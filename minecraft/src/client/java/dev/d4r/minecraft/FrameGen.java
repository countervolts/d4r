package dev.d4r.minecraft;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import com.mojang.renderpearl.backend.api.CommandEncoderBackend;
import com.mojang.renderpearl.backend.vulkan.VulkanCommandEncoder;
import com.mojang.renderpearl.backend.vulkan.VulkanDevice;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTexture;
import dev.d4r.minecraft.mixin.GameRendererAccessor;
import dev.d4r.minecraft.mixin.VulkanDeviceAccessor;
import dev.d4r.minecraft.mixin.VulkanGpuSurfaceAccessor;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.locks.LockSupport;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import org.joml.Matrix4d;
import org.joml.Matrix4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.KHRSwapchain;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkFenceCreateInfo;
import org.lwjgl.vulkan.VkPresentInfoKHR;
import org.lwjgl.vulkan.VkSemaphoreCreateInfo;
import org.lwjgl.vulkan.KHRSynchronization2;
import org.lwjgl.vulkan.VkCommandBufferSubmitInfo;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSemaphoreSubmitInfo;
import org.lwjgl.vulkan.VkSemaphoreTypeCreateInfo;
import org.lwjgl.vulkan.VkSubmitInfo2;

/**
 * DLSS-G Lite multi frame generation on Minecraft's Vulkan device (docs/architecture.md, docs/presentation.md).
 *
 * <pre>
 * GameRenderer.render():
 *   world (full resolution, reversed-Z depth)
 *   hudlessPoint():  copy world colour, depth -> engine depth + camera motion vectors          [own command buffer]
 *   GUI / HUD
 * VulkanGpuSurface.blitFromTexture():  generate N-1 frames (engine script), first one goes to the swapchain image
 * VulkanGpuSurface.present():          Minecraft presents it; afterPresent() presents the others and the real frame
 * </pre>
 */
public final class FrameGen {
	private enum State { IDLE, ACTIVE, FAILED }

	private static State state = State.IDLE;
	private static String failure = "";
	private static long bridge;
	private static int width, height, factor, builtFactor;

	// camera of the frame being drawn and of the previous one (camera-relative rendering: reproject in double precision)
	private static boolean projectionCaptured, worldCaptured, havePrevious;
	private static final Matrix4f currentProjection = new Matrix4f(), currentView = new Matrix4f();
	private static final Matrix4f previousProjection = new Matrix4f(), previousView = new Matrix4f();
	private static double camX, camY, camZ, prevCamX, prevCamY, prevCamZ;
	private static Object previousLevel;
	private static final float[] frameMatrix = new float[16];

	private static long seq;              // monotonic frame counter for the semaphore values (never reset)
	private static long frameIndex;          // frames since the engine (re)started its history; 0 = first frame
	private static int lastVariant;
	private static long lastColorImage;
	private static long cpuStart;
	private static double cpuAccumMs;
	private static int cpuFrames;
	private static volatile double cpuRenderMs = -1;

	// presentation
	private static long lastPresentNs;
	private static double framePeriodMs = 16.0;
	private static long presentedFrames, lastRateNs = System.nanoTime();
	private static volatile double shownFps = -1, realFps = -1;
	private static long realCount;
	private static final PresentRing ring = new PresentRing();

	private FrameGen() {
	}

	/** Frame generation is switched on (the swapchain is created before the engine is). */
	public static boolean isWanted() {
		return D4rConfig.get().enabled() && state != State.FAILED;
	}

	public static boolean isActive() {
		return state == State.ACTIVE && bridge != 0;
	}

	public static String status() {
		return switch (state) {
			case ACTIVE -> "active " + width + "x" + height + " " + factor + "x";
			case FAILED -> "disabled: " + failure;
			default -> "idle";
		};
	}

	public static int factor() {
		return factor;
	}

	public static long renderedFrames() {
		return realCount;
	}

	public static double cpuMs() {
		return cpuRenderMs;
	}

	public static double shownFps() {
		return shownFps;
	}

	public static double realFps() {
		return realFps;
	}

	public static void retry() {
		if (state == State.FAILED) {
			state = State.IDLE;
		}
	}

	public static void renderStart() {
		cpuStart = System.nanoTime();
	}

	public static void renderEnd() {
		cpuAccumMs += (System.nanoTime() - cpuStart) / 1e6;
		if (++cpuFrames >= 30) {
			cpuRenderMs = cpuAccumMs / cpuFrames;
			cpuAccumMs = 0;
			cpuFrames = 0;
		}
		realCount++;
	}

	// ------------------------------------------------------------------------------------------ frame begin

	public static void beginWorld(GameRenderer renderer) {
		GpuTimer.stamp(1);
		projectionCaptured = false;
		worldCaptured = false;
		D4rConfig config = D4rConfig.get();
		if (!config.enabled() || state == State.FAILED) {
			if (bridge != 0 && !config.enabled()) {
				destroy("frame generation turned off");
			}
			return;
		}
		VulkanDevice vk = VulkanAccess.device();
		if (vk == null) {
			fail("Minecraft is not running on its Vulkan backend (graphics API: " + VulkanAccess.backendName() + ")");
			return;
		}
		RenderTarget main = ((GameRendererAccessor) renderer).d4r$mainRenderTarget();
		ensure(vk, config, main.width, main.height);
	}

	private static int pendingFactor;
	private static long pendingSince;

	private static boolean ensure(VulkanDevice vk, D4rConfig config, int w, int h) {
		if (bridge != 0 && w == width && h == height && config.factor() != builtFactor) {
			// debounce quick clicks on the factor button: rebuild only once the value has been stable for half a second
			long now = System.nanoTime();
			if (config.factor() != pendingFactor) {
				pendingFactor = config.factor();
				pendingSince = now;
			}
			if (now - pendingSince < 500_000_000L) {
				return true;
			}
		}
		if (bridge != 0 && w == width && h == height && config.factor() == builtFactor) {
			return true;
		}
		destroy("recreate for " + w + "x" + h + " " + config.factor() + "x");
		String error = NativeBridge.load();
		if (error != null) {
			return fail(error);
		}
		if (!((VulkanDeviceAccessor) vk).d4r$enabledFeatures().contains(D4rFeatureSet.FEATURE_SET)) {
			return fail("the nvxrt layer is not active on Minecraft's device (needs an RDNA3 GPU on Mesa RADV, the 'layer' folder next to libd4r_mc.so, and a launch with the mod's pre-launch hook)");
		}
		Path script = Path.of(config.scriptDir(), w + "x" + h, "mfc" + (config.factor() - 1));
		if (!Files.isRegularFile(script.resolve("script.txt"))) {
			return fail("no frame-generation script for " + w + "x" + h + " at " + config.factor() + "x in " + script + " (build it with minecraft/scripts/build-fg-scripts.sh " + w + " " + h + ")");
		}
		try {
			vk.graphicsQueue().waitIdle();
			VulkanAccess.Handles hd = VulkanAccess.Handles.of(vk);
			bridge = NativeBridge.create(hd.instance(), hd.physicalDevice(), hd.device(), hd.graphicsQueue(), hd.graphicsFamily(), hd.computeFamily(), script.toString(), config.kernelCache(), w, h);
		} catch (RuntimeException e) {
			return fail(e.getMessage());
		}
		asyncMode = false;   // the compute-queue path measured no faster and does not copy into the output ring (see STATUS.md)
		if (asyncMode) {
			setupAsync(vk);
		}
		seq = 0;
		width = w;
		height = h;
		factor = NativeBridge.factor(bridge);
		builtFactor = config.factor();
		frameIndex = 0;
		havePrevious = false;
		state = State.ACTIVE;
		try {
			var vm = Minecraft.getInstance().getWindow().getActiveVideoMode();
			D4rClient.LOGGER.info("[D4R] display mode Minecraft is using: {} Hz{}", vm == null ? "unknown (windowed/desktop mode)" : vm.getRefreshRate(), Minecraft.getInstance().options.enableVsync().get() ? ", vsync on" : ", vsync off");
		} catch (RuntimeException ignored) {
		}
		D4rClient.LOGGER.info("[D4R] frame generation initialised on Minecraft's Vulkan device: {}x{}, {}x ({} generated frames per real frame), script {}", w, h, factor, factor - 1, script);
		return true;
	}

	// ------------------------------------------------------------------------------------------ camera

	/** The level projection (bobbing included) is final here. Nothing is changed; the camera is remembered for the motion vectors. */
	public static Matrix4f captureProjection(GameRenderer renderer, Matrix4f projection) {
		if (!isActive()) {
			return projection;
		}
		CameraRenderState camera = ((GameRendererAccessor) renderer).d4r$gameRenderState().levelRenderState.cameraRenderState;
		currentProjection.set(projection);
		currentView.set(camera.viewRotationMatrix);
		camX = camera.pos.x;
		camY = camera.pos.y;
		camZ = camera.pos.z;
		projectionCaptured = true;
		return projection;
	}

	public static boolean wantsConsistentDepth() {
		return isActive();
	}

	// ------------------------------------------------------------------------------------------ world finished, GUI not yet drawn

	public static void hudlessPoint(GameRenderer renderer) {
		GpuTimer.stamp(2);
		if (!isActive() || !projectionCaptured) {
			havePrevious = false;
			frameIndex = 0;
			return;
		}
		try {
			Minecraft mc = Minecraft.getInstance();
			boolean reset = !havePrevious || mc.level != previousLevel;
			previousLevel = mc.level;
			double dx = camX - prevCamX, dy = camY - prevCamY, dz = camZ - prevCamZ;
			if (dx * dx + dy * dy + dz * dz > 24.0 * 24.0) {
				reset = true;   // teleport / respawn / dimension change
			}
			if (reset) {
				frameIndex = 0;
			}
			// previous clip = Pprev * Vprev * T(camCur - camPrev) * inverse(Pcur * Vcur)   (camera-relative world space)
			Matrix4d m = new Matrix4d();
			if (havePrevious && !reset) {
				Matrix4d curInverse = new Matrix4d(currentProjection).mul(new Matrix4d(currentView)).invert();
				new Matrix4d(previousProjection).mul(new Matrix4d(previousView)).translate(dx, dy, dz).mul(curInverse, m);
			} else {
				m.identity();
			}
			for (int c = 0; c < 4; c++) {
				for (int r = 0; r < 4; r++) {
					frameMatrix[c * 4 + r] = (float) m.get(c, r);
				}
			}
			RenderTarget main = ((GameRendererAccessor) renderer).d4r$mainRenderTarget();
			VulkanGpuTexture color = (VulkanGpuTexture) main.getColorTexture();
			VulkanGpuTexture depth = (VulkanGpuTexture) main.getDepthTexture();
			VulkanDevice vk = VulkanAccess.device();
			VulkanCommandEncoder encoder = VulkanAccess.encoder(vk);
			if (asyncMode && seq > 0) {
				encoder.waitSemaphore(timeline, 2 * seq, 65536L);   // compute of the previous frame has read the engine inputs
			}
			VkCommandBuffer cb = encoder.allocateAndBeginTransientCommandBuffer();
			NativeBridge.captureWorld(bridge, cb.address(), color.vkImage(), depth.vkImage(), frameMatrix, D4rConfig.get().motionSign());
			end(cb);
			encoder.execute(cb);
			previousProjection.set(currentProjection);
			previousView.set(currentView);
			prevCamX = camX;
			prevCamY = camY;
			prevCamZ = camZ;
			havePrevious = true;
			worldCaptured = true;
		} catch (RuntimeException e) {
			fail("recording failed: " + e);
		}
	}

	private static void end(VkCommandBuffer cb) {
		int r = VK12.vkEndCommandBuffer(cb);
		if (r != VK12.VK_SUCCESS) {
			throw new IllegalStateException("vkEndCommandBuffer failed: " + r);
		}
	}

	// ------------------------------------------------------------------------------------------ presentation

	/** Replaces Minecraft's blit of the real frame: generation runs in front of it and the first generated frame is blitted instead. */
	public static boolean blit(VulkanGpuSurfaceAccessor s, CommandEncoderBackend encoderBackend, GpuTextureView view) {
		if (!isActive() || !worldCaptured) {
			return false;
		}
		worldCaptured = false;
		long tb = System.nanoTime();
		try {
			VulkanCommandEncoder encoder = (VulkanCommandEncoder) encoderBackend;
			VulkanGpuTexture color = (VulkanGpuTexture) view.texture();
			lastColorImage = color.vkImage();
			lastVariant = frameIndex == 0 ? 2 : (int) (frameIndex & 1);
			GpuTimer.stamp(3);
			long f = seq;
			long fi = frameIndex;
			waitForPresenter();
			VkCommandBuffer cb = encoder.allocateAndBeginTransientCommandBuffer();
			int ringSlot = (int) (f % RING);
			NativeBridge.generateGfx(bridge, cb.address(), lastColorImage, ringSlot);
			long swapImage = s.d4r$images().getLong(s.d4r$currentImageIndex());
			if (asyncMode) {
				// Minecraft shows the real frame of the previous iteration here; the generated frames in between come from the presenter
				int showSlot = fi == 0 ? 0 : (int) ((fi + 1) & 1);
				NativeBridge.blitReal(bridge, cb.address(), showSlot, swapImage, s.d4r$width(), s.d4r$height());
			} else {
				// everything on the graphics queue: generated frame 1 now, the rest and the real frame from the presenter
				NativeBridge.generateCompute(bridge, cb.address(), lastVariant, ringSlot, frameMatrix);
				NativeBridge.blitGenerated(bridge, cb.address(), ringSlot, 0, swapImage, s.d4r$width(), s.d4r$height());
			}
			end(cb);
			encoder.waitSemaphore(s.d4r$acquireSemaphores()[s.d4r$currentAcquireSemaphore()], 0L, 65536L);
			encoder.execute(cb);
			encoder.signalSemaphore(s.d4r$presentSemaphores()[s.d4r$currentImageIndex()], 0L, 4096L);
			if (asyncMode) {
				encoder.signalSemaphore(timeline, 2 * f + 1, 65536L);   // inputs of frame f are in place
				submitCompute(f, lastVariant);
			}
			seq++;
			GpuTimer.stamp(4);
			pendingExtra = true;
			blitNs += System.nanoTime() - tb;
			return true;
		} catch (RuntimeException e) {
			fail("generation failed: " + e);
			return false;
		}
	}

	/** Slots of generated/real images the presenter can still be reading while the engine works on newer frames (native kRing). */
	private static final int RING = 4;
	private static boolean pendingExtra, asyncMode;
	private static long timeline, blitsDone, computePool;
	private static VkQueue computeQueue;
	private static final int COMPUTE_RING = 6;
	private static final VkCommandBuffer[] computeCbs = new VkCommandBuffer[COMPUTE_RING];

	private static void setupAsync(VulkanDevice vk) {
		seq = 0;
		VkDevice device = vk.vkDevice();
		computeQueue = vk.computeQueue().vkQueue();
		try (MemoryStack stack = MemoryStack.stackPush()) {
			LongBuffer l = stack.longs(0);
			VkSemaphoreTypeCreateInfo ti = VkSemaphoreTypeCreateInfo.calloc(stack).sType$Default().semaphoreType(VK12.VK_SEMAPHORE_TYPE_TIMELINE).initialValue(0);
			VkSemaphoreCreateInfo si = VkSemaphoreCreateInfo.calloc(stack).sType$Default().pNext(ti.address());
			VK12.vkCreateSemaphore(device, si, null, l);
			timeline = l.get(0);
			VK12.vkCreateSemaphore(device, si, null, l);
			blitsDone = l.get(0);
			VkCommandPoolCreateInfo pci = VkCommandPoolCreateInfo.calloc(stack).sType$Default().flags(VK12.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
				.queueFamilyIndex(vk.computeQueue().queueFamilyIndex());
			VK12.vkCreateCommandPool(device, pci, null, l);
			computePool = l.get(0);
			PointerBuffer pb = stack.mallocPointer(COMPUTE_RING);
			VkCommandBufferAllocateInfo ai = VkCommandBufferAllocateInfo.calloc(stack).sType$Default().commandPool(computePool).level(VK12.VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(COMPUTE_RING);
			VK12.vkAllocateCommandBuffers(device, ai, pb);
			for (int i = 0; i < COMPUTE_RING; i++) {
				computeCbs[i] = new VkCommandBuffer(pb.get(i), device);
			}
		}
	}

	/** Records the generation script for frame f on the compute queue; it waits for the graphics-side inputs and for the previous users of the generated images. */
	private static void submitCompute(long f, int variant) {
		VkCommandBuffer cb = computeCbs[(int) (f % COMPUTE_RING)];
		try (MemoryStack stack = MemoryStack.stackPush()) {
			VK12.vkResetCommandBuffer(cb, 0);
			VK12.vkBeginCommandBuffer(cb, VkCommandBufferBeginInfo.calloc(stack).sType$Default().flags(VK12.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));
			NativeBridge.generateCompute(bridge, cb.address(), variant, 0, frameMatrix);
			end(cb);
			VkSubmitInfo2.Buffer submit = VkSubmitInfo2.calloc(1, stack).sType$Default();
			int nw = f >= 2 ? 2 : 1;
			VkSemaphoreSubmitInfo.Buffer waits = VkSemaphoreSubmitInfo.calloc(nw, stack);
			waits.get(0).sType$Default().semaphore(timeline).value(2 * f + 1).stageMask(65536L);
			if (f >= 2) {
				waits.get(1).sType$Default().semaphore(blitsDone).value(f - 1).stageMask(65536L);   // blits of frame f-2 (value f-2+1) are done
			}
			VkSemaphoreSubmitInfo.Buffer signals = VkSemaphoreSubmitInfo.calloc(1, stack);
			signals.get(0).sType$Default().semaphore(timeline).value(2 * f + 2).stageMask(65536L);
			VkCommandBufferSubmitInfo.Buffer cmd = VkCommandBufferSubmitInfo.calloc(1, stack);
			cmd.get(0).sType$Default().commandBuffer(cb);
			submit.pWaitSemaphoreInfos(waits).pCommandBufferInfos(cmd).pSignalSemaphoreInfos(signals);
			int r = KHRSynchronization2.vkQueueSubmit2KHR(computeQueue, submit, 0L);
			if (r != VK12.VK_SUCCESS) {
				throw new IllegalStateException("compute vkQueueSubmit2 failed: " + r);
			}
		}
	}

	/** Signal-only submission on the graphics queue (a dropped presenter job still has to release its images). */
	private static void signalBlitsDone(long value) {
		try (MemoryStack stack = MemoryStack.stackPush()) {
			VkSubmitInfo2.Buffer submit = VkSubmitInfo2.calloc(1, stack).sType$Default();
			VkSemaphoreSubmitInfo.Buffer signals = VkSemaphoreSubmitInfo.calloc(1, stack);
			signals.get(0).sType$Default().semaphore(blitsDone).value(value).stageMask(4096L);
			submit.pSignalSemaphoreInfos(signals);
			QUEUE.lock();
			try {
				KHRSynchronization2.vkQueueSubmit2KHR(surfaceRef.d4r$presentQueue(), submit, 0L);
			} finally {
				QUEUE.unlock();
			}
		}
	}

	// ---- presenter thread: shows the remaining generated frames and the real frame at even spacing while the render thread works on the next frame
	private record Job(long frame, int variant, int slot, int mcImage, long mcSemaphore, long arrivalNs) {
	}

	private static final java.util.concurrent.LinkedBlockingQueue<Job> jobs = new java.util.concurrent.LinkedBlockingQueue<>();
	private static final java.util.concurrent.atomic.AtomicLong completed = new java.util.concurrent.atomic.AtomicLong(-1);
	private static final java.util.concurrent.atomic.AtomicLong shownCount = new java.util.concurrent.atomic.AtomicLong();
	private static volatile VulkanGpuSurfaceAccessor surfaceRef;
	private static Thread presenter;
	private static volatile boolean presenterStop;
	/** Guards every use of Minecraft's graphics queue and swapchain that can come from two threads (see VulkanQueueMixin). */
	public static final java.util.concurrent.locks.ReentrantLock QUEUE = new java.util.concurrent.locks.ReentrantLock();

	/** Frame k reuses the generated/real images of frame k-2: its presenter job must be finished before frame k's generation is recorded. */
	public static long waitNs, blitNs, dropped, acquireNs, jobLagNs, jobLenNs, jobCount, presentNs, pAcq, pRec, pReset, pNative, pFence, pSub, pPres, pCount;

	private static void waitForPresenter() {
		long t0 = System.nanoTime();
		try {
			waitForPresenter0();
		} finally {
			waitNs += System.nanoTime() - t0;
		}
	}

	private static void waitForPresenter0() {
		long deadline = System.nanoTime() + 2_000_000_000L;
		while (completed.get() < seq - RING && presenter != null && System.nanoTime() < deadline) {
			LockSupport.parkNanos(200_000);
		}
	}

	/**
	 * Minecraft's own present is taken over: the image it blitted (generated frame 1, or the previous real frame in async mode) is presented by the
	 * pacer thread together with the other frames of the cycle, on one steady clock.
	 */
	public static boolean deferPresent(VulkanGpuSurfaceAccessor s) {
		if (!pendingExtra) {
			return false;
		}
		pendingExtra = false;
		capRate();
		int image = s.d4r$currentImageIndex();
		long semaphore = s.d4r$presentSemaphores()[image];
		s.d4r$setCurrentImageIndex(-1);
		surfaceRef = s;
		if (presenter == null) {
			presenterStop = false;
			presenter = new Thread(FrameGen::presenterLoop, "d4r-presenter");
			presenter.setDaemon(true);
			presenter.start();
		}
		jobs.add(new Job(seq - 1, lastVariant, (int) ((seq - 1) % RING), image, semaphore, System.nanoTime()));
		frameIndex++;
		return true;
	}

	private static long lastCapNs;

	/** Limits the real frame rate so that factor * real <= the display's refresh rate (frames above it only tear and make the pacing worse). */
	private static void capRate() {
		D4rConfig c = D4rConfig.get();
		if (!c.cap()) {
			return;
		}
		double hz = c.maxShownFps();
		if (hz <= 0) {
			try {
				var mode = Minecraft.getInstance().getWindow().getActiveVideoMode();
				hz = mode == null ? 0 : mode.getRefreshRate();
			} catch (RuntimeException e) {
				hz = 0;
			}
		}
		if (hz < 20) {
			return;
		}
		long minNs = (long) (1e9 * factor / hz);
		long due = lastCapNs + minNs;
		long now = System.nanoTime();
		if (now < due) {
			sleepUntil(due);
			now = due;
		}
		lastCapNs = now;
	}

	// pacer statistics (presenter thread writes, overlay reads)
	public static volatile double paceAvgMs = -1, paceSigmaMs, paceMaxMs, paceScale = 1, paceLeadItems;
	public static volatile long paceLate;

	private static void sleepUntil(long due) {
		long wait = due - System.nanoTime();
		if (wait > 300_000) {
			LockSupport.parkNanos(wait - 150_000);
		}
		while (System.nanoTime() < due) {
			Thread.onSpinWait();
		}
	}

	/**
	 * The pacer. Every Minecraft frame brings one cycle of N images (generated frames and the real frame). They are shown on a steady clock:
	 * interval d = smoothed frame period / N * scale. Scale is nudged so that the schedule stays about half a cycle ahead of the clock,
	 * which absorbs a late frame without a visible gap; a long idle (menu, load) restarts the clock with a one-interval lead.
	 */
	private static void presenterLoop() {
		double periodMs = 16.0;
		long lastArrival = 0, clock = 0, lastShown = 0;
		double scale = 1.0;
		double sum = 0, sumSq = 0, max = 0;
		int count = 0;
		long late = 0, statStart = System.nanoTime();
		try {
			while (!presenterStop) {
				Job job = jobs.poll(100, java.util.concurrent.TimeUnit.MILLISECONDS);
				if (job == null) {
					continue;
				}
				if (lastArrival != 0) {
					double p = (job.arrivalNs() - lastArrival) / 1e6;
					periodMs = periodMs * 0.95 + Math.max(1.0, Math.min(p, 100.0)) * 0.05;
				}
				lastArrival = job.arrivalNs();
				int n = factor;
				long dNs = (long) (periodMs / n * scale * 1e6);
				long now = System.nanoTime();
				if (clock < now - 3 * dNs) {
					clock = now + (long) (0.6 * n * dNs);   // restart with a buffer of 0.6 cycle
				}
				double lead = Math.max(0, (clock - now) / (double) dNs);
				// No feedback onto the frame rate: the interval follows the measured arrival period; only a backlog is drained a little faster
				// target: the schedule is 0.6 cycle ahead when a frame arrives; starved -> slow down a little, ahead -> speed up (max 4%)
				scale = jobs.size() >= 1 ? 0.9 : Math.max(0.96, Math.min(1.04, 1.0 + 0.04 * (0.6 * n - lead) / (0.6 * n)));
				paceScale = scale;
				paceLeadItems = lead;
				dNs = (long) (periodMs / n * scale * 1e6);
				VulkanGpuSurfaceAccessor s = surfaceRef;
				boolean skipMiddle = jobs.size() >= 1;   // another frame is already waiting: drop the in-between images of this one
				boolean lastGenDone = false;
				for (int i = 0; i < n && !presenterStop; i++) {
					boolean mc = i == 0;
					boolean realItem = !asyncMode && i == n - 1;
					if (!mc && !realItem && skipMiddle) {
						dropped++;
						continue;
					}
					long due = clock;
					clock += dNs;
					sleepUntil(due);
					long t = System.nanoTime();
					if (t - due > dNs / 2) {
						late++;
					}
					if (t - due > dNs) {
						clock = t + dNs;   // far behind: restart the schedule instead of presenting a burst
					}
					boolean ok;
					if (mc) {
						ok = ring.presentMc(s, job.mcImage(), job.mcSemaphore());
					} else if (asyncMode) {
						boolean last = i == n - 1;
						ok = ring.present(s, bridge, i - 1, job.variant(), 0, 2 * job.frame() + 2, last ? job.frame() + 1 : 0);
						lastGenDone |= last;
					} else {
						ok = ring.present(s, bridge, realItem ? -1 : i, job.slot(), job.slot(), 0, 0);
					}
					if (ok) {
						shownCount.incrementAndGet();
						long shown = System.nanoTime();
						if (lastShown != 0) {
							double iv = (shown - lastShown) / 1e6;
							sum += iv;
							sumSq += iv * iv;
							max = Math.max(max, iv);
							count++;
						}
						lastShown = shown;
					}
				}
				if (asyncMode && !lastGenDone) {
					signalBlitsDone(job.frame() + 1);   // the compute side waits for this before it reuses the images
				}
				completed.set(job.frame());
				long nowRate = System.nanoTime();
				if (nowRate - lastRateNs >= 500_000_000L) {
					shownFps = shownCount.getAndSet(0) * 1e9 / (nowRate - lastRateNs);
					realFps = shownFps / Math.max(1, factor);
					lastRateNs = nowRate;
				}
				if (System.nanoTime() - statStart >= 2_000_000_000L && count > 0) {
					double avg = sum / count;
					paceAvgMs = avg;
					paceSigmaMs = Math.sqrt(Math.max(0, sumSq / count - avg * avg));
					paceMaxMs = max;
					paceLate = late;
					if (D4rConfig.get().log()) {
						D4rClient.LOGGER.info("[D4R] pacing: present interval avg {} ms sigma {} max {}; {} late of {}; lead {} items, scale {}", String.format("%.2f", avg),
							String.format("%.2f", paceSigmaMs), String.format("%.1f", max), late, count, String.format("%.1f", lead), String.format("%.3f", scale));
					}
					sum = sumSq = max = 0;
					count = 0;
					late = 0;
					statStart = System.nanoTime();
				}
			}
		} catch (InterruptedException | RuntimeException e) {
			if (!presenterStop) {
				D4rClient.LOGGER.error("[D4R] presenter thread stopped: {}", e.toString());
				state = State.FAILED;
				failure = "presenter: " + e;
			}
		}
	}

	/** Own acquire / blit / present of images that Minecraft does not know about, with a small ring of command buffers. */
	private static final class PresentRing {
		private static final int SLOTS = 32;
		private long pool, swapchain;
		private final VkCommandBuffer[] cbs = new VkCommandBuffer[SLOTS];
		private final long[] fences = new long[SLOTS];
		private final boolean[] used = new boolean[SLOTS];
		private final long[] acquire = new long[2];
		private long[] present = new long[0];
		private int ringIndex, acquireIndex;
		private VkDevice device;

		boolean presentMc(VulkanGpuSurfaceAccessor s, int image, long semaphore) {
			try (MemoryStack stack = MemoryStack.stackPush()) {
				VkPresentInfoKHR pi = VkPresentInfoKHR.calloc(stack).sType$Default();
				pi.pWaitSemaphores(stack.longs(semaphore));
				pi.swapchainCount(1);
				pi.pSwapchains(stack.longs(s.d4r$swapchain()));
				pi.pImageIndices(stack.ints(image));
				QUEUE.lock();
				try {
					return KHRSwapchain.vkQueuePresentKHR(s.d4r$presentQueue(), pi) >= 0;
				} finally {
					QUEUE.unlock();
				}
			}
		}

		boolean present(VulkanGpuSurfaceAccessor s, long handle, int generated, int variant, int realSlot, long waitValue, long signalValue) {
			VulkanDevice vk = s.d4r$device();
			setup(vk, s);
			try (MemoryStack stack = MemoryStack.stackPush()) {
				long q0 = System.nanoTime();
				int a = acquireIndex++ & 1;
				IntBuffer idx = stack.ints(-1);
				int r = acquireLocked(device, swapchain, 5_000_000_000L, acquire[a], idx);
				if (r < 0 || r == VK12.VK_TIMEOUT || r == VK12.VK_NOT_READY) {
					return false;
				}
				int image = idx.get(0);
				long q1 = System.nanoTime();
				pAcq += q1 - q0;
				int i = ringIndex++ % SLOTS;
				long w0 = System.nanoTime();
				if (used[i]) {
					VK12.vkWaitForFences(device, fences[i], true, 5_000_000_000L);
					VK12.vkResetFences(device, fences[i]);
				}
				pFence += System.nanoTime() - w0;
				used[i] = true;
				long r0 = System.nanoTime();
				VK12.vkResetCommandBuffer(cbs[i], 0);
				VK12.vkBeginCommandBuffer(cbs[i], VkCommandBufferBeginInfo.calloc(stack).sType$Default().flags(VK12.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));
				long r1 = System.nanoTime();
				pReset += r1 - r0;
				long swapImage = s.d4r$images().getLong(image);
				if (generated >= 0) {
					NativeBridge.blitGenerated(handle, cbs[i].address(), variant, generated, swapImage, s.d4r$width(), s.d4r$height());
				} else {
					NativeBridge.blitReal(handle, cbs[i].address(), realSlot, swapImage, s.d4r$width(), s.d4r$height());
				}
				long r2 = System.nanoTime();
				pNative += r2 - r1;
				end(cbs[i]);
				long q2 = System.nanoTime();
				pRec += q2 - q1;
				VkSubmitInfo2.Buffer submit = VkSubmitInfo2.calloc(1, stack).sType$Default();
				VkSemaphoreSubmitInfo.Buffer waits = VkSemaphoreSubmitInfo.calloc(asyncMode ? 2 : 1, stack);
				waits.get(0).sType$Default().semaphore(acquire[a]).value(0).stageMask(4096L);
				if (asyncMode) {
					waits.get(1).sType$Default().semaphore(timeline).value(waitValue).stageMask(4096L);
				}
				VkSemaphoreSubmitInfo.Buffer signals = VkSemaphoreSubmitInfo.calloc(signalValue > 0 ? 2 : 1, stack);
				signals.get(0).sType$Default().semaphore(present[image]).value(0).stageMask(4096L);
				if (signalValue > 0) {
					signals.get(1).sType$Default().semaphore(blitsDone).value(signalValue).stageMask(4096L);
				}
				VkCommandBufferSubmitInfo.Buffer cmdInfo = VkCommandBufferSubmitInfo.calloc(1, stack);
				cmdInfo.get(0).sType$Default().commandBuffer(cbs[i]);
				submit.pWaitSemaphoreInfos(waits).pCommandBufferInfos(cmdInfo).pSignalSemaphoreInfos(signals);
				QUEUE.lock();
				int sr;
				try {
					sr = KHRSynchronization2.vkQueueSubmit2KHR(s.d4r$presentQueue(), submit, fences[i]);
				} finally {
					QUEUE.unlock();
				}
				long q3 = System.nanoTime();
				pSub += q3 - q2;
				if (sr != VK12.VK_SUCCESS) {
					throw new IllegalStateException("vkQueueSubmit failed: " + sr);
				}
				VkPresentInfoKHR pi = VkPresentInfoKHR.calloc(stack).sType$Default();
				pi.pWaitSemaphores(stack.longs(present[image]));
				pi.swapchainCount(1);
				pi.pSwapchains(stack.longs(swapchain));
				pi.pImageIndices(stack.ints(image));
				QUEUE.lock();
				try {
					return KHRSwapchain.vkQueuePresentKHR(s.d4r$presentQueue(), pi) >= 0;
				} finally {
					QUEUE.unlock();
					pPres += System.nanoTime() - q3;
					pCount++;
				}
			}
		}

		private void setup(VulkanDevice vk, VulkanGpuSurfaceAccessor s) {
			device = vk.vkDevice();
			long chain = s.d4r$swapchain();
			if (pool != 0 && chain == swapchain && present.length == s.d4r$images().size()) {
				return;
			}
			if (pool != 0) {
				vk.graphicsQueue().waitIdle();
				destroy();
			}
			swapchain = chain;
			try (MemoryStack stack = MemoryStack.stackPush()) {
				LongBuffer l = stack.longs(0);
				VkCommandPoolCreateInfo pci = VkCommandPoolCreateInfo.calloc(stack).sType$Default().flags(VK12.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
					.queueFamilyIndex(vk.graphicsQueue().queueFamilyIndex());
				VK12.vkCreateCommandPool(device, pci, null, l);
				pool = l.get(0);
				PointerBuffer pb = stack.mallocPointer(SLOTS);
				VkCommandBufferAllocateInfo ai = VkCommandBufferAllocateInfo.calloc(stack).sType$Default().commandPool(pool).level(VK12.VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(SLOTS);
				VK12.vkAllocateCommandBuffers(device, ai, pb);
				VkFenceCreateInfo fi = VkFenceCreateInfo.calloc(stack).sType$Default();
				VkSemaphoreCreateInfo si = VkSemaphoreCreateInfo.calloc(stack).sType$Default();
				for (int i = 0; i < SLOTS; i++) {
					cbs[i] = new VkCommandBuffer(pb.get(i), device);
					VK12.vkCreateFence(device, fi, null, l);
					fences[i] = l.get(0);
					used[i] = false;
				}
				for (int i = 0; i < 2; i++) {
					VK12.vkCreateSemaphore(device, si, null, l);
					acquire[i] = l.get(0);
				}
				present = new long[s.d4r$images().size()];
				for (int i = 0; i < present.length; i++) {
					VK12.vkCreateSemaphore(device, si, null, l);
					present[i] = l.get(0);
				}
			}
		}

		void destroy() {
			if (pool == 0) {
				return;
			}
			for (int i = 0; i < SLOTS; i++) {
				VK12.vkDestroyFence(device, fences[i], null);
			}
			for (long a : acquire) {
				VK12.vkDestroySemaphore(device, a, null);
			}
			for (long p : present) {
				VK12.vkDestroySemaphore(device, p, null);
			}
			VK12.vkDestroyCommandPool(device, pool, null);
			pool = 0;
		}
	}

	/** vkAcquireNextImageKHR that never blocks while holding the queue lock (the other thread may need it to free an image). */
	public static int acquireLocked(VkDevice device, long swapchain, long timeoutNs, long semaphore, IntBuffer index) {
		long t0 = System.nanoTime();
		try {
			return acquireLocked0(device, swapchain, timeoutNs, semaphore, index);
		} finally {
			if (Thread.currentThread().getName().equals("Render thread")) {
				acquireNs += System.nanoTime() - t0;
			}
		}
	}

	private static int acquireLocked0(VkDevice device, long swapchain, long timeoutNs, long semaphore, IntBuffer index) {
		long deadline = System.nanoTime() + timeoutNs;
		while (true) {
			int r;
			QUEUE.lock();
			try {
				r = KHRSwapchain.vkAcquireNextImageKHR(device, swapchain, 1_000_000L, semaphore, 0L, index);
			} finally {
				QUEUE.unlock();
			}
			if (r != VK12.VK_TIMEOUT && r != VK12.VK_NOT_READY) {
				return r;
			}
			if (System.nanoTime() >= deadline) {
				return VK12.VK_TIMEOUT;
			}
			Thread.onSpinWait();
		}
	}

	// ------------------------------------------------------------------------------------------ teardown

	private static boolean fail(String reason) {
		failure = reason;
		state = State.FAILED;
		D4rClient.LOGGER.error("[D4R] frame generation disabled: {}", reason);
		destroy("failure");
		return false;
	}

	private static void destroy(String reason) {
		pendingExtra = false;
		if (bridge != 0) {
			D4rClient.LOGGER.info("[D4R] releasing frame generation ({})", reason);
			VulkanDevice vk = VulkanAccess.device();
			if (vk != null) {
				vk.graphicsQueue().waitIdle();
			}
			presenterStop = true;
			if (presenter != null) {
				presenter.interrupt();
				try {
					presenter.join(2000);
				} catch (InterruptedException ignored) {
					Thread.currentThread().interrupt();
				}
				presenter = null;
			}
			for (Job j : jobs) {   // images Minecraft handed over must go back to the swapchain
				try {
					ring.presentMc(surfaceRef, j.mcImage(), j.mcSemaphore());
				} catch (RuntimeException ignored) {
				}
			}
			jobs.clear();
			completed.set(-1);
			if (vk != null) {
				vk.computeQueue().waitIdle();
			}
			ring.destroy();
			if (timeline != 0 && vk != null) {
				VK12.vkDestroySemaphore(vk.vkDevice(), timeline, null);
				VK12.vkDestroySemaphore(vk.vkDevice(), blitsDone, null);
				VK12.vkDestroyCommandPool(vk.vkDevice(), computePool, null);
				timeline = blitsDone = computePool = 0;
			}
			NativeBridge.destroy(bridge);
			bridge = 0;
		}
		havePrevious = false;
		frameIndex = 0;
		if (state == State.ACTIVE) {
			state = State.IDLE;
		}
	}
}
