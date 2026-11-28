package dev.d4r.minecraft;

import com.mojang.renderpearl.api.commands.GpuQueryPool;
import com.mojang.renderpearl.backend.vulkan.VulkanCommandEncoder;
import com.mojang.renderpearl.backend.vulkan.VulkanDevice;
import java.util.OptionalLong;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;

/**
 * GPU timestamps through Minecraft's own encoder: frame start, world start, HUD-less point, before generation, after generation.
 * Results are read two frames later (a ring of four frames), so nothing waits for the GPU.
 */
public final class GpuTimer {
	private static final int STAMPS = 5, RING = 4;
	private static GpuQueryPool pool;
	private static double nsPerTick;
	private static long frame;
	private static boolean failed;
	private static volatile double worldMs = -1, d4rMs = -1, frameMs = -1, preMs = -1, guiMs = -1;

	private GpuTimer() {
	}

	public static double worldMs() {
		return worldMs;
	}

	public static double d4rMs() {
		return d4rMs;
	}

	public static double frameMs() {
		return frameMs;
	}

	private static boolean ready(VulkanDevice vk) {
		if (failed) {
			return false;
		}
		if (pool == null) {
			try (MemoryStack stack = MemoryStack.stackPush()) {
				VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(stack);
				VK12.vkGetPhysicalDeviceProperties(vk.vkDevice().getPhysicalDevice(), props);
				nsPerTick = props.limits().timestampPeriod();
				pool = vk.createTimestampQueryPool(STAMPS * RING);
			} catch (RuntimeException e) {
				failed = true;
				D4rClient.LOGGER.warn("[D4R] GPU timers unavailable: {}", e.toString());
				return false;
			}
		}
		return true;
	}

	/** Writes stamp {@code which} (0 frame start, 1 world start, 2 HUD-less point, 3 GUI done / before generation, 4 after generation) for the current frame. */
	public static void stamp(int which) {
		if (!D4rConfig.get().overlay()) {
			return;
		}
		VulkanDevice vk = VulkanAccess.device();
		if (vk == null || !ready(vk)) {
			return;
		}
		VulkanCommandEncoder enc = VulkanAccess.encoder(vk);
		if (which == 0) {
			frame++;
			collect();
		}
		enc.writeTimestamp(pool, (int) (frame % RING) * STAMPS + which);
	}

	private static void collect() {
		long old = frame - 2;
		if (old < 1) {
			return;
		}
		OptionalLong[] v = pool.getValues((int) (old % RING) * STAMPS, STAMPS);
		for (OptionalLong o : v) {
			if (o.isEmpty()) {
				return;
			}
		}
		double k = nsPerTick / 1e6;
		preMs = (v[1].getAsLong() - v[0].getAsLong()) * k;
		worldMs = (v[2].getAsLong() - v[1].getAsLong()) * k;
		guiMs = (v[3].getAsLong() - v[2].getAsLong()) * k;
		d4rMs = (v[4].getAsLong() - v[3].getAsLong()) * k;
		frameMs = (v[4].getAsLong() - v[0].getAsLong()) * k;
	}

	public static String line() {
		if (frameMs < 0) {
			return null;
		}
		return String.format("GPU %.1f ms: world %.1f  FG %.1f  other %.1f", frameMs, worldMs, Math.max(d4rMs, 0), Math.max(frameMs - worldMs - Math.max(d4rMs, 0), 0));
	}
}
