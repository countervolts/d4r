package dev.d4r.minecraft;

import java.nio.file.Files;
import java.nio.file.Path;

/** JNI surface of libd4r_mc.so (minecraft/native/fg/fg_mc.cpp). Handles are raw Vulkan handles from LWJGL. */
public final class NativeBridge {
	private static boolean loaded;
	private static String loadError;

	private NativeBridge() {
	}

	/** @return null when the library is loaded, otherwise the reason it is not. */
	public static synchronized String load() {
		if (loaded) {
			return null;
		}
		if (loadError != null) {
			return loadError;
		}
		StringBuilder tried = new StringBuilder();
		for (Path dir : D4rConfig.searchDirs()) {
			Path p = dir.resolve("libd4r_mc.so").toAbsolutePath().normalize();
			tried.append(' ').append(p);
			if (Files.isRegularFile(p)) {
				try {
					System.load(p.toString());
					loaded = true;
					D4rClient.LOGGER.info("[D4R] native bridge: {}", p);
					return null;
				} catch (UnsatisfiedLinkError e) {
					loadError = "cannot load " + p + ": " + e.getMessage();
					return loadError;
				}
			}
		}
		loadError = "libd4r_mc.so not found (set d4rDir / D4R_MC_DIR); tried:" + tried;
		return loadError;
	}

	/** Sets VK_ADD_LAYER_PATH, VK_INSTANCE_LAYERS and the NVXRT_* options; only effective before Minecraft creates its VkInstance. */
	public static native String setupEnvironment(String layerDir, String kernelCache, boolean exact, String logFile);

	public static native long create(long vkInstance, long vkPhysicalDevice, long vkDevice, long vkQueue, int queueFamily, int computeQueueFamily, String scriptDir,
		String kernelCache, int width, int height);

	/** Generated frames per real frame + 1 (the presented rate multiple). */
	public static native int factor(long handle);

	/** HUD-less point: copies the world colour and derives depth + motion vectors from the world depth. */
	public static native void captureWorld(long handle, long commandBuffer, long colorImage, long depthImage, float[] prevClipFromCurClip, float motionSign);

	/** End of frame, graphics queue: copies the final colour into the engine's input and into real-frame snapshot slot {@code slot}. */
	public static native void generateGfx(long handle, long commandBuffer, long colorImage, int slot);

	/** Compute queue: the generation script (variant 2 = first frame, then 0/1 alternating). */
	public static native void generateCompute(long handle, long commandBuffer, int variant, int ringSlot, float[] prevClipFromCurClip);

	public static native void blitGenerated(long handle, long commandBuffer, int ringSlot, int k, long swapchainImage, int width, int height);

	/** The snapshot of the real frame that generate() stored in {@code slot} (frame index parity). */
	public static native void blitReal(long handle, long commandBuffer, int slot, long swapchainImage, int width, int height);

	/** Only after every frame recorded with this handle has completed on the GPU. */
	public static native void destroy(long handle);
}
