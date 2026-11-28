package dev.d4r.minecraft.mixin;

import com.mojang.renderpearl.api.textures.GpuTextureView;
import com.mojang.renderpearl.backend.api.CommandEncoderBackend;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuSurface;
import dev.d4r.minecraft.FrameGen;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import java.nio.IntBuffer;
import org.lwjgl.vulkan.KHRSwapchain;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPresentInfoKHR;
import org.lwjgl.vulkan.VkQueue;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Presentation hooks (docs/presentation.md). Per Minecraft frame, with frame generation active:
 *   blitFromTexture: instead of the real frame, the first generated frame goes into the acquired swapchain image
 *                    (the generation work is recorded in front of it, in the same submission)
 *   present:         Minecraft presents that image; afterwards the remaining generated frames and finally the real frame
 *                    are presented from here, each through its own acquire/blit/present
 */
@Mixin(VulkanGpuSurface.class)
public abstract class VulkanGpuSurfaceMixin {
	@Inject(method = "blitFromTexture", at = @At("HEAD"), cancellable = true)
	private void d4r$blit(CommandEncoderBackend encoder, GpuTextureView view, CallbackInfo ci) {
		if (FrameGen.blit((VulkanGpuSurfaceAccessor) (Object) this, encoder, view)) {
			ci.cancel();
		}
	}

	/** The pacer thread presents the image instead (on its own clock). */
	@Inject(method = "present", at = @At("HEAD"), cancellable = true)
	private void d4r$present(CallbackInfo ci) {
		if (FrameGen.deferPresent((VulkanGpuSurfaceAccessor) (Object) this)) {
			ci.cancel();
		}
	}

	@Redirect(method = "present", at = @At(value = "INVOKE", target = "Lorg/lwjgl/vulkan/KHRSwapchain;vkQueuePresentKHR(Lorg/lwjgl/vulkan/VkQueue;Lorg/lwjgl/vulkan/VkPresentInfoKHR;)I"))
	private int d4r$present(VkQueue queue, VkPresentInfoKHR info) {
		FrameGen.QUEUE.lock();
		try {
			return KHRSwapchain.vkQueuePresentKHR(queue, info);
		} finally {
			FrameGen.QUEUE.unlock();
		}
	}

	@Redirect(method = "acquireNextTexture", at = @At(value = "INVOKE", target = "Lorg/lwjgl/vulkan/KHRSwapchain;vkAcquireNextImageKHR(Lorg/lwjgl/vulkan/VkDevice;JJJJLjava/nio/IntBuffer;)I"))
	private int d4r$acquire(VkDevice device, long swapchain, long timeout, long semaphore, long fence, IntBuffer index) {
		return FrameGen.acquireLocked(device, swapchain, timeout, semaphore, index);
	}

	/** The presenter thread holds images of its own: more than Minecraft's 3 keep Minecraft's frame-start acquire from waiting for them. */
	@Redirect(method = "configure", at = @At(value = "INVOKE", target = "Lorg/lwjgl/vulkan/VkSwapchainCreateInfoKHR;minImageCount(I)Lorg/lwjgl/vulkan/VkSwapchainCreateInfoKHR;"))
	private org.lwjgl.vulkan.VkSwapchainCreateInfoKHR d4r$imageCount(org.lwjgl.vulkan.VkSwapchainCreateInfoKHR info, int count) {
		if (!FrameGen.isWanted()) {
			return info.minImageCount(count);
		}
		int want = 10;
		try (org.lwjgl.system.MemoryStack stack = org.lwjgl.system.MemoryStack.stackPush()) {
			VulkanGpuSurfaceAccessor acc = (VulkanGpuSurfaceAccessor) (Object) this;
			org.lwjgl.vulkan.VkSurfaceCapabilitiesKHR caps = org.lwjgl.vulkan.VkSurfaceCapabilitiesKHR.calloc(stack);
			org.lwjgl.vulkan.KHRSurface.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(acc.d4r$device().vkDevice().getPhysicalDevice(), acc.d4r$surface(), caps);
			if (caps.maxImageCount() > 0) {
				want = Math.min(want, caps.maxImageCount());
			}
		}
		return info.minImageCount(Math.max(count, want));
	}

	/** Vsync off: MAILBOX instead of IMMEDIATE (no tearing, no frame-rate cap: the display takes the newest image at each refresh). */
	@Redirect(method = "configure", at = @At(value = "INVOKE", target = "Lorg/lwjgl/vulkan/VkSwapchainCreateInfoKHR;presentMode(I)Lorg/lwjgl/vulkan/VkSwapchainCreateInfoKHR;"))
	private org.lwjgl.vulkan.VkSwapchainCreateInfoKHR d4r$presentMode(org.lwjgl.vulkan.VkSwapchainCreateInfoKHR info, int mode) {
		if (mode == 0 && FrameGen.isWanted() && dev.d4r.minecraft.D4rConfig.get().mailbox()) {
			try (org.lwjgl.system.MemoryStack stack = org.lwjgl.system.MemoryStack.stackPush()) {
				VulkanGpuSurfaceAccessor acc = (VulkanGpuSurfaceAccessor) (Object) this;
				java.nio.IntBuffer n = stack.callocInt(1);
				org.lwjgl.vulkan.VkPhysicalDevice pd = acc.d4r$device().vkDevice().getPhysicalDevice();
				org.lwjgl.vulkan.KHRSurface.vkGetPhysicalDeviceSurfacePresentModesKHR(pd, acc.d4r$surface(), n, null);
				java.nio.IntBuffer modes = stack.callocInt(n.get(0));
				org.lwjgl.vulkan.KHRSurface.vkGetPhysicalDeviceSurfacePresentModesKHR(pd, acc.d4r$surface(), n, modes);
				for (int i = 0; i < modes.limit(); i++) {
					if (modes.get(i) == 1) {
						return info.presentMode(1);
					}
				}
			}
		}
		return info.presentMode(mode);
	}
}
