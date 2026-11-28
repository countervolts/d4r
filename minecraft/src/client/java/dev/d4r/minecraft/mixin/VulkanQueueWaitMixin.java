package dev.d4r.minecraft.mixin;

import dev.d4r.minecraft.FrameGen;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkQueue;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

@Mixin(targets = "com.mojang.renderpearl.backend.vulkan.VulkanQueue")
public abstract class VulkanQueueWaitMixin {
	@Redirect(method = "waitIdle", at = @At(value = "INVOKE", target = "Lorg/lwjgl/vulkan/VK12;vkQueueWaitIdle(Lorg/lwjgl/vulkan/VkQueue;)I"))
	private int d4r$waitIdle(VkQueue queue) {
		FrameGen.QUEUE.lock();
		try {
			return VK12.vkQueueWaitIdle(queue);
		} finally {
			FrameGen.QUEUE.unlock();
		}
	}
}
