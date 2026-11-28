package dev.d4r.minecraft.mixin;

import dev.d4r.minecraft.FrameGen;
import org.lwjgl.vulkan.KHRSynchronization2;
import org.lwjgl.vulkan.VK12;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSubmitInfo2;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

/** The presenter thread shares Minecraft's graphics queue: every queue call of Minecraft takes FrameGen.QUEUE too. */
@Mixin(targets = "com.mojang.renderpearl.backend.vulkan.VulkanQueue$Submission")
public abstract class VulkanQueueMixin {
	@Redirect(method = "close", at = @At(value = "INVOKE", target = "Lorg/lwjgl/vulkan/KHRSynchronization2;vkQueueSubmit2KHR(Lorg/lwjgl/vulkan/VkQueue;Lorg/lwjgl/vulkan/VkSubmitInfo2$Buffer;J)I"))
	private int d4r$submit(VkQueue queue, VkSubmitInfo2.Buffer submits, long fence) {
		FrameGen.QUEUE.lock();
		try {
			return KHRSynchronization2.vkQueueSubmit2KHR(queue, submits, fence);
		} finally {
			FrameGen.QUEUE.unlock();
		}
	}
}
