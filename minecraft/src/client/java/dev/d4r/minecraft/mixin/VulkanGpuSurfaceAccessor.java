package dev.d4r.minecraft.mixin;

import com.mojang.renderpearl.backend.vulkan.VulkanDevice;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuSurface;
import it.unimi.dsi.fastutil.longs.LongList;
import org.lwjgl.vulkan.VkQueue;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Read access to the swapchain state of Minecraft's Vulkan surface (acquire/blit/present order is documented in docs/presentation.md). */
@Mixin(VulkanGpuSurface.class)
public interface VulkanGpuSurfaceAccessor {
	@Accessor("device")
	VulkanDevice d4r$device();

	@Accessor("presentQueue")
	VkQueue d4r$presentQueue();

	@Accessor("swapchain")
	long d4r$swapchain();

	@Accessor("surface")
	long d4r$surface();

	@Accessor("swapchainWidth")
	int d4r$width();

	@Accessor("swapchainHeight")
	int d4r$height();

	@Accessor("swapchainImages")
	LongList d4r$images();

	@Accessor("acquireSemaphores")
	long[] d4r$acquireSemaphores();

	@Accessor("currentAcquireSemaphore")
	int d4r$currentAcquireSemaphore();

	@Accessor("presentSemaphores")
	long[] d4r$presentSemaphores();

	@Accessor("currentImageIndex")
	int d4r$currentImageIndex();

	@Accessor("currentImageIndex")
	void d4r$setCurrentImageIndex(int index);
}
