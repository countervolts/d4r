package dev.d4r.minecraft;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.device.GpuDevice;
import com.mojang.renderpearl.backend.vulkan.VulkanCommandEncoder;
import com.mojang.renderpearl.backend.vulkan.VulkanDevice;
import com.mojang.renderpearl.frontend.FrontendGpuDevice;
import dev.d4r.minecraft.mixin.FrontendGpuDeviceAccessor;

/** Reaches Minecraft's own Vulkan objects (device, instance, queues, command encoder) from the render thread. */
public final class VulkanAccess {
	private VulkanAccess() {
	}

	/** @return the active Vulkan device, or null when Minecraft runs on another backend. */
	public static VulkanDevice device() {
		GpuDevice device = RenderSystem.tryGetDevice();
		if (device instanceof FrontendGpuDevice frontend && ((FrontendGpuDeviceAccessor) frontend).d4r$backend() instanceof VulkanDevice vulkan) {
			return vulkan;
		}
		return null;
	}

	public static String backendName() {
		GpuDevice device = RenderSystem.tryGetDevice();
		return device == null ? "none" : device.getDeviceInfo().backendName();
	}

	public static VulkanCommandEncoder encoder(VulkanDevice device) {
		return device.createCommandEncoder();
	}

	/** Handles that identify the engine's Vulkan context. */
	public record Handles(long instance, long physicalDevice, long device, long graphicsQueue, int graphicsFamily, int computeFamily) {
		public static Handles of(VulkanDevice d) {
			return new Handles(d.instance().vkInstance().address(), d.vkDevice().getPhysicalDevice().address(), d.vkDevice().address(),
				d.graphicsQueue().vkQueue().address(), d.graphicsQueue().queueFamilyIndex(), d.computeQueue().queueFamilyIndex());
		}
	}
}
