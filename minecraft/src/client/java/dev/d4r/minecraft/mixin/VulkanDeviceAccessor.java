package dev.d4r.minecraft.mixin;

import com.mojang.renderpearl.backend.vulkan.VulkanDevice;
import com.mojang.renderpearl.backend.vulkan.init.FeatureSet;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(VulkanDevice.class)
public interface VulkanDeviceAccessor {
	/** The FeatureSet the device was actually created with. */
	@Accessor("enabledFeatures")
	FeatureSet d4r$enabledFeatures();
}
