package dev.d4r.minecraft;

import com.mojang.renderpearl.backend.vulkan.init.FeatureSet;
import java.util.Set;

/**
 * The device extensions the frame-generation kernels launch through. The nvxrt layer advertises them (RDNA3 RADV only)
 * and adds the device features its SPIR-V kernels need (descriptor indexing, buffer device address, fp16, cooperative
 * matrices, ...) to vkCreateDevice itself, so only the extension names are requested here.
 */
public final class D4rFeatureSet {
	public static final FeatureSet FEATURE_SET = new FeatureSet(
		"D4R frame generation",
		Set.of("VK_NVX_binary_import", "VK_NVX_image_view_handle"),
		Set.of()
	);

	private D4rFeatureSet() {
	}
}
