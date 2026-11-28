package dev.d4r.minecraft.mixin;

import com.mojang.renderpearl.backend.vulkan.VulkanFeatureSets;
import com.mojang.renderpearl.backend.vulkan.init.FeatureSet;
import dev.d4r.minecraft.D4rFeatureSet;
import java.util.HashSet;
import java.util.Set;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Device creation hook. VulkanBackend.createDevice() enables every FeatureSet in optionalFeatureSets() whose extensions
 * and features the physical device supports, before vkCreateDevice. Adding D4R's set here is the only point where the
 * device's extension/feature list can still be extended.
 */
@Mixin(VulkanFeatureSets.class)
public abstract class VulkanFeatureSetsMixin {
	@Inject(method = "optionalFeatureSets", at = @At("RETURN"), cancellable = true)
	private static void d4r$addFeatureSet(CallbackInfoReturnable<Set<FeatureSet>> cir) {
		// Always requested, whatever the on/off switch says: device features cannot be added after the device exists,
		// so the in-game toggle would otherwise only work if it was already on at launch.
		Set<FeatureSet> sets = new HashSet<>(cir.getReturnValue());
		sets.add(D4rFeatureSet.FEATURE_SET);
		cir.setReturnValue(sets);
	}
}
