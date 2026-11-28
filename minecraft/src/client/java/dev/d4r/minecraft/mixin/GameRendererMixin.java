package dev.d4r.minecraft.mixin;

import dev.d4r.minecraft.FrameGen;
import net.minecraft.client.renderer.GameRenderer;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.ModifyVariable;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Frame lifecycle hooks (see docs/rendering-pipeline.md):
 *   render():       [start] ... renderLevel() ... [hudless point] GUI ... [end]
 *   renderLevel():  camera capture, forced consistent depth
 */
@Mixin(GameRenderer.class)
public abstract class GameRendererMixin {
	@Inject(method = "render", at = @At("HEAD"))
	private void d4r$renderStart(CallbackInfo ci) {
		FrameGen.renderStart();
		dev.d4r.minecraft.GpuTimer.stamp(0);
	}

	@Inject(method = "render", at = @At("RETURN"))
	private void d4r$renderEnd(CallbackInfo ci) {
		FrameGen.renderEnd();
	}

	@Inject(method = "render", at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/GameRenderer;renderLevel()V"))
	private void d4r$beginWorld(CallbackInfo ci) {
		FrameGen.beginWorld((GameRenderer) (Object) this);
	}

	/** World (post effects and 3D hand included) is finished and the GUI has not started. */
	@Inject(method = "render", at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/fog/FogRenderer;endFrame()V"))
	private void d4r$hudlessPoint(CallbackInfo ci) {
		FrameGen.hudlessPoint((GameRenderer) (Object) this);
	}

	/** The hand is drawn after the world with its own depth clear; consistent-depth mode keeps the world's depth in the main target. */
	@ModifyVariable(method = "renderLevel", at = @At(value = "STORE"), ordinal = 2)
	private boolean d4r$forceConsistentDepth(boolean consistentDepthRequired) {
		return consistentDepthRequired || FrameGen.wantsConsistentDepth();
	}

	/** The level projection (bobbing included) is final here. */
	@ModifyArg(method = "renderLevel", at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/ProjectionMatrixBuffer;getBuffer(Lorg/joml/Matrix4f;)Lcom/mojang/renderpearl/api/buffers/GpuBufferSlice;"), index = 0)
	private Matrix4f d4r$captureProjection(Matrix4f projection) {
		return FrameGen.captureProjection((GameRenderer) (Object) this, projection);
	}
}
