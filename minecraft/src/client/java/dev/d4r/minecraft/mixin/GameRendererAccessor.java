package dev.d4r.minecraft.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.state.GameRenderState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Mutable;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(GameRenderer.class)
public interface GameRendererAccessor {
	@Accessor("mainRenderTarget")
	RenderTarget d4r$mainRenderTarget();

	@Mutable
	@Accessor("mainRenderTarget")
	void d4r$setMainRenderTarget(RenderTarget target);

	@Accessor("hud3DTarget")
	RenderTarget d4r$hud3DTarget();

	@Mutable
	@Accessor("hud3DTarget")
	void d4r$setHud3DTarget(RenderTarget target);

	@Accessor("gameRenderState")
	GameRenderState d4r$gameRenderState();
}
