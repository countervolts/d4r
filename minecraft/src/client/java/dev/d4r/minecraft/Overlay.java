package dev.d4r.minecraft;

import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.hud.VanillaHudElements;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.resources.Identifier;

/** Small FPS / frame time / D4R readout in the top-left corner. */
public final class Overlay {
	private static long windowStart = System.nanoTime();
	private static int frames;
	private static double fps, frameMs;

	private Overlay() {
	}

	public static void register() {
		HudElementRegistry.attachElementAfter(VanillaHudElements.MISC_OVERLAYS, Identifier.fromNamespaceAndPath("d4r", "overlay"),
			(graphics, tick) -> draw(graphics));
	}

	private static void draw(GuiGraphicsExtractor g) {
		D4rConfig c = D4rConfig.get();
		if (!c.overlay()) {
			return;
		}
		frames++;
		long now = System.nanoTime();
		if (now - windowStart >= 500_000_000L) {
			fps = frames * 1e9 / (now - windowStart);
			frameMs = 1000.0 / Math.max(fps, 0.001);
			frames = 0;
			windowStart = now;
		}
		Minecraft mc = Minecraft.getInstance();
		String l1 = FrameGen.isActive() && FrameGen.shownFps() >= 0
			? String.format("%.0f FPS shown  (%.0f real, %dx)", FrameGen.shownFps(), FrameGen.realFps(), FrameGen.factor())
			: String.format("%.0f FPS  %.1f ms", fps, frameMs);
		double cpu = FrameGen.cpuMs();
		String lc = cpu < 0 ? null : String.format("CPU render %.1f ms", cpu);
		String l2 = "D4R " + FrameGen.status();
		String lg = GpuTimer.line();
		String lp = FrameGen.isActive() && FrameGen.paceAvgMs > 0 ? String.format("pace %.1f ms  sigma %.1f  max %.0f  late %d", FrameGen.paceAvgMs, FrameGen.paceSigmaMs, FrameGen.paceMaxMs, FrameGen.paceLate) : null;
		String l3 = null;
		int w = Math.max(Math.max(Math.max(mc.font.width(l1), mc.font.width(lg == null ? "" : lg)), mc.font.width(lc == null ? "" : lc)), mc.font.width(l2)) + 8;
		int h = (2 + (lg == null ? 0 : 1) + (lc == null ? 0 : 1) + (lp == null ? 0 : 1)) * 10 + 6;
		g.fill(2, 2, 2 + w, 2 + h, 0x90000000);
		g.text(mc.font, l1, 6, 5, 0xFFFFFFFF, true);
		g.text(mc.font, l2, 6, 15, 0xFFAAAAAA, true);
		int ly = 25;
		if (l3 != null) {
			g.text(mc.font, l3, 6, ly, 0xFFAAAAAA, true);
			ly += 10;
		}
		if (lg != null) {
			g.text(mc.font, lg, 6, ly, 0xFFAAAAAA, true);
			ly += 10;
		}
		if (lc != null) {
			g.text(mc.font, lc, 6, ly, 0xFFAAAAAA, true);
			ly += 10;
		}
		if (lp != null) {
			g.text(mc.font, lp, 6, ly, 0xFFAAAAAA, true);
		}
	}
}
