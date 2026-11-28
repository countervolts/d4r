package dev.d4r.minecraft;

import net.minecraft.client.Minecraft;
import net.minecraft.client.Screenshot;
import net.minecraft.client.gui.screens.TitleScreen;
import net.minecraft.client.gui.screens.worldselection.WorldOpenFlows;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.level.levelgen.WorldOptions;
import net.minecraft.world.level.levelgen.presets.WorldPresets;

/** Test automation, off unless 'autoWorld' is set: opens a fresh demo world from the title screen (client tick), turns left, screenshots. */
public final class AutoTest {
	private static boolean worldRequested, shot;
	private static long c0, n0;

	private AutoTest() {
	}

	public static void tick() {
		Minecraft mc = Minecraft.getInstance();
		String name = System.getProperty("d4r.autoWorld", "");
		if (name.isBlank()) {
			return;
		}
		if (mc.level != null && mc.player != null) {
			long t = mc.level.getGameTime();
			mc.options.keyLeft.setDown(t > 60 && t < 400);
			if (t == 100 && c0 == 0) {
				c0 = FrameGen.renderedFrames();
				FrameGen.waitNs = FrameGen.blitNs = FrameGen.acquireNs = FrameGen.dropped = 0;
				FrameGen.jobLagNs = FrameGen.jobLenNs = FrameGen.jobCount = FrameGen.pFence = FrameGen.pReset = FrameGen.pNative = FrameGen.pAcq = FrameGen.pRec = FrameGen.pSub = FrameGen.pPres = FrameGen.pCount = 0;
				n0 = System.nanoTime();
			}
			if (t >= 300 && !shot) {
				D4rClient.LOGGER.info("[D4R] auto test: rendered {} fps over ticks 100-300; per frame ms: waitPresenter {} blitHook {} acquire {}; dropped {}", String.format("%.1f", (FrameGen.renderedFrames() - c0) * 1e9 / (System.nanoTime() - n0)), String.format("%.2f", FrameGen.waitNs / 1e6 / Math.max(1, FrameGen.renderedFrames() - c0)), String.format("%.2f", FrameGen.blitNs / 1e6 / Math.max(1, FrameGen.renderedFrames() - c0)), String.format("%.2f", FrameGen.acquireNs / 1e6 / Math.max(1, FrameGen.renderedFrames() - c0)), FrameGen.dropped);
				shot = true;
				D4rClient.LOGGER.info("[D4R] presenter: job start lag {} ms, job length {} ms, jobs {}", String.format("%.2f", FrameGen.jobLagNs / 1e6 / Math.max(1, FrameGen.jobCount)), String.format("%.2f", FrameGen.jobLenNs / 1e6 / Math.max(1, FrameGen.jobCount)), FrameGen.jobCount);
				D4rClient.LOGGER.info("[D4R] per present ms: acquire {} record {} submit {} present {} reset+begin {} native {} fence {} ({} presents)", String.format("%.2f", FrameGen.pAcq / 1e6 / Math.max(1, FrameGen.pCount)), String.format("%.2f", FrameGen.pRec / 1e6 / Math.max(1, FrameGen.pCount)), String.format("%.2f", FrameGen.pSub / 1e6 / Math.max(1, FrameGen.pCount)), String.format("%.2f", FrameGen.pPres / 1e6 / Math.max(1, FrameGen.pCount)), String.format("%.2f", FrameGen.pReset / 1e6 / Math.max(1, FrameGen.pCount)), String.format("%.2f", FrameGen.pNative / 1e6 / Math.max(1, FrameGen.pCount)), String.format("%.2f", FrameGen.pFence / 1e6 / Math.max(1, FrameGen.pCount)), FrameGen.pCount);
				Screenshot.grab(mc.gameDirectory, "d4r_fg_main.png", mc.gameRenderer.mainRenderTarget(), 1, c -> D4rClient.LOGGER.info("[D4R] {}", c.getString()));
				D4rClient.LOGGER.info("[D4R] auto test: {} shown fps {} real {}", FrameGen.status(), FrameGen.shownFps(), FrameGen.realFps());
			}
			if (t >= 500) {
				D4rClient.LOGGER.info("[D4R] auto test done; stopping");
				mc.stop();
			}
			return;
		}
		if (worldRequested || !(mc.gui.screen() instanceof TitleScreen title)) {
			return;
		}
		worldRequested = true;
		WorldOpenFlows flows = mc.createWorldOpenFlows();
		flows.createFreshLevel(name, MinecraftServer.DEMO_SETTINGS, WorldOptions.DEMO_OPTIONS, WorldPresets::createNormalWorldDimensions, title);
	}
}
