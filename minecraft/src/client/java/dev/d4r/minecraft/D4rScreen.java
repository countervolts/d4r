package dev.d4r.minecraft;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphicsExtractor;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.network.chat.Component;

/** Settings panel (key: Insert). Plain rows: label on the left, value button on the right. */
public final class D4rScreen extends Screen {
	private static final int PANEL_W = 250, ROW = 22, BTN_W = 96;
	private final Screen parent;
	private Button enabled, factor, overlay, exact, cap;
	private int left, top, bottom;

	public D4rScreen(Screen parent) {
		super(Component.literal("D4R"));
		this.parent = parent;
	}

	@Override
	protected void init() {
		left = this.width / 2 - PANEL_W / 2;
		top = Math.max(10, this.height / 2 - 90);
		int bx = left + PANEL_W - 8 - BTN_W, y = top + 38;
		enabled = row(bx, y, b -> {
			D4rConfig.update("enabled", Boolean.toString(!D4rConfig.get().enabled()));
			FrameGen.retry();
			refresh();
		});
		factor = row(bx, y + ROW, b -> {
			int f = D4rConfig.get().factor() + 1;
			D4rConfig.update("factor", Integer.toString(f > 6 ? 2 : f));
			FrameGen.retry();
			refresh();
		});
		exact = row(bx, y + 2 * ROW, b -> {
			D4rConfig.update("exact", Boolean.toString(!D4rConfig.get().exact()));
			refresh();
		});
		cap = row(bx, y + 3 * ROW, b -> {
			D4rConfig.update("cap", Boolean.toString(!D4rConfig.get().cap()));
			refresh();
		});
		overlay = row(bx, y + 4 * ROW + 12, b -> {
			D4rConfig.update("overlay", Boolean.toString(!D4rConfig.get().overlay()));
			refresh();
		});
		bottom = y + 5 * ROW + 20;
		addRenderableWidget(Button.builder(Component.literal("Done"), b -> onClose()).bounds(left + PANEL_W / 2 - 50, bottom + 18, 100, 20).build());
		refresh();
	}

	private Button row(int x, int y, Button.OnPress press) {
		return addRenderableWidget(Button.builder(Component.empty(), press).bounds(x, y, BTN_W, 18).build());
	}

	private void refresh() {
		D4rConfig c = D4rConfig.get();
		enabled.setMessage(Component.literal(c.enabled() ? "On" : "Off"));
		factor.setMessage(Component.literal(c.factor() + "x"));
		exact.setMessage(Component.literal(c.exact() ? "Exact (restart)" : "Fast (restart)"));
		cap.setMessage(Component.literal(c.cap() ? "Refresh rate" : "Uncapped"));
		overlay.setMessage(Component.literal(c.overlay() ? "On" : "Off"));
	}

	@Override
	public void extractRenderState(GuiGraphicsExtractor g, int mouseX, int mouseY, float a) {
		int panelBottom = bottom + 46;
		g.fill(left, top, left + PANEL_W, panelBottom, 0xD0101010);
		g.fill(left, top, left + PANEL_W, top + 1, 0xFF5A5A5A);
		g.fill(left, panelBottom - 1, left + PANEL_W, panelBottom, 0xFF5A5A5A);
		g.fill(left, top, left + 1, panelBottom, 0xFF5A5A5A);
		g.fill(left + PANEL_W - 1, top, left + PANEL_W, panelBottom, 0xFF5A5A5A);
		g.text(this.font, "D4R", left + 8, top + 8, 0xFFFFFFFF, true);
		String right = "DLSS-G Lite";
		g.text(this.font, right, left + PANEL_W - 8 - this.font.width(right), top + 8, 0xFFAAAAAA, true);
		g.fill(left + 6, top + 22, left + PANEL_W - 6, top + 23, 0xFF3A3A3A);
		int y = top + 38;
		label(g, "Frame generation", y);
		label(g, "Frames shown per real", y + ROW);
		label(g, "Kernels", y + 2 * ROW);
		label(g, "Frame rate limit", y + 3 * ROW);
		g.fill(left + 6, y + 4 * ROW + 4, left + PANEL_W - 6, y + 4 * ROW + 5, 0xFF3A3A3A);
		label(g, "FPS overlay", y + 4 * ROW + 12);
		g.text(this.font, FrameGen.status(), left + 8, bottom + 2, 0xFF888888, true);
		super.extractRenderState(g, mouseX, mouseY, a);
	}

	private void label(GuiGraphicsExtractor g, String s, int y) {
		g.text(this.font, s, left + 8, y + 5, 0xFFDDDDDD, true);
	}

	@Override
	public void onClose() {
		Minecraft.getInstance().gui.setScreen(parent);
	}
}
