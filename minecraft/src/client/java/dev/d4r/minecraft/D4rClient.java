package dev.d4r.minecraft;

import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.minecraft.client.KeyMapping;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class D4rClient implements ClientModInitializer {
	public static final Logger LOGGER = LoggerFactory.getLogger("d4r");

	private static KeyMapping settingsKey;

	@Override
	public void onInitializeClient() {
		settingsKey = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.d4r.settings", InputConstants.KEY_INSERT, KeyMapping.Category.MISC));
		Overlay.register();
		ClientTickEvents.END_CLIENT_TICK.register(mc -> {
			AutoTest.tick();
			while (settingsKey.consumeClick()) {
				mc.gui.setScreen(new D4rScreen(mc.gui.screen()));
			}
		});
		LOGGER.info("[D4R] Frame Generation mod loaded; config file {} -> {}", D4rConfig.file(), D4rConfig.get());
		LOGGER.info("[D4R] D4R only runs on Minecraft's Vulkan renderer (launch with --graphicsBackend vulkan or set Options > Video > Graphics API).");
	}
}
