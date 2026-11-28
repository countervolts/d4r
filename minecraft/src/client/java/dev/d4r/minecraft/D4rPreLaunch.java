package dev.d4r.minecraft;

import net.fabricmc.loader.api.entrypoint.PreLaunchEntrypoint;

/**
 * Runs before Minecraft creates its Vulkan instance: puts the nvxrt layer (VK_NVX_binary_import emulation, which the
 * frame-generation kernels launch through) into the loader's environment. Without it the mod stays inert.
 */
public final class D4rPreLaunch implements PreLaunchEntrypoint {
	@Override
	public void onPreLaunch() {
		D4rConfig c = D4rConfig.get();
		if (!c.enabled()) {
			return;
		}
		String error = NativeBridge.load();
		if (error != null) {
			D4rClient.LOGGER.error("[D4R] {}", error);
			return;
		}
		java.nio.file.Path layer = D4rConfig.findDir("layer");
		if (layer == null) {
			D4rClient.LOGGER.error("[D4R] nvxrt layer folder 'layer' not found next to libd4r_mc.so (tried {})", D4rConfig.searchDirs());
			return;
		}
		String log = c.log() ? D4rConfig.d4rLogFile().toString() : "";
		NativeBridge.setupEnvironment(layer.toString(), c.kernelCache(), c.exact(), log);
		D4rClient.LOGGER.info("[D4R] nvxrt layer {} enabled (kernel cache {}, {} mode)", layer, c.kernelCache(), c.exact() ? "exact" : "fast");
	}
}
