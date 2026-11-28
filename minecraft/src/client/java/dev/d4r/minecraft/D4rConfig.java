package dev.d4r.minecraft;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Properties;
import net.fabricmc.loader.api.FabricLoader;

/**
 * Settings: {@code <gameDir>/config/d4r-minecraft.properties}, overridden by D4R_MC_* environment variables and -Dd4r.* properties.
 * The file is re-read about once a second; everything except {@code exact} and the layer settings can change while the game runs.
 *
 * <pre>
 * enabled=true
 * factor=4                  presented frames per real frame: 2..6 (needs a recorded script for the window size)
 * exact=false               true: bit-exact kernels (slower); false: d4r's fast mode
 * async=false               true: run the generation on the compute queue (restart to change)
 * d4rDir=/path/to/d4r       folder with libd4r_mc.so and layer/ (default: gameDir/d4r, then the dev build)
 * kernelCache=~/.cache/nvxrt-gfx11     kernel cache built by d4r's dlssg setup
 * scriptDir=~/.cache/d4r-mc-fg         scripts from scripts/build-fg-scripts.sh (WxH/mfcN)
 * motionSign=1
 * overlay=false
 * log=true
 * </pre>
 */
public final class D4rConfig {
	private static volatile D4rConfig current;
	private static long lastCheckNanos;
	private static long lastModified = -1;

	private final boolean enabled, exact, overlay, log, async, cap, mailbox;
	private final int maxShown;
	private final int factor;
	private final float motionSign;
	private final String d4rDir, kernelCache, scriptDir;

	private D4rConfig(Properties p) {
		String home = System.getProperty("user.home");
		this.enabled = bool(p, "enabled", true);
		this.factor = Math.max(2, Math.min(6, Integer.parseInt(get(p, "factor", "4"))));
		this.exact = bool(p, "exact", false);
		this.async = bool(p, "async", false);
		this.cap = bool(p, "cap", false);
		this.mailbox = bool(p, "mailbox", true);
		this.maxShown = Integer.parseInt(get(p, "maxShownFps", "0"));
		this.overlay = bool(p, "overlay", false);
		this.log = bool(p, "log", true);
		this.motionSign = Float.parseFloat(get(p, "motionSign", "1"));
		this.d4rDir = get(p, "d4rDir", "");
		this.kernelCache = get(p, "kernelCache", home + "/.cache/nvxrt-gfx11");
		this.scriptDir = get(p, "scriptDir", home + "/.cache/d4r-mc-fg");
	}

	private static String get(Properties p, String key, String fallback) {
		String env = System.getenv("D4R_MC_" + key.replaceAll("([A-Z])", "_$1").toUpperCase(Locale.ROOT));
		if (env != null && !env.isBlank()) {
			return env;
		}
		String prop = System.getProperty("d4r." + key);
		if (prop != null && !prop.isBlank()) {
			return prop;
		}
		return p.getProperty(key, fallback);
	}

	private static boolean bool(Properties p, String key, boolean fallback) {
		return Boolean.parseBoolean(get(p, key, Boolean.toString(fallback)));
	}

	public static Path file() {
		return FabricLoader.getInstance().getConfigDir().resolve("d4r-minecraft.properties");
	}

	public static Path d4rLogFile() {
		return FabricLoader.getInstance().getGameDir().resolve("logs").resolve("d4r-nvxrt.log");
	}

	/** Folders that may hold libd4r_mc.so and layer/, most specific first. */
	public static List<Path> searchDirs() {
		List<Path> dirs = new ArrayList<>();
		String configured = get(new Properties(), "d4rDir", "");
		if (current != null && !current.d4rDir.isBlank()) {
			configured = current.d4rDir;
		}
		if (!configured.isBlank()) {
			dirs.add(Path.of(configured));
		}
		dirs.add(FabricLoader.getInstance().getGameDir().resolve("d4r"));
		dirs.add(Path.of(System.getProperty("user.dir"), "build", "d4r"));
		dirs.add(Path.of(System.getProperty("user.dir"), "..", "build", "d4r"));
		return dirs;
	}

	public static Path findDir(String sub) {
		for (Path d : searchDirs()) {
			if (Files.isDirectory(d.resolve(sub))) {
				return d.resolve(sub).toAbsolutePath().normalize();
			}
		}
		return null;
	}

	public static D4rConfig get() {
		long now = System.nanoTime();
		D4rConfig c = current;
		if (c != null && now - lastCheckNanos < 1_000_000_000L) {
			return c;
		}
		lastCheckNanos = now;
		Path file = file();
		try {
			long modified = Files.exists(file) ? Files.getLastModifiedTime(file).toMillis() : 0;
			if (c == null || modified != lastModified) {
				Properties p = new Properties();
				if (modified != 0) {
					try (InputStream in = Files.newInputStream(file)) {
						p.load(in);
					}
				}
				lastModified = modified;
				current = new D4rConfig(p);
				if (c != null) {
					D4rClient.LOGGER.info("[D4R] config reloaded: {}", current);
				}
			}
		} catch (IOException | RuntimeException e) {
			D4rClient.LOGGER.warn("[D4R] cannot read {}: {}", file, e.toString());
			if (current == null) {
				current = new D4rConfig(new Properties());
			}
		}
		return current;
	}

	public boolean enabled() {
		return enabled;
	}

	public int factor() {
		return factor;
	}

	/** Generation on Minecraft's separate compute queue (measured no faster than the graphics queue, see STATUS.md). */
	public boolean async() {
		return async;
	}

	/** Limit the real frame rate so that shown frames never exceed the display's refresh rate (or maxShownFps). */
	public boolean cap() {
		return cap;
	}

	/** 0 = the display's refresh rate. */
	public int maxShownFps() {
		return maxShown;
	}

	/** With vsync off, present with MAILBOX (tear-free, still uncapped) instead of IMMEDIATE while frame generation is on. */
	public boolean mailbox() {
		return mailbox;
	}

	public boolean exact() {
		return exact;
	}

	public boolean overlay() {
		return overlay;
	}

	public boolean log() {
		return log;
	}

	public float motionSign() {
		return motionSign;
	}

	public String kernelCache() {
		return kernelCache;
	}

	public String scriptDir() {
		return scriptDir;
	}

	/** Writes one key into the properties file (the file is re-read right away). Environment overrides still win. */
	public static synchronized void update(String key, String value) {
		Properties p = new Properties();
		Path file = file();
		try {
			if (Files.exists(file)) {
				try (InputStream in = Files.newInputStream(file)) {
					p.load(in);
				}
			}
			p.setProperty(key, value);
			Files.createDirectories(file.getParent());
			try (java.io.OutputStream out = Files.newOutputStream(file)) {
				p.store(out, "D4R frame generation (changed in game)");
			}
		} catch (IOException e) {
			D4rClient.LOGGER.warn("[D4R] cannot write {}: {}", file, e.toString());
		}
		lastCheckNanos = 0;
		current = null;
		get();
	}

	@Override
	public String toString() {
		return "enabled=" + enabled + " factor=" + factor + " exact=" + exact + " motionSign=" + motionSign + " overlay=" + overlay;
	}
}
