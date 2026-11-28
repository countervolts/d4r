# D4R DLSS Frame Generation for Minecraft (Fabric, Vulkan, AMD RDNA3)

A Fabric client mod that runs NVIDIA's **DLSS-G Lite** multi frame generation (2x to 6x) on Minecraft 26.3's native Vulkan renderer on
RDNA3 + Mesa RADV, using d4r's engine: the real DLSS-G kernels, translated to SPIR-V and launched through d4r's `nvxrt` Vulkan layer.
No Wine, no D3D12 and no NVIDIA driver at run time.

Design/Docs: [docs/architecture.md](docs/architecture.md),
[docs/presentation.md](docs/presentation.md), [docs/rendering-pipeline.md](docs/rendering-pipeline.md),
[docs/vulkan-requirements.md](docs/vulkan-requirements.md), [docs/testing.md](docs/testing.md), [docs/mfg-feasibility.md](docs/mfg-feasibility.md).

## What you need
- RX 7000 series (RDNA3), Mesa RADV 26.2.5+, Minecraft 26.3 with Fabric (Java 25), run with `--graphicsBackend vulkan`.
- d4r's DLSS-G kernel cache for your DLSS-G DLL (`~/.cache/nvxrt-gfx11`, created by `d4r-features.py --fg dlssg` / `install.sh`) and the
  `nvngx_dlssg.dll` 310.9.16 (Lite) it was built from. NVIDIA files are not in this repository; the scripts below read them from your machine.
- A recorded launch script for your window size and factor (below). One-time, needs d4r's Proton harness.

## Build
```sh
scripts/build-native.sh          # libd4r_mc.so, the nvxrt layer, fg_replay_test  ->  build/d4r/
./gradlew build                  # build/libs/d4r-minecraft-1.0.0.jar
scripts/build-fg-scripts.sh 2560 1440 "1 2 3 4 5" /path/to/patched/vkd3d   # scripts for 2x..6x at 2560x1440
```
`build-fg-scripts.sh` runs the real DLL once per factor in d4r's harness (Proton, patched vkd3d-proton, `fg/dlssg/harness`) with a traced copy of the
layer (`patches/nvxrt-resource-trace.patch`), and converts the trace (`tools/mkscript.py`). Output: `~/.cache/d4r-mc-fg/WxH/mfcN/` including `weights.bin`
(a slice of the DLL's own weights: never commit or redistribute it).

## Install
Copy `build/libs/d4r-minecraft-1.0.0.jar` into the instance's `mods/` and `build/d4r/` (libd4r_mc.so + layer/) to `<instance>/minecraft/d4r/`.
Insert opens the settings screen; `config/d4r-minecraft.properties` is hot-reloaded (`factor`, `exact`, `overlay`, ...).
The window/main-target size must match a recorded script (2560x1440 by default); FSR/Vitrail style render scaling is not used.

## Dev
`./gradlew runClient` (adds `--graphicsBackend vulkan` and `-Dd4r.d4rDir=build/d4r`). `D4R_AUTO=name ./gradlew runClient` creates a demo world, turns left for ~20 s, saves `run/screenshots/d4r_fg_main.png` and quits (needs fullscreen at a recorded resolution).
