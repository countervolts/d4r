<!-- Modified in this fork for CUDA Ray Reconstruction support and validation (2026). -->
# Supported games

The following DirectX 12 games run DLSS Super Resolution through d4r on the tested Radeon RX 7700 XT Linux setup with GE-Proton 11-3. All three Super Resolution models in the table run in each game; image-quality issues are noted below. Experimental Ray Reconstruction has a separate RX 9070 XT validation described under Cyberpunk 2077.

| Game | E: DLSS 3 CNN | K: DLSS 4 | M: DLSS 4.5 |
|---|---|---|---|
| SILENT HILL Townfall | Works | Works | Works |
| Ghost of Tsushima DIRECTOR'S CUT | Works | Works | Works |
| Ready or Not | Works | Works; visible graphical artifacts with hdr enabled | Works |
| Clair Obscur: Expedition 33 | Works | Works | Works |
| Marvel's Spider-Man: Miles Morales | Works | Works; minor artifacts when HDR enabled and while the game is paused | Works |
| The Last of Us Part I | Works | Works; visible graphical artifacts, including with HDR enabled | Works |
| Control Ultimate Edition | Works | Works | Works |
| Cyberpunk 2077 | Works; occasional visual artifacts during gameplay | Works | Works |
| Alan Wake 2 | Works | Works | Works |
| SILENT HILL 2 (2024) | Works | Works | Works |
| Subnautica 2 | Works | Works; visible artifacts at Quality (other quality modes are fine) | Works |
| Horizon Zero Dawn Remastered | Works; stutters | Works | Works |
| Ratchet & Clank: Rift Apart | Works; occasional artifacts in the pause menu | Works; minor artifacts with HDR enabled and occasionally in the pause menu | Works; occasional artifacts in the pause menu |
| Dying Light: The Beast | Works | Works | Works; small visual artifacts throughout |

Unsupported games:

- **DOOM: The Dark Ages:** the game uses Vulkan and has no D3D12 mode. d4r currently supports D3D12 DLSS only, so its DLSS calls cannot reach d4r. The game's hardware ray-tracing requirement is separate from this limitation.

General notes:

- Select DLSS in the game's graphics settings where it is offered. OptiScaler can also feed DLSS from a game's FSR or XeSS inputs, but those are a fallback for games that hide DLSS and can ghost or show a black screen.
- Experimental Ray Reconstruction requires a CUDA-capable 3.10.7 denoiser, matching lifetime-aware d3d12.dll/d3d12core.dll, and native FP8 WMMA for the validated gfx1201 path. Cyberpunk 2077 is verified below; RR in Alan Wake 2 and Control remains unverified, so keep it off for their documented setups.
- Saving settings from the OptiScaler overlay ("Save INI") rewrites `OptiScaler.ini` and resets d4r's values to `auto`, including frame generation (`[FrameGen] Enabled=false`, `FGInput=nofg`, `FGOutput=nofg`) and `[Inputs] EnableDlssInputs=true`. Restore them afterwards, or keep a copy of the original file.

Game-specific setup:

- **The Last of Us Part I:** the game's Streamline integration crashes on boot with OptiScaler. Rename `sl.common.dll` in the game folder to `sl.common.dll.bak` and select FSR 3.1 in the game's graphics settings; OptiScaler passes it to DLSS through d4r.

- **Control Ultimate Edition:** keep ray tracing off for the documented setup. Older builds crashed when loading gameplay with ray tracing enabled; this game's integration has not been revalidated against the new RR route.

- **Cyberpunk 2077:** GE-Proton 11-3 loads Wine's own ICU libraries instead of the game's, and the game aborts at launch (`unimplemented function icuuc.dll.u_setMemoryFunctions_65`). Add `WINEDLLOVERRIDES="icuuc,icuin=n,b"` to the launch options (fixed in GE-Proton 11-4).
  - **Ray Reconstruction:** Cyberpunk 2077 2.31 has been verified in loaded local saves on RX 9070 XT (gfx1201), GE-Proton 11-3, at 1920×1080 and 3840×2160 Ultra Performance with path tracing and RR enabled. The actual backend is `DLSSD 310.7.0`, CUDA feature 13, preset D; observed gameplay snapshots showed about 53 FPS at 1080p and 20 FPS at 4K, not comparative benchmarks.
  - Enable `[Kernels] NativeFp8 = true` and supply the CUDA-capable 3.10.7 `nvngx_dlssd.dll` beside the shim. Update game-local d3d12.dll **and** d3d12core.dll too: local overrides shadow a patched Proton prefix. RR preset E has separate temporal/alpha harness validation, not a Cyberpunk performance result. Newer DLSS 4.5 RR preset F is not implemented or validated. Scalar FP8 temporal RR is not validated after producing nonfinite output; RTX image-quality parity remains unverified.

- **Alan Wake 2:** with the game's own DLSS inputs, every model shows heavy ghosting. Set `AutoExposure=true` under `[InitFlags]` in `OptiScaler.ini` (or enable auto exposure in the OptiScaler overlay), or select FSR 3.1 in the game instead so OptiScaler passes FSR inputs to DLSS.

- **Horizon Zero Dawn Remastered:** the game loads `dxgi.dll` only from the Windows system folder, so OptiScaler is never loaded under that name. Rename `dxgi.dll` to `version.dll` and add `WINEDLLOVERRIDES="version=n,b"` to the launch options. d4r's `d3d12core.dll` is still picked up from the game folder.

- **Ratchet & Clank: Rift Apart:** set `Dxgi=false` in `OptiScaler.ini`. With ray tracing on, OptiScaler's DXGI spoofing crashes the game at startup on AMD GPUs; the `PROTON_FORCE_NVAPI=1` launch option already reports an NVIDIA GPU.

- **Dying Light: The Beast:** use DLSS inputs; with FSR or XeSS inputs the screen stays black. The game passes its motion vectors as `R16G16B16A16_TYPELESS`, which needs a d4r build from 2026-09-28 or later (older builds reject every frame and show a black screen).

Choose a model with `Model = E`, `Model = K`, or `Model = M` under `[DLSS]` in the game's `d4r/d4r.ini`. Model compatibility does not imply equal performance; see the [Townfall measurements](docs/performance.md) for measured frame rates.
