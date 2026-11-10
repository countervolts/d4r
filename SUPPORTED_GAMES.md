<!-- Modified in this fork for CUDA Ray Reconstruction support and validation (2026). -->
# Supported games

The following DirectX 12 games run DLSS Super Resolution through d4r on the tested Radeon RX 7700 XT Linux setup with GE-Proton 11-3. All three Super Resolution models in the table run in each game unless a cell says otherwise; image-quality issues are noted below. Experimental Ray Reconstruction has a separate RX 9070 XT validation described under Cyberpunk 2077.

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
| Assassin's Creed Shadows | Not tested | Not tested | Works; DLSS only appears with NVAPI spoofing, see below |

General notes:

- Select DLSS in the game's graphics settings where it is offered. OptiScaler can also feed DLSS from a game's FSR or XeSS inputs, but those are a fallback for games that hide DLSS and can ghost or show a black screen.
- Experimental Ray Reconstruction requires a CUDA-capable 3.10.7 denoiser, matching lifetime-aware d3d12.dll/d3d12core.dll, and native FP8 WMMA for the validated gfx1201 path. Cyberpunk 2077 is verified below; RR in Alan Wake 2 and Control remains unverified, so keep it off for their documented setups.
- Change OptiScaler settings in the Insert overlay and use **Save INI** normally. Upscaler, preset, sharpening, FG and overlay choices are yours; d4r setup preserves them on later launches and runtime updates. OptiScaler can write `auto` for settings equal to its defaults. Keep the explicit `Libraries.NvngxPath` pointing at d4r when you want DLSS to use d4r; selecting FSR or XeSS uses that backend instead. The [FG setup command](docs/frame-generation.md) is optional and only changes configuration when you run it.

Game-specific setup:

- **Ghost of Tsushima DIRECTOR'S CUT (native Windows):** preset M runs its network on the model's native HIP layers (d3dhip) when `d4r\engine\m` and `m-ldr` were built with `compile_m.py --hip KERNEL_DIR --dxil yes` (add `--ldr` for the LDR package; the game passes the LDR flag), and `d4r_hip_net.dll` sits beside `d4r\nvngx.dll`. Verified 2026-10-30 on RX 9070 XT with OptiScaler, 1920x1080 → 3840x2160 preset M (LDR) at 55-68 FPS; `D4R_DX_NET=dxil` goes back to the engine's DXIL network.

- **The Last of Us Part I:** the game's Streamline integration crashes on boot with OptiScaler. Rename `sl.common.dll` in the game folder to `sl.common.dll.bak` and select FSR 3.1 in the game's graphics settings; OptiScaler passes it to DLSS through d4r.

- **Control Ultimate Edition:** keep ray tracing off for the documented setup. Older builds crashed when loading gameplay with ray tracing enabled; this game's integration has not been revalidated against the new RR route.

- **Cyberpunk 2077:** GE-Proton 11-3 loads Wine's own ICU libraries instead of the game's, and the game aborts at launch (`unimplemented function icuuc.dll.u_setMemoryFunctions_65`). Add `WINEDLLOVERRIDES="icuuc,icuin=n,b"` to the launch options (fixed in GE-Proton 11-4).
  - **Ray Reconstruction:** Cyberpunk 2077 2.31 has been verified in loaded local saves on RX 9070 XT (gfx1201), GE-Proton 11-3, at 1920×1080 and 3840×2160 Ultra Performance with path tracing and RR enabled. The actual backend is `DLSSD 310.7.0`, CUDA feature 13, preset D; observed gameplay snapshots showed about 53 FPS at 1080p and 20 FPS at 4K, not comparative benchmarks.
  - Enable `[Kernels] NativeFp8 = true` and supply the CUDA-capable 3.10.7 `nvngx_dlssd.dll` beside the shim. Update game-local d3d12.dll **and** d3d12core.dll too: local overrides shadow a patched Proton prefix. RR preset E has separate temporal/alpha harness validation, not a Cyberpunk performance result. Newer DLSS 4.5 RR preset F is not implemented or validated. Scalar FP8 temporal RR is not validated after producing nonfinite output; RTX image-quality parity remains unverified.

- **Alan Wake 2:** with the game's own DLSS inputs, every model shows heavy ghosting. Set `AutoExposure=true` under `[InitFlags]` in `OptiScaler.ini` (or enable auto exposure in the OptiScaler overlay), or select FSR 3.1 in the game instead so OptiScaler passes FSR inputs to DLSS.

- **Horizon Zero Dawn Remastered:** the game loads `dxgi.dll` only from the Windows system folder, so OptiScaler is never loaded under that name. Rename `dxgi.dll` to `version.dll` and add `WINEDLLOVERRIDES="version=n,b"` to the launch options. d4r's `d3d12core.dll` is still picked up from the game folder.

- **Ratchet & Clank: Rift Apart:** set `Dxgi=false` in `OptiScaler.ini`. With ray tracing on, OptiScaler's DXGI spoofing crashes the game at startup on AMD GPUs; d4r's `d4r_boot.dll` already reports an NVIDIA GPU to the game.

- **Dying Light: The Beast:** use DLSS inputs; with FSR or XeSS inputs the screen stays black. The game passes its motion vectors as `R16G16B16A16_TYPELESS`, which needs a d4r build from 2026-09-28 or later (older builds reject every frame and show a black screen).

- **Assassin's Creed Shadows:** the game exposes DLSS through Streamline 2.7.15 (`NVStreamline\production`), which OptiScaler's Streamline hook hands to d4r. DLSS only appears in the game's menu when NVAPI reports an NVIDIA GPU: add `PROTON_ENABLE_NVAPI=1 PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100` to the launch options (without them OptiScaler logs `isNvidia Detected: false`, `Not running on Nvidia, disabling DLSS`, and the option is missing), and `WINEDLLOVERRIDES="dxgi=n,b;d3d12=n,b;d3d12core=n,b"` for the game-local OptiScaler and d4r vkd3d-proton pair. Steam adds `-uplay_steam_mode`; launching `ACShadows.exe` directly without it exits once Ubisoft Connect has started. Verified 2026-10-31 on RX 9070 XT (gfx1201, Mesa RADV), GE-Proton10-34, `Model = M`, HDR: the game requests 2258x1270 → 3840x2160 and evaluates a fixed 1920x1080 subrect, `experimental vkhip (M/L) ready on gfx1201: 10 native dispatches for 1920x1080`, `feature 29185 runs preset M in the game's command list (1920x1080 -> 3840x2160), network on vkhip`, frames recorded continuously. A render-size change recreates the engine, which is what the fixed-size vkhip route needs. The game ships no `nvngx_dlssd.dll`, so Ray Reconstruction stays off; E and K are untested here. Ubisoft's `aegir_f.dll` is present, so keep this to the single-player game.

Choose a model with `Model = E`, `Model = K`, or `Model = M` under `[DLSS]` in the game's `d4r/d4r.ini`. Model compatibility does not imply equal performance; see the [Townfall measurements](docs/performance.md) for measured frame rates.

Vulkan games (native Vulkan frontend, engine path; see docs/vulkan.md):

- **Enshrouded** (Vulkan only): preset M verified in game on RX 9070 (gfx1201), GE-Proton11-7, through the game's own DLSS
  inputs. Install OptiScaler as `dbghelp.dll` (the exe imports it before it creates its Vulkan device; `dxgi.dll` loads
  too late) with `WINEDLLOVERRIDES="dbghelp=n,b"`, and set in `OptiScaler.ini`: `[Upscalers] VulkanUpscaler=dlss`,
  `[Spoofing] Vulkan=true`, `VulkanExtensionSpoofing=false`, `Dxgi=false` (a Vulkan game needs no DXGI spoofing; the
  verified setup ran without it), and `[InitFlags] AutoExposure=true` (the game sends no exposure; needed for K). The game's motion vectors are `R16G16_SNORM`, which the staging path rejects. Use the release's `dxgi.dll`
  renamed to `dbghelp.dll` (it loads `d4r_boot.dll`, so no further launch options are needed); another OptiScaler
  build needs `PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100`. The lsfg-vk layer made the game's swap chain creation
  fail under GE-Proton11-7 even without d4r; disable it (`DISABLE_LSFGVK=1`) if the game exits with
  `Graphics_InitializationFailed`.

- **Indiana Jones and the Great Circle** (native Windows, AMD driver): presets M and L verified on vkhip on RX 9070 XT
  (gfx1201), AMD driver 26.10.1 (LLPC), through the game's Streamline DLSS inputs, HDR, 2258x1270 -> 3840x2160. Install
  OptiScaler as `winmm.dll` (the exe imports WINMM) with `[Upscalers] VulkanUpscaler=dlss`, `NvngxPath` at d4r's
  `nvngx.dll`, and `[Spoofing] ForceNvidia=true`; the OptiScaler build needs `patches/optiscaler-windows/0002`, which adds
  the NGX module's Vulkan device extensions (`VK_KHR_pipeline_binary`, `VK_KHR_maintenance5`) and features to the game's
  device. Without them d4r cannot load vkhip's binaries and falls back to CUDA staging (~12 fps). The game's output is
  `B10G11R11_UFLOAT_PACK32`, which needs packages with `down_m_r11g11b10*.spv` (compile_m.py from 2026-10-31). Measured
  in game: M ~92 fps, L ~46 fps. Pick DLSS in the game's menu; `d4r.ini` needs `[Engine] Enabled = true`,
  `ExperimentalVkHip = true`.

## Community compatibility reports

The following reports were contributed by **ribbit902** from the [d4r Discord server](https://discord.gg/mgTWpSq8f5). These are separate from the model-by-model validation above; GPU, driver, d4r version and individual model coverage were not supplied. Timings are reporter observations, not directly comparable benchmarks.

| Game | Reported status | Notes |
|---|---|---|
| Control Ultimate Edition | 🟩 Works | Works with ray tracing disabled. |
| Crimson Desert | Not specified | Listed without a result or setup notes. |
| Cyberpunk 2077 | 🟩 Works | Both Ray Reconstruction and Super Resolution work. |
| Deep Rock Galactic | 🟩 Works | No additional setup notes supplied. |
| DOOM Eternal | 🟥 Does not work | Unable to get DLSS inputs with d4r without OptiPatcher. |
| DOOM: The Dark Ages | 🟥 Does not work | Reports a ray tracing incompatible GPU error. |
| Forza Horizon 5 | 🟥 Does not work | Crashes on startup with d4r without OptiPatcher. |
| Ghost of Tsushima DIRECTOR'S CUT | 🟩 Works | Use DLSS inputs; FSR and XeSS inputs produce black or grey screens. |
| Ghostrunner | 🟩 Works | No additional setup notes supplied. |
| Ghostrunner 2 | 🟩 Works | DLSS inputs add some performance overhead; XeSS inputs do not work. FSR inputs recommended. |
| Half-Life 2 RTX | 🟥 Does not work | “Failed to create D3D device” error and a driver update prompt; the reporter attributes the prompt to AMD driver version numbers being lower than NVIDIA's. |
| Helldivers 2 | 🟥 Does not work | Crashes on startup; the reporter attributes this to the game now using EAC. |
| Horizon Zero Dawn Remastered | 🟩 Works | Requires the specific wiki setup; see the game-specific setup above. |
| No Man's Sky | 🟩 Works | Install OptiScaler as `dbghelp.dll`. All DLSS presets have a larger performance cost than usual; preset M at Performance was reported at 6 ms. |
| Overwatch | 🟥 Does not work | OptiScaler loads, but no DLSS upscaler is available. |
| Palworld | 🟩 Works | No additional setup notes supplied. |
| Portal RTX | 🟥 Does not work | “Failed to create D3D device” error and a driver update prompt; the reporter attributes the prompt to AMD driver version numbers being lower than NVIDIA's. |
| Satisfactory | 🟩 Works | Install in `Engine/Binaries/Win64`. DLSS shows shimmering absent with FSR 4, more noticeable at lower resolutions. Preset M has a larger performance cost than usual: Performance (50%) 2.4 ms, Balanced (58%) 2.8 ms, Quality (66%) 3.3 ms, Native AA 6 ms. |
| Sons of the Forest | 🟥 Does not work | Reaches the menu and exposes DLSS inputs, but freezes when rendering gameplay. The reporter suspects the DX11 rendering path. |
| Subnautica 2 | 🟩 Works | No additional setup notes supplied. |
| Warhammer 40,000: Space Marine 2 | 🟩 Works | Requires launching without anti-cheat; the reporter used [this Nexus Mods setup](https://www.nexusmods.com/warhammer40000spacemarine2/mods/615). No online play with EAC disabled. |
