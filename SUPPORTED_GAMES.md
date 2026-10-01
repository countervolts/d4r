# Supported games

The following DirectX 12 games run DLSS Super Resolution through d4r on the tested Radeon RX 7700 XT Linux setup with GE-Proton 11-3. All three documented DLSS models run in each game; image-quality issues are noted in the table.

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
- d4r supports DLSS Super Resolution only. Keep DLSS Ray Reconstruction off in games that offer it (such as Cyberpunk 2077 and Alan Wake 2); Control requests it whenever ray tracing is on and crashes.
- Saving settings from the OptiScaler overlay ("Save INI") rewrites `OptiScaler.ini` and resets d4r's values to `auto`, including frame generation (`[FrameGen] Enabled=false`, `FGInput=nofg`, `FGOutput=nofg`) and `[Inputs] EnableDlssInputs=true`. Restore them afterwards, or keep a copy of the original file.

Game-specific setup:

- **The Last of Us Part I:** the game's Streamline integration crashes on boot with OptiScaler. Rename `sl.common.dll` in the game folder to `sl.common.dll.bak` and select FSR 3.1 in the game's graphics settings; OptiScaler passes it to DLSS through d4r.

- **Control Ultimate Edition:** turn ray tracing off. With ray tracing on, the game requests DLSS Ray Reconstruction, which d4r does not support, and crashes when loading into gameplay.

- **Cyberpunk 2077:** GE-Proton 11-3 loads Wine's own ICU libraries instead of the game's, and the game aborts at launch (`unimplemented function icuuc.dll.u_setMemoryFunctions_65`). Add `WINEDLLOVERRIDES="icuuc,icuin=n,b"` to the launch options (fixed in GE-Proton 11-4).

- **Alan Wake 2:** with the game's own DLSS inputs, every model shows heavy ghosting. Set `AutoExposure=true` under `[InitFlags]` in `OptiScaler.ini` (or enable auto exposure in the OptiScaler overlay), or select FSR 3.1 in the game instead so OptiScaler passes FSR inputs to DLSS.

- **Horizon Zero Dawn Remastered:** the game loads `dxgi.dll` only from the Windows system folder, so OptiScaler is never loaded under that name. Rename `dxgi.dll` to `version.dll` and add `WINEDLLOVERRIDES="version=n,b"` to the launch options. d4r's `d3d12core.dll` is still picked up from the game folder.

- **Ratchet & Clank: Rift Apart:** set `Dxgi=false` in `OptiScaler.ini`. With ray tracing on, OptiScaler's DXGI spoofing crashes the game at startup on AMD GPUs; the `PROTON_FORCE_NVAPI=1` launch option already reports an NVIDIA GPU.

- **Dying Light: The Beast:** use DLSS inputs; with FSR or XeSS inputs the screen stays black. The game passes its motion vectors as `R16G16B16A16_TYPELESS`, which needs a d4r build from 2026-09-28 or later (older builds reject every frame and show a black screen).

Choose a model with `Model = E`, `Model = K`, or `Model = M` under `[DLSS]` in the game's `d4r/d4r.ini`. Model compatibility does not imply equal performance; see the [Townfall measurements](docs/performance.md) for measured frame rates.
