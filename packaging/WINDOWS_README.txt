Modified in this fork for CUDA Ray Reconstruction support and validation (2026).

d4r @VERSION@ (native Windows): NVIDIA DLSS on AMD Radeon GPUs in DirectX 12 games on Windows
===========================================================================================

d4r runs NVIDIA's DLSS Super Resolution (models K, M and L) and Ray Reconstruction (presets D, E and F) on
AMD Radeon GPUs in DirectX 12 games on Windows. OptiScaler (included) catches the game's DLSS calls and hands
them to d4r. d4r runs the networks on its own D3D12 engine, recorded into the game's own frame, from models
install.cmd builds on your machine. Anything the engine does not cover runs NVIDIA's DLSS itself through
ZLUDA (CUDA on AMD), where d4r's native kernels replace the heaviest DLSS kernels.

This is a nightly build. It has been tested on a Radeon RX 9070 XT (RDNA4, gfx1201) with AMD's Windows
driver, mainly in Cyberpunk 2077 with Ray Reconstruction (preset F). Other GPUs and games are untested.
Expect problems.


What you need
-------------
- An AMD Radeon GPU. RDNA4 (RX 9000 series) is the tested one. The engine needs the WMMA matrix
  instructions of RDNA3 and RDNA4. RDNA3 (RX 7000 series) is untested on Windows: what the engine cannot
  run there goes through ZLUDA. RDNA2 (RX 6000 series) has no WMMA: there DLSS runs through ZLUDA only,
  with native kernels for DLSS 4 (K); untested on Windows. This zip has native kernels for gfx1030-gfx1036, gfx1100-gfx1103 and
  gfx1200-gfx1201.
- Windows 10 or 11 with AMD's current Adrenalin driver. It includes the HIP runtime ZLUDA runs on; ROCm
  and the HIP SDK are not needed.
- Internet access for install.cmd, which downloads NVIDIA's DLLs under NVIDIA's terms (about 1 GiB once,
  mostly the NVIDIA driver package NGX is extracted from). Nothing else needs installing: the model
  builder uses your Python 3.14 (or a Python with numpy and zstandard) if you have one, and otherwise sets
  up a private Python 3.14 with numpy in %LOCALAPPDATA%\d4r (about 25 MiB); its other tools are in the zip.


Install
-------
1. Extract everything in this zip into the folder that holds the game's main .exe (Cyberpunk 2077:
   bin\x64; Unreal Engine games: <game>\<Project>\Binaries\Win64, next to <Project>-Win64-Shipping.exe).
   You get dxgi.dll, OptiScaler.ini, the OptiScaler folder, install.cmd, install.ps1, this file and a
   d4r folder. If the folder already has a dxgi.dll or OptiScaler.ini (another OptiScaler install), move
   those out of the way first.
2. Double-click install.cmd. It downloads DLSS 310.7.0, DLSS Ray Reconstruction 310.9.1 and the NGX core
   from official NVIDIA sources, checks their SHA-256, and installs d4r\nvngx_dlss.dll,
   d4r\nvngx_dlssd.dll and d4r\ngx\_nvngx.dll. The driver package is extracted, never executed. It then
   builds the engine's models in d4r\engine from those DLLs:
   - Ray Reconstruction D, E and F straight from nvngx_dlssd.dll (about half a minute each);
   - Super Resolution K, M and L (with their LDR variants) by rendering one test frame per model through
     d4r (a few minutes; on RDNA2 the first build can take about half an hour while ZLUDA compiles
     DLSS's kernels).
   The DLLs and models are kept in %LOCALAPPDATA%\d4r, so installing d4r in the next game takes seconds.
   Failed builds keep their logs in %LOCALAPPDATA%\d4r\model-builds. Options: install.cmd -Yes accepts
   every question, -NoEngine only installs the DLLs, -EngineOnly only builds the models, -RebuildEngine
   builds them again.
3. Optional: double-click d4r\d4r-check.cmd to check the install.
4. Start the game and choose DLSS (and DLSS Ray Reconstruction) in its graphics settings.

The first time DLSS runs through ZLUDA, the game can freeze for a minute or more while DLSS's kernels are
compiled. They are cached in %LOCALAPPDATA%\zluda, so later starts are quick.


Settings
--------
d4r\d4r.ini holds d4r's settings for this game (each game has its own copy):
- [DLSS] Model: K (DLSS 4, default), M (DLSS 4.5) or L (DLSS 4.5 for Ultra Performance, especially 4K).
  Choosing the model does not change the render resolution.
- [RayReconstruction] Model: auto (default) follows the game; D, E or F picks a preset.
- [Engine] Enabled: auto (default) runs DLSS on d4r's D3D12 engine whenever its models are in d4r\engine;
  false always uses the ZLUDA path. Games and settings the engine does not cover keep using the ZLUDA path,
  and d4r\d4r_nvngx.log says which one ran and why.
- [Kernels] (the ZLUDA path only): PreferAccuracy = true selects separately built accuracy kernels, which
  favour fidelity over speed; NativeFp8 (RDNA4, on by default) selects native FP8 WMMA and the gfx12xx-fp8
  kernel folder; NativeKernels = off runs every DLSS kernel as ZLUDA translates it.
- Switching in game, from OptiScaler's menu (Insert): DLSS Settings > tick Render Presets Override, pick
  the preset and press Apply Changes ([OptiScaler] FollowPresets = false ignores it). OptiScaler.ini holds
  OptiScaler's own settings; [Libraries] NvngxPath=d4r\nvngx.dll selects d4r's shim and must stay.


Performance
-----------
GPU time of one frame on d4r's D3D12 engine, Radeon RX 9070 XT, AMD's Windows driver, measured with d4r's
D3D12 runners on the same test frames as the Linux figures (median of five trials):

  Output                 K        M        L      RR D     RR E     RR F
  2560x1440           1.49 ms  1.00 ms  2.34 ms  4.13 ms  4.19 ms  4.09 ms
  3840x2160           2.89 ms  2.10 ms  5.08 ms  8.81 ms  8.98 ms  8.99 ms

Every DLSS frame is shown in the frame it belongs to, so there is no added latency.


If something goes wrong
-----------------------
- d4r's log is d4r\d4r_nvngx.log, rewritten at every launch. "native D3D12 initialized" lists the models
  the engine loaded, and "engine frame 1 recorded on D3D12" means a feature runs on it. Lines starting
  with "d4r:" name missing files or settings.
- DLSS missing in the game's menu, or no d4r_nvngx.log: check that the files are next to the game's main
  .exe and run d4r\d4r-check.cmd. For OptiScaler's own log, set LogToFile=true in OptiScaler.ini's [Log]
  section: its "NVNGXProxy::InitNVNGX Loaded from" line should name d4r\nvngx.dll.
- install.cmd failed: it names the step and the log. Run it again after fixing the cause; finished
  downloads and models are reused.
- A wrong image: try another [DLSS] Model, or [Engine] Enabled = false, and report the problem with the log.
- Do not use d4r in games with anti-cheat. OptiScaler's DLL injection can get an account banned.
- XeFG, FSR FG and Nukem's DLL are included in OptiScaler\, with upstream licenses. FG is off by
  default; select it in OptiScaler's menu (Insert). FG with d4r is untested on Windows.
- Not supported: NVIDIA's DLSS Frame Generation network and DirectX 11 or Vulkan games.


Uninstall
---------
Delete dxgi.dll, OptiScaler.ini, OptiScaler.log (if present), install.cmd, install.ps1, D4R_README.txt,
the OptiScaler folder and the d4r folder from the game folder. %LOCALAPPDATA%\d4r (NVIDIA's DLLs, the
engine models and the model builder's Python) and %LOCALAPPDATA%\zluda (compiled kernels) can be deleted
too.


Licenses
--------
d4r is GNU General Public License, version 3 only (GPL-3.0-only). The release also contains ZLUDA (Apache
2.0 or MIT, patched), OptiScaler 10.0.0-pre1 (GPL 3.0, unmodified; pinned nightly-20261024 prerelease),
Khronos' glslangValidator, Microsoft's DirectX Shader Compiler and Visual C++ runtime (for the model
builder). d4r\source\SOURCES.txt says where each file comes from, and d4r\licenses has the license texts.
d4r is not affiliated with NVIDIA, AMD or the OptiScaler project, and is provided "as is", without
warranty of any kind (GPL-3.0, sections 15 and 16).

NVIDIA DLLs are downloaded by install.cmd from NVIDIA/DLSS SDK 310.7.0 and 310.9.1 and driver 596.36. The
engine models install.cmd builds from them in d4r\engine are NVIDIA's data, for your own use: they must
not be shared. The included kernels compiled from NVIDIA PTX remain NVIDIA's property and are not covered
by d4r's license. See d4r\source\SOURCES.txt for their provenance.
