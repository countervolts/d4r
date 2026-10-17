d4r for Windows: NVIDIA DLSS on AMD Radeon RDNA3 and RDNA4
===========================================================

d4r runs NVIDIA's own DLSS (DLSS 4 by default) on a Radeon GPU in DirectX 12 games. OptiScaler
(included) catches the game's upscaler calls and hands them to d4r.

This is an early release. RX 7700 XT has passed native Windows synthetic checks and limited
game-frame tests. RX 9070 XT also has recorded hardware tests; other targets are compile-tested.
These checks do not establish broad game support or Linux performance parity.
SUPPORTED_GAMES.md in the source repository lists the tested games. Expect problems.

What you need
-------------
- Windows 11, a Radeon RX 7000 or RX 9000 series GPU, a current AMD Adrenalin driver.
- A DirectX 12 game without anti-cheat (OptiScaler injects a DLL; it can get an account banned).
- NVIDIA DLSS files. The ZIP does not include them. Run install.bat to download official, pinned files.
  It downloads DLSS SR 310.9.1, Ray Reconstruction 310.9.1, and extracts _nvngx.dll from the
  official NVIDIA 617.14 driver without installing or changing the driver. SR is stored under d4r\dlss;
  RR is stored separately under d4r\rr so SR startup cannot auto-load an RR feature. It needs internet access
  and about 1 GiB temporary space. The driver archive and DLSS files are SHA256-verified.
  NVIDIA license terms apply. Files stay local and outside public d4r packages.

Install
-------
1. Extract this ZIP into the folder that holds the game's main .exe. For Unreal Engine games that
   is <game>\<Project>\Binaries\Win64, next to <Project>-Win64-Shipping.exe.
   If the folder already has a dxgi.dll or OptiScaler.ini (another OptiScaler install), or a
   d3d12.dll from a Linux install, move those out first.
2. Run install.bat and accept the download prompt. Existing d4r DLLs are backed up outside the game
   folder with their original paths and SHA256 hashes. Verified downloads are cached in
   %LOCALAPPDATA%\d4r\cache.
   Bundled isolated Python runs the installer; system Python, PowerShell and Git Bash are not required.
   Choose N to keep RR off, or D/E to build and enable an experimental private native RR model.
   Automatic RR building currently requires the gfx1101-only package. Model tools run from the
   external cache. Existing models remain intact; d4r.ini receives an exact external backup before
   selecting the new relative model path. Game compatibility and reference quality remain unverified.
   A model-build failure leaves the verified NVIDIA DLLs installed and does not enable the model.
3. Optional: check the install. In the game folder, open PowerShell and run
       powershell -ExecutionPolicy Bypass -File d4r\d4r-check.ps1
4. Start the game from Steam, Epic or a shortcut as usual. In its graphics settings choose DLSS
   (or FSR 3.1/XeSS if DLSS is hidden; OptiScaler then feeds DLSS). Keep frame generation off.
   Press Insert in game for OptiScaler's menu.

The first time DLSS starts, the game can freeze for several minutes while DLSS's GPU kernels are
compiled. They are cached in %LOCALAPPDATA%\d4r\cache, so later starts are quick.

Settings
--------
Edit d4r\d4r.ini (it is commented) and restart the game. Each game has its own copy.
- Model: E (preset 5), K (11, DLSS 4), L (12) or M (13, DLSS 4.5). E and L set NGX hints only; the standalone engine and L texture kernels are not integrated. Select quality mode in the game.
- Kernels/NativeFp8: experimental RDNA4 setting. Requires a matching gfx12 native-FP8 M bundle; keep false with the standard FP16 release. Token-lane and hardware-pack builds remain unvalidated on GPU.
- AsyncInterop, ValidateOutput, Log: for testing and bug reports; the defaults are the tested setting.
- RayReconstruction/Backend: off by default. native selects experimental native D3D12/Vulkan RR;
  ModelDir selects your locally generated D/E model (auto: d4r\engine\rr-d). Relative paths resolve
  beside d4r.ini. No model is bundled; install.bat can optionally generate and select D/E locally.
  fsr (experimental) answers the game's DLSS RR request with AMD's FSR Ray Regeneration denoiser,
  then DLSS Super Sampling upscales the result. AMD does not support FSR RR on RDNA3, so d4r runs
  AMD's unmodified denoiser DLL on a private vkd3d-proton device (d4r\fsr, built by d4r, LGPL; see
  d4r\fsr\SOURCE.txt) that emulates the FP8 math on RDNA3. AMD's amd_fidelityfx_denoiser_dx12.dll is
  not bundled: install.bat [F] downloads the pinned file (SHA256 48f1e588...), or set FsrDenoiser.
  FsrStabilityBias (default auto = AMD's 1; 0 is much sharper without upscaling, see d4r.ini) and
  FsrDetail (0-1) tune the denoiser.
  The first RR start compiles AMD's shaders and can freeze the game for about a minute; they are
  cached in %LOCALAPPDATA%\d4r\cache\vkd3d. Synthetic validation only: guides must be HDR with
  render-resolution, unjittered motion vectors; diffuse hit distance is not used. The exposure texture
  goes only to Super Sampling. Dynamic render sizes must stay within creation dimensions and the
  Super Sampling library's queried range; unsupported sizes fail before submission.
  Game compatibility and image quality against DLSS RR are unverified.
  FSR-only output accepts RGBA16F or RGBA32F, including format changes between frames. RGBA32F
  widens the completed half result on the GPU; it does not change denoiser arithmetic or produce alpha.
  Native RR accepts per-frame render dimensions from 64 pixels up to the feature's creation size.
  Changing dimensions resets temporal history and retains the loaded model and shared GPU packet.
  Output dimensions remain fixed. This behavior has synthetic D/E coverage only.
  RR accepts RGBA16F and RGBA32F output textures, including temporal format changes;
  float32 output is widened on the GPU. Other output encodings remain unsupported.
OptiScaler's own settings (overlay, render ratios) are in OptiScaler.ini.

If something goes wrong
-----------------------
- Run d4r\d4r-check.ps1: it prints ok / MISSING / note lines with the fix for each problem.
- d4r\d4r_nvngx.log is rewritten at every launch. A line "d4r: DLSS disabled: ..." says why.
- DLSS missing in the menu and no d4r_nvngx.log: the files are not next to the game's main .exe.
- For a bug report run  d4r\d4r-check.ps1 -Report  and send the ZIP it puts on your Desktop,
  with your GPU, the game, the model and what the image looked like. Logs can contain folder names.
- Not supported: frame generation, standalone inference engine selection in the game, DirectX 11,
  Vulkan game integration, or supported FSR RR (the fsr backend is experimental). Native DLSS RR and
  FSR RR have synthetic feature-13 validation only;
  it does not establish game compatibility or NVIDIA reference quality. E/L hints do not establish full parity.

Uninstall
---------
Delete dxgi.dll, OptiScaler.ini, OptiScaler.log (if present), WINDOWS_README.txt, install.bat,
install-dependencies.py, install-dlss.ps1 and the d4r folder
from the game folder. The cache in %LOCALAPPDATA%\d4r can be deleted too. Nothing else was changed.

Licenses
--------
d4r is GNU General Public License, version 3 only (GPL-3.0-only). This ZIP also contains ZLUDA (Apache 2.0 or MIT), OptiScaler (GPL 3.0,
patched; the patches are in the source repository), and AMD's HIP runtime. d4r\licenses has the
texts. d4r is not affiliated with NVIDIA, AMD or the OptiScaler project. NVIDIA's files are yours
to supply, under NVIDIA's terms.
Portable model tools include Python, NumPy and glslang; their license texts are under
d4r\model-tools\licenses, with additional dependency notices in the tool directories.
