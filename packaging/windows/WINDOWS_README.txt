d4r for Windows: NVIDIA DLSS on AMD Radeon RDNA3 and RDNA4
===========================================================

d4r runs NVIDIA's own DLSS (DLSS 4 by default) on a Radeon GPU in DirectX 12 games. OptiScaler
(included) catches the game's upscaler calls and hands them to d4r.

This is an early release. Only the Radeon RX 9070 XT has been tested on real hardware; every other
GPU in this release is compile-tested. SUPPORTED_GAMES.md in the source repository lists the
tested games. Expect problems.

What you need
-------------
- Windows 11, a Radeon RX 7000 or RX 9000 series GPU, a current AMD Adrenalin driver.
- A DirectX 12 game without anti-cheat (OptiScaler injects a DLL; it can get an account banned).
- Two files from NVIDIA, which this ZIP does not include:
    nvngx_dlss.dll  version 310.9.1 (tested) or 310.7.0. Many games ship one in their folder;
                    right-click > Properties > Details shows the version.
    _nvngx.dll      version 32.0.16.1714 (tested) or 32.0.15.9636. It is part of an NVIDIA display
                    driver package: extract the installer with 7-Zip and search for _nvngx.dll.

Install
-------
1. Extract this ZIP into the folder that holds the game's main .exe. For Unreal Engine games that
   is <game>\<Project>\Binaries\Win64, next to <Project>-Win64-Shipping.exe.
   If the folder already has a dxgi.dll or OptiScaler.ini (another OptiScaler install), or a
   d3d12.dll from a Linux install, move those out first.
2. Copy nvngx_dlss.dll into the d4r folder and _nvngx.dll into d4r\ngx.
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
- Model: K (DLSS 4, default) or M (DLSS 4.5, heavier). Select the quality mode in the game.
- AsyncInterop, ValidateOutput, Log: for testing and bug reports; the defaults are the tested setting.
OptiScaler's own settings (overlay, render ratios) are in OptiScaler.ini.

If something goes wrong
-----------------------
- Run d4r\d4r-check.ps1: it prints ok / MISSING / note lines with the fix for each problem.
- d4r\d4r_nvngx.log is rewritten at every launch. A line "d4r: DLSS disabled: ..." says why.
- DLSS missing in the menu and no d4r_nvngx.log: the files are not next to the game's main .exe.
- For a bug report run  d4r\d4r-check.ps1 -Report  and send the ZIP it puts on your Desktop,
  with your GPU, the game, the model and what the image looked like. Logs can contain folder names.
- Not supported: frame generation, ray reconstruction, DirectX 11, Vulkan, models E and L.

Uninstall
---------
Delete dxgi.dll, OptiScaler.ini, OptiScaler.log (if present), WINDOWS_README.txt and the d4r folder
from the game folder. The cache in %LOCALAPPDATA%\d4r can be deleted too. Nothing else was changed.

Licenses
--------
d4r is GNU General Public License, version 3 only (GPL-3.0-only). This ZIP also contains ZLUDA (Apache 2.0 or MIT), OptiScaler (GPL 3.0,
patched; the patches are in the source repository), and AMD's HIP runtime. d4r\licenses has the
texts. d4r is not affiliated with NVIDIA, AMD or the OptiScaler project. NVIDIA's files are yours
to supply, under NVIDIA's terms.
