d4r @VERSION@: NVIDIA DLSS on AMD Radeon RDNA3 and RDNA4 under Linux and Proton
=====================================================================

d4r runs NVIDIA's own DLSS Super Resolution (DLSS 4 by default) on AMD Radeon RDNA3 and RDNA4 GPUs in
DirectX 12 games under Proton. OptiScaler (included) catches the game's DLSS calls and hands them to
d4r. d4r then runs NVIDIA's DLSS on the Radeon through ZLUDA, with the heaviest parts replaced by
kernels written for RDNA3 and RDNA4.

This is an early release, tested on one GPU (Radeon RX 7700 XT) in about a dozen DirectX 12 games.
SUPPORTED_GAMES.md in the source repository lists them, with per-game setup notes. Expect problems
in other games.


What you need
-------------
- An AMD RDNA3 or RDNA4 GPU. This build includes native DLSS 4 and 4.5 network kernels for
  gfx1100–gfx1103 and gfx1200–gfx1201 unless the builder selected fewer targets. Only the RX 7700 XT
  (gfx1101) has been tested on a real GPU. gfx1201 has emulator testing; gfx1200 has compile testing
  only. Neither establishes RDNA4 runtime support, performance or driver stability. Texture-kernel
  optimisations are included only for targets supplied by the builder.
- Linux with the amdgpu kernel driver (/dev/kfd). ROCm itself is not needed: the zip includes the
  ROCm 7.2.4 runtime in d4r/rocm. RocmDir in d4r/d4r.ini selects another ROCm installation instead.
- GE-Proton 11 (tested: GE-Proton11-3), selected for the game in Steam.
@clean - Two NVIDIA files, which this zip does not include:
@clean   - nvngx_dlss.dll, the DLSS library, version 310.7 or 310.9 (tested: 310.7.0 and 310.9.1).
@clean     Many games ship one, but often an older version. The d4r kernels check the DLSS code they
@clean     replace, so other versions still run, just slower.
@clean   - _nvngx.dll, NVIDIA's NGX runtime from an NVIDIA Windows display driver (tested: the one from
@clean     driver 596.36, file version 32.0.15.9636). Extract the driver installer, for example with
@clean     7-Zip, and look for _nvngx.dll.
@full - Nothing from NVIDIA: this zip already contains NVIDIA's DLSS library (nvngx_dlss.dll 310.7.0),
@full   NVIDIA's NGX runtime (_nvngx.dll 32.0.15.9636) and output kernels built from NVIDIA's DLSS code.


Install
-------
1. Extract everything in this zip into the folder that holds the game's main .exe. For Unreal
   Engine games that is <game>/<Project>/Binaries/Win64/, next to <Project>-Win64-Shipping.exe.
   You get dxgi.dll, OptiScaler.ini, d3d12.dll, d3d12core.dll, this file and an d4r folder.
   When updating, replace both d3d12 DLLs with the shim: it requires their resource-lifetime extension.
   If the folder already has a dxgi.dll or OptiScaler.ini (another OptiScaler install), move those
   out of the way first. Also remove PROTON_USE_OPTISCALER from the game's launch options if you
   used GE-Proton's built-in OptiScaler: d4r brings its own.
@clean 2. Copy nvngx_dlss.dll into the d4r folder, and _nvngx.dll into d4r/ngx.
@clean 3. In Steam, open the game's Properties:
@full 2. In Steam, open the game's Properties:
   - Compatibility: force GE-Proton11-3.
   - Launch options:
       PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %command%
@clean 4. Optional: check the install from a terminal in that folder:
@full 3. Optional: check the install from a terminal in that folder:
       sh d4r/d4r-check.sh
@clean 5. Start the game and choose DLSS in its graphics settings.
@full 4. Start the game and choose DLSS in its graphics settings.

The first time DLSS starts, the game can freeze for a minute or more while DLSS's GPU kernels are
compiled. They are cached in ~/.cache/d4r, so later starts are quick.


Settings
--------
- In d4r/d4r.ini, [Kernels] PreferAccuracy = true prioritizes fidelity over performance for every
  native kernel and the translated fallback. It selects separately built accuracy kernels, restores
  intermediate FP8 quantization and per-MMA f16 rounding, preserves denormal handling, and keeps
  NGX's synchronization calls. It defaults to false. Restart the game after changing it; the first
  start may compile a separate cache. Missing accuracy variants use translated kernels, never the
  fast native set. The mode aims to match NVIDIA's output, but 1:1 RTX image quality is not proven.
- d4r/d4r.ini: d4r's settings for this game. The main one is the DLSS model:
  K (DLSS 4, default), E (DLSS 3 CNN, a little faster), M (DLSS 4.5), or L (DLSS 4.5
  for Ultra Performance, especially 4K; experimental). Set Model = L and select Ultra Performance
  in game or OptiScaler. Choosing the model does not change the render resolution.
- On RDNA4, [Kernels] NativeFp8 is on by default. It selects native FP8 WMMA and the matching
  gfx12-fp8 kernel folder; set it to false to use f16 widening. The FP8 path has not been tested
  on a real RDNA4 GPU.
- If L/M shows extra trails in motion, try [Kernels] NativeSwinEncoders = false with
  PreferAccuracy = true. This uses original translated enc1/enc2 layers while retaining other
  native acceleration. It reduced trails in a captured Townfall sequence on RX 7700 XT and costs
  some GPU time. Restart after changing it; other scenes and RDNA4 hardware still need testing.
- OptiScaler.ini: OptiScaler's settings, such as its fps overlay and the render resolution of each
  quality mode. Press Insert in game for OptiScaler's menu.

Each game has its own copy of both files.


Performance
-----------
Radeon RX 7700 XT, SILENT HILL Townfall at 2560x1440, a 62-second walk through the town, average fps:

  Mode                 DLSS 3 CNN (E)   DLSS 4 (K)   DLSS 4.5 (M)   FSR 4
  Quality                   72              69           51          76
  Balanced                  80              76           59          84
  Performance               88              82           69          94
  Ultra Performance         89              93           90         107
  Native 2560x1440 (TSR, no upscaling): 49

DLSS 4 in every mode, and E and M at Quality, were measured with this release. The other E and M
figures come from the development setup, which runs the same kernels. Every DLSS frame is shown in
the frame it belongs to, so there is no added latency.
@clean This zip lacks the kernels built from NVIDIA's code, so its DLSS 4 is about 3 fps slower.


Why the launch options
----------------------
- PROTON_FORCE_NVAPI=1 lets Proton's NVAPI run on an AMD GPU and present the GPU as NVIDIA's.
  OptiScaler only enables DLSS on NVIDIA hardware.
- DXVK_NVAPI_GPU_ARCH=AD100 reports an RTX 40 series (Ada) GPU. DLSS then loads the network
  weights that match the kernels d4r runs.


If something goes wrong
-----------------------
- d4r's log is d4r/d4r_nvngx.log. It is rewritten at every launch. Lines starting with "d4r:" and
  "nvcuda bridge:" name missing files or libraries.
- DLSS missing in the game's menu, or no d4r_nvngx.log: check the launch options and that the
  files are next to the game's main .exe. For OptiScaler's own log, set LogToFile=true in
  OptiScaler.ini's [Log] section.
- A wrong image: try Model = E in d4r/d4r.ini, and report the problem with both logs.
- Do not use d4r in games with anti-cheat. OptiScaler's DLL injection can get an account banned.
- Not supported: DLSS Frame Generation, DLSS Ray Reconstruction, DirectX 11 and Vulkan games.


Uninstall
---------
Delete dxgi.dll, OptiScaler.ini, OptiScaler.log (if present), d3d12.dll, d3d12core.dll,
D4R_README.txt and the d4r folder from the game folder, and clear the launch options. The kernel
cache in ~/.cache/d4r can be deleted too.


Licenses
--------
d4r is GNU General Public License, version 3 only (GPL-3.0-only). The release also contains ZLUDA (Apache 2.0 or MIT), vkd3d-proton
(LGPL 2.1, patched) and OptiScaler 0.9.4 (GPL 3.0, unmodified). d4r/source/SOURCES.txt says where
each file comes from, and d4r/licenses has the license texts. d4r is not affiliated with NVIDIA,
AMD or the OptiScaler project.
@clean NVIDIA's files are yours to supply, under NVIDIA's terms.
@full NVIDIA's files in this zip (d4r/nvngx_dlss.dll, d4r/ngx/_nvngx.dll) and the kernels built from
@full NVIDIA's code (listed in d4r/source/SOURCES.txt) are NVIDIA's property, are not covered by
@full d4r's license, and are included by whoever distributes this zip, not by NVIDIA.
