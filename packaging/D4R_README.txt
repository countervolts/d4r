Modified in this fork for CUDA Ray Reconstruction support and validation (2026).

d4r @VERSION@: NVIDIA DLSS on AMD Radeon RDNA2, RDNA3 and RDNA4 under Linux and Proton
=====================================================================================

d4r runs NVIDIA's own DLSS Super Resolution (DLSS 4 by default), plus experimental Ray Reconstruction,
on AMD Radeon GPUs in DirectX 12 games under Proton. OptiScaler (included) catches the game's DLSS calls
and hands them to d4r. d4r runs NVIDIA's DLSS through ZLUDA; the heaviest Super Resolution kernels are
replaced by kernels written for RDNA2, RDNA3 and RDNA4.

This is an early release. Super Resolution game testing primarily uses Radeon RX 7700 XT.
Ray Reconstruction presets D/E, temporal history, and separate alpha were validated on RX 9070 XT
(gfx1201) with native FP8 WMMA enabled. Other RR GPU targets and RTX image-quality parity are unverified.
SUPPORTED_GAMES.md in the source repository lists per-game setup notes. Expect problems in other games.


What you need
-------------
- An AMD RDNA2, RDNA3 or RDNA4 GPU. This build includes native DLSS 4 and 4.5 network kernels for
  gfx1100–gfx1103 and gfx1200–gfx1201 unless the builder selected fewer targets. Only the RX 7700 XT
  (gfx1101) has been tested on a real GPU. gfx1201 has emulator testing; gfx1200 has compile testing
  only. Neither establishes RDNA4 runtime support, performance or driver stability. Texture-kernel
  optimisations are included only for targets supplied by the builder.
  RDNA2 (RX 6000 series, Steam Deck, RDNA2 integrated GPUs; gfx1030–gfx1036) has no WMMA instructions:
  this build gives it native DLSS 4 (K) layers that do the matrix products in software, bit-identical
  to the WMMA kernels when run on RDNA3. Use K or E there; L runs fully translated (slow), M on d4r's
  engine (its v_dot2 stages; on the Steam Deck about 17.3 ms at 800p, K about 11.6 ms). Of RDNA2, only
  the Steam Deck has been tested.
- Linux with the amdgpu kernel driver (/dev/kfd). ROCm itself is not needed: the zip includes the
  ROCm 7.2.4 runtime in d4r/rocm. RocmDir in d4r/d4r.ini selects another ROCm installation instead.
- GE-Proton 11 (tested: GE-Proton11-3), selected for the game in Steam.
@installer - Internet access, Bash, curl, sha256sum and 7-Zip to run the included install.sh, and python3
@installer   with numpy for the engine models it builds. NVIDIA DLLs are downloaded during installation,
@installer   under NVIDIA's terms.
@clean - Two NVIDIA files, which this zip does not include:
@clean   - nvngx_dlss.dll, the DLSS library, version 310.7 or 310.9 (tested: 310.7.0 and 310.9.1).
@clean     Many games ship one, but often an older version. The d4r kernels check the DLSS code they
@clean     replace, so other versions still run, just slower.
@clean   - _nvngx.dll, NVIDIA's NGX runtime from an NVIDIA Windows display driver (tested: the one from
@clean     driver 596.36, file version 32.0.15.9636). Extract the driver installer, for example with
@clean     7-Zip, and look for _nvngx.dll.
@full - Nothing from NVIDIA: this zip already contains NVIDIA's DLSS library (nvngx_dlss.dll 310.7.0),
@full    NVIDIA's NGX runtime (_nvngx.dll 32.0.15.9636) and output kernels built from NVIDIA's DLSS code.


Install
-------
For an existing game-local OptiScaler installation, the separate -asi.zip add-on
uses its ASI loader and keeps its proxy DLL and unrelated INI settings. Follow
that archive's d4r/README.txt: setup runs automatically when the ASI loads.

1. Extract everything in this zip into the folder that holds the game's main .exe. For Unreal
   Engine games that is <game>/<Project>/Binaries/Win64/, next to <Project>-Win64-Shipping.exe.
   You get dxgi.dll, OptiScaler.ini, d3d12.dll, d3d12core.dll, AMD's amd_fidelityfx_dx12.dll and
   amd_fidelityfx_upscaler_dx12.dll (OptiScaler's FSR option), this file and a d4r folder.
   When updating, replace both d3d12 DLLs with the shim: it requires their resource-lifetime extension.
   If the folder already has a dxgi.dll or OptiScaler.ini (another OptiScaler install), move those
   out of the way first. Also remove PROTON_USE_OPTISCALER from the game's launch options if you
   used GE-Proton's built-in OptiScaler: d4r brings its own.
@installer 2. Run bash install.sh from a terminal (Bash, curl, sha256sum, 7zz/7z/7za and python3 with numpy
@installer    required). It downloads DLSS 310.7.0, DLSS Ray Reconstruction 310.9.1 and the NGX core from
@installer    official NVIDIA sources, checks SHA-256, and installs d4r/nvngx_dlss.dll, d4r/nvngx_dlssd.dll and
@installer    d4r/ngx/_nvngx.dll. No sudo is needed. The driver archive is about 915 MiB; it is extracted,
@installer    never executed. Downloads are temporary. Run it again to check the files; differing existing
@installer    DLLs are backed up before replacement.
@installer    Verified copies are kept in ~/.cache/d4r/nvidia, so installing d4r in the next game needs no
@installer    download. install.sh then asks whether to set up d4r's own DLSS engine (recommended: faster,
@installer    lower latency). That builds the engine's models in d4r/engine from the DLLs: Ray Reconstruction
@installer    D, E and F straight from nvngx_dlssd.dll, and Super Resolution K, M and L (with their LDR
@installer    variants) by rendering one test frame per model through d4r under GE-Proton 11, found in Steam's
@installer    compatibility tools (or set D4R_PROTON=/path/to/GE-Proton11-3). It takes a minute or two once;
@installer    the models are kept in ~/.cache/d4r/engine-models for the next game. bash install.sh --help lists
@installer    options for scripted installs.
@installer    RR D/E also have separate gfx11 models; devices without compatible matrix support retain CUDA
@installer    translation. Builds report progress and time out after 900 seconds per command (override with
@installer    D4R_MODEL_TIMEOUT). Failed captures and logs are retained under ~/.cache/d4r/model-builds;
@installer    D4R_MODEL_LOG_DIR changes that location. Model checksums are verified before cache reuse.
@installer 3. In Steam, open the game's Properties:
@clean 3. In Steam, open the game's Properties:
@full 2. In Steam, open the game's Properties:
   - Compatibility: force GE-Proton11-3.
   - Launch options:
       PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %command%
@installer 4. Optional: check the install from a terminal in that folder:
@clean 4. Optional: check the install from a terminal in that folder:
@full 3. Optional: check the install from a terminal in that folder:
       sh d4r/d4r-check.sh
@installer 5. Start the game and choose DLSS in its graphics settings.
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
@full - Ray Reconstruction: this zip includes NVIDIA's CUDA-capable denoiser (d4r/nvngx_dlssd.dll
@full    310.7.0). [RayReconstruction] Enable defaults to auto: when the game turns Ray
@full    Reconstruction on, d4r runs it through the denoiser. Model = D or E selects a denoiser
@full    preset explicitly; auto follows the game. RR presets D/E are validated on RX 9070 XT
@full    (gfx1201) with [Kernels] NativeFp8 = on, the packaged default on RDNA4. Keep NativeFp8 on:
@full    the scalar FP8 RR path is not validated.
- If L/M shows extra trails in motion, try [Kernels] NativeSwinEncoders = false with
  PreferAccuracy = true. This uses original translated enc1/enc2 layers while retaining other
  native acceleration. It reduced trails in a captured Townfall sequence on RX 7700 XT and costs
  some GPU time. Restart after changing it; other scenes and RDNA4 hardware still need testing.
- [Engine] runs DLSS 4 (K), 4.5 (M and L) and Ray Reconstruction (D, E and F) on d4r's own Vulkan
  inference engine: the work is recorded into the game's own frame instead of going through CUDA
  emulation, which is faster. Enabled = auto (the default) uses it whenever its models are in d4r/engine;
  true / false force it on / off. It needs Mesa RADV with VK_KHR_cooperative_matrix (on RDNA2, which
  lacks it, K, E and F run on the engine without it) and the models in d4r/engine: k (K), k-ldr (K in games without HDR), m and m-ldr (M), l and l-ldr (L),
  rr-d, rr-e and rr-f (Ray Reconstruction). Games and settings it does not cover (other models,
  display-resolution motion vectors with L or M) keep using the normal path,
  and d4r/d4r_nvngx.log says which one ran and why. Tested in a few games so far.
@installer   install.sh builds the models from your NVIDIA DLLs (d4r/tools/build-engine-models.sh). They hold
@installer   values read from NVIDIA's DLSS: they stay on your machine and must not be shared.
@full   Build the models with the tools in d4r's source repository (engine/README.md); until they are
@full   there, d4r uses the normal path.
@clean   Build the models with the tools in d4r's source repository (engine/README.md); until they are
@clean   there, d4r uses the normal path.
- Switching in game, from OptiScaler's menu (Insert):
  - DLSS model: DLSS Settings > tick Render Presets Override, pick the preset (E, K, L or M) and press
    Apply Changes. Unticking it goes back to Model above. ([OptiScaler] FollowPresets = false ignores it.)
  - FSR: set Upscaler (DX12) to FSR 3.X/4. RDNA3 and RDNA4 get FSR 4.1 from GE-Proton's AMD FSR 4
    library; other GPUs get FSR 3.1. d4r lets OptiScaler and AMD's FSR see the real GPU while the game
    still sees an NVIDIA one ([OptiScaler] Fsr4 = false turns that off).
- OptiScaler.ini: OptiScaler's settings, such as its fps overlay and the render resolution of each
  quality mode. Press Insert in game for OptiScaler's menu.
  [Libraries] NvngxPath=d4r\nvngx.dll explicitly selects d4r's shim; OptiDllPath=d4r is also
  needed for the other DLLs. Keep this route when restoring a customized INI during an update.
  Do not point NvngxPath at d4r\ngx\_nvngx.dll: that is NVIDIA's core for the shim's internal use.
  OptiScaler 0.9.4 resolves this relative override from the game's working directory. If a custom
  launcher uses a different working directory, use the shim's absolute Windows filename instead.

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


Ray Reconstruction launch options
---------------------------------
On the tested GE-Proton11-3 setup, Wine's builtin ICU libraries abort inside the denoiser
(u_setMemoryFunctions_65). With Ray Reconstruction (for example Cyberpunk 2077 2.31 with path
tracing and RR preset D or E), use launch options that prefer the game's native ICU DLLs and the
game-local d3d12.dll/d3d12core.dll over Wine's copies:

  PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 WINEDLLOVERRIDES='icuuc=n,b;icuin=n,b;icudt=n,b;d3d12=n,b;d3d12core=n,b' %command%

Merge that WINEDLLOVERRIDES value with any overrides another mod needs instead of replacing it.
Without Ray Reconstruction, the base launch options above are enough. In the game, select DLSS,
then Ray Reconstruction; d4r/d4r.ini ships [RayReconstruction] Enable = auto and
[Kernels] NativeFp8 = on for RDNA4. Keep the patched d3d12.dll/d3d12core.dll current, including
game-local copies.


If something goes wrong
-----------------------
- d4r's log is d4r/d4r_nvngx.log. It is rewritten at every launch. Lines starting with "d4r:" and
  "nvcuda bridge:" name missing files or libraries.
- DLSS missing in the game's menu, or no d4r_nvngx.log: check the launch options and that the
  files are next to the game's main .exe. Run sh d4r/d4r-check.sh: it checks file presence and
  OptiScaler's explicit NGX route separately. For OptiScaler's own log, set LogToFile=true in
  OptiScaler.ini's [Log] section. Its later "NVNGXProxy::InitNVNGX Loaded from" line should name
  d4r\nvngx.dll, not C:\windows\system32\_nvngx.dll. The startup "nvngx.dll not found!" warning
  can be expected with the shim under d4r/; do not move it into the game root to silence that check.
- A wrong image: try Model = E in d4r/d4r.ini, and report the problem with both logs.
- Do not use d4r in games with anti-cheat. OptiScaler's DLL injection can get an account banned.
- Ray Reconstruction needs the CUDA-capable 3.10.7 nvngx_dlssd.dll beside the shim, the matching
  patched d3d12.dll/d3d12core.dll, and the ICU launch overrides from the Ray Reconstruction section
  above. Enable [Kernels] NativeFp8; the scalar FP8 RR path is not validated.
- Optional frame generation: with the game closed, run
    python3 d4r/d4r-features.py . --fg xefg --input dlssg
  from its main executable folder. Use --input fsrfg for native FSR 3.1 FG or
  --input upscaler for games without native FG. --fg fsrfg selects FSR FG output.
  Restart, enable the native FG input in-game, and activate FG in OptiScaler.
  XeFG needs borderless mode and SDR/HDR10 (not scRGB). Verify with FG debug view.
- Optional Reflex-to-AL2/XeLL/LFX: python3 d4r/d4r-features.py . --latency auto
  Then enable Reflex in-game. The provider selects an available method; AL2 on
  Proton also requires host/vkd3d support. This does not change FrameAge.
  The router preserves NGX's real NVAPI topology queries. Both features are
  experimental and off by default. Each setup change creates an exact INI backup.
  Disable: python3 d4r/d4r-features.py . --fg off --latency proton
- Not supported: NVIDIA's DLSS Frame Generation network and DirectX 11.
  OptiScaler can instead convert DLSSG inputs into XeFG or FSR FG as above.
- Development builds expose an experimental native Vulkan NGX path using CPU staging.
  Youngblood is the first intended game test; gameplay remains unverified. See the
  project documentation (docs/vulkan.md) for its separate Vulkan setup and limits.


Uninstall
---------
Delete dxgi.dll, OptiScaler.ini, OptiScaler.log (if present), d3d12.dll, d3d12core.dll,
D4R_README.txt, install.sh (if present) and the d4r folder from the game folder, and clear the launch options. The kernel
cache in ~/.cache/d4r can be deleted too; it also holds install.sh's copies of the NVIDIA DLLs and the engine models.


Licenses
--------
d4r is GNU General Public License, version 3 only (GPL-3.0-only). The release also contains ZLUDA (Apache 2.0 or MIT), vkd3d-proton
(LGPL 2.1, patched) and OptiScaler 0.9.4 (GPL 3.0, unmodified). d4r/source/SOURCES.txt says where
each file comes from, and d4r/licenses has the license texts. d4r is not affiliated with NVIDIA,
AMD or the OptiScaler project, and is provided "as is", without warranty of any kind (GPL-3.0,
sections 15 and 16).
@clean NVIDIA's files are yours to supply, under NVIDIA's terms.
@full NVIDIA's files in this zip (d4r/nvngx_dlss.dll, d4r/ngx/_nvngx.dll) and the kernels built from
@full NVIDIA's code (listed in d4r/source/SOURCES.txt) are NVIDIA's property, are not covered by
@full d4r's license, and are included by whoever distributes this zip, not by NVIDIA.

@installer NVIDIA DLLs are downloaded by install.sh from NVIDIA/DLSS SDK 310.7.0 and 310.9.1 and driver
@installer 596.36. The engine models install.sh builds from them in d4r/engine are NVIDIA's data, for your
@installer own use. d4r/tools/models/bin/glslangValidator is Khronos' glslang 16.6.0 (d4r/licenses).
@installer The included kernels compiled from NVIDIA PTX remain NVIDIA's property and are not
@installer covered by d4r's license. See d4r/source/SOURCES.txt for their provenance.
