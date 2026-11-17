# Reconciling linux and windows into main

`main` starts from `windows` (fcafd9a). Linux commits from `f66ccca..linux` are ported onto it stage by stage.
Each port commit names the linux sha it ports. `windows` and `linux` are left untouched.

## What windows already had

`windows` commit 8437e24 ("Bring the main d4r codebase into the Windows branch") was a hand merge of the
linux tree, not a git merge. A hunk-level check (for each linux commit: are its added lines on main?) shows
it holds linux up to about 2486098 (2026-10-27). Older linux commits still show a few gaps. Most of them are
lines that later commits rewrote, and Stage 6 checks them. The three linux commits after 2486098 and before
the FG port (05e19d4, ad986e7, 40ba8f9) were not in it.

`fg/dlssg` on main is linux's tree as of 30ed3ec (Linux FG port), plus windows' own native D3D12 host
(45a6d1e: `host/cubin_d3d12.*`, `dlssg_host.cpp`, `host/build.sh`, `nvxrt/scan.cpp`,
`translator/build_cache.py`, `translator/ptxc.py`). Linux's later FG commits (4d88a2e, 970a013) were not on it.

Apart from the vkd3d-proton patches below, the only linux file missing from main was
`packaging/build/Dockerfile.glibc241`. Windows' branch reset (73994d4) deleted it, and docs/building.md and
build_release_glibc241.sh still use it. It is restored.

## vkd3d-proton patch numbering

Windows uses `patches/vkd3d-proton/0006`–`0013` for its dxil-spirv patches. Linux's later patches are renumbered
after them (the build script applies `*.patch` in name order):

| main | linux | from |
|---|---|---|
| 0014-cubin-launch-state | 0006 | 30ed3ec |
| 0015-command-list-d4r-unwrap | 0007 | #60 (243805a) |
| 0016-cuda-texture-handle-fallback | 0008 | #62 (c56d217) |
| 0017-descriptor-heap-optin | 0009 | #63 (ad10640) |

The base is now vkd3d-proton 3.2 (`f05a7710`, linux #63). Windows' `0011-dxil-spirv-exact-byte-address-divide`
did not apply on either base: dxil-spirv already has an `_MSC_VER` `<intrin.h>` block where the patch's include
hunk expects `namespace dxil_spv`. Its context is rebased. All 17 patches apply on f05a7710, and the patched
`d3d12.dll`/`d3d12core.dll` build. Windows' native FSR RR payload pins its own vkd3d-proton (`3dfc6f07`,
`patches/vkd3d-proton-windows/`) and is not affected.

## Fixes to main itself (not ports)

- `scripts/build_d4r_nvngx_shim.sh`: on a Linux host the d3dhip SDK probe picked `/opt/rocm`, and the MinGW
  link of `d4r_hip_net.dll` failed (`-lamdhip64` not found), so the shim did not build on Linux at all.
  A HIP SDK candidate now also needs a Windows import library (`lib/amdhip64.lib` or `libamdhip64.dll.a`).
  Windows HIP SDKs are unaffected. A Linux-hosted build has no d3dhip network, and it only matters on native
  Windows anyway.
- `fg/dlssg/host/build.sh`: `-I/usr/include` on the MinGW compile pulled in glibc's headers on Linux, so the
  DLSS-G host did not build there. Only `vulkan/` and `vk_video/` now go on the include path, as the shim build
  does.

## Test baseline (windows, fcafd9a)

`pytest tests` in a scratch venv (pytest 9.2, numpy 2.5), from a clean checkout of the tree:

- windows as is: 1 failed, 203 passed, 25 skipped.
- with the shim and ASI built (so `test_asi_setup`'s four ASI tests run): 5 failed.
- Already failing on windows:
  - `test_frame_completion::test_shim_does_not_signal_past_running_producer`: the test compiles a function cut
    out of the shim, and windows' version uses `Feature::splitFence`, which the test's stub `Feature` lacks.
  - `test_asi_setup`: four tests, `LoadLibrary failed: 126`. Windows' shim/ASI imports `d3d12.dll` and
    `dxgi.dll` at load time (native D3D12 engine), which the test's bare Wine prefix does not satisfy.

## Stage 1: small linux commits

Commits on main, oldest first. The history from 771eea3 on was rewritten to drop windows' Claude co-author trailers, so main and windows share history only up to fe53aa0:

| main | ports | what |
|---|---|---|
| 455fde3 | (fix) | shim build on Linux hosts (HIP SDK probe) |
| 8b6bfbf | 05e19d4 | `[Interop] NativeMotionVectors` opt-in for Northlight motion vectors |
| e7a7099 | 30ed3ec | Linux DLSS-G packaging, `d4r_boot.dll` NVIDIA identity (no launch options), EAC via umu, cubin vkd3d patch |
| 7417dd0 | #59 (3d7eb97) | NGX parameter object capacity 256 → 4096 (RE Requiem) |
| ef0f1c4 | (fix) | DLSS-G host build on Linux hosts (include path) |
| a54bf93 | 4d88a2e | DLSS-G from install.sh, FG Lite / `Model`, faster RDNA3 FG (t1 decoder groups, wave64), parallel kernel builds |
| ac97807 | #61 (f9595f9) | Vulkan engine accepts R32_UINT depth |
| d56fc3b | #60 (fc05bbe) | unwrap proxied command lists (ReShade); ReShade Wine patch |
| c1e3303 | 05dc7a0 | release the reference GetCommandList adds |
| 7ae0ac0 | 970a013 | comgr off Wine's signal handlers, locale-free nvxrt parsing, exposure views outside the cache budget |
| be8399c | #62 (432639f) | vkd3d heap-descriptor patches |
| b593464 | #63 (7d55fa8) | vkd3d-proton 3.2, descriptor heap opt-in, 0011 rebase |
| cb94d4a | (restore) | Dockerfile.glibc241 |
| dcb897b | (README) | linux's OptiScaler/FSR/vkhip/XeFG notes back into the README |

### Ported, with merge decisions

- **stage_optiscaler.py**: windows' `stage()` already takes extra positional arguments as settings files plus
  `--patched`. Linux passed the boot DLL as a third positional argument. It is now `--boot BOOT_DLL`, and
  `package_release.sh` passes it that way. `package_release_windows.sh` does not stage `d4r_boot.dll`
  (native Windows has no Wine identity to fake).
- **dlssg_host.cpp**: windows' host reads `d4r.ini [FrameGeneration]` itself (`Model = auto` means Lite). On
  Linux, `d4r_boot.dll` resolves the model (auto: Lite on RDNA3, full on RDNA4) and passes `D4R_DLSSG_ENABLED`
  and `D4R_DLSSG_LITE`. Those two now win when set. `d4r_boot.dll` exists only under Proton, so native Windows
  behaviour is unchanged.
- **cubin_d3d12.cpp** (windows' native FG executor) was not changed. Linux's new t1 decoder launch groups are
  data in the kernel cache, and it already runs groups with the same bounded grid barrier and single-launch
  fallback. It ignores the new `wave 64` meta, so those kernels stay wave32 on native Windows. That is perf
  only, a follow-up if wanted.
- **kernels/build.sh**: linux's parallel jobs (`D4R_KERNEL_JOBS`) and `D4R_KERNEL_CACHE`, merged with windows'
  `windows-flags.txt`, device-lib path and `D4R_VKHIP_LAYERS` lines. The one colliding hunk was merged by hand.
- **wine_nvcuda_bridge.c**: linux's comgr `sigaction` redirect is under `#ifndef D4R_NATIVE_WINDOWS` (it uses
  `<link.h>`/`<elf.h>`). Both the Linux and the native Windows bridge build.
- **README.md**: the Northlight, `d4r_boot.dll`/update and DLSS-G paragraphs are under "Linux (Proton)". Linux's
  OptiScaler options (`FollowPresets`, `Fsr4`, Save INI), FSR 4.1 on RDNA3/RDNA4, vkhip-by-default and the
  XeFG/XeLL notes are back in "Switching models and FSR in game".
- **INI files**: after Stage 1, `config/d4r.ini.default` and `packaging/d4r.ini` contain every key linux has.
  Main additionally has `[Engine] ExperimentalRrVkHip`.

### Skipped

- ad986e7, 40ba8f9 (readme): they edit linux's old Results section, which windows' README replaced.
- f779b8a (nightly link): main's README already links the nightly builds.
- f7c4e23, c310486, 619682c, 2486098, #53, #54, #57: already on main through 8437e24 (hunk check: 101/114,
  64/73, 116/145, 46/49, 4/4, 13/13, 65/66 hunks present; the rest are lines later commits rewrote). Stage 6
  rechecks the small remainders.
- The pre-#63 intermediate state of #62 (`0009-embedded-mutable-descs-optin`) was committed as linux had it and
  then removed by #63, as on linux.
- Not changed, left to decide: linux's Results tables (SILENT HILL Townfall, Cyberpunk 2077) and its GPU table's
  "Runtime testing" column and RDNA2 paragraph. Windows' README replaced them with its own Results and GPU tables.

### Built and tested here

- Builds: `d4r_nvngx.dll` + `d4r.asi`, `nvcuda.dll` (Wine bridge), the native Windows bridge, `d4r_boot.dll`,
  the DLSS-G host (`nvngx.dll_dlssg_host.dll`), `libVkLayer_nvxrt.so`, K kernels for gfx1101 through the merged
  `kernels/build.sh` (4 jobs, systemd-run MemoryMax=24G), patched vkd3d-proton 3.2.
- `pytest tests`: 5 failed (the same 5 as the windows baseline above), 219 passed, 21 skipped.

### Needs testing on hardware

- Linux, any game: the release's `dxgi.dll` imports `d4r_boot.dll` and DLSS works without Steam launch options
  (`d4r/d4r_boot.log`).
- Linux, vkd3d-proton 3.2 with the 17 patches: DLSS K/M in a D3D12 game (split command lists, external
  resources, RGBA8 output), and DLSS-G at x3+ without black generated frames (descriptor-heap opt-in).
  Windows' dxil-spirv patches (0006–0013) only ever ran on ef9a02e8, and 0011 needed a context rebase, so retest
  the shaders they target: the FSR RR path and any AGS WMMA titles.
- Linux, DLSS-G: `install.sh` setup on RDNA3 (Lite by default) and RDNA4 (full), x2/x4 in a game, and
  `[FrameGeneration] Backend = off` as a pass-through. The t1 decoder groups and wave64 kernels (RX 7700 XT).
- Native Windows, DLSS-G (RDNA3): unchanged settings path. Check FG still starts with the new t1 kernel cache
  (decoder groups) and that `dlssg_host.log` shows the expected model.
- Native Windows: `util.cpp`'s locale-free PTX/group parsing is shared with the native host.
- ReShade + d4r under Proton (#60): DLSS records into the native list (`evaluate: the command list is a proxy`
  in the shim log), with no reference leak over a long session (05dc7a0).
- Monster Hunter Wilds (970a013): startup with HIP preloaded (`comgr: N sigaction import(s) redirected` in the
  bridge trace), FG enabled, and exposure not falling back to CUDA.
- A game with R32_UINT depth (#61) on the Vulkan engine.
- RE Requiem (#59) with DLSS-G.
- Northlight (Control / Alan Wake 2): original motion vectors by default; `NativeMotionVectors = true` opts in.

## Added to the plan: linux commits after 65646d9

`origin/linux` has two commits the local `linux` branch does not (the local branch is left as is; these are
ported from `origin/linux`):

- **1dcfa80** (release: ship RR HIP stages and FSR RR; `LinearLighting` defaults to half): `build_rr_hip_release.sh`
  needs `engine/rr/hip/` (Stage 2), `patches/vkd3d-proton-fsr-rr/0002` applies on top of linux's 0001 (Stage 3),
  and `[DLSS] LinearLighting = half` needs the key from 65646d9 (Stage 5). Ported at the end of Stage 5.
- **4286001** (RDNA2 K: deep dot2 layers as phase dispatches, swizzled LDS rows, depth staged in shared memory):
  builds on ef0075f and 65646d9's `input_k.comp`, `compile_k.py` and `runtime.cpp`. Ported at the end of Stage 5,
  after 1dcfa80. Windows' native RDNA2 path (LLPC dot2 shaders, 130d119) is checked for an equivalent first.

## Stages 2-5 and linux up to 4b32f0e (2026-11-17)

Stage 1 had stopped at linux #63 (ad10640). The rest of `origin/linux` is now on main, which makes `linux` fully
merged. linux's history was rewritten and shares no commit with main, so the ports are content ports. A merge with
`-s ours` records `origin/linux` as a parent without taking its tree.

- **ef0075f, 1cdb649, 65646d9, 1dcfa80, 4286001, 7ca1e7f** (M and L on vkhip on RDNA3, DLSS and FSR RR on vkhip, K's
  linear lighting, the RR HIP release stages, RDNA2 K phase dispatches, RDNA 3.5 vkhip): one port of
  `7d55fa8..7ca1e7f`.
- **cbdb88e (#70), d6ccaa5 (#67), b4a635b (#69)**: Tk-Glitch's commits, each ported on its own with its author kept.
- **b912fbf**: the Linux Ghost of Tsushima table in README's Results uses its 2026-11-12 numbers. **793a100** (trailing
  spaces in the title) is skipped.

### Merge decisions

- **Two `dec0_l_head.comp`s.** main's (the D3D12 RDNA3 L head, m_d3d12_build.py) keeps the name. linux's Vulkan wide
  head is now `engine/dec0_l_head_wide.comp`. It still compiles to `dec0_l_head.spv`, so linux-built packages load
  unchanged.
- **Two M vkhip networks.** main's `vkhip::MNetwork` (create_engine_vkhip_net) and linux's `KNetwork::createM` (preset L's
  native enc0l/dec0l stages, the RDNA3 "wide" models) both stay. `engine_evaluate_m` takes linux's route for a wide
  model (engine_m_vkhip; it refuses when vkhip cannot load, as linux does) and main's otherwise. `createM` uses main's
  `Net` helpers: `loadEntryPoints` (AMD's pipeline binaries), `allocateArena`, `pipe`, `argBlock`, `finishContents`.
  `create_engine_vkhip_k` takes optional presetM, presetL and render-size arguments, so the native Vulkan frontend's K
  call is unchanged.
- **Two encoder-0 gathers.** `gen_enc0_m.py --gather` stays main's (scratch after the arena, kernels/rr/build_vkhip.sh).
  linux's (binding 23, engine/rr/hip/rr_enc0.hip) is `--gather-hip`. No linux script generates its
  `enc0_gather.spv` yet, and runtime_rr.inc skips vkhip encoder 0 without it.
- **RR on vkhip, presets D/E.** In the stage loop the model's own HIP pipelines (linux, RDNA3 packages, `model/hip`) come
  first, then main's stage recorder (RDNA4), then the Vulkan stage. The arena gets main's gather scratch, linux's
  TRANSFER_SRC usage and device address when either needs it. `debugArenaSnapshot` answers for D/E only (not v2/v3).
- **runtime_m.inc.** main creates the network and front stages lazily, so a wide model never compiles the FP8 layers
  that crash RADV on RDNA3. linux skipped creating them for the same reason. MEngine now refuses a wide model without
  an external network instead of reaching that path. linux's `downFor` / `downStore` replace main's
  rgb10Output/r11Output flags: `downFor` covers RGBA16F, RGB10A2, RGBA8, R11G11B10 and RGBA32F, and `hasDirectOutput`
  uses it. So `game_m.cpp`'s `packedOutput` and the Vulkan frontend's explicit RGB10/R11 checks use `hasDirectOutput`.
- **K's split dot2 layers (4286001)** are not loaded on AMD's Windows driver: they use the v_dot2 products its LLPC
  rejects, and there is no `_llpc` build of them, so those layers run whole there (their `_dot2_llpc.spv`).
- **Shaders shared with the D3D12 build** (hlsl_build.py):
  - input_k.comp's depth staging (STAGE_DEPTH) defaults off under D4R_HLSL, because the D3D12 stage's workgroups do not
    cover 32 x 32 pixels.
  - exposure_k.comp's 256-thread pass 0 is GLSL only.
  - output_k.comp keeps both reconstructions: main's D3D12 paths (D4R_WIN_OUTPUT's 2D walk, the f32 filter and sums) and
    linux's fast GLSL stages. Its packed running maximum (K_PACKED_YMAX) is GLSL only, because dialect_hlsl.h has no
    16-bit integer vectors. The HLSL builds keep the per-lane select, which gives the same values.
- **L's head functions** (layer_m.comp KIND 4) are `engine/l_head.glsl`, linux's file, with main's `M_AMD_HEAD` variant
  in it.
- **vkhip.cpp** accepts gfx1030-1036, gfx1100-1103, gfx1150-1153 and gfx1200/1201 code objects. main's `.rodata` handling
  (the whole section in the data image) replaces linux's narrower one.
- **Shim.** linux's 54 hunks were placed into the eight units, hunk by hand where main had changed the lines:
  - The f32 VRAM planes and outputs went into the vkd3d-proton VRAM branch. main's native Windows VRAM branch is
    unchanged.
  - The RDNA 3.5 adapter message is in vkhip_adapter.
- **Packaging.** linux's L-kernel accuracy rule is in `scripts/stage_native_kernels.sh` (the shared kernel staging),
  its f32 staging stores are in `scripts/build_vulkan_staging.py`, and `D4R_BUNDLE_RR_HIP` is in package_release.sh.

### Built and tested here (Windows)

- `scripts/build_d4r_nvngx_shim.sh` builds: the shim and ASI, the Vulkan engine (runtime, game*, vkhip, vkhip_net) and
  d4r_hip_net.dll.
- Shaders compiled:
  - **Vulkan (SPIR-V):** input/output/exposure K in the default, LDR, RGB10, E5B9G9R9, linear-lighting and accuracy
    variants; dec0_m, enc0_m and the planes and wide variants; down_m including the wide stores; dec0_l_head_wide;
    enc0_l_gather; layer_m's network layers and L's stages, with both plane layouts and M_AMD_HEAD; layer_dot2 whole
    (also D4R_PORTABLE_DOT2) and split (enc3, enc4, dec5, dec4, dec3).
  - **D3D12 (DXIL):** K's input, output (Windows and vkd3d targets) and exposure stages; M's enc0/dec0 (dot2), post,
    down, dec0_l_head, enc0_l_features and exposure.
- Python tests: no failure that main does not have. Two errors main has (test_dlssg_features route round trip,
  test_features v10 providers) pass.

### Needs testing on hardware (Linux)

- RDNA3: M and L on vkhip with wide models (install.sh builds them), including L's native input and expansion stages.
  RR D/E on vkhip with the rr-hip stages (D4R_BUNDLE_RR_HIP), and FSR RR.
- RDNA2: K with the split dot2 layers (Steam Deck: compare the frame time with 11-08's nightly).
- RDNA 3.5: K on vkhip.
- K's LinearLighting = half against the 11-08 nightly.
- The f32 VRAM planes and outputs on a vkd3d-proton game (D4R_SHIM_F32_INPUTS / D4R_SHIM_F32_OUTPUT).
- Native Windows: no regression for K, M, L and RR on RDNA3 and RDNA4. The shared shaders and runtime changed, the
  D3D12 paths did not.
