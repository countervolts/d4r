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
