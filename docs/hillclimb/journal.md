# Hill-climb journal (performance, native Windows, RX 9070 XT gfx1201)

Goal: docs/hillclimb/goal.md; method: docs/hillclimb/performance.md. Branch `hillclimb/perf` (from `windows`
fe53aa0). Never pushed.

## Environment (2026-11-04)

- RX 9070 XT (gfx1201), AMD Software driver 32.0.32015.7029 (26.10.1, dated 2026-10-18), Windows 11 Pro 26300.
- HAGS: `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\HwSchMode` is absent (driver/OS default). Not changed.
- **C: has only ~4 GB free.** Every build, runtime and run goes to `D:\d4r-hc` (103 GB free):
  - `D:\d4r-hc\build\tools`, `D:\d4r-hc\build\engine-win` (`scripts/build_windows_tools.ps1 -Out D:/d4r-hc/build`, 54 s)
  - the shim still builds into the repo's `build/` (`scripts/build_d4r_nvngx_shim.sh`, 52 s, ~15 MB)
  - `D:\d4r-hc\rt`: a portable d4r folder for the harnesses: `nvngx.dll` (= build/d4r_nvngx.dll), `d4r_hip_net.dll`,
    `packaging/d4r.ini` unchanged, NVIDIA's `nvngx_dlss.dll` (310.7) / `nvngx_dlssd.dll` (310.9.1) /
    `ngx/_nvngx.dll` from `%LOCALAPPDATA%\d4r\nvidia`, `nvcuda.dll` + `zluda\` + `nvapi\` copied from Cyberpunk's
    d4r folder (the Vulkan frontend's NGX init needs them), `engine\` a junction to the installed models
    `%LOCALAPPDATA%\d4r\engine-models\f94e8367186cb04b` (the same key Cyberpunk's install uses).
- Cyberpunk 2077 and Ghost of Tsushima already carry a release install of d4r (install.ps1, not windows-game.ps1;
  no `.d4r-backup\manifest.json`). Left untouched so far. Portal RTX has no d4r/OptiScaler.

## Benchmark (Phase 0)

Harness changes (tools only, no effect on the shim):
- `d3d12_dlss_harness`: `D4R_HARNESS_GPU_TIMING=<file>` writes "frame gpu_us" from two queue timestamps in the game's
  list around EvaluateFeature (the second lands in the split's second part when the shim splits the list, so it
  includes cross-queue waits). `D4R_HARNESS_GPU_SCENE=1` draws the motion scene with GPU copies in the evaluation's
  own list instead of six CPU uploads + waits per frame. Output byte-identical to the CPU scene (K 720p->1440p, 200
  frames, final frame sha 739591b7...).
- `vulkan_dlss_harness`: `D4R_HARNESS_SIZE="W H OW OH"` (was compile-time), `D4R_HARNESS_GPU_TIMING` (bottom-of-pipe
  timestamps around EvaluateFeature), `D4R_HARNESS_CHECK_EVERY=N` (read back/check every Nth + last frame),
  `D4R_HARNESS_PAN_CANVAS=1` (the pan scene as one canvas uploaded once and a GPU copy per frame).

Trap found: **the CPU-uploaded motion scene leaves ~7 ms between frames, and after ~70 frames the clock drops**: every
stage of K got 1.57x slower at once (gpu_eval 1.55 -> 2.43 ms at frame 70). With the GPU scene the frame interval is the
evaluation itself and K stays at ~1.5 ms over 200 frames.

Create flags 0x6b (HDR | MVLowRes | DepthInverted | 0x20 | AutoExposure) in both harnesses: the D3D12 harness default
(0) asks for display-resolution motion and the shim refuses render-size motion textures ("a texture does not cover
its rectangle"). The preset is forced with `D4R_DLSS_PRESET` (d4r.ini's `[DLSS] Model = K` otherwise overrides the
harness's hint and every SR case runs K). RR F: `D4R_HARNESS_FEATURE_ID=13 D4R_RR_PRESET=6`.

## First numbers with the shipped models (fe53aa0, 200 frames, median of frames 41-200, single run)

| Case | D3D12 (ms) | Vulkan (ms) | Target |
|---|---:|---:|---:|
| K 720p->1440p | 1.511 (DXIL network) | 3.668 (Vulkan engine network, no vkhip: package has no HIP layers) | 0.649 |
| M 720p->1440p | 1.641 (DXIL) | 29.6 (**CUDA staging fallback**: "missing hipnet.bin") | 1.081 |
| L 720p->1440p | 2.400 (DXIL) | **segfault** (LLPC) | 1.413 |
| RR F 720p->1440p | 4.066 (DXIL) | **segfault** | 2.100 |
| K 1080p->4K | 2.727 | | 1.532 |
| M 1080p->4K | 3.449 | | 2.553 |
| L 1080p->4K | 5.296 | | 3.161 |
| RR F 1080p->4K | 9.042 | | 4.658 |

K stage split at 720p->1440p (D4R_PROFILE, high clock): setup 0.008, exposure 0.002, input 0.107, network 0.86
(enc0 0.140, enc1 0.096, enc2 0.049, enc3 0.045, enc4 0.059, dec5 0.083, dec4 0.056, dec3 0.044, dec2 0.044, dec1 0.118,
dec0 0.150), output 0.550.

**The installer's models have no HIP layers on RDNA4**: `packaging/build_engine_models.py` adds `--hip` only for K and
only for gfx110x/gfx103x release kernels; M/L never. So on this machine a Vulkan game gets K on the slow Vulkan
network, M on CUDA staging, and L / RR F crash. The vkhip numbers in docs/experimental-vkhip.md were taken with
hand-built packages (`compile_m.py --hip KDIR`).

## Tried and rejected (seeded from earlier docs; do not retry without a new reason)

From docs/windows-k-d3dhip.md (gfx1101): memory-word handoff (hipStreamWaitValue32 on imported D3D12 memory never sees
D3D12 writes on Windows); splitting K's reconstruction around the HIP network (no gain); reconstruction as wave32
(slower, not identical); K input stage wave64 (slightly slower); DXIL prefetch depth 4 (-0.056 ms on 26.10.1, not
taken over).
From docs/performance.md (gfx1101 unless noted): f16 accumulation on WMMA (20x error); occupancy tweaks (VGPR caps,
waves-per-EU), persistent work-groups; non-temporal hints; splitting position-only layers by channel; HIP arrays over
Vulkan buffers (ROCm lacks the export); signalling Vulkan timeline semaphores from HIP.
From docs/experimental-vkhip.md (gfx1201): chaining the M plane-layout input stage into the first layer (RADV did not
overlap, no gain); RR F native blocks vs translated (now all translated, +0.5% vs Vulkan); rr2_m phases as vkhip kernels,
WGP mode for translated stages, partial unrolling, max-ilp/memory-clause scheduling for level 0-2 stages, source-order
schedule (all slower).

## Log

### 2026-11-04: Phase 0 setup

- Kernel sets for gfx1201 (D:\d4r-hc\kernels): `kernels/build.sh k` (K layers) and `D4R_NATIVE_FP8=1
  D4R_VKHIP_LAYERS=1 kernels/build.sh m` (M/L chained vkhip layers). ROCm 7.14, a few minutes each.
- Captures: the release builder (Cyberpunk's d4r\tools copy) with `--keep --models k,l,m` against D:\d4r-hc\rt-build
  (its own engine\ so the installed models are untouched): 1m55s; the rebuilt k/l/m are byte-identical to the
  installed ones.
- HIP packages in D:\d4r-hc\models: `compile_k.py ... --hip kernels/k-gfx1201 --capture`, `compile_m.py ... --hip
  kernels/m-vkhip-gfx1201` for m and l (adds hipnet.bin, hip/, *_planes.dxil). k-ldr/m-ldr/l-ldr/rr-d/rr-e are
  junctions to the installed packages.
- RR F from the repo (`extract_dlss_ptx.py`, `compile_rr2.py`, `compile_rr_dxil.py`, 2m06s) keeps native-work and
  has `enc0_llpc.spv` / `dec0_1_llpc.spv`; the installed rr-f has neither (it was built by the 2026-10-27 release kit),
  which is why the Vulkan harness segfaulted on F with the installed models. Its shaders differ from the installed
  ones (newer compile_rr2.py). `kernels/rr/build_vkhip_f.sh` wants `<stage>_m.preprocessed.comp` and
  `dec0_0_m.build.json`, which a fresh compile_rr2.py model does not have: generated them with
  `compile_rr_dxil.preprocess_f` (same defines), then the script built the 14 rr2 objects in 16 s, none with scratch.
- Vulkan with the HIP packages, 720p->1440p, single runs: K 3.73 (still the Vulkan network: the native Vulkan frontend
  never wired K's vkhip, only the vkd3d-proton adapter has it), M 1.430 (vkhip), L 5.110 (vkhip, soft-dot2 front),
  RR F 4.727 on its Vulkan route and **3.900 on vkhip (D4R_ENGINE_RR_VKHIP=1), byte-identical**.
- D3D12 M/L with HIP packages take d3dhip (GPU-fence handoff) by default; against `D4R_DX_NET=dxil` it is a wash:
  M 1.751 vs 1.624 (720p->1440p), 3.542 vs 3.527 (4K); L 2.520 vs 2.471, 5.430 vs 5.542; d3dhip p99 is worse.
  The matrix pins D3D12 SR on DXIL (the installed packages' route).
- `D4R_PREFER_ACCURACY=1` does not change D3D12 M at all (byte-identical to the fast set, same time): the D3D12 M
  engine has one set. The -acc cases stay in the matrix as the exact-set guard for any lane B change.

### 2026-11-04: Vulkan candidates (benchmarks interleaved after the GPU was free)

- **Contamination**: from 21:14 three `whisper-cli.exe` jobs (someone else's `worker3.sh slice_[pqr].tsv`, ~60% GPU
  each) ran until ~21:30. The Vulkan self-noise run made in that window is discarded. bench.py now samples the GPU
  engine counters before each case, waits while other processes use more than `max_other_gpu` (10%), and marks a case
  `busy` if they appear during it. Never kill other users' jobs.
- **RR F on vkhip by default with AMD's driver** (shim: `engine_rr_vkhip_capable`, F stays opt-in under RADV). Also
  fixed: `ExperimentalRrVkHip = auto` (d4r.ini.default's own value) was parsed as `true` with a "must be true or
  false" note; auto now keeps the default.
- **K's vkhip network in the native Vulkan frontend** (`create_engine_vkhip_k`, `NativeVkEngineState::kVkHip`): the
  vkd3d-proton adapter had it, the native Vulkan frontend never wired it.
- **Preset L's matrix stages on LLPC through `spirv-opt -O`**: the LLPC crash on L's FP8 cooperative-matrix modules
  (`enc0_l*.spv`, `dec0_l*.spv`, and the network layers `m_*.spv`) goes away after SPIRV-Tools' `-O`, exactly as it
  did for RR F's `enc0_llpc.spv`. The unoptimised module still segfaults (checked, `d4r-m` on a package without the
  dot2 plane files). `d4r-m` replay, 1280x720 -> 2560x1440 (GPU shared with the whisper jobs, so absolute numbers
  are high): input 613 / expansion 473 us (matrix, vkhip) against 2158 / 1832 (soft dot2); vkhip == hip
  byte-identical; PSNR against the D3D12 engine 51.75 / 51.83 dB (frames 0 / 3) against the soft-dot2 route's
  51.22 / 48.67 dB. The internal Vulkan matrix network (no vkhip) also compiles and runs then (51.6-53.0 dB against
  D3D12), which also removes the "Vulkan network fallback crashes on LLPC" known failure for L packages that have
  the `_llpc` files. Harness (pan scene, 200 frames, single run): L 1440p 5.153 -> 2.275 ms. Lane B for the Vulkan
  L route on AMD's driver (it changes that route's arithmetic from soft dot2 back to the matrix stages RADV runs);
  output against the pan scene's analytic truth (4x4 supersampled, `D:\d4r-hc\tools\pan_truth.py`): 27.85 dB
  against 27.83 dB for soft dot2, finite, same mean/std. The D3D12 path and RADV are untouched.
- LDR HIP packages built (`compile_m.py --ldr --hip`, `compile_k.py --ldr --hip`) into D:\d4r-hc\models\{k,m,l}-ldr
  (were junctions to the installed packages without HIP layers).

### 2026-11-05: workspace moved to C:\d4r-hc

- D: went away overnight (the queued Vulkan benchmarks died with it and were contaminated by other GPU work before
  that). C: now has ~199 GB free, so the whole workspace lives in `C:\d4r-hc` (copied from D:, junctions re-pointed,
  bench scripts and `runs/*.sh` repointed). Kernels and tools were rebuilt there too.
- bench.py now samples other processes' GPU use throughout each case (`GpuWatch`, a background thread), not only
  before and after; a case with another user above 10% at any sample is marked `busy`.
- The working tree also carries someone else's uncommitted D3D12 K input-copy work (`engine/d3d12/game_d3d12.cpp`,
  `prepare_live_k.py`, `replay_d3d12.cpp`, "black speckles"); it is not part of this climb's commits.

### 2026-11-05: Vulkan K on LLPC

- K's vkhip network in the native Vulkan frontend (wired up, uncommitted) never loads here: the gfx1201 K kernels
  spill (`enc0` 32 VGPRs spilled / 132 B scratch, `dec0` 25 / 104 B, `dec1` 9, `dec2` 10; all at 256 VGPRs) and the
  pipeline-binary injection refuses scratch. Fixing that means gfx12 flag lines in `kernels/k/*.hip`, which Linux
  shares: parked. bench: no change (same route).
- **Profile of the Vulkan K network on LLPC** (`D4R_PROFILE=1` through the shim, 720p -> 1440p): the deep layers are
  6-9x the D3D12 ones (enc3 0.28, enc4 0.38, dec5 0.77, dec4 0.38, dec3 0.27 ms against 0.045-0.083). Tools: a
  small `vkstats` (C:\d4r-hc\tools\vkstats.cpp: builds a compute pipeline with VK_KHR_pipeline_executable_properties
  and prints LLPC's statistics and ISA). dec5: 162 VGPRs (243 as wave32), no scratch, 136 WMMAs and **1168
  `buffer_load_u16`**: LLPC loads a row-major cooperative-matrix B operand one 16-bit element at a time (a lane of
  the WMMA B operand holds a column). `spirv-opt -O`: no change. Deeper prefetch (`PREFETCH_K/N=4`): no change.
- **Fix**: the D3D12 package's `weights_d3d12.bin` already transposes each 16x16 B tile (`D4R_WEIGHT_TILES_COLMAJOR`,
  HLSL only); the Vulkan layers with that define and the same file: 3.611 -> 1.794 ms (1440p), 5.473 -> 3.124 ms
  (4K), byte-identical (lane A). Remaining u16 loads come from the merge/expand/embed/head B operands and the bias
  tiles (accumulator layout, 16 equal rows): `D4R_LLPC_LOADS` transposes those too (`weights_llpc.bin`) and reads bias
  tiles with stride 0: dec5 208 -> 54 u16 loads, enc3 544 -> 48.

### 2026-11-05: profiles, D3D12 K output, d3dhip K

- **Tool install**: Radeon GPU Analyzer 2.15 (GPUOpen-Tools' GitHub release `rga-windows-x64-2.15.zip`, unpacked
  user-level into `C:\d4r-hc\.tools\rga`, profiling only). `rga -s dx12 -c gfx1201 --cs-blob X.dxil --isa ... -a ...`
  gives the ISA and register/scratch statistics of a DXIL shader as AMD's D3D12 compiler builds it (AGS WMMA works).
- Profiles at 720p -> 1440p (shim `D4R_PROFILE=1`, rt-cand5): D3D12 K 1.362 (input 0.089, network layers 0.789:
  enc0 0.130, dec0 0.127, dec1 0.100, enc1 0.085; **output 0.472**), D3D12 M 1.665 (network 1.041, recon 0.300),
  D3D12 L 2.449 (input 0.495, net 1.027, expansion 0.364, recon 0.379), D3D12 RR F 4.118 (enc0 0.840, dec0 0.948,
  dec1 0.466, output 0.683). Vulkan: L 2.249, RR F 3.765 (net 3.177), M 1.423 (net 0.798), K 1.65 (recon 0.371).
- D3D12 M per layer against its vkhip twin (us): enc1 204/205, enc2 135/98, tube 48-70/35, dec2 132/90, dec1 200/180.
  RGA: no scratch in any M DXIL layer. M DXIL `WPW=2` (two windows per wave): not identical and slower (network
  1015 -> 1073 us): rejected.
- RGA on D3D12 output_k: 118 VGPRs, 3042 instructions, no packed f16 (AMD's DXIL compiler scalarises SM 6.6 half
  vectors; the Vulkan build of the same stage has 69 VGPRs and packed f16). SM 6.9 is not an option for games.
  K network DXIL layers: no scratch except dec5 (48 B).
- **output_k as wave32** (D3D12): 1.424 against 1.377 ms and not byte-identical: rejected.
- **output_k ablation** (timing only, wrong output): the 12-tap filter cut to 3 taps saves 0.14 ms (so the filter
  costs ~0.19 ms), the luma-history Catmull-Rom removed saves 0.10 ms; tile load, motion, blend and stores ~0.18 ms.
- **K HIP kernels on gfx1201 spill under every flag set tried** (enc0 84-144 B, dec0 48-104 B, dec1 40-48 B, dec2
  32-44 B; enc2 also spills with `PWIN_NORM_OWNER`), so the vkhip pipeline-binary route (which refuses scratch) stays
  closed for Vulkan K.
- **d3dhip K on RDNA4**: `D4R_D3DHIP_K=1` (opt-in, written for RDNA3 26.10.1) on the RX 9070 XT: 1.300 against
  1.377 ms at 720p -> 1440p, 2.700 against 2.773 at 1080p -> 4K (interleaved, 2x2 runs). Not byte-identical to the
  DXIL network (different implementation: HIP accumulates in f32 where the DXIL layers accumulate in f16), so lane B.
  Harness quality scene (static and panning 3,1 px/frame, 60 frames): psnr 22.12 / 10.34 dB, blinds 24.36 / 7.12,
  flicker 0.00500 / 0.30493 against DXIL's 22.13 / 10.34, 24.36 / 7.12, 0.00501 / 0.30493: the same.
- gfx1201 HIP K flag sweep (24 one-flag variants from the gfx11 Windows set, each swapped into the package, C X X C):
  all byte-identical, all within the d3dhip route's run-to-run spread (+-0.02 ms); nothing clearly faster.
- **Kept (lane B): d3dhip K by default on RDNA4** in the fast set (`D4R_D3DHIP_K` default 2; `0` keeps DXIL;
  PreferAccuracy keeps DXIL, byte-identical to before). bench.py (3 rounds C X X C): dx-K-1440 1.371 -> 1.297
  (-5.40%), dx-K-4k 2.799 -> 2.744 (-1.95%), dx-K-q1440 1.406 -> 1.331 (-5.32%), dx-K-1440-rgb10 1.374 -> 1.304
  (-5.11%); candidate spreads up to 2.4% (the cross-queue handoff is noisier, p99 a little higher). LDR K 1.318 ->
  1.238, 55.5 dB against DXIL; HDR 57.2 dB, finite, same mean/std. The Cyberpunk check is blocked (BLOCKERS:
  release installs that windows-game.ps1 refuses). The HIP kernels are built by our LLVM, so AMD's Cyberpunk D3D12
  application profile (the half-accumulation hazard) cannot reach them, but the in-game check is still owed.
- **Kept (lane B): K on vkhip in the native Vulkan frontend, with scratch on gfx12.** The pipeline-binary injection
  now sets the PAL stage metadata's `.scratch_en` / `.scratch_memory_size` from the kernel descriptor on gfx12
  (architected flat scratch; `D4R_VKHIP_SCRATCH=0` refuses as before; gfx11 still refused) instead of refusing the
  four spilling K layers, and the native frontend records K's front, vkhip network and back (the uncommitted wiring
  from earlier). bench.py (3 rounds): vk-K-1440 1.652 -> 1.107 (-32.97%), vk-K-4k 2.938 -> 2.392 (-18.57%),
  dx-K-1440 identical. Output against the Vulkan network: pan scene 65.7 dB (truth PSNR 29.84 dB both), static
  107 dB, LDR 61.5 dB; finite, same mean/std. vk M, L and RR F byte-identical.
- **vkhip K at 4K was nondeterministic** (found by bench.py's hash check: two control hashes on vk-K-4k). 600-frame
  runs at 1920x1080 -> 3840x2160: 15-25% diverge (67 dB from the usual output, all over the frame: one early glitch
  spread by the history); 1440p 16/16 stable; the Vulkan network route 12/12 and d3dhip with the same kernels 12/12
  stable. Not scratch: LLPC's own metadata uses bytes per lane too (`vk_pipeline_stats` `VKSTATS_ELF` + a msgpack
  reader, `C:\d4r-hc\tools\palmeta.py`), and 2x / 4x / 32x the scratch size still diverged (4/12 at 2x). Not the
  barriers (MEMORY_READ/WRITE after every dispatch: 2/12), not `-mcumode` (enc0/dec0 without it: 3/16). **The
  trimmed grids**: `D4R_ENGINE_K_TRIM=0` was stable 16/16 with the usual output's hash. d3dhip never trims. Fix:
  vkhip K launches the full grids on AMD's driver (RADV keeps trimming): 32/32 stable; bench vk-K-1440 +0.62%,
  vk-K-4k -0.29% (noise), identical. Why trimming races is not understood: the trimmed windows' tokens are not read by
  the rule the trimming uses, so some HIP layer may read them after all (a padded window's halo) and pick up whatever
  another dispatch left there.
- **The vkhip K race is not the trimming.** The Windows-flag K kernels (below) diverged again at 4K with the full
  grids (enc0, enc2 and dec0 variants each alone: 1-2 of 12 runs); the unchanged kernels had just been lucky (0/32).
  vkhip M at 4K (no scratch): 12/12 stable; d3dhip K with the same kernels: 12/12 stable (both flag sets). So the race
  goes with the spilling layers on the pipeline-binary path. **Reverted to opt-in**: gfx12 scratch through pipeline
  binaries only with `D4R_VKHIP_SCRATCH=1` (K's Vulkan route is back on the Vulkan network, byte-identical to before
  cdc0787, vk-K-4k 2.922 against 2.916), and the trim change of 2291ccc is reverted (it only lowered the rate). The
  opt-in route keeps its speed (1.105 ms at 1440p). Spill-free gfx12 K layers are what would make it safe: a kernel
  change (register pressure in pwin_layer.h / pwin_pos.h) for a later iteration.
- **Found the K race: a missing barrier in `pos_core`** (kernels/k/pwin_pos.h, enc0 and dec0). In the default path
  (no `POS_VSHARE`), every wave computes V for each head from the `hb` rows of all 64 tokens; after the head loop it
  rewrites its own `hb` rows (residual -> MLP input) with no barrier between, so a wave that finishes its heads first
  overwrites rows another wave is still reading. `pwin_pos_recompute.h` has exactly this barrier ("All attention
  readers finish before the shared input rows are reused"); `pwin_layer.h` reads only its own rows or syncs per head.
  How it showed: the Windows-flag kernels diverged through d3dhip too (dx-K-q1440: 2 of 6 bench runs, then 1 of 40
  interleaved runs; control 0 of 68), and later the *control* (the sources' gfx12 flags, the committed d3dhip default)
  diverged once in 6 bench runs. Through vkhip at 4K (600 frames, `D4R_VKHIP_SCRATCH=1`) the unfixed kernels diverged
  in 2 of 24; with the barrier, **64 of 64 identical**, the same hash as the race-free runs. The scratch through
  pipeline binaries was innocent; the spills just skew the waves.
- Fix: `POS_HB_SYNC` (barrier plus a `d4r_pos_hb_sync` marker global), set for gfx1201 and gfx12 enc0/dec0 in
  kernels/windows-flags.txt (the shared flag lines, Linux and gfx11 compile unchanged; with the macro unset the
  preprocessor output is identical). Code of the nine other layers unchanged (objdump). enc0 scratch 132 -> 108 B.
  vkhip: gfx12 scratch back on by default (`D4R_VKHIP_SCRATCH=0` refuses); vkhip_net refuses gfx12 enc0/dec0 without
  the marker ("predates the gfx12 cross-wave barrier"), so old packages stay on the Vulkan network (checked: 1.654 ms,
  network on Vulkan) and new ones take vkhip (1.084 ms at 720p -> 1440p).
- d3dhip with the Windows-flag + barrier kernels (bench, 3 rounds): dx-K-1440 1.304 -> 1.285 (-1.48%), dx-K-4k 2.743
  -> 2.693 (-1.81%), dx-K-q1440 1.335 -> 1.306 (-2.17%), rgb10 -1.21%; candidate single-hashed everywhere.
- **Kept: the barrier, the Windows flag file, vkhip K by default again on gfx12.** Full matrix (3 rounds) against
  7c029f0: vk-K-1440 1.653 -> 1.085 (-34.36%), vk-K-4k 2.955 -> 2.378 (-19.53%) (lane B, the route of cdc0787 whose
  output checks stand: the fixed kernels give the race-free hash), dx-K-1440 -1.38%, dx-K-4k -1.46%, dx-K-q1440 -1.51%,
  dx-K-1440-rgb10 -1.96% (identical); all other cases within noise and identical; every candidate side single-hashed.
  Score 2.654 -> 2.557 (d3d12 2.499 -> 2.489, vulkan 2.819 -> 2.628). Local packages models\k and k-ldr now hold the
  kernels/build.sh k set built on Windows (k-gfx1201-sync2).
- Disk: bench.py kept every run's raw output (30-130 MB) in its work folder; 75 GB filled C: and broke a matrix run.
  It now deletes them after hashing (`BENCH_KEEP_OUTPUT=1` keeps them); the old work folders were removed.
- **D3D12 M on d3dhip, re-measured** (rt-cand13): 1.70 against DXIL 1.65 at 720p -> 1440p, 3.44 against 3.49 at 4K.
  Per-node HIP timing (a local bridge hack, `debugPhaseMicros`): K's isolated graph is 593 us, the same as vkhip's
  network, so d3dhip K loses only its ~0.12 ms handoff; M's is 956 us against vkhip's ~0.80 ms for the same code
  objects. The difference: vkhip chains M's ten layers through per-block flags (`d4r_dataflow`), while HIP put a full
  barrier between them. (Note: one.sh needs numeric presets for M/L too, 13/12; a letter silently runs K.)
- **Kept (lane A): the M chain in hip_net.cpp on Windows.** It copies vkhip's flag layout and `r69`-`r72` wiring. The
  consumers launch with `hipExtModuleLaunchKernel(..., hipExtAnyOrderLaunch)`, the flags are cleared each
  evaluation, and the chained nodes launch directly instead of as a graph (`D4R_HIP_M_CHAIN=0` keeps the barriers).
  The isolated graph goes from 956 to 787 us. Side matrix (`matrix-hip.json`: the dx-M/L cases without `D4R_DX_NET=dxil`, 3 rounds),
  all identical: dx-M-1440 -9.19%, dx-M-4k -1.75%, dx-M-720to4k -5.12%, dx-M-q1440 -2.48%, rgb10 -9.62%, dx-L-1440
  -4.74%, dx-L-4k -1.07%, dx-L-rgb10 -4.98%. Deterministic: 24 of 24 600-frame runs at 4K and at 1440p. The main
  matrix pins D3D12 M/L on DXIL (installed packages have no HIP layers), so its score does not move: dx-K and dx-M
  (DXIL) unchanged and identical. d3dhip M at 1440p is now ahead of DXIL (1.534 against 1.65).
- **Tried, rejected: a barrier-free K chain with per-dispatch completion counters.** Bound first (throwaway bridge,
  `hipExtAnyOrderLaunch` on chosen K nodes, wrong output): no barriers at all 602 -> 531 us isolated at 720p ->
  1440p (1196 -> 1079 at 4K), end-to-end 1.294 -> 1.231; only the deep part (nodes 3-11: enc3, enc4, dec5's phases,
  dec4, dec3, dec2) 550 us; only the enc0-2 / dec2-0 joins 590; only dec5's internal phases 594. Then the real thing:
  `PWIN_CHAIN` (a guard object at every K entry: wait until the previous dispatch's counter reaches its block count,
  count this block at exit; pointers in PwinParams' unused tail; `+FLAGS` glob lines in windows-flags.txt). Output
  byte-identical, but 643 us against 638 unchained. Worse, the guard code alone made the kernels slower with the
  chain off (595 -> 638 us). So the saving in the bound came from consecutive layers actually overlapping, not from
  relaunch latency; only per-block dataflow (as M's `SWIN_DATAFLOW`) can get it, and the deep layers' phase kernels
  with compile-time grids make that a large kernel project. Reverted.
- **d3dhip handoff (~0.12 ms on K, also on M/L)**: K's isolated HIP network matches vkhip's, so the handoff is the
  whole d3dhip overhead. It is two cross-API fence waits (D3D12 front -> HIP, HIP -> game queue). dxdiag reports
  "Hardware Scheduling: Enabled:True" with "Block List: DISABLE_HWSCH". If the OS resolves these waits on the CPU,
  2 x ~50-60 us fits. Scheduling settings are off limits, so not pursued; no code-side lever found.
- **RR F enc0 / dec0_1 as HIP**: the vkhip F route runs 14 stages as HIP but enc0 and dec0_1 (the two largest on
  D3D12, 0.84 / 0.95 ms) stay SPIR-V: they are translated-PTX shaders (`*.generated.comp`, 19.6k / 12.2k lines)
  with no `_m` source, and glsl2hip stops at enc0's `sampler2D` (texture sampling, which the pipeline-binary HIP
  path has no binding for). Not a quick candidate. D3D12 RR F on d3dhip exists only in the replay tool (not the
  shim), and would save at most ~0.1-0.2 ms of network minus the handoff: not pursued.
- **Tried, rejected: gfx1201 flag sweep of M's HIP layers** (vkhip M 720p -> 1440p, C X X C, 160 frames). The base
  flags reproduce the package's code exactly. Of 60 variants (the gfx1101 macros one at a time, cumode, wave priority,
  the memory-clause scheduler, each layer's gfx1101 line), 38 compile to the base's code (the SWIN_DATAFLOW /
  DIRECT_E4M3 path ignores those macros). The other 22: all identical, all within +-0.35%. The four best singles
  together (enc1 memory-clause, enc2 half-bpermute, tube both, dec1 cumode): 1.4152 -> 1.4068 (-0.59%), below the 1%
  bar.
- **Tried, rejected: steering LLPC on L's front stages.** Vulkan L spends ~1.09 ms in `enc0_l_planes_llpc.spv` /
  `dec0_l_planes_llpc.spv` (D3D12: 0.86). As the engine creates them (32-wide subgroups, required) LLPC uses 116 /
  109 VGPRs. Without the requirement it builds wave64 at 31 / 21 VGPRs, but `layer_m.comp` assumes 32-lane waves
  (`id & 31`, `id >> 5`, shuffles by 16), so that is not an option. spirv-opt variants (`-O --loop-unroll`, `-O -O`,
  `--loop-unroll-partial=2`, `--scalar-replacement=0`) give the same module as `-O`; `-Os` the same registers and LDS.
- Stage wave sizes on gfx1201 (outputs identical): K's stages as wave32 1.08 -> 1.16 ms (720p -> 1440p), 2.34 ->
  2.49 (4K); M's 1.42 -> 1.48, L's 2.23 -> 2.30. Wave64, the default, stays. d3dhip CPU handoff
  (`D4R_D3DHIP_SYNC=cpu`) against the GPU fence: K 1.35 against 1.28, M 1.60 against 1.53.
- **Stopped at the plateau condition** (10 iterations since ed503cc without a kept change, across >= 3
  opportunities); summary at the top of scoreboard.md. Game checks still owed (BLOCKERS).

### 2026-11-05: after the plateau stop (the stop was rejected: >5% items left untried)

- **K's output stage on Vulkan, ablations** (timing only, wrong output; vk-K-1440, 1.085 ms): the 12-tap filter
  removed 0.175 ms, the Catmull-Rom history 0.083, both 0.257. Inside the filter: the exps cost nothing measurable, the
  LDS reads nothing, the max-tap tracking 0.028 ms (2.6%). The filter is ~534 of the stage's 1546 VALU instructions
  (69 VGPRs either way), already packed f16. No byte-identical lever above ~0.03 ms: not pursued. The centre-only
  estimate (`D4R_GAUSSIAN_CENTER`) was dropped earlier for quality (compile_k.py), so it is not an option.
- **RR F's enc0 / dec0_1 on LLPC** (vk-RRF-1440, 3.77 ms): an empty enc0 saves 0.83 ms, an empty dec0_1 0.53 (36% of
  the case between them). enc0_llpc: 202 VGPRs, 10.7k instructions, no scratch; dec0_1_llpc: 189 VGPRs, 9.1k. Of
  enc0's instructions ~4.4k are the FP8 conversions' guards: `rr_exact.glsl`'s `FINITE` (infinity -> NaN before
  `saturatedConvertEXT`) is RADV's pattern for one conversion; LLPC keeps a compare and a select per element, each
  select waiting on its compare's VCC (`s_wait_alu depctr_va_vcc`), and writes MODE.FP16_OVFL before every
  conversion. LLPC test shaders: `saturatedConvertEXT` without FINITE is worse (clamp, class test, select per
  element); a plain conversion is the mode write and the conversion only; `clamp(v, +-448)` + plain conversion is one
  `v_med3` per element more. No LLPC form keeps a NaN's code more cheaply than FINITE.
  - Plain conversions everywhere (ablation): 3.77 -> 3.43 ms, output different (overflow becomes NaN).
  - `clamp` + plain conversion at every site: 3.53 ms, output different. Bisected: only `Q8(Nb, rv)` (the residual
    r = x + attention in dec0_1, KIND 5) changes the images, so r is not always finite there. ACT8, Q8 of the
    attention's f32 sums (q, k, v, o) and enc0's input are identical.
  - Kept: `RR_LLPC_SAT` (rr_exact.glsl `SAT8V`), clamp + plain conversion for the f32 sums of FP8 products; the
    residual and the tail (r + MLP) keep the guarded form (`Q8G`). compile_rr2.py compiles `<stage>_llpc.spv` with it
    on Windows hosts (`D4R_RR_LLPC_SAT`, default 1 on Windows, 0 elsewhere). With the define off the rebuilt model is
    byte-identical to HEAD's in all 99 files compile_rr2.py writes; on, only enc0_llpc.spv and dec0_1_llpc.spv change.
    The preprocessed sources the D3D12 and HIP builds use differ only in whitespace (dec0_0's glsl2hip output too).
    For finite values the conversion is the same (nearest even, everything above 448 saturates to 448). A NaN
    accumulator now gives +-448 instead of the NaN code: the harness shows none at these sites, but a NaN from a game
    input would reach them; NVIDIA's network propagates NaN there.
  - Bench (bench.py, full matrix, 3 rounds C X X C, rt-cand14 against the same runtime with the two rebuilt files):
    vk-RRF-1440 3.781 -> 3.561 (-5.82%, spread 0.82%), vk-RRF-4k 8.269 -> 7.755 (-6.21%, spread 0.46%); every
    case identical, nothing else outside its spread (worst vk-L-4k +0.18%). Scores D3D12 2.484 / 2.485, Vulkan
    2.620 -> 2.584. Lane A on the harness; the NaN case above is the only numerical difference.
- **Preset L's front stages on LLPC, the same conversions** (`layer_m.comp` `SAT2_` / `SATF2_`, enc0_l_planes_llpc /
  dec0_l_planes_llpc, vk-L-1440 2.245 ms). `M_LLPC_SAT`: enc0 4920 -> 4189 instructions, dec0 3899 -> 3386.
  - Every conversion clamped: 2.245 -> 2.155 ms (-4.0%), but the images change widely (8 of 29 MB of the output
    frame), deterministically. Each site group alone (`SAT2_`; attention sums `MK8_F8F`; GELU; ReLU; softmax
    probabilities; normalised tokens `H1N`) is identical. Cause: LLPC writes MODE.FP16_OVFL before every FP8
    conversion (1 for a saturating one, 0 for a plain one) and never restores it, so the binary16 arithmetic that
    follows saturates or overflows depending on the last conversion before it. The base dec0 runs all of it with
    FP16_OVFL set; with every conversion plain it never is. `saturatedConvertEXT(clamp(v))` does not help (LLPC keeps
    its clamp, class test and select).
  - Keeping one group guarded: the attention sums or the normalised tokens give identical images (2.220 -> 2.169 /
    2.240 -> 2.171 in single runs); GELU, ReLU or the softmax group do not. Kept: the normalised tokens always guarded
    (`SATN2_`, which are also NaN by design where a channel is infinite), everything else clamped under
    `M_LLPC_SAT`; compile_m.py builds the two `_planes_llpc.spv` with it on Windows hosts (`D4R_M_LLPC_SAT`). The
    non-planes `_llpc` modules keep the old form: on this driver the runtime never reaches them (without the planes
    matrix modules it takes the dot2 planes stages, 5.16 ms), so they cannot be tested here. Full compile_m.py
    rebuilds of M and L: M's files all identical, L's only the two modules (and model.json's hashes of them; its
    hsaco hashes differ because kernels/m-vkhip-gfx1201 was rebuilt at 02:42, after the installed models).
  - Caveat for this and for RR F's change above: identity holds where no binary16 operation overflows under a
    different FP16_OVFL state than before. The mode sequence moved in both (base dec0_l: 128 writes of 1; now
    16 x 0, 16 x 1, 64 x 0, 16 x 1, 16 x 0); the base's own sequence is a compiler artefact, not a design choice,
    and every bench case is identical. Content with binary16 overflow in those stages could differ.
  - Bench (Vulkan cases, 3 rounds C X X C; control = the RR F commit's runtime, candidate = the two rebuilt L
    modules): vk-L-1440 2.225 -> 2.181 (-1.97%, spread 1.33%), vk-L-4k 5.028 -> 4.886 (-2.82%, spread 0.72%), every
    case identical, the others within their spreads. Vulkan score 2.574 -> 2.561. D3D12 never loads these modules.
- **Tried, rejected: RR F's gather pass (enc0g) on LLPC**: an empty enc0g saves 0.40 ms of vk-RRF-1440's 3.53 (11%). 8.6k
  instructions, 293 branches, 139 mode writes: ptx2rr.py's `h2e4m3_hw` (`isnan(h) ? 0x7f : convert(clamp(h))`) is an
  exec-mask branch per element on LLPC (86 `v_cmpx_o_f16` + `s_cbranch_execz`), one conversion per element. Under
  `RR_LLPC_SAT` `f16x2_to_e4m3x2` converts both halves in one instruction and selects 0x7f for a NaN half afterwards:
  the same codes for every input, NaN included. 7.7k instructions, 207 branches, 45 conversions. compile_rr2.py
  writes `enc0g_llpc.spv` with it on Windows hosts (the runtime loads any stage's `_llpc.spv` on that driver). With the
  define off every .spv is identical to the previous build; the `*.generated.comp` texts carry the new helper, and
  their preprocessed text (what the D3D12 build reads) is unchanged apart from whitespace. enc0_llpc / dec0_1_llpc do
  not change (their few conversions of this kind are dead code there). Single runs: 3.539 -> 3.502 ms (-1.0%),
  identical.
  Bench (Vulkan cases, 3 rounds): vk-RRF-1440 3.547 -> 3.513 (-0.96%, spread 1.36%), vk-RRF-4k 7.747 -> 7.700
  (-0.60%), identical: under the 1% bar. Reverted.
- **Kept: RR F's gather pass (enc0g) on D3D12, hardware FP8 encode** (dx-RRF-1440, 4.09 ms). An empty enc0g saves
  0.47 ms; a trivial encoder (wrong output) 0.12 ms. Making the software codec branch-free (86 exec-mask branches
  gone, 8271 -> 7643 instructions) gave nothing (4.09 -> 4.11, identical). `rr2_pixel_opt._packed_encode` now emits
  `rrE4x2W` where the wave runs uniformly (main's body or the wave-uniform `if (rr2_valid)`): both halves clamped to
  +-448 and one AGS conversion (lane-local `v_cvt_pk_fp8_f32`), NaN halves selected back to 0x7f, so every input
  keeps its code. 53 of enc0g's 54 sites; 6493 instructions. vkd3d keeps its text (`#define rrE4x2W rrE4x2`; the
  preprocessed enc0g is identical). Full compile_rr_dxil.py rebuild: only enc0g.dxil changes (other .hlsl differ in
  blank lines only, their DXIL identical). Interleaved single runs (C X, 160 frames): 1440p 4.090/4.018,
  4.092/4.022, 4.096/4.033 (-1.6%); 4K 9.040 -> 8.888 (-1.7%); identical every time. Bench (dx cases, 3 rounds):
  dx-RRF-1440 4.122 -> 4.061 (-1.50%, spread 0.15%), dx-RRF-4k 9.083 -> 8.942 (-1.55%, spread 0.36%), all identical,
  the others within their spreads; D3D12 score 2.485 -> 2.480.
- **D3D12 RR F, where the rest goes** (empty-shader ablations on 0f61b35's base, dx-RRF-1440 4.034 ms, single runs):
  enc0 -0.66, post -0.56, dec0_0 -0.48, dec0_1 -0.47 ms (gather -0.40 before the change above). enc0's FP8
  conversions are already `v_maxmin` + `v_cvt_pk_fp8_f32` (640 / 320 of 4843 instructions, 215 VGPRs, 7 waves); dec0_1's
  selects (~190) are lane-shuffle selects around `v_permlane16_var`, not conversion guards. Per stage D3D12 is now close
  to the Vulkan route (enc0 ~0.6, dec0_1 ~0.45 on LLPC); the Vulkan network's remaining ~0.5 ms lead is its HIP stages,
  which the journal's d3dhip estimate (<= 0.1-0.2 ms after the handoff) already covers. No further cheap lever found.
- **Tried, rejected: RR F's network on d3dhip with the GPU-fence handoff.** Per-stage stamps (synthetic 720p -> 1440p
  case, `C:\d4r-hc\inputs\rr-syn`): F's 14 HIP stages take 1.61 ms on vkhip (Vulkan replay, `D4R_ENGINE_VKHIP=1
  D4R_ENGINE_RR_STAGE_STAMPS=1`) against 2.05 ms for the same stages as DXIL on D3D12, spread over every stage (enc4 39
  against 74 us, dec1 385 against 455, dec0_0 349 against ~460). Through the HIP runtime on D3D12 `HipNetworkRr`
  measured 2.08-2.21 ms, but its timed region held 14 synchronous parameter uploads: without them 1.63-1.66 ms.
  Prototype (reverted): `HipNetworkRr::useFence` / `launchAfter` (one asynchronous upload of all parameter blocks from a
  pinned ring, then the 14 launches between a wait on and a signal of a shared D3D12 fence) and a replay mode that
  records front and back up front and orders them with the fence. End to end 4.13-4.14 ms against DXIL's 3.87 (two
  runs each, 120 frames): the network's 0.4 ms gain is eaten by about 0.6 ms of cross-API handoff (the two fence
  waits; HAGS is block-listed here, so they may resolve on the CPU, and scheduling settings are off limits). Also not
  byte-identical to the DXIL network. The remaining DXIL-vs-HIP gap is per-stage code generation.
- **D3D12 RR F's phase split** (`D4R_RR2_SPLIT_WINDOWS`, default 48: levels with that many windows or fewer run as
  four phase dispatches): 0 / 16 / 24 / 48 / 100 give 4.060 / 4.052 / 4.035 / 4.032 / 4.027 ms (single runs, all
  identical): flat, no lever.
- **Why F's DXIL stages trail their HIP builds** (enc4: 74 against 39 us): RGA / llvm-objdump of enc4_m.dxil against
  rr2_enc4.hsaco: 256 VGPRs against 237, 3436 against 2223 static instructions. The HIP kernel sets the saturating FP8
  mode once and converts with `v_cvt_pk_fp8_f32` alone; the DXIL needs a `v_maxmin` clamp per element (264), because
  HLSL cannot set MODE and the driver's saturating AGS conversion is slower still (compare + select per element).
  The rest is address arithmetic (383 `v_add_nc_u32`) and 2x the unrolled WMMA count. That is AMD's D3D12 compiler,
  not a source-level lever.
- **RR F's enc0 / dec0_1 as vkhip HIP kernels, scoped (not built).** The Vulkan modules come from the hybrid
  `native-work/<stage>/<stage>_m.comp`, not the 19.6k-line translation. enc0's hybrid touches no texture, image or extra
  buffer in its body (declarations and an unused `fetch0` only): stripped of those, with `rr2_xf` passed to
  `rr2_block` and `gl_NumWorkGroups` taken from the parameter block (words 0 / 1 / 20, as the shader's `rr2_gx`),
  glsl2hip translates it, but its translated-PTX parts need GLSL that `kernels/rr/glsl_hip.h` lacks (`hf` literals,
  `subgroupBarrier` / `subgroupBallot`, `unpackHalf2x16` / `packHalf2x16`, `mix`, `roundEven`, `findMSB`, `frexp`,
  `u16vec2`, `ivec4`, float16 accumulator casts, ~30 kinds), and vkhip's F loader takes stages 1-14 only. Estimated
  gain: LLPC ran the other 14 stages ~33% slower than their HIP builds; enc0 ~0.45 ms -> ~0.30, about 4% of vk-RRF-1440,
  under the 5% bar. dec0_1's hybrid stores images (`sust_p4` / `sust_b4`) in its body, so it needs image descriptors in
  vkhip's F kernel ABI (D/E have them through VK_EXT_descriptor_buffer; F's ABI is {param, weights, arena}): with
  enc0g, post, down and pooled that is the multi-day project the summary lists.
- **Kept: RR F's enc0 on vkhip (Windows).** Built after all; most of the ~30 "missing" GLSL kinds sat in functions
  main never reaches. `kernels/rr/enc0_hip_prep.py`:
  - strips the bindings and accessors;
  - passes `rr2_xf` as a parameter;
  - takes the grid from the parameter block (`gl_NumWorkGroups.z` = 1);
  - prunes unreachable functions (79 -> 23);
  - rewrites `hf` literals, the `spirv_instruction` NMin, the `Q1023` table, `vec4(...)`, accumulator casts and one
    division.
  `kernels/rr/glsl_hip_enc0.h` (`glsl2hip.py --include=`) supplies the rest. One trap: `float(qx[wv][m][i])` on a
  non-const FP8 operand went through `Ref::operator float`, whose `floate4m3_t(*this)` picks the float constructor
  again. That is an out-of-line recursive call: 16 bytes of scratch and 58 SGPRs. Reading it through the const
  accessor (`glslF8f`) gives 0 scratch, 226 VGPRs, 20 SGPRs. The F loader, the replay and the shim now take stage 0.
  `build_vkhip_f.sh` builds it only on MSYS (Linux/RADV unchanged). Rebuilding all 15 code objects from scratch
  matches the tested ones in disassembly and notes. Bench (`rt-vc` vs `rt-ve`, same new shim; the candidate adds
  `hip/rr2_enc0.hsaco`), 3 rounds:
  - vk-RRF-1440 3.521 -> 3.412 (-3.09%, spread 0.73%);
  - vk-RRF-4k 7.731 -> 7.520 (-2.74%, spread 0.41%);
  - output byte-identical; Vulkan score 5.218 -> 5.066 over the RRF cases.
- enc0's `RR2_WAVES` 6 against 5: 3.425 against 3.429 ms (three ABAB pairs, identical output): flat, so 5 stays. 0 / 4 / 8
  spill (32 bytes of scratch).
- **Owner, 2026-11-05: byte-identical output is not a requirement.** From now on, numerics-changing levers are taken
  when they are faster and their quality holds (PSNR / max-diff against the current output, recorded here), not only
  lane A ones. Levers dropped partly for not being identical get a second look.
- **Tried, rejected (lane B): K's D3D12 output stage in f32.** AMD's DXIL compiler scalarises its half vectors anyway,
  so the scratch variant turned `float16_t` / `f16vec*` into f32 (bit casts through `f16tof32` / `f32tof16`):
  - RGA: 2112 against 2155 VALU, 107 against 10 conversions, 118 against 107 VGPRs, LDS 7.9 against 12.5 KB.
  - dx-K-1440: 1.327 -> 1.304 ms (-1.7%, three ABAB pairs).
  - The output moves 28.2 dB from the base: 30% of values change by more than 1e-3, max 0.97, finite.
  As output_k.comp's comments say, the reference filters in half precision: its sample positions sit on half
  rounding ties, and the recurrent history spreads any change over the frame. An f32 filter is another algorithm,
  not a rounding nuance. Not worth 1.7%.
- **RR F's dec0_1 as a vkhip HIP kernel: built and measured, parked.** Patch kept outside the repo
  (`C:\d4r-hc\tmp\d1hip\dec0_1-vkhip.patch`). What it took:
  - The hybrid prep (generalised to `rr2_hybrid_prep.py`) turns the four storage images (176 rgba8_snorm, 184 / 192
    rgba8, 200 rgba16f) into a binding-3 descriptor table.
  - glsl2hip gets `GLSL_IMAGES`: a fourth kernel argument and stores through `llvm.amdgcn.image.store.2d`, with the
    hardware's bounds check standing in for the shader's. Module-scope private arrays (`rr2_r16` / `rr2_m16`) become
    kernel locals, and their users lambdas.
  - The F loader writes the four descriptors (vkGetDescriptorEXT, `RStageResources::surfaceF`) after the arguments.
  Results:
  - Spill-free at `RR2_WAVES=6` (211 VGPRs). Replay stage stamps (720p -> 1440p synthetic): 421 us on LLPC against
    366-370 (WAVES 5: 390; 7 spills 8 B; rolled none 399; WGP mode 406). About -52 us, ~1.5% of vk-RRF-1440.
  - Lane B: the 8-bit surfaces differ by one LSB in <= 1% of values and kc by <= 0.016; the output settles at
    52.5 dB against the LLPC route over 8 frames, finite, same mean.
  Parked because the descriptors need VK_EXT_descriptor_buffer. The Vulkan frontend asks games only for cooperative
  matrix, float8, pipeline binary and maintenance5, so neither vulkan_dlss_harness nor a game has it. Asking every
  AMD Vulkan game for another device extension (and its feature bit, which Streamline-style games do not pass) is
  not worth 1.5% without a game check. An arena output plus a Vulkan store pass would cost about what it saves
  (~18 MB written and read). It becomes worth taking if the frontend ever gets descriptor_buffer for another reason.
- RR F's output passes on Vulkan (replay, 720p -> 1440p): post ~0.46 ms in three tile groups, down ~0.10, pooled
  0.045 (`D4R_ENGINE_RR_SKIP` ablations).
  - post on LLPC: 82 VGPRs, wave64, about 2700 instructions per thread (206 v_mul_f32, 158 v_add_f32, 154 v_fmac_f32,
    122 v_pk_fma_f16, 44 image_sample_lz, 52 branches). Over the ~2M threads that is VALU throughput for 0.46 ms.
  - `spirv-opt -O` variants (`<stage>_llpc.spv`) of post / down / pooled / downp: 542 -> 528 us (-0.4% of the case).
    Not kept.
  - post as wave32 (its shared scratch is sized for 32-wide subgroups): identical output but +20 us per group.
    Rejected.
- **RR F's dec0_1 on vkhip without descriptor_buffer: kept.** The parked patch above, with the descriptors coming
  from an ordinary descriptor set instead of vkGetDescriptorEXT. LLPC addresses a set's table as a user SGPR holding
  the low dword, with the high dword taken from `s_getpc_b64` sign-extended from 16 bits (seen in post's ISA). So:
  - A second donor, `donor_images.comp`, declares set 0 as four rgba8 storage images. AMD's driver puts that set's
    table in one more API user-data entry next to the push constant.
  - The HIP kernel enables the dispatch-id SGPRs (`llvm.amdgcn.dispatch.id`), and `palBinary` maps the table entry to
    s2 (`Kernel::table`: gfx12, kernarg + dispatch id, 4 user SGPRs). `glsl_hip.h` `glslImageTable` rebuilds the
    address as LLPC does. The kernarg goes back to 24 bytes.
  - The code-object parser accepts a bare gfx12 `s_getpc_b64; s_sext_i32_i16` with no add pair. LLVM bundles every
    PC-relative address, so a bare one cannot be a global's.
  - The F loader makes one set of `surfaceF` (GENERAL) and a layout of the set plus the 8-byte push, and binds it in
    recordStage. The engine rebinds its own set on every dispatch.
  Checks:
  - The replay output is byte-identical to the descriptor_buffer build, all 8 frames and every surface, so the
    driver's table puts binding k at 32 k bytes.
  - Against LLPC, the replay output is 64 dB on frame 0 and 53-56 dB after. Correction to the entry above: the
    recurrent 8-bit surfaces (rec / ka / kb) do diverge after frame 0, with 35-80% of values off by more than one
    LSB. The output does not drift.
  - Harness, candidate against control after 160 pan frames: 49 dB, finite, same means. After 120 still frames:
    59.4 dB. Candidate runs are deterministic (one hash over 6 runs).
  Timing, interleaved (bench `C:\d4r-hc\bench\d1set.json`, 3 rounds; rt-ve control, rt-vd candidate, same shim):
  - vk-RRF-1440: 3.418 -> 3.359 ms (-1.73%, spread 0.8%).
  - vk-RRF-4k: 7.516 -> 7.378 ms (-1.83%, spread 0.81%).
  - Replay: 3.39 -> 3.33 ms per frame.
  Housekeeping: C: ran out of space mid-replay (14 MB free; the climb's folder is 30 GB of 954 GB). A reference run
  wrote empty files. I deleted my own partial outputs and the rejected post-wave32 outputs (`tmp/rro_t`,
  `tmp/rro_l`, `tmp/rrout_rrp`) and reran.
- Vulkan RR F per stage after ffde45b (replay stamps, 720p -> 1440p synthetic, us). A temporary stamp after the
  gather (not committed) split the first column:
  - exposure + gather 255, enc0 (HIP) 430, enc1 234, enc2 115, enc3 59, enc4 39, enc5 29.
  - dec4 44 + 35, dec3 54 + 48, dec2 113 + 102, dec1 200 + 181, dec0_0 353, dec0_1 (HIP) 367.
  - post in three groups 153 / 195 / 195, pooled 45.
  post is per-pixel f32 filtering (5-tap history and the like), and its MMA helpers are not called from main, so
  `D4R_MMA_PACKED` does not apply. The RR F numerics are already at the most relaxed level (D4R_RR_RELAXED=3).
- **Tried again, rejected (lane B): the gather pass (enc0g) on LLPC with one FP8 conversion per pair and no NaN
  branch.** Under `RR_LLPC_SAT`, ptx2rr.py's `f16x2_to_e4m3x2` clamps the pair to +-448 and converts it once (a NaN
  half gives +-448, or 0 with relu, instead of 0x7f). compile_rr2.py writes `enc0g_llpc.spv` with it:
  - Every other file the build writes is byte-identical. The preprocessed texts the D3D12 build and the HIP hybrids
    read are unchanged.
  - LLPC: 8996 -> 7682 instructions, 139 -> 47 mode writes, 322 -> 232 branches, 86 -> 95 VGPRs.
  - The remaining mode writes are FP16_OVFL=0 before every plain conversion. LLPC never sets the bit here, and GLSL
    cannot hoist them.
  - Replay: 3.34 -> 3.28 ms, byte-identical.
  - Bench (rt-vd against rt-vg2, 3 rounds): vk-RRF-1440 3.353 -> 3.310 (-1.27%, spread 0.81%), vk-RRF-4k
    7.375 -> 7.300 (-1.02%, spread 0.76%), identical. Every candidate run was below every control run, but under
    2x the spread on both cases. Reverted.
  Housekeeping: `models-rrg` / `models-fullg` already existed from the first try, so this one used `models-rrg2`,
  `models-fullg2` and `rt-vg2`. A failed `ln` left two links inside `models-rrg/rr-f` (`hip/hip`,
  `native-work/native-work`); I removed them.
- **Bound: RR F's HIP stages without barriers between them** (timing only, wrong output; local hack, not committed).
  With the 15 `step()` barriers between the 16 vkhip stages gone (the gather's and post's kept), the replay goes
  3.33 -> 2.98 ms (network 2.72 -> 2.34). That is the largest lever left on vk-RRF, but only an upper bound:
  dependent stages overlap there. K's per-dispatch counters got nothing from a similar bound (above), so the real
  thing has to be per-block dataflow like M's `SWIN_DATAFLOW`. Unlike M, RR F needs:
  - Write-after-read waits. The arena regions are reused within a frame (rr.txt pointers: region 1024 is written by
    enc2, dec1_0 and dec0_0 and read in between). A conservative rule: stage N waits until stage N-2 has fully
    completed (a counter), which by induction completes every earlier stage. It also waits for its stage N-1
    producer blocks (flags), and for all of stage N-1 wherever N writes a region N-1 reads.
  - Mirror-aware producer ranges. Edge windows read tokens mirrored up to 6 inward, outside M's [t0, t1] range. With
    M's formula, a block at x0 = -6 would not wait for the producer of token 6.
  - The level mapping per stage: merge for enc1-enc5 (KIND 1), expand plus a skip for dec*_0 (KIND 2), the same grid
    for dec*_1. The skip comes from stage <= N-2, so it is already complete.
  - Waits and signals in both kernel families, rr2_native_body.h and the glsl2hip hybrids (enc0, dec0_1).
  - Flags cleared every evaluation, as vkhip_net's M chain does.
  Not started: race-prone work that needs its own validation (repeated runs, byte-identical against the barrier
  build).

### 2026-11-05 (afternoon): d3dhip only (owner: focus on d3dhip K, M, L, RR F; numerics may change)

- Current d3dhip profiles, harness 1280x720 -> 2560x1440 (`D4R_PROFILE=1`, rt-cand14): K 1.36 ms (input 0.09,
  network + handoff 0.76, output 0.47-0.53); M 1.61 (input 0.11, network 0.97-1.01, expansion 0.08, reconstruction
  0.30-0.36, downsample 0.10); L 2.27-2.47 (input 0.49-0.54, network 0.93-0.96, expansion 0.35-0.42, reconstruction
  0.37-0.44, downsample 0.10); RR F on DXIL 4.0 (gather + enc0 0.82, network ~2.1, dec0 0.93, output 0.63).
- **Why the first d3dhip RR F prototype lost**: REngine's external-network front (runtime_rr2_d3d12.inc
  `recordFront`) dispatched every network stage in DXIL as well (`dispatchStages(0, R2Network)`), so the frame ran
  the network twice. Up to `R2Dec0B` instead: replay front 1.25 -> 0.78-0.82 ms, frame 3.72-3.80 against DXIL
  3.86-4.01 (74ae648, which also adds the per-parity HIP graphs and the shared-fence path). A shared arena alone
  (`D3D12_HEAP_FLAG_SHARED`) did not slow the DXIL stages.
- **Kept (lane B): RR F on d3dhip in the shim** (947afda): dx-RRF-1440 -6.11%, dx-RRF-4k -7.17% (numbers in the commit).
  Split per part (D4R_PROFILE): front ~0.95, external_network_wait ~2.0, back ~1.2 ms at 1440p (the profiler's own
  marks add a little). The harness has no D3D12 debug-layer switch; the smoke ran without it.
- GPU sharing between agents: `C:\d4r-hc\tools\gpulock.sh` (a directory lock) wraps every timed run;
  `C:\d4r-hc\tools\ab.sh` runs C X X C rounds of one.sh cases under it. Helper agents (DeepSeek V4.1 Flash, own
  worktrees, briefed through `C:\d4r-hc\AGENTS-BRIEF.md`) work on: RR F's enc0 inside the HIP section, L's two plane
  stages' DXIL, and K's output / M's reconstruction DXIL.
- The working tree still has another session's K input copies in game_d3d12.cpp (on by default, D4R_K_DIRECT_INPUTS=1
  turns them off). Shims built from this tree carry them, so K timings from such a shim are not comparable with
  committed builds.
- **Tried, rejected: a GPU-side d3dhip handoff (memory words instead of the shared fence)**, RR replay prototype
  (reverted; kernels and shader in `C:\d4r-hc\tmp\spin`). An empty HIP section (wait + signal, no network) costs
  ~140-157 us between the D3D12 front's end and the back's start; that is the handoff every d3dhip preset pays.
  - D3D12 -> HIP: the front ends with `WriteBufferImmediate` (MARKER_OUT) of a frame number into a shared buffer; a
    one-wave HIP kernel polls it with system-scope acquire loads. Works (HIP sees CP writes; output byte-identical),
    empty gap 140 -> 106 us, but with the real network no end-to-end gain (3 rounds: 3.716 against 3.714 ms).
  - HIP -> D3D12: a HIP kernel stores the frame number with a system-scope release; the back's first dispatch (one
    thread, `globallycoherent` atomic polls, a bounded loop) waits for it. It never sees the word before its ~100 ms
    limit, with the word in the shared device buffer and also with it in host memory (`hipHostRegister` +
    `OpenExistingHeapFromAddress`); the HIP work then runs once the D3D12 dispatch has ended. So HIP's queue does not
    run while the D3D12 queue executes that dispatch: the contexts are not scheduled concurrently here (HAGS is
    block-listed on this machine), and part of the handoff is that context switch. With scheduling settings off
    limits there is no lever left in the handoff mechanism; only fewer handoffs (none possible below two) or less
    work across them.
- **Bound: RR F's d3dhip network without barriers** (timing only, wrong output; local hack, reverted): the fourteen
  stages launched directly with `hipExtAnyOrderLaunch` instead of the graph, replay 720p -> 1440p, two rounds: network
  gap 1.80 -> 1.57 ms, frame 3.72 -> 3.49 (-6%). As on vkhip, only per-block dataflow (M's `SWIN_DATAFLOW` style, with
  the write-after-read and mirror-range rules listed above) could take part of it.
- d3dhip profiles at 1920x1080 -> 3840x2160 (rt-rrh, `D4R_K_DIRECT_INPUTS=1`): K 2.69 ms (input 0.20, network 1.33,
  output 1.11); M 3.39 (input 0.26, network 1.86, expansion 0.19, reconstruction 0.76, downsample 0.23); L 5.27 (input
  1.22, network 1.92, expansion 0.89, reconstruction 0.95, downsample 0.23); RR F 8.39 (front 2.05, network 3.66, back
  2.63). Against NVIDIA's M stages at that size (post 0.836, downsample 0.254, Swin 1.078) M's whole gap is the
  network (~1.72 ms without the handoff). Two more helper agents: M's Swin HIP layers, RR F's D3D12 back stages.
- **Kept: RR F's encoder 0 in the d3dhip HIP section** (helper agent-rrenc0's patch): byte-identical, dx-RRF-1440
  -1.29%, dx-RRF-4k -1.82% in the harness (the replay showed -2.7%). Only packages with `hip/rr2_enc0.hsaco` take it
  (models-rre/rr-f here; models/rr-f lacks it). The replay's `hip N` figure for preset F is not meaningful (HIP events
  around the graph launch); use the D3D12 front / network / back split.
- **Kept: L's two d3dhip plane stages with two windows per wave** (a953312, helper agent-lplanes): `-DWPW=2` on
  `enc0_l_planes.dxil` / `dec0_l_planes.dxil` only; dx-L-1440 2.354 -> 2.317 ms, dx-L-4k 5.237 -> 5.154 (-1.59%
  each, tools/ab.sh 3 rounds), byte-identical. Per stage (D4R_PROFILE): input 490 -> 469 us, expansion 374 -> 348 at
  720p. Both stages are issue-bound (RGA: 1947 / 2555 instructions a wave, no spills; the FP8 packs need f32
  sources on gfx12, ~400 conversion instructions a wave). Rejected by the agent, all slower or equal: -O1, M_ACC32,
  PF/PFN prefetch, f32 embedding, binary16 MLP residual, MLP weight tiles staged in LDS (slower), removing the four
  group barriers (free already). A constant-address probe was 35% faster only because the compiler deleted half the
  program. Runtime with this package: `C:\d4r-hc\rt-lx` (rt-rrh plus the two files); rt-rrh stays the older control.
- **Tried, nothing kept: RR F's D3D12 back on d3dhip** (helper agent-rrback; report `C:\d4r-hc\agents\agent-rrback\report.md`).
  Per-pass timestamps of the back (720p / 1080p input): dec0_1 425 / 942 us, post 497 / 1157, down (12 / 16 tile
  passes) 135 / 323, pooled 11 / 22; back 1067 / 2445 us. RGA against LLPC: post 114 vs 82 VGPRs, 211 scalar
  `v_fma_f16` where LLPC has 122 `v_pk_fma_f16`, ~40% more ALU delay padding; dec0_1 192 VGPRs, 5426 instructions,
  twice its HIP sibling. Measured and rejected: post and dec0_1 at -O3 (neutral), post wave32 (497 -> 516 us,
  slower, as on Vulkan), dec0_1 wave64 (output 6.7 dB, wrong), other tile geometries, dispatch fixed cost (~0.9 us each, 22
  us in all), inter-group UAV barriers (~10 us). Post is ~84% issue-bound; removing its loads made it slower. The
  lever left is dec0_1 in HIP (it is one on vkhip, ffde45b), handed to agent-rrdec01.
- **Kept: K's D3D12 output stage with fewer instructions** (1f8d128, from helper agent-kout's patch, report
  `C:\d4r-hc\agents\agent-kout\REPORT.md`). dx-K-1440 1.3615 -> 1.2615 ms (-7.34%), dx-K-4k 2.759 -> 2.537 (-8.05%);
  lane B (motion scene 43 dB after 200 frames, quality-scene truth metrics equal, game captures 65-68 dB). The stage
  is VALU-issue-bound: 57600 waves x ~3200 instructions / 256 SIMDs matches the measured 0.49 ms. Ablations (wrong
  output): 12 -> 4 filter taps -27.5%, history Catmull-Rom -> bilinear -23%, re-exposure -2.6%; LDS banking and the
  transcendentals are not the limit. AMD's D3D12 compiler emits no packed f16 or VOPD at SM 6.6-6.9 (probe shaders):
  that is the whole Vulkan 0.37 against D3D12 0.49 ms gap. Rejected: wave32 (+8.7%), a separate luma LDS plane (+10%),
  512-thread groups. What I changed in review:
  - **Build flags**: compile_k.py builds output_k with `-DD4R_NATIVE_HALF=1` too (the `rounding` list in
    `ldr_define`). My first rebuilds lacked it, and every comparison of them with the agent's build came out ~3% slower
    than they were; three guard variants were wrongly rejected that way before I found it. Rebuild with all of
    compile_k's flags (`hlsl_build.py output_k.comp OUT -DD4R_NATIVE_HALF=1 -DD4R_FILTER_ACCUM_F32=1
    -DD4R_FILTER_F32=1 --wave 64`, D4R_DXC = the runtime's dxcoob 1.9.2609.5).
  - **The f32 filter can still underflow**: the agent dropped the running maximum because "f32 sums don't
    underflow", but exp2 below ~-126 is 0 and 0/0 zeroes the pixel through the finiteness check (the black-speckle
    mechanism). The guard sits in that check's cold branch (the sub-pixel takes the current bilinear luma).
  - **The "bright" quad exchange stays in shared memory**: the agent's shuffles were byte-identical only because
    neither the harness scenes nor five game captures (kcap 12739, 13181, 15078, 15518, 15955) have a quad of mixed
    brightness. Removing the exchange entirely is byte-identical on all of them too. `diff -rq` over the replay's
    output folders shows a difference in every case, from timings.csv; compare the .bin files only.
  M's reconstruction (post_m, 10x10, 1889 instructions, 88 VGPRs, 0.309 ms) is also issue-bound. Only algorithm
  changes (bilinear history, fewer taps) moved it; nothing kept (`M_AMD_POST_MATH` -1.3%, under the bar).
  The K target (0.649 ms) cannot come from this stage: input 0.09 + network ~0.73 is already 0.82 ms.
- **Kept: M/L's HIP Swin layers with paired weight loads** (helper agent-mswin, report
  `C:\d4r-hc\agents\agent-mswin\REPORT.md`; `SWIN_TOK_WPAIR` + `-mcumode` in kernels/windows-flags.txt). dx-M
  -2.01% / -1.77%, dx-L -1.55% / -1.09% (1440p / 4K), byte-identical. Per layer at 1080p input (us): enc1 425, enc2
  250, tube x6 604, dec2 228, dec1 431; sum 1937 at 129 TFLOPS (33% of the 390 TFLOPS FP8 peak). Roofline on this
  card (microbenchmarks `peakfp8.hip`, `wload.hip` there): WMMA alone 283-310 TFLOPS; with one 8-byte global load
  per WMMA 153; one 16-byte load per two WMMAs 200; LDS-staged 182. So the layers sit on the load-instruction
  ceiling: VGPR caps +16% / +73% (scratch), dropping the FP8 clamps nothing, GELU removed -5%, all operands from one
  address -20% (the load instruction, not the cache, costs). `SWIN_TOK_BUFFER_WEIGHTS` no gain.
  `SWIN_TOK_NO_LOAD_SYNC` is a trap: -20% end to end but 21 dB wrong after 100 frames, while a single frame's head
  can still look identical (the asm batch barrier keeps WMMAs below their loads). NVIDIA's ten layers run at ~226
  TFLOPS (1.078 ms). The next step is two windows per wave (N = 32 per weight operand): halves the loads again,
  +~64 VGPRs, occupancy 9 -> 6, a rewrite of swin_tok_core's token ownership. In review I reverted the patch's change
  to the `SWIN_TOK_WIDE_WEIGHTS` (gfx11 token) branch, which read operands that path never loads; L uses the same
  code objects and was measured here too. Package hip/ kernels = `kernels/build.sh m` with
  `D4R_GPU_ARCH=gfx1201 D4R_VKHIP_LAYERS=1 D4R_NATIVE_FP8=1` (ISA-identical to the shipped ones); hsaco files embed
  their source path, so compare `llvm-objdump -d` output, not bytes.
- **Tried, rejected: RR F's d3dhip network as a per-block dataflow chain** (helper agent-rrflow; report and patch in
  `C:\d4r-hc\agents\agent-rrflow`). Chained stages launched with `hipExtAnyOrderLaunch`; each block waits for the
  producer blocks of its own input rectangle (one 32-byte flag per block, a frame-generation word, so no clears),
  with the plan derived per frame from the stages' arena pointer words and geometry. A stage joins the chain only
  if it writes nothing its predecessor (or the chain's first stage) reads or writes, since regions are reused
  within a frame (region 1024: enc2, dec1_0, dec0_0). Mirror-aware producer ranges; at most three stages a chain.
  20 of 20 runs (8 frames x 15 repeats) byte-identical to the graph build. But: replay network 2211 -> 2191 us
  (-0.9%), harness -0.08%. The no-wait bound (wrong output) is -11.5% of the network, yet the isolated stages
  already sum to 2090 of ~2230 us: all fourteen barriers cost only ~110-180 us, and most of the bound came from
  overlapping stages that really depend on each other. Encoder chains: no gain, and about 1 run in 20 stalled
  9.5 ms (a 3807-block producer still has blocks unscheduled while consumer blocks spin in their slots; M's grids
  fit one dispatch, so M never hits this). A single fifteen-stage chain with an N-2 rule: no gain either.
  The agent also reported the graph build's output changing with HIP's allocation layout ("a stage reads past a
  buffer end"). On the committed code that does not reproduce: 1 MB blocks of 0xFF before every F weight buffer
  and after the parameter area leave all 8 replay frames byte-identical. The agent most likely compared different
  code objects (its RR2_FLOW build against one without it).
- **Tried, rejected: L's input stage block as HIP on d3dhip.** L's D3D12 `enc0_l_planes` reads only buffers
  (the feature slots `featsL`, per 8x8 block 64 pixels x 32 FP8 bytes; weights) and writes the skip (channel-linear,
  token * 32 + channel) and the network's tokens in NGX's planes, so HIP can take it; the expansion
  (`dec0_l_planes`) writes four textures (feat, gates, mix, cov), which HIP cannot. ZLUDA's native tail
  (kernels/tex/enc0l_tail.hip: embedding, the C32 Swin block, skip, patch merge) has exactly those outputs and
  layouts. Standalone (C:\d4r-hc\tmp\lhip: a loader from the featsL layout into the tail's shared-memory layout, then
  `d4r_enc0l_tail`; random weights) on the RX 9070 XT: 416 us at the 1280x736 token grid, 921 us at 1920x1088 (188
  VGPRs, occupancy 8, no scratch; four waves 433 / 960, -mcumode 412 / 912). I first set that against the profile's
  whole input stage (469 / 1160 us) and expected -2.4% / -4.8%. That was the wrong comparison: the gather pass
  (enc0_m.comp L_FEATURES) stays on D3D12 either way, and the block alone is ~0.35 ms at 720p.
  Built anyway, to measure it: the kernel packaged with NGX's enc0 weights, HipNetworkDx owning a shared featsL / skipL
  and launching it ahead of the graph, MEngine skipping its dispatch (`D4R_HIP_L_ENC0`; patch and sources in
  `C:\d4r-hc\tmp\lhip\route`). In the harness (L 720p -> 1440p, D4R_PROFILE), the input stage went from ~0.50 to
  0.14 ms, but the network wait grew from ~0.87 to ~1.37 ms, and the frame was 2.373 against 2.273 ms. The output is
  finite but 31.5 dB from the D3D12 stage after 200 frames, unexplained. Reverted. D3D12's plane stage (WPW=2, RGA 1947
  instructions a wave) beats ZLUDA's NVIDIA-order tail here.
- **Tried, below the bar: RR F's d3dhip HIP stages with paired weight loads** (helper agent-rrswin; notes, patch and
  profiling hook in `C:\d4r-hc\agents\agent-rrswin`). Isolated stage times at 720p input (us): enc0 460, dec0_0 372,
  enc1 240, dec1_0 213, dec1_1 187, enc2 121, dec2_0 115, dec2_1 104, enc3 61, dec3_0 55, dec3_1 49, dec4_0 48,
  enc4 42, dec4_1 37, enc5 29 (2123 in all; network 2280-2330). Unlike M's Swin layers these are not at the
  load-instruction ceiling: with the weight loads removed the stages gain only 12% (M: 20%), and doubling the WMMA
  count costs 30%, so the matrix pipe is about a third loaded. Each window's serial K32 accumulator chain sets the
  time, with little independent work next to it (dec0_0: one wave per window, ~4500 instructions for 272 WMMAs; the
  small stages have 20-70 windows for 64 CUs). More waves or more accumulators would be the lever, but the kernels
  are at 158-256 VGPRs (dec4_0 at 256 with 24 KiB of LDS). Built `RR2_WPAIR` (M's paired layout: in enc2,
  `global_load_b64` 105 -> 9), byte-identical: dx-RRF-1440 3.789 -> 3.767 ms (-0.59%), dx-RRF-4k 8.247 -> 8.215
  (-0.38%), network -1.2%. `-mno-cumode` on enc4 / enc5 / dec4_0 / dec4_1 is -5 to -14% a stage, ~-0.3% of the frame.
  Both are under the 1% bar, alone and together; not kept. The 14 translated stages build from preprocessed handoffs
  (`<stage>_m.preprocessed.comp`), so edits to rr2_m.comp only reach them after re-preprocessing; dec0_0 is
  rr2_w.comp. The packaged rr-f has no rr2_dec0_1.hsaco.
- **Decided against: two windows per wave in M/L's HIP Swin core.** agent-mswin2's ceiling probe
  (`C:\d4r-hc\agents\agent-mswin2\pair.hip`, rerun here): one 16-byte weight load per 2 FP8 WMMAs 204-213 TFLOPS,
  per 4 WMMAs (two windows) 258-266, no loads 339-355. The layers run at 120-146 TFLOPS, ~35% below even the current
  ceiling because of token-side VALU (norms, FP8 packs, softmax, GELU), and that work doubles with the WMMAs. A
  window's activations cost +~64 VGPRs: the C 96 / 128 layers (enc2 174, tube 163, dec2 235 VGPRs) cannot take that
  without scratch, and the C 64 layers (enc1 145, dec1 133) gained only 11-13% with no weight loads at all. Expected:
  about -40 us of M at 4K (~1%) for a rewrite of the core's token ownership and its patch merge / expand. Not done.
- **Housekeeping: a stray K enc0 code object in five runtimes.** A helper (agent-knet) wrote an experimental
  `dltss_pwin_enc0_layer.hsaco` through a hardlink into rt-lx from 16:35 to 17:20 on 2026-11-05. It restored rt-lx
  itself, but rt-kc, rt-ke, rt-km, rt-l0 and rt-rrswin had been copied in that window and kept the experimental file
  (disassembly 387565c6 against the original 69674d49). Restored from rt-lx (md5 892b4f8a). The M, L and RR F
  numbers do not involve that kernel. K's output-stage A/B (1f8d128) ran with the same file on both sides, so its
  delta holds, but its absolute K times came from that build.
- **Not committed: agent-knet's `-DPOS_VSHARE1` for gfx1201 K enc0 / dec0** (`C:\d4r-hc\agents\agent-knet\knet.patch`,
  kernels/windows-flags.txt only). Byte-identical (every wave computed V^T for all four key tiles; now each wave
  computes its own into LDS), spills down (enc0 108 -> 44 B). Network -1.6%; dx-K-1440 -0.89 / -1.01 / -1.66%,
  dx-K-4k -1.35 / -0.83 / -0.35% over three sessions. Borderline against the 1% bar and not yet re-measured here.
  The climb stopped at this point, on the user's call.
- **Audit of the helper agents' side effects (end of the climb).** No agent committed (no t3/* branch has commits
  outside hillclimb/perf); their edits stayed in their worktrees; no stray processes or stale GPU lock. Shared inputs
  untouched (model folders, `C:\d4r-hc\kernels\m-vkhip-gfx1201`, kcap). rt-lx was still hardlinked to agent package
  trees (1112 files), so I replaced it with a real copy (content unchanged: rt-rrh plus the two L plane stages,
  original K enc0). Linux / gfx11 builds re-checked against the parents of the agent-derived commits: M/L kernels for
  gfx1101 (fast and accuracy sets) are ISA-identical across 0feb1a0, and output_k.comp's SPIR-V (three flag sets)
  is byte-identical across 1f8d128.

## RDNA3 parity pass (RX 7700 XT gfx1101, 26.10.1, 2026-11-10; uncommitted worktree off origin/windows fcafd9a)

Environment: Ryzen 9 5900XT, Windows 11 Pro 26300, HAGS off (HwSchMode 1). Bench runtimes in `C:\d4r-rdna3-hc`
(rt-control = origin/windows shim + the release packages installed in Ghost of Tsushima; rt-final = all patches with
packages rebuilt from the worktree). Kernels with TheRock clang 24 (`.tools/therock-10.2.0a20260929`, the compiler of
the installed hsaco), shaders with the Vulkan SDK 1.4.363 dxc.

- **Kept: RR D/E on D3D12 for RDNA3** (patch 01). AGS layout probe first (agsprobe: A/B lane = row/column l & 15, element
  i = k i; accumulator element i = row 2i + (l >> 4); 255-256/256 exact with a seeded C). rr_hlsl.translate gfx11
  mode, compile_rr_dxil.py gfx11 build, runtime accepts `arithmetic gfx11` with the sidecar's line, shim loads
  engine\rr-d-gfx11 / rr-e-gfx11 on RDNA3. Exact build: replay 32.1-33.4 ms at 1024x576 -> 2048x1152 (Vulkan 30.7-32.0);
  relaxed (relaxed_math as gfx12): 27.2-27.5 ms, kept (61-67 dB against Vulkan either way). Harness 720p -> 1440p:
  39.6 / 39.4 ms (D / E). First launch compiles for ~47 s at NGX init, 0.6 s from the driver cache after.
- **Kept: RR output formats** (patch 02, issue 65). Reproduced: RGB9E5 output -> "refused: RR output must be RGBA16F" on
  the pre-patch shim; after, D and E run into RGB9E5 / RGB10A2 / R11G11B10 outputs (AMD reports typed UAV stores for
  all of them on RDNA3).
- **Kept: gfx1101 K layers without scratch** (patch 03). The released gfx1101 enc2 spills, a fresh clang 24 build's dec1
  spills 96 B; vkhip refuses spilling gfx11 kernels, so native Vulkan games on RDNA3 ran K's Vulkan network (9.7 ms with
  the released package). dec1 on its generic line has no scratch.
- **Kept with a caveat: d3dhip K default on RDNA3 from 26.10.1** (patch 04). Ghost's Quality size (1706x960) showed
  per-process bimodal d3dhip times in one session (2.40-2.52 or 2.88-3.09 ms, both shims alike, outputs identical);
  an earlier session had 2.39-2.41 steady. 1440p and 4K stable.
- **Found, not fixed: preset L on RDNA3 (DXIL) is nondeterministic.** Same shim, package and inputs: frames 1-12
  byte-identical across runs, from 25 frames on different (60 frames: 36.6-38.7 dB between runs, 2.9 M of 3.7 M pixels),
  with or without 200 ms gaps between frames: an intra-frame race that the history spreads. M (shared Swin layers) is
  identical over 160 frames, so the candidates are L's own RDNA3 stages (enc0_l / l_patch_dec0 / dec0_l / merge_l).
- **Measured, not changed: RDNA3 already gets K's faster D3D12 output stage and the f32 reconstruction** with a rebuilt
  package; together they are time-neutral on RDNA3 (dx-K-q1440 DXIL 2.594 -> 2.611).
- **Rejected:** the full gfx1101 flag sweep of windows-k-d3dhip.md as is (spills with clang 24; d3dhip within +-1.3%);
  ROCm 7.2 clang for the K layers (inline asm "operands are not valid for this GPU").

### RDNA3 pass, second session (2026-11-10)

- **Kept: preset L's RDNA3 reconstruction with a 12x12 tile** (`m_d3d12_build.py`, patch 06). The L nondeterminism was
  localized with per-stage captures (D4R_M_CAPTURE_DIR, every frame): input, network, head, gates, mix and covariance
  identical across runs; the first difference is `packed` (post_l), a few pixels of one 16x16 block on odd rows of one
  wave. post_l at 11x11: diverges within 25-160 frames (also built from current source); 12x12, wave64 and unpacked
  tiles: 3 x 160 frames identical. 11x11 is a texel short for L's -1..+2 taps now and then, and the read runs into shared
  memory another wave writes later. Time: dx-L-1440 +0.18%, dx-L-4k -0.25% (noise). The RDNA4 (FP8) build's post_l is
  11x11 too but wave64, which did not reproduce here; check it there.
- **Kept: the release builder makes rr-d-gfx11 / rr-e-gfx11 on RDNA3** (packaging, patch 07, approved by the user).
  Checked end to end with the kit's build-engine-models.py on this machine: rr-d, rr-d-gfx11, rr-e, rr-e-gfx11 in 3.5
  minutes; the D3D12 harness loads the installer-built rr-d-gfx11 and renders.
- **Game check, Ghost of Tsushima Quality (1712x960 -> 2560x1440, LDR), main menu, D4R_PROFILE gpu_eval, A B B A, 60 s
  each:** DXIL 3.241 / 3.238 ms, d3dhip (the new default) 2.529 / 2.527 ms: -22%. The shim in this install is
  d4r\d4r_dlssg_next.dll (the DLSS-G host's next core), not d4r\nvngx.dll; both were swapped and restored (hashes).
  The game's two start-up dialogs (no GPU / old driver: OptiScaler's spoofed adapter) were answered by posting the
  dialog's OK button (WM_COMMAND), without focus.
- **Not viable here: vkhip M / L on RDNA3 (AMD's Windows driver).** The gfx1101 dataflow Swin layers spill (enc1 152 B,
  enc2 24 B, dec2 188 B at the default bound; with SWIN_WAVES_PER_EU=1 only enc2 still spills, 8-68 B in every flag
  variant tried), and vkhip refuses spilling gfx11 kernels.
- **d3dhip M / L on RDNA3: blocked on non-FP8 plane stages.** The windows branch's enc0_m / dec0_m refuse DOT2 with
  PLANES (#error); the Linux branch has WIDE plane stages (FP8 values in f16 cooperative matrices) that the windows
  sources (~750-800 lines apart) lack. Needs: the WIDE port, their DXIL through hlsl_build's AGS path, an external-
  network mode in runtime_m_rdna3.inc, the shim branch. The gfx1101 layers (`D4R_VKHIP_LAYERS=1 kernels/build.sh m`)
  and an M package with hipnet.bin (`compile_m.py --hip`) build.
- **d3dhip M on RDNA3, how far it got.** Linux's WIDE plane stages (ef0075f) three-way-merge into the windows enc0_m /
  dec0_m (two small conflicts), but the merge also changes the existing FP8 variants' SPIR-V (enc0_m, dec0_m, both
  plane layouts), so it would alter RDNA4 output: kept out of the tree (a copy in the test machine's scratch). In a
  separate directory the WIDE stages build to DXIL, but under D4R_HLSL both shaders take the hand-written AGS branch
  (d3d12/enc0_m_wmma.hlsli and dec0's equivalent), which is FP8 only: the preprocessed main still calls mma8 / fp8Of.
  WIDE exists only in the GLSL cooperative-matrix branch, which the D3D12 build bypasses; making it compile through
  coopmat_ags.hlsli (as layer.comp does) means rewriting its buffer / shared-memory accesses to that header's
  wrappers. Remaining: that rewrite (or WIDE versions of the AGS branches), an external-network mode in
  runtime_m_rdna3.inc, the shim branch, validation. Over a day; stopped here.
- **Kept: d3dhip M and L on RDNA3 (patch 08), by default.** The RDNA 3 engine keeps its own input / expansion stages
  and converts its token-major `tokens` / `dec1` to and from the HIP network's plane layout (m_planes.hlsl, a byte
  permutation), so no FP8 shader is touched; the shim reuses the RDNA4 d3dhip split and GPU-fence handoff. Harness,
  3 rounds, same package: dx-M-1440 8.327 -> 5.527 (-33.6%), dx-M-4k 16.861 -> 12.596 (-25.3%), dx-L-1440 11.356 ->
  8.914 (-21.5%), dx-L-4k 23.494 -> 20.004 (-14.9%); quality-scene truth metrics equal (lane B). Ghost of Tsushima,
  1712x960 -> 1440p LDR, gpu_eval, A B B A: M 13.16 / 13.32 -> 8.76 / 8.75 ms, L 19.28 -> 15.09 ms (one pair); the
  game's files restored by hash afterwards. Release: package_release_windows.sh builds kernels/<gfx110x>-swin-hip and
  build-engine-models.py passes it to compile_m.py --hip (checked end to end with the kit builder). Vulkan M on RDNA3
  unchanged (staging; the layers spill for vkhip).
- **Measured, not kept: the Swin HIP layers with SWIN_WAVES_PER_EU=1** (fewer spills): byte-identical, dx-M-1440
  -3.25%, dx-M-4k -0.82% (under the bar); it would also change the Windows CUDA route's kernel builds.
- **Rejected: faster RR D on RDNA3 by relaxing the gfx11 DXIL further** (replay, 1024x576 -> 2048x1152, 5 repeats,
  relaxed build 25.2-27.9 ms): no binary16 rounding after each k32 step 27.1-27.8 ms (no gain; 60.7-64.9 dB against
  the exact Vulkan route, so harmless but useless); attention-head loops unrolled 26.1-30.4 ms (slower; enc1 2.4 ->
  3.6 ms). The cost is the exact gfx11 FP8 emulation itself: every 16x16 operand is widened from FP8 codes per load
  (four g11_sc4 per lane) and each k32 step is two f16 WMMAs with scaling. The next lever is keeping the activations
  widened (binary16) in LDS instead of FP8 codes, a rewrite of the RR_GFX11 operand path (rr_gfx11.glsl and the stage
  bodies): new kernel work, not started.
- **Kept: cheaper FP8 operand widening in the RDNA3 RR stages (patch 09, rr_gfx11.glsl g11_sc4), byte-identical.**
  Bound first (replay, 1024x576 -> 2048x1152, wrong output, timing only): widening and FP8 encode both free 26.5 ->
  14.3 ms (-46%); the encode alone free: no change; the widening alone free: ~17.7 ms (-33%). The widening now handles
  two codes per operation (bytes spread, then both halves widened at once), equal to the old form for every pair of
  codes (exhaustive check). bench.py, 3 rounds: dx-RRD-1440 40.149 -> 35.663 (-11.2%), dx-RRE-1440 40.018 -> 35.681
  (-10.8%), vk-RRD-1440 28.038 -> 27.316 (-2.6%), vk-RRE-1440 27.348 -> 26.770 (-2.1%), every output identical.
  Remaining (about another 20% by the bound): keep activations widened in LDS so each is widened once, not at every
  reuse; a rewrite of the RR_GFX11 staging in swin_m / dec0_m / enc0_m_tail / rr_attn. Not started.

## 2026-11-10: approximations on RDNA3 D3D12 (against the linux branch's docs/approximations.md)

Goal: the fast default kernels without approximations (the linux page's definition: a narrower format than NVIDIA's
at a step, or a different function), keeping the time saved. Full write-up: docs/approximations-windows-rdna3.md.

- **Found: M's RDNA3 expansion stage was int8** (m_d3d12_build.py built dec0_m.comp DOT2 INT8; runtime widenM
  stageInt8): tokens and matrix re-quantized to int8 with scales, a real approximation. Fixed in 10: dec0_m.comp
  D4R_M_EXPAND_F16, the product on v_wmma_f32_16x16x16_f16 with FP8 widened exactly and NVIDIA's binary16 rounding per
  32 products; weights column-major binary16 (runtime, D4RML007 only). Expansion 0.329 -> 0.271 ms at 1440p.
  Rejected first: DOT2 without INT8 (+21% dx-M-1440).
- **Found: AMD's D3D12 compiler converts f32 -> binary16 toward zero.** Probe (cvtprobe, RX 7700 XT, 26.10.1): f32tof16,
  (float16_t) casts and vector casts equal round-toward-zero on 2^20 of 2^20 inputs; binary16 add / mul / mad round to
  nearest (mad is one rounding: 100% equal to the correctly rounded fma). The RDNA3 M/L stages, K's D3D12 stages and
  RDNA4's M reconstruction use the native conversion (D4R_NATIVE_HALF).
- **Found: dialect_hlsl.h's emulated rounding is wrong for binary16 subnormals** on AMD's D3D12 compiler, which folds
  `precise float subnormal = (v + 0.75) - 0.75` (half of the inputs under 2^-14 truncated). D4R_RNE16_EXACT (round()
  for subnormals, overflow to infinity): 100% on the 2^20 inputs. Emulated in every RDNA3 stage it cost +7% (M) / +5%
  (L), rejected; and in L's enc0_l_network / dec0_l it gave 26 dB against NVIDIA (bisected per shader), not understood.
- **Kept: AGS's f32 -> f16 matrix conversion rounds to nearest even** (2^20 of 2^20, uniform, divergent branch and
  divergent loop alike); gfx11's f16 accumulator holds element r in the low half of register r (probe). rne16_ags.hlsli
  gives every RDNA3 M/L shader that rounding (pipelines bind the AGS UAV in D4RML007 packages). Check: a debug
  reconstruction computing both AGS's and the exact emulation's result on every conversion found no disagreement and
  gave the AGS build's frames byte for byte; enc0 / dec0 / down / exposure built either way are byte-identical. The
  emulated reconstruction alone differs (41 dB from AGS's), cause not found.
- **Kept: f32 output stores (down_m.comp WIDE_STORE, from the linux branch) and RGBA8 / BGRA8 output** (refused before:
  the LDR scene's RGBA8 output could not run).
- **Fixed, latent: rr_wmma.hlsli's AGS pair rounding** read gfx12's layout; on gfx11 the second value came back 0
  (probe). No RR package's DXIL changes (the calls were dead code).
- **RR D/E**: relaxed_math on gfx11 reduces to fma16 -> mad (exact on this driver, above) and hkpn without precise
  (rounding change). Exact and relaxed D3D12 builds are both 61.68 dB from the exact Vulkan build (RR D replay);
  stage dumps put the first difference at encoder 0's gate sig() (1/x and exp2 lowered differently by the two
  compilers; NVIDIA uses rcp.approx / ex2.approx there). Not an approximation; left.
- **Open: K's D3D12 stages** (compile_k.py, shared with RDNA4) still convert toward zero; needs an RDNA4 run.
- RDNA4: the shared files changed (dialect_hlsl.h, ags_wmma.hlsli, output_store.glsl, down_m.comp, dec0_m.comp) only
  under new defines; RDNA4's dec0_m, dec0_m_planes, down_m, down_m_rgb10, enc0_m, post_m and layer_m DXIL built from
  origin/windows and from the worktree are byte-identical.

## FG

Goal: DLSS Frame Generation (fg/dlssg) on native Windows as fast as on Linux, RX 7700 XT (gfx1101), AMD Software
26.10.1. Branch `hillclimb/fg` from `windows` f356317 (the owner's uncommitted FG installer work carried, not
committed). Workspace `D:\d4r-fg`: `rt\` is a portable d4r folder (d4r.ini and fg\dlssg copied from Ghost of Tsushima's
install), `bin\` the harness + nvapi64.dll, `scripts\run.sh` (one run), `bench.sh` (A B B A, medians), `guard.sh` (sha256
of the generated images, 60 frames, Lite/full x Exact 0/1 x 2x/4x/6x). The harness's `--lite` alone does not pick the
Lite DLL; the host's d4r.ini Model does (run.sh sets D4R_FG_MODEL). The guard baseline is deterministic (two runs equal).

Harness: 2560x1440 output, 1280x720 render, 400 frames, `--bench --pipe`, GPU ms per real frame.

### 2026-11-11: hand-offs (lead 1, 2)

- **Measured** with a new `[Host] SplitLog=1` (nvxrt.log: the kernels and D3D12 calls of a few lists, batch ends marked).
  At 2x one list per real frame is cut into 5 kernel batches; at 4x 7 (all three evaluates in one list). Barriers were
  already harmless. What ended batches: the first D3D12 work (unavoidable), `ClearUnorderedAccessViewUint/Float` (4 per
  evaluate: a raw R32 buffer cleared to 0x7fffffff, an R8_UNORM 1280x720 surface cleared to 0.0f), and between the
  evaluates of one list `EndEvent/EndQuery/ResolveQueryData/BeginEvent` (the host's GPU timer and the harness's
  timestamps).
- **Cost of a hand-off**: `SplitBarriers=1` (45 batches instead of 5) +6.0 ms at 2x: ~0.15 ms each. A measurement-only
  hack that let the clears not split (wrong order): 3.75 -> 3.43 ms at 2x. Hypothesis confirmed.
- **Kept: clears on the kernels' queue, timestamps and markers do not split** (`VkClears=1`, `SplitQueries=0` defaults).
  A clear with a batch open runs as a dispatch in the batch (host/clear_uav.comp): 2D surfaces through the same raw
  D3D12 UAV descriptor the kernels use (format-less store), raw / 32-bit buffers through the GPU VA; anything else
  stays on D3D12. EndQuery, PIX markers and resolves into readback memory go to the next D3D12 part. Every list is now
  1 batch (2 parts) at 2x-6x. Side effect: the host's per-MultiFrameIndex GPU times now put the whole list on index 1;
  their sum (per real frame) is unchanged. Guard: byte-identical, all 12 modes.

| FG (Lite) | before | after | delta |
|---|---:|---:|---:|
| 2x | 3.749 | 3.350 | -0.399 |
| 3x | 4.418 | 3.917 | -0.501 |
| 4x | 5.361 | 4.429 | -0.931 |
| 5x | 6.308 | 4.998 | -1.309 |
| 6x | 7.021 | 5.619 | -1.402 |

  Full model: 2x 5.819 -> 4.947, 4x 11.443 -> 10.439, 6x 16.985 -> 15.692. (3 rounds A B B A for Lite, 2 for full;
  every range within +-0.05 except Lite 6x after, 5.52-5.91.)

### 2026-11-11: noise, the knobs, where the rest goes (lead 3, 4)

- **The machine's state moves the numbers more than most changes do.** At 03:39 Firefox and Discord started; from then on
  the same builds measured much lower and moved minute to minute (the committed build at 6x: 5.62 before, then 4.52,
  4.80; 2x: 2.47 and 2.72 in two benches ten minutes apart). Likely clocks: other GPU work keeps the GPU up through
  the idle hand-off gaps. Only A B B A comparisons in one sitting are trusted below; absolute levels need the game.
- Baseline vs one-batch build again in the busy state (3 rounds, A B B A): 2x 3.282 -> 2.716, 3x 3.970 -> 3.252,
  4x 4.630 -> 3.769, 5x 5.290 -> 4.285, 6x 5.990 -> 4.801. The gain holds (-0.57 to -1.19 ms).
- **New: `[Host] KernelTiming=1`** (Vulkan timestamps after each dispatch, summed per kernel, logged every 1000
  batches). 2x Lite, quiet state: 2.77 ms on the kernels' queue of 3.35 ms per real frame, so ~0.58 ms is the remaining
  D3D12 -> Vulkan -> D3D12 round trip plus the D3D12 parts. Kernels per real frame: k_conv_fp16_nhwc 0.904 (16.8
  launches), DownsampleRGBAndEstimateUIFused 0.390, BlendCandidatesFGLite 0.288, EstimateIntermMvecsScatter 0.272,
  DL1Net_Input 0.142, k_element_wise 0.130 (4.9), EstimatePrev2CurrScatter 0.100, Prev2CurrUnpackPull 0.085, k_pooling
  0.085 (4.9), k_upscale 0.082 (4.9), InputMvecProcessing 0.079, the 4 clears 0.077, rest < 0.05 each.
- **Rejected: kernels on a compute-only queue family** (`[Host] ComputeQueue=1`, family 1, flags 0xe; kept as a knob,
  default off). The first single run said 2.74 vs 3.35 ms at 2x, a bench minutes later in the changed state said
  -0.44 / -0.68 / -0.09 ms, a block test and a 4-round A B B A then said +0.16 / +0.19 / +0.25 ms (2x / 4x / 6x).
  Guard byte-identical. The first numbers were the state change.
- **Ruled out: the deferred shader-only barriers.** Dropping them (measurement-only hack) saves 0.061 ms at 2x and
  0.079 ms at 6x per list. Not worth touching D3D12's state tracking.
- GpuSync=0 (CPU hand-offs) not timed: every hand-off would add a CPU round trip on top of the GPU one.
