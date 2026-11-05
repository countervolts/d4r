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
