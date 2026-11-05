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
