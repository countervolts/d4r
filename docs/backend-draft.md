# Windows D3D12 / HIP draft, 2026-10-22

Use D3D12 as the Windows game-facing host. Keep HIP as an optional inference
backend where a complete model benchmark justifies its integration and runtime
cost. These measurements do not establish a full-model performance winner.
Keep the existing Vulkan/HIP implementation for Linux.

This is a standalone diagnostic draft, not an installable all-model game build.
No game files or settings were changed. No commits were made.

## Local results

RX 7700 XT, gfx1101 / RDNA 3, Windows 11 build 26300, driver
32.0.31041.1004. DXC 1.9.2609.5, ROCm core/devel 10.1.0, LLVM-MinGW
20261020. No clock/power locking. Five trials, 100 ordered dispatches per batch,
GPU timing, uploads/readbacks excluded. D3D12 uses command lists and UAV barriers;
HIP uses dependent nodes in a captured graph, avoiding Python submission gaps.
Different implementations and ordering machinery remain part of the result.

Each tile computes `C + steps * A@B`, with unique seeded f16 A/B, nonzero f32 C,
and f32 accumulation. Every trial passes an independent float64 oracle and
produces bit-identical D3D12/HIP f32 outputs.

| Independent tiles | Products per tile | D3D12 median, us | HIP median, us |
|---:|---:|---:|---:|
| 1 | 1 | 0.675 | 2.970 |
| 128 | 1 | 0.747 | 2.930 |
| 128 | 128 | 4.126 | 5.792 |
| 4096 | 16 | 16.134 | 14.563 |

D3D12 leads the small/short workloads. HIP has about 10% lower latency on the
largest tested workload. D3D12 trial ranges for the last two cases are
4.117–7.040 us and 15.644–21.016 us; HIP ranges are 5.739–5.923 us and
14.251–15.395 us. Desktop activity and changing clocks limit confidence in
small differences. These are matrix tests, not DLSS or RR frame times.

An additional K plain-block test (C=160, 8 heads, 32x32 tokens) passes its NumPy
oracle at 88.40 dB. D3D12 baseline median is 165.746 us; the existing combined
prefetch/wave-fence experiment is 136.985 us, a 17.4% reduction. All four
variants are bit-identical. This does not select a packaged full-frame variant.

The existing full-frame D3D12 K engine also builds and runs locally with synthetic
weights and four ordinary input frames, 854x480 render to 1280x720 output.
All 24 dumped output/history/intermediate buffers are finite. The first run's
individual frame recordings take 1.422–1.556 ms. A warmed 25-frame-per-submit
run measures 0.936 ms GPU/frame median (0.930 ms minimum); its individual-frame
recordings take 1.415–1.584 ms. Batching changes submission and GPU utilization;
the batched figure is not a same-frame game latency prediction. Synthetic images
have no quality meaning. Cold model/engine setup was 11.88 s, dropping to 34.4 ms
on the next process launch, so pipeline startup remains an important usability
measurement. No complete HIP K-frame comparison was made.

Local artifacts under `build/backend-draft/`:

- `validated/comparison.json`: aggregate matrix results, with per-trial inputs,
  outputs, logs and correctness reports in subdirectories.
- `layer160-validated/ablation-result.json`: K block timing/accuracy results.
- `rdna4-compile/hip-result.json`: successful gfx1201 cross-compilation only.
- `fsr-rr-native/probe.json`: native FSR RR provider query.
- `d4r-k-layer.exe`, `d4r-k-d3d12.exe`, `fsr_rr_probe.exe`: built diagnostic runners.
- `k-d3d12/`: synthetic K model compiled with the existing package builder.
- `frames/`, `k-smoke/`, `k-warm.log`: the four-frame synthetic case, dumps,
  finite checks and full-frame timing log.

## Models and integration costs

| Feature | Existing implementation in this checkout | Practical Windows direction |
|---|---|---|
| DLSS K SR | Full-frame D3D12 engine; Vulkan frame engine; native HIP network kernels | Start with D3D12; validate real model, history/reset and borrowed game resources on RDNA 3/4. Benchmark a complete HIP network plus handoff before selecting it. |
| DLSS M SR | Vulkan engine and native HIP network, shared M/L machinery | D3D12 frame/shader port is still required. HIP reuses more existing inference work but its current host uses Linux fd interop. |
| DLSS L SR | M/L compiler/runtime and native HIP support; L has distinct weights/shapes and frame restrictions | Validate separately from M. K timings cannot predict L, and the Windows CMake build currently omits `d4r-m`. |
| DLSS RR D/E/F | Vulkan RR engine/stages, native HIP RR kernels, preset-specific extraction/build paths | Prefer preserving the validated Vulkan path until a D3D12 port or complete native Windows HIP backend passes stage and temporal replay. K's frame plan does not implement RR. |
| FSR RR / Ray Regeneration | AMD signed D3D12 denoiser runtime; prepare/compose shaders and SR integration in the Proton frontend | Use its D3D12 provider directly when available. It is not a HIP inference kernel that this draft can choose to run. Port prepare/compose and SR handoff for a native Windows integration. |

Relevant sources: `engine/CMakeLists.txt`, `engine/hip_net.cpp`,
`engine/compile_m.py`, `engine/rr/README.md`, `tools/d4r_fsr_rr_backend.inl`,
`tools/d4r_fsr_rr.h`, and `docs/windows-support-plan.md`.

The Linux HIP host imports `hipExternalMemoryHandleTypeOpaqueFd`; its native-network
path is not ready to use from a Windows game. A Windows port must prove adapter
matching, D3D12 shared-resource import support for the selected SDK/driver, shared
fences/semaphores, resource ownership, correct states and lifetimes. If compatible
shared-texture import is unavailable, staging copies must be counted. HIP kernel
latency alone does not account for any of that.

The d3dhip replay route now proves that import path on this machine for the
replay runners: `engine/d3d12/hip_net_d3d12.h` imports the token and head buffers
as opaque NT handles (`hipExternalMemoryHandleTypeOpaqueWin32`) after checking
that HIP device 0 is the D3D12 adapter by LUID and PCI identity, and exchanges
them with the D3D12 engines only while the game queue is idle (CPU-serialized
front submission, fence wait, HIP graph, back). Shared fences/semaphores are not
used or attempted, and a game integration still has to count the CPU round trip
measured there (~1.4 ms/frame at 1280x720 -> 2560x1440).

A D3D12 host can record supported inference into the game's command list and
use its resources directly. It also fits FSR RR's existing public API. This
reduces deployment dependencies and synchronization boundaries, although shader
and model ports still require substantial work. Do not replace the Linux
backend based on these Windows measurements.

## FSR RR and RDNA 4

The local native FSR probe loads AMD's unmodified denoiser DLL successfully but
returns **no provider** for the RX 7700 XT. No FSR RR frames were executed or
timed. DLL SHA256:
`48f1e5888ba6a0a3d59a98b9751e37c392b0f7b8c223d0082d5c1f40642879d3`.
The DLL came from AMD's public FidelityFX SDK `Kits/FidelityFX/signedbin`.

[AMD's current requirements](https://gpuopen.com/manuals/fsr_sdk/techniques/denoising/)
list RX 9000-series or later, D3D12 and SM6.6 for Ray Regeneration. The repo's
experimental Proton/RDNA 3 path does not demonstrate native Windows support.
FSR upscaling support is a separate capability from FSR RR.

RDNA 4 has a different WMMA register layout and higher theoretical matrix
throughput, as described in
[AMD's matrix-core guide](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/).
The HIP draft reuses `kernels/common/wmma_layout.h`, selecting the native gfx11
or gfx12 builtins at compile time; gfx1201 compiles successfully locally.
Cross-compilation proves neither execution nor performance on RDNA 4.

The repository's [RX 9070 XT K report](../engine/d3d12/windows-k-tuning.md)
records 2.62 ms at 4K and 1.39 ms at 1440p for synthetic D3D12 K, with different
arithmetic/reconstruction from Vulkan. Those historical numbers are not a
paired HIP comparison, an RDNA 3 projection, or evidence for M/L/RR speed.
Validate f16 accumulation, FP8 widening/conversion and temporal output for each
model and architecture before judging quality or enabling a backend.

## Reproduce

The local tools live in ignored `.tools/`; build products in ignored `build/`.
The draft requires no system compiler, HIP SDK installer or driver change.
For another machine, prepare a Python environment with NumPy and ROCm core/devel
([AMD installation instructions](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)),
a portable LLVM-MinGW C++ compiler, and Microsoft's DXC. The PowerShell script
accepts paths for all three tools; its defaults match the tools prepared here.
ROCm wheels were installed only into `.tools/backend-venv`, using
`pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[devel]'`.
These are AMD's modular ROCm wheels, not a test of every shipping HIP SDK version.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_backend_draft.ps1 -Arch gfx1101
```

Use gfx1100/1101/1102 for the matching RDNA 3 GPU, or gfx1200/1201 for RDNA 4.
The D3D12 runner selects the first usable AMD adapter and HIP defaults to device
zero; on a multi-GPU machine verify both report the same adapter before using
the results. Run GPU benchmarks sequentially, with games closed.

The standalone HIP companion can also cross-compile a fixture without execution:

```powershell
.tools/backend-venv/Scripts/python engine/d3d12/bench_hip.py FIXTURE_DIR --arch gfx1201 --compile-only
```

Use a separate fixture directory for cross-compilation; it writes `matrix.hsaco`
and `hip-result.json`. A valid fixture is generated by `test_matrix.py`.

The local full-frame K draft can be rerun directly:

```powershell
build/backend-draft/d4r-k-d3d12.exe build/backend-draft/k-d3d12 build/backend-draft/frames/case.txt - 25
build/backend-draft/fsr_rr_probe.exe build/backend-draft/amd_fidelityfx_denoiser_dx12.dll build/backend-draft/fsr-rr-recheck --query-only
```

The FSR probe currently exits with failure because this GPU has no provider;
its JSON records the capability failure. Do not treat that as a timing result.

Before choosing a performance default, replay matched real K/M/L and RR D/E/F
frames at 720p/1440p/4K, with resets, camera motion, HDR/LDR, exposure, dynamic
resolution and history checks. Measure total GPU frame cost, p50/p95 CPU
submission, cold/warm pipeline creation, peak VRAM, transfers and synchronization
on both generations. For FSR RR require an actual provider and full dispatch
quality checks; an unavailable provider is not a slow benchmark result.
