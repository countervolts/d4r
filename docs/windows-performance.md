# Windows RDNA4 measurements

Hardware: RX 9070 XT / gfx1201, Windows 11, display driver 32.0.31041.1004.
Runtime: TheRock 10.2.0a20260929, HIP 7.17.26386; ZLUDA b0161a4 with
`D4R_ZLUDA_WMMA=1`, FP8 widening, native FP8 disabled, f16 reference rounding.

The optional profiler uses HIP events on the actual launch stream, including
separate measurements of one-time weight preparation. It synchronizes each
sampled launch and changes scheduling: these numbers are not application FPS.
HIP occupancy predictions are not measured hardware occupancy. The native
objects' actual wave32, LDS, VGPR/SGPR and spill counts are available in
`llvm-readobj --notes` and can be attached to the JSON report. Memory bandwidth
and WMMA utilization were not measured; no utilization is inferred from timing.

The identical four-frame standalone D3D12/OptiScaler workloads output 512x288.
Summing the measured main-kernel event times and dividing by four gives:

| Preset | Fully translated, ms/frame | Native FP16-equivalent, ms/frame |
| --- | ---: | ---: |
| K | 157.745441 | 0.684675 |
| M | 7.428538 | 1.566500 |

Both rows include translated input/output tails. Native overrides cover all
11 K layers and five M layers; M's tube layer executes six times per frame.
M native and translated RGB are bit-exact, with max absolute/relative error 0.
K native remains bit-exact against its independently validated native control;
fully translated K has the previously documented arithmetic differences.
All outputs are finite. These are reference-correct codegen settings, not
measurements of a relaxed fast-math translated configuration.

M's first profiled Evaluate includes about 0.9 seconds of host initialization;
the four-sample CPU mean must not be treated as steady-state latency. Its
median Evaluate host completion is 4.855 ms. GPU events and prep phases are
reported separately in the raw JSON.

The initial game runner used PowerShell line-oriented output redirection.
It heavily throttled verbose stdout/stderr. The runner now concurrently drains
raw byte streams using the same approach as the standalone diagnostics.

| Silent Hill 2 stage, CPU mean | Initial capture, ms | Raw-byte capture with HIP profiler, ms |
| --- | ---: | ---: |
| Input fence wait | 5.641 | 0.833 |
| Input conversion and CUDA array upload | 13.196 | 0.668 |
| NGX Evaluate host call | 25.715 | 3.350 |
| NGX all-stream completion | 1.846 | 0.033 |
| Output CUDA array download | 4.288 | 0.167 |
| Output conversion and shared fence signal | 1.500 | 0.033 |

The first run additionally scanned every output (10.142 ms in that throttled
capture); the kernel-profiling run disables that scan. Both use a temporary
1280x720 window and current-frame VRAM interop. These are separate menu/game
runs, not a controlled whole-game FPS comparison. No synchronization was
removed to obtain the improvement. The raw-byte profiling run completed
2345 K frames, with no d4r failure, CPU image copy or previous-frame output.

Reproduce native or translated measurements with locally supplied NVIDIA DLLs
at the repository root and the separately built profiling runtime:

```powershell
.\scripts\windows\profile-windows-rdna4.ps1 -Preset 11
.\scripts\windows\profile-windows-rdna4.ps1 -Preset 11 -Translated
.\scripts\windows\profile-windows-rdna4.ps1 -Preset 13
.\scripts\windows\profile-windows-rdna4.ps1 -Preset 13 -Translated
```

`profile_report.py <results-directory> --output profile.json` summarizes raw
CPU stages and every kernel's event time, launch geometry, registers, private
memory, LDS and predicted blocks per multiprocessor. Add
`--metadata-directory <directory-of-llvm-readobj-notes>` for native wave sizes
and compiler spill counts. Reports and proprietary workload inputs remain local.

Native FP8 is not the validated baseline and has no performance claim here.
Replacing the exact per-MMA f16 rounding with f32 FP8 accumulation requires a
new layer/reference validation; it is not justified by the observed CPU stalls.

References: [HIP events](https://rocmdocs.amd.com/projects/HIP/en/develop/doxygen/html/group___event.html),
[HIP occupancy API](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___occupancy.html).
