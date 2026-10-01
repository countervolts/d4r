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

Further game coverage (HIP kernel profiling disabled, GPU output checks enabled):
M at 1920x1080 completes 4833 frames, all finite. K at 3840x2160 completes
4685 frames, all finite. Neither has a backend failure, CPU image copy or an
aged-frame fallback. Their mean interop/NGX CPU stages total about 13.6 and
10.1 ms respectively; these are different scenes, not a K/M comparison.

The no-debugger K/4K run completes 5774 frames, with 5774 finite output scans
and 63514 native transformer launches. The user reports 38-42 FPS. Additional
CPU measurements exclude replay/publication as the main cause of low FPS:

| CPU measurement | Mean, ms | Median, ms |
| --- | ---: | ---: |
| Command split setup | 0.284 | 0.275 |
| State replay within split setup | 0.025 | 0.023 |
| Submitted texture state publication | 0.029 | 0.029 |
| Consumer suffix completion | 0.405 | 0.409 |
| NGX recording interval | 27.474 | 28.466 |

Recording intervals are between NGX recording calls on the same thread, not
DXGI Present times. The interval includes rendering and application scheduling.
A second K/4K run with the attached debugger completes 3567 frames without
failure, with median recording interval 30.091 ms. The runs include different
menu/game states; this does not establish a precise debugger FPS speedup.
Its sparse debugger events also rule out a per-kernel OutputDebugString storm.
An earlier control that stayed on the startup screen completed zero DLSS frames
and is excluded. Normal game launches now omit the attached debugger; crash
diagnostics remain available with `-CaptureExceptions`.

The 2961-frame 4K K profile identifies the translated output tail as its largest
kernel: 4.401670 ms mean GPU time, versus about 2.875 ms for all eleven native
transformer layers combined. Ported upstream's texture-store recipe to native
Windows. It retains accuracy mode, wave32, FP16 subnormals and f16 reference
accumulation; native FP8 remains disabled.

Both output variants execute on every checked frame and match the previous
RGB output bit-exactly: eight consecutive 512x288 frames per variant, then
three consecutive 3840x2160 frames per variant. All RGBA components are finite.
Identical synthetic inputs and public NGX flags give these 4K event times
(three samples each; host initialization and readback are outside the event):

| K output variant | Translated mean, ms | Native stores mean, ms |
| --- | ---: | ---: |
| LDR, regular depth, high-resolution MV | 2.907823 | 1.803233 |
| HDR, inverted depth, low-resolution MV | 2.978547 | 2.088347 |

A subsequent game run executes 6221 frames with the native HDR output tail,
zero backend failures and mean tail GPU time 1.801918 ms. The earlier game
profile's 4.401670 ms is from a different scene/scheduling state; use the
identical-input table for the controlled comparison, not an inferred FPS
multiplier. This profiled game run did not enable the NaN scan.

Reproduce the private texture build and full-frame comparisons:

```powershell
.\scripts\windows\build-native-texture.ps1 -DlssDll "$PWD\nvngx_dlss.dll"
.\scripts\windows\build-native-texture.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -Kernel hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel
.\scripts\windows\test-native-texture.ps1 -OutputResolution 3840x2160 -Iterations 3
```

These objects contain code derived from the locally supplied NVIDIA PTX. They
stay in ignored local directories and are excluded from the public package.
The installer accepts them with `-LocalTextureKernels` only after validation,
checking the exact DLL identity, object and manifest hashes. M retains its
independently validated FP16-equivalent baseline.

References: [HIP events](https://rocmdocs.amd.com/projects/HIP/en/develop/doxygen/html/group___event.html),
[HIP occupancy API](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___occupancy.html).
