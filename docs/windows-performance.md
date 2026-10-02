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

Native FP8 remains a separate optional experiment; the package uses the
validated FP16-equivalent baseline. Its later checks and timings are below.

## K host synchronization and asynchronous submission

Current priority is K performance. The user's synchronous 4K reading is
49–51 FPS / 53% GPU utilization, versus approximately 80+ FPS / 100% with
FSR4. These readings do not identify a CPU hardware limit or provide a
controlled same-scene comparison. M/FP8 optimization is deferred.

ZLUDA candidate `a1c506f` adds `D4R_ZLUDA_PROFILE_API`, enabled by the game
runner's `-ProfileCudaApi`. It measures existing API host durations, aggregates
per thread and logs every two seconds without HIP events or additional waits.
The synchronous control shows five `cuCtxSynchronize` calls per frame with
about 3.9 ms aggregate host time. CUDA launch host overhead is approximately
0.46 ms per frame. Cached copy-list setup is only 0.011 ms median. These host
measurements include GPU waits and must not be added to GPU kernel times.

| K/4K command submission, CPU median | Synchronous | First async | Async without CUDA mutex |
| --- | ---: | ---: | ---: |
| Entire submission, ms | 10.512 | 2.676 | 0.181 |
| DLSS boundary, ms | 10.083 | 2.604 | 0.124 |
| Consumer suffix completion, ms | 0.350 | 0.005 | 0.004 |

The first async game run completes 9481 finite frames; the user reports
60 FPS / 63% GPU. The mutex-free run crashes during feature recreation after
366 finite frames, so its submission timing is not a successful gameplay
benchmark. The standalone burst/recreate reproducer subsequently identifies
caller-owned GPU backing freed after ReleaseFeature returns. Waiting for
pending D3D12 consumers in ReleaseFeature fixes the reproducer: all twelve
sync/async test frames pass, and corresponding RGB outputs are exact. Game
stability and FPS must be checked separately on the fixed build.

The fixed game retest on 2026-10-02 completes 16941 K/4K frames, all finite,
including feature recreation and color-format reimports, with zero backend
failures. The user reports 62 FPS / 63% GPU. Median CPU submission is
0.197 ms, NGX host completion on the worker is 4.400 ms and output validation
is 0.380 ms. The 10.162 ms median worker input wait includes queued game
rendering. These results establish a stable five-minute test of this fix,
not a conclusion about the remaining GPU idle time or FSR4 performance.

`-AsyncInterop` is opt-in. It queues input/output fences on the submission
thread, executes CUDA on a worker and retires owned command memory by GPU
completion. Three copy slots bound outstanding ownership. The queue consumes
the output of the same frame, with no CPU image copy. The worker's input wait
includes its lead behind submitted rendering and is not a CPU submission
stall. Neither that duration nor the NGX recording interval measures Present
latency. Reconfiguration waits for all consumers; ordinary submission does not
hold the CUDA mutex. The experimental runtime currently supports one queue.

```powershell
.\scripts\windows\test-async-k.ps1
```

The test generates fresh synchronous controls, submits three frames without
intermediate CPU completion, and repeats while releasing/recreating the DLSS
feature with work pending. Results include exact RGB comparisons, finite GPU
scans, stdout/stderr, exceptions and runtime provenance. To include the local
validated output-store optimization, pass its combined native directory with
`-NativeRoot`; proprietary objects remain local.

`-ProfileGpuBoundary` (requires `-AsyncInterop`) adds D3D12 timestamps before
and after the VRAM input/output copies. Resolved results are read only once
the copy slot's existing fence completes; 32 diagnostic bytes per frame are
read and no new completion wait is added. `external_span_ms` spans HIP and
host/driver scheduling across the external fence. `between_boundaries_ms`
also includes game rendering, and `input_interval_ms` is not Present timing.
These queries obey the queue's timestamp ordering and add GPU commands; they
are disabled for an ordinary FPS run. GPU clock behavior during idle can
affect intervals; no hardware occupancy is inferred. Timestamp semantics:
[Microsoft D3D12 timing](https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing).

The timestamp-enabled 4K burst/recreation gate completes twelve frames, all
finite; all six async RGB outputs still match their fresh controls exactly.
Each async mode emits all three timing records, including the final frame
at feature release. Results: `test-results/k-async-gpu-boundary-4k`.
All sixteen CTest gates pass on the rebuilt shim. Reproduce with:

```powershell
.\scripts\windows\test-async-k.ps1 -ProfileGpuBoundary
```

The 2026-10-02 game timestamp run completes 9384 K/4K frames without backend
errors (`test-results/silent-hill2-k-gpu-boundary-4k`). Median GPU intervals:
input copies 0.061 ms, external span 5.586 ms, output copy 0.208 ms, entire
boundary 5.856 ms, between boundaries 9.512 ms. Median input interval is
15.383 ms. Output scans and serializing kernel events are off. This locates
the remaining time outside the small D3D12 copies; it does not prove whether
that time is game GPU work, CPU waits or context scheduling.

### Present and GPU-busy capture

`capture-presentmon.ps1` attaches the official PresentMon 2.6.0 x64 CLI to
one running game process. It verifies SHA256
`b2a706bc6ad475749e3b7e3409263aa1e6906d45bdcf993f6dbc0f660188f1af`,
uses a unique ETW session and stops at its time bound/process exit, without
replacing any existing trace or installing a service. It saves CSV, console
output, driver version, target PID and the HWS registry setting. The default
downloads the pinned CLI into `.tools` if needed. Run elevated when ETW access
requires it; no driver or security setting is changed.

```powershell
.\scripts\windows\capture-presentmon.ps1 -Seconds 30
python .\tools\windows\profile_report.py <capture-directory> --output profile.json
```

Keep the game foreground and use the same scene/quality setting for comparison.
The report separates swapchains and preserves missing metrics. Present FPS
comes from present intervals, rather than NGX recording timestamps. GPU
busy/wait are ETW estimates, not a hardware sensor percentage; CPU busy is
frame work/wait attribution before Present, not physical CPU utilization.
HWS and cross-API context attribution can affect accuracy. Metric definitions:
[PresentMon console documentation](https://github.com/GameTechDev/PresentMon/blob/v2.6.0/README-ConsoleApplication.md).

The initial 60-second trace captures 2172 presents during a changing
Independent Flip / Composed Flip session, with mean present interval
27.593 ms (36.241 FPS), mean GPU busy 18.759 ms and GPU wait 2.303 ms.
This is not a controlled capture of the user's 62 FPS scene and is excluded
from speedup conclusions. Its containing game run completes 9627 K/4K frames
without backend errors; ordinary recording intervals later return to about
16 ms. A stable foreground-scene capture is needed before optimizing from
the ETW metrics. Raw files: `test-results/silent-hill2-k-presentmon-4k`.

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

The following no-debugger K/4K run completes 9982 frames and 9982 finite GPU
scans. Kernel event profiling is disabled; the validated private output tail
executes natively. The user reports 43-47 FPS. Mean NGX recording interval is
21.867 ms. No backend failure, CPU image copy or previous-frame output occurs.

Random one-in-64 hook samples then identify contention in the shared routing
table used by every intercepted D3D12 command. Per-thread weak lookup caches
now bypass that table on repeated calls to a list. Table mutations invalidate
cache generations; Reset/address reuse cannot route to a previous suffix, and
thread caches do not retain game resources. The per-list and submission locks
remain. A GPU regression resets and re-splits the exact same list with changed
root constants twice per iteration, checking both prefix and suffix results.

| Sampled command-hook access | Shared table | Per-thread weak cache |
| --- | ---: | ---: |
| Busiest recording thread, mean us/call | 0.203 | 0.066 |
| Estimated access + capture across threads, ms/s | 977.528 | 233.606 |

The latter estimate includes lock waits summed across concurrently running
threads, not CPU execution time or serial frame latency. The two game runs
have different thread populations (31 versus 29); there is no identical frame
trace. Generated forwarding methods have separate driver/capture markers;
special-case hooks contribute access samples but lack those body markers.

With the cache, 4770 complete K/4K frames and finite GPU scans pass, with zero
backend failures, CPU image copies or previous-frame output. Median NGX
recording interval is 19.473 ms. The user reports 49-51 FPS and 53% GPU usage
in the previously measured scene. These user readings are separate from the
sampled hook estimates and do not establish isolated WMMA utilization.
All sixteen CTest gates pass, including 64 same-object Reset/split checks in
each physical command-backend test and the software indirect-root regression.
Use `-ProfileCommandHooks -ProfileStages` to collect samples; the sampler is
disabled for ordinary play.

Native M FP8 was subsequently compiled and tested on this gfx1201. ISA confirms
`v_wmma_f32_16x16x16_fp8_fp8`, with packed e4m3 weight images; exact FP16
attention/PV remains. All logical outputs and merges in forty recorded launches
match the previously NumPy-validated FP16 baseline byte-for-byte. Full DLSS RGB
also matches exactly: four 512x288 LDR frames and four 3840x2160 HDR frames,
all RGBA finite. This establishes correctness for those fixtures, not a speedup.

The original software operand packing makes the FP8 network slower in the
four-frame 4K test: sum of five main-kernel event means (tube weighted six times)
is 27.789640 ms versus FP16's 23.550875 ms. A paired in-process replay confirms
the regression without relying solely on separate process timings. Both modules
and prepared weights remain resident, 512 launches warm the GPU, then sixty-four
AB/BA pairs time the same restored input using HIP events. Diagnostic CPU input
restoration is outside the timed event and is unrelated to game VRAM interop.

Added opt-in `D4R_FP8_HW_PACK`: native packed conversion of already quantised
operands replaces software exponent/subnormal reconstruction. It preserves all
254 finite OCP e4m3 encodings, including signed zero, in an exhaustive GPU check.
All forty recorded launches still match exactly. Four complete 4K HDR frames
also match FP16 RGB exactly with finite RGBA. ISA confirms both packed converts
and FP8 WMMA. The baseline build's sixteen objects remain byte-identical.

| Paired replay median, ms | FP16 (software-packing test) | FP8 software packing | FP16 (hardware-packing test) | FP8 hardware packing |
| --- | ---: | ---: | ---: | ---: |
| enc1 | 0.175450 | 0.217050 | 0.175100 | 0.177700 |
| enc2 | 0.141350 | 0.164550 | 0.132550 | 0.131550 |
| enc3 tube | 0.108000 | 0.112000 | 0.107600 | 0.099850 |
| dec2 | 0.129900 | 0.146000 | 0.143250 | 0.140850 |
| dec1 | 0.169650 | 0.194450 | 0.154250 | 0.156550 |

These are two paired tests on the captured 512x288 workload. Medians avoid large
isolated scheduling outliers; raw per-pair ratios are retained. They identify a
substantial cost in software packing but do not establish an overall FPS gain.
The separately run four-frame 4K hardware-packing network mean is 24.603463 ms;
it still does not beat the measured FP16 total. Default packaging retains FP16.
Hardware counters for bandwidth or WMMA utilization were not measured.

Reproduce the isolated build, encoding/replay checks and paired timings using
the existing private validated captures; no source editing is required:

```powershell
.\scripts\windows\build-native-m-fp8.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -HardwarePacking
.\scripts\windows\test-native-m-fp8.ps1 -ModuleDirectory "$PWD\build\native-m-fp8-hardware-pack" -CaptureDirectory "$PWD\test-results\ngx-m-exact-pv-capture\captures" -BaselineValidation "$PWD\test-results\m-native-exact-pv-temporal-reference\summary.json" -Benchmark
```

Omit `-HardwarePacking` to compile the original FP8 experiment. The replay
validator stops at the first mismatch, retains GPU stdout/stderr/exit code and
hashes every compared logical output. `profile_report.py` accepts replay logs
and reports per-pair GPU time and candidate/control ratios. This experiment is
not installed by the game runner or substituted silently for the baseline.
The conversion builtin is listed in the official
[Clang AMDGPU builtin reference](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html#builtin-amdgcn-cvt-pk-fp8-f32).

References: [HIP events](https://rocmdocs.amd.com/projects/HIP/en/develop/doxygen/html/group___event.html),
[HIP occupancy API](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___occupancy.html).
