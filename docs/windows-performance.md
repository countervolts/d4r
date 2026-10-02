# Windows RDNA4 measurements

Hardware: RX 9070 XT / gfx1201, Windows 11, display driver 32.0.31041.1004.
Runtime: TheRock 10.2.0a20260929, HIP 7.17.26386; ZLUDA b0161a4 with
`D4R_ZLUDA_WMMA=1`, FP8 widening, native FP8 disabled, f16 reference rounding.

The original serializing profiler uses HIP events on the actual launch stream, including
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

The following 30-second capture starts after the user confirms the same
foreground scene is ready. All 1902 presents use Hardware Independent Flip
and SyncInterval 0; input/output resolution remains 2259x1271 -> 3840x2160.
The mean present interval is 15.730 ms (**63.574 FPS**). PresentMon estimates
mean GPU busy/wait at 14.504/1.110 ms, GPU time at 15.614 ms and CPU wait at
0.052 ms. These frame-attributed metrics do not equal a hardware utilization
sensor and do not establish a physical CPU limit. FSR4 has not been captured
in this same scene, so its previously reported 80+ FPS remains a user reading.
Raw files: `test-results/silent-hill2-k-steady-profile-4k`.

The containing ten-minute diagnostic completes 36591 K frames, with zero
backend failures, CPU image copies, previous-frame outputs or recorded game
crashes. It deliberately stops at its time bound. Output scans and serializing
kernel events are off. Whole-run median boundary timestamps measure input
copies 0.063 ms, external span 5.399 ms and output copies 0.212 ms. The external
span includes HIP execution and host/driver scheduling, not only transformer
arithmetic. These medians cover the whole run; the PresentMon numbers above
come from the explicitly selected 30-second scene.

### Deferred kernel events and batched inputs

The new optional `-ProfileKernelsDeferred` samples every seventeenth launch
by default (`-KernelProfileEvery`). Its thread-local pool is bounded to 64
event pairs. It queries ready events before selected launches and after the
application's existing context synchronization, reusing a pair only after
reading its old timestamps. It adds no explicit event/stream/device completion
wait; timestamps, queries and logging still add overhead. Stream capture and
legacy default-stream launches are skipped by default. `-ProfileLegacyStream`
allows the latter explicitly, retaining their legacy cross-stream ordering.
This opt-in is necessary to profile this DLSS K binary, whose measured launches
use the legacy default stream. `completion_ms` now denotes collection lag for
deferred records. The report separates serializing/deferred records and retains
invalid timestamps and skipped samples outside kernel timing statistics.

HIP 7.17.26386 reports some negative elapsed times in the tiny queued PTX
diagnostic, including intervals exceeding one microsecond. These are logged as
`D4R_KERNEL_PROFILE_INVALID`; they are not clamped or used as valid GPU samples.
The cause is not established, and nonnegative samples alone cannot prove a
runtime's timestamp accuracy. The diagnostic verifies all 516 PTX launches
across primary/created contexts and individual/queued submission, checks final
device output and guards, accounts for every collected event and verifies safe
default-stream skipping. Results: `test-results/zluda-deferred-profile-final-ptx`.
The same diagnostic with stable HIP SDK 7.2 also verifies all 516 outputs and
reproduces negative intervals (`test-results/zluda-deferred-profile-stable-ptx`).
A native K dec4 sample is negative too; this is not confined to tiny PTX kernels.
Do not infer hardware/kernel utilization or a speedup from these event samples.
HIP event semantics: [AMD event management](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html).

The opt-in `-BatchInputCopies` (`D4R_BATCH_INPUT_COPIES=1`) enqueues native
pixel decoding and device-to-array input copies on the shared HIP/ZLUDA legacy
default stream. One all-stream completion remains before NGX can use any input
on its own streams. The independent D3D12 input-fence check, NGX completion and
same-frame output dependency remain. Output transfers retain their existing
completion. Every image stays in VRAM. The ZLUDA implementation adds
`cuMemcpy2DAsync_v2` using `hipMemcpyParam2DAsync`; format-80 array emulation
is explicitly rejected because its synchronous implementation uses temporary
host storage. The Windows path supplies canonical FP16/FP32 arrays instead.
This experiment remains disabled by default. Correctness checks pass, but the
controlled game comparison below shows only a small frame-time difference.

```powershell
.\scripts\windows\test-deferred-kernel-profile.ps1
.\scripts\windows\test-async-k.ps1 -BatchInputCopies -ProfileGpuBoundary -ZludaRoot "$PWD\dist\zluda-windows-deferred-profile"
```

The rebuilt Windows shim and new ZLUDA pass all seventeen CTest gates,
including the added asynchronous pitched FP16/FP32 array-copy test and
padding checks. Completed boundary timestamps are also read before reuse if
the copy fence advances between the first completion scan and slot selection;
this prevents diagnostic records being overwritten without adding a wait.

The depth/stencil and packed-format burst/recreation gates pass 24 full K/4K
frames. All twelve candidate RGB outputs match fresh unbatched synchronous
controls exactly, with finite RGBA. The former also enables deferred legacy
events; invalid timestamps remain separate. These short correctness fixtures
include initialization and do not establish steady-state FPS or a copy-stage
speedup. Raw results: `test-results/k-batch-deferred-4k` and
`test-results/k-batch-packed-4k`. ZLUDA candidate source is `67127dd`, isolated
in `dist/zluda-windows-deferred-profile`; patches 0016/0017 and all preceding
patches apply to the clean pinned base. The public prerelease is unchanged.
After committing/rebuilding ZLUDA with a clean worktree, the twelve-frame
four-way K/4K gate passes again with exact candidate RGB and finite RGBA
(`test-results/k-batch-committed-source-4k`). This warms the committed runtime's
JIT cache before the game comparison.

The following user-confirmed 30-second foreground-scene capture contains 1925
presents at **64.265 FPS**, versus the previous unbatched run's 63.574 FPS.
Both use 2259x1271 -> 3840x2160, Independent Flip and SyncInterval 0, without
kernel-event profiling, output scans or attached debuggers. Mean present time
is 15.561 versus 15.730 ms, a 0.169 ms difference. This single pair does not
establish repeatability or a large performance improvement. The containing
ten-minute candidate run completes 35673 K frames with zero backend failures,
CPU image copies, previous-frame outputs or recorded game crashes. Whole-run
median input preparation drops from 0.482 to 0.332 ms; median external span
is 5.438 versus 5.399 ms. Whole-run stages are not the selected PresentMon
window. No GPU utilization conclusion is drawn from ETW estimates. Results:
`test-results/silent-hill2-k-batch-inputs-4k`. The K performance gate stays open.

### AMD driver telemetry

A separate read-only diagnostic polls the installed driver's ADLX base C
interfaces for device-wide usage, core/VRAM clocks, board power and temperatures.
It adds no work to game queues and changes no tuning setting. Build it locally
using the official SDK pinned at `32b5a740d42295c5dfe9026b9f52683da0f3af91`:

```powershell
.\scripts\windows\build-gpu-telemetry.ps1
.\scripts\windows\capture-presentmon.ps1 -Seconds 30 -GpuTelemetry
```

The SDK retains its own AMD license; SDK headers, implementations, driver DLLs
and the SDK-dependent executable are excluded from public game packages.
The probe is independently written and uses the documented C ABI, including
with LLVM-MinGW. It loads `amdadlx64.dll` only from Windows System32 and
releases acquired interfaces before ADLX termination/unload. GPU selection
must match exactly one device (`-GpuName` selects a different name substring).
The process exits at its bound or target process exit. The capture retains
CSV, raw stderr, exit/timeout, driver DLL version/hash and SDK build identity.
No optional telemetry failure is silently converted to a valid measurement.

CSV records independent host UTC/QPC timestamps and the driver's raw sample
timestamp. On this installed ADLX 1.5.0.124, the latter is uptime despite the
SDK description of epoch time; do not interpret it as UTC. Unsupported metrics
are blank, zero utilization/clock remains valid, and the report uses each
driver timestamp once. Device-wide usage is distinct from PresentMon's
per-process frame attribution and is not measured WMMA occupancy. ADLX API:
[AMD GPU metrics sample](https://gpuopen.com/manuals/adlx/adlx-c__perf_g_p_u_metrics/).

The initial idle-device smoke test captures six valid polls, selects the real
RX 9070 XT and terminates ADLX successfully. Board power/core clock/usage
and temperatures are available; `GPUPower` is unsupported and stays absent
from the report. The first poll costs roughly 10 ms for initialization;
subsequent polls are about 0.08 ms at 500 ms intervals. This idle smoke test
does not establish game clocks, utilization or an overhead-free benchmark.
Results: `test-results/gpu-telemetry-smoke`.

The subsequent combined-capture smoke test collects 246 presents and ten
ADLX polls and exits cleanly (`test-results/gpu-telemetry-capture-smoke`).
It includes startup/scene changes and is excluded from FPS comparisons.
After the user confirms the foreground comparison scene, the 30-second run
with detailed d4r stages/boundary timestamps disabled records **64.661 FPS**
over 1935 presents. ADLX's sixty polls report mean usage **93.183%**, core
clock **3124.667 MHz**, board power **316.717 W**, temperature 57 C and hotspot
82.9 C. Mean PresentMon GPU busy/wait is 14.551/0.800 ms. All presents retain
Independent Flip / SyncInterval 0. This records high device-wide driver usage
in this run; the user identified the earlier 63% reading as Task Manager.
It is not the same measurement as ADLX device-wide usage. Do not infer a physical CPU
limit, isolated shader occupancy or a FSR4 speed comparison. Query overhead
is median 0.094 ms per 500 ms poll, with a roughly 12 ms first query. Results:
`test-results/silent-hill2-k-telemetry-unprofiled-4k`. The game stability
summary completes 37187 K frames with zero backend failures, CPU image copies,
previous-frame outputs or recorded crashes, stopping at its ten-minute bound.
Remaining K work now prioritizes transformer/output-kernel cost; frame
scheduling remains measured.

### Rejected packed FP16 accumulator experiment

An isolated `D4R_K_F16_WMMA` enc1 build uses gfx12's documented packed FP16
C/D intrinsic, retaining the current operand/accumulator lane mapping. It
removes most conversion instructions (1066 static F16/F32 conversions become
two) and reduces main-kernel VGPRs from 218 to 201; LDS remains 24576 bytes,
with zero spills. These are compiler/ISA counts, not GPU time or occupancy.
The candidate is rejected before moving to another layer: its real replay
changes 1187603 half codes in the output allocation, max absolute difference
0.0390625 from the validated baseline. All values are finite. Against the
current twelve-window NumPy reference, PSNR drops from baseline 93.7527 dB
to 69.2377 dB; max absolute error rises from 0.0078125 to 0.0390625.
The original comparison incorrectly selected a pre-normalization-fix saved
output (78.7702 dB); a fresh replay matches `k-real-strict-dual-reference/enc1`
exactly and supplies the corrected baseline here. The rejection still holds.
Two independent PTX windows give 70.4022/70.08 dB. These pass the existing
coarse reference thresholds but do not satisfy this optimization's baseline
preservation gate. No game installation or performance claim follows.

A public 64-matrix primitive reproducer isolates the difference to direct
instruction arithmetic. The packed intrinsic and the layout adapter agree
bit-for-bit. Direct FP16 output differs from FP32 WMMA rounded once to FP16
even after one K16 step (1674 of 4096 elements, max absolute 0.00390625).
After eight steps, 2874 elements differ, max absolute 0.0625. This rules out
the adapter packing as the source of this fixture's difference; the hardware
internal accumulation/rounding cause is not established. Production K keeps
its existing FP32-instruction/FP16-rounding baseline and the experimental flag
is off in every normal build. Results: `test-results/k-f16-wmma-enc1` and
`test-results/wmma-f16-arithmetic-compare-fixed`.
The finished `test-wmma-f16.ps1` repeats the same 64-case result. Rebuilding
every ordinary K/M object leaves all sixteen byte-identical; all seventeen
CTest gates pass and enc1 still compiles for gfx1101 (compile-only). Logs:
`test-results/wmma-f16-script-check` and `test-results/k-f16-baseline-ctest.log`.

```powershell
.\scripts\windows\test-wmma-f16.ps1
.\scripts\windows\build-native-k-f16-wmma.ps1 -Layer enc1
```

The first command compiles and runs the public arithmetic diagnostic without
NVIDIA inputs. Its success means the reproducer executed correctly, not that
the two instruction paths are equivalent; JSON retains `baselineEquivalent`
and `gameOptimizationAccepted`. The second compiles the rejected experiment
in a separate directory and generates no override manifest. Intrinsic API:
[Clang gfx12 FP16 WMMA](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html#builtin-amdgcn-wmma-f16-16x16x16-f16-w32-gfx12).

### Controlled native replay timings

`native_replay_probe` now supports `--benchmark-timing events|dispatch` and
`--benchmark-batch 1..256`. Dispatch mode uses the documented
`hipExtModuleLaunchKernel` start/stop events, with work-item dimensions derived
from each module's launch metadata and flags zero. It does not change wave
order or arithmetic. Control/candidate can have different native block/grid
metadata; each launch uses its own geometry. The default remains event markers.

Each AB/BA sample restores captured inputs outside timing, submits a series,
waits for its end event, and logs both the event interval and independent host
launch-to-completion interval divided by launch count. These are isolated
diagnostics: batch intervals include dispatch/queue gaps and are not exclusive
instruction execution times, frame latency, or game FPS. Zero, negative,
nonfinite or event intervals exceeding the enclosing host interval are retained
as `D4R_REPLAY_PROFILE_INVALID` and excluded. Reports keep timing sources,
batch counts and log files separate, retaining host statistics and invalid
samples; no partially valid pair is accepted by the benchmark runner.

The new runner first executes fresh ordinary control/candidate replays and
compares every allocation's SHA256 before timing. It stops at any mismatch,
then checks the control output again after every benchmark. A successful
timing run does not replace the independent NumPy/PTX correctness gates.
Captured weights/IR and GPU outputs remain private.

```powershell
.\scripts\windows\benchmark-native-replay.ps1 -Module "$PWD\build\native-k-gfx1201\dltss_pwin_enc1_layer.hsaco" -Candidate "$PWD\build\native-k-gfx1201\dltss_pwin_enc1_layer.hsaco" -FixtureDirectory "$PWD\test-results\ngx-k-fp16-fenced-capture\replay\replay-000001-dltss_pwin_enc1_layer"
```

The same-module enc1 control has all exact allocations and 32 valid pairs per
mode. Median per-launch times at the captured 512x288 workload are 0.054850 ms
with separate single-launch events and 0.089700 ms with single-launch dispatch
events. At batch 64 they converge to 0.048213/0.048017 ms. Thus dispatch events
alone do not remove timing overhead; use the batch/host cross-check. All eleven
K same-module controls pass sixteen pairs at batch 64 in both modes, including
dec4. This does not fix or explain the previously recorded negative ZLUDA
timestamps. Results: `test-results/k-replay-timing-methods` and
`test-results/k-all-layers-batch-profile`.

One-frame standalone K/4K HDR captures provide actual enc0/enc1/dec0 inputs.
Each harness run uses the eleven validated native layers and private output
tail, passes the finite-output scan and completes without network fallback. At this
synthetic resolution enc1 uses a 61x35 grid; this workload is distinct from
the game's 2259x1271 input. Experiments retain F32 WMMA instructions and strict
FP16 rounding. `build-native-k-experiment.ps1` builds separate objects and no
override manifest; no experimental object is installed by the game runner.

CU/WGP and non-temporal load/store/both on enc1 preserve every allocation but
do not improve its paired 4K timings. Position-only token tiling also preserves
outputs: enc1 with two tiles and shared V spills four VGPRs and is about 9%
slower; one-tile shared V has no measured gain. The two-tile shared-V enc0 run
initially has a roughly 4% lower median paired ratio amid scheduling variance.
A repeat with 64 AB/BA pairs at batch 64 gives control/candidate medians
0.297060/0.282559 ms (events) and 0.299988/0.285011 ms (dispatch), with median
paired ratios 0.950863/0.952240. A 512x288 replay independently gives ratios
0.950396/0.952582. Both tested dec0 shared-V variants are slower or indistinguishable.

Enc0 now selects shared V for the gfx12 strict baseline, preserving the existing
two-token tiles, F32 WMMA and FP16 rounding. Gfx11 and arithmetic experiments
retain their prior selection. `D4R_K_PRIVATE_V` restores the previous enc0 for
comparison; the build helper exposes `-PrivateValues`. The exact CMake-produced
object also passes a fresh 4K replay and 32 paired batch-64 samples, with ratios
0.955418/0.948494. Its ELF hash differs from the experimental object; `.text`
and `.rodata` match exactly, and numerical validation is repeated on the actual
object. The other fifteen ordinary K/M objects remain byte-identical.
Gfx1101 enc0/enc1 compile-only controls retain identical code and constant
sections to their original sources; no RDNA3 hardware execution is claimed.

Full K controls/candidates complete sixteen 4K frames, and twelve more with
the game's 2259x1271 input and 3840x2160 output. Every compared RGB half code
matches, all RGBA scans are finite. Results: `test-results/k-enc0-shared-full-4k`,
`k-enc0-shared-full-game-input`, `k-4k-enc0-pos2-shared-repeat`,
`k-enc0-pos2-shared-small` and `k-enc0-default-shared-4k`. The saving is about
0.015 ms for enc0 in the isolated 4K replay, not 5% application FPS.

The separate `D4R_K_PACKED_ACC` experiment stores already-rounded accumulators
as half vectors between steps, widening for each unchanged F32 WMMA and
rounding its F32 output back to half. The ordinary `acc8v` alias remains F32.
Enc0/enc1/dec0 4K replays match every allocation exactly. Enc1 main VGPRs fall
218 -> 195, LDS remains 24576 bytes and spills stay zero, but paired GPU time
does not improve; enc0 is about 4% slower and dec0 about 1% slower. Combining
packed storage with enc1 two-tile shared V is about 13% slower. The flag remains
off. Smaller register counts alone are not evidence of an optimization.
Raw results are under `test-results/k-4k-*-packed-acc` and
`test-results/k-4k-enc1-pos2-shared-packed`.

A direct recompilation of the private output tail's optimized LLVM IR with
TheRock Clang 24 fails in instruction selection with `unsupported library
call operation` (driver exit 1, backend exit 70, exception 0xE0000046). The generated crash
reproducer/log are retained locally; strict FP is not removed to work around
this. The working LLVM 22 output object remains selected. The HDR tail's
metadata is wave32, 133 VGPRs, 6912 LDS bytes and no spills; LDR uses 128 VGPRs
with the same LDS and no spills. These are static resource counts, not actual
occupancy. Compiler scheduling hints are documented in the
[LLVM AMDGPU backend guide](https://llvm.org/docs/AMDGPUUsage.html#llvm-ir-attributes).

### Private output-tail WGP experiment

ZLUDA source `fb0adc8714f57647320f0601533ec50331e378b0`, patch 0018, adds
opt-in `D4R_ZLUDA_WGP=1`. It changes only CU/WGP placement for gfx11/gfx12,
preserving wave size and arithmetic. The ordinary runtime stays in CU mode.
The offline texture builder accepts `-ShaderMode wgp` and verifies `-cumode`
on the emitted kernel before accepting the object. The compiled output-tail
objects carry this mode independently of the runtime's environment variable.
These NVIDIA-derived objects remain private; the public package excludes them.

The matched K/4K comparison uses identical public network objects and the
existing validated CU output kernels as control. Both HDR and LDR complete
eight frames per variant; repeating with candidate-first order completes
sixteen frames per variant. All 96 control/candidate frames are finite, and
all 48 candidate RGB outputs exactly match their fresh controls. Every frame
has a native output-tail hit and a valid serializing event measurement.

| Output-tail median, ms | CU, first run | WGP, first run | CU, reverse order | WGP, reverse order |
| --- | ---: | ---: | ---: | ---: |
| LDR | 1.864650 | 1.703100 | 1.840260 | 1.687500 |
| HDR | 2.012301 | 1.883395 | 2.016020 | 1.869250 |

These are separate-process serializing diagnostic measurements, not paired
resident-module timings or game FPS. Unchanged network kernels also vary
between processes. The repeat indicates a candidate worth testing in the
game; it does not establish a frame-rate improvement. Results:
`test-results/k-output-wgp-reference-4k-clean-env` and
`k-output-wgp-reference-4k-reverse`.
The game-sized WGP async burst/recreate gate completes twelve frames, all
finite; all six async RGB outputs match fresh sync controls exactly, with
no CPU image copy or aged output (`k-output-wgp-async-game-input`).
The exact CMake shared-V enc0 and WGP output tails together also pass twelve
game-sized async/burst/recreation frames, with all six async RGB images exact
(`k-enc0-shared-wgp-async-game-input`). Default selection of private output
objects remains CU until a game comparison establishes the candidate's effect.

### Combined K game check

The private WGP outputs are rebuilt from clean ZLUDA source `fb0adc8`; both
object hashes match the earlier validated experimental objects. A final
2259x1271 -> 3840x2160 comparison completes twelve finite control/candidate
frames, all six candidate RGB outputs exact, with all expected native hits and
valid timings (`k-output-wgp-clean-source-final-4k`). This supplies clean build
provenance without changing the tested binary identities.

The subsequent Silent Hill 2 game check combines the exact CMake shared-V enc0
and those private WGP outputs. Package source is clean `68b551b`; the runtime
remains the validated `67127dd` ZLUDA build with its warm cache, while the
offline output objects carry WGP placement. DLSS is the locally provided
310.9.1.0 DLL. Async interop and batched GPU inputs are enabled; debugger,
serializing/deferred kernel profiling, detailed stage/boundary timestamps and
output scans are disabled.

After the user confirms the stationary foreground comparison scene, thirty
seconds capture 1954 presents: **65.250 FPS**, mean/median/p95 frame intervals
15.326/15.318/16.306 ms. Every present uses Hardware Independent Flip with
SyncInterval 0. PresentMon mean GPU busy/wait is 14.403/0.807 ms. Sixty unique
ADLX polls record **93.500%** mean device usage, **3118.400 MHz** core,
2505 MHz VRAM and **318.817 W** board power, with 57 C temperature and 82.9 C
hotspot. Median telemetry query time is 0.083 ms per 500 ms poll.

The preceding accepted scene capture was 64.661 FPS: the new result is only
0.589 FPS higher, about 0.91%, with mean frame interval 0.139 ms shorter.
This is one historical control and one candidate capture, without a fresh
repeated game A/B. It does not establish a reproducible gain or isolate enc0
from output placement. WGP remains an optional experiment, and the overall
K performance gate stays open.

The full bounded run completes **35664 K frames** and **427968 native
launches**, including feature recreation and color-format reimports. There
are zero backend errors, CPU image copies, previous-frame outputs or recorded
crashes. `diagnostic_timeout` denotes the intentional ten-minute stop.
Numerical game scans are off, so the game's zero reported nonfinite count is
not a NaN/Inf validation result; use the separate finite/exact-frame gates.
Raw capture, environment, package identities and summary remain in
`test-results/silent-hill2-k-vshare-wgp-4k` and its diagnostic ZIP.

```powershell
.\scripts\windows\build-zluda-windows.ps1 -InstallRoot "$PWD\dist\zluda-windows-wgp-experiment"
.\scripts\windows\build-native-texture.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -ZludaRoot "$PWD\dist\zluda-windows-wgp-experiment" -ShaderMode wgp -OutputDirectory "$PWD\build\private-textures-wgp-gfx1201"
.\scripts\windows\build-native-texture.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -ZludaRoot "$PWD\dist\zluda-windows-wgp-experiment" -ShaderMode wgp -Kernel hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel -OutputDirectory "$PWD\build\private-textures-wgp-gfx1201"
.\scripts\windows\test-native-texture.ps1 -TextureRoot "$PWD\build\private-textures-wgp-gfx1201" -ControlTextureRoot "$PWD\build\private-textures-gfx1201" -ZludaRoot "$PWD\dist\zluda-windows-deferred-profile" -OutputResolution 3840x2160
```

The compiler placement unit passes, all seventeen hardware/software CTest gates
pass on the clean source build with ordinary CU defaults, and the six ZLUDA
gates also pass with WGP explicitly enabled. All seventeen patches (0002..0018)
apply to the clean pinned base `ee2f25a`. The numerical benchmark's rejected
F16 candidate stops after its two ordinary reference replays with differing
output and no timing samples (`k-replay-rejected-f16-gate-fixed`).

### Diagnostic environment and input dimensions

Some PowerShell/.NET hosts preserve removed process variables as empty strings.
Presence-based Rust diagnostic flags then select deferred profiling, while
empty codegen values can disable WMMA. The probe/game child environments now
omit empty `D4R_` keys; the probe restores codegen defaults for null or empty
values. Disabled WGP is also omitted so older runtimes keep their warm cache.
The texture test requires valid timing records instead of silently accepting
a profiling-free comparison. The initial candidate attempt times out during
NGX initialization with no accepted frame; later pre-fix comparisons have
correct RGB but no accepted GPU timings. Their logs remain in
`k-output-wgp-reference-small`, `k-output-wgp-reference-small-fixed` and
`k-output-wgp-reference-4k-fixed`; they are excluded from the table above.

The D3D12 harness now supports `--ngx-input-resolution WIDTHxHEIGHT`, exposed
as `-NgxInputResolution` in the probe runner and `-InputResolution` in the
texture/async gates. Output defaults and limits remain unchanged. This allows
the game's non-aligned input dimensions without inventing a larger output.
An initial attempt using 4518x2542 output is rejected by the harness's existing
range guard before GPU initialization; it is not a GPU or enc0 failure.

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
