# Windows / RDNA4 port

Branch: `windows-rdna4`. Experimental Windows build targets: **gfx1100..gfx1103,
gfx1150..gfx1154, gfx1200 / gfx1201**.
Validated physical hardware: Windows 11 x64, RX 9070 XT, **gfx1201**.
Priority: correct K, correct M, native Windows, same-frame output, then speed.
Upstream integration: [PR #11](https://github.com/countervolts/d4r/pull/11)
is merged into `countervolts/d4r:windows` at `c522101`.
The integration branch retains upstream's README with a Windows status section;
the fork's `main` keeps the dedicated English Windows README. The maintainer
invitation is accepted; GitHub confirms `xdfnx-dev` has push access and the
Windows branch is unprotected. Further validated Windows changes can go there
directly. K performance remains a separate open gate.
This file records current results followed by the dated development history. Standalone DLSS K/M
works on the real Windows GPU, including the patched OptiScaler frontend.
Silent Hill 2 now renders through the Windows backend; gameplay and output
validation coverage are recorded below. Performance and game coverage have
explicit limits; the validated package keeps conservative arithmetic.

## Milestone and gates

2026-10-02 issue #10 follow-up ([comment 5958769640](https://github.com/countervolts/d4r/issues/10#issuecomment-5958769640)):
four new logs confirm that RX 9060 XT / gfx1200 HIP discovery succeeds. The
Cyberpunk K/M runs never launch the transformer: numeric CUDA NGX capabilities
report `available=0, feature_init=BAD00004` (FeatureNotFound). The unmatched
adapter during enumeration is not the main Radeon: its runtime LUID matches.
GetFeatureRequirements now returns success with AdapterUnsupported for Intel /
WARP / another nonmatching adapter, while runtime initialization still requires
the exact HIP architecture and D3D12 LUID.

The feature discovery failure is reproduced on RX 9070 XT without a game,
using Cyberpunk's unchanged numeric application ID `100152211`. Keeping the
feature DLL only beside the official core in `d4r/vendor/` fails; placing it
beside the calling shim in `d4r/` succeeds. Putting it beside the EXE alone is
insufficient when the call originates in the shim. The installer now stages
the local feature DLL beside the shim and sets `Libraries.NvngxDlssPath` to
the directory, as required by OptiScaler's discovery logic. Project identity,
DLL contents, numeric driver ABI and capability checks remain unchanged.

The standalone reproduction also exposed concurrent/reentrant OptiScaler
device hook installation: Detours error `0x10dd` clears a live trampoline and
execution jumps to null. Source patch `0004` serializes installation/removal
and guards recursion; `0005` forwards numeric NGX init failures instead of
reporting success. The exact patched frontend source is
`86b21ea` (base `45a2001`), with pinned submodules verified directly before build.

The quick launcher no longer attaches a debugger by default: Assassin's Creed
Black Flag Resynced refused the previous debugger launch (`DEADC0DE`). Empty
stdout/stderr now parse correctly. A clean early process exit with zero DLSS
frames is inconclusive, as in the SM2 log; an external Steam restart is not
tracked and must not be reported as successful hardware validation.

Validation on the available RX 9070 XT: all eleven target builds pass; CTest
20/20; summary parser 9/9 including all four submitted logs. The layout
regression passes two expected FeatureNotFound cases (direct + OptiScaler),
then three finite frames each for direct K, OptiScaler K and OptiScaler M.
Every required native layer executes; GPU NaN/Inf checks pass, CPU frame copies
and frame age are zero. Direct/OptiScaler K RGB matches bit-for-bit for all
three frames (max absolute/relative error 0). Reports are under
`test-results/issue-10-5958769640/`, including `ctest.log`,
`summary-parser.json`, `layout-final/` and `build-all-targets.json`.
All five exported OptiScaler patches apply to the pinned upstream sources and
reproduce its exact committed files (`optiScaler-patches.json`). Reproduce the
standalone layout/error/coverage/RGB checks with:

```powershell
.\scripts\windows\test-ngx-feature-layout.ps1 -DiagnosticRoot .\dist\windows-gpu-coverage-gfx1201 -HipRoot .\.tools\therock-10.2.0a20260929\_rocm_sdk_core -ZludaRoot .\dist\zluda-windows-gpu-coverage -NativeRoot .\dist\windows-gpu-coverage-game-gfx1201\d4r\native -OptiScalerDll .\dist\optiscaler-windows-d4r\OptiScaler.dll -NgxCore .\_nvngx.dll -DlssDll .\nvngx_dlss.dll -OutputDirectory .\test-results\ngx-feature-layout
```

The rebuilt compact installer passes six isolated Windows PowerShell 5
fixtures (`quick-fixtures-final/summary.json`): exact backup/restore with the
original game DLSS DLL preserved, automatic gfx1201 selection, a child that
refuses debugging with empty stderr, and four negative input/package gates.
The child exits normally with no debugger and is classified as inconclusive,
with hardwareValidationCompleted=false. No installed game is changed by the
fixtures. Run `test-windows-quick-test.ps1` with the package and the local DLL
paths to reproduce them; it selects Windows PowerShell 5 when called from 7.
The follow-up compact release is `windows-quick-test-20261002-issue10` in the
fork, with all eleven targets and a separate exact GPL corresponding-source
asset. Native network arithmetic and conservative scheduling are unchanged.

These tests do not validate the reporter's gfx1200 kernels or game image.
The local Cyberpunk EXE is `D:\Games\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe`;
it is located but is not launched or modified while the user is away.

2026-10-02 quick-test packaging: one universal ZIP shares the identical
OptiScaler/ZLUDA/HIP runtime and selects among all eleven previously compiled
host/kernel targets using read-only HIP discovery. START-K.cmd / START-M.cmd
use file pickers, remember validated game/DLL paths, back up replaced files,
run synchronous output diagnostics, and produce one report ZIP. The initial
release attached a debugger; the issue #10 follow-up above removes that default.
RESTORE-GAME.cmd works without HIP discovery or NVIDIA DLL selection. No
runtime arithmetic changes or additional hardware-support claims are made.
The short English instructions are in `windows-quick-test.txt`; developer
sources and exact patched GPL OptiScaler/submodule sources are a separate asset.
An isolated Windows PowerShell 5 fixture verifies automatic gfx1201 selection,
spaces/Unicode paths, remembered selection, exact install/restore, and rejection
of wrong NGX/DLSS DLLs, vkd3d and modified package files before game writes.
No real game is launched by these packaging checks. Reproduce packaging with
`package-windows-quick-test.ps1 -PackageRoot <new-dir> -ArchivePath <zip>` and
`tools/windows/package_quick_test_sources.py --output <source-zip>` after commit.
The quick-test build is a correctness diagnostic, not the async/private-texture
performance configuration; K optimization remains open.

2026-10-02 coverage: all eleven listed RDNA3 / RDNA3.5 / RDNA4 targets compile
with separate host/device objects and guarded packages. Only RX 9070 XT has
physical validation here; no other-target GPU workloads are executed. The
newest realdody correction informs a Windows-only gfx11 LDS half-wave exchange
preserving native gfx11 WMMA layout. gfx12 and Linux keep their original
instruction path. A shared registry drives build, runtime, ELF checks and
scripts. HIP selection matches the D3D12 LUID before selecting an ordinal,
including same-target multi-GPU systems.

Issue #10 comment 5952822115 contains 32 logs / eight attempts: all discover
RX 9060 XT / gfx1200 but reject the old gfx1201 package before NGX runtime
initialization. Three also load a vkd3d-proton D3D12 proxy. The installer now
catches these conditions before changing game files and prints one diagnostic
ZIP. The next community gate is the gfx1200 package on native D3D12; those logs
do not establish a native K/M crash. Coverage, sources, fork credit, commands
and hardware limits are in [windows-gpu-support.md](windows-gpu-support.md).

The rebuilt gfx1201 passes 20/20 CTest gates on RX 9070 XT. K completes twelve
finite 4K burst/recreation frames at 2259x1271 input, with all six async RGB
images exactly matching sync. M completes eight finite control/candidate 4K
frames, with all four candidate RGB images matching the prior build exactly.
Both log every required native layer, zero CPU frame copies and frame age 0.
All 209 compiled objects have the correct ELF target. For all nineteen gfx1201
objects, instruction/constant sections match the previous validated build.

All eleven game directories stage strict K/M baselines without NVIDIA DLLs.
An isolated gfx1201 install/restore fixture preserves original bytes; wrong
gfx1200-on-gfx1201 and dummy vkd3d-proxy fixtures refuse before game changes
and produce one diagnostic ZIP each. The real game installation is untouched.
Results: `test-results/windows-gpu-coverage`; build logs and the compile-only
matrix report: `build/windows-gpu-coverage`. No physical multi-GPU, APU or
other-target validation is claimed. K performance remains open.

The final translator gate exposed a separate dependency gap: pinned LLVM 22
rejects `gfx1154` and aborts (`0xC0000409`) on a public PTX pattern. A minimal
source backport of official LLVM `7a0829e41228` adds the processor's real
11.5.4 features, parser and ELF identity. It is exported under
`patches/zluda-llvm`, applied idempotently by the LLVM builder and included with
its base/hash in package metadata. LLVM and ZLUDA are rebuilt; all 33 offline
pattern/F16-reference/WMMA compilations pass across the eleven targets without
GPU execution. The existing strict gfx12 F16 reference lowering remains intact.
The rebuilt ZLUDA passes 20/20 RX 9070 XT CTest gates. Its twelve further K
4K burst/recreation frames are finite and all six async RGB images are exact;
three sync RGB images also exactly match the previous compiler's captures.
Four further M 4K frames are finite and match all four previous-compiler RGB
images exactly. Both require native transformer hits, zero CPU frame copies
and frame age 0. Results: `k-new-zluda`, `m-new-zluda` and compiler comparison
logs under `test-results/windows-gpu-coverage`. Other GPUs remain unverified.

**Current milestone: K performance remains open.** The latest unprofiled 4K
Silent Hill 2 scene capture records 65.250 FPS. Functional K/M and the
reproducible archive do not close this gate. Prioritize measured K kernel cost
and retain queue/synchronization checks; postpone further M/FP8 optimization.
The user's approximately 80+ FPS with FSR4 is not a captured same-scene control.

The preceding unprofiled K/4K capture records **64.661 FPS** with direct AMD
ADLX mean usage **93.183%**, core clock **3124.667 MHz** and board power
**316.717 W**. The user identified the previous 63% reading as Task Manager,
which is distinct from ADLX's device-wide metric. This capture places GPU kernel cost
at the center of the next K investigation, while retaining queue measurements.

The shared-V enc0 + private WGP output candidate records **65.250 FPS** over
1954 presents in the user-confirmed stationary scene, at the same
2259x1271 -> 3840x2160 dimensions. ADLX mean usage is **93.500%**, core clock
**3118.400 MHz** and board power **318.817 W**. The 0.589 FPS historical
comparison is too small to establish a reproducible gain; a fresh repeated
game control remains necessary before selecting WGP by default. The bounded
ten-minute run completes **35664 K frames / 427968 native launches**, with
zero backend errors, CPU image copies, previous-frame outputs or recorded
crashes. Output scans are disabled in this unprofiled game run; standalone
finite/exact-image gates supply its numerical coverage. Results:
`test-results/silent-hill2-k-vshare-wgp-4k`. See
[windows-performance.md](windows-performance.md#combined-k-game-check).

Native replay benchmarks now compare fresh exact allocations before timing,
support event/dispatch profiling and paired batches, retain independent host
completion intervals, and reject invalid or incomplete timings. Eleven K
same-module controls pass both timing modes. Real K/4K enc0/enc1/dec0 captures
and CU/WGP, cache, token-tiling and packed-storage experiments are recorded in
[windows-performance.md](windows-performance.md#controlled-native-replay-timings).
Packed storage lowers enc1 VGPRs but does not speed it up. Enc0's shared-V
implementation preserves every tested output and gives about a 5% isolated
layer saving at both small and 4K workloads; the strict gfx12 baseline now
selects it. Other K layers, M and arithmetic remain unchanged. The previous direct-F16 experiment's
saved-control selection was corrected to the current strict-normalization
baseline; it remains rejected. K performance is still an open gate.

The private output-tail WGP candidate gives a lower event median in both
control-first and candidate-first K/4K tests, with exact RGB across 96 finite
control/candidate frames. This is a separate-process diagnostic result, not
game FPS. ZLUDA source `fb0adc8714f57647320f0601533ec50331e378b0` is rebuilt
cleanly in `dist/zluda-windows-wgp-experiment`; optional patch 0018 preserves
CU defaults and wave/arithmetic semantics. All seventeen CTest gates pass
with defaults, and all six ZLUDA gates also pass with WGP explicitly enabled.
All seventeen patches (0002..0018) apply to clean base `ee2f25a`.

Enc0 full-frame controls pass at 1920x1080 and the game's 2259x1271 input,
both outputting 3840x2160: 28 frames finite, all 14 candidate RGB images exact.
The exact CMake enc0 object passes a fresh replay/paired benchmark; the other
fifteen ordinary K/M objects retain their prior SHA256. The combined enc0 +
WGP-output async/burst/recreation gate passes twelve further game-sized frames
with all six async RGB images exact (`k-enc0-shared-wgp-async-game-input`).
`-InputResolution` exposes actual render dimensions without changing the
harness's output bounds. Empty task environment variables are removed from
probe/game children so disabled diagnostic flags stay disabled; GPU comparisons
now require actual valid timing samples. Commands, rejected experiments and
the initial diagnostic/cache failures are recorded in `windows-performance.md`.

A subsequent input-rounding CU/WGP experiment completes eighty finite frames
with all forty candidate RGB images exact, holding the other thirteen objects
identical. Neither input candidate improves standalone timing; the game keeps
translated input. The private recipe/test tools retain these reproducible
experiments, while the installer still accepts only the two validated output
variants. Rebuilt output objects retain their SHA256 values, and the ordinary
translated-output control comparison passes four further finite frames. See
[windows-performance.md](windows-performance.md#rejected-k-input-rounding-overrides).

The next diagnostics add `-ProfileCudaApi` (ZLUDA patch 0015): aggregate host
CUDA API durations without HIP-event waits. Boundary stages now separate
prepare, copy-list setup, input/output submission and output-copy drain.
Feature-owned input/output command lists and allocators are reused only after
the existing completion fence; `-UncachedInteropLists` restores per-frame
allocation for an A/B control. Submission drain fences/events are queue-owned
and reused. No completion dependency or current-frame output is removed.

The opt-in `-AsyncInterop` experiment reserves D3D12 input/output dependencies
at submission, runs CUDA on a feature worker, and retires owned suffix command
memory on a separate fence-completion service. Worker order is shared across
features using the context; this experimental runtime requires one D3D12 queue.
Three reusable copy contexts bound outstanding ownership. Every consumer waits
for **its own** output; no previous-image selection or CPU image staging occurs.
Steady-state submission uses immutable allocation metadata and per-frame
resource/parameter snapshots, avoiding the CUDA mutex. Resize/reimport and
feature creation/release drain D3D12 outside that mutex, keeping workers able to signal
already-published waits. Worker failures poison the context and are logged.

First async game run: 9481 K/4K frames and finite GPU scans, zero backend failures;
the user reports **60 FPS / 63% GPU**. Median command submission drops from
10.512 ms (synchronous control) to 2.676 ms. The next mutex-free candidate
submits in median 0.181 ms, but its initial game test exits with UE
`DXGI_ERROR_DEVICE_HUNG` during resource/feature recreation after 366 finite
frames. Its crash context and dump are in
`test-results/silent-hill2-k-async-nolock-4k/ue-crash`. Do not classify this as
success or enable the experiment by default until the game reconfiguration gate
passes. The new standalone burst + Release/CreateFeature mode reproduces the
GPU hang without the game. Retaining the d4r Feature through completion does
not retain OptiScaler's caller-owned root-CBV buffers: OptiScaler frees its
shader backing immediately after NGX ReleaseFeature returns. ReleaseFeature
now waits for pending workers, copy contexts and all submitted D3D12 consumers
before returning, outside the CUDA mutex. The fixed four-way regression
(sync/async x burst/recreate) completes twelve K/4K frames; all six async RGB
outputs exactly match their corresponding fresh synchronous controls, with
no NaN/Inf or GPU hang. Results:
`test-results/k-async-release-drain-4k/validation.json`. The fixed game retest
on 2026-10-02 completes **16941 K/4K frames**, 16941 finite GPU scans and
203292 native launches, with zero backend failures/CPU image copies/previous
frame outputs. It includes two feature creations and color-format reimports;
the previous recreation crash does not recur in this five-minute run. The
user reports **62 FPS / 63% GPU** in the comparison scene. Median command
submission is 0.197 ms; DLSS worker NGX completion is 4.400 ms and output
scan is 0.380 ms. Logs:
`test-results/silent-hill2-k-async-release-fix-4k`. The runner deliberately
stops at its bound (`diagnostic_timeout`); there is no recorded game crash.
The K performance gate remains open.

Optional `-ProfileGpuBoundary` now records D3D12 GPU timestamps around input
copies, the external-fence span and output copies. Copy-slot completion
controls query reuse/readback; only 32 timing bytes are read, with no extra
completion wait. The timestamp-enabled burst/recreate gate passes all twelve
K/4K frames with exact candidate RGB (`test-results/k-async-gpu-boundary-4k`),
and all sixteen CTest gates pass on the rebuilt shim. A 9384-frame game run
measures median input/output copies at 0.061/0.208 ms and external span at
5.586 ms, with zero backend failures. The span includes scheduling as well
as HIP. PresentMon 2.6.0 capture/report tooling is added; its first trace
contains changing presentation modes and is not a controlled capture of
the user's 62 FPS scene. See `windows-performance.md` for exact metrics and
the initial capture limitations. A subsequent user-confirmed foreground-scene
capture completes 1902 presents at 63.574 FPS, all Independent Flip / SyncInterval
0. PresentMon mean GPU busy/wait is 14.504/1.110 ms; this is not sensor
utilization. The containing ten-minute run completes 36591 K frames without
backend errors; median external span is 5.399 ms. Log bundle:
`test-results/silent-hill2-k-steady-profile-4k`.

The next candidate adds bounded, deferred HIP-event sampling and opt-in batched
GPU input conversion/array copies. It retains one all-stream input completion
before NGX and every D3D12/output dependency. All seventeen CTest gates pass,
including asynchronous pitched FP16/FP32 array copies and guard checks.
The deferred profiler's tiny PTX diagnostic verifies 516 launches and safe
default-stream skipping; negative HIP elapsed times are retained as invalid
samples, with cause unresolved. This DLSS K binary uses the legacy default
stream, which needs explicit opt-in for timestamp sampling. The depth/stencil
and packed-format burst/recreation gates pass 24 full K/4K frames: all twelve
candidate RGB images match their fresh unbatched synchronous controls exactly,
with finite RGBA. Results: `test-results/k-batch-deferred-4k` and
`test-results/k-batch-packed-4k`. The candidate remains
disabled by default; see `windows-performance.md` for semantics and commands.

Current candidate source is ZLUDA `67127dde599879ada022d6439383d682e09e0b36`
in `dist/zluda-windows-deferred-profile`, containing patches 0016/0017.
All sixteen patches (0002 through 0017) apply to clean pinned base `ee2f25a`.
The committed ZLUDA is rebuilt with a clean worktree and the four-way K/4K
gate is repeated: twelve frames pass, all six candidate RGB outputs match
their controls exactly, and all six async boundary timestamp records survive
slot reuse/release. Results: `test-results/k-batch-committed-source-4k`.
The same negative HIP event-timing issue reproduces with stable SDK 7.2 as
well as TheRock; PTX output/guards remain correct. It also affects a native K
dec4 timing sample, so valid-looking events alone are not a timing-accuracy
gate. No hardware utilization or kernel speedup is inferred from these samples.

The committed batched-input game run completes **35673 K/4K frames** without
backend errors, CPU image copies, previous-frame output or recorded crashes.
The user-confirmed 30-second PresentMon window records **64.265 FPS** (1925
presents), compared with the prior unbatched 63.574 FPS. Both keep Independent
Flip / SyncInterval 0 and 2259x1271 -> 3840x2160. This small single-pair
difference is not a repeatable speedup gate. Whole-run median input preparation
falls from 0.482 to 0.332 ms, while median external span is 5.438 versus
5.399 ms. Results: `test-results/silent-hill2-k-batch-inputs-4k`.
K performance remains open and batching remains opt-in. The next diagnostic
adds a separate read-only AMD ADLX usage/clock/power capture alongside
PresentMon. Its local RX 9070 XT smoke test passes, retaining unsupported
metrics as absent and preserving the driver's raw timestamp. See
`windows-performance.md` for build/capture commands and SDK packaging limits.
The combined capture smoke test passes, followed by the user-confirmed
1935-present/60-poll measurement recorded at the top of this document. Logs
and runtime/DLL hashes: `test-results/silent-hill2-k-telemetry-unprofiled-4k`.
Its ten-minute game summary completes **37187 K/4K frames**, zero backend
errors/CPU image copies/previous-frame outputs or recorded crashes.

The next isolated K enc1 experiment replaces FP32 WMMA + FP16 rounding with
gfx12's packed FP16 accumulator instruction. ISA conversions and VGPRs fall,
but the real output changes and reference PSNR falls. A 64-matrix public GPU
reproducer shows the direct instruction differs from FP32 + FP16 rounding
already at a single K16 step while the adapter is exact against the packed
instruction. The experiment is rejected at enc1; no subsequent layer or game
uses it. Baseline preservation takes priority over the instruction reduction.
`D4R_K_F16_WMMA` remains disabled in every normal build. Reproducer/results and
exact metrics are in `windows-performance.md`.
All sixteen ordinary K/M objects remain byte-identical after rebuilding with
the disabled experimental code. All seventeen CTest gates pass; gfx1101 enc1
compile-only compatibility and the ready-made public WMMA script also pass.

The public prerelease `windows-rdna4-dev-20261002-a346d76` is published to
`xdfnx-dev/d4r`, containing source commit `a346d76` and ZLUDA `a1c506f`.
Archive: `d4r-windows-rdna4-20261002.zip`, 88893567 bytes, SHA256
`5b9a47e9d0c9a042567c2096eb8eaafb0a28473614b656865d794a39f5165d8e`.
All 65 entries and the 274-entry source snapshot match the manifest; GitHub's
uploaded asset digest matches the local ZIP. NVIDIA DLLs/private kernels and
captures are excluded. This is a development artifact, not completion of
the K performance gate.

Three ordinary K/4K async frames and three queued together without intermediate
CPU completion all match their corresponding prior synchronous RGB exactly;
the temporal controls differ between frames. One burst attempt times out during
NGX creation before submitting a frame after the game GPU crash; HIP/D3D12
sanity probes pass and the 600-second retry completes and matches all frames.
`test-async-k.ps1` generates fresh synchronous controls and checks burst and
Release/CreateFeature with pending work, rather than depending on saved fixtures:

```powershell
.\scripts\windows\test-async-k.ps1
```

New game crashes are automatically copied from the current Silent Hill 2 UE
crash directory into the diagnostic bundle, including its error reason and
minidump; proprietary/private artifacts remain excluded from public packaging.

The profiling candidate is ZLUDA
`a1c506fc956cfc347c56c44311669700c0d500ac` in
`dist/zluda-windows-api-profile`. Patch 0015 adds optional per-thread host API
aggregates without extra synchronization; ordinary runs leave it disabled.
All fourteen patches (0002 through 0015) apply to pinned base `ee2f25a`.
All sixteen CTest gates also pass with `D4R_ASYNC_INTEROP=1` after the release
fix. The previously archived synchronous package described below remains a
separate artifact; it does not contain this async scheduling change.

Upstream refreshed and merged on 2026-10-02 through `aa447b2` (experimental L,
Linux resource lifetimes and event waits). Its Linux-only runtime helpers do
not enter the native Windows build; K/M native sources and the Windows
backend are unchanged by that update. Windows L remains outside the validated
manifest and game runner. The prior refresh on 2026-10-01 incorporated
[`f0d1a65`](https://github.com/countervolts/d4r/commit/f0d1a65e27aff6cbe7eeaa13227c80facb9056ab)
and artifact fix `65dfc9f`. Windows retains strict f16 rounding and the exact
M attention baseline. Upstream's optional denormal override now drops only
FP32 requirements, preserving FP16/FP64. All sixteen rebuilt native K/M
objects have the same SHA256 as the previously validated objects. All thirteen
ZLUDA patches apply to the pinned fresh base; all sixteen CTest gates pass.
Final ZLUDA source and binary are `2378af72e586e1730d15f75e0322daab674f0862`,
built cleanly into `dist/zluda-windows-final`; nvcuda SHA256 is
`ccc700974fa5d0b9a5f5c39e617c17ca22f34701d40f745f2f96f039a75b22f8`.
All sixteen CTest gates pass on it. Three K/4K HDR frames and four M baseline
frames remain bit-exact to their prior controls, all RGBA finite. Historical
performance runs below used `b0161a4` with the denormal override unset.
Package metadata records the binary build's actual commit.

The final-runtime M game test completes 1940 4K frames and 1940 finite GPU
scans, with 19410 native launches (including a partial final frame), zero
backend failures/CPU image copies/previous-frame outputs. Native M FP8 also
passes all logical output/merge values from forty NumPy-validated captures,
plus four 512x288 and four 4K complete frames. Its original software packing
regresses performance; the opt-in hardware packing preserves all 254 finite
FP8 encodings and all forty captures, with exact RGB in four additional 4K
frames. Paired in-process event tests confirm reduced packing cost, but the
full-network measurement does not establish a win over FP16. Production keeps
the baseline. See [windows-performance.md](windows-performance.md) for timings
and one-command replay/paired profiling.

Final production-mode K/4K verification completes 1838 frames and 22056 native
launches with the final runtime and locally validated output stores, zero
backend failures/CPU image copies/previous-frame outputs. Debugger, hook/stage
profiling, serializing kernel events and GPU output scans are disabled in that
run. The 60-second diagnostic is stopped deliberately by its process job;
its recorded `diagnostic_timeout` is not classified as a crash. Output accuracy
and finiteness are covered separately by the matching-runtime reference gates.
The clean-source public archive passes every hash check: 64 manifest-listed
entries, including a snapshot of 258 tracked source entries, no NVIDIA DLLs or
private artifacts. `test-results/final-package-validation.json` and the ZIP's
SHA256 sidecar record the exact final artifact identity.

Current native Windows build, after the pinned tool/source setup described
below (MSVC v143 / Windows SDK 10.0.26100 required for OptiScaler):

```powershell
.\scripts\windows\build-zluda-windows.ps1
.\scripts\windows\build-windows-rdna4.ps1 -RuntimeProfile therock -ZludaRoot "$PWD\dist\zluda-windows-final" -InstallDirectory "$PWD\dist\windows-rdna4-command-list"
.\scripts\windows\stage-native-k.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -PackageRoot "$PWD\dist\windows-rdna4-command-list"
.\scripts\windows\stage-native-m.ps1 -DlssDll "$PWD\nvngx_dlss.dll" -PackageRoot "$PWD\dist\windows-rdna4-command-list"
.\scripts\windows\build-optiscaler-windows.ps1
& .\.tools\python\cmake\data\bin\ctest.exe --test-dir build/windows-rdna4-therock --output-on-failure
.\scripts\windows\package-windows-game.ps1 -ArchivePath "$PWD\dist\windows-rdna4-game.zip"
```

The archive requires committed source. It includes a tracked-source snapshot,
all runtime file hashes/versions, dependency licenses and source patches, and
excludes local NVIDIA DLLs, PTX/weights/captures and private texture objects.
The optional texture optimization is built/validated locally and supplied with
`-LocalTextureKernels`; it is never folded into the public source snapshot.

Additional production-style runs: M at 1920x1080 completes 4833 frames with
4833 finite-output GPU scans and no backend failures. K at 3840x2160 completes
4685 frames, 51535 native launches and 4685 finite-output GPU scans. Both use
current-frame VRAM interop, no CPU image copy and no previous-frame fallback.
The user confirms visible imagery at 4K but reports low FPS and about 25% GPU
usage. This percentage is not an isolated DLSS/WMMA hardware measurement.

The game runner now launches without an attached debugger by default, while
retaining a kill-on-close process job, raw stdout/stderr, exit code, loaded
module inventory and frontend/driver logs. `-CaptureExceptions` enables the
original attached debugger and minidumps. A K/4K run without it completes
5774 frames / 63514 native launches / 5774 finite-output scans, zero failures.
The user reports 38-42 FPS. Recording/replay measurements: average split setup
0.284 ms, replay 0.025 ms, submitted texture-state publication 0.029 ms,
suffix completion 0.405 ms. These do not explain tens of milliseconds of
missing performance. NGX recording intervals average 27.474 ms, median
28.466 ms; they are not a DXGI Present measurement. The first debugger control
never left the startup screen and is explicitly not a valid FPS comparison.

K output bottleneck: the 2961-frame 4K profile measures its translated output
tail at 4.401670 ms, larger than all eleven native transformer layers combined.
The Windows texture build preserves upstream Accuracy mode and creates private
NVIDIA-derived output objects. Both LDR and the game's HDR/inverted-depth/low-MV
variants pass strict full-frame comparisons: eight 512x288 frames each, then
three 3840x2160 frames each, max absolute/relative RGB error 0 and all RGBA
components finite. Controlled 4K tail GPU means: LDR 2.907823 -> 1.803233 ms;
HDR 2.978547 -> 2.088347 ms. The 6221-frame game profile executes the native tail
every frame, with zero backend failures and mean tail time 1.801918 ms. Scene
differences preclude an exact game FPS speedup claim. `-LocalTextureKernels`
requires matching source DLL, object and manifest hashes after validation;
public ZIPs exclude these objects. Install/restore with private overrides
restores sentinel DLL/config byte-exactly, without duplicate slash/backslash
manifest paths.

Production K/4K with private output stores completes 9982 frames and 9982
finite GPU scans; the user reports 43-47 FPS. Random per-hook CPU sampling
then localizes contention to the shared command-list routing map. A per-thread
weak lookup cache bypasses that mutex for repeated calls, invalidating on map
mutations to preserve Reset/final Release/address reuse. Queue and per-list
synchronization remain. The regression resets/re-splits the same object twice
per iteration with different root constants and verifies both GPU outputs.
All sixteen CTest gates pass. The cached game run completes 4770 K/4K frames
and 4770 finite GPU scans, zero failures/CPU image copies/aged outputs. The user
reports 49-51 FPS and 53% GPU usage. Aggregated sampled access/capture time
falls from 977.528 to 233.606 ms/s across threads; it includes parallel lock
waits and must not be treated as serial frame time. Median recording interval
is 19.473 ms. `-ProfileCommandHooks` is optional and off during ordinary play.
Detailed scope and timings are in [windows-performance.md](windows-performance.md).
Post-cache full-pipeline checks also pass: three K/4K HDR frames and four M
512x288 frames match their previously validated controls with max absolute and
relative RGB error 0, all RGBA finite. Public archive validation checks all 63
entries and their SHA256 against package.json; NVIDIA DLLs and private texture
objects are absent. `package-windows-game.ps1 -ArchivePath <zip>` archives only
the staged manifest, including on Windows PowerShell 5.1.

Performance diagnostics (2026-10-01): the separate ZLUDA `b0161a4` runtime
passes 32 PTX launches in each context and all four native K/M frames with
bit-exact RGB against their controls. All thirteen exported patches (0002
through 0014) apply to the pinned fresh base. The full Windows build and all
16 CTest gates pass. Actual K profiling completes 2345 frames with zero backend
errors. Replacing PowerShell's line-oriented stdout/stderr collection with
concurrent raw byte copies removes substantial diagnostic backpressure:
mean Evaluate host time drops from 25.715 to 3.350 ms despite per-kernel HIP
event synchronization. Native/translated GPU times, scope and limitations are
in [windows-performance.md](windows-performance.md). Quiet successful API
logging is separately selectable via `D4R_QUIET_API=1`; error checks stay active.

An additional K game run (`silent-hill2-k-output-stages.zip`) completes 1688
frames / 18568 native launches / 1688 GPU output scans, all finite, with no
backend failure or previous-frame output. The user confirms visible imagery
after the depth-plane fix; low FPS is the current investigation.

Actual K rendering: `silent-hill2-k-depth-plane.zip` completes **1794 DLSS
frames / 19734 native K launches / zero d4r failures** at 753x424 -> 1280x720
with the game's depth-inverted HDR path and DXGI-19 depth plane. The captured
menu is rendered correctly; no CPU image copies or frame-age fallback are
used. Its restored graphics settings match the original SHA256.

GPU output validation is now available via `D4R_VALIDATE_OUTPUT=1` or the
game runner's `-ValidateOutput`. A native GPU scan classifies all canonical
RGBA16F components, returning only two 32-bit NaN/Inf counters. A negative
fixture with one NaN and two infinities passes. All four K/M standalone
validated frames have zero NaN/Inf and bit-exact RGB against their control
frames (`optiscaler-k-output-validation.zip`, `optiscaler-m-output-validation.zip`).
The pixel format tests including the negative fixture pass legacy/enhanced
barriers (`output-validation-formats-ctest.log`).

`silent-hill2-m-output-validation` now renders the actual saved gameplay
level with all five native M transformer layers. A screenshot shows the
character and street scene; 1349 full 1280x720 output scans have zero
NaN/Inf in all 3686400 components. There are 1348 complete frame log records,
13490 native launches and zero d4r failures; the bounded stop interrupts the
last frame log after its successful scan. The runner now counts only complete
records and reports validation counts and per-kernel native/translated hits.
The current-frame backend remains slow:
observed interop + NGX CPU completion is roughly 74–86 ms in these runs.
These are initial stage measurements, not isolated kernel GPU timings or
a controlled K/M performance comparison. Profile before optimizing.

First actual game runs (2026-10-01): Silent Hill 2 v1.1.258834 (the GSA settings
schema reports 1.2.0.0), XeSS input -> OptiScaler
DLSS K. `silent-hill2-k-first.zip` found the old Project ID wrapper initializing
CUDA NGX with app 4919 instead of the caller's identity, and back-buffer
tracking references blocking `ResizeBuffers` (DXGI_ERROR_INVALID_CALL).
OptiScaler patch `0003` preserves identity; resource-state records now follow
the public D3D12 caller-owned resource lifetime instead of retaining barriers'
resources. A composition swap-chain regression resizes while its submitted,
unreset command list remains alive; it passes on Radeon and WARP, along with
all 16 CTest gates (`game-resize-ctest.log`). K/M legacy Project ID standalone
frames have bit-exact RGB. The invalid-core test propagates the error from
legacy Init rather than pretending initialization succeeded.

`silent-hill2-k-project-resize.zip` successfully initializes CUDA DLSS and
presents thousands of startup frames with no d4r failures. It has **zero
completed DLSS frames** and is not a game acceptance pass. The next interactive
run reaches native K feature creation at 2259x1271 -> 3840x2160 after Enter;
the user reports a black screen during this phase. Capture and isolate this
before claiming successful in-game execution.

`silent-hill2-k-720p.zip` localizes the flashing “upscaler failed to run”
message to the actual depth resource: DXGI `R32G8X24_TYPELESS` (19). The
backend previously rejected it before executing CUDA Evaluate. Implemented
its D3D12 depth copy plane as R32; it is not an interleaved 64-bit texel.
The [public planar depth/stencil specification](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/PlanarDepthStencilDDISpec.md)
defines this plane mapping. `GetCopyableFootprints` is checked against R32
and width * 4 bytes; copies/barriers affect subresource 0 only. All 4403 depth
pixels match an independent D3D12 typed SRV exactly in each of two legacy and
two enhanced runs. Full build / CTest 16/16 pass
(`d32-stencil-full-ctest.log`). The NGX runner supports
`-PixelProfile depth-stencil` for full-network validation.
`optiscaler-k-depth-stencil.zip` and `optiscaler-m-depth-stencil.zip` both
pass four full frames, with all 44/40 native transformer launches and bit-exact
RGB against the respective non-stencil depth baselines.

The local game package is `dist/windows-rdna4-game`, produced by
`scripts/windows/package-windows-game.ps1`. Its native manifest combines all
11 K + 5 M objects; NVIDIA binaries are excluded. `windows-game.ps1` backs up
replaced files, verifies package hashes and the local DLSS identity, sets the
native runtime environment, and captures stdout/stderr, external debugger,
OptiScaler, exception and version/driver logs. Install/restore regression
restores original DLL/config sentinel bytes exactly. For Silent Hill 2,
`-DiagnosticResolution 1280x720` backs up and restores its two graphics config
files; this avoids modifying progression saves. Original settings after the
stopped failing run have an identical SHA256. `-CacheDirectory` can reuse an
already validated local JIT cache. Development package usage is documented in
`docs/windows-game.md`; K menu and M saved-level results are recorded above.

Startup coverage (2026-10-01): OptiScaler patch `0002` installs d4r at device
discovery, before NGX initialization. `optiscaler-k-preinit-signature.zip` and
`optiscaler-m-preinit-signature.zip` create a dispatch-only command signature
before NGX Init, execute it in the evaluation list, and run four full frames.
Both use all native transformer layers (44 K / 40 M launches), with bit-exact
RGB against their control frames. K uses inherited legacy states; M uses
inherited enhanced layouts. The full build and all 16 CTest gates pass. Reproduce
with `-EarlyIndirectProbe` in the standalone OptiScaler runner. This avoids
the previously documented unknown-signature rejection for observed startup
objects; signatures created before the frontend is loaded remain untracked.

Current milestone: M0-M9 functional gates pass on the tested Windows GPU/game.
The package uses FP16-equivalent K/M and conservative denormal handling.
Native NGX Create/Evaluate on the simple
requested E path passes with the user's DLLs. All 11 K layers now pass synthetic
and real-weight NumPy/PTX replay checks; native K Evaluate produces four finite
frames on Windows. M now passes all five synthetic and real-weight layers against
NumPy and independent PTX, and four native CUDA frames. The standalone D3D12
harness now produces four K and four M frames using imported VRAM and shared
fences, with bit-exact RGB agreement against the corresponding CUDA harnesses.
Silent Hill 2 K/M integration passes, including current-frame 4K VRAM interop.
The ordinary NGX D3D12 Evaluate API now
records a boundary in an open command list; a native queue interception backend
executes prefix, CUDA evaluation, then suffix in the same submitted frame.
A full temporal M scan
found rare attention P/V accumulation mismatches that the earlier twelve-block
sampling missed. `SWIN_EXACT_PV` fixes the native baseline; ZLUDA patch `0013`
adds the corresponding opt-in f16 reference accumulation. All spatial blocks in
40 actual native M launches now exactly match NumPy. Four complete native M RGB
frames exactly match fully translated M with the corrected reference option.
Do not integrate NGX until the integer PTX workload is stable on the real GPU.

| Gate | Status |
| --- | --- |
| M0: branch, audit, Windows build and diagnostics | Initial implementation complete |
| M1: native HIP gfx1201 allocation, kernel, CPU verification | PASS, 32 iterations + guard verification |
| M2: Windows ZLUDA integer PTX, primary and created contexts | PASS, 32 iterations each + guard verification |
| M3: Windows NGX initialization and simple DLSS path | Init, SR capabilities, Create/Evaluate PASS; E, K and M CUDA harnesses execute |
| M4: D3D12 / HIP external memory and fence round trip | PASS with TheRock; stable 7.2 has mapped-view leak |
| M5: independently validated gfx12 WMMA backend | PASS: raw + legacy adapter + upstream layout, max abs/relative error 0 |
| M6: K layers, full transformer and image validation | PASS: 11 real-weight layers, 44 native launches, four D3D12 frames bit-exact against native CUDA, 81.85–92.91 dB versus translated K |
| M7: M FP16-equivalent baseline and full transformer | PASS: all blocks of 40 native launches exactly match NumPy; targeted independent PTX regression matches; four D3D12 frames bit-exact against corrected translated M |
| M8: standalone Windows D3D12 NGX harness | PASS for ordinary recorded EvaluateFeature and explicit boundaries: K/M, four frames each, no shim CPU image copies; both bit-exact against explicit boundary baselines |
| M9: OptiScaler integration, profiling, installation | Patched frontend and game K/M PASS at 4K with finite output; native/translated/FP8 profiling, reversible install and manifest-only public archive PASS; K user reports 49-51 FPS |

## Public NGX identity and OptiScaler gate (2026-10-01)

The Windows shim now implements D3D12 GetFeatureRequirements and forwards
Init_ProjectID to CUDA Init_ProjectID with the caller's project/engine/version.
The runtime owns the feature path strings and pointer list. The physical HIP
gfx1201 architecture and full DXGI adapter LUID determine SR support; other
features remain unsupported. Numeric driver initialization retains the tested
three-argument CUDA Init ABI. `-NgxAbi project` selects the public project path
in both CUDA and D3D12 probes. Capability parameter allocation is transactional.

`ngx-project-api.zip` passes direct CUDA project initialization.
`d3d12-k-project-api.zip` and `d3d12-m-project-api.zip` pass four recorded
enhanced-barrier packed-format frames with 44 K / 40 M native launches.
All RGB values exactly match the independently quantised earlier baselines,
max absolute/relative error 0. Full rebuild and CTest pass 16/16 gates
(`project-api-ctest.log`).

The runner now accepts `-OptiScalerDll <absolute-path>`, creates a private
OptiScaler.ini and captures the frontend log. Native-network hit validation is
mandatory for the integration runs. The official 0.9.4 release silently chose
FSR despite DLSS=enabled on AMD: `optiscaler-m-first.zip` is correctly **FAIL**
with zero native launches. Current nightly 20261001 also gates DLSS on a
System32 NVIDIA driver; preloading our NVAPI triggered access violation
0xc0000005 in its loader hooks (`optiscaler-m-nightly-first.zip`). This is not
a DLSS success. The probe no longer preloads NVAPI before OptiScaler startup.
Crash capture avoids module resource enumeration in the exception filter,
guards reentrancy and records the minidump's module list instead.

A local OptiScaler source fork starts from
`45a2001303ddff632e279f77aef85ceede5832cb`. Its explicit
`[DLSS] AllowExternalBackend=true` option requires an absolute NvngxPath with
the `d4r_WindowsBackendVersion` ABI-1 marker; it does not spoof HIP architecture.
The marker is GPU-free and safe during discovery. The patch lets the external
backend load its own NVAPI provider through the original loader and preserves
the exact locally supplied CUDA NGX core path from frontend redirection.
Physical adapter capability is queried outside DLL startup instead of using
NVIDIA architecture as a proxy. GetFeatureRequirements spoofing is bypassed.
Explicit external DLSS selection returns errors instead of silently using FSR.
Microsoft Build Tools 17.14 / MSVC 14.44.35207 and SDK 10.0.26100 were installed
from the signature-verified official bootstrapper without restarting Windows.
At this historical frontend milestone no game files had been modified; the
later game installer and reversible file backups are described above.

The OptiScaler fork commit is `33bbac2` on `windows-rdna4-d4r`. Its reproducible
source patch and build instructions are in [patches/optiscaler](../patches/optiscaler/README.md).
`build-optiscaler-windows.ps1` produces a DLL, PDB, GPL license and dependency
metadata from pinned source/submodules; it disables upstream's unrelated
packaging post-build commands. The public D3D12 probe now supplies and submits
a real open creation command list, as required by OptiScaler's API.

`optiscaler-k-final.zip` and `optiscaler-m-final.zip` pass four complete
OptiScaler -> Windows d4r -> CUDA NGX -> ZLUDA/HIP evaluations on RX 9070 XT.
They contain 44 K and 40 M native transformer launches, respectively. Every
RGB value exactly matches the independently quantised direct-d4r baseline;
max absolute/relative error is 0. No FSR evaluation, NaN/Inf, shim CPU image copy
or previous-frame output is observed. The invalid-core regression intentionally
passes the shim as the CUDA core: it fails with an explicit frontend
`refusing FSR fallback` message and zero FSR evaluations, as required.

The loader crash was localized with a PDB to Util::IsSubpath calling
filesystem::relative/weakly_canonical from LdrLoadDll interception. The patched
check is lexical, case-insensitive and handles empty relative paths without
filesystem I/O. The separate `d4r_debug_launcher.exe`, enabled by
`-CaptureExceptions`, captures minidumps, first/second-chance exception codes,
module load/unload addresses and raw stack slots outside the hooked process.
It propagates the child's exit code and terminates its child on runner timeout
via a process lifetime job. `optiscaler-m-source-debug2.zip` contains the
successful dump capture and `optiscaler-source-stack-symbols.log` the source
attribution. The ordinary in-process crash filter remains available too.

## Initial PSO and enhanced texture layouts (2026-10-01)

The backend now captures the initial PSO supplied to public CreateCommandList,
in addition to Reset and SetPipelineState. Its suffix can dispatch without
rebinding that PSO. The public foundation probe's `initial-pso` mode passes
32 native RX 9070 XT iterations; the WARP diagnostic passes too. The runtime
now installs 81 documented method hooks, including CreateCommandList.

Resource tracking distinguishes legacy states from enhanced texture layouts
for mip/array/plane zero. Barriers on unrelated subresources do not overwrite
that state. An unfinished split barrier on an NGX texture explicitly rejects
Evaluate. The queued snapshot owns the layout/access metadata. Native interop
uses CommandList7 texture barriers to enter COPY_SOURCE/COPY_DEST and restore
the exact enhanced layout and access class; it does not reinterpret enhanced
layouts as legacy states. Legacy texture copies also affect only subresource
zero. Shared linear buffers continue to use their own legacy COMMON/copy states.
This follows [Microsoft's enhanced barrier interoperability rules](https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html).

`d3d12_gpu_enhanced_formats` independently tests all nineteen storage/plane
cases with a D3D12 enhanced producer and consumer. Full build and CTest pass
16/16 gates (`enhanced-final-ctest.log`, fifteen hardware and one WARP).
`d3d12-m-enhanced-packed.zip` and `d3d12-k-enhanced-packed.zip` each pass four
recorded NGX frames with their complete native transformer. Enhanced snapshot
logs report shader-resource layout 21 and unordered-access layout 20. All
four RGB images in both runs exactly match their independently quantised
FP16 baselines, max absolute/relative error 0. Frame age and shim CPU image
copy counts remain zero. Add `-BarrierMode enhanced` to the packed-run command
below to reproduce. The explicit harness can also supply layout metadata.

Tracking now also publishes explicit ordinary texture state/layout metadata
in the actual queue submission order. Each split segment stores its own
resource-state snapshot. An Evaluate input with no barrier
in its own list resolves the most recently submitted state at the callback,
after preceding lists in the batch. The metadata belongs to the resource, with
no global pointer cache or ownership cycle. Legacy buffer/simultaneous-access
decay is not treated as persistent texture state. Unknown states retain the
explicit caller metadata or NGX legacy default and are logged as tracked=0.

ExecuteCommandLists, Signal and Wait on one queue use a queue-owned submission
mutex: another thread cannot insert a submission/fence into prefix -> DLSS ->
suffix. Internal callback operations bypass that lock while the outer batch
owns it. The lock metadata has no reference back to its owning queue.
The backend now installs 83 public-method hooks.

`optiscaler-m-inherited-final.zip` passes four complete enhanced-layout M frames
with the transitions recorded only in predecessor A, none in the evaluated B.
`optiscaler-k-inherited.zip` covers legacy state inheritance. Both have all
five planes tracked=1 on all frames (M layouts 21/20; K states 0x40/0x8), full
native network hits, zero NaN/Inf and exact RGB baseline agreement, max
absolute/relative error 0. Full build/CTest still pass 16/16
(`inherited-queue-final-ctest.log`). Use `-BarrierMode inherited-enhanced` or
`inherited-legacy` to reproduce. Tracking still requires observing creation
or a transition after hook installation; initial enhanced states outside
that observation remain a game gate. Unknown pre-initialization indirect
signatures are still rejected.
OptiScaler's default restore hotfixes are false; integration will keep
RestoreComputeSignature, RestoreGraphicSignature and ExtendedStateRestore
false because their original-method trampolines can bypass logical-list routing.
Actual OptiScaler probes have exposed the frontend gate documented above;
no game files have been changed.

## Native GPU resource-format conversion (2026-10-01)

`pixel_convert_gfx1201.hsaco` converts imported D3D12 copy footprints to NGX's
RGBA16F/RG16F/R32F arrays and converts output back to the original resource
format. Canonical formats retain the existing direct path. Other formats use
pitched HIP allocations and native decode/encode kernels; images stay in VRAM.
Completion currently synchronizes the host before NGX uses its other streams.
The module is installed beside `d4r_nvngx.dll`; `D4R_FORMAT_MODULE` can override
its absolute path. The diagnostic runner copies it into its owned temporary
NGX runtime alongside the shim. No NVIDIA binaries are installed in packages.

Supported storage: RGBA16/32F, RGBA/BGRA8 UNORM, R11G11B10 FLOAT, RGB10A2 UNORM,
RG16/32F, RG16 SNORM/UNORM, R16F exposure, R32F/D32 depth, D24/R24 depth and
D16/R16 UNORM depth. Typeless R16 means floating-point exposure or UNORM depth,
according to the NGX plane. sRGB and D32S8 conversion are explicitly rejected
until independently validated. Optional mask textures are not supported yet.

The public `d4r_d3d12_pixel_probe` tests nineteen plane/format combinations at
259x17, with padded pitches, partial workgroups and row guards. D3D12 typed SRV
loads provide independent float32 decode values, with a mathematical binary16
nearest-value search for CUDA RNE conversion. HLSL `f32tof16` is not used as
an RNE reference: the current driver truncates that operation. Color encode
is compared bit-for-bit to typed UAV stores read through a D3D12 consumer SRV.
The reference includes HDR limits, subnormals, signed motion endpoints, UNORM
rounding/clamping and packed channel permutations. An initial RNE R11 encoder
failed this test; truncation now matches the driver, including finite HDR limits.
See [Microsoft's format-conversion specification](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm).

Results on RX 9070 XT / TheRock:

* `pixel-final-ctest.log`: 14/14 PASS (13 hardware tests and one WARP test).
* `pixel-milestone-diagnostics.zip`: all fourteen public diagnostic gates PASS.
* `d3d12-m-packed-formats.zip`: four M frames, forty native network launches,
  R11 color/output, RGBA32 motion, R16 exposure, zero shim CPU image copies.
* `d3d12-m-unorm-formats.zip`: four M frames with BGRA8 output, RG16 SNORM
  motion and RGBA32 exposure; forty native launches, frame age zero.
* `d3d12-k-packed-formats.zip`: four K frames and forty-four native launches.
* Each of those three complete runs matches its FP16 baseline after the
  independent D3D12 resource-format quantisation: RGB max absolute/relative
  error 0, PSNR infinite. No tolerance was relaxed for packed output.

Reproduce a complete packed M run with locally supplied NVIDIA DLLs:

```powershell
$env:D4R_ZLUDA_NATIVE_DIR="$PWD/build/native-m-gfx1201"
$env:ZLUDA_CACHE_DIR="$PWD/build/zluda-cache-windows"
powershell -NoProfile -File scripts/windows/test-windows-rdna4.ps1 -RuntimeProfile therock -ZludaRoot "$PWD/dist/zluda-windows-f16-reference" -PackageRoot "$PWD/dist/windows-rdna4-command-list" -NgxCore "$PWD/_nvngx.dll" -DlssDll "$PWD/nvngx_dlss.dll" -NgxMode d3d12 -Preset 13 -NgxOnly -RequireNativeNetwork -CommandListBackend -PixelProfile packed -Trace -Iterations 4 -OutputDirectory "$PWD/test-results/d3d12-m-packed-formats"
```

One public diagnostic command, without NVIDIA DLLs:

```powershell
powershell -NoProfile -File scripts/windows/test-windows-rdna4.ps1 -RuntimeProfile therock -ZludaRoot "$PWD/dist/zluda-windows-f16-reference" -PackageRoot "$PWD/dist/windows-rdna4-command-list"
```

The exact complete-frame comparison uses `frame_compare.py --exact` with
`--output-storage r11g11b10f` or `bgra8`. Ordinary comparisons still use the
existing FP16 output gates. Game integration next needs command-list initial
PSO handling, enhanced-layout tracking, optional mask import and OptiScaler
API/state-hook compatibility. The game directory is still untouched.

## M temporal precision regression (2026-09-30)

The earlier finite-frame and sampled-layer results did not cover every spatial
block. The second enc1 launch has a cancellation-sensitive P/V dot in block
`0,4`: the old native f32 WMMA route differed by up to 21 e4m3 code steps.
NumPy and the independent PTX interpreter agree exactly on that block.
`SWIN_EXACT_PV` stores each wave's sixteen P rows and V rows in LDS, sums sixteen
f16 products in f64, rounds directly to f16, then applies the existing e4m3
quantisation. Other GEMMs retain the gfx12 WMMA layout and FP16 widening.
The flag is enabled by `D4R_M_FP16_BASELINE`; upstream fast/Linux defaults are
preserved when that flag is absent. Intermediate snapshots are compiled only
with `D4R_SWIN_DIAGNOSTICS`; they are absent from packaged kernels.

The translated reference also differed: first frame, fourth enc3_tube launch,
block `0,0`, thirty stored codes, maximum fifteen e4m3 steps. Its inputs match
the corrected native launch exactly. Patch `0013` adds
`D4R_ZLUDA_WMMA_F16_REFERENCE=1` for gfx12 f16-accumulator MMA helpers, retaining
the established fragment gathers, pair fusion and conversions. It uses f64
sums and integer f16 RNE encoding, including subnormal bits. FP8 GEMMs continue
using the existing native WMMA widening. The default upstream helper behavior
remains available with this option unset or zero. Windows diagnostics select
the reference option by default. All d4r codegen options enter the JIT cache key.

This is a match to the independent mathematical NumPy/PTX baseline. NVIDIA's
[PTX floating-point MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
leaves accumulation order, rounding and subnormal handling unspecified; these
checks do not establish bit-exact agreement with NVIDIA hardware.

Recorded results on the RX 9070 XT / TheRock runtime:

* Public `cuda_mma_probe`: 512 synthetic matrix cases, 65,536 f16 outputs,
  zero differing bits with reference accumulation. The old fast WMMA route
  differs in 23,308 outputs on the same cancellation/subnormal stress fixture.
* `m-native-exact-pv-temporal-reference`: all spatial blocks of 40 actual native
  launches, every stored e4m3 value identical to NumPy, max abs/relative error 0.
* `m-exact-pv-enc1-regression-ptx`: original problem block `0,4`, independent
  PTX versus native and versus NumPy both max absolute/relative error 0.
* `m-exact-pv-complete-f16-reference.log`: four complete RGB frames against
  corrected fully translated M, max absolute/relative error 0, PSNR infinite.
* `m-translated-f16-temporal-reference`: all blocks of all forty translated
  M launches exactly match NumPy; no sampled-window exemption.
* `m-temporal-final-ctest.log`: 11/11 native Windows hardware tests PASS.
  Pinned fresh-source application of all twelve ZLUDA patches PASS.
* `d3d12-m-exact-pv-final.zip`: four finite frames, forty native launches,
  no shim CPU image copies, frame age zero. This remains the explicit queue
  boundary harness; open game command-list support is the next gate.

All captures, extracted PTX and real weights remain in ignored private result
directories. To repeat the full temporal scan without editing sources:

```powershell
$env:PYTHONPATH="$PWD/.tools/python/vendor"
python tools/windows/m_temporal_validate.py --capture-dir test-results/ngx-m-exact-pv-capture/captures --output-dir test-results/m-temporal-reference --exact
```

The game selected for the next integration test is Silent Hill 2 at
`D:\Games\SILENT HILL 2\SHProto\Binaries\Win64\SHProto-Win64-Shipping.exe`.
No game files have been modified at this milestone.

## Open D3D12 command lists (2026-09-30)

The backend intercepts documented D3D12 COM methods using pinned open-source
[MinHook 1.3.4](https://github.com/TsudaKageyu/minhook/releases/tag/v1.3.4), commit
`c3fcafdc10146beb5919319d0683e44e3c30d537`. No NVIDIA instructions, proprietary
DLL contents or private driver interfaces are patched. The typed forwarding
code is generated from public SDK CommandList0..7 declarations, with typed
CommandList8..10 additions audited in Microsoft's current public
DirectX-Headers, `adbd6f3ba40795c46a8d0f33af00bcb57ff0f0a4`. The AMD runtime
exposes CommandList10: eighty public-method hooks now install successfully.
SetProgram descriptors are copied by value. Work-graph replay omits INITIALIZE
to preserve already initialized backing memory; see the
[Microsoft work-graph specification](https://microsoft.github.io/DirectX-Specs/d3d/WorkGraphs.html#d3d12_set_work_graph_flags).
Full GPU work-graph execution is not a tested gate yet.

At NGX Evaluate, the caller's resource references and typed parameter values are
copied into an owned snapshot. The recorded prefix is closed. Subsequent calls
on the same logical object are forwarded into a fresh native suffix list with
its own allocator. State-setting calls are replayed with deep copies of host
arrays and retained COM references. Open PIX events are closed in the prefix
and reopened in the suffix. At ExecuteCommandLists, preceding lists in the
caller's batch execute first, followed by the prefix, actual CUDA evaluation
and output copy, then the suffix and following lists. All work completes in the
same frame. Completion waits currently stall the host; they do not age frames.

The callback module remains loaded while public D3D12 method interceptions are
installed. Original command-list pointers in the routing map are weak; final
Release removes metadata, suffixes, allocators and queued feature references.
This avoids a reference cycle that would keep every command list alive.
Failure at queue submission is logged and prevents submitting the consumer
suffix. No previous-frame fallback is substituted.

`d3d12-command-foundation` passes 32 iterations with no NVIDIA binaries. Each
batch contains A, B and C. A produces shader inputs, B dispatches before a
boundary, the callback verifies that prefix and replaces inputs on the GPU,
the suffix dispatches without rebinding root state, and C reads the result.
All values match; a temporary root-constant array may change after recording;
live recording metadata returns to zero after every iteration.

`d3d12-m-command-list.zip` passes four ordinary NGX EvaluateFeature frames with
forty native M launches. RGB is bit-exact to the earlier explicit boundary
harness. The initial producer remains unsubmitted until the same queue batch,
and the caller's Color/Reset parameters are deliberately changed after
Evaluate returns, testing snapshot ownership. `d3d12-k-command-list.zip` also
passes four finite frames and all forty-four native K launches; all four RGB
frames are bit-exact against `d3d12-k-final`. Complete CTest: 12/12 PASS
(`d3d12-command-final-ctest.log`).

Enable this backend with `D4R_D3D12_COMMAND_BACKEND=1`; the diagnostic runner
sets it through `-CommandListBackend`. Before game testing, remaining gates are
enhanced barrier layouts,
actual game texture formats and interaction with OptiScaler's own state hooks.
Closed lists, active render/query scopes, GPU predication and unknown indirect
state are currently explicitly rejected instead of guessing their semantics.

## Indirect commands and Windows diagnostics (2026-10-01)

CreateCommandSignature interception stores its public argument description
using ID3D12Object::SetPrivateData, owned by the original COM object. There is
no driver-private introspection or independent signature-lifetime table.
After ExecuteIndirect, replay restores exactly the documented zero/NULL values
for affected root constants, views, VB slots and IB; unaffected state remains
inherited. See [Microsoft's indirect drawing specification](https://learn.microsoft.com/en-us/windows/win32/direct3d12/indirect-drawing).
Signatures created before interception remain explicitly untracked.

The RX 9070 XT passes 32 queue/backend iterations alternating direct Dispatch
and dispatch-only ExecuteIndirect. Bindings, ordering, temporary host argument
copies and final Release are verified (`d3d12-indirect-plain-final`). Microsoft
WARP passes 32 backend iterations with an indirect signature that changes a
root constant, followed by direct Dispatch without rebinding: both prefix and
suffix observe the required zero (`d3d12-indirect-root-warp-backend.log`).
WARP is reported as software, never as gfx1201.
CTest now passes 13/13 (twelve physical GPU gates plus the WARP root-reset
regression), `d3d12-command-v10-final-ctest.log`. Four ordinary recorded M
frames still execute forty native transformer launches and remain bit-exact
against the previous CommandList7 implementation (`d3d12-m-command-v10.zip`,
`d3d12-m-command-v10-reference.log`).

**Unresolved native AMD root-mutating indirect regression:** that same valid
root-constant/Dispatch signature removes the AMD device (0x887a0006). It also
fails with d4r hooks disabled and with HIP omitted, under inbox D3D12Core and
Agility 1.619.5, with UPLOAD and DEFAULT argument buffers. DRED stops inside
ExecuteIndirect and reports 0x80015c000 when the argument-buffer GPU VA is
0x200057000. The exact fault address is four times that allocation's VA; this
is an observation, not a proven driver cause. This workload passes on WARP.
Do not classify native root-mutating indirect execution as validated. The
default physical test covers dispatch-only signatures; the root mutation case
is a separate reproducible diagnostic:

```powershell
powershell -NoProfile -File scripts/windows/test-d3d12-command.ps1 -NoHooks -NoHip -Mode indirect-root-reset -Iterations 2 -OutputDirectory test-results/amd-root-indirect-reproducer
```

Logs retain stdout/stderr, device-removed reason, DRED breadcrumbs/fault address,
loaded DLL versions and exception minidumps via D4R_DIAG_DIR. `-Warp` selects
the software comparison. No game or NVIDIA DLL is required.

Optional app-local debug target: set D4R_AGILITY_ROOT to unpacked official
Microsoft.Direct3D.D3D12 1.619.5. NuGet ZIP SHA256:
`0e9bcf32aac9a79343ede9b21e4864950ee54577e3d8e19bfcdf002bb4e9bfd6`.
On this machine, the public debug interfaces still return SDK_COMPONENT_MISSING
despite app-local SDKLayers loading. Installing Windows Graphics Tools failed
with DISM's component-store-corrupted error. No repair, reboot, security change
or game runtime dependency was introduced. DRED works without that layer.

## Source baseline (checked 2026-09-29)

* d4r: `ed1ab60`, main, 14 commits; https://github.com/countervolts/d4r.
* ZLUDA master: `ee2f25a180099fa42f36b2346732e1f2470a03ad`,
  2026-09-22. This is also the base named by d4r's existing patches.
* Windows binary baseline: ZLUDA `v7-preview.11`,
  `zluda-windows-ee2f25a.zip`, SHA256
  `7788c1ed43385e62f0cc5f4d1aead24c92b671c02f1d685f63182540dea16d74`.
* Local HIP SDK: `C:\Program Files\AMD\ROCm\7.2`, headers report
  `7.2.60201`, commit `38d754472`. Compiler: AMD clang 21,
  LLVM commit `590b9320a5be90e40268759c6203c01fde121e68`.
* Keep stable SDK and TheRock installations separate. Select one directory per
  build and test run; never mix headers, import libraries, runtime or COMGR.
* Tested isolated TheRock core: `10.2.0a20260929`, HIP `7.17.26386`, commit
  `0ad5f73253`; wheel SHA256
  `9623b97ca511eaa176905075a504393eca6fde594222af10b55e4fb1edfe5a25`.
  No machine-wide runtime replacement is performed.
* CMake 3.31.6, Ninja 1.11.1 (Python package 1.11.1.4).
* Synthetic K numerical reference: Python 3.11 and local NumPy `2.4.6`
  ([PyPI release](https://pypi.org/project/numpy/2.4.6/)); installed only
  into `.tools/python/vendor` by setup, outside the game runtime.
* Windows host toolchain: LLVM-MinGW `20260922`, UCRT x64, SHA256
  `e3ad77d117a4bea19a7a3b333341824d79a5a371004a10e25b8504e7b3047666`.
  This is a portable fallback on this machine, where MSVC is not installed.
  The device compiler remains the selected AMD HIP clang.

## Upstream RDNA4 update (checked 2026-09-30)

Fetched and integrated upstream main commit
[`dbef4b24f4bc974725c3b2bc74ca442a694b901d`](https://github.com/countervolts/d4r/commit/dbef4b24f4bc974725c3b2bc74ca442a694b901d),
"Add RDNA4 (gfx12) support and a 0.1.2 optimization round" (2026-09-30 UTC).
It changes 42 files and supplies real per-target fragment layouts rather than
renaming gfx11 builtins. Before the merge, an isolated archive was compiled
and its `enc1`/`enc2` objects were checked on this RX 9070 XT against our fixtures.

* `kernels/common/wmma_layout.h` separates gfx11/gfx12 operands, accumulator
  rows, weight/activation loading, packing and reconstruction. K, M and native
  texture tails use it. The gfx11 implementation and layout12 test shim remain.
* M has an optional native e4m3 FP8 variant, with matching ZLUDA/texture lowering
  and kernel directory selection. **Windows CMake keeps FP16 widening as the
  baseline; it does not enable native FP8.** M arithmetic on this GPU remains
  unvalidated until K and its full DLSS integration are correct.
* New ZLUDA patches `0006`/`0007` add k8 WMMA, prep slots, wave64 for MMA-free
  kernels, gfx12 helpers and FP8 selection/cache identity. These patches are
  now built into native Windows ZLUDA. Integer PTX and resource descriptor
  tests pass on hardware; complete K/M lowering remains to be validated.
* Upstream restored `kernels/tools/model_enc3.py`, resolving the missing M
  reference dependency found in the initial audit. New check/replay utilities
  cover K/M and texture tails. Texture artifacts still require local DLL/PTX
  extraction and a patched `d4r_emit`; they are not built by Windows CMake yet.
* There is still no native Windows runtime backend in this update. GPU selection
  still reads KFD topology; Wine, Linux shared descriptors and vkd3d-proton
  command-list splitting remain in the upstream path.

Upstream explicitly reports gfx1201 **emulator** checks and gfx1200 compile-only
coverage; it does not claim actual RDNA4 hardware verification
([native-kernel validation notes](https://github.com/countervolts/d4r/blob/dbef4b24f4bc974725c3b2bc74ca442a694b901d/docs/native-kernels.md)).
The hardware results below apply only to the named Windows probes and synthetic
layers, not to the complete network or the upstream FP8 implementation.

Local integration retains the Windows probes, NGX ABI correction, NVAPI topology
and D3D12/HIP interop. K now uses upstream's native gfx12 layout directly; the
legacy adapter remains only for its independent diagnostic regression. M gains
the same minimal device-header path as K. Its lane-mask constants no longer
require a host `<type_traits>` installation. Public HIP `uint2`/`uint4` storage
alignment and qualifiers are reproduced by the device-only header.

Windows CMake compiles all **11 K + 5 M** source modules with gfx1201 wave32,
`-O3` and each source's upstream `d4r-build-flags` (including `-mcumode`). Both
stable HIP 7.2 and isolated TheRock build successfully. The objects install to
`experimental/k` and `experimental/m`, outside the ZLUDA override directory.
All 11 K layers now have GPU synthetic replay/reference checks (results below).

2026-09-30 integration results on RX 9070 XT, Windows 11:

| Check | Result |
| --- | --- |
| TheRock diagnostic runner with the two local NVIDIA DLLs | 11/11 PASS; `test-results/upstream-rdna4-merged.zip` |
| TheRock CTest, including D3D12/HIP external memory/fences | 8/8 PASS |
| WMMA raw, legacy adapter and upstream native layout/packing | 32 cases, one/two K16 steps, nonzero C; max abs/relative error 0 |
| Upstream K enc1 full / merged reference | Max abs `0.000244140625`; PSNR `98.21 / 97.63 dB`; no NaN/Inf |
| Upstream K enc2 full / merged reference | Max abs `3.81469727e-06`; PSNR `130.39 / 132.49 dB`; no NaN/Inf |
| Stable HIP 7.2 non-interop CTest | 6/6 PASS |
| Stable-built K enc1/enc2 with stable runtime | 32 identity iterations each + both nonzero references PASS, same errors as TheRock |

The previously failing stable compiler/legacy-layout K path is superseded by
the native upstream layout: these two current layer fixtures pass under stable
HIP 7.2. Stable external-memory mapped-view ownership still requires the runtime
fix/TheRock; passing K does not resolve that leak. Build logs remain under each
`build/windows-rdna4*` directory; stable GPU logs/fixtures remain under
`test-results/upstream-stable-*-gpu.log` and `test-results/upstream-rdna4-stable`.

Next gate: validate K against real DLSS captures with the built Windows ZLUDA,
then implement same-frame Windows NGX Evaluate
with explicit D3D12 queue submission ownership. Full K precedes M and native
FP8/performance work. No complete DLSS or OptiScaler game result is claimed.

## Native Windows ZLUDA build and Evaluate work (2026-09-30)

### M strict FP16 baseline and independent reference (2026-09-30)

`D4R_M_FP16_BASELINE=ON` is the Windows CMake default and defines `SWIN_EXACT`.
Activations retain e4m3 requantisation; matrix accumulation rounds to f16 after
each K32 step. Gfx1201 WMMA widens FP8 operands to FP16. Native FP8 is disabled
until this baseline also passes the D3D12 harness.

All five `rrlite_{enc1,enc2,enc3_tube,dec2,dec1}_4x4` modules pass 32 iterations
on public synthetic 16x16 fixtures, including nonzero Q/V/WO/MLP weights,
decoder expansion/skip and patch merge. Every stored e4m3 code exactly matches
the NumPy model; max absolute and relative error are zero. The fixture's patch
merge deliberately uses large weights to exercise its output bound.

An independent CPU PTX check exposed an upstream omission: DLSS 310.9's patch
merge clamps f16 results to +/- f16(2*pi), before e4m3 rounding. Without this
bound, real `enc2` merge outputs reached +/-9 or 10 where the PTX stored +/-6.5.
The correction is limited to `SWIN_EXACT`; upstream fast arithmetic stays
unchanged. The Windows model enables the same audited bound explicitly; the
legacy Linux oracle remains the default. Both implementations now agree with
the independent PTX interpreter, rather than merely agreeing with each other.

`0011-audited-k-m-windows-replay.patch` adds exact M capture ABI sizes (56/88
bytes) and restricts K captures to the eleven audited names. The public CPU PTX
interpreter now handles divergent finite backward loops by suspending completed
lanes at the loop fallthrough. `tools/windows/test_ptx_reference.py` verifies
lane-specific loop counts and reconvergence with a public synthetic program.

With the user's 310.9 DLL, `test-results/ngx-m-bounded-capture.zip` passes four
frames with native counts enc1=4, enc2=4, enc3_tube=24, dec2=4, dec1=4. No NaN/Inf
or missing transformer is accepted. `test-results/m-real-final-dual-reference`
passes 12 NumPy blocks and two independent CPU PTX blocks per layer; all sampled
codes agree exactly (max abs/relative error 0, PSNR infinite). Capture files,
weights and extracted PTX remain private and must not be redistributed.

Reproduce the layer gates:

```powershell
& scripts/windows/test-m-replay.ps1
& scripts/windows/stage-native-m.ps1 -DlssDll "$PWD/nvngx_dlss.dll"
$env:D4R_ZLUDA_NATIVE_DIR="$PWD/build/native-m-gfx1201"
$env:D4R_ZLUDA_WMMA='1'; $env:D4R_ZLUDA_WMMA_FP8='1'; $env:D4R_ZLUDA_WMMA_FP8_NATIVE='0'
& scripts/windows/test-windows-rdna4.ps1 -RuntimeProfile therock -NgxOnly -NgxMode evaluate -Preset 13 -Iterations 4 -RequireNativeNetwork -NgxCore "$PWD/_nvngx.dll" -DlssDll "$PWD/nvngx_dlss.dll"
```

Earlier translated M attempts used incorrect environment names and an
unwritable default JIT cache. The runner now defaults to the actual
`D4R_ZLUDA_WMMA`, `D4R_ZLUDA_WMMA_FP8`, `D4R_ZLUDA_WMMA_FP8_NATIVE` options
(1, 1, 0), records them, and restores the caller's environment. Patch 0012 adds
absolute `ZLUDA_CACHE_DIR` and reports directory/open failures. The diagnostic
default is `build/zluda-cache-windows`; five Rust cache tests and all eleven
patches applied to the pinned clean upstream pass. A completed translated M
image comparison is not yet claimed. The user selected Silent Hill 2 at
`D:\Games\SILENT HILL 2\SHProto\Binaries\Win64\SHProto-Win64-Shipping.exe`
for the later game gate. No game files have been changed yet.

OptiScaler upstream inspected at `45a2001303ddff632e279f77aef85ceede5832cb`,
2026-09-30. Its DLSS backend forwards an open command list into Evaluate;
its swapchain hooks retain the game queue. A correct same-frame Windows path
must preserve submission order when earlier command lists have not yet been
submitted. Closing/submitting the current list immediately inside Evaluate
is insufficient for a general game integration. The next backend gate is a
D3D12 harness with explicit queue boundaries; command-list/queue integration
must be validated before the game gate.

### Native D3D12 harness milestone (2026-09-30)

The new `d4r_nvngx.dll` has a Windows backend independent of the Wine shim.
`d3d12_external.h` owns Win32 handles, shared DEFAULT buffers, HIP mappings and
the D3D12 fence/semaphore timeline. Canonical color/output RGBA16F, motion RG16F,
and depth/exposure R32F remain in VRAM. Texture-to-buffer copies, CUDA array
transfers and output copies execute on the GPU. Harness initialization uploads
synthetic inputs; final readback is confined to its verifier/output exporter.

The first complete attempt exposed a synchronization race: waiting on a HIP
external semaphore on its nonblocking stream and synchronizing that stream did
not reliably establish completion for work subsequently submitted through other
CUDA streams. Samples read back by the diagnostic could hide the race. The
baseline additionally waits for the producer's actual D3D12 fence event before
CUDA submission. It waits for completion, without transferring image data or
using another frame. `cuCtxSynchronize` covers NGX's internal streams, followed
by HIP fence signaling and a D3D12 queue wait before copying the output texture.
Reducing these host completion waits is a later profiling/ordering task.

Private results: `test-results/d3d12-k-direct-fence.zip` and
`test-results/d3d12-m-direct-fence.zip`, four finite, nonconstant frames each,
44 native K and 40 native M launches. Every RGB value agrees bit-for-bit with
`ngx-k-native-verified` / `ngx-m-bounded-capture`: max absolute/relative error 0,
PSNR infinite. CTest: 10/10 PASS including a five-plane D3D12-to-CUDA array
roundtrip and pitched CUDA device/array copies with padding sentinels.

Build/package: `dist/windows-rdna4-d3d12`. Reproduce K (use preset 13 and the
staged `native-m-gfx1201` directory for M):

```powershell
$env:D4R_ZLUDA_NATIVE_DIR="$PWD/build/native-k-gfx1201"
& scripts/windows/test-windows-rdna4.ps1 -RuntimeProfile therock -PackageRoot "$PWD/dist/windows-rdna4-d3d12" -NgxOnly -NgxMode d3d12 -Preset 11 -Iterations 4 -RequireNativeNetwork -NgxCore "$PWD/_nvngx.dll" -DlssDll "$PWD/nvngx_dlss.dll"
```

The runner places the shim beside the two private NVIDIA DLLs in an owned
temporary directory, matching NGX's caller-relative feature lookup. They are
never installed into the public package. `D4R_INTEROP_VERIFY` explicitly enables
extra CPU diagnostics and changes the logged copy flag; keep it unset for the
normal path. The open-command-list game export still rejects execution until
submission ordering is implemented. The explicit harness export requires that
all preceding producers have already been submitted. Other texture formats,
dynamic resolution and real-game resource-state tracking remain unvalidated.

### K real-weight native Evaluate and correctness baseline

The locally supplied DLSS 310.9.1.0 selects all eleven `dltss_pwin_*` functions
with preset 11 (K). `0010-native-identity-replay-windows-cache.patch` adds:

* FNV-1a PTX identity checking against upstream `d4r-kernels.txt`, with an
  explicit native miss/translated fallback log. A mismatched version cannot
  receive an override by function name alone.
* Owned native modules/images, released with their CUDA function/module, and
  direct native loading for fully replaced single-entry modules with only LDS
  declarations. Modules with exported globals or other functions retain JIT.
* Windows loaded-DLL size/mtime in the translator cache identity. Rebuilding
  an uncommitted translator no longer reuses its older cached code.
* Opt-in native Windows K replay capture, before and after a launch, using
  HIP allocation range queries. Captures are synchronous diagnostics, can
  contain proprietary weights, and are never enabled in the game fast path.
  `D4R_ZLUDA_VERBOSE=1` logs actual native/translated launches.

The Windows minimal `__syncthreads` previously omitted HIP's release/acquire
memory fences. Real, multi-window captures exposed nondeterministic LDS data
and intermediate NaNs, despite the one-window synthetic tests passing. The
shim now matches public HIP barrier semantics. Those earlier captures are
invalid for correctness claims.

Windows K defaults to `D4R_K_FP16_BASELINE=ON`: every K16 accumulator step and
the L2 reduction tree round to FP16. The norm grouping was independently
recovered and checked with the existing CPU PTX interpreter for C=32/64/96/128/160.
The reference model supports `NORM_FRAGMENT_ORDER=True`; its legacy default
remains unchanged. The original float norm approximation itself differed from
PTX by 0.13965 on one real decoder window, so it cannot serve as a bit-exact
oracle. Strict references and the independent PTX gate are both required.

Validated on RX 9070 XT with isolated TheRock:

* `test-results/k-strict-synthetic-final.zip`: 11/11, 32 launches per layer,
  **max absolute and relative error 0** with strict references.
* `test-results/k-real-strict-dual-reference`: all eleven real-weight layers
  pass. Up to 12 NumPy windows/layer, plus the two worst NumPy difference
  windows checked independently through PTX. NumPy PSNR 75.40–103.98 dB;
  PTX PSNR 67.11–122.39 dB, all selected output elements finite. Max relative
  error is reported with a 0.001 denominator floor; near-zero values can have
  a large relative error. Gates require PSNR >=60 dB and max absolute error
  <=max(0.001, 0.005*reference peak).
* `test-results/ngx-k-strict-norm-final.zip`: four K frames, all 44 native
  transformer launches logged, no NaN/Inf or unwritten output.
* Complete-frame comparison with translated WMMA K: PSNR 92.91, 81.85, 84.45,
  86.10 dB, max absolute error 0.01514. Differences are localized: at most
  21 RGB elements exceed 0.002, and at most 0.00294% exceed 1% of peak.
  These are approximate numerical results, not bit-exact NVIDIA hardware
  verification. The frame gate requires >=60 dB, max <=2.5% of peak, and
  no more than 0.01% of RGB elements above 1% of peak. An initial stricter
  1% single-element gate failed on these isolated temporal differences;
  the observed metrics and current bounds are recorded explicitly.
* `test-results/native-identity-final`: wrong hash rejects the override and
  runs translated PTX correctly; matching hash executes 32 native launches.
* CTest 9/9 and fresh-source application of all nine ZLUDA patches PASS.

Reproduce the actual K layer gate without editing sources:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/test-k-captures.ps1 `
  -CaptureDirectory C:\Users\Administrator\d4r\test-results\ngx-k-fp16-fenced-capture\replay `
  -DlssDll C:\Users\Administrator\d4r\nvngx_dlss.dll
```

This is CUDA NGX validation with synthetic image inputs and actual local DLSS
weights. It does not claim a completed Windows D3D12 shim, OptiScaler game test,
dynamic scene validation, or M support. These are the next gates.

### All K layers: native Windows synthetic replay (2026-09-30)

`native_replay_probe` reads upstream `manifest.txt`/`args.bin`/`alloc-N.bin`,
relocates allocation pointers, runs the module's prep and main kernels, and
saves GPU allocations for the existing `pwin_check`/`pwin_model` references.
No NVIDIA DLL, PTX or weights are needed for these synthetic fixtures.
Each fixture enables nonzero projections, attention, MLP and the layer's
embedding/merge/head operations; it covers one 8x8 window with sparse weights.
It does not replace validation with real network weights, multiple windows,
shifted borders, textures or complete DLSS output.

Run from the repository:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/test-k-replay.ps1
```

Hardware result: `test-results/k-all-layers-synthetic.zip`, **11/11 PASS**,
32 native launches per layer, all output elements finite. The acceptance limit
is max absolute error <= 0.001 and PSNR >= 60 dB; relative error uses a 0.001
denominator floor. Validation stops on the first mismatch.

| Layer | Max absolute error | Max relative error | PSNR (dB) |
| --- | --- | --- | --- |
| enc0 | 0.00048828125 | 0.0011682243 | 88.75 |
| enc1 | 0 | 0 | exact |
| enc2 | 0.0000305175781 | 0.0009319664 | 114.40 |
| enc3 | 0 | 0 | exact |
| enc4 | 0 | 0 | exact |
| dec5 | 0 | 0 | exact |
| dec4 | 0 | 0 | exact |
| dec3 | 0 | 0 | exact |
| dec2 | 0.0000305175781 | 0.0008741259 | 111.42 |
| dec1 | 0.00048828125 | 0.0011086475 | 99.33 |
| dec0 | 0.000122070312 | 0.0133333333 | 100.84 |

The next K gate is capture/reference validation with actual DLSS weights and
proof of native transformer launches during Evaluate. Native overrides must
check the originating PTX identity before replacing a function.

The local fork now builds natively under Windows, including d4r patches
`0002`–`0007`. New `0008-native-windows-build.patch` preserves MSVC/Linux builds
and adds LLVM-MinGW support: prebuilt LLVM selection, GNU delay-load flags,
Windows LLVM system-library propagation, configurable HiGHS C++ library,
public OCKL assertion reporting, and optional TaskDialog lookup. It does not
modify NVIDIA binaries. The SDK's `long` shuffle overloads need an LLP64 fix
in a private header mirror; `build-zluda-helpers.ps1` applies it without touching
the installed HIP SDK. Helper bitcode is rebuilt in all four wave/FP variants.

Pinned additional build dependencies:

* Portable official Rust/Cargo `1.98.1`, `x86_64-pc-windows-gnu`, release
  `2026-09-03`. `setup-rust-toolchain.ps1` verifies SHA256 of rustc, Cargo,
  rust-std and rust-mingw archives and installs only under `.tools`.
* ZLUDA LLVM submodule `ff4dc1f7c9e1c64d4d69e40f4ed30c2280a96dfd`,
  `llvm-config` reports `22.0.0git`. AMDGPU/LLVM/LLD are built with LLVM-MinGW
  `20260922`; no Linux build host or runtime is used. The Windows builder adds
  `patches/zluda-llvm/0001-backport-gfx1154.patch`, adapted from official LLVM
  `7a0829e41228513299c5108685b0bc127463c6a1`. Rebuild and relink ZLUDA when
  updating; native Clang 24 target support alone does not cover translated PTX.
* HiGHS submodule `364c83a51e44ba6c27def9c8fc1a49b1daf5ad5c`.
* The LLVM IR helper producer is stable HIP 7.2/LLVM 21. The final target is
  selected by the translator from HIP `gfx1201`; generic helper generation
  strips its temporary target attributes, as the existing Linux builder does.
* `dist/zluda-windows-native` contains the built CUDA/NVAPI/trace DLLs,
  `d4r_emit.exe`, local open-source libc++/libunwind DLLs and build hashes.
  This directory contains no NVIDIA DLLs, extracted PTX or weights.

Build sequence after preparing the pinned ZLUDA source/submodules:

```powershell
powershell -NoProfile -File scripts/windows/setup-rust-toolchain.ps1
powershell -NoProfile -File scripts/windows/prepare-zluda-source.ps1
powershell -NoProfile -File scripts/windows/build-zluda-llvm.ps1
powershell -NoProfile -File scripts/windows/build-zluda-helpers.ps1
powershell -NoProfile -File scripts/windows/build-zluda-windows.ps1
```

`prepare-zluda-source.ps1` refuses to reapply patches over a modified checkout.
For an already prepared checkout, rebuild only the affected components.
Configure/fetch/build logs remain under `build/zluda-*`.

The NGX probe now accepts `--ngx-mode evaluate`, `--preset 5|11|13` and a frame
count. Its synthetic CUDA-array path is a **debug harness** with host uploads
and final readback, not the game's fast path. It initializes output to NaN,
checks all RGBA components for finite values, checks RGB variance and saves
unmodified `.rgba16f` plus BMP previews. A successful Evaluate alone is logged
as `network_validation=pending`. The diagnostic exception handler also saves
first-chance fault details/minidumps before NGX's own final filter takes over.

The first stock-ZLUDA E test failed during CreateFeature (`0xc0000005`);
trace recorded missing `cuda_histogram_kernel`/`cuda_dldn_engine_histogram_kernel`
with unrecognized `tex.base.2d.v4.f32.s32`. This is a captured failure, not a
successful DLSS result. Bundle: `test-results/ngx-evaluate-e-stock.zip`.
The first custom build then exposed missing libc++ deployment and an eager
Common Controls v6 import; both are corrected. `0009-resource-descriptor-queries`
implements texture and 2D/3D array queries. Texture descriptors are tracked
from creation and removed on destruction: probing a surface as a texture
returns an API error instead of dereferencing HIP host metadata. This also
preserves CUDA flags/reserved bytes, which HIP's query does not fully initialize.

Hardware evidence with the patched runtime:

* Full runner `native-zluda-descriptors-milestone`: **12/12 PASS**, with 32 PTX
  iterations per context, 32 CUDA image iterations and unchanged HIP/K/interop
  checks. Package/default DLL selection prefers the locally built runtime.
* `cuda-images-kind.stdout.log`: 32 iterations of 1/2/4-channel arrays,
  texture/surface descriptor and storage round trips, plus wrong-kind errors.
* `ngx-evaluate-e-kindfix`: requested preset E Create/Evaluate **PASS** for one synthetic
  frame, finite RGBA, nonconstant output saved in the private result bundle.
  This establishes the first native Windows DLSS CUDA execution path. K/M
  transformer validation and D3D12 game integration are still pending.
* `ngx-evaluate-e-four-frames`: four consecutive finite synthetic frames,
  reset only on frame 0, saved RGBA16F/BMP outputs. CTest after self-contained
  CUDA dependency lookup: **9/9 PASS**. Fresh pinned-source application of all
  eight patches passes `check-zluda-patches.ps1`.
* First-chance capture successfully saved two minidumps for the preceding
  surface-as-texture HIP crash; `ngx-evaluate-e-arraydesc.zip` preserves it.

Example focused diagnostic (after the full foundational checks), using the
user's local DLLs:

```powershell
powershell -NoProfile -File scripts/windows/test-windows-rdna4.ps1 -RuntimeProfile therock -NgxOnly -NgxMode evaluate -Preset 5 -Iterations 4 -NgxCore C:\Users\Administrator\d4r\_nvngx.dll -DlssDll C:\Users\Administrator\d4r\nvngx_dlss.dll
```

`-NgxOnly` is for iterative NGX debugging; its summary explicitly says that
foundational checks were skipped. Private trace bundles can contain extracted
NVIDIA PTX/weights and must not be included in public release packages.

## Architecture and platform boundaries

Existing path:
`D3D12 -> OptiScaler -> d4r shim -> NGX CUDA -> Wine nvcuda -> ZLUDA -> HIP`.
The shim records Vulkan copies through vkd3d-proton interop. A patched vkd3d
splits recording into submissions and gates the second part on a semaphore.
The Wine bridge translates Windows/SysV ABIs, loads Linux libraries, chooses
native kernels from KFD topology and performs fd-based Vulkan memory imports.

Planned Windows path:
`D3D12 -> OptiScaler -> d4r shim -> NGX CUDA -> Windows ZLUDA -> HIP gfx1201`.
Platform-independent NGX parameter handling, formats, references, replay format
and kernel identity checks should be retained. A Windows backend must use
Win32 DLL loading, HIP device properties, shared D3D12 allocations and fences.
Wine builtins, KFD, file descriptors, Proton and vkd3d are excluded from runtime.

An NGX Evaluate call receives a **recording** command list, not a queue that can
be submitted halfway through by the shim. Native D3D12 has no equivalent to the
vkd3d split extension. Same-frame synchronization therefore needs an explicit
submission boundary supplied by the harness and then OptiScaler/queue integration.
A HIP wait inserted behind an unsplit D3D12 submission can deadlock. Merely
mapping VRAM does not solve command recording/submission ownership.

## Audit map and decisions

* `README.md`, `docs/{architecture,building,native-kernels,performance}.md`:
  Linux-tested baseline; upstream K translation can produce nonfinite results.
  A successful NGX return alone is not transformer validation.
* `tools/d4r_nvngx_shim.cpp`: Windows API/NGX entry points and C++ worker are
  reusable; Vulkan resource conversion, markers and split-frame functions form
  the platform boundary. Portable ini paths currently assume Wine translation.
* `tools/wine_nvcuda_bridge.c`: loading (`dlopen`, pthread_once), KFD target
  selection, descriptor bookkeeping, PTX manifests, replay/profile and CUDA
  forwarding are interleaved. Preserve identities/instrumentation, replace
  platform services rather than copy the Wine bridge into Windows.
* `patches/zluda/0002`: DLSS arrays/textures/surfaces, PTX instructions, dumps;
  `0003`: bounds and gfx11 MMA; `0004`: native override/prep, FP8, extra bitcode,
  handoffs; `0005`: null texture reads. The gfx11 MMA optimization is explicitly
  architecture-gated. It must not be enabled for gfx12 by changing that gate.
* `patches/vkd3d-proton/0001`: submission split, not a portable D3D12 interface.
* `kernels/k`: `pwin_common.h`, layer/position/wide templates and eleven entries.
  Operands are replicated 16-half rows; accumulator row is `2*i + (lane>>4)`;
  `operand_from_dt`, `wtile`, weight prep and stores depend on that convention.
* `kernels/m`: `swin_common.h`, `swin_block.h`, five standalone entries;
  FP8 weights expanded to FP16, `kslot`, `woff`, `bload`, `k32`, accumulator
  reconstruction and xor shuffles need layout audit. Preserve rounding per k32.
* `kernels/tex`: enc0 tail/dec0 head also use gfx11 WMMA and output shuffles;
  changing only K/M would leave part of M incorrect. Generated PTX is private.
* `kernels/build.sh`, texture build and `scripts/build_zluda_ptx_helpers.sh`:
  Linux tool paths, shell, bitcode cleanup and target assumptions need a Windows
  build path. Keep existing Linux scripts operational.
* `kernels/tools/{pwin_model,swin_model,ptxsim,psnr,kernel_manifest}.py` and
  `dump_runner.cpp`, `tools/replay_dlss_*`: use existing reference semantics
  and capture relocation format; add finite checks and quantitative comparison.
* `scripts/package_release.sh`, install/launch scripts and `packaging/`:
  Linux release bundles Proton components. Windows diagnostics/package must be
  separate and must never include NVIDIA DLLs or extracted NVIDIA PTX/weights.
* Upstream ZLUDA already has Windows delay-loading for HIP 7 and fallback HIP 6.
  Its Rust binding calls versioned `hipGetDevicePropertiesR0600`; a reported
  gfx1201 crash is not evidence of an ABI bug without a reproducer/trace here.

## gfx11 versus gfx12 WMMA

For wave32, gfx11 FP16 A/B are 16 halves per lane with replication across the
two 16-lane groups. gfx12 uses 8 halves per lane without replication. The gfx12
builtin has a `_gfx12` suffix; its FP32 accumulator is still eight floats, but
rows are contiguous within the lane group, rather than gfx11 even/odd rows.
The backend must explicitly convert operands **and** accumulator distribution,
or update all dependent loads/stores and intermediate fragments together.
Start with FP16 / FP32 accumulate. M native FP8 is a later measured optimization.
Compile success is not proof of lane layout or numerical correctness.

The initial `kernels/common/wmma_backend.h` adapter preserves the gfx11 call and maps d4r's
16-half operand / even-odd accumulator contract to gfx12's 8-half operand /
contiguous accumulator contract. The gfx12 mapping uses an unconditional
wave32 half-exchange for every source fragment element before selection.
Performing the exchange only on lanes selected by a group-dependent branch
returned channel 9 in place of channel 1 on this card: the opposite half of
the wave was inactive at that instruction. This was caught by the CPU reference.
`tools/windows/wmma_gfx1201.hip` tests raw gfx12 WMMA, the adapter, upstream's
native layout/operand packing, the half-exchange and lane ID with nonzero
accumulator and one/two K16 steps.
Both stable HIP 7.2 and TheRock 10.2 produced exact 16x16 outputs across 16
different inputs; max absolute and relative errors were both zero.

After the upstream merge, `kernels/k/pwin_common.h` calls the native
`wmma_layout.h` implementation; it no longer converts around every MMA with
the legacy adapter. The Windows device-only
compiler uses `kernels/common/hip_device_minimal.h`, exposing only public
Clang AMDGPU work-item/grid/barrier builtins and HIP-compatible qualifiers;
the ordinary Linux HIP include path remains. CMake builds every K/M module for
gfx1201 into `experimental/`. The `k_module_probe` resolves and launches prep
and transformer for `enc1`/`enc2`. The
identity fixture verifies all 4096 full-resolution and 1024/1536 merged FP16 values
bitwise. A second fixture enables nonzero V projections, position-only
attention, Wo, MLP/GELU and patch merge; it checks finite values and a patch
identity, then saves all inputs/outputs for `k_layer_validate.py`. That script
runs the existing `pwin_model.py` and measures max absolute/relative error and
PSNR. The code object remains in `experimental/k/`, outside ZLUDA's override
path, until the other K layers and real captured weights/activations pass.
The enc2 fixture additionally enables learned Q/K attention and softmax.

## External interop blockers

The installed HIP headers define D3D12Resource/D3D12Heap external memory and
D3D12Fence external semaphore types. Their presence is not proof the Windows
runtime implements them. Validate imports, mapping, GPU waits/signals and
cleanup using an independent round trip before modifying the shim.
Use HIP/DXGI device identity to prevent importing allocations across adapters.
NT HANDLE ownership must be explicit; Win32 import is not fd ownership transfer.
Test repeated create/destroy cycles as upstream noted a mapped-buffer leak on
Linux ROCm. CPU verification copies are allowed in tests, never in the fast path.

Implemented independent test: DXGI LUID matches HIP, an R32_UINT texture is
copied to a shared DEFAULT-heap linear buffer in VRAM, HIP maps that D3D12
resource, checks every input element and XORs it, D3D12 copies it back to the
texture and finally to a readback buffer solely for test verification. Shared
D3D12 fence values sequence both queues within the same iteration. There is no
CPU data copy between the APIs; native tiled texture import is not yet tested.
The future shim needs GPU texture/linear conversion or validated direct arrays.
The caller closes its original NT HANDLEs after HIP releases imports.

Stable HIP SDK `7.2.60201` fails mapped-buffer lifetime testing. `resource` and
`import` modes are stable, but `map` leaks one HANDLE and 262144 bytes per cycle.
The runtime's extra `view->retain()` is the cause; AMD removed it in
[CLR commit 529f6b1](https://github.com/ROCm/clr/commit/529f6b1641de436dafbb095d5438b0fbf773765d).
The source backport is retained at `patches/rocm/0001-external-memory-view-ownership.patch`.
No pointer/refcount or DLL binary hacks are used. Current TheRock passes both
the 64-cycle map lifetime test and the 32-cycle synchronized texture round trip.
The runner automatically isolates resource/import/stream lifetimes on failure.
Do not declare stable SDK 7.2 acceptable for the leak-free fast path.

## Windows NGX initialization

`tools/windows/ngx_cuda_init_probe.cpp` retains the physical HIP/PCI device
check, initializes a ZLUDA primary context, loads the user-supplied DLLs,
discovers DLSS CUDA capability and tests the existing MSVC parameter accessor.
This does not create/evaluate a DLSS feature or assert transformer correctness.

The supplied driver `_nvngx.dll` 32.0.16.1714 uses the **driver/snippet**
three-argument `NVSDK_NGX_CUDA_Init(appId, dataPath, sdkVersion)` ABI. The
four-argument SDK client interface used by the upstream shim passes a pointer
as the version and returns `0xBAD0000C` (OutOfDate) for this binary. The two
interfaces are distinguished in NVIDIA's public `nvsdk_ngx.h` by
`NGX_SNIPPET_BUILD`. The Windows backend must keep this distinction. The probe
has explicit `--ngx-abi driver` (default) / `sdk`; it does not guess and retry.

Upstream Windows ZLUDA NVAPI lacks physical GPU enumeration, architecture and
logical GPU/LUID functions used by NGX. The compatibility target
`nvapi-compat/nvapi64.dll` implements those **public NVAPI** APIs through HIP;
unknown queries are logged and forwarded to the supplied ZLUDA NVAPI backend.
It validates version/size and includes `NV_LOGICAL_GPU_DATA_V1.reserved[8]`
(568-byte x64 ABI). Concurrent initialization is synchronized. The AD102/AD100
identity is explicitly a CUDA/NGX network profile matching d4r's sm_89 kernels;
physical detection and native compilation remain gfx1201. It is currently a
probe backend, not a complete game-wide NVAPI replacement. A separate
`nvapi-trace` build forwards every response unchanged for diagnosis.

Driver init's default feature search needs the DLSS DLL next to the executable.
The runner creates a unique private `%TEMP%/d4r-ngx-<GUID>` directory containing
the probe and copies of the two supplied DLLs. Paths and original hashes/versions
are logged; the directory is removed after the subprocess ends. Neither DLL is
placed in dist/, git, or the diagnostics ZIP. No private NVAPI function is
invented and no NVIDIA binary is patched.

References: [NGX API signatures](https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx.h),
[public NVAPI ABI](https://github.com/NVIDIA/nvapi/blob/main/nvapi.h),
[public NVAPI interface IDs](https://github.com/NVIDIA/nvapi/blob/main/nvapi_interface.h).

## Authoritative references

* [AMD HIP SDK Windows support matrix](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)
  lists RX 9070 XT as gfx1201 with runtime and SDK support, release 7.2.0.
* [ZLUDA Windows SDK setup](https://github.com/vosen/ZLUDA/blob/ee2f25a180099fa42f36b2346732e1f2470a03ad/docs/src/hip_sdk.md)
  distinguishes official SDK and TheRock nightlies.
* [ZLUDA build](https://github.com/vosen/ZLUDA/blob/ee2f25a180099fa42f36b2346732e1f2470a03ad/docs/src/building.md),
  [trace instructions](https://github.com/vosen/ZLUDA/blob/ee2f25a180099fa42f36b2346732e1f2470a03ad/docs/src/troubleshooting.md).
* [LLVM AMDGPU target guide](https://llvm.org/docs/AMDGPUUsage.html),
  [Clang AMDGPU builtins](https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td).
* [AMD RDNA4 matrix cores](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/),
  [RDNA4 WMMA guide](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/),
  [RDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).
* [TheRock Windows development](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md).

## Build, test, package

Run from the repository root in PowerShell. The setup installs tools only inside
`.tools`, verifies downloaded SHA256 values, and does not install a driver/SDK.

```powershell
powershell -NoProfile -File scripts/windows/setup-windows-tools.ps1
powershell -NoProfile -File scripts/windows/build-windows-rdna4.ps1
powershell -NoProfile -File scripts/windows/test-windows-rdna4.ps1
```

For the currently validated external-memory runtime, use the separate profile:

```powershell
powershell -NoProfile -File scripts/windows/setup-windows-tools.ps1 -RuntimeProfile therock
powershell -NoProfile -File scripts/windows/build-windows-rdna4.ps1 -RuntimeProfile therock
powershell -NoProfile -File scripts/windows/test-windows-rdna4.ps1 -RuntimeProfile therock
```

This builds into `build/windows-rdna4-therock` and installs into
`dist/windows-rdna4-therock`. The stable profile stays available for reproducing
the runtime bug. It is expected to return failure at the lifetime gate.

One command adds the local NGX initialization gate after eight diagnostics:

```powershell
powershell -NoProfile -File scripts/windows/test-windows-rdna4.ps1 -RuntimeProfile therock -NgxCore "C:/Users/Administrator/d4r/_nvngx.dll" -DlssDll "C:/Users/Administrator/d4r/nvngx_dlss.dll"
```

Success with the two local NGX DLLs requires all eleven tests, including both `PASS K_REFERENCE` checks and
`PASS NGX_INIT ... sr_available=1`. The NGX probe still reports
`transformer_executed=0`: the standalone K launch has not been integrated into
DLSS. On NGX failure, CUDA trace and NVAPI query
logs are collected automatically. NVIDIA DLLs remain local test inputs.

The build accepts `-HipRoot`, `-ZludaRoot`, `-ToolchainRoot`, `-BuildDirectory`
and `-InstallDirectory`. MSVC users can invoke CMake from a Developer Shell
without the portable MinGW compiler. Equivalent direct build:

```powershell
cmake -S . -B build/windows-rdna4 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DD4R_HIP_ROOT=C:/Program Files/AMD/ROCm/7.2" "-DD4R_ZLUDA_ROOT=C:/path/to/zluda"
cmake --build build/windows-rdna4
ctest --test-dir build/windows-rdna4 --output-on-failure
cmake --install build/windows-rdna4 --prefix dist/windows-rdna4-diagnostics
```

The one-command diagnostic runner is also installed at
`dist/windows-rdna4-diagnostics/test-windows-rdna4.ps1`. It requires the selected
HIP SDK and ZLUDA directory; the developer setup auto-discovers local defaults.
Success: exit 0, every selected test passed in `summary.json`, `PASS HIP`, `PASS CUDA`,
`PASS INTEROP_LIFETIME` and `PASS INTEROP` in stdout. Failure: send the **single printed ZIP path**; it contains
stdout/stderr, timeout/exception exit code, DLL paths/versions/hashes, driver/OS
inventory and a minidump for an unhandled exception. Upstream trace is attempted
automatically on CUDA failure when `trace/nvcuda.dll` is available. The trace
DLL forwards to the real DLL through `ZLUDA_CUDA_LIB`; explicit loading no
longer bypasses the trace by accidentally passing the real DLL to the launcher.

The HIP module compiler uses the selected SDK, `--offload-arch=${D4R_GPU_ARCH}`
(gfx1201 by default, optionally gfx1200),
wave32 and no host/device library dependency. Host code uses the installed HIP
header ABI and loads that SDK's `amdhip64_7.dll` by absolute path. CUDA loads the
provided ZLUDA DLL by absolute path, matches the adapter against HIP PCI identity,
tests two context lifecycles, JITs embedded PTX and verifies an integer pattern
and out-of-range sentinels. No CUDA SDK is required.

No NVIDIA DLL is needed for M1/M2. Later NGX tests accept user-supplied absolute
paths; NVIDIA software is never downloaded or added to git by these scripts.

## Known limitations and results

2026-09-29: installed `hipInfo.exe` passed device enumeration in this workspace:
RX 9070 XT, gfx1201, warpSize 32, 15.92 GiB total, 15.77 GiB free.
2026-09-29 hardware results: Windows 11 Pro 10.0.26200, AMD display driver
`32.0.31041.1004`, HIP runtime/driver API `70260201`. HIP struct size 1472 bytes;
physical device PCI `0000:03:00`. Native HIP module: 32 successful iterations,
4099 elements with changing launch counts and seeds, guard verified. CUDA PTX:
32 iterations each in primary/created contexts, same pattern/guards verified,
all context/module/memory teardown calls successful. ZLUDA binary SHA256
`51dd32dc116a6c14c7a4bf5acf620bba0546ca7ec5f8a6e2148dd68f1714b165`.
Logs: local `test-results/m1-m2.zip`; ignored by git.

2026-09-29 M4: TheRock profile passed all HIP/PTX checks plus D3D12/HIP round
trip (32 iterations, 16384 elements each), exact integer output and GPU fence
synchronization. Map lifetime: 64 import/map/free/destroy cycles, HANDLE count
267 after both warmup and the last cycle, free VRAM 16926011392 bytes throughout
the measured steady state. Full interop cycle HANDLE count also stays constant
after warmup; stream teardown lowers it. Recreating streams did not resolve the
stable runtime mapped-view bug; the fix is the runtime ownership correction.

Local proprietary inputs supplied by the user: `nvngx_dlss.dll` 310.9.1.0,
`_nvngx.dll` 32.0.16.1714. Neither is packaged or tracked.

2026-09-29 NGX initialization: PASS on gfx1201 with those two DLLs and the native
NVAPI topology backend. SR Available=1, FeatureInitResult=0x1, NeedsUpdatedDriver=0,
MSVC parameter roundtrip Width=640, checked parameter/context/NGX cleanup.
Local logs: `test-results/ngx-init-private-runtime.zip`. This is only the first
part of M3; feature execution, K/M correctness and the Windows shim remain open.

2026-09-29 M5: TheRock full diagnostic run including user-provided NGX DLLs:
8/8 gates PASS (`test-results/m5-k-module.zip`). CTest: 7/7 PASS. Stable HIP 7.2:
5/5 non-interop gates PASS (HIP, gfx12 WMMA, K module load, both CUDA contexts);
stable mapped-view lifetime bug still excludes its interop fast path.
WMMA raw and legacy-layout adapter each match the scalar reference exactly
over 16 changing input matrices, with both one-step and two-step accumulation.
The `enc1` K module is built and loaded on gfx1201, with functions resolved;
`transformer_executed=0` was explicit in the module probe at that milestone.
The same K source also compiles for
`gfx1101` through the unchanged gfx11 builtin branch (compile-only regression;
no RDNA3 GPU is present in this Windows machine).

2026-09-29 M6 first kernel: `enc1` runs on the RX 9070 XT with zero and nonzero
synthetic weights. The nonzero fixture exercises V, position-only attention,
Wo, MLP/GELU and patch merge. Compared with `pwin_model.py`, full output
max absolute error `0.000244140625`, PSNR `98.21 dB`; merged output max
absolute error `0.000244140625`, PSNR `97.63 dB`. No NaN/Inf, all 4096 full
and 1024 merged elements written, 3886 full elements changed. TheRock
diagnostic bundle `test-results/m6-k-enc1-final.zip` has 9/9 gates PASS;
CTest has 7/7 PASS.
Reference dependency: pinned local NumPy `2.4.6` on Python 3.11, installed by
`setup-windows-tools.ps1` into `.tools/python/vendor` (not the game runtime).
This is a synthetic one-layer check, not proof of complete K or DLSS output.

2026-09-29 M6 second kernel: `enc2` exercises nonzero Q/K/V, learned attention
and softmax, Wo, MLP/GELU and patch merge on gfx1201. Full output max absolute
error `3.81469727e-06`, max relative error `0.00166893` (denominator floor
`1e-3`), PSNR `130.39 dB`; merged output max absolute error `3.81469727e-06`,
PSNR `132.49 dB`. All 4096 full and 1536 merged elements written, no NaN/Inf.
Both layers pass 32 identity iterations and the nonzero numpy reference.
Bundle `test-results/m6-k-enc2-final.zip`: 11/11 gates PASS. CTest: 8/8 PASS.
The generic probe/reference runner selects `--kernel-name enc1|enc2`; both
objects remain under `experimental/k` and are not used as DLSS overrides yet.

Historical stable HIP SDK 7.2 compiler caveat (pre-upstream legacy adapter):
its `-O2` code object for `enc1` produces
NaN already on the identity fixture (first element `0xfe00` instead of
`0xbc00`). Its `-O1` object also miscomputes the identity fixture; `-O0`
passes but is unsuitable for the fast path. Cross-testing isolates the problem
to the stable compiler/code object: the stable-built `-O2` object fails under
TheRock runtime, while the TheRock-built `-O2` object passes under stable HIP
runtime. The independent WMMA probe passes under both compilers. That legacy
path required the TheRock compiler for native K. The 2026-09-30 native-layout merge
supersedes this failing path: current enc1/enc2 fixtures pass with stable's
`-O3` objects too. Keep TheRock runtime for leak-free interop.

The initial audit found an upstream validation blocker: `swin_model.py` imported
absent `model_enc3`. Upstream commit `dbef4b2` restores this file; the missing
dependency is resolved, while real GPU M reference validation is still pending.

Environment-specific issues resolved: sandbox disallows writes to `.git` and
launching the MinGW child compiler, so branch/commit/build require scoped tool
approval in this managed environment. HIP_PATH ends in a backslash; build script
normalizes it before PowerShell 5 argument quoting. Test runner uses .NET Process
to retain reliable exit codes and concurrently drain both diagnostic streams.
No end-to-end K/M/DLSS/OptiScaler support is claimed by these diagnostic milestones.
