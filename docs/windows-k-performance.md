# Windows K performance work

## Current optimization pass: strict DS exchange, 1080p and 1440p (2026-10-17)

After the position-layer barrier fix, the explicit gfx11 DS cross-half exchange passes the strict gate.
`kernels/common/wave32_exchange.h` uses `ds_bpermute` with the current wave's lane XOR 16 instead of the
volatile LDS array and its fences. No arithmetic, half rounding, weights or tensor layouts change.
Windows gfx11 K selects it through `D4R_WINDOWS_GFX11_DS_EXCHANGE`; M retains its validated LDS path.
The experimental builder retains explicit `-LaneExchange lds|ds` selection for comparisons.

Final CMake-built candidates, authentic private K model, synthetic 24-frame temporal cases, 25 frames/submit,
12 timing trials with three warmup trials excluded. Timestamps span Vulkan front, HIP network, Vulkan back
and the CPU synchronization gaps; these are not network-only or game timings:

| Output | Fixed LDS baseline | Strict DS candidate | Change |
|---|---:|---:|---:|
| 1920x1080 | 2286.4 us | 1811.9 us | 20.8% faster |
| 2560x1440 | 3587.7 us | 2885.4 us | 19.6% faster |

All five frame products match baseline hashes exactly at both sizes (120 products per case).
Evidence: `build/k-ds-cmake-1080-final-v1/summary.json`, `build/k-ds-cmake-1440-final-v1/summary.json`.
Independent manual-build comparisons also pass, at 1808.7/2877.0 us. CMake/manual code objects differ as
artifacts; the actual CMake objects receive their own full output checks. Final native objects are under
`build/k-ds-native-cmake-v2/`. The v1 CMake candidate is rejected: an intermediate CMake regex overwrote the
captured per-source flags; the capture is now saved immediately, preserving CU and QKV-split flags.

Further experiments retained no reliable additional gain: position token counts 1/2/4, shared V values,
SPIR-V size optimization of input/output, and all-WGP scheduling. Every compared candidate here is exact;
many are slower. WGP measured 2862.3 vs 2847.2 us at 1440p and 1795.7 vs 1813.9 at 1080p, within small
run-to-run differences and inconsistent across sizes. It is not promoted. Evidence is under
`build/k-hip-opt-sweep-v1/`, `k-hip-opt-ds-share-v1/`, `k-hip-opt-stage-mt1-v1/`, and `k-ds-wgp-*-v1/`.

The HIP standalone engine/model is still separate from the package's NGX feature-1 path. The package can
use the same faster K layer objects through its existing ZLUDA/NGX route; that route requires separate
whole-frame validation and CPU interop timing. Linux targets of 1.5/2.47 ms remain unmet. Further substantial
work would require the Vulkan front/output stages or the cross-API synchronization path; it is outside this
bounded optimization pass. Broader K temporal/reference-quality and production-adapter work remains open.

The final v14 ZIP's loader/INI passes 64 exact synthetic NGX K frames per size in control/candidate/candidate/control
order. CPU interop medians: 1080p 3.900 -> 3.3915 ms (13.0%); 1440p 5.4015 -> 4.820 ms (10.8%). Unlike the
standalone timestamp table, these include CPU API/copy/synchronization/NGX overhead. They are not game FPS.
Evidence: `build/k-ds-local-v14-{1080,1440}-v1/summary.json`. An earlier legacy 1080p run overlaps build activity;
its exactness evidence remains valid, but the packaged player runs supersede its timings.

Stability diagnostic: `build/k-ds-repeat-node-final-v1.log`, 300 frozen-input rounds x 8 window phases (2,400
launches), zero output-hash mismatches. Its individually instrumented network nodes total 1,926.04 us at 1440p.
That instrumentation is separate from accepted frame timings. Raw readbacks created by this pass were checked
against recorded hashes and pruned to recover 5.05 GiB; logs, summaries, hashes and final previews remain.
Cleanup inventory: `build/k-opt-readback-cleanup-v1.json`. No earlier artifacts were deleted.

## Temporal isolation follow-up

The intermittent drift is not fixed or attributed to a specific layer. New
`benchmark_k_interop.py --reset-every-frame` passes an explicit diagnostic option
to the D3D12 harness, verifies its application and records it in the report.
This mode does not represent normal temporal inference or game performance.

With the validated private CU output, 16 frames per run and four player runs:

| Isolation | Exact evaluations | Evidence under build/ |
|---|---:|---|
| Reset every frame, ABBA input batching | 64 | k-player-reset-every-frame-1080-v1 |
| All K network layers translated, CU output retained, temporal ABBA | 64 | k-player-translated-network-temporal-v2 |
| All native K, unbatched identical controls | 64 | k-player-native-temporal-controls-v1 |
| Native encoders only, translated decoders, temporal ABBA | 64 | k-player-native-encoders-temporal-v1 |
| Translated encoders, native decoders only, temporal ABBA | 64 | k-player-native-decoders-temporal-v1 |
| All native K and CU output, temporal ABBA rerun | 64 | k-player-native-temporal-abba-v2 |

Each row has an authoritative `summary.json`. Reset-every-frame also produces
one identical output hash across all frames within each run. Thus accumulated
state matters in the earlier failing case, but that alone does not establish a
history implementation defect. The mixed native/translated cases do not isolate
an individual layer, and the full native ABBA rerun passes after the earlier
failure. Initialization, cache state, scheduling and arithmetic remain possible
investigation targets; no one cause is demonstrated.

Diagnostic manifests select translated or native K layers while preserving the
same object files, private CU output, dependencies and baseline. Runtime logs
verify actual translated enc0/dec0 selection for the all-translated case. Those
manifest variants remain private under `build/k-network-*-native-output-cu-v1/`
and `build/k-translated-network-output-cu-v1/`; no source kernel, runtime default,
game or delivered ZIP was changed to use them. First translated-network fixture
v1 failed before NGX because its setup command had a quoting error and did not
create the selected directory. A native Python fixture setup corrected it; v2
passes. No security policy block or bypass occurred.

The reset diagnostic's timings are not normal K timings. Mixed/translated cases
also are not proposed performance configurations. The full native rerun records
CPU medians 3.018/2.9155 ms, unbatched/batched, but earlier drift still prevents
promoting a default or claiming deterministic behavior. No GPU-time/Linux parity,
image-quality or real-game speed conclusion follows from these isolation runs.

## Latest: rejected kernel candidates and unresolved player temporal exactness

No new kernel or batching default is promoted. The current delivered ZIP remains
v8 with batching off. The working tree adds an experimental
`[Interop] BatchInputCopies` switch, default false, plus parser/environment tests
and checker support. Missing/invalid values keep older unbatched behavior.
CPU player-config tests pass; checker syntax parses. Exactness in the player
performance fixture remains unresolved, so keep the switch false.

`benchmark_k_interop.py --control-native DIR` selects an explicit verified native
baseline. `--player-config` stages dependencies and selects batching/native paths
through actual INI. It preserves per-run player logs. `--sequence controls` or
`batches` repeats identical settings to diagnose determinism. Failed comparisons
record mismatching frame names and hashes; they do not receive accepted timing.
`compare_k_outputs.py` measures preserved half-output differences without GPU work.

New private experiments, all artifacts retained and excluded from ZIPs:

- WGP scheduling for the existing native output override: 64 exact 1080p frames,
  with CU/WGP warmed CPU interop medians 3.048/3.0455 ms (no reliable gain).
  At 1440p the candidate failed the exact-output gate: five of 16 compared
  frames differ, 14,836 components, max absolute difference 0.00537109375;
  all values finite. Candidate rejected. Evidence:
  `build/k-output-wgp-vs-cu-{1080,1440}-v1/` and 1440p `mismatch-analysis.json`.
- CU input override: recipe changes only eight round-to-zero idioms; no surface
  stores matched this LDR input variant. Four of 16 frames differ, 993 components,
  max absolute difference 0.00048828125; finite. Candidate rejected. Evidence:
  `build/k-input-cu-vs-output-cu-1080-v1/mismatch-analysis.json`.
- Legacy/environment mode with validated private CU output plus existing input
  batching: 128 exact frames across 1080p/1440p ABBA comparisons. Warmed CPU
  interop medians: 3.072/2.9305 ms at 1080p, 4.4605/4.3345 ms at 1440p,
  unbatched/batched. `build/k-output-cu-batch-{1080,1440}-v1/summary.json`.

Player-mode checks exposed an important limit to those narrow successful runs:

- In the baseline-kernel 16-frame ABBA run, both batched runs match first control
  exactly, but the final unbatched control differs in 15 frames, 38,472
  components, max absolute difference 0.01318359375. This does not isolate a
  batching defect. `build/k-batch-player-1080-v1/repeat-analysis.json`.
- Four identical unbatched player runs of eight frames each pass all 32 exact
  frames. `build/k-player-control-repeat-1080-v1/summary.json`.
- A 16-frame player comparison using the validated private CU output differs
  between control and batched run in eight frames, 1,416 components, max absolute
  difference 0.00048828125. First difference is frame 8, with RGB differences
  spread across the image and unchanged alpha. Candidate is not accepted.
  `build/k-batch-player-output-cu-1080-v1/difference-locations.json`.

These results leave temporal determinism, arithmetic/initialization and ordering
as investigation targets. They do not prove any one of those causes. The kernel
candidates failed acceptance, but baseline drift prevents attributing every
difference solely to a candidate. No relaxed tolerance was substituted for the
exact gate. Private candidates, validated CU control and original runtime remain
separate. No GPU-time/Linux parity or game-speed claim, game access, driver/GPU/
security change, commit or publication occurred.

Linux's reported 1.5 ms at 1080p and 2.47 ms at 1440p remain targets, not verified
Windows results. The synthetic measurements below include CPU synchronization,
format/array copies and NGX execution. They are not directly comparable to a
GPU network-only timer or a game's frame time.

## Reproducible interop comparison

`tools/windows/benchmark_k_interop.py` uses an existing private gfx1101 runtime
and the native D3D12 harness. It runs control/batch/batch/control, with 16 frames
per run and the first four excluded from timing. Every frame is read back;
candidate output hashes must match the control frame for frame. The harness
also rejects nonfinite or uniform outputs. No game is launched or modified.

Use the current installer core, not an older isolated NGX fixture:

```powershell
python tools/windows/benchmark_k_interop.py `
  --probe build/windows-rdna4-therock-gfx1101/d4r_d3d12_dlss_probe.exe `
  --shim build/windows-rdna4-therock-gfx1101/d4r_nvngx.dll `
  --runtime build/got-k-test/work/package-gfx1101-current/d4r `
  --core build/driver-617.14-core-check/_nvngx.dll `
  --dlss build/got-k-test/isolated-sr/sr/nvngx_dlss.dll `
  --output build/k-new-benchmark --resolution 2560x1440
```

Paths above identify local ignored artifacts; provision matching artifacts on
another machine. The runtime argument supplies HIP, ZLUDA, NVAPI, the gfx1101
pixel converter and native K objects. Output must be a new folder under
`build/`. Each process has a 180-second limit. Reports retain executable/DLL
hashes and native object hashes. `--profile-stages` adds CPU stage diagnostics;
its timings include logging overhead and must be reported separately.

The harness otherwise defaults to half-resolution input. This runner explicitly
sets nominal Quality input: 1280x720 for 1920x1080 output; 1706x960 for 2560x1440.
Games can round the render width differently (GoT used 1712 at 1440p).

## Evidence, RX 7700 XT, 2026-10-16

The 617.14 core SHA256 is
`66767018c36b3bab46398dade3adf173daa3730fda75965689ea848c9bc4e79b`;
SR 310.9.1 matches the installer pin. Native K encoders are reported loaded in
the captured stderr. Baseline RGBA16F inputs/output need no format conversion.

At 1706x960 to 2560x1440, pooled warmed CPU interop medians were 5.2615 ms
control and 5.0800 ms batch. All 64 frame outputs passed and matched exactly
between modes. Evidence: `build/k-interop-batch-quality-1440-core617/summary.json`.
At 1280x720 to 1920x1080, the same check passed all 64 frames; pooled warmed
medians were 3.5730 ms control and 3.3820 ms batch. Evidence:
`build/k-interop-batch-quality-1080-core617/summary.json`.
These short runs suggest only a small batching benefit; defaults remain unchanged.
This does not establish image quality on realistic ray-traced/game content.

Earlier investigation runs used core 596.36 or stage profiling. Their artifacts
are retained separately and are not the current-core baseline. Batched input
copies alone do not establish Linux speed parity. Next optimization needs paired
GPU kernel timings and an exact-output comparison, particularly for the native
gfx11 kernels' Windows lane-exchange workaround and the translated texture stages.

## Kernel experiments and GPU profile

The experiment builder now accepts an explicit gfx11/gfx12 target. gfx11 keeps
its Windows LDS lane-exchange workaround and strict FP16 accumulation. Gfx12-only
tiling/storage experiments are rejected on gfx11. Compiler arguments, target,
object hash and ISA remain in the experiment output directory.

`benchmark_k_interop.py --candidate-native DIR` compares an experimental native
folder against the existing runtime, using identical interop settings. Candidate
runs must match every output file before their timing is accepted. No candidate
is installed into the existing runtime or a game.

A strict enc1 `ds_bpermute` replacement failed exact temporal output comparison:
eight of 16 frames differed, maximum absolute difference 0.00048828125. The fresh
LDS control passed the same comparison. The DS source experiment was removed;
its local artifacts remain under `build/k-bpermute-strict-enc1-quality-1080/`.
Earlier non-strict experiments omitted `D4R_K_FP16_BASELINE` and cannot isolate
shuffle behavior. The DS builtin also folded to `v_permlanex16`; explicit assembly
was needed to test an actual DS instruction.

Enc1 WGP mode passed all 64 frame comparisons but showed no reliable speed gain:
control/candidate pooled warmed CPU interop medians 3.5455 / 3.5865 ms. No default
or package change. Evidence: `build/k-enc1-wgp-quality-1080/summary.json`.

`--profile-kernels` enables deferred GPU events with legacy-stream profiling
explicitly allowed for this synthetic diagnostic. This changes stream dependencies
and substantially raises CPU interop time; do not compare its wall times with the
uninstrumented baseline. In the first control run, the last-four-sample GPU medians
were approximately:

| Stage | GPU ms |
|---|---:|
| translated output `hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel` | 0.820 |
| native dec0 | 0.301 |
| native enc0 | 0.293 |
| native dec1 | 0.261 |
| native enc1 | 0.213 |

All 32 instrumented frames passed exact output checks across control/candidate.
Raw logs and per-run stage analysis are in
`build/k-enc1-wgp-gpu-profile-1080/gpu-stage-analysis.json`. These are instrumented
stage measurements, not an uninstrumented total. The translated output pass is
the largest measured stage and is the next optimization target. The existing
private texture recipe needs `d4r_emit.exe`, which is absent from the current
workspace's tool/runtime/build directories. No Linux speed parity is claimed.

## Offline emitter provisioning

The dependency checkout is now local under ignored `external/ZLUDA`, pinned to
`ee2f25a180099fa42f36b2346732e1f2470a03ad` with this repository's Windows patches.
LLVM is pinned to `ff4dc1f7c9e1c64d4d69e40f4ed30c2280a96dfd` plus the audited
gfx1154 backport. Rust 1.98.1 and locked Cargo dependencies are provisioned locally.
All four PTX helper variants built with TheRock's compiler. They have not yet
been linked or exercised by the emitter. LLVM library compilation is in progress.

After the LLVM build succeeds, run:

```powershell
./scripts/windows/build-zluda-windows.ps1 -EmitterOnly
./scripts/windows/build-native-texture.ps1 `
  -DlssDll build/got-k-test/isolated-sr/sr/nvngx_dlss.dll `
  -Kernel hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel `
  -HipRoot .tools/therock-10.2.0a20260929/_rocm_sdk_core `
  -ZludaRoot dist/zluda-windows-emitter -GpuArch gfx1101 `
  -OutputDirectory build/private-k-output-gfx1101
```

Emitter-only mode does not build or replace `nvcuda.dll`, NVAPI, or the tracing
runtime. It writes the offline emitter and its local C++ DLL dependencies to a
separate directory. The texture recipe resolves compiler tools from either the
TheRock `lib/llvm/bin` layout or the HIP SDK `bin` layout; derived output must
stay under ignored `build/`. Each extraction/compile operation has a 600-second
limit. Output overrides remain unvalidated until full-frame comparison passes.

## K native output override validated (2026-10-16)

LLVM completed all 2,322 library build steps. The offline emitter now builds and
passes three gfx1101 compile fixtures (pattern, f16 reference MMA, f16 WMMA).
The validated CUDA/NVAPI runtime was not rebuilt or replaced.

Two build compatibility fixes were required for TheRock's LLVM 24 frontend:

- Remove stale ThinLTO summaries after stripping unused HIP/LLVM globals.
- Assemble helper IR with the pinned LLVM 22 assembler. A small compatibility
  pass preserves float bit patterns, translates the dynamic denormal attribute
  to LLVM 22's equivalent, and drops optimization promises the older reader
  cannot express. It does not enable relaxed math. Five tests cover finite float
  widening across exponent boundaries, NaN payloads, half constants, double/string
  preservation and conservative attribute handling. End-to-end exact output
  checks below are the runtime gate.

The private native store override for
`hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel` built for gfx1101. The PTX recipe
replaced 13 formatted stores, four b32 stores and eight round-to-zero stores.
The runtime log confirms the new native output module is loaded.

Paired control/candidate/control-order measurements (control, candidate,
candidate, control), 16 frames per run, first four excluded:

| Nominal Quality geometry | Control CPU interop ms | Candidate CPU interop ms |
|---|---:|---:|
| 1280x720 to 1920x1080 | 3.791 | 3.440 |
| 1706x960 to 2560x1440 | 5.318 | 4.744 |

All 128 frame outputs were finite/nonuniform and matched control bytes exactly.
These CPU measurements include synchronization and copies; they do not establish
Linux GPU-time parity or game speed. No game was accessed.

A separate deferred-event diagnostic passed 32 additional exact frame checks.
The output-stage GPU medians were 0.812/0.815 ms in the two controls and
0.535/0.532 ms in the two candidates. Legacy-stream events add dependencies;
these instrumented stage measurements are not uninstrumented frame totals.

Private artifacts: `build/private-k-output-gfx1101/` and combined
`build/k-output-native-gfx1101/`. Evidence:
`build/k-output-native-quality-{1080,1440}/summary.json` and
`build/k-output-native-gpu-profile-1080/output-stage-analysis.json`.
The override derives from the user's NVIDIA DLL and remains under ignored build/;
it is not included in the previously delivered local ZIP. Other K input/output
variants, game performance and automatic local override generation remain open.
