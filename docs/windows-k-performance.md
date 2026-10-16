# Windows K performance work

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
