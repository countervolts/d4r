# Linux startup and GPU compatibility diagnostics

Keep one complete runtime together when comparing GPUs: the shim, nvcuda bridge, ZLUDA, patched D3D12 pair and
native kernels. New shim/bridge logs include a source build identifier. Record file SHA-256 hashes as well when
reporting failures; an old log format is not a controlled comparison with a new build.

## Capture a failing run

Add these entries to the existing `[Env]` section of `d4r/d4r.ini` (or add that section once):

```ini
[Env]
D4R_CUDA_VERBOSE = 1
D4R_CUDA_TRACE = Z:\tmp\d4r-cuda-trace.log
```

Use no shell quotes in INI values. This saves the bridge trace to `/tmp/d4r-cuda-trace.log` on the machine running
the game. The bridge also writes to stderr. ZLUDA's own compiler/native-override messages require Proton's stderr
log separately. Verbose tracing can affect timing; remove it after collecting the run.

Save the CUDA trace and `d4r_nvngx.log` separately for each successful/failed launch. The trace appends, while the
default shim log is replaced on a fresh process. For a driver recovery or corrupted display, also save the kernel
journal covering that run:

```sh
journalctl -b -k -o short-monotonic --no-pager > d4r-kernel.log
```

`CUDA_CREATE_BEGIN`, `CUDA_CREATE_WORKER_BEGIN`, `CUDA_CREATE_NGX_BEGIN/END` and `CUDA_CREATE_END` bracket feature
creation. A worker still waiting logs progress every 15 seconds without abandoning resources still in use.
`CUDA_BEGIN api=...` is recorded before module loading and, with verbose tracing, uploads, allocations, launches
and synchronization. The following ordinary API line records its result. A launch return proves submission, not
correct pixels or completed GPU work. Bridge records include a monotonic timestamp and thread ID.

## GPU selection

The game adapter's PCI address is matched to the compute device. Other compute devices are not given its LUID,
and native kernels use the matching KFD topology entry rather than the largest GPU. When no PCI mapping is
available, one visible compute GPU is unambiguous; multiple devices are refused instead of guessed. Selection
failures report the rendering PCI address and visible compute devices. `D4R_GPU_ARCH` cannot override a known
target with an incompatible architecture.

RDNA2 keeps ZLUDA's translation fallback and its existing software-matrix K kernels. It is not assigned a native
FP8/WMMA Ray Reconstruction model. This does not establish that every translated model is stable or fast there.

## Model installer

The installer builds portable FP8 RR-D/E/F models plus separate `rr-d-gfx11` and `rr-e-gfx11` variants for compatible
RDNA3 devices. The runtime checks model arithmetic and physical matrix capabilities before constructing RR
pipelines. Unsupported models fall back to the existing CUDA path; advertised NGX availability is not a runtime
validation. Engine `auto` has the same model-directory detection in the portable shim and Python launcher.

Model build logs are printed as they start and progress every 15 seconds. Each child has a 900-second deadline;
`D4R_MODEL_TIMEOUT=SECONDS` changes it. The capture harness also limits individual GPU fence waits to 60 seconds
(`D4R_HARNESS_GPU_TIMEOUT_MS`, 1–600000, changes that bound). A failed process or incomplete/nonfinite capture
cannot be accepted solely because a replay filename exists.

Diagnostics are retained under `${XDG_CACHE_HOME:-$HOME/.cache}/d4r/model-builds/build.*`; `D4R_MODEL_LOG_DIR`
changes that root. Successful builds retain logs and remove temporary weights/captures/prefixes. Failed builds
retain their workspace, including proprietary model data; keep those artifacts local and remove them when no
longer needed. Model folders carry `files.sha256.json`; cached builds are checked before reuse, and compiler
arithmetic options participate in the cache key. Old caches without these manifests rebuild once.

These checks prevent known selection/setup failures and improve fault localization. They do not by themselves
prove that an intermittent GPU reset or corrupted frame is fixed; that requires matching traces and GPU testing.
