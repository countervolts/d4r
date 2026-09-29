# Windows / RDNA4 port

Branch: `windows-rdna4`. Target: Windows 11 x64, RX 9070 XT, **gfx1201**.
Priority: correct K, correct M, native Windows, same-frame output, then speed.
This is a work log for an implementation in progress, not a claim of DLSS support.

## Milestone and gates

Current milestone: M1/M2/M4/M5 passed; NGX init + DLSS CUDA capability discovery
passed with the user's DLLs. M6 is in progress: the first K `enc1` prep and
transformer run on gfx1201 and pass a synthetic nonzero numpy reference check.
Do not integrate NGX until the integer PTX workload is stable on the real GPU.

| Gate | Status |
| --- | --- |
| M0: branch, audit, Windows build and diagnostics | Initial implementation complete |
| M1: native HIP gfx1201 allocation, kernel, CPU verification | PASS, 32 iterations + guard verification |
| M2: Windows ZLUDA integer PTX, primary and created contexts | PASS, 32 iterations each + guard verification |
| M3: Windows NGX initialization and simple DLSS path | Init + SR capabilities PASS; Evaluate/transformer pending |
| M4: D3D12 / HIP external memory and fence round trip | PASS with TheRock; stable 7.2 has mapped-view leak |
| M5: independently validated gfx12 WMMA backend | PASS: raw + legacy adapter, 16 cases, max abs/relative error 0 |
| M6: K layers, full transformer and image validation | Partial: `enc1` executes, nonzero fixture matches reference; other layers/full network pending |
| M7: M FP16-equivalent baseline and full transformer | Pending |
| M8: standalone Windows D3D12 NGX harness | Pending |
| M9: OptiScaler integration, profiling, installation | Pending |

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

`kernels/common/wmma_backend.h` now preserves the gfx11 call and maps d4r's
16-half operand / even-odd accumulator contract to gfx12's 8-half operand /
contiguous accumulator contract. The gfx12 mapping uses an unconditional
wave32 half-exchange for every source fragment element before selection.
Performing the exchange only on lanes selected by a group-dependent branch
returned channel 9 in place of channel 1 on this card: the opposite half of
the wave was inactive at that instruction. This was caught by the CPU reference.
`tools/windows/wmma_gfx1201.hip` tests raw gfx12 WMMA, the adapter, the
half-exchange and lane ID with nonzero accumulator and one/two K16 steps.
Both stable HIP 7.2 and TheRock 10.2 produced exact 16x16 outputs across 16
different inputs; max absolute and relative errors were both zero.

`kernels/k/pwin_common.h` now calls the tested adapter. The Windows device-only
compiler uses `kernels/common/hip_device_minimal.h`, exposing only public
Clang AMDGPU work-item/grid/barrier builtins and HIP-compatible qualifiers;
the ordinary Linux HIP include path remains. CMake builds `enc1` for gfx1201
into `experimental/k/` and verifies that HIP loads both prep and transformer
entry points. The `k_module_probe` now launches prep and transformer. The
identity fixture verifies all 4096 full-resolution and 1024 merged FP16 values
bitwise. A second fixture enables nonzero V projections, position-only
attention, Wo, MLP/GELU and patch merge; it checks finite values and a patch
identity, then saves all inputs/outputs for `k_enc1_validate.py`. That script
runs the existing `pwin_model.py` and measures max absolute/relative error and
PSNR. The code object remains in `experimental/k/`, outside ZLUDA's override
path, until the other K layers and real captured weights/activations pass.

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

Success requires all nine tests, including `PASS K_REFERENCE` and
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
Success: exit 0, all five tests passed in `summary.json`, `PASS HIP`, `PASS CUDA`,
`PASS INTEROP_LIFETIME` and `PASS INTEROP` in stdout. Failure: send the **single printed ZIP path**; it contains
stdout/stderr, timeout/exception exit code, DLL paths/versions/hashes, driver/OS
inventory and a minidump for an unhandled exception. Upstream trace is attempted
automatically on CUDA failure when `trace/nvcuda.dll` is available. The trace
DLL forwards to the real DLL through `ZLUDA_CUDA_LIB`; explicit loading no
longer bypasses the trace by accidentally passing the real DLL to the launcher.

The HIP module compiler uses the selected SDK, fixed `--offload-arch=gfx1201`,
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

Stable HIP SDK 7.2 compiler caveat: its `-O2` code object for `enc1` produces
NaN already on the identity fixture (first element `0xfe00` instead of
`0xbc00`). Its `-O1` object also miscomputes the identity fixture; `-O0`
passes but is unsuitable for the fast path. Cross-testing isolates the problem
to the stable compiler/code object: the stable-built `-O2` object fails under
TheRock runtime, while the TheRock-built `-O2` object passes under stable HIP
runtime. The independent WMMA probe passes under both compilers. Until a
smaller compiler reproducer or source-level workaround is found, use the
TheRock compiler for native K and the TheRock runtime for leak-free interop.

Audit found an upstream validation blocker: `swin_model.py` imports
`model_enc3`, which is absent from the repository. M reference validation needs
that dependency restored or an independently checked replacement.

Environment-specific issues resolved: sandbox disallows writes to `.git` and
launching the MinGW child compiler, so branch/commit/build require scoped tool
approval in this managed environment. HIP_PATH ends in a backslash; build script
normalizes it before PowerShell 5 argument quoting. Test runner uses .NET Process
to retain reliable exit codes and concurrently drain both diagnostic streams.
No end-to-end K/M/DLSS/OptiScaler support is claimed by these diagnostic milestones.
