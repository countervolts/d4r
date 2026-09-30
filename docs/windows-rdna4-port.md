# Windows / RDNA4 port

Branch: `windows-rdna4`. Target: Windows 11 x64, RX 9070 XT, **gfx1201**.
Priority: correct K, correct M, native Windows, same-frame output, then speed.
This is a work log for an implementation in progress, not a claim of DLSS support.

## Milestone and gates

Current milestone: M1/M2/M4/M5 passed; native NGX Create/Evaluate on the simple
requested E path passes with the user's DLLs. M6 is in progress: K `enc1` and `enc2` prep and
transformers run on gfx1201 and pass synthetic nonzero numpy reference checks.
Do not integrate NGX until the integer PTX workload is stable on the real GPU.

| Gate | Status |
| --- | --- |
| M0: branch, audit, Windows build and diagnostics | Initial implementation complete |
| M1: native HIP gfx1201 allocation, kernel, CPU verification | PASS, 32 iterations + guard verification |
| M2: Windows ZLUDA integer PTX, primary and created contexts | PASS, 32 iterations each + guard verification |
| M3: Windows NGX initialization and simple DLSS path | Init, SR capabilities, Create/Evaluate PASS; four finite synthetic frames; K/M pending |
| M4: D3D12 / HIP external memory and fence round trip | PASS with TheRock; stable 7.2 has mapped-view leak |
| M5: independently validated gfx12 WMMA backend | PASS: raw + legacy adapter + upstream layout, max abs/relative error 0 |
| M6: K layers, full transformer and image validation | Partial: `enc1`/`enc2` execute and match nonzero references; other layers/full network pending |
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
Only `enc1`/`enc2` currently have GPU layer correctness checks.

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

Next gate: validate the remaining K layers against references/captures, then
build the d4r-patched Windows ZLUDA and implement same-frame Windows NGX Evaluate
with explicit D3D12 queue submission ownership. Full K precedes M and native
FP8/performance work. No complete DLSS or OptiScaler game result is claimed.

## Native Windows ZLUDA build and Evaluate work (2026-09-30)

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
  `20260922`; no Linux build host or runtime is used.
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
