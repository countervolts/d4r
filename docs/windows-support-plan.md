# Native Windows support implementation plan

Status: proposed implementation sequence, 2026-10-22. The existing D3D12
prototype is a validated starting point, not Windows game support.

## Target and scope

First release: preset K Super Resolution in D3D12 games on supported RDNA4
hardware, initially validated on RX 9070 XT. Run inference directly in the
game's D3D12 command list, with same-frame output and persistent GPU resources.
No Wine, patched vkd3d, CUDA bridge, ZLUDA or HIP dependency in the Windows
evaluation path. Prepare the model offline before launching a game.

Keep native Vulkan/HIP on Linux. The Linux baseline is materially faster than
the initial D3D12 port on RADV. Windows support must not make translating the
whole game through vkd3d a requirement.

Engine improvements made for Linux should reach Windows without a second manual
port. The Vulkan engine's shaders and host-side frame plan are where that work
lands; the HIP kernels and the Proton/vkd3d/bridge plumbing stay Linux-only.

M/L, Ray Reconstruction, E/F, D3D11, native Vulkan games on Windows and support
for additional GPU generations are follow-up milestones. Unsupported requests
must return accurate capabilities/errors so the host can choose another
upscaler. Do not promise a CUDA fallback on Windows until it actually exists.

## Evidence and outstanding questions

- The plain K block runs through AMD WMMA on an ordinary Windows-created D3D12
  device using the reserved AGS register space; no AGS DLL was needed in tests.
- The prototype preserves existing row-major weights and layer parameters.
  Matrix tiles and synthetic full-block output pass independent CPU and Vulkan
  comparisons. Captured-model and game-quality validation are still required.
- On Windows, the new scheduling variants reduced the synthetic 32x32 plain
  block from about 122 us to 72 us, with bit-identical output. This does not
  establish full-frame performance or a RADV improvement.
- The original Linux comparison measured 42.57 us native Vulkan versus 91.52 us
  D3D12/vkd3d. Retest the variants there before choosing any Linux D3D12 path.
  Variant rankings already differ between the two systems.
- The prototype owns its queue, uploads, waits and readbacks. None of that is
  yet an embeddable, asynchronous game engine.
- The Vulkan engine runtime (`runtime.cpp`, `vk_dispatch.cpp`) has no
  Linux-specific dependencies and already records into a caller-owned command
  buffer. AMD's Windows driver exposes `VK_KHR_cooperative_matrix` on RDNA3/4,
  so it is expected to run natively on Windows; this is unverified.
- `k_math.hlsli` and the 448-byte parameter layout already mirror GLSL helpers
  and C++ structs by hand: a second source of truth that can silently diverge.
- `tools/d4r_nvngx_shim.cpp` initializes CUDA/NGX libraries before engine routing;
  its resource-retention path depends on vkd3d extensions. Windows needs an
  earlier backend decision and an explicit submission/completion contract.
- Device-specific WMMA availability must be established on the game's borrowed
  device, not inferred merely from its vendor ID or Shader Model 6.6 support.
- Windows model provisioning must work without asking the game runtime to
  extract weights or initialize NVIDIA's CUDA backend.

References: [prototype and ablations](../engine/d3d12/README.md),
[Linux measurements](../engine/d3d12/linux-radv-benchmark.md),
[current architecture](architecture.md).

## Progress

Milestone 1, as of 2026-10-22 (RX 9070 XT, AMD 26.9.1). Details:
[host contract and results](../engine/d3d12/host-contract.md).

- Done: a record-only host on a borrowed device (`engine/d3d12/host.h`) with
  explicit submitted/discarded reporting and fence-based retirement. The
  `d4r-host-test` lifetime checks pass, including with GPU-based validation.
- Done: the Vulkan engine builds natively with MSVC (`engine/CMakeLists.txt`).
  The AMD Windows driver exposes `VK_KHR_cooperative_matrix`.
- Done: interop cost measured. Copies take 0.19–0.42 ms and the round trip
  about 0.1 ms per frame, depending on resolution.
- Done: native Windows `d4r-k` K timings with a synthetic-weight model
  (`engine/synth_k.py`; timing only, the images are meaningless). The model
  builds and loads on Windows. That needed two engine fixes:
  - AMD's Windows compiler crashed on 8 of 11 layer shaders, on one-element
    arrays of cooperative matrices (fixed in `layer.comp`; TPW=2 SPIR-V
    unchanged);
  - `d4r-k` did not enable `vulkanMemoryModel`, which the layer shaders use.
- Result, 1920x1080 -> 3840x2160, synthetic frames: 5.78 ms full frame,
  against 2.88 ms on RADV. The eleven layers sum to 4.36 ms on Windows,
  against 1.46 ms on RADV. Input is 448 us against 316 us; reconstruction is
  about the same (883 us against 904 us). Other sizes: 2560x1440 -> 3840x2160
  5.95 ms, 1708x960 -> 2560x1440 3.89 ms.
- Done: model provisioning on native Windows, with no Linux, Wine or HIP SDK.
  The Linux route is reused: NGX's CUDA backend runs on ZLUDA, and the
  bridge saves what NGX uploads. What made it work:
  - The bridge (`tools/wine_nvcuda_bridge.c`) builds as an ordinary
    `nvcuda.dll` with MinGW (`scripts/build_windows_nvcuda_bridge.sh`).
    `tools/d4r_bridge_host.h` holds the platform layer, so the source is
    shared with Wine.
  - `tools/d4r_nvapi_windows.c` reports the AMD adapter as an AD100. NGX
    loads `nvapi64.dll` from System32 only, so the shim preloads it
    (`D4R_NVAPI`).
  - ZLUDA is built for Windows. The PTX helpers come from the Windows HIP SDK,
    with LLVM 23 attributes written as ZLUDA's LLVM 22 reads them
    (`scripts/build_zluda_ptx_helpers.sh`).
  - The Vulkan test app (the shim's native Vulkan frontend) renders the frame.
    The D3D12 frontend needs vkd3d-proton.
  - A first run compiled DLSS's kernels: about 30 minutes, most of it the 11
    layers at about 100 s each. Since the bridge finds the GPU through HIP on
    Windows, the bundled native kernels run those layers on RDNA3/RDNA4 and a
    test frame takes seconds with ZLUDA's cache empty (K: 17 s, RX 9070 XT).
  - HIP comes from the AMD driver (`amdhip64_7.dll`, `amd_comgr_3.dll` in
    System32).
  - `extract_k.py`, `prepare_k.py --lut-only` and `compile_k.py` then run
    unchanged. `packaging/build_engine_models.py` is the Windows builder.
- Done: native Vulkan frontend + engine on Windows. The shim's Vulkan frontend
  records K into the application's command buffer with the Windows-built
  model. It did not detect cooperative matrices, so it loaded the v_dot2
  network on every GPU. That is fixed for both systems.
- Check, panning textured scene (Vulkan test app, `D4R_HARNESS_PAN=2`,
  640x360 -> 1280x720): engine against NGX on ZLUDA, same Windows machine.
  - RGB PSNR rises from 27.8 dB on frame 1 to 35.4, 39.6 and 41.2 dB on frames
    3, 7 and 11.
  - The early frames differ because NGX's reset frame comes out black in this
    app, while the engine's does not.
  - This sits inside the Linux engine-versus-reference range (README, about
    35–60 dB per frame). It is not a game-quality validation.
- Open: the `*_dot2` (RDNA2) shaders fail `vkCreateShaderModule` on AMD's
  Windows driver.
- Not started: the shim's early backend router, OptiScaler integration, and
  the choice of how the host reports submission (see the contract's open
  questions).
- Not set: the acceptable interop overhead for the decision.

## Intended architecture

```text
Shared NGX frontend
  Parameters, capability reporting, preset selection, feature lifecycle
                         |
Shared inference description and model data
  Shapes, frame settings, tensor layouts, temporal rules, test captures
  Generated parameter structs/constants; API-neutral frame plan
                         |
Shared shader source (HLSL + matrix adapter)
        DXC -> SPIR-V                  DXC -> DXIL
              /                          \
Linux Vulkan/HIP backend             Windows D3D12 backend
  Existing fast paths                 AMD WMMA via AGS intrinsics
  Existing Proton integration         Borrowed game device/command list
```

Share algorithms, data contracts and shader source; allow each backend its own
descriptors, resource transitions, pipeline cache and low-level scheduling.

- **Generated contracts.** Parameter structs, activation constants, rounding
  helpers and tensor layouts are emitted into C++, GLSL and HLSL from one Python
  description, next to `compile_k.py`. No hand-mirrored copies.
- **API-neutral frame plan.** The runtime produces a list of dispatches
  (pipeline, buffer bindings, group counts, parameters, barriers) that each
  backend executes. This is a compute-only data structure, not a general GPU
  abstraction. Pass fusion, buffer reuse and dispatch ordering then apply to
  both backends without porting. Extract CPU-only shape/frame calculations from
  the Vulkan-facing runtime header as part of this.
- **Shared shader source.** Each stage is written once in HLSL with a small
  matrix-operation adapter: AGS WMMA intrinsics for DXIL, `vk::khr` cooperative
  matrices for SPIR-V. A stage replaces its GLSL version only after it passes
  RADV correctness and performance checks; until then, the GLSL stays the Linux
  default and both are selectable.
- **Per-backend tuning as variants, not forks.** Scheduling and synchronization
  choices are defines or specialization constants. Defaults are chosen per
  (backend, GPU architecture) from a measured table, so a compiler-specific
  ranking does not split the source.

Algorithmic and structural gains carry over this way; micro-tuning specific to
ACO or AMD's Windows compiler may not, and is handled by the variant table.
Linux-only HIP kernel work does not carry over.

## Milestones and acceptance gates

### 1. Prove native host integration and model loading first

Deliver a minimal D3D12 host plus NGX/OptiScaler integration spike before porting
the remaining network. Use a simple diagnostic shader in place of inference.

- Borrow the application's device and command list; create resources and a
  pipeline without replacing D3D12 or recreating the game device through AGS.
- Show early native-backend initialization succeeds when the CUDA bridge,
  ZLUDA, NVIDIA core DLLs and vkd3d extensions are absent.
- Load/validate an existing K model, offsets and reconstruction tables on
  Windows. Detect missing/incompatible assets before reporting K available.
- Establish who supplies submission and completion information. Prefer a host
  contract with OptiScaler or a narrowly scoped queue/allocator integration.
  Merely receiving an NGX command-list pointer is insufficient for safely
  recycling upload memory or descriptors.
- Define handling of recorded-but-unsubmitted lists, allocator reset, feature
  release and device removal. Retain borrowed resources through their actual
  GPU use. Do not substitute a fixed delay for retirement.
- Build and run the existing Vulkan engine (`d4r-k`) natively on Windows against
  the same K captures. Record its full-frame and per-layer GPU times. This is
  the same-OS reference for judging the D3D12 port, and separates driver or
  compiler differences from port quality.
- Assess D3D12-to-Vulkan interop as a fallback game path: input copies into
  shared resources recorded in the game's list, a shared D3D12 fence imported
  into Vulkan, and queue-level ordering. It is subject to the same lifetime
  contract as above. It is not the intended end state, but its result decides
  the order of the remaining milestones (see the gate).

Gate: native texture input/output and repeated recording work in the harness
and an OptiScaler host without Linux components. The lifetime contract is
demonstrated, including a discarded recording and release while work is queued.
If the contract requires interception, resolve that design here rather than
after the engine is built. Windows Vulkan engine timings exist for the K
captures used in later gates.

Decision at this gate: if the native Windows Vulkan engine, run through
interop, passes the captured K checks and its complete per-frame cost
(engine time plus input/output copies and the cross-API fence wait) is
acceptable for a preview, ship the K preview on that path first. Take
milestone 6's frontend work and milestone 7's game validation directly, with
interop as the backend, and do milestones 2–5 afterwards as the replacement
D3D12 backend. Write down the acceptable overhead before measuring. Otherwise,
or if interop cannot satisfy the lifetime contract inside a game, continue in
the order below. In either case, record the result and the reason in this
document.

### 2. Establish shared contracts and the frame plan

- Generate parameter structs, constants, rounding helpers and layouts for C++,
  GLSL and HLSL from one description. Switch the existing GLSL and the prototype
  HLSL to the generated files; outputs must remain byte-identical.
- Extract the frame plan from the Vulkan runtime: the dispatch list with
  bindings, group counts, parameters and barriers. The Vulkan backend executes
  it with no change in recorded work, performance or output.
- Set up the shared HLSL build: DXC to DXIL and to SPIR-V, the matrix adapter,
  and the variant mechanism. Compile the prototype plain block both ways and
  compare its SPIR-V build against the GLSL block on RADV.
- Add the cross-backend replay check (see the policy section) to CI.

Gate: Linux replay checks and timings are unchanged after the switch to
generated contracts and the frame plan. The plain block builds for both
backends from one source and passes the same captured checks on each.

### 3. Build the reusable D3D12 runtime and asset package

- Separate immutable model/PSO ownership from per-feature activations, temporal
  history, descriptors and frame data. Share models across compatible features.
- Add a record-only engine entry point: caller-owned command list, explicit
  input/output resource descriptions and frame settings, no hidden submission.
  It executes the shared frame plan.
- Allocate persistent buffers and descriptor capacity at feature creation;
  compile shaders and upload weights at initialization. Use retirement-aware
  frame slots for immutable per-recording data.
- Read the existing weight/LUT/offset formats where possible. Add versioned
  backend manifests for DXIL, required capabilities and layout compatibility.
  Reject corruption and mismatched model/shader revisions deterministically.
- Add a Windows build/package path for the engine and its shaders. Keep the
  standalone replay executable as a separate diagnostic program.

Gate: two features can share a model, record multiple queued frames, and release
independently without per-frame allocations, CPU waits or premature reuse.
The runtime works with explicit completion notifications from milestone 1.

### 4. Complete preset K's network in shared source

Port in stages, retaining the existing captured tensor layout and arithmetic
boundaries so differences can be isolated. Each stage is written once in shared
HLSL and built for both backends; there is no D3D12-only copy.

1. Generalize the plain block into a shared transformer core, including
   position-only attention.
2. Add encoder embedding and patch merging (`enc0` through `enc4`).
3. Add decoder patch expansion, skip connections and output head (`dec4`
   through `dec0`).
4. Chain all eleven layers using the existing activation/skip-buffer reuse plan
   where valid for D3D12.

Add layer replays with actual extracted K weights and recorded inputs. Compare
individual outputs first, then a chained network; use multiple captures, shapes,
window shifts and resets. Preserve the current fast/accuracy distinctions.
Synthetic 60 dB thresholds are smoke tests, not a release-quality criterion;
set capture-specific tolerances from the existing validated implementations.

For each layer, compare the shared SPIR-V build against the current GLSL layer
on RADV for correctness and GPU time. Where it matches, make it selectable on
Linux; where it is slower, record why and keep the GLSL default for that layer.

Gate: all eleven layers and the chained network pass captured reference checks
on D3D12, produce finite output and have measured per-layer GPU costs on both
backends. Choose WMMA variants from measurements, with an explicit baseline
override for diagnosis.

### 5. Complete full-frame inference and texture handling

- Port K's input processing, exposure, reconstruction and temporal feedback
  stages to the shared source. Reuse frame mathematics rather than deriving a
  second implementation.
- Support jitter, motion scale/convention, render- and display-resolution motion,
  depth direction, reset, dynamic resolution, HDR/LDR and subrectangles to the
  extent advertised by the backend's capabilities.
- Handle common color/depth/motion/exposure/output formats. Start with GPU
  conversion/copy passes where necessary; optimize direct views later. Do not
  require arbitrary game textures to be shareable with another API.
- Use legal resource views and explicit resource-state requirements/restoration.
  Document the command-state contract with the host. Keep history on the GPU.
- Enforce temporal execution ordering; discarded recordings must not corrupt
  committed history or expose an unfinished result.

Gate: full multi-frame replay passes moving scenes, exposure changes, cuts,
resets, format/subrect cases and dynamic-resolution tests. Pixels outside the
requested output rectangle remain unchanged. Record complete inference without
readback, CPU synchronization or a second graphics API.

### 6. Integrate the native engine into the shipping frontend

- Introduce an early backend router and a native feature state separate from
  CUDA initialization and vkd3d interop. Extract reusable parameter/configuration
  code from the shim in small changes; do not fork the whole shim for Windows.
- Implement coherent capability discovery, create/evaluate/release and shutdown
  behavior for the supported K configuration, including queries before init.
- Ensure OptiScaler selects the d4r shim explicitly and can use its native K
  backend on a real AMD adapter. Resolve any host identity/eligibility checks
  without allowing them to leak into the compute implementation.
- Report unsupported presets, flags and hardware truthfully. Expose a clear
  failure reason and let the host select a known working alternative.
- Validate feature recreation, multiple in-flight frames, multiple features,
  queued release, model sharing and device-loss cleanup in the D3D12 harness.

Gate: K works through the real NGX/OptiScaler call path without CUDA/ZLUDA or
patched vkd3d. Debug validation and lifetime tests pass. Linux's existing routes
still pass their relevant tests and replay checks.

### 7. Game validation and Windows preview release

- Start with one repeatable D3D12 game and a fixed K configuration, then cover
  Unreal and a different rendering engine. Choose games with reproducible
  scenes, motion, HDR/exposure changes and resolution/preset switches.
- Measure complete GPU inference, CPU recording time, VRAM, cold/warm startup,
  frame-time distributions and sustained gameplay. Preserve captures and logs.
- Compare implementations on the same OS/card/inputs before drawing backend
  conclusions, including the Windows Vulkan engine from milestone 1. Check
  supported output resolutions and quality modes rather than extrapolating
  from the 16-workgroup plain-layer benchmark.
- Package the shim, engine and DXIL with a Windows setup/check tool and clear
  model provisioning. Import prebuilt model packages as an initial developer
  workflow; design reproducible Windows provisioning before a general release.
  Follow the repository's existing model acquisition/distribution policy.
- Check model/shader hashes, dependencies, adapter requirements and configuration
  before launch. Keep experimental tuning overrideable. Do not require the
  Linux launcher, Bash, ROCm or modifications to system DLLs.

Gate: a documented Windows preview supports K on the tested hardware/game set,
survives extended runs and setting changes, and has reproducible performance
and image-quality results. Explicitly list the unsupported features.

### 8. Retire duplicated GLSL, then expand coverage

For each K stage whose shared build matched or beat the GLSL version on RADV,
make the shared build the Linux default and remove the GLSL. Stages that still
lag keep their GLSL with a documented reason and a tracked follow-up.

Then prioritize RDNA3 capability/performance validation and M/L support, using
their existing model formats and reference cases. New stages for these presets
are written in the shared source from the start. Plan RDNA2, E/F, Ray
Reconstruction and other frontends separately rather than making them implicit
requirements for the first Windows release.

## Performance and correctness policy

- Keep a baseline and independently selectable synchronization/scheduling
  variants. No default change based solely on one Windows microbenchmark.
  Variant defaults are per backend and GPU architecture.
- Collect CPU/GPU costs, spills/scratch traffic, LDS and occupancy at realistic
  shapes. Record compiler/driver versions, fixtures and shader hashes.
- Use rotating trials, warmup and GPU timestamps. Keep validation out of timing
  runs and avoid simultaneous benchmark processes.
- Ordered dispatch latency and throughput are separate tests. A throughput test
  needs independent output regions or another valid dependency scheme; removing
  barriers while overwriting the same output is not a valid substitute.
- Require captured temporal/image-quality checks before promotion. Agreement
  with the existing engine does not itself establish parity with RTX hardware.
- Treat material Linux regressions as a reason to retain the existing Linux
  backend, rather than imposing one execution path on both systems.
- Every change to shared shaders, generated contracts or the frame plan runs
  the captured replay checks on both backends and records per-backend timings.
  A change that improves one backend and regresses the other must say so in
  its description and adjust that backend's variant defaults, not fork source.

## Suggested implementation order and sizing

Start milestone 1 next. If its interop decision selects the interop preview,
integrate and validate that first, then continue from milestone 2. Otherwise,
once its device/lifetime/model-loading contract is settled, establish the shared contracts and frame plan, build the record-only
runtime, finish the K network in shared source, complete temporal inference,
integrate the frontend and validate games, in that order.

For one experienced graphics developer, budget roughly 1–2 weeks for the
integration spike and Windows Vulkan baseline, 1–2 for shared contracts, the
frame plan and the dual shader build, 1–2 for the runtime/package, 3–5 for the
remaining network including RADV parity work, 2–4 for full-frame handling, and
2–4 for frontend/game hardening. This is an initial planning range of about
10–19 weeks for a constrained K preview, with some work overlapping. The extra
time up front replaces a later consolidation milestone and the manual porting
of Linux improvements made in the meantime. Unsupported hardware, shader
compiler behavior (including DXC's SPIR-V output on RADV), model provisioning
or host lifetime integration can extend it. Re-estimate after milestone 1; M/L
and broad compatibility are additional work.

No whole-project rewrite or Linux backend switch is a prerequisite for this
plan. Each milestone should leave a runnable harness and a reviewable result.
