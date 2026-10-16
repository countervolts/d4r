# Native RR D3D12/Vulkan bridge gate

## Runtime extraction in progress (2026-10-16)

The packet layout now has a geometry-only constructor with no fixture I/O.
The model recorder also accepts an owned `RFrameSettings` value independently
of the replay frame list. Existing fixture entry points delegate to the same
packet-layout and recording code. This prepares the production integration
described below.

`rr_native_device.h` now owns the shared LUID-matched Vulkan setup, compute queue
selection, and inference feature checks. `rr_native_packet.h` owns dedicated
D3D12 resource memory import and shared-fence timeline import. The existing
probe uses both owners instead of its inline setup. Imports retain Vulkan
references while the caller keeps ownership of exported Win32 handles. These
owners require retired submissions before destruction; the probe preserves its
existing quarantine behavior after an incomplete submission.

`rr_ngx_snapshot.h` adds typed RR creation/frame snapshots, independently of the
SR parameter whitelists. It copies both 16-float matrices, creation presets and
scalar baselines, per-frame overrides, render extents and required-guide subrect
origins. COM references retain required and optional guide/output resources.
D/E packed roughness uses normals.w only when a separate roughness texture is
absent, matching the Linux reference. Missing matrices and invalid/nonfinite
settings are rejected. Feature-13 evaluation uses this snapshot.

The CPU-only `d4r_rr_snapshot_probe` passes mutation, parameter-destruction,
creation-fallback, per-frame-override, and invalid-input checks with the real NGX
parameter implementation. It does not exercise GPU execution, COM resource
retirement, command-list ordering, Create/Evaluate/Release, or game integration.
This separate CPU test did not retry the policy-blocked GPU regression wrapper.

`rr_native_backend.cpp` now compiles into the Windows shim when the native engine
is available. It borrows the supplied D3D12 device/queue and actual input/output
textures, packs required guides on the GPU, records the existing model engine,
and copies output back on the GPU. It restores legacy/enhanced resource access,
supports bounded subrect copies in canonical formats, verifies device ownership
and preset selection, and retains all snapshot resources until both APIs finish.
Completion waits are bounded; a failed in-flight backend quarantines its full
ownership graph instead of freeing live GPU resources. Recording errors poison
history/upload state and require feature recreation.

The backend rejects unresolved/split states, unsupported formats/encodings,
optional guides/exposure textures, non-unit exposure scale, unsupported creation
flags, aliased input/output, and dynamic render extents. These paths still need
implementation for broad game support. Global Volk dispatch is serialized and
reloaded for each backend operation.

The shim now routes feature-13 Create/Evaluate/Release to the native backend
when `D4R_RR_BACKEND=native` and absolute `D4R_RR_MODEL_DIR` are configured.
Required-guide access is captured at Evaluate, inherited access is resolved at
submission, and the callback retains the feature and owned snapshot. Capability
and feature-requirement queries load-check the actual model, Vulkan adapter
features, and packet/fence import; they do not claim image-quality validation.
Default configurations continue to report RR unavailable. SR routing remains
separate. Native RR Init now defers the independent CUDA SR runtime, as described
in the independent-initialization evidence below.

`d4r_rr_ngx_probe` and `tools/windows/test_rr_ngx.py` add a bounded, synthetic
NGX lifecycle gate using fixture textures and same-engine reference images.
The gate mutates matrices/scalars after Evaluate and releases the last feature,
parameters, and fixture texture ownership before submission. It does not touch
a game or retry the policy-blocked interop executable/wrapper. Initial attempts
stopped before RR creation because of an invalid legacy application ID and a
missing `HIP_PATH`; those failed artifacts remain separate.

The actual native RX 7700 XT feature-13 gate now passes all six 64x64 to
128x128 camera/jitter/exposure/reset frames, byte-identical to standalone replay.
Every output is finite and nonuniform. The test includes parameter mutation
after Evaluate and feature/parameter/fixture-ownership release before the final
submission. Prefix guide uploads and suffix output readback exercise logical
command-list ordering through the real shim. Independent output-byte checks
also pass in the wrapper. Evidence: `build/rr-ngx-camera-reset-v3/summary.json`,
stdout/stderr and frame files. This is synthetic lifecycle/bridge consistency
against the same RR engine, not a game or NVIDIA quality reference. Additional
contract and inherited-state checks are recorded in the next section.

## Additional NGX contract and inherited-state validation (2026-10-16)

Unsupported frame contracts now fail synchronously during Evaluate, before a
logical command-list boundary is recorded. Submission revalidates the textures
and resolved states. This prevents unsupported guides/encodings/extents from
being reported as a successful Evaluate that fails only during submission.

The real-shim contract run verifies configured RR availability/init success,
preserved SR availability, feature-13 requirements on the actual adapter, WARP
rejection, unknown feature rejection, and eight creation/evaluation failures.
Those failures cover an unknown feature ID, zero creation width, missing matrix,
unsupported optional hit distance, dynamic render width, unsupported encoding,
out-of-bounds subrect, and an invalid/aliased output. All six valid camera/reset
frames still match standalone bytes after these rejected calls.
Evidence: `build/rr-ngx-contracts-camera-reset-v1/summary.json` and logs.

The harness can also omit all caller state overrides and upload guides in a
separate preceding list within the same ExecuteCommandLists batch. The NGX
list must resolve those inherited states at its delayed callback. That mode
passes six camera/reset frames at 64x64 to 128x128 and two full-size frames at
853x480 to 2560x1440, including the existing mutation/release-before-submit
checks. Every output remains finite/nonuniform and byte-identical to standalone.
Evidence: `build/rr-ngx-inherited-camera-reset-v1/summary.json` and
`build/rr-ngx-inherited-1440-v1/summary.json`. Enable the new gates with
`tools/windows/test_rr_ngx.py --contracts` or `--inherited`, alongside the
required explicit artifact/input/output arguments shown by `--help`.

These are synthetic native feature-13 contract/ordering results. The following
section adds enhanced access, legacy split rejection, and unavailable-backend
checks. Stale-but-loadable models, concurrency, optional guides, format conversion,
FSR RR, game integration, quality and performance parity remain open. No
additional SR presets or game content were tested.

## Enhanced access, split rejection, and unavailable configurations (2026-10-16)

`--enhanced --inherited` now validates feature-13 transfers when a preceding
list establishes enhanced shader-resource/UAV access and the NGX list has no
caller state overrides. Fixture transitions cross COMMON when switching
between legacy and enhanced barriers, as required by the
[Microsoft enhanced-barrier specification](https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html).
The backend restores enhanced access before suffix readback/fixture reuse. All
six camera/reset frames match standalone bytes and remain finite/nonuniform.
Evidence: `build/rr-ngx-enhanced-inherited-camera-reset-v1/summary.json`.

`--split` records a legacy BEGIN_ONLY transition, confirms Evaluate rejects
the unfinished split, completes the matching END_ONLY transition, and then
runs the normal feature-13 evaluation. The subsequent six camera/reset frames
still match standalone exactly. Evidence:
`build/rr-ngx-split-camera-reset-v1/summary.json`. This tests legacy split
rejection, not enhanced split rejection. No debug-layer validation is claimed.

`--unavailable disabled`, `--unavailable missing-model`, and
`--unavailable fsr` separately verify requirements stay unsupported, RR capability
availability stays zero with a failure init result, and feature-13 creation
fails with a null output handle. SR availability remains one in every case.
These runs dispatch no RR frames. Evidence is under
`build/rr-ngx-unavailable-{disabled,missing-model,fsr}-v1/summary.json`.
The FSR case proves honest rejection of the unavailable backend; it is not
FSR execution or provider support.

Native RR optional-guide/format conversion, dynamic geometry,
concurrent lifecycle, game integration, quality and performance
parity remain incomplete. The original full parity goal remains open.

## Independent native RR initialization (2026-10-16)

With an explicitly selected native RR backend, Init retains the borrowed D3D12
device and an owned initialization contract without loading HIP, CUDA, NVAPI,
NVIDIA SR/core DLLs or the SR pixel converter. Required SR capability/feature
requests initialize the separate SR runtime on demand. A missing SR dependency
reports SR unavailable without hiding a ready native RR backend. Default
configurations retain eager SR initialization. Shutdown releases both paths.

`RuntimeInitialization` owns the application data directory, project identity,
path strings, and path-list descriptor used by deferred SR startup. The harness
mutates those caller strings/descriptors after Init; subsequent SR capability
initialization still logs the original project/version and path. Both RR/SR
capabilities are available, negative contract checks pass, and all six inherited
camera/reset frames remain byte-identical to standalone.
Evidence: `build/rr-ngx-deferred-sr-init-v1/summary.json` and logs.

The `--contracts --without-sr --enhanced --inherited` gate removes all SR
dependency path configuration and does not add runtime directories to PATH.
It verifies RR requirements/capability readiness, SR unavailability, negative
contracts, and six exact native RR camera/reset frames. No SR runtime initializes.
Evidence: `build/rr-ngx-independent-camera-reset-v1/summary.json` and logs.
This proves native RR execution independent of configured SR dependencies;
it does not change the package's model provisioning or broader game support.

The default eager SR path still passes capability/readiness validation with RR
disabled: `build/rr-ngx-eager-sr-regression-v1/summary.json`. These checks add no
new SR-preset or game execution claim.

## Native sampled guide formats (2026-10-16)

The production packet now reserves each guide's maximum supported texel size,
so format changes do not recreate shared memory, the RR model, or temporal
history. The Vulkan replay caches sampled-image views by plane/format and keeps
old views alive for the engine's cached descriptors. New image variants receive
their own first-use layout transition. Production transfers still stay on GPU.
The larger reserved packet is a memory tradeoff, not a performance improvement.

The backend maps float16/float32 RGBA and UNORM RGBA8/RGB10A2 color/guide inputs,
float16/float32 two-channel motion, float16/float32 scalar depth, and scalar
float16/float32/UNORM8 roughness where applicable. Vulkan sampled/transfer support
is checked when creating a view. Unsupported encodings, typeless/depth-stencil
formats and optional guides remain rejected; output remains RGBA16F. Mapping a
format does not establish game-specific guide encoding or quality correctness.

Synthetic evidence tests RGBA32F color/normals, RG32F motion and R16F roughness,
with values exactly equivalent to the canonical fixtures. All six camera/reset
frames match standalone output bytes, including enhanced/inherited access,
negative contracts and independent SR initialization. A second run alternates
canonical and float guide formats each frame without feature recreation; all
six outputs remain exact. CPU conversions initialize fixture inputs only.
Evidence: `build/rr-ngx-float-guides-camera-reset-v1/summary.json` and
`build/rr-ngx-switch-guides-camera-reset-v1/summary.json`.
The unchanged canonical baseline also passes six frames:
`build/rr-ngx-format-layout-baseline-v1/summary.json`.

Other mapped formats have not received format-specific execution coverage.
Dynamic render extents remain rejected, as in the cited Linux reference's RR
adapter. Optional guides, output conversion, broader game formats/encodings,
concurrency, game integration, FSR RR and performance parity remain unfinished.

## Corrected RR creation-flag contract (2026-10-16)

Inspection found that the initial Windows gate accepted flagless synthetic RR
and rejected the actual HDR/render-motion flags. The production backend now
matches the Linux RR adapter: HDR and render-resolution, unjittered motion
vectors are required. Inverted depth remains supported. AutoExposure is accepted
because the D/E engine already performs its exposure reduction. Jittered motion,
high-resolution motion, non-HDR input, alpha upscaling and other unsupported
flags fail before Vulkan/model allocation. Public flag values are defined by
[NVIDIA's NGX header](https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx_defs.h).

Readiness checks and the fixture now use the required HDR/MVLowRes bits. Five
additional negative creation cases cover absent flags, high-resolution motion,
non-HDR input, jittered motion and alpha upscaling, bringing the contract gate
to thirteen rejections. Both the required-flags run and the AutoExposure run
pass all six camera/reset outputs exactly against standalone. The AutoExposure
run also alternates float guide formats, uses enhanced/inherited states, and
has no configured SR dependencies. Evidence:
`build/rr-ngx-hdr-render-motion-v1/summary.json` and
`build/rr-ngx-hdr-auto-exposure-v1/summary.json`.

Earlier flagless results remain synthetic bridge/lifetime evidence; they did
not prove a valid game-facing HDR/motion contract. These new runs correct that
gap. They do not establish game quality, high-resolution/jittered motion support,
FSR execution or performance parity.

## Optional native RR guides (2026-10-16)

The production packet and recorder now carry `DLSSD.SpecularHitDistance`,
`DLSS.DisocclusionMask`, `DLSSD.ScreenSpaceSubsurfaceScatteringGuide`, and
`DLSSD.ColorBeforeParticles` into the existing RR engine. Bound optional guides
receive the same device/format/subrect/access validation and GPU texture copies
as required guides. Snapshots retain optional COM resources and subrect origins;
submission resolves their inherited/enhanced states. Binding can change between
frames without recreating the model or resetting temporal history. Exposure
textures remain unsupported. Previous optional-guide rejection notes describe
the earlier backend and are superseded by this section.

Two six-frame sequences alternate all four optional guides between present and
absent, using R32F hit distance, R16F disocclusion/subsurface and RGBA16F
pre-particle color. Every output is finite/nonuniform and matches standalone
engine replay byte-for-byte. The second sequence keeps all optional guides
bound on the final frame, when the feature, parameters and fixture texture
ownership are released before submission. Both sequences also alternate
required guide formats, use enhanced/inherited access, set valid HDR/motion/
AutoExposure flags, run thirteen negative contracts and omit SR dependencies.
Evidence: `build/rr-ngx-optional-toggle-v2/summary.json` and
`build/rr-ngx-optional-final-bound-v1/summary.json`. Reports verify actual per-frame
optional binding masks as well as independent output bytes.

The first toggle attempt (`build/rr-ngx-optional-toggle-v1/`) matched five
frames, then encountered device removal. The harness had recorded uploads and
transitions for unbound optional textures and dropped their ownership before
the final submission. Fixture commands now skip unbound guides. The backend
quarantined incomplete work as designed. No driver, TDR, security or GPU setting
was changed. Fixed toggle and final-bound lifetime gates pass.

These are synthetic same-engine API/transfer/lifetime comparisons. They do not
validate NVIDIA reference quality, additional optional format/encoding cases,
nonzero optional GPU subrects, dynamic geometry, concurrency, game integration,
FSR execution, or performance parity.

## Native depth-plane formats and stencil preservation (2026-10-16)

The backend now accepts R32 typeless, D32 float, and typed/typeless D32S8 depth
resources in addition to ordinary float depth textures. It queries the actual
D3D12 subresource-0 footprint, validates the depth row size, and imports only
those float depth bytes into Vulkan. The stencil plane is not part of the RR
packet or resource transition. Depth-stencil copies require the complete depth
rectangle with zero origin; cropped depth-stencil input fails Evaluate before
recording a boundary. The whole-rectangle rule follows
[Microsoft's CopyTextureRegion contract](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-copytextureregion).
Earlier depth-stencil rejection notes are superseded for these tested formats.

Four six-frame real-shim runs cover D32 float, R32 typeless, D32S8 typed and
D32S8 typeless. All outputs remain finite/nonuniform and byte-identical to
standalone replay. D32S8 fixtures upload changing stencil sentinels to plane 1
and independently read them back after RR; every stencil byte is preserved.
The runs retain optional-guide toggles, inherited access and final-bound
Release-before-submit coverage; D32S8 also exercises enhanced access.
Evidence: `build/rr-ngx-depth-{d32,r32-typeless,d32s8,d32s8-typeless}-v1/summary.json`.

A further normalized, inverted hardware-depth sequence passes six exact frames
through typeless D32S8, with optional guides, enhanced/inherited access, changing
required formats, AutoExposure and no configured SR dependencies. Stencil
sentinels remain exact. Final report verifies all six hardware-depth/inversion
settings and records creation flags 75. Evidence:
`build/rr-ngx-inverted-hardware-depth-v2/summary.json` and logs. The final harness
also checks that cropped depth-stencil evaluation is rejected.

These are same-engine synthetic comparisons, not a game or NVIDIA quality
reference. D24 conversion, MSAA/array depth, cropped depth-stencil processing,
output conversion, broader encodings, dynamic geometry/concurrency/game
integration, FSR execution and performance parity remain incomplete.

## Player configuration without launcher overrides (2026-10-16)

`d4r.ini` now accepts `[RayReconstruction] Backend = off|native|fsr` and
`ModelDir = auto|PATH`. Off is the default and clears stale RR selection in
player mode. Native resolves relative model paths beside d4r.ini; auto selects
`engine/rr-d`. FSR selection remains explicit but unavailable, with a diagnostic
note and rejected RR requests. The existing readiness check still loads the
configured model before advertising native RR. The source package README/INI
describe these experimental settings. Existing delivered ZIP artifacts have
not been rebuilt or changed; install.bat does not generate RR models yet.

Pre-Init feature-13 requirements now establish player configuration before
checking readiness. An isolated staged player-folder test removes all launcher
path/RR overrides, selects the backend/model through the actual INI, and passes
six exact native RR hardware-depth frames. It also verifies capability contracts,
optional-guide toggling, enhanced/inherited state, inverted D32S8 and stencil
preservation, with final Release-before-submit. Evidence:
`build/rr-ngx-player-config-v1/summary.json` and the staged player's log/INI.
No game folder was read or modified.

The player parser/environment/pin probe passes all checks, including relative RR
paths, invalid backend fallback, FSR unavailability and stale-environment cleanup.
Evidence fixtures: `build/player-config-rr-v2/`. The PowerShell checker recognizes
the new keys and passes static parser syntax validation; its fixture suite was
not rerun under the current script execution policy. No policy bypass was used.

## Local package v4 architecture-loader gate (2026-10-16)

A new development ZIP refreshes the previously verified v3 runtime with the
current shim and INI/checker/installer documentation. The base and all output
manifest hashes verify; local modified provenance remains explicit. No NVIDIA
binaries, RR models or private optimized kernels are bundled. Existing ZIPs
remain unchanged. See `windows-local-package.md` for its hash and size.

The actual packaged architecture loader and refreshed target shim pass six
exact native RR hardware-depth frames with INI-only selection, optional guides,
enhanced/inherited states, stencil preservation and final Release-before-submit.
Pinned SR/core dependencies were staged separately for the test, and the local
RR model remains outside the ZIP. Evidence:
`build/rr-ngx-local-v4-loader-v1/summary.json`. No game, installer execution,
system change, commit or publishing occurred. Automatic RR model provisioning,
game integration, FSR execution and performance parity remain open.

The native interop probe rebuilt successfully after these changes. The requested
camera/reset regression wrapper was blocked by the current PowerShell execution
policy before running. No policy change, bypass, alternate launcher, or probe
retry followed. Runtime regression validation of this refactor remains pending;
the earlier results below describe the preceding build.

The RX 7700 XT now passes cross-API memory/fence transfers and actual RR model-D
inference through native D3D12 textures and the existing Vulkan engine. The NGX
feature-13 game adapter remains unfinished. The probe loads locally extracted
model files, not NVIDIA runtime DLLs, and does not touch a game.

## What is verified

`d4r_rr_interop_probe` matches the D3D12 adapter and Vulkan device by LUID. It
creates a committed shared D3D12 buffer, imports it into a dedicated Vulkan
allocation, and imports a shared D3D12 fence as a Vulkan timeline semaphore.
Each iteration queues this sequence:

1. D3D12 uploads a new deterministic pattern and signals the shared fence.
2. Vulkan waits on that GPU fence, reads the buffer, writes new output, and
   signals the same fence at a greater value.
3. D3D12 queues its own wait for Vulkan, copies the output for validation and
   signals completion. Only then does the CPU validate both directions.

The internal API handoff uses GPU synchronization and shared memory. CPU uploads
initialize synthetic inputs; CPU readbacks validate results. No tensor passes
through CPU memory between the two APIs.

The image cases also copy the input buffer into a sampled/storage-capable
RGBA16F Vulkan image, preserve its exact input pixels through image readback,
clear it to known half-float values, and copy that output back to the shared
buffer for D3D12 consumption. The padded case verifies untouched row-padding
sentinels separately from image pixels.

| Case | Shared bytes | Iterations |
|---|---:|---:|
| Buffer-only smoke | 1,048,576 | 8 |
| 1712x960 RGBA16F, 13,824-byte row pitch | 13,271,040 | 3 |
| 2560x1440 RGBA16F | 29,491,200 | 3 |

All 14 iterations pass exact comparisons in both directions on native Windows.
The last two cases exercise Vulkan texture transfers. The probe uses explicit
external queue ownership transitions and transfer/host memory barriers.
The imported resource requires dedicated allocation on this driver.

The timeline import follows the [Vulkan D3D12 fence guidance](https://docs.vulkan.org/refpages/latest/refpages/source/VkD3D12FenceSubmitInfoKHR.html).

## Reproduce

```powershell
cmake --build build/windows-rdna4-therock-gfx1101 --target d4r_rr_interop_probe
./scripts/windows/test-rr-interop.ps1 `
  -Probe build/windows-rdna4-therock-gfx1101/d4r_rr_interop_probe.exe `
  -OutputDirectory build/my-rr-interop-check
```

Use a new output directory under ignored `build/`. Each case has a 30-second
process limit. GPU completion waits have five-second limits. No driver, TDR,
security, or tuning setting is changed.

Evidence: `build/rr-interop-canonical-images-v2/summary.json` and per-case logs.
Earlier buffer-only evidence remains under `build/rr-interop-realistic/`.

## Bridged model inference

`rr_bridge_replay.h` packs all seven canonical inputs into one shared GPU packet.
The harness uploads synthetic data into real D3D12 guide textures, copies them to
the shared buffer, and queues its producer fence. Vulkan imports that memory,
copies each guide into its sampled image and runs `RModel`/`REngine` with the
fixture's matrices, jitter, exposure and reset settings. It copies the RGBA16F
result back to the packet and signals the shared timeline. D3D12 waits on the
GPU, copies the result into its UAV-capable output texture, then reads that
texture back for QA. There is no CPU tensor transfer between APIs.

Actual model-D results, all byte-identical to standalone Vulkan replay:

| Fixture | Render/output | Frames |
|---|---|---:|
| Initial history/reset | 64x64 / 128x128 | 3 |
| Camera translation, jitter, exposure, reset at frames 0 and 3 | 64x64 / 128x128 | 6 |
| Full output size | 853x480 / 2560x1440 | 2 |

All 11 textured frames are finite and nonuniform. The bridge wrapper also
compares output SHA256 independently after the executable's byte comparison.
Reference images come from the same existing RR engine; this proves bridge
consistency, not correctness against NVIDIA or image quality in a game.

```powershell
./scripts/windows/test-rr-bridge-model.ps1 `
  -Probe build/windows-rdna4-therock-gfx1101/d4r_rr_interop_probe.exe `
  -ModelDirectory build/native-rr-d-windows-gfx1101-final `
  -InputDirectory build/rr-1440-inputs `
  -ReferenceDirectory build/rr-1440-reference `
  -OutputDirectory build/my-rr-model-bridge
```

The process limit is 180 seconds, including pipeline creation. Evidence is under
`build/rr-bridge-d-texture-inference/`, `build/rr-bridge-camera-reset/`, and
`build/rr-bridge-1440/`. The original 14-iteration transfer gate also passes after
the model-mode changes (`build/rr-interop-after-model-mode/`).

The two standalone 1440p reference frames took 29.0 and 38.6 ms of GPU time. These
are two synthetic samples, not a steady-state measurement or Linux parity proof.
No inference timing is attributed to the bridge itself yet.

The harness accepts canonical D/E gfx11 models and the seven required guides.
It rejects optional guide files, duplicate frame IDs and nonfinite settings;
support for optional hit distances, masks, particles and game format conversion
still needs implementation. Feature-13 creation/evaluation/lifetime integration,
game-facing resource state contracts, FSR RR and performance parity remain open.
