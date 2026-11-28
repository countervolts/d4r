# DLSS-G Lite in Minecraft: feasibility (2026-11-28) - superseded by a working engine, see ../STATUS.md

The sections "What blocks" and "Possible routes" below were written before the engine existed; route A was implemented (docs/architecture.md).

Scope per the latest request: only the **Lite** model, as fast as it runs in normal d4r. Nothing here was committed.

## What exists in d4r (verified by reading the tree and running it)

- There is **no native-Vulkan DLSS-G path**. `nvngx_dlssg.dll` is a closed Windows DLL. It reaches the GPU only through
  D3D12 (vkd3d-proton) and the NVAPI/cubin launch API. All CUDA entry points in it are stubs (`0xBAD00001`).
- d4r's path: `game -> Streamline/OptiScaler -> d4r NGX host (fg/dlssg/host) -> nvngx_dlssg.dll -> vkd3d-proton
  (patch 0006) -> nvxrt Vulkan layer -> RADV compute`. nvxrt maps each cubin kernel (by PTX hash) to SPIR-V pre-translated
  into `~/.cache/nvxrt-gfx11` (174 kernels, 29 MB), plus hand-written T1 kernels and persistent launch groups.
- So the **kernels are already Vulkan compute shaders and are fast**; what is not portable is the *orchestration*: the DLL
  decides the launch sequence, grid sizes, scratch-buffer layout and the scalar fields of every 40-240 byte parameter block.

## Measured today (RX 7700 XT, Mesa RADV, 2560x1440, DLSS-G 310.9.16 Lite, harness, fast mode)

`fg/dlssg/harness/run.sh` with `--mfc 3` (4x), `HOST_INI_EXTRA=Lite=1`, `NVXRT_ARCH=gfx11 NVXRT_TEX=hw NVXRT_ACC=fast`:

- GPU time per real frame (real evaluate + 3 generated): **3.27 ms** (median of 4), evaluates 2.40 / 0.44 / 0.44 ms.
- Launch trace (`lite-launch-trace.txt`, NVXRT_TRACE=1): the real-frame evaluate is ~37 launches (Downsample, InputMvecProcessing,
  Prev2Curr push/pull chain, Scatter3d, DL1Net input -> ~25 conv/pool/upscale/elementwise layers -> DL1Net output); each generated
  frame is 3 launches (EstimateIntermMvecsScatter, WarpBlendingWeights, BlendCandidatesFGLite). Counts over the 6 evaluates of the run:
  85 `k_conv_fp16_nhwc`, 25 each pooling/upscale/elementwise, 15 each of the three generated-frame kernels.
- Inputs it needs (from the captured NGX parameter dump of a real game): depth, motion vectors (with `MvecScale`), HUD-less colour,
  backbuffer colour (and optional UI), camera position/up/right/forward/near/far/FOV/aspect, jitter, exposure, `MultiFrameCount/Index`,
  reset flag, `DepthInverted`. Output: N-1 extra frames at full output resolution (RGBA8 for LDR).
- Minecraft can supply all of these (reversed-Z depth, camera matrices already captured, HUD-less colour before the GUI pass);
  motion vectors are camera-only unless entity MVs are added.

## What blocks a native Minecraft integration

1. **The DLL cannot be loaded by Minecraft's Linux JVM.** It is a PE file that needs Wine + vkd3d-proton + NVAPI emulation.
   Minecraft (native Linux, LWJGL Vulkan) has no D3D12 device to hand it.
2. **Replaying the kernels natively requires re-deriving the DLL's orchestration.** That means reimplementing ~40 launches per real
   frame and 3 per generated frame with correct parameter blocks, scratch layout, and per-frame scalars derived from camera/jitter/MV
   data (e.g. the 240-byte BlendCandidatesFGLite block). This is a reverse-engineering project, not a port: launch captures
   (`NVXRT_TRACE`, `NVXRT_CAPTURE`) give the structure, but every scalar field must be identified and verified bit-exactly against
   the DLL on several scenes. Not done, and I will not claim it works.
3. **Presentation.** Even with correct frames, extra frames must go through Minecraft's swapchain (`VulkanGpuSurface`, 2 submits in
   flight, timeline semaphore). Pacing/present of N frames per game frame is a separate design problem (not started).

## Possible routes (ranked)

A. **Native replay engine ("d4r-fg-lite")**: implement the Lite pipeline as a Vulkan-only library inside d4r (reusing the SPIR-V/T1
   kernels, bindless slots from nvxrt), validated bit-exact against the harness (the harness already dumps real/generated frames as PPM,
   and the GoT capture in `~/.cache/dlssg-capture-got` is a ready replay corpus). Gives the same speed as d4r (kernels identical).
   Cost: large RE effort; the parameter-block decode is the critical path. Recommended start: freeze the block layouts for the 3
   generated-frame kernels (simplest, 0.44 ms each) and prove them against harness output; then Downsample/MV/Prev2Curr; the DL1Net
   chain last (its blocks are mostly pointers and constants).
B. **Wine sidecar**: run the real DLL in a Proton process and exchange frames with Minecraft over dma-buf/opaque-fd. Needs Wine to
   import Linux fds across processes plus cross-device sync; adds copies, so it would not match d4r's speed. Not recommended.
C. **Run Minecraft itself under Proton with the DLSS-G path**: d4r's DLSS-G host is D3D12-only; Minecraft is Vulkan. No.

## Status

Milestone A (this report) done. B-E (device features, bridge init, correct frame, presentation) **not started**: they depend on route A's
engine, which does not exist yet. The existing SR code is untouched in this pass.

## Parameter-block findings (tools/param_diff.py, 6 evaluates, harness, Lite 4x)

Raw blocks (`NVXRT_TRACE=1 NVXRT_TRACE_RAW=1`) show the per-frame variable bytes are mostly bindless image handles (small ints)
and device pointers; real scalars are few and often constant per resolution (e.g. generated-frame kernels carry the interpolation
fraction 0.25, 1.0, 40.0, 1280x720, a 0.0016 epsilon, a 0.0007 scale). Generated-frame kernels vary only in handles/pointers:
EstimateIntermMvecsScatter [16,24,34,35], WarpBlendingWeights [24,48], BlendCandidatesFGLite [0,8,16,40,128,146,147,176,208].
The real-frame stages (Downsample, InputMvecProcessing, Prev2Curr*, Scatter3d, DL1Net in/out) vary only in handles too.
(Conv/pool/upscale/elementwise blocks differ per layer, not per frame.) Caveat: the synthetic harness has a fixed camera, so
camera-derived scalars are constant here; they must be re-checked with the GoT capture or a moving-camera scene before the
engine claims to be correct. This makes route A look much more tractable than first feared: the orchestration is a fixed,
resolution-dependent launch list. Next step: dump (NVXRT_CAPTURE) the resources of one evaluate, replay the 3 generated-frame
kernels natively, and compare to the harness PPMs.
