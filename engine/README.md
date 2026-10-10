# Native weight-based inference engine

This directory implements standalone preset K and M inference engines. NGX is used **once to obtain weights
and reference data**. K loads extracted f16 matrices and a reconstruction table; M loads FP8 matrices and
f16 vectors. Both run d4r's own Vulkan shaders. M can run its network on d4r's native HIP kernels instead;
neither engine translates PTX or launches NVIDIA kernels.

The reusable library is `libd4r_engine.so`; `d4r-k` and `d4r-m` provide full-frame replay and benchmarking.
The D3D12 game shim supports both as opt-in engine backends (see [In a game](#in-a-game-d3d12-shim)).
The CUDA/ZLUDA backend remains the default and fallback. Presets L/E and RTX/game-quality parity are not
implemented or established here. Exposure is supplied by the application or measured on the GPU.

## Architecture

```
extracted model ──► shared Model (weights + LUT + 13 compiled pipelines)
                                │
frame color/depth/motion ──► Engine input stage ──► 11 fused Swin layers ──► reconstruction ──► output
                                ▲                    │                        │
                                │              encoder skip buffers           │
                                └──── previous color / luma / token features ◄─┘
```

- `Model` owns weights in device-local memory and compiled pipelines. Multiple `Engine` instances can share it.
  Shader compilation and weight uploads happen at initialization.
- Each `Engine` owns two reusable activation buffers, five encoder skip buffers, and two sets of temporal history.
  The input stage writes directly into the first activation buffer; the head feeds reconstruction directly.
  Intermediate activations and history stay on the GPU.
- A frame records thirteen dispatches into a caller-owned command buffer, with dependencies between stages and
  before reusing buffers. It performs no allocations, queue submissions, CPU waits, or intermediate readbacks.
- Input images are borrowed Vulkan views. The caller provides the device, synchronization, and command buffer,
  so a future game adapter can schedule inference with its own rendering work.
- Per-frame parameters include exposure, jitter, motion conventions, reset, and each layer's window shifts.
  Window shifts can change without rebuilding shaders or buffers. Resolution changes require a new `Engine`.
- Frame parameters and window shifts are written by commands inside the frame's own command buffer, so a frame can
  be recorded while earlier ones are still queued on the GPU. Each input/output view and layout combination gets
  immutable descriptors, cached and reused; first use allocates descriptors, steady recording does not. Borrowed
  images and views must outlive the `Engine`. Keep the `Model` alive until all engines using it have been destroyed.
  Submit recorded stateful frames once, in order.

The network shader uses 16x16 cooperative matrices with f32 accumulation, four wave32 subgroups per 8x8 token
window, fused attention/MLP/patch operations, padded LDS rows, and a residual stream held in registers. The output
shader uses wave64 and computes reconstruction and temporal accumulation in one dispatch.

## Build and extract a model

Requirements: C++17, Vulkan headers/loader, Python with NumPy, and `glslangValidator`. The GPU/driver must support
Vulkan 1.3, `VK_KHR_cooperative_matrix` with 16x16x16 f16 operands/f32 accumulation, wave32/wave64 subgroup control,
f16 storage/arithmetic, shader int16, and extended storage-image formats. The executable checks these capabilities.

A model directory holds values read from NVIDIA's DLSS (`weights.bin`, `offsets.bin`, `lut.bin`, `hipnet.bin`,
`model.json`) next to d4r's compiled shaders. Build it from your own `nvngx_dlss.dll` and keep it on your machine:
those files are not d4r's to license, `.gitignore` excludes them, and `scripts/package_release.sh` never packages them.

```sh
bash engine/build.sh                         # build/native-engine/{libd4r_engine.so,d4r-k,d4r-m}
# With ROCm installed (D4R_ROCM_DIR, default /opt/rocm) d4r-k and d4r-m also link the native HIP network.

# CAP contains replay/ and dump/ for the K input, 11 layers, and output launches.
# All replay allocations must be present for weight extraction.
python3 engine/extract_k.py "$CAP/replay" build/model-k
python3 engine/prepare_k.py "$CAP" build/k-case --frames 6
python3 engine/compile_k.py build/model-k build/k-case/lut.bin build/k-model

build/native-engine/d4r-k build/k-model build/k-case/case.txt build/k-results 25
python3 engine/test_frame.py build/k-case build/k-results
```

Capture the standard K HDR, inverse-depth, render-resolution-motion input/output kernels:
`hiluma_engine_input_depthinv_mvlo_hdr_v2_rel` and `hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel`.
The bridge's `D4R_CUDA_REPLAY_DUMP_DIR`, `D4R_CUDA_REPLAY_DUMP_LIMIT`, and `D4R_CUDA_LAUNCH_DUMP_DIR` capture the
arguments, uploaded weight allocations, textures, and reference outputs. Take references with the native K layers
and ZLUDA patch 0008; otherwise dec5's native phase output can be zero. Large captures require substantial disk space.

The reconstruction table is extracted separately from the output launch's argument 280, at its allocation offset.
Model packaging validates tensor dimensions, alignment and complete spans, records file hashes in `model.json`,
and compiles all shaders. The runtime also checks the binary format and every used tensor's bounds before upload.
Weights/tables are proprietary inputs and are not committed in this repository.

`test_frame.py` compares all four outputs across the sequence, using the engine's own history after the first frame.
It fails for missing/wrong-sized results, nonfinite values, or PSNR below its threshold (50 dB by default).
`test_k.py --chain` remains available for isolated-layer diagnosis; it now selects the first launch consistently and
fails on dispatch errors rather than continuing with a broken chain.

## Ordinary frames, without a capture

Create a JSON description with raw linear HDR color, inverse depth, and render-resolution motion vectors:

```json
{
  "output": [1280, 720],
  "frames": [
    {"color": "color-0.bin", "motion": "motion-0.bin", "depth": "depth-0.bin", "render": [640, 360]},
    {"color": "color-1.bin", "motion": "motion-1.bin", "depth": "depth-1.bin", "render": [640, 360], "jitter": [0.25, -0.25]}
  ]
}
```

Color is interleaved RGBA16F, motion RG16F, and depth R32F, tightly packed. `bits: 32` selects f32 color/motion.
Paths are relative to the JSON. The first frame resets; subsequent frames feed back the engine's own history.
Optional exposure/motion/sharpness settings are described by `prepare_frames.py --help` and `FrameSettings`.

```sh
python3 engine/prepare_frames.py frames.json build/frames.case
build/native-engine/d4r-k build/k-model build/frames.case build/frame-output
# GPU timing with no output readback or output files:
build/native-engine/d4r-k build/k-model build/frames.case - 25
```

`makeKShape` uses a token grid padded to multiples of 32, with a minimum of 256x256 tokens, and a four-token
window shift. NGX changes the shifts of the five deepest layers every frame: the bottleneck layer follows an
eight-frame cycle and each shallower layer's shift is (2 * shift + 4) mod 8 of the one below. `makeKWindows`
reproduces that cycle (observed over 64 evaluations); `d4r-k` replays use the captured shifts or, for ordinary
frames, the fixed one. On a captured sequence the fixed shift costs 4-8 dB against NGX's. The helper supports output dimensions at least
256x256 and render dimensions no larger than output. Device image/buffer limits can impose a lower maximum than
the helper's 8192-pixel dimension limit. Only standard HDR output with zero output origin is supported.

The eleven layers can also run on the native HIP kernels (`kernels/k`, the accuracy HSACOs) with the stages
around them still on Vulkan, through two storage buffers shared by file descriptor. `compile_k.py --hip
KERNEL_DIR --capture CAPTURE_DIR` adds `hipnet.bin` (the layers' weight allocations as NGX uploads them) and
`hip/dltss_pwin_*_layer.hsaco` copied from `KERNEL_DIR`:

```sh
python3 engine/compile_k.py build/model-k build/k-case/lut.bin build/k-model --hip build/kernels-gfx1201 --capture "$CAP"
build/native-engine/d4r-k build/k-model build/k-case/case.txt build/k-results 25 hip
```

`Engine` takes the caller's `ExternalNetwork` (`tokens`, `Engine::networkTokenBytes`; `head`,
`Engine::networkHeadBytes`) and records the frame in two parts, `recordFront` (exposure and the input stage, which
writes the tokens) and `recordBack` (the final store, which reads the head). The network derives its token grid
from the output dimensions, runs the layers between the two parts on the shared buffers, and picks one of eight
immutable graphs from an evaluation counter that advances every evaluation and is never restarted by a reset.
`HipNetwork` (`hip_net.h`, native Linux) and the CUDA bridge's `d4rEngineNet*` (the game path) are the same
logic; both dispatch on `hipnet.bin`'s magic, so one bridge serves preset M and preset K.

The optional HIP replay accepts the native eight-frame window sequence starting at evaluation zero.
Custom windows or a capture beginning mid-cycle are rejected rather than silently using a different graph;
use the Vulkan replay for those cases.

### Shared M/K engine cutover and K accuracy limits

Preset K packages now use `D4RK0002`; rebuild old K packages for the current runtime. Automatic exposure
samples the output/2 grid with the native coordinate and luma associations, truncates the measured exposure
to half, and measures on alternate evaluations starting with the first. Reconstruction confidence derives
its two geometry scales from output/render rather than assuming a fixed 2× ratio. The filtered current luma
is rounded to half before the f32 blend; the blend, bright-luma encoding and raw four-lane mean retain the
native operation order. These changes apply to both native-network and Vulkan-network K.

M and K share `ExternalNetwork`, direct RGB10A2 stores, lazy canonical input allocation, direct-input
barrier elision, native package loading, GPU input readiness and callback-released split completion.
K's native backend includes all three bottleneck post kernels and eight immutable window graphs; M retains
two graphs. External-network engines do not allocate the unused Vulkan network activation buffers.
This does **not** mean zero network VRAM: the native K backend owns its own activations and weights.

The old bundled ZLUDA library omitted K's bottleneck post phases; its captured decoder input and bottleneck
output were zero. It is not a valid K accuracy reference. The current comparison uses the existing
phase-capable bridge with the same accuracy HSACOs. Its three captured frames match the independent
eleven-frame reference byte-for-byte.

With captured native tokens, all three tested native K network heads are byte-identical to the reference.
With captured native head/history, current K reconstruction is byte-identical for the three LDR colour,
luma and final-output surfaces; HDR final-output stage PSNR is 116.1–123.4 dB. The final scalar FMA
normalisation removes all 34 first-frame feature differences (983,040 values), but does not remove the
coupled drift. **These isolated checks do not prove lossless temporal feedback.**
The full, independently evolving eleven-frame sequence then had first-frame RGB PSNR 114.86 dB HDR and exact
LDR, but subsequent frames fell to 47.30–50.95 dB HDR and 47.36–52.02 dB LDR (fixed peak 1, unclipped RGB;
`build/engine-quality/k-accuracy-v2/native-engine-fixed-reference-quality.json`).

#### Temporal feedback made exact (2026-10-10)

That fall was not accumulating error. K is recurrent and its reconstruction takes discrete decisions, so any
one-code half-float difference entering the history is spread over most tokens by the next frame's network and
turned into sparse large errors: about a thousand differing texels do as much damage as thirty thousand. Three
causes were found and removed, none of which costs time:

- **One reconstruction table for every ratio.** NGX holds sixteen 32 KiB tables on a uniform grid of the
  upscaling ratio (0.5 + index × 3/28) and gives the output kernel the nearest even one for the geometric mean
  of the two axes' ratios, at most index 14 (ratio 2). The engine packaged the single table of a 2× capture and
  used it everywhere, so at Quality (1.5×, table 10) every frame with history filtered its luma with the wrong
  table. `lut.bin` now holds all sixteen (`prepare_k.py` writes NGX's whole allocation), the frame parameters
  carry the table in bits 8–11 of the output flags (`kReconstructionTable`), and a single-table package keeps
  the old behaviour. The rule matches NGX at all 49 measured sizes, including unequal axis ratios; HDR and LDR
  use the same tables. **Rebuild K models to get this.**
- **Input stage, mismatch channel.** NGX rounds the compressed history before subtracting it from the current
  average; the compiler fused that product into the subtraction, leaving the channel one code off in about 3% of
  the sub-blocks of every frame with history (30,805 of 983,040 values on the test frame). Now `precise`.
- **Output stage, sub-pixel positions.** NGX fuses the scale and the jitter offset and subtracts the tile origin
  afterwards. At some size ratios a whole output column lands on a half-float rounding tie, where any other
  order moves the luma filter's samples by one half-float step (252 history and 909 luma texels on the test frame).

Measured on the RX 9070 XT with the native HIP network, 1706×960 → 2560×1440, create flags 0x4b, against the
phase-capable CUDA reference (`hdr-phase-accuracy-hip-11`): the per-stage tests show zero differing codes on
every surface of frames 0–2; the three-frame replay on the engine's own history is byte-identical on all four
surfaces of every frame; and all eleven frames through the shim's game path are byte-identical, where the same
run with the previous shaders and package reproduces 114.86 dB then 49.64–47.30 dB. Whole-frame time of the
three-frame replay is unchanged within run-to-run noise (about 1.6–1.9 ms). Not measured: LDR end to end, other
ratios end to end, and the Vulkan network, which is not bit-exact against NGX and therefore still settles near
50 dB on later frames.

K gameplay HUD before the final feature-normalisation association fix: 1.52 ms current / 1.58 ms average, effective 1712×960 → 2560×1440
Quality, preset 11, LDR, native HIP network and direct RGB10A2 output. This is total in-game upscaler time,
not a kernel sum. Evidence: `build/engine-quality/ghost-k-final-gameplay.png` and
`build/engine-quality/ghost-k-final/d4r_nvngx.log`.
The final feature-normalisation package subsequently ran 20,400 evaluations through the in-game native K
path and shut down cleanly; no HUD screenshot was captured from that run, so the earlier timing is not
relabelled as a new final-build measurement.

Final shared-runtime verification preserved all 22 complete M HDR/LDR outputs byte-for-byte, passed all
114 six-frame direct/fallback adapter cases (57 per colour mode), and passed the nine model-corruption tests.
All 64 packaged SPIR-V modules were validated, including a subsequent validation of the four rebuilt K
output modules. The three-frame native K replay completed with repeated submissions, and incompatible
offset window schedules were explicitly rejected.



## Embedding

Use [runtime.h](runtime.h). With a suitable Vulkan device and a command buffer owned by the caller:

```cpp
d4r::Model model(physicalDevice, device, modelDirectory, pipelineCache);
auto shape = d4r::makeKShape(1280, 720);
d4r::Engine engine(model, shape);
model.recordUpload(initializationCommand);
engine.recordInitialize(initializationCommand);
// Submit initialization and wait for its fence.

d4r::FrameSettings settings{640, 360};
settings.reset = true;
auto parameters = d4r::makeKFrameParams(shape, settings);
engine.setFrame({colorView, motionView, depthView}, parameters.input, parameters.output);
engine.recordFrame(frameCommand);
// engine.outputImage() is RGBA16F in GENERAL, ready for a transfer read.
// Submit with the caller's rendering dependencies. Later frames can be recorded while this one is queued.
```

Borrowed images must be sampleable, accessible to the recording queue family, and alive (including their views)
until Engine destruction after GPU completion. `FrameImages` supplies each sampled layout, defaulting to `GENERAL`.
The caller must enable the device features listed in `runtime.h`. Optional `FrameOutput{view, format}` passed to
`setFrame` receives reconstruction directly: RGBA16F or RGB10A2 storage, `GENERAL`, covering
`outOrigin + outSize`. In this mode `outputImage()` still names the owned, unwritten fallback image.

[game.h](game.h) is the adapter for game textures: provide `GameImage::view` for compatible sampled inputs and
an RGBA16F or RGB10A2 storage output. It samples inputs and writes the output directly, respecting rectangle origins.
Null views retain per-plane copies. Other output formats and packages without the RGB10A2 variant retain the final conversion blit;
input/output image aliasing also uses the output blit. Views and images must outlive the adapter. Input formats
are sampled in their original precision, so f32 colour/motion avoid the former f16 intermediate rounding.

`compile_k.py` and `compile_m.py` package both final-store formats; `hasRgb10Output()` reports the packed variant.
Both compilers accept `--ldr` for NGX evaluations without the HDR flag. RGB10A2 stores share `output_store.glsl`:
the mantissa mask reproduces RADV's truncating RGBA16F store before UNORM conversion, rather than changing rounding
when removing the intermediate image. A GPU corpus of 324,614 values (all half patterns, signed half ties and their
adjacent floats, RGB/alpha quantization boundaries, random f32 and low-payload NaNs) matched the old half-store +
blit on every pixel. This format-conversion proof applies to the tested RX 9070 XT/RADV driver, not all vendors.

Paired output-path timing (2026-10-10, LDR 1712×960 → 2560×1440, 20 alternating samples per mode,
Vulkan network, warm GPU timestamps; not the in-game acceptance number):

| Final output path | Downsample median | Tail / conversion median | Whole engine median |
|---|---:|---:|---:|
| RGBA16F + RGB10A2 blit | 160.96 µs | 36.12 µs | 3327.32 µs |
| Direct RGB10A2 | 134.68 µs | 3.80 µs | 3272.04 µs |


K and M share `game_internal.h`'s canonical input resources and copy barriers. Each conversion plane is allocated
and transitioned on its first fallback frame; an all-direct frame allocates none of these planes and records no
extra transfer barrier. Depth staging is allocated only for a copied depth-aspect input. Later direct/copy changes
remain supported, including first fallback use after earlier direct frames were recorded. This avoids unused
allocations; it is not a measured steady-state speedup.

## In a game (D3D12 shim)

`scripts/build_d4r_nvngx_shim.sh` compiles the engine into `d4r_nvngx.dll`. It is off unless requested:

```sh
D4R_ENGINE=1 D4R_ENGINE_MODEL_DIR=/path/to/build/k-model   # the output of compile_k.py
```

The shim loads each model on first use from a folder next to it, or from a variable:

| Preset, colour | Folder | Variable | Built by |
|---|---|---|---|
| K, HDR flag set | `engine\k` | `D4R_ENGINE_MODEL_DIR` (`[Engine] ModelDir`) | `compile_k.py` |
| K, no HDR flag | `engine\k-ldr` | `D4R_ENGINE_MODEL_LDR_DIR` | `compile_k.py --ldr` |
| M, HDR flag set | `engine\m` | `D4R_ENGINE_MODEL_M_DIR` | `compile_m.py` |
| M, no HDR flag | `engine\m-ldr` | `D4R_ENGINE_MODEL_M_LDR_DIR` | `compile_m.py --ldr` |

`scripts/install_d4r_runtime.sh` stages them from `D4R_ENGINE_MODEL`, `D4R_ENGINE_MODEL_LDR`, `D4R_ENGINE_MODEL_M`
and `D4R_ENGINE_MODEL_M_LDR`, copying each folder whole (with `hipnet.bin` and `hip/` from `--hip`). A model whose
`hipnet.bin` belongs to the other preset keeps the Vulkan network. `D4R_ENGINE_NETWORK=vulkan` (`[Engine] Network`)
forces the Vulkan network for both presets; `D4R_SHIM_ENGINE_GPU_WAIT=0` makes the HIP network wait for its inputs on
the CPU-visible marker instead of a GPU-side wait, for comparisons.

With the engine enabled, an evaluation of preset K is recorded entirely into the game's own command list through vkd3d-proton's
`BeginVkCommandBufferInterop`: thirteen to fifteen dispatches (automatic exposure adds one or two), with direct
sampled inputs and a direct RGBA16F or RGB10A2 storage output when compatible. Unsupported usages/views keep their individual
input copies or output blit. There is no NGX
evaluation, CUDA launch, worker thread, frame marker, shared-memory import or split frame, and the game presents
the result of the frame it just rendered. The shim log says `engine backend: feature N runs preset K in the game's
command list`.

The shim hands a feature to the CUDA backend, with a log line giving the reason, when: the preset is not K or M;
the feature is preset M with display-resolution motion vectors (MVLowRes clear), which M does not implement yet; the
motion vectors are display-resolution and the K package predates `D4RO0003` (K with a current package takes them, and
jittered vectors are handled by adding the jitter change, as NGX does); the depth is regular (not inverted) and the K
package predates `D4RO0002`; HDR K runs without AutoExposure and either has no float exposure texture the engine can
sample directly or its package predates `D4RO0003`; a non-positive or non-finite exposure scale or pre-exposure is
passed; or
a texture format is not handled (colour/output RGBA16F, RGBA32F, R11G11B10F, RGB10A2,
RGBA8/BGRA8 UNORM; motion RG16F/RG32F/RGBA16F/RGBA32F; depth D32F, D32F_S8, R32F, R16F, R16_UNORM).

Limits of this first integration, all untested in a real game:

- vkd3d-proton does not enable `VK_KHR_cooperative_matrix` on the game's device. RADV compiles and runs the
  shaders anyway (measured: `D4R_ENGINE_TEST_NO_COOPMAT=1 d4r-k ...` creates such a device and gives byte-identical
  results), but that is outside the Vulkan specification and may not hold for other drivers or Mesa versions.
  The proper fix is a vkd3d-proton patch that enables the feature.
- Display-resolution motion (preset K, flag 0x49 without MVLowRes, `D4RO0003`) is byte-exact against the CUDA
  backend under the project's parity recipe: eleven frames each at 1706x960 -> 2560x1440 with create flags 0x49 (HDR)
  and at 1712x960 -> 2560x1440 with 0x68 (LDR), plus three frames at 1280x720 -> 2560x1440, are byte-identical on
  every frame, with output-sized vectors in output pixels and no depth. The CUDA reference must use a phase-capable
  ZLUDA (patch 0008, with the native K accuracy layers); the old bundled library omits K's bottleneck post
  phases and makes its own output differ by about 33 dB. The harness's panning scene (`D4R_HARNESS_QUALITY_PAN=1,0`,
  content moving one output pixel per frame) keeps frame 1 exact and then drifts to about 70-73 dB, with the same
  figures for render-resolution motion on the same scene, so that residual is the moving-content history drift and
  not the display-resolution path. M still uses CUDA for it.
- With MVJittered set (create flags 0x4f render-resolution or 0x4d display-resolution) the harness's jitter scene
  (`D4R_HARNESS_JITTER_SCENE=1`) is 109.8 dB on frame 1 and then drifts to about 69-72 dB; the same scene with
  MVJittered clear shows the same drift, so that is the scene's own content/rounding limit, not the offset. The
  jitter change is added to the raw vectors as `delta jitter / MV.Scale` in both motion modes, before
  `makeKFrameParams` applies the output-pixel conversion; applying an extra output/render factor there moves the
  render-resolution case from about 70 dB to about 33 dB.
- NGX and the CUDA bridge are still loaded for initialization/capability APIs. With the engine enabled, K's CUDA
  feature, parameters, 64 MiB scratch buffer and completion/profile events are created only when a frame actually
  needs fallback. Supported K frames never create that feature or compile its inference shaders. Other presets and
  engine-disabled features retain eager CUDA creation. The first fallback frame resets CUDA temporal history;
  failed creation frees partial resources so a later evaluation can retry.
- Exposure: HDR K reproduces automatic exposure (AutoExposure) and a game's own exposure texture (AutoExposure off; a float
  texture whose (0,0) texel the engine samples directly; `D4RO0003` packages). The texture replaces the measurement. The
  game's exposure scale and pre-exposure apply as `source × scale / pre-exposure` in both paths. Two details were needed for
  byte-exact results: the automatic measurement scales the luma it averages by `1 / pre-exposure` (as NGX's
  `cuda_luma_convert_kernel` does), and the previous frame's game exposure is carried into the ratio as a half with NGX's
  truncating store. Measured against the CUDA backend at 1706×960 → 2560×1440 with the native HIP network over eleven
  frames: automatic exposure with scale and pre-exposure 1, 0.5, 0.7, 2, 3 and combinations is byte-identical; HDR game
  exposure is byte-identical for texture values 0.4, 0.5, 0.6, 0.75, 0.8, 1, 1.2, 1.25, 1.5, 1.6, 2.5, 3, 5, 8, 16, 32,
  and for texture 1 with scale 0.5/2/4 or pre-exposure 2/0.5. The one exception: an effective exposure of exactly 2 or 4
  leaves a one-code difference in a few hundred to a few thousand texels on the first frame (104.7 dB and 112.8 dB RGB PSNR,
  first three differing frames 104.7/49.5 dB and 112.8/48.2 dB), which the recurrent network spreads over later frames
  (≈49–63 dB). The CUDA launches carry the same exposure, ratio and scale parameters, so that gap is an unresolved
  half-rounding tie in the reconstruction, not the exposure formula; the same values stay exact with automatic exposure.
  A zero texel reads as 1; negative and +∞ pass through and are byte-identical; a NaN passes through but the backends keep
  different NaN bit patterns (both outputs are NaN). LDR ignores the exposure entirely and is byte-identical for any value,
  scale, pre-exposure and AutoExposure flag. Preset M ignores the AutoExposure flag, the texture, the pre-exposure and the
  scale, so the shim does not refuse them (see the Preset M section).
- DLSS indicator axes (`DLSS.Indicator.Invert.X/Y.Axis`) only place NGX's debug indicator: with them set NGX's K and M
  outputs are byte-identical to unset runs on the harness (`D4R_HARNESS_INDICATOR_INVERT=1`, CUDA backend), and so are
  the engine's, so the shim no longer refuses them.
- Regular depth (K): 47–49 dB against the CUDA backend on the moving-box harness scene, the same as inverted depth; the wrong
  convention gives about 41 dB. Output subrectangles (K and M) are passed to the adapters, whose regression test
  covers nonzero output origins; they have not been run through the harness.
- NGX's eight-frame window cycle is counted from the feature's first evaluation; whether NGX restarts it on a
  reset was not checked.

Lazy-fallback verification: eleven moving/jittered Vulkan frames remain byte-identical to eager feature creation,
with no `NVSDK_NGX_CUDA_CreateFeature` call. Releasing a recorded-but-discarded evaluation and a newly created,
unevaluated feature both complete without CUDA creation. Changing exposure scale at frame 3 creates the fallback
exactly once; its first three outputs match a fresh CUDA feature's three outputs byte-for-byte, establishing the
history reset. A first-frame fallback also matches eager creation. `D4R_HARNESS_EXPOSURE_CHANGE_FRAME=N` changes
exposure scale to 0.5 at frame N for this transition probe.

The KCD2 portable installation was checked with an empty CUDA cache at 1706x960 -> 2560x1440: three Vulkan frames,
no CUDA feature creation, direct inputs/output active, and a byte-identical final image to the eager installed shim.
The whole isolated process took 11.2 seconds including Proton startup and NGX initialization; that is not a measured
in-game loading time. NGX initialization may still compile its basic CUDA support code, but the unused fallback
inference feature and its shader compilation are skipped.

The shim caches borrowed texture views and retains their D3D12 resources until the feature's recorded lists have
retired. Immutable descriptor bindings allow texture rotation and queued frames. New views are capped at 64 and
512 MiB of image memory requirements per feature; new textures beyond that budget use copies. Cached textures
stay allocated until feature retirement. The shim sends aliased input/output resources to the CUDA backend because
their D3D12 state conventions are ambiguous. Inputs denying shader-resource usage keep the copy path; direct outputs
need RGBA16F or RGB10A2 and `ALLOW_UNORDERED_ACCESS`. `D4R_ENGINE_DIRECT_INPUTS=0` and `D4R_ENGINE_DIRECT_OUTPUT=0` independently
force the old paths for comparisons. Older model packages keep copies for nonzero input origins. Recompile with
`compile_k.py` to obtain `direct_origins.bin` and shaders with corrected motion-origin handling.

The native adapter regression test covers six temporal frames per case, rotating three texture sets,
nonzero K origins, odd K output sizes, K render size changes, resets, automatic and game exposure (K), display-resolution
motion (K), separate direct/copy planes, R32F/D32F depth, read-only input layouts, RGBA32F output fallback, and RGBA32F
colour/RG32F motion against the unquantized Engine. K and M switch each input to fallback for the first time after
startup, then back again. Both also rotate RGB10A2 outputs and mixed RGBA16F/RGB10A2 outputs, with queued and
individually submitted recordings, comparing every output byte against the old conversion path. The K checks need a
package from the current `compile_k.py` (`D4RO0003`). HDR and LDR model pairs each passed every case of six temporal
frames on the RX 9070 XT:

```sh
bash engine/build.sh
# Reuse the replay runner's device/test helpers for the adapter checks.
g++ -std=c++17 -O3 -Wall -Wextra -Wno-missing-field-initializers engine/test_game.cpp \
  -Lbuild/native-engine -ld4r_engine -lvulkan '-Wl,-rpath,$ORIGIN' -o build/native-engine/test-game
build/native-engine/test-game build/k-model
# Include M's direct-input and late-fallback checks:
build/native-engine/test-game build/k-model build/m-model
```

### Direct-texture measurements (2026-10-08, RX 9070 XT)

D3D12 harness, automatic exposure, three rotating-order rounds per mode/resolution, 140 frames per round with
frames 1–40 discarded and no pause between frames. Medians of the three run medians, evaluation call through the
completed submission fence (includes CPU recording/submission overhead, not game FPS):

| Adapter mode | 640x360 -> 1280x720 | 1920x1080 -> 3840x2160 |
|---|---:|---:|
| Input copies + output blit | 0.857 ms | 3.240 ms |
| Direct inputs + output blit | 0.924 ms | 3.212 ms |
| Input copies + direct output | 0.752 ms | 3.017 ms |
| Direct inputs + direct output | 0.728 ms | 2.989 ms |

The 4K runs are stable: about 0.028 ms saved by inputs, 0.223 ms by output, and 0.252 ms (7.8%) with both.
720p is bimodal/clock-sensitive: copying run medians range 0.837–1.017 ms and direct-input-only medians range
0.851–0.987 ms, so there is no established input-only speedup there. Both/direct-output medians range
0.717–0.745 / 0.743–0.761 ms, respectively. These are adapter measurements; do not add them to inference-only
benchmarks or treat the differences as isolated transfer timings.

Eleven moving/jittered quality frames match the copy adapter byte-for-byte, including a separate run recreating
the output every frame and using RGBA16F motion textures. Seventeen native adapter comparisons pass across six
frames each (including multiple frames recorded before submission). Preserved RGBA32F colour/RG32F motion matches
the unquantized Engine; it is not expected to match the old f16 conversion path. Results establish correctness
on this device; RTX perceptual parity remains unverified. A 20-frame 4K harness run recreating its output each
frame exercised the retained-view budget and switched new outputs to blits with a byte-identical final result.
RGB10A2 output fallback likewise matched the copy adapter.

### Harness results (2026-10-08, RX 9070 XT)

`tools/d3d12_dlss_harness.cpp`, quality scene (blinds, thin poles and fine diagonals near the render pixel pitch)
panning 2,1 output pixels per frame, 640x360 -> 1280x720, eleven frames, engine against the CUDA backend:

| Reference | Final RGB PSNR |
|---|---|
| CUDA backend with d4r's native K layer kernels | 53.2 dB on the reset frame, then 56.5-60.9 dB |
| CUDA backend with ZLUDA's translation of NVIDIA's K layers | 44.6-52.8 dB |
| the two CUDA variants against each other | 44.7-52.9 dB |

These earlier eleven-frame comparisons do not establish general numerical agreement; the longer, matched
accuracy-HIP comparisons below found larger Vulkan differences. Which variant is closer to an NVIDIA GPU is
not established here. Against the earlier scene's analytic ground truth the engine scores 20.07 dB and the CUDA
backend 19.48-19.59 dB; those scores alone do not establish equivalent image quality.

#### Matched accuracy-HIP comparison (2026-10-08)

All three paths ran identical panning (2,1 pixels/frame), jittered quality-scene inputs with automatic exposure.
HIP used split frames, so outputs are aligned to the current evaluation, as with Vulkan. The reference used
`PreferAccuracy=1`, the marked conservative native K layer set, and translated remaining kernels with the
accuracy compiler/synchronization policy. Fast HIP used `k-shipping-deep2-fast` with `PreferAccuracy=0`;
Vulkan used the folded-gain `build/k-store/final-model`, including the enc0 store optimization.
Logs verified all eleven native HIP overrides and the respective policies. The 1440p accuracy reference
was repeated and all sixteen output frames matched byte-for-byte.

RGB PSNR against accuracy HIP, fixed peak 1, without clipping; alpha excluded. Post-reset PSNR pools
squared error over all subsequent frames rather than averaging their dB values:

| Case | Fast HIP reset | Vulkan reset | Fast HIP post-reset | Vulkan post-reset |
|---|---:|---:|---:|---:|
| 640x360 -> 1280x720, 32 frames | 52.41 dB | 40.89 dB | 60.89 dB | 46.43 dB |
| 1280x720 -> 2560x1440, 16 frames | 52.77 dB | 41.49 dB | 59.74 dB | 48.02 dB |

Individual post-reset frames ranged 55.87–69.77 dB for fast HIP and 34.98–60.31 dB for Vulkan at 720p;
56.45–63.73 dB and 44.58–52.65 dB respectively at 1440p. Fast HIP is substantially closer to accuracy HIP
in these tests. These are synthetic temporal scenes on RX 9070 XT, not game or NVIDIA reference captures.
Per-frame metrics, exact kernel/model hashes and the comparison script are in `build/k-accuracy-compare/`;
raw outputs and run logs are `build/engine-work/hout/compare-{720,1440}-{accuracy,fast,vulkan}.*`.

Wall time of one evaluation in the harness, from the call to the completed fence, with both backends presenting
the frame's own result (CUDA: `D4R_SHIM_VRAM_INTEROP=1 D4R_SHIM_SPLIT_FRAME=1`, native K kernels); medians of 200
frames after 40 warm-up frames:

| Resolution | Engine | CUDA backend |
|---|---:|---:|
| 640x360 -> 1280x720 | 0.98 ms (bimodal, 0.74-1.03) | 1.18 ms |
| 1706x960 -> 2560x1440 | 1.78 ms | 2.28 ms |
| 1920x1080 -> 3840x2160 | 3.44 ms | 4.25 ms |

The harness submits one list per frame and waits for it, so these figures include submission and fence overhead
that a game shares with its own rendering. They are not game frame rates.

## Numerics and measured results (2026-10-08)

GPU: RX 9070 XT, RADV/Mesa 26.2.4. Timings use GPU timestamps around the **complete** input/network/reconstruction
graph. Nine measured batches follow three warmups; each batch repeats a fixed input as 25 successive temporal
frames, preserving dependencies and feedback. Uploads, compilation and validation readbacks are excluded.
These are sustained benchmarks, not game timings or matched-clock comparisons against another engine.

| Configuration | 1280x720 output | 3840x2160 output | Activation buffers at 4K |
|---|---:|---:|---:|
| folded gains | about 0.60 ms | 2.88 ms | 109.6 MiB |
| explicit f16 gains/residual boundaries | about 0.65 ms | 2.95 ms | 109.6 MiB |

Activation memory excludes weights, temporal images, output, and borrowed frame inputs. At 720p it is 17.2 MiB.
The 4K timing uses ordinary-frame parameters with the fixed window schedule. Full temporal capture comparison
was performed at 720p, where final RGB is 59.9–61.8 dB PSNR over six frames for folded gains, and 60.3–62.4 dB for
explicit gains. All outputs are finite. The first frame's network head is byte-identical between the persistent
runtime and the isolated shader chain (3,276,800 f16 values).

The default extractor folds per-channel gains into weight rows, rounding once to f16. `extract_k.py --unfold-gains`
preserves explicit f16 gain multiplication. Both modes now preserve the f16 residual additions at the MLP boundary
and use f32 matrix accumulation. Neither is bit-identical to NVIDIA: small head differences
can cross discrete reconstruction-table boundaries and cause larger errors at isolated pixels. PSNR is a numerical
check on these captures, not evidence of perceptual parity or state-of-the-art quality.

The input shader was additionally compared on synthetic motion/depth: current/history luma matched the translated
reference exactly, token feature and mismatch at 122 dB and 131 dB PSNR. It samples the history colour and the
previous token feature bilinearly, as NGX's texture objects are created; with point sampling (an earlier mistake
that zero-motion captures could not expose) a panning capture's token feature dropped to 23 dB. The output shader on synthetic motion/depth,
given the captured head, measured 78.4 dB for hi-res luma and 86.8 dB for final RGB. Translation is used only for
these reference comparisons, never by the engine. Both model modes' 26 SPIR-V modules passed `spirv-val`.

```sh
python3 -m unittest discover -s engine -p test_model.py
python3 -m unittest discover -s tests
```

## Per-stage K profiling

Set `D4R_ENGINE_PROFILE=1` when running `d4r-k` to record GPU timestamps at frame entry and after setup,
exposure, input, each of the eleven network layers, reconstruction, and final synchronization:

```sh
D4R_ENGINE_PROFILE=1 build/native-engine/d4r-k build/k-model build/k-case/case.txt - 25
```

With timing repeats, stdout reports each stage's median across the nine measured batches, after three warmups.
Each batch averages successive temporal frames. With readback enabled, `stages.csv` also records the initial
sequence's per-frame intervals in microseconds; these initial frames are not warmed-up benchmark samples.
Without repeats, the per-frame stages are printed. The default run records no extra timestamps. Embedders can
use the optional `FrameProfiler` callback in `runtime.h` to record their own markers.

Intervals include the dependencies preceding each dispatch and timestamp overhead; they are not isolated
kernel timings. Timestamps can perturb scheduling, so use profiling to locate costs and ordinary, uninstrumented
runs to assess an optimization. Input uploads, game adapter copies/blits, and readbacks are outside these timings.
The exposure interval contains only marker overhead when automatic exposure is disabled.

Checked on 2026-10-08, RX 9070 XT: profiling on and off both produced byte-identical head, final output and all
three histories to the pre-instrumentation executable over a five-frame 720p capture (25 binary files per mode).
All four reference checks passed the default 50 dB threshold; final RGB was 59.9–61.8 dB.

Three alternating profiling-on/off runs per resolution, 25 frames per batch, folded gains:

| Stage/group | 640x360 -> 1280x720 capture | 1920x1080 -> 3840x2160 synthetic |
|---|---:|---:|
| Input | 41.9 us | 316.1 us |
| Eleven network layers (sum of stage medians) | 409.9 us | 1456.0 us |
| Reconstruction | 93.8 us | 903.8 us |
| Whole frame, profiling enabled | 551.0 us | 2681.1 us |
| Whole frame, profiling disabled | 558.0 us | 2762.9 us |

The 4K case uses constant HDR colour, inverse depth and zero motion with the fixed window schedule, not a game
or quality reference. Automatic exposure is disabled in both cases. Whole-frame figures are medians of the three
run medians; stages are medians of the three stage medians and need not sum exactly to the whole-frame median.
The instrumented totals were 1.3% and 3.0% lower; this is a measurement perturbation, not an inference speedup.
Reconstruction accounts for about 34% of the profiled 4K total. The largest individual network stages there
were dec0 (300.1 us), enc0 (284.9 us), and dec1 (216.1 us).

### Reconstruction tuning (2026-10-08)

The Gaussian filter calls `exp2` directly on its f16 exponent, avoiding an explicit promotion to f32 and
conversion back to f16. On RX 9070 XT/RADV this produced byte-identical final output, head and all three histories
on both the five-frame capture and a five-frame synthetic stress sequence (50 binary files checked). The latter
covers fractional motion, off-screen reprojection, jitter, HDR highlights, exposure changes, mid-sequence reset,
and odd 1283x723 output dimensions. The captured reference quality checks still pass. This establishes numerical
agreement on this device, not on other drivers.

Five alternating baseline/candidate runs per resolution, profiling disabled, 50 temporal frames per batch:

| Case | Baseline run medians | Candidate run medians | Median paired reduction |
|---|---:|---:|---:|
| 720p capture | 582.4–604.8 us | 578.0–602.0 us | 3.1 us (about 0.5%) |
| 4K synthetic, as above | 2738.6–2781.8 us | 2732.5–2759.7 us | 23.0 us (about 0.8%) |

The 720p runs were bimodal, so comparing unpaired medians would overstate the gain. This is a small performance
change with no observed quality change. Explicit history/filter loop unrolling did not improve the 4K stage;
forcing loops to remain rolled increased it to about 1.14 ms. Reconstruction with wave32 was also slower,
about 1.06 ms versus the wave64 baseline's 0.90–0.91 ms. Those variants are not used.

### enc0 matrix stores and accuracy investigation (2026-10-08)

The 32-channel enc0 block now uses padded column-major shared token tiles. This lets RADV pack
cooperative-matrix stores into wider LDS writes without changing arithmetic or accumulation order.
On RX 9070 XT, static 16-bit LDS store instructions fell from 183 to 48, allocated VGPRs from
192 to 120, and shared memory from 10,240 to 9,216 bytes. Addressing uses cooperative-matrix
layouts, with no hardware fragment mapping. Other layers keep their existing shared layout.
The analogous dec0 variants did not show a useful improvement and were discarded.

Five alternating baseline/candidate runs, profiling disabled, 50 temporal frames per batch:

| Case | Baseline run medians | Candidate run medians | Median paired reduction |
|---|---:|---:|---:|
| 720p capture | 595.0–610.2 us | 584.2–600.3 us | 10.2 us (about 1.7%) |
| 1920x1080 -> 2560x1440 synthetic | 1418.8–1421.9 us | 1405.0–1409.0 us | 12.7 us (about 0.9%) |
| 1920x1080 -> 3840x2160 synthetic | 2770.8–2782.5 us | 2751.8–2756.3 us | 20.6 us (about 0.7%) |

Earlier 720p trials showed no gain, so the low-resolution benefit remains uncertain. Synthetic
cases use constant inputs and the fixed window schedule; these are engine GPU timings, not game
frame timings. Results and diagnostic variants are in `build/k-store/`, including
`final-benchmark.json`; the validated folded package is `build/k-store/final-model`.

Head, output and all three histories were byte-identical to the baseline across five captured,
two independent captured, and five synthetic stress frames (60 files). Explicit-gain output
also remained byte-identical across the five-frame capture (25 files). All 17 native game-adapter
comparisons passed, as did nine model tests and SPIR-V validation of both packages. Eleven moving,
jittered D3D harness frames and the final output matched the baseline byte-for-byte. The independent
capture's first-frame hi-res luma scores 49.2 dB against NGX, below the default 50 dB threshold in
both baseline and candidate; this change preserves that existing discrepancy.

Accuracy experiments covered explicit gains and f16 normalization/residual boundaries, per layer
and in combinations. Explicit enc1+dec0 improved mean RGB PSNR by 0.44 dB on the first capture,
but reduced it by 0.83 dB on the independent capture. All-explicit gains improved the first by
0.70 dB but reduced the second by 0.16 dB. No tested boundary variant improved both captures
consistently, so no accuracy variant became the default. `--unfold-gains` remains available;
the retained enc0 change preserves numerical behavior on this device.

## Preset M (runtime and shim path exist; experimental)

Preset M (DLSS 4.5, `rrlite_*`) is a different pipeline: an input stage that embeds 2x2-pixel tokens, a network of
4x4-token-window Swin blocks with FP8 (e4m3) weights and activations (enc1, enc2, six "tube" blocks, dec2, dec1),
a stage that expands the tokens to per-pixel reconstruction inputs, a reconstruction at 1.5 times the output
resolution, and a 3 -> 2 downsample. `M_NOTES.md` has the algorithms. Every stage has its own shader and a test
against captured launches:

| Stage | Shader | Against the capture (640x360 -> 1280x720, 2026-10-08) |
|---|---|---|
| input | `enc0_m.comp` | motion surfaces identical; tokens 99.2% identical on the reset frame, 94.3% with history |
| network | `layer_m.comp` | per layer on the captured inputs, all ten layers and all three captured frames: byte-identical to the capture (2026-10-09) |
| expansion | `dec0_m.comp` | 96-99.6% of the 8-bit values identical; covariance 70-73 dB |
| reconstruction | `post_m.comp` | 61.5 dB on the reset frame, 76.9 dB with history; packed output 94-98% identical |
| downsample | `down_m.comp` | 70-71 dB, 92% identical |

`test_frame_m.py` chains all of them over captured frames with the engine's own intermediate data and history.
Final image, three frames of the panning stress scene:

| Compared with | Final RGB PSNR |
|---|---|
| CUDA backend, d4r's native (fast) M kernels | 36.2, 35.9, 35.9 dB |
| CUDA backend, ZLUDA's translation of NVIDIA's kernels (exact arithmetic) | 33.1, 29.4 dB |
| the two CUDA variants against each other | 32.8, 29.2 dB |

The loss is in the network: its activations are FP8, one rounding flip is a 6-12% change, and the flips compound
through ten layers (the network output is at 24-30 dB whichever two implementations are compared). The engine is as
far from NVIDIA's exact arithmetic as d4r's shipping fast kernels are, no further. That was the state in 2026-10-08.
`layer_m.comp` now reproduces those ten layers bit for bit (2026-10-09): NVIDIA's binary16 rounding after every 32
products (through a per-invocation scratch the compiler cannot fold), the native norms' summation order
(`tok_rsq` on the register element order `tok_xnat` implies, which is not the shader's own accumulator order) and
the native attention tree. Every h1 and h2 FP8 operand, both residual streams and every output of all ten layers
are byte-identical to the capture, and the whole coupled engine (front, network, back) is byte-identical to the
same engine running the native HIP network on the three captured frames.

Current matched full-frame comparison against the old accuracy-HIP path (RX 9070 XT, 2026-10-10), fixed peak 1,
RGB only, unclipped RGBA16F outputs, eleven quality-scene frames with independent temporal feedback:

| Case | RGB PSNR range | Reset RGB PSNR | Last-frame RGB PSNR | Maximum absolute RGB error |
|---|---:|---:|---:|---:|
| HDR, 1706×960 → 2560×1440 | 68.65–70.36 dB | 69.12 dB | 69.66 dB | 0.008789 |
| LDR, 1712×960 → 2560×1440 | 79.19–80.76 dB | 79.19 dB | 80.45 dB | 0.001465 |

The previous feedback divergence is fixed. Native RGB10 half-history uses RGBA32F backing: packed Vulkan UNORM
bilinear filtering differs by up to three float ulps. M now uses the same float backing, with correctly rounded
`code/1023` normalization (a two-FMA residual correction; plain constant division is lowered to an inexact
reciprocal multiply). All 1024 codes and 524,288 native/Vulkan sampling coordinates match bit-for-bit.

Native paired reconstruction rows also round the odd row's own-cell offset to binary16 **before** subtracting
one cell in binary16. A single rounding relative to the even row's cell changed Gaussian weights, first appearing
at frame nine in this scene and causing the frame-ten feedback cliff. The corrected order reproduces all three
post outputs on the failing captured frames exactly. The packed and half-mean paths retain their explicit
`precise` rounding boundaries.

These are near-lossless comparisons, not bit-identical final output or an in-game timing claim. Raw comparisons:
`build/engine-quality/m-{hdr,ldr}-paired-f32-quality-11.raw.frameN` against
`m-{hdr,ldr}-matched-accuracy-11.raw.frameN`; metrics: `m-paired-f32-quality-11.json`.
Archived `.frame.001` scene captures are not interchangeable with this current harness scene.

Runtime: `MModel` / `MEngine` in `runtime.h` (implemented in `runtime_m.inc`), `GameUpscalerM` in `game_m.h`, the
model directory from `compile_m.py CAPTURE OUT` (weights of one captured frame plus the compiled shaders), and the
runner `d4r-m`. Exposure is measured on the GPU every frame (preset M's formula, see `M_NOTES.md`). NGX does that whatever
the AutoExposure flag, the game's exposure texture, the pre-exposure and the exposure scale say (CUDA output is identical for
each), and the engine output is byte-identical for each too. The shim records
preset M through `engine_evaluate_m` when `[Engine] Enabled` is set and the model is in `engine\m` next to the DLL
(or `D4R_ENGINE_MODEL_M_DIR`).

M packages now use `D4RM0003`. Rebuild with the current `compile_m.py`: the blocked FP8 operands,
RGBA32F half-history bindings and 8×64 downsample dispatch must match the runtime. `D4RM0001` and
`D4RM0002` packages are rejected rather than silently loading incompatible shaders. Native `hipnet.bin`
remains `D4RMHIP1`; its tensor and weight layout is unchanged.

Compile `--ldr` and stage the package in `engine\m-ldr` (or set `D4R_ENGINE_MODEL_M_LDR_DIR`) for evaluations without
NGX's HDR flag. The display's HDR setting does not determine this flag. M's adapter and native HIP network are
initialized from the first evaluation's effective render extent, not the nominal feature-creation dimensions:
Ghost of Tsushima creates a 1706×960 feature but evaluates 1712×960 inputs. Subsequent M extent changes still
require a new feature. K's adapter capacity also covers the first effective extent when it exceeds the nominal size.

D3D12 harness, engine against the CUDA backend with native M kernels (2026-10-09):

| | Engine | CUDA backend |
|---|---:|---:|
| final RGB PSNR between them, 11 panning frames at 640x360 -> 1280x720 | 32.2-37.3 dB | |
| against the scene's ground truth | 18.64 dB | 18.52 dB |
| one evaluation, call to fence, 640x360 -> 1280x720 | 1.00 ms | 1.33 ms |
| one evaluation, call to fence, 1920x1080 -> 3840x2160 | 6.21 ms | 5.96 ms |

That early harness measured an untuned HDR-only path, with copied inputs and no game verification. Those
limitations and the then-unverified half-history reconstruction do not describe the current implementation;
the matched temporal accuracy and actual game observations above and below supersede them.

### The network on the native HIP layers (2026-10-09)

The Vulkan layer shader runs preset M's network about 1.3x slower than d4r's native HIP layer kernels
(`kernels/m`, `rrlite_*_4x4.hsaco`): a paired same-session engine benchmark on the three captured frames
(1706x960 -> 2560x1440, 30 timed frames each, Vulkan and HIP alternating) measured 2.14-2.18 ms against
1.65-1.68 ms before the tiled-weight cutover. Both produced byte-identical images.
The FP8 weight packer now stores logical W[K][N] as [K/16][N/16][N16][K16]. Each 256-byte tile supplies
the shader's A[N16][K16] operand through a fixed-stride-16 row-major load, rather than a stride-N
column-major load. This is a byte permutation: FP8 values, f16 vectors, offsets, activation layouts,
front-end weights and the native HIP network package are unchanged. Compiler output replaces the
transposed global weight loads with ordinary loads.

Paired full-network measurements of that cutover on this GPU:

| Weight operands | Network median, first session | Network median, second session |
|---|---:|---:|
| Plain, column-major load | 2174.4 µs | 2181.4 µs |
| Tiled, row-major load | 2117.3 µs | 2119.0 µs |

All ten native captured layers across three frames, including merged outputs, remain byte-identical.
The integrated packages also preserve all three coupled full-frame outputs with either network backend.
These are network/component proofs, not an in-game total or a fix for the longer recurrent-quality drift.
`layer_m.comp`'s remaining FP8 conversion and register-scheduling costs are optimization targets. The Vulkan FP8
cooperative-matrix API exposes K16 on this device; these measurements do not establish a hardware limit or prove
that better Vulkan kernels cannot match HIP.

So the network can run on those HIP kernels while the stages around it, which are faster on Vulkan than their HIP
counterparts, stay where they are (`[Engine] Network`, default `auto` = HIP when available):

- `compile_m.py CAPTURE OUT --hip KERNEL_DIR` adds `hipnet.bin` (the ten layers' weight allocations as NGX
  uploads them, shifts and grid divisors), `hip/*.hsaco`, and the input and expansion stages compiled with
  `-DPLANES=1` (token buffers in NGX's layout, two planes of 32 channels, which the kernels read and write).
- `MEngine` takes two caller-owned buffers (`ExternalNetwork`) and records a frame in two parts, `recordFront`
  (exposure, input stage) and `recordBack` (expansion, reconstruction, downsample).
- `hip_net.h` (`HipNetwork`, native Linux, used by `d4r-m ... REPEATS hip`) and the CUDA bridge's `d4rEngineNet*`
  (the same logic in C for the game path) load the code objects with HIP directly, upload the weights once, run each
  layer's weight-preparation kernel once (the tube keeps one prepared image per weights pointer), and launch the ten
  layers per frame. No NGX, no PTX and no ZLUDA translation are involved; the bridge's library only supplies the
  HIP context.
- In the shim the game's command list is split at the network: front, frame marker,
  `SplitCommandListForExternalWait`, back. The native HIP stream waits for the front's GPU-published frame number.
  A native HIP completion callback signals the Vulkan timeline directly; Wine-worker polling retires resources
  but is not needed to release the back half. The game still shows the frame it just rendered. This needs the
  d4r vkd3d-proton and VRAM interop; otherwise the Vulkan network is used and the log says why.
- The native network uses a nonblocking stream at the device's greatest supported priority: it is on the frame's
  critical path between the two Vulkan parts. This changes scheduling, not the layer arithmetic. The front's
  invocation-private half-rounding scratch uses memory fences without subgroup execution barriers; all three
  coupled HDR frame outputs remained byte-identical to the prior front after this change.

The input GEMM keeps four accumulators live and shares each activation load across them. Captured tokens and
motion outputs remain exact in both token layouts. A further store-epilogue change hoists each lane's token
base address and uses one uniform bounds check for fully in-screen blocks; border blocks retain individual
row checks. Paired RX 9070 XT probes at 1706×960 measured 152.2 → 144.2 µs for interleaved tokens and
156.1 → 146.1 µs for plane-layout tokens. All eleven fresh captured frames preserved both output buffers
byte-for-byte in each layout. This is a stage result, not an in-game total.
The integrated `D4RM0002` runtime and both model packages also preserved all 22 complete independent-feedback
HDR/LDR frame outputs after the address hoist. Current HDR/LDR adapter checks passed all 114 six-frame cases,
including queued mixed-format output and late per-input fallback.

The native accuracy network pairs adjacent K16 operand loads in `dec1` and `enc3_tube`. The tube's MLP uses
two-chunk unrolling without changing accumulation order. Six alternating, clean network-only runs measured
median 1630.05 → 1611.05 µs (19 µs, 1.17%). All 22 complete HDR/LDR independent-feedback outputs remained
byte-identical after this native scheduling change. This is not an in-game total.

The expansion shares each activation tile across three accumulator tiles, then the remaining two, while
retaining the binary16 boundary after each K32 chunk. Six paired 32-lane replay rounds preserved all four
output buffers in both token layouts; native plane-layout medians were 100.55 → 97.95 µs.

Reconstruction uses 128 invocations per 16×16-pixel block, with each invocation owning a vertical pixel pair.
It materializes the vertical column sum before adding the adjacent column, retaining the native mean's
association and odd-row residual rounding. Captured HDR/LDR outputs and 15 partial-edge geometries remained
byte-identical. An initial isolated timing gain was not reproducible, so no stage-speed claim is made for this
change; acceptance depends on the integrated in-game total.

Downsampling uses an 8×64 output block, reducing the repeated horizontal filtering of border rows. All decoded
values, filter associations, tap-range clamps and final output rounding are unchanged. Bounds are checked
after both shared barriers against the feature output, not the borrowed image's potentially larger extent.
The combined `D4RM0003` shaders/runtime preserved all 22 complete HDR/LDR independent-feedback outputs,
byte-for-byte against the preceding native-tuned package. Weight payloads and offsets after the magic are
unchanged. The current runtime also passed 114 direct/fallback adapter cases, each with six temporal frames.
The standalone three-frame replay retained exact front tokens, network outputs and all expansion outputs;
its final output remained 68.8–71.5 dB against the captured accuracy HIP output.

Actual Ghost of Tsushima HUD observations (2026-10-10), preset 13/M, Quality, effective
1712×960 → 2560×1440, LDR, direct RGB10A2 output:

| Observation | Network | Current upscaler time | Average upscaler time |
|---|---|---:|---:|
| Verified gameplay before compact reconstruction tier | Native HIP | 2.61 ms | 2.63 ms |
| Later user HUD with compact tier; scene not shown in crop | Native HIP | 2.89 ms | 2.89 ms |
| User menu HUD with compact tier, no HIP handoff | Vulkan | 3.05 ms | 3.08 ms |
| Verified gameplay with corrected float half-history and paired-row rounding, before store-address hoisting | Native HIP | 2.59 ms | 2.53 ms |
| Verified gameplay with tiled Vulkan weights, corrected histories and store-address hoisting | Vulkan | 2.85 ms | 2.87 ms |
| Verified gameplay with corrected histories and store-address hoisting, before the final native load-scheduling change | Native HIP | 2.49 ms | 2.52 ms |
| Verified gameplay after the final native load-scheduling change, first observation | Native HIP | 2.52 ms | 2.50 ms |
| Same gameplay after settling, second observation | Native HIP | 2.51 ms | 2.51 ms |
| Verified gameplay with the combined `D4RM0003` reconstruction kernels | Native HIP | 2.44 ms | 2.47 ms |
| Same `D4RM0003` gameplay after settling, second observation | Native HIP | 2.49 ms | 2.49 ms |
| Final shared M/K runtime recheck, first gameplay observation | Native HIP | 2.47 ms | 2.54 ms |
| Final shared M/K runtime recheck, settled observation | Native HIP | 2.59 ms | 2.56 ms |

The combined `D4RM0003` package reached the 2.5 ms total in-game target in the earlier gameplay samples.
The final shared-runtime recheck averaged 2.54–2.56 ms, so a sustained ≤2.5 ms claim is **not** established.
The final samples are `build/engine-quality/ghost-m-final-gameplay{,-settled}.png`; the log confirms preset
13/M, Quality, effective 1712×960 → 2560×1440, native HIP network and direct RGB10A2 output.
No scene-independent timing guarantee is implied, and these measurements do not isolate whether the small
change comes from shared-runtime overhead, GPU clock state or the evolving scene.
The compact 11×11 reconstruction tile and conditional fallback fetch save about 26 µs in paired LDR stage
probes at this geometry, but that isolated saving alone did **not** establish an in-game improvement.
Removing the HIP handoff did not improve the observed total either; the native-network engine remains
the default. These observations do not isolate each new kernel's contribution.
The menu is not an acceptance scene; stage sums and callback launch-to-completion spans are not substitutes
for the HUD number. The runtime retains 10×10 for render/reconstruction ≤ 1/3 and 15×15 for wider ratios;
11×11 requires both ratios ≤ 0.45 and both jitter components in [-0.5, 0.5].
The compact tier and conditional fallback fetch preserved every output byte over two complete
eleven-frame feedback sequences (HDR and LDR), compared with the preceding engine package. The subsequent
float-history and paired-row rounding corrections fix the accuracy-HIP divergence independently of that tier.

Earlier measurements, before the current accuracy and output changes, on an RX 9070 XT:
`d4r-m`, synthetic frames, wall clock from recording to completion (GPU stage sum):

| | Vulkan network | HIP network |
|---|---:|---:|
| 1280x720 -> 3840x2160 | 3.22 ms (2.78) | 2.39 ms (2.19) |
| 1920x1080 -> 3840x2160 | 5.52 ms (5.16) | 3.81 ms (3.62) |
| network alone at 1080p | 3.57 ms | 2.07 ms |

D3D12 harness, one evaluation from the call to the fence, every path showing the frame's own result:

| | Engine, HIP network | Engine, Vulkan network | CUDA backend, native kernels |
|---|---:|---:|---:|
| 1280x720 -> 2560x1440 | 2.19 ms | 2.91 ms | 2.87 ms |
| 1280x720 -> 3840x2160 | 3.32 ms | 3.91 ms | 4.00 ms |
| 1920x1080 -> 3840x2160 | 4.54 ms | 6.23 ms | 5.38 ms |

Those earlier harness images measured 44–47 dB against the CUDA backend (HDR) and 36–37 dB without the HDR
flag; they are not current accuracy results. Feature release and re-creation at alternating render sizes ran
clean. The current game timing and longer accuracy comparison are reported above. The kernels are built per
GPU architecture (gfx1201 with FP8 here); another GPU falls back to Vulkan when they do not load.

## Files

| File | Role |
|---|---|
| `runtime.h`, `runtime.cpp` | Embeddable model/engine, persistent resources, frame parameter helpers, temporal loop |
| `game.h`, `game.cpp` | Adapter for a game's textures: format conversion, frame, output blit, in one command buffer |
| `vk_dispatch.h`, `vk_dispatch.cpp` | Vulkan function table for builds without a Vulkan import library (the MinGW shim) |
| `replay.cpp`, `build.sh` | Full-frame execution, capability checks, correctness readbacks, GPU timing |
| `extract_k.py`, `compile_k.py` | Weight extraction, validation, model/shader packaging |
| `capture.py`, `prepare_k.py` | Capture decoding and complete-frame reference case preparation |
| `prepare_frames.py` | Ordinary raw HDR frame preparation without NGX parameter blocks |
| `input_k.comp`, `layer.comp`, `output_k.comp` | Own input, fused network, and reconstruction shaders |
| `m_model.py`, `layer_m.comp`, `test_m.py` | Preset M: logical weights and numpy model, layer shader, tiled FP8 packer and per-layer test |
| `hip_net.h`, `hip_net.cpp` | Shared M/K native HIP network loading and immutable graph execution, for native Linux programs |
| `enc0_m.comp`, `dec0_m.comp`, `post_m.comp`, `down_m.comp` | Preset M: input, expansion, reconstruction and downsample stages |
| `activation_m.glsl`, `gen_activation_m.py` | Preset M: sigmoid/tanh/gate table of the 256 E4M3 codes and the script that computes it (`--check` verifies the table) |
| `test_enc0_m.py`, `test_dec0_m.py`, `test_post_m.py`, `test_down_m.py`, `test_frame_m.py` | Preset M: per-stage and whole-frame tests |
| `M_NOTES.md` | Preset M: the algorithms as reverse-engineered |
| `exposure_k.comp` | Automatic exposure: log-luma block sums, then the exposure state and the stages' exposure fields |
| `test_frame.py`, `test_model.py` | Temporal result checks and model corruption tests |
| `test_game.cpp` | Queued direct-texture, rectangle, layout, conversion and precision regression checks |
| `test_k.py`, `test_in*.py`, `test_out*.py` | Isolated shader diagnosis and reference comparisons |

`vulkan/ptx2glsl.py` and `vulkan/kvk.cpp` remain reference/research tools. They are not the engine backend.
