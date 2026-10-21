# Native weight-based inference engine

This directory implements standalone preset K, M and L inference engines. NGX is used **once to obtain weights
and reference data**. K loads extracted f16 matrices and a reconstruction table; M and L load FP8 matrices and
f16 vectors. All run d4r's own Vulkan shaders. M and L can run their network on d4r's native HIP kernels instead;
no engine translates PTX or launches NVIDIA kernels.

The reusable library is `libd4r_engine.so`; `d4r-k` and `d4r-m` provide full-frame replay and benchmarking.
The D3D12 game shim supports both as opt-in engine backends (see [In a game](#in-a-game-d3d12-shim)).
The CUDA/ZLUDA backend remains the default and fallback. Preset E and RTX/game-quality parity are not
implemented or established here (L and M are byte-identical to the CUDA backend running NVIDIA's translated kernels
in the harness; see [Preset L](#preset-l-runtime-and-shim-path-exist-experimental)). Exposure is supplied by the application or measured on the GPU.

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
  Window shifts and input resolution can change without rebuilding shaders or buffers. Output resolution changes require a new `Engine`.
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
K also runs without cooperative matrices (RDNA2): see [Preset K without cooperative matrices](#preset-k-without-cooperative-matrices-rdna2),
and M without cooperative matrices or FP8: see [Preset M without cooperative matrices or FP8](#preset-m-without-cooperative-matrices-or-fp8-rdna2).

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

## Models on the user's machine (install.sh)

A release's `install.sh` builds every model the shim loads, from the user's own DLLs, with
`d4r/tools/build-engine-models.sh` (`packaging/build_engine_models.sh`; `scripts/stage_engine_kit.sh` puts it and
d4r's compilers, shaders, the D3D12 harness and glslang into the release):

- Ray Reconstruction needs no capture. `nvngx_dlssd.dll` 310.9.1 holds presets D and E (the kernels and weights of
  310.7) and F. `engine/rr/build_native_model.sh --dll DLL D|E OUT` and `engine/compile_rr2.py --dll DLL OUT
  --ptx-dir DIR` take the PTX from the DLL (`kernels/tools/extract_dlss_ptx.py`), the weights as before
  (`dll_weights.py`, `rr2_pack.py`) and what the capture used to supply from the DLL too: the argument-block
  constants (`engine/rr/rr_templates.py`, `rr2_templates.py`, which replays the routine that builds NGX's model
  description), encoder 0's second table of preset F and its arena regions. Built this way, D, E and F are
  byte-identical to the capture-built models (every file, and `d4r-rr`'s outputs on the reference inputs).
- Super Resolution weights are packed inside `nvngx_dlss.dll`; they are read where NGX unpacks them. The harness
  renders one frame per preset (11, 13, 12) through the release's own d4r folder under GE-Proton with
  `D4R_CUDA_REPLAY_DUMP_DIR` set; `extract_k.py`, `prepare_k.py --lut-only` and `compile_k.py` / `compile_m.py`
  build K, M and L from that, and `--ldr` their LDR variants from the same frame (NGX's HDR and LDR stages read the
  same weights). About a minute per preset, ~280 MB of dumps that are deleted afterwards. The results are identical
  to the models built from full captures.
- On RDNA2 the builder makes E, F, K and M (L has no engine path there). M's and L's network kernels (`rrlite_enc0`
  ... `rrlite_dec0`) would be compiled by ZLUDA with emulated matrix products, minutes each, only for a frame whose
  launches compile_m reads: `model_tools.py stub-ptx DLL DIR` writes each of their modules with empty kernel bodies
  (same parameters and launch bounds) as `DIR/<FNV-1a 64 of the fatbin>.ptx`, and the test frame runs with
  `D4R_CUDA_REPLACE_DIR=DIR` (the bridge loads it in place of the fatbin); its pixels are then not checked. On the
  Deck the cold Super Resolution build with M takes about 150 s (M's frame about 30 s, over 10 minutes without the
  stubs); `D4R_ENGINE_STUB_NETWORK=0|1` forces either. The M model is byte-identical to one built from a real frame.
- The test frames run with `D4R_SHIM_GPU_INPUT_WAIT=0`: the replay dump synchronises and reads device memory at
  every launch, and behind the GPU-side input wait NGX's output then came out empty.
- NGX uploads every weight blob with `cuMemcpyHtoDAsync` from one reused host buffer and a `cuCtxSynchronize`
  before each reuse. CUDA's asynchronous copies from pageable memory return once the source is staged; HIP's may
  read it later, so with that synchronisation elided (`ElideNgxSync`) tensors arrived as zeros when the uploads
  ran back to back: M's input-stage weights with the stubs (every time on the Deck), and in every E/F capture the
  80-byte last head tensor of the reconstruction kernel. With the elision the bridge stages the source itself and
  frees the copy once an event behind the upload has completed.

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

The decoders skip the windows that only cover that padding (`kLayerGroups` in `runtime.cpp`). A window reads only
its own tokens (a decoder also the 4x4 low-resolution tokens under it and its encoder's skip tokens), so the
windows a frame's results depend on follow backwards from what the later stages read: the reconstruction reads the
head of the output's tokens, the next frame's input stage the token feature at most one token further (bilinear,
on-screen history only). The encoders all run: the deep layers' few windows mirror at the grid's edge and read
every token above them. 1280x720 has 180 of 256 token rows; dec0 then runs 24 of 33 window rows, dec1 13 of 17,
dec2 7 of 9, dec3 4 of 5. Every output is unchanged (final image, both histories, and the token feature and head
where anything reads them; the head and feature rows past 187 keep older values). Steam Deck, 640x360 -> 1280x720:
9.66 -> 8.84 ms per frame; RX 9070 XT, cooperative matrices: 713 -> 682 us. `D4R_ENGINE_K_TRIM=0` runs every
window. `test_frame.py` compares the token feature on the rows and columns the next frame can read.

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
| L, HDR flag set | `engine\l` | `D4R_ENGINE_MODEL_L_DIR` | `compile_m.py` on a capture of preset L |
| L, no HDR flag | `engine\l-ldr` | `D4R_ENGINE_MODEL_L_LDR_DIR` | `compile_m.py --ldr` on a capture of preset L |

The models the configuration will use are loaded at NGX init instead, each on a thread of its own (2026-10-20):
Ray Reconstruction's (the model a feature without a preset hint runs, preset F when `engine\rr-f` exists) when
`[RayReconstruction] Backend = dlss`, and the HDR model of the preset `[DLSS] Model` forces (K when `[Engine]` runs K,
L or M). On a cold shader cache (after a driver or model update) their pipelines take seconds to compile, and games
create their features long after they initialise NGX, so the first frame no longer waits for them. Only the models
load early (pipelines and weights: 8 MB for preset F, 2 MB for L); a feature still creates its own resources at its
first frame. `D4R_ENGINE_PRELOAD=0` loads on first use as before.

| D3D12 test app through the shim (`rr_backend_check.exe`, proton, 1280x720 -> 1440p) | Before | After |
|---|---|---|
| Ray Reconstruction preset F, cold cache, 8 s from NGX init to the feature: first frame | 3.05 s | 9.6 ms |
| Super Sampling preset L, the same | 0.89 s | 7.5 ms |
| Preset F / preset L, warm cache: first frame | 7.6 / 8.1 ms | 7.3 / 4.9 ms |

`scripts/install_d4r_runtime.sh` stages them from `D4R_ENGINE_MODEL`, `D4R_ENGINE_MODEL_LDR`, `D4R_ENGINE_MODEL_M`,
`D4R_ENGINE_MODEL_M_LDR`, `D4R_ENGINE_MODEL_L` and `D4R_ENGINE_MODEL_L_LDR`, copying each folder whole (with `hipnet.bin` and `hip/` from `--hip`). A model whose
`hipnet.bin` belongs to the other preset keeps the Vulkan network. The Vulkan network is the default since 2026-10-14;
`D4R_ENGINE_NETWORK=hip` (`[Engine] Network = hip`) selects the HIP network for both presets when the model has it; `D4R_SHIM_ENGINE_GPU_WAIT=0` makes the HIP network wait for its inputs on
the CPU-visible marker instead of a GPU-side wait, for comparisons.

With the engine enabled, an evaluation of preset K is recorded entirely into the game's own command list through vkd3d-proton's
`BeginVkCommandBufferInterop`: thirteen to fifteen dispatches (automatic exposure adds one or two), with direct
sampled inputs and a direct RGBA16F or RGB10A2 storage output when compatible. Unsupported usages/views keep their individual
input copies or output blit. There is no NGX
evaluation, CUDA launch, worker thread, frame marker, shared-memory import or split frame, and the game presents
the result of the frame it just rendered. The shim log says `engine backend: feature N runs preset K in the game's
command list`.

The shim hands a feature to the CUDA backend, with a log line giving the reason, when: the preset is not K, L or M;
the feature is preset L with display-resolution motion vectors (MVLowRes clear), which it does not implement yet, or M with
them and a package that predates `D4RM0005`; the
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
  not the display-resolution path. M takes display-resolution motion with a `D4RM0005` package (below); L still uses
  CUDA for it.
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

### Kernel-fusion attempts (2026-10-12)

Two dispatch boundaries were examined for fusion. Neither reproduced a gain; both variants were discarded and
the kernels are unchanged.

**dec5 phase fusion (preset K).** `kernels/k/dltss_pwin_dec5_layer.hip` was rewritten as one kernel: each block
owns one window and runs `phase_attn`, `phase_wo`, `phase_fc1` and `phase_fc2` in order with `block_sync()`
between them, keeping every phase's per-unit math and summation order (single-unit `ub` launches). `hip_net.cpp`
then takes its single-node path (`postCount == 0`), removing three dispatches per layer. The fused kernel
compiles at 127 VGPRs, 31,232 bytes of LDS and occupancy 4, against 115 VGPRs / 31,232 LDS / occupancy 4 for the
phase-split main kernel plus 48-93 VGPRs and **zero** LDS for each post kernel.

Outputs were byte-identical to the baseline across three captured 720p frames (head, output and all three
histories; only `timings.csv` differed). Three alternating baseline/candidate profiling runs, 25 frames per batch:

| Run | Baseline whole-frame median | Fused whole-frame median |
|---|---:|---:|
| 1 | 513.7 us | 611.1 us |
| 2 | 523.9 us | 612.2 us |
| 3 | 527.5 us | 613.9 us |

The fused kernel is about 90 us per frame **slower**: it holds the 31 KB window scratch across all four phases,
so the whole layer runs at occupancy 4, while the post kernels run at higher occupancy with no LDS. The saved
dispatches do not pay for the lost occupancy. It also runs only one block per window: dec5 has 4 windows at
720p, so 4 blocks on 64 CUs - the idleness the phase split exists to avoid (`pwin_phase.h`).

A second variant fixed the parallelism: a grid-stride fused kernel on a fixed 96-block grid (the baseline's
phase grids), keeping the baseline's unit -> block assignment exactly, with a software grid barrier (arrival
counter plus generation flip, `__threadfence` around it) where the dispatch boundaries were. `hip_net.cpp`
honors a kernel's declared `d4r_grid_x` in its single-node path. Outputs were again byte-identical over three
captured frames. Three alternating runs, 25 frames per batch: baseline 529.8 / 530.6 / 528.3 us, fused 519.3 /
532.2 / 525.4 us - no reproducible gain, and the first frame costs 9.9 ms instead of 0.97 ms (the barrier spins
until every block is resident). Both variants discarded; the phase-split kernel is retained.

**Exposure fold into `enc0_m.comp` (preset M).** Folding the two `cuda_dldn_engine_*` reduction passes into the
input stage would remove two dispatches and the `expSums` round-trip, but the reduction must finish before any
token is embedded: the fold needs a grid-wide barrier over the input stage's whole block grid (3726 blocks at
720p). Vulkan exposes no cooperative dispatch on this device (no grid-sync extension; `maxComputeWorkGroupInvocations`
is 1024), so the fold is not implementable in the engine's own stages. The measured cost it would save is small
anyway: on the captured 640x360 -> 1280x720 sequence (`d4r-m`, 10 repeats, Vulkan network) the exposure stage
medians 6.3 us against 26.9 us input, 598.6 us network, 21.4 us expansion, 83.0 us reconstruction and 22.6 us
downsample. Not attempted.

**Inter-layer barriers in the M network loop (2026-10-12).** The ten layer dispatches in `runtime_m.inc` were
separated by a full `vkCmdPipelineBarrier` with a global `memoryBarrier` flush. Only dispatch serialization is
required there: each layer's cross-block traffic goes through its own buffers, and the preceding stage's flush
already published them. Replacing the layer-loop barrier with an empty pipeline barrier (compute→compute, no
memory barriers) kept every stage byte-identical to the captured launches - `test_m.py --chain` reports 10/10
layers `identical 100.0%`, `test_frame_m.py` reports inf dB, and 8- and 16-frame Balanced outputs matched the
HEAD outputs byte-for-byte - while removing the flush cost: network median 1863.0 → 1765.6 µs, whole-frame
median 2515.0 → 2404.6 µs at 1488×837 → 2560×1440 (Quality 3818.9 → 3782.3, Performance 1892.6 → 1829.8, both
byte-identical). Removing the barriers entirely breaks frames 0-3; the full flush is retained at the
front→network, expand→post, post→down and exposure sites, where empty barriers measured within noise.

**Register-only half rounding in `layer_m.comp` (2026-10-12).** Replacing the LDS round-trip (`shared f16vec4
RS[128 * 2]` plus `subgroupMemoryBarrierShared`) with `roundHalfValue`'s integer bit arithmetic is byte-identical
in the standalone chain test (10/10 layers 100.0%) and neutral there, but pathological in-engine: network median
5658-5684 µs against 1798.8 µs for the HEAD shaders in the same package/runtime, across every layer (per-layer
skip deltas confirm the regression is uniform). The per-component rounding chain between dependent `K32H`
accumulators defeats latency hiding at Balanced wave counts; a branchless variant measured 4990 µs. The LDS
round-trip is the fast form and is retained. Rounding itself costs about 95 µs network-wide (NO_ROUND floor
1689-1711 µs); gelu about 10 µs; attention about 55 µs of enc1/dec1.

**Dead levers (2026-10-12).** At Balanced the compact reconstruction tier is already eligible and selected;
forcing post/quality/compact measured 321.3/340.7/325.6 µs - within noise. Weight traffic is cache-hot: global
versus LDS `coopMatLoad` measured neutral (249.8 vs 272.6 µs/dispatch). gfx1201 coopmat exposes only M16 N16 K16
(no K32), and RADV offers no VGPR-cap knob; enc1/enc2 sit at 192 VGPRs (2 workgroups/CU) with zero spills.

**Paired LDS fences in `layer_m.comp` (2026-10-12).** Batching two independent `K32H` chains into one
`subgroupMemoryBarrierShared` (`K32HP`/`R16P` over a `RS[128 * 4]` scratch) is byte-identical (chain test 10/10
100.0%) and halves the fence count, but it keeps both accumulators live across the fence: enc1/dec1 rise from
192 to 240 VGPRs, dropping to 2 workgroups/CU with worse scheduling - network median 1869.4 µs against 1765.6
for the unpaired form. Register pressure dominates fence count on this device; the unpaired macros are retained.

**Other `layer_m.comp` probes (2026-10-12), all discarded.** Rounding through `coopMatStore`/`coopMatLoad` on an
f16 LDS tile is not byte-identical (enc1 98.0%, tubes 12-33% identical): the HAC round trip does not reproduce the
f32->f16->f32 rounding, so the scalar `RS` scratch is required. Staging `h1` through wave-private LDS to free its
tiles across attention/MLP is byte-identical but raises enc1/dec1 to 240-256 VGPRs (RADV keeps the reloaded tiles
live plus the addressing) - 1892.3 µs. Splitting the attention loop into a q phase and a v phase, shortening the
`QV_G` weight-tile live ranges, and staging `v` through `VSH` right after the QV loop are each byte-identical and
leave enc1 at 192 VGPRs; all measured 1781-1807 µs, within noise of the 1765-1797 µs baseline. enc1/dec1 stay at
192 VGPRs (2 workgroups/CU) under every restructuring tried; the matmul-only floor (rounding, gelu and attention
removed) is about 1625 µs network, so no byte-identical engine restructuring reaches a 30% whole-frame gain.

**Non-network stage probes (2026-10-12), all discarded.** Staging the input stage's 10x10 depth/motion
neighbourhood into LDS (`enc0_m.comp` + `enc0_features.glsl` accessors) is byte-identical (tokens and all three
motion surfaces 100.0%) but slower: the twelve `texelFetch`es per pixel were already cache-hot, and the extra
barrier plus the LDS round-trip cost more than they saved - input stage 144.6-150.1 us against 124.5. Splitting
`enc0_m.comp`'s four independent K32 chains into two sequential passes (two accumulators live at a time) is
byte-identical and leaves the stage at 240 VGPRs; input 122.0-124.8 us, within noise. Padding the reconstruction's
shared render-texel tile rows (`TSTRIDE = TILE_W + 2` in `post_m.comp`, addressing only) is byte-identical
(history and packed output 100.0%) at 322.0-323.7 us against 324.5 - noise. The stages' texture traffic is
already coherent; none of these stages has a byte-identical restructuring that measures.

**Intra-frame barrier relaxation (2026-10-12), discarded.** Replacing the `nextStage` flushes at the exposure,
input-stage, expansion and reconstruction sites with empty pipeline barriers is byte-identical (8-frame outputs
match) and shrinks the measured stage spans (exposure 16.6, input 120.5, reconstruction 323.1), but the whole-frame
median is unchanged (2431.8 against 2431.7): dispatch pipelining already hides those flushes; only the ten
layer-loop barriers cost. The full flushes are retained.

**`colSum` subgroup shuffle in `post_m.comp` (2026-10-12), discarded.** The half-resolution mean's neighbour-column
sum can come from `subgroupShuffleXor` instead of the `colSum` LDS round-trip plus barrier (the reference itself
uses a `shfl.sync.bfly` there). Byte-identical (8- and 16-frame outputs match, post stage test 100.0%), but eight
GPU-locked alternating runs against the retained build measure 2446.6 against 2442.8 us whole-frame median - within
noise. The LDS round-trip is retained.

**Downsample block height and expansion LDS stride (2026-10-12), discarded.** Halving `down_m.comp`'s `BLOCK_Y`
(64 -> 32, with the matching dispatch grid) is byte-identical (8-frame outputs match) but measures 2441.7 against
2436.0 us whole-frame median over eight GPU-locked alternating runs - the smaller tile halves the LDS per workgroup
but doubles the workgroup count and the tile-fill redundancy. Padding the expansion's `EO` row stride (22 -> 24 in
`dec0_m.comp`) to cut LDS bank conflicts in the K32 rounding round-trip is byte-identical (8-frame outputs match)
at 2448.8 against 2438.9 us - noise. Both reverted.

**Rounding-free ceiling (2026-10-12).** Six GPU-locked alternating runs of the NO_ROUND model (per-K32 binary16
rounding removed - the most aggressive engine-only change, and one that *breaks* byte-identity) measure network
1629-1654 us and whole-frame 2267-2290 us against the banked build's 2437-2470 us: a 7.0% whole-frame gain. Even
abandoning the captured-launch accuracy reference entirely, no engine-only path reaches 30%; the frame is bounded
by the matmul itself.

**Cross-frame pipelining (2026-10-12), not implemented.** The frame dependency chain is `enc0 -> network ->
expansion -> post -> downsample`, and the next frame's `enc0` reads the `half`/`feat`/`mvA` surfaces the current
frame's `post`/`expansion` write, so only the downsample tail (93 us at Balanced) could hide behind the next
frame's front and network - 3.8% at best, and unsafe as recorded: `down_m.comp` reads the current frame's colour
texture alpha, which the next frame's input upload overwrites before the pipelined dispatch would run. Hiding
every non-network stage behind the network (the 652 us stage budget, 26.8%) is arithmetically blocked by the same
chain and would in any case compete with the compute-bound network for the same CUs.

## Dynamic input resolution (K, L and M)

The engine accepts changing render subrect dimensions with a fixed output size. K keeps its output-sized
network and temporal history, updates the frame's motion conversion and reconstruction table, and reserves
conversion inputs up to the output size. Use a K package with all sixteen reconstruction tables for changing
upscaling ratios; older single-table packages cannot match NGX at every ratio.

L and M update the Vulkan network's active token grid, exposure grid and reconstruction tier per frame.
Their input and history planes reserve the constructor's render capacity. The shim reserves the render size the
feature was created with (or the first frame's, when larger), as NGX does; a later frame beyond that goes to a second,
output-sized Vulkan engine, and the first is kept until the feature is released. Reserving the output size for every
feature cost 0.85 GB (M) and 1.08 GB (L) at 1280x720 -> 4K; the images are byte-identical either way (harness,
2026-10-14: created 1280x720 with 854x480 frames, created 854x480 with 1280x720 frames, and the second engine
forced). Bilinear reads clamp to the active rectangle. A render-size change resets
L/M temporal history. Input dimensions must be at least 64 on each axis and cannot exceed the output.
Standalone callers set `GameFrameM::renderWidth/renderHeight` or `MFrameSettings::renderWidth/renderHeight`;
zero keeps the constructor size.

L/M's HIP network retains its fixed constructor grid. When a feature renders at another size, the shim uses
one Vulkan engine for the changing sizes and retains the original HIP engine and shared buffers until queued
command lists retire. Returning to the original size resumes HIP with reset history. This uses additional
VRAM once the Vulkan engine is allocated. K's HIP network remains usable across all input-size changes.

Rebuild L and M packages with the current `compile_m.py` (`D4RL0002` / `D4RM0004`). Earlier packages retain fixed-size support but lack
active-rectangle sampling, so a size change uses the CUDA fallback. The dynamic regression can be run independently:

```sh
build/native-engine/test-game --dynamic-ml PATH_TO_M_OR_L_MODEL
```

On the RX 9070 XT, six-frame HDR sequences for both L and M pass shrinking/growing input sizes,
Quality/Performance tier changes and queued direct/copy comparisons. Direct and copied inputs are byte-identical,
and implicit resize resets match explicit resets. Increasing allocation capacity changes a few recurrent pixels
through normalized sampling and FP8 rounding: RMSE is 7.29e-6 (M) and 8.12e-6 (L), within the test's
5e-5 RMSE and 1% per-channel error limits. This is an adapter regression, not an in-game or NGX parity claim.

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
| downsample | `down_m.comp` | byte-identical (2026-10-11; was 70-71 dB, 92%: its weights, see L_NOTES.md) |

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

M packages now use `D4RM0004` (L: `D4RL0002`). Rebuild with the current `compile_m.py`: the blocked FP8 operands,
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

### This GPU's own arithmetic for the network: `M_RELAXED` (2026-10-14)

`M_DEFS=-DM_RELAXED python3 engine/compile_m.py CAPTURE OUT` builds `layer_m.comp` (presets M and L) without NVIDIA's
roundings: the accumulator stays f32 across the 32-deep steps (no scratch round trip), values go from f32 into FP8 two
per saturating conversion, and the activation is computed in f32. The default build is unchanged (first frame
byte-identical to the CUDA path through ZLUDA). Against NVIDIA's kernels on an RTX PRO 4500 Blackwell (captures
capHM / capHL: harness create flags 0x4B, 640x360 -> 1280x720, three frames; `build/rr/f/nv/srm.sh`, `srt4k.sh`):

| | M, dB | M at 1280x720 -> 4K | L, dB | L at 1280x720 -> 4K |
|---|---|---|---|---|
| default | 38.4 / 38.2 / 37.9 | 3.19 ms (network 1.42) | 40.8 / 40.9 / 40.9 | 4.47 ms (network 1.46, input 0.67, expansion 0.54) |
| `M_RELAXED` | 37.6 / 37.2 / 37.2 | 2.90 ms (network 1.13) | 40.1 / 39.8 / 39.8 | 3.83 ms (network 1.13, input 0.50, expansion 0.41) |

`M_WPW=2` (2026-10-14, `layer_m.comp` `WPW`) adds two 4x4-token windows per wave in the five network layers: a
workgroup is still one 8x8-token block, but its two waves each walk two windows and load every weight tile once for
both (a window reads the whole layer's weights; with every weight tile replaced by one tile loaded once the network
took 0.79 ms instead of 1.13). The default build with `M_WPW=2` is still byte-identical to the capture, on three
reset frames and on a ten-frame temporal capture (capTM), so the restructuring itself changes nothing; it only pays
together with `M_RELAXED` (the exact build's scratch round trips hide it: 1.41 ms network against 1.38). Four windows
per wave, or two in the 128-channel layers without spilling, would need fewer registers than the residual stream
takes (two windows: 256 vector registers, 119 spilled values in the tube layers, none in the others).

| 1280x720 -> 4K | M frame | M network | L frame | L network |
|---|---|---|---|---|
| default | 3.16 ms | 1.38 | 4.44 ms | 1.42 |
| `M_RELAXED` | 2.86 ms | 1.10 | 3.81 ms | 1.11 |
| `M_RELAXED`, `M_WPW=2` | 2.68 ms | 0.93 | 3.67 ms | 0.98 |
| the native HIP network (what `[Engine] Network = auto` picks when the model has `hipnet.bin`) | 2.90 ms | 1.08 | | |

`M_RELAXED M_WPW=2` against the NVIDIA card: M 37.6 / 37.3 / 37.2 dB, L 40.1 / 39.9 / 39.8 dB (default: 38.4 / 38.2 /
37.9 and 40.8 / 40.9 / 40.9). On the temporal capture the relaxed build settles at 34-35 dB from the ZLUDA
reference after four frames. Replay notes: the D3D12 harness resets every frame unless `D4R_HARNESS_TEMPORAL=1`, so
its captures are replayed with `D4R_M_RESET_ALL=1 d4r-m ...` (an earlier comparison without it looked like a
temporal drift of the engine; there is none). `M_DEFS_ENC1`, `M_DEFS_ENC3_TUBE`, ... pass defines to one layer.
Preset K (`layer.comp`) already accumulates in f32 and is 59.9 dB from the same card (the ZLUDA capture: 49.5 dB),
so there is no rounding to relax. Its four waves per window each load the layer's weights, and with every weight
tile replaced by one tile loaded once the eleven layers take 1.16 ms instead of 1.51 at 1280x720 -> 4K, so
`layer.comp` got the same restructuring (`TPW`: 16-token tiles per wave; `K_DEFS`, `K_DEFS_<LAYER>`, `K_TPW`). The
output is byte-identical at one, two and four tiles per wave (15 files of a three-frame capture), but K's operands
are binary16 (eight registers a tile where FP8 takes two): two tiles per wave only fit in the 32-channel layers
(enc0 277 -> 272 us, dec0 unchanged), four tiles spill there too (enc0 352 us), and in the wider layers two tiles
spill 15 to 568 values and the frame takes 3.16 ms. `compile_k.py` therefore builds two tiles per wave in the
32-channel layers only: 2.86 -> 2.81 ms. The native HIP network measures 2.83 - 2.85 ms on the same frames.

### Defaults since 2026-10-14

Every model builder now makes the fastest build measured above unless told otherwise, and the shim runs the network
on the Vulkan shaders unless `[Engine] Network = hip`:

| model | default build | the exact build (NVIDIA's roundings, bit-identical to the CUDA path) |
|---|---|---|
| K (`compile_k.py`) | two tiles per wave in the 32-channel layers (same output) | `K_TPW=1` for the previous layout |
| M, L (`compile_m.py`) | `M_RELAXED`, two windows per wave, `M_RESIDUAL_F32` | `M_EXACT=1` (implies `M_RESIDUAL_F32=0`) |
| RR D, E (`rr/build_native_model.sh`) | `D4R_RR_RELAXED=3` | `D4R_RR_RELAXED=0` |
| RR F (`compile_rr2.py`) | `D4R_RR_RELAXED=3`, `D4R_RR2_UNGUARDED=1`, `D4R_RR2_POOLED=rgb10_a2` | `D4R_RR_RELAXED=0 D4R_RR2_POOLED=rgba32f` |

When these defaults were introduced (2026-10-14), a model built with no options was compared shader by shader with
the explicitly optioned build of each row, and the exact switches still reproduce the captures
(`build/rr/f/nv/defaults.sh`).

### Byte-identical speed-ups (2026-10-19)

RX 9070, RADV 26.2.2, `d4r-m` 1280x720 -> 2560x1440 with `D4R_ENGINE_M_LAYER_STAMPS=1`. Every output frame, and
everything `D4R_M_DUMP` writes, is byte-identical to the previous build (M and L, HDR and LDR, the three
reconstruction tile tiers and the stress inputs).

| change | stage | before -> after (us) |
|---|---|---|
| `layer_m.comp`: hardware FP8 saturation where a value cannot be infinite, and the V tiles in a buffer that is dead during the attention (more waves per SIMD) | frame | M 1583 -> 1566, L 2349 -> 2293 |
| `post_m.comp`: the second history candidate reuses the first one's sample when both have the same vector | reconstruction, uniform motion | 325 -> 288 |
| `post_m.comp`: both rows of a pixel pair take their Gaussian taps in one pass | reconstruction | 287 -> 265 (L 355 -> 320) |
| `down_m.comp`: the horizontal pass reads its texels into registers, no shared source tile | downsample | L 102 -> 91, M 117 -> 113 |
| `layer_m.comp`, `compile_m.py`: L's expansion stage at two windows per wave (`M_DEFS_DEC0L=-DWPW=1` for one) | expansion | 383 -> 366 |

The network layers are separated by an execution-only barrier (`emptyStage`). A memory barrier there gives the same
output and costs about 60 us (M 1520 -> 1592), so it stays.

### `M_RESIDUAL_F32` (default since 2026-10-20)

Every relaxed build keeps the residual stream in f32 across the MLP's 32-channel chunks
and rounds it to binary16 once after them, instead of after every chunk; `M_RESIDUAL_F32=0` builds the previous
form. It needs the relaxed build (`M_EXACT=1` implies `M_RESIDUAL_F32=0`). 1280x720 -> 1440p: M 1521 -> 1410 us,
L 2195 -> 2077. Against the previous relaxed build, final RGB, peak 1: 43 - 46 dB (L 45 - 49) on smooth frames,
17 - 18 dB on noise. The exact build is as far from the previous relaxed build, so this moves no further from
NVIDIA's arithmetic than `M_RELAXED` already does. Not checked against an NVIDIA card.

## Preset L (runtime and shim path exist; experimental)

Preset L (render preset 12, NVIDIA's model for Ultra Performance) is M's pipeline with its own input, expansion and
reconstruction stages: a C 32 Swin block at render-pixel resolution behind an embedding (input) and in front of the
per-pixel head (expansion), where M has single folded products, and a reconstruction with 4x4-texel Gaussians and a
history blended in linear light. `L_NOTES.md` has the algorithms. The network is M's ten layers with L's weights.

The runtime is M's (`MModel`, `MEngine`, `GameUpscalerM`, `d4r-m`): `compile_m.py` recognises a capture of preset L by
its unfolded kernels and writes an L model directory (`offsets.bin` `D4RL0002`: the input and expansion stages as
layer blocks of `layer_m.comp` KIND 3 and 4, and L's five constants), which `MModel` loads as preset L
(`isPresetL()`). The shim runs preset 12 with the model in `engine\l` / `engine\l-ldr`; the log says `engine backend
(L): feature N runs preset L in the game's command list`.

```sh
# capture one frame of preset L with the CUDA backend (D4R_DLSS_PRESET=12; replay filter rrlite_), then
python3 engine/compile_m.py "$CAP" build/l-model                 # --ldr for a capture without the HDR flag
python3 engine/test_l.py enc0 "$CAP" 0 build/l-work vulkan/kvk   # also: dec0; test_post_m.py with POST_KERNEL
```

Every stage is byte-identical to NVIDIA's kernels as ZLUDA translates them on every captured frame (640x360 ->
1920x1080 HDR and LDR, 1280x720 -> 1920x1080); `d4r-m` on the captured frames and the D3D12 harness through the shim
are byte-identical to the CUDA backend except for isolated single-frame knife edges in three of 20 compared frames
(96-122 dB, gone the next frame). 1280x720 -> 3840x2160 takes 3.8 ms on an RX 9070 XT (3.6 ms with `--hip`).
Building L showed four engine issues that M shared, all fixed: the encoders' merge clamp, the exposure grid at
Ultra Performance, the downsample weights and the network's summation order (M is now byte-identical to the CUDA
backend in the harness as well: 1706x960 -> 2560x1440 and 640x360 -> 1920x1080, where it measured 36-41 dB with the
native accuracy kernels' order; `compile_m.py` builds both presets' networks with `TEXL`). Details: `L_NOTES.md`.

### The input stage in less shared memory (2026-10-20, night)

RX 9070 (56 compute units), `d4r-m` 1280x720 -> 2560x1440, median of interleaved runs. `layer_m.comp` KIND 3 keeps
the features and the merge rows inside x0's binary16 array (FP8 matrices at half the offsets and strides, feature
bytes stored in pairs; the merge rows and V tiles take x0's place after a barrier behind norm 1): 12 KiB of shared
memory become 7.5. That makes two windows per wave pay for the input stage, which `compile_m.py` now builds that way
(`M_DEFS_ENC0L=-DWPW=1` for one; at 12 KiB it ran five waves per SIMD and was slower, now eight).

| Build | Input stage (us) | Frame (us) |
|---|---|---|
| L, `M_RESIDUAL_F32=1` | 446 -> 414 | 2033 -> 1995 |
| L | 447 -> 414 | 2157 -> 2115 |
| L, stress input | 446 -> 419 | 2104 -> 2078 |

Every output frame and everything `D4R_M_DUMP` writes is byte-identical to the previous build (L and L-LDR, with and
without `M_RESIDUAL_F32`, `M_EXACT=1`; replay, odd-size and stress inputs); preset M's shaders are unchanged files.
Feature bytes stored in the wrong order change every frame.

Tried, no gain: the expansion stage's FP8 buffers inside its head buffer (9 -> 5 KiB, 7 -> 12 waves per SIMD:
expansion 355 -> 358 us; one window per wave there, 381). Both stages are bound by their own VALU issue (input stage
2961 instructions per wave, 1977 VALU of which 548 conversions; expansion 3383 and 2471), not by latency, so more
waves do not help them. The network layers' next MLP chunk loaded during the current one, by hand (what a compiler
hoisting loads across the loop's back-edge would do): the next chunk's biases and first tiles, every layer slower
(network 803 -> 833 us); all its tiles and biases, at the same registers and waves, enc1 and dec1 181 -> 204 us.

## Presets E and F: DLSS 3's CNN (runtime and shim path exist; experimental, the RDNA2 route)

Presets E (5) and F (6) are NGX 310.7's CNN upscaler, in the `nvngx_dlss.dll` the installer already fetches. Its
frame is an input kernel (`cuda_engine_input_kernel_rel_{ldr,hdr}_mvdiff_{mvhi,mvlo}`: the game's colour, depth and
motion, the previous output and motion to 16 channels per token of 4x4 output pixels, the warped history and this
frame's output-size motion), eleven `dltss_nchw8_*` network kernels (a U-Net over the token grid, the output padded
to 128 / 4, five levels) and a reconstruction kernel (`..._aniso_gaussian_dfn_{fastpath_3x3_128, mid_3x3_128,
3x3_256, fastpath_5x5_256, 5x5_256}`: the last decoder level, a 40-channel head per token and the anisotropic
Gaussian reconstruction of the output and the next history). E and F share the kernels; they differ in weights and
in two constants.

- The network is d4r's own `sr/e_net.comp` (five modes: the first encoder, the 3x3 + pool encoders, 1x1 convs, the
  upsampling 1x1, the decoders' double 1x1), packed binary16 like the RR CNN's packed flavor (`rr/cnn_pk.glsl`): a
  lane owns a pixel, the weights are wave-uniform scalar loads feeding `v_pk_fma_f16`, the input of a workgroup is
  staged in LDS. No cooperative matrices, so it is the engine's route on RDNA2. It writes NGX's tensor layouts
  (`[C/8][H][W][8]`, the full-size skips as 2x2 phases, 8-bit activations as binary16 bits 14..7), which the
  translated reconstruction kernel reads.
- The input and reconstruction kernels are translated from the user's DLL (`rr/ptx2rr.py`) in every variant; the
  runtime picks a variant for the ratio (5x5 above 2.5x, `fastpath_3x3` from 1.6x, `mid` when each render axis is
  floor/ceil of two thirds of its output axis, the generic kernels otherwise) and rewrites every frame-dependent
  field of their argument blocks (sizes, ratios,
  jitter, reset, MV scale, the HDR and MVLowRes flags; harness captures at seven sizes and ratios, 2026-10-18).
- `compile_sr_e.py REPLAY NVNGX_DLSS_DLL OUT --preset E|F` builds a model from one frame's replay dump (the weights are
  packed in the DLL; NGX uploads them unpacked), `runtime_e.inc` is `EModel` / `EEngine`, `game_e.cpp` the game
  adapter, `replay_e.cpp` the replay (`d4r-e`). The shim runs presets 5 and 6 with `engine\e` / `engine\f`
  (`D4R_ENGINE_MODEL_E_DIR`, `_F_DIR`), LDR or HDR, either motion resolution, any render size (a new size rebuilds
  the engine), the game's exposure texture or AutoExposure. `build-engine-models.sh` builds both.
- Exposure (HDR; harness captures with per-frame pre-exposure, exposure scale and exposure texture schedules,
  2026-10-19). Both translated kernels take this frame's exposure scale (twice), pre-exposure and 1 / pre-exposure
  (input words 244-256, reconstruction 592-604). They sample the exposure as 1x1 textures: the input kernel's 344 and
  360 and the reconstruction's 472 and 480 this frame's, the input kernel's 352 the previous frame's, which NGX's
  `cuda_copy_exposure_kernel` copies into a half surface after each frame (the current one on a reset frame):
  `sr/exposure_copy_e.comp`. With AutoExposure NGX runs K's measurement (`cuda_luma_convert_kernel`, three
  `cuda_reduce_sum_kernel`, `cuda_auto_exposure_copy_kernel`: key 0.18 / 0.82 times exp(-mean log luma), the luma
  scaled by 1 / pre-exposure, truncated to half) on every second frame from the first, on M's grid (half the output,
  half the render size but at least 256 above 2x), and ignores the game's texture: `exposure_k.comp` MODE 3. The
  engine used to leave all of this at the model dump's values (exposure 1, the current exposure as the previous one):
  at 1280x720 -> 1920x1080 with pre-exposure 4/2/8 and scale 0.5/0.75/1.25 it measured 39/18/29 dB against NGX with
  the game's texture, now 64/55/54 dB; with AutoExposure 65/54/54 dB, through the shim as through `d4r-e` (neutral
  exposure: 64-66 dB, unchanged). Models built before this keep the old binding and refuse AutoExposure.
- The reconstruction's jitter words 544-556 are the previous and this frame's jitter as (y, x). The runtime wrote
  (-x, -y), which agrees only on the harness's first three jitters (0, -+0.25), so three-frame checks passed while
  later frames of the jitter sequence measured 26-28 dB (every game's Halton jitter was affected). Twelve-frame
  captures at 640x360 -> 1280x720 now measure 60-66 dB (HDR), 61-66 dB (LDR), and with AutoExposure at
  960x540 -> 1920x1080 and pre-exposure 19-21 60-66 dB; on the Steam Deck through the shim 51-67 dB against ZLUDA.

With the f32 reconstruction head, accuracy against NGX through ZLUDA (three temporal frames): 63-68 dB at 640x360 -> 1280x720
(LDR and HDR, both motion resolutions, E and F), 1280x720 -> 1920x1080, 1001x563 -> 1501x845 (generic kernel),
427x240 -> 1280x720 (5x5), 742x418 -> 1280x720; DLAA 57-73 dB. Network tensors 45-60 dB each (binary16 accumulation).
The translation needed two `vulkan/ptx2glsl.py` fixes that apply to every translated kernel: `.relu` on f16 / f16x2
`fma`, `add` and `mul` was ignored, and `add.rz` was always taken for the `roundf` idiom (CUDA's `log1pf` starts with
an `add.rz`, which lost its second operand).

Steam Deck (gfx1033, RADV, best of 12), 640x360 -> 1280x720: 4.3 ms (input 0.78, network 1.86, reconstruction 1.68);
1280x720 -> 1920x1080: 10.1 ms. The network's ideal at the Deck's packed-f16 rate is 0.75 ms at 720p; the two
translated kernels are the larger share (the reconstruction is 5400 VALU instructions as translated, its emulated
mma about 0.3 ms of it), so native input and reconstruction kernels are the next step.

RDNA2 tuning (2026-10-18): the SR decoders now reuse the input's padded LDS rows for their hidden layer,
with a workgroup barrier before overwriting the input. This removes a second allocation and its bank conflicts.
RR's packed CNN uses smaller tiles on the stages limited by LDS occupancy; its decoder filters cache each bilinear
sample once per tile, retaining the original half-precision operation order. Rebuild the model directories with
`compile_sr_e.py` / `compile_rr3.py` to use these shaders; the existing runtime selects the packed RR flavor on RDNA2.

Steam Deck gfx1033, RADV, GPU fixed temporarily at its stock 1600 MHz, interleaved baseline/candidate runs,
best of 12 submissions per frame (four temporal frames for RR, three for SR), 640x360 -> 1280x720:

| Model | Before tuning, ms | After tuning, ms |
|---|---:|---:|
| SR E, complete frame | 3.13 | 3.09 |
| SR E, network | 1.24 | 1.20 |
| RR A, complete frame | 14.61 | 12.48 |
| RR A, input + network | 12.98 | 10.85 |

These are inference timings without output readback, with no other GPU workload; automatic clocks and games can
run slower. The changes preserve the packed path's arithmetic: all outputs and histories match byte for byte at
one submission per frame and in repeated runs. Additional checks cover RR's partially filled tiles (70x66 render),
and SR's HDR, output-resolution motion, odd sizes (1001x563 -> 1501x845), 1080p output and 5x5 reconstruction.
The packed convolutions still differ from NVIDIA's matrix accumulation as described above. The translated SR
input/reconstruction passes still take most of the frame, so these results do not establish hardware-optimal
performance. `tests/test_cnn_packed_gpu.py` checks decoder reuse and partial pixel sets against a CPU reference
without proprietary weights (`D4R_RUN_GPU_TESTS=1 D4R_EVK=/path/to/evk python3 -m unittest tests.test_cnn_packed_gpu`).

The reconstruction head now defaults to packed binary16 accumulation too (`D4R_MMA_PACKED` in
`cnn_sr/mma_f16.glsl`, enabled only for the SR model builder). Each lane computes two columns together with
`v_pk_fma_f16`, retaining the fragment layout and K order without expanding operands and accumulators to f32.
The translated stages also use aligned 32-bit arena accesses for packed words. This removes pairs of scalar
16-bit loads/stores; their binding and tensor layouts stay the same. Formatted image stores expand at the call
site, retaining the concrete SPIR-V image format instead of passing it through an unformatted function parameter.
All four input and five reconstruction variants pass `spirv-val`.

On the same Deck at a fixed stock 1600 MHz, best of 12 submissions per frame, the complete E pass with the
previous f32 head versus the new packed head (three-frame input sequence):

| Render -> output | f32 head, ms | Packed head, ms |
|---|---:|---:|
| 640x360 -> 1280x720, LDR | 3.09 | 2.55 |
| 640x360 -> 1280x720, HDR | 3.09 | 2.66 |
| 1001x563 -> 1501x845, generic reconstruction | 6.28 | 5.30 |
| 1280x720 -> 1920x1080 | 7.15 | 6.20 |
| 427x240 -> 1280x720, 5x5 reconstruction | 5.10 | 4.02 |

At 640x360 -> 1280x720 the reconstruction pass itself falls from about 1.35 to 0.80 ms. The packed head rounds
after each half FMA, so its image is **not byte-identical** to the previous head. At one submission per frame,
eight tested configurations (also HDR, both motion resolutions, balanced scaling and DLAA) give 64.9-92.6 dB
RGB PSNR against the f32-head engine over three temporal frames, with finite outputs throughout. These comparisons
measure the arithmetic change, not quality parity with NGX or a long gameplay sequence. `--head-accumulation f32`
retains the previous head arithmetic, with the word-access optimization; the default is `f16` for both E and F.
`tests/test_cnn_mma_gpu.py` checks both accumulation modes, fragment mapping and independent concurrent waves
against CPU references. Rebuild model shaders to pick up the new default; old model directories continue to run.

Rounded Quality dimensions also use the existing `mid_3x3_128` reconstruction kernel: each render axis must be
floor or ceil of two thirds of its output axis. The kernel still samples with the actual ratios and jitter.
This avoids the generic reconstruction fallback for 853x533 -> 1280x800 without changing the model arithmetic.
On the Deck at stock 1600 MHz, three interleaved before/after runs (120 temporal frames each, 30 warm-up frames
discarded, 270 measured samples per build) reduce median reconstruction from 2.44 to 1.23 ms and the complete
upscaling pass from 4.46 to 3.26 ms. This change only needs a runtime rebuild; existing model shaders work.
`tests/test_sr_reconstruction_gpu.py` compares automatic selection against forced generic reconstruction, including
mixed per-axis rounding, partial tiles, HDR/LDR, both motion resolutions, jitter, history and reset. On the Deck,
all 384 preset E test frames match bit for bit; an optional private preset F model enables the same checks for F.

```sh
python3 engine/compile_sr_e.py "$CAP/replay" nvngx_dlss.dll build/e-model --preset E
build/native-engine/d4r-e build/e-model INPUT_DIR OUT_DIR    # D4R_ENGINE_E_PROFILE=1: microseconds per pass
```

## Preset K without cooperative matrices (RDNA2)

RDNA2 (the Steam Deck's VanGogh, gfx103x) has no matrix instructions and RADV does not expose
`VK_KHR_cooperative_matrix` there, so `layer.comp` cannot run. `layer_dot2.comp` is the same Swin block (same
bindings, parameter block, build-time shape and roundings to binary16) on `v_dot2_f32_f16`: binary16 pairs, f32
accumulation, at the full packed rate of the GPU. Only the order of the f32 sums differs from `layer.comp`.

- GLSL reaches the instruction through `SPV_VALVE_mixed_float_dot_product` (`spirv_instruction`, `OpFDot2MixAcc32VALVE`);
  RADV lowers it to `v_dot2_f32_f16`, whose weight operand can be a scalar register.
- One wave32 per attention head (twice the heads, at most eight, in the deep layers C >= 96, which have too few
  windows to fill the GPU otherwise). A lane owns two tokens of the 8x8 window (lane and lane + 32), so every
  wave spans the window and each weight word feeds two products. The products with weights take the weights from
  scalar memory (the waves split the output channels); the activations are rows in shared memory.
- The weights are the model's own matrices with each pair of K rows interleaved (one 32-bit word holds W[2i][n]
  and W[2i+1][n]). The runtime does that when it loads `weights.bin`; every tensor keeps its offset, so a model
  directory serves both networks. `compile_k.py` writes the shaders as `<layer>_dot2.spv` next to `<layer>.spv`.
- Attention runs in registers: a lane holds its tokens' queries, keys and transposed values; the other tokens'
  keys and values come from the lanes holding them as DPP8 broadcasts fused into `v_dot2c`
  (`swizzleInvocationsMaskedAMD`, eight lanes at a time), and the registers rotate by eight lanes per step. Softmax
  is single-pass without the maximum (as the reference): the f32 sums are scaled by 1 / sum at the end instead of
  rounding each probability to binary16 first. That measured the same or 0.1-0.2 dB closer to NGX per layer
  than the exact two-pass order, and is faster.
- Patch merge and expand (the 16 low-resolution tokens of a window): a lane owns two output columns; weights
  are loaded 16 K pairs ahead. In the merge a lane computes `MMT` merged tokens (each weight pair feeds 2 `MMT`
  products); `compile_k.py` takes the largest of 2, 4, 8, 16 that keeps 60% of the lanes busy (enc0/enc1 8,
  enc2 16, enc3 4, enc4 8).
- Shared memory sets how many windows a CU holds. In the C = 64 layers (one wave per head, `AO_ROWS`) the heads'
  attention outputs go into the token rows once every wave has read them, and the MLP takes one hidden chunk of
  32 at a time: 14 KB instead of 19 KB, four windows per CU instead of three.

Selection: `d4r::KNetwork` on `Model` (`Auto`: the matrices when the device exposes the extension, else
`layer_dot2`; `D4R_ENGINE_K_NETWORK=dot2|matrix` overrides, which also runs `layer_dot2` on RDNA3/4 for testing).
`d4r-k` no longer requires cooperative matrices for K. The shim picks `layer_dot2` when the GPU has no f16 matrix
tuple. `[Engine] Enabled = auto` keeps K on the engine on RDNA2 (the native K layers emulate WMMA) and on RDNA4 (FP8
matrix tuple: RX 9070 XT, preset K HDR through the D3D12 harness, game-queue GPU time 1920x1080 -> 3840x2160 2.93 ms
against 9.19 ms for the CUDA path, 960x540 -> 1920x1080 0.95 ms against 4.35 ms, final image 49-52 dB apart), and
leaves it on the native kernels on RDNA3, where they are faster.
A K model built before this has no `_dot2` shaders: the shim then logs that it must be rebuilt and uses the CUDA
backend. On RDNA2 the input stage and the reconstruction run as wave32 (`D4R_ENGINE_K_STAGE_WAVE=32|64` forces
either, the outputs are identical).

Accuracy (`test_k.py --dot2`, each layer on NGX's captured inputs of a 1920x1080 frame, against NGX's outputs):
the same PSNR as `layer.comp` to 0.1 dB on every tensor, 64-80 dB, on the RX 9070 XT and on the Steam Deck. End to
end against the cooperative-matrix network (six temporal frames, 640x360 -> 1280x720): 61-66 dB on the final
image, 65-77 dB on the network head, which is about as far as the matrix network is from NGX.

Steam Deck (gfx1033, RADV 26.2.4, GPU fixed at 1600 MHz), ordinary frames, `d4r-k`, 25 frames per submission:

| Render -> output | Frame | Input | Network | Reconstruction | Frame before 2026-10-20 |
|---|---:|---:|---:|---:|---:|
| 640x360 -> 1280x720 | 8.31 ms | 0.49 | 6.1 | 1.73 | 9.66 ms |
| 640x400 -> 1280x800 | 8.76 ms | 0.51 | 6.4 | 1.91 | 9.87 ms |
| 854x480 -> 1280x720 | 8.50 ms | 0.64 | 6.1 | 1.78 | 9.83 ms |
| 1280x720 -> 1920x1080 | 15.47 ms | 1.30 | 10.3 | 3.96 | 16.64 ms |

2026-10-20: the decoders' padded windows (see the token grid above: 9.66 -> 8.84 ms at 720p), `AO_ROWS`
(enc1 1007 -> 895, dec1 979 -> 884, enc2 437 -> 392, dec2 322 -> 284 us) and `MMT` (enc2 392 -> 369, enc3
352 -> 340, enc4 278 -> 270 us). The network's output is bit-identical to the previous `layer_dot2` on the Deck
and on the RX 9070 XT (same order of every sum). Then the reconstruction: `output_k.comp` takes the low-exponent
fallback (the largest-weight sample where every weight underflows) in the filter loop itself instead of a second
pass, which most waves ran (RX 9070 XT 102.9 -> 94.4 us, same output bits), and runs as wave32 on RDNA2 (before
the change wave64 was faster). Together on the Deck: 1080p 4314 -> 3963 us, 720p 1898 -> 1729 us. The
reconstruction is VALU-bound; removing
parts of it costs (1080p): filter 1.40 ms, history colour 0.82, fallback 0.57, re-exposure 0.16, the rest under 0.1.

The eleven layers at 1920x1080 output (the capture) took 20.4 ms in the first version and 10.9 ms now; the ideal at
the Deck's v_dot2 rate (1.64 TMAC/s, measured 1.51 in a pure loop) is 6.9 ms. The full-resolution layers run at
65-75% of it (the rest is VALU work around the products: conversions, biases, GELU, softmax), the deep layers
lower (few windows, half the waves idle during attention).

Where the time goes (2026-10-20, Steam Deck; per-section shader clocks, ablations and the ACO listing):
- Occupancy dominates. enc1 with one, two and three windows per CU: 3866, 2137, 1630 us (1080p capture); enc0
  with three instead of four: +14%. 256 VGPRs (attention holds queries, keys, values, outputs and the residual
  accumulators) cap every layer at four waves per SIMD.
- In enc1 the products are 24.6k of 34.7k issued instructions per wave (other VALU 3.0k, SALU 5.0k: mostly
  copies between the scalar weight loads' registers). Without weight-cache misses (all weights read from one
  small region) enc1 is 8.5% faster.
- Not useful: weights through the vector path as DPP8 broadcasts, prefetched an iteration ahead (`v_dot2c` with
  DPP8 runs at 91% of peak in isolation, scalar weights at 96%; the layers were 3% slower to 5% faster); 16
  waves per window in dec5 (2.3x slower, probably because a workgroup stays on one CU); four waves per window in enc2/dec2
  (unchanged); skip-connection loads ahead of the expand; 4x8 register blocks in the expand (under 1%). VGPR
  bank conflicts are not the limit (the pure loop has the same 25% of same-bank operand pairs and reaches 96%).
- The deep layers (enc4, dec4, dec5: 4-6 windows at 720p, one per CU) spend their time inside single windows: in
  enc4 the MLP runs at about 75% of a CU, attention at 55% (half the waves wait), the merge at 26%. Splitting a
  window into two workgroups by token half (both compute every token's norm, keys and values, each the rest for
  its 32 tokens; same sums) was slower in every layer but dec5 at 720p (190 -> 155 us; enc4 270 -> 300, dec4
  246 -> 268, 1080p all slower): the repeated keys and values cost more than the idle waves. At most ~0.2 ms (1080p)
  to 0.6 ms (720p) is left there in total.
- int8 (`v_dot4_i32_i8`, twice the v_dot2 rate) for the MLPs of the four full-resolution layers, activations
  quantized per token in the epilogue: 1080p 15.51 -> 15.21 ms, final image 53-59 dB from the binary16 network.
  Not kept: the gain is small next to the quality loss (the conversions and scales eat most of the doubled rate).

Buffer robustness: vkd3d-proton enables `robustBufferAccess2` (`VK_EXT_robustness2`) on the device, and RADV then
loads storage buffers through the vector path instead of scalar loads, which makes this network about 1.6x slower
(the products take their weights from scalar registers). `patches/vkd3d-proton/0005-enable-pipeline-robustness.patch`
lets vkd3d-proton enable `VK_EXT_pipeline_robustness`, and the shim then creates the engine's pipelines with storage
and uniform buffer robustness disabled (`d4r::setBufferRobustnessDisabled`; `D4R_ENGINE_BUFFER_ROBUSTNESS=1` keeps
it; the engine's shaders stay inside their buffers). Harness, 960x540 -> 1920x1080 (GPU fixed at 1600 MHz):
engine GPU time 22.5 -> 16.6 ms, frame 24.9 -> 19.0 ms (18.5 with the reconstruction change above), output
byte-identical. About 1.1 ms of the rest is the harness resetting the history every frame. Without the patched
vkd3d-proton (stock Proton) the shim logs `buffer robustness on` and runs at the old speed.

Through the shim (the D3D12 harness under GE-Proton, `install.sh` run on the Deck, `d4r.ini` defaults, preset K,
640x360 -> 1280x720, HDR, 120-300 frames): `engine backend: ... (network: v_dot2, no cooperative matrices)`, GPU time
of the evaluation 14.8 ms against 17.1 ms for the CUDA path (ZLUDA with the gfx1033 native K layers). The harness
renders one frame every ~240 ms, so both start each evaluation on a cold GPU (the same engine frame takes 9.7 ms
back to back in `d4r-k`). Final image against the CUDA path on 12 frames: 48.0-56.0 dB, the same as the
cooperative-matrix engine on an RX 9070 XT (48.0-55.8 dB) on the same frames; the two engines differ by 57-60 dB,
the Deck's engine from the RX 9070 XT's `layer_dot2` by 58-60 dB and the two GPUs' CUDA paths by 58 dB. The
differences are isolated pixels where K's reconstruction takes a discrete decision.

What the compiler needed: the wave index from `gl_SubgroupID` (with `gl_LocalInvocationID.x >> 5` ACO cannot prove
the weight addresses uniform and loads every weight through the vector path: 14.8 -> 12.8 ms); copy loops that
issue all loads first (one memory latency instead of one per element: 1.9 ms of the frame); ACO fuses DPP8 into
`v_dot2c` only for up to three uses of a broadcast value and implements `row_share` on gfx10 as `v_permlane16`
(a full instruction); RADV caps shared-memory loads at 64 bits. One miscompile: an array copy `tb = tn` followed by
a store to `tn` read the new values; the shader copies element by element.

## Preset M with display-resolution motion vectors (D4RM0005)

Games that clear MVLowRes (Subnautica 2: flags 0x49) give one vector per output pixel, in output pixels; NGX then runs
`rrlite_enc0_4x4_mvhi_*_folded` and `rrlite_post_3_*_mvhi_*_folded` instead of the `mvlo` kernels (expansion,
network and downsample are shared). The two differ from `mvlo` only in where the motion comes from:

- Input stage (`enc0_m.comp` `displayMv`): no depth is read. A render pixel's vector is the motion texel at
  `floor((pixel + 0.5 - jitter) * output / render)`, clamped to the output rectangle, scaled by MV.Scale *
  reconstruction / output (1.5 at 640x360 -> 1280x720 against 3 for render-resolution vectors); it is all three
  candidates (also on the three surfaces dec0 and the next frame read), and the candidates' history positions are
  `fma(pixel + 0.5 - jitter, rnd2hi, mv)` (fused; `mvlo` rounds the product first). The fourth candidate keeps
  `mvlo`'s fused/unfused split between a lane's two pixels.
- Reconstruction (`post_m.comp` binding 12, `mvSize > 0`): the game's motion texture is read per reconstruction
  pixel at `int((x + 0.5) / reconstruction size * output size)`, in f32 (not the binary16 surfaces), and
  `q = fma(scale, texel + offset, x + 0.5)` serves both candidates; the candidate weights are chosen as before.
- `compile_m.py` writes `D4RM0005`; the runtime binds the motion texture for such models only (`MModel::
  hasDisplayMotion`), and the shim keeps older M packages and L on the CUDA backend for these games. The adapter's
  motion fallback is output-sized, as K's.

Validation: harness 640x360 -> 1280x720, flags 0x49, `D4R_HARNESS_MV_HIRES=1 D4R_HARNESS_MOTION_SCENE=1`, ten
frames: an `M_EXACT=1` package on the RX 9070 XT is byte-identical to the CUDA backend running NVIDIA's translated
kernels (`[Kernels] NativeKernels = off`) on every frame; the Steam Deck's v_dot2 stages 35.6 dB from it (35.5 dB
for render-resolution vectors). Render-resolution output is unchanged (`test_frame_m.py` on the capture: every stage
identical, and the harness output byte-identical to the previous build). In that motion scene the render-resolution
harness run is 36.8 dB from the same CUDA reference with the previous build as well, so that gap predates this.
Subnautica 2 on the Deck (640x400 -> 1280x800, Performance) runs preset M on the engine with it.

## Preset M without cooperative matrices or FP8 (RDNA2)

RDNA2 has neither the FP8 cooperative matrices of `layer_m.comp` nor `VK_EXT_shader_float8`, so preset M ran on the
CUDA backend there. `layer_m_dot2.comp` is `layer_m.comp`'s relaxed build (`M_RELAXED`, `M_RESIDUAL_F32`: what
RDNA3/4 run by default) on `v_dot2_f32_f16`, the same way `layer_dot2.comp` is K's: binary16 pairs, f32
accumulation. FP8 values are exact in binary16, so those products are exact; FP8 conversions are packed binary16 and
integer arithmetic (`fp8_soft.glsl`). The MLP, two thirds of the network's multiply-adds, and the attention output
projection run on `v_dot4_i32_i8` (twice the rate) instead: int8 inputs with one scale per token, int8 weights with
one scale per column. `enc0_m.comp`
and `dec0_m.comp` have `DOT2` builds of their products, and `merge_m_dot2.comp` is the encoders' patch merge (int8
as well).

- The weights: the runtime widens `weights8.bin`'s FP8 matrices to binary16 pairs when it loads the model
  (`runtime_m.inc` `widenM`; matrix W[K][N] at FP8 byte offset o becomes [K/2][N] pairs at dword o / 2), and packs
  the MLP's, the output projections' and the patch merges' as int8 words with their column scales there, so a model
  directory serves both networks. `compile_m.py` writes `<layer>_dot2.spv`, `m_merge1/2_dot2.spv`,
  `enc0_m_dot2.spv`, `dec0_m_dot2.spv` and `down_m[_rgb10]_dot2.spv` (shapes: `test_m.dot2_defines`,
  `dot2_merge_defines`).
- A workgroup is an 8x8-token block (four windows), two waves per head (`TSPLIT`), a lane owns two tokens. The
  products with weights take them from scalar memory (the waves split the output channels); attention runs in
  registers with DPP8 broadcasts within groups of eight lanes, as in K. The norms' sums of squares are per wave
  partial sums that meet in shared memory, so every wave normalises its own channels.
- The MLP in int8: norm 2 keeps its FP8-grid values in registers, the tokens' largest magnitudes meet in shared
  memory, and every wave writes its channels as int8 in place of the binary16 rows (no extra shared memory). A
  hidden chunk's 32 values get one scale per token and chunk the same way (one more barrier per chunk); the int32
  sums times the two scales go into the f32 residual. The weights' int8 is per column of each chunk's w1 and w2.
- The attention output projection in int8: every head writes its 32 outputs' largest magnitude per token to shared
  memory, the heads' outputs get one scale per token across the heads and replace the normalised rows as int8; wo is
  one [32 NH][C] int8 matrix with C column scales (the heads' [32][C] blocks are consecutive in `weights8.bin`; the
  runtime checks it). The layers 2-4.4% faster (network 11.44 -> 11.10 ms at 720p), the final image 0.4 dB further
  on the first frame, the second unchanged.
- The encoders' patch merge is its own dispatch after the block (8x8 merged tokens per workgroup, waves split the
  output channels, scalar weights): inside the block the weights would be per lane, about twice the time. Its
  products are int8, one scale per sub-token from its largest FP8 code (the codes' magnitudes are ordered like the
  values, so a shared-memory `atomicMax` finds it): merge1 323 -> 303 us, merge2 173 -> 160 us at 800p, the merged
  tokens 0.1 dB further from NGX's. Half of the merge's time is the products, the rest the loads and quantisation.
- `enc0_m`/`dec0_m` `DOT2`: workgroups of 2x2 blocks (four waves); every wave gathers or loads its block into
  shared memory, then lanes are tokens and waves split the channels, with scalar weights.
- Roundings. RADV folds `float(float16_t(x))` and the `packHalf2x16` round trip away (`round_half.glsl`), so
  `layer_m.comp`'s relaxed build in fact rounds its f32 accumulators straight to FP8. The dot2 stages do the same,
  exactly: `e4m3GridF32` is x + M - M with M = 1.5 * 2^(e + 20) for x's exponent e, as two fmas (equal to the
  integer conversion for every finite f32 but the sign of zero, checked on 4.2 million values on both GPUs).
  Rounding to binary16 first and then to FP8 in packed arithmetic was 1-2.6 dB further from the relaxed build per
  layer. The input and expansion stages' binary16 rounding after every 32 products uses the hardware conversion with
  a binary16 clamp in between, which RADV does not fold (`roundHalfSat`: equal to `roundHalfValue` but that sums from
  65520 saturate at 65504): input stage 1170 -> 933 us, expansion 642 -> 542 us on the Deck, same output bits.
- Barriers: between the dot2 network's dispatches the runtime records a memory barrier (`nextStage`), not the
  execution-only one the matrix layers use (`emptyStage`). With the execution-only barrier one 4x4 window came out
  wrong in about every second 720p frame, a different one run to run: the layers read their inputs through the
  vector caches, which then served lines of a ping-pong buffer read two layers earlier. The matrix network was
  deterministic over six runs with `emptyStage` and is unchanged.
- The stages around the network: with the dot2 network the reconstruction runs as wave32 (1362 -> 1252 us on the
  Deck; `D4R_ENGINE_M_STAGE_WAVE=32|64` forces either), and the downsample uses `down_m_dot2.spv`, built with
  `SEG=2` (two pixels per lane instead of four: 498 -> 393 us, same output; on the RX 9070 XT `SEG=2` is slower,
  16.4 -> 18.0 us, so the matrix path keeps 4).

Selection: `MModel(..., dot2)`; the shim (`engine_m_dot2()`) runs these stages when the device has neither
`VK_EXT_shader_float8` nor f16 cooperative matrices, so RDNA3 keeps its native HIP M kernels and L, which has no
such stages, stays on the CUDA backend without FP8; `D4R_ENGINE_M_NETWORK=dot2` forces them on a GPU without FP8
(and `dot2|matrix` overrides in `d4r-m`; dot2 also runs on RDNA3/4 for testing). `d4r-m` picks dot2 when the
device has no FP8 cooperative matrices. A model built before this has no `_dot2` shaders: loading it fails with
the message to rebuild it with the current `compile_m.py`, and the shim uses the CUDA backend.

Accuracy:
- `test_m.py --dot2`, each layer on NGX's captured inputs (640x360 -> 1280x720) against NGX's outputs: 34.4-41.7 dB
  (before the int8 output projection 34.7-42.0 dB); the relaxed matrix build
  (`M_DEFS='-DM_RELAXED -DM_RESIDUAL_F32 -DWPW=2'`) 35.4-42.6 dB, the dot2 stages with a binary16 MLP 35.3-42.5 dB.
  Against the relaxed matrix build on random FP8 inputs (RX 9070 XT): 38.5-41 dB (with the binary16 MLP 40-44 dB).
- `DOT2=1 test_frame_m.py`, two frames with the engine's own intermediate data and history, final image against
  the capture: 36.5 and 36.4 dB (before the int8 output projection 36.9 and 36.5 dB, before the int8 merge 37.1
  and 36.4 dB); the relaxed matrix build 37.5 and 36.8 dB, with the binary16 MLP 37.6 and 36.8 dB.
- The D3D12 harness (ten temporal frames, 640x360 -> 1280x720, RX 9070 XT): 35.5 dB against the exact build, the
  same as before the int8 output projection (then 35.7 dB against the relaxed matrix network, which is itself
  35.9 dB from the exact one; int8 MLP only: 35.6 and 35.9 dB; binary16 MLP: 35.8 and 35.9 dB).
  The dot2 output is the same on every run.

Steam Deck (gfx1033, RADV, GPU fixed at 1600 MHz), 2026-10-20/21:

| | First version | binary16 MLP | int8 MLP | int8 merges, wave32 reconstruction, SEG 2 | int8 output projection (now) |
|---|---:|---:|---:|---:|---:|
| `d4r-m`, 640x360 -> 1280x720, GPU time per frame | 24.6 ms | 17.2 ms | 14.8 ms | 14.5 ms (14.68) | 14.35 ms |
| of it: network | 20.6 ms | 13.8 ms | 11.4 ms | 11.35 ms (11.44) | 11.10 ms |
| input / expansion / reconstruction / downsample | | 0.93 / 0.54 / 1.36 / 0.50 ms | the same | 0.93 / 0.54 / 1.25 / 0.39 ms | the same |
| D3D12 harness through the shim, 640x400 -> 1280x800 | 28.4 ms | 20.6 ms | 18.0 ms | 17.7 ms | 17.3 ms |
| D3D12 harness through the shim, 960x540 -> 1920x1080 | 51.5 ms | 36.5 ms | 31.7 ms | 31.2 ms | 30.7 ms |

The last two columns are three interleaved `d4r-m` runs and two interleaved harness runs of 200 frames each
(medians; in brackets the earlier build in those runs).

K in the same harness: 11.6 ms at 800p, 18.7 ms at 1080p. The layers at 640x360 -> 1280x720 (us, merges included):
enc1 2424, enc2 1560, tube 559-621 each, dec2 1462, dec1 2347 (fp16 output projection: 2478, 1622, 576-644, 1515,
2392; binary16 MLP: 2876, 1953, 750-830, 1819, 2753). The network is 15.1 GMAC per frame there (the windows of the
shifted grid included), 9.7 of them in the MLP: 6.3 ms at the Deck's v_dot2 rate (1.64 TMAC/s, 1.51 measured in a
pure loop) and v_dot4 rate for the MLP, so the network runs at 55% of that; with the binary16 MLP it ran at 67% of
its 9.2 ms. The tube has 66 workgroups for eight CUs. Removing parts of a layer (binary16 MLP, 320x200 tokens,
enc1): the Q/V and output projections cost what the peak rate predicts
(320 and 175 us against 340 and 170), the MLP runs at about 80%; the rest is the FP8 roundings (12% before the fma
form, which took 2-5% off every layer), GELU (3%), the norms, loads, stores and barriers. Every layer has 128 VGPRs
(no spills) and 13 KiB (C = 64) or 22 KiB of shared memory: eight waves per SIMD either way.

What changed from the first version: the patch merge out of the block, the parallel norms, two waves per head
(`TSPLIT`), the expansion over 8 low-resolution tokens per task when the workgroup has more lanes than column quads,
the input and expansion stages over 2x2 blocks, the two roundings above, the int8 MLP (its first matrix alone: the
layers 9-15% faster, final image -0.3 dB; both: 12-22%, -0.4 dB), the int8 patch merges and the int8 output
projection.
Not kept: Q and V in int8 as well (norm 1 like norm 2): another 5-10% per layer, but 1-2.3 dB per layer against
NGX and the final image 0.7 dB further (36.4 and 36.1 dB): attention is sensitive to them. Not useful:
four-dword-aligned shared-memory reads in the input stage (unchanged). Also not kept, 2026-10-21:
- The decoders' expansion in int8 (one scale per low-resolution token): dec2 1513 -> 1470 us, dec1 2432 -> 2347 us,
  but those layers 1.6 and 1.8 dB further from NGX and the first frame 0.3 dB.
- The tube's tail: its time is linear in the workgroup count (7.7 us each), so the last partial round of the 66
  workgroups on eight CUs costs nothing extra.
- What is left outside the network is texture-bound: of the input stage's 934 us the product is 240 us (at the
  v_dot2 rate), the gather the rest; of the reconstruction's 1250 us the history fetches are about 350 us and the
  Gaussian loop 60 us, over a floor of about 680 us.

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
| `layer_dot2.comp` | Preset K's network without cooperative matrices (RDNA2), `<layer>_dot2.spv` |
| `layer_m_dot2.comp`, `merge_m_dot2.comp`, `fp8_soft.glsl` | Preset M's network and the encoders' patch merges without cooperative matrices or FP8 (RDNA2): v_dot2 and v_dot4 (int8 MLP and patch merges), FP8 in binary16 and integer arithmetic |
| `m_model.py`, `layer_m.comp`, `test_m.py` | Preset M: logical weights and numpy model, layer shader, tiled FP8 packer and per-layer test |
| `hip_net.h`, `hip_net.cpp` | Shared M/K native HIP network loading and immutable graph execution, for native Linux programs |
| `enc0_m.comp`, `dec0_m.comp`, `post_m.comp`, `down_m.comp` | Preset M: input, expansion, reconstruction and downsample stages |
| `activation_m.glsl`, `gen_activation_m.py` | Preset M: sigmoid/tanh/gate table of the 256 E4M3 codes and the script that computes it (`--check` verifies the table) |
| `test_enc0_m.py`, `test_dec0_m.py`, `test_post_m.py`, `test_down_m.py`, `test_frame_m.py` | Preset M: per-stage and whole-frame tests |
| `M_NOTES.md` | Preset M: the algorithms as reverse-engineered |
| `l_model.py`, `test_l.py`, `L_NOTES.md` | Preset L: weights of the input and expansion stages, their tests against a capture, the algorithms |
| `enc0_features.glsl` | Presets M and L: the input stage's 32 per-pixel feature slots |
| `exposure_k.comp` | Automatic exposure: log-luma block sums, then the exposure state and the stages' exposure fields |
| `sr/e_net.comp`, `compile_sr_e.py`, `runtime_e.inc`, `game_e.h`, `game_e.cpp`, `replay_e.cpp` | Presets E and F: the network shader, the model builder, the runtime, the game adapter and the replay (`d4r-e`) |
| `test_frame.py`, `test_model.py` | Temporal result checks and model corruption tests |
| `test_game.cpp` | Queued direct-texture, rectangle, layout, conversion and precision regression checks |
| `test_k.py`, `test_in*.py`, `test_out*.py` | Isolated shader diagnosis and reference comparisons |

`vulkan/ptx2glsl.py` and `vulkan/kvk.cpp` remain reference/research tools. They are not the engine backend.
