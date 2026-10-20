# Ray Reconstruction stages (DLSS-D, NGX feature 13)

d4r's own Vulkan compute shaders for the denoiser in `nvngx_dlssd.dll` 310.7 (presets D and E). One shader per
NVIDIA kernel of the frame; the engine records them in NVIDIA's order:

| stage | NVIDIA kernel | shader | grid (640x360 render, 1280x720 output) |
|---|---|---|---|
| exposure | `cuda_dldn_engine_luma_convert_kernel`, `reduce_sum` x2-3, `auto_exposure_copy_kernel` | `engine/exposure_k.comp` (MODE 1) | |
| enc0 | `cuda_dldn_engine_swin_enc0_kernel` (D: `..._enc0_diamond_wallaby_kernel`) | `enc0.comp` | 81x49, 32 threads |
| enc1-enc5 | `cuda_dldn_engine_swin_enc1_kernel` ... `enc5` | `enc1.comp` ... `enc5.comp` | |
| dec4-dec1 | `cuda_dldn_engine_swin_dec4_kernel` ... `dec1` | `dec4.comp` ... `dec1.comp` | |
| dec0 | `cuda_dldn_engine_swin_dec0_kernel` (D: `..._dec0_diamond_wallaby_kernel`) | `dec0.comp` | 81x49, 32 threads |
| output | `cuda_dldn_engine_hkpn_output_kernel_transformer` (D: `..._diamond_wallaby`) | `hkpn.comp` | 80x45, 16x16 threads |

The D variants of enc0, dec0 and the output kernel are different code; the other nine kernels are shared and only
their weights differ between presets.

## RDNA3 exact preset D (2026-10-14)

Build with `D4R_RR_GFX11=1 D4R_RR_RELAXED=0 engine/rr/build_native_model.sh CAPTURE OUT DLL`.
`rr_gfx11.glsl` reproduces ZLUDA's gfx11 FP8 widening, K positions, scaled f32 accumulators and final binary16
rounding using f16 cooperative matrices. The weight files include a widened copy. Relaxed and mix arithmetic are
unavailable in this build; `VK_EXT_shader_float8` is unnecessary. `kvk`, `evk` and `d4r-rr` enable float8 only when
the driver advertises the extension, because RADV gfx11 may report its feature bits without the extension.

`arithmetic gfx11` in `rr.txt` also selects NGX 310.7's CPU camera operation order. Its inversions, relative camera
translation and intermediate reprojection products leave small residues that affect specular motion history;
double-precision equivalents were insufficient for temporal equality. Other model builds keep their existing path.

RX 7700 XT (gfx1101), RADV, preset D, AW2 town at 853×480 → 2560×1440: enc0 and hkpn are byte-identical on all three
captured launches, joining the ten previously verified stages. Encoder surface 112's undefined alpha is excluded
from the isolated stage check. The complete presented RGBA16F output, including alpha, is byte-identical over all
24 replay frames in two engine runs versus the deterministic CUDA control. Engine GPU time is about 11.4 ms versus
27.9 ms for the current control; warmed samples varied by about 0.1 ms within each engine run. Older CUDA runs
included timing outliers, so their lower 26.7 ms figure is historical, not the paired control. No frame-age or
resolution change contributes to the result. Other RDNA3 GPUs, preset E, and other games require validation.

AW2 live at the same setting: CUDA / engine denoiser GPU medians 27.3 / 12.0 ms; frame intervals 42.6 / 24.4 ms
in the faster sessions and 53.4 / 36.2 ms in the slower sessions. These are separate session states, one launch per
backend in each state; the engine does not resolve the pre-existing slower game state. Frame intervals use the
frame-count/time slope over the final 30 seconds, with sampled instantaneous spreads of about 1 ms for CUDA and
3-4 ms for the engine. The same-frame engine is opt-in; configuration was restored after testing.

Put the local model under `d4r/engine/rr-d`, or set `D4R_ENGINE_MODEL_RR_D_DIR` to its Windows path. The shim runs it
with `[RayReconstruction] Backend = dlss` and `[Engine] Enabled = true`. `D4R_ENGINE_RR_PROFILE` log lines give
game-queue GPU time around the complete engine evaluation; `gpu_frame` identifies the earlier frame whose query
has completed, while `game_interval` is the current call interval. Query readback does not delay the image.

## Stage contract

Every stage shader reads the same data NVIDIA's kernel reads, in the same layouts, so each one can be checked on
its own against a captured launch (`test_layer.py`) and the stages chain exactly like NGX's kernels:

- binding 0: the parameter block, `std430` readonly. It is NVIDIA's argument block byte for byte, except that every
  pointer into the plane arena holds its **byte offset in the arena** (a 64-bit value), the weight pointer is zero,
  and a texture or surface handle is 1 where NGX bound one and 0 where it left the argument null (the kernels test
  optional inputs' handles for null). Fields are read at NVIDIA's argument offsets.
- binding 1: the layer's weight image exactly as NGX uploads it (argument 40's allocation), `std430` readonly.
- binding 2: the plane arena, `std430` read/write.
- bindings 3, 4, ...: the textures in ascending argument offset (combined image samplers, clamp-to-edge,
  normalized coordinates, linear or nearest as NVIDIA's texture object; 8-bit unsigned arrays are sampled as UNORM,
  `SNORM_INT8X4` as SNORM), then the surfaces in ascending argument offset (storage images, `rgba16f` or
  `rgba8_snorm`).
- The dispatch grid is NVIDIA's launch grid. The workgroup size is the shader's choice.

## Checking a stage

Captures (bridge launch and replay dumps of a harness run, see `rr_capture.py`) live outside the repository.

```sh
glslangValidator -V --target-env vulkan1.3 engine/rr/enc1.comp -o build/rr/enc1.spv
python3 engine/rr/test_layer.py CAPTURE swin_enc1 2 build/rr/enc1.spv build/rr/work-enc1 [--time 50]
```

It starts from the arena as it was before NVIDIA's launch, with zeroed surfaces, and compares every surface and
the whole arena with what NVIDIA's launch left. `build/rr/kvk` is `vulkan/kvk.cpp`
(`g++ -O2 -std=c++17 -o build/rr/kvk vulkan/kvk.cpp -lvulkan`).

## Native stages (preset E)

`build_native.py CAPTURE MODEL_DIR` turns a `compile_rr.py` model directory into one where every Swin stage is
d4r's own cooperative-matrix shader instead of translated PTX. Each was derived phase by phase from ZLUDA replays
(`phase_dump.py`: every mma's operands of one window) into an exact numpy reference, then written against it:

| stages | reference | shader | weights |
|---|---|---|---|
| enc1-enc5, dec4-dec1 | `swin_ref.py` (one generic model: C, heads, merge / expand) | `swin_m.comp` | `swin_pack.py` |
| dec0 | `dec0_ref.py` | `dec0_m.comp` | `dec0_pack.py` |
| enc0 | `enc0_ref.py` (after the per-pixel features) | `enc0_m.comp` from `gen_enc0_m.py` | `enc0_pack.py` |

`enc1_m.comp` / `dec1_m.comp` are the hand-written C = 64 instances `swin_m.comp` generalises; `enc1_ref.py` holds
the shared numerics (inter, mm, the cubic softmax, the norm trees). enc0 keeps ptx2rr's translation of NVIDIA's feature
gather (waves 0-2 run its pixel loop) and is native from the embedding on.

Binding 1 differs from the stage contract above: it holds the packed weights (enc0: NGX's image, then the packed
blob). Everything else is the contract, so `test_layer.py --weights FILE` checks a native stage the same way. Against
capE3 every native stage leaves the arena byte-identical to NVIDIA's; surfaces differ in a few values by one binary16
ulp (`exp2` approximation, and decoder texture 88 where NVIDIA's bilinear unit rounds differently), and the whole
network's output is ~96 dB from NVIDIA's.

The shaders run every product transposed (weights x activations) so results feed the next product in registers;
`enc1_m.comp`'s header explains the layouts and `rr_exact.glsl` the binary16 rounding RADV needs (it folds
`float(float16_t(x))` and does not fuse f16 `fma`).

## Native models and preset D

`engine/rr/build_native_model.sh CAPTURE OUT_DIR [NVNGX_DLSSD_DLL]` builds a model whose eleven network stages are
the native shaders (`swin_m.comp`, `dec0_m.comp`, the generated `enc0_m.comp`) for the preset of the capture, with the
translated output stage.

The weights are read from the user's DLL itself (`dll_weights.py`), not from the capture. The DLL holds each model's
tensors as plain binary16 arrays in natural `[out][in]` order, stage after stage; nothing names them, so they are
found by structure: every attention head has a 100 x 100 relative position bias table whose entries depend only on
the offset between two window positions, and a model is eleven consecutive stages with 2 2 4 4 8 8 8 4 4 2 2 heads.
What NGX uploads is the same data rearranged: FP8 codes in the kernels' fragment order, the norm gains times
sqrt(C), the bias times 2 sqrt(2) for the kernels' 64 x 96 query / key order, and (preset E) the 4096-byte noise
table encoder 0 dithers with. `dll_weights.py DLL OUT --model N --check DIR` compares a reading with a phase-dump
extraction: for 310.7 every tensor of both presets is identical, and so are the packed weight files. The capture
only selects which of the DLL's models it ran (and supplies the two argument-block templates); a DLL whose network
kernels are not the ones the shaders were written against (`native_kernels.txt`, SHA-256 of the PTX modules) is
refused unless `D4R_RR_ANY_KERNELS=1`.

`compile_rr.py` uses the translated output kernel for both presets: `engine/rr/hkpn.comp` has the reset path only
(`--reset-only` selects it for captures that reset every frame).

Preset D (`..._diamond_wallaby` kernels) differs from E in three places:

- **enc0**: another feature gather (kept as the translation, like E's) with 30 input features for E's 32, and no
  noise table (its weight image is 4096 bytes smaller).
- **dec0**: the eight bytes per pixel for the output stage are channels 12, 13, 0, 18, 19, 20, 21, 0
  (`-DRR_PRESET_D`; E: 12, 13, 18-23). Everything else is E's kernel.
- **output stage**: its own kernel (closest-depth motion from texture 296, a 4x4 bicubic history fetch). It is the
  translated stage; it has no subgroup operations and runs with 64-wide waves (`rr.txt`).

The reset flag (enc0 byte 365, output stage byte 440) follows `Reset`, not the jitter. With `Reset = 0` the output
stage reprojects and blends its history: check stages against a temporal capture (the harness with
`D4R_HARNESS_JITTER_SCENE=1`, and `D4R_CUDA_REPLAY_DUMP_LIMIT=400` so every launch is saved), not only against
captures that reset every frame. `engine/rr/hkpn.comp` implements the reset path only.

Preset D, RX 9070 XT, RADV 26.2.4, 2026-10-12, 1280x720 -> 3840x2160 (`d4r-rr`, `D4R_RR_TIME_REPEAT=20`): 5.0-5.2 ms
per frame (network 4.5, output 0.55-0.65), from 17.5 ms with every stage translated. Every network stage reproduces
the capture's arena byte for byte on reset and temporal frames. The presented image against the CUDA path over
temporal frames: 96-106 dB on the reset frame, then 41-52 dB (one-ulp differences in the filtered surfaces are
amplified by the recurrent feature; twelve frames stay at 41-45 dB).

RE Requiem (2026-10-12, 1280x720 -> 3840x2160, R9G9B9E5 colour, roughness packed in the normals' w): with
`hkpn.comp` the picture was 3-pixel blocks that shook with the jitter (no history); with the translated output stage
presets D and E both give a stable, reconstructed image on the engine. Not compared with the CUDA path in that game.

## Matching the reference bit for bit (2026-10-12)

The reference is the CUDA path through ZLUDA on the same GPU, so its `rcp.approx`, `ex2.approx` and bilinear texture
fetches are the GPU's own instructions and texture unit, not correctly rounded arithmetic. Two places in the native
filters had been "corrected" to exact arithmetic and were one ulp off in ~0.01-0.2% of the filtered surfaces:

- `rcp16` (the filters' softmax normalisation and gate) is now the plain hardware reciprocal;
- the decoders' texture 88 goes through the bilinear sampler at NVIDIA's coordinates (`rr_decfilter.glsl`,
  `dec0_m.comp`) instead of a hand-computed 1/4, 3/4 blend.

With both, every network stage (arena and surfaces 112 / 120 / 128 / 144 / 304) is identical to the capture on every
frame of capDT12, capD4K, capD1080 (preset D) and capE3 (preset E). What remains is the translated output stage:
~0.004% of surface 224 / 240 one ulp off (single channels, values just above 0.5; cause not found). The output is
107-112 dB on frames where the network's inputs are still exact. At 640x360 -> 1080p that holds for all twelve
frames of capDT12 (was 96 dB, then 41-45); at 1280x720 and 1920x1080 -> 4K the history's differing values reach
encoder 0 through texture 496 and the output falls to 52 / 45 / 43 dB (capD4K), 53 / 49 dB (capD1080) as before.

Speed: the compiled stages spend about half their instructions on float conversions. `spv_exact_convert.py` marks
the conversions NoContraction so RADV keeps a binary16 rounding without the opaque OR (`RR_EXACT_CONVERT`), which is
bit-exact and ~8% faster: preset D 4.8 ms at 1280x720 -> 4K, 9.5 ms at 1920x1080 -> 4K (was 5.1 / 10.3).
`D4R_RR_RELAXED=1|2` builds drop NVIDIA's double rounding instead (8.2 / 7.6 ms at 1080p) but are not exact: about 1%
of a layer's FP8 codes change and the output is 39-44 dB from the reference on every frame.

Profile of a full-resolution stage at 1080p (dec0, 3.0 ms): expand 0.31, key norm 0.16, attention 1.37 (per head:
projections 0.15, scores + softmax 0.26, P V 0.17, out-projection 0.10), MLP 0.69, output filter 0.50. The 136
matrix products themselves are 0.45-0.5 ms; weight and table memory traffic is under 8%. enc0's translated feature
gather is 0.28 of its 3.0 ms.

## Optional inputs: specular hit distance and disocclusion mask (2026-10-12)

RE Requiem passes `DLSSD.SpecularHitDistance` (R32_FLOAT) and `DLSS.DisocclusionMask` (R16_FLOAT). Harness captures
with synthetic ones (`D4R_HARNESS_RR_EXTRA=hit,mask`; capDX, capDXhit, capDXmask) show the wiring:

| input | encoder 0 texture | encoder 0 flag byte | output stage |
|---|---|---|---|
| specular hit distance | argument 248 | 536 | flag byte 160 |
| disocclusion mask | argument 296 | 539 | - |

Encoder 0 also tests the handle at argument 248 for null before it takes the hit distance path (it writes a
reprojected specular motion into channels 2 and 3 of surface 304), which is why the parameter block now carries the
handles' presence. With both bound the network is identical to the capture on every frame (capDX). The output
stage then differs from the CUDA path on the image's border pixels only (63-67 dB on temporal frames): NVIDIA's
kernel fetches neighbours outside the image there, CUDA clamps such integer fetches to the edge (every denoiser
texture's address mode is clamp; the translation and rrsim do the same), and ZLUDA's image load returns zero. With
zero outside the image the stage matches the capture like before (99.997%); the engine keeps CUDA's clamp.

Exact speed-ups since the first native build, all bit-identical per stage: conversions kept by NoContraction instead of
an opaque OR, the MLP activation and its FP8 conversion from a 64 KiB table (`enc1_pack.act_table`), FP8 conversions in
pairs, no saturation clamp on attention probabilities. Preset D: 4.4 ms at 1280x720 -> 4K, 8.6 ms at 1920x1080 -> 4K.

Output stage, hypotheses tested for its remaining ~0.004% one-ulp values and rejected (2026-10-12): correctly rounded
exp2 / reciprocal (rrsim's exact arithmetic is further from the capture than the GPU), fusing the kernel's 34
single-use multiply-subtract pairs into fma, merging the two constant multiplications after the log transform,
`precise` on the inlined exp2, NoContraction on its conversions. The differing values come in spatial clusters, one
colour channel at a time, always one ulp, mostly just above 0.5 in surface 224.

## HIP against Vulkan, and the fused rounding (2026-10-12)

Encoder 1 ported line for line to a standalone HIP kernel (`kernels/rr/swin_native.h`, runner
`kernels/tools/rr_swin_native_test.cpp`; capD1080 frame 2, 5 launches per round): 835 us against the Vulkan stage's
770-788 us, the arena byte-identical. A HIP port of the network is not worth making. What the port did show:

- The FP8 product alone costs 0.61 us per product per wave on this grid, 2.07 us when each is followed by the
  f32 -> binary16 -> f32 rounding of its eight accumulator registers. Products and that rounding are about half of
  a stage on either backend.
- Keeping the accumulator as binary16 and taking f16(P + acc) with `v_fma_mix` (as `kernels/m` does) halves the
  rounding's instructions: HIP 780 us. RADV emits the same instruction for `float16_t(p + float(acc))`, so the
  Vulkan stages have it as a build option: `D4R_RR_MIX=1 build_native_model.sh ...` (`RR_MIX`, the shaders' K32).

`RR_MIX` is one rounding of the exact sum where the reference has two, so it is not bit-exact by construction.
Preset D: every network stage still identical to the capture on all 21 frames of capDT12 / capD4K / capD1080 / capDX
/ capDXcp / capDXhw (the same figures as the default build), 4.22 ms at 1280x720 -> 4K and 8.3 ms at 1920x1080 -> 4K
(default 4.45 / 8.7). Preset E (capE3): frames 1 and 3 have a value of encoder 0 one step off (127 dB), which the
later stages amplify to 41 dB at the output; frame 2 is identical. The default build stays the exact one.

## Cyberpunk 2077: more inputs and settings (2026-10-12)

Found on harness captures (`D4R_HARNESS_RR_EXTRA=particles,sss`, `D4R_HARNESS_RR_HW_DEPTH=1`, `D4R_RR_PRESET=N`):

| what the game passes | where NGX puts it |
|---|---|
| `DLSSD.ScreenSpaceSubsurfaceScatteringGuide` (R16F) | encoder 0 texture 264 (its handle is the only switch) |
| `DLSSD.ColorBeforeParticles` (RGBA16F) | output stage texture 416, flag byte 432, and a second output-size history: surface 384, read back next frame as texture 392 (linear) |
| `DLSS.Use.HW.Depth = 1` | encoder 0 byte 980, output stage byte 443 |
| preset hint | 5 runs preset E; every other value (0-4, 6, 0x80000002, 0x80000005) runs preset D |

Integer texture fetches outside the image return zero (`fetch0`), which is what the reference does; CUDA would clamp.
With Cyberpunk's set (hit distance, particles, subsurface guide, hardware depth) the engine's presented image is
byte-identical to the CUDA path through the shim on the frames the harness can compare (640x360 -> 1920x1080), and
with RE Requiem's set the network is identical on every frame (output 107-109 dB). At 1280x720 and 1920x1080 -> 4K
without these inputs the output still falls to 43-53 dB after the first frame (the output stage's one-ulp values);
a correctly rounded log2 there matches the capture worse than the GPU instruction does.

## The output stage's one-step differences: fused multiply-convert (2026-10-13)

The rare one-binary16-step values of the translated output stage (above: ~0.004% of surfaces 224 / 240, "cause not
found") were the driver fusing an f32 product (or sum) with the conversion that follows it into `v_fma_mix`, which
rounds the exact result once. The kernel rounds twice, to f32 and then to binary16, and the two differ when the f32
result is exactly a binary16 midpoint, as `log2(x) * c1 * c2` of the output stage's tone curve sometimes is.
`ptx2rr.py` now ORs an opaque zero into the f32 bits before every `cvt.rn.f16.f32` (and into the binary16 bits
after, against the fold of `float(float16_t(x))`). With it presets D and E are byte-identical to the CUDA path on
every frame of every capture (capDT12, capD4K, capD1080, capDX, capDXcp, capDXhw, capE3; history and output), at
+0.05 ms: 4.27 / 4.45 ms at 1280x720 -> 4K. Found with `build/rr`-style bisection: probe every register of a block
through ZLUDA (debug stores added to the PTX, `zluda_replay.py`) and through the translated shader
(`ptx2rr.py --probe`) and list the first that differ.

Other reference behaviours the translation now follows, all found the same way on preset F's kernels:

- formatted stores to normalized surfaces round half away from zero in f32 (ZLUDA's `surface_formatted_store_bits`);
  UNORM 10:10:10:2 arrays are RGBA32F images holding the quantised values `q / 1023` (the correctly rounded quotient,
  from a table: the driver divides by multiplying with a reciprocal);
- `tanh.approx.f32` is OCML's `tanhF` with LLVM's AMDGPU lowering of `exp` (`lowerFEXP`, fast-FMA path): the same
  operations, the fused ones through f64 (the driver's f32 `fma` is a product and a sum whenever the product also
  exists on its own); `tanh.approx.f16` is the correctly rounded value (a table);
- the driver's GLSL `fma` on binary16 is a true fused operation, as the reference's is: no emulation there;
- texture fetches with a binary16 result keep that rounding.

## Preset F: DLSS 4.5 Ray Reconstruction (2026-10-13)

`nvngx_dlssd.dll` 310.9 has a third model with its own kernels, `rr2_*`, and runs it for every preset hint but 4 (D)
and 5 (E): none, 6, Cyberpunk 2077's 0x80000002. The engine runs it from a `D4RR0002` model:

```sh
python3 kernels/tools/extract_dlss_ptx.py NVNGX_DLSSD_DLL ~/.cache/d4r-rr-corpus-f       # its PTX (zstd in 310.9)
python3 engine/compile_rr2.py CAPTURE OUT_DIR --native NVNGX_DLSSD_DLL [--particles CAPTURE_WITH_PARTICLES]
```

The runtime is `engine/runtime_rr2.inc` (behind `RModel` / `REngine`); the shim uses `engine\rr-f` (or
`D4R_ENGINE_MODEL_RR_F_DIR`) and, when that model exists, NGX 310.9's hint mapping.

A frame, as NGX launches it:

| pass | NVIDIA kernel | engine |
|---|---|---|
| exposure | `rr2_histogram_auto_exposure_basic_kernel` | `exposure_f.comp` (two passes) |
| enc0 | `rr2_enc0_kernel` | translated feature gather and guide tail around the native block (`gen_rr2_enc0.py`) |
| enc1 - enc5 | `rr2_enc1_kernel` ... | `rr2_m.comp` |
| dec4 - dec1 | `rr2_decN_kernel_0`, `rr2_decN_kernel_1` | `rr2_m.comp` |
| dec0 | `rr2_dec0_kernel_0`, `rr2_dec0_kernel_1` | `rr2_m.comp`; the second: native block, translated output tail |
| output | `rr2_post_kernel`, `rr2_downsample_kernel_static_hdr[_no_transparency]` per tile, `rr2_pooled_history2_kernel` | translated (`ptx2rr.py`) |

The network (`rr2_ref.py` is its exact numpy reference and documents the block): D/E's levels, window grids and
arithmetic (FP8 operands, binary16 sums rounded per 32-deep step, the cubic softmax, the MLP activation), with a
different block. Each kernel is one block on 8x8-token windows that alternate their shift per frame: heads that
share one projection for queries and keys (so the scores are symmetric; bias `(T + T^T) sqrt(2)` of a 64 x 64
relative position table), all reading the block's input and summing their out-projections; MLP branches likewise;
both sums enter the residual times a binary16 scalar from the argument block. Encoders merge 2x2 for the next level,
a decoder's first kernel expands the level below and adds the skip times a third scalar. Next to the tokens every
level carries a three-channel guide image: encoders gate it (`sigmoid(head) * s0`) and average it 2x2 for the next
level, decoders upsample it bilinearly, gate it and blend it with the encoder's. All stages exchange data through
the arena (3152 units of W0 * H0 / 32 bytes; NGX's region offsets, which scale with the render size), tokens as FP8
planes of 32 channels, guides as four binary16 per token.

Weights: the DLL holds the model as plain binary16 tensors in natural `[out][in]` order, kernel after kernel
(`rr2_ref.dll_weights`; found by the sixteen kernels' relative position tables, 2 2 4 4 8 8 8 8 4 4 4 4 2 2 2 2
heads). `rr2_pack.py` packs them for `rr2_m.comp`; nothing of the network comes from the capture. The capture still
supplies every kernel's argument block (the runtime keeps their constants and fills the geometry, the frame's
values and the pointers) and encoder 0's second table.

The output stage filters an intermediate image of 1.5 times the output size in tiles (columns of at most 720, rows
of at most 1620 intermediate pixels, 4 pixels of overlap inside), reduces each tile to the output, and keeps two
histories for the next frame: the intermediate image, and a half-size copy whose quarter-size reduction
(`rr2_pooled_history2_kernel`) encoder 0 also reads (UNORM 10:10:10:2 arrays, see above). NGX runs filter and
reduction concurrently with progress counters; the kernels skip that when the counter pointers are null, which is how
the engine runs them, in order. They have no matrix products and run with 64-wide waves (their warp-level shuffles
address the lane's own half).

Options (harness captures capFX*): encoder 0 byte 466 packed roughness, 568 inverted depth, 569 / texture 272
`DLSSD.SpecularHitDistance`, 571 / texture 336 `DLSS.DisocclusionMask`, texture 288 the subsurface guide, 1028
`DLSS.Use.HW.Depth`; `DLSSD.ColorBeforeParticles` selects the other reduction kernel (model stage `downp`, texture
208, an output-size history 256 <-> 264, flag bytes 196-198). Encoder 0's surfaces 376 / 384 are the output filter's
textures 64 / 72. Pre-exposure does not reach these kernels.

Checked (`d4r-rr` on `rr_inputs.py` inputs; every stage surface and the presented image against the CUDA path):
byte-identical on every frame of capF (640x360 -> 1080p, 3 frames), capF900 (800x450 -> 1600x900), capF4K
(1280x720 -> 4K, 16 tiles) and of the option captures (motion, a real camera, inverted and hardware depth, packed
roughness, hit distance, disocclusion mask, subsurface guide, particles); through the shim (D3D12 harness under
Proton) the engine's frames are byte-identical to the CUDA backend's too (`D4R_RR_PRESET=6 D4R_RR_DLL=<310.9 DLL>
scripts/run_rr_engine_compare.sh`; that script still compares frame N with frame N, and the harness presents the
CUDA result one frame late in the jitter scene).

Speed, RX 9070 XT, 1280x720 -> 3840x2160: 6.2 / 6.6 ms per frame (exposure 0.03, network 4.1 - 4.2, output 2.1 -
2.4), from 14.5 ms with every kernel translated; preset D is 4.3 / 4.5. Per stage (us): enc0 1076 (its block 521),
dec0_1 775, dec0_0 630, enc1 340, dec1_0 316, dec1_1 269, the rest under 180; filter and reduction about 130 per
tile, 16 tiles. Not done: native output passes and feature gather.

### Where the translated parts' time went (2026-10-13, later)

Now about 5.8 / 6.2 ms (5.7 / 6.1 with the 10-bit histories that were later withdrawn), still byte-identical on every capture and through the
shim. Profiling the translated code (cutting it at points behind an opaque condition, `D4R_ENGINE_RR_SKIP=post,down,
pooled`) showed its arithmetic is spread evenly - no hot spot a rewrite would remove - and that the losses were in
how the passes were arranged:

- **Encoder 0 in two passes** (-0.2 ms). One shader ran NVIDIA's 32-lane gather on one of the block's four waves
  while the other three waited: 0.40 ms for the gather, 0.24 ms as a pass of its own in 32-thread groups.
  `gen_rr2_enc0.py --split gather | block` writes both; the gather pass (`stage enc0g`, the same bindings and
  parameter block as enc0) stores each window's record in the arena buffer (see Memory below): the first product's A
  fragments (1024 words) and, per lane, the values the tail reads from the gather (six binary16 on the reference
  build; the generator finds them and recomputes the tail's index arithmetic). The record's address is a word
  compile_rr2.py appends to enc0's argument block. Cost: 76 bytes per render pixel of arena.
  `D4R_RR2_SPLIT=0` builds the single shader.
- **Tile passes in overlapping groups** (-0.15 ms). The 16 tiles' post and down passes ran one after another, each
  behind a barrier, and the gaps between 32 short dispatches added up. Each tile now has a slot (its two regions;
  eight slots, 58 MB at 4K) and a group's post passes run without barriers between them, as do its down passes
  together with the next group's post passes on the other half of the slots - what NGX does with two slots and its
  progress counters. `D4R_ENGINE_RR_SKIP=serial` restores the old order.
- **Pooled histories as real 10:10:10:2 images** (the default since 2026-10-20). 4 bytes a pixel instead of
  RGBA32F's 16 saved 0.1 ms and was identical at 3x and 2x scaling, but the hardware's bilinear filter rounds a
  UNORM image differently from a float one: at 1.5x (capFQ) temporal frames came out at 44 - 51 dB (28.8 dB on
  random-motion noise's history frames, below). `D4R_RR2_POOLED=rgba32f` builds the reference form.

Tried without gain: 32-bit arena accesses for the translated code's paired 16-bit ones (0.01 ms), dropping the
subgroup shuffles or the depth search (they cost nothing measurable). What is left is NVIDIA's arithmetic: gather
0.24 - 0.33 ms, decoder 0's output tail 0.18, post 1.1 - 1.3, down 0.7.

### Other scaling ratios, and what a game-like configuration showed (2026-10-13, later)

Asked whether instability seen in RE Requiem came from this implementation, the engine was compared with the CUDA
path through the shim in configurations the earlier captures did not cover (`D4R_RR_PRESET=6 D4R_RR_DLL=<310.9>
scripts/run_rr_engine_compare.sh`, frames aligned by one):

- **Render widths that are not a multiple of 32 were wrong** (1114x626 -> 1080p: 15 dB): the post pass's argument 8
  is the row stride of the network's output, the padded width W0, not the render width. Fixed; identical now.
- **Below 4/3 scaling NGX runs another output filter**, `rr2_post_kernel_3_2` (1456x820 -> 1080p and larger render
  sizes, DLAA included; 1440x810 and smaller use `rr2_post_kernel`). The model has no such stage, so the shim hands
  those features to the CUDA backend. `rr2_histogram_auto_exposure_histogram_kernel` was never launched in any run.
- **DLSS.Roughness.Mode 1 with a roughness texture bound**: NGX follows the mode, the shim followed the texture.
  Fixed for preset F (RE Requiem binds none, so it was not affected).
- 1.5x, 1.72x, 2x and 4/3 scaling, and RE Requiem's inputs together (inverted and hardware depth, hit distance,
  mask, packed roughness, scaled motion vectors, a moving scene): identical.
- **Open: the camera matrices are not NGX's bit for bit.** NGX computes them in single precision: argument 644 is
  the f32 product view x clip, 836 its inverse by cofactors times 1 / det (glm::inverse's operation order; both
  reproduced bit for bit in build/rr/f/matfit.py), 900 is an inverse of the pixel matrix (it carries that inverse's
  residues, and 0.9999999 for 1 at 1114x626) and 964 was not reproduced. The engine computes them in double and
  rounds, a last-bit difference in a few elements. With a non-identity camera, hit distance and mask bound and a
  moving scene that showed as 40 - 50 dB on later frames in a synthetic scene whose affected feature is rounding
  noise around zero in both paths; no other configuration was affected.

### Memory (2026-10-14)

At 1280x720 -> 4K the engine needs about 600 MB for preset F; the two 1.5x output histories (2 x 149 MB, NGX has
the same) are half of it. NVIDIA's DLL, its allocations traced through the CUDA path and replayed on an RTX PRO 4500
Blackwell, takes 526 MiB of feature state, 106 MiB of input / output arrays and 2 MiB of code, plus 262 MiB of CUDA
context. Three cuts, every capture still byte-identical (`verify.sh`, `verify_x.sh`) and the shim's frames
byte-identical to the previous build, same speed:

- **The tile slots share the arena with the network's regions.** Every arena region is written and read within one
  frame; in the output stage only the network's output (dec0_1's argument 64) and something after it are read, so
  the slots that fit before the network's output go there (all 8 at 720p -> 4K) and the rest after the arena. A
  first try that put the slots right after the network's output broke 360p -> 1080p's output stage.
- **The gather records' rows of windows fill the gaps between encoder 0's regions**: the two largest (units
  1216 - 2240 and 2304 - 2560) and then from the end of its last region (unit 2816) on, instead of one piece after
  the tile slots. The records (76 bytes a render pixel, about 2450 units) are larger than all the gaps, so 46 of 93
  rows at 720p still reach past the arena (36 MB). Models from compile_rr2.py mark this with "s3" on the enc0g line
  and append six words to enc0's block (three ranges' addresses, the first two's rows; `gen_rr2_enc0.py`); older
  models keep the one-piece layout. Same speed as one piece (four alternating runs at 720p -> 4K).
- **The engine's own output image is created only for frames that do not write the game's texture**, and the
  weights go into the command buffer with `vkCmdUpdateBuffer` instead of through a host-visible staging copy that
  stayed allocated.

Through the shim (direct output) the harness process peaks 159 MiB lower (1100 -> 941 MiB). Not exact: the pooled
histories as RGB10A2 (`D4R_RR2_POOLED=rgb10_a2`, -70 MB, now the default, see above) hold q / 1023, which neither
binary16 nor UNORM16 represents exactly. `D4R_ENGINE_RR_ARENA`
keeps the old separate layout, so that arena dumps show the network's regions as NGX leaves them.

## Against a real NVIDIA card: `D4R_RR_FUSE` (2026-10-13)

Every reference above is the CUDA path through ZLUDA. NVIDIA's own kernels (310.9.1's `rr2_*` PTX, replayed through
plain CUDA on an RTX PRO 4500 Blackwell, driver 580; tools and logs in `build/rr/f/nv/`) give an image 39-42 dB from
that reference on every frame of capF, capF4K and capFQ. Register bisection between the two (`probe_pack.py`,
`probe_cmp.py`: a debug store after every instruction, run on both) found why:

- **Fused multiply-adds.** NVIDIA's compiler contracts a `mul.f16` / `mul.f16x2` whose product only feeds an
  `add` / `sub` into one fused multiply-add; the kernels give those instructions no rounding modifier. ZLUDA runs them
  as written. `ptx_contract.py` is the rule (single-use product, same block, the first operand when both are
  products): with it ZLUDA's result of every network kernel matches the card's (0 of ~5900 4 KiB chunks for most
  launches, a handful where an `ex2.approx` / `div.approx` differs by one f32 step). The FP8 products, the e4m3
  conversions and every unfiltered texture fetch were already identical. The sites are the residual steps
  `out = in + branch * s`, the MLP activation `x (0.5 + y (0.412 - 0.081 |y|))` and the guide's bilinear blend.
  Probing a product stops the card from fusing it: compare unprobed results, or probe everything.
- **Filtered fetches of 8-bit textures.** A filtered fetch of the SNORM8 normals returns this card's value (8-bit
  weights) rounded to a 16-bit step: `f32(roundEven(v * 32767) / 32767)`, exact on 256 of 256 probed values. The
  output filter's UNORM8 fetches differ too; no model for those yet.
- **Approximate instructions.** `lg2.approx` near 1 differs by ~8e-8 absolute, `ex2.approx` / `rcp.approx` /
  `div.approx` by one f32 step. Not emulated.

`D4R_RR_FUSE=1 compile_rr2.py CAPTURE OUT --ptx-dir <build/rr/f/nv/mkptx.sh output> --native DLL` builds the model
with the first two: `rr2_m.comp -DRR_FUSE` (`MADD` / `MADD4`), the activation table from `enc1_ref.act` with
`FUSE`, the translated parts from the rewritten modules, encoder 0's normals through `ptx2rr.py --snorm16 528`. All
14 native stages are byte-identical to ZLUDA running the rewritten modules (`build/rr/f/nv/nfuse.sh`), and so is
the whole model on capF's three frames. Presented image against the NVIDIA card, 640x360 -> 1080p:

| frame | default build | `D4R_RR_FUSE` |
|---|---|---|
| 1 (reset) | 40.9 dB, 27.3% of values equal | 85.6 dB, 99.5% equal |
| 2 | 39.2 dB | 43.0 dB |
| 3 | 39.3 dB | 41.8 dB |

Temporal frames stay near 42 dB: the remaining one-step differences (approximate instructions, the UNORM8 fetches)
enter encoder 0's reprojection and the recurrent history amplifies them, as the one-step values of the output stage
did before the fused-convert fix. 1280x720 -> 4K: 5.68 / 6.24 ms (default 5.78 / 6.35). The default build is
unchanged and still byte-identical to the captures.

### This GPU's own arithmetic for preset F: `D4R_RR_RELAXED` (2026-10-13)

Two NVIDIA setups differ from each other as much as a relaxed build differs from the reference: NVIDIA's kernels on
an RTX 4000 Ada (driver 595) and on the Blackwell cards (driver 580) give images 41.0 / 39.0 / 38.9 dB apart on
capF's three frames, although their approximate instructions and filtered texture fetches are bit-identical
(`build/rr/f/nv/math/`); the `D4R_RR_FUSE` build matches the Blackwell result only. So there is no single NVIDIA
image to reproduce, and `compile_rr2.py` now takes `D4R_RR_RELAXED=1|2` like `build_native_model.sh` does for D / E
(`rr_exact.glsl`: f32 accumulators across the 32-deep steps, one rounding into FP8; 2 also leaves the conversions to
the driver). Preset F, 1280x720 -> 4K: 5.40 / 5.99 ms against 5.78 / 6.35 (network 3.45 / 3.77 against 3.83 / 4.14);
40.8 / 39.1 / 39.2 dB from the exact build and 39-41 dB from either NVIDIA card. The default stays exact.

Why only 6%: with the rounding gone the products are about a sixth of a full-resolution stage. dec0's first kernel
at 720p, relaxed (`build/rr/f/nv/exp/t.sh`, cutting parts out): 626 us, of which the two attention heads ~370, the
four MLP branches ~100-160, expansion / residual / store ~120-160, an empty dispatch 10; inside the attention the
softmax table ~30, the bias tiles ~40, weight loads ~12. Its 88 products alone would take ~108 us at the measured
198 T MAC/s. The compiled stage still has 338 binary16 <-> f32 conversions, 176 FP8 conversions and 128 clamps for
44 product instructions (biases and tables are binary16, the activation and softmax tables are indexed by binary16
bits, FP8 conversions saturate): what is left is format traffic and the four waves' shared-memory hand-offs, not
arithmetic. NVIDIA's kernel runs the same stage in 203 us with one 32-thread warp per window.

`D4R_RR_RELAXED=3` adds `RR_NATIVE` (rr2_m.comp): the residual steps and the MLP activation in f32, straight into
FP8, with no binary16 in between and no activation table. 1280x720 -> 4K: 5.17 / 5.72 ms (network 3.24 / 3.48), still
39-41 dB from the exact build and from both NVIDIA cards. dec0's first kernel at that level: 588 us; the barriers
cost nothing; reading the constants (weight tiles, bias tiles, the softmax table) costs 170 us; the products about
108; the rest is the tokens' way through shared memory and the arena, conversions and the expansion
(`build/rr/f/nv/exp/run1.sh`).

What bounds a stage (2026-10-13; `build/rr/f/nv/exp/`). dec0's first kernel at level 3 takes 581 us at 720p and a
wave executes about 2500 instructions (`dyn.py`): 96 products, 356 binary16 <-> f32 conversions, 296 FP8
conversions, 224 clamps of the saturating FP8 conversions, 142 buffer loads, 75 shared-memory operations, about 400
wait / delay markers. What was tried on top of level 3, none of it a gain:

- the clamps removed: -30 us (2.2 ps per instruction and wave), but not usable: without the clamp the hardware
  conversion returns NaN above 464 where the clamp gives 448 (`satprobe.comp`);
- f32 copies of the biases: 176 fewer conversions and 78 more loads per wave, slightly slower (587-595 us): a load
  costs about three arithmetic instructions (on the later shaders they are faster, see 2026-10-20 below);
- the softmax cubic computed, not read from its table: the same time;
- no barriers: the same time;
- one wave per window walking the four token slices, every weight, key and value tile read once
  (`rr2_w.comp`, byte-identical to the four-wave build): 27% fewer instructions per window, 30% fewer buffer loads,
  60% fewer shared-memory loads, and the same time (600 us); it needs all 256 vector registers (four waves: 192).

So time here is not a function of the instruction count alone, and there is no model yet that predicts both the
clamp result and the one-wave result. The output stage is compute-bound too: 100 / 300 extra integer steps per
thread in the filter add 0.45 / 1.33 ms (`dz.sh`). NVIDIA's kernel runs dec0's first kernel in 203 us.

Shared memory throttles this GPU (2026-10-13, `build/rr/f/nv/cost/occ.py`, `occ2.py`). A dependent chain of f32
multiply-adds runs at full rate while the workgroup declares up to about 4 KiB of `shared` storage per wave; above
that the rate falls in proportion (one wave per workgroup: 6 KiB 1.24x slower, 8 KiB 1.55x, 13 KiB 3.1x, 62 KiB 12x;
four waves: 13 KiB full rate, 26 KiB 1.55x, 52 KiB 3.1x). The register count does not matter (0 to 224 live values:
the same rate). The decoders' first kernels held 21 to 46 KiB (LOW, EX, XE): `rr2_m.comp` now keeps the 4 x 4 low
tokens in KVS and the block's unquantised input in each lane's registers (`xr`), byte-identical on every stage
checked, and the network is 0.10 - 0.13 ms faster at 1280x720 -> 4K: exact build 5.68 / 6.25 ms (was 5.78 / 6.35),
`D4R_RR_RELAXED=3` 5.07 / 5.56 ms (was 5.17 / 5.72). It is also why the first one-wave prototype (13 KiB in one
wave) showed no gain: with 4 KiB (`exp/rr2_w.comp`: the low tokens borrow XT, the expansion borrows KVS, V^T is
stored, read into registers and overwritten by QK) it takes 534 us against the four-wave shader's 581, still
byte-identical to it. Extra arithmetic injected into either shader costs one instruction per cycle on about 128
wave slots (3.8e11 wave-instructions per second); NVIDIA's card measures 7.5e11 (`build/rr/f/nv/sass/rate.cpp`), and
its compiled kernel for this stage is 3,729 instructions per window (`sass/nv/dec0_0.sass`) against about 10,000
executed here. What an instruction costs here (`cost/cost.py`, `cost2.py`): an f32 multiply-add is the unit; a
32-bit integer multiply about seven; converting a pair to FP8 with the clamp about eight; a buffer read about 0.1 ps
per byte and lane (16 bytes: some forty multiply-adds), nearly twice that from a 1 MiB table.

`engine/rr/rr2_w.comp` (2026-10-13) is decoder 0's first kernel for `D4R_RR_RELAXED=3` with one wave per window:
every weight, key and value tile read once per window, the expansion plus skip computed once and kept in registers,
the scores' cross quadrant computed once and reused transposed (368 products for 384), FP8 values converted in
pairs, 4 KiB of shared memory. `compile_rr2.py` uses it for dec0_0 at level 3 (`ONE_WAVE`). The stage alone:
453 us against 581 us at 1280x720, arena byte-identical to `rr2_m.comp` at that level
(`build/rr/f/nv/invest-20261013/REPORT.md`: 20 variants, each change's instruction count and time). In the engine
every image of capF's three frames is identical to the four-wave level 3 model's, and 1280x720 -> 4K is about
5.00 / 5.55 ms (four-wave level 3: 5.10 / 5.61; network 3.05 / 3.31 against 3.15 / 3.39). The shader covers C = 32
only; the other stages still use `rr2_m.comp`.

`rr2_w.comp` also covers encoder 0's block now (KIND 4, embedded by `gen_rr2_enc0.py --one-wave` in NVIDIA's own
32-thread group, the translated guide tail on the same wave; `compile_rr2.py` `ONE_WAVE = {'dec0_0', 'enc0'}` at
`D4R_RR_RELAXED=3`). All 45 images of capF's three frames are identical to the four-wave level 3 model's.
1280x720 -> 4K, four interleaved runs each: four-wave 5.10 / 5.68 ms, dec0_0 on one wave 5.04 / 5.62, encoder 0 as
well 4.98 / 5.59. Decoder 0's second kernel is not converted: it hands r and the MLP sum (8 KiB of binary16) to its
translated tail through shared memory, which a single wave cannot hold without losing the 4 KiB rate.

Shared memory, later (2026-10-18, RX 9070 XT, RADV, 1280x720 -> 2560x1440, `d4r-rr` with `D4R_RR_TIME_REPEAT=40`
and `D4R_ENGINE_RR_STAGE_STAMPS=1`, five interleaved runs; every image of six temporal frames byte-identical to the
previous model, also at 1114x626 -> 1080p, 853x480 -> 1440p and with `D4R_RR_TEST_ROBUST2=1`). ACO sizes a shader's
registers to the waves its shared memory leaves room for (enc1 needed 97 vector registers before scheduling and got
240, dec0_1 79 and 192), so a stage that declares less shared memory also gets fewer registers and more waves:
- **Encoder 0 in one shader again at level 3.** With the one-wave block nothing waits while the gather runs, so
  `gen_rr2_enc0.py --one-wave` without `--split` runs the gather, the block and the tail in one 32-thread group and
  hands the features over in shared memory: XF shares its 4 KiB with the block's XT and KVS (`rr2_w.comp`
  `RR2_XF_SHARED`; every feature tile is in registers before the block first writes either). No gather pass, no
  record in the arena (76 bytes a render pixel, of which 36 MB lay past the arena at 720p). The gather now runs at the
  block's 6 waves per SIMD instead of 16 as a pass of its own, which takes back most of the saved traffic:
  enc0g + enc0 798 -> 772 us. `D4R_RR2_SPLIT=1` builds the gather pass.
- **enc1's merge input shares its 8 KiB with XT and KVS** (`rr2_m.comp` KIND 1): 17 -> 9 KiB of shared memory, 240 ->
  144 vector registers, 6 -> 10 waves per SIMD: 286 -> 266 us.
- **dec0_1 hands r and the MLP sum to its tail in XT and KVS' place** (KIND 5, one window per workgroup): written as
  binary16 pairs of natural channels after a barrier, read in those pairs by the tail; 14 -> 8 KiB, 192 -> 120 vector
  registers, 8 -> 12 waves per SIMD: 660 -> 623 us.
- **Lighter conversion guards in the hybrid stages** (enc0, dec0_1; `ptx2rr.py --exact-convert`): their modules go
  through `spv_exact_convert.py`, whose NoContraction on every conversion already stops the fold of
  `float(float16_t(x))`, so a `cvt.rn.f16.f32` keeps only the opaque zero on its f32 bits (against the fused
  multiply-convert, which `precise` alone does not stop: without it ACO emits `v_fma_mixlo_f16` there). In the ISA the
  same number of `v_cvt_f16_f32` / `v_cvt_f32_f16` as before and no `v_fma_mix*_f16`; 224 / 153 fewer instructions:
  enc0 747 -> 729 us, dec0_1 598 -> 592. The output passes (post, down, downp, pooled) now go through
  `spv_exact_convert.py` too and take the same guard. Their ISA already uses `v_fma_mix*_f16` for plain conversions,
  so the check there is the whole instruction mix: only `v_and_or_b32` / `v_or` / `v_perm` / shifts go (post -16
  VALU, down -9, downp -73), every `v_fma_f32` / `v_mul_f32` / `v_add_f32` / conversion / `v_exp` / `v_log` count is
  unchanged (post moves 5 of its 57 mixes from the low to the high half of their registers). Output 673 -> 665 us.
- **dec0_1's translated tail on two waves** (`gen_rr2_enc0.py --tail-waves 2`). The tail ran on wave 0 while the
  block's other three waves had nothing left to do; it is a prologue (y = r + MLP * s2 and the shuffles that give each
  lane its two pixels) and then the same per-pixel code twice. Each of two waves now runs its own copy of the tail with
  only one pixel's stores (the first six, the last six), and the compiler drops what a copy's stores do not need:
  every stored value comes from the same instructions on the same values. 120 -> 96 vector registers, 12 -> 16 waves
  per SIMD; dec0_1 565 -> 474 us. Four waves (each pixel's stores split in two) is slower, 529 us: the per-pixel work
  the parts share is then done twice. `D4R_RR2_TAIL_WAVES=1` builds the single-wave tail.
- **One unsigned comparison per axis in the translation's bounds-checked fetch** (`ptx2rr.py` `fetch0`: a negative
  coordinate is above every size): 231 instructions fewer in enc0, whose gather fetches 102 times, its scalar spills
  200 -> 176; enc0 745 -> 738 us.
- **Hardware saturation for the FP8 conversions of f32 values** (`rr_exact.glsl` `FINITE`). RADV lowers a saturating
  conversion (`saturatedConvertEXT`) as a NaN-preserving clamp per element (`v_minimummaximum_f32`) and then the
  conversion, because the hardware's FP16_OVFL saturation turns infinities into NaN instead of +-448 (ACO's comment;
  confirmed on all 2^32 f32 inputs, RX 9070 XT: FP16_OVFL is identical to the clamp for every finite value and NaN,
  only +-Inf differ). The values converted here never are infinite: sums of FP8 products (each at most 448^2) and
  binary16 biases stay far below 3.4e38. Written as "an infinity becomes NaN, then saturate", the conversion is the
  pattern `nir_opt_algebraic` turns into `f2e4m3fn_satfn` (vkd3d-proton's; Mesa >= 25.2, `has_f2e4m3fn_satfn`):
  the hardware conversion with FP16_OVFL and no clamp. 168 - 232 clamps fewer per network stage (enc0 960 -> 128);
  frame 3.92 -> 3.79 ms, every stage faster (dec0_1 607 -> 576 us, enc0 752 -> 727, enc1 262 -> 247). The binary16
  conversions (the decoders' first kernels' input) keep the clamp: NIR moves the infinity test through the widening
  and the pattern no longer matches. `RR_SAT_CLAMP=1` builds the old form.
Frame 4.09 -> 3.84 ms. Tried without gain: the decoders' first kernels' expansion in KVS (16 /
12 / 8 KiB, but the deep levels have too few windows for the waves to matter and the extra barrier cost 4 - 5 us
each), all twelve tiles of a 1440p frame in one group (sixteen tile slots: the same), the output passes on 32-wide
waves (0.67 -> 0.82 ms) and unrolled head / branch loops in the 160-channel stages (+10 - 20 us each). Per stage
(us): enc0 772, enc1 266, enc2 137, dec1_0 246, dec1_1 208, dec0_0 405, dec0_1 623, levels 3 - 5 (eight stages) 395,
output 670 (post 490, down 180). Levels 4 and 5 have 15 - 77 windows for 64 compute units.
Eight waves per window for those levels (each token tile's channel tiles split over two waves, the hidden layer's
two tiles one each, the halves meeting in shared memory; byte-identical) does not pay: enc4 / dec4_0 / dec4_1 46 / 45 /
36 -> 49 / 45 / 40 us, enc5 35 -> 31, level 3 64 - 66 -> 78 - 86. A workgroup stays on one compute unit, so the idle
units stay idle and the extra waves only add the shared work and barriers.

### Presets D and E against a real NVIDIA card, and level 3 for them (2026-10-14)

NVIDIA's own kernels for presets D and E were replayed on an RTX PRO 4500 Blackwell (drivers 595.91 and 580.178: the
two give identical images; `build/rr/f/nv/pack.py --corpus ~/.cache/d4r-rr-corpus --first luma_convert`, captures
capD / capE3 at 640x360 -> 1280x720, every frame a reset frame). `D4R_RR_RELAXED=3` now also exists for
`build_native_model.sh`: level 2 plus `RR_NATIVE` (`rr_exact.glsl` `ACTQ8`: the MLP activation in f32, two elements
per saturating conversion, no activation table), and from level 2 on the modules are no longer passed through
`spv_exact_convert.py`. Presented image against the NVIDIA card, and the time at 1280x720 -> 4K (reset / next frame):

| build | D, dB | D, ms | E, dB | E, ms |
|---|---|---|---|---|
| exact (identical to the CUDA path through ZLUDA) | 38.5 / 39.0 / 38.5 | 4.4 / 4.7 | 40.0 / 39.9 / 40.0 | 4.34 / 4.54 |
| `D4R_RR_RELAXED=1` | 36.8 / 37.4 / 36.8 | 3.86 / 4.08 | 38.5 / 38.3 / 38.3 | 3.74 / 3.98 |
| `D4R_RR_RELAXED=2` | 36.8 / 37.4 / 36.8 | 3.76 / 3.99 | 38.5 / 38.3 / 38.3 | 3.65 / 3.82 |
| `D4R_RR_RELAXED=3` | 36.7 / 37.4 / 36.7 | 3.65 / 3.95 | 38.5 / 38.3 / 38.3 | 3.51 / 3.70 |

The NVIDIA card runs D at 1280x720 -> 4K in 2.19 / 2.39 ms. Unlike preset F, relaxing costs about 1.6 dB here. What
was measured and not built for D / E (`build/rr/f/nv/dstage.sh`, `dnl.sh`, level 3, 720p):

- shared memory: encoder 1 with 4 / 8 / 16 KiB of extra shared storage takes 287 -> ~305 / ~320 / ~343 us, so the
  stages above the limit (decoder 1 at 21 KiB, decoder 0 at 17 KiB) would gain a few per cent each, under 1% of a
  frame;
- one wave per window: with every weight tile replaced by one tile loaded once (wrong values, the ceiling of sharing
  loads between a window's waves) decoder 0 takes the same time (803-818 us), encoder 1 the same, decoder 1 10% less.
  These windows also have 96 key tokens (3 KiB each for the tokens, the keys and the values at C = 32), which does
  not fit the 4 KiB a single wave can hold at full rate.

### Byte-identical speed-ups (2026-10-19)

RX 9070 XT, RADV 26.2.2, 1280x720 -> 2560x1440 with `D4R_ENGINE_RR_STAGE_STAMPS=1`. Every image of every frame is
byte-identical to the previous model on the replay and stress inputs, at levels 0 to 3.

- `ptx2rr.py`: OCML's tanh (decoder 0's second kernel) in f32 where f64 was not needed. Exp's error term becomes
  Dekker's exact product error; the polynomial below 0.625 stays in f64, because the driver folds away the f32
  rounding between its f64 steps. Identical to the previous function on all 2^32 inputs. dec0_1 475 -> 446 us, frame
  3.568 -> 3.537 ms.
- `exposure_f.comp` PASS 2: the exposure in one pass as NGX does it, each block adding its fixed-point sum to a 64-bit
  total with an atomic add and the last block finishing. It needs `shaderBufferInt64Atomics`, which vkd3d-proton
  enables; the shim reads the device's features through `GetDeviceFeatures`. Without it, or with a model that has no
  `exposure_f2.spv`, the two passes run (`D4R_RR_TEST_NO_INT64_ATOMICS=1` forces them). Exposure 25.2 -> 14.9 us.

### Opt-in builds that became defaults (2026-10-20)

Against the previous default build, presented image, RGB, peak 1, 1280x720 -> 1440p. For scale, the exact build
(`D4R_RR_RELAXED=0 D4R_RR2_POOLED=rgba32f`) is 43.0 - 44.4 dB from that build.

- Unguarded translated code (the default at level 3 since 2026-10-20; `D4R_RR2_UNGUARDED=0` keeps the guards):
  encoder 0's and decoder 0's per-pixel parts and the output
  passes without their exactness guards: no `precise`, no opaque zero on binary16 stores, no `spv_exact_convert.py`.
  56.0 dB on the reset frame, then 43.0 - 45.6 dB, as far from the exact build as the previous default was. Frame
  3.526 -> 3.501 ms.
- `D4R_RR2_POOLED=rgb10_a2` (since 2026-10-13, the default since 2026-10-20; `rgba32f` builds the reference form):
  the pooled histories as UNORM 10:10:10:2 images, 31 MB less at 1440p.
  Identical at 1280x720 -> 1440p and on reset frames; 43.7 - 47.2 dB on the history frames at 1114x626 -> 1080p and
  28.8 dB with random motion. Frame 3.526 -> 3.484 ms, 3.456 with the unguarded code as well.

### Byte-identical speed-ups (2026-10-20)

RX 9070 XT, RADV 26.2.2 with the ACO `v_mad_u32_u16` patch, 1280x720 -> 2560x1440 game-like input, median of 6
interleaved runs. Every image of every frame is byte-identical to the previous model on the replay, odd-size, 540p,
stress and game-like inputs, at levels 0 and 3 and in the `D4R_RR2_UNGUARDED=1 D4R_RR2_POOLED=rgb10_a2` build.

| Change | Stages (us) | Frame (ms) |
|---|---|---|
| `rr_exact.glsl` Q8PAIRS, `rr2_w.comp`: the pair widened into a variable, so binary16 pairs also reach the hardware saturation | dec0_0 372 -> 334, enc0 717 -> 695 | 3.549 -> 3.487 |
| `rr2_m.comp` KIND 2: x computed once per lane, FP8 decodes and binary16 multiply-adds in pairs | dec1_0 223 -> 203, dec2_0 118 -> 108, dec3_0 62 -> 55, dec4_0 43 -> 39 | 3.487 -> 3.442 |
| `rr2_m.comp` merge: a wave's channel tiles side by side, sharing each step's loads | enc1 236 -> 228, enc2 123 -> 112, enc3 68 -> 61, enc4 44 -> 40 | 3.442 -> 3.413 |
| `rr2_m.comp` residual step: FP8 decodes and conversions in pairs | dec0_1 436 -> 430 | 3.413 -> 3.401 |
| `rr2_pack.py` float32 copies of the biases (tables as column-major tiles), read by both block shaders | dec0_1 430 -> 412, dec1_1 184 -> 171, enc1 227 -> 218, dec1_0 202 -> 194 | 3.401 -> 3.329 |

The build that is now the default: 3.455 -> 3.215 ms, and 3.188 with encoder 0's gather as a pass of its own, which
the unguarded default selects (`compile_rr2.py`; encoder 0 650 -> 623 us, the same images; in the guarded build
the two are equally fast).

Tried, no gain: the translated code's texel fetch without its branch (enc0 within noise); the next MLP branch's
weights or biases prefetched (into L2, or into registers one branch ahead); two MLP branches per loop iteration; V^T
kept in registers in `rr2_w.comp`; weight loads through buffer device addresses. Not done: decoder 0's second kernel
on one wave, whose tail would need r and the MLP sum in 8 KiB of shared memory per wave (done later the same day,
below, by handing them over in two halves).

### Byte-identical speed-ups (2026-10-20, later)

Same setup as above (and the RADV buffer-offset pass), opt-in build (`D4R_RR2_UNGUARDED=1 D4R_RR2_POOLED=rgb10_a2`),
`D4R_RR_TEST_ROBUST2=1`. Every image of every frame is byte-identical to the previous model on the replay, odd-size,
540p, stress and game-like inputs, with and without `D4R_RR_TEST_ROBUST2=1`: preset F at levels 0 and 3 and in the
opt-in build, presets D and E at levels 0 and 3 (they share the translated code), and the RDNA3 builds of D and E
(`D4R_RR_GFX11=1`, run on the RX 9070 XT).

| Change | Stages (us) | Frame (ms) |
|---|---|---|
| `ptx2rr.py` fetch0: with `robustImageAccess2` the translated stages' texel fetches skip their bounds test | enc0 616 -> 600, output passes 607 -> 590 | 3.111 -> 3.082 |
| `rr2_w.comp` KIND 5: decoder 0's second kernel on one wave (opt-in build only, see below) | dec0_1 393 -> 374 | 3.081 -> 3.062 |
| `rr2_m.comp`, `rr2_w.comp`: FP8 pairs decoded as halves of `unpack8`, without byte permutes | dec0_0 324 -> 316, dec1_0 183 -> 181 | 3.057 -> 3.041 |
| `rr2_m.comp`: the stage's output tokens converted two at a time | dec1_1 160 -> 156, enc1 213 -> 211 | 3.049 -> 3.029 |
| `ptx2rr.py`: 64-bit address arithmetic in 32 bits where only the low 32 bits are used | output passes 592 -> 575 | 3.035 -> 3.014 |

The steps above were measured one after another before the RDNA2 preset E / F change was merged. All of it, against
models built from the branch these changes sit on, 6 interleaved runs: opt-in build 3.128 -> 3.024 ms, default build
3.256 -> 3.147, exact build 4.080 -> 4.004 (4 runs); presets D / E at level 3 3.383 -> 3.320 / 3.390 -> 3.287.

- The fetch: the reference reads an integer texel outside the image as zero. With `robustImageAccess2` the image load
  already returns zero there, except for a w component the format lacks, which comes back as one. The runtime sets
  specialization constant 0 (`RR_ROBUST_IMAGES`) when the device has the feature enabled: the shim reads it from
  vkd3d-proton's device features (vkd3d-proton requires robustness2; the shim logs `robustImageAccess2 enabled`),
  `d4r-rr` with `D4R_RR_TEST_ROBUST2=1`. Otherwise, and with models built before this change, nothing changes.
- Decoder 0's second kernel on one wave hands r and the MLP sum to the translated tail through 4 KiB of shared
  memory, half the window's tokens at a time. Only the opt-in build uses it: in the guarded build the tail's `kc[2]`
  comes out as a NaN of the other sign on the stress inputs, because the sign of a product of two NaNs follows the
  operand order the compiler picks (`ACO_DEBUG=nosched` alone flips it there). `D4R_RR2_ONE_WAVE_DEC0_1=1` / `0`
  forces either.
- Presets D and E at level 3 show the same on the stress inputs: with `ACO_DEBUG=nosched` alone 102 and 100 of 186
  images change at 720p, and every value that differs is a NaN on both sides, with the other sign. Comparisons that
  change the compiler or driver therefore hash those images with every NaN replaced by one canonical NaN (0 of 186
  then differ); the comparisons above are identical without it.

Tried, no gain: the output passes as wave32 (post 112 us slower); r of half the token slices kept in shared memory
during the MLP, for fewer registers (same time); the image size read once per texture in the gather instead of per
fetch (within noise); a driver without the RDNA4 hazard waits for SGPRs read by the VALU saves 2 - 4 us per
translated stage, so avoiding them is not worth much. Not done: the output filter's refinement loop runs until the
slowest lane of a wave converges; regrouping pixels by iteration count would save an estimated 25 us.

### Where the time of the top stages goes, and the encoder-0 order (2026-10-20, evening)

Same setup (RX 9070 XT, RADV 26.2.2 with the round-5b GL2 change, 1280x720 -> 2560x1440 game-like input, opt-in
build, `D4R_RR_TEST_ROBUST2=1`, median of 3 - 6 interleaved runs). Before writing more peepholes, each of the top
stages was timed with stub variants of its shader: stores behind an opaque false, texel fetches or arena reads
replaced by cheap arithmetic, every weight tile read from one hot 4 KiB (distinct loads, no CSE), the cubic table
lookups replaced by the value, one head / no MLP branches, extra shared memory to cut the waves per SIMD, the
shader returning at once. The variants' outputs are wrong, so only the stage's own time counts, and only for stages
without data-dependent loops (the native blocks, the gather, the decoder tail; not post and down, whose refinement
loops run longer or shorter with the data they get, see below).

| Stage (us) | Without stores | Without arena / fetches | Weights hot | Without cubic lookups | Half the waves per SIMD | Everything but the arithmetic removed |
|---|---|---|---|---|---|---|
| dec0_1 369 | 366 | 337 (tokens) | 344 | 332 | 484 (4 waves) / 882 (2) | 308 |
| dec0_0 317 | 315 | 308 | 308 | 285 | - | 282 |
| enc0 block ~339 | ~317 | ~242 (the gather's record) | ~318 | ~308 | +118 (3 waves) | ~215 |
| enc0 gather ~255 | ~208 | ~161 (fetches) | - | - | +334 (2 waves) | ~150 |

So decoder 0's two kernels and encoder 0's block are bound by their own VALU and WMMA issue (83 - 89% of their time
is left with every memory access removed; a WMMA 16x16x16 FP8 costs about 16 cycles of the SIMD on this GPU, a
quarter of the block's instruction time), and lose about a third more with half the waves: more waves would help,
but shared memory (4 KiB a wave, 32 KiB a SIMD) and 192 - 240 registers cap them at 6 - 8. The gather is VALU and
SALU issue plus its fetches. The record encoder 0's block reads (73 MB a frame, written by the gather just before)
is the one stage that reads from DRAM: the block's windows now run in the reverse of the gather's dispatch order
(`gen_rr2_enc0.py --split block`), so that the records written last are read first while they are still in the
64 MB last-level cache: encoder 0 597 -> 576 us, frame 2.985 -> 2.961 ms (6 interleaved runs). The levels' kernels are unchanged in what
they compute; every image of every frame is byte-identical to the previous model on the 7 input sets, with and
without `D4R_RR_TEST_ROBUST2=1`, for the opt-in, default and exact builds (presets D and E do not use this
generator; their models are file-identical).

The deep levels (3 - 5, 274 us in 7 dispatches) are not latency-bound in the way assumed: a dispatch of an empty
shader costs 1 us, and a window's four waves run on one CU with nothing else to do, so level 5's 24 windows use 24
of 64 CUs at full tilt (its time is the same at 540p, 720p and 1080p). The output filter (post, ~400 us) is where
the data matters: its refinement loop runs until a wave's slowest lane converges, and a stub that changes the
half-size history it reads next frame changes its time by up to 100 us, which looked like a store cost until the
same stores to a buffer, copied to the image by a pass of their own, cost exactly as much once the history was
right.

Tried, no gain (all byte-identical where they ran): the cubic numerator computed in packed binary16 arithmetic
instead of looked up (identical on all 65536 inputs; dec0_1 +7 us, dec0_0 +3, enc0 +9: the gathers hit the L0
cache); the deep levels with their heads and MLP branches on 2 - 4 groups of waves, folded in order through shared
memory (enc3 55 -> 77 / 103 us: one workgroup stays on one CU); 2 - 4 windows per workgroup for the same reason
(enc5 29 -> 102 us); the half-size history stored to the arena by the filter and written to its image by a copy
pass (+23 us for the pass, the filter unchanged); encoder 0's gather and block in 4 - 32 bands of rows with the
block of a band next to the gather of the next (-15 us with 4 bands, worse beyond: each band's barrier drains the
GPU); encoder 0 fused into one shader again (607 us against 575). Decoder 0's two kernels in reverse order gain
nothing (their tokens are still cached).

### The deep stages, and windows split over several workgroups (2026-10-20, night)

The card measured since 2026-10-18 is an RX 9070 (56 compute units; the sections above call it a 9070 XT). Same setup
as above, opt-in build. Each window of levels 3 - 5 was timed with the shader clock (start and end per workgroup):

- Between two stages the GPU idles 0.6 - 0.7 us; the barrier is not what costs time. Level 4 and 5 stages start all
  their windows within 0.4 us, and every window takes 28 - 35 us, so such a stage lasts one window. Inside one (enc5):
  input 0.8 us, attention 12.8 (eight heads), MLP 16.9 (20 branches); each head's and each branch's loop iteration
  waits for its weight tiles, with two waves per SIMD and nothing else to hide that.
- Level 3 has more windows than fit at once (252 for 224 places in enc3 and dec3_1, 240 for 168 in dec3_0), so each
  of its stages ends with a second, mostly empty round.

One dispatch for levels 3 - 5 with per-window completion counters (a window starts as soon as the windows it reads
are done) would recover at most 33 us of their 281 (the measured window times replayed with every dependency and
unlimited compute units), nearly all of it in the level-3 decoders, and it would also need its own arena regions
(enc4 writes its merge into encoder 3's input region). Not built.

What is built: a stage with fewer windows than the GPU has compute units runs in four dispatches, so that a window's
heads and MLP branches run on as many workgroups (`rr2_m.comp` `RR2_PHASE`): 1 one head per workgroup, 2 the heads'
out-projections in head order and r, 3 one MLP branch per workgroup, 4 the up-projections in branch order and the
rest of the stage. The values handed over are what the whole shader already keeps (Ob, Nb and Hb as FP8 tiles, r as
it is), through a record per window placed in the arena past the network's regions, so every window computes exactly
the same values. Phases 2 and 4 load the next head's or branch's tiles while the current one's products run.
`compile_rr2.py` builds the phases for enc3, enc4, enc5, dec4_1 and dec3_1 (a `phases` line in `rr.txt`;
`D4R_RR2_PHASES=0` builds none); the runtime uses them for a stage with at most 48 windows (`D4R_RR2_SPLIT_WINDOWS`,
0 for never). Older runtimes ignore the lines.

| Render size | Stages in phases | Stage times (us) | Frame (ms) |
|---|---|---|---|
| 1280x720 | enc5 | enc5 29 -> 19 | 2.954 -> 2.947 |
| 960x540 | enc4, enc5, dec4_1 | enc5 29 -> 17, enc4 31 -> 28, dec4_1 29 -> 23 | 1.721 -> 1.705 |
| 1920x1080 | enc5 | enc5 28 -> 23 | within noise |

Every image of every frame is byte-identical to the previous build on the 7 input sets and at 1280x720 -> 3840x2160,
opt-in, default and exact builds, with and without `D4R_RR_TEST_ROBUST2=1`, and with every stage in phases
(`D4R_RR2_SPLIT_WINDOWS=100000`); the single-stage shaders are unchanged files. Branch 1's Hb handed to branch 0 changes
53 of 90 images.

Tried, no gain: phases at 1280x720 for the stages with 66 - 77 windows (enc4 33 -> 38 us, dec4_1 29 -> 32, level 3
much slower: the extra workgroups only add work there); the next branch's tiles loaded early in the single shader,
with r kept in the arena meanwhile (enc5 29 -> 24, every other deep stage 2 - 9 us slower); WGP mode, which RADV
uses only for shaders without shared memory, forced on every stage in a scratch driver (frame 2.955 -> 3.256 ms).

### What the output filter's refinement steps cost (2026-10-20, night)

The output filter alone (`D4R_ENGINE_RR_SKIP=down,pooled`) takes 384 us at 1280x720 -> 1440p and 844 us at 4K. Its
translated code has no loop besides NGX's progress-counter wait (skipped here): what was called its refinement loop
is a cascade of up to four steps per row of a pixel pair, each five bilinear history samples, written out twice.
Counting active lanes on the game-like input, step 1 runs in 86% of the waves with 64% of their lanes, step 2 97% /
59%, step 3 93% / 56%, step 4 30% / 15%; on the stress input every step runs with about 95% of the lanes. Without
steps 2 - 4 (wrong images) the filter takes 336 / 742 us, so their idle lanes cost at most about 20 / 45 us: that is
all that regrouping pixels by step count could recover, and the rest is per-pixel arithmetic that a native rewrite
would have to repeat to stay exact (1564 VALU and 44 samples per 64-wide wave, of which the steps are 40 samples).
Not rewritten.
