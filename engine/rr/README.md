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
