# Preset L (rrlite, unfolded) - notes

Preset L (DLSS 4.5, render preset 12, NVIDIA's model for Ultra Performance) is preset M's pipeline (M_NOTES.md) with
three stages of its own. NGX runs the kernels without the `_folded` suffix: `rrlite_enc0_4x4_mv{hi,lo}_{hdr,ldr}`,
`rrlite_dec0_4x4` and `rrlite_post_3_{1,2}_mv{hi,lo}_{hdr,ldr}`. The exposure kernels, the ten Swin layers (with L's
own weights) and the downsample are M's. Sources: harness captures (`build/engine-work/lcap`, 640x360 -> 1920x1080;
`lcapq`, 1280x720 -> 1920x1080; `lcapldr`, LDR), the translated kernels (`vulkan/ptx2glsl.py` + `condense.py`, for
reading only) and d4r's native HIP parts of L's texture kernels (`kernels/tex/enc0l_tail.hip`, `dec0l_block.hip`).

## Input stage (`rrlite_enc0_4x4_*`; layer_m.comp KIND 3)
Grid 8x8-pixel blocks, block origin 8 b - (sx, sy); the shift alternates per frame like the layers': (2, 2), (0, 2).
1. The 32 FP8 feature slots per pixel, exactly as M's input stage (`enc0_features.glsl`, M_NOTES.md), including the
   mirrored pixels outside the image (their dither uses the unmirrored |position|).
2. Embedding, the shape of an MLP chunk: z = W1 f + b1 (K 32 -> 32), q8(relu(z)), x0 = W2 q8(relu(z)) + b2
   (32 -> 32, f16). Weights at 4096 of the allocation (after the 4096-byte dither table): W1 `woff_table(4096, 512,
   512, 32, 32)`, b1 at 5120, W2 at 5184 (rows in pair order), b2 at 6208 (pair order).
3. A Swin block of M's shape at pixel resolution: C 32, 2 heads (32 attention channels each), 4x4-pixel windows,
   4 MLP chunks; weights at 6272 laid out as a network layer (`swin_model.layout(32, 2)`).
4. Its FP8 output at every pixel of the 32-aligned pixel grid (P16 = render size rounded up to 32): the expansion's
   skip input, `[gh][gw][32]`.
5. Patch merge of 2x2 pixels (K 128) to 64 channels, clamped to +-2 pi as f16 before the FP8 encoding: the network's
   input tokens, `[gh/2][gw/2][64]`.

## Expansion stage (`rrlite_dec0_4x4`; layer_m.comp KIND 4)
Shift (0, 0) / (2, 0) alternating. The patch expand of the network's 64-channel output (64 -> 4 x 32), x0 =
f16(q8(expand) + skip), the same C 32 block, then the per-pixel head on the block's f16 output (no FP8 rounding).
The head reads NGX's pair-order columns j = 0, 1, 6 (gates), 2-5 (feature), 7-9 (mix), 10-12 (covariance a, b,
rho), which are natural channels 0, 1, 12; 4, 5, 8, 9; 13, 16, 17; 20, 21, 24. These are M's columns without M's
four feature gates:
- feature (SNORM8x4): tanh(v) - the poly below |t| 0.6, 1 - 2 / (e^2t + 1) above, as M's rho - with **no history**:
  L feeds the previous feature back only through the input stage's slots.
- gates (UNORM8x4): f16(sigmoid(v)), alpha 0. mix: softmax of three. covariance: a = sigma f16(sigmoid(clamp(v, +-20))),
  b likewise, (a^2, b^2, a b tanh(rho v)).
The f16 rounding of the sigmoid has to go through round_half.glsl: RADV folds `float(float16_t(x))`.

## Reconstruction (`rrlite_post_3_*`; post_m.comp PRESET_L)
M's body with two differences:
- the anisotropic Gaussians cover 4x4 render texels, offsets -1 .. +2 around floor(position - 0.5), instead of 3x3.
  The engine's tiles are one texel wider per tier: 11 (reconstruction/render >= 3), 12 (compact), 16 (post_3_2);
- the history is blended in linear light: hist = e1 (b1 h1 + b2 h2), each candidate h = expand(bicubic sample) / e2
  (when e2 > 0), and a candidate off screen or after a reset takes the colour texel itself (M: compress(e2 colour),
  blended compressed, expanded after the blend). Both candidates use the same fallback.

## Constants (identical in every captured mode)
| | M | L |
|---|---:|---:|
| input motion-change scale (enc0 P268) | 9.045066 | 5.137703 |
| input motion-change gain (enc0 P272) | 0.223568 | 0.082259 |
| expansion sigma, rho (dec0 P160, P164) | 8, 0.078633 | 8, 0.097860 |
| first gate gain (post P120) | 2.753011 | 2.979623 |

compile_m.py stores L's five in offsets.bin (`D4RL0001`), read from the capture.

## Found on the way (shared with M)
- **Merge clamp.** NVIDIA's encoders (enc1, enc2, L's enc0) clamp the patch merge to +-2 pi before the FP8 encoding
  (kernels/m `merge_clamp`); layer_m.comp did not. Without it enc1's merged output matched 98.6% and enc2's 88.3%.
- **Exposure grid.** NGX measures the luma on an endpoint-aligned grid of half the output, except where the output is
  more than twice the render size on an axis: there half the render size, at least 256 (640x360 -> 1920x1080:
  320x256; 854x480 -> 2560x1440: 427x256; 1280x720 -> 3840x2160: 640x360; presets M and L alike, eleven harness
  geometries). The engine used half the output everywhere, which changed the exposure at Ultra Performance and
  with it every value downstream (39 dB).
- **Downsample weights.** glslang folds `float16_t(<literal>)` toward zero, so the first weight was -0.041321 instead
  of NVIDIA's -0.041351 (cvt.rn of its f32 constant). As bit patterns the downsample is byte-identical (it was
  93.8% / 71 dB on every capture, M's included).
- **Network arithmetic.** With TEXL (ZLUDA's lowering: C enters the first WMMA of a k32 step, natural-order norm
  trees, ZLUDA's P V key slots, hardware rsq) the ten layers chain byte-identically against NVIDIA's translated
  kernels; the native accuracy kernels' K32_MIX order drifts to about 40 dB at dec1 after the tube.

## Results (RX 9070 XT, RADV, DLSS 310.7)
Stage tests (`test_l.py enc0|dec0`, `test_post_m.py` with `POST_KERNEL`, `test_down_m.py`, `test_m.py --chain`) on
the three captures, every captured frame: 0 differing values. Whole frames:
- `d4r-m` on the three captured frames (640x360 -> 1920x1080): byte-identical to NGX's downsample output.
- D3D12 harness through the shim, engine (Vulkan network) against the CUDA backend running NVIDIA's translated
  kernels, five frames each (the CUDA path presents one frame later): byte-identical at 640x360 -> 1920x1080 HDR and
  LDR. At 1280x720 -> 1920x1080, 960x540 -> 1920x1080 and 1280x720 -> 3840x2160 one frame each differs in 2,615,
  6,079 and 4,207 values (96-122 dB) and the next frame is identical again: an output-only knife edge (the packed
  11-11-10 store or the downsample), not state. Both paths are deterministic run to run.
- With d4r's native HIP layers (`compile_m.py --hip`, accuracy build) the network is K32_MIX: 48, 42, 40 dB on the
  three captured frames.

GPU time, `d4r-m`, 1280x720 -> 3840x2160 (stand-in frames), median of 60: 3.79 ms with the Vulkan network (exposure
0.01, input 0.67, network 1.44, expansion 0.54, reconstruction 0.92, downsample 0.22), 3.57 ms with the HIP layers
(network 1.12). The CUDA backend's native L tails measure 0.63 ms (enc0) and 0.54 ms (dec0) at the same render size.
