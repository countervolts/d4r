# Preset L: AMD decoder redesign, 2026-10-29

This records the earlier layout experiment. The subsequent
[AMD quality-budget implementation](l-amd-quality.md) allows numerical changes
and selects performance using a 40 dB per-frame reference PSNR threshold;
bit equality is not its acceptance criterion.

An opt-in decoder specialization improves the measured full L pass from
3.991 to 3.928 ms on the RX 9070 XT. The final network decoder (`dec0_l`,
reported as `expansion` by the runner) improves from 648 to 586 us, or 9.57%.
This is a 1.57% whole-pass improvement, not NVIDIA performance parity.

The control uses the current optimized relaxed L model: `WPW=2`,
`M_RESIDUAL_F32`, `M_F32ACT`, `M_RES32`, `M_PK16`, `M_BIAS32`, `M_LDAU`,
with the existing native vkhip inner stages. Defaults are unchanged.

## Register ownership and useful output columns

The last 32-channel MLP output feeds only 13 reconstruction inputs. The builder
copies those original FP8 weight columns into a 16-channel tile, preserving
their bytes and K accumulation order. It also selects their residuals. Padding
columns duplicate channel zero. This removes one of the two final output tiles
(eight WMMA instructions per window, or sixteen per wave at `WPW=2`).

Simply doing less matrix work was slower: rounding and half-active head code
consumed the saving. The successful `M_L_HEAD16_WAVE` layout assigns the two
windows in a wave to different lane halves, so every lane executes one complete
pixel's head. Four exchanges of packed binary16 pairs transfer the required
gate/feature and mixture/covariance values. The shared-memory output transpose
and its synchronization disappear. Attention V retains its own shared buffer.

The final variant also uses `M_L_EXPAND_DIRECT`: load the low-resolution FP8
expansion operands directly into their matrix-fragment layout and reuse them
across the wave's windows, removing the staging copy and barrier.

The original head rounds through binary16. An ordinary register-only cast can
be folded away; `packHalf2x16` can use a different rounding mode. Explicit
`float16_t` conversion plus a bitcast, exchange, and unpack preserves the
boundary without the initial expensive software integer rounding sequence.

## Paired measurements

1920x1080 to 3840x2160, RADV Mesa 26.2.5, gfx1201, default 330 W power cap.
Three interleaved trials per experiment, 48 warmup evaluations and 100 timed
evaluations per trial. Values below are medians of trial medians; controls must
be compared within each row, because clocks and other stages vary between runs.

| Variant | Decoder control → variant (us) | Whole pass control → variant (us) | Outcome |
|---|---:|---:|---|
| Selected columns, software rounding | 648.8 → 681.3 | 4027.02 → 4049.44 | Slower |
| Selected columns, packed exchange | 651.8 → 664.7 | 4036.16 → 4047.32 | Slower |
| Packed exchange + direct expansion loads | 651.8 → 636.1 | 4036.16 → 4014.92 | Small saving |
| Complete pixels across wave + direct expansion | 648.0 → 586.0 | 3991.14 → 3928.38 | Best exact candidate |
| FP16 normalized inputs in both outer blocks | 646.1 → 711.2 | 3992.72 → 4134.50 | Slower, changes arithmetic |

The FP16 experiment decodes the original FP8 weights exactly into FP16 and
retains normalized activations in FP16 to avoid FP8 packing. It loses the FP8
matrix throughput advantage. Its four patterned frames have 53.1–54.6 dB RGB
PSNR against the relaxed control (unit peak), not a reference-quality score.
It is an experiment, not an accepted replacement.

## Validation and use

The best candidate is byte-identical to the relaxed control on 15 temporal
frames across four geometries: 1080p→4K, 720p→4K, 360p→1080p, and a captured
moving/jittered 856x480→1440p scene. This is limited sequence validation;
application lifecycle, all HDR/LDR modes and all input values are not covered.
An isolated Vulkan rounding/exchange check matches binary16 round-to-nearest-even
on 1,304,576 values, including all binary16 encodings and neighboring midpoints
(NaN payload differences ignored). Rebuilding all four default outer-stage
SPIRVs produces byte-identical files to the pre-experiment build.

Build a separate model from a **current compiled L model with FP32 bias tiles**:

```sh
python3 tools/build_l_head16.py CURRENT_MODEL HEAD16_MODEL \
  --define=-DM_L_HEAD16_WAVE --define=-DM_L_EXPAND_DIRECT
```

The destination must not exist. The builder copies the source, appends selected
weights in unused metadata slots 44–47, compiles both decoder plane layouts,
and updates model hashes. Do not enable the shader flag on an unmodified model:
the appended weight layout is required. `--half-input` builds the rejected
precision experiment; it is not part of the best candidate.

Trial results and validation hashes:
[sr-l-amd-redesign.json](benchmarks/gpu-limits-20261029/sr-l-amd-redesign.json).
Private inputs, outputs, models and logs remain in `build/l-redesign-20261029/`.

The [FSR4 comparison](fsr4-l-instruction-comparison.md) provides a stronger
network design target. It suggests reducing the outer blocks' normalization,
attention and activation work, and the total matrix budget. That would require
a different network and training/quality validation; a kernel layout change
does not provide those changes by itself.
