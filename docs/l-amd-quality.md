# Preset L: AMD implementation with a 40 dB quality budget

Follow-up: the [RGP-guided outer-block redesign](l-outer-rdna4.md) replaces
shared transposes with register ownership, folds selected normalization gains,
and changes the skip layout. It also measures a separate feature dispatch and
retains the faster fused path.

The `fast` AMD profile runs the measured 1080p→4K L pass in **3.850 ms versus
4.046 ms** for the optimized relaxed control on an RX 9070 XT: **4.86% faster**.
It meets the requested **40 dB minimum on every one of 42 scored reference
frames**, with a worst result of **40.97 dB**. Numerical equality is not a
selection criterion.

The ready-to-use local model is `build/l-amd-quality-20261029/model-amd`.
Build and benchmark tools are [build_l_amd.py](../tools/build_l_amd.py) and
[bench_l_amd.py](../tools/bench_l_amd.py). The ordinary model compiler's defaults
are unchanged; this profile produces a separate model for the Vulkan/vkhip
backend. Measurements are for gfx1201 / RADV Mesa 26.2.5 at the default 330 W
cap, not a claim about all AMD GPUs or native Windows performance.

## What changes for AMD

The inner network uses **RDNA4 hardware-saturating FP8 conversions**.
`SWIN_AMD_SAT8` sets the executing wave's `MODE.FP16_OVFL` flag and removes
per-element clamps around ordinary FP8 conversions. Explicit smaller limits in
merge/expansion remain. AMD documents that this mode saturates FP32→FP8
overflow; it also changes binary16 overflow behavior, so final temporal PSNR
is checked. See section 7.6.2 of the
[RDNA4 ISA reference](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).
The final code object contains the mode-setting instruction followed by the
packed FP8 conversions.

This removes approximately 10–13% of static VALU instructions across the five
native code objects. The count includes their preparation code and is not a
dynamic instruction ratio or speedup estimate. The compiled decoder-1 register
count drops to 106 VGPRs with a 12-wave occupancy limit; decoder-2 uses 166
VGPRs with a 9-wave limit. These resource changes are judged by the measured
runtime, rather than treated as performance guarantees.

The full-resolution outer blocks change arithmetic and dataflow more broadly:

- `M_AMD_NORM`: sum squares in FP32 in the matrix fragment's natural channel
  layout and reduce once across lane halves, dropping the compatibility FP16
  reduction tree.
- `M_AMD_SOFTMAX`: use native exponentials and a cross-half FP32 reduction
  instead of reproducing the packed binary16 polynomial/exponent-bit trick.
- `M_AMD_PV8`: quantize probability and value matrices to FP8, transpose V in a
  2 KiB FP8 buffer, and run their product on FP8 WMMA. Probability is scaled by
  16 before conversion and the product divided by 16 afterwards. The original
  path materializes FP16 V and uses FP16 WMMA.
- `M_AMD_GELU`: use `x * clamp(0.28125*x + 0.5, 0, 1)` in the outer MLP,
  replacing its longer activation polynomial. This is deliberately confined to
  the two outer blocks in the selected profile.
- `M_AMD_HEAD`: keep complete pixel ownership across the wave, pass values
  directly in FP32, use native exponential sigmoid/tanh arithmetic, and let
  image stores perform normalized-format conversion. The binary16 rounding,
  explicit normalized-code rounding and shared output transpose are removed.

The [earlier head-column selection and direct expansion loads](l-amd-redesign.md)
are retained. The model weights and tensor interfaces remain compatible with
the existing runtime. The changed activation, softmax and quantization steps
need quality validation; this is not a retrained network.

## Paired performance

Three rotated/reversed interleaved trials per model, 48 warmup evaluations and
100 measured evaluations per trial, official-probe patterned inputs and four
jitters. Whole GPU evaluation time is the median of trial medians. Component
medians do not necessarily sum to the whole median.

| Implementation | Whole pass (ms) | Outer input (us) | Inner network (us) | Outer decoder (us) |
|---|---:|---:|---:|---:|
| Optimized relaxed control | 4.046 | 771.8 | 1647.4 | 658.6 |
| Earlier head-layout specialization | 3.978 | 769.6 | 1658.8 | 596.8 |
| AMD quality-budget profile | **3.850** | **756.3** | **1583.1** | **539.6** |

The AMD profile is 3.23% faster than the earlier layout specialization. Its
outer decoder is 18.07% faster than the control. The reconstruction and final
downsample remain the existing implementations. This does not establish
NVIDIA parity; it is a measured implementation improvement within the tested
quality budget.

## Reference PSNR and temporal gate

Metric: fixed peak 1, linear RGB, unclipped values, alpha excluded, FP64 MSE
accumulation. Every scored frame must reach 40 dB and have finite RGB. Pooled
PSNR is calculated from pooled MSE, not by averaging dB. The two NVIDIA/ZLUDA
frames exclude the known harness watermark rectangle; other frames use full RGB.

| Reference sequence | Frames | Control pooled PSNR | AMD pooled PSNR | AMD worst frame |
|---|---:|---:|---:|---:|
| Exact d4r D3D12 model, moving HDR 1080p→4K | 4 | 54.03 | **51.24** | **49.77** |
| Exact d4r D3D12 model, moving HDR 720p→1440p | 4 | 54.07 | **51.78** | **49.79** |
| Exact d4r D3D12 model, 32-frame temporal 128x96→384x288 | 32 | 50.02 | **45.59** | **40.97** |
| NVIDIA PTX through ZLUDA, captured 856x480→1440p | 2 | 45.61 | **42.17** | **42.07** |

The exact d4r D3D12 implementation is a numerical model reference, not NVIDIA
hardware or a ground-truth high-resolution render. The last row uses actual
NVIDIA kernels translated by ZLUDA. These fixtures are diagnostic content, not
an exhaustive game-quality test. Passing 42 frames does not guarantee 40 dB
on arbitrary content, exposure changes, LDR or application lifecycle events.

The 32-frame sequence rejected faster configurations that looked adequate on
four patterned frames. Hard GELU throughout the inner network fell below 40 dB;
the selected profile keeps its existing inner activation. A bilinear history
variant reduced reconstruction time but fell to 39.37 dB in the selected outer
configuration, so the accepted model retains bicubic history. Alternative hard
GELU slopes and the full native FP8-attention inner pipeline also failed the
per-frame gate. Their flags remain experimental and are not used by `fast`.

## Correction to the old NVIDIA/ZLUDA comparison

The earlier approximately 29.6 dB result used the wrong temporal alignment.
`build/l-vkhip-20261028/capture/run.log` explicitly records the asynchronous
displayed-result indices: saved frame 1 shows no result, frame 2 shows result 0,
and frame 3 shows result 1. Frame 4 repeats result 1 after feature retirement.
Thus captured input 0 must be compared with `out.raw.frame2`, input 1 with
`out.raw.frame3`, and input 2 has no saved completed reference.

The aligned control scores 45.51 / 45.70 dB, and the AMD profile scores
42.07 / 42.27 dB. Reference selection follows those logged indices, not a
search for the highest PSNR. The historical misaligned scores are marked
invalid in `sr-l-vector-work.json` and must not be used to infer a quality
limitation outside the outer blocks.

## Build and validate

From a current compiled L model with its FP32 bias tiles and native HIP layers:

```sh
python3 tools/build_l_amd.py CURRENT_L_MODEL AMD_L_MODEL \
  --profile fast --build-native --arch gfx1201
```

This compiles all five native HIP code objects with `SWIN_AMD_SAT8`, copies the
source model, appends the selected head weight columns, builds both outer-stage
plane layouts and updates file hashes. The destination and sibling native-build
directory must not exist. ROCm and glslangValidator are required. Alternatively
supply a separately compiled set with `--native-layers DIRECTORY`.

`--profile balanced --build-native` keeps the longer outer activation and
compatibility softmax/PV precision, while enabling natural FP32 normalization,
the native head and hardware FP8 saturation. The additional named profiles,
custom defines and `--history-bilinear` expose experiments; they do not inherit
the `fast` profile's measured quality acceptance.

Use the resulting model with the `vkhip` runner, or select it with
`D4R_ENGINE_MODEL_L_DIR` and `D4R_ENGINE_VKHIP=1` in the existing engine setup.
For quality, provide explicit input/reference associations in a case manifest:

```json
{
  "render": [1920, 1080],
  "output": [3840, 2160],
  "frames": [{
    "color": "0-color.bin", "motion": "0-motion.bin", "depth": "0-depth.bin",
    "jitter": [0, 0], "reference": "reference-0.rgba16f"
  }]
}
```

Paths resolve relative to the manifest. The runner resets on the first frame
and keeps history for later frames. Omit `reference` only for explicitly
unscored inputs. `exclude_rects` can specify known overlays; it is recorded in
the result. Do not silently mask nonfinite pixels or clip HDR to pass the gate.

```sh
python3 tools/bench_l_amd.py --runner /path/to/d4r-m \
  --case CASE.json --model control=CURRENT_L_MODEL --model amd=AMD_L_MODEL \
  --out RESULTS --trials 3 --min-psnr 40
# Quality-only temporal validation: --trials 1 --warmup-passes 0 --passes 1
```

The tool reports `meets_minimum_psnr` for every model; expected failing
experiments still produce a complete report. Timing and quality must both be
reviewed before selecting a profile for different weights or input sequences.

All measured data, per-frame scores, rejected experiments and model hashes are
in [sr-l-amd-quality.json](benchmarks/gpu-limits-20261029/sr-l-amd-quality.json).
Local models, raw outputs and logs remain in `build/l-amd-quality-20261029/`;
they are not redistributed in the summary.
