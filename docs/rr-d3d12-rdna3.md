# Ray Reconstruction on native D3D12 with RDNA3 (2026-11-10)

Before this, D3D12 games on RDNA3 had no Ray Reconstruction: every D3D12 RR package is built from the gfx12 (FP8 WMMA)
stages, whose pipelines remove an RDNA3 device, so the shim refused them ("needs an RDNA 4 GPU"). Presets D and E now
run on RDNA3 from the RDNA3 packages the Vulkan route already uses, `engine\rr-d-gfx11` and `engine\rr-e-gfx11`
(`D4R_RR_GFX11=1 engine/rr/build_native_model.sh --dll nvngx_dlssd.dll D|E DIR`), with D3D12 shaders added by
`compile_rr_dxil.py`. Preset F has no RDNA3 arithmetic on any route (Linux included) and stays RDNA4-only.

## How

- `rr_hlsl.translate(gfx11=True)` lowers the RR_GFX11 stages (`engine/rr/rr_gfx11.glsl`): f16 A/B operands are 16
  elements per lane (`RRHalf16`), accumulators 8, in gfx11's register layouts: A/B lane l holds row/column l & 15 with
  element i = k i; an accumulator lane holds column l & 15 with element i = row 2i + (l >> 4). The matrix product is
  AGS's `f32_16x16x16_f16` on all eight registers (`rr_wmma.hlsli`). The translator's FP8 passes are skipped; no FP8
  matrix or conversion reaches the shaders.
- AMD's D3D12 driver gives AGS matrices gfx11's hardware layout: a probe (one wave, A/B from known 16x16 matrices
  under that layout, C seeded with 1000 row + column) reads back 255-256 of 256 accumulators exactly; unseeded, the
  products differ from an f64 reference by at most one f32 ulp, the instruction's own non-associative sum
  (rr_gfx11.glsl measured the same on RADV).
- `compile_rr_dxil.py` accepts a gfx11 D/E model (`arithmetic gfx11` in rr.txt): it preprocesses the stages with
  build_native_model.sh's RDNA3 defines (W_WIDE from each weight file), uses the widened f16 weights as they are (no
  transposed FP8 tiles) and writes `arithmetic gfx11` into rr-d3d12.txt. The D3D12 runtime accepts the model's
  `arithmetic gfx11` only with that line and then runs the gfx11 camera arithmetic (`rr_frame_parameters.h`).
- The shim loads `engine\rr-d-gfx11` / `rr-e-gfx11` on an RDNA3 render GPU (`D4R_ENGINE_MODEL_RR_D_GFX11_DIR` /
  `_E_GFX11_DIR` override, `D4R_D3D12_RR_GFX11=0` turns it off), refuses an FP8 package there and a gfx11 package on
  RDNA4.
- Arithmetic: the gfx12 DXIL build's relaxations (`rr_hlsl.relaxed_math`) apply too; `D4R_RR_GFX11_EXACT=1` at build
  time keeps the exact stages.

## Measured (RX 7700 XT, AMD 26.10.1, HAGS off)

| | ms |
|---|---:|
| D3D12 harness, RR D, 1280x720 -> 2560x1440 (relaxed) | 39.7 |
| D3D12 harness, RR E, same | 39.7 |
| Vulkan harness, RR D / E (gfx11 SPIR-V, LLPC) | 27.7 / 23.5 |
| Replay 1024x576 -> 2048x1152, D, D3D12 exact / relaxed | 32.1-33.4 / 27.2-27.5 |
| Replay, same, D, Vulkan (d4r-rr) | 30.7-32.0 |

The two harnesses draw different scenes and time different spans; the replay pair (`d4r-rr` against `d4r-rr-d3d12`,
identical synthetic inputs, `scripts/windows/rr-synthetic-inputs.py`) is the like-for-like comparison: the D3D12 port
is within ~5% of the Vulkan route exact, ~12% faster relaxed. Quality against the Vulkan gfx11 route on 8 synthetic
frames: 61.0-67.0 dB PSNR (exact and relaxed alike, max abs error 0.0088, every value finite).

D3D12 stage split (replay, exact, us): enc0 4800, enc1 2980, enc2 1930, enc3 1900, enc4 1090, enc5 830, dec4 2310, dec3
1790, dec2 2600-3260, dec1 4450, dec0 6100, output 540. The cost is the exact RDNA3 arithmetic itself (ZLUDA's gfx11
FP8 lowering: every FP8 operand widened, two chained f16 WMMAs per k32 step with scaling); Linux/RADV runs the same
SPIR-V about twice as fast (11.4 ms at 853x480 -> 1440p, docs/native-kernels.md), so AMD's Windows compilers, both
LLPC and the D3D12 one, are the gap.

First launch: the driver compiles the D3D12 RR pipelines in about 47 s at NGX init; from its cache the next launch
loads them in 0.6 s.

## Not done

- No in-game check: no RR game is installed on the test machine.
- gfx1100 / gfx1102 / gfx1103: built from the same sources, untested.
- The release packages do not build rr-d-gfx11 / rr-e-gfx11 (packaging change left for review).
