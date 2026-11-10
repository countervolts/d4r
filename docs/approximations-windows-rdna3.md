# Approximations on RDNA3 (Windows, native D3D12)

This page lists where d4r's output on RDNA3 under Windows' native D3D12 engine differs from NVIDIA's arithmetic, and which of those differences are real approximations. It is the Windows counterpart of the linux branch's `docs/approximations.md` and uses the same definitions and reference.

**Short answer:** with the current packages, the default K, M, L and Ray Reconstruction D/E routes on RDNA3 use no approximations. They are not bit-exact either: like the Linux routes, they keep more precision than NVIDIA in some steps. Two things that *were* approximations or narrower than NVIDIA were found and removed while writing this page (M's int8 expansion stage, and binary16 conversions that rounded toward zero); see [What changed](#what-changed). [Open items](#open-items) lists what is not covered.

Everything here was measured on an RX 7700 XT (gfx1101) with AMD's 26.10.1 driver, HAGS off, on 2026-11-10. Other RDNA3 targets run the same sources but were not tested.

## What counts as an approximation

As on Linux, a step is an **approximation** if either:

1. it holds or multiplies a value in a narrower number format than NVIDIA's kernel does at that step, or
2. it computes a different function (the result would differ even with perfect arithmetic).

Everything else is a **widening** (same formula, wider format) or a **rounding change** (same formula and format, a rounding in a different place). Neither is an approximation; neither is bit-exact.

**Reference:** NVIDIA's kernels (the PTX in `nvngx_dlss.dll`) on the same GPU through ZLUDA, with `[Kernels] PreferAccuracy = true`, `NativeKernels = off` and `[Engine] Enabled = false`. **Metric:** RGB PSNR with peak 1, unclipped, squared error pooled over every frame after the first (8-bit output: codes / 255). **Scenes:** `tools/d3d12_dlss_harness.cpp`, 1920x1080 output: motion (1280x720, HDR, flags `0x4b`), jitter (960x540, HDR), LDR (1280x720, RGBA8 colour and output, flags `0x6a`).

## Summary

| Model | Default route on RDNA3 (D3D12) | Inside the network | Stages, input and output | Distance from NVIDIA's kernels |
|---|---|---|---|---|
| **K** | d3dhip: D3D12 stages, the 11 native HIP layers between them | No approximation (widening, rounding changes) | No approximation; binary16 conversions round toward zero ([Open items](#open-items)) | not re-measured on Windows |
| **M** | d3dhip: RDNA3 D3D12 stages, the native HIP Swin layers | No approximation (widening) | No approximation (exact package `D4RML007`) | motion 38.06 dB, jitter 45.86 dB, LDR 38.62 dB |
| **L** | d3dhip: RDNA3 D3D12 stages, the native HIP Swin layers | No approximation (widening) | No approximation (exact package `D4RML007`) | motion 36.93 dB, jitter 52.09 dB, LDR 36.77 dB |
| **RR D / E** | D3D12, the gfx11 packages (`rr-d-gfx11`, `rr-e-gfx11`) | No approximation (FP8 emulated exactly) | Rounding changes only ([Ray Reconstruction](#ray-reconstruction-d-and-e)) | 61.7 dB from the exact Vulkan build (RR D replay) |

The M and L figures are in the same ranges the Linux page reports for the same scenes (M 37.6 / 48.4 / 38.5 dB, L 36.8 / 50.1 / 36.4 dB on vkhip there). As that page explains, FP8 flips put any two implementations of M and L that round differently in this range; a lower figure is not by itself a sign of approximation.

## Presets M and L

### Network

The HIP Swin layers d3dhip runs are the same native kernels as the Linux vkhip and CUDA routes (`kernels/m`, built for gfx110x by `kernels/build.sh m`), with the same widening: f32 accumulators through the GEMM chains (`SWIN_F32ACC`) and intermediate activations kept in f16 where NVIDIA re-quantizes to FP8 (`SWIN_NO_Q8`). Stored layer outputs stay FP8. See the Linux page's tables; nothing differs.

### The RDNA3 D3D12 stages

Input (`enc0_m.comp`, L's `enc0_l_features` and `enc0_l_network`), expansion (`dec0_m.comp`, L's `dec0_l`, `l_patch_dec0`, `dec0_l_head`), reconstruction (`post_m.comp`), downsample (`down_m.comp`) and exposure, from `engine/m_d3d12_build.py`'s package `rdna3/`. In the current (`D4RML007`) package:

| Step | NVIDIA's kernel | RDNA3 D3D12, `D4RML007` | Kind |
|---|---|---|---|
| M's expansion GEMM (tokens x [64][80] matrix) | FP8 operands, f16 accumulator, one binary16 rounding per 32 products | FP8 values widened to f16 (exact) on `v_wmma_f32_16x16x16_f16`, f32 inside each 32, the same binary16 rounding per 32 (`dec0_m.comp D4R_M_EXPAND_F16`) | Rounding change (summation order inside 32) |
| Conversions f32 -> binary16 | `cvt.rn`: to nearest even | to nearest even, on AGS's matrix conversion (`engine/d3d12/rne16_ags.hlsli`) | None |
| Final store into the game's texture | formatted f32 store (`sust.p`) | the f32 result to the format's conversion (`down_m.comp WIDE_STORE`, as the Linux branch's RDNA3 stage set) | None |
| Output formats | any | RGBA16F, RGBA32F, RGB10A2, R11G11B10F, and RGBA8 / BGRA8 where the device has typed UAV stores for them | - |

### Instructions that look like approximations

As on Linux: `v_rcp_f32`, `v_rsq_f32`, `v_exp_f32`, `v_log_f32` are NVIDIA's own `.approx` instructions as ZLUDA lowers them; GELU, the softmax numerators and the reciprocals are DLSS's own design. d4r reproduces them rather than adding them. No RDNA3 D3D12 M/L shader contains an integer matrix or integer dot-product instruction in the `D4RML007` package.

## Preset K

d3dhip K runs the fast native layer set (`kernels/k`, `kernels/windows-flags.txt` for gfx1101): the Linux page's K network table applies unchanged (f32 accumulators in enc0-2 / dec0-2, f32 sums in the L2 norm and softmax: widening; the residual added once: a rounding change). vkhip K in Vulkan games runs the same layers.

K's D3D12 stages (input, output, exposure, from `engine/compile_k.py`) are built with `D4R_NATIVE_HALF`, so their binary16 conversions round toward zero ([Open items](#open-items)).

## Ray Reconstruction D and E

RDNA3 has no FP8 matrix instruction; the gfx11 packages emulate NVIDIA's `mma.m16n8k32.f16.e4m3.e4m3.f16` as ZLUDA lowers it for gfx11 (FP8 values widened exactly to binary16 scaled by 2^-8, two chained `v_wmma_f32_16x16x16_f16` per k32 with the accumulator scaled by 2^-16, one binary16 rounding; `engine/rr/rr_gfx11.glsl`). That is exact.

The default build applies `rr_hlsl.relaxed_math`. On gfx11 it does two things:

| Step | Exact build | Default | Kind |
|---|---|---|---|
| `fma16` (binary16 fma, one rounding) | f32 fma plus a midpoint correction | `mad` on binary16 operands | None: the driver emits a single-rounding fma (2^20 random operand triples: 100.000% equal to the correctly rounded result) |
| The output stage (hkpn) | `precise` | without `precise` | Rounding change (the compiler may fuse) |

Its other switches (`RR_QUANT_CLAMPONLY`, the external binary16 rounding) are not reached by any gfx11 stage. `D4R_RR_GFX11_EXACT=1` builds the exact form.

Both builds measure 61.7 dB from the exact Vulkan gfx11 build on the RR D replay (640x360 -> 1280x720, 8 frames), and 61.9 dB from each other. The D3D12-Vulkan gap starts in encoder 0 at the gate `sig(x) = 1 / (1 + exp2(-x log2 e))`, whose `1 / x` and `exp2` the two compilers lower differently; NVIDIA's own kernel uses `rcp.approx` and `ex2.approx` there. That is a rounding difference in an approximation that belongs to DLSS, not one d4r adds.

## What changed

Found and fixed on 2026-11-10 (`../rdna3-patches/10-ml-exact-rdna3.patch`):

1. **M's expansion stage was int8.** The RDNA3 package built `dec0_m.comp` with `DOT2 INT8`: each token's 64 channels re-quantized to int8 with one scale, the matrix int8 with column scales. That is a narrower format than NVIDIA's FP8, the kind of approximation the Linux page's `dot2` section describes. It now runs on f16 WMMA with NVIDIA's roundings (above), and is faster than the int8 form (expansion 0.33 -> 0.27 ms at 1440p).
2. **Binary16 conversions rounded toward zero.** AMD's D3D12 compiler converts f32 to binary16 with `v_cvt_pkrtz_f16_f32`: casts and `f32tof16` alike rounded toward zero on all of 2^20 random f32 inputs (the GPU's binary16 arithmetic itself rounds to nearest). The RDNA3 stages were built with that native conversion (`D4R_NATIVE_HALF`). They now round to nearest on AGS's f32 -> f16 matrix conversion, which is exact on the same 2^20 inputs, in uniform and divergent code alike, at one instruction per pair.
3. **The rounding emulation was wrong for binary16 subnormals.** `dialect_hlsl.h`'s emulated rounding relies on `precise` to keep `(v + 0.75) - 0.75`, which AMD's D3D12 compiler folds: half of the inputs under 2^-14 truncated. `D4R_RNE16_EXACT` is a form that is exact on all 2^20 inputs (subnormals through `round()`, overflow to infinity). It is not the default: built into L's `enc0_l_network` and `dec0_l` it gives a 10 dB worse image for a reason not found, while a debug build that computes AGS's and the emulation's result on every conversion of the reconstruction found them equal everywhere and gave AGS's output byte for byte.
4. **Output passed through half.** RGB10A2 output was rounded to binary16 toward zero before its store; the f32 result now goes to the format's conversion. RGBA8 and BGRA8 output were refused and are now accepted.
5. **A latent bug in relaxed RDNA3 RR.** `rr_wmma.hlsli`'s AGS rounding of a pair read gfx12's register layout and returned 0 for the second value on gfx11. Every RR package's DXIL is unchanged by the fix (the compiler removes the only calls), so no released output depended on it.

Cost and gain, `scripts/hillclimb/perf/bench.py`, `D4RML006` (old) against `D4RML007` packages, same shim, three interleaved rounds:

| Case | Old | Exact | Change |
|---|---|---|---|
| M Performance, 1280x720 -> 2560x1440 | 5.344 ms | 5.246 ms | -1.84% |
| M Performance, 1920x1080 -> 3840x2160 | 11.803 ms | 11.572 ms | -1.96% |
| L Ultra Performance, 854x480 -> 2560x1440 | 4.275 ms | 4.326 ms | +1.20% (spread 2.68%: noise) |
| L Ultra Performance, 1280x720 -> 3840x2160 | 9.616 ms | 9.642 ms | +0.27% (spread 2.03%: noise) |
| L Performance, 1280x720 -> 2560x1440 | 8.174 ms | 8.193 ms | +0.24% (noise) |
| L Performance, 1920x1080 -> 3840x2160 | 18.222 ms | 18.275 ms | +0.29% (noise) |

M and L run at their intended modes in the first four rows (`dx-M-*`, `dx-L-up-*` in `matrix-rdna3.json`).

Distance from NVIDIA's kernels, old against exact packages:

| Scene | M old | M exact | L old | L exact |
|---|---|---|---|---|
| Motion | 38.03 | 38.06 | 36.96 | 36.93 |
| Jitter | 45.25 | 45.86 | 49.74 | 52.09 |
| LDR (RGBA8 output) | refused | 38.62 | refused | 36.77 |

No frame of any run had non-finite values.

## Open items

- **K's D3D12 stages round binary16 conversions toward zero** (`compile_k.py` `D4R_NATIVE_HALF`), on RDNA4 too, since the K package's stages are shared. Same format, so not an approximation by the definition above, but a biased rounding NVIDIA does not have. Changing it changes RDNA4's DXIL and needs an RDNA4 run first.
- **The reference is not an RTX card**, and it shares d4r's front end (inputs and output through half), as the Linux page explains.
- **Only harness scenes** were compared. Ghost of Tsushima ran M and L through d3dhip at their intended modes without visible problems; there was no pixel comparison in game.
- **The emulated rounding's failure in L's `enc0_l_network` / `dec0_l`** (item 3 above) is not understood. The default does not use it.
