# The AMD vector-issue limit: RX 9070 XT against RTX PRO 4500 Blackwell

Measured 2026-10-29. This page records why the native and Vulkan DLSS stages land at 1.5–2.5× NVIDIA's time on an RX 9070 XT even though both cards' FP8 matrix units are almost equally fast, and what that means for the remaining kernel work. The preset-by-preset timings it explains are in [local-gpu-preset-comparison.md](local-gpu-preset-comparison.md); the raw data is in [benchmarks/gpu-limits-20261029/](benchmarks/gpu-limits-20261029/).

## Summary

- **Matrix math is nearly even.** Dense FP8 is about 1.1× NVIDIA's (392–403 against 418–470 TFLOPS).
- **Plain FP32 vector math is not.** An FP32 FMA loop runs 3.2× faster on the RTX PRO 4500 (28.1 against 8.9 T FMA/s). Integer work (1.14×) and packed f16 (1.22×) are nearly even.
- **The DLSS stages are mostly vector work.** In SR L's `enc0_l` and `dec0_l`, matrix instructions are only 5% of the vector instructions. FP32 math is 28–30%, packed 16-bit math 3%, and dual-issued (VOPD) instructions 5–7%. These stages are VALU-bound (about 60% VALU busy) and do about as much non-matrix work per matrix instruction as NVIDIA's own kernels.
- **So the ~2× on these stages is mostly hardware at the current instruction mix**, not a deficiency in the d4r kernels. It can be reduced only by changing that mix.
- **Changing the mix gained 13–14%.** Five changes cut the vector instructions per wave by 24% (`enc0_l` 3350 → 2542) and 23% (`dec0_l` 2780 → 2137). At 4K, `enc0_l` dropped from 896 to 772 µs (1.53× NVIDIA) and `dec0_l` from 750 to 653 µs (1.89×); the whole evaluation went from 4.22 to 4.00 ms. `compile_m.py` now builds them by default ([below](#what-was-tried-against-it)). Packed f16 helped only for the softmax: every value that ends in FP8 has to be f32 first (point 2 below).

Follow-up: [explicit dual issue, native residual/weight-load changes and RGP-guided occupancy experiments](l-architecture-workarounds.md) found no large additional gain. Explicitly paired GELU lost to optimized compiler output; raw-buffer weight loads saved about 0.6% of a whole L evaluation on the tested input sequence. All follow-up options remain experimental.

## Measured throughput

Same source on both cards (HIP and CUDA builds of [tools/bench_valu.h](../tools/bench_valu.h) and [tools/bench_gpu_peaks.*](../tools/bench_gpu_peaks.hip)). Each test runs independent chains, so the results are throughput, not latency.

| Operation | RX 9070 XT (330 W cap) | RTX PRO 4500 Blackwell (200 W) | NVIDIA / AMD |
|---|---:|---:|---:|
| FP8 matrix (`v_wmma_f32_16x16x16_fp8_fp8` / `mma.sync.m16n8k32.e4m3`) | 392–403 TFLOPS (350–383 at the 250 W cap) | 418–470 TFLOPS | ~1.1× |
| FP16 matrix, f32 accumulation | — | 209 TFLOPS | — |
| FP32 FMA (`v_fma_f32` / `FFMA`) | 8.1–8.9 T FMA/s | 27.3–28.1 T FMA/s | **3.2×** |
| Integer xor/shift/add | 15.8–17.2 T op/s | 19.4 T op/s | 1.14× |
| Packed f16 FMA (`v_pk_fma_f16` / `HFMA2`), per half | 22.1–24.2 T FMA/s | 29.1–29.4 T FMA/s | 1.22× |
| Read bandwidth | 324–446 GB/s (spec 640) | 817–825 GB/s | ~1.9–2.5× (the local read kernel is not tuned) |
| Copy bandwidth (read + write) | 412–523 GB/s | 729–743 GB/s | ~1.4–1.8× |

The FP32 row is the one that matters. NVIDIA's 10 496 FP32 lanes at about 2.7 GHz give 28 T FMA/s. The 9070 XT's 4096 lanes give a third of that unless the compiler pairs two instructions into one dual-issue (VOPD) slot. The FMA loop compiled to plain `v_fma_f32`, so it measures the unpaired rate, as does most of the shader code below. Packed f16 closes most of the gap because one `v_pk_fma_f16` does two operations per lane.

Neither card's spec-sheet peak is used here. The RTX PRO 4500's often-quoted 811 TFLOPS of FP8 is the server edition's *sparse* figure, and the 9070 XT's 48.7 TFLOPS of FP32 assumes every instruction dual-issues.

## Three RDNA4 specifics that cost vector instructions

1. **FP32 vector issue rate.** A third of NVIDIA's per clock, as measured above, outside dual issue. VOPD pairs only specific two-operand forms from both halves of a wave's register file; in the L stages it covers 5–7% of the VALU instructions.
2. **FP8 is made only from f32.** The FP8 conversions the compilers use here take f32 sources (`v_cvt_pk_fp8_f32`). An f16 value has to be widened first, so packing two f16 values into FP8 costs two `v_cvt_f32_f16` and one `v_cvt_pk_fp8_f32`. NVIDIA does it in one `cvt.rn.satfinite.e4m3x2.f16x2`. `enc0_l` spends 276 FP8 packs and about 600 other conversions per wave.
3. **FP8 matrix results are f32 only.** `v_wmma_f32_16x16x16_fp8_fp8` has no f16-accumulating form, while NVIDIA's kernels accumulate in f16 (`mma ... .f16.e4m3.e4m3.f16`). The exact builds therefore round through f16 after each 32-deep step. In the relaxed build RADV already folds the register-only f32 → f16 → f32 round trip: an explicit "stay f32 until the FP8 conversion" variant produced the same instruction counts (experiment 3 below). What does cost is the saturation: a value the shader declares binary16 might be infinite, so RADV clamps every element NaN-preserving (`v_minimummaximum_f32`, 224 per wave in each L stage) before `v_cvt_pk_fp8_f32`. Declared f32, the sum of FP8 products cannot be infinite, and the finite-only conversion needs no clamp (experiment 4).

## What this means stage by stage

| Stage class (4K) | Local / NVIDIA | What limits it | Floor that applies |
|---|---:|---|---|
| SR L `enc0_l` (Vulkan) | 1.77× (1.94× in the suite run); 1.53× after experiments 4–8 | VALU 60% busy; 20.8 VALU per WMMA (15.9 after); 28% FP32 math | FP32 issue: hardware at the current mix |
| SR L `dec0_l` (Vulkan) | 2.17× (2.25× before the unroll fix; 2.52× in the suite run); 1.89× after experiments 4–8 | VALU 59% busy; 19.5 VALU per WMMA (14.8 after); 30% FP32 math | the same |
| RR D `dec0` (native) | 2.4× | VALU 61% busy; 18 VALU per WMMA | the same |
| RR D `enc0` (native) | 2.3× | latency: VALU 17%, LDS 5%, VMEM 7% busy at 8 waves per SIMD | software (occupancy, memory latency) |
| SR network, RR CNN (native) | 1.5× | not profiled per instruction | between the matrix and vector floors |
| SR reconstruction, downsample | 0.7–0.8× | — | already faster than NVIDIA |

NVIDIA's own L kernels do about the same non-matrix work. Their PTX has 26.7 (`rrlite_dec0_4x4`) and 27.8 (`rrlite_enc0_4x4_mvhi_hdr`) statements per `mma`, with 304 and 336 `mma` per 8×8 block. d4r's builds do 288 WMMA per block (two waves) and about 20 VALU per WMMA. The PTX statement counts include `mov`s that ptxas removes, so this is an order-of-magnitude comparison, not exact: the d4r stages are not inflated relative to NVIDIA's. ([ptx-l-instruction-mix.txt](benchmarks/gpu-limits-20261029/ptx-l-instruction-mix.txt), [sr-l-4k-instruction-mix.txt](benchmarks/gpu-limits-20261029/rgp/sr-l-4k-instruction-mix.txt).)

A check that these stages run at their VALU instruction count: `enc0_l` issues about 2.1 × 10⁸ VALU wave-instructions per 4K evaluation (516 sampled waves of 3334, 66 034 waves in all). At 128 SIMDs and about 2.4 GHz that is 0.68 ms at full issue, or 1.13 ms at the measured 60% busy. The stage measured 1.16 ms in the same trace.

### What was tried against it

Five workarounds, judged first by the static instruction count of the compiled stage ([tools/radv_isa_mix.py](../tools/radv_isa_mix.py) on a `RADV_DEBUG=shaders` dump; the stages unroll completely, so static counts are per-wave counts), then timed. The numbers are in the experiment table below.

1. **Packed f16 instead of f32 elementwise math.** It helps where the result stays in f16, and the softmax numerator is the only such case (experiment 6, together with dropping the key permutation that only matters for ZLUDA-identical output). Everything else ends in an FP8 conversion that takes only f32 (point 2 above). GELU in packed f16, for example, would take about five instructions per element once the conversions back are counted, against 4.5 in f32 (counted, not built). What did pay was the opposite: keeping values in f32 and not converting at all. That covers the residual stream through the whole block (experiment 5), the f32 accumulators into the finite-only FP8 conversion (experiment 4) and f32 bias tiles in the model (experiment 7).
2. **Norm gains folded into the weights.** One multiply per normalised element fewer (about 80 instructions per stage), 1–2% faster. The folded weights are requantized to FP8, so it changes the model; it stays an opt-in experiment ([tools/fold_l_norm_gains.py](../tools/fold_l_norm_gains.py), experiment 9).
3. **Dual issue.** Not reachable from GLSL. ACO forms VOPD pairs after register allocation, and a GELU written only in pairable forms (bit-mask abs, separate min and max, multiply-add with a literal) compiled to the same instructions as before, because NIR folds it back into `v_maxmin_num_f32` and VOP3 FMAs (experiment 10).
4. **Native HIP code for these stages.** The same ten Swin layers take 1.75 ms as `layer_m.comp` on the Vulkan route and 1.68 ms as native code objects on the vkhip route, about 4%. `enc0_l` also samples six Vulkan textures, which a HIP kernel on the vkhip route cannot do. Not ported.
5. **Fewer conversions.** The 224 clamps per stage (experiment 4) and about 110 f16 → f32 bias widenings (experiment 7) are gone. The remaining 180–200 f32 → f16 conversions feed f16 stores (V for the attention, the embedding's output, the expansion head's input). Packing them would need `v_cvt_pk_rtz_f16_f32`, i.e. round-toward-zero, which GLSL cannot ask for.

Also kept: the weight tiles' address arithmetic (experiment 8). The cooperative load recomputed two lane terms for each of about 120 tile loads per wave; reading each lane's eight bytes from one precomputed lane offset removes that.

Not tried: occupancy for the latency-bound RR `enc0`. RGP reports that 48 fewer VGPRs and 2 KB less LDS would raise it.

## Experiments on SR L (4K, vkhip route, 330 W cap)

Median of three interleaved trials of 100 timed evaluations each (`d4r-m`, `D4R_M_TIME_EACH=1`; NVIDIA's L stages are 503 µs and 345 µs); outputs compared byte for byte ([sr-l-experiments.json](benchmarks/gpu-limits-20261029/sr-l-experiments.json)).

| # | Change | `enc0_l` µs | `dec0_l` µs | Whole µs | Output | Kept |
|---|---|---:|---:|---:|---|---|
| — | baseline (`WPW=2`) | 893 | 776 | 4296 | — | — |
| 1 | `dec0_l`'s patch-expand token copy unrolled (`engine/layer_m.comp`: the loop counts from 0, so the compiler unrolls it and issues all eight byte loads before waiting) | 891 | 748 | 4268 | byte-identical | yes |
| 2 | 1 plus one window per wave (`WPW=1`) for both stages | 942 | 814 | 4377 | byte-identical | no: slower |
| 3 | 1 plus `M_ACC8`: accumulators f32 from the WMMA to the FP8 conversion for Q, the MLP's hidden layer and the expansion | 889 | 745 | 4271 | byte-identical | no: identical instruction counts, RADV already folds it |

Experiments 4–10 build on experiment 1. Each has its own `layer_m.comp` define, all in the relaxed build only ([sr-l-vector-work.json](benchmarks/gpu-limits-20261029/sr-l-vector-work.json); [tools/build_l_stage_variant.py](../tools/build_l_stage_variant.py) rebuilds just the two stages of a model). "VALU" is per wave, `enc0_l` / `dec0_l`. Times are medians of 2–3 interleaved trials against the previous step in the same session; the last two rows are the final 3-trial run.

| # | Change | VALU | `enc0_l` µs | `dec0_l` µs | Whole µs | Kept |
|---|---|---:|---:|---:|---:|---|
| 1 | (the base for these rows) | 3350 / 2780 | 894 | 751 | 4216 | — |
| 4 | `M_F32ACT`: f32 accumulators wherever a product only feeds an FP8 conversion (Q, MLP hidden, expansion, embedding, merge), converted with the finite-only saturation | 3037 / 2568 | 846 | 718 | 4132 | yes |
| 5 | + `M_RES32`: the residual stream f32 for the whole block; norm 2 sums its squares in f32 | 2864 / 2435 | 819 | 699 | 4082 | yes |
| 6 | + `M_PK16`: the softmax numerator on packed f16 pairs; no ZLUDA key permutation before P·V | 2691 / 2262 | 790 | 670 | 4033 | yes |
| 7 | + `M_BIAS32`: f32 copies of the bias tiles in the model (`test_m.pack_layer`, slots o[32..40]) | 2577 / 2152 | 777 | 655 | 4033 | yes (output byte-identical to 6) |
| 8 | + `M_LDAU`: weight tiles read as each lane's 8 bytes from one precomputed lane offset | 2542 / 2137 | 774 | 657 | 4034 | yes |
| 9 | + norm gains folded into Q, V and W1 (FP8 requantized) | 2460 / 2062 | 761 | 646 | 3985 | no: changes the weights; opt-in tool |
| 10 | GELU in dual-issuable forms | 2542 / 2137 (unchanged) | — | — | — | no: same ISA |
| — | final: before (experiment 1) | 3350 / 2780 | 896 | 750 | 4220 | — |
| — | final: `compile_m.py` now (4–8) | 2542 / 2137 | 772 | 653 | 4005 | default |

Quality. The output of 6–8 is byte-identical to the relaxed build's output with 4–6 only. Against the exact build of the two stages (the arithmetic of NVIDIA's translated kernels) it is 51.7–54.9 dB per 4K frame, against 51.9–56.6 dB for the relaxed build before; with the gain fold it is 52.0–55.7 dB.

Correction: the earlier 29.5–29.6 dB comparison on `build/l-vkhip-20261028`
used misaligned reference frames. The capture's asynchronous queue displays
result 0 in saved frame 2 and result 1 in saved frame 3; saved frame 1 is black
and frame 4 repeats result 1 after feature retirement. The correctly aligned
optimized control scores 45.51 and 45.70 dB (unit peak, unclipped RGB, watermark
excluded). The old numbers do not establish an error outside these stages.
Historical variants have not all been rescored. See the
[AMD quality-budget implementation](l-amd-quality.md) for aligned references
and the new 40 dB acceptance gate.

RGP put 34% of `dec0_l`'s sampled cycles in the byte-copy loops of experiment 1, but removing them saved 5%. RGP's per-instruction cost counts every cycle a sampled wave waits, including waits that other waves on the SIMD hide, so it overstates what removing a stall saves in a stage that is throughput-bound.

## Measurement notes

- **Clock samples.** `tools/bench_local_presets.py` samples `sclk` every 200 ms. For SR (about 15% GPU busy, the runner spends about 120 ms of CPU time per evaluation) that catches idle time between bursts, not the bursts: the stage times are the same on routes sampled at 1.8 and 2.7 GHz. RR runs are 93–99% busy and their samples are meaningful.
- **Power.** Every RR route hit the 250 W cap; at the 330 W default they are 8–12% faster and sit at 330 W ([suite-*-330w.json](benchmarks/gpu-limits-20261029/)).
- **FLOP estimate** for the RR share-of-peak table ([tools/estimate_rr_swin_flops.py](../tools/estimate_rr_swin_flops.py)): every FP8 weight matrix in the native stage headers applied once per token, K and V for 96 keys per 64-query window, QKᵀ and PV at head dimension 32. Both cards run the same network, so the ratio of their shares does not depend on the estimate being exact ([rr-d-4k-fraction-of-peak.txt](benchmarks/gpu-limits-20261029/rr-d-4k-fraction-of-peak.txt)).

## Reproduction

Follow-up measurements: the [AMD decoder redesign](l-amd-redesign.md) provides
an exact layout improvement, and the [FSR4 comparison](fsr4-l-instruction-comparison.md)
measures an actual FP8 AMD reference. L's inner tube already has a similar
vector-to-matrix ratio to FSR4's core; the outer blocks and total matrix budget
remain larger targets.

```sh
# microbenchmarks, local
hipcc --offload-arch=gfx1201 -O3 tools/bench_gpu_peaks.hip -o /tmp/peaks && /tmp/peaks
hipcc --offload-arch=gfx1201 -O3 tools/bench_valu.hip -o /tmp/valu && /tmp/valu
# the same on an RTX PRO 4500 Blackwell (RunPod, image runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04;
# workstation edition, 200 W, driver 595.91.07 as NVIDIA's reference run; each session took 3-4 minutes at $0.72/h)
nvcc -O3 -arch=sm_120 tools/bench_gpu_peaks.cu -o peaks && ./peaks
nvcc -O3 -arch=sm_120 tools/bench_valu.cu -o valu && ./valu

# an RGP trace of one evaluation (RADV writes one .rgp per submit to /tmp)
MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 RADV_THREAD_TRACE_BUFFER_SIZE=$((512*1024*1024)) \
    D4R_M_WARMUP_PASSES=1 D4R_M_TIME_EACH=1 d4r-m MODEL 1920 1080 3840 2160 OUT 3 vkhip -- FRAMES...
# RGP headless: Radeon Developer Tool Suite 2026-10-02, under Xvfb (xcb, not Wayland)
Xvfb :78 -screen 0 2560x1440x24 &
env -u WAYLAND_DISPLAY DISPLAY=:78 QT_QPA_PLATFORM=xcb RadeonGPUProfiler trace.rgp &
# Events > Instruction timing, click the table, Ctrl+A, Ctrl+C (xdotool works), then:
gcc tools/rgp_clip.c -o /tmp/rgp_clip -lX11 && DISPLAY=:78 /tmp/rgp_clip > table.tsv
python3 tools/rgp_instruction_mix.py table.tsv
python3 tools/ptx_instruction_mix.py dlss-0148-00.ptx rrlite_enc0_4x4_mvhi_hdr dlss-0161-00.ptx rrlite_dec0_4x4

# experiments 4-10: rebuild only enc0_l / dec0_l of a compiled L model, count, time
python3 tools/build_l_stage_variant.py MODEL VARIANT -DM_F32ACT -DM_RES32      # LVAR_BASE= for the exact build
RADV_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=1 d4r-m VARIANT 1920 1080 3840 2160 OUT 1 vkhip -- FRAMES... > isa.txt 2>&1
python3 tools/radv_isa_mix.py isa.txt                                         # the plane-layout pair is the vkhip one
D4R_M_WARMUP_PASSES=12 D4R_M_TIME_EACH=1 d4r-m VARIANT 1920 1080 3840 2160 OUT 37 vkhip -- FRAMES...
```

M_BIAS32 reads f32 bias tiles that only a model compiled by the current `compile_m.py` carries; a variant built from an older model needs them added first.

If RGP shows no window, run `scripts/remove_shared_memory.sh` from the tool suite; a killed instance leaves its shared memory behind. The traces themselves (130–220 MB each) are not in the repository.
