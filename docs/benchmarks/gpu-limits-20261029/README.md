# GPU limits, RX 9070 XT against RTX PRO 4500 Blackwell (2026-10-29)

Raw data behind [amd-vector-issue-limit.md](../../amd-vector-issue-limit.md) and the "Hardware limitation or software limitation" section of [local-gpu-preset-comparison.md](../../local-gpu-preset-comparison.md). Local runs are on the RX 9070 XT under RADV (Mesa 26.2.5) at the 330 W default cap unless a file says otherwise. NVIDIA runs are on a rented RTX PRO 4500 Blackwell workstation edition (RunPod secure cloud, driver 595.91.07, 200 W, 82 SMs).

| File | What it holds | Made with |
|---|---|---|
| `peaks-rx9070xt-330w.txt`, `peaks-rtxpro4500.txt` | dense FP8 matrix, read and copy throughput | [tools/bench_gpu_peaks.hip](../../../tools/bench_gpu_peaks.hip), [.cu](../../../tools/bench_gpu_peaks.cu) |
| `valu-rx9070xt-330w.txt`, `valu-rtxpro4500.txt` | FP32 FMA, integer and packed-f16 vector throughput; the NVIDIA file starts with the inner loop's SASS opcode counts | [tools/bench_valu.hip](../../../tools/bench_valu.hip), [.cu](../../../tools/bench_valu.cu) |
| `rr-d-4k-flops.json` | estimated FP8 FLOPs per RR preset D stage at 4K | [tools/estimate_rr_swin_flops.py](../../../tools/estimate_rr_swin_flops.py) |
| `rr-d-4k-fraction-of-peak.txt` | those FLOPs over each card's stage times and measured peak | the same |
| `suite-{rr,sr}-{1440p,4k}-330w.json` | full preset suite re-run at the 330 W cap (the main comparison ran at 250 W) | [tools/bench_local_presets.py](../../../tools/bench_local_presets.py) |
| `sr-l-experiments.json` | SR L 4K vkhip: baseline, the kept `dec0_l` unroll fix, `WPW=1` and `M_ACC8`, per trial, with output identity | `d4r-m`, `D4R_M_TIME_EACH=1` |
| `sr-l-vector-work.json` | SR L 4K: instruction counts, timings and PSNR of the five vector-work changes (experiments 4–10) | [tools/radv_isa_mix.py](../../../tools/radv_isa_mix.py), [tools/build_l_stage_variant.py](../../../tools/build_l_stage_variant.py), [tools/fold_l_norm_gains.py](../../../tools/fold_l_norm_gains.py) |
| `sr-l-architecture-workarounds.json` | Follow-up explicit-VOPD GELU benchmark, six paired L experiments, frame hashes and optimized-control RGP counters | [l-architecture-workarounds.md](../../l-architecture-workarounds.md), [tools/bench_l_gelu.hip](../../../tools/bench_l_gelu.hip) |
| `sr-l-amd-redesign.json` | Decoder column selection, direct expansion loads and wave register ownership; paired trials, frame hashes and precision checks | [l-amd-redesign.md](../../l-amd-redesign.md), [tools/build_l_head16.py](../../../tools/build_l_head16.py) |
| `sr-l-amd-quality.json` | AMD hardware FP8 saturation and approximate outer-block dataflow, 40 dB per-frame gate on 42 references, paired timings, rejected experiments and corrected NVIDIA capture alignment | [l-amd-quality.md](../../l-amd-quality.md), [tools/build_l_amd.py](../../../tools/build_l_amd.py), [tools/bench_l_amd.py](../../../tools/bench_l_amd.py) |
| `sr-l-outer-rdna4.json` | RGP-guided register feature/embedding and V pipelines, selected gain folding, eight-channel skip planes, fused/split dispatch comparison, three-way paired timing and 42-frame quality gate | [l-outer-rdna4.md](../../l-outer-rdna4.md), [tools/build_l_outer.py](../../../tools/build_l_outer.py) |
| `fsr4-l-instruction-comparison.json` | FSR4 4.1.0 actual FP8 path, per-pass dynamic opcode counts and resource limits; L inner/outer comparison on the same GPU | [fsr4-l-instruction-comparison.md](../../fsr4-l-instruction-comparison.md), [tools/fsr_upscale_probe.cpp](../../../tools/fsr_upscale_probe.cpp) |
| `ptx-l-instruction-mix.txt` | static PTX instruction counts per `mma` of NVIDIA's L kernels `rrlite_enc0_4x4_mvhi_hdr` and `rrlite_dec0_4x4` | [tools/ptx_instruction_mix.py](../../../tools/ptx_instruction_mix.py) |
| `rgp/sr-l-4k-{enc0_l,dec0_l}-instruction-timing.tsv` | RGP instruction timing tables copied out of the GUI (line, opcode, operands, hits, cost, latency) | RGP 2.7.1 under Xvfb, [tools/rgp_clip.c](../../../tools/rgp_clip.c) |
| `rgp/sr-l-4k-instruction-mix.txt` | per-wave instruction mix of those two tables | [tools/rgp_instruction_mix.py](../../../tools/rgp_instruction_mix.py) |
| `rgp/rr-d-4k-{enc0,dec0}-*.png` | RGP instruction timing and pipeline state of RR preset D's `enc0` and `dec0` (4K, 250 W) | RGP screenshots |
| `rgp/sr-l-4k-*.png` | most expensive events of an SR L 4K evaluation, with instruction timing and pipeline state of `enc0_l` and `dec0_l` | the same |

The `.rgp` traces (130–220 MB per submit, about 1.9 GB in all) are not kept. [amd-vector-issue-limit.md](../../amd-vector-issue-limit.md#reproduction) has the commands to record new ones.
