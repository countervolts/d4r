# Local GPU comparison against the official preset timings

Measured 2026-10-29 on this machine: **Radeon RX 9070 XT** (Navi 48, `gfx1201`, PCI `1002:7550`), Mesa **26.2.5-arch3.1** RADV, kernel `7.3.0-5-cachyos`, DRM 3.64, `power_dpm_force_performance_level=auto` with the driver in the `performance` state, GPU power cap **250 W** (default 330 W, maximum 374 W). ROCm 7.2 (`hipconfig 7.2.53211`) supplied the code-object compiler; no CUDA and no ZLUDA installation exist here, so the CUDA backend was not available. These are isolated d4r engine evaluations, not game frame times.

The reference figures are the official NVIDIA measurements in [Blackwell preset comparisons](nvidia-blackwell-presets.md): an RTX PRO 4500 Blackwell at 200 W with driver 595.91.07 and the official Linux NGX CUDA libraries. That is a comparison GPU, not an RTX 5070 Ti, and the numbers are DLSS/RR evaluation cost rather than whole-game FPS. The machine-readable summary is [local-gpu-comparison-20261029.json](benchmarks/local-gpu-comparison-20261029.json); raw logs, commands, clocks and hashes are under `build/local-preset-compare-20261029/`.

## Method

- **Geometry:** the NVIDIA probe's two Performance cases, 1280×720 → 2560×1440 and 1920×1080 → 3840×2160 (render half the output in each axis).
- **Inputs recreated byte for byte** from [bench_official_ngx_cuda.cpp](../tools/bench_official_ngx_cuda.cpp): Super Resolution uses the probe's patterned RGBA16F (`0x3400 + ((x/8 + y/8) % 8) * 128`, G `0x3800`, B `0x3000`, A `0x3c00`) with zero motion and depth 0.5; Ray Reconstruction reuses one static noisy [fsr_rr_scene.h](../tools/fsr_rr_scene.h) frame (same camera, projection, `frameIndex=0`), sky depth `1e7`, packed roughness in the normal's `w`, linear RGBA8 albedos, specular hit distance 4, view and projection matrices, and the probe's jitter cycle (X by frame%2, Y by frame%4) with `Reset=1` only on the first evaluation.
- **Cadence:** at least 40 warmup and 100 measured evaluations per case, three interleaved trials (run order rotated and reversed per trial), reported as the median of the three trial medians — the same statistic as the NVIDIA table.
- **Timing boundary:** Vulkan timestamp queries bracket the whole evaluation, including exposure, input, network, reconstruction/expansion and output or downsample. Initialization, weight upload, per-frame input copies and the output readback are outside the bracket, as in the probe. Replay-runner knobs added for this: `D4R_M_WARMUP_PASSES`, `D4R_M_TIME_EACH`, `D4R_RR_NO_UPLOAD`, `D4R_RR_NO_WRITE` (`engine/replay_m.cpp`, `engine/replay_rr.cpp`); network-only time is reported separately by the same stamps.
- **Backend evidence, not assumption:** `d4r-m` refuses `vkhip`/`hip` when the model has no code objects and throws if the loader fails, `D4R_VKHIP_STAMPS=1` prints the per-layer GPU time of the native layers, and `d4r-rr` logs `vkhip: N Ray Reconstruction stages` or `vkhip unavailable: <reason>`. Every route below was checked against those messages and against its output hash.
- **Where the comparison is not like-for-like:** NVIDIA's figure brackets one `EvaluateFeature` with CUDA events; here the engine records one command buffer whose stages are separated by memory barriers, and its network arithmetic is native FP8 WMMA rather than NVIDIA's PTX. NVIDIA's stage numbers are summed kernel durations from Nsight traces, so they are labelled against whole-evaluation stamps wherever the two are put side by side.

## Fastest verified local route, per preset

| Preset | Resolution | Fastest verified local route | Local ms | NVIDIA ms | Local/NVIDIA |
|---|---|---|---:|---:|---:|
| SR M | 1280×720 → 2560×1440 | vkhip | 1.261 | 1.081 | 1.17× |
| SR M | 1920×1080 → 3840×2160 | vkhip | 2.865 | 2.553 | 1.12× |
| SR L | 1280×720 → 2560×1440 | vkhip | 2.002 | 1.413 | 1.42× |
| SR L | 1920×1080 → 3840×2160 | vkhip | 4.529 | 3.161 | 1.43× |
| RR A | 1280×720 → 2560×1440 | vkhip (11 CNN stages) | 2.051 | 1.358 | 1.51× |
| RR A | 1920×1080 → 3840×2160 | vkhip (11 CNN stages) | 4.619 | 2.979 | 1.55× |
| RR B | 1280×720 → 2560×1440 | vkhip (11 CNN stages) | 2.052 | 1.353 | 1.52× |
| RR B | 1920×1080 → 3840×2160 | vkhip (11 CNN stages) | 4.625 | 2.983 | 1.55× |
| RR C | 1280×720 → 2560×1440 | vkhip (11 CNN stages) | 2.054 | 1.358 | 1.51× |
| RR C | 1920×1080 → 3840×2160 | vkhip (11 CNN stages) | 4.627 | 2.988 | 1.55× |
| RR D | 1280×720 → 2560×1440 | vkhip (10 Swin stages) | 3.657 | 1.860 | 1.97× |
| RR D | 1920×1080 → 3840×2160 | vkhip (10 Swin stages) | 8.071 | 4.055 | 1.99× |
| RR E | 1280×720 → 2560×1440 | vkhip (10 Swin stages) | 3.649 | 1.863 | 1.96× |
| RR E | 1920×1080 → 3840×2160 | vkhip (10 Swin stages) | 8.029 | 4.057 | 1.98× |
| RR F | 1280×720 → 2560×1440 | Vulkan engine | 3.573 | 2.100 | 1.70× |
| RR F | 1920×1080 → 3840×2160 | Vulkan engine | 7.720 | 4.658 | 1.66× |

The winning backend is not uniform: vkhip wins for both Super Resolution presets and for the CNN (A/B/C) and Swin (D/E) networks, while preset F's Vulkan path is 1–2% ahead of its vkhip replacement.

## Every route measured

Whole-evaluation median of trial medians, milliseconds; 100 measured frames per run, three trials.

| Case | Vulkan | vkhip | HIP runtime |
|---|---:|---:|---:|
| SR M 1440p | 1.313 | **1.261** | 1.506 |
| SR M 4K | 2.928 | **2.865** | 3.021 |
| SR L 1440p | 2.043 | **2.002** | 2.229 |
| SR L 4K | 4.560 | **4.529** | 4.659 |
| RR A 1440p | 9.263 | **2.051** | — |
| RR A 4K | 18.990 | **4.619** | — |
| RR B 1440p | 9.270 | **2.052** | — |
| RR B 4K | 18.998 | **4.625** | — |
| RR C 1440p | 9.279 | **2.054** | — |
| RR C 4K | 19.000 | **4.627** | — |
| RR D 1440p | 3.964 | **3.657** | — |
| RR D 4K | 8.606 | **8.071** | — |
| RR E 1440p | 3.947 | **3.649** | — |
| RR E 4K | 8.565 | **8.029** | — |
| RR F 1440p | **3.573** | 3.608 | — |
| RR F 4K | **7.720** | 7.836 | — |

`HIP runtime` is `d4r-m ... hip`: the same code objects launched through the HIP runtime with the engine's stages split around it. It is 19–20% slower than vkhip here despite running at a higher sampled clock (2732 MHz against 1910 MHz), so the earlier impression that the HIP runtime route is the fast one does not survive interleaving. Presets A–C have no HIP-runtime route in the replay runner; only the vkhip CNN blocks exist.

The Ray Reconstruction CNN result is the largest local effect measured: A/B/C on the engine's own Vulkan CNN shaders cost 9.3 ms at 1440p and 19.0 ms at 4K, and replacing all eleven stages with the native code objects (`kernels/rr/build_vkhip_cnn.sh`) takes them to 2.05 ms and 4.62 ms — 4.5× faster, and within 1.5× of NVIDIA instead of 6.4×.

## Cost ratios: architecture versus implementation

| Ratio | Local (fastest route) | NVIDIA |
|---|---|---|
| L over M, 1440p | 1.588 | 1.307 |
| L over M, 4K | 1.581 | 1.238 |
| RR B over A, 1440p | 1.001 | 0.996 |
| RR B over A, 4K | 1.001 | 1.001 |
| RR C over A, 1440p | 1.002 | 1.000 |
| RR C over A, 4K | 1.002 | 1.003 |
| RR D over A, 1440p | 1.783 | 1.369 |
| RR D over A, 4K | 1.748 | 1.361 |
| RR E over A, 1440p | 1.780 | 1.372 |
| RR E over A, 4K | 1.738 | 1.362 |
| RR F over A, 1440p | 1.742 | 1.546 |
| RR F over A, 4K | 1.671 | 1.563 |
| RR F over D, 1440p | 0.977 | 1.129 |
| RR F over D, 4K | 0.956 | 1.149 |

**B/C over A is architecture on both GPUs** — the same network, differing only by run variation. **L over M** is partly architectural and partly implementation: NVIDIA's extra 0.607 ms at 4K comes from L's extra render-resolution Swin blocks plus a 4×4 reconstruction neighbourhood, and d4r pays 1.664 ms for the same design. The split is visible per stage below. **D/E over A** grows more on this card than on NVIDIA's, and **F relative to D/E** inverts: locally F is cheaper than D and E, whereas NVIDIA measures F 13–15% above them. F's own penalty over A is larger locally, but its rank relative to D/E is an implementation result, not a model property.

## Stage profile: Super Resolution L against M at 4K

Whole-evaluation stage medians in ms, vkhip route (the fastest one), against NVIDIA's Nsight stage medians for the same geometry (summed kernel durations, so a slightly different kind of number):

| Stage | Local M | Local L | Local L−M | NVIDIA M | NVIDIA L | NVIDIA L−M |
|---|---:|---:|---:|---:|---:|---:|
| Exposure | 0.067 | 0.067 | +0.001 | 0.038 | 0.037 | −0.000 |
| Input (`enc0`) | 0.206 | 0.977 | +0.771 | 0.214 | 0.503 | +0.289 |
| Network (ten Swin layers) | 1.664 | 1.686 | +0.022 | 1.078 | 1.092 | +0.014 |
| Expansion (`dec0`) | 0.157 | 0.868 | +0.710 | 0.117 | 0.345 | +0.228 |
| Reconstruction (`post`) | 0.590 | 0.748 | +0.158 | 0.836 | 0.917 | +0.080 |
| Downsample | 0.182 | 0.181 | −0.001 | 0.254 | 0.250 | −0.005 |
| **Whole evaluation** | **2.865** | **4.529** | **+1.664** | **2.553** | **3.161** | **+0.607** |

What separates:

- **The network is architectural and matched.** L's inner layers cost 22 µs more than M's here and 14 µs more on NVIDIA — L shares M's ten layers, so both GPUs pay almost nothing extra.
- **The input and expansion stages are the gap.** L's extra render-resolution Swin blocks cost 771 µs and 710 µs locally against 289 µs and 228 µs on NVIDIA: those two stages still run as Vulkan shaders (only the ten inner layers are native), and they run 1.94× and 2.52× NVIDIA's own kernels while M's `enc0` and `dec0` are at parity (0.96× and 1.34×). Roughly 1.0 ms of the 1.06 ms local excess over NVIDIA's L−M penalty sits in these two stages. They are VALU-bound (about 60% busy, 20 vector instructions per WMMA, about as many per block as NVIDIA's PTX), with 28–30% of their vector instructions in FP32 math, which issues at a third of NVIDIA's rate. At the current instruction mix much of that ratio is therefore hardware; see [amd-vector-issue-limit.md](amd-vector-issue-limit.md). Unrolling `dec0_l`'s token copy (2026-10-29) took it from 776 to 748 µs. Trimming their vector work (`compile_m.py`'s L defaults since 2026-10-29; [amd-vector-issue-limit.md](amd-vector-issue-limit.md#what-was-tried-against-it)) took them on to 772 and 653 µs, and the whole 4K evaluation from 4.22 to 4.00 ms (`d4r-m`, 330 W). The tables on this page are from before.
- **Reconstruction and downsample are cheaper here than on NVIDIA** (0.82× and 0.73× for L), so the local L penalty is not spread evenly: it is concentrated in the two Vulkan outer stages, which is the headroom the vkhip chain work targets.

Headroom, as a derived projection rather than a measurement: scale d4r's measured M cost by NVIDIA's L-over-M work ratio for the same geometry — the cost L would have if its outer stages were as efficient, relative to NVIDIA's kernels, as M's already are. At 4K that means multiplying each M stage by NVIDIA's per-stage L/M ratios (2.352 input, 1.013 network, 2.949 expansion, 1.096 reconstruction, 0.981 downsample, 0.993 exposure), giving **3.525 ms** (from 4.527; 0.493 input, 0.404 expansion, 0.102 reconstruction). At 1440p only NVIDIA's whole times are profiled, so scaling M's 1.265 ms by NVIDIA's 1.307 there gives **1.654 ms** (from 2.002). L over M would become **1.230–1.24 at 4K and 1.307 at 1440p — NVIDIA's own L/M ratios**, and L's ratio to NVIDIA would fall to M's existing 1.12× at 4K and 1.17× at 1440p, because an efficiency-matched L inherits M's ratio exactly. L cannot reach NVIDIA's absolute 3.161 ms with better kernels alone: its 1.54× network is hardware-side and M pays it too.

## Stage profile: Ray Reconstruction networks at 4K

Network stages from `D4R_ENGINE_RR_STAGE_STAMPS=1` (per-entry minima over ten repeats, medians over five profiled entries) against NVIDIA's per-kernel sums. "Outer" is the rest of the evaluation — exposure, input and output/post/downsample — taken as that profiled run's own whole-evaluation median minus its network stamps, so the local network and outer columns belong to the same run; the profiled runs sit within 2.4% of the unprofiled medians in the route table above. NVIDIA's outer is the non-network part of its kernel sum (`hkpn_input`, `hkpn_output`, `rr2_post`, `rr2_downsample` and the exposure helpers).

| Preset | NVIDIA network | NVIDIA outer | Local Vulkan network | Local vkhip network | Local vkhip outer | Whole local / whole NVIDIA | Network ratio vkhip |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1.854 | 1.077 | 17.230 | 2.769 | 1.756 | 4.62 / 2.98 = 1.55× | 1.49× |
| D | 3.227 | 0.823 | 7.365 | 6.846 | 1.220 | 8.07 / 4.05 = 1.99× | 2.12× |
| E | 3.284 | 0.765 | 7.585 | 7.034 | 1.016 | 8.03 / 4.06 = 1.98× | 2.14× |
| F | 3.117 | 1.472 | 6.362 | 6.391 | 1.517 | 7.72 / 4.66 = 1.66× | 2.05× |

- **Preset A/B/C's bottleneck is entirely implementation:** the same CNN network costs 1.85 ms on NVIDIA, 17.23 ms on this card's Vulkan shaders and 2.77 ms once native code objects replace it. NVIDIA's own CNN stage costs (PDL 0.462, Enc0 0.321, Dec0 0.200 ms) have local counterparts of 0.918, 0.533 and 0.327 ms — a uniform 1.5–2× rather than a different architecture.
- **Preset D/E's network is uniformly about 2.1× NVIDIA's**, stage by stage: `enc0` 2.393 against 1.021 ms, `dec0` 1.739 against 0.717, `dec1` 0.796 against 0.404, `enc1` 0.658 against 0.315. That is the whole 1.99× total; the outer group is 1.48× NVIDIA's.
- **Preset F's outer stages are at parity** — 1.517 ms local against NVIDIA's 1.472 ms of `rr2_post` and `rr2_downsample` — while its network is 2.05× NVIDIA's (`enc0` 1.545 against 0.712, `dec0_1` 0.993 against 0.383). F is therefore not "the expensive model" on this card; the expensive part is any network stage whose windows outsize the 64 compute units, which is why levels 3–5 dominate.

## Hardware limitation or software limitation

**Measured peaks, not spec sheets.** Both cards ran the same kind of microbenchmark: back-to-back FP8 matrix instructions with no dependencies between them (`v_wmma_f32_16x16x16_fp8_fp8` here, `mma.sync.m16n8k32.e4m3` there), plus a grid-stride read and copy. The NVIDIA side ran on a rented RTX PRO 4500 Blackwell (workstation edition, the same SKU, driver 595.91.07, 200 W limit, 82 SMs) on 2026-10-29.

| Metric | RX 9070 XT (here, 250 W) | RTX PRO 4500 Blackwell (200 W) | NVIDIA / local |
|---|---:|---:|---:|
| FP8 dense matrix | 350–383 TFLOPS | 418–470 TFLOPS (f16 and f32 accumulation identical) | **~1.15×** |
| FP16 dense matrix, f32 accumulation | — | 209 TFLOPS | — |
| Read bandwidth | 413–446 GB/s | 817–825 GB/s | ~1.85× (the local read kernel is probably under-tuned; spec 640 GB/s) |
| Copy bandwidth (read + write) | 505–523 GB/s | 729–743 GB/s | ~1.4× |
| FP32 FMA, vector (330 W) | 8.1–8.9 T FMA/s | 27.3–28.1 T FMA/s | **3.2×** |
| Integer xor/shift/add, vector (330 W) | 15.8–17.2 T op/s | 19.4 T op/s | 1.14× |
| Packed f16 FMA per half, vector (330 W) | 22.1–24.2 T FMA/s | 29.1–29.4 T FMA/s | 1.22× |

An earlier version of this table used 811 TFLOPS dense FP8 for the RTX PRO 4500. That is the *sparse* figure (and it belongs to the server edition), so it doubled the apparent compute gap to 2.09× and made the native stages look as if they already sat at the hardware limit. They don't.

**Share of peak on each card.** The per-stage FLOPs below are estimated from the weight layouts in the native stage headers (`native-work/<stage>/swin_stage.h`): every FP8 weight matrix applied once per token, K and V for 96 keys per 64-query window, QKᵀ and PV at head dimension 32. Both cards run the same network, so the ratio between the last two columns does not depend on the estimate being exact. Preset D at 4K, 250 W:

| Stage | Local TFLOPS | Local % of peak | NVIDIA TFLOPS | NVIDIA % of peak |
|---|---:|---:|---:|---:|
| `enc0` | 69 | 18% | 161 | 37% |
| `enc1` | 158 | 41% | 329 | 75% |
| `enc2`–`enc4`, `dec2`–`dec4` | 144–201 | 38–53% | — | — |
| `dec1` | 130 | 34% | 257 | 58% |
| `dec0` | 89 | 23% | 215 | 49% |
| Whole network | 109 | **29%** | 232 | **53%** |

NVIDIA's kernels get about twice the share of their card's peak. Measured against FP8 alone, about 1.15× of the 2.1× network gap is the hardware. That is only the floor for matrix work, though. Plain FP32 vector math is 3.2× faster on the RTX PRO 4500 (table above), and the stages that sit lowest in this table are the ones bound by vector work. [amd-vector-issue-limit.md](amd-vector-issue-limit.md) has the measurements and what they mean per stage.

**Where the full-resolution stages lose time** (RGP 2.7.1 instruction timing on a RADV SQTT capture of one preset D 4K evaluation, `MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1`):

| Dispatch | Time | VGPRs / LDS | Waves per SIMD (of 16) | VALU instructions per WMMA | VALU busy | LDS busy | VMEM busy |
|---|---:|---|---:|---:|---:|---:|---:|
| `enc0` (241×137 groups of 128) | 2516 µs | 192 / 14 336 B | 8 | 15.5 | 17% | 5% | 7% |
| `dec0` (241×137 groups of 32×4) | 1798 µs | 152 (168) / 14 336 B | 9 | 18.2 | 61% | 24% | 2% |

- **`dec0` is VALU-bound on non-matrix work.** It issues about 18 VALU instructions for every WMMA: fp8 packing, the table-driven cubic softmax and activation, f16 conversions, and the norms. The matrix units wait on that.
- **`enc0` is latency-bound.** Every unit is mostly idle. It issues 20× `dec0`'s vector-memory instructions per wave (its 16 texture inputs), and at 8 waves per SIMD there aren't enough waves to cover those memory waits and the barriers between phases. RGP reports that 2048 fewer bytes of LDS and 48 fewer VGPRs would raise its occupancy.

**Power.** The table above ran at a 250 W cap, which every RR run hit. With the cap back at the 330 W default (re-run 2026-10-29, same suite and statistic, logs in `logs-330w/`), every RR route got 8–12% faster and sat at the new cap. Preset D 4K went from 8.07 to 7.32 ms (1.80× NVIDIA), A/B/C 4K from 4.62 to 4.20 ms (1.41×), and F 4K Vulkan from 7.72 to 7.09 ms (1.52×). SR didn't change (within ±5%): those runs never get near either cap (15% busy, 53–124 W) and run at 1.7–1.9 GHz.

| Stage class | Measured ratio | Hardware floor | Reading |
|---|---:|---:|---|
| RR network, Swin (D/E) and rr2 (F) | 2.05–2.14× | FP8 ~1.15×; FP32 vector 3.2× | **mixed**: half NVIDIA's share of peak; `enc0` is latency-bound (software), `dec0` VALU-bound (hardware at its instruction mix) |
| SR network, ten Swin layers | 1.54× | FP8 ~1.15× | **mostly software** |
| RR CNN (A/B/C) | 1.49× | FP8 ~1.15× | **mostly software** |
| SR input (`enc0`) | 0.96× | bandwidth ~1.4× | software advantage |
| SR reconstruction, downsample | 0.71–0.82× | — | software advantage |
| RR outer group (exposure, input, output) | 1.33–1.63× | bandwidth ~1.4× | near the bandwidth floor |
| SR L's outer stages (`enc0_l`, `dec0_l`) | 1.77–1.94×, 2.17–2.52× | FP32 vector 3.2× | **hardware at the current instruction mix**: VALU-bound, about NVIDIA's instruction count; only packed f16 or dual issue lowers it |
| RR CNN on Vulkan shaders | 9.3× the native route on the same card | — | **software**, already removed by the vkhip CNN blocks |

The clear software headroom is RR `enc0` (occupancy and memory latency). For the VALU-bound stages (RR `dec0`, SR L `enc0_l`/`dec0_l`) the floor is set by the vector issue rate: a ratio near 1.15–1.4× is reachable only for work that runs on the matrix units, uses packed f16 or dual-issues, not for plain FP32 vector code.

## Verification of what actually ran

`build/local-preset-compare-20261029/logs/route-verification.json` from `tools/verify_local_routes.py`, plus `logs/route-verification-4k-rr.json` from `tools/check_local_rr_4k_routes.py` — the second one exists because the timed RR runs skip output writes (`D4R_RR_NO_WRITE=1`), so their 4K medians needed a separate output check.

- **Output coverage and finiteness:** Super Resolution outputs on all three routes were fully nonzero at both geometries (3 686 400 and 8 294 400 pixels) with zero non-finite half components; the same holds for all six Ray Reconstruction presets on the Vulkan and vkhip routes at 1440p and 4K.
- **Model selection:** `model.json` declares `preset: M` and `preset: L`; the RR manifests declare `preset 1`…`preset 6` for A–F, and the loader used those networks.
- **Native kernels:** `D4R_VKHIP_STAMPS=1` printed the chained layer timings (`b.enc1 … b.dec1`) for both SR presets; `d4r-rr` logged `vkhip: 11 / 10 / 14 Ray Reconstruction stages` for A/B/C, D/E and F, at both geometries. No run reported `vkhip unavailable`.
- **Route agreement:** SR `vkhip` and `hip` outputs are byte-identical to each other at both geometries, and differ from the engine's Vulkan network by 57.5 dB (M) and 55.4 dB (L) RGB PSNR. Ray Reconstruction vkhip outputs differ from the Vulkan engine's by 47.6–49.7 dB at 1440p and 48.1–50.5 dB at 4K, the range the native blocks have shown against NVIDIA's PTX reference; this is FP8 arithmetic, not a broken route.
- **Temporal state:** with a reset frame followed by a continued frame, every RR preset's final image changed on both routes at both geometries, so the history is read and written rather than re-derived each evaluation.

## Reproduction

Executables have to be built off the repository's NTFS mount — `ld.so` crashes on `$ORIGIN` resolution there — so the runners live in `/tmp/d4r-preset-compare-20261029/engine`, which means they have to be rebuilt after each reboot (the build takes about 20 seconds).

```sh
ROOT=$(git rev-parse --show-toplevel)
W=$ROOT/build/local-preset-compare-20261029
bash $ROOT/engine/build.sh /tmp/d4r-preset-compare-20261029/engine      # d4r-m, d4r-rr with HIP and vkhip
D4R_VKHIP_LAYERS=1 D4R_GPU_ARCH=gfx1201 D4R_NATIVE_FP8=1 \
    bash $ROOT/kernels/build.sh m $W/layers-m                            # the ten Swin layers, chained
M_EXACT=0 M_RESIDUAL_F32=1 M_WPW=2 D4R_NO_SCOPE=1 python3 $ROOT/engine/compile_m.py \
    $ROOT/build/linux-all-presets-handoff-20261022/private/captures/sr-m $W/models/sr-m --hip $W/layers-m
M_EXACT=0 M_RESIDUAL_F32=1 M_WPW=2 D4R_NO_SCOPE=1 python3 $ROOT/engine/compile_m.py \
    $ROOT/build/linux-all-presets-handoff-20261022/private/captures/sr-l $W/models/sr-l --hip $W/layers-m
python3 $ROOT/engine/compile_rr_cnn.py \
    $ROOT/build/linux-all-presets-handoff-20261022/private/dll/nvngx_dlssd_310.3.0.dll A $W/models-cnn/rr-a   # B and C likewise
D4R_GPU_ARCH=gfx1201 bash $ROOT/kernels/rr/build_vkhip_cnn.sh $W/models-cnn/rr-a      # and rr-b, rr-c
D4R_GPU_ARCH=gfx1201 bash $ROOT/kernels/rr/build_vkhip.sh $W/models/rr-d               # and rr-e
D4R_GPU_ARCH=gfx1201 bash $ROOT/kernels/rr/build_vkhip_f.sh $W/models/rr-f
python3 $ROOT/tools/make_local_sr_inputs.py $W/inputs/sr-1440p 1280 720 && python3 $ROOT/tools/make_local_sr_inputs.py $W/inputs/sr-4k 1920 1080
g++ -O2 -std=c++17 $ROOT/tools/make_local_rr_inputs.cpp -o /tmp/d4r-preset-compare-20261029/make_local_rr_inputs
/tmp/d4r-preset-compare-20261029/make_local_rr_inputs $W/inputs/rr-1440p 1280 720 2560 1440 140
/tmp/d4r-preset-compare-20261029/make_local_rr_inputs $W/inputs/rr-4k 1920 1080 3840 2160 140
python3 $ROOT/tools/bench_local_presets.py 1440p sr && python3 $ROOT/tools/bench_local_presets.py 4k sr \
    && python3 $ROOT/tools/bench_local_presets.py 1440p rr && python3 $ROOT/tools/bench_local_presets.py 4k rr
python3 $ROOT/tools/profile_local_rr_stages.py && python3 $ROOT/tools/verify_local_routes.py \
    && python3 $ROOT/tools/check_local_rr_4k_routes.py && python3 $ROOT/tools/build_local_comparison_summary.py
```

`bench_local_presets.py` prints each run's command; the exact commands are also recorded in `logs/suite-*.json` (written when those scripts still lived under `build/local-preset-compare-20261029/tools/`). One example SR run and one RR run:

```sh
D4R_M_WARMUP_PASSES=12 D4R_M_TIME_EACH=1 /tmp/d4r-preset-compare-20261029/engine/d4r-m \
    $W/models/sr-l 1280 720 2560 1440 $W/logs/sr-l-vkhip-trial0 37 vkhip -- \
    $W/inputs/sr-1440p/0-color.bin $W/inputs/sr-1440p/0-motion.bin $W/inputs/sr-1440p/0-depth.bin -0.25 -0.25 \
    ... four frames with the probe's jitter cycle
D4R_RR_TIME_REPEAT=1 D4R_RR_NO_UPLOAD=1 D4R_RR_NO_WRITE=1 D4R_ENGINE_VKHIP=1 \
    /tmp/d4r-preset-compare-20261029/engine/d4r-rr $W/models-cnn/rr-a $W/inputs/rr-1440p $W/logs/rr-a-vkhip-trial0
```

**Provenance** (`logs/environment.json` holds the SHA-256 of every runner, engine library, model manifest and code object):

| Component | Source | Version versus the NVIDIA reference |
|---|---|---|
| SR M, L models | `compile_m.py` from the retained captures of `nvngx_dlss.dll` 310.7.0 (`be6e434a…`) | 310.7.0 against NVIDIA's 310.9.16: a model-version mismatch |
| SR native layers | `kernels/build.sh m`, `D4R_VKHIP_LAYERS=1 D4R_NATIVE_FP8=1`, `gfx1201` | d4r kernels, built here |
| RR A, B, C | `compile_rr_cnn.py` from `nvngx_dlssd_310.3.0.dll` (`e43fdcea…`) | matches the NVIDIA A/B/C library exactly |
| RR D, E, F | the 2026-10-22 handoff models from `nvngx_dlssd_310.9.1.dll` (`4bc7ea5f…`) | 310.9.1 against NVIDIA's 310.9.16: a minor mismatch |
| RR vkhip code objects | `build_vkhip_cnn.sh`, `build_vkhip.sh`, `build_vkhip_f.sh`, `gfx1201` | built here; all spill-free (`private_segment_fixed_size: 0`) |
| NVIDIA references | `docs/benchmarks/nvidia-blackwell-{sr,rr}-20261029.json` | unchanged |

## Limits of this comparison

- SR M/L come from 310.7.0 weights and RR D/E/F from 310.9.1, against NVIDIA's 310.9.16; A/B/C match exactly. Ratios involving M/L or D/E/F therefore compare different model builds as well as different GPUs.
- The replay runners have no HDR or auto-exposure creation flags: the SR case runs the engine's exposure stage over the same patterned input, not NGX's flags `0x4b` path, and RR uses `preExposure=1` as the probe does.
- No CUDA backend and no ZLUDA exist on this machine, so "native HIP versus Vulkan" is the full backend choice here; the NVIDIA figures are not RTX 5070 Ti measurements and not game FPS.
- Runs are short (about one second of GPU work each), and the governor does not hold a steady clock: the sclk DPM state list is dynamic (snapshots showed `500 / 1826 / 2520 MHz` and `500 / 2765 / 2520 MHz`, `rocminfo` reports 2520 MHz nominal), sampled per-run medians spanned 1228–3178 MHz, GPU busy averaged 91%, and the heaviest runs sat at the 250 W cap (median-of-medians 242 W across all 108 runs, maximum sample 352 W). Every ratio therefore contains a power-limited clock as well as kernel cost. Interleaving and three trials bound run-to-run variation: the three trial medians for a case agreed within 1.3% for Super Resolution and 0.5% for Ray Reconstruction (median spread 0.4% and 0.2% across the 36 cases).
- Output equality across backends is PSNR-level, as noted above; nothing here certifies image quality against a reference render.
- The per-frame output dumps from the timed runs are not kept: `logs/` holds the suite JSON, per-run runner logs, the stage profile, the two route-verification records and `environment.json` with hashes. Byte-level outputs come from the verify runs above, which regenerate the inputs deterministically.
