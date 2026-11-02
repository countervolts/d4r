# Official NVIDIA preset timings on Blackwell

Measured 2026-10-29 (Europe/Amsterdam) on rented RunPod GPUs. These are isolated NGX CUDA evaluation timings using NVIDIA's official Linux libraries, not game frame times or d4r execution times. The GPU was an **RTX PRO 4500 Blackwell**, with a 200 W power limit and driver **595.91.07**; it is a comparison GPU, not an RTX 5070 Ti. The same presets measured on this machine's RX 9070 XT are in [Local GPU preset comparison](local-gpu-preset-comparison.md).

## Super Resolution: L versus M

Official DLSS **310.9.16**, Performance mode (render dimensions half the output in each axis). CUDA events bracket each evaluation; each process runs 40 warmup and 100 measured frames. Results are the median of three trial medians, alternating M and L at each geometry.

| Output | Input | M | L | Extra L time | L overhead |
|---|---|---:|---:|---:|---:|
| 2560×1440 | 1280×720 | 1.081 ms | 1.413 ms | 0.332 ms | 30.7% |
| 3840×2160 | 1920×1080 | 2.553 ms | 3.161 ms | 0.607 ms | 23.8% |

At 4K, M's three trial medians were 2.552, 2.553 and 2.555 ms; L's were 3.147, 3.161 and 3.168 ms. At 1440p M varied more (1.046–1.086 ms), while L ranged from 1.405 to 1.415 ms.

The input was a synthetic patterned RGBA16F image with zero RG16F motion, R32F depth and alternating jitter. Creation flags were `0x4b`: HDR, low-resolution motion, inverted depth and auto exposure. Both used an RGBA16F output surface; all six preset hints were explicitly set to L or M. Output hashes were stable across repeated trials and differed between presets. No image-quality conclusion follows from this smoke check.

### Where L spends more time

Separate Nsight Systems 2025.1.1 CUDA traces confirmed L's unfolded kernels and M's folded kernels. The following stage times are medians of summed kernel durations per frame over 30 measured frames after 40 warmups. Profiling can affect execution; the unprofiled evaluation times above are the primary comparison.

| Stage, 4K Performance | M | L |
|---|---:|---:|
| Input (`enc0`) | 0.214 ms | 0.503 ms |
| Shared ten inner Swin layers | 1.078 ms | 1.092 ms |
| Expansion (`dec0`) | 0.117 ms | 0.345 ms |
| Reconstruction (`post`) | 0.836 ms | 0.917 ms |
| Downsample | 0.254 ms | 0.250 ms |
| Exposure | 0.038 ms | 0.037 ms |

L's input and expansion stages account for about 0.52 ms of extra kernel execution. They contain the additional render-resolution Swin blocks described in [L_NOTES.md](../engine/L_NOTES.md). Its reconstruction also evaluates a 4×4 neighborhood instead of M's 3×3. NVIDIA therefore pays an architectural penalty too, while the inner network costs almost the same.

In d4r's vkhip route, only the ten inner layers use the native HIP kernels; L's outer blocks still execute as Vulkan shaders. This leaves implementation-specific optimization headroom. The NVIDIA measurements do **not** establish what fraction of a user's approximately 6 ms L execution time comes from architecture versus d4r's implementation: that requires comparable L/M stage profiles on that user's GPU. The older L/M examples in [experimental-vkhip.md](experimental-vkhip.md) use different input cases and cannot establish that fraction either.

A supplied Cyberpunk 2077 chart shows 103 versus 102 FPS for M/L at 1440p Performance on an RTX 5070 Ti. That is about 0.095 ms difference in **total game frame time**, rather than isolated DLSS cost. A sub-millisecond DLSS penalty is a smaller percentage of the whole game frame; scene, library version, GPU, resource contention and overlap also prevent directly predicting those FPS values from this synthetic Linux CUDA test.

### SR artifacts and cost

The [SR numerical summary](benchmarks/nvidia-blackwell-sr-20261029.json) retains trial medians, stage timings, hashes and cost. Raw frame logs, GPU information, Nsight reports, SQLite exports, source and summary script are under `build/nvidia-lm-20261029/`. The library SHA256 is `30982c937afed85c96b8d313323de9a650a2eec6de1bb0f714b21750f4ce98f6`. Additional ratio-3 runs used MaxPerf creation quality and are auxiliary tests, not standard Ultra Performance-mode measurements.

The pod ran for about 289 seconds at its actual quoted $0.72/hour: approximately **$0.058**, not a billing receipt. The advertised $0.34/hour minimum was unavailable for that secure-cloud deployment. All artifacts were downloaded, the pod was terminated, and a subsequent account query returned no pods. No persistent volume was created.

## Ray Reconstruction: A through F

A second RTX PRO 4500 Blackwell rental, again at 200 W with driver 595.91.07, measured the RR presets at the same Performance-mode geometries. These timings include RR's complete denoising and upscaling evaluation; they are not an additional pass to add to the SR table.

| RR preset | Official library | 1280×720 → 2560×1440 | 1920×1080 → 3840×2160 |
|---|---|---:|---:|
| A | 310.3.0 | 1.358 ms | 2.979 ms |
| B | 310.3.0 | 1.353 ms | 2.983 ms |
| C | 310.3.0 | 1.358 ms | 2.988 ms |
| D | 310.9.16 | 1.860 ms | 4.055 ms |
| E | 310.9.16 | 1.863 ms | 4.057 ms |
| F | 310.9.16 | 2.100 ms | 4.658 ms |

**D and E have effectively the same execution cost. F costs about 13% more at 1440p and 15% more at 4K.** The legacy CNN presets A/B/C cost about the same as each other; current D/E are about 36% more expensive than that group at 4K, and F about 56% more expensive. These are performance results, not an assessment of denoising quality or a recommendation to replace newer models with older ones.

### Library versions and model selection

Current [NVIDIA RR definitions](https://github.com/NVIDIA/DLSS/blob/9981c6350dc5/include/nvsdk_ngx_defs_dlssd.h) remove A/B/C and direct applications to D or E. Testing hints 1/2/3 against the current library would not measure the legacy CNNs. A/B/C were therefore measured with the official [310.3.0 library](https://github.com/NVIDIA/DLSS/tree/a8ed84e2d1cc/lib/Linux_x86_64/rel), and D/E/F with [310.9.16](https://github.com/NVIDIA/DLSS/tree/9981c6350dc5/lib/Linux_x86_64/rel).

To check the effect of mixing versions, D/E were also measured with 310.3.0:

| RR preset, 310.3.0 | 1440p Performance | 4K Performance |
|---|---:|---:|
| D | 1.856 ms | 4.071 ms |
| E | 1.860 ms | 4.062 ms |

The old/current D/E timings differ by at most about 0.4%, and each preset's final output hash matches across the two libraries in this workload. This supports the comparison for this test; it does not establish version equivalence for every scene or API.

Each run logs the actual snippet filename from `/proc/self/maps`, including libraries NGX loads into separate linker namespaces. Separate 4K Nsight traces confirm the CNN `Enc*`/`Dec*`/`PDL` pipeline for A/B/C, the `swin_*` transformer pipeline for D/E, and the distinct `rr2_*` pipeline for F. B/C produce matching final output hashes; D/E have similar cost but different output hashes.

### RR method, artifacts and cost

As with SR, the primary table uses CUDA events around each evaluation, 40 warmup frames and 100 measured frames, reporting the median of three trial medians. Each trial cycles through the presets at each geometry. Separate Nsight runs use 40 warmups and 10 measured frames; their durations are not used for the primary table.

The probe renders one noisy static scene using [fsr_rr_scene.h](../tools/fsr_rr_scene.h), then reuses that input with zero motion and alternating jitter. Guides are world normals with packed roughness (RGBA16F), linear diffuse/specular albedos (RGBA8 UNORM), linear depth (R32F), and specular hit distance (R32F, constant 4). It supplies the camera's view/projection matrices, an exposure texture of 1, and an RGBA16F output surface. Creation flags are `IsHDR|MVLowRes` (`0x3`), roughness mode is packed, and hardware depth is disabled. All six RR preset hints are set explicitly. A 64 MiB scratch allocation is supplied outside timing.

All **48 measured runs** completed successfully, produced nonzero output with no nonfinite values, and gave stable final output hashes across their three trials. The output surface starts with a NaN sentinel. This verifies execution and output coverage, not image quality: the input is reused rather than rendering and uploading a new noisy frame per evaluation, and no ground-truth comparison is made. Timings exclude initialization, scene generation, texture uploads and final output readback. CPU launch gaps within an evaluation remain included.

The [RR numerical summary](benchmarks/nvidia-blackwell-rr-20261029.json) preserves trial medians, output/library/source hashes, warmed kernel profiles and cost. Raw logs, Nsight reports, SQLite exports, the source snapshot and summarizer are under `build/nvidia-rr-20261029/`. The rental lasted about 340 seconds at $0.72/hour, approximately **$0.068** additional cost. The pod was terminated after downloading all results, and the account then listed no pods. No persistent volume was created.

## Reproduction

The follow-up timing probe is [tools/bench_official_ngx_cuda.cpp](../tools/bench_official_ngx_cuda.cpp). It uses NVIDIA's NGX core and SDK rather than calling snippet exports directly. Obtain the official SDK headers, `libnvsdk_ngx.a`, and the selected Linux snippet from [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS). NVIDIA's files retain their licensing terms.

Compile on a CUDA development environment, adapting the library paths to the installation:

```sh
g++ -O2 -std=c++17 -I/path/to/DLSS/include -I/usr/local/cuda/include \
    tools/bench_official_ngx_cuda.cpp /path/to/DLSS/lib/Linux_x86_64/libnvsdk_ngx.a \
    -L/usr/local/cuda/targets/x86_64-linux/lib/stubs -lcuda -ldl -pthread -o bench
./bench /absolute/path/to/snippet-directory 13 1920 1080 3840 2160 100
# Current RR preset F:
D4R_BENCH_RR=1 ./bench /absolute/path/to/310.9.16-directory 6 1920 1080 3840 2160 100
# Legacy RR preset A (310.3.0 lacks the newer Init1/CreateFeature1 exports):
D4R_BENCH_RR=1 D4R_BENCH_LEGACY=1 \
    ./bench /absolute/path/to/310.3.0-directory 1 1920 1080 3840 2160 100
```

Each snippet directory should contain only the selected library version. The probe reads output back after timing and checks for nonzero, finite output, with a NaN sentinel initialized before evaluations. The SR measurements above used the earlier probe retained with their raw artifacts; the follow-up probe adds RR guide inputs and stronger output checks.
