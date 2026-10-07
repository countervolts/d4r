# Performance

Unless stated otherwise, the frame rates and kernel timings on this page were measured on an RX 7700 XT (gfx1101). Native FP8 changes both arithmetic and cost, so the gfx11 results cannot be projected to RDNA4; the one RDNA4 measurement is in [RDNA4](#rdna4). No RDNA4 game frame rate has been measured.

## Method

- **Hardware and settings:** Radeon RX 7700 XT (RDNA3, 54 CUs), Ryzen 9 5900XT, Linux 6.18. SILENT HILL Townfall at 2560×1440 output.
- **Route:** every run plays the same scripted 62-second walk through a street, with fog, wires, fences and signage.
- **Frame rate:** from MangoHud's per-frame log over the walk, as frames divided by the sum of frame times.
- **Pairing:** runs are compared back-to-back in one session, because repeated runs of an identical setup vary by about 1%.
- **Screen recording:** it costs a few fps, so recorded videos show slightly lower numbers than the table.
- **Latency:** all DLSS numbers are same-frame (frame age 0). Every frame shows its own DLSS result, as with a native upscaler.
- **Upscaler modes:** every mode uses OptiScaler's render-ratio override (1.5, 1.72, 2.0, 3.0), so the result does not depend on the DLSS mode chosen in the game.

## Results

| Mode (render resolution) | DLSS 3 CNN (E) | DLSS 4 (K) | DLSS 4.5 (M) | FSR 4\* |
|---|---|---|---|---|
| Quality (1705×960) | **71.7** | 67.9 | 49.3 | 76.0 |
| Balanced (1488×837) | **80.8** | 75.9 | 58.0 | 84.5 |
| Performance (1280×720) | **89.7** | 84.1 | 69.0 | 94.0 |
| Ultra Performance (853×480) | 89.1 | **95.5** | 92.2 | 107.3 |
| Native 2560×1440, no upscaling (TSR at 100%) | | 49.1 | | |

\* FSR 4 and the native row are from 2026-09-27; the DLSS columns from 2026-09-29, when the machine ran about 2–4% slower overall (release 0.1.1 at Quality that day: E 71.2, K 67.9, M 48.4 fps, against 72.4, 69.4 and 51.5 two days earlier). The render resolution of every mode is set with OptiScaler's ratio override for whichever DLSS mode the game is set to (ratios 1.5, 1.72, 2.0 and 3.0).

DLSS 4's cost is almost constant across modes (about 2.8 ms of GPU time per frame). Its network runs on a grid set by the output resolution, not the render resolution, so the gap to FSR 4 widens as the render resolution drops. The CNN gets more expensive at the 3× ratio, so K is the better choice at Ultra Performance.

## What each step contributed

DLSS 4 (K) at Quality, frames per second on the walk:

| Step | fps |
|---|---|
| ZLUDA only: the transformer's outputs were non-finite, so the network had no effect | 58.5\* |
| All eleven K layers native | 65.0\* |
| Wide deep layers, f32 accumulation, cheaper operand transposes | 66.0 |
| NGX's CPU syncs removed, faster marker polling, no sync before the output copy | 66.7 |
| DLSS queued behind a GPU-side wait for the inputs (GPU busy 95% → 99%) | 68.0 |
| Native output kernel writes the game-side buffer directly | 68.4 |
| NGX samples the input buffers in place (no array copies) | 69.4 |

\* Measured from the shim's frame log on a similar stretch, before the MangoHud method was in use.

DLSS 4.5 (M) at Quality went from 28 fps (ZLUDA only) to 42 with the first native Swin layers, 46 with more waves per window, 50 with fast numerics and the texture-kernel tails, and 51.5 with the hand-off changes.

## DLSS 4 (K) tuning round, 2026-10-04

A separate work copy of this tree tuned the K kernels with byte-identical output (see [native-kernels.md](native-kernels.md)). Numbers from that copy, RX 7700 XT, D3D12 harness, motion scene, FrameAge 0, warmed medians of back-to-back runs; the shipped 0.1.4 kernels against the tuned set:

| | shipped | tuned |
|---|---|---|
| 2560×1440 Quality: DLSS GPU time (`gpu_eval`) | 2.89 ms | 2.41 ms |
| 1920×1080 Quality: DLSS GPU time | 1.86 ms | 1.50 ms |
| 2560×1440 Quality: sum of kernels | 2.90 ms | 2.40 ms |

In SILENT HILL Townfall (static scene, K Quality 1706×960 → 2560×1440, MangoHud, two alternating rounds): 63.1 / 63.3 fps → 65.4 / 65.1 fps with default settings (frame time −0.5 ms), DLSS GPU time 2.92 → 2.41 ms. At 1920×1080: 86.1 → 89.3 fps. DLSS 4.5 (M) gained little from its own round (DLSS GPU time 7.96 → 7.91 ms at 2560×1440 Quality, Townfall 1080p +0.7%).

These were measured in the work copy. They have not been re-measured from this repository since the port.

## Experimental: removing the output copies

After DLSS, the result normally goes from the shared buffer into the game's output texture with one copy, or with a copy and a blit when the output is not RGBA16F. Two opt-in settings remove those passes:

- `D4R_SHIM_OUTPUT_PACKED=1` (preset M): the native output kernel stores R10G10B10A2 texels itself, rounding as the RGBA16F store followed by the blit does, so the conversion image and blit are not needed.
- `D4R_SHIM_INPLACE=1` with `VKD3D_D4R_LINEAR_TEXTURES=1` (vkd3d-proton patch 0003): committed single-level RGBA16F and R10G10B10A2 textures are created linear on exportable memory, the shim imports the output texture into HIP, and the output kernel writes it directly.

Measured on an RX 9070 XT (gfx1201) in the D3D12 harness, preset M at 1280×720 → 3840×2160, median frame interval of 4 alternating runs of 400 frames. The harness runs DLSS back to back with no game rendering.

| Output texture | Passes after DLSS | Frame interval |
|---|---|---|
| RGBA16F, copy | 1 | 5.76 ms |
| RGBA16F, in place | 0 | 5.52 ms |
| R10G10B10A2, copy and blit | 2 | 5.97 ms |
| R10G10B10A2, packed store | 1 | 5.76 ms |
| R10G10B10A2, packed store in place | 0 | 5.65 ms |
| R10G10B10A2, accuracy kernels, copy and blit | 2 | 6.35 ms |
| R10G10B10A2, accuracy kernels, packed store in place | 0 | 5.91 ms |

Every variant was byte-identical to the path it replaces on moving scenes (`D4R_HARNESS_OUTPUT_RGB10A2=1` gives the harness an R10G10B10A2 output). In Ghost of Tsushima, which passes an R10G10B10A2 output, both settings engaged from the fourth frame.

Limits: the harness does not render into its textures, so the cost of linear render targets to a game is not measured, and every committed single-level texture of those two formats becomes linear, not only the DLSS output. The packed store is only built into M's downsample kernel. A frame recorded for packed output whose output kernel turns out not to be the native one is converted on the CPU; that path has not been exercised. The input copies remain; see [in-place-inputs.md](in-place-inputs.md).

## What did not help

- **Accumulating in f16 on the WMMA units.** The error grows about 20× and the image degrades.
- **Occupancy tweaks** (VGPR caps, waves-per-EU hints) and **persistent work-groups.**
- **Non-temporal hints** for activation traffic.
- **Splitting the position-only layers by channel** instead of by token. It halves weight traffic but serialises the compute.
- **Mapping the Vulkan buffers directly as HIP arrays.** ROCm 7.2 does not export `hipExternalMemoryGetMappedMipmappedArray`.
- **Signalling the game's Vulkan timeline semaphore from HIP.** HIP does not import timeline semaphores, so the end of DLSS is still signalled from the CPU.

## Where the time goes now

For DLSS 4 at Quality the GPU spends about 2.8 ms per frame in DLSS kernels:
- NVIDIA's output kernel: 0.82 ms. It is ALU-bound at close to the GPU's instruction rate.
- The eleven network layers: 1.6 ms.
- The input kernel, exposure and miscellaneous kernels: about 0.35 ms.

The game-side input and output copies add about 0.3 ms. [native-kernels.md](native-kernels.md) lists every kernel.

## RDNA4

DLSS 4.5 (M) on an RX 9070 XT (gfx1201, native FP8, fast set), 1280×720 → 3840×2160 in the D3D12 harness with DLSS 310.7. Median GPU time per frame in milliseconds, from `D4R_CUDA_KERNEL_PROFILE` (frames 21–40); "before" is the same texture set with the previous Swin layers.

| Kernel | before | now |
|---|---|---|
| enc3 tube (6 launches) | 1.50 | 0.52 |
| enc1 | 1.32 | 0.33 |
| dec1 | 0.94 | 0.33 |
| post (texture kernel) | 0.67 | 0.68 |
| enc2 | 0.51 | 0.19 |
| dec2 | 0.50 | 0.17 |
| downsample (texture kernel, now wave64) | 0.36 | 0.34 |
| dec0, enc0 (texture kernels) | 0.18, 0.13 | 0.18, 0.13 |
| **total** | **6.10** | **2.86** |

Fast and accuracy sets, on the native-FP8 path and on the 16-bit path (`NativeFp8` off), before and after the change, same harness and resolution. Median GPU time per frame in milliseconds from `D4R_CUDA_KERNEL_PROFILE`; the bracketed figures are the whole evaluation (`D4R_PROFILE`) with the GPU otherwise idle and at about 80% utilisation.

| Set | before | now |
|---|---|---|
| native FP8, fast | 6.11 (6.01 / 5.89) | 2.88 (2.77 / 2.38) |
| native FP8, accuracy | 6.38 (6.49 / 6.32) | 3.15 (3.06 / 2.59) |
| 16-bit, fast | 3.48 | 3.40 |
| 16-bit, accuracy | 4.38 | 4.20 |

The 16-bit path never re-encoded its operands, so it was already faster than the previous FP8 layers; its small gain here is the wave64 downsample (and post, in the accuracy set). Against a 9070 XT that was running with `NativeFp8` off, the FP8 fast set is about 17% faster at 1280×720 and 22% at 1920×1080 → 3840×2160 (4.63 ms against 5.94 ms, the latter with a 16-bit set from an earlier build), not twice as fast. In Ghost of Tsushima at 1920×1080 → 3840×2160 the upscaler time shown in game went from 7.0 ms (16-bit fast set) to 5.8 ms (this FP8 fast set).

The shim's whole-evaluation GPU time (`D4R_PROFILE`, no per-kernel synchronisation) went from 5.9–6.0 ms to 2.80 ms with the harness's 200 ms pause between frames, and to 2.39–2.43 ms with frames 0–4 ms apart (2000 frames at 4 ms spacing: median 2.39, maximum 2.56). The difference is the driver's clock governor: with the 200 ms pause the GPU is about 9% busy and runs at about 2790 MHz, with frames 0–4 ms apart it is 80–98% busy and runs at 3180–3300 MHz, as it does under a game's rendering load. Other GPU users on the desktop add occasional slower frames to any of these figures; the previous kernels show the same disturbances.

The change is in [native-kernels.md](native-kernels.md#rdna4): the FP8 layers keep their activations as e4m3 bytes, and the downsample kernel is built as wave64 on gfx12. The output image is byte-identical before and after.
