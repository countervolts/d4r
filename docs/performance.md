# Performance

The pre-existing Super Resolution game benchmarks below were measured on an RX 7700 XT (gfx1101); they are historical results, not measurements from this RR update. All new Ray Reconstruction measurements and Cyberpunk verification in this update use RX 9070 XT (gfx1201). The RR timings are synthetic harness measurements, not RDNA4 game-FPS benchmarks. Native FP8 may change both arithmetic and cost, so the gfx1101 results cannot be projected to gfx1201.

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

## Ray Reconstruction VRAM transport

Measured separately from the Super Resolution game benchmarks above: RX 9070 XT (gfx1201), GE-Proton11-3, CUDA-capable DLSS-Denoiser 310.7, RR preset E, native FP8 WMMA and native Swin encoders. The synthetic D3D12 harness rendered 640×360 into 1280×720 with separate R32F alpha. Split-frame presentation and pitch-linear inputs were enabled; verification/capture downloads were disabled.

Four back-to-back 40-frame runs used host guides, GPU guides, GPU guides, then host guides. The first five frames of each run were excluded; values below are warm-frame medians in milliseconds:

| Guide transport | Worker time, first/repeat | Frame interval, first/repeat |
|---|---:|---:|
| Host staging | 14.967 / 14.296 | 17.373 / 16.795 |
| Shared VRAM | 13.960 / 13.953 | 16.482 / 16.680 |

All four final RGBA captures were byte-identical. The GPU guide runs recorded no host staging readbacks. This is a small synthetic transport comparison, not a game-FPS claim; network execution and clocks contribute to run-to-run variation.

With the usual two RGBA16F and two RGBA8 guides, VRAM transport eliminates approximately 11.1 MB of guide payload crossing GPU↔CPU per frame at this render size (5.5 MB in each direction, excluding padding). Separate alpha also no longer forces RGB through host staging. Unsupported raw-integer guide/alpha formats retain their existing host conversion without disabling the other GPU routes.

Correctness checks additionally covered five guide-format sets at an odd 193-pixel input width: 35 final/per-frame RGBA files were byte-identical between host and GPU guide transport. Six-frame source/presentation captures preserved RGB bytes and alpha subrect sentinels. R32F alpha was exact; half/UNORM blits can differ from the CPU quantizer by one destination step.

An aligned 192×108→288×162 synthetic capture against native CUDA on RTX 4090 measured RGB PSNR 42.15 dB, mean absolute component error 0.00357, and maximum error 0.13623; neither image contained nonfinite RGB values. This spot check does not establish RTX image-quality parity or game-wide temporal correctness. Transport A/B comparisons on the same AMD backend were exact; cross-backend neural arithmetic remains a separate accuracy limitation.

The optimized route also rendered a loaded Cyberpunk 2077 2.31 save at 640×360→1920×1080 Ultra Performance with path tracing and RR preset E on gfx1201. The overlay identified DLSSD 310.7.0; the shim created CUDA feature 13 with split-frame VRAM interop. In-game verification found zero differing bytes for all six supplied guide arrays, and the user verified gameplay. This diagnostic run enabled GPU readbacks and does not establish an FPS improvement or a game-wide artifact/latency audit.

The extended diagnostic log also recorded prolonged split-frame waits and a GPU-marker timeout later in the run; their cause was not isolated. The successful rendering and guide comparisons are not evidence of stall-free operation.

