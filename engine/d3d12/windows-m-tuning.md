# Preset M on the native Windows D3D12 backend: tuning notes

2026-10-23, RX 9070 XT, AMD D3D12 driver 32.0.32015. Replay runner `d4r-m-d3d12` on the Linux handoff's
`m-1440p` / `m-4k` cases (`build/linux-all-presets-handoff-20261022/cases`), 160 timed frames, stage timestamps
(`D4R_ENGINE_M_LAYER_STAMPS=1`). Quality: final RGB PSNR, fixed peak 1, against NGX's output through ZLUDA on a
12-frame 640x360 -> 1280x720 harness capture (the relaxed arithmetic on both Linux and D3D12 differs from NVIDIA's
by FP8 rounding flips; the target is the same loss as Linux, not equality with Vulkan).

| 854x480 -> 2560x1440, us | session start | now | Linux Vulkan (handoff, 250 W cap) |
|---|---:|---:|---:|
| total (GPU median) | 1118.8 | 984.7 | 982.5 |
| input | 78.6 | 67.0 | 88.5 |
| network | 515.3 | 491.5 | 444.5 |
| expansion | 46.3 | 38.0 | 34.3 |
| reconstruction | 364.4 | 302.8 | 265.4 |
| downsample | 107.6 | 77.2 | 148.7 |

4K (1280x720 -> 3840x2160): 2356 -> 2115 us (Linux 1748). PSNR against ZLUDA, last three frames: 34.26 dB at
the start, 34.29 dB now (per frame within 0.1 dB). The Vulkan SPIR-V of the package is byte-identical.

LDR (`compile_m.py --ldr`, NGX's `*_ldr_*` kernels; the runtimes skip the exposure passes for an `ldr` model):
the same harness scene captured with create flags 0x4a instead of 0x4b, the package built from that capture.
Last three frames against ZLUDA: 36.39 dB relaxed, 38.60 dB `M_EXACT=1` (HDR: 34.29 / 37.94). 1440p: 943.6 us
(no exposure, reconstruction 267 us without the range compression).

What changed (the shaders carry the details and measurements):
- Parameter blocks of enc0/dec0/post/down are read-only raw views (`ByteAddressBuffer`, scalar loads); from a UAV
  every lane loaded its own copy into vector registers. Runtime and runner bind them as `R` / `raw()`.
- post_m: tile as packed words (the driver read a shared `f16vec4` with 16-bit loads), accumulators as packed
  words, the row's sums selected rather than indexed (88 instead of 120 VGPRs in wave64).
- post and down convert to binary16 natively (`D4R_NATIVE_HALF`); enc0/dec0 keep the emulated rounding (each lost
  0.1 dB for 2-4 us).
- layer_m.hlsl `M_ACC32`: f32 accumulators across the 32-deep steps (what RADV effectively compiles from the relaxed
  GLSL), on for the 64-channel layers and the decoders only; enc2 and the tube are slower with it.
- dec0 and enc0: FP8 rounding on the hardware conversion (`fp8Of`), eight values a call, instead of per-value
  software rounding; byte-identical output.

Tried and dropped: wave32 post (slower, 390 us), two windows per wave in the layers (slower everywhere), PF/PFN
other than 4/8, split f32 accumulator chains, one window per workgroup for the tube (no gain).

Tube timing: at 1280x720 render (M at 4K, L at 1440p) the last three tube blocks take ~67 us against ~48 for the
first three once the GPU is warm, and ~50 for all six on the first run after a minute idle, with byte-identical
DXIL; not the weights (swapped in from blocks 2-4: unchanged). Clock/power management in long frames, not code;
Linux (250 W cap) 41 each. Compare timings at a matched thermal state. The Vulkan M runner (`d4r-m`) crashes on
the Windows driver even with the Linux package, so no same-machine Vulkan comparison exists. `M_EXACT=1` D3D12 is
37.9 dB from ZLUDA on the capture, not byte-identical (not investigated here).

## Preset L (2026-10-23)

The same backend runs L's packages (`compile_m.py` on an L capture builds the DXIL too; offsets.bin `D4RL0002`):
- input stage in two dispatches: `enc0_lf` (enc0_m.comp `L_FEATURES`: the 32 FP8 feature slots of every 8x8 block
  of L's grid to a buffer, the motion surfaces as M's) and `enc0_l` (layer_m.hlsl KIND 3: embedding straight into
  the block's registers, the C 32 block, the pixel-resolution skip, the patch merge to the network's tokens);
- `dec0_l` (KIND 4: patch expand + skip, the block, the per-pixel head) and `post_l` / `_q` / `_c` (post_m.comp
  `PRESET_L`, tiles 11 / 16 / 12); network, exposure and downsample as M's.
- runtime_m_d3d12.cpp (`MModel::isPresetL`), the runner and the shim (preset 12 from `engine\l` / `engine\l-ldr`
  or `D4R_ENGINE_L_MODEL_DIR` / `_LDR_DIR`; render-resolution motion only). Native D3D12 initialisation now also
  succeeds with only M or L packages.

Quality, the harness scene at 640x360 -> 1920x1080 captured through NVIDIA's L kernels on ZLUDA, last three frames:
relaxed 37.34 dB, `M_EXACT=1` 48.64 dB (its first compared frames 101 / 100 dB: the stages reproduce NVIDIA's).
f32 accumulation off changes nothing (37.2 dB). The shim's path through the harness replay (`D4R_HARNESS_REPLAY_DIR`
with the capture's inputs) is byte-identical to the runner on all 12 frames, M likewise.

| L, us | 1280x720 -> 2560x1440 | Linux | 1920x1080 -> 3840x2160 | Linux |
|---|---:|---:|---:|---:|
| total | 2329 | 1961 | 5085 | 4677 |
| input | 479 | 401 | 1084 | 963 |
| network | 1016 | 837 | 1990 | 1830 |
| expansion | 355 | 336 | 826 | 838 |
| reconstruction | 378 | 302 | 892 | 752 |
