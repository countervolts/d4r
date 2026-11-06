# Experimental vkhip network

`[Engine] ExperimentalVkHip = true` is the shipped default: preset K runs its
native HIP kernels recorded inside the game's Vulkan command list. With
`Enabled = auto` on RDNA3, K uses vkhip when it loads and otherwise stays on the
native kernels (the engine's Vulkan network is slower there). With
`Enabled = true`, K falls back to the engine's Vulkan network instead.

```ini
[Engine]
ExperimentalVkHip = false   ; turn it off
```

`D4R_ENGINE_VKHIP=0|1` overrides the INI. `Enabled=false` still turns the engine
off. Other presets retain their existing routes.

## Preset M

With `ExperimentalVkHip` on, preset M runs the same way on RDNA3. `D4R_ENGINE_M_VKHIP=0` turns off M and L. M's five native layer kernels run inside the game's command list, with Vulkan stages around them. Those stages use f16
matrix operands, which hold every FP8 value exactly, and FP8 rounding in packed f16 arithmetic (`-DWIDE=1`). Their matrices are stored transposed, so a matrix operand is two 16-byte loads instead of sixteen 2-byte ones. The reconstruction and
downsample run as wave32. The half-resolution history is RGB10A2, the format of NGX's surface, which is about 0.09 ms cheaper at 1080p to write and filter than the RGBA32F image the CUDA backend has to use for it. Output uses f32 stores and
raw copies, as K's does. The network skips the windows nothing reads, past the render size (`D4R_ENGINE_M_TRIM=0` runs them); the frames are byte-identical either way.

With `PreferAccuracy`, or when the model has no `hipnet.bin` or `post_m_wide.spv` (rebuild the models), M stays on the CUDA backend. A change of render size also
moves the feature to the CUDA backend, because the network's token grid is fixed when it is created.

Harness results on an RX 7700 XT, with total GPU time per frame on the CUDA backend and then on vkhip (RGBA16F and RGBA8 from two A/B/B/A sets, RGB10A2 and R11G11B10F from one):

| Output, Performance mode | RGBA16F output | RGBA8 colour and output | RGB10A2 output | R11G11B10F output |
|---|---|---|---|---|
| 1080p | 3.03 → 2.35 ms | 3.30 → 2.29 ms | 3.31 → 2.33 ms | 3.32 → 2.35 ms |
| 1440p | 5.25 → 4.31 ms | 5.81 → 4.15 ms | 5.82 → 4.26 ms | 5.80 → 4.25 ms |
| 4K | 11.65 → 9.92 ms | Not measured | Not measured | Not measured |

Stage times at Performance mode, RGBA16F output (ms):

| Output | Exposure | Input | Network | Expansion | Reconstruction | Downsample |
|---|---|---|---|---|---|---|
| 1080p | 0.016 | 0.132 | 1.692 | 0.086 | 0.291 | 0.123 |
| 1440p | 0.035 | 0.240 | 3.028 | 0.154 | 0.583 | 0.255 |

In Ghost of Tsushima at Performance mode (RGBA8, no camera input from the test) it measured 2.35 ms at 1080p and 4.22 ms at 1440p, with frame generation running on top of it. That is one run per setting, in one game.

Its distance from the reference matches the CUDA backend's within FP8's usual variation: 37.60, 48.40, 43.11 and 38.53 dB
on the motion, jitter, pan and LDR scenes, against 38.03, 48.25, 43.14 and 38.74 dB.

The experimental loader requires an RDNA3 GPU (gfx1100, gfx1101, gfx1102 or
gfx1103), Mesa RADV and code objects built for that GPU. It identifies the real
render adapter from sysfs (its PCI vendor and the amdgpu KFD node at its PCI
address), so spoofed device IDs do not matter. `build_engine_models.sh` packages
the shipped `kernels/<gfx>/` layers for the GPU it runs on; after changing GPUs,
rebuild the engine models. It verifies the driver's cached shader layout before
replacing code. A missing model, unsupported device, or code object for another
GPU leaves the ordinary engine network available and logs the reason. Only
gfx1101 (RX 7700 XT) has been tested on hardware. The model needs its original
`hipnet.bin` and `hip/dltss_pwin_*_layer.hsaco` files. It uses original native
network instructions and weights; it does not run HIP on another queue.

Image handling is shared with `GameUpscaler`, rather than the deleted
prototype's fixed-format translated texture stages. Both regular and inverted
depth, render/display motion, texture rectangles, HDR/LDR, game/automatic
exposure and the adapter's normal format families are available:

| Input | Formats |
|---|---|
| Color/output | RGBA16F, RGBA32F, R11G11B10F, RGB9E5, RGBA8/BGRA8 UNORM, RGBA8 SNORM, RGBA16 UNORM/SNORM, RGB10A2 UNORM |
| Motion | RG16F, RG32F, RG16 SNORM, RGBA16F, RGBA32F |
| Depth | R32F, R16F, R16 UNORM, D32F, D16/D24 depth views |
| Exposure | The existing sampled float/UNORM exposure formats |

This does not mean every DXGI enum is meaningful as a DLSS input. Compressed,
multisampled, aliased and incompatible integer images retain validation and
fallback. Packed D16/D24 depth must have a sampleable depth view; its normalized
values must not be treated as raw float bits. Device format features still apply.
Output formats without a direct storage view use the adapter's conversion path.

The shared Vulkan input/reconstruction shaders have the existing engine's
numerics. This path does not claim byte identity to NVIDIA's CUDA texture stages
or a performance gain over the deleted exact prototype. Keep this opt-in until
per-game rendering, motion quality, frame generation and timing have been tested.

Build uses `scripts/build_vkhip_donor.py` and the source in `engine/vkhip/`.
The donor is generated from ordinary GLSL; no NVIDIA data or AMD code object is
embedded in source. Kernel and model inputs remain local runtime artifacts.

## Preset L

Preset L uses the same ten layers. Its input and expansion stages contain a full attention block, so the native kernels do that
work too (`kernels/tex/enc0l_net.hip`, `dec0l_net.hip`, built for every RDNA3 target). They use the numerics of the shared layers (`D4R_L_WIDENED`: f32 accumulators, f16
intermediates, FP8 where a result is stored, [approximations.md](approximations.md)). Two small Vulkan stages handle the textures
around them: `engine/enc0_l_gather.comp` gathers the features and `engine/dec0_l_head.comp` writes the reconstruction's inputs.
Reconstruction, downsample, the half-resolution history and the window skipping are M's. L stays on the CUDA backend for display-resolution motion and `PreferAccuracy`,
and moves there when the render size changes. M and L have only run on gfx1101.

Harness results on an RX 7700 XT at Ultra Performance, with total GPU time per frame on the CUDA backend and then on vkhip (1080p and 1440p and 4K RGBA16F from two A/B/B/A sets, 4K RGBA8 from one):

| Output | RGBA16F | RGBA8 colour and output |
|---|---|---|
| 1080p | 2.92 → 1.92 ms | 3.07 → 1.87 ms |
| 1440p | 4.79 → 3.37 ms | 5.18 → 3.30 ms |
| 4K | 10.49 → 7.66 ms | 11.40 → 7.52 ms |

Stage times at Ultra Performance, RGBA16F output (ms; the network includes the two native stages):

| Output | Exposure | Input | Network | Head | Reconstruction | Downsample |
|---|---|---|---|---|---|---|
| 1080p | 0.009 | 0.052 | 1.438 | 0.022 | 0.294 | 0.096 |
| 1440p | 0.010 | 0.085 | 2.456 | 0.034 | 0.537 | 0.238 |

In Ghost of Tsushima at Ultra Performance (RGBA8, no camera input from the test) it measured 1.89 ms at 1080p and 3.30 ms at 1440p, one run per setting.

Its distance from the reference matches the CUDA backend's within FP8's usual variation: 36.80, 50.06, 44.73 and 36.39 dB on the
motion, jitter, pan and LDR scenes, against 37.03, 49.91, 45.27 and 36.28 dB.
