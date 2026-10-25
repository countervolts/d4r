# Experimental vkhip network

`[Engine] ExperimentalVkHip = false` is the shipped default. To opt into native
preset-K HIP kernels recorded inside the game's Vulkan command list:

```ini
[Engine]
Enabled = true
ExperimentalVkHip = true
Network = vulkan
```

`D4R_ENGINE_VKHIP=0|1` overrides the INI. The explicit opt-in also enables K when
`Enabled=auto` would leave K on the normal CUDA path. `Enabled=false` still
turns the engine off. Other presets retain their existing routes.

The experimental loader currently requires a physically verified gfx1101
adapter (PCI device 1002:747e), Mesa RADV and compatible gfx1101 code objects.
It verifies the driver's cached shader layout before replacing code. A missing
model, unsupported device, or incompatible code object leaves the ordinary
engine network available and logs the reason. The model needs its original
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
