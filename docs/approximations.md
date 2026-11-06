# Approximations on RDNA3

This page explains exactly where d4r's DLSS Super Resolution output on RDNA3 differs from NVIDIA's arithmetic, and which of those differences are real approximations.

**Short answer:** the default K, M and L routes use no approximations on RDNA3. They are also not bit-exact: they keep more precision than NVIDIA in some steps, which changes the result slightly. Colour and motion stay f32 when half can't hold their format, and output reaches the game's format without passing through half. [Limits](#limits) lists the routes this doesn't cover, chiefly `PreferAccuracy` and the host route.

Everything here was checked on Linux with an RX 7700 XT (gfx1101) on 2026-11-05. The other RDNA3 targets (gfx1100, gfx1102, gfx1103) run the same sources but were not tested on hardware. This page does not cover Ray Reconstruction, frame generation, or RDNA2 and RDNA4.

## What counts as an approximation

"Approximation" is often used loosely, so this page uses one test. A step is an **approximation** if either of these is true:

1. It holds or multiplies a value in a narrower number format than NVIDIA's kernel does at that step.
2. It computes a different function, meaning the result would differ even with perfect arithmetic.

Everything else falls into one of two harmless categories:

| Term | Meaning | Example |
|---|---|---|
| **Widening** | Same formula, wider format. NVIDIA rounds here and d4r does not. | f32 accumulators where NVIDIA rounds to f16 |
| **Rounding change** | Same formula, same format, but a rounding happens in a different place. | a multiply and add fused into one rounding |

Neither is an approximation, but neither is bit-exact. Widening is closer to the true math and further from NVIDIA's output, because NVIDIA's rounding is part of what its kernels compute. It does not establish a better image.

"NVIDIA's kernel" below means the PTX inside `nvngx_dlss.dll`. Measurements use that PTX running on the RX 7700 XT through ZLUDA, not a real RTX card.

## Summary

| Model | Default route on RDNA3 | Inside the upscaler | At the input and output | Distance from NVIDIA's kernels |
|---|---|---|---|---|
| **K** | vkhip: native HIP layers inside Vulkan | No approximation (widening and rounding changes) | None | 49.7–59.7 dB |
| **M** | vkhip: native HIP layers inside Vulkan | No approximation (widening) | None | 37.6–48.4 dB |
| **L** | vkhip: native HIP kernels inside Vulkan | No approximation (widening) | None | 36.4–50.1 dB |

The dB ranges cover the four test scenes in [Measurements](#measurements). Low M and L figures do not indicate approximation; [Why M and L score lower](#why-m-and-l-score-lower) explains them.

## What runs by default

The route depends on the model and on what the GPU exposes. RDNA3 has f16 cooperative matrices but no `VK_EXT_shader_float8`:

| Model | Route | Why | Falls back to |
|---|---|---|---|
| K | vkhip: exposure, input and output stages in Vulkan, the 11 native layers recorded in the same command list | fastest K path on RDNA3 ([experimental-vkhip.md](experimental-vkhip.md)) | native kernels on the CUDA backend |
| M | vkhip: Vulkan stages with f16 matrices, and M's native layers in the same command list | fastest M path on RDNA3 ([experimental-vkhip.md](experimental-vkhip.md)) | native kernels on the CUDA backend |
| L | vkhip, with native stages and small Vulkan texture stages | fastest L path; the engine's own L stages need FP8 pipelines that RDNA3 cannot create | native kernels on the CUDA backend, then translated PTX, per kernel |

`d4r_nvngx.log` shows which route ran:

```text
experimental vkhip ready on gfx1101: 14 native dispatches, fast layers, ...
engine backend: K reconstruction uses fast stages
engine backend (M): feature 1 runs preset M in the game's command list (960x544 -> 1920x1080), network on vkhip
engine backend (L): feature 1 runs preset L in the game's command list (640x360 -> 1920x1080), network on vkhip
```

## Preset K

Default K has four parts: exposure, input, the 11-layer network, and output. All of its differences from NVIDIA's kernels are listed below.

### Network

The fast layers in `kernels/k` differ from the accuracy set in these steps:

| Step | NVIDIA's kernel | Default | Kind |
|---|---|---|---|
| Matrix accumulators in enc0, enc1, enc2, dec0, dec1, dec2 | f16, rounded after every k16 step | f32 through the chain, rounded where the value is used (`PWIN_F32ACC`) | Widening |
| Matrix accumulators in enc3, enc4, dec3, dec4, dec5 | f16, rounded after every k16 step | Same | None |
| Sum of squares in the L2 norm | fixed f16 tree | f32 sum of the f16 squares, rounded to f16 once | Widening |
| Softmax row sum | fixed f16 tree | f32 sum, rounded to f16 once | Widening |
| Residual in enc0 and dec0 | folded into the output projection, so it is rounded into every k16 step (`PWIN_RESIDUAL_IN_WO`) | added once afterwards | Rounding change |

The operands stay f16 in both sets. Every matrix instruction in the fast and accuracy sets is `v_wmma_f32_16x16x16_f16`, which takes f16 inputs and accumulates in f32. The fast layers simply skip the extra rounding back to f16 after each step.

Keeping the accumulator in f16 is not faster either. A native f16-output WMMA variant was slower (enc0 about 0.27 to 0.30 ms) because the f16 results sit in half-register slots that need extra packing, and it was not exact.

### Input, output and exposure stages

These Vulkan stages (`engine/input_k.comp`, `engine/output_k.comp`, `engine/exposure_k.comp`) reimplement NVIDIA's kernels, and `PreferAccuracy` selects stricter builds of them. They also convert their inputs to half under `PreferAccuracy` only ([Input](#input)). Apart from that, `input_k.comp` is the same in both. For the other two, a full diff of the preprocessed shaders shows these differences, and no others:

| Stage | Default | Accuracy build | Kind |
|---|---|---|---|
| Output: final store | f32 result, stored directly | rounded to half first ([Output](#output)) | Narrower in the accuracy build |
| Exposure | luma scale is `1.0` | luma scale is `scaled / scaled`, which NVIDIA's `scaled * rcp(scaled)` can leave one step below 1 | Rounding change |
| Output: Catmull-Rom weights | compiler may fuse products | each product rounded separately | Rounding change |
| Output: filter axis coefficients | compiler may fuse products | each product rounded separately | Rounding change |
| Output: `split * t2` term | compiler may fuse it | rounded separately | Rounding change |
| Output: filter weights `mx`, `my` | compiler may fuse them | rounded separately | Rounding change |
| Output: sub-pixel position | tile origin folded into the expression | fused multiply-add first, then subtract the tile origin, as NVIDIA does | Rounding change |

These stages also have differences that cannot change any value, so they are not in the table. The default output stage works out tile and head addresses with shifts and masked multiplies instead of integer division and full 32-bit multiplies, and the sums reach the same addresses. Automatic exposure fetches each block's 1024 terms across 256 threads, including under `PreferAccuracy` because its measuring pass has no separate accuracy build, and each block's terms are still added in the same serial order.

### Where default K's difference comes from

Swapping whole shipped files between the two sets separates the network from the stages. On the HDR motion scene:

| Network | Stages | Result |
|---|---|---|
| Fast | Accuracy | 53.35 dB, 0 of 48 frames exact |
| Accuracy | Fast | Exact, 48 of 48 frames |
| Accuracy | Accuracy | Exact, 48 of 48 frames |

So on that scene, which has RGBA16F output, the whole gap is the network's widening, and the fast stages change nothing. The stage differences above can still matter elsewhere: on RGBA8 and RGB10A2 output, where the final store differs, or with an exposure value that makes the luma scale land one step below 1.

### Other K routes

| Route | How to select it | Result |
|---|---|---|
| Native kernels on the CUDA backend | `ExperimentalVkHip = false` | Same dB as vkhip on all four scenes |
| Vulkan network (`layer.comp`) | `Enabled = true`, `ExperimentalVkHip = false`, `Network = vulkan` | Within 0.1 dB of vkhip on all four scenes |

The Vulkan network accumulates in f32 like the fast layers. It also folds the per-channel gains into the weights: `(x * g) W` becomes `x (diag(g) W)`, with the product computed in f64 and rounded once to f16 (`engine/extract_k.py`). That is the same function in the same format, so it is a rounding change.

## Presets M and L

### What runs

Default M runs its shared layers from the table below inside Vulkan. Its input and expansion stages are Vulkan shaders that keep NVIDIA's FP8 quantization and k32 roundings, with FP8 values held exactly in f16 for the matrix instructions, and its output uses f32 stores. Default L runs the same layers there, with native input and expansion kernels in the same command list and two small Vulkan stages for the textures. Those two kernels use the shared layers' widening (`D4R_L_WIDENED`), except that the embedding's and the patch expand's results stay FP8. The CUDA backend, which the table describes, is the fallback for both.

| Kernels | Source | Arithmetic |
|---|---|---|
| Shared layers: `rrlite_enc1`, `enc2`, `enc3_tube`, `dec2`, `dec1` | native, `kernels/m` | widened: `SWIN_F32ACC`, `SWIN_NO_Q8` |
| Input and expansion: `rrlite_enc0_*`, `rrlite_dec0_*` | NVIDIA's PTX with native heads or tails (`kernels/tex`) | NVIDIA's roundings (`SWIN_EXACT`) |
| Reconstruction and downsample: `rrlite_post_*`, `rrlite_downsample_*` | NVIDIA's PTX; only surface stores are replaced | NVIDIA's arithmetic |

Preset L has one RDNA3 special case. On gfx1101 fast sets, L's unfolded enc0 and dec0 use native four-wave kernels (`D4R_L_GFX11`). They keep NVIDIA's FP8 quantization and k32 rounding boundaries and passed byte comparison against the translated kernels ([rdna3-ml-optimization.md](rdna3-ml-optimization.md)). Other RDNA3 targets run the original bodies through ZLUDA.

### Widening in the shared layers

| Step | NVIDIA's kernel | Default | Kind |
|---|---|---|---|
| Matrix accumulators | f16, rounded after every k32 step | f32 through whole GEMM chains (`SWIN_F32ACC`) | Widening |
| Intermediate activations | re-quantized to FP8 e4m3 | stay f16 (`SWIN_NO_Q8`) | Widening |
| Reciprocal and reciprocal square root | `rcp.approx`, `rsqrt.approx` | a division in the source instead of the hardware reciprocal, then rounded to f16 | Rounding change |
| Stored layer outputs | FP8 | FP8 | None |
| Patch-merge clamp to ±2π | present | present | None |

`SWIN_NO_Q8` deserves one caveat. FP8 conversion also saturates at ±448, so an intermediate value beyond that keeps its size in the default build where NVIDIA's would clamp. That follows from the wider format having more range, so this page counts it as widening, but it is not purely a rounding difference.

### Widening in L's vkhip stages

`kernels/tex/enc0l_net.hip` and `dec0l_net.hip` differ from the CUDA backend's L kernels in these steps:

| Step | CUDA backend, as NVIDIA's kernel | vkhip | Kind |
|---|---|---|---|
| Matrix accumulators | f16, rounded after every k32 step | f32 through whole GEMM chains | Widening |
| Attention, norm and MLP activations | re-quantized to FP8 e4m3 | stay f16 | Widening |
| Key order of `P V`, K grouping of `Q Qᵀ` | ZLUDA's lowering of NVIDIA's instructions | the layout's own order | Rounding change |
| Matrix operands | scaled by 2^-8, the accumulator by 2^-16 around each k32 step | true values | Rounding change |
| Embedding and patch expand results, stored outputs, patch merge clamp | FP8, clamp to ±2π | the same | None |

Going one step further, keeping the embedding's and patch expand's results in f16 too, moved the pan scene 2.4 dB further from the reference for 0.017 ms, so they stay FP8.

### Half-resolution history

NGX's half-resolution history is a CUDA format 80 surface, `UNORM_INT101010_2`, and its kernels store 10-bit codes in it. M and L on vkhip use an RGB10A2 image, which is that format. The CUDA backend holds the same codes in an RGBA32F array, because HIP cannot create the packed one. RGB10A2 holds every code exactly, and the sample is the same bilinear filter. The hardware's result for UNORM texels differs from the RGBA32F array's by up to three f32 ulps, so this is a rounding change. Through FP8 it moves the frames (45.8 dB and 50.7 dB between the two on M's motion and pan scenes) and does not move their distance from the reference by more than the usual variation.

### FP8 on RDNA3

RDNA3 has no FP8 matrix instructions, and that does not cost any precision. Every e4m3 value is exactly representable in f16, so widening FP8 weights and activations to f16 for `v_wmma_f32_16x16x16_f16` loses nothing. Scaling those operands by a power of two, as the gfx1101 L path does, is also exact.

### Why M and L score lower

M and L measure 36–50 dB against NVIDIA's kernels, while K measures 50–60 dB, even though neither is approximated. The reason is FP8. A tiny difference can flip an FP8 value by a whole step, which is a 6–12% change, and those flips compound through ten layers and across frames. Any two implementations that round differently land in this range.

`PreferAccuracy` shows this clearly. It restores NVIDIA's roundings, yet produced 0 exact frames for M and L, and it is sometimes further from NVIDIA's kernels than the default. The native accuracy kernels keep NVIDIA's roundings, but not the exact order of sums or the hardware reciprocals of ZLUDA's translated code, and with FP8 flips that is enough to diverge. For M and L on RDNA3, it costs much more GPU time without a reliable gain.

## Instructions that look like approximations

These were all checked in the shipped code objects. None changes the conclusions above.

| Item | What it is | Verdict |
|---|---|---|
| `v_rcp_f32`, `v_rsq_f32`, `v_exp_f32`, `v_log_f32`, `v_sqrt_f32` | Hardware instructions with limited accuracy | Not d4r's choice. NVIDIA's PTX uses `.approx` forms of these operations, and ZLUDA lowers them to AMD's equivalents. |
| `v_rcp_f16` in the M and L softmax | The compiler turns `f16(1.0f / (float)s)` into it, in both sets | Exact. All 31,743 positive finite half values gave the correctly rounded result on the RX 7700 XT, the same as the f32 route. |
| `v_cvt_pk_rtz_f16_f32` | Round-toward-zero conversion | Only in surface stores (`kernels/tex/tex_common.h`), where it copies the rounding of AMD's image store. It is in the accuracy set too, and it never touches network math. |
| Wave64 builds | K output and M/L post kernels | Bit-identical to wave32. |
| `IgnoreDenormals` (default on) | Drops flush-to-zero mode switches in ZLUDA-compiled kernels | Only affects values below about 1e-38. `PreferAccuracy` turns it off. |

No shipped gfx1101 fast kernel contains an integer matrix or integer dot-product instruction.

### Approximations that belong to DLSS

DLSS has approximations built into its own design. They are in NVIDIA's kernels, so d4r reproduces them rather than adding them:

- **GELU** is a clamped quadratic fit: `x * (0.5 + c * (0.41216981 - 0.08108133 * |c|))` with `c = clamp(x, -2, 2)`.
- **Softmax numerators** in M and L are a polynomial plus an integer bit trick, and K's use a polynomial inside `exp2`.
- **Reciprocals, logarithms and exponentials** use PTX `.approx` instructions throughout K, M and L.

## Input and output formats

Half precision can't hold every value a game passes, so the default routes keep f32 wherever half would change a value. Half remains under `PreferAccuracy`, which stays byte-identical to the CUDA reference, and on the host route ([Limits](#limits)).

### Input

NVIDIA's kernels read textures with `tex.*.f32` and use the values directly in f32 arithmetic. Default K does the same: its Vulkan stages sample the game's texture directly, and textures they can't sample go through an RGBA32F or RG32F copy.

On the CUDA backend, colour and motion live in planes that NGX samples. Each plane holds half or f32, depending on the game's format:

| Input format | Plane on the default route | Under `PreferAccuracy`: largest error from half |
|---|---|---|
| Colour RGBA16F | RGBA16F | None |
| Colour R11G11B10F, RGB9E5 | RGBA16F, which holds them exactly | None |
| Colour RGBA8, BGRA8 | RGBA32F | 0.12 of one 8-bit code |
| Colour RGB10A2 | RGBA32F | 0.50 of one 10-bit code |
| Colour RGBA32F | RGBA32F | 0.098% of the value |
| Motion RG16F, RGBA16F | RG16F | None |
| Motion RG32F, RGBA32F | RG32F | 0.098% of the value (for vectors in pixels: under 0.06 px at 100 px, under 0.5 px at 1000 px) |
| Motion RGBA8 | RG32F | 0.12 of one 8-bit code |
| Depth R32F, D32F | R32F | None |
| Exposure | R32F | None |

Depth stays f32, and D24, D16 and R16 depth widen into it.

An f32 plane holding the same values as a half plane gives byte-identical frames, so the kernels read both alike. M, L and K on the CUDA backend all passed this with RGBA32F colour and RG32F motion. BGRA8 colour, which takes a blit into the plane, also matched RGBA8 colour, which takes a compute pass, on M and L.

### What the K input conversion costs

The conversion saves no time. On K's Vulkan route it is extra shader code, so default K uses a pipeline without it. With RGBA8 colour on the LDR route (two A/B/B/A sets per output size):

| Output | Without conversion | With conversion | Cost |
|---|---|---|---|
| 1080p | 1.070 ms | 1.090 ms | 0.020 ms (1.9%) |
| 1440p | 1.799 ms | 1.808 ms | 0.009 ms (0.5%) |

Games whose colour and motion are already exact in half run the same either way. Ghost of Tsushima passes RGBA8 colour, while Cyberpunk 2077 (RGBA16F) and The Outlast Trials (R11G11B10F colour, RG16F motion) do not need the conversion.

With and without the conversion, K's output differs by 59.5 dB on the motion scene and 61.1 dB on the jitter scene, with 7–10% of pixels changed. That is the size of its effect on K for RGBA8 colour. Reading the stored value matches how NVIDIA's kernels read the input, but proving that it is closer needs RTX output.

### Output

NVIDIA's output kernels use a formatted f32 store (`sust.p`) straight into the game's texture. The default routes get the same result in the ways below:

| Route | How the result reaches the game's texture |
|---|---|
| K, M and L, texture with unordered access, RGBA16F, RGBA32F, RGBA8, RGB10A2 or R11G11B10F | Stored directly (`engine/output_store.glsl`) |
| K, M and L, any other accepted case: BGRA8, RGB9E5, no unordered access, or an output that is also an input | Stored into an image with the texture's bit layout, then copied raw |
| The CUDA backend, RGBA16F | NGX's kernels store half |
| The CUDA backend, every other accepted format | NGX's kernels store f32 into an RGBA32F array. A Vulkan pass stores that into an image with the texture's bit layout, then copies it raw (`kernels/support/vulkan_write_f32.comp`). RGBA32F is copied as it is. Vulkan games store straight into their texture |

A blit can't be used for this. On the RX 7700 XT, a blit from RGBA32F into BGRA8 gave the same bytes as a blit from RGBA16F, so it narrows to half whatever the source format. The raw copies avoid it.

An RGBA16F store is half in any case, with hardware rounding toward zero. Rounding an RGBA32F result toward zero reproduces it exactly, on every route. The other stores match the format's own rounding of the RGBA32F result, with no code different on any frame (16 frames per route, 32 for K on vkhip):

| Output format | Rounding checked against the f32 result | Result | The half path, on the same check |
|---|---|---|---|
| RGBA8, BGRA8 | to nearest | Exact | 1 code off on 2.0% (M), 2.1% (L), 5.4% (K) of pixels |
| RGB10A2 | to nearest | Exact | Not checked |
| R11G11B10F | toward zero, which is what the hardware store does | Exact | Not checked |
| RGB9E5 | to nearest, as the Vulkan specification encodes it | Exact | Not checked |

### What f32 costs

**K costs nothing.** Removing the half step made no measurable difference on direct stores, and the raw copy is faster than the half store and blit it replaces:

| K output | Time with half, 1080p / 1440p | Half minus f32, 1080p / 1440p | Image change |
|---|---|---|---|
| RGBA8, direct | 1.068 / 1.794 ms | +0.0025 / +0.0012 ms | 5.53% of pixels, at most 1 code, 65.4 dB |
| RGB10A2, direct | 1.109 / 1.879 ms | +0.0003 / -0.0012 ms | 19.58% of pixels, at most 1 code, 71.7 dB |
| R11G11B10F, direct | 1.117 / 1.895 ms | +0.0048 / +0.0063 ms | None on this scene (byte-identical) |
| BGRA8, copied | 1.166 / 2.020 ms | +0.028 / +0.092 ms | Not measured |

The direct rows use three A/B/B/A sets and the copied row two. A control comparing identical shaders moved by up to 0.0055 ms, so the small differences are noise.

**M and L on vkhip cost nothing either**, for the same reason as K. They are faster than the CUDA backend on every format tested, as [experimental-vkhip.md](experimental-vkhip.md) shows.

**The CUDA backend does pay for f32**, but only when the game's formats need it. RGBA16F colour and output are unchanged, byte for byte. Otherwise, total GPU time per frame with half, then f32 (one A/B/B/A set, 1.5x upscaling):

| Model, output | RGBA8 colour and output | RGB10A2 output | R11G11B10F output |
|---|---|---|---|
| M, 1080p | 4.60 → 4.95 ms (+0.35) | 4.64 → 4.93 ms (+0.29) | 4.65 → 4.91 ms (+0.26) |
| M, 1440p | 8.26 → 8.93 ms (+0.67) | 8.31 → 8.83 ms (+0.52) | 8.33 → 8.84 ms (+0.51) |
| L, 1080p | 8.07 → 8.30 ms (+0.24) | 8.21 → 8.39 ms (+0.18) | 8.19 → 8.40 ms (+0.22) |
| L, 1440p | 14.03 → 14.49 ms (+0.46) | 14.61 → 14.93 ms (+0.32) | 14.62 → 14.93 ms (+0.31) |

On these formats the result goes through the array, because the direct store into the shared buffer writes half.

For M and L, RGBA8 colour in f32 moves the image more than the numbers above suggest: 43.5 dB and 39.8 dB between the two versions, because of the FP8 flips described in [Why M and L score lower](#why-m-and-l-score-lower). Their distance from the reference barely changes.

## Differences that are not approximations

These are real differences from NVIDIA's behaviour, but none of them reduces precision.

### Window schedule after a reset

K shifts its attention windows on an 8-frame cycle, and NVIDIA restarts that cycle at every reset. The accuracy set does too. The default fast set and the Vulkan network keep counting frames instead, so after a reset they run a different phase of the same cycle.

With a reset forced on every frame, `PreferAccuracy` stays exact (16 of 16 frames) while default K measures 44–49 dB, repeating every 8 frames. Putting the fast enc0 layer, which has no restart, into the accuracy set gives 56.6 dB in phase and about 45 dB out of phase. Both phases are valid cycle positions, so this is a difference from NVIDIA's schedule rather than lost precision. The offset lasts until a later reset happens to land in phase.

### Rounding ties in K's Vulkan stages

`PreferAccuracy` through vkhip is not always exact. On the panning scene it measured 75.6 dB, while the same accuracy layers on the CUDA backend were exact on all 32 frames, so the cause is in the Vulkan stages.

Nine of eleven pan tests stayed exact over 8 frames. In one that did not (pan `0,1`), frames 1–5 were identical, then frame 6 differed in a single pixel by a single half-float step (`0x3550` against `0x354f`). K's history then spread that to 938 pixels on frame 7 and 2,014 on frame 8, with larger errors at scattered pixels, because K's reconstruction makes discrete decisions. A different formula could not stay bit-exact for five frames of moving content, so this is a rounding tie. The other failing pan (`2,1`) first shows at frame 3 with 4,055 pixels; it has the same signature, but its first difference was in internal state and was not observed directly. Which operation causes the tie is still open.

### Ignored game inputs

The D3D12 path does not forward `TransparencyMask` or `DLSS.Input.Bias.Current.Color.Mask`: the shim sets both to null on the CUDA backend, and the engine has no input for them. A game that supplies them gets DLSS without those hints. The Vulkan frontend refuses such frames instead. This is from reading the code; the test harness does not pass masks.

`Sharpness` and the exposure scale in LDR mode were tested too. NVIDIA's kernels ignore both on these presets (24 of 24 frames identical with and without them), so not passing them on changes nothing.

### A translation detail in the accuracy set

The accuracy K layers reproduce one behaviour of ZLUDA's translated code: in the deep layers, `ds_bpermute` reads zero from lanes outside a store's execution mask, so some values at the edge of the token grid are zeroed (`ptx_mask_store_row`). The accuracy set matches this to stay byte-identical to the translated reference. It is a property of the translation, so it says nothing about RTX output.

## Settings

| Setting | Default | Effect on numerics |
|---|---|---|
| `[Kernels] PreferAccuracy` | `false` | `true` selects the accuracy layers and stages for K, and accuracy native kernels for M and L. Colour and motion formats half can't hold are converted to half, and output passes through half, on every model. It also disables `IgnoreDenormals`, wave64 translation and NGX sync elision. |
| `[Kernels] NativeKernels` | `on` | `off` runs NVIDIA's translated PTX for everything on the CUDA backend. This is the reference used below, and it is far slower. |
| `[Kernels] NativeSwinEncoders` | `true` | `false` runs the original enc1 and enc2 for L and M. |
| `[Engine] ExperimentalVkHip` | `true` | `false` moves K, M and L to the native kernels on the CUDA backend. |

`PreferAccuracy` does not forward the ignored masks.

### Real approximations you can opt into

Neither of these is on by default:

| Option | What it does |
|---|---|
| `D4R_ENGINE_M_NETWORK=dot2` | Runs M and L on the RDNA2 stages, which use int8 (below) |
| `D4R_ZLUDA_FAST_MATH=1` | Lets LLVM use approximate functions and reciprocal multiplies in translated kernels (`afn`, `arcp`) |

The K sources also have build-time switches that their comments call approximate (`D4R_FAST_MATH`, `PWIN_F16ACC_NATIVE`, `PWIN_NATIVE_F16ACC`, `PWIN_NATIVE_HRECIP`). Shipped builds define none of them.

The `dot2` stages exist for RDNA2 and are the clearest example of what a real approximation looks like. Their matrices are re-encoded from FP8 to int8 with one scale per column, and activations use one scale per token. Repeating that encoding on the real weights changes 98% of them (median error 0.8%; 1% of weights are off by 26% in M and 34% in L; 0.17% and 0.60% of nonzero weights become exactly zero). `install.sh --exact` does not affect these stages.

## Measurements

**Reference:** NVIDIA's kernels through ZLUDA, with `NativeKernels = off`, `Enabled = false` and `PreferAccuracy = true`. No native kernels loaded, and every repeat run was byte-identical to the first.

**Scenes** (`tools/d3d12_dlss_harness.cpp`, 1920×1080 output):

| Scene | Input | Create flags | Content | Frames |
|---|---|---|---|---|
| Motion | 1280×720 | `0x4b` | HDR, moving object | 48 |
| Jitter | 960×540 | `0x4b` | HDR, jittered | 48 |
| Pan | 1280×720 | `0x4b` | HDR, fine detail panning 2,1 pixels per frame | 32 |
| LDR | 1280×720 | `0x6a` | LDR, RGBA8, moving object | 32 |

**Metric:** RGB PSNR with peak 1, unclipped, pooling squared error over every frame after the first.

| Route | Motion | Jitter | Pan | LDR |
|---|---|---|---|---|
| K default (vkhip) | 53.35 | 59.71 | 49.73 | 52.14 |
| K `PreferAccuracy` (vkhip) | exact | exact | 75.62 | exact |
| K default (CUDA backend) | 53.35 | 59.71 | 49.73 | 52.14 |
| K `PreferAccuracy` (CUDA backend) | exact | exact | exact | exact |
| K Vulkan network | 53.31 | 59.70 | 49.80 | 52.09 |
| M default (vkhip) | 37.60 | 48.40 | 43.11 | 38.53 |
| M on the CUDA backend | 38.03 | 48.25 | 43.14 | 38.74 |
| M `PreferAccuracy` | 37.46 | 48.57 | 42.00 | 38.01 |
| M `dot2` int8 (opt-in) | 36.63 | 47.96 | 42.03 | 37.88 |
| L default (vkhip) | 36.80 | 50.06 | 44.73 | 36.39 |
| L on the CUDA backend | 37.03 | 49.91 | 45.27 | 36.28 |
| L `PreferAccuracy` | 36.99 | 49.69 | 45.14 | 36.06 |
| L `dot2` int8 (opt-in) | 35.15 | 48.39 | 40.84 | 33.59 |

No frame in any run had non-finite values. The LDR scene uses RGBA8 colour and RGBA8 output. The reference passes both through half and the default routes do not, so their figures there include that difference. Passing them through half too gives 52.28 dB for K through vkhip, 52.24 dB through the Vulkan network and 52.45 dB on the CUDA backend. The reference has the same step, so a lower figure does not show a worse image.

### Cost of `PreferAccuracy`

Median GPU time per frame from the shim's profile log, in ms, averaged over two alternating runs of each setting (240 frames each, first 60 discarded). The inputs are 1280×720 for 1080p and 1706×960 for 1440p.

| Model | 1080p default | 1080p accuracy | 1440p default | 1440p accuracy |
|---|---|---|---|---|
| K | 1.15 | 1.28 | 1.94 | 2.16 |
| M | 4.39 | 5.61 | 8.00 | 9.73 |
| L | 7.52 | 13.75 | 14.10 | 23.81 |

For K, `PreferAccuracy` costs about 11% more GPU time and gives byte identity with the translated kernels on the scenes that stayed exact. It costs about 25% for M and 70% for L, and was not reliably closer. The accuracy set does not have L's gfx1101 four-wave enc0 and dec0.

### Reproducing

Use a complete install with engine models, and a separate copy of the `d4r` folder for each configuration so that each has its own `d4r.ini`. The harness ships in `d4r/tools/models/bin/`; put it beside d4r's `d3d12.dll` and `d3d12core.dll`. For a reference run:

```ini
[Kernels]
PreferAccuracy = true
NativeKernels = off
[Engine]
Enabled = false
```

```sh
PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 \
WINEDLLOVERRIDES='d3d12=n,b;d3d12core=n,b;d4r12core=n,b' \
D4R_DLSS_PRESET=11 D4R_HARNESS_CREATE_FLAGS=0x4b D4R_HARNESS_MOTION_SCENE=1 \
D4R_HARNESS_SAVE_FRAMES=1 D4R_HARNESS_FRAME_WAIT_MS=0 D4R_SHIM_GPU_COMPLETION=0 \
proton run d3d12_dlss_harness.exe 'Z:\path\to\d4r\nvngx.dll' 'Z:\path\to\out.raw' 48 1280 720 1920 1080
```

Set the usual `STEAM_COMPAT_*` variables for `proton run`. Set `D4R_DLSS_PRESET` to 11 for K, 13 for M or 12 for L. Each frame is saved as `out.raw.frameN`. Compare runs byte for byte first, then by PSNR. `D4R_HARNESS_JITTER_SCENE=1`, `D4R_HARNESS_QUALITY_SCENE=<report file>` with `D4R_HARNESS_QUALITY_PAN=2,1`, `D4R_HARNESS_RGBA8=1` and `D4R_HARNESS_FORCE_RESET=1` select the other cases. `D4R_HARNESS_OUTPUT_DXGI=<format number>`, `D4R_HARNESS_COLOR_F32=1`, `D4R_HARNESS_COLOR_BGRA8=1`, `D4R_HARNESS_COLOR_RGB10A2=1` and `D4R_HARNESS_MV_F32=1` select the other formats. `D4R_SHIM_F32_INPUTS=0` and `D4R_SHIM_F32_OUTPUT=0` restore half on the CUDA backend for a comparison.

## Limits

- **The reference is not an RTX card.** It is NVIDIA's kernels on AMD hardware. Matrix accumulation order, the `.approx` instructions and texture filtering all differ between vendors, so even a byte-exact result here is not proven identical to RTX output.
- **The reference shares d4r's old front end.** It passes inputs and output through half and ignores the same masks, so it can't show how far those steps are from NVIDIA's arithmetic, and the default routes are measured against a reference that keeps a step they don't have. Those sections come from the code, NVIDIA's PTX and the separate format tests above.
- **The host route still uses half.** A feature leaves shared VRAM when interop is unavailable or the colour format is sRGB, and then the host path converts colour and motion to half and takes half output. The split-frame requirement of the CUDA backend's f32 output also applies: without it, output passes through half.
- **Older model packages keep the old paths.** K's raw copies and RGBA32F store, and M's vkhip route, need the current package. Without them, K falls back to its half store and blit, and M to the CUDA backend.
- **Vulkan games on the CUDA backend had a weaker test.** The Vulkan harness only clears flat colours, where the half path also matched. It shows that the route works with f32, not that it differs from half.
- **These are mostly harness scenes.** Real content can reach paths and rounding ties these scenes do not. M and L have also run in Ghost of Tsushima at their intended modes, and their timings and screenshots there looked right, but that is one game with no pixel comparison.
- **Only gfx1101 was tested.** The other RDNA3 targets are untested on hardware for these routes, and only gfx1101 has the native L enc0 and dec0 on the CUDA backend by default.
- **The DLSS version matters.** This covers DLSS 310.7 as d4r installs it. Native kernels only run when the PTX hash matches, and a different version falls back to translated PTX.
