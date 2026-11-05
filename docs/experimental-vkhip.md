# Experimental vkhip network

`[Engine] ExperimentalVkHip = true` is the shipped default: preset K runs its
native HIP kernels recorded inside the game's Vulkan command list. With
`Enabled = auto` on RDNA3, K uses vkhip when it loads and otherwise stays on the
native kernels (the engine's Vulkan network is slower there). With
`Enabled = true`, K falls back to the engine's Vulkan network instead. Presets
M and L take the same switch for their ten shared Swin layers (the section
below), and the Ray Reconstruction presets take `[Engine] ExperimentalRrVkHip`
(auto: A - C and D - E on, F on with AMD's Windows driver and off under RADV; all six on RDNA4 only - see
the RR sections).

```ini
[Engine]
ExperimentalVkHip = false   ; turn it off (RR with it)
ExperimentalRrVkHip = auto  ; true forces preset F too, false turns RR off
```

`D4R_ENGINE_VKHIP=0|1` overrides the INI, `D4R_ENGINE_RR_VKHIP=0|1` the RR
presets. `Enabled=false` still turns the engine off. A model without code
objects keeps its ordinary engine route and logs the reason.

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

## Presets M and L (RDNA3 and RDNA4)

`vkhip::MNetwork` runs the ten native Swin layers (`kernels/m/rrlite_*`) the same way for both presets; presets M and
L share those layers and differ only in the Vulkan input and expansion stages around them. The network runs between
the engine's input and expansion stages (`MEngine` with an external network). On RDNA4 (gfx1200/gfx1201) the
injector also handles gfx12's PC-relative addressing (`s_getpc_b64; s_sext_i32_i16` with interleaved
`s_wait_alu`) and `.rodata` constant tables.

Build the layers for the engine model with `D4R_VKHIP_LAYERS=1 kernels/build.sh m DIR` and package them with
`compile_m.py --hip DIR`. That set chains every layer to the next without a pipeline barrier: each 8x8-token
block sets a completion flag, and a block of the next layer waits only for the 2x2 (3x3 after a patch merge)
producer blocks that hold its input. Dispatches on one queue launch in order, so a waiting block never
starves its producers. It also encodes e4m3 straight from the f32 accumulators, starts the MLP from an f32
copy of its bias and widens/narrows f16 pairs with `v_fma_mix`. Its parameter blocks carry flag pointers,
so it must never be used as ZLUDA's native kernel set. `D4R_VKHIP_M_CHAIN=0` restores a barrier after
every layer.

Runner: `d4r-m MODEL ... vkhip` (`engine/replay_m.cpp`), for either preset's model. Shim: the native Vulkan frontend
takes M's or L's network on vkhip under the same switch as K, `[Engine] ExperimentalVkHip` / `D4R_ENGINE_VKHIP` (on by
default), for a fixed render size: a model without `hipnet.bin` and `hip/` (every released M and L model) logs that
vkhip is unavailable and runs the engine's Vulkan network. Other sizes and D3D12 games keep the engine's Vulkan
network. `D4R_VKHIP_STAMPS=1` logs per-layer GPU time, which is only meaningful with `D4R_VKHIP_M_CHAIN=0` because
chained layers overlap.

Measured on an RX 9070 XT (gfx1201, Mesa RADV 26.2.5), d4r-m synthetic cases, alternating runs, median GPU time:

| Case | Vulkan network | vkhip network | Vulkan frame | vkhip frame |
|---|---:|---:|---:|---:|
| 854x480 -> 2560x1440 | 425 us | 358 us (-16%) | 863 us | 792 us (-7.6%) |
| 1280x720 -> 3840x2160 | 855 us | 778 us (-9%) | 1835 us | 1768 us (-3.9%) |
| preset L, 856x480 -> 2560x1440 | 632 us | 480 us (-24%) | 1755 us | 1610 us (-8.3%) |

The preset L row is a captured Ultra Performance frame (HDR, low-resolution motion, inverted depth, auto exposure)
replayed from its own inputs: the median of five alternating trials of 25 repeats. The same case in a three-frame
temporal pattern with the captured jitter offsets gives 1827 us against 1682 us (-7.9%). PSNR against the pure
ZLUDA frames of that capture, RGB: the two usable frames 28.82 dB (Vulkan) and 28.83 dB (vkhip), 30.31 dB against
30.33 dB, so the native layers are as close to NVIDIA's PTX as the engine's own Vulkan network is.

Chaining the engine's plane-layout input stage into the first layer with the same flag protocol (each stage
workgroup counting itself into the per-block flag, the layer waiting for the two workgroups that wrote its eight
tokens) was implemented and measured: RADV does not overlap the stage's dispatch with the first layer's dispatch, so
every wait is already satisfied and the frame time is unchanged (1676.6 us against 1670.8 us over five interleaved
trials, within the +-100 us trial noise). It is not shipped; the barrier between the stages stays.

Profiles captured with RADV's `MESA_VK_TRACE=rgp` and opened in Radeon GPU Profiler show both paths' dispatches:
vkhip's code objects arrive as ordinary RADV compute pipelines, so their per-pipeline durations, VGPR counts and
occupancy are visible next to the Vulkan stages' (traced frame duration 1681 us against the Vulkan path's 1765 us
for the same submit).

Quality against pure ZLUDA (NVIDIA's PTX, no native kernels) in the Proton Vulkan harness, 640x360 -> 1280x720
panning pattern, RGB PSNR: frame 8 Vulkan 43.24 dB, vkhip 43.30 dB; 32 frames Vulkan 44.41 dB, vkhip
44.18 dB.

## Preset M on native Windows (AMD's driver)

AMD's own Vulkan driver (PAL/LLPC) keeps its shaders out of the pipeline cache data (the blob is a 52-byte header),
so there the injector goes through `VK_KHR_pipeline_binary` instead (`vkhip::Device::pipelineBinary`,
`NetFiles::pipelineBinary`). The donor's captured binary is an AMDPAL ELF: `.text` with `_amdgpu_cs_main`, and
msgpack metadata in an `NT_AMDGPU_METADATA` note. The ELF is rebuilt with the kernel's code. The compute stage's
metadata gets the kernel descriptor's settings (VGPRs, float mode, WGP mode, workgroup id enables, …), and its user
data map is rewritten so that the push constant lands in s[0:1]. The driver's donor map is `[GlobalTable, push0,
push1]`. The patched binary is loaded with `vkCreatePipelineBinariesKHR` under a key of its own. The driver compiles
the donor for the GPU it runs on, so the donor ELF's machine authenticates the code objects. There is no sysfs here,
and none is needed. Kernels that use scratch are refused on this path except on gfx12 (`D4R_VKHIP_SCRATCH=0` refuses
them there too): flat scratch is architected there, so the stage metadata's `.scratch_en` /
`.scratch_memory_size` (the kernel descriptor's private segment size, bytes per lane, the unit LLPC writes too) let
the driver set the kernel's scratch up as for a spilling LLPC shader.

The device needs `VK_KHR_pipeline_binary` and `VK_KHR_maintenance5` with their features. With
`D4R_ENGINE_VKHIP` (the default), the native Vulkan frontend asks the game for both through
`NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements`. It uses vkhip only if the game's device then resolves
`vkCreatePipelineBinariesKHR`. `D4R_HARNESS_ENGINE_DEVICE=1` makes the Vulkan harness enable what the shim asks
for. Native D3D12 games (no vkd3d-proton) cannot use vkhip: their command lists are not
Vulkan. They keep the D3D12 engine, or its d3dhip route (see below) where the model
ships HIP layers.

MModel now compiles its Vulkan network layers only for an engine without an external network. AMD's Windows driver
(26.10.1) crashes compiling the FP8 cooperative-matrix `layer_m` SPIR-V, so before this `d4r-m` could not create
the model there at all. The Vulkan network fallback (other render sizes) still hits that crash on this driver.

Build on Windows (Git Bash, AMD HIP SDK, Vulkan SDK); gfx12 layers need `D4R_NATIVE_FP8=1`:

```sh
D4R_ROCM_DIR=C:/ROCm/7.14 D4R_GPU_ARCH=gfx1201 D4R_NATIVE_FP8=1 D4R_VKHIP_LAYERS=1 kernels/build.sh m KDIR
python engine/compile_m.py CAPTURE MODEL --hip KDIR      # CAPTURE: build-engine-models.py --models m --keep
cmake -S engine -B BUILD && cmake --build BUILD --config Release --target d4r-m   # d4r-m with vkhip
```

Measured on an RX 9070 XT (gfx1201, Windows 11, driver 26.10.1), `d4r-m` with 8 synthetic panning frames,
1280x720 -> 2560x1440, 400 timed frames, GPU median over two runs:

| | vkhip | D3D12 engine (`d4r-m-d3d12`) |
|---|---:|---:|
| frame | 1.40-1.44 ms | 1.63-1.68 ms |
| network | 0.80-0.81 ms | 1.01-1.02 ms |

vkhip's output is byte-identical to `d4r-m ... hip` (the same code objects through AMD's HIP runtime) on all 8
frames, and 42.1-43.8 dB against the D3D12 engine. Through the shim in the Vulkan harness (640x360 -> 1280x720,
`D4R_HARNESS_PAN=2`, frame 16), it is 39.7 dB against NGX through ZLUDA, with the error spread evenly across the
image.

### Preset L on native Windows (2026-10-30)

Preset L's input and expansion stages are `layer_m.comp` KIND 3 and 4 on RDNA3/4, and LLPC crashes compiling their
FP8 cooperative-matrix SPIR-V (the same crash as the network layers). The plane-layout pair an external network
needs (`enc0_l_planes.spv`, `dec0_l_planes.spv`) is the same code with a different token layout, so on AMD's
Windows driver preset L had no front stages at all. `MModel` now takes the model's dot2 stages there when the
package has their plane variants - `enc0_l_planes_dot2.spv`, `dec0_l_planes_dot2.spv` and the merge that writes
the network's token buffer, `merge_l_dot2_planes.spv` - and only when the physical device reports
`VK_DRIVER_ID_AMD_PROPRIETARY`: RADV keeps the matrix stages and its files, byte for byte, and the internal
(Vulkan) network route is untouched.

LLPC rejects a second thing the dot2 family uses: `SPV_VALVE_mixed_float_dot_product` (capability 6912) is
unknown to it, and `vkCreateShaderModule` fails with `VK_ERROR_INITIALIZATION_FAILED` for any module that declares
it (`enc0_l_dot2.spv` is the same module as `enc0_l_planes_dot2.spv` there). The plane stages are therefore built
with `-DDOT2_SOFT=1`: the f16 pair products are two f32 FMAs of exact f16 products instead of `v_dot2_f32_f16`.
The packed int8 dot4 (`v_dot4`, `OpSDot`) and the AMD ballot swizzle compile on both drivers and stay intrinsics.
The internal dot2 stages keep `v_dot2` (RDNA2 on RADV).

RX 9070 XT, Windows 11, driver 26.10.1 (LLPC), `d4r-m models/sr-lb 1280 720 2560 1440`, 25 repeats,
`D4R_M_WARMUP_PASSES=6`, 76 timed frames: median **5446.0 us** (min 5000.9), input 2115.4, network 797.8,
expansion 1830.1, reconstruction 384.7, downsample 69.8 us. The same frame on the D3D12 engine
(`d4r-m-d3d12`, same inputs and repeats) is 2379.8 us, with its input stage at 490.0 and its expansion at 367.9
us against 2115.4 and 1830.1 here: the emulated `v_dot2` is what this route costs. `vkhip` is byte-identical to
`hip` on all four frames and 52.24-53.87 dB against the DXIL engine.

**The matrix stages on LLPC (2026-11-05).** The crash is in LLPC's handling of glslang's module, not in the FP8
cooperative matrices: the same modules through SPIRV-Tools' `spirv-opt -O` compile and run, as preset F's
`enc0_llpc.spv` already showed. `compile_m.py` now writes `<stem>_llpc.spv` for L's matrix stages (`enc0_l`,
`dec0_l`, their `_planes` variants and the five network layers `m_*.spv`) when `spirv-opt` is on PATH or in
`D4R_SPIRV_OPT`, and `MModel` takes them on AMD's driver when the package has all nine, ahead of the soft-dot2
stages; RADV loads the same files as before. glslang's module still segfaults in `vkCreateShaderModule` (checked
against a package without the dot2 plane files). `d4r-m` with the HIP L package: input 613 / expansion 473 us against
2158 / 1832 for soft dot2, vkhip byte-identical to hip, 51.75 / 51.83 dB against the D3D12 engine (soft dot2:
51.22 / 48.67). The internal Vulkan network (no vkhip) compiles too with the `m_*_llpc.spv` layers (51.6-53.0 dB
against D3D12). Through the shim (`vulkan_dlss_harness`, bench.py): 5.215 -> 2.252 ms at 1280x720 -> 2560x1440 and
11.309 -> 5.029 ms at 1920x1080 -> 3840x2160.

### Preset K's Vulkan network on AMD's driver (2026-11-05)

**Update (later the same day): K on vkhip in the native Vulkan frontend.** The frontend creates K's vkhip network
(`create_engine_vkhip_k`, shared with vkd3d-proton's adapter) and records the engine's front, the eleven native layers
and the engine's back, as for M. Four of the gfx1201 layers spill (`dltss_pwin_enc0/dec0/dec1/dec2`, 256 VGPRs,
108 / 64 / 40 / 44 bytes of scratch), which gfx12 takes through the pipeline binary (above). RX 9070 XT, 26.10.1,
1280x720 -> 2560x1440 1.649 -> 1.107 ms, 1080p -> 4K 2.94 -> 2.39 ms; output 65.7 dB against the Vulkan network, the
same pan-scene truth PSNR (29.84 dB).

At first 5-25% of the 600-frame 1080p -> 4K runs diverged slightly (67 dB from the usual output, a glitch the history
spreads over the frame), whatever the scratch size, barriers between dispatches, `-mcumode` or grid trimming, so the
route was briefly opt-in. The cause was in the kernels: `pos_core` (`kernels/k/pwin_pos.h`, enc0 and dec0) has every
wave read the `hb` rows of all 64 tokens in its attention loop and then rewrites its own rows in the residual, with no
barrier between, so a wave that finishes its heads first can overwrite rows another wave still reads. The spills
skew the waves enough to make it frequent; through HIP it showed in about 1 of 40 runs with `-mllvm
-amdgpu-set-wave-priority` and not at all in 68 runs without. `POS_HB_SYNC` adds the barrier (and the
`d4r_pos_hb_sync` marker global): 64 of 64 runs identical at 4K, the same output as the race-free runs. It is set in
`kernels/windows-flags.txt` for gfx1201, so the Windows builds have it; the Linux and gfx11 builds are unchanged. On
gfx12, vkhip refuses enc0 / dec0 without the marker (the engine's Vulkan network runs instead) unless
`D4R_VKHIP_SCRATCH=1`.

Before that, K's vkhip network did not load in the native Vulkan frontend on gfx1201: four of its eleven kernels spill
(`dltss_pwin_enc0/dec0/dec1/dec2`, 256 VGPRs, up to 132 bytes of scratch) and the pipeline-binary injection refused
scratch. K therefore ran the engine's Vulkan network, which on LLPC was 2.6x the D3D12 engine (3.61 against 1.37 ms
at 1280x720 -> 2560x1440). The cause is the weight loads: LLPC loads a row-major cooperative-matrix B operand one
16-bit element at a time (a WMMA B lane holds a column; dec5 had 1168 `buffer_load_u16` for 136 WMMAs).
`compile_k.py` now writes `<layer>_llpc.spv` with `D4R_WEIGHT_TILES_COLMAJOR` and `D4R_LLPC_LOADS` and a
`weights_llpc.bin` whose B-operand tiles (attention, MLP, patch merge / expansion, embedding and head matrices) are
transposed per 16x16 tile, and reads bias tiles (16 equal rows) with stride 0; the runtime takes them on AMD's driver
when the package has all of them. The logical operands are the same, so the output is byte-identical. dec5's 16-bit
loads drop to 54 (the position tables, read as accumulators, and a few vectors). Packages without the files, RADV and
the D3D12 engine are unchanged. `tools/vk_pipeline_stats.cpp` prints a SPIR-V compute shader's register, LDS and
scratch use and (`-ir`) its ISA on the installed driver through `VK_KHR_pipeline_executable_properties`, e.g.
`VKSTATS_WAVE=32 vk_pipeline_stats dec5_llpc.spv -ir b0=ssbo b1=ssbo b2=ssbo b3=ssbo b4=ssbo b5=ssbo`.

## RDNA2 on native Windows (unofficial, 2026-11-02)

Untested on RDNA2 hardware. Preset K takes vkhip on gfx1030-gfx1036 through AMD's driver only
(`VK_KHR_pipeline_binary`); RADV keeps refusing gfx10 code objects. `kernels/build.sh k` builds the RDNA2 layers
(software matrix products, `kernels/common/wmma_layout.h`), and `build_engine_models.py` packages a single gfx103x
folder as for gfx110x. The injector handles gfx10.3's own `s_getpc_b64`, `s_mov_b32` and `s_branch` encodings and
its kernel ABI: without architected flat scratch LLVM always enables the private segment buffer, so the kernarg
pointer arrives in s[4:5] (user SGPRs 6) and the donor's push constant is mapped there. The kernels use no scratch,
so s[0:3] is never read. Offline checks only: all eleven gfx1030 K layers (24 kernels) parse and their patched
data sites disassemble as gfx1030 `s_mov_b32` pairs.

The Vulkan network fallback needs the LLPC builds: `compile_k.py` writes `<layer>_dot2_llpc.spv` and `compile_m.py`
the M/L `*_dot2_llpc.spv` (`D4R_PORTABLE_DOT2` / `DOT2_SOFT`: no capability 6912), which the runtime loads on
`VK_DRIVER_ID_AMD_PROPRIETARY`. The K model compiles its network pipelines before vkhip is tried, so without them
K cannot load on RDNA2 under AMD's driver at all. M and L have no vkhip route on RDNA2: their Swin layers need WMMA.

## d3dhip: the HIP network inside the D3D12 engine (2026-10-30)

`d4r-m-d3d12` and `d4r-rr-d3d12` accept a trailing `hip` argument, and default to it
when the model directory contains the HIP layers: the D3D12 engine records only its
front (token/planestage) and back (expansion, downsample, output), the network itself
runs as the model's hsaco code objects on a HIP stream. Shared token/head buffers
travel as NT handles: `CreateSharedHandle` on the D3D12 side,
`hipImportExternalMemory` (`OPAQUE_WIN32`) on the HIP side, both kept in `COMMON`
state. The loader proves the two APIs see one GPU first: the HIP device must be AMD
(SetupAPI PCI identity) and the DXGI adapter the D3D12 device names by LUID must be
that card — by PCI vendor/device/subsystem, or, when the adapter's description is
spoofed (OptiScaler reports an NVIDIA card to the game), only when the machine has a
single AMD display adapter. HIP on ROCm 7.14 does not report the adapter LUID
(`hipDeviceAttributeLuid` fails), so the LUID comes from DXGI's enumeration of the
device's adapter.
Queue handoff is CPU-serialized — submit the front, wait on its fence, run the HIP
graph, `hipStreamSynchronize`, record the back; shared fences are not used. Any step
that fails logs `hip unavailable ...; using the DXIL network` and runs the ordinary
DXIL network, whose output is byte-identical to a build without this route.
`D4R_DX_NET=dxil` forces DXIL. The safety contract and the measured round trip
(~1.4 ms/frame CPU serialization at 2560x1440) are in `docs/backend-draft.md`.

### In the game: Ghost of Tsushima, preset M (2026-10-30)

The NGX shim runs this route for preset M when the model package has `hipnet.bin`,
`hip/` and the plane-layout stages (`compile_m.py --hip KERNEL_DIR --dxil yes`, also
with `--ldr` for games that pass the LDR flag), and the package has them now for both
HDR (`engine/m`) and LDR (`engine/m-ldr`) — the LDR color stages (`enc0_m`,
`enc0_m_planes`, `post_m*`, `down_m`) differ and must come from the `--ldr` build.
`D4R_DX_NET=dxil` forces the DXIL network.

The game's command list is split at the network (`d4r_d3d12_split.h`): the shim records
the front into the game's list, `begin()` redirects the rest into d4r's second list,
the back is recorded there, and `ExecuteCommandLists` runs the first part, calls the
split's `beforeSecond` hook on the submitting thread (wait for the front's fence on the
CPU, `HipNetworkDx::run()`, which blocks until the graph has finished), then signals the
split fence and runs the second part. The hook is the documented CPU round trip; with a
hook the split always satisfies its own fence, so a failing network cannot hang the
queue. The engine is created at the first frame's render size and is recreated when the
size changes (the HIP layers are built for one token grid): Ghost renders a fixed
1920x1080 at Performance, so the route is stable there; a dynamic-resolution game
would restart the engine at every size.

`amdhip64` is **not** imported by `d4r_nvngx.dll`: linking it made Ghost's device hang
(DXGI_ERROR_DEVICE_HUNG at `CreatePlacedResource`) before its first frame, next to
OptiScaler's driver hooks. The network lives in `d4r_hip_net.dll`
(`engine/d3d12/hip_net_bridge.cpp`, exports `d4rHipNet*`) beside `nvngx.dll`, loaded on
the first preset M frame that uses it.

Verified on RX 9070 XT (gfx1201), driver 32.0.32015.7029, Ghost of Tsushima
DIRECTOR'S CUT 1053.9.0623.1807 through OptiScaler 10.0.0-pre1, 1920x1080 -> 3840x2160
preset M, LDR flag: `d4r_hip_net.dll loaded`, `feature ... with its network on the model's
HIP layers (d3dhip)`, the menu and gameplay (including movement) render at 55-68 FPS with
3.7-5.8 ms upscaler time in the game's overlay and no device removal. That run used an LDR
package whose color stages were still the HDR ones (marker only); the package was then
rebuilt with `--ldr` and ran (feature created, frames recorded, no errors), without a
second visual inspection. The HDR package is built but Ghost passed the LDR flag in this
setup, so HDR gameplay is unverified.

## Ray Reconstruction vkhip on Windows (2026-10-30)

`ExperimentalRrVkHip = auto` (default) turns the RR vkhip route on for presets A - C
(CNN) and D - E (native Swin RR, ten stages: `hip/rr_<stage>.hsaco` +
`weights-<stage>m.bin`, stages enc0 - enc5 and dec4 - dec0). F (the F model's rr2
objects) is on by default with AMD's Windows driver, where its Vulkan path is LLPC's
compile and vkhip measures 17% faster with byte-identical output (RX 9070 XT,
1280x720 -> 2560x1440, frames back to back: 3.90 against 4.73 ms, 2026-11-04); under
RADV its vkhip route is within ~2% of (or slower than) its Vulkan path and F stays off
unless `true` forces it. `D4R_ENGINE_RR_VKHIP=0|1` overrides; turning
`ExperimentalVkHip` off turns RR vkhip off with it. On AMD's Windows driver the RR
route uses the same `VK_KHR_pipeline_binary` injection as M (replay_rr now enables
that plus `VK_KHR_maintenance5` on non-RADV drivers and hands the binaries to the
network — without it the driver rejects the pipeline with "cache data layout"). All
RR kernels are RDNA4-only (gfx12 FP8 WMMA), so the adapter gate stays gfx1200/1201.
Preset A - C print `vkhip: 11 Ray Reconstruction stages`, D - E `vkhip: 10 ...`.
On AMD's Windows driver (LLPC 32.0.32015) preset F crashed the process before the first frame: a null
dereference inside `vkCreateShaderModule` on exactly the two modules that carry translated PTX around a
native block (`enc0.spv`, `dec0_1.spv`) - every other stage of the model compiles, so it is neither the
FP8 cooperative matrices nor a size limit. `compile_rr2.py` now also emits `enc0_llpc.spv` and
`dec0_1_llpc.spv` (the same modules through `spirv-opt -O`), and `runtime_rr2.inc` loads those two when
the device reports `VK_DRIVER_ID_AMD_PROPRIETARY` and the files exist; RADV loads the model's own shaders
byte for byte. With that the F Vulkan route runs (1440p median 4.759 ms) and the vkhip route is its
faster variant (3.915 ms, byte-identical output); the 14 network stages with HIP kernels are built
lazily on first dispatch on that driver (vkhip compiles 10 shaders at load instead of 24). The F route
stays opt-in: `rr2_*.hsaco` through the HIP runtime still faults (`D4R_RR_HIP_F` keeps d3dhip on the
DXIL network), and the vkhip route is within ~2% of the D3D12 engine rather than ahead of it.

## Preset F on RDNA4 (2026-10-28)

An experimental backend now records fourteen HIP kernels on the Vulkan command
list: enc1–enc5 and both blocks of dec4–dec1 plus dec0's first block (all of them
translated from the Vulkan path's own shaders since, see the last section).
Encoder 0, decoder 0's pixel tail, exposure and output filtering stay on Vulkan.
There is no HIP runtime or second queue. This backend **does not meet the 10%
performance improvement target** and remains opt-in.

Build a default relaxed=3 preset F model with `compile_rr2.py`, keeping its
`native-work` directory, then:

```sh
kernels/rr/build_vkhip_f.sh MODEL
engine/build.sh /tmp/d4r-vkhip-build
mkdir -p OUTPUTS
D4R_ENGINE_VKHIP=1 D4R_RR_OUTPUT_ONLY=1 /tmp/d4r-vkhip-build/d4r-rr MODEL INPUTS OUTPUTS
```

`D4R_GPU_ARCH=gfx1200|gfx1201` selects the target (gfx1201 default). gfx1201 was
measured; gfx1200 is untested. The build refuses non-F and non-relaxed=3 models.
`D4R_VKHIP_RR_STAGES=enc1,dec0_0` restricts replacement to named blocks; missing
blocks use Vulkan. `D4R_ENGINE_VKHIP=0` selects the Vulkan control.

On Windows (AMD's driver) `build_vkhip_f.sh` also builds encoder 0, `rr2_enc0.hsaco` (2026-11-05). This is the
hybrid stage that runs after the gather pass `enc0g`. It reads its windows from the gather's records, not from its
textures, so `kernels/rr/rr2_hybrid_prep.py` drops those bindings, and `glsl2hip.py --include=glsl_hip_hybrid.h` translates
the rest. It is spill-free at 226 VGPRs. Output stays byte-identical, and vk-RRF went 3.521 → 3.412 ms at
1280x720 → 2560x1440 and 7.731 → 7.520 ms at 4k. Linux (RADV) keeps the model's enc0.

dec0's pixel tail, `rr2_dec0_1.hsaco`, is built the same way (spill-free at 211 VGPRs, `RR2_WAVES=6`). It stores
its four surfaces (176 / 184 / 192 / 200) to storage images, which a HIP kernel can only reach through their
descriptors. vkhip gets them there without VK_EXT_descriptor_buffer:

- The pipeline is built from a second donor, `engine/vkhip/donor_images.comp`, which declares set 0 as four storage
  images. Its layout has that set plus the 8-byte push constant (`createPipeline(..., images)`).
- The kernel enables the dispatch-id SGPRs (`Kernel::table`). The PAL user-data map gets the set's table pointer in
  s2, where the dispatch id's low dword would be.
- The kernel forms the table's address as LLPC does: that low dword, and the high dword of `s_getpc_b64`
  sign-extended from 16 bits (`glsl_hip.h` `glslImageTable`). The code-object parser lets this bare `s_getpc_b64`
  through (it has no add pair; a global's always does).
- The F loader writes one descriptor set of the engine's views (`RStageResources::surfaceF`) and binds it before the
  dispatch. The engine binds its own sets on every dispatch, so nothing else sees it.

The output is not byte-identical to the LLPC stage: 49 dB (pan) / 59 dB (still) against it after 120-160 harness
frames, finite, same means. vk-RRF went 3.418 → 3.359 ms at 1440p and 7.516 → 7.378 ms at 4k. RADV keeps dec0_1 on
Vulkan (no pipeline binaries).
`D4R_RR_OUTPUT_ONLY=1` saves and reads back only the final image; it does not change
frame evaluation or GPU timestamp boundaries.

For the vkd3d-proton game adapter, `D4R_ENGINE_RR_VKHIP=1` explicitly enables F
replacement, subject to `D4R_ENGINE_VKHIP` and verified RDNA4/RADV adapter checks.
Missing code or a loader failure logs a reason and keeps the Vulkan engine.
`GameDenoiser` owns the fixed-size engine, and the feature owns its stage recorder
until after the adapter is destroyed. The game hook compiles, but its live-game
validation is incomplete: the Proton harness exited before producing outputs in
this session, including its Vulkan/CUDA controls. Native Windows/D3D12 and the
native Vulkan NGX frontend's RR route are not enabled by this change.

The Vulkan RR game adapter retains at most 16 borrowed input/output view combinations. On further texture rotation,
it copies inputs and output through stable images, keeping the recorded descriptors alive. The engine reserves 16
additional combinations for the four optional guides being present or absent. This avoids exhausting the descriptor
cache and switching to CUDA during a long session; copying can add GPU work. This policy applies to D/E as well as F.

The native block in `kernels/rr/rr2_native_body.h` distributes attention heads and
MLP branches across groups of four waves. Each wave retains only its own channel
tiles, and accumulates heads and branches in their original order. Shared rows
are padded by eight bytes to reduce bank conflicts. V weight tiles are transposed
on loading so lanes read contiguous operands. Global-pointer provenance and
per-wave uniform indices avoid unnecessary flat addressing. Packed FP8 conversion
uses the GPU's saturation mode; this has the existing native kernels' binary16
overflow behavior, so these measurements establish PSNR similarity, not general
bit identity with Vulkan or NVIDIA.

`build_vkhip_f.sh` chooses 1, 2 or 4 head groups per block from a sweep; set
`D4R_RRVK_HS=1|2|4` to override it. Every shipped experimental block in the measured
set has zero private memory and zero VGPR spills in LLVM's code-object metadata.
The runtime gives the arena and live frame parameters device addresses and retains
the engine's barriers around each block. Weights and code objects remain local
model artifacts, with no NVIDIA weights embedded in the source.

gfx1201 (driver-reported RX 9070 XT), RADV Mesa 26.2.5-arch3.1, 1280x720 → 2560x1440 synthetic
capture, five alternating-order trials of 25 repeats per frame, median across the
runner's four per-frame GPU minima per trial:

| Path | Full frame GPU time |
|---|---:|
| Vulkan | 3.332 ms |
| vkhip, fourteen blocks | 3.934 ms (+18.1%) |

Quality uses separate single-evaluation replays, synchronous NVIDIA CUDA through
ZLUDA **without native kernel substitutions** as reference, fixed peak 1,
unclipped linear RGB and pooled MSE, excluding alpha:

| Case | Vulkan vs ZLUDA | vkhip vs ZLUDA |
|---|---:|---:|
| Four 1440p frames | 48.663 dB | 48.841 dB |
| 32 temporal diagnostic frames, 128x96 → 384x288 | 42.451 dB | 42.236 dB |

On individual 1440p frames the pairs are 50.741/50.744, 52.685/52.672,
47.194/47.431 and 46.673/46.917 dB. Final images have no nonfinite samples.
This is a similar distance from ZLUDA, but it is not a speed-up.

RGP ran in a dedicated Xvfb display `:98`. Paired RADV per-submit traces show the
native blocks as ordinary compute pipelines. For the reset-frame decoder-0
(161x92 windows), RGP reports 605.468 us, 128 VGPRs and occupancy 12 for vkhip;
Vulkan reports 396.412 us, 216 VGPRs and occupancy 7. These are traced first-frame
stage durations, not the warmed headline timings. Increasing occupancy alone did
not recover the cost of the native block's work; the Vulkan one-wave decoder also
reuses projections and symmetric scores. No performance improvement is claimed.

Local evidence is under `build/vkhip-f/`: `quality-final.json`,
`timing-final.json`, `sweep.json` and the associated logs. `rgp/vulkan-final.rgp`
and `rgp/vkhip-final.rgp` are paired 1440p traces; `rgp/final-pipelines.png` and
`rgp/vulkan-final-pipelines.png` show their pipeline metrics. The earlier decoder
port, all-loop unrolling, explicit phase dispatches, WGP mode and other wave
counts were also measured there; unused implementations were removed from source.

### Translated stages and the barrier ceiling (2026-10-28, later)

A second attempt compiled the Vulkan path's own stage shaders with LLVM instead of
the native block: `kernels/rr/glsl2hip.py` rewrites a stage's
`<stage>.preprocessed.comp` into HIP over `kernels/rr/glsl_hip.h` (cooperative
matrices as gfx12 FP8 WMMA, packed FP8 pairs, `--sat` for the saturation mode,
`--rolled=` for loops left rolled). `build_vkhip_f.sh` uses it for the stages where
it beats the native block (see the end of this section).
Warmed per-stage GPU times in us, 1440p case, Vulkan (ACO) vs vkhip (LLVM):

| Stage | Vulkan | vkhip | LLVM private bytes / VGPR spills |
|---|---:|---:|---|
| dec0_0 (`--sat --rolled=br`, 6 waves) | 318 | 343 | 0 / 0, byte-identical |
| enc1 | 227 | 266 | 0 / 0 |
| enc2 | 115 | 122 | 0 / 0 |
| enc5 | 22 | 139 | 24 / 5 |
| dec4_1 | 32 | 140 | 28 / 6 |
| dec4_0 | 36 | 252 | 508 / 131, byte-identical |
| enc3, enc4 | 54, 37 | 131, 160 | 136 / 33, 392 / 106; wrong output |

Every stage with a private segment loses about 100 us or more, whatever its spill
count; spill-free stages are within 5–20% of ACO. Leaving loops rolled replaces
spills with dynamically indexed private arrays, which are still scratch. The KIND 2
decoders and dec0_0 are byte-identical to Vulkan; the other translated stages differ
slightly, probably from LLVM's `v_rcp_f16` folding of `float16_t(1.0 / x)`.

Upper bound on the scheduling lever vkhip enables (per-window flag chaining instead
of the engine's barriers): removing the barriers entirely (wrong output, timing
only, five alternating trials) takes the Vulkan frame from 3.136 to 2.834 ms
(−9.6%) for enc1–dec1 and to 2.772 ms (−11.6%) for all sixteen stages. Single
barriers removed one at a time are within noise. A correct chain keeps the real
dependency waits and runs on the slower LLVM stages, so this route cannot reach a
10% gain either; it was not implemented.

`build_vkhip_f.sh` now builds enc2, dec2_0, dec2_1, dec1_0, dec1_1 and dec0_0 this
way and the other eight stages as native blocks (`D4R_RRVK_TRANSLATED` overrides the
list). The translator writes `1 / x` as `v_rcp_f32`, as RADV does; with that, every
translated stage in the list gives the native block's output, and the whole frame is
byte-identical to the all-native build on the four 1440p frames (48.841 dB against
ZLUDA; Vulkan 48.663 dB). All fourteen code objects have no private memory, which the
pipeline-binary path on AMD's Windows driver requires. Five alternating trials, same
setup:

| Path | Full frame GPU time |
|---|---:|
| Vulkan | 3.086 ms |
| vkhip, all native blocks | 3.724 ms (+18.6%, earlier trial set) |
| vkhip, six translated stages | 3.449 ms (+11.8%) |

Per stage (us, vkhip / Vulkan): enc1 263/229, enc2 125/116, enc3 72/55, enc4 62/39,
enc5 38/24, dec4_0 77/37, dec4_1 58/34, dec3_0 72/54, dec3_1 68/50, dec2_0 147/117,
dec2_1 110/103, dec1_0 222/193, dec1_1 190/177, dec0_0 344/322. dec3_0 translated
spill-free with `-mllvm -amdgpu-sched-strategy=iterative-minreg` took 275 us.

### Transposed V weights and window records (2026-10-28, latest)

Three changes brought the translated stages to Vulkan's speed, so `build_vkhip_f.sh`
now translates every stage:

- `--v-layout=rr2_stage.h` makes the loader transpose the heads' V weight tiles
  (`rr2_v_layout`, as for the native blocks): they are the only B operands read
  from the weights, and their loads become 8 contiguous bytes per lane instead of
  eight byte loads. This made the level 0–2 stages faster than ACO's.
- Levels 3–5 have about as many windows as the GPU has compute units, so one
  window's latency is the stage's time. There LLVM's schedule waited on each weight
  tile's load just before its WMMA, about ten exposed memory latencies per MLP
  branch: r (`rv[MT][8]`, 80 VGPRs, read only after the MLP) left no registers to
  issue the loads early with. `--record=rv` keeps r in the window's record in the
  arena (the words the phases use; the runtime now reserves the records of the
  levels with phases whenever the arena has a device address) through one uniform
  pointer and immediate offsets, and reads it back with an atomic load that LLVM
  does not forward the stored value to. In LDS instead, r cost the CU its second
  workgroup (enc4 49 us). dec4_0 also keeps its skip input there (`--record=rv,xr`);
  dec3_0 gets there with `-mllvm -amdgpu-sched-strategy=max-memory-clause`.
- `glsl_hip.h` takes `gl_SubgroupID` through `readfirstlane`: LLVM does not see that
  `threadIdx.x >> 5` is uniform, and kept it and the lane offsets derived from it in
  VGPRs, which made dec4_0 spill six of them (the native block it replaced: 60 us).

The script fails if a code object has a private segment. Output is byte-identical to
the all-native build on the four 1440p frames (48.841 dB against ZLUDA; Vulkan
48.663 dB), and the Vulkan path's output is unchanged. Fifteen alternating trials,
same setup, median of each trial's best frame:

| Path | Full frame GPU time |
|---|---:|
| Vulkan | 3.301 ms |
| vkhip | 3.316 ms (+0.5%) |
| vkhip, dec4_0 a native block (another trial set; Vulkan 3.308 ms) | 3.333 ms (+0.8%) |
| vkhip, previous build (eight translated stages) | 3.445 ms (+4.1%) |

Per stage (us, vkhip / Vulkan): enc1 227/226, enc2 113/115, enc3 56/55, enc4 37/36,
enc5 24/22, dec4_0 42/36, dec4_1 32/32, dec3_0 53/53, dec3_1 48/50, dec2_0 110/115,
dec2_1 99/102, dec1_0 190/193, dec1_1 172/178, dec0_0 335/319. Not effective: rr2_m's
phases as vkhip kernels (enc3 91, dec3_1 82), WGP mode for translated stages,
partial unrolling of the head and branch loops, max-ilp or memory-clause scheduling
for the level 0–2 stages, LLVM's source-order schedule (dec4_0 170 us).

## Windows backend comparison: vkhip vs d3dhip vs d3d12 (2026-10-30)

RX 9070 XT (gfx1201), driver 32.0.32015.7029, HIP SDK 7.14. Replay runners only
(`d4r-m.exe`, `d4r-rr.exe`, `d4r-m-d3d12.exe`, `d4r-rr-d3d12.exe`), never
concurrently; 25 repeats per frame, GPU median inside a run, three trials per route
with the route order rotated per trial; the table is the **best trial's median**,
because two trials of an earlier run were clock-throttled and medians of three
amplified that instead of rejecting it. Driver:
`build/backend-compare-20261030/tools/bench_windows_backends.py`, whole-frame times
ms:

| Case | vkhip | d3dhip | d3d12 (DXIL) |
|---|---:|---:|---:|
| M 1280x720 -> 2560x1440 | **1.372** | 1.717 | 1.605 |
| M 1920x1080 -> 3840x2160 | **3.076** | 3.361 | 3.355 |
| L 1280x720 -> 2560x1440 | 5.204 | 2.347 (fallback) | **2.317** |
| L 1920x1080 -> 3840x2160 | 12.331 | 5.954 (fallback) | **5.752** |
| RR D 1440p | **3.635** | 3.756 | 3.759 |
| RR E 1440p | **3.387** | 3.761 | 3.757 |
| RR D 4k | 7.950 | 7.747 | **7.717** |
| RR E 4k | 7.749 | 7.750 | **7.715** |
| RR F 1440p | **3.563** | 3.741 (fallback) | 3.758 |
| RR F 4k | **7.739** | 7.886 (fallback) | 7.948 |

Notes:

- **M**: vkhip stays the fastest route for Vulkan games. d3dhip runs the same code
  objects through HIP inside the D3D12 engine (`network: hip (native layers)`), and
  costs ~0.1–0.3 ms over the DXIL network at 1440p — the CPU round trip plus the
  plane-staging round trip roughly cancels the HIP network's advantage over DXIL.
  It is the route for native D3D12 games that ship the hip layers (`hipnet.bin`,
  `enc0_m_planes.dxil`, `dec0_m_planes.dxil`); measured in Ghost of Tsushima
  (LDR M, 1920x1080 -> 3840x2160, 55–68 FPS, no device removal, see todo).
- **L**: no plane-layout DXIL stages exist for L's unfolded layers, so d3dhip logs
  the fallback and equals d3d12. vkhip now runs on this driver through the
  driver-gated soft-dot2 plane stages (see "Preset L on native Windows"), but the
  emulated `v_dot2` makes it the slowest route here: 5.204 ms against 2.317 ms of
  DXIL at 1440p. Its output is byte-identical to `hip` and 52.2-53.9 dB against
  the DXIL engine. The internal (Vulkan network) route still fails on LLPC — the
  non-plane dot2 modules declare the Valve capability this driver rejects (it was
  broken here before, on the matrix stages).
- **RR D/E**: vkhip wins at 1440p; at 4k all three are within 3% and d3d12 is
  marginally fastest. d3dhip runs the native RR layers on the HIP stream
  (`network: hip (native layers)`), verified 54.8–58.6 dB against the DXIL network
  with byte-identical DXIL fallback output.
- **RR F**: the Vulkan and vkhip routes now run on this driver (the `enc0_llpc.spv`
  / `dec0_1_llpc.spv` variants above; the plain-Vulkan route measures 4.759 ms at
  1440p against vkhip's 3.563 ms, output byte-identical between the two, 55.27 dB
  against the DXIL engine), and vkhip is the fastest F route at both sizes. Only
  the HIP runtime route is broken: the `rr2_*.hsaco` the `rr-f` package ships
  fault on launch through AMD's Windows HIP runtime (a zeroed parameter block and
  a 1x1 grid still remove the device; the objects' `.group_segment_fixed_size`
  predates the current `rr2_native_body.h` pools), so d3dhip stays gated behind
  `D4R_RR_HIP_F=1` and otherwise logs the fallback — the `(fallback)` rows are the
  DXIL network, and the distance between a fallback row and the d3d12 row is trial
  noise. Rebuilding them with `kernels/rr/build_vkhip_f.sh` (Linux, RADV) is the
  open step; a device fault cannot fall back at runtime, which is why the gate is
  opt-in rather than auto.
- **K**: the HIP network was measured at 9.2 ms vs 2.9 ms DXIL (1440p whole frame)
  — the DXIL network stays K's default everywhere; hip only through an explicit
  argument.
- The safety contract for d3dhip (adapter identity, NT-handle import, CPU
  serialization, no shared fences) is in `docs/backend-draft.md`.

Defaults on this machine: `ExperimentalVkHip = true` (K, M, L through vkhip in
Vulkan games where the driver survives it), `ExperimentalRrVkHip = auto` (A - F on;
F off under RADV), d3dhip automatic for preset M in the shim's D3D12 engine and for M/D/E in
the replay runners when the model ships HIP layers, always with the logged DXIL
fallback and `D4R_DX_NET=dxil` to force DXIL.

