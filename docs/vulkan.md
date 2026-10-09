# Experimental native Vulkan support

The development shim now implements the Vulkan NGX Super Resolution API in
addition to D3D12. Wolfenstein: Youngblood is the intended first game test;
its gameplay, image quality, and performance have not been validated here.
Existing release archives do not acquire this support until rebuilt.

Build with `scripts/build_d4r_nvngx_shim.sh`. The result is
`build/d4r_nvngx.dll`; use it as `d4r/nvngx.dll` in a development install.
Keep NVIDIA's actual CUDA NGX core at `d4r/ngx/_nvngx.dll` and the official
DLSS feature at `d4r/nvngx_dlss.dll`. The existing Wine CUDA bridge,
ZLUDA, ROCm runtime, kernels and `d4r.ini` are still required.

## Youngblood testing

Route the game's Vulkan NGX calls through OptiScaler's native Vulkan DLSS
backend and point that backend at the new shim. Apply these values to an
existing OptiScaler installation, retaining its other settings:

```ini
[Upscalers]
VulkanUpscaler=dlss
[DLSS]
Enabled=true
[Libraries]
NvngxPath=d4r\nvngx.dll
[Inputs]
EnableDlssInputs=true
[Hooks]
HookOriginalNvngxOnly=true
[Spoofing]
Vulkan=true
VulkanExtensionSpoofing=false
[FrameGen]
Enabled=false
FGInput=nofg
FGOutput=nofg
```

Keep `HookOriginalNvngxOnly=true`. Setting it to false causes OptiScaler to
intercept the shim's own `_nvngx.dll` load and return the proxy instead of the
CUDA NGX core. Apply `packaging/optiscaler-vulkan.settings` with
`scripts/configure_optiscaler.py` to an existing INI to preserve the shared
routing requirements and the game's separate plugin/spoofing choices.

OptiScaler must actually be loaded into the game. The D3D12 release's
`dxgi.dll` proxy and its DXGI spoofing alone are not proof of Vulkan hook
activation or DLSS eligibility. Use the proxy/override supported by your
OptiScaler installation for Youngblood, and enable DLSS in the game's settings.
The patched `d3d12.dll` and `d3d12core.dll` are unnecessary for native Vulkan.
Do not select a Vulkan-to-D3D12 upscaler backend for this test.

The shim log should contain `native Vulkan NGX ready` and a successful
`NVSDK_NGX_CUDA_CreateFeature`. OptiScaler's log should identify the native
Vulkan DLSS backend and `d4r\nvngx.dll`. If DLSS is unavailable or output
fails, retain both logs; initialization alone does not establish rendering.

## Synchronization and limits

The frontend records sampled-image compute reads, a device event indicating input
completion, a wait on a host-signaled output event, and a storage-image compute write into
the caller's still-recording command buffer. An independent CPU waiter watches each input
event, the existing CUDA worker evaluates DLSS, and the waiter signals
output only after the CUDA result has been downloaded. Output belongs to the
same frame. The frontend never ends or submits the game's command buffer.

This first implementation uses host-coherent CPU staging, with format
conversion on the GPU. It is slower than the D3D12 VRAM path. Its event protocol needs no
NVIDIA-specific or external-memory Vulkan extensions and works with Vulkan
1.0 event/compute APIs. Staging buffers and descriptors are reclaimed only after
a GPU event following the output shader confirms consumption, rather than after
CPU completion. Evaluation and feature release do not call `vkDeviceWaitIdle`
on the game's concurrently submitted queues. At most eight recorded evaluations
per feature may be outstanding.

Completed staging buffers, descriptors and events are cached per feature, and
reused only after CPU production and GPU consumption finish and the same command
buffer is being recorded again. This removes per-frame buffer allocation overhead.

Staging prefers CPU-cached coherent memory, because scalar reads from uncached
mapped memory are expensive. CUDA's RGBA16F output is copied without expansion;
the output shader decodes its packed half values when writing the game's image.
GPU validation covers all 63,488 finite half encodings, including signed zero and
subnormals. This removes the CPU half-to-float expansion measured at about
28–29 ms per 2560×1440 DOOM frame. Output/image parity for every game is not implied.

Requirements and limitations:

- Use a Vulkan 1.1 instance (or enable `VK_KHR_get_physical_device_properties2`
  for Vulkan 1.0 adapter identity queries). Evaluate outside render passes/dynamic
  rendering, on a compute-capable queue. Building the shim requires `glslangValidator`.
- Inputs use `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` and `SAMPLED` usage;
  output uses `VK_IMAGE_LAYOUT_GENERAL` and `STORAGE` usage. These follow the
  [NVIDIA DLSS resource contract](https://raw.githubusercontent.com/NVIDIA/DLSS/main/doc/DLSS_Programming_Guide_Release.pdf).
  No image layout transitions or transfer usages are needed. A developer caller
  explicitly using GENERAL for all inputs can set `D4R_VULKAN_INPUT_GENERAL=1`.
- Images must be single-sample 2D views with one mip and one array layer.
  The caller must provide accurate metadata and maintain queue ownership.
- Input sampling converts float, half, UNORM and packed formats directly into
  canonical RGBA16F color, RG16F motion, and R32F depth/exposure staging. Half
  conversion uses round-to-even and preserves finite subnormals. Combined depth/stencil formats
  require a depth-only sampled view; stencil is untouched. Storage outputs support
  RGBA16F, RGBA32F, R11G11B10F, A2B10G10R10 and RGBA8 UNORM. Packed R11/R10
  outputs require `shaderStorageImageExtendedFormats` on the existing device.
  Unsupported storage formats are rejected.
- Output must match the feature's full output dimensions. Output subrect offsets,
  transparency masks and bias color masks are rejected rather than ignored.
- Each recorded evaluation is for one submission. Reset/discard command buffers
  before re-recording; never resubmit an evaluation or submit it after releasing
  its feature. The frontend reclaims its staging objects after completed batches.
- Release cancels waiting jobs and drains CPU/CUDA work; unfinished GPU objects
  remain retained for completion or shutdown. Shutdown requires the application
  to externally synchronize its queues and discard unsubmitted recordings before
  the final device-idle cleanup. Never submit recordings after feature release.
- An unsubmitted input times out after 30 seconds (`D4R_VULKAN_SUBMIT_TIMEOUT_MS`).
  Timeout/CUDA failure clears and signals the output to unblock submitted GPU
  work, logs the failure, and makes the next evaluation fail until recreation.
- One Vulkan device per shim instance. Do not mix native Vulkan and D3D12 handles.
- Motion-vector dilation requires the existing VRAM path and is unavailable here.
  DLSS Frame Generation and Ray Reconstruction remain unsupported.

The resource ABI and initialization signatures follow
[NVIDIA's Vulkan headers](https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx_vk.h)
and the NGX core signatures used by
[OptiScaler's proxy](https://github.com/optiscaler/OptiScaler/blob/master/OptiScaler/proxies/NVNGX_Proxy.h).
Host event signaling follows the
[Vulkan event synchronization contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWaitEvents.html).

## Synthetic harness

The harness creates an AMD Vulkan device without NGX-specific device extensions,
records constant color/depth/motion inputs, evaluates DLSS, and reads the result
back after the same command buffer completes. Every submitted frame must contain
nonzero finite RGBA16F output. It also supports invalid resource and discarded
command buffer tests.

```sh
scripts/build_vulkan_dlss_harness.sh
scripts/run_vulkan_dlss_harness_proton.sh PROTON_DIR PORTABLE_D4R_DIR build/vulkan.rgba16f 3
scripts/run_vulkan_dlss_harness_proton.sh PROTON_DIR PORTABLE_D4R_DIR build/unused.rgba16f 1 invalid
scripts/run_vulkan_dlss_harness_proton.sh PROTON_DIR PORTABLE_D4R_DIR build/unused.rgba16f 1 discard
```

The runner uses an isolated prefix at `/tmp/d4r-vulkan-harness-compat`; override
`D4R_PROTON_COMPAT_DATA` to choose another test prefix. It reads the supplied
runtime and does not install into any game or existing game prefix. The shim
log is `build/vulkan-harness.log`. A harness pass is synthetic execution proof,
not Youngblood acceptance.

Local validation on 2026-10-08: 38 tests pass (three optional motion GPU tests
skipped), including the x64 resource ABI and configuration regressions. The
corrected K Vulkan harness passes 16 normal and 24 dynamic-resolution frames
with finite, nonzero output on RX 7700 XT/GE-Proton11-3, plus discarded-recording
release and invalid-resource rejection. The harness accepts Proton's NVIDIA-spoofed
adapter identity and uses DOOM's NGX app ID/SDK version. This is fixture execution
proof; Youngblood rendering still needs separate validation. The packed input
path additionally passes 64 dynamic frames, 64 frames with 15 feature recreations
across quality modes, and discarded/invalid recording checks. Its final dynamic
output is byte-for-byte identical to the preceding CPU-conversion implementation.

DOOM Eternal has produced verified nonblack menu output at 2560×1440 on this
machine. User-reported frame rates improved from 6 to 17, then 33 and 42 FPS during
staging fixes; these are not controlled benchmarks. The latest packed-input
optimization still needs an in-game comparison. Graphics-ring resets were also
observed, including after the latest game session was closed normally; stability
and exit behavior remain unresolved. Synthetic passes do not establish long-term
gameplay stability or visual parity.

Check the current sampled/storage staging independently of CUDA/NGX:

```sh
bash scripts/build_vulkan_staging.sh
c++ -std=c++20 -O2 -Ibuild/vulkan-staging tools/vulkan_staging_probe.cpp -lvulkan -o build/vulkan_staging_probe
build/vulkan_staging_probe
```

All six input and five output format checks pass on RX 7700 XT/RADV, with
correct odd-width pixels and no Khronos validation errors. Inputs have no
TRANSFER_SRC usage; output writes need no TRANSFER_DST usage. This covers
DOOM's R11G11B10 color/output, RG16F motion, R16F exposure and D32/S8 depth.

Depth/stencil staging can be checked independently of CUDA/NGX with:

```sh
c++ -std=c++20 -O2 tools/vulkan_depth_copy_probe.cpp -lvulkan -o build/vulkan_depth_copy_probe
build/vulkan_depth_copy_probe
```

On the RX 7700 XT/RADV, D16/S8 and D32/S8 passed for depth-only and combined
view masks, with every sample matching the cleared depth value. D24/S8 was
reported unsupported by this driver and skipped. This proves the depth-copy
contract, not DOOM's complete DLSS evaluation.
