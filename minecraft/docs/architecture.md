# Architecture

```
Minecraft (Vulkan 1.2, LWJGL)                                 d4r
  GameRendererMixin ── FrameGen.java ── JNI ── libd4r_mc.so ── fg_engine (script replay)
  VulkanGpuSurfaceMixin (blit / present)                         └─ vkCmdCuLaunchKernelNVX ── nvxrt layer ── RADV
```

## Why a script
DLSS-G is a closed Windows DLL: it decides *what* to launch (about 37 kernels per real frame, 3 per generated frame), with which
parameter blocks, in which scratch buffers, and reaches the GPU only through D3D12 and NVIDIA's cubin launch API. d4r runs it under Proton
and answers the cubin API with `nvxrt`, which maps each CUDA kernel (by PTX hash) to a pre-translated SPIR-V kernel (plus hand-written T1 kernels).
A native Linux process cannot load the DLL, but the *kernels* are plain Vulkan compute, and measurement showed the orchestration is static:

- per real frame the launch list, grids and parameter blocks are identical frame after frame, except image handles / buffer pointers
  (which ping-pong between two sets: frames alternate variant 0/1) and `ClipToPrevClip` (16 floats);
- non-launch commands in steady state are only a few `vkCmdFillBuffer` (INT_MAX fills of the scatter buffers) and one R8 clear;
- first frame differs (init variant 2: history setup, three output copies); a few images start non-zero (init clears).

`tools/mkscript.py` records exactly that from a traced run of the real DLL (`patches/nvxrt-resource-trace.patch` adds resource/command logging to a *copy* of
the layer): images, views, samplers, buffers, handle tables, launch list (raw parameter bytes + symbolic patches for handles/pointers), fills, clears,
the weight buffer (`weights.bin`). `native/fg/fg_engine.cpp` rebuilds the resources on Minecraft's device, re-obtains handles from the layer
(`vkGetImageViewHandleNVX`), patches the parameter blocks and launches them. It is bit-identical to the DLL (docs/testing.md).

## Per Minecraft frame
1. `hudlessPoint` (before the GUI): copy world colour to the engine's *hudless* image; `prep.comp` copies depth (D32 -> R32F) and computes camera motion vectors
   (RG16F pixels, previous = current + mv, from depth and `Pprev*Vprev*T(camCur-camPrev)*inverse(Pcur*Vcur)` in double precision).
2. GUI draws. `blitFromTexture` hook: copy final colour to the *backbuffer* image, run the script variant (2 first frame, then 0/1), blit generated frame 1 into the acquired swapchain image.
3. `present` hook: Minecraft presents it; a presenter thread then presents generated frames 2..N-1 and the real frame snapshot (docs/presentation.md).

## Environment
`D4rPreLaunch` (Fabric pre-launch) loads `libd4r_mc.so` and sets `VK_ADD_LAYER_PATH`, `VK_INSTANCE_LAYERS=VK_LAYER_NVXRT_nvx`, `NVXRT_ARCH=gfx11`, `NVXRT_CACHE`
and in fast mode `NVXRT_TEX=hw NVXRT_ACC=fast NVXRT_SHADOW=b,c,r` before Minecraft creates its instance. The layer adds all device features its kernels need to `vkCreateDevice`;
`D4rFeatureSet` only requests `VK_NVX_binary_import` and `VK_NVX_image_view_handle` (advertised by the layer on RDNA3 only).
