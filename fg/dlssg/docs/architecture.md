# Architecture

```
game - Streamline - NGX (OptiScaler 0.9.4, NvngxPath) - nvngx.dll_dlssg_host.dll - feature 11: nvngx_dlssg.dll
                                                                                 - other features: next core or none
nvngx_dlssg.dll - LaunchCubinShaderEx (vkd3d-proton) - vkCmdCuLaunchKernelNVX - nvxrt - compute dispatch on RADV
```

## Host

Streamline and the game see an NVIDIA GPU through OptiScaler's DXGI spoofing and dxvk-nvapi (`dlssg-run` sets
`PROTON_FORCE_NVAPI=1`, `DXVK_NVAPI_ALLOW_OTHER_DRIVERS=1`, `DXVK_NVAPI_GPU_ARCH=AD100`).

DLSS-G takes the GPU architecture from its NGX core's init parameters (`Context.GPU.ArchID`) and then skips its own
NVAPI query. The host passes `0x1b0` (Blackwell), which enables multi-frame generation and makes DLSS-G load its
kernels as PTX. Streamline caps multi-frame generation at `DLSSG.MultiFrameCountMax` from the capability parameters.

The host does what an NGX core does for a feature DLL: `D3D12_Init_Ext` once per device with a parameter map, the
capability parameters, and Create/Evaluate/Release. Handles are wrapped so their ids cannot clash with the next
core's. `nvngx_dlssg.dll` only accepts calls from a module whose path contains `nvngx.dll`, and OptiScaler answers
loads of files ending in `nvngx.dll` itself, hence the name `nvngx.dll_dlssg_host.dll` and the d4r copy
`d4r_dlssg_next.dll`. The host is built with mingw-w64 using a hand-written MSVC-layout `NVSDK_NGX_Parameter` vtable.

## nvxrt

A Vulkan layer (enabled through `VK_INSTANCE_LAYERS`) that advertises `VK_NVX_binary_import` and
`VK_NVX_image_view_handle`, so vkd3d-proton's cubin path turns on.

`vkCreateCuModuleNVX` receives the DLL's fatbin. Each PTX entry is identified by the SHA-256 of its normalised PTX,
and the layer loads the matching SPIR-V from `~/.cache/nvxrt`. Nothing is translated at run time; the game runs in
the Steam runtime container, which has no Python or glslang.

`vkGetImageViewHandleNVX` returns indices into the layer's bindless descriptor sets, one slot per view. CUDA pointers
in parameter blocks are already buffer device addresses. `vkCmdCuLaunchKernelNVX` binds the pipeline, pushes the
parameter block as push constants (blocks above 248 bytes go through a host-visible ring) and dispatches, after a
memory barrier for CUDA stream order. Fixed launch sequences (the chained 8-layer DL2Net blocks) run as one persistent
dispatch with grid barriers.

Options in `dlssg/dlssg.env`: `NVXRT_TEX=hw` (texture-unit filtering), `NVXRT_ACC=fast` (fp32 accumulation),
`NVXRT_SHADOW=b,c` (RGBA8 storage for two intermediate images, only on DLL versions whose image roles are known),
`NVXRT_LOG`, `NVXRT_TIMING`.

## Translator

`tools/dlssg-setup` runs `nvxrt-scan` to extract the PTX modules and `translator/build_cache.py` to translate each
entry with `ptxc.py` into GLSL, compiled with glslangValidator.

- Structured control flow from a dominator-tree structurizer, so barriers stay in uniform control flow.
- `precise` and float-controls execution modes (FTZ fp32, preserve fp16) for exact float behaviour.
- `mma.sync`: one `v_wmma_f32_16x16x16_f16` per NVIDIA MMA, rounded to fp16 after each instruction, with operands
  built in RDNA4's WMMA register layout using subgroup shuffles.
- `rcp`/`div`/`ex2.approx` use RDNA's `v_rcp_f32`/`v_exp_f32` (within 1 ulp).
- Filtered `tex.2d` uses CUDA's 8-bit fixed-point bilinear weights over gathered texels, or the texture unit with
  `NVXRT_TEX=hw`.
- RADV folds f16 round trips on cooperative-matrix values even with `precise`; an XOR with a push-constant zero
  keeps the per-step rounding.
- Raw surface loads/stores (`suld.b`/`sust.b`) are rebuilt from typed image accesses using the view format the layer
  records per slot.

## T1 kernels

Replacements for the hot kernels, applied by `translator/t1_apply.py` after translation: DL1Net convolutions
(`dl1_conv_*`, `_dl1.py`), DL2Net blocks (`b0_*`, `b1_*`, `convPre`, `up_hf`, `_conv.py`), persistent launch groups
(`g_dl2_*`, `_chain.py`), and `pipe_bcf.py`, which reorders loads in the translated Kernel_BlendCandidatesFused. Each
file names the module hash and entry it replaces; other DLL versions keep the translated kernels. All are
bit-identical to the translation, except the fp32-accumulation variants (`*.fa.spv`) built for `NVXRT_ACC=fast`.
