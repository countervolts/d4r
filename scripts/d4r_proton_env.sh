# Source this to export the environment a Proton process needs to run
# official DLSS on AMD through d4r (ZLUDA + Wine CUDA bridge + NGX shim).
# Paths can be overridden before sourcing.
D4R_RUNTIME_DIR="${D4R_RUNTIME_DIR:-$HOME/.local/share/d4r-dlss}"
D4R_ZLUDA_DIR="${D4R_ZLUDA_DIR:-$HOME/.cache/d4r-zluda-current}"
# ROCm user-space runtime: a private copy under ~/.cache if present, otherwise the system /opt/rocm
if [[ -z "${D4R_ROCM_DIR:-}" ]]; then
  D4R_ROCM_DIR="$HOME/.cache/d4r-rocm-deps/root/opt/rocm"
  [[ -d "$D4R_ROCM_DIR" ]] || D4R_ROCM_DIR=/opt/rocm
fi
d4r_winpath() { printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"; }

# dxvk-nvapi on a non-NVIDIA driver, reporting Ada so NGX uploads the FP8
# network weights that match the sm_89 PTX ZLUDA compiles.
export PROTON_ENABLE_NVAPI=1
export DXVK_NVAPI_ALLOW_OTHER_DRIVERS=1
export DXVK_NVAPI_GPU_ARCH=AD100
# ZLUDA and the ROCm user-cache runtime.
export LD_LIBRARY_PATH="$D4R_ZLUDA_DIR:$D4R_ROCM_DIR/lib:$D4R_ROCM_DIR/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export D4R_ZLUDA_LIBCUDA="$D4R_ZLUDA_DIR/libcuda.so"
# Kernels without PTX launch bounds (DLSS's convolutions) are compiled for at
# most 256-thread blocks, as CUDA itself allows 255 registers for them; the
# default 1024 bound caps gfx11 waves at 96 VGPRs and makes them spill.
export D4R_ZLUDA_IMPLICIT_MAX_BLOCK="${D4R_ZLUDA_IMPLICIT_MAX_BLOCK:-256}"
# f16 tensor-core MMAs (DLSS's convolutions) run on gfx11/gfx12 WMMA instead of a
# dot-product fallback (patches/zluda/0003 and 0007); 0 selects the fallback.
export D4R_ZLUDA_WMMA="${D4R_ZLUDA_WMMA:-1}"
# DLSS frames in flight. The presented upscaled image is this many game frames
# old at most: 3 = highest frame rate (Townfall ~86 fps, age 2-3), 2 = age 2
# (~75 fps), 1 = age 1, the lowest possible with the CPU staging path (~53 fps).
export D4R_SHIM_MAX_IN_FLIGHT="${D4R_SHIM_MAX_IN_FLIGHT:-3}"
# 1 keeps DLSS inputs and output in VRAM (vkd3d-proton Vulkan interop, no CPU
# staging): Townfall ~141 fps at age 2-3 with 3 in flight, ~113 fps at age 1
# with D4R_SHIM_MAX_IN_FLIGHT=1. Needs vkd3d-proton; falls back automatically.
export D4R_SHIM_VRAM_INTEROP="${D4R_SHIM_VRAM_INTEROP:-0}"
# Shim configuration (Windows paths inside the prefix).
export D4R_NVCUDA_BRIDGE="$(d4r_winpath "$D4R_RUNTIME_DIR/bin/nvcuda.dll")"
export D4R_NGX_CORE="${D4R_NGX_CORE:-$(d4r_winpath "$D4R_RUNTIME_DIR/ngx/_nvngx.dll")}"
export D4R_NGX_FEATURE_DIR="$(d4r_winpath "$D4R_RUNTIME_DIR/dlss")"
export D4R_SHIM_LOG="${D4R_SHIM_LOG:-$(d4r_winpath "$D4R_RUNTIME_DIR/d4r_nvngx.log")}"
