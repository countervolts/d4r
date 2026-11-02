#!/usr/bin/env bash
set -euo pipefail
# Uses an isolated Proton prefix and an existing portable d4r runtime.
# usage: run_vulkan_dlss_harness_proton.sh PROTON_DIR D4R_DIR OUTPUT_RAW [FRAMES [discard|invalid]]
if [[ $# -lt 3 ]]; then
  printf 'usage: %s PROTON_DIR D4R_DIR OUTPUT_RAW [FRAMES [discard|invalid]]\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTON_DIR="$(realpath "$1")"
RUNTIME="$(realpath "$2")"
OUTPUT="$(realpath -m "$3")"
shift 3
for file in ngx/_nvngx.dll nvngx_dlss.dll nvcuda.dll zluda/libcuda.so; do
  [[ -f "$RUNTIME/$file" ]] || { printf 'Missing runtime file: %s\n' "$RUNTIME/$file" >&2; exit 2; }
done
"$ROOT/scripts/build_vulkan_dlss_harness.sh"
APP_DIR="$(mktemp -d "$ROOT/build/vulkan-harness-app-XXXXXX")"
cp "$ROOT/build/vulkan_dlss_harness.exe" "$ROOT/build/d4r_nvngx.dll" "$APP_DIR/"
ln -s "$RUNTIME/nvngx_dlss.dll" "$APP_DIR/nvngx_dlss.dll"
winpath() { printf 'Z:%s' "${1//\//\\}"; }
export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="${D4R_PROTON_COMPAT_DATA:-${TMPDIR:-/tmp}/d4r-vulkan-harness-compat}"
mkdir -p "$STEAM_COMPAT_DATA_PATH" "$(dirname "$OUTPUT")"
export D4R_NGX_CORE="$(winpath "$RUNTIME/ngx/_nvngx.dll")"
export D4R_NGX_FEATURE_DIR="$(winpath "$RUNTIME")"
export D4R_NVCUDA_BRIDGE="$(winpath "$RUNTIME/nvcuda.dll")"
export D4R_ZLUDA_LIBCUDA="$RUNTIME/zluda/libcuda.so"
export D4R_ZLUDA_NATIVE_DIR="${D4R_ZLUDA_NATIVE_DIR:-$RUNTIME/kernels}"
export D4R_ROCM_DIR="${D4R_ROCM_DIR:-$RUNTIME/rocm}"
export LD_LIBRARY_PATH="$RUNTIME/zluda:$D4R_ROCM_DIR/lib:$D4R_ROCM_DIR/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PROTON_ENABLE_NVAPI=1 DXVK_NVAPI_ALLOW_OTHER_DRIVERS=1 DXVK_NVAPI_GPU_ARCH=AD100
export D4R_CUDA_CAPTURE=0 D4R_ZLUDA_IMPLICIT_MAX_BLOCK=256 D4R_ZLUDA_WMMA=1
export D4R_SHIM_LOG="$(winpath "$ROOT/build/vulkan-harness.log")"
export WINEDEBUG="${WINEDEBUG:--all}"
"$PROTON_DIR/proton" run "$APP_DIR/vulkan_dlss_harness.exe" \
  "$(winpath "$APP_DIR/d4r_nvngx.dll")" "$(winpath "$OUTPUT")" "$@"
