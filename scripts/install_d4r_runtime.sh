#!/usr/bin/env bash
set -euo pipefail

# Stages the pieces a Proton game needs under D4R_RUNTIME_DIR
# (default ~/.local/share/d4r-dlss):
#   bin/nvcuda.dll      Wine CUDA bridge (ELF builtin, preloaded by path by the shim)
#   bin/d4r_nvngx.dll   D3D12-to-CUDA NGX core shim (OptiScaler NvngxPath target)
#   ngx/_nvngx.dll      official NGX core
#   dlss/nvngx_dlss.dll official DLSS SR feature DLL (also copied into bin/)
# usage: install_d4r_runtime.sh PATH_TO_NGX_CORE_DLL PATH_TO_NVNGX_DLSS_DLL
if [[ $# -ne 2 ]]; then
  printf 'usage: %s PATH_TO_NGX_CORE_DLL PATH_TO_NVNGX_DLSS_DLL\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${D4R_RUNTIME_DIR:-$HOME/.local/share/d4r-dlss}"
"$ROOT/scripts/build_d4r_nvngx_shim.sh"
"$ROOT/scripts/build_wine_nvcuda_bridge.sh"
mkdir -p "$RUNTIME/bin" "$RUNTIME/ngx" "$RUNTIME/dlss"
cp -f "$ROOT/build/wine-nvcuda/x86_64-unix/nvcuda.dll.so" "$RUNTIME/bin/nvcuda.dll"
cp -f "$ROOT/build/d4r_nvngx.dll" "$RUNTIME/bin/d4r_nvngx.dll"
copy_unless_same() { [[ "$(realpath "$1")" == "$(realpath -m "$2")" ]] || cp -f "$1" "$2"; }
copy_unless_same "$1" "$RUNTIME/ngx/_nvngx.dll"
copy_unless_same "$2" "$RUNTIME/dlss/nvngx_dlss.dll"
# The NGX core's CUDA init ignores the feature search paths it is given and
# scans the directory of the module that calls it, i.e. the shim's.
cp -f "$RUNTIME/dlss/nvngx_dlss.dll" "$RUNTIME/bin/nvngx_dlss.dll"
sha256sum "$RUNTIME"/ngx/_nvngx.dll "$RUNTIME"/dlss/nvngx_dlss.dll
printf 'Installed d4r runtime in %s\n' "$RUNTIME"
