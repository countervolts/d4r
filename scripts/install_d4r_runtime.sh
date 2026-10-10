#!/usr/bin/env bash
set -euo pipefail

# Stages the pieces a Proton game needs under D4R_RUNTIME_DIR
# (default ~/.local/share/d4r-dlss):
#   bin/nvcuda.dll      Wine CUDA bridge (ELF builtin, preloaded by path by the shim)
#   bin/d4r_nvngx.dll   D3D12-to-CUDA NGX core shim (OptiScaler NvngxPath target)
#   ngx/_nvngx.dll      official NGX core
#   dlss/nvngx_dlss.dll official DLSS SR feature DLL (also copied into bin/)
#   engine/k/, k-ldr/   models of preset K for the native engine ([Engine] in d4r.ini), when D4R_ENGINE_MODEL /
#                       D4R_ENGINE_MODEL_LDR name the folders engine/compile_k.py wrote (--ldr for the second)
#   engine/m/, m-ldr/   models of preset M, when D4R_ENGINE_MODEL_M / D4R_ENGINE_MODEL_M_LDR name the folders
#                       engine/compile_m.py wrote
# Each model folder is copied whole, with the native HIP network's hipnet.bin and hip/ when built with --hip.
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
# K and M models; the -ldr variants serve games without the HDR flag
for variant in k k-ldr m m-ldr; do
  case "$variant" in
    k) source_dir="${D4R_ENGINE_MODEL:-}" required="weights.bin offsets.bin lut.bin input_k.spv output_k.spv exposure_k0.spv exposure_k1.spv" compiler=compile_k.py ;;
    k-ldr) source_dir="${D4R_ENGINE_MODEL_LDR:-}" required="weights.bin offsets.bin lut.bin input_k.spv output_k.spv" compiler="compile_k.py --ldr" ;;
    m) source_dir="${D4R_ENGINE_MODEL_M:-}" required="offsets.bin enc0_m.spv" compiler=compile_m.py ;;
    m-ldr) source_dir="${D4R_ENGINE_MODEL_M_LDR:-}" required="offsets.bin enc0_m.spv" compiler="compile_m.py --ldr" ;;
  esac
  [[ -n "$source_dir" ]] || continue
  for f in $required; do
    [[ -f "$source_dir/$f" ]] || { printf '%s lacks %s (run engine/%s)\n' "$source_dir" "$f" "$compiler" >&2; exit 2; }
  done
  rm -rf "$RUNTIME/engine/$variant"
  mkdir -p "$RUNTIME/engine"
  cp -r "$source_dir" "$RUNTIME/engine/$variant"
  printf 'Installed the native engine model in %s\n' "$RUNTIME/engine/$variant"
done
sha256sum "$RUNTIME"/ngx/_nvngx.dll "$RUNTIME"/dlss/nvngx_dlss.dll
printf 'Installed d4r runtime in %s\n' "$RUNTIME"
