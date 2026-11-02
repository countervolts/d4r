#!/usr/bin/env bash
set -euo pipefail

# Runs the D3D12 harness against d4r_nvngx.dll in an isolated Wine prefix.
# usage: run_d3d12_dlss_harness.sh PATH_TO_NGX_CORE_DLL PATH_TO_NVNGX_DLSS_DLL OUTPUT_RAW [FRAMES [IN_W IN_H OUT_W OUT_H]]
if [[ $# -lt 3 ]]; then
  printf 'usage: %s PATH_TO_NGX_CORE_DLL PATH_TO_NVNGX_DLSS_DLL OUTPUT_RAW [FRAMES]\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_PATH="$(realpath "$1")"
FEATURE_DLL_PATH="$(realpath "$2")"
OUTPUT_PATH="$(realpath -m "$3")"
FRAMES="${4:-6}"
SIZES=("${@:5}")
WINE_BIN="${WINE:-wine}"
WINEPREFIX="${D4R_WINEPREFIX:-${TMPDIR:-/tmp}/d4r-wineprefix}"
APP_DIR="${D4R_HARNESS_APPDIR:-${TMPDIR:-/tmp}/d4r-d3d12-harness-app}"
ZLUDA_ROOT="${ZLUDA_ROOT:-}"
ROCM_ROOT="${ROCM_ROOT:-}"
if [[ ! -f "$ZLUDA_ROOT/libcuda.so" || ! -f "$ROCM_ROOT/lib/libamdhip64.so" ]]; then
  printf 'Set ZLUDA_ROOT and ROCM_ROOT to the ZLUDA and HIP runtimes\n' >&2
  exit 2
fi

mkdir -p "$APP_DIR" "$WINEPREFIX/drive_c/windows/system32"
"$ROOT/scripts/build_d3d12_dlss_harness.sh"
"$ROOT/scripts/build_wine_nvcuda_bridge.sh"
cp -f "$ROOT/build/wine-nvcuda/x86_64-windows/nvcuda.dll" "$WINEPREFIX/drive_c/windows/system32/nvcuda.dll"
cp -f "$ROOT/build/d3d12_dlss_harness.exe" "$ROOT/build/d4r_nvngx.dll" "$APP_DIR/"
cp -f "$FEATURE_DLL_PATH" "$APP_DIR/nvngx_dlss.dll"
winpath() { WINEPREFIX="$WINEPREFIX" winepath -w "$1"; }
WINE_OVERRIDES="${WINEDLLOVERRIDES:-}"
for override in 'dxgi=n,b' 'd3d12=n,b' 'd3d12core=n,b' 'nvcuda=b'; do
  dll_name="${override%%=*}"
  if [[ ";$WINE_OVERRIDES;" != *";$dll_name="* ]]; then
    WINE_OVERRIDES="${WINE_OVERRIDES:+$WINE_OVERRIDES;}$override"
  fi
done
LD_LIBRARY_PATH="$ZLUDA_ROOT:$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
WINEDLLPATH="$ROOT/build/wine-nvcuda/x86_64-windows:$ROOT/build/wine-nvcuda/x86_64-unix${WINEDLLPATH:+:$WINEDLLPATH}" \
WINEDLLOVERRIDES="$WINE_OVERRIDES" WINEPREFIX="$WINEPREFIX" WINEDEBUG="${D4R_WINEDEBUG:--all}" \
D4R_NGX_CORE="$(winpath "$CORE_PATH")" D4R_NGX_FEATURE_DIR="$(winpath "$APP_DIR")" \
D4R_SHIM_LOG="$(winpath "$APP_DIR/d4r_nvngx.log")" \
  "$WINE_BIN" "$(winpath "$APP_DIR/d3d12_dlss_harness.exe")" "$(winpath "$APP_DIR/d4r_nvngx.dll")" \
  "$(winpath "$OUTPUT_PATH")" "$FRAMES" "${SIZES[@]}"
printf '\nShim log (%s):\n' "$APP_DIR/d4r_nvngx.log"
cat "$APP_DIR/d4r_nvngx.log"
