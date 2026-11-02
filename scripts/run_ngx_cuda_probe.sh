#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'usage: %s PATH_TO_NGX_CORE_DLL PATH_TO_NVNGX_DLSS_DLL WRITABLE_APPDATA_DIRECTORY\n' "$0" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_PATH="$(realpath "$1")"
FEATURE_DLL_PATH="$(realpath "$2")"
DATA_PATH="$(realpath -m "$3")"
WINE_BIN="${WINE:-wine}"
WINEPREFIX="${D4R_WINEPREFIX:-${TMPDIR:-/tmp}/d4r-wineprefix}"
APP_DIR="${D4R_NGX_CUDA_APPDIR:-${TMPDIR:-/tmp}/d4r-ngxcuda-app}"
CAPTURE_DIR="${D4R_CUDA_CAPTURE_DIR:-${TMPDIR:-/tmp}/d4r-dlss-cuda-modules}"
ZLUDA_ROOT="${ZLUDA_ROOT:-}"
ROCM_ROOT="${ROCM_ROOT:-}"

TMP_ROOT="$(realpath -m "${TMPDIR:-/tmp}")"
WINEPREFIX="$(realpath -m "$WINEPREFIX")"
APP_DIR="$(realpath -m "$APP_DIR")"
DATA_PATH="$(realpath -m "$DATA_PATH")"
CAPTURE_DIR="$(realpath -m "$CAPTURE_DIR")"
case "$WINEPREFIX/" in
  "$TMP_ROOT/"*) ;;
  *)
    printf 'D4R_WINEPREFIX must be an isolated prefix under %s; refusing to modify a non-temporary prefix\n' "$TMP_ROOT" >&2
    exit 2
    ;;
esac
for isolated_path in "$APP_DIR" "$DATA_PATH" "$CAPTURE_DIR"; do
  case "$isolated_path/" in
    "$TMP_ROOT/"*) ;;
    *)
      printf 'App directory, appdata, and CUDA captures must stay under %s; refusing to write outside temporary storage\n' "$TMP_ROOT" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$ZLUDA_ROOT/libcuda.so" || ! -f "$ROCM_ROOT/lib/libamdhip64.so" ]]; then
  printf 'Set ZLUDA_ROOT and ROCM_ROOT to the temporary or installed ZLUDA/HIP runtimes\n' >&2
  exit 2
fi

mkdir -p "$WINEPREFIX/drive_c/windows/system32" "$DATA_PATH" "$APP_DIR" "$CAPTURE_DIR"
"$ROOT/scripts/build_ngx_cuda_probe.sh"
"$ROOT/scripts/build_wine_nvcuda_bridge.sh"
cp -f "$ROOT/build/wine-nvcuda/x86_64-windows/nvcuda.dll" "$WINEPREFIX/drive_c/windows/system32/nvcuda.dll"
APP_EXE_PATH="$APP_DIR/ngx_cuda_probe.exe"
APP_FEATURE_DLL_PATH="$APP_DIR/nvngx_dlss.dll"
cp -f "$ROOT/build/ngx_cuda_probe.exe" "$APP_EXE_PATH"
cp -f "$FEATURE_DLL_PATH" "$APP_FEATURE_DLL_PATH"
CORE_WIN_PATH="$(WINEPREFIX="$WINEPREFIX" winepath -w "$CORE_PATH")"
FEATURE_DLL_WIN_PATH="$(WINEPREFIX="$WINEPREFIX" winepath -w "$APP_FEATURE_DLL_PATH")"
DATA_WIN_PATH="$(WINEPREFIX="$WINEPREFIX" winepath -w "$DATA_PATH")"
APP_EXE_WIN_PATH="$(WINEPREFIX="$WINEPREFIX" winepath -w "$APP_EXE_PATH")"
WINE_OVERRIDES="${WINEDLLOVERRIDES:-}"
for override in 'dxgi=n,b' 'd3d12=n,b' 'd3d12core=n,b' 'nvcuda=b'; do
  dll_name="${override%%=*}"
  if [[ ";$WINE_OVERRIDES;" != *";$dll_name="* ]]; then
    WINE_OVERRIDES="${WINE_OVERRIDES:+$WINE_OVERRIDES;}$override"
  fi
done
LD_LIBRARY_PATH="$ZLUDA_ROOT:$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
WINEDLLPATH="$ROOT/build/wine-nvcuda/x86_64-windows:$ROOT/build/wine-nvcuda/x86_64-unix${WINEDLLPATH:+:$WINEDLLPATH}" \
WINEDLLOVERRIDES="$WINE_OVERRIDES" WINEPREFIX="$WINEPREFIX" \
WINEDEBUG="${D4R_WINEDEBUG:--all}" \
D4R_CUDA_CAPTURE_DIR="$CAPTURE_DIR" \
  "$WINE_BIN" "$APP_EXE_WIN_PATH" "$CORE_WIN_PATH" "$FEATURE_DLL_WIN_PATH" "$DATA_WIN_PATH"
