#!/usr/bin/env bash
set -euo pipefail

# Runs build/rr_backend_check.exe through a Proton build in its own prefix, against a portable d4r
# folder (d4r.ini, d4r_nvngx.dll, nvcuda.dll, ngx/, zluda/, and amd_fidelityfx_denoiser_dx12.dll or
# nvngx_dlssd.dll for the backend under test), with the d4r vkd3d-proton next to the exe.
# The GPU is reported as NVIDIA (PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100), as d4r's games run.
# usage: run_rr_backend_check_proton.sh PROTON_DIR VKD3D_DIR D4R_DIR OUT_DIR [check args...]
# Env: D4R_RR_BACKEND (dlss|fsr) and any other shim variable are passed through.
#      D4R_RR_CHECK_COMPAT  prefix (default ${TMPDIR:-/tmp}/d4r-rr-check-compat)
if [[ $# -lt 4 ]]; then
  printf 'usage: %s PROTON_DIR VKD3D_DIR D4R_DIR OUT_DIR [check args...]\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTON_DIR="$(realpath "$1")"
VKD3D_DIR="$(realpath "$2")"
D4R_DIR="$(realpath "$3")"
OUT="$(realpath -m "$4")"
shift 4
winpath() { printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"; }

[[ -f "$ROOT/build/rr_backend_check.exe" ]] || "$ROOT/scripts/build_rr_backend_check.sh" >/dev/null
APP="$OUT/app"
mkdir -p "$APP"
cp -f "$ROOT/build/rr_backend_check.exe" "$APP/"
cp -f "$VKD3D_DIR/d3d12.dll" "$VKD3D_DIR/d3d12core.dll" "$APP/"
{
  printf 'proton=%s\nd4r=%s\nbackend=%s\n' "$PROTON_DIR" "$D4R_DIR" "${D4R_RR_BACKEND:-ini}"
  sha256sum "$APP/d3d12.dll" "$APP/d3d12core.dll" "$D4R_DIR"/*.dll
} > "$OUT/run-manifest.txt"
export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="$(realpath -m "${D4R_RR_CHECK_COMPAT:-${TMPDIR:-/tmp}/d4r-rr-check-compat}")"
mkdir -p "$STEAM_COMPAT_DATA_PATH"
export WINEDLLOVERRIDES="d3d12,d3d12core=n,b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export DXIL_SPIRV_CONFIG="${DXIL_SPIRV_CONFIG-wmma_rdna3_workaround}"
export VKD3D_SHADER_CACHE_PATH="${VKD3D_SHADER_CACHE_PATH:-$(winpath "$OUT")}"
export PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100
export D4R_SHIM_LOG="$(winpath "$OUT/d4r_nvngx.log")"
set +e
"$PROTON_DIR/proton" run "$APP/rr_backend_check.exe" "$(winpath "$D4R_DIR/d4r_nvngx.dll")" "$(winpath "$OUT")" "$@" \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"
status=$?
set -e
printf 'exit=%s\n' "$status" >> "$OUT/run-manifest.txt"
grep -E '^(check=|overall=|tonemapped)' "$OUT/stdout.log" || true
exit "$status"
