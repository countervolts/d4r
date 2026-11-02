#!/usr/bin/env bash
set -euo pipefail

# Runs build/fsr_rr_probe.exe through a Proton build in its own prefix, with a given vkd3d-proton
# (d3d12.dll + d3d12core.dll) and AMD denoiser DLL copied next to the exe, as a game install would
# have them. Nothing outside APP_DIR and the prefix is touched.
# usage: run_fsr_rr_probe_proton.sh PROTON_DIR VKD3D_DIR DENOISER_DLL OUT_DIR [probe args...]
# Env: D4R_FSR_PROBE_COMPAT  prefix (default ${TMPDIR:-/tmp}/d4r-fsr-rr-probe-compat)
#      D4R_FSR_PROBE_APP     app directory (default OUT_DIR/app)
#      DXIL_SPIRV_CONFIG     passed through; defaults to wmma_rdna3_workaround (gfx11 WMMA path)
#      D4R_FSR_PROBE_SPOOF=1 report an NVIDIA GPU the way d4r's DLSS games run
#                            (PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100)
if [[ $# -lt 4 ]]; then
  printf 'usage: %s PROTON_DIR VKD3D_DIR DENOISER_DLL OUT_DIR [probe args...]\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTON_DIR="$(realpath "$1")"
VKD3D_DIR="$(realpath "$2")"
DENOISER="$(realpath "$3")"
OUT="$(realpath -m "$4")"
shift 4
winpath() { printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"; }

[[ -f "$ROOT/build/fsr_rr_probe.exe" ]] || "$ROOT/scripts/build_fsr_rr_probe.sh" >/dev/null
APP="${D4R_FSR_PROBE_APP:-$OUT/app}"
mkdir -p "$APP" "$OUT"
cp -f "$ROOT/build/fsr_rr_probe.exe" "$APP/"
cp -f "$VKD3D_DIR/d3d12.dll" "$VKD3D_DIR/d3d12core.dll" "$APP/"
cp -f "$DENOISER" "$APP/amd_fidelityfx_denoiser_dx12.dll"
{
  printf 'proton=%s\n' "$PROTON_DIR"
  sha256sum "$APP/d3d12.dll" "$APP/d3d12core.dll" "$APP/amd_fidelityfx_denoiser_dx12.dll" "$APP/fsr_rr_probe.exe"
} > "$OUT/run-manifest.txt"

export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="$(realpath -m "${D4R_FSR_PROBE_COMPAT:-${TMPDIR:-/tmp}/d4r-fsr-rr-probe-compat}")"
mkdir -p "$STEAM_COMPAT_DATA_PATH"
export WINEDLLOVERRIDES="d3d12,d3d12core=n,b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
export DXIL_SPIRV_CONFIG="${DXIL_SPIRV_CONFIG-wmma_rdna3_workaround}"
export VKD3D_SHADER_CACHE_PATH="${VKD3D_SHADER_CACHE_PATH:-$(winpath "$OUT")}"
export VKD3D_DEBUG="${VKD3D_DEBUG:-warn}"
if [[ "${D4R_FSR_PROBE_SPOOF:-0}" == 1 ]]; then
  export PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100
fi
printf 'DXIL_SPIRV_CONFIG=%s spoof=%s\n' "$DXIL_SPIRV_CONFIG" "${D4R_FSR_PROBE_SPOOF:-0}" >> "$OUT/run-manifest.txt"
set +e
"$PROTON_DIR/proton" run "$APP/fsr_rr_probe.exe" "$(winpath "$APP/amd_fidelityfx_denoiser_dx12.dll")" \
  "$(winpath "$OUT")" "$@" 2> "$OUT/stderr.log" | tee "$OUT/stdout.log"
status=${PIPESTATUS[0]}
set -e
printf 'exit=%s\n' "$status" >> "$OUT/run-manifest.txt"
exit "$status"
