#!/usr/bin/env bash
set -euo pipefail

# Runs the D3D12 harness through a Proton build's own launcher script, so the
# prefix, DXVK/vkd3d-proton/dxvk-nvapi and DLL overrides match a Steam game.
# usage: run_d3d12_dlss_harness_proton.sh PROTON_DIR OUTPUT_RAW [FRAMES [IN_W IN_H OUT_W OUT_H]]
if [[ $# -lt 2 ]]; then
  printf 'usage: %s PROTON_DIR OUTPUT_RAW [FRAMES [IN_W IN_H OUT_W OUT_H]]\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTON_DIR="$(realpath "$1")"
OUTPUT_PATH="$(realpath -m "$2")"
shift 2
source "$ROOT/scripts/d4r_proton_env.sh"
if [[ "${D4R_HARNESS_SKIP_BUILD:-0}" != 1 ]]; then
  "$ROOT/scripts/build_d3d12_dlss_harness.sh" >/dev/null
fi
HARNESS_APP_DIR="${D4R_HARNESS_APP_DIR:-$ROOT/build}"
mkdir -p "$HARNESS_APP_DIR"
HARNESS_EXE="$HARNESS_APP_DIR/d3d12_dlss_harness.exe"
if [[ "$(realpath -m "$HARNESS_EXE")" != "$(realpath -m "$ROOT/build/d3d12_dlss_harness.exe")" ]]; then
  cp -f "$ROOT/build/d3d12_dlss_harness.exe" "$HARNESS_EXE"
fi
export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="${D4R_PROTON_COMPAT_DATA:-${TMPDIR:-/tmp}/d4r-proton-harness-compat}"
mkdir -p "$STEAM_COMPAT_DATA_PATH"
# D4R_HARNESS_NGX_DLL selects the NGX DLL the harness drives (default: the
# shim; point it at OptiScaler installed as nvngx.dll to test OptiScaler).
NGX_DLL="${D4R_HARNESS_NGX_DLL:-$D4R_RUNTIME_DIR/bin/d4r_nvngx.dll}"
"$PROTON_DIR/proton" run "$HARNESS_EXE" "$(d4r_winpath "$NGX_DLL")" \
  "$(d4r_winpath "$OUTPUT_PATH")" "$@"
printf '\nShim log:\n'
if [[ -f "$D4R_RUNTIME_DIR/d4r_nvngx.log" ]]; then
  tail -n 20 "$D4R_RUNTIME_DIR/d4r_nvngx.log"
fi
