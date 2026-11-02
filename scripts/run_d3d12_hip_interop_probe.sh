#!/usr/bin/env bash
set -euo pipefail

# Zero-copy D3D12 <-> HIP sharing probe under Proton (see the .cpp).
# usage: run_d3d12_hip_interop_probe.sh PROTON_DIR REPORT_FILE [WIDTH HEIGHT]
if [[ $# -lt 2 ]]; then
  printf 'usage: %s PROTON_DIR REPORT_FILE [WIDTH HEIGHT]\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTON_DIR="$(realpath "$1")"
REPORT="$(realpath -m "$2")"
shift 2
source "$ROOT/scripts/d4r_proton_env.sh"
"$ROOT/scripts/build_d3d12_hip_interop_probe.sh" >/dev/null
export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="${D4R_PROTON_COMPAT_DATA:-${TMPDIR:-/tmp}/d4r-proton-harness-compat}"
mkdir -p "$STEAM_COMPAT_DATA_PATH"
"$PROTON_DIR/proton" run "$ROOT/build/d3d12_hip_interop_probe.exe" \
  "$(d4r_winpath "$D4R_RUNTIME_DIR/bin/nvcuda.dll")" "$(d4r_winpath "$REPORT")" "$@" || status=$?
cat "$REPORT"
exit "${status:-0}"
