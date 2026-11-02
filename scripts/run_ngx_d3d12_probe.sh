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

mkdir -p "$WINEPREFIX" "$DATA_PATH"
"$ROOT/scripts/build_ngx_d3d12_probe.sh"
CORE_WIN_PATH="$(WINEPREFIX="$WINEPREFIX" winepath -w "$CORE_PATH")"
FEATURE_DLL_WIN_PATH="$(WINEPREFIX="$WINEPREFIX" winepath -w "$FEATURE_DLL_PATH")"
DATA_WIN_PATH="$(WINEPREFIX="$WINEPREFIX" winepath -w "$DATA_PATH")"
WINEPREFIX="$WINEPREFIX" WINEDEBUG="${D4R_WINEDEBUG:--all}" \
  "$WINE_BIN" "$ROOT/build/ngx_d3d12_probe.exe" "$CORE_WIN_PATH" "$FEATURE_DLL_WIN_PATH" "$DATA_WIN_PATH"
