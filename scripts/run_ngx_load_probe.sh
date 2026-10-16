#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s PATH_TO_NVNGX_DLSS_DLL\n' "$0" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DLL_PATH="$(realpath "$1")"
WINE_BIN="${WINE:-wine}"
WINEPREFIX="${D4R_WINEPREFIX:-${TMPDIR:-/tmp}/d4r-wineprefix}"

"$ROOT/scripts/build_ngx_load_probe.sh"
mkdir -p "$WINEPREFIX"
DLL_WIN_PATH="$(WINEPREFIX="$WINEPREFIX" winepath -w "$DLL_PATH")"
WINEPREFIX="$WINEPREFIX" WINEDEBUG="${D4R_WINEDEBUG:--all}" \
  "$WINE_BIN" "$ROOT/build/ngx_load_probe.exe" "$DLL_WIN_PATH"
