#!/usr/bin/env bash
# NVAPI for the NGX core on native Windows (tools/d4r_nvapi_windows.c). Run from MSYS2 with MinGW-w64 GCC.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${MINGW_CC:-x86_64-w64-mingw32-gcc}"
OUT="$ROOT/build/windows-nvapi"

mkdir -p "$OUT"
"$CC" -std=c11 -O2 -Wall -Wextra -Werror -shared -static-libgcc "$ROOT/tools/d4r_nvapi_windows.c" -ldxgi -ldxguid \
  -o "$OUT/nvapi64.dll"
printf 'Built %s\n' "$OUT/nvapi64.dll"
