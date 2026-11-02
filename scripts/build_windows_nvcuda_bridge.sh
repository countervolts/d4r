#!/usr/bin/env bash
# Native Windows build of the CUDA bridge (tools/wine_nvcuda_bridge.c, the same source as the Wine build): an
# nvcuda.dll that forwards to ZLUDA's Windows nvcuda.dll in zluda\ beside it. Run from MSYS2 with MinGW-w64 GCC.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${MINGW_CC:-x86_64-w64-mingw32-gcc}"
OUT="$ROOT/build/windows-nvcuda"
BUILD_ID="$(bash "$ROOT/scripts/runtime_build_id.sh")"

mkdir -p "$OUT"
# the exports of the Wine spec: every entry is stdcall, which x64 does not decorate
{ echo 'LIBRARY nvcuda.dll'; echo 'EXPORTS'; awk '$1 == "@" { print "    " $NF }' "$ROOT/tools/wine_nvcuda_bridge.spec"; } \
  > "$OUT/nvcuda.def"
"$CC" -std=gnu11 -O2 -shared -static-libgcc -DD4R_NATIVE_WINDOWS "-DD4R_RUNTIME_BUILD_ID=\"$BUILD_ID\"" \
  "$ROOT/tools/wine_nvcuda_bridge.c" "$OUT/nvcuda.def" -Wl,-Bstatic -lpthread -Wl,-Bdynamic -o "$OUT/nvcuda.dll"
printf 'Built %s (put ZLUDA'"'"'s nvcuda.dll in %s)\n' "$OUT/nvcuda.dll" "$OUT/zluda"
