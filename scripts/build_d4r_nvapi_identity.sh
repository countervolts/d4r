#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${MINGW_CC:-x86_64-w64-mingw32-gcc}"
mkdir -p "$ROOT/build"
"$CC" -std=c11 -O2 -Wall -Wextra -shared -static-libgcc \
  "$ROOT/tools/d4r_nvapi_identity.c" -o "$ROOT/build/nvapi64.dll"
printf 'Built %s\n' "$ROOT/build/nvapi64.dll"
