#!/usr/bin/env bash
set -euo pipefail

# Builds build/d4r_boot.dll, the in-process NVIDIA identity (tools/d4r_boot.c) that the release's OptiScaler
# imports (scripts/pe_add_import.py), replacing the PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 launch options.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${MINGW_CC:-x86_64-w64-mingw32-gcc}"
mkdir -p "$ROOT/build"
"$CC" -std=c11 -O2 -Wall -Wextra -Werror -shared -static-libgcc -municode \
  "$ROOT/tools/d4r_boot.c" -o "$ROOT/build/d4r_boot.dll"
x86_64-w64-mingw32-strip "$ROOT/build/d4r_boot.dll"
printf 'Built %s\n' "$ROOT/build/d4r_boot.dll"
