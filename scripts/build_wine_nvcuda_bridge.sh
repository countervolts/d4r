#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WINEGCC="${WINEGCC:-winegcc}"
WINEBUILD="${WINEBUILD:-winebuild}"
BRIDGE_ROOT="$ROOT/build/wine-nvcuda"

mkdir -p "$BRIDGE_ROOT/x86_64-unix" "$BRIDGE_ROOT/x86_64-windows"
MOTION_FLAGS=()
if [[ -x "${D4R_ROCM_DIR:-/opt/rocm}/lib/llvm/bin/clang++" ]]; then
  "$ROOT/scripts/build_d4r_motion_kernels.sh" "$BRIDGE_ROOT/generated"
  MOTION_FLAGS=(-DD4R_MOTION_KERNELS -I"$BRIDGE_ROOT/generated")
fi
"$WINEGCC" -m64 -shared -o "$BRIDGE_ROOT/x86_64-unix/nvcuda.dll" \
  "${MOTION_FLAGS[@]}" \
  "$ROOT/tools/wine_nvcuda_bridge.c" \
  -Wb,--export="$ROOT/tools/wine_nvcuda_bridge.spec" -ldl -lpthread
"$WINEBUILD" --dll --fake-module -E "$ROOT/tools/wine_nvcuda_bridge.spec" \
  -F nvcuda.dll -b x86_64-windows -o "$BRIDGE_ROOT/x86_64-windows/nvcuda.dll"
printf 'Built Wine nvcuda bridge in %s\n' "$BRIDGE_ROOT"
