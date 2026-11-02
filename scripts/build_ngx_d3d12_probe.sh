#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"

if [[ -z "${NVSDK_NGX_INCLUDE:-}" || ! -f "$NVSDK_NGX_INCLUDE/nvsdk_ngx.h" ]]; then
  printf '%s\n' "Set NVSDK_NGX_INCLUDE to a directory containing NVIDIA's nvsdk_ngx.h and nvsdk_ngx_defs.h" >&2
  exit 2
fi

mkdir -p "$ROOT/build"
"$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -static -static-libgcc -static-libstdc++ \
  -I"$NVSDK_NGX_INCLUDE" \
  "$ROOT/tools/ngx_d3d12_probe.cpp" \
  -o "$ROOT/build/ngx_d3d12_probe.exe"
printf 'Built %s\n' "$ROOT/build/ngx_d3d12_probe.exe"
