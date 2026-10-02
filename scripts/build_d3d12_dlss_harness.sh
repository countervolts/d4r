#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"

"$ROOT/scripts/build_d4r_nvngx_shim.sh"
"$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -I"$ROOT/build/vulkan-include" -static -static-libgcc -static-libstdc++ \
  "$ROOT/tools/d3d12_dlss_harness.cpp" "$ROOT/tools/d4r_ngx_param_host.cpp" "$ROOT/build/d4r_ngx_param_msvc.obj" \
  -o "$ROOT/build/d3d12_dlss_harness.exe"
printf 'Built %s\n' "$ROOT/build/d3d12_dlss_harness.exe"
