#!/usr/bin/env bash
set -euo pipefail

# Builds build/rr_backend_check.exe (tools/rr_backend_check.cpp): a DLSS Ray Reconstruction request
# through d4r_nvngx.dll, measured against the synthetic scene's ground truth. Builds the shim first,
# for the NGX parameter object it links against.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"

"$ROOT/scripts/build_d4r_nvngx_shim.sh" >/dev/null
"$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers -static -static-libgcc -static-libstdc++ \
  "$ROOT/tools/rr_backend_check.cpp" "$ROOT/tools/d4r_ngx_param_host.cpp" "$ROOT/build/d4r_ngx_param_msvc.obj" \
  -ld3d12 -ldxgi -o "$ROOT/build/rr_backend_check.exe"
printf 'Built %s\n' "$ROOT/build/rr_backend_check.exe"
