#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"
CLANG_CL="${CLANG_CL:-clang-cl}"

if [[ -z "${NVSDK_NGX_INCLUDE:-}" || ! -f "$NVSDK_NGX_INCLUDE/nvsdk_ngx.h" ]]; then
  printf '%s\n' "Set NVSDK_NGX_INCLUDE to a directory containing NVIDIA's nvsdk_ngx.h and nvsdk_ngx_defs.h" >&2
  exit 2
fi

mkdir -p "$ROOT/build"
ABI_OBJECT="$ROOT/build/ngx_param_abi_probe_msvc.obj"
"$CLANG_CL" --target=x86_64-pc-windows-msvc /nologo /std:c++20 /O2 /c /GS- /GR- /EHs-c- /Zl \
  "/Fo$ABI_OBJECT" "$ROOT/tools/ngx_param_abi_probe_msvc.cpp"
"$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -static -static-libgcc -static-libstdc++ \
  -I"$NVSDK_NGX_INCLUDE" \
  "$ROOT/tools/ngx_cuda_probe.cpp" "$ABI_OBJECT" \
  -o "$ROOT/build/ngx_cuda_probe.exe"
printf 'Built %s\n' "$ROOT/build/ngx_cuda_probe.exe"
