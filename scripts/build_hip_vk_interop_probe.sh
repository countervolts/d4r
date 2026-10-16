#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-g++}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"

if [[ ! -f "$ROCM_ROOT/include/hip/hip_runtime_api.h" || ! -f "$ROCM_ROOT/lib/libamdhip64.so" ]]; then
  printf 'Set ROCM_ROOT to a ROCm installation containing HIP headers and libamdhip64.so\n' >&2
  exit 2
fi

mkdir -p "$ROOT/build"
"$CXX" -std=c++20 -O2 -Wall -Wextra -D__HIP_PLATFORM_AMD__=1 \
  -I"$ROCM_ROOT/include" \
  "$ROOT/tools/hip_vk_interop_probe.cpp" \
  -L"$ROCM_ROOT/lib" -lamdhip64 -lvulkan \
  -Wl,-rpath,"$ROCM_ROOT/lib" \
  -o "$ROOT/build/hip_vk_interop_probe"
printf 'Built %s\n' "$ROOT/build/hip_vk_interop_probe"
