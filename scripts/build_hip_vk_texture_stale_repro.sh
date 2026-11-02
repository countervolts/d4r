#!/usr/bin/env bash
set -euo pipefail

# Builds build/hip_vk_texture_stale_repro: a native Vulkan + HIP check that a HIP texture object on
# memory imported from Vulkan follows Vulkan's later writes (see tools/hip_vk_texture_stale_repro.cpp).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"
mkdir -p "$ROOT/build"
"$ROCM_ROOT/bin/hipcc" -std=c++17 -O2 "$ROOT/tools/hip_vk_texture_stale_repro.cpp" -lvulkan \
  -o "$ROOT/build/hip_vk_texture_stale_repro"
printf 'Built %s (run with LD_LIBRARY_PATH=%s/lib:%s/lib/llvm/lib)\n' "$ROOT/build/hip_vk_texture_stale_repro" "$ROCM_ROOT" "$ROCM_ROOT"
