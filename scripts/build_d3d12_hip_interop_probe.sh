#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"

VULKAN_INCLUDE="${VULKAN_INCLUDE:-/usr/include}"

# Only the Vulkan headers from the host include tree, not its libc headers.
mkdir -p "$ROOT/build/vulkan-include"
ln -sfn "$VULKAN_INCLUDE/vulkan" "$ROOT/build/vulkan-include/vulkan"
ln -sfn "$VULKAN_INCLUDE/vk_video" "$ROOT/build/vulkan-include/vk_video"
"$MINGW_CXX" -I"$ROOT/build/vulkan-include" -std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers -static -static-libgcc -static-libstdc++ \
  "$ROOT/tools/d3d12_hip_interop_probe.cpp" -ld3d12 -o "$ROOT/build/d3d12_hip_interop_probe.exe"
printf 'Built %s\n' "$ROOT/build/d3d12_hip_interop_probe.exe"
