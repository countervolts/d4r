#!/usr/bin/env bash
set -euo pipefail

# Builds build/fsr_rr_probe.exe, the standalone AMD FSR Ray Regeneration provider/execution check
# (tools/fsr_rr_probe.cpp). It needs no NVIDIA or d4r runtime files.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"

VULKAN_INCLUDE="${VULKAN_INCLUDE:-/usr/include}"
mkdir -p "$ROOT/build/vulkan-include"
ln -sfn "$VULKAN_INCLUDE/vulkan" "$ROOT/build/vulkan-include/vulkan"
ln -sfn "$VULKAN_INCLUDE/vk_video" "$ROOT/build/vulkan-include/vk_video"
"$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers -municode -static -static-libgcc \
  -static-libstdc++ -I"$ROOT/build/vulkan-include" "$ROOT/tools/fsr_rr_probe.cpp" -ld3d12 -ldxgi -o "$ROOT/build/fsr_rr_probe.exe"
printf 'Built %s\n' "$ROOT/build/fsr_rr_probe.exe"
