#!/usr/bin/env bash
set -euo pipefail

# Builds DLSS frame generation (fg/dlssg, GPL-3.0-only): the NGX host (fg/dlssg/host/build, mingw-w64) and the
# nvxrt Vulkan layer and PTX scanner (fg/dlssg/nvxrt/build). The layer runs inside Steam's container runtime, so
# its glibc requirement is checked like the bridge's. VULKAN_INCLUDE: Vulkan headers (default /usr/include).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FG="$ROOT/fg/dlssg"
VULKAN_INCLUDE="${VULKAN_INCLUDE:-/usr/include}"
bash "$FG/host/build.sh" >/dev/null
make -s -C "$FG/nvxrt" -j"${D4R_BUILD_JOBS:-4}" \
  CXXFLAGS="-O2 -g -std=c++17 -fPIC -fvisibility=hidden -Wall -Wno-unused-function -I$VULKAN_INCLUDE"
"$ROOT/scripts/check_glibc_compat.sh" "${D4R_MAX_GLIBC:-2.41}" "$FG/nvxrt/build/libVkLayer_nvxrt.so"
printf 'Built %s and %s\n' "$FG/host/build/nvngx.dll_dlssg_host.dll" "$FG/nvxrt/build/libVkLayer_nvxrt.so"
