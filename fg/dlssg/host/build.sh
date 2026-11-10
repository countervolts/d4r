#!/usr/bin/env bash
# Builds the Windows-side pieces with mingw-w64 (no NVIDIA code involved): the NGX host with its native-Windows cubin
# executor (cubin_d3d12.cpp; under Proton it stays off) and the PTX scanner for Windows.
# Native Windows (Git Bash): d4r's pinned llvm-mingw and the Vulkan SDK's headers. VULKAN_INCLUDE: Vulkan headers.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
CXX=${MINGW_CXX:-x86_64-w64-mingw32-g++}
if ! command -v "$CXX" >/dev/null; then
  for tc in "$ROOT"/.tools/llvm-mingw-*-ucrt-x86_64/bin; do [[ -x "$tc/x86_64-w64-mingw32-clang++" ]] && CXX="$tc/x86_64-w64-mingw32-clang++"; done
fi
if [[ -z "${VULKAN_INCLUDE:-}" ]]; then
  if [[ -n "${VULKAN_SDK:-}" ]]; then VULKAN_INCLUDE="$(cygpath -m "$VULKAN_SDK" 2>/dev/null || echo "$VULKAN_SDK")/Include"
  else VULKAN_INCLUDE=/usr/include; fi
fi
mkdir -p build
# Only the Vulkan headers from the include tree: on a Linux host /usr/include also holds glibc's headers, which break
# the MinGW build (as scripts/build_d4r_nvngx_shim.sh does).
rm -rf build/vulkan-include
mkdir -p build/vulkan-include
for dir in vulkan vk_video; do
  [[ -d "$VULKAN_INCLUDE/$dir" ]] || continue
  ln -s "$VULKAN_INCLUDE/$dir" "build/vulkan-include/$dir" 2>/dev/null || cp -r "$VULKAN_INCLUDE/$dir" "build/vulkan-include/$dir"
done
"$CXX" -std=c++17 -O2 -Wall -Wextra -Wno-cast-function-type -Wno-missing-field-initializers -Wno-unused-parameter \
  -Ibuild/vulkan-include -shared -static -static-libgcc -static-libstdc++ -o build/nvngx.dll_dlssg_host.dll \
  dlssg_host.cpp cubin_d3d12.cpp ../nvxrt/util.cpp -ldxguid -luuid
"$CXX" -O2 -std=c++17 -Wall -static -o build/nvxrt-scan.exe ../nvxrt/scan.cpp ../nvxrt/util.cpp
echo "built build/nvngx.dll_dlssg_host.dll"
