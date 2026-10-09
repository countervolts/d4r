#!/usr/bin/env bash
set -euo pipefail

# Builds build/d4r_nvngx.dll, the D3D12/Vulkan-to-CUDA NGX core shim loaded by
# OptiScaler through its NvngxPath setting.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"
CLANG_CL="${CLANG_CL:-clang-cl}"

VULKAN_INCLUDE="${VULKAN_INCLUDE:-/usr/include}"

mkdir -p "$ROOT/build"
bash "$ROOT/scripts/build_vulkan_staging.sh"
# Only the Vulkan headers from the host include tree, not its libc headers.
mkdir -p "$ROOT/build/vulkan-include"
ln -sfn "$VULKAN_INCLUDE/vulkan" "$ROOT/build/vulkan-include/vulkan"
ln -sfn "$VULKAN_INCLUDE/vk_video" "$ROOT/build/vulkan-include/vk_video"
PARAM_OBJECT="$ROOT/build/d4r_ngx_param_msvc.obj"
"$CLANG_CL" --target=x86_64-pc-windows-msvc /nologo /std:c++20 /O2 /c /GS- /GR- /EHs-c- /Zl \
  "/Fo$PARAM_OBJECT" -- "$ROOT/tools/d4r_ngx_param_msvc.cpp"
"$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers -I"$ROOT/build/vulkan-include" -I"$ROOT/build/vulkan-staging" -shared -static -static-libgcc -static-libstdc++ \
  "$ROOT/tools/d4r_nvngx_shim.cpp" "$ROOT/tools/d4r_ngx_param_host.cpp" "$PARAM_OBJECT" \
  -Wl,--entry,d4r_dll_entry -o "$ROOT/build/d4r_nvngx.dll"
printf 'Built %s\n' "$ROOT/build/d4r_nvngx.dll"
