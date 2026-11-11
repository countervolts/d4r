#!/usr/bin/env bash
set -euo pipefail

# Builds build/d4r_nvngx.dll, the D3D12/Vulkan-to-CUDA NGX core shim loaded by
# OptiScaler through its NvngxPath setting.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Native Windows (Git Bash or MSYS2): MSYS2's MinGW, the Vulkan SDK's headers and the HIP SDK (its clang-cl too) when
# they are not given. The HIP SDK's short path: "Program Files" would split the unquoted include and library flags.
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
  if ! command -v "${MINGW_CXX:-x86_64-w64-mingw32-g++}" >/dev/null && [[ -x /c/msys64/mingw64/bin/x86_64-w64-mingw32-g++ ]]; then
    export PATH="/c/msys64/mingw64/bin:$PATH"
  fi
  if [[ -z "${VULKAN_INCLUDE:-}" && ! -d /usr/include/vulkan && -n "${VULKAN_SDK:-}" ]]; then
    VULKAN_INCLUDE="$(cygpath -m "$VULKAN_SDK")/Include"
  fi
  if [[ -z "${D4R_ROCM_DIR:-}" && -n "${HIP_PATH:-}" && -f "$HIP_PATH/include/hip/hip_runtime_api.h" ]]; then
    D4R_ROCM_DIR="$(cygpath -m -s "$HIP_PATH")"
    D4R_ROCM_DIR="${D4R_ROCM_DIR%/}"
  fi
  if [[ -z "${CLANG_CL:-}" ]] && ! command -v clang-cl >/dev/null && [[ -x "${D4R_ROCM_DIR:-}/bin/clang-cl.exe" ]]; then
    CLANG_CL="$D4R_ROCM_DIR/bin/clang-cl.exe"
  fi
fi
MINGW_CXX="${MINGW_CXX:-x86_64-w64-mingw32-g++}"
CLANG_CL="${CLANG_CL:-clang-cl}"
BUILD_ID="$(bash "$ROOT/scripts/runtime_build_id.sh")"
python3 "$ROOT/scripts/build_vkhip_donor.py" "$ROOT/build/vkhip" >/dev/null

VULKAN_INCLUDE="${VULKAN_INCLUDE:-/usr/include}"

mkdir -p "$ROOT/build"
bash "$ROOT/scripts/build_d4r_nvapi_router.sh"
# Native configuration helper, launched automatically by InitializeASI.
"$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -Werror -static -municode \
  "$ROOT/tools/d4r_asi_setup.cpp" -o "$ROOT/build/d4r-setup.exe"
bash "$ROOT/scripts/build_vulkan_staging.sh"
# Only the Vulkan headers from the host include tree, not its libc headers.
# Recreated each time; MSYS2 (the native Windows build) may copy instead of linking.
rm -rf "$ROOT/build/vulkan-include"
mkdir -p "$ROOT/build/vulkan-include"
for dir in vulkan vk_video; do
  ln -s "$VULKAN_INCLUDE/$dir" "$ROOT/build/vulkan-include/$dir" 2>/dev/null ||
    cp -r "$VULKAN_INCLUDE/$dir" "$ROOT/build/vulkan-include/$dir"
done
# The native D3D12 VRAM path's shaders (tools/d4r_split_shaders.hlsl). Without dxc (DXC, PATH or the Vulkan SDK) the
# header is empty and the shim stages through host memory on native Windows instead.
SPLIT_DIR="$ROOT/build/d3d12-split"
mkdir -p "$SPLIT_DIR"
DXC="${DXC:-$(command -v dxc || true)}"
[[ -n "$DXC" || -z "${VULKAN_SDK:-}" ]] || DXC="$VULKAN_SDK/Bin/dxc.exe"
[[ -n "$DXC" && -x "$DXC" ]] || DXC="$(dirname "$VULKAN_INCLUDE")/Bin/dxc.exe"
# A generated header is replaced only when its bytes changed. The shim's objects live on between runs, and a
# header rewritten every time would ask every unit to compile again for nothing.
write_if_changed() {
  local target="$1" temp="$2"
  if [[ -f "$target" ]] && cmp -s "$target" "$temp"; then rm -f "$temp"; else mv -f "$temp" "$target"; fi
}
if [[ -x "$DXC" ]]; then
  : > "$SPLIT_DIR/d4r_split_shaders.h.new"
  for entry in to_canonical from_canonical; do
    "$DXC" -nologo -T cs_6_0 -E "$entry" -Vn "g_d4r_$entry" -Fh "$SPLIT_DIR/$entry.h.new" "$ROOT/tools/d4r_split_shaders.hlsl"
    write_if_changed "$SPLIT_DIR/$entry.h" "$SPLIT_DIR/$entry.h.new"
    printf '#include "%s.h"\n' "$entry" >> "$SPLIT_DIR/d4r_split_shaders.h.new"
  done
  write_if_changed "$SPLIT_DIR/d4r_split_shaders.h" "$SPLIT_DIR/d4r_split_shaders.h.new"
else
  printf '// built without dxc: no native D3D12 VRAM path\n#define D4R_NO_SPLIT_SHADERS 1\n' > "$SPLIT_DIR/d4r_split_shaders.h.new"
  write_if_changed "$SPLIT_DIR/d4r_split_shaders.h" "$SPLIT_DIR/d4r_split_shaders.h.new"
fi
PARAM_OBJECT="$ROOT/build/d4r_ngx_param_msvc.obj"
# Dash-style options and a separate -o: MSYS2 (the native Windows build) would rewrite /flags as paths.
"$CLANG_CL" --target=x86_64-pc-windows-msvc -nologo -std:c++20 -O2 -c -GS- -GR- -EHs-c- -Zl \
  -o "$PARAM_OBJECT" -- "$ROOT/tools/d4r_ngx_param_msvc.cpp"
# The engine (D4R_ENGINE) reaches Vulkan through a function table: MinGW has no Vulkan import library.
# engine/d3d12 is the native D3D12 engine (Windows without vkd3d-proton).
# d3dhip: the native HIP network (engine/hip_net.h) between the D3D12 engine's stages, from AMD's HIP SDK, behind
# d4r_hip_net.dll (engine/d3d12/hip_net_bridge.cpp). The shim loads that DLL on the first preset M frame that uses it:
# importing amdhip64 into the game at load time made Ghost of Tsushima's device hang before its first frame.
HIP_DEFINES=""
HIP_INCLUDES=""
HIP_SOURCES=""
HIP_LIBS=""
ROCM_DIR="${D4R_ROCM_DIR:-}"
# A HIP SDK for this DLL needs the Windows import library too: Linux ROCm (/opt/rocm) has the headers but no
# amdhip64.lib, and linking the MinGW DLL against it fails.
hip_sdk() {
  [[ -f "$1/include/hip/hip_runtime_api.h" ]] &&
    { [[ -f "$1/lib/amdhip64.lib" ]] || [[ -f "$1/lib/libamdhip64.dll.a" ]] || [[ -f "$1/lib/libamdhip64.a" ]]; }
}
if [[ -z "$ROCM_DIR" ]]; then
  for candidate in ${HIP_PATH:-} ${ROCM_PATH:-} C:/ROCm/*/ /opt/rocm; do
    if hip_sdk "$candidate"; then ROCM_DIR="$candidate"; fi
  done
fi
if [[ -n "$ROCM_DIR" ]] && hip_sdk "$ROCM_DIR"; then
  HIP_DEFINES="-DD4R_ENGINE_D3DHIP"
  HIP_INCLUDES="-I$ROCM_DIR/include"
  HIP_SOURCES="$ROOT/engine/hip_net.cpp $ROOT/engine/d3d12/hip_net_d3d12.cpp $ROOT/engine/d3d12/hip_net_rr.cpp"
  HIP_LIBS="-L$ROCM_DIR/lib -lamdhip64 -lsetupapi"
  printf 'd3dhip: HIP network from %s\n' "$ROCM_DIR"
else
  printf 'd3dhip: no HIP SDK (set D4R_ROCM_DIR); the shim has no native HIP network\n'
fi
if [[ -n "$HIP_SOURCES" ]]; then
  "$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers -DD4R_ENGINE_HIP -D__HIP_PLATFORM_AMD__ \
    $HIP_DEFINES $HIP_INCLUDES -shared -static -static-libgcc -static-libstdc++ \
    "$ROOT/engine/d3d12/hip_net_bridge.cpp" $HIP_SOURCES $HIP_LIBS -luuid -ld3d12 -ldxgi -o "$ROOT/build/d4r_hip_net.dll"
  printf 'Built %s\n' "$ROOT/build/d4r_hip_net.dll"
fi
# The shim is several translation units sharing tools/d4r_nvngx_shim_internal.h, so an edit to one of them
# recompiles that one file instead of the whole shim. They are compiled in parallel, their objects kept for
# the next run, and then linked with the engine sources into one DLL.
SHIM_FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers -DD4R_ENGINE_VK_TABLE $HIP_DEFINES
  -MMD -I"$ROOT/build/vulkan-include" -I"$ROOT/build/vulkan-staging" -I"$SPLIT_DIR")
SHIM_CORE="$ROOT/tools/d4r_nvngx_shim.cpp"
SHIM_HEADER="$ROOT/tools/d4r_nvngx_shim_internal.h"
SHIM_SOURCES=("$ROOT"/tools/d4r_nvngx_shim*.cpp)
SHIM_OBJ_DIR="$ROOT/build/shim-objects"
SHIM_ID_FILE="$SHIM_OBJ_DIR/.build-id"
SHIM_FLAGS_FILE="$SHIM_OBJ_DIR/.flags"
mkdir -p "$SHIM_OBJ_DIR"
# Only d4r_nvngx_shim.cpp prints the build id, so only its object is tied to it: an edit to another unit,
# or to any engine source, costs one recompile. These two files are stamped when their content changes,
# and an object older than its stamp is compiled again.
if [[ ! -f "$SHIM_ID_FILE" || "$(cat "$SHIM_ID_FILE")" != "$BUILD_ID" ]]; then
  printf '%s\n' "$BUILD_ID" > "$SHIM_ID_FILE"
fi
shim_flags="${SHIM_FLAGS[*]}"
if [[ ! -f "$SHIM_FLAGS_FILE" || "$(cat "$SHIM_FLAGS_FILE")" != "$shim_flags" ]]; then
  printf '%s\n' "$shim_flags" > "$SHIM_FLAGS_FILE"
fi
SHIM_JOBS=8
CPUS="$(nproc 2>/dev/null || echo 8)"
(( CPUS > SHIM_JOBS )) && SHIM_JOBS=$CPUS
# An object is kept when nothing it was compiled from is newer than it. What that was is in the -MMD file its
# last compile wrote; a unit that has none (a first build, or a compiler without -MMD) simply compiles again.
shim_up_to_date() {
  local object="$1" prerequisites="${object%.*}.d" file
  [[ -e "$object" ]] || return 1
  [[ -e "$prerequisites" ]] || return 1
  while read -r file; do
    [[ -n "$file" ]] || continue
    [[ -e "$file" ]] || return 1                    # a header went away: compile again
    if [[ "$file" -nt "$object" ]]; then return 1; fi
  done < <(sed -e '1s/^[^:]*://' -e 's/\\$//' "$prerequisites" | tr -s ' \\\n' '\n')
  return 0
}
SHIM_OBJECTS=()
shim_compiled=0
for source in "${SHIM_SOURCES[@]}"; do
  object="$SHIM_OBJ_DIR/$(basename "${source%.cpp}").obj"
  SHIM_OBJECTS+=("$object")
  if shim_up_to_date "$object" && [[ "$object" -nt "$SHIM_FLAGS_FILE" ]] &&
     { [[ "$source" != "$SHIM_CORE" ]] || [[ "$object" -nt "$SHIM_ID_FILE" ]]; }; then
    continue
  fi
  args=("${SHIM_FLAGS[@]}")
  [[ "$source" == "$SHIM_CORE" ]] && args+=("-DD4R_RUNTIME_BUILD_ID=\"$BUILD_ID\"")
  while (( $(jobs -rp | wc -l) >= SHIM_JOBS )); do wait -n; done
  "$MINGW_CXX" "${args[@]}" -c "$source" -o "$object" &
  shim_compiled=$((shim_compiled + 1))
done
while (( $(jobs -rp | wc -l) > 0 )); do wait -n; done
printf 'compiled %d of %d shim units\n' "$shim_compiled" "${#SHIM_SOURCES[@]}"
"$MINGW_CXX" -std=c++20 -O2 -Wall -Wextra -Wno-missing-field-initializers -DD4R_ENGINE_VK_TABLE $HIP_DEFINES "-DD4R_RUNTIME_BUILD_ID=\"$BUILD_ID\"" -I"$ROOT/build/vulkan-include" -I"$ROOT/build/vulkan-staging" -I"$SPLIT_DIR" -shared -static -static-libgcc -static-libstdc++ \
  "${SHIM_OBJECTS[@]}" "$ROOT/tools/d4r_ngx_param_host.cpp" "$PARAM_OBJECT" \
  -I"$ROOT/build/vkhip" "$ROOT/engine/vkhip/vkhip.cpp" "$ROOT/engine/vkhip/vkhip_net.cpp" \
  "$ROOT/engine/runtime.cpp" "$ROOT/engine/k_frame.cpp" "$ROOT/engine/game.cpp" "$ROOT/engine/game_m.cpp" "$ROOT/engine/game_rr.cpp" "$ROOT/engine/game_e.cpp" "$ROOT/engine/vk_dispatch.cpp" \
  "$ROOT/engine/d3d12/runtime_d3d12.cpp" "$ROOT/engine/d3d12/game_d3d12.cpp" \
  "$ROOT/engine/d3d12/runtime_rr_d3d12.cpp" "$ROOT/engine/d3d12/game_rr_d3d12.cpp" \
  "$ROOT/engine/d3d12/runtime_m_d3d12.cpp" "$ROOT/engine/d3d12/game_m_d3d12.cpp" \
  -ld3d12 -ldxgi -lbcrypt -lsetupapi -Wl,--entry,d4r_dll_entry -o "$ROOT/build/d4r_nvngx.dll"
printf 'Built %s\n' "$ROOT/build/d4r_nvngx.dll"
cp "$ROOT/build/d4r_nvngx.dll" "$ROOT/build/d4r.asi"
printf 'Built %s\n' "$ROOT/build/d4r.asi"
