#!/usr/bin/env bash
# Run inside packaging/build/Dockerfile.glibc241 with a recent official Rust toolchain and ROCm mounted.
# Supply isolated, patched ZLUDA/vkd3d checkouts and the usual package_release.sh inputs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${D4R_RELEASE_BUILD_ROOT:?set the build output directory}"
: "${D4R_ZLUDA_SRC:?set a build-local ZLUDA checkout with patches 0002-0007 applied}"
: "${D4R_VKD3D_SRC:?set a build-local vkd3d-proton checkout with patch 0001 applied}"
: "${D4R_ROCM_DIR:?set ROCm with clang and device libraries}"
BUILD="$(realpath -m "$D4R_RELEASE_BUILD_ROOT")"
[[ "$(getconf GNU_LIBC_VERSION)" == 'glibc 2.41' ]] || {
    echo "This release build requires GLIBC 2.41; use the pinned Debian build image" >&2; exit 2;
}
# Helper generation modifies its source checkout. Refuse the external source-cache directory.
case "$(realpath "$D4R_ZLUDA_SRC")" in "$BUILD"/*) ;; *) echo "ZLUDA source must be inside $BUILD" >&2; exit 2 ;; esac
case "$(realpath "$D4R_VKD3D_SRC")" in "$BUILD"/*) ;; *) echo "vkd3d source must be inside $BUILD" >&2; exit 2 ;; esac
mkdir -p "$BUILD/logs"
for source in "$D4R_ZLUDA_SRC/ext/llvm-project/llvm/CMakeLists.txt" "$D4R_ZLUDA_SRC/ext/HiGHS/CMakeLists.txt"; do
    [[ -f "$source" ]] || { echo "missing pinned source submodule: $source" >&2; exit 2; }
done
export CARGO_TARGET_DIR="$BUILD/zluda-target"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="$CARGO_BUILD_JOBS"
export LIBRARY_PATH="$D4R_ROCM_DIR/lib${D4R_ROCM_LINK_STUBS:+:$D4R_ROCM_LINK_STUBS}"
export LD_LIBRARY_PATH="$D4R_ROCM_DIR/lib"
export HIP_PATH="$D4R_ROCM_DIR"
export ROCM_PATH="$D4R_ROCM_DIR"
export HIP_CLANG_PATH="$D4R_ROCM_DIR/lib/llvm/bin"
{
    getconf GNU_LIBC_VERSION
    rustc --version
    cargo --version
    gcc --version | head -1
    winegcc --version | head -1
    "$D4R_ROCM_DIR/lib/llvm/bin/clang++" --version | head -1
} > "$BUILD/toolchain.txt"

ZLUDA_SOURCE_ROOT="$D4R_ZLUDA_SRC" ROCM_ROOT="$D4R_ROCM_DIR" \
    D4R_ZLUDA_PTX_HELPER_WORK="$BUILD/helpers" "$ROOT/scripts/build_zluda_ptx_helpers.sh" \
    > "$BUILD/logs/helpers.log" 2>&1
echo 'Device helpers rebuilt'
(cd "$D4R_ZLUDA_SRC" && cargo build --offline --locked --release -p zluda \
    && cargo build --offline --locked --release -p ptx --example d4r_emit) \
    > "$BUILD/logs/zluda.log" 2>&1
export D4R_ZLUDA_DIR="$CARGO_TARGET_DIR/release"
export D4R_ZLUDA_EMIT="$D4R_ZLUDA_DIR/examples/d4r_emit"
"$ROOT/scripts/check_glibc_compat.sh" 2.41 "$D4R_ZLUDA_DIR/libnvcuda.so"
echo 'ZLUDA and the offline emitter rebuilt; GLIBC check passed'
# A minimal ROCm compiler extraction may omit lld. Use the matching LLVM built above so
# all GPU targets link with the same AMDGPU/ELF support, inside the compatible environment.
for llvm_build in "$CARGO_TARGET_DIR"/release/build/llvm_zluda-*/out/build; do
    [[ -f "$llvm_build/build.ninja" ]] || continue
    ninja -j "$CARGO_BUILD_JOBS" -C "$llvm_build" lld > "$BUILD/logs/lld.log" 2>&1
    export PATH="$llvm_build/bin:$PATH"
    break
done
command -v lld >/dev/null || { echo 'missing AMDGPU linker' >&2; exit 2; }

"$ROOT/scripts/build_d4r_nvngx_shim.sh" > "$BUILD/logs/shim.log" 2>&1
"$ROOT/scripts/build_wine_nvcuda_bridge.sh" > "$BUILD/logs/bridge.log" 2>&1
"$ROOT/scripts/check_glibc_compat.sh" 2.41 "$ROOT/build/wine-nvcuda/x86_64-unix/nvcuda.dll.so"
if [[ ! -f "$BUILD/vkd3d-build/build.ninja" ]]; then
    meson setup --cross-file "$D4R_VKD3D_SRC/build-win64.txt" --buildtype release \
        -Denable_tests=false "$BUILD/vkd3d-build" "$D4R_VKD3D_SRC" > "$BUILD/logs/vkd3d.log" 2>&1
fi
ninja -j "$CARGO_BUILD_JOBS" -C "$BUILD/vkd3d-build" >> "$BUILD/logs/vkd3d.log" 2>&1
export D4R_VKD3D_DIR="$BUILD/vkd3d"
mkdir -p "$D4R_VKD3D_DIR"
x86_64-w64-mingw32-strip -o "$D4R_VKD3D_DIR/d3d12.dll" "$BUILD/vkd3d-build/libs/d3d12/d3d12.dll"
x86_64-w64-mingw32-strip -o "$D4R_VKD3D_DIR/d3d12core.dll" "$BUILD/vkd3d-build/libs/d3d12core/d3d12core.dll"
echo 'Shim, bridge and vkd3d-proton rebuilt'

export D4R_BUNDLE_TEX="$BUILD/kernels"
export D4R_DLSS_PTX_DIR="$BUILD/ptx"
python3 "$ROOT/kernels/tools/extract_dlss_ptx.py" "$D4R_BUNDLE_DLSS" "$D4R_DLSS_PTX_DIR" > "$BUILD/logs/extract.log"
build_textures() {
    local arch="$1" folder fp8 accuracy out
    folders=("$arch")
    [[ "$arch" == gfx12* ]] && folders+=("$arch-fp8")
    for folder in "${folders[@]}"; do
        fp8=0
        [[ "$folder" == *-fp8 ]] && fp8=1
        for accuracy in 0 1; do
            out="$D4R_BUNDLE_TEX/$folder"
            [[ "$accuracy" == 1 ]] && out="$D4R_BUNDLE_TEX/accuracy/$folder"
            D4R_PREFER_ACCURACY="$accuracy" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" \
                D4R_DLSS_DLL="$D4R_BUNDLE_DLSS" "$ROOT/kernels/build.sh" tex "$out" \
                > "$BUILD/logs/tex-$folder-$accuracy.log" 2>&1
            echo "Texture kernels rebuilt: $folder accuracy=$accuracy"
        done
    done
}
pending=()
failed=0
for arch in ${D4R_GPU_ARCHS:-gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201}; do
    build_textures "$arch" &
    pending+=("$!")
    if (( ${#pending[@]} >= 4 )); then
        wait "${pending[0]}" || failed=1
        pending=("${pending[@]:1}")
    fi
done
for pid in "${pending[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { echo 'Texture target build failed' >&2; exit 1; }
python3 -m unittest discover -s "$ROOT/tests" -v > "$BUILD/logs/tests.log" 2>&1
export D4R_SKIP_BUILD=1
export D4R_MAX_GLIBC=2.41
export D4R_BUILD_INFO="$BUILD/toolchain.txt"
export D4R_PACKAGE_KERNEL_JOBS=4
"$ROOT/scripts/package_release.sh" "$BUILD/dist" > "$BUILD/logs/package.log" 2>&1
echo "Release packaged in $BUILD/dist"
