#!/usr/bin/env bash
# Modified in this fork for CUDA Ray Reconstruction support and validation (2026).
# Builds the drag-in release: a zip whose contents are extracted into the folder that holds a game's
# main .exe, like an OptiScaler release. packaging/D4R_README.txt describes the result.
#
# usage: scripts/package_release.sh [OUT_DIR]          (default: dist/)
#
# Inputs (environment):
#   D4R_OPTISCALER   OptiScaler release archive (.7z or .zip) or extracted folder (required; 0.9.4 tested)
#   D4R_DLSS_DLLS    colon-separated nvngx_dlss.dll files whose kernel code the native kernels accept
#                    (required; only hashes of their PTX go into the release)
#   D4R_ZLUDA_DIR    ZLUDA build with patches/zluda applied (libnvcuda.so)   default ~/.cache/d4r-zluda-current
#   D4R_VKD3D_DIR    d4r-patched vkd3d-proton (d3d12.dll, d3d12core.dll)    default ~/.cache/d4r-vkd3d-d4r
#   D4R_ROCM_DIR     ROCm with clang, for the kernels                       default /opt/rocm
#   D4R_ROCM_RUNTIME ROCm runtime bundled as d4r/rocm (scripts/fetch_rocm_runtime.sh)
#                                                                           default ~/.cache/d4r-rocm-runtime
#   D4R_MAX_GLIBC     highest GLIBC ABI the Linux libraries may require     default 2.41
#   D4R_GPU_ARCHS    GPU targets to build kernels for          default gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201
#                    (RDNA4 targets also get a <target>-fp8 folder: the kernels for d4r.ini NativeFp8 = on)
#   D4R_OPTISCALER_LICENSE  OptiScaler's LICENSE (GPL-3.0) text; default: system SPDX or common-licenses copy
#   D4R_VKD3D_SRC    vkd3d-proton source checkout, for its license files     default ~/.cache/d4r-vkd3d-proton
#   D4R_ZLUDA_SRC    ZLUDA source checkout, for its license files            default: D4R_ZLUDA_DIR's ../d4r-zluda-upstream
#   SOURCE_DATE_EPOCH  timestamp given to every packaged file                default: the last commit's
#   D4R_BUILD_INFO    optional build-toolchain/provenance text to include in d4r/source/
#   D4R_PACKAGE_KERNEL_JOBS  independent target builds at once (default 1)
# The zip also contains NVIDIA's files and the texture kernels built from NVIDIA's PTX; redistributing
# those is up to whoever publishes it (they are not covered by d4r's license):
#   D4R_BUNDLE_DLSS  nvngx_dlss.dll to include      D4R_BUNDLE_NGX  _nvngx.dll to include
#   D4R_BUNDLE_DLSSD optional nvngx_dlssd.dll to include for Ray Reconstruction
#   D4R_BUNDLE_TEX   directory with texture-kernel code objects (kernels/build.sh tex), one subdirectory
#                    per target folder (gfx1101, gfx1201, gfx1201-fp8, ...), or a flat gfx1101 directory for
#                    older builds
#                    Accuracy texture sets go in accuracy/<target>/, built with D4R_PREFER_ACCURACY=1.
#   D4R_ZLUDA_EMIT  d4r_emit from the patched ZLUDA build; required in full builds when an accuracy
#                    texture set or L's unfolded texture variants are not supplied in D4R_BUNDLE_TEX.
# D4R_BUNDLE_NVIDIA=0 leaves them out (d4r-VERSION-nonvidia.zip; users then add the two DLLs themselves).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${1:-$ROOT/dist}")"
VERSION="$(cat "$ROOT/packaging/VERSION")"
VARIANT=full
[[ "${D4R_BUNDLE_NVIDIA:-1}" == 0 ]] && VARIANT=clean
NAME="d4r-$VERSION"
[[ "$VARIANT" == clean ]] && NAME="$NAME-nonvidia"
STAGE="$OUT/$NAME"
ZLUDA="${D4R_ZLUDA_DIR:-$HOME/.cache/d4r-zluda-current}"
VKD3D="${D4R_VKD3D_DIR:-$HOME/.cache/d4r-vkd3d-d4r}"
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
ROCM_RUNTIME="${D4R_ROCM_RUNTIME:-$HOME/.cache/d4r-rocm-runtime}"
ARCHS="${D4R_GPU_ARCHS:-gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201}"
: "${D4R_OPTISCALER:?set D4R_OPTISCALER to the OptiScaler release archive or folder}"
: "${D4R_DLSS_DLLS:?set D4R_DLSS_DLLS to the nvngx_dlss.dll files the kernel manifest accepts}"
for f in "$ZLUDA/libnvcuda.so" "$VKD3D/d3d12.dll" "$VKD3D/d3d12core.dll" "$ROCM_RUNTIME/lib/libamdhip64.so.7"; do
  [[ -f "$f" ]] || { echo "missing $f (the ROCm runtime comes from scripts/fetch_rocm_runtime.sh)" >&2; exit 2; }
done
OPTISCALER_LICENSE="${D4R_OPTISCALER_LICENSE:-/usr/share/licenses/spdx/GPL-3.0-only.txt}"
[[ -n "${D4R_OPTISCALER_LICENSE:-}" || -f "$OPTISCALER_LICENSE" ]] || OPTISCALER_LICENSE=/usr/share/common-licenses/GPL-3
[[ -f "$OPTISCALER_LICENSE" ]] || { echo "missing OptiScaler GPL-3.0 license; set D4R_OPTISCALER_LICENSE to an existing file" >&2; exit 2; }

# D4R_SKIP_BUILD=1 packages the shim and bridge already in build/ (e.g. the binaries that were tested)
if [[ "${D4R_SKIP_BUILD:-0}" != 1 ]]; then
  "$ROOT/scripts/build_d4r_nvngx_shim.sh" >/dev/null
  "$ROOT/scripts/build_wine_nvcuda_bridge.sh" >/dev/null
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/d4r/zluda" "$STAGE/d4r/rocm" "$STAGE/d4r/ngx" "$STAGE/d4r/kernels" "$STAGE/d4r/licenses" "$STAGE/d4r/source"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# OptiScaler, as dxgi.dll with its INI preconfigured for d4r
if [[ -d "$D4R_OPTISCALER" ]]; then
  OPTI="$D4R_OPTISCALER"
else
  OPTI="$TMP/optiscaler"
  mkdir -p "$OPTI"
  case "$D4R_OPTISCALER" in
    *.7z) 7z x -y -o"$OPTI" "$D4R_OPTISCALER" >/dev/null ;;
    *.zip) unzip -q -o "$D4R_OPTISCALER" -d "$OPTI" ;;
    *) echo "D4R_OPTISCALER must be a .7z, a .zip or a folder" >&2; exit 2 ;;
  esac
fi
cp "$OPTI/OptiScaler.dll" "$STAGE/dxgi.dll"
python3 "$ROOT/scripts/configure_optiscaler.py" \
  "$OPTI/OptiScaler.ini" "$ROOT/packaging/optiscaler.settings" "$STAGE/OptiScaler.ini"

# d4r-patched vkd3d-proton: same-frame DLSS results
cp "$VKD3D/d3d12.dll" "$VKD3D/d3d12core.dll" "$STAGE/"

# d4r itself
cp "$ROOT/build/d4r_nvngx.dll" "$STAGE/d4r/nvngx.dll"
cp "$ROOT/build/wine-nvcuda/x86_64-unix/nvcuda.dll.so" "$STAGE/d4r/nvcuda.dll"
cp "$ZLUDA/libnvcuda.so" "$STAGE/d4r/zluda/libcuda.so"
cp -r "$ROCM_RUNTIME/lib" "$STAGE/d4r/rocm/lib"
"$ROOT/scripts/check_glibc_compat.sh" "${D4R_MAX_GLIBC:-2.41}" \
  "$STAGE/d4r/nvcuda.dll" "$STAGE/d4r/zluda/libcuda.so" "$STAGE/d4r/rocm/lib/"*
cp "$ROOT/packaging/d4r.ini" "$STAGE/d4r/d4r.ini"
cp "$ROOT/packaging/d4r-check.sh" "$STAGE/d4r/d4r-check.sh"
if [[ "$VARIANT" == full ]]; then
  : "${D4R_BUNDLE_DLSS:?set D4R_BUNDLE_DLSS}" "${D4R_BUNDLE_NGX:?set D4R_BUNDLE_NGX}" "${D4R_BUNDLE_TEX:?set D4R_BUNDLE_TEX}"
  cp "$D4R_BUNDLE_DLSS" "$STAGE/d4r/nvngx_dlss.dll"
  cp "$D4R_BUNDLE_NGX" "$STAGE/d4r/ngx/_nvngx.dll"
  if [[ -n "${D4R_BUNDLE_DLSSD:-}" ]]; then
    cp "$D4R_BUNDLE_DLSSD" "$STAGE/d4r/nvngx_dlssd.dll"
  fi
else
  printf 'Put NVIDIA'"'"'s NGX runtime, _nvngx.dll, in this folder (see D4R_README.txt).\r\n' > "$STAGE/d4r/ngx/README.txt"
fi
IFS=: read -r -a DLLS <<< "$D4R_DLSS_DLLS"
# one folder per target; RDNA4 targets also get <target>-fp8 (native FP8 WMMA, d4r.ini NativeFp8), which the
# bridge serves instead when that setting is on
folders=()
for arch in $ARCHS; do
  folders+=("$arch")
  [[ "$arch" == gfx12* ]] && folders+=("$arch-fp8")
done
# Older texture bundles predate L. Complete them from the user's DLSS DLL rather
# than silently shipping only M's folded variants under the new L support claim.
ensure_l_textures() {
  local dir="$1" accuracy="$2" missing=0 mv range v name
  local names=(rrlite_dec0_4x4)
  for mv in mvhi mvlo; do
    for range in hdr ldr; do
      names+=("rrlite_enc0_4x4_${mv}_${range}")
      for v in 3_1 3_2; do names+=("rrlite_post_${v}_${mv}_${range}"); done
    done
  done
  for name in "${names[@]}"; do [[ -f "$dir/$name.hsaco" ]] || missing=1; done
  if [[ "$missing" == 1 ]]; then
    : "${D4R_ZLUDA_EMIT:?set D4R_ZLUDA_EMIT or supply the unfolded L texture variants in D4R_BUNDLE_TEX}"
    D4R_PREFER_ACCURACY="$accuracy" D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" \
      D4R_DLSS_DLL="$D4R_BUNDLE_DLSS" "$ROOT/kernels/build.sh" l "$dir" >/dev/null
  fi
}
build_target() {
  local folder="$1" arch fp8 tex_dir accurate accurate_tex f
  arch="${folder%-fp8}"
  fp8=0
  [[ "$folder" == *-fp8 ]] && fp8=1
  D4R_PREFER_ACCURACY=0 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" "$ROOT/kernels/build.sh" k "$STAGE/d4r/kernels/$folder" >/dev/null
  D4R_PREFER_ACCURACY=0 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" "$ROOT/kernels/build.sh" m "$STAGE/d4r/kernels/$folder" >/dev/null
  rm -f "$STAGE/d4r/kernels/$folder"/*.resolution.txt  # empty LTO notes from clang's -save-temps
  if [[ "$VARIANT" == full ]]; then
    tex_dir=""
    if [[ -d "$D4R_BUNDLE_TEX/$folder" ]]; then
      tex_dir="$D4R_BUNDLE_TEX/$folder"
    elif [[ "$folder" == gfx1101 ]]; then
      tex_dir="$D4R_BUNDLE_TEX"  # older, flat gfx1101 texture-kernel builds
    fi
    if [[ -n "$tex_dir" ]]; then
      for f in "$tex_dir"/*.hsaco; do  # texture kernels only; the layers above are built from source
        [[ -f "$f" ]] || continue
        [[ -e "$STAGE/d4r/kernels/$folder/$(basename "$f")" ]] || cp "$f" "$STAGE/d4r/kernels/$folder/"
      done
    fi
  fi
  [[ "$VARIANT" != full ]] || ensure_l_textures "$STAGE/d4r/kernels/$folder" 0
  python3 "$ROOT/kernels/tools/kernel_manifest.py" "$STAGE/d4r/kernels/$folder" "${DLLS[@]}"

  # Every release target also gets conservative network and texture variants. Do not copy fast
  # texture objects into this set: their compiler policy is fixed inside the binary.
  accurate="$STAGE/d4r/kernels/accuracy/$folder"
  D4R_PREFER_ACCURACY=1 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" "$ROOT/kernels/build.sh" k "$accurate" >/dev/null
  D4R_PREFER_ACCURACY=1 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" "$ROOT/kernels/build.sh" m "$accurate" >/dev/null
  if [[ "$VARIANT" == full ]]; then
    accurate_tex="$D4R_BUNDLE_TEX/accuracy/$folder"
    if [[ -f "$accurate_tex/d4r-accuracy.txt" && "$(cat "$accurate_tex/d4r-accuracy.txt")" == 1 ]]; then
      for f in "$accurate_tex"/*.hsaco; do
        [[ -f "$f" ]] || continue
        [[ -e "$accurate/$(basename "$f")" ]] || cp "$f" "$accurate/"
      done
    else
      : "${D4R_ZLUDA_EMIT:?set D4R_ZLUDA_EMIT or supply D4R_BUNDLE_TEX/accuracy/$folder}"
      D4R_PREFER_ACCURACY=1 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" \
        D4R_DLSS_DLL="$D4R_BUNDLE_DLSS" "$ROOT/kernels/build.sh" tex "$accurate" >/dev/null
    fi
  fi
  if [[ "$VARIANT" == full ]]; then
    ensure_l_textures "$accurate" 1
    for f in "$STAGE/d4r/kernels/$folder"/*.hsaco; do
      [[ -f "$accurate/$(basename "$f")" ]] || {
        echo "missing accuracy variant of $(basename "$f") for $folder" >&2; exit 2;
      }
    done
  fi
  rm -f "$accurate"/*.resolution.txt
  python3 "$ROOT/kernels/tools/kernel_manifest.py" "$accurate" "${DLLS[@]}"
}
kernel_jobs="${D4R_PACKAGE_KERNEL_JOBS:-1}"
[[ "$kernel_jobs" =~ ^[1-9][0-9]*$ ]] || { echo "D4R_PACKAGE_KERNEL_JOBS must be positive" >&2; exit 2; }
pending=()
kernel_failed=0
for folder in "${folders[@]}"; do
  build_target "$folder" &
  pending+=("$!")
  if (( ${#pending[@]} >= kernel_jobs )); then
    wait "${pending[0]}" || kernel_failed=1
    pending=("${pending[@]:1}")
  fi
done
for pid in "${pending[@]}"; do wait "$pid" || kernel_failed=1; done
[[ "$kernel_failed" == 0 ]] || { echo "native target build failed; no release ZIP created" >&2; exit 1; }

# licenses and sources
ZLUDA_SRC="${D4R_ZLUDA_SRC:-$(dirname "$ZLUDA")/d4r-zluda-upstream}"
VKD3D_SRC="${D4R_VKD3D_SRC:-$HOME/.cache/d4r-vkd3d-proton}"
L="$STAGE/d4r/licenses"
cp "$ROOT/LICENSE" "$L/d4r-LICENSE.txt"
cp "$ROOT/NOTICE" "$L/d4r-NOTICE.txt"
cp "$ZLUDA_SRC/LICENSE-APACHE" "$L/ZLUDA-LICENSE-APACHE.txt"
cp "$ZLUDA_SRC/LICENSE-MIT" "$L/ZLUDA-LICENSE-MIT.txt"
cp "$ZLUDA_SRC/ext/llvm-project/llvm/LICENSE.TXT" "$L/LLVM-LICENSE.txt"
cp "$VKD3D_SRC/LICENSE" "$L/vkd3d-proton-LICENSE.txt"
cp "$VKD3D_SRC/COPYING" "$L/vkd3d-proton-COPYING.txt"
cp "$OPTISCALER_LICENSE" "$L/OptiScaler-LICENSE-GPL-3.0.txt"
for f in "$ROCM_RUNTIME"/licenses/*; do cp "$f" "$L/ROCm-$(basename "$f")"; done
mkdir -p "$STAGE/d4r/source/patches"
cp -r "$ROOT/patches/zluda" "$ROOT/patches/vkd3d-proton" "$STAGE/d4r/source/patches/"
ZLUDA_COMMIT="$(git -C "$ZLUDA_SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
VKD3D_COMMIT="$(git -C "$VKD3D_SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
# "@clean " / "@full " lines belong to one variant only
variant() { sed -n -e "s/^@$VARIANT //" -e '/^@[a-z]* /d' -e p; }
file_version() { [[ -f "$1" ]] && strings -el "$1" | grep -A1 '^FileVersion$' | sed -n 2p | tr ',' '.' | tr -d ' '; }
sed -e "s/@VERSION@/$VERSION/g" -e "s/@ZLUDA_COMMIT@/$ZLUDA_COMMIT/g" -e "s/@VKD3D_COMMIT@/$VKD3D_COMMIT/g" \
  -e "s/@DLSS_VERSION@/$(file_version "$STAGE/d4r/nvngx_dlss.dll")/g" -e "s/@NGX_VERSION@/$(file_version "$STAGE/d4r/ngx/_nvngx.dll")/g" \
  "$ROOT/packaging/SOURCES.txt" | variant > "$STAGE/d4r/source/SOURCES.txt"
if [[ -n "${D4R_BUILD_INFO:-}" ]]; then
  cp "$D4R_BUILD_INFO" "$STAGE/d4r/source/BUILD_INFO.txt"
fi
sed -e "s/@VERSION@/$VERSION/g" "$ROOT/packaging/D4R_README.txt" | variant | sed 's/$/\r/' > "$STAGE/D4R_README.txt"

# One timestamp for every file (SOURCE_DATE_EPOCH, default the last commit): ZLUDA's kernel cache is keyed on
# its library's size and mtime, so every extraction of the zip shares one cache.
EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" log -1 --format=%ct)}"
find "$STAGE" -exec touch -h -d "@$EPOCH" {} +
# The zip holds the folder's contents, so it can be extracted straight into the game folder.
rm -f "$OUT/$NAME.zip"
(cd "$STAGE" && find . -type f | LC_ALL=C sort | sed 's|^\./||' | zip -q -X -9 "$OUT/$NAME.zip" -@)
(cd "$OUT" && sha256sum "$NAME.zip" > "$NAME.zip.sha256")
printf 'Built %s (%s)\n' "$OUT/$NAME.zip" "$(du -h "$OUT/$NAME.zip" | cut -f1)"
