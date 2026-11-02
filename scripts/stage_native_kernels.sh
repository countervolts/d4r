#!/usr/bin/env bash
# Builds d4r's native kernels into a release's d4r folder: d4r/kernels/<target> (the fast set) and
# d4r/kernels/accuracy/<target> (d4r.ini [Kernels] PreferAccuracy), each with its d4r-kernels.txt manifest.
# scripts/package_release.sh (Linux) and scripts/package_release_windows.sh run it; the code objects are the same
# on both, since ZLUDA serves them through the same bridge (tools/wine_nvcuda_bridge.c).
#
# usage: scripts/stage_native_kernels.sh D4R_FOLDER clean|installer|full
#   D4R_DLSS_DLLS    colon-separated nvngx_dlss.dll files whose kernel code the native kernels accept (required)
#   D4R_ROCM_DIR     ROCm with clang (default /opt/rocm; on Windows the HIP SDK, C:/ROCm/<version>)
#   D4R_GPU_ARCHS    GPU targets (default gfx1030-gfx1036, gfx1100-gfx1103, gfx1200, gfx1201)
#   D4R_PACKAGE_KERNEL_JOBS  independent target builds at once (default 1)
# installer and full variants also get the texture kernels built from NVIDIA's PTX:
#   D4R_BUNDLE_DLSS  nvngx_dlss.dll for texture builds              D4R_BUNDLE_DLSSD  optional nvngx_dlssd.dll (RR)
#   D4R_BUNDLE_TEX   prebuilt texture-kernel code objects, one subdirectory per target folder
#   D4R_ZLUDA_EMIT   ZLUDA's d4r_emit, for the texture sets D4R_BUNDLE_TEX does not supply
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(dirname "$(realpath "${1:?usage: stage_native_kernels.sh D4R_FOLDER clean|installer|full}")")"   # the release folder, which holds d4r/
VARIANT="${2:?usage: stage_native_kernels.sh D4R_FOLDER clean|installer|full}"
case "$VARIANT" in clean|installer|full) ;; *) echo "variant must be clean, installer or full" >&2; exit 2 ;; esac
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
ARCHS="${D4R_GPU_ARCHS:-gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
: "${D4R_DLSS_DLLS:?set D4R_DLSS_DLLS to the nvngx_dlss.dll files the kernel manifest accepts}"
if [[ "$VARIANT" != clean ]]; then
  : "${D4R_BUNDLE_DLSS:?set D4R_BUNDLE_DLSS}" "${D4R_BUNDLE_TEX:?set D4R_BUNDLE_TEX}"
fi
mkdir -p "$STAGE/d4r/kernels"

IFS=: read -r -a DLLS <<< "$D4R_DLSS_DLLS"
RR_DLLS=()
[[ -z "${D4R_BUNDLE_DLSSD:-}" ]] || RR_DLLS=(--rr-dll "$D4R_BUNDLE_DLSSD")
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
ensure_rr_textures() {
  local dir="$1" accuracy="$2" missing=0 name
  [[ -n "${D4R_BUNDLE_DLSSD:-}" ]] || return 0
  for name in cuda_dldn_engine_hkpn_output_kernel_transformer \
              cuda_dldn_engine_hkpn_output_kernel_transformer_diamond_wallaby; do
    [[ -f "$dir/$name.hsaco" ]] || missing=1
  done
  if [[ "$missing" == 1 ]]; then
    : "${D4R_ZLUDA_EMIT:?set D4R_ZLUDA_EMIT or supply the RR texture variants in D4R_BUNDLE_TEX}"
    D4R_PREFER_ACCURACY="$accuracy" D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" \
      D4R_DLSSD_DLL="$D4R_BUNDLE_DLSSD" "$ROOT/kernels/build.sh" rr "$dir" >/dev/null
  fi
}
build_target() {
  local folder="$1" arch fp8 tex_dir tmp_tex accurate accurate_tex f
  arch="${folder%-fp8}"
  fp8=0
  [[ "$folder" == *-fp8 ]] && fp8=1
  # RDNA2 (gfx103x): DLSS 4's layers (preset K), built with software matrix products
  # (kernels/common/wmma_layout.h), and K's hiluma output/input kernels translated from the PTX
  # (kernels/native, no WMMA; wave32, byte-identical to ZLUDA's compile). M, L, the texture tails and Ray
  # Reconstruction call WMMA and stay translated by ZLUDA. The accuracy set keeps ZLUDA's compile of hiluma.
  if [[ "$arch" == gfx103* ]]; then
    D4R_PREFER_ACCURACY=0 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" "$ROOT/kernels/build.sh" k "$STAGE/d4r/kernels/$folder" >/dev/null
    D4R_PREFER_ACCURACY=1 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" "$ROOT/kernels/build.sh" k "$STAGE/d4r/kernels/accuracy/$folder" >/dev/null
    if [[ "$VARIANT" != clean ]]; then
      tex_dir="$D4R_BUNDLE_TEX/$folder" tmp_tex=""
      if [[ ! -d "$tex_dir" ]]; then
        tex_dir="$(mktemp -d)" tmp_tex="$tex_dir"
        D4R_PREFER_ACCURACY=0 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_DLSS_DLL="$D4R_BUNDLE_DLSS" \
          "$ROOT/kernels/build.sh" tex "$tex_dir" >/dev/null
      fi
      # only the kernels validated for RDNA2, whatever else an older bundle folder holds
      for f in "$tex_dir"/hiluma_engine_output_*_hdr_max_v2_rel.hsaco "$tex_dir"/hiluma_engine_input_*_v2_rel.hsaco; do
        [[ -f "$f" ]] && cp "$f" "$STAGE/d4r/kernels/$folder/"
      done
      [[ -z "$tmp_tex" ]] || rm -rf "$tmp_tex"
    fi
    rm -f "$STAGE/d4r/kernels/$folder"/*.resolution.txt "$STAGE/d4r/kernels/accuracy/$folder"/*.resolution.txt
    "$PYTHON" "$ROOT/kernels/tools/kernel_manifest.py" "$STAGE/d4r/kernels/$folder" "${DLLS[@]}" "${RR_DLLS[@]}"
    "$PYTHON" "$ROOT/kernels/tools/kernel_manifest.py" "$STAGE/d4r/kernels/accuracy/$folder" "${DLLS[@]}" "${RR_DLLS[@]}"
    return 0
  fi
  D4R_PREFER_ACCURACY=0 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" "$ROOT/kernels/build.sh" k "$STAGE/d4r/kernels/$folder" >/dev/null
  D4R_PREFER_ACCURACY=0 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" "$ROOT/kernels/build.sh" m "$STAGE/d4r/kernels/$folder" >/dev/null
  rm -f "$STAGE/d4r/kernels/$folder"/*.resolution.txt  # empty LTO notes from clang's -save-temps
  if [[ "$VARIANT" != clean ]]; then
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
      # the RGBA8 store capability of the native LDR output kernels (kernels/build.sh), keyed by their code bytes
      [[ -f "$tex_dir/d4r-rgba8.txt" ]] && cp "$tex_dir/d4r-rgba8.txt" "$STAGE/d4r/kernels/$folder/"
    fi
  fi
  [[ "$VARIANT" == clean ]] || ensure_l_textures "$STAGE/d4r/kernels/$folder" 0
  [[ "$VARIANT" == clean ]] || ensure_rr_textures "$STAGE/d4r/kernels/$folder" 0
  "$PYTHON" "$ROOT/kernels/tools/kernel_manifest.py" "$STAGE/d4r/kernels/$folder" "${DLLS[@]}" "${RR_DLLS[@]}"

  # Every release target also gets conservative network and texture variants. Do not copy fast
  # texture objects into this set: their compiler policy is fixed inside the binary.
  accurate="$STAGE/d4r/kernels/accuracy/$folder"
  D4R_PREFER_ACCURACY=1 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" "$ROOT/kernels/build.sh" k "$accurate" >/dev/null
  D4R_PREFER_ACCURACY=1 D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$arch" D4R_NATIVE_FP8="$fp8" "$ROOT/kernels/build.sh" m "$accurate" >/dev/null
  if [[ "$VARIANT" != clean ]]; then
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
  if [[ "$VARIANT" != clean ]]; then
    ensure_l_textures "$accurate" 1
    ensure_rr_textures "$accurate" 1
    for f in "$STAGE/d4r/kernels/$folder"/*.hsaco; do
      # the fast set's native hiluma input kernels have no accuracy build: that set keeps ZLUDA's compile
      [[ "$(basename "$f")" == hiluma_engine_input_* ]] && continue
      [[ -f "$accurate/$(basename "$f")" ]] || {
        echo "missing accuracy variant of $(basename "$f") for $folder" >&2; exit 2;
      }
    done
  fi
  rm -f "$accurate"/*.resolution.txt
  "$PYTHON" "$ROOT/kernels/tools/kernel_manifest.py" "$accurate" "${DLLS[@]}" "${RR_DLLS[@]}"
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
