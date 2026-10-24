#!/usr/bin/env bash
set -euo pipefail

# Builds vkd3d-proton with patches/vkd3d-proton/ applied (split command lists
# for the shim's same-frame DLSS, D4R_SHIM_SPLIT_FRAME) into OUT_DIR, which
# the game launcher takes as D4R_VKD3D_DIR.
# usage: build_vkd3d_proton_d4r.sh OUT_DIR
# Env: VKD3D_PROTON_SRC (clone, default ~/.cache/d4r-vkd3d-proton),
#      VKD3D_PROTON_COMMIT (default ef9a02e8; tested with GE-Proton11-11),
#      D4R_VKD3D_FSR_RR=1: also what AMD's FSR Ray Regeneration needs on RDNA 3
#        ([RayReconstruction] Backend = fsr): dxil-spirv's accepted scalar FP8 fallback (upstream PR 307,
#        merge f9f6294f, cherry-picked onto the linked translator) and -Denable_extended_emulation=true,
#        in a build directory of its own. Run games with DXIL_SPIRV_CONFIG=wmma_rdna3_workaround.
if [[ $# -ne 1 ]]; then
  printf 'usage: %s OUT_DIR\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "$1")"
SRC="${VKD3D_PROTON_SRC:-$HOME/.cache/d4r-vkd3d-proton}"
COMMIT="${VKD3D_PROTON_COMMIT:-ef9a02e8}"
BUILD="$SRC-build"

[[ -d "$SRC/.git" ]] || git clone https://github.com/HansKristian-Work/vkd3d-proton.git "$SRC"
# -f: the clone is a build cache; the patch is reapplied below.
git -C "$SRC" checkout -q -f -B d4r-split "$COMMIT"
git -C "$SRC" submodule update --init --recursive -q
git -C "$SRC" apply "$ROOT"/patches/vkd3d-proton/*.patch
EXTRA=()
if [[ "${D4R_VKD3D_FSR_RR:-0}" == 1 ]]; then
  DXIL="$SRC/subprojects/dxil-spirv"
  git -C "$DXIL" fetch -q origin f9f6294f3a6795e071bef3a64d2bee99e7c1b7bc
  if ! git -C "$DXIL" merge-base --is-ancestor f9f6294f3a6795e071bef3a64d2bee99e7c1b7bc^2 HEAD; then
    git -C "$DXIL" -c user.name=d4r -c user.email=d4r@localhost cherry-pick -x \
      $(git -C "$DXIL" rev-list --reverse f9f6294f3a6795e071bef3a64d2bee99e7c1b7bc^1..f9f6294f3a6795e071bef3a64d2bee99e7c1b7bc^2)
  fi
  BUILD="$SRC-build-fsrrr"
  EXTRA=(-Denable_extended_emulation=true)
fi
[[ -d "$BUILD" ]] || meson setup --cross-file "$SRC/build-win64.txt" --buildtype release -Denable_tests=false "${EXTRA[@]}" "$BUILD" "$SRC"
ninja -C "$BUILD"
mkdir -p "$OUT"
x86_64-w64-mingw32-strip -o "$OUT/d3d12.dll" "$BUILD/libs/d3d12/d3d12.dll"
x86_64-w64-mingw32-strip -o "$OUT/d3d12core.dll" "$BUILD/libs/d3d12core/d3d12core.dll"
printf 'Built %s/d3d12.dll and d3d12core.dll\n' "$OUT"
