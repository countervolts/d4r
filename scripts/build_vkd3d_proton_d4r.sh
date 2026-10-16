#!/usr/bin/env bash
set -euo pipefail

# Builds vkd3d-proton with patches/vkd3d-proton/ applied (split command lists
# for the shim's same-frame DLSS, D4R_SHIM_SPLIT_FRAME) into OUT_DIR, which
# the game launcher takes as D4R_VKD3D_DIR.
# usage: build_vkd3d_proton_d4r.sh OUT_DIR
# Env: VKD3D_PROTON_SRC (clone, default ~/.cache/d4r-vkd3d-proton),
#      VKD3D_PROTON_COMMIT (default 3dfc6f07, the base GE-Proton11-3 ships).
if [[ $# -ne 1 ]]; then
  printf 'usage: %s OUT_DIR\n' "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "$1")"
SRC="${VKD3D_PROTON_SRC:-$HOME/.cache/d4r-vkd3d-proton}"
COMMIT="${VKD3D_PROTON_COMMIT:-3dfc6f07}"
BUILD="$SRC-build"

[[ -d "$SRC/.git" ]] || git clone https://github.com/HansKristian-Work/vkd3d-proton.git "$SRC"
# -f: the clone is a build cache; the patch is reapplied below.
git -C "$SRC" checkout -q -f -B d4r-split "$COMMIT"
git -C "$SRC" submodule update --init --recursive -q
git -C "$SRC" apply "$ROOT"/patches/vkd3d-proton/*.patch
[[ -d "$BUILD" ]] || meson setup --cross-file "$SRC/build-win64.txt" --buildtype release -Denable_tests=false "$BUILD" "$SRC"
ninja -C "$BUILD"
mkdir -p "$OUT"
x86_64-w64-mingw32-strip -o "$OUT/d3d12.dll" "$BUILD/libs/d3d12/d3d12.dll"
x86_64-w64-mingw32-strip -o "$OUT/d3d12core.dll" "$BUILD/libs/d3d12core/d3d12core.dll"
printf 'Built %s/d3d12.dll and d3d12core.dll\n' "$OUT"
