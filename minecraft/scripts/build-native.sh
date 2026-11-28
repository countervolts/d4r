#!/usr/bin/env bash
# Builds the Minecraft JNI bridge (libd4r_mc.so), the offline replay test and the stock nvxrt layer next to them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cmake -S "$HERE/native" -B "$HERE/build/native" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HERE/build/native" -j"$(nproc)"
# the stock layer (no tracing): VK_NVX_binary_import emulation on RADV, built out of tree so d4r's fg/ stays untouched
SRC="$HERE/build/nvxrt-fix"
mkdir -p "$SRC"
cp "$ROOT"/fg/dlssg/nvxrt/{nvxrt.cpp,util.cpp,util.h,Makefile,nvxrt_layer.json} "$SRC/"
(cd "$SRC" && patch -s -p0 nvxrt.cpp < "$HERE/patches/nvxrt-queuesubmit2-khr.patch")
make -C "$SRC" build/libVkLayer_nvxrt.so build/nvxrt_layer.json >/dev/null
LAYER="$SRC/build"
mkdir -p "$HERE/build/d4r/layer"
cp "$LAYER/libVkLayer_nvxrt.so" "$LAYER/nvxrt_layer.json" "$HERE/build/d4r/layer/"
cp "$HERE/build/native/libd4r_mc.so" "$HERE/build/d4r/"
echo "built: $HERE/build/d4r/{libd4r_mc.so,layer/} and $HERE/build/native/fg_replay_test"
