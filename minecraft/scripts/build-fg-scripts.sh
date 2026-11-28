#!/bin/bash
# Records DLSS-G Lite launch scripts for Minecraft's native-resolution inputs by running the real DLL once per
# (resolution, frame-generation factor) in d4r's harness (Proton + patched vkd3d + traced nvxrt), then converts the traces.
# Outputs ~/.cache/d4r-mc-fg/WxH/mfcN/{script.txt,weights.bin}. The weights are NVIDIA data: keep them out of the repository.
# usage: build-fg-scripts.sh WIDTH HEIGHT "MFC list" PATCHED_VKD3D_DIR [DLSSG_DLL]
set -euo pipefail
W=${1:-2560}; H=${2:-1440}; MFCS=${3:-"1 2 3 4 5"}; VKD3D=$(readlink -f "${4:?need the patched vkd3d dir (d3d12.dll + d3d12core.dll)}")
DLL=${5:-$HOME/.cache/d4r/nvidia/dlssg/nvngx_dlssg.dll}
MC=$(cd "$(dirname "$0")/.." && pwd); ROOT=$(cd "$MC/.." && pwd)
CACHE=${NVXRT_CACHE:-$HOME/.cache/nvxrt-gfx11}; OUT=${D4R_MC_FG:-$HOME/.cache/d4r-mc-fg}/${W}x${H}
TRACE_LAYER=$MC/build/nvxrt-trace
if [ ! -f "$TRACE_LAYER/build/libVkLayer_nvxrt.so" ]; then
  mkdir -p "$TRACE_LAYER"; cp "$ROOT"/fg/dlssg/nvxrt/{nvxrt.cpp,util.cpp,util.h,Makefile,nvxrt_layer.json} "$TRACE_LAYER/"
  (cd "$TRACE_LAYER" && patch -p0 nvxrt.cpp < "$MC/patches/nvxrt-resource-trace.patch" && make build/libVkLayer_nvxrt.so)
fi
WORK=$(mktemp -d "${D4R_TMP:-$HOME/.cache}/d4r-fg-rec.XXXXXX")
for M in $MFCS; do
  python3 -I "$MC/tools/mkcapture.py" "$WORK/cap$M" --w "$W" --h "$H" --frames 8 --mfc "$M"
  run() { # extra env, out dir, frames
    ( cd "$ROOT/fg/dlssg"
      env NVXRT_LAYER_DIR=$TRACE_LAYER/build NVXRT_ARCH=gfx11 NVXRT_CACHE=$CACHE VKD3D_DIR=$VKD3D HARNESS_NO_SETUP=1 \
          NVXRT_TRACE=1 NVXRT_TRACE_RAW=1 HOST_INI_EXTRA="Lite=1" $1 \
          harness/run.sh "$DLL" "$2" --mfc "$M" --replay "$WORK/cap$M" --frames "$3" >"$2.stdout" 2>&1 )
  }
  run "" "$WORK/rec$M" 8
  N=$(grep -m1 -o "launch #[0-9]* k_conv_fp16_nhwc" "$WORK/rec$M/nvxrt.log" | sed 's/launch #\([0-9]*\).*/\1/')
  run "NVXRT_DUMP=$WORK/dump$M NVXRT_CAPTURE=$N" "$WORK/recd$M" 3
  PRE=$(ls "$WORK/dump$M"/L$N/pre_*.bin | while read f; do echo "$(stat -c %s "$f") $f"; done | grep "^16777216 " | head -1 | cut -d' ' -f2)
  python3 -I "$MC/tools/mkscript.py" "$WORK/rec$M/nvxrt.log" "$OUT/mfc$M" --cache "$CACHE" --weights "$PRE"
done
rm -rf "$WORK"
echo "scripts in $OUT"
