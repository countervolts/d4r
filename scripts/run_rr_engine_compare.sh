#!/usr/bin/env bash
set -euo pipefail

# Runs the D3D12 harness's Ray Reconstruction path twice with identical inputs - once on the CUDA backend and once
# on the native engine (engine/runtime_rr.inc with a model from engine/compile_rr.py) - and compares the frames.
#
# usage: run_rr_engine_compare.sh OUTPUT_DIR MODEL_DIR
# environment: as scripts/run_rr_guide_compare.sh (D4R_PROTON_DIR, D4R_RUNTIME_DIR, D4R_VKD3D_DIR, D4R_ZLUDA_DIR,
#   D4R_HARNESS_SIZE, D4R_HARNESS_FRAMES); D4R_RR_PRESET 5 = E (default; the model must match)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$(realpath -m "${1:?usage: run_rr_engine_compare.sh OUTPUT_DIR MODEL_DIR}")"
MODEL_DIR="$(realpath "${2:?usage: run_rr_engine_compare.sh OUTPUT_DIR MODEL_DIR}")"
PROTON_DIR="${D4R_PROTON_DIR:-$HOME/.local/share/Steam/compatibilitytools.d/GE-Proton11-3}"
RUNTIME_DIR="${D4R_RUNTIME_DIR:-$HOME/.local/share/d4r-dlss}"
VKD3D_DIR="${D4R_VKD3D_DIR:-$RUNTIME_DIR/vkd3d}"
ZLUDA_DIR="${D4R_ZLUDA_DIR:-$HOME/.cache/d4r-zluda-rr}"
NVAPI_DIR="${D4R_NVAPI_DIR:-$PROTON_DIR/files/lib/wine/nvapi/x86_64-windows}"
FRAMES="${D4R_HARNESS_FRAMES:-6}"
read -r IN_W IN_H OUT_W OUT_H <<< "${D4R_HARNESS_SIZE:-640 360 1280 720}"
APP_DIR="$OUTPUT_DIR/app"
mkdir -p "$APP_DIR"
cp -f "$ROOT/build/d3d12_dlss_harness.exe" "$ROOT/build/d4r_nvngx.dll" \
      "$RUNTIME_DIR/bin/nvngx_dlssd.dll" "$RUNTIME_DIR/bin/nvngx_dlss.dll" \
      "$VKD3D_DIR/d3d12.dll" "$VKD3D_DIR/d3d12core.dll" "$NVAPI_DIR/nvapi64.dll" "$APP_DIR/"

export D4R_HARNESS_SKIP_BUILD=1 D4R_HARNESS_APP_DIR="$APP_DIR" D4R_HARNESS_NGX_DLL="$APP_DIR/d4r_nvngx.dll"
export D4R_RUNTIME_DIR="$RUNTIME_DIR" D4R_ZLUDA_DIR="$ZLUDA_DIR"
export D4R_ZLUDA_NATIVE_DIR="$RUNTIME_DIR/kernels" D4R_ZLUDA_IMPLICIT_MAX_BLOCK=256
export D4R_ZLUDA_WMMA=1 D4R_ZLUDA_WMMA_FP8=1 D4R_ZLUDA_WMMA_FP8_NATIVE=1
export D4R_SHIM_VRAM_INTEROP=1 D4R_RR_ENABLE=1 D4R_RR_PRESET="${D4R_RR_PRESET:-5}"
export D4R_HARNESS_FEATURE_ID=13 D4R_HARNESS_SAVE_FRAMES=1
export D4R_HARNESS_GUIDE_FORMATS="${D4R_HARNESS_GUIDE_FORMATS:-rgba16f,rgba16f,rgba8unorm,rgba8unorm}"
export WINEDLLOVERRIDES="d3d12=n,b;d3d12core=n,b"

run_one() { # run_one LABEL ENGINE
  printf '\n== %s\n' "$1" >&2
  D4R_ENGINE="$2" D4R_ENGINE_RR_SYNC=1 D4R_ENGINE_MODEL_RR_E_DIR="$MODEL_DIR" D4R_ENGINE_MODEL_RR_D_DIR="$MODEL_DIR" \
    "$ROOT/scripts/run_d3d12_dlss_harness_proton.sh" "$PROTON_DIR" "$OUTPUT_DIR/$1.raw" \
      "$FRAMES" "$IN_W" "$IN_H" "$OUT_W" "$OUT_H" > "$OUTPUT_DIR/$1.log" 2>&1 || true
  grep -E 'engine backend|engine frame|Ray Reconstruction|output read back|rror' "$OUTPUT_DIR/$1.log" | head -20 || true
}
run_one cuda 0
run_one engine 1
python3 "$ROOT/tools/compare_rr_output.py" --a "$OUTPUT_DIR/cuda.raw" --b "$OUTPUT_DIR/engine.raw" \
  --width "$OUT_W" --height "$OUT_H" --frames "$FRAMES" --same-frames
