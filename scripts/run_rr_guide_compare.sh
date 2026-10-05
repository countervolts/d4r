#!/usr/bin/env bash
set -euo pipefail

# Runs the D3D12 harness's Ray Reconstruction path twice with identical inputs -
# once with the shim's guides travelling on the GPU (D4R_SHIM_RR_VRAM_GUIDES=1,
# the default) and once with them staged through the host (=0) - and compares the
# rendered images, every saved frame and the alpha output.
#
# The two runs differ only in that one variable, so any difference in the image is
# the guide path's, not the denoiser's: same colour, depth, motion, exposure, jitter,
# sizes and frame count. The harness additionally checks the guides themselves when
# D4R_HARNESS_RR_GUIDE_PROBE=1 (a guide change must affect the rendered image;
# transformer attention may spread that response beyond the changed half).
#
# usage: run_rr_guide_compare.sh OUTPUT_DIR [GUIDE_FORMAT_SET ...]
#   each set is a comma-separated D4R_HARNESS_GUIDE_FORMATS value, one token per
#   plane (GBuffer.Normals, GBuffer.Roughness, GBuffer.DiffuseAlbedo,
#   GBuffer.SpecularAlbedo). The default is one set, rgba16f,rgba16f,rgba8unorm,rgba8unorm.
#
# environment:
#   D4R_PROTON_DIR       Proton build to launch through (default: GE-Proton11-3)
#   D4R_RUNTIME_DIR      install with nvngx_dlssd.dll, _nvngx.dll, kernels (default ~/.local/share/d4r-dlss)
#   D4R_VKD3D_DIR        d4r-patched d3d12.dll/d3d12core.dll (default ~/.cache/d4r-vkd3d-d4r)
#   D4R_ZLUDA_DIR        ZLUDA build whose libcuda.so the Wine bridge loads (default ~/.cache/d4r-zluda-current)
#   D4R_ROCM_DIR         ROCm user-space runtime (default ~/.cache/d4r-rocm-runtime, else /opt/rocm)
#   D4R_NVAPI_DIR        directory holding the real DXVK-NVAPI nvapi64.dll (default: from D4R_PROTON_DIR)
#   D4R_RR_PRESET        denoiser preset, 4 = D, 5 = E (default: E)
#   D4R_HARNESS_SIZE     "IN_W IN_H OUT_W OUT_H" (default "640 360 1280 720")
#   D4R_HARNESS_FRAMES   frames per run (default 6)
#   D4R_RR_ALPHA=1       also register DLSSD.Alpha / DLSSD.OutputAlpha
#   D4R_HARNESS_RR_ALPHA_FORMAT r32f (default), r16f, r8unorm, or r16unorm
#   D4R_RR_GUIDE_PROBE=1 also run the in-harness guide response probe
#   D4R_HARNESS_SKIP_BUILD=0  build the harness and shim first (default: skip)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$(realpath -m "${1:?usage: run_rr_guide_compare.sh OUTPUT_DIR [GUIDE_FORMAT_SET ...]}")"
shift
FORMAT_SETS=("$@")
if [[ ${#FORMAT_SETS[@]} -eq 0 ]]; then
  FORMAT_SETS=("rgba16f,rgba16f,rgba8unorm,rgba8unorm")
fi

PROTON_DIR="${D4R_PROTON_DIR:-$HOME/.local/share/Steam/compatibilitytools.d/GE-Proton11-3}"
RUNTIME_DIR="${D4R_RUNTIME_DIR:-$HOME/.local/share/d4r-dlss}"
VKD3D_DIR="${D4R_VKD3D_DIR:-$HOME/.cache/d4r-vkd3d-d4r}"
ZLUDA_DIR="${D4R_ZLUDA_DIR:-$HOME/.cache/d4r-zluda-current}"
ROCM_DIR="${D4R_ROCM_DIR:-$HOME/.cache/d4r-rocm-runtime}"
[[ -d "$ROCM_DIR/lib" ]] || ROCM_DIR=/opt/rocm
NVAPI_DIR="${D4R_NVAPI_DIR:-$PROTON_DIR/files/lib/wine/nvapi/x86_64-windows}"
FRAMES="${D4R_HARNESS_FRAMES:-6}"
read -r IN_W IN_H OUT_W OUT_H <<< "${D4R_HARNESS_SIZE:-640 360 1280 720}"
APP_DIR="${D4R_RR_APP_DIR:-$OUTPUT_DIR/app}"
HARNESS_EXE="$ROOT/build/d3d12_dlss_harness.exe"

# Every input the run needs, named so a missing one is a message, not a mystery.
require() { [[ -f "$1" ]] || { printf 'Missing %s (%s)\n' "$1" "$2" >&2; exit 2; }; }
require "$HARNESS_EXE" 'run scripts/build_d3d12_dlss_harness.sh (it also builds the shim)'
require "$ROOT/build/d4r_nvngx.dll" 'run scripts/build_d4r_nvngx_shim.sh'
require "$RUNTIME_DIR/bin/nvngx_dlssd.dll" 'NVIDIA DLSS-Denoiser 3.10.7 (CUDA-capable build)'
require "$RUNTIME_DIR/ngx/_nvngx.dll" 'NVIDIA NGX core'
require "$VKD3D_DIR/d3d12.dll" 'd4r-patched vkd3d-proton (scripts/build_vkd3d_proton_d4r.sh)'
require "$VKD3D_DIR/d3d12core.dll" 'd4r-patched vkd3d-proton'
require "$NVAPI_DIR/nvapi64.dll" 'DXVK-NVAPI from a Proton build'
require "$PROTON_DIR/proton" 'a Proton build'
[[ -e "$ZLUDA_DIR/libcuda.so" ]] || { printf 'Missing %s/libcuda.so (ZLUDA)\n' "$ZLUDA_DIR" >&2; exit 2; }
[[ -d "$RUNTIME_DIR/kernels" ]] || { printf 'Missing %s/kernels (native kernels)\n' "$RUNTIME_DIR" >&2; exit 2; }

mkdir -p "$APP_DIR" "$OUTPUT_DIR"
# One directory for everything the NGX core and the shim resolve relative to each
# other: the core loads nvngx_dlssd.dll from beside the shim, and the harness loads
# the d4r-patched vkd3d from beside its own executable.
cp -f "$HARNESS_EXE" "$ROOT/build/d4r_nvngx.dll" \
      "$RUNTIME_DIR/bin/nvngx_dlssd.dll" "$RUNTIME_DIR/bin/nvngx_dlss.dll" \
      "$VKD3D_DIR/d3d12.dll" "$VKD3D_DIR/d3d12core.dll" "$NVAPI_DIR/nvapi64.dll" "$APP_DIR/"

export D4R_HARNESS_SKIP_BUILD="${D4R_HARNESS_SKIP_BUILD:-1}"
export D4R_HARNESS_APP_DIR="$APP_DIR"
export D4R_HARNESS_NGX_DLL="$APP_DIR/d4r_nvngx.dll"
export D4R_RUNTIME_DIR="$RUNTIME_DIR"
export D4R_ZLUDA_DIR="$ZLUDA_DIR"
export D4R_ROCM_DIR="$ROCM_DIR"
export D4R_ZLUDA_NATIVE_DIR="$RUNTIME_DIR/kernels"
export D4R_ZLUDA_IMPLICIT_MAX_BLOCK=256
export D4R_ZLUDA_WMMA="${D4R_ZLUDA_WMMA:-1}"
export D4R_ZLUDA_WMMA_FP8="${D4R_ZLUDA_WMMA_FP8:-1}"
export D4R_ZLUDA_WMMA_FP8_NATIVE=1
export D4R_ZLUDA_NATIVE_SWIN_ENCODERS=1
export D4R_SHIM_VRAM_INTEROP=1
export D4R_RR_ENABLE=1
export D4R_RR_PRESET="${D4R_RR_PRESET:-5}"
export D4R_HARNESS_FEATURE_ID=13
export D4R_HARNESS_SAVE_FRAMES=1
export WINEDLLOVERRIDES="d3d12=n,b;d3d12core=n,b"

run_one() { # run_one LABEL OUTPUT_RAW GUIDE_PATH
  local label="$1" output="$2" guide_path="$3"
  printf '\n== %s: guides on the %s path (%s)\n' "$label" "$guide_path" "$4" >&2
  D4R_SHIM_RR_VRAM_GUIDES="$guide_path" \
    "$ROOT/scripts/run_d3d12_dlss_harness_proton.sh" "$PROTON_DIR" "$output" \
      "$FRAMES" "$IN_W" "$IN_H" "$OUT_W" "$OUT_H" 2>&1 | tee "$OUTPUT_DIR/$label.log"
  grep -E 'Ray Reconstruction|GetFeatureRequirements|CreateFeature|output read back|correlation|alpha:|Guide probe' \
    "$OUTPUT_DIR/$label.log" || true
}

status=0
for formats in "${FORMAT_SETS[@]}"; do
  tag="${formats//,/_}"
  printf '\n### guide formats: %s\n' "$formats"
  export D4R_HARNESS_GUIDE_FORMATS="$formats"
  if [[ "${D4R_RR_ALPHA:-0}" == 1 ]]; then export D4R_HARNESS_RR_ALPHA=1; else unset D4R_HARNESS_RR_ALPHA; fi
  if [[ "${D4R_RR_GUIDE_PROBE:-0}" == 1 ]]; then
    export D4R_HARNESS_RR_GUIDE_PHASE=1 D4R_HARNESS_RR_GUIDE_PROBE=1
  fi

  host_raw="$OUTPUT_DIR/$tag.host.raw"
  gpu_raw="$OUTPUT_DIR/$tag.gpu.raw"
  run_one "$tag.host" "$host_raw" 0 "$formats"
  run_one "$tag.gpu" "$gpu_raw" 1 "$formats"

  compare=(python3 "$ROOT/tools/compare_rr_output.py" --a "$host_raw" --b "$gpu_raw"
           --width "$OUT_W" --height "$OUT_H" --frames "$FRAMES" --same-frames)
  [[ "${D4R_RR_ALPHA:-0}" == 1 ]] && compare+=(--alpha --alpha-width $((OUT_W + 8)) --alpha-height $((OUT_H + 4))
                                  --alpha-base 4 2)
  if [[ "${D4R_HARNESS_RR_ALPHA_FORMAT:-r32f}" == r8unorm ||
        "${D4R_HARNESS_RR_ALPHA_FORMAT:-r32f}" == r16unorm ]]; then
    compare+=(--alpha-sentinel 0)
  fi
  printf '\n== comparing %s host guides against %s GPU guides\n' "$tag" "$tag"
  "${compare[@]}" || status=1

  if [[ "${D4R_RR_GUIDE_PROBE:-0}" == 1 ]]; then
    # The probe is a per-run check, so each guide path has to pass it on its own.
    for label in "$tag.host" "$tag.gpu"; do
      if [[ ! -f "$OUTPUT_DIR/$label.raw.probe.txt" ]]; then
        printf 'Guide response probe did not run or did not pass for %s\n' "$label" >&2
        status=1
      else
        cat "$OUTPUT_DIR/$label.raw.probe.txt"
      fi
    done
  fi
done

printf '\n%s\n' "$([[ $status -eq 0 ]] && echo 'All guide path comparisons passed.' || echo 'Guide path comparison FAILED.')"
exit $status