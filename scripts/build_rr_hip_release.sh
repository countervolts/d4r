#!/usr/bin/env bash
# Prebuilds DLSS Ray Reconstruction's HIP network stages (engine/rr/hip, run through vkhip) for each RDNA 3 target, for
# package_release.sh's D4R_BUNDLE_RR_HIP. The stages are d4r's own kernels; what they are compiled against is the
# model's layout (stage shapes, weight offsets, weight file sizes), which is the same on every machine for the
# nvngx_dlssd.dll install.sh pins. Each set carries that layout as rr-hip.txt (packaging/model_tools.py
# rr-hip-signature); the model builder installs a set only into a model with the same signature. No weights or other
# data read from the DLL are written to OUT_DIR.
#
# usage: scripts/build_rr_hip_release.sh NVNGX_DLSSD_DLL OUT_DIR
#   D4R_GPU_ARCHS  RDNA 3 targets (default gfx1100 gfx1101 gfx1102 gfx1103)
#   D4R_ROCM_DIR   ROCm with clang and the device libraries (default /opt/rocm)
# OUT_DIR/<arch>/rr-d-gfx11/{STAGE.hsaco, rr-hip.txt}, likewise rr-e-gfx11.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DLL="$(realpath "${1:?usage: build_rr_hip_release.sh NVNGX_DLSSD_DLL OUT_DIR}")"
OUT="$(realpath -m "${2:?usage: build_rr_hip_release.sh NVNGX_DLSSD_DLL OUT_DIR}")"
WANT=4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80   # nvngx_dlssd.dll 310.9.1, as install.sh
[[ "$(sha256sum "$DLL" | cut -d' ' -f1)" == "$WANT" ]] || { echo "$DLL is not nvngx_dlssd.dll 310.9.1" >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
rm -rf "$OUT"; mkdir -p "$OUT"
for p in D E; do
    name="rr-$(tr DE de <<< "$p")-gfx11"
    D4R_RR_GFX11=1 bash "$ROOT/engine/rr/build_native_model.sh" --dll "$DLL" "$p" "$WORK/$name" > "$WORK/$name.log" 2>&1 ||
        { tail -20 "$WORK/$name.log" >&2; exit 1; }
    python3 "$ROOT/packaging/model_tools.py" rr-hip-signature "$WORK/$name" > "$WORK/$name.sig"
    for arch in ${D4R_GPU_ARCHS:-gfx1100 gfx1101 gfx1102 gfx1103}; do
        [[ "$arch" == gfx11[05][0-3] ]] || { echo "$arch: the RR HIP stages are RDNA 3 / 3.5 only" >&2; exit 2; }
        D4R_GPU_ARCH="$arch" D4R_ROCM_RUNTIME='' bash "$ROOT/engine/rr/hip/build.sh" "$WORK/$name" "$OUT/$arch/$name" > /dev/null
        cp "$WORK/$name.sig" "$OUT/$arch/$name/rr-hip.txt"
        echo "$arch/$name: $(ls "$OUT/$arch/$name"/*.hsaco | wc -l) stages"
    done
done
