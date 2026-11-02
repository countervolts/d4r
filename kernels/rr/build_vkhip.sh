#!/usr/bin/env bash
set -euo pipefail
# Builds the Ray Reconstruction (presets D and E) network stages for vkhip (kernels/rr/rrvk_swin.hip, run by
# engine/vkhip/vkhip_net.h RRStages) into MODEL_DIR/hip/rr_<stage>.hsaco: one code object per stage, compiled against
# the stage's weight layout (MODEL_DIR/native-work/<stage>/swin_stage.h, dec0_m_w.h or enc0_m_w.h, which
# engine/rr/build_native_model.sh keeps). Encoder 0 runs after its translated gather, MODEL_DIR/enc0_g.spv (built by
# build_native_model.sh), which the engine dispatches first.
#
# usage: build_vkhip.sh MODEL_DIR
# environment: D4R_ROCM_DIR (default /opt/rocm), D4R_GPU_ARCH (gfx1200 or gfx1201, default gfx1201)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$(realpath "${1:?usage: build_vkhip.sh MODEL_DIR}")"
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
ARCH="${D4R_GPU_ARCH:-gfx1201}"
[[ "$ARCH" == gfx1200 || "$ARCH" == gfx1201 ]] || { echo "build_vkhip.sh: the stages need RDNA4 (gfx1200 / gfx1201) FP8 WMMA" >&2; exit 2; }
CLANG="$ROCM/lib/llvm/bin/clang++"
[[ -x "$CLANG" ]] || { echo "clang++ not found in $ROCM/lib/llvm/bin (set D4R_ROCM_DIR)" >&2; exit 2; }
DEVICE_LIBS="$(dirname "$(find "$ROCM" -name ocml.bc -print -quit)")"
mkdir -p "$MODEL/hip"
# per stage: query tiles per weight load (RRVK_QB) and heads at a time (RRVK_HS), measured on an RX 9070 XT at
# 1280x720 (the wave count follows: rrvk_swin.hip's NW)
declare -A QB=([enc1]=2 [enc2]=4 [enc3]=4 [enc4]=4 [enc5]=2 [dec4]=4 [dec3]=4 [dec2]=2 [dec1]=2)
declare -A HS=([enc1]=2 [enc2]=2 [enc3]=2 [enc4]=4 [enc5]=4 [dec4]=4 [dec3]=4 [dec2]=2 [dec1]=2)
# decoder 0: the same kernel (-DRRVK_DEC0) against dec0_pack.py's header; preset D's codes differ (rr.txt preset 4)
QB[dec0]=2; HS[dec0]=1
# encoder 0: the same kernel (-DRRVK_ENC0) against enc0_pack.py's header
QB[enc0]=2; HS[enc0]=1
PRESET=()
[[ "$(awk '$1 == "preset" { print $2 }' "$MODEL/rr.txt")" == 4 ]] && PRESET=(-DRR_PRESET_D)
for s in enc0 enc1 enc2 enc3 enc4 enc5 dec4 dec3 dec2 dec1 dec0; do
    stage="$MODEL/native-work/$s"
    extra=()
    [[ "$s" == dec0 ]] && extra=(-DRRVK_DEC0 "${PRESET[@]}")
    if [[ "$s" == enc0 ]]; then
        [[ -f "$MODEL/enc0_g.spv" ]] || { echo "enc0: no $MODEL/enc0_g.spv (rebuild the model with engine/rr/build_native_model.sh); skipped" >&2; continue; }
        extra=(-DRRVK_ENC0)
    fi
    [[ -f "$stage/swin_stage.h" || -f "$stage/dec0_m_w.h" || -f "$stage/enc0_m_w.h" ]] || { echo "no $stage/swin_stage.h (build the model with engine/rr/build_native_model.sh)" >&2; exit 1; }
    tmp="$(mktemp -d)"
    (cd "$tmp" && "$CLANG" -x hip --offload-arch="$ARCH" --offload-device-only -O3 -I"$ROCM/include" \
        --rocm-path="$ROCM" --rocm-device-lib-path="$DEVICE_LIBS" -I"$stage" ${D4R_RRVK_FLAGS:-} -DRRVK_NAME="rr_$s" -DRRVK_QB="${QB[$s]}" -DRRVK_HS="${HS[$s]}" "${extra[@]}" \
        -o "$MODEL/hip/rr_$s.hsaco" "$HERE/rrvk_swin.hip" -save-temps=cwd --no-gpu-bundle-output)
    printf '%-6s %s\n' "$s" "$(grep -hE '^; (NumVgprs|ScratchSize|Occupancy|LDSByteSize)' "$tmp"/*.s | paste -sd' ')"
    rm -rf "$tmp"
done
