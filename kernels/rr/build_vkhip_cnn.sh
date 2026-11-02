#!/usr/bin/env bash
set -euo pipefail
# Builds the Ray Reconstruction CNN's network stages (presets A - C, engine/compile_rr_cnn.py's "D4RR0003" model) for
# vkhip: kernels/rr/rrcnn_vkhip.hip once per stage into MODEL_DIR/hip/rr3_<stage>.hsaco, which engine/vkhip/vkhip_net.h
# RRStages records in place of the engine's stage shaders. The weights stay the model's (the parameter blocks point at
# them).
#
# usage: build_vkhip_cnn.sh MODEL_DIR
# environment: D4R_ROCM_DIR (default /opt/rocm), D4R_GPU_ARCH (gfx1200 or gfx1201, default gfx1201),
#   D4R_RR3_SCHEDULE="Stage=TH,R,NTW ..." to override stages' schedules, D4R_RR3_STAGES="Stage ..." to build only these
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$(realpath "${1:?usage: build_vkhip_cnn.sh MODEL_DIR}")"
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
ARCH="${D4R_GPU_ARCH:-gfx1201}"
[[ "$ARCH" == gfx1200 || "$ARCH" == gfx1201 ]] || { echo "build_vkhip_cnn.sh: the stages need RDNA4 (gfx1200 / gfx1201) WMMA" >&2; exit 2; }
[[ "$(head -1 "$MODEL/rr.txt")" == D4RR0003 ]] || { echo "build_vkhip_cnn.sh: not a Ray Reconstruction CNN model (rr.txt D4RR0003)" >&2; exit 2; }
CLANG="$ROCM/lib/llvm/bin/clang++"
[[ -x "$CLANG" ]] || { echo "clang++ not found in $ROCM/lib/llvm/bin (set D4R_ROCM_DIR)" >&2; exit 2; }
# per stage: the variant (engine/compile_rr_cnn.py NETWORK) and the schedule TH,R,NTW: tile rows, rows and 16-channel
# tiles per wave (rrcnn_vkhip.hip), measured on an RX 9070 XT
declare -A VARIANT=(
    [Enc0]="-DCIN=16 -DC=32" [Enc1]="-DCIN=32 -DC=64 -DIN8 -DTS=8" [Enc2]="-DCIN=64 -DC=96 -DIN8 -DSPLITK"
    [Enc3]="-DCIN=96 -DC=128 -DIN8 -DSPLITK" [Enc4]="-DCIN=128 -DC=160 -DIN8 -DSPLITK"
    [Dec4]="-DRR3_DEC -DCIN=160 -DC=160 -DSPLITK" [Dec3]="-DRR3_DEC -DCIN=160 -DC=128 -DUPS -DSPLITK"
    [Dec2]="-DRR3_DEC -DCIN=128 -DC=96 -DUPS -DSPLITK" [Dec1]="-DRR3_DEC -DCIN=96 -DC=64 -DUPS -DTS=8"
    [Dec0]="-DRR3_DEC -DCIN=64 -DC=32 -DUPS" [PDL]="-DRR3_DEC -DCIN=32 -DC=32 -DUPS -DPDL")
# (1280x720 render; occupancy, which the tile's LDS bounds, matters more than wide waves)
declare -A SCHEDULE=(
    [Enc0]=8,2,2 [Enc1]=4,2,2 [Enc2]=4,2,1 [Enc3]=2,2,1 [Enc4]=2,2,1
    [Dec4]=2,2,1 [Dec3]=2,2,1 [Dec2]=4,2,2 [Dec1]=4,2,1 [Dec0]=8,2,1 [PDL]=4,2,1)
for kv in ${D4R_RR3_SCHEDULE:-}; do SCHEDULE[${kv%%=*}]="${kv#*=}"; done
mkdir -p "$MODEL/hip"
for s in ${D4R_RR3_STAGES:-Enc0 Enc1 Enc2 Enc3 Enc4 Dec4 Dec3 Dec2 Dec1 Dec0 PDL}; do
    IFS=, read -r TH R NTW <<< "${SCHEDULE[$s]}"
    name="rr3_${s,,}"
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2086
    (cd "$tmp" && "$CLANG" -x hip --offload-arch="$ARCH" --offload-device-only -O3 -mcumode --rocm-path="$ROCM" \
        ${VARIANT[$s]} -DTH="$TH" -DR="$R" -DNTW="$NTW" -DRR3_NAME="$name" ${D4R_RR3_FLAGS:-} \
        -o "$MODEL/hip/$name.hsaco" "$HERE/rrcnn_vkhip.hip" -save-temps=cwd --no-gpu-bundle-output)
    "$ROCM/lib/llvm/bin/llvm-readelf" --notes "$MODEL/hip/$name.hsaco" | grep -q '\.private_segment_fixed_size: *0$' ||
        { echo "$name spills to scratch" >&2; rm -rf "$tmp"; exit 1; }
    printf '%-5s %-7s %s\n' "$s" "${SCHEDULE[$s]}" "$(grep -hE '^; (NumVgprs|Occupancy|LDSByteSize)' "$tmp"/*.s | paste -sd' ')"
    rm -rf "$tmp"
done
