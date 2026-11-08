#!/usr/bin/env bash
# Builds the HIP forms of a Ray Reconstruction model's Swin stages (engine/rr/hip/rr_swin.hip) for an RDNA3 target,
# from the model's own build (engine/rr/build_native_model.sh with D4R_RR_GFX11=1 leaves each stage's swin_stage.h in
# MODEL/native-work/STAGE) and its weight files (W_WIDE is recovered from the widened file's size).
#
# usage: engine/rr/hip/build.sh MODEL_DIR OUT_DIR [STAGE...]   (default: every Swin stage)
#   D4R_ROCM_DIR  ROCm with clang and the device libraries (default /opt/rocm)
#   D4R_GPU_ARCH  default gfx1101
# Also builds OUT_DIR/rr_hip_test (engine/rr/hip/rr_hip_test.cpp) when D4R_ROCM_RUNTIME (a lib directory with
# libamdhip64) is set.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$(realpath "${1:?usage: build.sh MODEL_DIR OUT_DIR [STAGE...]}")"
OUT="$(realpath -m "${2:?usage: build.sh MODEL_DIR OUT_DIR [STAGE...]}")"
shift 2
STAGES=("$@"); (( ${#STAGES[@]} )) || STAGES=(enc0 enc1 enc2 enc3 enc4 enc5 dec4 dec3 dec2 dec1 dec0)
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
ARCH="${D4R_GPU_ARCH:-gfx1101}"
mkdir -p "$OUT"
# W_WIDE: the weight file is the original image padded to 512 bytes, then two bytes per original byte
wide() {
    python3 - "$1" <<'EOF'
import os, sys
total = os.path.getsize(sys.argv[1])
for base in range(0, total + 1, 512):
    n = (total - base) // 2
    if (total - base) % 2 == 0 and (n + 511) // 512 * 512 == base:
        print(base // 2); break
else:
    sys.exit("cannot recover W_WIDE from " + sys.argv[1])
EOF
}
PRESET="$(awk '$1 == "preset" { print $2 }' "$MODEL/rr.txt")"
for s in "${STAGES[@]}"; do
    if [[ "$s" == enc0 ]]; then   # encoder 0's native rest (rr_enc0.hip); its gather is MODEL/enc0_gather.spv
        ww="$(wide "$MODEL/weights-enc0m.bin")"
        "$ROCM/llvm/bin/clang++" -x hip --offload-arch="$ARCH" --offload-device-only --no-gpu-bundle-output --rocm-path="$ROCM" -nogpuinc \
            -O3 -std=c++17 -ffp-contract=off -I"$MODEL/native-work/enc0" -I"$HERE" -DW_WIDE="${ww}u" ${D4R_RR_HIP_CFLAGS:-} \
            -o "$OUT/enc0.hsaco" "$HERE/rr_enc0.hip"
        echo "enc0: $OUT/enc0.hsaco (W_WIDE $ww, rr_enc0.hip)"
        continue
    fi
    if [[ "$s" == dec0 ]]; then   # decoder 0: its own kernel (rr_dec0.hip) and header
        ww="$(wide "$MODEL/weights-dec0m.bin")"
        defs=(); [[ "$PRESET" == 4 ]] && defs=(-DRR_PRESET_D)
        "$ROCM/llvm/bin/clang++" -x hip --offload-arch="$ARCH" --offload-device-only --no-gpu-bundle-output --rocm-path="$ROCM" -nogpuinc \
            -O3 -std=c++17 -ffp-contract=off -I"$MODEL/native-work/dec0" -I"$HERE" -DW_WIDE="${ww}u" "${defs[@]}" ${D4R_RR_HIP_CFLAGS:-} \
            -o "$OUT/dec0.hsaco" "$HERE/rr_dec0.hip"
        echo "dec0: $OUT/dec0.hsaco (W_WIDE $ww, rr_dec0.hip, preset $PRESET)"
        continue
    fi
    hdr="$MODEL/native-work/$s/swin_stage.h"
    [[ -f "$hdr" ]] || { echo "no $hdr (build the model with engine/rr/build_native_model.sh, D4R_RR_GFX11=1)" >&2; exit 1; }
    ww="$(wide "$MODEL/weights-${s}m.bin")"
    # D4R_RR_HIP_WIDE: the stages built from the sixteen-wave form (rr_swin_wide.hip, kernel rr_swin_w16)
    src="$HERE/rr_swin.hip"
    [[ " ${D4R_RR_HIP_WIDE:-enc4 enc5 dec4 dec3} " == *" $s "* ]] && src="$HERE/rr_swin_wide.hip"
    # D4R_RR_HIP_ROLLED: the four-wave stages built with rolled loops (rr_swin.hip RR_ROLLED)
    rolled=0; [[ " ${D4R_RR_HIP_ROLLED:-enc3 dec2} " == *" $s "* ]] && rolled=1
    "$ROCM/llvm/bin/clang++" -x hip --offload-arch="$ARCH" --offload-device-only --no-gpu-bundle-output --rocm-path="$ROCM" -nogpuinc \
        -O3 -std=c++17 -ffp-contract=off -I"$MODEL/native-work/$s" -I"$HERE" -DW_WIDE="${ww}u" -DRR_ROLLED=$rolled ${D4R_RR_HIP_CFLAGS:-} \
        -o "$OUT/$s.hsaco" "$src"
    echo "$s: $OUT/$s.hsaco (W_WIDE $ww, $(basename "$src"))"
done
if [[ -n "${D4R_ROCM_RUNTIME:-}" ]]; then
    g++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -I"$ROCM/include" "$HERE/rr_hip_test.cpp" \
        "$(ls "$D4R_ROCM_RUNTIME"/libamdhip64.so* | head -1)" -Wl,-rpath,"$D4R_ROCM_RUNTIME" -o "$OUT/rr_hip_test"
    echo "$OUT/rr_hip_test"
fi
