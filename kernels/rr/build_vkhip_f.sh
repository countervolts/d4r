#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$(realpath "${1:?usage: build_vkhip_f.sh MODEL_DIR}")"
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
ARCH="${D4R_GPU_ARCH:-gfx1201}"
[[ "$ARCH" == gfx1200 || "$ARCH" == gfx1201 ]] || { echo 'preset F vkhip requires RDNA4' >&2; exit 2; }
[[ "$(head -1 "$MODEL/rr.txt")" == D4RR0002 ]] || { echo 'preset F model required' >&2; exit 2; }
[[ -f "$MODEL/native-work/dec0_0/rr2_stage.h" ]] || { echo 'rebuild the F model with compile_rr2.py to retain native-work' >&2; exit 2; }
python3 - "$MODEL/dec0_0_m.build.json" <<'PYCODE'
import json, sys
args = json.load(open(sys.argv[1]))['argv']
if '-DRR_NATIVE' not in args or '-DRR_RELAXED=2' not in args:
    raise SystemExit('preset F vkhip currently requires the default relaxed=3 model')
PYCODE
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$MODEL/hip"
# Windows' ROCm keeps the device library under lib/llvm, where clang does not look with --rocm-path alone, and the
# MSVC <cmath> overloads it then compiles (isgreater & co., implicitly __host__ __device__ as constexpr) clash with
# clang's own __device__ declarations, which -fno-cuda-host-device-constexpr avoids
LIB=(); [[ -d "$ROCM/lib/llvm/amdgcn/bitcode" ]] && LIB+=(--rocm-device-lib-path="$ROCM/lib/llvm/amdgcn/bitcode")
case "$(uname -s)" in MSYS*|MINGW*|CYGWIN*) LIB+=(-Xclang -fno-cuda-host-device-constexpr) ;; esac
# Stages built from the engine's own stage shader (the model's <stage>_m.preprocessed.comp, translated by glsl2hip.py)
# instead of the native block: faster on gfx1201; same output as the native blocks. D4R_RRVK_TRANSLATED overrides the
# list (space-separated, empty for none; the rest are native blocks). Per stage, from sweeps
# (RX 9070 XT, 1280x720 render): the loops left rolled, the waves per SIMD LLVM aims for (0: its own choice), the arrays
# kept in the window's record instead of registers and the scheduler. Levels 3 - 5 have about as many windows as the
# GPU has compute units, so a window's latency is the stage's time; with r's 80 registers free (--record=rv) LLVM issues
# the weight loads ahead of their WMMAs (dec3_0 gets there with the memory-clause scheduler instead).
TRANSLATED="${D4R_RRVK_TRANSLATED-enc1 enc2 enc3 enc4 enc5 dec4_0 dec4_1 dec3_0 dec3_1 dec2_0 dec2_1 dec1_0 dec1_1 dec0_0}"
for STAGE in enc1 enc2 enc3 enc4 enc5 dec4_0 dec4_1 dec3_0 dec3_1 dec2_0 dec2_1 dec1_0 dec1_1 dec0_0; do
    if [[ " $TRANSLATED " == *" $STAGE "* ]]; then
        ROLLED=hd,br WAVES=5 RECORD='' SCHED=''
        case "$STAGE" in
            enc2) ROLLED=br WAVES=6 ;;
            dec2_0) ROLLED=none ;;
            enc3|enc4|enc5|dec4_1) ROLLED='hd,br,ks>2' WAVES=0 RECORD=rv ;;   # the merge's k loop rolled: unrolled, it spills
            dec4_0) ROLLED='hd,br,ks>2' WAVES=0 RECORD=rv,xr ;;              # and the skip input
            dec3_0) WAVES=0 SCHED=max-memory-clause ;;
            dec0_0) ROLLED=br WAVES=6 ;;                 # rr2_w.comp, one wave per window
        esac
        FLAGS=(); [[ "$WAVES" != 0 ]] && FLAGS+=(-DRR2_WAVES="$WAVES"); [[ -n "$SCHED" ]] && FLAGS+=(-mllvm -amdgpu-sched-strategy="$SCHED")
        python3 "$HERE/glsl2hip.py" --sat --rolled="$ROLLED" ${RECORD:+--record=$RECORD} --v-layout="$MODEL/native-work/$STAGE/rr2_stage.h" \
            "$MODEL/${STAGE}_m.preprocessed.comp" "rr2_$STAGE" "$TMP/rr2_$STAGE.hip"
        "$ROCM/lib/llvm/bin/clang++" -x hip --offload-arch="$ARCH" --offload-device-only -O3 -mcumode \
            -fno-hip-fp32-correctly-rounded-divide-sqrt ${D4R_RRVK_FLAGS:-} --rocm-path="$ROCM" ${LIB[@]+"${LIB[@]}"} -I"$HERE" "${FLAGS[@]}" \
            --no-gpu-bundle-output -o "$TMP/rr2_$STAGE.hsaco" "$TMP/rr2_$STAGE.hip"
        # no scratch: the Windows pipeline-binary path refuses kernels with private memory
        "$ROCM/lib/llvm/bin/llvm-readelf" --notes "$TMP/rr2_$STAGE.hsaco" | grep -q '\.private_segment_fixed_size: *0$' ||
            { echo "rr2_$STAGE spills to scratch" >&2; exit 1; }
        cp "$TMP/rr2_$STAGE.hsaco" "$MODEL/hip/rr2_$STAGE.hsaco"
        continue
    fi
    HS=2
    case "$STAGE" in enc3|enc5|dec4_0|dec3_0|dec3_1) HS=4 ;; dec0_0) HS=1 ;; esac
    HS="${D4R_RRVK_HS:-$HS}"
    # WGP mode (a workgroup's waves over both compute units' SIMDs) for the native level 4 and 5 blocks: latency-bound windows
    MODE=-mcumode; [[ "$STAGE" == enc4 || "$STAGE" == enc5 || "$STAGE" == dec4_* ]] && MODE=-mno-cumode
    "$ROCM/lib/llvm/bin/clang++" -x hip --offload-arch="$ARCH" --offload-device-only -O3 $MODE ${D4R_RRVK_FLAGS:-} --rocm-path="$ROCM" ${LIB[@]+"${LIB[@]}"} \
        -I"$MODEL/native-work/$STAGE" -DRRVK_NATIVE -DRRVK_HS="$HS" -DRRVK_SAT_MODE -DRRVK_NAME="rr2_$STAGE" \
        --no-gpu-bundle-output -o "$TMP/rr2_$STAGE.hsaco" "$HERE/rr2_vkhip.hip"
    cp "$TMP/rr2_$STAGE.hsaco" "$MODEL/hip/rr2_$STAGE.hsaco"
done
