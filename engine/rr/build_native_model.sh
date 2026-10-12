#!/usr/bin/env bash
set -euo pipefail

# Builds a Ray Reconstruction model directory whose network stages are the native shaders (engine/rr/*_m.comp),
# for the preset the capture was taken with (D or E).
#
# usage: build_native_model.sh CAPTURE_DIR OUT_DIR [NVNGX_DLSSD_DLL]
#
# CAPTURE_DIR is a bridge capture of a few denoiser frames (engine/rr/rr_capture.py); the DLL is the one it was taken
# with (default: CAPTURE_DIR/run/app/nvngx_dlssd.dll, then $D4R_RR_DLL). The steps:
#   1. engine/compile_rr.py: the translated model (the argument-block templates, the exposure passes, the translated
#      output stage and the translated encoder 0 whose feature gather the native one keeps)
#   2. engine/rr/dll_weights.py: the network's weights, read from the DLL itself. The capture only selects which of
#      the DLL's models it ran, and a DLL whose kernels are not the ones the native shaders were written against is
#      refused (D4R_RR_ANY_KERNELS=1 builds anyway)
#   3. per stage: the weights checked against the capture's frame (STAGE_ref.py), packed (STAGE_pack.py) and the
#      native shader compiled against that packing's header
#   4. rr.txt rewritten to the native stages; the weight images NGX uploaded in the capture are removed
# Preset D differs from E in three kernels: encoder 0 (its feature gather and 30 input features for E's 32), decoder 0
# (the eight bytes it hands the output stage: -DRR_PRESET_D) and the output stage, which is the translated kernel
# (it has no subgroup operations, so it runs with 64-wide waves).
# Everything written is derived from the user's own nvngx_dlssd.dll and stays on their machine.
#
# GPU jobs run one at a time, each in a memory-capped scope (a translated stage's pipeline compile can take tens of GB).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAP="$(realpath "${1:?usage: build_native_model.sh CAPTURE_DIR OUT_DIR}")"
OUT="$(realpath -m "${2:?usage: build_native_model.sh CAPTURE_DIR OUT_DIR}")"
RR="$ROOT/engine/rr"
WORK="$OUT/native-work"
DLL="${3:-${D4R_RR_DLL:-$CAP/run/app/nvngx_dlssd.dll}}"
[[ -f "$DLL" ]] || { echo "no nvngx_dlssd.dll at $DLL: pass the DLL the capture was taken with" >&2; exit 1; }
FRAME="${D4R_RR_MODEL_FRAME:-2}"
CAPPED=(systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0)
mkdir -p "$OUT" "$WORK"
cd "$ROOT"

python3 engine/compile_rr.py "$CAP" "$OUT" > "$WORK/compile_rr.log" 2>&1 || { tail -5 "$WORK/compile_rr.log"; exit 1; }
python3 "$RR/dll_weights.py" "$DLL" "$WORK/dll" --capture "$CAP" ${D4R_RR_ANY_KERNELS:+--any-kernels}
PRESET="$(awk '$1 == "preset" { print $2 }' "$OUT/rr.txt")"
# RR_EXACT_CONVERT: the compiled module's float conversions are decorated NoContraction (spv_exact_convert.py), which
# keeps the driver from folding a binary16 rounding away without the opaque OR the shaders otherwise need
DEFS=(-DRR_EXACT_CONVERT); [[ "$PRESET" == 4 ]] && DEFS+=(-DRR_PRESET_D)
# D4R_RR_RELAXED=1: one rounding where NVIDIA has two (engine/rr/rr_exact.glsl): faster, not bit-exact
[[ -n "${D4R_RR_RELAXED:-}" ]] && DEFS+=("-DRR_RELAXED=$D4R_RR_RELAXED")
glsl() { # glsl OUT.spv SOURCE INCLUDE_DIR...
    local out="$1" src="$2"; shift 2
    local dir inc=(); for dir in "$@"; do inc+=("-I$dir"); done
    "${CAPPED[@]}" glslangValidator --target-env vulkan1.3 -V "${DEFS[@]}" "${inc[@]}" -o "$out" "$src" > "$WORK/glsl.log" 2>&1 ||
        { grep -E 'ERROR' "$WORK/glsl.log" | head -5; exit 1; }
    python3 "$RR/spv_exact_convert.py" "$out" > /dev/null
}
native() { # native STAGE WEIGHT_FILE: point rr.txt's stage line at STAGE_m.spv
    sed -i -E "s/^stage $1 $1\.spv 32 weights-$1\.bin (.*)/stage $1 $1_m.spv 32 $2 \1/; /^stage $1 /s/ noopt\$//" "$OUT/rr.txt"
    grep -q "^stage $1 $1_m.spv " "$OUT/rr.txt" || { echo "rr.txt: no stage line for $1" >&2; exit 1; }
}

# the Swin stages shared by both presets (engine/rr/swin_m.comp)
for s in enc1 enc2 enc3 enc4 enc5 dec4 dec3 dec2 dec1; do
    b=1,1; case $s in enc1|enc2|dec1|dec2) b=5,5 ;; esac
    d="$WORK/$s"; mkdir -p "$d"
    cp "$WORK/dll/$s.npz" "$d/weights.npz"
    python3 "$RR/swin_ref.py" "$CAP" "$s" "$FRAME" "$d/weights.npz" --blocks "$b" 0,0 | tail -2 | sed "s/^/$s: /"
    python3 "$RR/swin_pack.py" "$s" "$d/weights.npz" "$OUT/weights-${s}m.bin" "$d/swin_stage.h" > /dev/null
    glsl "$OUT/${s}_m.spv" "$RR/swin_m.comp" "$d" "$RR"
    native "$s" "weights-${s}m.bin"
done

# decoder 0
d="$WORK/dec0"; mkdir -p "$d"
cp "$WORK/dll/dec0.npz" "$d/weights.npz"
python3 "$RR/dec0_ref.py" "$CAP" "$FRAME" "$d/weights.npz" --blocks 5,5 0,0 | tail -2 | sed 's/^/dec0: /'
python3 "$RR/dec0_pack.py" "$d/weights.npz" "$OUT/weights-dec0m.bin" "$d/dec0_m_w.h" > /dev/null
glsl "$OUT/dec0_m.spv" "$RR/dec0_m.comp" "$d" "$RR"
native dec0 weights-dec0m.bin

# encoder 0: the translated feature gather + the native rest
d="$WORK/enc0"; mkdir -p "$d"
cp "$WORK/dll/enc0.npz" "$d/weights.npz"
# preset E's gather dithers with the noise table that starts NGX's weight image: the table, then the packing
BASE=0; [[ "$PRESET" == 4 ]] || BASE=4096
python3 "$RR/enc0_pack.py" "$d/weights.npz" "$d/enc0_m.bin" "$d/enc0_m_w.h" "$BASE" > /dev/null
if [[ "$BASE" != 0 ]]; then
    [[ "$(stat -c %s "$WORK/dll/noise.bin" 2>/dev/null)" == 4096 ]] || { echo "no noise table found in $DLL" >&2; exit 1; }
    cat "$WORK/dll/noise.bin" "$d/enc0_m.bin" > "$OUT/weights-enc0m.bin"
else
    cp "$d/enc0_m.bin" "$OUT/weights-enc0m.bin"
fi
python3 "$RR/gen_enc0_m.py" "$OUT/enc0.generated.comp" "$d/enc0_m.comp" > /dev/null
glsl "$OUT/enc0_m.spv" "$d/enc0_m.comp" "$d" "$RR"
native enc0 weights-enc0m.bin

# the translated output stage (preset D; preset E's when engine/rr/hkpn.comp is not used) has no subgroup operations
if grep -q '"shader": "generated from the DLL"' <(python3 -c "import json,sys; print(json.dumps(json.load(open('$OUT/model.json'))['stages']['hkpn']))"); then
    sed -i -E 's/^stage hkpn hkpn\.spv 32 /stage hkpn hkpn.spv 64 /' "$OUT/rr.txt"
fi
for s in enc0 enc1 enc2 enc3 enc4 enc5 dec4 dec3 dec2 dec1 dec0; do rm -f "$OUT/weights-$s.bin"; done
printf 'Native Ray Reconstruction model (preset %s) in %s\n' "$([[ "$PRESET" == 4 ]] && echo D || echo E)" "$OUT"
grep '^stage' "$OUT/rr.txt" | awk '{ printf "  %s %s (%s-wide)\n", $2, $3, $4 }'
