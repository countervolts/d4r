#!/usr/bin/env bash
set -euo pipefail

# Builds a Ray Reconstruction model directory whose network stages are the native shaders (engine/rr/*_m.comp),
# for the preset the capture was taken with (D or E).
#
# usage: build_native_model.sh CAPTURE_DIR OUT_DIR [NVNGX_DLSSD_DLL]
#        build_native_model.sh --dll NVNGX_DLSSD_DLL D|E OUT_DIR
#
# With --dll no capture is needed: everything comes from the DLL (its PTX: kernels/tools/extract_dlss_ptx.py; the
# argument-block templates: engine/rr/rr_templates.py; the weights: dll_weights.py, model 0 = D, model 1 = E in 310.7's
# file order) and step 3's checks against a captured frame are skipped. This is what install.sh runs.
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
USAGE="usage: build_native_model.sh CAPTURE_DIR OUT_DIR [NVNGX_DLSSD_DLL] | --dll NVNGX_DLSSD_DLL D|E OUT_DIR"
CAP=""
if [[ "${1:-}" == --dll ]]; then
    DLL="$(realpath "${2:?$USAGE}")"; WANT="${3:?$USAGE}"; OUT="$(realpath -m "${4:?$USAGE}")"
    [[ "$WANT" == D || "$WANT" == E ]] || { echo "$USAGE" >&2; exit 2; }
else
    CAP="$(realpath "${1:?$USAGE}")"
    OUT="$(realpath -m "${2:?$USAGE}")"
    DLL="${3:-${D4R_RR_DLL:-$CAP/run/app/nvngx_dlssd.dll}}"
fi
RR="$ROOT/engine/rr"
WORK="$OUT/native-work"
[[ -f "$DLL" ]] || { echo "no nvngx_dlssd.dll at $DLL: pass the DLL the capture was taken with" >&2; exit 1; }
FRAME="${D4R_RR_MODEL_FRAME:-2}"
# each GPU / compiler job in a memory-capped scope where systemd provides one (D4R_NO_SCOPE=1: none)
CAPPED=()
if [[ "${D4R_NO_SCOPE:-0}" != 1 ]] && command -v systemd-run > /dev/null && systemd-run --user --scope -q true 2> /dev/null; then
    CAPPED=(systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0)
fi
mkdir -p "$OUT" "$WORK"
cd "$ROOT"

if [[ -n "$CAP" ]]; then
    python3 engine/compile_rr.py "$CAP" "$OUT" > "$WORK/compile_rr.log" 2>&1 || { tail -5 "$WORK/compile_rr.log"; exit 1; }
    python3 "$RR/dll_weights.py" "$DLL" "$WORK/dll" --capture "$CAP" ${D4R_RR_ANY_KERNELS:+--any-kernels}
else
    python3 kernels/tools/extract_dlss_ptx.py "$DLL" "$WORK/ptx" > /dev/null
    python3 engine/compile_rr.py --dll "$DLL" --preset "$WANT" "$OUT" --ptx-dir "$WORK/ptx" --network-bindings-only \
        > "$WORK/compile_rr.log" 2>&1 || { tail -5 "$WORK/compile_rr.log"; exit 1; }
    python3 "$RR/dll_weights.py" "$DLL" "$WORK/dll" --model "$([[ "$WANT" == D ]] && echo 0 || echo 1)" \
        ${D4R_RR_ANY_KERNELS:+--any-kernels} > /dev/null
fi
PRESET="$(awk '$1 == "preset" { print $2 }' "$OUT/rr.txt")"
# RR_EXACT_CONVERT: the compiled module's float conversions are decorated NoContraction (spv_exact_convert.py), which
# keeps the driver from folding a binary16 rounding away without the opaque OR the shaders otherwise need
DEFS=(-DRR_EXACT_CONVERT); [[ "$PRESET" == 4 ]] && DEFS+=(-DRR_PRESET_D)
# D4R_RR_RELAXED=1: one rounding where NVIDIA has two (engine/rr/rr_exact.glsl): faster, not bit-exact
# D4R_RR_RELAXED=3 adds RR_NATIVE: the MLP activation in f32 straight into FP8, no activation table
# Level 3 is the default (the fastest build; about 1.6 dB further from an NVIDIA card for D / E than the exact one);
# D4R_RR_RELAXED=0 builds NVIDIA's roundings exactly (bit-identical to the CUDA path through ZLUDA).
D4R_RR_RELAXED="${D4R_RR_RELAXED:-3}"
if (( D4R_RR_RELAXED > 0 )); then
    DEFS+=("-DRR_RELAXED=$(( D4R_RR_RELAXED > 2 ? 2 : D4R_RR_RELAXED ))"); (( D4R_RR_RELAXED >= 3 )) && DEFS+=(-DRR_NATIVE)
fi
# D4R_RR_MIX=1: the 32-deep steps round once through v_fma_mix (the shaders' K32): ~5% faster, not provably bit-exact
[[ "${D4R_RR_MIX:-0}" == 1 ]] && DEFS+=(-DRR_MIX)
# D4R_RR_GFX11=1: RDNA3 (no FP8 matrices, no VK_EXT_shader_float8): the stages reproduce ZLUDA's gfx11 lowering bit for
# bit (engine/rr/rr_gfx11.glsl); exact arithmetic only, and each native weight file carries a widened copy (W_WIDE)
GFX11=0
if [[ "${D4R_RR_GFX11:-0}" == 1 ]]; then
    GFX11=1; DEFS=(-DRR_EXACT_CONVERT -DRR_GFX11); [[ "$PRESET" == 4 ]] && DEFS+=(-DRR_PRESET_D)
    (( D4R_RR_RELAXED == 0 )) && [[ "${D4R_RR_MIX:-0}" != 1 ]] ||
        echo "D4R_RR_GFX11: relaxed / mix builds do not exist for RDNA3; building the exact model" >&2
    D4R_RR_RELAXED=0
fi
# The gfx11 runtime also reproduces NGX's CPU camera arithmetic, whose residues feed temporal specular motion.
(( GFX11 == 0 )) || sed -i '/^preset /a arithmetic gfx11' "$OUT/rr.txt"
# widen WEIGHTS: on RDNA3, append the widened copy and print -DW_WIDE=...; nothing otherwise
widen() { (( GFX11 )) && printf -- '-DW_WIDE=%su' "$(python3 "$RR/widen_weights.py" "$1")" || true; }
glsl() { # glsl OUT.spv SOURCE INCLUDE_DIR... [-DEXTRA]
    local out="$1" src="$2"; shift 2
    local dir inc=() extra=(); for dir in "$@"; do [[ "$dir" == -D* ]] && extra+=("$dir") || inc+=("-I$dir"); done
    "${CAPPED[@]}" glslangValidator --target-env vulkan1.3 -V "${DEFS[@]}" "${extra[@]}" "${inc[@]}" -o "$out" "$src" > "$WORK/glsl.log" 2>&1 ||
        { grep -E 'ERROR' "$WORK/glsl.log" | head -5; exit 1; }
    # from level 2 on the conversions are left to the driver (as compile_rr2.py does for preset F)
    (( D4R_RR_RELAXED >= 2 )) || python3 "$RR/spv_exact_convert.py" "$out" > /dev/null
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
    [[ -z "$CAP" ]] || python3 "$RR/swin_ref.py" "$CAP" "$s" "$FRAME" "$d/weights.npz" --blocks "$b" 0,0 | tail -2 | sed "s/^/$s: /"
    python3 "$RR/swin_pack.py" "$s" "$d/weights.npz" "$OUT/weights-${s}m.bin" "$d/swin_stage.h" > /dev/null
    glsl "$OUT/${s}_m.spv" "$RR/swin_m.comp" "$d" "$RR" $(widen "$OUT/weights-${s}m.bin")
    native "$s" "weights-${s}m.bin"
done

# decoder 0
d="$WORK/dec0"; mkdir -p "$d"
cp "$WORK/dll/dec0.npz" "$d/weights.npz"
[[ -z "$CAP" ]] || python3 "$RR/dec0_ref.py" "$CAP" "$FRAME" "$d/weights.npz" --blocks 5,5 0,0 | tail -2 | sed 's/^/dec0: /'
python3 "$RR/dec0_pack.py" "$d/weights.npz" "$OUT/weights-dec0m.bin" "$d/dec0_m_w.h" > /dev/null
glsl "$OUT/dec0_m.spv" "$RR/dec0_m.comp" "$d" "$RR" $(widen "$OUT/weights-dec0m.bin")
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
glsl "$OUT/enc0_m.spv" "$d/enc0_m.comp" "$d" "$RR" $(widen "$OUT/weights-enc0m.bin")
native enc0 weights-enc0m.bin

# the translated output stage (preset D; preset E's when engine/rr/hkpn.comp is not used) has no subgroup operations
if grep -q '"shader": "generated from the DLL"' <(python3 -c "import json,sys; print(json.dumps(json.load(open('$OUT/model.json'))['stages']['hkpn']))"); then
    sed -i -E 's/^stage hkpn hkpn\.spv 32 /stage hkpn hkpn.spv 64 /' "$OUT/rr.txt"
fi
for s in enc0 enc1 enc2 enc3 enc4 enc5 dec4 dec3 dec2 dec1 dec0; do rm -f "$OUT/weights-$s.bin"; done
# test_layer.py must bind optional images as well as the ones present in a capture. Keep a sidecar for every used
# shader, including native stages, so an absent input never shifts the shader's descriptor bindings.
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for line in (out / 'rr.txt').read_text().splitlines():
    fields = line.split()
    if not fields or fields[0] != 'stage':
        continue
    textures = [] if fields[5] == '-' else list(map(int, fields[5].split(',')))
    surfaces = [] if fields[6] == '-' else [[int(s.split(':')[0]), s.split(':')[1]] for s in fields[6].split(',')]
    (out / fields[2]).with_suffix('.json').write_text(json.dumps({'textures': textures, 'surfaces': surfaces}) + '\n')
PY
printf 'Native Ray Reconstruction model (preset %s) in %s\n' "$([[ "$PRESET" == 4 ]] && echo D || echo E)" "$OUT"
grep '^stage' "$OUT/rr.txt" | awk '{ printf "  %s %s (%s-wide)\n", $2, $3, $4 }'
