#!/usr/bin/env bash
# Opt-in HIP texture candidates for M/L. Does not modify installed/default sets.
# Usage: D4R_DLSS_DLL=... D4R_ROCM_DIR=... build_m_native_textures.sh OUT [KERNEL]
# Post/downsample retain the original arithmetic. Only folded M enc0/dec0 use
# native matrix tails; L's unfolded enc0/dec0 deliberately remain separate.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${1:?usage: build_m_native_textures.sh OUT [KERNEL]}")"
: "${D4R_DLSS_DLL:?set D4R_DLSS_DLL to the matching NVIDIA DLSS library}"
[[ "${D4R_PREFER_ACCURACY:-0}" == 0 ]] || { echo 'native M/L texture candidates require temporal accuracy validation before accuracy-set use' >&2; exit 2; }
mkdir -p "$OUT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PTX_DIR="${D4R_DLSS_PTX_DIR:-$WORK/ptx}"
if [[ -z "${D4R_DLSS_PTX_DIR:-}" ]]; then
    python3 "$ROOT/kernels/tools/extract_dlss_ptx.py" "$D4R_DLSS_DLL" "$PTX_DIR" >/dev/null
fi
names=()
if [[ -n "${2:-}" ]]; then names+=("$2"); else
    for mv in mvhi mvlo; do
        for range in hdr ldr; do
            names+=("rrlite_enc0_4x4_${mv}_${range}_folded")
            for v in 3_1 3_2; do
                names+=("rrlite_post_${v}_${mv}_${range}_folded" "rrlite_post_${v}_${mv}_${range}")
            done
        done
    done
    names+=(rrlite_dec0_4x4_folded)
    for mode in static dynamic; do
        for range in hdr ldr; do names+=("rrlite_downsample_kernel_${mode}_${range}"); done
    done
fi
for name in "${names[@]}"; do
    flags=(--as1)
    if [[ "$name" =~ ^rrlite_post_3_[12]_mv(hi|lo)_(hdr|ldr)(_folded)?$ ]]; then wave=64
    elif [[ "$name" =~ ^rrlite_downsample_kernel_(static|dynamic)_(hdr|ldr)$ ]]; then wave=32
    elif [[ "$name" =~ ^rrlite_enc0_4x4_mv(hi|lo)_(hdr|ldr)_folded$ || "$name" == rrlite_dec0_4x4_folded ]]; then
        wave=32; flags+=(--m-tail)
    else echo "unsupported M/L texture candidate: $name" >&2; exit 2
    fi
    [[ -z "${D4R_M_TEXTURE_MAX_VGPR:-}" ]] || flags+=("--max-vgpr=$D4R_M_TEXTURE_MAX_VGPR")
    mapfile -t modules < <(rg --no-ignore -l "\.entry[[:space:]]+$name[[:space:]]*\(" "$PTX_DIR"/*.ptx)
    [[ ${#modules[@]} -gt 0 ]] || { echo "no module for $name" >&2; exit 2; }
    ptx="${modules[0]}"
    for duplicate in "${modules[@]:1}"; do
        cmp -s "$ptx" "$duplicate" || { echo "different modules define $name; use a clean extraction directory" >&2; exit 2; }
    done
    python3 "$ROOT/kernels/native/ptx2hip.py" "${flags[@]}" "$ptx" "$name" > "$WORK/$name.hip"
    WAVE="${WAVE:-$wave}" bash "$ROOT/kernels/native/build_native.sh" "$WORK/$name.hip" "$OUT/$name.hsaco" \
        -DD4R_AS1_STORES "-DD4R_KERNEL_NAME=$name"
    echo 'experimental HIP translation; temporal validation required' > "$OUT/$name.hsaco.experimental.txt"
done
python3 "$ROOT/kernels/tools/kernel_manifest.py" "$OUT" "$D4R_DLSS_DLL"
