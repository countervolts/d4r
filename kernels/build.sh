#!/usr/bin/env bash
# Builds the native replacements for DLSS kernels into one directory that ZLUDA serves them from
# (D4R_ZLUDA_NATIVE_DIR, or NativeKernelDirFast in d4r.ini). See docs/native-kernels.md.
#
# usage: kernels/build.sh [all|k|l|m|tex] [OUT_DIR]      (default: all, kernels/out/native)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHAT="${1:-all}"
ACCURACY="${D4R_PREFER_ACCURACY:-0}"
[[ "$ACCURACY" == 0 || "$ACCURACY" == 1 ]] || { echo "D4R_PREFER_ACCURACY must be 0 or 1" >&2; exit 2; }
DEFAULT_OUT="$HERE/out/native"
[[ "$ACCURACY" == 1 ]] && DEFAULT_OUT+="/accuracy"
OUT="$(realpath -m "${2:-$DEFAULT_OUT}")"
ROCM="${D4R_ROCM_DIR:-/opt/rocm/core}"
ARCH="${D4R_GPU_ARCH:-gfx1101}"
FP8="${D4R_NATIVE_FP8:-0}"

# Путь к HIP хедерам из твоего venv Triton
TRITON_HIP_INC="${D4R_HIP_INC:-/home/try/vramvault/lib/python3.14/site-packages/triton/backends/amd/include}"

case "$ARCH" in
    gfx11[0-9][0-9]) [[ "$FP8" == 1 ]] && { echo "D4R_NATIVE_FP8 needs an RDNA4 (gfx12) target" >&2; exit 2; } ;;
    gfx12[0-9][0-9]) ;;
    *) echo "unsupported D4R_GPU_ARCH $ARCH (RDNA3 gfx110x or RDNA4 gfx120x)" >&2; exit 2 ;;
esac

if [[ -x "$ROCM/lib/llvm/bin/clang++" ]]; then
    CLANG="$ROCM/lib/llvm/bin/clang++"
elif [[ -x "$ROCM/llvm/bin/clang++" ]]; then
    CLANG="$ROCM/llvm/bin/clang++"
elif [[ -x "$ROCM/bin/clang++" ]]; then
    CLANG="$ROCM/bin/clang++"
else
    CLANG="$(which clang++)"
fi
[[ -x "$CLANG" ]] || { echo "clang++ not found (set D4R_ROCM_DIR)" >&2; exit 2; }

mkdir -p "$OUT"
case "$WHAT" in all|k|l|m|rr|tex) ;; *) echo "unknown kernel family: $WHAT" >&2; exit 2 ;; esac

# Never certify a directory containing older fast binaries as an accuracy set.
if [[ "$ACCURACY" == 1 ]] && compgen -G "$OUT/*.hsaco" >/dev/null &&
    { [[ ! -f "$OUT/d4r-accuracy.txt" ]] || [[ "$(cat "$OUT/d4r-accuracy.txt")" != 1 ]]; }; then
    echo "accuracy kernels need an empty output directory or an existing accuracy set: $OUT" >&2
    exit 2
fi
rm -f "$OUT/d4r-accuracy.txt"

BITCODE="${ROCM_DEVICE_LIB_PATH:-$ROCM/lib/llvm/amdgcn/bitcode}"
[[ -d "$BITCODE" ]] || BITCODE="$ROCM/amdgcn/bitcode"

# HIP source -> raw code object (the kernel name inside matches the DLSS kernel it replaces)
build_hip() {
    local src="$1" extra="${2:-}" name
    name="$(basename "$src" .hip)"
    local tmp flags
    tmp="$(mktemp -d)"
    # per-kernel compiler flags from a "// d4r-build-flags: ..." line in the source
    flags="$(sed -n 's|^// d4r-build-flags: *||p' "$src")"
    [[ "$ACCURACY" == 1 ]] && extra+=" -DD4R_ACCURACY"
    # shellcheck disable=SC2086
    (cd "$tmp" && "$CLANG" -x hip --offload-arch="$ARCH" --offload-device-only -O3 $flags $extra \
        -I"$TRITON_HIP_INC" \
        -I"$ROCM/include" \
        --rocm-path="$ROCM" \
        --rocm-device-lib-path="$BITCODE" \
        -I"$(dirname "$src")" \
        -o "$OUT/$name.hsaco" "$src" -save-temps=cwd --no-gpu-bundle-output)
    printf '%-52s %s\n' "$name" "$(grep -hE '^; (NumVgprs|ScratchSize|Occupancy)' "$tmp"/*.s | tail -3 | paste -sd' ')"
    rm -rf "$tmp"
}

if [[ "$WHAT" == all || "$WHAT" == k ]]; then
    echo "== DLSS 4 (preset K) transformer layers"
    for src in "$HERE"/k/dltss_pwin_*.hip; do build_hip "$src"; done
fi

if [[ "$WHAT" == all || "$WHAT" == m || "$WHAT" == l ]]; then
    echo "== DLSS 4.5 (presets L/M) shared Swin layers"
    for src in "$HERE"/m/rrlite_*.hip; do build_hip "$src" "$([[ "$FP8" == 1 ]] && echo -DD4R_FP8_WMMA)"; done
fi

if [[ "$WHAT" = all || "$WHAT" = rr ]]; then
    echo "== DLSS Ray Reconstruction (presets D, E, F) kernels"
    for src in "$HERE"/rr/*.hip; do
        [[ -f "$src" ]] && build_hip "$src"
    done
fi

if [[ "$WHAT" == all || "$WHAT" == tex || "$WHAT" == l ]]; then
    echo "== texture kernels (NVIDIA PTX with native parts)"
    : "${D4R_DLSS_DLL:?set D4R_DLSS_DLL to nvngx_dlss.dll}"
    : "${D4R_ZLUDA_EMIT:?set D4R_ZLUDA_EMIT to d4r_emit from a ZLUDA build with patches/zluda}"
    PTX_DIR="${D4R_DLSS_PTX_DIR:-$HERE/extracted/ptx}"
    if [[ -z "${D4R_DLSS_PTX_DIR:-}" ]]; then
        python3 "$HERE/tools/extract_dlss_ptx.py" "$D4R_DLSS_DLL" "$PTX_DIR"
    elif [[ ! -d "$PTX_DIR" ]]; then
        echo "missing extracted PTX directory: $PTX_DIR" >&2; exit 2
    fi

    specs=()
    if [[ "$WHAT" != l ]]; then
        specs+=(rrlite_dec0_4x4_folded:dec0_head)
        for mv in mvhi mvlo; do
            for range in hdr ldr; do
                specs+=("rrlite_enc0_4x4_${mv}_${range}_folded:enc0_tail")
                for v in 3_1 3_2; do specs+=("rrlite_post_${v}_${mv}_${range}_folded:sust_only:w64"); done
                for depth in depthinv depthreg; do
                    for kind in "" _max; do
                        [[ -z "$kind" ]] && specs+=("hiluma_engine_output_${depth}_${mv}_${range}_v1_rel:sust_only:w64")
                        specs+=("hiluma_engine_output_${depth}_${mv}_${range}${kind}_v2_rel:sust_only:w64")
                    done
                done
            done
        done
    fi
    specs+=(rrlite_dec0_4x4:sust_only)
    for mv in mvhi mvlo; do
        for range in hdr ldr; do
            specs+=("rrlite_enc0_4x4_${mv}_${range}:sust_only")
            for v in 3_1 3_2; do specs+=("rrlite_post_${v}_${mv}_${range}:sust_only:w64"); done
        done
    done
    for mode in static dynamic; do
        for range in hdr ldr; do specs+=("rrlite_downsample_kernel_${mode}_${range}:sust_only"); done
    done
    for spec in "${specs[@]}"; do
        IFS=: read -r kernel src mode <<< "$spec"
        D4R_PREFER_ACCURACY="$ACCURACY" \
            D4R_ZLUDA_WAVE64="$([[ "$mode" == w64 && "$ACCURACY" == 0 ]] && echo 1 || echo 0)" D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$ARCH" \
            D4R_TEX_FP8="$FP8" D4R_DLSS_PTX_DIR="$PTX_DIR" "$HERE/tex/build_tex.sh" "$kernel" "$src" "$OUT"
    done
fi
[[ "$ACCURACY" == 1 ]] && printf '1\n' > "$OUT/d4r-accuracy.txt"
echo "native kernels in $OUT"
