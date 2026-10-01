#!/usr/bin/env bash
# Builds the native replacements for DLSS kernels into one directory that ZLUDA serves them from
# (D4R_ZLUDA_NATIVE_DIR, or NativeKernelDirFast in d4r.ini). See docs/native-kernels.md.
#
# usage: kernels/build.sh [all|k|m|tex] [OUT_DIR]      (default: all, kernels/out/native)
#
#   D4R_ROCM_DIR    ROCm installation with clang and the HIP device libraries (default /opt/rocm)
#   D4R_GPU_ARCH    target GPU (default gfx1101): an RDNA3 (gfx110x) or RDNA4 (gfx120x) target; the kernels
#                   need wave32 WMMA, and kernels/common/wmma_layout.h picks the register layout per target
#   D4R_NATIVE_FP8  1 (RDNA4 only): the variant for ZLUDA's native FP8 WMMA (d4r.ini [Kernels] NativeFp8):
#                   M's layers on FP8 WMMA with NVIDIA's e4m3 activations, texture tails matching ZLUDA's
#                   D4R_ZLUDA_WMMA_FP8_NATIVE lowering. K is the same in both variants.
#   D4R_PREFER_ACCURACY  1: preserve FP8 quantization, per-MMA f16 rounding and denormal handling;
#                   build a separate set (default output: kernels/out/native/accuracy).
# Texture kernels (tex) are NVIDIA's PTX with parts replaced, so they additionally need:
#   D4R_DLSS_DLL    nvngx_dlss.dll 310.7 (its PTX is extracted into kernels/extracted/, never committed)
#   D4R_ZLUDA_EMIT  ZLUDA's d4r_emit (patches/zluda applied; `cargo build --release -p ptx --example d4r_emit`),
#                   which compiles PTX offline for D4R_GPU_ARCH (no GPU of that kind needed)
#   D4R_DLSS_PTX_DIR optional pre-extracted PTX directory for parallel per-target builds
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHAT="${1:-all}"
ACCURACY="${D4R_PREFER_ACCURACY:-0}"
[[ "$ACCURACY" == 0 || "$ACCURACY" == 1 ]] || { echo "D4R_PREFER_ACCURACY must be 0 or 1" >&2; exit 2; }
DEFAULT_OUT="$HERE/out/native"
[[ "$ACCURACY" == 1 ]] && DEFAULT_OUT+="/accuracy"
OUT="$(realpath -m "${2:-$DEFAULT_OUT}")"
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
ARCH="${D4R_GPU_ARCH:-gfx1101}"
FP8="${D4R_NATIVE_FP8:-0}"
case "$ARCH" in
    gfx11[0-9][0-9]) [[ "$FP8" == 1 ]] && { echo "D4R_NATIVE_FP8 needs an RDNA4 (gfx12) target" >&2; exit 2; } ;;
    gfx12[0-9][0-9]) ;;
    *) echo "unsupported D4R_GPU_ARCH $ARCH (RDNA3 gfx110x or RDNA4 gfx120x)" >&2; exit 2 ;;
esac
CLANG="$ROCM/lib/llvm/bin/clang++"
[[ -x "$CLANG" ]] || { echo "clang++ not found in $ROCM/lib/llvm/bin (set D4R_ROCM_DIR)" >&2; exit 2; }
mkdir -p "$OUT"
case "$WHAT" in all|k|m|tex) ;; *) echo "unknown kernel family: $WHAT" >&2; exit 2 ;; esac
# Never certify a directory containing older fast binaries as an accuracy set.
if [[ "$ACCURACY" == 1 ]] && compgen -G "$OUT/*.hsaco" >/dev/null &&
    { [[ ! -f "$OUT/d4r-accuracy.txt" ]] || [[ "$(cat "$OUT/d4r-accuracy.txt")" != 1 ]]; }; then
    echo "accuracy kernels need an empty output directory or an existing accuracy set: $OUT" >&2
    exit 2
fi
rm -f "$OUT/d4r-accuracy.txt"

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
    (cd "$tmp" && "$CLANG" -x hip --offload-arch="$ARCH" --offload-device-only -O3 $flags $extra -I"$ROCM/include" \
        --rocm-path="$ROCM" --rocm-device-lib-path="$ROCM/amdgcn/bitcode" -I"$(dirname "$src")" \
        -o "$OUT/$name.hsaco" "$src" -save-temps=cwd --no-gpu-bundle-output)
    printf '%-52s %s\n' "$name" "$(grep -hE '^; (NumVgprs|ScratchSize|Occupancy)' "$tmp"/*.s | tail -3 | paste -sd' ')"
    rm -rf "$tmp"
}

if [[ "$WHAT" == all || "$WHAT" == k ]]; then
    echo "== DLSS 4 (preset K) transformer layers"
    for src in "$HERE"/k/dltss_pwin_*.hip; do build_hip "$src"; done
fi
if [[ "$WHAT" == all || "$WHAT" == m ]]; then
    echo "== DLSS 4.5 (preset M) Swin layers"
    for src in "$HERE"/m/rrlite_*.hip; do build_hip "$src" "$([[ "$FP8" == 1 ]] && echo -DD4R_FP8_WMMA)"; done
fi
if [[ "$WHAT" == all || "$WHAT" == tex ]]; then
    echo "== texture kernels (NVIDIA PTX with native parts)"
    : "${D4R_DLSS_DLL:?set D4R_DLSS_DLL to nvngx_dlss.dll (310.7)}"
    : "${D4R_ZLUDA_EMIT:?set D4R_ZLUDA_EMIT to d4r_emit from a ZLUDA build with patches/zluda}"
    PTX_DIR="${D4R_DLSS_PTX_DIR:-$HERE/extracted/ptx}"
    if [[ -z "${D4R_DLSS_PTX_DIR:-}" ]]; then
        python3 "$HERE/tools/extract_dlss_ptx.py" "$D4R_DLSS_DLL" "$PTX_DIR"
    elif [[ ! -d "$PTX_DIR" ]]; then
        echo "missing extracted PTX directory: $PTX_DIR" >&2; exit 2
    fi
    # kernel : source of its native part [: w64 = compiled as wave64 (two CUDA warps per wave; RDNA3 issues
    # FP32 work for all 64 lanes at once). Only for kernels without MMAs; measured faster, bit-identical:
    # post 0.89 -> 0.77 ms, hiluma output 0.905 -> 0.887 ms (downsample was slower as wave64).]
    # Every flag combination DLSS selects (motion vectors hi/lo, HDR/LDR, depth inverted/regular, ...) gets
    # the same treatment, so games other than the one measured run the native versions too.
    specs=(rrlite_dec0_4x4_folded:dec0_head)
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
