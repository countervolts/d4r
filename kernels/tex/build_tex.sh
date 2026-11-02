#!/usr/bin/env bash
# build_tex.sh KERNEL SRC OUT_DIR: one texture kernel. NVIDIA's PTX for KERNEL, edited by make_ptx.py so
# part of it calls functions from SRC.hip, is compiled by ZLUDA with SRC's bitcode linked in
# (D4R_ZLUDA_EXTRA_BC) and the resulting code object is saved as OUT_DIR/KERNEL.hsaco.
# Called by kernels/build.sh, which sets D4R_ROCM_DIR, D4R_GPU_ARCH, D4R_DLSS_PTX_DIR, D4R_ZLUDA_EMIT,
# D4R_ZLUDA_WAVE64 (1 = wave64 code object; accuracy builds only with D4R_TEX_ACCURACY_WAVE64=1) and
# D4R_TEX_FP8 (1 = the gfx12 native-FP8 variant).
# ZLUDA compiles offline for D4R_GPU_ARCH with its d4r_emit tool, so any target builds on any machine.
set -euo pipefail
K=$1 SRC=$2 OUT=$3
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="${D4R_ROCM_DIR:-/opt/rocm}"
ARCH="${D4R_GPU_ARCH:-gfx1101}"
FP8="${D4R_TEX_FP8:-0}"
ACCURACY="${D4R_PREFER_ACCURACY:-0}"
LLVM="$R/lib/llvm/bin"
B="$(mktemp -d)"
trap 'rm -rf "$B"' EXIT
# D4R_TEX_CFLAGS: extra flags for SRC.hip (tests, e.g. -DD4R_WMMA_LAYOUT=12 for the gfx12-layout shim on gfx11)
read -r -a FLAGS <<< "${D4R_TEX_CFLAGS:-}"
[[ "$FP8" == 1 ]] && FLAGS+=(-DD4R_TEX_FP8)
WAVES="${D4R_TEX_WAVES:-1}"
[[ "$WAVES" == 1 || "$WAVES" == 4 ]] || { echo 'D4R_TEX_WAVES must be 1 or 4' >&2; exit 2; }
if [[ "$WAVES" == 4 ]]; then
    [[ "$SRC" == enc0l_tail || "$SRC" == dec0l_block ]] || { echo 'parallel windows require unfolded L kernels' >&2; exit 2; }
    FLAGS+=(-DD4R_TEX_PARALLEL_WINDOWS)
fi
[[ "$ACCURACY" == 1 ]] && FLAGS+=(-DD4R_ACCURACY)
# SRC.hip -> bitcode without target attributes, so ZLUDA's own target settings apply after linking.
# D4R_LLVM_AS: the llvm-as of ZLUDA's LLVM, when ROCm's clang is newer (the Windows HIP SDK 7.14 is LLVM 23, ZLUDA
# reads LLVM 22 bitcode). The last three rewrites turn LLVM 23's IR into LLVM 22's (llvm23_constants.py its constants
# after them); it has none of them:
# denormal_fpenv(dynamic) is LLVM 22's "denormal-fp-math" pair, nocreateundeforpoison only an intrinsic hint, and
# target_mem a memory location LLVM 22 has no name for (dropping it widens the access to the function's default).
# The module summary (^N = ... lines, ThinLTO's index, which LLVM 23 gives a noRenameOnPromotion flag) is dropped too.
LLVM_AS="${D4R_LLVM_AS:-$LLVM/llvm-as}"
"$LLVM/clang++" -x hip -std=c++20 -nogpuinc -nogpulib -O3 -mno-wavefrontsize64 --offload-device-only \
    --offload-arch="$ARCH" -fgpu-rdc -emit-llvm -c -Xclang -fdenormal-fp-math=dynamic -DD4R_KERNEL_NAME="$K" \
    "${FLAGS[@]}" -o "$B/raw.bc" "$D/$SRC.hip"
"$LLVM/llvm-dis" "$B/raw.bc" -o "$B/raw.ll"
sed -E -e '/^\^[0-9]+ = /d' -e '/@llvm.used/d' -e '/wchar_size/d' -e '/llvm.module.flags/d' -e '/__hip_cuid/d' -e 's/optnone//g' \
    -e "s/\"target-cpu\"=\"$ARCH\"//g" -e 's/"target-features"="[^"]+"//g' \
    -e 's/denormal_fpenv\(dynamic\)/"denormal-fp-math"="dynamic,dynamic" "denormal-fp-math-f32"="dynamic,dynamic"/g' \
    -e 's/ nocreateundeforpoison//g' -e 's/, target_mem[0-9]*: [a-z]+//g' "$B/raw.ll" |
    python3 "$D/llvm23_constants.py" | "$LLVM_AS" -o "$B/extra.bc" -
D4R_PREFER_ACCURACY="$ACCURACY" D4R_TEX_FP8="$FP8" D4R_TEX_SRC="$SRC" D4R_TEX_WAVES="$WAVES" python3 "$D/make_ptx.py" "$K" "$B/$K.ptx" >/dev/null
# the code-generation switches of the runtime (d4r.ini [Kernels]), which ZLUDA's module cache keys on
env LD_LIBRARY_PATH="$R/lib" D4R_ZLUDA_EXTRA_BC="$B/extra.bc" D4R_ZLUDA_WMMA=1 D4R_ZLUDA_WMMA_FP8=1 \
    D4R_ZLUDA_WMMA_FP8_NATIVE="$FP8" D4R_ZLUDA_IGNORE_DENORMAL="$((1 - ACCURACY))" \
    D4R_ZLUDA_WMMA_F32ACC=0 D4R_ZLUDA_FAST_MATH=0 D4R_ZLUDA_IMPLICIT_MAX_BLOCK=256 \
    D4R_ZLUDA_WAVE64="$([[ "$ACCURACY" == 1 && "${D4R_TEX_ACCURACY_WAVE64:-0}" != 1 ]] && echo 0 || echo "${D4R_ZLUDA_WAVE64:-0}")" \
    "$D4R_ZLUDA_EMIT" "$B/$K.ptx" "$B/dump" "$ARCH" >/dev/null
cp "$B/dump/module.hsaco" "$OUT/$K.hsaco"
printf '%-52s %s\n' "$K" "$(grep -E '^\s+\.(vgpr_count|private_segment_fixed_size|wavefront_size):' "$B/dump/asm.s" | tr -s ' ' | paste -sd' ')"
