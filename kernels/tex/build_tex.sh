#!/usr/bin/env bash
# build_tex.sh KERNEL SRC OUT_DIR: one texture kernel. NVIDIA's PTX for KERNEL, edited by make_ptx.py so
# part of it calls functions from SRC.hip, is compiled by ZLUDA with SRC's bitcode linked in
# (D4R_ZLUDA_EXTRA_BC) and the resulting code object is saved as OUT_DIR/KERNEL.hsaco.
# Called by kernels/build.sh, which sets D4R_ROCM_DIR, D4R_GPU_ARCH, D4R_DLSS_PTX_DIR, D4R_ZLUDA_EMIT,
# D4R_ZLUDA_WAVE64 (1 = wave64 code object) and D4R_TEX_FP8 (1 = the gfx12 native-FP8 variant).
# ZLUDA compiles offline for D4R_GPU_ARCH with its d4r_emit tool, so any target builds on any machine.
set -euo pipefail
K=$1 SRC=$2 OUT=$3
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="${D4R_ROCM_DIR:-/opt/rocm}"
ARCH="${D4R_GPU_ARCH:-gfx1101}"
FP8="${D4R_TEX_FP8:-0}"
LLVM="$R/lib/llvm/bin"
B="$(mktemp -d)"
trap 'rm -rf "$B"' EXIT
# D4R_TEX_CFLAGS: extra flags for SRC.hip (tests, e.g. -DD4R_WMMA_LAYOUT=12 for the gfx12-layout shim on gfx11)
read -r -a FLAGS <<< "${D4R_TEX_CFLAGS:-}"
[[ "$FP8" == 1 ]] && FLAGS+=(-DD4R_TEX_FP8)
# SRC.hip -> bitcode without target attributes, so ZLUDA's own target settings apply after linking
"$LLVM/clang++" -x hip -std=c++20 -nogpuinc -nogpulib -O3 -mno-wavefrontsize64 --offload-device-only \
    --offload-arch="$ARCH" -fgpu-rdc -emit-llvm -c -Xclang -fdenormal-fp-math=dynamic -DD4R_KERNEL_NAME="$K" \
    "${FLAGS[@]}" -o "$B/raw.bc" "$D/$SRC.hip"
"$LLVM/llvm-dis" "$B/raw.bc" -o "$B/raw.ll"
sed -E -e '/@llvm.used/d' -e '/wchar_size/d' -e '/llvm.module.flags/d' -e '/__hip_cuid/d' -e 's/optnone//g' \
    -e "s/\"target-cpu\"=\"$ARCH\"//g" -e 's/"target-features"="[^"]+"//g' "$B/raw.ll" | "$LLVM/llvm-as" -o "$B/extra.bc" -
python3 "$D/make_ptx.py" "$K" "$B/$K.ptx" >/dev/null
# the code-generation switches of the runtime (d4r.ini [Kernels]), which ZLUDA's module cache keys on
env LD_LIBRARY_PATH="$R/lib" D4R_ZLUDA_EXTRA_BC="$B/extra.bc" D4R_ZLUDA_WMMA=1 D4R_ZLUDA_WMMA_FP8=1 \
    D4R_ZLUDA_WMMA_FP8_NATIVE="$FP8" D4R_ZLUDA_IGNORE_DENORMAL=1 D4R_ZLUDA_IMPLICIT_MAX_BLOCK=256 \
    D4R_ZLUDA_WAVE64="${D4R_ZLUDA_WAVE64:-0}" "$D4R_ZLUDA_EMIT" "$B/$K.ptx" "$B/dump" "$ARCH" >/dev/null
cp "$B/dump/module.hsaco" "$OUT/$K.hsaco"
printf '%-52s %s\n' "$K" "$(grep -E '^\s+\.(vgpr_count|private_segment_fixed_size|wavefront_size):' "$B/dump/asm.s" | tr -s ' ' | paste -sd' ')"
