#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZLUDA_SOURCE_ROOT="${ZLUDA_SOURCE_ROOT:-}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"
HIPCC="${HIPCC:-$ROCM_ROOT/bin/hipcc}"
LLVM_DIS="${LLVM_DIS:-$ROCM_ROOT/lib/llvm/bin/llvm-dis}"
LLVM_AS="${LLVM_AS:-$ROCM_ROOT/lib/llvm/bin/llvm-as}"
OCML="${OCML:-$ROCM_ROOT/amdgcn/bitcode/ocml.bc}"
WORK_ROOT="${D4R_ZLUDA_PTX_HELPER_WORK:-${TMPDIR:-/tmp}/d4r-zluda-ptx-helpers}"
# D4R_ZLUDA_HELPER_FLAGS: extra compiler flags, e.g. -DD4R_WMMA12_SHIM for the gfx12-layout test build (never
# ship it: its gfx12 helpers use gfx11 WMMAs)
read -r -a EXTRA_FLAGS <<< "${D4R_ZLUDA_HELPER_FLAGS:-}"

if [[ -z "$ZLUDA_SOURCE_ROOT" || ! -f "$ZLUDA_SOURCE_ROOT/ptx/lib/zluda_ptx_impl.cpp" ]]; then
  printf 'Set ZLUDA_SOURCE_ROOT to a ZLUDA source checkout\n' >&2
  exit 2
fi
for dependency in "$HIPCC" "$LLVM_DIS" "$LLVM_AS" "$OCML"; do
  if [[ ! -e "$dependency" ]]; then
    printf 'Required ROCm/LLVM helper is missing: %s\n' "$dependency" >&2
    exit 2
  fi
done

mkdir -p "$WORK_ROOT"
SOURCE="$ZLUDA_SOURCE_ROOT/ptx/lib/zluda_ptx_impl.cpp"
PTX_LIB="$ZLUDA_SOURCE_ROOT/ptx/lib"
COMMON_FLAGS=(-DHIP_ENABLE_WARP_SYNC_BUILTINS -std=c++20 -Xclang -fdenormal-fp-math=dynamic
  --rocm-path="$ROCM_ROOT" -I"$ROCM_ROOT/include"
  -Wall -Wextra -Wsign-compare -Wconversion -x hip "$SOURCE" -nogpulib -O3
  -mno-wavefrontsize64 --offload-device-only --offload-arch=gfx1030 -emit-llvm -c
  -Xclang -mlink-bitcode-file -Xclang "$OCML" "${EXTRA_FLAGS[@]}")

build_one() {
  local suffix="$1"
  shift
  local raw="$WORK_ROOT/zluda_ptx_impl${suffix}.raw.bc"
  local clean="$WORK_ROOT/zluda_ptx_impl${suffix}.clean.bc"
  local ll="$WORK_ROOT/zluda_ptx_impl${suffix}.ll"
  local output="$PTX_LIB/zluda_ptx_impl${suffix}.bc"

  "$HIPCC" "${COMMON_FLAGS[@]}" "$@" -o "$raw"
  "$LLVM_DIS" "$raw" -o "$ll"
  sed -E \
      -e '/@llvm.used/d' \
      -e '/wchar_size/d' \
      -e '/llvm.module.flags/d' \
      -e '/__hip_cuid/d' \
      -e 's/optnone//g' \
      -e 's/define hidden/define linkonce_odr/g' \
      -e 's/"target-cpu"="gfx1030"//g' \
      -e 's/"target-features"="[^"]+"//g' "$ll" |
    "$LLVM_AS" -o "$clean" -
  mv "$clean" "$output"
  printf 'Wrote %s (%s bytes)\n' "$output" "$(stat -c '%s' "$output")"
}

build_one ""
build_one "_constrained" -ffp-model=strict -ffp-exception-behavior=ignore
# wave64 helpers (D4R_ZLUDA_WAVE64: two CUDA warps per wave; see d4r_warp_base in zluda_ptx_impl.cpp)
build_one "_w64" -mwavefrontsize64 -DD4R_WAVE64
build_one "_constrained_w64" -ffp-model=strict -ffp-exception-behavior=ignore -mwavefrontsize64 -DD4R_WAVE64
