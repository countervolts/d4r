#!/usr/bin/env bash
# build_native.sh SRC.hip OUT.hsaco [extra clang flags]: natively compiled DLSS texture kernel (gfx1101, wave64 default)
# WAVE=32|64 (default 64); CUMODE=1|0 (default 1): CU mode, or WGP mode (a workgroup's waves on all four SIMDs)
set -euo pipefail
ND=$(cd "$(dirname "$0")" && pwd)
SRC=$(realpath "$1") OUT=$(realpath -m "$2"); shift 2
[[ -s "$SRC" ]] || { echo "empty native source: $SRC" >&2; exit 2; }
R=${D4R_ROCM_DIR:-/opt/rocm}
ARCH=${D4R_GPU_ARCH:-gfx1101}
W=${WAVE:-64}
WF=$([[ $W == 64 ]] && echo -mwavefrontsize64 || echo -mno-wavefrontsize64)
CU=$([[ ${CUMODE:-1} == 1 ]] && echo -mcumode || echo -mno-cumode)
D=$(cd "$(dirname "$SRC")" && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$T" && "$R/lib/llvm/bin/clang++" -x hip -std=c++20 --offload-arch=$ARCH --offload-device-only -O3 $WF $CU \
  -ffp-contract=off -fgpu-flush-denormals-to-zero -fno-fast-math -I"$D" -I"$ND" -I"$R/include" --rocm-path="$R" \
  --rocm-device-lib-path="$R/amdgcn/bitcode" "$@" -o "$OUT" "$SRC" -save-temps=cwd --no-gpu-bundle-output)
S=$(ls "$T"/*.s | head -1)
cp "$S" "${OUT%.hsaco}.s"
printf '%s: %s\n' "$(basename "$OUT")" "$(grep -hE '^\s+\.(vgpr_count|sgpr_count|private_segment_fixed_size|vgpr_spill_count|group_segment_fixed_size):' "$S" | tr -s ' ' | paste -sd' ')"
