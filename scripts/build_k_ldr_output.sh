#!/usr/bin/env bash
# Compile K's LDR output using the same PTX semantics as the HDR native route. The packed-store
# sidecar authenticates the produced code bytes; it is not a substitute for temporal validation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${1:?usage: build_k_ldr_output.sh OUT_DIR [KERNEL]}")"
NAME="${2:-hiluma_engine_output_depthinv_mvlo_ldr_max_v2_rel}"
[[ "$NAME" =~ ^hiluma_engine_output_depth(inv|reg)_mv(hi|lo)_ldr_max_v2_rel$ ]] || exit 2
: "${D4R_DLSS_DLL:?set D4R_DLSS_DLL to the matching NVIDIA DLSS library}"
mkdir -p "$OUT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
python3 "$ROOT/kernels/tools/extract_dlss_ptx.py" "$D4R_DLSS_DLL" "$WORK/ptx" >/dev/null
PTX="$(rg -l "\.entry[[:space:]]+$NAME[[:space:]]*\(" "$WORK/ptx"/*.ptx)"
[[ -n "$PTX" && "$PTX" != *$'\n'* ]] || { echo 'kernel must appear in exactly one module' >&2; exit 2; }
# LDR uses another CFG/parameter layout than HDR: do not apply HDR's tile rewrite by register name.
python3 "$ROOT/kernels/native/ptx2hip.py" --as1 "$PTX" "$NAME" > "$WORK/output.hip"
# D4R_LDR_OUTPUT_MAX_VGPR=N (opt-in): cap the kernel's VGPRs. The DLAA path (path A, 100 VGPRs alone) otherwise sets
# the allocation of every block (101 VGPRs, 7 waves/SIMD); with 96 its spills stay in path A and Quality..UP blocks
# (path B, 88 VGPRs alone) run at 8 waves/SIMD: depthinv_mvlo output 0.825 -> 0.776 ms at 1440p, DLAA 0.840 ->
# 0.812, every frame byte-identical (private/notes/k1440-implementation.md, O1). Lower caps spill in path B.
if [[ -n "${D4R_LDR_OUTPUT_MAX_VGPR:-}" ]]; then
    [[ "$D4R_LDR_OUTPUT_MAX_VGPR" =~ ^[0-9]+$ ]] || { echo 'D4R_LDR_OUTPUT_MAX_VGPR must be a number' >&2; exit 2; }
    sed -i "s/__attribute__((amdgpu_flat_work_group_size(1, 256))) void $NAME(/__attribute__((amdgpu_flat_work_group_size(1, 256), amdgpu_num_vgpr($D4R_LDR_OUTPUT_MAX_VGPR))) void $NAME(/" "$WORK/output.hip"
    grep -q "amdgpu_num_vgpr($D4R_LDR_OUTPUT_MAX_VGPR)" "$WORK/output.hip" || { echo 'VGPR cap not applied' >&2; exit 2; }
fi
bash "$ROOT/kernels/native/build_native.sh" "$WORK/output.hip" "$OUT/$NAME.hsaco" \
    -DD4R_NATIVE_F2I -DD4R_AS1_STORES
python3 - "$OUT" "$NAME" <<'PY'
from pathlib import Path
import sys
out, name = Path(sys.argv[1]), sys.argv[2]
value = 14695981039346656037
for byte in (out / (name + '.hsaco')).read_bytes():
    value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
path = out / 'd4r-rgba8.txt'
rows = path.read_text().splitlines() if path.exists() else ['# RGBA8 store capability: NAME FNV1a64 of code object']
rows = [row for row in rows if not row.startswith(name + ' ')]
path.write_text('\n'.join(rows + [f'{name} {value:016x}']) + '\n')
PY
python3 "$ROOT/kernels/tools/kernel_manifest.py" "$OUT" "$D4R_DLSS_DLL"
