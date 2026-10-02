#!/usr/bin/env bash
# Build d4r-owned motion processing kernels and embed them in the Wine bridge.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build/wine-nvcuda/generated}"
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
CLANG="$ROCM/lib/llvm/bin/clang++"
[[ -x "$CLANG" ]] || { echo "motion kernels need ROCm clang (D4R_ROCM_DIR)" >&2; exit 2; }
mkdir -p "$OUT"
for arch in gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201; do
    "$CLANG" -x hip -nogpuinc -nogpulib --offload-device-only --offload-arch="$arch" \
        --no-gpu-bundle-output -O3 -mno-wavefrontsize64 -o "$OUT/$arch.hsaco" \
        "$ROOT/kernels/support/motion_dilate.hip"
done
python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
lines = ['/* Generated from d4r-owned HIP source; do not edit. */', '#include <string.h>']
for p in sorted(root.glob('gfx*.hsaco')):
    data = p.read_bytes()
    lines.append(f'static const unsigned char motion_{p.stem}[] __attribute__((aligned(64))) = {{')
    lines.extend(','.join(f'0x{x:02x}' for x in data[i:i+24]) + ',' for i in range(0,len(data),24))
    lines.append('};')
lines.append('static const void* d4r_motion_image(const char* arch) {')
for p in sorted(root.glob('gfx*.hsaco')):
    lines.append(f'  if (strcmp(arch, "{p.stem}") == 0) return motion_{p.stem};')
lines.append('  return NULL;\n}')
(root/'d4r_motion_kernels.h').write_text('\n'.join(lines)+'\n')
PY
