#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build/vulkan-staging}"
mkdir -p "$OUT"
glslangValidator -V --target-env vulkan1.0 -o "$OUT/read.spv" "$ROOT/kernels/support/vulkan_read.comp" >/dev/null
for format in rgba16f rgba32f r11f_g11f_b10f rgb10_a2 rgba8; do
    glslangValidator -V --target-env vulkan1.0 -DOUTPUT_FORMAT="$format" \
        -o "$OUT/write_$format.spv" "$ROOT/kernels/support/vulkan_write.comp" >/dev/null
done
python3 - "$OUT" <<'PY'
from pathlib import Path
import struct, sys
root = Path(sys.argv[1]); lines = ['#pragma once', '#include <cstdint>']
for path in sorted(root.glob('*.spv')):
    data = path.read_bytes()
    words = struct.unpack('<' + 'I' * (len(data)//4), data)
    lines.append(f'static const uint32_t d4r_vk_{path.stem}[] = {{')
    lines.extend(','.join(f'0x{w:08x}' for w in words[i:i+8])+',' for i in range(0,len(words),8))
    lines.append('};')
(root/'d4r_vulkan_staging_spv.h').write_text('\n'.join(lines)+'\n')
PY
