#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build/vulkan-staging}"
mkdir -p "$OUT"
glslangValidator -V --target-env vulkan1.0 -o "$OUT/read.spv" "$ROOT/kernels/support/vulkan_read.comp" >/dev/null
glslangValidator -V --target-env vulkan1.0 -o "$OUT/wait.spv" "$ROOT/kernels/support/vulkan_wait.comp" >/dev/null
# FSR Ray Regeneration backend (tools/d4r_fsr_rr_passes.h)
glslangValidator -V --target-env vulkan1.0 -o "$OUT/fsr_rr_prepare.spv" "$ROOT/kernels/support/fsr_rr_prepare.comp" >/dev/null
glslangValidator -V --target-env vulkan1.0 -o "$OUT/fsr_rr_compose.spv" "$ROOT/kernels/support/fsr_rr_compose.comp" >/dev/null
for format in rgba16f rgba32f r11f_g11f_b10f rgb10_a2 rgba8; do
    glslangValidator -V --target-env vulkan1.0 -DOUTPUT_FORMAT="$format" \
        -o "$OUT/write_$format.spv" "$ROOT/kernels/support/vulkan_write.comp" >/dev/null
done
for store in RGBA8:rgba8 BGRA8:bgra8 RGB10:rgb10 R11G11B10:r11g11b10 E5B9G9R9:e5b9g9r9 RGBA32F:rgba32f; do
    glslangValidator -V --target-env vulkan1.0 "-DOUTPUT_${store%%:*}=1" \
        -o "$OUT/write_f32_${store#*:}.spv" "$ROOT/kernels/support/vulkan_write_f32.comp" >/dev/null
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
