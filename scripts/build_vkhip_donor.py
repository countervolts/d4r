#!/usr/bin/env python3
"""Build vkhip's SPIR-V donors (ordinary, and with four storage images); no device or NVIDIA inputs required."""
from pathlib import Path
import os
import struct
import subprocess
import sys
root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'build/vkhip'
out.mkdir(parents=True, exist_ok=True)
def spirv(name):
    spv = out / f'{name}.spv'
    subprocess.run([os.environ.get('D4R_GLSLANG', 'glslangValidator'), '-V', '--target-env', 'vulkan1.2',
                    str(root / f'engine/vkhip/{name}.comp'), '-o', str(spv)], check=True)
    words = struct.unpack('<' + 'I' * (spv.stat().st_size // 4), spv.read_bytes())
    return (f'static constexpr uint32_t {name}_spv[] = {{\n' +
            ',\n'.join('    ' + ', '.join(f'0x{x:08x}' for x in words[i:i+8]) for i in range(0, len(words), 8)) +
            '\n};\n')
# donor_images: the same with descriptor set 0's four storage images (vkhip.h createPipeline's `images`)
(out / 'donor_spv.h').write_text('// Generated from engine/vkhip/donor.comp and donor_images.comp\n'
                               '#pragma once\n#include <cstdint>\n' + spirv('donor') + spirv('donor_images'))
