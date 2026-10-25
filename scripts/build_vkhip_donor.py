#!/usr/bin/env python3
"""Build vkhip's ordinary SPIR-V donor; no device or NVIDIA inputs required."""
from pathlib import Path
import struct
import subprocess
import sys
root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'build/vkhip'
out.mkdir(parents=True, exist_ok=True)
spv = out / 'donor.spv'
subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.2',
                str(root / 'engine/vkhip/donor.comp'), '-o', str(spv)], check=True)
words = struct.unpack('<' + 'I' * (spv.stat().st_size // 4), spv.read_bytes())
(out / 'donor_spv.h').write_text('// Generated from engine/vkhip/donor.comp\n#pragma once\n#include <cstdint>\n'
                               'static constexpr uint32_t donor_spv[] = {\n' +
                               ',\n'.join('    ' + ', '.join(f'0x{x:08x}' for x in words[i:i+8])
                                         for i in range(0, len(words), 8)) + '\n};\n')
