#!/usr/bin/env python3
"""Compile the shim's Vulkan staging shaders and embed them as d4r_vulkan_staging_spv.h.

usage: build_vulkan_staging.py [OUT_DIR]   (default build/vulkan-staging)
glslangValidator comes from GLSLANG_VALIDATOR, PATH or the Vulkan SDK (VULKAN_SDK).
"""
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUPPORT = ROOT / 'kernels/support'


def glslang():
    for candidate in (os.environ.get('GLSLANG_VALIDATOR'), shutil.which('glslangValidator')):
        if candidate:
            return candidate
    sdk = os.environ.get('VULKAN_SDK')
    for sub in ('Bin', 'bin'):
        if sdk and (Path(sdk) / sub / 'glslangValidator.exe').exists():
            return str(Path(sdk) / sub / 'glslangValidator.exe')
    sys.exit('glslangValidator not found (set GLSLANG_VALIDATOR or install the Vulkan SDK)')


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'build/vulkan-staging'
    out.mkdir(parents=True, exist_ok=True)
    compiler = glslang()
    jobs = [('read', 'vulkan_read.comp', []), ('wait', 'vulkan_wait.comp', []),
            # FSR Ray Regeneration backend (tools/d4r_fsr_rr_passes.h)
            ('fsr_rr_prepare', 'fsr_rr_prepare.comp', []), ('fsr_rr_compose', 'fsr_rr_compose.comp', [])]
    jobs += [(f'write_{f}', 'vulkan_write.comp', [f'-DOUTPUT_FORMAT={f}'])
             for f in ('rgba16f', 'rgba32f', 'r11f_g11f_b10f', 'rgb10_a2', 'rgba8')]
    for name, source, defines in jobs:
        subprocess.run([compiler, '-V', '--target-env', 'vulkan1.0', *defines, '-o', str(out / f'{name}.spv'),
                        str(SUPPORT / source)], check=True, stdout=subprocess.DEVNULL)
    lines = ['#pragma once', '#include <cstdint>']
    for path in sorted(out.glob('*.spv')):
        data = path.read_bytes()
        words = struct.unpack('<' + 'I' * (len(data) // 4), data)
        lines.append(f'static const uint32_t d4r_vk_{path.stem}[] = {{')
        lines.extend(','.join(f'0x{w:08x}' for w in words[i:i + 8]) + ',' for i in range(0, len(words), 8))
        lines.append('};')
    # written only when it changed, so a rebuild that touched nothing keeps its objects
    header = out / 'd4r_vulkan_staging_spv.h'
    text = '\n'.join(lines) + '\n'
    if not header.exists() or header.read_text() != text:
        header.write_text(text, newline='\n')


if __name__ == '__main__':
    main()
