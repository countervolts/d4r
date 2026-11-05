#!/usr/bin/env python3
"""Prepare a d4r-k replay from a triggered native D3D12 runtime snapshot.

Set D4R_D3D12_CAPTURE_DIR, then create capture-now in that directory while
the scene is visible. The runtime saves a frame-N directory after its fence.
Pass that directory here; both d4r-k and d4r-k-d3d12 can replay case.txt.
"""
import argparse
import json
from pathlib import Path
import struct


def prepare(directory):
    directory = Path(directory).resolve()
    params = (directory / 'params.bin').read_bytes()
    if len(params) < 448:
        raise ValueError('incomplete parameter capture')
    (directory / 'frame.input').write_bytes(params[:128])
    (directory / 'frame.output').write_bytes(params[188:376])
    shape = (directory / 'shape.bin').read_bytes()
    if len(shape) != 184:
        raise ValueError('invalid K shape capture')
    (directory / 'frame.windows').write_bytes(shape[8:])
    quote = lambda name: json.dumps(str(directory / name))
    lines = [f'shape {quote("shape.bin")}', 'history ' + ' '.join(
        quote('history-' + name + '.bin') for name in ('color', 'luma', 'feature'))]
    formats = {10: (4, 16), 2: (4, 32), 34: (2, 16), 16: (2, 32),
               41: (1, 32), 54: (1, 16), 56: (1, 16)}
    textures = (directory / 'textures.txt').read_text().splitlines()
    if len(textures) != 3:
        raise ValueError('expected color, motion and depth textures')
    for index, (name, geometry) in enumerate(zip(('color', 'motion', 'depth'), textures)):
        width, height, format_id = map(int, geometry.split())
        file = name + '.bin'
        # Replays consume floating point planes, not UNORM bit patterns.
        if format_id == 56:
            raise ValueError('R16_UNORM depth needs conversion to float before replay')
        if format_id == 24:
            # R10G10B10A2_UNORM colour (Ghost of Tsushima): decoded to RGBA16F as the view's hardware decode reads it
            # (10-bit unorm and the 2-bit alpha are exact in binary16)
            raw = (directory / file).read_bytes()
            if len(raw) != width * height * 4:
                raise ValueError(f'incomplete {name} texture')
            words = memoryview(raw).cast('I')
            out = bytearray(width * height * 8)
            texel = struct.Struct('<4e')
            for i, w in enumerate(words):
                texel.pack_into(out, 8 * i, (w & 1023) / 1023, (w >> 10 & 1023) / 1023, (w >> 20 & 1023) / 1023, (w >> 30) / 3)
            file = name + '-rgba16f.bin'
            (directory / file).write_bytes(out)
            format_id = 10
        if format_id in (20, 21):
            # D32_FLOAT_S8X24 depth: the capture holds its float plane (R32_FLOAT_X8X24's view), four bytes a texel
            format_id = 41
        if format_id not in formats:
            raise ValueError(f'unsupported capture DXGI format {format_id}')
        channels, bits = formats[format_id]
        if (directory / file).stat().st_size != width * height * channels * bits // 8:
            raise ValueError(f'incomplete {name} texture')
        lines.append(f'image {index} {quote(file)} {width} {height} {channels} {bits}')
    lines.append('frame ' + ' '.join(quote('frame.' + name) for name in ('input', 'output', 'windows')) + ' 0 1 2')
    (directory / 'case.txt').write_text('\n'.join(lines) + '\n')
    references = {'312': 'output.bin', '288': 'next-color.bin', '296': 'next-luma.bin', '304': 'next-feature.bin'}
    (directory / 'references.json').write_text(json.dumps([{k: str(directory / v) for k, v in references.items()}], indent=2))
    return directory / 'case.txt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(prepare(parser.parse_args().directory))
