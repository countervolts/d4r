#!/usr/bin/env python3
"""Generate deterministic RR replay inputs. This is a dispatch smoke, not a quality reference."""
import argparse
import json
import math
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--output-width', type=int)
    parser.add_argument('--output-height', type=int)
    args = parser.parse_args()
    if not (32 <= args.width <= 1024 and 32 <= args.height <= 1024 and 1 <= args.frames <= 32):
        parser.error('width/height must be 32..1024; frames must be 1..32')
    if bool(args.output_width) != bool(args.output_height):
        parser.error('supply both output dimensions')
    ow, oh = args.output_width or args.width * 2, args.output_height or args.height * 2
    if not (1 <= ow <= 2560 and 1 <= oh <= 1440):
        parser.error('output must fit 2560x1440')
    args.output.mkdir(parents=True, exist_ok=False)
    identity = '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'
    lines = [f'size {args.width} {args.height} {ow} {oh}']
    for frame in range(args.frames):
        data = {name: bytearray() for name in ('color', 'motion', 'depth', 'normals', 'roughness', 'diffuse', 'specular')}
        for y in range(args.height):
            for x in range(args.width):
                u, v = x / (args.width - 1), y / (args.height - 1)
                noise = math.sin(x * 1.7 + y * 2.3 + frame) * 0.025
                data['color'] += struct.pack('<4e', 0.1 + u + noise, 0.1 + v + noise, 0.2 + u * v, 1)
                data['motion'] += struct.pack('<2e', 0, 0)
                data['depth'] += struct.pack('<f', 1 + v)
                data['normals'] += struct.pack('<4e', 0, 0, 1, 1)
                data['roughness'] += struct.pack('<4e', 0.5, 0.5, 0.5, 0.5)
                data['diffuse'] += bytes((128, 160, 192, 255))
                data['specular'] += bytes((10, 10, 10, 255))
        for name, blob in data.items():
            (args.output / f'{frame}-{name}.bin').write_bytes(blob)
        lines.append(f'frame {frame} 0 0 1 1 1 {int(frame == 0)} 0 {identity} {identity}')
    (args.output / 'frames.txt').write_text('\n'.join(lines) + '\n', encoding='ascii')
    (args.output / 'synthetic.json').write_text(json.dumps({
        'kind': 'synthetic-dispatch-only', 'render': [args.width, args.height],
        'output': [ow, oh], 'frames': args.frames,
        'quality_reference': False,
    }, indent=2) + '\n', encoding='ascii')


if __name__ == '__main__':
    main()
