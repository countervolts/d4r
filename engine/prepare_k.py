#!/usr/bin/env python3
"""Prepare a full-frame native K replay from a capture. Does not translate kernels.

prepare_k.py CAPTURE_DIR OUTPUT_DIR [--frames N] [--start FRAME]
The model's reconstruction tables can be packaged using OUTPUT_DIR/lut.bin.
"""
import argparse
import json
from pathlib import Path
import struct
from capture import INPUT, OUTPUT, ORDER, launches, input_params, output_params


def quote(path):
    return json.dumps(str(Path(path).resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--frames', type=int, default=0, help='0 = all captured frames')
    parser.add_argument('--start', type=int, default=0)
    a = parser.parse_args()
    inputs, outputs = launches(a.capture, INPUT), launches(a.capture, OUTPUT)
    if not inputs or not outputs or len(outputs) > len(inputs):
        raise ValueError('capture must contain complete K input/output frame pairs')
    pairs = list(zip(inputs, outputs))
    for i, (ip, op) in enumerate(pairs):
        if not ip.sequence < op.sequence or (i+1 < len(inputs) and op.sequence >= inputs[i+1].sequence):
            raise ValueError('capture has a missing or out-of-order input/output launch')
    if a.start < 0 or a.start >= len(pairs) or a.frames < 0:
        raise ValueError('invalid frame selection')
    end = len(pairs) if a.frames == 0 else min(len(pairs), a.start + a.frames)
    selected = pairs[a.start:end]
    # Unselected later frames may be incomplete (for example a capture stopped while writing them).
    for ip, op in selected:
        for resource in (*ip.resources.values(), *op.resources.values()): resource.validate()
    layer_launches = {name: launches(a.capture, 'dltss_pwin_'+name+'_layer') for name in ORDER}
    first_in, first_out = selected[0]
    geometry = []
    for name in ORDER:
        matches = [p for p in layer_launches[name] if first_in.sequence < p.sequence < first_out.sequence]
        if len(matches) != 1:
            raise ValueError(f'frame {a.start}: expected one {name} launch, found {len(matches)}')
        geometry.append((*matches[0].i2(0), *matches[0].i2(56)))
    final = first_out.resources[312]
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'shape.bin').write_bytes(struct.pack('<2I', final.width, final.height) + b''.join(struct.pack('<4i', *s) for s in geometry))
    # NGX's whole allocation: sixteen reconstruction tables, of which a launch uses the one for its upscaling ratio
    (a.output / 'lut.bin').write_bytes(first_out.allocation(280, 16 * 32768))
    lines = [f'shape {quote(a.output / "shape.bin")}',
             f'history {quote(first_in.resources[176].path)} {quote(first_out.resources[232].path)} {quote(first_in.resources[224].path)}']
    resource_ids = {}
    references = []
    for number, (ip, op) in enumerate(selected):
        if not ip.sequence < op.sequence or (number and ip.sequence <= selected[number-1][1].sequence):
            raise ValueError('capture frames are not in network order')
        if op.resources[312].width != final.width or op.resources[312].height != final.height:
            raise ValueError('resolution changes require another Engine')
        frame_geometry = []
        for name, expected in zip(ORDER, geometry):
            matches = [p for p in layer_launches[name] if ip.sequence < p.sequence < op.sequence]
            if len(matches) != 1 or matches[0].i2(0) != expected[:2]:
                raise ValueError(f'frame {number}: missing or changed {name} geometry')
            frame_geometry.append((*matches[0].i2(0), *matches[0].i2(56)))
        ids = []
        for off in (168, 184, 192):
            r = ip.resources[off]
            if r.path not in resource_ids:
                i = len(resource_ids); resource_ids[r.path] = i
                lines.append(f'image {i} {quote(r.path)} {r.width} {r.height} {r.channels} {r.bits}')
            ids.append(resource_ids[r.path])
        inp, outp = a.output / f'frame-{number}.input', a.output / f'frame-{number}.output'
        inp.write_bytes(input_params(ip)); outp.write_bytes(output_params(op))
        windows = a.output / f'frame-{number}.windows'
        windows.write_bytes(b''.join(struct.pack('<4i', *s) for s in frame_geometry))
        lines.append(f'frame {quote(inp)} {quote(outp)} {quote(windows)} ' + ' '.join(map(str, ids)))
        references.append({str(off): str(op.resources[off].path) for off in (288,296,304,312)})
    (a.output / 'case.txt').write_text('\n'.join(lines) + '\n')
    (a.output / 'references.json').write_text(json.dumps(references, indent=2) + '\n')
    print(f'Prepared {len(selected)} frames, {final.width}x{final.height}, token grid {geometry[0][0]}x{geometry[0][1]}')


if __name__ == '__main__': main()
