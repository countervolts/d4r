#!/usr/bin/env python3
"""Compile an E/F CNN capture into Vulkan stages and a compact, alias-preserving tensor arena.

Usage: compile_e.py PTX_DIR CAPTURE_DIR OUTPUT_DIR [--preset E|F] [--hdr] [--frame N]
PTX and all model payloads are derived from the user's DLL and must not be redistributed.
"""
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def manifest(path):
    result = {'allocs': {}, 'pointers': [], 'textures': [], 'surfaces': []}
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == 'kernel':
            result['name'] = p[1]
        elif p[0] == 'launch':
            result['launch'] = list(map(int, p[1:7]))
        elif p[0] == 'alloc':
            result['allocs'][int(p[1])] = (int(p[2], 16), int(p[3]))
        elif p[0] == 'pointer':
            result['pointers'].append(tuple(map(int, p[1:4])))
        elif p[0] in ('texture', 'surface'):
            result[p[0] + 's'].append(int(p[1]))
    return result


def dump_shape(cap, stage, offset):
    number = int(stage['directory'].name.split('-')[1])
    prefix = f'launch-{number:03d}-{stage["name"]}-arg{offset:03d}-'
    paths = sorted((cap / 'dump').glob(prefix + '*'))
    for path in paths:
        m = re.search(r'-(texture|surface)-(\d+)x(\d+)x(\d+)-fmt(\d+)\.bin$', path.name)
        if m:
            _, w, h, channels, dtype = m.groups()
            channels, dtype = int(channels), int(dtype)
            if channels not in (1, 2, 4) or dtype not in (16, 32):
                raise ValueError(f'unsupported captured image: {path.name}')
            fmt = {1: 'r', 2: 'rg', 4: 'rgba'}[channels] + str(dtype) + 'f'
            return {'width': int(w), 'height': int(h), 'format': fmt}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ptx_dir', type=Path)
    ap.add_argument('capture_dir', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--preset', choices=('E', 'F'), default='E')
    ap.add_argument('--hdr', action='store_true')
    ap.add_argument('--frame', type=int, default=1)
    a = ap.parse_args()
    stages = []
    for d in sorted((a.capture_dir / 'replay').glob('replay-*')):
        s = manifest(d / 'manifest.txt')
        s['directory'] = d
        stages.append(s)
    starts = [i for i, s in enumerate(stages) if s['name'].startswith('cuda_engine_input_kernel_')]
    if not 1 <= a.frame <= len(starts):
        ap.error(f'capture contains {len(starts)} CNN frames')
    first = starts[a.frame - 1]
    end = starts[a.frame] if a.frame < len(starts) else len(stages)
    stages = [s for s in stages[first:end] if s['name'].startswith(('cuda_engine_input_kernel_', 'dltss_nchw8_'))]
    if len(stages) != 13 or 'aniso_gaussian_dfn' not in stages[-1]['name']:
        raise ValueError('capture does not contain the complete CNN input/network/reconstruction graph')
    if ('_ldr_' not in stages[0]['name']) != a.hdr:
        raise ValueError('--hdr does not match the captured input kernel')
    color = dump_shape(a.capture_dir, stages[0], 280)
    output = dump_shape(a.capture_dir, stages[-1], 464)
    if color is None or output is None:
        raise ValueError('capture lacks the current color or final output image')
    rw, rh, ow, oh = color['width'], color['height'], output['width'], output['height']

    # Merge only overlapping intervals, never CUDA address-space holes. A nonzero arena
    # prefix preserves null-pointer tests when a valid tensor was the first allocation.
    intervals = sorted({v for s in stages for v in s['allocs'].values()})
    regions = []
    for base, size in intervals:
        if regions and base < regions[-1]['end']:
            regions[-1]['end'] = max(regions[-1]['end'], base + size)
        else:
            regions.append({'base': base, 'end': base + size})
    arena_size = 256
    for r in regions:
        r['offset'] = arena_size
        arena_size = (arena_size + r['end'] - r['base'] + 255) & ~255
        r['mutable'] = False
    def region_for(address):
        return next(r for r in regions if r['base'] <= address < r['end'])
    def arena_offset(address):
        r = region_for(address)
        return r['offset'] + address - r['base']

    modules = {}
    for p in sorted(a.ptx_dir.glob('*.ptx')):
        text = p.read_text()
        for name in re.findall(r'\.entry\s+(\w+)\s*\(', text):
            if name not in modules or '.target sm_89' in text[:400]:
                modules[name] = p
    a.output.mkdir(parents=True, exist_ok=True)
    cache = {}
    lines = ['D4RE0001', f'size {rw} {rh} {ow} {oh}', f'preset {5 if a.preset == "E" else 6}', f'hdr {int(a.hdr)}',
             f'buffer 0 {arena_size} arena-init.bin']
    diagnostics = []
    for index, s in enumerate(stages):
        name, launch = s['name'], s['launch']
        if name not in cache:
            generated = a.output / (name + '.generated.comp')
            command = [sys.executable, str(HERE / 'rr/ptx2rr.py'), str(modules[name]), name, str(generated),
                       '--block', ','.join(map(str, launch[3:6])), '--weights-arg', '999', '--unified-arena']
            command += ['--buffer=' + str(off) for off, _, _ in s['pointers']]
            for off in s['surfaces']:
                shape = dump_shape(a.capture_dir, s, off)
                if shape:
                    command += [f'--surface={off}={shape["format"]}']
            subprocess.run(command, check=True)
            spv = a.output / (name + '.spv')
            subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', str(generated), '-o', str(spv)], check=True)
            info = json.loads(generated.with_suffix('.json').read_text())
            constants = generated.with_suffix('.globals.bin')
            globals_name = name + '.globals.bin'
            (a.output / globals_name).write_bytes(constants.read_bytes() if constants.exists() else bytes(16))
            cache[name] = (info, globals_name)
        info, globals_name = cache[name]
        for off, allocation, sub in s['pointers']:
            if off in info['written_buffers']:
                base, _ = s['allocs'][allocation]
                region_for(base + sub)['mutable'] = True
        params = bytearray((s['directory'] / 'args.bin').read_bytes())
        for off, allocation, sub in s['pointers']:
            base, _ = s['allocs'][allocation]
            struct.pack_into('<Q', params, off, arena_offset(base + sub))
        parameter_name = f'stage-{index:02d}.args'
        (a.output / parameter_name).write_bytes(params)
        lines.append(f'stage {index} {name}.spv {parameter_name} {globals_name} ' + ' '.join(map(str, launch[:3])))
        input_stage = index == 0
        for off in info['textures']:
            shape = dump_shape(a.capture_dir, s, off)
            roles = ({280: 'color', 288: 'history_read', 304: 'motion', 312: 'depth', 320: 'motion_read',
                      344: 'exposure', 360: 'previous_exposure'} if input_stage else
                     {456: 'color', 464: 'output', 472: 'exposure'})
            if shape is None:
                role = 'zero'
            elif off not in roles:
                raise ValueError(f'unmapped input image {name} argument {off}')
            else:
                role = roles[off]
            lines.append(f'texture {index} {off} {role}')
        for off, fmt in info['surfaces']:
            roles = ({328: 'motion_write', 384: 'output'} if input_stage else
                     {488: 'output', 496: 'history_write'})
            role = roles.get(off, 'discard')
            shape = dump_shape(a.capture_dir, s, off)
            if shape:
                fmt = shape['format']
            lines.append(f'surface {index} {off} {role} {fmt}')
        diagnostics.append({'name': name, 'params': parameter_name, 'grid': launch[:3], 'writtenArguments': info['written_buffers']})

    initial = bytearray(arena_size)
    copied = set()
    for s in stages:
        for allocation, (base, size) in s['allocs'].items():
            r = region_for(base)
            if r['mutable'] or base in copied:
                continue
            data = (s['directory'] / f'alloc-{allocation}.bin').read_bytes()
            if len(data) != size:
                raise ValueError('truncated captured allocation')
            at = arena_offset(base)
            initial[at:at+size] = data
            copied.add(base)
    (a.output / 'arena-init.bin').write_bytes(initial)
    (a.output / 'e.txt').write_text('\n'.join(lines) + '\n')
    (a.output / 'model.json').write_text(json.dumps({'format': 'D4RE0001', 'preset': a.preset,
        'geometry': [rw, rh, ow, oh], 'arenaBytes': arena_size, 'regions': regions, 'stages': diagnostics}, indent=2) + '\n')
    print(f'CNN {a.preset}: {len(stages)} stages, {arena_size} arena bytes, {rw}x{rh} -> {ow}x{oh}')


if __name__ == '__main__':
    main()
