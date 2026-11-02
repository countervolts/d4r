#!/usr/bin/env python3
"""Static instruction mix of the matrix shaders in a RADV shader dump (RADV_DEBUG=shaders), by the categories of
tools/rgp_instruction_mix.py.

The L stages unroll completely, so a static count is the per-wave count: an edit to layer_m.comp can be judged here
before it is timed. Shaders are told apart by their header (textures, images, shared memory); the highest VGPR named
bounds the allocation.

usage: radv_isa_mix.py DUMP.txt [DUMP.txt ...]
   e.g. RADV_DEBUG=shaders,nocache d4r-m MODEL 1920 1080 3840 2160 OUT 1 vkhip -- FRAMES... 2> DUMP.txt
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rgp_instruction_mix import kind  # noqa: E402


def shaders(path):
    out, cur = [], None
    for line in open(path, errors='replace'):
        if line.startswith('shader: MESA_SHADER_COMPUTE'):
            cur = {'head': {}, 'ops': [], 'asm': False}
            out.append(cur)
        elif cur is not None:
            m = re.match(r'(workgroup_size|num_textures|num_images|shared_size): (.*)', line)
            if m and not cur['asm']:
                cur['head'][m[1]] = m[2].strip()
            elif line.startswith('disasm:'):
                cur['asm'] = True
            elif cur['asm'] and line.startswith('\t'):
                cur['ops'].append(line.strip())
    return out


def report(path):
    print(path)
    for sh in shaders(path):
        ops = [o.split()[0] for o in sh['ops'] if o and not o.startswith(';')]
        wmma = sum(o.startswith('v_wmma') for o in ops)
        if not wmma:
            continue
        vg = max((int(x) for o in sh['ops'] for x in re.findall(r'\bv\[?(\d+)', o.split(';')[0])), default=0)
        valu = sum(o.startswith('v_') for o in ops)
        k = collections.Counter(kind(o) for o in ops if o.startswith('v_'))
        h = sh['head']
        print(f"  wg {h.get('workgroup_size')}  tex {h.get('num_textures', 0)}  img {h.get('num_images', 0)}  "
              f"lds {h.get('shared_size')}  v0..v{vg}: {valu} VALU, {wmma} WMMA ({valu / wmma:.1f} per WMMA)")
        print('    ' + ', '.join(f'{c} {n}' for c, n in k.most_common()))
        top = collections.Counter(o for o in ops if o.startswith('v_') and kind(o) in ('float', 'other conversion'))
        print('    float/conversion opcodes: ' + ', '.join(f'{o} {n}' for o, n in top.most_common(14)))


for p in sys.argv[1:]:
    report(p)
