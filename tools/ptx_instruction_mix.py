#!/usr/bin/env python3
"""Static instruction mix of one PTX entry: how many instructions NVIDIA's kernel spends per mma.

Counts PTX statements, so `mov` (mostly free after ptxas) and multi-instruction expansions are only approximate; the
number is meant for comparing the non-matrix work per matrix instruction with an RDNA4 instruction-timing table
(tools/rgp_instruction_mix.py). mma.sync.m16n8k32 e4m3 is 8192 FLOP, the same as one v_wmma_f32_16x16x16_fp8_fp8.

usage: ptx_instruction_mix.py FILE.ptx ENTRY [FILE.ptx ENTRY ...]
"""
import collections
import re
import sys


def entry(path, name):
    lines, out, on, depth = open(path).read().split('\n'), [], False, 0
    for line in lines:
        if not on and re.search(r'\.entry\s+' + re.escape(name) + r'\s*\(', line):
            on = True
        if on:
            out.append(line)
            depth += line.count('{') - line.count('}')
            if depth == 0 and any('{' in x for x in out) and len(out) > 3:
                break
    if not out:
        raise SystemExit(f'{path}: no entry {name}')
    return out


args = sys.argv[1:]
for path, name in zip(args[0::2], args[1::2]):
    ops = collections.Counter()
    for line in entry(path, name):
        s = line.strip().lstrip('{').strip()
        if not s or s.startswith(('//', '.', '}')) or s.endswith(':'):
            continue
        m = re.match(r'(@!?%p\d+\s+)?([a-z][\w.]*)', s)
        if m:
            ops[m.group(2).split('.')[0]] += 1
    total, mma = sum(ops.values()), ops['mma']
    print(f'{name} ({path}): {total} statements, {mma} mma, {total / max(mma, 1):.1f} per mma, '
          f'{ops["bar"]} bar, {ops["bra"]} bra')
    print('  ' + ', '.join(f'{k} {v}' for k, v in ops.most_common(20)))
