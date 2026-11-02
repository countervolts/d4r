#!/usr/bin/env python3
"""LDR "max" output kernels: replace path B's flattened tile fill (one signed division and remainder by the
footprint width per texel) with direct x/y traversal (output_tile.h output_tile_fill_ldr). Same texels, same
LDS addresses, same conversion expressions; the original fill stays for footprints the direct one does not
cover (width or height < 1). Refuses to generate if the region's CFG or register contract changed."""
import argparse
import re
from pathlib import Path

REGISTER = re.compile(r"\b(?:rd|rs|r|f|fd|p)\d+\b")
ASSIGN = re.compile(r"^\s*((?:rd|rs|r|f|fd|p)\d+)\s*=")
LABEL = re.compile(r"^(L__BB0_\d+):;$")
DECL = re.compile(r'^\s*(?:uint\d+_t|float|double|bool)\s+(?:rd|rs|r|f|fd|p)\d+(?:,|;)')


def rewrite(source):
    head, main = source.split('extern "C" __attribute__((global))', 1)
    lines = main.splitlines(keepends=True)
    labels = [(i, LABEL.match(line.strip()).group(1)) for i, line in enumerate(lines) if LABEL.match(line.strip())]
    names = [n for _, n in labels]
    # path B: the block that the footprint test (both sides < 24) jumps to, up to the barrier block
    m = re.search(r'p\d+ = \(\(int32_t\)r(\d+) < \(int32_t\)24\);\n\s*p\d+ = \(\(int32_t\)r(\d+) < \(int32_t\)24\);\n'
                  r'\s*p\d+ = \(p\d+ && p\d+\);\n\s*if \(p\d+\) goto (L__BB0_\d+);', main)
    if not m:
        raise ValueError('path-B footprint test not found')
    wm1, hm1, first = 'r' + m.group(1), 'r' + m.group(2), m.group(3)
    start = names.index(first)
    stop = next(k for k in range(start + 1, len(names))
                if 'bar_sync();' in ''.join(lines[labels[k][0]:labels[k][0] + 2]))
    region_names = names[start:stop]
    lo, hi = labels[start][0], labels[stop][0]
    barrier = names[stop]
    body = ''.join(lines[lo:hi])
    targets = set(re.findall(r'goto (L__BB0_\d+);', body))
    if not targets <= set(region_names) | {barrier}:
        raise ValueError(f'tile region exits elsewhere: {sorted(targets - set(region_names))}')
    if len(region_names) != 3 or body.count('tex_fetch(') != 6 or body.count('/ (int32_t)') != 3:
        raise ValueError(f'tile region shape changed: {region_names}, {body.count("tex_fetch(")} fetches')
    removed = set(range(lo, hi))
    definitions = {mm.group(1) for i in removed if (mm := ASSIGN.match(lines[i]))}
    outside = ''.join(line for i, line in enumerate(lines) if i not in removed and not DECL.match(line))
    live_out = definitions & set(REGISTER.findall(outside))
    live_in = set(REGISTER.findall(body)) - definitions  # 'f4' is the ::f4 vector type
    # the fill's own expressions, checked literally: parameter offsets relative to rd1 (= kernel arguments + 40)
    for needle in ('rd1 + 208', 'rd1 + 160', 'rd1 + 56', 'rd1 + 60', 'rd1 + 64', 'rd1 + 68', 'rd1 + 168',
                   'rd1 + 172', 'rd1 + 176', 'rd1 + 180', '* 192u +', '* 96u +', '<< (3u & 31)', '<< (2u & 31)',
                   'asf(0x3E800000u)', 'asf(0x3F000000u)', 'asf(0x00000000u)'):
        if needle not in body:
            raise ValueError(f'tile expression changed: {needle!r} missing')
    if 'rd1 = (uint64_t)(rd16 + (uint64_t)40ull);' not in main or 'rd16 = (uint64_t)(KARGU);' not in main:
        raise ValueError('parameter base changed')
    expected_in = {'f4', 'rd1', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', wm1, hm1}
    if not live_in <= expected_in:
        raise ValueError(f'tile input contract changed: {sorted(live_in)}')
    if live_out != {'r35', 'r37'}:
        raise ValueError(f'tile output contract changed: {sorted(live_out)}')
    original = ''.join(lines[lo:hi]).replace(f'{first}:;\n', '', 1)
    replacement = f'''{first}:;
    r35 = (uint32_t)__builtin_amdgcn_workitem_id_y();
    r37 = (uint32_t)__builtin_amdgcn_workitem_id_x();
    if ((int32_t)r7 >= 1 && (int32_t){hm1} >= 0)
    {{
        output_tile_fill_ldr(KARGP, (int32_t)r3, (int32_t)r4, (int32_t)r7, (int32_t)({hm1} + 1u),
                             (AS3 uint8_t*)sharedColor, (AS3 uint8_t*)sharedDepth);
        goto {barrier};
    }}
{original}'''
    out = [line for i, line in enumerate(lines) if i < lo or i >= hi]
    out.insert(lo, replacement)
    return head + '#include "output_tile.h"\n' + 'extern "C" __attribute__((global))' + ''.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(rewrite(args.source.read_text()), end='')
