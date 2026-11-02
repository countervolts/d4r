#!/usr/bin/env python3
"""Build the staged output rewrite from the exact translated kernel.

Replaces path B's cooperative tile fill with output_tile.h. All other phases and
the DLAA path remain intact. Refuses to generate if the region's register contract
changes, so new translator output cannot silently reuse an incompatible mapping.
"""
import argparse
import re
from pathlib import Path

REGISTER = re.compile(r"\b(?:rd|rs|r|f|fd|p)\d+\b")
ASSIGN = re.compile(r"^\s*((?:rd|rs|r|f|fd|p)\d+)\s*=")
LABEL = re.compile(r"^(L__BB0_\d+):;$")


def rewrite(source):
    head, main = source.split('extern "C" __attribute__((global))', 1)
    lines = main.splitlines(keepends=True)
    labels = [(i, LABEL.match(line.strip()).group(1)) for i, line in enumerate(lines) if LABEL.match(line.strip())]
    blocks = {name: (start, labels[k + 1][0] if k + 1 < len(labels) else len(lines))
              for k, (start, name) in enumerate(labels)}
    pending, region = ['L__BB0_124'], set()
    while pending:
        name = pending.pop()
        if name in region or name == 'L__BB0_138':
            continue
        region.add(name)
        start, end = blocks[name]
        body = ''.join(lines[start + 1:end])
        pending += re.findall(r'goto (L__BB0_\d+);', body)
        if not re.search(r'^\s*goto L__BB0_\d+;\s*$', lines[end - 1]):
            # This region's fallthroughs are only its internal branch joins.
            pending += [n for i, n in labels if i == end]
    expected = {'L__BB0_124'} | {f'L__BB0_{k}' for k in range(127, 138)}
    if region != expected:
        raise ValueError(f'tile CFG changed: {sorted(region)}')
    removed = {i for name in region for i in range(*blocks[name])}
    definitions = {m.group(1) for i in removed if (m := ASSIGN.match(lines[i]))}
    # Definitions declared at function scope count as uses. Ignore declaration-only lines.
    uses_outside = set(REGISTER.findall(''.join(line for i, line in enumerate(lines)
                            if i not in removed and not re.match(r'^\s*(?:uint\d+_t|float|double|bool)\s+(?:rd|rs|r|f|fd|p)\d+(?:,|;)', line))))
    live_out = definitions & uses_outside
    if live_out != {'r215', 'r217'}:
        raise ValueError(f'tile register contract changed: {sorted(live_out)}')
    # The replacement passes r3, r4, r7, r8 and f1 by name: the region's inputs must be exactly the ones it was
    # written for (other flag variants of the kernel reuse this region; the checks above alone only cover outputs).
    region_text = ''.join(lines[i] for i in sorted(removed))
    live_in = set(REGISTER.findall(region_text)) - definitions
    if live_in != {'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'rd1', 'f1', 'f4'}:
        raise ValueError(f'tile input contract changed: {sorted(live_in)}')
    replacement = '''L__BB0_124:;
    output_tile_fill(KARGP, (int32_t)r3, (int32_t)r4, (int32_t)r7, (int32_t)(r8 + 1u),
                     f1, (AS3 uint8_t*)sharedColor, (AS3 uint8_t*)sharedDepth);
    r215 = (uint32_t)__builtin_amdgcn_workitem_id_y();
    r217 = (uint32_t)__builtin_amdgcn_workitem_id_x();
    goto L__BB0_138;
'''
    result = []
    for i, line in enumerate(lines):
        if i == blocks['L__BB0_124'][0]:
            result.append(replacement)
        if i not in removed:
            result.append(line)
    return head + '#include "output_tile.h"\n' + 'extern "C" __attribute__((global))' + ''.join(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    print(rewrite(args.source.read_text()), end='')
