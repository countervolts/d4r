#!/usr/bin/env python3
"""Issue independent texture fetches of the translated hiluma input kernel earlier (opt-in experiment).

usage: in_hoist.py KERNEL.hip ANCHOR_REG HANDLE_REG [HANDLE_REG ...] > OUT.hip

The input kernel (ptx2hip --ifconv --sink + in_lds2reg.py) is one straight-line body. Its 3x3 color neighbourhood
fetches depend only on the thread's pixel position and kernel parameters, yet the compiler issues them after the
depth -> motion -> history chain, adding serial memory round trips to a latency-bound kernel. This pass moves every
fetch block `{ ::f4 t_ = tex_fetch(rdX, ...); ... }` whose handle register is one of HANDLE_REG's copies, together
with the backward slice of statements computing its operands, to right after the statement that assigns
ANCHOR_REG, followed by a scheduling barrier (instruction order only). Nothing else moves and no statement
changes, so every value is computed exactly as before.

Refuses unless: every register the moved statements assign is assigned exactly once in the kernel; every register
they read is assigned before the anchor (or by another moved statement, earlier in order); there is no memory store
and no control flow (label/goto/return) between the anchor and the last moved statement.
"""
import re
import sys

REG = re.compile(r'\b(?:rd|rs|r|f|p|shv\d+_|dpv\d+_)\d+\b|\bl\d+_\w+\b')
ASSIGN = re.compile(r'^\s*([A-Za-z_]\w*)\s*=(?!=)')


def stmts(body_lines):
    """group lines into statements: fetch blocks '{ ... }' become one statement"""
    out, i = [], 0
    while i < len(body_lines):
        line = body_lines[i]
        if line.strip().startswith('{ ::f4 t_ ='):
            j = i
            while body_lines[j].strip() != '}':
                j += 1
            out.append(body_lines[i:j + 1])
            i = j + 1
        else:
            out.append([line])
            i += 1
    return out


def defs_uses(st):
    text = ''.join(st)
    if text.strip().startswith('{ ::f4 t_ ='):
        defs = set(re.findall(r'^\s*([A-Za-z_]\w*) = t_\[\d\];', text, re.M))
        call = text.split(';', 1)[0].replace('::f4', '')
        return defs, set(REG.findall(call))
    m = ASSIGN.match(text)
    if m:
        lhs = m.group(1)
        rhs = text[m.end():]
        return {lhs}, set(REG.findall(rhs))
    return set(), set(REG.findall(text))


def main():
    src = open(sys.argv[1]).read().split('\n')
    anchor, handles = sys.argv[2], set(sys.argv[3:])
    start = next(i for i, l in enumerate(src) if l.startswith('extern "C" __attribute__((global))')) + 2
    end = max(i for i, l in enumerate(src) if l.strip() == '}')
    S = stmts([l + '\n' for l in src[start:end]])
    info = [defs_uses(s) for s in S]
    assigned = {}
    for k, (d, _) in enumerate(info):
        for r in d:
            assigned.setdefault(r, []).append(k)
    # handle copies: rdY = (uint64_t)(rdX) chains from the given handle registers
    alias = set(handles)
    for k, s in enumerate(S):
        m = re.match(r'^\s*(rd\d+) = \(uint64_t\)\((rd\d+)\);$', s[0])
        if m and m.group(2) in alias:
            alias.add(m.group(1))
    ia = [k for k, (d, _) in enumerate(info) if anchor in d]
    if len(ia) != 1:
        raise SystemExit(f'anchor {anchor} assigned {len(ia)} times')
    ia = ia[0]
    fetches = [k for k, s in enumerate(S) if s[0].strip().startswith('{ ::f4 t_ = tex_fetch(')
               and re.search(r'tex_fetch\((rd\d+)', s[0]).group(1) in alias and k > ia]
    if not fetches:
        raise SystemExit('no fetch to move')
    need, move = set(), set(fetches)
    for k in fetches:
        need |= info[k][1]
    while need:
        r = need.pop()
        ks = assigned.get(r, [])
        if not ks:
            continue  # parameters / builtins
        if len(ks) != 1:
            raise SystemExit(f'register {r} is assigned {len(ks)} times')
        k = ks[0]
        if k <= ia or k in move:
            continue
        move.add(k)
        need |= info[k][1]
    last = max(move)
    for k in range(ia + 1, last + 1):
        text = ''.join(S[k])
        if re.search(r'^\s*\*\(|\bgoto\b|\breturn\b|^L__BB', text, re.M):
            raise SystemExit(f'store or control flow between anchor and moved code: {text.strip()[:80]}')
    for k in sorted(move):
        for r in info[k][0]:
            if len(assigned[r]) != 1:
                raise SystemExit(f'moved register {r} assigned more than once')
        for r in info[k][1]:
            ks = assigned.get(r, [])
            if ks and not (ks[0] <= ia or ks[0] in move and ks[0] < k):
                raise SystemExit(f'moved statement reads {r} defined later')
    # a scheduling barrier keeps the compiler from sinking the fetches back to their uses (order only)
    barrier = ['    __builtin_amdgcn_sched_barrier(0); // in_hoist: fetches above are issued here\n']
    body = ''.join(''.join(S[k]) for k in range(0, ia + 1)) + ''.join(''.join(S[k]) for k in sorted(move)) + \
        ''.join(barrier) + ''.join(''.join(S[k]) for k in range(ia + 1, len(S)) if k not in move)
    sys.stderr.write(f'in_hoist: moved {len(move)} statements ({len(fetches)} fetches) after {anchor}\n')
    sys.stdout.write('\n'.join(src[:start]) + '\n' + body + '\n'.join(src[end:]))


if __name__ == '__main__':
    main()
