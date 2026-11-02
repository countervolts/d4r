#!/usr/bin/env python3
"""Instruction mix of one dispatch from an RGP instruction-timing table.

RGP 2.7 has no export for its instruction timing view, so the table is copied out of the GUI: open the trace (RGP under
Xvfb works: `QT_QPA_PLATFORM=xcb DISPLAY=:N RadeonGPUProfiler trace.rgp`), Events > Instruction timing, click into the
table, Ctrl+A, Ctrl+C, then read the X clipboard (tools/rgp_clip.c: `DISPLAY=:N ./rgp_clip > table.tsv`). Columns:
line, opcode, operands, hit count, instruction cost (%), latency.

Per-wave counts divide by the hit count of the shader's first instruction (every sampled wave runs it once), so loop
bodies count once per iteration. The cost column is RGP's share of the sampled waves' cycles: it includes stalls that
other waves hide, so a hot loop's cost overstates what removing it saves (docs/amd-vector-issue-limit.md).

usage: rgp_instruction_mix.py TABLE.tsv [...]
"""
import collections
import re
import sys


def rows(path):
    out = []
    for line in open(path):
        f = [x.strip() for x in line.rstrip('\n').split('\t')]
        if len(f) < 5 or not f[0].isdigit():
            continue
        try:
            hits, cost = int(f[3]), float(f[4])
        except ValueError:
            hits, cost = 0, 0.0
        out.append((int(f[0]), f[1], f[2], hits, cost))
    return out


def kind(op):
    if op.startswith('v_wmma'): return 'wmma'
    if op.startswith('v_cvt_pk_fp8') or op.startswith('v_cvt_pk_bf8'): return 'fp8 pack'
    if op.startswith('v_cvt'): return 'other conversion'
    if op.startswith('v_pk_'): return 'packed 16-bit'
    if op.startswith('v_dual_'): return 'dual issue (VOPD)'
    if op.startswith('v_') and re.search(r'_f32|_f16|_f64|minimummaximum|maxmin_num|minmax_num|fma_mix', op): return 'float'
    if op.startswith('v_'): return 'integer/bit/move/compare'
    if op.startswith('ds_'): return 'LDS'
    if op.startswith(('buffer_', 'global_', 'image_', 'scratch_', 'tbuffer_')): return 'vector memory'
    if op.startswith(('s_wait', 's_barrier', 's_delay_alu', 's_clause', 's_nop')): return 'waits/barriers/hints'
    if op.startswith('s_'): return 'scalar'
    return 'other'


def report(path):
    r = rows(path)
    waves = next(h for _, op, _, h, _ in r if h and not op.startswith('_'))
    by_kind = collections.defaultdict(lambda: [0, 0.0])
    by_op = collections.defaultdict(lambda: [0, 0.0])
    valu = 0
    for _, op, _, h, c in r:
        if not h:
            continue
        by_kind[kind(op)][0] += h; by_kind[kind(op)][1] += c
        by_op[op][0] += h; by_op[op][1] += c
        if op.startswith('v_'): valu += h
    wmma = by_kind['wmma'][0]
    print(f'{path}: {waves} sampled waves, {valu / waves:.0f} VALU and {wmma / waves:.0f} WMMA per wave '
          f'({valu / max(wmma, 1):.1f} VALU per WMMA)')
    for k, (h, c) in sorted(by_kind.items(), key=lambda x: -x[1][0]):
        print(f'  {k:26} {h / waves:7.0f} per wave  {100 * h / max(valu, 1) if k not in ("LDS", "vector memory", "scalar", "waits/barriers/hints", "other") else float("nan"):5.1f}% of VALU  cost {c:5.1f}%')
    print('  highest-cost opcodes:')
    for op, (h, c) in sorted(by_op.items(), key=lambda x: -x[1][1])[:15]:
        print(f'    {op:30} {h / waves:7.0f} per wave  cost {c:5.1f}%')
    # loops and blocks: the regions between labels, by cost
    regions, cur = [], None
    for line, op, _, h, c in r:
        if not h and (op.startswith('_L') or op.startswith('BB') or op.startswith('_amdgpu')):
            cur = [op, line, 0.0, 0, 0, 0]; regions.append(cur); continue
        if cur is None:
            continue
        cur[2] += c; cur[3] = max(cur[3], h)
        cur[4] += op.startswith('v_wmma'); cur[5] += op.startswith('v_')
    print('  costliest regions (label, first line, cost, iterations per wave, static VALU, static WMMA):')
    for name, line, c, h, wm, v in sorted(regions, key=lambda x: -x[2])[:8]:
        print(f'    {name:16} {line:5}  {c:5.1f}%  {h / waves:5.1f}  {v:5}  {wm:4}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        report(p)
