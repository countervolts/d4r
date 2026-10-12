#!/usr/bin/env python3
"""Dump every mma of a denoiser layer as matrices, from a ZLUDA replay of a captured launch: the oracle for porting
a layer to native shaders.

usage: phase_dump.py CAPTURE STAGE FRAME BX,BY OUT.npz [--phases A-B]

For one thread block, debug stores added to the PTX record every mma's A, B, C and D registers (each mma's last
execution) into padding behind the plane arena; the replay runs through ZLUDA (engine/rr/zluda_replay.py). The
fragments are then put back together with the m16n8k32 layouts (A row-major 16x32 e4m3, B column-major 32x8 e4m3,
C/D 16x8 f16) into one matrix per phase: phase P's mma are grouped into m-tiles by A fragment and into n-tiles by
B fragment, k-steps follow the C <- D chains. OUT.npz holds, per phase p and wave w: p{p}_w{w}_A (M x K, e4m3 codes),
_B (K x N codes), _C (M x N f16 of the first k-step), _D (M x N f16 of the last).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from rr_capture import Capture  # noqa: E402

REG = r'%r\d+'


def kernel_module(kernel):
    for p in sorted(Path('~/.cache/d4r-rr-corpus').expanduser().glob('*.ptx')):
        if f'.entry {kernel}(' in p.read_text(errors='replace'):
            return p
    raise SystemExit(f'no corpus module for {kernel}')


def mma_statements(lines, start, end):
    """(first line, last line, D, A, B, C) of every mma in [start, end)"""
    out, i = [], start
    while i < end:
        if 'mma.sync.aligned.m16n8k32' in lines[i]:
            j, text = i, ''
            while True:
                text += lines[j]
                if lines[j].rstrip().endswith(';'):
                    break
                j += 1
            groups = [re.findall(REG, g) for g in re.findall(r'\{([^}]*)\}', text)]
            out.append((i + 1, j + 1, *groups[:4]))
            i = j + 1
        else:
            i += 1
    return out


def phases_of(mmas, lines):
    """maximal runs of mma separated only by blank or comment lines"""
    phases, cur = [], []
    for m in mmas:
        if cur:
            gap = [l.strip() for l in lines[cur[-1][1]:m[0] - 1]]
            if any(g and not g.startswith('//') for g in gap):
                phases.append(cur)
                cur = []
        cur.append(m)
    if cur:
        phases.append(cur)
    return phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('stage')
    ap.add_argument('frame', type=int)
    ap.add_argument('block')
    ap.add_argument('out')
    ap.add_argument('--phases', help='first-last (1-based), default all')
    o = ap.parse_args()
    # the capture decides the kernel: a preset D capture runs cuda_dldn_engine_swin_{enc0,dec0}_diamond_wallaby_kernel
    cap = Capture(os.path.expanduser(o.capture))
    launch = [l for l in cap.launches if f'swin_{o.stage}' in l.kernel][o.frame - 1]
    kernel = launch.kernel
    module = kernel_module(kernel)
    lines = module.read_text().replace('\r', '').split('\n')
    k = next(i for i, l in enumerate(lines) if f'.entry {kernel}(' in l)
    e = next((i for i in range(k + 1, len(lines)) if '.entry ' in lines[i]), len(lines))
    phases = phases_of(mma_statements(lines, k, e), lines)
    first, last = (int(x) for x in o.phases.split('-')) if o.phases else (1, len(phases))
    T = launch.block[0] * launch.block[1] * launch.block[2]
    arena_bytes = len(launch.pre(launch.arena_alloc))
    # any argument pointing into the arena locates its base (enc0 has none at 8)
    aptr = next(p for p in launch.pointers if p.alloc == launch.arena_alloc)
    delta8, off8 = aptr.delta, aptr.offset
    bx, by = (int(v) for v in o.block.split(','))
    # one probe slot per register per mma: after the mma, all of A, B, C, D (C read after the mma is still C
    # unless C and D share registers, which the k-step chains never do for the first step)
    slots, inserts = [], {}
    pname = f'{kernel}_param_0'
    for p in range(first, last + 1):
        for (l0, l1, D, A, B, C) in phases[p - 1]:
            code = []
            for role, regs in (('D', D), ('A', A), ('B', B), ('C', C)):
                for r in regs:
                    s = len(slots)
                    slots.append((p, l0, role, r))
                    code.append(f'mov.b32 %dv, {r}; mad.wide.u32 %da, %dt, 4, %db; '
                                f'add.s64 %da, %da, {4 * T * s}; @%dq2 st.global.u32 [%da], %dv;')
            inserts[l1] = ('{ .reg .b64 %da, %db; .reg .b32 %dv, %dt, %dx, %dy; .reg .pred %dq, %dq2;\n'
                           f'mov.u32 %dx, %ctaid.x; mov.u32 %dy, %ctaid.y; setp.eq.u32 %dq, %dx, {bx}; '
                           f'setp.eq.and.u32 %dq2, %dy, {by}, %dq;\n'
                           f'mov.u32 %dt, %tid.z; mov.u32 %dx, %tid.y; mov.u32 %dy, %tid.x; '
                           f'mad.lo.u32 %dt, %dt, {launch.block[1]}, %dx; mad.lo.u32 %dt, %dt, {launch.block[0]}, %dy;\n'
                           f'ld.param.u64 %db, [{pname}+{off8}]; cvta.to.global.u64 %db, %db; '
                           f'add.s64 %db, %db, {arena_bytes - delta8};\n' + '\n'.join(code) + '\n}')
    out = list(lines)
    for l1 in sorted(inserts, reverse=True):
        out[l1:l1] = [inserts[l1]]
    work = ROOT / 'build/rr' / f'phase-{o.stage}'
    work.mkdir(parents=True, exist_ok=True)
    (work / 'probe.ptx').write_text('\n'.join(out))
    pad = 4 * T * len(slots)
    env = dict(os.environ, D4R_ZLUDA_WMMA='1', D4R_ZLUDA_WMMA_FP8='1', D4R_ZLUDA_WMMA_FP8_NATIVE='1',
               D4R_ZLUDA_IMPLICIT_MAX_BLOCK='256')
    cmd = ['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=24G', '-p', 'MemorySwapMax=0']
    cmd += [f'--setenv={k_}={v_}' for k_, v_ in env.items() if k_.startswith('D4R_ZLUDA')]
    r = subprocess.run(cmd + [sys.executable, str(HERE / 'zluda_replay.py'), os.path.expanduser(o.capture),
                              f'swin_{o.stage}' + ('_diamond' if 'diamond' in kernel else ''), str(o.frame), str(work),
                              '--ptx', str(work / 'probe.ptx'),
                              '--pad', str(pad)], capture_output=True, text=True)
    if 'failed' in r.stdout + r.stderr:
        sys.exit(r.stdout[-2000:] + r.stderr[-3000:])
    raw = np.frombuffer((work / 'arena.out.bin').read_bytes()[arena_bytes:], np.uint32).reshape(len(slots), T)
    val = {(p, l0, role, r): raw[s] for s, (p, l0, role, r) in enumerate(slots)}
    result = {}
    waves = T // 32
    for p in range(first, last + 1):
        mm = phases[p - 1]
        # k-step chains: C of a later mma = D of an earlier one
        dset = {tuple(m[2]): n for n, m in enumerate(mm)}
        prev = {n: dset.get(tuple(m[5])) for n, m in enumerate(mm)}
        step = {}
        for n in range(len(mm)):
            s_, q = 0, n
            while prev[q] is not None:
                q = prev[q]
                s_ += 1
            step[n] = s_
        ksteps = max(step.values()) + 1
        a_keys = list(dict.fromkeys(tuple(m[3]) for m in mm))
        b_keys = list(dict.fromkeys(tuple(m[4]) for m in mm))
        # m-tile of an A fragment: its index among the A fragments of the same k-step
        a_tile = {}
        for s_ in range(ksteps):
            for t_, key in enumerate(dict.fromkeys(tuple(m[3]) for n, m in enumerate(mm) if step[n] == s_)):
                a_tile[(s_, key)] = t_
        b_tile = {}
        for s_ in range(ksteps):
            for t_, key in enumerate(dict.fromkeys(tuple(m[4]) for n, m in enumerate(mm) if step[n] == s_)):
                b_tile[(s_, key)] = t_
        mt = max(t for (s_, _), t in a_tile.items()) + 1
        nt = max(t for (s_, _), t in b_tile.items()) + 1
        for w in range(waves):
            lanes = np.arange(32) + 32 * w
            A = np.zeros((16 * mt, 32 * ksteps), np.uint8)
            B = np.zeros((32 * ksteps, 8 * nt), np.uint8)
            C = np.zeros((16 * mt, 8 * nt), np.uint16)
            D = np.zeros((16 * mt, 8 * nt), np.uint16)
            for n, (l0, l1, Dr, Ar, Br, Cr) in enumerate(mm):
                s_ = step[n]
                i, j = a_tile[(s_, tuple(Ar))], b_tile[(s_, tuple(Br))]
                for lane in range(32):
                    g, t = lane >> 2, lane & 3
                    av = [int(val[(p, l0, 'A', r)][lanes[lane]]) for r in Ar]
                    bv = [int(val[(p, l0, 'B', r)][lanes[lane]]) for r in Br]
                    for q in range(4):          # A regs: (g, k 4t..), (g+8, 4t..), (g, 16+4t..), (g+8, 16+4t..)
                        row = 16 * i + g + 8 * (q & 1)
                        col = 32 * s_ + 4 * t + 16 * (q >> 1)
                        for byte in range(4):
                            A[row, col + byte] = (av[q] >> (8 * byte)) & 0xff
                    for q in range(2):          # B regs: column g, k 4t.. and 16+4t..
                        for byte in range(4):
                            B[32 * s_ + 4 * t + 16 * q + byte, 8 * j + g] = (bv[q] >> (8 * byte)) & 0xff
                    for q in range(2):          # C/D regs: row g (+8), columns 2t, 2t+1
                        row = 16 * i + g + 8 * q
                        if s_ == 0:
                            cv = int(val[(p, l0, 'C', Cr[q])][lanes[lane]])
                            C[row, 8 * j + 2 * t] = cv & 0xffff
                            C[row, 8 * j + 2 * t + 1] = cv >> 16
                        if s_ == ksteps - 1:
                            dv = int(val[(p, l0, 'D', Dr[q])][lanes[lane]])
                            D[row, 8 * j + 2 * t] = dv & 0xffff
                            D[row, 8 * j + 2 * t + 1] = dv >> 16
            for name, mat in (('A', A), ('B', B), ('C', C), ('D', D)):
                result[f'p{p}_w{w}_{name}'] = mat
        print(f'phase {p}: {len(mm)} mma, {mt} m-tiles x {nt} n-tiles x {ksteps} k-steps -> '
              f'{16 * mt}x{8 * nt}x{32 * ksteps}, lines {mm[0][0]}-{mm[-1][1]}')
    np.savez_compressed(o.out, **result)


if __name__ == '__main__':
    main()
