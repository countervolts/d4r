#!/usr/bin/env python3
"""Build a kvk manifest for one captured launch and compare kvk's outputs with the capture.

usage: kcase.py make CAPTURE_DIR KERNEL FRAME OUTDIR [--tex OFF:linear ...]   (FRAME counts that kernel's launches from 0)
       kcase.py cmp OUTDIR
CAPTURE_DIR holds dump/ (D4R_CUDA_LAUNCH_DUMP_DIR) and replay/ (D4R_CUDA_REPLAY_DUMP_DIR) of the same run.
"""
import glob, os, re, sys
import numpy as np
PING = set()

def launches(cap, kernel):
    return sorted({int(re.match(r'launch-(\d+)-', os.path.basename(f)).group(1)) for f in glob.glob(f'{cap}/dump/launch-*-{kernel}-arg*')})

def res(cap, kernel, seq):
    out = {}
    for f in glob.glob(f'{cap}/dump/launch-{seq:03d}-{kernel}-arg*'):
        m = re.search(r'-arg(\d+)-(texture|surface|buffer)-(.*)-fmt(-?\d+)\.bin$', f)
        out[int(m.group(1))] = (m.group(2), m.group(3), int(m.group(4)), f)
    return out

if sys.argv[1] == 'make':
    cap, kernel, frame, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    linear = {int(a.split(':')[0]) for a in sys.argv[6:] if a.endswith(':linear')}
    os.makedirs(outdir, exist_ok=True)
    seqs = launches(cap, kernel); seq = seqs[frame]
    cur = res(cap, kernel, seq)
    man = open(f'{cap}/replay/replay-{seq:06d}-{kernel}/manifest.txt').read().split('\n')
    grid = next(l for l in man if l.startswith('launch')).split()
    lines = [f'param {cap}/replay/replay-{seq:06d}-{kernel}/args.bin', f'grid {grid[1]} {grid[2]}']
    ptr = {int(l.split()[1]): int(l.split()[3]) for l in man if l.startswith('pointer')}
    expect = []
    for off, (kind, shape, fmt, f) in sorted(cur.items()):
        if kind == 'buffer':
            lines.append(f'buf {off} {f} {ptr[off]}'); continue
        w, h, c = (int(x) for x in shape.split('x'))
        bits = {32: 32, 194: 8, 192: 8, 193: 8, 200: 9, 198: 9, 199: 9}.get(fmt, 16)    # 19x: UNORM8, 198..200: SNORM8
        if kind == 'texture':
            lines.append(f'tex {off} {f} {w} {h} {c} {bits} {"linear" if off in linear else "nearest"}')
        else:
            # the surface's contents before this launch: its dump from the latest earlier launch that had the same data layout
            init = '-'
            for s in reversed([s for s in seqs if s < seq]):
                prev = res(cap, kernel, s)
                # ping-pong surfaces alternate; take the one whose size matches and which is not this frame's input
                if off in prev and prev[off][1] == shape and (s == seqs[frame - 2] if frame >= 2 and off in PING else True):
                    init = prev[off][3]; break
            lines.append(f'surf {off} {w} {h} {c} {bits} {init} {outdir}/out-{off}.bin')
            expect.append((off, f, c, bits))
    open(f'{outdir}/manifest', 'w').write('\n'.join(lines) + '\n')
    open(f'{outdir}/expect', 'w').write('\n'.join(f'{o} {f} {c} {b}' for o, f, c, b in expect) + '\n')
    print('\n'.join(lines))
else:
    outdir = sys.argv[2]
    ok = True
    for l in open(f'{outdir}/expect').read().split('\n'):
        if not l:
            continue
        off, f, c, bits = l.split()
        dt = np.uint16 if bits == '16' else np.uint32
        a = np.fromfile(f, dtype=dt); b = np.fromfile(f'{outdir}/out-{off}.bin', dtype=dt)
        diff = a != b
        fa = a.view(np.float16 if bits == '16' else np.float32).astype(np.float64); fb = b.view(np.float16 if bits == '16' else np.float32).astype(np.float64)
        with np.errstate(invalid='ignore'):
            d = np.abs(fa - fb); d = d[np.isfinite(d)]
        ok &= not diff.any()
        print(f'surface {off}: {a.size} values, {int(diff.sum())} differ ({diff.mean() * 100:.4f}%), max |diff| {d.max() if d.size else 0:.6g}, mean |ref| {np.nanmean(np.abs(fa[np.isfinite(fa)])):.4g}')
    print('BYTE-IDENTICAL' if ok else 'DIFFERENT')
