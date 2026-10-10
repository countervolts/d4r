#!/usr/bin/env python3
"""Compare the own K input kernel with the translated reference kernel on a captured frame, optionally with the motion
and depth replaced by synthetic fields (the captured scenes barely move).

usage: test_in_synth.py CAPTURE_DIR FRAME WORK_DIR KVK REF_SPV [seed | none]
"""
import glob, os, re, subprocess, sys
import numpy as np
K = 'hiluma_engine_input_depthinv_mvlo_hdr_v2_rel'
cap, frame, work, kvk, ref_spv = sys.argv[1:6]
seed = sys.argv[6] if len(sys.argv) > 6 else '1'
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{work}/ref', exist_ok=True)
seqs = sorted({int(re.match(r'launch-(\d+)-', os.path.basename(f)).group(1)) for f in glob.glob(f'{cap}/dump/launch-*-{K}-arg*')})
seq = seqs[int(frame)]
res = {}
for f in glob.glob(f'{cap}/dump/launch-{seq:03d}-{K}-arg*'):
    m = re.search(r'-arg(\d+)-(texture|surface|buffer)-(.*)-fmt(-?\d+)\.bin$', f)
    res[int(m.group(1))] = [m.group(2), m.group(3), int(m.group(4)), f]
override = ''
if seed != 'none':
    rng = np.random.default_rng(int(seed))
    def smooth(h, w, cells, amp):
        g = rng.uniform(-amp, amp, (cells + 1, cells + 1))
        y = np.linspace(0, cells, h)[:, None]; x = np.linspace(0, cells, w)[None, :]
        y0 = np.minimum(y.astype(int), cells - 1); x0 = np.minimum(x.astype(int), cells - 1); fy = y - y0; fx = x - x0
        return g[y0, x0] * (1 - fy) * (1 - fx) + g[y0, x0 + 1] * (1 - fy) * fx + g[y0 + 1, x0] * fy * (1 - fx) + g[y0 + 1, x0 + 1] * fy * fx
    w, h, c = (int(x) for x in res[184][1].split('x'))
    mv = np.stack([smooth(h, w, 6, 3.0), smooth(h, w, 6, 3.0)], axis=-1)
    mv[rng.uniform(size=(h, w)) < 0.02] += rng.uniform(-8, 8, 2)
    mv.astype(np.float16).tofile(f'{work}/mv.bin'); res[184][3] = f'{work}/mv.bin'
    w, h, c = (int(x) for x in res[192][1].split('x'))
    depth = 0.3 + 0.1 * smooth(h, w, 5, 1.0); depth[smooth(h, w, 12, 1.0) > 0.35] += 0.4
    depth.astype(np.float32).tofile(f'{work}/depth.bin'); res[192][3] = f'{work}/depth.bin'
    override = f'184={work}/mv.bin,192={work}/depth.bin'
rd = f'{cap}/replay/replay-{seq:06d}-{K}'
grid = next(l for l in open(f'{rd}/manifest.txt') if l.startswith('launch')).split()
n = os.path.getsize(res[240][3])
open(f'{work}/ref/zero.bin', 'wb').write(bytes(n))
lines = [f'param {rd}/args.bin', f'grid {grid[1]} {grid[2]}']
for off in (168, 176, 184, 192, 200, 208, 216, 224):
    kind, shape, fmt, f = res[off]; w, h, c = (int(x) for x in shape.split('x'))
    # NGX's texture objects: history colour and token feature linear, the rest point
    lines.append(f'tex {off} {f} {w} {h} {c} {32 if fmt == 32 else 16} {"linear" if off in (176, 224) else "nearest"}')
lines += [f'buf 240 {work}/ref/zero.bin 0', f'bufout 240 {work}/ref/out.bin']
open(f'{work}/ref/manifest', 'w').write('\n'.join(lines) + '\n')
subprocess.run([kvk, ref_spv, f'{work}/ref/manifest', '1', '32'], check=True, capture_output=True)
a = np.fromfile(res[240][3], np.uint16); b = np.fromfile(f'{work}/ref/out.bin', np.uint16)
if seed == 'none':
    print(f'translated reference vs captured buffer: {int((a != b).sum())} of {a.size} values differ')
env = dict(os.environ, IN_REF=f'{work}/ref/out.bin')
if override: env['IN_TEX_OVERRIDE'] = override
print(subprocess.run([sys.executable, f'{HERE}/test_in.py', cap, frame, work, kvk], env=env, capture_output=True, text=True).stdout.strip())
