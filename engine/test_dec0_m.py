#!/usr/bin/env python3
"""Run the engine's own preset M dec0 stage (dec0_m.comp) on a captured launch and compare its four surfaces.

usage: test_dec0_m.py CAPTURE_DIR FRAME WORK_DIR KVK [repeats]
"""
import glob, os, re, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m_model as mm
import model_enc3 as me
import swin_model as sm
from test_m import codes, token_codes
K = 'rrlite_dec0_4x4_folded'
cap, frame, work, kvk = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
reps = sys.argv[5] if len(sys.argv) > 5 else '1'
os.makedirs(work, exist_ok=True)
seq = sorted({int(re.match(r'launch-(\d+)-', os.path.basename(f)).group(1)) for f in glob.glob(f'{cap}/dump/launch-*-{K}-arg*')})[frame]
res = {int(re.search(r'-arg(\d+)-', f).group(1)): f for f in glob.glob(f'{cap}/dump/launch-{seq:03d}-{K}-arg*')}
d = f'{cap}/replay/replay-{seq:06d}-{K}'
P = open(d + '/args.bin', 'rb').read()
allocs = {}
for line in open(d + '/manifest.txt'):
    q = line.split()
    if q and q[0] == 'alloc': allocs[int(q[1])] = (int(q[2], 16), np.fromfile(f'{d}/alloc-{q[1]}.bin', np.uint8))
find = lambda ptr: next((a, b) for a, b in allocs.values() if a <= ptr < a + b.size)
wptr, inp = struct.unpack_from('<Q', P, 0)[0], struct.unpack_from('<Q', P, 24)[0]
i2 = lambda o: struct.unpack_from('<2i', P, o); f2 = lambda o: struct.unpack_from('<2f', P, o)
gw, gh = i2(16); rw, rh = i2(112)
wa, wbuf = find(wptr); wbuf = wbuf[wptr - wa:]
ab, arena = find(inp)
token_codes(arena, inp - ab, gw // 2, gh // 2, 64).tofile(f'{work}/tokens.bin')
# plain weights [64][128] and bias; the stage uses columns 32 q + j, j < 20
W = wbuf[sm.woff_table(0, 512, 1024, 64, 128)]
bias = wbuf[8192:8192 + 256].view(np.float16)
cols = (np.arange(4)[:, None] * 32 + np.arange(20)[None, :]).ravel()
np.concatenate([W[:, cols].ravel(), np.zeros(64, np.uint8)]).tofile(f'{work}/w8.bin')
np.concatenate([np.tile(bias[cols][None, :], (16, 1)).ravel(), np.zeros(32, np.float16)]).tofile(f'{work}/w16.bin')
open(f'{work}/params.bin', 'wb').write(struct.pack('<2iiI2f2f2f2f2f2f', rw, rh, gw // 2, P[120], *f2(72), *f2(88), float(i2(96)[0]), float(i2(96)[1]), *f2(80), *f2(160), *f2(104)))
lines = [f'param {work}/params.bin', f'grid {gw // 8} {gh // 8}',
         f'tex 1 {res[56]} {rw} {rh} 2 16 nearest', f'tex 2 {res[64]} {rw} {rh} 4 9 linear']
outs = ((128, 9, np.int8, 127.0, 'feature (SNORM8)'), (136, 8, np.uint8, 255.0, 'gates (UNORM8)'), (144, 8, np.uint8, 255.0, 'mix (UNORM8)'), (152, 16, np.float16, 1.0, 'covariance (f16)'))
for i, (off, bits, dt, scale, name) in enumerate(outs):
    lines.append(f'surf {3 + i} {rw} {rh} 4 {bits} - {work}/out-{off}.bin')
lines += [f'lut {work}/tokens.bin', f'lut {work}/w8.bin', f'lut {work}/w16.bin']
open(f'{work}/manifest', 'w').write('\n'.join(lines) + '\n')
spv = f'{work}/dec0_m.spv'
subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', f'{HERE}/dec0_m.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
r = subprocess.run([kvk, spv, f'{work}/manifest', reps, '32'], capture_output=True, text=True)
if r.returncode:
    print('FAILED', r.stderr[-600:]); sys.exit(1)
for l in r.stdout.split('\n'):
    if 'us per' in l: print(l)
print(f'frame {frame} (launch {seq}), reset {P[120]}')
for off, bits, dt, scale, name in outs:
    a = np.fromfile(res[off], dt).astype(np.float64) / scale; b = np.fromfile(f'{work}/out-{off}.bin', dt).astype(np.float64) / scale
    dd = np.abs(a - b)
    print(f'  {name:18s} psnr {10 * np.log10(max(np.abs(a).max(), 1e-9) ** 2 / max((dd ** 2).mean(), 1e-30)):5.1f} dB  identical {100 * (dd == 0).mean():5.1f}%  '
          f'within one step {100 * (dd <= 1.0 / scale + 1e-9).mean() if scale > 1 else 100 * (dd < 1e-3 * np.abs(a).max()).mean():5.1f}%  max |diff| {dd.max():.4g}')
