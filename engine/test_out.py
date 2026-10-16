#!/usr/bin/env python3
"""Run the engine's own K output kernel on a captured launch of NVIDIA's and compare the four output surfaces.

usage: test_out.py CAPTURE_DIR FRAME WORK_DIR KVK [repeats]
  CAPTURE_DIR: dump/ and replay/ of a run with the output kernel captured (see vulkan/README.md); KVK: the vulkan/kvk binary.
"""
import glob, os, re, struct, subprocess, sys
import numpy as np
LDR = os.environ.get('LDR') == '1'      # the *_ldr_* kernel variant and the shader's LDR path
K = 'hiluma_engine_output_depthinv_mvlo_ldr_max_v2_rel' if LDR else 'hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel'
cap, frame, work, kvk = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
reps = sys.argv[5] if len(sys.argv) > 5 else '1'
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(work, exist_ok=True)
seqs = sorted({int(re.match(r'launch-(\d+)-', os.path.basename(f)).group(1)) for f in glob.glob(f'{cap}/dump/launch-*-{K}-arg*')})
seq = seqs[frame]
res = {}
for f in glob.glob(f'{cap}/dump/launch-{seq:03d}-{K}-arg*'):
    m = re.search(r'-arg(\d+)-(texture|surface|buffer)-(.*)-fmt(-?\d+)\.bin$', f)
    res[int(m.group(1))] = (m.group(2), m.group(3), int(m.group(4)), f)
rd = f'{cap}/replay/replay-{seq:06d}-{K}'
P = open(f'{rd}/args.bin', 'rb').read()
man = open(f'{rd}/manifest.txt').read().split('\n')
grid = next(l for l in man if l.startswith('launch')).split()
ptr = {int(l.split()[1]): int(l.split()[3]) for l in man if l.startswith('pointer')}
f2 = lambda o: struct.unpack_from('<2f', P, o)
i2 = lambda o: struct.unpack_from('<2i', P, o)
h1 = lambda o: float(np.frombuffer(P, np.float16, 1, o)[0])
def expo_tex(off):
    if off not in res: return 1.0       # LDR: no exposure textures
    v = float(np.fromfile(res[off][3], np.float16)[0])
    return 1.0 if v == 0 else v
eA, eB = expo_tex(264), expo_tex(256)
s160 = struct.unpack_from('<f', P, 160)[0]; s168 = struct.unpack_from('<f', P, 168)[0]
expo = np.float32(s168) * np.float32(s160 * eB); expo = 1.0 if expo == 0 else float(expo)
ratio = float(np.float32(s160 * eB) / np.float32(s160 * eA))
blob = struct.pack('<18f', *f2(40), *f2(16), *f2(56), *f2(48), *f2(64), *f2(0), *f2(8), *f2(144), *f2(152))
blob += struct.pack('<16i', *i2(208), *i2(216), *i2(96), *i2(104), *i2(112), *i2(120), *i2(128), *i2(136))
blob += struct.pack('<2iI', *i2(80), P[32])
blob += struct.pack('<10f', expo, ratio, *struct.unpack_from('<3f', P, 172), h1(188), h1(190), h1(192), h1(184), h1(186))
open(f'{work}/params.bin', 'wb').write(blob)
flags = (P[89], P[90], P[91])
if any(flags):
    print('note: flag bytes 89..91 =', flags, '- the own kernel does not implement these modes')
# OUT_TEX_OVERRIDE="240=FILE,248=FILE" replaces captured textures; OUT_REF_DIR compares against DIR/out-OFF.bin
# (outputs of the translated reference kernel on the same inputs) instead of the captured surfaces.
for kv in filter(None, os.environ.get('OUT_TEX_OVERRIDE', '').split(',')):
    o, fn = kv.split('='); res[int(o)] = res[int(o)][:3] + (fn,)
lines = [f'param {work}/params.bin', f'grid {grid[1]} {grid[2]}']
for i, (off, filt) in enumerate(((200, 'nearest'), (224, 'linear'), (232, 'linear'), (240, 'nearest'), (248, 'nearest'))):
    kind, shape, fmt, f = res[off]; w, h, c = (int(x) for x in shape.split('x'))
    lines.append(f'tex {i + 1} {f} {w} {h} {c} {32 if fmt == 32 else 16} {filt}')
expect = []
for i, off in enumerate((288, 296, 304, 312)):
    kind, shape, fmt, f = res[off]; w, h, c = (int(x) for x in shape.split('x'))
    lines.append(f'surf {i + 6} {w} {h} {c} 16 - {work}/out-{off}.bin')
    expect.append((off, f'{os.environ["OUT_REF_DIR"]}/out-{off}.bin' if 'OUT_REF_DIR' in os.environ else f, c))
for off, name in ((272, 'head'), (280, 'lut')):
    data = np.fromfile(res[off][3], np.uint8)[ptr[off]:]
    data.tofile(f'{work}/{name}.bin'); lines.append(f'lut {work}/{name}.bin')
open(f'{work}/manifest', 'w').write('\n'.join(lines) + '\n')
spv = f'{work}/output_k.spv'
subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *os.environ.get('OUT_DEFS', '').split(), *(['-DLDR=1'] if LDR else []), f'{HERE}/output_k.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
# The reconstruction's 2x2 "bright" quad is a subgroup shuffle: 32 lanes put the pixel rows in the
# subgroup's two halves (x + 16 * (y & 1)), so the test must request the engine's subgroup size.
r = subprocess.run([kvk, spv, f'{work}/manifest', reps, '32'], capture_output=True, text=True)
if r.returncode:
    print('FAILED', r.stderr[-400:]); sys.exit(1)
for l in r.stdout.split('\n'):
    if 'us per' in l: print(l)
names = {288: 'history colour', 296: 'hi-res luma', 304: 'token feature', 312: 'final output'}
print(f'frame {frame} (launch {seq}), exposure scale {expo:.4g}, ratio {ratio:.6g}')
for off, f, c in expect:
    a = np.fromfile(f, np.float16).astype(np.float64); b = np.fromfile(f'{work}/out-{off}.bin', np.float16).astype(np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a - b)[ok]
    peak = np.abs(a[ok]).max()
    psnr = 10 * np.log10(peak ** 2 / max((d ** 2).mean(), 1e-30))
    exact = (np.fromfile(f, np.uint16) == np.fromfile(f'{work}/out-{off}.bin', np.uint16)).mean()
    print(f'  {names[off]:15s} psnr {psnr:6.1f} dB  exact {100 * exact:5.1f}%  max |diff| {d.max():.4g}  mean |diff| {d.mean():.3g}  peak {peak:.3g}  non-finite in own {int((~np.isfinite(b)).sum())}')
