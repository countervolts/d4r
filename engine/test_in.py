#!/usr/bin/env python3
"""Run the engine's own K input kernel on a captured launch of NVIDIA's and compare the network input it writes.

usage: test_in.py CAPTURE_DIR FRAME WORK_DIR KVK [repeats]
"""
import glob, os, re, struct, subprocess, sys
import numpy as np
LDR = os.environ.get('LDR') == '1'      # the *_ldr_* kernel variant and the shader's LDR path
K = 'hiluma_engine_input_depthinv_mvlo_ldr_v2_rel' if LDR else 'hiluma_engine_input_depthinv_mvlo_hdr_v2_rel'
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
P = open(f'{cap}/replay/replay-{seq:06d}-{K}/args.bin', 'rb').read()
f2 = lambda o: struct.unpack_from('<2f', P, o)
i2 = lambda o: struct.unpack_from('<2i', P, o)
f1 = lambda o: struct.unpack_from('<f', P, o)[0]
def expo_tex(off):
    if off not in res: return 1.0       # LDR: no exposure textures
    v = float(np.fromfile(res[off][3], np.float16)[0])
    return 1.0 if v == 0 else v
F = np.float32
a, b, c = expo_tex(208), expo_tex(200), expo_tex(216)
f152, f158 = F(f1(144)) * F(a), F(f1(144)) * F(b)
expo = F(f1(156)) * f158; expo = F(1) if expo == 0 else expo
expo_in = F(f1(148)) * F(c) / f158
ratio = f158 / f152
blob = struct.pack('<14f', *f2(40), *f2(24), *f2(8), *f2(16), *f2(120), *f2(136), *f2(128))
blob += struct.pack('<12i', *i2(72), *i2(80), *i2(88), *i2(96), *i2(104), *i2(112))
blob += struct.pack('<iI4f', struct.unpack_from('<i', P, 64)[0], P[32], float(expo), float(expo_in), float(ratio), f1(160))
open(f'{work}/params.bin', 'wb').write(blob)
tokW = struct.unpack_from('<i', P, 64)[0]
kind, shape, fmt, fref = res[240]
nbytes = os.path.getsize(fref)
tokH = 256 if tokW == 320 else None
# IN_TEX_OVERRIDE="184=FILE,192=FILE" replaces captured textures; IN_REF compares against that file (the translated
# reference kernel's output on the same inputs) instead of the captured buffer.
for kv in filter(None, os.environ.get('IN_TEX_OVERRIDE', '').split(',')):
    o, fn = kv.split('='); res[int(o)] = res[int(o)][:3] + (fn,)
fref = os.environ.get('IN_REF', fref)
lines = [f'param {work}/params.bin']
dims = {}
for i, (off, filt) in enumerate(((168, 'nearest'), (176, 'linear'), (184, 'nearest'), (192, 'nearest'), (224, 'linear'))):
    kind, shape, fmt, f = res[off]; w, h, ch = (int(x) for x in shape.split('x'))
    dims[off] = (w, h)
    lines.append(f'tex {i + 1} {f} {w} {h} {ch} {32 if fmt == 32 else 16} {filt}')
tokH = dims[224][1]
lines.insert(1, f'grid {tokW // 8} {tokH // 8}')
lines.append(f'outbuf {work}/out.bin {tokW * tokH * 32}')
open(f'{work}/manifest', 'w').write('\n'.join(lines) + '\n')
spv = f'{work}/input_k.spv'
subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *(['-DLDR=1'] if LDR else []), f'{HERE}/input_k.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
# K requires a 32-lane subgroup (the reconstruction's 2x2 "bright" quad is a subgroup shuffle and the
# network layers use 16x16x16 cooperative matrices); wave64 changes both.
r = subprocess.run([kvk, spv, f'{work}/manifest', reps, '32'], capture_output=True, text=True)
if r.returncode:
    print('FAILED', r.stderr[-400:]); sys.exit(1)
for l in r.stdout.split('\n'):
    if 'us per' in l: print(l)
ref = np.fromfile(fref, np.float16)[:tokW * tokH * 16].astype(np.float64).reshape(tokH, tokW, 4, 4)
got = np.fromfile(f'{work}/out.bin', np.float16)[:tokW * tokH * 16].astype(np.float64).reshape(tokH, tokW, 4, 4)
print(f'frame {frame} (launch {seq}), tokens {tokW}x{tokH}, reset {P[32]}, exposure {float(expo):.4g} / {float(expo_in):.4g}, ratio {float(ratio):.6g}')
for ch, name in enumerate(('current luma', 'history luma', 'token feature', 'mismatch')):
    a, b = ref[..., ch], got[..., ch]
    d = np.abs(a - b); peak = max(np.abs(a).max(), 1e-9)
    print(f'  {name:14s} psnr {10 * np.log10(peak ** 2 / max((d ** 2).mean(), 1e-30)):6.1f} dB  max |diff| {d.max():.4g}  mean |ref| {np.abs(a).mean():.4g}  non-finite {int((~np.isfinite(b)).sum())}')
