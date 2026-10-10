#!/usr/bin/env python3
"""Run the engine's own preset M reconstruction (post_m.comp) on a captured launch and compare its outputs.

usage: test_post_m.py CAPTURE_DIR FRAME WORK_DIR KVK [repeats]
"""
import glob, os, re, struct, subprocess, sys
import numpy as np
cap, frame, work, kvk = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
# the kernel name can be overridden; post_3_2 (reconstruction/render < 3) needs the larger shared tile
K = os.environ.get('POST_KERNEL') or next((k for k in ('rrlite_post_3_1_mvlo_hdr_folded', 'rrlite_post_3_2_mvlo_hdr_folded')
    if glob.glob(f'{cap}/replay/replay-*-{k}')), 'rrlite_post_3_1_mvlo_hdr_folded')
TILE = ['-DTILE_W=15', '-DTILE_H=15'] if '3_2' in K else []
reps = sys.argv[5] if len(sys.argv) > 5 else '1'
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(work, exist_ok=True)
seq = sorted({int(re.match(r'launch-(\d+)-', os.path.basename(f)).group(1)) for f in glob.glob(f'{cap}/dump/launch-*-{K}-arg*')})[frame]
res = {int(re.search(r'-arg(\d+)-', f).group(1)): f for f in glob.glob(f'{cap}/dump/launch-{seq:03d}-{K}-arg*')}
P = open(f'{cap}/replay/replay-{seq:06d}-{K}/args.bin', 'rb').read()
i2 = lambda o: struct.unpack_from('<2i', P, o); f2 = lambda o: struct.unpack_from('<2f', P, o)
expo = lambda o: float(np.fromfile(res[o], np.float16)[0])
hw, hh = i2(140); rw, rh = i2(148)
open(f'{work}/params.bin', 'wb').write(struct.pack('<2i2i2i2f2f2ffIff', *i2(8), hw, hh, rw, rh, *f2(124), *f2(132), *f2(160),
                                                   struct.unpack_from('<f', P, 120)[0], P[168], expo(16), expo(24)))
def tex(binding, off, ch, bits, mode, w=rw, h=rh): return f'tex {binding} {res[off]} {w} {h} {ch} {bits} {mode}'
lines = [f'param {work}/params.bin', f'grid {(hw + 15) // 16} {(hh + 15) // 16}',
         tex(1, 0, 4, 16, 'nearest'), tex(2, 40, 4, 8, 'nearest'), tex(3, 32, 4, 8, 'nearest'), tex(4, 48, 4, 16, 'nearest'),
         tex(5, 56, 4, 16, 'linear', hw, hh), tex(6, 64, 2, 16, 'nearest'), tex(7, 72, 2, 16, 'nearest'), tex(8, 80, 2, 16, 'linear'),
         f'surf 9 {hw} {hh} 4 16 - {work}/hist.bin', f'surf 10 {(hw + 1) // 2} {(hh + 1) // 2} 4 32 - {work}/half.bin',
         f'outbuf {work}/packed.bin {hw * hh * 4}']
open(f'{work}/manifest', 'w').write('\n'.join(lines) + '\n')
spv = f'{work}/post_m.spv'
subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *TILE, f'{HERE}/post_m.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
r = subprocess.run([kvk, spv, f'{work}/manifest', reps, '64'], capture_output=True, text=True)
if r.returncode:
    print('FAILED', r.stderr[-600:]); sys.exit(1)
for l in r.stdout.split('\n'):
    if 'us per' in l: print(l)
print(f'frame {frame} (launch {seq}) {K} {rw}x{rh} -> {hw}x{hh}, reset {P[168]}, exposure {expo(16):.4g} / {expo(24):.4g}')
a = np.fromfile(res[96], np.float16).astype(np.float64); b = np.fromfile(f'{work}/hist.bin', np.float16).astype(np.float64)
d = np.abs(a - b)
print(f'  new history   psnr {10 * np.log10(np.abs(a).max() ** 2 / max((d ** 2).mean(), 1e-30)):5.1f} dB  identical {100 * (np.fromfile(res[96], np.uint16) == np.fromfile(f"{work}/hist.bin", np.uint16)).mean():5.1f}%  max |diff| {d.max():.4g}')
def unpack(f):
    w = np.fromfile(f, np.uint32)
    return np.stack([w & 2047, (w >> 11) & 2047, w >> 22], -1).astype(np.float64)
a, b = unpack(res[104]), unpack(f'{work}/packed.bin')
d = np.abs(a - b)
print(f'  packed output identical {100 * (d == 0).mean():5.1f}%  within one step {100 * (d <= 1).mean():5.1f}%  psnr {10 * np.log10(2047.0 ** 2 / max((d ** 2).mean(), 1e-30)):5.1f} dB  max step {int(d.max())}')
if 112 in res:      # the half-resolution output; the launch dumps do not always carry it
    codes = np.fromfile(res[112], np.uint32)
    a = np.stack([codes & 1023, (codes >> 10) & 1023, (codes >> 20) & 1023], -1).astype(np.float32) / np.float32(1023)
    b = np.fromfile(f'{work}/half.bin', np.float32).reshape(-1, 4)[:, :3]
    d = np.abs(a.astype(np.float64) - b)
    exact = a.view(np.uint32) == b.view(np.uint32)
    print(f'  half output   psnr {10 * np.log10(1.0 / max((d ** 2).mean(), 1e-30)):5.1f} dB  identical {100 * exact.mean():5.1f}%  max |diff| {d.max():.4g}')
