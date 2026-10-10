#!/usr/bin/env python3
"""Run the engine's own preset M downsample stage (down_m.comp) on a captured launch and compare the output.

usage: test_down_m.py CAPTURE_DIR FRAME WORK_DIR KVK [repeats]
"""
import glob, os, re, struct, subprocess, sys
import numpy as np
K = 'rrlite_downsample_kernel_static_hdr'
cap, frame, work, kvk = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
reps = sys.argv[5] if len(sys.argv) > 5 else '1'
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(work, exist_ok=True)
seq = sorted({int(re.match(r'launch-(\d+)-', os.path.basename(f)).group(1)) for f in glob.glob(f'{cap}/dump/launch-*-{K}-arg*')})[frame]
res = {int(re.search(r'-arg(\d+)-', f).group(1)): f for f in glob.glob(f'{cap}/dump/launch-{seq:03d}-{K}-arg*')}
P = open(f'{cap}/replay/replay-{seq:06d}-{K}/args.bin', 'rb').read()
i2 = lambda o: struct.unpack_from('<2i', P, o); f2 = lambda o: struct.unpack_from('<2f', P, o)
expo = float(np.fromfile(res[48], np.float16)[0])
open(f'{work}/params.bin', 'wb').write(struct.pack('<8i4ff', *i2(0), *i2(40), *i2(64), *i2(72), *f2(80), *f2(88), expo))
dims = lambda f: [int(x) for x in re.search(r'-(\d+x\d+x\d+)-fmt', f).group(1).split('x')]
cw, ch, cc = dims(res[56]); ow, oh, oc = dims(res[32])
open(f'{work}/manifest', 'w').write('\n'.join([
    f'param {work}/params.bin', f'grid {(ow + 7) // 8} {(oh + 63) // 64}', f'tex 1 {res[56]} {cw} {ch} {cc} 16 nearest',
    f'surf 2 {ow} {oh} 4 16 - {work}/out.bin', f'lut {res[24]}']) + '\n')
spv = f'{work}/down_m.spv'
subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', f'{HERE}/down_m.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
r = subprocess.run([kvk, spv, f'{work}/manifest', reps, '64'], capture_output=True, text=True)
if r.returncode:
    print('FAILED', r.stderr[-400:]); sys.exit(1)
for l in r.stdout.split('\n'):
    if 'us per' in l: print(l)
a = np.fromfile(res[32], np.float16).astype(np.float64); b = np.fromfile(f'{work}/out.bin', np.float16).astype(np.float64)
d = np.abs(a - b)
print(f'frame {frame} (launch {seq}) {ow}x{oh}: psnr {10 * np.log10(np.abs(a).max() ** 2 / max((d ** 2).mean(), 1e-30)):.1f} dB, '
      f'identical {100 * (np.fromfile(res[32], np.uint16) == np.fromfile(f"{work}/out.bin", np.uint16)).mean():.1f}%, max |diff| {d.max():.4g}')
