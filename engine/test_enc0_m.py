#!/usr/bin/env python3
"""Run the engine's own preset M input stage (enc0_m.comp) on a captured launch and compare the tokens and the three
motion surfaces.

usage: test_enc0_m.py CAPTURE_DIR FRAME WORK_DIR KVK [repeats]
The half-resolution previous result (texture 152) is not in the dumps; frames after the first use the engine's
own post stage output if WORK_DIR/half-prev.bin exists, zeros otherwise.
"""
import glob, os, re, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m_model as mm
import model_enc3 as me
import swin_model as sm
from test_m import token_codes, report
K = 'rrlite_enc0_4x4_mvlo_hdr_folded'
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
i2 = lambda o: struct.unpack_from('<2i', P, o); f2 = lambda o: struct.unpack_from('<2f', P, o); f1 = lambda o: struct.unpack_from('<f', P, o)[0]
wptr, outp = struct.unpack_from('<Q', P, 0)[0], struct.unpack_from('<Q', P, 48)[0]
wa, wbuf = find(wptr); wbuf = wbuf[wptr - wa:]
rw, rh = i2(56); gw, gh = i2(16)
expo = lambda o: float(np.fromfile(res[o], np.float16)[0])
g = sm.gp(np.arange(64))      # the embedding's columns are in the network's pair order
np.ascontiguousarray(wbuf[sm.woff_table(4096, 512, 2048, 128, 64)][:, g]).tofile(f'{work}/w8.bin')
np.tile(wbuf[12288:12288 + 128].view(np.float16)[g][None, :], (16, 1)).tofile(f'{work}/w16.bin')
wbuf[:4096].tofile(f'{work}/noise.bin')
open(f'{work}/params.bin', 'wb').write(struct.pack('<2i2i2i2f2f2f2f2f2f2fff3Iffffff', rw, rh, *i2(8), gw // 2, gh // 2, *f2(168), *f2(176), *f2(200), *f2(208),
                                                   *f2(192), *f2(184), float(i2(248)[0]), float(i2(248)[1]), f1(268), f1(272), P[216], P[316], i2(312)[0], expo(112), expo(120), *f2(256), *f2(64)))
hw, hh = i2(248)
# The native RGB10 formatted history has RGBA32F backing, not a Vulkan UNORM sampler.
codes = np.fromfile(res[152], np.uint32) if 152 in res else np.zeros((hh // 2) * (hw // 2), np.uint32)
half = f'{work}/half-prev.bin'
values = np.empty((codes.size, 4), np.float32)
for channel, shift in enumerate((0, 10, 20)):
    values[:, channel] = ((codes >> shift) & 1023).astype(np.float32) / np.float32(1023)
values[:, 3] = (codes >> 30).astype(np.float32) / np.float32(3)
values.tofile(half)
lines = [f'param {work}/params.bin', f'grid {(gw + i2(8)[0] + 7) // 8} {(gh + i2(8)[1] + 7) // 8}',
         f'tex 1 {res[80]} {rw} {rh} 4 16 nearest', f'tex 2 {res[96]} {rw} {rh} 2 16 nearest', f'tex 3 {res[136]} {rw} {rh} 1 32 nearest',
         f'tex 4 {res[128]} {rw} {rh} 2 16 linear', f'tex 5 {half} {hw // 2} {hh // 2} 4 32 linear', f'tex 6 {res[160]} {rw} {rh} 4 9 linear']
for i, off in enumerate((280, 288, 296)): lines.append(f'surf {7 + i} {rw} {rh} 2 16 - {work}/mv-{off}.bin')
lines += [f'lut {work}/w8.bin', f'lut {work}/w16.bin', f'lut {work}/noise.bin', f'outbuf {work}/tokens.bin {(gw // 2) * (gh // 2) * 64}']
open(f'{work}/manifest', 'w').write('\n'.join(lines) + '\n')
spv = f'{work}/enc0_m.spv'
subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', f'{HERE}/enc0_m.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
r = subprocess.run([kvk, spv, f'{work}/manifest', reps, '32'], capture_output=True, text=True)
if r.returncode:
    print('FAILED', r.stderr[-600:]); sys.exit(1)
for l in r.stdout.split('\n'):
    if 'us per' in l: print(l)
print(f'frame {frame} (launch {seq}) {rw}x{rh}, tokens {gw // 2}x{gh // 2}, reset {P[316]}, dither frame {i2(312)[0]}')
for off, name in ((280, 'A'), (288, 'B'), (296, 'C')):
    a = np.fromfile(res[off], np.float16).astype(np.float64); b = np.fromfile(f'{work}/mv-{off}.bin', np.float16).astype(np.float64)
    print(f'  motion {name}: identical {100 * (a == b).mean():5.1f}%  max |diff| {np.abs(a - b).max():.4g}')
after = np.fromfile(res[40], np.uint8); ab, _ = find(outp)
ref = token_codes(after, outp - ab, gw // 2, gh // 2, 64)
got = np.fromfile(f'{work}/tokens.bin', np.uint8).reshape(gh // 2, gw // 2, 64)
print('  ' + report('tokens', got, ref))
vis = rh // 2
print('  ' + report('tokens, rows inside the image', got[:vis], ref[:vis]))
