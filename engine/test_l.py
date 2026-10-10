#!/usr/bin/env python3
"""Run the engine's own preset L stages on a captured launch and compare them with its outputs.

usage: test_l.py enc0|dec0 CAPTURE_DIR FRAME WORK_DIR KVK [repeats]
  enc0: layer_m.comp KIND 3 (features, embedding, block, merge) against rrlite_enc0_4x4_mv*_*: the block output
        (dec0's skip input), the merged tokens (the network's input) and the three motion surfaces
  dec0: layer_m.comp KIND 4 (expand, block, head) against rrlite_dec0_4x4: its four surfaces
CAPTURE_DIR as for test_enc0_m.py (replay/ before each launch, dump/ after it, filter `rrlite_`). KVK is
vulkan/kvk built with its `bindings` manifest line. The half-resolution previous result (texture 152) is not in the
dumps: enc0 uses WORK_DIR/half-prev.bin if it exists (zeros otherwise), which only matters after the first frame.
"""
import glob, os, re, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l_model as lm
import model_enc3 as me
from test_m import pack_layer, defines, token_codes, report


def launch(cap, kernel_re, frame):
    """(sequence number, replay dir, dumped resources by parameter offset) of the frame's launch of the kernel"""
    names = sorted({re.match(r'launch-(\d+)-(.*)-arg', os.path.basename(f)).groups() for f in glob.glob(f'{cap}/dump/launch-*-arg*')})
    hits = [(int(s), k) for s, k in names if re.fullmatch(kernel_re, k)]
    if len(hits) <= frame:
        raise SystemExit(f'no launch of {kernel_re} for frame {frame}')
    seq, k = hits[frame]
    res = {int(re.search(r'-arg(\d+)-', f).group(1)): f for f in glob.glob(f'{cap}/dump/launch-{seq:03d}-{k}-arg*')}
    return seq, k, f'{cap}/replay/replay-{seq:06d}-{k}', res


class Replay:
    def __init__(self, d):
        self.P = open(d + '/args.bin', 'rb').read()
        self.allocs = {}
        for line in open(d + '/manifest.txt'):
            q = line.split()
            if q and q[0] == 'alloc':
                self.allocs[int(q[1])] = (int(q[2], 16), np.fromfile(f'{d}/alloc-{q[1]}.bin', np.uint8))

    def at(self, ptr):
        """the replayed allocation bytes from address ptr on"""
        a, b = next((a, b) for a, b in self.allocs.values() if a <= ptr < a + b.size)
        return b[ptr - a:]

    def q(self, o): return struct.unpack_from('<Q', self.P, o)[0]
    def i2(self, o): return struct.unpack_from('<2i', self.P, o)
    def f2(self, o): return struct.unpack_from('<2f', self.P, o)
    def f1(self, o): return struct.unpack_from('<f', self.P, o)[0]


def compile_stage(name, out, extra=()):
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defines(name), *extra, f'{HERE}/layer_m.comp', '-o', out],
                   check=True, stdout=subprocess.DEVNULL)


def run(kvk, spv, manifest, reps):
    r = subprocess.run([kvk, spv, manifest, reps, '32'], capture_output=True, text=True)
    if r.returncode:
        print('FAILED', r.stderr[-800:]); sys.exit(1)
    for l in r.stdout.split('\n'):
        if 'us per' in l: print(l)


def enc0(cap, frame, work, kvk, reps):
    seq, k, d, res = launch(cap, r'rrlite_enc0_4x4_mv(hi|lo)_(hdr|ldr)', frame)
    R = Replay(d)
    wbuf = R.at(R.q(0))
    t = lm.enc0_layer(wbuf)
    o, b8, b16 = pack_layer(t, 'enc0l')
    rw, rh = R.i2(56); gw, gh = R.i2(16); sx, sy = R.i2(8)
    b8.tofile(f'{work}/w8.bin'); b16.tofile(f'{work}/w16.bin'); wbuf[:4096].tofile(f'{work}/noise.bin')
    np.zeros(64, np.uint8).tofile(f'{work}/dummy.bin')
    open(f'{work}/lp.bin', 'wb').write(struct.pack('<4i72I', gw, gh, sx, sy, *o))
    expo = lambda off: float(np.fromfile(res[off], np.float16)[0])
    hw, hh = R.i2(248)
    open(f'{work}/ep.bin', 'wb').write(struct.pack('<2i2i2i2f2f2f2f2f2f2fff3Iffffff', rw, rh, sx, sy, gw // 2, gh // 2, *R.f2(168), *R.f2(176),
                                                   *R.f2(200), *R.f2(208), *R.f2(192), *R.f2(184), float(hw), float(hh), R.f1(268), R.f1(272),
                                                   R.P[216], R.P[316], R.i2(312)[0], expo(112), expo(120), *R.f2(256), *R.f2(64)))
    half = f'{work}/half-prev.bin'
    if not os.path.exists(half):
        np.zeros((hh // 2) * (hw // 2) * 4, np.float32).tofile(half)
    lines = [f'param {work}/lp.bin', f'grid {(gw + sx + 7) // 8} {(gh + sy + 7) // 8}',
             f'tex 1 {res[80]} {rw} {rh} 4 16 nearest', f'tex 2 {res[96]} {rw} {rh} 2 16 nearest', f'tex 3 {res[136]} {rw} {rh} 1 32 nearest',
             f'tex 4 {res[128]} {rw} {rh} 2 16 linear', f'tex 5 {half} {hw // 2} {hh // 2} 4 32 linear', f'tex 6 {res[160]} {rw} {rh} 4 9 linear']
    for i, off in enumerate((280, 288, 296)): lines.append(f'surf {7 + i} {rw} {rh} 2 16 - {work}/mv-{off}.bin')
    lines += [f'lut {work}/w8.bin', f'lut {work}/w16.bin', f'lut {work}/dummy.bin', f'lut {work}/dummy.bin',
              f'outbuf {work}/skip.bin {gw * gh * 32}', f'outbuf {work}/merged.bin {(gw // 2) * (gh // 2) * 64}',
              f'lut {work}/ep.bin', f'lut {work}/noise.bin',
              'bindings 0 8 9 10 11 12 13 14 15 16 1 2 3 4 5 6 7 17']
    open(f'{work}/manifest', 'w').write('\n'.join(lines) + '\n')
    spv = f'{work}/enc0_l.spv'
    compile_stage('enc0l', spv, ['-DLDR=1'] if k.endswith('_ldr') else [])
    run(kvk, spv, f'{work}/manifest', reps)
    print(f'frame {frame} ({k}, launch {seq}) {rw}x{rh}, pixel grid {gw}x{gh}, shift {sx},{sy}, reset {R.P[316]}')
    for off, name in ((280, 'A'), (288, 'B'), (296, 'C')):
        a = np.fromfile(res[off], np.float16).astype(np.float64); b = np.fromfile(f'{work}/mv-{off}.bin', np.float16).astype(np.float64)
        print(f'  motion {name}: identical {100 * (a == b).mean():5.1f}%  max |diff| {np.abs(a - b).max():.4g}')
    out, merged = R.q(40), R.q(48)
    after = np.fromfile(res[40], np.uint8); base = out
    ref = token_codes(after, 0, gw, gh, 32)
    got = np.fromfile(f'{work}/skip.bin', np.uint8).reshape(gh, gw, 32)
    print('  ' + report('block output', got, ref))
    print('  ' + report('block output, rows inside the image', got[:rh], ref[:rh]))
    mref = token_codes(after, merged - base, gw // 2, gh // 2, 64)
    mgot = np.fromfile(f'{work}/merged.bin', np.uint8).reshape(gh // 2, gw // 2, 64)
    print('  ' + report('merged', mgot, mref))
    print('  ' + report('merged, rows inside the image', mgot[:rh // 2], mref[:rh // 2]))


def dec0(cap, frame, work, kvk, reps):
    seq, k, d, res = launch(cap, r'rrlite_dec0_4x4', frame)
    R = Replay(d)
    t = lm.dec0_layer(R.at(R.q(0)))
    o, b8, b16 = pack_layer(t, 'dec0l')
    sx, sy = R.i2(8); gw, gh = R.i2(16)
    rw, rh = [int(x) for x in re.search(r'surface-(\d+)x(\d+)x', res[128]).groups()]
    b8.tofile(f'{work}/dw8.bin'); b16.tofile(f'{work}/dw16.bin')
    np.zeros(64, np.uint8).tofile(f'{work}/dummy.bin')
    token_codes(R.at(R.q(24)), 0, gw // 2, gh // 2, 64).tofile(f'{work}/din.bin')
    token_codes(R.at(R.q(32)), 0, gw, gh, 32).tofile(f'{work}/dskip.bin')
    open(f'{work}/dlp.bin', 'wb').write(struct.pack('<4i72I', gw, gh, sx, sy, *o))
    open(f'{work}/dp.bin', 'wb').write(struct.pack('<2iiI2f2f2f2fff2f', rw, rh, gw // 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, R.f1(160), R.f1(164), 0, 0))
    lines = [f'param {work}/dlp.bin', f'grid {(gw + sx + 7) // 8} {(gh + sy + 7) // 8}']
    fmt = {128: (4, 9), 136: (4, 8), 144: (4, 8), 152: (4, 16)}
    for off in (128, 136, 144, 152): lines.append(f'surf {off} {rw} {rh} {fmt[off][0]} {fmt[off][1]} - {work}/s{off}.bin')
    lines += [f'lut {work}/dw8.bin', f'lut {work}/dw16.bin', f'lut {work}/din.bin', f'lut {work}/dskip.bin',
              f'outbuf {work}/dump.bin {gw * gh * 64}', f'lut {work}/dummy.bin', f'lut {work}/dp.bin', 'bindings 0 8 9 10 11 1 2 3 4 5 6 7']
    open(f'{work}/dmanifest', 'w').write('\n'.join(lines) + '\n')
    spv = f'{work}/dec0_l.spv'
    compile_stage('dec0l', spv, [a for a in os.environ.get('L_DEFS', '').split()])
    run(kvk, spv, f'{work}/dmanifest', reps)
    print(f'frame {frame} (launch {seq}) {rw}x{rh}, pixel grid {gw}x{gh}, shift {sx},{sy}, sigma {R.f1(160):.4g} rho {R.f1(164):.4g}')
    for off, name, dt in ((128, 'feature', np.int8), (136, 'gates', np.uint8), (144, 'mix', np.uint8), (152, 'covariance', np.float16)):
        a = np.fromfile(res[off], dt).reshape(rh, rw, 4).astype(np.float64); b = np.fromfile(f'{work}/s{off}.bin', dt).reshape(rh, rw, 4).astype(np.float64)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        print(f'  {name:10s}: identical {100 * same.mean():6.2f}% (per channel ' + ' '.join(f'{100 * same[..., c].mean():6.2f}' for c in range(4)) +
              f'), max |diff| {np.nanmax(np.abs(a - b)):.4g}')


if __name__ == '__main__':
    stage, cap, frame, work, kvk = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    reps = sys.argv[6] if len(sys.argv) > 6 else '1'
    os.makedirs(work, exist_ok=True)
    {'enc0': enc0, 'dec0': dec0}[stage](cap, frame, work, kvk, reps)
