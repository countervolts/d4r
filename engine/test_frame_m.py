#!/usr/bin/env python3
"""Run whole preset M frames on the engine's own stages and compare the output with the capture.

usage: test_frame_m.py CAPTURE_DIR WORK_DIR KVK EVK [frames]
Every stage runs as its own process (kvk for the texture stages, evk for the network layers). Only the frame's
colour, motion and depth, the exposure values and the kernels' parameter blocks come from the capture; all
intermediate data and all temporal history are the engine's own.
"""
import glob, os, re, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m_model as mm
import swin_model as sm
from test_m import pack_layer, defines, token_codes, report
cap, work, kvk, evk = sys.argv[1:5]
frames = int(sys.argv[5]) if len(sys.argv) > 5 else 2
os.makedirs(work, exist_ok=True)
LDR = os.environ.get('LDR') == '1'      # a capture without the HDR flag: NGX's *_ldr_* kernels, the shaders' LDR paths
R = 'ldr' if LDR else 'hdr'
# below a reconstruction/render ratio of 3 the capture holds NGX's rrlite_post_3_2; the engine then needs the
# larger shared tile (post_m.comp's TILE_W/TILE_H)
POST = os.environ.get('POST_KERNEL') or next((k for k in (f'rrlite_post_3_1_mvlo_{R}_folded', f'rrlite_post_3_2_mvlo_{R}_folded')
    if glob.glob(f'{cap}/dump/launch-*-{k}-arg*')), f'rrlite_post_3_1_mvlo_{R}_folded')
POST_TILE = ['-DTILE_W=15', '-DTILE_H=15'] if '3_2' in POST else []
NAMES = {'enc0': f'rrlite_enc0_4x4_mvlo_{R}_folded', 'dec0': 'rrlite_dec0_4x4_folded', 'post': POST, 'down': f'rrlite_downsample_kernel_static_{R}'}
STAGE = ['-DLDR=1'] if LDR else []


class Launch:
    def __init__(self, kernel, frame):
        seqs = sorted({int(re.match(r'launch-(\d+)-', os.path.basename(f)).group(1)) for f in glob.glob(f'{cap}/dump/launch-*-{kernel}-arg*')})
        self.seq = seqs[frame]
        self.res = {int(re.search(r'-arg(\d+)-', f).group(1)): f for f in glob.glob(f'{cap}/dump/launch-{self.seq:03d}-{kernel}-arg*')}
        self.dir = f'{cap}/replay/replay-{self.seq:06d}-{kernel}'
        self.P = open(self.dir + '/args.bin', 'rb').read()
        self.allocs = {}
        for line in open(self.dir + '/manifest.txt'):
            q = line.split()
            if q and q[0] == 'alloc': self.allocs[int(q[1])] = (int(q[2], 16), np.fromfile(f'{self.dir}/alloc-{q[1]}.bin', np.uint8))
    i2 = lambda s, o: struct.unpack_from('<2i', s.P, o)
    f2 = lambda s, o: struct.unpack_from('<2f', s.P, o)
    f1 = lambda s, o: struct.unpack_from('<f', s.P, o)[0]
    u8 = lambda s, o: struct.unpack_from('<Q', s.P, o)[0]
    expo = lambda s, o: float(np.fromfile(s.res[o], np.float16)[0]) if o in s.res else 1.0
    def blob(self, ptr):
        a, b = next((a, b) for a, b in self.allocs.values() if a <= ptr < a + b.size)
        return b[ptr - a:]


def shader(name, flags=()):
    spv = f'{work}/{name}{"".join(flags).replace("-D", "_").replace("=", "")}.spv'
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *flags, f'{HERE}/{name}.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
    return spv


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print('FAILED', ' '.join(cmd[:3]), r.stderr[-500:]); sys.exit(1)


def zeros(path, n, dtype=np.float16):
    np.zeros(n, dtype).tofile(path); return path


prev = None      # the previous frame's files
for f in range(frames):
    D = f'{work}/f{f}'; os.makedirs(D, exist_ok=True)
    # ---------------------------------------------------------------- input stage
    L = Launch(NAMES['enc0'], f)
    rw, rh = L.i2(56); gw, gh = L.i2(16); tw, th = gw // 2, gh // 2
    hw, hh = L.i2(248)
    wb = L.blob(L.u8(0))
    g = sm.gp(np.arange(64))
    np.ascontiguousarray(wb[sm.woff_table(4096, 512, 2048, 128, 64)][:, g]).tofile(f'{D}/e0.w8')
    np.tile(wb[12288:12288 + 128].view(np.float16)[g][None, :], (16, 1)).tofile(f'{D}/e0.w16')
    wb[:4096].tofile(f'{D}/e0.noise')
    open(f'{D}/e0.params', 'wb').write(struct.pack('<2i2i2i2f2f2f2f2f2f2fff3Iffffff', rw, rh, *L.i2(8), tw, th, *L.f2(168), *L.f2(176), *L.f2(200), *L.f2(208),
                                                   *L.f2(192), *L.f2(184), float(hw), float(hh), L.f1(268), L.f1(272), L.P[216], L.P[316], L.i2(312)[0], L.expo(112), L.expo(120), *L.f2(256), *L.f2(64)))
    colour, motion, depth = L.res[80], L.res[96], L.res[136]
    e1, e2 = L.expo(112), L.expo(120)
    pmv = prev['mvA'] if prev else zeros(f'{D}/zero.mv', rw * rh * 2)
    phalf = prev['half'] if prev else zeros(f'{D}/zero.half', (hw // 2) * (hh // 2) * 4, np.float32)
    pfeat = prev['feat'] if prev else zeros(f'{D}/zero.feat', rw * rh * 4, np.int8)
    phist = prev['hist'] if prev else zeros(f'{D}/zero.hist', hw * hh * 4)
    open(f'{D}/e0.manifest', 'w').write('\n'.join([
        f'param {D}/e0.params', f'grid {(gw + L.i2(8)[0] + 7) // 8} {(gh + L.i2(8)[1] + 7) // 8}',
        f'tex 1 {colour} {rw} {rh} 4 16 nearest', f'tex 2 {motion} {rw} {rh} 2 16 nearest', f'tex 3 {depth} {rw} {rh} 1 32 nearest',
        f'tex 4 {pmv} {rw} {rh} 2 16 linear', f'tex 5 {phalf} {hw // 2} {hh // 2} 4 32 linear', f'tex 6 {pfeat} {rw} {rh} 4 9 linear',
        f'surf 7 {rw} {rh} 2 16 - {D}/mvA', f'surf 8 {rw} {rh} 2 16 - {D}/mvB', f'surf 9 {rw} {rh} 2 16 - {D}/mvC',
        f'lut {D}/e0.w8', f'lut {D}/e0.w16', f'lut {D}/e0.noise', f'outbuf {D}/tokens {tw * th * 64}']) + '\n')
    run([kvk, shader('enc0_m', STAGE), f'{D}/e0.manifest', '1', '32'])
    ref_tokens = token_codes(np.fromfile(L.res[40], np.uint8), L.u8(48) - next(a for a, b in L.allocs.values() if a <= L.u8(48) < a + b.size), tw, th, 64)
    print(f'frame {f}: ' + report('input tokens', np.fromfile(f'{D}/tokens', np.uint8).reshape(th, tw, 64), ref_tokens), flush=True)
    # ---------------------------------------------------------------- network
    Ps, dirs = mm.launches(cap, f)
    own = {Ps[0].inp: f'{D}/tokens'}
    for i, (P, d, name) in enumerate(zip(Ps, dirs, mm.ORDER)):
        C, heads, NPM, CIN = mm.LAYERS[name]
        o, b8, b16 = pack_layer(mm.plain_layer(P, name), name)
        W, H = P.W, P.H
        pre = f'{D}/l{i}'
        open(pre + '.params', 'wb').write(struct.pack('<4i72I', W, H, P.sx, P.sy, *o)); b8.tofile(pre + '.w8'); b16.tofile(pre + '.w16')
        skip = own[P.p32] if CIN else zeros(pre + '.noskip', 64, np.uint8)
        msize = (H // 2) * (W // 2) * NPM if NPM else 64
        run([evk, shader('layer_m', defines(name)), str((W + P.sx + 7) // 8), str((H + P.sy + 7) // 8), '1', f'in:0={pre}.params', f'in:1={pre}.w8', f'in:2={pre}.w16',
             f'in:3={own[P.inp]}', f'in:4={skip}', f'out:5={pre}.out:{H * W * C}', f'out:6={pre}.merged:{msize}'])
        own[P.out] = pre + '.out'
        if NPM: own[P.p48] = pre + '.merged'
    seq = int(os.path.basename(dirs[-1]).split('-')[1])
    after = np.fromfile(glob.glob(f'{cap}/dump/launch-{seq:03d}-*-arg024-*')[0], np.uint8)
    P = Ps[-1]
    print(f'frame {f}: ' + report('network output', np.fromfile(own[P.out], np.uint8).reshape(P.H, P.W, 64), token_codes(after, P.out - P.abase, P.W, P.H, 64)), flush=True)
    # ---------------------------------------------------------------- dec0
    L = Launch(NAMES['dec0'], f)
    wb = L.blob(L.u8(0))
    cols = (np.arange(4)[:, None] * 32 + np.arange(20)[None, :]).ravel()
    np.concatenate([wb[sm.woff_table(0, 512, 1024, 64, 128)][:, cols].ravel(), np.zeros(64, np.uint8)]).tofile(f'{D}/d0.w8')
    np.concatenate([np.tile(wb[8192:8192 + 256].view(np.float16)[cols][None, :], (16, 1)).ravel(), np.zeros(32, np.float16)]).tofile(f'{D}/d0.w16')
    open(f'{D}/d0.params', 'wb').write(struct.pack('<2iiI2f2f2f2f2f2f', rw, rh, tw, L.P[120], *L.f2(72), *L.f2(88), float(L.i2(96)[0]), float(L.i2(96)[1]), *L.f2(80), *L.f2(160), *L.f2(104)))
    open(f'{D}/d0.manifest', 'w').write('\n'.join([
        f'param {D}/d0.params', f'grid {gw // 8} {gh // 8}', f'tex 1 {D}/mvA {rw} {rh} 2 16 nearest', f'tex 2 {pfeat} {rw} {rh} 4 9 linear',
        f'surf 3 {rw} {rh} 4 9 - {D}/feat', f'surf 4 {rw} {rh} 4 8 - {D}/gates', f'surf 5 {rw} {rh} 4 8 - {D}/mix', f'surf 6 {rw} {rh} 4 16 - {D}/cov',
        f'lut {own[Ps[-1].out]}', f'lut {D}/d0.w8', f'lut {D}/d0.w16']) + '\n')
    run([kvk, shader('dec0_m'), f'{D}/d0.manifest', '1', '32'])
    for off, name, dt in ((128, 'feat', np.int8), (136, 'gates', np.uint8), (144, 'mix', np.uint8)):
        a = np.fromfile(L.res[off], dt).astype(np.int32); b = np.fromfile(f'{D}/{name}', dt).astype(np.int32)
        print(f'frame {f}: dec0 {name:5s} identical {100 * (a == b).mean():5.1f}%, within one step {100 * (np.abs(a - b) <= 1).mean():5.1f}%, mean |diff| {np.abs(a - b).mean():.3f} steps', flush=True)
    # ---------------------------------------------------------------- reconstruction and output
    L = Launch(NAMES['post'], f)
    open(f'{D}/post.params', 'wb').write(struct.pack('<2i2i2i2f2f2ffIff', *L.i2(8), hw, hh, rw, rh, *L.f2(124), *L.f2(132), *L.f2(160), L.f1(120), L.P[168], L.expo(16), L.expo(24)))
    open(f'{D}/post.manifest', 'w').write('\n'.join([
        f'param {D}/post.params', f'grid {(hw + 15) // 16} {(hh + 15) // 16}',
        f'tex 1 {colour} {rw} {rh} 4 16 nearest', f'tex 2 {D}/mix {rw} {rh} 4 8 nearest', f'tex 3 {D}/gates {rw} {rh} 4 8 nearest', f'tex 4 {D}/cov {rw} {rh} 4 16 nearest',
        f'tex 5 {phist} {hw} {hh} 4 16 linear', f'tex 6 {D}/mvA {rw} {rh} 2 16 nearest', f'tex 7 {D}/mvB {rw} {rh} 2 16 nearest', f'tex 8 {D}/mvC {rw} {rh} 2 16 linear',
        f'surf 9 {hw} {hh} 4 16 - {D}/hist', f'surf 10 {hw // 2} {hh // 2} 4 32 - {D}/half', f'outbuf {D}/packed {hw * hh * 4}']) + '\n')
    run([kvk, shader('post_m', STAGE + POST_TILE), f'{D}/post.manifest', '1', '64'])
    L = Launch(NAMES['down'], f)
    ow, oh = L.i2(8)
    open(f'{D}/down.params', 'wb').write(struct.pack('<8i4ff', *L.i2(0), *L.i2(40), *L.i2(64), *L.i2(72), *L.f2(80), *L.f2(88), L.expo(48)))
    open(f'{D}/down.manifest', 'w').write('\n'.join([
        f'param {D}/down.params', f'grid {(ow + 7) // 8} {(oh + 63) // 64}', f'tex 1 {colour} {rw} {rh} 4 16 nearest',
        f'surf 2 {ow} {oh} 4 16 - {D}/output', f'lut {D}/packed']) + '\n')
    run([kvk, shader('down_m', STAGE), f'{D}/down.manifest', '1', '64'])
    a = np.fromfile(L.res[32], np.float16).astype(np.float64).reshape(oh, ow, 4)[..., :3]
    b = np.fromfile(f'{D}/output', np.float16).astype(np.float64).reshape(oh, ow, 4)[..., :3]
    d = a - b
    print(f'frame {f}: FINAL OUTPUT {ow}x{oh}: {10 * np.log10(1.0 / np.mean(d * d)):.1f} dB PSNR (peak 1.0), max |diff| {np.abs(d).max():.3f}, non-finite {int((~np.isfinite(b)).sum())}', flush=True)
    prev = {'mvA': f'{D}/mvA', 'half': f'{D}/half', 'feat': f'{D}/feat', 'hist': f'{D}/hist'}
