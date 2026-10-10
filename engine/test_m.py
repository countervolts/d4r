#!/usr/bin/env python3
"""Run the engine's own preset M layers (layer_m.comp) on a captured frame and compare with the captured outputs and
with the plain numpy model (m_model.py).

usage: test_m.py CAPTURE_DIR WORK_DIR EVK [--frame N] [--time N] [--model] [--chain] [layer index ...]
  CAPTURE_DIR/replay: layer launches before they ran; CAPTURE_DIR/dump: buffers after each launch.
  --model also runs the numpy model (slow) and reports the shader against it.
  --chain feeds each layer the engine's own previous outputs instead of the captured inputs.

The layers' FP8 matrices are exported in the shader's blocked fixed-16 tile layout (pack8 below); the front-end
embedding/expansion weights that compile_m.py packages for enc0_m/dec0_m stay in NGX's own plain layout.
"""
import glob, os, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m_model as mm
import model_enc3 as me

F16 = np.float16


def codes(values):
    """e4m3 codes of exact e4m3 values"""
    v = np.asarray(values, np.float64); a = np.abs(v)
    c = np.searchsorted(me._POS, a).astype(np.uint8)
    assert np.array_equal(me._POS[c], a), 'not an FP8 value'
    return (c | np.where(np.signbit(v) & (a > 0), 128, 0)).astype(np.uint8)


class Pack:
    def __init__(self, dtype, align):
        self.parts, self.n, self.dtype, self.align = [], 0, dtype, align

    def add(self, a):
        a = np.ascontiguousarray(a, self.dtype).ravel()
        pad = (-self.n) % self.align
        if pad: self.parts.append(np.zeros(pad, self.dtype)); self.n += pad
        off = self.n; self.parts.append(a); self.n += a.size
        return off

    def data(self):
        return np.concatenate(self.parts + [np.zeros((-self.n) % 64 + 64, self.dtype)])


def pack8(values):
    """blocked FP8 bytes of a logical weight matrix W[K][N] (K, N multiples of 16), the layout layer_m.comp loads:
    [K/16][N/16][n16][k16], i.e. reshape(K//16, 16, N//16, 16).transpose(0, 2, 3, 1). contiguous (no arithmetic).
    A fixed 16x16 tile is 256 bytes and starts at 256*(kblock*(N//16) + nblock); the shader's A operand is the tile
    [N16][K16] read row-major with stride 16 (A[n][k] = W[kblock*16 + k][nblock*16 + n]), so a K16 row block of
    every output channel tile stays contiguous and the per-head slice of wo is still base + 32*h*C."""
    W = codes(values)
    K, N = W.shape
    assert K % 16 == 0 and N % 16 == 0, (K, N)
    return np.ascontiguousarray(W.reshape(K // 16, 16, N // 16, 16).transpose(0, 2, 3, 1)).ravel()


def pack_layer(t, name):
    """(o[72], FP8 bytes, f16 words) of one layer's tensors; the FP8 matrices are blocked by pack8()"""
    C, heads, NPM, CIN = mm.LAYERS[name]
    b8, b16 = Pack(np.uint8, 16), Pack(F16, 16)
    tile = lambda v: np.tile(np.asarray(v, F16)[None, :], (16, 1))
    o = [0] * 72
    o[0], o[1], o[2], o[3] = b16.add(t['g1']), b16.add(tile(t['bo'])), b16.add(t['g2']), b16.add(tile(t['b2']))
    assert t['wo'].shape == (32 * heads, C), t['wo'].shape
    wo = b8.add(pack8(t['wo']))
    for h in range(heads):
        assert t['q'][h].shape == (C, 32) and t['v'][h].shape == (C, 32), (t['q'][h].shape, t['v'][h].shape)
        o[4 + 4 * h], o[5 + 4 * h], o[6 + 4 * h], o[7 + 4 * h] = b8.add(pack8(t['q'][h])), b8.add(pack8(t['v'][h])), b16.add(t['bias'][h]), wo + 32 * h * C
    for c in range(C // 8):
        assert t['w1'][c].shape == (C, 32) and t['w2'][c].shape == (32, C), (t['w1'][c].shape, t['w2'][c].shape)
        o[20 + 3 * c], o[21 + 3 * c], o[22 + 3 * c] = b8.add(pack8(t['w1'][c])), b16.add(tile(t['b1'][c])), b8.add(pack8(t['w2'][c]))
    if NPM:
        assert t['merge_w'].shape == (4 * C, NPM), t['merge_w'].shape
        o[68], o[69] = b8.add(pack8(t['merge_w'])), b16.add(tile(t['merge_b']))
    if CIN:
        assert t['expand_w'].shape == (CIN, 4 * C), t['expand_w'].shape
        o[68], o[69] = b8.add(pack8(t['expand_w'])), b16.add(tile(t['expand_b']))
    return o, b8.data(), b16.data()


def defines(name):
    C, heads, NPM, CIN = mm.LAYERS[name]
    return [f'-DC={C}', f'-DNH={heads}', f'-DKIND={1 if NPM else 2 if CIN else 0}', f'-DXC={NPM or CIN or 16}'] + os.environ.get('M_DEFS', '').split()


def token_codes(buf, off, W, H, C):
    a = buf[off:off + (C // 32) * H * W * 32].reshape(C // 32, H, W, 32)
    return np.ascontiguousarray(a.transpose(1, 2, 0, 3).reshape(H, W, C))


def report(tag, got, ref):
    rank = lambda v: np.searchsorted(me._POS, np.abs(v)) * np.sign(v)
    g, r = me.E4M3[got].astype(np.float64), me.E4M3[ref].astype(np.float64)
    steps = np.abs(rank(g) - rank(r))
    return (f'{tag} {mm.psnr(g, r):5.1f} dB, identical {100 * (got == ref).mean():5.1f}%, within one step {100 * (steps <= 1).mean():5.1f}%')


def main():
    a = sys.argv[1:]
    def opt(flag, default):
        if flag in a:
            i = a.index(flag); v = a[i + 1]; del a[i:i + 2]; return v
        return default
    frame, reps = int(opt('--frame', 0)), opt('--time', '1')
    model = '--model' in a
    if model: a.remove('--model')
    chain = '--chain' in a
    if chain: a.remove('--chain')
    own = {}    # arena address -> the engine's own tokens there
    cap, work, evk = a[:3]
    only = [int(x) for x in a[3:]]
    os.makedirs(work, exist_ok=True)
    Ps, dirs = mm.launches(cap, frame)
    total = 0.0
    for i, (P, d, name) in enumerate(zip(Ps, dirs, mm.ORDER)):
        if only and i not in only: continue
        C, heads, NPM, CIN = mm.LAYERS[name]
        seq = int(os.path.basename(d).split('-')[1])
        after = np.fromfile(glob.glob(f'{cap}/dump/launch-{seq:03d}-*-arg024-*')[0], np.uint8)
        t = mm.plain_layer(P, name)
        o, b8, b16 = pack_layer(t, name)
        W, H = P.W, P.H
        pre = f'{work}/{i}-{name}'
        open(pre + '.params', 'wb').write(struct.pack('<4i72I', W, H, P.sx, P.sy, *o))
        b8.tofile(pre + '.w8'); b16.tofile(pre + '.w16')
        def source(ptr, w, h_, c):
            return own[ptr] if chain and ptr in own else token_codes(P.arena, ptr - P.abase, w, h_, c)
        if CIN:
            source(P.inp, W // 2, H // 2, CIN).tofile(pre + '.in')
            source(P.p32, W, H, C).tofile(pre + '.skip')
        else:
            source(P.inp, W, H, C).tofile(pre + '.in')
            np.zeros(64, np.uint8).tofile(pre + '.skip')
        spv = pre + '.spv'
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defines(name), f'{HERE}/layer_m.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
        gx, gy = (W + P.sx + 7) // 8, (H + P.sy + 7) // 8
        msize = (H // 2) * (W // 2) * NPM if NPM else 64
        r = subprocess.run([evk, spv, str(gx), str(gy), reps, f'in:0={pre}.params', f'in:1={pre}.w8', f'in:2={pre}.w16', f'in:3={pre}.in',
                            f'in:4={pre}.skip', f'out:5={pre}.out:{H * W * C}', f'out:6={pre}.merged:{msize}'], capture_output=True, text=True)
        if r.returncode:
            print(name, 'FAILED', r.stderr[-600:]); sys.exit(1)
        us = [float(l.split(':')[1].split()[0]) for l in r.stdout.split('\n') if 'us per dispatch' in l]
        total += us[0] if us else 0
        got = np.fromfile(pre + '.out', np.uint8).reshape(H, W, C)
        own[P.out] = got
        ref = token_codes(after, P.out - P.abase, W, H, C)
        line = f'{i} {name:9s} {W}x{H}x{C} {us[0] if us else 0:7.1f} us: ' + report('vs capture', got, ref)
        if NPM:
            gm = np.fromfile(pre + '.merged', np.uint8).reshape(H // 2, W // 2, NPM)
            own[P.p48] = gm
            line += '; ' + report('merged', gm, token_codes(after, P.p48 - P.abase, W // 2, H // 2, NPM))
        print(line, flush=True)
        if model:
            if CIN:
                out, merged = mm.forward(t, name, mm.tokens(P.arena, P.inp - P.abase, W // 2, H // 2, CIN), mm.tokens(P.arena, P.p32 - P.abase, W, H, C), P.sx, P.sy)
            else:
                out, merged = mm.forward(t, name, mm.tokens(P.arena, P.inp - P.abase, W, H, C), None, P.sx, P.sy)
            line = '            ' + report('shader vs numpy model', got, codes(out))
            if NPM: line += '; ' + report('merged', gm, codes(merged))
            print(line, flush=True)
    print(f'total {total:.1f} us')


if __name__ == '__main__':
    main()
