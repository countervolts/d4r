#!/usr/bin/env python3
"""Preset M (rrlite): plain weights of the Swin layers and a plain numpy model of them.

NGX keeps the weights as FP8 (e4m3) bytes in a layout arranged for its tensor-core fragments, and half of the
activations in a permuted ("pair") channel order. plain_layer() undoes both: every tensor becomes a row-major
matrix W[K][N] (y = x W) over natural channel order, f16 exact (every e4m3 value is an f16 value).
forward() is the layer's mathematics on those matrices. Every activation that feeds a matrix product as its left
operand is rounded to FP8 first, as in NVIDIA's arithmetic (its tensor-core operands are FP8) and in d4r's native
FP8 kernels; sums are accumulated without the intermediate f16 rounding of NVIDIA's k32 steps.

usage: m_model.py CAPTURE_DIR [frame]     compares forward() with the captured outputs of each layer launch
"""
import glob, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kernels', 'tools'))
import model_enc3 as me
import swin_model as sm

F16 = np.float16
# kernel -> (C, heads, merged channels, low-resolution input channels)
LAYERS = {'enc1': (64, 2, 96, 0), 'enc2': (96, 4, 128, 0), 'enc3_tube': (128, 4, 0, 0), 'dec2': (96, 4, 0, 128), 'dec1': (64, 2, 0, 96)}
ORDER = ['enc1', 'enc2'] + ['enc3_tube'] * 6 + ['dec2', 'dec1']


def gp(n):
    return sm.gp(np.arange(n))


def plain_layer(P, name):
    """dict of plain f16 tensors of one layer launch (P: model_enc3.Params)"""
    C, heads, NPM, CIN = LAYERS[name]
    fp8 = lambda T: me.E4M3[P.wbuf[T]]
    pre = 4 * C * CIN + 8 * C if CIN else 0
    L = sm.layout(C, heads, pre)
    g = gp(C)
    t = {'g1': P.f16vec(L['g1'], C)[g], 'bo': P.f16vec(L['bo'], C)[g], 'g2': P.f16vec(L['g2'], C)[g], 'b2': P.f16vec(L['b2'], C)[g],
         'q': [], 'v': [], 'bias': [], 'w1': [], 'b1': [], 'w2': []}
    wo = np.zeros((32 * heads, C))
    g32 = gp(32)
    for h in range(heads):
        hb = L['head'][h]
        t['q'].append(fp8(sm.woff_table(hb, 1024, 512, C, 32)))
        t['v'].append(fp8(sm.woff_table(hb + 32 * C, 1024, 512, C, 32)))
        t['bias'].append(sm.rel_bias(P, hb + 64 * C))
        raw = fp8(sm.woff_table(hb + 64 * C + 512, 0, 512, 32, C))          # rows: pair-ordered attention outputs
        wo[32 * h + g32] = raw[:, g]                                       # Wo[gp(a)][c] = raw[a][gp(c)]
    t['wo'] = wo
    for B1 in L['b1']:
        t['w1'].append(fp8(sm.woff_table(B1, 512, 16 * C, C, 32)))
        t['b1'].append(P.f16vec(B1 + 32 * C, 32))
        raw = fp8(sm.woff_table(B1 + 32 * C + 64, 0, 512, 32, C))
        w2 = np.zeros((32, C)); w2[g32] = raw[:, g]
        t['w2'].append(w2)
    if NPM:
        PM0 = L['end']
        W = np.concatenate([fp8(sm.woff_table(PM0 + k * 128 * C, 512, 64 * C, 4 * C, 32)) for k in range(NPM // 32)], 1)
        b = np.concatenate([P.f16vec(PM0 + 4 * C * NPM + 64 * k, 32) for k in range(NPM // 32)])
        gm = gp(NPM)
        t['merge_w'], t['merge_b'] = W[:, gm], b[gm]
    if CIN:
        cols = 4 * C // heads
        W = np.concatenate([fp8(sm.woff_table(w * CIN * cols, 512, 16 * CIN, CIN, cols)) for w in range(heads)], 1)   # [CIN][4C]
        b = np.concatenate([P.f16vec(4 * C * CIN + 2 * cols * w, cols) for w in range(heads)])
        idx = (np.arange(4)[:, None] * C + g[None, :]).ravel()            # x0[.., q, c] = E[q C + gp(c)]
        t['expand_w'], t['expand_b'] = W[:, idx], b[idx]
    return {k: ([np.asarray(a, F16) for a in v] if isinstance(v, list) else np.asarray(v, F16)) for k, v in t.items()}


def h(a):
    return np.asarray(a, np.float64).astype(F16).astype(np.float64)


def q8(a):
    return me.q8(a)


def softmax_rows(S):
    t = np.clip(h(S * me.C_SCALE), -float(me.CLAMP), float(me.CLAMP))
    poly = h(t * h(t * (-t) + me.P1) + float(me.P0))
    w = me.exp_trick(poly).astype(np.float64)
    s = h(w.sum(1, keepdims=True))
    return h(w * h(1.0 / s))


def block(t, x0, C, heads):
    """x0: [windows][16][C] f16 values; returns the block output before the FP8 rounding"""
    f = lambda a: np.asarray(a, np.float64)
    r = h(1.0 / np.sqrt(h((x0 * x0).sum(-1, keepdims=True) + 2.0 ** -13)))
    h1 = q8(h(x0 * h(r * f(t['g1']))))
    O = np.zeros(x0.shape[:2] + (32 * heads,))
    for k in range(heads):
        Q = h(h1 @ f(t['q'][k])); V = h(h1 @ f(t['v'][k]))
        for w in range(x0.shape[0]):
            Qq = q8(Q[w])
            O[w, :, 32 * k:32 * k + 32] = q8(h(softmax_rows(h(Qq @ Qq.T + f(t['bias'][k]))) @ V[w]))
    x1 = h(O @ f(t['wo']) + x0 + f(t['bo']))
    r = h(1.0 / np.sqrt(h((x1 * x1).sum(-1, keepdims=True) + 2.0 ** -13)))
    h2 = q8(h(x1 * h(r * f(t['g2']))))
    x2 = x1 + f(t['b2'])
    for w1, b1, w2 in zip(t['w1'], t['b1'], t['w2']):
        x2 = x2 + q8(me.gelu_poly(h(h2 @ f(w1) + f(b1)))) @ f(w2)
    return h(x2)


def tokens(buf, off, W, H, C):
    """planes of 32 e4m3 channels at byte `off` -> [H][W][C] values"""
    a = me.E4M3[buf[off:off + (C // 32) * H * W * 32]].reshape(C // 32, H, W, 32)
    return a.transpose(1, 2, 0, 3).reshape(H, W, C)


def mirror(v, n):
    v = np.abs(v); return np.minimum(v, 2 * n - 2 - v)


def forward(t, name, x, skip, sx, sy):
    """x: [H][W][C] input tokens (decoders: [H/2][W/2][CIN]); returns (out [H][W][C], merged or None), FP8-rounded"""
    C, heads, NPM, CIN = LAYERS[name]
    f = lambda a: np.asarray(a, np.float64)
    if CIN:
        H, W = skip.shape[:2]
    else:
        H, W = x.shape[:2]
    gx, gy = (W + sx + 7) // 8, (H + sy + 7) // 8
    Y, X = np.meshgrid(np.arange(8 * gy) - sy, np.arange(8 * gx) - sx, indexing='ij')
    if CIN:
        e = h(x @ f(t['expand_w']) + f(t['expand_b'])).reshape(x.shape[0], x.shape[1], 4, C)
        # the low-resolution token of block row r is taken at (8 bx - sx) // 2 + r % 4 (floor), then mirrored
        Yb, Xb = (Y + sy) // 8 * 8 - sy, (X + sx) // 8 * 8 - sx
        ly, lx = mirror(Yb // 2 + (Y - Yb) // 2, H // 2), mirror(Xb // 2 + (X - Xb) // 2, W // 2)
        q = ((Y - Yb) & 1) * 2 + ((X - Xb) & 1)
        x0 = h(q8(e[ly, lx, q]) + skip[mirror(Y, H), mirror(X, W)])
    else:
        x0 = x[mirror(Y, H), mirror(X, W)]
    win = x0.reshape(2 * gy, 4, 2 * gx, 4, C).transpose(0, 2, 1, 3, 4).reshape(-1, 16, C)
    y = block(t, win, C, heads).reshape(2 * gy, 2 * gx, 4, 4, C).transpose(0, 2, 1, 3, 4).reshape(8 * gy, 8 * gx, C)
    out = q8(y)
    merged = None
    if NPM:
        A = out.reshape(4 * gy, 2, 4 * gx, 2, C).transpose(0, 2, 1, 3, 4).reshape(4 * gy, 4 * gx, 4 * C)
        m = q8(h(A @ f(t['merge_w']) + f(t['merge_b'])))
        # merged token (my, mx) of a block sits at (8 by - sy) // 2 + my
        my = ((np.arange(4 * gy) // 4) * 8 - sy) // 2 + np.arange(4 * gy) % 4
        mx = ((np.arange(4 * gx) // 4) * 8 - sx) // 2 + np.arange(4 * gx) % 4
        merged = np.zeros((H // 2, W // 2, NPM))
        oky, okx = (my >= 0) & (my < H // 2), (mx >= 0) & (mx < W // 2)
        merged[np.ix_(my[oky], mx[okx])] = m[np.ix_(oky, okx)]
    oy, ox = (Y[:, 0] >= 0) & (Y[:, 0] < H), (X[0] >= 0) & (X[0] < W)
    return out[np.ix_(oy, ox)], merged


def launches(cap, frame=0):
    """the ten layer launches of one frame, in network order, as model_enc3.Params"""
    dirs = sorted(d for d in glob.glob(f'{cap}/replay/replay-*-rrlite_*_4x4') if any(d.endswith(f'rrlite_{k}_4x4') for k in LAYERS))
    if len(dirs) < 10 * (frame + 1): raise ValueError('capture lacks the layer launches of that frame')
    return [me.Params(d) for d in dirs[10 * frame:10 * frame + 10]], dirs[10 * frame:10 * frame + 10]


def psnr(a, b):
    d = (a - b) ** 2
    return 10 * np.log10(np.abs(b).max() ** 2 / max(d.mean(), 1e-30))


def main():
    cap = sys.argv[1]; frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    Ps, dirs = launches(cap, frame)
    for i, (P, d, name) in enumerate(zip(Ps, dirs, ORDER)):
        C, heads, NPM, CIN = LAYERS[name]
        seq = int(os.path.basename(d).split('-')[1])
        after = np.fromfile(glob.glob(f'{cap}/dump/launch-{seq:03d}-*-arg024-*')[0], np.uint8)
        t = plain_layer(P, name)
        W, H = P.W, P.H
        if CIN:
            x = tokens(P.arena, P.inp - P.abase, W // 2, H // 2, CIN); skip = tokens(P.arena, P.p32 - P.abase, W, H, C)
        else:
            x = tokens(P.arena, P.inp - P.abase, W, H, C); skip = None
        out, merged = forward(t, name, x, skip, P.sx, P.sy)
        ref = tokens(after, P.out - P.abase, W, H, C)
        line = f'{name:10s} launch {seq:3d} {W}x{H}x{C} shift {P.sx},{P.sy}: output {psnr(out, ref):5.1f} dB, identical {100 * (out == ref).mean():5.1f}%'
        if NPM:
            mref = tokens(after, P.p48 - P.abase, W // 2, H // 2, NPM)
            line += f'; merged {psnr(merged, mref):5.1f} dB, identical {100 * (merged == mref).mean():5.1f}%'
        print(line, flush=True)


if __name__ == '__main__':
    main()
