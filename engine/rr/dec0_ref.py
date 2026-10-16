#!/usr/bin/env python3
"""Exact numpy reference of the denoiser's decoder 0 (cuda_dldn_engine_swin_dec0_kernel), one 8x8 window at a time,
from its weights and real inputs only: the specification the native shader implements. Derived like
engine/rr/enc1_ref.py, phase by phase against ZLUDA replays (engine/rr/phase_dump.py).

  expand    the low-resolution tokens under the window and its border (6 x 6 of dec1's 64-channel output, arena order)
            -> 128 = 4 sub-positions x 32 channels (+ bias), FP8; sub-position 2 (y & 1) + (x & 1)
  prologue  key token = expanded + skip token (enc0's 32-channel output at argument 24), f16; RMS norm (enc1's per-word
            tree on one plane) times the gain at W+8448; the residual is the token + the table at W+39232
  attention two heads of 32 channels (K = 32, one k32 step), as enc1
  MLP       RMS norm, four branches of 32 -> 32 (+ bias, ramp activation) -> 32 onto the residual (+ table)

usage: dec0_ref.py CAPTURE FRAME WEIGHTS.npz [--blocks X,Y ...]   (verifies the phases of a dump, see --dump)
       dec0_ref.py CAPTURE FRAME OUT.npz --extract DUMP.npz      (weights from a phase dump)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rrmath import e4m3, f16, q8  # noqa: E402
from enc1_ref import inter, mm, softmax_cubic, act, mirror, add, H, F  # noqa: E402

I32 = inter(32)
UN32 = [I32.index(i) for i in range(32)]
I96 = inter(96)


def norm_plane(xn, gain):
    """RMS norm of one 32-channel token (natural order) in NVIDIA's tree over the arena order; returns codes in k order"""
    xa = xn[I32]
    sq = (xa * xa).astype(H)
    h = []
    for t in range(4):
        def pair(w):
            return add(sq[4 * w:4 * w + 2], sq[4 * w + 2:4 * w + 4])
        h.append(add(pair(t), pair(t + 4)))
    h01 = [add(h[t], h[t ^ 1]) for t in range(4)]
    h2 = add(h01[0], h01[2])
    s = H(H(h2[0] + h2[1]) + H(2.0 ** -13))
    r = H(F(1) / np.sqrt(F(s)))
    return q8((xn * (r * gain).astype(H)).astype(H))[I32]


def keys(x0, y0):
    border = ([(x, y0 - 1) for x in range(x0, x0 + 8)] + [(x, y0 + 8) for x in range(x0, x0 + 8)]
              + [(x0 - 1, y) for y in range(y0, y0 + 8)] + [(x0 + 8, y) for y in range(y0, y0 + 8)])
    return [(x, y) for y in range(y0, y0 + 8) for x in range(x0, x0 + 8)] + border


def window(w, lo, skip, x0, y0, ex, ey, dbg=None):
    """lo: dec1's output [2][ey/2][ex/2][32] codes; skip: enc0's [ey][ex][32] codes. Returns the residual stream after
    the block (64 x 32 f16, natural channels)"""
    dbg = {} if dbg is None else dbg
    hx, hy = ex // 2, ey // 2
    X0, Y0 = x0 // 2 - 1, y0 // 2 - 1
    low = np.zeros((48, 64), np.uint8)
    for r in range(36):
        X, Y = mirror(X0 + r % 6, hx), mirror(Y0 + r // 6, hy)
        low[r] = np.concatenate([lo[0, Y, X], lo[1, Y, X]])
    Ex = q8(mm(low, w['we'], np.repeat(w['be'][None, :], 48, 0)))
    dbg['p1_A'] = low
    xs, ns = [], []
    for (x, y) in keys(x0, y0):
        xx, yy = mirror(x, ex), mirror(y, ey)
        r = (yy // 2 - Y0) * 6 + (xx // 2 - X0) if False else (y // 2 - Y0) * 6 + (x // 2 - X0)
        s = 2 * (y & 1) + (x & 1)
        xn = (e4m3(Ex[r, 32 * s:32 * s + 32]).astype(H) + e4m3(skip[yy, xx][UN32]).astype(H)).astype(H)
        xs.append(xn)
        ns.append(norm_plane(xn, w['gain0']))
    X = np.array(ns)
    dbg['p2_A'] = X
    R = None
    for h in range(2):
        qk = mm(X, w[f'wk{h}'], np.zeros((96, 32)))
        v = mm(X, w[f'wv{h}'], np.zeros((96, 32)))
        Q = q8(qk)[:, I32]
        S = mm(Q[:64], Q.T, w[f'bias{h}'])
        P = softmax_cubic(S)
        O = mm(q8(P)[:, I96], q8(v)[I96], np.zeros((64, 32)))
        C = add(np.array(xs[:64]), w['table0'][None, :]) if h == 0 else R
        R = mm(q8(O)[:, I32], w[f'wo{h}'], C)
        dbg[f'p{4 + 5 * h}_A'], dbg[f'p{5 + 5 * h}_A'], dbg[f'p{6 + 5 * h}_A'] = Q[:64], q8(P)[:, I96], q8(O)[:, I32]
        dbg[f'p{6 + 5 * h}_C'], dbg[f'p{6 + 5 * h}_D'] = C, R
    N = np.array([norm_plane(R[t], w['gain1']) for t in range(64)])
    acc = add(R, w['table1'][None, :])
    for b in range(4):
        hid = act(mm(N, w[f'w1_{b}'], np.repeat(w[f'b1_{b}'][None, :], 64, 0)))
        dbg[f'p{12 + 2 * b}_A'] = N
        dbg[f'p{13 + 2 * b}_A'], dbg[f'p{13 + 2 * b}_C'] = q8(hid)[:, I32], acc
        acc = mm(q8(hid)[:, I32], w[f'w2_{b}'], acc)
        dbg[f'p{13 + 2 * b}_D'] = acc
    return acc


OUT_CH = [12, 13, 18, 19, 20, 21, 22, 23]    # the 8 FP8 channels of the output stage's input (argument 48)
OUT_CH_D = [12, 13, None, 18, 19, 20, 21, None]   # preset D (..._dec0_diamond_wallaby_kernel): bytes 2 and 7 are zero
TANH_CH = [14, 15, 16, 17]                    # surface 144 (rgba8 snorm)


def fma16(a, b, c):
    return H(np.float64(a) * np.float64(b) + np.float64(c))


def tanh_nv(t):
    """NVIDIA's tanh of the surface-144 channels: an odd polynomial below 0.6, 1 - 2 / (e^2x + 1) above, f32"""
    t = F(t)
    m = abs(t)
    big = F(1) if m >= F(9.0109138) else F(np.float64(F(1) / F(np.exp2(F(m * F(2.8853900432586670))) + F(1))) * -2 + 1)
    s = F(t * t)
    p = F(0.01573968306183815)
    for c in (-0.05230396240949631, 0.13315297663211823, -0.33332768082618713, 0.0):
        p = F(np.float64(p) * np.float64(s) + np.float64(F(c)))
    small = F(np.float64(p) * np.float64(t) + np.float64(t))
    return np.copysign(big, t) if m >= F(0.6) else small


def bilinear88(t88, xi, yi):
    """texture 88 (half resolution, linear) at the full-resolution texel centre (xi, yi): weights 1/4 and 3/4,
    clamped at the edge; f32 sum rounded to binary16 (NVIDIA's texture unit rounds a few of these differently)"""
    def axis(i):
        k = i // 2
        return ((k - 1, 0.25), (k, 0.75)) if i % 2 == 0 else ((k, 0.75), (k + 1, 0.25))
    h, w = t88.shape[:2]
    v = np.zeros(4)
    for (yy, wy) in axis(yi):
        for (xx, wx) in axis(xi):
            v += wx * wy * t88[min(max(yy, 0), h - 1), min(max(xx, 0), w - 1)].astype(np.float64)
    return v.astype(H)


def outputs(acc, t88, t96, t104, x0, y0, ex, ey):
    """per window token: the 8 output codes, surface 144 (4 floats before the snorm store), surface 112 (3 f16)"""
    codes, s144, s112 = [], [], []
    for t in range(64):
        x, y = x0 + t % 8, y0 + t // 8
        a = acc[t].astype(H)
        codes.append(q8(a)[OUT_CH])
        s144.append([tanh_nv(a[c]) for c in TANH_CH])
        lg = a[:9]
        m01 = np.maximum(np.maximum(np.maximum(lg[0:2], lg[2:4]), lg[4:6]), lg[6:8])
        m = max(max(m01[0], m01[1]), lg[8])
        e = [np.exp2(F(H(lg[i] - m)) * F(1.4426950216293335)) for i in range(9)]
        pair = [F(e[2 * k] + e[2 * k + 1]) for k in range(4)]
        sm = F(pair[0] + F(0))
        for k in (1, 2, 3):
            sm = F(sm + pair[k])
        sm = F(sm + e[8])
        r = H(F(1) / sm)
        wg = [H(H(e[i]) * r) for i in range(9)]

        def cl(v, n):
            return 1 if v < 0 else (n - 2 if v >= n else v)
        xs, ys = [cl(x - 1, ex), cl(x, ex), cl(x + 1, ex)], [cl(y - 1, ey), cl(y, ey), cl(y + 1, ey)]
        f = np.zeros(3, H)
        k = 0
        for yy in ys:
            for xx in xs:
                T = bilinear88(t88, xx, yy)
                for c in range(3):
                    f[c] = fma16(T[c], wg[k], f[c])
                k += 1
        g = H(F(1) / (np.exp2(F(a[9]) * F(-1.4426950216293335)) + F(1)))
        om = H(H(1) - g)
        o = np.zeros(3, H)
        for c in range(3):
            if 0 <= x < ex and 0 <= y < ey:
                o[c] = fma16(t96[y, x, c], om, H(f[c] * g))
                if not np.isnan(t104[y, x, c]):
                    o[c] = H(o[c] + t104[y, x, c])
        s112.append(o)
    return np.array(codes), np.array(s144), np.array(s112)


def extract(dump, weights, out):
    z = np.load(dump)
    W = np.frombuffer(weights, np.uint8)

    def B(p):
        return z[f'p{p}_w0_B']

    def C(p):
        return f16(z[f'p{p}_w0_C']).astype(H)
    w = {'we': B(1), 'be': C(1)[0], 'gain0': W[8448:8448 + 64].view(H), 'table0': W[39232:39232 + 64].view(H),
         'gain1': W[39296:39296 + 64].view(H), 'table1': W[41472:41472 + 64].view(H)}
    for h, base in ((0, 2), (1, 7)):
        w[f'wk{h}'], w[f'wv{h}'] = B(base), B(base + 1)
        w[f'bias{h}'] = C(base + 2)
        w[f'wo{h}'] = B(base + 4)
    for b in range(4):
        w[f'w1_{b}'], w[f'b1_{b}'] = B(12 + 2 * b), C(12 + 2 * b)[0]
        w[f'w2_{b}'] = B(13 + 2 * b)
    np.savez_compressed(out, **w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('frame', type=int)
    ap.add_argument('weights')
    ap.add_argument('--extract', metavar='DUMP')
    ap.add_argument('--dump', help='phase dump of --blocks[0] to verify phase by phase')
    ap.add_argument('--blocks', nargs='*', default=['5,5'])
    o = ap.parse_args()
    from rr_capture import Capture
    c = Capture(os.path.expanduser(o.capture))
    l = c.find('swin_dec0', o.frame)
    if o.extract:
        extract(o.extract, l.weights, o.weights)
        return
    ex, ey = (int(v) for v in np.frombuffer(l.args[:8], np.uint32))
    sx, sy = (int(v) for v in np.frombuffer(l.args[80:84], np.uint16))
    ptr = {p.offset: p.delta for p in l.pointers}
    pre = np.frombuffer(l.pre(l.arena_alloc), np.uint8)
    lo = pre[ptr[8]:ptr[8] + 2 * (ex // 2) * (ey // 2) * 32].reshape(2, ey // 2, ex // 2, 32)
    skip = pre[ptr[24]:ptr[24] + ex * ey * 32].reshape(ey, ex, 32)
    w = dict(np.load(o.weights))
    img = {i.offset: i for i in l.images}

    def surf(off, dt=np.float16):
        i = img[off]
        return np.frombuffer(i.path.read_bytes(), dt).reshape(i.height, i.width, i.channels)
    t88, t96, t104, r112, r144 = surf(88), surf(96), surf(104), surf(112), surf(144, np.int8)
    post = np.frombuffer(l.post_buffer(c, l.arena_alloc), np.uint8)
    rout = post[ptr[48]:ptr[48] + ex * ey * 8].reshape(ey, ex, 8)
    for blk in o.blocks:
        bx, by = (int(v) for v in blk.split(','))
        d = {}
        x0, y0 = 8 * bx - sx, 8 * by - sy
        acc = window(w, lo, skip, x0, y0, ex, ey, d)
        codes, s144, s112 = outputs(acc, t88, t96, t104, x0, y0, ex, ey)
        if 'diamond' in l.kernel:
            codes = np.array([[0 if ch is None else q8(acc[t].astype(H))[ch] for ch in OUT_CH_D] for t in range(64)])
        n = ok_c = ok_s = ok_t = 0
        for t in range(64):
            x, y = x0 + t % 8, y0 + t // 8
            if not (0 <= x < ex and 0 <= y < r144.shape[0]):
                continue
            n += 1
            ok_c += np.array_equal(codes[t], rout[y, x])
            ok_t += np.array_equal(np.clip(np.round(np.array(s144[t], np.float64) * 127), -127, 127).astype(np.int8), r144[y, x])
            ok_s += np.array_equal(s112[t], r112[y, x, :3])
        print(f'block {bx},{by}: output codes {ok_c}/{n}, surface 144 {ok_t}/{n}, surface 112 {ok_s}/{n}')
        if o.dump:
            z = np.load(o.dump)
            for k in sorted(d, key=lambda s: (int(s[1:].split('_')[0]), s)):
                ref = z[f'{k.split("_")[0]}_w0_{k.split("_")[1]}']
                ours = np.asarray(d[k])
                ours = ours.view(np.uint16) if ours.dtype == H else ours
                n = min(len(ref), len(ours))
                print(f'{k}: rows equal {int((ref[:n] == ours[:n]).all(1).sum())}/{n}')
            o.dump = None


if __name__ == '__main__':
    main()
