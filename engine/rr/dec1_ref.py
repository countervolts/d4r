#!/usr/bin/env python3
"""Exact numpy reference of the denoiser's decoder 1 (cuda_dldn_engine_swin_dec1_kernel), one 8x8 window at a time.
Derived like engine/rr/enc1_ref.py, phase by phase against ZLUDA replays (engine/rr/phase_dump.py); NVIDIA runs it with
two warps, each one head and half of the expand's columns.

  expand    the 6 x 6 level-2 tokens under the window and its border (dec2's 96 channels, arena order) -> 256 =
            4 sub-positions x 64 channels (+ bias), FP8; sub-position 2 (y & 1) + (x & 1) (dec0_ref)
  prologue  key token = expanded + skip token (enc1's 64-channel output at argument 24), f16; enc1's input norm (two
            planes) times the gain at W+25088; the residual is the token + the table at W+62080
  block     enc1's: two 32-channel heads, eight MLP branches 64 -> 32 -> 64 (gain W+62208, table W+66496)
  outputs   the tokens (argument 48); the 64 -> 16 head and dec0's filter (bilinear texture 88 from level 2, gate,
            textures 96 / 104) to surface 112

usage: dec1_ref.py CAPTURE FRAME WEIGHTS.npz [--blocks X,Y ...]   (verifies against the capture)
       dec1_ref.py CAPTURE FRAME OUT.npz --extract DUMP.npz      (weights from a phase dump)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rrmath import e4m3, f16, q8  # noqa: E402
from enc1_ref import inter, mm, softmax_cubic, act, mirror, add, input_norm, nv_rowsum_sq, H, F  # noqa: E402
from dec0_ref import keys, fma16, bilinear88  # noqa: E402

I32, I64, I96 = inter(32), inter(64), inter(96)
UN64 = [I64.index(i) for i in range(64)]
GAIN0, TABLE7, GAIN14, TABLE16 = 25088, 62080, 62208, 66496


def window(w, lo, skip, x0, y0, ex, ey):
    """lo: dec2's output [3][ey/2][ex/2][32] codes; skip: enc1's [2][ey][ex][32]. Returns (tokens 64 x 64 codes in arena
    order, head 64 x 16 f16)"""
    hx, hy = ex // 2, ey // 2
    X0, Y0 = x0 // 2 - 1, y0 // 2 - 1
    low = np.zeros((48, 96), np.uint8)
    for r in range(36):
        X, Y = mirror(X0 + r % 6, hx), mirror(Y0 + r // 6, hy)
        low[r] = np.concatenate([lo[p, Y, X] for p in range(3)])
    Ex = q8(mm(low, w['we'], np.repeat(w['be'][None, :], 48, 0)))
    xs, ns = [], []
    for (x, y) in keys(x0, y0):
        xx, yy = mirror(x, ex), mirror(y, ey)
        r = (y // 2 - Y0) * 6 + (x // 2 - X0)
        s = 2 * (y & 1) + (x & 1)
        sk = np.concatenate([skip[0, yy, xx], skip[1, yy, xx]])[UN64]
        xn = (e4m3(Ex[r, 64 * s:64 * s + 64]).astype(H) + e4m3(sk).astype(H)).astype(H)
        xs.append(xn)
        xa = xn[I64]
        n, _ = input_norm([xa[:32], xa[32:]], w['gain0'])
        ns.append(q8(n)[I64])
    X = np.array(ns)
    res = np.array(xs[:64])
    R = None
    for h in range(2):
        qk = mm(X[:, 32:], w[f'wk{h}b'], mm(X[:, :32], w[f'wk{h}a'], np.zeros((96, 32))))
        v = mm(X[:, 32:], w[f'wv{h}b'], mm(X[:, :32], w[f'wv{h}a'], np.zeros((96, 32))))
        Q = q8(qk)[:, I32]
        S = mm(Q[:64], Q.T, w[f'bias{h}'])
        P = softmax_cubic(S)
        O = mm(q8(P)[:, I96], q8(v)[I96], np.zeros((64, 32)))
        C = add(res, w['table7'][None, :]) if h == 0 else R
        R = mm(q8(O)[:, I32], w[f'wo{h}'], C)
    N = q8((R * (nv_rowsum_sq(R)[:, None] * w['gain14'][None, :]).astype(H)).astype(H))[:, I64]
    acc = add(R, w['table16'][None, :])
    for b in range(8):
        hid = act(mm(N, w[f'w1_{b}'], np.repeat(w[f'b1_{b}'][None, :], 64, 0)))
        acc = mm(q8(hid)[:, I32], w[f'w2_{b}'], acc)
    tokens = q8(acc)[:, I64]
    head = mm(tokens, w['wh'], np.repeat(w['bh'][None, :], 64, 0))
    return tokens, head


def filter112(head, t88, t96, t104, x0, y0, ex, ey):
    """dec0_ref.outputs' surface 112 on the head's channels 0-8 (logits) and 9 (gate)"""
    out = []
    for t in range(64):
        x, y = x0 + t % 8, y0 + t // 8
        a = head[t].astype(H)
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
        f = np.zeros(3, H)
        k = 0
        for yy in (cl(y - 1, ey), cl(y, ey), cl(y + 1, ey)):
            for xx in (cl(x - 1, ex), cl(x, ex), cl(x + 1, ex)):
                T = bilinear88(t88, xx, yy)
                for c in range(3):
                    f[c] = fma16(T[c], wg[k], f[c])
                k += 1
        g = H(F(1) / (np.exp2(F(a[9]) * F(-1.4426950216293335)) + F(1)))
        om = H(H(1) - g)
        o = np.zeros(3, H)
        if 0 <= x < ex and 0 <= y < ey:
            for c in range(3):
                o[c] = fma16(t96[y, x, c], om, H(f[c] * g))
                if not np.isnan(t104[y, x, c]):
                    o[c] = H(o[c] + t104[y, x, c])
        out.append(o)
    return np.array(out)


def extract(dump, weights, out):
    z = np.load(dump)
    W = np.frombuffer(weights, np.uint8)

    def B(p, wv=0):
        return z[f'p{p}_w{wv}_B']

    def C(p, wv=0):
        return f16(z[f'p{p}_w{wv}_C']).astype(H)

    def table(off, n=64):
        return W[off:off + 2 * n].view(H)
    w = {'we': np.hstack([B(1, 0), B(1, 1)]), 'be': np.concatenate([C(1, 0)[0], C(1, 1)[0]]),
         'gain0': table(GAIN0), 'table7': table(TABLE7), 'gain14': table(GAIN14), 'table16': table(TABLE16)}
    for h in range(2):
        w[f'wk{h}a'], w[f'wv{h}a'], w[f'wk{h}b'], w[f'wv{h}b'] = B(2, h), B(3, h), B(4, h), B(5, h)
        w[f'bias{h}'] = C(6, h)
        w[f'wo{h}'] = B(8 + h)
    for b in range(8):
        w[f'w1_{b}'], w[f'b1_{b}'] = B(10 + 2 * b), C(10 + 2 * b)[0]
        w[f'w2_{b}'] = B(11 + 2 * b)
    w['wh'], w['bh'] = B(26), C(26)[0]
    np.savez_compressed(out, **w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('frame', type=int)
    ap.add_argument('weights')
    ap.add_argument('--extract', metavar='DUMP')
    ap.add_argument('--blocks', nargs='*', default=['5,5'])
    o = ap.parse_args()
    from rr_capture import Capture
    c = Capture(os.path.expanduser(o.capture))
    l = c.find('swin_dec1', o.frame)
    if o.extract:
        extract(o.extract, l.weights, o.weights)
        return
    ex, ey = (int(v) for v in np.frombuffer(l.args[:8], np.uint32))
    sx, sy = (int(v) for v in np.frombuffer(l.args[80:84], np.uint16))
    ptr = {p.offset: p.delta for p in l.pointers}
    pre = np.frombuffer(l.pre(l.arena_alloc), np.uint8)
    post = np.frombuffer(l.post_buffer(c, l.arena_alloc), np.uint8)
    lo = pre[ptr[8]:ptr[8] + 3 * (ex // 2) * (ey // 2) * 32].reshape(3, ey // 2, ex // 2, 32)
    skip = pre[ptr[24]:ptr[24] + 2 * ex * ey * 32].reshape(2, ey, ex, 32)
    outp = post[ptr[48]:ptr[48] + 2 * ex * ey * 32].reshape(2, ey, ex, 32)
    img = {i.offset: i for i in l.images}

    def surf(off):
        i = img[off]
        return np.frombuffer(i.path.read_bytes(), np.float16).reshape(i.height, i.width, 4)
    t88, t96, t104, r112 = surf(88), surf(96), surf(104), surf(112)
    w = dict(np.load(o.weights))
    for blk in o.blocks:
        bx, by = (int(v) for v in blk.split(','))
        x0, y0 = 8 * bx - sx, 8 * by - sy
        tokens, head = window(w, lo, skip, x0, y0, ex, ey)
        s112 = filter112(head, t88, t96, t104, x0, y0, ex, ey)
        inside = [t for t in range(64) if 0 <= x0 + t % 8 < ex and 0 <= y0 + t // 8 < ey]
        tk = sum(np.array_equal(tokens[t], np.concatenate([outp[0, y0 + t // 8, x0 + t % 8], outp[1, y0 + t // 8, x0 + t % 8]]))
                 for t in inside)
        sf = sum(np.array_equal(s112[t], r112[y0 + t // 8, x0 + t % 8, :3]) for t in inside)
        print(f'block {bx},{by}: tokens {tk}/{len(inside)}, surface 112 {sf}/{len(inside)}')


if __name__ == '__main__':
    main()
