#!/usr/bin/env python3
"""Exact numpy reference of the denoiser's encoder 1 (cuda_dldn_engine_swin_enc1_kernel), one 8x8 window at a time,
from its weights and real inputs only: the specification the native shader implements.

Derived phase by phase against ZLUDA replays (engine/rr/phase_dump.py), every step bit-exact:
  prologue  the 96 key tokens (the window plus a one-token border, corners excluded; mirrored at the image edge)
            from the arena's two 32-channel planes, f16 RMS norm (NVIDIA's summation order) times the gain at W+0
  attention two heads: QK and V projections (K = 64 in two k32 steps), scores = q8(QK[:64]) q8(QK)^T + bias,
            the cubic-exponent softmax, P V, out-projection onto the residual (input tile + table at W+36992)
  MLP       RMS norm times the gain at W+37120, eight branches of 64 -> 32 (bias, ramp activation) -> 64,
            accumulated onto the residual (+ table at W+41408)
  outputs   q8 tokens to param+48, the 2x2 merge (16 x 256 -> 96 + bias) to param+56, the 64 -> 16 head and its
            3x3 kernel-prediction filter of texture 88 to surfaces 112, 120 and (half resolution) 128

usage: enc1_ref.py CAPTURE FRAME WEIGHTS.npz [--blocks X,Y ...]   (verifies against the capture)
       WEIGHTS.npz: engine/rr/enc1_ref.py --extract DUMP.npz CAPTURE FRAME OUT.npz builds it from a phase dump
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rrmath import e4m3, f16, q8, mma_k32  # noqa: E402

H = np.float16
F = np.float32


def inter(n):
    perm = []
    for b in range(n // 32):
        for m in range(8):
            perm += [32 * b + 2 * m, 32 * b + 2 * m + 1, 32 * b + 16 + 2 * m, 32 * b + 17 + 2 * m]
    return perm


INTER64 = inter(64)
# arena channel order (NVIDIA's A-fragment order) -> natural: natural channel k is arena channel ARENA[k]
ARENA = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27,
         30, 31]
ARENA = ARENA + [32 + c for c in ARENA]


def add(a, b):
    return (a + b).astype(H)


def mm(a_codes, b_codes, c):
    """a (M x K e4m3 codes, NVIDIA k order) . b (K x N codes) + c (M x N f16 values) with ZLUDA's k32 rounding"""
    return f16(mma_k32(a_codes, b_codes, np.asarray(c, H).view(np.uint16))).astype(H)


def nv_rowsum_sq(x, eps=2.0 ** -13):
    """f16(rsqrt(sum of squares + eps)) per row of a 64-column C/D tile, NVIDIA's order"""
    sq = (x * x).astype(H)

    def tile(j, t):
        return sq[:, 8 * j + 2 * t: 8 * j + 2 * t + 2]
    h = []
    for t in range(4):
        a, b = add(tile(0, t), tile(2, t)), add(tile(1, t), tile(3, t))
        c, d = add(tile(4, t), tile(6, t)), add(tile(5, t), tile(7, t))
        h.append(add(add(a, c), add(b, d)))
    h1 = [add(h[t], h[t ^ 1]) for t in range(4)]
    h2 = add(h1[0], h1[2])
    s = add(add(h2[:, 0], h2[:, 1]), np.full(x.shape[0], H(eps)))
    return (F(1) / np.sqrt(s.astype(F))).astype(H)


def input_norm(tok_arena, gain):
    """the prologue's per-token norm: tok_arena (2 planes x 32, arena order) -> f16 normalised token (natural)"""
    sq = [(v * v).astype(H) for v in tok_arena]
    h = []
    for t in range(4):
        def pair(p, w):
            s = sq[p][4 * w: 4 * w + 4]
            return add(s[0:2], s[2:4])
        h.append(add(add(pair(0, t), pair(1, t)), add(pair(0, t + 4), pair(1, t + 4))))
    h01 = [add(h[t], h[t ^ 1]) for t in range(4)]
    h2 = add(h01[0], h01[2])
    s = H(H(h2[0] + h2[1]) + H(2.0 ** -13))
    r = H(F(1) / np.sqrt(F(s)))
    x = np.concatenate(tok_arena)[ARENA]
    return (x * (r * gain).astype(H)).astype(H), x


def softmax_cubic(x):
    """E5: the cubic-exponent softmax of a 64 x 96 score block (natural columns)"""
    def fma16(a, b, c):
        return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(H)
    m = np.clip((x * H(0.017181396484375)).astype(H), H(-0.55615234375), H(0.55615234375))
    t = fma16(m, -m, np.full_like(m, H(0.92724609375)))
    u = fma16(m, t, np.full_like(m, H(1.375)))
    e = ((((u.view(np.uint16).astype(np.uint32)) << 5) & 0x7FE0).astype(np.uint16)).view(H)

    def tile(j, tt, half):
        return e[:, 8 * j + 2 * tt + half]
    s = {}
    for tt in range(4):
        for half in range(2):
            ev = add(add(add(tile(0, tt, half), tile(2, tt, half)), add(tile(4, tt, half), tile(6, tt, half))),
                     add(tile(8, tt, half), tile(10, tt, half)))
            od = add(add(add(tile(1, tt, half), tile(3, tt, half)), add(tile(5, tt, half), tile(7, tt, half))),
                     add(tile(9, tt, half), tile(11, tt, half)))
            s[tt, half] = add(ev, od)
    out = [add(add(s[0, h_], s[1, h_]), add(s[2, h_], s[3, h_])) for h_ in range(2)]
    rs = add(out[0], out[1])
    r = (F(1) / rs.astype(F)).astype(H)
    return (e * r[:, None]).astype(H)


# D4R_RR_FUSE=1: the arithmetic as an NVIDIA card runs it. NVIDIA's compiler contracts a binary16 multiply whose
# product only feeds an add / sub into a fused multiply-add (the kernels give those no rounding modifier); ZLUDA, and
# so the captures, round the product first. build/rr/f/nv has the measurements (ptx_contract.py is the rule).
FUSE = os.environ.get('D4R_RR_FUSE') == '1'


def fma16(a, b, c):
    """binary16 a * b + c with one rounding (the exact value fits a double)"""
    with np.errstate(all='ignore'):
        return (np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64)).astype(H)


def act(x):
    y = np.clip(x, H(-2), H(2)).astype(H)
    if FUSE:
        g = fma16(y, fma16(-np.abs(y), H(0.0810546875), H(0.412109375)), H(0.5))
        return (x * g).astype(H)
    g = add(H(0.5), (y * (H(0.412109375) - (H(0.0810546875) * np.abs(y)).astype(H)).astype(H)).astype(H))
    return (x * g).astype(H)


MERGE_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 2),
               (2, 3), (3, 2), (3, 3)]


def mirror(v, n):
    v = abs(v)
    return min(v, 2 * n - 2 - v)


def window(w, planes, tex, x0, y0, ex, ey, dbg=None):
    """one window: returns (tokens out [64 x 64 codes, arena order], merged [16 x 96 codes, A order],
    surf112 / surf120 [64 x 3 f16], surf128 contributions)"""
    border = ([(x, y0 - 1) for x in range(x0, x0 + 8)] + [(x, y0 + 8) for x in range(x0, x0 + 8)]
              + [(x0 - 1, y) for y in range(y0, y0 + 8)] + [(x0 + 8, y) for y in range(y0, y0 + 8)])
    coords = [(x, y) for y in range(y0, y0 + 8) for x in range(x0, x0 + 8)] + border
    norm, raw = [], []
    for (x, y) in coords:
        xx, yy = mirror(x, ex), mirror(y, ey)
        n_, r_ = input_norm([e4m3(planes[p, yy, xx]).astype(H) for p in (0, 1)], w['gain0'])
        norm.append(n_)
        raw.append(r_)
    X = q8(np.array(norm))[:, INTER64]                       # 96 x 64 codes, NVIDIA k order
    res = np.array(raw[:64])
    dbg = {} if dbg is None else dbg
    dbg['XQ'], dbg['RES0'] = X, add(res, w['table7'][None, :])
    R = None
    for h in range(2):
        qk = mm(X[:, 32:], w[f'wk{h}b'], mm(X[:, :32], w[f'wk{h}a'], np.zeros((96, 32))))
        v = mm(X[:, 32:], w[f'wv{h}b'], mm(X[:, :32], w[f'wv{h}a'], np.zeros((96, 32))))
        Q = q8(qk)[:, inter(32)]
        dbg[f'QK{h}'], dbg[f'V{h}'] = Q, q8(v)[inter(96)]
        S = mm(Q[:64], Q.T, w[f'bias{h}'])
        P = softmax_cubic(S)
        O = mm(q8(P)[:, inter(96)], q8(v)[inter(96)], np.zeros((64, 32)))
        dbg[f'S{h}'], dbg[f'P{h}'], dbg[f'O{h}'] = S, q8(P)[:, inter(96)], O
        C = add(res, w['table7'][None, :]) if h == 0 else R
        R = mm(q8(O)[:, inter(32)], w[f'wo{h}'], C)
        dbg[f'R{h}'] = R
    N = q8((R * (nv_rowsum_sq(R)[:, None] * w['gain14'][None, :]).astype(H)).astype(H))[:, INTER64]
    acc = add(R, w['table16'][None, :])
    for b in range(8):
        hid = act(mm(N, w[f'w1_{b}'], np.repeat(w[f'b1_{b}'][None, :], 64, 0)))
        acc = mm(q8(hid)[:, inter(32)], w[f'w2_{b}'], acc)
    dbg['ACC'] = acc
    tokens = q8(acc)[:, INTER64]
    Mg = np.zeros((16, 256), np.uint8)
    for r, (mx, my) in enumerate(MERGE_ORDER):
        for blk, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            Mg[r, 64 * blk: 64 * blk + 64] = tokens[(2 * my + dy) * 8 + 2 * mx + dx]
    merged = q8(mm(Mg, w['wm'], np.repeat(w['bm'][None, :], 16, 0)))[:, inter(96)]
    head = mm(tokens, w['wh'], np.repeat(w['bh'][None, :], 64, 0))
    s112, s120 = filter_head(head, tex, x0, y0, ex, ey)
    return tokens, merged, s112, s120


def filter_head(D, tex, x0, y0, ex, ey, ungated=False):
    """surfaces 112 and 120 of the encoders (the gate's split of the filtered value); ungated (the bottleneck enc5):
    the filtered value itself, for surface 112"""
    def sig(v):
        return H(F(1) / (np.exp2(F(v) * F(-1.4426950216293335)) + F(1)))

    def ex2(v):
        return np.exp2(F(v) * F(1.4426950216293335))
    out112, out120 = np.zeros((64, 3), H), np.zeros((64, 3), H)
    for tk in range(64):
        x, y = x0 + tk % 8, y0 + tk // 8
        l_ = D[tk].astype(H)
        m01 = np.maximum(np.maximum(np.maximum(l_[0:2], l_[2:4]), l_[4:6]), l_[6:8])
        m = max(max(m01[0], m01[1]), l_[8])
        e = [ex2(H(l_[i] - m)) for i in range(9)]
        pair = [F(e[2 * k] + e[2 * k + 1]) for k in range(4)]
        s = F(F(pair[0]) + F(0))
        for k in (1, 2, 3):
            s = F(s + pair[k])
        s = F(s + e[8])
        r = H(F(1) / s)
        wgt = [H(H(e[i]) * r) for i in range(9)]

        def cl(v, n):
            return 1 if v < 0 else (n - 2 if v >= n else v)
        xm, xc, xp = cl(x - 1, ex), cl(x, ex), cl(x + 1, ex)
        ym, yc, yp = cl(y - 1, ey), cl(y, ey), cl(y + 1, ey)

        def T(xx, yy):
            return tex[yy, xx].astype(H)
        pts = [((xm, ym), (xc, ym)), ((xp, ym), (xm, yc)), ((xc, yc), (xp, yc)), ((xm, yp), (xc, yp))]
        acc = np.zeros((3, 2), H)
        for k, (pa, pb) in enumerate(pts):
            for c in range(3):
                for h_, pt in enumerate((pa, pb)):
                    acc[c, h_] = H(F(T(*pt)[c]) * F(wgt[2 * k + h_]) + F(acc[c, h_]))
        f = [H(acc[c, 0] + acc[c, 1]) for c in range(3)]
        f = [H(F(T(xp, yp)[c]) * F(wgt[8]) + F(f[c])) for c in range(3)]
        if ungated:
            out112[tk] = f
            continue
        g = [sig(l_[9]), sig(l_[10]), sig(l_[11])]
        a = [H(g[c] * f[c]) for c in range(3)]
        out120[tk] = a
        out112[tk] = [H(f[c] - a[c]) for c in range(3)]
    return out112, out120


def extract(dump, cap, frame, out):
    """the weights, from a phase dump of any block (they do not depend on the block) and the weight image"""
    z = np.load(dump)

    def B(p):
        return z[f'p{p}_w0_B']

    def C(p):
        return f16(z[f'p{p}_w0_C']).astype(H)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rr_capture import Capture
    c = Capture(os.path.expanduser(cap))
    W = np.frombuffer(c.find('swin_enc1', frame).weights, np.uint8)

    def table(off, n=64):
        return W[off:off + 2 * n].view(np.float16)
    w = {'gain0': table(0), 'gain14': table(37120), 'table7': table(36992), 'table16': table(41408)}
    for h, base in ((0, 1), (1, 8)):
        w[f'wk{h}a'], w[f'wv{h}a'], w[f'wk{h}b'], w[f'wv{h}b'] = B(base), B(base + 1), B(base + 2), B(base + 3)
        w[f'bias{h}'] = C(base + 4)
        w[f'wo{h}'] = B(base + 6)
    for b in range(8):
        w[f'w1_{b}'], w[f'b1_{b}'] = B(15 + 2 * b), C(15 + 2 * b)[0]
        w[f'w2_{b}'] = B(16 + 2 * b)
    w['wm'], w['bm'] = B(31), C(31)[0]
    w['wh'], w['bh'] = B(32), C(32)[0]
    np.savez_compressed(out, **w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('frame', type=int)
    ap.add_argument('weights')
    ap.add_argument('--extract', metavar='DUMP')
    ap.add_argument('--blocks', nargs='*', default=['5,5'])
    o = ap.parse_args()
    if o.extract:
        extract(o.extract, o.capture, o.frame, o.weights)
        return
    from rr_capture import Capture
    c = Capture(os.path.expanduser(o.capture))
    l = c.find('swin_enc1', o.frame)
    ex, ey = (int(v) for v in np.frombuffer(l.args[:8], np.uint32))
    sx, sy = (int(v) for v in np.frombuffer(l.args[80:84], np.uint16))
    pre = np.frombuffer(l.pre(l.arena_alloc), np.uint8)
    post = np.frombuffer(l.post_buffer(c, l.arena_alloc), np.uint8)
    ptr = {p.offset: p.delta for p in l.pointers}
    planes = pre[ptr[8]:ptr[8] + 2 * ex * ey * 32].reshape(2, ey, ex, 32)
    outp = post[ptr[48]:ptr[48] + 2 * ex * ey * 32].reshape(2, ey, ex, 32)
    mrg = post[ptr[56]:ptr[56] + 3 * (ex // 2) * (ey // 2) * 32].reshape(3, ey // 2, ex // 2, 32)
    img = {i.offset: i for i in l.images}

    def surf(off):
        i = img[off]
        return np.frombuffer(i.path.read_bytes(), np.float16).reshape(i.height, i.width, 4)
    tex, s112, s120 = surf(88), surf(112), surf(120)
    w = dict(np.load(o.weights))
    for blk in o.blocks:
        bx, by = (int(v) for v in blk.split(','))
        x0, y0 = 8 * bx - sx, 8 * by - sy
        tokens, merged, o112, o120 = window(w, planes, tex, x0, y0, ex, ey)
        tk_ok = sum(np.array_equal(tokens[t], np.concatenate([outp[0, y0 + t // 8, x0 + t % 8],
                                                              outp[1, y0 + t // 8, x0 + t % 8]]))
                    for t in range(64) if 0 <= x0 + t % 8 < ex and 0 <= y0 + t // 8 < ey)
        mg_ok = sum(np.array_equal(merged[r], np.concatenate([mrg[p, y0 // 2 + my, x0 // 2 + mx] for p in range(3)]))
                    for r, (mx, my) in enumerate(MERGE_ORDER)
                    if 0 <= x0 // 2 + mx < ex // 2 and 0 <= y0 // 2 + my < ey // 2)
        inside = [t for t in range(64) if 0 <= x0 + t % 8 < ex and 0 <= y0 + t // 8 < ey]
        s_ok = sum(np.array_equal(o112[t], s112[y0 + t // 8, x0 + t % 8, :3]) and
                   np.array_equal(o120[t], s120[y0 + t // 8, x0 + t % 8, :3]) for t in inside)
        print(f'block {bx},{by}: tokens {tk_ok}/{len(inside)}, merged {mg_ok}, surfaces {s_ok}/{len(inside)}')


if __name__ == '__main__':
    main()


def pairs_tree(v):
    """NVIDIA's reduction over a list: (v0 + v1) + v2 for three, (v0 + v1) + (v2 + v3) for four, then the same
    recursively on halves rounded up ((v0 + v1) + (v2 + v3)) + v4 for five"""
    if len(v) == 1:
        return v[0]
    if len(v) == 2:
        return add(v[0], v[1])
    half = 1 << ((len(v) - 1).bit_length() - 1)
    return add(pairs_tree(v[:half]), pairs_tree(v[half:]))


def rowsum_sq(x, eps=2.0 ** -13):
    """nv_rowsum_sq for any C = 8 n8-tiles: per lane word t the tiles of each parity (tile pairs 2 apart, tree),
    the parities, then t ^ 1, t ^ 2 and the halves"""
    sq = (x * x).astype(H)
    nt = x.shape[1] // 8

    def tile(j, t):
        return sq[:, 8 * j + 2 * t: 8 * j + 2 * t + 2]
    h = []
    for t in range(4):
        ev = pairs_tree([add(tile(j, t), tile(j + 2, t)) for j in range(0, nt, 4)] if nt % 4 == 0 else
                        [tile(j, t) for j in range(0, nt, 2)])
        od = pairs_tree([add(tile(j, t), tile(j + 2, t)) for j in range(1, nt, 4)] if nt % 4 == 0 else
                        [tile(j, t) for j in range(1, nt, 2)])
        h.append(add(ev, od))
    h1 = [add(h[t], h[t ^ 1]) for t in range(4)]
    h2 = add(h1[0], h1[2])
    s = add(add(h2[:, 0], h2[:, 1]), np.full(x.shape[0], H(eps)))
    return (F(1) / np.sqrt(s.astype(F))).astype(H)
