#!/usr/bin/env python3
"""Exact numpy reference of the denoiser's Swin stages enc1-enc5 and dec1-dec4 (cuda_dldn_engine_swin_*_kernel), one
8x8 window at a time, from weights and real inputs only. Generalises engine/rr/enc1_ref.py (encoders) and
engine/rr/dec1_ref.py (decoders); every stage was checked against its capture (main).

A stage has C channels (32 * KC), NH heads of 32 channels, NB = C / 8 MLP branches of C -> 32 -> C:
  prologue  encoders: the 96 key tokens from the arena (C / 32 planes); decoders: the expand of the 6 x 6 tokens
            one level down (C_LO channels -> 4 sub-positions x C) + the skip token. RMS norm (input_norm: per lane word
            the planes summed pairwise (pairs_tree), the two word halves, then lanes ^1, ^2 and the halves) times gain0; the
            residual is the token + table7
  heads     Q/K and V projections over KC k32 steps, scores + bias, the cubic softmax, P V, out-projection onto the
            residual, head after head
  MLP       RMS norm (rowsum_sq) times gain14, residual + table16, NB branches
  outputs   the tokens; encoders: the 2x2 merge (4C -> C_M) and the 64 -> 16 head with enc1's filter (surfaces 112,
            120, 128); decoders: the head with dec0's filter (surface 112)

usage: swin_ref.py CAPTURE STAGE FRAME WEIGHTS.npz [--blocks X,Y ...]   (verifies against the capture)
       swin_ref.py CAPTURE STAGE FRAME OUT.npz --extract DUMP.npz      (weights from a phase dump)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rrmath import e4m3, f16, q8  # noqa: E402
from enc1_ref import (inter, mm, softmax_cubic, act, mirror, add, rowsum_sq, pairs_tree, filter_head,  # noqa: E402
                      MERGE_ORDER, H, F)
from dec0_ref import keys  # noqa: E402
from dec1_ref import filter112  # noqa: E402

# kind, C, C_LO (decoders: channels one level down) or C_M (encoders: merged channels), heads, warps
STAGES = {
    'enc1': ('enc', 64, 96, 2, 1), 'enc2': ('enc', 96, 128, 4, 4), 'enc3': ('enc', 128, 160, 4, 4),
    'enc4': ('enc', 160, 160, 8, 8), 'enc5': ('enc', 160, 0, 8, 8),
    'dec4': ('dec', 160, 160, 8, 8), 'dec3': ('dec', 128, 160, 4, 4), 'dec2': ('dec', 96, 128, 4, 4),
    'dec1': ('dec', 64, 96, 2, 2),
}


def input_norm(xn, gain):
    """xn: one token, natural channels (f16). NVIDIA's tree over the arena order; returns the normalised token
    (natural, f16)"""
    c = len(xn)
    xa = xn[inter(c)]
    sq = (xa * xa).astype(H)

    def pair(p, w):
        return add(sq[32 * p + 4 * w:32 * p + 4 * w + 2], sq[32 * p + 4 * w + 2:32 * p + 4 * w + 4])

    fold = pairs_tree   # (p0 + p1) + p2, (p0 + p1) + (p2 + p3), ((p0 + p1) + (p2 + p3)) + p4
    h = [add(fold([pair(p, t) for p in range(c // 32)]), fold([pair(p, t + 4) for p in range(c // 32)]))
         for t in range(4)]
    h01 = [add(h[t], h[t ^ 1]) for t in range(4)]
    h2 = add(h01[0], h01[2])
    s = H(H(h2[0] + h2[1]) + H(2.0 ** -13))
    r = H(F(1) / np.sqrt(F(s)))
    return (xn * (r * gain).astype(H)).astype(H)


def block(stage, w, x0, y0, ex, ey, inp):
    """the window: returns (tokens 64 x C codes in arena order, merged 16 x C_M codes or None, head 64 x 16 f16).
    inp: 'tok' [C/32][ey][ex][32] (encoders) or 'lo' [C_LO/32][ey/2][ex/2][32] and 'skip' [C/32][ey][ex][32]"""
    kind, C, CX, NH, _ = STAGES[stage]
    IC = inter(C)
    UN = [IC.index(i) for i in range(C)]
    xs = []
    if kind == 'dec':
        hx, hy = ex // 2, ey // 2
        X0, Y0 = x0 // 2 - 1, y0 // 2 - 1
        low = np.zeros((48, CX), np.uint8)
        for r in range(36):
            X, Y = mirror(X0 + r % 6, hx), mirror(Y0 + r // 6, hy)
            low[r] = np.concatenate([inp['lo'][p, Y, X] for p in range(CX // 32)])
        Ex = q8(mm(low, w['we'], np.repeat(w['be'][None, :], 48, 0)))
        for (x, y) in keys(x0, y0):
            xx, yy = mirror(x, ex), mirror(y, ey)
            r, s = (y // 2 - Y0) * 6 + (x // 2 - X0), 2 * (y & 1) + (x & 1)
            sk = np.concatenate([inp['skip'][p, yy, xx] for p in range(C // 32)])[UN]
            xs.append((e4m3(Ex[r, C * s:C * s + C]).astype(H) + e4m3(sk).astype(H)).astype(H))
    else:
        for (x, y) in keys(x0, y0):
            xx, yy = mirror(x, ex), mirror(y, ey)
            xs.append(e4m3(np.concatenate([inp['tok'][p, yy, xx] for p in range(C // 32)])[UN]).astype(H))
    X = np.array([q8(input_norm(x, w['gain0']))[IC] for x in xs])
    R = add(np.array(xs[:64]), w['table7'][None, :])
    I32, I96 = inter(32), inter(96)
    for h in range(NH):
        qk, v = np.zeros((96, 32)), np.zeros((96, 32))
        for c in range(C // 32):
            qk = mm(X[:, 32 * c:32 * c + 32], w[f'wk{h}'][32 * c:32 * c + 32], qk)
            v = mm(X[:, 32 * c:32 * c + 32], w[f'wv{h}'][32 * c:32 * c + 32], v)
        Q = q8(qk)[:, I32]
        P = softmax_cubic(mm(Q[:64], Q.T, w[f'bias{h}']))
        O = mm(q8(P)[:, I96], q8(v)[I96], np.zeros((64, 32)))
        R = mm(q8(O)[:, I32], w[f'wo{h}'], R)
    N = q8((R * (rowsum_sq(R)[:, None] * w['gain14'][None, :]).astype(H)).astype(H))[:, IC]
    acc = add(R, w['table16'][None, :])
    for b in range(C // 8):
        hid = act(mm(N, w[f'w1_{b}'], np.repeat(w[f'b1_{b}'][None, :], 64, 0)))
        acc = mm(q8(hid)[:, I32], w[f'w2_{b}'], acc)
    tokens = q8(acc)[:, IC]
    merged = None
    if kind == 'enc' and CX:
        Mg = np.zeros((16, 4 * C), np.uint8)
        for r, (mx, my) in enumerate(MERGE_ORDER):
            for blk, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                Mg[r, C * blk:C * blk + C] = tokens[(2 * my + dy) * 8 + 2 * mx + dx]
        merged = q8(mm(Mg, w['wm'], np.repeat(w['bm'][None, :], 16, 0)))[:, inter(CX)]
    head = mm(tokens, w['wh'], np.repeat(w['bh'][None, :], 64, 0))
    return tokens, merged, head


def phase_map(stage):
    """dump phase numbers: projections [h][c] = (QK, V), bias[h] (scores phase, wave), out[h], mlp up / down, merge,
    head, expand"""
    kind, C, CX, NH, waves = STAGES[stage]
    KC = C // 32
    m = {'proj': {}, 'bias': {}, 'out': {}}
    if waves == 1:   # heads in sequence: per head 2 KC projections, scores, P V, out-projection
        p = 1
        for h in range(NH):
            for c in range(KC):
                m['proj'][h, c] = (p, p + 1, 0)
                p += 2
            m['bias'][h] = (p, 0)
            m['out'][h] = p + 2
            p += 3
    else:            # one head per warp
        p = 2 if kind == 'dec' else 1
        for h in range(NH):
            for c in range(KC):
                m['proj'][h, c] = (p + 2 * c, p + 2 * c + 1, h)
        p += 2 * KC
        for h in range(NH):
            m['bias'][h] = (p, h)
            m['out'][h] = p + 2 + h
        p += 2 + NH
    m['mlp'] = [(p + 2 * b, p + 2 * b + 1) for b in range(C // 8)]
    p += 2 * (C // 8)
    if kind == 'enc' and CX:
        m['merge'] = p
        p += 1
    m['head'] = p
    return m


def read_b(W, base, K):
    """an FP8 B matrix (K x 32) as NVIDIA stores it in the weight image: 4-byte k groups, offset 512 (kg >> 3) +
    16 (kg & 3) + 4 ((kg >> 2) & 1) + 64 (n & 7) + 8 ((n >> 3) & 1) + 16 K (n >> 4)"""
    b = np.zeros((K, 32), np.uint8)
    for kg in range(K // 4):
        for n in range(32):
            o = base + 512 * (kg >> 3) + 16 * (kg & 3) + 4 * ((kg >> 2) & 1) + 64 * (n & 7) + 8 * ((n >> 3) & 1) + 16 * K * (n >> 4)
            b[4 * kg:4 * kg + 4, n] = W[o:o + 4]
    return b


def merge_weights(z, W, p, waves, C, CX):
    """the merge (4C -> CX) of a multi-warp encoder: warp w computes the 32-column chunks w, w + warps, ...; the dump
    holds each warp's last chunk, so the chunks come from the weight image (located through warp 1's, chunk 1) and the
    dumped ones check the layout"""
    K, size = 4 * C, 4 * C * 32
    W32 = W[:len(W) // 4 * 4].view(np.uint32)
    first = z[f'p{p}_w1_B'][0:4, 0].copy().view(np.uint32)[0]
    pos1 = int(next(i for i in np.nonzero(W32 == first)[0] * 4 if np.array_equal(read_b(W, int(i), K), z[f'p{p}_w1_B'])))
    base0 = pos1 - size
    chunks = [read_b(W, base0 + size * j, K) for j in range(CX // 32)]
    for v in range(waves):
        last = max(j for j in range(CX // 32) if j % waves == v) if any(j % waves == v for j in range(CX // 32)) else None
        if last is not None:
            assert np.array_equal(chunks[last], z[f'p{p}_w{v}_B']), (v, last)
    b1 = f16(z[f'p{p}_w1_C']).astype(H)[0].tobytes()
    bpos1 = W.tobytes().find(b1)
    bias = np.concatenate([W[bpos1 + 64 * (j - 1):bpos1 + 64 * j].view(H) for j in range(CX // 32)])
    for v in range(waves):
        last = max((j for j in range(CX // 32) if j % waves == v), default=None)
        if last is not None:
            assert np.array_equal(bias[32 * last:32 * last + 32].view(np.uint16), z[f'p{p}_w{v}_C'][0]), ('bias', v)
    return np.hstack(chunks), bias


def extract(stage, dump, weights, out, inp, x0, y0, ex, ey):
    """weights from the phase dump; gains and tables are found in the weight image by matching the dump"""
    kind, C, CX, NH, waves = STAGES[stage]
    z = np.load(dump)
    W = np.frombuffer(weights, np.uint8)

    def B(p, wv=0):
        return z[f'p{p}_w{wv}_B']

    def Cm(p, wv=0):
        return f16(z[f'p{p}_w{wv}_C']).astype(H)
    pm = phase_map(stage)
    w = {}
    if kind == 'dec':
        w['we'] = np.hstack([B(1, v) for v in range(waves)])
        w['be'] = np.concatenate([Cm(1, v)[0] for v in range(waves)])
    for h in range(NH):
        w[f'wk{h}'] = np.vstack([B(pm['proj'][h, c][0], pm['proj'][h, c][2]) for c in range(C // 32)])
        w[f'wv{h}'] = np.vstack([B(pm['proj'][h, c][1], pm['proj'][h, c][2]) for c in range(C // 32)])
        w[f'bias{h}'] = Cm(*pm['bias'][h])
        w[f'wo{h}'] = B(pm['out'][h])
    for b, (up, dn) in enumerate(pm['mlp']):
        w[f'w1_{b}'], w[f'b1_{b}'], w[f'w2_{b}'] = B(up), Cm(up)[0], B(dn)
    if 'merge' in pm:
        mw = min(waves, 4)   # the merge runs on (at most) four warps
        if waves > 1 and CX // 32 > mw:   # a warp computes several chunks: read them from the weight image
            w['wm'], w['bm'] = merge_weights(z, W, pm['merge'], mw, C, CX)
        elif waves > 1:
            w['wm'] = np.hstack([B(pm['merge'], v) for v in range(CX // 32)])
            w['bm'] = np.concatenate([Cm(pm['merge'], v)[0] for v in range(CX // 32)])
        else:
            w['wm'], w['bm'] = B(pm['merge']), Cm(pm['merge'])[0]
    w['wh'], w['bh'] = B(pm['head']), Cm(pm['head'])[0]
    # the four f16 vectors: candidates at every even offset, checked against the first token rows of the dump
    tables = [W[o:o + 2 * C].view(H) for o in range(0, len(W) - 2 * C + 1, 2)]
    w0 = {k: v for k, v in w.items()}
    # gain0: the key tokens' norm against the first projection's A (window row 0..)
    xs = key_tokens(stage, w0, x0, y0, ex, ey, inp)
    IC = inter(C)
    A1 = np.hstack([z[f'p{pm["proj"][0, c][0]}_w{pm["proj"][0, c][2]}_A'] for c in range(C // 32)])
    w['gain0'] = next(t for t in tables if np.array_equal(q8(input_norm(xs[0], t))[IC], A1[0]) and
                      np.array_equal(q8(input_norm(xs[5], t))[IC], A1[5]))
    p_out = pm['out'][0]
    C0 = Cm(p_out, 0)
    rows = np.array(xs[:64])
    w['table7'] = next(t for t in tables if np.array_equal(add(rows[:len(C0)][:2], t[None]).view(np.uint16), C0[:2].view(np.uint16)))
    up, dn = pm['mlp'][0]
    Rt = f16(z[f'p{pm["out"][NH - 1]}_w0_D']).astype(H)
    Aup, Cdn = z[f'p{up}_w0_A'], Cm(dn)
    w['gain14'] = next(t for t in tables if np.array_equal(
        q8((Rt[:2] * (rowsum_sq(Rt[:2])[:, None] * t[None]).astype(H)).astype(H))[:, IC], Aup[:2]))
    w['table16'] = next(t for t in tables if np.array_equal(add(Rt[:2], t[None]).view(np.uint16), Cdn[:2].view(np.uint16)))
    np.savez_compressed(out, **w)


def key_tokens(stage, w, x0, y0, ex, ey, inp):
    kind, C, CX, NH, _ = STAGES[stage]
    IC = inter(C)
    UN = [IC.index(i) for i in range(C)]
    xs = []
    if kind == 'dec':
        hx, hy = ex // 2, ey // 2
        X0, Y0 = x0 // 2 - 1, y0 // 2 - 1
        low = np.zeros((48, CX), np.uint8)
        for r in range(36):
            X, Y = mirror(X0 + r % 6, hx), mirror(Y0 + r // 6, hy)
            low[r] = np.concatenate([inp['lo'][p, Y, X] for p in range(CX // 32)])
        Ex = q8(mm(low, w['we'], np.repeat(w['be'][None, :], 48, 0)))
    for (x, y) in keys(x0, y0):
        xx, yy = mirror(x, ex), mirror(y, ey)
        if kind == 'dec':
            r, s = (y // 2 - Y0) * 6 + (x // 2 - X0), 2 * (y & 1) + (x & 1)
            sk = np.concatenate([inp['skip'][p, yy, xx] for p in range(C // 32)])[UN]
            xs.append((e4m3(Ex[r, C * s:C * s + C]).astype(H) + e4m3(sk).astype(H)).astype(H))
        else:
            xs.append(e4m3(np.concatenate([inp['tok'][p, yy, xx] for p in range(C // 32)])[UN]).astype(H))
    return xs


def load(cap, stage, frame):
    from rr_capture import Capture
    c = Capture(os.path.expanduser(cap))
    l = c.find(f'swin_{stage}', frame)
    kind, C, CX, NH, _ = STAGES[stage]
    ex, ey = (int(v) for v in np.frombuffer(l.args[:8], np.uint32))
    sx, sy = (int(v) for v in np.frombuffer(l.args[80:84], np.uint16))
    ptr = {p.offset: p.delta for p in l.pointers}
    pre = np.frombuffer(l.pre(l.arena_alloc), np.uint8)
    post = np.frombuffer(l.post_buffer(c, l.arena_alloc), np.uint8)
    inp = {}
    if kind == 'dec':
        inp['lo'] = pre[ptr[8]:ptr[8] + CX * (ex // 2) * (ey // 2)].reshape(CX // 32, ey // 2, ex // 2, 32)
        inp['skip'] = pre[ptr[24]:ptr[24] + C * ex * ey].reshape(C // 32, ey, ex, 32)
    else:
        inp['tok'] = pre[ptr[8]:ptr[8] + C * ex * ey].reshape(C // 32, ey, ex, 32)
    outs = {'tok': post[ptr[48]:ptr[48] + C * ex * ey].reshape(C // 32, ey, ex, 32)}
    if kind == 'enc' and CX:
        outs['mrg'] = post[ptr[56]:ptr[56] + CX * (ex // 2) * (ey // 2)].reshape(CX // 32, ey // 2, ex // 2, 32)
    img = {i.offset: i for i in l.images}

    def surf(off):
        i = img[off]
        return np.frombuffer(i.path.read_bytes(), np.float16).reshape(i.height, i.width, 4)
    tex = {k: surf(k) for k in img}
    return l, ex, ey, sx, sy, inp, outs, tex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('stage')
    ap.add_argument('frame', type=int)
    ap.add_argument('weights')
    ap.add_argument('--extract', metavar='DUMP')
    ap.add_argument('--dump-block', default='1,1', help='the block the dump was taken of (for --extract)')
    ap.add_argument('--blocks', nargs='*', default=['1,1'])
    o = ap.parse_args()
    l, ex, ey, sx, sy, inp, outs, tex = load(o.capture, o.stage, o.frame)
    kind, C, CX, NH, _ = STAGES[o.stage]
    if o.extract:
        bx, by = (int(v) for v in o.dump_block.split(','))
        extract(o.stage, o.extract, l.weights, o.weights, inp, 8 * bx - sx, 8 * by - sy, ex, ey)
        return
    w = dict(np.load(o.weights))
    for blk in o.blocks:
        bx, by = (int(v) for v in blk.split(','))
        x0, y0 = 8 * bx - sx, 8 * by - sy
        tokens, merged, head = block(o.stage, w, x0, y0, ex, ey, inp)
        inside = [t for t in range(64) if 0 <= x0 + t % 8 < ex and 0 <= y0 + t // 8 < ey]
        tk = sum(np.array_equal(tokens[t], np.concatenate([outs['tok'][p, y0 + t // 8, x0 + t % 8] for p in range(C // 32)]))
                 for t in inside)
        msg = f'block {bx},{by}: tokens {tk}/{len(inside)}'
        if merged is not None:
            mg = sum(np.array_equal(merged[r], np.concatenate([outs['mrg'][p, y0 // 2 + my, x0 // 2 + mx] for p in range(CX // 32)]))
                     for r, (mx, my) in enumerate(MERGE_ORDER) if 0 <= x0 // 2 + mx < ex // 2 and 0 <= y0 // 2 + my < ey // 2)
            msg += f', merged {mg}'
        if kind == 'enc' and not CX:   # the bottleneck: the ungated filter to surface 112 only
            o112, _ = filter_head(head, tex[88], x0, y0, ex, ey, ungated=True)
            ok = sum(np.array_equal(o112[t], tex[112][y0 + t // 8, x0 + t % 8, :3]) for t in inside)
        elif kind == 'enc':
            o112, o120 = filter_head(head, tex[88], x0, y0, ex, ey)
            ok = sum(np.array_equal(o112[t], tex[112][y0 + t // 8, x0 + t % 8, :3]) and
                     np.array_equal(o120[t], tex[120][y0 + t // 8, x0 + t % 8, :3]) for t in inside)
        else:
            s112 = filter112(head, tex[88], tex[96], tex[104], x0, y0, ex, ey)
            ok = sum(np.array_equal(s112[t], tex[112][y0 + t // 8, x0 + t % 8, :3]) for t in inside)
        print(msg + f', surfaces {ok}/{len(inside)}')


if __name__ == '__main__':
    main()
