#!/usr/bin/env python3
"""Exact numpy reference of the denoiser's encoder 0 (cuda_dldn_engine_swin_enc0_kernel) after its input features,
one 8x8 window at a time. The 32 FP8 features of the 96 key pixels (NVIDIA's per-pixel gather from the game's guides
and the temporal state, which engine/rr/enc0_m.comp keeps as ptx2rr.py's translation) are an input here.

  embedding  F (96 x 32) -> q8(relu(F W1 + b1)) -> W2 + b2: the key tokens x (32 channels, f16)
  block      dec0_ref's: RMS norm (gain W+6272), residual x + table W+37056, two heads, four MLP branches
  outputs    the tokens (argument 48, dec0's skip), the 2x2 merge 128 -> 64 (argument 56, enc1's input) and the
             32 -> 16 head with enc1's filter to surfaces 112, 120 and 128

usage: enc0_ref.py CAPTURE FRAME WEIGHTS.npz --dump DUMP.npz [--block X,Y]    (verifies the dump's block)
       enc0_ref.py CAPTURE FRAME OUT.npz --extract DUMP.npz                   (weights from a phase dump)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rrmath import f16, q8  # noqa: E402
from enc1_ref import inter, mm, softmax_cubic, act, add, filter_head, MERGE_ORDER, H  # noqa: E402
from dec0_ref import norm_plane  # noqa: E402

I32, I64, I96 = inter(32), inter(64), inter(96)
GAIN0, TABLE0, GAIN1, TABLE1 = 6272, 37056, None, None


def block(w, feat):
    """feat: the 96 key pixels' features (codes, NVIDIA's k order). Returns (tokens 64 x 32 codes in arena order,
    merged 16 x 64 codes in arena order, head 64 x 16 f16, dbg)"""
    dbg = {}
    e1 = mm(feat, w['w1'], np.repeat(w['b1'][None, :], 96, 0))
    x = mm(q8(np.maximum(e1, H(0)))[:, I32], w['w2'], np.repeat(w['b2'][None, :], 96, 0))
    dbg['x'] = x
    X = np.array([norm_plane(x[k], w['gain0']) for k in range(96)])
    R = None
    for h in range(2):
        qk = mm(X, w[f'wk{h}'], np.zeros((96, 32)))
        v = mm(X, w[f'wv{h}'], np.zeros((96, 32)))
        Q = q8(qk)[:, I32]
        P = softmax_cubic(mm(Q[:64], Q.T, w[f'bias{h}']))
        O = mm(q8(P)[:, I96], q8(v)[I96], np.zeros((64, 32)))
        C = add(x[:64], w['table0'][None, :]) if h == 0 else R
        R = mm(q8(O)[:, I32], w[f'wo{h}'], C)
    N = np.array([norm_plane(R[t], w['gain1']) for t in range(64)])
    acc = add(R, w['table1'][None, :])
    for b in range(4):
        hid = act(mm(N, w[f'w1_{b}'], np.repeat(w[f'b1_{b}'][None, :], 64, 0)))
        acc = mm(q8(hid)[:, I32], w[f'w2_{b}'], acc)
    tokens = q8(acc)[:, I32]
    Mg = np.zeros((16, 128), np.uint8)
    for r, (mx, my) in enumerate(MERGE_ORDER):
        for blk, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            Mg[r, 32 * blk:32 * blk + 32] = tokens[(2 * my + dy) * 8 + 2 * mx + dx]
    merged = q8(mm(Mg, w['wm'], np.repeat(w['bm'][None, :], 16, 0)))[:, I64]
    head = mm(tokens, w['wh'], np.repeat(w['bh'][None, :], 64, 0))
    return tokens, merged, head, dbg


def extract(dump, weights, out):
    z = np.load(dump)
    W = np.frombuffer(weights, np.uint8)

    def B(p):
        return z[f'p{p}_w0_B']

    def C(p):
        return f16(z[f'p{p}_w0_C']).astype(H)
    w = {'w1': B(1), 'b1': C(1)[0], 'w2': B(2), 'b2': C(2)[0], 'gain0': W[GAIN0:GAIN0 + 64].view(H),
         'table0': W[TABLE0:TABLE0 + 64].view(H)}
    # the input norm's gain and the residual table sit elsewhere in preset D's (smaller) weight image: when the preset
    # E offsets do not reproduce the dump, find them by matching it, like the MLP's below
    x = mm(q8(np.maximum(mm(z['p1_w0_A'], w['w1'], np.repeat(w['b1'][None, :], 96, 0)), H(0)))[:, I32], w['w2'],
           np.repeat(w['b2'][None, :], 96, 0))
    cands = [W[o:o + 64].view(H) for o in range(0, len(W) - 63, 2)]
    if not all(np.array_equal(norm_plane(x[k], w['gain0']), z['p3_w0_A'][k]) for k in (0, 9)):
        w['gain0'] = next(t for t in cands if all(np.array_equal(norm_plane(x[k], t), z['p3_w0_A'][k]) for k in (0, 9)))
    if not np.array_equal(add(x[:2], w['table0'][None]).view(np.uint16), C(7)[:2].view(np.uint16)):
        w['table0'] = next(t for t in cands if np.array_equal(add(x[:2], t[None]).view(np.uint16), C(7)[:2].view(np.uint16)))
    for h, base in ((0, 3), (1, 8)):
        w[f'wk{h}'], w[f'wv{h}'] = B(base), B(base + 1)
        w[f'bias{h}'] = C(base + 2)
        w[f'wo{h}'] = B(base + 4)
    for b in range(4):
        w[f'w1_{b}'], w[f'b1_{b}'] = B(13 + 2 * b), C(13 + 2 * b)[0]
        w[f'w2_{b}'] = B(14 + 2 * b)
    w['wm'], w['bm'] = B(21), C(21)[0]
    w['wh'], w['bh'] = B(22), C(22)[0]
    # the MLP norm's gain and table, found by matching the dump
    R = f16(z['p12_w0_D']).astype(H)
    tables = [W[o:o + 64].view(H) for o in range(0, len(W) - 63, 2)]
    w['gain1'] = next(t for t in tables if all(np.array_equal(norm_plane(R[k], t), z['p13_w0_A'][k]) for k in (0, 9)))
    w['table1'] = next(t for t in tables if np.array_equal(add(R[:2], t[None]).view(np.uint16), C(14)[:2].view(np.uint16)))
    np.savez_compressed(out, **w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('frame', type=int)
    ap.add_argument('weights')
    ap.add_argument('--extract', metavar='DUMP')
    ap.add_argument('--dump')
    ap.add_argument('--block', default='5,5')
    o = ap.parse_args()
    from rr_capture import Capture
    c = Capture(os.path.expanduser(o.capture))
    l = c.find('swin_enc0', o.frame)
    if o.extract:
        extract(o.extract, l.weights, o.weights)
        return
    z = np.load(o.dump)
    w = dict(np.load(o.weights))
    ex, ey = (int(v) for v in np.frombuffer(l.args[:8], np.uint32))
    sx, sy = (int(v) for v in np.frombuffer(l.args[80:84], np.uint16))
    bx, by = (int(v) for v in o.block.split(','))
    x0, y0 = 8 * bx - sx, 8 * by - sy
    tokens, merged, head, dbg = block(w, z['p1_w0_A'])
    for p, name in ((3, 'X'), ):
        pass
    ptr = {p.offset: p.delta for p in l.pointers}
    post = np.frombuffer(l.post_buffer(c, l.arena_alloc), np.uint8)
    otok = post[ptr[48]:ptr[48] + ex * ey * 32].reshape(ey, ex, 32)
    omrg = post[ptr[56]:ptr[56] + 2 * (ex // 2) * (ey // 2) * 32].reshape(2, ey // 2, ex // 2, 32)
    img = {i.offset: i for i in l.images}
    tex = {k: np.frombuffer(i.path.read_bytes(), np.float16).reshape(i.height, i.width, 4) for k, i in img.items()
           if k in (112, 120, 160)}
    inside = [t for t in range(64) if 0 <= x0 + t % 8 < ex and 0 <= y0 + t // 8 < ey]
    tk = sum(np.array_equal(tokens[t], otok[y0 + t // 8, x0 + t % 8]) for t in inside)
    mg = sum(np.array_equal(merged[r], np.concatenate([omrg[p, y0 // 2 + my, x0 // 2 + mx] for p in range(2)]))
             for r, (mx, my) in enumerate(MERGE_ORDER))
    print(f'block {bx},{by}: tokens {tk}/{len(inside)}, merged {mg}/16')
    return head, x0, y0, ex, ey


if __name__ == '__main__':
    main()
