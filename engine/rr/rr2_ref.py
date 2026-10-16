#!/usr/bin/env python3
"""Exact numpy reference of preset F's encoder stages (rr2_enc1_kernel ... rr2_enc4_kernel of nvngx_dlssd.dll
310.9), one 8x8 window at a time, from weights and real inputs only; checked against a capture (main).

A stage has C channels and works on windows of 64 tokens (origin 8 * block - shift). Arithmetic is binary16 with FP8
(e4m3) operands in the products, exactly as in presets D and E (enc1_ref.py: mm, the cubic softmax, the activation):

  merge     x = W1 . [the four tokens of the 2x2 group one level up, (0,0) (1,0) (0,1) (1,1)] + b1; window tokens
            outside the level are the mirrored tokens
  heads     per head, all from q8(x): QK = Wqk . x and V = Wv . x (32 channels each, no bias). The scores are
            QK . QK^T + bias: symmetric, so NVIDIA computes the blocks (queries 0-31, keys 0-31), (0-31, 32-63) and
            (32-63, 32-63) and transposes the second for (32-63, 0-31). Cubic softmax over the 64 keys, P . V, and the
            out-projections of all heads accumulate into one sum a (the first one's C is the bias bo)
  residual  r = dq(q8(x)) + a * s1          (dq: the FP8 code back as binary16; s1: binary16 at argument 94)
  MLP       NB branches from q8(r): C -> 32 (+ bias) -> act -> C, accumulated into m (first C: the bias bu)
  residual  y = r + m * s2                  (s2: argument 96); the stage's tokens are q8(y)
  outputs   merged = Wm . [y of the 2x2 group] + bm for the next encoder (q8), and a 16-channel head h = Wh . y + bh
            whose first three channels gate the stage's guide image (argument 48, three binary16 channels per token):
            out = min(guide * (sigmoid(h) * s0), 65504) (s0: argument 92) at argument 64, and its 2x2 mean
            ((a + b) + (c + d) of the quarters) at argument 72 for the next encoder

Arena layouts: tokens are FP8 codes in planes of 32 channels ([plane][y][x][32], NVIDIA's k order within a plane);
the guide images are four binary16 per token.

usage: rr2_ref.py CAPTURE STAGE FRAME WEIGHTS.npz [--blocks X,Y ...]   (verifies against the capture)
       rr2_ref.py CAPTURE STAGE FRAME OUT.npz --extract DUMP.npz      (weights from a phase dump, phase_dump.py)
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rrmath import e4m3, q8  # noqa: E402
from enc1_ref import inter, mm, act, add, pairs_tree, mirror, H, F  # noqa: E402
from rr_capture import Capture  # noqa: E402

# stage: C, channels one level up (merge input), merged channels for the next encoder, heads, MLP branches
STAGES = {'enc0': (32, 0, 0, 2, 4), 'enc1': (64, 32, 96, 2, 8), 'enc2': (96, 0, 128, 4, 12), 'enc3': (128, 0, 160, 4, 16),
          'enc4': (160, 0, 160, 8, 20), 'enc5': (160, 0, 0, 8, 20),
          'dec0_1': (32, 0, 0, 2, 4), 'dec4_1': (160, 0, 0, 8, 20), 'dec3_1': (128, 0, 0, 4, 16), 'dec2_1': (96, 0, 0, 4, 12), 'dec1_1': (64, 0, 0, 2, 8)}
# the decoders' first kernels (decN_0): C, channels one level down, heads, MLP branches. They expand the 4 x 4 tokens
# one level down to the window, e = We . low + be (4C outputs: sub-position (y & 1) * 2 + (x & 1), then channel), and
# start from x = dq(q8(e)) + dq(skip) * s3 (s3: binary16 at argument 98; skip: the encoder's tokens of this level,
# argument 16); then the heads and the MLP as above, the first residual on x itself (not dq(q8(x))). No merge, no guide.
EXPAND = {'dec4_0': (160, 160, 8, 20), 'dec3_0': (128, 160, 4, 16), 'dec2_0': (96, 128, 4, 12), 'dec1_0': (64, 96, 2, 8),
          'dec0_0': (32, 64, 2, 4)}
# the network's sixteen kernels in the DLL's order, with their head counts
ORDER = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4_0', 'dec4_1', 'dec3_0', 'dec3_1', 'dec2_0', 'dec2_1',
         'dec1_0', 'dec1_1', 'dec0_0', 'dec0_1']
HEADS = [2, 2, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2]


def find_tables(dll):
    """byte offsets of the sixteen kernels' relative position tables in nvngx_dlssd.dll: runs of 64 x 64 binary16
    tables (one per head) with a constant diagonal and at most 225 distinct values, in ORDER"""
    a = np.frombuffer(dll[:len(dll) & ~1], np.uint16)
    n = len(a) - 4096
    m = (a[:n] & 0x7fff) != 0
    for k in range(1, 8):
        m &= a[:n] == a[65 * k:65 * k + n]
    m &= a[1:n + 1] == a[66:66 + n]
    groups = []
    for i in np.nonzero(m)[0]:
        t = a[i:i + 4096].reshape(64, 64)
        if not (np.all(t.diagonal() == t[0, 0]) and 50 < len(np.unique(t)) <= 225):
            continue
        if groups and 2 * int(i) - groups[-1][-1] == 8192:
            groups[-1].append(2 * int(i))
        else:
            groups.append([2 * int(i)])
    for start in range(len(groups) - 15):
        if [len(g) for g in groups[start:start + 16]] == HEADS:
            return {name: g[0] for name, g in zip(ORDER, groups[start:start + 16])}
    raise SystemExit('no preset F model found in the DLL (its sixteen relative position table groups)')


def dll_weights(dll, stage):
    """an encoder stage's tensors from the DLL: plain binary16 arrays in natural [out][in] order, one after another.
    Before the tables: {s0, 0, 0, 0} and (enc1) the merge of encoder 0's tokens, W1 then b1. After them, per tensor
    kind all heads / branches together: Wqk, Wv (heads x 32 x C), Wo (C x heads x 32), bo, Wd (branches x 32 x C), bd,
    Wu (C x branches x 32), bu, the merge
    for the next encoder Wm (C_M x 4C) and bm, the guide head Wh (3 x C) and bh. A merge's input index is
    channel * 4 + sub-position; the kernels' bias is (T + T^T) * sqrt(2)."""
    if stage in EXPAND:
        c, c_lo, nh, nb = EXPAND[stage]
        c_up = c_m = 0
    else:
        c, c_up, c_m, nh, nb = STAGES[stage]
    g = find_tables(dll)[stage]
    f = lambda off, n: np.frombuffer(dll[off:off + 2 * n], H)

    def B(mat, merge=0):
        """[out][in] binary16 -> the product's B operand (in x out FP8 codes, the inputs in NVIDIA's k order)"""
        k = mat.shape[1]
        if merge:   # channel * 4 + q -> q * channels + channel
            mat = mat.reshape(mat.shape[0], merge, 4).transpose(0, 2, 1).reshape(mat.shape[0], k)
        return q8(mat).T[inter(k)]
    w = {}
    if c_up:
        o = g - 8 - 2 * (3 * c_up + 4) - 2 * c - 2 * c * 4 * c_up   # before encoder 0's guide head (3 x C_UP, bias, 0)
        w['w1'] = B(f(o, c * 4 * c_up).reshape(c, 4 * c_up), c_up)
        w['b1'] = f(o + 2 * c * 4 * c_up, c)
    if stage in EXPAND:   # We (C x 4 x C_LO) and be (C x 4) end where the tables start
        o = g - 8 * c - 8 * c * c_lo
        we = f(o, 4 * c * c_lo).reshape(c, 4, c_lo).transpose(1, 0, 2).reshape(4 * c, c_lo)
        w['we'] = q8(we).T[inter(c_lo)]
        w['be'] = f(g - 8 * c, 4 * c).reshape(c, 4).T.reshape(-1)
    if stage == 'enc0':
        # encoder 0's embedding of the 43 per-pixel features (the kernel's 64 feature slots, the rest unused): two MLP
        # branches 43 -> 32 -> act -> 32, x = bu + sum: Wd (2 x 32 x 43), bd, Wu (32 x 2 x 32), bu, then 16 bytes
        o = g - 16 - 64 - 4096 - 128 - 2 * 2 * 32 * 43
        wd = np.zeros((2, 32, 64), H); wd[:, :, :43] = f(o, 2 * 32 * 43).reshape(2, 32, 43); o += 2 * 2 * 32 * 43
        w['ewd'] = np.array([B(x) for x in wd])
        w['ebd'] = f(o, 64).reshape(2, 32); o += 128
        w['ewu'] = np.array([B(x) for x in f(o, 2048).reshape(32, 2, 32).transpose(1, 0, 2)]); o += 4096
        w['ebu'] = f(o, 32)
    o = g
    t = f(o, nh * 4096).reshape(nh, 64, 64).astype(np.float64); o += nh * 8192
    w['bias'] = ((t + t.transpose(0, 2, 1)) * np.sqrt(2)).astype(H)
    w['wqk'] = np.array([B(x) for x in f(o, nh * 32 * c).reshape(nh, 32, c)]); o += nh * 64 * c
    w['wv'] = np.array([B(x) for x in f(o, nh * 32 * c).reshape(nh, 32, c)]); o += nh * 64 * c
    w['wo'] = np.array([B(x) for x in f(o, nh * 32 * c).reshape(c, nh, 32).transpose(1, 0, 2)]); o += nh * 64 * c
    w['bo'] = f(o, c); o += 2 * c
    w['wd'] = np.array([B(x) for x in f(o, nb * 32 * c).reshape(nb, 32, c)]); o += nb * 64 * c
    w['bd'] = f(o, nb * 32).reshape(nb, 32); o += nb * 64
    w['wu'] = np.array([B(x) for x in f(o, nb * 32 * c).reshape(c, nb, 32).transpose(1, 0, 2)]); o += nb * 64 * c
    w['bu'] = f(o, c); o += 2 * c
    if c_m:
        w['wm'] = B(f(o, c_m * 4 * c).reshape(c_m, 4 * c), c); o += 2 * c_m * 4 * c
        w['bm'] = f(o, c_m); o += 2 * c_m
    if stage in EXPAND:
        return w
    if stage == 'enc0':   # the merge of its tokens (enc1's W1, b1) comes before its guide head
        o += 2 * 64 * 128 + 2 * 64
    nhd = 4 if stage.startswith('dec') else 3   # the guide head: three gates; decoders also a blend factor
    wh = np.zeros((16, c), H); wh[:nhd] = f(o, nhd * c).reshape(nhd, c); o += 2 * nhd * c
    bh = np.zeros(16, H); bh[:nhd] = f(o, nhd)
    w['wh'], w['bh'] = B(wh), bh
    return w


def dq(x):
    """binary16 values through FP8 and back"""
    return e4m3(q8(x)).astype(H)


def A(x):
    """binary16 rows -> the product's A operand: FP8 codes in NVIDIA's k order"""
    return q8(x)[:, inter(x.shape[1])]


def softmax(s):
    """the cubic-exponent softmax of presets D and E over the 64 keys of a row"""
    def fma16(a, b, c):
        return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(H)
    m = np.clip((s * H(0.017181396484375)).astype(H), H(-0.55615234375), H(0.55615234375))
    t = fma16(m, -m, np.full_like(m, H(0.92724609375)))
    u = fma16(m, t, np.full_like(m, H(1.375)))
    e = ((((u.view(np.uint16).astype(np.uint32)) << 5) & 0x7FE0).astype(np.uint16)).view(H)
    nt = e.shape[1] // 8
    part = {}
    for tt in range(4):
        for half in range(2):
            ev = pairs_tree([e[:, 8 * j + 2 * tt + half] for j in range(0, nt, 2)])
            od = pairs_tree([e[:, 8 * j + 2 * tt + half] for j in range(1, nt, 2)])
            part[tt, half] = add(ev, od)
    out = [add(add(part[0, h], part[1, h]), add(part[2, h], part[3, h])) for h in range(2)]
    r = (F(1) / add(out[0], out[1]).astype(F)).astype(H)
    return (e * r[:, None]).astype(H)


def sigmoid(d):
    """1 / (1 + 2^(-d log2 e)) in f32, rounded to binary16 (the GPU's exp2 and division; see rr_exact.glsl sig)"""
    f = F(1) / (np.exp2(d.astype(F) * F(-1.4426950216293335)).astype(F) + F(1))
    return f.astype(H)


def upsample(low, x, y):
    """the decoders' 2x bilinear fetch of the guide one level down (half-pixel centres, edge clamped), with NVIDIA's
    binary16 rounding: every weight product and the three sums in tap order"""
    def axis(v, n):
        if v == 0:
            return 0, H(0)
        b = (v - 1) // 2
        return b, H(0.25) if v & 1 else H(0.75)
    hh, hw = low.shape[:2]
    bx, wx = axis(x, hw)
    by, wy = axis(y, hh)
    x1, y1 = min(bx + 1, hw - 1), min(by + 1, hh - 1)
    ux, uy = H(1) - wx, H(1) - wy
    m = lambda v, a, b: ((v * a).astype(H) * b).astype(H)
    return add(add(add(m(low[by, bx], ux, uy), m(low[by, x1], wx, uy)), m(low[y1, bx], ux, wy)), m(low[y1, x1], wx, wy))


def window(w, scal, xin, x=None):
    """xin: 64 x 4*C_UP FP8 codes (the merge input, already in k order). Returns y (64 x C binary16), the merged
    tokens (16 x C_M binary16) and the head (64 x 16 binary16)."""
    s0, s1, s2 = scal
    base = None
    if x is not None:   # a decoder's first kernel: the expanded tokens plus the skip; the residual keeps it unquantised
        xa, base = A(x), x
    elif 'w1' in w:
        x = mm(xin, w['w1'], np.tile(w['b1'], (64, 1)))
        xa = A(x)
    else:   # the tokens themselves (already FP8)
        x, xa = e4m3(xin)[:, np.argsort(inter(xin.shape[1]))].astype(H), xin
    acc = np.tile(w['bo'], (64, 1))
    for h in range(len(w['wqk'])):
        qk = mm(xa, w['wqk'][h], np.zeros((64, 32), H))
        v = mm(xa, w['wv'][h], np.zeros((64, 32), H))
        qa = A(qk)
        s = mm(qa, qa.T, w['bias'][h])
        # NVIDIA computes three 32 x 32 blocks and transposes one; the full product is the same values
        p = softmax(s)
        o = mm(A(p), A(v.T).T if False else q8(v)[inter(64), :], np.zeros((64, 32), H))
        acc = mm(A(o), w['wo'][h], acc)
    r = add(dq(x) if base is None else base, (acc * s1).astype(H))
    ra = A(r)
    m = np.tile(w['bu'], (64, 1))
    for j in range(len(w['wd'])):
        d = mm(ra, w['wd'][j], np.tile(w['bd'][j], (64, 1)))
        m = mm(A(act(d)), w['wu'][j], m)
    y = add(r, (m * s2).astype(H))
    ya = A(y)
    # the 2x2 groups of the window: token order (0,0) (1,0) (0,1) (1,1), 16 groups row-major
    g = np.zeros((16, 4 * y.shape[1]), np.uint8)
    for n in range(16):
        gx, gy = n % 4, n // 4
        for q, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            g[n, q * y.shape[1]:(q + 1) * y.shape[1]] = ya[(2 * gy + dy) * 8 + 2 * gx + dx]
    merged = mm(g, w['wm'], np.tile(w['bm'], (16, 1))) if 'wm' in w else None
    head = mm(ya, w['wh'], np.tile(w['bh'], (64, 1))) if 'wh' in w else None
    return y, merged, head


def extract(z, out):
    """weights of enc1 from a phase dump (phases as phase_dump.py numbers them)"""
    P = lambda p, n: z[f'p{p}_w0_{n}']
    Ch = lambda p: P(p, 'C').view(H)
    w = {'w1': P(1, 'B'), 'b1': Ch(1)[0], 'wqk': [], 'wv': [], 'bias': [], 'wo': [], 'bo': Ch(11)[0]}
    for h in range(2):
        b = 2 + 10 * h
        w['wqk'].append(np.vstack([P(b, 'B'), P(b + 2, 'B')]))
        w['wv'].append(np.vstack([P(b + 1, 'B'), P(b + 3, 'B')]))
        tl, tr, br = Ch(b + 4), Ch(b + 5), Ch(b + 7)
        w['bias'].append(np.block([[tl, tr], [tr.T, br]]))
        w['wo'].append(P(b + 9, 'B'))
    w['wd'] = [P(22 + 2 * j, 'B') for j in range(8)]
    w['bd'] = [Ch(22 + 2 * j)[0] for j in range(8)]
    w['wu'] = [P(23 + 2 * j, 'B') for j in range(8)]
    w['bu'] = Ch(23)[0]
    w['wm'], w['bm'], w['wh'], w['bh'] = P(38, 'B'), Ch(38)[0], P(39, 'B'), Ch(39)[0]
    np.savez_compressed(out, **{k: np.array(v) for k, v in w.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('stage')
    ap.add_argument('frame', type=int)
    ap.add_argument('weights')
    ap.add_argument('--extract')
    ap.add_argument('--blocks', nargs='*', default=['5,5'])
    o = ap.parse_args()
    if o.extract:
        return extract(np.load(o.extract), o.weights)
    expand = o.stage in EXPAND
    if expand:
        c, c_lo, heads, nb = EXPAND[o.stage]
        c_up = c_m = 0
    else:
        c, c_up, c_m, heads, nb = STAGES[o.stage]
    w = dll_weights(open(o.weights, 'rb').read(), o.stage) if o.weights.lower().endswith('.dll') else dict(np.load(o.weights))
    cap = Capture(os.path.expanduser(o.capture))
    kernel = f'rr2_{o.stage}_kernel' if o.stage.startswith('enc') else f'rr2_{o.stage[:4]}_kernel_{o.stage[5]}'
    l = [l for l in cap.launches if l.kernel == kernel][o.frame - 1]
    pre = np.frombuffer(l.pre(l.arena_alloc), np.uint8)
    post = np.frombuffer(l.post_buffer(cap, l.arena_alloc), np.uint8)
    ptr = {p.offset: p.delta for p in l.pointers if p.alloc == l.arena_alloc}
    wd, ht = struct.unpack_from('<2I', l.args, 0)
    sx, sy = struct.unpack_from('<2H', l.args, 80)
    scal = np.frombuffer(l.args[92:98], H)
    s3 = np.frombuffer(l.args[98:100], H)[0]
    planes = lambda buf, off, w_, h_, ch: buf[off:off + w_ * h_ * ch].reshape(ch // 32, h_, w_, 32)
    up = (planes(pre, ptr[8], wd // 2, ht // 2, c_lo) if expand else planes(pre, ptr[8], 2 * wd, 2 * ht, c_up) if c_up else
          planes(pre, ptr[8], wd, ht, c))
    skip = planes(pre, ptr[16], wd, ht, c) if expand else None
    guided = 48 in ptr and o.stage.startswith('enc')
    dec_guide = o.stage.startswith('dec') and 56 in ptr
    tok_ref = planes(post, ptr[32], wd, ht, c)
    mer_ref = planes(post, ptr[40], wd // 2, ht // 2, c_m) if c_m else None
    img = lambda buf, off, w_, h_: np.frombuffer(bytes(buf[off:off + w_ * h_ * 8]) + bytes(8), H)[:w_ * h_ * 4].reshape(h_, w_, 4)
    guide, out_ref = (img(pre, ptr[48], wd, ht), img(post, ptr[64], wd, ht)) if guided else (None, None)
    half_ref = img(post, ptr[72], wd // 2, ht // 2) if c_m else None
    if dec_guide:   # the guide one level down (argument 48), the encoder's of this level (56), the output (64)
        g_low, g_enc, out_ref = img(pre, ptr[48], wd // 2, ht // 2), img(pre, ptr[56], wd, ht), img(post, ptr[64], wd, ht)
    ok = True
    for blk in o.blocks:
        bx, by = (int(v) for v in blk.split(','))
        x0, y0 = 8 * bx - sx, 8 * by - sy
        inside = np.array([0 <= x0 + t % 8 < wd and 0 <= y0 + t // 8 < ht for t in range(64)])
        xin = np.zeros((64, 4 * c_up if c_up else c), np.uint8)
        xe = None
        if expand:
            lows = np.array([np.concatenate([up[p, mirror(y0 // 2 + n // 4, ht // 2), mirror(x0 // 2 + n % 4, wd // 2)]
                                             for p in range(c_lo // 32)]) for n in range(16)])
            e = mm(lows, w['we'], np.tile(w['be'], (16, 1)))
            un = np.argsort(inter(c))
            xe = np.zeros((64, c), H)
            for t in range(64):
                tx, ty = t % 8, t // 8
                sub = (ty % 2) * 2 + tx % 2
                sk = np.concatenate([skip[p, mirror(y0 + ty, ht), mirror(x0 + tx, wd)] for p in range(c // 32)])
                xe[t] = add(dq(e[(ty // 2) * 4 + tx // 2, sub * c:(sub + 1) * c]), (e4m3(sk)[un].astype(H) * s3).astype(H))
        for t in range(64 if not expand else 0):   # tokens outside the level are the mirrored ones (no edge repeat), as in presets D and E
            X, Y = mirror(x0 + t % 8, wd), mirror(y0 + t // 8, ht)
            if not c_up:
                xin[t] = np.concatenate([up[p, Y, X] for p in range(c // 32)])
                continue
            for q, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                xin[t, q * c_up:(q + 1) * c_up] = np.concatenate([up[p, 2 * Y + dy, 2 * X + dx] for p in range(c_up // 32)])
        y, merged, head = window(w, scal, xin, xe)
        ya, ma = A(y), (A(merged) if c_m else None)
        bad_t = bad_m = bad_o = bad_h = 0
        out = np.zeros((64, 4), H)
        for t in np.nonzero(inside)[0]:
            X, Y = x0 + t % 8, y0 + t // 8
            bad_t += int(np.sum(np.concatenate([tok_ref[p, Y, X] for p in range(c // 32)]) != ya[t]))
            if dec_guide:
                # out = min(enc * (1 - a) + (upsampled * sigmoid(h[0:3]) * s0) * a, 65504), a = sigmoid(h[3])
                sg = sigmoid(head[t, :4])
                gate = np.array([(sg[0] * scal[0]).astype(H), (sg[1] * scal[0]).astype(H), (sg[2] * scal[0]).astype(H), H(0)])
                gt = (upsample(g_low, X, Y) * gate).astype(H)
                gt[3] = H(0)
                mix = add((g_enc[Y, X] * (H(1) - sg[3])).astype(H), (gt * sg[3]).astype(H))
                out[t] = np.minimum(mix, H(65504))
                bad_o += int(np.sum(out[t].view(np.uint16) != out_ref[Y, X].view(np.uint16)))
            if not guided:
                continue
            gate = (sigmoid(head[t, :3]) * scal[0]).astype(H)
            out[t, :3] = np.minimum((guide[Y, X, :3] * gate).astype(H), H(65504))
            bad_o += int(np.sum(out[t].view(np.uint16) != out_ref[Y, X].view(np.uint16)))
        for n in range(16):
            X, Y = x0 // 2 + n % 4, y0 // 2 + n // 4
            t0 = (2 * (n // 4)) * 8 + 2 * (n % 4)
            if not c_m or not (inside[t0] and x0 % 2 == 0 and y0 % 2 == 0):
                continue
            bad_m += int(np.sum(np.concatenate([mer_ref[p, Y, X] for p in range(c_m // 32)]) != ma[n]))
            qv = [(out[t0 + d] * H(0.25)).astype(H) for d in (0, 1, 8, 9)]
            bad_h += int(np.sum(add(add(qv[0], qv[1]), add(qv[2], qv[3])).view(np.uint16) != half_ref[Y, X].view(np.uint16)))
        print(f'block {blk}: {int(inside.sum())} tokens inside; differing tokens bytes {bad_t}, merged bytes {bad_m}, '
              f'guide values {bad_o}, half guide values {bad_h}')
        ok &= bad_t + bad_m + bad_o + bad_h == 0
    print('IDENTICAL' if ok else 'DIFFERENT')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
