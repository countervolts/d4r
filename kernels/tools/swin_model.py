#!/usr/bin/env python3
"""Parametrised numpy model of the rrlite Swin blocks (enc3 spec generalised); checked stage by stage
against ptxsim MMA captures."""
import numpy as np
from model_enc3 import (q8, f16, Params, E4M3, F16, exp_trick, row_sum16, C_SCALE, CLAMP, P1, P0,
                        gelu_poly, mma_chain)

def gp(c):
    c = np.asarray(c)
    return 32 * (c // 32) + 2 * ((c % 32) // 4) + (c % 2) + 16 * ((c % 4) // 2)

def pair_to_nat(M):
    return M[:, gp(np.arange(M.shape[1]))]

def nat_to_pair(M):
    out = np.empty_like(M); out[:, gp(np.arange(M.shape[1]))] = M; return out

def woff_table(base, Ks, Ns, K, N):
    k = np.arange(K)[:, None]; n = np.arange(N)[None, :]; j = n // 8
    return base + Ks * (k // 32) + Ns * (j // 2) + 64 * (n % 8) + 16 * ((k % 16) // 4) + 8 * (j % 2) + 4 * ((k % 32) // 16) + k % 4

def load_tokens(P, bx, by, planes):
    X, Y = P.token_xy(bx, by); X, Y = P.mirror(X, P.W), P.mirror(Y, P.H)
    out = np.zeros((64, 32 * planes))
    for p in range(planes):
        off = P.inp - P.abase + ((p * P.H + Y) * P.W + X) * 32
        out[:, 32 * p:32 * p + 32] = E4M3[P.arena[off[:, None] + np.arange(32)]]
    return out

def sumsq_tokens(x):
    """Kernel-exact f16 sum of squares of natural-order token rows (planes of 32; lane t holds words t, t+4)."""
    sq = f16(x * x); planes = x.shape[1] // 32
    def add(a, b): return f16(a[0] + b[0]), f16(a[1] + b[1])
    lanes = []
    for t in range(4):
        def pair(pp, word, k):
            c = 32 * pp + 4 * word + 2 * k
            return sq[:, c], sq[:, c + 1]
        def plane_sum(word):
            p = [add(pair(pp, word, 0), pair(pp, word, 1)) for pp in range(planes)]
            if planes == 4: return add(add(p[0], p[1]), add(p[2], p[3]))
            if planes == 3: return add(add(p[0], p[1]), p[2])
            if planes == 2: return add(p[0], p[1])
            raise ValueError(planes)
        lanes.append(add(plane_sum(t), plane_sum(t + 4)))
    a = [(f16(lanes[t][0] + lanes[t ^ 1][0]), f16(lanes[t][1] + lanes[t ^ 1][1])) for t in range(4)]
    b = (f16(a[0][0] + a[2][0]), f16(a[0][1] + a[2][1]))
    return f16(b[0] + b[1])[:, None]

def rms_from_sum(x, s, gamma):
    r = f16(1.0 / np.sqrt(f16(s + f16(2.0 ** -13))))
    return q8(f16(x * f16(r * gamma[None, :])))

def rel_bias(P, off):
    raw = P.f16vec(off, 256)
    i, j = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
    lane = 4 * (i % 8) + (j % 8) // 2; word = i // 8 + 2 * (j // 8)
    return raw[8 * lane + 2 * word + (j % 2)]

def attention(Qq, V, bias):
    S = f16(Qq @ Qq.T + bias)
    t = np.clip(f16(S * C_SCALE), -CLAMP, CLAMP)
    e1 = f16(t * (-t) + P1); poly = f16(t * e1 + P0)
    w = exp_trick(poly); s = row_sum16(w)
    r = f16(np.float32(1.0) / s.astype(np.float32)); Pm = f16(w * r)
    return f16(Pm @ V)

def tree_norm2(x):
    """Kernel-exact f16 sum of squares of pair-ordered C-fragment rows (NT8 = C/8 n8 tiles)."""
    sq = f16(x * x); nt8 = x.shape[1] // 8
    def add(a, b): return f16(a[0] + b[0]), f16(a[1] + b[1])
    lanes = []
    for t in range(4):
        T = [(sq[:, 8 * j + 2 * t], sq[:, 8 * j + 2 * t + 1]) for j in range(nt8)]
        def quad(js): return add(add(T[js[0]], T[js[1]]), add(T[js[2]], T[js[3]]))
        if nt8 == 16:
            even = add(quad([0, 2, 4, 6]), quad([8, 10, 12, 14])); odd = add(quad([1, 3, 5, 7]), quad([9, 11, 13, 15]))
        elif nt8 == 12:
            even = add(quad([0, 2, 4, 6]), add(T[8], T[10])); odd = add(quad([1, 3, 5, 7]), add(T[9], T[11]))
        elif nt8 == 8:
            even = quad([0, 2, 4, 6]); odd = quad([1, 3, 5, 7])
        else:
            raise ValueError(nt8)
        lanes.append(add(even, odd))
    a = [(f16(lanes[t][0] + lanes[t ^ 1][0]), f16(lanes[t][1] + lanes[t ^ 1][1])) for t in range(4)]
    b = (f16(a[0][0] + a[2][0]), f16(a[0][1] + a[2][1]))
    return f16(b[0] + b[1])[:, None]

def layout(C, heads, pre=0):
    """Weight offsets of a Swin block with C channels starting at byte `pre`."""
    HS = 96 * C + 512
    L = dict(g1=pre, head=[pre + 2 * C + HS * h for h in range(heads)])
    E = pre + 2 * C + heads * HS
    L.update(bo=E, g2=E + 2 * C)
    M0 = E + 4 * C; NC = C // 8
    L['b1'] = [M0 if c == 0 else M0 + 66 * C + 64 + (64 * C + 64) * (c - 1) for c in range(NC)]
    L['b2'] = M0 + 64 * C + 64
    L['end'] = M0 + 66 * C + 64 + (64 * C + 64) * (NC - 1)
    return L

def swin_block(P, x0, C, heads, pre=0, stages=None):
    """x0: 64 x C natural (f16 values). Returns (x2 pair order, out natural fp8 values)."""
    fp8 = lambda T: E4M3[P.wbuf[T]]
    L = layout(C, heads, pre)
    h1 = rms_from_sum(x0, sumsq_tokens(x0), P.f16vec(L['g1'], C)[gp(np.arange(C))])
    O = np.zeros((64, 32 * heads))
    for h in range(heads):
        hb = L['head'][h]
        Q = mma_chain(h1, fp8(woff_table(hb, 1024, 512, C, 32)))
        V = mma_chain(h1, fp8(woff_table(hb + 32 * C, 1024, 512, C, 32)))
        B = rel_bias(P, hb + 64 * C)
        for m in range(4):
            O[16 * m:16 * m + 16, 32 * h:32 * h + 32] = attention(q8(Q[16 * m:16 * m + 16]), V[16 * m:16 * m + 16], B)
    Wo = np.zeros((32 * heads, C), np.int64)
    for s in range(heads):
        Wo[32 * s:32 * s + 32] = woff_table(L['head'][s] + 64 * C + 512, 0, 512, 32, C)
    x1 = mma_chain(pair_to_nat(q8(O)), fp8(Wo), C0=f16(nat_to_pair(x0) + P.f16vec(L['bo'], C)[None, :]))
    h2 = pair_to_nat(rms_from_sum(x1, tree_norm2(x1), P.f16vec(L['g2'], C)))
    if stages is not None:
        stages.update(h1=h1, x1=pair_to_nat(x1), h2=h2, O=pair_to_nat(q8(O)))
    x2 = f16(x1 + P.f16vec(L['b2'], C)[None, :])
    idx = gp(np.arange(32))
    for c, B1 in enumerate(L['b1']):
        z = mma_chain(h2, fp8(woff_table(B1, 512, 16 * C, C, 32)), C0=np.broadcast_to(P.f16vec(B1 + 32 * C, 32), (64, 32)).copy())
        g = q8(gelu_poly(z))[:, idx]
        x2 = f16(g @ fp8(woff_table(B1 + 32 * C + 64, 0, 512, 32, C)) + x2)
    return x2, q8(pair_to_nat(x2)), L

TOK_X = (np.arange(64) & 3) + 4 * ((np.arange(64) >> 4) & 1)
TOK_Y = ((np.arange(64) >> 2) & 3) + 4 * (np.arange(64) >> 5)

def patch_merge_input(out, C):
    """16 merged rows (r -> (r%4, r//4)) x 4C: column C*q + ch = out[token(2mx+q%2, 2my+q//2), ch]."""
    A = np.zeros((16, 4 * C))
    for r in range(16):
        mx, my = r % 4, r // 4
        for q in range(4):
            t = np.nonzero((TOK_X == 2 * mx + (q & 1)) & (TOK_Y == 2 * my + (q >> 1)))[0][0]
            A[r, C * q:C * q + C] = out[t]
    return A

def load_at(P, ptr, bx, by, planes, W=None, H=None, low=False):
    W = W or P.W; H = H or P.H
    X, Y = P.token_xy(bx, by); X, Y = P.mirror(X, W), P.mirror(Y, H)
    out = np.zeros((64, 32 * planes))
    for p in range(planes):
        off = ptr - P.abase + ((p * H + Y) * W + X) * 32
        out[:, 32 * p:32 * p + 32] = E4M3[P.arena[off[:, None] + np.arange(32)]]
    return out

def decoder_input(P, bx, by, C, CIN, NW, skip_ptr):
    """x0 = f16(q8(patch_expand(low-res CIN)) + skip), expand output col G = (4C/NW) w + n -> q = G // C, ch = ginv(G % C)."""
    fp8 = lambda T: E4M3[P.wbuf[T]]
    W2, H2 = P.W // 2, P.H // 2
    A = np.zeros((16, CIN))
    for r in range(16):
        MX = P.mirror(np.array((8 * bx - P.sx) // 2 + r % 4), W2); MY = P.mirror(np.array((8 * by - P.sy) // 2 + r // 4), H2)
        for p in range(CIN // 32):
            off = P.inp - P.abase + ((p * H2 + MY) * W2 + MX) * 32
            A[r, 32 * p:32 * p + 32] = E4M3[P.arena[off:off + 32]]
    cols = 4 * C // NW
    E_ = np.concatenate([mma_chain(A, fp8(woff_table(w * CIN * cols, 512, 16 * CIN, CIN, cols)),
                                   C0=np.broadcast_to(P.f16vec(4 * C * CIN + 2 * cols * w, cols), (16, cols)).copy()) for w in range(NW)], 1)
    skip = load_at(P, skip_ptr, bx, by, C // 32)
    x0 = np.zeros((64, C))
    for T in range(64):
        tx, ty = TOK_X[T], TOK_Y[T]; r = tx // 2 + 4 * (ty // 2); q = (tx & 1) + 2 * (ty & 1)
        G = np.arange(C) + q * C          # global expand columns of sub-position q (pair order within C)
        x0[T] = q8(E_[r, G])[gp(np.arange(C))]
    return f16(x0 + skip)
