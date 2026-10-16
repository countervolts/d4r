#!/usr/bin/env python3
"""Numpy reference of DLSS 4 (preset K) DltssPaddedWinLayer blocks, built stage by stage against the
PTX interpreter (kernels/tools/ptxsim.py). f16 arithmetic in the kernel's operation order."""
import struct
import numpy as np

F16, F32, F64 = np.float16, np.float32, np.float64

# Legacy replay reports retain their original approximate norm by default.
# Windows strict K validation selects the PTX fragment's rounded reduction.
NORM_FRAGMENT_ORDER = False


def norm_rows(x):
    if not NORM_FRAGMENT_ORDER:
        return f16(np.sum(x.astype(F32).astype(F64) ** 2, axis=1))
    tokens, channels = x.shape
    assert channels in (32, 64, 96, 128, 160)
    square = f16(x.astype(F64) ** 2)
    partial = []
    for lane in range(4):
        halves = []
        for parity in range(2):
            pairs = [add16(square[:, 2 * lane + 8 * parity + 32 * g:2 * lane + 8 * parity + 32 * g + 2],
                           square[:, 2 * lane + 8 * parity + 32 * g + 16:2 * lane + 8 * parity + 32 * g + 18])
                     for g in range(channels // 32)]
            value = pairs[0]
            if channels >= 64:
                value = add16(value, pairs[1])
            if channels == 96:
                value = add16(value, pairs[2])
            if channels >= 128:
                value = add16(value, add16(pairs[2], pairs[3]))
            if channels == 160:
                value = add16(value, pairs[4])
            halves.append(value)
        partial.append(add16(halves[0], halves[1]))
    total = add16(add16(partial[0], partial[1]), add16(partial[2], partial[3]))
    return add16(total[:, 0], total[:, 1])


class Dump:
    def __init__(self, d):
        self.dir = d
        man = open(d + '/manifest.txt').read().split('\n')
        self.kernel = man[0].split()[1]
        la = man[1].split()
        self.grid, self.block = tuple(map(int, la[1:4])), tuple(map(int, la[4:7]))
        self.args = open(d + '/args.bin', 'rb').read()
        self.allocs = {}
        for l in man:
            p = l.split()
            if p and p[0] == 'alloc':
                self.allocs[int(p[1])] = (int(p[2], 16), np.fromfile(f'{d}/alloc-{p[1]}.bin', np.uint8))

    def u64(self, off):
        return struct.unpack_from('<Q', self.args, off)[0]

    def i32x2(self, off):
        return struct.unpack_from('<ii', self.args, off)

    def ptr(self, off):
        a = self.u64(off)
        for i, (base, buf) in self.allocs.items():
            if base <= a < base + buf.size:
                return buf, a - base
        raise ValueError(hex(a))

    def f16(self, off_param, byte, n):
        buf, o = self.ptr(off_param)
        return buf[o + byte:o + byte + 2 * n].view(F16).copy()


def f16(x):
    return np.asarray(x, F64).astype(F16)


def mirror(v, n):
    v = abs(v)
    return min(v, 2 * n - 2 - v)


def load_window(P, bx, by, C):
    """64 tokens (raster order of the 8x8 window) x C channels, f16, mirror padded."""
    W, H = P.i32x2(0)
    sx, sy = P.i32x2(56)
    buf, o = P.ptr(8)
    x = np.zeros((64, C), F16)
    for r in range(64):
        ty, tx = r >> 3, r & 7
        Y = mirror(8 * by - sy + ty, H)
        X = mirror(8 * bx - sx + tx, W)
        a = o + 2 * ((Y * W + X) * C)
        x[r] = buf[a:a + 2 * C].view(F16)
    return x


# --- reductions in the kernel's order ---------------------------------------------------------
def sumsq_rows(x):
    """f16 sum of squares per token, reproducing the kernel's tree for C = 64 (per lane: 16 products
    over 8 words (t + 4k), pairwise f16x2 adds, butterfly over the 4 lanes of a row, final hi+lo)."""
    T, C = x.shape
    assert C == 64
    sq = f16(x.astype(F64) * x.astype(F64))  # f16 products
    out = np.zeros(T, F16)
    for r in range(T):
        lanes = []
        for t in range(4):
            # word k holds channels 2(t+4k), 2(t+4k)+1 as an f16 pair
            w = [np.array([sq[r, 2 * (t + 4 * k)], sq[r, 2 * (t + 4 * k) + 1]], F16) for k in range(8)]
            a = [f16(w[2 * i].astype(F64) + w[2 * i + 1].astype(F64)) for i in range(4)]  # (0+1),(2+3),(4+5),(6+7)
            b = [f16(a[0].astype(F64) + a[2].astype(F64)), f16(a[1].astype(F64) + a[3].astype(F64))]
            lanes.append(f16(b[0].astype(F64) + b[1].astype(F64)))
        s1 = [f16(lanes[t].astype(F64) + lanes[t ^ 1].astype(F64)) for t in range(4)]
        s2 = f16(s1[0].astype(F64) + s1[2].astype(F64))
        out[r] = f16(s2[0].astype(F64) + s2[1].astype(F64))
    return out


def rsqrt16(s):
    return f16(1.0 / np.sqrt(s.astype(F32).astype(F64)))


def mma(A, B, C):
    """one wmma k16 step: D = f16(A @ B + C) with exact sums (the interpreter's semantics)."""
    return f16(A.astype(F64) @ B.astype(F64) + C.astype(F64))


def frag_offset(k, n):
    """byte offset of logical B[k][n] (k, n < 16) inside one 512-byte pre-swizzled f16 fragment"""
    return 64 * (n & 7) + 16 * ((k & 7) >> 1) + 8 * (n >> 3) + 4 * (k >> 3) + 2 * (k & 1)


FRAG = np.array([[frag_offset(k, n) for n in range(16)] for k in range(16)])


def weights(P, byte, K, N, nt_stride=None):
    """logical W[K][N] from fragments at weights + byte + 512 * kt + nt_stride * nt"""
    buf, o = P.ptr(64)
    nt_stride = nt_stride or 512 * (K // 16)
    W = np.zeros((K, N), F16)
    for kt in range(K // 16):
        for nt in range(N // 16):
            base = o + byte + 512 * kt + nt_stride * nt
            vals = buf[(base + FRAG)[..., None] + np.arange(2)].reshape(16, 16, 2).view(F16)[..., 0]
            W[16 * kt:16 * kt + 16, 16 * nt:16 * nt + 16] = vals
    return W


def gemm(A, W, C0=None):
    """chained wmma k16 steps (f16 rounding after each step), per 16x16 output tile"""
    M, K = A.shape
    N = W.shape[1]
    D = np.zeros((M, N), F16) if C0 is None else C0.astype(F16).copy()
    for kt in range(K // 16):
        D = mma(A[:, 16 * kt:16 * kt + 16], W[16 * kt:16 * kt + 16], D)
    return D


def hv(bits):
    return np.array(bits, np.uint16).view(F16)


def fma16(a, b, c):
    return f16(a.astype(F64) * b.astype(F64) + c.astype(F64))


def add16(a, b):
    return f16(a.astype(F64) + b.astype(F64))


def mul16(a, b):
    return f16(a.astype(F64) * b.astype(F64))


def softmax_exp(S):
    """e = ex2(t * (1 + a t^2) + c), t = clamp(S, +-21.64) (f16x2 ops of the kernel)"""
    hi, lo = hv(0x4D69), hv(0xCD69)
    a, one, c = hv(0x91D5), hv(0x3C00), hv(0xC50D)
    t = np.maximum(np.minimum(S, hi), lo)
    u = mul16(t, t)
    v = fma16(u, a, one)
    w = fma16(t, v, c)
    return f16(np.exp2(w.astype(F64)))


def gelu16(x):
    """x * (0.5 + c (0.41216 - 0.081081 |c|)), c = clamp(x, +-2) (f16x2 ops of the kernel)"""
    k1, k2 = f16(np.float32(np.uint32(0x3ED306EB).view(F32))), f16(np.float32(np.uint32(0x3DA60DD6).view(F32)))
    half, two = F16(0.5), F16(2.0)
    c = np.minimum(np.maximum(x, -two), two)
    u = f16(k1.astype(F64) - mul16(np.full_like(c, k2), np.abs(c)).astype(F64))
    return mul16(x, add16(np.full_like(c, half), mul16(c, u)))


def bias_table(P, byte):
    """64x64 f16 table stored in C-fragment order, 512 bytes per 16x16 tile (tile 4 mt + nt)"""
    buf, o = P.ptr(64)
    B = np.zeros((64, 64), F16)
    for i in range(64):
        for j in range(64):
            mt, nt, r, c = i >> 4, j >> 4, i & 15, j & 15
            off = 512 * (4 * nt + mt) + 64 * (r & 7) + 16 * ((c & 7) >> 1) + 2 * (c & 1) + 4 * (r >> 3) + 8 * (c >> 3)
            B[i, j] = buf[o + byte + off:o + byte + off + 2].view(F16)[0]
    return B


def vec(P, byte, n):
    return P.f16(64, byte, n)


def pwin_enc2(P, bx, by, pm_order=((0, 0), (0, 1), (1, 0), (1, 1)), b2_in_c0=True, dbg=None):
    """logical enc2 layer (H 2, C 64 -> merged 96) for one window; returns (y [64 x 64], merged [16 x 96])"""
    C, H, Dh = 64, 2, 32
    x0 = load_window(P, bx, by, C)
    s = norm_rows(x0)
    r = rsqrt16(s)
    h1 = mul16(x0, mul16(r[:, None], vec(P, 0, C)[None, :]))
    Os = []
    for h in range(H):
        hb = 128 + 12288 * h
        q, k, v = [gemm(h1[:, 32:64], weights(P, hb + 2048 * (j + 3), 32, 32, 1024),
                        gemm(h1[:, 0:32], weights(P, hb + 2048 * j, 32, 32, 1024))) for j in range(3)]
        S = gemm(q, k.T.copy(), bias_table(P, 24704 + 8192 * h))
        E = softmax_exp(S)
        rs = f16(np.sum(E.astype(F64), axis=1))
        rc = f16(1.0 / rs.astype(F32).astype(F64))
        Pm = mul16(E, rc[:, None])
        Os.append(gemm(Pm, v))
        if dbg is not None:
            dbg[f'q{h}'], dbg[f'k{h}'], dbg[f'v{h}'], dbg[f'S{h}'], dbg[f'P{h}'], dbg[f'O{h}'] = q, k, v, S, Pm, Os[-1]
    O = np.concatenate(Os, axis=1)
    bo = vec(P, 41088, C)
    Y = gemm(O[:, 32:64], weights(P, 45312, 32, 64, 1024), gemm(O[:, 0:32], weights(P, 41216, 32, 64, 1024), np.tile(bo, (64, 1))))
    x1 = add16(Y, x0)
    if dbg is not None:
        dbg['Y'], dbg['x1'] = Y, x1
    m = mul16(x1, vec(P, 49408, C)[None, :])
    b2 = vec(P, 49536, C)
    acc = add16(x1, np.tile(b2, (64, 1))) if b2_in_c0 else x1
    for c in range(8):
        Hc = gemm(m, weights(P, 49664 + 4096 * c, 64, 32, 2048), np.tile(vec(P, 82432 + 64 * c, 32), (64, 1)))
        acc = gemm(gelu16(Hc), weights(P, 82944 + 4096 * c, 32, 64, 1024), acc)
        if dbg is not None:
            dbg[f'H{c}'], dbg[f'acc{c}'] = Hc, acc
    y = acc
    # patch merge: 16 merged tokens (4x4), K = 4 sub-tokens x 64
    A = np.zeros((16, 256), F16)
    for mt in range(16):
        my, mx = mt >> 2, mt & 3
        for q, (dy, dx) in enumerate(pm_order):
            A[mt, 64 * q:64 * q + 64] = y[8 * (2 * my + dy) + 2 * mx + dx]
    merged = np.concatenate([gemm(A, weights(P, 115712 + 24576 * w, 256, 48, 8192), np.tile(vec(P, 164864 + 96 * w, 48), (16, 1)))
                             for w in range(2)], axis=1)
    return y, merged


class Layout:
    """weight-buffer layout of a plain DltssPaddedWinLayer (encoder with patch merge), bytes"""
    def __init__(self, H, C, COUT):
        self.H, self.C, self.COUT = H, C, COUT
        self.G1 = 0
        self.QKV = 2 * C
        self.HEAD = 192 * C
        self.BIAS = self.QKV + self.HEAD * H
        self.BO = self.BIAS + 8192 * H
        self.WO = self.BO + 2 * C
        self.G2 = self.WO + 64 * C * H
        self.B2 = self.G2 + 2 * C
        self.W1 = self.B2 + 2 * C
        self.B1 = self.W1 + 8 * C * C
        self.W2 = self.B1 + 8 * C
        self.PM = self.W2 + 8 * C * C
        self.NPW = COUT // H                  # merged channels per original warp
        self.NPA = -(-self.NPW // 16) * 16    # allocated (padded to whole 16-column tiles)
        self.PMB = self.PM + 8 * C * self.NPA * H


def swin_core(P, L, x0, dbg=None, posattn=False):
    """norm -> attention -> Wo + residual -> MLP (+ residual): the block output y [64 x C]"""
    C, H = L.C, L.H
    NCH = C // 32
    s = norm_rows(x0)
    r = rsqrt16(s)
    h1 = mul16(x0, mul16(r[:, None], vec(P, L.G1, C)[None, :]))
    Os = []
    for h in range(H):
        hb = L.QKV + L.HEAD * h
        proj = []
        for j in range(3):
            acc = None
            for ch in range(NCH):
                acc = gemm(h1[:, 32 * ch:32 * ch + 32], weights(P, hb + 2048 * (3 * ch + j), 32, 32, 1024), acc)
            proj.append(acc)
        q, k, v = proj
        if posattn:
            # position-only attention (trait flag 2): the table is the probability matrix itself
            Os.append(gemm(bias_table(P, L.BIAS + 8192 * h), v))
            if dbg is not None:
                dbg[f'v{h}'], dbg[f'O{h}'] = v, Os[-1]
            continue
        S = gemm(q, k.T.copy(), bias_table(P, L.BIAS + 8192 * h))
        E = softmax_exp(S)
        rs = f16(np.sum(E.astype(F64), axis=1))
        rc = f16(1.0 / rs.astype(F32).astype(F64))
        Os.append(gemm(mul16(E, rc[:, None]), v))
        if dbg is not None:
            dbg[f'q{h}'], dbg[f'k{h}'], dbg[f'v{h}'], dbg[f'O{h}'], dbg[f'S{h}'], dbg[f'E{h}'], dbg[f'rs{h}'] = q, k, v, Os[-1], S, E, rs
    Y = np.tile(vec(P, L.BO, C), (64, 1))
    for h in range(H):
        Y = gemm(Os[h], weights(P, L.WO + 64 * C * h, 32, C, 1024), Y)
    x1 = add16(Y, x0)
    m = mul16(x1, vec(P, L.G2, C)[None, :])
    acc = add16(x1, np.tile(vec(P, L.B2, C), (64, 1)))
    if dbg is not None:
        dbg['Y'], dbg['x1'] = Y, x1
    for c in range(C // 8):
        Hc = gemm(m, weights(P, L.W1 + 64 * C * c, C, 32, 32 * C), np.tile(vec(P, L.B1 + 64 * c, 32), (64, 1)))
        acc = gemm(gelu16(Hc), weights(P, L.W2 + 64 * C * c, 32, C, 1024), acc)
        if dbg is not None:
            dbg[f'H{c}'], dbg[f'acc{c}'] = Hc, acc
    return acc


def pwin_encoder(P, bx, by, H, C, COUT, pm_bias_padded=True, posattn=False):
    L = Layout(H, C, COUT)
    x0 = load_window(P, bx, by, C)
    y = swin_core(P, L, x0, posattn=posattn)
    A = np.zeros((16, 4 * C), F16)
    for mt in range(16):
        my, mx = mt >> 2, mt & 3
        for q, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            A[mt, C * q:C * q + C] = y[8 * (2 * my + dy) + 2 * mx + dx]
    Wp = weights(P, L.PM, 4 * C, L.NPA * H, 128 * C)
    bias = vec(P, L.PMB, L.NPA * H) if pm_bias_padded else None
    cols = []
    for w in range(H):
        sl = slice(L.NPA * w, L.NPA * w + L.NPW)
        b = bias[sl] if pm_bias_padded else vec(P, L.PMB + 2 * L.NPW * w, L.NPW)
        cols.append(gemm(A, Wp[:, L.NPA * w:L.NPA * (w + 1)], np.tile(np.concatenate([b, np.zeros(L.NPA - L.NPW, F16)]), (16, 1)))[:, :L.NPW])
    merged = np.concatenate(cols, axis=1)
    return y, merged


class Shifted:
    """view of the weight buffer starting at a byte offset (decoders: core weights follow the expand)"""
    def __init__(self, P, base):
        self.P, self.base = P, base

    def ptr(self, off):
        buf, o = self.P.ptr(off)
        return buf, o + (self.base if off == 64 else 0)

    def f16(self, off_param, byte, n):
        return self.P.f16(off_param, byte + (self.base if off_param == 64 else 0), n)

    def i32x2(self, off):
        return self.P.i32x2(off)


def load_lowres(P, bx, by, CL):
    W, H = P.i32x2(0)
    sx, sy = P.i32x2(56)
    buf, o = P.ptr(8)
    W2, H2 = W // 2, H // 2
    lx0, ly0 = int((8 * bx - sx) / 2), int((8 * by - sy) / 2)
    x = np.zeros((16, CL), F16)
    for r in range(16):
        Y = mirror(ly0 + (r >> 2), H2); X = mirror(lx0 + (r & 3), W2)
        a = o + 2 * ((Y * W2 + X) * CL)
        x[r] = buf[a:a + 2 * CL].view(F16)
    return x


def load_skip(P, bx, by, C):
    W, H = P.i32x2(0)
    sx, sy = P.i32x2(56)
    buf, o = P.ptr(16)
    x = np.zeros((64, C), F16)
    for r in range(64):
        Y = mirror(8 * by - sy + (r >> 3), H); X = mirror(8 * bx - sx + (r & 7), W)
        a = o + 2 * ((Y * W + X) * C)
        x[r] = buf[a:a + 2 * C].view(F16)
    return x


def pwin_decoder(P, bx, by, H, C, CL, order=((0, 0), (0, 1), (1, 0), (1, 1))):
    xl = load_lowres(P, bx, by, CL)
    EXP = 2 * CL * 4 * C
    E = gemm(xl, weights(P, 0, CL, 4 * C, 512 * (CL // 16)), np.tile(vec(P, EXP, 4 * C), (16, 1)))
    skip = load_skip(P, bx, by, C)
    x0 = np.zeros((64, C), F16)
    for r in range(16):
        ly, lx = r >> 2, r & 3
        for q, (dy, dx) in enumerate(order):
            t = 8 * (2 * ly + dy) + 2 * lx + dx
            x0[t] = add16(E[r, q * C:(q + 1) * C], skip[t])
    L = Layout(H, C, C)
    return swin_core(Shifted(P, EXP + 8 * C), L, x0)


def pwin_enc0(P, bx, by, H=2, C=32, COUT=64, CIN=16):
    """first layer: token embedding x0 = in (CIN) W_e + b_e, then the core (position-only attention) and merge"""
    xin = load_window(P, bx, by, CIN)
    x0 = gemm(xin, weights(P, 2 * C, CIN, C, 512 * (CIN // 16)), np.tile(vec(P, 0, C), (64, 1)))
    x0 = np.maximum(x0, F16(0))   # ReLU after the embedding
    CB = 2 * C + 2 * CIN * C
    L = Layout(H, C, COUT)
    Pw = Shifted(P, CB)
    y = swin_core(Pw, L, x0, posattn=True)
    A = np.zeros((16, 4 * C), F16)
    for mt in range(16):
        my, mx = mt >> 2, mt & 3
        for q, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            A[mt, C * q:C * q + C] = y[8 * (2 * my + dy) + 2 * mx + dx]
    Wp = weights(Pw, L.PM, 4 * C, L.NPA * H, 128 * C)
    bias = vec(Pw, L.PMB, L.NPA * H)
    cols = [gemm(A, Wp[:, L.NPA * w:L.NPA * (w + 1)], np.tile(bias[L.NPA * w:L.NPA * (w + 1)], (16, 1)))[:, :L.NPW] for w in range(H)]
    return y, np.concatenate(cols, axis=1)


def pwin_dec0(P, bx, by, H=2, C=32, CL=64, NOUT=40, NOUTA=48):
    """last layer: patch expand + skip, core (position-only attention), output head y W_h + b_h (NOUT of NOUTA)"""
    xl = load_lowres(P, bx, by, CL)
    EXP = 2 * CL * 4 * C
    E = gemm(xl, weights(P, 0, CL, 4 * C, 512 * (CL // 16)), np.tile(vec(P, EXP, 4 * C), (16, 1)))
    skip = load_skip(P, bx, by, C)
    x0 = np.zeros((64, C), F16)
    for r in range(16):
        ly, lx = r >> 2, r & 3
        for q, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            t = 8 * (2 * ly + dy) + 2 * lx + dx
            x0[t] = add16(E[r, q * C:(q + 1) * C], skip[t])
    L = Layout(H, C, C)
    Pw = Shifted(P, EXP + 8 * C)
    y = swin_core(Pw, L, x0, posattn=True)
    HB = L.PM              # head bias, then the head weights
    Wh = weights(Pw, HB + 2 * NOUTA, C, NOUTA, 1024)
    out = gemm(y, Wh, np.tile(vec(Pw, HB, NOUTA), (64, 1)))
    return y, out[:, :NOUT]
