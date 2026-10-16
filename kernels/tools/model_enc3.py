#!/usr/bin/env python3
"""Numeric primitives and replay-dump access for swin_model.py (the rrlite Swin blocks of DLSS 4.5, preset M).

Everything here is NVIDIA's arithmetic as the native kernels reproduce it (kernels/m/swin_common.h,
swin_block.h): f16 rounding after every operation, e4m3 (OCP, satfinite, RNE) re-quantisation, one f16 rounding
per k32 MMA step, and the softmax's bit-level exponent trick.
"""
import struct

import numpy as np

F16, F32, F64 = np.float16, np.float32, np.float64


def f16(x):
    """round to f16; the result stays float64 (on the f16 grid), so products and sums of such values are exact
    until the next f16() (numpy would otherwise do f16 arithmetic, e.g. in matmul)"""
    return np.asarray(x, F64).astype(F16).astype(F64)


def _e4m3_table():
    t = np.zeros(256, F64)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = m * 2.0 ** -9
        else:
            v = (1 + m / 8) * 2.0 ** (e - 7)
        t[c] = -v if s else v
    return t


E4M3 = _e4m3_table()
_POS = E4M3[:127]  # codes 0..126: the finite non-negative values, ascending


def q8(x):
    """e4m3 RNE satfinite round trip of f16 values (NaN stays NaN), as f16"""
    h = f16(x).astype(F64)
    a = np.abs(h)
    i = np.clip(np.searchsorted(_POS, a), 1, 126)
    lo, hi = _POS[i - 1], _POS[i]
    # nearest; ties to the even code
    pick_hi = (hi - a < a - lo) | ((hi - a == a - lo) & ((i & 1) == 0))
    v = np.where(pick_hi, hi, lo)
    v = np.where(a >= _POS[126], _POS[126], v)
    v = np.where(a <= 0, 0.0, v)
    v = np.where(np.isnan(h), np.nan, np.copysign(v, h))
    return f16(v)


# softmax constants (swin_block.h stage 2)
C_SCALE = f16(0.0972222164273262 / 5.656854152679443)
CLAMP = F16(0.55615234375)
P1 = f16(0.92730712890625)
P0 = F16(1.375)


def exp_trick(poly):
    """per row, columns (2j, 2j+1): the 32-bit word even | odd << 16 of the f16 bits, << 5, + 0x7FF88000; the even
    column takes its low half, the odd column its high half (as f16 bits)"""
    b = poly.astype(F16).view(np.uint16).astype(np.uint64)
    out = np.empty_like(b)
    even, odd = b[:, 0::2], b[:, 1::2]
    word = ((even | (odd << np.uint64(16))) << np.uint64(5)) + np.uint64(0x7FF88000)
    word &= np.uint64(0xFFFFFFFF)
    out[:, 0::2] = word & np.uint64(0xFFFF)
    out[:, 1::2] = word >> np.uint64(16)
    return out.astype(np.uint16).view(F16)


def row_sum16(w):
    """f16 butterfly over the 16 columns: partner c ^ 8, then ^ 2, ^ 4, ^ 1 (every column ends with the same sum)"""
    s = w.astype(F16)
    c = np.arange(16)
    for m in (8, 2, 4, 1):
        s = f16(s.astype(F64) + s[:, c ^ m].astype(F64))
    return s


def gelu_poly(z):
    z = f16(z)
    cz = np.clip(z, F16(-2.0), F16(2.0))
    inner = f16(f16(0.41216981).astype(F64) - f16(f16(0.08108133).astype(F64) * np.abs(cz).astype(F64)).astype(F64))
    return f16(z.astype(F64) * f16(0.5 + f16(cz.astype(F64) * inner.astype(F64)).astype(F64)).astype(F64))


def mma_chain(A, W, C0=None):
    """chained k32 MMA steps with f16 accumulation: D = f16(A[:, 32k:32k+32] W[32k:32k+32] + D), exact sums"""
    M, K = A.shape
    D = np.zeros((M, W.shape[1]), F64) if C0 is None else f16(C0)
    A = np.asarray(A, F64)
    W = np.asarray(W, F64)
    for k in range(0, K, 32):
        D = f16(A[:, k:k + 32] @ W[k:k + 32] + D.astype(F64))
    return D


class Params:
    """one rrlite Swin-block launch from a replay dump (CommonParams: w 0, sx sy 8, tw th 16, in 24, p32 32,
    out 40, p48 48)"""

    def __init__(self, d):
        man = open(d + '/manifest.txt').read().split('\n')
        self.kernel = man[0].split()[1]
        la = man[1].split()
        self.grid = tuple(map(int, la[1:4]))
        self.args = open(d + '/args.bin', 'rb').read()
        self.allocs = {}
        for line in man:
            p = line.split()
            if p and p[0] == 'alloc':
                self.allocs[int(p[1])] = (int(p[2], 16), np.fromfile(f'{d}/alloc-{p[1]}.bin', np.uint8))
        self.w_ptr, = struct.unpack_from('<Q', self.args, 0)
        self.sx, self.sy, self.W, self.H = struct.unpack_from('<iiii', self.args, 8)
        self.inp, self.p32, self.out, self.p48 = struct.unpack_from('<QQQQ', self.args, 24)
        wbase, wbuf = self.alloc_of(self.w_ptr)
        self.wbuf = wbuf[self.w_ptr - wbase:]
        self.abase, self.arena = self.alloc_of(self.inp)

    def alloc_of(self, a):
        for base, buf in self.allocs.values():
            if base <= a < base + buf.size:
                return base, buf
        raise ValueError(hex(a))

    def f16vec(self, off, n):
        return self.wbuf[off:off + 2 * n].view(F16).copy()

    def token_xy(self, bx, by):
        T = np.arange(64)
        tx = (T & 3) + 4 * ((T >> 4) & 1)
        ty = ((T >> 2) & 3) + 4 * (T >> 5)
        return 8 * bx - self.sx + tx, 8 * by - self.sy + ty

    @staticmethod
    def mirror(v, n):
        v = np.abs(v)
        return np.minimum(v, 2 * n - 2 - v)
