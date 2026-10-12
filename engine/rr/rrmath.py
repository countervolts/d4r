"""Shared numeric helpers for the Ray Reconstruction port: e4m3 codes, f16, and ZLUDA's m16n8k32 accumulation."""
import numpy as np

_E4M3 = np.zeros(256)
for _c in range(256):
    _s, _e, _m = _c >> 7, (_c >> 3) & 15, _c & 7
    _v = np.nan if (_e == 15 and _m == 7) else (_m * 2.0 ** -9 if _e == 0 else (1 + _m / 8) * 2.0 ** (_e - 7))
    _E4M3[_c] = -_v if _s else _v


def e4m3(codes):
    return _E4M3[np.asarray(codes, np.uint8)]


def f16(bits):
    return np.asarray(bits, np.uint16).view(np.float16).astype(np.float64)


def f16bits(x):
    return np.asarray(x, np.float64).astype(np.float16).view(np.uint16)


def mma_k32(a_codes, b_codes, c_f16bits, ksteps=None):
    """D = A.B + C with ZLUDA's lowering: per 32-deep k-step, f32 sum of the exact products and C, one f16 rounding;
    the next k-step takes that f16 value as C. (f64 stands in for f32: the products and short sums are exact.)"""
    a, b = e4m3(a_codes), e4m3(b_codes)
    acc = f16(c_f16bits)
    for k in range(0, a.shape[1], 32):
        acc = (a[:, k:k + 32] @ b[k:k + 32, :] + acc).astype(np.float32).astype(np.float16).astype(np.float64)
    return f16bits(acc)

# cvt.rn.satfinite.e4m3 of every f16 bit pattern (the hardware conversion, exhaustively equal to PTX's semantics;
# NaN -> 0x7f), see engine/rr/ptx2rr.py h2e4m3_hw
import os as _os
_Q8 = np.load(_os.path.join(_os.path.dirname(__file__), 'f16_to_e4m3.npy'))


def q8(x_f16):
    """f16 values (float array) -> e4m3 codes"""
    return _Q8[np.asarray(x_f16, np.float64).astype(np.float16).view(np.uint16)]
