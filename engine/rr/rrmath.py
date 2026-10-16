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

# cvt.rn.satfinite.e4m3 of every f16 bit pattern. Generate this deterministic
# format table locally; never ship an extracted or generated binary table.
def _f16_to_e4m3_table():
    h = np.arange(65536, dtype=np.uint32)
    mag = h & 0x7fff
    normal = (mag - 0x2000 + 0x3f + ((mag >> 7) & 1)) >> 7
    exponent = mag >> 10
    significand = (mag & 0x3ff) | np.where(exponent != 0, 0x400, 0)
    clamped_exponent = np.clip(exponent, 1, 8)
    shift = 16 - clamped_exponent
    subnormal = (significand + (np.left_shift(1, shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift
    code = np.where(mag >= 0x2400, normal, subnormal)
    code = np.where(mag > 0x5f00, 0x7e, code)
    result = ((h >> 8) & 0x80) | code
    result = np.where(mag > 0x7c00, 0x7f, result)  # All NaNs map to the canonical positive NaN.
    return result.astype(np.uint8)


_Q8 = _f16_to_e4m3_table()


def q8(x_f16):
    """f16 values (float array) -> e4m3 codes"""
    return _Q8[np.asarray(x_f16, np.float64).astype(np.float16).view(np.uint16)]
