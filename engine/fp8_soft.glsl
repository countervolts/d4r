// E4M3 (FP8, the "fn" variant: no infinities, NaN = 0x7f / 0xff) in integer arithmetic, for GPUs without
// VK_EXT_shader_float8 (RDNA2). Values are carried as binary16: every E4M3 value is exact there.
#ifndef D4R_FP8_SOFT_GL
#define D4R_FP8_SOFT_GL
// The code of v, rounded to nearest even and saturated to +-448 (saturatedConvertEXT: an infinity becomes 448,
// NaN stays NaN). Normal results (|v| >= 2^-6) round the f32 mantissa to 3 bits, with the carry running into the
// exponent; below that the subnormal step is 2^-9, and 8 steps are the smallest normal code.
uint e4m3Code(float v)
{
    const uint bits = floatBitsToUint(v), sign = (bits >> 24) & 0x80u;
    if (isnan(v)) return sign | 0x7fu;
    const float a = min(abs(v), 448.0);
    const uint u = floatBitsToUint(a);
    const uint normal = ((u + 0x7ffffu + ((u >> 20) & 1u)) >> 20) - (120u << 3);
    return sign | (a < 0.015625 ? uint(roundEven(a * 512.0)) : normal);
}
// The value of a code as binary16: its seven magnitude bits shifted into the binary16 exponent and mantissa give the
// value times 2^-8 (subnormals included, since both formats' subnormals continue their normals' steps)
float16_t e4m3Value(uint c)
{
    return uint16BitsToFloat16(uint16_t(((c & 0x80u) << 8) | ((c & 0x7fu) << 7))) * float16_t(256.0);
}
// four codes (bytes of a word, low first) as two binary16 pairs: (b0, b1) and (b2, b3)
uvec2 e4m3Unpack4(uint w)
{
    const uint lo = (w & 0xffu) | ((w & 0xff00u) << 8), hi = ((w >> 16) & 0xffu) | ((w >> 8) & 0xff0000u);
    const uint s = 0x00800080u, m = 0x007f007fu;
    return uvec2(packFloat2x16(unpackFloat2x16(((lo & s) << 8) | ((lo & m) << 7)) * f16vec2(256.0)),
                 packFloat2x16(unpackFloat2x16(((hi & s) << 8) | ((hi & m) << 7)) * f16vec2(256.0)));
}
// v rounded to the E4M3 grid (saturating), as binary16
float16_t e4m3Round(float v) { return e4m3Value(e4m3Code(v)); }
// A binary16 pair rounded to the E4M3 grid (saturating), in packed binary16 arithmetic: x + M - M with
// M = 1.5 * 2^(e + 7) for x's exponent e (at least -6: the subnormal step) rounds once, to nearest even, to the E4M3
// step 2^(e - 3); zeros keep their sign. Equal to e4m3Value(e4m3Code(x)) for every binary16 x (NaN aside).
uint e4m3RoundHalf2(f16vec2 x)
{
    const f16vec2 v = clamp(x, f16vec2(-448.0), f16vec2(448.0));
    const uint b = packFloat2x16(v);
    const f16vec2 m = uint16BitsToFloat16(max(unpack16(b) & u16vec2(0x7c00), u16vec2(9u << 10)) + u16vec2((7u << 10) | 0x200u));
    precise const f16vec2 r = (v + m) - m;
    return (packFloat2x16(r) & 0x7fff7fffu) | (b & 0x80008000u);
}
// The same for an f32 value, in f32: M = 1.5 * 2^(e + 20) for f32's 23 mantissa bits, applied as two fmas of
// T = 1.5 * 2^e and 2^20. The rounding is symmetric, so M carries no sign; results that round to zero are +0.
// Equal to e4m3Value(e4m3Code(x)) for every finite f32 x but the sign of zero.
float e4m3GridF32(float x)
{
    const float v = clamp(x, -448.0, 448.0);
    const float t = uintBitsToFloat(max((floatBitsToUint(v) & 0x7f800000u) | 0x400000u, (121u << 23) | 0x400000u));
    precise const float r = fma(t, 1048576.0, v);
    precise const float g = fma(t, -1048576.0, r);
    return g;
}
// two f32 values on the E4M3 grid as a binary16 pair: grid values are exact in binary16, so packHalf2x16's rounding
// mode (RADV: toward zero, one v_cvt_pkrtz_f16_f32) does not matter
uint e4m3PackF32(float a, float b) { return packHalf2x16(vec2(e4m3GridF32(a), e4m3GridF32(b))); }
// the codes of a binary16 pair on the E4M3 grid (e4m3RoundHalf2's result), two bytes, low first: the value times
// 2^-8 has the code's magnitude bits at bit 7 of its binary16 bits
uint e4m3CodesHalf2(uint v)
{
    const u16vec2 h = unpack16(packFloat2x16(unpackFloat2x16(v) * f16vec2(0.00390625))) >> 7;
    const u16vec2 c = (h & u16vec2(0x7f)) | ((h >> 1) & u16vec2(0x80));
    return uint(c.x) | (uint(c.y) << 8);
}
// v rounded to binary16 on the hardware conversion, where a binary16 clamp keeps RADV from folding the f32 -> f16 -> f32
// round trip away (round_half.glsl): equal to roundHalfValue except that magnitudes from 65520 saturate to 65504
// instead of becoming infinities
float roundHalfSat(float v) { return float(clamp(float16_t(v), float16_t(-65504.0), float16_t(65504.0))); }
#endif
