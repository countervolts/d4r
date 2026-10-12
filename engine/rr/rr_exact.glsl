// Exact NVIDIA arithmetic helpers for the native Ray Reconstruction stages (engine/rr/*_m.comp, enc1_native.comp).
// Requires GL_EXT_shader_explicit_arithmetic_types, GL_EXT_float_e4m3.
//
// RADV folds a register-only float(float16_t(x)) back to x, so every binary16 rounding whose result is used again goes
// through a conversion with an opaque zero (RR_ZERO16, set in main from something the compiler cannot evaluate, e.g.
// gl_NumWorkGroups.z - 1) OR-ed into its bits: one instruction more than the plain conversion.
uint16_t RR_ZERO16;
#if defined(RR_RELAXED)
// Relaxed rounding (build_native_model.sh with D4R_RR_RELAXED=1): the accumulators stay f32 between 32-deep steps and
// are converted to FP8 in one rounding instead of NVIDIA's two (binary16, then FP8). About 1% of the FP8 codes of a
// layer then differ by one step, so a stage no longer reproduces the reference bit for bit; it saves the conversions
// that are a quarter of a stage's instructions.
float rnd16(float x) { return x; }
#elif defined(RR_EXACT_CONVERT)
// the module's conversions are decorated NoContraction (engine/rr/spv_exact_convert.py): the driver keeps them
float rnd16(float x) { return float(float16_t(x)); }
#else
float rnd16(float x) { return float(uint16BitsToFloat16(float16BitsToUint16(float16_t(x)) | RR_ZERO16)); }
#endif
// x rounded to binary16 (nearest even), as f32 / as f16, through a conversion the compiler keeps (see rnd16)
float r16(float x) { return rnd16(x); }
#if (defined(RR_RELAXED) && RR_RELAXED >= 2) || defined(RR_EXACT_CONVERT)
float16_t h16(float x) { return float16_t(x); }
#else
float16_t h16(float x) { return uint16BitsToFloat16(float16BitsToUint16(float16_t(x)) | RR_ZERO16); }
#endif
// NVIDIA's k order within a 32-block: slot j holds natural channel inter(j); uninter is the inverse (and also the
// arena's channel order: arena slot uninter(k) holds natural channel k)
uint inter(uint j) { return (j & ~31u) | (((j & 31u) >> 2u) << 1u) | (j & 1u) | (((j >> 1u) & 1u) << 4u); }
uint uninter(uint c) { return (c & ~31u) | (((c & 15u) >> 1u) << 2u) | (c & 1u) | (((c >> 4u) & 1u) << 1u); }
int mirror(int v, int n) { v = abs(v); return min(v, 2 * n - 2 - v); }

// cvt.rn.satfinite.e4m3 (NaN -> 0x7f)
floate4m3_t q8(float16_t h)
{
    return isnan(h) ? uintBitsToFloate4m3EXT(uint8_t(0x7fu)) : floate4m3_t(clamp(h, float16_t(-448.0), float16_t(448.0)));
}
// the same from f32 / f16 values in one saturating hardware conversion (satfinite; for an f32 value that is a binary16
// number the result equals the f16 conversion's)
floate4m3_t q8s(float x) { floate4m3_t r; saturatedConvertEXT(r, x); return r; }
floate4m3_t q8h(float16_t x) { floate4m3_t r; saturatedConvertEXT(r, x); return r; }

uint code(floate4m3_t v) { return uint(floate4m3BitsToUintEXT(v)); }
float16_t hb(uint b) { return uint16BitsToFloat16(uint16_t(b)); }
uint bh(float16_t h) { return uint(float16BitsToUint16(h)); }

// m * x compared with 1, exactly (the fma recovers the product's rounding error): -1, 0 or 1
int cmp1(float m, float x)
{
    precise float t = m * x;
    precise float e = fma(m, x, -t);
    return t > 1.0 ? 1 : (t < 1.0 ? -1 : (e > 0.0 ? 1 : (e < 0.0 ? -1 : 0)));
}
// 1 / x of the filters' softmax and gate, rounded to binary16. The reference is the GPU's own reciprocal instruction
// (rcp.approx; ZLUDA lowers it to llvm.amdgcn.rcp.f32, which is what 1.0 / x compiles to here), not the correctly
// rounded quotient: correcting it left ~0.01% of the filtered surfaces one ulp off (capDT, 2026-10-12).
float16_t rcp16(float x) { return h16(1.0 / x); }
// binary16 rounding of 1 / sqrt(s) (s > 0)
float16_t rsq16(float16_t s16)
{
    const float s = float(s16);
    float16_t q = h16(inversesqrt(s));
#if defined(RR_RELAXED) && RR_RELAXED >= 2
    return q;
#endif
    for (int i = 0; i < 2; ++i) {
        const uint b = bh(q);
        const float lo = (float(q) + float(hb(b - 1u))) * 0.5, hi = (float(q) + float(hb(b + 1u))) * 0.5;
        const int cl = cmp1(lo * lo, s), ch = cmp1(hi * hi, s);
        if (cl > 0 || (cl == 0 && (b & 1u) != 0u)) q = hb(b - 1u);
        else if (ch < 0 || (ch == 0 && (b & 1u) != 0u)) q = hb(b + 1u);
    }
    return q;
}
// fma of binary16 operands with one binary16 rounding, exactly. The product is exact in f32, so an f32 fma rounds the
// exact result once; converting that to binary16 rounds twice, which differs only when the f32 value lands exactly
// on a binary16 midpoint: then the sum's rounding error (TwoSum, the rare branch) decides the direction.
float16_t fma16(float16_t a, float16_t b, float16_t c)
{
    const float fa = float(a), fb = float(b), fc = float(c);
    precise float s = fma(fa, fb, fc);
#if defined(RR_RELAXED) && RR_RELAXED >= 2
    return float16_t(s);
#endif
    const uint u = floatBitsToUint(s);
    if ((u & 0x1fffu) == 0x1000u && abs(s) >= 6.103515625e-05) {
        precise float p = fa * fb;
        precise float bb = s - p;
        precise float err = (p - (s - bb)) + (fc - bb);
        if (err != 0.0) return h16(uintBitsToFloat(err * s > 0.0 ? u + 1u : u - 1u));
    }
    return h16(s);
}
// the same in one f32 fma, rounded once more to binary16: mid is set when the f32 result lies on a binary16 midpoint,
// where that double rounding can differ (the caller then redoes the computation with fma16)
float16_t fma16q(float16_t a, float16_t b, float16_t c, inout bool mid)
{
    precise float s = fma(float(a), float(b), float(c));
#if defined(RR_RELAXED) && RR_RELAXED >= 2
    return float16_t(s);
#endif
    mid = mid || (floatBitsToUint(s) & 0x1fffu) == 0x1000u;
#ifdef RR_EXACT_CONVERT
    return float16_t(s);
#else
    return hb(bh(float16_t(s)) | uint(RR_ZERO16));
#endif
}
// The saturating FP8 conversion of a whole accumulator, two elements per conversion instruction
#define Q8PAIRS(dst, src) [[unroll]] for (int i_ = 0; i_ < 8; i_ += 2) { fe4m3vec2 t_; saturatedConvertEXT(t_, vec2((src)[i_], (src)[i_ + 1])); (dst)[i_] = t_.x; (dst)[i_ + 1] = t_.y; }
// q8(act(h)) from the packed table (enc1_pack.act_table; the weight buffer's W8U view at W_ACT)
#define ACTQ(h) uintBitsToFloate4m3EXT(w8u.w[W_ACT + bh(h)])
float16_t act(float16_t x)
{
    precise float16_t y = clamp(x, float16_t(-2.0), float16_t(2.0));
    precise float16_t a = float16_t(0.0810546875) * abs(y);
    precise float16_t b = float16_t(0.412109375) - a;
    precise float16_t c = y * b;
    precise float16_t g = float16_t(0.5) + c;
    precise float16_t r = x * g;
    return r;
}
float16_t sig(float16_t v) { return rcp16(exp2(float(v) * -1.4426950216293335) + 1.0); }
// the cubic-exponent softmax numerator (enc1_ref.softmax_cubic): a cubic in the clamped scaled score whose binary16
// bits, shifted into the exponent field, are the exponential
// the same in plain f32 fmas (double rounding); bad is set when an f32 result lies exactly on a binary16 midpoint,
// the only case where that differs: the caller then recomputes with cubic_e (subgroupAny, rare)
float16_t cubic_e_fast(float16_t x, inout bool bad)
{
    precise float16_t m = clamp(x * float16_t(0.017181396484375), float16_t(-0.55615234375), float16_t(0.55615234375));
    const float fm = float(m);
    precise float t = fma(fm, -fm, 0.92724609375);
    const float tr = rnd16(t);
    precise float u = fma(fm, tr, 1.375);
    // t - c and u - 1.375 are exact (Sterbenz), so one more fma gives the rounding error's sign: only an inexact
    // midpoint is a double rounding
    precise float et = fma(fm, fm, t - 0.92724609375), eu = fma(fm, -tr, u - 1.375);
    bad = bad || ((floatBitsToUint(t) & 0x1fffu) == 0x1000u && et != 0.0) || ((floatBitsToUint(u) & 0x1fffu) == 0x1000u && eu != 0.0);
    return hb((bh(float16_t(u)) << 5u) & 0x7FE0u);
}
float16_t cubic_e(float16_t x)
{
    precise float16_t m = clamp(x * float16_t(0.017181396484375), float16_t(-0.55615234375), float16_t(0.55615234375));
    const float16_t t = fma16(m, -m, float16_t(0.92724609375));
    const float16_t u = fma16(m, t, float16_t(1.375));
    return hb((bh(u) << 5u) & 0x7FE0u);
}
