// What preset F's hybrid encoder 0 (kernels/rr/enc0_hip_prep.py) uses besides kernels/rr/glsl_hip.h: included after
// it (glsl2hip.py --include=glsl_hip_enc0.h).
#pragma once
typedef uint16_t u16vec2 __attribute__((ext_vector_type(2)));
#define GLSL_TABLE __device__ const
namespace glsl {
GD f16vec2 uint16BitsToFloat16(u16vec2 v) { return __builtin_bit_cast(f16vec2, v); }
GD u16vec2 float16BitsToUint16(f16vec2 v) { return __builtin_bit_cast(u16vec2, v); }
template <class A, class B> GD u16vec2 glslU16x2(A a, B b) { return u16vec2{uint16_t(a), uint16_t(b)}; }
// GLSL.std.450 NMin / NMax (the shader's spirv_instruction declarations): the operand that is not NaN
GD f16vec2 pmin2(f16vec2 a, f16vec2 b) { return __builtin_elementwise_min(a, b); }
GD f16vec2 pmax2(f16vec2 a, f16vec2 b) { return __builtin_elementwise_max(a, b); }
GD _Float16 pmin(_Float16 a, _Float16 b) { return __builtin_fminf16(a, b); }
GD _Float16 pmax(_Float16 a, _Float16 b) { return __builtin_fmaxf16(a, b); }
GD float roundEven(float v) { return __builtin_rintf(v); }
GD void subgroupBarrier() { __builtin_amdgcn_wave_barrier(); }
GD uvec4 subgroupBallot(bool p) { return uvec4{uint(__builtin_amdgcn_ballot_w32(p)), 0u, 0u, 0u}; }
GD vec2 unpackHalf2x16(uint w) {
    const f16vec2 h = __builtin_bit_cast(f16vec2, w);
    return vec2{float(h.x), float(h.y)};
}
// the stage packs only values already rounded to binary16 (roundHalfValue), where every rounding mode agrees
GD uint packHalf2x16(vec2 v) { return __builtin_bit_cast(uint, f16vec2{_Float16(v.x), _Float16(v.y)}); }
GD vec4 glslV4(float a) { return vec4{a, a, a, a}; }
GD vec4 glslV4(float a, float b, float c, float d) { return vec4{a, b, c, d}; }
GD vec4 glslV4(vec2 a, vec2 b) { return vec4{a.x, a.y, b.x, b.y}; }
// coopmat<T, ...>(accumulator of another element type)
template <class T, class F, int S, int M, int N, int U> GD coopmat<T, S, M, N, U> glslAccCvt(const coopmat<F, S, M, N, U>& c) {
    coopmat<T, S, M, N, U> r;
    for (int i = 0; i < 8; ++i) r.v[i] = T(c.v[i]);
    return r;
}
template <int S, int U> GD float glslF8f(const coopmat<floate4m3_t, S, 16, 16, U>& c, int i) { return float(c[i]); }
}  // namespace glsl
#define u16vec2(a, b) glsl::glslU16x2(a, b)
