// What preset F's hybrid stages enc0 and dec0_1 (kernels/rr/rr2_hybrid_prep.py) use besides kernels/rr/glsl_hip.h:
// included after it (glsl2hip.py --include=glsl_hip_hybrid.h).
#pragma once
typedef uint16_t u16vec2 __attribute__((ext_vector_type(2)));
#define GLSL_TABLE __device__ const
namespace glsl {
// storage images: AMD's eight-dword descriptors, read from descriptor set 0's table (glsl_hip.h glslImageTable; vkhip
// puts binding k at 32 k bytes). The hardware drops a store outside the image, which stands in for the shader's own
// bounds test (an image size of 2^30)
typedef int glslV8i __attribute__((ext_vector_type(8)));
struct glslImage { glslV8i d; };
__device__ void glslImageStore2d(vec4 v, int dmask, int x, int y, glslV8i rsrc, int tfe, int aux)
    __asm("llvm.amdgcn.image.store.2d.v4f32.i32");
GD ivec2 glslImageSize(const glslImage&) { return ivec2{1 << 30, 1 << 30}; }
GD void glslImageStore(const glslImage& image, ivec2 c, vec4 v) { glslImageStore2d(v, 15, c.x, c.y, image.d, 0, 0); }
GD f16vec2 uint16BitsToFloat16(u16vec2 v) { return __builtin_bit_cast(f16vec2, v); }
GD u16vec2 float16BitsToUint16(f16vec2 v) { return __builtin_bit_cast(u16vec2, v); }
template <class A, class B> GD u16vec2 glslU16x2(A a, B b) { return u16vec2{uint16_t(a), uint16_t(b)}; }
// GLSL.std.450 NMin / NMax (the shader's spirv_instruction declarations): the operand that is not NaN
GD f16vec2 pmin2(f16vec2 a, f16vec2 b) { return __builtin_elementwise_min(a, b); }
GD f16vec2 pmax2(f16vec2 a, f16vec2 b) { return __builtin_elementwise_max(a, b); }
GD _Float16 pmin(_Float16 a, _Float16 b) { return __builtin_fminf16(a, b); }
GD _Float16 pmax(_Float16 a, _Float16 b) { return __builtin_fmaxf16(a, b); }
GD float roundEven(float v) { return __builtin_rintf(v); }
// dec0_1's image values (q_unorm8 / q_snorm8): vec4 GLSL
struct glslB4 { bool x, y, z, w; };
GD glslB4 isnan(vec4 v) { return {bool(__builtin_isnan(v.x)), bool(__builtin_isnan(v.y)), bool(__builtin_isnan(v.z)), bool(__builtin_isnan(v.w))}; }
GD vec4 mix(vec4 a, vec4 b, glslB4 m) { return vec4{m.x ? b.x : a.x, m.y ? b.y : a.y, m.z ? b.z : a.z, m.w ? b.w : a.w}; }
GD vec4 clamp(vec4 v, vec4 lo, vec4 hi) { return __builtin_elementwise_min(__builtin_elementwise_max(v, lo), hi); }
GD vec4 abs(vec4 v) { return __builtin_elementwise_abs(v); }
GD vec4 floor(vec4 v) { return __builtin_elementwise_floor(v); }
GD float glslSign(float v) { return v > 0.f ? 1.f : v < 0.f ? -1.f : v; }
GD vec4 sign(vec4 v) { return vec4{glslSign(v.x), glslSign(v.y), glslSign(v.z), glslSign(v.w)}; }
GD ivec2 glslIV2(int a, int b) { return ivec2{a, b}; }
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
