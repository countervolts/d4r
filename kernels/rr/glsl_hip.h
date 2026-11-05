// GLSL as HIP for kernels/rr/glsl2hip.py: the types, built-ins and KHR cooperative matrices (RDNA4 WMMA layout)
// that the engine's preset F stage shaders (engine/rr/rr2_w.comp, rr2_m.comp after glslangValidator -E) use, so
// that the same source compiles with LLVM into a vkhip code object. Arithmetic follows what RADV makes of the GLSL:
// 1.0 / x is v_rcp_f32 (build with -fno-hip-fp32-correctly-rounded-divide-sqrt), FP8 conversions round to nearest
// even, saturated ones clamp to +-448 and keep NaN.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#pragma clang fp contract(off)
using uint = uint32_t;
using float16_t = _Float16;
typedef float vec2 __attribute__((ext_vector_type(2)));
typedef float vec4 __attribute__((ext_vector_type(4)));
typedef _Float16 f16vec2 __attribute__((ext_vector_type(2)));
typedef _Float16 f16vec3 __attribute__((ext_vector_type(3)));
typedef _Float16 f16vec4 __attribute__((ext_vector_type(4)));
typedef uint uvec2 __attribute__((ext_vector_type(2)));
typedef uint uvec4 __attribute__((ext_vector_type(4)));
typedef int ivec2 __attribute__((ext_vector_type(2)));
#define GD __device__ __forceinline__
// everything in namespace glsl, so that GLSL's overloads hide HIP's math wrappers (glsl2hip.py wraps the shader too)
namespace glsl {
struct GlslArgs { uint64_t param, weights, arena; };
#ifdef GLSL_IMAGES
// A stage that writes storage images (rr2_hybrid_prep.py): their descriptors are descriptor set 0's table, which vkhip
// (Kernel::table) hands the kernel in the dispatch id's low dword (s2). The high dword is the program counter's, as
// LLPC forms it (s_getpc_b64; s_sext_i32_i16 of the high half).
__device__ uint64_t glslDispatchId() __asm("llvm.amdgcn.dispatch.id");
GD uint64_t glslImageTable() {
    const uint32_t hi = uint32_t(int32_t(int16_t(uint16_t(__builtin_amdgcn_s_getpc() >> 32))));
    return (uint64_t(hi) << 32) | uint32_t(glslDispatchId());
}
#endif
template <class T> GD T* glslGlobal(uint64_t a) { return (T*)(__attribute__((address_space(1))) T*)a; }

// ---------------------------------------------------------------- scalars
GD uint floatBitsToUint(float v) { return __builtin_bit_cast(uint, v); }
GD float uintBitsToFloat(uint v) { return __builtin_bit_cast(float, v); }
// a word this thread wrote earlier (glsl2hip.py --record): an atomic load, which LLVM neither forwards the stored value
// to nor hoists, so the value stays out of the registers in between
GD uint glslRecordLoad(const uint* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); }
GD _Float16 uint16BitsToFloat16(uint16_t b) { return __builtin_bit_cast(_Float16, b); }
GD uint16_t float16BitsToUint16(_Float16 h) { return __builtin_bit_cast(uint16_t, h); }
GD uint packFloat2x16(f16vec2 v) { return __builtin_bit_cast(uint, v); }
GD f16vec2 unpackFloat2x16(uint v) { return __builtin_bit_cast(f16vec2, v); }
GD bool isnan(float v) { return __builtin_isnan(v); }
GD bool isinf(float v) { return __builtin_isinf(v); }
GD bool isnan(_Float16 v) { return __builtin_isnan(v); }
GD bool isinf(_Float16 v) { return __builtin_isinf(v); }
GD int abs(int v) { return v < 0 ? -v : v; }
GD float abs(float v) { return __builtin_fabsf(v); }
GD _Float16 abs(_Float16 v) { return __builtin_fabsf16(v); }
GD vec2 abs(vec2 v) { return __builtin_elementwise_abs(v); }
GD f16vec2 abs(f16vec2 v) { return __builtin_elementwise_abs(v); }
GD int min(int a, int b) { return a < b ? a : b; }
GD int max(int a, int b) { return a > b ? a : b; }
GD uint min(uint a, uint b) { return a < b ? a : b; }
GD uint max(uint a, uint b) { return a > b ? a : b; }
GD float min(float a, float b) { return __builtin_fminf(a, b); }
GD float max(float a, float b) { return __builtin_fmaxf(a, b); }
GD _Float16 min(_Float16 a, _Float16 b) { return __builtin_fminf16(a, b); }
GD _Float16 max(_Float16 a, _Float16 b) { return __builtin_fmaxf16(a, b); }
GD vec2 min(vec2 a, vec2 b) { return __builtin_elementwise_min(a, b); }
GD vec2 max(vec2 a, vec2 b) { return __builtin_elementwise_max(a, b); }
GD f16vec2 min(f16vec2 a, f16vec2 b) { return __builtin_elementwise_min(a, b); }
GD f16vec2 max(f16vec2 a, f16vec2 b) { return __builtin_elementwise_max(a, b); }
GD f16vec4 min(f16vec4 a, f16vec4 b) { return __builtin_elementwise_min(a, b); }
GD f16vec4 max(f16vec4 a, f16vec4 b) { return __builtin_elementwise_max(a, b); }
template <class T> GD T clamp(T x, T lo, T hi) { return min(max(x, lo), hi); }
GD float fma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
GD _Float16 fma(_Float16 a, _Float16 b, _Float16 c) { return __builtin_fmaf16(a, b, c); }
GD vec2 fma(vec2 a, vec2 b, vec2 c) { return __builtin_elementwise_fma(a, b, c); }
GD f16vec2 fma(f16vec2 a, f16vec2 b, f16vec2 c) { return __builtin_elementwise_fma(a, b, c); }
GD float exp2(float v) { return __builtin_amdgcn_exp2f(v); }
GD float inversesqrt(float v) { return __builtin_amdgcn_rsqf(v); }
GD float sqrt(float v) { return __builtin_sqrtf(v); }
GD float floor(float v) { return __builtin_floorf(v); }
GD float fract(float v) { return v - __builtin_floorf(v); }

// constructors (function-like macros: the type names stay usable in declarations)
template <class A> GD vec2 glslVec2(A a) { return vec2{float(a), float(a)}; }
template <class A, class B> GD vec2 glslVec2(A a, B b) { return vec2{float(a), float(b)}; }
GD vec2 glslVec2(vec2 a) { return a; }
GD vec2 glslVec2(f16vec2 a) { return vec2{float(a.x), float(a.y)}; }
template <class A> GD f16vec2 glslH2(A a) { return f16vec2{_Float16(a), _Float16(a)}; }
template <class A, class B> GD f16vec2 glslH2(A a, B b) { return f16vec2{_Float16(a), _Float16(b)}; }
GD f16vec2 glslH2(f16vec2 a) { return a; }
GD f16vec2 glslH2(vec2 a) { return f16vec2{_Float16(a.x), _Float16(a.y)}; }
template <class A, class B, class C> GD f16vec3 glslH3(A a, B b, C c) { return f16vec3{_Float16(a), _Float16(b), _Float16(c)}; }
template <class A> GD f16vec4 glslH4(A a) { return f16vec4{_Float16(a), _Float16(a), _Float16(a), _Float16(a)}; }
template <class A, class B, class C, class E> GD f16vec4 glslH4(A a, B b, C c, E d) { return f16vec4{_Float16(a), _Float16(b), _Float16(c), _Float16(d)}; }
GD f16vec4 glslH4(f16vec3 a, _Float16 b) { return f16vec4{a.x, a.y, a.z, b}; }
template <class A, class B> GD uvec2 glslU2(A a, B b) { return uvec2{uint(a), uint(b)}; }
template <class A> GD uvec2 glslU2(A a) { return uvec2{uint(a), uint(a)}; }

// ---------------------------------------------------------------- FP8 (e4m3)
// GLSL_FP8_SAT: MODE.FP16_OVFL is set (glslInit), so the conversion itself saturates (one instruction); it also
// makes binary16 arithmetic saturate instead of overflowing to infinity, which glsl2hip.py --sat leaves to the
// shaders it is used for (their binary16 values stay finite or are clamped)
#ifdef GLSL_FP8_SAT
GD float glslSat8(float v) { return v; }
GD void glslInit() { __builtin_amdgcn_s_setreg(1 | (23 << 6), 1); }
#else
GD float glslSat8(float v) { return __builtin_isnan(v) ? v : __builtin_fminf(__builtin_fmaxf(v, -448.f), 448.f); }
GD void glslInit() {}
#endif
struct floate4m3_t {
    uint8_t b;
    GD floate4m3_t() = default;
    GD floate4m3_t(float v) { b = uint8_t(__builtin_amdgcn_cvt_pk_fp8_f32(glslSat8(v), 0.f, 0, false)); }
    GD floate4m3_t(_Float16 v) : floate4m3_t(float(v)) {}
    GD operator float() const { return __builtin_amdgcn_cvt_f32_fp8(b, 0); }
    GD static floate4m3_t bits(uint v) { floate4m3_t r; r.b = uint8_t(v); return r; }
};
// A pair keeps where its bytes came from: the 16 bits at half `hi` of `w`, and for a conversion its two source values
// (src), so that glslSetPair can convert straight into an operand's half word and glslH2 converts both bytes with
// one packed instruction; the fields a use does not need are dead code.
struct fe4m3vec2 {
    vec2 src;
    uint w;
    bool hi;
    floate4m3_t x, y;
    GD fe4m3vec2() = default;
    GD fe4m3vec2(floate4m3_t a, floate4m3_t b) : src{float(a), float(b)}, w(uint(a.b) | uint(b.b) << 8), hi(false), x(a), y(b) {}
    GD fe4m3vec2(uint word, bool high) : w(word), hi(high), x(floate4m3_t::bits(word >> (high ? 16 : 0))), y(floate4m3_t::bits(word >> (high ? 24 : 8))) {
        src = high ? __builtin_amdgcn_cvt_pk_f32_fp8(int(word), true) : __builtin_amdgcn_cvt_pk_f32_fp8(int(word), false);
    }
    GD explicit fe4m3vec2(vec2 v) : src{glslSat8(v.x), glslSat8(v.y)} {
        w = uint(__builtin_amdgcn_cvt_pk_fp8_f32(src.x, src.y, 0, false)); hi = false;
        x = floate4m3_t::bits(w); y = floate4m3_t::bits(w >> 8);
    }
    GD explicit fe4m3vec2(f16vec2 v) : fe4m3vec2(vec2{float(v.x), float(v.y)}) {}
};
struct glslU8Pair { uint w; bool hi; };
struct glslU8x4 { glslU8Pair xy, zw; };
GD glslU8x4 unpack8(uint w) { return {{w, false}, {w, true}}; }
GD floate4m3_t uintBitsToFloate4m3EXT(uint8_t v) { return floate4m3_t::bits(v); }
GD fe4m3vec2 uintBitsToFloate4m3EXT(glslU8Pair v) { return fe4m3vec2(v.w, v.hi); }
GD uint8_t floate4m3BitsToUintEXT(floate4m3_t v) { return v.b; }
// FP8 values are exact in binary16, so the round-towards-zero pack loses nothing
GD vec2 glslVec2(fe4m3vec2 v) { return v.hi ? __builtin_amdgcn_cvt_pk_f32_fp8(int(v.w), true) : __builtin_amdgcn_cvt_pk_f32_fp8(int(v.w), false); }
GD f16vec2 glslH2(fe4m3vec2 v) { const vec2 f = glslVec2(v); return __builtin_bit_cast(f16vec2, __builtin_amdgcn_cvt_pkrtz(f.x, f.y)); }
GD void saturatedConvertEXT(floate4m3_t& r, float v) { r = floate4m3_t(v); }
GD void saturatedConvertEXT(floate4m3_t& r, _Float16 v) { r = floate4m3_t(float(v)); }
GD void saturatedConvertEXT(fe4m3vec2& r, vec2 v) { r = fe4m3vec2(v); }
GD void saturatedConvertEXT(fe4m3vec2& r, f16vec2 v) { r = fe4m3vec2(v); }

// ---------------------------------------------------------------- invocation, subgroup, barriers
#define gl_SubgroupInvocationID (threadIdx.x & 31u)
// uniform per wave (wave32), which LLVM cannot see in threadIdx.x >> 5: in scalar registers, it and what derives from it
// leave the VGPRs alone (translated dec4_0 spilled lane-derived offsets without it)
#define gl_SubgroupID (__builtin_amdgcn_readfirstlane(threadIdx.x >> 5))
#define gl_LocalInvocationIndex (threadIdx.x)
#define gl_WorkGroupID (glslDim3{blockIdx.x, blockIdx.y, blockIdx.z})
#define gl_NumWorkGroups (glslDim3{gridDim.x, gridDim.y, gridDim.z})
struct glslDim3 { uint x, y, z; };
GD uint subgroupShuffleXor(uint v, uint mask) { return __shfl_xor(v, int(mask), 32); }
GD uint subgroupShuffle(uint v, uint lane) { return __shfl(v, int(lane), 32); }
GD void subgroupMemoryBarrierShared() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront", "local"); }
GD void memoryBarrierShared() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup", "local"); }
GD void memoryBarrierBuffer() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup", "global"); }
GD void barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// ---------------------------------------------------------------- cooperative matrices, 16 x 16, one wave of 32
// gfx12 WMMA: A (use 0) lane l holds row l & 15, k 8 (l >> 4) + i; B (use 1) column l & 15, k 8 (l >> 4) + i;
// accumulators row 8 (l >> 4) + i, column l & 15
enum : int { gl_ScopeSubgroup = 3, gl_MatrixUseA = 0, gl_MatrixUseB = 1, gl_MatrixUseAccumulator = 2,
             gl_CooperativeMatrixLayoutRowMajor = 0, gl_CooperativeMatrixLayoutColumnMajor = 1 };
template <class T, int S, int M, int N, int U> struct coopmat {
    T v[8];
    GD coopmat() = default;
    GD explicit coopmat(float x) { for (int i = 0; i < 8; ++i) v[i] = T(x); }
    GD T& operator[](int i) { return v[i]; }
    GD const T& operator[](int i) const { return v[i]; }
};
template <int S, int M, int N, int U> struct coopmat<floate4m3_t, S, M, N, U> {
    uint w[2] = {0, 0};
    struct Ref {
        uint* w; int i;
        GD operator floate4m3_t() const { return floate4m3_t::bits(w[i >> 2] >> (8 * (i & 3))); }
        GD operator float() const { return float(floate4m3_t(*this)); }
        GD Ref& operator=(floate4m3_t f) { const uint m = 255u << (8 * (i & 3)); w[i >> 2] = (w[i >> 2] & ~m) | (uint(f.b) << (8 * (i & 3))); return *this; }
    };
    GD coopmat() = default;
    GD Ref operator[](int i) { return {w, i}; }
    GD floate4m3_t operator[](int i) const { return floate4m3_t::bits(w[i >> 2] >> (8 * (i & 3))); }
};
// elements i and i + 1 (i even) of an FP8 operand from a pair (glsl2hip.py rewrites the pair stores into this):
// a converted pair goes straight into its half word
template <int S, int U> GD void glslSetPair(coopmat<floate4m3_t, S, 16, 16, U>& d, uint i, const fe4m3vec2& t) {
    uint& w = d.w[i >> 2];
    w = (i & 2) ? uint(__builtin_amdgcn_cvt_pk_fp8_f32(t.src.x, t.src.y, int(w), true))
                : uint(__builtin_amdgcn_cvt_pk_fp8_f32(t.src.x, t.src.y, int(w), false));
}
GD void glslSetPair(floate4m3_t* d, uint i, const fe4m3vec2& t) { d[i] = t.x; d[i + 1] = t.y; }
template <class T, int S, int U, class P>
GD void coopMatLoad(coopmat<T, S, 16, 16, U>& dst, const P* src, uint offset, uint stride, int layout) {
    const uint lane = threadIdx.x & 31u, col = lane & 15u, hl = lane >> 4;
    _Pragma("unroll") for (uint i = 0; i < 8; ++i) {
        const uint row = U == 0 ? col : 8 * hl + i, column = U == 0 ? 8 * hl + i : col;
        const char* p = (const char*)src + (offset + (layout == 0 ? row : column) * stride) * sizeof(P) + (layout == 0 ? column : row) * sizeof(T);
        dst[i] = *(const T*)p;
    }
}
template <class T, int S, int U, class P>
GD void coopMatStore(const coopmat<T, S, 16, 16, U>& src, P* dst, uint offset, uint stride, int layout) {
    const uint lane = threadIdx.x & 31u, col = lane & 15u, hl = lane >> 4;
    _Pragma("unroll") for (uint i = 0; i < 8; ++i) {
        const uint row = U == 0 ? col : 8 * hl + i, column = U == 0 ? 8 * hl + i : col;
        char* p = (char*)dst + (offset + (layout == 0 ? row : column) * stride) * sizeof(P) + (layout == 0 ? column : row) * sizeof(T);
        *(T*)p = src[i];
    }
}
// FP8 operands: eight consecutive bytes per lane when the layout runs along k, else eight byte accesses
template <int S, int U, class P>
GD void coopMatLoad(coopmat<floate4m3_t, S, 16, 16, U>& dst, const P* src, uint offset, uint stride, int layout) {
    const uint lane = threadIdx.x & 31u, col = lane & 15u, hl = lane >> 4;
    if ((U == 0 && layout == 0) || (U == 1 && layout == 1)) {
        const uint64_t v = *(const uint64_t*)((const char*)src + (offset + col * stride) * sizeof(P) + 8 * hl);
        dst.w[0] = uint(v); dst.w[1] = uint(v >> 32);
    } else {
        uint64_t bits = 0;
        _Pragma("unroll") for (uint i = 0; i < 8; ++i)
            bits |= uint64_t(*((const uint8_t*)src + (offset + (8 * hl + i) * stride) * sizeof(P) + col)) << (8 * i);
        dst.w[0] = uint(bits); dst.w[1] = uint(bits >> 32);
    }
}
#ifdef GLSL_B_WEIGHTS_TRANSPOSED
// B operands from the weights (FP8 elements): the loader transposed those tiles (rr2_v_layout), so a row-major tile's
// column is eight consecutive bytes per lane
template <int S>
GD void coopMatLoad(coopmat<floate4m3_t, S, 16, 16, 1>& dst, const floate4m3_t* src, uint offset, uint stride, int layout) {
    const uint lane = threadIdx.x & 31u, col = lane & 15u, hl = lane >> 4;
    if (layout != 0 || stride != 16u) __builtin_trap();
    const uint64_t v = *(const uint64_t*)((const char*)src + offset + col * 16u + 8 * hl);
    dst.w[0] = uint(v); dst.w[1] = uint(v >> 32);
}
#endif
template <int S, int U, class P>
GD void coopMatStore(const coopmat<floate4m3_t, S, 16, 16, U>& src, P* dst, uint offset, uint stride, int layout) {
    const uint lane = threadIdx.x & 31u, col = lane & 15u, hl = lane >> 4;
    if ((U == 0 && layout == 0) || (U == 1 && layout == 1)) {
        *(uint64_t*)((char*)dst + (offset + col * stride) * sizeof(P) + 8 * hl) = uint64_t(src.w[0]) | uint64_t(src.w[1]) << 32;
    } else {
        _Pragma("unroll") for (uint i = 0; i < 8; ++i)
            *((uint8_t*)dst + (offset + (8 * hl + i) * stride) * sizeof(P) + col) = uint8_t(src.w[i >> 2] >> (8 * (i & 3)));
    }
}
typedef int glslI2 __attribute__((ext_vector_type(2)));
typedef float glslF8 __attribute__((ext_vector_type(8)));
template <int S>
GD coopmat<float, S, 16, 16, 2> coopMatMulAdd(const coopmat<floate4m3_t, S, 16, 16, 0>& a, const coopmat<floate4m3_t, S, 16, 16, 1>& b,
                                              const coopmat<float, S, 16, 16, 2>& c) {
    const glslF8 r = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(glslI2{int(a.w[0]), int(a.w[1])}, glslI2{int(b.w[0]), int(b.w[1])},
                                                                          __builtin_bit_cast(glslF8, c.v));
    return __builtin_bit_cast(coopmat<float, S, 16, 16, 2>, r);
}
} // namespace glsl
