// Device helpers for the HIP forms of the native Ray Reconstruction stages (engine/rr/hip/*.hip), which run on RDNA3
// either through HIP (engine/rr/hip/rr_hip_test.cpp) or inside the engine's Vulkan command buffer (engine/vkhip).
//
// The arithmetic is engine/rr/rr_exact.glsl + rr_gfx11.glsl's (RR_GFX11), which reproduce ZLUDA's gfx11 lowering of
// NVIDIA's kernels bit for bit; only the instructions differ:
//   - FP8 operands are widened with v_perm (ZLUDA's e4m3x4_to_scaled_f16x2x2) and encoded two at a time with packed
//     16-bit integer ops (ZLUDA's f16x2_to_e4m3x2_satfinite_bits, equal to rr_gfx11.glsl's g11_h2e4m3 for all 65536
//     binary16 inputs);
//   - every global access goes through a buffer descriptor sized to the buffer, so an access outside it reads zero
//     and a store outside it is dropped (what the Vulkan stages get from robustBufferAccess), never a page fault.
// Register layouts are the GLSL stages' (rr_gfx11.glsl): an A operand's lane holds row g11_row(lane & 15), so an
// accumulator's element i in half h is row i + 8h, the gfx12 layout the stages' elementwise code is written for.
#pragma once
#include <stdint.h>

#pragma clang fp contract(off)

#define DEV __attribute__((device, always_inline)) static inline
// compiled with -nogpuinc: clang builtins only (the device libraries supply the __ockl image functions)
// RR_WAVES (build.sh per stage): at least that many waves per SIMD, i.e. a VGPR cap (occupancy for the many-window stages)
#ifdef RR_WAVES
#define KERNEL128 extern "C" __attribute__((global, amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(RR_WAVES)))
#else
#define KERNEL128 extern "C" __attribute__((global, amdgpu_flat_work_group_size(128, 128)))
#endif
#define KERNEL512 extern "C" __attribute__((global, amdgpu_flat_work_group_size(512, 512)))
#define TID() __builtin_amdgcn_workitem_id_x()
#define WGX() __builtin_amdgcn_workgroup_id_x()
#define WGY() __builtin_amdgcn_workgroup_id_y()
#define SHARED __attribute__((shared))
#define SYNC() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); \
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); } while (0)

typedef _Float16 f16;
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef float f4v __attribute__((ext_vector_type(4)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t u2v __attribute__((ext_vector_type(2)));
typedef uint16_t u16x2 __attribute__((ext_vector_type(2)));
typedef int16_t i16x2 __attribute__((ext_vector_type(2)));
typedef int32_t i2v __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(4))) unsigned int tsharp_t;

extern "C" __attribute__((device)) f4v __ockl_image_load_2D(tsharp_t* image, i2v coord);
extern "C" __attribute__((device)) f4v __ockl_image_sample_lod_2D(tsharp_t* image, tsharp_t* sampler,
                                                                 __attribute__((ext_vector_type(2))) float coord, float lod);
extern "C" __attribute__((device)) void __ockl_image_store_2D(tsharp_t* image, i2v coord, f4v value);

// ---------------------------------------------------------------- bounds-checked buffers
struct Buf {
    __amdgpu_buffer_rsrc_t r;
};
DEV Buf make_buf(const void* p, uint32_t bytes)
{
    return Buf{__builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), (short)0, (int)bytes, 0x31004000)};
}
DEV uint32_t ld32(const Buf& b, uint32_t byteOffset) { return __builtin_amdgcn_raw_buffer_load_b32(b.r, byteOffset, 0, 0); }
DEV u2v ld64(const Buf& b, uint32_t byteOffset) { return __builtin_amdgcn_raw_buffer_load_b64(b.r, byteOffset, 0, 0); }
DEV u4v ld128(const Buf& b, uint32_t byteOffset) { return __builtin_amdgcn_raw_buffer_load_b128(b.r, byteOffset, 0, 0); }
DEV uint32_t ld16(const Buf& b, uint32_t byteOffset) { return __builtin_amdgcn_raw_buffer_load_b16(b.r, byteOffset, 0, 0); }
DEV uint32_t ld8(const Buf& b, uint32_t byteOffset) { return __builtin_amdgcn_raw_buffer_load_b8(b.r, byteOffset, 0, 0); }
// a store offset, or one past every buffer when !c: the bounds-checked store is dropped. Stores of values computed
// by WMMA must not sit in a lane-divergent branch: the compiler may sink the products into it, and WMMA needs every
// lane (2026-11-07: rr_swin_wide.hip's expansion rows came out as garbage that way)
DEV uint32_t keep(bool c, uint32_t byteOffset) { return c ? byteOffset : 0xfffffff0u; }
DEV void st32(const Buf& b, uint32_t byteOffset, uint32_t v) { __builtin_amdgcn_raw_buffer_store_b32(v, b.r, byteOffset, 0, 0); }
DEV void st64(const Buf& b, uint32_t byteOffset, u2v v) { __builtin_amdgcn_raw_buffer_store_b64(v, b.r, byteOffset, 0, 0); }

// Brings workgroup `wg` of `wgs`'s slice of a buffer into the caches with one burst of independent loads per thread (at
// most 16 x 16 bytes; a slice beyond that is left to demand misses). The stages' chains are latency-bound where a level
// has few windows, and between the engine's stages the weights have left L2. The result only feeds a store that never
// happens (keeps the loads), so it changes nothing else.
DEV void warm(const Buf& b, uint32_t bytes, uint32_t wg, uint32_t wgs, uint32_t tid, uint32_t threads, const Buf& sink)
{
    const uint32_t slice = ((bytes + wgs - 1u) / wgs + 15u) & ~15u, lo = wg * slice;
    const uint32_t hi = lo + slice < bytes ? lo + slice : bytes;
    u4v v[16];
#pragma unroll
    for (uint32_t i = 0u; i < 16u; ++i) {
        const uint32_t o = lo + 16u * (tid + i * threads);
        v[i] = ld128(b, o < hi ? o : 0xfffffff0u);
    }
    uint32_t x = 0u;
#pragma unroll
    for (uint32_t i = 0u; i < 16u; ++i) x ^= v[i].x ^ v[i].w;
    if (x == 0x9e3779b9u && bytes == 0u) __builtin_amdgcn_raw_buffer_store_b32(x, sink.r, 0u, 0, 0);
}

// ---------------------------------------------------------------- scalars
DEV f16 hb(uint32_t b) { return __builtin_bit_cast(f16, (uint16_t)b); }
DEV uint32_t bh(f16 h) { return (uint32_t)__builtin_bit_cast(uint16_t, h); }
DEV uint32_t pk(hv2 v) { return __builtin_bit_cast(uint32_t, v); }
DEV hv2 upk(uint32_t u) { return __builtin_bit_cast(hv2, u); }
DEV float r16(float x) { return (float)(f16)x; }
DEV f16 h16(float x) { return (f16)x; }
// two f32 values that are binary16 numbers as an f16 pair (round toward zero is exact on them)
DEV uint32_t pk2(float a, float b) { return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(a, b)); }

DEV uint32_t inter(uint32_t j) { return (j & ~31u) | (((j & 31u) >> 2u) << 1u) | (j & 1u) | (((j >> 1u) & 1u) << 4u); }
DEV int mirror(int v, int n) { v = v < 0 ? -v : v; const int m = 2 * n - 2 - v; return v < m ? v : m; }
DEV int clampTap(int v, int n) { return v < 0 ? 1 : (v >= n ? n - 2 : v); }
// the A operand row a lane holds: physical row r carries row (r >> 1) + 8 (r & 1)
DEV uint32_t g11_row(uint32_t r) { return (r >> 1u) | ((r & 1u) << 3u); }

// value of the lane with the same index in the other half of the wave
DEV uint32_t other_half(uint32_t v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false); }

// m * x compared with 1, exactly: -1, 0 or 1
DEV int cmp1(float m, float x)
{
    const float t = m * x;
    const float e = __builtin_fmaf(m, x, -t);
    return t > 1.0f ? 1 : (t < 1.0f ? -1 : (e > 0.0f ? 1 : (e < 0.0f ? -1 : 0)));
}
// the GPU's reciprocal instruction (the reference's rcp.approx), rounded to binary16
#ifdef RR_HIP_IEEE_RCP
DEV f16 rcp16(float x) { return h16(1.0f / x); }
#else
DEV f16 rcp16(float x) { return h16(__builtin_amdgcn_rcpf(x)); }
#endif
#ifdef RR_HIP_OCML_EXP2
DEV float ex2(float x) { return __builtin_exp2f(x); }
#else
DEV float ex2(float x) { return __builtin_amdgcn_exp2f(x); }
#endif
// binary16 rounding of 1 / sqrt(s) (s > 0)
DEV f16 rsq16(f16 s16)
{
    const float s = (float)s16;
    f16 q = h16(__builtin_amdgcn_rsqf(s));
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t b = bh(q);
        const float lo = ((float)q + (float)hb(b - 1u)) * 0.5f, hi = ((float)q + (float)hb(b + 1u)) * 0.5f;
        const int cl = cmp1(lo * lo, s), ch = cmp1(hi * hi, s);
        if (cl > 0 || (cl == 0 && (b & 1u) != 0u)) q = hb(b - 1u);
        else if (ch < 0 || (ch == 0 && (b & 1u) != 0u)) q = hb(b + 1u);
    }
    return q;
}
// fma of binary16 operands with one binary16 rounding, exactly (rr_exact.glsl fma16)
DEV f16 fma16(f16 a, f16 b, f16 c)
{
    const float fa = (float)a, fb = (float)b, fc = (float)c;
    const float s = __builtin_fmaf(fa, fb, fc);
    const uint32_t u = __builtin_bit_cast(uint32_t, s);
    if ((u & 0x1fffu) == 0x1000u && __builtin_fabsf(s) >= 6.103515625e-05f) {
        const float p = fa * fb;
        const float bb = s - p;
        const float err = (p - (s - bb)) + (fc - bb);
        if (err != 0.0f) return h16(__builtin_bit_cast(float, err * s > 0.0f ? u + 1u : u - 1u));
    }
    return h16(s);
}
DEV f16 fma16q(f16 a, f16 b, f16 c, bool& mid)
{
    const float s = __builtin_fmaf((float)a, (float)b, (float)c);
    mid = mid || (__builtin_bit_cast(uint32_t, s) & 0x1fffu) == 0x1000u;
    return h16(s);
}
// f16(p + a) with ONE rounding (the exact sum to binary16): v_fma_mixlo_f16 1.0, p, a. The Vulkan stages' compiler
// forms this from their r16(p + a) wherever p is an f32 product of binary16 values (exact) and a a binary16
// accumulator, and the reference is that compile, so the HIP stages use the same instruction there.
DEV f16 mixadd(float p, f16 a)
{
    uint32_t r;
    asm("v_fma_mixlo_f16 %0, 1.0, %1, %2 op_sel_hi:[0,0,1]" : "=v"(r) : "v"(p), "v"(bh(a)));
    return hb(r & 0xffffu);
}
// f16(a * b) with one rounding and a -0 addend (v_fma_mixlo_f16 a, b, neg(0)): what the Vulkan stages' compiler emits
// for r16(a * b). LLVM folds (f16)(a * b) into v_fma_mixlo_f16 with a +0 addend, which turns an exact -0 product into
// +0 (FP8 code 0x00 for the reference's 0x80): every product rounded to binary16 goes through this.
DEV f16 mulh(float a, float b)
{
    uint32_t r;
    asm("v_fma_mixlo_f16 %0, %1, %2, neg(0)" : "=v"(r) : "v"(a), "v"(b));
    return hb(r & 0xffffu);
}
DEV f16 sig(f16 v) { return rcp16(ex2((float)v * -1.4426950216293335f) + 1.0f); }

// ---------------------------------------------------------------- FP8 (e4m3 codes)
// RNE satfinite e4m3 of both binary16 halves (NaN -> 0x7f with the sign), low half -> low byte
DEV uint32_t e4m3x2(uint32_t bits)
{
    u16x2 half = __builtin_bit_cast(u16x2, bits);
    u16x2 magnitude = half & (uint16_t)0x7fffu;
    u16x2 normal = (magnitude + ((magnitude >> (uint16_t)7) & (uint16_t)1) + (uint16_t)0xe03fu) >> (uint16_t)7;
    // e4m3 subnormals: f16 RNE of |x| + 2 lands on a multiple of 2^-9, whose ulp count above 2.0 is the code (equal to
    // the shift-and-round form for all 65536 inputs)
    const hv2 sm = __builtin_bit_cast(hv2, magnitude) + (hv2){(f16)2.0f, (f16)2.0f};
    u16x2 subnormal = __builtin_bit_cast(u16x2, sm) - (uint16_t)0x4000u;
    u16x2 below = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)(magnitude - (uint16_t)0x2400u)) >> (int16_t)15));
    u16x2 code = (subnormal & below) | (normal & ~below);
    code = __builtin_elementwise_min(code, (u16x2)0x7e);
    u16x2 nan = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)((uint16_t)0x7c00u - magnitude)) >> (int16_t)15));
    code |= nan & (uint16_t)0x7fu;
    code |= (half >> (uint16_t)8) & (uint16_t)0x80u;
    return uint32_t(code.x) | (uint32_t(code.y) << 8);
}
// the code of one binary16 value
DEV uint32_t q8h(f16 x) { return e4m3x2(bh(x)) & 0xffu; }
// four codes of f32 values that are binary16 numbers, bytes in order
DEV uint32_t codes4(float a, float b, float c, float d) { return e4m3x2(pk2(a, b)) | (e4m3x2(pk2(c, d)) << 16); }
// an accumulator's codes: element i as byte i & 3 of word i >> 2
DEV u2v codes8(const f8v& x) { return (u2v){codes4(x[0], x[1], x[2], x[3]), codes4(x[4], x[5], x[6], x[7])}; }

// e4m3 bytes -> f16 pairs holding value * 2^-8 (the magnitude at bits 7..14, NaN with bit 14 set), {bytes 0,1}, {2,3}
DEV void e4m3x4_scaled(uint32_t packed, uint32_t& low, uint32_t& high)
{
    uint32_t magnitude = packed & 0x7f7f7f7fu;
    magnitude |= (magnitude + 0x01010101u) & 0x80808080u;
    const uint32_t sign = packed & 0x80808080u;
    low = (__builtin_amdgcn_perm(0u, magnitude, 0x0c010c00u) << 7) | __builtin_amdgcn_perm(0u, sign, 0x010c000cu);
    high = (__builtin_amdgcn_perm(0u, magnitude, 0x0c030c02u) << 7) | __builtin_amdgcn_perm(0u, sign, 0x030c020cu);
}
// the exact binary16 values of four codes, as pairs {bytes 0,1}, {2,3} (NaN payloads aside)
DEV void e4h4(uint32_t packed, hv2& low, hv2& high)
{
    uint32_t lo, hi;
    e4m3x4_scaled(packed, lo, hi);
    low = upk(lo) * (hv2){(f16)256.0f, (f16)256.0f};
    high = upk(hi) * (hv2){(f16)256.0f, (f16)256.0f};
}

// ---------------------------------------------------------------- WMMA operands
// a k16 operand from 16 codes in k order (four words): ZLUDA's gfx11 positions (register j holds k 4j, 4j + 1,
// register 4 + j holds k 4j + 2, 4j + 3)
DEV u8v operand16(u4v d)
{
    u8v v;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lo, hi;
        e4m3x4_scaled(d[j], lo, hi);
        v[j] = lo;
        v[4 + j] = hi;
    }
    return v;
}
DEV f8v wmma(const u8v& a, const u8v& b, f8v c)
{
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16v, a), __builtin_bit_cast(h16v, b), c);
}
// one 32-deep step: C * 2^-16, the two halves, * 2^16, one rounding to binary16 (rr_gfx11_k32.glsl K32)
DEV void k32(f8v& acc, const u8v& a0, const u8v& b0, const u8v& a1, const u8v& b1)
{
    f8v c = acc * 1.52587890625e-05f;
    c = wmma(a0, b0, c);
    c = wmma(a1, b1, c);
    // f16(c * 2^16) per element (exact product, one rounding). LLVM's v_fma_mixlo/hi_f16 has a +0 addend, which would
    // turn c = -0 into +0, but c is a WMMA result and WMMA never returns -0 on RDNA3 (2026-11-07: 1.3e9 fuzzed outputs,
    // 5e8 of them zero, including -0 accumulators with all -0 products). Bit casts of the WMMA result's elements
    // miscompile with LLVM 22 (v_perm / and masks read element 0 for element 1): do not add any here.
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (float)(f16)(c[i] * 65536.0f);
}
// the B operand that is the accumulator-layout codes c (rr_gfx11.glsl G11_B)
DEV u8v g11_b(u2v c, bool h)
{
    const u2v o = (u2v){other_half(c.x), other_half(c.y)};
    const u2v e0 = h ? o : c, e1 = h ? c : o;
    return operand16((u4v){e0.x, e0.y, e1.x, e1.y});
}
// a widened weight operand: A (rows permuted) from the f16 copy at element W_WIDE (two 16-byte loads, k in order ->
// ZLUDA's positions by register choice alone)
DEV u8v lda_at(const Buf& w, uint32_t element)
{
    const u4v u0 = ld128(w, 2u * element), u1 = ld128(w, 2u * element + 16u);
    return (u8v){u0.x, u0.z, u1.x, u1.z, u0.y, u0.w, u1.y, u1.w};
}
// a B operand from a row-major 16 x 16 f16 tile at element `element`: this lane's column, k in order
DEV u8v ldb_at(const Buf& w, uint32_t element, uint32_t col)
{
    uint32_t s[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = ld16(w, 2u * (element + 16u * (uint32_t)i + col));
    u8v v;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        v[q] = s[4 * q] | (s[4 * q + 1] << 16);
        v[4 + q] = s[4 * q + 2] | (s[4 * q + 3] << 16);
    }
    return v;
}

// ---------------------------------------------------------------- stage helpers (the GLSL stages' LDS8A / LDS8B / G11_ST)
// FP8 operands from shared words: an A row-major (rows permuted), a B column-major
DEV u8v lds8a(const uint32_t* s, uint32_t off, uint32_t stride, uint32_t col)
{
    const uint32_t* p = s + off + g11_row(col) * stride;
    return operand16((u4v){p[0], p[1], p[2], p[3]});
}
DEV u8v lds8b(const uint32_t* s, uint32_t off, uint32_t stride, uint32_t col)
{
    const uint32_t* p = s + off + col * stride;
    return operand16((u4v){p[0], p[1], p[2], p[3]});
}
// an accumulator's codes stored column-major (rr_gfx11.glsl G11_ST)
DEV void st8(uint32_t* s, u2v c, uint32_t off, uint32_t stride, uint32_t lane)
{
    uint32_t* p = s + off + (lane & 15u) * stride + (lane >= 16u ? 2u : 0u);
    p[0] = c.x;
    p[1] = c.y;
}
// eight binary16 values at a byte offset (16-byte aligned), as f32
DEV f8v ldh8(const Buf& w, uint32_t byteOffset)
{
    const u4v u = ld128(w, byteOffset);
    f8v r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const hv2 v = upk(u[j]);
        r[2 * j] = (float)v.x;
        r[2 * j + 1] = (float)v.y;
    }
    return r;
}
DEV f16 w16(const Buf& w, uint32_t element) { return hb(ld16(w, 2u * element)); }

DEV i2v keyXY(uint32_t t, int x0, int y0)
{
    if (t < 64u) return (i2v){x0 + (int)(t & 7u), y0 + (int)(t >> 3u)};
    const int i = (int)(t & 7u), side = (int)((t - 64u) >> 3u);
    return (i2v){side < 2 ? x0 + i : (side == 2 ? x0 - 1 : x0 + 8), side == 0 ? y0 - 1 : (side == 1 ? y0 + 8 : y0 + i)};
}
DEV uint32_t mergeR(uint32_t r, bool y) { return y ? (r < 8u ? (r & 1u) : 2u + (r & 1u)) : (r < 8u ? (r >> 1u) : ((r - 8u) >> 1u)); }

// Widened shared operands: rows of f16 pairs holding value * 2^-8 in natural k order (16 values = 8 words per k16
// chunk), written once from codes (stw8: an accumulator's codes as G11_ST stores them, widened) and read with no
// conversion (ldw8a / ldw8b: the register choice gives ZLUDA's positions, as lda_at)
DEV u8v ldw8a(const uint32_t* s, uint32_t off, uint32_t stride, uint32_t col)
{
    const uint32_t* p = s + off + g11_row(col) * stride;
    const u4v u0 = (u4v){p[0], p[1], p[2], p[3]}, u1 = (u4v){p[4], p[5], p[6], p[7]};
    return (u8v){u0.x, u0.z, u1.x, u1.z, u0.y, u0.w, u1.y, u1.w};
}
DEV u8v ldw8b(const uint32_t* s, uint32_t off, uint32_t stride, uint32_t col)
{
    const uint32_t* p = s + off + col * stride;
    const u4v u0 = (u4v){p[0], p[1], p[2], p[3]}, u1 = (u4v){p[4], p[5], p[6], p[7]};
    return (u8v){u0.x, u0.z, u1.x, u1.z, u0.y, u0.w, u1.y, u1.w};
}
DEV void stw8(uint32_t* s, u2v c, uint32_t off, uint32_t stride, uint32_t lane)
{
    uint32_t l0, h0, l1, h1;
    e4m3x4_scaled(c.x, l0, h0);
    e4m3x4_scaled(c.y, l1, h1);
    uint32_t* p = s + off + (lane & 15u) * stride + (lane >= 16u ? 4u : 0u);
    p[0] = l0; p[1] = h0; p[2] = l1; p[3] = h1;
}
// one code word widened to two words at p
DEV void stw4(uint32_t* p, uint32_t w)
{
    uint32_t lo, hi;
    e4m3x4_scaled(w, lo, hi);
    p[0] = lo; p[1] = hi;
}

// ---------------------------------------------------------------- operands straight from binary16 values
// The widened e4m3 of two binary16 values (bits): e4m3x4_scaled(e4m3x2(h)) without the codes (equal for all 65536
// inputs: RNE to 3 mantissa bits and the bias change in the bits for e4m3 normals, saturated at 448; e4m3
// subnormals by an f16 RNE to a multiple of 2^-9 (|x| + 2) and an exact fma to value * 2^-8; NaN, sign)
DEV uint32_t wid2(uint32_t bits)
{
    const u16x2 hv = __builtin_bit_cast(u16x2, bits);
    const u16x2 m = hv & (uint16_t)0x7fffu;
    const u16x2 rn = (m + ((m >> (uint16_t)7) & (uint16_t)1) + (uint16_t)0x3fu) & (uint16_t)0xff80u;
    const u16x2 wn = __builtin_elementwise_min(rn, (u16x2)0x5f00) - (uint16_t)0x2000u;
    const hv2 sm = __builtin_bit_cast(hv2, m) + (hv2){(f16)2.0f, (f16)2.0f};
    const hv2 ws = __builtin_elementwise_fma(sm, (hv2){(f16)0.00390625f, (f16)0.00390625f}, (hv2){(f16)-0.0078125f, (f16)-0.0078125f});
    const u16x2 below = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)(m - (uint16_t)0x2400u)) >> (int16_t)15));
    u16x2 w = (__builtin_bit_cast(u16x2, ws) & below) | (wn & ~below);
    const u16x2 nan = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)((uint16_t)0x7c00u - m)) >> (int16_t)15));
    w |= nan & (uint16_t)0x7f80u;
    w |= hv & (uint16_t)0x8000u;
    return __builtin_bit_cast(uint32_t, w);
}
// an accumulator's eight binary16 values (f32 holding binary16), widened: pairs (0,1) (2,3) (4,5) (6,7)
DEV u4v wid8(const f8v& x)
{
    return (u4v){wid2(pk2(x[0], x[1])), wid2(pk2(x[2], x[3])), wid2(pk2(x[4], x[5])), wid2(pk2(x[6], x[7]))};
}
// the B operand from an accumulator layout's widened values (g11_b without the codes): rows 0-7 are half-wave 0's
DEV u8v g11_bw(u4v w, bool h)
{
    const u4v o = (u4v){other_half(w.x), other_half(w.y), other_half(w.z), other_half(w.w)};
    const u4v a = h ? o : w, b = h ? w : o;
    return (u8v){a.x, a.z, b.x, b.z, a.y, a.w, b.y, b.w};
}
