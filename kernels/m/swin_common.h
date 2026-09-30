// Shared device code for native rrlite Swin-block kernels (DLSS 310.7.0; gfx11 / gfx12 wave32 WMMA).
// Semantics: ../ENC3_SPEC.md generalised (numpy reference ../swin_model.py).
//
// Activations are the WMMA A operand (lane = token row), weights the B operand (lane = channel column), and
// the f32 result has lane = channel column, VGPR i = token row wm_acc_row(i) (../common/wmma_layout.h).
// D4R_FP8_WMMA (gfx12 layout only): the GEMMs whose operands are both e4m3 values run as native FP8 WMMAs on
// e4m3 bytes (weights prepared as bytes, activations encoded when loaded); P V stays f16.
#pragma once
#ifdef D4R_DEVICE_ONLY_MINIMAL
#include "../common/hip_device_minimal.h"
#else
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>
#include "../common/wmma_layout.h"
#if defined(D4R_FP8_WMMA) && D4R_WMMA_LAYOUT != 12 && defined(__AMDGCN__)
#error "D4R_FP8_WMMA needs the gfx12 WMMA layout"
#endif
// FP8 operands must hold e4m3 values, so the FP8 build re-quantises the intermediate activations as NVIDIA's
// network does (the gfx11 "fast numerics" SWIN_NO_Q8 keeps them f16, which no gfx12 WMMA can multiply with
// e4m3 weights)
// (SWIN_FORCE_Q8 does the same on any target: the gfx11 reference of the FP8 build's numerics)
#if (defined(D4R_FP8_WMMA) || defined(SWIN_FORCE_Q8) || defined(SWIN_EXACT)) && defined(SWIN_NO_Q8)
#undef SWIN_NO_Q8
#endif
// SWIN_EXACT (tests): NVIDIA's numerics throughout, the semantics of tools/swin_model.py (e4m3 activations and
// an f16 accumulator rounded after every k32 step)
#if defined(SWIN_EXACT) && defined(SWIN_F32ACC)
#undef SWIN_F32ACC
#endif

#pragma clang fp contract(off)

typedef _Float16 half_t;
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8v __attribute__((ext_vector_type(8)));

struct CommonParams
{
    const uint8_t* w; // 0
    int sx, sy;       // 8
    int tw, th;       // 16
    const uint8_t* in;  // 24
    const uint8_t* p32; // 32 (decoders: skip input)
    uint8_t* out;       // 40
    uint8_t* p48;       // 48 (encoders with patch merge: merged output)
};
static_assert(sizeof(CommonParams) == 56, "param block");

struct TubeParams
{
    CommonParams c;
    int r69, r70;
    int r71, r72;
    const uint8_t* in_flags;
    uint8_t* out_flags;
};
static_assert(sizeof(TubeParams) == 88, "param block");

// ---------------------------------------------------------------- numerics
__device__ __forceinline__ half_t e4m3_to_half(uint32_t code)
{
    uint16_t bits = (uint16_t)(((code & 0x7fu) << 7) | ((code & 0x80u) << 8));
    return __builtin_bit_cast(half_t, bits) * (half_t)256.0f;
}

__device__ __forceinline__ uint32_t half_to_e4m3(half_t h)
{
    uint32_t half = __builtin_bit_cast(uint16_t, h);
    uint32_t magnitude = half & 0x7fffu;
    uint32_t normal = (magnitude - 0x2000u + 0x3fu + ((magnitude >> 7) & 1u)) >> 7;
    uint32_t exponent = magnitude >> 10;
    uint32_t significand = (magnitude & 0x3ffu) | (exponent != 0 ? 0x400u : 0u);
    uint32_t clamped = exponent < 1u ? 1u : (exponent > 8u ? 8u : exponent);
    uint32_t shift = 16u - clamped;
    uint32_t subnormal = (significand + (1u << (shift - 1u)) - 1u + ((significand >> shift) & 1u)) >> shift;
    uint32_t code = magnitude >= 0x2400u ? normal : subnormal;
    code = magnitude > 0x5f00u ? 0x7eu : code;
    code = magnitude > 0x7c00u ? 0x7fu : code;
    return ((half >> 8) & 0x80u) | code;
}

// e4m3 RNE satfinite round trip evaluated in the f16 domain (exhaustively equal to decode(encode(h)))
__device__ __forceinline__ half_t q8_exact(half_t h);
__device__ __forceinline__ half_t q8(half_t h)
{
#ifdef SWIN_NO_Q8
    return h; // fast mode: intermediate activations stay f16 (stored outputs are still FP8)
#endif
    return q8_exact(h);
}
__device__ __forceinline__ half_t q8_exact(half_t h)
{
    const uint32_t u = __builtin_bit_cast(uint16_t, h);
    const uint32_t m = u & 0x7fffu;
    uint32_t r = (m + 0x3fu + ((m >> 7) & 1u)) & 0x7f80u;
    r = r < 0x5f00u ? r : 0x5f00u;
    const half_t mag = __builtin_bit_cast(half_t, (uint16_t)m);
    const uint32_t sub = __builtin_bit_cast(uint16_t, (half_t)((half_t)(mag + (half_t)2.0f) - (half_t)2.0f));
    r = m < 0x2400u ? sub : r;
    r = m > 0x7c00u ? m : r;
    return __builtin_bit_cast(half_t, (uint16_t)((u & 0x8000u) | r));
}

typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));
typedef short i16x2 __attribute__((ext_vector_type(2)));

// q8_exact on two halves with packed 16-bit integer / f16 ops (same per-half logic; no carries cross halves)
__device__ __forceinline__ hv2 q8x2_exact(hv2 h)
{
    const u16x2 u = __builtin_bit_cast(u16x2, h);
    const u16x2 m = u & (unsigned short)0x7fff;
    u16x2 r = (m + (unsigned short)0x3f + ((m >> 7) & (unsigned short)1)) & (unsigned short)0x7f80;
    r = __builtin_elementwise_min(r, (u16x2){0x5f00, 0x5f00});
    const hv2 mag = __builtin_bit_cast(hv2, m);
    const hv2 two = {(half_t)2.0f, (half_t)2.0f};
    const u16x2 sub = __builtin_bit_cast(u16x2, (hv2)((hv2)(mag + two) - two));
    const u16x2 lt = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)(m - (unsigned short)0x2400)) >> 15));
    r = (sub & lt) | (r & ~lt);
    const u16x2 gt = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)((unsigned short)0x7c00 - m)) >> 15));
    r = (m & gt) | (r & ~gt);
    return __builtin_bit_cast(hv2, (u16x2)(r | (u & (unsigned short)0x8000)));
}

__device__ __forceinline__ hv2 q8x2(hv2 h)
{
#ifdef SWIN_NO_Q8
    return h;
#endif
    return q8x2_exact(h);
}

// enc8 of two halves with packed ops: returns the two e4m3 codes in the low bytes of the 16-bit
// halves, and q = q8_exact(h) (the values the codes decode to)
__device__ __forceinline__ uint32_t codes8x2(hv2 q);
__device__ __forceinline__ uint32_t enc8x2(hv2 h, hv2& q)
{
    q = q8x2_exact(h);
    return codes8x2(q);
}

// the e4m3 codes of two halves that already hold e4m3 values (q8_exact results), in the low bytes of the
// 16-bit halves
__device__ __forceinline__ uint32_t codes8x2(hv2 q)
{
    const u16x2 qb = __builtin_bit_cast(u16x2, q);
    const u16x2 m = qb & (unsigned short)0x7fff;
    // normals: the exponent moves from bias 15 to bias 7
    const u16x2 norm = (u16x2)(m >> 7) - (unsigned short)64;
    // subnormals are k * 2^-9 (k = 0..7): f16(k + 1024) has k in its low bits
    const hv2 mag = __builtin_bit_cast(hv2, m);
    const hv2 k = (hv2)(mag * (hv2){(half_t)512.0f, (half_t)512.0f}) + (hv2){(half_t)1024.0f, (half_t)1024.0f};
    const u16x2 sub = __builtin_bit_cast(u16x2, k) - (unsigned short)0x6400;
    const u16x2 lt = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)(m - (unsigned short)0x2400)) >> 15));
    u16x2 code = (sub & lt) | (norm & ~lt);
    const u16x2 gt = __builtin_bit_cast(u16x2, (i16x2)(__builtin_bit_cast(i16x2, (u16x2)((unsigned short)0x7c00 - m)) >> 15));
    code = ((u16x2){0x7f, 0x7f} & gt) | (code & ~gt);
    return __builtin_bit_cast(uint32_t, (u16x2)(((qb >> 8) & (unsigned short)0x80) | code));
}

// four e4m3 codes (a, b from enc8x2) as the bytes of one word in element order
__device__ __forceinline__ uint32_t pack_codes(uint32_t a, uint32_t b)
{
    return __builtin_amdgcn_perm(b, a, 0x06040200u);
}

// e4m3 code of h (RNE, satfinite) derived from q8(h): normals shift the exponent, subnormals are k * 2^-9
__device__ __forceinline__ uint32_t enc8(half_t h)
{
    const uint32_t qb = __builtin_bit_cast(uint16_t, q8_exact(h));
    const uint32_t m = qb & 0x7fffu;
    const half_t mag = __builtin_bit_cast(half_t, (uint16_t)m);
    uint32_t code = m >= 0x2400u ? (m >> 7) - 64u : (uint32_t)(uint16_t)(mag * (half_t)512.0f);
    code = m > 0x7c00u ? 0x7fu : code;
    return ((qb >> 8) & 0x80u) | code;
}

__device__ __forceinline__ half_t f16(float v)
{
    return (half_t)v;
}

__device__ __forceinline__ half_t rsqrt16(half_t s)
{
    return f16(1.0f / __builtin_sqrtf((float)(s + (half_t)(1.0f / 8192.0f))));
}

// natural channel c -> pair-order column, and inverse (per 32-channel group)
__device__ __forceinline__ int gperm(int c)
{
    return 32 * (c >> 5) + 2 * ((c & 31) >> 2) + (c & 1) + 16 * ((c & 3) >> 1);
}
__device__ __forceinline__ int ginv(int n)
{
    int m = n & 31, e = m >> 4, a = (m & 15) >> 1, b = m & 1;
    return (n & ~31) + 4 * a + 2 * e + b;
}

// WMMA k16 step s of a k32 chunk: operand slot i holds logical k = kslot(s, i) (matches the weight
// memory runs). Activations live in LDS at apos(k), so an A operand is 16 contiguous halves.
__device__ __forceinline__ int kslot(int s, int i)
{
    int a = 2 * s + (i >> 3), r = i & 7;
    return (r < 4) ? (4 * a + r) : (16 + 4 * a + (r - 4));
}
__device__ __forceinline__ int apos(int k)
{
    return (k & ~31) + 8 * ((k & 15) >> 2) + 4 * ((k >> 4) & 1) + (k & 3);
}

__device__ __forceinline__ int woff(int base, int Ks, int Ns, int k, int n)
{
    int j = n >> 3;
    return base + Ks * (k >> 5) + Ns * (j >> 1) + 64 * (n & 7) + 16 * ((k & 15) >> 2) + 8 * (j & 1) +
           4 * ((k & 31) >> 4) + (k & 3);
}

// ---------------------------------------------------------------- WMMA operands
// half-wave of this lane: blocks are 32 x 1 x waves, so the lane is threadIdx.x (the same value as wm_half(),
// but from the source the kernels already use for their lane index)
__device__ __forceinline__ int m_half()
{
    return (int)threadIdx.x >> 4;
}
// token row that accumulator VGPR i holds in this lane
__device__ __forceinline__ int mrow(int i)
{
    return wm_acc_row_h(i, m_half());
}
// sop: an f16 operand (gfx11: the row's 16 K values; gfx12: this half's 8). gop: an operand of the e4m3 GEMMs
// (k32 steps): sop, or with D4R_FP8_WMMA the 8 e4m3 bytes of this half's K values.
#if D4R_WMMA_LAYOUT == 12
typedef h8 sop;
#else
typedef h16 sop;
#endif
typedef uint32_t u2v __attribute__((ext_vector_type(2)));
typedef uint32_t u4v_t __attribute__((ext_vector_type(4)));
#ifdef D4R_FP8_WMMA
typedef u2v gop;
typedef u4v_t wslot; // weight image slot: 16 e4m3 bytes (K slots 0..15)
#else
typedef sop gop;
typedef h16 wslot; // weight image slot: 16 f16 values
#endif

__device__ __forceinline__ f8v wmma(sop a, sop b, f8v c)
{
#if D4R_WMMA_LAYOUT == 12
    return wm_mma(__builtin_bit_cast(wm_op, a), __builtin_bit_cast(wm_op, b), c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}

// f16 operand from 16 contiguous halves in LDS (gfx12: this half's 8 of them)
__device__ __forceinline__ sop lds16(const half_t* p)
{
#if D4R_WMMA_LAYOUT == 12
    return *(const h8*)(p + 8 * m_half());
#else
    h8 lo = *(const h8*)p, hi = *(const h8*)(p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#endif
}

#ifdef D4R_FP8_WMMA
// e4m3 bytes of an f16 operand whose values are e4m3 values (K order kept: byte j = element j)
__device__ __forceinline__ gop to_fp8(sop v)
{
    uint32_t c[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        c[j] = codes8x2((hv2){v[2 * j], v[2 * j + 1]});
    return (u2v){__builtin_amdgcn_perm(c[1], c[0], 0x06040200u), __builtin_amdgcn_perm(c[3], c[2], 0x06040200u)};
}
// f32 += A B over 16 K for e4m3 operands
__device__ __forceinline__ f8v wmma8(gop a, gop b, f8v c)
{
#if defined(__GFX12__)
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(wm_i2v, a), __builtin_bit_cast(wm_i2v, b), c);
#else
    // layout shim: the e4m3 values widened to f16 (exact), then the f16 WMMA
    auto widen = [](gop q) {
        sop r;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            r[j] = e4m3_to_half((q[j >> 2] >> (8 * (j & 3))) & 0xffu);
        return r;
    };
    return wmma(widen(a), widen(b), c);
#endif
}
__device__ __forceinline__ gop lda(const half_t* p)
{
    return to_fp8(lds16(p));
}
#else
__device__ __forceinline__ f8v wmma8(gop a, gop b, f8v c)
{
    return wmma(a, b, c);
}
// GEMM A operand (activations) from LDS
__device__ __forceinline__ gop lda(const half_t* p)
{
    return lds16(p);
}
#endif
// an f16 operand built in registers -> GEMM operand
__device__ __forceinline__ gop as_gop(sop v)
{
#ifdef D4R_FP8_WMMA
    return to_fp8(v);
#else
    return v;
#endif
}

__device__ __forceinline__ f8v round16(f8v v)
{
#pragma unroll
    for (int i = 0; i < 8; ++i)
        v[i] = (float)(half_t)v[i];
    return v;
}

// A C-fragment tile: element i is row wm_acc_row(i). Exact mode stores packed f16 (NVIDIA's f16
// accumulator, rounded after every k32 step); SWIN_F32ACC keeps f32 through whole GEMM chains
// (more precise than NVIDIA; values are rounded to f16 where they are consumed).
struct T16
{
#ifdef SWIN_F32ACC
    f8v f;
    __device__ hv2 pair(int k) const { return (hv2){(half_t)f[2 * k], (half_t)f[2 * k + 1]}; }
    __device__ void set_pair(int k, hv2 p) { f[2 * k] = (float)p[0]; f[2 * k + 1] = (float)p[1]; }
    __device__ half_t get(int i) const { return (half_t)f[i]; }
    __device__ void set(int i, half_t h) { f[i] = (float)h; }
#else
    hv2 v[4];
    __device__ hv2 pair(int k) const { return v[k]; }
    __device__ void set_pair(int k, hv2 p) { v[k] = p; }
    __device__ half_t get(int i) const { return v[i / 2][i & 1]; }
    __device__ void set(int i, half_t h) { v[i / 2][i & 1] = h; }
#endif
};

__device__ __forceinline__ T16 t16_splat(half_t h)
{
    T16 t;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        t.set_pair(k, (hv2){h, h});
    return t;
}

#ifndef MIX_PAD
#define MIX_PAD ""
#endif
// acc = f16(P + acc), rounded once (v_fma_mix: verified == f16(exact sum) on 8.4M cases)
__device__ __forceinline__ hv2 mix2(float p0, float p1, hv2 c)
{
    hv2 r = c;
    asm(MIX_PAD "v_fma_mixlo_f16 %0, %1, 1.0, %2 op_sel_hi:[0,0,1]" : "+v"(r) : "v"(p0), "v"(c));
    asm("v_fma_mixhi_f16 %0, %1, 1.0, %2 op_sel:[0,0,1] op_sel_hi:[0,0,1]" : "+v"(r) : "v"(p1), "v"(c));
    return r;
}
__device__ __forceinline__ void mixacc(T16& acc, f8v P)
{
#pragma unroll
    for (int k = 0; k < 4; ++k)
        acc.set_pair(k, mix2(P[2 * k], P[2 * k + 1], acc.pair(k)));
}

#ifdef K32_CHECK
__device__ unsigned d4r_k32_bad[4 + 64 * 4];
#endif
// f16 accumulate of one k32 step (NVIDIA m16n8k32 semantics: products summed, one rounding with C)
__device__ __forceinline__ void k32(T16& acc, gop a0, gop b0, gop a1, gop b1)
{
#if defined(SWIN_F32ACC)
    acc.f = wmma8(a1, b1, wmma8(a0, b0, acc.f));
#elif defined(K32_MIX)
    f8v P = wmma8(a0, b0, (f8v){0, 0, 0, 0, 0, 0, 0, 0});
    P = wmma8(a1, b1, P);
    mixacc(acc, P);
#else // exact: C added inside the f32 WMMA, f16 rounding after the k32 step
    f8v c;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        c[i] = (float)acc.get(i);
    c = wmma8(a1, b1, wmma8(a0, b0, c));
#pragma unroll
    for (int i = 0; i < 8; ++i)
        acc.set(i, (half_t)c[i]);
#endif
}

__device__ __forceinline__ f8v splat(float v)
{
    return (f8v){v, v, v, v, v, v, v, v};
}

// lane xor shuffles without LDS: DPP within rows of 16, permlanex16 across the two rows
template <int MASK> __device__ __forceinline__ uint32_t xor_lane(uint32_t v)
{
    if constexpr (MASK == 1)
        return __builtin_amdgcn_update_dpp(0, (int)v, 0xB1, 0xf, 0xf, false);
    else if constexpr (MASK == 2)
        return __builtin_amdgcn_update_dpp(0, (int)v, 0x4E, 0xf, 0xf, false);
    else if constexpr (MASK == 4)
        return __builtin_amdgcn_update_dpp(0, (int)v, 0x164, 0xf, 0xf, false);
    else if constexpr (MASK == 8)
        return __builtin_amdgcn_update_dpp(0, (int)v, 0x168, 0xf, 0xf, false);
    else
        return __builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210, 0xfedcba98, false, false);
}
template <int MASK> __device__ __forceinline__ half_t hxor(half_t v)
{
    return __builtin_bit_cast(half_t, (uint16_t)xor_lane<MASK>(__builtin_bit_cast(uint16_t, v)));
}

__device__ __forceinline__ int mirror(int v, int n)
{
    v = v < 0 ? -v : v;
    int o = 2 * n - 2 - v;
    return v < o ? v : o;
}

__device__ __forceinline__ half_t hload(const uint8_t* w, int off)
{
    return __builtin_bit_cast(half_t, *(const uint16_t*)(w + off));
}

__device__ __forceinline__ void wave_sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Token T (0..63) of a block: 8x8 tokens, windows of 4x4 (T >> 4)
__device__ __forceinline__ int tok_x(int T)
{
    return (T & 3) + 4 * ((T >> 4) & 1);
}
__device__ __forceinline__ int tok_y(int T)
{
    return ((T >> 2) & 3) + 4 * (T >> 5);
}

// ---------------------------------------------------------------- weight images
// A GEMM's B operands in WMMA lane order: slot index dst + ((kc * 2 + s) * NT + nt) * 16 + lane16 holds the 16
// K slots of logical k = 32 kc + kslot(s, i), n = 16 nt + lane16 (i = 0..15), as f16 values (wslot = h16) or,
// with D4R_FP8_WMMA, as the e4m3 bytes. gfx12 lanes read the half of the slot with their 8 K slots.
struct GemmDesc
{
    int dst; // in slots
    int base, Ks, Ns, KC, NT;
};

// The tile index is wave-uniform (it depends only on the wave, the loop counters and constants), so it
// is formed in scalar registers and the load uses a scalar base plus the lane's constant offset.
__device__ __forceinline__ gop bload(const wslot* w16, int gemm_dst, int NT, int kc, int s, int nt, int l16)
{
#ifdef SWIN_SCALAR_BLOAD
    const int tile = __builtin_amdgcn_readfirstlane(gemm_dst + ((kc * 2 + s) * NT + nt) * 16);
    const wslot* t = w16 + tile;
    const int at = l16;
#else
    const wslot* t = w16;
    const int at = gemm_dst + ((kc * 2 + s) * NT + nt) * 16 + l16;
#endif
#if D4R_WMMA_LAYOUT == 12
    return ((const gop*)t)[2 * at + m_half()];
#else
    return t[at];
#endif
}

__device__ __forceinline__ void expand_weights(const uint8_t* w, wslot* w16, const GemmDesc* descs, int ndesc, int idx)
{
    int d = 0;
    while (d + 1 < ndesc && idx >= descs[d + 1].dst)
        ++d;
    const GemmDesc g = descs[d];
    const int local = idx - g.dst;
    if (local >= g.KC * 2 * g.NT * 16)
        return;
    const int l16 = local & 15, rest = local >> 4;
    const int nt = rest % g.NT, s = (rest / g.NT) & 1, kc = rest / g.NT / 2;
    const int n = 16 * nt + l16;
#ifdef D4R_FP8_WMMA
    u4v_t v;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b)
            word |= (uint32_t)w[woff(g.base, g.Ks, g.Ns, 32 * kc + kslot(s, 4 * j + b), n)] << (8 * b);
        v[j] = word;
    }
#else
    h16 v;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        v[i] = e4m3_to_half(w[woff(g.base, g.Ks, g.Ns, 32 * kc + kslot(s, i), n)]);
#endif
    w16[idx] = v;
}
