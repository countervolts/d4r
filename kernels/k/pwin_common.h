// Native kernels (gfx11 / gfx12 wave32 WMMA) for the DLSS 4 (preset K) DltssPaddedWinLayer blocks.
// Semantics: kernels/tools/pwin_model.py (verified against the PTX interpreter, kernels/tools/ptxsim.py).
//
// Layout conventions (transposed WMMA, as in the preset M kernels): a GEMM Y = X W is computed as
// Y^T = W^T X^T with the weights as the WMMA A operand (lane l: output channel l & 15) and the activations
// as the B operand (lane l: token l & 15). The f32 result then has lane l = token l & 15 and VGPR i =
// output channel wm_acc_row(i) of the 16-channel tile ("D^T"); ../common/wmma_layout.h has the per-target
// register layouts (gfx11: channel 2i + (l >> 4); gfx12: channel i + 8 (l >> 4)).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
// The accuracy set rounds every k16 step to f16 and widens it again (mma16); the inline-asm pair conversion hides
// the last step's f16 from the compiler, which then converts it a third time. Without it the extra conversions
// fold away (enc0 2739 -> 2147, dec0 3011 -> 2355; gfx1101 vkhip 1080p 181 -> 170 and 185 -> 174 us, byte-identical).
#if defined(D4R_ACCURACY) && defined(WM_CVT_ASM)
#undef WM_CVT_ASM
#endif
#include "../common/wmma_layout.h"
#if defined(D4R_ACCURACY) && defined(PWIN_F32ACC)
#undef PWIN_F32ACC
#endif
// D4R_FAST_MATH: the opt-in approximate variant (FMA contraction allowed; not bit-identical to NVIDIA's rounding)
#ifndef D4R_FAST_MATH
#pragma clang fp contract(off)
#endif

typedef _Float16 half_t;
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));

__device__ __forceinline__ half_t half_reciprocal(half_t h)
{
#ifdef PWIN_NATIVE_HRECIP
    return __builtin_amdgcn_rcph(h);
#else
    return (half_t)__builtin_amdgcn_rcpf((float)h);
#endif
}

__device__ __forceinline__ half_t half_rsqrt(half_t h)
{
#ifdef PWIN_NATIVE_HRECIP
    return __builtin_amdgcn_rsqh(h);
#else
    return (half_t)__builtin_amdgcn_rsqf((float)h);
#endif
}

// LDS row padding in halves: activation rows of N halves are stored N + PWIN_PAD apart, so the 16 token rows a
// WMMA operand load (op_lds / lds_row16) touches start on distinct banks (row stride = 16 mod 32 bytes). With
// unpadded 128- or 256-byte rows every lane hits the same banks (measured: 6x slower WMMA feed). 0 = old layout.
#ifndef PWIN_PAD
#define PWIN_PAD 8
#endif
#define LDSR(N) ((N) + PWIN_PAD)

__device__ __forceinline__ uint32_t lane_id()
{
    return __builtin_amdgcn_mbcnt_lo(~0u, 0u);
}

// RDNA2: the f16-accumulator WMMA variants need the instruction; the f32 path (software products) serves them
#ifdef __GFX10__
#undef PWIN_NATIVE_F16ACC
#undef PWIN_F16ACC_NATIVE
#endif
// RDNA2: the weight images use plain global loads. The raw-buffer path below builds its resource with gfx11's
// dword3 (0x31004000: unified format 4, 8_SSCALED), which gfx11 ignores for untyped loads; gfx10 descriptors want
// a 32-bit format there (AMD's own libraries use 0x31014000 on gfx103x), and no gfx10 encoding has been validated
// on RDNA2 hardware, so RDNA2 takes no descriptor at all. Keyed on D4R_WMMA_SOFT (always set on gfx10) so that the
// RDNA3 validation build of the RDNA2 kernels runs the same loads.
#ifdef D4R_WMMA_SOFT
#ifndef PWIN_NO_BUFFER_IMG
#define PWIN_NO_BUFFER_IMG
#endif
#undef PWIN_GELU_LUT_BUFFER
#endif
typedef wm_op op_t; // WMMA operand: u8v (gfx11) or u4v (gfx12)

// one k16 step with the f16 accumulator of NVIDIA's f16 wmma (rounded after the step)
// PWIN_F32ACC: keep the accumulator in f32 through the chain (rounded to f16 where the values are used)
__device__ __forceinline__ f8v mma16(const op_t& a, const op_t& b, f8v c)
{
#if defined(PWIN_NATIVE_F16ACC) && !defined(PWIN_F32ACC) && D4R_WMMA_LAYOUT == 11
    h16 hc = {};
#pragma unroll
    for (int i = 0; i < 8; ++i)
        hc[2 * i] = (half_t)c[i];
    const h16 hd = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
        __builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b), hc, false);
    f8v d;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        d[i] = (float)hd[2 * i];
#else
    f8v d = wm_mma(a, b, c);
#ifndef PWIN_F32ACC
#pragma unroll
    for (int i = 0; i < 8; ++i)
        d[i] = (float)(half_t)d[i];
#endif
#endif
    return d;
}

__device__ __forceinline__ f8v splat8(float v)
{
    return (f8v){v, v, v, v, v, v, v, v};
}

__device__ __forceinline__ uint32_t other_half(uint32_t v)
{
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
}

__device__ __forceinline__ uint32_t pack2(half_t lo, half_t hi)
{
    return __builtin_bit_cast(uint32_t, (hv2){lo, hi});
}

__device__ __forceinline__ half_t lo16(uint32_t v)
{
    return __builtin_bit_cast(hv2, v)[0];
}
__device__ __forceinline__ half_t hi16(uint32_t v)
{
    return __builtin_bit_cast(hv2, v)[1];
}

// D^T f16 values (own[i] = channel wm_acc_row(i) of token l & 15) -> operand of the same token (the
// channels become K)
__device__ __forceinline__ op_t operand_from_dt(const half_t own[8])
{
    return wm_op_from_acc(own);
}

__device__ __forceinline__ op_t operand_from_f8(f8v d)
{
    return wm_op_from_f8(d);
}

// element c (0..15) of a row of 16 halves (VALU code; not a WMMA operand)
__device__ __forceinline__ half_t op_get(const u8v& v, int c)
{
    return (c & 1) ? hi16(v[c >> 1]) : lo16(v[c >> 1]);
}

__device__ __forceinline__ u8v lds_row16(const half_t* p)
{
    const u4v a = *(const u4v*)p, b = *(const u4v*)(p + 8);
    return (u8v){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
}

// Global activation traffic (tokens in, rows out): PWIN_NT_LOAD / PWIN_NT_STORE mark it non-temporal so
// it does not evict the weight images from the caches. Same data either way.
__device__ __forceinline__ u8v gload_row16(const half_t* p)
{
#ifdef PWIN_NT_LOAD
    const u4v a = __builtin_nontemporal_load((const u4v*)p), b = __builtin_nontemporal_load((const u4v*)(p + 8));
#else
    const u4v a = *(const u4v*)p, b = *(const u4v*)(p + 8);
#endif
    return (u8v){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
}
__device__ __forceinline__ void gstore16(half_t* p, u4v v)
{
#ifdef PWIN_NT_STORE
    __builtin_nontemporal_store(v, (u4v*)p);
#else
    *(u4v*)p = v;
#endif
}
__device__ __forceinline__ void gstore_row16(half_t* dst, const u8v& v)
{
    gstore16(dst, (u4v){v[0], v[1], v[2], v[3]});
    gstore16(dst + 8, (u4v){v[4], v[5], v[6], v[7]});
}

// WMMA operands from rows of 16 channels: LDS, global (PWIN_NT_LOAD as above), prep images (32-byte slots)
__device__ __forceinline__ op_t op_lds(const half_t* p)
{
    return wm_op_load(p);
}
__device__ __forceinline__ op_t op_gload(const half_t* p)
{
#if D4R_WMMA_LAYOUT == 12
#ifdef PWIN_NT_LOAD
    return __builtin_nontemporal_load((const u4v*)(p + 8 * wm_half()));
#else
    return *(const u4v*)(p + 8 * wm_half());
#endif
#else
    return gload_row16(p);
#endif
}
__device__ __forceinline__ op_t op_img(const u8v* __restrict__ img, int idx)
{
    return wm_op_image(img, idx);
}
// 8 floats of image slot `slot` + this lane's half (f32 accumulator-init vectors the prep kernel stores after the
// operand tiles: slot pair (2 v, 2 v + 1) = rows wm_acc_row_h(., h) of vector v), slot uniform
__device__ __forceinline__ f8v img_vec(const u8v* __restrict__ img, int slot)
{
    const uint32_t h = lane_id() >> 4;
#if defined(PWIN_VEC_SCALAR) // both layouts: slot pair (2 v, 2 v + 1) = the two halves' rows
    // both halves' slots with wave-uniform (scalar) loads, then one select per value (instead of a per-lane
    // vector load, or 8 shifts + 8 f16 -> f32 conversions from the weights); the same f32 values
    const u8v x0 = img[slot], x1 = img[slot + 1];
    u8v x;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        x[i] = h ? x1[i] : x0[i];
    return __builtin_bit_cast(f8v, x);
#elif D4R_WMMA_LAYOUT == 12 || defined(PWIN_NO_BUFFER_IMG)
    const u8v x = img[slot + (int)h];
    return __builtin_bit_cast(f8v, x);
#else
    const __amdgpu_buffer_rsrc_t r = __builtin_amdgcn_make_buffer_rsrc((void*)img, (short)0, 0x7fffffff, 0x31004000);
    const uint32_t vo = h * 32u, so = 32u * (uint32_t)slot;
    const u4v a = __builtin_amdgcn_raw_buffer_load_b128(r, vo, so, 0), b = __builtin_amdgcn_raw_buffer_load_b128(r, vo + 16, so, 0);
    return __builtin_bit_cast(f8v, (u8v){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]});
#endif
}
// slot (tile, lane l & 15) of an operand image in global memory: tile is wave-uniform, so the row address stays a
// scalar base and only the lane indexes it (global loads with an SGPR base and a VGPR offset; written as one
// index tile * 16 + m every load paid 64-bit VALU address arithmetic)
__device__ __forceinline__ op_t op_imgt(const u8v* __restrict__ img, int tile)
{
#if D4R_WMMA_LAYOUT == 12 || defined(PWIN_NO_BUFFER_IMG)
    return wm_op_image(img + 16 * tile, (int)(lane_id() & 15));
#else
    // gfx11: a raw buffer load (resource from img: base in SGPRs, 0x31004000 = gfx11 raw dword3, validated), the
    // tile offset as the scalar offset and the lane's 32-byte slot as the vector offset: no VALU address math
    const __amdgpu_buffer_rsrc_t r = __builtin_amdgcn_make_buffer_rsrc((void*)img, (short)0, 0x7fffffff, 0x31004000);
    const uint32_t vo = (lane_id() & 15) * 32u, so = 512u * (uint32_t)tile;
    const u4v a = __builtin_amdgcn_raw_buffer_load_b128(r, vo, so, 0), b = __builtin_amdgcn_raw_buffer_load_b128(r, vo + 16, so, 0);
    return (u8v){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
#endif
}
// operand -> slot idx of an operand image in LDS (read back with op_img)
__device__ __forceinline__ void op_img_store(u8v* img, int idx, const op_t& v)
{
#if D4R_WMMA_LAYOUT == 12
    ((u4v*)img)[2 * idx + wm_half()] = v;
#else
    if (!wm_half())
        img[idx] = v;
#endif
}
// operand (16 channels of this lane's token) -> row of `count` (8 or 16) channels in global memory, if ok
__device__ __forceinline__ void op_gstore(half_t* dst, const op_t& v, bool ok, int count = 16)
{
#if D4R_WMMA_LAYOUT == 12
    if (ok && (wm_half() == 0 || count >= 16))
        gstore16(dst + 8 * wm_half(), v);
#else
    if (!wm_half() && ok)
    {
        gstore16(dst, (u4v){v[0], v[1], v[2], v[3]});
        if (count >= 16)
            gstore16(dst + 8, (u4v){v[4], v[5], v[6], v[7]});
    }
#endif
}
// operand -> row of 16 channels in LDS
__device__ __forceinline__ void op_store(half_t* dst, const op_t& v)
{
    wm_op_store(dst, v);
}

__device__ __forceinline__ void block_sync()
{
    __syncthreads();
}

__device__ __forceinline__ int mirror(int v, int n)
{
    v = v < 0 ? -v : v;
    return v < 2 * n - 2 - v ? v : 2 * n - 2 - v;
}

// ------------------------------------------------ pre-swizzled f16 weight fragments (NVIDIA)
// logical B[k][n] of one 16x16 fragment (512 bytes)
__device__ __forceinline__ int frag_offset(int k, int n)
{
    return 64 * (n & 7) + 16 * ((k & 7) >> 1) + 8 * (n >> 3) + 4 * (k >> 3) + 2 * (k & 1);
}

// Diagnostic phase stamps (-DPWIN_PROF): per-wave SHADER_CYCLES deltas summed into d4r_dbg[k], wave counts into
// d4r_dbg[32 + k] (read back by bench/native/pwbench). Region k ends at PROF(k). Off: no code.
#ifdef PWIN_PROF
__device__ unsigned long long d4r_dbg[64];
#define PROF_INIT uint32_t prof_last_ = (uint32_t)__builtin_readcyclecounter()
#define PROF(k)                                                                                                    \
    do                                                                                                             \
    {                                                                                                              \
        const uint32_t n_ = (uint32_t)__builtin_readcyclecounter();                                                \
        if (lane_id() == 0)                                                                                        \
        {                                                                                                          \
            __hip_atomic_fetch_add(&d4r_dbg[k], (unsigned long long)((n_ - prof_last_) & 0xfffffu), __ATOMIC_RELAXED, \
                                   __HIP_MEMORY_SCOPE_AGENT);                                                      \
            __hip_atomic_fetch_add(&d4r_dbg[32 + (k)], 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);          \
        }                                                                                                          \
        prof_last_ = n_;                                                                                           \
    } while (0)
#else
#define PROF_INIT do {} while (0)
#define PROF(k) do {} while (0)
#endif
