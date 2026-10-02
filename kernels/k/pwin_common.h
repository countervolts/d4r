// Native kernels (gfx11 / gfx12 wave32 WMMA) for the DLSS 4 (preset K) DltssPaddedWinLayer blocks.
// Semantics: kernels/tools/pwin_model.py (verified against the PTX interpreter, kernels/tools/ptxsim.py).
//
// Layout conventions (transposed WMMA, as in the preset M kernels): a GEMM Y = X W is computed as
// Y^T = W^T X^T with the weights as the WMMA A operand (lane l: output channel l & 15) and the activations
// as the B operand (lane l: token l & 15). The f32 result then has lane l = token l & 15 and VGPR i =
// output channel wm_acc_row(i) of the 16-channel tile ("D^T"); ../common/wmma_layout.h has the per-target
// register layouts (gfx11: channel 2i + (l >> 4); gfx12: channel i + 8 (l >> 4)).
#pragma once
#ifdef D4R_DEVICE_ONLY_MINIMAL
#include "../common/hip_device_minimal.h"
#else
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>
#include "../common/wmma_layout.h"
#if defined(D4R_ACCURACY) && defined(PWIN_F32ACC)
#undef PWIN_F32ACC
#endif
#pragma clang fp contract(off)

typedef _Float16 half_t;
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8v __attribute__((ext_vector_type(8)));
#ifdef D4R_K_PACKED_ACC
#if !defined(__GFX12__) || !defined(D4R_K_FP16_BASELINE) || defined(D4R_K_F16_WMMA)
#error "packed accumulator storage requires gfx12 strict F32 WMMA baseline"
#endif
// Keep already-rounded values packed between steps. Every matrix instruction
// still uses F32 C/D and rounds once to half afterwards, like the baseline.
typedef _Float16 acc8v __attribute__((ext_vector_type(8)));
#else
typedef f8v acc8v;
#endif
typedef uint32_t u8v __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint32_t lane_id()
{
    return __builtin_amdgcn_mbcnt_lo(~0u, 0u);
}

typedef wm_op op_t; // WMMA operand: u8v (gfx11) or u4v (gfx12)

// The CUDA fragment reduction rounds products and every tree node to f16.
// A float sum of the squared halves changes rsqrt and can amplify differences
// through attention. The strict baseline preserves its channel grouping.
template <int C>
__device__ __forceinline__ half_t l2_sum(const half_t* row)
{
#ifdef D4R_K_FP16_BASELINE
    static_assert(C % 32 == 0 && C <= 160, "audited K channel counts");
    hv2 partial[4];
#pragma unroll
    for (int t = 0; t < 4; ++t)
    {
        hv2 halves[2];
#pragma unroll
        for (int parity = 0; parity < 2; ++parity)
        {
            hv2 pairs[C / 32];
#pragma unroll
            for (int group = 0; group < C / 32; ++group)
            {
                const hv2 a = *(const hv2*)(row + 2 * t + 8 * parity + 32 * group);
                const hv2 b = *(const hv2*)(row + 2 * t + 8 * parity + 32 * group + 16);
                pairs[group] = (a * a) + (b * b);
            }
            hv2 value = pairs[0];
            if constexpr (C >= 64) value = value + pairs[1];
            if constexpr (C == 96) value = value + pairs[2];
            if constexpr (C >= 128) value = value + (pairs[2] + pairs[3]);
            if constexpr (C == 160) value = value + pairs[4];
            halves[parity] = value;
        }
        partial[t] = halves[0] + halves[1];
    }
    const hv2 sum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
    return sum[0] + sum[1];
#else
    float sum = 0;
#pragma unroll
    for (int c = 0; c < C; ++c) sum += (float)(half_t)(row[c] * row[c]);
    return (half_t)sum;
#endif
}

// one k16 step with the f16 accumulator of NVIDIA's f16 wmma (rounded after the step)
// PWIN_F32ACC: keep the accumulator in f32 through the chain (rounded to f16 where the values are used)
__device__ __forceinline__ acc8v mma16(const op_t& a, const op_t& b, acc8v c)
{
#ifdef D4R_K_PACKED_ACC
    const f8v fc = __builtin_convertvector(c, f8v);
    const f8v d = wm_mma(a, b, fc);
    return __builtin_convertvector(d, acc8v);
#elif defined(D4R_K_F16_WMMA)
#if !defined(__GFX12__) || D4R_WMMA_LAYOUT != 12 || !defined(D4R_K_FP16_BASELINE)
#error "experimental packed F16 WMMA requires gfx12 native layout and strict K baseline"
#endif
    return wm_mma_f16_step(a, b, c);
#else
    f8v d = wm_mma(a, b, c);
#if !defined(PWIN_F32ACC) || defined(D4R_K_FP16_BASELINE)
#pragma unroll
    for (int i = 0; i < 8; ++i)
        d[i] = (float)(half_t)d[i];
#endif
    return d;
#endif
}

__device__ __forceinline__ acc8v splat8(float v)
{
#ifdef D4R_K_PACKED_ACC
    const half_t h = (half_t)v;
    return (acc8v){h, h, h, h, h, h, h, h};
#else
    return (acc8v){v, v, v, v, v, v, v, v};
#endif
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

__device__ __forceinline__ op_t operand_from_f8(acc8v d)
{
#ifdef D4R_K_PACKED_ACC
    const half_t h[8] = {d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]};
    return wm_op_from_acc(h);
#else
    return wm_op_from_f8(d);
#endif
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
