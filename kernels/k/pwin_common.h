// Native RDNA3 (gfx11 wave32 WMMA) kernels for the DLSS 4 (preset K) DltssPaddedWinLayer blocks.
// Semantics: kernels/tools/pwin_model.py (verified against the PTX interpreter, kernels/tools/ptxsim.py).
//
// Layout conventions (transposed WMMA, as in the preset M kernels): a GEMM Y = X W is computed as
// Y^T = W^T X^T with the weights as the WMMA A operand (lane l: output channel l & 15, 16 K values)
// and the activations as the B operand (lane l: token l & 15, 16 K values). The f32 result then has
// lane l = token l & 15 and VGPR i = output channel 2i + (l >> 4) of the 16-channel tile ("D^T").
#pragma once
#ifdef D4R_DEVICE_ONLY_MINIMAL
#include "../common/hip_device_minimal.h"
#else
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>

#pragma clang fp contract(off)

typedef _Float16 half_t;
typedef _Float16 hv2 __attribute__((ext_vector_type(2)));
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
#include "../common/wmma_backend.h"

__device__ __forceinline__ uint32_t lane_id()
{
    return __builtin_amdgcn_mbcnt_lo(~0u, 0u);
}

__device__ __forceinline__ f8v wmma(u8v a, u8v b, f8v c)
{
    return d4r_wmma_legacy_layout(__builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b), c);
}

// one k16 step with the f16 accumulator of NVIDIA's f16 wmma (rounded after the step)
// PWIN_F32ACC: keep the accumulator in f32 through the chain (rounded to f16 where the values are used)
__device__ __forceinline__ f8v mma16(u8v a, u8v b, f8v c)
{
    f8v d = wmma(a, b, c);
#ifndef PWIN_F32ACC
#pragma unroll
    for (int i = 0; i < 8; ++i)
        d[i] = (float)(half_t)d[i];
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

// D^T f16 values (own[i] = channel 2i + hf of token l & 15) -> B operand of the same token (all 16
// channels in order): the other parity comes from lane l ^ 16
__device__ __forceinline__ u8v operand_from_dt(const half_t own[8])
{
    const uint32_t hf = lane_id() >> 4;
    uint32_t mine[4], theirs[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        mine[j] = pack2(own[2 * j], own[2 * j + 1]);
        theirs[j] = other_half(mine[j]);
    }
#ifndef PWIN_NO_PERM
    // v_perm_b32(theirs, mine): bytes 0-3 = mine, 4-7 = theirs. Even j: the low halves, odd j: the high
    // halves, in the order (mine, theirs) for hf = 0 and (theirs, mine) for hf = 1.
    const uint32_t sel_lo = hf ? 0x01000504u : 0x05040100u, sel_hi = hf ? 0x03020706u : 0x07060302u;
    u8v r;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        r[j] = __builtin_amdgcn_perm(theirs[j >> 1], mine[j >> 1], (j & 1) ? sel_hi : sel_lo);
    return r;
#else
    u8v r;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        const half_t a = (j & 1) ? hi16(mine[j >> 1]) : lo16(mine[j >> 1]);
        const half_t b = (j & 1) ? hi16(theirs[j >> 1]) : lo16(theirs[j >> 1]);
        r[j] = hf ? pack2(b, a) : pack2(a, b);
    }
    return r;
#endif
}

__device__ __forceinline__ u8v operand_from_f8(f8v d)
{
    half_t h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        h[i] = (half_t)d[i];
    return operand_from_dt(h);
}

// element c (0..15) of a B operand (16 halves)
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

// A-operand image of a K x N weight: tile (kt, nt) = 16 lanes x 8 dwords (lane = output channel)
struct WDesc
{
    int src;       // byte offset of the fragments in the weight buffer
    int kt_stride; // bytes between k tiles
    int nt_stride; // bytes between n tiles
    int KT, NT;    // tiles
    int dst;       // u8v index of the image in the prep buffer
};

// weight operand tile `tile` of an A-operand image (16 lanes x 32 bytes): lanes l and l + 16 need the same
// row, so each half-wave loads one 16-byte half and the halves are swapped across (halves L2 traffic)
__device__ __forceinline__ u8v wtile(const u8v* __restrict__ img, int tile)
{
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const u4v mine = ((const u4v*)&img[tile * 16 + m])[hf];
    u4v other;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        other[j] = other_half(mine[j]);
    const u4v lo = hf ? other : mine, hi = hf ? mine : other;
    return (u8v){lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
}
