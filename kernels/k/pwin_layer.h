// Native DltssPaddedWinLayer (DLSS 4, preset K) template: 8x8 windows, H heads x 32, C channels.
// One block per window, 4 waves (one 16-token tile each). Semantics: kernels/tools/pwin_model.py (swin_core,
// pwin_encoder); weight layout: pwin_model.Layout.
#pragma once
#include "pwin_common.h"
#ifndef PWIN_VGPR
#define PWIN_VGPR
#endif

struct PwinParams
{
    int W, H;              // 0: token grid
    const uint8_t* in;     // 8
    const uint8_t* skip;   // 16 (decoders)
    uint8_t* out24;        // 24: encoders: merged output; decoders: output
    uint8_t* out32;        // 32: encoders: full-resolution output
    uint64_t pad40, pad48; // 40
    int sx, sy;            // 56
    const uint8_t* w;      // 64
    uint8_t rest[176 - 72];
};
static_assert(sizeof(PwinParams) == 176, "param block");

#if defined(D4R_ACCURACY) && defined(__GFX11__) && D4R_WMMA_LAYOUT == 11
// The translated reference's deep-layer epilogues materialize the NVIDIA accumulator fragment inside
// each predicated store. Its two half values come from WMMA lanes g + 8*b and g + 8*b + 16, where
// g = token % 8 and b = (token % 16) / 8. ds_bpermute returns zero for a source lane outside that
// store's execution mask. Reproduce that mask before writing the canonical rows; the register/LDS
// values used by an encoder's patch merge remain intact.
__device__ __forceinline__ bool ptx_store_source_live(const PwinParams& p, int bx, int by, int token, int parity)
{
    const int b = (token & 15) >> 3;
    const int source = (token & 7) + 8 * b + 16 * parity;
    const int srcToken = (token & ~15) + (source >> 2) + 8 * b;
    const int x = 8 * bx - p.sx + (srcToken & 7), y = 8 * by - p.sy + (srcToken >> 3);
    return x >= 0 && x < p.W && y >= 0 && y < p.H;
}
__device__ __forceinline__ u8v ptx_mask_store_row(const PwinParams& p, int bx, int by, int token, u8v row)
{
    const uint32_t mask = (ptx_store_source_live(p, bx, by, token, 0) ? 0x0000ffffu : 0u) |
                          (ptx_store_source_live(p, bx, by, token, 1) ? 0xffff0000u : 0u);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        row[i] &= mask;
    return row;
}
#endif

template <int H_, int C_, int COUT_, bool POS_ = false> struct PwinLayout
{
    static constexpr bool POS = POS_; // position-only attention (trait flag 2): the table is P itself
    static constexpr int H = H_, C = C_, COUT = COUT_, KT = C / 16, NCH = C / 32, NMLP = C / 8;
    static constexpr int G1 = 0, QKV = 2 * C, HEAD = 192 * C, BIAS = QKV + HEAD * H, BO = BIAS + 8192 * H,
                         WO = BO + 2 * C, G2 = WO + 64 * C * H, B2 = G2 + 2 * C, W1 = B2 + 2 * C, B1 = W1 + 8 * C * C,
                         W2 = B1 + 8 * C, PM = W2 + 8 * C * C;
    // patch merge: NVIDIA's warps own COUT / H channels each, allocated as whole 16-column tiles
    static constexpr int NPW = COUT / H, NPA = (NPW + 15) / 16 * 16, PMT = H * NPA / 16, PMB = PM + 8 * C * NPA * H;
    // prep image tiles (16 lanes of u8v each)
    static constexpr int T_QKV = 0;                           // [h][j][kt KT][nt 2]
    static constexpr int T_WO = T_QKV + H * 3 * KT * 2;        // [ks 2H][nt KT]
    static constexpr int T_W1 = T_WO + 2 * H * KT;             // [c][kt KT][nt 2]
    static constexpr int T_W2 = T_W1 + NMLP * KT * 2;          // [c][ks 2][nt KT]
    static constexpr int T_PM = T_W2 + NMLP * 2 * KT;          // [ks C/4][nt COUT/16]
    static constexpr int T_END = T_PM + (C / 4) * PMT;
    static constexpr int NBIAS = H * 4 * 4 * 32;               // u4v per lane
    // f32 accumulator-init vectors in image slots after the tiles (see img_vec): Wo bias, MLP fc1 biases, merge bias
    static constexpr int V_BO = 0, V_B1 = V_BO + KT, V_PM = V_B1 + 2 * NMLP, NV = V_PM + PMT;
    static constexpr int VEC0 = T_END * 16, IMG_SLOTS = VEC0 + 2 * NV;
    static constexpr int PREP_ITEMS = T_END * 16 + NBIAS + 2 * NV;

    __device__ static const uint8_t* extra_vec(const uint8_t*, const uint8_t*, int) { return nullptr; }

    __device__ static int src_of(int tile)
    {
        if (tile < T_WO)
        {
            const int nt = tile & 1, kt = (tile >> 1) % KT, j = (tile / (2 * KT)) % 3, h = tile / (6 * KT);
            return QKV + HEAD * h + 2048 * (3 * (kt >> 1) + j) + 512 * (kt & 1) + 1024 * nt;
        }
        if (tile < T_W1)
        {
            const int t = tile - T_WO, nt = t % KT, ks = t / KT;
            return WO + 64 * C * (ks >> 1) + 512 * (ks & 1) + 1024 * nt;
        }
        if (tile < T_W2)
        {
            const int t = tile - T_W1, nt = t & 1, kt = (t >> 1) % KT, c = t / (2 * KT);
            return W1 + 64 * C * c + 512 * kt + 32 * C * nt;
        }
        if (tile < T_PM)
        {
            const int t = tile - T_W2, nt = t % KT, ks = (t / KT) & 1, c = t / (2 * KT);
            return W2 + 64 * C * c + 512 * ks + 1024 * nt;
        }
        const int t = tile - T_PM, nt = t % PMT, ks = t / PMT;
        return PM + 512 * ks + 128 * C * nt; // allocated tiles are contiguous
    }
};

__device__ __forceinline__ half_t wload(const uint8_t* w, int byte)
{
    return *(const half_t*)(w + byte);
}

// channels 16 nt + wm_acc_row(i) (i = 0..7) of an f16 vector at byte offset `byte` (16 nt already added)
__device__ __forceinline__ void dvec8(const uint8_t* w, int byte, half_t out[8])
{
    wm_acc_vec_load((const half_t*)(w + byte), out);
}

__device__ __forceinline__ u8v img_tile(const uint8_t* w, int src, int n)
{
    u8v r;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        r[j] = pack2(wload(w, src + frag_offset(2 * j, n)), wload(w, src + frag_offset(2 * j + 1, n)));
    return r;
}

// one (vector, half) slot of the f32 vector table: the 8 rows wm_acc_row_h(i, h) of the 16 halves at src, as floats
__device__ __forceinline__ void vec_slot(u8v* img, int slot, const uint8_t* src, int h)
{
    f8v f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        f[i] = (float)wload(src, 2 * wm_acc_row_h(i, h));
    img[slot] = __builtin_bit_cast(u8v, f);
}

// Optional 640-byte module table for G2/B2 in each half-wave's accumulator order.
#if defined(PWIN_PACKED_RESIDUAL) && D4R_WMMA_LAYOUT == 11
__device__ __attribute__((aligned(16))) u4v g_residual_vectors[40];
#endif

template <class L> __device__ __forceinline__ void residual_vector_prep(const uint8_t* w, int idx)
{
#if defined(PWIN_PACKED_RESIDUAL) && D4R_WMMA_LAYOUT == 11
    static_assert(L::KT <= 10, "residual table capacity");
    if (idx >= 0 && idx < 4 * L::KT)
    {
        const int kind = idx / (2 * L::KT), nt = (idx >> 1) % L::KT, h = idx & 1;
        const int base = (kind ? L::B2 : L::G2) + 32 * nt;
        u4v v;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = pack2(wload(w, base + 2 * (4 * j + h)), wload(w, base + 2 * (4 * j + 2 + h)));
        g_residual_vectors[idx] = v;
    }
#endif
}

template <class L> __device__ __forceinline__ void residual_vector(const uint8_t* w, int kind, int nt, half_t out[8])
{
#if defined(PWIN_PACKED_RESIDUAL) && D4R_WMMA_LAYOUT == 11
    static_assert(L::KT <= 10, "residual table capacity");
    const u4v v = g_residual_vectors[2 * L::KT * kind + 2 * nt + (lane_id() >> 4)];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        out[2 * j] = lo16(v[j]);
        out[2 * j + 1] = hi16(v[j]);
    }
#else
    dvec8(w, (kind ? L::B2 : L::G2) + 32 * nt, out);
#endif
}

// prep items of the f32 vector table (j = 0 .. 2 NV - 1): core vectors from the core weights wc, the derived
// layouts' extra vectors from w0 (the layer's weights pointer). MERGE: the layer has a patch merge (its merge bias
// exists; for decoders and the bottleneck that offset lies past the layer's weights, so it is never read).
template <class L, bool MERGE> __device__ void pwin_vec_prep(const uint8_t* w0, const uint8_t* wc, u8v* img, int j)
{
    const int v = j >> 1, h = j & 1;
    const uint8_t* src = nullptr;
    if (v < L::V_B1)
        src = wc + L::BO + 32 * (v - L::V_BO);
    else if (v < L::V_PM)
        src = wc + L::B1 + 32 * (v - L::V_B1);
    else if (v < L::V_PM + L::PMT)
    {
        if (!MERGE)
            return;
        src = wc + L::PMB + 32 * (v - L::V_PM);
    }
    else
        src = L::extra_vec(w0, wc, v);
    if (src)
        vec_slot(img, L::VEC0 + j, src, h);
}

// accumulator init vector v (16 halves at w + byte as floats in accumulator row order): from the prepared f32 table
// (one 32-byte load per lane), or with PWIN_VEC_FROM_WEIGHTS from the weights (load + 8 shifts + 8 conversions;
// faster in some layers). The same values.
template <class L> __device__ __forceinline__ f8v acc_vec(const u8v* img, const uint8_t* w, int byte, int v)
{
#ifdef PWIN_VEC_FROM_WEIGHTS
    half_t t[8];
    dvec8(w, byte, t);
    f8v d;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        d[i] = (float)t[i];
    (void)img; (void)v;
    return d;
#else
    (void)w; (void)byte;
    return img_vec(img, L::VEC0 + 2 * v);
#endif
}

template <class L> __device__ void pwin_prep_at(const PwinParams& p, u8v* img, u4v* bias, int idx)
{
    const uint8_t* w = p.w;
    residual_vector_prep<L>(w, idx);
    if (idx < L::T_END * 16)
    {
        img[idx] = img_tile(w, L::src_of(idx >> 4), idx & 15);
        return;
    }
    const int b = idx - L::T_END * 16;
    if (b >= L::NBIAS)
        return;
    if constexpr (L::POS)
    {
        // B operand of P V: lane = query 16 qt + l16, 16 keys 16 kt .. 16 kt + 15 (u8v per lane; 16 lanes)
        if (b >= L::H * 4 * 4 * 16)
            return;
        const int l16 = b & 15, kt = (b >> 4) & 3, qt = (b >> 6) & 3, h = b >> 8, r = l16;
        u8v v;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            half_t e[2];
#pragma unroll
            for (int u = 0; u < 2; ++u)
            {
                const int c = 2 * j + u;
                const int off = 512 * (4 * kt + qt) + 64 * (r & 7) + 16 * ((c & 7) >> 1) + 2 * (c & 1) + 4 * (r >> 3) + 8 * (c >> 3);
                e[u] = wload(w, L::BIAS + 8192 * h + off);
            }
            v[j] = pack2(e[0], e[1]);
        }
        ((u8v*)bias)[b] = v;
        return;
    }
    const int lane = b & 31, kt = (b >> 5) & 3, qt = (b >> 7) & 3, h = b >> 9;
    const int q = 16 * qt + (lane & 15);
    half_t v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const int key = 16 * kt + wm_acc_row_h(i, lane >> 4), r = q & 15, c = key & 15;
        const int off = 512 * (4 * kt + qt) + 64 * (r & 7) + 16 * ((c & 7) >> 1) + 2 * (c & 1) + 4 * (r >> 3) + 8 * (c >> 3);
        v[i] = wload(w, L::BIAS + 8192 * h + off);
    }
    bias[b] = (u4v){pack2(v[0], v[1]), pack2(v[2], v[3]), pack2(v[4], v[5]), pack2(v[6], v[7])};
}

template <class L, bool MERGE = true> __device__ void pwin_prep(const PwinParams& p, u8v* img, u4v* bias)
{
    const int idx = blockIdx.x * 128 + threadIdx.x, j = idx - (L::T_END * 16 + L::NBIAS);
    if (j >= 0)
    {
        if (j < 2 * L::NV)
            pwin_vec_prep<L, MERGE>(p.w, p.w, img, j);
        return;
    }
    pwin_prep_at<L>(p, img, bias, idx);
}
template <class L> __device__ void pwin_prep_nomerge(const PwinParams& p, u8v* img, u4v* bias)
{
    pwin_prep<L, false>(p, img, bias);
}

__device__ __forceinline__ half_t hmin(half_t a, half_t b) { return a < b ? a : b; }
__device__ __forceinline__ half_t hmax(half_t a, half_t b) { return a > b ? a : b; }

__device__ __forceinline__ half_t softmax_e(half_t s)
{
    const half_t hi = __builtin_bit_cast(half_t, (uint16_t)0x4D69), lo = __builtin_bit_cast(half_t, (uint16_t)0xCD69);
    const half_t a = __builtin_bit_cast(half_t, (uint16_t)0x91D5), c = __builtin_bit_cast(half_t, (uint16_t)0xC50D);
    const half_t t = hmax(hmin(s, hi), lo);
    const half_t u = t * t;
    const half_t v = __builtin_fmaf16(u, a, (half_t)1.0f);
    const half_t x = __builtin_fmaf16(t, v, c);
#ifdef PWIN_NATIVE_HEXP
    return __builtin_elementwise_exp2(x);
#else
    return (half_t)__builtin_amdgcn_exp2f((float)x);
#endif
}

__device__ __forceinline__ half_t gelu(half_t x)
{
    const half_t k1 = (half_t)__builtin_bit_cast(float, 0x3ED306EBu), k2 = (half_t)__builtin_bit_cast(float, 0x3DA60DD6u);
    const half_t c = hmin(hmax(x, (half_t)-2.0f), (half_t)2.0f);
    const half_t ac = c < (half_t)0.0f ? -c : c;
    const half_t u = k1 - k2 * ac;
    return x * ((half_t)0.5f + c * u);
}

// gelu of 8 accumulator values as 4 packed f16 pairs (the same f16 operations as gelu(), element-wise)
__device__ __forceinline__ void gelu8_math(const f8v& d, half_t g[8])
{
    const hv2 k1 = (hv2)(half_t)__builtin_bit_cast(float, 0x3ED306EBu), k2 = (hv2)(half_t)__builtin_bit_cast(float, 0x3DA60DD6u);
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const hv2 x = (hv2){(half_t)d[2 * j], (half_t)d[2 * j + 1]};
        const hv2 c = __builtin_elementwise_min(__builtin_elementwise_max(x, (hv2)(half_t)-2.0f), (hv2)(half_t)2.0f);
        const hv2 u = k1 - k2 * __builtin_elementwise_abs(c);
        const hv2 r = x * ((hv2)(half_t)0.5f + c * u);
        g[2 * j] = r[0];
        g[2 * j + 1] = r[1];
    }
}

// Exact GELU table experiment; enabled only by the POS wrappers that initialize it in prep.
#ifdef PWIN_GELU_LUT
#include "pwin_gelu_lut.h"
#else
constexpr int pwin_gelu_prep_items(int n) { return n; }
__device__ __forceinline__ void pwin_gelu_prep() {}
#endif

__device__ __forceinline__ void gelu8(const f8v& d, half_t g[8])
{
#ifdef PWIN_GELU_LUT
    gelu8_lut(d, g);
#else
    gelu8_math(d, g);
#endif
}

// gelu of the 8 accumulator values as the next GEMM's operand. With WM_CVT_ASM the pairs stay packed from the
// conversion through the packed f16 math to the operand exchange (the same f16 operations as gelu8_math).
__device__ __forceinline__ op_t gelu_op(const f8v& d)
{
#if defined(PWIN_GELU_PACKED) && defined(WM_HAVE_FROM_MINE) && !defined(PWIN_GELU_LUT)
    const hv2 k1 = (hv2)(half_t)__builtin_bit_cast(float, 0x3ED306EBu), k2 = (hv2)(half_t)__builtin_bit_cast(float, 0x3DA60DD6u);
    uint32_t mine[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
#ifdef PWIN_GELU_PACKED_C
        const hv2 x = (hv2){(half_t)d[2 * j], (half_t)d[2 * j + 1]}; // C conversions (no asm in the MLP)
#else
        const hv2 x = __builtin_bit_cast(hv2, wm_cvt2(d[2 * j], d[2 * j + 1]));
#endif
        const hv2 c = __builtin_elementwise_min(__builtin_elementwise_max(x, (hv2)(half_t)-2.0f), (hv2)(half_t)2.0f);
        const hv2 u = k1 - k2 * __builtin_elementwise_abs(c);
        mine[j] = __builtin_bit_cast(uint32_t, x * ((hv2)(half_t)0.5f + c * u));
    }
    return wm_op_from_mine(mine);
#else
    half_t g[8];
    gelu8(d, g);
    return operand_from_dt(g);
#endif
}

// ---- accumulator abstraction for the POS layers (pwin_pos.h).
// Exact builds: pacc = f8v and every helper is the operation the code used before (bit-identical).
// PWIN_F16ACC_NATIVE (opt-in approximate variant): accumulators stay in f16 registers through
// the native f16-output WMMA (value i in the low half of dword i); this rounds every k16 step to f16 instead of
// keeping f32 through the chain (harness PSNR ~62 dB vs the exact output), and removes most f32 -> f16 conversions.
#ifdef PWIN_F16ACC_NATIVE
static_assert(D4R_WMMA_LAYOUT == 11, "f16 accumulator variant: gfx11 layout");
typedef h16 pacc;
__device__ __forceinline__ pacc acc_zero() { return (pacc){}; }
template <class L> __device__ __forceinline__ pacc acc_vec_p(const u8v* img, const uint8_t* w, int byte, int v)
{
    const f8v f = acc_vec<L>(img, w, byte, v);
    pacc r = {};
#pragma unroll
    for (int i = 0; i < 8; ++i)
        r[2 * i] = (half_t)f[i];
    return r;
}
__device__ __forceinline__ pacc mma16p(const op_t& a, const op_t& b, pacc c)
{
    return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(__builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b), c, false);
}
// elements 2 j, 2 j + 1 as one packed register (low halves of dwords 2 j and 2 j + 1: one byte permutation)
__device__ __forceinline__ uint32_t acc_pair(const pacc& d, int j)
{
    const u8v u = __builtin_bit_cast(u8v, d);
    return __builtin_amdgcn_perm(u[2 * j + 1], u[2 * j], 0x05040100u);
}
__device__ __forceinline__ void acc_set_pair(pacc& d, int j, hv2 y)
{
    d[4 * j] = y[0];
    d[4 * j + 2] = y[1];
}
__device__ __forceinline__ op_t operand_from_acc_p(const pacc& d)
{
    uint32_t mine[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        mine[j] = acc_pair(d, j);
    return wm_op_from_mine(mine);
}
__device__ __forceinline__ op_t gelu_op_p(const pacc& d)
{
    const hv2 k1 = (hv2)(half_t)__builtin_bit_cast(float, 0x3ED306EBu), k2 = (hv2)(half_t)__builtin_bit_cast(float, 0x3DA60DD6u);
    uint32_t mine[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const hv2 x = __builtin_bit_cast(hv2, acc_pair(d, j));
        const hv2 c = __builtin_elementwise_min(__builtin_elementwise_max(x, (hv2)(half_t)-2.0f), (hv2)(half_t)2.0f);
        const hv2 u = k1 - k2 * __builtin_elementwise_abs(c);
        mine[j] = __builtin_bit_cast(uint32_t, x * ((hv2)(half_t)0.5f + c * u));
    }
    return wm_op_from_mine(mine);
}
#else
typedef f8v pacc;
__device__ __forceinline__ pacc acc_zero() { return splat8(0.0f); }
template <class L> __device__ __forceinline__ pacc acc_vec_p(const u8v* img, const uint8_t* w, int byte, int v) { return acc_vec<L>(img, w, byte, v); }
__device__ __forceinline__ pacc mma16p(const op_t& a, const op_t& b, pacc c) { return mma16(a, b, c); }
__device__ __forceinline__ uint32_t acc_pair(const pacc& d, int j) { return wm_cvt2(d[2 * j], d[2 * j + 1]); }
__device__ __forceinline__ void acc_set_pair(pacc& d, int j, hv2 y)
{
    d[2 * j] = (float)y[0];
    d[2 * j + 1] = (float)y[1];
}
__device__ __forceinline__ op_t operand_from_acc_p(const pacc& d) { return operand_from_f8(d); }
__device__ __forceinline__ op_t gelu_op_p(const pacc& d) { return gelu_op(d); }
#endif


// store the D^T tile of token m as 16 contiguous channels (lanes of one half-wave write)
__device__ __forceinline__ void store_row16(half_t* dst, const u8v& v)
{
    *(u4v*)dst = (u4v){v[0], v[1], v[2], v[3]};
    *(u4v*)(dst + 8) = (u4v){v[4], v[5], v[6], v[7]};
}

#ifdef PWIN_STAMP
// per-block stage timestamps of wave 0 (s_memtime), slot k
__device__ uint64_t g_stamp[16384][16];
#define STAMP(k)                                                                                                   \
    do                                                                                                             \
    {                                                                                                              \
        if (threadIdx.z == 0 && lane_id() == 0)                                                                    \
        {                                                                                                          \
            const int bid = blockIdx.y * gridDim.x + blockIdx.x;                                                   \
            if (bid < 16384)                                                                                       \
                g_stamp[bid][k] = __builtin_readcyclecounter();                                                    \
        }                                                                                                          \
    } while (0)
#elif defined(PWIN_PROF)
#define STAMP(k) PROF(k)
#else
#define STAMP(k) do {} while (0)
#endif
#ifdef PWIN_DBG
__device__ float g_dbg[12][64][160];
#define DBG_ON (blockIdx.x == DBX && blockIdx.y == DBY)
#endif
// Softmax row sum over the 64 keys of this lane's query: this lane's 32 accumulator rows (summed into rs as
// the scores are computed), plus the other half's. The layout shim (gfx12 layout on gfx11) instead redoes the
// sum in gfx11's order (keys 16 kt + 2 i + h, then the other parity), which the f32 result depends on, so
// that its output stays bit-identical to the gfx11 build.
#ifdef D4R_WMMA_SHIM
constexpr bool kShimRowSum = true;
#else
constexpr bool kShimRowSum = false;
#endif
__device__ __forceinline__ float row_sum_total(float rs, const half_t (&e)[4][8])
{
    if constexpr (!kShimRowSum)
        return rs + __builtin_bit_cast(float, other_half(__builtin_bit_cast(uint32_t, rs)));
    // this lane: keys 16 kt + 8 h + j; the other half: keys 16 kt + 8 (1 - h) + j
    const bool hf = wm_half() != 0;
    float s[2] = {0.0f, 0.0f};
#pragma unroll
    for (int kt = 0; kt < 4; ++kt)
    {
        half_t lo[8], hi[8];
#pragma unroll
        for (int j = 0; j < 8; j += 2)
        {
            const uint32_t own = pack2(e[kt][j], e[kt][j + 1]), other = other_half(own);
            const uint32_t l = hf ? other : own, u = hf ? own : other;
            lo[j] = lo16(l), lo[j + 1] = hi16(l), hi[j] = lo16(u), hi[j + 1] = hi16(u);
        }
#pragma unroll
        for (int par = 0; par < 2; ++par)
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const int key = 2 * i + par;
                s[par] += (float)(key < 8 ? lo[key] : hi[key - 8]);
            }
    }
    return s[0] + s[1];
}
#ifdef D4R_ACCURACY
// The reference's softmax row sum as f16 pairs (keys 2p, 2p + 1), reduced in its order (ptx_reduce).
// gfx11 splits each key pair across half-waves; gfx12 assigns eight consecutive keys to each half-wave.
__device__ __forceinline__ half_t ptx_row_sum(const half_t (&e)[4][8])
{
    const bool hf = wm_half() != 0;
    uint32_t pairs[32];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt)
#pragma unroll
        for (int j = 0; j < 8; j += 2)
        {
            const uint32_t own = pack2(e[kt][j], e[kt][j + 1]), other = other_half(own);
#if D4R_WMMA_LAYOUT == 12
            pairs[8 * kt + 4 * int(hf) + j / 2] = own;
            pairs[8 * kt + 4 * int(!hf) + j / 2] = other;
#else
#pragma unroll
            for (int b = 0; b < 2; ++b)
            {
                const half_t mine = b ? hi16(own) : lo16(own), theirs = b ? hi16(other) : lo16(other);
                pairs[8 * kt + j + b] = hf ? pack2(theirs, mine) : pack2(mine, theirs);
            }
#endif
        }
    return ptx_reduce<64, false>([&](int q) { return pairs[q]; });
}
#define PWIN_SOFTMAX_RC(rs, e) half_reciprocal(ptx_row_sum(e))
#else
#define PWIN_SOFTMAX_RC(rs, e) half_reciprocal((half_t)row_sum_total(rs, e))
#endif

// The Swin core for one window: x0 rows of this wave's 16 tokens are in act (f16, C per token).
// On return act holds the block output y. hb: normalised input rows; K/V of one head at a time in kl/vt.
#ifdef PWIN_VT_DIRECT
// V^T operands in the vt buffer: slot r (0..127) at byte 48 r (32 B operand + 16 B pad: 16 lanes reading 16
// consecutive slots with b128 loads hit disjoint LDS banks; an unpadded 32 B pitch conflicts 2-way)
#define PWIN_VT_ROWS 43 /* 43 x LDSR(64) halves = 6192 B >= 128 x 48 B */
static_assert(PWIN_VT_ROWS * LDSR(64) * 2 >= 128 * 48, "vt operand slots");
static_assert(sizeof(op_t) == 32, "gfx11 operand layout");
__device__ __forceinline__ void vt_op_store(half_t* vt, int r, const op_t& v)
{
    if (!wm_half())
    {
        u4v* q = (u4v*)((uint8_t*)vt + 48 * r);
        q[0] = (u4v){v[0], v[1], v[2], v[3]};
        q[1] = (u4v){v[4], v[5], v[6], v[7]};
    }
}
__device__ __forceinline__ op_t vt_op_load(const half_t* vt, int r)
{
    const u4v* q = (const u4v*)((const uint8_t*)vt + 48 * r);
    const u4v a = q[0], b = q[1];
    return (op_t){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
}
#else
#define PWIN_VT_ROWS 32
#endif

struct PwinNoX0
{
    static constexpr bool active = false;
    __device__ void operator()() const {}
};
template <class F, class B> struct PwinX0
{
    static constexpr bool active = true;
    F& f;
    B b;
    __device__ void operator()() const { f(b); }
};

// X0F (PWIN_DEC_RECOMPUTE): act and hb are one buffer, so the residual's x0 is recomputed after attention by x0f()
// into x0b (rows of this layer's tokens, aliasing kl / vt, which are dead by then)
template <class L, class X0F = PwinNoX0>
__device__ void pwin_core(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias,
                          half_t (*act)[LDSR(L::C)], half_t (*hb)[LDSR(L::C)], half_t (*kl)[LDSR(32)], half_t (*vt)[LDSR(64)],
                          const X0F& x0f = X0F(), half_t (*x0b)[LDSR(L::C)] = nullptr)
{
    constexpr int C = L::C, KT = L::KT, H = L::H;
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.z);
    const uint8_t* w = p.w;
    half_t* arow = act[16 * wv + m];
    half_t* hrow = hb[16 * wv + m];

    PROF_INIT;
    STAMP(1);
    // ---- L2 norm: h1 = x0 * (rsqrt(sum x0^2) * gamma1) -> hb
#if defined(PWIN_NORM_OWNER) && !defined(D4R_ACCURACY)
    static_assert(C == 64 && !L::POS, "norm ownership is validated for the general C64 layers");
    // Each of two producer waves normalizes 32 distinct rows. The original four-wave WMMA
    // mapping stays unchanged; both cross-wave barriers are required because input staging
    // was wave-local and later attention reads the producer waves' normalized rows.
    block_sync();
    if (wv < 2)
    {
        const int token = 32*wv + int(l);
        const half_t* normA = act[token];
        half_t* normH = hb[token];
        float ss = 0.0f;
        for (int kt = 0; kt < KT; ++kt)
        {
            const u8v xv = lds_row16(normA + 16 * kt);
#pragma unroll
#if defined(PWIN_PK_ELEM) && !defined(PWIN_PK_NO_SS)
            for (int j = 0; j < 8; ++j)
            {
                const uint32_t xj = xv[j]; // (no __builtin_bit_cast of a vector subscript: clang bug, REWRITE.md)
                const hv2 v = __builtin_bit_cast(hv2, xj), q = v * v;
                ss += (float)q[0];
                ss += (float)q[1];
            }
#else
            for (int c = 0; c < 16; ++c)
            {
                const half_t v = op_get(xv, c);
                ss += (float)(half_t)(v * v);
            }
#endif
        }
        const half_t r16 = PWIN_NORM_R16(ss, C, normA);
        for (int kt = 0; kt < KT; ++kt)
        {
            const u8v xv = lds_row16(normA + 16 * kt);
            u8v hv;
            const u8v gv = lds_row16((const half_t*)(w + L::G1) + 16 * kt); // 16 contiguous gammas
#pragma unroll
            for (int j = 0; j < 8; ++j)
#if defined(PWIN_PK_ELEM) && !defined(PWIN_PK_NO_APPLY)
            {
                const uint32_t xj = xv[j], gj = gv[j];
                hv[j] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(hv2, xj) * ((hv2)r16 * __builtin_bit_cast(hv2, gj)));
            }
#else
                hv[j] = pack2(op_get(xv, 2 * j) * (half_t)(r16 * op_get(gv, 2 * j)),
                              op_get(xv, 2 * j + 1) * (half_t)(r16 * op_get(gv, 2 * j + 1)));
#endif
            store_row16(normH + 16 * kt, hv);
        }
    }
    block_sync();
#else
    float ss = 0.0f;
    for (int kt = 0; kt < KT; ++kt)
    {
        const u8v xv = lds_row16(arow + 16 * kt);
#pragma unroll
#if defined(PWIN_PK_ELEM) && !defined(PWIN_PK_NO_SS)
        for (int j = 0; j < 8; ++j)
        {
            const uint32_t xj = xv[j]; // (no __builtin_bit_cast of a vector subscript: clang bug, REWRITE.md)
            const hv2 v = __builtin_bit_cast(hv2, xj), q = v * v;
            ss += (float)q[0];
            ss += (float)q[1];
        }
#else
        for (int c = 0; c < 16; ++c)
        {
            const half_t v = op_get(xv, c);
            ss += (float)(half_t)(v * v);
        }
#endif
    }
    const half_t r16 = PWIN_NORM_R16(ss, C, arow);
    for (int kt = (int)hf; kt < KT; kt += 2)
    {
        const u8v xv = lds_row16(arow + 16 * kt);
        u8v hv;
        const u8v gv = lds_row16((const half_t*)(w + L::G1) + 16 * kt); // 16 contiguous gammas
#pragma unroll
        for (int j = 0; j < 8; ++j)
#if defined(PWIN_PK_ELEM) && !defined(PWIN_PK_NO_APPLY)
        {
            const uint32_t xj = xv[j], gj = gv[j];
            hv[j] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(hv2, xj) * ((hv2)r16 * __builtin_bit_cast(hv2, gj)));
        }
#else
            hv[j] = pack2(op_get(xv, 2 * j) * (half_t)(r16 * op_get(gv, 2 * j)),
                          op_get(xv, 2 * j + 1) * (half_t)(r16 * op_get(gv, 2 * j + 1)));
#endif
        store_row16(hrow + 16 * kt, hv);
    }
#endif
    __builtin_amdgcn_wave_barrier();
#ifdef PWIN_DBG
    if (DBG_ON && !hf)
        for (int c = 0; c < C; ++c)
            g_dbg[0][16 * wv + m][c] = (float)hrow[c];
#endif

    if constexpr (L::POS)
        block_sync(); // V of all 64 tokens is computed by every wave from hb
    STAMP(2);
    // ---- attention, one head at a time; the output projection accumulates head by head
    f8v acc[KT];
#pragma unroll
    for (int nt = 0; nt < KT; ++nt)
{ acc[nt] = acc_vec<L>(img, w, L::BO + 32 * nt, L::V_BO + nt); }
    for (int h = 0; h < H; ++h)
    {
        // position-only attention: V^T A operands straight from D = h1 Wv (lane = dim, 16 keys per tile)
        op_t vtop[2][4];
        if constexpr (L::POS)
        {
            // each wave computes the V^T operands of its own key tile; shared through LDS, one slot per head
            __shared__ __attribute__((aligned(16))) u8v vsh[H][2][4][16];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                f8v d = splat8(0.0f);
                for (int ks = 0; ks < KT; ++ks)
                    d = mma16(op_lds(&hb[16 * wv + m][16 * ks]), op_imgt(img, L::T_QKV + ((h * 3 + 2) * KT + ks) * 2 + nt), d);
                op_img_store(&vsh[h][nt][0][0], wv * 16 + m, operand_from_f8(d));
            }
            block_sync();
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int kt = 0; kt < 4; ++kt)
                    vtop[nt][kt] = op_img(&vsh[h][nt][0][0], kt * 16 + m);
        }
        op_t pop[4];
        if constexpr (L::POS)
        {
#pragma unroll
            for (int kt = 0; kt < 4; ++kt)
                pop[kt] = op_imgt((const u8v*)bias, (h * 4 + wv) * 4 + kt);
        }
        else
        {
        op_t qop[2];
#ifdef PWIN_BIAS_PREFETCH
        // this head's score bias tiles, loaded before the QKV chains (used right after the next barrier)
        u4v bpre[4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt)
            bpre[kt] = bias[((h * 4 + wv) * 4 + kt) * 32 + l];
#endif
#ifdef PWIN_QKV_SPLIT
        // the six chains (V, K, Q x 2 n-tiles; each still over kt in order) two at a time: 2 live accumulators
#pragma unroll
        for (int j = 2; j >= 0; --j)
        {
            f8v dj[2] = {splat8(0.0f), splat8(0.0f)};
            for (int kt = 0; kt < KT; ++kt)
            {
                const op_t row = op_lds(hrow + 16 * kt);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                {
                    const op_t wt = op_imgt(img, L::T_QKV + ((h * 3 + j) * KT + kt) * 2 + nt);
#ifdef PWIN_VT_DIRECT
                    dj[nt] = j == 2 ? mma16(row, wt, dj[nt]) : mma16(wt, row, dj[nt]);
#else
                    dj[nt] = mma16(wt, row, dj[nt]);
#endif
                }
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                const f8v d = dj[nt];
                if (j == 0)
                    qop[nt] = operand_from_f8(d);
                else if (j == 1)
                    op_store(&kl[16 * wv + m][16 * nt], operand_from_f8(d));
                else
                {
#ifdef PWIN_VT_DIRECT
                    vt_op_store(&vt[0][0], 64 * nt + wv * 16 + m, operand_from_f8(d));
#else
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        vt[16 * nt + wm_acc_row(i)][16 * wv + m] = (half_t)d[i];
#endif
                }
            }
        }
#else
        // six independent chains (Q, K, V x 2 n-tiles), each over kt in order
        f8v dq[6];
#pragma unroll
        for (int u = 0; u < 6; ++u)
            dq[u] = splat8(0.0f);
        for (int kt = 0; kt < KT; ++kt)
        {
            const op_t row = op_lds(hrow + 16 * kt);
#ifdef PWIN_VT_DIRECT
            // V chains with the operands swapped (as in the POS layers): D = V^T of this wave's 16 keys, lane = dim
#pragma unroll
            for (int u = 0; u < 4; ++u)
                dq[u] = mma16(op_imgt(img, L::T_QKV + ((h * 3 + (u >> 1)) * KT + kt) * 2 + (u & 1)), row, dq[u]);
#pragma unroll
            for (int u = 4; u < 6; ++u)
                dq[u] = mma16(row, op_imgt(img, L::T_QKV + ((h * 3 + 2) * KT + kt) * 2 + (u & 1)), dq[u]);
#else
#pragma unroll
            for (int u = 0; u < 6; ++u)
                dq[u] = mma16(op_imgt(img, L::T_QKV + ((h * 3 + (u >> 1)) * KT + kt) * 2 + (u & 1)), row, dq[u]);
#endif
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                const f8v d = dq[2 * j + nt];
                if (j == 0)
                    qop[nt] = operand_from_f8(d);
                else if (j == 1)
                {
                    op_store(&kl[16 * wv + m][16 * nt], operand_from_f8(d));
                }
                else
                {
#ifdef PWIN_VT_DIRECT
                    // V^T operand of key tile wv -> slot nt * 64 + wv * 16 + m of the vt buffer (128 x 48 B)
                    vt_op_store(&vt[0][0], 64 * nt + wv * 16 + m, operand_from_f8(d));
#else
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        vt[16 * nt + wm_acc_row(i)][16 * wv + m] = (half_t)d[i];
#endif
                }
            }
#endif
        block_sync();
        half_t e[4][8];
        float rs = 0.0f;
#pragma unroll
        for (int kt = 0; kt < 4; ++kt)
        {
#ifdef PWIN_BIAS_PREFETCH
            const u4v bi = bpre[kt];
#else
            const u4v bi = bias[((h * 4 + wv) * 4 + kt) * 32 + l];
#endif
            f8v d;
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                d[2 * i] = (float)lo16(bi[i]);
                d[2 * i + 1] = (float)hi16(bi[i]);
            }
#pragma unroll
            for (int ds = 0; ds < 2; ++ds)
                d = mma16(op_lds(&kl[16 * kt + m][16 * ds]), qop[ds], d);
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                e[kt][i] = softmax_e((half_t)d[i]);
                if constexpr (!kShimRowSum)
                    rs += (float)e[kt][i];
            }
        }
        const half_t rc = PWIN_SOFTMAX_RC(rs, e);
#pragma unroll
        for (int kt = 0; kt < 4; ++kt)
        {
            half_t pv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                pv[i] = e[kt][i] * rc;
            pop[kt] = operand_from_dt(pv);
        }
        }
        op_t oop[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
        {
            f8v d = splat8(0.0f);
#pragma unroll
            for (int kt = 0; kt < 4; ++kt)
            {
                if constexpr (L::POS)
                    d = mma16(vtop[nt][kt], pop[kt], d);
                else
                {
#ifdef PWIN_VT_DIRECT
                    d = mma16(vt_op_load(&vt[0][0], 64 * nt + kt * 16 + m), pop[kt], d);
#else
                    d = mma16(op_lds(&vt[16 * nt + m][16 * kt]), pop[kt], d);
#endif
                }
            }
            oop[nt] = operand_from_f8(d);
#ifdef PWIN_DBG
            if (DBG_ON)
                for (int i = 0; i < 8; ++i)
                    g_dbg[2 + h][16 * wv + m][16 * nt + wm_acc_row(i)] = (float)(half_t)d[i];
#endif
        }
#pragma unroll
        for (int nt = 0; nt < KT; ++nt)
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
                acc[nt] = mma16(op_imgt(img, L::T_WO + (2 * h + ks) * KT + nt), oop[ks], acc[nt]);
        if constexpr (!L::POS)
            block_sync(); // kl / vt are reused by the next head
    }

#ifdef PWIN_DBG
    if (DBG_ON)
        for (int nt = 0; nt < KT; ++nt)
            for (int i = 0; i < 8; ++i)
                g_dbg[1][16 * wv + m][16 * nt + wm_acc_row(i)] = acc[nt][i];
#endif
    STAMP(3);
    // ---- residual; MLP init acc = x1 + b2, MLP input m = x1 * g2 (-> hb)
    if constexpr (X0F::active)
    {
        x0f();       // every wave's attention is done (block barrier at the end of the last head)
        block_sync(); // x0 rows of all waves' tokens written
    }
#pragma unroll
    for (int nt = 0; nt < KT; ++nt)
    {
        const op_t xv = X0F::active ? op_lds(&x0b[16 * wv + m][16 * nt]) : op_lds(arow + 16 * nt);
        half_t mv[8], g2[8], b2[8];
        residual_vector<L>(w, 0, nt, g2);
        residual_vector<L>(w, 1, nt, b2);
#if defined(PWIN_PK_ELEM) && !defined(PWIN_PK_NO_RESID)
        uint32_t mine[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const hv2 x1 = __builtin_bit_cast(hv2, wm_cvt2(acc[nt][2 * j], acc[nt][2 * j + 1])) +
                           __builtin_bit_cast(hv2, wm_op_acc_pair(xv, j));
            mine[j] = __builtin_bit_cast(uint32_t, x1 * (hv2){g2[2 * j], g2[2 * j + 1]});
            const hv2 y = x1 + (hv2){b2[2 * j], b2[2 * j + 1]};
            acc[nt][2 * j] = (float)y[0];
            acc[nt][2 * j + 1] = (float)y[1];
        }
        (void)mv;
        op_store(hrow + 16 * nt, wm_op_from_mine(mine));
#else
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const half_t x1 = (half_t)acc[nt][i] + wm_op_acc_elem(xv, i);
            mv[i] = x1 * g2[i];
            acc[nt][i] = (float)(half_t)(x1 + b2[i]);
        }
        op_store(hrow + 16 * nt, operand_from_dt(mv));
#endif
    }
    __builtin_amdgcn_wave_barrier();

    // ---- MLP: C/8 chunks of 32 hidden
    for (int c = 0; c < L::NMLP; ++c)
    {
        op_t gop[2];
#pragma unroll
        for (int hn = 0; hn < 2; ++hn)
        {
            f8v d;
{ d = acc_vec<L>(img, w, L::B1 + 64 * c + 32 * hn, L::V_B1 + 2 * c + hn); }
            for (int kt = 0; kt < KT; ++kt)
                d = mma16(op_imgt(img, L::T_W1 + (c * KT + kt) * 2 + hn), op_lds(hrow + 16 * kt), d);
            gop[hn] = gelu_op(d);
        }
#pragma unroll
        for (int nt = 0; nt < KT; ++nt)
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
                acc[nt] = mma16(op_imgt(img, L::T_W2 + (c * 2 + ks) * KT + nt), gop[ks], acc[nt]);
    }
    STAMP(4);
    // y -> act (rows of this wave)
#pragma unroll
    for (int nt = 0; nt < KT; ++nt)
    {
        op_store(arow + 16 * nt, operand_from_f8(acc[nt]));
    }
    __builtin_amdgcn_wave_barrier();
    PROF(5);
}

// first layer: token embedding x0 = relu(in (CIN) W_e + b_e) in front of the core (core weights at CB)
template <int H_, int C_, int COUT_, bool POS_, int CIN_> struct PwinEmbLayout : PwinLayout<H_, C_, COUT_, POS_>
{
    using B = PwinLayout<H_, C_, COUT_, POS_>;
    static constexpr int CIN = CIN_, KE = CIN / 16, CB = CIN ? 2 * C_ + 2 * CIN * C_ : 0;
    static constexpr int T_EMB = B::T_END, T_END = T_EMB + KE * (C_ / 16);
    // + the embedding bias (C / 16 vectors at the layer's weights pointer)
    static constexpr int V_EMB = B::NV, NV = V_EMB + (CIN ? C_ / 16 : 0);
    static constexpr int VEC0 = T_END * 16, IMG_SLOTS = VEC0 + 2 * NV;
    static constexpr int PREP_ITEMS = T_END * 16 + B::NBIAS + 2 * NV;
    __device__ static const uint8_t* extra_vec(const uint8_t* w0, const uint8_t*, int v) { return w0 + 32 * (v - V_EMB); }
};

template <class L> __device__ void pwin_emb_prep(const PwinParams& p, u8v* img, u4v* bias)
{
    const int idx = blockIdx.x * 128 + threadIdx.x, j = idx - (L::T_END * 16 + L::B::NBIAS);
    if (j >= 0)
    {
        if (j < 2 * L::NV)
            pwin_vec_prep<L, true>(p.w, p.w + L::CB, img, j);
        return;
    }
    if (idx >= L::T_EMB * 16 && idx < L::T_END * 16)
    {
        const int t = (idx >> 4) - L::T_EMB, nt = t % (L::C / 16), ks = t / (L::C / 16);
        img[idx] = img_tile(p.w, 2 * L::C + 512 * ks + 512 * L::KE * nt, idx & 15);
        return;
    }
    PwinParams q = p;
    q.w = p.w + L::CB;
    // core tiles and the bias / P images (indices after T_END are shifted down to the base layout's)
    const int base_idx = idx < L::T_EMB * 16 ? idx : idx - (L::T_END - L::T_EMB) * 16;
    if (idx >= L::T_EMB * 16 && idx < L::T_END * 16)
        return;
    pwin_prep_at<typename L::B>(q, img, bias, base_idx);
}

// encoder: core + full-resolution output + 2x2 patch merge
template <class L>
__device__ void pwin_encoder(const PwinParams& p0, const u8v* __restrict__ img, const u4v* __restrict__ bias)
{
    PwinParams p = p0;
    if constexpr (L::CB != 0)
        p.w = p0.w + L::CB;
    constexpr int C = L::C, COUT = L::COUT;
    __shared__ __attribute__((aligned(16))) half_t act[64][LDSR(C)];
    __shared__ __attribute__((aligned(16))) half_t hb[64][LDSR(C)];
    __shared__ __attribute__((aligned(16))) half_t kl[64][LDSR(32)];
    __shared__ __attribute__((aligned(16))) half_t vt[PWIN_VT_ROWS][LDSR(64)];
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.z), tok = 16 * wv + m, ty = tok >> 3, tx = tok & 7;
    const int bx = blockIdx.x, by = blockIdx.y;

    PROF_INIT;
    STAMP(0);
    // input rows (mirror padded) -> act (first layer: embedding + ReLU)
    {
        const int X = mirror(8 * bx - p.sx + tx, p.W), Y = mirror(8 * by - p.sy + ty, p.H);
        if constexpr (L::CIN != 0)
        {
            const half_t* xin = (const half_t*)p.in + (size_t)(Y * p.W + X) * L::CIN;
            op_t xe[L::KE];
#pragma unroll
            for (int ks = 0; ks < L::KE; ++ks)
                xe[ks] = op_gload(xin + 16 * ks);
#pragma unroll
            for (int nt = 0; nt < C / 16; ++nt)
            {
                f8v d;
                { d = acc_vec<L>(img, p0.w, 32 * nt, L::V_EMB + nt); }
#pragma unroll
                for (int ks = 0; ks < L::KE; ++ks)
                    d = mma16(op_imgt(img, L::T_EMB + ks * (C / 16) + nt), xe[ks], d);
                half_t xo[8];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    xo[i] = hmax((half_t)d[i], (half_t)0.0f);
                op_store(&act[tok][16 * nt], operand_from_dt(xo));
            }
        }
        else
        {
            const half_t* xin = (const half_t*)p.in + (size_t)(Y * p.W + X) * C;
            for (int kt = (int)hf; kt < L::KT; kt += 2)
                store_row16(&act[tok][16 * kt], gload_row16(xin + 16 * kt));
        }
    }
    __builtin_amdgcn_wave_barrier();
    pwin_core<L>(p, img, bias, act, hb, kl, vt);

    STAMP(5);
    // full-resolution output
    const int Y0 = 8 * by - p.sy + ty, X0 = 8 * bx - p.sx + tx;
    if (Y0 >= 0 && Y0 < p.H && X0 >= 0 && X0 < p.W)
    {
        half_t* yout = (half_t*)p.out32 + (size_t)(Y0 * p.W + X0) * C;
        for (int kt = (int)hf; kt < L::KT; kt += 2)
            gstore_row16(yout + 16 * kt, lds_row16(&act[tok][16 * kt]));
    }
    block_sync();

    STAMP(6);
    // patch merge: 16 merged tokens, K = 4 sub-tokens x C, COUT/16 n-tiles over the 4 waves
    const int my = m >> 2, mx = m & 3;
    const int Ym = (8 * by - p.sy) / 2 + my, Xm = (8 * bx - p.sx) / 2 + mx;
    const bool minb = Ym >= 0 && Ym < p.H / 2 && Xm >= 0 && Xm < p.W / 2;
    half_t* mout = (half_t*)p.out24 + (size_t)(Ym * (p.W / 2) + Xm) * COUT;
    const uint8_t* w = p.w;
    for (int vt = wv; vt < L::PMT; vt += 4)
    {
        f8v d;
{ d = acc_vec<L>(img, w, L::PMB + 32 * vt, L::V_PM + vt);

    STAMP(7); }
        for (int ks = 0; ks < C / 4; ++ks)
        {
            const int sub = ks / L::KT, dy = sub >> 1, dx = sub & 1;
            const int src = 8 * (2 * my + dy) + 2 * mx + dx;
            d = mma16(op_imgt(img, L::T_PM + ks * L::PMT + vt), op_lds(&act[src][16 * (ks % L::KT)]), d);
        }
        // allocated column 16 vt + c belongs to original warp g, local column 16 u + c
        const int g = vt / (L::NPA / 16), u = vt % (L::NPA / 16), valid = L::NPW - 16 * u;
        op_gstore(mout + L::NPW * g + 16 * u, operand_from_f8(d), minb, valid);
    }
}

#define PWIN_ENCODER_EMB(NAME, H, C, COUT, POS, CIN)                                                               \
    using NAME##_L = PwinEmbLayout<H, C, COUT, POS, CIN>;                                                                       \
    __device__ u8v g_img[NAME##_L::IMG_SLOTS];                                                                    \
    __device__ u4v g_bias[NAME##_L::NBIAS];                                                                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::PREP_ITEMS + 127) / 128;     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4;                                          \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_offset = 64; /* prep reads only p.w */       \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p)                                    \
    {                                                                                                              \
        pwin_emb_prep<NAME##_L>(p, g_img, g_bias);                                                                     \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) PWIN_VGPR NAME(PwinParams p)                                 \
    {                                                                                                              \
        pwin_encoder<NAME##_L>(p, g_img, g_bias);                                                                  \
    }
#define PWIN_ENCODER(NAME, H, C, COUT) PWIN_ENCODER_EMB(NAME, H, C, COUT, false, 0)
#define PWIN_ENCODER_POS(NAME, H, C, COUT, POS) PWIN_ENCODER_EMB(NAME, H, C, COUT, POS, 0)

// bottleneck (dec5): core only, output at the same resolution to +24
template <class L>
__device__ void pwin_plain(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias)
{
    constexpr int C = L::C;
    __shared__ __attribute__((aligned(16))) half_t act[64][LDSR(C)];
    __shared__ __attribute__((aligned(16))) half_t hb[64][LDSR(C)];
    __shared__ __attribute__((aligned(16))) half_t kl[64][LDSR(32)];
    __shared__ __attribute__((aligned(16))) half_t vt[PWIN_VT_ROWS][LDSR(64)];
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.z), tok = 16 * wv + m, ty = tok >> 3, tx = tok & 7;
    const int bx = blockIdx.x, by = blockIdx.y;
    {
        const int X = mirror(8 * bx - p.sx + tx, p.W), Y = mirror(8 * by - p.sy + ty, p.H);
        const half_t* xin = (const half_t*)p.in + (size_t)(Y * p.W + X) * C;
        for (int kt = (int)hf; kt < L::KT; kt += 2)
            store_row16(&act[tok][16 * kt], gload_row16(xin + 16 * kt));
    }
    __builtin_amdgcn_wave_barrier();
    pwin_core<L>(p, img, bias, act, hb, kl, vt);
    const int Y0 = 8 * by - p.sy + ty, X0 = 8 * bx - p.sx + tx;
    if (Y0 >= 0 && Y0 < p.H && X0 >= 0 && X0 < p.W)
    {
        half_t* yout = (half_t*)p.out24 + (size_t)(Y0 * p.W + X0) * C;
        for (int kt = (int)hf; kt < L::KT; kt += 2)
            gstore_row16(yout + 16 * kt, lds_row16(&act[tok][16 * kt]));
    }
}

#define PWIN_KERNEL(NAME, H, C, COUT, BODY)                                                                        \
    using NAME##_L = PwinLayout<H, C, COUT>;                                                                       \
    __device__ u8v g_img[NAME##_L::IMG_SLOTS];                                                                    \
    __device__ u4v g_bias[NAME##_L::NBIAS];                                                                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::PREP_ITEMS + 127) / 128;     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4;                                          \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_offset = 64; /* prep reads only p.w */       \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p)                                    \
    {                                                                                                              \
        pwin_prep<NAME##_L>(p, g_img, g_bias);                                                                     \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME(PwinParams p)                                           \
    {                                                                                                              \
        BODY<NAME##_L>(p, g_img, g_bias);                                                                          \
    }

// ---------------------------------------------------------------- decoders (patch expand + skip)
template <int H_, int C_, int CL_, bool POS_ = false, int NOUT_ = 0, int NOUTA_ = 0>
struct PwinDecLayout : PwinLayout<H_, C_, 16, POS_>
{
    using B = PwinLayout<H_, C_, 16, POS_>;
    // last layer: output head out = y W_h + b_h (NOUTA computed, NOUT stored); b_h at B::PM, W_h after it
    static constexpr int NOUT = NOUT_, NOUTA = NOUTA_, HB = B::PM, HW = B::PM + 2 * NOUTA_;
    static constexpr int CL = CL_, KL = CL / 16, EXPN = 4 * C_ / 16; // expand: K CL, N 4C
    static constexpr int EXPB = 2 * CL * 4 * C_, CB = EXPB + 8 * C_;  // expand bias, core weights base
    static constexpr int T_EXP = B::T_PM;                              // expand tiles replace the merge tiles
    static constexpr int T_HEAD = T_EXP + KL * EXPN;
    static constexpr int T_END = T_HEAD + B::KT * (NOUTA_ / 16);
    // + the expand biases (EXPN vectors at w0 + EXPB) and the output head bias (NOUTA / 16 at core + HB)
    static constexpr int V_EXP = B::NV, V_HB = V_EXP + EXPN, NV = V_HB + NOUTA_ / 16;
    static constexpr int VEC0 = T_END * 16, IMG_SLOTS = VEC0 + 2 * NV;
    static constexpr int PREP_ITEMS = T_END * 16 + B::NBIAS + 2 * NV;
    __device__ static const uint8_t* extra_vec(const uint8_t* w0, const uint8_t* wc, int v)
    {
        return v < V_HB ? w0 + EXPB + 32 * (v - V_EXP) : wc + HB + 32 * (v - V_HB);
    }
};

template <class L> __device__ void pwin_dec_prep(const PwinParams& p, u8v* img, u4v* bias)
{
    const int idx = blockIdx.x * 128 + threadIdx.x, j = idx - (L::T_END * 16 + L::B::NBIAS);
    if (j >= 0)
    {
        if (j < 2 * L::NV)
            pwin_vec_prep<L, false>(p.w, p.w + L::CB, img, j);
        return;
    }
    if (idx >= L::T_EXP * 16 && idx < L::T_HEAD * 16)
    {
        const int t = (idx >> 4) - L::T_EXP, nt = t % L::EXPN, ks = t / L::EXPN;
        img[idx] = img_tile(p.w, 512 * ks + 32 * L::CL * nt, idx & 15);
        return;
    }
    if (idx >= L::T_HEAD * 16 && idx < L::T_END * 16)
    {
        const int t = (idx >> 4) - L::T_HEAD, nt = t % (L::NOUTA / 16), ks = t / (L::NOUTA / 16);
        img[idx] = img_tile(p.w + L::CB, L::HW + 512 * ks + 512 * L::KT * nt, idx & 15);
        return;
    }
    PwinParams q = p;
    q.w = p.w + L::CB;
    // core tiles lie below T_EXP; bias / P images follow the base layout's (unused) merge tiles
    pwin_prep_at<typename L::B>(q, img, bias, idx < L::T_EXP * 16 ? idx : L::B::T_END * 16 + (idx - L::T_END * 16));
}

template <class L>
__device__ void pwin_decoder(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias)
{
    constexpr int C = L::C;
    __shared__ __attribute__((aligned(16))) half_t act[64][LDSR(C)];
    constexpr int KVR = L::POS ? 1 : 64, KVC = L::POS ? 1 : PWIN_VT_ROWS;
#ifdef PWIN_DEC_RECOMPUTE
    // one buffer for act / hb; kl, vt and (after attention) the recomputed x0 rows share one region
    static_assert(!L::POS, "PWIN_DEC_RECOMPUTE: full-attention decoders");
    constexpr int KLB = KVR * LDSR(32) * 2, VTB = KVC * LDSR(64) * 2, X0B = 64 * LDSR(C) * 2;
    constexpr int KVXB = (KLB + VTB) > X0B ? (KLB + VTB) : X0B;
    __shared__ __attribute__((aligned(16))) uint8_t kvx[KVXB];
    half_t (*hb)[LDSR(C)] = act;
    half_t (*kl)[LDSR(32)] = (half_t (*)[LDSR(32)])kvx;
    half_t (*vt)[LDSR(64)] = (half_t (*)[LDSR(64)])(kvx + KLB);
    half_t (*x0b)[LDSR(C)] = (half_t (*)[LDSR(C)])kvx;
#else
    __shared__ __attribute__((aligned(16))) half_t hb[64][LDSR(C)];
    __shared__ __attribute__((aligned(16))) half_t kl[KVR][LDSR(32)];
    __shared__ __attribute__((aligned(16))) half_t vt[KVC][LDSR(64)];
#endif
#ifdef PWIN_LDS_PAD
    // occupancy-sensitivity probe only: PWIN_LDS_PAD bytes of unused LDS per block (one dword written)
    __shared__ uint32_t lds_pad[PWIN_LDS_PAD / 4];
    if (threadIdx.x == 0 && threadIdx.z == 0)
        ((volatile uint32_t*)lds_pad)[0] = 0u;
#endif
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.z), bx = blockIdx.x, by = blockIdx.y;
    PROF_INIT;

    // ---- patch expand: wave wv produces sub-token q = wv (dy, dx) of all 16 low-resolution tokens -> dst rows
    auto expand = [&](half_t (*dst)[LDSR(C)]) {
        const int W2 = p.W / 2, H2 = p.H / 2;
        const int ly = m >> 2, lx = m & 3;
        const int LX = mirror((8 * bx - p.sx) / 2 + lx, W2), LY = mirror((8 * by - p.sy) / 2 + ly, H2);
        const half_t* xl = (const half_t*)p.in + (size_t)(LY * W2 + LX) * L::CL;
        const int dy = wv >> 1, dx = wv & 1, t = 8 * (2 * ly + dy) + 2 * lx + dx;
        const int SX = mirror(8 * bx - p.sx + (t & 7), p.W), SY = mirror(8 * by - p.sy + (t >> 3), p.H);
        const half_t* sk = (const half_t*)p.skip + (size_t)(SY * p.W + SX) * C;
        for (int nt = 0; nt < C / 16; ++nt)
        {
            const int gnt = wv * (C / 16) + nt; // expand output column tile
            f8v d;
{ d = acc_vec<L>(img, p.w, L::EXPB + 32 * gnt, L::V_EXP + gnt); }
            for (int ks = 0; ks < L::KL; ++ks)
                d = mma16(op_imgt(img, L::T_EXP + ks * L::EXPN + gnt), op_gload(xl + 16 * ks), d);
            const op_t sv = op_gload(sk + 16 * nt);
#if defined(PWIN_PK_ELEM) && !defined(PWIN_PK_NO_EXPAND)
            uint32_t mine[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                mine[j] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(hv2, wm_cvt2(d[2 * j], d[2 * j + 1])) +
                                                           __builtin_bit_cast(hv2, wm_op_acc_pair(sv, j)));
            op_store(&dst[t][16 * nt], wm_op_from_mine(mine));
#else
            half_t xo[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                xo[i] = (half_t)d[i] + wm_op_acc_elem(sv, i);
            op_store(&dst[t][16 * nt], operand_from_dt(xo));
#endif
        }
    };
    expand(act);
    block_sync();
    PROF(8);
    PwinParams q = p;
    q.w = p.w + L::CB;
#ifdef PWIN_DEC_RECOMPUTE
    pwin_core<L>(q, img, bias, act, hb, kl, vt, PwinX0<decltype(expand), half_t (*)[LDSR(C)]>{expand, x0b}, x0b);
#else
    pwin_core<L>(q, img, bias, act, hb, kl, vt);
#endif
    PROF(9);

    const int tok = 16 * wv + m, Y0 = 8 * by - p.sy + (tok >> 3), X0 = 8 * bx - p.sx + (tok & 7);
    const bool inb = Y0 >= 0 && Y0 < p.H && X0 >= 0 && X0 < p.W;
    if constexpr (L::NOUTA != 0)
    {
        half_t* hout = (half_t*)p.out24 + (size_t)(Y0 * p.W + X0) * L::NOUT;
#pragma unroll
        for (int nt = 0; nt < L::NOUTA / 16; ++nt)
        {
            f8v d;
{ d = acc_vec<L>(img, q.w, L::HB + 32 * nt, L::V_HB + nt); }
#pragma unroll
            for (int kt = 0; kt < L::KT; ++kt)
                d = mma16(op_imgt(img, L::T_HEAD + kt * (L::NOUTA / 16) + nt), op_lds(&act[tok][16 * kt]), d);
            op_gstore(hout + 16 * nt, operand_from_f8(d), inb, L::NOUT - 16 * nt);
        }
    }
    else if (inb)
    {
        half_t* yout = (half_t*)p.out24 + (size_t)(Y0 * p.W + X0) * C;
        for (int kt = (int)hf; kt < L::KT; kt += 2)
            gstore_row16(yout + 16 * kt, lds_row16(&act[tok][16 * kt]));
    }
    PROF(10);
}

#define PWIN_DECODER_EX(NAME, H, C, CL, POS, NOUT, NOUTA)                                                          \
    using NAME##_L = PwinDecLayout<H, C, CL, POS, NOUT, NOUTA>;                                                                      \
    __device__ u8v g_img[NAME##_L::IMG_SLOTS];                                                                    \
    __device__ u4v g_bias[NAME##_L::NBIAS];                                                                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::PREP_ITEMS + 127) / 128;     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4;                                          \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_offset = 64; /* prep reads only p.w */       \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p)                                    \
    {                                                                                                              \
        pwin_dec_prep<NAME##_L>(p, g_img, g_bias);                                                                 \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) PWIN_VGPR NAME(PwinParams p)                                 \
    {                                                                                                              \
        pwin_decoder<NAME##_L>(p, g_img, g_bias);                                                                  \
    }

#define PWIN_DECODER(NAME, H, C, CL) PWIN_DECODER_EX(NAME, H, C, CL, false, 0, 0)
