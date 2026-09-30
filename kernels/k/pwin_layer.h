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
    static constexpr int PREP_ITEMS = T_END * 16 + NBIAS;

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

template <class L> __device__ void pwin_prep_at(const PwinParams& p, u8v* img, u4v* bias, int idx)
{
    const uint8_t* w = p.w;
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

template <class L> __device__ void pwin_prep(const PwinParams& p, u8v* img, u4v* bias)
{
    pwin_prep_at<L>(p, img, bias, blockIdx.x * 128 + threadIdx.x);
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
    return (half_t)__builtin_amdgcn_exp2f((float)x);
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
__device__ __forceinline__ void gelu8(const f8v& d, half_t g[8])
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

// The Swin core for one window: x0 rows of this wave's 16 tokens are in act (f16, C per token).
// On return act holds the block output y. hb: normalised input rows; K/V of one head at a time in kl/vt.
template <class L>
__device__ void pwin_core(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias,
                          half_t (*act)[L::C], half_t (*hb)[L::C], half_t (*kl)[32], half_t (*vt)[64])
{
    constexpr int C = L::C, KT = L::KT, H = L::H;
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z;
    const uint8_t* w = p.w;
    half_t* arow = act[16 * wv + m];
    half_t* hrow = hb[16 * wv + m];

    STAMP(1);
    // ---- L2 norm: h1 = x0 * (rsqrt(sum x0^2) * gamma1) -> hb
    float ss = 0.0f;
    for (int kt = 0; kt < KT; ++kt)
    {
        const u8v xv = lds_row16(arow + 16 * kt);
#pragma unroll
        for (int c = 0; c < 16; ++c)
        {
            const half_t v = op_get(xv, c);
            ss += (float)(half_t)(v * v);
        }
    }
    const half_t r16 = (half_t)__builtin_amdgcn_rsqf((float)(half_t)ss);
    for (int kt = (int)hf; kt < KT; kt += 2)
    {
        const u8v xv = lds_row16(arow + 16 * kt);
        u8v hv;
        const u8v gv = lds_row16((const half_t*)(w + L::G1) + 16 * kt); // 16 contiguous gammas
#pragma unroll
        for (int j = 0; j < 8; ++j)
            hv[j] = pack2(op_get(xv, 2 * j) * (half_t)(r16 * op_get(gv, 2 * j)),
                          op_get(xv, 2 * j + 1) * (half_t)(r16 * op_get(gv, 2 * j + 1)));
        store_row16(hrow + 16 * kt, hv);
    }
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
{

    half_t vv_[8];

    dvec8(w, L::BO + 32 * nt, vv_);

#pragma unroll

    for (int i = 0; i < 8; ++i)

        acc[nt][i] = (float)vv_[i];

}
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
                    d = mma16(op_lds(&hb[16 * wv + m][16 * ks]), op_img(img, (L::T_QKV + ((h * 3 + 2) * KT + ks) * 2 + nt) * 16 + m), d);
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
                pop[kt] = op_img((const u8v*)bias, ((h * 4 + wv) * 4 + kt) * 16 + m);
        }
        else
        {
        op_t qop[2];
        // six independent chains (Q, K, V x 2 n-tiles), each over kt in order
        f8v dq[6];
#pragma unroll
        for (int u = 0; u < 6; ++u)
            dq[u] = splat8(0.0f);
        for (int kt = 0; kt < KT; ++kt)
        {
            const op_t row = op_lds(hrow + 16 * kt);
#pragma unroll
            for (int u = 0; u < 6; ++u)
                dq[u] = mma16(op_img(img, (L::T_QKV + ((h * 3 + (u >> 1)) * KT + kt) * 2 + (u & 1)) * 16 + m), row, dq[u]);
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
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        vt[16 * nt + wm_acc_row(i)][16 * wv + m] = (half_t)d[i];
                }
            }
        block_sync();
        half_t e[4][8];
        float rs = 0.0f;
#pragma unroll
        for (int kt = 0; kt < 4; ++kt)
        {
            const u4v bi = bias[((h * 4 + wv) * 4 + kt) * 32 + l];
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
        rs = row_sum_total(rs, e);
        const half_t rc = (half_t)__builtin_amdgcn_rcpf((float)(half_t)rs);
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
                    d = mma16(op_lds(&vt[16 * nt + m][16 * kt]), pop[kt], d);
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
                acc[nt] = mma16(op_img(img, (L::T_WO + (2 * h + ks) * KT + nt) * 16 + m), oop[ks], acc[nt]);
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
#pragma unroll
    for (int nt = 0; nt < KT; ++nt)
    {
        const op_t xv = op_lds(arow + 16 * nt);
        half_t mv[8], g2[8], b2[8];
        dvec8(w, L::G2 + 32 * nt, g2);
        dvec8(w, L::B2 + 32 * nt, b2);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const half_t x1 = (half_t)acc[nt][i] + wm_op_acc_elem(xv, i);
            mv[i] = x1 * g2[i];
            acc[nt][i] = (float)(half_t)(x1 + b2[i]);
        }
        op_store(hrow + 16 * nt, operand_from_dt(mv));
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
{

    half_t vv_[8];

    dvec8(w, L::B1 + 64 * c + 32 * hn, vv_);

#pragma unroll

    for (int i = 0; i < 8; ++i)

        d[i] = (float)vv_[i];

}
            for (int kt = 0; kt < KT; ++kt)
                d = mma16(op_img(img, (L::T_W1 + (c * KT + kt) * 2 + hn) * 16 + m), op_lds(hrow + 16 * kt), d);
            half_t g[8];
            gelu8(d, g);
            gop[hn] = operand_from_dt(g);
        }
#pragma unroll
        for (int nt = 0; nt < KT; ++nt)
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
                acc[nt] = mma16(op_img(img, (L::T_W2 + (c * 2 + ks) * KT + nt) * 16 + m), gop[ks], acc[nt]);
    }
    STAMP(4);
    // y -> act (rows of this wave)
#pragma unroll
    for (int nt = 0; nt < KT; ++nt)
    {
        op_store(arow + 16 * nt, operand_from_f8(acc[nt]));
    }
    __builtin_amdgcn_wave_barrier();
}

// first layer: token embedding x0 = relu(in (CIN) W_e + b_e) in front of the core (core weights at CB)
template <int H_, int C_, int COUT_, bool POS_, int CIN_> struct PwinEmbLayout : PwinLayout<H_, C_, COUT_, POS_>
{
    using B = PwinLayout<H_, C_, COUT_, POS_>;
    static constexpr int CIN = CIN_, KE = CIN / 16, CB = CIN ? 2 * C_ + 2 * CIN * C_ : 0;
    static constexpr int T_EMB = B::T_END, T_END = T_EMB + KE * (C_ / 16);
    static constexpr int PREP_ITEMS = T_END * 16 + B::NBIAS;
};

template <class L> __device__ void pwin_emb_prep(const PwinParams& p, u8v* img, u4v* bias)
{
    const int idx = blockIdx.x * 128 + threadIdx.x;
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
    __shared__ __attribute__((aligned(16))) half_t act[64][C];
    __shared__ __attribute__((aligned(16))) half_t hb[64][C];
    __shared__ __attribute__((aligned(16))) half_t kl[64][32];
    __shared__ __attribute__((aligned(16))) half_t vt[32][64];
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z, tok = 16 * wv + m, ty = tok >> 3, tx = tok & 7;
    const int bx = blockIdx.x, by = blockIdx.y;

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
                {
                    half_t vv_[8];
                    dvec8(p0.w, 32 * nt, vv_);
#pragma unroll
                    for (int i = 0; i < 8; ++i)
                        d[i] = (float)vv_[i];
                }
#pragma unroll
                for (int ks = 0; ks < L::KE; ++ks)
                    d = mma16(op_img(img, (L::T_EMB + ks * (C / 16) + nt) * 16 + m), xe[ks], d);
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
{

    half_t vv_[8];

    dvec8(w, L::PMB + 32 * vt, vv_);

#pragma unroll

    for (int i = 0; i < 8; ++i)

        d[i] = (float)vv_[i];

    STAMP(7);
}
        for (int ks = 0; ks < C / 4; ++ks)
        {
            const int sub = ks / L::KT, dy = sub >> 1, dx = sub & 1;
            const int src = 8 * (2 * my + dy) + 2 * mx + dx;
            d = mma16(op_img(img, (L::T_PM + ks * L::PMT + vt) * 16 + m), op_lds(&act[src][16 * (ks % L::KT)]), d);
        }
        // allocated column 16 vt + c belongs to original warp g, local column 16 u + c
        const int g = vt / (L::NPA / 16), u = vt % (L::NPA / 16), valid = L::NPW - 16 * u;
        op_gstore(mout + L::NPW * g + 16 * u, operand_from_f8(d), minb, valid);
    }
}

#define PWIN_ENCODER_EMB(NAME, H, C, COUT, POS, CIN)                                                               \
    using NAME##_L = PwinEmbLayout<H, C, COUT, POS, CIN>;                                                                       \
    __device__ u8v g_img[NAME##_L::T_END * 16];                                                                    \
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
    __shared__ __attribute__((aligned(16))) half_t act[64][C];
    __shared__ __attribute__((aligned(16))) half_t hb[64][C];
    __shared__ __attribute__((aligned(16))) half_t kl[64][32];
    __shared__ __attribute__((aligned(16))) half_t vt[32][64];
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z, tok = 16 * wv + m, ty = tok >> 3, tx = tok & 7;
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
    __device__ u8v g_img[NAME##_L::T_END * 16];                                                                    \
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
    static constexpr int PREP_ITEMS = T_END * 16 + B::NBIAS;
};

template <class L> __device__ void pwin_dec_prep(const PwinParams& p, u8v* img, u4v* bias)
{
    const int idx = blockIdx.x * 128 + threadIdx.x;
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
    __shared__ __attribute__((aligned(16))) half_t act[64][C];
    __shared__ __attribute__((aligned(16))) half_t hb[64][C];
    constexpr int KVR = L::POS ? 1 : 64, KVC = L::POS ? 1 : 32;
    __shared__ __attribute__((aligned(16))) half_t kl[KVR][32];
    __shared__ __attribute__((aligned(16))) half_t vt[KVC][64];
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;

    // ---- patch expand: wave wv produces sub-token q = wv (dy, dx) of all 16 low-resolution tokens
    {
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
{

    half_t vv_[8];

    dvec8(p.w, L::EXPB + 32 * gnt, vv_);

#pragma unroll

    for (int i = 0; i < 8; ++i)

        d[i] = (float)vv_[i];

}
            for (int ks = 0; ks < L::KL; ++ks)
                d = mma16(op_img(img, (L::T_EXP + ks * L::EXPN + gnt) * 16 + m), op_gload(xl + 16 * ks), d);
            const op_t sv = op_gload(sk + 16 * nt);
            half_t xo[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                xo[i] = (half_t)d[i] + wm_op_acc_elem(sv, i);
            op_store(&act[t][16 * nt], operand_from_dt(xo));
        }
    }
    block_sync();
    PwinParams q = p;
    q.w = p.w + L::CB;
    pwin_core<L>(q, img, bias, act, hb, kl, vt);

    const int tok = 16 * wv + m, Y0 = 8 * by - p.sy + (tok >> 3), X0 = 8 * bx - p.sx + (tok & 7);
    const bool inb = Y0 >= 0 && Y0 < p.H && X0 >= 0 && X0 < p.W;
    if constexpr (L::NOUTA != 0)
    {
        half_t* hout = (half_t*)p.out24 + (size_t)(Y0 * p.W + X0) * L::NOUT;
#pragma unroll
        for (int nt = 0; nt < L::NOUTA / 16; ++nt)
        {
            f8v d;
{

    half_t vv_[8];

    dvec8(q.w, L::HB + 32 * nt, vv_);

#pragma unroll

    for (int i = 0; i < 8; ++i)

        d[i] = (float)vv_[i];

}
#pragma unroll
            for (int kt = 0; kt < L::KT; ++kt)
                d = mma16(op_img(img, (L::T_HEAD + kt * (L::NOUTA / 16) + nt) * 16 + m), op_lds(&act[tok][16 * kt]), d);
            op_gstore(hout + 16 * nt, operand_from_f8(d), inb, L::NOUT - 16 * nt);
        }
    }
    else if (inb)
    {
        half_t* yout = (half_t*)p.out24 + (size_t)(Y0 * p.W + X0) * C;
        for (int kt = (int)hf; kt < L::KT; kt += 2)
            gstore_row16(yout + 16 * kt, lds_row16(&act[tok][16 * kt]));
    }
}

#define PWIN_DECODER_EX(NAME, H, C, CL, POS, NOUT, NOUTA)                                                          \
    using NAME##_L = PwinDecLayout<H, C, CL, POS, NOUT, NOUTA>;                                                                      \
    __device__ u8v g_img[NAME##_L::T_END * 16];                                                                    \
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
