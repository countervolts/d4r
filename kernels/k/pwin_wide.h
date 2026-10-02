// Wide variant of the DltssPaddedWinLayer template for the deep, latency-bound levels (few windows):
// 4 * NG waves per window. Wave wv = threadIdx.z works on token tile t = wv & 3 and channel group
// g = wv >> 2: heads g, g + NG, ... (one head per group and round), output tiles nt = g + NG j of the
// output projection and MLP, and MLP hidden chunks c = NG rr + g. Every output element goes through
// the same f16 operations in the same order as pwin_layer.h (bit-identical results).
//
// Shared memory: hb (64 x C) = normalised input, then MLP input, then the block output y; R = union of
// x0 (64 x C), the K / V images of the NG heads of an attention round, their outputs O, and the double
// buffered MLP hidden chunks of a round.
#pragma once
#include "pwin_layer.h"

template <class L, int NG> struct WideDims
{
    // NA: heads per attention round (their K / V images share R)
    static constexpr int NA = NG < 4 ? NG : 4;
    static constexpr int C = L::C, KT = L::KT, H = L::H, NTL = (KT + NG - 1) / NG, ROUNDS = (H + NA - 1) / NA;
    static constexpr int X0_BYTES = 64 * C * 2, KV_BYTES = NA * 2 * 64 * 32 * 2;
    // MLP hidden chunks: one per group and round, double-buffered while that fits
    static constexpr int NB = NG <= 4 ? 2 : 1, MROUNDS = (L::NMLP + NG - 1) / NG, HID_BYTES = NB * NG * 64 * 32 * 2;
    static constexpr int R_BYTES = X0_BYTES > KV_BYTES ? (X0_BYTES > HID_BYTES ? X0_BYTES : HID_BYTES)
                                                       : (KV_BYTES > HID_BYTES ? KV_BYTES : HID_BYTES);
    static_assert(!L::POS, "position-only attention layers use pwin_pos.h");
};

// Core for one window. On entry R holds x0 (64 x C); on return hb holds the block output y.
template <class L, int NG>
__device__ void wide_core(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias,
                          half_t (*hb)[L::C], uint8_t* R)
{
    using D = WideDims<L, NG>;
    constexpr int C = L::C, KT = L::KT, H = L::H, NTL = D::NTL;
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z, t = wv & 3, g = wv >> 2, tok = 16 * t + m;
    const uint8_t* w = p.w;
    half_t(*act)[C] = (half_t(*)[C])R;
    const half_t* arow = act[tok];
    half_t* hrow = hb[tok];

    // ---- L2 norm: h1 = x0 * (rsqrt(sum x0^2) * gamma1) -> hb (all groups of a token tile compute the sum)
    const half_t ss = l2_sum<C>(arow);
    const half_t r16 = (half_t)__builtin_amdgcn_rsqf((float)(half_t)ss);
    for (int kt = 2 * g + (int)hf; kt < KT; kt += 2 * NG)
    {
        const u8v xv = lds_row16(arow + 16 * kt);
        u8v hv;
        const u8v gv = lds_row16((const half_t*)(w + L::G1) + 16 * kt);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            hv[j] = pack2(op_get(xv, 2 * j) * (half_t)(r16 * op_get(gv, 2 * j)),
                          op_get(xv, 2 * j + 1) * (half_t)(r16 * op_get(gv, 2 * j + 1)));
        store_row16(hrow + 16 * kt, hv);
    }
    // residual input of the owned output tiles (D^T order: channel 16 nt + wm_acc_row(i))
    half_t x0[NTL][8];
#pragma unroll
    for (int j = 0; j < NTL; ++j)
    {
        const int nt = g + NG * j;
        if (nt < KT)
        {
            const op_t xv = op_lds(arow + 16 * nt);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                x0[j][i] = wm_op_acc_elem(xv, i);
        }
    }
    block_sync(); // hb complete, R free

    // ---- attention: NG heads per round; the output projection accumulates head by head
    acc8v acc[NTL];
#pragma unroll
    for (int j = 0; j < NTL; ++j)
    {
        const int nt = g + NG * j;
        if (nt < KT)
        {
            half_t vv[8];
            dvec8(w, L::BO + 32 * nt, vv);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[j][i] = (float)vv[i];
        }
    }
    half_t(*kl)[64][32] = (half_t(*)[64][32])R;                   // [group][token][dim]
    constexpr int NA = D::NA;
    half_t(*vt)[32][64] = (half_t(*)[32][64])(R + NA * 64 * 32 * 2); // [group][dim][token]
    half_t(*ob)[64][32] = (half_t(*)[64][32])R;                   // [group][token][dim] (after K / V)
    for (int r = 0; r < D::ROUNDS; ++r)
    {
        const int h = g < NA ? r * NA + g : H;
        op_t qop[2], oop[2];
        if (h < H)
        {
            // six independent chains (Q, K, V x 2 n-tiles), each over kt in order
            acc8v dq[6];
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
                    const acc8v d = dq[2 * j + nt];
                    if (j == 0)
                        qop[nt] = operand_from_f8(d);
                    else if (j == 1)
                    {
                        op_store(&kl[g][tok][16 * nt], operand_from_f8(d));
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 8; ++i)
                            vt[g][16 * nt + wm_acc_row(i)][tok] = (half_t)d[i];
                    }
                }
        }
        block_sync();
        if (h < H)
        {
            half_t e[4][8];
            float rs = 0.0f;
#pragma unroll
            for (int kt = 0; kt < 4; ++kt)
            {
                const u4v bi = bias[((h * 4 + t) * 4 + kt) * 32 + l];
                acc8v d;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    d[2 * i] = (float)lo16(bi[i]);
                    d[2 * i + 1] = (float)hi16(bi[i]);
                }
#pragma unroll
                for (int ds = 0; ds < 2; ++ds)
                    d = mma16(op_lds(&kl[g][16 * kt + m][16 * ds]), qop[ds], d);
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
            op_t pop[4];
#pragma unroll
            for (int kt = 0; kt < 4; ++kt)
            {
                half_t pv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    pv[i] = e[kt][i] * rc;
                pop[kt] = operand_from_dt(pv);
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                acc8v d = splat8(0.0f);
#pragma unroll
                for (int kt = 0; kt < 4; ++kt)
                    d = mma16(op_lds(&vt[g][16 * nt + m][16 * kt]), pop[kt], d);
                oop[nt] = operand_from_f8(d);
            }
        }
        block_sync(); // K / V of the round consumed
        if (h < H)
        {
            op_store(&ob[g][tok][0], oop[0]);
            op_store(&ob[g][tok][16], oop[1]);
        }
        block_sync();
        for (int jh = 0; jh < NA; ++jh)
        {
            const int hh = r * NA + jh;
            if (hh >= H)
                break;
            const op_t o0 = op_lds(&ob[jh][tok][0]), o1 = op_lds(&ob[jh][tok][16]);
#pragma unroll
            for (int j = 0; j < NTL; ++j)
            {
                const int nt = g + NG * j;
                if (nt < KT)
                {
                    acc[j] = mma16(op_img(img, (L::T_WO + (2 * hh) * KT + nt) * 16 + m), o0, acc[j]);
                    acc[j] = mma16(op_img(img, (L::T_WO + (2 * hh + 1) * KT + nt) * 16 + m), o1, acc[j]);
                }
            }
        }
        if (r + 1 < D::ROUNDS)
            block_sync(); // ob is overwritten by the next round's K / V
    }

    // ---- residual; MLP init acc = x1 + b2, MLP input m = x1 * g2 (-> hb)
#pragma unroll
    for (int j = 0; j < NTL; ++j)
    {
        const int nt = g + NG * j;
        if (nt < KT)
        {
            half_t mv[8], g2[8], b2[8];
            dvec8(w, L::G2 + 32 * nt, g2);
            dvec8(w, L::B2 + 32 * nt, b2);
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const half_t x1 = (half_t)acc[j][i] + x0[j][i];
                mv[i] = x1 * g2[i];
                acc[j][i] = (float)(half_t)(x1 + b2[i]);
            }
            op_store(hrow + 16 * nt, operand_from_dt(mv));
        }
    }
    block_sync();

    // ---- MLP: rounds of NG chunks of 32 hidden (double-buffered in R)
    half_t(*hid)[NG][64][32] = (half_t(*)[NG][64][32])R;
    for (int rr = 0; rr < D::MROUNDS; ++rr)
    {
        const int c = rr * NG + g;
#pragma unroll
        for (int hn = 0; hn < 2; ++hn)
        {
            if (c >= L::NMLP)
                break;
            acc8v d;
            {
                half_t vv[8];
                dvec8(w, L::B1 + 64 * c + 32 * hn, vv);
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    d[i] = (float)vv[i];
            }
            for (int kt = 0; kt < KT; ++kt)
                d = mma16(op_img(img, (L::T_W1 + (c * KT + kt) * 2 + hn) * 16 + m), op_lds(hrow + 16 * kt), d);
            half_t gg[8];
            gelu8(d, gg);
            op_store(&hid[rr % D::NB][g][tok][16 * hn], operand_from_dt(gg));
        }
        block_sync();
        for (int jc = 0; jc < NG; ++jc)
        {
            const int cc = rr * NG + jc;
            if (cc >= L::NMLP)
                break;
            const op_t h0 = op_lds(&hid[rr % D::NB][jc][tok][0]), h1 = op_lds(&hid[rr % D::NB][jc][tok][16]);
#pragma unroll
            for (int j = 0; j < NTL; ++j)
            {
                const int nt = g + NG * j;
                if (nt < KT)
                {
                    acc[j] = mma16(op_img(img, (L::T_W2 + (cc * 2) * KT + nt) * 16 + m), h0, acc[j]);
                    acc[j] = mma16(op_img(img, (L::T_W2 + (cc * 2 + 1) * KT + nt) * 16 + m), h1, acc[j]);
                }
            }
        }
        if (D::NB == 1 && rr + 1 < D::MROUNDS)
            block_sync();
    }
    // y -> hb (the last reads of hb were the W1 operands before the final barrier)
#pragma unroll
    for (int j = 0; j < NTL; ++j)
    {
        const int nt = g + NG * j;
        if (nt < KT)
        {
            op_store(hrow + 16 * nt, operand_from_f8(acc[j]));
        }
    }
    block_sync();
}

// x0 rows of a window (mirror padded) -> R
template <class L, int NG> __device__ void wide_load_x0(const PwinParams& p, uint8_t* R)
{
    constexpr int C = L::C;
    half_t(*act)[C] = (half_t(*)[C])R;
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z, t = wv & 3, g = wv >> 2, tok = 16 * t + m;
    const int X = mirror(8 * (int)blockIdx.x - p.sx + (tok & 7), p.W), Y = mirror(8 * (int)blockIdx.y - p.sy + (tok >> 3), p.H);
    const half_t* xin = (const half_t*)p.in + (size_t)(Y * p.W + X) * C;
    for (int kt = 2 * g + (int)hf; kt < L::KT; kt += 2 * NG)
        store_row16(&act[tok][16 * kt], gload_row16(xin + 16 * kt));
}

// y rows of the window's own tokens -> dst (C per token)
template <class L, int NG> __device__ void wide_store_y(const PwinParams& p, half_t (*hb)[L::C], uint8_t* dst)
{
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z, t = wv & 3, g = wv >> 2, tok = 16 * t + m;
    const int Y0 = 8 * (int)blockIdx.y - p.sy + (tok >> 3), X0 = 8 * (int)blockIdx.x - p.sx + (tok & 7);
    if (Y0 >= 0 && Y0 < p.H && X0 >= 0 && X0 < p.W)
    {
        half_t* yout = (half_t*)dst + (size_t)(Y0 * p.W + X0) * L::C;
        for (int kt = 2 * g + (int)hf; kt < L::KT; kt += 2 * NG)
            gstore_row16(yout + 16 * kt, lds_row16(&hb[tok][16 * kt]));
    }
}

template <class L, int NG>
__device__ void wide_encoder(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias)
{
    constexpr int C = L::C, COUT = L::COUT;
    __shared__ __attribute__((aligned(16))) half_t hb[64][C];
    __shared__ __attribute__((aligned(16))) uint8_t R[WideDims<L, NG>::R_BYTES];
    wide_load_x0<L, NG>(p, R);
    block_sync();
    wide_core<L, NG>(p, img, bias, hb, R);
    wide_store_y<L, NG>(p, hb, p.out32);

    // patch merge: 16 merged tokens, K = 4 sub-tokens x C, the allocated n-tiles over the waves
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z;
    const int my = m >> 2, mx = m & 3;
    const int Ym = (8 * (int)blockIdx.y - p.sy) / 2 + my, Xm = (8 * (int)blockIdx.x - p.sx) / 2 + mx;
    const bool minb = Ym >= 0 && Ym < p.H / 2 && Xm >= 0 && Xm < p.W / 2;
    half_t* mout = (half_t*)p.out24 + (size_t)(Ym * (p.W / 2) + Xm) * COUT;
    const uint8_t* w = p.w;
    for (int vt = wv; vt < L::PMT; vt += 4 * NG)
    {
        acc8v d;
        {
            half_t vv[8];
            dvec8(w, L::PMB + 32 * vt, vv);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                d[i] = (float)vv[i];
        }
        for (int ks = 0; ks < C / 4; ++ks)
        {
            const int sub = ks / L::KT, dy = sub >> 1, dx = sub & 1;
            const int src = 8 * (2 * my + dy) + 2 * mx + dx;
            d = mma16(op_img(img, (L::T_PM + ks * L::PMT + vt) * 16 + m), op_lds(&hb[src][16 * (ks % L::KT)]), d);
        }
        const int gw = vt / (L::NPA / 16), u = vt % (L::NPA / 16), valid = L::NPW - 16 * u;
        op_gstore(mout + L::NPW * gw + 16 * u, operand_from_f8(d), minb, valid);
    }
}

// bottleneck (dec5): core only, output at the same resolution to +24
template <class L, int NG>
__device__ void wide_plain(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias)
{
    __shared__ __attribute__((aligned(16))) half_t hb[64][L::C];
    __shared__ __attribute__((aligned(16))) uint8_t R[WideDims<L, NG>::R_BYTES];
    wide_load_x0<L, NG>(p, R);
    block_sync();
    wide_core<L, NG>(p, img, bias, hb, R);
    wide_store_y<L, NG>(p, hb, p.out24);
}

template <class L, int NG>
__device__ void wide_decoder(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias)
{
    constexpr int C = L::C;
    __shared__ __attribute__((aligned(16))) half_t hb[64][C];
    __shared__ __attribute__((aligned(16))) uint8_t R[WideDims<L, NG>::R_BYTES];
    half_t(*act)[C] = (half_t(*)[C])R;
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z, q = wv & 3, g = wv >> 2, bx = blockIdx.x, by = blockIdx.y;

    // ---- patch expand: waves of sub-token q = (dy, dx) produce its tiles nt = g + NG j for the 16 low-res tokens
    {
        const int W2 = p.W / 2, H2 = p.H / 2;
        const int ly = m >> 2, lx = m & 3;
        const int LX = mirror((8 * bx - p.sx) / 2 + lx, W2), LY = mirror((8 * by - p.sy) / 2 + ly, H2);
        const half_t* xl = (const half_t*)p.in + (size_t)(LY * W2 + LX) * L::CL;
        const int dy = q >> 1, dx = q & 1, tt = 8 * (2 * ly + dy) + 2 * lx + dx;
        const int SX = mirror(8 * bx - p.sx + (tt & 7), p.W), SY = mirror(8 * by - p.sy + (tt >> 3), p.H);
        const half_t* sk = (const half_t*)p.skip + (size_t)(SY * p.W + SX) * C;
        for (int nt = g; nt < C / 16; nt += NG)
        {
            const int gnt = q * (C / 16) + nt;
            acc8v d;
            {
                half_t vv[8];
                dvec8(p.w, L::EXPB + 32 * gnt, vv);
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    d[i] = (float)vv[i];
            }
            for (int ks = 0; ks < L::KL; ++ks)
                d = mma16(op_img(img, (L::T_EXP + ks * L::EXPN + gnt) * 16 + m), op_gload(xl + 16 * ks), d);
            const op_t sv = op_gload(sk + 16 * nt);
            half_t xo[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                xo[i] = (half_t)d[i] + wm_op_acc_elem(sv, i);
            op_store(&act[tt][16 * nt], operand_from_dt(xo));
        }
    }
    block_sync();
    PwinParams qp = p;
    qp.w = p.w + L::CB;
    wide_core<L, NG>(qp, img, bias, hb, R);
    static_assert(L::NOUTA == 0, "output head: use pwin_layer.h");
    wide_store_y<L, NG>(p, hb, p.out24);
}

#define WIDE_KERNEL(NAME, PREP, BODY, NG, ...)                                                                  \
    using NAME##_L = __VA_ARGS__;                                                                                     \
    __device__ u8v g_img[NAME##_L::T_END * 16];                                                                    \
    __device__ u4v g_bias[NAME##_L::NBIAS];                                                                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::PREP_ITEMS + 127) / 128;     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4 * (NG);                                   \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_offset = 64; /* prep reads only p.w */       \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p)                                    \
    {                                                                                                              \
        PREP<NAME##_L>(p, g_img, g_bias);                                                                          \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128 * (NG)) PWIN_VGPR NAME(PwinParams p)                          \
    {                                                                                                              \
        BODY<NAME##_L, NG>(p, g_img, g_bias);                                                                      \
    }

#define WIDE_ENCODER(NAME, H, C, COUT, NG) WIDE_KERNEL(NAME, pwin_prep, wide_encoder, NG, PwinLayout<H, C, COUT>)
#define WIDE_PLAIN(NAME, H, C, NG) WIDE_KERNEL(NAME, pwin_prep, wide_plain, NG, PwinLayout<H, C, 16>)
#define WIDE_DECODER(NAME, H, C, CL, NG) WIDE_KERNEL(NAME, pwin_dec_prep, wide_decoder, NG, PwinDecLayout<H, C, CL>)
