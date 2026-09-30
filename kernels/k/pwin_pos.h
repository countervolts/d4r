// Position-only-attention layers (trait flag 2: enc0, enc1, dec0) with MT token tiles per wave, so every
// weight operand loaded from L2 is used MT times (4 / MT waves per window). Same math as pwin_core.
#pragma once
#include "pwin_layer.h"

template <class L, int MT>
__device__ void pos_core(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias,
                         half_t (*act)[L::C], half_t (*hb)[L::C])
{
    constexpr int C = L::C, KT = L::KT, H = L::H, NW = 4 / MT;
    static_assert(L::POS, "position-only attention layers");
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z;
    const uint8_t* w = p.w;

    // ---- L2 norm of every tile of this wave -> hb
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
    {
        const half_t* arow = act[16 * (wv * MT + mi) + m];
        half_t* hrow = hb[16 * (wv * MT + mi) + m];
        float ss = 0.0f;
#pragma unroll
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
            const u8v gv = lds_row16((const half_t*)(w + L::G1) + 16 * kt);
            u8v hv;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                hv[j] = pack2(op_get(xv, 2 * j) * (half_t)(r16 * op_get(gv, 2 * j)),
                              op_get(xv, 2 * j + 1) * (half_t)(r16 * op_get(gv, 2 * j + 1)));
            store_row16(hrow + 16 * kt, hv);
        }
    }
    block_sync(); // V of all 64 tokens is computed from hb
    STAMP(2);

    // ---- attention: O = P V per head (P from the table), output projection accumulated head by head
    f8v acc[MT][KT];
#pragma unroll
    for (int nt = 0; nt < KT; ++nt)
    {
        half_t vv[8];
        dvec8(w, L::BO + 32 * nt, vv);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[mi][nt][i] = (float)vv[i];
    }
#ifdef POS_VSHARE
    // V^T operands of every head: each wave computes those of its own key tiles, shared through LDS
    // (one slot per head, so a head's slot is never rewritten while it is read)
    __shared__ __attribute__((aligned(16))) u8v vsh[H][2][4][16];
#endif
    for (int h = 0; h < H; ++h)
    {
        op_t vtop[2][4];
#ifdef POS_VSHARE
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
            {
                const int kt = wv * MT + mi;
                f8v d = splat8(0.0f);
#pragma unroll
                for (int ks = 0; ks < KT; ++ks)
                    d = mma16(op_lds(&hb[16 * kt + m][16 * ks]), op_img(img, (L::T_QKV + ((h * 3 + 2) * KT + ks) * 2 + nt) * 16 + m), d);
                op_img_store(&vsh[h][nt][0][0], kt * 16 + m, operand_from_f8(d));
            }
        block_sync();
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int kt = 0; kt < 4; ++kt)
                vtop[nt][kt] = op_img(&vsh[h][nt][0][0], kt * 16 + m);
#else
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int kt = 0; kt < 4; ++kt)
            {
                f8v d = splat8(0.0f);
#pragma unroll
                for (int ks = 0; ks < KT; ++ks)
                    d = mma16(op_lds(&hb[16 * kt + m][16 * ks]), op_img(img, (L::T_QKV + ((h * 3 + 2) * KT + ks) * 2 + nt) * 16 + m), d);
                vtop[nt][kt] = operand_from_f8(d);
            }
#endif
        op_t oop[MT][2];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
        {
            const int qt = wv * MT + mi;
            op_t pop[4];
#pragma unroll
            for (int kt = 0; kt < 4; ++kt)
                pop[kt] = op_img((const u8v*)bias, ((h * 4 + qt) * 4 + kt) * 16 + m);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                f8v d = splat8(0.0f);
#pragma unroll
                for (int kt = 0; kt < 4; ++kt)
                    d = mma16(vtop[nt][kt], pop[kt], d);
                oop[mi][nt] = operand_from_f8(d);
            }
        }
#pragma unroll
        for (int nt = 0; nt < KT; ++nt)
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
            {
                const op_t wo = op_img(img, (L::T_WO + (2 * h + ks) * KT + nt) * 16 + m);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi)
                    acc[mi][nt] = mma16(wo, oop[mi][ks], acc[mi][nt]);
            }
    }

    STAMP(3);
    // ---- residual, MLP init acc = x1 + b2, MLP input m = x1 * g2 (-> hb)
#pragma unroll
    for (int nt = 0; nt < KT; ++nt)
    {
        half_t g2[8], b2[8];
        dvec8(w, L::G2 + 32 * nt, g2);
        dvec8(w, L::B2 + 32 * nt, b2);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
        {
            const int tok = 16 * (wv * MT + mi) + m;
            const op_t xv = op_lds(&act[tok][16 * nt]);
            half_t mv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const half_t x1 = (half_t)acc[mi][nt][i] + wm_op_acc_elem(xv, i);
                mv[i] = x1 * g2[i];
                acc[mi][nt][i] = (float)(half_t)(x1 + b2[i]);
            }
            op_store(&hb[tok][16 * nt], operand_from_dt(mv));
        }
    }
    __builtin_amdgcn_wave_barrier();

    STAMP(4);
    // ---- MLP: C/8 chunks of 32 hidden; each weight operand serves the MT tiles
    for (int c = 0; c < L::NMLP; ++c)
    {
        op_t gop[MT][2];
#pragma unroll
        for (int hn = 0; hn < 2; ++hn)
        {
            half_t bb[8];
            dvec8(w, L::B1 + 64 * c + 32 * hn, bb);
            f8v d[MT];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    d[mi][i] = (float)bb[i];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt)
            {
                const op_t w1 = op_img(img, (L::T_W1 + (c * KT + kt) * 2 + hn) * 16 + m);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi)
                    d[mi] = mma16(w1, op_lds(&hb[16 * (wv * MT + mi) + m][16 * kt]), d[mi]);
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
            {
                half_t g[8];
                gelu8(d[mi], g);
                gop[mi][hn] = operand_from_dt(g);
            }
        }
#pragma unroll
        for (int nt = 0; nt < KT; ++nt)
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
            {
                const op_t w2 = op_img(img, (L::T_W2 + (c * 2 + ks) * KT + nt) * 16 + m);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi)
                    acc[mi][nt] = mma16(w2, gop[mi][ks], acc[mi][nt]);
            }
    }
    STAMP(5);
    // y -> act
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int nt = 0; nt < KT; ++nt)
        {
            op_store(&act[16 * (wv * MT + mi) + m][16 * nt], operand_from_f8(acc[mi][nt]));
        }
    block_sync();
}

// encoder (enc0 with embedding, enc1): core + full-resolution output + patch merge
template <class L, int MT>
__device__ void pos_encoder(const PwinParams& p0, const u8v* __restrict__ img, const u4v* __restrict__ bias)
{
    constexpr int C = L::C, NW = 4 / MT;
    __shared__ __attribute__((aligned(16))) half_t act[64][C];
    __shared__ __attribute__((aligned(16))) half_t hb[64][C];
    PwinParams p = p0;
    if constexpr (L::CB != 0)
        p.w = p0.w + L::CB;
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;
    STAMP(0);
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
    {
        const int tok = 16 * (wv * MT + mi) + m, ty = tok >> 3, tx = tok & 7;
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
                half_t vv[8];
                dvec8(p0.w, 32 * nt, vv);
                f8v d;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    d[i] = (float)vv[i];
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
    STAMP(1);
    pos_core<L, MT>(p, img, bias, act, hb);
    STAMP(6);

#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
    {
        const int tok = 16 * (wv * MT + mi) + m;
        const int Y0 = 8 * by - p.sy + (tok >> 3), X0 = 8 * bx - p.sx + (tok & 7);
        if (Y0 >= 0 && Y0 < p.H && X0 >= 0 && X0 < p.W)
        {
            half_t* yout = (half_t*)p.out32 + (size_t)(Y0 * p.W + X0) * C;
            for (int kt = (int)hf; kt < L::KT; kt += 2)
                gstore_row16(yout + 16 * kt, lds_row16(&act[tok][16 * kt]));
        }
    }
    STAMP(7);
    const int my = m >> 2, mx = m & 3;
    const int Ym = (8 * by - p.sy) / 2 + my, Xm = (8 * bx - p.sx) / 2 + mx;
    const bool minb = Ym >= 0 && Ym < p.H / 2 && Xm >= 0 && Xm < p.W / 2;
    half_t* mout = (half_t*)p.out24 + (size_t)(Ym * (p.W / 2) + Xm) * L::COUT;
    for (int vt = wv; vt < L::PMT; vt += NW)
    {
        half_t vv[8];
        dvec8(p.w, L::PMB + 32 * vt, vv);
        f8v d;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            d[i] = (float)vv[i];
        for (int ks = 0; ks < C / 4; ++ks)
        {
            const int sub = ks / L::KT, dy = sub >> 1, dx = sub & 1;
            const int src = 8 * (2 * my + dy) + 2 * mx + dx;
            d = mma16(op_img(img, (L::T_PM + ks * L::PMT + vt) * 16 + m), op_lds(&act[src][16 * (ks % L::KT)]), d);
        }
        const int g = vt / (L::NPA / 16), u = vt % (L::NPA / 16), valid = L::NPW - 16 * u;
        op_gstore(mout + L::NPW * g + 16 * u, operand_from_f8(d), minb, valid);
    }
    STAMP(8);
}

// dec0: patch expand + skip, core, output head
template <class L, int MT>
__device__ void pos_decoder(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias)
{
    constexpr int C = L::C, NW = 4 / MT;
    __shared__ __attribute__((aligned(16))) half_t act[64][C];
    __shared__ __attribute__((aligned(16))) half_t hb[64][C];
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;
    {
        const int W2 = p.W / 2, H2 = p.H / 2, ly = m >> 2, lx = m & 3;
        const int LX = mirror((8 * bx - p.sx) / 2 + lx, W2), LY = mirror((8 * by - p.sy) / 2 + ly, H2);
        const half_t* xl = (const half_t*)p.in + (size_t)(LY * W2 + LX) * L::CL;
        op_t xls[L::KL];
#pragma unroll
        for (int ks = 0; ks < L::KL; ++ks)
            xls[ks] = op_gload(xl + 16 * ks);
        for (int q = wv; q < 4; q += NW)
        {
            const int dy = q >> 1, dx = q & 1, t = 8 * (2 * ly + dy) + 2 * lx + dx;
            const int SX = mirror(8 * bx - p.sx + (t & 7), p.W), SY = mirror(8 * by - p.sy + (t >> 3), p.H);
            const half_t* sk = (const half_t*)p.skip + (size_t)(SY * p.W + SX) * C;
#pragma unroll
            for (int nt = 0; nt < C / 16; ++nt)
            {
                const int gnt = q * (C / 16) + nt;
                half_t vv[8];
                dvec8(p.w, L::EXPB + 32 * gnt, vv);
                f8v d;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    d[i] = (float)vv[i];
#pragma unroll
                for (int ks = 0; ks < L::KL; ++ks)
                    d = mma16(op_img(img, (L::T_EXP + ks * L::EXPN + gnt) * 16 + m), xls[ks], d);
                const op_t sv = op_gload(sk + 16 * nt);
                half_t xo[8];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    xo[i] = (half_t)d[i] + wm_op_acc_elem(sv, i);
                op_store(&act[t][16 * nt], operand_from_dt(xo));
            }
        }
    }
    block_sync();
    PwinParams q = p;
    q.w = p.w + L::CB;
    pos_core<L, MT>(q, img, bias, act, hb);
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
    {
        const int tok = 16 * (wv * MT + mi) + m, Y0 = 8 * by - p.sy + (tok >> 3), X0 = 8 * bx - p.sx + (tok & 7);
        const bool inb = Y0 >= 0 && Y0 < p.H && X0 >= 0 && X0 < p.W;
        half_t* hout = (half_t*)p.out24 + (size_t)(Y0 * p.W + X0) * L::NOUT;
#pragma unroll
        for (int nt = 0; nt < L::NOUTA / 16; ++nt)
        {
            half_t vv[8];
            dvec8(q.w, L::HB + 32 * nt, vv);
            f8v d;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                d[i] = (float)vv[i];
#pragma unroll
            for (int kt = 0; kt < L::KT; ++kt)
                d = mma16(op_img(img, (L::T_HEAD + kt * (L::NOUTA / 16) + nt) * 16 + m), op_lds(&act[tok][16 * kt]), d);
            op_gstore(hout + 16 * nt, operand_from_f8(d), inb, L::NOUT - 16 * nt);
        }
    }
}

#define POS_ENCODER(NAME, H, C, COUT, CIN, MT)                                                                     \
    using NAME##_L = PwinEmbLayout<H, C, COUT, true, CIN>;                                                         \
    __device__ u8v g_img[NAME##_L::T_END * 16];                                                                    \
    __device__ u4v g_bias[NAME##_L::NBIAS];                                                                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::PREP_ITEMS + 127) / 128;     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4 / MT;                                     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_offset = 64; /* prep reads only p.w */       \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p)                                    \
    {                                                                                                              \
        pwin_emb_prep<NAME##_L>(p, g_img, g_bias);                                                                 \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(32 * 4 / MT) PWIN_VGPR NAME(PwinParams p)                                 \
    {                                                                                                              \
        pos_encoder<NAME##_L, MT>(p, g_img, g_bias);                                                               \
    }

#define POS_DECODER(NAME, H, C, CL, NOUT, NOUTA, MT)                                                               \
    using NAME##_L = PwinDecLayout<H, C, CL, true, NOUT, NOUTA>;                                                   \
    __device__ u8v g_img[NAME##_L::T_END * 16];                                                                    \
    __device__ u4v g_bias[NAME##_L::NBIAS];                                                                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::PREP_ITEMS + 127) / 128;     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4 / MT;                                     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_offset = 64; /* prep reads only p.w */       \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p)                                    \
    {                                                                                                              \
        pwin_dec_prep<NAME##_L>(p, g_img, g_bias);                                                                 \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(32 * 4 / MT) PWIN_VGPR NAME(PwinParams p)                                 \
    {                                                                                                              \
        pos_decoder<NAME##_L, MT>(p, g_img, g_bias);                                                               \
    }
