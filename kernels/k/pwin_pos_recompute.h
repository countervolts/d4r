// Single-buffer position-only-attention encoder experiment (trait flag 2: enc0, enc1, dec0) with MT token tiles per wave, so every
// weight operand loaded from L2 is used MT times (4 / MT waves per window). Same math as pwin_core.
#pragma once
#include "pwin_layer.h"

// POS_DIRECT_NORM (no-embedding MT=1 layer, enc1): waves 0 and 1 load one token per lane straight from global
// memory and write its normalized row, instead of four waves staging the raw rows in LDS and both half-waves of
// each re-reading and summing every row. The same sums, roundings and gamma products; act/hb alias, and the raw
// input is never needed in LDS (the residual reloads it from global memory).
template <class L, int MT>
constexpr bool pos_direct_norm()
{
#ifdef POS_DIRECT_NORM
    return L::CIN == 0 && MT == 1;
#else
    return false;
#endif
}

template <class L, int MT>
__device__ void pos_recompute_core(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias,
                         half_t (*act)[LDSR(L::C)], half_t (*hb)[LDSR(L::C)])
{
    constexpr int C = L::C, KT = L::KT, H = L::H, NW = 4 / MT;
    static_assert(L::POS, "position-only attention layers");
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.z);
    const uint8_t* w = p.w;
    PROF_INIT;

    // ---- L2 norm of every tile of this wave -> hb (done by pos_recompute_encoder for pos_direct_norm)
#pragma unroll
    for (int mi = 0; mi < (pos_direct_norm<L, MT>() ? 0 : MT); ++mi)
    {
        const half_t* arow = act[16 * (wv * MT + mi) + m];
        half_t* hrow = hb[16 * (wv * MT + mi) + m];
        float ss = 0.0f;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt)
        {
            const u8v xv = lds_row16(arow + 16 * kt);
#pragma unroll
#if defined(PWIN_PK_ELEM) && !defined(PWIN_PK_NO_SS)
            for (int j = 0; j < 8; ++j)
            {
                const uint32_t xj = xv[j];
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
            const u8v gv = lds_row16((const half_t*)(w + L::G1) + 16 * kt);
            u8v hv;
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
    }
    block_sync(); // V of all 64 tokens is computed from hb
    STAMP(2);

    // ---- attention: O = P V per head (P from the table), output projection accumulated head by head
    f8v acc[MT][KT];
#pragma unroll
    for (int nt = 0; nt < KT; ++nt)
    {
        const f8v bo = acc_vec<L>(img, w, L::BO + 32 * nt, L::V_BO + nt);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
            acc[mi][nt] = bo;
    }
#ifdef POS_VSHARE
    // V^T operands of every head: each wave computes those of its own key tiles, shared through LDS
    // (one slot per head, so a head's slot is never rewritten while it is read)
    __shared__ __attribute__((aligned(16))) u8v vsh[H][2][4][16];
#endif
#ifdef POS_VSHARE1
    // V^T operands shared through ONE 4 KB slot reused per head (a barrier before it is rewritten); each wave
    // computes its own key tiles, the PV loop reads the operands from LDS when it needs them (no 8 operands live
    // in registers). Same values as computing all of V in every wave.
    __shared__ __attribute__((aligned(16))) u8v vsh1[2][4][16];
#endif
    for (int h = 0; h < H; ++h)
    {
        op_t vtop[2][4];
#ifdef POS_VSHARE1
        if (h > 0)
            block_sync(); // every wave has finished reading the previous head's V^T
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
            {
                const int kt = wv * MT + mi;
                f8v d = splat8(0.0f);
#pragma unroll
                for (int ks = 0; ks < KT; ++ks)
                    d = mma16(op_lds(&hb[16 * kt + m][16 * ks]), op_imgt(img, L::T_QKV + ((h * 3 + 2) * KT + ks) * 2 + nt), d);
                op_img_store(&vsh1[nt][0][0], kt * 16 + m, operand_from_f8(d));
            }
        block_sync();
        (void)vtop;
#elif defined(POS_VSHARE)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
            {
                const int kt = wv * MT + mi;
                f8v d = splat8(0.0f);
#pragma unroll
                for (int ks = 0; ks < KT; ++ks)
                    d = mma16(op_lds(&hb[16 * kt + m][16 * ks]), op_imgt(img, L::T_QKV + ((h * 3 + 2) * KT + ks) * 2 + nt), d);
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
                    d = mma16(op_lds(&hb[16 * kt + m][16 * ks]), op_imgt(img, L::T_QKV + ((h * 3 + 2) * KT + ks) * 2 + nt), d);
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
                pop[kt] = op_imgt((const u8v*)bias, (h * 4 + qt) * 4 + kt);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                f8v d = splat8(0.0f);
#pragma unroll
                for (int kt = 0; kt < 4; ++kt)
                {
#ifdef POS_VSHARE1
                    d = mma16(op_img(&vsh1[nt][0][0], kt * 16 + m), pop[kt], d);
#else
                    d = mma16(vtop[nt][kt], pop[kt], d);
#endif
                }
                oop[mi][nt] = operand_from_f8(d);
            }
        }
#pragma unroll
        for (int nt = 0; nt < KT; ++nt)
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
            {
                const op_t wo = op_imgt(img, L::T_WO + (2 * h + ks) * KT + nt);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi)
                    acc[mi][nt] = mma16(wo, oop[mi][ks], acc[mi][nt]);
            }
    }

    STAMP(3);
    // All attention readers finish before the shared input rows are reused.
    block_sync();
    // Recover x0 exactly from the unchanged input (repeat the original embedding,
    // or reload a no-embedding layer), then write the MLP input into the same LDS rows.
#pragma unroll
    for (int nt = 0; nt < KT; ++nt)
    {
        half_t g2[8], b2[8];
        residual_vector<L>(w, 0, nt, g2);
        residual_vector<L>(w, 1, nt, b2);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
        {
            const int tok = 16 * (wv * MT + mi) + m;
            const int X = mirror(8 * (int)blockIdx.x - p.sx + (tok & 7), p.W);
            const int Y = mirror(8 * (int)blockIdx.y - p.sy + (tok >> 3), p.H);
            half_t x0[8];
            if constexpr (L::CIN != 0)
            {
                const half_t* xin = (const half_t*)p.in + (size_t)(Y * p.W + X) * L::CIN;
                f8v d = acc_vec<L>(img, w - L::CB, 32 * nt, L::V_EMB + nt);
#pragma unroll
                for (int ks = 0; ks < L::KE; ++ks)
                    d = mma16(op_imgt(img, L::T_EMB + ks * KT + nt), op_gload(xin + 16 * ks), d);
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    x0[i] = hmax((half_t)d[i], (half_t)0.0f);
            }
            else
            {
                const half_t* xin = (const half_t*)p.in + (size_t)(Y * p.W + X) * C;
                const op_t xv = op_gload(xin + 16 * nt);
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    x0[i] = wm_op_acc_elem(xv, i);
            }
            half_t mv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const half_t x1 = (half_t)acc[mi][nt][i] + x0[i];
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
            const f8v bb = acc_vec<L>(img, w, L::B1 + 64 * c + 32 * hn, L::V_B1 + 2 * c + hn);
            f8v d[MT];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
                d[mi] = bb;
#pragma unroll
            for (int kt = 0; kt < KT; ++kt)
            {
                const op_t w1 = op_imgt(img, L::T_W1 + (c * KT + kt) * 2 + hn);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi)
                    d[mi] = mma16(w1, op_lds(&hb[16 * (wv * MT + mi) + m][16 * kt]), d[mi]);
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
            {
                gop[mi][hn] = gelu_op(d[mi]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < KT; ++nt)
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
            {
                const op_t w2 = op_imgt(img, L::T_W2 + (c * 2 + ks) * KT + nt);
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
    PROF(6);
}

// encoder (enc0 with embedding, enc1): core + full-resolution output + patch merge
template <class L, int MT>
__device__ void pos_recompute_encoder(const PwinParams& p0, const u8v* __restrict__ img, const u4v* __restrict__ bias)
{
    constexpr int C = L::C, NW = 4 / MT;
    __shared__ __attribute__((aligned(16))) half_t act[64][LDSR(C)];
    half_t (*hb)[LDSR(C)] = act; // one allocation: x0 -> norm -> MLP input -> final output
    PwinParams p = p0;
    if constexpr (L::CB != 0)
        p.w = p0.w + L::CB;
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int wv = __builtin_amdgcn_readfirstlane(threadIdx.z), bx = blockIdx.x, by = blockIdx.y;
    PROF_INIT;
    STAMP(20);
    if constexpr (pos_direct_norm<L, MT>())
    {
        // wave-uniform producer branch, no barrier inside it: pos_recompute_core's first block_sync orders these
        // hb rows before every wave's attention
        if (wv < 2)
        {
            const int tok = 32 * wv + (int)l;
            const int X = mirror(8 * bx - p.sx + (tok & 7), p.W), Y = mirror(8 * by - p.sy + (tok >> 3), p.H);
            const half_t* xin = (const half_t*)p.in + (size_t)(Y * p.W + X) * C;
            u8v xr[L::KT];
#pragma unroll
            for (int kt = 0; kt < L::KT; ++kt)
                xr[kt] = gload_row16(xin + 16 * kt);
            float ss = 0.0f;
#pragma unroll
            for (int kt = 0; kt < L::KT; ++kt)
            {
                const u8v xv = xr[kt];
#pragma unroll
#if defined(PWIN_PK_ELEM) && !defined(PWIN_PK_NO_SS)
                for (int j = 0; j < 8; ++j)
                {
                    const uint32_t xj = xv[j];
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
#ifdef D4R_ACCURACY
            const half_t r16 = half_rsqrt(ptx_sumsq<L::C>([&](int q) { return (uint32_t)xr[q >> 3][q & 7]; }));
#else
            const half_t r16 = half_rsqrt((half_t)ss);
#endif
#pragma unroll
            for (int kt = 0; kt < L::KT; ++kt)
            {
                const u8v xv = xr[kt];
                const u8v gv = lds_row16((const half_t*)(p.w + L::G1) + 16 * kt);
                u8v hv;
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
                store_row16(&hb[tok][16 * kt], hv);
            }
        }
    }
    else
    {
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
                    f8v d = acc_vec<L>(img, p0.w, 32 * nt, L::V_EMB + nt);
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
    }
    STAMP(21);
    pos_recompute_core<L, MT>(p, img, bias, act, hb);
    STAMP(26);

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
    STAMP(27);
    const int my = m >> 2, mx = m & 3;
    const int Ym = (8 * by - p.sy) / 2 + my, Xm = (8 * bx - p.sx) / 2 + mx;
    const bool minb = Ym >= 0 && Ym < p.H / 2 && Xm >= 0 && Xm < p.W / 2;
    half_t* mout = (half_t*)p.out24 + (size_t)(Ym * (p.W / 2) + Xm) * L::COUT;
    for (int vt = wv; vt < L::PMT; vt += NW)
    {
        f8v d = acc_vec<L>(img, p.w, L::PMB + 32 * vt, L::V_PM + vt);
        for (int ks = 0; ks < C / 4; ++ks)
        {
            const int sub = ks / L::KT, dy = sub >> 1, dx = sub & 1;
            const int src = 8 * (2 * my + dy) + 2 * mx + dx;
            d = mma16(op_imgt(img, L::T_PM + ks * L::PMT + vt), op_lds(&act[src][16 * (ks % L::KT)]), d);
        }
        const int g = vt / (L::NPA / 16), u = vt % (L::NPA / 16), valid = L::NPW - 16 * u;
        op_gstore(mout + L::NPW * g + 16 * u, operand_from_f8(d), minb, valid);
    }
    STAMP(28);
}

#define POS_RECOMPUTE_ENCODER(NAME, H, C, COUT, CIN, MT)                                                                     \
    using NAME##_L = PwinEmbLayout<H, C, COUT, true, CIN>;                                                         \
    __device__ u8v g_img[NAME##_L::IMG_SLOTS];                                                                    \
    __device__ u4v g_bias[NAME##_L::NBIAS];                                                                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (pwin_gelu_prep_items(NAME##_L::PREP_ITEMS) + 127) / 128;     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4 / MT;                                     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_offset = 64; /* prep reads only p.w */       \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p)                                    \
    {                                                                                                              \
        pwin_gelu_prep();                                                                                         \
        pwin_emb_prep<NAME##_L>(p, g_img, g_bias);                                                                 \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(32 * 4 / MT) PWIN_VGPR NAME(PwinParams p)                                 \
    {                                                                                                              \
        pos_recompute_encoder<NAME##_L, MT>(p, g_img, g_bias);                                                               \
    }

