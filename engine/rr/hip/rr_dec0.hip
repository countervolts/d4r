// HIP form of engine/rr/dec0_m.comp (RR_GFX11): decoder 0 of the DLSS-D/E denoiser on RDNA3, against the model's
// dec0_m_w.h (engine/rr/dec0_pack.py) and W_WIDE. Same inputs, outputs and arithmetic as the GLSL stage (see
// rr_hip.h for what differs: instructions only). One workgroup per 8x8-token window at the render resolution, four
// wave32 waves. Descriptor table as rr_swin.hip's, with surface 144 (RGBA8 SNORM) in the S120 slot.
#include "rr_hip.h"
#include "dec0_m_w.h"

#ifndef W_WIDE
#error "W_WIDE (the widened weight copy's f16 element offset) is required"
#endif
#define HEAD (W_WK1 - W_WK0)
#define BRANCH (W_W1_1 - W_W1_0)

struct RrArgs {
    const uint32_t* param;
    const void* weights;
    uint32_t* arena;
    const uint32_t* desc;
    uint32_t weightBytes, arenaBytes;
    const uint32_t* aux;   // encoder 0: the gather stage's per-window words; unused elsewhere
    uint32_t auxBytes, pad;
};

#define TS(o) ((tsharp_t*)(a.desc + (o)))
#define LDA(base, k, mt, kt) lda_at(W, W_WIDE + (base) + 256u * ((mt) * ((k) / 16u) + (kt)) + 16u * g11_row(col))
#define LDB(base, n, kt, nt) ldb_at(W, W_WIDE + (base) + 256u * ((kt) * ((n) / 16u) + (nt)), col)
// shared words: [LOW (48 x 16) | EX (36 x 32 used of 48 x 32)], then the attention's K (0..767) and V^T (768..)
#define EX_AT 768u
#define KS0 0u
#define VS0 768u

// NVIDIA's tanh (dec0_m.comp tanhNV): the GPU's exp and reciprocal instructions
DEV float tanhNV(float t)
{
    const float m = __builtin_fabsf(t), s = t * t;
    const float big = m >= 9.0109138f ? 1.0f : __builtin_fmaf(__builtin_amdgcn_rcpf(ex2(m * 2.885390043258667f) + 1.0f), -2.0f, 1.0f);
    return m >= 0.6f ? (t < 0.0f ? -big : big)
                     : __builtin_fmaf(__builtin_fmaf(__builtin_fmaf(__builtin_fmaf(__builtin_fmaf(0.01573968306183815f, s, -0.05230396240949631f), s,
                                                                                   0.13315297663211823f), s, -0.33332768082618713f), s, 0.0f), t, t);
}

KERNEL128 void rr_dec0(RrArgs a)
{
    SHARED uint32_t POOL[48u * 16u + 48u * 32u];
    SHARED uint32_t XT[96u * 8u];
    SHARED f16 XR[64u * 32u];   // the window tokens before the norm, then (OUTF) the final tokens, natural channels
    SHARED f16 TAP[100u * 4u];
    const uint32_t tid = TID(), wv = tid >> 5u, lane = tid & 31u;
    const uint32_t hl = lane >> 4u, col = lane & 15u;
    const bool H = hl != 0u;
    const Buf W = make_buf(a.weights, a.weightBytes), A = make_buf(a.arena, a.arenaBytes);
    const uint32_t* p = a.param;
    const int ex = (int)p[0], ey = (int)p[1], hx = ex / 2, hy = ey / 2;
    const uint32_t loOff = p[2], skOff = p[6], outOff = p[12], shift = p[20];
    const int x0 = 8 * (int)WGX() - (int)(shift & 0xffffu), y0 = 8 * (int)WGY() - (int)(shift >> 16u);
    const int X0 = (x0 >> 1) - 1, Y0 = (y0 >> 1) - 1;
    const uint32_t lplane = (uint32_t)(hx * hy) * 32u;

    // ---------------------------------------------------------------- expand
    for (uint32_t i = tid; i < 48u * 16u; i += 128u) {
        const uint32_t r = i >> 4u, wd = i & 15u;
        uint32_t u = 0u;
        if (r < 36u) {
            const int X = mirror(X0 + (int)(r % 6u), hx), Y = mirror(Y0 + (int)(r / 6u), hy);
            u = ld32(A, loOff + (wd >> 3u) * lplane + (uint32_t)(Y * hx + X) * 32u + 4u * (wd & 7u));
        }
        POOL[i] = u;
    }
    SYNC();
    {
        u8v we[2][4];
#pragma unroll
        for (uint32_t mm = 0u; mm < 2u; ++mm)
#pragma unroll
            for (uint32_t kt = 0u; kt < 4u; ++kt) we[mm][kt] = LDA(W_WE, 64u, 2u * wv + mm, kt);
#pragma unroll
        for (uint32_t n = 0u; n < 3u; ++n) {
            u8v lb[4];
#pragma unroll
            for (uint32_t kt = 0u; kt < 4u; ++kt) lb[kt] = lds8b(POOL, n * 256u + 4u * kt, 16u, col);
#pragma unroll
            for (uint32_t mm = 0u; mm < 2u; ++mm) {
                const uint32_t m = 2u * wv + mm;
                f8v acc = ldh8(W, W_BE + 2u * (16u * m + 8u * hl));
                k32(acc, we[mm][0], lb[0], we[mm][1], lb[1]);
                k32(acc, we[mm][2], lb[2], we[mm][3], lb[3]);
                st8(POOL, codes8(acc), EX_AT + n * 512u + 4u * m, 32u, lane);
            }
        }
    }
    SYNC();

    // ---------------------------------------------------------------- prologue: one key token per thread
    if (tid < 96u) {
        const i2v k = keyXY(inter(tid), x0, y0);
        const uint32_t r = (uint32_t)(((k.y >> 1) - Y0) * 6 + ((k.x >> 1) - X0)), s = (uint32_t)(2 * (k.y & 1) + (k.x & 1));
        const int xx = mirror(k.x, ex), yy = mirror(k.y, ey);
        const uint32_t sb = skOff + (uint32_t)(yy * ex + xx) * 32u;
        hv2 v[16];
#pragma unroll
        for (uint32_t wd = 0u; wd < 8u; ++wd) {
            hv2 e0, e1, s0, s1;
            e4h4(POOL[EX_AT + r * 32u + 8u * s + wd], e0, e1);
            e4h4(ld32(A, sb + 4u * wd), s0, s1);
            v[2u * wd] = e0 + s0;
            v[2u * wd + 1u] = e1 + s1;
        }
        // dec0_ref.norm_plane: per word t the pair sums, words t and t + 4, then t ^ 1, t ^ 2, the halves
        hv2 h[4];
#pragma unroll
        for (uint32_t t = 0u; t < 4u; ++t) {
            const hv2 a0 = v[2u * t] * v[2u * t], a1 = v[2u * t + 1u] * v[2u * t + 1u];
            const hv2 b0 = v[8u + 2u * t] * v[8u + 2u * t], b1 = v[9u + 2u * t] * v[9u + 2u * t];
            const hv2 p0 = a0 + a1, p1 = b0 + b1;
            h[t] = p0 + p1;
        }
        const hv2 h0 = h[0] + h[1], h2_ = h[2] + h[3];
        const hv2 hs = h0 + h2_;
        const f16 sm = hs.x + hs.y;
        const f16 se = sm + (f16)0.0001220703125f;
        const f16 rr = rsq16(se);
        const hv2 r2 = (hv2){rr, rr};
#pragma unroll
        for (uint32_t wd = 0u; wd < 8u; ++wd) {
            uint32_t c2[2];
#pragma unroll
            for (uint32_t b = 0u; b < 2u; ++b) {
                const hv2 g = r2 * upk(ld32(W, W_GAIN0 + 2u * inter(4u * wd + 2u * b)));
                const hv2 nn = v[2u * wd + b] * g;
                c2[b] = e4m3x2(pk(nn));
            }
            XT[tid * 8u + wd] = c2[0] | (c2[1] << 16);
        }
        if (tid < 64u)
#pragma unroll
            for (uint32_t j = 0u; j < 16u; ++j) { XR[tid * 32u + 2u * j] = v[j].x; XR[tid * 32u + 2u * j + 1u] = v[j].y; }
    }
    SYNC();

    // ---------------------------------------------------------------- the residual: window tokens + table 0
    f8v R[2];
#pragma unroll
    for (uint32_t m = 0u; m < 2u; ++m) {
        const u4v tab = ld128(W, W_TABLE0 + 2u * (16u * m + 8u * hl));
#pragma unroll
        for (uint32_t q = 0u; q < 4u; ++q) {
            const uint32_t at = wv * 512u + 16u * m + col * 32u + 8u * hl + 2u * q;
            const hv2 c = (hv2){XR[at], XR[at + 1u]} + upk(tab[q]);
            R[m][2u * q] = (float)c.x;
            R[m][2u * q + 1u] = (float)c.y;
        }
    }

    // ---------------------------------------------------------------- attention, two heads
#pragma unroll
    for (uint32_t hd = 0u; hd < 2u; ++hd) {
        if (hd == 1u) SYNC();
        for (uint32_t j = 0u; j < 3u; ++j) {   // 12 jobs (token tile n, channel tile c)
            const uint32_t job = wv + 4u * j, n = job >> 1u, c = job & 1u;
            const u8v xb0 = lds8b(XT, n * 128u, 8u, col), xb1 = lds8b(XT, n * 128u + 4u, 8u, col);
            const u8v xa0 = lds8a(XT, n * 128u, 8u, col), xa1 = lds8a(XT, n * 128u + 4u, 8u, col);
            f8v ak = (f8v)0.0f, av = (f8v)0.0f;
            k32(ak, LDA(W_WK0 + HEAD * hd, 32u, c, 0u), xb0, LDA(W_WK0 + HEAD * hd, 32u, c, 1u), xb1);
            st8(POOL, codes8(ak), KS0 + n * 128u + 4u * c, 8u, lane);
            k32(av, xa0, LDB(W_WV0 + HEAD * hd, 32u, 0u, c), xa1, LDB(W_WV0 + HEAD * hd, 32u, 1u, c));
            st8(POOL, codes8(av), VS0 + c * 384u + 4u * n, 24u, lane);
        }
        SYNC();
        u8v Ob[2];
        {   // engine/rr/rr_attn.glsl
            u8v Qb[2];
#pragma unroll
            for (uint32_t c = 0u; c < 2u; ++c) Qb[c] = lds8b(POOL, KS0 + wv * 128u + 4u * c, 8u, col);
            hv2 ep[6][4];
#pragma unroll
            for (uint32_t kt = 0u; kt < 6u; ++kt) {
                f8v cc;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    cc[i] = (float)w16(W, (W_BIAS0 + HEAD * hd) / 2u + kt * 1024u + 16u * wv + (8u * hl + (uint32_t)i) * 64u + col);
                k32(cc, lds8a(POOL, KS0 + kt * 128u, 8u, col), Qb[0], lds8a(POOL, KS0 + kt * 128u + 4u, 8u, col), Qb[1]);
                f16 e[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) e[i] = hb(ld16(W, W_CUBIC + 2u * bh((f16)cc[i])));
#pragma unroll
                for (uint32_t q = 0u; q < 4u; ++q) ep[kt][q] = (hv2){e[2u * q], e[2u * q + 1u]};
            }
            hv2 st[2];
#pragma unroll
            for (uint32_t tq = 0u; tq < 2u; ++tq) {
                const uint32_t q0 = 2u * tq, q1 = q0 + 1u;
                const hv2 ev = ((ep[0][q0] + ep[0][q1]) + (ep[2][q0] + ep[2][q1])) + (ep[4][q0] + ep[4][q1]);
                const hv2 od = ((ep[1][q0] + ep[1][q1]) + (ep[3][q0] + ep[3][q1])) + (ep[5][q0] + ep[5][q1]);
                st[tq] = ev + od;
            }
            const hv2 loc = st[0] + st[1];
            const hv2 o = loc + upk(other_half(pk(loc)));
            const f16 rs = o.x + o.y;
            const f16 r = h16(__builtin_amdgcn_rcpf((float)rs));
            const hv2 r2 = (hv2){r, r};
            u8v Pb[6];
#pragma unroll
            for (uint32_t kt = 0u; kt < 6u; ++kt) {
                u4v pw;
#pragma unroll
                for (uint32_t q = 0u; q < 4u; ++q) pw[q] = wid2(pk(ep[kt][q] * r2));
                Pb[kt] = g11_bw(pw, H);
            }
#pragma unroll
            for (uint32_t c = 0u; c < 2u; ++c) {
                f8v o8 = (f8v)0.0f;
#pragma unroll
                for (uint32_t ks = 0u; ks < 3u; ++ks)
                    k32(o8, lds8a(POOL, VS0 + c * 384u + 8u * ks, 24u, col), Pb[2u * ks], lds8a(POOL, VS0 + c * 384u + 8u * ks + 4u, 24u, col),
                        Pb[2u * ks + 1u]);
                Ob[c] = g11_bw(wid8(o8), H);
            }
        }
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m) k32(R[m], LDA(W_WO0 + HEAD * hd, 32u, m, 0u), Ob[0], LDA(W_WO0 + HEAD * hd, 32u, m, 1u), Ob[1]);
    }

    // ---------------------------------------------------------------- MLP
    u8v Nb[2];
    {
        f16 rv[2][8];
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m)
#pragma unroll
            for (int i = 0; i < 8; ++i) rv[m][i] = (f16)R[m][i];
        f16 h1[2];
#pragma unroll
        for (uint32_t c = 0u; c < 2u; ++c) {
            f16 ht[2];
#pragma unroll
            for (uint32_t tq = 0u; tq < 2u; ++tq) {
                const uint32_t i0 = 4u * tq + c, i1 = i0 + 2u;
                const f16 a0 = rv[0][i0] * rv[0][i0], a1 = rv[0][i1] * rv[0][i1];
                const f16 b0 = rv[1][i0] * rv[1][i0], b1 = rv[1][i1] * rv[1][i1];
                const f16 p0 = a0 + a1, p1 = b0 + b1;
                ht[tq] = p0 + p1;
            }
            h1[c] = ht[0] + ht[1];
        }
        const uint32_t other = other_half(bh(h1[0]) | (bh(h1[1]) << 16u));
        const f16 g0 = h1[0] + hb(other & 0xffffu), g1 = h1[1] + hb(other >> 16u);
        const f16 s = g0 + g1;
        const f16 se = s + (f16)0.0001220703125f;
        const f16 r = rsq16(se);
        const hv2 r2 = (hv2){r, r};
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m) {
            const u4v gain = ld128(W, W_GAIN1 + 2u * (16u * m + 8u * hl));
            const u4v tab = ld128(W, W_TABLE1 + 2u * (16u * m + 8u * hl));
            u4v nw;
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) {
                const hv2 x = (hv2){rv[m][2u * q], rv[m][2u * q + 1u]};
                const hv2 g = r2 * upk(gain[q]);
                const hv2 nn = x * g;
                nw[q] = wid2(pk(nn));
                const hv2 ad = x + upk(tab[q]);
                R[m][2u * q] = (float)ad.x;
                R[m][2u * q + 1u] = (float)ad.y;
            }
            Nb[m] = g11_bw(nw, H);
        }
    }
#pragma unroll
    for (uint32_t br = 0u; br < 4u; ++br) {
        const uint32_t w1 = W_W1_0 + BRANCH * br, b1 = W_B1_0 + BRANCH * br, w2 = W_W2_0 + BRANCH * br;
        u8v Hb[2];
#pragma unroll
        for (uint32_t c = 0u; c < 2u; ++c) {
            f8v hh = ldh8(W, b1 + 2u * (16u * c + 8u * hl));
            k32(hh, LDA(w1, 32u, c, 0u), Nb[0], LDA(w1, 32u, c, 1u), Nb[1]);
            u4v hw;   // the widened activation codes: the weight file's widened copy of the table
#pragma unroll
            for (int q = 0; q < 4; ++q)
                hw[q] = ld16(W, 2u * (W_WIDE + W_ACT + bh((f16)hh[2 * q]))) | (ld16(W, 2u * (W_WIDE + W_ACT + bh((f16)hh[2 * q + 1]))) << 16);
            Hb[c] = g11_bw(hw, H);
        }
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m) k32(R[m], LDA(w2, 32u, m, 0u), Hb[0], LDA(w2, 32u, m, 1u), Hb[1]);
    }

    // ---------------------------------------------------------------- outputs, one token per thread
    {
        const uint32_t t = inter(16u * wv + col);
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m)
#pragma unroll
            for (int i = 0; i < 8; ++i) XR[t * 32u + inter(16u * m + 8u * hl + (uint32_t)i)] = (f16)R[m][i];
    }
    SYNC();
    if (tid < 100u) {   // texture 88 at the window's 10 x 10 tap positions, through the bilinear sampler
        const int xi = clampTap(x0 - 1 + (int)(tid % 10u), ex), yi = clampTap(y0 - 1 + (int)(tid / 10u), ey);
        const float iw = __builtin_bit_cast(float, p[34]), ih = __builtin_bit_cast(float, p[35]);
        const float cu = __builtin_fmaf(iw, (float)xi, iw * 0.5f), cv = __builtin_fmaf(ih, (float)yi, ih * 0.5f);
        const f4v bl = __ockl_image_sample_lod_2D(TS(0), TS(48), (__attribute__((ext_vector_type(2))) float){cu, cv}, 0.0f);
#pragma unroll
        for (uint32_t c = 0u; c < 3u; ++c) TAP[tid * 4u + c] = (f16)bl[c];
    }
    SYNC();
    {   // two threads per token: half 0 filters channels 0 and 1, half 1 channel 2 and writes the output stage's codes
        // and surface 144
        const uint32_t t = tid >> 1u, hf = tid & 1u;
        const int x = x0 + (int)(t & 7u), y = y0 + (int)(t >> 3u);
        if (x < 0 || y < 0 || x >= ex || y >= ey) return;
        f16 av[10];
#pragma unroll
        for (uint32_t c = 0u; c < 10u; ++c) av[c] = XR[t * 32u + c];
        if (hf == 1u) {
            f16 b[12];
#pragma unroll
            for (uint32_t c = 0u; c < 12u; ++c) b[c] = XR[t * 32u + 12u + c];
            const uint32_t o = outOff + (uint32_t)(y * ex + x) * 8u;
#ifdef RR_PRESET_D
            // the diamond_wallaby kernel: channels 12, 13, 0, 18, 19, 20, 21, 0
            const uint32_t w0 = q8h(b[0]) | (q8h(b[1]) << 8u) | (q8h(b[6]) << 24u);
            const uint32_t w1 = q8h(b[7]) | (q8h(b[8]) << 8u) | (q8h(b[9]) << 16u);
#else
            const uint32_t w0 = q8h(b[0]) | (q8h(b[1]) << 8u) | (q8h(b[6]) << 16u) | (q8h(b[7]) << 24u);
            const uint32_t w1 = q8h(b[8]) | (q8h(b[9]) << 8u) | (q8h(b[10]) << 16u) | (q8h(b[11]) << 24u);
#endif
            st64(A, o, (u2v){w0, w1});
            __ockl_image_store_2D(TS(32), (i2v){x, y},
                                  (f4v){tanhNV((float)b[2]), tanhNV((float)b[3]), tanhNV((float)b[4]), tanhNV((float)b[5])});
        }
        const f16 ma = __builtin_fmaxf16(__builtin_fmaxf16(__builtin_fmaxf16(av[0], av[2]), av[4]), av[6]);
        const f16 mb = __builtin_fmaxf16(__builtin_fmaxf16(__builtin_fmaxf16(av[1], av[3]), av[5]), av[7]);
        const f16 mx = __builtin_fmaxf16(__builtin_fmaxf16(ma, mb), av[8]);
        float e[9];
#pragma unroll
        for (uint32_t i = 0u; i < 9u; ++i) { const f16 d = av[i] - mx; e[i] = ex2((float)d * 1.4426950216293335f); }
        float s = (e[0] + e[1]) + 0.0f;
        s = s + (e[2] + e[3]);
        s = s + (e[4] + e[5]);
        s = s + (e[6] + e[7]);
        s = s + e[8];
        const f16 rr = rcp16(s);
        f16 wg[9];
#pragma unroll
        for (uint32_t i = 0u; i < 9u; ++i) wg[i] = h16(r16(e[i]) * (float)rr);
        const uint32_t txl = (uint32_t)(x - x0), tyl = (uint32_t)(y - y0), c0 = 2u * hf;
        f16 f[2] = {(f16)0.0f, (f16)0.0f};
        bool mid = false;
#pragma unroll
        for (uint32_t k = 0u; k < 9u; ++k) {
            const uint32_t at = ((tyl + k / 3u) * 10u + txl + k % 3u) * 4u + c0;
#pragma unroll
            for (uint32_t c = 0u; c < 2u; ++c) f[c] = fma16q(TAP[at + c], wg[k], f[c], mid);
        }
        if (mid) {
            f[0] = (f16)0.0f; f[1] = (f16)0.0f;
            for (uint32_t k = 0u; k < 9u; ++k) {
                const uint32_t at = ((tyl + k / 3u) * 10u + txl + k % 3u) * 4u + c0;
                for (uint32_t c = 0u; c < 2u; ++c) f[c] = fma16(TAP[at + c], wg[k], f[c]);
            }
        }
        const f16 gs = sig(av[9]);
        const f16 om = (f16)1.0f - gs;
        const f4v t96 = __ockl_image_load_2D(TS(8), (i2v){x, y}), t104 = __ockl_image_load_2D(TS(16), (i2v){x, y});
        f16 o[2];
#pragma unroll
        for (uint32_t c = 0u; c < 2u; ++c) {
            const f16 fg = f[c] * gs;
            f16 v = fma16((f16)t96[c0 + c], om, fg);
            const f16 t4 = (f16)t104[c0 + c];
            if (!__builtin_isnan(t4)) v = v + t4;
            o[c] = v;
        }
        const uint32_t other = (uint32_t)__builtin_amdgcn_ds_bpermute((int)((lane ^ 1u) * 4u), (int)bh(o[0]));
        if (hf == 0u) __ockl_image_store_2D(TS(24), (i2v){x, y}, (f4v){(float)o[0], (float)o[1], (float)hb(other), 0.0f});
    }
}
