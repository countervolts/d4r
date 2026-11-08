// HIP form of encoder 0's native part (engine/rr/enc0_m_tail.glsl, RR_GFX11) for RDNA3, against the model's
// enc0_m_w.h (engine/rr/enc0_pack.py) and W_WIDE. The translated feature gather stays a Vulkan stage
// (engine/rr/gen_enc0_m.py --gather): it leaves each window's shared buffer sh (1000 words: the noise grid, 10 x 10
// pixels x 4 f16, at byte 0; the two feature planes, 10 x 10 pixels x 16 FP8, at bytes 800 and 2400) in the aux
// buffer at word 1000 x window; this kernel is the rest, with the GLSL stage's arithmetic. One workgroup per 8x8
// window, four wave32 waves. Descriptor table as rr_swin.hip's (surfaces 112, 120, 128; no textures).
#include "rr_hip.h"
#include "enc0_m_w.h"

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
    const uint32_t* aux;   // encoder 0: the gather stage's per-window words
    uint32_t auxBytes, pad;
};

#define TS(o) ((tsharp_t*)(a.desc + (o)))
#define LDA(base, k, mt, kt) lda_at(W, W_WIDE + (base) + 256u * ((mt) * ((k) / 16u) + (kt)) + 16u * g11_row(col))
#define LDB(base, n, kt, nt) ldb_at(W, W_WIDE + (base) + 256u * ((kt) * ((n) / 16u) + (nt)), col)
#define XF KVS   // before the attention: the key pixels' 32 FP8 features (plane 0 | plane 1), slot-major

KERNEL128 void rr_enc0(RrArgs a)
{
    SHARED uint32_t KVS[2u * 768u];   // the key features, then the current head's K (0..767) and V^T (768..)
    SHARED uint32_t XT[96u * 8u];     // the normalised key tokens; then the merge rows (16 x 128 FP8)
    SHARED uint32_t NOISE[200u];      // the noise grid (sh words 0-199)
    SHARED f16 O112[64u * 4u];
    const uint32_t tid = TID(), wv = tid >> 5u, lane = tid & 31u;
    const uint32_t hl = lane >> 4u, col = lane & 15u;
    const bool H = hl != 0u;
    const Buf W = make_buf(a.weights, a.weightBytes), A = make_buf(a.arena, a.arenaBytes), G = make_buf(a.aux, a.auxBytes);
    const uint32_t* p = a.param;
    const int ex = (int)p[0], ey = (int)p[1];
    const uint32_t outOff = p[12], mrgOff = p[14], shift = p[20];
    const int x0 = 8 * (int)WGX() - (int)(shift & 0xffffu), y0 = 8 * (int)WGY() - (int)(shift >> 16u);
    const uint32_t lplane = (uint32_t)((ex / 2) * (ey / 2)) * 32u;
    const uint32_t gx = ((uint32_t)ex + (shift & 0xffffu) + 7u) / 8u;
    const uint32_t sh = 4000u * ((uint32_t)WGY() * gx + (uint32_t)WGX());   // this window's sh, bytes in the aux buffer

    // the key slots' features from the 10 x 10 grid (origin x0 - 1, y0 - 1); the noise grid
    if (tid < 96u) {
        const i2v k = keyXY(inter(tid), x0, y0);
        const uint32_t g = (uint32_t)((k.y - y0 + 1) * 10 + (k.x - x0 + 1));
        const u4v f0 = ld128(G, sh + 800u + g * 16u), f1 = ld128(G, sh + 2400u + g * 16u);
#pragma unroll
        for (uint32_t wd = 0u; wd < 4u; ++wd) { XF[tid * 8u + wd] = f0[wd]; XF[tid * 8u + 4u + wd] = f1[wd]; }
    }
    for (uint32_t i = tid; i < 200u; i += 128u) NOISE[i] = ld32(G, sh + 4u * i);
    SYNC();
    // the embedding x = W2 q8(relu(W1 f + b1)) + b2 for token tiles n = wv (this wave's queries: x is also their
    // residual) and n = wv + 4 (the border keys, waves 0 and 1), then each key's norm in registers to XT
    f8v R[2];
    for (uint32_t n = wv; n < 6u; n += 4u) {
        const u8v fb0 = lds8b(XF, n * 128u, 8u, col), fb1 = lds8b(XF, n * 128u + 4u, 8u, col);
        u8v hbm[2];
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m) {
            f8v acc = ldh8(W, W_B1 + 2u * (16u * m + 8u * hl));
            k32(acc, LDA(W_W1, 32u, m, 0u), fb0, LDA(W_W1, 32u, m, 1u), fb1);
            f8v rl;
#pragma unroll
            for (int i = 0; i < 8; ++i) rl[i] = __builtin_fmaxf(acc[i], 0.0f);
            hbm[m] = g11_bw(wid8(rl), H);
        }
        f16 xv[2][8];
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m) {
            f8v acc = ldh8(W, W_B2 + 2u * (16u * m + 8u * hl));
            k32(acc, LDA(W_W2, 32u, m, 0u), hbm[0], LDA(W_W2, 32u, m, 1u), hbm[1]);
#pragma unroll
            for (int i = 0; i < 8; ++i) xv[m][i] = (f16)acc[i];
            if (n == wv) {   // the residual: x + table 0
                const u4v tab = ld128(W, W_TABLE0 + 2u * (16u * m + 8u * hl));
#pragma unroll
                for (uint32_t q = 0u; q < 4u; ++q) {
                    const hv2 c = (hv2){xv[m][2u * q], xv[m][2u * q + 1u]} + upk(tab[q]);
                    R[m][2u * q] = (float)c.x;
                    R[m][2u * q + 1u] = (float)c.y;
                }
            }
        }
        f16 h1[2];
#pragma unroll
        for (uint32_t c = 0u; c < 2u; ++c) {
            f16 ht[2];
#pragma unroll
            for (uint32_t tq = 0u; tq < 2u; ++tq) {
                const uint32_t i0 = 4u * tq + c, i1 = i0 + 2u;
                const f16 a0 = xv[0][i0] * xv[0][i0], a1 = xv[0][i1] * xv[0][i1];
                const f16 b0 = xv[1][i0] * xv[1][i0], b1 = xv[1][i1] * xv[1][i1];
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
            u2v xq;
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) {
                const hv2 g = r2 * upk(ld32(W, W_GAIN0 + 2u * inter(16u * m + 8u * hl + 2u * q)));
                const hv2 nn = (hv2){xv[m][2u * q], xv[m][2u * q + 1u]} * g;
                const uint32_t cd = e4m3x2(pk(nn));
                if (q & 1u) xq[q >> 1u] |= cd << 16u; else xq[q >> 1u] = cd;
            }
            st8(XT, xq, n * 128u + 4u * m, 8u, lane);
        }
    }
    SYNC();   // XT complete; XF is read (the attention reuses its memory)

    // ---------------------------------------------------------------- attention, two heads
#pragma unroll
    for (uint32_t hd = 0u; hd < 2u; ++hd) {
        if (hd == 1u) SYNC();
        for (uint32_t j = 0u; j < 3u; ++j) {
            const uint32_t job = wv + 4u * j, n = job >> 1u, c = job & 1u;
            const u8v xb0 = lds8b(XT, n * 128u, 8u, col), xb1 = lds8b(XT, n * 128u + 4u, 8u, col);
            const u8v xa0 = lds8a(XT, n * 128u, 8u, col), xa1 = lds8a(XT, n * 128u + 4u, 8u, col);
            f8v ak = (f8v)0.0f, av = (f8v)0.0f;
            k32(ak, LDA(W_WK0 + HEAD * hd, 32u, c, 0u), xb0, LDA(W_WK0 + HEAD * hd, 32u, c, 1u), xb1);
            st8(KVS, codes8(ak), n * 128u + 4u * c, 8u, lane);
            k32(av, xa0, LDB(W_WV0 + HEAD * hd, 32u, 0u, c), xa1, LDB(W_WV0 + HEAD * hd, 32u, 1u, c));
            st8(KVS, codes8(av), 768u + c * 384u + 4u * n, 24u, lane);
        }
        SYNC();
        u8v Ob[2];
        {   // engine/rr/rr_attn.glsl
            u8v Qb[2];
#pragma unroll
            for (uint32_t c = 0u; c < 2u; ++c) Qb[c] = lds8b(KVS, wv * 128u + 4u * c, 8u, col);
            hv2 ep[6][4];
#pragma unroll
            for (uint32_t kt = 0u; kt < 6u; ++kt) {
                f8v cc;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    cc[i] = (float)w16(W, (W_BIAS0 + HEAD * hd) / 2u + kt * 1024u + 16u * wv + (8u * hl + (uint32_t)i) * 64u + col);
                k32(cc, lds8a(KVS, kt * 128u, 8u, col), Qb[0], lds8a(KVS, kt * 128u + 4u, 8u, col), Qb[1]);
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
                    k32(o8, lds8a(KVS, 768u + c * 384u + 8u * ks, 24u, col), Pb[2u * ks], lds8a(KVS, 768u + c * 384u + 8u * ks + 4u, 24u, col),
                        Pb[2u * ks + 1u]);
                Ob[c] = g11_bw(wid8(o8), H);
            }
        }
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m) k32(R[m], LDA(W_WO0 + HEAD * hd, 32u, m, 0u), Ob[0], LDA(W_WO0 + HEAD * hd, 32u, m, 1u), Ob[1]);
    }

    // ---------------------------------------------------------------- MLP (dec0's)
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

    // ---------------------------------------------------------------- outputs: the tokens (dec0's skip), the merge rows
    const uint32_t t = inter(16u * wv + col);
    const int tx = x0 + (int)(t & 7u), ty = y0 + (int)(t >> 3u);
    const bool inside = tx >= 0 && ty >= 0 && tx < ex && ty < ey;
    u8v Tb[2];
    {
        const uint32_t ty2 = t >> 3u, tx2 = t & 7u;
        const uint32_t mr = (ty2 >> 1u) < 2u ? 2u * (tx2 >> 1u) + (ty2 >> 1u) : 8u + 2u * (tx2 >> 1u) + (ty2 >> 1u) - 2u;
        const uint32_t blk = 2u * (ty2 & 1u) + (tx2 & 1u);
        const uint32_t o = outOff + (uint32_t)(ty * ex + tx) * 32u;
        SYNC();   // XT is read; the merge rows reuse it
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m) {
            const u2v tc = codes8(R[m]);
            Tb[m] = g11_b(tc, H);
            st64(A, keep(inside, o + 16u * m + 8u * hl), tc);
            const uint32_t gm = mr * 32u + blk * 8u + 4u * m + 2u * hl;
            XT[gm] = tc.x;
            XT[gm + 1u] = tc.y;
        }
    }
    {   // the 32 -> 16 head and encoder 0's filter of the noise grid (two f16 fma chains, taps even / odd)
        f8v hv = ldh8(W, W_BH + 2u * (8u * hl));
        k32(hv, LDA(W_WH, 32u, 0u, 0u), Tb[0], LDA(W_WH, 32u, 0u, 1u), Tb[1]);
        f16 l[12];
#pragma unroll
        for (int i = 0; i < 8; ++i) l[i] = (f16)hv[i];
        const uint32_t u0 = other_half(bh(l[0]) | (bh(l[1]) << 16u)), u1 = other_half(bh(l[2]) | (bh(l[3]) << 16u));
        l[8] = hb(u0 & 0xffffu); l[9] = hb(u0 >> 16u); l[10] = hb(u1 & 0xffffu); l[11] = hb(u1 >> 16u);
        const f16 ma = __builtin_fmaxf16(__builtin_fmaxf16(__builtin_fmaxf16(l[0], l[2]), l[4]), l[6]);
        const f16 mb = __builtin_fmaxf16(__builtin_fmaxf16(__builtin_fmaxf16(l[1], l[3]), l[5]), l[7]);
        const f16 mx = __builtin_fmaxf16(__builtin_fmaxf16(ma, mb), l[8]);
        float e[9];
#pragma unroll
        for (uint32_t i = 0u; i < 9u; ++i) { const f16 d = l[i] - mx; e[i] = ex2((float)d * 1.4426950216293335f); }
        float s = (e[0] + e[1]) + 0.0f;
        s = s + (e[2] + e[3]);
        s = s + (e[4] + e[5]);
        s = s + (e[6] + e[7]);
        s = s + e[8];
        const f16 rr = rcp16(s);
        f16 wg[9];
#pragma unroll
        for (uint32_t i = 0u; i < 9u; ++i) wg[i] = h16(r16(e[i]) * (float)rr);
        const uint32_t gxt = t & 7u, gyt = t >> 3u;   // the token's 3 x 3 taps in the grid
        auto noise = [&](uint32_t x, uint32_t y, uint32_t c) {
            const uint32_t b = 80u * y + 8u * x + 2u * c;
            return hb((NOISE[b >> 2] >> (8u * (b & 2u))) & 0xffffu);
        };
        f16 o112[3], o120[3];
#pragma unroll
        for (uint32_t c = 0u; c < 3u; ++c) {
            f16 lo = (f16)0.0f, hi = (f16)0.0f;
#pragma unroll
            for (uint32_t k = 0u; k < 8u; k += 2u) {
                lo = fma16(noise(gxt + k % 3u, gyt + k / 3u, c), wg[k], lo);
                hi = fma16(noise(gxt + (k + 1u) % 3u, gyt + (k + 1u) / 3u, c), wg[k + 1u], hi);
            }
            const f16 sum = lo + hi;
            const f16 f = fma16(noise(gxt + 2u, gyt + 2u, c), wg[8], sum);
            const f16 am = sig(l[9u + c]) * f;
            const f16 bm = f - am;
            o120[c] = am; o112[c] = bm;
        }
        if (hl == 0u) {
            if (inside) {
                __ockl_image_store_2D(TS(24), (i2v){tx, ty}, (f4v){(float)o112[0], (float)o112[1], (float)o112[2], 0.0f});
                __ockl_image_store_2D(TS(32), (i2v){tx, ty}, (f4v){(float)o120[0], (float)o120[1], (float)o120[2], 0.0f});
            }
#pragma unroll
            for (uint32_t c = 0u; c < 3u; ++c) O112[t * 4u + c] = o112[c];
        }
    }
    SYNC();
    {   // the 2x2 merge (128 -> 64): wave w computes channel tile w
        const int X = x0 / 2 + (int)mergeR(col, false), Y = y0 / 2 + (int)mergeR(col, true), hx = ex / 2, hy = ey / 2;
        const uint32_t m = wv;
        f8v acc = ldh8(W, W_BM + 2u * (16u * m + 8u * hl));
#pragma unroll
        for (uint32_t ks = 0u; ks < 4u; ++ks)
            k32(acc, LDA(W_WM, 128u, m, 2u * ks), lds8b(XT, 8u * ks, 32u, col), LDA(W_WM, 128u, m, 2u * ks + 1u), lds8b(XT, 8u * ks + 4u, 32u, col));
        const u2v q = codes8(acc);
        st64(A, keep(X >= 0 && Y >= 0 && X < hx && Y < hy, mrgOff + (m >> 1u) * lplane + (uint32_t)(Y * hx + X) * 32u + 16u * (m & 1u) + 8u * hl), q);
    }
    if (wv == 3u && lane < 16u) {   // surface 128: 2x2 sums of 0.25 * surface 112, rightwards first, then downwards
        const uint32_t mx = lane & 3u, my = lane >> 2u;
        const int x = x0 + 2 * (int)mx, y = y0 + 2 * (int)my;
        if (x >= 0 && y >= 0 && x < ex && y < ey) {
            const uint32_t t00 = (2u * my) * 8u + 2u * mx;
            f4v out = (f4v){0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (uint32_t c = 0u; c < 3u; ++c) {
                const f16 q4 = (f16)0.25f;
                const f16 a00 = O112[t00 * 4u + c] * q4, a01 = O112[(t00 + 1u) * 4u + c] * q4;
                const f16 a10 = O112[(t00 + 8u) * 4u + c] * q4, a11 = O112[(t00 + 9u) * 4u + c] * q4;
                const f16 b0 = a00 + a01, b1 = a10 + a11;
                out[c] = (float)(f16)(b0 + b1);
            }
            __ockl_image_store_2D(TS(40), (i2v){x >> 1, y >> 1}, out);
        }
    }
}
