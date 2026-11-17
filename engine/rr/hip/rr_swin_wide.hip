// Sixteen-wave form of engine/rr/hip/rr_swin.hip for the Swin stages with few windows (the deep levels): one
// workgroup of 16 wave32 waves per 8x8-token window. Same inputs, outputs and arithmetic per element as rr_swin.hip
// (and swin_m.comp): every product chain runs in NVIDIA's order; only which wave runs it changes.
//
// Wave w: query tile qt = w & 3 (window token slots 16 qt .. 16 qt + 15) and role ro = w >> 2, which owns the
// residual channel tiles m = ro, ro + 4, ... < MT. Phases:
//   expand (decoders): the 3 x 4MT (row tile, channel tile) products spread over all waves
//   prologue: one key token per thread (threads 0-95); residual: each wave its own (qt, m) tiles, kept in registers
//   attention, G heads at a time: the 12 G (token tile, channel tile) K / V jobs over all waves; the 4 G (head,
//     query tile) attention cores; then each wave adds the G heads' out-projections onto its tiles, heads in order
//   MLP: the norm per query tile (every wave of the tile computes it, from the tiles the waves exchanged through
//     LDS); branches four at a time: wave (qt, ro) computes branch 4 g + ro's hidden codes, then every wave adds the
//     four branches' second products onto its tiles, branches in order
//   outputs: arena tokens and merge rows per wave; the head and the filter by waves 0-3 (one per query tile); the
//     2x2 merge's channel tiles over all waves
#include "rr_hip.h"
#include "swin_stage.h"

#ifndef W_WIDE
#error "W_WIDE (the widened weight copy's f16 element offset) is required"
#endif
#define KC (C / 32)
#define MT (C / 16)
#define NB (C / 8)
#define HEAD (W_WK1 - W_WK0)
#define BRANCH (W_W1_1 - W_W1_0)
#define MO ((MT + 3) / 4)   // residual tiles per wave at most
#if NB % 4
#error "the MLP runs branches four at a time"
#endif

#if KC == 1
#define TREE(v) (v[0])
#elif KC == 2
#define TREE(v) (v[0] + v[1])
#elif KC == 3
#define TREE(v) ((v[0] + v[1]) + v[2])
#elif KC == 4
#define TREE(v) ((v[0] + v[1]) + (v[2] + v[3]))
#else
#define TREE(v) (((v[0] + v[1]) + (v[2] + v[3])) + v[4])
#endif

struct RrArgs {
    const uint32_t* param;
    const void* weights;
    uint32_t* arena;
    const uint32_t* desc;
    uint32_t weightBytes, arenaBytes;
    const uint32_t* aux;   // encoder 0: the gather stage's per-window words; unused elsewhere
    uint32_t auxBytes, pad;
};

// ---------------------------------------------------------------- shared memory (words), by phase
#if KIND == 1
#define EX_AT 0u
#define EX_W (36u * C)              // the expansion: 36 low tokens x 4 sub-positions x C FP8 channels
#define LOW_AT EX_W                 // the expand's low-resolution tokens, then the prologue's XT
#define XT_AT EX_W
#define B_W (12u * CX > 24u * C ? 12u * CX : 24u * C)
#define ATT_AT (EX_W + B_W)
#else
#define XT_AT 0u
#define ATT_AT (24u * C)
#endif
// 64 KiB of LDS less the stage's other arrays (TAP + OUTF, or O112)
#if KIND == 1
#define POOL_CAP (16384u - 704u)
#else
#define POOL_CAP (16384u - 128u)
#endif
#define G_FIT ((POOL_CAP - ATT_AT) / 2048u)
#if NH >= 4 && G_FIT >= 4
#define G 4
#elif NH >= 2 && G_FIT >= 2
#define G 2
#else
#define G 1
#endif
// per local head: K (768 words), V^T (768), O codes (2 channel tiles x 4 query tiles x 32 lanes x 2 words)
#define KS_AT(lh) (ATT_AT + 2048u * (lh))
#define VS_AT(lh) (ATT_AT + 2048u * (lh) + 768u)
#define O_AT(lh, c, q) (ATT_AT + 2048u * (lh) + 1536u + 64u * (4u * (c) + (q)))
#define ATT_END (ATT_AT + 2048u * G)
#define TT_AT (ATT_AT + 256u)                       // the prologue's per-word terms (96 x C / 4), before attention
#define TT_END (TT_AT + 24u * C)
// MLP and outputs (everything above is dead by then)
#define RX_AT 0u                                    // the residual tiles, binary16 (4 words per lane)
#define RX(q, m) (RX_AT + 128u * ((q) * MT + (m)))
#define H_AT (512u * MT)                            // hidden codes, two buffers of 4 branches x 2 x 4 query tiles
#define HX(buf, b, c, q) (H_AT + 2048u * (buf) + 64u * (8u * (b) + 4u * (c) + (q)))
#define MLP_END (H_AT + 4096u)
#define MG_AT 0u                                    // the merge rows (16 x C words)
#define TB_AT (16u * C)                             // the output tokens' codes for the head
#define TBX(q, m) (TB_AT + 64u * ((q) * MT + (m)))
#define OUT_END (TB_AT + 256u * MT)
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#if KIND == 1
#define POOL_W MAX2(MAX2(MAX2(ATT_END, MLP_END), OUT_END), TT_END)
#else
#define POOL_W MAX2(MAX2(MAX2(ATT_END, MLP_END), OUT_END), TT_END)
#endif

#define TS(o) ((tsharp_t*)(a.desc + (o)))
#define LDA(base, k, mt, kt) lda_at(W, W_WIDE + (base) + 256u * ((mt) * ((k) / 16u) + (kt)) + 16u * g11_row(col))
#define LDB(base, n, kt, nt) ldb_at(W, W_WIDE + (base) + 256u * ((kt) * ((n) / 16u) + (nt)), col)

DEV void stx2(uint32_t* s, uint32_t at, u2v v, uint32_t lane) { s[at + 2u * lane] = v.x; s[at + 2u * lane + 1u] = v.y; }
DEV u2v ldx2(const uint32_t* s, uint32_t at, uint32_t lane) { return (u2v){s[at + 2u * lane], s[at + 2u * lane + 1u]}; }

#ifdef RR_STAMPS
// timing builds only: s_memtime at phase boundaries, wave 0 of window (0, 0), written to the arena's last 128 bytes
#define STAMP(i) do { if (stampOn) stamps[i] = __builtin_readsteadycounter(); } while (0)
#else
#define STAMP(i) do { } while (0)
#endif
KERNEL512 void rr_swin_w16(RrArgs a)
{
#ifdef RR_STAMPS
    uint64_t stamps[16] = {};
    uint64_t gstamp[8] = {};
    const bool stampOn = WGX() == 0u && WGY() == 0u && TID() < 32u;
#endif
#ifdef RR_IPF
#define RR_STR2(x) #x
#define RR_STR(x) RR_STR2(x)
    asm volatile("s_set_inst_prefetch_distance " RR_STR(RR_IPF));
#endif
    SHARED uint32_t SM[POOL_W];
#if KIND == 1
    SHARED f16 TAP[100 * 4];
    SHARED f16 OUTF[64 * 16];
#elif CX
    SHARED f16 O112[64 * 4];
#endif
    const uint32_t tid = TID(), wv = tid >> 5u, lane = tid & 31u;
    const uint32_t hl = lane >> 4u, col = lane & 15u;
    const uint32_t qt = wv & 3u, ro = wv >> 2u;
    const bool H = hl != 0u;
    const Buf W = make_buf(a.weights, a.weightBytes), A = make_buf(a.arena, a.arenaBytes);
    const uint32_t* p = a.param;
    const int ex = (int)p[0], ey = (int)p[1], hx = ex / 2, hy = ey / 2;
    const uint32_t inOff = p[2], skOff = p[6], outOff = p[12], shift = p[20];
#if KIND == 0 && CX
    const uint32_t mrgOff = p[14];
#endif
    const int x0 = 8 * (int)WGX() - (int)(shift & 0xffffu), y0 = 8 * (int)WGY() - (int)(shift >> 16u);
    {   // this window's share of the stage's weights into the caches (the grid is the engine's: level + shift, 8x8)
        const uint32_t gx = ((uint32_t)ex + (shift & 0xffffu) + 7u) / 8u, gy = ((uint32_t)ey + (shift >> 16u) + 7u) / 8u;
        warm(W, a.weightBytes, (uint32_t)WGY() * gx + (uint32_t)WGX(), gx * gy, TID(), 512u, A);
    }
    const uint32_t plane = (uint32_t)(ex * ey) * 32u;
    const uint32_t lplane = (uint32_t)(hx * hy) * 32u;
    (void)lplane;

    STAMP(0);
#if KIND == 1
    // ---------------------------------------------------------------- expand: (row tile n, channel tile m) jobs
    const int X0 = (x0 >> 1) - 1, Y0 = (y0 >> 1) - 1;
    for (uint32_t i = tid; i < 48u * (CX / 4u); i += 512u) {
        const uint32_t r = i / (CX / 4u), wd = i % (CX / 4u);
        uint32_t u = 0u;
        if (r < 36u) {
            const int X = mirror(X0 + (int)(r % 6u), hx), Y = mirror(Y0 + (int)(r / 6u), hy);
            u = ld32(A, inOff + (wd >> 3u) * lplane + (uint32_t)(Y * hx + X) * 32u + 4u * (wd & 7u));
        }
        SM[LOW_AT + i] = u;
    }
    SYNC();
    {
        u8v lb[CX / 16];
        uint32_t have = 3u;
        for (uint32_t job = wv; job < 12u * MT; job += 16u) {
            const uint32_t n = job / (4u * MT), m = job % (4u * MT);
            if (n != have) {
#pragma unroll
                for (uint32_t kt = 0u; kt < CX / 16; ++kt) lb[kt] = lds8b(SM, LOW_AT + n * 4u * CX + 4u * kt, CX / 4u, col);
                have = n;
            }
            f8v acc = ldh8(W, W_BE + 2u * (16u * m + 8u * hl));
#pragma unroll
            for (uint32_t kt = 0u; kt < CX / 16; kt += 2u) k32(acc, LDA(W_WE, CX, m, kt), lb[kt], LDA(W_WE, CX, m, kt + 1u), lb[kt + 1u]);
            // rows 36-47 are never read: they go to the attention region (unused yet) with no branch, since a
            // lane-divergent store lets the compiler sink the products into it, and WMMA needs every lane
            const uint32_t at = 16u * n + col < 36u ? EX_AT + n * 16u * C + 4u * m : ATT_AT + 4u * m - (lane & 15u) * C;
            st8(SM, codes8(acc), at, C, lane);
        }
    }
    SYNC();
#endif

    STAMP(7);
    // ---------------------------------------------------------------- prologue: one key token per thread
    // every thread stages the raw key tokens (into XT, rewritten in place below) and the norm's gains (in the
    // attention region, unused until then); then one thread per token
    for (uint32_t i = tid; i < 96u * (C / 4u); i += 512u) {
        const uint32_t tok = i / (C / 4u), wd = i % (C / 4u);
        const i2v k = keyXY(inter(tok), x0, y0);
        const uint32_t tb = (KIND == 1 ? skOff : inOff) + (uint32_t)(mirror(k.y, ey) * ex + mirror(k.x, ex)) * 32u;
        SM[XT_AT + i] = ld32(A, tb + (wd >> 3u) * plane + 4u * (wd & 7u));
    }
    for (uint32_t i = tid; i < C / 2u; i += 512u) SM[ATT_AT + i] = ld32(W, W_GAIN0 + 2u * inter(2u * i));
    SYNC();
    if (tid < 96u) {
        // rolled (the stage runs few windows, so its code mostly runs from cold instruction caches): pass 1 the
        // words' pair-sum terms to LDS, the norm tree from them in NVIDIA's order, pass 2 the values again, encoded
        const i2v k = keyXY(inter(tid), x0, y0);
#if KIND == 1
        const uint32_t r = (uint32_t)(((k.y >> 1) - Y0) * 6 + ((k.x >> 1) - X0)), s = (uint32_t)(2 * (k.y & 1) + (k.x & 1));
#else
        (void)k;
#endif
        auto values = [&](uint32_t wd, hv2& t0, hv2& t1) {
            e4h4(SM[XT_AT + tid * (C / 4u) + wd], t0, t1);
#if KIND == 1
            hv2 e0, e1;
            e4h4(SM[EX_AT + r * C + s * (C / 4u) + wd], e0, e1);
            t0 = e0 + t0;
            t1 = e1 + t1;
#endif
        };
#pragma unroll 1
        for (uint32_t wd = 0u; wd < C / 4; ++wd) {
            hv2 t0, t1;
            values(wd, t0, t1);
            const hv2 sa = t0 * t0, sc = t1 * t1;
            SM[TT_AT + tid * (C / 4u) + wd] = pk(sa + sc);
        }
        // swin_ref.input_norm: per lane word q (and q + 4) the planes' terms (TREE), the halves, then q ^ 1, q ^ 2
        hv2 h[4];
#pragma unroll
        for (uint32_t q = 0u; q < 4u; ++q) {
            hv2 hh[2];
#pragma unroll
            for (uint32_t half_ = 0u; half_ < 2u; ++half_) {
                hv2 pr[KC];
#pragma unroll
                for (uint32_t pl = 0u; pl < KC; ++pl) pr[pl] = upk(SM[TT_AT + tid * (C / 4u) + 8u * pl + q + 4u * half_]);
                hh[half_] = TREE(pr);
            }
            h[q] = hh[0] + hh[1];
        }
        const hv2 h0 = h[0] + h[1], h2_ = h[2] + h[3];
        const hv2 hs = h0 + h2_;
        const f16 sm = hs.x + hs.y;
        const f16 se = sm + (f16)0.0001220703125f;
        const f16 rr = rsq16(se);
        const hv2 r2 = (hv2){rr, rr};
#pragma unroll 1
        for (uint32_t wd = 0u; wd < C / 4; ++wd) {
            hv2 t[2];
            values(wd, t[0], t[1]);
            uint32_t c2[2];
#pragma unroll
            for (uint32_t b = 0u; b < 2u; ++b) {
                const hv2 g = r2 * upk(SM[ATT_AT + 2u * wd + b]);
                const hv2 nn = t[b] * g;
                c2[b] = e4m3x2(pk(nn));
            }
            SM[XT_AT + tid * (C / 4u) + wd] = c2[0] | (c2[1] << 16);
        }
    }

    // ---------------------------------------------------------------- the residual: this wave's tiles
    f8v R[MO];
    {
        const uint32_t tq = inter(16u * qt + col);
        const int qx = x0 + (int)(tq & 7u), qy = y0 + (int)(tq >> 3u);
        const uint32_t tb = (KIND == 1 ? skOff : inOff) + (uint32_t)(mirror(qy, ey) * ex + mirror(qx, ex)) * 32u;
#if KIND == 1
        const uint32_t r = (uint32_t)(((qy >> 1) - Y0) * 6 + ((qx >> 1) - X0)), s = (uint32_t)(2 * (qy & 1) + (qx & 1));
#endif
#pragma unroll
        for (uint32_t k = 0u; k < MO; ++k) {
            const uint32_t m = ro + 4u * k;
            if (m >= MT) break;
            const u2v t = ld64(A, tb + (m >> 1u) * plane + 16u * (m & 1u) + 8u * hl);
            const u4v tab = ld128(W, W_TABLE7 + 2u * (16u * m + 8u * hl));
#if KIND == 1
            const uint32_t ea = EX_AT + r * C + s * (C / 4u) + 4u * m + 2u * hl;
            const u2v e = (u2v){SM[ea], SM[ea + 1u]};
#endif
#pragma unroll
            for (uint32_t w2 = 0u; w2 < 2u; ++w2) {
                hv2 x0p, x1p;
                e4h4(t[w2], x0p, x1p);
#if KIND == 1
                hv2 e0p, e1p;
                e4h4(e[w2], e0p, e1p);
                x0p = e0p + x0p;
                x1p = e1p + x1p;
#endif
                const hv2 c0 = x0p + upk(tab[2u * w2]), c1 = x1p + upk(tab[2u * w2 + 1u]);
                R[k][4u * w2] = (float)c0.x;
                R[k][4u * w2 + 1u] = (float)c0.y;
                R[k][4u * w2 + 2u] = (float)c1.x;
                R[k][4u * w2 + 3u] = (float)c1.y;
            }
        }
    }
    SYNC();   // XT complete; the decoders' EX no longer read
    STAMP(1);

    // ---------------------------------------------------------------- attention, G heads at a time
#pragma nounroll
    for (uint32_t g = 0u; g < NH / G; ++g) {
        // K / V^T of the group's heads: 12 jobs per head (token tile n, channel tile c)
        for (uint32_t job = wv; job < 12u * G; job += 16u) {
            const uint32_t lh = job / 12u, jj = job % 12u, n = jj >> 1u, c = jj & 1u, hd = G * g + lh;
            f8v ak = (f8v)0.0f, av = (f8v)0.0f;
#pragma unroll 1
            for (uint32_t kt = 0u; kt < 2u * KC; kt += 2u) {
                const u8v xb0 = lds8b(SM, XT_AT + n * 4u * C + 4u * kt, C / 4u, col);
                const u8v xb1 = lds8b(SM, XT_AT + n * 4u * C + 4u * kt + 4u, C / 4u, col);
                const u8v xa0 = lds8a(SM, XT_AT + n * 4u * C + 4u * kt, C / 4u, col);
                const u8v xa1 = lds8a(SM, XT_AT + n * 4u * C + 4u * kt + 4u, C / 4u, col);
                k32(ak, LDA(W_WK0 + HEAD * hd, C, c, kt), xb0, LDA(W_WK0 + HEAD * hd, C, c, kt + 1u), xb1);
                k32(av, xa0, LDB(W_WV0 + HEAD * hd, 32u, kt, c), xa1, LDB(W_WV0 + HEAD * hd, 32u, kt + 1u, c));
            }
            st8(SM, codes8(ak), KS_AT(lh) + n * 128u + 4u * c, 8u, lane);
            st8(SM, codes8(av), VS_AT(lh) + c * 384u + 4u * n, 24u, lane);
        }
        SYNC();
        // the attention cores: (head, query tile) jobs (engine/rr/rr_attn.glsl)
        for (uint32_t job = wv; job < 4u * G; job += 16u) {
            const uint32_t lh = job >> 2u, q = job & 3u, hd = G * g + lh;
            u8v Qb[2];
#pragma unroll
            for (uint32_t c = 0u; c < 2u; ++c) Qb[c] = lds8b(SM, KS_AT(lh) + q * 128u + 4u * c, 8u, col);
            hv2 ep[6][4];
#pragma unroll
            for (uint32_t kt = 0u; kt < 6u; ++kt) {
                f8v cc;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    cc[i] = (float)w16(W, (W_BIAS0 + HEAD * hd) / 2u + kt * 1024u + 16u * q + (8u * hl + (uint32_t)i) * 64u + col);
                k32(cc, lds8a(SM, KS_AT(lh) + kt * 128u, 8u, col), Qb[0], lds8a(SM, KS_AT(lh) + kt * 128u + 4u, 8u, col), Qb[1]);
                f16 e[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) e[i] = hb(ld16(W, W_CUBIC + 2u * bh((f16)cc[i])));
#pragma unroll
                for (uint32_t qq = 0u; qq < 4u; ++qq) ep[kt][qq] = (hv2){e[2u * qq], e[2u * qq + 1u]};
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
                for (uint32_t qq = 0u; qq < 4u; ++qq) pw[qq] = wid2(pk(ep[kt][qq] * r2));
                Pb[kt] = g11_bw(pw, H);
            }
#pragma unroll
            for (uint32_t c = 0u; c < 2u; ++c) {
                f8v o8 = (f8v)0.0f;
#pragma unroll
                for (uint32_t ks = 0u; ks < 3u; ++ks)
                    k32(o8, lds8a(SM, VS_AT(lh) + c * 384u + 8u * ks, 24u, col), Pb[2u * ks],
                        lds8a(SM, VS_AT(lh) + c * 384u + 8u * ks + 4u, 24u, col), Pb[2u * ks + 1u]);
                stx2(SM, O_AT(lh, c, q), codes8(o8), lane);
            }
        }
        SYNC();
        // the out-projections onto this wave's tiles, heads in order
#pragma unroll 1
        for (uint32_t lh = 0u; lh < G; ++lh) {
            const uint32_t hd = G * g + lh;
            const u8v Ob0 = g11_b(ldx2(SM, O_AT(lh, 0u, qt), lane), H), Ob1 = g11_b(ldx2(SM, O_AT(lh, 1u, qt), lane), H);
#pragma unroll
            for (uint32_t k = 0u; k < MO; ++k) {
                const uint32_t m = ro + 4u * k;
                if (m >= MT) break;
                k32(R[k], LDA(W_WO0 + HEAD * hd, 32u, m, 0u), Ob0, LDA(W_WO0 + HEAD * hd, 32u, m, 1u), Ob1);
            }
        }
        SYNC();   // the next group's K / V / O overwrite this one's
    }

    // ---------------------------------------------------------------- MLP
    STAMP(2);
    // the residual tiles through LDS: every wave of a query tile computes the tile's norm
#pragma unroll
    for (uint32_t k = 0u; k < MO; ++k) {
        const uint32_t m = ro + 4u * k;
        if (m >= MT) break;
        uint32_t* d = SM + RX(qt, m) + 4u * lane;
        d[0] = pk2(R[k][0], R[k][1]); d[1] = pk2(R[k][2], R[k][3]); d[2] = pk2(R[k][4], R[k][5]); d[3] = pk2(R[k][6], R[k][7]);
    }
    SYNC();
    u8v Nb[MT];
    {
        f16 rv[MT][8];
#pragma unroll
        for (uint32_t m = 0u; m < MT; ++m) {
            const uint32_t* s = SM + RX(qt, m) + 4u * lane;
#pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) { const hv2 v = upk(s[j]); rv[m][2u * j] = v.x; rv[m][2u * j + 1u] = v.y; }
        }
        f16 h1[2];
#pragma unroll
        for (uint32_t c = 0u; c < 2u; ++c) {
            f16 ht[2];
#pragma unroll
            for (uint32_t tq = 0u; tq < 2u; ++tq) {
                const uint32_t i0 = 4u * tq + c, i1 = i0 + 2u;
                f16 par[2];
#pragma unroll
                for (uint32_t pa = 0u; pa < 2u; ++pa) {
                    f16 pr[KC];
#pragma unroll
                    for (uint32_t b = 0u; b < KC; ++b) {
                        const f16 s0 = rv[2u * b + pa][i0] * rv[2u * b + pa][i0], s1 = rv[2u * b + pa][i1] * rv[2u * b + pa][i1];
                        pr[b] = s0 + s1;
                    }
                    par[pa] = TREE(pr);
                }
                ht[tq] = par[0] + par[1];
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
        for (uint32_t m = 0u; m < MT; ++m) {
            const u4v gain = ld128(W, W_GAIN14 + 2u * (16u * m + 8u * hl));
            u4v nw;
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) {
                const hv2 x = (hv2){rv[m][2u * q], rv[m][2u * q + 1u]};
                const hv2 gg = r2 * upk(gain[q]);
                const hv2 nn = x * gg;
                nw[q] = wid2(pk(nn));
            }
            Nb[m] = g11_bw(nw, H);
        }
#pragma unroll
        for (uint32_t k = 0u; k < MO; ++k) {
            const uint32_t m = ro + 4u * k;
            if (m >= MT) break;
            const u4v tab = ld128(W, W_TABLE16 + 2u * (16u * m + 8u * hl));
            const uint32_t* sx = SM + RX(qt, m) + 4u * lane;   // rv[m] (m is per wave: a register array index would go to scratch)
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) {
                const hv2 ad = upk(sx[q]) + upk(tab[q]);
                R[k][2u * q] = (float)ad.x;
                R[k][2u * q + 1u] = (float)ad.y;
            }
        }
    }
    STAMP(3);
#pragma nounroll
    for (uint32_t gb = 0u; gb < NB / 4u; ++gb) {
        const uint32_t buf = gb & 1u;
#ifdef RR_STAMPS
        if (stampOn && gb < 8u) gstamp[gb] = __builtin_readsteadycounter();
#endif
        {   // branch 4 gb + ro's hidden codes for this query tile
            const uint32_t br = 4u * gb + ro;
            const uint32_t w1 = W_W1_0 + BRANCH * br, b1 = W_B1_0 + BRANCH * br;
#pragma unroll
            for (uint32_t c = 0u; c < 2u; ++c) {
                f8v hh = ldh8(W, b1 + 2u * (16u * c + 8u * hl));
#pragma unroll
                for (uint32_t kt = 0u; kt < MT; kt += 2u) k32(hh, LDA(w1, C, c, kt), Nb[kt], LDA(w1, C, c, kt + 1u), Nb[kt + 1u]);
                u2v hc = (u2v){0u, 0u};
#pragma unroll
                for (int i = 0; i < 8; ++i) hc[i >> 2] |= ld8(W, W_ACT + bh((f16)hh[i])) << (8u * (uint32_t)(i & 3));
                stx2(SM, HX(buf, ro, c, qt), hc, lane);
            }
        }
        SYNC();
        // the four branches' second products onto this wave's tiles, branches in order
#pragma unroll 1
        for (uint32_t bl = 0u; bl < 4u; ++bl) {
            const uint32_t w2 = W_W2_0 + BRANCH * (4u * gb + bl);
            const u8v Hb0 = g11_b(ldx2(SM, HX(buf, bl, 0u, qt), lane), H), Hb1 = g11_b(ldx2(SM, HX(buf, bl, 1u, qt), lane), H);
#pragma unroll
            for (uint32_t k = 0u; k < MO; ++k) {
                const uint32_t m = ro + 4u * k;
                if (m >= MT) break;
                k32(R[k], LDA(w2, 32u, m, 0u), Hb0, LDA(w2, 32u, m, 1u), Hb1);
            }
        }
    }
    SYNC();   // the hidden codes and residual exchange are dead; MG and TB reuse them
    STAMP(4);

    // ---------------------------------------------------------------- outputs
    const uint32_t t = inter(16u * qt + col);
    const int tx = x0 + (int)(t & 7u), ty = y0 + (int)(t >> 3u);
    const bool inside = tx >= 0 && ty >= 0 && tx < ex && ty < ey;
    {
        const uint32_t o = outOff + (uint32_t)(ty * ex + tx) * 32u;
#if KIND == 0 && CX
        const uint32_t ty2 = t >> 3u, tx2 = t & 7u;
        const uint32_t mr = (ty2 >> 1u) < 2u ? 2u * (tx2 >> 1u) + (ty2 >> 1u) : 8u + 2u * (tx2 >> 1u) + (ty2 >> 1u) - 2u;
        const uint32_t blk = 2u * (ty2 & 1u) + (tx2 & 1u);
#endif
#pragma unroll
        for (uint32_t k = 0u; k < MO; ++k) {
            const uint32_t m = ro + 4u * k;
            if (m >= MT) break;
            const u2v tc = codes8(R[k]);
            st64(A, keep(inside, o + (m >> 1u) * plane + 16u * (m & 1u) + 8u * hl), tc);
#if KIND == 0 && CX
            const uint32_t gi = MG_AT + mr * C + blk * (C / 4u) + 4u * m + 2u * hl;
            SM[gi] = tc.x;
            SM[gi + 1u] = tc.y;
#endif
            stx2(SM, TBX(qt, m), tc, lane);
        }
    }
    SYNC();
    STAMP(5);
    if (ro == 0u) {   // the C -> 16 head of this query tile
        f8v hv = ldh8(W, W_BH + 2u * (8u * hl));
#pragma unroll 1
        for (uint32_t kt = 0u; kt < MT; kt += 2u)
            k32(hv, LDA(W_WH, C, 0u, kt), g11_b(ldx2(SM, TBX(qt, kt), lane), H), LDA(W_WH, C, 0u, kt + 1u),
                g11_b(ldx2(SM, TBX(qt, kt + 1u), lane), H));
#if KIND == 0
        // enc's kernel-prediction filter (lanes 0-15; 16-31 hold head channels 8-15)
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
        float wg[9];
#pragma unroll
        for (uint32_t i = 0u; i < 9u; ++i) wg[i] = r16(r16(e[i]) * (float)rr);
        const int x = tx, y = ty;
        const int xm = x - 1 < 0 ? 1 : x - 1, xc = x < 0 ? 1 : (x >= ex ? ex - 2 : x), xp = x + 1 >= ex ? ex - 2 : x + 1;
        const int ym = y - 1 < 0 ? 1 : y - 1, yc = y < 0 ? 1 : (y >= ey ? ey - 2 : y), yp = y + 1 >= ey ? ey - 2 : y + 1;
        const i2v pts[9] = {(i2v){xm, ym}, (i2v){xc, ym}, (i2v){xp, ym}, (i2v){xm, yc}, (i2v){xc, yc},
                            (i2v){xp, yc}, (i2v){xm, yp}, (i2v){xc, yp}, (i2v){xp, yp}};
        f16 acc[3][2];
#pragma unroll
        for (uint32_t k = 0u; k < 8u; ++k) {
            const f4v tv = __ockl_image_load_2D(TS(0), pts[k]);
#pragma unroll
            for (uint32_t c = 0u; c < 3u; ++c) {
                acc[c][k & 1u] = k < 2u ? mulh(tv[c], wg[k]) : mixadd(tv[c] * wg[k], acc[c][k & 1u]);
            }
        }
        const f4v t8 = __ockl_image_load_2D(TS(0), pts[8]);
        float o112[3];
#if CX
        float o120[3];
#endif
#pragma unroll
        for (uint32_t c = 0u; c < 3u; ++c) {
            const f16 f0 = acc[c][0] + acc[c][1];
            const float f = (float)mixadd(t8[c] * wg[8], f0);
#if CX
            const float ar = (float)mulh((float)sig(l[9u + c]), f);
            const float bm = f - ar;
            o120[c] = ar; o112[c] = r16(bm);
#else
            o112[c] = f;
#endif
        }
        if (hl == 0u && inside) {
            __ockl_image_store_2D(TS(24), (i2v){x, y}, (f4v){o112[0], o112[1], o112[2], 0.0f});
#if CX
            __ockl_image_store_2D(TS(32), (i2v){x, y}, (f4v){o120[0], o120[1], o120[2], 0.0f});
#endif
        }
#if CX
        if (hl == 0u)
#pragma unroll
            for (uint32_t c = 0u; c < 3u; ++c) O112[t * 4u + c] = (f16)o112[c];
#endif
#else
#pragma unroll
        for (int i = 0; i < 8; ++i) OUTF[t * 16u + 8u * hl + (uint32_t)i] = (f16)hv[i];
#endif
    }
#if KIND == 0
#if CX
    SYNC();
    {   // the 2x2 merge: channel tiles over all waves; surface 128 by the last wave
        const int X = x0 / 2 + (int)mergeR(col, false), Y = y0 / 2 + (int)mergeR(col, true);
        const uint32_t o = mrgOff + (uint32_t)(Y * hx + X) * 32u;
        for (uint32_t m = wv; m < CX / 16u; m += 16u) {
            f8v acc = ldh8(W, W_BM + 2u * (16u * m + 8u * hl));
#pragma unroll 1
            for (uint32_t ks = 0u; ks < C / 8u; ++ks)
                k32(acc, LDA(W_WM, 4u * C, m, 2u * ks), lds8b(SM, MG_AT + 8u * ks, C, col), LDA(W_WM, 4u * C, m, 2u * ks + 1u),
                    lds8b(SM, MG_AT + 8u * ks + 4u, C, col));
            const u2v q = codes8(acc);
            st64(A, keep(X >= 0 && Y >= 0 && X < hx && Y < hy, o + (m >> 1u) * lplane + 16u * (m & 1u) + 8u * hl), q);
        }
        if (wv == 15u && lane < 16u) {
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
#endif
#else
    // ---------------------------------------------------------------- the decoders' output filter (rr_decfilter.glsl)
    SYNC();
    if (tid < 100u) {
        const int xi = clampTap(x0 - 1 + (int)(tid % 10u), ex), yi = clampTap(y0 - 1 + (int)(tid / 10u), ey);
        const float iw = __builtin_bit_cast(float, p[34]), ih = __builtin_bit_cast(float, p[35]);
        const float cu = __builtin_fmaf(iw, (float)xi, iw * 0.5f), cv = __builtin_fmaf(ih, (float)yi, ih * 0.5f);
        const f4v bl = __ockl_image_sample_lod_2D(TS(0), TS(48), (__attribute__((ext_vector_type(2))) float){cu, cv}, 0.0f);
#pragma unroll
        for (uint32_t c = 0u; c < 3u; ++c) TAP[tid * 4u + c] = (f16)bl[c];
    }
    SYNC();
    if (tid < 128u) {
        const uint32_t tt = tid >> 1u, hf = tid & 1u;
        const int x = x0 + (int)(tt & 7u), y = y0 + (int)(tt >> 3u);
        if (x < 0 || y < 0 || x >= ex || y >= ey) return;
        f16 av[10];
#pragma unroll
        for (uint32_t c = 0u; c < 10u; ++c) av[c] = OUTF[tt * 16u + c];
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
#endif
#ifdef RR_STAMPS
    SYNC();
    STAMP(6);
    if (stampOn && TID() == 0u)
        for (int i = 0; i < 8; ++i) {
            st64(A, a.arenaBytes - 128u + 8u * (uint32_t)i, (u2v){(uint32_t)stamps[i], (uint32_t)(stamps[i] >> 32)});
            st64(A, a.arenaBytes - 64u + 8u * (uint32_t)i, (u2v){(uint32_t)gstamp[i], (uint32_t)(gstamp[i] >> 32)});
        }
#endif
}
