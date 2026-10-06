// Token-lane rrlite Swin block for RDNA4 native FP8 (SWIN_A8), selected with SWIN_TOKEN_LANES.
//
// swin_block.h computes every GEMM as activations x weights: the result has lane = channel column and
// VGPR = token row, so each result is transposed through LDS (byte stores, barriers) before it can be the
// next GEMM's operand. Here the weights are the first WMMA operand and the activations the second, which
// gives the transposed result: lane = token, VGPR i = row 8 half + i of the output. A gfx12 operand is this
// half's 8 K values, so an encoded result IS the next GEMM's operand and stays in registers. The products
// and the K slot of every product are the same as in swin_block.h, so the values are the same (operand
// roles swapped: bit-identical on an RX 9070 XT for the FP8 and the f16 WMMA).
//
// One wave owns one 4x4 window (16 tokens) for the whole block: norms, attention and the MLP need no LDS
// and no barriers. Only the patch expand (decoders) and the patch merge (encoders) cross windows and go
// through an e4m3 byte image in LDS.
//
// A half-wave owns the channels whose bit 2 equals the half (dword 4 t + 2 g + half of tile t, g = 0, 1):
// both halves of a k32 operand's K slots are then two of the lane's own code dwords.
#pragma once

enum : int { TOK_ID, TOK_PI, TOK_X, TOK_NU, TOK_PE };
// A GEMM's weight tiles as in GemmDesc (slot dst + ((kc * 2 + s) * NT + nt) * 16 + r holds the 16 K slots of
// row r of tile nt), with the raw weight column of (nt, r) chosen by perm:
//   TOK_ID  16 nt + r                  (Q: the attention's K order)
//   TOK_PI  gperm(kslot(nt, r))        (V, fc1: row r is K slot r of step nt of the GEMM that consumes it)
//   TOK_X   gperm(tok_xnat(nt, r))     (output projection, fc2: the x layout)
//   TOK_NU  gperm(16 nt + r)           (patch merge: natural channel order, 8 contiguous output bytes)
//   TOK_PE  col0 + gperm(16 nt + r) over weight blocks of blkcols columns / blkbytes bytes (patch expand)
struct TokDesc
{
    int dst, base, Ks, Ns, KC, NT, perm, col0, blkcols, blkbytes;
};
enum : int { TC_X, TC_PI, TC_NU, TC_ATT };
// Per-channel f16 vectors as 8-value slots: slot dst + 2 j + half holds rows 8 half + i of tile j in the
// layout of kind (TC_ATT: j = query token, row = key token, the transposed attention bias)
struct TokConst
{
    int dst, kind, base;
};

// x layout: natural channel of row r (= 8 half + i) of tile t
__device__ __forceinline__ int tok_xnat(int t, int r)
{
    return 16 * t + 8 * ((r >> 2) & 1) + 4 * (r >> 3) + (r & 3);
}

template <int C, int NH, int NPM, int CIN> struct TokLayout
{
    using L = SwinLayout<C, NH, NPM, CIN>;
    static constexpr int NT = C / 16, NC = C / 8, NTP = NPM / 16;
    // weight image (slots)
    __host__ __device__ static constexpr int qv(int h, int q) { return (2 * h + q) * 2 * C; }
    static constexpr int WO = 4 * C * NH;
    static constexpr int W1 = WO + 2 * NH * C;
    __host__ __device__ static constexpr int w1(int c) { return W1 + 4 * C * c; }
    __host__ __device__ static constexpr int w2(int c) { return W1 + 4 * C * c + 2 * C; }
    static constexpr int PM = W1 + 4 * C * NC;
    static constexpr int PE = PM + NPM * C / 4;
    __host__ __device__ static constexpr int pe(int q) { return PE + q * (CIN * C / 16); }
    static constexpr int TOTAL = PE + CIN * C / 4;
    static constexpr int NDESC = 2 * NH + 1 + 2 * NC + (NPM ? 1 : 0) + (CIN ? 4 : 0);
    // constant image (slots of 8 f16)
    static constexpr int CBO = 0, CG1 = 2 * NT, CG2 = 4 * NT, CB2 = 6 * NT, CB1 = 8 * NT;
    static constexpr int CATT = CB1 + 4 * NC, CPMB = CATT + 32 * NH, CPEB = CPMB + 2 * NTP;
    static constexpr int CTOTAL = CPEB + (CIN ? 8 * NT : 0);
    static constexpr int NCONST = 4 + NC + NH + (NPM ? 1 : 0) + (CIN ? 4 : 0);
};

template <int C, int NH, int NPM, int CIN> struct TokTables
{
    TokDesc d[TokLayout<C, NH, NPM, CIN>::NDESC];
    TokConst c[TokLayout<C, NH, NPM, CIN>::NCONST];
};

template <int C, int NH, int NPM, int CIN> constexpr TokTables<C, NH, NPM, CIN> make_tok_tables()
{
    using L = SwinLayout<C, NH, NPM, CIN>;
    using K = TokLayout<C, NH, NPM, CIN>;
    TokTables<C, NH, NPM, CIN> r{};
    int n = 0;
    for (int h = 0; h < NH; ++h)
    {
        r.d[n++] = TokDesc{K::qv(h, 0), L::head(h), 1024, 512, C / 32, 2, TOK_ID, 0, 0, 0};
        r.d[n++] = TokDesc{K::qv(h, 1), L::head(h) + 32 * C, 1024, 512, C / 32, 2, TOK_PI, 0, 0, 0};
    }
    r.d[n++] = TokDesc{K::WO, L::head(0) + 64 * C + 512, L::HS, 512, NH, C / 16, TOK_X, 0, 0, 0};
    for (int c = 0; c < K::NC; ++c)
    {
        r.d[n++] = TokDesc{K::w1(c), L::b1(c), 512, 16 * C, C / 32, 2, TOK_PI, 0, 0, 0};
        r.d[n++] = TokDesc{K::w2(c), L::b1(c) + 32 * C + 64, 0, 512, 1, C / 16, TOK_X, 0, 0, 0};
    }
    if (NPM)
        r.d[n++] = TokDesc{K::PM, L::PM0, 512, 64 * C, C / 8, NPM / 16, TOK_NU, 0, 0, 0};
    for (int q = 0; q < (CIN ? 4 : 0); ++q)
        r.d[n++] = TokDesc{K::pe(q), 0, 512, 16 * CIN, CIN / 32, C / 16, TOK_PE, q * C, 4 * C / NH, CIN * 4 * C / NH};
    int m = 0;
    r.c[m++] = TokConst{K::CBO, TC_X, L::BO};
    r.c[m++] = TokConst{K::CG1, TC_X, L::G1};
    r.c[m++] = TokConst{K::CG2, TC_X, L::G2};
    r.c[m++] = TokConst{K::CB2, TC_X, L::B2};
    for (int c = 0; c < K::NC; ++c)
        r.c[m++] = TokConst{K::CB1 + 4 * c, TC_PI, L::b1(c) + 32 * C};
    for (int h = 0; h < NH; ++h)
        r.c[m++] = TokConst{K::CATT + 32 * h, TC_ATT, L::head(h) + 64 * C};
    if (NPM)
        r.c[m++] = TokConst{K::CPMB, TC_NU, L::PMB};
    for (int q = 0; q < (CIN ? 4 : 0); ++q)
        r.c[m++] = TokConst{K::CPEB + 2 * K::NT * q, TC_NU, 4 * C * CIN + 2 * q * C};
    return r;
}

__device__ __forceinline__ void expand_tok_weights(const uint8_t* w, wslot* w16, const TokDesc* descs, int ndesc, int idx)
{
    int d = 0;
    while (d + 1 < ndesc && idx >= descs[d + 1].dst)
        ++d;
    const TokDesc g = descs[d];
    const int local = idx - g.dst;
    if (local >= g.KC * 2 * g.NT * 16)
        return;
    const int r = local & 15, rest = local >> 4;
    const int nt = rest % g.NT, s = (rest / g.NT) & 1, kc = rest / g.NT / 2;
    int n = 16 * nt + r, base = g.base;
    if (g.perm == TOK_PI)
        n = gperm(kslot(nt, r));
    else if (g.perm == TOK_X)
        n = gperm(tok_xnat(nt, r));
    else if (g.perm == TOK_NU || g.perm == TOK_PE)
        n = gperm(16 * nt + r);
    if (g.perm == TOK_PE)
    {
        const int gc = g.col0 + n;
        base += (gc / g.blkcols) * g.blkbytes;
        n = gc % g.blkcols;
    }
    u4v_t v;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b)
            word |= (uint32_t)w[woff(base, g.Ks, g.Ns, 32 * kc + kslot(s, 4 * j + b), n)] << (8 * b);
        v[j] = word;
    }
    w16[idx] = v;
}

__device__ __forceinline__ void expand_tok_consts(const uint8_t* w, h8* c16, const TokConst* cs, int ncs, int idx)
{
    int d = 0;
    while (d + 1 < ncs && idx >= cs[d + 1].dst)
        ++d;
    const TokConst g = cs[d];
    const int local = idx - g.dst, hf = local & 1, j = local >> 1;
    h8 v;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const int r = 8 * hf + i;
        int off;
        if (g.kind == TC_X)
            off = 2 * gperm(tok_xnat(j, r));
        else if (g.kind == TC_PI)
            off = 2 * gperm(kslot(j, r));
        else if (g.kind == TC_NU)
            off = 2 * gperm(16 * j + r);
        else
        {
            // the attention bias of (query j, key r)
            const int bl = 4 * (j & 7) + ((r & 7) >> 1), word = (j >> 3) + 2 * (r >> 3);
            off = 2 * (8 * bl + 2 * word + (r & 1));
        }
        v[i] = hload(w, g.base + off);
    }
    c16[idx] = v;
}

// ---------------------------------------------------------------- register helpers
typedef float tok_f2v __attribute__((ext_vector_type(2)));

// two f32 -> f16 (round to nearest even) packed into one register
__device__ __forceinline__ hv2 tok_pk16(float a, float b)
{
    return (hv2){(half_t)a, (half_t)b};
}

// two e4m3 codes of a word (bytes 0, 1 or 2, 3) as f16. Equal to e4m3_to_half for every code except the NaN
// codes 0x7f / 0xff, which the conversion decodes as NaN (e4m3_to_half: +-480). The FP8 WMMAs read the
// same bytes with the hardware's decoding, and no encoder here stores those codes for a non-NaN value.
template <bool UPPER> __device__ __forceinline__ hv2 tok_dec2(uint32_t word)
{
    const tok_f2v f = __builtin_amdgcn_cvt_pk_f32_fp8((int)word, UPPER);
    return tok_pk16(f[0], f[1]);
}

// the same lane of the other half-wave
__device__ __forceinline__ hv2 tok_other(hv2 v)
{
    return __builtin_bit_cast(hv2, xor_lane<16>(__builtin_bit_cast(uint32_t, v)));
}

__device__ __forceinline__ T16 tok_t16(h8 c)
{
    T16 t;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        t.set_pair(k, (hv2){c[2 * k], c[2 * k + 1]});
    return t;
}

// e4m3 codes of a tile's 8 rows: this half's 8 K slots of an operand
__device__ __forceinline__ gop tok_codes(const T16& t)
{
    return (u2v){codes4(t.pair(0), t.pair(1)), codes4(t.pair(2), t.pair(3))};
}

// 1 / sqrt(sum of squares + eps) of a token's C values, with the summation tree of norm1 and norm2 of
// swin_block.h (both are this tree in the x layout): per (g, element) the squares of a dword's two pairs,
// the 32-channel groups, the two 16-channel halves of a group, then the half-waves, g and the elements
template <int NT> __device__ __forceinline__ half_t tok_rsq(const hv2 (&v)[NT][4])
{
    constexpr int NPL = NT / 2;
    hv2 u[2];
#pragma unroll
    for (int g = 0; g < 2; ++g)
    {
        hv2 ws[2];
#pragma unroll
        for (int ww = 0; ww < 2; ++ww)
        {
            hv2 pp[NPL];
#pragma unroll
            for (int pl = 0; pl < NPL; ++pl)
            {
                const hv2 a = v[2 * pl + ww][2 * g], b = v[2 * pl + ww][2 * g + 1];
                pp[pl] = (hv2)(a * a) + (hv2)(b * b);
            }
            if constexpr (NPL == 4)
                ws[ww] = (pp[0] + pp[1]) + (pp[2] + pp[3]);
            else if constexpr (NPL == 3)
                ws[ww] = (pp[0] + pp[1]) + pp[2];
            else
                ws[ww] = pp[0] + pp[1];
        }
        const hv2 a = ws[0] + ws[1];
        u[g] = a + tok_other(a);
    }
    const hv2 s = u[0] + u[1];
    return rsqrt16(s[0] + s[1]);
}

template <int C, int NH, int NPM, bool TUBE, int CIN>
__device__ __forceinline__ void swin_tok_block(const CommonParams& p, const TubeParams* tp, const wslot* w16, const h8* c16,
                                               const int bx, const int by, const int gx)
{
    using K = TokLayout<C, NH, NPM, CIN>;
    constexpr int NT = C / 16, NKC = C / 32, NC = C / 8;
    static_assert(C % 32 == 0, "32-channel groups");

    const int lane = threadIdx.x, wv = threadIdx.z;
    const int l16 = lane & 15, hf = lane >> 4;
    const int T = 16 * wv + l16; // this lane's token: window wv, position l16
    // The lane's bases into the two images, hidden from constant folding: every tile and constant is then
    // one load at a fixed offset from a register (folded, each load rebuilt the image's address first).
    uintptr_t chv = (uintptr_t)(c16 + hf), wlv = (uintptr_t)((const gop*)w16 + 2 * l16 + hf);
    asm("" : "+v"(chv));
    asm("" : "+v"(wlv));
    const h8* const ch = (const h8*)chv;
    // row l16 of a weight tile: this half's 8 K slots
    const gop* const wl = (const gop*)wlv;
    auto wt = [&](int dst, int nt_count, int kc, int s, int nt) { return wl[2 * (dst + ((kc * 2 + s) * nt_count + nt) * 16)]; };

    if constexpr (TUBE)
    {
        if (tp->in_flags != nullptr)
        {
            if (lane <= 3 && wv == 0)
            {
                int nx = bx + (lane & 1) + tp->r71, ny = by + (lane >> 1) + tp->r72;
                if (nx >= 0 && nx < tp->r69 && ny >= 0 && ny < tp->r70)
                {
                    uint8_t* f = (uint8_t*)tp->in_flags + (size_t)(ny * tp->r69 + nx) * 32;
                    // bounded spin: a missing producer must never hang the GPU
                    for (int it = 0; it < (1 << 22); ++it)
                    {
                        if (__hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0)
                            break;
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }
    }

    // ------------------------------------------------ block input x0 (encoders: the fp8 input; decoders: skip)
    const int X = 8 * bx - p.sx + tok_x(T), Y = 8 * by - p.sy + tok_y(T);
    const int Xm = mirror(X, p.tw), Ym = mirror(Y, p.th);
    uint32_t inw[NT][2];
    {
        const uint8_t* const src = CIN ? p.p32 : p.in;
#pragma unroll
        for (int pl = 0; pl < NKC; ++pl)
        {
            const uint8_t* b = src + ((size_t)(pl * p.th + Ym) * p.tw + Xm) * 32 + 4 * hf;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                inw[2 * pl + (j >> 1)][j & 1] = input_word((const uint32_t*)(b + 8 * j));
        }
    }

    hv2 v[NT][4]; // x0, then x1, as f16 pairs: [tile][2 g + pair of the dword]
    if constexpr (CIN > 0)
    {
        // patch expand: lanes = the block's 16 low-res tokens, wave = quadrant; the codes of q8(expand) go to
        // their child tokens' rows of a byte image, where the windows' waves pick them up
        __shared__ __attribute__((aligned(16))) uint8_t E8[64 * C];
        const int W2 = p.tw / 2, H2 = p.th / 2;
        const int MX = mirror((8 * bx - p.sx) / 2 + (l16 & 3), W2), MY = mirror((8 * by - p.sy) / 2 + (l16 >> 2), H2);
        gop pa[CIN / 32][2];
#pragma unroll
        for (int kc = 0; kc < CIN / 32; ++kc)
        {
            const uint32_t* b = (const uint32_t*)(p.in + ((size_t)(kc * H2 + MY) * W2 + MX) * 32 + 4 * hf);
#pragma unroll
            for (int s = 0; s < 2; ++s)
                pa[kc][s] = (u2v){input_word(b + 2 * s), input_word(b + 4 + 2 * s)};
        }
        const int tx = 2 * (l16 & 3) + (wv & 1), ty = 2 * (l16 >> 2) + (wv >> 1);
        const int Tc = (tx & 3) + 4 * (ty & 3) + 16 * (tx >> 2) + 32 * (ty >> 2);
        const int pew = K::pe(0) + wv * (CIN * C / 16);
#pragma unroll
        for (int ct = 0; ct < NT; ++ct)
        {
            T16 e = tok_t16(ch[K::CPEB + 2 * (NT * wv + ct)]);
#pragma unroll
            for (int kc = 0; kc < CIN / 32; ++kc)
                k32(e, wt(pew, NT, kc, 0, ct), pa[kc][0], wt(pew, NT, kc, 1, ct), pa[kc][1]);
            *(u2v*)(E8 + Tc * C + 16 * ct + 8 * hf) = tok_codes(e);
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int g = 0; g < 2; ++g)
            {
                const uint32_t ew = *(const uint32_t*)(E8 + T * C + 16 * t + 8 * g + 4 * hf);
                v[t][2 * g] = tok_dec2<false>(ew) + tok_dec2<false>(inw[t][g]);
                v[t][2 * g + 1] = tok_dec2<true>(ew) + tok_dec2<true>(inw[t][g]);
            }
    }
    else
    {
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int g = 0; g < 2; ++g)
            {
                v[t][2 * g] = tok_dec2<false>(inw[t][g]);
                v[t][2 * g + 1] = tok_dec2<true>(inw[t][g]);
            }
    }

    // residual of the output projection: f16(x0 + b_o)
    hv2 res[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t)
    {
        const h8 bo = ch[K::CBO + 2 * t];
#pragma unroll
        for (int k = 0; k < 4; ++k)
            res[t][k] = v[t][k] + (hv2){bo[2 * k], bo[2 * k + 1]};
    }

    // ------------------------------------------------ norm1: h1 codes, the K operands of Q and V
    gop hop[NKC][2];
    {
        const half_t rn = tok_rsq<NT>(v);
        const hv2 rn2 = {rn, rn};
        uint32_t c1[NT][2];
#pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            const h8 g1 = ch[K::CG1 + 2 * t];
#pragma unroll
            for (int g = 0; g < 2; ++g)
                c1[t][g] = codes4(v[t][2 * g] * (hv2)(rn2 * (hv2){g1[4 * g], g1[4 * g + 1]}),
                                  v[t][2 * g + 1] * (hv2)(rn2 * (hv2){g1[4 * g + 2], g1[4 * g + 3]}));
        }
#pragma unroll
        for (int kc = 0; kc < NKC; ++kc)
#pragma unroll
            for (int s = 0; s < 2; ++s)
                hop[kc][s] = (u2v){c1[2 * kc][s], c1[2 * kc + 1][s]};
    }

    // ------------------------------------------------ attention per head: O codes, the K operands of the projection
    gop oc[NH][2];
#pragma unroll
    for (int hd = 0; hd < NH; ++hd)
    {
        // Q transposed (lane = token, rows = Q's 32 columns), V as in swin_block.h (lane = column, rows = tokens)
        T16 qt[2], vt[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
        {
            qt[nt] = t16_splat((half_t)0.0f);
            vt[nt] = t16_splat((half_t)0.0f);
        }
#pragma unroll
        for (int kc = 0; kc < NKC; ++kc)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                k32(qt[nt], wt(K::qv(hd, 0), 2, kc, 0, nt), hop[kc][0], wt(K::qv(hd, 0), 2, kc, 1, nt), hop[kc][1]);
                k32(vt[nt], hop[kc][0], wt(K::qv(hd, 1), 2, kc, 0, nt), hop[kc][1], wt(K::qv(hd, 1), 2, kc, 1, nt));
            }
        const gop q0 = tok_codes(qt[0]), q1 = tok_codes(qt[1]);
        // S transposed: lane = query token, rows = key tokens
        T16 S = tok_t16(ch[K::CATT + 32 * hd + 2 * l16]);
        k32(S, q0, q0, q1, q1);
        const half_t cs = f16(0.0972222164273262f / 5.656854152679443f);
        const half_t cl = (half_t)0.55615234375f;
        const half_t p1 = f16(0.92730712890625f), p0 = (half_t)1.375f;
        hv2 wg[4], v1[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            hv2 tv = S.pair(k) * (hv2){cs, cs};
            tv = __builtin_elementwise_max(__builtin_elementwise_min(tv, (hv2){cl, cl}), (hv2){-cl, -cl});
            const hv2 e1 = __builtin_elementwise_fma(tv, -tv, (hv2){p1, p1});
            const hv2 poly = __builtin_elementwise_fma(tv, e1, (hv2){p0, p0});
            // the pair is the (even, odd) key word of the exponent trick
            wg[k] = __builtin_bit_cast(hv2, (__builtin_bit_cast(uint32_t, poly) << 5) + 0x7FF88000u);
            v1[k] = wg[k] + tok_other(wg[k]); // keys c and c ^ 8
        }
        const hv2 v3 = (v1[0] + v1[1]) + (v1[2] + v1[3]); // c ^ 2, then c ^ 4
        const half_t sum = v3[0] + v3[1];                  // c ^ 1
        const half_t rinv = f16(1.0f / (float)sum);
        hv2 pw[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
            pw[k] = wg[k] * (hv2){rinv, rinv};
        const sop pb = __builtin_bit_cast(sop, pw);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
        {
            hv2 vp[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                vp[k] = vt[nt].pair(k);
            const f8v o = wmma(__builtin_bit_cast(sop, vp), pb, splat(0.0f));
            oc[hd][nt] = (u2v){codes4(tok_pk16(o[0], o[1]), tok_pk16(o[2], o[3])), codes4(tok_pk16(o[4], o[5]), tok_pk16(o[6], o[7]))};
        }
    }

    // ------------------------------------------------ output projection + residual
    T16 x[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int k = 0; k < 4; ++k)
            x[t].set_pair(k, res[t][k]);
#pragma unroll
    for (int kc = 0; kc < NH; ++kc)
#pragma unroll
        for (int t = 0; t < NT; ++t)
            k32(x[t], wt(K::WO, NT, kc, 0, t), oc[kc][0], wt(K::WO, NT, kc, 1, t), oc[kc][1]);

    // ------------------------------------------------ norm2: h2 codes, the K operands of fc1
    gop fop[NKC][2];
    {
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int k = 0; k < 4; ++k)
                v[t][k] = x[t].pair(k);
        const half_t rn = tok_rsq<NT>(v);
        const hv2 rn2 = {rn, rn};
        uint32_t c2[NT][2];
#pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            const h8 g2 = ch[K::CG2 + 2 * t], b2 = ch[K::CB2 + 2 * t];
#pragma unroll
            for (int g = 0; g < 2; ++g)
                c2[t][g] = codes4(v[t][2 * g] * (hv2)(rn2 * (hv2){g2[4 * g], g2[4 * g + 1]}),
                                  v[t][2 * g + 1] * (hv2)(rn2 * (hv2){g2[4 * g + 2], g2[4 * g + 3]}));
#pragma unroll
            for (int k = 0; k < 4; ++k)
                x[t].set_pair(k, v[t][k] + (hv2){b2[2 * k], b2[2 * k + 1]});
        }
#pragma unroll
        for (int kc = 0; kc < NKC; ++kc)
#pragma unroll
            for (int s = 0; s < 2; ++s)
                fop[kc][s] = (u2v){c2[2 * kc][s], c2[2 * kc + 1][s]};
    }

    // ------------------------------------------------ MLP: fc1 transposed, its codes are fc2's operands
#pragma unroll 1
    for (int c = 0; c < NC; ++c)
    {
        const int w1 = K::W1 + 4 * C * c;
        // the round's operands first: 2 bias slots, fc1's and fc2's weight tiles
        h8 zb[2];
        gop wa[NKC][2][2], wb[2][NT];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            zb[nt] = ch[K::CB1 + 4 * c + 2 * nt];
#pragma unroll
        for (int kc = 0; kc < NKC; ++kc)
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    wa[kc][s][nt] = wt(w1, 2, kc, s, nt);
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int t = 0; t < NT; ++t)
                wb[s][t] = wt(w1 + 2 * C, NT, 0, s, t);
#ifndef SWIN_TOK_NO_LOAD_SYNC
        // one wait for the whole batch here instead of one before each WMMA
        asm("" : "+v"(wb[1][NT - 1]));
#endif
        T16 z[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            z[nt] = tok_t16(zb[nt]);
#pragma unroll
        for (int kc = 0; kc < NKC; ++kc)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                k32(z[nt], wa[kc][0][nt], fop[kc][0], wa[kc][1][nt], fop[kc][1]);
        gop gz[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
        {
            hv2 gl[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const hv2 zv = z[nt].pair(k);
                // minimum / maximum instead of swin_block.h's minnum / maxnum: they differ only for a NaN z, and
                // then the product with z below is NaN either way (no canonicalising instruction per pair)
                const hv2 cz = __builtin_elementwise_maximum(__builtin_elementwise_minimum(zv, (hv2){2.0f16, 2.0f16}),
                                                             (hv2){-2.0f16, -2.0f16});
                const hv2 az = __builtin_bit_cast(hv2, (u16x2)(__builtin_bit_cast(u16x2, cz) & (unsigned short)0x7fff));
                const hv2 inner = (hv2){f16(0.41216981f), f16(0.41216981f)} -
                                  (hv2)((hv2){f16(0.08108133f), f16(0.08108133f)} * az);
                gl[k] = zv * (hv2)((hv2){0.5f16, 0.5f16} + (hv2)(cz * inner));
            }
            gz[nt] = (u2v){codes4(gl[0], gl[1]), codes4(gl[2], gl[3])};
        }
#pragma unroll
        for (int t = 0; t < NT; ++t)
            k32(x[t], wb[0][t], gz[0], wb[1][t], gz[1]);
    }

    // ------------------------------------------------ output codes
    uint32_t co[NT][2];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int g = 0; g < 2; ++g)
            co[t][g] = codes4(x[t].pair(2 * g), x[t].pair(2 * g + 1));
    if (X >= 0 && X < p.tw && Y >= 0 && Y < p.th)
    {
#pragma unroll
        for (int pl = 0; pl < NKC; ++pl)
        {
            uint32_t* b = (uint32_t*)(p.out + ((size_t)(pl * p.th + Y) * p.tw + X) * 32 + 4 * hf);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                b[2 * j] = co[2 * pl + (j >> 1)][j & 1];
        }
    }

    // ------------------------------------------------ patch merge: 2x2 tokens (4C) -> NPM, half resolution
    if constexpr (NPM > 0)
    {
        // lanes = the block's 16 merged tokens; the merge reads the output codes from a byte image
        constexpr int XB = C + 4, NTP = NPM / 16;
        __shared__ __attribute__((aligned(16))) uint8_t U[64 * XB];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int g = 0; g < 2; ++g)
                *(uint32_t*)(U + T * XB + 16 * t + 8 * g + 4 * hf) = co[t][g];
        __syncthreads();
        if (wv < NTP)
        {
            const int mx = l16 & 3, my = l16 >> 2;
            // K slot 8 half + 4 rr + b of step (kc, s) is value k = 32 kc + 16 rr + 8 s + 4 half + b of the merged
            // token's 4C inputs: channel k % C of its source token k / C. The four source tokens are consecutive
            // in x and 4 apart in y, and 4 half never crosses a token, so every run of four codes is at a fixed
            // offset from the lane's first source token.
            const uint8_t* const ub = U + ((2 * mx & 3) + 4 * (2 * my & 3) + 16 * (mx >> 1) + 32 * (my >> 1)) * XB + 4 * hf;
            gop as[C / 8][2];
#pragma unroll
            for (int kc = 0; kc < C / 8; ++kc)
#pragma unroll
                for (int s = 0; s < 2; ++s)
                {
                    uint32_t run4[2];
#pragma unroll
                    for (int rr = 0; rr < 2; ++rr)
                    {
                        const int k0 = 32 * kc + 16 * rr + 8 * s, q = k0 / C;
                        run4[rr] = *(const uint32_t*)(ub + ((q & 1) + 4 * (q >> 1)) * XB + k0 % C);
                    }
                    as[kc][s] = (u2v){run4[0], run4[1]};
                }
            const int W2 = p.tw / 2, H2 = p.th / 2;
            const int MX = (8 * bx - p.sx) / 2 + mx, MY = (8 * by - p.sy) / 2 + my;
            const bool inside = MX >= 0 && MX < W2 && MY >= 0 && MY < H2;
            // output tiles u = wv, wv + 4, ...: rows = 16 natural channels, 8 contiguous bytes per half-wave
#pragma unroll
            for (int j = 0; j < (NTP + 3) / 4; ++j)
            {
                const int u = wv + 4 * j;
                if (u < NTP)
                {
                    T16 y = tok_t16(ch[K::CPMB + 2 * u]);
#pragma unroll
                    for (int kc = 0; kc < C / 8; ++kc)
                        k32(y, wt(K::PM, NTP, kc, 0, u), as[kc][0], wt(K::PM, NTP, kc, 1, u), as[kc][1]);
                    const u2v code = tok_codes(y);
                    if (inside)
                        *(u2v*)(p.p48 + ((size_t)((u >> 1) * H2 + MY) * W2 + MX) * 32 + 16 * (u & 1) + 8 * hf) = code;
                }
            }
        }
    }

    if constexpr (TUBE)
    {
        if (tp->out_flags != nullptr)
        {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __syncthreads();
            if (lane == 0 && wv == 0)
                __hip_atomic_store(tp->out_flags + (size_t)(by * gx + bx) * 32, (uint8_t)1, __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

#if defined(SWIN_TOK_MAX_VGPR)
#define SWIN_TOK_VGPR_ATTR __attribute__((amdgpu_num_vgpr(SWIN_TOK_MAX_VGPR)))
#else
#define SWIN_TOK_VGPR_ATTR
#endif

// Module boilerplate: the weight and constant images, their prep kernel, and the 4-wave main kernel
#define SWIN_TOK_MODULE(NAME, C, NH, NPM, TUBE, PARAMS, CIN)                                                            \
    using NAME##_K = TokLayout<C, NH, NPM, CIN>;                                                                        \
    static_assert(SWIN_PREP_SLOTS == 0, "one prepared image");                                                         \
    __device__ wslot g_w16[NAME##_K::TOTAL];                                                                            \
    __device__ h8 g_c16[NAME##_K::CTOTAL];                                                                              \
    __constant__ TokTables<C, NH, NPM, CIN> g_tok = make_tok_tables<C, NH, NPM, CIN>();                                 \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4;                                              \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks =                                             \
        ((NAME##_K::TOTAL > NAME##_K::CTOTAL ? NAME##_K::TOTAL : NAME##_K::CTOTAL) + 127) / 128;                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_at = SWIN_PREP_KEY_AT;                             \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_slots = 0;                                       \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PARAMS p)                                             \
    {                                                                                                                   \
        const int idx = blockIdx.x * 128 + threadIdx.x;                                                                 \
        const uint8_t* w = ((const CommonParams*)&p)->w;                                                                \
        if (idx < NAME##_K::TOTAL)                                                                                      \
            expand_tok_weights(w, g_w16, g_tok.d, NAME##_K::NDESC, idx);                                               \
        if (idx < NAME##_K::CTOTAL)                                                                                     \
            expand_tok_consts(w, g_c16, g_tok.c, NAME##_K::NCONST, idx);                                               \
    }                                                                                                                   \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_grid_x = 0;                                               \
    extern "C" __global__ void __launch_bounds__(128) SWIN_TOK_VGPR_ATTR NAME(PARAMS p)                                 \
    {                                                                                                                   \
        const CommonParams& cp = *(const CommonParams*)&p;                                                              \
        const int gx = (cp.tw + cp.sx + 7) / 8;                                                                         \
        swin_tok_block<C, NH, NPM, TUBE, CIN>(cp, (const TubeParams*)&p, g_w16, g_c16, blockIdx.x, blockIdx.y, gx);     \
    }
