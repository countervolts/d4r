// Templated rrlite Swin block (64 tokens = 4 windows of 4x4 per block, NW waves = heads of 32 dims).
//   C    channels (64 / 96 / 128), NW waves (2 or 4)
//   NPM  patch-merge output channels (0 = none): 2x2 merge of the block output, 4C -> NPM
//   TUBE 88-byte parameter block with the cross-launch flag protocol
//   CIN  patch-expand input channels (0 = none): decoder input = f16(q8(expand(low-res CIN)) + skip)
#pragma once
#include "swin_common.h"
#include <type_traits>
#define SWIN_PRAGMA_IMPL(x) _Pragma(#x)
#define SWIN_PRAGMA(x) SWIN_PRAGMA_IMPL(x)

template <int C, int NW, int NPM, int CIN = 0> struct SwinLayout
{
    static constexpr int PREB = CIN ? 4 * C * CIN + 8 * C : 0; // patch-expand weights + bias first
    static constexpr int NTE = CIN ? 4 * C / NW / 16 : 0;
    static constexpr int NPL = C / 32, NT = C / 16, NT8 = C / 8, INNER = 32 * NW, NC = C / 8;
    static constexpr int HS = 96 * C + 512;
    static constexpr int G1 = PREB, E = PREB + 2 * C + NW * HS, BO = E, G2 = E + 2 * C, M0 = E + 4 * C;
    static constexpr int B2 = M0 + 64 * C + 64;
    __host__ __device__ static constexpr int head(int h) { return PREB + 2 * C + HS * h; }
    __host__ __device__ static constexpr int b1(int c) { return c == 0 ? M0 : M0 + 66 * C + 64 + (64 * C + 64) * (c - 1); }
    static constexpr int PM0 = M0 + 66 * C + 64 + (64 * C + 64) * (NC - 1);
    static constexpr int PMB = PM0 + 4 * C * NPM;
    // weight image (slots of 16 K values, see expand_weights)
    __host__ __device__ static constexpr int qv(int h, int q) { return (2 * h + q) * 2 * C; }
    static constexpr int WO = 4 * C * NW;
    static constexpr int W1 = WO + 2 * NW * C;
    __host__ __device__ static constexpr int w1(int c) { return W1 + 4 * C * c; }
    __host__ __device__ static constexpr int w2(int c) { return W1 + 4 * C * c + 2 * C; }
    static constexpr int PM = W1 + 4 * C * NC;
    __host__ __device__ static constexpr int pm(int g) { return PM + 8 * C * g; }
    static constexpr int PE = PM + (NPM / 32) * 8 * C;
    __host__ __device__ static constexpr int pe(int w) { return PE + w * (CIN / 32) * 2 * NTE * 16; }
    static constexpr int TOTAL = PE + (CIN ? NW * (CIN / 32) * 2 * NTE * 16 : 0);
    static constexpr int NDESC = 2 * NW + 1 + 2 * NC + NPM / 32 + (CIN ? NW : 0);
};

// Start (in halves) of token T's row in the output staging buffer: stride C + 4 plus a shift by bits 4 and 5
// of T. The patch merge's lanes read 8 bytes each from 16 tokens that differ in bits 1, 3, 4 and 5 (plus a
// constant); this placement puts those 16 reads on 16 distinct 8-byte bank pairs for C = 64, 96 and 128.
template <int C> __host__ __device__ constexpr int xo_row(int T)
{
    return (C + 4) * T + 4 * ((T >> 4) & 1) + 16 * (T >> 5);
}

template <int C, int NW, int NPM, int CIN> struct SwinDescs
{
    GemmDesc d[SwinLayout<C, NW, NPM, CIN>::NDESC];
};

template <int C, int NW, int NPM, int CIN> constexpr SwinDescs<C, NW, NPM, CIN> make_descs()
{
    using L = SwinLayout<C, NW, NPM, CIN>;
    SwinDescs<C, NW, NPM, CIN> r{};
    int n = 0;
    for (int h = 0; h < NW; ++h)
        for (int q = 0; q < 2; ++q)
            r.d[n++] = GemmDesc{L::qv(h, q), L::head(h) + (q ? 32 * C : 0), 1024, 512, C / 32, 2};
    r.d[n++] = GemmDesc{L::WO, L::head(0) + 64 * C + 512, L::HS, 512, NW, C / 16};
    for (int c = 0; c < L::NC; ++c)
    {
        r.d[n++] = GemmDesc{L::w1(c), L::b1(c), 512, 16 * C, C / 32, 2};
        r.d[n++] = GemmDesc{L::w2(c), L::b1(c) + 32 * C + 64, 0, 512, 1, C / 16};
    }
    for (int g = 0; g < NPM / 32; ++g)
        r.d[n++] = GemmDesc{L::pm(g), L::PM0 + g * 128 * C, 512, 64 * C, C / 8, 2};
    for (int w = 0; w < (CIN ? NW : 0); ++w)
        r.d[n++] = GemmDesc{L::pe(w), w * CIN * 4 * C / NW, 512, 16 * CIN, CIN / 32, L::NTE};
    return r;
}

// NH heads (NVIDIA's warps per block) run on NW waves (NW = NH, or 2 NH to halve LDS per wave)
template <int C, int NH, int NW, int NPM, bool TUBE, int CIN>
__device__ __forceinline__ void swin_block(const CommonParams& p, const TubeParams* tp, const wslot* w16, const int bx,
                                           const int by, const int gx)
{
    using L = SwinLayout<C, NH, NPM, CIN>;
    static_assert(NW % NH == 0 && NW / NH <= 4, "waves per head");
    constexpr int NPL = L::NPL, NT = L::NT, INNER = L::INNER;
    constexpr int AST = (C > INNER ? C : INNER) + 8;
    constexpr int GST = 40, QST = 40, PST = 24;
#ifndef SWIN_MLP_R
#define SWIN_MLP_R(NW) ((NW) / 2)
#endif
    constexpr int R = SWIN_MLP_R(NW); // MLP chunks per round
    constexpr int MG = (NT % NW == 0) ? 1 : ((2 * NT) % NW == 0 ? 2 : 4), MT = 4 / MG, NTW = NT * MG / NW;
    static_assert(NTW * NW == NT * MG, "column split");
    constexpr int PAST = CIN + 8;
    constexpr int SCR = NW * 16 * QST * 2 > (CIN ? 16 * PAST * 2 : 0) ? NW * 16 * QST * 2 : 16 * PAST * 2;
    constexpr int U_ST4 = R * 64 * GST * 2;
    constexpr int USIZE = SCR > U_ST4 ? SCR : U_ST4;

    __shared__ __attribute__((aligned(16))) half_t A[64 * AST];
    __shared__ __attribute__((aligned(16))) uint8_t U[USIZE];
    __shared__ half_t R2[64];
    half_t* X0 = A;                            // stages 0-1: block input f16 [64][AST], natural order
    half_t* QP = (half_t*)U;                   // stage 2: per-wave Q / P scratch [NW][16][QST]
    half_t* PA = QP;                           // stage 0: low-res tokens f16 [16][PAST] (apos order)
    half_t* G = (half_t*)U;                    // stage 4: fc1 outputs [R][64][GST]
#ifdef SWIN_A8
    // e4m3 byte images of the GEMM operands: the same rows and columns as the f16 images, one byte per value.
    // A token's bytes start where its f16 row starts, so they stay inside that row.
    constexpr int ARB = 2 * AST;
    uint8_t* const A8 = (uint8_t*)A;
    uint8_t* const G8 = U;
    // codes of one column in four rows
    auto put22 = [](uint8_t* p0, uint8_t* p1, uint8_t* p2, uint8_t* p3, hv2 a, hv2 b) {
        const uint32_t c = codes4(a, b);
        *p0 = (uint8_t)c;
        *p1 = (uint8_t)(c >> 8);
        *p2 = (uint8_t)(c >> 16);
        *p3 = (uint8_t)(c >> 24);
    };
#else
    constexpr int ARB = AST;
    half_t* const A8 = A;
    half_t* const G8 = G;
#endif

    const int lane = threadIdx.x, wv = threadIdx.z, tid = lane + 32 * wv;
    const int l16 = lane & 15, hi = lane >> 4;
    const uint8_t* W = p.w;

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

    // block input words (encoders: fp8 input, decoders: fp8 skip), issued first so their latency
    // overlaps the patch expand; word k of thread tid is idx = tid + 32 NW k
    constexpr int NSW = 64 * C / 4 / (32 * NW);
    static_assert(NSW * 32 * NW == 64 * C / 4, "input words per thread");
    uint32_t inw[NSW];
#pragma unroll
    for (int k = 0; k < NSW; ++k)
    {
        const int idx = tid + 32 * NW * k;
        const int T = idx / (C / 4), c4 = 4 * (idx % (C / 4));
        const int X = mirror(8 * bx - p.sx + tok_x(T), p.tw), Y = mirror(8 * by - p.sy + tok_y(T), p.th);
        const uint8_t* src = CIN ? p.p32 : p.in;
        inw[k] = input_word((const uint32_t*)(src + ((size_t)((c4 >> 5) * p.th + Y) * p.tw + X) * 32 + (c4 & 31)));
    }
    // the input words as f16 into X0 (natural channel order); decoders add the patch expand to them later
    auto stage_input_words = [&]() {
#pragma unroll
        for (int k = 0; k < NSW; ++k)
        {
            const int idx = tid + 32 * NW * k;
            const int T = idx / (C / 4), c4 = 4 * (idx % (C / 4));
            const uint32_t bytes = inw[k];
#ifdef SWIN_PACKED_INPUT
            *(h4*)(X0 + T * AST + c4) = decode4(bytes);
#else
#pragma unroll
            for (int b = 0; b < 4; ++b)
                X0[T * AST + c4 + b] = e4m3_to_half((bytes >> (8 * b)) & 0xffu);
#endif
        }
    };

    // ------------------------------------------------ stage 0: block input x0 (f16) into A
    // encoders: the fp8 input; decoders: q8(patch expand(low-res)) + skip
    if constexpr (CIN > 0)
    {
        const int W2 = p.tw / 2, H2 = p.th / 2;
        for (int idx = tid; idx < 16 * CIN / 4; idx += 32 * NW)
        {
            const int r = idx / (CIN / 4), c4 = 4 * (idx % (CIN / 4));
            const int MX = mirror((8 * bx - p.sx) / 2 + (r & 3), W2), MY = mirror((8 * by - p.sy) / 2 + (r >> 2), H2);
            const uint32_t bytes =
                input_word((const uint32_t*)(p.in + ((size_t)((c4 >> 5) * H2 + MY) * W2 + MX) * 32 + (c4 & 31)));
#ifdef SWIN_A8
            *(uint32_t*)((uint8_t*)PA + r * PAST + apos(c4)) = bytes; // apos(c4 + b) = apos(c4) + b
#elif defined(SWIN_PACKED_INPUT)
            *(h4*)(PA + r * PAST + apos(c4)) = decode4(bytes);
#else
#pragma unroll
            for (int b = 0; b < 4; ++b)
                PA[r * PAST + apos(c4 + b)] = e4m3_to_half((bytes >> (8 * b)) & 0xffu);
#endif
        }
        stage_input_words(); // the skip input
        __syncthreads();
        constexpr int NTEW = 4 * C / NW / 16; // expand tiles per wave (L::NTE per weight block)
        T16 e[NTEW];
#pragma unroll
        for (int j = 0; j < NTEW; ++j)
            e[j] = t16_splat(hload(W, 4 * C * CIN + 2 * (16 * (NTEW * wv + j) + l16)));
#pragma unroll
        for (int kc = 0; kc < CIN / 32; ++kc)
        {
#ifdef SWIN_A8
            const gop a0 = lda((const uint8_t*)PA + l16 * PAST + 32 * kc), a1 = lda((const uint8_t*)PA + l16 * PAST + 32 * kc + 16);
#else
            const gop a0 = lda(PA + l16 * PAST + 32 * kc), a1 = lda(PA + l16 * PAST + 32 * kc + 16);
#endif
#pragma unroll
            for (int j = 0; j < NTEW; ++j)
            {
                const int t = NTEW * wv + j, blk = t / L::NTE, lt = t % L::NTE;
                k32(e[j], a0, bload(w16, L::pe(blk), L::NTE, kc, 0, lt, l16), a1, bload(w16, L::pe(blk), L::NTE, kc, 1, lt, l16));
            }
        }
#pragma unroll
        for (int j = 0; j < NTEW; ++j)
        {
            const int Gc = 16 * (NTEW * wv + j) + l16, q = Gc / C, ch = ginv(Gc % C);
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const int r = mrow(i);
                const int tx = 2 * (r & 3) + (q & 1), ty = 2 * (r >> 2) + (q >> 1);
                const int T = (tx & 3) + 4 * (ty & 3) + 16 * (tx >> 2) + 32 * (ty >> 2);
                // e4m3_to_half is exact in f16, so adding the staged skip value is the same sum
                X0[T * AST + ch] = (half_t)(q8(e[j].get(i)) + X0[T * AST + ch]);
            }
        }
    }
    else
        stage_input_words();
    __syncthreads();

    // residual accumulators of the output projection: f16(x0 + b_o) in the C-fragment layout
    // (wave owns pair columns of n tiles ng*NTW.. for m tiles mg*MT..)
    const int ng = wv % (NW / MG), mg = wv / (NW / MG);
#ifdef SWIN_PACKED_RESIDUAL
    hv2 residual[MT][NTW][4]; // Only f16 is needed until the output projection.
#else
    T16 x[MT][NTW];
#endif
#pragma unroll
    for (int mm = 0; mm < MT; ++mm)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
        {
            const int n = 16 * (ng * NTW + j) + l16;
            const half_t bo = hload(W, L::BO + 2 * n);
            const int ch = ginv(n);
#pragma unroll
            for (int i = 0; i < 8; i += 2)
            {
                const hv2 v = (hv2){X0[(16 * (mg * MT + mm) + mrow(i)) * AST + ch],
                                   X0[(16 * (mg * MT + mm) + mrow(i + 1)) * AST + ch]} + (hv2){bo, bo};
#ifdef SWIN_PACKED_RESIDUAL
                residual[mm][j][i / 2] = v;
#else
                x[mm][j].set_pair(i / 2, v);
#endif
            }
        }
    __syncthreads(); // stage 1 overwrites x0 rows with h1

    // ------------------------------------------------ stage 1: norm1 (kernel-exact tree)
    {
        const int g = lane >> 2, t = lane & 3;
#pragma unroll
        for (int r = 0; r < 8 / NW; ++r)
        {
            const int T = (64 / NW) * wv + g + 8 * r;
            half_t v[NPL][2][4];
#pragma unroll
            for (int pl = 0; pl < NPL; ++pl)
#pragma unroll
                for (int ww = 0; ww < 2; ++ww)
#pragma unroll
                    for (int b = 0; b < 4; ++b)
                        v[pl][ww][b] = X0[T * AST + 32 * pl + 4 * (t + 4 * ww) + b];
            // the 4 lanes of a token read the whole row before any of them writes h1 over it
            wave_sync();
#ifdef SWIN_PACKED_NORM1
            hv2 ws[2];
#pragma unroll
            for (int ww = 0; ww < 2; ++ww)
            {
                hv2 pp[NPL];
#pragma unroll
                for (int pl = 0; pl < NPL; ++pl)
                {
                    hv2 a = {v[pl][ww][0], v[pl][ww][1]}, b = {v[pl][ww][2], v[pl][ww][3]};
                    pp[pl] = (hv2)(a * a) + (hv2)(b * b);
                }
                if constexpr (NPL == 4)
                    ws[ww] = (pp[0] + pp[1]) + (pp[2] + pp[3]);
                else if constexpr (NPL == 3)
                    ws[ww] = (pp[0] + pp[1]) + pp[2];
                else
                    ws[ww] = pp[0] + pp[1];
            }
            hv2 a = ws[0] + ws[1];
            a += __builtin_bit_cast(hv2, xor_lane<1>(__builtin_bit_cast(uint32_t, a)));
            a += __builtin_bit_cast(hv2, xor_lane<2>(__builtin_bit_cast(uint32_t, a)));
            const half_t rn = rsqrt16(a[0] + a[1]);
#else
            half_t ws[2][2];
#pragma unroll
            for (int ww = 0; ww < 2; ++ww)
            {
                half_t pp[NPL][2];
#pragma unroll
                for (int pl = 0; pl < NPL; ++pl)
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        half_t s0 = v[pl][ww][e] * v[pl][ww][e];
                        half_t s1 = v[pl][ww][2 + e] * v[pl][ww][2 + e];
                        pp[pl][e] = s0 + s1;
                    }
#pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    if constexpr (NPL == 4)
                        ws[ww][e] = (pp[0][e] + pp[1][e]) + (pp[2][e] + pp[3][e]);
                    else if constexpr (NPL == 3)
                        ws[ww][e] = (pp[0][e] + pp[1][e]) + pp[2][e];
                    else
                        ws[ww][e] = pp[0][e] + pp[1][e];
                }
            }
            half_t a0 = ws[0][0] + ws[1][0], a1 = ws[0][1] + ws[1][1];
            a0 = a0 + hxor<1>(a0);
            a1 = a1 + hxor<1>(a1);
            a0 = a0 + hxor<2>(a0);
            a1 = a1 + hxor<2>(a1);
            const half_t rn = rsqrt16(a0 + a1);
#endif
            const hv2 rn2 = {rn, rn};
#ifdef SWIN_A8
#pragma unroll
            for (int pl = 0; pl < NPL; ++pl)
#pragma unroll
                for (int ww = 0; ww < 2; ++ww)
                {
                    const int c = 32 * pl + 4 * (t + 4 * ww);
                    const hv2 g0 = __builtin_bit_cast(hv2, *(const uint32_t*)(W + L::G1 + 2 * gperm(c)));
                    const hv2 g1 = __builtin_bit_cast(hv2, *(const uint32_t*)(W + L::G1 + 2 * gperm(c + 2)));
                    // apos(c + b) = apos(c) + b for b < 4: the four codes are one word
                    *(uint32_t*)(A8 + T * ARB + apos(c)) = codes4((hv2){v[pl][ww][0], v[pl][ww][1]} * (hv2)(rn2 * g0),
                                                                  (hv2){v[pl][ww][2], v[pl][ww][3]} * (hv2)(rn2 * g1));
                }
#else
#pragma unroll
            for (int pl = 0; pl < NPL; ++pl)
#pragma unroll
                for (int ww = 0; ww < 2; ++ww)
#pragma unroll
                    for (int b = 0; b < 4; b += 2)
                    {
                        const int c = 32 * pl + 4 * (t + 4 * ww) + b;
                        // gperm(c + 1) = gperm(c) + 1 for even c: one 32-bit gamma load per pair
                        const hv2 gm = __builtin_bit_cast(hv2, *(const uint32_t*)(W + L::G1 + 2 * gperm(c)));
                        const hv2 hq = q8x2((hv2){v[pl][ww][b], v[pl][ww][b + 1]} * (hv2)(rn2 * gm));
                        // apos(c + 1) = apos(c) + 1 for even c
                        *(hv2*)(A + T * AST + apos(c)) = hq;
                    }
#endif
        }
    }
    __syncthreads();

    // ------------------------------------------------ stage 2: Q/V projection + attention (wave = head)
    {
        constexpr int MW = 4 / (NW / NH); // windows per wave
        const int h = NW == NH ? wv : wv % NH, m0 = NW == NH ? 0 : MW * (wv / NH);
        hv2 qp[MW][2][4], vp[MW][2][4];
#ifdef SWIN_QV_SPLIT
        // Q and V have independent k32 chains. Retain Q as packed halves while
        // calculating V, instead of keeping both sets of f32 chains live.
        auto project = [&](auto value_tag, hv2 (&dest)[MW][2][4]) {
            constexpr bool VALUE = decltype(value_tag)::value;
            T16 part[MW][2];
#pragma unroll
            for (int m = 0; m < MW; ++m)
#pragma unroll
                for (int q = 0; q < 2; ++q)
                    part[m][q] = t16_splat((half_t)0.0f);
#pragma unroll
            for (int kc = 0; kc < C / 32; ++kc) {
                gop b[2][2];
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int q = 0; q < 2; ++q)
                        b[s][q] = bload(w16, L::qv(h, VALUE ? 1 : 0), 2, kc, s, q, l16);
#pragma unroll
                for (int m = 0; m < MW; ++m) {
                    const gop a0 = lda(A + (16 * (m0 + m) + l16) * AST + 32 * kc);
                    const gop a1 = lda(A + (16 * (m0 + m) + l16) * AST + 32 * kc + 16);
#pragma unroll
                    for (int q = 0; q < 2; ++q)
                        k32(part[m][q], a0, b[0][q], a1, b[1][q]);
                }
            }
#pragma unroll
            for (int m = 0; m < MW; ++m)
#pragma unroll
                for (int q = 0; q < 2; ++q)
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        if constexpr (VALUE) dest[m][q][k] = part[m][q].pair(k);
                        else dest[m][q][k] = q8x2(part[m][q].pair(k));
                    }
        };
        project(std::false_type{}, qp);
        project(std::true_type{}, vp);
#else
        T16 acc[MW][4]; // [window][Q nt0, Q nt1, V nt0, V nt1]
#pragma unroll
        for (int m = 0; m < MW; ++m)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc[m][q] = t16_splat((half_t)0.0f);
#pragma unroll
        for (int kc = 0; kc < C / 32; ++kc)
        {
            gop b[2][4];
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    b[s][q] = bload(w16, L::qv(h, q >> 1), 2, kc, s, q & 1, l16);
#pragma unroll
            for (int m = 0; m < MW; ++m)
            {
                const gop a0 = lda(A8 + (16 * (m0 + m) + l16) * ARB + 32 * kc);
                const gop a1 = lda(A8 + (16 * (m0 + m) + l16) * ARB + 32 * kc + 16);
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    k32(acc[m][q], a0, b[0][q], a1, b[1][q]);
            }
        }
        // pack: Q as q8 pairs, V as f16 pairs (all that attention needs), halving the live registers
#pragma unroll
        for (int m = 0; m < MW; ++m)
#pragma unroll
            for (int q = 0; q < 2; ++q)
#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
#ifdef SWIN_A8
                    qp[m][q][i / 2] = acc[m][q].pair(i / 2); // encoded when staged
#else
                    qp[m][q][i / 2] = q8x2(acc[m][q].pair(i / 2));
#endif
                    vp[m][q][i / 2] = acc[m][2 + q].pair(i / 2);
                }
#endif
        __syncthreads(); // every wave is done reading h1; O may overwrite A

        half_t* Qs = QP + wv * 16 * QST;
        const int bbase = L::head(h) + 64 * C;
        T16 bias;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int ri = mrow(i), cj = l16;
            const int bl = 4 * (ri & 7) + ((cj & 7) >> 1), word = (ri >> 3) + 2 * (cj >> 3);
            bias.set(i, hload(W, bbase + 2 * (8 * bl + 2 * word + (cj & 1))));
        }
        const half_t cs = f16(0.0972222164273262f / 5.656854152679443f);
        const half_t cl = (half_t)0.55615234375f;
        const half_t p1 = f16(0.92730712890625f), p0 = (half_t)1.375f;
#pragma unroll
        for (int m = 0; m < MW; ++m)
        {
#ifdef SWIN_A8
            uint8_t* const Q8 = (uint8_t*)Qs;
#pragma unroll
            for (int q = 0; q < 2; ++q)
#pragma unroll
                for (int i = 0; i < 8; i += 4)
                    put22(Q8 + mrow(i) * QST + 16 * q + l16, Q8 + mrow(i + 1) * QST + 16 * q + l16,
                          Q8 + mrow(i + 2) * QST + 16 * q + l16, Q8 + mrow(i + 3) * QST + 16 * q + l16, qp[m][q][i / 2],
                          qp[m][q][i / 2 + 1]);
            wave_sync();
            const gop a0 = lda(Q8 + l16 * QST), a1 = lda(Q8 + l16 * QST + 16);
#else
#pragma unroll
            for (int q = 0; q < 2; ++q)
#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    Qs[mrow(i) * QST + 16 * q + l16] = qp[m][q][i / 2][0];
                    Qs[mrow(i + 1) * QST + 16 * q + l16] = qp[m][q][i / 2][1];
                }
            wave_sync();
            const gop a0 = lda(Qs + l16 * QST), a1 = lda(Qs + l16 * QST + 16);
#endif
            T16 S = bias;
            k32(S, a0, a0, a1, a1);
            // rows mrow(2k), mrow(2k+1) of this lane's column, packed: scale, clamp and the cubic
            // polynomial as f16x2, the exponent trick per row on the (even, odd) column word
            hv2 wg[4];
            const bool oddc = l16 & 1;
            const uint32_t sel_a = oddc ? 0x01000504u : 0x05040100u, sel_b = oddc ? 0x03020706u : 0x07060302u;
            const uint32_t sel_r = oddc ? 0x07060302u : 0x05040100u;
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                hv2 tv = S.pair(k) * (hv2){cs, cs};
                tv = __builtin_elementwise_max(__builtin_elementwise_min(tv, (hv2){cl, cl}), (hv2){-cl, -cl});
                const hv2 e1 = __builtin_elementwise_fma(tv, -tv, (hv2){p1, p1});
                const hv2 poly = __builtin_elementwise_fma(tv, e1, (hv2){p0, p0});
                const uint32_t mine = __builtin_bit_cast(uint32_t, poly);
                const uint32_t other = xor_lane<1>(mine);
                const uint32_t ua = (__builtin_amdgcn_perm(other, mine, sel_a) << 5) + 0x7FF88000u;
                const uint32_t ub = (__builtin_amdgcn_perm(other, mine, sel_b) << 5) + 0x7FF88000u;
                wg[k] = __builtin_bit_cast(hv2, __builtin_amdgcn_perm(ub, ua, sel_r));
            }
            wave_sync(); // Q scratch is dead: P overwrites it
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                auto xr = [](hv2 v, auto mask) {
                    return __builtin_bit_cast(hv2, xor_lane<decltype(mask)::value>(__builtin_bit_cast(uint32_t, v)));
                };
                const hv2 v1 = wg[k] + xr(wg[k], std::integral_constant<int, 8>{});
                const hv2 v2 = v1 + xr(v1, std::integral_constant<int, 2>{});
                const hv2 v3 = v2 + xr(v2, std::integral_constant<int, 4>{});
                const hv2 sum = v3 + xr(v3, std::integral_constant<int, 1>{});
                const hv2 rinv = {f16(1.0f / (float)sum[0]), f16(1.0f / (float)sum[1])};
                const hv2 pw = wg[k] * rinv;
                Qs[mrow(2 * k) * PST + l16] = pw[0];
                Qs[mrow(2 * k + 1) * PST + l16] = pw[1];
            }
            wave_sync();
            const sop pa = lds16(Qs + l16 * PST);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                // V as B operand: lane holds column l16 over the 16 tokens (rows, the K of P V)
#if D4R_WMMA_LAYOUT == 12
                // own pair k holds rows (2k + 8hi, 2k + 1 + 8hi): this half's K values in order
                const sop vb = __builtin_bit_cast(sop, vp[m][nt]);
#else
                // own pair k holds rows (4k+hi, 4k+2+hi); the other half-wave has (4k+1-hi, 4k+3-hi)
                uint32_t vw[8];
                const uint32_t s_lo = hi ? 0x01000504u : 0x05040100u, s_hi = hi ? 0x03020706u : 0x07060302u;
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    const uint32_t own = __builtin_bit_cast(uint32_t, vp[m][nt][k]);
                    const uint32_t other = xor_lane<16>(own);
                    vw[2 * k] = __builtin_amdgcn_perm(other, own, s_lo);     // rows 4k, 4k+1
                    vw[2 * k + 1] = __builtin_amdgcn_perm(other, own, s_hi); // rows 4k+2, 4k+3
                }
                const h16 vb = __builtin_bit_cast(h16, vw);
#endif
                const f8v o = wmma(pa, vb, splat(0.0f));
                const int ocol = apos(32 * h + ginv(16 * nt + l16));
#ifdef SWIN_A8
#pragma unroll
                for (int i = 0; i < 8; i += 4)
                {
                    uint8_t* const oc = A8 + 16 * (m0 + m) * ARB + ocol;
                    put22(oc + mrow(i) * ARB, oc + mrow(i + 1) * ARB, oc + mrow(i + 2) * ARB, oc + mrow(i + 3) * ARB,
                          (hv2){(half_t)o[i], (half_t)o[i + 1]}, (hv2){(half_t)o[i + 2], (half_t)o[i + 3]});
                }
#else
#pragma unroll
                for (int i = 0; i < 8; i += 2)
                {
                    const hv2 oq = q8x2((hv2){(half_t)o[i], (half_t)o[i + 1]});
                    A[(16 * (m0 + m) + mrow(i)) * AST + ocol] = oq[0];
                    A[(16 * (m0 + m) + mrow(i + 1)) * AST + ocol] = oq[1];
                }
#endif
            }
            wave_sync();
        }
    }
    __syncthreads();

    // ------------------------------------------------ stage 3: output projection + residual
#ifdef SWIN_PACKED_RESIDUAL
    T16 x[MT][NTW];
#pragma unroll
    for (int mm = 0; mm < MT; ++mm)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k)
                x[mm][j].set_pair(k, residual[mm][j][k]);
#endif
#pragma unroll
    for (int kc = 0; kc < NH; ++kc)
    {
        gop b[2][NTW];
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int j = 0; j < NTW; ++j)
                b[s][j] = bload(w16, L::WO, NT, kc, s, ng * NTW + j, l16);
#pragma unroll
        for (int mm = 0; mm < MT; ++mm)
        {
            const gop a0 = lda(A8 + (16 * (mg * MT + mm) + l16) * ARB + 32 * kc);
            const gop a1 = lda(A8 + (16 * (mg * MT + mm) + l16) * ARB + 32 * kc + 16);
#pragma unroll
            for (int j = 0; j < NTW; ++j)
                k32(x[mm][j], a0, b[0][j], a1, b[1][j]);
        }
    }
    __syncthreads(); // X0b and O are dead

    // ------------------------------------------------ norm2 (kernel-exact tree), x1 staged in A
#pragma unroll
    for (int mm = 0; mm < MT; ++mm)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                A[(16 * (mg * MT + mm) + mrow(i)) * AST + 16 * (ng * NTW + j) + l16] = x[mm][j].get(i);
    __syncthreads();
#ifdef SWIN_PARALLEL_NORM2
    // Four lanes per token. Preserve the original per-column and even/odd trees.
#pragma unroll
    for (int r = 0; r < 8 / NW; ++r)
    {
        const int T = (64 / NW) * wv + (lane >> 2) + 8 * r;
        const int c = 2 * (lane & 3);
        hv2 sq[L::NT8];
#pragma unroll
        for (int j = 0; j < L::NT8; ++j)
        {
            const hv2 v = *(const hv2*)(A + T * AST + 8 * j + c);
            sq[j] = v * v;
        }
        hv2 even, odd;
        if constexpr (L::NT8 == 16)
        {
            even = ((sq[0] + sq[2]) + (sq[4] + sq[6])) + ((sq[8] + sq[10]) + (sq[12] + sq[14]));
            odd = ((sq[1] + sq[3]) + (sq[5] + sq[7])) + ((sq[9] + sq[11]) + (sq[13] + sq[15]));
        }
        else if constexpr (L::NT8 == 12)
        {
            even = ((sq[0] + sq[2]) + (sq[4] + sq[6])) + (sq[8] + sq[10]);
            odd = ((sq[1] + sq[3]) + (sq[5] + sq[7])) + (sq[9] + sq[11]);
        }
        else
        {
            even = (sq[0] + sq[2]) + (sq[4] + sq[6]);
            odd = (sq[1] + sq[3]) + (sq[5] + sq[7]);
        }
        hv2 sum = even + odd;
        sum += __builtin_bit_cast(hv2, xor_lane<1>(__builtin_bit_cast(uint32_t, sum)));
        sum += __builtin_bit_cast(hv2, xor_lane<2>(__builtin_bit_cast(uint32_t, sum)));
        if ((lane & 3) == 0)
            R2[T] = rsqrt16(sum[0] + sum[1]);
    }
#else
    if (tid < 64)
    {
        // the row as 128-bit LDS loads: element 8 j + c of n8 tile j
        h8 row[L::NT8];
#pragma unroll
        for (int j = 0; j < L::NT8; ++j)
            row[j] = *(const h8*)(A + tid * AST + 8 * j);
        half_t L8[8];
#pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            half_t sq[L::NT8];
#pragma unroll
            for (int j = 0; j < L::NT8; ++j)
                sq[j] = row[j][c] * row[j][c];
            half_t even, odd;
            if constexpr (L::NT8 == 16)
            {
                even = ((sq[0] + sq[2]) + (sq[4] + sq[6])) + ((sq[8] + sq[10]) + (sq[12] + sq[14]));
                odd = ((sq[1] + sq[3]) + (sq[5] + sq[7])) + ((sq[9] + sq[11]) + (sq[13] + sq[15]));
            }
            else if constexpr (L::NT8 == 12)
            {
                even = ((sq[0] + sq[2]) + (sq[4] + sq[6])) + (sq[8] + sq[10]);
                odd = ((sq[1] + sq[3]) + (sq[5] + sq[7])) + (sq[9] + sq[11]);
            }
            else
            {
                even = (sq[0] + sq[2]) + (sq[4] + sq[6]);
                odd = (sq[1] + sq[3]) + (sq[5] + sq[7]);
            }
            L8[c] = even + odd;
        }
        const half_t b0 = (L8[0] + L8[2]) + (L8[4] + L8[6]);
        const half_t b1 = (L8[1] + L8[3]) + (L8[5] + L8[7]);
        R2[tid] = rsqrt16(b0 + b1);
    }
#endif
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NTW; ++j)
    {
        const int n = 16 * (ng * NTW + j) + l16;
        const half_t g2 = hload(W, L::G2 + 2 * n), b2 = hload(W, L::B2 + 2 * n);
        const int col = apos(ginv(n));
#pragma unroll
        for (int mm = 0; mm < MT; ++mm)
#ifdef SWIN_A8
#pragma unroll
            for (int i = 0; i < 8; i += 4)
            {
                hv2 h[2];
                uint8_t* at[4];
#pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    const int row = 16 * (mg * MT + mm) + mrow(i + 2 * e), row2 = 16 * (mg * MT + mm) + mrow(i + 2 * e + 1);
                    const hv2 v = x[mm][j].pair(i / 2 + e);
                    const hv2 r2 = {R2[row], R2[row2]};
                    h[e] = v * (hv2)(r2 * (hv2){g2, g2});
                    at[2 * e] = A8 + row * ARB + col;
                    at[2 * e + 1] = A8 + row2 * ARB + col;
                    x[mm][j].set_pair(i / 2 + e, v + (hv2){b2, b2});
                }
                put22(at[0], at[1], at[2], at[3], h[0], h[1]);
            }
#else
#pragma unroll
            for (int i = 0; i < 8; i += 2)
            {
                const int row = 16 * (mg * MT + mm) + mrow(i), row2 = 16 * (mg * MT + mm) + mrow(i + 1);
                const hv2 v = x[mm][j].pair(i / 2);
                const hv2 r2 = {R2[row], R2[row2]};
                const hv2 hq = q8x2(v * (hv2)(r2 * (hv2){g2, g2}));
                A[row * AST + col] = hq[0];
                A[row2 * AST + col] = hq[1];
                x[mm][j].set_pair(i / 2, v + (hv2){b2, b2});
            }
#endif
    }
    __syncthreads();

    // ------------------------------------------------ stage 4: MLP, R chunks per round
    {
        constexpr int WPC = NW / R, MPW = 4 / WPC; // waves per chunk, fc1 m tiles per wave
        const int cw = wv / WPC, mb = MPW * (wv % WPC);
#ifdef SWIN_UNROLL_MLP
    SWIN_PRAGMA(unroll SWIN_UNROLL_MLP)
#else
#pragma unroll 1
#endif
        for (int r = 0; r < L::NC / R; ++r)
        {
            const int c = R * r + cw;
            const int B1 = L::b1(c);
            T16 z[MPW][2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
            {
                const half_t b1 = hload(W, B1 + 32 * C + 2 * (16 * nt + l16));
#pragma unroll
                for (int mm = 0; mm < MPW; ++mm)
                    z[mm][nt] = t16_splat(b1);
            }
#pragma unroll
            for (int kc = 0; kc < C / 32; ++kc)
            {
                gop b[2][2];
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt)
                        b[s][nt] = bload(w16, L::w1(c), 2, kc, s, nt, l16);
#pragma unroll
                for (int mm = 0; mm < MPW; ++mm)
                {
                    const gop a0 = lda(A8 + (16 * (mb + mm) + l16) * ARB + 32 * kc);
                    const gop a1 = lda(A8 + (16 * (mb + mm) + l16) * ARB + 32 * kc + 16);
                    k32(z[mm][0], a0, b[0][0], a1, b[1][0]);
                    k32(z[mm][1], a0, b[0][1], a1, b[1][1]);
                }
            }
            half_t* Gc = G + cw * 64 * GST;
#pragma unroll
            for (int mm = 0; mm < MPW; ++mm)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                {
                    const int col = apos(ginv(16 * nt + l16));
#ifdef SWIN_A8
#pragma unroll
                    for (int i = 0; i < 8; i += 4)
                    {
                        hv2 gl[2];
#pragma unroll
                        for (int e = 0; e < 2; ++e)
                        {
                            const hv2 zv = z[mm][nt].pair(i / 2 + e);
                            const hv2 cz = __builtin_elementwise_max(__builtin_elementwise_min(zv, (hv2){2.0f16, 2.0f16}),
                                                                     (hv2){-2.0f16, -2.0f16});
                            const hv2 az = __builtin_bit_cast(hv2, (u16x2)(__builtin_bit_cast(u16x2, cz) & (unsigned short)0x7fff));
                            const hv2 inner = (hv2){f16(0.41216981f), f16(0.41216981f)} -
                                              (hv2)((hv2){f16(0.08108133f), f16(0.08108133f)} * az);
                            gl[e] = zv * (hv2)((hv2){0.5f16, 0.5f16} + (hv2)(cz * inner));
                        }
                        uint8_t* const gc = G8 + (cw * 64 + 16 * (mb + mm)) * GST + col;
                        put22(gc + mrow(i) * GST, gc + mrow(i + 1) * GST, gc + mrow(i + 2) * GST, gc + mrow(i + 3) * GST, gl[0],
                              gl[1]);
                    }
#else
#pragma unroll
                    for (int i = 0; i < 8; i += 2)
                    {
                        const hv2 zv = z[mm][nt].pair(i / 2);
                        const hv2 cz = __builtin_elementwise_max(__builtin_elementwise_min(zv, (hv2){2.0f16, 2.0f16}),
                                                                 (hv2){-2.0f16, -2.0f16});
                        const hv2 az = __builtin_bit_cast(hv2, (u16x2)(__builtin_bit_cast(u16x2, cz) & (unsigned short)0x7fff));
                        const hv2 inner = (hv2){f16(0.41216981f), f16(0.41216981f)} -
                                          (hv2)((hv2){f16(0.08108133f), f16(0.08108133f)} * az);
                        const hv2 gl = zv * (hv2)((hv2){0.5f16, 0.5f16} + (hv2)(cz * inner));
                        const hv2 gq = q8x2(gl);
                        Gc[(16 * (mb + mm) + mrow(i)) * GST + col] = gq[0];
                        Gc[(16 * (mb + mm) + mrow(i + 1)) * GST + col] = gq[1];
                    }
#endif
                }
            __syncthreads();
#pragma unroll
            for (int cc = 0; cc < R; ++cc)
            {
                const int c2 = R * r + cc;
                gop b[2][NTW];
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int j = 0; j < NTW; ++j)
                        b[s][j] = bload(w16, L::w2(c2), NT, 0, s, ng * NTW + j, l16);
#pragma unroll
                for (int mm = 0; mm < MT; ++mm)
                {
                    const gop a0 = lda(G8 + (cc * 64 + 16 * (mg * MT + mm) + l16) * GST);
                    const gop a1 = lda(G8 + (cc * 64 + 16 * (mg * MT + mm) + l16) * GST + 16);
#pragma unroll
                    for (int j = 0; j < NTW; ++j)
                        k32(x[mm][j], a0, b[0][j], a1, b[1][j]);
                }
            }
            __syncthreads();
        }
    }

    // ------------------------------------------------ output: f16 staged in LDS, 16-value segments encoded by a loop
    // XO: [64 tokens][C channels] natural order; the row padding keeps the patch merge's 8-byte reads of 16
    // tokens on distinct LDS banks (see xo_row)
    half_t* XO = A;
#ifdef SWIN_A8
    constexpr int XB = C + 4; // byte row stride of the patch merge's code image (rows stay 4-byte aligned)
#endif
    static_assert(xo_row<C>(63) + C <= 64 * AST, "XO fits in A");
#pragma unroll
    for (int j = 0; j < NTW; ++j)
    {
        const int ch = ginv(16 * (ng * NTW + j) + l16);
#pragma unroll
        for (int mm = 0; mm < MT; ++mm)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                XO[xo_row<C>(16 * (mg * MT + mm) + mrow(i)) + ch] = x[mm][j].get(i);
    }
    __syncthreads();
#pragma unroll 1
    for (int seg = tid; seg < 64 * C / 16; seg += 32 * NW)
    {
        const int T = seg / (C / 16), c16 = 16 * (seg % (C / 16));
        // rows are 8-byte aligned (xo_row), so the segment moves as four 8-byte words of two f16 pairs
        uint2* xs = (uint2*)(XO + xo_row<C>(T) + c16);
        uint32_t w[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            const uint2 pr = xs[k];
#ifdef SWIN_A8
            w[k] = codes4(__builtin_bit_cast(hv2, pr.x), __builtin_bit_cast(hv2, pr.y));
#else
            hv2 qa, qb;
            uint32_t ca, cb;
#ifdef SWIN_DIRECT_CODES
            if constexpr (NPM == 0) {
                ca = direct_codes8x2(__builtin_bit_cast(hv2, pr.x));
                cb = direct_codes8x2(__builtin_bit_cast(hv2, pr.y));
            } else
#endif
            {
                ca = enc8x2(__builtin_bit_cast(hv2, pr.x), qa);
                cb = enc8x2(__builtin_bit_cast(hv2, pr.y), qb);
            }
            w[k] = pack_codes(ca, cb);
            // the patch merge consumes decode(code) = q8_exact(x), written back over x (this thread's own values)
            if constexpr (NPM > 0)
                xs[k] = (uint2){__builtin_bit_cast(uint32_t, qa), __builtin_bit_cast(uint32_t, qb)};
#endif
        }
        const uint4 v = {w[0], w[1], w[2], w[3]};
#ifdef SWIN_A8
        // the patch merge's operands are these codes: a byte image [64][XB] in U (G is dead)
        if constexpr (NPM > 0)
        {
            static_assert(64 * XB <= USIZE, "merge byte image fits in U");
            uint32_t* xb = (uint32_t*)(U + T * XB + c16);
#pragma unroll
            for (int k = 0; k < 4; ++k)
                xb[k] = w[k];
        }
#endif
        const int X = 8 * bx - p.sx + tok_x(T), Y = 8 * by - p.sy + tok_y(T);
        if (X >= 0 && X < p.tw && Y >= 0 && Y < p.th)
            *(uint4*)(p.out + ((size_t)((c16 >> 5) * p.th + Y) * p.tw + X) * 32 + (c16 & 31)) = v;
    }
    if constexpr (NPM > 0)
        __syncthreads();

    // ------------------------------------------------ patch merge: 2x2 tokens (4C) -> NPM, half resolution
    if constexpr (NPM > 0)
    {
        // merge tiles split evenly over the largest wave count dividing them (remaining waves idle)
        constexpr int PT = NPM / 16 ? NPM / 16 : 1;
        constexpr int PMW = PT % NW == 0 ? NW : (PT % 4 == 0 && NW >= 4 ? 4 : (PT % 3 == 0 && NW >= 3 ? 3 : (PT % 2 == 0 && NW >= 2 ? 2 : 1)));
        constexpr int NTP = PT / PMW;
        constexpr bool PM_EVEN = true;
        if (wv < PMW)
        {
        T16 y[NTP];
#pragma unroll
        for (int j = 0; j < NTP; ++j)
            y[j] = t16_splat(hload(W, L::PMB + 2 * (16 * (PM_EVEN ? wv * NTP + j : (wv * NTP + j) % (NPM / 16)) + l16)));
        const int mx = l16 & 3, my = l16 >> 2;
#pragma unroll
        for (int kc = 0; kc < C / 8; ++kc)
        {
            gop as[2];
#pragma unroll
            for (int s = 0; s < 2; ++s)
            {
#ifdef SWIN_A8
                // this half's two runs of four K slots, as stored codes
                uint32_t run4[2];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                {
                    const int k = 32 * kc + kslot(s, 4 * (2 * hi + rr));
                    const int q = k / C, ch = k % C;
                    const int tx = 2 * mx + (q & 1), ty = 2 * my + (q >> 1);
                    const int T = (tx & 3) + 4 * (ty & 3) + 16 * (tx >> 2) + 32 * (ty >> 2);
                    run4[rr] = *(const uint32_t*)(U + T * XB + ch);
                }
                as[s] = (u2v){run4[0], run4[1]};
#else
                sop a;
                // the operand's K slots in runs of 4 (gfx12: this half's slots 8 hi .. 8 hi + 7)
                constexpr int RUNS = sizeof(sop) / sizeof(half_t) / 4;
#pragma unroll
                for (int rr = 0; rr < RUNS; ++rr)
                {
                    const int run = RUNS == 4 ? rr : 2 * hi + rr;
                    const int k = 32 * kc + kslot(s, 4 * run);
                    const int q = k / C, ch = k % C;
                    const int tx = 2 * mx + (q & 1), ty = 2 * my + (q >> 1);
                    const int T = (tx & 3) + 4 * (ty & 3) + 16 * (tx >> 2) + 32 * (ty >> 2);
                    typedef _Float16 h4 __attribute__((ext_vector_type(4)));
                    const h4 v = *(const h4*)(XO + xo_row<C>(T) + ch);
#pragma unroll
                    for (int b = 0; b < 4; ++b)
                        a[4 * rr + b] = v[b];
                }
                as[s] = as_gop(a);
#endif
            }
#pragma unroll
            for (int j = 0; j < NTP; ++j)
            {
                // out-of-range tiles (uneven split) recompute a valid one and are not stored
                const int t16 = PM_EVEN ? wv * NTP + j : (wv * NTP + j) % (NPM / 16);
                k32(y[j], as[0], bload(w16, L::pm(t16 >> 1), 2, kc, 0, t16 & 1, l16), as[1],
                    bload(w16, L::pm(t16 >> 1), 2, kc, 1, t16 & 1, l16));
            }
        }
        const int W2 = p.tw / 2, H2 = p.th / 2;
#pragma unroll
        for (int j = 0; j < NTP; ++j)
        {
            if (!PM_EVEN && wv * NTP + j >= NPM / 16)
                continue;
            const int ch = ginv(16 * (wv * NTP + j) + l16);
#ifdef SWIN_A8
            const uint32_t code[2] = {codes4(y[j].pair(0), y[j].pair(1)), codes4(y[j].pair(2), y[j].pair(3))};
#endif
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const int r = mrow(i);
                const int MX = (8 * bx - p.sx) / 2 + (r & 3), MY = (8 * by - p.sy) / 2 + (r >> 2);
                if (MX < 0 || MX >= W2 || MY < 0 || MY >= H2)
                    continue;
                p.p48[((size_t)((ch >> 5) * H2 + MY) * W2 + MX) * 32 + (ch & 31)] =
#ifdef SWIN_A8
                    (uint8_t)(code[i >> 2] >> (8 * (i & 3)));
#else
                    (uint8_t)enc8(y[j].get(i));
#endif
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
            if (tid == 0)
                __hip_atomic_store(tp->out_flags + (size_t)(by * gx + bx) * 32, (uint8_t)1, __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

#if defined(SWIN_WAVES_PER_EU)
#define SWIN_VGPR_ATTR __attribute__((amdgpu_waves_per_eu(SWIN_WAVES_PER_EU)))
#elif defined(SWIN_MAX_VGPR)
#define SWIN_VGPR_ATTR __attribute__((amdgpu_num_vgpr(SWIN_MAX_VGPR)))
#else
#define SWIN_VGPR_ATTR
#endif
#ifndef SWIN_PERSIST
#define SWIN_PERSIST 0 // persistent workgroups in x (0: one workgroup per original block)
#endif
#ifndef SWIN_PREP_SLOTS
#define SWIN_PREP_SLOTS 0 // weight images kept per weights pointer (0: one image, re-prepared when the pointer changes)
#endif
#ifndef SWIN_PREP_KEY_AT
#define SWIN_PREP_KEY_AT 1 // weights pointer at offset 0; 0 forces preparation on every launch
#endif

// Optional weight-image slots for several weight sets with stable address identity. The ZLUDA
// hook can run prep only for a weights pointer it has not seen (d4r_prep_key_at / d4r_prep_key_slots);
// enc3 disables this policy because NGX recycles weight addresses across feature lifetimes.
// the prep claims the first free slot for that pointer, and pointers beyond S share the overflow slot S,
// which the hook re-prepares on every launch. Keys are written by one thread of the prep; any prep
// thread that reads before that write finds the same (first free) slot.
template <int S> __device__ __forceinline__ int prep_slot_find(const uint64_t* keys, uint64_t w)
{
    for (int i = 0; i < S; ++i)
        if (__hip_atomic_load(keys + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == w)
            return i;
    return S;
}
template <int S> __device__ __forceinline__ int prep_slot_claim(uint64_t* keys, uint64_t w, bool writer)
{
    // one pass: the first slot holding w or nothing. Keys only ever go from 0 to a pointer, so every
    // prep thread picks the same slot whether it reads that slot before or after the writer's store.
    for (int i = 0; i < S; ++i)
    {
        const uint64_t k = __hip_atomic_load(keys + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (k == w)
            return i;
        if (k == 0)
        {
            if (writer)
                __hip_atomic_store(keys + i, w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            return i;
        }
    }
    return S;
}

// Module boilerplate: f16 weight image(s), prep kernel and its grid for the ZLUDA override hook.
#define SWIN_MODULE(NAME, C, NW, NPM, TUBE, PARAMS, CIN) SWIN_MODULE_W(NAME, C, NW, NW, NPM, TUBE, PARAMS, CIN)
// NH heads on NWAVES waves; d4r_block_z tells the ZLUDA hook the block z size to launch with
#define SWIN_MODULE_W(NAME, C, NH, NWAVES, NPM, TUBE, PARAMS, CIN)                                                      \
    using NAME##_L = SwinLayout<C, NH, NPM, CIN>;                                                                       \
    constexpr int NAME##_S = SWIN_PREP_SLOTS;                                                                           \
    __device__ wslot g_w16[(NAME##_S + 1) * NAME##_L::TOTAL];                                                           \
    __device__ uint64_t g_prep_keys[NAME##_S + 1];                                                                      \
    __constant__ SwinDescs<C, NH, NPM, CIN> g_descs = make_descs<C, NH, NPM, CIN>();                                    \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = NWAVES;                                         \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::TOTAL + 127) / 128;              \
    /* enc3 disables address-only reuse: a recreated feature may recycle a weights address */                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_at = SWIN_PREP_KEY_AT;                             \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_slots = NAME##_S;                                \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PARAMS p)                                             \
    {                                                                                                                   \
        const int idx = blockIdx.x * 128 + threadIdx.x;                                                                 \
        const uint8_t* w = ((const CommonParams*)&p)->w;                                                                \
        const int slot = NAME##_S ? prep_slot_claim<NAME##_S>(g_prep_keys, (uint64_t)w, idx == 0) : 0;                 \
        if (idx < NAME##_L::TOTAL)                                                                                      \
            expand_weights(w, g_w16 + slot * NAME##_L::TOTAL, g_descs.d, NAME##_L::NDESC, idx);                        \
    }                                                                                                                   \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_grid_x = SWIN_PERSIST;                                     \
    extern "C" __global__ void __launch_bounds__(32 * NWAVES) SWIN_VGPR_ATTR NAME(PARAMS p)                             \
    {                                                                                                                   \
        const CommonParams& cp = *(const CommonParams*)&p;                                                              \
        const int slot = NAME##_S ? prep_slot_find<NAME##_S>(g_prep_keys, (uint64_t)cp.w) : 0;                         \
        const wslot* w16 = g_w16 + slot * NAME##_L::TOTAL;                                                              \
        /* the original grid: 8x8-token blocks covering the shifted token grid */                                        \
        const int gx = (cp.tw + cp.sx + 7) / 8, gy = (cp.th + cp.sy + 7) / 8;                                           \
        if (SWIN_PERSIST == 0)                                                                                          \
        {                                                                                                               \
            swin_block<C, NH, NWAVES, NPM, TUBE, CIN>(cp, (const TubeParams*)&p, w16, blockIdx.x, blockIdx.y, gx);      \
            return;                                                                                                     \
        }                                                                                                               \
        _Pragma("unroll 1") for (int b = blockIdx.x; b < gx * gy; b += gridDim.x)                                      \
        {                                                                                                               \
            swin_block<C, NH, NWAVES, NPM, TUBE, CIN>(cp, (const TubeParams*)&p, w16, b % gx, b / gx, gx);              \
            __syncthreads();                                                                                            \
        }                                                                                                               \
    }

// RDNA4 native FP8: a module that defines SWIN_TOKEN_LANES is built from the token-lane block instead
// (swin_tok.h: one wave per window, transposed GEMMs, results stay in registers)
#if defined(SWIN_A8) && defined(SWIN_TOKEN_LANES)
#include "swin_tok.h"
#undef SWIN_MODULE_W
#define SWIN_MODULE_W(NAME, C, NH, NWAVES, NPM, TUBE, PARAMS, CIN) SWIN_TOK_MODULE(NAME, C, NH, NPM, TUBE, PARAMS, CIN)
#endif
