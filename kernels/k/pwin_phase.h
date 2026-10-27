// Phase-split DltssPaddedWinLayer: the deep layers have few windows (dec5: 9), so one block
// per window leaves most of the GPU idle. Here a layer runs as four kernels (the main kernel plus NAME_post1..3,
// launched back to back by the patched ZLUDA hook, patches/zluda/0008-native-post-phase-kernels.patch), each a fixed-size grid
// looping over independent units:
//   P0 (main)  per (window, head):              L2 norm + QKV + attention          -> O    (f16, global)
//   P1 (post1) per (window, out tile, tok tile): Wo chain over heads + residual     -> m, a (f16, global)
//   P2 (post2) per (window, hidden tile, tok):   fc1 chain + gelu                   -> hid  (f16, global)
//   P3 (post3) per (window, out tile, tok tile): fc2 chain over hidden from a       -> y    (layer output)
// Every value goes through the same f16 / f32 operations in the same order as pwin_wide.h (and pwin_layer.h), and
// everything passed between phases is an f16 value already, so the output is bit-identical. Units are distributed
// with compile-time grid sizes (no gridDim), the window count comes from the parameters (W, H, shift).
// Scratch buffers are module globals sized for MAXW windows; a launch with more windows does nothing (guard
// against out-of-bounds writes; the layer output is then not written).
#pragma once
#include "pwin_wide.h"

// layer kinds: core weights at p.w (+ CB for decoders); x0 rows from the layer input (mirror padded) or, for
// decoders, from the expand phase's scratch rows gX[window][token][C]
template <class L> struct PhaseCB
{
    template <class T> static constexpr int cb(decltype(T::CB)*) { return T::CB; }
    template <class T> static constexpr int cb(...) { return 0; }
    static constexpr int value = cb<L>(nullptr);
};
template <class L> __device__ __forceinline__ const half_t* phase_x0_row(const PwinParams& p, const half_t* gX, int wi, int bx,
                                                                         int by, int tok)
{
    if (gX)
        return gX + ((size_t)wi * 64 + tok) * L::C;
    const int X = mirror(8 * bx - p.sx + (tok & 7), p.W), Y = mirror(8 * by - p.sy + (tok >> 3), p.H);
    return (const half_t*)p.in + (size_t)(Y * p.W + X) * L::C;
}

template <class L> struct PhaseWin
{
    int GX, GY, n;
    __device__ PhaseWin(const PwinParams& p) : GX((p.W + p.sx + 7) / 8), GY((p.H + p.sy + 7) / 8), n(GX * GY) {}
};

// P0: unit u = (window w, head h); 4 waves = the 4 token tiles of the window
template <class L, int MAXW, int GRID>
__device__ void phase_attn(const PwinParams& p, const u8v* __restrict__ img, const u4v* __restrict__ bias, half_t* gO,
                           const half_t* gX = nullptr, int ub = -1)
{
    constexpr int C = L::C, KT = L::KT, H = L::H;
    __shared__ __attribute__((aligned(16))) half_t hb[64][LDSR(C)];
    __shared__ __attribute__((aligned(16))) half_t kl[1][64][LDSR(32)];
    __shared__ __attribute__((aligned(16))) half_t vt[1][32][LDSR(64)];
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const int t = __builtin_amdgcn_readfirstlane(threadIdx.z), tok = 16 * t + m;
    const PhaseWin<L> win(p);
    if (win.n > MAXW)
        return;
    const uint8_t* w = p.w + PhaseCB<L>::value;
    half_t* hrow = hb[tok];
    // ub >= 0 (single-unit launches): exactly unit ub; otherwise grid-strided over all units
    for (int u = ub >= 0 ? ub : (int)blockIdx.x; u < win.n * H; u += ub >= 0 ? 0x3fffffff : GRID)
    {
        const int wi = u / H, h = u % H, bx = wi % win.GX, by = wi / win.GX;
        const half_t* xin = phase_x0_row<L>(p, gX, wi, bx, by, tok);
        // L2 norm (wide_core): h1 = x0 * (rsqrt(sum x0^2) * gamma1)
        float ss = 0.0f;
        for (int kt = 0; kt < KT; ++kt)
        {
            const u8v xv = gload_row16(xin + 16 * kt);
#pragma unroll
            for (int c = 0; c < 16; ++c)
            {
                const half_t v = op_get(xv, c);
                ss += (float)(half_t)(v * v);
            }
        }
        const half_t r16 = PWIN_NORM_R16(ss, L::C, xin);
        for (int kt = (int)hf; kt < KT; kt += 2)
        {
            const u8v xv = gload_row16(xin + 16 * kt);
            const u8v gv = lds_row16((const half_t*)(w + L::G1) + 16 * kt);
            u8v hv;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                hv[j] = pack2(op_get(xv, 2 * j) * (half_t)(r16 * op_get(gv, 2 * j)),
                              op_get(xv, 2 * j + 1) * (half_t)(r16 * op_get(gv, 2 * j + 1)));
            store_row16(hrow + 16 * kt, hv);
        }
        __builtin_amdgcn_wave_barrier();
        op_t qop[2], oop[2];
        wide_qkv<L, 0, 6>(img, hrow, h, 0, tok, qop, kl, vt);
        block_sync();
        {
            half_t e[4][8];
            float rs = 0.0f;
#pragma unroll
            for (int kt = 0; kt < 4; ++kt)
            {
                const u4v bi = bias[((h * 4 + t) * 4 + kt) * 32 + l];
                f8v d;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    d[2 * i] = (float)lo16(bi[i]);
                    d[2 * i + 1] = (float)hi16(bi[i]);
                }
#pragma unroll
                for (int ds = 0; ds < 2; ++ds)
                    d = mma16(op_lds(&kl[0][16 * kt + m][16 * ds]), qop[ds], d);
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    e[kt][i] = softmax_e((half_t)d[i]);
                    if constexpr (!kShimRowSum)
                        rs += (float)e[kt][i];
                }
            }
            const half_t rc = PWIN_SOFTMAX_RC(rs, e);
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
                f8v d = splat8(0.0f);
#pragma unroll
                for (int kt = 0; kt < 4; ++kt)
                    d = mma16(op_lds(&vt[0][16 * nt + m][16 * kt]), pop[kt], d);
                oop[nt] = operand_from_f8(d);
            }
        }
        half_t* orow = gO + ((size_t)(wi * H + h) * 64 + tok) * 32;
        op_gstore(orow, oop[0], true);
        op_gstore(orow + 16, oop[1], true);
        block_sync(); // hb / kl / vt are rewritten by the next unit
    }
}

// global wave index helpers for the per-wave-unit phases (blocks of 4 waves, GRID blocks)
// (ub >= 0: only the 4 wave units of chunk ub, see phase_attn)
#define PHASE_WAVE_LOOP(UNITS, GRID)                                                                               \
    for (int uw = (ub >= 0 ? ub : (int)blockIdx.x) * 4 + (int)__builtin_amdgcn_readfirstlane(threadIdx.z); uw < (UNITS); \
         uw += ub >= 0 ? 0x3fffffff : (GRID) * 4)

// P1: unit = (window, output tile nt, token tile): Wo chain over heads, residual; writes the MLP input m and the
// fc2 accumulator init a = f16(x1 + b2) (rows of C halves per token)
template <class L, int MAXW, int GRID>
__device__ void phase_wo(const PwinParams& p, const u8v* __restrict__ img, const half_t* gO, half_t* gM, half_t* gA,
                         const half_t* gX = nullptr, int ub = -1)
{
    constexpr int C = L::C, KT = L::KT, H = L::H;
    const uint32_t m = lane_id() & 15;
    const PhaseWin<L> win(p);
    if (win.n > MAXW)
        return;
    const uint8_t* w = p.w + PhaseCB<L>::value;
    PHASE_WAVE_LOOP(win.n * KT * 4, GRID)
    {
        const int wi = uw / (KT * 4), r = uw % (KT * 4), nt = r >> 2, tt = r & 3, tok = 16 * tt + m;
        const int bx = wi % win.GX, by = wi / win.GX;
        f8v acc = acc_vec<L>(img, w, L::BO + 32 * nt, L::V_BO + nt);
        for (int hh = 0; hh < H; ++hh)
        {
            const half_t* orow = gO + ((size_t)(wi * H + hh) * 64 + tok) * 32;
            acc = mma16(op_imgt(img, L::T_WO + (2 * hh) * KT + nt), op_gload(orow), acc);
            acc = mma16(op_imgt(img, L::T_WO + (2 * hh + 1) * KT + nt), op_gload(orow + 16), acc);
        }
        const op_t xv = op_gload(phase_x0_row<L>(p, gX, wi, bx, by, tok) + 16 * nt);
        half_t mv[8], av[8], g2[8], b2[8];
        residual_vector<L>(w, 0, nt, g2);
        residual_vector<L>(w, 1, nt, b2);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const half_t x1 = (half_t)acc[i] + wm_op_acc_elem(xv, i);
            mv[i] = x1 * g2[i];
            av[i] = (half_t)(x1 + b2[i]);
        }
        const size_t row = ((size_t)wi * 64 + tok) * C + 16 * nt;
        op_gstore(gM + row, operand_from_dt(mv), true);
        op_gstore(gA + row, operand_from_dt(av), true);
    }
}

// P2: unit = (window, hidden tile j = 2 c + hn, token tile): fc1 chain over kt + gelu -> hidden rows (4C halves)
template <class L, int MAXW, int GRID>
__device__ void phase_fc1(const PwinParams& p, const u8v* __restrict__ img, const half_t* gM, half_t* gH, int ub = -1)
{
    constexpr int C = L::C, KT = L::KT, NJ = 2 * L::NMLP;
    const uint32_t m = lane_id() & 15;
    const PhaseWin<L> win(p);
    if (win.n > MAXW)
        return;
    const uint8_t* w = p.w + PhaseCB<L>::value;
    PHASE_WAVE_LOOP(win.n * NJ * 4, GRID)
    {
        const int wi = uw / (NJ * 4), r = uw % (NJ * 4), j = r >> 2, tt = r & 3, tok = 16 * tt + m;
        const int c = j >> 1, hn = j & 1;
        f8v d = acc_vec<L>(img, w, L::B1 + 64 * c + 32 * hn, L::V_B1 + 2 * c + hn);
        const half_t* mrow = gM + ((size_t)wi * 64 + tok) * C;
        for (int kt = 0; kt < KT; ++kt)
            d = mma16(op_imgt(img, L::T_W1 + (c * KT + kt) * 2 + hn), op_gload(mrow + 16 * kt), d);
        op_gstore(gH + ((size_t)wi * 64 + tok) * (4 * C) + 16 * j, gelu_op(d), true);
    }
}

// P3: unit = (window, output tile nt, token tile): fc2 chain over the hidden chunks from a; y -> layer output
template <class L, int MAXW, int GRID>
__device__ void phase_fc2(const PwinParams& p, const u8v* __restrict__ img, const half_t* gA, const half_t* gH, uint8_t* dst,
                          half_t* gY = nullptr, int ub = -1)
{
    constexpr int C = L::C, KT = L::KT;
    const uint32_t m = lane_id() & 15;
    const PhaseWin<L> win(p);
    if (win.n > MAXW)
        return;
    PHASE_WAVE_LOOP(win.n * KT * 4, GRID)
    {
        const int wi = uw / (KT * 4), r = uw % (KT * 4), nt = r >> 2, tt = r & 3, tok = 16 * tt + m;
        const int bx = wi % win.GX, by = wi / win.GX;
        const op_t av = op_gload(gA + ((size_t)wi * 64 + tok) * C + 16 * nt);
        f8v acc;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc[i] = (float)wm_op_acc_elem(av, i);
        const half_t* hrow = gH + ((size_t)wi * 64 + tok) * (4 * C);
        for (int cc = 0; cc < L::NMLP; ++cc)
        {
            acc = mma16(op_imgt(img, L::T_W2 + (cc * 2) * KT + nt), op_gload(hrow + 32 * cc), acc);
            acc = mma16(op_imgt(img, L::T_W2 + (cc * 2 + 1) * KT + nt), op_gload(hrow + 32 * cc + 16), acc);
        }
        const int Y0 = 8 * by - p.sy + (tok >> 3), X0 = 8 * bx - p.sx + (tok & 7);
        const bool inb = Y0 >= 0 && Y0 < p.H && X0 >= 0 && X0 < p.W;
        half_t* yout = (half_t*)dst + (size_t)(Y0 * p.W + X0) * C;
        op_t y = operand_from_f8(acc);
#if defined(D4R_ACCURACY) && defined(__GFX11__) && D4R_WMMA_LAYOUT == 11
        if constexpr (C == 160)
            y = ptx_mask_store_row(p, bx, by, tok, y);
#endif
        op_gstore(yout + 16 * nt, y, inb);
        if (gY)
            op_gstore(gY + ((size_t)wi * 64 + tok) * C + 16 * nt, y, true);
    }
}

// decoder expand (wide_decoder): unit = (window, sub-token q, tile nt) per wave; lanes = the 16 low-resolution
// tokens; x0 = patch expand of the previous decoder output + encoder skip -> gX rows (all 64 tokens of the window)
template <class L, int MAXW, int GRID>
__device__ void phase_expand(const PwinParams& p, const u8v* __restrict__ img, half_t* gX, int ub = -1)
{
    constexpr int C = L::C, KT = L::KT;
    const uint32_t m = lane_id() & 15;
    const PhaseWin<L> win(p);
    if (win.n > MAXW)
        return;
    PHASE_WAVE_LOOP(win.n * 4 * KT, GRID)
    {
        const int wi = uw / (4 * KT), r = uw % (4 * KT), q = r / KT, nt = r % KT;
        const int bx = wi % win.GX, by = wi / win.GX;
        const int W2 = p.W / 2, H2 = p.H / 2;
        const int ly = m >> 2, lx = m & 3;
        const int LX = mirror((8 * bx - p.sx) / 2 + lx, W2), LY = mirror((8 * by - p.sy) / 2 + ly, H2);
        const half_t* xl = (const half_t*)p.in + (size_t)(LY * W2 + LX) * L::CL;
        const int dy = q >> 1, dx = q & 1, tt = 8 * (2 * ly + dy) + 2 * lx + dx;
        const int SX = mirror(8 * bx - p.sx + (tt & 7), p.W), SY = mirror(8 * by - p.sy + (tt >> 3), p.H);
        const half_t* sk = (const half_t*)p.skip + (size_t)(SY * p.W + SX) * C;
        const int gnt = q * (C / 16) + nt;
        f8v d = acc_vec<L>(img, p.w, L::EXPB + 32 * gnt, L::V_EXP + gnt);
        for (int ks = 0; ks < L::KL; ++ks)
            d = mma16(op_imgt(img, L::T_EXP + ks * L::EXPN + gnt), op_gload(xl + 16 * ks), d);
        const op_t sv = op_gload(sk + 16 * nt);
        half_t xo[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            xo[i] = (half_t)d[i] + wm_op_acc_elem(sv, i);
        op_gstore(gX + ((size_t)wi * 64 + tt) * C + 16 * nt, operand_from_dt(xo), true);
    }
}

// encoder patch merge (wide_encoder): unit = (window, merge tile vt) per wave; lanes = the 16 merged tokens
template <class L, int MAXW, int GRID>
__device__ void phase_merge(const PwinParams& p, const u8v* __restrict__ img, const half_t* gY, int ub = -1)
{
    constexpr int C = L::C, KT = L::KT;
    const uint32_t m = lane_id() & 15;
    const PhaseWin<L> win(p);
    if (win.n > MAXW)
        return;
    const uint8_t* w = p.w;
    PHASE_WAVE_LOOP(win.n * L::PMT, GRID)
    {
        const int wi = uw / L::PMT, vt = uw % L::PMT;
        const int bx = wi % win.GX, by = wi / win.GX;
        const int my = m >> 2, mx = m & 3;
        const int Ym = (8 * by - p.sy) / 2 + my, Xm = (8 * bx - p.sx) / 2 + mx;
        const bool minb = Ym >= 0 && Ym < p.H / 2 && Xm >= 0 && Xm < p.W / 2;
        half_t* mout = (half_t*)p.out24 + (size_t)(Ym * (p.W / 2) + Xm) * L::COUT;
        f8v d = acc_vec<L>(img, w, L::PMB + 32 * vt, L::V_PM + vt);
        for (int ks = 0; ks < C / 4; ++ks)
        {
            const int sub = ks / KT, dy = sub >> 1, dx = sub & 1;
            const int src = 8 * (2 * my + dy) + 2 * mx + dx;
            d = mma16(op_imgt(img, L::T_PM + ks * L::PMT + vt), op_gload(gY + ((size_t)wi * 64 + src) * C + 16 * (ks % KT)), d);
        }
        const int gw = vt / (L::NPA / 16), u = vt % (L::NPA / 16), valid = L::NPW - 16 * u;
        op_gstore(mout + L::NPW * gw + 16 * u, operand_from_f8(d), minb, valid);
    }
}

// bottleneck layer (dec5 shape: core only, output to +24) as four phase kernels
#define PHASE_PLAIN(NAME, H, C, MAXW, G0, G1, G2, G3)                                                              \
    using NAME##_L = PwinLayout<H, C, 16>;                                                                         \
    __device__ u8v g_img[NAME##_L::IMG_SLOTS];                                                                     \
    __device__ u4v g_bias[NAME##_L::NBIAS];                                                                        \
    __device__ half_t g_O[(MAXW) * (H) * 64 * 32];                                                                 \
    __device__ half_t g_M[(MAXW) * 64 * (C)];                                                                      \
    __device__ half_t g_A[(MAXW) * 64 * (C)];                                                                      \
    __device__ half_t g_H[(MAXW) * 64 * 4 * (C)];                                                                  \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::PREP_ITEMS + 127) / 128;     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4;                                          \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_grid_x = (G0);                                        \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_offset = 64;                                 \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_post1_grid_x = (G1);                                  \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_post1_block_z = 4;                                    \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_post2_grid_x = (G2);                                  \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_post2_block_z = 4;                                    \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_post3_grid_x = (G3);                                  \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_post3_block_z = 4;                                    \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p)                                    \
    {                                                                                                              \
        pwin_prep_nomerge<NAME##_L>(p, g_img, g_bias);                                                             \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME(PwinParams p)                                           \
    {                                                                                                              \
        phase_attn<NAME##_L, MAXW, G0>(p, g_img, g_bias, g_O);                                                     \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post1(PwinParams p)                                   \
    {                                                                                                              \
        phase_wo<NAME##_L, MAXW, G1>(p, g_img, g_O, g_M, g_A);                                                     \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post2(PwinParams p)                                   \
    {                                                                                                              \
        phase_fc1<NAME##_L, MAXW, G2>(p, g_img, g_M, g_H);                                                         \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post3(PwinParams p)                                   \
    {                                                                                                              \
        phase_fc2<NAME##_L, MAXW, G3>(p, g_img, g_A, g_H, p.out24);                                                \
    }

#define PHASE_GLOBALS(NAME, H, C, MAXW)                                                                            \
    __device__ u8v g_img[NAME##_L::IMG_SLOTS];                                                                     \
    __device__ u4v g_bias[NAME##_L::NBIAS];                                                                        \
    __device__ half_t g_O[(MAXW) * (H) * 64 * 32];                                                                 \
    __device__ half_t g_M[(MAXW) * 64 * (C)];                                                                      \
    __device__ half_t g_A[(MAXW) * 64 * (C)];                                                                      \
    __device__ half_t g_H[(MAXW) * 64 * 4 * (C)];                                                                  \
    __device__ half_t g_X[(MAXW) * 64 * (C)];                                                                      \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_blocks = (NAME##_L::PREP_ITEMS + 127) / 128;     \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_block_z = 4;                                          \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_prep_key_offset = 64;
#define PHASE_POST(K, G)                                                                                           \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_post##K##_grid_x = (G);                               \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_post##K##_block_z = 4;

// decoder (wide shape, patch expand + skip, no output head): expand, attention, Wo, fc1, fc2
#define PHASE_DECODER(NAME, H, C, CL, MAXW, GE, G0, G1, G2, G3)                                                    \
    using NAME##_L = PwinDecLayout<H, C, CL>;                                                                      \
    PHASE_GLOBALS(NAME, H, C, MAXW)                                                                                \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_grid_x = (GE);                                        \
    PHASE_POST(1, G0) PHASE_POST(2, G1) PHASE_POST(3, G2) PHASE_POST(4, G3)                                        \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p) { pwin_dec_prep<NAME##_L>(p, g_img, g_bias); } \
    extern "C" __global__ void __launch_bounds__(128) NAME(PwinParams p) { phase_expand<NAME##_L, MAXW, GE>(p, g_img, g_X); } \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post1(PwinParams p)                                   \
    {                                                                                                              \
        phase_attn<NAME##_L, MAXW, G0>(p, g_img, g_bias, g_O, g_X);                                                \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post2(PwinParams p)                                   \
    {                                                                                                              \
        phase_wo<NAME##_L, MAXW, G1>(p, g_img, g_O, g_M, g_A, g_X);                                                \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post3(PwinParams p) { phase_fc1<NAME##_L, MAXW, G2>(p, g_img, g_M, g_H); } \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post4(PwinParams p)                                   \
    {                                                                                                              \
        phase_fc2<NAME##_L, MAXW, G3>(p, g_img, g_A, g_H, p.out24);                                                \
    }

// encoder (wide shape): attention, Wo, fc1, fc2 (+ full-resolution output), patch merge
#define PHASE_ENCODER(NAME, H, C, COUT, MAXW, G0, G1, G2, G3, GM)                                                  \
    using NAME##_L = PwinLayout<H, C, COUT>;                                                                       \
    PHASE_GLOBALS(NAME, H, C, MAXW)                                                                                \
    __device__ half_t g_Y[(MAXW) * 64 * (C)];                                                                      \
    extern "C" __device__ __attribute__((used)) uint32_t d4r_grid_x = (G0);                                        \
    PHASE_POST(1, G1) PHASE_POST(2, G2) PHASE_POST(3, G3) PHASE_POST(4, GM)                                        \
    extern "C" __global__ void __launch_bounds__(128) NAME##_prep(PwinParams p) { pwin_prep<NAME##_L>(p, g_img, g_bias); } \
    extern "C" __global__ void __launch_bounds__(128) NAME(PwinParams p) { phase_attn<NAME##_L, MAXW, G0>(p, g_img, g_bias, g_O); } \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post1(PwinParams p) { phase_wo<NAME##_L, MAXW, G1>(p, g_img, g_O, g_M, g_A); } \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post2(PwinParams p) { phase_fc1<NAME##_L, MAXW, G2>(p, g_img, g_M, g_H); } \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post3(PwinParams p)                                   \
    {                                                                                                              \
        phase_fc2<NAME##_L, MAXW, G3>(p, g_img, g_A, g_H, p.out32, g_Y);                                           \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME##_post4(PwinParams p) { phase_merge<NAME##_L, MAXW, GM>(p, g_img, g_Y); }
