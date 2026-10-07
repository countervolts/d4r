// Native tail of preset L's unfolded rrlite_enc0_4x4_* (all flag combinations), RDNA4 native FP8 only.
//
// The kernel is ZLUDA's compile of NVIDIA's PTX up to the barrier that ends the feature staging: one wave per
// 8x8-pixel block has then written 32 e4m3 features per pixel to shared memory. Everything after it is this
// tail (make_ptx.py calls it instead of the original code):
//   embedding   x0 = W2 q8(relu(W1 f + b1)) + b2            (K 32 -> 32 -> 32, the layout of an MLP chunk)
//   Swin block  C 32, 2 heads, 4x4 windows                  (../m/swin_tok.h, one window after the other)
//   output      the block's e4m3 codes at pixel resolution  (the skip input of dec0)
//   patch merge 2x2 pixels (K 128) -> 64 channels, clamped to +-2 pi, at half resolution (the input of enc1)
// with NVIDIA's arithmetic (e4m3 activations, an f16 accumulator rounded after every k32 step) in the order of
// ZLUDA's lowering of it (swin_tex.h): harness frames are byte-identical to the original kernel's.
#include "tex_common.h"

#if !defined(D4R_TEX_FP8) || D4R_WMMA_LAYOUT != 12
#error "enc0l_tail.hip needs the gfx12 native-FP8 build (D4R_TEX_FP8=1)"
#endif

#include "swin_tex.h"

namespace
{
using namespace d4r_swin;
constexpr int EC = 32, ENH = 2, ENPM = 64, ENT = EC / 16;
using EL = TokLayout<EC, ENH, ENPM, 0>;
// weights: the embedding (w1 at 4096, b1, w2, b2), then the Swin block and the patch merge
constexpr int E_EMB = 4096, E_BLOCK = E_EMB + 2 * (32 * EC + 2 * EC);
// images: the block's tiles and constants (bases relative to the block), then the embedding's
constexpr int EW1 = EL::TOTAL, EW2 = EW1 + 64, EW_TOTAL = EW2 + 64;
constexpr int ECB1 = EL::CTOTAL, ECB2 = ECB1 + 4, EC_TOTAL = ECB2 + 2 * ENT;
constexpr TokTables<EC, ENH, ENPM, 0> g_el_tok = make_tok_tables<EC, ENH, ENPM, 0>();
constexpr TokDesc g_el_emb[2] = {{EW1, E_EMB, 512, 16 * EC, 1, 2, TOK_PI, 0, 0, 0},
                                 {EW2, E_EMB + 32 * EC + 2 * EC, 0, 512, 1, ENT, TOK_X, 0, 0, 0}};
constexpr TokConst g_el_embc[2] = {{ECB1, TC_PI, E_EMB + 32 * EC}, {ECB2, TC_X, E_EMB + 2 * (32 * EC) + 2 * EC}};
} // namespace

__attribute__((device)) d4r_swin::wslot g_el_w16[EW_TOTAL];
__attribute__((device)) d4r_swin::h8 g_el_c16[EC_TOTAL];
DEV __attribute__((used)) uint32_t d4r_prep_blocks = (EW_TOTAL + 127) / 128;
// the prep reads only prm.w (offset 0): the hook skips it while that pointer repeats
DEV __attribute__((used)) uint32_t d4r_prep_key_at = 0 + 1;

struct Enc0LParams
{
    uint64_t w;
    uint8_t rest[352];
};

#ifndef D4R_KERNEL_NAME
#define D4R_KERNEL_NAME rrlite_enc0_4x4_mvhi_hdr
#endif
#define D4R_CAT2(a, b) a##b
#define D4R_CAT(a, b) D4R_CAT2(a, b)
KERNEL __attribute__((amdgpu_flat_work_group_size(128, 128))) void D4R_CAT(D4R_KERNEL_NAME, _prep)(Enc0LParams prm)
{
    const int idx = (int)(__builtin_amdgcn_workgroup_id_x() * 128 + __builtin_amdgcn_workitem_id_x());
    const uint8_t* w = (const uint8_t*)prm.w;
    if (idx == 0)
        d4r_swin_one = (float)(prm.w != 0);
    if (idx < EL::TOTAL)
        expand_tok_weights(w + E_BLOCK, g_el_w16, g_el_tok.d, EL::NDESC, idx);
    else if (idx < EW_TOTAL)
        expand_tok_weights(w, g_el_w16, g_el_emb, 2, idx);
    if (idx < EL::CTOTAL)
        expand_tok_consts(w + E_BLOCK, g_el_c16, g_el_tok.c, EL::NCONST, idx);
    else if (idx < EC_TOTAL)
        expand_tok_consts(w, g_el_c16, g_el_embc, 2, idx);
}

// e4m3 codes of four halves that are already within +-448 (no saturation clamp)
__attribute__((device, always_inline)) static inline uint32_t el_codes4(d4r_swin::hv2 a, d4r_swin::hv2 b)
{
    const uint32_t w = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32((float)a[0], (float)a[1], 0, false);
    return (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32((float)b[0], (float)b[1], (int)w, true);
}

// smem: the staged features, byte k of pixel (px, py) of the block at 1024 (k >> 4) + 128 py + 16 px + (k & 15).
// out: e4m3 planes of 32 channels over tw x th tokens (pixels); merged: two such planes over tw / 2 x th / 2.
// (sx, sy): offset of the block grid against the token grid; (bx, by): this block.
DEV void d4r_enc0l_tail(uint32_t smem, uint64_t wp, uint64_t outp, uint64_t mergedp, int32_t sx, int32_t sy, int32_t tw,
                        int32_t th, int32_t bx, int32_t by)
{
    using K = EL;
    constexpr int C = EC, NT = ENT, XB = C + 4, NTP = ENPM / 16;
    const int lane = (int)lane_id(), l16 = lane & 15, hf = lane >> 4;
    const lds_u8* const s = (const lds_u8*)(uintptr_t)smem;
    (void)wp; // the images were prepared from it
    // this lane's bases into the two images, as in swin_tok_block
    uintptr_t chv = (uintptr_t)(g_el_c16 + hf), wlv = (uintptr_t)((const gop*)g_el_w16 + 2 * l16 + hf);
    asm("" : "+v"(chv));
    asm("" : "+v"(wlv));
    const h8* const ch = (const h8*)chv;
    const gop* const wl = (const gop*)wlv;
    auto wt = [&](int dst, int nt_count, int kc, int st, int nt) { return wl[2 * (dst + ((kc * 2 + st) * nt_count + nt) * 16)]; };

    // the output codes of the block's 64 tokens, for the patch merge
    __shared__ __attribute__((aligned(16))) uint8_t U[64 * XB];

#pragma unroll 1
    for (int wv = 0; wv < 4; ++wv)
    {
        const int T = 16 * wv + l16, px = tok_x(T), py = tok_y(T);
        // features: K slot 8 half + i of WMMA st is feature 16 st + 8 half + i, 8 contiguous staged bytes
        gop fa[2];
        {
            const lds_u8* f = s + 128 * py + 16 * px + 8 * hf;
#pragma unroll
            for (int st = 0; st < 2; ++st)
                fa[st] = *(const __attribute__((address_space(3))) u2v*)(f + 1024 * st);
        }
        // embedding: the layout of an MLP chunk (32 hidden as two tiles), ReLU instead of the GELU
        gop gz[2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
        {
            T16 z = tok_t16(ch[ECB1 + 2 * nt]);
            k32(z, wt(EW1, 2, 0, 0, nt), fa[0], wt(EW1, 2, 0, 1, nt), fa[1]);
            hv2 r[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                r[k] = __builtin_elementwise_maximumnum(z.pair(k), (hv2){(half_t)0.0f, (half_t)0.0f});
            gz[nt] = (u2v){codes4(r[0], r[1]), codes4(r[2], r[3])};
        }
        hv2 v[NT][4];
#pragma unroll
        for (int t = 0; t < NT; ++t)
        {
            T16 x0 = tok_t16(ch[ECB2 + 2 * t]);
            k32(x0, wt(EW2, NT, 0, 0, t), gz[0], wt(EW2, NT, 0, 1, t), gz[1]);
#pragma unroll
            for (int k = 0; k < 4; ++k)
                v[t][k] = x0.pair(k);
        }

        uint32_t co[NT][2];
        swin_tok_core<EC, ENH, ENPM, 0>(ch, wl, l16, v, co);

        const int X = 8 * bx - sx + px, Y = 8 * by - sy + py;
        if (X >= 0 && X < tw && Y >= 0 && Y < th)
        {
            uint8_t* b = (uint8_t*)outp + ((size_t)Y * (size_t)tw + (size_t)X) * 32 + tok_half(hf);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                *(uint32_t*)(b + tok_word(j >> 1, j & 1)) = co[j >> 1][j & 1];
        }
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int g = 0; g < 2; ++g)
                *(uint32_t*)(U + T * XB + tok_word(t, g) + tok_half(hf)) = co[t][g];
    }
    // every lane's codes are in U before any lane reads them
    wave_sync();

    // patch merge: lanes = the block's 16 merged tokens, the four source pixels' codes as in swin_tok_block
    {
        const int mx = l16 & 3, my = l16 >> 2;
        const uint8_t* const ub = U + ((2 * mx & 3) + 4 * (2 * my & 3) + 16 * (mx >> 1) + 32 * (my >> 1)) * XB + tok_half(hf);
        gop as[C / 8][2];
#pragma unroll
        for (int kc = 0; kc < C / 8; ++kc)
#pragma unroll
            for (int st = 0; st < 2; ++st)
            {
                uint32_t run4[2];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                {
                    const int k0 = tok_merge_k(kc, st, rr), q = k0 / C;
                    run4[rr] = *(const uint32_t*)(ub + ((q & 1) + 4 * (q >> 1)) * XB + k0 % C);
                }
                as[kc][st] = (u2v){run4[0], run4[1]};
            }
        const int W2 = tw / 2, H2 = th / 2;
        const int MX = (8 * bx - sx) / 2 + mx, MY = (8 * by - sy) / 2 + my;
        const bool inside = MX >= 0 && MX < W2 && MY >= 0 && MY < H2;
        const hv2 lim = {(half_t)6.2831855f, (half_t)6.2831855f};
#pragma unroll 1
        for (int u = 0; u < NTP; ++u)
        {
            T16 y = tok_t16(ch[K::CPMB + 2 * u]);
#pragma unroll
            for (int kc = 0; kc < C / 8; ++kc)
                k32(y, wt(K::PM, NTP, kc, 0, u), as[kc][0], wt(K::PM, NTP, kc, 1, u), as[kc][1]);
            hv2 c[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                c[k] = __builtin_elementwise_minimumnum(__builtin_elementwise_maximumnum(y.pair(k), -lim), lim);
            const u2v code = {el_codes4(c[0], c[1]), el_codes4(c[2], c[3])};
            if (inside)
                *(u2v*)((uint8_t*)mergedp + ((size_t)((u >> 1) * H2 + MY) * (size_t)W2 + (size_t)MX) * 32 + 16 * (u & 1) + 8 * hf) = code;
        }
    }
}
