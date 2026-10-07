// Native block of preset L's unfolded rrlite_dec0_4x4, RDNA4 native FP8 only.
//
// The original kernel is, per 8x8-pixel block (one wave): the patch expand of the block's 16 half-resolution
// tokens (K 64 -> 4 x 32 channels), x0 = f16(q8(expand) + skip) with enc0's pixel tokens as the skip input, a
// Swin block (C 32, 2 heads, 4x4 windows), and output code that turns the block's f16 result into four surfaces.
// make_ptx.py keeps the output code as ZLUDA's compile of the PTX and replaces everything before it with a call
// of d4r_dec0l_block. The function leaves the block's result in shared memory in the register layout the
// output code reads (the D fragments of NVIDIA's last 16 MMAs), which the edited PTX loads back.
// The arithmetic is the token-lane block of ../m/swin_tok.h in the order of ZLUDA's lowering (see
// enc0l_tail.hip): harness frames are byte-identical to the original kernel's.
#include "tex_common.h"

#if !defined(D4R_TEX_FP8) || D4R_WMMA_LAYOUT != 12
#error "dec0l_block.hip needs the gfx12 native-FP8 build (D4R_TEX_FP8=1)"
#endif

#include "swin_tex.h"

namespace
{
using namespace d4r_swin;
constexpr int DC = 32, DNH = 2, DCIN = 64, DNT = DC / 16;
using DL = TokLayout<DC, DNH, 0, DCIN>;
constexpr TokTables<DC, DNH, 0, DCIN> g_dl_tok = make_tok_tables<DC, DNH, 0, DCIN>();
} // namespace

__attribute__((device)) d4r_swin::wslot g_dl_w16[DL::TOTAL];
__attribute__((device)) d4r_swin::h8 g_dl_c16[DL::CTOTAL];
DEV __attribute__((used)) uint32_t d4r_prep_blocks = ((DL::TOTAL > DL::CTOTAL ? DL::TOTAL : DL::CTOTAL) + 127) / 128;
// the prep reads only prm.w (offset 0): the hook skips it while that pointer repeats
DEV __attribute__((used)) uint32_t d4r_prep_key_at = 0 + 1;

struct Dec0LParams
{
    uint64_t w;
    uint8_t rest[160]; // 168-byte parameter block
};

KERNEL __attribute__((amdgpu_flat_work_group_size(128, 128))) void rrlite_dec0_4x4_prep(Dec0LParams prm)
{
    const int idx = (int)(__builtin_amdgcn_workgroup_id_x() * 128 + __builtin_amdgcn_workitem_id_x());
    const uint8_t* w = (const uint8_t*)prm.w;
    if (idx == 0)
        d4r_swin_one = (float)(prm.w != 0);
    if (idx < DL::TOTAL)
        expand_tok_weights(w, g_dl_w16, g_dl_tok.d, DL::NDESC, idx);
    if (idx < DL::CTOTAL)
        expand_tok_consts(w, g_dl_c16, g_dl_tok.c, DL::NCONST, idx);
}

// in: e4m3 planes of 32 channels (two) over tw / 2 x th / 2 tokens; skip: one such plane over tw x th tokens
// (pixels). (sx, sy): offset of the block grid against the token grid; (bx, by): this block.
// Returns the shared-memory address of the result image: register R = 2 (4 m + j) + h of lane L (4 bytes at
// 4 (32 R + L)) is D register h of NVIDIA's MMA for token tile m (window) and column tile j, i.e. rows
// 16 m + (L >> 2) + 8 h, pair-order columns 8 j + 2 (L & 3), + 1 of the block's f16 output.
DEV uint32_t d4r_dec0l_block(uint64_t wp, uint64_t inp, uint64_t skipp, int32_t sx, int32_t sy, int32_t tw, int32_t th,
                             int32_t bx, int32_t by)
{
    using K = DL;
    constexpr int C = DC, NT = DNT, CIN = DCIN;
    const int lane = (int)lane_id(), l16 = lane & 15, hf = lane >> 4;
    (void)wp; // the images were prepared from it
    uintptr_t chv = (uintptr_t)(g_dl_c16 + hf), wlv = (uintptr_t)((const gop*)g_dl_w16 + 2 * l16 + hf);
    asm("" : "+v"(chv));
    asm("" : "+v"(wlv));
    const h8* const ch = (const h8*)chv;
    const gop* const wl = (const gop*)wlv;
    auto wt = [&](int dst, int nt_count, int kc, int st, int nt) { return wl[2 * (dst + ((kc * 2 + st) * nt_count + nt) * 16)]; };

    __shared__ __attribute__((aligned(16))) uint8_t E8[64 * C];
    __shared__ __attribute__((aligned(16))) uint32_t X2[32 * 32];

    // patch expand: lanes = the block's 16 half-resolution tokens, one sub-position (quadrant) after the other;
    // the codes of q8(expand) go to their child tokens' rows of E8 (as swin_tok_block, which uses a wave each)
    {
        const int W2 = tw / 2, H2 = th / 2;
        const int MX = mirror((8 * bx - sx) / 2 + (l16 & 3), W2), MY = mirror((8 * by - sy) / 2 + (l16 >> 2), H2);
        gop pa[CIN / 32][2];
#pragma unroll
        for (int kc = 0; kc < CIN / 32; ++kc)
        {
            const uint8_t* b = (const uint8_t*)inp + ((size_t)(kc * H2 + MY) * W2 + MX) * 32 + tok_half(hf);
#pragma unroll
            for (int st = 0; st < 2; ++st)
                pa[kc][st] = (u2v){*(const uint32_t*)(b + tok_word(st, 0)), *(const uint32_t*)(b + tok_word(st, 1))};
        }
#pragma unroll 1
        for (int q = 0; q < 4; ++q)
        {
            const int tx = 2 * (l16 & 3) + (q & 1), ty = 2 * (l16 >> 2) + (q >> 1);
            const int Tc = (tx & 3) + 4 * (ty & 3) + 16 * (tx >> 2) + 32 * (ty >> 2);
            const int pew = K::pe(0) + q * (CIN * C / 16);
#pragma unroll
            for (int ct = 0; ct < NT; ++ct)
            {
                T16 e = tok_t16(ch[K::CPEB + 2 * (NT * q + ct)]);
#pragma unroll
                for (int kc = 0; kc < CIN / 32; ++kc)
                    k32(e, wt(pew, NT, kc, 0, ct), pa[kc][0], wt(pew, NT, kc, 1, ct), pa[kc][1]);
                *(u2v*)(E8 + Tc * C + 16 * ct + 8 * hf) = tok_codes(e);
            }
        }
    }
    wave_sync();

#pragma unroll 1
    for (int wv = 0; wv < 4; ++wv)
    {
        const int T = 16 * wv + l16;
        const int Xm = mirror(8 * bx - sx + tok_x(T), tw), Ym = mirror(8 * by - sy + tok_y(T), th);
        const uint8_t* const sb = (const uint8_t*)skipp + ((size_t)Ym * (size_t)tw + (size_t)Xm) * 32 + tok_half(hf);
        hv2 v[NT][4];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int g = 0; g < 2; ++g)
            {
                const uint32_t ew = *(const uint32_t*)(E8 + T * C + tok_word(t, g) + tok_half(hf));
                const uint32_t sw = *(const uint32_t*)(sb + tok_word(t, g));
                v[t][2 * g] = tok_dec2<false>(ew) + tok_dec2<false>(sw);
                v[t][2 * g + 1] = tok_dec2<true>(ew) + tok_dec2<true>(sw);
            }

        uint32_t co[NT][2];
        hv2 x2[NT][4];
        swin_tok_core<DC, DNH, 0, DCIN>(ch, wl, l16, v, co, x2);
        (void)co;

        // pair k of tile t holds natural channels 16 t + 8 half + 2 k, + 1: pair-order columns 8 j + 2 u, + 1 with
        // j = t + 2 (k & 1), u = 2 half + (k >> 1); token row l16 of tile wv
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const int R = 2 * (4 * wv + t + 2 * (k & 1)) + (l16 >> 3), L = 4 * (l16 & 7) + 2 * hf + (k >> 1);
                X2[32 * R + L] = __builtin_bit_cast(uint32_t, x2[t][k]);
            }
    }
    wave_sync();
    return (uint32_t)(uintptr_t)(__attribute__((address_space(3))) void*)X2;
}
