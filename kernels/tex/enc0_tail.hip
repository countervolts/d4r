// Native tail of rrlite_enc0_4x4_*_folded (all flag combinations): the 2x2-pixel patch embedding GEMM (16 tokens x
// K 128 x N 64, e4m3 inputs, f16 accumulation per k32 step, f16 bias), clamp to +-2pi and e4m3 store.
// The feature code before it is ZLUDA's compile of the original PTX (enc0_tail.ptx calls this after
// the kernel's own shared-memory staging); the arithmetic reproduces ZLUDA's m16n8k32 lowering
// (scaled e4m3 widening, same k slot order, C * 2^-16 inside the f32 WMMA, f16 rounding per k32).
#include "tex_common.h"

// B operands (weights) as WMMA A images: index ((p * 4 + c) * 2 + h) * 16 + n16 for channel pair p
// (channels 16p..16p+15), k32 chunk c, k16 half h.
#if D4R_WMMA_LAYOUT == 12
__attribute__((device)) kslot g_enc0_w16[4 * 4 * 2 * 16];
#else
__attribute__((device)) u8v g_enc0_w16[4 * 4 * 2 * 16];
#endif
DEV __attribute__((used)) uint32_t d4r_prep_blocks = 4;
// the prep reads only prm.w (offset 0): the hook skips it while that pointer repeats
DEV __attribute__((used)) uint32_t d4r_prep_key_at = 0 + 1;

struct Enc0Params
{
    uint64_t w;
    uint8_t rest[352];
};

// every flag combination (mvhi/mvlo, hdr/ldr) has this tail; build_tex.sh names the prep after the kernel
#ifndef D4R_KERNEL_NAME
#define D4R_KERNEL_NAME rrlite_enc0_4x4_mvhi_hdr_folded
#endif
#define D4R_CAT2(a, b) a##b
#define D4R_CAT(a, b) D4R_CAT2(a, b)
KERNEL __attribute__((amdgpu_flat_work_group_size(128, 128))) void D4R_CAT(D4R_KERNEL_NAME, _prep)(Enc0Params prm)
{
    const uint32_t idx = __builtin_amdgcn_workgroup_id_x() * 128 + __builtin_amdgcn_workitem_id_x();
    if (idx >= 512)
        return;
    const uint8_t* w = (const uint8_t*)prm.w;
    const int n16 = idx & 15, h = (idx >> 4) & 1, c = (idx >> 5) & 3, p = idx >> 7;
    // NVIDIA B fragments: v4 load 4p + c of lane 4 * column + j holds {b0, b1} of tile 2p then 2p + 1
    const int g = n16 & 7, second = n16 >> 3;
    u4v d;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        d[j] = *(const uint32_t*)(w + 4096 + 512 * (4 * p + c) + 16 * (4 * g + j) + 4 * (2 * second + h));
#if D4R_WMMA_LAYOUT == 12
    g_enc0_w16[idx] = kslot_from16(d);
#else
    g_enc0_w16[idx] = operand16(d);
#endif
}

DEV void d4r_enc0_tail(uint32_t smem, uint64_t wp, uint64_t outp, uint32_t dimx, uint32_t dimy, int32_t ox, int32_t oy)
{
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const uint8_t* w = (const uint8_t*)wp;
    lds_u8* s = (lds_u8*)(uintptr_t)smem;

#if D4R_WMMA_LAYOUT == 12
    // activations: token m's 128 e4m3 features (addresses as below); this half's 8 bytes of each k16 part
    const uint32_t rowoff = 32 * (m & 3) + 256 * (m >> 2);
    const uint32_t choff[4] = {0, 16, 128, 144};
    kop X[4][2];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int h = 0; h < 2; ++h)
            X[c][h] = kop_from8(*(const __attribute__((address_space(3))) uint2_t*)(s + 1024 * h + rowoff + choff[c] + 8 * hf));

    const hv2 lo = {(half_t)-6.2831855f, (half_t)-6.2831855f}, hi = {(half_t)6.2831855f, (half_t)6.2831855f};
    // E[p]: e4m3 codes of channels 16p + 8hf .. 16p + 8hf + 7 (D^T element i = channel 16p + i + 8hf)
    uint32_t E[4][2];
#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        f8v cf;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            cf[i] = (float)*(const half_t*)(w + 12288 + 2 * (16 * p + i + 8 * hf));
        half_t acc[8];
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            const int slot = ((p * 4 + c) * 2) * 16 + m;
            cf = k32_e4m3(cf, kop_image(g_enc0_w16, slot), X[c][0], kop_image(g_enc0_w16, slot + 16), X[c][1]);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[i] = (half_t)cf[i];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                cf[i] = (float)acc[i];
        }
        uint32_t codes[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            hv2 v = {acc[2 * k], acc[2 * k + 1]};
            v = __builtin_elementwise_minimumnum(__builtin_elementwise_maximumnum(v, lo), hi);
            codes[k] = e4m3x2(__builtin_bit_cast(uint32_t, v));
        }
        E[p][0] = codes[0] | (codes[1] << 16);
        E[p][1] = codes[2] | (codes[3] << 16);
    }

    // plane pl, word 4 j + q = columns 2q, 2q + 1 of tile (2 pl, j) then of tile (2 pl + 1, j) (tile (p, j) =
    // channels 16p + 8j ..): this lane holds tiles j = hf, i.e. words 4 hf .. 4 hf + 3 of both planes
    const uint32_t H2 = dimy >> 1, W2 = dimx >> 1, rowstride = W2 * 8, plane = H2 * rowstride;
    const int32_t Y = oy / 2 + (int32_t)(m >> 2), Xc = ox / 2 + (int32_t)(m & 3);
    if ((uint32_t)Y < H2 && (uint32_t)Xc < W2)
    {
#pragma unroll
        for (int pl = 0; pl < 2; ++pl)
        {
            const uint32_t* p0 = E[2 * pl];
            const uint32_t* p1 = E[2 * pl + 1];
            const u4v words = {__builtin_amdgcn_perm(p1[0], p0[0], 0x05040100u), __builtin_amdgcn_perm(p1[0], p0[0], 0x07060302u),
                               __builtin_amdgcn_perm(p1[1], p0[1], 0x05040100u), __builtin_amdgcn_perm(p1[1], p0[1], 0x07060302u)};
            *(u4v*)((uint32_t*)outp + rowstride * (uint32_t)Y + 8 * (uint32_t)Xc + plane * pl + 4 * hf) = words;
        }
    }
#else
    // activations: token m's 128 e4m3 features; A[m][k] of chunk c at
    // 1024 * (k >> 4) + 32 * (m & 3) + 256 * (m >> 2) + (k & 15) + {0, 16, 128, 144}[c]
    const uint32_t rowoff = 32 * (m & 3) + 256 * (m >> 2);
    const uint32_t choff[4] = {0, 16, 128, 144};
    u8v X[4][2];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int h = 0; h < 2; ++h)
            X[c][h] = operand16(*(const __attribute__((address_space(3))) u4v*)(s + 1024 * h + rowoff + choff[c]));

    const hv2 lo = {(half_t)-6.2831855f, (half_t)-6.2831855f}, hi = {(half_t)6.2831855f, (half_t)6.2831855f};
    uint32_t E[4][2];
#pragma unroll
    for (int p = 0; p < 4; ++p)
    {
        // D^T layout: element i is channel 16p + 2i + hf of token m
        half_t acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc[i] = *(const half_t*)(w + 12288 + 2 * (16 * p + 2 * i + hf));
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            f8v cf;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                cf[i] = (float)acc[i];
            cf *= 0x1p-16f;
            const u8v* wv = &g_enc0_w16[((p * 4 + c) * 2) * 16 + m];
            cf = wmma(wv[0], X[c][0], cf);
            cf = wmma(wv[16], X[c][1], cf);
            cf = cf * 0x1p16f;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[i] = (half_t)cf[i];
        }
        uint32_t codes[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            hv2 v = {acc[2 * k], acc[2 * k + 1]};
            v = __builtin_elementwise_minimumnum(__builtin_elementwise_maximumnum(v, lo), hi);
            codes[k] = e4m3x2(__builtin_bit_cast(uint32_t, v));
        }
        // byte i of E[p][0] (E[p][1]) is column 2i + hf (2(i - 4) + hf) of tile 2p (2p + 1)
        E[p][0] = codes[0] | (codes[1] << 16);
        E[p][1] = codes[2] | (codes[3] << 16);
    }

    // lanes 0..15 write plane 0 (tiles 0..3), lanes 16..31 plane 1 (tiles 4..7): swap the other
    // plane's codes with the partner lane holding the other column parity
    uint32_t mine[2][2], theirs[2][2];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const uint32_t send = hf ? E[a][j] : E[2 + a][j];
            theirs[a][j] = other_half(send);
            mine[a][j] = hf ? E[2 + a][j] : E[a][j];
        }
    u8v words;
#pragma unroll
    for (int j = 0; j < 2; ++j)
    {
        const uint32_t ev0 = hf ? theirs[0][j] : mine[0][j], od0 = hf ? mine[0][j] : theirs[0][j];
        const uint32_t ev1 = hf ? theirs[1][j] : mine[1][j], od1 = hf ? mine[1][j] : theirs[1][j];
        const uint32_t a01 = __builtin_amdgcn_perm(od0, ev0, 0x05010400u), a23 = __builtin_amdgcn_perm(od0, ev0, 0x07030602u);
        const uint32_t b01 = __builtin_amdgcn_perm(od1, ev1, 0x05010400u), b23 = __builtin_amdgcn_perm(od1, ev1, 0x07030602u);
        words[4 * j + 0] = __builtin_amdgcn_perm(b01, a01, 0x05040100u);
        words[4 * j + 1] = __builtin_amdgcn_perm(b01, a01, 0x07060302u);
        words[4 * j + 2] = __builtin_amdgcn_perm(b23, a23, 0x05040100u);
        words[4 * j + 3] = __builtin_amdgcn_perm(b23, a23, 0x07060302u);
    }

    const uint32_t H2 = dimy >> 1, W2 = dimx >> 1, rowstride = W2 * 8, plane = H2 * rowstride;
    const int32_t Y = oy / 2 + (int32_t)(m >> 2), Xc = ox / 2 + (int32_t)(m & 3);
    if ((uint32_t)Y < H2 && (uint32_t)Xc < W2)
    {
        uint32_t* out = (uint32_t*)outp + rowstride * (uint32_t)Y + 8 * (uint32_t)Xc + plane * hf;
        *(u8v*)out = words;
    }
#endif
}
