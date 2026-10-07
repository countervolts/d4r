// Native head of rrlite_dec0_4x4_folded: the token GEMM (16 half-resolution tokens x K 64 (two
// e4m3 planes) x N 128, f16 bias, f16 accumulation per k32 step) and the e4m3 staging of its result
// in shared memory, which the rest of the kernel (ZLUDA's compile of NVIDIA's PTX, dec0_head.ptx)
// reads. Arithmetic as ZLUDA's m16n8k32 lowering (see enc0_tail.hip); tokens are mirror padded.
#include "tex_common.h"

// weights as WMMA A images: index ((p * 2 + c) * 2 + h) * 16 + n16 for channel pair p (16p..16p+15),
// k32 chunk c (input plane), k16 half h
#if D4R_WMMA_LAYOUT == 12
__attribute__((device)) kslot g_dec0_w16[8 * 2 * 2 * 16];
#else
__attribute__((device)) u8v g_dec0_w16[8 * 2 * 2 * 16];
#endif
DEV __attribute__((used)) uint32_t d4r_prep_blocks = 4;
// the prep reads only prm.w (offset 0): the hook skips it while that pointer repeats
DEV __attribute__((used)) uint32_t d4r_prep_key_at = 0 + 1;

struct Dec0Params
{
    uint64_t w;
    uint8_t rest[160]; // 168-byte parameter block
};

KERNEL __attribute__((amdgpu_flat_work_group_size(128, 128))) void rrlite_dec0_4x4_folded_prep(Dec0Params prm)
{
    const uint32_t idx = __builtin_amdgcn_workgroup_id_x() * 128 + __builtin_amdgcn_workitem_id_x();
    if (idx >= 512)
        return;
    const uint8_t* w = (const uint8_t*)prm.w;
    const int n16 = idx & 15, h = (idx >> 4) & 1, c = (idx >> 5) & 1, p = idx >> 6;
    const int g = n16 & 7, second = n16 >> 3;
    u4v d;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        d[j] = *(const uint32_t*)(w + 512 * (2 * p + c) + 16 * (4 * g + j) + 4 * (2 * second + h));
#if D4R_WMMA_LAYOUT == 12
    g_dec0_w16[idx] = kslot_from16(d);
#else
    g_dec0_w16[idx] = operand16(d);
#endif
}

__attribute__((device)) static inline uint32_t mirror_clamp(int32_t v, uint32_t full, uint32_t half)
{
    const int32_t a = v < 0 ? -v : v;
    const int32_t m = __builtin_elementwise_min(a, (int32_t)(full & ~1u) - a - 2);
    return __builtin_elementwise_min(half - 1u, (uint32_t)m);
}

DEV void d4r_dec0_head(uint32_t smem, uint64_t wp, uint64_t inp, uint32_t dimx, uint32_t dimy, int32_t ox, int32_t oy)
{
    const uint32_t l = lane_id(), m = l & 15, hf = l >> 4;
    const uint8_t* w = (const uint8_t*)wp;
    lds_u8* s = (lds_u8*)(uintptr_t)smem;

    const uint32_t H2 = dimy >> 1, W2 = dimx >> 1, rowstride = W2 * 8, plane = H2 * rowstride;
    const uint32_t Y = mirror_clamp(oy / 2 + (int32_t)(m >> 2), dimy, H2);
    const uint32_t Xc = mirror_clamp(ox / 2 + (int32_t)(m & 3), dimx, W2);
    const uint32_t* in = (const uint32_t*)inp + rowstride * Y + 8 * Xc;
#if D4R_WMMA_LAYOUT == 12
    kop X[2][2];
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int h = 0; h < 2; ++h)
            X[c][h] = kop_from8(*(const uint2_t*)(in + plane * c + 4 * h + 2 * hf));

    // E[p]: e4m3 codes of channels 16p + 8hf .. 16p + 8hf + 7
    uint32_t E[8][2];
#pragma unroll
    for (int p = 0; p < 8; ++p)
    {
        f8v cf;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            cf[i] = (float)*(const half_t*)(w + 8192 + 2 * (16 * p + i + 8 * hf));
        half_t acc[8];
#pragma unroll
        for (int c = 0; c < 2; ++c)
        {
            const int slot = ((p * 2 + c) * 2) * 16 + m;
            cf = k32_e4m3(cf, kop_image(g_dec0_w16, slot), X[c][0], kop_image(g_dec0_w16, slot + 16), X[c][1]);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[i] = (half_t)cf[i];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                cf[i] = (float)acc[i];
        }
#ifdef D4R_TEX_HW_E4M3
        // cf holds the f16 results as f32. A NaN becomes -448 here (e4m3x2: the NaN code).
#pragma unroll
        for (int i = 0; i < 8; ++i)
            cf[i] = __builtin_amdgcn_fmed3f(cf[i], -448.0f, 448.0f);
        E[p][0] = e4m3x4_f32(cf[0], cf[1], cf[2], cf[3]);
        E[p][1] = e4m3x4_f32(cf[4], cf[5], cf[6], cf[7]);
#else
        uint32_t codes[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
            codes[k] = e4m3x2(pack2(acc[2 * k], acc[2 * k + 1]));
        E[p][0] = codes[0] | (codes[1] << 16);
        E[p][1] = codes[2] | (codes[3] << 16);
#endif
    }

    // staging (layout as below): this lane holds tiles 4G + hf and 4G + 2 + hf (channels 16p + 8hf ..), so it
    // writes H = hf itself
    const uint32_t base = 1024 * hf + 32 * (m & 3) + 256 * (m >> 2);
    const uint32_t goff[4] = {0, 16, 128, 144};
#pragma unroll
    for (int G = 0; G < 4; ++G)
    {
        const uint32_t* p0 = E[2 * G];
        const uint32_t* p1 = E[2 * G + 1];
        u4v words = {__builtin_amdgcn_perm(p1[0], p0[0], 0x05040100u), __builtin_amdgcn_perm(p1[0], p0[0], 0x07060302u),
                     __builtin_amdgcn_perm(p1[1], p0[1], 0x05040100u), __builtin_amdgcn_perm(p1[1], p0[1], 0x07060302u)};
        *(__attribute__((address_space(3))) u4v*)(s + base + goff[G]) = words;
    }
#else
    u8v X[2][2];
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int h = 0; h < 2; ++h)
            X[c][h] = operand16(*(const u4v*)(in + plane * c + 4 * h));

    uint32_t E[8][2];
#pragma unroll
    for (int p = 0; p < 8; ++p)
    {
        half_t acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc[i] = *(const half_t*)(w + 8192 + 2 * (16 * p + 2 * i + hf));
#pragma unroll
        for (int c = 0; c < 2; ++c)
        {
            f8v cf;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                cf[i] = (float)acc[i];
            cf *= 0x1p-16f;
            const u8v* wv = &g_dec0_w16[((p * 2 + c) * 2) * 16 + m];
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
            codes[k] = e4m3x2(pack2(acc[2 * k], acc[2 * k + 1]));
        E[p][0] = codes[0] | (codes[1] << 16);
        E[p][1] = codes[2] | (codes[3] << 16);
    }

    // staging: word t of token m at 1024 * H + 32 * (m & 3) + 256 * (m >> 2) + {0, 16, 128, 144}[G] + 4t
    // holds columns 2t, 2t + 1 of tiles 4G + H and 4G + 2 + H; lanes 0..15 write H = 0, 16..31 H = 1
    uint32_t EV[8], OD[8];
#pragma unroll
    for (int p = 0; p < 8; ++p)
    {
        const uint32_t theirs = other_half(hf ? E[p][0] : E[p][1]);
        EV[p] = hf ? theirs : E[p][0];
        OD[p] = hf ? E[p][1] : theirs;
    }
    const uint32_t base = 1024 * hf + 32 * (m & 3) + 256 * (m >> 2);
    const uint32_t goff[4] = {0, 16, 128, 144};
#pragma unroll
    for (int G = 0; G < 4; ++G)
    {
        const uint32_t a01 = __builtin_amdgcn_perm(OD[2 * G], EV[2 * G], 0x05010400u);
        const uint32_t a23 = __builtin_amdgcn_perm(OD[2 * G], EV[2 * G], 0x07030602u);
        const uint32_t b01 = __builtin_amdgcn_perm(OD[2 * G + 1], EV[2 * G + 1], 0x05010400u);
        const uint32_t b23 = __builtin_amdgcn_perm(OD[2 * G + 1], EV[2 * G + 1], 0x07030602u);
        u4v words = {__builtin_amdgcn_perm(b01, a01, 0x05040100u), __builtin_amdgcn_perm(b01, a01, 0x07060302u),
                     __builtin_amdgcn_perm(b23, a23, 0x05040100u), __builtin_amdgcn_perm(b23, a23, 0x07060302u)};
        *(__attribute__((address_space(3))) u4v*)(s + base + goff[G]) = words;
    }
#endif
}
