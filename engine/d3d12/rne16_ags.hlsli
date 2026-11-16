// RDNA 3 (m_d3d12_build.py, D4R_RNE16_AGS_GFX11): dialect_hlsl.h's binary16 rounding (d4rRne16) on AGS's f32 -> f16
// accumulator conversion. AMD's D3D12 compiler converts f32 to binary16 toward zero (v_cvt_pkrtz; casts and f32tof16
// alike); NVIDIA's conversions round to nearest even, and so does this one (RX 7700 XT, 26.10.1: equal to it for all
// of 2^20 f32 inputs over binary16's range, subnormals and overflow to infinity included). One conversion per pair,
// where the integer emulation (D4R_RNE16_EXACT) takes about a dozen instructions a value. gfx11's f16 accumulator holds
// element r in the low half of register r: the pair goes in registers 0 and 1 and only those two are read back.
// Included by dialect_hlsl.h; ags_wmma.hlsli, if a shader includes it as well, shares the intrinsic UAV declared here.
#ifndef D4R_RNE16_AGS
#define D4R_RNE16_AGS
#ifndef D4R_AMD_INTRINSIC_UAV
#define D4R_AMD_INTRINSIC_UAV
RWByteAddressBuffer d4rAmdIntrinsic : register(u0, space2147420894);
#endif
uint d4rRneAmdOp(uint phase, uint data, uint a, uint b)
{
    uint result;
    d4rAmdIntrinsic.InterlockedCompareExchange(0x50000000u | (phase << 24) | (data << 8) | 0x32u, a, b, result);
    return result;
}
float2 d4rRne16(float2 v)
{
    // ags_wmma.hlsli's matrixConvert(MatrixAcc32 (0x26), MatrixAcc16 (0x24)) with registers 2..7 zero
    d4rRneAmdOp(0, 0, 0, 0);
    d4rRneAmdOp(0, 0, asuint(v.x), asuint(v.y));
    [unroll] for (uint i = 1; i < 4; ++i) d4rRneAmdOp(0, (i & 1u) | ((i >> 1) << 4), 0, 0);
    d4rRneAmdOp(1, 0x26, 0x24, 0);
    const uint h0 = d4rRneAmdOp(2, 0, 0, 0), h1 = d4rRneAmdOp(2, 1, 0, 0);
    const float2 r = float2(f16tof32(h0), f16tof32(h1));
#if D4R_RNE16_AGS_CHECK   // debugging only: NaN where the integer emulation (dialect_hlsl.h D4R_RNE16_EXACT) disagrees
    const uint2 b = asuint(v);
    const float2 sub = round(v * 16777216.0) * 5.9604644775390625e-08;
    const float2 nor = asfloat((b + 4095u + ((b >> 13u) & 1u)) & 0xffffe000u);
    float2 e = select(abs(v) < 6.103515625e-05, sub, nor);
    e = select(and(abs(e) > 65504.0, !isnan(e)), asfloat((b & 0x80000000u) | 0x7f800000u), e);
    const bool2 same = or(asuint(e) == asuint(r), and(isnan(e), isnan(r)));
    return float2(same.x ? r.x : asfloat(0x7fc00000u), same.y ? r.y : asfloat(0x7fc00000u));
#else
    return r;
#endif
}
float d4rRne16(float v) { return d4rRne16(float2(v, 0.0)).x; }
#endif
