// AMD D3D12 scalar f32 -> f16 RNE conversion via AGS opcode 0x1c (imm 1: v_cvt_f16_f32).
// Safe in wave32, wave64, and divergent control flow without extra fence overhead.
#ifndef D4R_RNE16_AGS_SCALAR_H
#define D4R_RNE16_AGS_SCALAR_H
#ifndef D4R_AMD_INTRINSIC_UAV
#define D4R_AMD_INTRINSIC_UAV
RWByteAddressBuffer d4rAmdIntrinsic : register(u0, space2147420894);
#endif
uint d4rRne16Bits(float v)
{
    uint result;
    d4rAmdIntrinsic.InterlockedCompareExchange(0x5000011cu, asuint(v), 0u, result);
    return result & 0xffffu;
}
float d4rRne16(float v) { return f16tof32(d4rRne16Bits(v)); }
float2 d4rRne16(float2 v) { return float2(d4rRne16(v.x), d4rRne16(v.y)); }
#endif
