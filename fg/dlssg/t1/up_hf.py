# T1: custom_upsample_hf_kernel of nvngx_dlssg 310.9.1 (DL2Net): per slice z (3 tensors)
# a bilinear 2x upsample (align_corners = false) of an NHWC f16 tensor into 16x16-pixel output tiles. NVIDIA's loop
# gives each thread one (pixel, channel) element and recomputes the pixel's source coordinates and weights for every
# channel, with scalar 16-bit loads and stores. T1: one invocation per tile pixel, all channels, channel pairs as
# 32-bit loads/stores, the same per-element operations: coordinates/weights in f32 exactly as the PTX, weights
# rounded to f16, and per channel the arithmetic of T0's compiled code (its ISA: row0 = fma(wx, v01, RNE((1-wx) v00)),
# t = RNE((1-wy) row0), row1 = fma(wx, v11, RNE((1-wx) v10)), out = fma(wy, row1, t)), written out with precise.
# Taken only when the slice's input and output channel counts are equal (as in the game); otherwise T0's own loop.
import _conv
HASH = '6b71a77fa712fd41d00673dc5be1319db61bd90861bce97604c0566db793c391'
ENTRY = 'custom_upsample_hf_kernel'

BODY = r'''
// the per-element arithmetic of T0's compiled code for one channel value (explicit, so no compiler can contract it
// differently): row0 = fma(wx, v01, RNE((1-wx) v00)), t = RNE((1-wy) row0), row1 = fma(wx, v11, RNE((1-wx) v10)),
// out = fma(wy, row1, t); f16vec2 form for two channels at once (per half identical)
f16vec2 nvx_uphf(f16vec2 wx, f16vec2 wy, f16vec2 wx1, f16vec2 wy1, f16vec2 v00, f16vec2 v01, f16vec2 v10, f16vec2 v11)
{
    precise f16vec2 m00 = wx1 * v00;
    precise f16vec2 row0 = fma(wx, v01, m00);
    precise f16vec2 t0 = wy1 * row0;
    precise f16vec2 m10 = wx1 * v10;
    precise f16vec2 row1 = fma(wx, v11, m10);
    precise f16vec2 o = fma(wy, row1, t0);
    return o;
}

void main() {
    uint z = gl_WorkGroupID.z;
    if (z > 2u)
        return;     // never launched so (the PTX would use undefined registers)
    uint b = 24u * z;
    uint64_t op = nvx_p64(72u + b), ip = nvx_p64(0u + b);
    uint r117 = nvx_p32(12u + b), r116 = nvx_p32(16u + b), r115 = nvx_p32(20u + b);
    uint r114 = nvx_p32(84u + b), r118 = nvx_p32(88u + b), r119 = nvx_p32(92u + b);
    int x0 = int(gl_WorkGroupID.x << 4u), y0 = int(gl_WorkGroupID.y << 4u);
    if (x0 >= int(r118 + 4294967295u) || y0 >= int(r114 + 4294967295u))
        return;
    uint tid = (gl_LocalInvocationID.z * gl_WorkGroupSize.y + gl_LocalInvocationID.y) * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    uint nthr = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    // the PTX's element loop: e = (pixel, channel) with pixel = e / r115; T1: when input and output channel counts are
    // equal (always, in the game) one invocation takes a whole pixel, else the elements one by one as the PTX does
    bool whole = r115 == r119 && r115 != 0u;
    uint n = whole ? 256u : (r119 << 8u);
    for (uint e = tid; int(e) < int(n); e += nthr) {
        uint pix = whole ? e : uint(int(e) / int(r115));
        if (int(pix) > 255)
            continue;
        int x = int(pix & 15u) + x0, y = int(pix >> 4u) + y0;
        if ((x | y) < 0 || x >= int(r118) || y >= int(r114))
            continue;
        precise float f6 = float(x);
        precise float f7 = f6 + uintBitsToFloat(0xBF000000U);
        precise float f8 = f7 * uintBitsToFloat(0x3F000000U);
        precise float f9 = float(y);
        precise float f10 = f9 + uintBitsToFloat(0xBF000000U);
        precise float f11 = f10 * uintBitsToFloat(0x3F000000U);
        uint r83 = uint(trunc(f8)), r84 = uint(trunc(f11));
        precise float f12 = f8 + uintBitsToFloat(0x3F800000U);
        uint r85 = uint(trunc(f12));
        bool p14 = r85 >= r116;
        uint r43 = r116 + 4294967295u, r44 = r117 + 4294967295u;
        uint r86 = p14 ? r43 : r83, r87 = p14 ? r43 : r85;
        precise float f13 = f11 + uintBitsToFloat(0x3F800000U);
        uint r88 = uint(trunc(f13));
        bool p15 = r88 >= r117;
        uint r89 = p15 ? r44 : r84, r90 = p15 ? r44 : r88;
        bool p16 = f8 < uintBitsToFloat(0x00000000U);
        uint r91 = p16 ? 0u : r86, r92 = p16 ? 0u : r87;
        bool p17 = f11 < uintBitsToFloat(0x00000000U);
        uint r93 = p17 ? 0u : r89, r94 = p17 ? 0u : r90;
        precise float f14 = float(r93);
        precise float f2 = f11 - f14;
        precise float f15 = float(r91);
        precise float f3 = f8 - f15;
        precise float f16 = f14 - f11;
        precise float f4 = f16 + uintBitsToFloat(0x3F800000U);
        precise float f17 = f15 - f8;
        precise float f5 = f17 + uintBitsToFloat(0x3F800000U);
        f16vec2 wy = unpackFloat2x16(uint(nvx_f2h(f2)) * 0x10001u), wx = unpackFloat2x16(uint(nvx_f2h(f3)) * 0x10001u);
        f16vec2 wy1 = unpackFloat2x16(uint(nvx_f2h(f4)) * 0x10001u), wx1 = unpackFloat2x16(uint(nvx_f2h(f5)) * 0x10001u);
        uint a00 = (r93 * r116 + r91) * r115, a01 = (r93 * r116 + r92) * r115;
        uint a10 = (r94 * r116 + r91) * r115, a11 = (r94 * r116 + r92) * r115;
        uint ao = uint(y * int(r118) + x) * r119;
        uint c0 = whole ? 0u : uint(int(e) % int(r115)), c1 = whole ? r115 : c0 + 1u;
        uint c = c0;
        // pairs (32-bit loads/stores) where both the pixel offsets and the channel index are even
        if (whole && (r115 & 1u) == 0u) {
            for (; c < c1; c += 2u) {
                f16vec2 o = nvx_uphf(wx, wy, wx1, wy1,
                    unpackFloat2x16(NvxU32(ip + uint64_t(a00 + c) * 2ul).v), unpackFloat2x16(NvxU32(ip + uint64_t(a01 + c) * 2ul).v),
                    unpackFloat2x16(NvxU32(ip + uint64_t(a10 + c) * 2ul).v), unpackFloat2x16(NvxU32(ip + uint64_t(a11 + c) * 2ul).v));
                NvxU32(op + uint64_t(ao + c) * 2ul).v = packFloat2x16(o);
            }
        }
        for (; c < c1; c++) {
            f16vec2 o = nvx_uphf(wx, wy, wx1, wy1,
                f16vec2(uint16BitsToFloat16(NvxU16(ip + uint64_t(a00 + c) * 2ul).v)), f16vec2(uint16BitsToFloat16(NvxU16(ip + uint64_t(a01 + c) * 2ul).v)),
                f16vec2(uint16BitsToFloat16(NvxU16(ip + uint64_t(a10 + c) * 2ul).v)), f16vec2(uint16BitsToFloat16(NvxU16(ip + uint64_t(a11 + c) * 2ul).v)));
            NvxU16(op + uint64_t(ao + c) * 2ul).v = float16BitsToUint16(o.x);
        }
    }
}
'''


def build(src):
    return _conv.header(16) + BODY, ['t1_block 16 16 1']
