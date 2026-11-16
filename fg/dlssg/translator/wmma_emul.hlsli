// Debug stand-in for engine/d3d12/ags_wmma.hlsli (glsl2hlsl.py kernels, -DNVX_WMMA_EMULATE): the same matrix
// operations in plain HLSL, to tell a conversion error from an AGS one. Not exact to the hardware's summation order.
// Layout: lane L holds row L & 15, columns (L >> 4) * 8 .. + 7 as f32 bits in reg[0..7], for every matrix type.
#ifndef NVX_WMMA_EMUL
#define NVX_WMMA_EMUL
struct Matrix16 { uint reg[8]; };
static const uint MatrixA16 = 0x04, MatrixB16 = 0x14, MatrixAcc32 = 0x26, MatrixAcc16 = 0x24;
groupshared uint matrixShared[D4R_WMMA_SHARED_WORDS];
float nvx_emul_h(uint halfIndex) { return float(asfloat16(uint16_t((matrixShared[halfIndex >> 1] >> ((halfIndex & 1u) * 16u)) & 0xffffu))); }
Matrix16 matrixSharedLoad(uint type, uint wordOffset, uint wordStride)
{
    Matrix16 m;
    uint lane = WaveGetLaneIndex(), r = lane & 15u, c0 = (lane >> 4) * 8u;
    bool col = (type & 0x80u) != 0u;
    [unroll] for (uint j = 0; j < 8; ++j) {
        uint c = c0 + j;
        uint h = wordOffset * 2u + (col ? c * wordStride * 2u + r : r * wordStride * 2u + c);
        m.reg[j] = asuint(nvx_emul_h(h));
    }
    return m;
}
void matrixSharedStore(uint type, uint wordOffset, uint wordStride, Matrix16 m)
{
    uint lane = WaveGetLaneIndex(), r = lane & 15u, c0 = (lane >> 4) * 8u, o;
    bool col = (type & 0x80u) != 0u;
    [unroll] for (uint j = 0; j < 8; ++j) {
        uint c = c0 + j;
        uint h = wordOffset * 2u + (col ? c * wordStride * 2u + r : r * wordStride * 2u + c);
        uint sh = (h & 1u) * 16u;
        InterlockedAnd(matrixShared[h >> 1], ~(0xffffu << sh), o);
        InterlockedOr(matrixShared[h >> 1], uint(nvx_f32_to_f16_rne(asfloat(m.reg[j]))) << sh, o);
    }
}
Matrix16 matrixFill(uint type, uint value)
{
    Matrix16 m;
    [unroll] for (uint j = 0; j < 8; ++j) m.reg[j] = type == MatrixAcc32 ? value : asuint(float(asfloat16(uint16_t(value))));
    return m;
}
Matrix16 matrixConvert(uint from, uint to, Matrix16 m)
{
    if (to == MatrixAcc16)
        [unroll] for (uint j = 0; j < 8; ++j) m.reg[j] = asuint(float(asfloat16(nvx_f32_to_f16_rne(asfloat(m.reg[j])))));
    return m;
}
Matrix16 matrixMultiplyAdd(Matrix16 a, Matrix16 b, Matrix16 c)
{
    uint lane = WaveGetLaneIndex(), r = lane & 15u, c0 = (lane >> 4) * 8u;
    Matrix16 res;
    [unroll] for (uint j = 0; j < 8; ++j) {
        float s = asfloat(c.reg[j]);
        uint cc = c0 + j;
        [unroll] for (uint k = 0; k < 16; ++k) {
            float av = asfloat(WaveReadLaneAt(a.reg[k & 7u], r | (k >= 8u ? 16u : 0u)));
            float bv = asfloat(WaveReadLaneAt(b.reg[cc & 7u], k | (cc >= 8u ? 16u : 0u)));
            s = mad(av, bv, s);
        }
        res.reg[j] = asuint(s);
    }
    return res;
}
#endif
