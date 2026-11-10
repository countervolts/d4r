// Minimal AMD AGS WMMA wire protocol for f16 operands / f32 accumulators.
// Encoding reference: dxil-spirv shaders/ags/wmma_ags.h and vkd3d-proton's
// tests/shaders/sm_advanced. No AGS binary is required for the reserved space.
// All calls must execute uniformly across a wave32. Matrix register layout is
// opaque; access through load/store operations, never assumed lane indices.
#ifndef D4R_AGS_WMMA
#define D4R_AGS_WMMA
#ifndef D4R_AMD_INTRINSIC_UAV   // declared once with d3d12/rne16_ags.hlsli
#define D4R_AMD_INTRINSIC_UAV
RWByteAddressBuffer d4rAmdIntrinsic : register(u0, space2147420894);
#endif
struct Matrix16 { uint reg[8]; };
static const uint MatrixA16 = 0x04, MatrixB16 = 0x14, MatrixAcc32 = 0x26, MatrixAcc16 = 0x24;

uint amdOp(uint op, uint phase, uint data, uint a, uint b)
{
    uint result;
    d4rAmdIntrinsic.InterlockedCompareExchange(0x50000000u | (phase << 24) | (data << 8) | op, a, b, result);
    return result;
}
uint matrixIO(uint pair, uint role) { return (pair & 1u) | ((pair >> 1) << 4) | (role << 8); }
// D4R_AGS_SCALAR_REGISTERS (compile_m.py / compile_rr_dxil.py --d3d12-target vkd3d): M, L and RR read and write the
// registers of AGS results as scalars (gfx12's layout, see layer_m.hlsl), which AMD's driver allows. dxil-spirv
// translates an AGS matrix to an opaque cooperative matrix and none of that register arithmetic, so every wrapper
// below takes and returns the scalar layout through AGS element operations instead: matrixScalarUnpack extracts the
// elements (FP8 packs four to a register as AGS does, f16 two, f32 one), matrixScalarPack inserts them into a zero
// fill of the type. Element moves only, no arithmetic changes. Without it both are the identity.
#if D4R_AGS_SCALAR_REGISTERS
Matrix16 matrixScalarUnpack(uint type, Matrix16 nativeValue)
{
    Matrix16 result;
    [unroll] for (uint i = 0; i < 8; ++i) result.reg[i] = 0;
    const uint format = type & 15u;
    const uint elements = format == 10u ? 2u : 8u;
    [unroll] for (uint i = 0; i < elements; ++i) {
        [unroll] for (uint j = 0; j < 4; ++j)
            amdOp(0x30, 0, matrixIO(j, 0), nativeValue.reg[2*j], nativeValue.reg[2*j+1]);
        const uint v = amdOp(0x30, 1, type, i, 0);
        if (format == 4u) result.reg[i >> 1] |= (v & 65535u) << (16u * (i & 1u));
        else result.reg[i] = v;
    }
    return result;
}
Matrix16 matrixScalarPack(uint type, Matrix16 scalarValue)
{
    amdOp(0x33, 0, type, 0, 0);
    Matrix16 result;
    [unroll] for (uint i = 0; i < 8; ++i)
        result.reg[i] = amdOp(0x33, 1, (i & 3u) | ((i >> 2) << 4), 0, 0);
    const uint format = type & 15u;
    const uint elements = format == 10u ? 2u : 8u;
    [unroll] for (uint i = 0; i < elements; ++i) {
        const uint v = format == 4u ? (scalarValue.reg[i >> 1] >> (16u * (i & 1u))) & 65535u : scalarValue.reg[i];
        [unroll] for (uint j = 0; j < 4; ++j)
            amdOp(0x2f, 0, matrixIO(j, 0), result.reg[2*j], result.reg[2*j+1]);
        amdOp(0x2f, 1, type, i, v);
        [unroll] for (uint j = 0; j < 8; ++j)
            result.reg[j] = amdOp(0x2f, 2, (j & 3u) | ((j >> 2) << 4), 0, 0);
    }
    return result;
}
#else
#define matrixScalarUnpack(type, value) (value)
#define matrixScalarPack(type, value) (value)
#endif
Matrix16 matrixFill(uint type, uint value)
{
    amdOp(0x33, 0, type, value, 0);
    Matrix16 result;
    [unroll] for (uint i = 0; i < 8; ++i)
        result.reg[i] = amdOp(0x33, 1, (i & 3u) | ((i >> 2) << 4), 0, 0);
    return matrixScalarUnpack(type, result);
}
Matrix16 matrixConvert(uint from, uint to, Matrix16 m)
{
    m = matrixScalarPack(from, m);
    amdOp(0x32, 0, 0, 0, 0);
    [unroll] for (uint i = 0; i < 4; ++i)
        amdOp(0x32, 0, matrixIO(i, 0), m.reg[2*i], m.reg[2*i+1]);
    amdOp(0x32, 1, from, to, 0);
    Matrix16 result;
    [unroll] for (uint i = 0; i < 8; ++i)
        result.reg[i] = amdOp(0x32, 2, (i & 3u) | ((i >> 2) << 4), 0, 0);
    return matrixScalarUnpack(to, result);
}
Matrix16 matrixLoad(uint type, ByteAddressBuffer source, uint byteOffset, uint byteStride)
{
    uint address = amdOp(0x29, 0, 0, byteOffset, byteStride);
    uint hook = source.Load(address);
    amdOp(0x29, 1, type, hook, 0);
    Matrix16 m;
    [unroll] for (uint i = 0; i < 8; ++i)
        m.reg[i] = amdOp(0x29, 2, (i & 3u) | ((i >> 2) << 4), 0, 0);
    return matrixScalarUnpack(type, m);
}
void matrixStore(uint type, RWByteAddressBuffer target, uint byteOffset, uint byteStride, Matrix16 m)
{
    m = matrixScalarPack(type, m);
    uint address = amdOp(0x2a, 0, 0, byteOffset, byteStride);
    uint hook = target.Load(address);
    [unroll] for (uint i = 0; i < 4; ++i)
        amdOp(0x2a, 1, matrixIO(i, 0), m.reg[2*i], m.reg[2*i+1]);
    amdOp(0x2a, 2, type, hook, 0);
}
#if D4R_ACCUMULATE_F16
#define D4R_AGS_MMA_ACC MatrixAcc16
#else
#define D4R_AGS_MMA_ACC MatrixAcc32
#endif
Matrix16 matrixMultiplyAdd(Matrix16 a, Matrix16 b, Matrix16 c)
{
    a = matrixScalarPack(MatrixA16, a); b = matrixScalarPack(MatrixB16, b);
    c = matrixScalarPack(D4R_AGS_MMA_ACC, c);
    [unroll] for (uint i = 0; i < 4; ++i)
        amdOp(0x28, 0, matrixIO(i, 1), a.reg[2*i], a.reg[2*i+1]);
    [unroll] for (uint i = 0; i < 4; ++i)
        amdOp(0x28, 0, matrixIO(i, 2), b.reg[2*i], b.reg[2*i+1]);
    [unroll] for (uint i = 0; i < 4; ++i)
        amdOp(0x28, 0, matrixIO(i, 3), c.reg[2*i], c.reg[2*i+1]);
#if D4R_ACCUMULATE_F16
    amdOp(0x28, 1, 1, 0, 0); // f16_16x16x16_f16 (explicit approximate mode)
#else
    amdOp(0x28, 1, 5, 0, 0); // f32_16x16x16_f16
#endif
    Matrix16 result;
    [unroll] for (uint i = 0; i < 8; ++i)
        result.reg[i] = amdOp(0x28, 2, (i & 3u) | ((i >> 2) << 4), 0, 0);
    return matrixScalarUnpack(D4R_AGS_MMA_ACC, result);
}
// Element-wise a + b of two matrices of one type (AGS MatrixElementWise, op 1 = add).
Matrix16 matrixAdd(uint type, Matrix16 a, Matrix16 b)
{
    a = matrixScalarPack(type, a); b = matrixScalarPack(type, b);
    [unroll] for (uint i = 0; i < 4; ++i)
        amdOp(0x35, 0, matrixIO(i, 1), a.reg[2*i], a.reg[2*i+1]);
    [unroll] for (uint i = 0; i < 4; ++i)
        amdOp(0x35, 0, matrixIO(i, 2), b.reg[2*i], b.reg[2*i+1]);
    amdOp(0x35, 1, type, 1, 0);
    Matrix16 result;
    [unroll] for (uint i = 0; i < 8; ++i)
        result.reg[i] = amdOp(0x35, 2, (i & 3u) | ((i >> 2) << 4), 0, 0);
    return matrixScalarUnpack(type, result);
}
#ifdef D4R_WMMA_SHARED_WORDS
groupshared uint matrixShared[D4R_WMMA_SHARED_WORDS];
// AGS LDS offsets and strides are DWORD counts, unlike byte buffer operations.
Matrix16 matrixSharedLoad(uint type, uint wordOffset, uint wordStride)
{
    uint address = amdOp(0x2d, 0, 0, wordOffset, wordStride), hook;
    InterlockedAdd(matrixShared[address], 0, hook);
    amdOp(0x2d, 1, type, hook, 0);
    Matrix16 result;
    [unroll] for (uint i = 0; i < 8; ++i)
        result.reg[i] = amdOp(0x2d, 2, (i & 3u) | ((i >> 2) << 4), 0, 0);
    return matrixScalarUnpack(type, result);
}
void matrixSharedStore(uint type, uint wordOffset, uint wordStride, Matrix16 m)
{
    m = matrixScalarPack(type, m);
    uint address = amdOp(0x2e, 0, 0, wordOffset, wordStride), hook;
    InterlockedAdd(matrixShared[address], 0, hook);
    [unroll] for (uint i = 0; i < 4; ++i)
        amdOp(0x2e, 1, matrixIO(i, 0), m.reg[2*i], m.reg[2*i+1]);
    amdOp(0x2e, 2, type, hook, 0);
}
#endif
#endif
