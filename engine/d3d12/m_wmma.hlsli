// d4r engine, preset M on AMD's D3D12 driver: the FP8 / binary16 matrices of layer_m.hlsl and the input and expansion
// stages (enc0_m.comp, dec0_m.comp) on AGS WMMA (ags_wmma.hlsli). Register layouts and conversion rounding: see
// layer_m.hlsl. Include after ags_wmma.hlsli (with D4R_WMMA_SHARED_WORDS for ldsB8 / stsA8). M_EXACT selects the
// conversions NVIDIA's roundings need; KEEP_ZERO (default 0) is a zero the driver cannot prove (layer_m.hlsl).
#ifndef D4R_M_WMMA
#define D4R_M_WMMA
#ifndef M_EXACT
#define M_EXACT 0
#endif
#ifndef KEEP_ZERO
#define KEEP_ZERO 0u
#endif
static const uint MatrixA8 = 0x0a, MatrixB8 = 0x1a;
// FP8 operands are their two registers (uint2), binary16 accumulators their four (uint4), f32 accumulators Matrix16
Matrix16 m16(uint2 v) { Matrix16 m; m.reg[0] = v.x; m.reg[1] = v.y; [unroll] for (uint i = 2; i < 8; ++i) m.reg[i] = 0; return m; }
Matrix16 m16(uint4 v)
{
    Matrix16 m; m.reg[0] = v.x; m.reg[1] = v.y; m.reg[2] = v.z; m.reg[3] = v.w;
    [unroll] for (uint i = 4; i < 8; ++i) m.reg[i] = 0;
    return m;
}
Matrix16 mmaOp(uint op, Matrix16 a, Matrix16 b, Matrix16 c)
{
    // the scalar register layout through AGS element operations on vkd3d (ags_wmma.hlsli D4R_AGS_SCALAR_REGISTERS)
    a = matrixScalarPack(op == 7u ? MatrixA8 : MatrixA16, a);
    b = matrixScalarPack(op == 7u ? MatrixB8 : MatrixB16, b);
    c = matrixScalarPack(MatrixAcc32, c);
    [unroll] for (uint i = 0; i < 4; ++i) amdOp(0x28, 0, matrixIO(i, 1), a.reg[2*i], a.reg[2*i+1]);
    [unroll] for (uint i = 0; i < 4; ++i) amdOp(0x28, 0, matrixIO(i, 2), b.reg[2*i], b.reg[2*i+1]);
    [unroll] for (uint i = 0; i < 4; ++i) amdOp(0x28, 0, matrixIO(i, 3), c.reg[2*i], c.reg[2*i+1]);
    amdOp(0x28, 1, op, 0, 0);
    Matrix16 r;
    [unroll] for (uint i = 0; i < 8; ++i) r.reg[i] = amdOp(0x28, 2, (i & 3u) | ((i >> 2) << 4), 0, 0);
    return matrixScalarUnpack(MatrixAcc32, r);
}
Matrix16 mma8(uint2 a, uint2 b, Matrix16 c) { return mmaOp(7, m16(a), m16(b), c); }      // f32_16x16x16_fp8_fp8
Matrix16 mma16(uint4 a, uint4 b, Matrix16 c) { return mmaOp(5, m16(a), m16(b), c); }     // f32_16x16x16_f16
Matrix16 convert(uint from, uint to, uint saturate, Matrix16 m)
{
    m = matrixScalarPack(from, m);
    amdOp(0x32, 0, 0, 0, 0);
    [unroll] for (uint i = 0; i < 4; ++i) amdOp(0x32, 0, matrixIO(i, 0), m.reg[2*i], m.reg[2*i+1]);
    amdOp(0x32, 1, from, to, saturate);
    Matrix16 r;
    [unroll] for (uint i = 0; i < 8; ++i) r.reg[i] = amdOp(0x32, 2, (i & 3u) | ((i >> 2) << 4), 0, 0);
    return matrixScalarUnpack(to, r);
}
Matrix16 f32of(uint4 h) { return convert(MatrixAcc16, MatrixAcc32, 0, m16(h)); }
#if M_EXACT
uint4 halfOf(Matrix16 m) { const Matrix16 r = convert(MatrixAcc32, MatrixAcc16, 0, m); return uint4(r.reg[0], r.reg[1], r.reg[2], r.reg[3]) | KEEP_ZERO; }
uint2 fp8Of(Matrix16 m) { const Matrix16 r = convert(MatrixAcc32, MatrixB8, 1, m); return uint2(r.reg[0], r.reg[1]); }
uint2 fp8Of(uint4 h) { const Matrix16 r = convert(MatrixAcc16, MatrixB8, 1, m16(h)); return uint2(r.reg[0], r.reg[1]); }
#elif D4R_VKD3D
// M_RELAXED on vkd3d-proton (compile_m.py --d3d12-target vkd3d): the saturating conversion, as the Vulkan build's
// saturatedConvertEXT (RADV: the hardware conversion with its clamp; NaN stays NaN)
uint4 halfOf(Matrix16 m) { const Matrix16 r = convert(MatrixAcc32, MatrixAcc16, 0, m); return uint4(r.reg[0], r.reg[1], r.reg[2], r.reg[3]); }
uint2 fp8Of(Matrix16 m) { const Matrix16 r = convert(MatrixAcc32, MatrixB8, 1, m); return uint2(r.reg[0], r.reg[1]); }
// from binary16 through f32 (exact): RADV saturates an f32 source with one v_minimummaximum_f32, a binary16 one with
// an infinity compare and select per element (the hardware converts through f32 either way)
uint2 fp8Of(uint4 h) { return fp8Of(f32of(h)); }
#else
// M_RELAXED: the driver's saturating conversion maps +-infinity to +-448 with two compares and two selects per
// element; a clamp in f32 (one instruction; a NaN becomes a bound, as with SATF2_'s finite inputs) and the plain
// conversion instead
uint4 halfOf(Matrix16 m) { const Matrix16 r = convert(MatrixAcc32, MatrixAcc16, 0, m); return uint4(r.reg[0], r.reg[1], r.reg[2], r.reg[3]); }
uint2 fp8Of(Matrix16 m)
{
    [unroll] for (uint i = 0; i < 8; ++i) m.reg[i] = asuint(clamp(asfloat(m.reg[i]), -448.0, 448.0));
    const Matrix16 r = convert(MatrixAcc32, MatrixB8, 0, m); return uint2(r.reg[0], r.reg[1]);
}
uint2 fp8Of(uint4 h)
{
    [unroll] for (uint i = 0; i < 4; ++i)
    {
        const half2 v = clamp(asfloat16(uint16_t2(uint16_t(h[i] & 0xffffu), uint16_t(h[i] >> 16))), half(-448.0), half(448.0));
        const uint16_t2 b = asuint16(v); h[i] = uint(b.x) | (uint(b.y) << 16);
    }
    const Matrix16 r = convert(MatrixAcc16, MatrixB8, 0, m16(h)); return uint2(r.reg[0], r.reg[1]);
}
#endif
Matrix16 zero32() { Matrix16 m; [unroll] for (uint i = 0; i < 8; ++i) m.reg[i] = 0; return m; }
Matrix16 f32m(float v[8]) { Matrix16 m; [unroll] for (uint i = 0; i < 8; ++i) m.reg[i] = asuint(v[i]); return m; }
// one logical k32 (M_RELAXED): both k16 products chained in f32 from the binary16 accumulator, rounded once
uint4 k32h(uint4 dst, uint2 a0, uint2 b0, uint2 a1, uint2 b1) { return halfOf(mma8(a1, b1, mma8(a0, b0, f32of(dst)))); }
// a B operand of FP8 bytes in shared memory, column-major: element (k, n) at byte at + n * stride + k
uint2 ldsB8(uint at, uint stride)
{
    const uint lane = WaveGetLaneIndex(), w = (at + (lane & 15u) * stride + 8u * (lane >> 4)) >> 2;
    return uint2(matrixShared[w], matrixShared[w + 1u]);
}
// an A operand's eight FP8 elements (row lane & 15, columns 8 (lane >> 4) + i) to shared memory, row-major
void stsA8(uint at, uint stride, uint2 v)
{
    const uint lane = WaveGetLaneIndex(), w = (at + (lane & 15u) * stride + 8u * (lane >> 4)) >> 2;
    matrixShared[w] = v.x; matrixShared[w + 1u] = v.y;
}
#endif
