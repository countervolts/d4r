// GLSL_KHR_cooperative_matrix as layer.comp uses it (16x16x16 f16 products, f32 accumulators, subgroup scope),
// on AMD's AGS WMMA intrinsics (ags_wmma.hlsli) for the D3D12 build. Include after defining
// D4R_WMMA_SHARED_WORDS: the shader's shared arrays are regions of AGS's matrixShared word array.
//
// Types: D4rMatA / D4rMatB (operands) and D4rAcc (f32 accumulator). A float16_t accumulator (HACC) is a D4rAcc
// whose values are binary16 numbers: HACC(m) rounds every element to binary16 (round to nearest even), ACC(h) is
// then exact, and HACC + HACC is the binary16 sum (f32 addition rounded once more to binary16, which for binary16
// operands gives the correctly rounded binary16 result). An f32 accumulator holds one element per register (8 per
// lane of a wave32), in the same order for every accumulator, so element-wise operations work on the registers.
// Memory operands: a ByteAddressBuffer wrapper (D4rBuf4: offsets and strides in f16vec4 units, as the GLSL) or a
// shared region (D4rLds4: f16vec4 units, two words each). Wrappers holding resources are initialized from the
// global resource itself and never copied (see coopMatLoad).
// Packaged D3D12 kernels select D4R_ACCUMULATE_F16: D4rAcc carries packed
// binary16 accumulators and matrix products round after each K tile. The f32
// implementation remains available to the isolated matrix/layer tests.
#ifndef D4R_COOPMAT_AGS
#define D4R_COOPMAT_AGS
#include "ags_wmma.hlsli"

#define d4rRoundHalf d4rRne16   // dialect_hlsl.h
#if D4R_ACCUMULATE_F16
static const uint D4rAccumulatorType = MatrixAcc16;
#else
static const uint D4rAccumulatorType = MatrixAcc32;
#endif

struct D4rMatA { Matrix16 m; };
struct D4rMatB { Matrix16 m; };
struct D4rAcc
{
    Matrix16 m;
    // only binary16 accumulators are added (layer.comp: HACC + HACC)
    D4rAcc operator+(D4rAcc b)
    {
        D4rAcc r;
#if D4R_ACCUMULATE_F16 && D4R_AGS_ELEMENTWISE_ADD
        // binary16 add of the packed accumulators, without register access: dxil-spirv (vkd3d-proton) only
        // translates matrices whose registers pass between AGS operations unchanged
        r.m = matrixAdd(MatrixAcc16, m, b.m);
        return r;
#endif
#if D4R_ACCUMULATE_F16
        Matrix16 a32 = matrixConvert(MatrixAcc16, MatrixAcc32, m);
        Matrix16 b32 = matrixConvert(MatrixAcc16, MatrixAcc32, b.m);
#else
        Matrix16 a32 = m, b32 = b.m;
#endif
        [unroll] for (uint i = 0; i < 8; ++i) r.m.reg[i] = asuint(d4rRoundHalf(asfloat(a32.reg[i]) + asfloat(b32.reg[i])));
#if D4R_ACCUMULATE_F16
        r.m = matrixConvert(MatrixAcc32, MatrixAcc16, r.m);
#endif
        return r;
    }
};
D4rAcc d4r_acc(float v) { D4rAcc r; r.m = matrixFill(D4rAccumulatorType, D4rAccumulatorType == MatrixAcc16 ? f32tof16(v) : asuint(v)); return r; }
D4rAcc d4r_acc(D4rAcc v) { return v; }
D4rAcc d4r_hacc(D4rAcc v)
{
#if D4R_ACCUMULATE_F16
    return v;
#else
    D4rAcc r;
    [unroll] for (uint i = 0; i < 8; ++i) r.m.reg[i] = asuint(d4rRoundHalf(asfloat(v.m.reg[i])));
    return r;
#endif
}
D4rAcc d4r_mma(D4rMatA a, D4rMatB b, D4rAcc c) { D4rAcc r; r.m = matrixMultiplyAdd(a.m, b.m, c.m); return r; }

// ---------------------------------------------------------------- memory
struct D4rBuf4
{
    ByteAddressBuffer b;
    float16_t4 operator[](uint i) { return b.Load<float16_t4>(i * 8u); }
};
struct D4rLds4 { uint b; };
#define D4R_COL(layout) ((layout) != 0u ? 0x80u : 0u)
void d4r_load(out D4rMatA m, ByteAddressBuffer s, uint off, uint stride, uint layout)
{ m.m = matrixLoad(MatrixA16 | D4R_COL(layout), s, off * 8u, stride * 8u); }
void d4r_load(out D4rMatB m, ByteAddressBuffer s, uint off, uint stride, uint layout)
{ m.m = matrixLoad(MatrixB16 | D4R_COL(layout), s, off * 8u, stride * 8u); }
void d4r_load(out D4rAcc m, ByteAddressBuffer s, uint off, uint stride, uint layout)
{ m.m = matrixConvert(MatrixAcc16, D4rAccumulatorType, matrixLoad(MatrixAcc16 | D4R_COL(layout), s, off * 8u, stride * 8u)); }
void d4r_load(out D4rMatA m, uint s, uint off, uint stride, uint layout)
{ m.m = matrixSharedLoad(MatrixA16 | D4R_COL(layout), s + off * 2u, stride * 2u); }
void d4r_load(out D4rMatB m, uint s, uint off, uint stride, uint layout)
{ m.m = matrixSharedLoad(MatrixB16 | D4R_COL(layout), s + off * 2u, stride * 2u); }
void d4r_load(out D4rAcc m, uint s, uint off, uint stride, uint layout)
{ m.m = matrixConvert(MatrixAcc16, D4rAccumulatorType, matrixSharedLoad(MatrixAcc16 | D4R_COL(layout), s + off * 2u, stride * 2u)); }
// stores take a binary16 accumulator (GLSL: coopMatStore(HACC(m), ...)); the f32 -> f16 conversion is exact then
void d4r_store(D4rAcc m, uint s, uint off, uint stride, uint layout)
{ matrixSharedStore(MatrixAcc16 | D4R_COL(layout), s + off * 2u, stride * 2u, matrixConvert(D4rAccumulatorType, MatrixAcc16, m.m)); }

// the shared region as f16vec4 words, for the shader's scalar accesses
float16_t4 d4rLdsRead4(uint word)
{
    const uint a = matrixShared[word], b = matrixShared[word + 1u];
    return float16_t4(asfloat16(uint16_t(a & 0xffffu)), asfloat16(uint16_t(a >> 16)),
                      asfloat16(uint16_t(b & 0xffffu)), asfloat16(uint16_t(b >> 16)));
}
void d4rLdsWrite4(uint word, float16_t4 v)
{
    matrixShared[word] = uint(asuint16(v.x)) | (uint(asuint16(v.y)) << 16);
    matrixShared[word + 1u] = uint(asuint16(v.z)) | (uint(asuint16(v.w)) << 16);
}

// GLSL names
// The operand is the wrapper's member b (the buffer, or the region's first word): DXC (1.8) loses the binding of
// a resource-holding struct that is copied, into a function argument or another static, so none is copied.
#define coopMatLoad(m, arr, off, stride, layout) d4r_load(m, (arr).b, off, stride, layout)
#define coopMatStore(m, arr, off, stride, layout) d4r_store(m, (arr).b, off, stride, layout)
#define coopMatMulAdd d4r_mma
#endif
