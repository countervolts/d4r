// M/L shared Swin network on RDNA 3: FP8 storage, widened FP16 matrix operands,
// FP32 WMMA accumulators. One wave owns one 4x4-token window. Weight matrices
// retain pack8's blocked [K/16][N/16][N16][K16] layout, widened to binary16.
// This experimental schedule uses LDS between phases to bound live registers.
#include "../dialect_hlsl.h"
#include "../fp8_soft.glsl"
// Optional NVIDIA-like binary16 rounding after each K32. The relaxed schedule
// retains FP32 sums until their consumer, reducing conversion instructions.
#ifndef D4R_M_ACC_ROUND_K32
#define D4R_M_ACC_ROUND_K32 1
#endif
#ifndef D4R_M_NO_Q8
#define D4R_M_NO_Q8 0
#endif
#ifndef D4R_M_EMBED
#define D4R_M_EMBED 0
#endif
#ifndef D4R_M_OUTPUT_F16
#define D4R_M_OUTPUT_F16 0
#endif
#ifndef D4R_M_INPUT_F16
#define D4R_M_INPUT_F16 0
#endif
#ifndef D4R_M_UNROLL_C32_IO
#define D4R_M_UNROLL_C32_IO 0
#endif
#ifndef D4R_M_VECTOR_IO
#define D4R_M_VECTOR_IO 0
#endif
#if D4R_M_VECTOR_IO && (KIND != 0 || C % 16 != 0 || D4R_M_EMBED || D4R_M_OUTPUT_F16)
#error VECTOR_IO requires a non-embedding network block with FP8 output and C divisible by 16
#endif
// Only C32 uses the bounded schedule: larger shapes regress when their buffer
// loops are fully unrolled. The general loop path remains the default.
#ifndef D4R_M_PAIR_GEMM
#define D4R_M_PAIR_GEMM 0
#endif
#ifndef D4R_M_COMPACT_ATTN
#define D4R_M_COMPACT_ATTN 0
#endif
#ifndef D4R_M_DIRECT_V
#define D4R_M_DIRECT_V 0
#endif
#ifndef D4R_M_STREAM_Q
#define D4R_M_STREAM_Q 0
#endif
#if D4R_M_STREAM_Q && (!D4R_M_DIRECT_V || !D4R_M_NO_Q8)
#error STREAM_Q requires direct V and fast arithmetic
#endif
#ifndef D4R_M_DIRECT_MLP
#define D4R_M_DIRECT_MLP 0
#endif
#ifndef D4R_M_STREAM_MLP
#define D4R_M_STREAM_MLP 0
#endif
#if D4R_M_DIRECT_MLP && !D4R_M_NO_Q8
#error DIRECT_MLP requires fast arithmetic
#endif
#define RS (C + 8)
#define AS ((32 * NH > XC ? 32 * NH : XC) + 8)
#define XBASE 0
// Normalize in place after loading the residual into registers. The decoder
// holds selected expansion outputs in registers until all four products finish.
#define NBASE XBASE
#define QBASE (16 * RS / 2)
#if D4R_M_DIRECT_V
#define SBASE QBASE
#define VBASE QBASE
#define ATTN_WORDS (QBASE + 16 * 40 / 2)
#elif D4R_M_COMPACT_ATTN
#define SBASE QBASE
#define VBASE (QBASE + 16 * 24 / 2)
#define ATTN_WORDS (VBASE + 16 * 40 / 2)
#else
#define VBASE (QBASE + 16 * 40 / 2)
#define SBASE (VBASE + 16 * 40 / 2)
#define ATTN_WORDS (SBASE + 16 * 24 / 2)
#endif
#define ABASE QBASE
#if D4R_M_COMPACT_ATTN
#define HBASE VBASE
#else
#define HBASE QBASE
#endif
#if KIND == 2
#define D4R_WMMA_SHARED_WORDS (ATTN_WORDS > (ABASE + 16 * AS / 2) ? ATTN_WORDS : (ABASE + 16 * AS / 2))
#else
#define D4R_WMMA_SHARED_WORDS ATTN_WORDS
#endif
#include "ags_wmma.hlsli"
#include "m_layer_params.hlsli"
ByteAddressBuffer weights : register(t1), vectors : register(t2);
#if D4R_M_VECTOR_IO
Buffer<uint4> inputs : register(t3);
RWBuffer<uint4> output : register(u5);
#else
Buffer<uint> inputs : register(t3);
RWBuffer<uint> output : register(u5);
#endif
Buffer<uint> skips : register(t4);
uint pk(float2 v) { return packHalf2x16(v); }
float2 unpk(uint v) { return unpackHalf2x16(v); }
uint quant(float2 v) {
#if D4R_M_NO_Q8
    return pk(v);
#else
    return e4m3PackF32(v.x, v.y);
#endif
}
int mirror(int v, int n) { v = abs(v); return min(v, 2 * n - 2 - v); }
Matrix16 zeroAcc() { return matrixFill(MatrixAcc32, 0); }
Matrix16 halfAcc(Matrix16 v) { return matrixConvert(MatrixAcc16, MatrixAcc32, matrixConvert(MatrixAcc32, MatrixAcc16, v)); }
Matrix16 loadAcc(uint base, uint stride, uint n)
{
    return matrixConvert(MatrixAcc16, MatrixAcc32,
        matrixSharedLoad(MatrixAcc16 | 0x80u, base + n / 2, stride / 2));
}
Matrix16 biasAcc(uint at, uint stride)
{
    return matrixConvert(MatrixAcc16, MatrixAcc32, matrixLoad(MatrixAcc16 | 0x80u, vectors, 2 * at, 2 * stride));
}
Matrix16 addAcc(Matrix16 a, Matrix16 b)
{
    Matrix16 r;
    [unroll] for (uint i = 0; i < 8; ++i) r.reg[i] = asuint(asfloat(a.reg[i]) + asfloat(b.reg[i]));
    return r;
}
void storeAcc(Matrix16 a, uint base, uint stride, uint n)
{
    matrixSharedStore(MatrixAcc16 | 0x80u, base + n / 2, stride / 2, matrixConvert(MatrixAcc32, MatrixAcc16, a));
}
Matrix16 gemm(Matrix16 acc, uint at, uint ksize, uint nsize, uint n, uint base, uint stride)
{
    [loop] for (uint k = 0; k < ksize; k += 32) {
        [unroll] for (uint j = 0; j < 2; ++j) {
            const uint tile = (k / 16 + j) * (nsize / 16) + n / 16;
            Matrix16 a = matrixLoad(MatrixA16, weights, 2 * at + tile * 512, 32);
            Matrix16 b = matrixSharedLoad(MatrixB16 | 0x80u, base + (k + 16 * j) / 2, stride / 2);
            acc = matrixMultiplyAdd(a, b, acc);
        }
#if D4R_M_ACC_ROUND_K32
        acc = halfAcc(acc);
#endif
    }
    return acc;
}
// M_RESIDUAL_F32: the second MLP product keeps its accumulator across chunks.
Matrix16 gemm32(Matrix16 acc, uint at, uint n, uint base)
{
    [unroll] for (uint k = 0; k < 2; ++k) {
        Matrix16 a = matrixLoad(MatrixA16, weights, 2 * at + (k * (C / 16) + n / 16) * 512, 32);
        Matrix16 b = matrixSharedLoad(MatrixB16 | 0x80u, base + k * 8, 20);
        acc = matrixMultiplyAdd(a, b, acc);
    }
    return acc;
}
// Two adjacent output tiles share each activation operand. Keep product order
// unchanged so this schedule can be compared byte-for-byte with scalar tiles.
void gemmPair(uint bias, uint at, uint ksize, uint base, uint stride, uint destination, uint outStride)
{
    Matrix16 c0, c1;
    if (bias == ~0u) { c0 = zeroAcc(); c1 = zeroAcc(); }
    else { c0 = biasAcc(bias, 32); c1 = biasAcc(bias + 16, 32); }
    [loop] for (uint k = 0; k < ksize; k += 32) {
        [unroll] for (uint j = 0; j < 2; ++j) {
            const uint tile = (k / 16 + j) * 2;
            Matrix16 b = matrixSharedLoad(MatrixB16 | 0x80u, base + (k + 16 * j) / 2, stride / 2);
            c0 = matrixMultiplyAdd(matrixLoad(MatrixA16, weights, 2 * at + tile * 512, 32), b, c0);
            c1 = matrixMultiplyAdd(matrixLoad(MatrixA16, weights, 2 * at + (tile + 1) * 512, 32), b, c1);
        }
#if D4R_M_ACC_ROUND_K32
        c0 = halfAcc(c0); c1 = halfAcc(c1);
#endif
    }
    storeAcc(c0, destination, outStride, 0);
    storeAcc(c1, destination, outStride, 16);
}
void normalize(uint lane, uint gain)
{
    if (lane < 16) {
        float sum = 0;
        [unroll] for (uint d = 0; d < C / 2; ++d) {
            const float2 x = unpk(matrixShared[XBASE + lane * RS / 2 + d]);
            sum += dot(x, x);
        }
        const float r = d4rRne16(rsqrt(d4rRne16(d4rRne16(sum) + 0.0001220703125)));
        [unroll] for (uint d = 0; d < C / 2; ++d) {
            const float2 x = unpk(matrixShared[XBASE + lane * RS / 2 + d]);
            const float2 g = unpk(vectors.Load(2 * gain + 4 * d));
            matrixShared[NBASE + lane * RS / 2 + d] = quant(x * (r * g));
        }
    }
    GroupMemoryBarrierWithGroupSync();
}
uint expPair(float2 s)
{
    const float16_t2 t = clamp((float16_t2)s * (float16_t)(0.0972222164273262 / 5.656854152679443),
                                (float16_t)-0.55615234375, (float16_t)0.55615234375);
    const float16_t2 q = mad(t, mad(t, -t, (float16_t2)0.92730712890625), (float16_t2)1.375);
    return (pk(float2(q)) << 5) + 0x7ff88000;
}
float gelu(float x)
{
    const float y = clamp(x, -2.0, 2.0);
    return x * mad(y, mad(-abs(y), 0.08108133, 0.41216981), 0.5);
}
[WaveSize(32)] [numthreads(32, 1, 1)]
void main(uint3 group : SV_GroupID)
{
    const uint lane = WaveGetLaneIndex();
    const int w = asint(mParameter(0)), h = asint(mParameter(1));
    const int2 start = int2(group.xy) * 4 - int2(asint(mParameter(2)), asint(mParameter(3)));
    // Each thread owns complete DWORDs in shared memory; no packed-half write races.
#if KIND == 2
    [loop] for (uint i = lane; i < 16 * XC / 4; i += 32) {
        const uint row = i / (XC / 4), d = i % (XC / 4);
        const int2 pos = start / 2 + int2((row & 3) / 2, (row >> 2) / 2);
        const uint token = mirror(pos.y, h / 2) * (w / 2) + mirror(pos.x, w / 2);
        const uint2 v = e4m3Unpack4(inputs[token * (XC / 4) + d]);
        matrixShared[ABASE + row * AS / 2 + 2 * d] = v.x;
        matrixShared[ABASE + row * AS / 2 + 2 * d + 1] = v.y;
    }
    GroupMemoryBarrierWithGroupSync();
    uint decoded[C / 8][2];
    [loop] for (uint q = 0; q < 4; ++q) {
        [unroll] for (uint n = 0; n < C; n += 16) {
            Matrix16 a = gemm(biasAcc(off(69) + q * C + n, 4 * C), off(68), XC, 4 * C, q * C + n, ABASE, AS);
            storeAcc(a, NBASE, RS, n);
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint e = 0; e < C / 8; ++e) {
            const uint i = lane + 32 * e, row = i / (C / 4), d = i % (C / 4);
            if (((row >> 2) & 1) * 2 + (row & 1) == q) {
                const int2 pos = start + int2(row & 3, row >> 2);
                const uint token = mirror(pos.y, h) * w + mirror(pos.x, w);
                const uint2 s = e4m3Unpack4(skips[token * (C / 4) + d]);
                decoded[e][0] = pk(unpk(quant(unpk(matrixShared[NBASE + row * RS / 2 + 2 * d]))) + unpk(s.x));
                decoded[e][1] = pk(unpk(quant(unpk(matrixShared[NBASE + row * RS / 2 + 2 * d + 1]))) + unpk(s.y));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll] for (uint e = 0; e < C / 8; ++e) {
        const uint i = lane + 32 * e, row = i / (C / 4), d = i % (C / 4);
        matrixShared[XBASE + row * RS / 2 + 2 * d] = decoded[e][0];
        matrixShared[XBASE + row * RS / 2 + 2 * d + 1] = decoded[e][1];
    }
    GroupMemoryBarrierWithGroupSync();
#elif D4R_M_INPUT_F16 || D4R_M_EMBED
#if D4R_M_VECTOR_IO
    [loop] for (uint i = lane; i < 16 * C / 8; i += 32) {
        const uint row = i / (C / 8), d = i % (C / 8);
        const int2 pos = start + int2(row & 3, row >> 2);
        const uint4 v = inputs[((pos.y + 2) * (w + 4) + pos.x + 2) * (C / 8) + d];
        [unroll] for (uint j = 0; j < 4; ++j)
            matrixShared[XBASE + row * RS / 2 + 4 * d + j] = v[j];
    }
#else
#if D4R_M_UNROLL_C32_IO && C == 32
    [unroll] for (uint e = 0; e < C / 4; ++e) {
        const uint i = lane + 32 * e;
#else
    [loop] for (uint i = lane; i < 16 * C / 2; i += 32) {
#endif
        const uint row = i / (C / 2), d = i % (C / 2);
        const int2 pos = start + int2(row & 3, row >> 2);
        matrixShared[XBASE + row * RS / 2 + d] = inputs[((pos.y + 2) * (w + 4) + pos.x + 2) * (C / 2) + d];
    }
#endif
    GroupMemoryBarrierWithGroupSync();
#if D4R_M_EMBED
    [unroll] for (uint n = 0; n < 32; n += 16)
        storeAcc(gemm(biasAcc(off(61) + n, 32), off(60), 32, 32, n, XBASE, RS), HBASE, 40, n);
    GroupMemoryBarrierWithGroupSync();
    [loop] for (uint i = lane; i < 16 * 16; i += 32) {
        const uint at = HBASE + (i / 16) * 20 + i % 16;
        const float2 v = max(unpk(matrixShared[at]), 0.0);
        matrixShared[at] = e4m3PackF32(v.x, v.y);
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint n = 0; n < 32; n += 16)
        storeAcc(gemm(biasAcc(off(63) + n, 32), off(62), 32, 32, n, HBASE, 40), XBASE, RS, n);
    GroupMemoryBarrierWithGroupSync();
#endif
#else
#if D4R_M_VECTOR_IO
    [loop] for (uint i = lane; i < 16 * C / 16; i += 32) {
        const uint row = i / (C / 16), d = i % (C / 16);
        const int2 pos = start + int2(row & 3, row >> 2);
        const uint token = mirror(pos.y, h) * w + mirror(pos.x, w);
        const uint4 v = inputs[token * (C / 16) + d];
        [unroll] for (uint j = 0; j < 4; ++j) {
            const uint2 x = e4m3Unpack4(v[j]);
            matrixShared[XBASE + row * RS / 2 + 8 * d + 2 * j] = x.x;
            matrixShared[XBASE + row * RS / 2 + 8 * d + 2 * j + 1] = x.y;
        }
    }
#else
    [loop] for (uint i = lane; i < 16 * C / 4; i += 32) {
        const uint row = i / (C / 4), d = i % (C / 4);
        const int2 pos = start + int2(row & 3, row >> 2);
        const uint token = mirror(pos.y, h) * w + mirror(pos.x, w);
        const uint2 v = e4m3Unpack4(inputs[token * (C / 4) + d]);
        matrixShared[XBASE + row * RS / 2 + 2 * d] = v.x;
        matrixShared[XBASE + row * RS / 2 + 2 * d + 1] = v.y;
    }
#endif
    GroupMemoryBarrierWithGroupSync();
#endif
    // Keep the residual in WMMA registers across heads; each logical head rounds once.
    Matrix16 residual[C / 16];
    [unroll] for (uint n = 0; n < C / 16; ++n)
        residual[n] = halfAcc(addAcc(loadAcc(XBASE, RS, 16 * n), biasAcc(off(1) + 16 * n, C)));
    normalize(lane, off(0));
    [loop] for (uint head = 0; head < NH; ++head) {
#if D4R_M_STREAM_Q
        Matrix16 scores = matrixConvert(MatrixAcc16, MatrixAcc32,
            matrixLoad(MatrixAcc16, vectors, 2 * off(6 + 4 * head), 32));
        [unroll] for (uint k = 0; k < 2; ++k) {
            Matrix16 q = gemm(zeroAcc(), off(4 + 4 * head), C, 32, 16 * k, NBASE, RS);
            storeAcc(q, QBASE, 24, 0);
            GroupMemoryBarrierWithGroupSync();
            Matrix16 qa = matrixSharedLoad(MatrixA16, QBASE, 12);
            Matrix16 qb = matrixConvert(MatrixAcc32, MatrixB16, q);
            scores = matrixMultiplyAdd(qa, qb, scores);
            GroupMemoryBarrierWithGroupSync();
        }
#else
#if D4R_M_PAIR_GEMM
        gemmPair(~0u, off(4 + 4 * head), C, NBASE, RS, QBASE, 40);
#if !D4R_M_COMPACT_ATTN && !D4R_M_DIRECT_V
        gemmPair(~0u, off(5 + 4 * head), C, NBASE, RS, VBASE, 40);
#endif
#else
        [unroll] for (uint n = 0; n < 32; n += 16) {
            storeAcc(gemm(zeroAcc(), off(4 + 4 * head), C, 32, n, NBASE, RS), QBASE, 40, n);
#if !D4R_M_COMPACT_ATTN && !D4R_M_DIRECT_V
            storeAcc(gemm(zeroAcc(), off(5 + 4 * head), C, 32, n, NBASE, RS), VBASE, 40, n);
#endif
        }
#endif
        GroupMemoryBarrierWithGroupSync();
#if !D4R_M_NO_Q8
        [loop] for (uint i = lane; i < 16 * 16; i += 32)
            matrixShared[QBASE + (i / 16) * 20 + i % 16] = quant(unpk(matrixShared[QBASE + (i / 16) * 20 + i % 16]));
        GroupMemoryBarrierWithGroupSync();
#endif
        // Q * Q^T runs on WMMA as well. Scalar per-key dot products dominate
        // instruction count on the Windows compiler, even with native WMMA GEMMs.
        Matrix16 scores = matrixConvert(MatrixAcc16, MatrixAcc32,
            matrixLoad(MatrixAcc16, vectors, 2 * off(6 + 4 * head), 32));
        [unroll] for (uint k = 0; k < 2; ++k) {
            Matrix16 qa = matrixSharedLoad(MatrixA16, QBASE + 8 * k, 20);
            Matrix16 qb = matrixSharedLoad(MatrixB16 | 0x80u, QBASE + 8 * k, 20);
            scores = matrixMultiplyAdd(qa, qb, scores);
        }
#endif
        matrixSharedStore(MatrixAcc16, SBASE, 12, matrixConvert(MatrixAcc32, MatrixAcc16, scores));
        GroupMemoryBarrierWithGroupSync();
        if (lane < 16) {
            float2 e[8];
            float sum = 0;
            [unroll] for (uint k = 0; k < 8; ++k) {
                const float2 score = unpk(matrixShared[SBASE + lane * 12 + k]);
                e[k] = unpk(expPair(unpk(pk(score))));
                sum += e[k].x + e[k].y;
            }
            const float rc = d4rRne16(1.0 / d4rRne16(sum));
            [unroll] for (uint k = 0; k < 8; ++k) matrixShared[SBASE + lane * 12 + k] = pk(e[k] * rc);
        }
        GroupMemoryBarrierWithGroupSync();
#if D4R_M_DIRECT_V
        // MatrixCopy converts the accumulator to an A operand directly. The
        // logical channel/token layout is unchanged; V never needs an LDS copy.
        Matrix16 s = matrixSharedLoad(MatrixB16 | 0x80u, SBASE, 12);
        [unroll] for (uint n = 0; n < 32; n += 16) {
            Matrix16 acc = gemm(zeroAcc(), off(5 + 4 * head), C, 32, n, NBASE, RS);
            Matrix16 v = matrixConvert(MatrixAcc32, MatrixA16, acc);
            storeAcc(matrixMultiplyAdd(v, s, zeroAcc()), HBASE, 40, n);
        }
#else
#if D4R_M_COMPACT_ATTN
        // Q is dead after its score product. Scores occupy Q's old storage;
        // each output tile of H replaces the consumed channels of V in place.
#if D4R_M_PAIR_GEMM
        gemmPair(~0u, off(5 + 4 * head), C, NBASE, RS, VBASE, 40);
#else
        [unroll] for (uint n = 0; n < 32; n += 16)
            storeAcc(gemm(zeroAcc(), off(5 + 4 * head), C, 32, n, NBASE, RS), VBASE, 40, n);
#endif
        GroupMemoryBarrierWithGroupSync();
#endif
        [unroll] for (uint n = 0; n < 32; n += 16) {
            Matrix16 v = matrixSharedLoad(MatrixA16 | 0x80u, VBASE + n / 2, 20);
            Matrix16 s = matrixSharedLoad(MatrixB16 | 0x80u, SBASE, 12);
            // Both operands transpose their token-major shared arrays.
            storeAcc(matrixMultiplyAdd(v, s, zeroAcc()), HBASE, 40, n);
        }
#endif
        GroupMemoryBarrierWithGroupSync();
#if !D4R_M_NO_Q8
        [loop] for (uint i = lane; i < 16 * 16; i += 32)
            matrixShared[HBASE + (i / 16) * 20 + i % 16] = quant(unpk(matrixShared[HBASE + (i / 16) * 20 + i % 16]));
        GroupMemoryBarrierWithGroupSync();
#endif
        [unroll] for (uint n = 0; n < C / 16; ++n)
            residual[n] = gemm(residual[n], off(7 + 4 * head), 32, C, 16 * n, HBASE, 40);
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll] for (uint n = 0; n < C / 16; ++n) storeAcc(residual[n], XBASE, RS, 16 * n);
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint n = 0; n < C / 16; ++n)
        residual[n] = halfAcc(addAcc(loadAcc(XBASE, RS, 16 * n), biasAcc(off(3) + 16 * n, C)));
    normalize(lane, off(2));
    [loop] for (uint chunk = 0; chunk < C / 8; ++chunk) {
#if D4R_M_STREAM_MLP && D4R_M_NO_Q8 && C >= 64
        // Consume one hidden tile at a time to bound live state. Each residual
        // keeps the original K0 then K1 products and binary16 boundaries.
        [unroll] for (uint k = 0; k < 2; ++k) {
            Matrix16 hidden = halfAcc(gemm(biasAcc(off(21 + 3 * chunk) + 16 * k, 32),
                off(20 + 3 * chunk), C, 32, 16 * k, NBASE, RS));
            // Matrix narrowing matches the pre-GELU store. The native scalar
            // helper matches pk's post-GELU conversion before the B operand;
            // replacing either boundary with the other changes output bits.
            [unroll] for (uint i = 0; i < 8; ++i)
                hidden.reg[i] = asuint(d4rRne16(gelu(asfloat(hidden.reg[i]))));
            hidden = matrixConvert(MatrixAcc32, MatrixB16, hidden);
            [unroll] for (uint n = 0; n < C / 16; ++n) {
                Matrix16 a = matrixLoad(MatrixA16, weights,
                    2 * off(22 + 3 * chunk) + (k * (C / 16) + n) * 512, 32);
                residual[n] = matrixMultiplyAdd(a, hidden, residual[n]);
            }
        }
#elif D4R_M_DIRECT_MLP && C == 32
        Matrix16 hidden[2];
        [unroll] for (uint k = 0; k < 2; ++k) {
            hidden[k] = halfAcc(gemm(biasAcc(off(21 + 3 * chunk) + 16 * k, 32),
                off(20 + 3 * chunk), C, 32, 16 * k, NBASE, RS));
            // Match the existing half store before GELU and half operand after it.
            [unroll] for (uint i = 0; i < 8; ++i)
                hidden[k].reg[i] = asuint(gelu(asfloat(hidden[k].reg[i])));
            hidden[k] = matrixConvert(MatrixAcc32, MatrixB16, hidden[k]);
        }
        [unroll] for (uint n = 0; n < C / 16; ++n) {
            [unroll] for (uint k = 0; k < 2; ++k) {
                Matrix16 a = matrixLoad(MatrixA16, weights,
                    2 * off(22 + 3 * chunk) + (k * (C / 16) + n) * 512, 32);
                residual[n] = matrixMultiplyAdd(a, hidden[k], residual[n]);
            }
        }
#else
#if D4R_M_PAIR_GEMM
        gemmPair(off(21 + 3 * chunk), off(20 + 3 * chunk), C, NBASE, RS, HBASE, 40);
#else
        [unroll] for (uint n = 0; n < 32; n += 16)
            storeAcc(gemm(biasAcc(off(21 + 3 * chunk) + n, 32), off(20 + 3 * chunk), C, 32, n, NBASE, RS), HBASE, 40, n);
#endif
        GroupMemoryBarrierWithGroupSync();
        [loop] for (uint i = lane; i < 16 * 16; i += 32) {
            const uint at = HBASE + (i / 16) * 20 + i % 16;
            const float2 x = unpk(matrixShared[at]);
            matrixShared[at] = quant(float2(gelu(x.x), gelu(x.y)));
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint n = 0; n < C / 16; ++n)
            residual[n] = gemm32(residual[n], off(22 + 3 * chunk), 16 * n, HBASE);
        GroupMemoryBarrierWithGroupSync();
#endif
    }
    [unroll] for (uint n = 0; n < C / 16; ++n) storeAcc(residual[n], XBASE, RS, 16 * n);
    GroupMemoryBarrierWithGroupSync();
#if D4R_M_OUTPUT_F16
#if D4R_M_UNROLL_C32_IO && C == 32
    [unroll] for (uint e = 0; e < C / 4; ++e) {
        const uint i = lane + 32 * e;
#else
    [loop] for (uint i = lane; i < 16 * C / 2; i += 32) {
#endif
        const uint row = i / (C / 2), d = i % (C / 2);
        const int2 pos = start + int2(row & 3, row >> 2);
        if (all(pos >= 0) && pos.x < w && pos.y < h)
            output[(pos.y * w + pos.x) * (C / 2) + d] = matrixShared[XBASE + row * RS / 2 + d];
    }
#else
#if D4R_M_VECTOR_IO
    // Group four adjacent packed words per typed transaction. FP8 conversion
    // and token ownership are unchanged; only the I/O schedule differs.
    [loop] for (uint i = lane; i < 16 * C / 16; i += 32) {
        const uint row = i / (C / 16), d = i % (C / 16);
        const int2 pos = start + int2(row & 3, row >> 2);
        if (all(pos >= 0) && pos.x < w && pos.y < h) {
            uint4 packed;
            [unroll] for (uint j = 0; j < 4; ++j) {
                const float2 a = unpk(matrixShared[XBASE + row * RS / 2 + 8 * d + 2 * j]);
                const float2 b = unpk(matrixShared[XBASE + row * RS / 2 + 8 * d + 2 * j + 1]);
                packed[j] = e4m3Code(a.x) | (e4m3Code(a.y) << 8) | (e4m3Code(b.x) << 16) | (e4m3Code(b.y) << 24);
            }
            output[(pos.y * w + pos.x) * (C / 16) + d] = packed;
        }
    }
#else
#if D4R_M_UNROLL_C32_IO && C == 32
    [unroll] for (uint e = 0; e < C / 8; ++e) {
        const uint i = lane + 32 * e;
#else
    [loop] for (uint i = lane; i < 16 * C / 4; i += 32) {
#endif
        const uint row = i / (C / 4), d = i % (C / 4);
        const int2 pos = start + int2(row & 3, row >> 2);
        if (all(pos >= 0) && pos.x < w && pos.y < h) {
            const float2 a = unpk(matrixShared[XBASE + row * RS / 2 + 2 * d]);
            const float2 b = unpk(matrixShared[XBASE + row * RS / 2 + 2 * d + 1]);
            output[(pos.y * w + pos.x) * (C / 4) + d] = e4m3Code(a.x) | (e4m3Code(a.y) << 8) | (e4m3Code(b.x) << 16) | (e4m3Code(b.y) << 24);
        }
    }
#endif
#endif
}
