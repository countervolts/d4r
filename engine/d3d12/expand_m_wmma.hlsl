// Decoder patch expansion at the parent-token grid. Each parent is processed
// once, instead of being duplicated into four rows of every shifted Swin window.
// The two-pixel halo preserves expansion-before-mirroring at boundary windows.
#include "../dialect_hlsl.h"
#include "../fp8_soft.glsl"
#ifndef D4R_M_NO_Q8
#define D4R_M_NO_Q8 0
#endif
#ifndef D4R_M_ACC_ROUND_K32
#define D4R_M_ACC_ROUND_K32 1
#endif
#define IS (XC + 8)
#define OS (C + 8)
#define TEMP (16 * IS / 2)
#define D4R_WMMA_SHARED_WORDS (TEMP + 16 * OS / 2)
#include "ags_wmma.hlsli"
#include "m_layer_params.hlsli"
ByteAddressBuffer weights : register(t1), vectors : register(t2);
Buffer<uint> inputs : register(t3), skips : register(t4);
RWBuffer<uint> output : register(u5);
int mirror(int v, int n) { v = abs(v); return min(v, 2 * n - 2 - v); }
Matrix16 halfAcc(Matrix16 v) { return matrixConvert(MatrixAcc16, MatrixAcc32, matrixConvert(MatrixAcc32, MatrixAcc16, v)); }
[WaveSize(32)] [numthreads(32, 1, 1)]
void main(uint3 group : SV_GroupID)
{
    const uint lane = WaveGetLaneIndex();
    const int w = asint(mParameter(0)), h = asint(mParameter(1));
    const int2 start = int2(group.xy) * 4 - 1;
    [loop] for (uint i = lane; i < 16 * XC / 4; i += 32) {
        const uint row = i / (XC / 4), d = i % (XC / 4);
        const int2 pos = start + int2(row & 3, row >> 2);
        const uint token = mirror(pos.y, h / 2) * (w / 2) + mirror(pos.x, w / 2);
        const uint2 v = e4m3Unpack4(inputs[token * (XC / 4) + d]);
        matrixShared[row * IS / 2 + 2 * d] = v.x;
        matrixShared[row * IS / 2 + 2 * d + 1] = v.y;
    }
    GroupMemoryBarrierWithGroupSync();
    [loop] for (uint q = 0; q < 4; ++q) {
        [unroll] for (uint n = 0; n < C; n += 16) {
            Matrix16 acc = matrixConvert(MatrixAcc16, MatrixAcc32,
                matrixLoad(MatrixAcc16 | 0x80u, vectors, 2 * (off(69) + q * C + n), 8 * C));
            [loop] for (uint k = 0; k < XC; k += 32) {
                [unroll] for (uint j = 0; j < 2; ++j) {
                    const uint tile = (k / 16 + j) * (4 * C / 16) + (q * C + n) / 16;
                    Matrix16 a = matrixLoad(MatrixA16, weights, 2 * off(68) + tile * 512, 32);
                    Matrix16 b = matrixSharedLoad(MatrixB16 | 0x80u, (k + 16 * j) / 2, IS / 2);
                    acc = matrixMultiplyAdd(a, b, acc);
                }
#if D4R_M_ACC_ROUND_K32
                acc = halfAcc(acc);
#endif
            }
            matrixSharedStore(MatrixAcc16 | 0x80u, TEMP + n / 2, OS / 2, matrixConvert(MatrixAcc32, MatrixAcc16, acc));
        }
        GroupMemoryBarrierWithGroupSync();
        [loop] for (uint i = lane; i < 16 * C / 4; i += 32) {
            const uint row = i / (C / 4), d = i % (C / 4);
            const int2 pos = 2 * (start + int2(row & 3, row >> 2)) + int2(q & 1, q >> 1);
            if (all(pos >= -2) && pos.x < w + 2 && pos.y < h + 2) {
                const uint token = mirror(pos.y, h) * w + mirror(pos.x, w);
                const uint2 skip = e4m3Unpack4(skips[token * (C / 4) + d]);
                float2 a = unpackHalf2x16(matrixShared[TEMP + row * OS / 2 + 2 * d]);
                float2 b = unpackHalf2x16(matrixShared[TEMP + row * OS / 2 + 2 * d + 1]);
#if !D4R_M_NO_Q8
                a = unpackHalf2x16(e4m3PackF32(a.x, a.y));
                b = unpackHalf2x16(e4m3PackF32(b.x, b.y));
#endif
                const uint at = ((pos.y + 2) * (w + 4) + pos.x + 2) * (C / 2) + 2 * d;
                output[at] = packHalf2x16(a + unpackHalf2x16(skip.x));
                output[at + 1] = packHalf2x16(b + unpackHalf2x16(skip.y));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
