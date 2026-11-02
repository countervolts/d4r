// M/L encoder patch merge on RDNA 3 WMMA. Reads the preceding layer's FP8
// output and combines each 2x2-token patch into one token. The weight file is
// pack8's blocked matrices widened to binary16; vector offsets stay in halves.
#include "../dialect_hlsl.h"
#include "../fp8_soft.glsl"
#define STRIDE (C + 8)
#define TEMP (16 * STRIDE / 2)
#define D4R_WMMA_SHARED_WORDS (TEMP + 16 * 24 / 2)
#include "ags_wmma.hlsli"
#include "m_layer_params.hlsli"
ByteAddressBuffer weights : register(t1), vectors : register(t2);
Buffer<uint> inputs : register(t3);
RWBuffer<uint> output : register(u4);
[WaveSize(32)] [numthreads(32, 1, 1)]
void main(uint3 group : SV_GroupID)
{
    const uint lane = WaveGetLaneIndex(), w = mParameter(0) / 2, h = mParameter(1) / 2;
    Matrix16 acc[XC / 16];
    [unroll] for (uint n = 0; n < XC / 16; ++n)
        acc[n] = matrixConvert(MatrixAcc16, MatrixAcc32,
            matrixLoad(MatrixAcc16 | 0x80u, vectors, 2 * (off(69) + 16 * n), 2 * XC));
    // Stream one sub-position at a time instead of keeping all 4C rows in LDS.
    [loop] for (uint q = 0; q < 4; ++q) {
        [loop] for (uint i = lane; i < 16 * (C / 4); i += 32) {
            const uint row = i / (C / 4), ch = i % (C / 4);
            const uint2 pos = min(4 * group.xy + uint2(row & 3, row >> 2), uint2(w - 1, h - 1));
            const uint token = (2 * pos.y + (q >> 1)) * (2 * w) + 2 * pos.x + (q & 1);
            const uint2 v = e4m3Unpack4(inputs[token * (C / 4) + ch]);
            matrixShared[row * STRIDE / 2 + 2 * ch] = v.x;
            matrixShared[row * STRIDE / 2 + 2 * ch + 1] = v.y;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint n = 0; n < XC / 16; ++n) {
            [loop] for (uint k = 0; k < C; k += 16) {
                Matrix16 a = matrixLoad(MatrixA16, weights, 2 * off(68) + (((q * C + k) / 16) * (XC / 16) + n) * 512, 32);
                Matrix16 b = matrixSharedLoad(MatrixB16 | 0x80u, k / 2, STRIDE / 2);
                acc[n] = matrixMultiplyAdd(a, b, acc[n]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll] for (uint tile = 0; tile < XC / 16; ++tile) {
        const uint n = tile * 16;
        matrixSharedStore(MatrixAcc16 | 0x80u, TEMP, 12, matrixConvert(MatrixAcc32, MatrixAcc16, acc[tile]));
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint e = 0; e < 2; ++e) {
            const uint i = lane + 32 * e, row = i / 4, ch = i % 4;
            const uint2 pos = 4 * group.xy + uint2(row & 3, row >> 2);
            if (pos.x < w && pos.y < h) {
                const float2 a = clamp(unpackHalf2x16(matrixShared[TEMP + row * 12 + 2 * ch]), -6.2831855, 6.2831855);
                const float2 b = clamp(unpackHalf2x16(matrixShared[TEMP + row * 12 + 2 * ch + 1]), -6.2831855, 6.2831855);
                output[(pos.y * w + pos.x) * (XC / 4) + (n / 4) + ch] =
                    e4m3Code(a.x) | (e4m3Code(a.y) << 8) | (e4m3Code(b.x) << 16) | (e4m3Code(b.y) << 24);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
