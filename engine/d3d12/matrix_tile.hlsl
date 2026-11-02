// Packed per tile: A=f16[16][16], B=f16[16][16], C=f32[16][16].
// Output: f32[16][16]. Independent tiles exercise load + f32 accumulate + store.
ByteAddressBuffer parameters : register(t0);
ByteAddressBuffer inputs : register(t2);
RWByteAddressBuffer output : register(u0);
#include "ags_wmma.hlsli"
#ifndef MATRIX_STEPS
#define MATRIX_STEPS 1
#endif
[WaveSize(32)]
[numthreads(32, 1, 1)]
void main(uint3 group : SV_GroupID)
{
    const uint base = group.x * 2048;
    Matrix16 a = matrixLoad(MatrixA16, inputs, base, 32);
    Matrix16 b = matrixLoad(MatrixB16, inputs, base + 512, 32);
    Matrix16 c = matrixLoad(MatrixAcc32, inputs, base + 1024, 64);
    [loop] for (uint i = 0; i < MATRIX_STEPS; ++i) c = matrixMultiplyAdd(a, b, c);
    matrixStore(MatrixAcc32, output, group.x * 1024, 64, c);
}
