// Native M input product. Feature gathering and its DWORD layout stay shared
// with the scalar GLSL path. Each wave embeds its own 4x4-token block.
int2 blockOf(uint b) { return 8 * (2 * int2(gl_WorkGroupID.xy) + int2(b & 1u, b >> 1)) - p.shift; }
void main()
{
    const uint lane = gl_SubgroupInvocationID, wave = gl_SubgroupID;
    const float e1 = p.expo == 0.0 ? 1.0 : p.expo, e2 = p.expoPrev == 0.0 ? 1.0 : p.expoPrev;
    gatherPixel(lane, blockOf(wave), 16u * wave, e1, e2, true);
    gatherPixel(lane + 32u, blockOf(wave), 16u * wave, e1, e2, false);
    GroupMemoryBarrierWithGroupSync();
    const uint base = wave * 16 * 68;
    Matrix16 acc[4];
    [unroll] for (uint n = 0; n < 4; ++n)
        acc[n] = matrixConvert(MatrixAcc16, MatrixAcc32,
            matrixLoad(MatrixAcc16 | 0x80u, w16_raw, n * 32, 128));
    [loop] for (uint k = 0; k < 128; k += 32) {
        Matrix16 b0 = matrixSharedLoad(MatrixB16 | 0x80u, base + k / 2, 68);
        Matrix16 b1 = matrixSharedLoad(MatrixB16 | 0x80u, base + k / 2 + 8, 68);
        [unroll] for (uint n = 0; n < 4; ++n) {
            Matrix16 sum = matrixFill(MatrixAcc32, 0);
            sum = matrixMultiplyAdd(matrixLoad(MatrixA16 | 0x80u, w8_raw, k * 128 + n * 32, 128), b0, sum);
            sum = matrixMultiplyAdd(matrixLoad(MatrixA16 | 0x80u, w8_raw, (k + 16) * 128 + n * 32, 128), b1, sum);
            [unroll] for (uint i = 0; i < 8; ++i)
                acc[n].reg[i] = asuint(roundHalfSat(asfloat(sum.reg[i]) + asfloat(acc[n].reg[i])));
        }
    }
    // All input products finish before each wave reuses its feature region.
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint n = 0; n < 4; ++n) {
        matrixSharedStore(MatrixAcc16 | 0x80u, base, 12, matrixConvert(MatrixAcc32, MatrixAcc16, acc[n]));
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint e = 0; e < 2; ++e) {
            const uint i = lane + 32 * e, row = i / 4, ch = i % 4;
            const int2 pos = (blockOf(wave) >> 1) + int2(row & 3, row >> 2);
            if (all(pos >= 0) && pos.x < p.tok.x && pos.y < p.tok.y) {
                const float2 a = clamp(unpackHalf2x16(matrixShared[base + row * 12 + 2 * ch]), -6.2831855, 6.2831855);
                const float2 b = clamp(unpackHalf2x16(matrixShared[base + row * 12 + 2 * ch + 1]), -6.2831855, 6.2831855);
                outb.a[(pos.y * p.tok.x + pos.x) * 16 + 4 * n + ch] =
                    e4m3Code(a.x) | (e4m3Code(a.y) << 8) | (e4m3Code(b.x) << 16) | (e4m3Code(b.y) << 24);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
