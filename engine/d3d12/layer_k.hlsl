// Preset K plain Swin block, ported from ../layer_dot2.comp.
// Existing pair-interleaved weights and <4i108I parameters are unchanged.
// Feasibility backend: no embedding, patch merge/expand, head or matrix acceleration yet.
// DXC: -T cs_6_6 -E main -enable-16bit-types -DC=160 -DNH=8
#ifndef C
#define C 160
#endif
#ifndef NH
#define NH 8
#endif
#define NW 2
#define CW (C / NW)
#define HC (C / 2)
#define RS (HC + 4)
#define RH (64 * RS)
#define KR 20
#define RHS (64 * KR)
#define G 2
#define HRS (16 * G + 4)
#if C < 32 || C % 32 != 0 || C > 160 || NH < 2 || NH % NW != 0 || NH > 8
#error Unsupported plain K shape
#endif
#if 4 * (RH + NW * RHS) > 32768
#error Shared memory exceeds the D3D12 32 KiB limit
#endif

// Typed DWORD SRVs keep the 448-byte parameter ABI tightly packed (cbuffer arrays do not).
Buffer<uint> parameters : register(t0);
Buffer<uint> weights : register(t1);
Buffer<uint> inputs : register(t2);
RWBuffer<uint> output : register(u0);
groupshared uint L[RH + NW * RHS];
groupshared uint waveCounter;
static uint lane, wave, gi;
uint offset(uint i) { return parameters[4 + i]; } // offsets in f16 elements
#include "k_math.hlsli"
uint pk(float a, float b) { return pack(float16_t2(a, b)); }
float lo(uint v) { return f16tof32(v & 65535); }
float hi(uint v) { return f16tof32(v >> 16); }
float d2(uint a, uint b, float c) { return dot2add(unpack(a), unpack(b), c); }
#define GEMM(NAME, NN) \
void NAME(inout float acc[2][NN], uint ab, uint stride, uint kpCount, uint base, uint ws) \
{ \
    [loop] for (uint kp = 0; kp < kpCount; ++kp) \
    { \
        const uint a0 = L[ab + lane * stride + kp], a1 = L[ab + (lane + 32) * stride + kp]; \
        [unroll] for (uint n = 0; n < NN; ++n) \
        { \
            const uint w = weights[base + kp * ws + n]; \
            acc[0][n] = d2(a0, w, acc[0][n]); acc[1][n] = d2(a1, w, acc[1][n]); \
        } \
    } \
}
GEMM(gemm32, 32)
GEMM(gemmCW, CW)
#define STORE_ROWS(NN, a, ab, stride, col) \
    [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < NN; n += 2) \
        L[(ab) + (lane + 32 * t) * (stride) + (col) + n / 2] = pk(a[t][n], a[t][n + 1]);
// AMD's masked 8-lane broadcast expressed with standard wave operations.
#define BC8(v, r) WaveReadLaneAt(v, (lane & ~7u) | (r))
#define SET_TABLE(dst, d, half_) \
    [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint j = 0; j < 4; ++j) \
        dst[t][j] = weights[otab / 2 + (lane + 32 * t) * 32 + ((((lane & ~7u) + 8 * (d)) & 31) + 32 * (half_)) / 2 + j];

[WaveSize(32)]
[numthreads(32 * NW, 1, 1)]
void main(uint thread : SV_GroupIndex, uint3 group : SV_GroupID)
{
    // Do not assume SV_GroupIndex / 32 equals the wave index. Assign an index once
    // per physical wave, then use that wave's logical lanes for every token copy.
    if (thread == 0) waveCounter = 0;
    GroupMemoryBarrierWithGroupSync();
    uint index = 0;
    if (WaveIsFirstLane()) InterlockedAdd(waveCounter, 1, index);
    wave = WaveReadLaneFirst(index); lane = WaveGetLaneIndex(); gi = wave * 32 + lane;
    const int W = int(parameters[0]), H = int(parameters[1]);
    const int X0 = 8 * int(group.x) - asint(parameters[2]), Y0 = 8 * int(group.y) - asint(parameters[3]);
    for (uint i = gi; i < 64 * HC; i += 32 * NW)
    {
        const uint tok = i / HC, d = i % HC;
        const uint src = uint(mirror(Y0 + int(tok >> 3), H) * W + mirror(X0 + int(tok & 7), W));
        L[tok * RS + d] = inputs[src * HC + d];
    }
    GroupMemoryBarrierWithGroupSync();
    uint residual[2][CW / 2];
    [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint d = 0; d < CW / 2; ++d)
        residual[t][d] = L[(lane + 32 * t) * RS + wave * CW / 2 + d];
    GroupMemoryBarrierWithGroupSync();
    const uint r = gi * RS;
    float2 sumsq = float2(0, 0);
    for (uint d = 0; d < HC; ++d) { const float2 x = float2(unpack(L[r + d])); sumsq += x * x; }
    const float16_t inv = float16_t(rsqrt(float(float16_t(sumsq.x + sumsq.y))));
    for (uint d = 0; d < HC; ++d) L[r + d] = pack(unpack(L[r + d]) * inv);
    GroupMemoryBarrierWithGroupSync();
    float acc[2][CW];
    [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < CW; n += 2)
    {
        const uint b = weights[(offset(1) + wave * CW + n) / 2];
        acc[t][n] = lo(b); acc[t][n + 1] = hi(b);
    }
    for (uint h0 = 0; h0 < NH; h0 += NW)
    {
        const uint h = h0 + wave, hb = RH + wave * RHS, otab = offset(7 + 5 * h);
        uint kc[2][16], vc[2][16], q[2][16];
        {
            float a[2][32];
            [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < 32; ++n) a[t][n] = 0;
            gemm32(a, 0, RS, HC, offset(6 + 5 * h) / 2, 32);
            [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < 32; n += 2)
            {
                const uint own = pk(a[t][n], a[t][n + 1]), other = WaveReadLaneAt(own, lane ^ 1);
                vc[t][n / 2] = (lane & 1) ? ((other >> 16) | (own & 0xffff0000)) : ((own & 65535) | (other << 16));
            }
        }
        {
            float a[2][32];
            [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < 32; ++n) a[t][n] = 0;
            gemm32(a, 0, RS, HC, offset(4 + 5 * h) / 2, 32);
            [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < 32; n += 2) q[t][n / 2] = pk(a[t][n], a[t][n + 1]);
        }
        {
            float a[2][32];
            [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < 32; ++n) a[t][n] = 0;
            gemm32(a, 0, RS, HC, offset(5 + 5 * h) / 2, 32);
            [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < 32; n += 2) kc[t][n / 2] = pk(a[t][n], a[t][n + 1]);
        }
        float2 sums[2] = {float2(0, 0), float2(0, 0)};
        float o[2][32];
        [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < 32; ++n) o[t][n] = 0;
        uint tn[2][4]; SET_TABLE(tn, 0, 0)
        [loop] for (uint d = 0; d < 4; ++d)
        {
            [unroll] for (uint h2 = 0; h2 < 2; ++h2)
            {
                uint e[2][4]; float s[2][8];
                [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint j = 0; j < 4; ++j)
                { s[t][2 * j] = lo(tn[t][j]); s[t][2 * j + 1] = hi(tn[t][j]); }
                SET_TABLE(tn, h2 == 0 ? d : min(d + 1, 3u), h2 ^ 1)
                [unroll] for (uint kp = 0; kp < 16; ++kp)
                {
#define SC(r_) { const uint k = BC8(kc[h2][kp], r_); s[0][r_] = d2(k, q[0][kp], s[0][r_]); s[1][r_] = d2(k, q[1][kp], s[1][r_]); }
                    SC(0u) SC(1u) SC(2u) SC(3u) SC(4u) SC(5u) SC(6u) SC(7u)
                }
                [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint j = 0; j < 4; ++j)
                {
                    const float16_t2 x = softmax_exp(float16_t2(s[t][2 * j], s[t][2 * j + 1]));
                    sums[t] += float2(x); e[t][j] = pack(x);
                }
#define PV(rp, c) { const uint v = BC8(vc[h2][k], 2u * rp + c); o[0][2u * k + c] = d2(v, e[0][rp], o[0][2u * k + c]); o[1][2u * k + c] = d2(v, e[1][rp], o[1][2u * k + c]); }
                [unroll] for (uint k = 0; k < 16; ++k) { PV(0u, 0u) PV(0u, 1u) }
                [unroll] for (uint k = 0; k < 16; ++k) { PV(1u, 0u) PV(1u, 1u) }
                [unroll] for (uint k = 0; k < 16; ++k) { PV(2u, 0u) PV(2u, 1u) }
                [unroll] for (uint k = 0; k < 16; ++k) { PV(3u, 0u) PV(3u, 1u) }
                [unroll] for (uint k = 0; k < 16; ++k)
                { kc[h2][k] = WaveReadLaneAt(kc[h2][k], (lane + 8) & 31); vc[h2][k] = WaveReadLaneAt(vc[h2][k], (lane + 8) & 31); }
            }
        }
        [unroll] for (uint t = 0; t < 2; ++t)
        {
            const float invsum = float(float16_t(1.0 / float(float16_t(sums[t].x + sums[t].y))));
            [unroll] for (uint n = 0; n < 32; ++n) o[t][n] *= invsum;
        }
        STORE_ROWS(32, o, hb, KR, 0)
        GroupMemoryBarrierWithGroupSync();
        for (uint i = 0; i < NW; ++i) gemmCW(acc, RH + i * RHS, KR, 16, offset(8 + 5 * (h0 + i)) / 2 + wave * CW, C);
        GroupMemoryBarrierWithGroupSync();
    }
    [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < CW; n += 2)
    {
        const float16_t2 y = float16_t2(acc[t][n], acc[t][n + 1]) + unpack(residual[t][n / 2]);
        L[(lane + 32 * t) * RS + (wave * CW + n) / 2] = pack(y);
        const float2 z = float2(y + unpack(weights[(offset(3) + wave * CW + n) / 2]));
        acc[t][n] = z.x; acc[t][n + 1] = z.y;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint g = 0; g < C / 8; g += G)
    {
        const uint ch = g + wave;
        float hh[2][32];
        [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < 32; n += 2)
        { const uint b = weights[(offset(45 + 3 * ch) + n) / 2]; hh[t][n] = lo(b); hh[t][n + 1] = hi(b); }
        gemm32(hh, 0, RS, HC, offset(44 + 3 * ch) / 2, 32);
        [unroll] for (uint t = 0; t < 2; ++t) [unroll] for (uint n = 0; n < 32; n += 2)
            L[RH + (lane + 32 * t) * HRS + (wave * 32 + n) / 2] = pack(gelu(float16_t2(hh[t][n], hh[t][n + 1])));
        GroupMemoryBarrierWithGroupSync();
        for (uint i = 0; i < G; ++i) gemmCW(acc, RH + 16 * i, HRS, 16, offset(46 + 3 * (g + i)) / 2 + wave * CW, C);
        GroupMemoryBarrierWithGroupSync();
    }
    STORE_ROWS(CW, acc, 0, RS, wave * CW / 2)
    GroupMemoryBarrierWithGroupSync();
    for (uint i = gi; i < 64 * HC; i += 32 * NW)
    {
        const uint tok = i / HC, d = i % HC;
        const int y = Y0 + int(tok >> 3), x = X0 + int(tok & 7);
        if (y >= 0 && y < H && x >= 0 && x < W) output[uint(y * W + x) * HC + d] = L[tok * RS + d];
    }
}
