// Product-only fixture for the production M input WMMA implementation.
#include "../dialect_hlsl.h"
#include "../fp8_soft.glsl"
#define D4R_WMMA_SHARED_WORDS (64 * 68)
#include "ags_wmma.hlsli"
Buffer<uint> parameters : register(t0), features : register(t3);
ByteAddressBuffer w8_raw : register(t1), w16_raw : register(t2);
RWBuffer<uint> result : register(u5);
struct Output { RWBuffer<uint> a; };
static Output outb = { result };
struct Params { int2 tok, shift; float expo, expoPrev; };
static Params p;
static uint3 groupState;
static uint waveState;
groupshared uint waveCounter;
#define gl_WorkGroupID groupState
#define gl_SubgroupID waveState
#define gl_SubgroupInvocationID WaveGetLaneIndex()
void gatherPixel(uint pi, int2 block, uint first, float e1, float e2, bool fused)
{
    const uint px = pi & 7, py = pi >> 3;
    const int2 pos = clamp((block >> 1) + int2(px >> 1, py >> 1), int2(0, 0), p.tok - 1);
    const uint pixel = (py & 1) * 2 + (px & 1);
    const uint at = (first + (py >> 1) * 4 + (px >> 1)) * 68 + pixel * 16;
    [unroll] for (uint k = 0; k < 16; ++k)
        matrixShared[at + k] = features[(pos.y * p.tok.x + pos.x) * 64 + pixel * 16 + k];
}
#define main embedMain
#include "enc0_m_wmma.hlsli"
#undef main
[WaveSize(32)] [numthreads(128, 1, 1)]
void main(uint3 group : SV_GroupID, uint index : SV_GroupIndex)
{
    if (index == 0) waveCounter = 0;
    GroupMemoryBarrierWithGroupSync();
    uint wave = 0;
    if (WaveIsFirstLane()) InterlockedAdd(waveCounter, 1, wave);
    waveState = WaveReadLaneFirst(wave);
    groupState = group;
    p.tok = int2(parameters[0], parameters[1]);
    p.shift = int2(parameters[2], parameters[3]);
    p.expo = p.expoPrev = 0;
    embedMain();
}
