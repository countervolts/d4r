// Storage ABI shared by M/L software-FP8 network and patch-merge shaders.
// Parameter arrays use a structured SRV so o[72] stays tightly packed.
#ifndef D4R_M_BUFFERS
#define D4R_M_BUFFERS
struct MLayerParams { int W; int Hh; int sx; int sy; uint o[72]; };
#ifdef D4R_M_PARAMETER_WORDS
Buffer<uint> parameterWords : register(t0);
struct MOffsets { Buffer<uint> data; uint operator[](uint i) { return data[4u + i]; } };
struct MParams { int W; int Hh; int sx; int sy; MOffsets o; };
static MParams p = {0, 0, 0, 0, {parameterWords}};
#define D4R_M_LOAD_PARAMS p.W = asint(parameterWords[0]); p.Hh = asint(parameterWords[1]); p.sx = asint(parameterWords[2]); p.sy = asint(parameterWords[3]);
#else
StructuredBuffer<MLayerParams> parameterBuffer : register(t0);
static MLayerParams p;
#define D4R_M_LOAD_PARAMS p = parameterBuffer[0];
#endif
struct MWeightBuffer { Buffer<uint> w; };
struct MTokenBuffer { Buffer<uint> a; };
struct MOutputBuffer { RWBuffer<uint> a; };
Buffer<uint> weightWords : register(t1), vectorWords : register(t2), inputWords : register(t3), skipWords : register(t4);
// DXC loses resource bindings if these wrappers are passed by value.
static MWeightBuffer wb = { weightWords }, w16 = { vectorWords };
static MTokenBuffer inb = { inputWords }, skb = { skipWords };
#ifdef D4R_M_MERGE
RWBuffer<uint> outputWords : register(u4);
static MOutputBuffer outb = { outputWords };
#else
RWBuffer<uint> outputWords : register(u5);
static MOutputBuffer out0 = { outputWords };
#endif
static uint d4rMWave;
groupshared uint mWaveCounter;
void d4rInitWaves()
{
    if (gl_LocalInvocationIndex == 0) mWaveCounter = 0;
    barrier();
    uint index = 0;
    if (WaveIsFirstLane()) InterlockedAdd(mWaveCounter, 1, index);
    d4rMWave = WaveReadLaneFirst(index);
}
#define gl_SubgroupID d4rMWave
#define gl_SubgroupInvocationID WaveGetLaneIndex()
#define D4R_HLSL_INIT_CALL D4R_M_LOAD_PARAMS d4rInitWaves();
float d2(uint a, uint b, float c) { return dot2add(unpackFloat2x16(a), unpackFloat2x16(b), c); }
#endif
