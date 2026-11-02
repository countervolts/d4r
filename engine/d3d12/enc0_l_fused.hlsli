// d4r engine, preset L on vkd3d-proton: the input stage in one dispatch (layer_m.hlsl KIND 3 with L_FUSED, built by
// compile_m.py as enc0_lx). The layer's workgroup gathers its block's 64 pixels' feature slots itself, as the Vulkan
// build's layer_m.comp does, instead of enc0_m.comp L_FEATURES writing them to a buffer that enc0_l reads back (32
// bytes a pixel each way). The registers t0-u13 are enc0_m.comp's (the runtime's Enc0B), the layer's follow at 14.
// Included last, before main: dialect_hlsl.h's GLSL spellings stay out of the layer's own code.
#include "dialect_hlsl.h"
struct Enc0Params      // enc0_m.comp's Params (runtime_m_d3d12.cpp MEnc0Params, 144 bytes)
{
    ivec2 size;             // render size
    ivec2 shift;            // pixels the block grid is shifted by
    ivec2 tok;              // token grid
    vec2 jitter, prevJitter;
    vec2 mvScale, mvOffs;   // motion texel -> pixels of the 3x image
    vec2 rnd2hi, hi2rnd;
    vec2 hiSize;
    float changeScale, changeGain;
    uint depthFlag, reset, frame;
    float expo, expoPrev;
    float halfInvX, halfInvY, renderInvX, renderInvY; // host-rounded CUDA texture coordinates
    // display-resolution motion (MVLowRes clear: NGX's rrlite_enc0_4x4_mvhi_*): a pixel's one vector is the motion
    // texel at floor((pixel + 0.5 - jitter) * mvPix), clamped to [0, mvMax]; it serves all three candidates
    uint displayMv;
    ivec2 mvMax;
    vec2 mvPix;             // output / render
};
ByteAddressBuffer p_buf : register(t0);
static Enc0Params ep;
D4R_SAMPLER(colorTex, t1, s1)
D4R_SAMPLER(motionTex, t2, s2)
D4R_SAMPLER(depthTex, t3, s3)
D4R_SAMPLER(prevMvTex, t4, s4)
D4R_SAMPLER(halfTex, t5, s5)
D4R_SAMPLER(featTex, t6, s6)
RWTexture2D<float4> mvAOut : register(u7);
RWTexture2D<float4> mvBOut : register(u8);
RWTexture2D<float4> mvCOut : register(u9);
ByteAddressBuffer noise_b : register(t12);  // L's dither table, 8 x 16 x 16 binary16 bits
#define ENC0_NOISE(i) ((noise_b.Load(((i) * 2u) & ~3u) >> (16u * ((i) & 1u))) & 0xffffu)
#define EP ep
#define ENC0_DISPLAY_MV 1
#define F8T float16_t
#define F8OF(h) (h)
// enc0_features.glsl through hlsl_build.translate (compile_m.py writes it next to the build)
#include "enc0_features_hlsl.hlsli"
// the block's 64 pixels' 32 FP8 slots, eight words a pixel (enc0_m.comp L_FEATURES' buffer layout)
groupshared uint LF[64 * 8];
void gatherBlockFeatures(uint li, uint2 gid)
{
    ep = p_buf.Load<Enc0Params>(0);
    const int2 block = 8 * int2(gid) - ep.shift;
    const float e1 = ep.expo == 0.0 ? 1.0 : ep.expo, e2 = ep.expoPrev == 0.0 ? 1.0 : ep.expoPrev;
    // enc0_m.comp gathers pixels li and li + 32 in one lane, the first of them fusing its fourth history coordinate
    float16_t fe[32];
    gatherFeatures(li, block, e1, e2, li < 32u, fe);
    [unroll] for (uint k = 0u; k < 4u; ++k)
    {
        uint4 h;
        [unroll] for (uint i = 0u; i < 4u; ++i) h[i] = packFloat2x16(f16vec2(fe[8u * k + 2u * i], fe[8u * k + 2u * i + 1u]));
        const uint2 c = fp8Of(h);
        LF[li * 8u + 2u * k] = c.x; LF[li * 8u + 2u * k + 1u] = c.y;
    }
}
