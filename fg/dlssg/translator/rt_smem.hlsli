// Aliased shared-memory views for glsl2hlsl.py kernels: GLSL declares smem (uint), smv (uvec4), smh (float16_t) and
// smf (float) as shared memory blocks over the same storage; HLSL has one groupshared uint smem[] and these functions.
uint4 nvx_smv_ld(uint i) { return uint4(smem[4u * i], smem[4u * i + 1u], smem[4u * i + 2u], smem[4u * i + 3u]); }
void nvx_smv_st(uint i, uint4 v)
{
    smem[4u * i] = v.x;
    smem[4u * i + 1u] = v.y;
    smem[4u * i + 2u] = v.z;
    smem[4u * i + 3u] = v.w;
}
float16_t nvx_smh_ld(uint i) { return asfloat16(uint16_t((smem[i >> 1] >> ((i & 1u) * 16u)) & 0xffffu)); }
// a 16-bit store touches only its half of the word (the other half may be stored by another lane at the same time)
void nvx_smh_st(uint i, float16_t v)
{
    uint sh = (i & 1u) * 16u, o;
    InterlockedAnd(smem[i >> 1], ~(0xffffu << sh), o);
    InterlockedOr(smem[i >> 1], uint(asuint16(v)) << sh, o);
}
float nvx_smf_ld(uint i) { return asfloat(smem[i]); }
void nvx_smf_st(uint i, float v) { smem[i] = asuint(v); }
