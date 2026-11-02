// M texture-translation helpers. Kept separate from K runtime code.
#pragma once
#include "ptx_rt.h"
extern "C" __attribute__((device)) h1 __ocml_tanh_f16(h1);
extern "C" __attribute__((device)) h2 m_min2(h2, h2) __asm("llvm.minimumnum.v2f16");
extern "C" __attribute__((device)) h2 m_max2(h2, h2) __asm("llvm.maximumnum.v2f16");
RT uint32_t prmt32(uint32_t a, uint32_t b, uint32_t select)
{
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t s = (select >> (4 * i)) & 15u;
        uint32_t byte = ((s & 4u) ? b : a) >> (8 * (s & 3u));
        byte = (s & 8u) ? ((byte & 128u) ? 255u : 0u) : (byte & 255u);
        out |= byte << (8 * i);
    }
    return out;
}
RT uint32_t shfl_idx(uint32_t input, uint32_t index, uint32_t opts, bool* in_bounds)
{
    uint32_t mask = (opts >> 8) & 31u;
    uint32_t subsection = laneid() & mask;
    uint32_t end = subsection | (opts & ~mask & 31u);
    uint32_t src = subsection | (index & ~mask & 31u);
    bool valid = src <= end;
    if (in_bounds) *in_bounds = valid;
    return __builtin_amdgcn_ds_bpermute(4u * (warp_base() | (valid ? src : laneid())), input);
}
