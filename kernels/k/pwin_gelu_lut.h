// PWIN_GELU_LUT: exact f16 GELU by exhaustive lookup, populated on the prep stream.
// Include after gelu8_math. The POS prep wrappers reserve at least 65536 work-items.
// Every index comes from a uint16_t f16 bit pattern, so it is inside this allocation.
#pragma once

__device__ __attribute__((aligned(256))) uint16_t g_gelu_lut[65536];

constexpr int pwin_gelu_prep_items(int n) { return n < 65536 ? 65536 : n; }

__device__ __forceinline__ void pwin_gelu_prep()
{
    const unsigned i = blockIdx.x * 128u + threadIdx.x;
    if (i < 65536u)
    {
        const half_t x = __builtin_bit_cast(half_t, (uint16_t)i);
        half_t g[8];
        gelu8_math(splat8((float)x), g);
        g_gelu_lut[i] = __builtin_bit_cast(uint16_t, g[0]);
    }
}

__device__ __forceinline__ void gelu8_lut(const f8v& d, half_t g[8])
{
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const uint16_t bits = __builtin_bit_cast(uint16_t, (half_t)d[i]);
#ifdef PWIN_GELU_LUT_BUFFER
        const __amdgpu_buffer_rsrc_t r = __builtin_amdgcn_make_buffer_rsrc(
            (void*)g_gelu_lut, (short)0, sizeof(g_gelu_lut), 0x31004000);
        const uint16_t value = __builtin_amdgcn_raw_buffer_load_b16(r, (unsigned)bits * 2u, 0u, 0);
#else
        const uint16_t value = g_gelu_lut[bits];
#endif
        g[i] = __builtin_bit_cast(half_t, value);
    }
}
