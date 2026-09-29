#pragma once
#include <stdint.h>

// d4r's gfx11 interface: each lane supplies 16 FP16 K operands. The 8 FP32
// outputs have logical row 2*i + lane/16 and logical column lane%16.
// gfx12 consumes only 8 FP16 operands per lane and emits 8 consecutive rows
// per lane group. Keep the old interface for K/M/texture call sites until their
// surrounding fragments are ported and validated individually.
using d4r_wmma_h16 = _Float16 __attribute__((ext_vector_type(16)));
using d4r_wmma_h8 = _Float16 __attribute__((ext_vector_type(8)));
using d4r_wmma_f8 = float __attribute__((ext_vector_type(8)));

__attribute__((device, always_inline)) static inline float d4r_wmma_other_half(float value)
{
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    const uint32_t swapped = __builtin_amdgcn_permlanex16(
        bits, bits, 0x76543210u, 0xfedcba98u, false, false);
    return __builtin_bit_cast(float, swapped);
}

__attribute__((device, always_inline)) static inline d4r_wmma_f8
d4r_wmma_legacy_layout(d4r_wmma_h16 a, d4r_wmma_h16 b, d4r_wmma_f8 c)
{
#if defined(__gfx1200__) || defined(__gfx1201__)
    const uint32_t group = __builtin_amdgcn_mbcnt_lo(~0u, 0u) >> 4;
    d4r_wmma_h8 a12, b12;
    d4r_wmma_f8 c12;
    d4r_wmma_f8 remote_c;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        remote_c[i] = d4r_wmma_other_half(c[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a12[i] = a[8u * group + i];
        b12[i] = b[8u * group + i];
        const uint32_t row = 8u * group + i;
        const uint32_t old_index = row >> 1;
        c12[i] = (row & 1u) == group ? c[old_index] : remote_c[old_index];
    }
    const d4r_wmma_f8 d12 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a12, b12, c12);
    d4r_wmma_f8 result, remote_d;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        remote_d[i] = d4r_wmma_other_half(d12[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t row = 2u * i + group;
        const uint32_t new_index = row & 7u;
        result[i] = (row >> 3) == group ? d12[new_index] : remote_d[new_index];
    }
    return result;
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
#error "d4r WMMA backend requires a validated gfx11 or gfx12 target"
#endif
}
