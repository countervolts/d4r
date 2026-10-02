#pragma once
#include <stdint.h>

// Windows gfx11 portability workaround informed by realdody/d4r 1897b4a.
// The reported permlanex16 failure duplicates the even K value into the odd
// slot. Retain the native gfx11 layout and exchange through per-wave LDS.
// Other platforms and gfx12 keep their original instruction sequence.
__attribute__((device, always_inline)) static inline uint32_t d4r_wave32_other_half(uint32_t value)
{
#if defined(D4R_WINDOWS_GFX11_LANE_EXCHANGE) && defined(__GFX11__)
    __attribute__((shared)) volatile uint32_t scratch[512];
    const uint32_t tid = __builtin_amdgcn_workitem_id_x() + __builtin_amdgcn_workgroup_size_x() *
        (__builtin_amdgcn_workitem_id_y() + __builtin_amdgcn_workgroup_size_y() * __builtin_amdgcn_workitem_id_z());
    // All native K/M launch bounds are <=512. Do not silently corrupt LDS if
    // this helper is later reused by a larger workgroup.
    if (tid >= 512) __builtin_trap();
    scratch[tid] = value;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    const uint32_t result = scratch[tid ^ 16u];
    // Protect scratch reuse by the next exchange in this wave. No block-wide
    // rendezvous: different waves can reach GEMMs in a different order.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    return result;
#else
    return __builtin_amdgcn_permlanex16(value, value, 0x76543210u, 0xfedcba98u, false, false);
#endif
}
