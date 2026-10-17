// Stress the bounded FP32 path against the original sequential FP64 dot.
#define SWIN_BOUNDED_PV
#include "../../kernels/m/swin_common.h"
struct PvParams { uint32_t* output; };
__device__ uint32_t pv_random(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
extern "C" __global__ void exact_pv_probe(PvParams p) {
    const unsigned index = blockIdx.x * 32 + threadIdx.x;
    uint32_t state = index + 1;
    half_t a[16], b[16];
    for (int k = 0; k < 16; ++k) {
        uint16_t x = pv_random(state), y = pv_random(state);
        x = (x & 0x8000u) | ((x & 0x7fffu) % 0x7c00u);
        y = (y & 0x8000u) | ((y & 0x7fffu) % 0x7c00u);
        a[k] = __builtin_bit_cast(half_t, x);
        b[k] = __builtin_bit_cast(half_t, y);
    }
    switch (index % 4) {
        case 0: break; // Entire finite FP16 exponent range, signed products.
        case 1: // Cancellation, including products too small for an FP32 sum.
            for (int k = 0; k < 8; ++k) { a[k + 8] = -a[k]; b[k + 8] = b[k]; }
            b[15] = (half_t)0.0f;
            break;
        case 2: // Every finite magnitude with exact midpoint perturbations.
        case 3:
            for (int k = 0; k < 16; ++k) { a[k] = (half_t)0.0f; b[k] = (half_t)1.0f; }
            a[0] = __builtin_bit_cast(half_t, (uint16_t)((index / 4) % 0x7c00u));
            {
                const uint16_t bits = __builtin_bit_cast(uint16_t, a[0]);
                const half_t next = __builtin_bit_cast(half_t, (uint16_t)(bits + (bits < 0x7bffu)));
                a[1] = (half_t)((float)next - (float)a[0]);
            }
            b[1] = (half_t)0.5f;
            a[2] = __builtin_bit_cast(half_t, (uint16_t)1);
            b[2] = __builtin_bit_cast(half_t, (uint16_t)((index & 1) ? 0x8001u : 1u));
            break;
    }
    double reference = 0.0;
    for (int k = 0; k < 16; ++k) reference += (double)a[k] * (double)b[k];
    const uint16_t expected = __builtin_bit_cast(uint16_t, (half_t)reference);
    const uint16_t actual = __builtin_bit_cast(uint16_t, exact_pv_dot<1>(a, b));
    p.output[index] = uint32_t(expected) | (uint32_t(actual) << 16);
}
