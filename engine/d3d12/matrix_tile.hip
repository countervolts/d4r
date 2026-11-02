// Same byte layout and operation as matrix_tile.hlsl. Uses d4r's architecture
// helpers, including the distinct RDNA 3 and RDNA 4 WMMA register layouts.
#include "kernels/common/wmma_layout.h"
#ifndef MATRIX_STEPS
#define MATRIX_STEPS 1
#endif
extern "C" __global__ __launch_bounds__(32) void matrix_tile(
    const unsigned char* input, float* output)
{
    const unsigned tile = blockIdx.x, lane = threadIdx.x, col = lane & 15;
    const wm_f16* a = (const wm_f16*)(input + tile * 2048);
    const wm_f16* b = a + 256;
    const float* c = (const float*)(b + 256);
    wm_op av = wm_op_load(a + col * 16);
    // B is row-major in the common fixture; gather its column into the operand.
    wm_h16 bv = {};
    for (int k = 0; k < 16; ++k) bv[k] = b[k * 16 + col];
#if D4R_WMMA_LAYOUT == 12
    wm_h8 bh;
    for (int k = 0; k < 8; ++k) bh[k] = bv[k + 8 * (lane >> 4)];
    wm_op bo = __builtin_bit_cast(wm_op, bh);
#else
    wm_op bo = __builtin_bit_cast(wm_op, bv);
#endif
    wm_f8v acc;
    for (int i = 0; i < 8; ++i) acc[i] = c[wm_acc_row(i) * 16 + col];
    for (int step = 0; step < MATRIX_STEPS; ++step) acc = wm_mma(av, bo, acc);
    for (int i = 0; i < 8; ++i)
        output[tile * 256 + wm_acc_row(i) * 16 + col] = acc[i];
}
