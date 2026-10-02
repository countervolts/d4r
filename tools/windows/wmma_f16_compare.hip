// Public synthetic reproducer: compare packed F16 WMMA with F32 WMMA followed
// by one FP16 rounding after each K16 step. Never part of game packaging.
#define D4R_K_F16_WMMA
#include "../../kernels/common/wmma_layout.h"

// The replay utility passes a single by-value parameter block, like K/M.
struct CompareParams { const _Float16* a; const _Float16* b; const float* c; float* output; };
extern "C" __attribute__((global)) void d4r_wmma_f16_compare(CompareParams params)
{
    const _Float16* a = params.a;
    const _Float16* b = params.b;
    const float* c = params.c;
    float* output = params.output;
    const unsigned test = __builtin_amdgcn_workgroup_id_x();
    const unsigned token = wm_lane() & 15;
    const unsigned base = test * 256;
    const wm_op aa = wm_op_load(a + base + token * 16);
    const wm_op bb = wm_op_load(b + base + token * 16);
    wm_f8v baseline, adapted;
    wm_h8 packed;
    for (unsigned i = 0; i < 8; ++i) {
        baseline[i] = c[base + wm_acc_row(i) * 16 + token];
        packed[i] = (_Float16)baseline[i];
    }
    adapted = baseline;
    const unsigned steps = 1u << (test / 16);
    for (unsigned step = 0; step < steps; ++step) {
        baseline = wm_mma(aa, bb, baseline);
        for (unsigned i = 0; i < 8; ++i) baseline[i] = (float)(_Float16)baseline[i];
        packed = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(
            __builtin_bit_cast(wm_h8, aa), __builtin_bit_cast(wm_h8, bb), packed);
        adapted = wm_mma_f16_step(aa, bb, adapted);
    }
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned index = wm_acc_row(i) * 16 + token;
        output[test * 768 + index] = baseline[i];
        output[test * 768 + 256 + index] = (float)packed[i];
        output[test * 768 + 512 + index] = adapted[i];
    }
}
