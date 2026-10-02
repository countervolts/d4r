// Independent gfx12 WMMA layout probe. This deliberately does not use d4r fragments.
// A and B are row-major [logical row/column][K]; the intrinsic receives B,A
// so its output is row=A, column=B, as in AMD's RDNA 4 WMMA guide.
using h8 = _Float16 __attribute__((ext_vector_type(8)));
using f8 = float __attribute__((ext_vector_type(8)));
#include "../../kernels/common/wmma_backend.h"
#include "../../kernels/common/wmma_layout.h"

extern "C" __attribute__((global)) void d4r_wmma_swap_test(float* output)
{
    const unsigned int lane = __builtin_amdgcn_workitem_id_x();
    if (lane < 32u) {
        output[lane] = d4r_wmma_other_half(static_cast<float>(lane));
        output[32u + lane] = static_cast<float>(__builtin_amdgcn_mbcnt_lo(~0u, 0u));
    }
}

extern "C" __attribute__((global)) void d4r_wmma_gfx1201(
    const unsigned short* a_bits, const unsigned short* b_bits,
    const float* initial_c, float* output, unsigned int steps)
{
    const unsigned int lane = __builtin_amdgcn_workitem_id_x();
    if (lane >= 32u) return;
    const unsigned int row = lane & 15u;
    const unsigned int group = lane >> 4;
    const _Float16* a = reinterpret_cast<const _Float16*>(a_bits);
    const _Float16* b = reinterpret_cast<const _Float16*>(b_bits);
    h8 a_frag, b_frag;
    f8 c_frag;
    for (unsigned int i = 0; i < 8u; ++i) {
        a_frag[i] = a[row * 16u + group * 8u + i];
        b_frag[i] = b[row * 16u + group * 8u + i];
        c_frag[i] = initial_c[row * 16u + group * 8u + i];
    }
    f8 result = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        b_frag, a_frag, c_frag);
    if (steps == 2u)
        result = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
            b_frag, a_frag, result);
    for (unsigned int i = 0; i < 8u; ++i)
        output[row * 16u + group * 8u + i] = result[i];
}

// d4r's original gfx11 fragment contract, exercised independently of K/M.
extern "C" __attribute__((global)) void d4r_wmma_legacy_contract(
    const unsigned short* a_bits, const unsigned short* b_bits,
    const float* initial_c, float* output, unsigned int steps)
{
    const unsigned int lane = __builtin_amdgcn_workitem_id_x();
    if (lane >= 32u) return;
    const unsigned int token = lane & 15u;
    const unsigned int group = lane >> 4;
    const _Float16* a = reinterpret_cast<const _Float16*>(a_bits);
    const _Float16* b = reinterpret_cast<const _Float16*>(b_bits);
    d4r_wmma_h16 a_frag, b_frag;
    d4r_wmma_f8 c_frag;
    for (unsigned int i = 0; i < 16u; ++i) {
        a_frag[i] = a[token * 16u + i];
        b_frag[i] = b[token * 16u + i];
    }
    for (unsigned int i = 0; i < 8u; ++i)
        c_frag[i] = initial_c[(2u * i + group) * 16u + token];
    d4r_wmma_f8 result = d4r_wmma_legacy_layout(a_frag, b_frag, c_frag);
    if (steps == 2u)
        result = d4r_wmma_legacy_layout(a_frag, b_frag, result);
    for (unsigned int i = 0; i < 8u; ++i)
        output[(2u * i + group) * 16u + token] = result[i];
}

// Upstream's native per-target layout, including accumulator -> next-operand
// packing. It must match the same scalar reference as the independent probe.
extern "C" __attribute__((global)) void d4r_wmma_upstream_contract(
    const unsigned short* a_bits, const unsigned short* b_bits,
    const float* initial_c, float* output, unsigned int steps)
{
    const unsigned int lane = __builtin_amdgcn_workitem_id_x();
    if (lane >= 32u) return;
    const unsigned int token = lane & 15u;
    const _Float16* a = reinterpret_cast<const _Float16*>(a_bits);
    const _Float16* b = reinterpret_cast<const _Float16*>(b_bits);
    const wm_op a_frag = wm_op_load(a + token * 16u);
    const wm_op b_frag = wm_op_load(b + token * 16u);
    wm_f8v c_frag;
    for (unsigned int i = 0; i < 8u; ++i)
        c_frag[i] = initial_c[wm_acc_row(i) * 16u + token];
    wm_f8v result = wm_mma(a_frag, b_frag, c_frag);
    if (steps == 2u)
        result = wm_mma(a_frag, b_frag, result);
    // Fixture values are exactly representable as FP16. Checking this roundtrip
    // catches a wrong gfx12 fragment reconstruction without a tolerance mask.
    const wm_op next_operand = wm_op_from_f8(result);
    for (unsigned int i = 0; i < 8u; ++i)
        output[wm_acc_row(i) * 16u + token] = static_cast<float>(wm_op_acc_elem(next_operand, i));
}
