// The token-lane Swin block (../m/swin_tok.h) for the native parts of ZLUDA-compiled texture kernels: preset L's
// enc0 (enc0l_tail.hip) and dec0 (dec0l_block.hip).
#pragma once

// The Swin sources are written against the HIP headers; a texture tail is built without them. These are the
// runtime names they use.
#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_H
#define D4R_NO_HIP_RUNTIME
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __forceinline__ inline __attribute__((always_inline))
#define __shared__ __attribute__((shared))
typedef __SIZE_TYPE__ size_t;
struct d4r_tid_t
{
    struct X { __attribute__((device, always_inline)) operator unsigned() const { return __builtin_amdgcn_workitem_id_x(); } } x;
    struct Z { __attribute__((device, always_inline)) operator unsigned() const { return __builtin_amdgcn_workitem_id_z(); } } z;
};
static const d4r_tid_t threadIdx;
__attribute__((device, always_inline)) static inline void __syncthreads()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
#endif

// 1.0 for the k32 rounding of the D4R_SWIN_K32_MIX variant (see mix2 in ../m/swin_common.h). The prep kernel stores
// it, so that it is not a constant to the compiler.
__attribute__((device, used)) float d4r_swin_one = 1.0f;
#define MIX_ONE d4r_swin_one

namespace d4r_swin
{
#define SWIN_EXACT
// The original code is ZLUDA's compile of the PTX; these make the f32 arithmetic the same instruction for
// instruction where it is not exact (see ../m/swin_tok.h, ../m/swin_common.h): the K order and the C operand of
// the FP8 WMMAs, the key order of the f16 P V product, and the hardware reciprocals.
#define SWIN_TOK_NATURAL
#ifndef D4R_SWIN_K32_MIX
#define K32_NO_MIX
#endif
#define SWIN_PV_ZLUDA_ORDER
#define SWIN_APPROX_RECIP
#ifndef D4R_FP8_WMMA
#define D4R_FP8_WMMA
#endif
#include "../m/swin_common.h"
#include "../m/swin_tok.h"
} // namespace d4r_swin
