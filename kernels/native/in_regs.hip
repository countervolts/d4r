// hiluma input kernel rebuilt from in_unit.h (kernels/native/in_fuse.py on in_ifs.hip): the translated kernel's
// per-thread LDS scratch and 32-byte local array live in registers (the local array compiled to scratch-memory
// flat stores/loads; -21 % in the replay bench). Same launch (16x16 blocks), same output.
// Build: kernels/build.sh (generates in_unit.h from the extracted PTX); by hand: WAVE=32 build_native.sh in_regs.hip OUT.hsaco
#include "in_unit.h"
struct __attribute__((aligned(8))) hiluma_engine_input_depthinv_mvhi_hdr_v2_rel_params { uint8_t b[248]; };
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1, 256))) void hiluma_engine_input_depthinv_mvhi_hdr_v2_rel(hiluma_engine_input_depthinv_mvhi_hdr_v2_rel_params)
{
    const AS4 uint8_t* KARGP = (const AS4 uint8_t*)__builtin_amdgcn_kernarg_segment_ptr();
    const uint32_t gx = __builtin_amdgcn_workgroup_id_x() * 16u + __builtin_amdgcn_workitem_id_x();
    const uint32_t gy = __builtin_amdgcn_workgroup_id_y() * 16u + __builtin_amdgcn_workitem_id_y();
    uint16_t o[4];
    in_unit(KARGP, gx, gy, o);
    // the original's store: network pixel (gx >> 2, gy), channels 4 (gx & 3) .. + 3; row stride W (u32 at +64),
    // buffer at +240 (16 halves per pixel)
    const uint32_t W = *(const AS4 uint32_t*)(KARGP + 64);
    uint16_t* d = (uint16_t*)*(const AS4 uint64_t*)(KARGP + 240) + ((size_t)(W * gy + (gx >> 2)) * 16 + 4 * (gx & 3));
    d[0] = o[0];
    d[1] = o[1];
    d[2] = o[2];
    d[3] = o[3];
}
