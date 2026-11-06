// Preset L's expansion stage for vkhip (engine/vkhip): the native block of dec0l_block.hip as a kernel of its own. Its
// result image goes to a buffer, which a Vulkan stage turns into the reconstruction's inputs (engine/dec0_l_head.comp).
#include "dec0l_block.hip"

struct Dec0LNetParams
{
    uint64_t w;
    int32_t sx, sy, tw, th;
    uint64_t in, skip;
    uint64_t out;             // per block (stride (tw + 9) / 8 blocks) the 4096 bytes of d4r_dec0l_block's result image
};
KERNEL __attribute__((amdgpu_flat_work_group_size(128, 128))) void rrlite_dec0l_net(Dec0LNetParams prm)
{
    const int bx = (int)__builtin_amdgcn_workgroup_id_x(), by = (int)__builtin_amdgcn_workgroup_id_y();
    const int i = (int)(__builtin_amdgcn_workitem_id_x() + 32 * __builtin_amdgcn_workitem_id_z());
    const uint32_t image = d4r_dec0l_block(prm.w, prm.in, prm.skip, prm.sx, prm.sy, prm.tw, prm.th, bx, by);
    const __attribute__((address_space(3))) uint32_t* const src = (const __attribute__((address_space(3))) uint32_t*)(uintptr_t)image;
    uint32_t* const dst = (uint32_t*)prm.out + ((size_t)by * (size_t)((prm.tw + 9) / 8) + (size_t)bx) * 1024;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        dst[i * 8 + j] = src[i * 8 + j];
}
