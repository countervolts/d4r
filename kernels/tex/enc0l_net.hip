// Preset L's input stage for vkhip (engine/vkhip): the native tail of enc0l_tail.hip as a kernel of its own, with the
// features a Vulkan stage gathered (engine/enc0_l_gather.comp) instead of NVIDIA's texture code in front of it.
// d4r-tex-flags: see kernels/build.sh
#define D4R_KERNEL_NAME rrlite_enc0l_net
#include "enc0l_tail.hip"

struct Enc0LNetParams
{
    uint64_t w;
    int32_t sx, sy, tw, th;   // the block grid's offset and the pixel grid
    uint64_t features;        // per block (stride (tw + 9) / 8 blocks) the 2048 bytes the tail reads from shared memory
    uint64_t out, merged;     // the tail's outputs
};
KERNEL __attribute__((amdgpu_flat_work_group_size(128, 128))) void rrlite_enc0l_net(Enc0LNetParams prm)
{
    __shared__ __attribute__((aligned(16))) uint32_t stage[512];
    const int bx = (int)__builtin_amdgcn_workgroup_id_x(), by = (int)__builtin_amdgcn_workgroup_id_y();
    const int i = (int)(__builtin_amdgcn_workitem_id_x() + 32 * __builtin_amdgcn_workitem_id_z());
    const uint32_t* const src = (const uint32_t*)prm.features + ((size_t)by * (size_t)((prm.tw + 9) / 8) + (size_t)bx) * 512 + i * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        stage[i * 4 + j] = src[j];
    d4r_enc0l_tail((uint32_t)(uintptr_t)(__attribute__((address_space(3))) void*)stage, prm.w, prm.out, prm.merged, prm.sx, prm.sy,
                   prm.tw, prm.th, bx, by);
}
