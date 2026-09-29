#pragma once

// Device-only Windows HIP build shim. The diagnostic/kernel code objects are
// compiled with -nogpuinc/-nogpulib; no host HIP or C++ standard library is
// needed. Definitions below mirror public HIP qualifiers and LLVM AMDGPU
// work-item/grid builtins. The normal Linux HIP include path is untouched.
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __forceinline__ __attribute__((always_inline)) inline
#define __launch_bounds__(n) __attribute__((amdgpu_flat_work_group_size(1, n)))

struct d4r_device_dim3 { unsigned x, y, z; };
__device__ __forceinline__ d4r_device_dim3 d4r_thread_idx() {
    return {__builtin_amdgcn_workitem_id_x(), __builtin_amdgcn_workitem_id_y(),
        __builtin_amdgcn_workitem_id_z()};
}
__device__ __forceinline__ d4r_device_dim3 d4r_block_idx() {
    return {__builtin_amdgcn_workgroup_id_x(), __builtin_amdgcn_workgroup_id_y(),
        __builtin_amdgcn_workgroup_id_z()};
}
__device__ __forceinline__ d4r_device_dim3 d4r_grid_dim() {
    return {__builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x(),
        __builtin_amdgcn_grid_size_y() / __builtin_amdgcn_workgroup_size_y(),
        __builtin_amdgcn_grid_size_z() / __builtin_amdgcn_workgroup_size_z()};
}
#define threadIdx d4r_thread_idx()
#define blockIdx d4r_block_idx()
#define gridDim d4r_grid_dim()
__device__ __attribute__((convergent, always_inline)) inline void __syncthreads() {
    __builtin_amdgcn_s_barrier();
}
