#pragma once
#include "diagnostic.h"
// CUDA Driver API ABI subset. No CUDA SDK, CUDA runtime or NVIDIA binary is needed.
using CUdevice = int;
using CUdeviceptr = uint64_t;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
struct CudaApi {
    d4r::diag::Library library;
#define D4R_CU(name, ret, ...) using name##_fn = ret(WINAPI*)(__VA_ARGS__); name##_fn name = library.symbol<name##_fn>(#name)
    D4R_CU(cuInit, int, unsigned);
    D4R_CU(cuDriverGetVersion, int, int*);
    D4R_CU(cuDeviceGetCount, int, int*);
    D4R_CU(cuDeviceGet, int, CUdevice*, int);
    D4R_CU(cuDeviceGetName, int, char*, int, CUdevice);
    D4R_CU(cuDeviceGetAttribute, int, int*, int, CUdevice);
    D4R_CU(cuDevicePrimaryCtxRetain, int, CUcontext*, CUdevice);
    D4R_CU(cuDevicePrimaryCtxRelease_v2, int, CUdevice);
    D4R_CU(cuCtxSetCurrent, int, CUcontext);
    D4R_CU(cuCtxCreate_v2, int, CUcontext*, unsigned, CUdevice);
    D4R_CU(cuCtxDestroy_v2, int, CUcontext);
    D4R_CU(cuCtxSynchronize, int);
    D4R_CU(cuMemAlloc_v2, int, CUdeviceptr*, size_t);
    D4R_CU(cuMemFree_v2, int, CUdeviceptr);
    D4R_CU(cuMemcpyHtoD_v2, int, CUdeviceptr, const void*, size_t);
    D4R_CU(cuMemcpyDtoH_v2, int, void*, CUdeviceptr, size_t);
    D4R_CU(cuModuleLoadData, int, CUmodule*, const void*);
    D4R_CU(cuModuleUnload, int, CUmodule);
    D4R_CU(cuModuleGetFunction, int, CUfunction*, CUmodule, const char*);
    D4R_CU(cuLaunchKernel, int, CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
        unsigned, void*, void**, void**);
    D4R_CU(cuGetErrorName, int, int, const char**);
    D4R_CU(cuGetErrorString, int, int, const char**);
#undef D4R_CU
    explicit CudaApi(const std::string& path) : library(d4r::diag::wide(path)) {}
    void check(int result, const char* call) {
        const char* name = "unknown", *message = "unknown";
        if (result) { cuGetErrorName(result, &name); cuGetErrorString(result, &message); }
        else name = "CUDA_SUCCESS";
        std::printf("CUDA %s -> %d (%s)\n", call, result, name ? name : "unknown");
        if (result) throw std::runtime_error(std::string(call) + ": " + (message ? message : "unknown"));
    }
};
