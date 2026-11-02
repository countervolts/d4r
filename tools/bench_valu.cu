#include <cuda_fp16.h>
#include <cstdio>
#define KG __global__
#define DEVPROP cudaDeviceProp
#define GETPROP cudaGetDeviceProperties
#define MALLOC cudaMalloc
#define EVENT cudaEvent_t
#define ECREATE cudaEventCreate
#define ERECORD(e) cudaEventRecord(e)
#define ESYNC cudaEventSynchronize
#define ELAPSED cudaEventElapsedTime
#include "bench_valu.h"
#include "bench_valu_main.h"
