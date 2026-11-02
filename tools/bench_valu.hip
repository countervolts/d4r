#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#define KG __global__
#define DEVPROP hipDeviceProp_t
#define GETPROP hipGetDeviceProperties
#define MALLOC hipMalloc
#define EVENT hipEvent_t
#define ECREATE hipEventCreate
#define ERECORD(e) hipEventRecord(e)
#define ESYNC hipEventSynchronize
#define ELAPSED hipEventElapsedTime
#include "bench_valu.h"
#include "bench_valu_main.h"
