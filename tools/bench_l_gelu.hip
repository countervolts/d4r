// Isolated scheduling experiment, NOT a preset-L evaluation benchmark.
// hipcc --offload-arch=gfx1201 -O3 tools/bench_l_gelu.hip -o /tmp/bench_l_gelu
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../kernels/m/gelu_dual.h"

#define HIP_OK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); exit(1); } } while (0)

template <int Mode>
__device__ __forceinline__ void activation(float (&x)[8])
{
    if constexpr (Mode == 0) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float c = __builtin_amdgcn_fmed3f(x[k], -2.0f, 2.0f);
            x[k] *= __builtin_fmaf(c,
                __builtin_fmaf(-0.0810546875f, __builtin_fabsf(c), 0.412109375f), 0.5f);
        }
    } else {
        swin_gelu_scheduled<Mode == 2>(x);
    }
}

template <int Mode>
__global__ void gelu(float* out, const float* in, int n, int iterations)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n / 8) return;
    float x[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) x[k] = in[8 * t + k];
    for (int j = 0; j < iterations; ++j) {
        activation<Mode>(x);
        // Keep long timing loops out of the subnormal range. No offset in the
        // single-iteration correctness run.
        if (iterations != 1) {
#pragma unroll
            for (int k = 0; k < 8; ++k) x[k] += 0.25f;
        }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) out[8 * t + k] = x[k];
}

int main(int argc, char** argv)
{
    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    if (strncmp(prop.gcnArchName, "gfx12", 5) != 0) {
        fprintf(stderr, "This experiment requires gfx12 wave32.\n"); return 1;
    }
    const int blocks_per_unit = argc > 1 ? atoi(argv[1]) : 32;
    if (blocks_per_unit < 2 || blocks_per_unit > 128) return 1;
    const int n = std::max(65536, 8 * 128 * prop.multiProcessorCount * blocks_per_unit);
    std::vector<float> in(n), reference(n), result(n);
    // Every binary16 encoding plus pseudorandom binary32 bit patterns, including
    // overflow, subnormals, infinities and NaNs. Eight channels remain distinct.
    for (int i = 0; i < n; ++i) {
        if (i < 65536) {
            const _Float16 h = __builtin_bit_cast(_Float16, (unsigned short)i);
            in[i] = (float)h;
        } else {
            unsigned u = (unsigned)i * 747796405u + 2891336453u;
            u = ((u >> ((u >> 28) + 4)) ^ u) * 277803737u;
            u = (u >> 22) ^ u;
            memcpy(&in[i], &u, sizeof(u));
        }
    }
    float *di, *dout;
    HIP_OK(hipMalloc(&di, n * sizeof(float)));
    HIP_OK(hipMalloc(&dout, n * sizeof(float)));
    HIP_OK(hipMemcpy(di, in.data(), n * sizeof(float), hipMemcpyHostToDevice));
    using Kernel = void (*)(float*, const float*, int, int);
    Kernel kernels[] = {gelu<0>, gelu<1>, gelu<2>};
    const char* names[] = {"compiler", "scalar-asm", "dual-asm"};
    for (int mode = 0; mode < 3; ++mode) {
        kernels[mode]<<<n / 8 / 128, 128>>>(dout, di, n, 1);
        HIP_OK(hipGetLastError());
        HIP_OK(hipMemcpy(result.data(), dout, n * sizeof(float), hipMemcpyDeviceToHost));
        if (!mode) reference = result;
        size_t mismatches = 0;
        for (int i = 0; i < n; ++i) {
            if (std::isnan(reference[i]) && std::isnan(result[i])) continue;
            if (memcmp(&reference[i], &result[i], sizeof(float))) {
                if (mismatches < 4) fprintf(stderr, "%s mismatch %d: %a / %a\n",
                    names[mode], i, reference[i], result[i]);
                ++mismatches;
            }
        }
        printf("correctness %s: %zu mismatches / %d (NaN payloads ignored)\n", names[mode], mismatches, n);
        if (mismatches) return 2;
    }
    for (int i = 0; i < n; ++i) in[i] = float(i % 1024) / 128.0f - 4.0f;
    HIP_OK(hipMemcpy(di, in.data(), n * sizeof(float), hipMemcpyHostToDevice));
    hipEvent_t start, end;
    HIP_OK(hipEventCreate(&start)); HIP_OK(hipEventCreate(&end));
    std::vector<float> times[3];
    const int iterations = 4096;
    for (int warmup = 0; warmup < 100; ++warmup)
        for (int mode = 0; mode < 3; ++mode)
            kernels[mode]<<<n / 8 / 128, 128>>>(dout, di, n, iterations);
    HIP_OK(hipDeviceSynchronize());
    for (int trial = 0; trial < 7; ++trial) {
        for (int k = 0; k < 3; ++k) {
            int mode = trial % 2 ? (trial + 2 - k) % 3 : (trial + k) % 3;
            HIP_OK(hipEventRecord(start));
            for (int repeat = 0; repeat < 10; ++repeat)
                kernels[mode]<<<n / 8 / 128, 128>>>(dout, di, n, iterations);
            HIP_OK(hipGetLastError());
            HIP_OK(hipEventRecord(end)); HIP_OK(hipEventSynchronize(end));
            float ms;
            HIP_OK(hipEventElapsedTime(&ms, start, end));
            ms /= 10;
            times[mode].push_back(ms);
            printf("trial %d %-12s %.6f ms\n", trial, names[mode], ms);
        }
    }
    printf("%s n=%d iterations=%d\n", prop.name, n, iterations);
    for (int mode = 0; mode < 3; ++mode) {
        std::sort(times[mode].begin(), times[mode].end());
        printf("median %-12s %.6f ms\n", names[mode], times[mode][times[mode].size() / 2]);
    }
    HIP_OK(hipEventDestroy(start)); HIP_OK(hipEventDestroy(end));
    HIP_OK(hipFree(di)); HIP_OK(hipFree(dout));
}
