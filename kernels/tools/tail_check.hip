// Standalone check of the texture-kernel GEMM tails (kernels/tex/enc0_tail.hip, dec0_head.hip) outside the
// ZLUDA-compiled kernels they are linked into: random e4m3 weights and activations, several tile positions
// (including partial tiles at the image border), all outputs written to one file. Builds of different WMMA
// layouts (gfx11, the gfx12-layout shim, gfx12, gfx12 native FP8) are then compared byte for byte (same
// arithmetic) or by value (gfx12 rounding). The tails touch no images, so this also runs in rocjitsu.
//
// build: clang++ -x hip --offload-arch=ARCH -O3 [-DD4R_WMMA_LAYOUT=12] [-DD4R_TEX_FP8] -I kernels/tex
//        kernels/tools/tail_check.hip -o tail_check           usage: tail_check OUT.bin
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// the surface-store helpers of tex_common.h reference ZLUDA's helper library, which the tails do not use
extern "C" __device__ void __zluda_ptx_impl_surfobj_b_2d_v2_b16_zero(uint64_t, int __attribute__((ext_vector_type(2))),
                                                                     short __attribute__((ext_vector_type(2))))
{
}

// both tails define the same hook globals
#define D4R_KERNEL_NAME tail_enc0
#define d4r_prep_blocks d4r_prep_blocks_enc0
#define d4r_prep_key_at d4r_prep_key_at_enc0
#include "enc0_tail.hip"
#undef d4r_prep_blocks
#undef d4r_prep_key_at
#define d4r_prep_blocks d4r_prep_blocks_dec0
#define d4r_prep_key_at d4r_prep_key_at_dec0
#include "dec0_head.hip"

extern "C" __global__ void __launch_bounds__(32) run_enc0(const uint32_t* lds_in, const uint8_t* w, uint32_t* out,
                                                          uint32_t dimx, uint32_t dimy, int ox, int oy)
{
    __shared__ uint32_t s[512 + 4];
    for (int i = threadIdx.x; i < 512; i += 32)
        s[i] = lds_in[i];
    __syncthreads();
    d4r_enc0_tail((uint32_t)(uintptr_t)(__attribute__((address_space(3))) void*)s, (uint64_t)w, (uint64_t)out, dimx,
                  dimy, ox, oy);
}

extern "C" __global__ void __launch_bounds__(32) run_dec0(const uint8_t* w, const uint32_t* in, uint32_t* staged,
                                                          uint32_t dimx, uint32_t dimy, int ox, int oy)
{
    __shared__ uint32_t s[512 + 4];
    for (int i = threadIdx.x; i < 512; i += 32)
        s[i] = 0xdeadbeefu;
    __syncthreads();
    d4r_dec0_head((uint32_t)(uintptr_t)(__attribute__((address_space(3))) void*)s, (uint64_t)w, (uint64_t)in, dimx,
                  dimy, ox, oy);
    __syncthreads();
    for (int i = threadIdx.x; i < 512; i += 32)
        staged[i] = s[i];
}

#define CHECK(x)                                                                                                   \
    do                                                                                                             \
    {                                                                                                              \
        hipError_t e_ = (x);                                                                                       \
        if (e_ != hipSuccess)                                                                                      \
        {                                                                                                          \
            fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_));                                                \
            return 1;                                                                                              \
        }                                                                                                          \
    } while (0)

static uint8_t e4m3(uint32_t r)
{
    // Keep the matrix values in a useful range so saturation cannot hide a wrong WMMA result.
    const uint8_t magnitude = (r & 31) == 0 ? 0 : (uint8_t)(0x18 + (r & 31));
    return (uint8_t)(magnitude | ((r >> 8) & 0x80));
}
static uint16_t f16_bits(float v)
{
    return __builtin_bit_cast(uint16_t, (_Float16)v);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: %s OUT.bin\n", argv[0]);
        return 2;
    }
    srand(12345);
    auto rnd = []() { return (uint32_t)rand() * 2654435761u + (uint32_t)rand(); };
    // weights: e4m3 fragments, f16 biases (enc0 at 12288, dec0 at 8192)
    std::vector<uint8_t> w(16384 + 512);
    for (auto& b : w)
        b = e4m3(rnd());
    for (int i = 0; i < 64; ++i)
        *(uint16_t*)&w[12288 + 2 * i] = f16_bits((float)(int)(rnd() % 2001 - 1000) / 256.0f);
    for (int i = 0; i < 128; ++i)
        *(uint16_t*)&w[8192 + 2 * i] = f16_bits((float)(int)(rnd() % 2001 - 1000) / 256.0f);
    const uint32_t dimx = 20, dimy = 14, W2 = dimx / 2, H2 = dimy / 2, rowstride = W2 * 8, plane = H2 * rowstride;
    std::vector<uint32_t> lds(512), in(2 * plane);
    for (auto& v : lds)
        v = (uint32_t)e4m3(rnd()) | (uint32_t)e4m3(rnd()) << 8 | (uint32_t)e4m3(rnd()) << 16 | (uint32_t)e4m3(rnd()) << 24;
    for (auto& v : in)
        v = (uint32_t)e4m3(rnd()) | (uint32_t)e4m3(rnd()) << 8 | (uint32_t)e4m3(rnd()) << 16 | (uint32_t)e4m3(rnd()) << 24;

    uint8_t *dw, *dummy;
    uint32_t *dlds, *din, *dout, *dstaged;
    CHECK(hipMalloc(&dw, w.size()));
    CHECK(hipMalloc(&dlds, lds.size() * 4));
    CHECK(hipMalloc(&din, in.size() * 4));
    CHECK(hipMalloc(&dout, 2 * plane * 4));
    CHECK(hipMalloc(&dstaged, 512 * 4));
    CHECK(hipMalloc(&dummy, 512));
    CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dlds, lds.data(), lds.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(din, in.data(), in.size() * 4, hipMemcpyHostToDevice));
    // the prep kernels read the weight pointer at parameter offset 0
    Enc0Params pe{};
    pe.w = (uint64_t)dw;
    Dec0Params pd{};
    pd.w = (uint64_t)dw;
    hipLaunchKernelGGL(tail_enc0_prep, dim3(4), dim3(128), 0, 0, pe);
    hipLaunchKernelGGL(rrlite_dec0_4x4_folded_prep, dim3(4), dim3(128), 0, 0, pd);
    CHECK(hipDeviceSynchronize());

    FILE* f = fopen(argv[1], "wb");
    const int offsets[][2] = {{0, 0}, {8, 4}, {12, 6}, {16, 8}, {-4, -2}, {18, 12}};
    std::vector<uint32_t> host(2 * plane);
    for (const auto& o : offsets)
    {
        CHECK(hipMemset(dout, 0x5a, 2 * plane * 4));
        hipLaunchKernelGGL(run_enc0, dim3(1), dim3(32), 0, 0, dlds, dw, dout, dimx, dimy, o[0], o[1]);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(host.data(), dout, 2 * plane * 4, hipMemcpyDeviceToHost));
        fwrite(host.data(), 4, 2 * plane, f);
        hipLaunchKernelGGL(run_dec0, dim3(1), dim3(32), 0, 0, dw, din, dstaged, dimx, dimy, o[0], o[1]);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(host.data(), dstaged, 512 * 4, hipMemcpyDeviceToHost));
        fwrite(host.data(), 4, 512, f);
    }
    fclose(f);
    printf("wrote %s\n", argv[1]);
    return 0;
}
