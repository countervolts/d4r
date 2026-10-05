// Layout probe: which k does each byte of the A (weight) and B (activation) operand hold?
//
// Method, with no assumption about byte order:
//   * A (the prepared weight slot) row r holds a distinct e4m3 value per byte of its k16 row.
//   * B (the staged activation) is an identity selector: column c is 1.0 at k = c, 0 elsewhere.
//   * The WMMA computes D[r][c] = sum_k A[r][k] * B[k][c] = A[r][k(c)], so the value that
//     comes back names the k that column c's byte was paired with.  Both operands are filled
//     byte-by-byte in memory order, so the printout is a direct read of the hardware's slot
//     order for each side.
//
// Build and run:
//   hipcc --offload-arch=gfx1201 -DD4R_TEX_FP8 -DD4R_WMMA_LAYOUT=12 -O2 \
//         -o /tmp/probe kernels/tools/rrswin_operand_probe.cpp && /tmp/probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "../rr/rrswin_engine.h"

#define CHECK(expr)                                                                              \
    do                                                                                           \
    {                                                                                            \
        hipError_t e_ = (expr);                                                                  \
        if (e_ != hipSuccess)                                                                    \
        {                                                                                        \
            std::printf("HIP error at %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,              \
                        hipGetErrorString(e_));                                                  \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

static const int kRows = 16, kChunk = 16, kRowBytes = 32;

// 16 distinct, exactly representable e4m3 values indexed by k
static uint8_t value_for(int k)
{
    const int e = 1 + (k & 7);      // 1..8
    const int m = (k >> 3) & 1;     // 0 or 1
    return (uint8_t)((e << 3) | (m ? 4 : 0));
}

static double dec(uint8_t c)
{
    const int sign = c & 0x80, e = (c >> 3) & 0xf, m = c & 7;
    const double mag = e == 0 ? (m / 8.0) * std::pow(2.0, -6.0) : (1.0 + m / 8.0) * std::pow(2.0, e - 7.0);
    return sign ? -mag : mag;
}

__global__ void probe_kernel(const uint8_t* x_img, const uint8_t* w_prep, float* out)
{
    __shared__ uint8_t img[kRows * kRowBytes];
    for (int i = threadIdx.x; i < kRows * kRowBytes; i += blockDim.x)
        img[i] = x_img[i];
    __syncthreads();
    f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = rrswin::swin_gemm_tile(c, (const kslot*)w_prep, 0, img, kRowBytes, 0);
    for (int i = 0; i < 8; ++i)
        out[i * 32 + threadIdx.x] = c[i];
}

int main()
{
    // A: weight row r, byte b of chunk 0 = value_for(b); the other chunk zero.
    // B: activation column c = 1.0 at byte c of chunk 0, zero elsewhere.
    const uint8_t one = 0x38;   // e4m3 1.0
    std::vector<uint8_t> w(32 * kChunk, 0), x(kRows * kRowBytes, 0);
    // A row r carries only 2^(r-8) at byte 0, so the value read back names the row the
    // engine actually selected; B column c is 1.0 at k = c.
    // A row r = 1.0 at byte 0 only; B column c holds a distinct value per byte of its k16 row.
    for (int r = 0; r < kRows; ++r)
        w[(size_t)(r) * kChunk + 0] = one;
    // B row r must be distinguishable by its byte 0 as well, or "which row" is invisible:
    // byte b of row r carries value_for((b + r) % 16).
    for (int c = 0; c < kRows; ++c)
        for (int b = 0; b < 16; ++b)
            x[(size_t)c * kRowBytes + b] = value_for((b + c) % 16);

    uint8_t *dx, *dw;
    float* dout;
    CHECK(hipMalloc(&dx, x.size()));
    CHECK(hipMalloc(&dw, w.size()));
    CHECK(hipMalloc(&dout, 8 * 32 * sizeof(float)));
    CHECK(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    probe_kernel<<<1, 32>>>(dx, dw, dout);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<float> got(8 * 32);
    CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));

    // The engine's assumed indexing: value at D[row = i + 8h][col = lane & 15].
    // A is the identity at byte 0, so D(row, col) = B[k0][col] = value_for((k0 + row(col)) % 16).
    // Print the value_for index of every (vgpr, lane) so the true (row, column) assignment can
    // be read off the table instead of assumed.
    std::printf("value_for index at got[vgpr i][lane l] (rows = l 0..31, cols = i 0..7):\n");
    std::printf("      ");
    for (int i = 0; i < 8; ++i) std::printf("  i%-3d", i);
    std::printf("\n");
    for (int l = 0; l < 32; ++l)
    {
        std::printf("l%-4d ", l);
        for (int i = 0; i < 8; ++i)
        {
            const double v = got[i * 32 + l];
            int n = -1;
            for (int q = 0; q < 16; ++q)
                if (std::fabs(v - dec(value_for(q))) < 1e-3) n = q;
            if (n < 0) std::printf("  ?   ");
            else std::printf("  %-3d ", n);
        }
        std::printf("\n");
    }
    CHECK(hipFree(dx));
    CHECK(hipFree(dw));
    CHECK(hipFree(dout));
    return 0;
}
