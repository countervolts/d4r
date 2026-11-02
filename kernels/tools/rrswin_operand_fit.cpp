// Fit the engine's D mapping by measurement.
//
// The WMMA computes D[m][n] = sum_k A[m][k] B[k][n].  With A (the prepared weight slot) filled
// as value_for((b + c * r) % 16) in byte b of row r, and B (the activation image) an identity
// selector (row n holds 1.0 at byte n, zero elsewhere), the value that comes out at a given
// (vgpr, lane) is value_for((n + c * m) % 16): one equation in the two indices the engine's
// indexing must report.  Two runs with different c (1 and 3, both coprime with 16) solve for
// both.  Nothing here assumes which (r, n) is which -- the point is to measure it.
//
// Build and run:
//   hipcc --offload-arch=gfx1201 -DD4R_TEX_FP8 -DD4R_WMMA_LAYOUT=12 -O2 \
//         -o /tmp/fit kernels/tools/rrswin_operand_fit.cpp && /tmp/fit
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

static uint8_t value_for(int k)
{
    const int e = 1 + (k & 7), m = (k >> 3) & 1;
    return (uint8_t)((e << 3) | (m ? 4 : 0));
}

static double dec(uint8_t c)
{
    const int sign = c & 0x80, e = (c >> 3) & 0xf, m = c & 7;
    const double mag = e == 0 ? (m / 8.0) * std::pow(2.0, -6.0) : (1.0 + m / 8.0) * std::pow(2.0, e - 7.0);
    return sign ? -mag : mag;
}

static int value_index(double v)
{
    for (int q = 0; q < 16; ++q)
        if (std::fabs(v - dec(value_for(q))) < 1e-3)
            return q;
    return -1;
}

__global__ void fit_kernel(const uint8_t* x_img, const uint8_t* w_prep, float* out)
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

// run one pattern and return the value_for index at every (vgpr i, lane l)
static std::vector<int> run(int coefficient, const std::vector<uint8_t>& x)
{
    std::vector<uint8_t> w(32 * kChunk, 0);
    for (int r = 0; r < kRows; ++r)
        for (int b = 0; b < 16; ++b)
            w[(size_t)r * kChunk + b] = value_for((b + coefficient * r) % 16);

    uint8_t *dx, *dw;
    float* dout;
    CHECK(hipMalloc(&dx, x.size()));
    CHECK(hipMalloc(&dw, w.size()));
    CHECK(hipMalloc(&dout, 8 * 32 * sizeof(float)));
    CHECK(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    fit_kernel<<<1, 32>>>(dx, dw, dout);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<float> got(8 * 32);
    CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<int> idx(8 * 32);
    for (int i = 0; i < 8 * 32; ++i)
        idx[i] = value_index(got[i]);
    CHECK(hipFree(dx));
    CHECK(hipFree(dw));
    CHECK(hipFree(dout));
    return idx;
}

// Experiment 3: A row r carries a distinct value at byte 0 only, and every B row is 1.0 at
// byte 0, so D[i][l] reads back the A row that element was actually computed from.  This is
// the operand row selection, which experiment 2 cannot separate from the D indexing.
static std::vector<int> run_rows(const std::vector<uint8_t>& x)
{
    std::vector<uint8_t> w(32 * kChunk, 0);
    for (int r = 0; r < kRows; ++r)
        w[(size_t)r * kChunk + 0] = value_for(r);
    uint8_t *dx, *dw;
    float* dout;
    CHECK(hipMalloc(&dx, x.size()));
    CHECK(hipMalloc(&dw, w.size()));
    CHECK(hipMalloc(&dout, 8 * 32 * sizeof(float)));
    CHECK(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    fit_kernel<<<1, 32>>>(dx, dw, dout);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<float> got(8 * 32);
    CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<int> idx(8 * 32);
    for (int i = 0; i < 8 * 32; ++i)
        idx[i] = value_index(got[i]);
    CHECK(hipFree(dx));
    CHECK(hipFree(dw));
    CHECK(hipFree(dout));
    return idx;
}

int main()
{
    // B: row n holds 1.0 at byte n only.
    const uint8_t one = 0x38;
    std::vector<uint8_t> x(kRows * kRowBytes, 0);
    for (int n = 0; n < kRows; ++n)
        x[(size_t)n * kRowBytes + n] = one;

    const std::vector<int> a = run(1, x);
    const std::vector<int> b = run(3, x);
    const std::vector<int> c = run(5, x);

    std::printf("(vgpr i, lane l) -> (sum1, sum3) = ((n+m)%%16, (n+3m)%%16), solved m,n:\n");
    int shown = 0;
    for (int l = 0; l < 32 && shown < 14; l += 4)
        for (int i = 0; i < 8 && shown < 14; ++i, ++shown)
        {
            const int s1 = a[i * 32 + l], s3 = b[i * 32 + l], s5 = c[i * 32 + l];
            if (s1 < 0 || s3 < 0 || s5 < 0)
            {
                std::printf("  i%d l%-3d  unreadable (sums %d %d)\n", i, l, s1, s3);
                continue;
            }
            // Three equations n + c*m = s_c (mod 16) for c = 1, 3, 5 pin both indices down.
            int m = -1, n = -1, hits = 0;
            for (int mm = 0; mm < 16; ++mm)
                for (int nn = 0; nn < 16; ++nn)
                    if ((nn + mm) % 16 == s1 && (nn + 3 * mm) % 16 == s3 && (nn + 5 * mm) % 16 == s5)
                    {
                        if (m < 0) { m = mm; n = nn; }
                        ++hits;
                    }
            std::printf("  i%d l%-3d  sums %2d %2d %2d  -> m=%2d n=%2d%s\n", i, l, s1, s3, s5, m, n,
                        hits == 1 ? "" : " (AMBIGUOUS)");
        }
    // experiment 3
    {
        std::vector<uint8_t> xb(kRows * kRowBytes, 0);
        for (int n = 0; n < kRows; ++n)
            xb[(size_t)n * kRowBytes + 0] = one;
        const std::vector<int> rows = run_rows(xb);
        std::printf("\nA row used at (vgpr i, lane l):\n      ");
        for (int i = 0; i < 8; ++i) std::printf("  i%-2d", i);
        std::printf("\n");
        for (int l = 0; l < 32; ++l)
        {
            std::printf("l%-4d ", l);
            for (int i = 0; i < 8; ++i)
            {
                const int v = rows[i * 32 + l];
                if (v < 0) std::printf("  ?  ");
                else std::printf("  %-2d ", v);
            }
            std::printf("\n");
        }
    }
    return 0;
}
