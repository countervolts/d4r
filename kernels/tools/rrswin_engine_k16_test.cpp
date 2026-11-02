// Steps 1 and 2 of the engine's validation: e4m3 GEMMs in real shared memory, every HIP call
// checked, compared against an independently computed host model.
//
//   step 1  one k16 GEMM (the second k16 half is zero)   -- operand form, lane/half indexing,
//                                                           and the D layout the engine assumes
//   step 2  a 64-wide k chain as two k32 steps with the accumulator rounded to f16 between
//           them, which is what kernels/tex/enc0_tail.hip does (`acc[i] = (half_t)cf[i];
//           cf[i] = (float)acc[i];`)
//
// The host model is e4m3 decode plus a double-precision sum, written from the layout facts in
// kernels/common/wmma_layout.h and the usage in kernels/tex/enc0_tail.hip; it shares no code
// with the device side.
//
// Build and run:
//   hipcc --offload-arch=gfx1201 -DD4R_TEX_FP8 -DD4R_WMMA_LAYOUT=12 -O2 \
//         -o /tmp/engine_test kernels/tools/rrswin_engine_k16_test.cpp && /tmp/engine_test
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
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

// the documented enc0 weight layout, written here independently of the device header
static int rrswin_waddr_host(int k, int n)
{
    return 512 * (n >> 4) + 64 * (n & 7) + 16 * ((k >> 1) & 3) + 8 * ((n >> 3) & 1)
         + 4 * ((k >> 3) & 1) + 2 * ((k >> 4) & 1) + (k & 1);
}

static const int kRows = 16;
static const int kChunk = 16;          // bytes per k16 chunk
static const int kChunks = 4;          // k = 64 in four k16 chunks (= two k32 steps)
static const int kRowBytes = kChunk * kChunks;
static const int kSlots = 128;         // prepared weight slots: 32 per k32 step (16 rows x 2 k16), two steps per GEMM

static uint8_t enc(double v)
{
    const int sign = v < 0;
    v = std::fabs(v);
    if (v < 0.001953125)
        return (uint8_t)(sign ? 0x80 : 0x00);
    int e = 0;
    while (v >= 2.0) { v /= 2.0; ++e; }
    while (v < 1.0) { v *= 2.0; --e; }
    const int code = ((e + 7) << 3) | (int)std::lround((v - 1.0) * 8.0);
    return (uint8_t)(code | (sign ? 0x80 : 0));
}

static double dec(uint8_t c)
{
    const int sign = c & 0x80, e = (c >> 3) & 0xf, m = c & 7;
    const double mag = e == 0 ? (m / 8.0) * std::pow(2.0, -6.0) : (1.0 + m / 8.0) * std::pow(2.0, e - 7.0);
    return sign ? -mag : mag;
}

// D VGPR i of half h is row i + 8h and lane l is column l & 15, so a value's (row, column)
// fixes the lane and slot that must hold it.
static void check(const char* what, const std::vector<float>& got, const std::vector<uint8_t>& w,
                  const std::vector<uint8_t>& x, int kk, double tol)
{
    int bad = 0, checked = 0;
    for (int l = 0; l < 32; ++l)
    {
        const int h = l >> 4, rcol = l & 15;
        for (int i = 0; i < 8; ++i)
        {
            const int row = i + 8 * h;
            // weight row `row`'s k16 chunk h of k32 step s is stored at slot 32*s + 16*h + row,
            // which is byte 16*slot -- NOT inside the same 16 bytes as the row's first chunk.
            auto wslot = [&](int s, int chunk, int r) { return (size_t)(32 * s + 16 * chunk + r); };
            auto dot = [&](int from, int to) {
                double s = 0.0;
                for (int c = from; c < to; ++c)
                    s += dec(w[wslot(c / 32, (c % 32) / 16, row) * kChunk + (c % 16)]) *
                         dec(x[(size_t)rcol * kRowBytes + c]);
                return s;
            };
            double ref = dot(0, kk);
            if (kk > 32)
            {
                // the caller's f16 rounding of the accumulator between the two k32 steps
                const double first = (double)(_Float16)dot(0, 32);
                ref = first + dot(32, kk);
            }
            const float g = got[i * 32 + l];
            ++checked;
            if (std::fabs((double)g - ref) > tol * (1.0 + std::fabs(ref)))
            {
                if (bad < 6)
                    std::printf("  lane %2d vgpr %d: D[row %2d][col %2d]  got %+.6f  ref %+.6f\n",
                                l, i, row, rcol, g, ref);
                ++bad;
            }
        }
    }
    std::printf("%s: checked %d, mismatched %d\n", what, checked, bad);
    if (bad)
    {
        // report which (row, column) each mismatching value *does* correspond to, which is
        // the mapping the engine would have to use
        int shown = 0;
        for (int l = 0; l < 32 && shown < 4; ++l)
        {
            for (int i = 0; i < 8 && shown < 4; ++i)
            {
                const float g = got[i * 32 + l];
                int best_row = -1, best_col = -1;
                double best = 1e30;
                for (int row = 0; row < 16; ++row)
                    for (int col = 0; col < 16; ++col)
                    {
                        double ref = 0.0;
                        for (int c = 0; c < kk; ++c)
                            ref += dec(w[(size_t)(32 * (c / 32) + 16 * ((c % 32) / 16) + row) * kChunk + (c % 16)]) *
                                   dec(x[(size_t)col * kRowBytes + c]);
                        if (std::fabs((double)g - ref) < best) { best = std::fabs((double)g - ref); best_row = row; best_col = col; }
                    }
                if (best < tol)
                {
                    std::printf("  lane %2d vgpr %d = D[row %2d][col %2d] (engine assumed row %2d col %2d)\n",
                                l, i, best_row, best_col, i + 8 * (l >> 4), l & 15);
                    ++shown;
                }
            }
        }
        std::exit(1);
    }
}

// The requantisation the device performs in swin_stage_operand, written independently here:
// RNE satfinite e4m3 of an f16 pair, low half to the low byte.
static uint8_t requant_code(uint16_t h)
{
    const uint32_t mag = h & 0x7fffu;
    uint32_t normal = ((mag - 0x2000u + 0x3fu + ((mag >> 7) & 1u)) & 0xffffu) >> 7;
    const uint32_t e = mag >> 10;
    const uint32_t clamped = e < 1u ? 1u : (e > 8u ? 8u : e);
    const uint32_t shift = 16u - clamped;
    const uint32_t sig = (mag & 0x3ffu) | (e != 0 ? 0x400u : 0u);
    const uint32_t sub = (sig + (1u << (shift - 1u)) - 1u + ((sig >> shift) & 1u)) >> shift;
    uint32_t code = mag >= 0x2400u ? normal : sub;
    if (mag > 0x5f00u)
        code = 0x7eu;
    if (mag > 0x7c00u)
        code = 0x7fu;
    return (uint8_t)(((h >> 8) & 0x80u) | code);
}

// the f16 bits the device would round to, then the e4m3 code
static uint8_t requant_code_bytes(double v)
{
    const _Float16 half = (_Float16)v;
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return requant_code(bits);
}

static double requant(double v)
{
    const _Float16 half = (_Float16)v;
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return dec(requant_code(bits));
}

__global__ void step1_kernel(const uint8_t* x_img, const uint8_t* w_prep, float* out)
{
    __shared__ uint8_t img[kRows * kRowBytes];
    for (int i = threadIdx.x; i < kRows * kRowBytes; i += blockDim.x)
        img[i] = x_img[i];
    __syncthreads();
    f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = rrswin::swin_gemm_tile(c, (const kslot*)w_prep, 0, 16, img, kRowBytes, 0);
    for (int i = 0; i < 8; ++i)
        out[i * 32 + threadIdx.x] = c[i];
}

__global__ void step2_kernel(const uint8_t* x_img, const uint8_t* w_prep, float* out)
{
    __shared__ uint8_t img[kRows * kRowBytes];
    for (int i = threadIdx.x; i < kRows * kRowBytes; i += blockDim.x)
        img[i] = x_img[i];
    __syncthreads();
    f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = rrswin::swin_gemm_tile(c, (const kslot*)w_prep, 0, 16, img, kRowBytes, 0);
    half_t acc[8];
    for (int i = 0; i < 8; ++i)
        acc[i] = (half_t)c[i];
    for (int i = 0; i < 8; ++i)
        c[i] = (float)acc[i];
    // the second k32 step: slots 32..63 and the image's chunks 2 and 3
    c = rrswin::swin_gemm_tile(c, (const kslot*)w_prep, 32, 16, img + 2 * kChunk, kRowBytes, 0);
    for (int i = 0; i < 8; ++i)
        out[i * 32 + threadIdx.x] = c[i];
}

// Step 3: the accumulator-to-operand mapping.  A first k32 GEMM produces D, D is staged as
// the next phase's activation image with swin_stage_operand, and a second k32 GEMM consumes
// it against a second weight slot.  The host model re-does exactly that, requantising with
// its own e4m3 encoder, so a wrong mapping (row/column/k-half) shows up as a mismatch.
__global__ void step3_kernel(const uint8_t* x_img, const uint8_t* w_prep, uint8_t* staged, float* out)
{
    __shared__ uint8_t img[kRows * kRowBytes];
    for (int i = threadIdx.x; i < kRows * kRowBytes; i += blockDim.x)
        img[i] = x_img[i];
    __syncthreads();
    f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = rrswin::swin_gemm_tile(c, (const kslot*)w_prep, 0, 16, img, kRowBytes, 0);
    __syncthreads();
    // the staging writes 16 rows, i.e. k 0..15; the image's second k16 chunk must not keep
    // the values the first GEMM's operand left there, or the second GEMM sums them too
    for (int i = threadIdx.x; i < kRows * kRowBytes; i += blockDim.x)
        if ((i % kRowBytes) >= 16)
            img[i] = 0;
    __syncthreads();
    rrswin::swin_stage_operand(img, kRowBytes, 0, 0, c);
    __syncthreads();
    for (int i = threadIdx.x; i < kRows * kRowBytes; i += blockDim.x)
        staged[i] = img[i];
    __syncthreads();
    f8v d = {0, 0, 0, 0, 0, 0, 0, 0};
    d = rrswin::swin_gemm_tile(d, (const kslot*)w_prep, 64, 16, img, kRowBytes, 0);
    for (int i = 0; i < 8; ++i)
        out[i * 32 + threadIdx.x] = d[i];
}

// Step 4: the weight preparation.  Lane t prepares weight row t's two k16 slots (32 bytes), then
// the phase runs with a known activation image and the result is compared against a host model
// that reads the weight image through the documented waddr.  This is what proves the prep's slot
// addressing, which the GEMM alone cannot see.
__global__ void step4_kernel(const uint8_t* x_img, const uint8_t* wimg, uint8_t* slots, float* out)
{
    __shared__ uint8_t img[kRows * kRowBytes];
    for (int i = threadIdx.x; i < kRows * kRowBytes; i += blockDim.x)
        img[i] = x_img[i];
    __syncthreads();
    const int t = threadIdx.x & 15;
    if ((threadIdx.x >> 4) == 0)
        rrswin::swin_prep_phase(slots, 0, 16, 1, wimg, rrswin::swin_waddr_enc0);
    __syncthreads();
    f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = rrswin::swin_gemm_tile(c, (const kslot*)slots, 0, 16, img, kRowBytes, 0);
    (void)t;
    for (int i = 0; i < 8; ++i)
        out[i * 32 + threadIdx.x] = c[i];
}

// Step 4c: a 32-row phase (two m-tiles of 16 weight rows).  slot_step is n_rows = 32, so tile 1's
// rows sit past tile 0's second k16 slot; with a fixed step of 16 it would read them as tile 0's
// second k16 operand.
__global__ void step4c_kernel(const uint8_t* x_img, const uint8_t* wimg, uint8_t* slots, float* out)
{
    __shared__ uint8_t img[kRows * kRowBytes];
    for (int i = threadIdx.x; i < kRows * kRowBytes; i += blockDim.x)
        img[i] = x_img[i];
    __syncthreads();
    if (threadIdx.x == 0)
        rrswin::swin_prep_phase(slots, 0, 32, 1, wimg, rrswin::swin_waddr_enc0);
    __syncthreads();
    f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
    c = rrswin::swin_gemm_tile(c, (const kslot*)slots, 0, 32, img, kRowBytes, 0);
    for (int i = 0; i < 8; ++i)
        out[i * 32 + threadIdx.x] = c[i];
    f8v d = {0, 0, 0, 0, 0, 0, 0, 0};
    d = rrswin::swin_gemm_tile(d, (const kslot*)slots, 16, 32, img, kRowBytes, 0);
    for (int i = 0; i < 8; ++i)
        out[256 + i * 32 + threadIdx.x] = d[i];
}

// Step 5: the phase loop.  Phase 1 is a 32-row (two m-tiles) x 16-token x k64 GEMM that stages
// its accumulator; phase 2 is 16x16x32 reading that staged image.  The host model redoes the
// whole chain, including the e4m3 requantisation between the phases.
__global__ void step5_kernel(const uint8_t* x_img, const uint8_t* wimg, uint8_t* slots,
                             uint8_t* staged, float* out)
{
    __shared__ uint8_t img[16 * 64];
    __shared__ uint8_t mid[16 * 32];
    for (int i = threadIdx.x; i < 16 * 64; i += blockDim.x)
        img[i] = x_img[i];
    for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x)
        mid[i] = 0;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        rrswin::swin_prep_phase(slots + 0, 0, 32, 2, wimg, rrswin::swin_waddr_enc0);      // phase 1
        rrswin::swin_prep_phase(slots + 32 * 32 * 2, 0, 16, 1, wimg + 8192, rrswin::swin_waddr_enc0);
    }
    __syncthreads();
    rrswin::Phase p1 = {2, 1, 2, 0, 0, 0};
    rrswin::swin_phase(p1, (const kslot*)slots, 32, img, 64, mid, 32, nullptr);
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x)
        staged[i] = mid[i];
    rrswin::Phase p2 = {1, 1, 1, 128, 1, 0};   // slot 128 = byte 2048, past phase 1's slots
    f8v acc[1];
    rrswin::swin_phase(p2, (const kslot*)slots, 16, mid, 32, nullptr, 0, acc);
    for (int i = 0; i < 8; ++i)
        out[i * 32 + threadIdx.x] = acc[0][i];
}

// Step 6: the clamped cubic activation.  The host model below is written from the specification
// in kernels/rr/rrswin_enc0.hip's comment (y = clamp(x, +-2), out = x * (0.5 + y * (0.412109375
// - 0.0810546875*|y|)), f16 throughout, constants 0x3698/0x2d30/0x3800/0x4000) rather than from
// the device code, so agreement means the port is faithful.
static uint16_t act_model(uint16_t bits)
{
    const auto as_half = [](uint16_t b) { _Float16 h; std::memcpy(&h, &b, sizeof(h)); return h; };
    const auto as_bits = [](const _Float16& h) { uint16_t b; std::memcpy(&b, &h, sizeof(b)); return b; };
    const _Float16 x = as_half(bits);
    const _Float16 two = as_half(0x4000), slope = as_half(0x2d30), mid = as_half(0x3698), half0 = as_half(0x3800);
    _Float16 y = x > -two ? x : -two;
    y = y < two ? y : two;
    y = y > -y ? y : -y;
    const _Float16 g = (_Float16)(half0 + y * (mid - slope * y));
    return as_bits((_Float16)(x * g));
}

__global__ void step6_kernel(const uint32_t* pairs, uint32_t* out)
{
    const uint32_t p = pairs[threadIdx.x];
    out[threadIdx.x] = rrswin::swin_act_pair(p);
}

// Step 7: fragment-to-image staging.  A host model of the documented layout builds the fragment
// registers from a known image, the device stages them back, and the result must equal the
// original image -- a round trip that fails if any (row, k) mapping is wrong.
static void fragment_model(const std::vector<uint8_t>& img, int stride, int m_tiles,
                           std::vector<uint32_t>& frag, int lane)
{
    const int g = lane >> 2, t = lane & 3;
    uint32_t* mine = frag.data() + (size_t)4 * m_tiles * lane;   // each lane's own registers
    for (int mi = 0; mi < m_tiles; ++mi)
        for (int j = 0; j < 4; ++j)
        {
            const uint32_t k0 = 2 * t + (j & 1) + 16u * ((j >> 1) & 1);
            const uint32_t k2 = 8u + k0;
            mine[4 * mi + 0] |= (uint32_t)img[(size_t)(16 * mi + g) * stride + k0] << (8 * j);
            mine[4 * mi + 1] |= (uint32_t)img[(size_t)(16 * mi + g + 8) * stride + k0] << (8 * j);
            mine[4 * mi + 2] |= (uint32_t)img[(size_t)(16 * mi + g) * stride + k2] << (8 * j);
            mine[4 * mi + 3] |= (uint32_t)img[(size_t)(16 * mi + g + 8) * stride + k2] << (8 * j);
        }
}

__global__ void step7_kernel(const uint32_t* frag, uint8_t* staged, int m_tiles)
{
    __shared__ uint8_t img[64 * 32];
    for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x)
        img[i] = 0;
    __syncthreads();
    rrswin::swin_stage_fragment(img, 32, frag + 4 * m_tiles * threadIdx.x, m_tiles);
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x)
        staged[i] = img[i];
}

int main()
{
    std::vector<uint8_t> x(kRows * kRowBytes, 0), w(kSlots * kChunk, 0);
    // Well-spread e4m3 codes: small/huge and equal sums must not collide, or the mapping
    // search below cannot tell one (row, column) from another.
    uint32_t lcg = 0x12345678u;
    auto next_code = [&lcg]() -> uint8_t {
        lcg = lcg * 1664525u + 1013904223u;
        uint32_t mag = (lcg >> 13) & 0x3fu;      // exponents 0..7 and mantissas
        if (mag == 0)
            mag = 0x21;
        const uint32_t sign = (lcg >> 7) & 1u;
        return (uint8_t)((sign << 7) | mag);
    };
    for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kChunks * kChunk; ++c)
        {
            // row r's k16 chunk h of k32 step `step` lives at slot 32*step + 16*h + r
            const int step = c / 32, h = (c % 32) / 16, kk = c % 16;
            w[(size_t)(32 * step + 16 * h + r) * kChunk + kk] = next_code();
            x[(size_t)r * kRowBytes + c] = next_code();
        }

    uint8_t *dx, *dw;
    float* dout;
    CHECK(hipMalloc(&dx, x.size()));
    CHECK(hipMalloc(&dw, w.size()));
    CHECK(hipMalloc(&dout, 8 * 32 * sizeof(float)));
    std::vector<float> got(8 * 32);

    CHECK(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
    step1_kernel<<<1, 32>>>(dx, dw, dout);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    check("step 1  one k32 step (k 0..31)", got, w, x, 32, 2e-3);

    step2_kernel<<<1, 32>>>(dx, dw, dout);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    check("step 2  k64 as two k32 with f16 rounding", got, w, x, 64, 2e-3);

    // step 3: two chained GEMMs with the accumulator staged as the next operand
    {
        std::vector<uint8_t> w2 = w;                      // slots 64..95 are zero so far
        for (int r = 0; r < kRows; ++r)
            for (int c = 0; c < 32; ++c)
            {
                const int h = c / 16, kk = c % 16;
                w2[(size_t)(64 + 16 * h + r) * kChunk + kk] = next_code();
            }
        CHECK(hipMemcpy(dw, w2.data(), w2.size(), hipMemcpyHostToDevice));
        uint8_t* dstaged;
        CHECK(hipMalloc(&dstaged, kRows * kRowBytes));
        step3_kernel<<<1, 32>>>(dx, dw, dstaged, dout);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));

        std::vector<uint8_t> staged(kRows * kRowBytes, 0);
        CHECK(hipMemcpy(staged.data(), dstaged, staged.size(), hipMemcpyDeviceToHost));
        auto w1_at = [&](int m, int k) { return w[(size_t)(32 * (k / 32) + 16 * ((k % 32) / 16) + m) * kChunk + (k % 16)]; };
        // where did each expected staged byte land, and what is at its expected slot?
        int pl = 0;
        for (int n = 0; n < 16 && pl < 6; ++n)
            for (int m = 0; m < 16 && pl < 6; ++m)
            {
                double d1 = 0.0;
                for (int j = 0; j < 32; ++j)
                    d1 += dec(w1_at(m, j)) * dec(x[(size_t)n * kRowBytes + j]);
                const uint8_t want = requant_code(((_Float16)(float)d1) ? [&] {
                    const _Float16 h = (_Float16)d1;
                    uint16_t b;
                    std::memcpy(&b, &h, sizeof(b));
                    return b;
                }() : 0);
                const uint8_t have = staged[(size_t)n * kRowBytes + m];
                std::printf("  staged[col %2d][k %2d] = 0x%02x, expected 0x%02x%s\n", n, m, have, want,
                            have == want ? "" : "  <-- differs");
                ++pl;
            }
        auto w2_at = [&](int m, int k) { return w2[(size_t)(64 + 32 * (k / 32) + 16 * ((k % 32) / 16) + m) * kChunk + (k % 16)]; };
        int bad = 0, checked = 0;
        for (int l = 0; l < 32; ++l)
        {
            const int h = l >> 4, col = l & 15;
            for (int i = 0; i < 8; ++i)
            {
                const int m = i + 8 * h;
                // The staging covers 16 rows (the first GEMM's m), so the second GEMM's
                // k 16..31 operand is zero; the model sums the same 16 k.
                double ref = 0.0;
                for (int k = 0; k < 16; ++k)
                {
                    double d1 = 0.0;
                    for (int j = 0; j < 32; ++j)
                        d1 += dec(w1_at(k, j)) * dec(x[(size_t)col * kRowBytes + j]);
                    ref += dec(w2_at(m, k)) * requant(d1);
                }
                const float g = got[i * 32 + l];
                ++checked;
                if (std::fabs((double)g - ref) > 2e-3 * (1.0 + std::fabs(ref)))
                {
                    if (bad < 4)
                        std::printf("  step3 lane %2d vgpr %d: got %+.6f ref %+.6f\n", l, i, g, ref);
                    ++bad;
                }
            }
        }
        std::printf("step 3  accumulator staged as the next operand: checked %d, mismatched %d\n",
                    checked, bad);
        CHECK(hipFree(dstaged));
        if (bad)
            std::exit(1);
    }
    // step 4: weight preparation, checked through the GEMM against the documented layout
    {
        std::vector<uint8_t> wimg(65536, 0);
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 16; ++n)
                wimg[rrswin_waddr_host(k, n)] = next_code();

        uint8_t* dwimg;
        uint8_t* dslots;
        CHECK(hipMalloc(&dwimg, wimg.size()));
        CHECK(hipMalloc(&dslots, 16 * 16 * 2));
        CHECK(hipMemset(dslots, 0, 16 * 16 * 2));
        CHECK(hipMemcpy(dwimg, wimg.data(), wimg.size(), hipMemcpyHostToDevice));
        step4_kernel<<<1, 32>>>(dx, dwimg, dslots, dout);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));

        std::vector<uint8_t> slotsHost(16 * 16 * 2, 0);
        CHECK(hipMemcpy(slotsHost.data(), dslots, slotsHost.size(), hipMemcpyDeviceToHost));
        int bad = 0, checked = 0;
        for (int hus = 0; hus < 2; ++hus)
            for (int n = 0; n < 16; ++n)
                for (int j = 0; j < 16; ++j)
                {
                    const uint8_t have = slotsHost[16 * (16 * hus + n) + j];
                    const uint8_t want = wimg[rrswin_waddr_host(16 * hus + j, n)];
                    ++checked;
                    if (have != want)
                    {
                        if (bad < 4)
                            std::printf("  slot[h%u n%2d j%2d] = 0x%02x, expected 0x%02x\n", hus, n, j, have, want);
                        ++bad;
                    }
                }
        std::printf("step 4a prep slots: checked %d, mismatched %d\n", checked, bad);

        int gbad = 0, gchecked = 0;
        for (int l = 0; l < 32; ++l)
        {
            const int h = l >> 4, col = l & 15;
            for (int i = 0; i < 8; ++i)
            {
                const int n = i + 8 * h;
                double ref = 0.0;
                for (int k = 0; k < 32; ++k)
                    ref += dec(wimg[rrswin_waddr_host(k, n)]) * dec(x[(size_t)col * kRowBytes + k]);
                const float g = got[i * 32 + l];
                ++gchecked;
                if (std::fabs((double)g - ref) > 2e-3 * (1.0 + std::fabs(ref)))
                {
                    if (gbad < 4)
                        std::printf("  step4 GEMM lane %2d vgpr %d: got %+.6f ref %+.6f\n", l, i, g, ref);
                    ++gbad;
                }
            }
        }
        std::printf("step 4b prep through the GEMM: checked %d, mismatched %d\n", gchecked, gbad);


        CHECK(hipFree(dwimg));
        CHECK(hipFree(dslots));
        if (bad || gbad)
            std::exit(1);
    }
    // step 4c: 32 rows, two m-tiles, slot_step = n_rows
    {
        std::vector<uint8_t> wimg(131072, 0);
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 32; ++n)
                wimg[rrswin_waddr_host(k, n)] = next_code();

        uint8_t* dwimg;
        uint8_t* dslots;
        float* dout2;
        CHECK(hipMalloc(&dwimg, wimg.size()));
        CHECK(hipMalloc(&dslots, 32 * 16 * 2));
        CHECK(hipMalloc(&dout2, 512 * sizeof(float)));
        CHECK(hipMemset(dslots, 0, 32 * 16 * 2));
        CHECK(hipMemcpy(dwimg, wimg.data(), wimg.size(), hipMemcpyHostToDevice));
        step4c_kernel<<<1, 32>>>(dx, dwimg, dslots, dout2);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<float> got2(512);
        CHECK(hipMemcpy(got2.data(), dout2, got2.size() * sizeof(float), hipMemcpyDeviceToHost));

        int bad = 0, checked = 0;
        for (int tile = 0; tile < 2; ++tile)
            for (int l = 0; l < 32; ++l)
            {
                const int h = l >> 4, col = l & 15;
                for (int i = 0; i < 8; ++i)
                {
                    const int n = 16 * tile + i + 8 * h;
                    double ref = 0.0;
                    for (int k = 0; k < 32; ++k)
                        ref += dec(wimg[rrswin_waddr_host(k, n)]) * dec(x[(size_t)col * kRowBytes + k]);
                    const float g = got2[tile * 256 + i * 32 + l];
                    ++checked;
                    if (std::fabs((double)g - ref) > 2e-3 * (1.0 + std::fabs(ref)))
                    {
                        if (bad < 4)
                            std::printf("  step4c tile %d lane %2d vgpr %d: got %+.6f ref %+.6f\n",
                                        tile, l, i, g, ref);
                        ++bad;
                    }
                }
            }
        std::printf("step 4c 32-row phase, two m-tiles: checked %d, mismatched %d\n", checked, bad);
        CHECK(hipFree(dwimg));
        CHECK(hipFree(dslots));
        CHECK(hipFree(dout2));
        if (bad)
            std::exit(1);
    }
    // step 5: the phase loop, two phases chained through a staged image
    {
        std::vector<uint8_t> wimg(32768, 0);
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 48; ++n)
                wimg[(n < 32 ? rrswin_waddr_host(k, n) : 8192 + rrswin_waddr_host(k, n - 32))] = next_code();

        uint8_t *dwimg, *dslots, *dstaged;
        float* dout2;
        CHECK(hipMalloc(&dwimg, wimg.size()));
        CHECK(hipMalloc(&dslots, (32 * 32 * 2 + 16 * 16 * 2)));
        CHECK(hipMalloc(&dstaged, 16 * 32));
        CHECK(hipMalloc(&dout2, 256 * sizeof(float)));
        CHECK(hipMemset(dslots, 0, 32 * 32 * 2 + 16 * 16 * 2));
        CHECK(hipMemcpy(dwimg, wimg.data(), wimg.size(), hipMemcpyHostToDevice));
        step5_kernel<<<1, 32>>>(dx, dwimg, dslots, dstaged, dout2);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<float> got2(256);
        CHECK(hipMemcpy(got2.data(), dout2, got2.size() * sizeof(float), hipMemcpyDeviceToHost));

        std::vector<uint8_t> staged(16 * 32, 0);
        CHECK(hipMemcpy(staged.data(), dstaged, staged.size(), hipMemcpyDeviceToHost));
        auto w1 = [&](int m, int k) { return wimg[rrswin_waddr_host(k, m)]; };
        {
            int sbad = 0, schecked = 0;
            for (int col = 0; col < 16 && sbad < 4; ++col)
                for (int m = 0; m < 32; ++m)
                {
                    double d1 = 0.0;
                    for (int j = 0; j < 64; ++j)
                        d1 += dec(w1(m, j)) * dec(x[(size_t)col * kRowBytes + j]);
                    const uint8_t want = requant_code_bytes(d1);
                    const uint8_t have = staged[(size_t)col * 32 + m];
                    ++schecked;
                    if (have != want)
                    {
                        if (sbad < 4)
                            std::printf("  staged5[col %2d][k %2d] = 0x%02x, expected 0x%02x\n", col, m, have, want);
                        ++sbad;
                    }
                }
            std::printf("step 5a staged image: checked %d, mismatched %d\n", schecked, sbad);
        }
        auto w2 = [&](int m, int k) { return wimg[8192 + rrswin_waddr_host(k, m)]; };
        int bad = 0, checked = 0;
        for (int l = 0; l < 32; ++l)
        {
            const int h = l >> 4, col = l & 15;
            for (int i = 0; i < 8; ++i)
            {
                const int n = i + 8 * h;
                // phase 1's staged image: row = token col, k = phase 1's m
                // phase 2's k spans 32: phase 1's two m-tiles land in the staged image's two
                // k16 chunks
                double ref = 0.0;
                for (int k = 0; k < 32; ++k)
                {
                    double d1 = 0.0;
                    for (int j = 0; j < 64; ++j)
                        d1 += dec(w1(k, j)) * dec(x[(size_t)col * kRowBytes + j]);
                    ref += dec(w2(n, k)) * requant(d1);
                }
                const float g = got2[i * 32 + l];
                ++checked;
                if (std::fabs((double)g - ref) > 3e-3 * (1.0 + std::fabs(ref)))
                {
                    if (bad < 4)
                        std::printf("  step5 lane %2d vgpr %d: got %+.6f ref %+.6f\n", l, i, g, ref);
                    ++bad;
                }
            }
        }
        std::printf("step 5 phase loop, two phases: checked %d, mismatched %d\n", checked, bad);
        CHECK(hipFree(dwimg));
        CHECK(hipFree(dslots));
        CHECK(hipFree(dstaged));
        CHECK(hipFree(dout2));
        if (bad)
            std::exit(1);
    }
    // step 6: the clamped cubic activation, over f16 pairs spanning and crossing the clamps
    {
        std::vector<uint32_t> pairs(64);
        for (int i = 0; i < 64; ++i)
        {
            const _Float16 lo = (_Float16)((i - 32) * 0.21f);
            const _Float16 hi = (_Float16)((i - 16) * -0.13f + 1.7f);
            uint16_t bl, bh;
            std::memcpy(&bl, &lo, sizeof(bl));
            std::memcpy(&bh, &hi, sizeof(bh));
            pairs[i] = (uint32_t)bl | ((uint32_t)bh << 16);
        }
        uint32_t *dp, *dop;
        CHECK(hipMalloc(&dp, pairs.size() * 4));
        CHECK(hipMalloc(&dop, pairs.size() * 4));
        CHECK(hipMemcpy(dp, pairs.data(), pairs.size() * 4, hipMemcpyHostToDevice));
        step6_kernel<<<1, 64>>>(dp, dop);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> got6(pairs.size());
        CHECK(hipMemcpy(got6.data(), dop, got6.size() * 4, hipMemcpyDeviceToHost));
        int bad = 0, checked = 0;
        for (int i = 0; i < 64; ++i)
        {
            const uint32_t want = (uint32_t)act_model((uint16_t)(pairs[i] & 0xffff))
                                | ((uint32_t)act_model((uint16_t)(pairs[i] >> 16)) << 16);
            ++checked;
            // The device applies the formula with packed f16x2 operations, which is what the PTX
            // does; this scalar _Float16 model can land one ulp away.  A logic error moves whole
            // clamps, so one ulp is the right bar here; the oracle comparison is the bit-exact
            // check, later.
            auto within_ulp = [](uint16_t a, uint16_t b) {
                // compare as f16 by ordering the sign-magnitude representation
                const auto key = [](uint16_t v) -> int {
                    return (v & 0x8000) ? (int)(0x8000 - (v & 0x7fff)) : (int)(v & 0x7fff);
                };
                const int d = key(a) - key(b);
                return d >= -1 && d <= 1;
            };
            if (!within_ulp((uint16_t)(got6[i] & 0xffff), (uint16_t)(want & 0xffff)) ||
                !within_ulp((uint16_t)(got6[i] >> 16), (uint16_t)(want >> 16)))
            {
                if (bad < 4)
                    std::printf("  act pair %2d: got %08x want %08x\n", i, got6[i], want);
                ++bad;
            }
        }
        std::printf("step 6 clamped cubic activation (1 ulp): checked %d, mismatched %d\n", checked, bad);
        CHECK(hipFree(dp));
        CHECK(hipFree(dop));
        if (bad)
            std::exit(1);
    }
    // step 7: fragment -> image staging round trip
    {
        const int m_tiles = 4;
        std::vector<uint8_t> src(64 * 32, 0);
        uint32_t lcg7 = 0x2468ace0u;
        for (auto& v : src)
        {
            lcg7 = lcg7 * 1664525u + 1013904223u;
            v = (uint8_t)((lcg7 >> 21) & 0x7fu);
        }
        std::vector<uint32_t> frag(32 * 4 * m_tiles, 0);
        for (int lane = 0; lane < 32; ++lane)
            fragment_model(src, 32, m_tiles, frag, lane);

        uint32_t* dfrag;
        uint8_t* dstaged7;
        CHECK(hipMalloc(&dfrag, frag.size() * 4));
        CHECK(hipMalloc(&dstaged7, src.size()));
        CHECK(hipMemcpy(dfrag, frag.data(), frag.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemset(dstaged7, 0, src.size()));
        step7_kernel<<<1, 32>>>(dfrag, dstaged7, m_tiles);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint8_t> got7(src.size(), 0);
        CHECK(hipMemcpy(got7.data(), dstaged7, src.size(), hipMemcpyDeviceToHost));
        int bad = 0, checked = 0;
        for (size_t i = 0; i < src.size(); ++i)
        {
            ++checked;
            if (got7[i] != src[i])
            {
                if (bad < 4)
                    std::printf("  stage byte %zu (row %zu k %zu): got 0x%02x want 0x%02x\n", i, i / 32, i % 32,
                                got7[i], src[i]);
                ++bad;
            }
        }
        std::printf("step 7 fragment staging round trip: checked %d, mismatched %d\n", checked, bad);
        CHECK(hipFree(dfrag));
        CHECK(hipFree(dstaged7));
        if (bad)
            std::exit(1);
    }
    CHECK(hipFree(dx));
    CHECK(hipFree(dw));
    CHECK(hipFree(dout));
    return 0;
}
