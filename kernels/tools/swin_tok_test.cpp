// Build with HIP clang: -x hip --offload-arch=gfx1101 -O2 (ROCm headers/libs).
// Exhaustive finite binary16 quantization and nonuniform half-wave layout
// checks.
#define SWIN_GFX11_TOKEN
#define SWIN_TOKEN_LANES
#define WM_HALF_BPERMUTE
#include "../m/swin_block.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define HIP(call)                                                              \
  do {                                                                         \
    auto e = (call);                                                           \
    if (e != hipSuccess) {                                                     \
      std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e));           \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
__global__ void quantize(uint16_t *out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const hv2 h = {__builtin_bit_cast(half_t, (uint16_t)i),
                 __builtin_bit_cast(half_t, (uint16_t)(65535 - i))};
  const hv2 q = q8x2_exact(h);
  const u16x2 bits = __builtin_bit_cast(u16x2, q);
  out[2 * i] = bits[0];
  out[2 * i + 1] = bits[1];
}
__global__ void layout_check(unsigned *out) {
  const int l = threadIdx.x, hf = l >> 4;
  T16 t;
  for (int k = 0; k < 4; ++k)
    t.set_pair(k,
               (hv2){(half_t)(l * 16 + 2 * k), (half_t)(l * 16 + 2 * k + 1)});
  for (int k = 0; k < 4; ++k) {
    const hv2 p = t.pair(k);
    out[8 * l + 2 * k] = (unsigned)p[0];
    out[8 * l + 2 * k + 1] = (unsigned)p[1];
  }
  // Also check physical WMMA row ownership directly, rather than a circular
  // round trip.
  for (int i = 0; i < 8; ++i) {
    const int row = 2 * i + hf, owner = (l & 15) + 16 * (row / 8);
    out[256 + 8 * l + i] = (unsigned)t.v[i / 2][i & 1];
    out[512 + 8 * l + i] = owner * 16 + row % 8;
  }
}
__global__ void code_check(uint32_t *out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const hv2 h = {__builtin_bit_cast(half_t, (uint16_t)i),
                 __builtin_bit_cast(half_t, (uint16_t)(65535 - i))};
  hv2 rounded;
  out[2 * i] = direct_codes8x2(h);
  out[2 * i + 1] = enc8x2(h, rounded);
}
static float half_value(uint16_t bits) {
  const unsigned m = bits & 1023, e = (bits >> 10) & 31;
  const float v = e ? __builtin_ldexpf(float(1024 + m), int(e) - 25)
                    : __builtin_ldexpf(float(m), -24);
  return bits & 0x8000 ? -v : v;
}
int main() {
  uint16_t *d;
  HIP(hipMalloc(&d, 65536 * 2 * sizeof(uint16_t)));
  hipLaunchKernelGGL(quantize, dim3(256), dim3(256), 0, 0, d);
  HIP(hipGetLastError());
  std::vector<uint16_t> q(65536 * 2);
  HIP(hipMemcpy(q.data(), d, q.size() * 2, hipMemcpyDeviceToHost));
  unsigned tested = 0;
  for (unsigned i = 0; i < 65536; ++i) {
    const unsigned m = i & 0x7fff;
    if (m > 0x7c00) {
      // The register quantizer retains the original NaN payload/sign;
      // this is deliberately separate from the finite E4M3 oracle.
      if (q[2 * i] != i || q[2 * (65535 - i) + 1] != i)
        return 1;
      continue;
    }
    const float a = m == 0x7c00 ? 1e30f : half_value(m);
    unsigned best = 0;
    float delta = 1e30f;
    for (unsigned c = 0; c < 127; ++c) {
      const float value =
          c < 8 ? c / 512.0f
                : __builtin_ldexpf(float(8 + (c & 7)), int(c / 8) - 10);
      const float diff = __builtin_fabsf(a - value);
      if (diff < delta || (diff == delta && !(c & 1))) {
        best = c;
        delta = diff;
      }
    }
    if (m >= 0x5f00)
      best = 126;
    const float expected =
        best < 8 ? best / 512.0f
                 : __builtin_ldexpf(float(8 + (best & 7)), int(best / 8) - 10);
    for (unsigned pos = 0; pos < 2; ++pos) {
      const uint16_t got = q[2 * (pos ? 65535 - i : i) + pos];
      if ((got & 0x8000) != (i & 0x8000) ||
          half_value(got & 0x7fff) != expected) {
        std::fprintf(stderr,
                     "quantization mismatch half=%04x pos=%u got=%04x\n", i,
                     pos, got);
        return 1;
      }
    }
    ++tested;
  }
  HIP(hipFree(d));
  unsigned *ld;
  HIP(hipMalloc(&ld, 768 * sizeof(unsigned)));
  hipLaunchKernelGGL(layout_check, dim3(1), dim3(32), 0, 0, ld);
  HIP(hipGetLastError());
  std::vector<unsigned> got(768);
  HIP(hipMemcpy(got.data(), ld, got.size() * sizeof(unsigned),
                hipMemcpyDeviceToHost));
  for (unsigned l = 0; l < 32; ++l)
    for (unsigned i = 0; i < 8; ++i)
      if (got[8 * l + i] != 16 * l + i ||
          got[256 + 8 * l + i] != got[512 + 8 * l + i]) {
        std::fprintf(stderr, "accumulator layout mismatch lane=%u row=%u\n", l,
                     i);
        return 1;
      }
  HIP(hipFree(ld));
  uint32_t *cd;
  HIP(hipMalloc(&cd, 65536 * 2 * sizeof(uint32_t)));
  hipLaunchKernelGGL(code_check, dim3(256), dim3(256), 0, 0, cd);
  HIP(hipGetLastError());
  std::vector<uint32_t> codes(65536 * 2);
  HIP(hipMemcpy(codes.data(), cd, codes.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < 65536; ++i)
    if (codes[2 * i] != codes[2 * i + 1]) {
      std::fprintf(stderr, "direct output codes mismatch half=%04x\n", i);
      return 1;
    }
  HIP(hipFree(cd));
  std::puts("65536 direct output encodings match the original encoder in both packed positions");
  std::printf("%u finite/infinity and 2046 NaN half encodings in both packed "
              "positions; 256 nonuniform accumulator rows passed\n",
              tested);
}
