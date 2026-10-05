// Test a translated PTX MMA, not a direct call to its lowering helpers.
// The first warp catches masked-EXEC padding (A=1, B=0.5, C=1 => D=17).
// The second checks every row/column with A[r,k]=(r+1)/8, B[k,c]=(c+1)/8.
// Its nonuniform bias C[r,c]=8*r+c+0.25 also checks accumulator layout conversion.
// A second MMA checks partial accumulator writes separated by a dependent read:
// double the low-row bias through the shadow, negate the high-row bias.
// All expected values are exactly representable in f16.
//
// Build: hipcc -O2 kernels/tools/rrswin_fp8_unpaired_test.cpp -o /tmp/fp8_unpaired_test
// Emit with the patched ZLUDA d4r_emit example:
//   env D4R_ZLUDA_WMMA=1 D4R_ZLUDA_WMMA_FP8=1 D4R_ZLUDA_WMMA_FP8_NATIVE=1 \
//     d4r_emit kernels/tools/rrswin_fp8_unpaired.ptx /tmp/fp8_unpaired gfx1201
// Run: /tmp/fp8_unpaired_test /tmp/fp8_unpaired/module.hsaco
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(expr) do { \
    const hipError_t error = (expr); \
    if (error != hipSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #expr, hipGetErrorString(error)); \
        std::exit(2); \
    } \
} while (0)

static uint16_t half_bits(float value)
{
    const _Float16 half = static_cast<_Float16>(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s module.hsaco\n", argv[0]);
        return 2;
    }
    constexpr unsigned lanes = 64;
    constexpr uint8_t eighths[16] = {
        0x20, 0x28, 0x2c, 0x30, 0x32, 0x34, 0x36, 0x38,
        0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40
    };
    uint32_t input[lanes * 8], output[lanes * 4];
    uint16_t expected[lanes * 8];
    for (unsigned i = 0; i < lanes; ++i) {
        const unsigned g = (i % 32) / 4, t = i % 4;
        const uint32_t low = i < 32 ? 0x38383838u : uint32_t(eighths[g]) * 0x01010101u;
        const uint32_t high = i < 32 ? low : uint32_t(eighths[g + 8]) * 0x01010101u;
        const uint32_t b = i < 32 ? 0x30303030u : uint32_t(eighths[g]) * 0x01010101u;
        uint32_t fragment[8] = {low, high, low, high, b, b, 0, 0};
        for (unsigned row = 0; row < 2; ++row)
            for (unsigned column = 0; column < 2; ++column) {
                const float bias = i < 32 ? 1.f : float(8 * (g + 8 * row) + 2 * t + column) + 0.25f;
                fragment[6 + row] |= uint32_t(half_bits(bias)) << (16 * column);
                const float product = i < 32 ? 16.f : float((g + 8 * row + 1) * (2 * t + column + 1)) / 2.f;
                expected[i * 8 + row * 2 + column] = half_bits(bias + product);
                expected[i * 8 + 4 + row * 2 + column] = half_bits((row == 0 ? 2.f : -1.f) * bias + product);
            }
        std::memcpy(input + i * 8, fragment, sizeof(fragment));
    }
    hipModule_t module;
    hipFunction_t function;
    CHECK(hipModuleLoad(&module, argv[1]));
    CHECK(hipModuleGetFunction(&function, module, "fp8_unpaired"));
    uint32_t *device_input, *device_output;
    CHECK(hipMalloc(&device_input, sizeof(input)));
    CHECK(hipMalloc(&device_output, sizeof(output)));
    CHECK(hipMemcpy(device_input, input, sizeof(input), hipMemcpyHostToDevice));
    CHECK(hipMemset(device_output, 0, sizeof(output)));
    void *args[] = {&device_input, &device_output};
    CHECK(hipModuleLaunchKernel(function, 2, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));
    uint16_t observed[lanes * 8];
    std::memcpy(observed, output, sizeof(observed));
    unsigned bad = 0;
    for (unsigned i = 0; i < lanes * 8; ++i)
        if (observed[i] != expected[i]) {
            if (bad < 8)
                std::fprintf(stderr, "warp=%u lane=%u case=%u half=%u: got %04x, expected %04x\n",
                             i / 256, (i / 8) % 32, (i / 4) % 2, i % 4, observed[i], expected[i]);
            ++bad;
        }
    CHECK(hipFree(device_output));
    CHECK(hipFree(device_input));
    CHECK(hipModuleUnload(module));
    std::printf("unpaired FP8 MMA and partial accumulators: %u/512 differing half values\n", bad);
    return bad ? 1 : 0;
}
