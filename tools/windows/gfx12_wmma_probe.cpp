#include "hip_api.h"
#include <array>
#include <cmath>
#include <limits>

namespace {
uint16_t half_integer(int value)
{
    const unsigned n = static_cast<unsigned>(std::abs(value));
    if (n > 3u) throw std::runtime_error("test half integer out of range");
    const uint16_t bits[] = {0u, 0x3c00u, 0x4000u, 0x4200u};
    return static_cast<uint16_t>(bits[n] | (value < 0 ? 0x8000u : 0u));
}
struct Allocation {
    d4r::diag::HipApi& hip;
    void* pointer = nullptr;
    Allocation(d4r::diag::HipApi& api, size_t bytes) : hip(api) {
        hip.check(hip.hipMalloc(&pointer, bytes), "hipMalloc(WMMA)");
    }
    ~Allocation() { if (pointer) (void)hip.hipFree(pointer); }
};
}

int main(int argc, char** argv)
{
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        if (args.module.empty()) throw std::runtime_error("--module is required");
        HipApi hip(args.hip_root);
        hipDeviceProp_t properties{};
        hip.select_architecture(args.device, properties);
        hipModule_t module = nullptr;
        hip.check(hip.hipModuleLoad(&module, args.module.c_str()), "hipModuleLoad(WMMA)");
        struct ModuleCleanup {
            HipApi& hip; hipModule_t module;
            ~ModuleCleanup() { if (module) (void)hip.hipModuleUnload(module); }
        } cleanup{hip, module};
        hipFunction_t raw_kernel = nullptr, adapter_kernel = nullptr, upstream_kernel = nullptr, swap_kernel = nullptr;
        hip.check(hip.hipModuleGetFunction(&raw_kernel, module, "d4r_wmma_gfx1201"),
            "hipModuleGetFunction(WMMA)");
        hip.check(hip.hipModuleGetFunction(&adapter_kernel, module, "d4r_wmma_legacy_contract"),
            "hipModuleGetFunction(WMMA adapter)");
        hip.check(hip.hipModuleGetFunction(&upstream_kernel, module, "d4r_wmma_upstream_contract"),
            "hipModuleGetFunction(WMMA upstream)");
        hip.check(hip.hipModuleGetFunction(&swap_kernel, module, "d4r_wmma_swap_test"),
            "hipModuleGetFunction(WMMA swap)");
        constexpr size_t elements = 16u * 16u;
        std::array<uint16_t, elements> a{}, b{};
        std::array<float, elements> initial{}, output{};
        Allocation da(hip, sizeof(a)), db(hip, sizeof(b));
        Allocation dc(hip, sizeof(initial)), dout(hip, sizeof(output));
        void* swap_out = dout.pointer;
        void* swap_params[] = {&swap_out};
        hip.check(hip.hipModuleLaunchKernel(swap_kernel, 1, 1, 1, 32, 1, 1, 0,
            nullptr, swap_params, nullptr), "hipModuleLaunchKernel(WMMA swap)");
        hip.check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize(WMMA swap)");
        hip.check(hip.hipMemcpy(output.data(), dout.pointer, sizeof(output), hipMemcpyDeviceToHost), "hipMemcpy(WMMA swap)");
        for (unsigned lane = 0; lane < 32u; ++lane) {
            if (output[lane] != static_cast<float>(lane ^ 16u)) {
                std::fprintf(stderr, "FAIL WMMA swap lane=%u expected=%u actual=%g\n",
                    lane, lane ^ 16u, output[lane]);
                return 5;
            }
            if (output[32u + lane] != static_cast<float>(lane)) {
                std::fprintf(stderr, "FAIL WMMA lane-id lane=%u mbcnt=%g\n", lane, output[32u + lane]);
                return 5;
            }
        }
        float max_absolute_error = 0.0f;
        float max_relative_error = 0.0f;
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            const unsigned steps = 1u + (iteration & 1u);
            for (unsigned row = 0; row < 16u; ++row) {
                for (unsigned k = 0; k < 16u; ++k) {
                    a[row * 16u + k] = half_integer(static_cast<int>((row * 3u + k * 5u + iteration) % 7u) - 3);
                    b[row * 16u + k] = half_integer(static_cast<int>((row * 5u + k * 2u + iteration * 3u) % 7u) - 3);
                    initial[row * 16u + k] = static_cast<float>(static_cast<int>((row * 11u + k * 3u + iteration) % 17u) - 8) * 0.125f;
                }
            }
            hip.check(hip.hipMemcpy(da.pointer, a.data(), sizeof(a), hipMemcpyHostToDevice), "hipMemcpy(A)");
            hip.check(hip.hipMemcpy(db.pointer, b.data(), sizeof(b), hipMemcpyHostToDevice), "hipMemcpy(B)");
            hip.check(hip.hipMemcpy(dc.pointer, initial.data(), sizeof(initial), hipMemcpyHostToDevice), "hipMemcpy(C)");
            void* a_arg = da.pointer;
            void* b_arg = db.pointer;
            void* c_arg = dc.pointer;
            void* output_arg = dout.pointer;
            unsigned steps_arg = steps;
            void* params[] = {&a_arg, &b_arg, &c_arg, &output_arg, &steps_arg};
            const hipFunction_t kernels[] = {raw_kernel, adapter_kernel, upstream_kernel};
            const char* modes[] = {"raw", "legacy_adapter", "upstream_layout"};
            for (unsigned mode = 0; mode < 3u; ++mode) {
            output.fill(std::numeric_limits<float>::quiet_NaN());
            hip.check(hip.hipMemcpy(dout.pointer, output.data(), sizeof(output), hipMemcpyHostToDevice), "hipMemcpy(output sentinel)");
            hip.check(hip.hipModuleLaunchKernel(kernels[mode], 1, 1, 1, 32, 1, 1, 0, nullptr,
                params, nullptr), modes[mode]);
            hip.check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize(WMMA)");
            hip.check(hip.hipMemcpy(output.data(), dout.pointer, sizeof(output), hipMemcpyDeviceToHost), "hipMemcpy(output)");
            for (unsigned row = 0; row < 16u; ++row) {
                for (unsigned col = 0; col < 16u; ++col) {
                    int dot = 0;
                    for (unsigned k = 0; k < 16u; ++k) {
                        const int av = static_cast<int>((row * 3u + k * 5u + iteration) % 7u) - 3;
                        const int bv = static_cast<int>((col * 5u + k * 2u + iteration * 3u) % 7u) - 3;
                        dot += av * bv;
                    }
                    const float expected = initial[row * 16u + col] + static_cast<float>(static_cast<int>(steps) * dot);
                    const float got = output[row * 16u + col];
                    const float absolute = std::fabs(got - expected);
                    const float relative = absolute / std::max(1.0f, std::fabs(expected));
                    if (!std::isfinite(got) || absolute > 0.0001f) {
                        std::fprintf(stderr, "FAIL WMMA mode=%s iteration=%u row=%u col=%u steps=%u expected=%.9g actual=%.9g abs=%.9g\n",
                            modes[mode], iteration, row, col, steps, expected, got, absolute);
                        return 5;
                    }
                    max_absolute_error = std::max(max_absolute_error, absolute);
                    max_relative_error = std::max(max_relative_error, relative);
                }
            }
            }
        }
        std::printf("PASS GFX12_WMMA architecture=%s modes=raw,legacy_adapter,upstream_layout iterations=%u max_abs=%.9g max_rel=%.9g\n", d4r::diag::HipApi::target_arch(),
            args.iterations, max_absolute_error, max_relative_error);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL GFX12_WMMA %s\n", error.what());
        loaded_modules();
        return 4;
    }
}
