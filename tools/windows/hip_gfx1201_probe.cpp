#include "hip_api.h"

int main(int argc, char** argv)
{
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        if (args.module.empty()) throw std::runtime_error("--module is required");
        HipApi api(args.hip_root);
        hipDeviceProp_t props{};
        api.select_architecture(args.device, props);
        loaded_modules();
        size_t free = 0, total = 0;
        api.check(api.hipMemGetInfo(&free, &total), "hipMemGetInfo");
        std::printf("VRAM free=%zu total=%zu\n", free, total);
        hipModule_t module = nullptr;
        api.check(api.hipModuleLoad(&module, args.module.c_str()), "hipModuleLoad(target)");
        struct ModuleCleanup { HipApi& api; hipModule_t m; ~ModuleCleanup() { if (m) (void)api.hipModuleUnload(m); } } cleanup{api, module};
        hipFunction_t kernel = nullptr;
        api.check(api.hipModuleGetFunction(&kernel, module, "d4r_probe_pattern"), "hipModuleGetFunction");
        constexpr uint32_t count = 4099, guard = 61;
        std::vector<uint32_t> result(count + guard, 0xdeadbeefu);
        const size_t bytes = result.size() * sizeof(uint32_t);
        void* allocation = nullptr;
        api.check(api.hipMalloc(&allocation, bytes), "hipMalloc(VRAM)");
        struct AllocCleanup { HipApi& api; void* p; ~AllocCleanup() { if (p) (void)api.hipFree(p); } } allocation_cleanup{api, allocation};
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            uint32_t n = count - iteration % 129u, seed = 0x12345678u + iteration * 31u;
            std::fill(result.begin(), result.end(), 0xdeadbeefu);
            api.check(api.hipMemcpy(allocation, result.data(), bytes, hipMemcpyHostToDevice), "hipMemcpy(HtoD sentinel)");
            void* parameters[] = {&allocation, &n, &seed};
            api.check(api.hipModuleLaunchKernel(kernel, (n + 127u) / 128u, 1, 1, 128, 1, 1,
                0, nullptr, parameters, nullptr), "hipModuleLaunchKernel");
            api.check(api.hipDeviceSynchronize(), "hipDeviceSynchronize");
            api.check(api.hipMemcpy(result.data(), allocation, bytes, hipMemcpyDeviceToHost), "hipMemcpy(DtoH)");
            if (!verify(result, n, seed)) return 5;
        }
        // Explicitly validate teardown too; disable RAII cleanup after success.
        api.check(api.hipFree(allocation), "hipFree");
        allocation_cleanup.p = nullptr;
        api.check(api.hipModuleUnload(module), "hipModuleUnload");
        cleanup.m = nullptr;
        std::printf("PASS HIP architecture=%s iterations=%u guard_verified=1\n", d4r::diag::HipApi::target_arch(), args.iterations);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL HIP %s\n", e.what());
        loaded_modules();
        return 4;
    }
}
