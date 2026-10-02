#include "hip_api.h"

// Isolate Windows HIP stream lifetime from D3D12 and external-resource imports.
int main(int argc, char** argv)
{
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        HipApi hip(args.hip_root);
        hipDeviceProp_t props{};
        hip.select_architecture(args.device, props);
        hipModule_t module = nullptr;
        hipFunction_t kernel = nullptr;
        void* output = nullptr;
        struct ResourceCleanup {
            HipApi& hip; hipModule_t& module; void*& output;
            ~ResourceCleanup() {
                if (output) (void)hip.hipFree(output);
                if (module) (void)hip.hipModuleUnload(module);
            }
        } resources{hip, module, output};
        if (!args.module.empty()) {
            hip.check(hip.hipModuleLoad(&module, args.module.c_str()), "hipModuleLoad");
            hip.check(hip.hipModuleGetFunction(&kernel, module, "d4r_probe_pattern"), "hipModuleGetFunction");
            hip.check(hip.hipMalloc(&output, 128 * sizeof(uint32_t)), "hipMalloc");
        }
        DWORD baseline = 0, count = 0;
        for (unsigned i = 0; i < args.iterations; ++i) {
            hipStream_t stream = nullptr;
            hip.check(hip.hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
            struct StreamCleanup {
                HipApi& hip; hipStream_t& stream;
                ~StreamCleanup() { if (stream) { (void)hip.hipStreamSynchronize(stream); (void)hip.hipStreamDestroy(stream); } }
            } cleanup{hip, stream};
            if (kernel) {
                uint32_t n = 128, seed = i;
                void* parameters[] = {&output, &n, &seed};
                hip.check(hip.hipModuleLaunchKernel(kernel, 1, 1, 1, 128, 1, 1, 0,
                    stream, parameters, nullptr), "hipModuleLaunchKernel");
            }
            hip.check(hip.hipStreamSynchronize(stream), "hipStreamSynchronize");
            hip.check(hip.hipStreamDestroy(stream), "hipStreamDestroy");
            stream = nullptr;
            if (!GetProcessHandleCount(GetCurrentProcess(), &count))
                throw std::runtime_error("GetProcessHandleCount failed");
            if (i == 1 || (args.iterations == 1 && i == 0)) baseline = count;
            std::printf("STREAM_CLEANUP iteration=%u handles=%lu baseline=%lu\n", i, count, baseline);
        }
        if (output) { hip.check(hip.hipFree(output), "hipFree"); output = nullptr; }
        if (module) { hip.check(hip.hipModuleUnload(module), "hipModuleUnload"); module = nullptr; }
        if (count > baseline + 4) throw std::runtime_error("HIP stream lifecycle handle growth");
        std::printf("PASS HIP_STREAM iterations=%u handles=%lu\n", args.iterations, count);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL HIP_STREAM %s\n", e.what());
        loaded_modules();
        return 4;
    }
}
