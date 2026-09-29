#include "hip_api.h"
#include <algorithm>

#include "cuda_api.h"

static constexpr char ptx[] = R"PTX(
.version 7.0
.target sm_75
.address_size 64
.visible .entry d4r_ptx_pattern(
    .param .u64 output, .param .u32 count, .param .u32 seed)
{
    .reg .pred %p;
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd0, [output];
    ld.param.u32 %r0, [count];
    ld.param.u32 %r1, [seed];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p, %r5, %r0;
    @%p bra done;
    mad.lo.u32 %r6, %r5, 1664525, %r1;
    xor.b32 %r7, %r6, 0xa5a55a5a;
    mul.wide.u32 %rd1, %r5, 4;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r7;
done:
    ret;
}
)PTX";

int main(int argc, char** argv)
{
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        if (args.cuda_dll.empty()) throw std::runtime_error("--cuda-dll is required (absolute ZLUDA nvcuda.dll path)");
        if (args.context != "primary" && args.context != "created") throw std::runtime_error("context must be primary or created");
        // Require the real HIP architecture, then match the CUDA device by PCI identity.
        HipApi hip(args.hip_root);
        hipDeviceProp_t props{};
        hip.select_gfx1201(args.device, props);
        CudaApi api(args.cuda_dll);
        loaded_modules();
        std::printf("STAGE cuInit\n");
        api.check(api.cuInit(0), "cuInit");
        int driver = 0, count = 0;
        api.check(api.cuDriverGetVersion(&driver), "cuDriverGetVersion");
        api.check(api.cuDeviceGetCount(&count), "cuDeviceGetCount");
        std::printf("CUDA driver_api=%d devices=%d\n", driver, count);
        CUdevice device = -1;
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            CUdevice candidate;
            api.check(api.cuDeviceGet(&candidate, ordinal), "cuDeviceGet");
            char name[256]{};
            int bus = -1, pci_device = -1, domain = -1;
            api.check(api.cuDeviceGetName(name, sizeof(name), candidate), "cuDeviceGetName");
            api.check(api.cuDeviceGetAttribute(&bus, 33, candidate), "cuDeviceGetAttribute(PCI_BUS_ID)");
            api.check(api.cuDeviceGetAttribute(&pci_device, 34, candidate), "cuDeviceGetAttribute(PCI_DEVICE_ID)");
            api.check(api.cuDeviceGetAttribute(&domain, 50, candidate), "cuDeviceGetAttribute(PCI_DOMAIN_ID)");
            std::printf("CUDA GPU ordinal=%d name=%s pci=%04x:%02x:%02x\n", ordinal, name, domain, bus, pci_device);
            if (bus == props.pciBusID && pci_device == props.pciDeviceID && domain == props.pciDomainID) device = candidate;
        }
        if (device < 0) throw std::runtime_error("CUDA/HIP PCI identity mismatch");
        CUcontext context = nullptr;
        const bool primary = args.context == "primary";
        if (primary) {
            api.check(api.cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
        } else api.check(api.cuCtxCreate_v2(&context, 0, device), "cuCtxCreate_v2");
        struct ContextCleanup {
            CudaApi& api; CUcontext context; CUdevice device; bool primary;
            ~ContextCleanup() { if (context) { api.cuCtxSetCurrent(nullptr);
                if (primary) api.cuDevicePrimaryCtxRelease_v2(device); else api.cuCtxDestroy_v2(context); } }
        } cleanup{api, context, device, primary};
        api.check(api.cuCtxSetCurrent(context), "cuCtxSetCurrent");
        CUmodule module = nullptr;
        std::printf("STAGE cuModuleLoadData (PTX JIT)\n");
        api.check(api.cuModuleLoadData(&module, ptx), "cuModuleLoadData");
        struct ModuleCleanup { CudaApi& api; CUmodule m; ~ModuleCleanup() { if (m) api.cuModuleUnload(m); } } mc{api, module};
        loaded_modules();
        CUfunction function = nullptr;
        api.check(api.cuModuleGetFunction(&function, module, "d4r_ptx_pattern"), "cuModuleGetFunction");
        constexpr uint32_t nmax = 4099;
        std::vector<uint32_t> result(nmax + 61, 0xdeadbeefu);
        const size_t bytes = result.size() * sizeof(uint32_t);
        CUdeviceptr pointer = 0;
        api.check(api.cuMemAlloc_v2(&pointer, bytes), "cuMemAlloc_v2");
        struct MemCleanup { CudaApi& api; CUdeviceptr p; ~MemCleanup() { if (p) api.cuMemFree_v2(p); } } ac{api, pointer};
        for (unsigned i = 0; i < args.iterations; ++i) {
            uint32_t n = nmax - i % 129u, seed = 0x12345678u + i * 31u;
            std::fill(result.begin(), result.end(), 0xdeadbeefu);
            api.check(api.cuMemcpyHtoD_v2(pointer, result.data(), bytes), "cuMemcpyHtoD_v2(sentinel)");
            void* parameters[] = {&pointer, &n, &seed};
            api.check(api.cuLaunchKernel(function, (n + 127u) / 128u, 1, 1, 128, 1, 1,
                0, nullptr, parameters, nullptr), "cuLaunchKernel");
            api.check(api.cuCtxSynchronize(), "cuCtxSynchronize");
            api.check(api.cuMemcpyDtoH_v2(result.data(), pointer, bytes), "cuMemcpyDtoH_v2");
            if (!verify(result, n, seed)) return 5;
        }
        api.check(api.cuMemFree_v2(pointer), "cuMemFree_v2"); ac.p = 0;
        api.check(api.cuModuleUnload(module), "cuModuleUnload"); mc.m = nullptr;
        api.check(api.cuCtxSetCurrent(nullptr), "cuCtxSetCurrent(NULL)");
        if (primary) api.check(api.cuDevicePrimaryCtxRelease_v2(device), "cuDevicePrimaryCtxRelease_v2");
        else api.check(api.cuCtxDestroy_v2(context), "cuCtxDestroy_v2");
        cleanup.context = nullptr;
        std::printf("PASS CUDA architecture=gfx1201 context=%s iterations=%u guard_verified=1\n",
            args.context.c_str(), args.iterations);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL CUDA %s\n", e.what());
        loaded_modules();
        return 4;
    }
}
