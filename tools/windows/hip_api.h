#pragma once
#include "diagnostic.h"
#include <hip/hip_runtime_api.h>

namespace d4r::diag {
struct HipApi {
    SearchDirectory search;
    Library library;
#define D4R_HIP_FUNCTION(name) decltype(&::name) name = library.symbol<decltype(&::name)>(#name)
    D4R_HIP_FUNCTION(hipInit);
    D4R_HIP_FUNCTION(hipGetDeviceCount);
    // hipGetDeviceProperties is an SDK macro; the ABI-versioned export is intentional.
    decltype(&::hipGetDeviceProperties) hipGetDeviceProperties =
        library.symbol<decltype(&::hipGetDeviceProperties)>("hipGetDevicePropertiesR0600");
    D4R_HIP_FUNCTION(hipSetDevice);
    D4R_HIP_FUNCTION(hipRuntimeGetVersion);
    D4R_HIP_FUNCTION(hipDriverGetVersion);
    D4R_HIP_FUNCTION(hipGetErrorName);
    D4R_HIP_FUNCTION(hipGetErrorString);
    using MallocFn = hipError_t (*)(void**, size_t);
    MallocFn hipMalloc = library.symbol<MallocFn>("hipMalloc");
    D4R_HIP_FUNCTION(hipFree);
    D4R_HIP_FUNCTION(hipMemcpy);
    D4R_HIP_FUNCTION(hipMemGetInfo);
    D4R_HIP_FUNCTION(hipModuleLoad);
    D4R_HIP_FUNCTION(hipModuleUnload);
    D4R_HIP_FUNCTION(hipModuleGetFunction);
    D4R_HIP_FUNCTION(hipModuleLaunchKernel);
    D4R_HIP_FUNCTION(hipDeviceSynchronize);
#undef D4R_HIP_FUNCTION
    explicit HipApi(const std::string& root) :
        search(std::filesystem::path(wide(root)) / L"bin"),
        library(std::filesystem::path(wide(root)) / L"bin" / L"amdhip64_7.dll") {}
    void check(hipError_t result, const char* call) const {
        std::printf("HIP %s -> %d (%s)\n", call, static_cast<int>(result), hipGetErrorName(result));
        if (result != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(result));
    }
    int select_gfx1201(int requested, hipDeviceProp_t& selected) const {
        // Architecture spoofing invalidates a gfx12 correctness test.
        for (const char* name : {"HSA_OVERRIDE_GFX_VERSION", "HSA_OVERRIDE_GFX_VERSION_0"})
            if (std::getenv(name)) throw std::runtime_error(std::string("Unset architecture override ") + name);
        check(hipInit(0), "hipInit");
        int version = 0, driver = 0, count = 0;
        check(hipRuntimeGetVersion(&version), "hipRuntimeGetVersion");
        check(hipDriverGetVersion(&driver), "hipDriverGetVersion");
        std::printf("HIP runtime=%d driver_api=%d headers=%d.%d.%d sizeof(properties)=%zu\n",
            version, driver, HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH, sizeof(hipDeviceProp_t));
        check(hipGetDeviceCount(&count), "hipGetDeviceCount");
        int found = -1;
        for (int i = 0; i < count; ++i) {
            hipDeviceProp_t props{};
            check(hipGetDeviceProperties(&props, i), "hipGetDevicePropertiesR0600");
            std::printf("GPU ordinal=%d name=%s gcnArchName=%s wave=%d vram=%zu pci=%04x:%02x:%02x\n",
                i, props.name, props.gcnArchName, props.warpSize, props.totalGlobalMem,
                props.pciDomainID, props.pciBusID, props.pciDeviceID);
            const std::string arch = props.gcnArchName;
            if (arch.substr(0, arch.find(':')) == "gfx1201" && (requested < 0 || requested == i) && found < 0) {
                found = i;
                selected = props;
            }
        }
        if (found < 0) throw std::runtime_error("No selected gfx1201 device; never substitute gfx110x");
        check(hipSetDevice(found), "hipSetDevice");
        std::printf("SELECTED HIP ordinal=%d architecture=gfx1201\n", found);
        return found;
    }
};
}
