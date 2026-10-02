#pragma once
#include "diagnostic.h"
#include "gpu_target.h"
#include <hip/hip_runtime_api.h>

namespace d4r::diag {
struct HipApi {
    bool verbose = true; // Probes log every call; the game can suppress successful API calls.
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
    D4R_HIP_FUNCTION(hipModuleGetGlobal);
    D4R_HIP_FUNCTION(hipModuleLaunchKernel);
    D4R_HIP_FUNCTION(hipDeviceSynchronize);
    D4R_HIP_FUNCTION(hipStreamCreateWithFlags);
    D4R_HIP_FUNCTION(hipStreamSynchronize);
    D4R_HIP_FUNCTION(hipStreamDestroy);
#undef D4R_HIP_FUNCTION
    explicit HipApi(const std::string& root) :
        search(std::filesystem::path(wide(root)) / L"bin"),
        library(std::filesystem::path(wide(root)) / L"bin" / L"amdhip64_7.dll") {}
    void check(hipError_t result, const char* call) const {
        if (verbose || result != hipSuccess) std::printf("HIP %s -> %d (%s)\n", call, static_cast<int>(result), hipGetErrorName(result));
        if (result != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(result));
    }
    static const char* target_arch() { return D4R_TARGET_ARCH; }
    int select_architecture(int requested, hipDeviceProp_t& selected, const void* required_luid = nullptr) const {
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
            if (gpu_architecture(arch) == target_arch() && (requested < 0 || requested == i) && found < 0 &&
                (!required_luid || std::memcmp(props.luid, required_luid, sizeof(props.luid)) == 0)) {
                found = i;
                selected = props;
            }
        }
        if (found < 0) throw std::runtime_error(std::string("No selected ") + target_arch() +
            " device for this adapter; use the package matching the actual HIP architecture (see GPU log)");
        require_gpu_target(selected.gcnArchName, target_arch());
        check(hipSetDevice(found), "hipSetDevice");
        std::printf("SELECTED HIP ordinal=%d architecture=%s validation=%s\n", found, target_arch(),
            std::string(target_arch()) == "gfx1201" ? "previously_tested_target" : "hardware_unverified_target");
        return found;
    }
};
}
