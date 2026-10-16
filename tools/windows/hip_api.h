#pragma once
#include "diagnostic.h"
#include "gpu_target.h"
#include <hip/hip_runtime_api.h>
#include <cstdlib>
#include <string>

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
    // The package's backend sets D4R_GPU_ARCH to the GPU type of the game's adapter before it loads this DLL, so one
    // build serves every supported type (a compile-time-only target locked RDNA3 / gfx1201 cards out: "No selected
    // gfx1200 device"). Unknown or missing values fall back to the build's own target.
    static const char* target_arch() {
        static const std::string arch = [] {
            const char* e = std::getenv("D4R_GPU_ARCH");
            const std::string v = e ? gpu_architecture(e) : std::string(); // "gfx1101:xnack-" -> "gfx1101"
            return supported_gpu_target(v) ? v : std::string(D4R_TARGET_ARCH);
        }();
        return arch.c_str();
    }
    static int requested_device(int requested) {
        if (requested >= 0) return requested;
        const char* e = std::getenv("D4R_HIP_DEVICE");
        return e && *e ? std::atoi(e) : -1;
    }
    int select_architecture(int requested, hipDeviceProp_t& selected, const void* required_luid = nullptr,
        bool allow_no_match = false) const {
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
        requested = requested_device(requested);
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
        if (found < 0 && allow_no_match) return -1; // Capability enumeration can include Intel/WARP adapters.
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
