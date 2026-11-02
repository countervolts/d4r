#pragma once
#include "diagnostic.h"
#include "gpu_selection.h"
#include <hip/hip_runtime_api.h>
#include <cstdlib>
#include <string>

namespace d4r::diag {
// The HIP runtime this process uses: the release's own (ROOT\bin\amdhip64_7.dll) when it initializes and sees a GPU,
// otherwise the AMD driver's (System32), e.g. when the bundled runtime is newer than the installed driver (issue 58).
// D4R_HIP_RUNTIME=bundled|driver skips the probe; the choice is published there too, so the loader's choice holds for
// the shim it loads. A runtime that failed is unloaded before the next is tried, so name-based loads (ZLUDA's) then
// resolve to the chosen one, which stays loaded for the process.
inline std::filesystem::path hip_runtime_path(const std::string& root) {
    static const std::filesystem::path chosen = [&] {
        const auto bundled = std::filesystem::path(wide(root)) / L"bin" / L"amdhip64_7.dll";
        wchar_t system[MAX_PATH]{};
        GetSystemDirectoryW(system, MAX_PATH);
        const auto driver = std::filesystem::path(system) / L"amdhip64_7.dll";
        const char* forced = std::getenv("D4R_HIP_RUNTIME");
        if (forced && std::string(forced) == "bundled") return bundled;
        if (forced && std::string(forced) == "driver") return driver;
        auto usable = [](const std::filesystem::path& path) {
            if (!std::filesystem::exists(path)) return false;
            HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
            if (!module) return false;
            auto init = reinterpret_cast<hipError_t (*)(unsigned)>(reinterpret_cast<void*>(GetProcAddress(module, "hipInit")));
            auto count = reinterpret_cast<hipError_t (*)(int*)>(reinterpret_cast<void*>(GetProcAddress(module, "hipGetDeviceCount")));
            int devices = 0;
            const bool ok = init && count && init(0) == hipSuccess && count(&devices) == hipSuccess && devices > 0;
            if (!ok) { FreeLibrary(module); return false; }
            HMODULE pinned = nullptr;   // stays loaded: HIP is never unloaded once initialized
            GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                reinterpret_cast<LPCWSTR>(reinterpret_cast<void*>(init)), &pinned);
            return true;
        };
        const bool useDriver = !usable(bundled) && usable(driver);
        if (useDriver) std::fprintf(stderr, "d4r: the bundled HIP runtime found no GPU; using the driver's %s\n", utf8(driver.c_str()).c_str());
        SetEnvironmentVariableA("D4R_HIP_RUNTIME", useDriver ? "driver" : "bundled");
        _putenv_s("D4R_HIP_RUNTIME", useDriver ? "driver" : "bundled");
        return useDriver ? driver : bundled;
    }();
    return chosen;
}
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
        library(hip_runtime_path(root)) {}
    void check(hipError_t result, const char* call) const {
        if (verbose || result != hipSuccess) std::printf("HIP %s -> %d (%s)\n", call, static_cast<int>(result), hipGetErrorName(result));
        if (result != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(result));
    }
    // The architecture this process runs for: D4R_GPU_ARCH when it names a supported target, otherwise the build's
    // own D4R_TARGET_ARCH. The loader and the shim publish the adapter's real architecture there before the
    // NVAPI/NGX components start (publish_selected_gpu in gpu_environment.h), so every component agrees. A value that
    // is set but unsupported is reported and ignored; select_architecture still refuses any device that is not of
    // the architecture in use.
    static const char* target_arch() {
        static const std::string arch = [] {
            const auto resolved = resolve_target_arch(std::getenv("D4R_GPU_ARCH"), D4R_TARGET_ARCH);
            if (!resolved.warning.empty()) std::fprintf(stderr, "d4r: %s\n", resolved.warning.c_str());
            return resolved.architecture;
        }();
        return arch.c_str();
    }
    static int requested_device(int requested) {
        if (requested >= 0) return requested;
        const char* e = std::getenv("D4R_HIP_DEVICE");
        return e && *e ? std::atoi(e) : -1;
    }
    // The adapter the loader or shim selected (D4R_GPU_LUID), for components that start before any D3D12 device is
    // at hand. Absent means "any device of the target architecture"; a malformed value is an error, never ignored.
    static bool configured_luid(AdapterLuid& luid) {
        const char* text = std::getenv("D4R_GPU_LUID");
        if (!text || !*text) return false;
        if (!parse_luid(text, luid)) throw std::runtime_error(std::string("D4R_GPU_LUID='") + text + "' is not HHHHHHHH:LLLLLLLL");
        return true;
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
        std::vector<GpuCandidate> devices;
        requested = requested_device(requested);
        AdapterLuid wanted{};
        const AdapterLuid* luid = nullptr;
        if (required_luid) { std::memcpy(wanted.data(), required_luid, wanted.size()); luid = &wanted; }
        else if (configured_luid(wanted)) luid = &wanted;
        std::vector<hipDeviceProp_t> properties;
        for (int i = 0; i < count; ++i) {
            hipDeviceProp_t props{};
            check(hipGetDeviceProperties(&props, i), "hipGetDevicePropertiesR0600");
            std::printf("GPU ordinal=%d name=%s gcnArchName=%s wave=%d vram=%zu pci=%04x:%02x:%02x\n",
                i, props.name, props.gcnArchName, props.warpSize, props.totalGlobalMem,
                props.pciDomainID, props.pciBusID, props.pciDeviceID);
            GpuCandidate candidate{props.name, gpu_architecture(props.gcnArchName), {}};
            std::memcpy(candidate.luid.data(), props.luid, candidate.luid.size());
            devices.push_back(candidate);
            properties.push_back(props);
        }
        const int found = find_gpu_device(devices, target_arch(), requested, luid);
        if (found < 0 && allow_no_match) return -1; // Capability enumeration can include Intel/WARP adapters.
        if (found < 0) throw std::runtime_error(no_device_message(devices, target_arch(), luid));
        selected = properties[found];
        require_gpu_target(selected.gcnArchName, target_arch());
        check(hipSetDevice(found), "hipSetDevice");
        std::printf("SELECTED HIP ordinal=%d architecture=%s validation=%s\n", found, target_arch(),
            std::string(target_arch()) == "gfx1201" ? "previously_tested_target" : "hardware_unverified_target");
        return found;
    }
};
}
