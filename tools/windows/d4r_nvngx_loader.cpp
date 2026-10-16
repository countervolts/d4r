// Drag-in releases ship one d4r folder for every supported GPU. This small
// _nvngx.dll is what OptiScaler loads: on first use it reads the real HIP
// architecture of the game's adapter (matched by LUID) and loads that
// target's shim, d4r\_nvngx_<arch>.dll, from the same folder. The per-target
// shim still enforces its own compiled target, so a mismatch cannot run.
// Keeping every file in one folder preserves NGX's search for
// nvngx_dlss.dll beside the calling module.
#include "d3d12_external.h"
#include "ngx_public_abi.h"
#include <dxgi.h>
#include <fstream>
#include <mutex>

namespace {
constexpr unsigned failure = 0xBAD00002, unsupported = 4;
std::mutex selectMutex;
HMODULE shim = nullptr;
bool selectionFailed = false;

std::filesystem::path module_directory() {
    HMODULE module = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&module_directory), &module);
    wchar_t path[32768]{}; GetModuleFileNameW(module, path, 32768);
    return std::filesystem::path(path).parent_path();
}
// The shim never ran, so the loader writes the player's log itself.
// Fresh at each launch, then appended: OptiScaler can query several adapters.
void report(const std::string& message) {
    static bool fresh = true;
    std::ofstream(module_directory() / L"d4r_nvngx.log", fresh ? std::ios::trunc : std::ios::app)
        << "d4r: DLSS disabled: " << message << "\n";
    fresh = false;
    std::fprintf(stderr, "D4R_WINDOWS_FAILURE %s\n", message.c_str());
}
std::string supported_list(const std::filesystem::path& dir) {
    std::string list;
    for (const auto& target : d4r::diag::gpu_targets)
        if (std::filesystem::exists(dir / (L"_nvngx_" + d4r::diag::wide(target.architecture) + L".dll")))
            list += (list.empty() ? "" : ", ") + std::string(target.architecture);
    return list.empty() ? "none (the d4r folder is incomplete; extract the ZIP again)" : list;
}

// Loads the matching shim once. A null LUID (calls made before NGX init)
// accepts the only HIP device with a shim in this folder.
HMODULE select(const LUID* luid) {
    std::lock_guard<std::mutex> lock(selectMutex);
    if (shim || selectionFailed) return shim;
    const auto dir = module_directory();
    try {
        // Spoofed architectures invalidate the target guarantee; see hip_api.h.
        for (const char* name : {"HSA_OVERRIDE_GFX_VERSION", "HSA_OVERRIDE_GFX_VERSION_0"})
            if (std::getenv(name)) throw std::runtime_error(std::string("unset ") + name + "; d4r does not support architecture overrides");
        d4r::diag::HipApi hip((dir / L"hip").string());
        if (hip.hipInit(0) != hipSuccess) throw std::runtime_error("the AMD HIP runtime found no usable GPU; update the AMD graphics driver");
        int count = 0; (void)hip.hipGetDeviceCount(&count);
        std::string found, detected;
        for (int i = 0; i < count; ++i) {
            hipDeviceProp_t props{};
            if (hip.hipGetDeviceProperties(&props, i) != hipSuccess) continue;
            const auto arch = d4r::diag::gpu_architecture(props.gcnArchName);
            detected += (detected.empty() ? "" : ", ") + arch + " (" + props.name + ")";
            if (luid && std::memcmp(props.luid, luid, sizeof(*luid))) continue;
            if (!luid && !std::filesystem::exists(dir / (L"_nvngx_" + d4r::diag::wide(arch) + L".dll"))) continue;
            if (!found.empty() && found != arch) throw std::runtime_error("several GPUs with different architectures; start the game once DLSS selects one");
            found = arch;
        }
        if (found.empty()) throw std::runtime_error("no supported AMD GPU for this game's adapter. Detected: " +
            (detected.empty() ? std::string("none") : detected) + ". This release supports: " + supported_list(dir) + ".");
        const auto path = dir / (L"_nvngx_" + d4r::diag::wide(found) + L".dll");
        if (!std::filesystem::exists(path))
            throw std::runtime_error("this release has no build for your GPU (" + found + "). It supports: " + supported_list(dir) + ".");
        shim = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!shim) {
            selectionFailed = true; // A broken install will not fix itself during this session.
            throw std::runtime_error("cannot load " + d4r::diag::utf8(path.c_str()) + " (Win32 error " + std::to_string(GetLastError()) + ")");
        }
    } catch (const std::exception& error) {
        // An adapter without a matching GPU (an iGPU queried first) must not
        // prevent selecting the discrete GPU on a later call.
        report(error.what());
    }
    return shim;
}
template <typename Fn> Fn resolve(const char* name, const LUID* luid = nullptr) {
    HMODULE module = select(luid);
    return module ? reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(module, name))) : nullptr;
}
LUID device_luid(ID3D12Device* device) { return d4r::win::adapter_luid(device); }
}

#define API extern "C" __declspec(dllexport)
// Discovery marker for OptiScaler: no GPU or runtime initialization.
API unsigned d4r_WindowsBackendVersion() { return 1; }

API unsigned NVSDK_NGX_D3D12_Init_Ext(unsigned long long app, const wchar_t* data, ID3D12Device* device, unsigned sdk, const void* info) {
    if (!device) return 0xBAD00005;
    const LUID luid = device_luid(device);
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_Init_Ext)>("NVSDK_NGX_D3D12_Init_Ext", &luid);
    return fn ? fn(app, data, device, sdk, info) : failure;
}
API unsigned NVSDK_NGX_D3D12_Init(unsigned long long app, const wchar_t* data, ID3D12Device* device, const void* info, unsigned sdk) {
    if (!device) return 0xBAD00005;
    const LUID luid = device_luid(device);
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_Init)>("NVSDK_NGX_D3D12_Init", &luid);
    return fn ? fn(app, data, device, info, sdk) : failure;
}
API unsigned NVSDK_NGX_D3D12_Init_ProjectID(const char* project, int engine, const char* version, const wchar_t* data, ID3D12Device* device, unsigned sdk, const void* info) {
    if (!device) return 0xBAD00005;
    const LUID luid = device_luid(device);
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_Init_ProjectID)>("NVSDK_NGX_D3D12_Init_ProjectID", &luid);
    return fn ? fn(project, engine, version, data, device, sdk, info) : failure;
}
API unsigned NVSDK_NGX_D3D12_GetFeatureRequirements(IDXGIAdapter* adapter, const d4r::ngx::FeatureDiscoveryInfo* info,
    d4r::ngx::FeatureRequirement* requirements) {
    if (!adapter || !info || !requirements) return 0xBAD00005;
    DXGI_ADAPTER_DESC desc{};
    if (FAILED(adapter->GetDesc(&desc))) return failure;
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_GetFeatureRequirements)>("NVSDK_NGX_D3D12_GetFeatureRequirements", &desc.AdapterLuid);
    if (fn) return fn(adapter, info, requirements);
    // Same answer as the shim for an adapter without a matching HIP device.
    *requirements = {}; requirements->FeatureSupported = unsupported;
    std::strcpy(requirements->MinOSVersion, "10.0.22000.0");
    return 1;
}
API unsigned NVSDK_NGX_D3D12_AllocateParameters(void** p) {
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_AllocateParameters)>("NVSDK_NGX_D3D12_AllocateParameters");
    return fn ? fn(p) : failure;
}
API unsigned NVSDK_NGX_D3D12_GetCapabilityParameters(void** p) {
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_GetCapabilityParameters)>("NVSDK_NGX_D3D12_GetCapabilityParameters");
    return fn ? fn(p) : failure;
}
API unsigned NVSDK_NGX_D3D12_GetParameters(void** p) {
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_GetParameters)>("NVSDK_NGX_D3D12_GetParameters");
    return fn ? fn(p) : failure;
}
API unsigned NVSDK_NGX_D3D12_DestroyParameters(void* p) {
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_DestroyParameters)>("NVSDK_NGX_D3D12_DestroyParameters");
    return fn ? fn(p) : failure;
}
API unsigned NVSDK_NGX_D3D12_GetScratchBufferSize(unsigned feature, const void* p, size_t* size) {
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_GetScratchBufferSize)>("NVSDK_NGX_D3D12_GetScratchBufferSize");
    return fn ? fn(feature, p, size) : failure;
}
API unsigned NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList* list, unsigned feature, void* p, void** out) {
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_CreateFeature)>("NVSDK_NGX_D3D12_CreateFeature");
    return fn ? fn(list, feature, p, out) : failure;
}
API unsigned d4r_D3D12_EvaluateAtBoundary(ID3D12CommandQueue* queue, void* handle, void* p) {
    auto fn = resolve<decltype(&d4r_D3D12_EvaluateAtBoundary)>("d4r_D3D12_EvaluateAtBoundary");
    return fn ? fn(queue, handle, p) : failure;
}
API unsigned NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList* list, const void* handle, void* p, void* callback) {
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_EvaluateFeature)>("NVSDK_NGX_D3D12_EvaluateFeature");
    return fn ? fn(list, handle, p, callback) : failure;
}
API unsigned NVSDK_NGX_D3D12_EvaluateFeature_C(ID3D12GraphicsCommandList* list, const void* handle, void* p, void* callback) {
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_EvaluateFeature_C)>("NVSDK_NGX_D3D12_EvaluateFeature_C");
    return fn ? fn(list, handle, p, callback) : failure;
}
API unsigned NVSDK_NGX_D3D12_ReleaseFeature(void* handle) {
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_ReleaseFeature)>("NVSDK_NGX_D3D12_ReleaseFeature");
    return fn ? fn(handle) : failure;
}
API unsigned NVSDK_NGX_D3D12_Shutdown() {
    // Nothing to shut down if no shim was ever selected.
    { std::lock_guard<std::mutex> lock(selectMutex); if (!shim) return 1; }
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_Shutdown)>("NVSDK_NGX_D3D12_Shutdown");
    return fn ? fn() : failure;
}
API unsigned NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* device) {
    { std::lock_guard<std::mutex> lock(selectMutex); if (!shim) return 1; }
    auto fn = resolve<decltype(&NVSDK_NGX_D3D12_Shutdown1)>("NVSDK_NGX_D3D12_Shutdown1");
    return fn ? fn(device) : failure;
}
