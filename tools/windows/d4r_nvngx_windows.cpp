// Native Windows backend. The Linux/Wine implementation remains separate.
#include "ngx_windows_runtime.h"
#include <dxgi.h>
#include <unordered_map>
#include <functional>

namespace {
std::mutex apiMutex;
std::shared_ptr<d4r::win::Runtime> runtime;
struct Handle { unsigned Id; };
unsigned nextId = 1;
std::unordered_map<Handle*, std::shared_ptr<d4r::win::Feature>> features;
std::unordered_map<void*, bool> parameters;
constexpr unsigned failure = 0xBAD00002, invalid = 0xBAD00005;
unsigned call(const std::function<unsigned()>& fn) noexcept {
    static std::once_flag diagnosticOutput;
    std::call_once(diagnosticOutput, [] {
        if (std::getenv("D4R_DIAG_DIR")) {
            std::setvbuf(stdout, nullptr, _IONBF, 0);
            std::setvbuf(stderr, nullptr, _IONBF, 0);
        }
    });
    try { return fn(); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "D4R_WINDOWS_FAILURE %s\n", error.what()); return failure;
    }
}
unsigned allocate(void** out, bool caps) {
    if (!runtime || !out) return invalid;
    void* value = d4r_ngx_parameters_create();
    if (!value) return failure;
    *out = nullptr;
    struct LocalParameters {
        void* p;
        ~LocalParameters() { if (p) d4r_ngx_parameters_destroy(p); }
    } owned{value};
    if (caps) {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        runtime->current(); void* official = nullptr;
        d4r::win::ngx_check(runtime->capabilities(&official), "CUDA capabilities");
        struct Cleanup { d4r::win::Runtime& rt; void* p; ~Cleanup() { (void)rt.destroy(p); } } cleanup{*runtime, official};
        for (const char* name : {"SuperSampling.Available", "SuperSampling.NeedsUpdatedDriver", "SuperSampling.FeatureInitResult"})
            d4r_ngx_set_int(value, name, d4r::ngx::int_value(official, name));
        for (const char* name : {"SuperSampling.MinDriverVersionMajor", "SuperSampling.MinDriverVersionMinor"})
            d4r_ngx_set_uint(value, name, d4r::ngx::uint_value(official, name));
        for (const char* name : {"DLSSOptimalSettingsCallback", "DLSSGetStatsCallback"}) {
            void* callback = nullptr;
            if (d4r_ngx_get_void(official, name, &callback) == 1) d4r_ngx_set_void(value, name, callback);
        }
    }
    parameters.emplace(value, true); *out = value; owned.p = nullptr;
    return 1;
}
}
#define API extern "C" __declspec(dllexport)
// Capability marker for an explicit external backend selection. This function
// is safe to call during DLL discovery: no GPU/runtime initialization occurs.
API unsigned d4r_WindowsBackendVersion() { return 1; }
API unsigned NVSDK_NGX_D3D12_Init_Ext(unsigned long long app, const wchar_t* data, ID3D12Device* device, unsigned sdk, const void* info) {
    std::lock_guard<std::mutex> lock(apiMutex);
    return call([&] {
        if (!runtime) runtime = std::make_shared<d4r::win::Runtime>(device, app, data, sdk, static_cast<const d4r::ngx::FeatureCommonInfo*>(info));
        if (const char* backend = std::getenv("D4R_D3D12_COMMAND_BACKEND"); backend && std::string(backend) == "1")
            d4r::win::commands::install(device);
        return 1u;
    });
}
API unsigned NVSDK_NGX_D3D12_Init(unsigned long long app, const wchar_t* data, ID3D12Device* device, const void* info, unsigned sdk) {
    return NVSDK_NGX_D3D12_Init_Ext(app, data, device, sdk, info);
}
API unsigned NVSDK_NGX_D3D12_Init_ProjectID(const char* project, int engine, const char* version, const wchar_t* data, ID3D12Device* device, unsigned sdk, const void* info) {
    std::lock_guard<std::mutex> lock(apiMutex);
    return call([&] {
        if (!project || !*project) return invalid;
        d4r::ngx::ProjectIdentity identity{project, engine, version ? version : ""};
        if (!runtime) runtime = std::make_shared<d4r::win::Runtime>(device, 0, data, sdk, static_cast<const d4r::ngx::FeatureCommonInfo*>(info), &identity);
        if (const char* backend = std::getenv("D4R_D3D12_COMMAND_BACKEND"); backend && std::string(backend) == "1") d4r::win::commands::install(device);
        return 1u;
    });
}
API unsigned NVSDK_NGX_D3D12_GetFeatureRequirements(IDXGIAdapter* adapter, const d4r::ngx::FeatureDiscoveryInfo* info,
    d4r::ngx::FeatureRequirement* requirements) {
    if (!adapter || !info || !requirements) return invalid;
    return call([&] {
        *requirements = {}; requirements->FeatureSupported = 16; // Unsupported feature.
        if (info->FeatureID != 1) return 1u;
        DXGI_ADAPTER_DESC desc{}; d4r::win::dx(adapter->GetDesc(&desc), "NGX requirements adapter");
        // OptiScaler can override the DXGI vendor ID. The physical HIP
        // architecture and full adapter LUID are the authoritative identity.
        d4r::diag::HipApi hip(d4r::win::env_path("D4R_HIP_ROOT")); hipDeviceProp_t props{}; hip.select_architecture(-1,props);
        requirements->FeatureSupported = std::memcmp(&desc.AdapterLuid,props.luid,sizeof(LUID)) ? 4u : 0u;
        std::strcpy(requirements->MinOSVersion,"10.0.22000.0");
        std::printf("D4R_REQUIREMENTS feature=%u support=%u architecture=%s\n",info->FeatureID,requirements->FeatureSupported,d4r::diag::HipApi::target_arch());
        return 1u;
    });
}
API unsigned NVSDK_NGX_D3D12_AllocateParameters(void** p) {
    std::lock_guard<std::mutex> lock(apiMutex); return call([&] { return allocate(p, false); });
}
API unsigned NVSDK_NGX_D3D12_GetCapabilityParameters(void** p) {
    std::lock_guard<std::mutex> lock(apiMutex); return call([&] { return allocate(p, true); });
}
API unsigned NVSDK_NGX_D3D12_GetParameters(void** p) { return NVSDK_NGX_D3D12_GetCapabilityParameters(p); }
API unsigned NVSDK_NGX_D3D12_DestroyParameters(void* p) {
    std::lock_guard<std::mutex> lock(apiMutex);
    if (!parameters.erase(p)) return invalid;
    d4r_ngx_parameters_destroy(p); return 1;
}
API unsigned NVSDK_NGX_D3D12_GetScratchBufferSize(unsigned, const void*, size_t* size) {
    if (!size) return invalid; *size = 0; return 1;
}
API unsigned NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList*, unsigned feature, void* p, Handle** out) {
    std::lock_guard<std::mutex> lock(apiMutex);
    return call([&] {
        if (!runtime || feature != 1 || !p || !out) return invalid;
        auto value = std::make_shared<d4r::win::Feature>(runtime, p);
        auto handle = std::make_unique<Handle>(); handle->Id = nextId++;
        features.emplace(handle.get(), std::move(value)); *out = handle.release(); return 1u;
    });
}
API unsigned d4r_D3D12_EvaluateAtBoundary(ID3D12CommandQueue* queue, Handle* handle, void* p) {
    std::shared_ptr<d4r::win::Feature> feature;
    { std::lock_guard<std::mutex> lock(apiMutex);
      auto it = features.find(handle); if (it == features.end() || !p) return invalid; feature = it->second; }
    return call([&] { feature->evaluate_boundary(queue, p); return 1u; });
}
API unsigned NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList* list, const Handle* handle, void* p, void*) {
    std::shared_ptr<d4r::win::Feature> feature;
    { std::lock_guard<std::mutex> lock(apiMutex);
      auto it = features.find(const_cast<Handle*>(handle));
      if (it == features.end() || !p || !list) return invalid; feature = it->second; }
    return call([&] {
        struct Snapshot {
            void* parameters = d4r_ngx_parameters_create();
            std::vector<d4r::win::ComPtr<ID3D12Resource>> resources;
            d4r::win::ResourceAccess access[5];
            ~Snapshot() { if (parameters) d4r_ngx_parameters_destroy(parameters); }
        };
        auto snapshot = std::make_shared<Snapshot>();
        if (!snapshot->parameters) return failure;
        d4r::ngx::copy_create(p, snapshot->parameters); d4r::ngx::copy_frame(p, snapshot->parameters);
        const char* names[] = {"Color", "Depth", "MotionVectors", "Output", "ExposureTexture"};
        const char* states[] = {"D4R.Color.State", "D4R.Depth.State", "D4R.Motion.State", "D4R.Output.State", "D4R.Exposure.State"};
        for (unsigned i = 0; i < 5; ++i) {
            ID3D12Resource* resource = nullptr; (void)d4r_ngx_get_d3d12_resource(p, names[i], &resource);
            snapshot->resources.emplace_back(resource);
            d4r_ngx_set_d3d12_resource(snapshot->parameters, names[i], resource);
            const auto state = d4r::win::commands::resource_access(list, resource, d4r::win::ngx_resource_access(p, states[i],
                i == 3 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
            snapshot->access[i] = state;
            if (state.pending_split) throw std::runtime_error("NGX resource has an unfinished split barrier");
            d4r_ngx_set_uint(snapshot->parameters, states[i], unsigned(state.legacy));
            const std::string prefix(states[i]);
            d4r_ngx_set_uint(snapshot->parameters, (prefix + ".Enhanced").c_str(), state.enhanced);
            d4r_ngx_set_uint(snapshot->parameters, (prefix + ".Layout").c_str(), unsigned(state.layout));
            d4r_ngx_set_uint(snapshot->parameters, (prefix + ".Access").c_str(), unsigned(state.access));
            if (!std::getenv("D4R_QUIET_API")) std::printf("D4R_RESOURCE_ACCESS plane=%s enhanced=%u legacy=0x%x layout=%u access=0x%x\n",
                names[i], state.enhanced, unsigned(state.legacy), unsigned(state.layout), unsigned(state.access));
        }
        d4r::win::commands::record_boundary(list, [feature, snapshot](ID3D12CommandQueue* queue) {
            const char* names[] = {"Color", "Depth", "MotionVectors", "Output", "ExposureTexture"};
            const char* states[] = {"D4R.Color.State", "D4R.Depth.State", "D4R.Motion.State", "D4R.Output.State", "D4R.Exposure.State"};
            for (unsigned i = 0; i < 5; ++i) if (snapshot->access[i].inherited) {
                const auto resolved = d4r::win::commands::submitted_resource_access(snapshot->resources[i].Get(), snapshot->access[i]);
                if (resolved.pending_split) throw std::runtime_error("Inherited NGX texture has an unfinished split barrier");
                const std::string prefix(states[i]);
                d4r_ngx_set_uint(snapshot->parameters, states[i], unsigned(resolved.legacy));
                d4r_ngx_set_uint(snapshot->parameters, (prefix + ".Enhanced").c_str(), resolved.enhanced);
                d4r_ngx_set_uint(snapshot->parameters, (prefix + ".Layout").c_str(), unsigned(resolved.layout));
                d4r_ngx_set_uint(snapshot->parameters, (prefix + ".Access").c_str(), unsigned(resolved.access));
                if (!std::getenv("D4R_QUIET_API")) std::printf("D4R_INHERITED_ACCESS plane=%s tracked=%u enhanced=%u layout=%u legacy=0x%x\n",
                    names[i], !resolved.inherited, resolved.enhanced, unsigned(resolved.layout), unsigned(resolved.legacy));
            }
            feature->evaluate_boundary(queue, snapshot->parameters);
        });
        return 1u;
    });
}
API unsigned NVSDK_NGX_D3D12_EvaluateFeature_C(ID3D12GraphicsCommandList* list, const Handle* handle, void* p, void* callback) {
    return NVSDK_NGX_D3D12_EvaluateFeature(list, handle, p, callback);
}
API unsigned NVSDK_NGX_D3D12_ReleaseFeature(Handle* handle) {
    std::shared_ptr<d4r::win::Feature> feature;
    { std::lock_guard<std::mutex> lock(apiMutex);
      auto it = features.find(handle); if (it == features.end()) return invalid;
      feature = std::move(it->second); features.erase(it); delete handle; }
    // OptiScaler frees its root-CBV/descriptor backing after Release returns.
    // A retained C++ Feature alone does not keep that caller-owned memory alive.
    return call([&] { feature->drain(); feature.reset(); return 1u; });
}
API unsigned NVSDK_NGX_D3D12_Shutdown() {
    std::lock_guard<std::mutex> lock(apiMutex);
    return call([&] {
        if (runtime && std::getenv("D4R_ASYNC_INTEROP")) runtime->drain_async_queue();
        for (auto& pair : features) delete pair.first;
        features.clear();
        for (auto& pair : parameters) d4r_ngx_parameters_destroy(pair.first);
        parameters.clear(); runtime.reset(); return 1u;
    });
}
API unsigned NVSDK_NGX_D3D12_Shutdown1(ID3D12Device*) { return NVSDK_NGX_D3D12_Shutdown(); }

// Public diagnostics exercise queue/list integration without NVIDIA binaries.
API unsigned d4r_D3D12_InstallCommandBackend(ID3D12Device* device) {
    return call([&] { d4r::win::commands::install(device); return 1u; });
}
using DiagnosticBoundary = void(WINAPI*)(ID3D12CommandQueue*, void*);
API unsigned d4r_D3D12_RecordDiagnosticBoundary(ID3D12GraphicsCommandList* list, DiagnosticBoundary callback, void* context) {
    if (!callback) return invalid;
    return call([&] { d4r::win::commands::record_boundary(list, [=](ID3D12CommandQueue* queue) { callback(queue, context); }); return 1u; });
}
API unsigned long long d4r_D3D12_LiveRecordings() { return d4r::win::commands::live_recordings(); }
