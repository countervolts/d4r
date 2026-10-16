// Native Windows backend. The Linux/Wine implementation remains separate.
#include "ngx_windows_runtime.h"
#include "ngx_runtime_initialization.h"
#include "player_config.h"
#if defined(D4R_NATIVE_RR_BACKEND)
#include "rr_native_backend.h"
#endif
#include <dxgi.h>
#include <unordered_map>
#include <functional>

namespace {
std::mutex apiMutex;
std::shared_ptr<d4r::win::Runtime> runtime;
std::shared_ptr<d4r::win::RuntimeInitialization> initialization;
bool deferredSr = false;
void ensure_sr_runtime() {
    if (!runtime) {
        if (!initialization) throw std::runtime_error("NGX is not initialized");
        runtime = initialization->create();
    }
}
void initialize(ID3D12Device* device, unsigned long long app, const wchar_t* data, unsigned sdk,
    const d4r::ngx::FeatureCommonInfo* common, const d4r::ngx::ProjectIdentity* project = nullptr) {
    if (initialization) {
        if (initialization->device.Get() != device) throw std::runtime_error("NGX already initialized on another device");
        return;
    }
    auto candidate = std::make_shared<d4r::win::RuntimeInitialization>(device, app, data, sdk, common, project);
    bool native = false;
#if defined(D4R_NATIVE_RR_BACKEND)
    const char* backend = std::getenv("D4R_RR_BACKEND"); native = backend && std::string(backend) == "native";
#endif
    // Default SR startup remains eager. Native RR owns its Vulkan execution
    // independently; SR initializes only when its capabilities/feature are used.
    if (!native) runtime = candidate->create();
    initialization = std::move(candidate); deferredSr = native;
}
struct Handle { unsigned Id; };
unsigned nextId = 1;
std::unordered_map<Handle*, std::shared_ptr<d4r::win::Feature>> features;
#if defined(D4R_NATIVE_RR_BACKEND)
struct RrFeature {
    d4r::win::rr::Creation creation;
    d4r::win::rr::Backend backend;
    RrFeature(ID3D12Device* device, void* parameters, const std::string& directory)
        : creation(parameters), backend(device, creation, directory) {}
};
std::unordered_map<Handle*, std::shared_ptr<RrFeature>> rrFeatures;
std::string rr_directory() {
    const char* backend = std::getenv("D4R_RR_BACKEND");
    if (!backend || std::string(backend) != "native")
        throw std::runtime_error("Native RR requires D4R_RR_BACKEND=native");
    return d4r::win::env_path("D4R_RR_MODEL_DIR");
}
bool rr_ready(ID3D12Device* device) {
    try {
        const auto directory = rr_directory();
        void* p = d4r_ngx_parameters_create();
        if (!p) return false;
        struct Cleanup { void* p; ~Cleanup() { d4r_ngx_parameters_destroy(p); } } cleanup{p};
        for (const char* name : {"Width", "Height", "OutWidth", "OutHeight"}) d4r_ngx_set_uint(p, name, 64);
        d4r_ngx_set_uint(p, "DLSS.Feature.Create.Flags", 3);
        d4r::win::rr::Creation creation(p);
        // Readiness checks the actual configured model's pipeline/weight load,
        // adapter features and packet/fence import. No synthetic frame runs.
        d4r::win::rr::Backend candidate(device, creation, directory);
        return true;
    } catch (const std::exception& error) {
        if (std::getenv("D4R_RR_BACKEND")) std::fprintf(stderr, "D4R_RR_UNAVAILABLE %s\n", error.what());
        return false;
    }
}
struct RrSnapshot {
    d4r::win::rr::Snapshot frame;
    std::array<d4r::win::ResourceAccess, d4r::win::rr::Snapshot::Count> access;
    std::array<bool, d4r::win::rr::Snapshot::Count> explicitState{};
    RrSnapshot(const d4r::win::rr::Creation& creation, void* p, ID3D12GraphicsCommandList* list)
        : frame(creation, p) {
        const char* names[] = {"D4R.Color.State", "D4R.Motion.State", "D4R.Depth.State", "D4R.Normals.State",
            "D4R.Roughness.State", "D4R.Diffuse.State", "D4R.Specular.State", "D4R.Output.State", "D4R.Exposure.State",
            "D4R.HitDistance.State", "D4R.Disocclusion.State", "D4R.Subsurface.State", "D4R.BeforeParticles.State"};
        for (size_t i = 0; i < access.size(); ++i) {
            if (!frame.resources[i]) {
                if (i < 8) throw std::runtime_error("Missing required NGX RR guide/output");
                continue;
            }
            const auto fallback = d4r::win::ngx_resource_access(p, names[i], i == 7 ?
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            unsigned declared = 0;
            explicitState[i] = d4r_ngx_get_uint(p, names[i], &declared) == 1;
            access[i] = list ? d4r::win::commands::resource_access(list, frame.resources[i].Get(), fallback) :
                d4r::win::commands::submitted_resource_access(frame.resources[i].Get(), fallback);
            if (access[i].pending_split) throw std::runtime_error("NGX RR texture has an unfinished split barrier");
        }
        if (frame.settings.roughnessPacked) {
            access[4] = access[3]; explicitState[4] = explicitState[3];
        }
    }
    void resolve() {
        for (size_t i = 0; i < access.size(); ++i) if (access[i].inherited) {
            access[i] = d4r::win::commands::submitted_resource_access(frame.resources[i].Get(), access[i]);
            // Explicit caller state remains valid when no prior submitted list
            // supplied a tracked transition for this resource.
            if (access[i].inherited && explicitState[i]) access[i].inherited = false;
            if (access[i].pending_split || access[i].inherited) throw std::runtime_error("NGX RR inherited texture state unresolved");
        }
    }
};
#endif
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
    if (!initialization || !out) return invalid;
    void* value = d4r_ngx_parameters_create();
    if (!value) return failure;
    *out = nullptr;
    struct LocalParameters {
        void* p;
        ~LocalParameters() { if (p) d4r_ngx_parameters_destroy(p); }
    } owned{value};
    if (caps) {
        try { ensure_sr_runtime(); }
        catch (const std::exception& error) {
            if (!deferredSr) throw;
            std::fprintf(stderr, "D4R_SR_UNAVAILABLE %s\n", error.what());
            d4r_ngx_set_int(value, "SuperSampling.Available", 0);
            d4r_ngx_set_int(value, "SuperSampling.NeedsUpdatedDriver", 0);
            d4r_ngx_set_int(value, "SuperSampling.FeatureInitResult", int(failure));
        }
        if (runtime) {
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
        bool rrAvailable = false;
#if defined(D4R_NATIVE_RR_BACKEND)
        rrAvailable = rr_ready(initialization->device.Get());
#endif
        d4r_ngx_set_int(value, "SuperSamplingDenoising.Available", rrAvailable ? 1 : 0);
        d4r_ngx_set_int(value, "SuperSamplingDenoising.NeedsUpdatedDriver", 0);
        d4r_ngx_set_int(value, "SuperSamplingDenoising.FeatureInitResult", rrAvailable ? 1 : int(0xBAD00001u));
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
        d4r::win::player::ensure_environment();
        initialize(device, app, data, sdk, static_cast<const d4r::ngx::FeatureCommonInfo*>(info));
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
        d4r::win::player::ensure_environment();
        d4r::ngx::ProjectIdentity identity{project, engine, version ? version : ""};
        initialize(device, 0, data, sdk, static_cast<const d4r::ngx::FeatureCommonInfo*>(info), &identity);
        if (const char* backend = std::getenv("D4R_D3D12_COMMAND_BACKEND"); backend && std::string(backend) == "1") d4r::win::commands::install(device);
        return 1u;
    });
}
API unsigned NVSDK_NGX_D3D12_GetFeatureRequirements(IDXGIAdapter* adapter, const d4r::ngx::FeatureDiscoveryInfo* info,
    d4r::ngx::FeatureRequirement* requirements) {
    if (!adapter || !info || !requirements) return invalid;
    return call([&] {
        *requirements = {}; requirements->FeatureSupported = 16; // Unsupported feature.
        if (info->FeatureID != 1) {
#if defined(D4R_NATIVE_RR_BACKEND)
            if (info->FeatureID == 13) {
                d4r::win::player::ensure_environment();
                d4r::win::ComPtr<ID3D12Device> device;
                if (SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))) && rr_ready(device.Get())) {
                    requirements->FeatureSupported = 0;
                    std::strcpy(requirements->MinOSVersion, "10.0.22000.0");
                }
            }
#endif
            return 1u;
        }
        d4r::win::player::ensure_environment();
        DXGI_ADAPTER_DESC desc{}; d4r::win::dx(adapter->GetDesc(&desc), "NGX requirements adapter");
        // OptiScaler can override the DXGI vendor ID. The physical HIP
        // architecture and full adapter LUID are the authoritative identity.
        d4r::diag::HipApi hip(d4r::win::env_path("D4R_HIP_ROOT")); hipDeviceProp_t props{};
        const int ordinal = hip.select_architecture(-1,props,&desc.AdapterLuid,true);
        requirements->FeatureSupported = ordinal < 0 ? 4u : 0u;
        std::strcpy(requirements->MinOSVersion,"10.0.22000.0");
        std::printf("D4R_REQUIREMENTS feature=%u support=%u architecture=%s adapter_luid=%08lx:%08lx hip_ordinal=%d\n",
            info->FeatureID,requirements->FeatureSupported,d4r::diag::HipApi::target_arch(),
            static_cast<unsigned long>(desc.AdapterLuid.HighPart),static_cast<unsigned long>(desc.AdapterLuid.LowPart),ordinal);
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
        if (!initialization || !p || !out) return invalid;
        *out = nullptr;
#if defined(D4R_NATIVE_RR_BACKEND)
        if (feature == 13) {
            auto value = std::make_shared<RrFeature>(initialization->device.Get(), p, rr_directory());
            d4r::win::commands::install(initialization->device.Get());
            auto handle = std::make_unique<Handle>(); handle->Id = nextId++;
            rrFeatures.emplace(handle.get(), std::move(value)); *out = handle.release(); return 1u;
        }
#endif
        if (feature != 1) return invalid;
        ensure_sr_runtime();
        d4r::win::player::apply_preset(p);
        auto value = std::make_shared<d4r::win::Feature>(runtime, p);
        auto handle = std::make_unique<Handle>(); handle->Id = nextId++;
        features.emplace(handle.get(), std::move(value)); *out = handle.release(); return 1u;
    });
}
API unsigned d4r_D3D12_EvaluateAtBoundary(ID3D12CommandQueue* queue, Handle* handle, void* p) {
#if defined(D4R_NATIVE_RR_BACKEND)
    std::shared_ptr<RrFeature> rr;
    { std::lock_guard<std::mutex> lock(apiMutex);
      auto it = rrFeatures.find(handle); if (it != rrFeatures.end()) rr = it->second; }
    if (rr) return call([&] {
        RrSnapshot snapshot(rr->creation, p, nullptr); snapshot.resolve();
        rr->backend.evaluate(queue, snapshot.frame, snapshot.access); return 1u;
    });
#endif
    std::shared_ptr<d4r::win::Feature> feature;
    { std::lock_guard<std::mutex> lock(apiMutex);
      auto it = features.find(handle); if (it == features.end() || !p) return invalid; feature = it->second; }
    return call([&] { feature->evaluate_boundary(queue, p); return 1u; });
}
API unsigned NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList* list, const Handle* handle, void* p, void*) {
#if defined(D4R_NATIVE_RR_BACKEND)
    std::shared_ptr<RrFeature> rr;
    { std::lock_guard<std::mutex> lock(apiMutex);
      auto it = rrFeatures.find(const_cast<Handle*>(handle)); if (it != rrFeatures.end()) rr = it->second; }
    if (rr) return call([&] {
        if (!p || !list) return invalid;
        auto snapshot = std::make_shared<RrSnapshot>(rr->creation, p, list);
        // Reject unsupported contracts before splitting the logical command
        // list. Submission still rechecks resources and resolved access.
        rr->backend.validate_frame(snapshot->frame);
        d4r::win::commands::record_boundary(list, [rr, snapshot](ID3D12CommandQueue* queue) {
            snapshot->resolve(); rr->backend.evaluate(queue, snapshot->frame, snapshot->access);
        });
        return 1u;
    });
#endif
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
        d4r::win::player::apply_preset(p);
        d4r::ngx::copy_create(p, snapshot->parameters); d4r::ngx::copy_frame(p, snapshot->parameters);
        // NVIDIA's library stamps "DLSS SDK - DO NOT DISTRIBUTE" over its output unless the SDK's
        // Disable.Watermark parameter is set (the Linux shim sets it too). The game's own value is
        // not used. [DLSS] ShowWatermark = true (D4R_SHIM_WATERMARK=1) keeps the stamp.
        static const bool watermark = [] { const char* v = std::getenv("D4R_SHIM_WATERMARK"); return v && v[0] == '1'; }();
        d4r_ngx_set_int(snapshot->parameters, "Disable.Watermark", watermark ? 0 : 1);
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
#if defined(D4R_NATIVE_RR_BACKEND)
    std::shared_ptr<RrFeature> rr;
    { std::lock_guard<std::mutex> lock(apiMutex);
      auto it = rrFeatures.find(handle);
      if (it != rrFeatures.end()) { rr = std::move(it->second); rrFeatures.erase(it); delete handle; } }
    if (rr) return call([&] { rr->backend.drain(); rr.reset(); return 1u; });
#endif
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
#if defined(D4R_NATIVE_RR_BACKEND)
        for (auto& pair : rrFeatures) pair.second->backend.drain();
        for (auto& pair : rrFeatures) delete pair.first;
        rrFeatures.clear();
#endif
        for (auto& pair : features) delete pair.first;
        features.clear();
        for (auto& pair : parameters) d4r_ngx_parameters_destroy(pair.first);
        parameters.clear(); runtime.reset(); initialization.reset(); deferredSr = false; return 1u;
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
