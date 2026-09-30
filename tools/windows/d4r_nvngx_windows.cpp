// Native Windows backend. The Linux/Wine implementation remains separate.
#include "ngx_windows_runtime.h"
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
    try { return fn(); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "D4R_WINDOWS_FAILURE %s\n", error.what()); return failure;
    }
}
unsigned allocate(void** out, bool caps) {
    if (!runtime || !out) return invalid;
    void* value = d4r_ngx_parameters_create();
    if (!value) return failure;
    parameters.emplace(value, true); *out = value;
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
    return 1;
}
}
#define API extern "C" __declspec(dllexport)
API unsigned NVSDK_NGX_D3D12_Init_Ext(unsigned long long app, const wchar_t* data, ID3D12Device* device, unsigned sdk, const void*) {
    std::lock_guard<std::mutex> lock(apiMutex);
    return call([&] {
        if (!runtime) runtime = std::make_shared<d4r::win::Runtime>(device, app, data, sdk);
        return 1u;
    });
}
API unsigned NVSDK_NGX_D3D12_Init(unsigned long long app, const wchar_t* data, ID3D12Device* device, const void* info, unsigned sdk) {
    return NVSDK_NGX_D3D12_Init_Ext(app, data, device, sdk, info);
}
API unsigned NVSDK_NGX_D3D12_Init_ProjectID(const char*, int, const char*, const wchar_t* data, ID3D12Device* device, unsigned sdk, const void* info) {
    // Driver core's CUDA ABI uses the numerical application identity.
    return NVSDK_NGX_D3D12_Init_Ext(241534723ull, data, device, sdk, info);
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
API unsigned NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList*, const Handle*, void*, void*) {
    // Explicitly refuse an unsubmitted game list until queue integration exists.
    std::fprintf(stderr, "D4R_WINDOWS_FAILURE command-list queue backend pending; no frame-age fallback\n");
    return failure;
}
API unsigned NVSDK_NGX_D3D12_EvaluateFeature_C(ID3D12GraphicsCommandList* list, const Handle* handle, void* p, void* callback) {
    return NVSDK_NGX_D3D12_EvaluateFeature(list, handle, p, callback);
}
API unsigned NVSDK_NGX_D3D12_ReleaseFeature(Handle* handle) {
    std::shared_ptr<d4r::win::Feature> feature;
    { std::lock_guard<std::mutex> lock(apiMutex);
      auto it = features.find(handle); if (it == features.end()) return invalid;
      feature = std::move(it->second); features.erase(it); delete handle; }
    return call([&] { feature.reset(); return 1u; });
}
API unsigned NVSDK_NGX_D3D12_Shutdown() {
    std::lock_guard<std::mutex> lock(apiMutex);
    return call([&] {
        for (auto& pair : features) delete pair.first;
        features.clear();
        for (auto& pair : parameters) d4r_ngx_parameters_destroy(pair.first);
        parameters.clear(); runtime.reset(); return 1u;
    });
}
API unsigned NVSDK_NGX_D3D12_Shutdown1(ID3D12Device*) { return NVSDK_NGX_D3D12_Shutdown(); }
