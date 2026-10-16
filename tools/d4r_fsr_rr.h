// AMD FSR Ray Regeneration (FidelityFX denoiser 1.2) runtime access, shared by the NGX shim's
// FSR Ray Reconstruction backend and tools/fsr_rr_probe.cpp.
//
// The denoiser is AMD's signed amd_fidelityfx_denoiser_dx12.dll, loaded from an explicit path
// (never from a search path, never patched) and driven through its five FFX API exports. It
// records its compute work into the caller's D3D12 command list; nothing here involves CUDA,
// ZLUDA or NVIDIA's denoiser. Struct layouts come from AMD's MIT-licensed headers in
// third_party/fidelityfx (SDK 2.3.0 / FSR Ray Regeneration 1.2.0).
//
// Each stage reports separately (see FsrRrStage): a missing DLL, a missing export, a provider
// query that returns no denoiser for this device, a failed context creation (where shader
// pipelines are built) and a failed dispatch are different failures and are never folded together.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../third_party/fidelityfx/api/include/ffx_api.h"
#include "../third_party/fidelityfx/api/include/ffx_api_types.h"
#include "../third_party/fidelityfx/denoisers/include/ffx_denoiser.h"

// ffx_api_dx12.h's backend descriptor, declared here because that header's inline helpers call
// ID3D12Resource::GetDesc() by value, which does not compile with WIDL_EXPLICIT_AGGREGATE_RETURNS.
#ifndef FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12
#define FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12 FFX_API_MAKE_BACKEND_SUB_ID(FFX_API_BACKEND_ID_DX12, 0x02)
struct ffxCreateBackendDX12Desc
{
    ffxCreateContextDescHeader header;
    ID3D12Device* device;
};
#endif
static_assert(sizeof(ffxCreateBackendDX12Desc) == 24, "ffx_api_dx12.h layout");
// The 1.2 descriptors as the signed runtime reads them (x64, natural alignment, same as MSVC).
static_assert(sizeof(FfxApiResource) == 48 && sizeof(FfxApiDenoiserSignal) == 104, "ffx_api_types.h layout");
static_assert(sizeof(ffxCreateContextDescDenoiser) == 40, "ffx_denoiser.h 1.2 create layout");
static_assert(sizeof(ffxDispatchDescDenoiser) == 448 && offsetof(ffxDispatchDescDenoiser, view) == 296 &&
                  offsetof(ffxDispatchDescDenoiser, renderSize) == 432,
              "ffx_denoiser.h 1.2 dispatch layout");

enum class FsrRrStage
{
    NotLoaded,        // nothing tried yet
    DllMissing,       // no file at the configured path
    DllLoadFailed,    // LoadLibrary failed (dependencies, architecture)
    ExportsMissing,   // not an FFX API DLL
    NoProvider,       // the runtime returned no denoiser version for this device
    Ready,            // a provider was enumerated for this device; contexts may still fail
};

inline const char* fsr_rr_stage_name(FsrRrStage stage)
{
    switch (stage)
    {
    case FsrRrStage::NotLoaded: return "not loaded";
    case FsrRrStage::DllMissing: return "denoiser DLL missing";
    case FsrRrStage::DllLoadFailed: return "denoiser DLL failed to load";
    case FsrRrStage::ExportsMissing: return "denoiser DLL lacks the FFX API exports";
    case FsrRrStage::NoProvider: return "no FSR Ray Regeneration provider for this device";
    case FsrRrStage::Ready: return "provider available";
    }
    return "?";
}

inline const char* fsr_rr_return_name(ffxReturnCode_t code)
{
    switch (code)
    {
    case FFX_API_RETURN_OK: return "OK";
    case FFX_API_RETURN_ERROR: return "ERROR";
    case FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE: return "ERROR_UNKNOWN_DESCTYPE";
    case FFX_API_RETURN_ERROR_RUNTIME_ERROR: return "ERROR_RUNTIME_ERROR";
    case FFX_API_RETURN_NO_PROVIDER: return "NO_PROVIDER";
    case FFX_API_RETURN_ERROR_MEMORY: return "ERROR_MEMORY";
    case FFX_API_RETURN_ERROR_PARAMETER: return "ERROR_PARAMETER";
    case FFX_API_RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE: return "PROVIDER_NO_SUPPORT_NEW_DESCTYPE";
    }
    return "unknown";
}

// The loaded runtime and the provider the device was offered. One per process: the DLL keeps
// global state (its debug callback), so it is loaded once and never unloaded while contexts exist.
struct FsrRrRuntime
{
    HMODULE module = nullptr;
    std::wstring path;        // as configured
    std::wstring mappedPath;  // as the loader mapped it
    PfnFfxCreateContext createContext = nullptr;
    PfnFfxDestroyContext destroyContext = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    FsrRrStage stage = FsrRrStage::NotLoaded;
    ffxReturnCode_t queryResult = FFX_API_RETURN_OK;
    std::vector<uint64_t> versionIds;
    std::vector<std::string> versionNames;
    uint64_t providerId = 0;   // the provider contexts are created with (the first enumerated)
    std::string providerName;

    // Loads `dll` with its own directory searched for dependencies. Does not query providers.
    bool load(const std::wstring& dll)
    {
        path = dll;
        if (GetFileAttributesW(dll.c_str()) == INVALID_FILE_ATTRIBUTES)
        {
            stage = FsrRrStage::DllMissing;
            return false;
        }
        module = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (module == nullptr)
        {
            stage = FsrRrStage::DllLoadFailed;
            return false;
        }
        wchar_t mapped[MAX_PATH] = {};
        GetModuleFileNameW(module, mapped, MAX_PATH);
        mappedPath = mapped;
        createContext = reinterpret_cast<PfnFfxCreateContext>(reinterpret_cast<void*>(GetProcAddress(module, "ffxCreateContext")));
        destroyContext = reinterpret_cast<PfnFfxDestroyContext>(reinterpret_cast<void*>(GetProcAddress(module, "ffxDestroyContext")));
        configure = reinterpret_cast<PfnFfxConfigure>(reinterpret_cast<void*>(GetProcAddress(module, "ffxConfigure")));
        query = reinterpret_cast<PfnFfxQuery>(reinterpret_cast<void*>(GetProcAddress(module, "ffxQuery")));
        dispatch = reinterpret_cast<PfnFfxDispatch>(reinterpret_cast<void*>(GetProcAddress(module, "ffxDispatch")));
        if (!createContext || !destroyContext || !configure || !query || !dispatch)
        {
            stage = FsrRrStage::ExportsMissing;
            return false;
        }
        return true;
    }

    // Routes the runtime's own error/warning messages to `callback` (may be called on any thread).
    ffxReturnCode_t set_message_callback(ffxApiMessage callback, uint32_t level)
    {
        ffxConfigureDescGlobalDebug desc = {};
        desc.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG;
        desc.effectId = FFX_API_EFFECT_ID_DENOISER;
        desc.fpMessage = callback;
        desc.debugLevel = level;
        return configure(nullptr, &desc.header);
    }

    // The two-call version enumeration AMD's sample uses, against the real device. Zero
    // versions is the "no provider for this device" outcome, distinct from a query error.
    bool query_providers(ID3D12Device* device)
    {
        versionIds.clear();
        versionNames.clear();
        uint64_t count = 0;
        ffxQueryDescGetVersions desc = {};
        desc.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        desc.createDescType = FFX_API_EFFECT_ID_DENOISER;
        desc.device = device;
        desc.outputCount = &count;
        queryResult = query(nullptr, &desc.header);
        if (queryResult != FFX_API_RETURN_OK || count == 0)
        {
            stage = FsrRrStage::NoProvider;
            return false;
        }
        versionIds.resize(count);
        std::vector<const char*> names(count, nullptr);
        desc.versionIds = versionIds.data();
        desc.versionNames = names.data();
        queryResult = query(nullptr, &desc.header);
        if (queryResult != FFX_API_RETURN_OK || count == 0)
        {
            stage = FsrRrStage::NoProvider;
            return false;
        }
        versionIds.resize(count);
        for (uint64_t index = 0; index < count; ++index)
            versionNames.emplace_back(names[index] != nullptr ? names[index] : "");
        providerId = versionIds[0];
        providerName = versionNames[0];
        stage = FsrRrStage::Ready;
        return true;
    }
};

// GPU memory the denoiser would allocate for this configuration (total, aliasable).
inline ffxReturnCode_t fsr_rr_query_memory(FsrRrRuntime& runtime, ID3D12Device* device, uint32_t width,
                                           uint32_t height, uint32_t signals, uint32_t flags,
                                           FfxApiEffectMemoryUsage& usage)
{
    ffxOverrideVersion version = {};
    version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version.versionId = runtime.providerId;
    ffxQueryDescDenoiserGetGPUMemoryUsage desc = {};
    desc.header.type = FFX_API_QUERY_DESC_TYPE_DENOISER_GPU_MEMORY_USAGE;
    desc.header.pNext = &version.header;
    desc.device = device;
    desc.maxRenderSize = {width, height};
    desc.signalFlags = signals;
    desc.flags = flags;
    desc.gpuMemoryUsage = &usage;
    return runtime.query(nullptr, &desc.header);
}

// Creates one denoiser context on the enumerated provider. The descriptor chain only has to live
// for the call: ffx_api.h requires pointers *inside* descriptors to outlive the context, and
// none of these carry any.
inline ffxReturnCode_t fsr_rr_create_context(FsrRrRuntime& runtime, ID3D12Device* device, uint32_t maxWidth,
                                             uint32_t maxHeight, uint32_t signals, uint32_t flags,
                                             ffxContext& context)
{
    ffxCreateContextDescDenoiser desc = {};
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER;
    desc.version = FFX_DENOISER_VERSION;
    desc.maxRenderSize = {maxWidth, maxHeight};
    desc.signalFlags = signals;
    desc.flags = flags;
    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = device;
    ffxOverrideVersion version = {};
    version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version.versionId = runtime.providerId;
    desc.header.pNext = &backend.header;
    backend.header.pNext = &version.header;
    context = nullptr;
    return runtime.createContext(&context, &desc.header, nullptr);
}

// Asks a live context which provider actually serves it (the runtime's own account, logged as
// the "actual backend").
inline std::string fsr_rr_context_provider(FsrRrRuntime& runtime, ffxContext& context, uint64_t* id = nullptr)
{
    ffxQueryGetProviderVersion desc = {};
    desc.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (runtime.query(&context, &desc.header) != FFX_API_RETURN_OK || desc.versionName == nullptr)
        return std::string();
    if (id != nullptr)
        *id = desc.versionId;
    return desc.versionName;
}

// An FfxApiResource for a 2D texture whose description the caller already knows (the shim never
// calls GetDesc() by value; see above).
inline FfxApiResource fsr_rr_texture(ID3D12Resource* resource, const D3D12_RESOURCE_DESC& desc, uint32_t format,
                                     uint32_t state)
{
    FfxApiResource result = {};
    result.resource = resource;
    result.state = state;
    if (resource == nullptr)
        return result;
    result.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    result.description.format = format;
    result.description.width = static_cast<uint32_t>(desc.Width);
    result.description.height = desc.Height;
    result.description.depth = desc.DepthOrArraySize;
    result.description.mipCount = desc.MipLevels;
    result.description.usage = FFX_API_RESOURCE_USAGE_READ_ONLY;
    if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        result.description.usage |= FFX_API_RESOURCE_USAGE_UAV;
    return result;
}

// The FFX surface formats the adapter's own intermediate textures use.
inline uint32_t fsr_rr_surface_format(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R32_FLOAT: return FFX_API_SURFACE_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R8G8B8A8_UNORM: return FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R16_FLOAT: return FFX_API_SURFACE_FORMAT_R16_FLOAT;
    case DXGI_FORMAT_R8_UNORM: return FFX_API_SURFACE_FORMAT_R8_UNORM;
    default: return FFX_API_SURFACE_FORMAT_UNKNOWN;
    }
}
