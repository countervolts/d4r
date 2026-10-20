// FSR Ray Regeneration as the Ray Reconstruction backend ([RayReconstruction] Backend = fsr,
// D4R_RR_BACKEND=fsr). Included by d4r_nvngx_shim.cpp.
//
// A game's Ray Reconstruction feature (NGX feature 13) then runs, in the game's command list and in
// the game's own frame:
//   1. fsr_rr_prepare.comp (Vulkan, recorded through vkd3d-proton's interop): the game's DLSS-RR inputs
//      become the denoiser's signed linear depth, 2.5D motion, octahedral normals and linear albedos,
//      and its noisy colour is split by albedo into demodulated diffuse and specular signals plus a
//      residual the denoiser never sees (particles, sky, what demodulation could not carry);
//   2. AMD's FSR Ray Regeneration 1.2 (amd_fidelityfx_denoiser_dx12.dll, unmodified; ffxDispatch);
//   3. fsr_rr_compose.comp: the denoised signals remodulated, plus the residual, at render resolution;
//   4. Super Sampling to the output size the game created feature 13 with, by an internal Super
//      Sampling feature (feature 1) configured like the user's own - [DLSS] Model and the engine or
//      CUDA backend - and evaluated in the same command list. Ray Reconstruction is never run a
//      second time as the upscaler.
// Nothing reaches NVIDIA's denoiser before step 4, and AMD's denoiser never goes through CUDA or
// ZLUDA. A frame that cannot be denoised fails its evaluation (logged); it is never upscaled noisy as
// if it had been denoised.
//
// Lifetime: the game's command lists retain the feature (DeferredResources), so the context, the
// intermediate textures and the views are destroyed in fsr_rr_destroy only once the GPU has
// finished every list that used them. The internal Super Sampling feature is released with its owner.

#include "d4r_fsr_rr_dispatch_profile.h"

// D4R_FSR_RR_HALO_LIMIT (default 1, 0 = off): each denoised lobe is capped at this multiple of the brightest noisy
// input of that lobe around the texel. The denoiser leaks a thin object's large demodulated signal across its edge,
// where the neighbour's albedo turns it into a bright fringe (poles and wires against the sky, hair, the edge of water).
static float haloLimitSetting()
{
    const std::string halo = env_string("D4R_FSR_RR_HALO_LIMIT");
    const float value = halo.empty() ? 1.0f : std::strtof(halo.c_str(), nullptr);
    return value > 0.0f ? value : 0.0f;
}

struct FsrRrGlobal
{
    bool attempted = false;
    bool ready = false;
    std::string reason = "the FSR Ray Regeneration runtime has not been loaded yet";
    std::wstring path;
    FsrRrRuntime runtime;
    std::string sha256, fileVersion;
    // The two Vulkan passes, created once on vkd3d-proton's device.
    std::once_flag passesOnce;
    bool passesReady = false;
    D4rFsrRrPasses passes;
    VkDevice vkDevice = VK_NULL_HANDLE;
    ID3D12DXVKInteropDevice1* interop = nullptr;
    std::atomic<unsigned> messages{0};
};
static FsrRrGlobal g_fsr;

static RrBackend rr_backend()
{
    const std::string value = ascii_lower(env_string("D4R_RR_BACKEND"));
    if (value.empty() || value == "auto" || value == "dlss")
        return RrBackend::Dlss;
    if (value == "fsr")
        return RrBackend::Fsr;
    return RrBackend::Invalid;
}

static const char* rr_backend_name(RrBackend backend)
{
    return backend == RrBackend::Fsr ? "FSR Ray Regeneration" : backend == RrBackend::Dlss ? "DLSS Ray Reconstruction"
                                                                                           : "invalid";
}

static std::wstring fsr_rr_dll_path()
{
    wchar_t value[MAX_PATH] = {};
    const DWORD length = GetEnvironmentVariableW(L"D4R_FSR_RR_DLL", value, MAX_PATH);
    if (length > 0 && length < MAX_PATH)
    {
        std::wstring path(value, length);
        if (!path.empty() && path[0] == L'/') // a Linux path from d4r_config.py
            path = L"Z:" + path;
        return path;
    }
    ensure_portable_config();
    if (!g_portable.dir.empty())
        return g_portable.dir + L"\\amd_fidelityfx_denoiser_dx12.dll";
    wchar_t self[MAX_PATH] = {};
    const DWORD selfLength = GetModuleFileNameW(g_selfModule, self, MAX_PATH);
    std::wstring directory(self, selfLength);
    directory.resize(directory.find_last_of(L"\\/") == std::wstring::npos ? 0 : directory.find_last_of(L"\\/"));
    return directory + L"\\amd_fidelityfx_denoiser_dx12.dll";
}

// SHA-256 of a file (bcrypt), for the log: which exact runtime served the frames.
static std::string fsr_rr_file_sha256(const std::wstring& path)
{
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return "unreadable";
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    std::string text = "unavailable";
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0 &&
        BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) >= 0)
    {
        std::vector<unsigned char> buffer(1 << 20);
        DWORD read = 0;
        bool ok = true;
        while (ok && ReadFile(file, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr) && read > 0)
            ok = BCryptHashData(hash, buffer.data(), read, 0) >= 0;
        unsigned char digest[32];
        if (ok && BCryptFinishHash(hash, digest, sizeof(digest), 0) >= 0)
        {
            char hex[65];
            for (int i = 0; i < 32; ++i)
                std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);
            text = hex;
        }
    }
    if (hash != nullptr)
        BCryptDestroyHash(hash);
    if (algorithm != nullptr)
        BCryptCloseAlgorithmProvider(algorithm, 0);
    CloseHandle(file);
    return text;
}

static std::string fsr_rr_file_version(HMODULE module)
{
    HRSRC found = FindResourceW(module, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(16));
    if (found == nullptr)
        return "no version resource";
    const DWORD size = SizeofResource(module, found);
    HGLOBAL loaded = LoadResource(module, found);
    const unsigned char* bytes = loaded != nullptr ? static_cast<const unsigned char*>(LockResource(loaded)) : nullptr;
    for (DWORD at = 0; bytes != nullptr && at + 16 <= size; at += 4)
    {
        uint32_t words[4];
        std::memcpy(words, bytes + at, sizeof(words));
        if (words[0] != 0xFEEF04BDu)
            continue;
        char text[64];
        std::snprintf(text, sizeof(text), "%u.%u.%u.%u", words[2] >> 16, words[2] & 0xffffu, words[3] >> 16,
                      words[3] & 0xffffu);
        return text;
    }
    return "no version resource";
}

static void fsr_rr_message(uint32_t type, const wchar_t* message)
{
    // The runtime's own diagnostics; bounded, since a misconfigured frame repeats them every frame.
    const unsigned count = ++g_fsr.messages;
    if (count <= 64)
        logf("FSR Ray Regeneration runtime %s: %ls", type == FFX_API_MESSAGE_TYPE_WARNING ? "warning" : "error",
             message);
    else if (count == 65)
        logf("FSR Ray Regeneration runtime: further messages suppressed");
}

// Loads the runtime and asks it for a provider on the game's device. Called once after NGX Init (the
// device is known then), with g.mutex held. Cheap: no shader is built until a feature creates its context.
static void fsr_rr_prepare_runtime()
{
    d4r_fsr4_identity::OriginalIdentityScope originalIdentity;
    if (g_fsr.attempted)
        return;
    g_fsr.attempted = true;
    g_fsr.path = fsr_rr_dll_path();
    if (g.device == nullptr)
    {
        g_fsr.reason = "FSR Ray Regeneration needs the game's D3D12 device (NVSDK_NGX_D3D12_Init with a device)";
        logf("Ray Reconstruction backend fsr: %s", g_fsr.reason.c_str());
        return;
    }
    std::string extension;
    const bool extensionOk = d4r_fsr_rr_driver_extension(g.device, extension);
    logf("FSR Ray Regeneration: %s", extension.c_str());
    if (!g_fsr.runtime.load(g_fsr.path))
    {
        g_fsr.reason = std::string(fsr_rr_stage_name(g_fsr.runtime.stage)) + " (" +
                       std::string(g_fsr.path.begin(), g_fsr.path.end()) + "); copy AMD's "
                       "amd_fidelityfx_denoiser_dx12.dll (FidelityFX SDK 2.3, FSR Ray Regeneration 1.2) there or set "
                       "[RayReconstruction] FsrDenoiser";
        logf("Ray Reconstruction backend fsr unavailable: %s", g_fsr.reason.c_str());
        return;
    }
    g_fsr.sha256 = fsr_rr_file_sha256(g_fsr.runtime.mappedPath);
    g_fsr.fileVersion = fsr_rr_file_version(g_fsr.runtime.module);
    g_fsr.runtime.set_message_callback(&fsr_rr_message, FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS);
    if (!g_fsr.runtime.query_providers(g.device))
    {
        g_fsr.reason = std::string(fsr_rr_stage_name(g_fsr.runtime.stage)) + " (query " +
                       fsr_rr_return_name(g_fsr.runtime.queryResult) + ")" +
                       (extensionOk ? "" : "; Proton's amdxc64.dll could not be loaded");
        logf("Ray Reconstruction backend fsr unavailable: %ls (sha256 %s, file version %s): %s",
             g_fsr.runtime.mappedPath.c_str(), g_fsr.sha256.c_str(), g_fsr.fileVersion.c_str(), g_fsr.reason.c_str());
        return;
    }
    g_fsr.ready = true;
    g_fsr.reason = "available";
    logf("Ray Reconstruction backend fsr: %ls (sha256 %s, file version %s), provider 0x%016llx \"%s\" (%zu offered)",
         g_fsr.runtime.mappedPath.c_str(), g_fsr.sha256.c_str(), g_fsr.fileVersion.c_str(),
         static_cast<unsigned long long>(g_fsr.runtime.providerId), g_fsr.runtime.providerName.c_str(),
         g_fsr.runtime.versionIds.size());
}

static bool fsr_rr_ready()
{
    return g_fsr.ready;
}

static const char* fsr_rr_denial_reason()
{
    return g_fsr.reason.c_str();
}

// Before Init there is no device to ask: offer the backend when its DLL is where it will be loaded from.
static bool fsr_rr_present()
{
    return file_exists(fsr_rr_dll_path());
}

// --- per-feature state ---------------------------------------------------------------

struct FsrRrTexture
{
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_DESC desc = {};
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL; // while in UNORDERED_ACCESS
};

// A view of one of the game's textures, kept for the feature's lifetime (the texture is AddRef'd).
struct FsrRrBorrowed
{
    ID3D12Resource* resource = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
};

struct FsrRrFeature
{
    enum Texture
    {
        LinearDepth,
        Motion,
        Normals,
        DiffuseAlbedo,
        SpecularAlbedo,
        Diffuse,
        Specular,
        DiffuseOut,
        SpecularOut,
        Residual,
        Composed,
        TextureCount
    };
    Feature* sr = nullptr;      // the internal Super Sampling feature
    void* srParams = nullptr;   // its per-frame parameters (ours)
    ffxContext context = nullptr;
    uint32_t signals = 0;       // FfxApiDenoiserSignalFlags the context was created with
    bool contextFailed = false;
    FsrRrTexture textures[TextureCount];
    bool texturesReady = false, texturesFailed = false;
    std::vector<FsrRrBorrowed> borrowed;
    float previousView[16] = {};
    bool havePrevious = false;
    std::string lastRefusal;
    uint32_t refusals = 0;
    bool configurationLogged = false;
    // D4R_PROFILE: GPU timestamps around the four stages, kSlots frames deep
    ID3D12QueryHeap* queries = nullptr;
    ID3D12Resource* queryReadback = nullptr;
    volatile uint64_t* queryValues = nullptr;
    double ticksPerMs = 0.0;
    bool queriesFailed = false;
    double stageSum[4] = {};
    uint32_t stageCount = 0;
    double cpuStageSum[4] = {};
    uint32_t cpuStageCount = 0;
};

static constexpr int kFsrRrStamps = 5; // before prepare, after prepare, after denoise, after compose, after SR

static NgxResult create_feature_locked(unsigned int featureId, void* parameters, NgxHandle** handle, bool nativeVulkan);
static NgxResult evaluate_unlocked(ID3D12GraphicsCommandList* list, const NgxHandle* handle, void* parameters);

// Feature 13 with Backend = fsr. Called from create_feature with the API lock and g.mutex held, after
// the feature's creation parameters were read.
static NgxResult fsr_rr_create(Feature& feature, void* /*parameters: read by create_feature already*/)
{
    if (!fsr_rr_ready())
        return NGX_FAIL_FEATURE_NOT_SUPPORTED;
    auto* fsr = new FsrRrFeature();
    feature.fsr = fsr;
    // The Super Sampling stage: an ordinary feature 1 at the same render and output sizes, with the game's
    // creation flags (HDR, motion-vector resolution and jitter, inverted depth) and the user's [DLSS] Model
    // (D4R_DLSS_PRESET applies to it as to any Super Sampling feature).
    fsr->srParams = d4r_ngx_parameters_create();
    void* sp = fsr->srParams;
    d4r_ngx_set_uint(sp, "Width", feature.width);
    d4r_ngx_set_uint(sp, "Height", feature.height);
    d4r_ngx_set_uint(sp, "OutWidth", feature.outWidth);
    d4r_ngx_set_uint(sp, "OutHeight", feature.outHeight);
    d4r_ngx_set_int(sp, "PerfQualityValue", feature.quality);
    d4r_ngx_set_int(sp, "DLSS.Feature.Create.Flags", feature.flags);
    d4r_ngx_set_int(sp, "DLSS.Enable.Output.Subrects", feature.outputSubrects);
    NgxHandle* srHandle = nullptr;
    const NgxResult created = create_feature_locked(NGX_FEATURE_SUPER_SAMPLING, sp, &srHandle, false);
    if (created != NGX_SUCCESS || srHandle == nullptr)
    {
        logf("FSR Ray Regeneration: the Super Sampling stage could not be created (0x%08x); feature 13 refused",
             created);
        return created != NGX_SUCCESS ? created : NGX_FAIL_PLATFORM_ERROR;
    }
    {
        std::lock_guard<std::mutex> lock(g_featuresMutex);
        for (Feature* candidate : g_features)
            if (&candidate->handle == srHandle)
                fsr->sr = candidate;
    }
    if (fsr->sr == nullptr)
        return NGX_FAIL_PLATFORM_ERROR;
    fsr->sr->owner = &feature;
    logf("Ray Reconstruction feature: requested backend fsr, serving with %s (provider 0x%016llx "
         "\"%s\"); Super Sampling by internal feature %u (preset %u); %ux%u -> %ux%u; switching backends needs the "
         "feature recreated",
         rr_backend_name(RrBackend::Fsr), static_cast<unsigned long long>(g_fsr.runtime.providerId),
         g_fsr.runtime.providerName.c_str(), fsr->sr->handle.Id, fsr->sr->preset, feature.width, feature.height, feature.outWidth, feature.outHeight);
    return NGX_SUCCESS;
}

// Called by release_feature before the feature itself drains: the internal Super Sampling feature goes
// first (it follows the same deferred path as any feature: its resources live until the GPU is done).
static void fsr_rr_release(Feature& feature)
{
    FsrRrFeature* fsr = feature.fsr;
    if (fsr == nullptr || fsr->sr == nullptr)
        return;
    Feature* sr = fsr->sr;
    fsr->sr = nullptr;
    sr->retiring.store(true);
    release_feature(sr);
}

// After the GPU has finished every command list that recorded this feature.
static void fsr_rr_destroy(Feature& feature)
{
    d4r_fsr4_identity::OriginalIdentityScope originalIdentity;
    FsrRrFeature* fsr = feature.fsr;
    if (fsr == nullptr)
        return;
    feature.fsr = nullptr;
    fsr_rr_dp::dump_flush();
    if (fsr->context != nullptr)
        g_fsr.runtime.destroyContext(&fsr->context, nullptr);
    for (FsrRrTexture& texture : fsr->textures)
    {
        if (texture.view != VK_NULL_HANDLE)
            g_fsr.passes.destroyView(g_fsr.vkDevice, texture.view, nullptr);
        if (texture.resource != nullptr)
            texture.resource->Release();
    }
    for (FsrRrBorrowed& view : fsr->borrowed)
    {
        g_fsr.passes.destroyView(g_fsr.vkDevice, view.view, nullptr);
        view.resource->Release();
    }
    if (fsr->srParams != nullptr)
        d4r_ngx_parameters_destroy(fsr->srParams);
    if (fsr->queries != nullptr)
        fsr->queries->Release();
    if (fsr->queryReadback != nullptr)
        fsr->queryReadback->Release();
    delete fsr;
}

static bool fsr_rr_passes_ready()
{
    std::call_once(g_fsr.passesOnce, [] {
        if (FAILED(g.device->QueryInterface(__uuidof(ID3D12DXVKInteropDevice1), reinterpret_cast<void**>(&g_fsr.interop))))
        {
            logf("FSR Ray Regeneration: vkd3d-proton's Vulkan interop is unavailable");
            return;
        }
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        if (FAILED(g_fsr.interop->GetVulkanHandles(&instance, &physical, &g_fsr.vkDevice)))
            return;
        HMODULE vulkan = GetModuleHandleA("vulkan-1.dll");
        if (vulkan == nullptr)
            vulkan = LoadLibraryA("vulkan-1.dll");
        PFN_vkGetInstanceProcAddr instanceProc = nullptr;
        if (vulkan == nullptr || !load_export(vulkan, "vkGetInstanceProcAddr", instanceProc))
            return;
        auto deviceProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(instanceProc(instance, "vkGetDeviceProcAddr"));
        auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            instanceProc(instance, "vkGetPhysicalDeviceProperties"));
        if (deviceProc == nullptr || properties == nullptr)
            return;
        VkPhysicalDeviceProperties limits = {};
        properties(physical, &limits);
        g_fsr.passesReady = g_fsr.passes.init(g_fsr.vkDevice, deviceProc, limits.limits.maxPushConstantsSize);
        logf("FSR Ray Regeneration: conversion and composition passes %s (push constants %u bytes)",
             g_fsr.passesReady ? "ready" : "could not be created", limits.limits.maxPushConstantsSize);
    });
    return g_fsr.passesReady;
}

static bool fsr_rr_create_texture(FsrRrTexture& texture, UINT width, UINT height, DXGI_FORMAT format)
{
    texture.desc = {};
    texture.desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.desc.Width = width;
    texture.desc.Height = height;
    texture.desc.DepthOrArraySize = 1;
    texture.desc.MipLevels = 1;
    texture.desc.Format = format;
    texture.desc.SampleDesc.Count = 1;
    texture.desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    texture.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if (FAILED(g.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture.desc, texture.state, nullptr,
                                                 __uuidof(ID3D12Resource), reinterpret_cast<void**>(&texture.resource))))
        return false;
    UINT64 handle = 0, offset = 0;
    VkFormat vkFormat = VK_FORMAT_UNDEFINED;
    if (FAILED(g_fsr.interop->GetVulkanResourceInfo1(texture.resource, &handle, &offset, &vkFormat)) || handle == 0)
        return false;
    g_fsr.interop->GetVulkanImageLayout(texture.resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &texture.layout);
    VkImageViewCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = reinterpret_cast<VkImage>(handle);
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = vkFormat;
    info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return g_fsr.passes.createView(g_fsr.vkDevice, &info, nullptr, &texture.view) == VK_SUCCESS;
}

static bool fsr_rr_create_textures(Feature& feature, FsrRrFeature& fsr)
{
    if (fsr.texturesReady || fsr.texturesFailed)
        return fsr.texturesReady;
    fsr.texturesFailed = true;
    const UINT w = feature.width, h = feature.height;
    for (int index = 0; index < FsrRrFeature::TextureCount; ++index)
    {
        const DXGI_FORMAT format = index == FsrRrFeature::LinearDepth ? DXGI_FORMAT_R32_FLOAT
                                                                      : DXGI_FORMAT_R16G16B16A16_FLOAT;
        if (!fsr_rr_create_texture(fsr.textures[index], w, h, format))
        {
            logf("FSR Ray Regeneration: intermediate texture %d (%ux%u) could not be created", index, w, h);
            return false;
        }
    }
    fsr.texturesFailed = false;
    fsr.texturesReady = true;
    return true;
}

// A sampled view of one of the game's textures (depth formats through their depth aspect).
static bool fsr_rr_borrow(FsrRrFeature& fsr, ID3D12Resource* resource, D3D12_RESOURCE_STATES state,
                          D4rFsrRrImage& image, VkFormat& format)
{
    UINT64 handle = 0, offset = 0;
    if (FAILED(g_fsr.interop->GetVulkanResourceInfo1(resource, &handle, &offset, &format)) || handle == 0)
        return false;
    const VkImageAspectFlags aspect =
        format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D16_UNORM ||
                format == VK_FORMAT_X8_D24_UNORM_PACK32 || format == VK_FORMAT_D24_UNORM_S8_UINT ||
                format == VK_FORMAT_D16_UNORM_S8_UINT
            ? VK_IMAGE_ASPECT_DEPTH_BIT
            : VK_IMAGE_ASPECT_COLOR_BIT;
    auto found = std::find_if(fsr.borrowed.begin(), fsr.borrowed.end(), [&](const FsrRrBorrowed& view) {
        return view.resource == resource && view.format == format && view.aspect == aspect;
    });
    if (found == fsr.borrowed.end())
    {
        if (fsr.borrowed.size() >= 64)
            return false; // a game cycling through more textures than this is not supported
        VkImageViewUsageCreateInfo usage = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        usage.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        VkImageViewCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &usage};
        info.image = reinterpret_cast<VkImage>(handle);
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = format;
        info.subresourceRange = {aspect, 0, 1, 0, 1};
        FsrRrBorrowed view;
        if (g_fsr.passes.createView(g_fsr.vkDevice, &info, nullptr, &view.view) != VK_SUCCESS)
            return false;
        resource->AddRef();
        view.resource = resource;
        view.format = format;
        view.aspect = aspect;
        fsr.borrowed.push_back(view);
        found = fsr.borrowed.end() - 1;
    }
    image.view = found->view;
    g_fsr.interop->GetVulkanImageLayout(resource, state, &image.layout);
    return true;
}

static bool fsr_rr_unorm(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_R16G16B16A16_UNORM:
        return true;
    default:
        return false;
    }
}

// Row-major, row-vector 4x4 helpers (the NGX/FFX convention), in double.
static bool fsr_rr_invert(const float (&m)[16], double (&out)[16])
{
    double a[16];
    for (int i = 0; i < 16; ++i)
        a[i] = m[i];
    double inv[16];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
    const double det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if (!std::isfinite(det) || std::fabs(det) < 1e-30)
        return false;
    for (int i = 0; i < 16; ++i)
    {
        out[i] = inv[i] / det;
        if (!std::isfinite(out[i]))
            return false;
    }
    return true;
}

static void fsr_rr_multiply(const double (&a)[16], const double (&b)[16], double (&out)[16])
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
        {
            double sum = 0;
            for (int k = 0; k < 4; ++k)
                sum += a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = sum;
        }
}

static bool fsr_rr_queries_ready(FsrRrFeature& fsr)
{
    if (fsr.queries != nullptr)
        return true;
    if (fsr.queriesFailed)
        return false;
    fsr.queriesFailed = true;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    UINT64 frequency = 0;
    if (SUCCEEDED(g.device->CreateCommandQueue(&queueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue))))
    {
        queue->GetTimestampFrequency(&frequency);
        queue->Release();
    }
    D3D12_QUERY_HEAP_DESC heapDesc = {};
    heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    heapDesc.Count = kFsrRrStamps * kOutputSlots;
    uint8_t* mapped = nullptr;
    if (frequency == 0 ||
        FAILED(g.device->CreateQueryHeap(&heapDesc, __uuidof(ID3D12QueryHeap), reinterpret_cast<void**>(&fsr.queries))))
    {
        fsr.queries = nullptr;
        return false;
    }
    if (!create_buffer(D3D12_HEAP_TYPE_READBACK, 8ull * kFsrRrStamps * kOutputSlots, &fsr.queryReadback, &mapped))
    {
        fsr.queries->Release();
        fsr.queries = nullptr;
        return false;
    }
    fsr.queryValues = reinterpret_cast<volatile uint64_t*>(mapped);
    for (int i = 0; i < kFsrRrStamps * kOutputSlots; ++i)
        fsr.queryValues[i] = 0;
    fsr.ticksPerMs = static_cast<double>(frequency) / 1000.0;
    fsr.queriesFailed = false;
    return true;
}

// The frame `kOutputSlots` back has certainly retired by the time the game records this one (the
// swapchain and the shim's in-flight cap bound the queue), so its slot can be read and reused.
static void fsr_rr_profile(FsrRrFeature& fsr, ID3D12GraphicsCommandList* list, uint32_t frame, int point, bool resolve)
{
    if (!fsr_rr_queries_ready(fsr))
        return;
    const UINT slot = frame % kOutputSlots;
    if (point == 0)
    {
        volatile uint64_t* values = fsr.queryValues + slot * kFsrRrStamps;
        if (values[0] != 0 && values[kFsrRrStamps - 1] > values[0])
        {
            for (int stage = 0; stage < 4; ++stage)
                fsr.stageSum[stage] += static_cast<double>(values[stage + 1] - values[stage]) / fsr.ticksPerMs;
            if (++fsr.stageCount == profile_log_every())
            {
                const double count = fsr.stageCount;
                if (profile_log_enabled())
                    logf("D4R_PROFILE FSR Ray Regeneration (mean of %u frames, GPU ms): prepare %.3f, denoise %.3f, "
                         "compose %.3f, super sampling %.3f, total %.3f",
                         fsr.stageCount, fsr.stageSum[0] / count, fsr.stageSum[1] / count, fsr.stageSum[2] / count,
                         fsr.stageSum[3] / count,
                         (fsr.stageSum[0] + fsr.stageSum[1] + fsr.stageSum[2] + fsr.stageSum[3]) / count);
                fsr.stageCount = 0;
                for (double& sum : fsr.stageSum)
                    sum = 0;
            }
        }
        for (int i = 0; i < kFsrRrStamps; ++i)
            values[i] = 0;
    }
    list->EndQuery(fsr.queries, D3D12_QUERY_TYPE_TIMESTAMP, slot * kFsrRrStamps + point);
    if (resolve)
        list->ResolveQueryData(fsr.queries, D3D12_QUERY_TYPE_TIMESTAMP, slot * kFsrRrStamps, kFsrRrStamps,
                               fsr.queryReadback, 8ull * slot * kFsrRrStamps);
}

// One frame of feature 13 on the FSR backend. Called from evaluate with the API lock held; the game's
// parameters are read here, on its thread, and nothing is kept from them past this call.
static NgxResult fsr_rr_evaluate(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                                 ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                                 ID3D12Resource* exposure, void* parameters, const FrameParams& p, uint32_t frame)
{
    d4r_fsr4_identity::OriginalIdentityScope originalIdentity;
    FsrRrFeature& fsr = *feature.fsr;
    const bool profile = profile_enabled();
    ProfileClock::time_point cpuStamp[5];
    if (profile)
        cpuStamp[0] = ProfileClock::now();
    auto refuse = [&](const std::string& reason, NgxResult result = NGX_FAIL_FEATURE_NOT_SUPPORTED) {
        // A frame that cannot be denoised is reported as failed, never upscaled as if it had been.
        if (reason != fsr.lastRefusal || ++fsr.refusals % 600 == 0)
            logf("FSR Ray Regeneration: feature %u frame %u not evaluated: %s", feature.handle.Id, frame, reason.c_str());
        fsr.lastRefusal = reason;
        return result;
    };
    if (fsr.sr == nullptr)
        return refuse("no Super Sampling stage");
    if (!fsr_rr_passes_ready())
        return refuse("the conversion passes are unavailable (vkd3d-proton interop)");
    if (g_vk.lifetime == nullptr || FAILED(g_vk.lifetime->RetainExternalResources(list, feature.resources)))
        return refuse("the command list cannot retain the feature's resources", NGX_FAIL_PLATFORM_ERROR);

    // The guides the conversion needs; the same names and fallbacks the native RR engine reads.
    ID3D12Resource* normals = get_resource(parameters, "GBuffer.Normals");
    ID3D12Resource* roughness = get_resource(parameters, "GBuffer.Roughness");
    ID3D12Resource* diffuseAlbedo = get_resource(parameters, "DLSS.Input.DiffuseAlbedo");
    if (diffuseAlbedo == nullptr)
        diffuseAlbedo = get_resource(parameters, "GBuffer.DiffuseAlbedo");
    ID3D12Resource* specularAlbedo = get_resource(parameters, "DLSS.Input.SpecularAlbedo");
    if (specularAlbedo == nullptr)
        specularAlbedo = get_resource(parameters, "GBuffer.SpecularAlbedo");
    // evaluate_unlocked resolves each scalar against the feature's creation baseline.
    // A game may use a fresh parameter object for evaluation and omit creation-only settings.
    const bool roughnessPacked = p.auxScalar[1] == 1 || roughness == nullptr; // DLSS.Roughness.Mode
    if (normals == nullptr || diffuseAlbedo == nullptr || specularAlbedo == nullptr)
        return refuse("the game did not pass normals and both albedos (GBuffer.Normals, DiffuseAlbedo, SpecularAlbedo)");
    bool specularInW = false, diffuseInW = false;
    ID3D12Resource* specularHit = get_resource(parameters, "DLSSD.SpecularHitDistance");
    if (specularHit == nullptr && (specularHit = get_resource(parameters, "DLSSD.SpecularRayDirectionHitDistance")))
        specularInW = true;
    ID3D12Resource* diffuseHit = get_resource(parameters, "DLSSD.DiffuseHitDistance");
    if (diffuseHit == nullptr && (diffuseHit = get_resource(parameters, "DLSSD.DiffuseRayDirectionHitDistance")))
        diffuseInW = true;
    ID3D12Resource* beforeParticles = get_resource(parameters, "DLSSD.ColorBeforeParticles");
    if (get_resource(parameters, "DLSSD.OutputAlpha") != nullptr)
        return refuse("DLSSD.OutputAlpha is requested, and the FSR backend does not produce alpha");
    if (!p.auxMatrixSet[0] || !p.auxMatrixSet[1])
        return refuse("the camera matrices (WorldToViewMatrix, ViewToClipMatrix) are missing");
    double clipToView[16], viewToWorld[16];
    if (!fsr_rr_invert(p.auxMatrix[1], clipToView) || !fsr_rr_invert(p.auxMatrix[0], viewToWorld))
        return refuse("a camera matrix is not invertible");
    // NVSDK_NGX_DLSS_Feature_Flags: MVLowRes 2, MVJittered 4, DepthInverted 8
    if ((feature.flags & 0x4) != 0)
        return refuse("jittered motion vectors (MVJittered) are not converted");
    for (size_t i = 0; i < sizeof(kAuxSubrectNames) / sizeof(kAuxSubrectNames[0]); ++i)
        if (p.auxSubrect[i][0] != 0 || p.auxSubrect[i][1] != 0)
            return refuse("guide subrect origins are not implemented");
    const UINT rw = p.renderWidth, rh = p.renderHeight;
    if (rw == 0 || rh == 0 || rw > feature.width || rh > feature.height)
        return refuse("the render size exceeds the size the feature was created with");
    if (!fsr_rr_create_textures(feature, fsr))
        return refuse("the intermediate textures could not be created", NGX_FAIL_PLATFORM_ERROR);

    // Which path each lobe takes: indirect (reading a ray length) only when the game supplies one.
    const std::string specularMode = env_string("D4R_FSR_RR_SPECULAR_SIGNAL");
    if (!specularMode.empty() && specularMode != "auto" && specularMode != "direct")
        return refuse("D4R_FSR_RR_SPECULAR_SIGNAL must be auto or direct");
    const bool useIndirectSpecular = specularHit != nullptr && specularMode != "direct";
    const uint32_t signals = (diffuseHit ? FFX_DENOISER_SIGNAL_INDIRECT_DIFFUSE : FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE) |
                             (useIndirectSpecular ? FFX_DENOISER_SIGNAL_INDIRECT_SPECULAR : FFX_DENOISER_SIGNAL_DIRECT_SPECULAR);
    if (fsr.context != nullptr && fsr.signals != signals)
    {
        // The game started or stopped passing a hit distance: the context's signal set no longer fits. The old
        // context may still be referenced by lists in flight, so it is destroyed only with the feature.
        return refuse("the game's hit-distance guides changed after the denoiser was created (recreate the feature)");
    }
    if (fsr.context == nullptr)
    {
        if (fsr.contextFailed)
            return refuse("the denoiser context could not be created");
        FfxApiEffectMemoryUsage memory = {};
        fsr_rr_query_memory(g_fsr.runtime, g.device, feature.width, feature.height, signals, 0, memory);
        fsr_rr_dp::hook_device(g.device);
        fsr_rr_dp::t_creating = true;
        const auto start = std::chrono::steady_clock::now();
        const ffxReturnCode_t created =
            fsr_rr_create_context(g_fsr.runtime, g.device, feature.width, feature.height, signals, 0, fsr.context);
        fsr_rr_dp::t_creating = false;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (created != FFX_API_RETURN_OK)
        {
            fsr.contextFailed = true;
            fsr.context = nullptr;
            return refuse(std::string("ffxCreateContext failed: ") + fsr_rr_return_name(created));
        }
        fsr.signals = signals;
        // AMD's tweakable settings (FfxApiConfigureDenoiserKey 1-6): the defaults are logged, and
        // D4R_FSR_RR_KEY_<n>=<float> overrides key n (diagnostics). [RayReconstruction] FsrStabilityBias
        // (D4R_FSR_RR_STABILITY_BIAS) sets key 2: at AMD's default of 1 its history blurs reflections and fine detail
        // (Alan Wake 2 water: 0.5x DLSS RR's detail); 0 halved the error against ground truth, moving or not.
        {
            static const char* const kKeyNames[7] = {"", "cross bilateral normal strength", "stability bias",
                                                     "max radiance", "radiance clip std k",
                                                     "gaussian kernel relaxation", "disocclusion threshold"};
            std::string defaults;
            for (uint64_t key = 1; key <= 6; ++key)
            {
                float value = 0.0f;
                ffxQueryDescDenoiserGetDefaultKeyValue query = {};
                query.header.type = FFX_API_QUERY_DESC_TYPE_DENOISER_GET_DEFAULT_KEYVALUE;
                query.key = key;
                query.count = 1;
                query.data = &value;
                char text[96];
                if (g_fsr.runtime.query(&fsr.context, &query.header) == FFX_API_RETURN_OK)
                    std::snprintf(text, sizeof(text), "%s%s %g", defaults.empty() ? "" : ", ", kKeyNames[key], value);
                else
                    std::snprintf(text, sizeof(text), "%s%s ?", defaults.empty() ? "" : ", ", kKeyNames[key]);
                defaults += text;
                char name[32];
                std::snprintf(name, sizeof(name), "D4R_FSR_RR_KEY_%u", static_cast<unsigned>(key));
                std::string override = env_string(name);
                if (override.empty() && key == 2)
                {
                    override = env_string("D4R_FSR_RR_STABILITY_BIAS");
                    if (ascii_lower(override) == "auto")
                        override.clear();
                }
                if (override.empty())
                    continue;
                char* end = nullptr;
                const float wanted = std::strtof(override.c_str(), &end);
                if (end == override.c_str() || !std::isfinite(wanted))
                {
                    logf("FSR Ray Regeneration: %s override '%s' is not a number; ignored", kKeyNames[key],
                         override.c_str());
                    continue;
                }
                ffxConfigureDescDenoiserKeyValue configure = {};
                configure.header.type = FFX_API_CONFIGURE_DESC_TYPE_DENOISER_KEYVALUE;
                configure.key = key;
                configure.count = 1;
                configure.data = &wanted;
                const ffxReturnCode_t configured = g_fsr.runtime.configure(&fsr.context, &configure.header);
                logf("FSR Ray Regeneration: %s set to %g (%s)", kKeyNames[key], wanted, fsr_rr_return_name(configured));
            }
            logf("FSR Ray Regeneration: denoiser defaults: %s", defaults.c_str());
        }
        logf("FSR Ray Regeneration: feature %u context created in %.0f ms at max %ux%u (GPU memory %.1f MB, %.1f MB "
             "aliasable); signals: diffuse %s, specular %s",
             feature.handle.Id, ms, feature.width, feature.height, memory.totalUsageInBytes / 1048576.0,
             memory.aliasableUsageInBytes / 1048576.0, diffuseHit ? "indirect (game hit distance)" : "direct",
             useIndirectSpecular ? "indirect (game hit distance)" : "direct");
    }

    // --- 1. conversion (Vulkan) ----------------------------------------------------------
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const D3D12_RESOURCE_STATES sampled = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    struct Input
    {
        ID3D12Resource* resource;
        D3D12_RESOURCE_STATES state;
    };
    ID3D12Resource* const roughnessSource = roughnessPacked ? normals : roughness;
    const Input inputs[D4rFsrRrPasses::kPrepareInputs] = {
        {color, inputState},
        {depth, depthState},
        {motion, inputState},
        {normals, inputState},
        {roughnessSource, inputState},
        {diffuseAlbedo, inputState},
        {specularAlbedo, inputState},
        {specularHit ? specularHit : normals, inputState},
        {diffuseHit ? diffuseHit : normals, inputState},
        {beforeParticles ? beforeParticles : color, inputState},
    };
    for (const Input& input : inputs)
        if (input.resource == output)
            return refuse("an input texture is also the output");
    D4rFsrRrImage images[D4rFsrRrPasses::kPrepareInputs + D4rFsrRrPasses::kPrepareOutputs];
    VkFormat formats[D4rFsrRrPasses::kPrepareInputs] = {};
    for (uint32_t i = 0; i < D4rFsrRrPasses::kPrepareInputs; ++i)
        if (!fsr_rr_borrow(fsr, inputs[i].resource, sampled, images[i], formats[i]))
            return refuse("a game texture cannot be viewed through Vulkan");
    static const int kPrepareTargets[D4rFsrRrPasses::kPrepareOutputs] = {
        FsrRrFeature::LinearDepth, FsrRrFeature::Motion, FsrRrFeature::Normals, FsrRrFeature::DiffuseAlbedo,
        FsrRrFeature::SpecularAlbedo, FsrRrFeature::Diffuse, FsrRrFeature::Specular, FsrRrFeature::Residual};
    for (uint32_t i = 0; i < D4rFsrRrPasses::kPrepareOutputs; ++i)
        images[D4rFsrRrPasses::kPrepareInputs + i] = {fsr.textures[kPrepareTargets[i]].view,
                                                       fsr.textures[kPrepareTargets[i]].layout};

    D4rFsrRrPrepareConstants c = {};
    for (int i = 0; i < 16; ++i)
        c.clipToView[i] = static_cast<float>(clipToView[i]);
    // A camera cut or the first frame has no previous view: the depth delta is then zero.
    const bool reset = p.reset != 0 || !fsr.havePrevious;
    double previousView[16], viewToPrevious[16];
    for (int i = 0; i < 16; ++i)
        previousView[i] = reset ? p.auxMatrix[0][i] : fsr.previousView[i];
    fsr_rr_multiply(viewToWorld, previousView, viewToPrevious);
    for (int i = 0; i < 16; ++i)
        c.viewToPreviousView[i] = static_cast<float>(viewToPrevious[i]);
    c.renderSize[0] = static_cast<float>(rw);
    c.renderSize[1] = static_cast<float>(rh);
    c.renderSize[2] = 1.0f / rw;
    c.renderSize[3] = 1.0f / rh;
    // Motion vectors at render resolution (MVLowRes) or at the output's; DLSS's are in that texture's pixels,
    // pointing from this frame to the previous one, scaled by MV.Scale.
    const bool lowResMotion = (feature.flags & 0x2) != 0;
    const float motionWidth = lowResMotion ? static_cast<float>(rw) : static_cast<float>(feature.outWidth);
    const float motionHeight = lowResMotion ? static_cast<float>(rh) : static_cast<float>(feature.outHeight);
    c.motion[0] = p.mvScaleX / motionWidth;
    c.motion[1] = p.mvScaleY / motionHeight;
    c.motion[2] = motionWidth / rw;
    c.motion[3] = motionHeight / rh;
    c.colorDepthBase[0] = p.colorBaseX;
    c.colorDepthBase[1] = p.colorBaseY;
    c.colorDepthBase[2] = p.depthBaseX;
    c.colorDepthBase[3] = p.depthBaseY;
    c.motionBaseFlags[0] = p.mvBaseX;
    c.motionBaseFlags[1] = p.mvBaseY;
    const bool hardwareDepth = p.auxScalar[2] != 0; // DLSS.Use.HW.Depth (resolved creation/frame value)
    uint32_t flags = 0;
    flags |= hardwareDepth ? kFsrRrHardwareDepth : 0u;
    flags |= roughnessPacked ? kFsrRrRoughnessPacked : 0u;
    flags |= specularHit ? uint32_t(kFsrRrSpecularHit) | (specularInW ? uint32_t(kFsrRrSpecularHitInW) : 0u) : 0u;
    flags |= diffuseHit ? uint32_t(kFsrRrDiffuseHit) | (diffuseInW ? uint32_t(kFsrRrDiffuseHitInW) : 0u) : 0u;
    flags |= fsr_rr_unorm(formats[3]) ? kFsrRrNormalsUnorm : 0u;
    flags |= beforeParticles ? kFsrRrBeforeParticles : 0u;
    flags |= env_uint("D4R_FSR_RR_ZERO_NORMAL_PASSTHROUGH", 1) ? kFsrRrInvalidNormalPassthrough : 0u;
    c.motionBaseFlags[2] = flags;
    // Row-vector projection: clip.w = view.z * ViewToClip[2][3], positive in front of the camera, so the sign
    // of that element is the sign of view z there (+1 left-handed, -1 right-handed).
    const float viewZSign = p.auxMatrix[1][11] < 0.0f ? -1.0f : 1.0f;
    c.depthParams[0] = viewZSign;
    c.depthParams[1] = (feature.flags & 0x8) != 0 ? 0.0f : 1.0f; // the hardware depth of the sky
    c.depthParams[2] = 1.0e6f;                                    // linear depth at or beyond this is sky
    // [RayReconstruction] FsrDetail (D4R_FSR_RR_DETAIL, default 0): the share of the colour that goes around the
    // denoiser, keeping the fine detail it removes as noise. D4R_FSR_RR_DEMOD_FLOOR (default 0.1): the smallest albedo
    // a signal is divided by; below it, demodulation turned fog and edge light into clipped or spread signals.
    {
        const std::string detail = env_string("D4R_FSR_RR_DETAIL");
        const float share = detail.empty() ? 0.0f : std::strtof(detail.c_str(), nullptr);
        if (!(share >= 0.0f && share <= 1.0f))
            return refuse("D4R_FSR_RR_DETAIL must be between 0 and 1");
        const std::string demodFloor = env_string("D4R_FSR_RR_DEMOD_FLOOR");
        const float floorValue = demodFloor.empty() ? 0.1f : std::strtof(demodFloor.c_str(), nullptr);
        c.detail[0] = share;
        c.detail[1] = floorValue > 1e-3f ? floorValue : 1e-3f;
    }
    if (!fsr.configurationLogged)
    {
        fsr.configurationLogged = true;
        logf("FSR Ray Regeneration: feature %u inputs: depth %s%s, motion %s-resolution (scale %.3f,%.3f), normals %s "
             "(assumed world space), roughness %s, specular hit distance %s, diffuse hit distance %s, colour before "
             "particles %s, view z %s in front of the camera; formats colour %d depth %d motion %d normals %d",
             feature.handle.Id, hardwareDepth ? "hardware" : "linear", (feature.flags & 0x8) ? " (inverted)" : "",
             lowResMotion ? "render" : "output", p.mvScaleX, p.mvScaleY, (flags & kFsrRrNormalsUnorm) ? "[0,1]" : "signed",
             roughnessPacked ? "packed in the normals' w" : "texture", specularHit ? "bound" : "none",
             diffuseHit ? "bound" : "none", beforeParticles ? "bound (particles bypass the denoiser)" : "none",
             viewZSign > 0 ? "positive" : "negative", formats[0], formats[1], formats[2], formats[3]);
        logf("FSR Ray Regeneration: feature %u signal split: detail share %.2f, demodulation floor %.3f, halo limit %.2f",
             feature.handle.Id, c.detail[0], c.detail[1], haloLimitSetting());
    }

    if (profile)
        fsr_rr_profile(fsr, list, frame, 0, false);
    // The game's inputs are sampled in NON_PIXEL_SHADER_RESOURCE; ours are written as UAVs.
    std::vector<ID3D12Resource*> transitioned;
    auto to_sampled = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES from) {
        if (std::find(transitioned.begin(), transitioned.end(), resource) != transitioned.end())
            return;
        transitioned.push_back(resource);
        transition(list, resource, from, sampled);
    };
    for (const Input& input : inputs)
        to_sampled(input.resource, input.state);
    auto set_state = [&](int index, D3D12_RESOURCE_STATES state) {
        FsrRrTexture& texture = fsr.textures[index];
        transition(list, texture.resource, texture.state, state);
        texture.state = state;
    };
    for (int index = 0; index < FsrRrFeature::TextureCount; ++index)
        set_state(index, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (FAILED(g_fsr.interop->BeginVkCommandBufferInterop(list, &command)))
        return refuse("BeginVkCommandBufferInterop failed", NGX_FAIL_PLATFORM_ERROR);
    g_fsr.passes.record_prepare(command, images, c, rw, rh);
    g_fsr.interop->EndVkCommandBufferInterop(list);
    // the game's textures back to the states it handed them over in
    for (ID3D12Resource* resource : transitioned)
        for (const Input& input : inputs)
            if (input.resource == resource)
            {
                transition(list, resource, sampled, input.state);
                break;
            }
    if (profile)
        fsr_rr_profile(fsr, list, frame, 1, false);
    if (profile)
        cpuStamp[1] = ProfileClock::now();

    // --- 2. AMD FSR Ray Regeneration (D3D12) ------------------------------------------------
    const D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    for (int index : {FsrRrFeature::LinearDepth, FsrRrFeature::Motion, FsrRrFeature::Normals, FsrRrFeature::DiffuseAlbedo,
                      FsrRrFeature::SpecularAlbedo, FsrRrFeature::Diffuse, FsrRrFeature::Specular})
        set_state(index, read);
    auto ffx = [&](int index, uint32_t state) {
        const FsrRrTexture& texture = fsr.textures[index];
        return fsr_rr_texture(texture.resource, texture.desc, fsr_rr_surface_format(texture.desc.Format), state);
    };
    const uint32_t ffxRead = FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ;
    {
        ID3D12Resource* const inputs[7] = {
            fsr.textures[FsrRrFeature::LinearDepth].resource, fsr.textures[FsrRrFeature::Motion].resource,
            fsr.textures[FsrRrFeature::Normals].resource,     fsr.textures[FsrRrFeature::DiffuseAlbedo].resource,
            fsr.textures[FsrRrFeature::SpecularAlbedo].resource, fsr.textures[FsrRrFeature::Diffuse].resource,
            fsr.textures[FsrRrFeature::Specular].resource};
        static const char* const names[7] = {"in_depth", "in_motion", "in_normals", "in_diffuse_albedo",
                                             "in_specular_albedo", "in_diffuse", "in_specular"};
        fsr_rr_dp::dump_record(0, g.device, list, frame, 7, inputs, names, read);
    }
    ffxDispatchDescDenoiser desc = {};
    desc.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER;
    desc.commandList = list;
    desc.linearDepth = ffx(FsrRrFeature::LinearDepth, ffxRead);
    desc.motionVectors = ffx(FsrRrFeature::Motion, ffxRead);
    desc.normals = ffx(FsrRrFeature::Normals, ffxRead);
    desc.specularAlbedo = ffx(FsrRrFeature::SpecularAlbedo, ffxRead);
    desc.diffuseAlbedo = ffx(FsrRrFeature::DiffuseAlbedo, ffxRead);
    desc.motionVectorScale = {1.0f, 1.0f, 1.0f}; // the conversion wrote UV and view-z deltas
    desc.jitterOffsets = {p.jitterX, p.jitterY};  // render pixels, as DLSS's
    // camera position = the inverse view's translation row; delta = previous - current
    const double position[3] = {viewToWorld[12], viewToWorld[13], viewToWorld[14]};
    double previousPosition[3] = {position[0], position[1], position[2]};
    if (!reset)
    {
        double previousInverse[16];
        if (fsr_rr_invert(fsr.previousView, previousInverse))
            previousPosition[0] = previousInverse[12], previousPosition[1] = previousInverse[13],
            previousPosition[2] = previousInverse[14];
    }
    desc.cameraPositionDelta = {static_cast<float>(previousPosition[0] - position[0]),
                                static_cast<float>(previousPosition[1] - position[1]),
                                static_cast<float>(previousPosition[2] - position[2])};
    std::memcpy(&desc.view, p.auxMatrix[0], sizeof(desc.view));       // row-major, row vectors: as FFX wants
    std::memcpy(&desc.projection, p.auxMatrix[1], sizeof(desc.projection));
    desc.linearDepthBounds = {0.0f, 1.0e6f};
    desc.renderSize = {rw, rh};
    desc.frameIndex = frame;
    desc.flags = uint32_t(FFX_DENOISER_DISPATCH_NON_GAMMA_ALBEDO) | (reset ? uint32_t(FFX_DENOISER_DISPATCH_RESET) : 0u);
    ffxDispatchDescDenoiserDirectDiffuse directDiffuse = {};
    ffxDispatchDescDenoiserIndirectDiffuse indirectDiffuse = {};
    ffxDispatchDescDenoiserDirectSpecular directSpecular = {};
    ffxDispatchDescDenoiserIndirectSpecular indirectSpecular = {};
    FfxApiDenoiserSignal* diffuseSignal;
    FfxApiDenoiserSignal* specularSignal;
    ffxApiHeader *diffuseHeader, *specularHeader;
    if (diffuseHit)
    {
        indirectDiffuse.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_INDIRECT_DIFFUSE;
        diffuseSignal = &indirectDiffuse.signal, diffuseHeader = &indirectDiffuse.header;
    }
    else
    {
        directDiffuse.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_DIFFUSE;
        diffuseSignal = &directDiffuse.signal, diffuseHeader = &directDiffuse.header;
    }
    if (useIndirectSpecular)
    {
        indirectSpecular.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_INDIRECT_SPECULAR;
        specularSignal = &indirectSpecular.signal, specularHeader = &indirectSpecular.header;
    }
    else
    {
        directSpecular.header.type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DIRECT_SPECULAR;
        specularSignal = &directSpecular.signal, specularHeader = &directSpecular.header;
    }
    diffuseSignal->input = ffx(FsrRrFeature::Diffuse, ffxRead);
    diffuseSignal->output = ffx(FsrRrFeature::DiffuseOut, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    specularSignal->input = ffx(FsrRrFeature::Specular, ffxRead);
    specularSignal->output = ffx(FsrRrFeature::SpecularOut, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    desc.header.pNext = diffuseHeader;
    diffuseHeader->pNext = specularHeader;
    const bool skipDispatch = env_uint("D4R_FSR_RR_SKIP_DISPATCH", 0) != 0;
    ffxReturnCode_t dispatched = FFX_API_RETURN_OK;
    if (skipDispatch)
    {
        // Diagnostic only: retain the same game RR request and allocations, but remove AMD's
        // neural work. This distinguishes game RR-path cost from denoiser cost.
        for (const auto pair : {std::pair{FsrRrFeature::Diffuse, FsrRrFeature::DiffuseOut},
                               std::pair{FsrRrFeature::Specular, FsrRrFeature::SpecularOut}})
        {
            set_state(pair.first, D3D12_RESOURCE_STATE_COPY_SOURCE);
            set_state(pair.second, D3D12_RESOURCE_STATE_COPY_DEST);
            list->CopyResource(fsr.textures[pair.second].resource, fsr.textures[pair.first].resource);
            set_state(pair.first, read);
            set_state(pair.second, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (frame == 1)
            logf("FSR Ray Regeneration: DIAGNOSTIC dispatch bypass active; output is not denoised");
    }
    else
    {
        fsr_rr_dp::begin(g.device, list, frame);
        fsr_rr_dp::t_creating = fsr_rr_dp::enabled();
        dispatched = g_fsr.runtime.dispatch(&fsr.context, &desc.header);
        fsr_rr_dp::t_creating = false;
        fsr_rr_dp::end(list);
    }
    if (dispatched != FFX_API_RETURN_OK)
        return refuse(std::string("ffxDispatch failed: ") + fsr_rr_return_name(dispatched), NGX_FAIL_PLATFORM_ERROR);
    // The denoiser's writes before the composition's reads.
    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &uav);
    {
        ID3D12Resource* const denoised[2] = {fsr.textures[FsrRrFeature::DiffuseOut].resource,
                                             fsr.textures[FsrRrFeature::SpecularOut].resource};
        static const char* const names[2] = {"out_diffuse", "out_specular"};
        fsr_rr_dp::dump_record(1, g.device, list, frame, 2, denoised, names, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    if (profile)
        fsr_rr_profile(fsr, list, frame, 2, false);
    if (profile)
        cpuStamp[2] = ProfileClock::now();

    // --- 3. composition (Vulkan) -------------------------------------------------------------
    for (int index = 0; index < FsrRrFeature::TextureCount; ++index)
        set_state(index, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const D4rFsrRrImage compose[D4rFsrRrPasses::kComposeImages] = {
        {fsr.textures[FsrRrFeature::DiffuseOut].view, fsr.textures[FsrRrFeature::DiffuseOut].layout},
        {fsr.textures[FsrRrFeature::SpecularOut].view, fsr.textures[FsrRrFeature::SpecularOut].layout},
        {fsr.textures[FsrRrFeature::DiffuseAlbedo].view, fsr.textures[FsrRrFeature::DiffuseAlbedo].layout},
        {fsr.textures[FsrRrFeature::SpecularAlbedo].view, fsr.textures[FsrRrFeature::SpecularAlbedo].layout},
        {fsr.textures[FsrRrFeature::Residual].view, fsr.textures[FsrRrFeature::Residual].layout},
        {fsr.textures[FsrRrFeature::Composed].view, fsr.textures[FsrRrFeature::Composed].layout},
        {fsr.textures[FsrRrFeature::Diffuse].view, fsr.textures[FsrRrFeature::Diffuse].layout},
        {fsr.textures[FsrRrFeature::Specular].view, fsr.textures[FsrRrFeature::Specular].layout},
    };
    const float haloLimit = haloLimitSetting();
    if (FAILED(g_fsr.interop->BeginVkCommandBufferInterop(list, &command)))
        return refuse("BeginVkCommandBufferInterop failed", NGX_FAIL_PLATFORM_ERROR);
    g_fsr.passes.record_compose(command, compose, rw, rh, haloLimit);
    g_fsr.interop->EndVkCommandBufferInterop(list);
    if (profile)
        fsr_rr_profile(fsr, list, frame, 3, false);
    if (profile)
        cpuStamp[3] = ProfileClock::now();

    std::memcpy(fsr.previousView, p.auxMatrix[0], sizeof(fsr.previousView));
    fsr.havePrevious = true;

    // --- 4. Super Sampling to the output size -------------------------------------------------
    // The composed colour is handed over the way a game hands DLSS its colour (D4R_SHIM_INPUT_STATE).
    set_state(FsrRrFeature::Composed, inputState);
    void* sp = fsr.srParams;
    d4r_ngx_set_d3d12_resource(sp, "Color", fsr.textures[FsrRrFeature::Composed].resource);
    d4r_ngx_set_d3d12_resource(sp, "Depth", depth);
    d4r_ngx_set_d3d12_resource(sp, "MotionVectors", motion);
    d4r_ngx_set_d3d12_resource(sp, "Output", output);
    d4r_ngx_set_d3d12_resource(sp, "ExposureTexture", exposure);
    d4r_ngx_set_float(sp, "Jitter.Offset.X", p.jitterX);
    d4r_ngx_set_float(sp, "Jitter.Offset.Y", p.jitterY);
    d4r_ngx_set_float(sp, "MV.Scale.X", p.mvScaleX);
    d4r_ngx_set_float(sp, "MV.Scale.Y", p.mvScaleY);
    d4r_ngx_set_float(sp, "Sharpness", p.sharpness);
    d4r_ngx_set_float(sp, "DLSS.Pre.Exposure", p.preExposure);
    d4r_ngx_set_float(sp, "DLSS.Exposure.Scale", p.exposureScale);
    d4r_ngx_set_float(sp, "FrameTimeDeltaInMsec", p.frameTime);
    d4r_ngx_set_int(sp, "Reset", p.reset);
    d4r_ngx_set_uint(sp, "DLSS.Render.Subrect.Dimensions.Width", rw);
    d4r_ngx_set_uint(sp, "DLSS.Render.Subrect.Dimensions.Height", rh);
    d4r_ngx_set_uint(sp, "DLSS.Input.Color.Subrect.Base.X", 0);
    d4r_ngx_set_uint(sp, "DLSS.Input.Color.Subrect.Base.Y", 0);
    d4r_ngx_set_uint(sp, "DLSS.Input.Depth.Subrect.Base.X", p.depthBaseX);
    d4r_ngx_set_uint(sp, "DLSS.Input.Depth.Subrect.Base.Y", p.depthBaseY);
    d4r_ngx_set_uint(sp, "DLSS.Input.MV.Subrect.Base.X", p.mvBaseX);
    d4r_ngx_set_uint(sp, "DLSS.Input.MV.Subrect.Base.Y", p.mvBaseY);
    d4r_ngx_set_uint(sp, "DLSS.Output.Subrect.Base.X", p.outputBaseX);
    d4r_ngx_set_uint(sp, "DLSS.Output.Subrect.Base.Y", p.outputBaseY);
    d4r_ngx_set_int(sp, "DLSS.Indicator.Invert.X.Axis", p.invertX);
    d4r_ngx_set_int(sp, "DLSS.Indicator.Invert.Y.Axis", p.invertY);
    const NgxResult upscaled = evaluate_unlocked(list, &fsr.sr->handle, sp);
    if (profile)
        fsr_rr_profile(fsr, list, frame, 4, true);
    if (profile && frame > 5)
    {
        cpuStamp[4] = ProfileClock::now();
        for (int stage = 0; stage < 4; ++stage)
            fsr.cpuStageSum[stage] += profile_ms(cpuStamp[stage], cpuStamp[stage + 1]);
        if (++fsr.cpuStageCount == profile_log_every())
        {
            const double count = fsr.cpuStageCount;
            if (profile_log_enabled())
                logf("D4R_PROFILE FSR CPU (mean of %u frames, recording ms): prepare %.3f, denoise %.3f, "
                     "compose %.3f, super sampling %.3f, total %.3f",
                     fsr.cpuStageCount, fsr.cpuStageSum[0] / count, fsr.cpuStageSum[1] / count,
                     fsr.cpuStageSum[2] / count, fsr.cpuStageSum[3] / count,
                     (fsr.cpuStageSum[0] + fsr.cpuStageSum[1] + fsr.cpuStageSum[2] + fsr.cpuStageSum[3]) / count);
            fsr.cpuStageCount = 0;
            for (double& sum : fsr.cpuStageSum)
                sum = 0;
        }
    }
    if (upscaled != NGX_SUCCESS)
        return refuse("the Super Sampling stage failed", upscaled);
    if (frame <= 3 || frame % 600 == 0)
        logf("FSR Ray Regeneration frame %u recorded for feature %u: %ux%u -> %ux%u (reset %d, jitter %.4f,%.4f)",
             frame, feature.handle.Id, rw, rh, feature.outWidth, feature.outHeight, reset ? 1 : 0, p.jitterX, p.jitterY);
    fsr.lastRefusal.clear();
    return NGX_SUCCESS;
}
