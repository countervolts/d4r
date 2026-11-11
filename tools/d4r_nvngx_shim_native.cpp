// d4r_nvngx_shim_native.cpp: the native engine backend (engine/). Runs the networks through the
// Vulkan engine instead of NGX: no NGX evaluation, CUDA, worker thread or frame split.

// the command-list stubs live in d4r_nvngx_shim.cpp, which assembles them once
#define D4R_D3D12_SPLIT_DECL 1
#include "d4r_nvngx_shim_internal.h"

bool engine_requested()
{
    static const bool requested = env_uint("D4R_ENGINE", 0) != 0;
    return requested && !g_engineGpuOff.load();
}

// [Engine] Enabled = auto (D4R_ENGINE_K=0) keeps K on the native kernels only on RDNA3: there they beat the Vulkan
// network, on RDNA2 (no f16 matrix tuple) they emulate WMMA and on RDNA4 (FP8 matrix tuple) the engine is faster
// (2026-10-21, RX 9070 XT, preset K HDR: 1920x1080 -> 3840x2160 2.93 ms vs 9.19 ms, 960x540 -> 1920x1080 0.95 ms
// vs 4.35 ms on the game queue). Needs init_engine_vulkan.

bool engine_k_native_rdna3()
{
    return env_string("D4R_ENGINE_K") == "0" && env_string("D4R_ENGINE_NETWORK") != "hip" && g_vk.matrixF16 &&
           !g_vk.float8;
}

// ... except through vkhip (on by default): there K runs on the engine when vkhip loads, and on the native
// kernels when it does not (engine_evaluate).

bool engine_k_refused()
{
    return engine_k_native_rdna3() && env_uint("D4R_ENGINE_VKHIP", 1) == 0;
}

// Presets M and L on the engine's v_dot2 stages (layer_m_dot2.comp, layer_l_dot2.comp): on GPUs without FP8 and
// without the f16 matrix tuple (RDNA2), where the CUDA backend's M and L kernels are translated. RDNA3 keeps its
// native kernels unless D4R_ENGINE_M_NETWORK=dot2.

static bool engine_m_dot2()
{
    return !g_vk.float8 && (!g_vk.matrixF16 || env_string("D4R_ENGINE_M_NETWORK") == "dot2");
}

// The engine's arithmetic on this GPU: the f16 matrix tuple (RDNA3/4) and, with VK_EXT_shader_float8 and FP8
// matrices, the FP8 one (RDNA4). Shared by the vkd3d-proton and native Vulkan routes.

void detect_engine_matrices(PFN_vkGetInstanceProcAddr instanceProc, VkInstance instance, VkPhysicalDevice physical)
{
    auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        instanceProc(instance, "vkEnumerateDeviceExtensionProperties"));
    uint32_t count = 0;
    if (enumerate != nullptr && enumerate(physical, nullptr, &count, nullptr) == VK_SUCCESS)
    {
        std::vector<VkExtensionProperties> extensions(count);
        if (enumerate(physical, nullptr, &count, extensions.data()) == VK_SUCCESS)
            for (uint32_t index = 0; index < count; ++index)
                g_vk.float8 |= std::strcmp(extensions[index].extensionName, "VK_EXT_shader_float8") == 0;
    }
    auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(instanceProc(instance, "vkGetPhysicalDeviceFeatures2"));
    auto matrices = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        instanceProc(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    bool fp8 = false;
    if (features2 && matrices)
    {
        VkPhysicalDeviceShaderFloat8FeaturesEXT f8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        cm.pNext = g_vk.float8 ? &f8 : nullptr;
        VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &cm};
        features2(physical, &features);
        uint32_t n = 0;
        if (cm.cooperativeMatrix && matrices(physical, &n, nullptr) == VK_SUCCESS)
        {
            std::vector<VkCooperativeMatrixPropertiesKHR> properties(n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
            if (matrices(physical, &n, properties.data()) == VK_SUCCESS)
                for (uint32_t i = 0; i < n; ++i)
                {
                    g_vk.matrixF16 |= d4r_matrix_tuple(properties[i], VK_COMPONENT_TYPE_FLOAT16_KHR);
                    fp8 |= d4r_matrix_tuple(properties[i], VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT);
                }
        }
        fp8 = fp8 && f8.shaderFloat8 && f8.shaderFloat8CooperativeMatrix;
    }
    g_vk.float8 = g_vk.float8 && fp8;
}

void init_engine_vulkan()
{
    VkInstance instance = VK_NULL_HANDLE;
    if (g_vk.interop == nullptr || g_vk.lifetime == nullptr ||
        FAILED(g_vk.interop->GetVulkanHandles(&instance, &g_vk.physical, &g_vk.device)))
    {
        logf("engine backend: the vkd3d-proton interop interfaces are unavailable");
        return;
    }
    HMODULE vulkan = GetModuleHandleA("vulkan-1.dll");
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    if (vulkan == nullptr || !load_export(vulkan, "vkGetInstanceProcAddr", instanceProc) ||
        !d4r::loadVulkan(instanceProc, instance, g_vk.device))
    {
        logf("engine backend: Vulkan entry points are unavailable");
        return;
    }
    detect_engine_matrices(instanceProc, instance, g_vk.physical);
    logf("engine backend: f16 matrix tuple %s; FP8 matrix tuple %s", g_vk.matrixF16 ? "supported" : "unsupported",
         g_vk.float8 ? "supported" : "unsupported");
    logf("engine backend: VK_EXT_shader_float8 %s", g_vk.float8 ? "available (M and L engine layers)"
         : engine_m_dot2() ? "unavailable: M and L run their v_dot2 stages" : "unavailable: M and L run on the CUDA backend");
    // the features vkd3d-proton created its device with
    const VkPhysicalDeviceFeatures2* features = nullptr;
    if (SUCCEEDED(g_vk.interop->GetDeviceFeatures(&features)) && features != nullptr)
        for (auto* next = static_cast<const VkBaseInStructure*>(features->pNext); next != nullptr; next = next->pNext)
        {
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                g_vk.int64Atomics = reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(next)->shaderBufferInt64Atomics == VK_TRUE;
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES)
                g_vk.int64Atomics = reinterpret_cast<const VkPhysicalDeviceShaderAtomicInt64Features*>(next)->shaderBufferInt64Atomics == VK_TRUE;
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT)
                g_vk.robustImages = reinterpret_cast<const VkPhysicalDeviceRobustness2FeaturesEXT*>(next)->robustImageAccess2 == VK_TRUE;
        }
    logf("engine backend: 64-bit buffer atomics %s", g_vk.int64Atomics ? "enabled" : "not enabled: preset F's exposure in two passes");
    logf("engine backend: robustImageAccess2 %s", g_vk.robustImages ? "enabled" : "not enabled");
}

void init_engine_backend()
{
    // SetEnvironmentVariable (d4r.ini) does not reach the C runtime's getenv
    std::string directory = env_string("D4R_ENGINE_MODEL_DIR");
    if (directory.empty())
    {
        char folder[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (g_portable.dir + L"\\engine\\k").c_str(), -1, folder, sizeof(folder), nullptr, nullptr);
        directory = folder;
    }
    if ((!g_vk.native && (g_vk.interop == nullptr || g_vk.lifetime == nullptr)) || g_vk.device == VK_NULL_HANDLE || d4r::vkTable.CreateBuffer == nullptr)
        return;
    // A Unix path names the same file through Wine's Z: drive.
    const std::string path = directory[0] == '/' ? "Z:" + directory : directory;
    const auto start = std::chrono::steady_clock::now();
    try
    {
        // without the f16 matrix tuple (RDNA2) the network runs layer_dot2.comp's shaders
        g_engine.model = std::make_unique<d4r::Model>(g_vk.physical, g_vk.device, path, VK_NULL_HANDLE,
                                                      g_vk.matrixF16 ? d4r::KNetwork::Matrix : d4r::KNetwork::Dot2,
                                                      env_uint("D4R_PREFER_ACCURACY", 0) != 0);
        g_engine.directory = path;
    }
    catch (const std::exception& error)
    {
        logf("engine backend: cannot load the model from %s: %s (see D4R_README.txt / engine/README.md for how to "
             "create it)", path.c_str(), error.what());
        return;
    }
    logf("engine backend: model %s loaded and pipelines compiled in %.0f ms (network: %s)", path.c_str(),
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
         g_engine.model->dot2Network() ? "v_dot2, no cooperative matrices" : "cooperative matrices");
    logf("engine backend: K reconstruction uses %s stages", g_engine.model->hasAccuracyStages() ? "accuracy" : "fast");
}

void init_engine_backend_ldr()
{
    std::string directory = env_string("D4R_ENGINE_MODEL_LDR_DIR");
    if (directory.empty())
    {
        char folder[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (g_portable.dir + L"\\engine\\k-ldr").c_str(), -1, folder, sizeof(folder), nullptr, nullptr);
        directory = folder;
    }
    if ((!g_vk.native && (g_vk.interop == nullptr || g_vk.lifetime == nullptr)) || g_vk.device == VK_NULL_HANDLE || d4r::vkTable.CreateBuffer == nullptr)
        return;
    const std::string path = directory[0] == '/' ? "Z:" + directory : directory;
    try
    {
        g_engine.modelLdr = std::make_unique<d4r::Model>(g_vk.physical, g_vk.device, path, VK_NULL_HANDLE,
                                                         g_vk.matrixF16 ? d4r::KNetwork::Matrix : d4r::KNetwork::Dot2,
                                                         env_uint("D4R_PREFER_ACCURACY", 0) != 0);
        g_engine.directoryLdr = path;
        logf("engine backend: LDR model %s loaded", path.c_str());
        logf("engine backend: K LDR reconstruction uses %s stages", g_engine.modelLdr->hasAccuracyStages() ? "accuracy" : "fast");
    }
    catch (const std::exception& error)
    {
        logf("engine backend: cannot load the LDR model from %s: %s", path.c_str(), error.what());
    }
}

// K and M borrow compatible views through the same bounded, resource-owning cache.

void prepare_engine_view(Feature& feature, ID3D12Resource* resource, const D3D12_RESOURCE_DESC& desc,
                                d4r::GameImage& image, bool isOutput, bool originSupported,
                                D3D12_RESOURCE_STATES& state)
{
    if (image.format == VK_FORMAT_D32_SFLOAT || image.format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
        image.format == VK_FORMAT_D16_UNORM || image.format == VK_FORMAT_D16_UNORM_S8_UINT ||
        image.format == VK_FORMAT_D24_UNORM_S8_UINT || image.format == VK_FORMAT_X8_D24_UNORM_PACK32)
        image.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    const VkImageUsageFlags usage = isOutput ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT;
    VkFormatProperties properties{};
    d4r::vkTable.GetPhysicalDeviceFormatProperties(g_vk.physical, image.format, &properties);
    const bool enabled = env_uint(isOutput ? "D4R_ENGINE_DIRECT_OUTPUT" : "D4R_ENGINE_DIRECT_INPUTS", 1) != 0;
    const bool compatible = desc.DepthOrArraySize == 1 && originSupported && (isOutput ?
        // the final-store formats (K packages a store per format; other models copy what they cannot store)
        (image.format == VK_FORMAT_R16G16B16A16_SFLOAT || image.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 ||
         image.format == VK_FORMAT_R8G8B8A8_UNORM || image.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) &&
            (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) &&
            (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) :
        !(desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) &&
            (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT));
    if (!enabled || !compatible)
        return;
    auto found = std::find_if(feature.engineViews.begin(), feature.engineViews.end(), [&](const auto& view) {
        return view->resource == resource && view->format == image.format && view->aspect == image.aspect &&
            view->usage == usage;
    });
    if (found == feature.engineViews.end())
    {
        VkMemoryRequirements memory{};
        d4r::vkTable.GetImageMemoryRequirements(g_vk.device, image.image, &memory);
        constexpr uint64_t cacheBudget = 512ull * 1024 * 1024;
        // Exposure has no copy path, so a game that cycles its render targets must not cost it a view: tiny images
        // sit outside the budget, which then only limits the textures that can fall back to copies.
        const bool tiny = memory.size <= 64 * 1024;
        if (tiny ? feature.engineSmallViews < 1024 :
            feature.engineViews.size() - feature.engineSmallViews < 64 && memory.size <= cacheBudget - feature.engineViewBytes)
        {
            auto view = std::make_unique<EngineBorrowedView>();
            view->device = g_vk.device; view->format = image.format;
            view->aspect = image.aspect; view->usage = usage;
            VkImageViewUsageCreateInfo restricted{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO}; restricted.usage = usage;
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &restricted};
            info.image = image.image; info.viewType = VK_IMAGE_VIEW_TYPE_2D; info.format = view->format;
            info.subresourceRange = {view->aspect, 0, 1, 0, 1};
            if (d4r::vkTable.CreateImageView(g_vk.device, &info, nullptr, &view->view) == VK_SUCCESS)
            {
                resource->AddRef(); view->resource = resource;
                feature.engineViews.push_back(std::move(view));
                if (tiny) ++feature.engineSmallViews;
                else feature.engineViewBytes += memory.size;
                found = feature.engineViews.end() - 1;
            }
        }
        else if (!feature.engineViewBudgetReported)
        {
            feature.engineViewBudgetReported = true;
            logf("engine view cache budget reached; new textures use copies until feature retirement");
        }
    }
    if (found != feature.engineViews.end())
    {
        image.view = (*found)->view;
        state = isOutput ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
}

void init_engine_backend_m_variant(bool presetL, bool ldr)
{
    const char* const name = presetL ? "L" : "M";
    std::unique_ptr<d4r::MModel>& slot = g_engineM.models[presetL][ldr];
    std::string directory = env_string(presetL ? (ldr ? "D4R_ENGINE_MODEL_L_LDR_DIR" : "D4R_ENGINE_MODEL_L_DIR")
                                               : (ldr ? "D4R_ENGINE_MODEL_M_LDR_DIR" : "D4R_ENGINE_MODEL_M_DIR"));
    if (directory.empty())
    {
        char folder[MAX_PATH * 3] = {};
        const wchar_t* const sub = presetL ? (ldr ? L"\\engine\\l-ldr" : L"\\engine\\l") : (ldr ? L"\\engine\\m-ldr" : L"\\engine\\m");
        WideCharToMultiByte(CP_ACP, 0, (g_portable.dir + sub).c_str(), -1, folder, sizeof(folder), nullptr, nullptr);
        directory = folder;
    }
    if ((!g_vk.native && (g_vk.interop == nullptr || g_vk.lifetime == nullptr)) || g_vk.device == VK_NULL_HANDLE || d4r::vkTable.CreateBuffer == nullptr)
    {
        logf("engine backend (%s): Vulkan interop is unavailable", name);
        return;
    }
    const std::string path = directory[0] == '/' ? "Z:" + directory : directory;
    const auto start = std::chrono::steady_clock::now();
    try
    {
        // without FP8 (RDNA2) M and L run their v_dot2 stages (layer_m_dot2.comp, layer_l_dot2.comp)
        auto model = std::make_unique<d4r::MModel>(g_vk.physical, g_vk.device, path, VK_NULL_HANDLE, !g_vk.float8);
        if (model->isPresetL() != presetL)
            throw std::runtime_error(presetL ? "the directory holds a preset M model" : "the directory holds a preset L model");
        slot = std::move(model);
        g_engineM.directories[presetL][ldr] = path;
    }
    catch (const std::exception& error)
    {
        logf("engine backend (%s): cannot load the model from %s: %s", name, path.c_str(), error.what());
        return;
    }
    logf("engine backend (%s): model %s loaded and pipelines compiled in %.0f ms", name, path.c_str(),
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

static EngineNetExports g_engineNet;

VkResult signal_split_semaphore(Feature& feature, uint64_t value)
{
    if (feature.splitFence != nullptr)
        return SUCCEEDED(feature.splitFence->Signal(value)) ? VK_SUCCESS : VK_ERROR_DEVICE_LOST;
    if (feature.engineNetTimeline)
        return static_cast<VkResult>(g_engineNet.signalTimeline(feature.engineNet, value));
    if (feature.ngxTimeline != nullptr)
        return static_cast<VkResult>(g.cu.timelineSignal(feature.ngxTimeline, value));
    VkSemaphoreSignalInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    info.semaphore = feature.splitSemaphore;
    info.value = value;
    return g_vk.signalSemaphore(g_vk.device, &info);
}

static bool read_file(const std::string& path, std::vector<char>& bytes)
{
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr)
        return false;
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    bytes.resize(size > 0 ? static_cast<size_t>(size) : 0);
    const bool ok = size > 0 && fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
    fclose(file);
    return ok;
}

// A storage buffer the engine's shaders and HIP both use. Pooled like the interop buffers, for the same ROCm leak.

static bool create_engine_shared_buffer(VramBuffer& target, size_t bytes, bool host = false)
{
    const uint8_t kind = host ? 2 : 1;
    if (take_pooled_vram_buffer(target, bytes, kind))
    {
        if (target.mapped != nullptr)
        {
            std::memset(target.mapped, 0, target.bytes);
            MemoryBarrier();
        }
        return true;
    }
    VramBuffer buffer;
    buffer.kind = kind;
    VkExternalMemoryBufferCreateInfo external = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = g_vk.createBuffer(g_vk.device, &info, nullptr, &buffer.buffer);
    VkMemoryRequirements requirements = {};
    if (result == VK_SUCCESS)
        g_vk.bufferRequirements(g_vk.device, buffer.buffer, &requirements);
    int type = device_local_type(requirements.memoryTypeBits);
    if (host)
    {
        type = -1;
        const auto coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (int pass = 0; pass < 2 && type < 0; ++pass)
            for (uint32_t index = 0; index < g_vk.memory.memoryTypeCount; ++index)
            {
                const auto flags = g_vk.memory.memoryTypes[index].propertyFlags;
                if ((requirements.memoryTypeBits & (1u << index)) && (flags & coherent) == coherent &&
                    (pass != 0 || !(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
                {
                    type = static_cast<int>(index);
                    break;
                }
            }
    }
    VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = buffer.buffer;
    VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dedicated};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exportInfo};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = static_cast<uint32_t>(type);
    if (result == VK_SUCCESS && type < 0)
        result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (result == VK_SUCCESS)
        result = g_vk.allocate(g_vk.device, &allocation, nullptr, &buffer.memory);
    if (result == VK_SUCCESS)
        result = g_vk.bindBuffer(g_vk.device, buffer.buffer, buffer.memory, 0);
    if (result == VK_SUCCESS && host)
        result = g_vk.map(g_vk.device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped);
    int imported = -1;
    if (result == VK_SUCCESS)
        imported = g.worker.call([&] {
            return g_vk.import(g_vk.device, reinterpret_cast<uint64_t>(buffer.memory), requirements.size, &buffer.device,
                               &buffer.external);
        });
    if (result != VK_SUCCESS || imported != 0)
    {
        logf("engine backend: shared buffer of %zu bytes failed (VkResult %d, import %d)", bytes, result, imported);
        if (buffer.mapped != nullptr)
            g_vk.unmap(g_vk.device, buffer.memory);
        if (buffer.memory != VK_NULL_HANDLE)
            g_vk.free(g_vk.device, buffer.memory, nullptr);
        if (buffer.buffer != VK_NULL_HANDLE)
            g_vk.destroyBuffer(g_vk.device, buffer.buffer, nullptr);
        return false;
    }
    if (buffer.mapped != nullptr)
    {
        std::memset(buffer.mapped, 0, bytes);
        MemoryBarrier();
    }
    buffer.bytes = bytes;
    target = buffer;
    return true;
}

bool initialize_ngx_gpu_wait(Feature& feature)
{
    if (env_uint("D4R_SHIM_GPU_INPUT_WAIT", 1) == 0)
        return false;
    // Age-zero output only; keep queue depth and the existing split-frame contract unchanged.
    if (!feature.split || feature.rayReconstruction || g_vk.fill == nullptr ||
        g.cu.streamWaitValue32Eq == nullptr ||
        !create_engine_shared_buffer(feature.ngxInputSync, 256, true))
    {
        logf("NGX feature %u: GPU-side input wait unavailable (needs split VRAM, Super Sampling, coherent export "
             "and the matching bridge); using CPU marker polling", feature.handle.Id);
        return false;
    }
    static std::once_flag watchdog;
    std::call_once(watchdog, [] { std::thread(ngx_input_watchdog).detach(); });
    feature.ngxGpuWait.store(true);
    // D4R_SHIM_GPU_COMPLETION (default 1): the game's queue waits for DLSS on the GPU, polling the frame's word
    // in the same coherent buffer, instead of a split list released by a CPU timeline signal (one host round trip,
    // ~0.1 ms, per frame). The pool is never freed: game command lists still reference the set after retirement.
    if (env_uint("D4R_SHIM_GPU_COMPLETION", 1) != 0)
    {
        if (feature.completionSet.set == VK_NULL_HANDLE && g_waitReady && g.cu.ngxWriteValue32 != nullptr)
            feature.gpuCompletion = g_waitStage.wait_set(feature.completionSet, feature.ngxInputSync.buffer, 256);
        else
            feature.gpuCompletion = feature.completionSet.set != VK_NULL_HANDLE && g.cu.ngxWriteValue32 != nullptr;
        for (uint32_t index = 0; index < kOutputSlots; ++index)
            static_cast<volatile uint32_t*>(feature.ngxInputSync.mapped)[kCompletionWord + index] = 0;
        logf("NGX feature %u: GPU-side completion %s", feature.handle.Id,
             feature.gpuCompletion ? "enabled (the game's queue polls DLSS's completion word)"
                                   : "unavailable (wait shader or bridge export missing); using the split timeline");
    }
    return true;
}

// The game's queue holds here until DLSS wrote this frame's completion word (or the bounded poll gives up).

bool record_gpu_completion_wait(Feature& feature, ID3D12GraphicsCommandList* list, uint32_t frame)
{
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &command)))
        return false;
    g_waitStage.record_wait(command, feature.completionSet, kCompletionWord + frame % kOutputSlots, frame,
                            kCompletionPollLimit);
    // Everything after the wait, including the output copy, sees DLSS's writes.
    const VkMemoryBarrier after = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
                                   VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    g_vk.barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after,
                 0, nullptr, 0, nullptr);
    return SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
}

bool record_ngx_input_ready(Feature& feature, ID3D12GraphicsCommandList* list, int slotIndex, uint32_t frame)
{
    if (!feature.ngxGpuWait.load())
        return false;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &command)))
        return false;
    // All input copies/conversions precede this release marker. The imported inputs and the shared word
    // use the same memory barrier protocol as the engine's GPU handoff. Keep this before the list split:
    // placing the fill in its second half would deadlock that half against its own NGX input wait.
    const VkMemoryBarrier before = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                    VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.barrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before,
                 0, nullptr, 0, nullptr);
    g_vk.fill(command, feature.ngxInputSync.buffer, sizeof(uint32_t) * (1 + slotIndex), sizeof(uint32_t), frame);
    const VkMemoryBarrier after = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    g_vk.barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after,
                 0, nullptr, 0, nullptr);
    return SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
}

// Only once no command list uses the engine and no network launch is pending.

void release_engine_network(Feature& feature)
{
    feature.engineVkHip.reset();
    if (feature.engineNet != nullptr && g_engineNet.destroy != nullptr)
        g.worker.call([&feature] {
            g_engineNet.destroy(feature.engineNet);
            return 0;
        });
    feature.engineNet = nullptr;
    feature.engineNetTimeline = false;
    for (VramBuffer* buffer : {&feature.engineTokens, &feature.engineHead, &feature.engineSync})
        recycle_vram_buffer(*buffer);
}

// The HIP network of a feature: the buffers shared with HIP and the bridge's network over them. False: the
// engine's own Vulkan network is used instead (reason logged). `preset` is "K", "M" or "L"; `width`
// and `height` are the dimensions the bridge derives the token grid from (M: render, K: output); `tokenBytes`
// and `headBytes` size the two shared buffers (M: both MEngine::networkBufferBytes, K: Engine::networkTokenBytes
// and Engine::networkHeadBytes). The layer kernel names come from hipnet.bin, so no preset list is repeated here.

bool create_engine_network(Feature& feature, const std::string& directory, const char* preset,
                                  bool stagesAvailable, unsigned int width, unsigned int height, size_t tokenBytes,
                                  size_t headBytes)
{
    // the Vulkan network unless [Engine] Network = hip asks for the native layers (2026-10-14: the shader with
    // M_RELAXED and two windows per wave is faster than the HIP layers, engine/README.md)
    const std::string choice = env_string("D4R_ENGINE_NETWORK");
    if (choice != "hip")
        return false;
    auto unavailable = [&](const char* reason) {
        logf("engine backend (%s): HIP network unavailable (%s); using the Vulkan network", preset, reason);
        release_engine_network(feature);
        return false;
    };
    std::call_once(g_engineNet.once, [] {
        g_engineNet.ready = load_export(g.cuda, "d4rEngineNetCreate", g_engineNet.create) &&
                            load_export(g.cuda, "d4rEngineNetLaunch", g_engineNet.launch) &&
                            load_export(g.cuda, "d4rEngineNetDone", g_engineNet.done) &&
                            load_export(g.cuda, "d4rEngineNetDestroy", g_engineNet.destroy) &&
                            load_export(g.cuda, "d4rEngineNetInputWaitReady", g_engineNet.inputWaitReady) &&
                            load_export(g.cuda, "d4rEngineNetLaunchAt", g_engineNet.launchAt) &&
                            load_export(g.cuda, "d4rEngineNetReleaseInput", g_engineNet.releaseInput) &&
                            load_export(g.cuda, "d4rEngineNetSetTimeline", g_engineNet.setTimeline) &&
                            load_export(g.cuda, "d4rEngineNetSignalTimeline", g_engineNet.signalTimeline);
    });
    if (!g_engineNet.ready)
        return unavailable("the CUDA bridge lacks d4rEngineNet*");
    if (!stagesAvailable)
        return unavailable("the model has no stages for the native network; rebuild it with --hip");
    if (g_vk.split == nullptr || g_vk.import == nullptr || feature.marker == nullptr || feature.markerValue == nullptr)
        return unavailable("split frames need the d4r vkd3d-proton and VRAM interop");
    std::vector<char> net;
    if (!read_file(directory + "\\hipnet.bin", net))
        return unavailable("no hipnet.bin in the model directory");
    // The bridge dispatches on the magic alone, and the shared buffers are sized for `preset`.
    const char* const magic = std::strcmp(preset, "K") == 0 ? "D4RKHIP1" : "D4RMHIP1";
    if (net.size() < 16 || std::memcmp(net.data(), magic, 8) != 0)
        return unavailable("hipnet.bin is not a native network of this preset");
    uint32_t count = 0;
    std::memcpy(&count, net.data() + 8, sizeof(count));
    if (count == 0 || net.size() < 16 + 64ull * count)
        return unavailable("invalid hipnet.bin");
    // The kernel names live in the layer table; load the code object of each distinct one (the M tube repeats).
    std::vector<std::string> names;
    names.reserve(count);
    for (uint32_t index = 0; index < count; ++index)
    {
        char name[33] = {};
        std::memcpy(name, net.data() + 16 + 64 * index, 32);
        if (name[0] == '\0')
            return unavailable("invalid kernel name in hipnet.bin");
        if (std::find(names.begin(), names.end(), name) == names.end())
            names.emplace_back(name);
    }
    std::vector<std::vector<char>> code(names.size());
    std::vector<EngineNetCode> codes(names.size());
    for (size_t index = 0; index < names.size(); ++index)
    {
        if (!read_file(directory + "\\hip\\" + names[index] + ".hsaco", code[index]))
            return unavailable("a layer kernel is missing in the model's hip folder");
        codes[index] = {names[index].c_str(), code[index].data(), code[index].size()};
    }
    if (!create_engine_shared_buffer(feature.engineTokens, tokenBytes) ||
        !create_engine_shared_buffer(feature.engineHead, headBytes) ||
        !create_engine_shared_buffer(feature.engineSync, 256, true))
        return unavailable("shared buffers");
    const int created = g.worker.call([&] {
        return g_engineNet.create(reinterpret_cast<const uint8_t*>(net.data()), net.size(), codes.data(),
                                  static_cast<uint32_t>(codes.size()), width, height, feature.engineTokens.device,
                                  feature.engineHead.device, &feature.engineNet);
    });
    if (created != 0 || feature.engineNet == nullptr)
    {
        feature.engineNet = nullptr;
        logf("engine backend (%s): d4rEngineNetCreate failed (%d)", preset, created);
        return unavailable("the kernels could not be loaded for this GPU");
    }
    // Input readiness on the GPU: the engine's front fills engineSync with the frame number (the shim records the
    // fill right after recordFront) and the network's stream waits for it, so the layers start the moment the
    // front is complete, without the CPU round trip through the readback marker. The bridge checks HIP's
    // capability and a round trip on this buffer. Without wait-value support (or D4R_SHIM_ENGINE_GPU_WAIT=0)
    // the worker keeps waiting for the CPU-visible marker before launching.
    const bool gpuWaitEnabled = env_uint("D4R_SHIM_ENGINE_GPU_WAIT", 1) != 0;
    const int waitProbe = gpuWaitEnabled ? g.worker.call([&] {
        return g_engineNet.inputWaitReady(feature.engineNet, feature.engineSync.device,
                                         static_cast<uint32_t*>(feature.engineSync.mapped));
    }) : 0;
    if (waitProbe < 0)
        return unavailable("the HIP stream cannot complete a wait-value test");
    feature.engineGpuWait = waitProbe == 1;
    logf("engine backend (%s): HIP network input readiness on %s", preset, feature.engineGpuWait.load()
             ? "the GPU (the front publishes the frame number; the layers wait on it)"
             : "the CPU (marker polling)");
    if (feature.splitSemaphore == VK_NULL_HANDLE && !create_split_semaphore(feature))
        return unavailable("timeline semaphore");
    if (g.worker.call([&] {
            return g_engineNet.setTimeline(feature.engineNet, g_vk.device,
                                          reinterpret_cast<uint64_t>(feature.splitSemaphore));
        }) != 0)
        return unavailable("native HIP completion timeline");
    feature.engineNetTimeline = true;
    logf("engine backend (%s): network completion releases the split timeline from the native HIP callback", preset);
    return true;
}

// Records the end of the front: a fill of this frame's number into the buffer the HIP stream waits on
// (d4rEngineNetLaunchAt). The engine's recordFront ends the input stage with its own ALL_COMMANDS barrier (the
// tokens are then available); the fill is ordered after it and, once it lands, the ten layers may read them.

void record_engine_input_ready(Feature& feature, VkCommandBuffer command, uint32_t frame)
{
    const VkMemoryBarrier before = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                    VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.barrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr,
                 0, nullptr);
    g_vk.fill(command, feature.engineSync.buffer, 0, sizeof(uint32_t), frame);
    const VkMemoryBarrier after = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    g_vk.barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr,
                 0, nullptr);
}

// The network's inputs are written by the front of the game's list, so its stream waits for the frame's fill of
// the shared u32 (d4rEngineNetLaunchAt). A wait for a frame whose front never reached the GPU (the game recorded
// its list but did not execute it, or the game stalled) can only be satisfied by a later frame's fill: writing
// the value it waits for releases it, so the stream keeps moving. The frame's output is undefined either way.
// `disable`: the GPU-side wait itself looks broken for this feature, so later frames wait for the CPU marker.

void release_engine_input_wait(Feature& feature, uint32_t frame, const char* reason, bool disable)
{
    if (feature.engineNet == nullptr || g_engineNet.releaseInput == nullptr)
        return;
    const int result = g_engineNet.releaseInput(feature.engineNet, frame);
    if (disable)
        feature.engineGpuWait.store(false);
    logf("engine frame %u: %s; releasing the GPU-side input wait (%d)", frame, reason, result);
}

// CUDA worker: runs the network of `frame` (behind the GPU-side input wait, or once the CPU marker shows the
// front has run), then lets the second part of the game's command list (the engine's reconstruction) run.

void run_engine_network(Feature* feature, int slotIndex, uint32_t frame)
{
    const bool gpuWait = feature->engineGpuWait.load() && g_engineNet.launchAt != nullptr;
    if (!gpuWait && !wait_for_input_marker(feature, slotIndex, frame))
        return; // dropped and retired there
    const auto start = std::chrono::steady_clock::now();
    int launched;
    if (gpuWait)
    {
        feature->engineWaitFrame.store(frame);
        launched = g_engineNet.launchAt(feature->engineNet, frame);
        if (launched == 801 /* CUDA_ERROR_NOT_SUPPORTED: no wait was queued */)
        {
            feature->engineWaitFrame.store(0);
            // The bridge could not queue the wait (no hipStreamWaitValue32, or the stream rejected it): wait
            // for the input marker on the CPU, as without the GPU-side wait.
            feature->engineGpuWait.store(false);
            logf("engine frame %u: GPU-side input wait unavailable (%d); waiting for the input marker on the CPU", frame, launched);
            if (!wait_for_input_marker(feature, slotIndex, frame))
                return;
            launched = g_engineNet.launch(feature->engineNet, frame);
        }
    }
    else
        launched = g_engineNet.launch(feature->engineNet, frame);
    float gpuMs = 0.0f;
    int done = launched == 0 ? 0 : -1;
    bool released = false, discarded = false;
    // Completion polling (as the CUDA path's output event): sleep while far from the predicted completion,
    // yield around it (a Wine sleep rounds up to ~50 us), back off when the GPU runs late. The prediction
    // learns from the earlier frames; with the GPU-side wait it also covers the front's remaining GPU time.
    double lastNotReadyUs = -1.0;
    while (done == 0)
    {
        done = g_engineNet.done(feature->engineNet, &gpuMs);
        if (done != 0)
            break;
        const double elapsedUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        if (gpuWait && !released)
        {
            // This frame's own marker (written with the aggregate one, at the end of its front) tells whether
            // its front ran at all. A later frame's front running while this one's never did means the game
            // discarded this list: its wait can only be satisfied by that later front, which would make the
            // network read another frame's tokens. Release it and drop the frame, as the CPU marker path does.
            const uint32_t slotMarker = feature->markerValue[1 + slotIndex];
            if (slotMarker != frame && static_cast<int32_t>(*feature->markerValue - frame) > 0)
            {
                release_engine_input_wait(*feature, frame, "its command list was discarded", false);
                discarded = true;
                break;
            }
            // The front ran but the stream's wait did not start the network (a broken wait would leave every
            // later frame queued behind it): release it and wait for the CPU marker from now on.
            if (slotMarker == frame && elapsedUs > 1000000.0)
            {
                release_engine_input_wait(*feature, frame, "the GPU-side wait did not start the network", true);
                released = true;
            }
        }
        if (elapsedUs > 2000000.0)
        {
            done = -2;
            break;
        }
        lastNotReadyUs = elapsedUs;
        constexpr double kMarginUs = 120.0, kStepUs = 20.0, kChunkUs = 200.0, kSpinAfterUs = 150.0;
        const double predictedUs = feature->engineNetWaitEmaUs;
        const double remaining = predictedUs - kMarginUs - elapsedUs;
        if (remaining > kStepUs)
            d4r_sleep_us(static_cast<unsigned>(std::min(remaining, kChunkUs)));
        else if (elapsedUs < predictedUs + kSpinAfterUs)
            d4r_sleep_us(0); // polling retires resources; the native callback releases the GPU's split wait
        else
            d4r_sleep_us(static_cast<unsigned>(std::min(
                kStepUs * (1.0 + (elapsedUs - predictedUs - kSpinAfterUs) / 160.0), kChunkUs)));
    }
    feature->engineWaitFrame.store(0);
    const double waitedUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    if (done == 1)
    {
        // Learn the completion time, not our own wait: the work finished between the last NOT_READY poll and the
        // successful one (at most one chunk apart), so the midpoint cannot ratchet upwards.
        double sample = lastNotReadyUs >= 0.0 ? 0.5 * (lastNotReadyUs + waitedUs) : waitedUs;
        sample = std::min(sample, 20000.0);
        feature->engineNetWaitEmaUs = feature->engineNetWaitEmaUs == 0.0
                                          ? sample : 0.8 * feature->engineNetWaitEmaUs + 0.2 * sample;
    }
    if (discarded)
        logf("engine frame %u: the game discarded its command list; the frame is dropped", frame);
    else if (done != 1)
        logf("engine frame %u: HIP network failed (launch %d, state %d); the frame's output is undefined", frame, launched, done);
    else if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u: HIP network %.2f ms GPU, %.2f ms %s", frame, gpuMs, waitedUs / 1000.0,
             gpuWait ? "from launch to completion (GPU-side input wait)" : "from input readiness to completion");
    if (gpuWait && !discarded && done != 1)
        release_engine_input_wait(*feature, frame, "the network failed before completing its input wait", true);
    feature->inputs[slotIndex].busy = false;
    frame_retired(feature, frame);
}

std::unique_ptr<vkhip::MNetwork> create_engine_vkhip_net(const std::string& directory, uint32_t renderWidth,
                                                                uint32_t renderHeight, uint64_t bufferBytes);

// Presets M (13) and L (12): record the evaluation into `list`, borrowing compatible inputs and copying only
// fallbacks. L is M's pipeline with its own input, expansion and reconstruction stages (engine/L_NOTES.md).

bool engine_evaluate_m(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                              ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                              const FrameParams& p, uint32_t frame)
{
    const bool presetL = feature.preset == 12;
    const char* const name = presetL ? "L" : "M";
    auto refuse = [&](const char* reason) {
        feature.engineRefused = true;
        logf("engine backend (%s): %s; feature %u uses the CUDA backend from frame %u", name, reason, feature.handle.Id, frame);
        return false;
    };
    // NVSDK_NGX_DLSS_Feature_Flags: IsHDR 1, MVLowRes 2, MVJittered 4, DepthInverted 8, AutoExposure 0x40.
    const bool ldr = (feature.flags & 0x1) == 0; // NGX's LDR stages: a model directory of its own, no exposure
    // The FP8 pipelines need VK_EXT_shader_float8: creating them without it crashes the driver (RADV on RDNA3,
    // vkCreateComputePipelines), so never load such a model. M and L load their v_dot2 stages instead where
    // engine_m_dot2().
    if (!g_vk.float8 && !engine_m_dot2())
        return refuse("the GPU has no VK_EXT_shader_float8 for the FP8 layers");
    std::call_once(g_engineM.once[presetL][ldr], [presetL, ldr] { init_engine_backend_m_variant(presetL, ldr); });
    d4r::MModel* const model = g_engineM.models[presetL][ldr].get();
    if (model == nullptr)
    {
        if (!ldr)
            return refuse("not available");
        return refuse(presetL ? "no LDR model (engine\\l-ldr) for a game without the HDR flag"
                              : "no LDR model (engine\\m-ldr) for a game without the HDR flag");
    }
    // MVLowRes clear: display-resolution vectors (one per output pixel, in output pixels), NGX's *_mvhi_* kernels
    const bool displayMotion = (feature.flags & 0x2) == 0;
    // (L: its v_dot2 stages only; its FP8 stages have no such variant)
    if (displayMotion && !model->hasDisplayMotion())
        return refuse(presetL ? "display-resolution motion vectors need the v_dot2 stages (no FP8) and the current compile_m.py"
                              : "display-resolution motion vectors need an M model rebuilt by the current compile_m.py");
    if (!model->hasDynamicResolution() && feature.engineM != nullptr &&
        (p.renderWidth != feature.engineMRenderWidth || p.renderHeight != feature.engineMRenderHeight))
        return refuse("dynamic render sizes need a model rebuilt by the current compile_m.py");
    // NGX measures the exposure for this preset, so the AutoExposure flag, the game's exposure texture, the
    // pre-exposure and the exposure scale are all ignored here (CUDA output is identical for each of them).
    if (color == output || depth == output || motion == output)
        return refuse("input/output texture aliasing is not supported");
    d4r::GameFrameM game;
    game.profile = engine_stage_profile(frame, name);
    using Source = EngineFrameSource;
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    Source sources[4] = {
        {color, &game.color, p.colorBaseX, p.colorBaseY, p.renderWidth, p.renderHeight, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {depth, &game.depth, p.depthBaseX, p.depthBaseY, p.renderWidth, p.renderHeight, depthState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {motion, &game.motion, p.mvBaseX, p.mvBaseY, displayMotion ? feature.outWidth : p.renderWidth,
         displayMotion ? feature.outHeight : p.renderHeight, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {output, &game.output, p.outputBaseX, p.outputBaseY, feature.outWidth, feature.outHeight, outputState,
         D3D12_RESOURCE_STATE_COPY_DEST},
    };
    for (Source& source : sources)
    {
        UINT64 handle = 0, offset = 0;
        D3D12_RESOURCE_DESC desc;
        source.resource->GetDesc(&desc);
        if (FAILED(g_vk.interop->GetVulkanResourceInfo1(source.resource, &handle, &offset, &source.image->format)) ||
            handle == 0 || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
            source.x + source.width > desc.Width || source.y + source.height > desc.Height)
            return refuse("a texture is not a single-sample 2D image covering its rectangle");
        source.image->image = reinterpret_cast<VkImage>(handle);
        source.image->x = source.x;
        source.image->y = source.y;
        prepare_engine_view(feature, source.resource, desc, *source.image, source.image == &game.output,
                            source.x == 0 && source.y == 0, source.copyState);
        g_vk.interop->GetVulkanImageLayout(source.resource, source.copyState, &source.image->layout);
    }
    if (game.depth.format == VK_FORMAT_D32_SFLOAT || game.depth.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        game.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!d4r::gameColorFormat(game.color.format) || !d4r::gameMotionFormat(game.motion.format) ||
        !d4r::gameDepthFormat(game.depth.format, game.depth.aspect) || !d4r::gameOutputFormat(game.output.format))
    {
        logf("engine backend (%s): Vulkan formats colour %d, depth %d, motion %d, output %d", name, game.color.format,
             game.depth.format, game.motion.format, game.output.format);
        return refuse("a texture format is not implemented");
    }
    game.jitter[0] = p.jitterX;
    game.jitter[1] = p.jitterY;
    game.motionToRender[0] = p.mvScaleX;
    game.motionToRender[1] = p.mvScaleY;
    game.displayMotion = displayMotion;
    if ((feature.flags & 0x4) != 0 && p.mvScaleX != 0.0f && p.mvScaleY != 0.0f)
    {
        game.motionOffset[0] = (p.jitterX - feature.engineJitter[0]) / p.mvScaleX;
        game.motionOffset[1] = (p.jitterY - feature.engineJitter[1]) / p.mvScaleY;
    }
    game.renderWidth = p.renderWidth;
    game.renderHeight = p.renderHeight;
    game.reset = p.reset != 0 || feature.engineM == nullptr ||
        p.renderWidth != feature.engineMLastWidth || p.renderHeight != feature.engineMLastHeight;
    game.depthInverted = (feature.flags & 0x8) != 0;
    try
    {
        if (feature.engineM == nullptr)
        {
            const size_t bytes = static_cast<size_t>(d4r::MEngine::networkBufferBytes(p.renderWidth, p.renderHeight));
            const bool hip = create_engine_network(feature, g_engineM.directories[presetL][ldr], name,
                                                   model->hasExternalNetworkStages(), p.renderWidth, p.renderHeight,
                                                   bytes, bytes);
            // vkhip (D4R_ENGINE_VKHIP): M's and L's native layers in the game's command list, on a fixed grid of
            // this frame's render size; other sizes go to engineMDynamic
            if (!hip && model->hasExternalNetworkStages())
                feature.engineMVkHip = create_engine_vkhip_net(g_engineM.directories[presetL][ldr], p.renderWidth,
                                                               p.renderHeight, bytes);
            const bool vkhip = feature.engineMVkHip != nullptr;
            const d4r::ExternalNetwork external = vkhip ?
                d4r::ExternalNetwork{feature.engineMVkHip->tokens(), feature.engineMVkHip->head()} :
                d4r::ExternalNetwork{feature.engineTokens.buffer, feature.engineHead.buffer};
            // The Vulkan network of a dynamic-resolution model reserves the render size the feature was created with
            // (NGX's maximum; at least this frame's), not the output size: at 1280x720 -> 4K that is 1.1 GB less for
            // M and 1.4 GB for L. A larger frame goes to the output-sized engine below (engineMDynamic).
            const bool dynamic = !hip && !vkhip && model->hasDynamicResolution();
            const uint32_t capacityWidth = dynamic ? std::min(feature.outWidth, std::max(feature.width, p.renderWidth)) : p.renderWidth;
            const uint32_t capacityHeight = dynamic ? std::min(feature.outHeight, std::max(feature.height, p.renderHeight)) : p.renderHeight;
            feature.engineM = std::make_unique<d4r::GameUpscalerM>(*model, g_vk.physical, g_vk.device, feature.outWidth,
                                                                   feature.outHeight, capacityWidth, capacityHeight,
                                                                   hip || vkhip ? &external : nullptr);
            feature.engineMRenderWidth = capacityWidth;
            feature.engineMRenderHeight = capacityHeight;
            logf("engine backend (%s): feature %u runs preset %s in the game's command list (%ux%u -> %ux%u), network on %s, RGB10A2 direct output %d",
                 name, feature.handle.Id, name, p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight,
                 hip ? "the native HIP layers (list split around it)" : vkhip ? "vkhip" : "Vulkan",
                 model->hasRgb10Output() && game.output.view != VK_NULL_HANDLE &&
                     game.output.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
                     game.output.layout == VK_IMAGE_LAYOUT_GENERAL && game.output.image != game.color.image &&
                     game.output.image != game.motion.image && game.output.image != game.depth.image);
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend (%s): %s", name, error.what());
        release_engine_network(feature);
        feature.engineMVkHip.reset();
        return refuse("the engine could not be created");
    }
    d4r::GameUpscalerM* adapter = feature.engineM.get();
    // the HIP and vkhip networks' grids are fixed; the Vulkan engine takes render sizes up to the one it reserved
    const bool fixedGrid = feature.engineNet != nullptr || feature.engineMVkHip != nullptr;
    const bool resizedHip = fixedGrid &&
        (p.renderWidth != feature.engineMRenderWidth || p.renderHeight != feature.engineMRenderHeight);
    const bool beyondCapacity = !fixedGrid &&
        (p.renderWidth > feature.engineMRenderWidth || p.renderHeight > feature.engineMRenderHeight);
    if (resizedHip || beyondCapacity)
    {
        try
        {
            if (!feature.engineMDynamic)
            {
                feature.engineMDynamic = std::make_unique<d4r::GameUpscalerM>(*model, g_vk.physical, g_vk.device,
                    feature.outWidth, feature.outHeight, feature.outWidth, feature.outHeight);
                logf("engine backend (%s): render size %ux%u beyond %ux%u: an output-sized Vulkan engine takes such frames; "
                     "the original engine is retained", name, p.renderWidth, p.renderHeight, feature.engineMRenderWidth,
                     feature.engineMRenderHeight);
            }
            adapter = feature.engineMDynamic.get();
        }
        catch (const std::exception& error)
        {
            logf("engine backend (%s): %s", name, error.what());
            return refuse("the dynamic engine could not be created");
        }
    }
    feature.engineMLastWidth = p.renderWidth;
    feature.engineMLastHeight = p.renderHeight;
    if (feature.engineNet != nullptr && !resizedHip)
    {
        // The native HIP network: the game's list is split around it. Its native completion callback releases
        // the second part, which records expansion, reconstruction and the final store (or the conversion blit
        // for incompatible outputs). The Wine worker retires resources, not frame timing.
        auto recordPart = [&](bool front) -> const char* {
            VkCommandBuffer command = VK_NULL_HANDLE;
            if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &command)))
                return "BeginVkCommandBufferInterop failed";
            const char* failure = nullptr;
            try
            {
                if (front)
                {
                    adapter->recordFront(command, game);
                    // The HIP network's stream waits for this frame's number (record_engine_input_ready): the
                    // fill is part of the first half, so the network starts the moment the front's GPU work ends.
                    if (feature.engineGpuWait.load())
                        record_engine_input_ready(feature, command, frame);
                }
                else
                    adapter->recordBack(command, game);
            }
            catch (const std::exception& error)
            {
                logf("engine backend (%s): %s", name, error.what());
                failure = "the frame could not be recorded";
            }
            if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && failure == nullptr)
                failure = "EndVkCommandBufferInterop failed";
            return failure;
        };
        auto abandon = [&](const char* reason) {
            feature.engineRefused = true;
            logf("engine backend (%s): %s; feature %u uses the CUDA backend after frame %u", name, reason, feature.handle.Id, frame);
            return true;
        };
        return record_engine_network_frame(feature, list, sources, frame, name, game.reset, p.mvScaleX, p.mvScaleY,
                                           p.jitterX, p.jitterY, recordPart, refuse, abandon);
    }
    if (FAILED(g_vk.lifetime->RetainExternalResources(list, feature.resources)))
        return refuse("the command list cannot retain the engine's resources");
    for (const Source& source : sources)
        transition(list, source.resource, source.state, source.copyState);
    VkCommandBuffer command = VK_NULL_HANDLE;
    bool recorded = SUCCEEDED(g_vk.interop->BeginVkCommandBufferInterop(list, &command));
    const char* failure = recorded ? nullptr : "BeginVkCommandBufferInterop failed";
    if (recorded)
    {
        try
        {
            if (feature.engineMVkHip && !resizedHip)
            {
                adapter->recordFront(command, game);
                feature.engineMVkHip->record(command);
                adapter->recordBack(command, game);
            }
            else
                adapter->record(command, game);
        }
        catch (const std::exception& error)
        {
            logf("engine backend (%s): %s", name, error.what());
            failure = "the frame could not be recorded";
            recorded = false;
        }
        if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && recorded)
        {
            failure = "EndVkCommandBufferInterop failed";
            recorded = false;
        }
    }
    for (const Source& source : sources)
        transition(list, source.resource, source.copyState, source.state);
    if (!recorded)
        return refuse(failure);
    feature.engineJitter[0] = p.jitterX;
    feature.engineJitter[1] = p.jitterY;
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded, preset %s (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f)", frame, name, p.jitterX, p.jitterY,
             game.reset, p.mvScaleX, p.mvScaleY);
    return true;
}

static EngineBackendE g_engineE;

static void init_engine_backend_e(int slot)
{
    const char* const name = slot ? "F" : "E";
    std::string directory = env_string(slot ? "D4R_ENGINE_MODEL_F_DIR" : "D4R_ENGINE_MODEL_E_DIR");
    if (directory.empty())
    {
        char folder[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (g_portable.dir + (slot ? L"\\engine\\f" : L"\\engine\\e")).c_str(), -1, folder,
                            sizeof(folder), nullptr, nullptr);
        directory = folder;
    }
    if (g_vk.interop == nullptr || g_vk.lifetime == nullptr || g_vk.device == VK_NULL_HANDLE || d4r::vkTable.CreateBuffer == nullptr)
    {
        logf("engine backend (%s): Vulkan interop is unavailable", name);
        return;
    }
    const std::string path = directory[0] == '/' ? "Z:" + directory : directory;
    const auto start = std::chrono::steady_clock::now();
    try
    {
        g_engineE.models[slot] = std::make_unique<d4r::EModel>(g_vk.physical, g_vk.device, path);
    }
    catch (const std::exception& error)
    {
        logf("engine backend (%s): cannot load the model from %s: %s", name, path.c_str(), error.what());
        return;
    }
    logf("engine backend (%s): model %s loaded in %.0f ms", name, path.c_str(),
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

bool engine_evaluate_e(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                              ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                              ID3D12Resource* exposure, const FrameParams& p, uint32_t frame)
{
    const int slot = feature.preset == 6 ? 1 : 0;
    const char* const name = slot ? "F" : "E";
    auto refuse = [&](const char* reason) {
        feature.engineRefused = true;
        logf("engine backend (%s): %s; feature %u uses the CUDA backend from frame %u", name, reason, feature.handle.Id, frame);
        return false;
    };
    std::call_once(g_engineE.once[slot], [slot] { init_engine_backend_e(slot); });
    d4r::EModel* const model = g_engineE.models[slot].get();
    if (model == nullptr)
        return refuse(slot ? "no preset F model (engine\\f)" : "no preset E model (engine\\e)");
    // NVSDK_NGX_DLSS_Feature_Flags: IsHDR 1, MVLowRes 2, MVJittered 4, DepthInverted 8, AutoExposure 0x40.
    const bool hdr = (feature.flags & 0x1) != 0, lowRes = (feature.flags & 0x2) != 0;
    // AutoExposure: NGX measures the exposure from the colour (HDR only; an LDR feature has no exposure at all)
    const bool autoExposure = hdr && (feature.flags & 0x40) != 0;
    if (autoExposure && !model->hasAutoExposure())
        return refuse("auto exposure needs a model built by this d4r version (run d4r/tools/build-engine-models.sh)");
    if (hdr && !(std::isfinite(p.preExposure) && p.preExposure > 0.0f))
        return refuse("the pre-exposure is not a positive number");
    if (hdr && !(std::isfinite(p.exposureScale) && p.exposureScale > 0.0f))
        return refuse("the exposure scale is not a positive number");
    if (color == output || depth == output || motion == output)
        return refuse("input/output texture aliasing is not supported");
    if (p.renderWidth > feature.outWidth || p.renderHeight > feature.outHeight)
        return refuse("the render size exceeds the output size");
    d4r::GameFrameE game;
    game.profile = engine_stage_profile(frame, feature.preset == 6 ? "F" : "E");
    using Source = EngineFrameSource;
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    const uint32_t mw = lowRes ? p.renderWidth : feature.outWidth, mh = lowRes ? p.renderHeight : feature.outHeight;
    // HDR: the game's exposure texture is NGX's exposure (its copy kernel moves texel 0 into the 1x1 the kernels read);
    // with AutoExposure NGX measures instead and does not read it
    const bool gameExposure = hdr && !autoExposure && exposure != nullptr;
    Source sources[5] = {
        {color, &game.color, p.colorBaseX, p.colorBaseY, p.renderWidth, p.renderHeight, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {depth, &game.depth, p.depthBaseX, p.depthBaseY, p.renderWidth, p.renderHeight, depthState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {motion, &game.motion, p.mvBaseX, p.mvBaseY, mw, mh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {output, &game.output, p.outputBaseX, p.outputBaseY, feature.outWidth, feature.outHeight, outputState,
         D3D12_RESOURCE_STATE_COPY_DEST},
        {exposure, &game.exposure, 0, 0, 1, 1, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
    };
    const size_t sourceCount = gameExposure ? 5 : 4;
    for (size_t index = 0; index < sourceCount; ++index)
    {
        Source& source = sources[index];
        UINT64 handle = 0, offset = 0;
        D3D12_RESOURCE_DESC desc;
        source.resource->GetDesc(&desc);
        if (FAILED(g_vk.interop->GetVulkanResourceInfo1(source.resource, &handle, &offset, &source.image->format)) ||
            handle == 0 || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
            source.x + source.width > desc.Width || source.y + source.height > desc.Height)
            return refuse("a texture is not a single-sample 2D image covering its rectangle");
        source.image->image = reinterpret_cast<VkImage>(handle);
        source.image->x = source.x;
        source.image->y = source.y;
        prepare_engine_view(feature, source.resource, desc, *source.image, source.image == &game.output,
                            source.x == 0 && source.y == 0, source.copyState);
        g_vk.interop->GetVulkanImageLayout(source.resource, source.copyState, &source.image->layout);
    }
    if (gameExposure && (game.exposure.view == VK_NULL_HANDLE || (game.exposure.format != VK_FORMAT_R32_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R16_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R32G32_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R16G16_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R32G32B32A32_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R16G16B16A16_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R8G8B8A8_UNORM)))
        return refuse("the exposure texture must be a float or unorm texture the engine samples directly");
    if (game.depth.format == VK_FORMAT_D32_SFLOAT || game.depth.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        game.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!d4r::gameColorFormat(game.color.format) || !d4r::gameMotionFormat(game.motion.format) ||
        !d4r::gameDepthFormat(game.depth.format, game.depth.aspect) || !d4r::gameOutputFormat(game.output.format))
    {
        logf("engine backend (%s): Vulkan formats colour %d, depth %d, motion %d, output %d", name, game.color.format,
             game.depth.format, game.motion.format, game.output.format);
        return refuse("a texture format is not implemented");
    }
    game.jitter[0] = p.jitterX;
    game.jitter[1] = p.jitterY;
    game.motionScale[0] = p.mvScaleX;
    game.motionScale[1] = p.mvScaleY;
    if (hdr)
    {
        game.preExposure = p.preExposure;
        game.exposureScale = p.exposureScale;
    }
    const bool resized = feature.engineE != nullptr &&
        (p.renderWidth != feature.engineERenderWidth || p.renderHeight != feature.engineERenderHeight);
    game.reset = p.reset != 0 || feature.engineE == nullptr || resized;
    try
    {
        if (feature.engineE == nullptr || resized)
        {
            feature.engineE.reset();
            feature.engineE = std::make_unique<d4r::GameUpscalerE>(*model, g_vk.physical, g_vk.device, feature.outWidth,
                                                                   feature.outHeight, p.renderWidth, p.renderHeight, hdr, lowRes,
                                                                   autoExposure);
            feature.engineERenderWidth = p.renderWidth;
            feature.engineERenderHeight = p.renderHeight;
            logf("engine backend (%s): feature %u runs preset %s in the game's command list (%ux%u -> %ux%u, %s, %s motion%s)",
                 name, feature.handle.Id, name, p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight,
                 hdr ? "HDR" : "LDR", lowRes ? "render-size" : "output-size", autoExposure ? ", auto exposure" : "");
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend (%s): %s", name, error.what());
        return refuse("the engine could not be created");
    }
    if (FAILED(g_vk.lifetime->RetainExternalResources(list, feature.resources)))
        return refuse("the command list cannot retain the engine's resources");
    for (size_t index = 0; index < sourceCount; ++index)
        transition(list, sources[index].resource, sources[index].state, sources[index].copyState);
    VkCommandBuffer command = VK_NULL_HANDLE;
    bool recorded = SUCCEEDED(g_vk.interop->BeginVkCommandBufferInterop(list, &command));
    const char* failure = recorded ? nullptr : "BeginVkCommandBufferInterop failed";
    if (recorded)
    {
        try
        {
            feature.engineE->record(command, game);
        }
        catch (const std::exception& error)
        {
            logf("engine backend (%s): %s", name, error.what());
            failure = "the frame could not be recorded";
            recorded = false;
        }
        if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && recorded)
        {
            failure = "EndVkCommandBufferInterop failed";
            recorded = false;
        }
    }
    for (size_t index = 0; index < sourceCount; ++index)
        transition(list, sources[index].resource, sources[index].copyState, sources[index].state);
    if (!recorded)
        return refuse(failure);
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded, preset %s (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f)", frame, name, p.jitterX,
             p.jitterY, game.reset, p.mvScaleX, p.mvScaleY);
    return true;
}

std::unique_ptr<vkhip::RRStages> create_engine_rr_vkhip(const std::string& directory, d4r::REngine& engine,
                                                               int slot);

static const uint32_t g_rrPreset[6] = {4, 5, 6, 1, 2, 3};

std::string rr_model_directory(int slot)
{
    static const char* const variables[6] = {"D4R_ENGINE_MODEL_RR_D_DIR", "D4R_ENGINE_MODEL_RR_E_DIR",
                                             "D4R_ENGINE_MODEL_RR_F_DIR", "D4R_ENGINE_MODEL_RR_A_DIR",
                                             "D4R_ENGINE_MODEL_RR_B_DIR", "D4R_ENGINE_MODEL_RR_C_DIR"};
    static const wchar_t* const folders[6] = {L"\\engine\\rr-d", L"\\engine\\rr-e", L"\\engine\\rr-f",
                                              L"\\engine\\rr-a", L"\\engine\\rr-b", L"\\engine\\rr-c"};
    std::string directory = env_string(variables[slot]);
    if (directory.empty())
    {
        char folder[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (g_portable.dir + folders[slot]).c_str(), -1, folder, sizeof(folder), nullptr, nullptr);
        directory = folder;
        if (slot < 2 && !g_vk.float8)
            directory += "-gfx11";
    }
    return directory[0] == '/' ? "Z:" + directory : directory;
}

static void init_engine_backend_rr(int slot)
{
    const std::string path = rr_model_directory(slot);
    if ((!g_vk.native && (g_vk.interop == nullptr || g_vk.lifetime == nullptr)) || g_vk.device == VK_NULL_HANDLE || d4r::vkTable.CreateBuffer == nullptr)
    {
        logf("engine backend (Ray Reconstruction): Vulkan interop is unavailable");
        g_engineR.state[slot] = 3;
        return;
    }
    logf("engine backend (Ray Reconstruction %c): loading the model from %s and compiling its pipelines; the CUDA "
         "backend serves frames until it is ready", g_rrLetter[slot], path.c_str());
    const auto start = std::chrono::steady_clock::now();
    try
    {
        std::string metadata;
        if (!read_whole_file(widen(path + "\\rr.txt"), metadata) ||
            !d4r_rr_model_compatible(metadata, g_vk.matrixF16, g_vk.float8))
            throw std::runtime_error("model arithmetic is unsupported by this GPU (no pipelines created)");
        auto model = std::make_unique<d4r::RModel>(g_vk.physical, g_vk.device, path, VK_NULL_HANDLE, g_vk.int64Atomics,
                                                   g_vk.robustImages);
        if (model->preset() != g_rrPreset[slot])
            throw std::runtime_error("the directory holds a model of another preset");
        g_engineR.models[slot] = std::move(model);
    }
    catch (const std::exception& error)
    {
        logf("engine backend (Ray Reconstruction): cannot load the model from %s: %s", path.c_str(), error.what());
        g_engineR.state[slot] = 3;
        return;
    }
    g_engineR.state[slot] = 2;
    logf("engine backend (Ray Reconstruction %c): model %s loaded and pipelines compiled in %.0f ms", g_rrLetter[slot],
         path.c_str(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

// Starts loading a Ray Reconstruction model (once per slot) on a thread of its own, or in place with
// D4R_ENGINE_RR_SYNC=1. The thread keeps the game's device, and with it the VkDevice, alive until the model is loaded.

static void begin_engine_model_rr(int slot)
{
    std::call_once(g_engine.vulkanOnce, init_engine_vulkan);
    std::call_once(g_engineR.once[slot], [slot] {
        g_engineR.state[slot] = 1;
        // D4R_ENGINE_RR_SYNC=1 (harness comparisons): load in place so the engine serves the first frame
        if (env_uint("D4R_ENGINE_RR_SYNC", 0) != 0)
            init_engine_backend_rr(slot);
        else
        {
            ID3D12Device* const device = g.device;
            if (device != nullptr)
                device->AddRef();
            std::thread([slot, device] {
                init_engine_backend_rr(slot);
                if (device != nullptr)
                    device->Release();
            }).detach();
        }
    });
}

// The slot of the Ray Reconstruction model that serves a preset hint

int rr_engine_slot(uint32_t preset)
{
    static const bool haveF = GetFileAttributesA((rr_model_directory(2) + "\\rr.txt").c_str()) != INVALID_FILE_ATTRIBUTES;
    // A - C (hints 1 - 3, which NGX 310.4 and later map to D or F): the CNN's model when one is installed
    static const bool haveCnn[3] = {
        GetFileAttributesA((rr_model_directory(3) + "\\rr.txt").c_str()) != INVALID_FILE_ATTRIBUTES,
        GetFileAttributesA((rr_model_directory(4) + "\\rr.txt").c_str()) != INVALID_FILE_ATTRIBUTES,
        GetFileAttributesA((rr_model_directory(5) + "\\rr.txt").c_str()) != INVALID_FILE_ATTRIBUTES};
    if (preset >= 1 && preset <= 3 && haveCnn[preset - 1])
        return int(preset) + 2;
    return preset == 5 ? 1 : (haveF && preset != 4) ? 2 : 0;
}

// Starts loading the Ray Reconstruction model `feature` runs on (once per preset) and returns its slot. Called at
// CreateFeature too, so the model is usually ready by the first frame.

int start_engine_model_rr(const Feature& feature)
{
    // NGX 310.7 runs preset E's kernels for the hint 5 only and preset D's (diamond_wallaby) for every other value:
    // none, the retired A / B / C (1-3), 6, and values with the top bit set such as Cyberpunk 2077's 0x80000002 or
    // 0x80000005 (harness captures with D4R_RR_PRESET, 2026-10-12).
    // NGX 310.9 has preset F (DLSS 4.5, the rr2_* kernels) and runs it for every hint but 4 (D) and 5 (E): none,
    // 6, Cyberpunk 2077's 0x80000002 (harness runs with D4R_RR_PRESET, 2026-10-13). The shim cannot see which DLL
    // version the CUDA backend loaded, so that mapping applies when the user built a preset F model (engine\rr-f),
    // which only a 310.9 DLL yields.
    const int slot = rr_engine_slot(feature.preset);
    begin_engine_model_rr(slot);
    return slot;
}

// Whether this game can use Ray Reconstruction: it ships NVIDIA's denoiser, as a game that offers it must, beside its
// executable or in a folder it gave NGX init. d4r's own copy (the one the NGX core loads, also on the feature search
// path) does not count; it is recognised by file identity, since Wine can reach the same folder through two drive
// letters.

static bool game_ships_denoiser()
{
    auto identify = [](const std::wstring& path, BY_HANDLE_FILE_INFORMATION& info) {
        const HANDLE file = CreateFileW(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            return false;
        const bool known = GetFileInformationByHandle(file, &info) != 0;
        CloseHandle(file);
        return known;
    };
    auto same = [](const BY_HANDLE_FILE_INFORMATION& a, const BY_HANDLE_FILE_INFORMATION& b) {
        return a.nFileIndexHigh == b.nFileIndexHigh && a.nFileIndexLow == b.nFileIndexLow &&
               a.nFileSizeHigh == b.nFileSizeHigh && a.nFileSizeLow == b.nFileSizeLow;
    };
    auto directory_of = [](HMODULE module) {
        wchar_t path[MAX_PATH] = {};
        const DWORD length = GetModuleFileNameW(module, path, MAX_PATH);
        std::wstring directory = length > 0 && length < MAX_PATH ? std::wstring(path, length) : std::wstring();
        const size_t slash = directory.find_last_of(L"\\/");
        directory.resize(slash == std::wstring::npos ? 0 : slash);
        return directory;
    };
    BY_HANDLE_FILE_INFORMATION own = {};
    const std::wstring self = directory_of(g_selfModule);
    const bool haveOwn = !self.empty() && identify(self + L"\\nvngx_dlssd.dll", own);
    std::vector<std::wstring> folders = g.paths;
    folders.push_back(directory_of(nullptr));
    for (std::wstring folder : folders)
    {
        while (!folder.empty() && (folder.back() == L'\\' || folder.back() == L'/'))
            folder.pop_back();
        if (folder.empty())
            continue;
        BY_HANDLE_FILE_INFORMATION info = {};
        if (!identify(folder + L"\\nvngx_dlssd.dll", info) || (haveOwn && same(info, own)))
            continue;
        logf("engine backend: the game ships a denoiser (%ls\\nvngx_dlssd.dll)", folder.c_str());
        return true;
    }
    return false;
}

// At NGX init, starts loading the engine models this configuration will use: Ray Reconstruction's (the model a feature
// without a preset hint runs, normally preset F) when the game can use it (game_ships_denoiser, or [RayReconstruction]
// Enable = true) and Super Sampling's when [DLSS] Model forces K, L or M (its HDR model; LDR models still load at
// first use). On a cold shader cache their pipelines take seconds to compile (preset F 3.7 s in Witcher 3 after a
// driver update), and games create their features well after they initialise NGX (64 s there), so the compile no
// longer holds up the first frame. Only the models (pipelines and weights) load early; every feature still creates
// its own resources at its first frame. D4R_ENGINE_PRELOAD=0 turns this off.

void start_engine_preload()
{
    static std::once_flag once;
    if (env_uint("D4R_ENGINE_PRELOAD", 1) == 0 || !engine_requested() || g.device == nullptr)
        return;
    std::call_once(once, [] {
        // the game's thread, inside its NGX init: the vkd3d-proton interop and the engine's Vulkan entry points
        std::call_once(g_vk.once, init_vram_interop);
        if (g_vk.interop == nullptr || g_vk.lifetime == nullptr)
            return;
        std::call_once(g_engine.vulkanOnce, init_engine_vulkan);
        if (g_vk.device == VK_NULL_HANDLE || d4r::vkTable.CreateBuffer == nullptr)
            return;
        // without the game's denoiser, Ray Reconstruction's model loads at its first feature (most DLSS games have
        // none: 4.7 - 5.4 s of one core on a cold shader cache and ~7 MiB of VRAM for nothing)
        const bool rr = rr_enabled() && rr_backend() == RrBackend::Dlss &&
                        (!env_string("D4R_RR_ENABLE").empty() || game_ships_denoiser());
        if (rr_enabled() && rr_backend() == RrBackend::Dlss && !rr)
            logf("engine backend: the game ships no nvngx_dlssd.dll: Ray Reconstruction's model loads at its first "
                 "feature ([RayReconstruction] Enable = true loads it at NGX init)");
        const auto forced = d4r_parse_render_preset_override(env_string(d4r_render_preset_variable(false)).c_str());
        const uint32_t sr = forced.enabled ? forced.value : 0;
        const bool srM = (sr == 13 || sr == 12) && (g_vk.float8 || engine_m_dot2());
        const bool srK = sr == 11 && !engine_k_refused();
        if (!rr && !srM && !srK)
            return;
        logf("engine backend: loading the models this configuration uses at NGX init:%s%s%s", rr ? " Ray Reconstruction" : "",
             rr && (srM || srK) ? "," : "", srM ? (sr == 12 ? " Super Sampling preset L" : " Super Sampling preset M")
                                         : srK ? " Super Sampling preset K" : "");
        if (rr)
            begin_engine_model_rr(rr_engine_slot(0));
        if (srM || srK)
        {
            ID3D12Device* const device = g.device;
            device->AddRef();
            std::thread([device, srK, presetL = sr == 12] {
                if (srK)
                    std::call_once(g_engine.once, init_engine_backend);
                else
                    std::call_once(g_engineM.once[presetL][0], [presetL] { init_engine_backend_m_variant(presetL, false); });
                device->Release();
            }).detach();
        }
    });
}

// Ray Reconstruction on the engine: d4r's stages in the game's command list, with the denoiser's guides sampled
// straight from the game's textures. False: the CUDA backend serves the feature (reason logged).

bool engine_evaluate_rr(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                               ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output, void* parameters,
                               const FrameParams& p, uint32_t frame)
{
    // set below once the preset is known: preset F was asked for but nvngx_dlssd.dll (the CUDA backend) has none
    bool cudaLacksF = false;
    auto warn_not_f = [&](const char* when) {
        if (!cudaLacksF)
            return;
        logf("WARNING: Ray Reconstruction preset F (DLSS 4.5) was selected, but %s feature %u is served by the CUDA "
             "backend, whose nvngx_dlssd.dll %u.%u.%u.%u has no preset F: those frames show preset D, NOT F",
             when, feature.handle.Id, g.denoiserVersion[0], g.denoiserVersion[1], g.denoiserVersion[2],
             g.denoiserVersion[3]);
    };
    auto refuse = [&](const char* reason) {
        feature.engineRefused = true;
        logf("engine backend (Ray Reconstruction): %s; feature %u uses the CUDA backend from frame %u", reason,
             feature.handle.Id, frame);
        warn_not_f("from this frame on");
        return false;
    };
    const int slot = start_engine_model_rr(feature);
    cudaLacksF = slot == 2 && !denoiser_has_preset_f();
    int state = g_engineR.state[slot].load();
    if (state == 1 && feature.cudaCreationDeferred)
    {
        // No CUDA feature yet: making one for these frames compiles NGX's kernels (44 s with 310.9 on a cold ZLUDA
        // cache, RE Requiem 2026-10-14; the model was ready 7 ms after the first frame). Wait for the model instead.
        logf("engine backend (Ray Reconstruction %c): frame %u waits for the model rather than creating the CUDA "
             "fallback", g_rrLetter[slot], frame);
        while ((state = g_engineR.state[slot].load()) == 1)
            d4r_sleep_us(1000);
    }
    if (state == 1)
    {
        // still compiling and the CUDA feature exists: it serves this frame, the engine is tried again next frame
        if (!feature.rrCompileWarned)
        {
            feature.rrCompileWarned = true;
            warn_not_f("while the preset F model compiles,");
        }
        return false;
    }
    d4r::RModel* const model = g_engineR.models[slot].get();
    if (state != 2 || model == nullptr)
        return refuse(slot == 1 ? "no preset E model (engine\\rr-e)" : slot == 2 ? "no preset F model (engine\\rr-f)" :
                      slot >= 3 ? "no preset A - C model (engine\\rr-a, rr-b, rr-c)" : "no preset D model (engine\\rr-d)");
    // NVSDK_NGX_DLSS_Feature_Flags: IsHDR 1, MVLowRes 2, MVJittered 4, DepthInverted 8.
    if ((feature.flags & 0x3) != 0x3)
        return refuse("needs the HDR flag and render-resolution motion vectors");
    if ((feature.flags & 0x4) != 0)
        return refuse("jittered motion vectors are not implemented");
    if (!p.auxMatrixSet[0] || !p.auxMatrixSet[1])
        return refuse("the camera matrices are missing");
    // Preset F below 4/3 scaling (DLAA and the like): NGX runs another output filter there, rr2_post_kernel_3_2,
    // which the model does not have (harness runs at 1920x1080 output, 2026-10-13: 1440x810 and smaller render sizes
    // use rr2_post_kernel, 1456x820 and larger rr2_post_kernel_3_2)
    if (slot == 2 && (3ull * feature.outWidth < 4ull * p.renderWidth || 3ull * feature.outHeight < 4ull * p.renderHeight))
        return refuse("preset F below 4/3 scaling uses an output filter the engine does not have (rr2_post_kernel_3_2)");
    if (feature.engineR != nullptr &&
        (p.renderWidth != feature.engineMRenderWidth || p.renderHeight != feature.engineMRenderHeight))
        return refuse("changing the effective render resolution after initialization is not implemented");
    const char* const guideNames[4][2] = {{"GBuffer.Normals", nullptr}, {"GBuffer.Roughness", nullptr},
                                          {"DLSS.Input.DiffuseAlbedo", "GBuffer.DiffuseAlbedo"},
                                          {"DLSS.Input.SpecularAlbedo", "GBuffer.SpecularAlbedo"}};
    ID3D12Resource* guides[4] = {};
    // DLSS.Roughness.Mode 1 (packed): the roughness is the normals' w. Preset F follows the mode also when a roughness
    // texture is bound (harness run against the CUDA path, 2026-10-13); D and E use the bound texture (AW2 D capture:
    // enc0 byte 366 is zero despite mode 1). Match that choice for presets D/E.
    const bool roughnessPacked = p.auxScalar[AuxRoughnessMode] == 1 &&
        (slot == 2 || get_resource(parameters, "GBuffer.Roughness") == nullptr);
    for (int i = 0; i < 4; ++i)
    {
        guides[i] = get_resource(parameters, guideNames[i][0]);
        if (guides[i] == nullptr && guideNames[i][1] != nullptr)
            guides[i] = get_resource(parameters, guideNames[i][1]);
        if (i == 1 && roughnessPacked)
            guides[i] = guides[0];
        if (guides[i] == nullptr)
            return refuse("a guide the engine needs (normals, roughness, diffuse or specular albedo) is missing");
    }
    // the optional inputs a game may add (RE Requiem passes both): bound when present, like NGX
    ID3D12Resource* const hitDistance = get_resource(parameters, "DLSSD.SpecularHitDistance");
    ID3D12Resource* const disocclusion = get_resource(parameters, "DLSS.DisocclusionMask");
    // and Cyberpunk 2077's: the subsurface scattering guide and the colour before particles
    ID3D12Resource* const subsurface = get_resource(parameters, "DLSSD.ScreenSpaceSubsurfaceScatteringGuide");
    ID3D12Resource* const beforeParticles = get_resource(parameters, "DLSSD.ColorBeforeParticles");
    for (ID3D12Resource* input : {color, depth, motion, guides[0], guides[1], guides[2], guides[3], hitDistance, disocclusion, subsurface, beforeParticles})
        if (input == output)
            return refuse("input/output texture aliasing is not supported");
    d4r::GameFrameR game;
    game.profile = engine_stage_profile(frame, feature.preset == 6 ? "RR_F" : (feature.preset == 4 ? "RR_D" : "RR_E"));
    using Source = EngineFrameSource;
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    const unsigned int rw = p.renderWidth, rh = p.renderHeight;
    std::vector<Source> sources = {
        {color, &game.color, p.colorBaseX, p.colorBaseY, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {depth, &game.depth, p.depthBaseX, p.depthBaseY, rw, rh, depthState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {motion, &game.motion, p.mvBaseX, p.mvBaseY, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {guides[0], &game.normals, 0, 0, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {guides[1], &game.roughness, 0, 0, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {guides[2], &game.diffuseAlbedo, 0, 0, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {guides[3], &game.specularAlbedo, 0, 0, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {output, &game.output, p.outputBaseX, p.outputBaseY, feature.outWidth, feature.outHeight, outputState,
         D3D12_RESOURCE_STATE_COPY_DEST},
    };
    if (hitDistance != nullptr)
        sources.push_back({hitDistance, &game.specularHitDistance, 0, 0, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE});
    if (disocclusion != nullptr)
        sources.push_back({disocclusion, &game.disocclusionMask, 0, 0, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE});
    if (subsurface != nullptr)
        sources.push_back({subsurface, &game.subsurfaceGuide, 0, 0, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE});
    if (beforeParticles != nullptr)
        sources.push_back({beforeParticles, &game.colorBeforeParticles, 0, 0, rw, rh, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE});
    // a guide with its own subrect origin (DLSS.Input.Normals.Subrect.Base etc.) is not implemented
    for (size_t i = 0; i < sizeof(kAuxSubrectNames) / sizeof(kAuxSubrectNames[0]); ++i)
        if (p.auxSubrect[i][0] != 0 || p.auxSubrect[i][1] != 0)
            return refuse("guide subrect origins are not implemented");
    for (Source& source : sources)
    {
        UINT64 handle = 0, offset = 0;
        D3D12_RESOURCE_DESC desc;
        source.resource->GetDesc(&desc);
        if (FAILED(g_vk.interop->GetVulkanResourceInfo1(source.resource, &handle, &offset, &source.image->format)) ||
            handle == 0 || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
            source.x + source.width > desc.Width || source.y + source.height > desc.Height)
            return refuse("a texture is not a single-sample 2D image covering its rectangle");
        source.image->image = reinterpret_cast<VkImage>(handle);
        source.image->x = source.x;
        source.image->y = source.y;
        // The stages sample inputs with normalized coordinates over the whole view: a texture larger than the render
        // size (dynamic resolution, padded targets) goes through the copy into an exactly sized plane instead.
        const bool exact = source.image == &game.output || (desc.Width == source.width && desc.Height == source.height);
        prepare_engine_view(feature, source.resource, desc, *source.image, source.image == &game.output,
                            source.x == 0 && source.y == 0 && exact, source.copyState);
        g_vk.interop->GetVulkanImageLayout(source.resource, source.copyState, &source.image->layout);
    }
    // the same resource can appear twice (a game may pass one texture as two guides): transition it once
    for (size_t i = 0; i < sources.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (sources[i].resource == sources[j].resource &&
                (sources[i].state != sources[j].state || sources[i].copyState != sources[j].copyState))
                return refuse("shared inputs require conflicting resource states");
    // Read-only guides may alias even when D/E use the bound roughness texture. Each resource is transitioned once.
    auto firstUse = [&](size_t i) {
        for (size_t j = 0; j < i; ++j)
            if (sources[j].resource == sources[i].resource) return false;
        return true;
    };
    if (game.depth.format == VK_FORMAT_D32_SFLOAT || game.depth.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        game.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!d4r::gameColorFormat(game.color.format) || !d4r::gameMotionFormat(game.motion.format) ||
        !d4r::gameDepthFormat(game.depth.format, game.depth.aspect) || !d4r::gameOutputFormat(game.output.format))
    {
        logf("engine backend (Ray Reconstruction): Vulkan formats colour %d, depth %d, motion %d, output %d",
             game.color.format, game.depth.format, game.motion.format, game.output.format);
        return refuse("a texture format is not implemented");
    }
    if (frame <= 2)
    {
        const char* names[8] = {"color", "depth", "motion", "normals", "roughness", "diffuse", "specular", "output"};
        for (int i = 0; i < 8; ++i)
        {
            D3D12_RESOURCE_DESC desc;
            sources[i].resource->GetDesc(&desc);
            logf("engine backend (Ray Reconstruction): %s format %d layout %d origin %u,%u size %llux%u view %p aspect %u",
                 names[i], sources[i].image->format, sources[i].image->layout, sources[i].image->x, sources[i].image->y,
                 static_cast<unsigned long long>(desc.Width), desc.Height, reinterpret_cast<void*>(sources[i].image->view),
                 sources[i].image->aspect);
        }
    }
    auto& s = game.settings;
    s.jitter[0] = p.jitterX; s.jitter[1] = p.jitterY;
    s.motionScale[0] = p.mvScaleX; s.motionScale[1] = p.mvScaleY;
    s.preExposure = p.preExposure;
    s.frameTimeMs = p.frameTime;
    s.reset = p.reset != 0 || feature.engineR == nullptr;
    s.depthInverted = (feature.flags & 0x8) != 0;
    s.hardwareDepth = p.auxScalar[AuxHardwareDepth] != 0;
    s.roughnessPacked = roughnessPacked;
    if (frame <= 2)
        logf("engine backend (Ray Reconstruction): roughness %s, specular hit distance %s, disocclusion mask %s, "
             "subsurface guide %s, colour before particles %s, hardware depth %d",
             roughnessPacked ? "packed in the normals' w" : "texture", hitDistance != nullptr ? "bound" : "none",
             disocclusion != nullptr ? "bound" : "none", subsurface != nullptr ? "bound" : "none",
             beforeParticles != nullptr ? "bound" : "none", s.hardwareDepth ? 1 : 0);
    std::copy(std::begin(p.auxMatrix[0]), std::end(p.auxMatrix[0]), s.worldToView);
    std::copy(std::begin(p.auxMatrix[1]), std::end(p.auxMatrix[1]), s.viewToClip);
    try
    {
        if (feature.engineR == nullptr)
        {
            const bool rrVkHip = engine_rr_vkhip_capable(slot);
            feature.engineR = std::make_unique<d4r::GameDenoiser>(*model, g_vk.physical, g_vk.device, feature.outWidth,
                                                                  feature.outHeight, rw, rh, rrVkHip);
            if (rrVkHip)
                feature.engineRVkHip = create_engine_rr_vkhip(rr_model_directory(slot), feature.engineR->networkEngine(),
                                                              slot);
            feature.engineMRenderWidth = rw;
            feature.engineMRenderHeight = rh;
            logf("engine backend (Ray Reconstruction %c): feature %u runs in the game's command list (%ux%u -> %ux%u)",
                 g_rrLetter[slot], feature.handle.Id, rw, rh, feature.outWidth, feature.outHeight);
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend (Ray Reconstruction): %s", error.what());
        return refuse("the engine could not be created");
    }
    if (FAILED(g_vk.lifetime->RetainExternalResources(list, feature.resources)))
        return refuse("the command list cannot retain the engine's resources");
    const bool profile = profile_enabled() && env_uint("D4R_PROFILE_GPU_TIMESTAMPS", 1) != 0 &&
        profile_timestamps_ready(feature);
    if (profile)
    {
        FrameTiming measured;
        read_profile_timestamps(feature, frame, measured);
        const auto now = ProfileClock::now();
        const double interval = feature.engineRProfileAt.time_since_epoch().count() ?
            profile_ms(feature.engineRProfileAt, now) : 0.0;
        feature.engineRProfileAt = now;
        if (profile_log_frame(frame) && measured.vkTotal >= 0)
            logf("D4R_ENGINE_RR_PROFILE frame=%u gpu_frame=%u render=%ux%u output=%ux%u gpu_eval=%.3f "
                 "game_interval=%.3f age=0", frame, frame - kOutputSlots, rw, rh, feature.outWidth,
                 feature.outHeight, measured.vkTotal, interval);
        profile_timestamp(feature, list, frame, 0);
        profile_timestamp(feature, list, frame, 1);
    }
    for (size_t i = 0; i < sources.size(); ++i)
        if (firstUse(i))
            transition(list, sources[i].resource, sources[i].state, sources[i].copyState);
    VkCommandBuffer command = VK_NULL_HANDLE;
    bool recorded = SUCCEEDED(g_vk.interop->BeginVkCommandBufferInterop(list, &command));
    const char* failure = recorded ? nullptr : "BeginVkCommandBufferInterop failed";
    if (recorded)
    {
        try
        {
            feature.engineR->record(command, game);
        }
        catch (const std::exception& error)
        {
            logf("engine backend (Ray Reconstruction): %s", error.what());
            failure = "the frame could not be recorded";
            recorded = false;
        }
        if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && recorded)
        {
            failure = "EndVkCommandBufferInterop failed";
            recorded = false;
        }
    }
    for (size_t i = 0; i < sources.size(); ++i)
        if (firstUse(i))
            transition(list, sources[i].resource, sources[i].copyState, sources[i].state);
    if (!recorded)
        return refuse(failure);
    if (profile)
    {
        profile_timestamp(feature, list, frame, 2);
        profile_timestamp(feature, list, frame, 3);
    }
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded, Ray Reconstruction %c (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f)", frame,
             g_rrLetter[slot], p.jitterX, p.jitterY, s.reset, p.mvScaleX, p.mvScaleY);
    return true;
}

// The render adapter for vkhip: its code object machine (gfx11Mach; with `rdna4` also gfx12Mach), its name and its
// timestamp period; false and `reason` when vkhip cannot inject native code there. RADV is authenticated through sysfs.
// With `amd`, AMD's own driver (native Windows) qualifies too when the device has VK_KHR_pipeline_binary: `binaries` is
// set and `mach` 0, since the injector checks every code object against the driver's own compile for the GPU.

static bool vkhip_adapter(bool rdna4, bool amd, uint32_t& mach, bool& binaries, std::string& gfx, float& timestampPeriod,
                          std::string& reason)
{
    auto unavailable = [&](const std::string& why) { reason = why; return false; };
    mach = 0; binaries = false;
    auto properties = g_vk.instanceProc ? reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        g_vk.instanceProc(g_vk.instance, "vkGetPhysicalDeviceProperties2")) : nullptr;
    if (!properties || !g_vk.deviceProc) return unavailable("Vulkan device properties unavailable");
    // The driver from VkPhysicalDeviceVulkan12Properties on a 1.2 device: OptiScaler's Vulkan spoofing rewrites
    // VkPhysicalDeviceDriverProperties (and the device name) to NVIDIA's for every caller in the game's process.
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties(g_vk.physical, &props);
    VkPhysicalDeviceVulkan12Properties v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceDriverProperties driverOnly{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    const bool core = props.properties.apiVersion >= VK_API_VERSION_1_2;
    props.pNext = core ? static_cast<void*>(&v12) : static_cast<void*>(&driverOnly);
    properties(g_vk.physical, &props);
    struct { VkDriverId driverID; const char* driverInfo; } driver = {
        core ? v12.driverID : driverOnly.driverID, core ? v12.driverInfo : driverOnly.driverInfo};
    timestampPeriod = props.properties.limits.timestampPeriod;
    if (amd && driver.driverID == VK_DRIVER_ID_AMD_PROPRIETARY)
    {
        if (!g_vk.pipelineBinary)
            return unavailable("AMD's driver needs VK_KHR_pipeline_binary and VK_KHR_maintenance5 on the device");
        binaries = true;
        gfx = std::string(props.properties.deviceName) + " (AMD driver " + driver.driverInfo + ")";
        return true;
    }
    if (driver.driverID != VK_DRIVER_ID_MESA_RADV)
        return unavailable(amd ? "requires Mesa RADV or AMD's driver" : "requires Mesa RADV");
    // Device IDs can be spoofed by Proton/OptiScaler. Authenticate the actual render adapter before injecting
    // gfx11 machine code: its PCI vendor, then the gfx target of the amdgpu KFD node at the same PCI address.
    const std::string pci = env_string("D4R_RENDER_PCI");
    // sysfs reports a synthetic file size; read until the end.
    auto sysfs = [](const std::wstring& path, std::string& value) {
        HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  nullptr, OPEN_EXISTING, 0, nullptr);
        if (file == INVALID_HANDLE_VALUE) return false;
        value.clear();
        char bytes[4096]; DWORD count = 0;
        while (value.size() < 65536 && ReadFile(file, bytes, sizeof(bytes), &count, nullptr) && count != 0)
            value.append(bytes, count);
        CloseHandle(file);
        return !value.empty();
    };
    // "key value" lines of a KFD properties file
    auto property = [](const std::string& text, const char* key, unsigned long long& value) {
        const std::string want = std::string(key) + " ";
        for (size_t at = 0; at < text.size();) {
            const size_t end = std::min(text.find('\n', at), text.size());
            if (text.compare(at, want.size(), want) == 0) {
                value = std::strtoull(text.c_str() + at + want.size(), nullptr, 0);
                return true;
            }
            at = end + 1;
        }
        return false;
    };
    std::string vendor;
    unsigned domain = 0, bus = 0, slot = 0, function = 0;
    if (pci.empty() || std::sscanf(pci.c_str(), "%x:%x:%x.%x", &domain, &bus, &slot, &function) != 4 ||
        !sysfs(widen("Z:\\sys\\bus\\pci\\devices\\" + pci + "\\vendor"), vendor) ||
        std::strtoul(vendor.c_str(), nullptr, 0) != 0x1002)
        return unavailable("the physical render adapter is not a verified AMD GPU");
    uint32_t target = 0;
    const std::wstring nodes = L"Z:\\sys\\class\\kfd\\kfd\\topology\\nodes\\";
    WIN32_FIND_DATAW node{};
    HANDLE find = FindFirstFileW((nodes + L"*").c_str(), &node);
    for (bool more = find != INVALID_HANDLE_VALUE; more && !target; more = FindNextFileW(find, &node)) {
        if (node.cFileName[0] == L'.') continue;
        std::string text;
        unsigned long long location = 0, nodeDomain = 0, version = 0;
        if (sysfs(nodes + node.cFileName + L"\\properties", text) && property(text, "location_id", location) &&
            property(text, "gfx_target_version", version) && location == ((bus << 8) | (slot << 3) | function) &&
            (!property(text, "domain", nodeDomain) || nodeDomain == domain))
            target = static_cast<uint32_t>(version);
    }
    if (find != INVALID_HANDLE_VALUE) FindClose(find);
    if (!target) return unavailable("no amdgpu KFD node for the render adapter " + pci);
    mach = vkhip::gfx11Mach(target) ? vkhip::gfx11Mach(target) : rdna4 ? vkhip::gfx12Mach(target) : 0;
    char name[16];
    std::snprintf(name, sizeof(name), "gfx%u%u%x", target / 10000, target / 100 % 100, target % 100);
    gfx = name;
    if (!mach) return unavailable((rdna4 ? "requires an RDNA3 or RDNA4 GPU, not " : "requires an RDNA3 GPU (gfx1100-gfx1103), not ") + gfx);
    return true;
}

static bool vkhip_load(const std::wstring& name, std::vector<uint8_t>& bytes)
{
    HANDLE file = CreateFileW(name.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER size{};
    bool ok = GetFileSizeEx(file, &size) && size.QuadPart > 0 && size.QuadPart <= 128 * 1024 * 1024;
    if (ok) {
        bytes.resize(static_cast<size_t>(size.QuadPart)); DWORD read = 0;
        ok = ReadFile(file, bytes.data(), static_cast<DWORD>(bytes.size()), &read, nullptr) && read == bytes.size();
    }
    CloseHandle(file);
    return ok;
}

// The native network stages of every Ray Reconstruction preset, on by default where they are the faster route:
// presets A - C (kernels/rr/build_vkhip_cnn.sh) are about 8x faster than their Vulkan shaders, presets D and E
// (kernels/rr/build_vkhip.sh) about 8% faster. Preset F (kernels/rr/build_vkhip_f.sh) measures within 1% of its
// Vulkan path on RADV, where it stays an explicit experiment (D4R_ENGINE_RR_VKHIP=1); on AMD's own driver (native
// Windows) its Vulkan path is LLPC's compile and vkhip is 17% faster with byte-identical output (RX 9070 XT,
// 1280x720 -> 2560x1440, frames back to back: 3.90 against 4.73 ms), so there F takes it by default too. All six need
// RDNA4. On AMD's own driver (VK_KHR_pipeline_binary) the injector checks each code object against the driver's
// compile for the GPU, which refuses a code object of another architecture.

bool engine_rr_vkhip_capable(int slot)
{
    const unsigned requested = env_uint("D4R_ENGINE_RR_VKHIP", 2); // 2: not set (auto)
    if (requested == 0 || !env_uint("D4R_ENGINE_VKHIP", 1)) return false;
    std::string gfx, reason; uint32_t mach = 0; bool binaries = false; float period = 0;
    if (!vkhip_adapter(true, true, mach, binaries, gfx, period, reason) ||
        (!binaries && mach != vkhip::gfx12Mach(120000) && mach != vkhip::gfx12Mach(120001))) {
        logf("experimental vkhip (RR %c) unavailable: %s; using Vulkan", g_rrLetter[slot],
             reason.empty() ? "requires RDNA4" : reason.c_str());
        return false;
    }
    if (slot == 2 && requested == 2 && !binaries) return false; // F under RADV: opt-in
    return true;
}

std::unique_ptr<vkhip::RRStages> create_engine_rr_vkhip(const std::string& directory, d4r::REngine& engine,
                                                               int slot)
{
    vkhip::NetFiles files;
    std::string gfx, reason; bool binaries = false; float period = 0;
    if (!vkhip_adapter(true, true, files.mach, binaries, gfx, period, reason)) return nullptr;
    files.pipelineBinary = binaries;   // AMD's own Vulkan driver injects through VK_KHR_pipeline_binary
    std::map<std::string, std::vector<uint8_t>> weights;
    const std::string only = env_string("D4R_VKHIP_RR_STAGES");
    const char* set = slot >= 3 ? "A - C" : slot == 2 ? "F" : "D/E";
    // presets A - C: the parameter blocks point at the model's weights
    if (slot >= 3)
        for (const char* stage : {"enc0", "enc1", "enc2", "enc3", "enc4", "dec4", "dec3", "dec2", "dec1", "dec0", "pdl"}) {
            if (!only.empty() && ("," + only + ",").find("," + std::string(stage) + ",") == std::string::npos) continue;
            const std::string name = std::string("rr3_") + stage;
            if (!vkhip_load(widen(directory + "/hip/" + name + ".hsaco"), files.code[name])) files.code.erase(name);
        }
    else if (slot == 2)   // preset F: rr2 blocks, the model's weights rearranged for the kernels
        for (const char* stage : {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec4_0", "dec4_1", "dec3_0", "dec3_1",
                                  "dec2_0", "dec2_1", "dec1_0", "dec1_1", "dec0_0", "dec0_1"}) {
            if (!only.empty() && ("," + only + ",").find("," + std::string(stage) + ",") == std::string::npos) continue;
            const std::string name = std::string("rr2_") + stage;
            std::vector<uint8_t> code, weight;
            if (!vkhip_load(widen(directory + "/hip/" + name + ".hsaco"), code)) continue;
            if (!vkhip_load(widen(directory + "/weights-" + stage + "m.bin"), weight)) {
                logf("experimental vkhip (RR F) unavailable: missing weights for %s; using Vulkan", stage); return nullptr;
            }
            files.code[name] = std::move(code); weights[stage] = std::move(weight);
        }
    else   // presets D and E: the Swin stages (kernels/rr/rrvk_swin.hip), encoder 0 after the engine's gather
        for (const char* stage : {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec4", "dec3", "dec2", "dec1", "dec0"}) {
            if (!only.empty() && ("," + only + ",").find("," + std::string(stage) + ",") == std::string::npos) continue;
            std::vector<uint8_t> code, weight;
            if (!vkhip_load(widen(directory + "/hip/rr_" + stage + ".hsaco"), code)) continue;
            if (!vkhip_load(widen(directory + "/weights-" + stage + "m.bin"), weight)) {
                logf("experimental vkhip (RR %s) unavailable: missing weights for %s; using Vulkan", set, stage); return nullptr;
            }
            files.code[std::string("rr_") + stage] = std::move(code); weights[stage] = std::move(weight);
        }
    if (files.code.empty()) {
        logf("experimental vkhip (RR %s) unavailable: no HIP code objects in %s/hip; using Vulkan", set,
             directory.c_str()); return nullptr;
    }
    auto stages = std::make_unique<vkhip::RRStages>();
    if (!stages->create(g_vk.physical, g_vk.instanceProc, g_vk.instance, g_vk.device, g_vk.deviceProc, g_vk.memory,
                        files, weights, engine, reason)) {
        logf("experimental vkhip (RR %s) unavailable: %s; using Vulkan", set, reason.c_str()); return nullptr;
    }
    engine.setStageRecorder(stages.get());
    logf("experimental vkhip (RR %s) ready on %s: %u native %s", set, gfx.c_str(), stages->stageCount(),
         slot == 2 ? "blocks" : "stages");
    return stages;
}

// The native layers of presets M and L (the model's hipnet.bin and hip/rrlite_*.hsaco; the ten shared Swin layers
// are identical for both presets) as vkhip pipelines for a fixed render size; null (logged) when vkhip or the layers
// are unavailable. RDNA3 and RDNA4.

std::unique_ptr<vkhip::MNetwork> create_engine_vkhip_net(const std::string& directory, uint32_t renderWidth,
                                                                uint32_t renderHeight, uint64_t bufferBytes)
{
    if (env_uint("D4R_ENGINE_VKHIP", 1) == 0) return nullptr;
    auto unavailable = [&](const std::string& reason) {
        logf("experimental vkhip (M/L) unavailable: %s; using the standard engine network", reason.c_str());
        return nullptr;
    };
    std::string gfx, reason;
    float timestampPeriod = 1.0f;
    uint32_t mach = 0;
    bool binaries = false;
    if (!vkhip_adapter(true, true, mach, binaries, gfx, timestampPeriod, reason)) return unavailable(reason);
    vkhip::NetFiles files;
    files.mach = mach;
    files.pipelineBinary = binaries;
    if (!vkhip_load(widen(directory + "/hipnet.bin"), files.hipnet)) return unavailable("missing hipnet.bin");
    for (const char* layer : {"enc1", "enc2", "enc3_tube", "dec2", "dec1"}) {
        const std::string name = std::string("rrlite_") + layer + "_4x4";
        if (!vkhip_load(widen(directory + "/hip/" + name + ".hsaco"), files.code[name]))
            return unavailable(name + ".hsaco missing");
    }
    auto network = std::make_unique<vkhip::MNetwork>();
    std::string error;
    if (!network->create(g_vk.device, g_vk.deviceProc, g_vk.memory, files, renderWidth, renderHeight, bufferBytes, error))
        return unavailable(error);
    logf("experimental vkhip (M/L) ready on %s: %u native dispatches for %ux%u", gfx.c_str(), network->dispatchesPerFrame(),
         renderWidth, renderHeight);
    if (env_uint("D4R_VKHIP_STAMPS", 0) != 0)
        network->enableStamps([](const char* line) { logf("%s", line); }, timestampPeriod);
    if (const std::string dump = env_string("D4R_VKHIP_DUMP_DIR"); !dump.empty())
        network->enableDump(dump, env_uint("D4R_VKHIP_DUMP_FRAMES", 4));
    return network;
}

// Preset K's eleven native layers on vkhip (D4RHIP, compile_k.py --hip) for a feature with this output size, or null
// (logged) when the adapter, the model or the injection cannot run them. Both Vulkan frontends use it: vkd3d-proton's
// adapter and the native Vulkan frontend (d4r_vulkan_frontend.inl).

std::unique_ptr<vkhip::KNetwork> create_engine_vkhip_k(const std::string& directory, uint32_t outWidth,
                                                              uint32_t outHeight, size_t tokenBytes, size_t headBytes)
{
    if (env_uint("D4R_ENGINE_VKHIP", 1) == 0) return nullptr;
    auto unavailable = [&](const std::string& reason) {
        logf("experimental vkhip unavailable: %s; using the standard engine network", reason.c_str());
        return nullptr;
    };
    std::string gfx, reason;
    float timestampPeriod = 1.0f;
    uint32_t mach = 0;
    bool binaries = false;
    // AMD's own driver (native Windows, vkd3d-proton with VK_KHR_pipeline_binary) qualifies as for M, RDNA2 (gfx103x
    // code objects) included, unofficially
    if (!vkhip_adapter(false, true, mach, binaries, gfx, timestampPeriod, reason)) return unavailable(reason);
    auto& load = vkhip_load;
    vkhip::NetFiles files;
    files.mach = mach;
    files.pipelineBinary = binaries;
    if (!load(widen(directory + "/hipnet.bin"), files.hipnet)) return unavailable("missing hipnet.bin");
    // [Kernels] PreferAccuracy: the accuracy layers (NVIDIA's f16 rounding after every k16 step; byte-identical to
    // the CUDA backend, about 0.12 ms slower at 1080p) when the model has them, otherwise the fast set in hip/
    const bool accuracy = env_uint("D4R_PREFER_ACCURACY", 0) != 0 &&
        GetFileAttributesW(widen(directory + "/hip-accuracy/dltss_pwin_enc0_layer.hsaco").c_str()) != INVALID_FILE_ATTRIBUTES;
    const std::string hipDir = directory + (accuracy ? "/hip-accuracy/" : "/hip/");
    for (const char* layer : {"enc0", "enc1", "enc2", "enc3", "enc4", "dec5", "dec4", "dec3", "dec2", "dec1", "dec0"}) {
        const std::string name = std::string("dltss_pwin_") + layer + "_layer";
        if (!load(widen(hipDir + name + ".hsaco"), files.code[name]))
            return unavailable(name + ".hsaco missing");
    }
    auto network = std::make_unique<vkhip::KNetwork>();
    std::string error;
    if (!network->create(g_vk.device, g_vk.deviceProc, g_vk.memory, files, outWidth,
                         outHeight, tokenBytes, headBytes, error)) return unavailable(error);
    logf("experimental vkhip ready on %s: %u native dispatches, %s layers, shared image-format adapter, regular/inverted depth",
         gfx.c_str(), network->dispatchesPerFrame(), accuracy ? "accuracy" : "fast");
    if (env_uint("D4R_VKHIP_STAMPS", 0) != 0)
        network->enableStamps([](const char* line) { logf("%s", line); }, timestampPeriod);
    if (const std::string dump = env_string("D4R_VKHIP_DUMP_DIR"); !dump.empty())
        network->enableDump(dump, env_uint("D4R_VKHIP_DUMP_FRAMES", 4));
    return network;
}

bool create_engine_vkhip(Feature& feature, const std::string& directory,
                                size_t tokenBytes, size_t headBytes)
{
    feature.engineVkHip = create_engine_vkhip_k(directory, feature.outWidth, feature.outHeight, tokenBytes, headBytes);
    return feature.engineVkHip != nullptr;
}

// The engine's frame settings for NGX's K parameters (either engine). first: the feature's first engine frame.

d4r::FrameSettings engine_k_settings(const Feature& feature, const FrameParams& p, bool ldr, bool displayMotion,
                                            bool autoExposure, bool first)
{
    d4r::FrameSettings settings;
    settings.renderWidth = p.renderWidth;
    settings.renderHeight = p.renderHeight;
    settings.jitter[0] = p.jitterX;
    settings.jitter[1] = p.jitterY;
    settings.displayMotion = displayMotion;
    // The engine takes motion in output pixels; MV.Scale gives render pixels (display-resolution vectors are output pixels already).
    // NGX's association: the reciprocal of render * (1 / output) (1.5000001 at 720 -> 1080, not 1080 / 720)
    const float toOutput = displayMotion ? 1.0f :
        1.0f / (static_cast<float>(p.renderWidth) * (1.0f / static_cast<float>(feature.outWidth)));
    const float toOutputY = displayMotion ? 1.0f :
        1.0f / (static_cast<float>(p.renderHeight) * (1.0f / static_cast<float>(feature.outHeight)));
    settings.motionScale[0] = p.mvScaleX * toOutput;
    settings.motionScale[1] = p.mvScaleY * toOutputY;
    // MVJittered: NGX adds the change of the jitter (render pixels) to the vectors before scaling them
    // (observed with MV.Scale 1: offset = jitter - previous jitter, the first frame's previous being 0).
    // The offset is added to the raw vector, before motionScale's own output-pixel conversion, so it carries
    // no extra output/render factor in either motion mode.
    if ((feature.flags & 0x4) != 0 && p.mvScaleX != 0.0f && p.mvScaleY != 0.0f)
    {
        settings.motionOffset[0] = (p.jitterX - feature.engineJitter[0]) / p.mvScaleX;
        settings.motionOffset[1] = (p.jitterY - feature.engineJitter[1]) / p.mvScaleY;
    }
    settings.reset = p.reset != 0 || first;
    settings.depthInverted = (feature.flags & 0x8) != 0;
    settings.autoExposure = !ldr && autoExposure; // LDR: no exposure at all
    settings.exposureScale = ldr ? 1.0f : p.exposureScale;
    // NGX's automatic K measurement scales the luma by 1 / pre-exposure before the logarithm
    // (cuda_luma_convert_kernel), so the stored exposure is pre times the raw measurement and the stage's
    // 1 / pre factor recovers it; the game texture path applies the same stage factor.
    settings.preExposure = ldr ? 1.0f : p.preExposure;
    return settings;
}
