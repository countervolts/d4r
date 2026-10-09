// Native Vulkan NGX frontend. Included after the shared CUDA NGX implementation.
// Resource ABI follows NVIDIA/DLSS include/nvsdk_ngx_defs_vk.h. No SDK dependency.
// CPU staging is intentionally conservative: events keep the result in the
// evaluating command buffer without ending/submitting an application-owned list.
#include "d4r_vulkan_ngx.h"
#include "d4r_vulkan_depth.h"
#include "d4r_vulkan_staging.h"

struct NativeVkApi
{
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory = {};
    PFN_vkCreateBuffer createBuffer = nullptr;
    PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements requirements = nullptr;
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    PFN_vkBindBufferMemory bind = nullptr;
    PFN_vkMapMemory map = nullptr;
    PFN_vkUnmapMemory unmap = nullptr;
    PFN_vkCreateEvent createEvent = nullptr;
    PFN_vkDestroyEvent destroyEvent = nullptr;
    PFN_vkGetEventStatus eventStatus = nullptr;
    PFN_vkSetEvent setEvent = nullptr;
    PFN_vkResetEvent resetEvent = nullptr;
    PFN_vkCmdSetEvent cmdSetEvent = nullptr;
    PFN_vkCmdWaitEvents waitEvents = nullptr;
    PFN_vkCmdPipelineBarrier barrier = nullptr;
    PFN_vkCmdFillBuffer fill = nullptr;
    PFN_vkDeviceWaitIdle idle = nullptr;
    int(WINAPI* import)(VkDevice, uint64_t, uint64_t, CudaDevicePtr*, void**) = nullptr;
    int(WINAPI* release)(void*) = nullptr;
    uint32_t queueFamily = UINT32_MAX; // exclusive owner when the device has one compute-capable family
    // Every compute-capable family: shared buffers are CONCURRENT across them, since a game may record
    // DLSS on its async compute queue (Indiana Jones) as well as on the graphics queue (DOOM Eternal).
    uint32_t families[8] = {};
    uint32_t familyCount = 0;
    bool external = false;
};
static NativeVkApi g_nativeVk;
static D4rVkStaging g_nativeVkStaging;
static std::mutex g_nativeVkMutex; // all native API calls, including init/teardown

struct NativeVkPlane
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* mapped = nullptr;
    NgxVkImage image = {};
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    size_t rowBytes = 0;
    size_t capacity = 0;
    CudaDevicePtr cuda = 0;
    void* external = nullptr;
    bool externalOwned = false;
    D4rVkStagingSet staging;
    // Pitch-linear texture on `cuda`, handed to NGX without a buffer->array copy
    // (worker only). Pooled buffers keep it; the key detects a geometry change.
    CudaObject texture = 0;
    uint64_t textureKey = 0;
};
struct NativeVkFrame
{
    NativeVkPlane planes[5]; // color, depth, motion, optional exposure, output
    VkEvent inputReady = VK_NULL_HANDLE, outputReady = VK_NULL_HANDLE, consumed = VK_NULL_HANDLE;
    std::atomic<bool> cancel{false}, done{false};
    std::atomic<NgxResult> result{NGX_SUCCESS};
    std::thread waiter; // submission waits must not serialize across command buffers
    FrameParams params;
    uint32_t number = 0;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
};
struct NativeVkFeature
{
    Feature* feature;
    std::vector<NativeVkFrame*> frames;
    std::vector<NativeVkFrame*> reusable;
};
static std::vector<NativeVkFeature> g_nativeVkFeatures;
// Release can happen while recordings are still pending. Their GPU objects
// outlive CUDA feature state and are reclaimed at completion or shutdown.
static std::vector<NativeVkFrame*> g_nativeVkRetired;
// ROCm retains imported mappings; reuse across feature recreation rather than
// exporting and importing the same-sized allocations again on each mode change.
static std::vector<NativeVkPlane> g_nativeVkBufferPool;
static unsigned g_nativeVkExternalAllocations = 0;

static DXGI_FORMAT native_vk_format(VkFormat format, size_t& bytes)
{
    bytes = 4;
    switch (format)
    {
    case VK_FORMAT_R16G16B16A16_SFLOAT: bytes = 8; return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case VK_FORMAT_R32G32B32A32_SFLOAT: bytes = 16; return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case VK_FORMAT_R16G16_SFLOAT: return DXGI_FORMAT_R16G16_FLOAT;
    case VK_FORMAT_R32G32_SFLOAT: bytes = 8; return DXGI_FORMAT_R32G32_FLOAT;
    case VK_FORMAT_R32_SFLOAT: return DXGI_FORMAT_R32_FLOAT;
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return DXGI_FORMAT_D32_FLOAT; // depth-only buffer copy is 4 bytes
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case VK_FORMAT_R16_SFLOAT: bytes = 2; return DXGI_FORMAT_R16_FLOAT;
    case VK_FORMAT_R16_UNORM: bytes = 2; return DXGI_FORMAT_R16_UNORM;
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT: bytes = 2; return DXGI_FORMAT_D16_UNORM;
    case VK_FORMAT_R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case VK_FORMAT_B8G8R8A8_UNORM: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return DXGI_FORMAT_R11G11B10_FLOAT;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    default: bytes = 0; return DXGI_FORMAT_UNKNOWN;
    }
}

static void native_vk_free_plane(NativeVkPlane& p, bool final = false)
{
    g_nativeVkStaging.free(p.staging);
    if (p.external && !final) {
        g_nativeVkBufferPool.push_back(p); p = {}; return;
    }
    if (p.external) g.worker.call([&] {
        if (p.texture) g.cu.texObjectDestroy(p.texture);
        g.cu.memFree(p.cuda); return g_nativeVk.release(p.external);
    });
    if (p.mapped) g_nativeVk.unmap(g_nativeVk.device, p.memory);
    if (p.buffer) g_nativeVk.destroyBuffer(g_nativeVk.device, p.buffer, nullptr);
    if (p.memory) g_nativeVk.free(g_nativeVk.device, p.memory, nullptr);
    p = {};
}

// Exports a new device-local buffer and imports it into CUDA. The import can block on GPU work of
// the game's queues, which may in turn wait for one of our pending frames: callers make sure none is.
static bool native_vk_import(NativeVkPlane& p, VkDeviceSize bytes)
{
    if (g_nativeVkExternalAllocations >= 64) return false;
    VkExternalMemoryBufferCreateInfo external = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = bytes; info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = g_nativeVk.familyCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    info.queueFamilyIndexCount = g_nativeVk.familyCount > 1 ? g_nativeVk.familyCount : 0;
    info.pQueueFamilyIndices = g_nativeVk.familyCount > 1 ? g_nativeVk.families : nullptr;
    VkResult result = g_nativeVk.createBuffer(g_nativeVk.device, &info, nullptr, &p.buffer);
    VkMemoryRequirements requirements = {};
    if (result == VK_SUCCESS) g_nativeVk.requirements(g_nativeVk.device, p.buffer, &requirements);
    uint32_t type = UINT32_MAX;
    for (int pass = 0; pass < 2 && type == UINT32_MAX; ++pass)
        for (uint32_t i = 0; i < g_nativeVk.memory.memoryTypeCount; ++i) {
            const auto flags = g_nativeVk.memory.memoryTypes[i].propertyFlags;
            if ((requirements.memoryTypeBits & (1u << i)) && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (pass || !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) { type = i; break; }
        }
    VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = p.buffer;
    VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dedicated};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exportInfo};
    allocation.allocationSize = requirements.size; allocation.memoryTypeIndex = type;
    if (result == VK_SUCCESS && type != UINT32_MAX) result = g_nativeVk.allocate(g_nativeVk.device, &allocation, nullptr, &p.memory);
    else result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (result == VK_SUCCESS) result = g_nativeVk.bind(g_nativeVk.device, p.buffer, p.memory, 0);
    int imported = -1;
    if (result == VK_SUCCESS) imported = g.worker.call([&] {
        return g_nativeVk.import(g_nativeVk.device, reinterpret_cast<uint64_t>(p.memory), requirements.size, &p.cuda, &p.external);
    });
    if (result == VK_SUCCESS && imported == 0) {
        p.capacity = bytes; ++g_nativeVkExternalAllocations;
        logf("native Vulkan VRAM buffer: %llu bytes, type=%u", static_cast<unsigned long long>(bytes), type);
        return true;
    }
    logf("native Vulkan VRAM export/import unavailable: Vulkan=%d CUDA=%d; using host staging", result, imported);
    const auto image = p.image; const auto format = p.format; const auto row = p.rowBytes;
    native_vk_free_plane(p, true); p.image = image; p.format = format; p.rowBytes = row;
    // Avoid repeating an unsupported import for every subsequent plane/frame.
    g_nativeVk.external = false;
    return false;
}

static bool native_vk_vram_plane(NativeVkPlane& p, VkDeviceSize bytes, bool allowImport)
{
    if (!g_nativeVk.external) return false;
    auto best = g_nativeVkBufferPool.end();
    for (auto it = g_nativeVkBufferPool.begin(); it != g_nativeVkBufferPool.end(); ++it)
        if (it->capacity >= bytes && (best == g_nativeVkBufferPool.end() || it->capacity < best->capacity)) best = it;
    if (best != g_nativeVkBufferPool.end()) {
        const auto image = p.image; const auto format = p.format; const auto row = p.rowBytes;
        p = *best; p.image = image; p.format = format; p.rowBytes = row;
        g_nativeVkBufferPool.erase(best); return true;
    }
    return allowImport && native_vk_import(p, bytes); // otherwise host staging for this frame
}

// With no frame pending, import spare buffers for the frames that will be in flight later
// (D4R_VULKAN_SPARE_SETS, default 3), so steady state never imports while the GPU waits on us.
static void native_vk_prefill(const NativeVkFrame& frame)
{
    static const unsigned spares = env_uint("D4R_VULKAN_SPARE_SETS", 3);
    std::vector<VkDeviceSize> sizes;
    for (const auto& p : frame.planes) if (p.cuda) sizes.push_back(p.rowBytes * p.image.height);
    std::sort(sizes.begin(), sizes.end(), std::greater<>());
    std::vector<bool> claimed(g_nativeVkBufferPool.size(), false);
    for (const auto bytes : sizes)
        for (unsigned k = 0; k < spares && g_nativeVk.external; ++k) {
            size_t best = claimed.size();
            for (size_t i = 0; i < claimed.size(); ++i)
                if (!claimed[i] && g_nativeVkBufferPool[i].capacity >= bytes &&
                    (best == claimed.size() || g_nativeVkBufferPool[i].capacity < g_nativeVkBufferPool[best].capacity)) best = i;
            if (best != claimed.size()) { claimed[best] = true; continue; }
            NativeVkPlane spare;
            if (!native_vk_import(spare, bytes)) return;
            g_nativeVkBufferPool.push_back(spare); claimed.push_back(true);
        }
}

static void native_vk_free_frame(NativeVkFrame* frame)
{
    if (frame->waiter.joinable()) frame->waiter.join();
    for (auto& p : frame->planes)
    {
        native_vk_free_plane(p);
    }
    if (frame->inputReady) g_nativeVk.destroyEvent(g_nativeVk.device, frame->inputReady, nullptr);
    if (frame->outputReady) g_nativeVk.destroyEvent(g_nativeVk.device, frame->outputReady, nullptr);
    if (frame->consumed) g_nativeVk.destroyEvent(g_nativeVk.device, frame->consumed, nullptr);
    delete frame;
}

static bool native_vk_plane(NativeVkPlane& plane, const NgxVkResource* resource, int index, bool allowImport)
{
    if (!resource || resource->type != 0) return false;
    plane.image = resource->resource.image;
    const auto& image = plane.image;
    const VkImageAspectFlags aspect = image.range.aspectMask;
    if (!image.image || !image.view || !image.width || !image.height ||
        image.width > 16384 || image.height > 16384 || image.range.levelCount != 1 ||
        image.range.layerCount != 1 || !d4r_vk_copy_aspect(image.format, aspect))
        return false;
    if (d4r_vk_depth_bytes(image.format) && aspect != VK_IMAGE_ASPECT_DEPTH_BIT)
        return false; // a sampled depth view must not include the stencil aspect
    size_t bytes;
    plane.format = native_vk_format(image.format, bytes);
    if (!bytes || (index == 4 ? !D4rVkStaging::supported_output(image.format) :
                                !supported_input(static_cast<Plane>(index), plane.format))) return false;
    // Pack directly into CUDA's canonical layouts on the GPU. Sampling still
    // avoids packed depth/stencil copies and requires no transfer image usage.
    plane.format = index == 0 || index == 4 ? DXGI_FORMAT_R16G16B16A16_FLOAT :
                   index == 2 ? DXGI_FORMAT_R16G16_FLOAT : DXGI_FORMAT_R32_FLOAT;
    plane.rowBytes = (index == 0 || index == 4 ? 8 : 4) * image.width;
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = plane.rowBytes * image.height;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    const auto inputLayout = env_uint("D4R_VULKAN_INPUT_GENERAL", 0) != 0
        ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (plane.buffer && plane.capacity >= info.size)
        return g_nativeVkStaging.set(plane.staging, image.view, image.format, plane.buffer,
                                    info.size, index == 4, inputLayout);
    if (plane.buffer)
    {
        const auto descriptor = plane.image;
        const auto format = plane.format;
        const auto pitch = plane.rowBytes;
        native_vk_free_plane(plane);
        plane.image = descriptor; plane.format = format; plane.rowBytes = pitch;
    }
    if (native_vk_vram_plane(plane, info.size, allowImport))
        return g_nativeVkStaging.set(plane.staging, image.view, image.format, plane.buffer,
                                    info.size, index == 4, inputLayout);
    if (g_nativeVk.createBuffer(g_nativeVk.device, &info, nullptr, &plane.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements;
    g_nativeVk.requirements(g_nativeVk.device, plane.buffer, &requirements);
    uint32_t type = UINT32_MAX;
    const auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // Reading float4 pixels from uncached mapped GTT/BAR memory makes every
    // scalar conversion pay uncached access latency. Prefer cached, coherent
    // host memory, retaining the coherent fallback on devices without it.
    for (int cached = 1; cached >= 0 && type == UINT32_MAX; --cached)
        for (uint32_t i = 0; i < g_nativeVk.memory.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (g_nativeVk.memory.memoryTypes[i].propertyFlags & flags) == flags &&
                (!cached || (g_nativeVk.memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)))
            { type = i; break; }
    if (type == UINT32_MAX) return false;
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    static unsigned memoryReports = 0;
    if ((index == 0 || index == 4) && memoryReports < 6)
    {
        ++memoryReports;
        logf("Vulkan staging %s memory: type=%u flags=0x%x bytes=%llu",
             index == 4 ? "output" : "input", type,
             g_nativeVk.memory.memoryTypes[type].propertyFlags,
             static_cast<unsigned long long>(info.size));
    }
    if (!(g_nativeVk.allocate(g_nativeVk.device, &allocation, nullptr, &plane.memory) == VK_SUCCESS &&
           g_nativeVk.bind(g_nativeVk.device, plane.buffer, plane.memory, 0) == VK_SUCCESS &&
           g_nativeVk.map(g_nativeVk.device, plane.memory, 0, VK_WHOLE_SIZE, 0,
                          reinterpret_cast<void**>(&plane.mapped)) == VK_SUCCESS)) return false;
    plane.capacity = info.size;
    return g_nativeVkStaging.set(plane.staging, image.view, image.format, plane.buffer,
                                info.size, index == 4, inputLayout);
}

static bool native_vk_consumed(const NativeVkFrame& frame)
{
    return frame.done.load() && g_nativeVk.eventStatus(g_nativeVk.device, frame.consumed) == VK_EVENT_SET;
}

static void native_vk_reap(std::vector<NativeVkFrame*>& frames)
{
    for (auto it = frames.begin(); it != frames.end();)
        if (native_vk_consumed(**it)) { native_vk_free_frame(*it); it = frames.erase(it); }
        else ++it;
}

static NativeVkFrame* native_vk_reuse(NativeVkFeature& owner, VkCommandBuffer cmd)
{
    for (auto it = owner.frames.begin(); it != owner.frames.end();)
        if (native_vk_consumed(**it)) {
            if ((*it)->waiter.joinable()) (*it)->waiter.join();
            owner.reusable.push_back(*it); it = owner.frames.erase(it);
        } else ++it;
    if (owner.frames.size() >= 8) return nullptr;
    // Re-recording this same command buffer proves its previous recording was
    // reset. Reuse only after CPU production AND GPU consumption completed.
    for (auto it = owner.reusable.begin(); it != owner.reusable.end(); ++it)
        if ((*it)->commandBuffer == cmd) {
            auto* frame = *it; owner.reusable.erase(it);
            if (g_nativeVk.resetEvent(g_nativeVk.device, frame->inputReady) != VK_SUCCESS ||
                g_nativeVk.resetEvent(g_nativeVk.device, frame->outputReady) != VK_SUCCESS ||
                g_nativeVk.resetEvent(g_nativeVk.device, frame->consumed) != VK_SUCCESS) {
                native_vk_free_frame(frame); return nullptr;
            }
            frame->cancel = false; frame->done = false; frame->result = NGX_SUCCESS; frame->params = {};
            return frame;
        }
    while (owner.reusable.size() > 8) {
        native_vk_free_frame(owner.reusable.front()); owner.reusable.erase(owner.reusable.begin());
    }
    return nullptr;
}

// Worker only. D4R_SHIM_LINEAR_INPUTS: NGX samples the shared VRAM buffer through a
// pitch-linear texture (reported to NGX as an array by the bridge), skipping the
// buffer->array copy. Rows must meet the texture pitch alignment; others copy.
static CudaObject native_vk_linear_texture(NativeVkPlane& plane, int index)
{
    static const bool enabled = env_uint("D4R_SHIM_LINEAR_INPUTS", 0) != 0;
    if (!enabled || !plane.cuda || g.cu.registerLinearTexture == nullptr || plane.rowBytes % 256 != 0)
        return 0;
    const uint64_t key = (static_cast<uint64_t>(index) << 60) ^ (static_cast<uint64_t>(plane.rowBytes) << 32) ^
                         (static_cast<uint64_t>(plane.image.width) << 16) ^ plane.image.height;
    if (plane.texture && plane.textureKey == key) return plane.texture;
    if (plane.texture) g.cu.texObjectDestroy(plane.texture);
    plane.texture = 0;
    const Plane kind = static_cast<Plane>(index);
    struct Pitch2D { CudaDevicePtr pointer; uint32_t format, channels; size_t width, height, pitch; } pitch2D =
        {plane.cuda, plane_format(kind), plane_channels(kind), plane.image.width, plane.image.height, plane.rowBytes};
    static_assert(sizeof(Pitch2D) == 40, "CUDA_RESOURCE_DESC pitch2D");
    CudaResourceDesc resource = {};
    resource.resType = 3; // CU_RESOURCE_TYPE_PITCH2D
    memcpy(&resource.res, &pitch2D, sizeof(pitch2D));
    const CudaTextureDesc sampler = input_sampler(plane.image.width, pitch2D.channels, kind == Plane::Color);
    CudaObject object = 0;
    int result = g.cu.texObjectCreate(&object, &resource, &sampler, nullptr);
    if (result == 0)
        result = g.cu.registerLinearTexture(object, plane.image.width, plane.image.height, pitch2D.format, pitch2D.channels);
    if (result != 0) {
        logf("Vulkan linear input %d (%ux%u, pitch %zu) failed: %d; copying", index, plane.image.width,
             plane.image.height, plane.rowBytes, result);
        if (object) g.cu.texObjectDestroy(object);
        return 0;
    }
    plane.texture = object; plane.textureKey = key;
    return object;
}

// Polls the input event. D4R_VULKAN_INPUT_POLL_US=N polls every N us; unset is
// adaptive like the shim's output wait: sleep in chunks until shortly before the
// predicted readiness (smoothed over earlier frames), then yield, backing off
// when the game runs late. Detection latency is on the frame's critical path.
static VkResult native_vk_wait_input(Feature* feature, NativeVkFrame* frame)
{
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(env_uint("D4R_VULKAN_SUBMIT_TIMEOUT_MS", 30000));
    static const int fixedPoll = [] {
        const char* v = std::getenv("D4R_VULKAN_INPUT_POLL_US");
        return v != nullptr && *v != '\0' ? static_cast<int>(env_uint("D4R_VULKAN_INPUT_POLL_US", 200)) : -1;
    }();
    constexpr double kMarginUs = 120.0, kStepUs = 20.0, kChunkUs = 200.0, kSpinAfterUs = 150.0;
    const double predicted = feature->vkInputWaitEmaUs.load();
    double lastNotReadyUs = -1.0;
    VkResult ready = VK_EVENT_RESET;
    while (!frame->cancel && (ready = g_nativeVk.eventStatus(g_nativeVk.device, frame->inputReady)) == VK_EVENT_RESET &&
           std::chrono::steady_clock::now() < deadline)
    {
        if (fixedPoll >= 0) { d4r_sleep_us(static_cast<unsigned>(fixedPoll)); continue; }
        const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        lastNotReadyUs = elapsed;
        const double remaining = predicted - kMarginUs - elapsed;
        if (remaining > kStepUs) d4r_sleep_us(static_cast<unsigned>(std::min(remaining, kChunkUs)));
        else if (elapsed < predicted + kSpinAfterUs) d4r_sleep_us(0);
        else d4r_sleep_us(static_cast<unsigned>(std::min(kStepUs * (1.0 + (elapsed - predicted - kSpinAfterUs) / 160.0), kChunkUs)));
    }
    if (ready == VK_EVENT_SET && fixedPoll < 0) {
        const double waited = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        // Learn the readiness time (between the last NOT_READY and the successful query), not our own wait.
        double sample = std::min(lastNotReadyUs >= 0.0 ? 0.5 * (lastNotReadyUs + waited) : waited, 20000.0);
        feature->vkInputWaitEmaUs.store(predicted == 0.0 ? sample : 0.8 * predicted + 0.2 * sample);
    }
    return ready;
}

static void native_vk_run(Feature* feature, NativeVkFrame* frame)
{
    const auto waitStarted = ProfileClock::now();
    // Independent submission waiter: recording order need not match submit
    // order. Never block the CUDA worker on an unsubmitted command buffer.
    const VkResult ready = native_vk_wait_input(feature, frame);
    NgxResult result = NGX_FAIL_PLATFORM_ERROR;
    const auto inputsReadyAt = ProfileClock::now();
    if (!frame->cancel && ready == VK_EVENT_SET)
    {
        result = g.worker.call([&]() -> NgxResult {
            const auto stageStart = ProfileClock::now();
            InputSlot& slot = feature->inputs[0];
            for (int i = 0; i < 4; ++i)
            {
                auto& p = frame->planes[i];
                if (!p.buffer) continue;
                if (p.cuda) {
                    slot.host[i].width = p.image.width; slot.host[i].height = p.image.height;
                    slot.host[i].rowBytes = p.rowBytes;
                    continue;
                }
                Staging staging;
                staging.mapped = p.mapped;
                staging.width = p.image.width; staging.height = p.image.height;
                staging.format = p.format;
                staging.layout.Footprint.RowPitch = static_cast<UINT>(p.rowBytes);
                stage_plane(static_cast<Plane>(i), staging, slot.host[i], nullptr, i);
            }
            auto& p = frame->params;
            const auto staged = ProfileClock::now();
            bool ok = true, copied = false;
            CudaImage* images[] = {&feature->color, &feature->depth, &feature->motion, &feature->exposure};
            CudaObject linear[4] = {};
            for (int i = 0; i < 4 && ok; ++i) {
                auto& plane = frame->planes[i];
                if (!plane.buffer) continue;
                copied = copied || !(linear[i] = native_vk_linear_texture(plane, i));
                if (linear[i]) continue;
                if (!plane.cuda) {
                    ok = upload_plane(static_cast<Plane>(i), slot.host[i], *images[i], frame->number, nullptr, i);
                    continue;
                }
                ok = ensure_cuda_image(*images[i], plane.image.width, plane.image.height,
                                       plane_format(static_cast<Plane>(i)), plane_channels(static_cast<Plane>(i)), false, i == 0);
                CudaMemcpy2D copy = {};
                copy.srcMemoryType = CUDA_MEMORY_DEVICE; copy.srcDevice = plane.cuda; copy.srcPitch = plane.rowBytes;
                copy.dstMemoryType = CUDA_MEMORY_ARRAY; copy.dstArray = images[i]->array;
                copy.WidthInBytes = plane.rowBytes; copy.Height = plane.image.height;
                if (ok) ok = copy_2d(copy) == 0;
            }
            ok = ok && ensure_cuda_image(feature->output, feature->outWidth, feature->outHeight, CUDA_FORMAT_HALF, 4, true, false);
            CudaObject dilated = ok ? prepare_dilated_motion(*feature, slot, p) : 0;
            // NGX does not evaluate on the copies' stream: they must finish first. Linear
            // inputs were completed by the game's GPU before the input event.
            if (!ok || ((copied || dilated) && g.cu.ctxSynchronize() != 0)) return NGX_FAIL_PLATFORM_ERROR;
            const auto uploaded = ProfileClock::now();
            if (feature->evaluatedFrames++ == 0) p.reset = 1;
            set_evaluation_parameters(*feature, p);
            CudaObject* handles[] = {&feature->colorHandle, &feature->depthHandle, &feature->motionHandle, &feature->exposureHandle};
            for (int i = 0; i < 4; ++i)
                if (linear[i]) *handles[i] = linear[i]; // the parameters point at these variables
            if (dilated) feature->motionHandle = dilated;
            const auto& destination = frame->planes[4];
            // D4R_SHIM_OUTPUT_DIRECT: the native output kernel stores straight into the
            // shared output buffer. Set while the previous output kernel was native,
            // undone below when this frame's was not (it then wrote the array).
            static const bool outputDirect = env_uint("D4R_SHIM_OUTPUT_DIRECT", 0) != 0;
            const bool nativeOutput = g.cu.outputKernelNative == nullptr || g.cu.outputKernelNative() == 1;
            if (outputDirect && feature->outputDirectAllowed && g.cu.setArrayRedirect != nullptr &&
                feature->output.array != nullptr && (nativeOutput || feature->outputRedirected)) {
                const bool redirect = nativeOutput && destination.cuda &&
                    g.cu.setArrayRedirect(feature->output.array, destination.cuda, static_cast<uint32_t>(destination.rowBytes)) == 0;
                if (!redirect && feature->outputRedirected) g.cu.setArrayRedirect(feature->output.array, 0, 0);
                static bool logged = false;
                if (redirect && !logged) { logf("Vulkan frame %u: direct output (no array copy)", frame->number); logged = true; }
                feature->outputRedirected = redirect;
            }
            NgxResult evaluated = g.ngx.evaluateFeature(feature->cudaHandle, feature->cudaParams, nullptr);
            if (evaluated == NGX_FAIL_INVALID_PARAMETER && recreate_for_render_size(*feature, p, frame->number))
                evaluated = g.ngx.evaluateFeature(feature->cudaHandle, feature->cudaParams, nullptr);
            if (feature->outputRedirected && g.cu.outputKernelNative != nullptr && g.cu.outputKernelNative() != 1) {
                g.cu.setArrayRedirect(feature->output.array, 0, 0);
                feature->outputRedirected = false;
            }
            if (synchronize_default_stream(*feature, nullptr, &feature->evalWaitEmaUs) != 0) return NGX_FAIL_PLATFORM_ERROR;
            if (evaluated != NGX_SUCCESS) return evaluated;
            const auto evaluatedAt = ProfileClock::now();
            if (destination.cuda && feature->outputRedirected) {
                // already in the shared buffer
            } else if (destination.cuda) {
                CudaMemcpy2D copy = {};
                copy.srcMemoryType = CUDA_MEMORY_ARRAY; copy.srcArray = feature->output.array;
                copy.dstMemoryType = CUDA_MEMORY_DEVICE; copy.dstDevice = destination.cuda; copy.dstPitch = destination.rowBytes;
                copy.WidthInBytes = destination.rowBytes; copy.Height = destination.image.height;
                if (copy_2d(copy) != 0 || g.cu.ctxSynchronize() != 0) return NGX_FAIL_PLATFORM_ERROR;
            } else if (!download_output(*feature, feature->outputHost[0], frame->number, nullptr)) return NGX_FAIL_PLATFORM_ERROR;
            const auto downloaded = ProfileClock::now();
            if (!destination.cuda) std::memcpy(destination.mapped, feature->outputHost[0].bytes, feature->outputHost[0].size());
            feature->completedFrames++;
            if (frame->number <= 3 || frame->number % 120 == 0 || profile_enabled())
                logf("D4R_VULKAN_PROFILE frame=%u stage=%.3f upload=%.3f evaluate=%.3f download=%.3f encode=%.3f ms",
                     frame->number, profile_ms(stageStart, staged), profile_ms(staged, uploaded),
                     profile_ms(uploaded, evaluatedAt), profile_ms(evaluatedAt, downloaded),
                     profile_ms(downloaded, ProfileClock::now()));
            if (frame->number <= 3) logf("native Vulkan frame %u CUDA output ready (%ux%u)",
                                         frame->number, destination.image.width, destination.image.height);
            return NGX_SUCCESS;
        });
    }
    if (result != NGX_SUCCESS)
    {
        // Never leave the GPU permanently waiting on an abandoned/failed job.
        // Signal a cleared output and expose failure on the next API call.
        auto& output = frame->planes[4];
        if (output.mapped) std::memset(output.mapped, 0, output.rowBytes * output.image.height);
        // VRAM output is cleared by the recorded GPU commands before releasing
        // it to CUDA. A cancelled, unsubmitted job must not write unowned memory.
        logf("Vulkan frame %u failed/cancelled (0x%08x, input event %d)", frame->number, result, ready);
    }
    frame->result = result;
    const auto signalStarted = ProfileClock::now();
    if (g_nativeVk.setEvent(g_nativeVk.device, frame->outputReady) != VK_SUCCESS)
        frame->result = NGX_FAIL_PLATFORM_ERROR;
    frame->done = true;
    if (frame->number <= 3 || frame->number % 120 == 0 || profile_enabled())
        logf("D4R_VULKAN_WAIT frame=%u input_ready=%.3f signal=%.3f total=%.3f ms",
             frame->number, profile_ms(waitStarted, inputsReadyAt),
             profile_ms(signalStarted, ProfileClock::now()), profile_ms(waitStarted, ProfileClock::now()));
}

static void native_vk_process_exit();
static void WINAPI native_vk_shutdown_hook() { native_vk_process_exit(); }

static NgxResult native_vk_init(unsigned long long id, const wchar_t* path, VkInstance instance,
                               VkPhysicalDevice physical, VkDevice device, PFN_vkGetInstanceProcAddr instanceProc,
                               PFN_vkGetDeviceProcAddr deviceProc, const NgxFeatureCommonInfo* info,
                               unsigned int version, const ProjectIdentity* project = nullptr)
{
    std::lock_guard<std::mutex> lock(g_nativeVkMutex);
    if (!instance || !physical || !device || g.device) return NGX_FAIL_INVALID_PARAMETER;
    if (g_nativeVk.device) return g_nativeVk.device == device ? NGX_SUCCESS : NGX_FAIL_INVALID_PARAMETER;
    if (!instanceProc)
    {
        HMODULE module = LoadLibraryA("vulkan-1.dll");
        if (!module || !load_export(module, "vkGetInstanceProcAddr", instanceProc)) return NGX_FAIL_PLATFORM_ERROR;
    }
    if (!deviceProc) deviceProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(instanceProc(instance, "vkGetDeviceProcAddr"));
    auto memory = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(instanceProc(instance, "vkGetPhysicalDeviceMemoryProperties"));
    if (!deviceProc || !memory) return NGX_FAIL_PLATFORM_ERROR;
    NativeVkApi api;
    api.device = device;
    memory(physical, &api.memory);
    bool ok = true;
    auto load = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(deviceProc(device, name));
        ok &= fn != nullptr;
    };
    load(api.createBuffer, "vkCreateBuffer"); load(api.destroyBuffer, "vkDestroyBuffer");
    load(api.requirements, "vkGetBufferMemoryRequirements"); load(api.allocate, "vkAllocateMemory");
    load(api.free, "vkFreeMemory"); load(api.bind, "vkBindBufferMemory");
    load(api.map, "vkMapMemory"); load(api.unmap, "vkUnmapMemory");
    load(api.createEvent, "vkCreateEvent"); load(api.destroyEvent, "vkDestroyEvent");
    load(api.eventStatus, "vkGetEventStatus"); load(api.setEvent, "vkSetEvent");
    load(api.resetEvent, "vkResetEvent");
    load(api.cmdSetEvent, "vkCmdSetEvent"); load(api.waitEvents, "vkCmdWaitEvents");
    load(api.barrier, "vkCmdPipelineBarrier"); load(api.idle, "vkDeviceWaitIdle");
    load(api.fill, "vkCmdFillBuffer");
    if (!ok) return NGX_FAIL_PLATFORM_ERROR;
    // Match NGX's CUDA adapter identity to the supplied Vulkan physical device.
    auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(instanceProc(instance, "vkGetPhysicalDeviceProperties2"));
    if (!properties2) properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(instanceProc(instance, "vkGetPhysicalDeviceProperties2KHR"));
    if (properties2)
    {
        VkPhysicalDeviceIDProperties identity = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &identity};
        properties2(physical, &properties);
        if (identity.deviceLUIDValid)
        {
            uint32_t low, high;
            std::memcpy(&low, identity.deviceLUID, 4); std::memcpy(&high, identity.deviceLUID + 4, 4);
            char value[24];
            std::snprintf(value, sizeof(value), "0x%08x", low); SetEnvironmentVariableA("D4R_CUDA_LUID_LOW", value);
            std::snprintf(value, sizeof(value), "0x%08x", high); SetEnvironmentVariableA("D4R_CUDA_LUID_HIGH", value);
            std::snprintf(value, sizeof(value), "%u", identity.deviceNodeMask); SetEnvironmentVariableA("D4R_CUDA_NODE_MASK", value);
        }
    }
    D4rVkStaging staging;
    if (!staging.init(device, deviceProc)) { staging.destroy(); return NGX_FAIL_PLATFORM_ERROR; }
    const auto result = initialize(id, path, nullptr, info, version, project);
    if (result == NGX_SUCCESS)
    {
        auto families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(instanceProc(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
        if (families) {
            uint32_t count = 0; families(physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> properties(count); families(physical, &count, properties.data());
            for (uint32_t i = 0; i < count && api.familyCount < 8; ++i)
                if (properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) api.families[api.familyCount++] = i;
            if (api.familyCount == 1) api.queueFamily = api.families[0];
        }
        load_export(g.cuda, "d4rImportVulkanMemory", api.import);
        load_export(g.cuda, "d4rReleaseVulkanMemory", api.release);
        // NGX's own detach tears CUDA down before ours runs; the bridge calls this first.
        void(WINAPI* setShutdownHook)(void(WINAPI*)()) = nullptr;
        if (load_export(g.cuda, "d4rSetShutdownHook", setShutdownHook)) setShutdownHook(native_vk_shutdown_hook);
        else logf("CUDA bridge lacks d4rSetShutdownHook: a game exit during DLSS can still reset the GPU ring");
        api.external = env_uint("D4R_VULKAN_VRAM_INTEROP", 0) != 0 && api.import && api.release &&
                       api.familyCount > 0 && deviceProc(device, "vkGetMemoryWin32HandleKHR") != nullptr;
        g_nativeVk = api;
        g_nativeVkStaging = staging;
        logf("native Vulkan NGX ready: device=%p, sampled/storage compute staging (experimental), VRAM interop=%d, "
             "%u compute-capable queue famil%s", static_cast<void*>(device), api.external, api.familyCount,
             api.familyCount == 1 ? "y" : "ies (concurrent sharing)");
    }
    else staging.destroy();
    return result;
}

static void native_vk_ownership(VkCommandBuffer cmd, NativeVkPlane& p, bool external)
{
    if (!p.cuda || p.externalOwned == external) return;
    VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b.buffer = p.buffer; b.offset = 0; b.size = VK_WHOLE_SIZE;
    // Concurrent buffers are shared by every family the game may record on (RADV keeps buffers
    // uncompressed, so CUDA's view needs no transfer); an exclusive one moves to/from EXTERNAL.
    const bool concurrent = g_nativeVk.familyCount > 1;
    b.srcQueueFamilyIndex = concurrent ? VK_QUEUE_FAMILY_IGNORED : external ? g_nativeVk.queueFamily : VK_QUEUE_FAMILY_EXTERNAL;
    b.dstQueueFamilyIndex = concurrent ? VK_QUEUE_FAMILY_IGNORED : external ? VK_QUEUE_FAMILY_EXTERNAL : g_nativeVk.queueFamily;
    b.srcAccessMask = external ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : 0;
    b.dstAccessMask = external ? 0 : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    g_nativeVk.barrier(cmd, external ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       external ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 1, &b, 0, nullptr);
    p.externalOwned = external;
}

D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_Init(unsigned long long id, const wchar_t* path, VkInstance instance,
    VkPhysicalDevice physical, VkDevice device, PFN_vkGetInstanceProcAddr ip, PFN_vkGetDeviceProcAddr dp,
    const NgxFeatureCommonInfo* info, unsigned int version)
{ return native_vk_init(id, path, instance, physical, device, ip, dp, info, version); }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_Init_Ext(unsigned long long id, const wchar_t* path, VkInstance instance,
    VkPhysicalDevice physical, VkDevice device, unsigned int version, const NgxFeatureCommonInfo* info)
{ return native_vk_init(id, path, instance, physical, device, nullptr, nullptr, info, version); }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_Init_Ext2(unsigned long long id, const wchar_t* path, VkInstance instance,
    VkPhysicalDevice physical, VkDevice device, PFN_vkGetInstanceProcAddr ip, PFN_vkGetDeviceProcAddr dp,
    unsigned int version, const NgxFeatureCommonInfo* info)
{ return native_vk_init(id, path, instance, physical, device, ip, dp, info, version); }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_Init_with_ProjectID(const char* id, int engine, const char* engineVersion,
    const wchar_t* path, VkInstance instance, VkPhysicalDevice physical, VkDevice device,
    PFN_vkGetInstanceProcAddr ip, PFN_vkGetDeviceProcAddr dp, const NgxFeatureCommonInfo* info, unsigned int version)
{
    ProjectIdentity project{id ? id : "", engine, engineVersion ? engineVersion : ""};
    return native_vk_init(0, path, instance, physical, device, ip, dp, info, version, &project);
}
// The NGX core/OptiScaler project-ID exports put SDKVersion before FeatureInfo.
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_Init_ProjectID(const char* id, int engine, const char* engineVersion,
    const wchar_t* path, VkInstance instance, VkPhysicalDevice physical, VkDevice device,
    PFN_vkGetInstanceProcAddr ip, PFN_vkGetDeviceProcAddr dp, unsigned int version, const NgxFeatureCommonInfo* info)
{ return NVSDK_NGX_VULKAN_Init_with_ProjectID(id, engine, engineVersion, path, instance, physical, device, ip, dp, info, version); }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_Init_ProjectID_Ext(const char* id, int engine, const char* engineVersion,
    const wchar_t* path, VkInstance instance, VkPhysicalDevice physical, VkDevice device,
    PFN_vkGetInstanceProcAddr ip, PFN_vkGetDeviceProcAddr dp, unsigned int version, const NgxFeatureCommonInfo* info)
{ return NVSDK_NGX_VULKAN_Init_ProjectID(id, engine, engineVersion, path, instance, physical, device, ip, dp, version, info); }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_GetParameters(void** p)
{ return p ? NVSDK_NGX_D3D12_GetParameters(p) : NGX_FAIL_INVALID_PARAMETER; }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_AllocateParameters(void** p)
{ return p ? NVSDK_NGX_D3D12_AllocateParameters(p) : NGX_FAIL_INVALID_PARAMETER; }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_GetCapabilityParameters(void** p)
{ return p ? NVSDK_NGX_D3D12_GetCapabilityParameters(p) : NGX_FAIL_INVALID_PARAMETER; }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_DestroyParameters(void* p)
{ return NVSDK_NGX_D3D12_DestroyParameters(p); }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_GetScratchBufferSize(unsigned int feature, const void* p, size_t* size)
{ return size ? NVSDK_NGX_D3D12_GetScratchBufferSize(feature, p, size) : NGX_FAIL_INVALID_PARAMETER; }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_RequiredExtensions(unsigned int* ic, const char*** ie, unsigned int* dc, const char*** de)
{
    if (!ic || !ie || !dc || !de) return NGX_FAIL_INVALID_PARAMETER;
    *ic = *dc = 0; *ie = *de = nullptr;
    return NGX_SUCCESS; // CPU staging requires no NVIDIA or external-memory extensions
}
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(const void*, uint32_t* count, VkExtensionProperties** p)
{
    // Callers query the count with a null list first (Streamline does).
    if (!count) return NGX_FAIL_INVALID_PARAMETER;
    *count = 0;
    if (p) *p = nullptr;
    return NGX_SUCCESS;
}
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(VkInstance, VkPhysicalDevice,
    const void* info, uint32_t* count, VkExtensionProperties** p)
{ return NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(info, count, p); }
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_GetFeatureRequirements(VkInstance, VkPhysicalDevice, const void*, NgxFeatureRequirement* p)
{ return p ? NVSDK_NGX_D3D12_GetFeatureRequirements(nullptr, nullptr, p) : NGX_FAIL_INVALID_PARAMETER; }

D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_CreateFeature(VkCommandBuffer, unsigned int id, void* p, NgxHandle** handle)
{
    std::lock_guard<std::mutex> lock(g_nativeVkMutex);
    if (!g_nativeVk.device) return NGX_FAIL_NOT_INITIALIZED;
    const auto result = create_feature(id, p, handle, true);
    if (result == NGX_SUCCESS) g_nativeVkFeatures.push_back({find_feature(*handle), {}});
    return result;
}
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_CreateFeature1(VkDevice device, VkCommandBuffer cmd, unsigned int id, void* p, NgxHandle** handle)
{
    if (device != g_nativeVk.device) return NGX_FAIL_INVALID_PARAMETER;
    return NVSDK_NGX_VULKAN_CreateFeature(cmd, id, p, handle);
}

D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_EvaluateFeature(VkCommandBuffer cmd, const NgxHandle* handle, void* p, void*)
{
    std::lock_guard<std::mutex> lock(g_nativeVkMutex);
    const auto apiStarted = ProfileClock::now();
    if (!g_nativeVk.device || !g.ngxInitialized) return NGX_FAIL_NOT_INITIALIZED;
    auto it = std::find_if(g_nativeVkFeatures.begin(), g_nativeVkFeatures.end(),
                           [&](const auto& f) { return &f.feature->handle == handle; });
    if (!cmd || !p || it == g_nativeVkFeatures.end()) return NGX_FAIL_INVALID_PARAMETER;
    auto& owner = *it;
    if (owner.feature->retiring) return NGX_FAIL_INVALID_PARAMETER;
    for (auto* frame : owner.frames)
        if (frame->done && frame->result != NGX_SUCCESS) return frame->result;
    native_vk_reap(g_nativeVkRetired);
    auto* reusable = native_vk_reuse(owner, cmd);
    const auto reapedAt = ProfileClock::now();
    if (owner.frames.size() >= 8) return NGX_FAIL_PLATFORM_ERROR;
    auto* frame = reusable ? reusable : new NativeVkFrame;
    frame->commandBuffer = cmd;
    // An import while any recording of ours may be waiting on the GPU can deadlock against it
    // (Indiana Jones: 10 s compute ring timeout). Then only pooled buffers or host staging are used.
    bool pending = !g_nativeVkRetired.empty();
    for (const auto& f : g_nativeVkFeatures)
        for (const auto* other : f.frames) pending = pending || !native_vk_consumed(*other);
    const char* names[] = {"Color", "Depth", "MotionVectors", "ExposureTexture", "Output"};
    for (int i = 0; i < 5; ++i)
    {
        void* resource = nullptr;
        d4r_ngx_get_void(p, names[i], &resource);
        if (i == 3 && !resource) { native_vk_free_plane(frame->planes[3]); continue; }
        if (!native_vk_plane(frame->planes[i], static_cast<NgxVkResource*>(resource), i, !pending))
        {
            if (resource)
            {
                const auto* descriptor = static_cast<NgxVkResource*>(resource);
                const auto& image = descriptor->resource.image;
                logf("Vulkan rejected resource %s: type=%d image=%p view=%p format=%d "
                     "size=%ux%u aspect=0x%x mip=%u+%u layer=%u+%u", names[i], descriptor->type,
                     reinterpret_cast<void*>(image.image), reinterpret_cast<void*>(image.view), image.format,
                     image.width, image.height, image.range.aspectMask, image.range.baseMipLevel,
                     image.range.levelCount, image.range.baseArrayLayer, image.range.layerCount);
            }
            else logf("Vulkan missing resource %s", names[i]);
            native_vk_free_frame(frame); return NGX_FAIL_UNSUPPORTED_FORMAT;
        }
    }
    if (!pending && g_nativeVk.external) native_vk_prefill(*frame);
    auto* feature = owner.feature;
    const auto allocatedAt = ProfileClock::now();
    if (feature->frame == 0)
        for (int i = 0; i < 5; ++i)
        {
            const auto& plane = frame->planes[i];
            if (plane.buffer) logf("Vulkan resource %s: format=%d size=%ux%u aspect=0x%x copyAspect=0x%x rowBytes=%zu",
                                   names[i], plane.image.format, plane.image.width, plane.image.height,
                                   plane.image.range.aspectMask, d4r_vk_copy_aspect(plane.image.format, plane.image.range.aspectMask),
                                   plane.rowBytes);
        }
    if (frame->planes[4].image.width != feature->outWidth || frame->planes[4].image.height != feature->outHeight)
    { native_vk_free_frame(frame); return NGX_FAIL_INVALID_PARAMETER; }
    auto& params = frame->params;
    params.jitterX = get_float_or(p, "Jitter.Offset.X", 0); params.jitterY = get_float_or(p, "Jitter.Offset.Y", 0);
    params.mvScaleX = get_float_or(p, "MV.Scale.X", 1); params.mvScaleY = get_float_or(p, "MV.Scale.Y", 1);
    params.sharpness = get_float_or(p, "Sharpness", 0); params.reset = get_int_or(p, "Reset", 0);
    params.preExposure = get_float_or(p, "DLSS.Pre.Exposure", 1); params.exposureScale = get_float_or(p, "DLSS.Exposure.Scale", 1);
    params.frameTime = get_float_or(p, "FrameTimeDeltaInMsec", 16.6f);
    params.renderWidth = get_uint_or(p, "DLSS.Render.Subrect.Dimensions.Width", feature->width);
    params.renderHeight = get_uint_or(p, "DLSS.Render.Subrect.Dimensions.Height", feature->height);
    if (!params.renderWidth) params.renderWidth = feature->width;
    if (!params.renderHeight) params.renderHeight = feature->height;
    params.colorBaseX = get_uint_or(p, "DLSS.Input.Color.Subrect.Base.X", 0);
    params.colorBaseY = get_uint_or(p, "DLSS.Input.Color.Subrect.Base.Y", 0);
    params.depthBaseX = get_uint_or(p, "DLSS.Input.Depth.Subrect.Base.X", 0);
    params.depthBaseY = get_uint_or(p, "DLSS.Input.Depth.Subrect.Base.Y", 0);
    params.mvBaseX = get_uint_or(p, "DLSS.Input.MV.Subrect.Base.X", 0);
    params.mvBaseY = get_uint_or(p, "DLSS.Input.MV.Subrect.Base.Y", 0);
    params.outputBaseX = get_uint_or(p, "DLSS.Output.Subrect.Base.X", 0);
    params.outputBaseY = get_uint_or(p, "DLSS.Output.Subrect.Base.Y", 0);
    params.hasExposure = frame->planes[3].buffer != VK_NULL_HANDLE;
    params.invertX = get_int_or(p, "DLSS.Indicator.Invert.X.Axis", 0);
    params.invertY = get_int_or(p, "DLSS.Indicator.Invert.Y.Axis", 0);
    // Output subrect conversion and auxiliary masks are not implemented yet.
    void* mask = nullptr;
    d4r_ngx_get_void(p, "DLSS.Input.Bias.Current.Color.Mask", &mask);
    void* transparency = nullptr; d4r_ngx_get_void(p, "TransparencyMask", &transparency);
    if (params.outputBaseX || params.outputBaseY || mask || transparency ||
        params.colorBaseX > frame->planes[0].image.width || params.colorBaseY > frame->planes[0].image.height ||
        params.renderWidth > frame->planes[0].image.width - params.colorBaseX ||
        params.renderHeight > frame->planes[0].image.height - params.colorBaseY)
    { native_vk_free_frame(frame); return NGX_FAIL_INVALID_PARAMETER; }
    VkEventCreateInfo event = {VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    if (!frame->inputReady && (g_nativeVk.createEvent(g_nativeVk.device, &event, nullptr, &frame->inputReady) != VK_SUCCESS ||
        g_nativeVk.createEvent(g_nativeVk.device, &event, nullptr, &frame->outputReady) != VK_SUCCESS ||
        g_nativeVk.createEvent(g_nativeVk.device, &event, nullptr, &frame->consumed) != VK_SUCCESS))
    { native_vk_free_frame(frame); return NGX_FAIL_PLATFORM_ERROR; }
    frame->number = ++feature->frame;
    try { frame->waiter = std::thread([feature, frame] { native_vk_run(feature, frame); }); }
    catch (...) { native_vk_free_frame(frame); return NGX_FAIL_PLATFORM_ERROR; }
    // All validation/allocation precedes the first recorded command.
    VkMemoryBarrier readable = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    readable.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    readable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    g_nativeVk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &readable, 0, nullptr, 0, nullptr);
    for (int i = 0; i < 4; ++i)
    {
        auto& plane = frame->planes[i];
        if (!plane.buffer) continue;
        native_vk_ownership(cmd, plane, false);
        g_nativeVkStaging.record(cmd, plane.staging, plane.image.width, plane.image.height, i + 1);
    }
    auto& resultPlane = frame->planes[4];
    if (resultPlane.cuda) {
        native_vk_ownership(cmd, resultPlane, false);
        g_nativeVk.fill(cmd, resultPlane.buffer, 0, resultPlane.rowBytes * resultPlane.image.height, 0);
    }
    for (auto& plane : frame->planes) native_vk_ownership(cmd, plane, true);
    VkMemoryBarrier readback = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    readback.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    g_nativeVk.barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readback, 0, nullptr, 0, nullptr);
    g_nativeVk.cmdSetEvent(cmd, frame->inputReady, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkMemoryBarrier upload = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    upload.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; upload.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    g_nativeVk.waitEvents(cmd, 1, &frame->outputReady, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          1, &upload, 0, nullptr, 0, nullptr);
    auto& output = frame->planes[4];
    native_vk_ownership(cmd, output, false);
    g_nativeVkStaging.record(cmd, output.staging, output.image.width, output.image.height);
    VkMemoryBarrier visible = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    visible.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    visible.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    g_nativeVk.barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       0, 1, &visible, 0, nullptr, 0, nullptr);
    // CPU production is not GPU consumption. This marker follows the image
    // write and all earlier commands which reference the staging objects.
    g_nativeVk.cmdSetEvent(cmd, frame->consumed, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    owner.frames.push_back(frame);
    if (frame->number <= 3 || frame->number % 120 == 0 || profile_enabled())
        logf("D4R_VULKAN_API frame=%u reclaim=%.3f allocate=%.3f record=%.3f total=%.3f ms",
             frame->number, profile_ms(apiStarted, reapedAt), profile_ms(reapedAt, allocatedAt),
             profile_ms(allocatedAt, ProfileClock::now()), profile_ms(apiStarted, ProfileClock::now()));
    return NGX_SUCCESS;
}
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_EvaluateFeature_C(VkCommandBuffer cmd, const NgxHandle* handle, void* p, void* callback)
{ return NVSDK_NGX_VULKAN_EvaluateFeature(cmd, handle, p, callback); }

static NgxResult native_vk_release(std::vector<NativeVkFeature>::iterator it)
{
    for (auto* frame : it->frames) frame->cancel = true;
    // Every waiter signals output before CUDA state disappears.
    for (auto* frame : it->frames)
        if (frame->waiter.joinable()) frame->waiter.join();
    for (auto* frame : it->frames)
        if (native_vk_consumed(*frame)) native_vk_free_frame(frame);
        else g_nativeVkRetired.push_back(frame);
    for (auto* frame : it->reusable) native_vk_free_frame(frame);
    logf("native Vulkan feature %u released after %u completed frames",
         it->feature->handle.Id, it->feature->completedFrames.load());
    it->feature->retiring = true;
    release_feature(it->feature);
    g_nativeVkFeatures.erase(it);
    return NGX_SUCCESS;
}
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_ReleaseFeature(NgxHandle* handle)
{
    std::lock_guard<std::mutex> lock(g_nativeVkMutex);
    auto it = std::find_if(g_nativeVkFeatures.begin(), g_nativeVkFeatures.end(),
                           [&](const auto& f) { return &f.feature->handle == handle; });
    return it == g_nativeVkFeatures.end() ? NGX_FAIL_INVALID_PARAMETER : native_vk_release(it);
}
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_Shutdown()
{
    std::lock_guard<std::mutex> lock(g_nativeVkMutex);
    NgxResult result = NGX_SUCCESS;
    while (!g_nativeVkFeatures.empty())
        if (native_vk_release(g_nativeVkFeatures.begin()) != NGX_SUCCESS) result = NGX_FAIL_PLATFORM_ERROR;
    if (g_nativeVk.device)
    {
        // Shutdown is the application's teardown boundary: it must externally
        // synchronize its queues and discard unsubmitted recordings first.
        if (g_nativeVk.idle(g_nativeVk.device) != VK_SUCCESS) return NGX_FAIL_PLATFORM_ERROR;
        for (auto* frame : g_nativeVkRetired) native_vk_free_frame(frame);
        g_nativeVkRetired.clear();
        for (auto& plane : g_nativeVkBufferPool) native_vk_free_plane(plane, true);
        g_nativeVkBufferPool.clear(); g_nativeVkExternalAllocations = 0;
        g_nativeVkStaging.destroy();
        NVSDK_NGX_D3D12_Shutdown();
        g_nativeVk = {};
    }
    return result;
}
// DLL_PROCESS_DETACH. A game may exit without NGX shutdown while a submitted
// command buffer still waits in vkCmdWaitEvents for an output event that only the
// (now killed) waiter thread would set. The wait sits inside the GPU ring, so the
// kernel cannot cancel it: the ring times out and resets 10 s after the exit.
// Release every such wait. No locking: other threads are gone, and one may have
// died holding the mutex.
static void native_vk_process_exit()
{
    static std::atomic<bool> ran{false}; // the bridge's hook, then DLL_PROCESS_DETACH
    if (!g_nativeVk.device || !g_nativeVk.setEvent || ran.exchange(true)) return;
    unsigned released = 0;
    auto release = [&](NativeVkFrame* frame) {
        if (frame && !frame->done.load() && frame->outputReady &&
            g_nativeVk.setEvent(g_nativeVk.device, frame->outputReady) == VK_SUCCESS) ++released;
    };
    for (auto& owner : g_nativeVkFeatures)
        for (auto* frame : owner.frames) release(frame);
    for (auto* frame : g_nativeVkRetired) release(frame);
    if (released) logf("process exit: released %u pending Vulkan output wait(s)", released);
}
D4R_EXPORT NgxResult NVSDK_NGX_VULKAN_Shutdown1(VkDevice device)
{
    if (device != g_nativeVk.device) return NGX_FAIL_INVALID_PARAMETER;
    return NVSDK_NGX_VULKAN_Shutdown();
}
