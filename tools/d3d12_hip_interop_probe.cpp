// Probe for zero-copy D3D12 <-> CUDA/HIP sharing under Proton.
//
// D3D12 shared NT handles are a dead end under Wine (no unix fd behind them;
// vkd3d-proton exports only textures), so this goes through vkd3d-proton's
// Vulkan interop instead:
//   - a VkBuffer in device-local memory, allocated on vkd3d-proton's own
//     VkDevice with VkExportMemoryAllocateInfo (winevulkan makes it a host
//     OPAQUE_FD export),
//   - imported into HIP by the d4r nvcuda bridge (d4rImportVulkanMemory),
//   - filled from / copied into a D3D12 texture by raw Vulkan copy commands
//     recorded inside a D3D12 command list (ID3D12DXVKInteropDevice1::
//     BeginVkCommandBufferInterop), which is where the NGX shim records its
//     staging copies today.
//
// usage: d3d12_hip_interop_probe.exe NVCUDA_BRIDGE_DLL REPORT_FILE [WIDTH HEIGHT]
// (Proton does not connect the console, so results go to REPORT_FILE.)
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <windows.h>
#include <d3d12.h>
#include <vulkan/vulkan_core.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using CUresult = int;
using CUdeviceptr = unsigned long long;

// vkd3d-proton's interop interface (include/vkd3d_device_vkd3d_ext.idl).
MIDL_INTERFACE("39da4e09-bd1c-4198-9fae-86bbe3be41fd")
ID3D12DXVKInteropDevice : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetDXGIAdapter(REFIID iid, void** object) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetInstanceExtensions(UINT* count, const char** extensions) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceExtensions(UINT* count, const char** extensions) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceFeatures(const VkPhysicalDeviceFeatures2** features) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanHandles(VkInstance* instance, VkPhysicalDevice* physical, VkDevice* device) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanQueueInfo(ID3D12CommandQueue* queue, VkQueue* vk_queue, UINT32* family) = 0;
    virtual void STDMETHODCALLTYPE GetVulkanImageLayout(ID3D12Resource* resource, D3D12_RESOURCE_STATES state,
                                                        VkImageLayout* layout) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo(ID3D12Resource* resource, UINT64* handle, UINT64* offset) = 0;
    virtual HRESULT STDMETHODCALLTYPE LockCommandQueue(ID3D12CommandQueue* queue) = 0;
    virtual HRESULT STDMETHODCALLTYPE UnlockCommandQueue(ID3D12CommandQueue* queue) = 0;
};

MIDL_INTERFACE("902d8115-59eb-4406-9518-fe00f991ee65")
ID3D12DXVKInteropDevice1 : public ID3D12DXVKInteropDevice
{
    virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo1(ID3D12Resource* resource, UINT64* handle, UINT64* offset,
                                                             VkFormat* format) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandQueue(const D3D12_COMMAND_QUEUE_DESC* desc, UINT32 family,
                                                                ID3D12CommandQueue** queue) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandAllocator(D3D12_COMMAND_LIST_TYPE type, UINT32 family,
                                                                    ID3D12CommandAllocator** allocator) = 0;
    virtual HRESULT STDMETHODCALLTYPE BeginVkCommandBufferInterop(ID3D12CommandList* list, VkCommandBuffer* buffer) = 0;
    virtual HRESULT STDMETHODCALLTYPE EndVkCommandBufferInterop(ID3D12CommandList* list) = 0;
};
__CRT_UUID_DECL(ID3D12DXVKInteropDevice1, 0x902d8115, 0x59eb, 0x4406, 0x95, 0x18, 0xfe, 0x00, 0xf9, 0x91, 0xee, 0x65)

static ID3D12Device* g_device;
static ID3D12CommandQueue* g_queue;
static ID3D12CommandAllocator* g_allocator;
static ID3D12GraphicsCommandList* g_list;
static ID3D12Fence* g_fence;
static HANDLE g_event;
static UINT64 g_fenceValue;

static bool check(HRESULT hr, const char* what)
{
    if (FAILED(hr))
        std::printf("%s failed: 0x%08lx\n", what, hr);
    return SUCCEEDED(hr);
}

static bool flush()
{
    if (!check(g_list->Close(), "Close"))
        return false;
    ID3D12CommandList* lists[] = {g_list};
    g_queue->ExecuteCommandLists(1, lists);
    g_queue->Signal(g_fence, ++g_fenceValue);
    g_fence->SetEventOnCompletion(g_fenceValue, g_event);
    WaitForSingleObject(g_event, INFINITE);
    return check(g_allocator->Reset(), "allocator Reset") && check(g_list->Reset(g_allocator, nullptr), "list Reset");
}

static double elapsed_ms(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static ID3D12Resource* create_buffer(D3D12_HEAP_TYPE type, UINT64 size, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = type;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* resource = nullptr;
    if (!check(g_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                 __uuidof(ID3D12Resource), reinterpret_cast<void**>(&resource)),
               "CreateCommittedResource(buffer)"))
        return nullptr;
    return resource;
}

static void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    g_list->ResourceBarrier(1, &barrier);
}

static uint64_t pattern(uint64_t i)
{
    return (i * 0x9E3779B97F4A7C15ull) ^ (i >> 7);
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s NVCUDA_BRIDGE_DLL REPORT_FILE [WIDTH HEIGHT]\n", argv[0]);
        return 2;
    }
    if (std::freopen(argv[2], "w", stdout) == nullptr)
        return 2;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const UINT width = argc > 4 ? static_cast<UINT>(std::strtoul(argv[3], nullptr, 0)) : 2560;
    const UINT height = argc > 4 ? static_cast<UINT>(std::strtoul(argv[4], nullptr, 0)) : 1440;
    const UINT64 texel = 8; // R16G16B16A16_FLOAT, the DLSS output format
    const UINT64 pitch = width * texel;
    const UINT64 bytes = pitch * height;
    const UINT64 words = bytes / 8;
    if (pitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT != 0)
    {
        std::printf("width must make a 256-byte aligned row pitch\n");
        return 2;
    }

    if (!check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device),
                                 reinterpret_cast<void**>(&g_device)),
               "D3D12CreateDevice"))
        return 1;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (!check(g_device->CreateCommandQueue(&queueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&g_queue)),
               "CreateCommandQueue") ||
        !check(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
                                                reinterpret_cast<void**>(&g_allocator)),
               "CreateCommandAllocator") ||
        !check(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_allocator, nullptr,
                                           __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&g_list)),
               "CreateCommandList") ||
        !check(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&g_fence)),
               "CreateFence"))
        return 1;
    g_event = CreateEventA(nullptr, FALSE, FALSE, nullptr);

    ID3D12DXVKInteropDevice1* interop = nullptr;
    if (!check(g_device->QueryInterface(__uuidof(ID3D12DXVKInteropDevice1), reinterpret_cast<void**>(&interop)),
               "QueryInterface(ID3D12DXVKInteropDevice1)"))
        return 1;
    VkInstance vkInstance = VK_NULL_HANDLE;
    VkPhysicalDevice vkPhysical = VK_NULL_HANDLE;
    VkDevice vkDevice = VK_NULL_HANDLE;
    if (!check(interop->GetVulkanHandles(&vkInstance, &vkPhysical, &vkDevice), "GetVulkanHandles"))
        return 1;
    UINT extensionCount = 0;
    interop->GetDeviceExtensions(&extensionCount, nullptr);
    std::vector<const char*> extensions(extensionCount);
    interop->GetDeviceExtensions(&extensionCount, extensions.data());
    bool win32Memory = false;
    for (const char* name : extensions)
        win32Memory |= std::strcmp(name, "VK_KHR_external_memory_win32") == 0;
    std::printf("vkd3d-proton VkDevice %p, %u extensions, VK_KHR_external_memory_win32 %s\n",
                static_cast<void*>(vkDevice), extensionCount, win32Memory ? "enabled" : "MISSING");

    HMODULE vulkan = LoadLibraryA("vulkan-1.dll");
    auto getInstanceProc = vulkan != nullptr ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                                                   reinterpret_cast<void*>(GetProcAddress(vulkan, "vkGetInstanceProcAddr")))
                                             : nullptr;
    if (getInstanceProc == nullptr)
    {
        std::printf("vulkan-1.dll unavailable\n");
        return 1;
    }
    auto getDeviceProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(getInstanceProc(vkInstance, "vkGetDeviceProcAddr"));
    auto getMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        getInstanceProc(vkInstance, "vkGetPhysicalDeviceMemoryProperties"));
#define DEVICE_PROC(name) auto name = reinterpret_cast<PFN_##name>(getDeviceProc(vkDevice, #name))
    DEVICE_PROC(vkCreateBuffer);
    DEVICE_PROC(vkGetBufferMemoryRequirements);
    DEVICE_PROC(vkAllocateMemory);
    DEVICE_PROC(vkBindBufferMemory);
    DEVICE_PROC(vkCmdPipelineBarrier);
    DEVICE_PROC(vkCmdCopyImageToBuffer);
    DEVICE_PROC(vkCmdCopyBufferToImage);
#undef DEVICE_PROC

    // Exportable device-local buffer on vkd3d-proton's device.
    VkExternalMemoryBufferCreateInfo externalBuffer = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    externalBuffer.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &externalBuffer};
    bufferInfo.size = bytes;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer vkBuffer = VK_NULL_HANDLE;
    VkResult vr = vkCreateBuffer(vkDevice, &bufferInfo, nullptr, &vkBuffer);
    VkMemoryRequirements requirements = {};
    vkGetBufferMemoryRequirements(vkDevice, vkBuffer, &requirements);
    VkPhysicalDeviceMemoryProperties memoryProperties = {};
    getMemoryProperties(vkPhysical, &memoryProperties);
    int memoryType = -1;
    for (int pass = 0; pass < 2 && memoryType < 0; ++pass)
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount && memoryType < 0; ++i)
        {
            const VkMemoryPropertyFlags flags = memoryProperties.memoryTypes[i].propertyFlags;
            if ((requirements.memoryTypeBits & (1u << i)) && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (pass == 1 || !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)))
                memoryType = static_cast<int>(i);
        }
    VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = vkBuffer;
    VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dedicated};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkMemoryAllocateInfo allocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exportInfo};
    allocateInfo.allocationSize = requirements.size;
    allocateInfo.memoryTypeIndex = static_cast<uint32_t>(memoryType);
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
    if (vr == VK_SUCCESS && memoryType >= 0)
        vr = vkAllocateMemory(vkDevice, &allocateInfo, nullptr, &vkMemory);
    if (vr == VK_SUCCESS)
        vr = vkBindBufferMemory(vkDevice, vkBuffer, vkMemory, 0);
    std::printf("exportable VkBuffer: %llu bytes (allocation %llu, memory type %d): VkResult %d\n",
                static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(requirements.size), memoryType,
                vr);
    if (vr != VK_SUCCESS || memoryType < 0)
        return 1;

    HMODULE bridge = LoadLibraryA(argv[1]);
    if (bridge == nullptr)
    {
        std::printf("LoadLibrary(%s) failed: %lu\n", argv[1], GetLastError());
        return 1;
    }
    using ImportFn = CUresult(WINAPI*)(void*, uint64_t, uint64_t, CUdeviceptr*, void**);
    using DtoHFn = CUresult(WINAPI*)(void*, CUdeviceptr, size_t);
    using MemsetFn = CUresult(WINAPI*)(CUdeviceptr, unsigned int, size_t);
    using SyncFn = CUresult(WINAPI*)();
    using AllocFn = CUresult(WINAPI*)(CUdeviceptr*, size_t);
    using DtoDFn = CUresult(WINAPI*)(CUdeviceptr, CUdeviceptr, size_t);
    auto import = reinterpret_cast<ImportFn>(reinterpret_cast<void*>(GetProcAddress(bridge, "d4rImportVulkanMemory")));
    auto dtoh = reinterpret_cast<DtoHFn>(reinterpret_cast<void*>(GetProcAddress(bridge, "cuMemcpyDtoH")));
    auto memset32 = reinterpret_cast<MemsetFn>(reinterpret_cast<void*>(GetProcAddress(bridge, "cuMemsetD32")));
    auto synchronize = reinterpret_cast<SyncFn>(reinterpret_cast<void*>(GetProcAddress(bridge, "cuCtxSynchronize")));
    auto alloc = reinterpret_cast<AllocFn>(reinterpret_cast<void*>(GetProcAddress(bridge, "cuMemAlloc")));
    auto dtod = reinterpret_cast<DtoDFn>(reinterpret_cast<void*>(GetProcAddress(bridge, "cuMemcpyDtoD")));
    if (import == nullptr || dtoh == nullptr || memset32 == nullptr || synchronize == nullptr || alloc == nullptr ||
        dtod == nullptr)
    {
        std::printf("bridge exports missing\n");
        return 1;
    }
    CUdeviceptr device = 0;
    void* external = nullptr;
    CUresult result = import(vkDevice, reinterpret_cast<uint64_t>(vkMemory), requirements.size, &device, &external);
    std::printf("d4rImportVulkanMemory -> %d, device 0x%llx\n", result, device);
    if (result != 0)
        return 1;

    // A game-like R16G16B16A16_FLOAT texture, filled through an upload buffer.
    D3D12_HEAP_PROPERTIES defaultHeap = {};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* texture = nullptr;
    ID3D12Resource* upload = create_buffer(D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* readback = create_buffer(D3D12_HEAP_TYPE_READBACK, bytes, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!check(g_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &textureDesc,
                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource),
                                                 reinterpret_cast<void**>(&texture)),
               "CreateCommittedResource(texture)") ||
        upload == nullptr || readback == nullptr)
        return 1;
    UINT64 vkImageHandle = 0, unusedOffset = 0;
    interop->GetVulkanResourceInfo(texture, &vkImageHandle, &unusedOffset);
    const VkImage vkImage = reinterpret_cast<VkImage>(vkImageHandle);

    uint64_t* mapped = nullptr;
    D3D12_RANGE none = {0, 0};
    if (!check(upload->Map(0, &none, reinterpret_cast<void**>(&mapped)), "Map upload"))
        return 1;
    for (UINT64 i = 0; i < words; ++i)
        mapped[i] = pattern(i) & 0x3BFF3BFF3BFF3BFFull; // finite halves: copies must not canonicalize NaNs
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    footprint.Footprint.Format = textureDesc.Format;
    footprint.Footprint.Width = width;
    footprint.Footprint.Height = height;
    footprint.Footprint.Depth = 1;
    footprint.Footprint.RowPitch = static_cast<UINT>(pitch);
    D3D12_TEXTURE_COPY_LOCATION textureLocation = {};
    textureLocation.pResource = texture;
    textureLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION uploadLocation = {};
    uploadLocation.pResource = upload;
    uploadLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    uploadLocation.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION readbackLocation = uploadLocation;
    readbackLocation.pResource = readback;
    g_list->CopyTextureRegion(&textureLocation, 0, 0, 0, &uploadLocation, nullptr);
    transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    if (!flush())
        return 1;

    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width, height, 1};
    VkMemoryBarrier toTransfer = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toTransfer.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    VkMemoryBarrier fromTransfer = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    fromTransfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fromTransfer.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    // Texture -> exportable buffer, recorded inside the D3D12 command list.
    auto recordTextureToBuffer = [&]() {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (!check(interop->BeginVkCommandBufferInterop(g_list, &cmd), "BeginVkCommandBufferInterop"))
            return false;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        interop->GetVulkanImageLayout(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, &layout);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toTransfer,
                             0, nullptr, 0, nullptr);
        vkCmdCopyImageToBuffer(cmd, vkImage, layout, vkBuffer, 1, &region);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                             &fromTransfer, 0, nullptr, 0, nullptr);
        return check(interop->EndVkCommandBufferInterop(g_list), "EndVkCommandBufferInterop");
    };
    if (!recordTextureToBuffer() || !flush())
        return 1;
    std::vector<uint64_t> host(words);
    result = dtoh(host.data(), device, bytes);
    size_t mismatches = 0;
    for (UINT64 i = 0; i < words; ++i)
        mismatches += host[i] != (pattern(i) & 0x3BFF3BFF3BFF3BFFull);
    std::printf("D3D12 texture -> Vulkan buffer -> CUDA: result %d, %zu of %llu texels mismatched\n", result,
                mismatches, static_cast<unsigned long long>(words));

    // CUDA write -> buffer -> D3D12 texture (the DLSS output direction).
    result = memset32(device, 0x3C003800u, bytes / 4);
    if (result == 0)
        result = synchronize();
    transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto recordBufferToTexture = [&]() {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (!check(interop->BeginVkCommandBufferInterop(g_list, &cmd), "BeginVkCommandBufferInterop"))
            return false;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        interop->GetVulkanImageLayout(texture, D3D12_RESOURCE_STATE_COPY_DEST, &layout);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &toTransfer,
                             0, nullptr, 0, nullptr);
        vkCmdCopyBufferToImage(cmd, vkBuffer, vkImage, layout, 1, &region);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                             &fromTransfer, 0, nullptr, 0, nullptr);
        return check(interop->EndVkCommandBufferInterop(g_list), "EndVkCommandBufferInterop");
    };
    if (!recordBufferToTexture())
        return 1;
    transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    g_list->CopyTextureRegion(&readbackLocation, 0, 0, 0, &textureLocation, nullptr);
    if (!flush())
        return 1;
    uint64_t* back = nullptr;
    D3D12_RANGE all = {0, static_cast<SIZE_T>(bytes)};
    readback->Map(0, &all, reinterpret_cast<void**>(&back));
    size_t reverse = 0;
    for (UINT64 i = 0; i < words; ++i)
        reverse += back[i] != 0x3C0038003C003800ull;
    readback->Unmap(0, nullptr);
    std::printf("CUDA -> Vulkan buffer -> D3D12 texture: result %d, %zu of %llu texels mismatched\n", result, reverse,
                static_cast<unsigned long long>(words));

    // Timings: interop copies vs today's texture -> readback copy, and a CUDA
    // device copy out of the imported buffer.
    double toBuffer = 1e9, toTexture = 1e9, toReadback = 1e9, cudaCopy = 1e9;
    for (int i = 0; i < 10; ++i)
    {
        auto start = std::chrono::steady_clock::now();
        recordTextureToBuffer();
        flush();
        toBuffer = std::min(toBuffer, elapsed_ms(start));

        transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        flush();
        start = std::chrono::steady_clock::now();
        recordBufferToTexture();
        flush();
        toTexture = std::min(toTexture, elapsed_ms(start));
        transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        flush();

        start = std::chrono::steady_clock::now();
        g_list->CopyTextureRegion(&readbackLocation, 0, 0, 0, &textureLocation, nullptr);
        flush();
        toReadback = std::min(toReadback, elapsed_ms(start));
    }
    CUdeviceptr local = 0;
    if (alloc(&local, bytes) == 0)
        for (int i = 0; i < 10; ++i)
        {
            synchronize();
            const auto start = std::chrono::steady_clock::now();
            dtod(local, device, bytes);
            synchronize();
            cudaCopy = std::min(cudaCopy, elapsed_ms(start));
        }
    std::printf("%ux%u RGBA16F (%.1f MB), best of 10 incl. submit+wait:\n", width, height, bytes / 1048576.0);
    std::printf("  interop texture -> VRAM buffer:  %.3f ms\n", toBuffer);
    std::printf("  interop VRAM buffer -> texture:  %.3f ms\n", toTexture);
    std::printf("  D3D12 texture -> readback (today): %.3f ms\n", toReadback);
    std::printf("  CUDA device copy from imported:  %.3f ms\n", cudaCopy);
    const bool ok = mismatches == 0 && reverse == 0;
    std::printf("zero-copy interop probe: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
