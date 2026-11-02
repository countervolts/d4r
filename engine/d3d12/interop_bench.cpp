// Milestone 1 interop measurement: the cost of running the Vulkan engine for a D3D12 game through shared memory.
// Per frame, on the game's D3D12 queue: copy color/depth/motion into shared buffers and signal a shared fence;
// a Vulkan queue waits on it (imported as a timeline semaphore), runs its work and signals; the D3D12 queue waits
// and copies the shared output buffer into the game's output texture. The Vulkan work is a small copy that proves
// the data path; the engine's inference time is measured separately (d4r-k) and adds to these numbers.
// The baseline is the same copies on the D3D12 queue without the Vulkan round trip.
//   d4r-interop-bench [RENDER_W RENDER_H OUTPUT_W OUTPUT_H] [--frames N]
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace {
void check(HRESULT hr, const char* operation)
{
    if (FAILED(hr)) {
        char error[256];
        std::snprintf(error, sizeof(error), "%s: HRESULT 0x%08lx", operation, static_cast<unsigned long>(hr));
        throw std::runtime_error(error);
    }
}
void checkVk(VkResult r, const char* operation)
{
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(r));
}
#define CK(call) check((call), #call)
#define VK(call) checkVk((call), #call)
using Clock = std::chrono::steady_clock;
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct Input { const char* name; DXGI_FORMAT format; UINT bytes; bool output; };
constexpr Input Surfaces[] = {
    {"color", DXGI_FORMAT_R16G16B16A16_FLOAT, 8, false},
    {"depth", DXGI_FORMAT_R32_FLOAT, 4, false},
    {"motion", DXGI_FORMAT_R16G16_FLOAT, 4, false},
    {"output", DXGI_FORMAT_R16G16B16A16_FLOAT, 8, true},
};
constexpr size_t SurfaceCount = std::size(Surfaces);
constexpr VkDeviceSize ProbeBytes = 65536;      // what the Vulkan side copies from color to output

struct Bench {
    UINT renderW, renderH, outputW, outputH;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator, scratch;     // static frame lists; one-off verification lists
    ComPtr<ID3D12GraphicsCommandList> pre, post;
    ComPtr<ID3D12Fence> shared;
    ComPtr<ID3D12QueryHeap> queries;
    ComPtr<ID3D12Resource> timing;
    ComPtr<ID3D12Resource> textures[SurfaceCount], buffers[SurfaceCount];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[SurfaceCount]{};
    UINT64 sizes[SurfaceCount]{}, value = 0, frequency = 0;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    LUID luid{};

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice vk = VK_NULL_HANDLE;
    VkQueue vkQueue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    VkBuffer vkBuffers[SurfaceCount]{};
    VkDeviceMemory vkMemory[SurfaceCount]{};

    ~Bench()
    {
        if (vk) {
            vkDeviceWaitIdle(vk);
            for (size_t i = 0; i < SurfaceCount; ++i) { vkDestroyBuffer(vk, vkBuffers[i], nullptr); vkFreeMemory(vk, vkMemory[i], nullptr); }
            vkDestroySemaphore(vk, semaphore, nullptr);
            vkDestroyCommandPool(vk, pool, nullptr);
            vkDestroyDevice(vk, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
        if (event) CloseHandle(event);
    }

    void createD3D12()
    {
        ComPtr<IDXGIFactory4> factory; CK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        for (UINT i = 0; !device; ++i) {
            ComPtr<IDXGIAdapter1> adapter;
            const HRESULT result = factory->EnumAdapters1(i, &adapter);
            if (result == DXGI_ERROR_NOT_FOUND) throw std::runtime_error("no hardware D3D12 adapter");
            CK(result);
            DXGI_ADAPTER_DESC1 desc{}; CK(adapter->GetDesc1(&desc));
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
                luid = desc.AdapterLuid;
                std::printf("D3D12 adapter: %ls\n", desc.Description);
            }
        }
        D3D12_COMMAND_QUEUE_DESC q{}; q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CK(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
        CK(queue->GetTimestampFrequency(&frequency));
        CK(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&allocator)));
        CK(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&scratch)));
        CK(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&shared)));
        D3D12_QUERY_HEAP_DESC qh{}; qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qh.Count = 4;
        CK(device->CreateQueryHeap(&qh, IID_PPV_ARGS(&queries)));
        D3D12_HEAP_PROPERTIES readback{}; readback.Type = D3D12_HEAP_TYPE_READBACK;
        timing = buffer(32, readback, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        D3D12_HEAP_PROPERTIES local{}; local.Type = D3D12_HEAP_TYPE_DEFAULT;
        for (size_t i = 0; i < SurfaceCount; ++i) {
            const auto& s = Surfaces[i];
            D3D12_RESOURCE_DESC t{};
            t.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            t.Width = s.output ? outputW : renderW; t.Height = s.output ? outputH : renderH;
            t.DepthOrArraySize = t.MipLevels = 1; t.Format = s.format; t.SampleDesc.Count = 1;
            // The game's own textures: ordinary, not shareable.
            CK(device->CreateCommittedResource(&local, D3D12_HEAP_FLAG_NONE, &t,
                                               s.output ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE,
                                               nullptr, IID_PPV_ARGS(&textures[i])));
            device->GetCopyableFootprints(&t, 0, 1, 0, &layouts[i], nullptr, nullptr, &sizes[i]);
            buffers[i] = buffer(sizes[i], local, D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_STATE_COMMON);   // copies promote it
        }
        // Static lists: copies in (pre) and out (post), each bracketed by timestamps.
        CK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&pre)));
        pre->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        for (size_t i = 0; i < SurfaceCount; ++i) if (!Surfaces[i].output) copy(pre.Get(), i);
        pre->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        CK(pre->Close());
        CK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&post)));
        post->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
        copy(post.Get(), SurfaceCount - 1);
        post->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
        post->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, timing.Get(), 0);
        CK(post->Close());
    }
    ComPtr<ID3D12Resource> buffer(UINT64 bytes, const D3D12_HEAP_PROPERTIES& heap, D3D12_HEAP_FLAGS flags, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC b{};
        b.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; b.Width = bytes; b.Height = b.DepthOrArraySize = b.MipLevels = 1;
        b.SampleDesc.Count = 1; b.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> result;
        CK(device->CreateCommittedResource(&heap, flags, &b, state, nullptr, IID_PPV_ARGS(&result)));
        return result;
    }
    void copy(ID3D12GraphicsCommandList* list, size_t i)
    {
        D3D12_TEXTURE_COPY_LOCATION texture{textures[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}}, linear{};
        linear.pResource = buffers[i].Get(); linear.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; linear.PlacedFootprint = layouts[i];
        if (Surfaces[i].output) list->CopyTextureRegion(&texture, 0, 0, 0, &linear, nullptr);
        else list->CopyTextureRegion(&linear, 0, 0, 0, &texture, nullptr);
    }

    void createVulkan()
    {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
        VK(vkCreateInstance(&ici, nullptr, &instance));
        uint32_t count = 0; VK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
        std::vector<VkPhysicalDevice> devices(count); VK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
        for (auto pd : devices) {
            VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p.pNext = &id;
            vkGetPhysicalDeviceProperties2(pd, &p);
            if (id.deviceLUIDValid && std::memcmp(id.deviceLUID, &luid, VK_LUID_SIZE) == 0) {
                physical = pd;
                std::printf("Vulkan device: %s (same LUID)\n", p.properties.deviceName);
            }
        }
        if (!physical) throw std::runtime_error("no Vulkan device with the D3D12 adapter's LUID");
        uint32_t families = 0; vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
        std::vector<VkQueueFamilyProperties> qf(families); vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, qf.data());
        uint32_t family = UINT32_MAX;
        for (uint32_t i = 0; i < families && family == UINT32_MAX; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) family = i;
        const float priority = 1;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
        const char* extensions[] = {VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.timelineSemaphore = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext = &f12;
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 2; dci.ppEnabledExtensionNames = extensions;
        VK(vkCreateDevice(physical, &dci, nullptr, &vk));
        vkGetDeviceQueue(vk, family, 0, &vkQueue);
        auto getProperties = reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(vkGetDeviceProcAddr(vk, "vkGetMemoryWin32HandlePropertiesKHR"));
        auto importSemaphore = reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(vkGetDeviceProcAddr(vk, "vkImportSemaphoreWin32HandleKHR"));
        if (!getProperties || !importSemaphore) throw std::runtime_error("Win32 external memory/semaphore entry points missing");

        VkPhysicalDeviceMemoryProperties memory; vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        for (size_t i = 0; i < SurfaceCount; ++i) {
            HANDLE handle = nullptr;
            CK(device->CreateSharedHandle(buffers[i].Get(), nullptr, GENERIC_ALL, nullptr, &handle));
            VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.pNext = &ext; bci.size = sizes[i];
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VK(vkCreateBuffer(vk, &bci, nullptr, &vkBuffers[i]));
            VkMemoryRequirements req; vkGetBufferMemoryRequirements(vk, vkBuffers[i], &req);
            VkMemoryWin32HandlePropertiesKHR hp{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
            VK(getProperties(vk, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, handle, &hp));
            uint32_t type = UINT32_MAX;
            for (uint32_t t = 0; t < memory.memoryTypeCount && type == UINT32_MAX; ++t)
                if ((req.memoryTypeBits & hp.memoryTypeBits & (1u << t)) &&
                    (memory.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) type = t;
            if (type == UINT32_MAX) throw std::runtime_error("no device-local memory type for the shared buffer");
            VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO}; dedicated.buffer = vkBuffers[i];
            VkImportMemoryWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
            import.pNext = &dedicated; import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT; import.handle = handle;
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.pNext = &import;
            mai.allocationSize = req.size; mai.memoryTypeIndex = type;
            VK(vkAllocateMemory(vk, &mai, nullptr, &vkMemory[i]));
            VK(vkBindBufferMemory(vk, vkBuffers[i], vkMemory[i], 0));
            CloseHandle(handle);     // D3D12_RESOURCE imports do not take ownership of the handle
        }
        VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO}; timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; sci.pNext = &timeline;
        VK(vkCreateSemaphore(vk, &sci, nullptr, &semaphore));
        HANDLE fenceHandle = nullptr;
        CK(device->CreateSharedHandle(shared.Get(), nullptr, GENERIC_ALL, nullptr, &fenceHandle));
        VkImportSemaphoreWin32HandleInfoKHR isi{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
        isi.semaphore = semaphore; isi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT; isi.handle = fenceHandle;
        VK(importSemaphore(vk, &isi));
        CloseHandle(fenceHandle);

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = family;
        VK(vkCreateCommandPool(vk, &cpi, nullptr, &pool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        VK(vkAllocateCommandBuffers(vk, &cai, &command));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        VK(vkBeginCommandBuffer(command, &begin));
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        const VkBufferCopy region{layouts[0].Offset, layouts[SurfaceCount - 1].Offset, ProbeBytes};
        vkCmdCopyBuffer(command, vkBuffers[0], vkBuffers[SurfaceCount - 1], 1, &region);
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        VK(vkEndCommandBuffer(command));
    }

    // One frame. Returns the fence value that marks its completion on the D3D12 queue.
    UINT64 frame(bool interop)
    {
        ID3D12CommandList* first[] = {pre.Get()};
        queue->ExecuteCommandLists(1, first);
        if (interop) {
            const UINT64 ready = ++value, done = ++value;
            CK(queue->Signal(shared.Get(), ready));
            VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            ts.waitSemaphoreValueCount = 1; ts.pWaitSemaphoreValues = &ready;
            ts.signalSemaphoreValueCount = 1; ts.pSignalSemaphoreValues = &done;
            const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.pNext = &ts;
            si.waitSemaphoreCount = 1; si.pWaitSemaphores = &semaphore; si.pWaitDstStageMask = &stage;
            si.commandBufferCount = 1; si.pCommandBuffers = &command;
            si.signalSemaphoreCount = 1; si.pSignalSemaphores = &semaphore;
            VK(vkQueueSubmit(vkQueue, 1, &si, VK_NULL_HANDLE));
            CK(queue->Wait(shared.Get(), done));
        }
        ID3D12CommandList* second[] = {post.Get()};
        queue->ExecuteCommandLists(1, second);
        CK(queue->Signal(shared.Get(), ++value));
        return value;
    }
    void wait(UINT64 v)
    {
        if (shared->GetCompletedValue() >= v) return;
        CK(shared->SetEventOnCompletion(v, event));
        if (WaitForSingleObject(event, 10000) != WAIT_OBJECT_0) {
            CK(device->GetDeviceRemovedReason());
            throw std::runtime_error("GPU wait timed out");
        }
    }
    double copyMicroseconds()
    {
        UINT64 t[4]; void* mapped = nullptr; const D3D12_RANGE range{0, sizeof(t)}, none{0, 0};
        CK(timing->Map(0, &range, &mapped)); std::memcpy(t, mapped, sizeof(t)); timing->Unmap(0, &none);
        return double((t[1] - t[0]) + (t[3] - t[2])) * 1e6 / double(frequency);
    }

    // Fill color with a pattern, run one interop frame and check the Vulkan copy reached the output texture.
    void verify()
    {
        D3D12_HEAP_PROPERTIES upload{}; upload.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_HEAP_PROPERTIES readback{}; readback.Type = D3D12_HEAP_TYPE_READBACK;
        auto source = buffer(ProbeBytes, upload, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto result = buffer(ProbeBytes, readback, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        uint32_t* p = nullptr; const D3D12_RANGE none{0, 0};
        CK(source->Map(0, &none, reinterpret_cast<void**>(&p)));
        for (UINT i = 0; i < ProbeBytes / 4; ++i) p[i] = 0x9e3779b9u * (i + 1);
        source->Unmap(0, nullptr);
        ComPtr<ID3D12GraphicsCommandList> list;
        auto run = [&](auto&& record) {
            CK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, scratch.Get(), nullptr, IID_PPV_ARGS(&list)));
            record(); CK(list->Close());
            ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
            CK(queue->Signal(shared.Get(), ++value)); wait(value);
        };
        // Write the pattern into the first rows of the color texture through its staging buffer.
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT rows = layouts[0];
        rows.Offset = 0; rows.Footprint.Height = UINT(ProbeBytes / rows.Footprint.RowPitch);
        if (rows.Footprint.Height == 0) throw std::runtime_error("render width too large for the probe");
        auto transition = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
            D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b}; list->ResourceBarrier(1, &barrier);
        };
        run([&] {
            transition(textures[0].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION dst{textures[0].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}}, src{};
            src.pResource = source.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = rows;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            transition(textures[0].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        });
        wait(frame(true));
        // Output texture rows -> readback (row pitches match: both are RGBA16F at 256-byte-aligned widths).
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT out = layouts[SurfaceCount - 1];
        out.Offset = 0; out.Footprint.Height = UINT(ProbeBytes / out.Footprint.RowPitch);
        run([&] {
            transition(textures[SurfaceCount - 1].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION src{textures[SurfaceCount - 1].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}}, dst{};
            dst.pResource = result.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = out;
            D3D12_BOX box{0, 0, 0, out.Footprint.Width, out.Footprint.Height, 1};
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
            transition(textures[SurfaceCount - 1].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        });
        // Vulkan copied ProbeBytes linearly between the two pitched buffers, so byte b of the output rows is byte b
        // of the color rows wherever both are inside a texture row (not row padding).
        const D3D12_RANGE all{0, size_t(ProbeBytes)};
        uint8_t* got = nullptr; CK(result->Map(0, &all, reinterpret_cast<void**>(&got)));
        CK(source->Map(0, &all, reinterpret_cast<void**>(&p)));
        const auto* want = reinterpret_cast<const uint8_t*>(p);
        const UINT64 inPitch = layouts[0].Footprint.RowPitch, outPitch = out.Footprint.RowPitch;
        const UINT64 inWidth = UINT64(layouts[0].Footprint.Width) * 8, outWidth = UINT64(out.Footprint.Width) * 8;
        bool ok = true;
        for (UINT64 byte = 0; byte < UINT64(rows.Footprint.Height) * inPitch && ok; ++byte)
            if (byte % inPitch < inWidth && byte % outPitch < outWidth && byte / outPitch < out.Footprint.Height)
                ok = got[byte] == want[byte];
        source->Unmap(0, &none); result->Unmap(0, &none);
        if (!ok) throw std::runtime_error("interop data check failed: Vulkan output did not reach the D3D12 texture");
        std::printf("Data path: color texture -> shared buffer -> Vulkan copy -> shared buffer -> output texture: ok\n");
    }
};
}

int main(int argc, char** argv)
{
    try {
        UINT dims[4] = {2560, 1440, 3840, 2160}; int frames = 200, positional = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string a(argv[i]);
            if (a == "--frames" && i + 1 < argc) frames = std::max(20, std::stoi(argv[++i]));
            else if (positional < 4) dims[positional++] = UINT(std::stoul(a));
            else { std::fprintf(stderr, "usage: d4r-interop-bench [RENDER_W RENDER_H OUTPUT_W OUTPUT_H] [--frames N]\n"); return 2; }
        }
        if (positional != 0 && positional != 4) throw std::runtime_error("give all four dimensions or none");
        Bench b{dims[0], dims[1], dims[2], dims[3]};
        b.createD3D12();
        b.createVulkan();
        b.verify();
        std::printf("Render %ux%u (color RGBA16F, depth R32F, motion RG16F) -> output %ux%u RGBA16F, %d frames per mode\n",
                    dims[0], dims[1], dims[2], dims[3], frames);
        UINT64 bytes = 0; for (auto s : b.sizes) bytes += s;
        std::printf("Copied per frame: %.1f MiB\n", double(bytes) / (1 << 20));
        for (bool interop : {false, true, false, true}) b.wait(b.frame(interop));      // warm-up
        // Rotate modes frame by frame so clock changes affect both equally.
        std::vector<double> latency[2], copies[2];
        for (int n = 0; n < frames; ++n)
            for (int mode = 0; mode < 2; ++mode) {
                const auto t0 = Clock::now();
                b.wait(b.frame(mode == 1));
                latency[mode].push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
                copies[mode].push_back(b.copyMicroseconds());
            }
        // Throughput: three frames in flight, as a game's queue would run them.
        double throughput[2];
        for (int mode = 0; mode < 2; ++mode) {
            std::vector<UINT64> marks;
            const auto t0 = Clock::now();
            for (int n = 0; n < frames; ++n) {
                marks.push_back(b.frame(mode == 1));
                if (marks.size() > 3) b.wait(marks[marks.size() - 4]);
            }
            b.wait(marks.back());
            throughput[mode] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / frames;
        }
        std::printf("%-34s %12s %12s\n", "median per frame (us)", "D3D12 only", "interop");
        std::printf("%-34s %12.1f %12.1f\n", "input+output copies (GPU)", median(copies[0]), median(copies[1]));
        std::printf("%-34s %12.1f %12.1f\n", "frame latency, idle GPU (wall)", median(latency[0]), median(latency[1]));
        std::printf("%-34s %12.1f %12.1f\n", "frame time, 3 in flight (wall)", throughput[0], throughput[1]);
        std::printf("Interop round trip, latency: +%.1f us; throughput: +%.1f us per frame (beyond the copies)\n",
                    median(latency[1]) - median(latency[0]), throughput[1] - throughput[0]);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "d4r-interop-bench: %s\n", e.what()); return 1; }
}
