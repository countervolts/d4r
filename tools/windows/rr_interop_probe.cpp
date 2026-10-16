// D3D12 -> Vulkan -> D3D12 shared-buffer/fence gate for native RR integration.
// Synthetic transfers and optional local RR model replay. No NGX, CUDA, or game.
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <volk.h>
#include <vulkan/vulkan_win32.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "rr_bridge_replay.h"
#include "rr_native_packet.h"
using Microsoft::WRL::ComPtr;
namespace {
void dx(HRESULT value, const char* operation) {
    if (FAILED(value)) throw std::runtime_error(std::string(operation) + " HRESULT=" + std::to_string(value));
}
void vk(VkResult value, const char* operation) {
    if (value != VK_SUCCESS) throw std::runtime_error(std::string(operation) + " VkResult=" + std::to_string(value));
}
struct Handle { HANDLE value = nullptr; ~Handle() { if (value) CloseHandle(value); } };
ComPtr<ID3D12Resource> buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE type,
                             D3D12_HEAP_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = type;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes; desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (type == D3D12_HEAP_TYPE_DEFAULT) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> result;
    dx(device->CreateCommittedResource(&heap, flags, &desc, state, nullptr, IID_PPV_ARGS(&result)), "Create buffer");
    return result;
}
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &b);
}
struct Vulkan {
    std::unique_ptr<rr_bridge::NativeDevice> native;
    std::unique_ptr<rr_bridge::NativePacket> packet;
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer shared = VK_NULL_HANDLE, readback = VK_NULL_HANDLE;
    VkDeviceMemory host = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    std::unique_ptr<rr_bridge::Replay> replay;
    ~Vulkan() {
        // The outer process timeout also bounds a stalled driver call. Never
        // destroy resources that a failed/timed-out submission might still use.
        if (submitted) { (void)replay.release(); (void)packet.release(); (void)native.release(); return; }
        replay.reset();
        if (device) {
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);


            if (readback) vkDestroyBuffer(device, readback, nullptr);
            if (image) vkDestroyImage(device, image, nullptr);

            if (host) vkFreeMemory(device, host, nullptr);
            if (imageMemory) vkFreeMemory(device, imageMemory, nullptr);
        }
    }
};
}
int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    UINT64 bytes = 1u << 20;
    unsigned iterations = 8;
    uint32_t imageWidth = 0, imageHeight = 0;
    std::string rrModel, rrInputs, rrOutput, rrReference;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("Missing argument value");
        const std::string key = argv[i];
        size_t end = 0;
        const std::string text = argv[i + 1];
        if (key == "--rr-model") { rrModel = text; continue; }
        if (key == "--rr-inputs") { rrInputs = text; continue; }
        if (key == "--rr-output") { rrOutput = text; continue; }
        if (key == "--rr-reference") { rrReference = text; continue; }
        if (text.empty() || text[0] == '-') throw std::runtime_error("Expected positive argument");
        const auto value = std::stoull(text, &end);
        if (end != text.size()) throw std::runtime_error("Invalid numeric argument");
        if (key == "--bytes") bytes = value;
        else if (key == "--iterations" && value <= 32) iterations = unsigned(value);
        else if (key == "--image-width" && value <= 8192) imageWidth = uint32_t(value);
        else if (key == "--image-height" && value <= 8192) imageHeight = uint32_t(value);
        else throw std::runtime_error("Unknown or out-of-range argument");
    }
    std::unique_ptr<rr_bridge::Layout> rrLayout;
    if (!rrModel.empty()) {
        if (rrInputs.empty() || rrOutput.empty() || imageWidth || imageHeight) throw std::runtime_error("RR replay needs model/inputs/output and no image-clear mode");
        rrLayout = std::make_unique<rr_bridge::Layout>(rrInputs, rrOutput, rrReference);
        bytes = rrLayout->bytes; iterations = unsigned(rrLayout->frames.size());
    } else if (!rrInputs.empty() || !rrOutput.empty() || !rrReference.empty()) throw std::runtime_error("Missing RR model");
    if (!bytes || bytes % 4 || bytes > (64u << 20) || !iterations)
        throw std::runtime_error("Need a 4-byte-aligned buffer (4 bytes..64MiB) and 1..32 iterations");
    const UINT64 rowPitch = (UINT64(imageWidth) * 8 + 255) & ~UINT64(255);
    const bool imageMode = imageWidth && imageHeight;
    if (bool(imageWidth) != bool(imageHeight) || (imageMode && bytes != rowPitch * imageHeight))
        throw std::runtime_error("Image dimensions must match the 256-byte-padded RGBA16F buffer size");
    ComPtr<IDXGIFactory4> factory; dx(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 description{};
    bool selected = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        dx(adapter->GetDesc1(&description), "DXGI adapter description");
        if (description.VendorId == 0x1002 && !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { selected = true; break; }
        adapter.Reset();
    }
    if (!selected) throw std::runtime_error("No AMD D3D12 adapter");
    ComPtr<ID3D12Device> d3d; dx(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d)), "D3D12 device");
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC q{}; q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    dx(d3d->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)), "D3D12 queue");
    auto shared = buffer(d3d.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_STATE_COMMON);
    auto upload = buffer(d3d.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = buffer(d3d.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    std::unique_ptr<rr_bridge::D3D12Frames> d3dFrames;
    if (rrLayout) d3dFrames = std::make_unique<rr_bridge::D3D12Frames>(*rrLayout, d3d.Get());
    ComPtr<ID3D12Fence> d3dFence; dx(d3d->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&d3dFence)), "Shared D3D12 fence");
    Handle memoryHandle, fenceHandle, event;
    dx(d3d->CreateSharedHandle(shared.Get(), nullptr, GENERIC_ALL, nullptr, &memoryHandle.value), "Shared buffer handle");
    dx(d3d->CreateSharedHandle(d3dFence.Get(), nullptr, GENERIC_ALL, nullptr, &fenceHandle.value), "Shared fence handle");
    event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event.value) throw std::runtime_error("Fence event creation failed");
    Vulkan v;
    v.native = std::make_unique<rr_bridge::NativeDevice>();
    v.native->initialize(description.AdapterLuid, bool(rrLayout));
    v.device = v.native->device;
    const auto physical = v.native->physical;
    const auto family = v.native->family;
    const auto vkQueue = v.native->queue;
    if (rrLayout) v.replay = std::make_unique<rr_bridge::Replay>(*rrLayout, physical, v.device, rrModel);
    v.packet = std::make_unique<rr_bridge::NativePacket>(*v.native);
    v.packet->initialize(bytes, memoryHandle.value, fenceHandle.value);
    v.shared = v.packet->buffer; v.semaphore = v.packet->semaphore;
    VkPhysicalDeviceMemoryProperties memory{}; vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    auto memoryType = [&](uint32_t bits, VkMemoryPropertyFlags flags) {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
        throw std::runtime_error("No compatible Vulkan memory type");
    };
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryRequirements req{};
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    vk(vkCreateBuffer(v.device, &bi, nullptr, &v.readback), "Vulkan readback buffer");
    vkGetBufferMemoryRequirements(v.device, v.readback, &req);
    ai.pNext = nullptr; ai.allocationSize = req.size;
    ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vk(vkAllocateMemory(v.device, &ai, nullptr, &v.host), "Vulkan readback memory");
    vk(vkBindBufferMemory(v.device, v.readback, v.host, 0), "Bind readback");
    if (imageMode) {
        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.imageType = VK_IMAGE_TYPE_2D; image.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        image.extent = {imageWidth, imageHeight, 1}; image.mipLevels = image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        vk(vkCreateImage(v.device, &image, nullptr, &v.image), "Canonical RR image");
        vkGetImageMemoryRequirements(v.device, v.image, &req);
        ai.allocationSize = req.size; ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vk(vkAllocateMemory(v.device, &ai, nullptr, &v.imageMemory), "Canonical RR image memory");
        vk(vkBindImageMemory(v.device, v.image, v.imageMemory, 0), "Bind canonical image");
    }
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = family;
    vk(vkCreateCommandPool(v.device, &pci, nullptr, &v.pool), "Vulkan command pool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = v.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb; vk(vkAllocateCommandBuffers(v.device, &cai, &cb), "Vulkan command buffer");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; vk(vkCreateFence(v.device, &fci, nullptr, &v.fence), "Vulkan completion fence");
    std::puts("RR_INTEROP_IMPORTED memory=dedicated_d3d12_resource semaphore=d3d12_timeline_fence luid_match=1");
    ComPtr<ID3D12CommandAllocator> producerAllocator, consumerAllocator;
    ComPtr<ID3D12GraphicsCommandList> producer, consumer;
    dx(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&producerAllocator)), "Producer allocator");
    dx(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&consumerAllocator)), "Consumer allocator");
    dx(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, producerAllocator.Get(), nullptr, IID_PPV_ARGS(&producer)), "Producer list");
    dx(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, consumerAllocator.Get(), nullptr, IID_PPV_ARGS(&consumer)), "Consumer list");
    dx(producer->Close(), "Initial producer close"); dx(consumer->Close(), "Initial consumer close");
    auto pattern = [](unsigned iteration, uint32_t word) { return (word * 2654435761u) ^ (0x13579bdfu + iteration * 97u); };
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        uint32_t* input = nullptr; D3D12_RANGE noRead{};
        dx(upload->Map(0, &noRead, reinterpret_cast<void**>(&input)), "Upload map");
        if (rrLayout) rrLayout->upload(input, iteration);
        else for (uint32_t word = 0; word < bytes / 4; ++word) input[word] = pattern(iteration, word);
        upload->Unmap(0, nullptr);
        dx(producerAllocator->Reset(), "Producer allocator reset"); dx(producer->Reset(producerAllocator.Get(), nullptr), "Producer reset");
        transition(producer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        if (d3dFrames) d3dFrames->recordInputs(producer.Get(), upload.Get(), shared.Get());
        else producer->CopyBufferRegion(shared.Get(), 0, upload.Get(), 0, bytes);
        transition(producer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        dx(producer->Close(), "Producer close"); ID3D12CommandList* p[] = {producer.Get()}; queue->ExecuteCommandLists(1, p);
        const uint64_t ready = iteration * 3 + 1, done = ready + 1, complete = ready + 2;
        dx(queue->Signal(d3dFence.Get(), ready), "D3D12 producer signal");
        vk(vkResetCommandPool(v.device, v.pool, 0), "Vulkan pool reset");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vk(vkBeginCommandBuffer(cb, &begin), "Vulkan begin");
        VkBufferMemoryBarrier acquire{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; acquire.buffer = v.shared; acquire.size = VK_WHOLE_SIZE;
        acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL; acquire.dstQueueFamilyIndex = family; acquire.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        if (rrLayout) acquire.dstAccessMask |= VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &acquire, 0, nullptr);
        VkBufferImageCopy imageRegion{};
        imageRegion.bufferRowLength = uint32_t(rowPitch / 8);
        imageRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        imageRegion.imageExtent = {imageWidth, imageHeight, 1};
        auto imageBarrier = [&](VkImageLayout from, VkImageLayout to, VkAccessFlags source, VkAccessFlags destination) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.image = v.image; b.oldLayout = from; b.newLayout = to;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; b.srcAccessMask = source; b.dstAccessMask = destination;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        };
        if (rrLayout) {
            v.replay->record(cb, v.shared, iteration);
        } else if (imageMode) {
            vkCmdFillBuffer(cb, v.readback, 0, bytes, 0xdeadbeefu);
            VkBufferMemoryBarrier fill{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; fill.buffer = v.readback; fill.size = VK_WHOLE_SIZE;
            fill.srcQueueFamilyIndex = fill.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            fill.srcAccessMask = fill.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &fill, 0, nullptr);
            imageBarrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            vkCmdCopyBufferToImage(cb, v.shared, v.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageRegion);
            imageBarrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            vkCmdCopyImageToBuffer(cb, v.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, v.readback, 1, &imageRegion);
            imageBarrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkClearColorValue clear{}; clear.float32[0] = iteration % 2 ? .5f : .25f;
            clear.float32[1] = .5f; clear.float32[2] = .75f; clear.float32[3] = 1.f;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, v.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
            imageBarrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        } else {
            VkBufferCopy region{0, 0, bytes}; vkCmdCopyBuffer(cb, v.shared, v.readback, 1, &region);
        }
        VkBufferMemoryBarrier afterRead{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; afterRead.buffer = v.shared; afterRead.size = VK_WHOLE_SIZE;
        afterRead.srcQueueFamilyIndex = afterRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        afterRead.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; afterRead.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        if (!rrLayout) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &afterRead, 0, nullptr);
        const uint32_t output = 0xa5a50000u | iteration;
        if (!rrLayout) vkCmdFillBuffer(cb, v.shared, 0, bytes, output);
        if (imageMode) {
            afterRead.srcAccessMask = afterRead.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &afterRead, 0, nullptr);
            vkCmdCopyImageToBuffer(cb, v.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, v.shared, 1, &imageRegion);
        }
        VkBufferMemoryBarrier release = acquire; release.srcQueueFamilyIndex = family; release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        release.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; release.dstAccessMask = 0;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1, &release, 0, nullptr);
        VkBufferMemoryBarrier hostRead{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; hostRead.buffer = v.readback; hostRead.size = VK_WHOLE_SIZE;
        hostRead.srcQueueFamilyIndex = hostRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; hostRead.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hostRead, 0, nullptr);
        vk(vkEndCommandBuffer(cb), "Vulkan end");
        VkTimelineSemaphoreSubmitInfo values{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        values.waitSemaphoreValueCount = values.signalSemaphoreValueCount = 1; values.pWaitSemaphoreValues = &ready; values.pSignalSemaphoreValues = &done;
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, &values}; submit.waitSemaphoreCount = submit.signalSemaphoreCount = 1;
        submit.pWaitSemaphores = submit.pSignalSemaphores = &v.semaphore; submit.pWaitDstStageMask = &waitStage; submit.commandBufferCount = 1; submit.pCommandBuffers = &cb;
        vk(vkResetFences(v.device, 1, &v.fence), "Vulkan fence reset");
        vk(vkQueueSubmit(vkQueue, 1, &submit, v.fence), "Vulkan shared-buffer submission"); v.submitted = true;
        dx(queue->Wait(d3dFence.Get(), done), "D3D12 wait for Vulkan");
        dx(consumerAllocator->Reset(), "Consumer allocator reset"); dx(consumer->Reset(consumerAllocator.Get(), nullptr), "Consumer reset");
        transition(consumer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        if (d3dFrames) d3dFrames->recordOutput(consumer.Get(), shared.Get(), readback.Get());
        else consumer->CopyBufferRegion(readback.Get(), 0, shared.Get(), 0, bytes);
        transition(consumer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        dx(consumer->Close(), "Consumer close"); ID3D12CommandList* c[] = {consumer.Get()}; queue->ExecuteCommandLists(1, c);
        dx(queue->Signal(d3dFence.Get(), complete), "D3D12 consumer signal");
        dx(d3dFence->SetEventOnCompletion(complete, event.value), "Completion event");
        if (WaitForSingleObject(event.value, 5000) != WAIT_OBJECT_0) throw std::runtime_error("D3D12/Vulkan handoff timeout");
        vk(vkWaitForFences(v.device, 1, &v.fence, VK_TRUE, 5000000000ull), "Vulkan completion wait"); v.submitted = false;
        if (rrLayout) {
            void* data = nullptr; D3D12_RANGE range{0, SIZE_T(bytes)};
            dx(readback->Map(0, &range, &data), "RR D3D12 output map");
            try { rrLayout->validate(data, iteration); } catch (...) { D3D12_RANGE empty{}; readback->Unmap(0, &empty); throw; }
            D3D12_RANGE empty{}; readback->Unmap(0, &empty);
            continue;
        }
        uint32_t* first = nullptr; vk(vkMapMemory(v.device, v.host, 0, bytes, 0, reinterpret_cast<void**>(&first)), "Vulkan readback map");
        auto padding = [&](uint32_t word) { return imageMode && word % (rowPitch / 4) >= imageWidth * 2; };
        size_t mismatches = 0;
        for (uint32_t word = 0; word < bytes / 4; ++word)
            mismatches += first[word] != (padding(word) ? 0xdeadbeefu : pattern(iteration, word));
        vkUnmapMemory(v.device, v.host);
        uint32_t* second = nullptr; D3D12_RANGE readRange{0, SIZE_T(bytes)};
        dx(readback->Map(0, &readRange, reinterpret_cast<void**>(&second)), "D3D12 readback map");
        for (uint32_t word = 0; word < bytes / 4; ++word) {
            const uint32_t imageWord = word % 2 ? 0x3c003a00u : (iteration % 2 ? 0x38003800u : 0x38003400u);
            const uint32_t expected = imageMode && !padding(word) ? imageWord : output;
            mismatches += second[word] != expected;
        }
        D3D12_RANGE noWrite{}; readback->Unmap(0, &noWrite);
        if (mismatches) throw std::runtime_error("Shared-buffer contents differ: " + std::to_string(mismatches));
        std::printf("RR_INTEROP_ITERATION index=%u bytes=%llu exact=1 producer_value=%llu vulkan_value=%llu consumer_value=%llu\n",
            iteration, static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(ready), static_cast<unsigned long long>(done), static_cast<unsigned long long>(complete));
    }
    if (rrLayout) std::printf("PASS RR_BRIDGED_INFERENCE frames=%u guides=7 d3d12_textures=1 gpu_fence_handoff=1 standalone_reference=%u\n", iterations, !rrReference.empty());
    else std::printf("PASS RR_INTEROP_TRANSFER iterations=%u d3d12_to_vulkan_exact=1 vulkan_to_d3d12_exact=1 gpu_fence_handoff=1 canonical_image=%u inference_executed=0\n", iterations, imageMode);
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "FAIL RR_INTEROP_TRANSFER %s\n", error.what()); return 1; }
