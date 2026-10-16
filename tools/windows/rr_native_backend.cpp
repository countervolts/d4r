#include <volk.h>
#include "rr_native_backend.h"
#include "rr_native_packet.h"
#include "rr_bridge_replay.h"
#include "d3d12_command_hooks.h"
#include <mutex>

namespace d4r::win::rr {
namespace {
// The engine uses Volk's global dispatch. Serialize device creation, recording,
// submission and destruction; reload the owning dispatch before each operation.
std::mutex dispatchMutex;
using Packet = rr_bridge::NativePacket;
using Device = rr_bridge::NativeDevice;
size_t resource_index(size_t plane) { return plane < 7 ? plane : plane + 2; }
rr_bridge::Plane input_format(DXGI_FORMAT format, size_t plane) {
    // Match Linux's direct sampled-image route. These formats preserve the
    // stored texel values; no CPU conversion or reinterpretation is involved.
    if (plane == Snapshot::Motion) {
        if (format == DXGI_FORMAT_R16G16_FLOAT) return {nullptr, VK_FORMAT_R16G16_SFLOAT, 4};
        if (format == DXGI_FORMAT_R32G32_FLOAT) return {nullptr, VK_FORMAT_R32G32_SFLOAT, 8};
    } else if (plane == Snapshot::Depth || plane == Snapshot::HitDistance || plane == Snapshot::Disocclusion || plane == Snapshot::Subsurface) {
        if (plane == Snapshot::Depth && (format == DXGI_FORMAT_R32_TYPELESS || format == DXGI_FORMAT_D32_FLOAT ||
            format == DXGI_FORMAT_R32G8X24_TYPELESS || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT || format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS))
            return {nullptr, VK_FORMAT_R32_SFLOAT, 4};
        if (format == DXGI_FORMAT_R32_FLOAT) return {nullptr, VK_FORMAT_R32_SFLOAT, 4};
        if (format == DXGI_FORMAT_R16_FLOAT) return {nullptr, VK_FORMAT_R16_SFLOAT, 2};
        if (plane != Snapshot::Depth && format == DXGI_FORMAT_R8_UNORM) return {nullptr, VK_FORMAT_R8_UNORM, 1};
    } else {
        if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) return {nullptr, VK_FORMAT_R16G16B16A16_SFLOAT, 8};
        if (format == DXGI_FORMAT_R32G32B32A32_FLOAT) return {nullptr, VK_FORMAT_R32G32B32A32_SFLOAT, 16};
        if (format == DXGI_FORMAT_R8G8B8A8_UNORM) return {nullptr, VK_FORMAT_R8G8B8A8_UNORM, 4};
        if (format == DXGI_FORMAT_R10G10B10A2_UNORM) return {nullptr, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4};
        if (plane == Snapshot::Roughness) {
            if (format == DXGI_FORMAT_R16_FLOAT) return {nullptr, VK_FORMAT_R16_SFLOAT, 2};
            if (format == DXGI_FORMAT_R32_FLOAT) return {nullptr, VK_FORMAT_R32_SFLOAT, 4};
            if (format == DXGI_FORMAT_R8_UNORM) return {nullptr, VK_FORMAT_R8_UNORM, 1};
        }
    }
    throw std::runtime_error("Unsupported native RR sampled input format");
}
DXGI_FORMAT copy_format(ID3D12Device* device, ID3D12Resource* texture, size_t plane) {
    const auto desc = resource_desc(texture);
    if (plane != Snapshot::Depth) return desc.Format;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT rows = 0; UINT64 rowBytes = 0, bytes = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &bytes);
    const auto format = input_format(desc.Format, plane);
    if (!bytes || bytes == UINT64_MAX || rows != desc.Height || rowBytes != desc.Width * format.bytes ||
        footprint.Footprint.Width != desc.Width || footprint.Footprint.Height != desc.Height)
        throw std::runtime_error("Unsupported RR depth-plane footprint");
    return footprint.Footprint.Format;
}
}
struct Backend::Impl {
    Creation creation;
    rr_bridge::Layout layout;
    ComPtr<ID3D12Device> d3d;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Resource> shared;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12CommandAllocator> producerAllocator, consumerAllocator;
    ComPtr<ID3D12GraphicsCommandList> producer, consumer;
    Handle event;
    std::unique_ptr<Device> native;
    std::unique_ptr<Packet> packet;
    std::unique_ptr<rr_bridge::Replay> replay;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence completion = VK_NULL_HANDLE;
    uint64_t sequence = 0;
    bool pending = false, failed = false;
    std::unique_ptr<Snapshot> retained;
    Impl(ID3D12Device* device, const Creation& c)
        : creation(c), layout(c.width, c.height, c.outWidth, c.outHeight), d3d(device) {
        // Reserve each guide's largest supported texel size once. Format
        // changes can then preserve the packet, model and temporal history.
        VkDeviceSize offset = 0;
        for (auto& plane : layout.planes) {
            plane.offset = (offset + 511) & ~VkDeviceSize(511);
            plane.pitch = (VkDeviceSize(layout.rw) * 16 + 255) & ~VkDeviceSize(255);
            offset = plane.offset + plane.pitch * layout.rh;
        }
        layout.outputOffset = (offset + 511) & ~VkDeviceSize(511);
        layout.bytes = layout.outputOffset + layout.outputPitch * layout.oh;
    }
    void dispatch() {
        if (native && native->instance) volkLoadInstance(native->instance);
        if (native && native->device) volkLoadDevice(native->device);
    }
    ~Impl() {
        dispatch();
        replay.reset();
        if (native && native->device) {
            if (completion) vkDestroyFence(native->device, completion, nullptr);
            if (pool) vkDestroyCommandPool(native->device, pool, nullptr);
        }
        packet.reset(); native.reset();
    }
    void initialize(const std::string& directory) {
        if (!d3d) throw std::runtime_error("Null native RR D3D12 device");
        // Match the Linux RR adapter's contract: HDR and render-resolution,
        // unjittered motion. D/E already perform their own exposure reduction.
        // NVIDIA/DLSS include/nvsdk_ngx_defs.h defines these public flag bits.
        if ((creation.flags & 3u) != 3u)
            throw std::runtime_error("Native RR requires HDR and render-resolution motion vectors");
        if (creation.flags & ~(3u | 8u | 64u))
            throw std::runtime_error("Native RR creation flags require an unimplemented conversion");
        if (creation.quality < 0 || creation.quality > 5) throw std::runtime_error("Unsupported RR quality mode");
        native = std::make_unique<Device>(); native->initialize(adapter_luid(d3d.Get()), true);
        replay = std::make_unique<rr_bridge::Replay>(layout, native->physical, native->device, directory);
        const unsigned indices[] = {3, 2, 1, 4, 5, 0};
        const auto requested = creation.presets[indices[creation.quality]];
        if (requested && requested != replay->model->preset()) throw std::runtime_error("Configured RR model differs from requested preset");
        shared = make_buffer(d3d.Get(), layout.bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_STATE_COMMON);
        dx(d3d->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)), "RR shared fence");
        Handle memoryHandle, fenceHandle;
        dx(d3d->CreateSharedHandle(shared.Get(), nullptr, GENERIC_ALL, nullptr, &memoryHandle.value), "RR packet handle");
        dx(d3d->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &fenceHandle.value), "RR fence handle");
        packet = std::make_unique<Packet>(*native); packet->initialize(layout.bytes, memoryHandle.value, fenceHandle.value);
        event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event.value) throw std::runtime_error("RR completion event creation failed");
        auto createList = [&](auto& allocator, auto& list) {
            dx(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "RR allocator");
            dx(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "RR command list");
            dx(list->Close(), "RR initial close");
        };
        createList(producerAllocator, producer); createList(consumerAllocator, consumer);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = native->family;
        Device::check(vkCreateCommandPool(native->device, &pci, nullptr, &pool), "RR command pool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        Device::check(vkAllocateCommandBuffers(native->device, &cai, &command), "RR command buffer");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        Device::check(vkCreateFence(native->device, &fci, nullptr, &completion), "RR Vulkan completion fence");
    }
    void validate(ID3D12CommandQueue* q, const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>& access) {
        if (failed) throw std::runtime_error("Native RR backend is poisoned after failed submission");
        if (!q) throw std::runtime_error("Null RR submission queue");
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
#if defined(__MINGW32__)
        q->GetDesc(&queueDesc);
#else
        queueDesc = q->GetDesc();
#endif
        if (queueDesc.Type != D3D12_COMMAND_LIST_TYPE_DIRECT) throw std::runtime_error("RR requires a direct queue");
        ComPtr<ID3D12Device> owner; dx(q->GetDevice(IID_PPV_ARGS(&owner)), "RR queue device");
        if (owner.Get() != d3d.Get()) throw std::runtime_error("RR queue belongs to another device");
        for (size_t i = 0; i < access.size(); ++i)
            if (frame.resources[i] && (access[i].pending_split || access[i].inherited)) throw std::runtime_error("Unresolved RR texture state");
        validate_frame(frame);
    }
    void validate_frame(const Snapshot& frame) {
        if (failed) throw std::runtime_error("Native RR backend is poisoned after failed submission");
        if (frame.renderWidth != layout.rw || frame.renderHeight != layout.rh)
            throw std::runtime_error("Dynamic RR render extent requires engine reconfiguration");
        if (frame.resources[Snapshot::Exposure]) throw std::runtime_error("RR exposure texture is not implemented");
        if (frame.exposureScale != 1 || frame.depthType || frame.normalsType || frame.roughnessType || frame.diffuseType || frame.specularType)
            throw std::runtime_error("RR guide encoding or exposure scale requires conversion");
        for (size_t i = 0; i < Snapshot::Count; ++i) {
            if (i == Snapshot::Exposure) continue;
            auto* resource = frame.resources[i].Get();
            if (!resource && i >= Snapshot::HitDistance) continue;
            if (!resource) throw std::runtime_error("Missing required native RR texture");
            ComPtr<ID3D12Device> resourceOwner; dx(resource->GetDevice(IID_PPV_ARGS(&resourceOwner)), "RR texture device");
            if (resourceOwner.Get() != d3d.Get()) throw std::runtime_error("RR texture belongs to another device");
            const auto desc = resource_desc(resource);
            if (i != Snapshot::Output) (void)input_format(desc.Format, i);
            if (i == Snapshot::Depth) {
                (void)copy_format(d3d.Get(), resource, i);
                // D3D12 requires whole-subresource copies for depth-stencil.
                // The copied subresource is depth plane 0, never stencil.
                if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) &&
                    (frame.origins[i][0] || frame.origins[i][1] || layout.rw != desc.Width || layout.rh != desc.Height))
                    throw std::runtime_error("RR depth-stencil requires a full depth-plane rectangle");
            }
            const uint32_t width = i == Snapshot::Output ? layout.ow : layout.rw;
            const uint32_t height = i == Snapshot::Output ? layout.oh : layout.rh;
            if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || (i == Snapshot::Output && desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT) || desc.SampleDesc.Count != 1 || desc.DepthOrArraySize != 1 ||
                frame.origins[i][0] > desc.Width || width > desc.Width - frame.origins[i][0] ||
                frame.origins[i][1] > desc.Height || height > desc.Height - frame.origins[i][1])
                throw std::runtime_error("Unsupported native RR texture format or subrect");
            if (i == Snapshot::Output && !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
                throw std::runtime_error("RR output lacks UAV capability");
        }
        for (size_t i = 0; i < Snapshot::Count; ++i)
            if (i != Snapshot::Output && frame.resources[i] == frame.resources[Snapshot::Output]) throw std::runtime_error("RR input/output alias is unsupported");
    }
    void evaluate(ID3D12CommandQueue* q, const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>& access) {
        validate(q, frame, access); dispatch();
        for (size_t i = 0; i < layout.planes.size(); ++i) {
            const auto resource = resource_index(i);
            layout.planes[i].enabled = frame.resources[resource] != nullptr;
            if (!layout.planes[i].enabled) continue;
            const auto format = input_format(resource_desc(frame.resources[resource].Get()).Format, resource);
            replay->configure(i, format.format);
            layout.planes[i].format = format.format; layout.planes[i].bytes = format.bytes;
        }
        // Recording mutates engine upload/history bookkeeping. Any later error
        // requires recreation, even if no GPU submission has started yet.
        failed = true;
        commands::InternalScope internal;
        dx(producerAllocator->Reset(), "RR producer allocator reset"); dx(producer->Reset(producerAllocator.Get(), nullptr), "RR producer reset");
        transition(producer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        for (size_t i = 0; i < layout.planes.size(); ++i) {
            const auto resource = resource_index(i);
            auto* texture = frame.resources[resource].Get(); const auto& plane = layout.planes[i];
            if (!plane.enabled) continue;
            texture_copy_barrier(producer.Get(), texture, access[resource], false, false);
            auto source = rr_bridge::D3D12Frames::texture(texture);
            auto dest = rr_bridge::D3D12Frames::placed(shared.Get(), plane.offset, plane.pitch,
                copy_format(d3d.Get(), texture, resource), layout.rw, layout.rh);
            const auto x = frame.origins[resource][0], y = frame.origins[resource][1];
            D3D12_BOX box{x, y, 0, x + layout.rw, y + layout.rh, 1};
            const bool depthStencil = (resource_desc(texture).Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
            producer->CopyTextureRegion(&dest, 0, 0, 0, &source, depthStencil ? nullptr : &box);
            texture_copy_barrier(producer.Get(), texture, access[resource], false, true);
        }
        transition(producer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        dx(producer->Close(), "RR producer close");
        dx(consumerAllocator->Reset(), "RR consumer allocator reset"); dx(consumer->Reset(consumerAllocator.Get(), nullptr), "RR consumer reset");
        transition(consumer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto* output = frame.resources[Snapshot::Output].Get();
        texture_copy_barrier(consumer.Get(), output, access[Snapshot::Output], true, false);
        auto source = rr_bridge::D3D12Frames::placed(shared.Get(), layout.outputOffset, layout.outputPitch,
            DXGI_FORMAT_R16G16B16A16_FLOAT, layout.ow, layout.oh);
        auto dest = rr_bridge::D3D12Frames::texture(output);
        consumer->CopyTextureRegion(&dest, frame.origins[Snapshot::Output][0], frame.origins[Snapshot::Output][1], 0, &source, nullptr);
        texture_copy_barrier(consumer.Get(), output, access[Snapshot::Output], true, true);
        transition(consumer.Get(), shared.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        dx(consumer->Close(), "RR consumer close");
        Device::check(vkResetCommandPool(native->device, pool, 0), "RR Vulkan pool reset");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        Device::check(vkBeginCommandBuffer(command, &begin), "RR Vulkan begin");
        VkBufferMemoryBarrier acquire{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; acquire.buffer = packet->buffer; acquire.size = VK_WHOLE_SIZE;
        acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL; acquire.dstQueueFamilyIndex = native->family;
        acquire.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &acquire, 0, nullptr);
        replay->record(command, packet->buffer, frame.settings);
        VkBufferMemoryBarrier release = acquire; release.srcQueueFamilyIndex = native->family; release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        release.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; release.dstAccessMask = 0;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1, &release, 0, nullptr);
        Device::check(vkEndCommandBuffer(command), "RR Vulkan end");
        const uint64_t ready = sequence + 1, done = sequence + 2, complete = sequence + 3;
        VkTimelineSemaphoreSubmitInfo values{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        values.waitSemaphoreValueCount = values.signalSemaphoreValueCount = 1; values.pWaitSemaphoreValues = &ready; values.pSignalSemaphoreValues = &done;
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, &values}; submit.waitSemaphoreCount = submit.signalSemaphoreCount = 1;
        submit.pWaitSemaphores = submit.pSignalSemaphores = &packet->semaphore; submit.pWaitDstStageMask = &stage;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
        Device::check(vkResetFences(native->device, 1, &completion), "RR completion reset");
        retained = std::make_unique<Snapshot>(frame); queue = q; pending = true;
        try {
            ID3D12CommandList* inputs[] = {producer.Get()}; q->ExecuteCommandLists(1, inputs);
            dx(q->Signal(fence.Get(), ready), "RR input ready signal");
            Device::check(vkQueueSubmit(native->queue, 1, &submit, completion), "RR model submission");
            dx(q->Wait(fence.Get(), done), "RR output queue wait");
            ID3D12CommandList* outputs[] = {consumer.Get()}; q->ExecuteCommandLists(1, outputs);
            dx(q->Signal(fence.Get(), complete), "RR completion signal");
            dx(fence->SetEventOnCompletion(complete, event.value), "RR completion event");
            if (WaitForSingleObject(event.value, 5000) != WAIT_OBJECT_0) throw std::runtime_error("RR D3D12 completion timeout");
            Device::check(vkWaitForFences(native->device, 1, &completion, VK_TRUE, 5000000000ull), "RR Vulkan completion wait");
            if (fence->GetCompletedValue() == UINT64_MAX) throw std::runtime_error("RR D3D12 device removed");
            pending = false; failed = false; retained.reset(); queue.Reset(); sequence = complete;
        } catch (...) { failed = true; throw; }
    }
};
Backend::Backend(ID3D12Device* device, const Creation& creation, const std::string& directory) {
    std::lock_guard<std::mutex> lock(dispatchMutex);
    auto candidate = std::make_unique<Impl>(device, creation);
    candidate->initialize(directory); impl = std::move(candidate);
}
Backend::~Backend() {
    std::lock_guard<std::mutex> lock(dispatchMutex);
    if (impl && impl->pending) {
        std::fprintf(stderr, "D4R_RR_QUARANTINE incomplete GPU work; retaining backend and guide resources\n");
        (void)impl.release();
    } else impl.reset();
}
void Backend::evaluate(ID3D12CommandQueue* queue, const Snapshot& frame, const std::array<ResourceAccess, Snapshot::Count>& access) {
    std::lock_guard<std::mutex> lock(dispatchMutex);
    impl->evaluate(queue, frame, access);
}
void Backend::validate_frame(const Snapshot& frame) {
    std::lock_guard<std::mutex> lock(dispatchMutex);
    impl->validate_frame(frame);
}
void Backend::drain() {
    std::lock_guard<std::mutex> lock(dispatchMutex);
    if (impl->pending || impl->failed) throw std::runtime_error("RR backend cannot retire failed GPU work");
}
}
