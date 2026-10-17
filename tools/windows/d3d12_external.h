#pragma once
#if defined(__MINGW32__)
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#endif
#include "hip_api.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <mutex>
#include <wrl/client.h>

namespace d4r::win {
using Microsoft::WRL::ComPtr;
using diag::HipApi;
inline void dx(HRESULT result, const char* operation) {
    if (FAILED(result)) throw std::runtime_error(std::string(operation) + " HRESULT=" + std::to_string(result));
}
struct Handle {
    HANDLE value = nullptr;
    Handle() = default;
    Handle(const Handle&) = delete;
    ~Handle() { if (value) CloseHandle(value); }
};
struct ExternalApi {
    HipApi& hip;
#define D4R_EXTERNAL(name) decltype(&::name) name = hip.library.symbol<decltype(&::name)>(#name)
    D4R_EXTERNAL(hipImportExternalMemory);
    D4R_EXTERNAL(hipExternalMemoryGetMappedBuffer);
    D4R_EXTERNAL(hipDestroyExternalMemory);
    D4R_EXTERNAL(hipImportExternalSemaphore);
    D4R_EXTERNAL(hipDestroyExternalSemaphore);
    D4R_EXTERNAL(hipWaitExternalSemaphoresAsync);
    D4R_EXTERNAL(hipSignalExternalSemaphoresAsync);
#undef D4R_EXTERNAL
};
inline D3D12_RESOURCE_DESC resource_desc(ID3D12Resource* resource) {
#if defined(__MINGW32__)
    D3D12_RESOURCE_DESC desc{}; resource->GetDesc(&desc); return desc;
#else
    return resource->GetDesc();
#endif
}
inline LUID adapter_luid(ID3D12Device* device) {
#if defined(__MINGW32__)
    LUID luid{}; device->GetAdapterLuid(&luid); return luid;
#else
    return device->GetAdapterLuid();
#endif
}
inline ComPtr<ID3D12Resource> make_buffer(ID3D12Device* device, uint64_t bytes,
    D3D12_HEAP_TYPE heapType, D3D12_HEAP_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = heapType;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> resource;
    dx(device->CreateCommittedResource(&heap, flags, &desc, state, nullptr,
        IID_PPV_ARGS(resource.GetAddressOf())), "CreateCommittedResource(buffer)");
    return resource;
}
inline void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                       D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
}
struct ResourceAccess {
    D3D12_RESOURCE_STATES legacy = D3D12_RESOURCE_STATE_COMMON;
    bool enhanced = false, pending_split = false, inherited = false;
    D3D12_BARRIER_LAYOUT layout = D3D12_BARRIER_LAYOUT_COMMON;
    D3D12_BARRIER_ACCESS access = D3D12_BARRIER_ACCESS_COMMON;
    D3D12_BARRIER_SYNC sync = D3D12_BARRIER_SYNC_ALL;
    ResourceAccess() = default;
    ResourceAccess(D3D12_RESOURCE_STATES state) : legacy(state) {}
};
inline void texture_copy_barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* texture,
    const ResourceAccess& original, bool output, bool restore) {
    if (original.pending_split) throw std::runtime_error("Cannot use a texture inside an unfinished split barrier");
    if (!original.enhanced) {
        const auto copy = output ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
        if (original.legacy == copy) return;
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {texture, 0, restore ? copy : original.legacy, restore ? original.legacy : copy};
        list->ResourceBarrier(1, &barrier); return;
    }
    ComPtr<ID3D12GraphicsCommandList7> newer;
    dx(list->QueryInterface(IID_PPV_ARGS(newer.GetAddressOf())), "Enhanced interop command list");
    D3D12_TEXTURE_BARRIER barrier{}; barrier.pResource = texture;
    barrier.LayoutBefore = restore ? (output ? D3D12_BARRIER_LAYOUT_COPY_DEST : D3D12_BARRIER_LAYOUT_COPY_SOURCE) : original.layout;
    barrier.LayoutAfter = restore ? original.layout : (output ? D3D12_BARRIER_LAYOUT_COPY_DEST : D3D12_BARRIER_LAYOUT_COPY_SOURCE);
    barrier.AccessBefore = restore ? (output ? D3D12_BARRIER_ACCESS_COPY_DEST : D3D12_BARRIER_ACCESS_COPY_SOURCE) : original.access;
    barrier.AccessAfter = restore ? original.access : (output ? D3D12_BARRIER_ACCESS_COPY_DEST : D3D12_BARRIER_ACCESS_COPY_SOURCE);
    // Interop runs in a separate submitted direct list. ALL is a conservative
    // synchronization scope for the app's original access class/layout.
    barrier.SyncBefore = restore ? D3D12_BARRIER_SYNC_COPY : D3D12_BARRIER_SYNC_ALL;
    barrier.SyncAfter = restore ? D3D12_BARRIER_SYNC_ALL : D3D12_BARRIER_SYNC_COPY;
    barrier.Subresources.IndexOrFirstMipLevel = 0; // Single subresource 0.
    D3D12_BARRIER_GROUP group{}; group.Type = D3D12_BARRIER_TYPE_TEXTURE; group.NumBarriers = 1; group.pTextureBarriers = &barrier;
    newer->Barrier(1, &group);
}

class SharedPlane {
    ExternalApi& api_;
    hipExternalMemory_t memory_ = nullptr;
    ComPtr<ID3D12Heap> heap_; // only for the explicit-heap export; outlives buffer (declared first)
    void release() noexcept {
        if (mapped) { (void)api_.hip.hipFree(mapped); mapped = nullptr; }
        if (memory_) { (void)api_.hipDestroyExternalMemory(memory_); memory_ = nullptr; }
    }
    // Export a committed shared buffer (the tested path). Returns false when the driver refuses
    // the export: Cyberpunk 2077 saw CreateSharedHandle E_INVALIDARG on a fresh shared buffer
    // after a DLSS feature recreate, although the same sequence works outside the game.
    bool export_committed(ID3D12Device* device, Handle& handle, UINT64& importBytes) {
        buffer = make_buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_STATE_COMMON);
        const auto desc = resource_desc(buffer.Get());
        D3D12_RESOURCE_ALLOCATION_INFO allocation{};
#if defined(__MINGW32__)
        device->GetResourceAllocationInfo(&allocation, 0, 1, &desc);
#else
        allocation = device->GetResourceAllocationInfo(0, 1, &desc);
#endif
        const HRESULT result = device->CreateSharedHandle(buffer.Get(), nullptr, GENERIC_ALL, nullptr, &handle.value);
        if (SUCCEEDED(result)) { importBytes = allocation.SizeInBytes; return true; }
        // A wrapping layer can hand back a resource whose owner is not the device it was created on.
        ComPtr<ID3D12Device> owner; (void)buffer->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()));
        std::fprintf(stderr, "D4R_SHARED_PLANE_EXPORT committed HRESULT=0x%08lx bytes=%llu allocation=%llu device_removed=0x%08lx owner_matches=%d\n",
            static_cast<unsigned long>(result), static_cast<unsigned long long>(bytes),
            static_cast<unsigned long long>(allocation.SizeInBytes), static_cast<unsigned long>(device->GetDeviceRemovedReason()),
            owner.Get() == device);
        buffer.Reset(); handle.value = nullptr;
        return false;
    }
    // Last resort when the device refuses every export (Cyberpunk 2077 with the FSR RR backend refused
    // both the committed and the heap export): allocate and export on a private device of the same
    // adapter, then open that allocation on the caller's device for the copies. HIP imports the
    // private device's handle, which names the same memory.
    ComPtr<ID3D12Device> privateDevice_;
    ComPtr<ID3D12Resource> privateBuffer_;
    static ComPtr<ID3D12Device> private_device(ID3D12Device* device) {
        static std::mutex lock; static ComPtr<ID3D12Device> cached; static LUID cachedLuid{};
        std::lock_guard<std::mutex> guard(lock);
        const LUID luid = adapter_luid(device);
        if (cached && cachedLuid.LowPart == luid.LowPart && cachedLuid.HighPart == luid.HighPart) return cached;
        using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
        HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
        auto create = dxgi ? reinterpret_cast<CreateFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1"))) : nullptr;
        if (!create) throw std::runtime_error("CreateDXGIFactory1 unavailable (private shared-plane device)");
        ComPtr<IDXGIFactory4> factory; dx(create(IID_PPV_ARGS(factory.GetAddressOf())), "CreateDXGIFactory1(private shared-plane device)");
        ComPtr<IDXGIAdapter1> adapter; dx(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(adapter.GetAddressOf())), "EnumAdapterByLuid(private shared-plane device)");
        // D3D12CreateDevice returns the adapter's existing singleton (the game's own device), so an
        // independent device comes from the device factory, neither reusing nor replacing the singleton.
        using GetInterface = HRESULT(WINAPI*)(REFCLSID, REFIID, void**);
        HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
        auto getInterface = d3d12 ? reinterpret_cast<GetInterface>(reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12GetInterface"))) : nullptr;
        if (!getInterface) throw std::runtime_error("D3D12GetInterface unavailable (private shared-plane device)");
        static const GUID deviceFactoryClsid = {0x114863bf, 0xc386, 0x4aee, {0xb3, 0x9d, 0x8f, 0x0b, 0xbb, 0x06, 0x29, 0x55}};
        ComPtr<ID3D12DeviceFactory> deviceFactory;
        dx(getInterface(deviceFactoryClsid, IID_PPV_ARGS(deviceFactory.GetAddressOf())), "D3D12GetInterface(device factory)");
        dx(deviceFactory->SetFlags(D3D12_DEVICE_FACTORY_FLAG_DISALLOW_STORING_NEW_DEVICE_AS_SINGLETON), "Device factory flags");
        ComPtr<ID3D12Device> created;
        dx(deviceFactory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(created.GetAddressOf())), "Create private shared-plane device");
        ComPtr<IUnknown> createdIdentity, callerIdentity;
        (void)created->QueryInterface(IID_PPV_ARGS(createdIdentity.GetAddressOf()));
        (void)device->QueryInterface(IID_PPV_ARGS(callerIdentity.GetAddressOf()));
        std::fprintf(stderr, "D4R_SHARED_PLANE_PRIVATE_DEVICE independent=%d\n", createdIdentity.Get() != callerIdentity.Get());
        cached = created; cachedLuid = luid;
        return cached;
    }
    void export_private(ID3D12Device* device, Handle& handle, UINT64& importBytes) {
        privateDevice_ = private_device(device);
        privateBuffer_ = make_buffer(privateDevice_.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_STATE_COMMON);
        const auto desc = resource_desc(privateBuffer_.Get());
        D3D12_RESOURCE_ALLOCATION_INFO allocation{};
#if defined(__MINGW32__)
        privateDevice_->GetResourceAllocationInfo(&allocation, 0, 1, &desc);
#else
        allocation = privateDevice_->GetResourceAllocationInfo(0, 1, &desc);
#endif
        dx(privateDevice_->CreateSharedHandle(privateBuffer_.Get(), nullptr, GENERIC_ALL, nullptr, &handle.value), "CreateSharedHandle(private device)");
        dx(device->OpenSharedHandle(handle.value, IID_PPV_ARGS(buffer.GetAddressOf())), "OpenSharedHandle(private shared plane)");
        importBytes = allocation.SizeInBytes;
    }
    // Fallback: share an explicit heap that holds the buffer as a placed resource.
    bool export_heap(ID3D12Device* device, Handle& handle, UINT64& importBytes) {
        constexpr UINT64 alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        D3D12_HEAP_DESC heap{};
        heap.SizeInBytes = (bytes + alignment - 1) / alignment * alignment;
        heap.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.Properties.CreationNodeMask = heap.Properties.VisibleNodeMask = 1;
        heap.Alignment = alignment;
        heap.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        dx(device->CreateHeap(&heap, IID_PPV_ARGS(heap_.GetAddressOf())), "CreateHeap(shared plane)");
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = bytes;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        dx(device->CreatePlacedResource(heap_.Get(), 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(buffer.GetAddressOf())), "CreatePlacedResource(shared plane)");
        const HRESULT result = device->CreateSharedHandle(heap_.Get(), nullptr, GENERIC_ALL, nullptr, &handle.value);
        if (SUCCEEDED(result)) { importBytes = heap.SizeInBytes; return true; }
        std::fprintf(stderr, "D4R_SHARED_PLANE_EXPORT heap HRESULT=0x%08lx\n", static_cast<unsigned long>(result));
        buffer.Reset(); heap_.Reset(); handle.value = nullptr;
        return false;
    }
public:
    ComPtr<ID3D12Resource> buffer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    void* mapped = nullptr;
    SharedPlane(ExternalApi& api, ID3D12Device* device, const D3D12_RESOURCE_DESC& texture) : api_(api) {
        UINT rows = 0; UINT64 rowBytes = 0;
        device->GetCopyableFootprints(&texture, 0, 1, 0, &footprint, &rows, &rowBytes, &bytes);
        if (texture.Format == DXGI_FORMAT_R32G8X24_TYPELESS || texture.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT || texture.Format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS) {
            // D3D12 exposes D32S8 as separate depth/stencil copy planes. Only
            // subresource 0 (R32 depth) belongs to NGX; never copy interleaved
            // 64-bit texels or touch the stencil plane.
            if ((footprint.Footprint.Format != DXGI_FORMAT_R32_FLOAT && footprint.Footprint.Format != DXGI_FORMAT_R32_TYPELESS) || rowBytes != texture.Width * 4)
                throw std::runtime_error("Unexpected D32S8 depth-plane copy footprint");
        }
        if (!bytes || bytes == UINT64_MAX || !footprint.Footprint.RowPitch)
            throw std::runtime_error("Invalid shared texture footprint");
        Handle handle;
        UINT64 importBytes = 0;
        // D4R_SHARED_PLANE_EXPORT=heap or =private forces that fallback, for testing it.
        const char* forcedValue = std::getenv("D4R_SHARED_PLANE_EXPORT");
        const std::string forced = forcedValue ? forcedValue : "";
        const char* path = "committed";
        bool exported = forced.empty() && (export_committed(device, handle, importBytes) || export_committed(device, handle, importBytes));
        if (!exported && forced != "private") { path = "heap"; exported = export_heap(device, handle, importBytes); }
        if (!exported) { path = "private"; export_private(device, handle, importBytes); }
        const bool heapImport = std::string(path) == "heap";
        try {
            hipExternalMemoryHandleDesc import{};
            import.type = heapImport ? hipExternalMemoryHandleTypeD3D12Heap : hipExternalMemoryHandleTypeD3D12Resource;
            import.handle.win32.handle = handle.value;
            import.size = importBytes; import.flags = heapImport ? 0 : hipExternalMemoryDedicated;
            api.hip.check(api.hipImportExternalMemory(&memory_, &import), "hipImportExternalMemory(D3D12)");
            hipExternalMemoryBufferDesc map{}; map.size = bytes;
            api.hip.check(api.hipExternalMemoryGetMappedBuffer(&mapped, memory_, &map), "hipExternalMemoryGetMappedBuffer");
        } catch (...) { release(); throw; }
        // Win32 imports retain their own reference; the exporter closes its HANDLE.
        std::printf("D4R_IMPORT width=%u height=%u pitch=%u allocation=%llu export=%s cpu_copy=0\n",
            footprint.Footprint.Width, footprint.Footprint.Height, footprint.Footprint.RowPitch, importBytes, path);
    }
    SharedPlane(const SharedPlane&) = delete;
    ~SharedPlane() { release(); }
    void copy_input(ID3D12GraphicsCommandList* list, ID3D12Resource* texture, ResourceAccess state) {
        texture_copy_barrier(list, texture, state, false, false);
        transition(list, buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION source{}, dest{};
        source.pResource = texture; source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dest.pResource = buffer.Get(); dest.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dest.PlacedFootprint = footprint;
        list->CopyTextureRegion(&dest, 0, 0, 0, &source, nullptr);
        transition(list, buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        texture_copy_barrier(list, texture, state, false, true);
    }
    void copy_output(ID3D12GraphicsCommandList* list, ID3D12Resource* texture, ResourceAccess state) {
        texture_copy_barrier(list, texture, state, true, false);
        transition(list, buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION source{}, dest{};
        source.pResource = buffer.Get(); source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; source.PlacedFootprint = footprint;
        dest.pResource = texture; dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dest, 0, 0, 0, &source, nullptr);
        transition(list, buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        texture_copy_barrier(list, texture, state, true, true);
    }
};

class SharedTimeline {
    ExternalApi& api_;
    hipExternalSemaphore_t semaphore_ = nullptr;
    hipStream_t stream_ = nullptr;
    Handle event_;
    uint64_t value_ = 0;
public:
    struct Ticket { uint64_t input, output; };
    ComPtr<ID3D12Fence> fence;
    SharedTimeline(ExternalApi& api, ID3D12Device* device) : api_(api) {
        dx(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence.GetAddressOf())), "CreateFence(D3D12/HIP)");
        Handle handle;
        dx(device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle.value), "CreateSharedHandle(fence)");
        event_.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_.value) throw std::runtime_error("CreateEvent(fence) failed");
        hipExternalSemaphoreHandleDesc import{};
        import.type = hipExternalSemaphoreHandleTypeD3D12Fence;
        import.handle.win32.handle = handle.value;
        api.hip.check(api.hipImportExternalSemaphore(&semaphore_, &import), "hipImportExternalSemaphore(D3D12Fence)");
        try { api.hip.check(api.hip.hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate(interop)"); }
        catch (...) { (void)api.hipDestroyExternalSemaphore(semaphore_); semaphore_ = nullptr; throw; }
    }
    ~SharedTimeline() {
        if (stream_) { (void)api_.hip.hipStreamSynchronize(stream_); (void)api_.hip.hipStreamDestroy(stream_); }
        if (semaphore_) (void)api_.hipDestroyExternalSemaphore(semaphore_);
    }
    // Caller must serialize transactions and submit the producer prefix first.
    void wait_input(ID3D12CommandQueue* queue) {
        const uint64_t ready = ++value_;
        dx(queue->Signal(fence.Get(), ready), "Queue Signal(input)");
        wait_ready(ready);
    }
    // Reserve both dependencies on the submission thread. The CUDA worker
    // never submits to the D3D12 queue, so later game Signal/Wait calls retain
    // their original ordering without waiting on a CPU callback.
    Ticket reserve() { return {++value_, ++value_}; }
    void enqueue(ID3D12CommandQueue* queue, Ticket ticket) {
        dx(queue->Signal(fence.Get(), ticket.input), "Queue Signal(async input)");
        dx(queue->Wait(fence.Get(), ticket.output), "Queue Wait(async output)");
    }
    void wait_ready(uint64_t ready) {
        hipExternalSemaphoreWaitParams wait{}; wait.params.fence.value = ready;
        api_.hip.check(api_.hipWaitExternalSemaphoresAsync(&semaphore_, &wait, 1, stream_), "HIP wait(D3D12 input)");
        api_.hip.check(api_.hip.hipStreamSynchronize(stream_), "HIP input wait completion");
        // Establish host-side completion as well: the Windows HIP semaphore
        // wait currently needs a separately verified completion boundary before
        // work is launched on CUDA's other streams. No image crosses the CPU.
        if (fence->GetCompletedValue() < ready) {
            dx(fence->SetEventOnCompletion(ready, event_.value), "SetEventOnCompletion(input)");
            if (WaitForSingleObject(event_.value, 30000) != WAIT_OBJECT_0)
                throw std::runtime_error("D3D12 input completion timeout");
        }
    }
    void signal_output(ID3D12CommandQueue* queue) {
        const uint64_t done = ++value_;
        signal_ready(done);
        dx(queue->Wait(fence.Get(), done), "Queue Wait(HIP output)");
    }
    void signal_ready(uint64_t done) {
        hipExternalSemaphoreSignalParams signal{}; signal.params.fence.value = done;
        api_.hip.check(api_.hipSignalExternalSemaphoresAsync(&semaphore_, &signal, 1, stream_), "HIP signal(output)");
    }
    void drain(ID3D12CommandQueue* queue) {
        const uint64_t done = ++value_;
        dx(queue->Signal(fence.Get(), done), "Queue Signal(completion)");
        if (fence->GetCompletedValue() < done) {
            dx(fence->SetEventOnCompletion(done, event_.value), "SetEventOnCompletion");
            if (WaitForSingleObject(event_.value, 30000) != WAIT_OBJECT_0)
                throw std::runtime_error("D3D12 completion timeout; resources must remain alive");
        }
    }
};
}
