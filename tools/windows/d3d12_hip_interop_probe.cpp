#if defined(__MINGW32__)
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#endif
#include "hip_api.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;
using namespace d4r::diag;

static void handle_checkpoint(const char* stage)
{
    DWORD count = 0;
    if (!GetProcessHandleCount(GetCurrentProcess(), &count))
        throw std::runtime_error("GetProcessHandleCount failed");
    std::printf("HANDLE_STAGE %s count=%lu\n", stage, count);
}

static void dx(HRESULT hr, const char* call)
{
    std::printf("D3D12 %s -> 0x%08lx\n", call, static_cast<unsigned long>(hr));
    if (FAILED(hr)) throw std::runtime_error(std::string(call) + " HRESULT=" + std::to_string(hr));
}
struct NativeHandle {
    HANDLE handle = nullptr;
    ~NativeHandle() { if (handle) CloseHandle(handle); }
};
struct ExternalApi {
    HipApi& hip;
#define EXT(name) decltype(&::name) name = hip.library.symbol<decltype(&::name)>(#name)
    EXT(hipImportExternalMemory);
    EXT(hipExternalMemoryGetMappedBuffer);
    EXT(hipDestroyExternalMemory);
    EXT(hipImportExternalSemaphore);
    EXT(hipDestroyExternalSemaphore);
    EXT(hipWaitExternalSemaphoresAsync);
    EXT(hipSignalExternalSemaphoresAsync);
#undef EXT
};
struct Imported {
    ExternalApi& api;
    hipExternalMemory_t memory = nullptr;
    hipExternalSemaphore_t semaphore = nullptr;
    hipStream_t stream = nullptr;
    bool owns_stream = true;
    void* mapped = nullptr;
    void* errors = nullptr;
    ~Imported() {
        // No resource may be released while the HIP stream is still using it.
        if (stream) (void)api.hip.hipStreamSynchronize(stream);
        if (stream && owns_stream) (void)api.hip.hipStreamDestroy(stream);
        if (errors) (void)api.hip.hipFree(errors);
        if (mapped) (void)api.hip.hipFree(mapped);
        if (memory) (void)api.hipDestroyExternalMemory(memory);
        if (semaphore) (void)api.hipDestroyExternalSemaphore(semaphore);
    }
};
static D3D12_RESOURCE_DESC buffer_desc(uint64_t bytes)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1;
    desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}
static ComPtr<ID3D12Resource> make_buffer(ID3D12Device* device, uint64_t bytes,
    D3D12_HEAP_TYPE type, D3D12_HEAP_FLAGS flags, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = type;
    properties.CreationNodeMask = properties.VisibleNodeMask = 1;
    auto desc = buffer_desc(bytes);
    ComPtr<ID3D12Resource> resource;
    dx(device->CreateCommittedResource(&properties, flags, &desc, state, nullptr,
        IID_PPV_ARGS(resource.GetAddressOf())), "CreateCommittedResource(buffer)");
    return resource;
}
static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before; barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

static void import_lifetime(ID3D12Device* device, ExternalApi& api, const Args& args)
{
    if (args.interop_mode != "resource" && args.interop_mode != "import" && args.interop_mode != "map")
        throw std::runtime_error("--interop-mode must be roundtrip, resource, import or map");
    DWORD baseline = 0, handles = 0;
    for (unsigned i = 0; i < args.iterations; ++i) {
        {
            auto resource = make_buffer(device, 65536, D3D12_HEAP_TYPE_DEFAULT,
                D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_STATE_COMMON);
            NativeHandle handle;
            dx(device->CreateSharedHandle(resource.Get(), nullptr, GENERIC_ALL, nullptr,
                &handle.handle), "CreateSharedHandle(lifecycle)");
            Imported imported{api};
            if (args.interop_mode != "resource") {
                hipExternalMemoryHandleDesc desc{};
                desc.type = hipExternalMemoryHandleTypeD3D12Resource;
                desc.handle.win32.handle = handle.handle;
                desc.size = 65536;
                desc.flags = hipExternalMemoryDedicated;
                api.hip.check(api.hipImportExternalMemory(&imported.memory, &desc), "hipImportExternalMemory(lifecycle)");
                if (args.interop_mode == "map") {
                    hipExternalMemoryBufferDesc map{}; map.size = 65536;
                    api.hip.check(api.hipExternalMemoryGetMappedBuffer(&imported.mapped, imported.memory, &map), "hipExternalMemoryGetMappedBuffer(lifecycle)");
                    api.hip.check(api.hip.hipFree(imported.mapped), "hipFree(lifecycle)"); imported.mapped = nullptr;
                }
                api.hip.check(api.hipDestroyExternalMemory(imported.memory), "hipDestroyExternalMemory(lifecycle)"); imported.memory = nullptr;
            }
        }
        if (!GetProcessHandleCount(GetCurrentProcess(), &handles)) throw std::runtime_error("GetProcessHandleCount failed");
        size_t free_bytes = 0, total_bytes = 0;
        api.hip.check(api.hip.hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo");
        if (i == 1 || (args.iterations == 1 && i == 0)) baseline = handles;
        std::printf("IMPORT_CLEANUP mode=%s iteration=%u handles=%lu baseline=%lu free_vram=%zu\n",
            args.interop_mode.c_str(), i, handles, baseline, free_bytes);
    }
    if (handles > baseline + 4) throw std::runtime_error("External memory lifecycle handle growth");
    std::printf("PASS INTEROP_LIFETIME mode=%s iterations=%u\n", args.interop_mode.c_str(), args.iterations);
}

int main(int argc, char** argv)
{
    start();
    try {
        Args args(argc, argv);
        if (args.module.empty() && args.interop_mode == "roundtrip") throw std::runtime_error("--module required");
        if (args.stream_lifetime != "persistent" && args.stream_lifetime != "cycle")
            throw std::runtime_error("--stream-lifetime must be persistent or cycle");
        HipApi hip(args.hip_root);
        hipDeviceProp_t props{};
        hip.select_gfx1201(args.device, props);
        ExternalApi external{hip};
        ComPtr<IDXGIFactory4> factory;
        dx(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> adapter;
        bool matched = false;
        for (UINT i = 0; ; ++i) {
            const HRESULT enumerated = factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf());
            if (enumerated == DXGI_ERROR_NOT_FOUND) break;
            dx(enumerated, "EnumAdapters1");
            DXGI_ADAPTER_DESC1 desc{};
            dx(adapter->GetDesc1(&desc), "GetDesc1");
            std::printf("DXGI adapter=%s LUID=%08lx:%08lx\n", utf8(desc.Description).c_str(),
                static_cast<unsigned long>(desc.AdapterLuid.HighPart), desc.AdapterLuid.LowPart);
            if (std::memcmp(&desc.AdapterLuid, props.luid, sizeof(LUID)) == 0) { matched = true; break; }
        }
        if (!matched) throw std::runtime_error("HIP/DXGI LUID mismatch: cross-adapter import prohibited");
        ComPtr<ID3D12Device> device;
        dx(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.GetAddressOf())), "D3D12CreateDevice");
        if (args.interop_mode != "roundtrip") { import_lifetime(device.Get(), external, args); return 0; }
        ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        dx(device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.GetAddressOf())), "CreateCommandQueue");
        ComPtr<ID3D12Fence> shared_fence, complete;
        dx(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(shared_fence.GetAddressOf())), "CreateFence(shared)");
        dx(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(complete.GetAddressOf())), "CreateFence(completion)");
        NativeHandle fence_handle, event;
        event.handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event.handle) throw std::runtime_error("CreateEvent failed");
        dx(device->CreateSharedHandle(shared_fence.Get(), nullptr, GENERIC_ALL, nullptr,
            &fence_handle.handle), "CreateSharedHandle(fence)");

        // R32_UINT texture -> linear shared buffer -> HIP -> texture -> verification readback.
        constexpr UINT width = 256, height = 64, count = width * height;
        const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(uint32_t);
        auto upload = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = width; td.Height = height; td.DepthOrArraySize = 1; td.MipLevels = 1;
        td.SampleDesc.Count = 1; td.Format = DXGI_FORMAT_R32_UINT;
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        hp.CreationNodeMask = hp.VisibleNodeMask = 1;
        ComPtr<ID3D12Resource> texture;
        dx(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(texture.GetAddressOf())), "CreateCommittedResource(texture)");
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT rows; UINT64 row_bytes, total_bytes;
        device->GetCopyableFootprints(&td, 0, 1, 0, &footprint, &rows, &row_bytes, &total_bytes);
        if (total_bytes != bytes || footprint.Footprint.RowPitch != width * 4)
            throw std::runtime_error("Unexpected footprint; test requires no padding");
        auto texture_copy = D3D12_TEXTURE_COPY_LOCATION{};
        texture_copy.pResource = texture.Get();
        texture_copy.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        auto linear_copy = D3D12_TEXTURE_COPY_LOCATION{};
        linear_copy.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        linear_copy.PlacedFootprint = footprint;
        hipModule_t module = nullptr;
        hip.check(hip.hipModuleLoad(&module, args.module.c_str()), "hipModuleLoad");
        struct ModuleCleanup { HipApi& hip; hipModule_t m; ~ModuleCleanup() { (void)hip.hipModuleUnload(m); } } mc{hip, module};
        hipFunction_t kernel = nullptr;
        hip.check(hip.hipModuleGetFunction(&kernel, module, "d4r_probe_interop"), "hipModuleGetFunction");
        struct StreamCleanup {
            HipApi& hip;
            hipStream_t stream = nullptr;
            ~StreamCleanup() { if (stream) {
                (void)hip.hipStreamSynchronize(stream);
                (void)hip.hipStreamDestroy(stream);
            } }
        } persistent{hip};
        if (args.stream_lifetime == "persistent")
            hip.check(hip.hipStreamCreateWithFlags(&persistent.stream, hipStreamNonBlocking), "hipStreamCreateWithFlags(persistent)");
        loaded_modules();
        DWORD handles_before = 0; GetProcessHandleCount(GetCurrentProcess(), &handles_before);
        DWORD warmed_handles = 0;
        DWORD final_cycle_handles = 0;
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            const uint32_t seed = 0x12345678u + iteration * 31u;
            auto shared = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED,
                D3D12_RESOURCE_STATE_COPY_DEST);
            const auto shared_desc = buffer_desc(bytes);
            D3D12_RESOURCE_ALLOCATION_INFO allocation_info{};
#if defined(__MINGW32__)
            device->GetResourceAllocationInfo(&allocation_info, 0, 1, &shared_desc);
#else
            allocation_info = device->GetResourceAllocationInfo(0, 1, &shared_desc);
#endif
            NativeHandle memory_handle;
            dx(device->CreateSharedHandle(shared.Get(), nullptr, GENERIC_ALL, nullptr,
                &memory_handle.handle), "CreateSharedHandle(resource)");
            Imported imported{external};
            handle_checkpoint("created_resource");
            hipExternalMemoryHandleDesc memory_desc{};
            memory_desc.type = hipExternalMemoryHandleTypeD3D12Resource;
            memory_desc.handle.win32.handle = memory_handle.handle;
            memory_desc.size = allocation_info.SizeInBytes;
            memory_desc.flags = hipExternalMemoryDedicated;
            std::printf("STAGE hipImportExternalMemory(D3D12Resource) size=%llu\n", memory_desc.size);
            hip.check(external.hipImportExternalMemory(&imported.memory, &memory_desc), "hipImportExternalMemory(D3D12Resource)");
            handle_checkpoint("imported_memory");
            hipExternalMemoryBufferDesc map_desc{}; map_desc.size = bytes;
            hip.check(external.hipExternalMemoryGetMappedBuffer(&imported.mapped, imported.memory, &map_desc), "hipExternalMemoryGetMappedBuffer");
            handle_checkpoint("mapped_memory");
            hipExternalSemaphoreHandleDesc sem_desc{};
            sem_desc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
            sem_desc.handle.win32.handle = fence_handle.handle;
            std::printf("STAGE hipImportExternalSemaphore(D3D12Fence)\n");
            hip.check(external.hipImportExternalSemaphore(&imported.semaphore, &sem_desc), "hipImportExternalSemaphore(D3D12Fence)");
            handle_checkpoint("imported_semaphore");
            if (persistent.stream) { imported.stream = persistent.stream; imported.owns_stream = false; }
            else hip.check(hip.hipStreamCreateWithFlags(&imported.stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
            handle_checkpoint("created_stream");
            hip.check(hip.hipMalloc(&imported.errors, sizeof(uint32_t)), "hipMalloc(errors)");
            const uint32_t zero = 0;
            hip.check(hip.hipMemcpy(imported.errors, &zero, sizeof(zero), hipMemcpyHostToDevice), "hipMemcpy(error initialization)");
            uint32_t* host_upload = nullptr;
            D3D12_RANGE no_read{0, 0};
            dx(upload->Map(0, &no_read, reinterpret_cast<void**>(&host_upload)), "Map(upload initialization)");
            for (uint32_t i = 0; i < count; ++i) host_upload[i] = pattern(i, seed);
            D3D12_RANGE written{0, static_cast<SIZE_T>(bytes)}; upload->Unmap(0, &written);
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList> first, second;
            dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf())), "CreateCommandAllocator");
            dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                IID_PPV_ARGS(first.GetAddressOf())), "CreateCommandList(input)");
            if (iteration) transition(first.Get(), texture.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            linear_copy.pResource = upload.Get();
            first->CopyTextureRegion(&texture_copy, 0, 0, 0, &linear_copy, nullptr);
            transition(first.Get(), texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            linear_copy.pResource = shared.Get();
            first->CopyTextureRegion(&linear_copy, 0, 0, 0, &texture_copy, nullptr);
            transition(first.Get(), shared.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            dx(first->Close(), "Close(input)");
            ID3D12CommandList* submit[] = {first.Get()}; queue->ExecuteCommandLists(1, submit);
            const UINT64 ready = 2ull * iteration + 1, done = ready + 1;
            dx(queue->Signal(shared_fence.Get(), ready), "Signal(D3D12 input ready)");
            hipExternalSemaphoreWaitParams wait{}; wait.params.fence.value = ready;
            hip.check(external.hipWaitExternalSemaphoresAsync(&imported.semaphore, &wait, 1, imported.stream), "hipWaitExternalSemaphoresAsync");
            uint32_t n = count, mutable_seed = seed;
            void* parameters[] = {&imported.mapped, &n, &mutable_seed, &imported.errors};
            hip.check(hip.hipModuleLaunchKernel(kernel, (n + 127) / 128, 1, 1, 128, 1, 1, 0,
                imported.stream, parameters, nullptr), "hipModuleLaunchKernel(shared VRAM)");
            hipExternalSemaphoreSignalParams signal{}; signal.params.fence.value = done;
            hip.check(external.hipSignalExternalSemaphoresAsync(&imported.semaphore, &signal, 1, imported.stream), "hipSignalExternalSemaphoresAsync");
            dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                IID_PPV_ARGS(second.GetAddressOf())), "CreateCommandList(output)");
            transition(second.Get(), shared.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            transition(second.Get(), texture.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            second->CopyTextureRegion(&texture_copy, 0, 0, 0, &linear_copy, nullptr);
            transition(second.Get(), texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            linear_copy.pResource = readback.Get();
            second->CopyTextureRegion(&linear_copy, 0, 0, 0, &texture_copy, nullptr);
            dx(second->Close(), "Close(output)");
            dx(queue->Wait(shared_fence.Get(), done), "Wait(HIP output ready)");
            submit[0] = second.Get(); queue->ExecuteCommandLists(1, submit);
            dx(queue->Signal(complete.Get(), iteration + 1), "Signal(completion)");
            dx(complete->SetEventOnCompletion(iteration + 1, event.handle), "SetEventOnCompletion");
            if (WaitForSingleObject(event.handle, 30000) != WAIT_OBJECT_0) throw std::runtime_error("D3D12 fence timeout");
            hip.check(hip.hipStreamSynchronize(imported.stream), "hipStreamSynchronize");
            uint32_t errors = ~0u;
            hip.check(hip.hipMemcpy(&errors, imported.errors, sizeof(errors), hipMemcpyDeviceToHost), "hipMemcpy(error verification)");
            if (errors) throw std::runtime_error("HIP input mismatch count=" + std::to_string(errors));
            uint32_t* host_result = nullptr;
            D3D12_RANGE read{0, static_cast<SIZE_T>(bytes)};
            dx(readback->Map(0, &read, reinterpret_cast<void**>(&host_result)), "Map(final verification)");
            bool good = true;
            for (uint32_t i = 0; i < count; ++i) if (host_result[i] != (pattern(i, seed) ^ 0x3579bdf1u)) { good = false; break; }
            readback->Unmap(0, &no_read);
            if (!good) throw std::runtime_error("D3D12 output mismatch");
            // Check each release in the successful path; RAII handles all failures.
            if (!persistent.stream) hip.check(hip.hipStreamDestroy(imported.stream), "hipStreamDestroy");
            imported.stream = nullptr;
            handle_checkpoint("released_stream_reference");
            hip.check(hip.hipFree(imported.errors), "hipFree(errors)"); imported.errors = nullptr;
            handle_checkpoint("freed_errors");
            hip.check(hip.hipFree(imported.mapped), "hipFree(external mapping)"); imported.mapped = nullptr;
            handle_checkpoint("freed_mapping");
            hip.check(external.hipDestroyExternalMemory(imported.memory), "hipDestroyExternalMemory"); imported.memory = nullptr;
            handle_checkpoint("destroyed_memory");
            hip.check(external.hipDestroyExternalSemaphore(imported.semaphore), "hipDestroyExternalSemaphore"); imported.semaphore = nullptr;
            handle_checkpoint("destroyed_semaphore");
            shared.Reset();
            if (!CloseHandle(memory_handle.handle)) throw std::runtime_error("CloseHandle(shared memory) failed");
            memory_handle.handle = nullptr;
            first.Reset(); second.Reset(); allocator.Reset();
            DWORD cycle_handles = 0; GetProcessHandleCount(GetCurrentProcess(), &cycle_handles);
            if (iteration == 0) warmed_handles = cycle_handles;
            final_cycle_handles = cycle_handles;
            std::printf("CLEANUP iteration=%u handles=%lu warmed_baseline=%lu\n", iteration, cycle_handles, warmed_handles);
            std::printf("ROUNDTRIP iteration=%u verified=%u vram_transfer=1 cpu_staging_between_apis=0\n", iteration, count);
        }
        if (persistent.stream) {
            hip.check(hip.hipStreamDestroy(persistent.stream), "hipStreamDestroy(persistent)");
            persistent.stream = nullptr;
        }
        DWORD handles_after = 0; GetProcessHandleCount(GetCurrentProcess(), &handles_after);
        std::printf("HANDLES before=%lu after=%lu\n", handles_before, handles_after);
        if (final_cycle_handles > warmed_handles + 4) throw std::runtime_error("Handle count grew after first-cycle warmup");
        std::printf("PASS INTEROP architecture=gfx1201 iterations=%u texture=R32_UINT external_fence=1\n", args.iterations);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL INTEROP %s\n", e.what());
        loaded_modules();
        return 4;
    }
}
