#include "d3d12_command_hooks.h"
#include <MinHook.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace d4r::win::commands {
namespace {
thread_local bool internal = false;
std::mutex map_mutex, installation_mutex, error_mutex;
std::string last_error;
bool installed = false;
struct Event { UINT metadata; std::vector<uint8_t> data; };
using State = std::function<void(ID3D12GraphicsCommandList*)>;
struct Segment {
    // The application's original list is a weak pointer; holding it in its
    // own routing map would prevent final Release and leak the entire feature.
    ID3D12GraphicsCommandList* list = nullptr;
    ComPtr<ID3D12GraphicsCommandList> owner;
    ComPtr<ID3D12CommandAllocator> allocator;
    Boundary boundary;
};
struct Recording {
    std::recursive_mutex mutex;
    ID3D12GraphicsCommandList* original;
    ComPtr<ID3D12GraphicsCommandList> current;
    ComPtr<ID3D12CommandAllocator> current_allocator;
    std::vector<Segment> segments;
    std::vector<State> state;
    std::vector<Event> events;
    std::unordered_map<ID3D12Resource*, D3D12_RESOURCE_STATES> resource_states;
    unsigned active_queries = 0;
    bool render_pass = false, predication = false, indirect_state_unknown = false, closed = false;
    explicit Recording(ID3D12GraphicsCommandList* list) : original(list) {}
};
std::unordered_map<ID3D12GraphicsCommandList*, std::shared_ptr<Recording>> recordings;
std::shared_ptr<Recording> get_recording(ID3D12GraphicsCommandList* self, bool create = true) {
    if (internal) return {};
    std::lock_guard<std::mutex> lock(map_mutex);
    auto found = recordings.find(self);
    if (found != recordings.end()) return found->second;
    if (!create) return {};
    auto value = std::make_shared<Recording>(self);
    recordings.emplace(self, value); return value;
}
void forget(ID3D12GraphicsCommandList* self) {
    std::shared_ptr<Recording> released;
    { std::lock_guard<std::mutex> lock(map_mutex);
      auto it = recordings.find(self);
      if (it != recordings.end()) { released = std::move(it->second); recordings.erase(it); } }
    // Release retained COM objects without holding the map mutex.
}
struct Access {
    std::shared_ptr<Recording> recording;
    std::unique_lock<std::recursive_mutex> lock;
    ID3D12GraphicsCommandList* self;
    explicit Access(ID3D12GraphicsCommandList* list) : recording(get_recording(list)), self(list) {
        if (recording) lock = std::unique_lock<std::recursive_mutex>(recording->mutex);
    }
    ID3D12GraphicsCommandList* target() const {
        return recording && recording->current ? recording->current.Get() : self;
    }
};
template<class T> ComPtr<T> retain(T* pointer) { return ComPtr<T>(pointer); }
template<class T> std::vector<T> copy_values(const T* pointer, size_t count) {
    return pointer ? std::vector<T>(pointer, pointer + count) : std::vector<T>();
}
template<class T> T* values_pointer(std::vector<T>& values) { return values.empty() ? nullptr : values.data(); }
template<class T> struct RetainedArray { std::vector<ComPtr<T>> owners; std::vector<T*> raw; };
template<class T> RetainedArray<T> retain_values(T* const* pointer, size_t count) {
    RetainedArray<T> result;
    for (size_t i = 0; pointer && i < count; ++i) { result.owners.emplace_back(pointer[i]); result.raw.push_back(pointer[i]); }
    return result;
}
struct Hook { void* target; void* replacement; };
std::vector<Hook> attached;
void attach_method(void* target, void* replacement, void** original, const char* name) {
    for (const auto& hook : attached) if (hook.target == target)
        throw std::runtime_error(std::string("Shared D3D12 implementation needs an explicit alias adapter: ") + name);
    const auto result = MH_CreateHook(target, replacement, original);
    if (result != MH_OK) throw std::runtime_error(std::string("Cannot intercept D3D12 ") + name + ": " + MH_StatusToString(result));
    attached.push_back({target, replacement});
    std::printf("D4R_D3D12_HOOK method=%s\n", name);
}

#include "d3d12_command_hooks.generated.h"

using ReleaseFn = ULONG(STDMETHODCALLTYPE*)(IUnknown*);
ReleaseFn original_release = nullptr;
ULONG STDMETHODCALLTYPE hook_release(IUnknown* self) {
    const ULONG count = original_release(self);
    if (!count) forget(reinterpret_cast<ID3D12GraphicsCommandList*>(self));
    return count;
}
HRESULT STDMETHODCALLTYPE hook_Close(ID3D12GraphicsCommandList* self) {
    Access access(self);
    const auto result = original_Close(access.target());
    if (SUCCEEDED(result) && access.recording) access.recording->closed = true;
    return result;
}
HRESULT STDMETHODCALLTYPE hook_Reset(ID3D12GraphicsCommandList* self, ID3D12CommandAllocator* allocator, ID3D12PipelineState* initial_state) {
    if (internal) return original_Reset(self, allocator, initial_state);
    forget(self);
    const auto result = original_Reset(self, allocator, initial_state);
    if (SUCCEEDED(result)) {
        auto recording = get_recording(self);
        auto pipeline = retain(initial_state);
        recording->state.emplace_back([pipeline](ID3D12GraphicsCommandList* target) {
            if (pipeline) original_SetPipelineState(target, pipeline.Get());
        });
    }
    return result;
}
void STDMETHODCALLTYPE hook_ClearState(ID3D12GraphicsCommandList* self, ID3D12PipelineState* pipeline_state) {
    Access access(self);
    if (access.recording) {
        access.recording->state.clear(); access.recording->indirect_state_unknown = false;
        auto pipeline = retain(pipeline_state);
        access.recording->state.emplace_back([pipeline](ID3D12GraphicsCommandList* target) { original_ClearState(target, pipeline.Get()); });
    }
    original_ClearState(access.target(), pipeline_state);
}
void STDMETHODCALLTYPE hook_ResourceBarrier(ID3D12GraphicsCommandList* self, UINT barrier_count, const D3D12_RESOURCE_BARRIER* barriers) {
    Access access(self);
    if (access.recording) for (UINT i = 0; i < barrier_count; ++i)
        if (barriers[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION && barriers[i].Flags != D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY)
            access.recording->resource_states[barriers[i].Transition.pResource] = barriers[i].Transition.StateAfter;
    original_ResourceBarrier(access.target(), barrier_count, barriers);
}
void STDMETHODCALLTYPE hook_Barrier(ID3D12GraphicsCommandList* self, UINT32 barrier_groups_count, const D3D12_BARRIER_GROUP* barrier_groups) {
    Access access(self);
    original_Barrier(access.target(), barrier_groups_count, barrier_groups);
    // Enhanced layout-to-legacy conversion is an explicit subsequent gate.
}
void STDMETHODCALLTYPE hook_BeginEvent(ID3D12GraphicsCommandList* self, UINT metadata, const void* data, UINT size) {
    Access access(self);
    if (access.recording) access.recording->events.push_back({metadata, copy_values(static_cast<const uint8_t*>(data), size)});
    original_BeginEvent(access.target(), metadata, data, size);
}
void STDMETHODCALLTYPE hook_EndEvent(ID3D12GraphicsCommandList* self) {
    Access access(self);
    if (access.recording && !access.recording->events.empty()) access.recording->events.pop_back();
    original_EndEvent(access.target());
}
void STDMETHODCALLTYPE hook_BeginRenderPass(ID3D12GraphicsCommandList* self, UINT render_target_count,
    const D3D12_RENDER_PASS_RENDER_TARGET_DESC* render_targets, const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC* depth_stencil, D3D12_RENDER_PASS_FLAGS flags) {
    Access access(self); if (access.recording) access.recording->render_pass = true;
    original_BeginRenderPass(access.target(), render_target_count, render_targets, depth_stencil, flags);
}
void STDMETHODCALLTYPE hook_EndRenderPass(ID3D12GraphicsCommandList* self) {
    Access access(self); if (access.recording) access.recording->render_pass = false;
    original_EndRenderPass(access.target());
}
void STDMETHODCALLTYPE hook_BeginQuery(ID3D12GraphicsCommandList* self, ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT index) {
    Access access(self); if (access.recording) ++access.recording->active_queries;
    original_BeginQuery(access.target(), heap, type, index);
}
void STDMETHODCALLTYPE hook_EndQuery(ID3D12GraphicsCommandList* self, ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT index) {
    Access access(self); if (access.recording && access.recording->active_queries) --access.recording->active_queries;
    original_EndQuery(access.target(), heap, type, index);
}
void STDMETHODCALLTYPE hook_SetPredication(ID3D12GraphicsCommandList* self, ID3D12Resource* buffer, UINT64 aligned_buffer_offset, D3D12_PREDICATION_OP operation) {
    Access access(self);
    if (access.recording) {
        access.recording->predication = buffer != nullptr;
        auto retained = retain(buffer);
        access.recording->state.emplace_back([=](ID3D12GraphicsCommandList* target) { original_SetPredication(target, retained.Get(), aligned_buffer_offset, operation); });
    }
    original_SetPredication(access.target(), buffer, aligned_buffer_offset, operation);
}
void STDMETHODCALLTYPE hook_ExecuteBundle(ID3D12GraphicsCommandList* self, ID3D12GraphicsCommandList* command_list) {
    Access access(self);
    auto bundle = get_recording(command_list, false);
    if (access.recording && bundle) {
        std::lock_guard<std::recursive_mutex> lock(bundle->mutex);
        access.recording->state.insert(access.recording->state.end(), bundle->state.begin(), bundle->state.end());
        access.recording->indirect_state_unknown |= bundle->indirect_state_unknown;
    } else if (access.recording) access.recording->indirect_state_unknown = true;
    original_ExecuteBundle(access.target(), command_list);
}
void STDMETHODCALLTYPE hook_ExecuteIndirect(ID3D12GraphicsCommandList* self, ID3D12CommandSignature* command_signature,
    UINT max_command_count, ID3D12Resource* arg_buffer, UINT64 arg_buffer_offset, ID3D12Resource* count_buffer, UINT64 count_buffer_offset) {
    Access access(self);
    if (access.recording) access.recording->indirect_state_unknown = true;
    original_ExecuteIndirect(access.target(), command_signature, max_command_count, arg_buffer, arg_buffer_offset, count_buffer, count_buffer_offset);
}

using ExecuteFn = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
ExecuteFn original_execute = nullptr;
void complete(ID3D12CommandQueue* queue) {
    ComPtr<ID3D12Device> device; dx(queue->GetDevice(IID_PPV_ARGS(device.GetAddressOf())), "Queue device(submit drain)");
    ComPtr<ID3D12Fence> fence; dx(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())), "Queue fence(submit drain)");
    Handle event; event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event.value) throw std::runtime_error("CreateEvent(submit drain)");
    dx(queue->Signal(fence.Get(), 1), "Queue Signal(submit drain)");
    if (fence->GetCompletedValue() < 1) {
        dx(fence->SetEventOnCompletion(1, event.value), "Queue event(submit drain)");
        if (WaitForSingleObject(event.value, 30000) != WAIT_OBJECT_0) throw std::runtime_error("GPU submit boundary timed out");
    }
}
void STDMETHODCALLTYPE hook_execute(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists) {
    if (internal) { original_execute(queue, count, lists); return; }
    std::vector<std::shared_ptr<Recording>> owners;
    std::vector<ID3D12CommandList*> pending;
    InternalScope scope;
    auto flush = [&] { if (!pending.empty()) { original_execute(queue, UINT(pending.size()), pending.data()); pending.clear(); } };
    bool boundaries = false;
    try {
        for (UINT i = 0; i < count; ++i) {
            std::shared_ptr<Recording> recording;
            { std::lock_guard<std::mutex> lock(map_mutex);
              auto found = recordings.find(static_cast<ID3D12GraphicsCommandList*>(lists[i]));
              if (found != recordings.end()) recording = found->second; }
            if (!recording || recording->segments.empty()) { pending.push_back(lists[i]); continue; }
            std::lock_guard<std::recursive_mutex> lock(recording->mutex);
            if (!recording->closed) throw std::runtime_error("Logical command list submitted before Close");
            owners.push_back(recording); boundaries = true;
            for (const auto& segment : recording->segments) {
                pending.push_back(segment.list); flush();
                segment.boundary(queue);
            }
            pending.push_back(recording->current.Get());
        }
        flush();
        if (boundaries) complete(queue);
    } catch (const std::exception& failure) {
        { std::lock_guard<std::mutex> lock(error_mutex); last_error = failure.what(); }
        std::fprintf(stderr, "D4R_D3D12_SUBMIT_FAILURE %s\n", failure.what());
        try { complete(queue); } catch (...) {}
        // Do not submit a consumer suffix after a failed DLSS boundary.
    }
}
}

InternalScope::InternalScope() : previous(internal) { internal = true; }
InternalScope::~InternalScope() { internal = previous; }
std::string error() { std::lock_guard<std::mutex> lock(error_mutex); return last_error; }
size_t live_recordings() { std::lock_guard<std::mutex> lock(map_mutex); return recordings.size(); }
D3D12_RESOURCE_STATES resource_state(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES fallback) {
    auto recording = get_recording(list, false);
    if (!recording) return fallback;
    std::lock_guard<std::recursive_mutex> lock(recording->mutex);
    auto found = recording->resource_states.find(resource);
    return found == recording->resource_states.end() ? fallback : found->second;
}
void install(ID3D12Device* device) {
    std::lock_guard<std::mutex> lock(installation_mutex);
    if (installed) return;
    InternalScope scope;
    // An application may unload its NGX handle while native D3D12 objects are
    // still alive. Public method callbacks must remain callable until exit.
    HMODULE owner = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(&install), &owner)) throw std::runtime_error("Cannot retain the D3D12 callback module");
    const auto init = MH_Initialize();
    if (init != MH_OK && init != MH_ERROR_ALREADY_INITIALIZED) throw std::runtime_error(MH_StatusToString(init));
    ComPtr<ID3D12CommandAllocator> allocator; ComPtr<ID3D12GraphicsCommandList> list;
    dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf())), "Create hook allocator");
    dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(list.GetAddressOf())), "Create hook command list");
    try {
        // The public COM interface inherits one ordered method table.
        unsigned version = 0;
        ComPtr<IUnknown> newest;
        const GUID* ids[] = {&IID_ID3D12GraphicsCommandList, &IID_ID3D12GraphicsCommandList1, &IID_ID3D12GraphicsCommandList2,
            &IID_ID3D12GraphicsCommandList3, &IID_ID3D12GraphicsCommandList4, &IID_ID3D12GraphicsCommandList5,
            &IID_ID3D12GraphicsCommandList6, &IID_ID3D12GraphicsCommandList7};
        for (unsigned n = 0; n < 8; ++n) if (SUCCEEDED(list->QueryInterface(*ids[n], reinterpret_cast<void**>(newest.ReleaseAndGetAddressOf())))) version = n;
        auto table = *reinterpret_cast<void***>(list.Get());
        attach_recording_methods(table, version);
        attach_method(table[2], reinterpret_cast<void*>(hook_release), reinterpret_cast<void**>(&original_release), "Release");
        D3D12_COMMAND_QUEUE_DESC desc{}; desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue; dx(device->CreateCommandQueue(&desc, IID_PPV_ARGS(queue.GetAddressOf())), "Create hook queue");
        auto queueTable = *reinterpret_cast<void***>(queue.Get());
        // Queue inherits IUnknown, Object, DeviceChild; UpdateTileMappings,
        // CopyTileMappings precede ExecuteCommandLists in the public SDK.
        attach_method(queueTable[10], reinterpret_cast<void*>(hook_execute), reinterpret_cast<void**>(&original_execute), "ExecuteCommandLists");
        for (const auto& hook : attached) {
            const auto result = MH_QueueEnableHook(hook.target);
            if (result != MH_OK) throw std::runtime_error(MH_StatusToString(result));
        }
        const auto result = MH_ApplyQueued();
        if (result != MH_OK) throw std::runtime_error(MH_StatusToString(result));
        installed = true;
        std::printf("D4R_D3D12_COMMAND_BACKEND installed=1 interface=%u hooks=%zu frame_age=0\n", version, attached.size());
    } catch (...) {
        for (const auto& hook : attached) { (void)MH_DisableHook(hook.target); (void)MH_RemoveHook(hook.target); }
        attached.clear(); throw;
    }
}
void record_boundary(ID3D12GraphicsCommandList* list, Boundary callback) {
    if (!installed || !list || !callback) throw std::runtime_error("D3D12 command backend is not initialized");
    if (!error().empty()) throw std::runtime_error(error());
    if (list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) throw std::runtime_error("DLSS requires a direct command list");
    Access access(list);
    auto& recording = *access.recording;
    if (recording.closed || recording.render_pass || recording.active_queries || recording.predication || recording.indirect_state_unknown)
        throw std::runtime_error("Cannot split a closed list or an active render/query/predication/unknown indirect state scope");
    // Construct/replay the suffix before modifying the original list so an
    // allocation failure leaves the caller's recording intact.
    InternalScope scope;
    ComPtr<ID3D12Device> device; dx(list->GetDevice(IID_PPV_ARGS(device.GetAddressOf())), "Boundary device");
    Segment segment; segment.list = access.target(); segment.owner = recording.current;
    segment.allocator = recording.current_allocator; segment.boundary = std::move(callback);
    ComPtr<ID3D12CommandAllocator> nextAllocator; ComPtr<ID3D12GraphicsCommandList> next;
    dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(nextAllocator.GetAddressOf())), "Suffix allocator");
    dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, nextAllocator.Get(), nullptr, IID_PPV_ARGS(next.GetAddressOf())), "Suffix command list");
    for (auto& state : recording.state) state(next.Get());
    recording.segments.reserve(recording.segments.size() + 1);
    for (const auto& event : recording.events) original_BeginEvent(next.Get(), event.metadata, event.data.data(), UINT(event.data.size()));
    for (size_t n = 0; n < recording.events.size(); ++n) original_EndEvent(segment.list);
    dx(original_Close(segment.list), "Close boundary prefix");
    // Each suffix owns an independent allocator until GPU completion/reset.
    recording.segments.push_back(std::move(segment));
    recording.current_allocator = std::move(nextAllocator);
    recording.current = std::move(next);
    std::printf("D4R_D3D12_RECORD boundary=%zu state_calls=%zu frame_age=0\n", recording.segments.size(), recording.state.size());
}
}
