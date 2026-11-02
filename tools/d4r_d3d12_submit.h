// d4r_d3d12_submit.h: tells the native D3D12 engine whether, and when, the game executes a command list it recorded
// into (engine/d3d12/runtime_d3d12.h: every recording is reported once, submitted with a fence value or discarded).
// NGX hands the shim an open command list and never says what becomes of it, so ID3D12CommandQueue's
// ExecuteCommandLists and ID3D12GraphicsCommandList's Reset are hooked in their vtables (shared by every queue and
// list the game's device creates): after a batch that holds a tracked list the hook signals d4r's own fence on that
// queue, and a tracked list reset before it was executed was discarded. Without the hooks (install failed) or for a
// recording no hook saw, the shim reports it submitted without a fence (the engine then keeps its resources for a
// fixed number of frames).
#pragma once
#include <d3d12.h>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace d4r_submit {
struct Pending {
    void* owner = nullptr;
    void (*submitted)(void* owner, ID3D12Fence* fence, uint64_t value, uint64_t timestampFrequency) = nullptr;
    void (*discarded)(void* owner) = nullptr;
};

namespace detail {
using Execute = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
using Reset = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*, ID3D12PipelineState*);
// vtable slots: IUnknown (3), ID3D12Object (4), ID3D12DeviceChild (1), then ID3D12CommandQueue's
// UpdateTileMappings, CopyTileMappings, ExecuteCommandLists; ID3D12CommandList's GetType, then
// ID3D12GraphicsCommandList's Close, Reset
constexpr size_t ExecuteSlot = 10, ResetSlot = 10;
struct State {
    std::mutex mutex;
    std::unordered_map<ID3D12CommandList*, std::vector<Pending>> pending;
    ID3D12Fence* fence = nullptr;
    uint64_t value = 0;
    Execute execute = nullptr;
    Reset reset = nullptr;
    std::atomic<bool> installed{false};
    // d4r_d3d12_split.h: executes a batch holding a split list itself (true), and sees every Reset
    std::atomic<bool (*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*, Execute)> splitExecute{nullptr};
    std::atomic<void (*)(ID3D12GraphicsCommandList*)> splitReset{nullptr};
};
inline State& state() { static State s; return s; }

inline bool patch(void** vtable, size_t slot, void* hook, void** original) {
    DWORD protection = 0;
    if (!VirtualProtect(&vtable[slot], sizeof(void*), PAGE_EXECUTE_READWRITE, &protection)) return false;
    if (vtable[slot] != hook) { *original = vtable[slot]; vtable[slot] = hook; }
    VirtualProtect(&vtable[slot], sizeof(void*), protection, &protection);
    FlushInstructionCache(GetCurrentProcess(), &vtable[slot], sizeof(void*));
    return true;
}

inline void STDMETHODCALLTYPE execute(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists) {
    State& s = state();
    const auto split = s.splitExecute.load();
    if (split == nullptr || !split(queue, count, lists, s.execute))
        s.execute(queue, count, lists);
    std::vector<Pending> done;
    uint64_t value = 0, frequency = 0;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.pending.empty()) return;
        for (UINT i = 0; i < count; ++i) {
            auto it = s.pending.find(lists[i]);
            if (it == s.pending.end()) continue;
            done.insert(done.end(), it->second.begin(), it->second.end());
            s.pending.erase(it);
        }
        if (done.empty()) return;
        queue->GetTimestampFrequency(&frequency);
        value = ++s.value;
        // after the batch on the same queue: reached once every list of it has executed
        if (FAILED(queue->Signal(s.fence, value))) value = 0;
    }
    for (const Pending& p : done) p.submitted(p.owner, value ? s.fence : nullptr, value, frequency);
}

inline HRESULT STDMETHODCALLTYPE reset(ID3D12GraphicsCommandList* list, ID3D12CommandAllocator* allocator,
                                       ID3D12PipelineState* pipeline) {
    State& s = state();
    std::vector<Pending> dropped;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.pending.find(list);
        if (it != s.pending.end()) { dropped = std::move(it->second); s.pending.erase(it); }
    }
    for (const Pending& p : dropped) p.discarded(p.owner);
    if (const auto split = s.splitReset.load()) split(list);
    return s.reset(list, allocator, pipeline);
}
} // namespace detail

// Hooks the vtables of the queues and lists `device` creates (once per process). False: recordings cannot be
// tracked (see the header comment).
inline bool install(ID3D12Device* device, const char** failure = nullptr) {
    using namespace detail;
    State& s = state();
    static std::mutex installMutex;
    std::lock_guard<std::mutex> lock(installMutex);
    if (s.installed.load()) return true;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    bool ok = SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.fence))) &&
              SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))) &&
              SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) &&
              SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list)));
    if (!ok && failure) *failure = "the probe queue or list could not be created";
    if (ok) {
        void* executeOriginal = nullptr; void* resetOriginal = nullptr;
        ok = patch(*reinterpret_cast<void***>(queue), ExecuteSlot, reinterpret_cast<void*>(&detail::execute), &executeOriginal) &&
             patch(*reinterpret_cast<void***>(list), ResetSlot, reinterpret_cast<void*>(&detail::reset), &resetOriginal);
        if (ok) {
            if (executeOriginal) s.execute = reinterpret_cast<Execute>(executeOriginal);
            if (resetOriginal) s.reset = reinterpret_cast<Reset>(resetOriginal);
            ok = s.execute && s.reset;
        }
        if (!ok && failure) *failure = "the vtables could not be patched";
    }
    if (list) { list->Close(); list->Release(); }
    if (allocator) allocator->Release();
    if (queue) queue->Release();
    if (!ok && s.fence) { s.fence->Release(); s.fence = nullptr; }
    s.installed = ok;
    return ok;
}
inline bool installed() { return detail::state().installed.load(); }

// The recording just made into `list`: reported through `p` when the list is executed or reset.
inline void track(ID3D12GraphicsCommandList* list, const Pending& p) {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.pending[list].push_back(p);
}
// Takes back an owner's unreported recordings (a feature being released); returns how many there were.
inline size_t forget(void* owner) {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    size_t count = 0;
    for (auto it = s.pending.begin(); it != s.pending.end();) {
        auto& v = it->second;
        for (auto p = v.begin(); p != v.end();) {
            if (p->owner == owner) { p = v.erase(p); ++count; } else ++p;
        }
        it = v.empty() ? s.pending.erase(it) : std::next(it);
    }
    return count;
}
// Whether `owner` has a recording no hook has reported yet.
inline bool pending(void* owner) {
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (auto& entry : s.pending)
        for (auto& p : entry.second)
            if (p.owner == owner) return true;
    return false;
}
} // namespace d4r_submit
