// Diagnostic per-dispatch GPU timing of AMD's denoiser (D4R_FSR_RR_DISPATCH_PROFILE=1, off by default).
//
// While ffxDispatch records into the game's list, the list's Dispatch/ExecuteIndirect entries (vkd3d-proton's
// shared command-list vtable) are wrapped to put a timestamp before every dispatch; the device's compute PSO
// creation is wrapped to tag each pipeline with vkd3d's shader hash (FNV-1 64 of the DXBC blob, the name
// VKD3D_SHADER_DUMP_PATH and VKD3D_SHADER_OVERRIDE use). Stamp i..i+1 is the time from the end of the work
// before dispatch i to the end of dispatch i (bottom-of-pipe), which is dispatch i's cost when the denoiser
// separates its passes with barriers. Wrappers forward unchanged outside the recording thread's window.
#pragma once

#include <d3d12.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace fsr_rr_dp
{
constexpr UINT kSlots = 4;
constexpr UINT kStampsPerSlot = 2048;

using DispatchFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
using ExecuteIndirectFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandSignature*, UINT,
                                                   ID3D12Resource*, UINT64, ID3D12Resource*, UINT64);
using ResourceBarrierFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, const D3D12_RESOURCE_BARRIER*);
using SetRootCbvFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, D3D12_GPU_VIRTUAL_ADDRESS);
using SetPipelineStateFn = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12PipelineState*);
using CreateComputeFn = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID,
                                                    void**);
using CreateStreamFn = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID,
                                                   void**);

struct Record
{
    uint64_t hash;
    UINT x, y, z;
    bool indirect;
    uint32_t barriers, uavBarriers; // recorded by the denoiser between the previous dispatch and this one
};

struct Stat
{
    uint64_t hash = 0;
    UINT x = 0, y = 0, z = 0;
    bool indirect = false;
    uint32_t barriers = 0, uavBarriers = 0;
    double sum = 0;
    uint32_t count = 0;
};

struct State
{
    std::mutex mutex;
    std::unordered_map<const void*, uint64_t> psoHash;
    DispatchFn dispatch = nullptr;
    ExecuteIndirectFn executeIndirect = nullptr;
    SetPipelineStateFn setPipelineState = nullptr;
    ResourceBarrierFn resourceBarrier = nullptr;
    SetRootCbvFn setRootCbv = nullptr;
    uint32_t barriersSinceDispatch = 0, uavBarriersSinceDispatch = 0;
    CreateComputeFn createCompute = nullptr;
    CreateStreamFn createStream = nullptr;
    bool listHooked = false, deviceHooked = false, failed = false;
    ID3D12QueryHeap* heap = nullptr;
    ID3D12Resource* readback = nullptr;
    volatile uint64_t* values = nullptr;
    double ticksPerMs = 0;
    std::vector<Record> records[kSlots];
    UINT used[kSlots] = {};
    std::map<UINT, Stat> stats; // by dispatch index within the frame
    uint32_t frames = 0;
    uint64_t lastHash = 0;
};

inline State& state()
{
    static State s;
    return s;
}

thread_local ID3D12GraphicsCommandList* t_list = nullptr;
thread_local UINT t_slot = 0;
thread_local uint64_t t_pso = 0;
thread_local D3D12_GPU_VIRTUAL_ADDRESS t_cbv = 0;
inline void snapshot_after_dispatch(ID3D12GraphicsCommandList* list);

inline uint64_t fnv1(const void* data, size_t size)
{
    uint64_t h = 0xcbf29ce484222325ull;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i)
        h = (h * 0x100000001b3ull) ^ p[i];
    return h;
}

inline bool patch(void** slot, void* hook, void** original)
{
    if (*slot == hook)
        return true;
    DWORD old = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old))
        return false;
    *original = *slot;
    *slot = hook;
    VirtualProtect(slot, sizeof(void*), old, &old);
    return true;
}

inline void stamp(ID3D12GraphicsCommandList* list, uint64_t hash, UINT x, UINT y, UINT z, bool indirect)
{
    State& s = state();
    UINT& used = s.used[t_slot];
    if (used + 1 >= kStampsPerSlot)
        return;
    list->EndQuery(s.heap, D3D12_QUERY_TYPE_TIMESTAMP, t_slot * kStampsPerSlot + used);
    s.records[t_slot].push_back({hash, x, y, z, indirect, s.barriersSinceDispatch, s.uavBarriersSinceDispatch});
    s.barriersSinceDispatch = s.uavBarriersSinceDispatch = 0;
    ++used;
}

static void STDMETHODCALLTYPE hook_set_pipeline_state(ID3D12GraphicsCommandList* list, ID3D12PipelineState* pso)
{
    State& s = state();
    if (t_list == list)
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto found = s.psoHash.find(pso);
        t_pso = found != s.psoHash.end() ? found->second : 0;
    }
    s.setPipelineState(list, pso);
}

inline bool serialize()
{
    static const bool on = env_uint("D4R_FSR_RR_DP_SERIALIZE", 0) != 0;
    return on;
}

static void STDMETHODCALLTYPE hook_resource_barrier(ID3D12GraphicsCommandList* list, UINT count,
                                                    const D3D12_RESOURCE_BARRIER* barriers)
{
    State& s = state();
    if (t_list == list)
        for (UINT i = 0; i < count; ++i)
        {
            ++s.barriersSinceDispatch;
            if (barriers[i].Type == D3D12_RESOURCE_BARRIER_TYPE_UAV)
                ++s.uavBarriersSinceDispatch;
        }
    s.resourceBarrier(list, count, barriers);
}

// Diagnostic: a full UAV barrier before every dispatch of the denoiser (tests for missing barriers).
inline void serialize_barrier(ID3D12GraphicsCommandList* list)
{
    if (!serialize())
        return;
    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    state().resourceBarrier(list, 1, &uav);
}

static void STDMETHODCALLTYPE hook_set_root_cbv(ID3D12GraphicsCommandList* list, UINT index,
                                                D3D12_GPU_VIRTUAL_ADDRESS va)
{
    if (t_list == list)
        t_cbv = va;
    state().setRootCbv(list, index, va);
}

static void STDMETHODCALLTYPE hook_dispatch(ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z)
{
    State& s = state();
    if (t_list == list)
    {
        serialize_barrier(list);
        stamp(list, t_pso, x, y, z, false);
    }
    s.dispatch(list, x, y, z);
    if (t_list == list)
        snapshot_after_dispatch(list);
}

static void STDMETHODCALLTYPE hook_execute_indirect(ID3D12GraphicsCommandList* list, ID3D12CommandSignature* sig,
                                                    UINT maxCount, ID3D12Resource* args, UINT64 argsOffset,
                                                    ID3D12Resource* count, UINT64 countOffset)
{
    State& s = state();
    if (t_list == list)
    {
        serialize_barrier(list);
        stamp(list, t_pso, maxCount, 0, 0, true);
    }
    s.executeIndirect(list, sig, maxCount, args, argsOffset, count, countOffset);
    if (t_list == list)
        snapshot_after_dispatch(list);
}

static HRESULT STDMETHODCALLTYPE hook_create_compute(ID3D12Device* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc,
                                                     REFIID riid, void** pso)
{
    State& s = state();
    const HRESULT hr = s.createCompute(device, desc, riid, pso);
    if (SUCCEEDED(hr) && pso != nullptr && *pso != nullptr && desc != nullptr && desc->CS.pShaderBytecode != nullptr)
    {
        const uint64_t hash = fnv1(desc->CS.pShaderBytecode, desc->CS.BytecodeLength);
        std::lock_guard<std::mutex> lock(s.mutex);
        s.psoHash[*pso] = hash;
    }
    return hr;
}

// The CS bytecode of a pipeline state stream (D3DX12-style subobjects, each aligned to a pointer).
inline const D3D12_SHADER_BYTECODE* stream_cs(const D3D12_PIPELINE_STATE_STREAM_DESC* desc)
{
    auto* p = static_cast<const uint8_t*>(desc->pPipelineStateSubobjectStream);
    const uint8_t* end = p + desc->SizeInBytes;
    while (p + sizeof(UINT) <= end)
    {
        const UINT type = *reinterpret_cast<const UINT*>(p);
        size_t size;
        switch (type)
        {
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE: size = 16; break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS:
            return reinterpret_cast<const D3D12_SHADER_BYTECODE*>(p + 8);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO: size = 24; break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS: size = 8; break;
        default: return nullptr;
        }
        p += size;
    }
    return nullptr;
}

static HRESULT STDMETHODCALLTYPE hook_create_stream(ID3D12Device* device, const D3D12_PIPELINE_STATE_STREAM_DESC* desc,
                                                    REFIID riid, void** pso)
{
    State& s = state();
    const HRESULT hr = s.createStream(device, desc, riid, pso);
    const D3D12_SHADER_BYTECODE* cs = desc != nullptr ? stream_cs(desc) : nullptr;
    if (SUCCEEDED(hr) && pso != nullptr && *pso != nullptr && cs != nullptr && cs->pShaderBytecode != nullptr)
    {
        const uint64_t hash = fnv1(cs->pShaderBytecode, cs->BytecodeLength);
        std::lock_guard<std::mutex> lock(s.mutex);
        s.psoHash[*pso] = hash;
    }
    return hr;
}

// D4R_FSR_RR_DP_SNAPSHOT=k1,k2,...: on frame D4R_FSR_RR_DP_SNAPSHOT_FRAME (default 1), copy every UAV buffer the
// denoiser created after its dispatch k (0-based), to <D4R_FSR_RR_DUMP_DIR>/snap_<k>_<buffer>.raw at teardown.
using CreateCommittedFn = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                                      const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                                      const D3D12_CLEAR_VALUE*, REFIID, void**);
using CreatePlacedFn = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Heap*, UINT64, const D3D12_RESOURCE_DESC*,
                                                   D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void**);
using CreateCommitted1Fn = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                                       const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                                       const D3D12_CLEAR_VALUE*, ID3D12ProtectedResourceSession*,
                                                       REFIID, void**);
struct Created
{
    ID3D12Resource* resource;
    D3D12_RESOURCE_DESC desc;
    const char* how;
};
inline std::vector<Created>& created()
{
    static std::vector<Created> c;
    return c;
}
thread_local bool t_creating = false;
inline CreateCommittedFn& orig_committed() { static CreateCommittedFn f = nullptr; return f; }
inline CreatePlacedFn& orig_placed() { static CreatePlacedFn f = nullptr; return f; }
inline CreateCommitted1Fn& orig_committed1() { static CreateCommitted1Fn f = nullptr; return f; }

inline void note_created(HRESULT hr, const D3D12_RESOURCE_DESC* desc, void** out, const char* how)
{
    if (!t_creating || FAILED(hr) || out == nullptr || *out == nullptr || desc == nullptr)
        return;
    // Only ID3D12Resource-compatible pointers are recorded; every resource interface starts with ID3D12Resource.
    created().push_back({static_cast<ID3D12Resource*>(*out), *desc, how});
    logf("D4R_FSR_RR_DP resource %zu (%s): dim %d %llux%ux%u format %d flags 0x%x", created().size() - 1, how,
         desc->Dimension, static_cast<unsigned long long>(desc->Width), desc->Height, desc->DepthOrArraySize,
         desc->Format, desc->Flags);
}
static HRESULT STDMETHODCALLTYPE hook_committed(ID3D12Device* d, const D3D12_HEAP_PROPERTIES* hp, D3D12_HEAP_FLAGS hf,
                                                const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES st,
                                                const D3D12_CLEAR_VALUE* cv, REFIID riid, void** out)
{
    const HRESULT hr = orig_committed()(d, hp, hf, desc, st, cv, riid, out);
    note_created(hr, desc, out, "committed");
    return hr;
}
static HRESULT STDMETHODCALLTYPE hook_placed(ID3D12Device* d, ID3D12Heap* heap, UINT64 offset,
                                             const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES st,
                                             const D3D12_CLEAR_VALUE* cv, REFIID riid, void** out)
{
    const HRESULT hr = orig_placed()(d, heap, offset, desc, st, cv, riid, out);
    note_created(hr, desc, out, "placed");
    return hr;
}
static HRESULT STDMETHODCALLTYPE hook_committed1(ID3D12Device* d, const D3D12_HEAP_PROPERTIES* hp, D3D12_HEAP_FLAGS hf,
                                                 const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES st,
                                                 const D3D12_CLEAR_VALUE* cv, ID3D12ProtectedResourceSession* ps,
                                                 REFIID riid, void** out)
{
    const HRESULT hr = orig_committed1()(d, hp, hf, desc, st, cv, ps, riid, out);
    note_created(hr, desc, out, "committed1");
    return hr;
}

struct Snapshot
{
    UINT dispatch;
    std::vector<std::pair<size_t, ID3D12Resource*>> copies; // created() index, readback buffer
};
inline std::vector<Snapshot>& snapshots()
{
    static std::vector<Snapshot> s;
    return s;
}
inline const std::vector<UINT>& snapshot_points()
{
    static const std::vector<UINT> points = [] {
        std::vector<UINT> p;
        const std::string list = env_string("D4R_FSR_RR_DP_SNAPSHOT");
        size_t at = 0;
        while (at < list.size())
        {
            const size_t comma = list.find(',', at);
            p.push_back(static_cast<UINT>(std::stoul(list.substr(at, comma - at))));
            at = comma == std::string::npos ? list.size() : comma + 1;
        }
        return p;
    }();
    return points;
}
thread_local uint32_t t_frame = 0;
thread_local UINT t_dispatchIndex = 0;
thread_local ID3D12Device* t_device = nullptr;

inline bool enabled()
{
    static const bool on = env_uint("D4R_FSR_RR_DISPATCH_PROFILE", 0) != 0;
    return on;
}

// Before the denoiser's context is created: tag the PSOs it builds.
inline void hook_device(ID3D12Device* device)
{
    State& s = state();
    if (!enabled() || s.deviceHooked)
        return;
    void** vtable = *reinterpret_cast<void***>(device);
    ID3D12Device2* device2 = nullptr;
    bool ok = patch(&vtable[11], reinterpret_cast<void*>(&hook_create_compute), reinterpret_cast<void**>(&s.createCompute));
    if (SUCCEEDED(device->QueryInterface(__uuidof(ID3D12Device2), reinterpret_cast<void**>(&device2))))
    {
        void** vtable2 = *reinterpret_cast<void***>(device2);
        ok = ok && patch(&vtable2[47], reinterpret_cast<void*>(&hook_create_stream), reinterpret_cast<void**>(&s.createStream));
        device2->Release();
    }
    if (!snapshot_points().empty())
    {
        ID3D12Device4* device4 = nullptr;
        ok = ok && patch(&vtable[27], reinterpret_cast<void*>(&hook_committed), reinterpret_cast<void**>(&orig_committed())) &&
             patch(&vtable[29], reinterpret_cast<void*>(&hook_placed), reinterpret_cast<void**>(&orig_placed()));
        if (SUCCEEDED(device->QueryInterface(__uuidof(ID3D12Device4), reinterpret_cast<void**>(&device4))))
        {
            void** vtable4 = *reinterpret_cast<void***>(device4);
            ok = ok && patch(&vtable4[53], reinterpret_cast<void*>(&hook_committed1),
                             reinterpret_cast<void**>(&orig_committed1()));
            device4->Release();
        }
    }
    s.deviceHooked = ok;
    logf("D4R_FSR_RR_DISPATCH_PROFILE: PSO creation hooks %s", ok ? "installed" : "FAILED");
}

inline bool ensure(ID3D12Device* device, ID3D12GraphicsCommandList* list)
{
    State& s = state();
    if (s.failed)
        return false;
    if (!s.listHooked)
    {
        void** vtable = *reinterpret_cast<void***>(list);
        const bool ok =
            patch(&vtable[14], reinterpret_cast<void*>(&hook_dispatch), reinterpret_cast<void**>(&s.dispatch)) &&
            patch(&vtable[25], reinterpret_cast<void*>(&hook_set_pipeline_state),
                  reinterpret_cast<void**>(&s.setPipelineState)) &&
            patch(&vtable[59], reinterpret_cast<void*>(&hook_execute_indirect),
                  reinterpret_cast<void**>(&s.executeIndirect)) &&
            patch(&vtable[26], reinterpret_cast<void*>(&hook_resource_barrier),
                  reinterpret_cast<void**>(&s.resourceBarrier)) &&
            patch(&vtable[37], reinterpret_cast<void*>(&hook_set_root_cbv), reinterpret_cast<void**>(&s.setRootCbv));
        if (!ok)
        {
            s.failed = true;
            logf("D4R_FSR_RR_DISPATCH_PROFILE: command-list hooks FAILED");
            return false;
        }
        s.listHooked = true;
    }
    if (s.heap == nullptr)
    {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ID3D12CommandQueue* queue = nullptr;
        UINT64 frequency = 0;
        if (SUCCEEDED(device->CreateCommandQueue(&queueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue))))
        {
            queue->GetTimestampFrequency(&frequency);
            queue->Release();
        }
        D3D12_QUERY_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        heapDesc.Count = kSlots * kStampsPerSlot;
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = 8ull * kSlots * kStampsPerSlot;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        void* mapped = nullptr;
        if (frequency == 0 ||
            FAILED(device->CreateQueryHeap(&heapDesc, __uuidof(ID3D12QueryHeap), reinterpret_cast<void**>(&s.heap))) ||
            FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, __uuidof(ID3D12Resource),
                                                   reinterpret_cast<void**>(&s.readback))) ||
            FAILED(s.readback->Map(0, nullptr, &mapped)))
        {
            s.failed = true;
            logf("D4R_FSR_RR_DISPATCH_PROFILE: query heap FAILED");
            return false;
        }
        s.values = static_cast<volatile uint64_t*>(mapped);
        for (UINT i = 0; i < kSlots * kStampsPerSlot; ++i)
            s.values[i] = 0;
        s.ticksPerMs = static_cast<double>(frequency) / 1000.0;
    }
    return true;
}

inline void log_stats()
{
    State& s = state();
    double total = 0;
    for (const auto& [index, stat] : s.stats)
        total += stat.sum / std::max(1u, stat.count);
    logf("D4R_FSR_RR_DISPATCH_PROFILE: %u frames, %zu dispatches/frame, %.3f ms GPU/frame (mean)", s.frames,
         s.stats.size(), total);
    for (const auto& [index, stat] : s.stats)
        logf("D4R_FSR_RR_DISPATCH %4u %016llx %s %5u %5u %5u  %.4f ms (n=%u) barriers %u uav %u", index,
             static_cast<unsigned long long>(stat.hash), stat.indirect ? "ind" : "dir", stat.x, stat.y, stat.z,
             stat.sum / std::max(1u, stat.count), stat.count, stat.barriers, stat.uavBarriers);
    std::map<uint64_t, std::pair<double, uint32_t>> byHash;
    for (const auto& [index, stat] : s.stats)
    {
        auto& entry = byHash[stat.hash];
        entry.first += stat.sum / std::max(1u, stat.count);
        ++entry.second;
    }
    std::vector<std::pair<double, uint64_t>> ranked;
    for (const auto& [hash, entry] : byHash)
        ranked.push_back({entry.first, hash});
    std::sort(ranked.rbegin(), ranked.rend());
    for (const auto& [ms, hash] : ranked)
        logf("D4R_FSR_RR_SHADER %016llx %.4f ms/frame over %u dispatches (%.1f%%)", static_cast<unsigned long long>(hash),
             ms, byHash[hash].second, total > 0 ? 100.0 * ms / total : 0.0);
}

// Around the denoiser's ffxDispatch. frame selects a slot; the slot recorded kSlots frames ago has retired.
inline void begin(ID3D12Device* device, ID3D12GraphicsCommandList* list, uint32_t frame)
{
    if (!enabled() || !ensure(device, list))
        return;
    State& s = state();
    const UINT slot = frame % kSlots;
    volatile uint64_t* values = s.values + slot * kStampsPerSlot;
    const UINT used = s.used[slot];
    if (used > 1 && values[0] != 0 && values[used - 1] >= values[0])
    {
        // Frames 1-5 carry the cold start; the stats keep warm frames only.
        if (frame > 5 + kSlots)
        {
            for (UINT i = 0; i + 1 < used; ++i)
            {
                Stat& stat = s.stats[i];
                const Record& r = s.records[slot][i];
                stat.hash = r.hash, stat.x = r.x, stat.y = r.y, stat.z = r.z, stat.indirect = r.indirect;
                stat.barriers = r.barriers, stat.uavBarriers = r.uavBarriers;
                stat.sum += static_cast<double>(values[i + 1] - values[i]) / s.ticksPerMs;
                ++stat.count;
            }
            if (++s.frames % profile_log_every() == 0)
                log_stats();
        }
    }
    for (UINT i = 0; i < kStampsPerSlot; ++i)
        values[i] = 0;
    s.records[slot].clear();
    s.used[slot] = 0;
    t_slot = slot;
    t_pso = 0;
    t_list = list;
    t_frame = frame;
    t_dispatchIndex = 0;
    t_device = device;
    s.barriersSinceDispatch = s.uavBarriersSinceDispatch = 0;
    stamp(list, 0, 0, 0, 0, false); // the start
}

inline void end(ID3D12GraphicsCommandList* list)
{
    State& s = state();
    if (t_list != list)
        return;
    t_list = nullptr;
    stamp(list, 0, 0, 0, 0, false); // after the last dispatch
    list->ResolveQueryData(s.heap, D3D12_QUERY_TYPE_TIMESTAMP, t_slot * kStampsPerSlot, s.used[t_slot], s.readback,
                           8ull * t_slot * kStampsPerSlot);
}
// D4R_FSR_RR_DUMP_DIR=<Windows dir>: also copy the denoiser's two outputs of every frame (RGBA16F, rows packed)
// to <dir>/denoised_<frame>_<diffuse|specular>.raw, for checking replacement shaders against the original.
// The copies are written once the frame has retired (4 frames later, or at teardown).
struct Dump
{
    uint32_t frame = 0;
    ID3D12Resource* buffer = nullptr;
    UINT count = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint[8] = {};
    UINT rows[8] = {};
    UINT64 rowBytes[8] = {};
    const char* names[8] = {};
    bool pending = false;
};
inline std::vector<Dump>& dumps(int set)
{
    static std::vector<Dump> d[2] = {std::vector<Dump>(8), std::vector<Dump>(8)};
    return d[set];
}

inline void dump_write(Dump& d)
{
    if (!d.pending)
        return;
    d.pending = false;
    static const std::string dir = env_string("D4R_FSR_RR_DUMP_DIR");
    void* mapped = nullptr;
    if (FAILED(d.buffer->Map(0, nullptr, &mapped)))
        return;
    for (UINT i = 0; i < d.count; ++i)
    {
        char name[512];
        std::snprintf(name, sizeof(name), "%s\\%s_%03u.raw", dir.c_str(), d.names[i], d.frame);
        FILE* f = std::fopen(name, "wb");
        if (f == nullptr)
            continue;
        const auto* base = static_cast<const uint8_t*>(mapped) + d.footprint[i].Offset;
        for (UINT row = 0; row < d.rows[i]; ++row)
            std::fwrite(base + static_cast<size_t>(row) * d.footprint[i].Footprint.RowPitch, 1, d.rowBytes[i], f);
        std::fclose(f);
    }
    d.buffer->Unmap(0, nullptr);
}

// set 0/1: independent rings (e.g. the denoiser's inputs and its outputs). Textures are in `state` and return to it.
inline void dump_record(int set, ID3D12Device* device, ID3D12GraphicsCommandList* list, uint32_t frame, UINT count,
                        ID3D12Resource* const* textures, const char* const* names, const D3D12_RESOURCE_STATES state)
{
    static const bool on = !env_string("D4R_FSR_RR_DUMP_DIR").empty();
    if (!on || count > 8)
        return;
    Dump& d = dumps(set)[frame % dumps(set).size()];
    dump_write(d); // the frame that used this slot has retired
    if (d.buffer == nullptr)
    {
        UINT64 offset = 0;
        for (UINT i = 0; i < count; ++i)
        {
            D3D12_RESOURCE_DESC desc = textures[i]->GetDesc();
            UINT64 total = 0;
            device->GetCopyableFootprints(&desc, 0, 1, offset, &d.footprint[i], &d.rows[i], &d.rowBytes[i], &total);
            offset = (offset + total + 65535) & ~65535ull;
        }
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer = {};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = offset;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&d.buffer))))
        {
            d.buffer = nullptr;
            return;
        }
    }
    d.count = count;
    D3D12_RESOURCE_BARRIER barriers[8] = {};
    for (UINT i = 0; i < count; ++i)
    {
        d.names[i] = names[i];
        barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[i].Transition.pResource = textures[i];
        barriers[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[i].Transition.StateBefore = state;
        barriers[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    list->ResourceBarrier(count, barriers);
    for (UINT i = 0; i < count; ++i)
    {
        D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
        dst.pResource = d.buffer;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = d.footprint[i];
        src.pResource = textures[i];
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    for (UINT i = 0; i < count; ++i)
        std::swap(barriers[i].Transition.StateBefore, barriers[i].Transition.StateAfter);
    list->ResourceBarrier(count, barriers);
    d.frame = frame;
    d.pending = true;
}

inline void snapshot_after_dispatch(ID3D12GraphicsCommandList* list)
{
    const UINT index = ++t_dispatchIndex; // 1-based: matches the D4R_FSR_RR_DISPATCH table (row 0 = start stamp)
    static const uint32_t frame = env_uint("D4R_FSR_RR_DP_SNAPSHOT_FRAME", 1);
    const auto& points = snapshot_points();
    if (t_frame != frame || std::find(points.begin(), points.end(), index) == points.end())
        return;
    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    state().resourceBarrier(list, 1, &uav);
    Snapshot snap{index, {}};
    // the dispatch's root CBV, read through the CPU mapping of whichever recorded buffer holds it
    for (size_t i = 0; i < created().size() && t_cbv != 0; ++i)
    {
        const Created& c = created()[i];
        if (c.desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
            continue;
        const D3D12_GPU_VIRTUAL_ADDRESS va = c.resource->GetGPUVirtualAddress();
        if (t_cbv < va || t_cbv + 1376 > va + c.desc.Width)
            continue;
        void* mapped = nullptr;
        D3D12_RANGE range = {static_cast<SIZE_T>(t_cbv - va), static_cast<SIZE_T>(t_cbv - va + 1376)};
        if (SUCCEEDED(c.resource->Map(0, &range, &mapped)))
        {
            static const std::string dir = env_string("D4R_FSR_RR_DUMP_DIR");
            char name[512];
            std::snprintf(name, sizeof(name), "%s\\snap_%03u_cbv.raw", dir.c_str(), index);
            if (FILE* f = std::fopen(name, "wb"))
            {
                std::fwrite(static_cast<const uint8_t*>(mapped) + (t_cbv - va), 1, 1376, f);
                std::fclose(f);
            }
            D3D12_RANGE none = {0, 0};
            c.resource->Unmap(0, &none);
            logf("D4R_FSR_RR_DP_SNAPSHOT: dispatch %u CBV in resource %zu at +%llu", index, i,
                 static_cast<unsigned long long>(t_cbv - va));
        }
        break;
    }
    // CPU-visible buffers (upload copies, e.g. the weights): read through their mapping
    for (size_t i = 0; i < created().size(); ++i)
    {
        const Created& c = created()[i];
        D3D12_HEAP_PROPERTIES props = {};
        D3D12_HEAP_FLAGS flags = {};
        if (c.desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || FAILED(c.resource->GetHeapProperties(&props, &flags)) ||
            props.Type != D3D12_HEAP_TYPE_UPLOAD)
            continue;
        void* mapped = nullptr;
        if (SUCCEEDED(c.resource->Map(0, nullptr, &mapped)))
        {
            static const std::string dir = env_string("D4R_FSR_RR_DUMP_DIR");
            char name[512];
            std::snprintf(name, sizeof(name), "%s\\snap_%03u_upload%02zu.raw", dir.c_str(), index, i);
            if (FILE* f = std::fopen(name, "wb"))
            {
                std::fwrite(mapped, 1, c.desc.Width, f);
                std::fclose(f);
            }
            D3D12_RANGE none = {0, 0};
            c.resource->Unmap(0, &none);
        }
    }
    if (t_cbv != 0)
        logf("D4R_FSR_RR_DP_SNAPSHOT: dispatch %u root CBV va 0x%llx", index, static_cast<unsigned long long>(t_cbv));
    for (size_t i = 0; i < created().size(); ++i)
    {
        const Created& c = created()[i];
        if (c.desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            (c.desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)
            continue;
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer = c.desc;
        buffer.Flags = D3D12_RESOURCE_FLAG_NONE;
        ID3D12Resource* readback = nullptr;
        if (FAILED(t_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&readback))))
            continue;
        D3D12_RESOURCE_BARRIER t = {};
        t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        t.Transition.pResource = c.resource;
        t.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        t.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        state().resourceBarrier(list, 1, &t);
        list->CopyBufferRegion(readback, 0, c.resource, 0, c.desc.Width);
        std::swap(t.Transition.StateBefore, t.Transition.StateAfter);
        state().resourceBarrier(list, 1, &t);
        snap.copies.push_back({i, readback});
    }
    snapshots().push_back(std::move(snap));
}

inline void snapshot_flush()
{
    static const std::string dir = env_string("D4R_FSR_RR_DUMP_DIR");
    for (Snapshot& snap : snapshots())
    {
        for (auto& [index, readback] : snap.copies)
        {
            void* mapped = nullptr;
            if (SUCCEEDED(readback->Map(0, nullptr, &mapped)))
            {
                char name[512];
                std::snprintf(name, sizeof(name), "%s\\snap_%03u_res%02zu.raw", dir.c_str(), snap.dispatch, index);
                if (FILE* f = std::fopen(name, "wb"))
                {
                    std::fwrite(mapped, 1, created()[index].desc.Width, f);
                    std::fclose(f);
                }
                readback->Unmap(0, nullptr);
            }
            readback->Release();
        }
        logf("D4R_FSR_RR_DP_SNAPSHOT: dispatch %u written (%zu buffers)", snap.dispatch, snap.copies.size());
    }
    snapshots().clear();
}

inline void dump_flush()
{
    snapshot_flush();
    for (int set = 0; set < 2; ++set)
        for (Dump& d : dumps(set))
            dump_write(d);
}
} // namespace fsr_rr_dp
