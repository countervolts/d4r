// Record-only D3D12 host for the native Windows backend (milestone 1 spike, see host-contract.md).
// The game owns the device, queues and command lists. d4r records into the caller's open list and is told, per
// recording, either the fence value that retires it (submitted) or that it never reached the GPU (discarded).
// Per-recording memory and borrowed resources are reused/released only after that fence value completes.
#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace d4r::dx {
using Microsoft::WRL::ComPtr;

enum class Status { ok, busy, pending, invalid, deviceLost };
const char* statusName(Status status);

struct FeatureDesc {
    uint32_t maxWidth = 0, maxHeight = 0;
    uint32_t frameSlots = 3;        // recordings that may be in flight at once
};
// Entry states, left unchanged on exit: color NON_PIXEL_SHADER_RESOURCE, output UNORDERED_ACCESS.
// Both R16G16B16A16_FLOAT. Color is read from (0, 0); the result goes to the output rectangle only.
struct FrameDesc {
    ID3D12Resource* color = nullptr;
    ID3D12Resource* output = nullptr;
    uint32_t x = 0, y = 0, width = 0, height = 0;
    float scale = 1;
    bool reset = false;
};
struct Ticket { uint32_t slot = UINT32_MAX; uint64_t serial = 0; };

class Backend;
class Feature {
public:
    // Records the frame. busy: every slot is still in flight. pending: the previous recording was neither
    // submitted nor discarded yet (temporal history has a single successor).
    Status record(ID3D12GraphicsCommandList* list, const FrameDesc& frame, Ticket* ticket);
    // After ExecuteCommandLists and a Signal of `value` on `fence`, on the queue that runs this feature's work.
    // One fence per feature, strictly increasing values. Commits the recording's history.
    Status submitted(const Ticket& ticket, ID3D12Fence* fence, uint64_t value);
    // The list was closed/reset without executing. History stays at the last submitted frame.
    Status discarded(const Ticket& ticket);
    size_t inFlight() const;

private:
    friend class Backend;
    struct Slot {
        enum State { free, recorded, submitted } state = free;
        uint64_t serial = 0, value = 0;
        ComPtr<ID3D12Fence> fence;
        ComPtr<ID3D12Resource> color, output;   // borrowed, retained until retired
    };
    Feature(Backend& backend, const FeatureDesc& desc);
    void retire();
    Slot* find(const Ticket& ticket, Slot::State state);

    Backend& backend;
    FeatureDesc desc;
    ComPtr<ID3D12Resource> history[2], constants;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    uint8_t* mapped = nullptr;
    std::vector<Slot> slots;
    ComPtr<ID3D12Fence> fence;
    uint64_t lastValue = 0, serial = 0;
    unsigned current = 0;
    bool historyValid = false, pending = false, released = false;
};

class Backend {
public:
    // Borrows the application's device: no device, queue or global D3D12 state of its own.
    static std::unique_ptr<Backend> create(ID3D12Device* device, std::string* reason);
    ~Backend();     // waits (bounded) for retirement of remaining submitted work, then frees everything
    Status createFeature(const FeatureDesc& desc, Feature** feature);
    void release(Feature* feature);     // safe with queued work: destroyed by collect() once it retires
    void collect();
    size_t features() const { return owned.size(); }
    size_t pendingReleases() const;
    bool lost();

private:
    friend class Feature;
    Backend() = default;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    UINT increment = 0;
    bool removed = false;
    std::vector<std::unique_ptr<Feature>> owned;
};
}
