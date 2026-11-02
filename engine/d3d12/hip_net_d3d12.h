// d3dhip: the native HIP network (engine/hip_net.h) between the stages of the D3D12 engine. The engine's front
// writes the tokens and its back reads the head; both buffers are D3D12 committed resources shared with HIP as
// opaque NT handles and imported by hip_net's Windows path.
//
// Synchronization is CPU-serialized and deliberately conservative ("the safe way"):
//   recordFront(list) -> ExecuteCommandLists -> wait on a D3D12 fence on the CPU -> run() -> recordBack(list)
// run() launches the captured HIP graph and blocks until the HIP stream has finished, so the back may only be
// recorded (and submitted) after it returns. Both resources are D3D12 buffers with
// D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, created in D3D12_RESOURCE_STATE_COMMON (the all-access state for
// buffers, D3D12's counterpart of Vulkan's GENERAL) and never transitioned: HIP touches them only while the game
// queue is idle, and D3D12 touches them only before the front's fence or after run() has returned. Zero-copy
// fences/semaphores between the two queues are intentionally not attempted.
//
// The model directory needs hipnet.bin and hip/*.hsaco (compile_m.py --hip for preset M, compile_k.py --hip for
// preset K). Construction throws std::runtime_error with the reason when they are missing, when the D3D12 device is
// not the HIP device's adapter (or not AMD), or when HIP cannot import the shared resources; callers then run their
// own network and log the fallback.
#pragma once
#include <cstdint>
#include <memory>
#include <string>

struct ID3D12Device;
struct ID3D12Resource;

namespace d4r {
class HipNetwork;
}

namespace d4r::dx {
// The parts the native HIP networks share (d3d12/hip_net_rr.h's Ray Reconstruction host uses them too):
// Throws unless HIP device 0 is the D3D12 device's adapter: AMD (vendor 0x1002 on both sides), the same PCI
// vendor/device/subsystem from the PnP device tree, and one matching AMD adapter only.
void checkHipAdapter(ID3D12Device* device);
// A D3D12 buffer the native network imports as an opaque NT handle: committed, D3D12_HEAP_FLAG_SHARED,
// D3D12_RESOURCE_STATE_COMMON (the all-access buffer state, never transitioned by the network) and
// D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, rounded up to 64 KiB. The caller owns the resource (Release).
ID3D12Resource* createSharedBuffer(ID3D12Device* device, uint64_t bytes);
// Exports that buffer as an NT handle, imports it into HIP (OPAQUE_WIN32; HIP closes the handle it is given) and
// returns the device pointer of the imported memory; *memory receives the hipExternalMemory_t to destroy later.
void* importSharedBuffer(ID3D12Device* device, ID3D12Resource* resource, uint64_t bytes, void** memory);
class HipNetworkDx {
public:
    // width/height: preset M's render dimensions, preset K's output dimensions (as d4r::HipNetwork).
    // tokensBytes/headBytes: the payload each resource must hold (MEngine::networkBufferBytes for M/L,
    // Engine::networkTokenBytes/networkHeadBytes for K); the D3D12 allocation is rounded up to 64 KiB for sharing
    // and HIP imports the rounded size.
    HipNetworkDx(ID3D12Device* device, const std::string& modelDirectory, uint32_t width, uint32_t height,
                 uint64_t tokensBytes, uint64_t headBytes);
    ~HipNetworkDx();
    HipNetworkDx(const HipNetworkDx&) = delete;
    HipNetworkDx& operator=(const HipNetworkDx&) = delete;
    // The two shared resources (COMMON, ALLOW_UNORDERED_ACCESS), owned here and valid for this object's lifetime.
    ID3D12Resource* tokens() const;
    ID3D12Resource* head() const;
    // Launches the network's graph and blocks until it has finished. The queue that wrote the tokens must have
    // retired on the CPU first; the head is complete when run() returns.
    void run();
    // GPU time of the last run() (HIP events around the graph), 0 before the first run.
    float lastMilliseconds();
    // One per run(), never reset.
    uint32_t launches() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r::dx
