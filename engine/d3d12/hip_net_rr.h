// d3dhip for Ray Reconstruction: the model's native HIP rr/rr2 kernels as the network between the D3D12 engine's
// stages (d3d12/runtime_rr_d3d12.h's ExternalNetworkRr). Presets D and E (ten stages, MODEL/hip/rr_<stage>.hsaco with
// weights-<stage>m.bin, the same code objects the Vulkan vkhip route records) and preset F (fourteen
// MODEL/hip/rr2_<stage>.hsaco, the Vulkan vkhip F route).
//
// The network's tensors live in HIP memory: the plane arena the engine also writes is one D3D12 buffer imported as an
// opaque NT handle, everything else is hipMalloc'd. Presets D and E also exchange three skip images with the engine
// through D3D12 buffers shared the same way: the network reads the front's three (surface 128 / 112 / 120 at level 0,
// encoder 0's outputs) and writes the back's (decoder 0's surface 112 and its surface 144 for each frame parity). Those
// are row-major images; the network samples and stores them through hardware image descriptors built by HIP's own
// texture objects for a pitched 2D resource, so no shader-side format is re-derived here.
//
// Synchronization is the d3dhip way (d3d12/hip_net_d3d12.h): the caller submits the engine's front and waits on a
// D3D12 fence on the CPU, calls run() (the stages on a private HIP stream, then hipStreamSynchronize), then records
// the engine's back. Preset F can instead order the two queues on the GPU through a shared D3D12 fence (useFence(),
// launchAfter()), as preset K's HipNetworkDx does.
//
// Preset F launches its fourteen stages as one HIP graph per frame parity: the parameter blocks live in two device
// areas (one per parity, a fixed address per stage), each graph launches the stages with its area's addresses, and a
// frame's blocks are copied from pinned memory only when they differ from what that area last held (the engine's
// blocks change only with the frame parity's shift, so after two frames there is no copy). A graph is recaptured
// whenever a stage's grid or block size changes.
//
// Construction throws std::runtime_error with the reason when the model has no HIP network of this preset, the code
// objects are incomplete, the D3D12 device is not HIP device 0's adapter, or HIP cannot import the shared arena; the
// replay then keeps the model's DXIL network.
#pragma once
#include "runtime_rr_d3d12.h"
#include <cstdint>
#include <memory>
#include <string>

struct ID3D12Device;
struct ID3D12Resource;
struct ID3D12Fence;

namespace d4r::dx {
class HipNetworkRr final : public ExternalNetworkRr::Network {
public:
    // modelDirectory: an RR package with rr.txt and hip/rr_*.hsaco (presets D/E) or hip/rr2_*.hsaco (preset F).
    // presetF: the package is D4RR0002 (preset 6). renderWidth/Height and outputWidth/Height are the engine's (the
    // network's image sizes follow its level dimensions). arena: the engine's plane arena, arenaBytes of it, which the
    // engine must expose (REngine::arena): this imports it as an opaque NT handle, so it must be a shared buffer.
    HipNetworkRr(ID3D12Device* device, const std::string& modelDirectory, bool presetF, uint32_t renderWidth,
                 uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight, ID3D12Resource* arena,
                 uint64_t arenaBytes);
    ~HipNetworkRr() override;
    HipNetworkRr(const HipNetworkRr&) = delete;
    HipNetworkRr& operator=(const HipNetworkRr&) = delete;
    // The engine's half of the interface: this network and the shared images it created (presets D/E).
    const ExternalNetworkRr& external() const;
    // One of the network's stages (ExternalNetworkRr::Network).
    void stage(unsigned index, const void* block, size_t bytes, uint32_t gx, uint32_t gy, uint32_t shift) override;
    void run() override;
    bool runs(unsigned index) const override;
    // Preset F only. GPU-side ordering instead of run()'s CPU waits: `fence` (created with D3D12_FENCE_FLAG_SHARED on
    // this object's device) is imported into HIP as an external semaphore. Throws when HIP cannot import it.
    void useFence(ID3D12Fence* fence);
    // Queues the stages recorded since the last run, without blocking: a wait on the HIP stream until the fence reaches
    // `wait` (the game's queue signals it after the engine's front), the network, then a signal of `signal` (the queue
    // waits for it before the engine's back). The arena stays in COMMON; the fence orders every access.
    void launchAfter(uint64_t wait, uint64_t signal);
    // Blocks until everything launchAfter() queued has finished (lastMilliseconds() then holds the last one).
    void finish();
    // GPU time of the last run() (HIP events around the whole network), 0 before the first run.
    float lastMilliseconds();
    // One per run(), never reset.
    uint32_t frames() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    ExternalNetworkRr link;   // the engine's half (`interface` is a Windows macro)
};
} // namespace d4r::dx
