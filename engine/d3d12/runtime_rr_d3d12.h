// Native D3D12 RR D/E and preset F. D/E resources, history, reset behavior and packed model ABI follow runtime_rr.inc;
// F (D4RR0002) follows engine/runtime_rr2.inc with the stage DXIL of compile_rr_dxil.py. Caller ownership follows
// runtime_d3d12.h.
#pragma once
#include "runtime_d3d12.h"
#include <cstddef>
#include <vector>

namespace d4r::dx
{
struct RFrameImages
{
    FrameTexture color, motion, depth, normals, roughness, diffuseAlbedo, specularAlbedo;
    FrameTexture specularHitDistance, disocclusionMask, subsurfaceGuide, colorBeforeParticles;
};
// The network run by something else on its own queue (d3d12/hip_net_rr.h, the native HIP rr stages): the engine
// records only its front (parameters, reset, exposure, encoder 0) and its back (the output stage for D/E; decoder 0's
// second block, post, down and pooled for F), and hands the network's own stages to `network` in between. The queue
// handoff is CPU-serialized by the caller: submit the front and wait on its fence, run() the network, then record and
// submit the back.
struct ExternalNetworkRr
{
    // An image shared with the network as an opaque NT handle: a D3D12 buffer holding the image row-major (rowPitch
    // bytes a row, the texture copy's footprint), in D3D12_RESOURCE_STATE_COMMON and never transitioned by the engine
    // except around its own copies. Presets D and E: front[0] is encoder 0's surface 128 level 0, front[1] and front[2]
    // its surfaces 112 and 120 level 0 (the network samples them first); back[0] receives decoder 0's surface 112 (the
    // output stage's texture 448) and back[1], back[2] its surface 144 for frame parity 0 and 1. Preset F has none (its
    // network stages work in the arena the network imports).
    struct Image
    {
        ID3D12Resource *resource = nullptr;
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
        uint32_t width = 0, height = 0, rowPitch = 0;
    };
    // The network the engine hands its stages to (the engine only calls it between the front and the back).
    class Network
    {
      public:
        virtual ~Network() = default;
        // One of the network's stages, in network order: its parameter block (preset F; null for D/E, whose blocks the
        // network builds from the engine's fixed stage layout), the dispatch grid and the stage's window alignment for
        // this frame (packed x | y << 16; D/E). Called while recording the front; launches happen in run().
        virtual void stage(unsigned index, const void *block, size_t bytes, uint32_t gx, uint32_t gy, uint32_t shift) = 0;
        // Launches the stages of the recorded frame and blocks until they have finished (one HIP stream synchronize).
        virtual void run() = 0;
        // The stage indices the network runs (presets D/E: 1..10; F: 0..14 when the package has encoder 0's code object,
        // 1..14 otherwise). The engine dispatches everything else.
        virtual bool runs(unsigned index) const = 0;
    };
    Network *network = nullptr;
    Image front[3], back[3];
};
struct RFrameSettings
{
    float jitter[2] = {0, 0}, motionScale[2] = {1, 1};
    float preExposure = 1, frameTimeMs = 16.666667f;
    bool reset = false, depthInverted = false, hardwareDepth = false, roughnessPacked = false;
    float worldToView[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float viewToClip[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};
class RModel
{
  public:
    RModel(ID3D12Device *device, const std::string &directory);
    ~RModel();
    RModel(const RModel &) = delete;
    RModel &operator=(const RModel &) = delete;
    uint32_t preset() const;
    void upload();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class REngine;
};
class REngine
{
  public:
    // shareArena: create the plane arena as a buffer a native HIP network can import (d3d12/hip_net_rr.h); it then
    // rests in D3D12_RESOURCE_STATE_COMMON instead of UNORDERED_ACCESS. Pass true when setExternalNetwork will be used.
    REngine(RModel &model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight,
            bool shareArena = false);
    ~REngine();
    REngine(const REngine &) = delete;
    REngine &operator=(const REngine &) = delete;
    // Inputs in NON_PIXEL_SHADER_RESOURCE; RGBA16F output in UNORDERED_ACCESS.
    // Dimensions are fixed for this engine; recreate it on resize.
    // The network's plane arena (the network stages' tensors; a D3D12 buffer, UNORDERED_ACCESS, and while an external
    // network is set COMMON), exported to that network as an opaque NT handle. Valid until the engine is destroyed.
    ID3D12Resource *arena() const;
    uint64_t arenaBytes() const;
    // Split the frame around an external network (d3d12/hip_net_rr.h), like the Vulkan engine's setStageRecorder.
    // Call it once, before the external network is created (it imports the arena) and before the first setFrame; the
    // stages the network runs must then cover every network stage of the model, or it throws. recordFrame and
    // recordFrameForProfiling are then unavailable: record the front, run the network, record the back.
    void setExternalNetwork(const ExternalNetworkRr *external);
    const ExternalNetworkRr *externalNetwork() const;
    // The external network's halves. The front (the parameters, the reset, the exposure and encoder 0, then the three
    // skip images into the network's shared buffers) must be submitted and fenced on the CPU before the network runs;
    // the back (the network's results into the engine's images, then the output stage for D/E and decoder 0's second
    // block, post, down and pooled for F) is recorded only after the network has finished. Together they dispatch
    // what recordFrame does. The shared buffers are in COMMON apart from the engine's own copies.
    void recordFront(ID3D12GraphicsCommandList *list);
    void recordBack(ID3D12GraphicsCommandList *list);
    void setFrame(const RFrameImages &images, const RFrameSettings &settings, FrameOutput output);
    void recordFrame(ID3D12GraphicsCommandList *list, const FrameProfiler &profiler = {});
    // Diagnostic replay only: mark may submit/wait/reset this same command list
    // at stage boundaries. Descriptor heaps are rebound after each mark.
    void recordFrameForProfiling(ID3D12GraphicsCommandList *list, const FrameProfiler &profiler);
    void submitted(ID3D12Fence *fence, uint64_t value);
    void discarded();
    bool idle() const;
    struct DebugImage
    {
        std::string name;
        ID3D12Resource *resource;
        DXGI_FORMAT format;
        uint32_t width, height;
    };
    std::vector<DebugImage> debugImages() const;

  private:
    void recordFrameImpl(ID3D12GraphicsCommandList *list, const FrameProfiler &profiler, bool rebindAfterMark);
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r::dx
