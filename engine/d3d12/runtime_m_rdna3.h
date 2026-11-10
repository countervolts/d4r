// Native D3D12 M/L frames, recorded on the caller's device and command list.
#pragma once
#include "game_d3d12.h"
#include "../m_shared.h"
namespace d4r::dx::rdna3 {
class MModel {
public:
    MModel(ID3D12Device* device, const std::string& directory);
    ~MModel();
    MModel(const MModel&) = delete;
    MModel& operator=(const MModel&) = delete;
    bool isPresetL() const;
    bool isLdr() const;
    static std::string missingShaders(const std::string& directory, bool presetL);
    // d3dhip: the package has the plane-layout conversion (rdna3/m_planes.dxil) an external network needs
    bool hasExternalStages() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class GameUpscalerM;
};
struct MGameFrame {
    GameTexture color, depth, motion, output;
    d4r::MFrameSettings settings;
};
class GameUpscalerM {
public:
    GameUpscalerM(MModel& model, uint32_t outputWidth, uint32_t outputHeight,
                  uint32_t maxRenderWidth, uint32_t maxRenderHeight);
    // d3dhip: the ten Swin layers run as the native HIP network (d3d12/hip_net_d3d12.h) between recordFront and
    // recordBack. `external` holds its two shared buffers (NGX's plane layout, COMMON, never transitioned); the engine
    // converts its token-major tokens / dec1 to and from them (m_planes.hlsl). The network's token grid is fixed:
    // every frame must have this render size.
    GameUpscalerM(MModel& model, uint32_t outputWidth, uint32_t outputHeight,
                  uint32_t renderWidth, uint32_t renderHeight, const ExternalNetworkDx* external);
    ~GameUpscalerM(); // only after idle() or a host-side GPU drain
    GameUpscalerM(const GameUpscalerM&) = delete;
    GameUpscalerM& operator=(const GameUpscalerM&) = delete;
    // Exactly one submitted/discarded notification must follow a recording.
    void record(ID3D12GraphicsCommandList* list, const MGameFrame& frame, const FrameProfiler& profiler = {});
    // The external network's frame in halves: recordFront records the front (setup, exposure, the input stage, the
    // tokens into the network's buffer) with the game textures' entry transitions; recordBack the back (the network's
    // head into dec1, expansion, reconstruction, downsample) with their exit transitions. Submit the front, run the
    // network, then record the back. submitted / discarded follow once for the whole frame, as after record().
    void recordFront(ID3D12GraphicsCommandList* list, const MGameFrame& frame, const FrameProfiler& profiler = {});
    void recordBack(ID3D12GraphicsCommandList* list, const FrameProfiler& profiler = {});
    void submitted(ID3D12Fence* fence, uint64_t value);
    void discarded();
    bool idle() const;
    uint32_t maxRenderWidth() const;
    uint32_t maxRenderHeight() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r::dx::rdna3
