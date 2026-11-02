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
    ~GameUpscalerM(); // only after idle() or a host-side GPU drain
    GameUpscalerM(const GameUpscalerM&) = delete;
    GameUpscalerM& operator=(const GameUpscalerM&) = delete;
    // Exactly one submitted/discarded notification must follow a recording.
    void record(ID3D12GraphicsCommandList* list, const MGameFrame& frame, const FrameProfiler& profiler = {});
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
