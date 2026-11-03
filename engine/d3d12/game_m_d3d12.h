// Adapter between a D3D12 game's own textures and the preset M engine (runtime_m_d3d12.h): the D3D12 counterpart of
// ../game_m.h. Everything is recorded into the game's command list; the inputs are read where they are (origin
// zero) and the output is stored directly with typed UAV stores.
#pragma once
#include "game_d3d12.h"
#include "runtime_m_d3d12.h"

namespace d4r::dx {
struct GameFrameM {
    GameTexture color, depth, motion, output;       // as GameFrame (origins must be zero)
    MFrameSettings settings;
};
class GameUpscalerM {
public:
    // The model must outlive this object. Render sizes up to maxRenderWidth x maxRenderHeight; uploads the model on
    // its own queue (once per model), waiting for the GPU.
    GameUpscalerM(MModel& model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth, uint32_t maxRenderHeight);
    // An external network between the stages (d3d12/hip_net_d3d12.h, d3dhip): the caller owns the shared token and
    // head buffers and runs the network. The model needs its plane-layout stages (compile_m.py --dxil yes) and must
    // be preset M; the engine throws the reason otherwise. record() is not available then.
    GameUpscalerM(MModel& model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth, uint32_t maxRenderHeight,
                  const ExternalNetworkDx* external);
    ~GameUpscalerM();   // only once every recorded frame has retired (idle)
    GameUpscalerM(const GameUpscalerM&) = delete;
    GameUpscalerM& operator=(const GameUpscalerM&) = delete;
    // Records the frame; then exactly one of submitted or discarded. The game's textures are transitioned from their
    // given states and back; the descriptor heap, root signatures and pipelines are d4r's afterwards.
    // profiler: MEngine's boundaries (recordFrame, recordFront and recordBack respectively).
    void record(ID3D12GraphicsCommandList* list, const GameFrameM& frame, const FrameProfiler& profiler = {});
    // The external network's frame in halves: recordFront records the front (and the game's textures' entry
    // transitions), recordBack the back (and their exit transitions). Submit the front, run the network, then record
    // the back. recordFront reports submitted/discarded like record() does, once the whole frame has run.
    void recordFront(ID3D12GraphicsCommandList* list, const GameFrameM& frame, const FrameProfiler& profiler = {});
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
} // namespace d4r::dx
