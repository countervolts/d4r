// Adapter between a D3D12 game's own textures and the RR D/E/F engine
// (runtime_rr_d3d12.h): the D3D12 counterpart of
// ../game_rr.h. Everything is recorded into the game's command list; the inputs
// are read where they are (origin zero) and the output is stored directly with
// typed UAV stores.
#pragma once
#include "game_d3d12.h"
#include "runtime_rr_d3d12.h"

namespace d4r::dx
{
struct GameFrameR
{
    GameTexture color, depth, motion, output, normals, roughness, diffuseAlbedo, specularAlbedo;
    GameTexture specularHitDistance, disocclusionMask, subsurfaceGuide,
        colorBeforeParticles; // as GameFrame (origins must be zero)
    RFrameSettings settings;
};
class GameDenoiser
{
  public:
    // The model must outlive this object. Render size is fixed at maxRenderWidth x
    // maxRenderHeight; uploads the model on its own queue (once per model),
    // waiting for the GPU.
    GameDenoiser(RModel &model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth,
                 uint32_t maxRenderHeight);
    ~GameDenoiser(); // only once every recorded frame has retired (idle)
    GameDenoiser(const GameDenoiser &) = delete;
    GameDenoiser &operator=(const GameDenoiser &) = delete;
    // Records the frame; then exactly one of submitted or discarded. The game's
    // textures are transitioned from their given states and back; the descriptor
    // heap, root signatures and pipelines are d4r's afterwards.
    // profiler: REngine::recordFrame's boundaries.
    void record(ID3D12GraphicsCommandList *list, const GameFrameR &frame, const FrameProfiler &profiler = {});
    void submitted(ID3D12Fence *fence, uint64_t value);
    void discarded();
    bool idle() const;
    uint32_t maxRenderWidth() const;
    uint32_t maxRenderHeight() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r::dx
