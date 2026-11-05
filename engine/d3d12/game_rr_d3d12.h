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
    // shareArena: the engine's plane arena is shareable (REngine), for an external network (setExternalNetwork).
    GameDenoiser(RModel &model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth,
                 uint32_t maxRenderHeight, bool shareArena = false);
    ~GameDenoiser(); // only once every recorded frame has retired (idle)
    GameDenoiser(const GameDenoiser &) = delete;
    GameDenoiser &operator=(const GameDenoiser &) = delete;
    // Records the frame; then exactly one of submitted or discarded. The game's
    // textures are transitioned from their given states and back; the descriptor
    // heap, root signatures and pipelines are d4r's afterwards.
    // profiler: REngine::recordFrame's boundaries.
    // With an external network set, record() records only the engine's front (REngine::recordFront) and leaves the
    // game's textures in the engine's states; recordBack (into the list that runs after the network) records the back
    // and restores them. submitted / discarded then follow the back's list.
    void record(ID3D12GraphicsCommandList *list, const GameFrameR &frame, const FrameProfiler &profiler = {});
    void recordBack(ID3D12GraphicsCommandList *list);
    // The engine's plane arena (for the external network to import) and the network itself (REngine's, before the
    // first record).
    ID3D12Resource *arena() const;
    uint64_t arenaBytes() const;
    void setExternalNetwork(const ExternalNetworkRr *external);
    bool external() const;
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
