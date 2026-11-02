// Adapter between a D3D12 game's own textures and the preset K engine (runtime_d3d12.h): the D3D12 counterpart of
// ../game.h, with the same frame plan (planKGameFrame). Everything is recorded into the game's command list. D3D12
// reads every input format the engine needs through a shader view, so no input is copied or converted; the output
// is stored directly into any format with typed UAV stores.
#pragma once
#include "runtime_d3d12.h"

namespace d4r::dx {
struct GameTexture {
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; // at record time; restored after
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;       // the view format (UNKNOWN: from the resource, see FrameTexture)
    uint32_t x = 0, y = 0;                          // origin of the used rectangle
};
struct GameFrame {
    GameTexture color, depth, motion, output;
    // Optional 1x1 exposure (texel 0 is read): the game's value replaces measurement. Exclusive with settings.autoExposure.
    GameTexture exposure;
    FrameSettings settings;
};
// Whether the adapter takes a texture: a single-sample 2D texture that shaders can read (inputs) or that has typed
// UAV stores (output), with a usable view format. reason: why not.
bool gameInputTexture(ID3D12Device* device, const GameTexture& texture, const char** reason = nullptr);
bool gameOutputTexture(ID3D12Device* device, const GameTexture& texture, const Model& model, const char** reason = nullptr);

class GameUpscaler {
public:
    // The model must outlive this object. Inputs up to maxRenderWidth x maxRenderHeight. Uploads the model (once per
    // Model) and clears the history on the model's own queue, waiting for the GPU (Model::execute).
    GameUpscaler(Model& model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth, uint32_t maxRenderHeight);
    ~GameUpscaler(); // only once every recorded frame has retired (idle)
    GameUpscaler(const GameUpscaler&) = delete;
    GameUpscaler& operator=(const GameUpscaler&) = delete;
    // Records the frame; then exactly one of submitted or discarded (Engine's contract). Inputs and output are
    // transitioned from their given states and back, so the list's states are unchanged afterwards; the descriptor
    // heaps, root signature and pipeline state are d4r's (the caller restores its own, as with NGX).
    void record(ID3D12GraphicsCommandList* list, const GameFrame& frame);
    void submitted(ID3D12Fence* fence, uint64_t value);
    void discarded();
    // Every submitted frame has executed (its fence was reached) and none awaits submitted/discarded.
    bool idle() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r::dx
