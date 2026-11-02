// Preset M on D3D12: the Vulkan runtime's M frame (../runtime_m.inc: parameters, bindings, dispatches, the per-frame
// window shift and the two history sets), recorded into the caller's command list on the caller's device. The
// network's matrices are AMD WMMA through AGS's reserved register space (m_wmma.hlsli); the other stages are their
// GLSL through ../hlsl_build.py. The package is compile_m.py's with its DXIL (--dxil). Binding N is register tN / uN
// (sampler sN), as in replay_m_d3d12.cpp, which runs the same stages on raw frames.
//
// Host contract (host-contract.md, as runtime_d3d12.h): a recording sets d4r's descriptor heap, root signatures and
// pipelines (the caller restores its own); every engine-owned resource ends the recording in the state it started
// in, so a recording may be discarded. Each recorded frame is reported exactly once, submitted or discarded.
#pragma once
#include "runtime_d3d12.h"

namespace d4r::dx {

class MModel {
public:
    // directory: an M or L package with its DXIL; throws std::runtime_error when files or device capabilities are missing.
    MModel(ID3D12Device* device, const std::string& directory);
    ~MModel();
    MModel(const MModel&) = delete;
    MModel& operator=(const MModel&) = delete;
    // Uploads the weights on the model's own queue and waits (once; later calls return at once).
    void upload();
    bool isLdr() const;              // NGX's LDR stages: no exposure measurement
    bool isPresetL() const;          // a preset L package (offsets.bin D4RL0002): its own input, expansion and reconstruction
    bool hasDisplayMotion() const;
    bool hasRgb10Output() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class MEngine;
};

struct MFrameSettings {
    uint32_t renderWidth = 0, renderHeight = 0;     // up to the engine's capacity
    float jitter[2] = {0, 0};
    float motionScale[2] = {1, 1};                  // (motion texel + motionOffset) * this = render pixels (output
                                                    // pixels with displayMotion)
    float motionOffset[2] = {0, 0};
    bool reset = false, depthInverted = true, displayMotion = false;
};

// The network run by something else (the native HIP layers, d3d12/hip_net_d3d12.h): ExternalNetworkDx, declared by
// runtime_d3d12.h because both engines share it. The caller owns the two resources, each
// MEngine::networkBufferBytes and shared with that network. The front stage writes the tokens, the back's expansion
// reads the head, both in NGX's plane layout ([2][tok.y][tok.x][32], the token grid of the render size rounded up to
// 32 and halved). Both resources must be in D3D12_RESOURCE_STATE_COMMON (D3D12's all-access buffer state) with
// D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS and must not be used by another queue while this engine's frames run.

class MEngine {
public:
    // Render sizes up to renderWidth x renderHeight (each frame's within that); the history is cleared by the first
    // frame's reset.
    MEngine(MModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight);
    // With an external network (the native HIP layers, d3d12/hip_net_d3d12.h) the caller owns the token and head
    // buffers: the front writes the tokens (NGX's plane layout, both buffers MEngine::networkBufferBytes) and the back
    // reads the head. The model then needs its plane-layout DXIL stages (enc0_m_planes/dec0_m_planes). recordFrame is
    // not available on such an engine.
    MEngine(MModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight,
            const ExternalNetworkDx* external);
    ~MEngine();
    // The bytes of each of an external network's two buffers (as the Vulkan MEngine's networkBufferBytes): the token
    // grid is the render size rounded up to 32 and halved, 64 bytes a token.
    static uint64_t networkBufferBytes(uint32_t renderWidth, uint32_t renderHeight);
    MEngine(const MEngine&) = delete;
    MEngine& operator=(const MEngine&) = delete;
    // The game's colour, motion and depth in NON_PIXEL_SHADER_RESOURCE at origin zero; the output in
    // UNORDERED_ACCESS with typed UAV stores (RGB10A2 through the packaged rgb10 downsample). Retained until the frames
    // using them retire.
    void setFrame(const FrameTextures& textures, const MFrameSettings& settings, FrameOutput output);
    void recordFrame(ID3D12GraphicsCommandList* list);
    // The external network's halves, in the Vulkan runtime's order (../runtime_m.inc): the front (exposure and the input
    // stage, which writes the external tokens; submit it and wait on a fence) and the back (expansion, reconstruction
    // and downsample, which read the external head; record it only after the external network has finished). Together
    // they dispatch exactly what recordFrame does. The external resources stay in COMMON and are never transitioned.
    void recordFront(ID3D12GraphicsCommandList* list);
    void recordBack(ID3D12GraphicsCommandList* list);
    void submitted(ID3D12Fence* fence, uint64_t value);
    void discarded();
    bool idle() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r::dx
