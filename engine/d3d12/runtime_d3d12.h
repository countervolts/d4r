// Preset K on D3D12: the same frame as the Vulkan runtime (../runtime.h), recorded into the caller's command list on
// the caller's device. The shaders are the Vulkan backend's GLSL compiled to DXIL (../hlsl_build.py); the frame's
// parameters, windows, dispatch sizes and model files come from the shared ../k_frame.h. The network's cooperative
// matrices are AMD WMMA through AGS's reserved register space (coopmat_ags.hlsli): no AGS library or device.
//
// Host contract (host-contract.md): the caller owns the device, queue and command lists. A recording sets d4r's
// descriptor heap, root signatures and pipelines (the caller restores its own afterwards), and every engine-owned
// resource ends the recording in the state it started in, so a recording may be discarded. Each recorded frame is
// reported exactly once: submitted (with the fence value that retires it) or discarded (history stays at the last
// submitted frame). Descriptors and borrowed resources are kept until the frames using them have retired.
#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include "../k_frame.h"

namespace d4r::dx {
using Microsoft::WRL::ComPtr;

// One of the game's textures. Typeless and depth formats are read through their usual colour view (D32_FLOAT as
// R32_FLOAT, D24S8 as R24_UNORM_X8, ...); format overrides the view format.
struct FrameTexture {
    ID3D12Resource* resource = nullptr;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};
// The game's inputs, in NON_PIXEL_SHADER_RESOURCE (PIXEL_SHADER_RESOURCE may be included); left in that state.
// exposure: the 1x1 exposure texture, read with ExposureControl::game.
struct FrameTextures { FrameTexture color, motion, depth, exposure; };
// A borrowed output in UNORDERED_ACCESS covering outOrigin + outSize, with a format that supports typed UAV
// stores (RGBA16F; RGB10A2 uses the packaged rgb10 store); left in that state.
struct FrameOutput {
    ID3D12Resource* resource = nullptr;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};

// The network run by something else (the native HIP layers, d3d12/hip_net_d3d12.h): the caller owns the two
// resources, Engine::networkTokenBytes/networkHeadBytes (16 and 40 f16 per token, row-major), shared with that
// network. The front stage writes the tokens and the final store reads the head. Both must be in
// D3D12_RESOURCE_STATE_COMMON (D3D12's all-access buffer state) with D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS and must not be used by another
// queue while this engine's frames run.
struct ExternalNetworkDx { ID3D12Resource* tokens = nullptr; ID3D12Resource* head = nullptr; };

// Optional timestamps around the frame's stages, as the Vulkan FrameProfiler: boundary 0 at entry, 1 after setup,
// 2 after exposure, 3 after the input stage, 4 + i after layer i, 4 + KLayers after the output stage and
// KTimingStages at the end.
struct FrameProfiler {
    void (*mark)(ID3D12GraphicsCommandList* list, uint32_t boundary, void* user) = nullptr;
    void* user = nullptr;
};

class Model {
public:
    // directory: a K package with its DXIL (compile_k.py on Windows); throws std::runtime_error when the files or the
    // device's capabilities (Shader Model 6.6, 16-bit types, wave32, AMD WMMA) are missing.
    Model(ID3D12Device* device, const std::string& directory);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    // Once, before any Engine records. The upload buffer lives until the Model is destroyed.
    void recordUpload(ID3D12GraphicsCommandList* list);
    bool uploadRecorded() const;
    // Records with `commands` into a list of the Model's own, executes it on a direct queue of its own and waits for
    // it (one-off setup, such as the upload and an Engine's recordInitialize, outside the host's lists).
    void execute(const std::function<void(ID3D12GraphicsCommandList*)>& commands);
    bool hasAutoExposure() const;
    bool hasRegularDepth() const;
    bool hasDisplayMotion() const;
    bool hasGameExposure() const;
    bool hasRgb10Output() const;
    bool hasDirectOrigins() const;      // nonzero texture rectangle origins (shader revision 1 and later)
    // The package's DXIL files that are missing ("" when complete).
    static std::string missingShaders(const std::string& directory);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class Engine;
};

class Engine {
public:
    Engine(Model& model, const Shape& shape);
    // With an external network (the native HIP layers) the caller owns the token and head buffers
    // (ExternalNetworkDx) and the eleven layers run outside the engine: recordFront writes the tokens, recordBack
    // reads the head. recordFrame is not available on such an engine.
    Engine(Model& model, const Shape& shape, const ExternalNetworkDx* external);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    // The external network's buffers: the input stage's 16 f16 per token, the head's 40 f16 per token, both at the
    // token grid makeKShape derives from the output (as the Vulkan Engine's).
    static uint64_t networkTokenBytes(const Shape& shape);
    static uint64_t networkHeadBytes(const Shape& shape);
    // Once, before the first frame: clears the temporal history.
    void recordInitialize(ID3D12GraphicsCommandList* list);
    // As the Vulkan Engine::setFrame. The textures and output are retained until the frames using them retire.
    void setFrame(const FrameTextures& textures, const InputParams& input, const OutputParams& output,
                  const std::array<LayerShape, KLayers>* windows = nullptr, const ExposureControl* exposure = nullptr,
                  FrameOutput finalOutput = {});
    // Records the frame (no allocation, CPU wait, submission or readback in normal operation); then submitted
    // or discarded. Opt-in D4R_D3D12_CAPTURE_DIR snapshots allocate and read back only when capture-now exists.
    void recordFrame(ID3D12GraphicsCommandList* list, const FrameProfiler& profiler = {});
    // The external network's halves, as the Vulkan Engine::recordFront/recordBack: the front (setup, exposure and the
    // input stage, which writes the external tokens; submit it and wait on a fence) and the back (the final store,
    // which reads the external head; record and submit it only after the external network has finished). Together they
    // dispatch what recordFrame does. The external resources stay in COMMON and are never transitioned.
    // Profiler boundaries: the front's 0 at entry, 1 after setup, 2 after exposure and 3 after the input stage; the
    // back's 0 at entry, 1 after the final store and 2 at the end.
    void recordFront(ID3D12GraphicsCommandList* list, const FrameProfiler& profiler = {});
    void recordBack(ID3D12GraphicsCommandList* list, const FrameProfiler& profiler = {});
    // The last recorded frame was executed, and fence reaches value once it has run on the GPU. A null fence: the
    // host cannot tell, and the frame's resources are kept for 16 more frames.
    void submitted(ID3D12Fence* fence, uint64_t value);
    // The last recorded frame was never executed: the history stays at the frame before it.
    void discarded();
    // No recording awaits submitted/discarded and every submitted frame has retired (its fence reached; frames
    // submitted without a fence count as retired 16 frames later, so they never make an engine idle by themselves).
    bool idle() const;
    // The engine's own RGBA16F output (frames without a borrowed output), in NON_PIXEL_SHADER_RESOURCE.
    ID3D12Resource* outputImage() const;
    // The history images of the most recently recorded frame (color, luma, feature), in NON_PIXEL_SHADER_RESOURCE.
    std::array<ID3D12Resource*, 3> historyImages() const;
    // The network's head (40 f16 per token), in UNORDERED_ACCESS; headBytes of it are the frame's.
    ID3D12Resource* headBuffer() const;
    uint64_t headBytes() const;
    // The input stage's tokens (16 f16 per token; the layers reuse the buffer: see kRecordedLayers).
    ID3D12Resource* tokenBuffer() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r::dx
