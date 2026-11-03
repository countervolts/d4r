#include "game_d3d12.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace d4r::dx {
namespace {
// The colour view of a typeless or depth format (the Engine's FrameTexture mapping, for the support checks).
DXGI_FORMAT viewFormat(const GameTexture& t) {
    if (t.format != DXGI_FORMAT_UNKNOWN) return t.format;
    switch (const DXGI_FORMAT f = t.resource->GetDesc().Format) {
    case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    default: return f;
    }
}
bool texture2d(const GameTexture& t, const char** reason) {
    if (!t.resource) { if (reason) *reason = "a texture is missing"; return false; }
    const auto desc = t.resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 || desc.DepthOrArraySize != 1) {
        if (reason) *reason = "a texture is not a single-sample 2D texture";
        return false;
    }
    return true;
}
bool formatSupport(ID3D12Device* device, DXGI_FORMAT format, D3D12_FORMAT_SUPPORT1 one, D3D12_FORMAT_SUPPORT2 two) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)))) return false;
    return (support.Support1 & one) == one && (support.Support2 & two) == two;
}
// the state bits a shader read needs; a state that already includes them is left alone
constexpr D3D12_RESOURCE_STATES ReadState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
constexpr D3D12_RESOURCE_STATES WriteState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
} // namespace

bool gameInputTexture(ID3D12Device* device, const GameTexture& t, const char** reason) {
    if (!texture2d(t, reason)) return false;
    if (t.resource->GetDesc().Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) {
        if (reason) *reason = "an input denies shader resource views";
        return false;
    }
    if (!formatSupport(device, viewFormat(t), D3D12_FORMAT_SUPPORT1_SHADER_LOAD, D3D12_FORMAT_SUPPORT2_NONE)) {
        if (reason) *reason = "an input format cannot be read by shaders";
        return false;
    }
    return true;
}
bool gameOutputTexture(ID3D12Device* device, const GameTexture& t, const Model& model, const char** reason) {
    if (!texture2d(t, reason)) return false;
    if (!(t.resource->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
        if (reason) *reason = "the output does not allow unordered access";
        return false;
    }
    const DXGI_FORMAT f = viewFormat(t);
    if (f == DXGI_FORMAT_R10G10B10A2_UNORM && !model.hasRgb10Output()) {
        if (reason) *reason = "an RGB10A2 output needs the packaged output_k_rgb10 shader";
        return false;
    }
    if (!formatSupport(device, f, D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW, D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)) {
        if (reason) *reason = "the output format has no typed UAV stores";
        return false;
    }
    return true;
}

struct GameUpscaler::Impl {
    Model& model;
    Shape shape;
    Engine engine;
    uint32_t maxWidth, maxHeight;
    uint32_t frames = 0;
    bool external = false;
    std::vector<D3D12_RESOURCE_BARRIER> exit;   // recordFront's: the textures' states, restored by recordBack
    Impl(Model& m, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh, const ExternalNetworkDx* network)
        : model(m), shape(makeKShape(ow, oh)), engine(m, shape, network), maxWidth(rw), maxHeight(rh),
          external(network != nullptr) {}
    // Validates the frame, sets it on the engine and returns the barriers into and out of the engine's states.
    void prepare(const GameFrame& frame, std::vector<D3D12_RESOURCE_BARRIER>& entry, std::vector<D3D12_RESOURCE_BARRIER>& leave);
};

GameUpscaler::GameUpscaler(Model& model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth, uint32_t maxRenderHeight)
    : GameUpscaler(model, outputWidth, outputHeight, maxRenderWidth, maxRenderHeight, nullptr) {}
GameUpscaler::GameUpscaler(Model& model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth, uint32_t maxRenderHeight,
                           const ExternalNetworkDx* external)
    : impl(std::make_unique<Impl>(model, outputWidth, outputHeight, maxRenderWidth, maxRenderHeight, external)) {
    if (!maxRenderWidth || !maxRenderHeight || maxRenderWidth > outputWidth || maxRenderHeight > outputHeight)
        throw std::runtime_error("render dimensions must be positive and no larger than the output");
    // Setup on the model's own queue: a discarded game list cannot lose the upload or the cleared history
    impl->model.execute([&](ID3D12GraphicsCommandList* list) {
        if (!impl->model.uploadRecorded()) impl->model.recordUpload(list);
        impl->engine.recordInitialize(list);
    });
}
GameUpscaler::~GameUpscaler() = default;

void GameUpscaler::Impl::prepare(const GameFrame& frame, std::vector<D3D12_RESOURCE_BARRIER>& entry,
                                 std::vector<D3D12_RESOURCE_BARRIER>& exit) {
    auto& g = *this;
    const uint32_t w = frame.settings.renderWidth, h = frame.settings.renderHeight;
    if (!w || !h || w > g.maxWidth || h > g.maxHeight) throw std::runtime_error("render size exceeds the adapter's inputs");
    if (frame.output.resource == frame.color.resource || frame.output.resource == frame.motion.resource ||
        frame.output.resource == frame.depth.resource)
        throw std::runtime_error("input/output texture aliasing is not supported");
    const bool gameExposure = frame.exposure.resource != nullptr;
    // every texture is read where it is, so every rectangle is direct
    auto plan = planKGameFrame(g.shape, frame.settings, g.model.hasDirectOrigins(), {true, frame.color.x, frame.color.y},
                               {true, frame.motion.x, frame.motion.y}, {true, frame.depth.x, frame.depth.y},
                               {true, frame.output.x, frame.output.y}, gameExposure, g.frames);
    // the game's states to the engine's and back (a resource listed twice, such as colour and exposure, once)
    std::vector<ID3D12Resource*> seen;
    auto require = [&](const GameTexture& t, D3D12_RESOURCE_STATES needed) {
        if (!t.resource || std::find(seen.begin(), seen.end(), t.resource) != seen.end()) return;
        seen.push_back(t.resource);
        if ((t.state & needed) == needed && (needed != WriteState || t.state == WriteState)) return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {t.resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, t.state, needed};
        entry.push_back(b);
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        exit.push_back(b);
    };
    require(frame.output, WriteState);
    for (auto* t : {&frame.color, &frame.motion, &frame.depth, &frame.exposure}) require(*t, ReadState);
    FrameTextures textures{{frame.color.resource, frame.color.format}, {frame.motion.resource, frame.motion.format},
                           {frame.depth.resource, frame.depth.format}, {frame.exposure.resource, frame.exposure.format}};
    // setFrame validates everything before a command is recorded
    g.engine.setFrame(textures, plan.params.input, plan.params.output, &plan.windows, plan.exposureOn ? &plan.exposure : nullptr,
                      FrameOutput{frame.output.resource, frame.output.format});
}
void GameUpscaler::record(ID3D12GraphicsCommandList* list, const GameFrame& frame, const FrameProfiler& profiler) {
    if (impl->external) throw std::runtime_error("this upscaler's network is external: record the front and the back");
    std::vector<D3D12_RESOURCE_BARRIER> entry, exit;
    impl->prepare(frame, entry, exit);
    if (!entry.empty()) list->ResourceBarrier(UINT(entry.size()), entry.data());
    impl->engine.recordFrame(list, profiler);
    if (!exit.empty()) list->ResourceBarrier(UINT(exit.size()), exit.data());
}
void GameUpscaler::recordFront(ID3D12GraphicsCommandList* list, const GameFrame& frame, const FrameProfiler& profiler) {
    if (!impl->external) throw std::runtime_error("this upscaler's network is internal: record the whole frame");
    std::vector<D3D12_RESOURCE_BARRIER> entry;
    impl->exit.clear();
    impl->prepare(frame, entry, impl->exit);
    if (!entry.empty()) list->ResourceBarrier(UINT(entry.size()), entry.data());
    impl->engine.recordFront(list, profiler);
}
void GameUpscaler::recordBack(ID3D12GraphicsCommandList* list, const FrameProfiler& profiler) {
    impl->engine.recordBack(list, profiler);
    if (!impl->exit.empty()) list->ResourceBarrier(UINT(impl->exit.size()), impl->exit.data());
    impl->exit.clear();
}
uint64_t GameUpscaler::networkTokenBytes(uint32_t outputWidth, uint32_t outputHeight) {
    return Engine::networkTokenBytes(makeKShape(outputWidth, outputHeight));
}
uint64_t GameUpscaler::networkHeadBytes(uint32_t outputWidth, uint32_t outputHeight) {
    return Engine::networkHeadBytes(makeKShape(outputWidth, outputHeight));
}
void GameUpscaler::submitted(ID3D12Fence* fence, uint64_t value) { impl->engine.submitted(fence, value); }
void GameUpscaler::discarded() { impl->engine.discarded(); }
bool GameUpscaler::idle() const { return impl->engine.idle(); }
} // namespace d4r::dx
