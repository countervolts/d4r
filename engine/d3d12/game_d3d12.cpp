#include "game_d3d12.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>
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
    // D4R_K_DIRECT_INPUTS=0: the game's colour, motion and depth, copied into the engine's own textures when the
    // frame starts, and the stages read the copies. The copies were the fix for preset K's "black speckles" in Ghost
    // of Tsushima and Cyberpunk 2077 (RX 9070 XT, driver 26.10.1): whole 16x16 groups of the history went black on
    // bright surfaces and stayed black. The reconstruction stage reads these inputs again after the network, and the
    // suspicion was that the game had changed them by then. With the reconstruction filtering in f32
    // (D4R_FILTER_F32), Ghost of Tsushima was checked clean in game without the copies (1080p -> 4K, LDR, d3dhip,
    // 2026-11-05), so by default the stages read the game's textures in place and the copies' time is saved.
    struct Staged { ID3D12Resource* source = nullptr; ComPtr<ID3D12Resource> copy; };
    Staged staged[3];
    bool direct = true;
    std::vector<D3D12_RESOURCE_BARRIER> before, after;    // this frame's copies: into the copy states and back
    std::vector<std::pair<ID3D12Resource*, ID3D12Resource*>> copies;
    Impl(Model& m, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh, const ExternalNetworkDx* network)
        : model(m), shape(makeKShape(ow, oh)), engine(m, shape, network), maxWidth(rw), maxHeight(rh),
          external(network != nullptr) {
        const char* v = std::getenv("D4R_K_DIRECT_INPUTS");
        direct = !(v && *v == '0');
    }
    // The engine-owned copy of a game texture (one per distinct resource of the frame), created or replaced to
    // match the texture's description.
    ID3D12Resource* stage(unsigned slot, const GameTexture& t);
    // Validates the frame, sets it on the engine and returns the barriers into and out of the engine's states.
    void prepare(const GameFrame& frame, std::vector<D3D12_RESOURCE_BARRIER>& entry, std::vector<D3D12_RESOURCE_BARRIER>& leave);
    // The copies prepare planned, recorded before the engine's stages.
    void recordCopies(ID3D12GraphicsCommandList* list) {
        if (!before.empty()) list->ResourceBarrier(UINT(before.size()), before.data());
        for (auto& [source, copy] : copies) list->CopyResource(copy, source);
        if (!after.empty()) list->ResourceBarrier(UINT(after.size()), after.data());
    }
};

ID3D12Resource* GameUpscaler::Impl::stage(unsigned slot, const GameTexture& t) {
    auto& s = staged[slot];
    D3D12_RESOURCE_DESC desc = t.resource->GetDesc();
    desc.Alignment = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags &= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;    // a depth format cannot be created without it
    bool fits = false;
    if (s.copy) {
        const auto have = s.copy->GetDesc();
        fits = have.Width == desc.Width && have.Height == desc.Height && have.Format == desc.Format &&
               have.MipLevels == desc.MipLevels && have.Flags == desc.Flags;
    }
    if (!fits) {
        // the engine's binding blocks hold the replaced copy until the frames that read it have retired
        ComPtr<ID3D12Device> device;
        if (FAILED(t.resource->GetDevice(IID_PPV_ARGS(&device)))) throw std::runtime_error("no device for an input copy");
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        s.copy.Reset();
        // created in the engine's read state; every frame overwrites the whole texture before a stage reads it
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, ReadState, nullptr, IID_PPV_ARGS(&s.copy))))
            throw std::runtime_error("cannot create an input copy");
        s.copy->SetName(slot == 0 ? L"d4r K colour copy" : slot == 1 ? L"d4r K motion copy" : L"d4r K depth copy");
    }
    s.source = t.resource;
    return s.copy.Get();
}

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
    auto transition = [](ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
        return b;
    };
    auto require = [&](const GameTexture& t, D3D12_RESOURCE_STATES needed) {
        if (!t.resource || std::find(seen.begin(), seen.end(), t.resource) != seen.end()) return;
        seen.push_back(t.resource);
        if ((t.state & needed) == needed && (needed != WriteState || t.state == WriteState)) return;
        entry.push_back(transition(t.resource, t.state, needed));
        exit.push_back(transition(t.resource, needed, t.state));
    };
    require(frame.output, WriteState);
    FrameTextures textures{{frame.color.resource, frame.color.format}, {frame.motion.resource, frame.motion.format},
                           {frame.depth.resource, frame.depth.format}, {frame.exposure.resource, frame.exposure.format}};
    g.before.clear(); g.after.clear(); g.copies.clear();
    if (g.direct) {
        for (auto* t : {&frame.color, &frame.motion, &frame.depth, &frame.exposure}) require(*t, ReadState);
    } else {
        // each distinct game texture is copied once, in its own state's place: into COPY_SOURCE and straight back,
        // so the game's textures need no transition around the stages
        const GameTexture* inputs[3] = {&frame.color, &frame.motion, &frame.depth};
        FrameTexture* targets[3] = {&textures.color, &textures.motion, &textures.depth};
        for (unsigned i = 0; i < 3; ++i) {
            const GameTexture& t = *inputs[i];
            ID3D12Resource* copy = nullptr;
            for (unsigned j = 0; j < i; ++j)
                if (inputs[j]->resource == t.resource) copy = targets[j]->resource;
            if (!copy) {
                copy = g.stage(i, t);
                const auto copySource = D3D12_RESOURCE_STATE_COPY_SOURCE;
                if (!(t.state & copySource)) {
                    g.before.push_back(transition(t.resource, t.state, copySource));
                    g.after.push_back(transition(t.resource, copySource, t.state));
                }
                g.before.push_back(transition(copy, ReadState, D3D12_RESOURCE_STATE_COPY_DEST));
                g.after.push_back(transition(copy, D3D12_RESOURCE_STATE_COPY_DEST, ReadState));
                g.copies.emplace_back(t.resource, copy);
            }
            targets[i]->resource = copy;
            // the game's exposure texture read where it is, unless it is one of the copied textures
            if (frame.exposure.resource == t.resource) textures.exposure.resource = copy;
        }
        if (textures.exposure.resource == frame.exposure.resource) require(frame.exposure, ReadState);
    }
    // setFrame validates everything before a command is recorded
    g.engine.setFrame(textures, plan.params.input, plan.params.output, &plan.windows, plan.exposureOn ? &plan.exposure : nullptr,
                      FrameOutput{frame.output.resource, frame.output.format});
}
void GameUpscaler::record(ID3D12GraphicsCommandList* list, const GameFrame& frame, const FrameProfiler& profiler) {
    if (impl->external) throw std::runtime_error("this upscaler's network is external: record the front and the back");
    std::vector<D3D12_RESOURCE_BARRIER> entry, exit;
    impl->prepare(frame, entry, exit);
    impl->recordCopies(list);
    if (!entry.empty()) list->ResourceBarrier(UINT(entry.size()), entry.data());
    impl->engine.recordFrame(list, profiler);
    if (!exit.empty()) list->ResourceBarrier(UINT(exit.size()), exit.data());
}
void GameUpscaler::recordFront(ID3D12GraphicsCommandList* list, const GameFrame& frame, const FrameProfiler& profiler) {
    if (!impl->external) throw std::runtime_error("this upscaler's network is internal: record the whole frame");
    std::vector<D3D12_RESOURCE_BARRIER> entry;
    impl->exit.clear();
    impl->prepare(frame, entry, impl->exit);
    impl->recordCopies(list);
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
