#include "game_m_d3d12.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace d4r::dx {
namespace {
// The game's texture states to the engine's and back (a resource listed twice once).
void prepareFrame(MEngine& engine, std::vector<D3D12_RESOURCE_BARRIER>& entry, std::vector<D3D12_RESOURCE_BARRIER>& exit,
                  const GameFrameM& frame)
{
    if (frame.output.resource == frame.color.resource || frame.output.resource == frame.motion.resource ||
        frame.output.resource == frame.depth.resource)
        throw std::runtime_error("input/output texture aliasing is not supported");
    for (const auto* t : {&frame.color, &frame.depth, &frame.motion, &frame.output})
        if (t->x || t->y) throw std::runtime_error("preset M on D3D12 reads its textures at origin zero");
    constexpr D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    entry.clear();
    exit.clear();
    std::vector<ID3D12Resource*> seen;
    auto require = [&](const GameTexture& t, D3D12_RESOURCE_STATES needed) {
        if (!t.resource || std::find(seen.begin(), seen.end(), t.resource) != seen.end()) return;
        seen.push_back(t.resource);
        if ((t.state & needed) == needed && (needed != write || t.state == write)) return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {t.resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, t.state, needed};
        entry.push_back(b);
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        exit.push_back(b);
    };
    require(frame.output, write);
    for (const auto* t : {&frame.color, &frame.motion, &frame.depth}) require(*t, read);
    // setFrame validates everything before a command is recorded
    engine.setFrame({{frame.color.resource, frame.color.format}, {frame.motion.resource, frame.motion.format},
                     {frame.depth.resource, frame.depth.format}, {}},
                    frame.settings, FrameOutput{frame.output.resource, frame.output.format});
}
} // namespace

struct GameUpscalerM::Impl {
    MEngine engine;
    uint32_t maxWidth, maxHeight;
    std::vector<D3D12_RESOURCE_BARRIER> exit; // recordFront's exit transitions, for recordBack
    bool front = false;
    Impl(MModel& m, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh, const ExternalNetworkDx* external)
        : engine(m, rw, rh, ow, oh, external), maxWidth(rw), maxHeight(rh) {}
};

GameUpscalerM::GameUpscalerM(MModel& model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth, uint32_t maxRenderHeight)
    : impl(std::make_unique<Impl>(model, outputWidth, outputHeight, maxRenderWidth, maxRenderHeight, nullptr)) {}
GameUpscalerM::GameUpscalerM(MModel& model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth,
                             uint32_t maxRenderHeight, const ExternalNetworkDx* external)
    : impl(std::make_unique<Impl>(model, outputWidth, outputHeight, maxRenderWidth, maxRenderHeight, external)) {}
GameUpscalerM::~GameUpscalerM() = default;

void GameUpscalerM::record(ID3D12GraphicsCommandList* list, const GameFrameM& frame) {
    auto& g = *impl;
    std::vector<D3D12_RESOURCE_BARRIER> entry, exit;
    prepareFrame(g.engine, entry, exit, frame);
    if (!entry.empty()) list->ResourceBarrier(UINT(entry.size()), entry.data());
    g.engine.recordFrame(list);
    if (!exit.empty()) list->ResourceBarrier(UINT(exit.size()), exit.data());
}

void GameUpscalerM::recordFront(ID3D12GraphicsCommandList* list, const GameFrameM& frame) {
    auto& g = *impl;
    if (g.front) throw std::runtime_error("recordBack before the next recordFront");
    std::vector<D3D12_RESOURCE_BARRIER> entry;
    prepareFrame(g.engine, entry, g.exit, frame);
    if (!entry.empty()) list->ResourceBarrier(UINT(entry.size()), entry.data());
    g.engine.recordFront(list);
    g.front = true;
}

void GameUpscalerM::recordBack(ID3D12GraphicsCommandList* list) {
    auto& g = *impl;
    if (!g.front) throw std::runtime_error("recordFront before recordBack");
    g.engine.recordBack(list);
    if (!g.exit.empty()) list->ResourceBarrier(UINT(g.exit.size()), g.exit.data());
    g.exit.clear();
    g.front = false;
}
void GameUpscalerM::submitted(ID3D12Fence* fence, uint64_t value) { impl->engine.submitted(fence, value); }
void GameUpscalerM::discarded() { impl->engine.discarded(); }
bool GameUpscalerM::idle() const { return impl->engine.idle(); }
uint32_t GameUpscalerM::maxRenderWidth() const { return impl->maxWidth; }
uint32_t GameUpscalerM::maxRenderHeight() const { return impl->maxHeight; }
} // namespace d4r::dx
