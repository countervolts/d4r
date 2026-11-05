#include "game_rr_d3d12.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
#include <wrl/client.h>

namespace d4r::dx
{
// Diagnostic capture: with D4R_RR_CAPTURE_DIR set, creating DIR\capture-now records the game's
// inputs and the engine's output for D4R_RR_CAPTURE_COUNT frames (default 8) into readback
// buffers, written as NNNN-NAME.raw (rows unpadded) and NNNN-meta.txt once the GPU is done.
// engine/rr/d3d12_capture_case.py turns them into a replay case.
namespace
{
struct CapturePlane
{
    std::string name;
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 rowBytes = 0;
};
struct CaptureFrame
{
    uint32_t index = 0;
    std::vector<CapturePlane> planes;
    std::string meta;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    uint64_t value = 0;
};
} // namespace

struct GameDenoiser::Impl
{
    REngine engine;
    uint32_t maxWidth, maxHeight;
    std::string captureDir;
    uint32_t captureLeft = 0, captureIndex = 0;
    bool captureStart = false; // the trigger was just consumed: D4R_RR_CAPTURE_RESET restarts the history here
    uint32_t frames = 0;       // the engine's frame counter (recordings less discards)
    std::vector<CaptureFrame> pending; // recorded; fence set once submitted
    // with an external network: the front's exit barriers and output capture, for recordBack
    std::vector<D3D12_RESOURCE_BARRIER> backExit;
    CaptureFrame *backCapture = nullptr;
    GameTexture backOutput;
    bool frontRecorded = false;
    Impl(RModel &m, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh, bool shareArena)
        : engine(m, rw, rh, ow, oh, shareArena), maxWidth(rw), maxHeight(rh)
    {
        captureDir = env("D4R_RR_CAPTURE_DIR");
    }
    ~Impl()
    {
        flush(true);
    }
    // The process environment (the shim sets d4r.ini's [Env] there), not the CRT's startup copy.
    static std::string env(const char *name)
    {
        char value[MAX_PATH];
        const DWORD n = GetEnvironmentVariableA(name, value, MAX_PATH);
        return n && n < MAX_PATH ? std::string(value, n) : std::string();
    }
    bool armed()
    {
        if (captureDir.empty())
            return false;
        if (captureLeft == 0)
        {
            const std::string trigger = captureDir + "\\capture-now";
            // start on the engine's frame phase 0, as a replay does
            if (frames % 8 || GetFileAttributesA(trigger.c_str()) == INVALID_FILE_ATTRIBUTES)
                return false;
            DeleteFileA(trigger.c_str());
            const std::string count = env("D4R_RR_CAPTURE_COUNT");
            captureLeft = count.empty() ? 8u : uint32_t(std::max(1, std::atoi(count.c_str())));
            captureStart = env("D4R_RR_CAPTURE_RESET") == "1";
        }
        return true;
    }
    // Copies subresource 0 of t (in state, restored afterwards) to a new readback buffer.
    static void copy(ID3D12GraphicsCommandList *list, CaptureFrame &f, const char *name, ID3D12Resource *t,
                     D3D12_RESOURCE_STATES state, DXGI_FORMAT view)
    {
        if (!t)
            return;
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        if (FAILED(t->GetDevice(IID_PPV_ARGS(&device))))
            return;
        const auto desc = t->GetDesc();
        CapturePlane p;
        p.name = name;
        UINT64 total = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &p.footprint, &p.rows, &p.rowBytes, &total);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = total;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&p.buffer))))
            return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {t, 0, state, D3D12_RESOURCE_STATE_COPY_SOURCE};
        const bool transition = state != D3D12_RESOURCE_STATE_COPY_SOURCE;
        if (transition)
            list->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = p.buffer.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = p.footprint;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = t;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        if (transition)
        {
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            list->ResourceBarrier(1, &b);
        }
        f.meta += "plane " + p.name + " " + std::to_string(int(view != DXGI_FORMAT_UNKNOWN ? view : desc.Format)) +
                  " " + std::to_string(p.footprint.Footprint.Width) + " " + std::to_string(p.rows) + " " +
                  std::to_string(p.rowBytes) + "\n";
        f.planes.push_back(std::move(p));
    }
    // Writes every capture whose fence has passed (all of them, waiting, when wait is set).
    void flush(bool wait)
    {
        for (auto it = pending.begin(); it != pending.end();)
        {
            if (!it->fence || (!wait && it->fence->GetCompletedValue() < it->value))
            {
                ++it;
                continue;
            }
            if (it->fence->GetCompletedValue() < it->value)
            {
                HANDLE event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
                it->fence->SetEventOnCompletion(it->value, event);
                WaitForSingleObject(event, 10000);
                CloseHandle(event);
            }
            char prefix[32];
            std::snprintf(prefix, sizeof(prefix), "\\%04u-", it->index);
            for (auto &p : it->planes)
            {
                void *data = nullptr;
                if (FAILED(p.buffer->Map(0, nullptr, &data)))
                    continue;
                if (FILE *out = std::fopen((captureDir + prefix + p.name + ".raw").c_str(), "wb"))
                {
                    for (UINT r = 0; r < p.rows; ++r)
                        std::fwrite(static_cast<const char *>(data) + p.footprint.Offset +
                                        UINT64(r) * p.footprint.Footprint.RowPitch,
                                    1, size_t(p.rowBytes), out);
                    std::fclose(out);
                }
                p.buffer->Unmap(0, nullptr);
            }
            if (FILE *out = std::fopen((captureDir + prefix + "meta.txt").c_str(), "w"))
            {
                std::fputs(it->meta.c_str(), out);
                std::fclose(out);
            }
            it = pending.erase(it);
        }
    }
};

GameDenoiser::GameDenoiser(RModel &model, uint32_t outputWidth, uint32_t outputHeight, uint32_t maxRenderWidth,
                           uint32_t maxRenderHeight, bool shareArena)
    : impl(std::make_unique<Impl>(model, outputWidth, outputHeight, maxRenderWidth, maxRenderHeight, shareArena))
{
}
GameDenoiser::~GameDenoiser() = default;

void GameDenoiser::record(ID3D12GraphicsCommandList *list, const GameFrameR &frame, const FrameProfiler &profiler)
{
    auto &g = *impl;
    if (g.frontRecorded)
        throw std::runtime_error("RR front recorded twice without its back");
    const GameTexture *inputs[] = {&frame.color,
                                   &frame.depth,
                                   &frame.motion,
                                   &frame.normals,
                                   &frame.roughness,
                                   &frame.diffuseAlbedo,
                                   &frame.specularAlbedo,
                                   &frame.specularHitDistance,
                                   &frame.disocclusionMask,
                                   &frame.subsurfaceGuide,
                                   &frame.colorBeforeParticles};
    for (auto *t : inputs)
    {
        if (t->resource && t->resource == frame.output.resource)
            throw std::runtime_error("RR input/output aliasing");
        if (t->x || t->y)
            throw std::runtime_error("RR D3D12 requires origin zero");
        if (t->resource)
        {
            auto d = t->resource->GetDesc();
            if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1 || d.Width != g.maxWidth ||
                d.Height != g.maxHeight)
                throw std::runtime_error("RR D3D12 input dimensions must match the render size");
        }
    }
    for (auto *a : inputs)
        for (auto *b : inputs)
            if (a->resource && a->resource == b->resource && a->state != b->state)
                throw std::runtime_error("RR aliased inputs declare conflicting resource states");
    if (frame.output.x || frame.output.y)
        throw std::runtime_error("RR D3D12 output origin must be zero");
    // the game's states to the engine's and back (a resource listed twice once)
    constexpr D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                    write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    std::vector<D3D12_RESOURCE_BARRIER> entry, exit;
    std::vector<ID3D12Resource *> seen;
    auto require = [&](const GameTexture &t, D3D12_RESOURCE_STATES needed) {
        if (!t.resource || std::find(seen.begin(), seen.end(), t.resource) != seen.end())
            return;
        seen.push_back(t.resource);
        if ((t.state & needed) == needed && (needed != write || t.state == write))
            return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {t.resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, t.state, needed};
        entry.push_back(b);
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        exit.push_back(b);
    };
    require(frame.output, write);
    for (const auto *t : inputs)
        require(*t, read);
    // setFrame validates everything before a command is recorded
    auto view = [](const GameTexture &t) { return FrameTexture{t.resource, t.format}; };
    RFrameImages images{view(frame.color),
                        view(frame.motion),
                        view(frame.depth),
                        view(frame.normals),
                        view(frame.roughness),
                        view(frame.diffuseAlbedo),
                        view(frame.specularAlbedo),
                        view(frame.specularHitDistance),
                        view(frame.disocclusionMask),
                        view(frame.subsurfaceGuide),
                        view(frame.colorBeforeParticles)};
    // a capture that starts from a reset replays bit for bit offline
    const bool capturing = g.armed();
    RFrameSettings settings = frame.settings;
    if (capturing && g.captureStart)
        settings.reset = true;
    g.captureStart = false;
    g.engine.setFrame(images, settings, {frame.output.resource, frame.output.format});
    g.flush(false);
    CaptureFrame *capture = nullptr;
    if (capturing)
    {
        capture = &g.pending.emplace_back();
        capture->index = g.captureIndex++;
        --g.captureLeft;
        const auto &st = settings;
        char line[256];
        std::snprintf(line, sizeof(line), "settings %u %u %.9g %.9g %.9g %.9g %.9g %.9g %d %d %d %d\n", g.maxWidth,
                      g.maxHeight, st.jitter[0], st.jitter[1], st.motionScale[0], st.motionScale[1], st.preExposure,
                      st.frameTimeMs, int(st.reset), int(st.depthInverted), int(st.hardwareDepth),
                      int(st.roughnessPacked));
        capture->meta = line;
        for (auto [name, m] : {std::pair{"worldToView", st.worldToView}, std::pair{"viewToClip", st.viewToClip}})
        {
            capture->meta += name;
            for (int i = 0; i < 16; ++i)
            {
                std::snprintf(line, sizeof(line), " %.9g", m[i]);
                capture->meta += line;
            }
            capture->meta += "\n";
        }
        const char *names[] = {"color",    "depth",       "motion",       "normals", "roughness", "diffuse",
                               "specular", "hitdistance", "disocclusion", "sss",     "particles"};
        static_assert(std::size(names) == std::size(inputs));
        for (size_t i = 0; i < std::size(inputs); ++i)
            Impl::copy(list, *capture, names[i], inputs[i]->resource, inputs[i]->state, inputs[i]->format);
    }
    if (!entry.empty())
        list->ResourceBarrier(UINT(entry.size()), entry.data());
    if (g.engine.externalNetwork())
    {
        g.engine.recordFront(list);
        g.backExit = std::move(exit);
        g.backCapture = capture;
        g.backOutput = frame.output;
        g.frontRecorded = true;
        return;
    }
    g.engine.recordFrame(list, profiler);
    ++g.frames;
    if (!exit.empty())
        list->ResourceBarrier(UINT(exit.size()), exit.data());
    if (capture)
        Impl::copy(list, *capture, "output", frame.output.resource, frame.output.state, frame.output.format);
}
void GameDenoiser::recordBack(ID3D12GraphicsCommandList *list)
{
    auto &g = *impl;
    if (!g.frontRecorded)
        throw std::runtime_error("RR back recorded without its front");
    g.frontRecorded = false;
    g.engine.recordBack(list);
    ++g.frames;
    if (!g.backExit.empty())
        list->ResourceBarrier(UINT(g.backExit.size()), g.backExit.data());
    if (g.backCapture)
        Impl::copy(list, *g.backCapture, "output", g.backOutput.resource, g.backOutput.state, g.backOutput.format);
    g.backExit.clear();
    g.backCapture = nullptr;
}
ID3D12Resource *GameDenoiser::arena() const
{
    return impl->engine.arena();
}
uint64_t GameDenoiser::arenaBytes() const
{
    return impl->engine.arenaBytes();
}
void GameDenoiser::setExternalNetwork(const ExternalNetworkRr *external)
{
    impl->engine.setExternalNetwork(external);
}
bool GameDenoiser::external() const
{
    return impl->engine.externalNetwork() != nullptr;
}
void GameDenoiser::submitted(ID3D12Fence *fence, uint64_t value)
{
    impl->engine.submitted(fence, value);
    for (auto &f : impl->pending)
        if (!f.fence)
            f.fence = fence, f.value = value;
}
void GameDenoiser::discarded()
{
    impl->engine.discarded();
    --impl->frames;
    impl->frontRecorded = false;
    impl->backCapture = nullptr;
    impl->pending.erase(std::remove_if(impl->pending.begin(), impl->pending.end(),
                                       [](const CaptureFrame &f) { return !f.fence; }),
                        impl->pending.end());
}
bool GameDenoiser::idle() const
{
    return impl->engine.idle();
}
uint32_t GameDenoiser::maxRenderWidth() const
{
    return impl->maxWidth;
}
uint32_t GameDenoiser::maxRenderHeight() const
{
    return impl->maxHeight;
}
} // namespace d4r::dx
