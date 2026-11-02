#include "runtime_rr_d3d12.h"
#include "../rr_camera.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace d4r::dx
{
namespace
{
#include "runtime_helpers.h"
struct ExposureParams
{
    int32_t origin[2], size[2];
    float invCount, scale, key;
    uint32_t flags, blocks, blocksX;
    int32_t sampleSize[2];
    float gameScale = 1, invPre = 1;
};
uint32_t exposureGrid(uint32_t r, uint32_t o)
{
    return o / 2 <= r ? o / 2 : std::max(r / 2, 256u);
}
// d3dhip: the plane arena as a buffer the native HIP network imports as an opaque NT handle (hip_net_d3d12.h's
// createSharedBuffer has the same shape: shared heap, COMMON, 64 KiB-rounded).
ComPtr<ID3D12Resource> sharedArena(ID3D12Device *device, UINT64 bytes)
{
    D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::max<UINT64>((bytes + 65535) / 65536 * 65536, 65536);
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> resource;
    DXCHECK(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_SHARED, &desc,
                                            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource)));
    return resource;
}
bool rrPresent(const FrameTexture &t)
{
    return t.resource != nullptr;
}
#include "../rr_frame_parameters.h"
struct Slot
{
    ID3D12Resource *resource = nullptr;
    UINT64 offset = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    bool game = false;
};
std::vector<std::string> split(const std::string &s)
{
    std::vector<std::string> r;
    std::istringstream in(s);
    if (s != "-")
        for (std::string x; std::getline(in, x, ',');)
            r.push_back(x);
    return r;
}
DXGI_FORMAT surfaceFormat(const std::string &s)
{
    if (s == "rgba16f")
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (s == "rgba8_snorm")
        return DXGI_FORMAT_R8G8B8A8_SNORM;
    if (s == "r16f")
        return DXGI_FORMAT_R16_FLOAT;
    if (s == "rg16f")
        return DXGI_FORMAT_R16G16_FLOAT;
    throw std::runtime_error("unsupported RR surface " + s);
}
} // namespace
#include "runtime_rr2_d3d12.inc"

struct RModel::Impl
{
    ComPtr<ID3D12Device> device;
    uint32_t preset = 0;
    bool uploaded = false, gfx11 = false;
    std::unique_ptr<R2Model> f; // preset F, when the model is D4RR0002
    std::array<std::vector<uint32_t>, RStages> textures;
    std::array<std::vector<std::pair<uint32_t, std::string>>, RStages> surfaces;
    std::array<std::vector<Binding>, RStages> layouts;
    std::array<uint32_t, RStages> scratchWords{};
    std::array<bool, RStages> loopZero{};
    std::array<std::string, RStages> dxWeights;
    std::array<Pipeline, RStages> stages;
    std::array<ComPtr<ID3D12Resource>, RStages> weights;
    std::array<std::vector<char>, RStages> bytes;
    std::vector<uint8_t> enc0Template, hkpnTemplate;
    std::vector<Binding> exp0B{US(sizeof(ExposureParams)), T, US(4)},
        exp1B{US(sizeof(ExposureParams)), US(4), US(8), UT, UT};
    std::array<Pipeline, 2> exposure;
    std::mutex setup;
    void execute(const std::function<void(ID3D12GraphicsCommandList *)> &commands);
};
RModel::RModel(ID3D12Device *device, const std::string &directory) : impl(std::make_unique<Impl>())
{
    auto &m = *impl;
    m.device = device;
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_6};
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 options{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) ||
        sm.HighestShaderModel < D3D_SHADER_MODEL_6_6)
        throw std::runtime_error("RR requires shader model 6.6");
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options, sizeof(options))) ||
        !options.Native16BitShaderOpsSupported)
        throw std::runtime_error("RR requires native 16-bit shader operations");
    std::ifstream dx(directory + "/rr-d3d12.txt");
    std::string dxline;
    if (!std::getline(dx, dxline) || dxline != "D4RRDX01")
        throw std::runtime_error("rebuild RR model with compile_rr_dxil.py");
    std::vector<std::string> sidecar;
    while (std::getline(dx, dxline))
        sidecar.push_back(dxline);
    auto text = slurp(directory + "/rr.txt");
    std::istringstream in(std::string(text.begin(), text.end()));
    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("empty RR model");
    if (line == "D4RR0002")
    {
        m.f = std::make_unique<R2Model>();
        m.f->load(device, directory, sidecar);
        m.preset = m.f->preset;
        return;
    }
    if (line != "D4RR0001")
        throw std::runtime_error("requires an RR model (D4RR0001 or D4RR0002)");
    for (const auto &entry : sidecar)
    {
        std::istringstream f(entry);
        std::string key, name;
        uint32_t words = 0;
        f >> key >> name;
        if (key.empty())
            continue;
        if ((key != "scratch" && key != "weights" && key != "loopzero") || !f)
            throw std::runtime_error("invalid RR D3D12 manifest");
        auto at = std::find(RNames, RNames + RStages, name);
        if (at == RNames + RStages)
            throw std::runtime_error("invalid RR scratch stage");
        if (key == "weights")
        {
            f >> m.dxWeights[at - RNames];
            if (!f || std::filesystem::path(m.dxWeights[at - RNames]).filename() != m.dxWeights[at - RNames])
                throw std::runtime_error("invalid RR DXIL weight file");
        }
        else if (key == "loopzero")
        {
            f >> words;
            const auto index = at - RNames;
            if (!f || words != RParamBytes[index] || m.loopZero[index])
                throw std::runtime_error("invalid RR loop-zero parameter ABI");
            m.loopZero[index] = true;
        }
        else
        {
            f >> words;
            if (!f || !words || words > 16384)
                throw std::runtime_error("invalid RR scratch size");
            m.scratchWords[at - RNames] = words;
        }
    }
    unsigned stage = 0;
    while (std::getline(in, line))
    {
        std::istringstream f(line);
        std::string key;
        f >> key;
        if (key == "preset")
            f >> m.preset;
        else if (key == "arithmetic")
            throw std::runtime_error("D3D12 RR currently requires gfx12 FP8 weights/arithmetic");
        else if (key == "stage")
        {
            std::string name, shader, weight, ts, ss;
            unsigned wave;
            f >> name >> shader >> wave >> weight >> ts >> ss;
            if (stage >= RStages || name != RNames[stage] || ss.empty())
                throw std::runtime_error("invalid RR stage: " + line);
            auto &b = m.layouts[stage];
            b = {R, R, UR};
            for (const auto &t : split(ts))
            {
                m.textures[stage].push_back(uint32_t(std::stoul(t)));
                b.push_back(TL);
            }
            // Only inputs marked linear by the Vulkan resolver use filtering.
            for (size_t i = 0; i < m.textures[stage].size(); ++i)
            {
                uint32_t off = m.textures[stage][i];
                bool linear = false;
                if (stage == REnc0)
                    linear = off == 160 || off == 224 || off == 488 || off == 496;
                else if (stage <= RDec0)
                    linear = off == 88;
                else
                    linear = off == 176 || off == 216 || off == 392;
                b[3 + i] = linear ? TL : T;
            }
            for (const auto &s : split(ss))
            {
                auto p = s.find(':');
                if (p == std::string::npos)
                    throw std::runtime_error("bad RR surface");
                m.surfaces[stage].emplace_back(uint32_t(std::stoul(s.substr(0, p))), s.substr(p + 1));
                b.push_back(UT);
            }
            if (m.scratchWords[stage])
            {
                b.resize(41);
                b[40] = US(4);
            }
            auto path = std::filesystem::path(shader);
            path.replace_extension(".dxil");
            m.stages[stage].create(device, directory + "/" + path.string(), b);
            m.bytes[stage] = slurp(directory + "/" + (m.dxWeights[stage].empty() ? weight : m.dxWeights[stage]));
            m.weights[stage] =
                buffer(device, m.bytes[stage].size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
            ++stage;
        }
        else if (key == "template")
        {
            std::string which, file;
            f >> which >> file;
            auto bytes = slurp(directory + "/" + file);
            if (which != "enc0" && which != "hkpn")
                throw std::runtime_error("unknown RR template");
            auto &dst = which == "enc0" ? m.enc0Template : m.hkpnTemplate;
            if (bytes.size() != RParamBytes[which == "enc0" ? REnc0 : RHkpn])
                throw std::runtime_error("invalid RR template size");
            dst.assign(bytes.begin(), bytes.end());
        }
    }
    if (stage != RStages || (m.preset != 4 && m.preset != 5) || m.enc0Template.empty() || m.hkpnTemplate.empty())
        throw std::runtime_error("incomplete RR model");
    m.exposure[0].create(device, directory + "/exposure_rr0.dxil", m.exp0B);
    m.exposure[1].create(device, directory + "/exposure_rr1.dxil", m.exp1B);
}
RModel::~RModel() = default;
uint32_t RModel::preset() const
{
    return impl->preset;
}
void RModel::upload()
{
    auto &m = *impl;
    std::lock_guard<std::mutex> lock(m.setup);
    if (m.uploaded)
        return;
    if (m.f)
    {
        auto &f = *m.f;
        struct Upload
        {
            ComPtr<ID3D12Resource> source;
            ID3D12Resource *target;
            UINT64 bytes;
            D3D12_RESOURCE_STATES after;
        };
        std::vector<Upload> copies;
        auto add = [&](ComPtr<ID3D12Resource> &target, const void *data, size_t bytes, D3D12_RESOURCE_STATES after) {
            auto source =
                buffer(m.device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            void *mapped = nullptr;
            DXCHECK(source->Map(0, nullptr, &mapped));
            std::memcpy(mapped, data, bytes);
            source->Unmap(0, nullptr);
            copies.push_back({std::move(source), target.Get(), UINT64(bytes), after});
        };
        for (auto &st : f.stages)
            if (st.weights)
                add(st.weights, st.pending.data(), st.pending.size(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        add(f.aux, f.auxPending.data(), f.auxPending.size(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m.execute([&](ID3D12GraphicsCommandList *list) {
            for (auto &copy : copies)
            {
                list->CopyBufferRegion(copy.target, 0, copy.source.Get(), 0, copy.bytes);
                auto b = transition(copy.target, D3D12_RESOURCE_STATE_COPY_DEST, copy.after);
                list->ResourceBarrier(1, &b);
            }
        });
        for (auto &st : f.stages)
            st.pending.clear();
        f.auxPending.clear();
        m.uploaded = true;
        return;
    }
    std::array<ComPtr<ID3D12Resource>, RStages> uploads;
    for (unsigned i = 0; i < RStages; ++i)
    {
        uploads[i] =
            buffer(m.device.Get(), m.bytes[i].size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void *data = nullptr;
        DXCHECK(uploads[i]->Map(0, nullptr, &data));
        std::memcpy(data, m.bytes[i].data(), m.bytes[i].size());
        uploads[i]->Unmap(0, nullptr);
    }
    m.execute([&](ID3D12GraphicsCommandList *list) {
        for (unsigned i = 0; i < RStages; ++i)
        {
            list->CopyBufferRegion(m.weights[i].Get(), 0, uploads[i].Get(), 0, m.bytes[i].size());
            auto b = transition(m.weights[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &b);
        }
    });
    for (auto &b : m.bytes)
        b.clear();
    m.uploaded = true;
}

struct REngine::Impl
{
    RModel::Impl &model;
    ID3D12Device *device;
    std::unique_ptr<R2Engine> f; // preset F
    uint32_t rw, rh, ow, oh, w0, h0, frame = 0;
    bool reset = true, frameSet = false;
    std::array<ComPtr<ID3D12Resource>, RStages> params;
    std::array<ComPtr<ID3D12Resource>, RStages> scratch;
    std::array<std::vector<uint8_t>, RStages> blocks;
    ComPtr<ID3D12Resource> arena, expP, expSums, expState, ags, zero;
    std::array<ComPtr<ID3D12Resource>, 6> s112, s120;
    std::array<ComPtr<ID3D12Resource>, 5> s128, d112, d120;
    std::array<ComPtr<ID3D12Resource>, 2> q, r, h, e, pb;
    std::map<std::string, ComPtr<ID3D12Resource>> spare;
    ExposureParams exposure{};
    float prevJitter[2] = {}, savedJitter[2] = {};
    rr_camera::Matrix previousView{}, previousClip{}, savedView{}, savedClip{};
    std::unordered_map<ID3D12Resource *, D3D12_RESOURCE_STATES> rest, now;
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    ComPtr<ID3D12DescriptorHeap> heap, clearHeap;
    UINT increment = 0;
    static constexpr UINT Blocks = 16, BlockSlots = 512, ClearBase = Blocks * BlockSlots;
    struct Bindings
    {
        RFrameImages images{};
        FrameOutput output{};
        std::vector<ComPtr<ID3D12Resource>> retained;
        ComPtr<ID3D12Fence> fence;
        uint64_t value = 0;
        bool pending = false, inFlight = false;
        std::array<std::array<UINT, 2>, RStages> stages{};
        UINT exp0 = 0, exp1 = 0;
    };
    std::vector<Bindings> bindings;
    size_t current = 0, recorded = SIZE_MAX;
    std::map<UINT, std::vector<Slot>> tables;
    // an external network (the native HIP stages): the front is recorded, the network runs, then the back
    const ExternalNetworkRr *external = nullptr;
    bool frontDone = false;
    explicit Impl(RModel::Impl &m) : model(m), device(m.device.Get())
    {
    }
    uint32_t lw(int l) const
    {
        return w0 >> l;
    }
    uint32_t lh(int l) const
    {
        return h0 >> l;
    }
    uint64_t arenaOffset(int units) const
    {
        return uint64_t(units) * (uint64_t(w0) * h0 / 32);
    }
    ComPtr<ID3D12Resource> buf(UINT64 n)
    {
        auto x = buffer(device, n, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        rest[x.Get()] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        return x;
    }
    ComPtr<ID3D12Resource> tex(UINT w, UINT h, DXGI_FORMAT f)
    {
        auto x = texture(device, w, h, f);
        rest[x.Get()] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        return x;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT i) const
    {
        auto x = heap->GetCPUDescriptorHandleForHeapStart();
        x.ptr += SIZE_T(i) * increment;
        return x;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE clearCpu(UINT i) const
    {
        auto x = clearHeap->GetCPUDescriptorHandleForHeapStart();
        x.ptr += SIZE_T(i) * increment;
        return x;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT i) const
    {
        auto x = heap->GetGPUDescriptorHandleForHeapStart();
        x.ptr += UINT64(i) * increment;
        return x;
    }
    void write(UINT first, const std::vector<Binding> &kinds, const std::vector<Slot> &slots);
    void use(ID3D12Resource *r, D3D12_RESOURCE_STATES s)
    {
        auto &old = now.at(r);
        if (old != s)
        {
            barriers.push_back(transition(r, old, s));
            old = s;
        }
    }
    void flush(ID3D12GraphicsCommandList *list)
    {
        if (!barriers.empty())
            list->ResourceBarrier(UINT(barriers.size()), barriers.data());
        barriers.clear();
    }
    void dispatch(ID3D12GraphicsCommandList *list, const Pipeline &p, UINT first, UINT x, UINT y)
    {
        auto &slots = tables.at(first);
        auto &kinds = *p.bindings;
        for (size_t i = 0; i < slots.size(); ++i)
            if (slots[i].resource && !slots[i].game)
                use(slots[i].resource, kinds[i].kind == Kind::Raw || kinds[i].kind == Kind::Tex
                                           ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                           : D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        flush(list);
        list->SetComputeRootSignature(p.root.Get());
        list->SetPipelineState(p.state.Get());
        list->SetComputeRootDescriptorTable(0, gpu(first));
        list->SetComputeRootUnorderedAccessView(1, ags->GetGPUVirtualAddress());
        list->Dispatch(x, y, 1);
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barriers.push_back(b);
    }
    Slot resolve(unsigned stage, bool surface, uint32_t offset, uint32_t par, const RFrameImages &in,
                 ID3D12Resource *output) const;
    void bind(Bindings &b, UINT first);
    // the frame's parameter blocks, the history's reset and the exposure into the recorded list
    void uploadParameters(ID3D12GraphicsCommandList *list);
    // an engine image to and from one of the external network's shared buffers (row-major, its row pitch)
    void copyToShared(ID3D12GraphicsCommandList *list, ID3D12Resource *source, const ExternalNetworkRr::Image &target);
    void copyFromShared(ID3D12GraphicsCommandList *list, const ExternalNetworkRr::Image &source, ID3D12Resource *target);
};

REngine::REngine(RModel &model, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh, bool shareArena)
    : impl(std::make_unique<Impl>(*model.impl))
{
    if (rw < 64 || rh < 64 || rw > ow || rh > oh || ow > 16384 || oh > 16384)
        throw std::runtime_error("unsupported RR dimensions");
    model.upload();
    auto &e = *impl;
    e.rw = rw;
    e.rh = rh;
    e.ow = ow;
    e.oh = oh;
    e.w0 = (rw + 31) & ~31u;
    e.h0 = (rh + 31) & ~31u;
    if (e.model.f)
    {
        e.f = std::make_unique<R2Engine>(*e.model.f, e.device, rw, rh, ow, oh, shareArena);
        return;
    }
    // the arena is the external network's too (d3dhip): then it must be shareable, so it starts and rests in COMMON
    e.arena = shareArena ? sharedArena(e.device, e.arenaOffset(RArenaUnits)) : e.buf(e.arenaOffset(RArenaUnits));
    if (shareArena)
        e.rest[e.arena.Get()] = D3D12_RESOURCE_STATE_COMMON;
    e.expP = e.buf(sizeof(ExposureParams));
    e.expState = e.buf(8);
    e.ags = e.buf(256);
    uint32_t gx = exposureGrid(rw, ow), gy = exposureGrid(rh, oh);
    e.exposure.blocksX = (gx + 31) / 32;
    e.exposure.blocks = e.exposure.blocksX * ((gy + 31) / 32);
    e.expSums = e.buf(e.exposure.blocks * 4);
    for (unsigned i = 0; i < RStages; ++i)
    {
        const uint32_t bytes = RParamBytes[i] + (e.model.loopZero[i] ? 4 : 0);
        e.params[i] = e.buf(bytes);
        e.blocks[i].resize(bytes);
    }
    for (unsigned i = 0; i < RStages; ++i)
        if (e.model.scratchWords[i])
        {
            int l = RLevel[i];
            e.scratch[i] = e.buf(uint64_t(e.model.scratchWords[i]) * 4 * ((e.lw(l) + 13) / 8) * ((e.lh(l) + 13) / 8));
        }
    auto half4 = DXGI_FORMAT_R16G16B16A16_FLOAT;
    for (int l = 0; l < 6; ++l)
    {
        e.s112[l] = e.tex(e.lw(l), e.lh(l), half4);
        e.s120[l] = e.tex(e.lw(l), e.lh(l), half4);
    }
    for (int l = 0; l < 5; ++l)
    {
        e.s128[l] = e.tex(e.lw(l + 1), e.lh(l + 1), half4);
        e.d112[l] = e.tex(e.lw(l), e.lh(l), half4);
        e.d120[l] = e.tex(e.lw(l), e.lh(l), half4);
    }
    for (unsigned i = 0; i < 2; ++i)
    {
        e.q[i] = e.tex(rw, rh, half4);
        e.r[i] = e.tex(rw, rh, DXGI_FORMAT_R8G8B8A8_SNORM);
        e.h[i] = e.tex(ow, oh, half4);
        e.e[i] = e.tex(1, 1, DXGI_FORMAT_R16_FLOAT);
        e.pb[i] = e.tex(ow, oh, half4);
    }
    e.zero = e.tex(1, 1, half4);
    for (auto &stage : e.model.surfaces)
        for (auto &[off, fmt] : stage)
            if (!e.spare.count(fmt))
                e.spare[fmt] = e.tex(1, 1, surfaceFormat(fmt));
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, Impl::ClearBase + 2,
                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    DXCHECK(e.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&e.heap)));
    e.increment = e.device->GetDescriptorHandleIncrementSize(hd.Type);
    e.write(Impl::ClearBase, {UT, UT}, {{e.q[0].Get()}, {e.q[1].Get()}});
    hd.NumDescriptors = 2;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    DXCHECK(e.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&e.clearHeap)));
    for (unsigned i = 0; i < 2; ++i)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
        v.Format = e.q[i]->GetDesc().Format;
        v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        e.device->CreateUnorderedAccessView(e.q[i].Get(), nullptr, &v, e.clearCpu(i));
    }
    for (unsigned stage = REnc1; stage <= RDec0; ++stage)
    {
        auto &p = e.blocks[stage];
        int l = RLevel[stage];
        put32(p, 0, e.lw(l));
        put32(p, 4, e.lh(l));
        int offsets[] = {8, 24, 48, 56};
        for (int k = 0; k < 4; ++k)
            if (RArena[stage][k] >= 0)
                put64(p, offsets[k], e.arenaOffset(RArena[stage][k]));
        putF(p, 136, 1.f / float(e.lw(l)));
        putF(p, 140, 1.f / float(e.lh(l)));
    }
}
REngine::~REngine() = default;
void REngine::setFrame(const RFrameImages &images, const RFrameSettings &settings, FrameOutput output)
{
    auto &e = *impl;
    if (e.f)
    {
        e.f->setFrame(images, settings, output);
        return;
    }
    if (e.recorded != SIZE_MAX)
        throw std::runtime_error("report the previous RR recording first");
    for (auto *t : {&images.color, &images.motion, &images.depth, &images.normals, &images.roughness,
                    &images.diffuseAlbedo, &images.specularAlbedo})
        if (!t->resource)
            throw std::runtime_error("RR requires color, motion, depth and four guides");
    if (!output.resource || viewFormat({output.resource, output.format}) != DXGI_FORMAT_R16G16B16A16_FLOAT)
        throw std::runtime_error("RR output must be RGBA16F");
    auto od = output.resource->GetDesc();
    if (od.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || od.SampleDesc.Count != 1 || od.Width != e.ow ||
        od.Height != e.oh || !(od.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
        throw std::runtime_error("RR output must match the output size and allow UAV access");
    auto same = [](const FrameTexture &a, const FrameTexture &b) {
        return a.resource == b.resource && a.format == b.format;
    };
    auto list = [](const RFrameImages &i) {
        return std::array<FrameTexture, 11>{i.color,
                                            i.motion,
                                            i.depth,
                                            i.normals,
                                            i.roughness,
                                            i.diffuseAlbedo,
                                            i.specularAlbedo,
                                            i.specularHitDistance,
                                            i.disocclusionMask,
                                            i.subsurfaceGuide,
                                            i.colorBeforeParticles};
    };
    auto inputs = list(images);
    for (auto &t : inputs)
        if (t.resource == output.resource)
            throw std::runtime_error("RR input/output alias");
    auto found = std::find_if(e.bindings.begin(), e.bindings.end(), [&](const Impl::Bindings &b) {
        auto old = list(b.images);
        for (size_t i = 0; i < old.size(); ++i)
            if (!same(old[i], inputs[i]))
                return false;
        return b.output.resource == output.resource && b.output.format == output.format;
    });
    if (found == e.bindings.end())
    {
        for (auto &b : e.bindings)
            if (b.inFlight && b.fence && b.fence->GetCompletedValue() >= b.value)
            {
                b.inFlight = false;
                b.fence.Reset();
            }
        size_t at = e.bindings.size();
        if (at >= Impl::Blocks)
        {
            at = 0;
            while (at < e.bindings.size() && (e.bindings[at].pending || e.bindings[at].inFlight))
                ++at;
            if (at == e.bindings.size())
                throw std::runtime_error("too many RR input sets in flight");
        }
        Impl::Bindings b;
        b.images = images;
        b.output = output;
        for (auto &t : inputs)
            if (t.resource)
                b.retained.emplace_back(t.resource);
        b.retained.emplace_back(output.resource);
        e.bind(b, UINT(at) * Impl::BlockSlots);
        if (at == e.bindings.size())
            e.bindings.push_back(std::move(b));
        else
            e.bindings[at] = std::move(b);
        found = e.bindings.begin() + at;
    }
    e.current = size_t(found - e.bindings.begin());
    std::copy_n(e.prevJitter, 2, e.savedJitter);
    e.savedView = e.previousView;
    e.savedClip = e.previousClip;
    setRRParameters(e, images, settings);
    for (unsigned stage : {unsigned(REnc0), unsigned(RHkpn)})
    {
        for (uint32_t off : e.model.textures[stage])
            put64(e.blocks[stage], off, e.resolve(stage, false, off, 0, images, output.resource).resource ? 1 : 0);
        for (auto &[off, fmt] : e.model.surfaces[stage])
            put64(e.blocks[stage], off, e.resolve(stage, true, off, 0, images, output.resource).resource ? 1 : 0);
    }
    // An opaque, uniform zero prevents the native driver
    // from expanding whole attention/MLP loops. Keep it outside the CUDA ABI;
    // setRRParameters restores enc0/hkpn from their original templates.
    for (unsigned stage = 0; stage < RStages; ++stage)
        if (e.model.loopZero[stage])
        {
            e.blocks[stage].resize(RParamBytes[stage] + 4);
            put32(e.blocks[stage], RParamBytes[stage], 0);
        }
    e.prevJitter[0] = settings.jitter[0];
    e.prevJitter[1] = settings.jitter[1];
    e.frameSet = true;
}
void REngine::recordFrame(ID3D12GraphicsCommandList *list, const FrameProfiler &profiler)
{
    recordFrameImpl(list, profiler, false);
}
void REngine::recordFrameForProfiling(ID3D12GraphicsCommandList *list, const FrameProfiler &profiler)
{
    recordFrameImpl(list, profiler, true);
}
// The frame's parameter blocks, the history's reset and the exposure: the front of every frame (recordFrame's
// first stages, and the external network's front).
void REngine::Impl::uploadParameters(ID3D12GraphicsCommandList *list)
{
    auto &e = *this;
    auto &b = bindings[current];
    ComPtr<ID3D12GraphicsCommandList2> list2;
    DXCHECK(list->QueryInterface(IID_PPV_ARGS(&list2)));
    std::vector<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> values;
    auto put = [&](ID3D12Resource *r, const void *data, size_t size) {
        e.use(r, D3D12_RESOURCE_STATE_COPY_DEST);
        for (size_t i = 0; i < size; i += 4)
        {
            uint32_t v;
            std::memcpy(&v, static_cast<const char *>(data) + i, 4);
            values.push_back({r->GetGPUVirtualAddress() + i, v});
        }
    };
    for (unsigned i = 0; i < RStages; ++i)
        put(e.params[i].Get(), e.blocks[i].data(), e.blocks[i].size());
    put(e.expP.Get(), &e.exposure, sizeof(e.exposure));
    e.flush(list);
    list2->WriteBufferImmediate(UINT(values.size()), values.data(), nullptr);
    if (e.reset)
    {
        const FLOAT z[4] = {};
        for (unsigned i = 0; i < 2; ++i)
        {
            e.use(e.q[i].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            e.flush(list);
            list->ClearUnorderedAccessViewFloat(e.gpu(Impl::ClearBase + i), e.clearCpu(i), e.q[i].Get(), z, 0, nullptr);
        }
    }
    D3D12_RESOURCE_BARRIER u{};
    u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    e.barriers.push_back(u);
    e.dispatch(list, e.model.exposure[0], b.exp0, e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX);
    e.dispatch(list, e.model.exposure[1], b.exp1, 1, 1);
}

// An engine image into one of the external network's shared buffers: row-major, the buffer's row pitch (a multiple of
// 256, as D3D12's placed footprints require), the buffer COMMON apart from this copy.
void REngine::Impl::copyToShared(ID3D12GraphicsCommandList *list, ID3D12Resource *source,
                                 const ExternalNetworkRr::Image &target)
{
    auto &e = *this;
    e.use(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
    e.barriers.push_back(transition(target.resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));
    e.flush(list);
    D3D12_TEXTURE_COPY_LOCATION destination{target.resource, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    destination.PlacedFootprint = {0, {target.format, target.width, target.height, 1, target.rowPitch}};
    D3D12_TEXTURE_COPY_LOCATION origin{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    list->CopyTextureRegion(&destination, 0, 0, 0, &origin, nullptr);
    e.barriers.push_back(transition(target.resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON));
}

void REngine::Impl::copyFromShared(ID3D12GraphicsCommandList *list, const ExternalNetworkRr::Image &source,
                                   ID3D12Resource *target)
{
    auto &e = *this;
    e.use(target, D3D12_RESOURCE_STATE_COPY_DEST);
    e.barriers.push_back(transition(source.resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE));
    e.flush(list);
    D3D12_TEXTURE_COPY_LOCATION origin{source.resource, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    origin.PlacedFootprint = {0, {source.format, source.width, source.height, 1, source.rowPitch}};
    D3D12_TEXTURE_COPY_LOCATION destination{target, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    list->CopyTextureRegion(&destination, 0, 0, 0, &origin, nullptr);
    e.barriers.push_back(transition(source.resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON));
}

void REngine::recordFrameImpl(ID3D12GraphicsCommandList *list, const FrameProfiler &profiler, bool rebindAfterMark)
{
    auto &e = *impl;
    if (e.f)
    {
        e.f->recordFrame(list, profiler, rebindAfterMark);
        return;
    }
    if (e.external)
        throw std::runtime_error("this engine's network is external: record the front and the back");
    if (!e.frameSet || e.recorded != SIZE_MAX)
        throw std::runtime_error("set RR frame before recording");
    auto &b = e.bindings[e.current];
    e.now = e.rest;
    ID3D12DescriptorHeap *heaps[] = {e.heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    auto stamp = [&](uint32_t i) {
        if (profiler.mark)
            profiler.mark(list, i, profiler.user);
        if (rebindAfterMark)
            list->SetDescriptorHeaps(1, heaps);
    };
    stamp(0);
    e.uploadParameters(list);
    stamp(1);
    for (unsigned stage = 0; stage < RHkpn; ++stage)
    {
        uint32_t shift = rShift(stage, e.frame);
        int l = RLevel[stage];
        e.dispatch(list, e.model.stages[stage], b.stages[stage][e.frame & 1], (e.lw(l) + (shift & 65535u) + 7) / 8,
                   (e.lh(l) + (shift >> 16) + 7) / 8);
        stamp(2 + stage);
    }
    e.dispatch(list, e.model.stages[RHkpn], b.stages[RHkpn][e.frame & 1], (e.ow + 15) / 16, (e.oh + 15) / 16);
    stamp(13);
    for (auto &[r, state] : e.now)
        if (state != e.rest.at(r))
            e.barriers.push_back(transition(r, state, e.rest.at(r)));
    e.flush(list);
    b.pending = true;
    e.recorded = e.current;
    ++e.frame;
}
ID3D12Resource *REngine::arena() const
{
    auto &e = *impl;
    return e.f ? e.f->arena.Get() : e.arena.Get();
}
uint64_t REngine::arenaBytes() const
{
    ID3D12Resource *a = arena();
    return a ? a->GetDesc().Width : 0;
}
const ExternalNetworkRr *REngine::externalNetwork() const
{
    auto &e = *impl;
    return e.f ? e.f->external : e.external;
}
// The external network's halves for presets D and E (the front of ../runtime_rr.inc recordFront): the parameters, the
// reset, the exposure and encoder 0, whose three skip surfaces the network samples, then the network's stages. The
// engine's own resources are back in their resting states when this returns; the shared buffers are COMMON apart from
// their copies. Submit it and wait on a fence before the network runs.
void REngine::recordFront(ID3D12GraphicsCommandList *list)
{
    auto &e = *impl;
    if (e.f)
    {
        e.f->recordFront(list);
        return;
    }
    if (!e.external)
        throw std::runtime_error("recordFront needs an engine with an external network");
    if (!e.frameSet || e.recorded != SIZE_MAX)
        throw std::runtime_error("set RR frame before recording");
    if (e.frontDone)
        throw std::runtime_error("recordFront before recordBack");
    auto &b = e.bindings[e.current];
    e.now = e.rest;
    ID3D12DescriptorHeap *heaps[] = {e.heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    e.uploadParameters(list);
    {
        const uint32_t shift = rShift(REnc0, e.frame);
        const int l = RLevel[REnc0];
        e.dispatch(list, e.model.stages[REnc0], b.stages[REnc0][e.frame & 1],
                   (e.lw(l) + (shift & 65535u) + 7) / 8, (e.lh(l) + (shift >> 16) + 7) / 8);
    }
    // encoder 0's surfaces 128, 112 and 120 at level 0 into the network's shared buffers
    e.copyToShared(list, e.s128[0].Get(), e.external->front[0]);
    e.copyToShared(list, e.s112[0].Get(), e.external->front[1]);
    e.copyToShared(list, e.s120[0].Get(), e.external->front[2]);
    // the network's ten stages, with the engine's windows and grids
    for (unsigned stage = REnc1; stage <= RDec0; ++stage)
    {
        const uint32_t shift = rShift(stage, e.frame);
        const int l = RLevel[stage];
        e.external->network->stage(stage, e.blocks[stage].data(), e.blocks[stage].size(),
                                   (e.lw(l) + (shift & 65535u) + 7) / 8, (e.lh(l) + (shift >> 16) + 7) / 8, shift);
    }
    for (auto &[r, state] : e.now)
        if (state != e.rest.at(r))
            e.barriers.push_back(transition(r, state, e.rest.at(r)));
    e.flush(list);
    e.frontDone = true;
}
// The back (../runtime_rr.inc recordBack): decoder 0's surface 112 and its surface 144 for this frame's parity from the
// network's shared buffers, then the output stage. Record and submit it only after the network has finished.
void REngine::recordBack(ID3D12GraphicsCommandList *list)
{
    auto &e = *impl;
    if (e.f)
    {
        e.f->recordBack(list);
        return;
    }
    if (!e.external)
        throw std::runtime_error("recordBack needs an engine with an external network");
    if (!e.frontDone)
        throw std::runtime_error("recordFront before recordBack");
    auto &b = e.bindings[e.current];
    e.now = e.rest;
    ID3D12DescriptorHeap *heaps[] = {e.heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    // the network wrote them after the front retired on the CPU: this UAV barrier invalidates the caches the output
    // stage reads them through
    D3D12_RESOURCE_BARRIER uav{};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &uav);
    e.copyFromShared(list, e.external->back[0], e.d112[0].Get());
    e.copyFromShared(list, e.external->back[1 + (e.frame & 1)], e.r[e.frame & 1].Get());
    e.dispatch(list, e.model.stages[RHkpn], b.stages[RHkpn][e.frame & 1], (e.ow + 15) / 16, (e.oh + 15) / 16);
    for (auto &[r, state] : e.now)
        if (state != e.rest.at(r))
            e.barriers.push_back(transition(r, state, e.rest.at(r)));
    e.flush(list);
    b.pending = true;
    e.recorded = e.current;
    ++e.frame;
    e.frontDone = false;
}
void REngine::setExternalNetwork(const ExternalNetworkRr *external)
{
    auto &e = *impl;
    if (e.frameSet || e.frontDone || e.recorded != SIZE_MAX)
        throw std::runtime_error("set the external network before the first frame");
    if (external && !external->network)
        throw std::runtime_error("an external network needs its host");
    if (external)
    {
        // the front dispatches encoder 0 only (and the gather for preset F): the network must run every network stage
        const unsigned last = e.f ? 14u : 10u;
        for (unsigned stage = 1; stage <= last; ++stage)
            if (!external->network->runs(stage))
                throw std::runtime_error("the external network does not run Ray Reconstruction stage " +
                                         std::to_string(stage));
        if (!e.f)
            for (int i = 0; i < 3; ++i)
                if (!external->front[i].resource || !external->back[i].resource)
                    throw std::runtime_error("the external network needs its three front and three back images");
    }
    e.external = external;
    if (e.f)
        e.f->external = external;
}
void REngine::submitted(ID3D12Fence *fence, uint64_t value)
{
    auto &e = *impl;
    if (e.f)
    {
        e.f->submitted(fence, value);
        return;
    }
    if (e.recorded == SIZE_MAX)
        throw std::runtime_error("no RR recording");
    auto &b = e.bindings[e.recorded];
    b.pending = false;
    b.inFlight = true;
    b.fence = fence;
    b.value = value;
    e.recorded = SIZE_MAX;
    e.frameSet = false;
}
void REngine::discarded()
{
    auto &e = *impl;
    if (e.f)
    {
        e.f->discarded();
        return;
    }
    if (e.recorded == SIZE_MAX)
        throw std::runtime_error("no RR recording");
    e.bindings[e.recorded].pending = false;
    e.recorded = SIZE_MAX;
    e.frameSet = false;
    --e.frame;
    std::copy_n(e.savedJitter, 2, e.prevJitter);
    e.previousView = e.savedView;
    e.previousClip = e.savedClip;
}
bool REngine::idle() const
{
    auto &e = *impl;
    if (e.f)
        return e.f->idle();
    if (e.recorded != SIZE_MAX)
        return false;
    for (auto &b : e.bindings)
        if (b.inFlight && (!b.fence || b.fence->GetCompletedValue() < b.value))
            return false;
    return true;
}
void RModel::Impl::execute(const std::function<void(ID3D12GraphicsCommandList *)> &commands)
{
    D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    DXCHECK(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)));
    DXCHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
    DXCHECK(
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
    DXCHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    commands(list.Get());
    DXCHECK(list->Close());
    ID3D12CommandList *lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    DXCHECK(queue->Signal(fence.Get(), 1));
    if (fence->GetCompletedValue() < 1)
    {
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event)
            throw std::runtime_error("CreateEvent failed");
        const HRESULT hr = fence->SetEventOnCompletion(1, event);
        if (SUCCEEDED(hr))
            WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
        DXCHECK(hr);
    }
    if (FAILED(device->GetDeviceRemovedReason()))
        throw std::runtime_error("the device was removed during setup");
}
void REngine::Impl::write(UINT first, const std::vector<Binding> &kinds, const std::vector<Slot> &slots)
{
    for (UINT i = 0; i < slots.size(); ++i)
    {
        const Slot &s = slots[i];
        const Binding &b = kinds[i];
        const auto h = cpu(first + i);
        if (b.kind == Kind::Tex)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC v{};
            v.Format = s.format != DXGI_FORMAT_UNKNOWN ? s.format : readFormat(s.resource->GetDesc().Format);
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            v.Texture2D.MipLevels = 1;
            device->CreateShaderResourceView(s.resource, &v, h);
        }
        else if (b.kind == Kind::Raw)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC v{};
            v.Format = DXGI_FORMAT_R32_TYPELESS;
            v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            v.Buffer = {s.offset / 4, UINT((s.resource->GetDesc().Width - s.offset) / 4), 0, D3D12_BUFFER_SRV_FLAG_RAW};
            device->CreateShaderResourceView(s.resource, &v, h);
        }
        else if (b.kind == Kind::RwTex)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
            v.Format = s.format != DXGI_FORMAT_UNKNOWN ? s.format : readFormat(s.resource->GetDesc().Format);
            v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            device->CreateUnorderedAccessView(s.resource, nullptr, &v, h);
        }
        else if (b.kind == Kind::RwStruct)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
            v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            v.Buffer = {0, UINT(s.resource->GetDesc().Width / b.stride), b.stride, 0, D3D12_BUFFER_UAV_FLAG_NONE};
            device->CreateUnorderedAccessView(s.resource, nullptr, &v, h);
        }
        else if (b.kind == Kind::RwRaw)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
            v.Format = DXGI_FORMAT_R32_TYPELESS;
            v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            v.Buffer = {0, UINT(s.resource->GetDesc().Width / 4), 0, 0, D3D12_BUFFER_UAV_FLAG_RAW};
            device->CreateUnorderedAccessView(s.resource, nullptr, &v, h);
        }
    }
    tables[first] = slots;
}

Slot REngine::Impl::resolve(unsigned stage, bool surface, uint32_t offset, uint32_t par, const RFrameImages &in,
                            ID3D12Resource *output) const
{
    const int l = RLevel[stage];
    auto T = [](const ComPtr<ID3D12Resource> &t, bool lin = false) { return Slot{t.Get()}; };
    auto G = [](const FrameTexture &t, bool lin = false) { return Slot{t.resource, 0, t.format, true}; };
    if (stage == REnc0)
    {
        if (!surface)
            switch (offset)
            {
            case 160:
                return G(in.color, true);
            case 168:
                return G(in.motion);
            case 184:
                return G(in.normals);
            case 192:
                return G(in.roughness);
            case 200:
                return T(e[0]);
            case 208:
                return T(e[1]);
            case 224:
                return T(q[par ^ 1], true);
            case 232:
                return G(in.diffuseAlbedo);
            case 240:
                return G(in.specularAlbedo);
            // the optional inputs (capDX, 2026-10-12): unbound without them, like NGX
            case 248:
                return in.specularHitDistance.resource ? G(in.specularHitDistance) : Slot{};
            case 296:
                return in.disocclusionMask.resource ? G(in.disocclusionMask) : Slot{};
            case 264:
                return in.subsurfaceGuide.resource ? G(in.subsurfaceGuide) : Slot{}; // capDXsss
            case 488:
                return T(r[par ^ 1], true);
            case 496:
                return T(h[par ^ 1], true);
            case 512:
                return G(in.depth);
            }
        else
            switch (offset)
            {
            case 112:
                return T(s112[0]);
            case 120:
                return T(s120[0]);
            case 128:
                return T(s128[0]);
            case 304:
                return T(q[par]);
            }
    }
    else if (stage <= REnc5)
    {
        if (!surface && offset == 88)
            return T(s128[l - 1], true);
        if (surface && offset == 112)
            return T(s112[l]);
        if (surface && offset == 120)
            return T(s120[l]);
        if (surface && offset == 128 && l < 5)
            return T(s128[l]);
    }
    else if (stage <= RDec0)
    {
        if (!surface && offset == 88)
            return T(stage == RDec4 ? s112[5] : d112[l + 1], true);
        if (!surface && offset == 96)
            return T(s112[l]);
        if (!surface && offset == 104)
            return T(s120[l]);
        if (surface && offset == 112)
            return T(d112[l]);
        if (surface && offset == 120)
            return T(d120[l]);
        if (surface && offset == 144 && stage == RDec0)
            return T(r[par]);
    }
    else
    {
        if (!surface)
            switch (offset)
            {
            case 176:
                return G(in.color, true);
            case 192:
                return G(in.depth);
            case 200:
                return T(e[0]);
            case 208:
                return T(e[1]);
            case 216:
                return T(h[par ^ 1], true);
            case 280:
                return G(in.diffuseAlbedo);
            case 288:
                return G(in.specularAlbedo);
            case 296:
                return T(q[par]);
            case 448:
                return T(d112[0]);
            // DLSSD.ColorBeforeParticles and its history (capDXparticles, 2026-10-12)
            case 416:
                return in.colorBeforeParticles.resource ? G(in.colorBeforeParticles) : Slot{};
            case 392:
                return in.colorBeforeParticles.resource ? T(pb[par ^ 1], true) : Slot{};
            }
        else
            switch (offset)
            {
            case 224:
                return T(h[par]);
            case 240:
                return Slot{output, 0, DXGI_FORMAT_UNKNOWN, true};
            case 384:
                return in.colorBeforeParticles.resource ? T(pb[par]) : Slot{};
            }
    }
    return {};
}

void REngine::Impl::bind(Bindings &b, UINT first)
{
    auto own = [](const ComPtr<ID3D12Resource> &r) { return Slot{r.Get()}; };
    for (unsigned stage = 0; stage < RStages; ++stage)
        for (unsigned par = 0; par < 2; ++par)
        {
            std::vector<Slot> slots{own(params[stage]), Slot{model.weights[stage].Get(), 0, DXGI_FORMAT_UNKNOWN, true},
                                    own(arena)};
            for (uint32_t off : model.textures[stage])
            {
                auto t = resolve(stage, false, off, par, b.images, b.output.resource);
                slots.push_back(t.resource ? t : own(zero));
            }
            for (auto &[off, fmt] : model.surfaces[stage])
            {
                auto t = resolve(stage, true, off, par, b.images, b.output.resource);
                slots.push_back(t.resource ? t : own(spare.at(fmt)));
            }
            if (model.scratchWords[stage])
            {
                slots.resize(41);
                slots[40] = own(scratch[stage]);
            }
            b.stages[stage][par] = first;
            write(first, model.layouts[stage], slots);
            first += UINT(slots.size());
        }
    b.exp0 = first;
    write(first, model.exp0B, {own(expP), Slot{b.images.color.resource, 0, b.images.color.format, true}, own(expSums)});
    first += 3;
    b.exp1 = first;
    write(first, model.exp1B, {own(expP), own(expSums), own(expState), own(e[0]), own(e[1])});
}
std::vector<REngine::DebugImage> REngine::debugImages() const
{
    auto &e = *impl;
    if (e.f)
        return e.f->debugImages();
    std::vector<DebugImage> images;
    auto put = [&](const std::string &name, const ComPtr<ID3D12Resource> &r) {
        auto d = r->GetDesc();
        images.push_back({name, r.Get(), d.Format, uint32_t(d.Width), d.Height});
    };
    for (int l = 0; l < 6; ++l)
    {
        put("enc" + std::to_string(l) + "-112", e.s112[l]);
        put("enc" + std::to_string(l) + "-120", e.s120[l]);
        if (l < 5)
        {
            put("enc" + std::to_string(l) + "-128", e.s128[l]);
            put("dec" + std::to_string(l) + "-112", e.d112[l]);
        }
    }
    for (int i = 0; i < 2; ++i)
    {
        put("q" + std::to_string(i), e.q[i]);
        put("r" + std::to_string(i), e.r[i]);
        put("h" + std::to_string(i), e.h[i]);
        put("e" + std::to_string(i), e.e[i]);
    }
    return images;
}

} // namespace d4r::dx
