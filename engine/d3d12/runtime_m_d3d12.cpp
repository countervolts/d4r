#include "runtime_m_d3d12.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace d4r::dx {
namespace {
#include "runtime_helpers.h"
constexpr unsigned MLayers = 10;
constexpr const char* MShader[MLayers] = {"m_enc1", "m_enc2", "m_tube", "m_tube", "m_tube", "m_tube", "m_tube", "m_tube", "m_dec2", "m_dec1"};
constexpr int32_t MShift[MLayers] = {2, 2, 2, 0, 2, 0, 2, 0, 0, 0};
constexpr unsigned MScale[MLayers] = {1, 2, 4, 4, 4, 4, 4, 4, 2, 1};
// m_tube_w2 (optional, compile_m.py --d3d12-target vkd3d): the tube with two windows per wave, for dispatches of at
// least this many workgroups; fewer leave too few waves to hide its latency (RX 9070 XT on vkd3d-proton, a block:
// 112 groups (M 854x480) 25 -> 34 us, 144 (960x540) 52 -> 38, 252 (1280x720) 64 -> 52). D4R_ENGINE_M_TUBE_W2_GROUPS
// overrides it.
constexpr uint32_t MTubeW2Groups = 128;
// the stages' parameter blocks (runtime_m.inc)
struct MEnc0Params {
    int32_t size[2], shift[2], tok[2];
    float jitter[2], prevJitter[2], mvScale[2], mvOffs[2], rnd2hi[2], hi2rnd[2], hiSize[2];
    float changeScale, changeGain;
    uint32_t depthFlag, reset, frame;
    float expo, expoPrev;
    float halfInv[2], renderInv[2];
    uint32_t displayMv; int32_t mvMax[2]; float mvPix[2];
};
struct MDec0Params { int32_t size[2], tokW; uint32_t reset; float rndOffs[2], rnd2hi[2], hiSize[2], histOffs[2], sigma, rho, renderInv[2]; };
struct MPostParams {
    int32_t colOrigin[2], hiSize[2], rndSize[2]; float hi2rnd[2], rndOffs[2], invRnd[2], gain; uint32_t reset; float expo, expoPrev;
    int32_t mvSize[2]; float mvScale[2], mvOffs[2];
};
struct MDownParams { int32_t srcSize[2], outOrigin[2], colOrigin[2], colSize[2]; float out2rnd[2], rndOffs[2], expo; };
struct MLayerParams { int32_t w, h, sx, sy; uint32_t o[72]; };
struct MExposureParams { int32_t origin[2], size[2]; float invCount, scale, key; uint32_t flags, blocks, blocksX; int32_t sampleSize[2];
                         float gameScale = 1, invPre = 1; };
static_assert(sizeof(MEnc0Params) == 144 && sizeof(MDec0Params) == 64 && sizeof(MPostParams) == 88 && sizeof(MDownParams) == 52 &&
              sizeof(MLayerParams) == 304 && sizeof(MExposureParams) == 56, "the stages' parameter blocks");
uint32_t exposureGrid(uint32_t render, uint32_t output) { return output / 2 <= render ? output / 2 : std::max(render / 2, 256u); }

const std::vector<Binding> LayerB{R, R, R, R, R, UR, UR};
const std::vector<Binding> Enc0B{R, T, T, T, TL, TL, TL, UT, UT, UT, R, R, R, UR};
const std::vector<Binding> Dec0B{R, T, TL, UT, UT, UT, UT, R, R, R};
const std::vector<Binding> Dec0LB{R, R, R, R, R, UR, UR, R, UT, UT, UT, UT};
// preset L's input stage in one dispatch (enc0_lx, vkd3d-proton packages): Enc0B's slots, then LayerB's
const std::vector<Binding> Enc0LXB{R, T, T, T, TL, TL, TL, UT, UT, UT, R, R, R, UR, R, R, R, R, R, UR, UR};
const std::vector<Binding> PostB{R, T, T, T, T, TL, T, T, TL, UT, UT, US(4), T};
const std::vector<Binding> Exp0B{US(sizeof(MExposureParams)), T, US(4)};
const std::vector<Binding> Exp1B{US(sizeof(MExposureParams)), US(4), US(8), US(4), US(4), US(4)};
const std::vector<Binding> DownB{R, T, UT, US(4)};

} // namespace

// ---------------------------------------------------------------- model
struct MModel::Impl {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12Resource> w8, w16;
    std::vector<char> b8, b16;          // until uploaded
    uint32_t enc0Off[3]{}, dec0Off[2]{};
    UINT64 enc0W = 0, dec0W = 0;        // the rearranged matrices in w8
    std::array<std::array<uint32_t, 74>, MLayers> layers{};
    std::array<Pipeline, MLayers> network;
    Pipeline tubeW2;
    uint32_t tubeW2Groups = MTubeW2Groups;
    Pipeline enc0, dec0, post, postQ, postC, down, downRgb10, exp0, exp1;
    // the input and expansion stages with the token buffers in NGX's plane layout, for an external (HIP) network; L's are
    // its input stage's block and its expansion (layer_m.hlsl PLANES, KIND 3 and 4), after the same feature slots
    Pipeline enc0Planes, dec0Planes;
    // preset L (offsets.bin D4RL0002): its input stage's and expansion stage's blocks, the dither table, its constants
    // (input motion-change scale and gain, covariance sigma and rho, first-gate gain; M's otherwise) and the input
    // stage's block (layer_m.hlsl KIND 3; enc0 is then its feature slots, enc0_m.comp L_FEATURES)
    std::array<uint32_t, 74> lEnc0{}, lDec0{};
    uint32_t dither16 = 0;
    float consts[5] = {9.045065879821777f, 0.22356846928596497f, 8.f, 0.07863261550664902f, 2.7530112f};
    Pipeline enc0L, enc0LX;
    bool presetL = false, ldr = false, displayMotion = false, uploaded = false;
    std::mutex setup;
    void execute(const std::function<void(ID3D12GraphicsCommandList*)>& commands) {
        std::lock_guard<std::mutex> lock(setup);
        D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        ComPtr<ID3D12CommandQueue> queue;
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        ComPtr<ID3D12Fence> fence;
        DXCHECK(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)));
        DXCHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        DXCHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
        DXCHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        commands(list.Get());
        DXCHECK(list->Close());
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        DXCHECK(queue->Signal(fence.Get(), 1));
        if (fence->GetCompletedValue() < 1) {
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!event) throw std::runtime_error("CreateEvent failed");
            const HRESULT hr = fence->SetEventOnCompletion(1, event);
            if (SUCCEEDED(hr)) WaitForSingleObject(event, INFINITE);
            CloseHandle(event);
            DXCHECK(hr);
        }
        if (FAILED(device->GetDeviceRemovedReason())) throw std::runtime_error("the device was removed during setup");
    }
};

MModel::MModel(ID3D12Device* device, const std::string& directory) : impl(std::make_unique<Impl>()) {
    auto& m = *impl;
    m.device = device;
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) || model.HighestShaderModel < D3D_SHADER_MODEL_6_6)
        throw std::runtime_error("the device does not support Shader Model 6.6");
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4))) || !options4.Native16BitShaderOpsSupported)
        throw std::runtime_error("the device has no native 16-bit shader operations");
    const auto head = slurp(directory + "/offsets.bin");
    m.presetL = head.size() == 40 + (MLayers + 2) * 74 * 4 && !std::memcmp(head.data(), "D4RL0002", 8);
    if (!m.presetL && (head.size() != 16 + 5 * 4 + MLayers * 74 * 4 || std::memcmp(head.data(), "D4RM000", 7)))
        throw std::runtime_error("not a preset M or L model (offsets.bin)");
    const char* in = m.presetL ? "enc0_lf" : "enc0_m";
    const char* ex = m.presetL ? "dec0_l" : "dec0_m";
    const std::string po = m.presetL ? "post_l" : "post_m";
    for (const std::string& f : {std::string("m_enc1"), std::string("m_enc2"), std::string("m_tube"), std::string("m_dec2"),
                                 std::string("m_dec1"), std::string(in), std::string(ex), po, po + "_q", po + "_c", std::string("down_m"),
                                 std::string("exposure_m0"), std::string("exposure_m1"), std::string(m.presetL ? "enc0_l" : in)})
        if (!std::filesystem::exists(directory + "/" + f + ".dxil"))
            throw std::runtime_error(directory + " has no " + f + ".dxil: build the model with compile_m.py --dxil yes on Windows");
    // L's stages take render-resolution motion only (its matrix stages have no display-resolution variant)
    m.displayMotion = !m.presetL && head[7] >= '5';
    uint32_t n8, n16;
    std::memcpy(&n8, head.data() + 8, 4); std::memcpy(&n16, head.data() + 12, 4);
    if (m.presetL) {
        std::memcpy(&m.dither16, head.data() + 16, 4); std::memcpy(m.consts, head.data() + 20, sizeof(m.consts));
        std::memcpy(m.lEnc0.data(), head.data() + 40, sizeof(m.lEnc0));
        std::memcpy(m.layers.data(), head.data() + 40 + sizeof(m.lEnc0), sizeof(m.layers));
        std::memcpy(m.lDec0.data(), head.data() + 40 + sizeof(m.lEnc0) + sizeof(m.layers), sizeof(m.lDec0));
        for (float c : m.consts) if (!(c > 0.f && c < 100.f)) throw std::runtime_error("invalid L model constants");
    } else {
        std::memcpy(m.enc0Off, head.data() + 16, 12); std::memcpy(m.dec0Off, head.data() + 28, 8);
        std::memcpy(m.layers.data(), head.data() + 36, sizeof(m.layers));
    }
    m.b8 = slurp(directory + "/weights8.bin"); m.b16 = slurp(directory + "/weights16.bin");
    if (m.b8.size() != n8 || m.b16.size() != n16 || (!m.presetL && (m.enc0Off[0] + 128 * 64 > n8 || m.dec0Off[0] + 64 * 80 > n8)))
        throw std::runtime_error("invalid M model sizes");
    // The FP8 matrices of the input and expansion stages rearranged after the package's, column-major, so that a lane's
    // WMMA operand bytes are adjacent (the package's row-major layout takes a byte load per element: input stage 143 ->
    // 79 us at 854x480 -> 2560x1440). Input stage, [128][64]: the bytes of channel c are [2][4][8], half h, k32 step s,
    // k = 32 s + 16 j + 8 h + i at c * 128 + 64 h + 16 s + 8 j + i (a k32 step's A operands are one 16-byte load).
    // Expansion, [64][80]: the bytes of column c are [2][4][8], k = 16 t + 8 h + i at c * 64 + 32 h + 8 t + i.
    // (as replay_m_d3d12.cpp)
    m.enc0W = (m.b8.size() + 255) / 256 * 256; m.dec0W = m.enc0W + 128 * 64;
    auto& b8 = m.b8;
    if (!m.presetL) b8.resize(m.dec0W + 64 * 80);
    if (!m.presetL) for (uint32_t k = 0; k < 128; ++k)
        for (uint32_t c = 0; c < 64; ++c)
            b8[m.enc0W + c * 128 + 64 * (k >> 3 & 1) + 16 * (k >> 5) + 8 * (k >> 4 & 1) + (k & 7)] = b8[m.enc0Off[0] + k * 64 + c];
    if (!m.presetL) for (uint32_t k = 0; k < 64; ++k)
        for (uint32_t c = 0; c < 80; ++c)
            b8[m.dec0W + c * 64 + 32 * (k >> 3 & 1) + 8 * (k >> 4) + (k & 7)] = b8[m.dec0Off[0] + k * 80 + c];
    m.ldr = std::filesystem::exists(directory + "/ldr");
    m.w8 = buffer(device, b8.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    m.w16 = buffer(device, m.b16.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    auto path = [&](const std::string& name) { return directory + "/" + name + ".dxil"; };
    for (unsigned l = 0; l < MLayers; ++l) m.network[l].create(device, path(MShader[l]), LayerB);
    if (std::filesystem::exists(path("m_tube_w2"))) m.tubeW2.create(device, path("m_tube_w2"), LayerB);
    if (const char* g = std::getenv("D4R_ENGINE_M_TUBE_W2_GROUPS")) m.tubeW2Groups = uint32_t(std::strtoul(g, nullptr, 10));
    m.enc0.create(device, path(in), Enc0B); m.dec0.create(device, path(ex), m.presetL ? Dec0LB : Dec0B);
    // the same stages for an external network (the native HIP layers): tokens/head in NGX's plane layout
    const std::string inPlanes = m.presetL ? "enc0_l_planes" : "enc0_m_planes", exPlanes = m.presetL ? "dec0_l_planes" : "dec0_m_planes";
    if (std::filesystem::exists(path(inPlanes)) && std::filesystem::exists(path(exPlanes))) {
        m.enc0Planes.create(device, path(inPlanes), m.presetL ? LayerB : Enc0B);
        m.dec0Planes.create(device, path(exPlanes), m.presetL ? Dec0LB : Dec0B);
    }
    if (m.presetL) m.enc0L.create(device, path("enc0_l"), LayerB);
    if (m.presetL && std::filesystem::exists(path("enc0_lx"))) m.enc0LX.create(device, path("enc0_lx"), Enc0LXB);
    m.post.create(device, path(po), PostB); m.postQ.create(device, path(po + "_q"), PostB);
    m.postC.create(device, path(po + "_c"), PostB);
    m.down.create(device, path("down_m"), DownB);
    if (std::filesystem::exists(path("down_m_rgb10"))) m.downRgb10.create(device, path("down_m_rgb10"), DownB);
    m.exp0.create(device, path("exposure_m0"), Exp0B); m.exp1.create(device, path("exposure_m1"), Exp1B);
}
MModel::~MModel() = default;
bool MModel::isLdr() const { return impl->ldr; }
bool MModel::isPresetL() const { return impl->presetL; }
bool MModel::hasDisplayMotion() const { return impl->displayMotion; }
bool MModel::hasRgb10Output() const { return bool(impl->downRgb10); }
void MModel::upload() {
    auto& m = *impl;
    if (m.uploaded) return;
    const UINT64 at16 = (m.b8.size() + 255) / 256 * 256;
    auto up = buffer(m.device.Get(), at16 + m.b16.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr; D3D12_RANGE none{0, 0};
    DXCHECK(up->Map(0, &none, &mapped));
    std::memcpy(mapped, m.b8.data(), m.b8.size());
    std::memcpy(static_cast<char*>(mapped) + at16, m.b16.data(), m.b16.size());
    up->Unmap(0, nullptr);
    m.execute([&](ID3D12GraphicsCommandList* list) {
        list->CopyBufferRegion(m.w8.Get(), 0, up.Get(), 0, m.b8.size());
        list->CopyBufferRegion(m.w16.Get(), 0, up.Get(), at16, m.b16.size());
        const D3D12_RESOURCE_BARRIER b[] = {transition(m.w8.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                                            transition(m.w16.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
        list->ResourceBarrier(2, b);
    });
    m.uploaded = true;
    std::vector<char>().swap(m.b8); std::vector<char>().swap(m.b16);
}

// ---------------------------------------------------------------- engine
namespace {
// A slot of a descriptor table: an engine resource (its view and the state a dispatch needs) or a game texture.
struct Slot {
    ID3D12Resource* resource = nullptr;
    UINT64 offset = 0;              // Raw: byte offset of the view
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    bool game = false;              // a game texture: in the state the caller gave it, not transitioned
};
}
struct MEngine::Impl {
    MModel::Impl& model;
    ID3D12Device* device;
    uint32_t maxRw, maxRh, ow, oh, hw, hh, rw = 0, rh = 0, gw = 0, gh = 0;
    ComPtr<ID3D12Resource> enc0P, dec0P, postP, downP, expP, layerP, expSums, expState, dummyIn, dummyOut, ags, packed;
    ComPtr<ID3D12Resource> tokens, skip1, merged1, skip2, merged2, tubeA, tubeB, dec2Out, dec1Out;
    ComPtr<ID3D12Resource> skipL, featsL, dummyOut2;   // preset L: the input stage's pixel output and feature slots
    std::array<ComPtr<ID3D12Resource>, 2> mvA, feat, hist, half;
    ComPtr<ID3D12Resource> mvB, mvC, gates, mix, cov;
    // resting states: textures NON_PIXEL_SHADER_RESOURCE, buffers UNORDERED_ACCESS
    std::unordered_map<ID3D12Resource*, D3D12_RESOURCE_STATES> rest, now;
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    ComPtr<ID3D12DescriptorHeap> heap;
    UINT increment = 0;
    // static tables: the layers (preset L's input stage block after them), the expansion (per parity) and exposure pass
    // 1; then one block per distinct set of game textures: input stage and reconstruction per parity, exposure pass 0
    // and the downsample
    static constexpr UINT LayerBase = 0, Enc0LBase = LayerBase + MLayers * 7, Dec0Base = Enc0LBase + 7, Dec0Slots = 12;
    static constexpr UINT Exp1Base = Dec0Base + 2 * Dec0Slots, BlockBase = Exp1Base + 6;
    static constexpr UINT Enc0Slot = 0, PostSlot = 28, Exp0Slot = 54, DownSlot = 57, Enc0LXSlot = 64, BlockSlots = 106, Blocks = 16;
    struct Bindings {
        FrameTextures textures{};
        FrameOutput output{};
        ComPtr<ID3D12Resource> retained[4];
        ComPtr<ID3D12Fence> fence;
        uint64_t value = 0, lastUse = 0;
        bool pending = false, inFlight = false;
        UINT block = 0;
    };
    std::vector<Bindings> bindings;
    size_t current = 0, recorded = SIZE_MAX;
    uint64_t uses = 0;
    MEnc0Params enc0{}; MDec0Params dec0{}; MPostParams post{}; MDownParams down{}; MExposureParams exposure{};
    float prevJitter[2] = {0, 0}, savedJitter[2] = {0, 0};
    unsigned parity = 0, frame = 0;
    bool frameSet = false, quality = false, compact = false, rgb10 = false;
    bool external = false, frontDone = false;
    ID3D12Resource* externalTokens = nullptr;   // the caller's (an external network's), in COMMON, never transitioned
    ID3D12Resource* externalHead = nullptr;
    Slot tokenSlot() const { return external ? Slot{externalTokens, 0, DXGI_FORMAT_UNKNOWN, true} : Slot{tokens.Get()}; }
    Slot headSlot() const { return external ? Slot{externalHead, 0, DXGI_FORMAT_UNKNOWN, true} : Slot{dec1Out.Get()}; }
    static constexpr uint64_t UnfencedFrames = 16;

    Impl(MModel::Impl& m) : model(m), device(m.device.Get()) {}
    ComPtr<ID3D12Resource> buf(UINT64 bytes) {
        auto r = buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        rest[r.Get()] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        return r;
    }
    ComPtr<ID3D12Resource> tex(UINT w, UINT h, DXGI_FORMAT f) {
        auto r = texture(device, w, h, f);
        rest[r.Get()] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        return r;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT i) const { auto h = heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * increment; return h; }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT i) const { auto h = heap->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(i) * increment; return h; }
    void write(UINT first, const std::vector<Binding>& kinds, const std::vector<Slot>& slots) {
        for (UINT i = 0; i < slots.size(); ++i) {
            const Slot& s = slots[i]; const Binding& b = kinds[i];
            const auto h = cpu(first + i);
            if (b.kind == Kind::Tex) {
                D3D12_SHADER_RESOURCE_VIEW_DESC v{};
                v.Format = s.format != DXGI_FORMAT_UNKNOWN ? s.format : readFormat(s.resource->GetDesc().Format);
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; v.Texture2D.MipLevels = 1;
                device->CreateShaderResourceView(s.resource, &v, h);
            } else if (b.kind == Kind::Raw) {
                D3D12_SHADER_RESOURCE_VIEW_DESC v{}; v.Format = DXGI_FORMAT_R32_TYPELESS; v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                v.Buffer = {s.offset / 4, UINT((s.resource->GetDesc().Width - s.offset) / 4), 0, D3D12_BUFFER_SRV_FLAG_RAW};
                device->CreateShaderResourceView(s.resource, &v, h);
            } else if (b.kind == Kind::RwTex) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
                v.Format = s.format != DXGI_FORMAT_UNKNOWN ? s.format : readFormat(s.resource->GetDesc().Format);
                v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                device->CreateUnorderedAccessView(s.resource, nullptr, &v, h);
            } else if (b.kind == Kind::RwStruct) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC v{}; v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                v.Buffer = {0, UINT(s.resource->GetDesc().Width / b.stride), b.stride, 0, D3D12_BUFFER_UAV_FLAG_NONE};
                device->CreateUnorderedAccessView(s.resource, nullptr, &v, h);
            } else if (b.kind == Kind::RwRaw) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC v{}; v.Format = DXGI_FORMAT_R32_TYPELESS; v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                v.Buffer = {0, UINT(s.resource->GetDesc().Width / 4), 0, 0, D3D12_BUFFER_UAV_FLAG_RAW};
                device->CreateUnorderedAccessView(s.resource, nullptr, &v, h);
            }
        }
    }
    // the slots of every table, for the state transitions of their dispatches
    std::vector<std::vector<Slot>> tables;
    std::unordered_map<UINT, size_t> tableAt;
    void table(UINT first, const std::vector<Binding>& kinds, std::vector<Slot> slots) {
        write(first, kinds, slots);
        tableAt[first] = tables.size();
        tables.push_back(std::move(slots));
    }
    void use(ID3D12Resource* r, D3D12_RESOURCE_STATES state) {
        auto& s = now[r];
        if (s == state) return;
        barriers.push_back(transition(r, s, state));
        s = state;
    }
    void flush(ID3D12GraphicsCommandList* list) {
        if (!barriers.empty()) list->ResourceBarrier(UINT(barriers.size()), barriers.data());
        barriers.clear();
    }
    void dispatch(ID3D12GraphicsCommandList* list, const Pipeline& p, UINT first, UINT x, UINT y) {
        const auto& slots = tables[tableAt.at(first)];
        const auto& kinds = *p.bindings;
        for (UINT i = 0; i < slots.size(); ++i) {
            if (slots[i].game || !slots[i].resource) continue;
            const Kind k = kinds[i].kind;
            use(slots[i].resource, k == Kind::Tex || k == Kind::Raw ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                                                     : D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        flush(list);
        list->SetComputeRootSignature(p.root.Get());
        list->SetPipelineState(p.state.Get());
        list->SetComputeRootDescriptorTable(0, gpu(first));
        list->SetComputeRootUnorderedAccessView(1, ags->GetGPUVirtualAddress());
        list->Dispatch(x, y, 1);
        D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barriers.push_back(uav);
    }
    void retire() {
        for (auto& b : bindings)
            if (b.inFlight && (b.fence ? b.fence->GetCompletedValue() >= b.value : uses >= b.value + UnfencedFrames)) {
                b.inFlight = false; b.fence.Reset();
            }
    }
    void writeBindings(Bindings& b);
};

void MEngine::Impl::writeBindings(Bindings& b) {
    auto& m = model;
    const UINT base = BlockBase + b.block * BlockSlots;
    const auto& t = b.textures;
    const Slot color{t.color.resource, 0, viewFormat(t.color), true}, motion{t.motion.resource, 0, viewFormat(t.motion), true};
    const Slot depth{t.depth.resource, 0, viewFormat(t.depth), true};
    const Slot out{b.output.resource, 0, viewFormat(FrameTexture{b.output.resource, b.output.format}), true};
    auto s = [](const ComPtr<ID3D12Resource>& r, UINT64 offset = 0) { return Slot{r.Get(), offset}; };
    // the tables of a block that a later frame reuses replace the old entries (tableAt keeps the first index)
    auto put = [&](UINT first, const std::vector<Binding>& kinds, std::vector<Slot> slots) {
        write(first, kinds, slots);
        if (auto it = tableAt.find(first); it != tableAt.end()) tables[it->second] = std::move(slots);
        else { tableAt[first] = tables.size(); tables.push_back(std::move(slots)); }
    };
    for (unsigned i = 0; i < 2; ++i) {      // i: the index of this frame's history outputs
        const unsigned prev = 1 - i;
        if (m.presetL)      // L: the feature slots (its weights unused), L's dither table
            put(base + Enc0Slot + 14 * i, Enc0B, {s(enc0P), color, motion, depth, s(mvA[prev]), s(half[prev]), s(feat[prev]), s(mvA[i]),
                                                 s(mvB), s(mvC), s(m.w8), s(m.w16), s(m.w16, m.dither16), s(featsL)});
        else
            put(base + Enc0Slot + 14 * i, Enc0B, {s(enc0P), color, motion, depth, s(mvA[prev]), s(half[prev]), s(feat[prev]), s(mvA[i]),
                                                 s(mvB), s(mvC), s(m.w8, m.enc0W), s(m.w16, m.enc0Off[1]), s(m.w16, m.enc0Off[2]), tokenSlot()});
        if (m.enc0LX)       // L in one dispatch: the input stage's slots, then its layer block's (as Enc0LBase's)
            put(base + Enc0LXSlot + 21 * i, Enc0LXB, {s(enc0P), color, motion, depth, s(mvA[prev]), s(half[prev]), s(feat[prev]),
                                                     s(mvA[i]), s(mvB), s(mvC), s(m.w8), s(m.w16), s(m.w16, m.dither16), s(featsL),
                                                     s(layerP, 512 * MLayers), s(m.w8, m.lEnc0[0]), s(m.w16, m.lEnc0[1]), s(featsL),
                                                     s(dummyIn), s(skipL), tokenSlot()});
        put(base + PostSlot + 13 * i, PostB, {s(postP), color, s(mix), s(gates), s(cov), s(hist[prev]), s(mvA[i]), s(mvB), s(mvC),
                                             s(hist[i]), s(half[i]), s(packed), motion});
    }
    put(base + Exp0Slot, Exp0B, {s(expP), color, s(expSums)});
    put(base + DownSlot, DownB, {s(downP), color, out, s(packed)});
}

uint64_t MEngine::networkBufferBytes(uint32_t renderWidth, uint32_t renderHeight) {
    return uint64_t((renderWidth + 31) / 32 * 16) * ((renderHeight + 31) / 32 * 16) * 64 + 64;
}

MEngine::MEngine(MModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight)
    : MEngine(model, renderWidth, renderHeight, outputWidth, outputHeight, nullptr) {}

MEngine::MEngine(MModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight,
                 const ExternalNetworkDx* external)
    : impl(std::make_unique<Impl>(*model.impl)) {
    auto& e = *impl; auto& m = e.model;
    if (renderWidth < 64 || renderHeight < 64 || outputWidth % 2 || outputHeight % 2 || outputWidth > 16384 || outputHeight > 16384 ||
        renderWidth > outputWidth || renderHeight > outputHeight)
        throw std::runtime_error("unsupported preset M dimensions");
    if (external) {
        if (!external->tokens || !external->head) throw std::runtime_error("an external network needs its tokens and head resources");
        if (!m.enc0Planes || !m.dec0Planes)
            throw std::runtime_error("the model has no plane-layout stages for an external network (compile_m.py --dxil yes)");
        e.external = true; e.externalTokens = external->tokens; e.externalHead = external->head;
    }
    model.upload();
    e.maxRw = renderWidth; e.maxRh = renderHeight; e.ow = outputWidth; e.oh = outputHeight;
    e.hw = outputWidth / 2 * 3; e.hh = outputHeight / 2 * 3;
    const uint32_t gw = (renderWidth + 31) / 32 * 32, gh = (renderHeight + 31) / 32 * 32, tw = gw / 2, th = gh / 2;
    e.enc0P = e.buf(sizeof(MEnc0Params)); e.dec0P = e.buf(sizeof(MDec0Params)); e.postP = e.buf(sizeof(MPostParams));
    e.downP = e.buf(sizeof(MDownParams)); e.expP = e.buf(sizeof(MExposureParams)); e.layerP = e.buf(512 * (MLayers + 2));
    e.expSums = e.buf(UINT64((std::max(outputWidth / 2, 256u) + 31) / 32) * ((std::max(outputHeight / 2, 256u) + 31) / 32) * 4);
    e.expState = e.buf(8); e.dummyIn = e.buf(256); e.dummyOut = e.buf(256); e.ags = e.buf(256);
    e.packed = e.buf(UINT64(e.hw) * e.hh * 4);
    auto tok = [&](unsigned scale, unsigned channels) { return e.buf(UINT64(tw / scale) * (th / scale) * channels + 64); };
    if (!e.external) {
        e.tokens = tok(1, 64); e.skip1 = tok(1, 64); e.merged1 = tok(2, 96); e.skip2 = tok(2, 96); e.merged2 = tok(4, 128);
        e.tubeA = tok(4, 128); e.tubeB = tok(4, 128); e.dec2Out = tok(2, 96); e.dec1Out = tok(1, 64);
    }
    // L: the pixel-resolution block output [gh][gw][32] and the feature slots of every 8x8 block of the input stage's
    // grid ((gw + 9) / 8 x (gh + 9) / 8 blocks: shifts of up to two pixels), 32 bytes a pixel
    e.skipL = e.buf(m.presetL ? UINT64(gw) * gh * 32 + 64 : 256);
    e.featsL = e.buf(m.presetL ? UINT64((gw + 9) / 8) * ((gh + 9) / 8) * 64 * 32 : 256);
    e.dummyOut2 = e.buf(256);
    for (unsigned i = 0; i < 2; ++i) {
        e.mvA[i] = e.tex(renderWidth, renderHeight, DXGI_FORMAT_R16G16_FLOAT);
        e.feat[i] = e.tex(renderWidth, renderHeight, DXGI_FORMAT_R8G8B8A8_SNORM);
        e.hist[i] = e.tex(e.hw, e.hh, DXGI_FORMAT_R16G16B16A16_FLOAT);
        e.half[i] = e.tex(e.hw / 2, e.hh / 2, DXGI_FORMAT_R32G32B32A32_FLOAT);
    }
    e.mvB = e.tex(renderWidth, renderHeight, DXGI_FORMAT_R16G16_FLOAT); e.mvC = e.tex(renderWidth, renderHeight, DXGI_FORMAT_R16G16_FLOAT);
    e.gates = e.tex(renderWidth, renderHeight, DXGI_FORMAT_R8G8B8A8_UNORM); e.mix = e.tex(renderWidth, renderHeight, DXGI_FORMAT_R8G8B8A8_UNORM);
    e.cov = e.tex(renderWidth, renderHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
    e.rest[m.w8.Get()] = e.rest[m.w16.Get()] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, Impl::BlockBase + Impl::Blocks * Impl::BlockSlots,
                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    DXCHECK(e.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&e.heap)));
    e.increment = e.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto s = [](const ComPtr<ID3D12Resource>& r, UINT64 offset = 0) { return Slot{r.Get(), offset}; };
    // the network: tokens -> enc1 -> enc2 -> six tube blocks (ping-pong) -> dec2 -> dec1 (an external network's own
    // buffers stand in for the tokens and the head, and its layers are not dispatched here)
    if (!e.external) {
        ID3D12Resource* in[MLayers] = {e.tokens.Get(), e.merged1.Get(), e.merged2.Get(), e.tubeA.Get(), e.tubeB.Get(), e.tubeA.Get(), e.tubeB.Get(),
                                       e.tubeA.Get(), e.tubeB.Get(), e.dec2Out.Get()};
        ID3D12Resource* skip[MLayers] = {e.dummyIn.Get(), e.dummyIn.Get(), e.dummyIn.Get(), e.dummyIn.Get(), e.dummyIn.Get(), e.dummyIn.Get(),
                                         e.dummyIn.Get(), e.dummyIn.Get(), e.skip2.Get(), e.skip1.Get()};
        ID3D12Resource* out[MLayers] = {e.skip1.Get(), e.skip2.Get(), e.tubeA.Get(), e.tubeB.Get(), e.tubeA.Get(), e.tubeB.Get(), e.tubeA.Get(),
                                        e.tubeB.Get(), e.dec2Out.Get(), e.dec1Out.Get()};
        ID3D12Resource* merged[MLayers] = {e.merged1.Get(), e.merged2.Get(), e.dummyOut.Get(), e.dummyOut.Get(), e.dummyOut.Get(),
                                           e.dummyOut.Get(), e.dummyOut.Get(), e.dummyOut.Get(), e.dummyOut.Get(), e.dummyOut.Get()};
        for (unsigned l = 0; l < MLayers; ++l)
            e.table(Impl::LayerBase + 7 * l, LayerB, {s(e.layerP, 512 * l), s(m.w8, m.layers[l][0]), s(m.w16, m.layers[l][1]), Slot{in[l]},
                                                      Slot{skip[l]}, Slot{out[l]}, Slot{merged[l]}});
    }
    // (with an external network, the tokens and the head are its buffers: L's plane-layout block and expansion take these)
    if (m.presetL)
        e.table(Impl::Enc0LBase, LayerB, {s(e.layerP, 512 * MLayers), s(m.w8, m.lEnc0[0]), s(m.w16, m.lEnc0[1]), s(e.featsL), s(e.dummyIn),
                                          s(e.skipL), e.tokenSlot()});
    for (unsigned i = 0; i < 2; ++i) {
        const unsigned prev = 1 - i;
        if (m.presetL)
            e.table(Impl::Dec0Base + Impl::Dec0Slots * i, Dec0LB, {s(e.layerP, 512 * (MLayers + 1)), s(m.w8, m.lDec0[0]), s(m.w16, m.lDec0[1]),
                                                                   e.headSlot(), s(e.skipL), s(e.dummyOut), s(e.dummyOut2), s(e.dec0P),
                                                                   s(e.feat[i]), s(e.gates), s(e.mix), s(e.cov)});
        else
            e.table(Impl::Dec0Base + Impl::Dec0Slots * i, Dec0B, {s(e.dec0P), s(e.mvA[i]), s(e.feat[prev]), s(e.feat[i]), s(e.gates), s(e.mix),
                                                                  s(e.cov), e.headSlot(), s(m.w8, m.dec0W), s(m.w16, m.dec0Off[1])});
    }
    e.table(Impl::Exp1Base, Exp1B, {s(e.expP), s(e.expSums), s(e.expState), s(e.enc0P), s(e.postP), s(e.downP)});

    // the layers' tensor offsets, once (each frame writes the grid and window shift before them)
    std::vector<char> blocks(512 * (MLayers + 2));
    for (unsigned l = 0; l < MLayers + 2; ++l) {
        const auto& o = l < MLayers ? m.layers[l] : l == MLayers ? m.lEnc0 : m.lDec0;
        MLayerParams lp{l < MLayers ? int32_t(tw / MScale[l]) : int32_t(gw), l < MLayers ? int32_t(th / MScale[l]) : int32_t(gh),
                        l < MLayers ? MShift[l] : 0, l < MLayers ? MShift[l] : 0, {}};
        std::memcpy(lp.o, o.data() + 2, sizeof(lp.o));
        std::memcpy(blocks.data() + 512 * l, &lp, sizeof(lp));
    }
    auto up = buffer(e.device, blocks.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr; D3D12_RANGE none{0, 0};
    DXCHECK(up->Map(0, &none, &mapped)); std::memcpy(mapped, blocks.data(), blocks.size()); up->Unmap(0, nullptr);
    m.execute([&](ID3D12GraphicsCommandList* list) {
        auto b = transition(e.layerP.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        list->ResourceBarrier(1, &b);
        list->CopyBufferRegion(e.layerP.Get(), 0, up.Get(), 0, blocks.size());
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        list->ResourceBarrier(1, &b);
    });
}
MEngine::~MEngine() = default;

void MEngine::setFrame(const FrameTextures& textures, const MFrameSettings& s, FrameOutput output) {
    auto& e = *impl; auto& m = e.model;
    if (e.recorded != SIZE_MAX) throw std::runtime_error("the previous recording was neither submitted nor discarded");
    if (!textures.color.resource || !textures.motion.resource || !textures.depth.resource || !output.resource)
        throw std::runtime_error("frame has a null texture");
    const uint32_t rw = s.renderWidth ? s.renderWidth : e.maxRw, rh = s.renderHeight ? s.renderHeight : e.maxRh;
    if (rw < 64 || rh < 64 || rw > e.maxRw || rh > e.maxRh) throw std::runtime_error("render size exceeds the engine's capacity or is below 64");
    if (s.displayMotion && !m.displayMotion) throw std::runtime_error("display-resolution motion needs an M model rebuilt by the current compile_m.py");
    const bool reset = s.reset || e.frame == 0 || rw != e.rw || rh != e.rh;
    e.rw = rw; e.rh = rh;
    e.gw = (rw + 31) / 32 * 32; e.gh = (rh + 31) / 32 * 32;
    e.quality = uint64_t(e.hw) < 3ull * rw || uint64_t(e.hh) < 3ull * rh;
    e.compact = e.quality && 20ull * rw <= 9ull * e.hw && 20ull * rh <= 9ull * e.hh;
    // (runtime_m.inc setFrame)
    const float fx = float(e.hw) / float(rw), fy = float(e.hh) / float(rh);
    const float mx = s.displayMotion ? float(e.hw) / float(e.ow) : fx, my = s.displayMotion ? float(e.hh) / float(e.oh) : fy;
    e.enc0 = {{int32_t(rw), int32_t(rh)}, {2, 2}, {int32_t(e.gw / 2), int32_t(e.gh / 2)}, {s.jitter[0], s.jitter[1]}, {e.prevJitter[0], e.prevJitter[1]},
              {s.motionScale[0] * mx, s.motionScale[1] * my}, {s.motionOffset[0], s.motionOffset[1]}, {fx, fy}, {1.f / fx, 1.f / fy},
              {float(e.hw), float(e.hh)}, m.consts[0], m.consts[1], s.depthInverted ? 1u : 0u, reset ? 1u : 0u, e.frame & 7u,
              1.f, 0.f, {1.f / float(e.hw), 1.f / float(e.hh)}, {1.f / float(e.maxRw), 1.f / float(e.maxRh)},
              s.displayMotion ? 1u : 0u, {int32_t(e.ow) - 1, int32_t(e.oh) - 1}, {float(e.ow) / float(rw), float(e.oh) / float(rh)}};
    e.dec0 = {{int32_t(rw), int32_t(rh)}, int32_t(e.gw / 2), reset ? 1u : 0u, {s.jitter[0], s.jitter[1]}, {fx, fy}, {float(e.hw), float(e.hh)},
              {e.prevJitter[0], e.prevJitter[1]}, m.consts[2], m.consts[3], {1.f / float(e.maxRw), 1.f / float(e.maxRh)}};
    // L's input stage alternates its horizontal window shift every frame, (2, 2) / (0, 2); its feature slots follow it
    if (m.presetL) e.enc0.shift[0] = 2 ^ int32_t((e.frame & 1u) * 2u);
    e.post = {{0, 0}, {int32_t(e.hw), int32_t(e.hh)}, {int32_t(rw), int32_t(rh)}, {1.f / fx, 1.f / fy}, {s.jitter[0], s.jitter[1]},
              {1.f / float(e.maxRw), 1.f / float(e.maxRh)}, m.consts[4], reset ? 1u : 0u, 1.f, 0.f,
              {s.displayMotion ? int32_t(e.ow) : 0, s.displayMotion ? int32_t(e.oh) : 0}, {e.enc0.mvScale[0], e.enc0.mvScale[1]},
              {s.motionOffset[0], s.motionOffset[1]}};
    e.down = {{int32_t(e.hw), int32_t(e.hh)}, {0, 0}, {0, 0}, {int32_t(rw), int32_t(rh)}, {float(rw) / float(e.ow), float(rh) / float(e.oh)},
              {s.jitter[0], s.jitter[1]}, 1.f};
    const uint32_t sw = exposureGrid(rw, e.ow), sh = exposureGrid(rh, e.oh), bx = (sw + 31) / 32, by = (sh + 31) / 32;
    e.exposure = {{0, 0}, {int32_t(rw), int32_t(rh)}, 1.f / (float(sw) * float(sh)), 1.f, 0.23547087609767914f, 1u | (reset ? 2u : 0u),
                  bx * by, bx, {int32_t(sw), int32_t(sh)}};
    e.savedJitter[0] = e.prevJitter[0]; e.savedJitter[1] = e.prevJitter[1];
    e.prevJitter[0] = s.jitter[0]; e.prevJitter[1] = s.jitter[1];
    const DXGI_FORMAT outFormat = viewFormat(FrameTexture{output.resource, output.format});
    e.rgb10 = outFormat == DXGI_FORMAT_R10G10B10A2_UNORM && m.downRgb10;

    auto same = [](const FrameTexture& a, const FrameTexture& b) { return a.resource == b.resource && a.format == b.format; };
    auto found = std::find_if(e.bindings.begin(), e.bindings.end(), [&](const Impl::Bindings& b) {
        return same(b.textures.color, textures.color) && same(b.textures.motion, textures.motion) && same(b.textures.depth, textures.depth) &&
               b.output.resource == output.resource && b.output.format == output.format;
    });
    if (found == e.bindings.end()) {
        e.retire();
        UINT block = UINT(e.bindings.size());
        if (e.bindings.size() >= Impl::Blocks) {
            // reuse the least recently used block whose frames have all retired
            auto victim = e.bindings.end();
            for (auto it = e.bindings.begin(); it != e.bindings.end(); ++it)
                if (!it->inFlight && !it->pending && (victim == e.bindings.end() || it->lastUse < victim->lastUse)) victim = it;
            if (victim == e.bindings.end()) throw std::runtime_error("too many distinct frame resources in flight");
            block = victim->block;
            e.bindings.erase(victim);
        }
        Impl::Bindings b;
        b.textures = textures; b.output = output; b.block = block;
        ID3D12Resource* keep[] = {textures.color.resource, textures.motion.resource, textures.depth.resource, output.resource};
        for (unsigned i = 0; i < 4; ++i) b.retained[i] = keep[i];
        e.writeBindings(b);
        e.bindings.push_back(std::move(b));
        found = e.bindings.end() - 1;
    }
    e.current = size_t(found - e.bindings.begin());
    e.frameSet = true;
}

void MEngine::recordFrame(ID3D12GraphicsCommandList* list) {
    auto& e = *impl; auto& m = e.model;
    if (e.external) throw std::runtime_error("this engine's network is external: record the front and the back");
    if (!e.frameSet) throw std::runtime_error("setFrame before recording");
    if (e.recorded != SIZE_MAX) throw std::runtime_error("the previous recording was neither submitted nor discarded");
    ComPtr<ID3D12GraphicsCommandList2> list2;
    DXCHECK(list->QueryInterface(IID_PPV_ARGS(&list2)));
    auto& b = e.bindings[e.current];
    const UINT base = Impl::BlockBase + b.block * Impl::BlockSlots;
    const unsigned i = e.parity;
    e.now = e.rest;
    ID3D12DescriptorHeap* heaps[] = {e.heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    // the frame's parameters, in the command list (frames may be recorded while earlier ones are queued)
    std::vector<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> values;
    auto put = [&](ID3D12Resource* r, UINT64 offset, const void* data, size_t bytes) {
        e.use(r, D3D12_RESOURCE_STATE_COPY_DEST);
        const auto at = r->GetGPUVirtualAddress() + offset;
        for (size_t k = 0; k < bytes; k += 4) { UINT32 v; std::memcpy(&v, static_cast<const char*>(data) + k, 4); values.push_back({at + k, v}); }
    };
    put(e.enc0P.Get(), 0, &e.enc0, sizeof(e.enc0)); put(e.dec0P.Get(), 0, &e.dec0, sizeof(e.dec0));
    put(e.postP.Get(), 0, &e.post, sizeof(e.post)); put(e.downP.Get(), 0, &e.down, sizeof(e.down));
    put(e.expP.Get(), 0, &e.exposure, sizeof(e.exposure));
    for (unsigned l = 0; l < MLayers; ++l) {
        const int32_t grid[4] = {int32_t(e.gw / 2 / MScale[l]), int32_t(e.gh / 2 / MScale[l]), MShift[l] ^ int32_t((e.frame & 1u) * 2u), MShift[l]};
        put(e.layerP.Get(), 512 * l, grid, sizeof(grid));
    }
    if (m.presetL) {
        // L: the input stage (2, 2) / (0, 2), the expansion (0, 0) / (2, 0), on the 32-aligned pixel grid
        const int32_t lin[4] = {int32_t(e.gw), int32_t(e.gh), 2 ^ int32_t((e.frame & 1u) * 2u), 2};
        const int32_t lex[4] = {int32_t(e.gw), int32_t(e.gh), int32_t((e.frame & 1u) * 2u), 0};
        put(e.layerP.Get(), 512 * MLayers, lin, sizeof(lin)); put(e.layerP.Get(), 512 * (MLayers + 1), lex, sizeof(lex));
    }
    e.flush(list);
    list2->WriteBufferImmediate(UINT(values.size()), values.data(), nullptr);
    // earlier work on the game's textures and the engine's resources (a previous frame) completes first
    D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    e.barriers.push_back(uav);
    if (!m.ldr) {
        e.dispatch(list, m.exp0, base + Impl::Exp0Slot, e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX);
        e.dispatch(list, m.exp1, Impl::Exp1Base, 1, 1);
    }
    if (m.enc0LX) e.dispatch(list, m.enc0LX, base + Impl::Enc0LXSlot + 21 * i, (e.gw + 9) / 8, (e.gh + 9) / 8);
    else {
        e.dispatch(list, m.enc0, base + Impl::Enc0Slot + 14 * i, (e.gw + 2 + 7) / 8, (e.gh + 2 + 7) / 8);
        if (m.presetL) e.dispatch(list, m.enc0L, Impl::Enc0LBase, (e.gw + 9) / 8, (e.gh + 9) / 8);
    }
    for (unsigned l = 0; l < MLayers; ++l) {
        const uint32_t w = e.gw / 2 / MScale[l], h = e.gh / 2 / MScale[l];
        const uint32_t gx = (w + (MShift[l] ^ ((e.frame & 1u) * 2u)) + 7) / 8, gy = (h + MShift[l] + 7) / 8;
        const bool w2 = m.tubeW2.state && MScale[l] == 4 && gx * gy >= m.tubeW2Groups;
        e.dispatch(list, w2 ? m.tubeW2 : m.network[l], Impl::LayerBase + 7 * l, gx, gy);
    }
    // L: the blocks with a pixel of the render size, one more in x on frames whose expansion shift is 2
    if (m.presetL) e.dispatch(list, m.dec0, Impl::Dec0Base + Impl::Dec0Slots * i, (e.rw + (e.frame & 1u) * 2u + 7) / 8, (e.rh + 7) / 8);
    else e.dispatch(list, m.dec0, Impl::Dec0Base + Impl::Dec0Slots * i, e.gw / 8, e.gh / 8);
    const bool compact = e.compact && e.post.rndOffs[0] >= -0.5f && e.post.rndOffs[0] <= 0.5f && e.post.rndOffs[1] >= -0.5f && e.post.rndOffs[1] <= 0.5f;
    e.dispatch(list, compact ? m.postC : e.quality ? m.postQ : m.post, base + Impl::PostSlot + 13 * i, (e.hw + 15) / 16, (e.hh + 15) / 16);
    e.dispatch(list, e.rgb10 ? m.downRgb10 : m.down, base + Impl::DownSlot, (e.ow + 7) / 8, (e.oh + 63) / 64);
    // every engine resource back to its resting state
    for (auto& [r, state] : e.now)
        if (state != e.rest.at(r)) e.barriers.push_back(transition(r, state, e.rest.at(r)));
    e.flush(list);
    b.pending = true; b.lastUse = ++e.uses;
    e.recorded = e.current;
    e.parity = 1 - e.parity; ++e.frame;
}

// The front of an external network's frame (../runtime_m.inc recordFront): the parameters, the history's reset and the
// exposure and input stages, which write the caller's tokens. The engine's own resources are back in their resting
// states when this returns; the tokens are in COMMON and are not transitioned (hip_net_d3d12.h).
void MEngine::recordFront(ID3D12GraphicsCommandList* list) {
    auto& e = *impl; auto& m = e.model;
    if (!e.external) throw std::runtime_error("recordFront needs an engine with an external network");
    if (!e.frameSet) throw std::runtime_error("setFrame before recording");
    if (e.recorded != SIZE_MAX) throw std::runtime_error("the previous recording was neither submitted nor discarded");
    if (e.frontDone) throw std::runtime_error("recordFront before recordBack");
    ComPtr<ID3D12GraphicsCommandList2> list2;
    DXCHECK(list->QueryInterface(IID_PPV_ARGS(&list2)));
    auto& b = e.bindings[e.current];
    const UINT base = Impl::BlockBase + b.block * Impl::BlockSlots;
    const unsigned i = e.parity;
    e.now = e.rest;
    ID3D12DescriptorHeap* heaps[] = {e.heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    std::vector<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> values;
    auto put = [&](ID3D12Resource* r, UINT64 offset, const void* data, size_t bytes) {
        e.use(r, D3D12_RESOURCE_STATE_COPY_DEST);
        const auto at = r->GetGPUVirtualAddress() + offset;
        for (size_t k = 0; k < bytes; k += 4) { UINT32 v; std::memcpy(&v, static_cast<const char*>(data) + k, 4); values.push_back({at + k, v}); }
    };
    put(e.enc0P.Get(), 0, &e.enc0, sizeof(e.enc0)); put(e.dec0P.Get(), 0, &e.dec0, sizeof(e.dec0));
    put(e.postP.Get(), 0, &e.post, sizeof(e.post)); put(e.downP.Get(), 0, &e.down, sizeof(e.down));
    put(e.expP.Get(), 0, &e.exposure, sizeof(e.exposure));
    if (m.presetL) {
        // L's input stage block and expansion grids (as recordFrame); the network's own layers are external
        const int32_t lin[4] = {int32_t(e.gw), int32_t(e.gh), 2 ^ int32_t((e.frame & 1u) * 2u), 2};
        const int32_t lex[4] = {int32_t(e.gw), int32_t(e.gh), int32_t((e.frame & 1u) * 2u), 0};
        put(e.layerP.Get(), 512 * MLayers, lin, sizeof(lin)); put(e.layerP.Get(), 512 * (MLayers + 1), lex, sizeof(lex));
    }
    e.flush(list);
    list2->WriteBufferImmediate(UINT(values.size()), values.data(), nullptr);
    D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    e.barriers.push_back(uav);
    if (!m.ldr) {
        e.dispatch(list, m.exp0, base + Impl::Exp0Slot, e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX);
        e.dispatch(list, m.exp1, Impl::Exp1Base, 1, 1);
    }
    if (m.presetL) {
        // L: the feature slots, then the block and merge in the plane layout, which write the network's tokens
        e.dispatch(list, m.enc0, base + Impl::Enc0Slot + 14 * i, (e.gw + 2 + 7) / 8, (e.gh + 2 + 7) / 8);
        e.dispatch(list, m.enc0Planes, Impl::Enc0LBase, (e.gw + 9) / 8, (e.gh + 9) / 8);
    } else
        e.dispatch(list, m.enc0Planes, base + Impl::Enc0Slot + 14 * i, (e.gw + 2 + 7) / 8, (e.gh + 2 + 7) / 8);
    for (auto& [r, state] : e.now)
        if (state != e.rest.at(r)) e.barriers.push_back(transition(r, state, e.rest.at(r)));
    e.flush(list);
    e.frontDone = true;
}

// The back (../runtime_m.inc recordBack): the expansion (which reads the caller's head), the reconstruction and the
// downsample. Record and submit it only after the external network has finished; the head is in COMMON.
void MEngine::recordBack(ID3D12GraphicsCommandList* list) {
    auto& e = *impl; auto& m = e.model;
    if (!e.external) throw std::runtime_error("recordBack needs an engine with an external network");
    if (!e.frontDone) throw std::runtime_error("recordFront before recordBack");
    auto& b = e.bindings[e.current];
    const UINT base = Impl::BlockBase + b.block * Impl::BlockSlots;
    const unsigned i = e.parity;
    e.now = e.rest;
    ID3D12DescriptorHeap* heaps[] = {e.heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    // the external network wrote the head after the front retired on the CPU: this UAV barrier invalidates the caches
    // the expansion reads it through
    D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &uav);
    if (m.presetL) e.dispatch(list, m.dec0Planes, Impl::Dec0Base + Impl::Dec0Slots * i, (e.rw + (e.frame & 1u) * 2u + 7) / 8, (e.rh + 7) / 8);
    else e.dispatch(list, m.dec0Planes, Impl::Dec0Base + Impl::Dec0Slots * i, e.gw / 8, e.gh / 8);
    const bool compact = e.compact && e.post.rndOffs[0] >= -0.5f && e.post.rndOffs[0] <= 0.5f && e.post.rndOffs[1] >= -0.5f && e.post.rndOffs[1] <= 0.5f;
    e.dispatch(list, compact ? m.postC : e.quality ? m.postQ : m.post, base + Impl::PostSlot + 13 * i, (e.hw + 15) / 16, (e.hh + 15) / 16);
    e.dispatch(list, e.rgb10 ? m.downRgb10 : m.down, base + Impl::DownSlot, (e.ow + 7) / 8, (e.oh + 63) / 64);
    for (auto& [r, state] : e.now)
        if (state != e.rest.at(r)) e.barriers.push_back(transition(r, state, e.rest.at(r)));
    e.flush(list);
    b.pending = true; b.lastUse = ++e.uses;
    e.recorded = e.current;
    e.parity = 1 - e.parity; ++e.frame;
    e.frontDone = false;
}

void MEngine::submitted(ID3D12Fence* fence, uint64_t value) {
    auto& e = *impl;
    if (e.recorded == SIZE_MAX) throw std::runtime_error("no recording to report");
    auto& b = e.bindings[e.recorded];
    b.pending = false; b.inFlight = true; b.fence = fence; b.value = fence ? value : e.uses;
    e.recorded = SIZE_MAX; e.frameSet = false;
}
void MEngine::discarded() {
    auto& e = *impl;
    if (e.recorded == SIZE_MAX) throw std::runtime_error("no recording to report");
    e.bindings[e.recorded].pending = false;
    e.recorded = SIZE_MAX; e.frameSet = false;
    // the history stays at the last submitted frame
    e.parity = 1 - e.parity; --e.frame;
    e.frontDone = false;
    e.prevJitter[0] = e.savedJitter[0]; e.prevJitter[1] = e.savedJitter[1];
}
bool MEngine::idle() const {
    auto& e = *impl;
    if (e.recorded != SIZE_MAX) return false;
    e.retire();
    return std::none_of(e.bindings.begin(), e.bindings.end(), [](const Impl::Bindings& b) { return b.inFlight; });
}
} // namespace d4r::dx
