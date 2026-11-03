// RDNA 3 M/L native frame implementation (shaders in <package>/rdna3). Included inside d4r::dx (as d4r::dx::rdna3), sharing the K
// backend's D3D12 allocation, pipeline, descriptor-heap, and transition helpers.
namespace rdna3 {
namespace {
using namespace d4r::m_detail;
constexpr auto MRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
constexpr auto MWrite = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
struct MResource {
    ComPtr<ID3D12Resource> resource;
    D3D12_RESOURCE_STATES state = MRead;
    void to(ID3D12GraphicsCommandList* list, D3D12_RESOURCE_STATES next) {
        if (state != next) { auto b = transition(resource.Get(), state, next); list->ResourceBarrier(1, &b); state = next; }
    }
};
bool mSrv(Bind kind) { return kind == Bind::Texture || kind == Bind::Raw || kind == Bind::Structured || kind == Bind::Typed || kind == Bind::Typed4; }
struct MView { MResource* resource = nullptr; UINT64 offset = 0, bytes = 0; DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN; };
struct MSet { const Pipeline* pipeline = nullptr; UINT table = 0; std::vector<MView> views; };
const Binding MT{Bind::Typed}, MT4{Bind::Typed4}, MU{Bind::RwTyped}, MU4{Bind::RwTyped4}, MI{Bind::RwTexture}, MR{Bind::Raw};
std::vector<Binding> mLayerBindings(bool merge = false, bool cbv = false) {
    if (merge) return {cbv ? N : MT, MR, MR, MT, MU};
    return {cbv ? N : MT, MR, MR, MT, MT, MU};
}
std::vector<Binding> mInputBindings(bool l, bool wmma) {
    if (l) return {rw(sizeof(MEnc0Params)), T, T, T, TL, TL, TL, MI, MI, MI, MR, MU};
    return {rw(sizeof(MEnc0Params)), T, T, T, TL, TL, TL, MI, MI, MI, wmma ? MR : MT, wmma ? MR : MT, MR, MU};
}
std::vector<Binding> mExpandBindings(bool l) {
    if (l) return {rw(sizeof(MLayerParams)), MT, MT, MT, MT, MU, N, rw(sizeof(MDec0Params)), MI, MI, MI, MI};
    return {rw(sizeof(MDec0Params)), T, TL, MI, MI, MI, MI, MT, MT, MT};
}
std::vector<Binding> mPostBindings() { return {rw(sizeof(MPostParams)), T, T, T, T, TL, T, T, TL, MI, MI, MU, T}; }
std::vector<Binding> mDownBindings() { return {rw(sizeof(MDownParams)), T, MI, MT}; }
void mExecute(ID3D12Device* device, const std::function<void(ID3D12GraphicsCommandList*)>& record) {
    ComPtr<ID3D12CommandQueue> queue; ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list; ComPtr<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    DXCHECK(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)));
    DXCHECK(device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(&allocator)));
    DXCHECK(device->CreateCommandList(0, desc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
    DXCHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    record(list.Get()); DXCHECK(list->Close()); ID3D12CommandList* commands[] = {list.Get()}; queue->ExecuteCommandLists(1, commands);
    DXCHECK(queue->Signal(fence.Get(), 1));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("M/L setup: CreateEvent failed");
    const HRESULT result = fence->SetEventOnCompletion(1, event);
    const DWORD waited = SUCCEEDED(result) ? WaitForSingleObject(event, 30000) : WAIT_FAILED;
    CloseHandle(event); DXCHECK(result);
    if (waited != WAIT_OBJECT_0) throw std::runtime_error("M/L setup GPU wait timed out");
    DXCHECK(device->GetDeviceRemovedReason());
}
void mWrite(ID3D12GraphicsCommandList* list, MResource& target, const void* data, UINT bytes) {
    ComPtr<ID3D12GraphicsCommandList2> list2; DXCHECK(list->QueryInterface(IID_PPV_ARGS(&list2)));
    target.to(list, D3D12_RESOURCE_STATE_COPY_DEST);
    std::vector<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> values(bytes / 4);
    for (UINT i = 0; i < bytes / 4; ++i) { uint32_t word; std::memcpy(&word, static_cast<const char*>(data) + 4 * i, 4); values[i] = {target.resource->GetGPUVirtualAddress() + 4 * i, word}; }
    list2->WriteBufferImmediate(UINT(values.size()), values.data(), nullptr);
    target.to(list, MRead);
}
} // namespace

struct MModel::Impl {
    ComPtr<ID3D12Device> device;
    MResource matrixWeights, dotWeights, vectors;
    std::vector<ComPtr<ID3D12Resource>> uploads;
    std::array<std::array<uint32_t, 74>, MLayers> layers{};
    std::array<uint32_t, 74> lEnc0{}, lDec0{};
    uint32_t enc0[3]{}, dec0[2]{}, dither = 0;
    float constants[5]{};
    bool l = false, ldr = false, cbv = false, split = false, inputWmma = false, vectorIo = false;
    std::array<Pipeline, MLayers> network;
    std::array<Pipeline, 2> merge, exposure, patch;
    Pipeline lPatch;
    Pipeline input, inputLBlock, expand, headL, post, postQuality, postCompact, down, downRgb10, mergeL;
};
std::string MModel::missingShaders(const std::string& directory, bool l) {
    std::string missing;
    auto need = [&](const std::string& name) { if (!fileExists(directory + "/rdna3/" + name + ".dxil")) missing += (missing.empty() ? "" : ", ") + name + ".dxil"; };
    for (const char* name : {"m_enc1", "m_enc2", "m_tube", "m_dec2", "m_dec1", "m_merge1", "m_merge2", "down_m", "down_m_rgb10", "exposure_m0", "exposure_m1"}) need(name);
    const std::string suffix = l ? "_l" : "_m";
    need("enc0" + suffix); need("dec0" + suffix); need("post" + suffix); need("post" + suffix + "_q"); need("post" + suffix + "_c");
    if (l) { need("merge_l"); need("enc0_l_network"); need("dec0_l_head"); }
    return missing;
}
MModel::MModel(ID3D12Device* device, const std::string& directory) : impl(std::make_unique<Impl>()) {
    auto& m = *impl; m.device = device; m.ldr = fileExists(directory + "/ldr");
    const auto head = readFile(directory + "/offsets.bin");
    m.l = head.size() >= 8 && !std::memcmp(head.data(), "D4RL0002", 8);
    if ((!m.l && (head.size() < 8 || std::memcmp(head.data(), "D4RM0007", 8))) ||
        head.size() != (m.l ? 40 + (MLayers + 2) * 296 : 36 + MLayers * 296))
        throw std::runtime_error("native M/L requires a current D4RM0007 or D4RL0002 model");
    const auto marker = readFile(directory + "/rdna3/d3d12_ml.bin");
    m.vectorIo = marker.size() == 8 && !std::memcmp(marker.data(), "D4RML006", 8);
    m.inputWmma = (m.vectorIo && !m.l) || (marker.size() == 8 && !std::memcmp(marker.data(), "D4RML005", 8));
    m.split = m.vectorIo || m.inputWmma || (marker.size() == 8 && !std::memcmp(marker.data(), "D4RML004", 8));
    if (m.inputWmma && m.l) throw std::runtime_error("M WMMA input marker is invalid for L");
    m.cbv = m.split || (marker.size() == 8 && !std::memcmp(marker.data(), "D4RML003", 8));
    if (marker.size() != 8 || (!m.cbv && std::memcmp(marker.data(), "D4RML002", 8) && (m.l || std::memcmp(marker.data(), "D4RML001", 8)))) throw std::runtime_error("incompatible native M/L shader package");
    if (const auto missing = missingShaders(directory, m.l); !missing.empty()) throw std::runtime_error("native M/L shaders missing: " + missing);
    uint32_t n8, n16; std::memcpy(&n8, head.data() + 8, 4); std::memcpy(&n16, head.data() + 12, 4);
    auto b8 = readFile(directory + "/weights8.bin"), b16 = readFile(directory + "/weights16.bin");
    if (b8.size() != n8 || b16.size() != n16 || n16 % 64 || !n8 || !n16) throw std::runtime_error("invalid M/L weights size");
    if (m.l) {
        std::memcpy(&m.dither, head.data() + 16, 4); std::memcpy(m.constants, head.data() + 20, 20);
        std::memcpy(m.lEnc0.data(), head.data() + 40, 296); std::memcpy(m.layers.data(), head.data() + 336, MLayers * 296);
        std::memcpy(m.lDec0.data(), head.data() + 336 + MLayers * 296, 296);
        for (float v : m.constants) if (!(v > 0 && v < 100)) throw std::runtime_error("invalid L stage constants");
    } else {
        std::memcpy(m.enc0, head.data() + 16, 12); std::memcpy(m.dec0, head.data() + 28, 8); std::memcpy(m.layers.data(), head.data() + 36, MLayers * 296);
    }
    auto inside = [&](const auto& block) { if (block[0] >= n8 || block[1] >= n16 || block[0] % 64 || block[1] % 64) throw std::runtime_error("invalid M/L weight block"); };
    for (const auto& block : m.layers) inside(block);
    if (m.l) { inside(m.lEnc0); inside(m.lDec0); }
    auto dot = widenM(b8, m.layers, m.enc0, m.dec0, true, true, m.l ? &m.lEnc0 : nullptr, m.l ? &m.lDec0 : nullptr);
    std::vector<uint16_t> widened(b8.size());
    for (size_t i = 0; i < b8.size(); ++i) {
        const unsigned c = uint8_t(b8[i]), mag = c & 127, sign = (c & 128) << 8;
        unsigned bits = 0;
        if (mag >= 8) bits = (mag << 7) + (8u << 10);
        else if (mag) { unsigned e = 1; while (!((mag << e) & 8)) ++e; bits = (((mag << e) & 7) << 7) | ((9 - e) << 10); }
        widened[i] = uint16_t(sign | bits);
    }
    mExecute(device, [&](ID3D12GraphicsCommandList* list) {
        auto upload = [&](MResource& target, const void* data, size_t bytes) {
            target.resource = buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST); target.state = D3D12_RESOURCE_STATE_COPY_DEST;
            auto source = buffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            void* mapped; D3D12_RANGE none{0, 0}; DXCHECK(source->Map(0, &none, &mapped)); std::memcpy(mapped, data, bytes); source->Unmap(0, nullptr);
            list->CopyBufferRegion(target.resource.Get(), 0, source.Get(), 0, bytes); target.to(list, MRead); m.uploads.push_back(source);
        };
        upload(m.matrixWeights, widened.data(), widened.size() * 2); upload(m.dotWeights, dot.data(), dot.size()); upload(m.vectors, b16.data(), b16.size());
    });
    m.uploads.clear();
    auto shader = [&](const std::string& name) { return readFile(directory + "/rdna3/" + name + ".dxil"); };
    for (unsigned i = 0; i < MLayers; ++i) {
        if (i > 2 && i < 8) m.network[i] = m.network[2];
        else {
            auto bindings = mLayerBindings(false, m.cbv);
            if (m.vectorIo) { bindings[3] = MT4; bindings[5] = MU4; }
            m.network[i].create(device, shader(MShader[i]), bindings, true, m.cbv, MShader[i]);
        }
    }
    for (unsigned i = 0; i < 2; ++i) {
        m.merge[i].create(device, shader("m_merge" + std::to_string(i + 1)), mLayerBindings(true, m.cbv), true, m.cbv);
        m.exposure[i].create(device, shader("exposure_m" + std::to_string(i)), i == 0 ? std::vector<Binding>{rw(sizeof(ExposureParams)), T, rw(4)} : std::vector<Binding>{rw(sizeof(ExposureParams)), rw(4), rw(8), rw(4), rw(4), rw(4)}, false);
    }
    if (m.split) {
        m.patch[0].create(device, shader("m_patch_dec2"), mLayerBindings(false, true), true, true, "m_patch_dec2");
        m.patch[1].create(device, shader("m_patch_dec1"), mLayerBindings(false, true), true, true, "m_patch_dec1");
        if (m.l) m.lPatch.create(device, shader("l_patch_dec0"), mLayerBindings(false, true), true, true, "l_patch_dec0");
    }
    const std::string suffix = m.l ? "_l" : "_m";
    m.input.create(device, shader("enc0" + suffix), mInputBindings(m.l, m.inputWmma), m.inputWmma, false, ("enc0" + suffix).c_str());
    m.expand.create(device, shader("dec0" + suffix), m.l ? mLayerBindings(false, m.cbv) : mExpandBindings(false), m.l, m.l && m.cbv, ("dec0" + suffix).c_str());
    if (m.l) {
        m.inputLBlock.create(device, shader("enc0_l_network"), mLayerBindings(false, m.cbv), true, m.cbv, "enc0_l_network");
        m.headL.create(device, shader("dec0_l_head"), {rw(sizeof(MDec0Params)), MT, MI, MI, MI, MI}, false, false, "dec0_l_head");
    }
    m.post.create(device, shader("post" + suffix), mPostBindings(), false, false, ("post" + suffix).c_str());
    m.postQuality.create(device, shader("post" + suffix + "_q"), mPostBindings(), false);
    m.postCompact.create(device, shader("post" + suffix + "_c"), mPostBindings(), false);
    m.down.create(device, shader("down_m"), mDownBindings(), false);
    m.downRgb10.create(device, shader("down_m_rgb10"), mDownBindings(), false);
    if (m.l) m.mergeL.create(device, shader("merge_l"), mLayerBindings(true, m.cbv), true, m.cbv);
}
MModel::~MModel() = default;
bool MModel::isPresetL() const { return impl->l; }
bool MModel::isLdr() const { return impl->ldr; }

struct GameUpscalerM::Impl {
    MModel::Impl& model;
    ID3D12Device* device;
    uint32_t ow, oh, maxRw, maxRh, hw, hh;
    uint32_t frame = 0, rw = 0, rh = 0, current = 0;
    float previousJitter[2]{};
    bool pending = false;
    d4r::MFrameSettings pendingSettings{};
    MResource agsDummy, empty, inputP, expandP, postP, downP, exposureP, exposureSums, exposureState, packed;
    std::array<MResource, MLayers> layerP;
    MResource expanded, encLP, decLP, pixelFeatures, pixelHead, tokens, skip1, merged1, skip2, merged2, tubeA, tubeB, dec2, dec1, skipL;
    std::array<MResource, 2> mvA, feature, history, half;
    MResource mvB, mvC, gates, mix, covariance;
    std::vector<MResource*> owned;
    Heap heap;
    UINT nextSlot = 0;
    std::array<MSet, MLayers> layerSets;
    std::array<MSet, 2> mergeSets, patchSets;
    MSet exposure1, inputLBlock, expandLBlock, mergeL, lPatchSet;
    struct Cached {
        std::array<MResource, 4> images;
        std::array<DXGI_FORMAT, 4> formats{};
        std::array<MSet, 2> input, expand, post;
        MSet exposure0, down;
    };
    std::vector<std::unique_ptr<Cached>> cached;
    struct Flight { ComPtr<ID3D12Fence> fence; uint64_t value, uses; };
    std::vector<Flight> flights;
    std::string captureDirectory;
    std::vector<std::unique_ptr<FrameCapture>> captures;
    FrameCapture* recordingCapture = nullptr;
    uint64_t uses = 0;
    Impl(MModel::Impl& m, uint32_t outputW, uint32_t outputH, uint32_t maxW, uint32_t maxH)
        : model(m), device(m.device.Get()), ow(outputW), oh(outputH), maxRw(maxW), maxRh(maxH), hw(outputW / 2 * 3), hh(outputH / 2 * 3) {}
    void allocate(MResource& target, UINT64 bytes) {
        target.resource = buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, MRead, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        owned.push_back(&target);
    }
    void image(MResource& target, UINT w, UINT h, DXGI_FORMAT format) {
        target.resource = texture(device, w, h, format); owned.push_back(&target);
    }
    void view(UINT slot, const Binding& binding, const MView& view) {
        auto* resource = view.resource ? view.resource->resource.Get() : nullptr;
        if (!resource) throw std::logic_error("M/L null descriptor resource");
        const auto desc = resource->GetDesc(); const auto cpu = heap.cpuHandle(slot);
        if (binding.kind == Bind::Texture) {
            D3D12_SHADER_RESOURCE_VIEW_DESC v{}; v.Format = view.format == DXGI_FORMAT_UNKNOWN ? readFormat(desc.Format) : view.format;
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; v.Texture2D.MipLevels = 1;
            device->CreateShaderResourceView(resource, &v, cpu); return;
        }
        if (binding.kind == Bind::RwTexture) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{}; v.Format = view.format == DXGI_FORMAT_UNKNOWN ? desc.Format : view.format; v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            device->CreateUnorderedAccessView(resource, nullptr, &v, cpu); return;
        }
        const UINT64 bytes = view.bytes ? view.bytes : desc.Width - view.offset;
        const bool raw = binding.kind == Bind::Raw || binding.kind == Bind::RwRaw;
        const bool typed4 = binding.kind == Bind::Typed4 || binding.kind == Bind::RwTyped4;
        const bool typed = typed4 || binding.kind == Bind::Typed || binding.kind == Bind::RwTyped;
        const UINT stride = raw ? 4 : typed ? (typed4 ? 16 : 4) : binding.stride;
        const DXGI_FORMAT format = raw ? DXGI_FORMAT_R32_TYPELESS : typed ? (typed4 ? DXGI_FORMAT_R32G32B32A32_UINT : DXGI_FORMAT_R32_UINT) : DXGI_FORMAT_UNKNOWN;
        if (!stride || view.offset % stride || bytes % stride) throw std::logic_error("M/L unaligned buffer descriptor");
        if (mSrv(binding.kind)) {
            D3D12_SHADER_RESOURCE_VIEW_DESC v{}; v.Format = format; v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            v.Buffer = {view.offset / stride, UINT(bytes / stride), raw || typed ? 0u : stride, raw ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE};
            device->CreateShaderResourceView(resource, &v, cpu);
        } else {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{}; v.Format = format; v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            v.Buffer = {view.offset / stride, UINT(bytes / stride), raw || typed ? 0u : stride, 0, raw ? D3D12_BUFFER_UAV_FLAG_RAW : D3D12_BUFFER_UAV_FLAG_NONE};
            device->CreateUnorderedAccessView(resource, nullptr, &v, cpu);
        }
    }
    MSet set(const Pipeline& pipeline, std::vector<MView> views) {
        if (views.size() != pipeline.bindings.size()) throw std::logic_error("M/L descriptor count mismatch");
        if (nextSlot + views.size() > heap.size) throw std::runtime_error("M/L descriptor cache exhausted; recreate the feature");
        MSet result{&pipeline, nextSlot, std::move(views)}; nextSlot += UINT(result.views.size());
        for (UINT i = 0; i < result.views.size(); ++i) if (pipeline.bindings[i].kind != Bind::None) view(result.table + i, pipeline.bindings[i], result.views[i]);
        return result;
    }
    void dispatch(ID3D12GraphicsCommandList* list, const MSet& set, UINT x, UINT y, const Pipeline* variant = nullptr) {
        const auto& pipeline = variant ? *variant : *set.pipeline;
        for (UINT i = 0; i < set.views.size(); ++i) {
            const auto kind = pipeline.bindings[i].kind;
            if (kind != Bind::None) set.views[i].resource->to(list, mSrv(kind) ? MRead : MWrite);
        }
        list->SetComputeRootSignature(pipeline.root.Get()); list->SetPipelineState(pipeline.state.Get());
        list->SetComputeRootDescriptorTable(0, heap.gpuHandle(set.table));
        if (pipeline.ags) list->SetComputeRootUnorderedAccessView(1, agsDummy.resource->GetGPUVirtualAddress());
        if (pipeline.layerCbv) {
            auto& parameters = *set.views[0].resource;
            parameters.to(list, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            list->SetComputeRootConstantBufferView(pipeline.ags ? 2 : 1, parameters.resource->GetGPUVirtualAddress());
        }
        list->Dispatch(x, y, 1); uavBarrier(list);
    }
    void retire() {
        for (auto it = captures.begin(); it != captures.end();) {
            auto& capture = **it;
            if (!capture.pending && (capture.fence ? capture.fence->GetCompletedValue() >= capture.value : uses >= capture.value + 16)) {
                for (const auto& readback : capture.data) readback.save();
                it = captures.erase(it);
            } else ++it;
        }
        flights.erase(std::remove_if(flights.begin(), flights.end(), [&](const Flight& f) {
            return f.fence ? f.fence->GetCompletedValue() >= f.value : uses >= f.uses + 16;
        }), flights.end());
    }
    Cached& bindings(const MGameFrame& frame);
};

GameUpscalerM::GameUpscalerM(MModel& model, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh)
    : impl(std::make_unique<Impl>(*model.impl, ow, oh, rw, rh)) {
    auto& e = *impl; auto& m = e.model;
    if (const char* directory = std::getenv("D4R_M_CAPTURE_DIR"); directory && *directory)
        e.captureDirectory = directory;
    if (rw < 64 || rh < 64 || rw > ow || rh > oh || ow % 2 || oh % 2 || ow > 16384 || oh > 16384)
        throw std::runtime_error("unsupported native M/L dimensions");
    e.heap.create(e.device, 32768);
    e.allocate(e.agsDummy, 256); e.allocate(e.empty, 256);
    e.allocate(e.inputP, sizeof(MEnc0Params)); e.allocate(e.expandP, sizeof(MDec0Params)); e.allocate(e.postP, sizeof(MPostParams));
    e.allocate(e.downP, sizeof(MDownParams)); e.allocate(e.exposureP, sizeof(ExposureParams)); e.allocate(e.exposureState, 16);
    e.allocate(e.exposureSums, UINT64((std::max(ow / 2, 256u) + 31) / 32) * ((std::max(oh / 2, 256u) + 31) / 32) * 4);
    e.allocate(e.packed, UINT64(e.hw) * e.hh * 4);
    const UINT tw = (rw + 31) / 32 * 16, th = (rh + 31) / 32 * 16;
    auto token = [&](MResource& r, UINT divisor, UINT channels) { e.allocate(r, UINT64(tw / divisor) * (th / divisor) * channels + 64); };
    token(e.tokens, 1, 64); token(e.skip1, 1, 64); token(e.merged1, 2, 96); token(e.skip2, 2, 96); token(e.merged2, 4, 128);
    if (m.split) e.allocate(e.expanded, m.l ? UINT64(2 * tw + 4) * (2 * th + 4) * 64 : UINT64(tw + 4) * (th + 4) * 128);
    token(e.tubeA, 4, 128); token(e.tubeB, 4, 128); token(e.dec2, 2, 96); token(e.dec1, 1, 64);
    if (m.l) { e.allocate(e.pixelFeatures, UINT64(2 * tw + 4) * (2 * th + 4) * 64 + 64); e.allocate(e.pixelHead, UINT64(2 * tw) * (2 * th) * 64 + 64); e.allocate(e.skipL, UINT64(2 * tw) * (2 * th) * 32 + 64); e.allocate(e.encLP, 512); e.allocate(e.decLP, 512); }
    for (unsigned i = 0; i < 2; ++i) {
        e.image(e.mvA[i], rw, rh, DXGI_FORMAT_R16G16_FLOAT); e.image(e.feature[i], rw, rh, DXGI_FORMAT_R8G8B8A8_SNORM);
        e.image(e.history[i], e.hw, e.hh, DXGI_FORMAT_R16G16B16A16_FLOAT);
        e.image(e.half[i], e.hw / 2, e.hh / 2, DXGI_FORMAT_R10G10B10A2_UNORM);
    }
    e.image(e.mvB, rw, rh, DXGI_FORMAT_R16G16_FLOAT); e.image(e.mvC, rw, rh, DXGI_FORMAT_R16G16_FLOAT);
    e.image(e.gates, rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM); e.image(e.mix, rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM);
    e.image(e.covariance, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    MResource* ins[MLayers] = {&e.tokens, &e.merged1, &e.merged2, &e.tubeA, &e.tubeB, &e.tubeA, &e.tubeB, &e.tubeA, &e.tubeB, &e.dec2};
    MResource* skips[MLayers] = {&e.empty, &e.empty, &e.empty, &e.empty, &e.empty, &e.empty, &e.empty, &e.empty, &e.skip2, &e.skip1};
    MResource* outs[MLayers] = {&e.skip1, &e.skip2, &e.tubeA, &e.tubeB, &e.tubeA, &e.tubeB, &e.tubeA, &e.tubeB, &e.dec2, &e.dec1};
    for (unsigned i = 0; i < MLayers; ++i) {
        e.allocate(e.layerP[i], 512);
        e.layerSets[i] = e.set(m.network[i], {{&e.layerP[i]}, {&m.matrixWeights, UINT64(m.layers[i][0]) * 2}, {&m.vectors, m.layers[i][1]}, {m.split && i >= 8 ? &e.expanded : ins[i]}, {skips[i]}, {outs[i]}});
        if (m.split && i >= 8) e.patchSets[i - 8] = e.set(m.patch[i - 8], {{&e.layerP[i]}, {&m.matrixWeights, UINT64(m.layers[i][0]) * 2}, {&m.vectors, m.layers[i][1]}, {ins[i]}, {skips[i]}, {&e.expanded}});
        if (i < 2) e.mergeSets[i] = e.set(m.merge[i], {{&e.layerP[i]}, {&m.matrixWeights, UINT64(m.layers[i][0]) * 2}, {&m.vectors, m.layers[i][1]}, {outs[i]}, {i == 0 ? &e.merged1 : &e.merged2}});
    }
    e.exposure1 = e.set(m.exposure[1], {{&e.exposureP}, {&e.exposureSums}, {&e.exposureState, 0, 8}, {&e.inputP}, {&e.postP}, {&e.downP}});
    if (m.l) {
        e.inputLBlock = e.set(m.inputLBlock, {{&e.encLP}, {&m.matrixWeights, UINT64(m.lEnc0[0]) * 2}, {&m.vectors, m.lEnc0[1]}, {&e.pixelFeatures}, {&e.empty}, {&e.skipL}});
        e.mergeL = e.set(m.mergeL, {{&e.encLP}, {&m.matrixWeights, UINT64(m.lEnc0[0]) * 2}, {&m.vectors, m.lEnc0[1]}, {&e.skipL}, {&e.tokens}});
        e.expandLBlock = e.set(m.expand, {{&e.decLP}, {&m.matrixWeights, UINT64(m.lDec0[0]) * 2}, {&m.vectors, m.lDec0[1]}, {m.split ? &e.expanded : &e.dec1}, {&e.skipL}, {&e.pixelHead}});
        if (m.split) e.lPatchSet = e.set(m.lPatch, {{&e.decLP}, {&m.matrixWeights, UINT64(m.lDec0[0]) * 2}, {&m.vectors, m.lDec0[1]}, {&e.dec1}, {&e.skipL}, {&e.expanded}});
    }
    mExecute(e.device, [&](ID3D12GraphicsCommandList* list) {
        ID3D12DescriptorHeap* heaps[] = {e.heap.gpu.Get()}; list->SetDescriptorHeaps(1, heaps);
        const float zero[4]{};
        for (auto* r : e.owned) if (r->resource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            r->to(list, MWrite);
            const UINT slot = e.nextSlot++;
            e.view(slot, MI, {r});
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{}; view.Format = r->resource->GetDesc().Format; view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            e.device->CreateUnorderedAccessView(r->resource.Get(), nullptr, &view, e.heap.cpuOnly(slot));
            list->ClearUnorderedAccessViewFloat(e.heap.gpuHandle(slot), e.heap.cpuOnly(slot), r->resource.Get(), zero, 0, nullptr);
            r->to(list, MRead);
        }
        const uint32_t zeros[2]{}; mWrite(list, e.exposureState, zeros, 8);
        // The AGS dummy is an intrinsic protocol binding, never a tensor input.
        e.agsDummy.state = MRead; e.agsDummy.to(list, MWrite);
    });
}
GameUpscalerM::~GameUpscalerM() = default;

GameUpscalerM::Impl::Cached& GameUpscalerM::Impl::bindings(const MGameFrame& frame) {
    const GameTexture source[] = {frame.color, frame.depth, frame.motion, frame.output};
    std::array<DXGI_FORMAT, 4> formats{};
    for (unsigned i = 0; i < 4; ++i) formats[i] = source[i].format == DXGI_FORMAT_UNKNOWN ? readFormat(source[i].resource->GetDesc().Format) : source[i].format;
    for (auto& candidate : cached) {
        bool same = candidate->formats == formats;
        for (unsigned i = 0; i < 4; ++i) same &= candidate->images[i].resource.Get() == source[i].resource;
        if (same) return *candidate;
    }
    auto owner = std::make_unique<Cached>(); auto& b = *owner; b.formats = formats;
    for (unsigned i = 0; i < 4; ++i) b.images[i].resource = source[i].resource;
    MView color{&b.images[0], 0, 0, formats[0]}, depth{&b.images[1], 0, 0, formats[1]}, motion{&b.images[2], 0, 0, formats[2]}, output{&b.images[3], 0, 0, formats[3]};
    auto& m = model;
    for (unsigned i = 0; i < 2; ++i) {
        const unsigned prev = 1 - i;
        if (m.l) {
            b.input[i] = set(m.input, {{&inputP}, color, motion, depth, {&mvA[prev]}, {&half[prev]}, {&feature[prev]}, {&mvA[i]}, {&mvB}, {&mvC}, {&m.vectors, m.dither}, {&pixelFeatures}});
            b.expand[i] = set(m.headL, {{&expandP}, {&pixelHead}, {&feature[i]}, {&gates}, {&mix}, {&covariance}});
        } else {
            b.input[i] = set(m.input, {{&inputP}, color, motion, depth, {&mvA[prev]}, {&half[prev]}, {&feature[prev]}, {&mvA[i]}, {&mvB}, {&mvC},
                                     {m.inputWmma ? &m.matrixWeights : &m.dotWeights, UINT64(m.enc0[0]) * 2}, {&m.vectors, m.enc0[1]}, {&m.vectors, m.enc0[2]}, {&tokens}});
            b.expand[i] = set(m.expand, {{&expandP}, {&mvA[i]}, {&feature[prev]}, {&feature[i]}, {&gates}, {&mix}, {&covariance}, {&dec1},
                                        {&m.dotWeights, UINT64(m.dec0[0]) * 2}, {&m.vectors, m.dec0[1]}});
        }
        b.post[i] = set(m.post, {{&postP}, color, {&mix}, {&gates}, {&covariance}, {&history[prev]}, {&mvA[i]}, {&mvB}, {&mvC}, {&history[i]}, {&half[i]}, {&packed}, motion});
    }
    b.exposure0 = set(m.exposure[0], {{&exposureP}, color, {&exposureSums}});
    b.down = set(m.down, {{&downP}, color, output, {&packed}});
    cached.push_back(std::move(owner)); return *cached.back();
}

void GameUpscalerM::record(ID3D12GraphicsCommandList* list, const MGameFrame& game, const FrameProfiler& profiler) {
    auto& e = *impl; auto& m = e.model;
    if (e.pending) throw std::runtime_error("M/L recording still awaits submitted/discarded");
    const auto& s = game.settings;
    const UINT rw = s.renderWidth ? s.renderWidth : e.maxRw, rh = s.renderHeight ? s.renderHeight : e.maxRh;
    if (rw < 64 || rh < 64 || rw > e.maxRw || rh > e.maxRh || !std::isfinite(s.jitter[0]) || !std::isfinite(s.jitter[1]) ||
        std::abs(s.jitter[0]) > 0.5f || std::abs(s.jitter[1]) > 0.5f)
        throw std::runtime_error("M/L render dimensions or jitter exceed supported bounds");
    for (unsigned i = 0; i < 2; ++i) if (!std::isfinite(s.motionScale[i]) || !std::isfinite(s.motionOffset[i])) throw std::runtime_error("M/L nonfinite motion settings");
    const GameTexture source[] = {game.color, game.depth, game.motion, game.output};
    const UINT widths[] = {rw, rw, s.displayMotion ? e.ow : rw, e.ow};
    const UINT heights[] = {rh, rh, s.displayMotion ? e.oh : rh, e.oh};
    for (unsigned i = 0; i < 4; ++i) {
        const auto& t = source[i];
        if (!t.resource) throw std::runtime_error("M/L frame has a missing texture");
        const auto desc = t.resource->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.DepthOrArraySize != 1 || desc.SampleDesc.Count != 1 ||
            UINT64(t.x) + widths[i] > desc.Width || UINT64(t.y) + heights[i] > desc.Height)
            throw std::runtime_error("M/L texture does not cover its frame rectangle");
        if (i < 3 && (t.x || t.y)) throw std::runtime_error("M/L nonzero input origins are not implemented");
        if (i < 3) { const char* reason; if (!gameInputTexture(e.device, t, &reason)) throw std::runtime_error(reason); }
        else {
            if (!(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) throw std::runtime_error("M/L output does not allow UAV stores");
            const DXGI_FORMAT format = t.format == DXGI_FORMAT_UNKNOWN ? readFormat(desc.Format) : t.format;
            if (format != DXGI_FORMAT_R16G16B16A16_FLOAT && format != DXGI_FORMAT_R32G32B32A32_FLOAT && format != DXGI_FORMAT_R10G10B10A2_UNORM &&
                format != DXGI_FORMAT_R11G11B10_FLOAT)
                throw std::runtime_error("M/L output format is unsupported");
        }
    }
    if (game.output.resource == game.color.resource || game.output.resource == game.depth.resource || game.output.resource == game.motion.resource)
        throw std::runtime_error("M/L input/output texture aliasing is unsupported");
    auto& b = e.bindings(game); e.retire();
    for (unsigned i = 0; i < 4; ++i) b.images[i].state = source[i].state;
    const UINT gw = (rw + 31) / 32 * 32, gh = (rh + 31) / 32 * 32;
    const bool reset = s.reset || e.frame == 0 || rw != e.rw || rh != e.rh;
    const float fx = float(e.hw) / rw, fy = float(e.hh) / rh;
    const float mx = s.displayMotion ? float(e.hw) / e.ow : fx, my = s.displayMotion ? float(e.hh) / e.oh : fy;
    MEnc0Params input{{int32_t(rw), int32_t(rh)}, {2, 2}, {int32_t(gw / 2), int32_t(gh / 2)}, {s.jitter[0], s.jitter[1]}, {e.previousJitter[0], e.previousJitter[1]},
        {s.motionScale[0] * mx, s.motionScale[1] * my}, {s.motionOffset[0], s.motionOffset[1]}, {fx, fy}, {1.f / fx, 1.f / fy}, {float(e.hw), float(e.hh)},
        m.l ? m.constants[0] : 9.045065879821777f, m.l ? m.constants[1] : 0.22356846928596497f, s.depthInverted ? 1u : 0u, reset ? 1u : 0u, e.frame & 7u,
        1.f, 0.f, {1.f / e.hw, 1.f / e.hh}, {1.f / e.maxRw, 1.f / e.maxRh}, s.displayMotion ? 1u : 0u,
        {int32_t(e.ow) - 1, int32_t(e.oh) - 1}, {float(e.ow) / rw, float(e.oh) / rh}};
    if (m.l) input.shift[0] = 2 ^ int32_t((e.frame & 1) * 2);
    MDec0Params expand{{int32_t(rw), int32_t(rh)}, int32_t(gw / 2), reset ? 1u : 0u, {s.jitter[0], s.jitter[1]}, {fx, fy}, {float(e.hw), float(e.hh)},
        {e.previousJitter[0], e.previousJitter[1]}, m.l ? m.constants[2] : 8.f, m.l ? m.constants[3] : 0.07863261550664902f, {1.f / e.maxRw, 1.f / e.maxRh}};
    MPostParams post{{0, 0}, {int32_t(e.hw), int32_t(e.hh)}, {int32_t(rw), int32_t(rh)}, {1.f / fx, 1.f / fy}, {s.jitter[0], s.jitter[1]},
        {1.f / e.maxRw, 1.f / e.maxRh}, m.l ? m.constants[4] : 2.7530112f, reset ? 1u : 0u, 1.f, 0.f,
        {s.displayMotion ? int32_t(e.ow) : 0, s.displayMotion ? int32_t(e.oh) : 0}, {input.mvScale[0], input.mvScale[1]}, {s.motionOffset[0], s.motionOffset[1]}};
    MDownParams down{{int32_t(e.hw), int32_t(e.hh)}, {int32_t(game.output.x), int32_t(game.output.y)}, {0, 0}, {int32_t(rw), int32_t(rh)},
        {float(rw) / e.ow, float(rh) / e.oh}, {s.jitter[0], s.jitter[1]}, 1.f};
    const UINT sw = exposureGrid(rw, e.ow), sh = exposureGrid(rh, e.oh), bx = (sw + 31) / 32, by = (sh + 31) / 32;
    ExposureParams exposure{{0, 0}, {int32_t(rw), int32_t(rh)}, 1.f / (float(sw) * sh), 1.f, 0.23547087609767914f,
        1u | (reset ? 2u : 0u), bx * by, bx, {int32_t(sw), int32_t(sh)}};
    auto stamp = [&](uint32_t boundary) { if (profiler.mark) profiler.mark(list, boundary, profiler.user); };
    // Opt-in first-submission diagnostics. Captures retire on the submission
    // fence; discarded command lists never publish their readback contents.
    std::string capturePath;
    if (e.uses == 0 && !e.captureDirectory.empty()) {
        capturePath = e.captureDirectory + "/frame-0";
        const auto base = capturePath;
        for (unsigned suffix = 1; std::filesystem::exists(capturePath); ++suffix)
            capturePath = base + "-" + std::to_string(suffix);
        std::filesystem::create_directories(capturePath);
        e.captures.push_back(std::make_unique<FrameCapture>());
        e.recordingCapture = e.captures.back().get();
        std::ofstream metadata(capturePath + "/shape.txt");
        metadata << rw << ' ' << rh << ' ' << e.hw << ' ' << e.hh << ' ' << e.ow << ' ' << e.oh << '\n';
    }
    auto capture = [&](MResource& resource, const char* name) {
        if (e.recordingCapture)
            e.recordingCapture->add(e.device, list, resource.resource.Get(), resource.state, capturePath + "/" + name);
    };
    stamp(0);
    ID3D12DescriptorHeap* heaps[] = {e.heap.gpu.Get()}; list->SetDescriptorHeaps(1, heaps);
    mWrite(list, e.inputP, &input, sizeof(input)); mWrite(list, e.expandP, &expand, sizeof(expand));
    mWrite(list, e.postP, &post, sizeof(post)); mWrite(list, e.downP, &down, sizeof(down)); mWrite(list, e.exposureP, &exposure, sizeof(exposure));
    for (unsigned i = 0; i < MLayers; ++i) {
        MLayerParams p{int32_t(gw / 2 / MScale[i]), int32_t(gh / 2 / MScale[i]), MShift[i] ^ int32_t((e.frame & 1) * 2), MShift[i], {}};
        std::memcpy(p.o, m.layers[i].data() + 2, sizeof(p.o)); mWrite(list, e.layerP[i], &p, sizeof(p));
    }
    if (m.l) {
        MLayerParams p{int32_t(gw), int32_t(gh), 2 ^ int32_t((e.frame & 1) * 2), 2, {}};
        std::memcpy(p.o, m.lEnc0.data() + 2, sizeof(p.o)); mWrite(list, e.encLP, &p, sizeof(p));
        p.sx = int32_t((e.frame & 1) * 2); p.sy = 0; std::memcpy(p.o, m.lDec0.data() + 2, sizeof(p.o)); mWrite(list, e.decLP, &p, sizeof(p));
    }
    stamp(1);
    if (!m.ldr) { e.dispatch(list, b.exposure0, bx, by); e.dispatch(list, e.exposure1, 1, 1); }
    stamp(2);
    const UINT inputBlocks = m.l ? 1 : 2;
    e.dispatch(list, b.input[e.current], ((gw + 9) / 8 + inputBlocks - 1) / inputBlocks, ((gh + 9) / 8 + inputBlocks - 1) / inputBlocks);
    if (m.l) {
        e.dispatch(list, e.inputLBlock, (gw + (2 ^ ((e.frame & 1) * 2)) + 3) / 4, (gh + 5) / 4);
        e.dispatch(list, e.mergeL, (gw / 2 + 3) / 4, (gh / 2 + 3) / 4);
    }
    stamp(3);
    for (unsigned i = 0; i < MLayers; ++i) {
        const UINT w = gw / 2 / MScale[i], h = gh / 2 / MScale[i], sx = MShift[i] ^ ((e.frame & 1) * 2);
        if (m.split && i >= 8) e.dispatch(list, e.patchSets[i - 8], (w / 2 + 5) / 4, (h / 2 + 5) / 4);
        e.dispatch(list, e.layerSets[i], (w + sx + 3) / 4, (h + MShift[i] + 3) / 4);
        if (i < 2) e.dispatch(list, e.mergeSets[i], (w / 2 + 3) / 4, (h / 2 + 3) / 4);
    }
    stamp(4);
    if (m.l) {
        if (m.split) e.dispatch(list, e.lPatchSet, (gw / 2 + 5) / 4, (gh / 2 + 5) / 4);
        e.dispatch(list, e.expandLBlock, (rw + (e.frame & 1) * 2 + 3) / 4, (rh + 3) / 4);
        e.dispatch(list, b.expand[e.current], (rw + 7) / 8, (rh + 7) / 8);
    }
    else e.dispatch(list, b.expand[e.current], (gw / 8 + 1) / 2, (gh / 8 + 1) / 2);
    stamp(5);
    capture(e.covariance, "covariance.bin");
    capture(e.gates, "gates.bin");
    capture(e.mix, "mix.bin");
    const bool quality = uint64_t(e.hw) < 3ull * rw || uint64_t(e.hh) < 3ull * rh;
    const bool compact = quality && 20ull * rw <= 9ull * e.hw && 20ull * rh <= 9ull * e.hh;
    e.dispatch(list, b.post[e.current], (e.hw + 15) / 16, (e.hh + 15) / 16, compact ? &m.postCompact : quality ? &m.postQuality : &m.post);
    stamp(6);
    capture(e.packed, "packed.bin");
    e.dispatch(list, b.down, (e.ow + 7) / 8, (e.oh + 63) / 64, b.formats[3] == DXGI_FORMAT_R10G10B10A2_UNORM ? &m.downRgb10 : &m.down);
    capture(b.images[3], "output.bin");
    for (auto* r : e.owned) if (r != &e.agsDummy) r->to(list, MRead);
    for (unsigned i = 0; i < 4; ++i) b.images[i].to(list, source[i].state);
    stamp(7);
    e.pendingSettings = s; e.pendingSettings.renderWidth = rw; e.pendingSettings.renderHeight = rh; e.pending = true;
}
void GameUpscalerM::submitted(ID3D12Fence* fence, uint64_t value) {
    auto& e = *impl;
    if (!e.pending) throw std::runtime_error("no M/L recording to submit");
    ++e.uses; e.flights.push_back({fence, value, e.uses});
    if (e.recordingCapture) {
        e.recordingCapture->pending = false;
        e.recordingCapture->fence = fence;
        e.recordingCapture->value = fence ? value : e.uses;
        e.recordingCapture = nullptr;
    }
    e.rw = e.pendingSettings.renderWidth; e.rh = e.pendingSettings.renderHeight;
    e.previousJitter[0] = e.pendingSettings.jitter[0]; e.previousJitter[1] = e.pendingSettings.jitter[1];
    e.current = 1 - e.current; ++e.frame; e.pending = false; e.retire();
}
void GameUpscalerM::discarded() {
    if (!impl->pending) throw std::runtime_error("no M/L recording to discard");
    auto& e = *impl;
    if (e.recordingCapture) {
        e.captures.erase(std::remove_if(e.captures.begin(), e.captures.end(), [&](const auto& c) {
            return c.get() == e.recordingCapture;
        }), e.captures.end());
        e.recordingCapture = nullptr;
    }
    e.pending = false;
}
bool GameUpscalerM::idle() const { impl->retire(); return !impl->pending && impl->flights.empty(); }
uint32_t GameUpscalerM::maxRenderWidth() const { return impl->maxRw; }
uint32_t GameUpscalerM::maxRenderHeight() const { return impl->maxRh; }

} // namespace rdna3
