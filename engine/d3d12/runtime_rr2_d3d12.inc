// Ray Reconstruction preset F on D3D12: the runtime around the stage DXIL of a "D4RR0002" model
// (engine/compile_rr2.py; the shaders are translated by compile_rr_dxil.py). The frame, resource geometry, parameter
// blocks and the Vulkan resolver's optional-input fallbacks follow engine/runtime_rr2.inc; this file is included by
// runtime_rr_d3d12.cpp after its helpers, the shared RR parameter ABI and Slot.
namespace
{
constexpr unsigned R2Network = 16, R2Post = 16, R2Down = 17, R2Pooled = 18, R2DownP = 19, R2Gather = 20,
                   R2Stages = 21;
constexpr unsigned R2Enc0 = 0, R2Enc5 = 5, R2Dec0B = 15;
constexpr const char *R2Names[R2Stages] = {"enc0",  "enc1",   "enc2",  "enc3",  "enc4",  "enc5",  "dec4_0", "dec4_1",
                                           "dec3_0", "dec3_1", "dec2_0", "dec2_1", "dec1_0", "dec1_1", "dec0_0", "dec0_1",
                                           "post",  "down",   "pooled", "downp", "enc0g"};
constexpr int R2Level[R2Network] = {0, 1, 2, 3, 4, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0};
constexpr uint32_t R2ArenaUnits = 3152;
constexpr uint32_t R2TileSlots = 8;
constexpr UINT R2BindingBlocks = 16;

uint32_t r2Shift(unsigned stage, uint32_t frame)
{
    const bool even = (frame & 1u) == 0;
    if (stage < R2Enc5)
        return even ? 0x00060002u : 0x00020006u;
    const bool first = stage == R2Enc5 || ((stage - R2Enc5) & 1u) == 1;
    return (even == first) ? 0x00000004u : 0x00040000u;
}
DXGI_FORMAT r2SurfaceFormat(const std::string &s)
{
    if (s == "rgba16f")
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (s == "rgba32f")
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    if (s == "rgb10_a2")
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    if (s == "rg16f")
        return DXGI_FORMAT_R16G16_FLOAT;
    if (s == "r16f")
        return DXGI_FORMAT_R16_FLOAT;
    if (s == "rgba8_snorm")
        return DXGI_FORMAT_R8G8B8A8_SNORM;
    if (s == "rgba8")
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    throw std::runtime_error("Ray Reconstruction stage surface format " + s + " is not implemented");
}
// The Vulkan resolver's filter modes: every other texture is sampled point, as NGX's texture objects.
bool r2TextureLinear(unsigned stage, uint32_t offset)
{
    if (stage == R2Enc0 || stage == R2Gather)
        return offset == 144 || offset == 264 || offset == 528 || offset == 536 || offset == 544;
    if (stage == R2Post)
        return offset == 32 || offset == 48 || offset == 72 || offset == 80;
    if (stage == R2DownP)
        return offset == 256;
    return false;
}
// A stage's descriptor table: slot 0 parameter block (raw SRV), 1 weights (raw SRV), 2 arena (raw UAV), then the
// textures as tN, the surfaces as typed UAVs uN and the byte-address buffers as raw UAVs uN. A raw SRV with a stride
// is a structured one (the exposure pass's block sums). The spill buffer (translator's scratch) is u40.
void r2WriteTable(ID3D12Device *device, D3D12_CPU_DESCRIPTOR_HANDLE first, UINT increment,
                  const std::vector<Binding> &kinds, const std::vector<Slot> &slots)
{
    for (size_t i = 0; i < slots.size(); ++i)
    {
        const Slot &s = slots[i];
        const Binding &b = kinds[i];
        if (b.kind == Kind::None)
            continue;
        if (!s.resource)
            throw std::runtime_error("Ray Reconstruction descriptor slot " + std::to_string(i) + " has no resource");
        const auto handle = D3D12_CPU_DESCRIPTOR_HANDLE{first.ptr + SIZE_T(i) * increment};
        const auto width = s.resource->GetDesc().Width;
        if (b.kind == Kind::Tex)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC v{};
            v.Format = s.format != DXGI_FORMAT_UNKNOWN ? s.format : readFormat(s.resource->GetDesc().Format);
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            v.Texture2D.MipLevels = 1;
            device->CreateShaderResourceView(s.resource, &v, handle);
        }
        else if (b.kind == Kind::Raw)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC v{};
            v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if (b.stride) // a structured buffer: exposure's block sums
            {
                v.Format = DXGI_FORMAT_UNKNOWN;
                v.Buffer = {UINT(s.offset / b.stride), UINT((width - s.offset) / b.stride), b.stride,
                            D3D12_BUFFER_SRV_FLAG_NONE};
            }
            else
            {
                v.Format = DXGI_FORMAT_R32_TYPELESS;
                v.Buffer = {UINT(s.offset / 4), UINT((width - s.offset) / 4), 0, D3D12_BUFFER_SRV_FLAG_RAW};
            }
            device->CreateShaderResourceView(s.resource, &v, handle);
        }
        else if (b.kind == Kind::RwTex)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
            v.Format = s.format != DXGI_FORMAT_UNKNOWN ? s.format : readFormat(s.resource->GetDesc().Format);
            v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            device->CreateUnorderedAccessView(s.resource, nullptr, &v, handle);
        }
        else if (b.kind == Kind::RwStruct)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
            v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            v.Buffer = {UINT(s.offset / b.stride), UINT((width - s.offset) / b.stride), b.stride, 0,
                        D3D12_BUFFER_UAV_FLAG_NONE};
            device->CreateUnorderedAccessView(s.resource, nullptr, &v, handle);
        }
        else if (b.kind == Kind::RwRaw)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
            v.Format = DXGI_FORMAT_R32_TYPELESS;
            v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            v.Buffer = {UINT(s.offset / 4), UINT((width - s.offset) / 4), 0, 0, D3D12_BUFFER_UAV_FLAG_RAW};
            device->CreateUnorderedAccessView(s.resource, nullptr, &v, handle);
        }
    }
}

struct R2Stage
{
    Pipeline pipeline;
    std::array<Pipeline, 4> phases; // RR2_PHASE 1 - 4; null when the model has none
    uint32_t heads = 0, branches = 0, recordWords = 0;
    std::vector<Binding> layout;
    std::vector<uint8_t> args; // the captured argument block (+ the translator's loop-zero word)
    size_t argsBytes = 0;      // the captured block's file length, before any appended word
    std::vector<uint32_t> textures, buffers;
    std::vector<std::pair<uint32_t, std::string>> surfaces;
    std::vector<std::pair<uint32_t, double>> pointers;
    uint32_t group = 1, record = 0;
    bool ranges = false;
    std::string dxWeights;
    ComPtr<ID3D12Resource> weights;
    std::vector<char> pending;
    uint32_t scratchWords = 0;
    bool loopZero = false;
    uint32_t loopZeroWords = 0;
};

struct R2Model
{
    ID3D12Device *device = nullptr;
    std::array<R2Stage, R2Stages> stages;
    std::vector<char> auxPending;
    ComPtr<ID3D12Resource> aux;
    // exposure_f0/f1 as rr_hlsl.translate declares them: Params t0 (byte-address), the block sums t1/u2, the values
    // u2, the previous exposure t3 and this frame's r16f exposure u4
    std::vector<Binding> exp0B{R, T, UR};
    std::vector<Binding> exp1B{R, R, UR, T, UT};
    std::array<Pipeline, 2> exposure;
    // exposure_f2 (the vkd3d-proton build): both passes in one with 64-bit atomics on the sums (the total and the
    // count, zero between frames), as the Vulkan build's exposure_f2.spv; Params t0, the colour t1, the sums u2, the
    // values u3, the previous exposure t4 and this frame's u5
    std::vector<Binding> exp2B{R, T, UR, UR, T, UT};
    Pipeline exposureOne;
    bool particles = false, gather = false;
    uint32_t preset = 0;

    static std::vector<std::string> list(const std::string &s)
    {
        return split(s);
    }
    unsigned find(const std::string &name) const
    {
        for (unsigned i = 0; i < R2Stages; ++i)
            if (name == R2Names[i])
                return i;
        return R2Stages;
    }
    void load(ID3D12Device *d, const std::string &directory, const std::vector<std::string> &sidecar)
    {
        device = d;
        std::array<bool, R2Stages> weightsSeen{}, scratchSeen{}, loopSeen{};
        for (const auto &line : sidecar)
        {
            std::istringstream f(line);
            std::string key, name;
            f >> key >> name;
            if (key.empty())
                continue;
            const unsigned at = find(name);
            if (at >= R2Stages)
                throw std::runtime_error("invalid RR D3D12 manifest stage: " + line);
            if (key == "weights")
            {
                f >> stages[at].dxWeights;
                if (!f || stages[at].dxWeights.empty() || weightsSeen[at] ||
                    std::filesystem::path(stages[at].dxWeights).filename() != stages[at].dxWeights)
                    throw std::runtime_error("invalid RR DXIL weight file");
                weightsSeen[at] = true;
            }
            else if (key == "scratch")
            {
                uint32_t words = 0;
                f >> words;
                if (!f || !words || words > 16384 || scratchSeen[at])
                    throw std::runtime_error("invalid RR scratch size");
                stages[at].scratchWords = words;
                scratchSeen[at] = true;
            }
            else if (key == "loopzero")
            {
                uint32_t words = 0;
                f >> words;
                if (!f || !words || loopSeen[at])
                    throw std::runtime_error("invalid RR loop-zero parameter ABI");
                stages[at].loopZero = true;
                stages[at].loopZeroWords = words;
                loopSeen[at] = true;
            }
            else
                throw std::runtime_error("invalid RR D3D12 manifest");
        }
        auto text = slurp(directory + "/rr.txt");
        std::istringstream in(std::string(text.begin(), text.end()));
        std::string line;
        if (!std::getline(in, line) || line != "D4RR0002")
            throw std::runtime_error("requires an RR preset F model (D4RR0002)");
        std::array<bool, R2Stages> seen{};
        while (std::getline(in, line))
        {
            std::istringstream f(line);
            std::string key;
            f >> key;
            if (key.empty())
                continue;
            if (key == "preset")
                f >> preset;
            else if (key == "arithmetic")
                throw std::runtime_error("D3D12 RR currently requires gfx12 arithmetic");
            else if (key == "phases")
            {
                std::string name, files[4];
                uint32_t heads = 0, branches = 0, words = 0;
                f >> name >> files[0] >> files[1] >> files[2] >> files[3] >> heads >> branches >> words;
                const unsigned at = find(name);
                if (!f || at >= R2Network || !seen[at] || !heads || !branches || !words)
                    throw std::runtime_error("invalid RR phases line: " + line);
                std::string path[4];
                bool all = true;
                for (int k = 0; k < 4; ++k)
                {
                    auto p = std::filesystem::path(files[k]);
                    p.replace_extension(".dxil");
                    path[k] = directory + "/" + p.string();
                    all = all && std::filesystem::exists(path[k]);
                }
                if (!all)
                    continue; // a model packaged without the phase shaders runs the stage's own shader
                auto &st = stages[at];
                for (int k = 0; k < 4; ++k)
                    st.phases[k].create(device, path[k], st.layout);
                st.heads = heads;
                st.branches = branches;
                st.recordWords = words;
            }
            else if (key == "stage")
            {
                std::string name, shader, weightFile, argsFile, ts, ss, bs, ps, option;
                unsigned wave = 0;
                f >> name >> shader >> wave >> weightFile >> argsFile >> ts >> ss >> bs >> ps;
                (void)wave;
                const unsigned at = find(name);
                if (!f || at >= R2Stages || seen[at])
                    throw std::runtime_error("invalid RR stage: " + line);
                seen[at] = true;
                auto &st = stages[at];
                while (f >> option)
                {
                    if (option.size() == 2 && option[0] == 'x' && option[1] >= '1' && option[1] <= '4')
                        st.group = uint32_t(option[1] - '0');
                    if (option.size() > 1 && option[0] == 'r')
                        st.record = uint32_t(std::stoul(option.substr(1)));
                    if (option == "s3")
                        st.ranges = true;
                }
                for (const auto &item : list(ts))
                    st.textures.push_back(uint32_t(std::stoul(item)));
                for (const auto &item : list(bs))
                    st.buffers.push_back(uint32_t(std::stoul(item)));
                for (const auto &item : list(ss))
                {
                    const auto colon = item.find(':');
                    if (colon == std::string::npos)
                        throw std::runtime_error("bad RR surface: " + item);
                    st.surfaces.emplace_back(uint32_t(std::stoul(item.substr(0, colon))), item.substr(colon + 1));
                }
                for (const auto &item : list(ps))
                {
                    const auto colon = item.find(':');
                    if (colon == std::string::npos)
                        throw std::runtime_error("bad RR pointer: " + item);
                    st.pointers.emplace_back(uint32_t(std::stoul(item.substr(0, colon))), std::stod(item.substr(colon + 1)));
                }
                { // the captured argument block, then the translator's loop-zero word
                    const auto bytes = slurp(directory + "/" + argsFile);
                    st.args.assign(bytes.begin(), bytes.end());
                    if (st.args.empty() || st.args.size() % 4)
                        throw std::runtime_error("invalid RR argument block " + argsFile);
                    st.argsBytes = st.args.size();
                    if (st.loopZero)
                    {
                        // Non-phased stages guard at the block's end; the phased stages keep their record-arena
                        // pointer at the end and guard four bytes further on (104 + 4).
                        if (st.loopZeroWords != st.argsBytes && st.loopZeroWords != st.argsBytes + 4)
                            throw std::runtime_error("RR loop-zero offset does not match the argument block of " + name);
                        st.args.resize(std::max<size_t>(st.args.size(), size_t(st.loopZeroWords) + 4), 0);
                    }
                }
                // the DXIL weights: the sidecar's transposed/tiled blob, or a 16-byte placeholder (stages without)
                if (!st.dxWeights.empty())
                {
                    auto bytes = slurp(directory + "/" + st.dxWeights);
                    st.pending.assign(bytes.begin(), bytes.end());
                }
                else if (weightFile != "-")
                    throw std::runtime_error("RR stage " + name + " has no D3D12 weights: rebuild the model with compile_rr_dxil.py");
                else
                    st.pending.assign(16, 0);
                st.weights = buffer(device, st.pending.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
                // the descriptor table, whose registers the translated shaders declare
                st.layout = {R, R, UR};
                for (uint32_t offset : st.textures)
                    st.layout.push_back(r2TextureLinear(at, offset) ? TL : T);
                st.layout.insert(st.layout.end(), st.surfaces.size(), UT);
                st.layout.insert(st.layout.end(), st.buffers.size(), UR);
                if (st.scratchWords)
                {
                    if (st.layout.size() > 40)
                        throw std::runtime_error("RR stage " + name + " has too many bindings for the u40 scratch slot");
                    st.layout.resize(41);
                    st.layout[40] = US(4);
                }
                auto path = std::filesystem::path(shader);
                path.replace_extension(".dxil");
                st.pipeline.create(device, directory + "/" + path.string(), st.layout);
            }
        }
        for (unsigned stage = 0; stage < R2DownP; ++stage)
            if (!seen[stage])
                throw std::runtime_error("incomplete Ray Reconstruction preset F model");
        for (unsigned stage = 0; stage < R2Stages; ++stage)
            if ((loopSeen[stage] || scratchSeen[stage] || weightsSeen[stage]) && !seen[stage])
                throw std::runtime_error(std::string("RR D3D12 manifest names a stage the model lacks: ") + R2Names[stage]);
        // A phased stage's record-arena pointer occupies its block's last word; the translator's loop-zero guard must
        // sit after it (args length + 4). There is no legacy layout: a 104-style guard would loop recordsAt times.
        for (unsigned stage = 0; stage < R2Network; ++stage)
        {
            const auto &st = stages[stage];
            if (st.phases[0] && loopSeen[stage] && st.loopZeroWords != st.argsBytes + 4)
                throw std::runtime_error(std::string("RR phased stage ") + R2Names[stage] +
                                         " requires a loop-zero word at " + std::to_string(st.argsBytes + 4));
        }
        particles = seen[R2DownP];
        gather = seen[R2Gather] && stages[R2Gather].record != 0 && stages[R2Enc0].args.size() >= 8;
        { // encoder 0's second table
            auto bytes = slurp(directory + "/enc0-aux.bin");
            auxPending.assign(bytes.begin(), bytes.end());
            aux = buffer(device, auxPending.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        }
        exposure[0].create(device, directory + "/exposure_f0.dxil", exp0B);
        exposure[1].create(device, directory + "/exposure_f1.dxil", exp1B);
        if (std::filesystem::exists(directory + "/exposure_f2.dxil"))
            exposureOne.create(device, directory + "/exposure_f2.dxil", exp2B);
    }
};

struct R2ExposureParams
{
    int32_t last[2];
    uint32_t blocksX, blocks;
    float invCount;
    uint32_t reset;
};

struct R2Engine
{
    R2Model &model;
    ID3D12Device *device;
    uint32_t rw, rh, ow, oh, w0, h0, iw, ih, cw, ch, qw, qh;
    uint32_t nx, ny, maxW, maxH;
    uint64_t unit, tileB, tileBytes = 0;
    std::vector<uint64_t> slotAt;
    uint32_t tileGroup = 1;
    std::array<uint32_t, 5> gatherWords{};
    std::array<bool, R2Network> split{};
    // the stages whose phases' records exist in the arena (argument 104): the split ones and, with an external network
    // that reaches the arena by address (d3dhip), every stage with phases
    std::array<bool, R2Network> records{};
    uint64_t recordsAt = 0;
    struct Rect
    {
        uint32_t x0, y0, x1, y1;
    };
    std::vector<Rect> src, dst;
    std::vector<uint64_t> downPAt;
    std::vector<std::vector<uint8_t>> downPBlocks;
    std::array<ComPtr<ID3D12Resource>, 2> pb;
    std::array<ComPtr<ID3D12Resource>, 2> feat, hist, e, mv;
    ComPtr<ID3D12Resource> rec, ka, kb, kc, half, quarter, zero;
    std::map<std::string, ComPtr<ID3D12Resource>> spare;
    std::vector<ComPtr<ID3D12Resource>> scratch;
    ComPtr<ID3D12Resource> arena, values, sums, dummy, ags;
    std::array<uint64_t, R2Network> paramAt{};
    std::vector<uint64_t> postAt, downAt;
    uint64_t pooledAt = 0, expAt = 0, paramBytes = 0;
    std::array<std::vector<uint8_t>, R2Network> blocks;
    std::vector<std::vector<uint8_t>> postBlocks, downBlocks;
    std::vector<uint8_t> pooledBlock;
    R2ExposureParams exposure{};
    // descriptors: one block per distinct input view set; parameter blocks are per binding (stable descriptors)
    ComPtr<ID3D12DescriptorHeap> heap;
    UINT increment = 0, slotsPerBind = 0;
    struct Bindings
    {
        RFrameImages images{};
        FrameOutput output{};
        std::vector<ComPtr<ID3D12Resource>> retained;
        ComPtr<ID3D12Fence> fence;
        uint64_t value = 0;
        bool pending = false, inFlight = false;
        UINT base = 0;
        ComPtr<ID3D12Resource> paramAll;
        std::array<std::array<UINT, 2>, R2Network> network{};
        std::array<UINT, 2> gather{};
        std::vector<std::array<UINT, 2>> post;
        std::vector<UINT> down;
        std::vector<std::array<UINT, 2>> downP;
        UINT pooled = 0, exp0 = 0;
        std::array<UINT, 2> exp1{}, expOne{};
    };
    std::vector<Bindings> bindings;
    size_t current = 0, recorded = SIZE_MAX;
    // per-recording parameter upload buffers: the bytes a deferred copy reads until its frame has retired. A copy is
    // owned from recordFrame to submitted/discarded; only its submission fence releases it.
    struct ParamCopy
    {
        ComPtr<ID3D12Resource> staging;
        void *mapped = nullptr;
        ComPtr<ID3D12Fence> fence;
        uint64_t value = 0;
        bool used = false;
    };
    std::vector<ParamCopy> paramCopies;
    size_t recordingParam = SIZE_MAX;
    std::map<ID3D12Resource *, D3D12_RESOURCE_STATES> rest, now;
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    float prevJitter[2] = {0, 0};
    float savedJitter[2] = {0, 0};
    uint32_t frame = 0;
    bool frameSet = false;
    // an external network (the native HIP stages, d3d12/hip_net_rr.h): the front is recorded, the network runs, then
    // the back; the network's own stages are not dispatched here
    const ExternalNetworkRr *external = nullptr;
    bool frontDone = false;
    // TMP: D4R_RR_TMP_STAGEDUMP=DIR with D4R_RR_TMP_STAGEDUMP_FRAME=N saves the arena (NGX's 3152 units) before the
    // gather and after every network stage of frame N, for checking NVIDIA's kernels on this engine's state
    std::vector<std::pair<std::string, ComPtr<ID3D12Resource>>> stageDumps;
    void stageDump(ID3D12GraphicsCommandList *list, const std::string &name)
    {
        const char *dir = std::getenv("D4R_RR_TMP_STAGEDUMP"), *at = std::getenv("D4R_RR_TMP_STAGEDUMP_FRAME");
        if (!dir || !at || frame != uint32_t(std::atoi(at)))
            return;
        uav();
        use(arena.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
        flush(list);
        auto rb = buffer(device, unit * R2ArenaUnits, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(rb.Get(), 0, arena.Get(), 0, unit * R2ArenaUnits);
        stageDumps.emplace_back(std::string(dir) + "/" + name + ".bin", rb);
    }
    void writeStageDumps(ID3D12Fence *fence, uint64_t value)
    {
        if (stageDumps.empty())
            return;
        DXCHECK(fence->SetEventOnCompletion(value, nullptr));
        for (auto &[path, rb] : stageDumps)
        {
            void *data;
            DXCHECK(rb->Map(0, nullptr, &data));
            std::ofstream(path, std::ios::binary).write(static_cast<const char *>(data), std::streamsize(unit * R2ArenaUnits));
            rb->Unmap(0, nullptr);
        }
        stageDumps.clear();
    }

    uint32_t lw(int level) const { return w0 >> level; }
    uint32_t lh(int level) const { return h0 >> level; }
    uint32_t tileSlot(size_t i) const
    {
        const uint32_t g = uint32_t(i) / tileGroup, k = uint32_t(i) % tileGroup;
        return (g & 1u) * tileGroup + k;
    }
    DXGI_FORMAT pooledFormat() const
    {
        if (std::getenv("D4R_RR_TMP_POOLED32")) // TMP: the reference's RGBA32F pooled histories
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        for (const auto &[offset, format] : model.stages[R2Post].surfaces)
            if (offset == 120)
                return r2SurfaceFormat(format);
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    }
    ComPtr<ID3D12Resource> buf(UINT64 bytes, D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
    {
        auto x = buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, state);
        rest[x.Get()] = state;
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
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT i) const
    {
        auto x = heap->GetGPUDescriptorHandleForHeapStart();
        x.ptr += UINT64(i) * increment;
        return x;
    }
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
    void uav()
    {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barriers.push_back(b);
    }
    void dispatch(ID3D12GraphicsCommandList *list, const Pipeline &p, UINT first, UINT x, UINT y, bool barrier = true)
    {
        auto &slots = tables.at(first);
        const auto &kinds = *p.bindings;
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
        if (barrier)
            uav();
    }
    std::map<UINT, std::vector<Slot>> tables;

    Slot resolve(unsigned stage, bool surface, uint32_t offset, uint32_t par, const RFrameImages &in,
                 ID3D12Resource *output) const
    {
        auto T = [](const ComPtr<ID3D12Resource> &t) { return Slot{t.Get()}; };
        auto G = [](const FrameTexture &t) { return Slot{t.resource, 0, t.format, true}; };
        if (stage == R2Gather)
            stage = R2Enc0;
        if (stage == R2Enc0)
        {
            if (!surface)
                switch (offset)
                {
                case 144:
                    return G(in.color);
                case 160:
                    return G(in.diffuseAlbedo);
                case 176:
                    return G(in.specularAlbedo);
                case 192:
                    return G(in.motion);
                case 224:
                    return G(in.normals);
                case 240:
                    return G(in.roughness);
                case 264:
                    return T(feat[par ^ 1]);
                case 528:
                    return T(rec);
                case 536:
                    return T(half);
                case 544:
                    return T(quarter);
                case 552:
                    return G(in.depth);
                // the optional inputs: unbound without them, like NGX
                case 272:
                    return in.specularHitDistance.resource ? G(in.specularHitDistance) : Slot{};
                case 336:
                    return in.disocclusionMask.resource ? G(in.disocclusionMask) : Slot{};
                case 288:
                    return in.subsurfaceGuide.resource ? G(in.subsurfaceGuide) : Slot{};
                default:
                    return {};
                }
            else
                switch (offset)
                {
                case 368:
                    return T(feat[par]);
                case 376:
                    return T(mv[0]);
                case 384:
                    return T(mv[1]);
                default:
                    return {};
                }
        }
        else if (stage == R2Dec0B)
        {
            if (surface)
                switch (offset)
                {
                case 176:
                    return T(rec);
                case 184:
                    return T(ka);
                case 192:
                    return T(kb);
                case 200:
                    return T(kc);
                default:
                    return {};
                }
        }
        else if (stage == R2Post)
        {
            if (!surface)
                switch (offset)
                {
                case 24:
                    return T(ka);
                case 32:
                    return T(kb);
                case 40:
                    return T(kc);
                case 48:
                    return T(hist[par ^ 1]);
                case 56:
                    return T(feat[par]);
                case 64:
                    return T(mv[0]);
                case 72:
                    return T(mv[1]);
                case 80:
                    return G(in.color);
                default:
                    return {};
                }
            else
                switch (offset)
                {
                case 96:
                    return T(hist[par]);
                case 120:
                    return T(half);
                default:
                    return {};
                }
        }
        else if (stage == R2Down)
        {
            if (surface && offset == 40)
                return Slot{output, 0, DXGI_FORMAT_UNKNOWN, true};
            return {};
        }
        else if (stage == R2DownP)
        {
            if (surface && offset == 40)
                return Slot{output, 0, DXGI_FORMAT_UNKNOWN, true};
            if (!surface && offset == 80)
                return G(in.color);
            if (surface && offset == 264)
                return T(pb[par]);
            if (!surface && offset == 208)
                return in.colorBeforeParticles.resource ? G(in.colorBeforeParticles) : Slot{};
            if (!surface && offset == 256)
                return T(pb[par ^ 1]);
            return {};
        }
        else if (stage == R2Pooled)
            return surface ? T(quarter) : T(half);
        return {};
    }
    std::vector<Slot> stageSlots(R2Stage &st, uint32_t par, const RFrameImages &in, ID3D12Resource *output,
                                 bool gather)
    {
        std::vector<Slot> slots{Slot{}, Slot{st.weights.Get()}, Slot{arena.Get()}};
        const unsigned stage = gather ? unsigned(R2Enc0) : unsigned(&st - &model.stages[0]);
        for (uint32_t offset : st.textures)
        {
            auto t = resolve(stage, false, offset, par, in, output);
            slots.push_back(t.resource ? t : Slot{zero.Get()});
        }
        for (const auto &[offset, format] : st.surfaces)
        {
            auto t = resolve(stage, true, offset, par, in, output);
            slots.push_back(t.resource ? t : Slot{spare.at(format).Get()});
        }
        for (size_t i = 0; i < st.buffers.size(); ++i)
        {
            const uint32_t offset = st.buffers[i];
            ID3D12Resource *r = dummy.Get();
            if (stage == R2Enc0 && offset == 104)
                r = model.aux.Get();
            else if ((stage == R2Enc0 && offset == 256) || (stage == R2Post && offset == 16) ||
                     ((stage == R2Down || stage == R2DownP) && offset == 72))
                r = values.Get();
            slots.push_back(Slot{r});
        }
        if (st.scratchWords)
        {
            slots.resize(41);
            slots[40] = Slot{scratch[stage].Get()};
        }
        return slots;
    }

    R2Engine(R2Model &m, ID3D12Device *d, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth,
             uint32_t outputHeight, bool shareArena = false)
        : model(m), device(d), rw(renderWidth), rh(renderHeight), ow(outputWidth), oh(outputHeight)
    {
        w0 = (rw + 31) & ~31u;
        h0 = (rh + 31) & ~31u;
        iw = (ow * 3 + 1) / 2;
        ih = (oh * 3 + 1) / 2;
        cw = iw / 2;
        ch = ih / 2;
        qw = (cw + 1) / 2;
        qh = (ch + 1) / 2;
        nx = (iw + 719) / 720;
        ny = (ih + 1619) / 1620;
        for (uint32_t ty = 0; ty < ny; ++ty)
            for (uint32_t tx = 0; tx < nx; ++tx)
            {
                const uint32_t x0 = tx * iw / nx, x1 = (tx + 1) * iw / nx;
                const uint32_t y0 = ty * ih / ny, y1 = (ty + 1) * ih / ny;
                src.push_back({x0 >= 4 ? x0 - 4 : 0, y0 >= 4 ? y0 - 4 : 0, std::min(iw, x1 + 4), std::min(ih, y1 + 4)});
                dst.push_back({tx * ow / nx, ty * oh / ny, (tx + 1) * ow / nx, (ty + 1) * oh / ny});
            }
        maxW = maxH = 0;
        for (const auto &r : src)
        {
            maxW = std::max(maxW, r.x1 - r.x0);
            maxH = std::max(maxH, r.y1 - r.y0);
        }
        unit = uint64_t(w0) * h0 / 32;
        const uint64_t tilePixels = uint64_t(maxW + 8) * (maxH + 8);
        tileBytes = tilePixels * 6;
        const uint32_t tileCount = nx * ny;
        const uint32_t slots = std::min(tileCount, R2TileSlots);
        tileGroup = tileCount <= R2TileSlots ? tileCount : R2TileSlots / 2;
        const uint64_t arenaBytes = unit * R2ArenaUnits;
        // arena regions: the tile slots fit in dead space before the network's output, the gather records in the gaps
        // of encoder 0 (s3 models) and after its regions; the phases' records after the network's regions
        std::vector<uint32_t> starts{R2ArenaUnits};
        for (unsigned stage = 0; stage < R2Network; ++stage)
            for (const auto &[offset, units] : model.stages[stage].pointers)
                starts.push_back(uint32_t(std::lround(units)));
        std::sort(starts.begin(), starts.end());
        auto regionEnd = [&](uint32_t start) { return *std::upper_bound(starts.begin(), starts.end(), start); };
        uint32_t netOutStart = 0, enc0End = 0;
        for (const auto &[offset, units] : model.stages[R2Dec0B].pointers)
            if (offset == 64)
                netOutStart = uint32_t(std::lround(units));
        for (unsigned stage : {unsigned(R2Enc0), unsigned(R2Gather)})
            for (const auto &[offset, units] : model.stages[stage].pointers)
                enc0End = std::max(enc0End, regionEnd(uint32_t(std::lround(units))));
        const uint32_t before = std::getenv("D4R_RR_TMP_NOREUSE") ? 0u // TMP: no tile slots in the dead space
                                                                 : uint32_t(std::min<uint64_t>(slots, unit * netOutStart / tileBytes));
        for (uint32_t k = 0; k < slots; ++k)
            slotAt.push_back(k < before ? tileBytes * k : arenaBytes + tileBytes * (k - before));
        tileB = tilePixels * 4;
        const uint64_t tileEnd = arenaBytes + tileBytes * (slots - before);
        const uint64_t tail = (unit * enc0End + 255) & ~uint64_t(255);
        const uint32_t rows = h0 / 8 + 1;
        const uint64_t rowBytes =
            model.gather ? uint64_t(w0 / 8 + 1) * model.stages[R2Gather].record * 4 : 1;
        std::vector<std::pair<uint64_t, uint32_t>> ranges;
        if (model.gather && model.stages[R2Gather].ranges)
        {
            std::vector<std::pair<uint32_t, uint32_t>> used;
            for (unsigned stage : {unsigned(R2Enc0), unsigned(R2Gather)})
                for (const auto &[offset, units] : model.stages[stage].pointers)
                {
                    const uint32_t start = uint32_t(std::lround(units));
                    used.push_back({start, regionEnd(start)});
                }
            std::sort(used.begin(), used.end());
            uint32_t at = 0;
            for (const auto &[start, end] : used)
            {
                if (start > at)
                {
                    const uint64_t from = (unit * at + 255) & ~uint64_t(255), to = unit * start;
                    if (to > from && (to - from) / rowBytes > 0)
                        ranges.push_back({from, uint32_t((to - from) / rowBytes)});
                }
                at = std::max(at, end);
            }
            std::sort(ranges.begin(), ranges.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
            if (ranges.size() > 2)
                ranges.resize(2);
        }
        uint32_t placed = 0;
        for (const auto &r : ranges)
            placed += r.second;
        ranges.resize(2, {tail, 0});
        gatherWords = {uint32_t(ranges[0].first), ranges[0].second, uint32_t(ranges[1].first), ranges[1].second,
                       uint32_t(tail)};
        const uint64_t gatherTail = placed >= rows ? 0 : (rows - placed) * rowBytes;
        const char *splitEnv = std::getenv("D4R_RR2_SPLIT_WINDOWS");
        const uint32_t splitWindows = splitEnv ? uint32_t(std::strtoul(splitEnv, nullptr, 10)) : 48u;
        uint64_t recordBytes = 0;
        recordsAt = (arenaBytes + 255) & ~uint64_t(255);
        for (unsigned stage = 0; stage < R2Network; ++stage)
        {
            const auto &st = model.stages[stage];
            if (!st.phases[0])
                continue;
            const int l = R2Level[stage];
            uint32_t windows = 0;
            for (uint32_t f = 0; f < 2; ++f)
            {
                const uint32_t shift = r2Shift(stage, f), sx = shift & 0xffffu, sy = shift >> 16;
                windows = std::max(windows, ((lw(l) + sx + 7) / 8) * ((lh(l) + sy + 7) / 8));
            }
            split[stage] = windows <= splitWindows;
            records[stage] = split[stage] || shareArena;
            if (records[stage])
                recordBytes = std::max(recordBytes, uint64_t(windows) * st.recordWords * 4);
        }
        // and the stages without phases there (dec4_0, dec3_0; kernels/rr/glsl2hip.py --record): their level's records,
        // laid out for a stage with the same heads, branches and channels (as runtime_rr2.inc with arena addresses, the
        // contract the native HIP stages run under)
        if (shareArena)
            for (unsigned stage = 0; stage < R2Network; ++stage)
                for (unsigned other = 0; other < R2Network && !records[stage]; ++other)
                    records[stage] = records[other] && R2Level[other] == R2Level[stage];
        const uint64_t arenaTotal = std::max({tileEnd, model.gather ? tail + gatherTail : 0, recordsAt + recordBytes});
        if (arenaTotal > 0xffffffffull)
            throw std::runtime_error("Ray Reconstruction: the render size is too large for the arena");
        // the arena is the external network's too (d3dhip): then it must be shareable, so it starts and rests in COMMON
        arena = shareArena ? sharedArena(device, arenaTotal) : buf(arenaTotal);
        if (shareArena)
            rest[arena.Get()] = D3D12_RESOURCE_STATE_COMMON;
        values = buf(16);
        dummy = buf(4096);
        ags = buf(256);
        exposure.blocksX = (rw + 15) / 16;
        exposure.blocks = exposure.blocksX * ((rh + 15) / 16);
        exposure.last[0] = int32_t(rw - 1);
        exposure.last[1] = int32_t(rh - 1);
        exposure.invCount = float(10000.0 / (4294967296.0 * rw * rh));
        sums = buf(std::max(exposure.blocks, 2u) * 8);
        const size_t tiles = src.size();
        auto place = [&](size_t bytes) {
            const uint64_t at = paramBytes;
            paramBytes += (bytes + 255) & ~uint64_t(255);
            return at;
        };
        for (unsigned i = 0; i < R2Network; ++i)
        {
            blocks[i] = model.stages[i].args;
            if (records[i])
                blocks[i].resize(std::max<size_t>(blocks[i].size(), 108));
            paramAt[i] = place(blocks[i].size());
        }
        postBlocks.assign(tiles, model.stages[R2Post].args);
        downBlocks.assign(tiles, model.stages[R2Down].args);
        if (model.particles)
            downPBlocks.assign(tiles, model.stages[R2DownP].args);
        for (size_t i = 0; i < tiles; ++i)
        {
            postAt.push_back(place(postBlocks[i].size()));
            downAt.push_back(place(downBlocks[i].size()));
            if (model.particles)
                downPAt.push_back(place(downPBlocks[i].size()));
        }
        pooledBlock = model.stages[R2Pooled].args;
        pooledAt = place(pooledBlock.size());
        expAt = place(sizeof(R2ExposureParams));
        // a few parameter upload buffers up front: one per frame that may still be in flight
        for (int i = 0; i < 4; ++i)
        {
            ParamCopy copy;
            copy.staging = buffer(device, paramBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            DXCHECK(copy.staging->Map(0, nullptr, &copy.mapped));
            paramCopies.push_back(std::move(copy));
        }
        const DXGI_FORMAT half4 = DXGI_FORMAT_R16G16B16A16_FLOAT;
        for (int i = 0; i < 2; ++i)
        {
            feat[i] = tex(rw, rh, half4);
            hist[i] = tex(iw, ih, half4);
            e[i] = tex(1, 1, DXGI_FORMAT_R16_FLOAT);
            mv[i] = tex(rw, rh, DXGI_FORMAT_R16G16_FLOAT);
        }
        rec = tex(rw, rh, DXGI_FORMAT_R8G8B8A8_SNORM);
        ka = tex(rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM);
        kb = tex(rw, rh, DXGI_FORMAT_R8G8B8A8_UNORM);
        kc = tex(rw, rh, half4);
        const DXGI_FORMAT pooledF = pooledFormat();
        half = tex(cw, ch, pooledF);
        quarter = tex(qw, qh, pooledF);
        zero = tex(1, 1, half4);
        for (const auto &st : model.stages)
            for (const auto &[offset, format] : st.surfaces)
                if (!spare.count(format))
                    spare[format] = tex(1, 1, r2SurfaceFormat(format));
        for (const auto &st : model.stages)
        {
            if (st.weights && !rest.count(st.weights.Get()))
                rest[st.weights.Get()] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        rest[model.aux.Get()] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        scratch.resize(R2Stages);
        for (unsigned stage = 0; stage < R2Network; ++stage)
        {
            const auto &st = model.stages[stage];
            if (!st.scratchWords)
                continue;
            const int l = R2Level[stage];
            uint64_t groups = uint64_t((lw(l) + 13) / 8) * ((lh(l) + 13) / 8);
            groups *= std::max({1u, st.heads, st.branches});
            scratch[stage] = buf(uint64_t(st.scratchWords) * 4 * groups);
        }
        // the fixed fields: size, arena regions, 1 / size
        for (unsigned stage = 0; stage < R2Network; ++stage)
        {
            auto &p = blocks[stage];
            const int l = R2Level[stage];
            if (records[stage])
                put32(p, 104, uint32_t(recordsAt));
            put32(p, 0, lw(l));
            put32(p, 4, lh(l));
            putF(p, 84, 1.0f / float(lw(l)));
            putF(p, 88, 1.0f / float(lh(l)));
            for (const auto &[offset, units] : model.stages[stage].pointers)
                put64(p, offset, uint64_t(std::llround(units * double(unit))));
        }
        const float fr = float(rw), fh = float(rh), fiw = float(iw), fih = float(ih);
        const float fow = float(ow), foh = float(oh);
        auto &p = blocks[R2Enc0];
        const size_t argsBytes = model.stages[R2Enc0].argsBytes;
        if (model.gather && model.stages[R2Gather].ranges)
            for (unsigned i = 0; i < 5; ++i)
                put32(p, uint32_t(argsBytes - 24 + 4 * i), gatherWords[i]);
        else if (model.gather)
            put64(p, uint32_t(argsBytes - 8), gatherWords[4]);
        put32(p, 128, rw);
        put32(p, 132, rh);
        put32(p, 136, w0);
        put32(p, 140, h0);
        putF(p, 408, fr / fiw);
        putF(p, 412, fh / fih);
        putF(p, 416, fiw / fr);
        putF(p, 420, fih / fh);
        putF(p, 424, 1.0f / fr);
        putF(p, 428, 1.0f / fh);
        putF(p, 432, 1.0f / fiw);
        putF(p, 436, 1.0f / fih);
        put32(p, 468, rw - 1);
        put32(p, 472, rh - 1);
        put32(p, 488, iw);
        put32(p, 492, ih);
        put32(p, 496, cw);
        put32(p, 500, ch);
        putF(p, 504, 1.0f / fiw);
        putF(p, 508, 1.0f / fih);
        putF(p, 512, 1.0f / float(2 * qw));
        putF(p, 516, 1.0f / float(2 * qh));
        put32(p, 1076, rw);
        put32(p, 1080, rh);
        uint64_t netOut = 0;
        for (const auto &[offset, units] : model.stages[R2Dec0B].pointers)
            if (offset == 64)
                netOut = uint64_t(std::llround(units * double(unit)));
        const uint32_t postCount = ((maxW + 15) / 16) * ((maxH + 31) / 32);
        for (size_t i = 0; i < tiles; ++i)
        {
            auto &o = postBlocks[i];
            put64(o, 0, netOut);
            put32(o, 8, w0);
            put64(o, 104, slotAt[tileSlot(i)]);
            put64(o, 112, slotAt[tileSlot(i)] + tileB);
            putF(o, 132, fr / fiw);
            putF(o, 136, fh / fih);
            put32(o, 148, iw);
            put32(o, 152, ih);
            putF(o, 160, 1.0f / fiw);
            putF(o, 164, 1.0f / fih);
            put32(o, 168, rw);
            put32(o, 172, rh);
            put32(o, 176, src[i].x0);
            put32(o, 180, src[i].y0);
            put32(o, 184, src[i].x1);
            put32(o, 188, src[i].y1);
            put32(o, 192, maxW);
            put32(o, 196, maxH);
            putF(o, 200, 1.0f / fr);
            putF(o, 204, 1.0f / fh);
            put32(o, 272, 0xffffffffu);
            put32(o, 276, 0);
            put32(o, 280, uint32_t(i));
            for (auto *wb : {&downBlocks[i], model.particles ? &downPBlocks[i] : nullptr})
            {
                if (!wb)
                    continue;
                auto &w = *wb;
                put32(w, 0, iw);
                put32(w, 4, ih);
                put32(w, 8, ow);
                put32(w, 12, oh);
                put32(w, 16, ow);
                put32(w, 20, oh);
                put64(w, 24, slotAt[tileSlot(i)]);
                put64(w, 32, slotAt[tileSlot(i)] + tileB);
                put32(w, 96, rw - 1);
                put32(w, 100, rh - 1);
                putF(w, 104, fr / fow);
                putF(w, 108, fh / foh);
                putF(w, 112, fow / fiw);
                putF(w, 116, foh / fih);
                put32(w, 128, dst[i].x0);
                put32(w, 132, dst[i].y0);
                put32(w, 136, dst[i].x1);
                put32(w, 140, dst[i].y1);
                put32(w, 144, maxW);
                put32(w, 148, maxH);
                put32(w, 152, src[i].x0);
                put32(w, 156, src[i].y0);
                put32(w, 160, src[i].x1);
                put32(w, 164, src[i].y1);
                put32(w, 184, uint32_t(i));
                put32(w, 188, postCount);
                put32(w, 192, uint32_t(i) + 2);
            }
        }
        put32(pooledBlock, 16, cw);
        put32(pooledBlock, 20, ch);
        put32(pooledBlock, 24, qw);
        put32(pooledBlock, 28, qh);
        // descriptor blocks and the per-binding parameter buffers
        slotsPerBind = 0;
        for (unsigned stage = 0; stage < R2Network; ++stage)
            slotsPerBind += 2 * UINT(model.stages[stage].layout.size());
        if (model.gather)
            slotsPerBind += 2 * UINT(model.stages[R2Gather].layout.size());
        slotsPerBind += 2 * UINT(tiles) * UINT(model.stages[R2Post].layout.size());
        slotsPerBind += UINT(tiles) * UINT(model.stages[R2Down].layout.size());
        if (model.particles)
            slotsPerBind += 2 * UINT(tiles) * UINT(model.stages[R2DownP].layout.size());
        slotsPerBind += UINT(model.stages[R2Pooled].layout.size());
        slotsPerBind += UINT(model.exp0B.size()) + 2 * UINT(model.exp1B.size()) + 2 * UINT(model.exp2B.size());
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, R2BindingBlocks * slotsPerBind,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        DXCHECK(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
        increment = device->GetDescriptorHandleIncrementSize(hd.Type);
    }
    ~R2Engine()
    {
        for (auto &p : paramCopies)
            if (p.staging && p.mapped)
                p.staging->Unmap(0, nullptr);
    }

    void setFrame(const RFrameImages &images, const RFrameSettings &settings, FrameOutput output)
    {
        if (recorded != SIZE_MAX)
            throw std::runtime_error("report the previous RR recording first");
        for (auto *t : {&images.color, &images.motion, &images.depth, &images.normals,
                        &images.diffuseAlbedo, &images.specularAlbedo})
            if (!t->resource)
                throw std::runtime_error("RR requires color, motion, depth, normals and albedos");
        if (!settings.roughnessPacked && !images.roughness.resource)
            throw std::runtime_error("RR requires roughness unless packed in normals");
        if (!output.resource || viewFormat({output.resource, output.format}) != DXGI_FORMAT_R16G16B16A16_FLOAT)
            throw std::runtime_error("RR output must be RGBA16F");
        auto od = output.resource->GetDesc();
        if (od.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || od.SampleDesc.Count != 1 || od.Width != ow ||
            od.Height != oh || !(od.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            throw std::runtime_error("RR output must match the output size and allow UAV access");
        auto same = [](const FrameTexture &a, const FrameTexture &b) {
            return a.resource == b.resource && a.format == b.format;
        };
        auto list = [](const RFrameImages &i) {
            return std::array<FrameTexture, 11>{i.color,        i.motion,           i.depth,     i.normals,
                                                i.roughness,    i.diffuseAlbedo,    i.specularAlbedo,
                                                i.specularHitDistance, i.disocclusionMask, i.subsurfaceGuide,
                                                i.colorBeforeParticles};
        };
        auto inputs = list(images);
        for (auto &t : inputs)
            if (t.resource == output.resource)
                throw std::runtime_error("RR input/output alias");
        const bool rst = settings.reset || frame == 0;
        if (images.colorBeforeParticles.resource && !model.particles)
            throw std::runtime_error(
                "this preset F model has no particle-layer downsample stage (downp)");
        if (images.colorBeforeParticles.resource)
            for (auto &t : pb)
                if (!t)
                    t = tex(ow, oh, DXGI_FORMAT_R16G16B16A16_FLOAT);
        auto found = std::find_if(bindings.begin(), bindings.end(), [&](const Bindings &b) {
            auto old = list(b.images);
            for (size_t i = 0; i < old.size(); ++i)
                if (!same(old[i], inputs[i]))
                    return false;
            return b.output.resource == output.resource && b.output.format == output.format;
        });
        if (found == bindings.end())
        {
            for (auto &b : bindings)
                if (b.inFlight && b.fence && b.fence->GetCompletedValue() >= b.value)
                {
                    b.inFlight = false;
                    b.fence.Reset();
                }
            size_t at = bindings.size();
            if (at >= R2BindingBlocks)
            {
                at = 0;
                while (at < bindings.size() && (bindings[at].pending || bindings[at].inFlight))
                    ++at;
                if (at == bindings.size())
                    throw std::runtime_error("too many RR input sets in flight");
            }
            Bindings b;
            b.images = images;
            b.output = output;
            for (auto &t : inputs)
                if (t.resource)
                    b.retained.emplace_back(t.resource);
            b.retained.emplace_back(output.resource);
            b.base = UINT(at) * slotsPerBind;
            b.paramAll = buf(paramBytes, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            bind(b);
            if (at == bindings.size())
                bindings.push_back(std::move(b));
            else
            {
                rest.erase(bindings[at].paramAll.Get());
                bindings[at] = std::move(b);
            }
            found = bindings.begin() + at;
        }
        current = size_t(found - bindings.begin());
        std::copy_n(prevJitter, 2, savedJitter);
        const uint32_t f = frame;
        const float fr = float(rw), fh = float(rh), fiw = float(iw), fih = float(ih);
        const float fow = float(ow), foh = float(oh);
        const float pj[2] = {rst ? 0.0f : prevJitter[0], rst ? 0.0f : prevJitter[1]};
        for (unsigned stage = 0; stage < R2Network; ++stage)
            put32(blocks[stage], 80, r2Shift(stage, f));
        auto &p = blocks[R2Enc0];
        putF(p, 392, pj[0]);
        putF(p, 396, pj[1]);
        putF(p, 400, settings.jitter[0]);
        putF(p, 404, settings.jitter[1]);
        putF(p, 440, pj[0] / fr);
        putF(p, 444, pj[1] / fh);
        putF(p, 448, settings.motionScale[0] * (fiw / fr));
        putF(p, 452, settings.motionScale[1] * (fih / fh));
        put32(p, 464, (rst ? 256u : 0u) | (settings.roughnessPacked ? 65536u : 0u));
        put32(p, 568, (settings.depthInverted ? 1u : 0u) | (images.specularHitDistance.resource ? 256u : 0u) |
                          (images.disocclusionMask.resource ? 0x1000000u : 0u));
        p[1028] = settings.hardwareDepth ? 1u : 0u;
        Mat4 view, clip;
        for (int i = 0; i < 16; ++i)
        {
            view[i] = settings.worldToView[i];
            clip[i] = settings.viewToClip[i];
        }
        view[12] = view[13] = view[14] = 0.0;
        const Mat4 rc = mul(view, clip), inv = inverse(rc);
        const Mat4 toNdc{2.0 / rw, 0, 0, 0, 0, -2.0 / rh, 0, 0, 0, 0, 1, 0, -1, 1, 0, 1};
        const Mat4 toPixels{rw / 2.0, 0, 0, 0, 0, -(rh / 2.0), 0, 0, 0, 0, 1, 0, rw / 2.0, rh / 2.0, 0, 1};
        putMat(p, 644, rc);
        putMat(p, 708, clip);
        putMat(p, 772, clip);
        putMat(p, 836, inv);
        putMat(p, 900, mul(toNdc, inv));
        putMat(p, 964, mul(rc, toPixels));
        putF(p, 804, settings.jitter[0] * (2.0f / fr));
        putF(p, 808, -settings.jitter[1] * (2.0f / fh));
        put32(p, 1032, f & 7u);
        for (size_t i = 0; i < postBlocks.size(); ++i)
        {
            auto &o = postBlocks[i];
            putF(o, 140, settings.jitter[0]);
            putF(o, 144, settings.jitter[1]);
            put32(o, 208, rst ? 1u : 0u);
            for (auto *wb : {&downBlocks[i], model.particles ? &downPBlocks[i] : nullptr})
            {
                if (!wb)
                    continue;
                auto &w = *wb;
                putF(w, 120, settings.jitter[0]);
                putF(w, 124, settings.jitter[1]);
                putF(w, 288, settings.jitter[0] * (fow / fr));
                putF(w, 292, settings.jitter[1] * (foh / fh));
                put32(w, 196, wb == &downBlocks[i] ? (rst ? 1u : 0u) : (rst ? 0x010101u : 0x010000u));
            }
        }
        exposure.reset = rst ? 1u : 0u;
        // texture and surface handles: 1 where something is bound, 0 where NGX leaves the argument null
        auto handles = [&](unsigned stage, std::vector<uint8_t> &block) {
            const auto &st = model.stages[stage];
            for (uint32_t offset : st.textures)
                put64(block, offset, resolve(stage, false, offset, 0, images, output.resource).resource ? 1u : 0u);
            for (const auto &[offset, format] : st.surfaces)
                put64(block, offset, resolve(stage, true, offset, 0, images, output.resource).resource ? 1u : 0u);
            for (uint32_t offset : st.buffers)
                put64(block, offset, 0);
        };
        for (unsigned stage = 0; stage < R2Network; ++stage)
            handles(stage, blocks[stage]);
        if (model.gather)
            handles(R2Gather, blocks[R2Enc0]);
        for (auto &o : postBlocks)
            handles(R2Post, o);
        for (auto &w : downBlocks)
            handles(R2Down, w);
        for (auto &w : downPBlocks)
            handles(R2DownP, w);
        handles(R2Pooled, pooledBlock);
        prevJitter[0] = settings.jitter[0];
        prevJitter[1] = settings.jitter[1];
        frameSet = true;
    }

    size_t acquireParamCopy()
    {
        for (auto &p : paramCopies)
            if (p.used && p.fence && p.fence->GetCompletedValue() >= p.value)
            {
                p.used = false;
                p.fence.Reset();
            }
        for (size_t i = 0; i < paramCopies.size(); ++i)
            if (!paramCopies[i].used)
            {
                auto &p = paramCopies[i];
                p.used = true;
                p.fence.Reset();
                p.value = 0;
                return i;
            }
        ParamCopy p;
        p.staging = buffer(device, paramBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        DXCHECK(p.staging->Map(0, nullptr, &p.mapped));
        p.used = true;
        paramCopies.push_back(std::move(p));
        return paramCopies.size() - 1;
    }

    // The frame's parameter blocks (every network stage's, the post/down/pooled passes' and the exposure's) and the
    // exposure passes: the front of every frame (recordFrame's first stages, and the external network's front).
    void uploadParameters(ID3D12GraphicsCommandList *list, const FrameProfiler *profiler, bool rebindAfterMark)
    {
        auto &b = bindings[current];
        ID3D12DescriptorHeap *heaps[] = {heap.Get()};
        auto stamp = [&](uint32_t i) {
            if (profiler && profiler->mark)
                profiler->mark(list, i, profiler->user);
            if (rebindAfterMark)
                list->SetDescriptorHeaps(1, heaps);
        };
        stamp(0);
        // the whole frame's parameter blocks: written into a free staging buffer, then copied to the descriptor's stable
        // parameter buffer by the command list (no per-word inline writes)
        recordingParam = acquireParamCopy();
        auto &copy = paramCopies[recordingParam];
        auto *out = static_cast<uint8_t *>(copy.mapped);
        for (unsigned i = 0; i < R2Network; ++i)
            std::memcpy(out + paramAt[i], blocks[i].data(), blocks[i].size());
        for (size_t i = 0; i < postBlocks.size(); ++i)
        {
            std::memcpy(out + postAt[i], postBlocks[i].data(), postBlocks[i].size());
            std::memcpy(out + downAt[i], downBlocks[i].data(), downBlocks[i].size());
        }
        for (size_t i = 0; i < downPBlocks.size(); ++i)
            std::memcpy(out + downPAt[i], downPBlocks[i].data(), downPBlocks[i].size());
        std::memcpy(out + pooledAt, pooledBlock.data(), pooledBlock.size());
        std::memcpy(out + expAt, &exposure, sizeof(exposure));
        use(b.paramAll.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        flush(list);
        list->CopyBufferRegion(b.paramAll.Get(), 0, copy.staging.Get(), 0, paramBytes);
        const uint32_t cur = frame & 1u;
        if (model.exposureOne)
            dispatch(list, model.exposureOne, b.expOne[cur], exposure.blocksX, exposure.blocks / exposure.blocksX);
        else if (model.exposure[0])
        {
            dispatch(list, model.exposure[0], b.exp0, exposure.blocksX, exposure.blocks / exposure.blocksX);
            dispatch(list, model.exposure[1], b.exp1[cur], 1, 1);
        }
        stamp(1);
    }
    // The network stages [first, last): the engine's own dispatches, or (an external network's own stages) the blocks
    // and grids handed to it, which it runs between the front and the back.
    void dispatchStages(ID3D12GraphicsCommandList *list, unsigned first, unsigned last, const FrameProfiler *profiler,
                        bool rebindAfterMark)
    {
        auto &b = bindings[current];
        ID3D12DescriptorHeap *heaps[] = {heap.Get()};
        auto stamp = [&](uint32_t i) {
            if (profiler && profiler->mark)
                profiler->mark(list, i, profiler->user);
            if (rebindAfterMark)
                list->SetDescriptorHeaps(1, heaps);
        };
        const uint32_t f = frame, cur = f & 1u;
        for (unsigned stage = first; stage < last; ++stage)
        {
            const int l = R2Level[stage];
            const uint32_t shift = r2Shift(stage, f), sx = shift & 0xffffu, sy = shift >> 16;
            const uint32_t gx = (lw(l) + sx + 7) / 8;
            const uint32_t gy = (lh(l) + sy + 7) / 8;
            if (stage == R2Enc0)
                stageDump(list, "00-start");
            if (stage == R2Enc0 && model.gather)
            {
                dispatch(list, model.stages[R2Gather].pipeline, b.gather[cur], gx, gy);
                if (std::getenv("D4R_RR_TMP_FINE")) stamp(14); // TMP
            }
            // The external network's stage: the same block the engine would bind, its grid and (its host derives the
            // windows from) this frame's shift. Stages 1 - 14 are always its own; encoder 0 only when the network runs
            // it too (a package with hip/rr2_enc0.hsaco, d3d12/hip_net_rr.cpp), after the gather pass above.
            if (external && (stage >= 1 ? stage <= 14 : external->network->runs(stage)))
            {
                external->network->stage(stage, blocks[stage].data(), blocks[stage].size(), gx, gy, shift);
                continue;
            }
            const auto &st = model.stages[stage];
            if (split[stage])
            {
                const uint32_t parts[4] = {st.heads, 1, st.branches, 1};
                for (int k = 0; k < 4; ++k)
                    dispatch(list, st.phases[k], b.network[stage][cur], gx * parts[k], gy);
            }
            else
                dispatch(list, st.pipeline, b.network[stage][cur], (gx + st.group - 1) / st.group, gy);
            stageDump(list, (stage < 9 ? "0" : "") + std::to_string(stage + 1) + "-" + R2Names[stage]);
            // profiling: one boundary per encoder stage, then one per decoder pair (14 stamps as presets D/E)
            if (std::getenv("D4R_RR_TMP_FINE") && stage > R2Enc5 && (stage - R2Enc5) % 2 == 1) stamp(15 + (stage - R2Enc5) / 2); // TMP
            if (stage < R2Enc5 + 1)
                stamp(2 + stage);
            else if ((stage - R2Enc5) % 2 == 0)
                stamp(8 + (stage - (R2Enc5 + 1)) / 2);
        }
    }
    // The output passes: every tile's post and down pass (adjacent groups in alternating slots), then the pooled history
    void dispatchOutput(ID3D12GraphicsCommandList *list, const FrameProfiler *profiler, bool rebindAfterMark)
    {
        auto &b = bindings[current];
        ID3D12DescriptorHeap *heaps[] = {heap.Get()};
        auto stamp = [&](uint32_t i) {
            if (profiler && profiler->mark)
                profiler->mark(list, i, profiler->user);
            if (rebindAfterMark)
                list->SetDescriptorHeaps(1, heaps);
        };
        const uint32_t cur = frame & 1u;
        const bool particles = b.images.colorBeforeParticles.resource != nullptr;
        const size_t tiles = src.size(), groups = (tiles + tileGroup - 1) / tileGroup;
        // Adjacent output groups use alternating tile slots: a previous down pass can run beside the next post pass.
        for (size_t g = 0; g <= groups; ++g)
        {
            if (g > 0)
                for (size_t i = (g - 1) * tileGroup; i < std::min(tiles, g * tileGroup); ++i)
                    dispatch(list, model.stages[particles ? R2DownP : R2Down].pipeline,
                             particles ? b.downP[i][cur] : b.down[i], (dst[i].x1 - dst[i].x0 + 15) / 16,
                             (dst[i].y1 - dst[i].y0 + 15) / 16, false);
            for (size_t i = g * tileGroup; i < std::min(tiles, (g + 1) * tileGroup); ++i)
                dispatch(list, model.stages[R2Post].pipeline, b.post[i][cur], (maxW + 15) / 16, (maxH + 31) / 32,
                         false);
            if (g < groups)
                uav();
        }
        dispatch(list, model.stages[R2Pooled].pipeline, b.pooled, (qw + 15) / 16, (qh + 15) / 16, false);
        uav();
        stamp(13);
    }
    // every engine resource back to its resting state; the recording is then reported by submitted or discarded
    void completeRecording(ID3D12GraphicsCommandList *list)
    {
        auto &b = bindings[current];
        for (auto &[r, state] : now)
            if (state != rest.at(r))
                barriers.push_back(transition(r, state, rest.at(r)));
        flush(list);
        b.pending = true;
        recorded = current;
        ++frame;
    }
    void recordFrame(ID3D12GraphicsCommandList *list, const FrameProfiler &profiler, bool rebindAfterMark)
    {
        if (!frameSet || recorded != SIZE_MAX)
            throw std::runtime_error("set RR frame before recording");
        if (external)
            throw std::runtime_error("this engine's network is external: record the front and the back");
        now = rest;
        ID3D12DescriptorHeap *heaps[] = {heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        uploadParameters(list, &profiler, rebindAfterMark);
        dispatchStages(list, 0, R2Network, &profiler, rebindAfterMark);
        dispatchOutput(list, &profiler, rebindAfterMark);
        completeRecording(list);
    }
    // The front of an external network's frame: the parameters, the exposure, the gather pass, then encoder 0 and the
    // network's fourteen stages handed over with this frame's blocks and grids (encoder 0 stays here when the network
    // does not run it). The engine's own resources are back in their resting states when this returns; submit it and
    // wait on a fence before the network runs.
    void recordFront(ID3D12GraphicsCommandList *list)
    {
        if (!frameSet || recorded != SIZE_MAX)
            throw std::runtime_error("set RR frame before recording");
        if (!external)
            throw std::runtime_error("recordFront needs an engine with an external network");
        if (frontDone)
            throw std::runtime_error("recordFront before recordBack");
        now = rest;
        ID3D12DescriptorHeap *heaps[] = {heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        uploadParameters(list, nullptr, false);
        // up to decoder 0's second block, which belongs to the back: it reads the network's output
        dispatchStages(list, 0, R2Dec0B, nullptr, false);
        for (auto &[r, state] : now)
            if (state != rest.at(r))
                barriers.push_back(transition(r, state, rest.at(r)));
        flush(list);
        frontDone = true;
    }
    // The back: decoder 0's second block (which reads the network's output in the arena), then the output passes.
    // Record and submit it only after the network has finished.
    void recordBack(ID3D12GraphicsCommandList *list)
    {
        if (!frontDone)
            throw std::runtime_error("recordFront before recordBack");
        now = rest;
        ID3D12DescriptorHeap *heaps[] = {heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        // the network wrote the arena after the front retired on the CPU: this UAV barrier invalidates the caches
        // decoder 0's second block reads it through
        D3D12_RESOURCE_BARRIER uavBarrier{};
        uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uavBarrier.UAV = {arena.Get()};
        list->ResourceBarrier(1, &uavBarrier);
        dispatchStages(list, R2Dec0B, R2Network, nullptr, false);
        dispatchOutput(list, nullptr, false);
        completeRecording(list);
        frontDone = false;
    }
    void submitted(ID3D12Fence *fence, uint64_t value)
    {
        if (recorded == SIZE_MAX)
            throw std::runtime_error("no RR recording");
        if (!fence)
            throw std::runtime_error("Ray Reconstruction preset F requires a submission fence");
        auto &b = bindings[recorded];
        b.pending = false;
        b.inFlight = true;
        b.fence = fence;
        b.value = value;
        writeStageDumps(fence, value); // TMP
        if (recordingParam != SIZE_MAX)
        {
            auto &p = paramCopies[recordingParam];
            p.fence = fence;
            p.value = value;
            recordingParam = SIZE_MAX;
        }
        recorded = SIZE_MAX;
        frameSet = false;
    }
    void discarded()
    {
        if (recorded == SIZE_MAX)
            throw std::runtime_error("no RR recording");
        bindings[recorded].pending = false;
        if (recordingParam != SIZE_MAX)
        {
            auto &p = paramCopies[recordingParam];
            p.used = false;
            p.fence.Reset();
            recordingParam = SIZE_MAX;
        }
        recorded = SIZE_MAX;
        frameSet = false;
        --frame;
        std::copy_n(savedJitter, 2, prevJitter);
    }
    bool idle() const
    {
        if (recorded != SIZE_MAX)
            return false;
        for (const auto &b : bindings)
            if (b.inFlight && (!b.fence || b.fence->GetCompletedValue() < b.value))
                return false;
        return true;
    }
    std::vector<REngine::DebugImage> debugImages() const
    {
        std::vector<REngine::DebugImage> out;
        auto put = [&](const std::string &name, const ComPtr<ID3D12Resource> &r) {
            if (!r)
                return;
            const auto d = r->GetDesc();
            out.push_back({name, r.Get(), d.Format, uint32_t(d.Width), d.Height});
        };
        for (int i = 0; i < 2; ++i)
        {
            const std::string n = std::to_string(i);
            put("feat" + n, feat[i]);
            put("hist" + n, hist[i]);
            put("e" + n, e[i]);
            put("mv" + n, mv[i]);
        }
        put("rec", rec);
        put("ka", ka);
        put("kb", kb);
        put("kc", kc);
        put("half", half);
        put("quarter", quarter);
        return out;
    }
    UINT writeBlock(const std::vector<Binding> &layout, const std::vector<Slot> &slots, UINT first)
    {
        r2WriteTable(device, cpu(first), increment, layout, slots);
        tables[first] = slots;
        return first + UINT(slots.size());
    }
    void bind(Bindings &b)
    {
        UINT first = b.base;
        for (unsigned stage = 0; stage < R2Network; ++stage)
            for (uint32_t par = 0; par < 2; ++par)
            {
                auto slots = stageSlots(model.stages[stage], par, b.images, b.output.resource, false);
                slots[0] = Slot{b.paramAll.Get(), paramAt[stage]};
                b.network[stage][par] = first;
                first = writeBlock(model.stages[stage].layout, slots, first);
            }
        if (model.gather)
            for (uint32_t par = 0; par < 2; ++par)
            {
                auto slots = stageSlots(model.stages[R2Gather], par, b.images, b.output.resource, true);
                slots[0] = Slot{b.paramAll.Get(), paramAt[R2Enc0]};
                b.gather[par] = first;
                first = writeBlock(model.stages[R2Gather].layout, slots, first);
            }
        const size_t tiles = src.size();
        for (size_t i = 0; i < tiles; ++i)
        {
            std::array<UINT, 2> post{};
            for (uint32_t par = 0; par < 2; ++par)
            {
                auto slots = stageSlots(model.stages[R2Post], par, b.images, b.output.resource, false);
                slots[0] = Slot{b.paramAll.Get(), postAt[i]};
                post[par] = first;
                first = writeBlock(model.stages[R2Post].layout, slots, first);
            }
            b.post.push_back(post);
        }
        for (size_t i = 0; i < tiles; ++i)
        {
            auto slots = stageSlots(model.stages[R2Down], 0, b.images, b.output.resource, false);
            slots[0] = Slot{b.paramAll.Get(), downAt[i]};
            b.down.push_back(first);
            first = writeBlock(model.stages[R2Down].layout, slots, first);
        }
        if (b.images.colorBeforeParticles.resource)
            for (size_t i = 0; i < tiles; ++i)
            {
                std::array<UINT, 2> downP{};
                for (uint32_t par = 0; par < 2; ++par)
                {
                    auto slots = stageSlots(model.stages[R2DownP], par, b.images, b.output.resource, false);
                    slots[0] = Slot{b.paramAll.Get(), downPAt[i]};
                    downP[par] = first;
                    first = writeBlock(model.stages[R2DownP].layout, slots, first);
                }
                b.downP.push_back(downP);
            }
        {
            auto slots = stageSlots(model.stages[R2Pooled], 0, b.images, b.output.resource, false);
            slots[0] = Slot{b.paramAll.Get(), pooledAt};
            b.pooled = first;
            first = writeBlock(model.stages[R2Pooled].layout, slots, first);
        }
        b.exp0 = first;
        first = writeBlock(model.exp0B,
                           {Slot{b.paramAll.Get(), expAt},
                            Slot{b.images.color.resource, 0, b.images.color.format, true}, Slot{sums.Get(), 0}},
                           first);
        for (uint32_t par = 0; par < 2; ++par)
        {
            b.exp1[par] = first;
            first = writeBlock(model.exp1B,
                               {Slot{b.paramAll.Get(), expAt}, Slot{sums.Get()}, Slot{values.Get()},
                                Slot{e[par ^ 1].Get()}, Slot{e[par].Get()}},
                               first);
            b.expOne[par] = first;
            first = writeBlock(model.exp2B,
                               {Slot{b.paramAll.Get(), expAt},
                                Slot{b.images.color.resource, 0, b.images.color.format, true}, Slot{sums.Get()},
                                Slot{values.Get()}, Slot{e[par ^ 1].Get()}, Slot{e[par].Get()}},
                               first);
        }
    }
};
} // namespace
