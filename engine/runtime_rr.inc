// Ray Reconstruction (DLSS-D presets D and E): the runtime around engine/rr's stage shaders. Included by runtime.cpp
// inside namespace d4r, after runtime_m.inc (it uses that file's Tex, bindRange and exposureGrid).
//
// Every stage reads and writes NGX's own layouts (engine/rr/README.md): its parameter block is NVIDIA's argument
// block with arena byte offsets in place of pointers, the network's activations live in one plane arena at NGX's
// offsets, and the per-level surfaces have NGX's sizes. So a model is NGX's weight images plus two argument-block
// templates (enc0's and the output stage's constants), and this file fills every field that depends on the frame or
// on the geometry.
namespace {
constexpr unsigned RStages = 12;
enum : unsigned { REnc0, REnc1, REnc2, REnc3, REnc4, REnc5, RDec4, RDec3, RDec2, RDec1, RDec0, RHkpn };
constexpr const char* RNames[RStages] = {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5",
                                         "dec4", "dec3", "dec2", "dec1", "dec0", "hkpn"};
constexpr uint32_t RParamBytes[RStages] = {1032, 144, 144, 144, 144, 144, 144, 144, 144, 144, 152, 480};
constexpr int RLevel[RStages] = {0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, -1};
// The plane arena: 70 * P0 bytes, P0 = the render size rounded up to 32 on both axes (W0 x H0). Every region starts
// at a multiple of P0 / 32 (NGX's layout, the same at 640x360 and 800x450): per stage, the arena offset of its
// input (argument 8), skip (24), output (48) and merged output (56) in units of P0 / 32, -1 where unused.
constexpr uint32_t RArenaUnits = 70 * 32;
constexpr int RArena[RStages][4] = {
    {-1, -1, 0, 1024},        // enc0: 32-channel tokens (dec0's skip), the 2x2 merge for enc1
    {1024, -1, 1536, 2048},   // enc1
    {2048, -1, 1024, 1216},   // enc2
    {1216, -1, 1280, 1344},   // enc3
    {1344, -1, 1216, 1236},   // enc4
    {1236, -1, 1241, -1},     // enc5, the bottleneck
    {1241, 1216, 1344, -1},   // dec4
    {1344, 1280, 1216, -1},   // dec3
    {1216, 1024, 2048, -1},   // dec2
    {2048, 1536, 1024, -1},   // dec1
    {1024, 0, 1536, -1},      // dec0: its output is the output stage's argument 256
    {-1, -1, -1, -1}};
constexpr uint32_t RHkpnArena = 1536;
// NGX's window shift of a stage, packed (x | y << 16) at argument 80: it alternates every frame, the decoders opposite
// to the encoders; the bottleneck has none
uint32_t rShift(unsigned stage, uint32_t frame) {
    if (stage == REnc5 || stage == RHkpn) return 0;
    const bool even = ((frame & 1u) == 0) == (stage <= REnc5);
    return even ? 0x00060002u : 0x00020006u;
}
void put32(std::vector<uint8_t>& p, uint32_t offset, uint32_t value) { std::memcpy(p.data() + offset, &value, 4); }
void putF(std::vector<uint8_t>& p, uint32_t offset, float value) { std::memcpy(p.data() + offset, &value, 4); }
void put64(std::vector<uint8_t>& p, uint32_t offset, uint64_t value) { std::memcpy(p.data() + offset, &value, 8); }
using Mat4 = std::array<double, 16>;
Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
    return r;
}
Mat4 inverse(const Mat4& m) {
    // Gauss-Jordan with partial pivoting; a singular camera falls back to the identity
    Mat4 a = m, r{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    for (int c = 0; c < 4; ++c) {
        int pivot = c;
        for (int i = c + 1; i < 4; ++i) if (std::fabs(a[i * 4 + c]) > std::fabs(a[pivot * 4 + c])) pivot = i;
        if (std::fabs(a[pivot * 4 + c]) < 1e-30) return Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        for (int j = 0; j < 4; ++j) { std::swap(a[c * 4 + j], a[pivot * 4 + j]); std::swap(r[c * 4 + j], r[pivot * 4 + j]); }
        const double d = a[c * 4 + c];
        for (int j = 0; j < 4; ++j) { a[c * 4 + j] /= d; r[c * 4 + j] /= d; }
        for (int i = 0; i < 4; ++i) {
            if (i == c) continue;
            const double f = a[i * 4 + c];
            for (int j = 0; j < 4; ++j) { a[i * 4 + j] -= f * a[c * 4 + j]; r[i * 4 + j] -= f * r[c * 4 + j]; }
        }
    }
    return r;
}
void putMat(std::vector<uint8_t>& p, uint32_t offset, const Mat4& m) {
    for (int i = 0; i < 16; ++i) putF(p, offset + 4 * i, static_cast<float>(m[i]));
}
} // namespace

struct RModel::Impl {
    Device device;
    uint32_t preset = 0;
    // per stage, the shader's bindings after the three buffers (engine/rr/README.md): texture argument offsets, then
    // surface argument offsets with their GLSL image format, both ascending
    std::array<std::vector<uint32_t>, RStages> textures;
    std::array<std::vector<std::pair<uint32_t, std::string>>, RStages> surfaces;
    std::array<Buffer, RStages> weights, uploads;
    std::array<Pipeline, RStages> stages;
    std::array<Pipeline, 2> exposure;
    std::vector<uint8_t> enc0Template, hkpnTemplate;
    bool uploaded = false;
    Impl(VkPhysicalDevice physical, VkDevice dev) : device(physical, dev) {}
};
RModel::RModel(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache)
    : impl(std::make_unique<Impl>(physical, device)) {
    auto& m = *impl;
    // rr.txt: "D4RR0001", "preset N", one "stage NAME SHADER.spv SUBGROUP WEIGHTS TEXTURES SURFACES" line per stage in
    // RNames order (TEXTURES: comma-separated argument offsets or "-"; SURFACES: OFFSET:FORMAT,... or "-") and
    // "template enc0|hkpn FILE" (the argument blocks whose constants the runtime keeps)
    auto text = read(directory + "/rr.txt");
    std::istringstream lines(std::string(text.begin(), text.end()));
    std::string line;
    if (!std::getline(lines, line) || line.rfind("D4RR0001", 0) != 0) throw std::runtime_error("not a Ray Reconstruction model (rr.txt D4RR0001)");
    unsigned stage = 0;
    const auto b = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, t = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    auto split = [](const std::string& list) {
        std::vector<std::string> items;
        if (list == "-") return items;
        std::istringstream in(list);
        for (std::string item; std::getline(in, item, ',');) items.push_back(item);
        return items;
    };
    while (std::getline(lines, line)) {
        std::istringstream f(line);
        std::string key; f >> key;
        if (key == "preset") f >> m.preset;
        else if (key == "stage") {
            std::string name, shader, weightFile, textureList, surfaceList, option; uint32_t subgroup = 32;
            f >> name >> shader >> subgroup >> weightFile >> textureList >> surfaceList >> option;
            // "noopt": the stage is compiled without the driver's optimizations. RADV's ACO scheduler (Mesa 26.2,
            // gfx1201) reorders the translated stages' cross-lane code so that enc1 lands at 20-43 dB against the
            // reference; VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT gives 42-64 dB at about twice the time.
            const VkPipelineCreateFlags flags = option == "noopt" ? VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT : 0;
            if (stage >= RStages || name != RNames[stage] || (subgroup != 32 && subgroup != 64) || surfaceList.empty())
                throw std::runtime_error("rr.txt: stages out of order or invalid: " + line);
            for (const auto& item : split(textureList)) m.textures[stage].push_back(static_cast<uint32_t>(std::stoul(item)));
            for (const auto& item : split(surfaceList)) {
                const auto colon = item.find(':');
                if (colon == std::string::npos) throw std::runtime_error("rr.txt: surface without a format: " + line);
                m.surfaces[stage].emplace_back(static_cast<uint32_t>(std::stoul(item.substr(0, colon))), item.substr(colon + 1));
            }
            std::vector<VkDescriptorType> types{b, b, b};
            types.insert(types.end(), m.textures[stage].size(), t);
            types.insert(types.end(), m.surfaces[stage].size(), s);
            m.stages[stage].create(device, directory + "/" + shader, types, subgroup, cache, flags);
            auto blob = read(directory + "/" + weightFile);
            const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            m.weights[stage].create(m.device, blob.size(), usage);
            m.uploads[stage].create(m.device, blob.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            std::memcpy(m.uploads[stage].mapped, blob.data(), blob.size());
            ++stage;
        } else if (key == "template") {
            std::string which, file; f >> which >> file;
            auto bytes = read(directory + "/" + file);
            auto& target = which == "enc0" ? m.enc0Template : m.hkpnTemplate;
            if (bytes.size() != RParamBytes[which == "enc0" ? REnc0 : RHkpn]) throw std::runtime_error("rr.txt: bad template " + file);
            target.assign(bytes.begin(), bytes.end());
        }
    }
    if (stage != RStages || m.enc0Template.empty() || m.hkpnTemplate.empty() || (m.preset != 4 && m.preset != 5))
        throw std::runtime_error("rr.txt: incomplete Ray Reconstruction model");
    m.exposure[0].create(device, directory + "/exposure_rr0.spv", {b, t, b}, 64, cache);
    m.exposure[1].create(device, directory + "/exposure_rr1.spv", {b, b, b, s, s}, 64, cache);
}
RModel::~RModel() = default;
bool RModel::uploadRecorded() const { return impl->uploaded; }
uint32_t RModel::preset() const { return impl->preset; }
void RModel::recordUpload(VkCommandBuffer cb) {
    auto& m = *impl;
    if (m.uploaded) throw std::runtime_error("model upload already recorded");
    for (unsigned i = 0; i < RStages; ++i) {
        VkBufferCopy copy{0, 0, m.uploads[i].size};
        vkCmdCopyBuffer(cb, m.uploads[i].handle, m.weights[i].handle, 1, &copy);
    }
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    m.uploaded = true;
}

struct REngine::Impl {
    RModel::Impl& model;
    uint32_t rw, rh, ow, oh, w0, h0;
    Buffer arena, expP, expSums, expState;
    std::array<Buffer, RStages> params;
    std::array<std::vector<uint8_t>, RStages> blocks;   // this frame's parameter blocks
    std::array<Tex, 6> s112, s120;  // per level: the encoders' surfaces (decoders' skip textures 96, 104)
    std::array<Tex, 5> s128;        // the encoders' half-resolution surface, the next encoder's texture 88
    std::array<Tex, 5> d112, d120;  // the decoders' surfaces (texture 88 of the next decoder; dec0's 112: output stage)
    std::array<Tex, 2> q, r, h, e;  // ping-pong: enc0's 304 / 224, dec0's 144 / enc0's 488, the output-size history,
                                    // and e: this frame's and the previous frame's exposure (1x1)
    Tex out;                        // the presented image (the output stage's surface 240)
    Tex zero;                       // a zeroed 1x1 texture for the handles NGX leaves null (they read zero)
    std::map<std::string, Tex> spare;   // per format, a 1x1 surface for stores NGX leaves unbound
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet exp1Set = VK_NULL_HANDLE;
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    ExposureParams exposure{};
    struct Bindings {
        RFrameImages images{};
        FrameOutput finalOutput{};
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<std::array<VkDescriptorSet, 2>, RStages> stages{};   // per stage and frame parity
        VkDescriptorSet exposure = VK_NULL_HANDLE;
    };
    // what a stage's texture or surface at an argument offset is (the capture's wiring, engine/rr/README.md), for
    // frame parity `par`; null view: NGX leaves it unbound
    struct Resource { VkImageView view = VK_NULL_HANDLE; bool linear = false; bool game = false; };
    Resource resolve(unsigned stage, bool surface, uint32_t offset, uint32_t par, const RFrameImages& in, VkImageView output) const {
        const int l = RLevel[stage];
        auto T = [](const Tex& t, bool lin = false) { return Resource{t.view, lin, false}; };
        auto G = [](VkImageView v, bool lin = false) { return Resource{v, lin, true}; };
        if (stage == REnc0) {
            if (!surface) switch (offset) {
                case 160: return G(in.color, true);   case 168: return G(in.motion);   case 184: return G(in.normals);
                case 192: return G(in.roughness);     case 200: return T(e[0]);        case 208: return T(e[1]);
                case 224: return T(q[par ^ 1], true); case 232: return G(in.diffuseAlbedo); case 240: return G(in.specularAlbedo);
                // the optional inputs (capDX, 2026-10-12): unbound without them, like NGX
                case 248: return in.specularHitDistance ? G(in.specularHitDistance) : Resource{};
                case 296: return in.disocclusionMask ? G(in.disocclusionMask) : Resource{};
                case 488: return T(r[par ^ 1], true); case 496: return T(h[par ^ 1], true); case 512: return G(in.depth);
            } else switch (offset) {
                case 112: return T(s112[0]); case 120: return T(s120[0]); case 128: return T(s128[0]); case 304: return T(q[par]);
            }
        } else if (stage <= REnc5) {
            if (!surface && offset == 88) return T(s128[l - 1], true);
            if (surface && offset == 112) return T(s112[l]);
            if (surface && offset == 120) return T(s120[l]);
            if (surface && offset == 128 && l < 5) return T(s128[l]);
        } else if (stage <= RDec0) {
            if (!surface && offset == 88) return T(stage == RDec4 ? s112[5] : d112[l + 1], true);
            if (!surface && offset == 96) return T(s112[l]);
            if (!surface && offset == 104) return T(s120[l]);
            if (surface && offset == 112) return T(d112[l]);
            if (surface && offset == 120) return T(d120[l]);
            if (surface && offset == 144 && stage == RDec0) return T(r[par]);
        } else {
            if (!surface) switch (offset) {
                case 176: return G(in.color, true); case 192: return G(in.depth); case 200: return T(e[0]); case 208: return T(e[1]);
                case 216: return T(h[par ^ 1], true); case 280: return G(in.diffuseAlbedo); case 288: return G(in.specularAlbedo);
                case 296: return T(q[par]); case 448: return T(d112[0]);
            } else switch (offset) {
                case 224: return T(h[par]); case 240: return Resource{output, false, false};
            }
        }
        return {};
    }
    std::vector<Bindings> bindings;
    const Bindings* bound = nullptr;
    float prevJitter[2] = {0, 0};
    uint32_t frame = 0;
    bool reset = true;
    explicit Impl(RModel::Impl& m) : model(m) {}
    VkFormat surfaceFormat(const std::string& f) const {
        if (f == "rgba16f") return VK_FORMAT_R16G16B16A16_SFLOAT;
        if (f == "rg16f") return VK_FORMAT_R16G16_SFLOAT;
        if (f == "r16f") return VK_FORMAT_R16_SFLOAT;
        if (f == "rgba8_snorm") return VK_FORMAT_R8G8B8A8_SNORM;
        throw std::runtime_error("Ray Reconstruction stage surface format " + f + " is not implemented");
    }
    ~Impl() {
        VkDevice d = model.device.handle;
        for (auto& b : bindings) vkDestroyDescriptorPool(d, b.pool, nullptr);
        if (pool) vkDestroyDescriptorPool(d, pool, nullptr);
        if (nearest) vkDestroySampler(d, nearest, nullptr);
        if (linear) vkDestroySampler(d, linear, nullptr);
    }
    uint32_t lw(int level) const { return w0 >> level; }
    uint32_t lh(int level) const { return h0 >> level; }
    uint64_t arenaOffset(int units) const { return static_cast<uint64_t>(units) * (static_cast<uint64_t>(w0) * h0 / 32); }
};

REngine::REngine(RModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight)
    : impl(std::make_unique<Impl>(*model.impl)) {
    auto& e = *impl;
    const Device& d = e.model.device;
    if (renderWidth < 64 || renderHeight < 64 || renderWidth > outputWidth || renderHeight > outputHeight || outputWidth > 16384 || outputHeight > 16384)
        throw std::runtime_error("unsupported Ray Reconstruction dimensions");
    e.rw = renderWidth; e.rh = renderHeight; e.ow = outputWidth; e.oh = outputHeight;
    e.w0 = (renderWidth + 31) & ~31u; e.h0 = (renderHeight + 31) & ~31u;
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    e.arena.create(d, e.arenaOffset(RArenaUnits), storage);
    for (unsigned i = 0; i < RStages; ++i) {
        e.params[i].create(d, RParamBytes[i], storage);
        e.blocks[i].assign(RParamBytes[i], 0);
    }
    e.expP.create(d, sizeof(ExposureParams), storage);
    const uint32_t gx = exposureGrid(e.rw, e.ow), gy = exposureGrid(e.rh, e.oh);
    e.exposure.blocksX = (gx + 31) / 32;
    e.exposure.blocks = e.exposure.blocksX * ((gy + 31) / 32);
    e.expSums.create(d, e.exposure.blocks * 4, storage);
    e.expState.create(d, 8, storage);
    const VkFormat half4 = VK_FORMAT_R16G16B16A16_SFLOAT;
    for (int l = 0; l < 6; ++l) { e.s112[l].create(d, e.lw(l), e.lh(l), half4); e.s120[l].create(d, e.lw(l), e.lh(l), half4); }
    for (int l = 0; l < 5; ++l) {
        e.s128[l].create(d, e.lw(l + 1), e.lh(l + 1), half4);
        e.d112[l].create(d, e.lw(l), e.lh(l), half4); e.d120[l].create(d, e.lw(l), e.lh(l), half4);
    }
    for (int i = 0; i < 2; ++i) {
        e.q[i].create(d, e.rw, e.rh, half4);
        e.r[i].create(d, e.rw, e.rh, VK_FORMAT_R8G8B8A8_SNORM);
        e.h[i].create(d, e.ow, e.oh, half4);
        e.e[i].create(d, 1, 1, VK_FORMAT_R16_SFLOAT);
    }
    e.out.create(d, e.ow, e.oh, half4);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = 0;
    VKCHECK(vkCreateSampler(d.handle, &si, nullptr, &e.nearest));
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    VKCHECK(vkCreateSampler(d.handle, &si, nullptr, &e.linear));
    e.zero.create(d, 1, 1, half4);
    for (unsigned stage = 0; stage < RStages; ++stage)
        for (const auto& [offset, format] : e.model.surfaces[stage])
            if (!e.spare.count(format)) e.spare[format].create(d, 1, 1, e.surfaceFormat(format));
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 1; pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
    VKCHECK(vkCreateDescriptorPool(d.handle, &pi, nullptr, &e.pool));
    e.exp1Set = descriptorSet(d.handle, e.pool, e.model.exposure[1].descriptors);
    bindBuffer(d.handle, e.exp1Set, 0, e.expP);
    bindBuffer(d.handle, e.exp1Set, 1, e.expSums);
    bindBuffer(d.handle, e.exp1Set, 2, e.expState);
    bindImage(d.handle, e.exp1Set, 3, e.e[0].view);
    bindImage(d.handle, e.exp1Set, 4, e.e[1].view);
    // the parameter blocks' fixed fields
    for (unsigned stage = REnc1; stage <= RDec0; ++stage) {
        auto& p = e.blocks[stage];
        const int l = RLevel[stage];
        put32(p, 0, e.lw(l)); put32(p, 4, e.lh(l));
        const int offsets[4] = {8, 24, 48, 56};
        for (int k = 0; k < 4; ++k)
            if (RArena[stage][k] >= 0) put64(p, offsets[k], e.arenaOffset(RArena[stage][k]));
        putF(p, 136, 1.0f / static_cast<float>(e.lw(l))); putF(p, 140, 1.0f / static_cast<float>(e.lh(l)));
    }
}
REngine::~REngine() = default;
VkImage REngine::outputImage() const { return impl->out.handle; }
std::vector<REngine::DebugImage> REngine::debugImages() const {
    const auto& e = *impl;
    std::vector<DebugImage> list;
    const VkFormat half4 = VK_FORMAT_R16G16B16A16_SFLOAT;
    for (int l = 0; l < 6; ++l) {
        const std::string enc = "enc" + std::to_string(l);
        list.push_back({enc + "-112", e.s112[l].handle, half4, e.lw(l), e.lh(l)});
        list.push_back({enc + "-120", e.s120[l].handle, half4, e.lw(l), e.lh(l)});
        if (l < 5) list.push_back({enc + "-128", e.s128[l].handle, half4, e.lw(l + 1), e.lh(l + 1)});
        if (l < 5) list.push_back({"dec" + std::to_string(l) + "-112", e.d112[l].handle, half4, e.lw(l), e.lh(l)});
    }
    for (int i = 0; i < 2; ++i) {
        list.push_back({"q" + std::to_string(i), e.q[i].handle, half4, e.rw, e.rh});
        list.push_back({"r" + std::to_string(i), e.r[i].handle, VK_FORMAT_R8G8B8A8_SNORM, e.rw, e.rh});
        list.push_back({"h" + std::to_string(i), e.h[i].handle, half4, e.ow, e.oh});
        list.push_back({"e" + std::to_string(i), e.e[i].handle, VK_FORMAT_R16_SFLOAT, 1, 1});
    }
    list.push_back({"out", e.out.handle, half4, e.ow, e.oh});
    return list;
}

void REngine::recordInitialize(VkCommandBuffer cb) {
    auto& e = *impl;
    std::vector<VkImage> images;
    for (auto* group : {&e.s112, &e.s120}) for (auto& t : *group) images.push_back(t.handle);
    for (auto* group : {&e.s128, &e.d112, &e.d120}) for (auto& t : *group) images.push_back(t.handle);
    for (auto* group : {&e.q, &e.r, &e.h, &e.e}) for (auto& t : *group) images.push_back(t.handle);
    images.push_back(e.out.handle);
    images.push_back(e.zero.handle);
    for (auto& [format, t] : e.spare) images.push_back(t.handle);
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (VkImage image : images) {
        imageBarrier(cb, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }
    vkCmdFillBuffer(cb, e.arena.handle, 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cb, e.expState.handle, 0, VK_WHOLE_SIZE, 0);
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    e.frame = 0; e.reset = true;
}

void REngine::setFrame(const RFrameImages& images, const RFrameSettings& settings, FrameOutput finalOutput) {
    auto& e = *impl;
    const Device& d = e.model.device;
    if (!images.color || !images.motion || !images.depth || !images.normals || !images.roughness || !images.diffuseAlbedo || !images.specularAlbedo)
        throw std::runtime_error("Ray Reconstruction needs colour, motion, depth and the four guides");
    if (finalOutput.view && finalOutput.format != VK_FORMAT_R16G16B16A16_SFLOAT)
        throw std::runtime_error("Ray Reconstruction writes RGBA16F output only");
    const bool reset = settings.reset || e.frame == 0;
    const uint32_t f = e.frame;
    const float rw = static_cast<float>(e.rw), rh = static_cast<float>(e.rh), ow = static_cast<float>(e.ow), oh = static_cast<float>(e.oh);
    const float prevJitter[2] = {reset ? 0.0f : e.prevJitter[0], reset ? 0.0f : e.prevJitter[1]};
    // the network layers: only the window shift changes
    for (unsigned stage = REnc1; stage <= RDec0; ++stage) put32(e.blocks[stage], 80, rShift(stage, f));
    // enc0 (the argument block's constants from the model; offsets as observed on harness captures, 2026-10-11)
    auto& p = e.blocks[REnc0];
    p = e.model.enc0Template;
    put32(p, 0, e.w0); put32(p, 4, e.h0);
    put64(p, 48, e.arenaOffset(RArena[REnc0][2])); put64(p, 56, e.arenaOffset(RArena[REnc0][3]));
    put32(p, 80, rShift(REnc0, f));
    putF(p, 136, 1.0f / static_cast<float>(e.w0)); putF(p, 140, 1.0f / static_cast<float>(e.h0));
    put32(p, 144, e.rw); put32(p, 148, e.rh); put32(p, 152, e.w0); put32(p, 156, e.h0);
    putF(p, 312, prevJitter[0]); putF(p, 316, prevJitter[1]); putF(p, 320, settings.jitter[0]); putF(p, 324, settings.jitter[1]);
    putF(p, 328, rw / ow); putF(p, 332, rh / oh); putF(p, 336, ow / rw); putF(p, 340, oh / rh);
    putF(p, 344, settings.motionScale[0] * (ow / rw)); putF(p, 348, settings.motionScale[1] * (oh / rh));
    // byte 365 (and the output stage's byte 440): the reset flag. Temporal harness captures (capDT, capDfr, capDst,
    // 2026-10-12) show it follows Reset alone, not the jitter: set with reset and jitter, clear without both.
    put32(p, 364, reset ? 256u : 0u);
    // byte 366: DLSS.Roughness.Mode packed. enc0 then takes the roughness from the normals' w and never reads the
    // roughness texture (argument 192, bound to the normals as well)
    p[366] = settings.roughnessPacked ? 1u : 0u;
    // bytes 536 / 539: DLSSD.SpecularHitDistance (texture 248) / DLSS.DisocclusionMask (texture 296) are bound. The hit
    // distance also sets the output stage's byte 160 (harness captures capDXhit, capDXmask, capDX, 2026-10-12)
    p[536] = images.specularHitDistance ? 1u : 0u;
    p[539] = images.disocclusionMask ? 1u : 0u;
    put32(p, 432, e.rw - 1); put32(p, 436, e.rh - 1); put32(p, 472, e.ow); put32(p, 476, e.oh);
    put32(p, 528, settings.depthInverted ? 1u : 0u);
    // the camera: the view's rotation (its translation row dropped) times the projection, its inverse, and both
    // composed with the render viewport (NDC <-> render pixels, y down)
    Mat4 view, clip;
    for (int i = 0; i < 16; ++i) { view[i] = settings.worldToView[i]; clip[i] = settings.viewToClip[i]; }
    view[12] = view[13] = view[14] = 0.0;
    const Mat4 rc = mul(view, clip), inv = inverse(rc);
    const Mat4 toNdc{2.0 / e.rw, 0, 0, 0, 0, -2.0 / e.rh, 0, 0, 0, 0, 1, 0, -1, 1, 0, 1};
    const Mat4 toPixels{e.rw / 2.0, 0, 0, 0, 0, -(e.rh / 2.0), 0, 0, 0, 0, 1, 0, e.rw / 2.0, e.rh / 2.0, 0, 1};
    putMat(p, 596, rc); putMat(p, 660, clip); putMat(p, 724, clip); putMat(p, 788, inv);
    putMat(p, 852, mul(toNdc, inv)); putMat(p, 916, mul(rc, toPixels));
    putF(p, 756, settings.jitter[0] * (2.0f / rw)); putF(p, 760, -settings.jitter[1] * (2.0f / rh));
    put32(p, 984, f);
    put32(p, 1024, e.rw); put32(p, 1028, e.rh);
    // the output stage
    auto& o = e.blocks[RHkpn];
    o = e.model.hkpnTemplate;
    put32(o, 0, e.ow); put32(o, 4, e.oh); put32(o, 8, e.ow); put32(o, 12, e.oh); put32(o, 16, e.rw); put32(o, 20, e.rh);
    putF(o, 24, rw / ow); putF(o, 28, rh / oh); putF(o, 32, ow / rw); putF(o, 36, oh / rh);
    putF(o, 40, settings.jitter[0]); putF(o, 44, settings.jitter[1]);
    put32(o, 56, e.w0); put32(o, 60, e.h0); put32(o, 140, e.rw - 1); put32(o, 144, e.rh - 1);
    putF(o, 152, settings.frameTimeMs);
    o[160] = images.specularHitDistance ? 1u : 0u;
    put64(o, 256, e.arenaOffset(RHkpnArena));
    putF(o, 336, settings.motionScale[0] * (ow / rw)); putF(o, 340, settings.motionScale[1] * (oh / rh));
    put32(o, 440, (reset ? 1u : 0u) | (settings.depthInverted ? 0x10000u : 0u));
    // the exposure: NGX's MODE 1 measurement (engine/exposure_k.comp) on 1 / pre-exposure scaled colour
    const uint32_t gx = exposureGrid(e.rw, e.ow), gy = exposureGrid(e.rh, e.oh);
    e.exposure.origin[0] = e.exposure.origin[1] = 0;
    e.exposure.size[0] = static_cast<int32_t>(e.rw); e.exposure.size[1] = static_cast<int32_t>(e.rh);
    e.exposure.sampleSize[0] = static_cast<int32_t>(gx); e.exposure.sampleSize[1] = static_cast<int32_t>(gy);
    e.exposure.invCount = 1.0f / static_cast<float>(gx * gy);
    e.exposure.scale = 1.0f / (settings.preExposure > 0.0f ? settings.preExposure : 1.0f);
    // cuda_dldn_engine_auto_exposure_copy_kernel divides a constant of its own (0d3FCE23E8E0000000), not its argument 0
    // (0.2195...), by max(mean luma, 0.005): the capture's 0.919 = 0.23547 / 0.2562
    e.exposure.key = 0.23547087609767914f;
    // NGX's reset keeps the exposure state, the recurrent feature and the output history and only clears the previous
    // frame's enc0 feature q (a harness run resetting every frame: enc0's textures 208, 488 and 496 carry the previous
    // frame from frame 2 on, 224 stays zero; capture capE3, 2026-10-11). The first frame starts from zero entirely.
    e.exposure.flags = 1u | (e.frame == 0 ? 2u : 0u);
    e.reset = reset;
    // descriptors for the caller's images, cached per combination: every stage, both frame parities, bound by the
    // argument offsets its shader declares (rr.txt)
    // texture and surface handles of the two stages with optional inputs: 1 where something is bound, 0 where NGX
    // leaves the argument null. The kernels branch on that (enc0 takes its specular hit distance path only when
    // argument 248's handle is not null; capDX, 2026-10-12).
    for (unsigned stage : {static_cast<unsigned>(REnc0), static_cast<unsigned>(RHkpn)}) {
        for (uint32_t offset : e.model.textures[stage])
            put64(e.blocks[stage], offset, e.resolve(stage, false, offset, 0, images, e.out.view).view ? 1u : 0u);
        for (const auto& [offset, format] : e.model.surfaces[stage])
            put64(e.blocks[stage], offset, e.resolve(stage, true, offset, 0, images, e.out.view).view ? 1u : 0u);
    }
    auto same = [&](const Impl::Bindings& b) {
        return !std::memcmp(&b.images, &images, sizeof(images)) && b.finalOutput.view == finalOutput.view;
    };
    auto found = std::find_if(e.bindings.begin(), e.bindings.end(), same);
    if (found == e.bindings.end()) {
        if (e.bindings.size() >= 16) throw std::runtime_error("too many distinct Ray Reconstruction input view sets");
        Impl::Bindings b;
        b.images = images; b.finalOutput = finalOutput;
        uint32_t textures = 0, surfaces = 0;
        for (unsigned stage = 0; stage < RStages; ++stage) {
            textures += static_cast<uint32_t>(e.model.textures[stage].size());
            surfaces += static_cast<uint32_t>(e.model.surfaces[stage].size());
        }
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 3 * RStages + 3},
                                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * textures + 1},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * surfaces}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 2 * RStages + 1; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
        VKCHECK(vkCreateDescriptorPool(d.handle, &pi, nullptr, &b.pool));
        const VkImageView output = finalOutput.view ? finalOutput.view : e.out.view;
        for (unsigned stage = 0; stage < RStages; ++stage) {
            for (uint32_t par = 0; par < 2; ++par) {
                VkDescriptorSet set = b.stages[stage][par] = descriptorSet(d.handle, b.pool, e.model.stages[stage].descriptors);
                bindBuffer(d.handle, set, 0, e.params[stage]);
                bindBuffer(d.handle, set, 1, e.model.weights[stage]);
                bindBuffer(d.handle, set, 2, e.arena);
                uint32_t binding = 3;
                for (uint32_t offset : e.model.textures[stage]) {
                    const auto res = e.resolve(stage, false, offset, par, images, output);
                    bindImage(d.handle, set, binding++, res.view ? res.view : e.zero.view, res.linear ? e.linear : e.nearest,
                              res.game ? images.layout : VK_IMAGE_LAYOUT_GENERAL);
                }
                for (const auto& [offset, format] : e.model.surfaces[stage]) {
                    const auto res = e.resolve(stage, true, offset, par, images, output);
                    bindImage(d.handle, set, binding++, res.view ? res.view : e.spare.at(format).view);
                }
            }
        }
        b.exposure = descriptorSet(d.handle, b.pool, e.model.exposure[0].descriptors);
        bindBuffer(d.handle, b.exposure, 0, e.expP);
        bindImage(d.handle, b.exposure, 1, images.color, e.nearest, images.layout);
        bindBuffer(d.handle, b.exposure, 2, e.expSums);
        e.bindings.push_back(b);
        found = e.bindings.end() - 1;
    }
    e.bound = &*found;
    // debugging: D4R_ENGINE_RR_DUMP=DIR writes every frame's parameter blocks as DIR/F-STAGE.bin (F counts from 1)
    if (const char* dump = std::getenv("D4R_ENGINE_RR_DUMP")) {
        for (unsigned stage = 0; stage < RStages; ++stage)
            std::ofstream(std::string(dump) + "/" + std::to_string(f + 1) + "-" + RNames[stage] + ".bin", std::ios::binary)
                .write(reinterpret_cast<const char*>(e.blocks[stage].data()), static_cast<std::streamsize>(e.blocks[stage].size()));
        std::ofstream(std::string(dump) + "/" + std::to_string(f + 1) + "-exposure.bin", std::ios::binary)
            .write(reinterpret_cast<const char*>(&e.exposure), sizeof(e.exposure));
    }
    e.prevJitter[0] = settings.jitter[0]; e.prevJitter[1] = settings.jitter[1];
}

void REngine::recordFrame(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) {
    auto& e = *impl;
    if (!e.bound) throw std::runtime_error("REngine::setFrame was not called");
    const auto& b = *e.bound;
    const uint32_t f = e.frame, cur = f & 1u;
    auto stamp = [&](uint32_t n) { if (stamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, stamps, first + n); };
    auto step = [&] {
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    };
    stamp(0);
    // the frame's parameter blocks, then (on a reset) a clean temporal state
    memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    for (unsigned i = 0; i < RStages; ++i) vkCmdUpdateBuffer(cb, e.params[i].handle, 0, e.blocks[i].size(), e.blocks[i].data());
    vkCmdUpdateBuffer(cb, e.expP.handle, 0, sizeof(ExposureParams), &e.exposure);
    if (e.reset) {
        VkClearColorValue zero{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (auto& t : e.q) vkCmdClearColorImage(cb, t.handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    // exposure
    dispatch(cb, e.model.exposure[0], b.exposure, e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX);
    step();
    dispatch(cb, e.model.exposure[1], e.exp1Set, 1, 1);
    step();
    stamp(1);
    // the network, in NGX's order; each stage's grid covers its level plus that frame's window shift
    for (unsigned stage = REnc0; stage <= RDec0; ++stage) {
        const int l = RLevel[stage];
        const uint32_t shift = rShift(stage, f), sx = shift & 0xffffu, sy = shift >> 16;
        const VkDescriptorSet set = b.stages[stage][cur];
        dispatch(cb, e.model.stages[stage], set, (e.lw(l) + sx + 7) / 8, (e.lh(l) + sy + 7) / 8);
        step();
    }
    stamp(2);
    dispatch(cb, e.model.stages[RHkpn], b.stages[RHkpn][cur], (e.ow + 15) / 16, (e.oh + 15) / 16);
    memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    stamp(3);
    ++e.frame;
}
