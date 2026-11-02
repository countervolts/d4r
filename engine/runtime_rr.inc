// Ray Reconstruction (DLSS-D presets D and E): the runtime around engine/rr's stage shaders. Included by runtime.cpp
// inside namespace d4r, after runtime_m.inc (it uses that file's Tex, bindRange and exposureGrid).
//
// Every stage reads and writes NGX's own layouts (engine/rr/README.md): its parameter block is NVIDIA's argument
// block with arena byte offsets in place of pointers, the network's activations live in one plane arena at NGX's
// offsets, and the per-level surfaces have NGX's sizes. So a model is NGX's weight images plus two argument-block
// templates (enc0's and the output stage's constants), and this file fills every field that depends on the frame or
// on the geometry.
namespace {
#include "rr_frame_parameters.h"
// AMD's own driver (Windows, LLPC) reports VK_DRIVER_ID_AMD_PROPRIETARY: it crashes compiling preset F's encoder 0
// and decoder 0 kernel 1 shaders, so there the runtime loads the model's <shader>_llpc.spv variants when it has them
// (engine/runtime_rr2.inc, compile_rr2.py). Everywhere else, RADV included, the model's own shaders are loaded.
bool amdProprietaryDriver(VkPhysicalDevice physical) { return driverId(physical) == VK_DRIVER_ID_AMD_PROPRIETARY; }
} // namespace
#include "runtime_rr2.inc"
#include "runtime_rr_cnn.inc"

struct RModel::Impl {
    Device device;
    uint32_t preset = 0;
    bool gfx11 = false;
    std::unique_ptr<R2Model> v2;   // preset F ("D4RR0002"): engine/runtime_rr2.inc
    std::unique_ptr<R3Model> v3;   // presets A - C ("D4RR0003"): engine/runtime_rr_cnn.inc
    // per stage, the shader's bindings after the three buffers (engine/rr/README.md): texture argument offsets, then
    // surface argument offsets with their GLSL image format, both ascending
    std::array<std::vector<uint32_t>, RStages> textures;
    std::array<std::vector<std::pair<uint32_t, std::string>>, RStages> surfaces;
    std::array<Buffer, RStages> weights;
    std::array<std::vector<char>, RStages> weightBytes;   // until recordUpload
    std::array<Pipeline, RStages> stages;
    // optional (vkhip, kernels/rr/build_vkhip.sh): enc0's translated feature gather alone, which writes each window's
    // features and noise grid to scratch after the arena for a stage recorder that runs the rest of encoder 0
    Pipeline enc0Gather;
    std::array<Pipeline, 2> exposure;
    std::vector<uint8_t> enc0Template, hkpnTemplate;
    bool uploaded = false;
    Impl(VkPhysicalDevice physical, VkDevice dev) : device(physical, dev) {}
};
RModel::RModel(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache, bool int64Atomics,
               bool robustImages)
    : impl(std::make_unique<Impl>(physical, device)) {
    auto& m = *impl;
    // the translated stages' specialization constant 0 (engine/rr/ptx2rr.py RR_ROBUST_IMAGES), set for every stage of the
    // model; shaders without it ignore it. Constant 0 is reserved for this: an RR shader must not declare it otherwise.
    const VkBool32 robust = VK_TRUE;
    const VkSpecializationMapEntry robustEntry{0, 0, sizeof(VkBool32)};
    const VkSpecializationInfo robustInfo{1, &robustEntry, sizeof(VkBool32), &robust};
    const VkSpecializationInfo* specialization = robustImages ? &robustInfo : nullptr;
    // rr.txt: "D4RR0001", "preset N", one "stage NAME SHADER.spv SUBGROUP WEIGHTS TEXTURES SURFACES" line per stage in
    // RNames order (TEXTURES: comma-separated argument offsets or "-"; SURFACES: OFFSET:FORMAT,... or "-") and
    // "template enc0|hkpn FILE" (the argument blocks whose constants the runtime keeps)
    auto text = read(directory + "/rr.txt");
    std::istringstream lines(std::string(text.begin(), text.end()));
    std::string line;
    if (!std::getline(lines, line)) throw std::runtime_error("not a Ray Reconstruction model (rr.txt)");
    if (line.rfind("D4RR0002", 0) == 0) {
        m.preset = 6;
        m.v2 = std::make_unique<R2Model>();
        m.v2->load(m.device, directory, lines, cache, int64Atomics, specialization, amdProprietaryDriver(physical));
        return;
    }
    if (line.rfind("D4RR0003", 0) == 0) {
        std::string key;
        if (!std::getline(lines, line) || !(std::istringstream(line) >> key >> m.preset) || key != "preset" || m.preset < 1 || m.preset > 3)
            throw std::runtime_error("rr.txt: a D4RR0003 model's second line is its preset, 1 - 3");
        m.v3 = std::make_unique<R3Model>();
        m.v3->load(m.device, directory, lines, cache, specialization);
        return;
    }
    if (line.rfind("D4RR0001", 0) != 0) throw std::runtime_error("not a Ray Reconstruction model (rr.txt D4RR0001)");
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
        else if (key == "arithmetic") {
            std::string arithmetic; f >> arithmetic;
            if (arithmetic != "gfx11" && arithmetic != "gfx12") throw std::runtime_error("rr.txt: unknown arithmetic " + arithmetic);
            m.gfx11 = arithmetic == "gfx11";
        }
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
            traceStage(directory + "/" + shader);
            m.stages[stage].create(device, directory + "/" + shader, types, subgroup, cache, flags, specialization);
            if (stage == REnc0 && std::ifstream(directory + "/enc0_g.spv").good()) {
                traceStage(directory + "/enc0_g.spv");
                m.enc0Gather.create(device, directory + "/enc0_g.spv", types, subgroup, cache, flags, specialization);
            }
            auto blob = read(directory + "/" + weightFile);
            const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            m.weights[stage].create(m.device, blob.size(), usage);
            m.weightBytes[stage] = std::move(blob);
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
    if (m.v2 || m.v3) {
        if (m.v2) m.v2->recordUpload(cb); else m.v3->recordUpload(cb);
        memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        m.uploaded = true;
        return;
    }
    for (unsigned i = 0; i < RStages; ++i) {
        recordContents(cb, m.weights[i], m.weightBytes[i]);
    }
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    m.uploaded = true;
}

struct REngine::Impl {
    RModel::Impl& model;
    std::unique_ptr<R2Engine> v2;   // preset F
    std::unique_ptr<R3Engine> v3;   // presets A - C
    uint32_t rw, rh, ow, oh, w0, h0;
    Buffer arena, expP, expSums, expState;
    std::array<Buffer, RStages> params;
    std::array<std::vector<uint8_t>, RStages> blocks;   // this frame's parameter blocks
    std::array<Tex, 6> s112, s120;  // per level: the encoders' surfaces (decoders' skip textures 96, 104)
    std::array<Tex, 5> s128;        // the encoders' half-resolution surface, the next encoder's texture 88
    std::array<Tex, 5> d112, d120;  // the decoders' surfaces (texture 88 of the next decoder; dec0's 112: output stage)
    std::array<Tex, 2> q, r, h, e;  // ping-pong: enc0's 304 / 224, dec0's 144 / enc0's 488, the output-size history,
                                    // and e: this frame's and the previous frame's exposure (1x1)
    Tex out;                        // the presented image (the output stage's surface 240) when the caller passes none
    bool outNew = false;            // created by setFrame (first frame without the caller's output)
    // the output stage's second history (surface 384, read back as texture 392), kept only for games that pass
    // DLSSD.ColorBeforeParticles; created on first use
    Tex pb[2];
    bool pbNew = false;
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
                case 264: return in.subsurfaceGuide ? G(in.subsurfaceGuide) : Resource{};   // capDXsss
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
                // DLSSD.ColorBeforeParticles and its history (capDXparticles, 2026-10-12)
                case 416: return in.colorBeforeParticles ? G(in.colorBeforeParticles) : Resource{};
                case 392: return in.colorBeforeParticles ? T(pb[par ^ 1], true) : Resource{};
            } else switch (offset) {
                case 224: return T(h[par]); case 240: return Resource{output, false, false};
                case 384: return in.colorBeforeParticles ? T(pb[par]) : Resource{};
            }
        }
        return {};
    }
    std::vector<Bindings> bindings;
    const Bindings* bound = nullptr;
    float prevJitter[2] = {0, 0};
    rr_camera::Matrix previousView{}, previousClip{};
    uint32_t frame = 0;
    bool reset = true;
    RStageRecorder* recorder = nullptr;   // network stages recorded by the caller (vkhip)
    uint32_t gatherWindows = 0;           // the enc0 windows the scratch after the arena holds (0: no scratch)
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

REngine::REngine(RModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight,
                 bool arenaAddress)
    : impl(std::make_unique<Impl>(*model.impl)) {
    auto& e = *impl;
    const Device& d = e.model.device;
    if (e.model.v2) {
        if (renderWidth < 64 || renderHeight < 64 || renderWidth > outputWidth || renderHeight > outputHeight || outputWidth > 16384 || outputHeight > 16384)
            throw std::runtime_error("unsupported Ray Reconstruction dimensions");
        e.v2 = std::make_unique<R2Engine>(d, *e.model.v2, renderWidth, renderHeight, outputWidth, outputHeight, arenaAddress);
        return;
    }
    if (e.model.v3) {
        if (renderWidth < 64 || renderHeight < 64 || renderWidth > outputWidth || renderHeight > outputHeight || outputWidth > 16384 || outputHeight > 16384)
            throw std::runtime_error("unsupported Ray Reconstruction dimensions");
        e.v3 = std::make_unique<R3Engine>(d, *e.model.v3, renderWidth, renderHeight, outputWidth, outputHeight);
        return;
    }
    if (renderWidth < 64 || renderHeight < 64 || renderWidth > outputWidth || renderHeight > outputHeight || outputWidth > 16384 || outputHeight > 16384)
        throw std::runtime_error("unsupported Ray Reconstruction dimensions");
    e.rw = renderWidth; e.rh = renderHeight; e.ow = outputWidth; e.oh = outputHeight;
    e.w0 = (renderWidth + 31) & ~31u; e.h0 = (renderHeight + 31) & ~31u;
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    // with an address (a stage recorder) and enc0's gather: the scratch for the most windows either shift dispatches
    if (arenaAddress && e.model.enc0Gather.handle) e.gatherWindows = ((e.lw(0) + 14) / 8) * ((e.lh(0) + 14) / 8);
    e.arena.create(d, e.arenaOffset(RArenaUnits) + uint64_t(e.gatherWindows) * RGatherBytes, storage, false, arenaAddress);
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
VkImage REngine::outputImage() const { return impl->v2 ? impl->v2->out.handle : impl->v3 ? impl->v3->out.handle : impl->out.handle; }
std::vector<REngine::DebugImage> REngine::debugImages() const {
    const auto& e = *impl;
    if (e.v2) return e.v2->debugImages();
    if (e.v3) return e.v3->debugImages();
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
    if (e.out.handle) list.push_back({"out", e.out.handle, half4, e.ow, e.oh});
    return list;
}

void REngine::setStageRecorder(RStageRecorder* recorder) {
    if (impl->v2) { impl->v2->recorder = recorder; return; }
    if (impl->v3) { impl->v3->recorder = recorder; return; }
    impl->recorder = recorder;
}
VkBuffer REngine::arenaBuffer() const {
    if (impl->v2) return impl->v2->arena.handle;
    if (impl->v3) return impl->v3->arena.handle;
    return impl->arena.handle;
}
uint64_t REngine::stageParameterAddress(unsigned stage) const {
    if (impl->v3 && stage < R3Network) return impl->v3->address(impl->v3->paramAll) + impl->v3->paramAt[stage];
    if (!impl->v2 || stage >= R2Network) throw std::runtime_error("preset F stage required");
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = impl->v2->paramAll.handle;
    const auto base = vkGetBufferDeviceAddress(impl->model.device.handle, &info);
    return base ? base + impl->v2->paramAt[stage] : 0;
}
RStageResources REngine::stageResources(unsigned stage) const {
    const auto& e = *impl;
    if (e.v3 && stage < R3Network) {   // presets A - C: the size of the stage's level
        RStageResources r;
        r.width = e.v3->lw(R3Level[stage]); r.height = e.v3->lh(R3Level[stage]);
        return r;
    }
    if (e.v2 || e.v3 || stage > RDec0) throw std::runtime_error("REngine::stageResources: the network stages of presets D and E only");
    RStageResources r;
    const int l = RLevel[stage];
    r.width = e.lw(l); r.height = e.lh(l);
    for (int k = 0; k < 4; ++k) r.arena[k] = RArena[stage][k] >= 0 ? e.arenaOffset(RArena[stage][k]) : 0;
    // the stage's images do not depend on the frame's inputs, nor (but decoder 0's surface 144) its parity (resolve)
    const RFrameImages none{};
    auto view = [&](bool surface, uint32_t offset, uint32_t par = 0) { return e.resolve(stage, surface, offset, par, none, VK_NULL_HANDLE).view; };
    r.texture88 = view(false, 88);
    if (stage >= RDec4) { r.texture96 = view(false, 96); r.texture104 = view(false, 104); }
    r.surface112 = view(true, 112);
    if (stage <= REnc5) { r.surface120 = view(true, 120); r.surface128 = view(true, 128); }
    if (stage == RDec0) { r.surface144[0] = view(true, 144, 0); r.surface144[1] = view(true, 144, 1); }
    if (stage == REnc0) r.scratchEnd = e.arenaOffset(RArenaUnits) + uint64_t(e.gatherWindows) * RGatherBytes;
    return r;
}

void REngine::recordInitialize(VkCommandBuffer cb) {
    auto& e = *impl;
    if (e.v2) { e.v2->recordInitialize(cb); return; }
    if (e.v3) { e.v3->recordInitialize(cb); return; }
    std::vector<VkImage> images;
    for (auto* group : {&e.s112, &e.s120}) for (auto& t : *group) images.push_back(t.handle);
    for (auto* group : {&e.s128, &e.d112, &e.d120}) for (auto& t : *group) images.push_back(t.handle);
    for (auto* group : {&e.q, &e.r, &e.h, &e.e}) for (auto& t : *group) images.push_back(t.handle);
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
    if (e.v2) { e.v2->setFrame(images, settings, finalOutput); return; }
    if (e.v3) { e.v3->setFrame(images, settings, finalOutput); return; }
    const Device& d = e.model.device;
    if (!images.color || !images.motion || !images.depth || !images.normals || !images.roughness || !images.diffuseAlbedo || !images.specularAlbedo)
        throw std::runtime_error("Ray Reconstruction needs colour, motion, depth and the four guides");
    if (finalOutput.view && finalOutput.format != VK_FORMAT_R16G16B16A16_SFLOAT)
        throw std::runtime_error("Ray Reconstruction writes RGBA16F output only");
    setRRParameters(e, images, settings);
    const uint32_t f = e.frame;
    // descriptors for the caller's images, cached per combination: every stage, both frame parities, bound by the
    // argument offsets its shader declares (rr.txt)
    if (images.colorBeforeParticles && !e.pb[0].handle) {
        for (auto& t : e.pb) t.create(d, e.ow, e.oh, VK_FORMAT_R16G16B16A16_SFLOAT);
        e.pbNew = true;
    }
    if (!finalOutput.view && !e.out.handle) {
        e.out.create(d, e.ow, e.oh, VK_FORMAT_R16G16B16A16_SFLOAT);
        e.outNew = true;
    }
    const VkImageView frameOutput = finalOutput.view ? finalOutput.view : e.out.view;
    // texture and surface handles of the two stages with optional inputs: 1 where something is bound, 0 where NGX
    // leaves the argument null. The kernels branch on that (enc0 takes its specular hit distance path only when
    // argument 248's handle is not null; capDX, 2026-10-12).
    for (unsigned stage : {static_cast<unsigned>(REnc0), static_cast<unsigned>(RHkpn)}) {
        for (uint32_t offset : e.model.textures[stage])
            put64(e.blocks[stage], offset, e.resolve(stage, false, offset, 0, images, frameOutput).view ? 1u : 0u);
        for (const auto& [offset, format] : e.model.surfaces[stage])
            put64(e.blocks[stage], offset, e.resolve(stage, true, offset, 0, images, frameOutput).view ? 1u : 0u);
    }
    auto same = [&](const Impl::Bindings& b) {
        return !std::memcmp(&b.images, &images, sizeof(images)) && b.finalOutput.view == finalOutput.view;
    };
    auto found = std::find_if(e.bindings.begin(), e.bindings.end(), same);
    if (found == e.bindings.end()) {
        if (e.bindings.size() >= kRRMaxBindings) throw std::runtime_error("too many distinct Ray Reconstruction input view sets");
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
        const VkImageView output = frameOutput;
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
    if (e.v2) { e.v2->recordFrame(cb, stamps, first); return; }
    if (e.v3) { e.v3->recordFrame(cb, stamps, first); return; }
    if (!e.bound) throw std::runtime_error("REngine::setFrame was not called");
    const auto& b = *e.bound;
    const uint32_t f = e.frame, cur = f & 1u;
    auto stamp = [&](uint32_t n) { if (stamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, stamps, first + n); };
    // D4R_ENGINE_RR_STAGE_STAMPS=1: one more stamp after every network stage (queries first + 4 ...)
    static const bool stageStamps = std::getenv("D4R_ENGINE_RR_STAGE_STAMPS") != nullptr;
    auto step = [&] {
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    };
    if (e.pbNew || e.outNew) {   // images created by setFrame start from zero, like the rest of the temporal state
        if (e.pbNew) for (auto& t : e.pb) clearNewImage(cb, t.handle);
        if (e.outNew) clearNewImage(cb, e.out.handle);
        memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        e.pbNew = e.outNew = false;
    }
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
        const uint32_t gx = (e.lw(l) + sx + 7) / 8, gy = (e.lh(l) + sy + 7) / 8;
        if (stage == REnc0 && e.gatherWindows && e.recorder && e.recorder->records(REnc0)) {
            dispatch(cb, e.model.enc0Gather, set, gx, gy);   // the features to scratch, then the recorder's block
            step();
            e.recorder->recordStage(cb, stage, gx, gy, shift);
        } else if (!e.recorder || !e.recorder->recordStage(cb, stage, gx, gy, shift))
            dispatch(cb, e.model.stages[stage], set, gx, gy);
        step();
        if (stageStamps) stamp(4 + stage - REnc0);
    }
    stamp(2);
    dispatch(cb, e.model.stages[RHkpn], b.stages[RHkpn][cur], (e.ow + 15) / 16, (e.oh + 15) / 16);
    memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    stamp(3);
    ++e.frame;
}
