// Preset M runtime; included by runtime.cpp inside namespace d4r (it shares that file's Vulkan helpers).
// GCC treats an included file as a header and warns that these types use the including file's anonymous namespace.
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
namespace {
constexpr unsigned MLayers = 10;
constexpr const char* MShader[MLayers] = {"m_enc1", "m_enc2", "m_tube", "m_tube", "m_tube", "m_tube", "m_tube", "m_tube", "m_dec2", "m_dec1"};
constexpr int32_t MShift[MLayers] = {2, 2, 2, 0, 2, 0, 2, 0, 0, 0};
constexpr unsigned MScale[MLayers] = {1, 2, 4, 4, 4, 4, 4, 4, 2, 1};       // token grid divisor
constexpr unsigned MChannels[MLayers] = {64, 96, 128, 128, 128, 128, 128, 128, 96, 64};
struct MEnc0Params {
    int32_t size[2], shift[2], tok[2];
    float jitter[2], prevJitter[2], mvScale[2], mvOffs[2], rnd2hi[2], hi2rnd[2], hiSize[2];
    float changeScale, changeGain;
    uint32_t depthFlag, reset, frame;
    float expo, expoPrev;
    float halfInv[2], renderInv[2]; // CUDA texture coordinates use host-rounded reciprocals
};
struct MDec0Params { int32_t size[2], tokW; uint32_t reset; float rndOffs[2], rnd2hi[2], hiSize[2], histOffs[2], sigma, rho, renderInv[2]; };
struct MPostParams { int32_t colOrigin[2], hiSize[2], rndSize[2]; float hi2rnd[2], rndOffs[2], invRnd[2], gain; uint32_t reset; float expo, expoPrev; };
struct MDownParams { int32_t srcSize[2], outOrigin[2], colOrigin[2], colSize[2]; float out2rnd[2], rndOffs[2], expo; };
struct MLayerParams { int32_t w, h, sx, sy; uint32_t o[72]; };
static_assert(sizeof(MEnc0Params) == 124 && offsetof(MEnc0Params, expo) == 100 && offsetof(MEnc0Params, halfInv) == 108, "enc0_m.comp and exposure_k.comp (words 25, 26)");
static_assert(sizeof(MDec0Params) == 64, "dec0_m.comp");
static_assert(sizeof(MPostParams) == 64 && offsetof(MPostParams, expo) == 56, "post_m.comp and exposure_k.comp (words 14, 15)");
static_assert(sizeof(MDownParams) == 52 && offsetof(MDownParams, expo) == 48, "down_m.comp and exposure_k.comp (word 12)");
static_assert(sizeof(MLayerParams) == 304, "layer_m.comp");
struct Tex {
    const Device* device = nullptr;
    VkImage handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    ~Tex() {
        if (!device) return;
        if (view) vkDestroyImageView(device->handle, view, nullptr);
        if (handle) vkDestroyImage(device->handle, handle, nullptr);
        if (memory) vkFreeMemory(device->handle, memory, nullptr);
    }
    void create(const Device& d, uint32_t w, uint32_t h, VkFormat format) {
        device = &d;
        if (!w || !h || w > d.properties.limits.maxImageDimension2D || h > d.properties.limits.maxImageDimension2D)
            throw std::runtime_error("image dimensions exceed device limits");
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D; info.format = format; info.extent = {w, h, 1}; info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VKCHECK(vkCreateImage(d.handle, &info, nullptr, &handle));
        VkMemoryRequirements requirements; vkGetImageMemoryRequirements(d.handle, handle, &requirements);
        memory = d.allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VKCHECK(vkBindImageMemory(d.handle, handle, memory, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = handle; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VKCHECK(vkCreateImageView(d.handle, &vi, nullptr, &view));
    }
};
void bindRange(VkDevice device, VkDescriptorSet set, uint32_t binding, const Buffer& buffer, VkDeviceSize offset) {
    VkDescriptorBufferInfo info{buffer.handle, offset, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set; write.dstBinding = binding; write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}
} // namespace

struct MModel::Impl {
    Device device;
    Buffer w8, w16, up8, up16;
    uint32_t enc0[3]{}, dec0[2]{};
    std::array<std::array<uint32_t, 74>, MLayers> layers{};
    std::array<Pipeline, MLayers> network;      // one per layer (five distinct shaders)
    Pipeline input, expand, post, postQuality, postCompact, down, downRgb10;
    Pipeline inputPlanes, expandPlanes;         // token buffers in NGX's plane layout, for an external (HIP) network
    bool planes = false;
    std::array<Pipeline, 2> exposure;
    bool uploaded = false, ldr = false;
    Impl(VkPhysicalDevice physical, VkDevice dev) : device(physical, dev) {}
};
MModel::MModel(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache)
    : impl(std::make_unique<Impl>(physical, device)) {
    auto& m = *impl;
    m.ldr = exists(directory + "/ldr");     // the stages were compiled for a game without the HDR flag: no exposure
    auto head = read(directory + "/offsets.bin");
    const size_t expect = 16 + 5 * 4 + MLayers * 74 * 4;
    if (head.size() != expect || std::memcmp(head.data(), "D4RM0003", 8)) throw std::runtime_error("incompatible M model metadata");
    uint32_t n8 = 0, n16 = 0;
    std::memcpy(&n8, head.data() + 8, 4); std::memcpy(&n16, head.data() + 12, 4);
    auto b8 = read(directory + "/weights8.bin"), b16 = read(directory + "/weights16.bin");
    if (b8.size() != n8 || b16.size() != n16 || n16 % 64) throw std::runtime_error("invalid M model sizes");
    std::memcpy(m.enc0, head.data() + 16, 12); std::memcpy(m.dec0, head.data() + 28, 8);
    std::memcpy(m.layers.data(), head.data() + 36, sizeof(m.layers));
    auto inside = [&](uint32_t off8, uint32_t off16) { if (off8 >= n8 || off16 >= n16 || off8 % 64 || off16 % 64) throw std::runtime_error("invalid M model offsets"); };
    inside(m.enc0[0], m.enc0[1]); inside(m.dec0[0], m.dec0[1]); inside(0, m.enc0[2]);
    for (auto& l : m.layers) inside(l[0], l[1]);
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    m.w8.create(m.device, b8.size(), usage); m.w16.create(m.device, b16.size(), usage);
    m.up8.create(m.device, b8.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true); m.up16.create(m.device, b16.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    std::memcpy(m.up8.mapped, b8.data(), b8.size()); std::memcpy(m.up16.mapped, b16.data(), b16.size());
    const auto b = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, t = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    for (unsigned i = 0; i < MLayers; ++i) m.network[i].create(device, directory + "/" + MShader[i] + ".spv", {b,b,b,b,b,b,b}, 32, cache);
    m.input.create(device, directory + "/enc0_m.spv", {b,t,t,t,t,t,t,s,s,s,b,b,b,b}, 32, cache);
    m.expand.create(device, directory + "/dec0_m.spv", {b,t,t,s,s,s,s,b,b,b}, 32, cache);
    if ((m.planes = exists(directory + "/enc0_m_planes.spv") && exists(directory + "/dec0_m_planes.spv"))) {
        m.inputPlanes.create(device, directory + "/enc0_m_planes.spv", {b,t,t,t,t,t,t,s,s,s,b,b,b,b}, 32, cache);
        m.expandPlanes.create(device, directory + "/dec0_m_planes.spv", {b,t,t,s,s,s,s,b,b,b}, 32, cache);
    }
    m.post.create(device, directory + "/post_m.spv", {b,t,t,t,t,t,t,t,t,s,s,b}, 64, cache);
    m.postQuality.create(device, directory + "/post_m_q.spv", {b,t,t,t,t,t,t,t,t,s,s,b}, 64, cache);
    if (exists(directory + "/post_m_c.spv"))
        m.postCompact.create(device, directory + "/post_m_c.spv", {b,t,t,t,t,t,t,t,t,s,s,b}, 64, cache);
    m.down.create(device, directory + "/down_m.spv", {b,t,s,b}, 64, cache);
    if (exists(directory + "/down_m_rgb10.spv"))
        m.downRgb10.create(device, directory + "/down_m_rgb10.spv", {b,t,s,b}, 64, cache);
    m.exposure[0].create(device, directory + "/exposure_m0.spv", {b,t,b}, 64, cache);
    m.exposure[1].create(device, directory + "/exposure_m1.spv", {b,b,b,b,b,b}, 64, cache);
}
MModel::~MModel() = default;
bool MModel::uploadRecorded() const { return impl->uploaded; }
bool MModel::hasExternalNetworkStages() const { return impl->planes; }
bool MModel::hasRgb10Output() const { return impl->downRgb10.handle != VK_NULL_HANDLE; }
void MModel::recordUpload(VkCommandBuffer cb) {
    auto& m = *impl;
    if (m.uploaded) throw std::runtime_error("model upload already recorded");
    VkBufferCopy copy{0, 0, m.up8.size}; vkCmdCopyBuffer(cb, m.up8.handle, m.w8.handle, 1, &copy);
    copy.size = m.up16.size; vkCmdCopyBuffer(cb, m.up16.handle, m.w16.handle, 1, &copy);
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    m.uploaded = true;
}

struct MEngine::Impl {
    MModel::Impl& model;
    uint32_t rw, rh, ow, oh, hw, hh, gw, gh;       // render, output, reconstruction, token pixel grid
    Buffer enc0P, dec0P, postP, downP, expP, expSums, expState, dummy, packed;
    std::array<Buffer, MLayers> layerP;
    Buffer tokens, skip1, merged1, skip2, merged2, tubeA, tubeB, dec2Out, dec1Out;
    std::array<Tex, 2> mvA, feat, hist, half;
    Tex mvB, mvC, gates, mix, cov, output;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MLayers> layerSets{};
    std::array<VkDescriptorSet, 2> enc0Sets{}, dec0Sets{}, postSets{}, expSets{};
    VkDescriptorSet downSet = VK_NULL_HANDLE;
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    MEnc0Params enc0{}; MDec0Params dec0{}; MPostParams post{}; MDownParams down{}; ExposureParams exposure{};
    struct Bindings {
        FrameImages images{};
        FrameOutput finalOutput{};
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> input{}, post{};
        VkDescriptorSet exposure = VK_NULL_HANDLE, down = VK_NULL_HANDLE;
    };
    std::vector<Bindings> bindings;
    float prevJitter[2] = {0, 0};
    unsigned current = 0, frame = 0;
    bool initialized = false, frameSet = false, external = false, frontDone = false, quality = false, compact = false, rgb10Output = false;
    Impl(MModel::Impl& m) : model(m) {}
    ~Impl() {
        auto dev = model.device.handle;
        for (auto& binding : bindings) if (binding.pool) vkDestroyDescriptorPool(dev, binding.pool, nullptr);
        if (pool) vkDestroyDescriptorPool(dev, pool, nullptr);
        if (nearest) vkDestroySampler(dev, nearest, nullptr);
        if (linear) vkDestroySampler(dev, linear, nullptr);
    }
};
VkDeviceSize MEngine::networkBufferBytes(uint32_t renderWidth, uint32_t renderHeight) {
    return VkDeviceSize((renderWidth + 31) / 32 * 16) * ((renderHeight + 31) / 32 * 16) * 64 + 64;
}
MEngine::MEngine(MModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight, const ExternalNetwork* external)
    : impl(std::make_unique<Impl>(*model.impl)) {
    auto& e = *impl; const auto& d = e.model.device; auto dev = d.handle;
    if (external && (!external->tokens || !external->head)) throw std::runtime_error("external network without its buffers");
    if (external && !e.model.planes) throw std::runtime_error("the M model has no stages for an external network");
    e.external = external != nullptr;
    if (renderWidth < 64 || renderHeight < 64 || outputWidth % 2 || outputHeight % 2 || outputWidth > 16384 || outputHeight > 16384)
        throw std::runtime_error("unsupported preset M dimensions");
    e.rw = renderWidth; e.rh = renderHeight; e.ow = outputWidth; e.oh = outputHeight;
    e.hw = outputWidth / 2 * 3; e.hh = outputHeight / 2 * 3;
    if (renderWidth > outputWidth || renderHeight > outputHeight)
        throw std::runtime_error("preset M render dimensions must not exceed the output");
    e.quality = uint64_t(e.hw) < 3ull * renderWidth || uint64_t(e.hh) < 3ull * renderHeight;
    // An 11x11 render tile covers a 16x16 reconstruction block when h <= 0.45; keep margin below 7/15.5.
    e.compact = e.quality && e.model.postCompact.handle &&
        20ull * renderWidth <= 9ull * e.hw && 20ull * renderHeight <= 9ull * e.hh;
    e.gw = (renderWidth + 31) / 32 * 32; e.gh = (renderHeight + 31) / 32 * 32;
    const uint32_t tw = e.gw / 2, th = e.gh / 2;
    const VkBufferUsageFlags sb = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sbt = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    e.enc0P.create(d, sizeof(MEnc0Params), sbt); e.dec0P.create(d, sizeof(MDec0Params), sbt); e.postP.create(d, sizeof(MPostParams), sbt);
    e.downP.create(d, sizeof(MDownParams), sbt); e.expP.create(d, sizeof(ExposureParams), sbt);
    e.expSums.create(d, VkDeviceSize((outputWidth / 2 + 31) / 32) * ((outputHeight / 2 + 31) / 32) * 4, sb);
    e.expState.create(d, 8, sbt); e.dummy.create(d, 256, sb);
    e.packed.create(d, VkDeviceSize(e.hw) * e.hh * 4, sb);
    auto tok = [&](Buffer& b, unsigned scale, unsigned channels) { b.create(d, VkDeviceSize(tw / scale) * (th / scale) * channels + 64, sb); };
    if (external) {
        // the caller's buffers (shared with its network): not owned, the layers' buffers are not needed
        e.tokens.handle = external->tokens; e.dec1Out.handle = external->head; e.tokens.size = e.dec1Out.size = networkBufferBytes(renderWidth, renderHeight);
    } else {
        tok(e.tokens, 1, 64); tok(e.skip1, 1, 64); tok(e.merged1, 2, 96); tok(e.skip2, 2, 96); tok(e.merged2, 4, 128);
        tok(e.tubeA, 4, 128); tok(e.tubeB, 4, 128); tok(e.dec2Out, 2, 96); tok(e.dec1Out, 1, 64);
    }
    for (auto& t : e.mvA) t.create(d, renderWidth, renderHeight, VK_FORMAT_R16G16_SFLOAT);
    e.mvB.create(d, renderWidth, renderHeight, VK_FORMAT_R16G16_SFLOAT); e.mvC.create(d, renderWidth, renderHeight, VK_FORMAT_R16G16_SFLOAT);
    for (auto& t : e.feat) t.create(d, renderWidth, renderHeight, VK_FORMAT_R8G8B8A8_SNORM);
    e.gates.create(d, renderWidth, renderHeight, VK_FORMAT_R8G8B8A8_UNORM); e.mix.create(d, renderWidth, renderHeight, VK_FORMAT_R8G8B8A8_UNORM);
    e.cov.create(d, renderWidth, renderHeight, VK_FORMAT_R16G16B16A16_SFLOAT);
    for (auto& t : e.hist) t.create(d, e.hw, e.hh, VK_FORMAT_R16G16B16A16_SFLOAT);
    // The native formatted RGB10 history has RGBA32F backing. UNORM filtering differs by up to three ulps.
    for (auto& t : e.half) t.create(d, e.hw / 2, e.hh / 2, VK_FORMAT_R32G32B32A32_SFLOAT);
    e.output.create(d, outputWidth, outputHeight, VK_FORMAT_R16G16B16A16_SFLOAT);
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 48}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 24; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
    VKCHECK(vkCreateDescriptorPool(dev, &pi, nullptr, &e.pool));
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VKCHECK(vkCreateSampler(dev, &si, nullptr, &e.nearest));
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    VKCHECK(vkCreateSampler(dev, &si, nullptr, &e.linear));
    auto& m = e.model;
    // the network: tokens -> enc1 -> enc2 -> six tube blocks (ping-pong) -> dec2 -> dec1
    const Buffer* in[MLayers] = {&e.tokens, &e.merged1, &e.merged2, &e.tubeA, &e.tubeB, &e.tubeA, &e.tubeB, &e.tubeA, &e.tubeB, &e.dec2Out};
    const Buffer* skip[MLayers] = {&e.dummy, &e.dummy, &e.dummy, &e.dummy, &e.dummy, &e.dummy, &e.dummy, &e.dummy, &e.skip2, &e.skip1};
    const Buffer* out[MLayers] = {&e.skip1, &e.skip2, &e.tubeA, &e.tubeB, &e.tubeA, &e.tubeB, &e.tubeA, &e.tubeB, &e.dec2Out, &e.dec1Out};
    const Buffer* merged[MLayers] = {&e.merged1, &e.merged2, &e.dummy, &e.dummy, &e.dummy, &e.dummy, &e.dummy, &e.dummy, &e.dummy, &e.dummy};
    for (unsigned i = 0; i < MLayers && !external; ++i) {
        e.layerP[i].create(d, sizeof(MLayerParams), sbt, true);
        MLayerParams lp{int32_t(tw / MScale[i]), int32_t(th / MScale[i]), MShift[i], MShift[i], {}};
        std::memcpy(lp.o, m.layers[i].data() + 2, sizeof(lp.o));
        std::memcpy(e.layerP[i].mapped, &lp, sizeof(lp));
        auto set = e.layerSets[i] = descriptorSet(dev, e.pool, m.network[i].descriptors);
        bindBuffer(dev, set, 0, e.layerP[i]); bindRange(dev, set, 1, m.w8, m.layers[i][0]); bindRange(dev, set, 2, m.w16, m.layers[i][1]);
        bindBuffer(dev, set, 3, *in[i]); bindBuffer(dev, set, 4, *skip[i]); bindBuffer(dev, set, 5, *out[i]); bindBuffer(dev, set, 6, *merged[i]);
    }
    for (unsigned i = 0; i < 2; ++i) {      // i: the index of this frame's history outputs
        const unsigned prev = 1 - i;
        auto s0 = e.enc0Sets[i] = descriptorSet(dev, e.pool, (external ? m.inputPlanes : m.input).descriptors);
        bindBuffer(dev, s0, 0, e.enc0P);
        bindImage(dev, s0, 4, e.mvA[prev].view, e.linear); bindImage(dev, s0, 5, e.half[prev].view, e.linear); bindImage(dev, s0, 6, e.feat[prev].view, e.linear);
        bindImage(dev, s0, 7, e.mvA[i].view); bindImage(dev, s0, 8, e.mvB.view); bindImage(dev, s0, 9, e.mvC.view);
        bindRange(dev, s0, 10, m.w8, m.enc0[0]); bindRange(dev, s0, 11, m.w16, m.enc0[1]); bindRange(dev, s0, 12, m.w16, m.enc0[2]); bindBuffer(dev, s0, 13, e.tokens);
        auto s1 = e.dec0Sets[i] = descriptorSet(dev, e.pool, (external ? m.expandPlanes : m.expand).descriptors);
        bindBuffer(dev, s1, 0, e.dec0P); bindImage(dev, s1, 1, e.mvA[i].view, e.nearest); bindImage(dev, s1, 2, e.feat[prev].view, e.linear);
        bindImage(dev, s1, 3, e.feat[i].view); bindImage(dev, s1, 4, e.gates.view); bindImage(dev, s1, 5, e.mix.view); bindImage(dev, s1, 6, e.cov.view);
        bindBuffer(dev, s1, 7, e.dec1Out); bindRange(dev, s1, 8, m.w8, m.dec0[0]); bindRange(dev, s1, 9, m.w16, m.dec0[1]);
        auto s2 = e.postSets[i] = descriptorSet(dev, e.pool, m.post.descriptors);
        bindBuffer(dev, s2, 0, e.postP);
        bindImage(dev, s2, 2, e.mix.view, e.nearest); bindImage(dev, s2, 3, e.gates.view, e.nearest); bindImage(dev, s2, 4, e.cov.view, e.nearest);
        bindImage(dev, s2, 5, e.hist[prev].view, e.linear); bindImage(dev, s2, 6, e.mvA[i].view, e.nearest); bindImage(dev, s2, 7, e.mvB.view, e.nearest);
        bindImage(dev, s2, 8, e.mvC.view, e.linear); bindImage(dev, s2, 9, e.hist[i].view); bindImage(dev, s2, 10, e.half[i].view); bindBuffer(dev, s2, 11, e.packed);
        e.expSets[i] = descriptorSet(dev, e.pool, m.exposure[i].descriptors);
        bindBuffer(dev, e.expSets[i], 0, e.expP);
    }
    bindBuffer(dev, e.expSets[0], 2, e.expSums);
    bindBuffer(dev, e.expSets[1], 1, e.expSums); bindBuffer(dev, e.expSets[1], 2, e.expState);
    bindBuffer(dev, e.expSets[1], 3, e.enc0P); bindBuffer(dev, e.expSets[1], 4, e.postP); bindBuffer(dev, e.expSets[1], 5, e.downP);
    e.downSet = descriptorSet(dev, e.pool, m.down.descriptors);
    bindBuffer(dev, e.downSet, 0, e.downP); bindImage(dev, e.downSet, 2, e.output.view); bindBuffer(dev, e.downSet, 3, e.packed);
}
MEngine::~MEngine() = default;
void MEngine::recordInitialize(VkCommandBuffer cb) {
    auto& e = *impl;
    if (e.initialized) throw std::runtime_error("engine initialization already recorded");
    VkClearColorValue zero{}; VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (Tex* t : {&e.mvA[0], &e.mvA[1], &e.mvB, &e.mvC, &e.feat[0], &e.feat[1], &e.gates, &e.mix, &e.cov, &e.hist[0], &e.hist[1], &e.half[0], &e.half[1], &e.output}) {
        imageBarrier(cb, t->handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdClearColorImage(cb, t->handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }
    const float none[2] = {0, 0};
    vkCmdUpdateBuffer(cb, e.expState.handle, 0, sizeof(none), none);
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    e.initialized = true;
}
void MEngine::setFrame(const FrameImages& images, const MFrameSettings& s, FrameOutput finalOutput) {
    auto& e = *impl;
    if (!images.color || !images.motion || !images.depth) throw std::runtime_error("frame has a null image view");
    if (finalOutput.view && finalOutput.format != VK_FORMAT_R16G16B16A16_SFLOAT &&
        (finalOutput.format != VK_FORMAT_A2B10G10R10_UNORM_PACK32 || !e.model.downRgb10.handle))
        throw std::runtime_error("unsupported borrowed output format or missing RGB10A2 shader");
    const bool reset = s.reset || e.frame == 0;
    const float fx = float(e.hw) / float(e.rw), fy = float(e.hh) / float(e.rh);
    e.enc0 = {{int32_t(e.rw), int32_t(e.rh)}, {2, 2}, {int32_t(e.gw / 2), int32_t(e.gh / 2)}, {s.jitter[0], s.jitter[1]}, {e.prevJitter[0], e.prevJitter[1]},
              {s.motionScale[0] * fx, s.motionScale[1] * fy}, {s.motionOffset[0], s.motionOffset[1]}, {fx, fy}, {1.f / fx, 1.f / fy}, {float(e.hw), float(e.hh)}, 9.045065879821777f, 0.22356846928596497f,
              s.depthInverted ? 1u : 0u, reset ? 1u : 0u, e.frame & 7u, 1.f, 0.f, {1.f / float(e.hw), 1.f / float(e.hh)}, {1.f / float(e.rw), 1.f / float(e.rh)}};
    e.dec0 = {{int32_t(e.rw), int32_t(e.rh)}, int32_t(e.gw / 2), reset ? 1u : 0u, {s.jitter[0], s.jitter[1]}, {fx, fy}, {float(e.hw), float(e.hh)},
              {e.prevJitter[0], e.prevJitter[1]}, 8.f, 0.07863261550664902f, {1.f / float(e.rw), 1.f / float(e.rh)}};
    e.post = {{0, 0}, {int32_t(e.hw), int32_t(e.hh)}, {int32_t(e.rw), int32_t(e.rh)}, {1.f / fx, 1.f / fy}, {s.jitter[0], s.jitter[1]},
              {1.f / float(e.rw), 1.f / float(e.rh)}, 2.7530112f, reset ? 1u : 0u, 1.f, 0.f};
    e.down = {{int32_t(e.hw), int32_t(e.hh)}, {0, 0}, {0, 0}, {int32_t(e.rw), int32_t(e.rh)}, {float(e.rw) / float(e.ow), float(e.rh) / float(e.oh)}, {s.jitter[0], s.jitter[1]}, 1.f};
    const uint32_t sw = e.ow / 2, sh = e.oh / 2;
    const uint32_t bx = (sw + 31) / 32, by = (sh + 31) / 32;
    e.exposure = {{0, 0}, {int32_t(e.rw), int32_t(e.rh)}, 1.f / (float(sw) * float(sh)), 1.f, 0.23547087609767914f, 1u | (reset ? 2u : 0u), bx * by, bx, {int32_t(sw), int32_t(sh)}};
    e.prevJitter[0] = s.jitter[0]; e.prevJitter[1] = s.jitter[1];
    auto dev = e.model.device.handle;
    auto& m = e.model;
    auto found = std::find_if(e.bindings.begin(), e.bindings.end(), [&](const auto& b) {
        return b.images.color == images.color && b.images.motion == images.motion && b.images.depth == images.depth &&
            b.images.colorLayout == images.colorLayout && b.images.motionLayout == images.motionLayout &&
            b.images.depthLayout == images.depthLayout && b.finalOutput.view == finalOutput.view &&
            b.finalOutput.format == finalOutput.format;
    });
    if (found == e.bindings.end()) {
        Impl::Bindings b;
        b.images = images; b.finalOutput = finalOutput;
        if (e.bindings.empty()) {
            b.input = e.enc0Sets; b.post = e.postSets; b.exposure = e.expSets[0]; b.down = e.downSet;
        } else {
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16}};
            VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            info.maxSets = 6; info.poolSizeCount = 3; info.pPoolSizes = sizes;
            VKCHECK(vkCreateDescriptorPool(dev, &info, nullptr, &b.pool));
        }
        try {
            if (b.pool) {
                b.exposure = descriptorSet(dev, b.pool, m.exposure[0].descriptors);
                b.down = descriptorSet(dev, b.pool, m.down.descriptors);
            }
            bindBuffer(dev, b.exposure, 0, e.expP); bindImage(dev, b.exposure, 1, images.color, e.nearest, images.colorLayout);
            bindBuffer(dev, b.exposure, 2, e.expSums);
            bindBuffer(dev, b.down, 0, e.downP); bindImage(dev, b.down, 1, images.color, e.nearest, images.colorLayout);
            bindImage(dev, b.down, 2, finalOutput.view ? finalOutput.view : e.output.view); bindBuffer(dev, b.down, 3, e.packed);
            for (unsigned i = 0; i < 2; ++i) {
                if (b.pool) {
                    b.input[i] = descriptorSet(dev, b.pool, (e.external ? m.inputPlanes : m.input).descriptors);
                    b.post[i] = descriptorSet(dev, b.pool, m.post.descriptors);
                }
                auto in = b.input[i], post = b.post[i];
                bindBuffer(dev, in, 0, e.enc0P);
                bindImage(dev, in, 1, images.color, e.nearest, images.colorLayout);
                bindImage(dev, in, 2, images.motion, e.nearest, images.motionLayout);
                bindImage(dev, in, 3, images.depth, e.nearest, images.depthLayout);
                bindImage(dev, in, 4, e.mvA[1 - i].view, e.linear); bindImage(dev, in, 5, e.half[1 - i].view, e.linear);
                bindImage(dev, in, 6, e.feat[1 - i].view, e.linear);
                bindImage(dev, in, 7, e.mvA[i].view); bindImage(dev, in, 8, e.mvB.view); bindImage(dev, in, 9, e.mvC.view);
                bindRange(dev, in, 10, m.w8, m.enc0[0]); bindRange(dev, in, 11, m.w16, m.enc0[1]);
                bindRange(dev, in, 12, m.w16, m.enc0[2]); bindBuffer(dev, in, 13, e.tokens);
                bindBuffer(dev, post, 0, e.postP); bindImage(dev, post, 1, images.color, e.nearest, images.colorLayout);
                bindImage(dev, post, 2, e.mix.view, e.nearest); bindImage(dev, post, 3, e.gates.view, e.nearest);
                bindImage(dev, post, 4, e.cov.view, e.nearest); bindImage(dev, post, 5, e.hist[1 - i].view, e.linear);
                bindImage(dev, post, 6, e.mvA[i].view, e.nearest); bindImage(dev, post, 7, e.mvB.view, e.nearest);
                bindImage(dev, post, 8, e.mvC.view, e.linear); bindImage(dev, post, 9, e.hist[i].view);
                bindImage(dev, post, 10, e.half[i].view); bindBuffer(dev, post, 11, e.packed);
            }
            e.bindings.push_back(b);
        } catch (...) {
            if (b.pool) vkDestroyDescriptorPool(dev, b.pool, nullptr);
            throw;
        }
        found = e.bindings.end() - 1;
    }
    e.enc0Sets = found->input; e.postSets = found->post;
    e.expSets[0] = found->exposure; e.downSet = found->down;
    e.rgb10Output = finalOutput.view && finalOutput.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    e.frameSet = true;
}
namespace {
const auto compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
void nextStage(VkCommandBuffer cb) { memoryBarrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT); }
void openStages(VkCommandBuffer cb) {
    memoryBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, compute | VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}
void closeStages(VkCommandBuffer cb) { memoryBarrier(cb, compute, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT); }
}
// exposure and the input stage; writes three stamps
void MEngine::recordFront(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) {
    auto& e = *impl; auto& m = e.model;
    if (!e.initialized || !m.uploaded || !e.frameSet || e.frontDone) throw std::runtime_error("initialize/upload/setFrame before recording inference");
    openStages(cb);
    vkCmdUpdateBuffer(cb, e.enc0P.handle, 0, sizeof(e.enc0), &e.enc0); vkCmdUpdateBuffer(cb, e.dec0P.handle, 0, sizeof(e.dec0), &e.dec0);
    vkCmdUpdateBuffer(cb, e.postP.handle, 0, sizeof(e.post), &e.post); vkCmdUpdateBuffer(cb, e.downP.handle, 0, sizeof(e.down), &e.down);
    vkCmdUpdateBuffer(cb, e.expP.handle, 0, sizeof(e.exposure), &e.exposure);
    if (!e.external) for (unsigned l = 0; l < MLayers; ++l) {
        const int32_t shifts[2] = {MShift[l] ^ int32_t((e.frame & 1u) * 2u), MShift[l]};
        vkCmdUpdateBuffer(cb, e.layerP[l].handle, offsetof(MLayerParams, sx), sizeof(shifts), shifts);
    }
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, compute, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    uint32_t query = first;
    auto stamp = [&] { if (stamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, stamps, query++); };
    stamp();
    if (!m.ldr) {
        dispatch(cb, m.exposure[0], e.expSets[0], e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX); nextStage(cb);
        dispatch(cb, m.exposure[1], e.expSets[1], 1, 1); nextStage(cb);
    }
    stamp();
    dispatch(cb, e.external ? m.inputPlanes : m.input, e.enc0Sets[e.current], (e.gw + 2 + 7) / 8, (e.gh + 2 + 7) / 8);
    stamp();
    if (e.external) closeStages(cb); else nextStage(cb);
    e.frontDone = true;
}
// expansion, reconstruction and downsample; writes four stamps (the first before the expansion)
void MEngine::recordBack(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) {
    auto& e = *impl; auto& m = e.model;
    if (!e.frontDone) throw std::runtime_error("recordFront before recordBack");
    if (e.external) openStages(cb);
    const unsigned i = e.current;
    uint32_t query = first;
    auto stamp = [&] { if (stamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, stamps, query++); };
    stamp();
    dispatch(cb, e.external ? m.expandPlanes : m.expand, e.dec0Sets[i], e.gw / 8, e.gh / 8); nextStage(cb);
    stamp();
    const bool compact = e.compact && e.post.rndOffs[0] >= -0.5f && e.post.rndOffs[0] <= 0.5f &&
        e.post.rndOffs[1] >= -0.5f && e.post.rndOffs[1] <= 0.5f;
    dispatch(cb, compact ? m.postCompact : (e.quality ? m.postQuality : m.post),
             e.postSets[i], (e.hw + 15) / 16, (e.hh + 15) / 16); nextStage(cb);
    stamp();
    // D4RM0003's downsample workgroup covers 8x64 output pixels.
    dispatch(cb, e.rgb10Output ? m.downRgb10 : m.down, e.downSet, (e.ow + 7) / 8, (e.oh + 63) / 64);
    stamp();
    closeStages(cb);
    e.current = 1 - e.current; ++e.frame; e.frameSet = false; e.frontDone = false;
}
void MEngine::recordFrame(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) {
    auto& e = *impl; auto& m = e.model;
    if (e.external) throw std::runtime_error("this engine's network is external: record the front and the back");
    recordFront(cb, stamps, first);
    for (unsigned l = 0; l < MLayers; ++l) {
        const uint32_t w = e.gw / 2 / MScale[l], h = e.gh / 2 / MScale[l];
        dispatch(cb, m.network[l], e.layerSets[l], (w + (MShift[l] ^ ((e.frame & 1u) * 2u)) + 7) / 8, (h + MShift[l] + 7) / 8); nextStage(cb);
    }
    recordBack(cb, stamps, stamps ? first + 3 : 0);
}
VkImage MEngine::outputImage() const { return impl->output.handle; }
