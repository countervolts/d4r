// Super Resolution presets E and F (the CNN of nvngx_dlss.dll 310.7): the runtime around engine/sr/e_net.comp and
// the translated input and reconstruction kernels of a "D4RE0002" model (engine/compile_sr_e.py). Included by
// runtime.cpp after runtime_rr.inc (its helpers: Tex, bindRange, put32 / putF / put64).
//
// The frame: the input kernel (the game's colour, depth and motion, the previous output history and motion to the
// network's 16 channels per token, the warped history and this frame's output-size motion), the eleven network
// stages, the reconstruction kernel (the last decoder level, the head, the output and the next history). One buffer
// (the arena) holds every tensor; the translated kernels take its byte offsets in place of NGX's pointers.
//
// The two translated kernels' argument blocks are NGX's (harness captures at seven sizes and ratios, 2026-10-18):
// the dump's block as a template, every field that depends on the sizes, the frame or the flags rewritten here.
namespace {
constexpr unsigned ENet = 11;
constexpr const char* ENames[ENet] = {"enc0", "enc1", "enc2", "enc3", "mid0", "mid1", "up4", "red4", "dec3", "dec2", "dec1"};
// the network's tensors: name, channels, level (L0 = the token grid), 8-bit
struct ETensor { const char* name; uint32_t channels, level; bool e8; };
constexpr ETensor ETensors[] = {
    {"A", 16, 0, false}, {"B", 32, 0, true}, {"S1", 64, 1, true}, {"P1", 64, 2, true}, {"K2", 128, 2, false},
    {"P2", 128, 3, false}, {"K3", 128, 3, false}, {"P3", 128, 4, false}, {"K4", 256, 4, false}, {"P4", 256, 5, false},
    {"Q5", 512, 5, false}, {"R5", 256, 5, false}, {"D4", 256, 4, false}, {"E4", 128, 4, false}, {"D3", 128, 3, false},
    {"D2", 64, 2, true}, {"D1", 32, 1, true}};
constexpr unsigned ETensorCount = sizeof(ETensors) / sizeof(ETensors[0]);
// per stage: the level it runs at and its tensors (input, skip or second output, output, third output; -1 none)
struct EStageIO { uint32_t level; int in, skip, out, out2; };
constexpr EStageIO EStages[ENet] = {
    {0, 0, 1, 2, 3}, {2, 3, 4, 5, -1}, {3, 5, 6, 7, -1}, {4, 7, 8, 9, -1}, {5, 9, -1, 10, -1}, {5, 10, -1, 11, -1},
    {4, 11, 8, 12, -1}, {4, 12, -1, 13, -1}, {3, 13, 6, 14, -1}, {2, 14, 4, 15, -1}, {1, 15, 2, 16, -1}};

struct ETranslated {
    std::string shader, globalsFile;
    std::vector<uint32_t> textures;
    std::vector<std::pair<uint32_t, std::string>> surfaces;
    std::unique_ptr<Pipeline> pipeline;
    Buffer globals;
    std::vector<char> pending;
};
} // namespace

struct EModel::Impl {
    Device device;
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::string directory;
    std::array<Pipeline, ENet> stages;
    std::array<std::array<uint32_t, 4>, ENet> weightAt{};   // word offsets in weights.bin: w1 b1 w2 b2
    std::array<uint32_t, ENet> pixelSets{};                    // pixel sets per lane (e_net.comp PS)
    struct Head { uint32_t arg; VkDeviceSize at, bytes; };
    std::vector<Head> head;
    Buffer weights;
    std::vector<char> weightsPending;
    std::map<std::string, std::unique_ptr<ETranslated>> input, final;
    std::vector<uint8_t> inputTemplate, finalTemplate;
    bool uploaded = false;
    std::array<Pipeline, 2> exposure;   // engine/exposure_k.comp MODE 3 (NGX's AutoExposure), when the model has it
    bool hasExposure = false;
    Pipeline exposureCopy;              // engine/sr/exposure_copy_e.comp: the previous frame's exposure (HDR)
    bool hasExposureCopy = false;
    uint32_t preset = 5;            // 5: E, 6: F
    float kspread = 0.035f;         // the reconstruction's constant at 584 times rx ry (E 0.035, F 0.1)
    Impl(VkPhysicalDevice physical, VkDevice dev, const std::string& dir, VkPipelineCache c)
        : device(physical, dev), cache(c), directory(dir) {
        const auto text = read(dir + "/e.txt");
        std::istringstream lines(std::string(text.begin(), text.end()));
        std::string line;
        if (!std::getline(lines, line) || line.rfind("D4RE0002", 0) != 0) throw std::runtime_error("not a preset E/F model (e.txt D4RE0002)");
        const auto b = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        std::array<bool, ENet> seen{};
        while (std::getline(lines, line)) {
            std::istringstream f(line);
            std::string key; f >> key;
            if (key == "preset") f >> preset;
            else if (key == "kspread") f >> kspread;
            else if (key == "weights") {
                std::string file; f >> file;
                weightsPending = read(dir + "/" + file);
                weights.create(device, weightsPending.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            } else if (key == "stage") {
                std::string name, shader; f >> name >> shader;
                unsigned s = 0;
                while (s < ENet && name != ENames[s]) ++s;
                if (s >= ENet || seen[s]) throw std::runtime_error("e.txt: invalid stage: " + line);
                for (auto& w : weightAt[s]) f >> w;
                if (!(f >> pixelSets[s])) pixelSets[s] = 1;
                if (pixelSets[s] != 1 && pixelSets[s] != 2) throw std::runtime_error("e.txt: invalid stage: " + line);
                stages[s].create(device.handle, dir + "/" + shader, {b, b, b}, 32, cache);
                seen[s] = true;
            } else if (key == "head") {
                for (std::string item; f >> item;) {
                    Head h{};
                    if (std::sscanf(item.c_str(), "%u:%llu:%llu", &h.arg, reinterpret_cast<unsigned long long*>(&h.at),
                                    reinterpret_cast<unsigned long long*>(&h.bytes)) != 3)
                        throw std::runtime_error("e.txt: invalid head record");
                    head.push_back(h);
                }
            } else if (key == "input" || key == "final") {
                // "input|final VARIANT SHADER.spv GLOBALS TEXTURES SURFACES"
                std::string variant, textureList, surfaceList;
                auto entry = std::make_unique<ETranslated>();
                ETranslated& k = *entry;
                f >> variant >> k.shader >> k.globalsFile >> textureList >> surfaceList;
                std::istringstream tl(textureList == "-" ? "" : textureList), sl(surfaceList == "-" ? "" : surfaceList);
                for (std::string item; std::getline(tl, item, ',');) k.textures.push_back(static_cast<uint32_t>(std::stoul(item)));
                for (std::string item; std::getline(sl, item, ',');) {
                    const auto colon = item.find(':');
                    k.surfaces.emplace_back(static_cast<uint32_t>(std::stoul(item.substr(0, colon))), item.substr(colon + 1));
                }
                (key == "input" ? input : final)[variant] = std::move(entry);
            } else if (key == "exposure") {
                // "exposure PASS0.spv PASS1.spv": the measurement over the colour, then the 1x1 exposure image
                std::string measure, update; f >> measure >> update;
                const auto t = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                exposure[0].create(device.handle, dir + "/" + measure, {b, t, b}, 32, cache);
                exposure[1].create(device.handle, dir + "/" + update, {b, b, b, s}, 32, cache);
                hasExposure = true;
            } else if (key == "exposure_copy") {
                std::string shader; f >> shader;
                exposureCopy.create(device.handle, dir + "/" + shader, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                                        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}, 32, cache);
                hasExposureCopy = true;
            } else if (key == "template") {
                std::string which, file; f >> which >> file;
                auto bytes = read(dir + "/" + file);
                (which == "input" ? inputTemplate : finalTemplate).assign(bytes.begin(), bytes.end());
            }
        }
        for (unsigned s = 0; s < ENet; ++s)
            if (!seen[s]) throw std::runtime_error("e.txt: incomplete preset E/F model");
        if (!weights.handle || head.size() != 4 || inputTemplate.size() != 408 || finalTemplate.size() != 656 || input.empty() || final.empty())
            throw std::runtime_error("e.txt: incomplete preset E/F model");
    }
    // a translated kernel, its pipeline built on first use (the engine needs one input and one reconstruction variant)
    ETranslated& kernel(bool isInput, const std::string& variant) {
        auto& table = isInput ? input : final;
        auto found = table.find(variant);
        if (found == table.end()) throw std::runtime_error("e.txt: no " + std::string(isInput ? "input" : "reconstruction") + " kernel " + variant);
        auto& k = *found->second;
        if (!k.pipeline) {
            const auto b = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, t = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            std::vector<VkDescriptorType> types{b, b, b};
            types.insert(types.end(), k.textures.size(), t);
            types.insert(types.end(), k.surfaces.size(), s);
            k.pipeline = std::make_unique<Pipeline>();
            k.pipeline->create(device.handle, directory + "/" + k.shader, types, 32, cache);
            k.pending = read(directory + "/" + k.globalsFile);
            k.globals.create(device, k.pending.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }
        return k;
    }
};
EModel::EModel(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache)
    : impl(std::make_unique<Impl>(physical, device, directory, cache)) {}
EModel::~EModel() = default;
void EModel::recordUpload(VkCommandBuffer cb) {
    auto& m = *impl;
    if (m.uploaded) throw std::runtime_error("model upload already recorded");
    recordContents(cb, m.weights, m.weightsPending);
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    m.uploaded = true;
}
bool EModel::uploadRecorded() const { return impl->uploaded; }
bool EModel::hasAutoExposure() const { return impl->hasExposure; }

struct EEngine::Impl {
    EModel::Impl& model;
    const Device& d;
    uint32_t rw, rh, ow, oh, pw, ph, h0, w0;   // render, output, the output padded to 128, the token grid
    bool hdr, lowRes, autoExposure;
    ETranslated *input = nullptr, *final = nullptr;
    std::string finalVariant;
    uint32_t finalGrid[2] = {0, 0};
    Buffer arena, params;
    std::array<VkDeviceSize, ETensorCount> at{}, bytes{};
    std::array<VkDeviceSize, 4> headAt{};
    VkDeviceSize scratch = 0, arenaBytes = 0;
    std::array<VkDeviceSize, ENet> paramAt{};
    VkDeviceSize inAt = 0, finAt = 0, paramBytes = 0;
    std::vector<uint8_t> paramStaging, inBlock, finBlock;
    std::array<Tex, 2> history, motionHistory;
    Tex warped, discard, zero, exposureOne, out;
    // AutoExposure: NGX's measurement (exposure_k.comp MODE 3) into `measured`, the 1x1 the kernels then sample
    Buffer expP, expSums, expState;
    ExposureParams exposure{};
    Tex measured;
    VkDescriptorSet exp1Set = VK_NULL_HANDLE;
    // HDR with a model that has the copy shader: the previous frame's exposure, the input kernel's texture 352
    Tex previousExposure;
    bool outNew = false;
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    VkDescriptorPool fixedPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, ENet> netSets{};
    struct Bindings {
        FrameImages images{};
        FrameOutput finalOutput{};
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> input{}, final{};
        VkDescriptorSet exposure = VK_NULL_HANDLE;   // the measurement over this colour view (AutoExposure)
        VkDescriptorSet copy = VK_NULL_HANDLE;       // this frame's exposure into previousExposure
    };
    std::vector<Bindings> bindings;
    const Bindings* bound = nullptr;
    VkQueryPool profile = VK_NULL_HANDLE;   // D4R_ENGINE_E_PROFILE=1: microseconds per pass, printed a frame later
    bool profiled = false;
    float prevJitter[2] = {0, 0};
    uint32_t frame = 0;
    bool reset = true;

    uint32_t lh(uint32_t l) const { return h0 >> l; }
    uint32_t lw(uint32_t l) const { return w0 >> l; }

    Impl(EModel::Impl& m, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight, bool isHdr,
         bool lowResMotion, bool isAutoExposure)
        : model(m), d(m.device), rw(renderWidth), rh(renderHeight), ow(outputWidth), oh(outputHeight), hdr(isHdr), lowRes(lowResMotion),
          autoExposure(isAutoExposure) {
        if (rw < 32 || rh < 32 || rw > ow || rh > oh || ow > 16384 || oh > 16384)
            throw std::runtime_error("unsupported preset E/F dimensions");
        if (autoExposure && (!hdr || !model.hasExposure))
            throw std::runtime_error(hdr ? "the preset E/F model has no exposure shaders (rebuild it with compile_sr_e.py)"
                                         : "automatic exposure needs HDR colour");
        pw = (ow + 127) & ~127u; ph = (oh + 127) & ~127u; w0 = pw / 4; h0 = ph / 4;
        // The mid kernel uses the actual scale/jitter and also covers the integer-rounded 2/3 render sizes.
        // Requiring exact 3:2 dimensions sends e.g. 853x533 -> 1280x800 through the much larger generic kernel.
        // Restrict this extension to floor/ceil rounding on each axis; retain the generic path at other ratios.
        const auto qualityAxis = [](uint32_t render, uint32_t output) {
            return render == (2 * output) / 3 || render == (2 * output + 2) / 3;
        };
        input = &model.kernel(true, std::string(hdr ? "hdr" : "ldr") + "_mvdiff_" + (lowRes ? "mvlo" : "mvhi"));
        const double sx = double(ow) / rw, sy = double(oh) / rh, s = std::min(sx, sy);
        if (s > 2.5) finalVariant = "fastpath_5x5_256";
        else if (s >= 1.6) finalVariant = "fastpath_3x3_128";
        else if (qualityAxis(rw, ow) && qualityAxis(rh, oh)) finalVariant = "mid_3x3_128";
        else finalVariant = "3x3_256";
        if (const char* v = std::getenv("D4R_ENGINE_E_FINAL")) finalVariant = v;   // development: force a variant
        final = &model.kernel(false, finalVariant);
        const bool generic = finalVariant == "3x3_256" || finalVariant == "5x5_256";
        finalGrid[0] = (ow + 63) / 64; finalGrid[1] = generic ? (oh + 63) / 64 : (oh + 31) / 32;
        // the arena: the network's tensors, the head weights, a scratch word block (NGX's unused pointer 256)
        VkDeviceSize a = 0;
        auto region = [&](VkDeviceSize n) { const VkDeviceSize r = a; a += (n + 255) & ~VkDeviceSize(255); return r; };
        for (unsigned t = 0; t < ETensorCount; ++t) {
            const auto& e = ETensors[t];
            bytes[t] = VkDeviceSize(e.channels) * lh(e.level) * lw(e.level) * (e.e8 ? 1 : 2);
            at[t] = region(bytes[t]);
        }
        for (unsigned i = 0; i < 4; ++i) headAt[i] = region(model.head[i].bytes);
        scratch = region(256);
        arenaBytes = a;
        const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        arena.create(d, arenaBytes, storage);
        auto place = [&](size_t n) { const VkDeviceSize r = paramBytes; paramBytes += (n + 255) & ~VkDeviceSize(255); return r; };
        for (unsigned s = 0; s < ENet; ++s) paramAt[s] = place(64);
        inAt = place(model.inputTemplate.size()); finAt = place(model.finalTemplate.size());
        params.create(d, paramBytes, storage);
        paramStaging.assign(static_cast<size_t>(paramBytes), 0);
        for (unsigned s = 0; s < ENet; ++s) {
            const auto& io = EStages[s];
            uint32_t p[10] = {lh(io.level), lw(io.level), uint32_t(at[io.in]), io.skip >= 0 ? uint32_t(at[io.skip]) : 0u,
                              uint32_t(at[io.out]), io.out2 >= 0 ? uint32_t(at[io.out2]) : 0u,
                              model.weightAt[s][0], model.weightAt[s][1], model.weightAt[s][2], model.weightAt[s][3]};
            std::memcpy(paramStaging.data() + paramAt[s], p, sizeof(p));
        }
        const VkFormat half4 = VK_FORMAT_R16G16B16A16_SFLOAT, half2 = VK_FORMAT_R16G16_SFLOAT;
        for (int i = 0; i < 2; ++i) { history[i].create(d, ow, oh, half4); motionHistory[i].create(d, ow, oh, half2); }
        warped.create(d, ow, oh, half4);
        discard.create(d, 1, 1, half4);
        zero.create(d, 1, 1, half4);
        exposureOne.create(d, 1, 1, VK_FORMAT_R32_SFLOAT);
        if (autoExposure) {
            // NGX's cuda_luma_convert_kernel grid (harness captures 2026-10-19, like M's and L's): half the output,
            // half the render size but at least 256 where the output is more than twice the render size
            const uint32_t gx = exposureGrid(rw, ow), gy = exposureGrid(rh, oh);
            exposure.size[0] = int32_t(rw); exposure.size[1] = int32_t(rh);
            exposure.sampleSize[0] = int32_t(gx); exposure.sampleSize[1] = int32_t(gy);
            exposure.invCount = 1.0f / float(gx * gy);
            exposure.key = 0.18f / 0.82f;   // cuda_auto_exposure_copy_kernel's argument 0
            exposure.blocksX = (gx + 31) / 32;
            exposure.blocks = exposure.blocksX * ((gy + 31) / 32);
            expP.create(d, sizeof(ExposureParams), storage);
            expSums.create(d, VkDeviceSize(exposure.blocks) * 4, storage);
            expState.create(d, 8, storage);
            measured.create(d, 1, 1, VK_FORMAT_R16_SFLOAT);
        }
        if (hdr && model.hasExposureCopy) previousExposure.create(d, 1, 1, VK_FORMAT_R16_SFLOAT);
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = 0;
        VKCHECK(vkCreateSampler(d.handle, &si, nullptr, &nearest));
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        VKCHECK(vkCreateSampler(d.handle, &si, nullptr, &linear));
        VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * ENet + 3}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = ENet + 1; pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
        VKCHECK(vkCreateDescriptorPool(d.handle, &pi, nullptr, &fixedPool));
        if (autoExposure) {
            exp1Set = descriptorSet(d.handle, fixedPool, model.exposure[1].descriptors);
            bindBuffer(d.handle, exp1Set, 0, expP);
            bindBuffer(d.handle, exp1Set, 1, expSums);
            bindBuffer(d.handle, exp1Set, 2, expState);
            bindImage(d.handle, exp1Set, 3, measured.view);
        }
        for (unsigned s = 0; s < ENet; ++s) {
            netSets[s] = descriptorSet(d.handle, fixedPool, model.stages[s].descriptors);
            bindRange(d.handle, netSets[s], 0, params, paramAt[s]);
            bindBuffer(d.handle, netSets[s], 1, model.weights);
            bindBuffer(d.handle, netSets[s], 2, arena);
        }
    }
    ~Impl() {
        if (profile) vkDestroyQueryPool(d.handle, profile, nullptr);
        for (auto& b : bindings) vkDestroyDescriptorPool(d.handle, b.pool, nullptr);
        if (fixedPool) vkDestroyDescriptorPool(d.handle, fixedPool, nullptr);
        if (nearest) vkDestroySampler(d.handle, nearest, nullptr);
        if (linear) vkDestroySampler(d.handle, linear, nullptr);
    }
    struct Resource { VkImageView view = VK_NULL_HANDLE; bool linear = false; VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL; };
    // what a translated kernel's texture or surface argument is (NGX's wiring in the captures; null where NGX leaves
    // it unbound). par: this frame's parity (history written this frame, the other one read)
    Resource resolve(bool isInput, bool surface, uint32_t offset, uint32_t par, const FrameImages& in, VkImageView output) const {
        const VkImageView exposure = !hdr ? VK_NULL_HANDLE : autoExposure ? measured.view : in.exposure ? in.exposure : exposureOne.view;
        const VkImageLayout exposureLayout = !autoExposure && in.exposure ? in.exposureLayout : VK_IMAGE_LAYOUT_GENERAL;
        if (isInput) {
            if (!surface) switch (offset) {
                case 280: return {in.color, true, in.colorLayout};
                case 288: return {history[par ^ 1].view, true};
                case 304: return {in.motion, false, in.motionLayout};
                case 312: return {in.depth, false, in.depthLayout};
                case 320: return {motionHistory[par ^ 1].view, true};
                case 344: case 360: return {exposure, false, exposureLayout};
                // NGX: the previous frame's exposure (its copy kernel's surface; the current one on a reset frame,
                // where the engine copies first). Models without the copy shader bind the current exposure.
                case 352: return previousExposure.view ? Resource{previousExposure.view} : Resource{exposure, false, exposureLayout};
            } else switch (offset) {
                case 328: return {motionHistory[par].view};
                case 368: return {discard.view};
                case 384: return {warped.view};
            }
        } else {
            if (!surface) switch (offset) {
                case 456: return {in.color, true, in.colorLayout};
                case 464: return {warped.view, false};
                case 472: case 480: return {exposure, false, exposureLayout};
            } else switch (offset) {
                case 488: return {output};
                case 496: return {history[par].view};
            }
        }
        return {};
    }
    VkDescriptorSet translatedSet(Bindings& b, bool isInput, uint32_t par, VkImageView output) {
        ETranslated& k = isInput ? *input : *final;
        VkDescriptorSet set = descriptorSet(d.handle, b.pool, k.pipeline->descriptors);
        bindRange(d.handle, set, 0, params, isInput ? inAt : finAt);
        bindBuffer(d.handle, set, 1, k.globals);
        bindBuffer(d.handle, set, 2, arena);
        uint32_t binding = 3;
        for (uint32_t offset : k.textures) {
            const auto r = resolve(isInput, false, offset, par, b.images, output);
            bindImage(d.handle, set, binding++, r.view ? r.view : zero.view, r.linear ? linear : nearest, r.view ? r.layout : VK_IMAGE_LAYOUT_GENERAL);
        }
        for (const auto& [offset, format] : k.surfaces) {
            const auto r = resolve(isInput, true, offset, par, b.images, output);
            bindImage(d.handle, set, binding++, r.view ? r.view : discard.view);
        }
        return set;
    }
    void setFrame(const FrameImages& images, const EFrameSettings& settings, FrameOutput finalOutput) {
        if (!images.color || !images.motion || (lowRes && !images.depth))
            throw std::runtime_error("preset E/F needs colour and motion (and depth with render-size motion)");
        if (finalOutput.view && finalOutput.format != VK_FORMAT_R16G16B16A16_SFLOAT)
            throw std::runtime_error("preset E/F writes RGBA16F output only");
        if (!(settings.preExposure > 0.0f) || !std::isfinite(settings.preExposure) || !(settings.exposureScale > 0.0f) ||
            !std::isfinite(settings.exposureScale))
            throw std::runtime_error("the pre-exposure and exposure scale must be positive numbers");
        const bool rst = settings.reset || frame == 0;
        const float frw = float(rw), frh = float(rh), fow = float(ow), foh = float(oh);
        const float rx = frw / fow, ry = frh / foh;
        const float pj[2] = {rst ? 0.0f : prevJitter[0], rst ? 0.0f : prevJitter[1]};
        auto& p = inBlock;
        p = model.inputTemplate;
        put32(p, 0, ow); put32(p, 4, oh); put32(p, 8, pw); put32(p, 12, ph); put32(p, 16, pw / 2); put32(p, 20, ph / 2);
        put32(p, 24, ow); put32(p, 28, oh); putF(p, 32, 1.0f / fow); putF(p, 36, 1.0f / foh);
        put32(p, 40, rw - 1); put32(p, 44, rh - 1);
        put32(p, 48, (lowRes ? rw : ow) - 1); put32(p, 52, (lowRes ? rh : oh) - 1);
        putF(p, 56, settings.motionScale[0] * (lowRes ? fow / frw : 1.0f)); putF(p, 60, settings.motionScale[1] * (lowRes ? foh / frh : 1.0f));
        putF(p, 72, lowRes ? rx : 1.0f); putF(p, 76, lowRes ? ry : 1.0f);
        putF(p, 80, settings.jitter[0]); putF(p, 84, settings.jitter[1]);
        put32(p, 108, hdr ? 1u : 0u); put32(p, 112, rst ? 1u : 0u);
        putF(p, 120, rx); putF(p, 124, ry); putF(p, 128, rx); putF(p, 132, ry);
        put32(p, 136, rw); put32(p, 140, rh); putF(p, 144, rx); putF(p, 148, ry);
        putF(p, 152, 1.0f / fow); putF(p, 156, 1.0f / foh); putF(p, 160, model.preset == 5 ? 500.0f / foh : 1.0f);
        put32(p, 184, lowRes ? 1u : 0u); put32(p, 188, lowRes ? 1u : 0u);
        // the game's exposure scale (twice), pre-exposure and its inverse: both kernels take this frame's values, with
        // or without AutoExposure (captures with per-frame schedules, 2026-10-19); the kept measurement is not rescaled
        const float exposureFields[4] = {settings.exposureScale, settings.exposureScale, settings.preExposure, 1.0f / settings.preExposure};
        for (unsigned i = 0; i < 4; ++i) putF(p, 244 + 4 * i, exposureFields[i]);
        put64(p, 376, at[0]);
        auto& o = finBlock;
        o = model.finalTemplate;
        put32(o, 40, lh(1)); put32(o, 44, lw(1)); put32(o, 60, h0); put32(o, 64, w0);
        put32(o, 88, lw(1) * 32); put32(o, 92, w0 * 32); put32(o, 96, w0 * 32);
        put32(o, 264, h0); put32(o, 268, w0); put32(o, 284, h0); put32(o, 288, w0);
        put32(o, 312, w0 * 32); put32(o, 316, w0 * 32); put32(o, 320, w0 * 40);
        put32(o, 504, oh); put32(o, 508, ow); put32(o, 512, hdr ? 1u : 0u);
        putF(o, 528, ry); putF(o, 532, 1.0f / ry); putF(o, 536, rx); putF(o, 540, 1.0f / rx);
        // the previous and this frame's jitter as (y, x): (-x, -y) agrees with it only on the harness's first three
        // jitters (0 / -+0.25), so later frames of a Halton sequence measured 26-28 dB (six-frame captures, 2026-10-19)
        putF(o, 544, pj[1]); putF(o, 548, pj[0]); putF(o, 552, settings.jitter[1]); putF(o, 556, settings.jitter[0]);
        put32(o, 564, rh - 1); put32(o, 572, rw - 1);
        putF(o, 584, model.kspread / (rx * ry));
        for (unsigned i = 0; i < 4; ++i) putF(o, 592 + 4 * i, exposureFields[i]);
        put64(o, 0, at[16]); put64(o, 8, at[1]);
        for (unsigned i = 0; i < 4; ++i) put64(o, model.head[i].arg, headAt[i]);
        put64(o, 256, scratch);
        if (!finalOutput.view && !out.handle) {
            out.create(d, ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT);
            outNew = true;
        }
        const VkImageView output = finalOutput.view ? finalOutput.view : out.view;
        // texture and surface handles: 1 where something is bound, 0 where NGX leaves the argument null
        for (uint32_t offset : input->textures) put64(p, offset, resolve(true, false, offset, 0, images, output).view ? 1u : 0u);
        for (const auto& [offset, format] : input->surfaces) put64(p, offset, offset == 368 ? 0u : 1u);   // NGX leaves 368 null
        for (uint32_t offset : final->textures) put64(o, offset, resolve(false, false, offset, 0, images, output).view ? 1u : 0u);
        for (const auto& [offset, format] : final->surfaces) put64(o, offset, 1u);
        auto same = [&](const Bindings& b) { return !std::memcmp(&b.images, &images, sizeof(images)) && b.finalOutput.view == finalOutput.view; };
        auto found = std::find_if(bindings.begin(), bindings.end(), same);
        if (found == bindings.end()) {
            if (bindings.size() >= 16) throw std::runtime_error("too many distinct preset E/F input view sets");
            Bindings b;
            b.images = images; b.finalOutput = finalOutput;
            const uint32_t tex = 2 * uint32_t(input->textures.size() + final->textures.size());
            const uint32_t surf = 2 * uint32_t(input->surfaces.size() + final->surfaces.size());
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 14}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex + 2},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, surf + 1}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = 6; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
            VKCHECK(vkCreateDescriptorPool(d.handle, &pi, nullptr, &b.pool));
            for (uint32_t par = 0; par < 2; ++par) {
                b.input[par] = translatedSet(b, true, par, output);
                b.final[par] = translatedSet(b, false, par, output);
            }
            if (autoExposure) {
                b.exposure = descriptorSet(d.handle, b.pool, model.exposure[0].descriptors);
                bindBuffer(d.handle, b.exposure, 0, expP);
                bindImage(d.handle, b.exposure, 1, images.color, nearest, images.colorLayout);
                bindBuffer(d.handle, b.exposure, 2, expSums);
            }
            if (previousExposure.view) {
                const auto current = resolve(true, false, 344, 0, images, output);
                b.copy = descriptorSet(d.handle, b.pool, model.exposureCopy.descriptors);
                bindImage(d.handle, b.copy, 0, current.view, nearest, current.layout);
                bindImage(d.handle, b.copy, 1, previousExposure.view);
            }
            bindings.push_back(std::move(b));
            found = bindings.end() - 1;
        }
        bound = &*found;
        if (const char* dump = std::getenv("D4R_ENGINE_E_DUMP")) {   // development: the frame's two argument blocks
            std::ofstream(std::string(dump) + "/" + std::to_string(frame + 1) + "-input.args", std::ios::binary)
                .write(reinterpret_cast<const char*>(inBlock.data()), static_cast<std::streamsize>(inBlock.size()));
            std::ofstream(std::string(dump) + "/" + std::to_string(frame + 1) + "-final.args", std::ios::binary)
                .write(reinterpret_cast<const char*>(finBlock.data()), static_cast<std::streamsize>(finBlock.size()));
        }
        if (autoExposure) {
            // NGX measures on every second evaluation from the first (and K's rule: on a reset too); the luma is
            // scaled by 1 / pre-exposure, and the image keeps the last measurement on the frames between
            exposure.scale = 1.0f / settings.preExposure;
            exposure.flags = (frame % 2 == 0 || rst ? 1u : 0u) | (rst ? 2u : 0u);
        }
        reset = rst;
        prevJitter[0] = settings.jitter[0]; prevJitter[1] = settings.jitter[1];
    }
    void recordInitialize(VkCommandBuffer cb) {
        VkClearColorValue clear{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        std::vector<VkImage> images{warped.handle, discard.handle, zero.handle, exposureOne.handle};
        if (autoExposure) images.push_back(measured.handle);
        if (previousExposure.handle) images.push_back(previousExposure.handle);
        for (int i = 0; i < 2; ++i) { images.push_back(history[i].handle); images.push_back(motionHistory[i].handle); }
        for (VkImage image : images) {
            imageBarrier(cb, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        }
        VkClearColorValue one{}; one.float32[0] = 1.0f;
        vkCmdClearColorImage(cb, exposureOne.handle, VK_IMAGE_LAYOUT_GENERAL, &one, 1, &range);
        vkCmdFillBuffer(cb, arena.handle, 0, VK_WHOLE_SIZE, 0);
        if (autoExposure) vkCmdFillBuffer(cb, expState.handle, 0, VK_WHOLE_SIZE, 0x3F800000u);   // 1.0 until the first measurement
        memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT);
        // the head weights into the arena (the reconstruction kernel reads them there)
        for (unsigned i = 0; i < 4; ++i) {
            VkBufferCopy region{model.head[i].at, headAt[i], model.head[i].bytes};
            vkCmdCopyBuffer(cb, model.weights.handle, arena.handle, 1, &region);
        }
        for (const auto* k : {input, final})
            if (!k->pending.empty()) recordContents(cb, k->globals, const_cast<ETranslated*>(k)->pending);
        memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        frame = 0; reset = true;
    }
    void recordFrame(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) {
        if (!bound) throw std::runtime_error("EEngine::setFrame was not called");
        const auto& b = *bound;
        const uint32_t cur = frame & 1u;
        auto stamp = [&](uint32_t n) { if (stamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, stamps, first + n); };
        auto step = [&] {
            memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        };
        if (outNew) {
            clearNewImage(cb, out.handle);
            outNew = false;
        }
        constexpr uint32_t passes = ENet + 2;
        if (!profile && std::getenv("D4R_ENGINE_E_PROFILE")) {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType = VK_QUERY_TYPE_TIMESTAMP; qi.queryCount = passes + 1;
            VKCHECK(vkCreateQueryPool(d.handle, &qi, nullptr, &profile));
        }
        if (profile && profiled) {
            uint64_t t[passes + 1];
            if (vkGetQueryPoolResults(d.handle, profile, 0, passes + 1, sizeof(t), t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                double total = 0;
                std::fprintf(stderr, "sr cnn profile (us):");
                for (uint32_t i = 0; i < passes; ++i) {
                    const double us = (t[i + 1] - t[i]) * d.properties.limits.timestampPeriod * 1e-3;
                    total += us;
                    std::fprintf(stderr, " %s %.1f", i == 0 ? "input" : i == passes - 1 ? "final" : ENames[i - 1], us);
                }
                std::fprintf(stderr, " total %.1f\n", total);
            }
        }
        uint32_t mark = 0;
        auto pass = [&] { if (profile) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, profile, mark++); };
        if (profile) vkCmdResetQueryPool(cb, profile, 0, passes + 1);
        stamp(0);
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        std::memcpy(paramStaging.data() + inAt, inBlock.data(), inBlock.size());
        std::memcpy(paramStaging.data() + finAt, finBlock.data(), finBlock.size());
        for (VkDeviceSize a = 0; a < paramBytes; a += 65536)
            vkCmdUpdateBuffer(cb, params.handle, a, std::min<VkDeviceSize>(65536, paramBytes - a), paramStaging.data() + a);
        if (autoExposure) vkCmdUpdateBuffer(cb, expP.handle, 0, sizeof(ExposureParams), &exposure);
        memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        pass();
        if (autoExposure) {   // counted in the input pass's time
            if (exposure.flags & 1u) {
                dispatch(cb, model.exposure[0], b.exposure, exposure.blocksX, exposure.blocks / exposure.blocksX);
                step();
            }
            dispatch(cb, model.exposure[1], exp1Set, 1, 1);
            step();
        }
        if (b.copy && reset) {   // NGX's reset frame reads the current exposure as the previous one
            dispatch(cb, model.exposureCopy, b.copy, 1, 1);
            step();
        }
        dispatch(cb, *input->pipeline, b.input[cur], (ow + 15) / 16, (oh + 15) / 16);
        step(); stamp(1); pass();
        for (unsigned s = 0; s < ENet; ++s) {
            const uint32_t l = EStages[s].level;
            const uint32_t ps = model.pixelSets[s];
            uint32_t gx, gy = 1;
            if (s == 0) { gx = w0 / 16; gy = h0 / (8 * ps); }
            else if (s <= 3) { gx = (lw(l) + 7) / 8; gy = (lh(l) + 4 * ps - 1) / (4 * ps); }
            else gx = (lw(l) * lh(l) + 32 * ps - 1) / (32 * ps);
            dispatch(cb, model.stages[s], netSets[s], gx, gy);
            step(); pass();
        }
        stamp(2);
        dispatch(cb, *final->pipeline, b.final[cur], finalGrid[0], finalGrid[1]);
        // the next frame's previous exposure (the input kernel read this frame's before the network)
        if (b.copy) dispatch(cb, model.exposureCopy, b.copy, 1, 1);
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
        stamp(3); pass();
        profiled = profile != VK_NULL_HANDLE;
        ++frame;
    }
};
EEngine::EEngine(EModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight, bool hdr,
                 bool lowResMotion, bool autoExposure)
    : impl(std::make_unique<Impl>(*model.impl, renderWidth, renderHeight, outputWidth, outputHeight, hdr, lowResMotion, autoExposure)) {
    if (!model.impl->uploaded) {}   // the caller records the model's upload before the first frame
}
EEngine::~EEngine() = default;
void EEngine::recordInitialize(VkCommandBuffer cb) { impl->recordInitialize(cb); }
void EEngine::setFrame(const FrameImages& images, const EFrameSettings& settings, FrameOutput finalOutput) { impl->setFrame(images, settings, finalOutput); }
void EEngine::recordFrame(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) { impl->recordFrame(cb, stamps, first); }
VkImage EEngine::outputImage() const { return impl->out.handle; }
std::vector<EEngine::DebugBuffer> EEngine::debugBuffers() const {
    std::vector<DebugBuffer> list;
    for (unsigned t = 0; t < ETensorCount; ++t) list.push_back({ETensors[t].name, impl->arena.handle, impl->at[t], impl->bytes[t]});
    return list;
}
