// Ray Reconstruction presets A - C (the CNN of nvngx_dlssd.dll 310.3): the runtime around the stage shaders of a
// "D4RR0003" model (engine/compile_rr_cnn.py). Included by runtime_rr.inc after its helpers.
//
// NGX's frame: the exposure (the same kernels as presets D and E), the input kernel, five encoder levels, five decoder
// levels and the last one (PDL), then the output kernel. The input and output kernels are translated from the DLL
// (engine/rr/ptx2rr.py) and take arena offsets for pointers; the network stages are d4r's own shaders
// (engine/rr/cnn_enc.comp, cnn_dec.comp), which take device addresses. Their parameter blocks are NVIDIA's argument
// blocks: the network's are tensor descriptors this file fills for the render size, the input and output kernels' are
// presets D and E's encoder 0 and output blocks at other offsets (harness captures of presets A - C, 2026-10-17;
// engine/rr/README.md). Every buffer NGX allocates for the network lives in one arena, at offsets that scale with
// W0 * H0 (the render size rounded up to 32 on both axes).
namespace {
constexpr unsigned R3Network = 11, R3Input = 11, R3Output = 12, R3Blocks = 13;
constexpr unsigned R3Enc4 = 4, R3Dec4 = 5, R3Pdl = 10;
constexpr const char* R3Names[R3Network] = {"Enc0", "Enc1", "Enc2", "Enc3", "Enc4", "Dec4", "Dec3", "Dec2", "Dec1", "Dec0", "PDL"};
constexpr uint32_t R3InputBytes = 848, R3OutputBytes = 424, R3NetworkBytes = 296;
// per stage: its level (the encoders' and decoders' output level; Dec4 works at level 5) and its input and output
// channels (engine/rr/cnn_dll.py CHANNELS)
constexpr int R3Level[R3Network] = {0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0};
constexpr uint32_t R3In[R3Network] = {16, 32, 64, 96, 128, 160, 160, 128, 96, 64, 32};
constexpr uint32_t R3Out[R3Network] = {32, 64, 96, 128, 160, 160, 128, 96, 64, 32, 32};
// The arena in units of W0 * H0 / 2 bytes: NGX's allocations one after another (harness captures, 640x360). The main
// one (0 - 64) holds most levels' activations; 64 and 80 are the input kernel's two outputs (read by Enc0, PDL and the
// output kernel), 96 encoder 0's skip tensor (PDL's), then the smaller per-level tensors and images.
constexpr uint32_t R3ArenaUnits = 234;
struct R3Pointer { uint32_t offset, units; };
constexpr R3Pointer R3InputPointers[] = {{296, 0}, {304, 64}, {312, 80}};
constexpr R3Pointer R3OutputPointers[] = {{224, 0}, {232, 64}, {240, 80}};
const std::vector<R3Pointer> R3Arena[R3Network] = {
    {{0, 0}, {120, 96}, {152, 160}, {184, 176}, {192, 208}, {200, 64}},      // Enc0
    {{0, 160}, {120, 0}, {152, 32}, {184, 40}, {192, 59}, {200, 208}},       // Enc1
    {{0, 32}, {120, 160}, {152, 208}, {184, 214}, {192, 60}, {200, 59}},     // Enc2
    {{0, 208}, {120, 32}, {152, 59}, {184, 48}, {192, 230}, {200, 60}},      // Enc3
    {{0, 59}, {120, 208}, {152, 60}, {184, 56}, {192, 61}, {200, 230}},      // Enc4
    {{0, 60}, {192, 59}, {240, 230}, {256, 61}},                             // Dec4
    {{0, 59}, {32, 208}, {192, 60}, {240, 61}, {256, 230}, {264, 56}},       // Dec3
    {{0, 60}, {32, 32}, {192, 56}, {240, 230}, {256, 61}, {264, 48}},        // Dec2
    {{0, 56}, {32, 160}, {192, 48}, {240, 61}, {256, 230}, {264, 214}},      // Dec1
    {{0, 48}, {32, 0}, {192, 214}, {240, 230}, {256, 61}, {264, 40}},        // Dec0
    {{0, 214}, {32, 96}, {216, 0}, {240, 64}, {256, 230}, {264, 176}},       // PDL
};

struct R3Model {
    Pipeline input, output;
    std::vector<uint32_t> textures[2];                              // the input and output kernels'
    std::vector<std::pair<uint32_t, std::string>> surfaces[2];
    Buffer consts;                                                  // the input kernel's constant tables (binding 1)
    std::vector<char> constsPending;
    std::array<Pipeline, R3Network> stages;
    std::array<uint32_t, R3Network> tileRows{};                     // a workgroup's tile: 16 x tileRows pixels
    std::array<std::vector<std::pair<uint32_t, uint64_t>>, R3Network> weightPointers;   // argument offset, byte in weights
    Buffer weights;
    std::vector<char> weightsPending;
    std::array<Pipeline, 2> exposure;
    void load(const Device& d, const std::string& directory, std::istringstream& lines, VkPipelineCache cache,
              const VkSpecializationInfo* specialization) {
        const auto b = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, t = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        auto split = [](const std::string& list) {
            std::vector<std::string> items;
            if (list == "-") return items;
            std::istringstream in(list);
            for (std::string item; std::getline(in, item, ',');) items.push_back(item);
            return items;
        };
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        std::array<bool, R3Network> seen{};
        bool translated[2] = {}, weightsSeen = false;
        std::string line;
        while (std::getline(lines, line)) {
            std::istringstream f(line);
            std::string key; f >> key;
            if (key == "input" || key == "output") {
                // "input|output SHADER.spv CONSTS|- TEXTURES SURFACES": the translated kernels' bindings (engine/rr/README.md)
                const int k = key == "output";
                std::string shader, constFile, textureList, surfaceList;
                f >> shader >> constFile >> textureList >> surfaceList;
                for (const auto& item : split(textureList)) textures[k].push_back(static_cast<uint32_t>(std::stoul(item)));
                for (const auto& item : split(surfaceList)) {
                    const auto colon = item.find(':');
                    surfaces[k].emplace_back(static_cast<uint32_t>(std::stoul(item.substr(0, colon))), item.substr(colon + 1));
                }
                std::vector<VkDescriptorType> types{b, b, b};
                types.insert(types.end(), textures[k].size(), t);
                types.insert(types.end(), surfaces[k].size(), s);
                traceStage(directory + "/" + shader);
                (k ? output : input).create(d.handle, directory + "/" + shader, types, 32, cache, 0, specialization);
                if (!k) {
                    auto bytes = constFile == "-" ? std::vector<char>(16) : read(directory + "/" + constFile);
                    consts.create(d, bytes.size(), usage);
                    constsPending = std::move(bytes);
                }
                translated[k] = true;
            } else if (key == "stage") {
                // "stage NAME SHADER.spv TILE_ROWS ARG:BYTE,...": a network stage, its tile height and its weight pointers
                std::string name, shader, pointerList; uint32_t rows = 8;
                f >> name >> shader >> rows >> pointerList;
                unsigned stage = 0;
                while (stage < R3Network && name != R3Names[stage]) ++stage;
                if (stage >= R3Network || seen[stage] || (rows != 4 && rows != 8))
                    throw std::runtime_error("rr.txt: unknown, repeated or invalid stage: " + line);
                seen[stage] = true;
                tileRows[stage] = rows;
                for (const auto& item : split(pointerList)) {
                    const auto colon = item.find(':');
                    weightPointers[stage].emplace_back(static_cast<uint32_t>(std::stoul(item.substr(0, colon))), std::stoull(item.substr(colon + 1)));
                }
                traceStage(directory + "/" + shader);
                stages[stage].create(d.handle, directory + "/" + shader, {b}, 32, cache, 0, specialization);
            } else if (key == "weights") {
                std::string file; f >> file;
                auto bytes = read(directory + "/" + file);
                weights.create(d, bytes.size(), usage, false, true);
                weightsPending = std::move(bytes);
                weightsSeen = true;
            }
        }
        for (bool s0 : seen) if (!s0) throw std::runtime_error("rr.txt: incomplete Ray Reconstruction CNN model");
        if (!translated[0] || !translated[1] || !weightsSeen) throw std::runtime_error("rr.txt: incomplete Ray Reconstruction CNN model");
        traceStage(directory + "/exposure_rr0.spv");
        exposure[0].create(d.handle, directory + "/exposure_rr0.spv", {b, t, b}, 64, cache);
        traceStage(directory + "/exposure_rr1.spv");
        exposure[1].create(d.handle, directory + "/exposure_rr1.spv", {b, b, b, s, s}, 64, cache);
    }
    void recordUpload(VkCommandBuffer cb) {
        recordContents(cb, consts, constsPending);
        recordContents(cb, weights, weightsPending);
    }
};

struct R3Engine {
    RStageRecorder* recorder = nullptr;
    const Device& d;
    R3Model& model;
    uint32_t rw, rh, ow, oh, w0, h0;
    uint64_t unit;                    // W0 * H0 / 2 bytes
    Buffer arena, paramAll, expP, expSums, expState;
    std::array<VkDeviceSize, R3Blocks> paramAt{};
    VkDeviceSize paramBytes = 0;
    std::array<std::vector<uint8_t>, R3Blocks> blocks;
    ExposureParams exposure{};
    // ping-pong: the input kernel's render-size state (its surface 160, read back as texture 80 and by the output
    // kernel as texture 264) and the output-size history (the output kernel's surface 200, read back as its texture 192
    // and the input kernel's 352). e: this frame's and the previous frame's exposure (1x1), as presets D and E
    std::array<Tex, 2> state, hist;
    std::array<Tex, 2> e;
    Tex out, zero;
    bool outNew = false;
    std::map<std::string, Tex> spare;
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet exp1Set = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, R3Network> network{};
    struct Bindings {
        RFrameImages images{};
        FrameOutput finalOutput{};
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> input{}, output{};
        VkDescriptorSet exposure0 = VK_NULL_HANDLE;
    };
    std::vector<Bindings> bindings;
    const Bindings* bound = nullptr;
    float prevJitter[2] = {0, 0};
    uint32_t frame = 0;

    uint32_t lw(int level) const { return w0 >> level; }
    uint32_t lh(int level) const { return h0 >> level; }
    uint64_t at(uint32_t units) const { return unit * units; }
    uint64_t address(const Buffer& buffer) const {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buffer.handle;
        return vkGetBufferDeviceAddress(d.handle, &info);
    }
    struct Resource { VkImageView view = VK_NULL_HANDLE; bool linear = false; bool game = false; };
    // what the input (k = 0) or output (k = 1) kernel's texture or surface at an argument offset is, for frame parity
    // `par`; null view: NGX leaves it unbound. Presets D and E's encoder 0 and output wiring (runtime_rr.inc) at the
    // CNN kernels' offsets; filter modes are NGX's texture objects'
    Resource resolve(int k, bool surface, uint32_t offset, uint32_t par, const RFrameImages& in, VkImageView output) const {
        auto T = [](const Tex& t, bool lin = false) { return Resource{t.view, lin, false}; };
        auto G = [](VkImageView v, bool lin = false) { return Resource{v, lin, true}; };
        if (k == 0) {
            if (!surface) switch (offset) {
                case 16: return G(in.color, true);   case 24: return G(in.motion);       case 40: return G(in.normals);
                case 48: return G(in.roughness);     case 56: return T(e[0]);             case 64: return T(e[1]);
                case 80: return T(state[par ^ 1], true); case 88: return G(in.diffuseAlbedo); case 96: return G(in.specularAlbedo);
                case 104: return in.specularHitDistance ? G(in.specularHitDistance) : Resource{};
                case 152: return in.disocclusionMask ? G(in.disocclusionMask) : Resource{};
                case 352: return T(hist[par ^ 1], true); case 368: return G(in.depth);
            } else if (offset == 160) return T(state[par]);
        } else {
            if (!surface) switch (offset) {
                case 160: return G(in.color, true);  case 168: return G(in.depth);       case 176: return T(e[0]);
                case 184: return T(e[1]);            case 192: return T(hist[par ^ 1], true);
                case 248: return G(in.diffuseAlbedo); case 256: return G(in.specularAlbedo); case 264: return T(state[par]);
            } else switch (offset) {
                case 200: return T(hist[par]); case 216: return Resource{output, false, false};
            }
        }
        return {};
    }

    R3Engine(const Device& device, R3Model& m, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight)
        : d(device), model(m), rw(renderWidth), rh(renderHeight), ow(outputWidth), oh(outputHeight) {
        w0 = (rw + 31) & ~31u; h0 = (rh + 31) & ~31u;
        unit = static_cast<uint64_t>(w0) * h0 / 2;
        const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (at(R3ArenaUnits) > 0xffffffffull) throw std::runtime_error("Ray Reconstruction: the render size is too large for the arena");
        // the network stages reach the arena and the weights by address
        arena.create(d, at(R3ArenaUnits), storage, std::getenv("D4R_ENGINE_RR_ARENA") != nullptr, true);
        const uint32_t sizes[R3Blocks] = {R3NetworkBytes, R3NetworkBytes, R3NetworkBytes, R3NetworkBytes, R3NetworkBytes,
                                          R3NetworkBytes, R3NetworkBytes, R3NetworkBytes, R3NetworkBytes, R3NetworkBytes,
                                          R3NetworkBytes, R3InputBytes, R3OutputBytes};
        for (unsigned i = 0; i < R3Blocks; ++i) {
            blocks[i].assign(sizes[i], 0);
            paramAt[i] = paramBytes;
            paramBytes += (sizes[i] + 255) & ~VkDeviceSize(255);
        }
        paramAll.create(d, paramBytes, storage, false, true);   // a stage recorder reads the blocks by address
        expP.create(d, sizeof(ExposureParams), storage);
        const uint32_t gx = exposureGrid(rw, ow), gy = exposureGrid(rh, oh);
        exposure.blocksX = (gx + 31) / 32;
        exposure.blocks = exposure.blocksX * ((gy + 31) / 32);
        expSums.create(d, exposure.blocks * 4, storage);
        expState.create(d, 8, storage);
        const VkFormat half4 = VK_FORMAT_R16G16B16A16_SFLOAT;
        for (int i = 0; i < 2; ++i) {
            state[i].create(d, rw, rh, half4);
            hist[i].create(d, ow, oh, half4);
            e[i].create(d, 1, 1, VK_FORMAT_R16_SFLOAT);
        }
        zero.create(d, 1, 1, half4);
        for (int k = 0; k < 2; ++k)
            for (const auto& [offset, format] : model.surfaces[k])
                if (!spare.count(format)) spare[format].create(d, 1, 1, R2Engine::surfaceFormat(format));
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = 0;
        VKCHECK(vkCreateSampler(d.handle, &si, nullptr, &nearest));
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        VKCHECK(vkCreateSampler(d.handle, &si, nullptr, &linear));
        // the network's descriptor sets (their parameter blocks only) and the exposure's second pass
        VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, R3Network + 3}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = R3Network + 1; pi.poolSizeCount = 2; pi.pPoolSizes = poolSizes;
        VKCHECK(vkCreateDescriptorPool(d.handle, &pi, nullptr, &pool));
        for (unsigned stage = 0; stage < R3Network; ++stage) {
            network[stage] = descriptorSet(d.handle, pool, model.stages[stage].descriptors);
            bindRange(d.handle, network[stage], 0, paramAll, paramAt[stage]);
        }
        exp1Set = descriptorSet(d.handle, pool, model.exposure[1].descriptors);
        bindBuffer(d.handle, exp1Set, 0, expP);
        bindBuffer(d.handle, exp1Set, 1, expSums);
        bindBuffer(d.handle, exp1Set, 2, expState);
        bindImage(d.handle, exp1Set, 3, e[0].view);
        bindImage(d.handle, exp1Set, 4, e[1].view);
        // the network's blocks do not change from frame to frame: tensor descriptors (height, width, channels, row
        // stride in elements) and constants as NGX passes them, the arena's and the weights' addresses
        const uint64_t arenaAt = address(arena), weightsAt = address(model.weights);
        for (unsigned stage = 0; stage < R3Network; ++stage) {
            auto& p = blocks[stage];
            const int l = R3Level[stage];
            const uint32_t cin = R3In[stage], c = R3Out[stage];
            auto tensor = [&](uint32_t offset, int level, uint32_t channels) {
                put32(p, offset, lh(level)); put32(p, offset + 4, lw(level));
                put32(p, offset + 8, channels); put32(p, offset + 12, lw(level) * channels);
            };
            if (stage <= R3Enc4) {
                tensor(8, l, cin);   // the input (Enc0: the input kernel's 16 halves per pixel at level 0)
                put32(p, 32, 16); put32(p, 48, 16); put32(p, 64, 16); put32(p, 80, c); put32(p, 96, c); put32(p, 112, 16);
                tensor(128, l, c);   // the skip tensor
                put32(p, 144, 1);
                tensor(160, l + 1, c);   // its max-pool
                put32(p, 176, 1);
                put32(p, 208, lh(l)); put32(p, 212, lw(l)); put32(p, 216, 4);   // the level's images
            } else {
                const bool pdl = stage == R3Pdl, ups = stage != R3Dec4;
                tensor(8, ups ? l + 1 : l, cin);   // the level below's features (Dec4: the bottleneck, level 5)
                put32(p, 24, 1);
                if (ups) tensor(40, l, R3Out[R3Enc4 - (stage - R3Dec4) + 1]);   // the encoder's skip tensor
                put32(p, 56, 1);
                for (uint32_t o : {72u, 88u, 104u}) put32(p, o, 16);
                if (!pdl) put32(p, 120, 16);
                for (uint32_t o : {136u, 152u, 168u}) put32(p, o, c);
                if (!pdl) put32(p, 184, 16);
                tensor(pdl ? 224 : 200, l, c);   // the level's features
                put32(p, 272, lh(l)); put32(p, 276, lw(l)); put32(p, 280, 4);
            }
            for (const auto& [offset, units] : R3Arena[stage]) put64(p, offset, arenaAt + at(units));
            for (const auto& [offset, byte] : model.weightPointers[stage]) put64(p, offset, weightsAt + byte);
        }
        // the input and output kernels' fixed fields (presets D and E's encoder 0 block 144 bytes earlier; the output
        // block 16 - 32 bytes earlier than theirs) and their arena offsets
        auto& i = blocks[R3Input];
        put32(i, 0, rw); put32(i, 4, rh); put32(i, 8, w0); put32(i, 12, h0);
        const float fr = static_cast<float>(rw), fh = static_cast<float>(rh), fow = static_cast<float>(ow), foh = static_cast<float>(oh);
        putF(i, 184, fr / fow); putF(i, 188, fh / foh); putF(i, 192, fow / fr); putF(i, 196, foh / fh);
        putF(i, 216, 1.0f);
        put32(i, 288, rw - 1); put32(i, 292, rh - 1);
        put32(i, 328, ow); put32(i, 332, oh);
        put32(i, 388, 2);
        for (const auto& [offset, units] : R3InputPointers) put64(i, offset, at(units));
        auto& o = blocks[R3Output];
        put32(o, 0, ow); put32(o, 4, oh); put32(o, 8, ow); put32(o, 12, oh); put32(o, 16, rw); put32(o, 20, rh);
        putF(o, 24, fr / fow); putF(o, 28, fh / foh); putF(o, 32, fow / fr); putF(o, 36, foh / fh);
        put32(o, 56, w0); put32(o, 60, h0);
        put32(o, 124, rw - 1); put32(o, 128, rh - 1);
        putF(o, 132, 1.0f);
        putF(o, 404, 0.25f);
        for (const auto& [offset, units] : R3OutputPointers) put64(o, offset, at(units));
    }
    ~R3Engine() {
        for (auto& b : bindings) vkDestroyDescriptorPool(d.handle, b.pool, nullptr);
        if (pool) vkDestroyDescriptorPool(d.handle, pool, nullptr);
        if (nearest) vkDestroySampler(d.handle, nearest, nullptr);
        if (linear) vkDestroySampler(d.handle, linear, nullptr);
    }
    std::vector<REngine::DebugImage> debugImages() const {
        std::vector<REngine::DebugImage> list;
        const VkFormat half4 = VK_FORMAT_R16G16B16A16_SFLOAT;
        for (int i = 0; i < 2; ++i) {
            const std::string n = std::to_string(i);
            list.push_back({"state" + n, state[i].handle, half4, rw, rh});
            list.push_back({"hist" + n, hist[i].handle, half4, ow, oh});
            list.push_back({"e" + n, e[i].handle, VK_FORMAT_R16_SFLOAT, 1, 1});
        }
        if (out.handle) list.push_back({"out", out.handle, half4, ow, oh});
        return list;
    }
    void recordInitialize(VkCommandBuffer cb) {
        std::vector<VkImage> images{zero.handle};
        for (auto* group : {&state, &hist, &e}) for (auto& t : *group) images.push_back(t.handle);
        for (auto& [format, t] : spare) images.push_back(t.handle);
        VkClearColorValue clear{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (VkImage image : images) {
            imageBarrier(cb, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        }
        for (auto* b : {&arena, &expState}) vkCmdFillBuffer(cb, b->handle, 0, VK_WHOLE_SIZE, 0);
        memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        frame = 0;
    }
    void setFrame(const RFrameImages& images, const RFrameSettings& settings, FrameOutput finalOutput) {
        if (!images.color || !images.motion || !images.depth || !images.normals || !images.roughness || !images.diffuseAlbedo || !images.specularAlbedo)
            throw std::runtime_error("Ray Reconstruction needs colour, motion, depth and the four guides");
        if (finalOutput.view && finalOutput.format != VK_FORMAT_R16G16B16A16_SFLOAT)
            throw std::runtime_error("Ray Reconstruction writes RGBA16F output only");
        const bool rst = settings.reset || frame == 0;
        // debugging: D4R_ENGINE_RR_ARENA=DIR keeps the arena host-visible and writes it as DIR/F-arena.bin before frame
        // F + 1 (the previous frame's, F counting from 1)
        if (const char* dir = std::getenv("D4R_ENGINE_RR_ARENA"); dir && frame > 0 && arena.mapped)
            std::ofstream(std::string(dir) + "/" + std::to_string(frame) + "-arena.bin", std::ios::binary)
                .write(static_cast<const char*>(arena.mapped), static_cast<std::streamsize>(at(R3ArenaUnits)));
        const float fr = static_cast<float>(rw), fh = static_cast<float>(rh), fow = static_cast<float>(ow), foh = static_cast<float>(oh);
        const float pj[2] = {rst ? 0.0f : prevJitter[0], rst ? 0.0f : prevJitter[1]};
        // the input kernel: presets D and E's encoder 0 fields (rr_frame_parameters.h) 144 bytes earlier
        auto& p = blocks[R3Input];
        putF(p, 168, pj[0]); putF(p, 172, pj[1]); putF(p, 176, settings.jitter[0]); putF(p, 180, settings.jitter[1]);
        putF(p, 200, settings.motionScale[0] * (fow / fr)); putF(p, 204, settings.motionScale[1] * (foh / fh));
        put32(p, 220, rst ? 256u : 0u);
        p[222] = settings.roughnessPacked ? 1u : 0u;
        put32(p, 384, settings.depthInverted ? 1u : 0u);
        p[388] = 2u;
        p[392] = images.specularHitDistance ? 1u : 0u;
        p[395] = images.disocclusionMask ? 1u : 0u;
        p[836] = settings.hardwareDepth ? 1u : 0u;
        Mat4 view, clip;
        for (int k = 0; k < 16; ++k) { view[k] = settings.worldToView[k]; clip[k] = settings.viewToClip[k]; }
        view[12] = view[13] = view[14] = 0.0;
        const Mat4 rc = mul(view, clip), inv = inverse(rc);
        const Mat4 toNdc{2.0 / rw, 0, 0, 0, 0, -2.0 / rh, 0, 0, 0, 0, 1, 0, -1, 1, 0, 1};
        const Mat4 toPixels{rw / 2.0, 0, 0, 0, 0, -(rh / 2.0), 0, 0, 0, 0, 1, 0, rw / 2.0, rh / 2.0, 0, 1};
        putMat(p, 452, rc); putMat(p, 516, clip); putMat(p, 580, clip); putMat(p, 644, inv);
        putMat(p, 708, mul(toNdc, inv)); putMat(p, 772, mul(rc, toPixels));
        putF(p, 612, settings.jitter[0] * (2.0f / fr)); putF(p, 616, -settings.jitter[1] * (2.0f / fh));
        // the output kernel: presets D and E's fields 16 (up to byte 160) or 32 bytes earlier
        auto& o = blocks[R3Output];
        putF(o, 40, settings.jitter[0]); putF(o, 44, settings.jitter[1]);
        putF(o, 136, settings.frameTimeMs);
        o[144] = images.specularHitDistance ? 1u : 0u;
        putF(o, 304, settings.motionScale[0] * (fow / fr)); putF(o, 308, settings.motionScale[1] * (foh / fh));
        put32(o, 408, (rst ? 1u : 0u) | (settings.depthInverted ? 0x10000u : 0u) | (settings.hardwareDepth ? 0x1000000u : 0u));
        // the exposure: NGX's MODE 1 measurement on 1 / pre-exposure scaled colour, as presets D and E
        const uint32_t gx = exposureGrid(rw, ow), gy = exposureGrid(rh, oh);
        exposure.origin[0] = exposure.origin[1] = 0;
        exposure.size[0] = static_cast<int32_t>(rw); exposure.size[1] = static_cast<int32_t>(rh);
        exposure.sampleSize[0] = static_cast<int32_t>(gx); exposure.sampleSize[1] = static_cast<int32_t>(gy);
        exposure.invCount = 1.0f / static_cast<float>(gx * gy);
        exposure.scale = 1.0f / (settings.preExposure > 0.0f ? settings.preExposure : 1.0f);
        exposure.key = 0.23547087609767914f;
        exposure.flags = 1u | (frame == 0 ? 2u : 0u);
        if (!finalOutput.view && !out.handle) {
            out.create(d, ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT);
            outNew = true;
        }
        const VkImageView output = finalOutput.view ? finalOutput.view : out.view;
        // texture and surface handles: 1 where something is bound, 0 where NGX leaves the argument null
        for (int k = 0; k < 2; ++k) {
            auto& block = blocks[k ? R3Output : R3Input];
            for (uint32_t offset : model.textures[k]) put64(block, offset, resolve(k, false, offset, 0, images, output).view ? 1u : 0u);
            for (const auto& [offset, format] : model.surfaces[k]) put64(block, offset, resolve(k, true, offset, 0, images, output).view ? 1u : 0u);
        }
        auto same = [&](const Bindings& b) { return !std::memcmp(&b.images, &images, sizeof(images)) && b.finalOutput.view == finalOutput.view; };
        auto found = std::find_if(bindings.begin(), bindings.end(), same);
        if (found == bindings.end()) {
            if (bindings.size() >= kRRMaxBindings) throw std::runtime_error("too many distinct Ray Reconstruction input view sets");
            Bindings b;
            b.images = images; b.finalOutput = finalOutput;
            const uint32_t textures = static_cast<uint32_t>(model.textures[0].size() + model.textures[1].size());
            const uint32_t surfaces = static_cast<uint32_t>(model.surfaces[0].size() + model.surfaces[1].size());
            VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 2 * 3 + 2},
                                                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * textures + 1},
                                                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * surfaces}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = 5; pi.poolSizeCount = 3; pi.pPoolSizes = poolSizes;
            VKCHECK(vkCreateDescriptorPool(d.handle, &pi, nullptr, &b.pool));
            for (int k = 0; k < 2; ++k)
                for (uint32_t par = 0; par < 2; ++par) {
                    VkDescriptorSet set = (k ? b.output : b.input)[par] = descriptorSet(d.handle, b.pool, (k ? model.output : model.input).descriptors);
                    bindRange(d.handle, set, 0, paramAll, paramAt[k ? R3Output : R3Input]);
                    bindBuffer(d.handle, set, 1, k ? arena : model.consts);   // the output kernel has no constants
                    bindBuffer(d.handle, set, 2, arena);
                    uint32_t binding = 3;
                    for (uint32_t offset : model.textures[k]) {
                        const auto res = resolve(k, false, offset, par, images, output);
                        bindImage(d.handle, set, binding++, res.view ? res.view : zero.view, res.linear ? linear : nearest,
                                  res.game ? images.layout : VK_IMAGE_LAYOUT_GENERAL);
                    }
                    for (const auto& [offset, format] : model.surfaces[k]) {
                        const auto res = resolve(k, true, offset, par, images, output);
                        bindImage(d.handle, set, binding++, res.view ? res.view : spare.at(format).view);
                    }
                }
            b.exposure0 = descriptorSet(d.handle, b.pool, model.exposure[0].descriptors);
            bindBuffer(d.handle, b.exposure0, 0, expP);
            bindImage(d.handle, b.exposure0, 1, images.color, nearest, images.layout);
            bindBuffer(d.handle, b.exposure0, 2, expSums);
            bindings.push_back(std::move(b));
            found = bindings.end() - 1;
        }
        bound = &*found;
        // debugging: D4R_ENGINE_RR_DUMP=DIR writes every frame's parameter blocks as DIR/F-STAGE.bin (F counts from 1)
        if (const char* dump = std::getenv("D4R_ENGINE_RR_DUMP")) {
            auto write = [&](const std::string& name, const std::vector<uint8_t>& block) {
                std::ofstream(std::string(dump) + "/" + std::to_string(frame + 1) + "-" + name + ".bin", std::ios::binary)
                    .write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(block.size()));
            };
            for (unsigned stage = 0; stage < R3Network; ++stage) write(R3Names[stage], blocks[stage]);
            write("input", blocks[R3Input]); write("output", blocks[R3Output]);
        }
        prevJitter[0] = settings.jitter[0]; prevJitter[1] = settings.jitter[1];
    }
    void recordFrame(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) {
        if (!bound) throw std::runtime_error("REngine::setFrame was not called");
        const auto& b = *bound;
        const uint32_t cur = frame & 1u;
        auto stamp = [&](uint32_t n) { if (stamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, stamps, first + n); };
        // D4R_ENGINE_RR_STAGE_STAMPS=1: one more stamp after every network stage (queries first + 4 ...)
        static const bool stageStamps = std::getenv("D4R_ENGINE_RR_STAGE_STAMPS") != nullptr;
        auto step = [&] {
            memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        };
        if (outNew) {   // created by setFrame: starts from zero, like the rest of the temporal state
            clearNewImage(cb, out.handle);
            memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            outNew = false;
        }
        stamp(0);
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        std::vector<uint8_t> all(paramBytes);
        for (unsigned i = 0; i < R3Blocks; ++i) std::memcpy(all.data() + paramAt[i], blocks[i].data(), blocks[i].size());
        for (VkDeviceSize at0 = 0; at0 < paramBytes; at0 += 65536)   // vkCmdUpdateBuffer takes at most 65536 bytes
            vkCmdUpdateBuffer(cb, paramAll.handle, at0, std::min<VkDeviceSize>(65536, paramBytes - at0), all.data() + at0);
        vkCmdUpdateBuffer(cb, expP.handle, 0, sizeof(ExposureParams), &exposure);
        memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        dispatch(cb, model.exposure[0], b.exposure0, exposure.blocksX, exposure.blocks / exposure.blocksX);
        step();
        dispatch(cb, model.exposure[1], exp1Set, 1, 1);
        step();
        dispatch(cb, model.input, b.input[cur], (w0 + 15) / 16, (h0 + 15) / 16);
        step();
        stamp(1);
        for (unsigned stage = 0; stage < R3Network; ++stage) {
            const int l = R3Level[stage];
            const uint32_t gx = (lw(l) + 15) / 16, gy = (lh(l) + model.tileRows[stage] - 1) / model.tileRows[stage];
            if (!recorder || !recorder->recordStage(cb, stage, gx, gy, 0)) dispatch(cb, model.stages[stage], network[stage], gx, gy);
            step();
            if (stageStamps) stamp(4 + stage);
        }
        stamp(2);
        dispatch(cb, model.output, b.output[cur], (ow + 15) / 16, (oh + 15) / 16);
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
        stamp(3);
        ++frame;
    }
};
} // namespace
