// Ray Reconstruction preset F (DLSS 4.5, the rr2_* kernels of nvngx_dlssd.dll 310.9): the runtime around the stage
// shaders of a "D4RR0002" model (engine/compile_rr2.py). Included by runtime_rr.inc after its helpers.
//
// NGX's frame: the exposure, encoder 0 to 5, five decoders of two kernels each, then the output stage, which filters
// an intermediate image of 1.5 times the output size in tiles (post) and reduces each tile to the output (down), and
// a quarter-size copy of the half-size history for the next frame's encoder 0 (pooled). Every stage's parameter
// block is NVIDIA's argument block: the model carries one captured block per kernel and this file fills the fields
// that depend on the frame or the geometry (offsets as observed on harness captures at three sizes, 2026-10-13).
// The network's activations live in one arena whose regions are NGX's, in units of W0 * H0 / 32 bytes (they scale
// with the render size); the output passes' two tile regions follow it. NGX runs post and down concurrently and
// has them wait on progress counters; here they run in order and the counter pointers stay null, which the kernels
// test.
namespace {
// R2DownP: the downsample kernel NGX runs instead of R2Down when the game passes DLSSD.ColorBeforeParticles
// (rr2_downsample_kernel_static_hdr); optional in a model
// R2Gather: encoder 0's feature gather as a pass of its own, before R2Enc0 (engine/rr/gen_rr2_enc0.py --split); optional
constexpr unsigned R2Network = 16, R2Post = 16, R2Down = 17, R2Pooled = 18, R2DownP = 19, R2Gather = 20, R2Stages = 21;
constexpr unsigned R2Enc0 = 0, R2Enc5 = 5, R2Dec0B = 15;
constexpr const char* R2Names[R2Stages] = {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec4_0", "dec4_1", "dec3_0", "dec3_1",
                                           "dec2_0", "dec2_1", "dec1_0", "dec1_1", "dec0_0", "dec0_1", "post", "down", "pooled", "downp", "enc0g"};
constexpr int R2Level[R2Network] = {0, 1, 2, 3, 4, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0};
constexpr uint32_t R2ArenaUnits = 3152;
// The window shift of a network stage (x | y << 16, argument 80): it alternates every frame
uint32_t r2Shift(unsigned stage, uint32_t frame) {
    const bool even = (frame & 1u) == 0;
    if (stage < R2Enc5) return even ? 0x00060002u : 0x00020006u;
    const bool first = stage == R2Enc5 || ((stage - R2Enc5) & 1u) == 1;   // enc5 and the decoders' kernel 0
    return (even == first) ? 0x00000004u : 0x00040000u;
}
struct R2Stage {
    Pipeline pipeline;
    Buffer weights;
    std::vector<char> pending;   // the weights until recordUpload puts them in the command buffer
    std::vector<uint8_t> args;
    std::vector<uint32_t> textures, buffers;
    std::vector<std::pair<uint32_t, std::string>> surfaces;
    std::vector<std::pair<uint32_t, double>> pointers;   // argument offset, arena offset in units
    uint32_t group = 1;    // "xN": a workgroup holds N windows along x (the stages generated around translated code)
    uint32_t record = 0;   // R2Gather, "rN": 32-bit words per window in the arena
    bool ranges = false;   // R2Gather, "s3": the records' rows lie in three ranges (six words end enc0's block; else one address)
};
struct R2Model {
    std::array<R2Stage, R2Stages> stages;
    Buffer aux;                   // encoder 0's second table (argument 104)
    std::vector<char> auxPending;
    std::array<Pipeline, 2> exposure;
    Pipeline exposureOne;         // both exposure passes in one (exposure_f2.spv, 64-bit atomics); else null
    bool particles = false;       // the model has R2DownP
    bool gather = false;          // the model has R2Gather
    void load(const Device& d, const std::string& directory, std::istringstream& lines, VkPipelineCache cache, bool int64Atomics,
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
        auto blob = [&](Buffer& buffer, std::vector<char>& pending, std::vector<char> bytes) {
            buffer.create(d, bytes.size(), usage);
            pending = std::move(bytes);
        };
        std::array<bool, R2Stages> seen{};
        std::string line;
        while (std::getline(lines, line)) {
            std::istringstream f(line);
            std::string key; f >> key;
            if (key != "stage") continue;
            // "stage NAME SHADER.spv SUBGROUP WEIGHTS|- ARGS TEXTURES SURFACES BUFFERS POINTERS": the lists are
            // comma-separated argument offsets ("-": none), surfaces OFFSET:FORMAT, pointers OFFSET:UNITS; an optional "xN"
            std::string name, shader, weightFile, argsFile, textureList, surfaceList, bufferList, pointerList, option; uint32_t subgroup = 32;
            f >> name >> shader >> subgroup >> weightFile >> argsFile >> textureList >> surfaceList >> bufferList >> pointerList;
            unsigned stage = 0;
            while (stage < R2Stages && name != R2Names[stage]) ++stage;
            if (stage >= R2Stages || seen[stage] || pointerList.empty())
                throw std::runtime_error("rr.txt: unknown, repeated or invalid stage: " + line);
            seen[stage] = true;
            auto& st = stages[stage];
            while (f >> option) {
                if (option.size() == 2 && option[0] == 'x' && option[1] >= '1' && option[1] <= '4') st.group = static_cast<uint32_t>(option[1] - '0');
                if (option.size() > 1 && option[0] == 'r') st.record = static_cast<uint32_t>(std::stoul(option.substr(1)));
                if (option == "s3") st.ranges = true;
            }
            for (const auto& item : split(textureList)) st.textures.push_back(static_cast<uint32_t>(std::stoul(item)));
            for (const auto& item : split(bufferList)) st.buffers.push_back(static_cast<uint32_t>(std::stoul(item)));
            for (const auto& item : split(surfaceList)) {
                const auto colon = item.find(':');
                st.surfaces.emplace_back(static_cast<uint32_t>(std::stoul(item.substr(0, colon))), item.substr(colon + 1));
            }
            for (const auto& item : split(pointerList)) {
                const auto colon = item.find(':');
                st.pointers.emplace_back(static_cast<uint32_t>(std::stoul(item.substr(0, colon))), std::stod(item.substr(colon + 1)));
            }
            std::vector<VkDescriptorType> types{b, b, b};
            types.insert(types.end(), st.textures.size(), t);
            types.insert(types.end(), st.surfaces.size(), s);
            types.insert(types.end(), st.buffers.size(), b);
            st.pipeline.create(d.handle, directory + "/" + shader, types, subgroup, cache, 0, specialization);
            { const auto bytes = read(directory + "/" + argsFile); st.args.assign(bytes.begin(), bytes.end()); }
            blob(st.weights, st.pending, weightFile == "-" ? std::vector<char>(16) : read(directory + "/" + weightFile));
        }
        for (unsigned stage = 0; stage < R2DownP; ++stage)
            if (!seen[stage]) throw std::runtime_error("rr.txt: incomplete Ray Reconstruction preset F model");
        particles = seen[R2DownP];
        gather = seen[R2Gather] && stages[R2Gather].record != 0 && stages[R2Enc0].args.size() >= 8;
        for (unsigned stage : {R2DownP, R2Gather})
            if (!seen[stage]) blob(stages[stage].weights, stages[stage].pending, std::vector<char>(16));
        blob(aux, auxPending, read(directory + "/enc0-aux.bin"));
        exposure[0].create(d.handle, directory + "/exposure_f0.spv", {b, t, b}, 64, cache);
        exposure[1].create(d.handle, directory + "/exposure_f1.spv", {b, b, b, t, s}, 64, cache);
        if (int64Atomics && exists(directory + "/exposure_f2.spv"))
            exposureOne.create(d.handle, directory + "/exposure_f2.spv", {b, t, b, b, t, s}, 64, cache);
    }
    void recordUpload(VkCommandBuffer cb) {   // the weights into the command buffer (recordContents)
        for (auto& st : stages) recordContents(cb, st.weights, st.pending);
        recordContents(cb, aux, auxPending);
    }
};
struct R2ExposureParams { int32_t last[2]; uint32_t blocksX, blocks; float invCount; uint32_t reset; };

struct R2Engine {
    const Device& d;
    R2Model& model;
    uint32_t rw, rh, ow, oh, w0, h0;
    uint32_t iw, ih, cw, ch, qw, qh;          // the intermediate (1.5x output), its half and quarter
    uint32_t nx, ny, maxW, maxH;              // output tiles and the largest tile rectangle (with its overlap)
    uint64_t unit, tileB;                     // arena unit; where a tile slot's second region starts in it
    uint64_t tileBytes = 0;                   // a tile slot's bytes
    std::vector<uint64_t> slotAt;             // each tile slot's place in the arena buffer
    // Tile slots (a pair of regions each) and tiles per group. The post passes of a group have a slot each and run
    // without waiting for one another, as do the down passes that read them; a group's down passes also overlap the
    // next group's post passes, which write the other half of the slots (NGX overlaps one post with one down the same
    // way, on two slots). Serialising all 32 passes of a 4K frame cost 0.17 ms in the gaps (RX 9070 XT, 2026-10-13).
    static constexpr uint32_t R2TileSlots = 8;
    uint32_t tileGroup = 1;
    // the gather pass's records (R2Gather): byte address of range 0, its rows of windows, range 1, its rows, range 2
    std::array<uint32_t, 5> gatherWords{};
    struct Rect { uint32_t x0, y0, x1, y1; };
    std::vector<Rect> src, dst;               // per tile: intermediate rectangle (with overlap), output rectangle
    std::vector<VkDeviceSize> downPAt;
    std::vector<std::vector<uint8_t>> downPBlocks;
    // the particle layer's output-size history (R2DownP's surface 264, read back as texture 256): created on first use
    Tex pb[2];
    bool pbNew = false;
    // the engine's own output image: only for frames that do not write the caller's texture (created on first use)
    bool outNew = false;
    Buffer arena, values, sums, dummy;
    // every parameter block of the frame in one buffer (one upload per frame): the network stages', per tile the two
    // output passes', the pooling pass's and the exposure's, each at a 256-byte boundary
    Buffer paramAll;
    std::array<VkDeviceSize, R2Network> paramAt{};
    std::vector<VkDeviceSize> postAt, downAt;
    VkDeviceSize pooledAt = 0, expAt = 0, paramBytes = 0;
    std::array<std::vector<uint8_t>, R2Network> blocks;
    std::vector<std::vector<uint8_t>> postBlocks, downBlocks;
    std::vector<uint8_t> pooledBlock;
    R2ExposureParams exposure{};
    // enc0's feature (surface 368; next frame's texture 264), the intermediate history (post's surface 96 / texture
    // 48) and the exposure, ping-pong; the rest is written and read within a frame or carried as is:
    // mv[0] / mv[1]: enc0's surfaces 376 / 384 (post's textures 64 / 72; capFXmot); rec: dec0_1's 176 (enc0's 528 next frame);
    // ka / kb / kc: dec0_1's 184 / 192 / 200 (post's 24 / 32 / 40); half: post's 120 (enc0's 536, pooled's input);
    // quarter: pooled's output (enc0's 544)
    std::array<Tex, 2> feat, hist, e, mv;
    Tex rec, ka, kb, kc, half, quarter, out, zero;
    std::map<std::string, Tex> spare;
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    struct Bindings {
        RFrameImages images{};
        FrameOutput finalOutput{};
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::array<std::array<VkDescriptorSet, 2>, R2Network> network{};
        std::array<VkDescriptorSet, 2> gather{};
        std::vector<std::array<VkDescriptorSet, 2>> post;
        std::vector<VkDescriptorSet> down;
        std::vector<std::array<VkDescriptorSet, 2>> downP;
        VkDescriptorSet pooled = VK_NULL_HANDLE, exposure0 = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 2> exposure1{}, exposureOne{};
    };
    std::vector<Bindings> bindings;
    const Bindings* bound = nullptr;
    float prevJitter[2] = {0, 0};
    uint32_t frame = 0;
    bool reset = true;

    static VkFormat surfaceFormat(const std::string& f) {
        if (f == "rgba16f") return VK_FORMAT_R16G16B16A16_SFLOAT;
        if (f == "rgba32f") return VK_FORMAT_R32G32B32A32_SFLOAT;
        if (f == "rgb10_a2") return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        if (f == "rg16f") return VK_FORMAT_R16G16_SFLOAT;
        if (f == "r16f") return VK_FORMAT_R16_SFLOAT;
        if (f == "rgba8_snorm") return VK_FORMAT_R8G8B8A8_SNORM;
        if (f == "rgba8") return VK_FORMAT_R8G8B8A8_UNORM;
        throw std::runtime_error("Ray Reconstruction stage surface format " + f + " is not implemented");
    }
    // the pooled histories: the model's format of the post pass's surface 120 (UNORM 10:10:10:2, or RGBA32F holding
    // the same q / 1023 in models built with D4R_RR2_POOLED=rgba32f)
    VkFormat pooledFormat() const {
        for (const auto& [offset, format] : model.stages[R2Post].surfaces)
            if (offset == 120) return surfaceFormat(format);
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    // tile i's slot: groups alternate between the two halves of the slots (one group when every tile has a slot)
    uint32_t tileSlot(size_t i) const {
        const uint32_t g = static_cast<uint32_t>(i) / tileGroup, k = static_cast<uint32_t>(i) % tileGroup;
        return (g & 1u) * tileGroup + k;
    }
    uint32_t lw(int level) const { return w0 >> level; }
    uint32_t lh(int level) const { return h0 >> level; }
    struct Resource { VkImageView view = VK_NULL_HANDLE; bool linear = false; bool game = false; };
    // what a stage's texture or surface at an argument offset is, for frame parity `par` (the capture's wiring;
    // filter modes are NGX's texture objects'); null view: NGX leaves it unbound
    Resource resolve(unsigned stage, bool surface, uint32_t offset, uint32_t par, const RFrameImages& in, VkImageView output) const {
        auto T = [](const Tex& t, bool lin = false) { return Resource{t.view, lin, false}; };
        auto G = [](VkImageView v, bool lin = false) { return Resource{v, lin, true}; };
        if (stage == R2Enc0) {
            if (!surface) switch (offset) {
                case 144: return G(in.color, true);    case 160: return G(in.diffuseAlbedo); case 176: return G(in.specularAlbedo);
                case 192: return G(in.motion);         case 224: return G(in.normals);       case 240: return G(in.roughness);
                case 264: return T(feat[par ^ 1], true); case 528: return T(rec, true);      case 536: return T(half, true);
                case 544: return T(quarter, true);     case 552: return G(in.depth);
                // the optional inputs (harness captures capFX*, 2026-10-13): unbound without them, like NGX
                case 272: return in.specularHitDistance ? G(in.specularHitDistance) : Resource{};
                case 336: return in.disocclusionMask ? G(in.disocclusionMask) : Resource{};
                case 288: return in.subsurfaceGuide ? G(in.subsurfaceGuide) : Resource{};
            } else switch (offset) {
                case 368: return T(feat[par]); case 376: return T(mv[0]); case 384: return T(mv[1]);
            }
        } else if (stage == R2Dec0B) {
            if (surface) switch (offset) {
                case 176: return T(rec); case 184: return T(ka); case 192: return T(kb); case 200: return T(kc);
            }
        } else if (stage == R2Post) {
            if (!surface) switch (offset) {
                case 24: return T(ka); case 32: return T(kb, true); case 40: return T(kc); case 48: return T(hist[par ^ 1], true);
                case 56: return T(feat[par]); case 64: return T(mv[0]); case 72: return T(mv[1], true); case 80: return G(in.color, true);
            } else switch (offset) {
                case 96: return T(hist[par]); case 120: return T(half);
            }
        } else if (stage == R2Down) {
            if (surface && offset == 40) return Resource{output, false, false};
        } else if (stage == R2DownP) {
            if (surface && offset == 40) return Resource{output, false, false};
            // the game's colour, which this variant reads at the render rectangle (arguments 88 - 100) where it
            // composites the particle layer. NGX binds it (a CUDA-path replay of a harness frame with
            // ColorBeforeParticles, 2026-10-14); left null, the kernel's output is NaN (Cyberpunk 2077 showed black)
            if (!surface && offset == 80) return G(in.color);
            if (surface && offset == 264) return T(pb[par]);
            if (!surface && offset == 208) return in.colorBeforeParticles ? G(in.colorBeforeParticles) : Resource{};
            if (!surface && offset == 256) return T(pb[par ^ 1], true);
        } else if (stage == R2Pooled) {
            return surface ? T(quarter) : T(half);
        }
        return {};
    }

    R2Engine(const Device& device, R2Model& m, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight)
        : d(device), model(m), rw(renderWidth), rh(renderHeight), ow(outputWidth), oh(outputHeight) {
        w0 = (rw + 31) & ~31u; h0 = (rh + 31) & ~31u;
        iw = (ow * 3 + 1) / 2; ih = (oh * 3 + 1) / 2;
        cw = iw / 2; ch = ih / 2; qw = (cw + 1) / 2; qh = (ch + 1) / 2;
        // tiles: columns of at most 720 and rows of at most 1620 intermediate pixels, 4 pixels of overlap inside
        nx = (iw + 719) / 720; ny = (ih + 1619) / 1620;
        for (uint32_t ty = 0; ty < ny; ++ty)
            for (uint32_t tx = 0; tx < nx; ++tx) {
                const uint32_t x0 = tx * iw / nx, x1 = (tx + 1) * iw / nx, y0 = ty * ih / ny, y1 = (ty + 1) * ih / ny;
                src.push_back({x0 >= 4 ? x0 - 4 : 0, y0 >= 4 ? y0 - 4 : 0, std::min(iw, x1 + 4), std::min(ih, y1 + 4)});
                dst.push_back({tx * ow / nx, ty * oh / ny, (tx + 1) * ow / nx, (ty + 1) * oh / ny});
            }
        maxW = maxH = 0;
        for (const auto& r : src) { maxW = std::max(maxW, r.x1 - r.x0); maxH = std::max(maxH, r.y1 - r.y0); }
        unit = static_cast<uint64_t>(w0) * h0 / 32;
        const uint64_t tilePixels = static_cast<uint64_t>(maxW + 8) * (maxH + 8);
        tileBytes = tilePixels * 6;
        const uint32_t tileCount = nx * ny;
        const uint32_t slots = std::min(tileCount, R2TileSlots);
        // every tile its own slot when they fit; otherwise groups of half the slots
        tileGroup = tileCount <= R2TileSlots ? tileCount : R2TileSlots / 2;
        const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        // debugging: D4R_ENGINE_RR_ARENA=DIR keeps the arena and the exposure values host-visible and writes them at the
        // next setFrame (DIR/F-arena.bin, F-values.bin), when the caller has waited for frame F
        const bool debug = std::getenv("D4R_ENGINE_RR_ARENA") != nullptr;
        // Every arena region is written and read within one frame, so the tile slots (live in the output stage only) and
        // the gather records (live from the gather pass to the end of encoder 0) can share the buffer with the network's
        // regions where those are dead: the tile slots that fit before the network's output (dec0_1's argument 64; the
        // output stage also reads the arena after it), the records' rows of windows in the two largest gaps between
        // encoder 0's regions ("s3" models) and from the end of its last region on (a region ends at the next region's
        // start). Saves 105 MB at 720p -> 4K. D4R_ENGINE_RR_ARENA keeps the regions apart, so that dumps show the
        // network's regions as NGX leaves them.
        std::vector<uint32_t> starts{R2ArenaUnits};
        for (unsigned stage = 0; stage < R2Network; ++stage)
            for (const auto& [offset, units] : model.stages[stage].pointers) starts.push_back(static_cast<uint32_t>(std::lround(units)));
        std::sort(starts.begin(), starts.end());
        auto regionEnd = [&](uint32_t start) { return *std::upper_bound(starts.begin(), starts.end(), start); };
        uint32_t netOutStart = 0, enc0End = 0;
        for (const auto& [offset, units] : model.stages[R2Dec0B].pointers)
            if (offset == 64) netOutStart = static_cast<uint32_t>(std::lround(units));
        for (unsigned stage : {R2Enc0, R2Gather})
            for (const auto& [offset, units] : model.stages[stage].pointers) enc0End = std::max(enc0End, regionEnd(static_cast<uint32_t>(std::lround(units))));
        const uint64_t arenaBytes = unit * R2ArenaUnits;
        const uint32_t before = debug ? 0 : static_cast<uint32_t>(std::min<uint64_t>(slots, unit * netOutStart / tileBytes));
        for (uint32_t k = 0; k < slots; ++k) slotAt.push_back(k < before ? tileBytes * k : arenaBytes + tileBytes * (k - before));
        tileB = tilePixels * 4;
        const uint64_t tileEnd = arenaBytes + tileBytes * (slots - before);
        const uint64_t tail = debug ? (tileEnd + 255) & ~uint64_t(255) : (unit * enc0End + 255) & ~uint64_t(255);
        // the gather pass's records: one per window of level 0, (W0 / 8 + 1) to a row (a shifted level has one more)
        const uint32_t rows = h0 / 8 + 1;
        const uint64_t rowBytes = model.gather ? uint64_t(w0 / 8 + 1) * model.stages[R2Gather].record * 4 : 1;
        std::vector<std::pair<uint64_t, uint32_t>> ranges;   // (byte address, rows) of the gaps used
        if (!debug && model.gather && model.stages[R2Gather].ranges) {
            std::vector<std::pair<uint32_t, uint32_t>> used;
            for (unsigned stage : {R2Enc0, R2Gather})
                for (const auto& [offset, units] : model.stages[stage].pointers) {
                    const uint32_t start = static_cast<uint32_t>(std::lround(units));
                    used.push_back({start, regionEnd(start)});
                }
            std::sort(used.begin(), used.end());
            uint32_t at = 0;
            for (const auto& [start, end] : used) {
                if (start > at) {
                    const uint64_t from = (unit * at + 255) & ~uint64_t(255), to = unit * start;
                    if (to > from && (to - from) / rowBytes > 0) ranges.push_back({from, static_cast<uint32_t>((to - from) / rowBytes)});
                }
                at = std::max(at, end);
            }
            std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            if (ranges.size() > 2) ranges.resize(2);
        }
        uint32_t placed = 0;
        for (const auto& r : ranges) placed += r.second;
        ranges.resize(2, {tail, 0});
        gatherWords = {static_cast<uint32_t>(ranges[0].first), ranges[0].second, static_cast<uint32_t>(ranges[1].first),
                       ranges[1].second, static_cast<uint32_t>(tail)};
        const uint64_t gatherTail = placed >= rows ? 0 : (rows - placed) * rowBytes;
        const uint64_t arenaTotal = std::max({tileEnd, model.gather ? tail + gatherTail : 0});
        if (arenaTotal > 0xffffffffull) throw std::runtime_error("Ray Reconstruction: the render size is too large for the arena");
        arena.create(d, arenaTotal, storage, debug);
        values.create(d, 16, storage, debug);
        dummy.create(d, 4096, storage);
        exposure.blocksX = (rw + 15) / 16; exposure.blocks = exposure.blocksX * ((rh + 15) / 16);
        exposure.last[0] = static_cast<int32_t>(rw - 1); exposure.last[1] = static_cast<int32_t>(rh - 1);
        exposure.invCount = static_cast<float>(10000.0 / (4294967296.0 * rw * rh));
        sums.create(d, std::max(exposure.blocks, 2u) * 8, storage);   // the one-pass exposure: the total and the count
        const size_t tiles = src.size();
        auto place = [&](size_t bytes) { const VkDeviceSize at = paramBytes; paramBytes += (bytes + 255) & ~VkDeviceSize(255); return at; };
        for (unsigned i = 0; i < R2Network; ++i) { blocks[i] = model.stages[i].args; paramAt[i] = place(blocks[i].size()); }
        postBlocks.assign(tiles, model.stages[R2Post].args); downBlocks.assign(tiles, model.stages[R2Down].args);
        if (model.particles) downPBlocks.assign(tiles, model.stages[R2DownP].args);
        for (size_t i = 0; i < tiles; ++i) {
            postAt.push_back(place(postBlocks[i].size())); downAt.push_back(place(downBlocks[i].size()));
            if (model.particles) downPAt.push_back(place(downPBlocks[i].size()));
        }
        pooledBlock = model.stages[R2Pooled].args;
        pooledAt = place(pooledBlock.size());
        expAt = place(sizeof(R2ExposureParams));
        paramAll.create(d, paramBytes, storage);
        const VkFormat half4 = VK_FORMAT_R16G16B16A16_SFLOAT;
        for (int i = 0; i < 2; ++i) {
            feat[i].create(d, rw, rh, half4); hist[i].create(d, iw, ih, half4);
            e[i].create(d, 1, 1, VK_FORMAT_R16_SFLOAT); mv[i].create(d, rw, rh, VK_FORMAT_R16G16_SFLOAT);
        }
        rec.create(d, rw, rh, VK_FORMAT_R8G8B8A8_SNORM);
        ka.create(d, rw, rh, VK_FORMAT_R8G8B8A8_UNORM); kb.create(d, rw, rh, VK_FORMAT_R8G8B8A8_UNORM); kc.create(d, rw, rh, half4);
        half.create(d, cw, ch, pooledFormat()); quarter.create(d, qw, qh, pooledFormat());
        zero.create(d, 1, 1, half4);
        for (const auto& st : model.stages)
            for (const auto& [offset, format] : st.surfaces)
                if (!spare.count(format)) spare[format].create(d, 1, 1, surfaceFormat(format));
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = 0;
        VKCHECK(vkCreateSampler(d.handle, &si, nullptr, &nearest));
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        VKCHECK(vkCreateSampler(d.handle, &si, nullptr, &linear));
        // the fixed fields. Network stages: size, arena regions, 1 / size
        for (unsigned stage = 0; stage < R2Network; ++stage) {
            auto& p = blocks[stage];
            const int l = R2Level[stage];
            put32(p, 0, lw(l)); put32(p, 4, lh(l));
            putF(p, 84, 1.0f / static_cast<float>(lw(l))); putF(p, 88, 1.0f / static_cast<float>(lh(l)));
            for (const auto& [offset, units] : model.stages[stage].pointers)
                put64(p, offset, static_cast<uint64_t>(std::llround(units * static_cast<double>(unit))));
        }
        const float fr = static_cast<float>(rw), fh = static_cast<float>(rh), fiw = static_cast<float>(iw), fih = static_cast<float>(ih);
        const float fow = static_cast<float>(ow), foh = static_cast<float>(oh);
        auto& p = blocks[R2Enc0];
        if (model.gather && model.stages[R2Gather].ranges)   // the six words compile_rr2.py appends to the block
            for (unsigned i = 0; i < 5; ++i) put32(p, p.size() - 24 + 4 * i, gatherWords[i]);
        else if (model.gather)   // an older model: one address, the records in one piece
            put64(p, p.size() - 8, gatherWords[4]);
        put32(p, 128, rw); put32(p, 132, rh); put32(p, 136, w0); put32(p, 140, h0);
        putF(p, 408, fr / fiw); putF(p, 412, fh / fih); putF(p, 416, fiw / fr); putF(p, 420, fih / fh);
        putF(p, 424, 1.0f / fr); putF(p, 428, 1.0f / fh); putF(p, 432, 1.0f / fiw); putF(p, 436, 1.0f / fih);
        put32(p, 468, rw - 1); put32(p, 472, rh - 1); put32(p, 488, iw); put32(p, 492, ih); put32(p, 496, cw); put32(p, 500, ch);
        putF(p, 504, 1.0f / fiw); putF(p, 508, 1.0f / fih);
        putF(p, 512, 1.0f / static_cast<float>(2 * qw)); putF(p, 516, 1.0f / static_cast<float>(2 * qh));
        put32(p, 1076, rw); put32(p, 1080, rh);
        // the network's output for the output stage: dec0_1's argument 64
        uint64_t netOut = 0;
        for (const auto& [offset, units] : model.stages[R2Dec0B].pointers)
            if (offset == 64) netOut = static_cast<uint64_t>(std::llround(units * static_cast<double>(unit)));
        const uint32_t postCount = ((maxW + 15) / 16) * ((maxH + 31) / 32);
        for (size_t i = 0; i < tiles; ++i) {
            auto& o = postBlocks[i];
            put64(o, 0, netOut); put32(o, 8, w0);   // the row stride of the network's output: the padded width
            put64(o, 104, slotAt[tileSlot(i)]); put64(o, 112, slotAt[tileSlot(i)] + tileB);
            putF(o, 132, fr / fiw); putF(o, 136, fh / fih); put32(o, 148, iw); put32(o, 152, ih);
            putF(o, 160, 1.0f / fiw); putF(o, 164, 1.0f / fih); put32(o, 168, rw); put32(o, 172, rh);
            put32(o, 176, src[i].x0); put32(o, 180, src[i].y0); put32(o, 184, src[i].x1); put32(o, 188, src[i].y1);
            put32(o, 192, maxW); put32(o, 196, maxH);
            putF(o, 200, 1.0f / fr); putF(o, 204, 1.0f / fh);
            put32(o, 272, 0xffffffffu); put32(o, 276, 0); put32(o, 280, static_cast<uint32_t>(i));
            for (auto* wb : {&downBlocks[i], model.particles ? &downPBlocks[i] : nullptr}) {
                if (!wb) continue;
                auto& w = *wb;
                put32(w, 0, iw); put32(w, 4, ih); put32(w, 8, ow); put32(w, 12, oh); put32(w, 16, ow); put32(w, 20, oh);
                put64(w, 24, slotAt[tileSlot(i)]); put64(w, 32, slotAt[tileSlot(i)] + tileB);
                put32(w, 96, rw - 1); put32(w, 100, rh - 1);
                putF(w, 104, fr / fow); putF(w, 108, fh / foh); putF(w, 112, fow / fiw); putF(w, 116, foh / fih);
                put32(w, 128, dst[i].x0); put32(w, 132, dst[i].y0); put32(w, 136, dst[i].x1); put32(w, 140, dst[i].y1);
                put32(w, 144, maxW); put32(w, 148, maxH);
                put32(w, 152, src[i].x0); put32(w, 156, src[i].y0); put32(w, 160, src[i].x1); put32(w, 164, src[i].y1);
                put32(w, 184, static_cast<uint32_t>(i)); put32(w, 188, postCount); put32(w, 192, static_cast<uint32_t>(i) + 2);
            }
        }
        put32(pooledBlock, 16, cw); put32(pooledBlock, 20, ch); put32(pooledBlock, 24, qw); put32(pooledBlock, 28, qh);
    }
    ~R2Engine() {
        for (auto& b : bindings) vkDestroyDescriptorPool(d.handle, b.pool, nullptr);
        if (nearest) vkDestroySampler(d.handle, nearest, nullptr);
        if (linear) vkDestroySampler(d.handle, linear, nullptr);
    }
    std::vector<REngine::DebugImage> debugImages() const {
        std::vector<REngine::DebugImage> list;
        const VkFormat half4 = VK_FORMAT_R16G16B16A16_SFLOAT;
        for (int i = 0; i < 2; ++i) {
            const std::string n = std::to_string(i);
            list.push_back({"feat" + n, feat[i].handle, half4, rw, rh});
            list.push_back({"hist" + n, hist[i].handle, half4, iw, ih});
            list.push_back({"e" + n, e[i].handle, VK_FORMAT_R16_SFLOAT, 1, 1});
            list.push_back({"mv" + n, mv[i].handle, VK_FORMAT_R16G16_SFLOAT, rw, rh});
        }
        list.push_back({"rec", rec.handle, VK_FORMAT_R8G8B8A8_SNORM, rw, rh});
        list.push_back({"ka", ka.handle, VK_FORMAT_R8G8B8A8_UNORM, rw, rh});
        list.push_back({"kb", kb.handle, VK_FORMAT_R8G8B8A8_UNORM, rw, rh});
        list.push_back({"kc", kc.handle, half4, rw, rh});
        list.push_back({"half", half.handle, pooledFormat(), cw, ch});
        list.push_back({"quarter", quarter.handle, pooledFormat(), qw, qh});
        if (out.handle) list.push_back({"out", out.handle, half4, ow, oh});
        return list;
    }
    void recordInitialize(VkCommandBuffer cb) {
        std::vector<VkImage> images;
        for (auto* group : {&feat, &hist, &e, &mv}) for (auto& t : *group) images.push_back(t.handle);
        for (auto* t : {&rec, &ka, &kb, &kc, &half, &quarter, &zero}) images.push_back(t->handle);
        for (auto& [format, t] : spare) images.push_back(t.handle);
        VkClearColorValue clear{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (VkImage image : images) {
            imageBarrier(cb, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        }
        for (auto* b : {&arena, &values, &sums, &dummy}) vkCmdFillBuffer(cb, b->handle, 0, VK_WHOLE_SIZE, 0);
        memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        frame = 0; reset = true;
    }
    // a stage's descriptor set: parameter block, weights, arena, textures, surfaces, further buffers
    VkDescriptorSet stageSet(Bindings& b, unsigned pass, VkDeviceSize param, uint32_t par, VkImageView output) {
        const auto& st = model.stages[pass];
        const unsigned stage = pass == R2Gather ? R2Enc0 : pass;   // the gather pass has encoder 0's bindings
        VkDescriptorSet set = descriptorSet(d.handle, b.pool, st.pipeline.descriptors);
        bindRange(d.handle, set, 0, paramAll, param);
        bindBuffer(d.handle, set, 1, st.weights);
        bindBuffer(d.handle, set, 2, arena);
        uint32_t binding = 3;
        for (uint32_t offset : st.textures) {
            const auto res = resolve(stage, false, offset, par, b.images, output);
            bindImage(d.handle, set, binding++, res.view ? res.view : zero.view, res.linear ? linear : nearest,
                      res.game ? b.images.layout : VK_IMAGE_LAYOUT_GENERAL);
        }
        for (const auto& [offset, format] : st.surfaces) {
            const auto res = resolve(stage, true, offset, par, b.images, output);
            bindImage(d.handle, set, binding++, res.view ? res.view : spare.at(format).view);
        }
        for (uint32_t offset : st.buffers) {
            // the exposure values (enc0 256, post 16, down and downp 72), enc0's second table (104); the progress counters
            // are never accessed (their pointers are null). downp scales its output by 1 / the exposure: with the
            // zeroed dummy there every pixel was NaN (Cyberpunk 2077, which passes ColorBeforeParticles, showed black)
            const bool vals = (stage == R2Enc0 && offset == 256) || (stage == R2Post && offset == 16) ||
                              ((stage == R2Down || stage == R2DownP) && offset == 72);
            bindBuffer(d.handle, set, binding++, vals ? values : (stage == R2Enc0 && offset == 104) ? model.aux : dummy);
        }
        return set;
    }
    void setFrame(const RFrameImages& images, const RFrameSettings& settings, FrameOutput finalOutput) {
        if (!images.color || !images.motion || !images.depth || !images.normals || !images.roughness || !images.diffuseAlbedo || !images.specularAlbedo)
            throw std::runtime_error("Ray Reconstruction needs colour, motion, depth and the four guides");
        if (finalOutput.view && finalOutput.format != VK_FORMAT_R16G16B16A16_SFLOAT)
            throw std::runtime_error("Ray Reconstruction writes RGBA16F output only");
        const bool rst = settings.reset || frame == 0;
        const uint32_t f = frame;
        if (const char* dir = std::getenv("D4R_ENGINE_RR_ARENA"); dir && f > 0 && arena.mapped) {
            std::ofstream(std::string(dir) + "/" + std::to_string(f) + "-arena.bin", std::ios::binary)
                .write(static_cast<const char*>(arena.mapped), static_cast<std::streamsize>(unit * R2ArenaUnits));
            std::ofstream(std::string(dir) + "/" + std::to_string(f) + "-values.bin", std::ios::binary)
                .write(static_cast<const char*>(values.mapped), 16);
        }
        const float fr = static_cast<float>(rw), fh = static_cast<float>(rh), fiw = static_cast<float>(iw), fih = static_cast<float>(ih);
        const float fow = static_cast<float>(ow), foh = static_cast<float>(oh);
        const float pj[2] = {rst ? 0.0f : prevJitter[0], rst ? 0.0f : prevJitter[1]};
        for (unsigned stage = 0; stage < R2Network; ++stage) put32(blocks[stage], 80, r2Shift(stage, f));
        auto& p = blocks[R2Enc0];
        putF(p, 392, pj[0]); putF(p, 396, pj[1]); putF(p, 400, settings.jitter[0]); putF(p, 404, settings.jitter[1]);
        putF(p, 440, pj[0] / fr); putF(p, 444, pj[1] / fh);
        putF(p, 448, settings.motionScale[0] * (fiw / fr)); putF(p, 452, settings.motionScale[1] * (fih / fh));
        // bytes 465 / 466: the reset flag and DLSS.Roughness.Mode packed (the roughness is the normals' w); 568: inverted
        // depth; 569 / 571: DLSSD.SpecularHitDistance / DLSS.DisocclusionMask are bound (textures 272 / 336); 1028:
        // DLSS.Use.HW.Depth (harness captures capFXrough, capFXinv, capFXhit, capFXmask, capFXhw, 2026-10-13)
        put32(p, 464, (rst ? 256u : 0u) | (settings.roughnessPacked ? 65536u : 0u));
        put32(p, 568, (settings.depthInverted ? 1u : 0u) | (images.specularHitDistance ? 256u : 0u) | (images.disocclusionMask ? 0x1000000u : 0u));
        p[1028] = settings.hardwareDepth ? 1u : 0u;
        // the camera, as presets D and E (runtime_rr.inc), 48 bytes further on
        Mat4 view, clip;
        for (int i = 0; i < 16; ++i) { view[i] = settings.worldToView[i]; clip[i] = settings.viewToClip[i]; }
        view[12] = view[13] = view[14] = 0.0;
        const Mat4 rc = mul(view, clip), inv = inverse(rc);
        const Mat4 toNdc{2.0 / rw, 0, 0, 0, 0, -2.0 / rh, 0, 0, 0, 0, 1, 0, -1, 1, 0, 1};
        const Mat4 toPixels{rw / 2.0, 0, 0, 0, 0, -(rh / 2.0), 0, 0, 0, 0, 1, 0, rw / 2.0, rh / 2.0, 0, 1};
        putMat(p, 644, rc); putMat(p, 708, clip); putMat(p, 772, clip); putMat(p, 836, inv);
        putMat(p, 900, mul(toNdc, inv)); putMat(p, 964, mul(rc, toPixels));
        putF(p, 804, settings.jitter[0] * (2.0f / fr)); putF(p, 808, -settings.jitter[1] * (2.0f / fh));
        // byte 1032: NGX's frame index modulo 8 (0 - 7, then 0 again; 12-frame jitter capture, 2026-10-14). The raw
        // count matched the first eight frames only and made every later frame differ
        put32(p, 1032, f & 7u);
        for (size_t i = 0; i < postBlocks.size(); ++i) {
            auto& o = postBlocks[i];
            putF(o, 140, settings.jitter[0]); putF(o, 144, settings.jitter[1]);
            put32(o, 208, rst ? 1u : 0u);
            for (auto* wb : {&downBlocks[i], model.particles ? &downPBlocks[i] : nullptr}) {
                if (!wb) continue;
                auto& w = *wb;
                putF(w, 120, settings.jitter[0]); putF(w, 124, settings.jitter[1]);
                putF(w, 288, settings.jitter[0] * (fow / fr)); putF(w, 292, settings.jitter[1] * (foh / fh));
                // byte 196: the reset flag; with the particle layer also byte 197, and byte 198 set (capFXparticles)
                put32(w, 196, wb == &downBlocks[i] ? (rst ? 1u : 0u) : (rst ? 0x010101u : 0x010000u));
            }
        }
        if (images.colorBeforeParticles) {
            if (!model.particles) throw std::runtime_error("this preset F model has no particle-layer downsample stage (downp)");
            if (!pb[0].handle) {
                for (auto& t : pb) t.create(d, ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT);
                pbNew = true;
            }
        }
        if (!finalOutput.view && !out.handle) {
            out.create(d, ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT);
            outNew = true;
        }
        exposure.reset = rst ? 1u : 0u;
        reset = rst;
        auto same = [&](const Bindings& b) {
            return !std::memcmp(&b.images, &images, sizeof(images)) && b.finalOutput.view == finalOutput.view;
        };
        auto found = std::find_if(bindings.begin(), bindings.end(), same);
        if (found == bindings.end()) {
            if (bindings.size() >= 16) throw std::runtime_error("too many distinct Ray Reconstruction input view sets");
            Bindings b;
            b.images = images; b.finalOutput = finalOutput;
            const uint32_t tiles = static_cast<uint32_t>(src.size());
            auto count = [&](unsigned stage, uint32_t sets, uint32_t& bufs, uint32_t& tex, uint32_t& surf) {
                const auto& st = model.stages[stage];
                bufs += sets * (3 + static_cast<uint32_t>(st.buffers.size()));
                tex += sets * static_cast<uint32_t>(st.textures.size()); surf += sets * static_cast<uint32_t>(st.surfaces.size());
            };
            uint32_t bufs = 8, tex = 4, surf = 4, sets = 3;
            for (unsigned stage = 0; stage < R2Network; ++stage) { count(stage, 2, bufs, tex, surf); sets += 2; }
            if (model.gather) { count(R2Gather, 2, bufs, tex, surf); sets += 2; }
            count(R2Post, 2 * tiles, bufs, tex, surf); count(R2Down, tiles, bufs, tex, surf); count(R2Pooled, 1, bufs, tex, surf);
            if (images.colorBeforeParticles) count(R2DownP, 2 * tiles, bufs, tex, surf);
            sets += 5 * tiles + 1;
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bufs}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, surf}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = sets; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
            VKCHECK(vkCreateDescriptorPool(d.handle, &pi, nullptr, &b.pool));
            const VkImageView output = finalOutput.view ? finalOutput.view : out.view;
            for (unsigned stage = 0; stage < R2Network; ++stage)
                for (uint32_t par = 0; par < 2; ++par) b.network[stage][par] = stageSet(b, stage, paramAt[stage], par, output);
            if (model.gather)
                for (uint32_t par = 0; par < 2; ++par) b.gather[par] = stageSet(b, R2Gather, paramAt[R2Enc0], par, output);
            for (uint32_t i = 0; i < tiles; ++i) {
                b.post.push_back({stageSet(b, R2Post, postAt[i], 0, output), stageSet(b, R2Post, postAt[i], 1, output)});
                b.down.push_back(stageSet(b, R2Down, downAt[i], 0, output));
                if (images.colorBeforeParticles)
                    b.downP.push_back({stageSet(b, R2DownP, downPAt[i], 0, output), stageSet(b, R2DownP, downPAt[i], 1, output)});
            }
            b.pooled = stageSet(b, R2Pooled, pooledAt, 0, output);
            if (model.exposureOne.handle)
                for (uint32_t par = 0; par < 2; ++par) {
                    VkDescriptorSet set = b.exposureOne[par] = descriptorSet(d.handle, b.pool, model.exposureOne.descriptors);
                    bindRange(d.handle, set, 0, paramAll, expAt);
                    bindImage(d.handle, set, 1, images.color, nearest, images.layout);
                    bindBuffer(d.handle, set, 2, sums);
                    bindBuffer(d.handle, set, 3, values);
                    bindImage(d.handle, set, 4, e[par ^ 1].view, nearest);
                    bindImage(d.handle, set, 5, e[par].view);
                }
            else {
                b.exposure0 = descriptorSet(d.handle, b.pool, model.exposure[0].descriptors);
                bindRange(d.handle, b.exposure0, 0, paramAll, expAt);
                bindImage(d.handle, b.exposure0, 1, images.color, nearest, images.layout);
                bindBuffer(d.handle, b.exposure0, 2, sums);
                for (uint32_t par = 0; par < 2; ++par) {
                    VkDescriptorSet set = b.exposure1[par] = descriptorSet(d.handle, b.pool, model.exposure[1].descriptors);
                    bindRange(d.handle, set, 0, paramAll, expAt);
                    bindBuffer(d.handle, set, 1, sums);
                    bindBuffer(d.handle, set, 2, values);
                    bindImage(d.handle, set, 3, e[par ^ 1].view, nearest);
                    bindImage(d.handle, set, 4, e[par].view);
                }
            }
            bindings.push_back(std::move(b));
            found = bindings.end() - 1;
        }
        bound = &*found;
        // texture and surface handles: 1 where something is bound, 0 where NGX leaves the argument null (the kernels
        // test optional inputs' handles); buffer pointers other than the arena's are zero
        const VkImageView output = finalOutput.view ? finalOutput.view : out.view;
        auto handles = [&](unsigned stage, std::vector<uint8_t>& block) {
            const auto& st = model.stages[stage];
            for (uint32_t offset : st.textures) put64(block, offset, resolve(stage, false, offset, 0, images, output).view ? 1u : 0u);
            for (const auto& [offset, format] : st.surfaces) put64(block, offset, resolve(stage, true, offset, 0, images, output).view ? 1u : 0u);
            for (uint32_t offset : st.buffers) put64(block, offset, 0);
        };
        for (unsigned stage = 0; stage < R2Network; ++stage) handles(stage, blocks[stage]);
        for (auto& o : postBlocks) handles(R2Post, o);
        for (auto& w : downBlocks) handles(R2Down, w);
        for (auto& w : downPBlocks) handles(R2DownP, w);
        handles(R2Pooled, pooledBlock);
        if (const char* dump = std::getenv("D4R_ENGINE_RR_DUMP")) {
            auto write = [&](const std::string& name, const std::vector<uint8_t>& block) {
                std::ofstream(std::string(dump) + "/" + std::to_string(f + 1) + "-" + name + ".bin", std::ios::binary)
                    .write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(block.size()));
            };
            for (unsigned stage = 0; stage < R2Network; ++stage) write(R2Names[stage], blocks[stage]);
            for (size_t i = 0; i < postBlocks.size(); ++i) { write("post" + std::to_string(i), postBlocks[i]); write("down" + std::to_string(i), downBlocks[i]); }
        }
        prevJitter[0] = settings.jitter[0]; prevJitter[1] = settings.jitter[1];
    }
    void recordFrame(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) {
        if (!bound) throw std::runtime_error("REngine::setFrame was not called");
        const auto& b = *bound;
        const uint32_t f = frame, cur = f & 1u;
        auto stamp = [&](uint32_t n) { if (stamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, stamps, first + n); };
        // timing only: D4R_ENGINE_RR_STAGE_STAMPS=1 also stamps after every network stage and output group, from query
        // first + 4 on (the caller's pool needs 32 queries)
        static const bool stageStamps = std::getenv("D4R_ENGINE_RR_STAGE_STAMPS") != nullptr;
        uint32_t extra = 4;
        auto stageStamp = [&] { if (stageStamps) stamp(extra++); };
        auto step = [&] {
            memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        };
        if (pbNew || outNew) {   // images created on first use start from zero, like the rest of the temporal state
            VkClearColorValue clear{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            std::vector<VkImage> fresh;
            if (pbNew) for (auto& t : pb) fresh.push_back(t.handle);
            if (outNew) fresh.push_back(out.handle);
            for (VkImage image : fresh) {
                imageBarrier(cb, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
                vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
            }
            memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            pbNew = outNew = false;
        }
        stamp(0);
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        std::vector<uint8_t> all(paramBytes);
        auto put = [&](VkDeviceSize at, const void* data, size_t bytes) { std::memcpy(all.data() + at, data, bytes); };
        for (unsigned i = 0; i < R2Network; ++i) put(paramAt[i], blocks[i].data(), blocks[i].size());
        for (size_t i = 0; i < postBlocks.size(); ++i) { put(postAt[i], postBlocks[i].data(), postBlocks[i].size()); put(downAt[i], downBlocks[i].data(), downBlocks[i].size()); }
        for (size_t i = 0; i < downPBlocks.size(); ++i) put(downPAt[i], downPBlocks[i].data(), downPBlocks[i].size());
        put(pooledAt, pooledBlock.data(), pooledBlock.size());
        put(expAt, &exposure, sizeof(exposure));
        for (VkDeviceSize at = 0; at < paramBytes; at += 65536)   // vkCmdUpdateBuffer takes at most 65536 bytes
            vkCmdUpdateBuffer(cb, paramAll.handle, at, std::min<VkDeviceSize>(65536, paramBytes - at), all.data() + at);
        memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        // the exposure: one pass with 64-bit atomics (the last block finishes, as NGX's kernel), or the block sums and
        // one workgroup that adds them
        if (model.exposureOne.handle)
            dispatch(cb, model.exposureOne, b.exposureOne[cur], exposure.blocksX, exposure.blocks / exposure.blocksX);
        else {
            dispatch(cb, model.exposure[0], b.exposure0, exposure.blocksX, exposure.blocks / exposure.blocksX);
            step();
            dispatch(cb, model.exposure[1], b.exposure1[cur], 1, 1);
        }
        step();
        stamp(1);
        for (unsigned stage = 0; stage < R2Network; ++stage) {
            const int l = R2Level[stage];
            const uint32_t shift = r2Shift(stage, f), sx = shift & 0xffffu, sy = shift >> 16;
            const uint32_t gx = (lw(l) + sx + 7) / 8;
            if (stage == R2Enc0 && model.gather) {
                dispatch(cb, model.stages[R2Gather].pipeline, b.gather[cur], gx, (lh(l) + sy + 7) / 8);
                step();
            }
            dispatch(cb, model.stages[stage].pipeline, b.network[stage][cur], (gx + model.stages[stage].group - 1) / model.stages[stage].group, (lh(l) + sy + 7) / 8);
            step();
            stageStamp();
        }
        stamp(2);
        // timing only: D4R_ENGINE_RR_SKIP=post,down,pooled leaves those passes out (the picture is wrong then)
        static const std::string skip = std::getenv("D4R_ENGINE_RR_SKIP") ? std::getenv("D4R_ENGINE_RR_SKIP") : "";
        static const bool noPost = skip.find("post") != skip.npos, noDown = skip.find("down") != skip.npos, noPooled = skip.find("pooled") != skip.npos;
        static const bool serial = skip.find("serial") != skip.npos;   // a barrier after every output pass, as before
        // group g: its post passes, then (after one barrier) its down passes together with group g + 1's post passes
        const bool particles = b.images.colorBeforeParticles != VK_NULL_HANDLE;
        const size_t tiles = src.size(), groups = (tiles + tileGroup - 1) / tileGroup;
        for (size_t g = 0; g <= groups; ++g) {
            if (g > 0)
                for (size_t i = (g - 1) * tileGroup; i < std::min(tiles, g * tileGroup); ++i) {
                    if (!noDown) dispatch(cb, model.stages[particles ? R2DownP : R2Down].pipeline, particles ? b.downP[i][cur] : b.down[i],
                                          (dst[i].x1 - dst[i].x0 + 15) / 16, (dst[i].y1 - dst[i].y0 + 15) / 16);
                    if (serial) step();
                }
            for (size_t i = g * tileGroup; i < std::min(tiles, (g + 1) * tileGroup); ++i) {
                if (!noPost) dispatch(cb, model.stages[R2Post].pipeline, b.post[i][cur], (maxW + 15) / 16, (maxH + 31) / 32);
                if (serial) step();
            }
            if (g < groups) step();
            stageStamp();
        }
        // (the pooling pass reads what the post passes wrote: the last group's down passes do not touch it)
        if (!noPooled) dispatch(cb, model.stages[R2Pooled].pipeline, b.pooled, (qw + 15) / 16, (qh + 15) / 16);
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
        stamp(3);
        ++frame;
    }
};
} // namespace
