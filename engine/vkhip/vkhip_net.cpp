// vkhip::KNetwork: see vkhip_net.h. The plan (buffers, wiring, window cycle, launch shapes) is engine/hip_net.cpp's
// preset K plan; only the backend differs.
#include "vkhip_net.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vkhip {
namespace {

struct LayerParamsK { int32_t W, H; uint64_t in, skip, out24, out32, pad40, pad48; int32_t sx, sy; uint64_t w; uint8_t rest[104]; };
static_assert(sizeof(LayerParamsK) == 176, "K params");
// preset M's kernels' parameter block (kernels/m/swin_common.h); engine/hip_net.cpp's LayerParamsM
struct LayerParamsM { uint64_t w; int32_t sx, sy, tw, th; uint64_t in, skip, out, merged; int32_t r69, r70, r71, r72; uint64_t inFlags, outFlags; };
static_assert(sizeof(LayerParamsM) == 88, "M params");
// preset L's native stages (kernels/tex/enc0l_net.hip, dec0l_net.hip): a, b, c are features, out, merged or in, skip, out
struct StageParamsL { uint64_t w; int32_t sx, sy, tw, th; uint64_t a, b, c; };
static_assert(sizeof(StageParamsL) == 48, "L stage params");
struct KLayer { const char* name; uint32_t divisor, cin, c, out24ch; int role; };
constexpr KLayer KPlan[11] = {
    {"enc0", 1, 16, 32, 64, -1}, {"enc1", 2, 64, 64, 64, -1}, {"enc2", 4, 64, 64, 96, -1},
    {"enc3", 8, 96, 96, 128, 2}, {"enc4", 16, 128, 128, 160, 1}, {"dec5", 32, 160, 160, 160, 0},
    {"dec4", 16, 160, 128, 128, 1}, {"dec3", 8, 128, 96, 96, 2}, {"dec2", 4, 96, 64, 64, -1},
    {"dec1", 2, 64, 64, 64, -1}, {"dec0", 1, 64, 32, 40, -1},
};
constexpr int32_t KCycle[8][2] = {{0, 2}, {5, 6}, {4, 0}, {1, 4}, {7, 5}, {2, 1}, {3, 7}, {6, 3}};
void kShift(int role, uint32_t frame, int32_t& sx, int32_t& sy) {
    if (role < 0) { sx = sy = 4; return; }
    int32_t x = KCycle[frame % 8][0], y = KCycle[frame % 8][1];
    for (int d = 0; d < role; ++d) { x = (2 * x + 4) & 7; y = (2 * y + 4) & 7; }
    sx = x; sy = y;
}
// The windows of each layer (x, y group counts) the frame's results depend on: engine/runtime.cpp's kLayerGroups
// (the same rule, kept identical): nothing reads the windows that only cover the padded token grid's rows and
// columns past the output, so they are not launched. w/h/sx/sy per layer, out = output width or height.
void trimmedGroups(const int32_t (&size)[2][11], const int32_t (&shift)[2][11], const uint32_t (&out)[2],
                   uint32_t (&groups)[2][11]) {
    auto mirror = [](int v, int n) { v = std::abs(v); return std::min(v, 2 * n - 2 - v); };
    auto tdiv2 = [](int v) { return v < 0 ? -(-v / 2) : v / 2; };
    for (unsigned a = 0; a < 2; ++a) {
        auto full = [&](unsigned i) { return uint32_t(size[a][i] + shift[a][i] + 7) / 8; };
        for (unsigned i = 0; i < 11; ++i) groups[a][i] = full(i);
        auto windows = [&](unsigned i, int n) { return std::clamp(uint32_t(n + shift[a][i] + 7) / 8, 1u, full(i)); };
        auto reads = [&](unsigned i, bool half) {
            int last = 0;
            for (int w = 0; w < int(groups[a][i]); ++w)
                for (int t = 0; t < (half ? 4 : 8); ++t) {
                    const int origin = 8 * w - shift[a][i];
                    last = std::max(last, half ? mirror(tdiv2(origin) + t, size[a][i] / 2) : mirror(origin + t, size[a][i]));
                }
            return last + 1;
        };
        int need = std::min(size[a][10], int(out[a] + 3) / 4 + 2), skip[5] = {};
        for (unsigned i = 10; i >= 6; --i) {
            groups[a][i] = windows(i, need);
            skip[10 - i] = reads(i, false);
            need = reads(i, true);
        }
        groups[a][5] = windows(5, need);
        int merged = reads(5, false);
        for (int i = 4; i >= 0; --i) {
            groups[a][i] = windows(unsigned(i), std::max(skip[i], 2 * merged));
            merged = reads(unsigned(i), false);
        }
    }
}
// The same rule for presets M and L's ten layers, per axis: nothing reads the block rows and columns past the render size. A
// block's output is its own tokens and, in the encoders, 16 half-size tokens; size and shift per layer, stageShift is
// L's expansion stage's (the stage only needs the blocks with a pixel of the render size), below zero for preset M.
constexpr unsigned MLayerCount = 10;
void trimmedGroupsM(const int32_t (&size)[MLayerCount], const int32_t (&shift)[MLayerCount], int render, int stageShift, uint32_t (&groups)[MLayerCount]) {
    auto mirror = [](int v, int n) { v = std::abs(v); return std::min(v, 2 * n - 2 - v); };
    auto tdiv2 = [](int v) { return v < 0 ? -(-v / 2) : v / 2; };
    auto reads = [&](unsigned i, bool half) {
        int last = 0;
        for (int w = 0; w < int(groups[i]); ++w)
            for (int t = 0; t < (half ? 4 : 8); ++t) {
                const int origin = 8 * w - shift[i];
                last = std::max(last, half ? mirror(tdiv2(origin) + t, size[i] / 2) : mirror(origin + t, size[i]));
            }
        return last + 1;
    };
    auto windows = [&](unsigned i, int n) { return std::clamp(uint32_t(n + shift[i] + 7) / 8, 1u, uint32_t(size[i] + shift[i] + 7) / 8); };
    // the last decoder's output: tokens are 2x2 pixels (M), or L's expansion stage reads half-size tokens of them
    int need = (render + 1) / 2;
    if (stageShift >= 0) {
        need = 0;
        for (int w = 0; w < (render + stageShift + 7) / 8; ++w)
            for (int t = 0; t < 4; ++t) need = std::max(need, mirror(tdiv2(8 * w - stageShift) + t, size[9]) + 1);
    }
    int skip[2] = {};
    for (unsigned i = 9; i >= 8; --i) {
        groups[i] = windows(i, need);
        skip[9 - i] = reads(i, false);
        need = reads(i, true);
    }
    for (unsigned i = 7; i >= 2; --i) { groups[i] = windows(i, need); need = reads(i, false); }
    groups[1] = windows(1, std::max(skip[1], 2 * need));
    groups[0] = windows(0, std::max(skip[0], 2 * reads(1, false)));
}
uint32_t global32(const Module& m, const std::string& name, uint32_t fallback) {
    auto it = m.globals.find(name);
    if (it == m.globals.end() || it->second.size != 4 || it->second.va < m.dataLo || it->second.va + 4 > m.dataHi) return fallback;
    uint32_t v = 0; std::memcpy(&v, m.dataInit.data() + (it->second.va - m.dataLo), 4);
    return v ? v : fallback;
}
constexpr uint64_t kAlign = 65536;
constexpr uint64_t kWeightTail = 1ull << 20;  // dec5's prep reads its generic layout's unused merge tiles past its weights
constexpr uint64_t kTail = 65536, kGuard = 4ull << 20;
// Donor keys must not collide when features are recreated at different GPU
// addresses on the same device. Reusing a key could reuse relocated code.
// They must not repeat across processes either: Mesa's on-disk shader cache keeps the patched kernels under the
// donor keys, so a fixed first id handed a later run's donors an earlier run's patched code (the layout check then
// refused the layers that use scratch and the feature left vkhip). Each process starts at its own id.
uint64_t firstPipelineId() {
    uint64_t x = uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()) ^
                 uint64_t(std::chrono::system_clock::now().time_since_epoch().count()) ^
                 reinterpret_cast<uintptr_t>(&x);
    x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return (x ^ (x >> 31)) & 0x7fffffffu;  // leaves 2^31 ids before the key space check
}
std::atomic<uint64_t> nextPipelineId{firstPipelineId()};

} // namespace

struct KNetwork::Impl {
    VkDevice device = VK_NULL_HANDLE;
    Device vk;
    PFN_vkCreateBuffer createBuffer = nullptr; PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements requirements = nullptr; PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory freeMemory = nullptr; PFN_vkBindBufferMemory bind = nullptr;
    PFN_vkGetBufferDeviceAddress address = nullptr; PFN_vkCreatePipelineLayout createLayout = nullptr;
    PFN_vkDestroyPipelineLayout destroyLayout = nullptr;
    PFN_vkCmdBindPipeline cmdBind = nullptr; PFN_vkCmdPushConstants cmdPush = nullptr; PFN_vkCmdDispatch cmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier cmdBarrier = nullptr; PFN_vkCmdUpdateBuffer cmdUpdate = nullptr; PFN_vkCmdFillBuffer cmdFill = nullptr;
    PFN_vkCreateQueryPool createQueries = nullptr; PFN_vkDestroyQueryPool destroyQueries = nullptr;
    PFN_vkGetQueryPoolResults queryResults = nullptr; PFN_vkCmdResetQueryPool cmdResetQueries = nullptr;
    PFN_vkCmdWriteTimestamp cmdStamp = nullptr; PFN_vkCreateEvent createEvent = nullptr; PFN_vkDestroyEvent destroyEvent = nullptr;
    PFN_vkResetEvent resetEvent = nullptr; PFN_vkGetEventStatus eventStatus = nullptr; PFN_vkCmdSetEvent cmdSetEvent = nullptr;
    PFN_vkCmdCopyBuffer cmdCopy = nullptr; PFN_vkMapMemory mapMemory = nullptr; PFN_vkDeviceWaitIdle waitIdle = nullptr;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer arena = VK_NULL_HANDLE, tokens = VK_NULL_HANDLE, head = VK_NULL_HANDLE;
    uint64_t arenaSize = 0, base = 0;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<VkPipeline> pipelines;
    struct Dispatch { VkPipeline pipeline; uint32_t gx, gy; uint64_t args; std::string label; };
    std::vector<Dispatch> preps, nodes[8];
    // contents written once, on the first record: (arena offset, bytes)
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> contents;
    bool uploaded = false;
    uint32_t frame = 0, windowFrame = 0, phases = 8;
    bool resetWindowCycle = false;

    // optional per-dispatch timestamps: slot s holds queries [s*stampCount, (s+1)*stampCount)
    static constexpr uint32_t StampSlots = 8;
    void (*stampLog)(const char*) = nullptr;
    VkQueryPool stampPool = VK_NULL_HANDLE;
    uint32_t stampCount = 0;
    double nsPerTick = 0;
    struct StampSlot { VkEvent done = VK_NULL_HANDLE; bool pending = false; uint32_t frame = 0; };
    StampSlot stampSlots[StampSlots];
    int stampActive = -1;
    uint32_t stampNext = 0;

    void stamp(VkCommandBuffer cb, VkPipelineStageFlagBits stage) {
        if (stampActive >= 0) cmdStamp(cb, stage, stampPool, uint32_t(stampActive) * stampCount + stampNext++);
    }
    void collectStamps(const std::vector<Dispatch>& list) {
        for (uint32_t i = 0; i < StampSlots; ++i) {
            auto& slot = stampSlots[i];
            if (!slot.pending || eventStatus(device, slot.done) != VK_EVENT_SET) continue;
            std::vector<uint64_t> t(stampCount);
            if (queryResults(device, stampPool, i * stampCount, stampCount, t.size() * 8, t.data(), 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) continue;
            slot.pending = false;
            auto ms = [&](uint32_t a, uint32_t b) { return double(int64_t(t[b] - t[a])) * nsPerTick / 1e6; };
            // t[0] network start, t[1 + 2k] dispatch k start, t[2 + 2k] dispatch k end, t[last] network end
            const uint32_t last = stampCount - 1;
            std::string text = "D4R_VKHIP_STAMPS frame=" + std::to_string(slot.frame);
            char item[96];
            std::snprintf(item, sizeof(item), " total=%.4f", ms(0, last)); text += item;
            double busy = 0, gaps = 0;
            for (uint32_t k = 0; k < list.size(); ++k) {
                const double b = ms(1 + 2 * k, 2 + 2 * k), g = ms(k ? 2 * k : 0, 1 + 2 * k);
                busy += b; gaps += g;
                std::snprintf(item, sizeof(item), " b.%s=%.4f g.%s=%.4f", list[k].label.c_str(), b, list[k].label.c_str(), g);
                text += item;
            }
            std::snprintf(item, sizeof(item), " tail=%.4f busy=%.4f gaps=%.4f", ms(last - 1, last), busy, gaps); text += item;
            stampLog(text.c_str());
        }
    }

    // optional dumps of the tokens and the head (debug)
    std::string dumpDir;
    uint32_t dumpFrames = 0;
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    uint64_t tokenBytes = 0, headBytes = 0;
    VkBuffer dumpBuffer = VK_NULL_HANDLE;
    VkDeviceMemory dumpMemory = VK_NULL_HANDLE;
    void writeDumps() {
        if (!dumpBuffer || !frame) return;
        waitIdle(device);
        void* mapped = nullptr;
        if (mapMemory(device, dumpMemory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) return;
        const uint8_t* bytes = static_cast<const uint8_t*>(mapped);
        for (uint32_t f = 0; f < std::min(frame, dumpFrames); ++f)
            for (int part = 0; part < 2; ++part) {
                const std::string path = dumpDir + (part ? "/head-" : "/tokens-") + std::to_string(f) + ".bin";
                if (FILE* out = std::fopen(path.c_str(), "wb")) {
                    const uint64_t at = f * (tokenBytes + headBytes) + (part ? tokenBytes : 0);
                    std::fwrite(bytes + at, 1, part ? headBytes : tokenBytes, out);
                    std::fclose(out);
                }
            }
    }

    ~Impl() {
        writeDumps();
        if (dumpBuffer) destroyBuffer(device, dumpBuffer, nullptr);
        if (dumpMemory) freeMemory(device, dumpMemory, nullptr);
        for (auto& slot : stampSlots) if (slot.done) destroyEvent(device, slot.done, nullptr);
        if (stampPool) destroyQueries(device, stampPool, nullptr);
        for (VkPipeline p : pipelines) if (p) vk.destroyPipeline(device, p, nullptr);
        if (layout) destroyLayout(device, layout, nullptr);
        for (VkBuffer b : {tokens, head, arena}) if (b) destroyBuffer(device, b, nullptr);
        if (memory) freeMemory(device, memory, nullptr);
    }

    template <class F> bool load(PFN_vkGetDeviceProcAddr proc, F& f, const char* name) {
        f = reinterpret_cast<F>(proc(device, name));
        return f != nullptr;
    }
    bool loadEntries(PFN_vkGetDeviceProcAddr proc) {
        bool ok = vk.load(device, proc);
    ok &= load(proc, createBuffer, "vkCreateBuffer") && load(proc, destroyBuffer, "vkDestroyBuffer") &&
          load(proc, requirements, "vkGetBufferMemoryRequirements") && load(proc, allocate, "vkAllocateMemory") &&
          load(proc, freeMemory, "vkFreeMemory") && load(proc, bind, "vkBindBufferMemory") &&
          load(proc, address, "vkGetBufferDeviceAddress") && load(proc, createLayout, "vkCreatePipelineLayout") &&
          load(proc, destroyLayout, "vkDestroyPipelineLayout") && load(proc, cmdBind, "vkCmdBindPipeline") &&
          load(proc, cmdPush, "vkCmdPushConstants") && load(proc, cmdDispatch, "vkCmdDispatch") &&
          load(proc, cmdBarrier, "vkCmdPipelineBarrier") && load(proc, cmdUpdate, "vkCmdUpdateBuffer") &&
          load(proc, cmdFill, "vkCmdFillBuffer") && load(proc, createQueries, "vkCreateQueryPool") &&
          load(proc, destroyQueries, "vkDestroyQueryPool") && load(proc, queryResults, "vkGetQueryPoolResults") &&
          load(proc, cmdResetQueries, "vkCmdResetQueryPool") && load(proc, cmdStamp, "vkCmdWriteTimestamp") &&
          load(proc, createEvent, "vkCreateEvent") && load(proc, destroyEvent, "vkDestroyEvent") &&
          load(proc, resetEvent, "vkResetEvent") && load(proc, eventStatus, "vkGetEventStatus") &&
          load(proc, cmdSetEvent, "vkCmdSetEvent") && load(proc, cmdCopy, "vkCmdCopyBuffer") &&
          load(proc, mapMemory, "vkMapMemory") && load(proc, waitIdle, "vkDeviceWaitIdle");
        return ok;
    }
    // The arena and the caller's two buffers, in one device-local allocation, and the pipeline layout
    bool allocateArena(const VkPhysicalDeviceMemoryProperties& memoryProperties, uint64_t tokensSize, uint64_t headSize,
                       uint64_t tokensAt, uint64_t headAt, std::string& error) {
        arena = buffer(arenaSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        tokens = buffer(tokensSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        head = buffer(headSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        if (!arena || !tokens || !head) { error = "buffers"; return false; }
        VkMemoryRequirements ra, rt, rh;
        requirements(device, arena, &ra); requirements(device, tokens, &rt); requirements(device, head, &rh);
        const uint32_t bits = ra.memoryTypeBits & rt.memoryTypeBits & rh.memoryTypeBits;
        if (tokensAt % rt.alignment || headAt % rh.alignment) { error = "alias alignment"; return false; }
        uint32_t type = ~0u;
        for (uint32_t t = 0; t < memoryProperties.memoryTypeCount && type == ~0u; ++t)
            if ((bits & (1u << t)) && (memoryProperties.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) type = t;
        if (type == ~0u) { error = "no device-local memory type for the network"; return false; }
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        mai.allocationSize = std::max(ra.size, arenaSize); mai.memoryTypeIndex = type;
        if (allocate(device, &mai, nullptr, &memory) != VK_SUCCESS) { error = "network memory"; return false; }
        if (bind(device, arena, memory, 0) != VK_SUCCESS || bind(device, tokens, memory, tokensAt) != VK_SUCCESS ||
            bind(device, head, memory, headAt) != VK_SUCCESS) { error = "bind"; return false; }
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; ai.buffer = arena;
        base = address(device, &ai);
        if (!base) { error = "no device address"; return false; }
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &range;
        if (createLayout(device, &lci, nullptr, &layout) != VK_SUCCESS) { error = "pipeline layout"; return false; }
        return true;
    }
    VkBuffer buffer(uint64_t size, VkBufferUsageFlags usage) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size; info.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer b = VK_NULL_HANDLE;
        return createBuffer(device, &info, nullptr, &b) == VK_SUCCESS ? b : VK_NULL_HANDLE;
    }
    static void barrier(PFN_vkCmdPipelineBarrier f, VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags srcAccess,
                        VkPipelineStageFlags dst, VkAccessFlags dstAccess) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, srcAccess, dstAccess};
        f(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void run(VkCommandBuffer cb, const std::vector<Dispatch>& list) {
        for (const auto& d : list) {
            cmdBind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);
            cmdPush(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &d.args);
            stamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);  // RADV emits the pending barrier first
            cmdDispatch(cb, d.gx, d.gy, 1);
            stamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
            // the next layer's vector and scalar loads (SHADER_STORAGE_READ invalidates both caches on RADV)
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
    }
};

KNetwork::KNetwork() : impl(std::make_unique<Impl>()) {}
KNetwork::~KNetwork() = default;
VkBuffer KNetwork::tokens() const { return impl->tokens; }
VkBuffer KNetwork::head() const { return impl->head; }
uint32_t KNetwork::dispatchesPerFrame() const { return uint32_t(impl->nodes[0].size()); }

bool KNetwork::create(VkDevice device, PFN_vkGetDeviceProcAddr proc, const VkPhysicalDeviceMemoryProperties& memoryProperties,
                      const NetFiles& files, uint32_t width, uint32_t height, uint64_t tokenBytes, uint64_t headBytes,
                      std::string& error) {
    auto& n = *impl;
    n.device = device;
    n.memoryProperties = memoryProperties;
    bool ok = n.loadEntries(proc);
    if (!ok) { error = "missing Vulkan entry points"; return false; }
    const auto& file = files.hipnet;
    uint32_t count = 0;
    if (file.size() < 16 + 64 * 11 || std::memcmp(file.data(), "D4RKHIP1", 8)) { error = "not a preset K hipnet.bin"; return false; }
    std::memcpy(&count, file.data() + 8, 4);
    if (count != 11) { error = "hipnet.bin does not have eleven layers"; return false; }
    if (width < 256 || height < 256 || width > 8192 || height > 8192) { error = "output dimensions must be 256..8192"; return false; }
    const uint32_t tw = std::max(256u, ((width + 3) / 4 + 31) / 32 * 32), th = std::max(256u, ((height + 3) / 4 + 31) / 32 * 32);

    // code objects
    std::vector<Module> modules(11);
    std::vector<std::string> names(11);
    std::vector<uint64_t> weightAt(11), weightBytes(11);
    for (uint32_t i = 0; i < 11; ++i) {
        const uint8_t* e = file.data() + 16 + 64 * i;
        char name[33]{}; std::memcpy(name, e, 32);
        std::memcpy(&weightAt[i], e + 48, 8); std::memcpy(&weightBytes[i], e + 56, 8);
        if (weightAt[i] > file.size() || weightBytes[i] > file.size() - weightAt[i]) { error = "invalid hipnet.bin layer"; return false; }
        names[i] = name;
        if (names[i] != std::string("dltss_pwin_") + KPlan[i].name + "_layer") { error = "unexpected K layer order: " + names[i]; return false; }
        auto it = files.code.find(names[i]);
        if (it == files.code.end()) { error = names[i] + ".hsaco missing"; return false; }
        if (!modules[i].parse(it->second)) { error = names[i] + ": " + modules[i].error; return false; }
        if (i == 0) n.resetWindowCycle = global32(modules[i], "d4r_reset_window_cycle", 0) != 0;
        if (modules[i].mach != (files.mach ? files.mach : modules[0].mach)) {
            error = names[i] + ".hsaco was built for another GPU (rebuild the engine models)"; return false;
        }
    }

    // the pooled layout: tokens, head, weights, activations, per-module globals, kernarg blocks
    uint64_t used = 0;
    auto carve = [&](uint64_t bytes) { used = (used + kAlign - 1) / kAlign * kAlign; const uint64_t at = used; used += bytes; return at; };
    const uint64_t planTokens = uint64_t(tw) * th * 16 * 2, planHead = uint64_t(tw) * th * 40 * 2;
    const uint64_t tokensSize = std::max(planTokens, tokenBytes), headSize = std::max(planHead, headBytes);
    n.tokenBytes = tokensSize; n.headBytes = headSize;
    const uint64_t tokensAt = carve(tokensSize + kTail), headAt = carve(headSize + kTail);
    std::vector<uint64_t> wAt(11), act(11, ~0ull), full(5, ~0ull), gAt(11);
    for (uint32_t i = 0; i < 11; ++i) wAt[i] = carve(weightBytes[i] + kWeightTail);
    for (uint32_t i = 0; i < 11; ++i) {
        const auto& p = KPlan[i];
        const uint32_t w = tw / p.divisor, h = th / p.divisor;
        const uint32_t ow = i < 5 ? tw / (2 * p.divisor) : w, oh = i < 5 ? th / (2 * p.divisor) : h;
        if (i + 1 != 11) act[i] = carve(uint64_t(ow) * oh * p.out24ch * 2 + kTail);
        if (i < 5) full[i] = carve(uint64_t(w) * h * p.c * 2 + kTail);
    }
    for (uint32_t i = 0; i < 11; ++i) gAt[i] = carve(modules[i].dataHi - modules[i].dataLo + kTail);
    const uint64_t argsAt = carve(256 * (11 + 8 * 32));
    n.arenaSize = used + kGuard;

    if (!n.allocateArena(memoryProperties, tokensSize, headSize, tokensAt, headAt, error)) return false;

    // parameter blocks (hip_net.cpp's wiring)
    std::vector<LayerParamsK> baseParams(11);
    for (uint32_t i = 0; i < 11; ++i) {
        const auto& p = KPlan[i];
        auto& q = baseParams[i]; std::memset(&q, 0, sizeof(q));
        q.W = int32_t(tw / p.divisor); q.H = int32_t(th / p.divisor);
        q.in = n.base + (i == 0 ? tokensAt : act[i - 1]);
        q.skip = i >= 6 ? n.base + full[10 - i] : 0;
        q.out24 = n.base + (i + 1 == 11 ? headAt : act[i]);
        q.out32 = i < 5 ? n.base + full[i] : 0;
        kShift(p.role, 0, q.sx, q.sy);
        q.w = n.base + wAt[i];
    }
    // pipelines: one per kernel and launch shape
    std::map<std::string, VkPipeline> byKey;
    auto pipe = [&](uint32_t layer, const std::string& kernel, Launch launch) -> VkPipeline {
        const std::string key = kernel + "/" + std::to_string(launch.bx) + "/" + std::to_string(launch.by) + "/" + std::to_string(launch.bz);
        auto it = byKey.find(key);
        if (it != byKey.end()) return it->second;
        auto k = modules[layer].kernels.find(kernel);
        if (k == modules[layer].kernels.end()) { error = kernel + " missing"; return VK_NULL_HANDLE; }
        const uint64_t id = nextPipelineId.fetch_add(1, std::memory_order_relaxed);
        if (id > UINT32_MAX) { error = "vkhip pipeline key space exhausted"; return VK_NULL_HANDLE; }
        VkPipeline p = createPipeline(n.vk, n.layout, modules[layer], k->second, launch, n.base + gAt[layer], uint32_t(id), error);
        if (p) { n.pipelines.push_back(p); byKey[key] = p; }
        return p;
    };
    std::vector<uint8_t> args;
    auto argBlock = [&](const LayerParamsK& q) {
        const uint64_t at = argsAt + args.size();
        args.resize(args.size() + 256);
        std::memcpy(args.data() + args.size() - 256, &q, sizeof(q));
        return n.base + at;
    };
    for (uint32_t i = 0; i < 11; ++i) {
        const std::string prep = names[i] + "_prep";
        if (!modules[i].kernels.count(prep)) continue;
        const uint32_t blocks = global32(modules[i], "d4r_prep_blocks", 0);
        if (!blocks) { error = names[i] + ": prep kernel without d4r_prep_blocks"; return false; }
        VkPipeline p = pipe(i, prep, {128, 1, 1});
        if (!p) return false;
        n.preps.push_back({p, blocks, 1, argBlock(baseParams[i]), prep});
    }
    const char* trimEnv = std::getenv("D4R_ENGINE_K_TRIM");
    const bool trim = !trimEnv || std::strcmp(trimEnv, "0") != 0;
    for (uint32_t phase = 0; phase < 8; ++phase) {
        int32_t size[2][11], shift[2][11];
        for (uint32_t i = 0; i < 11; ++i) {
            size[0][i] = baseParams[i].W; size[1][i] = baseParams[i].H;
            kShift(KPlan[i].role, phase, shift[0][i], shift[1][i]);
        }
        uint32_t groups[2][11];
        trimmedGroups(size, shift, {width, height}, groups);
        for (uint32_t i = 0; i < 11; ++i) {
            const auto& m = modules[i];
            LayerParamsK params = baseParams[i];
            kShift(KPlan[i].role, phase, params.sx, params.sy);
            const uint32_t blockZ = global32(m, "d4r_block_z", 4);
            std::vector<std::string> posts;
            for (int j = 1; j <= 8; ++j) {
                const std::string t = names[i] + "_post" + std::to_string(j);
                if (!m.kernels.count(t)) break;
                posts.push_back(t);
            }
            const uint64_t a = argBlock(params);
            if (!posts.empty()) {
                const uint32_t gx = global32(m, "d4r_grid_x", 0), gy = global32(m, "d4r_grid_y", 1);
                if (!gx) { error = names[i] + ": phase kernel without d4r_grid_x"; return false; }
                VkPipeline p = pipe(i, names[i], {32, 1, blockZ});
                if (!p) return false;
                n.nodes[phase].push_back({p, gx, gy, a, KPlan[i].name});
                for (const auto& post : posts) {
                    const std::string s = post.substr(names[i].size());
                    uint32_t pgx = global32(m, "d4r" + s + "_grid_x", 0);
                    if (!pgx) pgx = gx;
                    VkPipeline pp = pipe(i, post, {global32(m, "d4r" + s + "_block_x", 32), global32(m, "d4r" + s + "_block_y", 1),
                                                    global32(m, "d4r" + s + "_block_z", 1)});
                    if (!pp) return false;
                    n.nodes[phase].push_back({pp, pgx, global32(m, "d4r" + s + "_grid_y", 1), a, KPlan[i].name + s});
                }
            } else {
                VkPipeline p = pipe(i, names[i], {32, 1, blockZ});
                if (!p) return false;
                const uint32_t gx = trim ? groups[0][i] : uint32_t(params.W + params.sx + 7) / 8;
                const uint32_t gy = trim ? groups[1][i] : uint32_t(params.H + params.sy + 7) / 8;
                n.nodes[phase].push_back({p, gx, gy, a, KPlan[i].name});
            }
        }
    }
    if (args.size() > 256 * (11 + 8 * 32)) { error = "kernarg space"; return false; }
    // first-record contents: weights, the globals' initialized data, the kernarg blocks
    for (uint32_t i = 0; i < 11; ++i)
        n.contents.push_back({wAt[i], std::vector<uint8_t>(file.begin() + weightAt[i], file.begin() + weightAt[i] + weightBytes[i])});
    for (uint32_t i = 0; i < 11; ++i) {
        auto data = modules[i].dataInit;
        while (!data.empty() && data.back() == 0) data.pop_back();
        if (!data.empty()) n.contents.push_back({gAt[i], std::move(data)});
    }
    n.contents.push_back({argsAt, std::move(args)});
    return true;
}

// Preset M: hip_net.cpp's plan. Ten layers (one code object shared by the six tube blocks, with one weight image per
// weights pointer in its globals) in two window-alignment phases: NGX alternates the horizontal alignment every evaluation.
bool KNetwork::createM(VkDevice device, PFN_vkGetDeviceProcAddr proc, const VkPhysicalDeviceMemoryProperties& memoryProperties,
                       const NetFiles& files, uint32_t width, uint32_t height, uint64_t tokenBytes, uint64_t headBytes,
                       std::string& error) {
    auto& n = *impl;
    n.device = device;
    n.memoryProperties = memoryProperties;
    if (!n.loadEntries(proc)) { error = "missing Vulkan entry points"; return false; }
    constexpr uint32_t Layers = MLayerCount;
    const auto& file = files.hipnet;
    uint32_t count = 0;
    if (file.size() < 16 + 64 * Layers || std::memcmp(file.data(), "D4RMHIP1", 8)) { error = "not a preset M hipnet.bin"; return false; }
    std::memcpy(&count, file.data() + 8, 4);
    if (count != Layers) { error = "hipnet.bin does not have ten layers"; return false; }
    if (width < 64 || height < 64 || width > 8192 || height > 8192) { error = "render dimensions must be 64..8192"; return false; }
    const uint32_t tw = (width + 31) / 32 * 16, th = (height + 31) / 32 * 16;
    std::vector<Module> modules;
    std::vector<std::string> kernelNames;
    uint32_t layerKernel[Layers]; int32_t shiftX[Layers], shiftY[Layers]; uint32_t divisor[Layers];
    uint64_t weightAt[Layers], weightBytes[Layers];
    std::string label[Layers];
    for (uint32_t i = 0; i < Layers; ++i) {
        const uint8_t* e = file.data() + 16 + 64 * i;
        char name[33]{}; std::memcpy(name, e, 32);
        std::memcpy(&shiftX[i], e + 32, 4); std::memcpy(&shiftY[i], e + 36, 4); std::memcpy(&divisor[i], e + 40, 4);
        std::memcpy(&weightAt[i], e + 48, 8); std::memcpy(&weightBytes[i], e + 56, 8);
        if (!divisor[i] || weightAt[i] > file.size() || weightBytes[i] > file.size() - weightAt[i]) { error = "invalid hipnet.bin layer"; return false; }
        label[i] = std::string(name).substr(7) + "." + std::to_string(i);  // past "rrlite_"
        uint32_t k = 0;
        while (k < kernelNames.size() && kernelNames[k] != name) ++k;
        if (k == kernelNames.size()) {
            auto it = files.code.find(name);
            if (it == files.code.end()) { error = std::string(name) + ".hsaco missing"; return false; }
            modules.emplace_back();
            if (!modules.back().parse(it->second)) { error = std::string(name) + ": " + modules.back().error; return false; }
            if (modules.back().mach != (files.mach ? files.mach : modules[0].mach)) {
                error = std::string(name) + ".hsaco was built for another GPU (rebuild the engine models)"; return false;
            }
            kernelNames.emplace_back(name);
        }
        layerKernel[i] = k;
    }
    for (uint32_t k = 0; k < kernelNames.size(); ++k)
        if (kernelNames[k].find("tube") != std::string::npos && global32(modules[k], "d4r_prep_key_slots", 0) < 6) {
            error = kernelNames[k] + ": fewer than six weight image slots"; return false;
        }
    uint64_t used = 0;
    auto carve = [&](uint64_t bytes) { used = (used + kAlign - 1) / kAlign * kAlign; const uint64_t at = used; used += bytes; return at; };
    n.tokenBytes = tokenBytes; n.headBytes = headBytes;
    const uint64_t tokensAt = carve(tokenBytes + kTail), headAt = carve(headBytes + kTail);
    uint64_t wAt[Layers];
    for (uint32_t i = 0; i < Layers; ++i) wAt[i] = carve(weightBytes[i] + kWeightTail);
    // preset L: the stages' kernels, weights and pixel-resolution skip, and the network's own tokens and head
    const bool stages = !files.enc0lWeights.empty();
    uint32_t stageKernel[2] = {};
    uint64_t stageW[2] = {}, netTokens = tokensAt, netHead = headAt, skipL = 0;
    const uint32_t gw = (width + 31) / 32 * 32, gh = (height + 31) / 32 * 32;
    if (stages) {
        const std::vector<uint8_t>* const blobs[2] = {&files.enc0lWeights, &files.dec0lWeights};
        const char* const names[2] = {"rrlite_enc0l_net", "rrlite_dec0l_net"};
        for (int i = 0; i < 2; ++i) {
            auto it = files.code.find(names[i]);
            if (it == files.code.end() || blobs[i]->empty()) { error = std::string(names[i]) + " missing"; return false; }
            stageKernel[i] = uint32_t(modules.size());
            modules.emplace_back();
            if (!modules.back().parse(it->second)) { error = std::string(names[i]) + ": " + modules.back().error; return false; }
            if (modules.back().mach != (files.mach ? files.mach : modules[0].mach)) {
                error = std::string(names[i]) + ".hsaco was built for another GPU (rebuild the engine models)"; return false;
            }
            kernelNames.emplace_back(names[i]);
            stageW[i] = carve(blobs[i]->size() + kWeightTail);
        }
        const uint64_t planes = uint64_t(tw) * th * 64;
        netTokens = carve(planes + kTail); netHead = carve(planes + kTail); skipL = carve(uint64_t(gw) * gh * 32 + kTail);
    }
    const uint64_t skip1 = carve(uint64_t(tw) * th * 64 + kTail), merged1 = carve(uint64_t(tw / 2) * (th / 2) * 96 + kTail);
    const uint64_t skip2 = carve(uint64_t(tw / 2) * (th / 2) * 96 + kTail), merged2 = carve(uint64_t(tw / 4) * (th / 4) * 128 + kTail);
    const uint64_t tubeA = carve(uint64_t(tw / 4) * (th / 4) * 128 + kTail), tubeB = carve(uint64_t(tw / 4) * (th / 4) * 128 + kTail);
    const uint64_t dec2 = carve(uint64_t(tw / 2) * (th / 2) * 96 + kTail);
    std::vector<uint64_t> gAt(modules.size());
    for (size_t k = 0; k < modules.size(); ++k) gAt[k] = carve(modules[k].dataHi - modules[k].dataLo + kTail);
    const uint64_t argsAt = carve(256 * (Layers * 3 + 8));
    n.arenaSize = used + kGuard;
    if (!n.allocateArena(memoryProperties, tokenBytes, headBytes, tokensAt, headAt, error)) return false;
    // tokens -> enc1 -> enc2 -> six tube blocks (ping-pong) -> dec2 -> dec1
    const uint64_t in[Layers] = {netTokens, merged1, merged2, tubeA, tubeB, tubeA, tubeB, tubeA, tubeB, dec2};
    const uint64_t skip[Layers] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, skip2, skip1};
    const uint64_t out[Layers] = {skip1, skip2, tubeA, tubeB, tubeA, tubeB, tubeA, tubeB, dec2, netHead};
    const uint64_t merged[Layers] = {merged1, merged2, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
    LayerParamsM baseParams[Layers];
    const auto at = [&](uint64_t offset) { return offset == ~0ull ? 0ull : n.base + offset; };
    for (uint32_t i = 0; i < Layers; ++i) {
        auto& q = baseParams[i]; std::memset(&q, 0, sizeof(q));
        q.w = n.base + wAt[i]; q.sx = shiftX[i]; q.sy = shiftY[i];
        q.tw = int32_t(tw / divisor[i]); q.th = int32_t(th / divisor[i]);
        q.in = at(in[i]); q.skip = at(skip[i]); q.out = at(out[i]); q.merged = at(merged[i]);
    }
    std::map<std::string, VkPipeline> byKey;
    auto pipe = [&](uint32_t k, const std::string& kernel, Launch launch) -> VkPipeline {
        const std::string key = kernel + "/" + std::to_string(launch.bx) + "/" + std::to_string(launch.by) + "/" + std::to_string(launch.bz);
        auto it = byKey.find(key);
        if (it != byKey.end()) return it->second;
        auto found = modules[k].kernels.find(kernel);
        if (found == modules[k].kernels.end()) { error = kernel + " missing"; return VK_NULL_HANDLE; }
        const uint64_t id = nextPipelineId.fetch_add(1, std::memory_order_relaxed);
        if (id > UINT32_MAX) { error = "vkhip pipeline key space exhausted"; return VK_NULL_HANDLE; }
        VkPipeline p = createPipeline(n.vk, n.layout, modules[k], found->second, launch, n.base + gAt[k], uint32_t(id), error);
        if (p) { n.pipelines.push_back(p); byKey[key] = p; }
        return p;
    };
    std::vector<uint8_t> args;
    auto argBlock = [&](const auto& q) {
        const uint64_t offset = argsAt + args.size();
        args.resize(args.size() + 256);
        std::memcpy(args.data() + args.size() - 256, &q, sizeof(q));
        return n.base + offset;
    };
    // the input stage shifts by (2, 2) / (0, 2) and the expansion by (0, 0) / (2, 0), alternating every evaluation
    const auto stageParams = [&](int stage, uint32_t phase) {
        StageParamsL q{};
        q.w = n.base + stageW[stage]; q.tw = int32_t(gw); q.th = int32_t(gh);
        q.sx = stage == 0 ? 2 ^ int32_t(phase * 2) : int32_t(phase * 2); q.sy = stage == 0 ? 2 : 0;
        if (stage == 0) { q.a = n.base + tokensAt; q.b = n.base + skipL; q.c = n.base + netTokens; }
        else { q.a = n.base + netHead; q.b = n.base + skipL; q.c = n.base + headAt; }
        return q;
    };
    if (stages)
        for (int stage = 0; stage < 2; ++stage) {
            const uint32_t k = stageKernel[stage];
            const std::string prep = stage == 0 ? "rrlite_enc0l_net_prep" : "rrlite_dec0_4x4_prep";
            const uint32_t blocks = global32(modules[k], "d4r_prep_blocks", 0);
            if (!blocks) { error = kernelNames[k] + ": prep kernel without d4r_prep_blocks"; return false; }
            VkPipeline p = pipe(k, prep, {128, 1, 1});
            if (!p) return false;
            n.preps.push_back({p, blocks, 1, argBlock(stageParams(stage, 0)), prep});
        }
    // every layer's weight image, once: a shared code object gets one preparation per weights pointer
    for (uint32_t i = 0; i < Layers; ++i) {
        const uint32_t k = layerKernel[i];
        const std::string prep = kernelNames[k] + "_prep";
        if (!modules[k].kernels.count(prep)) continue;
        const uint32_t blocks = global32(modules[k], "d4r_prep_blocks", 0);
        if (!blocks) { error = kernelNames[k] + ": prep kernel without d4r_prep_blocks"; return false; }
        VkPipeline p = pipe(k, prep, {128, 1, 1});
        if (!p) return false;
        n.preps.push_back({p, blocks, 1, argBlock(baseParams[i]), prep});
    }
    n.phases = 2;
    const auto stageNode = [&](int stage, uint32_t phase) -> bool {
        const uint32_t k = stageKernel[stage];
        VkPipeline p = pipe(k, kernelNames[k], {32, 1, global32(modules[k], "d4r_block_z", 4)});
        if (!p) return false;
        const StageParamsL q = stageParams(stage, phase);
        // L's expansion: the blocks dec0_l_head.comp reads
        const uint32_t gx = stage == 1 ? std::min(uint32_t(q.tw + q.sx + 7) / 8, (width + uint32_t(q.sx) + 7) / 8) : uint32_t(q.tw + q.sx + 7) / 8;
        const uint32_t gy = stage == 1 ? std::min(uint32_t(q.th + q.sy + 7) / 8, (height + uint32_t(q.sy) + 7) / 8) : uint32_t(q.th + q.sy + 7) / 8;
        n.nodes[phase].push_back({p, gx, gy, argBlock(q), stage == 0 ? "enc0l" : "dec0l"});
        return true;
    };
    const char* trimEnv = std::getenv("D4R_ENGINE_M_TRIM");
    const bool trim = !trimEnv || std::strcmp(trimEnv, "0") != 0;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        if (stages && !stageNode(0, phase)) return false;
        int32_t size[2][Layers], shift[2][Layers];
        for (uint32_t i = 0; i < Layers; ++i) {
            size[0][i] = baseParams[i].tw; size[1][i] = baseParams[i].th;
            shift[0][i] = shiftX[i] ^ int32_t(phase * 2); shift[1][i] = shiftY[i];
        }
        uint32_t groups[2][Layers];
        trimmedGroupsM(size[0], shift[0], int(width), stages ? int(phase * 2) : -1, groups[0]);
        trimmedGroupsM(size[1], shift[1], int(height), stages ? 0 : -1, groups[1]);
        for (uint32_t i = 0; i < Layers; ++i) {
            const uint32_t k = layerKernel[i];
            LayerParamsM params = baseParams[i];
            params.sx = shiftX[i] ^ int32_t(phase * 2);
            VkPipeline p = pipe(k, kernelNames[k], {32, 1, global32(modules[k], "d4r_block_z", 4)});
            if (!p) return false;
            const uint32_t gx = trim ? groups[0][i] : uint32_t(params.tw + params.sx + 7) / 8, gy = trim ? groups[1][i] : uint32_t(params.th + params.sy + 7) / 8;
            n.nodes[phase].push_back({p, gx, gy, argBlock(params), label[i]});
        }
        if (stages && !stageNode(1, phase)) return false;
    }
    if (args.size() > 256 * (Layers * 3 + 8)) { error = "kernarg space"; return false; }
    for (uint32_t i = 0; i < Layers; ++i)
        n.contents.push_back({wAt[i], std::vector<uint8_t>(file.begin() + weightAt[i], file.begin() + weightAt[i] + weightBytes[i])});
    if (stages) {
        n.contents.push_back({stageW[0], files.enc0lWeights});
        n.contents.push_back({stageW[1], files.dec0lWeights});
    }
    for (size_t k = 0; k < modules.size(); ++k) {
        auto data = modules[k].dataInit;
        while (!data.empty() && data.back() == 0) data.pop_back();
        if (!data.empty()) n.contents.push_back({gAt[k], std::move(data)});
    }
    n.contents.push_back({argsAt, std::move(args)});
    return true;
}

void KNetwork::record(VkCommandBuffer cb, bool reset) {
    auto& n = *impl;
    if (!n.uploaded) {
        // the tokens were just written by the front: keep the token range out of the clear
        Impl::barrier(n.cmdBarrier, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        const uint64_t afterHead = n.contents.empty() ? 0 : std::min_element(n.contents.begin(), n.contents.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; })->first;
        n.cmdFill(cb, n.arena, afterHead, (n.arenaSize - afterHead) & ~3ull, 0);
        Impl::barrier(n.cmdBarrier, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        for (auto& [at, bytes] : n.contents) {
            bytes.resize((bytes.size() + 3) & ~size_t(3));
            for (size_t off = 0; off < bytes.size(); off += 65536)
                n.cmdUpdate(cb, n.arena, at + off, std::min<size_t>(65536, bytes.size() - off), bytes.data() + off);
            std::vector<uint8_t>().swap(bytes);
        }
        n.contents.clear();
        Impl::barrier(n.cmdBarrier, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        n.run(cb, n.preps);
        n.uploaded = true;
    }
    // The accuracy set follows the reference's reset schedule. Keep the evaluation count separate for
    // timestamps and dumps, and preserve the legacy window schedule for sets without this contract.
    if (reset && n.resetWindowCycle) n.windowFrame = 0;
    const bool dump = n.dumpBuffer && n.frame < n.dumpFrames;
    if (dump) {
        Impl::barrier(n.cmdBarrier, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        const VkBufferCopy copy{0, n.frame * (n.tokenBytes + n.headBytes), n.tokenBytes};
        n.cmdCopy(cb, n.tokens, n.dumpBuffer, 1, &copy);
        Impl::barrier(n.cmdBarrier, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    }
    const uint32_t dumpFrame = n.frame;
    const auto& list = n.nodes[n.windowFrame % n.phases];
    n.stampActive = -1;
    if (n.stampPool) {
        n.collectStamps(list);
        auto& slot = n.stampSlots[n.frame % Impl::StampSlots];
        if (!slot.pending && n.resetEvent(n.device, slot.done) == VK_SUCCESS) {
            n.stampActive = int(n.frame % Impl::StampSlots); n.stampNext = 0;
            slot.pending = true; slot.frame = n.frame;
            n.cmdResetQueries(cb, n.stampPool, uint32_t(n.stampActive) * n.stampCount, n.stampCount);
            n.stamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        }
    }
    n.run(cb, list);
    ++n.frame;
    ++n.windowFrame;
    Impl::barrier(n.cmdBarrier, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    if (dump) {
        Impl::barrier(n.cmdBarrier, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        const VkBufferCopy copy{0, dumpFrame * (n.tokenBytes + n.headBytes) + n.tokenBytes, n.headBytes};
        n.cmdCopy(cb, n.head, n.dumpBuffer, 1, &copy);
        Impl::barrier(n.cmdBarrier, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    }
    if (n.stampActive >= 0) {
        n.stamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        n.cmdSetEvent(cb, n.stampSlots[n.stampActive].done, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        n.stampActive = -1;
    }
}

void KNetwork::enableDump(const std::string& directory, uint32_t frames) {
    auto& n = *impl;
    if (n.dumpBuffer || directory.empty() || !frames || n.frame) return;
    frames = std::min(frames, 8u);
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = frames * (n.tokenBytes + n.headBytes); info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (n.createBuffer(n.device, &info, nullptr, &n.dumpBuffer) != VK_SUCCESS) { n.dumpBuffer = VK_NULL_HANDLE; return; }
    VkMemoryRequirements r; n.requirements(n.device, n.dumpBuffer, &r);
    uint32_t type = ~0u;
    for (uint32_t t = 0; t < n.memoryProperties.memoryTypeCount && type == ~0u; ++t)
        if ((r.memoryTypeBits & (1u << t)) && (n.memoryProperties.memoryTypes[t].propertyFlags &
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) type = t;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = r.size; mai.memoryTypeIndex = type;
    if (type == ~0u || n.allocate(n.device, &mai, nullptr, &n.dumpMemory) != VK_SUCCESS ||
        n.bind(n.device, n.dumpBuffer, n.dumpMemory, 0) != VK_SUCCESS) {
        n.destroyBuffer(n.device, n.dumpBuffer, nullptr); n.dumpBuffer = VK_NULL_HANDLE;
        if (n.dumpMemory) { n.freeMemory(n.device, n.dumpMemory, nullptr); n.dumpMemory = VK_NULL_HANDLE; }
        return;
    }
    n.dumpDir = directory; n.dumpFrames = frames;
}

void KNetwork::enableStamps(void (*log)(const char*), float timestampPeriod) {
    auto& n = *impl;
    if (n.stampPool || !log || !(timestampPeriod > 0) || n.nodes[0].empty()) return;
    n.nsPerTick = timestampPeriod;
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    n.stampCount = 2 * uint32_t(n.nodes[0].size()) + 2;
    qi.queryCount = Impl::StampSlots * n.stampCount;
    if (n.createQueries(n.device, &qi, nullptr, &n.stampPool) != VK_SUCCESS) { n.stampPool = VK_NULL_HANDLE; return; }
    for (auto& slot : n.stampSlots) {
        VkEventCreateInfo ei{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        if (n.createEvent(n.device, &ei, nullptr, &slot.done) != VK_SUCCESS) {
            for (auto& s : n.stampSlots) if (s.done) { n.destroyEvent(n.device, s.done, nullptr); s.done = VK_NULL_HANDLE; }
            n.destroyQueries(n.device, n.stampPool, nullptr); n.stampPool = VK_NULL_HANDLE;
            return;
        }
    }
    n.stampLog = log;
}

} // namespace vkhip
