// vkhip::KNetwork and vkhip::MNetwork: see vkhip_net.h. The plans (buffers, wiring, window cycle, launch shapes) are
// engine/hip_net.cpp's preset K and M plans; only the backend differs. detail::Net is the backend both share: one
// device-local arena (tokens and head aliased into it), the patched pipelines, the first-record upload and the
// per-frame dispatch lists.
#include "vkhip_net.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vkhip {
namespace {

struct LayerParamsK { int32_t W, H; uint64_t in, skip, out24, out32, pad40, pad48; int32_t sx, sy; uint64_t w; uint8_t rest[104]; };
static_assert(sizeof(LayerParamsK) == 176, "K params");
// kernels/m/swin_common.h: CommonParams, TubeParams (hip_net.cpp's LayerParamsM)
struct LayerParamsM {
    uint64_t w; int32_t sx, sy, tw, th; uint64_t in, skip, out, merged;
    int32_t r69, r70, r71, r72; uint64_t inFlags, outFlags;
};
static_assert(sizeof(LayerParamsM) == 88 && offsetof(LayerParamsM, in) == 24, "M params");
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

namespace detail {
struct Net {
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
    // descriptor set 0 of four storage images, for kernels that read its table (Kernel::table: RR F's dec0_1)
    PFN_vkCreateDescriptorSetLayout createSetLayout = nullptr; PFN_vkDestroyDescriptorSetLayout destroySetLayout = nullptr;
    PFN_vkCreateDescriptorPool createPool = nullptr; PFN_vkDestroyDescriptorPool destroyPool = nullptr;
    PFN_vkAllocateDescriptorSets allocateSets = nullptr; PFN_vkUpdateDescriptorSets updateSets = nullptr;
    PFN_vkCmdBindDescriptorSets cmdBindSets = nullptr;
    VkDescriptorSetLayout imageSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout imageLayout = VK_NULL_HANDLE;
    VkDescriptorPool imagePool = VK_NULL_HANDLE;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer arena = VK_NULL_HANDLE, tokens = VK_NULL_HANDLE, head = VK_NULL_HANDLE;
    uint64_t arenaSize = 0, base = 0;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<VkPipeline> pipelines;
    std::map<std::string, VkPipeline> byKey;
    // chained: no barrier after it; the next dispatch waits for its producers' flags itself (M's tube blocks)
    struct Dispatch { VkPipeline pipeline; uint32_t gx, gy; uint64_t args; std::string label; bool chained = false; };
    std::vector<Dispatch> preps, nodes[8];
    // contents written once, on the first record: (arena offset, bytes)
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> contents;
    std::vector<uint8_t> args;  // the kernarg blocks, 256 bytes each, at argsAt
    uint64_t argsAt = 0;
    uint64_t flagsAt = 0, flagsBytes = 0;  // the chained dispatches' completion flags, cleared every evaluation
    bool uploaded = false;
    uint32_t frame = 0, windowFrame = 0;
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

    ~Net() {
        writeDumps();
        if (dumpBuffer) destroyBuffer(device, dumpBuffer, nullptr);
        if (dumpMemory) freeMemory(device, dumpMemory, nullptr);
        for (auto& slot : stampSlots) if (slot.done) destroyEvent(device, slot.done, nullptr);
        if (stampPool) destroyQueries(device, stampPool, nullptr);
        for (VkPipeline p : pipelines) if (p) vk.destroyPipeline(device, p, nullptr);
        if (layout) destroyLayout(device, layout, nullptr);
        if (imageLayout) destroyLayout(device, imageLayout, nullptr);
        if (imagePool) destroyPool(device, imagePool, nullptr);
        if (imageSetLayout) destroySetLayout(device, imageSetLayout, nullptr);
        for (VkBuffer b : {tokens, head, arena}) if (b) destroyBuffer(device, b, nullptr);
        if (memory) freeMemory(device, memory, nullptr);
    }

    template <class F> bool load(PFN_vkGetDeviceProcAddr proc, F& f, const char* name) {
        f = reinterpret_cast<F>(proc(device, name));
        return f != nullptr;
    }
    bool loadEntryPoints(VkDevice d, PFN_vkGetDeviceProcAddr proc, const VkPhysicalDeviceMemoryProperties& properties,
                         bool binaries) {
        device = d;
        memoryProperties = properties;
        bool ok = vk.load(device, proc, binaries);
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
    VkBuffer buffer(uint64_t size, VkBufferUsageFlags usage) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size; info.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer b = VK_NULL_HANDLE;
        return createBuffer(device, &info, nullptr, &b) == VK_SUCCESS ? b : VK_NULL_HANDLE;
    }
    // The arena of `used` bytes (plus a guard) in one device-local allocation, with the tokens and the head aliased
    // into it at their offsets, its device address and the push-constant-only pipeline layout.
    bool allocateArena(uint64_t used, uint64_t tokensAt, uint64_t tokensSize, uint64_t headAt, uint64_t headSize, std::string& error) {
        tokenBytes = tokensSize; headBytes = headSize;
        arenaSize = used + kGuard;
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
    // The storage-image set layout (bindings 0 - 3, as engine/vkhip/donor_images.comp declares them), its pipeline
    // layout (that set and the 8-byte push constant) and a pool of `sets` sets
    bool imageSets(PFN_vkGetDeviceProcAddr proc, uint32_t sets, std::string& error) {
        if (!(load(proc, createSetLayout, "vkCreateDescriptorSetLayout") && load(proc, destroySetLayout, "vkDestroyDescriptorSetLayout") &&
              load(proc, createPool, "vkCreateDescriptorPool") && load(proc, destroyPool, "vkDestroyDescriptorPool") &&
              load(proc, allocateSets, "vkAllocateDescriptorSets") && load(proc, updateSets, "vkUpdateDescriptorSets") &&
              load(proc, cmdBindSets, "vkCmdBindDescriptorSets"))) { error = "missing Vulkan descriptor entry points"; return false; }
        VkDescriptorSetLayoutBinding b[4];
        for (uint32_t i = 0; i < 4; ++i) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; sli.bindingCount = 4; sli.pBindings = b;
        if (createSetLayout(device, &sli, nullptr, &imageSetLayout) != VK_SUCCESS) { error = "image set layout"; return false; }
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1; lci.pSetLayouts = &imageSetLayout; lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &range;
        if (createLayout(device, &lci, nullptr, &imageLayout) != VK_SUCCESS) { error = "image pipeline layout"; return false; }
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * sets};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pci.maxSets = sets; pci.poolSizeCount = 1; pci.pPoolSizes = &size;
        if (createPool(device, &pci, nullptr, &imagePool) != VK_SUCCESS) { error = "image descriptor pool"; return false; }
        return true;
    }
    // a set of the four storage images (GENERAL)
    VkDescriptorSet imageSet(const VkImageView (&views)[4], std::string& error) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = imagePool; ai.descriptorSetCount = 1; ai.pSetLayouts = &imageSetLayout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (allocateSets(device, &ai, &set) != VK_SUCCESS) { error = "image descriptor set"; return VK_NULL_HANDLE; }
        VkDescriptorImageInfo ii[4];
        VkWriteDescriptorSet w[4];
        for (uint32_t i = 0; i < 4; ++i) {
            ii[i] = {VK_NULL_HANDLE, views[i], VK_IMAGE_LAYOUT_GENERAL};
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[i].pImageInfo = &ii[i];
        }
        updateSets(device, 4, w, 0, nullptr);
        return set;
    }
    // one pipeline per kernel and launch shape
    VkPipeline pipe(const Module& module, uint64_t dataAt, const std::string& kernel, Launch launch, std::string& error) {
        const std::string key = kernel + "/" + std::to_string(launch.bx) + "/" + std::to_string(launch.by) + "/" + std::to_string(launch.bz);
        auto it = byKey.find(key);
        if (it != byKey.end()) return it->second;
        auto k = module.kernels.find(kernel);
        if (k == module.kernels.end()) { error = kernel + " missing"; return VK_NULL_HANDLE; }
        const uint64_t id = nextPipelineId.fetch_add(1, std::memory_order_relaxed);
        if (id > UINT32_MAX) { error = "vkhip pipeline key space exhausted"; return VK_NULL_HANDLE; }
        VkPipeline p = createPipeline(vk, k->second.table ? imageLayout : layout, module, k->second, launch, base + dataAt,
                                      uint32_t(id), error, k->second.table);
        if (p) { pipelines.push_back(p); byKey[key] = p; }
        return p;
    }
    template <class P> uint64_t argBlock(const P& q) {
        static_assert(sizeof(P) <= 256, "kernarg block");
        const uint64_t at = argsAt + args.size();
        args.resize(args.size() + 256);
        std::memcpy(args.data() + args.size() - 256, &q, sizeof(q));
        return base + at;
    }
    // the first record's contents: weights (added by the plan), every module's initialized data, the kernarg blocks
    void finishContents(const std::vector<const Module*>& modules, const std::vector<uint64_t>& dataAt) {
        for (size_t i = 0; i < modules.size(); ++i) {
            auto data = modules[i]->dataInit;
            while (!data.empty() && data.back() == 0) data.pop_back();
            if (!data.empty()) contents.push_back({dataAt[i], std::move(data)});
        }
        contents.push_back({argsAt, std::move(args)});
        args.clear();
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
            if (d.chained) continue;
            // the next layer's vector and scalar loads (SHADER_STORAGE_READ invalidates both caches on RADV)
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
    }
    // The first evaluation's arena clear (past the tokens and the head), upload and weight preparation
    void upload(VkCommandBuffer cb) {
        if (!uploaded) {
            // the tokens were just written by the front: keep the token range out of the clear
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            const uint64_t afterHead = contents.empty() ? 0 : std::min_element(contents.begin(), contents.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; })->first;
            cmdFill(cb, arena, afterHead, (arenaSize - afterHead) & ~3ull, 0);
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            for (auto& [at, bytes] : contents) {
                bytes.resize((bytes.size() + 3) & ~size_t(3));
                for (size_t off = 0; off < bytes.size(); off += 65536)
                    cmdUpdate(cb, arena, at + off, std::min<size_t>(65536, bytes.size() - off), bytes.data() + off);
                std::vector<uint8_t>().swap(bytes);
            }
            contents.clear();
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            run(cb, preps);
            uploaded = true;
        }
    }
    // One evaluation: on the first, the upload; the optional dumps and timestamps around the frame's dispatches.
    void record(VkCommandBuffer cb, const std::vector<Dispatch>& list) {
        upload(cb);
        const bool dump = dumpBuffer && frame < dumpFrames;
        if (dump) {
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            const VkBufferCopy copy{0, frame * (tokenBytes + headBytes), tokenBytes};
            cmdCopy(cb, tokens, dumpBuffer, 1, &copy);
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        }
        const uint32_t dumpFrame = frame;
        stampActive = -1;
        if (stampPool) {
            collectStamps(list);
            auto& slot = stampSlots[frame % StampSlots];
            if (!slot.pending && resetEvent(device, slot.done) == VK_SUCCESS) {
                stampActive = int(frame % StampSlots); stampNext = 0;
                slot.pending = true; slot.frame = frame;
                cmdResetQueries(cb, stampPool, uint32_t(stampActive) * stampCount, stampCount);
                stamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
            }
        }
        if (flagsBytes) {
            // this evaluation's flags start clear (the previous evaluation's dispatches completed behind its final barrier)
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            cmdFill(cb, arena, flagsAt, flagsBytes, 0);
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        run(cb, list);
        ++frame;
        ++windowFrame;
        barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        if (dump) {
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            const VkBufferCopy copy{0, dumpFrame * (tokenBytes + headBytes) + tokenBytes, headBytes};
            cmdCopy(cb, head, dumpBuffer, 1, &copy);
            barrier(cmdBarrier, cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        }
        if (stampActive >= 0) {
            stamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
            cmdSetEvent(cb, stampSlots[stampActive].done, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            stampActive = -1;
        }
    }

    void enableDump(const std::string& directory, uint32_t frames) {
        if (dumpBuffer || directory.empty() || !frames || frame) return;
        frames = std::min(frames, 8u);
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = frames * (tokenBytes + headBytes); info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (createBuffer(device, &info, nullptr, &dumpBuffer) != VK_SUCCESS) { dumpBuffer = VK_NULL_HANDLE; return; }
        VkMemoryRequirements r; requirements(device, dumpBuffer, &r);
        uint32_t type = ~0u;
        for (uint32_t t = 0; t < memoryProperties.memoryTypeCount && type == ~0u; ++t)
            if ((r.memoryTypeBits & (1u << t)) && (memoryProperties.memoryTypes[t].propertyFlags &
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) type = t;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = r.size; mai.memoryTypeIndex = type;
        if (type == ~0u || allocate(device, &mai, nullptr, &dumpMemory) != VK_SUCCESS ||
            bind(device, dumpBuffer, dumpMemory, 0) != VK_SUCCESS) {
            destroyBuffer(device, dumpBuffer, nullptr); dumpBuffer = VK_NULL_HANDLE;
            if (dumpMemory) { freeMemory(device, dumpMemory, nullptr); dumpMemory = VK_NULL_HANDLE; }
            return;
        }
        dumpDir = directory; dumpFrames = frames;
    }
    void enableStamps(void (*log)(const char*), float timestampPeriod) {
        if (stampPool || !log || !(timestampPeriod > 0) || nodes[0].empty()) return;
        nsPerTick = timestampPeriod;
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        stampCount = 2 * uint32_t(nodes[0].size()) + 2;
        qi.queryCount = StampSlots * stampCount;
        if (createQueries(device, &qi, nullptr, &stampPool) != VK_SUCCESS) { stampPool = VK_NULL_HANDLE; return; }
        for (auto& slot : stampSlots) {
            VkEventCreateInfo ei{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
            if (createEvent(device, &ei, nullptr, &slot.done) != VK_SUCCESS) {
                for (auto& s : stampSlots) if (s.done) { destroyEvent(device, s.done, nullptr); s.done = VK_NULL_HANDLE; }
                destroyQueries(device, stampPool, nullptr); stampPool = VK_NULL_HANDLE;
                return;
            }
        }
        stampLog = log;
    }
};
} // namespace detail

KNetwork::KNetwork() : impl(std::make_unique<detail::Net>()) {}
KNetwork::~KNetwork() = default;
VkBuffer KNetwork::tokens() const { return impl->tokens; }
VkBuffer KNetwork::head() const { return impl->head; }
uint32_t KNetwork::dispatchesPerFrame() const { return uint32_t(impl->nodes[0].size()); }
void KNetwork::enableDump(const std::string& directory, uint32_t frames) { impl->enableDump(directory, frames); }
void KNetwork::enableStamps(void (*log)(const char*), float timestampPeriod) { impl->enableStamps(log, timestampPeriod); }

bool KNetwork::create(VkDevice device, PFN_vkGetDeviceProcAddr proc, const VkPhysicalDeviceMemoryProperties& memoryProperties,
                      const NetFiles& files, uint32_t width, uint32_t height, uint64_t tokenBytes, uint64_t headBytes,
                      std::string& error) {
    auto& n = *impl;
    if (!n.loadEntryPoints(device, proc, memoryProperties, files.pipelineBinary)) { error = "missing Vulkan entry points"; return false; }
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
        // gfx12: enc0 / dec0 built without the attention -> residual barrier (pwin_pos.h POS_HB_SYNC) race between
        // waves, rarely on HIP but in 5-25% of 600-frame 4K runs here (their spills skew the waves); such packages
        // stay on the engine's network unless D4R_VKHIP_SCRATCH=1
        const bool gfx12 = modules[i].mach == gfx12Mach(120000) || modules[i].mach == gfx12Mach(120001);
        const char* scratchEnv = std::getenv("D4R_VKHIP_SCRATCH");
        if (gfx12 && (names[i] == "dltss_pwin_enc0_layer" || names[i] == "dltss_pwin_dec0_layer") &&
            global32(modules[i], "d4r_pos_hb_sync", 0) == 0 && !(scratchEnv && scratchEnv[0] == '1')) {
            error = names[i] + ".hsaco predates the gfx12 cross-wave barrier (rebuild with kernels/build.sh k on Windows)";
            return false;
        }
        if (modules[i].mach != (files.mach ? files.mach : modules[0].mach)) {
            error = names[i] + ".hsaco was built for another GPU (rebuild the engine models)"; return false;
        }
    }

    // the pooled layout: tokens, head, weights, activations, per-module globals, kernarg blocks
    uint64_t used = 0;
    auto carve = [&](uint64_t bytes) { used = (used + kAlign - 1) / kAlign * kAlign; const uint64_t at = used; used += bytes; return at; };
    const uint64_t planTokens = uint64_t(tw) * th * 16 * 2, planHead = uint64_t(tw) * th * 40 * 2;
    const uint64_t tokensSize = std::max(planTokens, tokenBytes), headSize = std::max(planHead, headBytes);
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
    n.argsAt = carve(256 * (11 + 8 * 32));
    if (!n.allocateArena(used, tokensAt, tokensSize, headAt, headSize, error)) return false;

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
    auto pipe = [&](uint32_t layer, const std::string& kernel, Launch launch) {
        return n.pipe(modules[layer], gAt[layer], kernel, launch, error);
    };
    for (uint32_t i = 0; i < 11; ++i) {
        const std::string prep = names[i] + "_prep";
        if (!modules[i].kernels.count(prep)) continue;
        const uint32_t blocks = global32(modules[i], "d4r_prep_blocks", 0);
        if (!blocks) { error = names[i] + ": prep kernel without d4r_prep_blocks"; return false; }
        VkPipeline p = pipe(i, prep, {128, 1, 1});
        if (!p) return false;
        n.preps.push_back({p, blocks, 1, n.argBlock(baseParams[i]), prep});
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
            const uint64_t a = n.argBlock(params);
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
    if (n.args.size() > 256 * (11 + 8 * 32)) { error = "kernarg space"; return false; }
    for (uint32_t i = 0; i < 11; ++i)
        n.contents.push_back({wAt[i], std::vector<uint8_t>(file.begin() + weightAt[i], file.begin() + weightAt[i] + weightBytes[i])});
    std::vector<const Module*> all;
    for (const auto& m : modules) all.push_back(&m);
    n.finishContents(all, gAt);
    return true;
}

void KNetwork::record(VkCommandBuffer cb, bool reset) {
    auto& n = *impl;
    // The accuracy set follows the reference's reset schedule. Keep the evaluation count separate for
    // timestamps and dumps, and preserve the legacy window schedule for sets without this contract.
    if (reset && n.resetWindowCycle) n.windowFrame = 0;
    n.record(cb, n.nodes[n.windowFrame % 8]);
}

MNetwork::MNetwork() : impl(std::make_unique<detail::Net>()) {}
MNetwork::~MNetwork() = default;
VkBuffer MNetwork::tokens() const { return impl->tokens; }
VkBuffer MNetwork::head() const { return impl->head; }
uint32_t MNetwork::dispatchesPerFrame() const { return uint32_t(impl->nodes[0].size()); }
void MNetwork::enableDump(const std::string& directory, uint32_t frames) { impl->enableDump(directory, frames); }
void MNetwork::enableStamps(void (*log)(const char*), float timestampPeriod) { impl->enableStamps(log, timestampPeriod); }

bool MNetwork::create(VkDevice device, PFN_vkGetDeviceProcAddr proc, const VkPhysicalDeviceMemoryProperties& memoryProperties,
                      const NetFiles& files, uint32_t renderWidth, uint32_t renderHeight, uint64_t bufferBytes,
                      std::string& error) {
    constexpr uint32_t Layers = 10;
    auto& n = *impl;
    if (!n.loadEntryPoints(device, proc, memoryProperties, files.pipelineBinary)) { error = "missing Vulkan entry points"; return false; }
    const auto& file = files.hipnet;
    uint32_t count = 0;
    if (file.size() < 16 + 64 * Layers || std::memcmp(file.data(), "D4RMHIP1", 8)) { error = "not an M/L hipnet.bin"; return false; }
    std::memcpy(&count, file.data() + 8, 4);
    if (count != Layers) { error = "hipnet.bin does not have ten layers"; return false; }
    if (renderWidth < 64 || renderHeight < 64 || renderWidth > 16384 || renderHeight > 16384) { error = "render dimensions must be 64..16384"; return false; }
    // MEngine's token pixel grid halved: one token per 2x2 render pixels, the grid aligned to 32 pixels
    const uint32_t tw = (renderWidth + 31) / 32 * 16, th = (renderHeight + 31) / 32 * 16;
    if (bufferBytes < uint64_t(tw) * th * 64) { error = "token buffers smaller than the plan"; return false; }

    // code objects: the tube's six layers share one (and its data image, where the six prepared weight sets live)
    std::vector<Module> modules;
    modules.reserve(Layers);
    std::vector<std::string> names;
    std::vector<uint32_t> layerModule(Layers), divisor(Layers);
    std::vector<int32_t> shiftX(Layers), shiftY(Layers);
    std::vector<uint64_t> weightAt(Layers), weightBytes(Layers);
    for (uint32_t i = 0; i < Layers; ++i) {
        const uint8_t* e = file.data() + 16 + 64 * i;
        char name[33]{}; std::memcpy(name, e, 32);
        std::memcpy(&shiftX[i], e + 32, 4); std::memcpy(&shiftY[i], e + 36, 4); std::memcpy(&divisor[i], e + 40, 4);
        std::memcpy(&weightAt[i], e + 48, 8); std::memcpy(&weightBytes[i], e + 56, 8);
        if (!divisor[i] || weightAt[i] > file.size() || weightBytes[i] > file.size() - weightAt[i]) { error = "invalid hipnet.bin layer"; return false; }
        const auto known = std::find(names.begin(), names.end(), name);
        if (known != names.end()) { layerModule[i] = uint32_t(known - names.begin()); continue; }
        auto it = files.code.find(name);
        if (it == files.code.end()) { error = std::string(name) + ".hsaco missing"; return false; }
        layerModule[i] = uint32_t(modules.size());
        names.emplace_back(name); modules.emplace_back();
        auto& m = modules.back();
        if (!m.parse(it->second)) { error = std::string(name) + ": " + m.error; return false; }
        if (m.mach != (files.mach ? files.mach : modules[0].mach)) {
            error = std::string(name) + ".hsaco was built for another GPU (rebuild the engine models)"; return false;
        }
        if (!m.kernels.count(name)) { error = std::string(name) + " missing"; return false; }
    }
    for (uint32_t i = 0; i < Layers; ++i) {
        const auto& m = modules[layerModule[i]];
        const uint32_t uses = uint32_t(std::count(layerModule.begin(), layerModule.end(), layerModule[i]));
        // a module keeps one prepared weight image per weights pointer while its slots last
        if (uses > 1 && global32(m, "d4r_prep_key_slots", 0) < uses) { error = names[layerModule[i]] + ": too few weight image slots"; return false; }
    }

    // Barrier-free chains (kernels/m/swin_tok.h in_flags/out_flags): block (x, y) of a layer sets its 32-byte flag
    // once its outputs are written, and a block of the next layer waits for the producer blocks that hold its
    // input. A dispatch's blocks all launch before the next dispatch's, so the waits cannot starve their producers.
    // Layers built with SWIN_DATAFLOW (d4r_dataflow) chain all ten layers; otherwise the six tube blocks chain with
    // the kernel's original 2x2 wait. D4R_VKHIP_M_CHAIN=0 keeps a barrier after every layer.
    const char* chainEnv = std::getenv("D4R_VKHIP_M_CHAIN");
    const bool chain = !chainEnv || std::atoi(chainEnv) != 0;
    bool dataflow = chain;
    for (const auto& m : modules) dataflow = dataflow && global32(m, "d4r_dataflow", 0) == 1;
    const uint32_t first = dataflow ? 0 : 2, last = dataflow ? Layers - 1 : 7;  // the chained range

    // the pooled layout: tokens, head, weights, activations (hip_net.cpp's M buffers, FP8), module data, kernarg blocks
    uint64_t used = 0;
    auto carve = [&](uint64_t bytes) { used = (used + kAlign - 1) / kAlign * kAlign; const uint64_t at = used; used += bytes; return at; };
    const uint64_t tokensAt = carve(bufferBytes + kTail), headAt = carve(bufferBytes + kTail);
    std::vector<uint64_t> wAt(Layers);
    for (uint32_t i = 0; i < Layers; ++i) wAt[i] = carve(weightBytes[i] + kTail);
    auto tok = [&](uint32_t scale, uint32_t channels) { return carve(uint64_t(tw / scale) * (th / scale) * channels + kTail); };
    const uint64_t skip1 = tok(1, 64), merged1 = tok(2, 96), skip2 = tok(2, 96), merged2 = tok(4, 128);
    const uint64_t tubeA = tok(4, 128), tubeB = tok(4, 128), dec2 = tok(2, 96);
    std::vector<uint64_t> gAt(modules.size());
    for (size_t k = 0; k < modules.size(); ++k) gAt[k] = carve(modules[k].dataHi - modules[k].dataLo + kTail);
    uint64_t flagsAt[Layers] = {};
    if (chain) {
        auto grid = [&](uint32_t i) { return uint64_t((tw / divisor[i] + 2 + 7) / 8 + 1) * ((th / divisor[i] + 2 + 7) / 8 + 1) * 32; };
        uint64_t bytes = 0;
        for (uint32_t i = first; i < last; ++i) bytes += grid(i);
        n.flagsAt = carve(bytes);
        n.flagsBytes = bytes;
        for (uint32_t i = first, at = 0; i < last; at += uint32_t(grid(i)), ++i) flagsAt[i] = n.flagsAt + at;
    }
    n.argsAt = carve(256 * Layers * 3);
    if (!n.allocateArena(used, tokensAt, bufferBytes, headAt, bufferBytes, error)) return false;

    // tokens -> enc1 -> enc2 -> six tube blocks (ping-pong) -> dec2 -> dec1 (hip_net.cpp's wiring)
    const uint64_t in[Layers] = {tokensAt, merged1, merged2, tubeA, tubeB, tubeA, tubeB, tubeA, tubeB, dec2};
    const uint64_t skip[Layers] = {0, 0, 0, 0, 0, 0, 0, 0, skip2, skip1};
    const uint64_t out[Layers] = {skip1, skip2, tubeA, tubeB, tubeA, tubeB, tubeA, tubeB, dec2, headAt};
    const uint64_t merged[Layers] = {merged1, merged2, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<LayerParamsM> baseParams(Layers);
    for (uint32_t i = 0; i < Layers; ++i) {
        auto& q = baseParams[i]; std::memset(&q, 0, sizeof(q));
        q.w = n.base + wAt[i]; q.sx = shiftX[i]; q.sy = shiftY[i];
        q.tw = int32_t(tw / divisor[i]); q.th = int32_t(th / divisor[i]);
        q.in = n.base + in[i]; q.skip = skip[i] ? n.base + skip[i] : 0;
        q.out = n.base + out[i]; q.merged = merged[i] ? n.base + merged[i] : 0;
    }
    auto pipe = [&](uint32_t layer, const std::string& kernel, Launch launch) {
        return n.pipe(modules[layerModule[layer]], gAt[layerModule[layer]], kernel, launch, error);
    };
    // one preparation per layer, i.e. per distinct weights pointer (the tube: six)
    for (uint32_t i = 0; i < Layers; ++i) {
        const auto& m = modules[layerModule[i]];
        const std::string prep = names[layerModule[i]] + "_prep";
        if (!m.kernels.count(prep)) continue;
        const uint32_t blocks = global32(m, "d4r_prep_blocks", 0);
        if (!blocks) { error = prep + " without d4r_prep_blocks"; return false; }
        VkPipeline p = pipe(i, prep, {128, 1, 1});
        if (!p) return false;
        n.preps.push_back({p, blocks, 1, n.argBlock(baseParams[i]), prep});
    }
    static const char* label[Layers] = {"enc1", "enc2", "tube0", "tube1", "tube2", "tube3", "tube4", "tube5", "dec2", "dec1"};
    // NGX alternates the horizontal window alignment every evaluation
    for (uint32_t phase = 0; phase < 2; ++phase)
        for (uint32_t i = 0; i < Layers; ++i) {
            LayerParamsM params = baseParams[i];
            params.sx = shiftX[i] ^ int32_t(phase * 2);
            const auto& m = modules[layerModule[i]];
            VkPipeline p = pipe(i, names[layerModule[i]], {32, 1, global32(m, "d4r_block_z", 4)});
            if (!p) return false;
            const uint32_t gx = uint32_t(params.tw + params.sx + 7) / 8, gy = uint32_t(params.th + params.sy + 7) / 8;
            const bool chained = chain && i >= first && i <= last;
            if (chained && i < last) params.outFlags = n.base + flagsAt[i];
            if (chained && i > first) {
                // the producer grid, and where this block's input lies in it: block x covers tokens 8 x - sx ..
                // 8 x - sx + 7 of its own layer
                const int32_t psx = shiftX[i - 1] ^ int32_t(phase * 2), psy = shiftY[i - 1];
                const int32_t ptw = int32_t(tw / divisor[i - 1]), pth = int32_t(th / divisor[i - 1]);
                params.inFlags = n.base + flagsAt[i - 1];
                params.r69 = (ptw + psx + 7) / 8; params.r70 = (pth + psy + 7) / 8;
                if (dataflow) {
                    // mapping: 1 a patch merge (the producer at twice the resolution), 2 a patch expand (half)
                    const int32_t mode = divisor[i - 1] < divisor[i] ? 1 : divisor[i - 1] > divisor[i] ? 2 : 0;
                    params.r71 = psx | mode << 8; params.r72 = psy | mode << 8;
                } else {
                    params.r71 = psx < params.sx ? -1 : 0; params.r72 = psy < params.sy ? -1 : 0;
                }
            }
            n.nodes[phase].push_back({p, gx, gy, n.argBlock(params), label[i], chained && i < last});
        }
    for (uint32_t i = 0; i < Layers; ++i)
        n.contents.push_back({wAt[i], std::vector<uint8_t>(file.begin() + weightAt[i], file.begin() + weightAt[i] + weightBytes[i])});
    std::vector<const Module*> all;
    for (const auto& m : modules) all.push_back(&m);
    n.finishContents(all, gAt);
    return true;
}

void MNetwork::record(VkCommandBuffer cb) {
    auto& n = *impl;
    n.record(cb, n.nodes[n.frame % 2]);
}

// ---------------------------------------------------------------- Ray Reconstruction presets D and E
namespace {
// kernels/rr/rrvk_swin.hip's Args (passed by value: the push constant is the address of this block)
struct RRArgs {
    uint64_t arena, w;
    int32_t ex, ey;
    uint32_t inOff, skOff, outOff, mrgOff, shift;
    float iw, ih;
    uint32_t biasF;   // the attention bias as f32 ([head][query][key]) after the packed weights
    alignas(32) int32_t t88[8];
    int32_t t96[8], t104[8], s112[8], s120[8], s128[8];
    alignas(16) int32_t linear[4];
};
static_assert(offsetof(RRArgs, t88) == 64 && offsetof(RRArgs, linear) == 256 && sizeof(RRArgs) == 288, "RR args");
constexpr const char* RRNames[] = {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec4", "dec3", "dec2", "dec1", "dec0"};
float halfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
    float v = e == 0 ? std::ldexp(float(m), -24) : (e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(m | 1024u), int(e) - 25));
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    bits |= sign;
    std::memcpy(&v, &bits, 4);
    return v;
}
// the two window shifts NGX alternates (engine/rr_frame_parameters.h rShift)
constexpr uint32_t RRShifts[2] = {0x00060002u, 0x00020006u};
} // namespace

struct RRStages::Stage {
    unsigned index = 0;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint64_t args[2] = {};   // per window shift (RRShifts)
    VkDescriptorSet set = VK_NULL_HANDLE;  // Kernel::table: descriptor set 0 (Net::imageSet)
    uint32_t groups[2] = {}; // presets A - C: the stage's own grid (its tile height differs from the engine shader's)
};

RRStages::RRStages() : impl(std::make_unique<detail::Net>()) {}
RRStages::~RRStages() {
    if (linear) destroySampler(impl->device, linear, nullptr);
}
uint32_t RRStages::stageCount() const { return uint32_t(stages.size()); }

bool RRStages::create(VkPhysicalDevice physical, PFN_vkGetInstanceProcAddr instanceProc, VkInstance instance, VkDevice device,
                      PFN_vkGetDeviceProcAddr proc, const VkPhysicalDeviceMemoryProperties& memoryProperties,
                      const NetFiles& files, const std::map<std::string, std::vector<uint8_t>>& weights,
                      d4r::REngine& engine, std::string& error) {
    auto& n = *impl;
    if (!n.loadEntryPoints(device, proc, memoryProperties, files.pipelineBinary)) { error = "missing Vulkan entry points"; return false; }
    // presets A - C (kernels/rr/rrcnn_vkhip.hip): the kernel takes its parameter block's address, which holds every
    // pointer (the arena's and the engine's weights); it exports its workgroup size and tile height
    const char* cnnNames[] = {"enc0", "enc1", "enc2", "enc3", "enc4", "dec4", "dec3", "dec2", "dec1", "dec0", "pdl"};
    {
        struct CLoaded { unsigned index; std::string name; Module module; uint64_t gAt; uint32_t threads, rows; };
        std::vector<CLoaded> loaded;
        uint64_t used = kAlign;
        auto carve = [&](uint64_t size) { uint64_t at = used; used = (at + size + kTail + kAlign - 1) / kAlign * kAlign; return at; };
        for (unsigned i = 0; i < 11; ++i) {
            const std::string name = std::string("rr3_") + cnnNames[i];
            auto code = files.code.find(name);
            if (code == files.code.end()) continue;
            CLoaded l{}; l.index = i; l.name = name;
            if (!l.module.parse(code->second)) { error = name + ": " + l.module.error; return false; }
            if (l.module.mach != gfx12Mach(120000) && l.module.mach != gfx12Mach(120001)) { error = "the CNN vkhip stages require RDNA4"; return false; }
            if (files.mach && l.module.mach != files.mach) { error = name + ".hsaco was built for another GPU"; return false; }
            auto k = l.module.kernels.find(name);
            if (k == l.module.kernels.end() || k->second.kernarg != 8) { error = name + ": unexpected kernel arguments"; return false; }
            for (auto [symbol, value] : {std::pair<const char*, uint32_t*>{"rr3_threads", &l.threads}, {"rr3_rows", &l.rows}}) {
                auto g = l.module.globals.find(symbol);
                if (g == l.module.globals.end() || g->second.va < l.module.dataLo || g->second.va + 4 > l.module.dataHi) {
                    error = name + ": no " + symbol; return false;
                }
                std::memcpy(value, l.module.dataInit.data() + g->second.va - l.module.dataLo, 4);
            }
            if (l.threads < 32 || l.threads > 1024 || l.threads % 32 || !l.rows) { error = name + ": invalid launch metadata"; return false; }
            l.gAt = carve(l.module.dataHi - l.module.dataLo);
            loaded.push_back(std::move(l));
        }
        if (!loaded.empty()) {
            n.argsAt = used;
            if (!n.allocateArena(n.argsAt + 512 * loaded.size(), 0, kAlign, 0, kAlign, error)) return false;
            std::vector<const Module*> modules;
            std::vector<uint64_t> globals;
            for (const auto& l : loaded) {
                const uint64_t param = engine.stageParameterAddress(l.index);
                if (!param) { error = "the CNN's parameters need a device address"; return false; }
                const d4r::RStageResources r = engine.stageResources(l.index);
                Stage st; st.index = l.index;
                st.pipeline = n.pipe(l.module, l.gAt, l.name, {l.threads, 1, 1}, error);
                if (!st.pipeline) return false;
                const size_t offset = n.args.size(); n.args.resize(offset + 512);
                std::memcpy(n.args.data() + offset, &param, sizeof(param));
                st.args[0] = st.args[1] = n.base + n.argsAt + offset;
                st.groups[0] = (r.width + 15) / 16; st.groups[1] = (r.height + l.rows - 1) / l.rows;
                stages.push_back(st);
                modules.push_back(&l.module); globals.push_back(l.gAt);
            }
            n.finishContents(modules, globals);
            return true;
        }
    }
    // F network blocks use live device-address parameters and arena guide planes.
    const char* fNames[] = {"enc0", "enc1", "enc2", "enc3", "enc4", "enc5", "dec4_0", "dec4_1",
                            "dec3_0", "dec3_1", "dec2_0", "dec2_1", "dec1_0", "dec1_1", "dec0_0", "dec0_1"};
    struct FLoaded { unsigned index; std::string name; Module module; std::vector<uint8_t> weights; uint64_t wAt, gAt; uint32_t threads; bool images; };
    // dec0_1 (kernels/rr/rr2_hybrid_prep.py) stores its four surfaces through storage images, descriptor set 0, whose
    // table the kernel reads itself (Kernel::table): only through pipeline binaries (AMD's driver), otherwise it stays
    // on the engine's shader
    std::vector<FLoaded> fLoaded;
    uint64_t fUsed = kAlign;
    auto fCarve = [&](uint64_t size) { uint64_t at = fUsed; fUsed = (at + size + kTail + kAlign - 1) / kAlign * kAlign; return at; };
    for (unsigned i = 0; i < 16; ++i) {
        const std::string name = std::string("rr2_") + fNames[i];
        auto code = files.code.find(name);
        if (code == files.code.end()) continue;
        FLoaded l{}; l.index = i; l.name = name;
        if (!l.module.parse(code->second)) { error = l.module.error; return false; }
        auto& module = l.module;
        if (module.mach != gfx12Mach(120000) && module.mach != gfx12Mach(120001)) {
            error = "preset F vkhip requires RDNA4"; return false;
        }
        if (files.mach && module.mach != files.mach) { error = "preset F code object targets another GPU"; return false; }
        auto k = module.kernels.find(name);
        if (k == module.kernels.end() || k->second.kernarg != 24 || (k->second.table && i != 15)) {
            error = "unexpected preset F kernel ABI"; return false;
        }
        l.images = k->second.table;
        if (l.images && !files.pipelineBinary) continue;
        auto w = weights.find(fNames[i]);
        if (w == weights.end()) { error = "missing preset F weights for " + name; return false; }
        l.weights = w->second;
        if (auto g = module.globals.find("rr2_v_layout"); g != module.globals.end()) {
            uint32_t v[4];
            if (g->second.size != sizeof(v) || g->second.va < module.dataLo || g->second.va + sizeof(v) > module.dataHi) {
                error = "invalid F V-weight layout"; return false;
            }
            std::memcpy(v, module.dataInit.data() + g->second.va - module.dataLo, sizeof(v));
            if (!v[0] || v[0] % 16 || !v[1] || uint64_t(v[2]) + uint64_t(v[3])*(v[1]-1) + uint64_t(v[0]/16)*512 > l.weights.size()) {
                error = "F V weights exceed model"; return false;
            }
            for (uint32_t h = 0; h < v[1]; ++h)
                for (uint32_t t = 0; t < v[0]/16*2; ++t) {
                    auto* tile = l.weights.data() + v[2] + v[3]*h + 256*t;
                    for (int r = 0; r < 16; ++r)
                        for (int c = 0; c < r; ++c) std::swap(tile[16*r+c], tile[16*c+r]);
                }
        }
        l.wAt = fCarve(l.weights.size());
        l.gAt = fCarve(module.dataHi - module.dataLo);
        l.threads = 32;
        if (auto t = module.globals.find("rr2_threads"); t != module.globals.end()) {
            if (t->second.va < module.dataLo || t->second.va + 4 > module.dataHi) { error = "invalid F launch metadata"; return false; }
            std::memcpy(&l.threads, module.dataInit.data() + t->second.va - module.dataLo, 4);
        }
        if (l.threads < 32 || l.threads > 1024 || l.threads % 32) { error = "invalid F workgroup size"; return false; }
        fLoaded.push_back(std::move(l));
    }
    if (!fLoaded.empty()) {
        n.argsAt = fUsed;
        if (!n.allocateArena(n.argsAt + 512 * fLoaded.size(), 0, kAlign, 0, kAlign, error)) return false;
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; ai.buffer = engine.arenaBuffer();
        const uint64_t arena = n.address(device, &ai);
        if (!arena) { error = "preset F arena needs a device address"; return false; }
        const uint32_t imageStages = uint32_t(std::count_if(fLoaded.begin(), fLoaded.end(), [](const FLoaded& l) { return l.images; }));
        if (imageStages && !n.imageSets(proc, imageStages, error)) return false;
        std::vector<const Module*> modules;
        std::vector<uint64_t> globals;
        for (const auto& l : fLoaded) {
            const uint64_t args[3] = {engine.stageParameterAddress(l.index), n.base + l.wAt, arena};
            if (!args[0]) { error = "preset F parameters need a device address"; return false; }
            const VkPipeline pipeline = n.pipe(l.module, l.gAt, l.name, {l.threads,1,1}, error);
            if (!pipeline) return false;
            const uint64_t at = n.base + n.argsAt + n.args.size();
            const size_t offset = n.args.size(); n.args.resize(offset + 512);
            std::memcpy(n.args.data() + offset, args, sizeof(args));
            Stage st; st.index = l.index; st.pipeline = pipeline; st.args[0] = st.args[1] = at;
            if (l.images) {   // surfaces 176, 184, 192, 200: the engine's images, the same for both window shifts
                const d4r::RStageResources r = engine.stageResources(l.index);
                for (VkImageView v : r.surfaceF) if (!v) { error = "preset F dec0_1 surfaces missing"; return false; }
                st.set = n.imageSet(r.surfaceF, error);
                if (!st.set) return false;
            }
            stages.push_back(st);
            n.contents.push_back({l.wAt, l.weights});
            modules.push_back(&l.module); globals.push_back(l.gAt);
        }
        n.finishContents(modules, globals);
        return true;
    }
    auto getDescriptor = reinterpret_cast<PFN_vkGetDescriptorEXT>(proc(device, "vkGetDescriptorEXT"));
    auto createSampler = reinterpret_cast<PFN_vkCreateSampler>(proc(device, "vkCreateSampler"));
    destroySampler = reinterpret_cast<PFN_vkDestroySampler>(proc(device, "vkDestroySampler"));
    auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(instanceProc(instance, "vkGetPhysicalDeviceProperties2"));
    if (!getDescriptor) { error = "VK_EXT_descriptor_buffer is not enabled (vkGetDescriptorEXT)"; return false; }
    if (!createSampler || !destroySampler || !properties2) { error = "missing Vulkan entry points"; return false; }
    VkPhysicalDeviceDescriptorBufferPropertiesEXT db{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &db};
    properties2(physical, &p2);
    // the kernels take AMD's 8-dword image and 4-dword sampler descriptors
    if (db.sampledImageDescriptorSize != 32 || db.storageImageDescriptorSize != 32 || db.samplerDescriptorSize != 16) {
        error = "unexpected descriptor sizes"; return false;
    }

    // the code objects, one per stage; a stage without one stays on the engine's shader
    struct Loaded { unsigned index; Module module; uint32_t layout[8]; std::vector<uint8_t> w; uint32_t biasF; };
    std::vector<Loaded> loaded;
    for (unsigned i = 0; i <= 10; ++i) {
        const std::string name = std::string("rr_") + RRNames[i];
        auto code = files.code.find(name);
        if (code == files.code.end()) continue;
        if (i == 0 && !engine.stageResources(0).scratchEnd) continue;   // no gather (enc0_g.spv): the engine's enc0
        Loaded l{i, {}, {}, {}, 0};
        if (!l.module.parse(code->second)) { error = name + ": " + l.module.error; return false; }
        if (files.mach && l.module.mach != files.mach) { error = name + ".hsaco was built for another GPU (rebuild the engine models)"; return false; }
        auto k = l.module.kernels.find(name);
        if (k == l.module.kernels.end()) { error = name + " missing"; return false; }
        if (k->second.kernarg < sizeof(RRArgs)) { error = name + ": unexpected kernel arguments"; return false; }
        auto g = l.module.globals.find("rrvk_layout");
        if (g == l.module.globals.end() || g->second.size != sizeof(l.layout) || g->second.va < l.module.dataLo ||
            g->second.va + sizeof(l.layout) > l.module.dataHi) { error = name + ": no rrvk_layout"; return false; }
        std::memcpy(l.layout, l.module.dataInit.data() + (g->second.va - l.module.dataLo), sizeof(l.layout));
        auto wf = weights.find(RRNames[i]);
        if (wf == weights.end()) { error = std::string("weights-") + RRNames[i] + "m.bin missing"; return false; }
        // rrvk_swin.hip's layout: the V tiles [n][k] instead of [k][n], and the attention bias as f32 [query][key]
        // (instead of binary16 [key][query]) after the packed weights, so that every operand is contiguous per lane and
        // the scores start from the bias without conversions
        l.w = wf->second;
        const uint32_t C = l.layout[1], NH = l.layout[3], wv0 = l.layout[5], bias0 = l.layout[6], head = l.layout[7];
        if (bias0 + uint64_t(head) * (NH - 1) + 96 * 64 * 2 > l.w.size() || wv0 + uint64_t(head) * (NH - 1) + 256 * (C / 8) > l.w.size()) {
            error = name + ": weights do not match its layout"; return false;
        }
        for (uint32_t hd = 0; hd < NH; ++hd) {
            uint8_t* v = l.w.data() + wv0 + head * hd;
            for (uint32_t t = 0; t < (C / 16) * 2; ++t)
                for (int r = 0; r < 16; ++r)
                    for (int c = 0; c < r; ++c) std::swap(v[256 * t + 16 * r + c], v[256 * t + 16 * c + r]);
        }
        l.biasF = uint32_t((l.w.size() + 255) / 256 * 256);
        std::vector<uint8_t> w = l.w;
        w.resize(l.biasF + size_t(NH) * 96 * 64 * 4);
        for (uint32_t hd = 0; hd < NH; ++hd) {
            uint16_t b[96 * 64];
            std::memcpy(b, l.w.data() + bias0 + head * hd, sizeof(b));
            float* o = reinterpret_cast<float*>(w.data() + l.biasF) + hd * 96 * 64;
            for (int key = 0; key < 96; ++key)
                for (int q = 0; q < 64; ++q) o[q * 96 + key] = halfToFloat(b[key * 64 + q]);
        }
        l.w = std::move(w);
        loaded.push_back(std::move(l));
    }
    if (loaded.empty()) { error = "no rr_<stage>.hsaco"; return false; }

    // the pooled layout: weights, module data, kernarg blocks (512 bytes each)
    uint64_t used = 0;
    auto carve = [&](uint64_t bytes) { used = (used + kAlign - 1) / kAlign * kAlign; const uint64_t at = used; used += bytes; return at; };
    const uint64_t tokensAt = carve(kAlign), headAt = carve(kAlign);
    std::vector<uint64_t> wAt, gAt;
    for (const auto& l : loaded) wAt.push_back(carve(l.w.size() + kTail));
    for (const auto& l : loaded) gAt.push_back(carve(l.module.dataHi - l.module.dataLo + kTail));
    n.argsAt = carve(512 * 2 * loaded.size());
    if (!n.allocateArena(used, tokensAt, kAlign, headAt, kAlign, error)) return false;

    VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    ai.buffer = engine.arenaBuffer();
    const uint64_t arena = n.address(device, &ai);
    if (!arena) { error = "the engine's arena has no device address"; return false; }

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};   // REngine's linear sampler
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = 0;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    if (createSampler(device, &si, nullptr, &linear) != VK_SUCCESS) { linear = VK_NULL_HANDLE; error = "sampler"; return false; }
    auto image = [&](int32_t (&out)[8], VkImageView view, bool storage) {
        std::memset(out, 0, sizeof(out));
        if (!view) return;
        VkDescriptorImageInfo ii{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorGetInfoEXT gi{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
        gi.type = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        if (storage) gi.data.pStorageImage = &ii; else gi.data.pSampledImage = &ii;
        getDescriptor(device, &gi, sizeof(out), out);
    };

    for (size_t i = 0; i < loaded.size(); ++i) {
        const auto& l = loaded[i];
        const std::string name = std::string("rr_") + RRNames[l.index];
        const d4r::RStageResources r = engine.stageResources(l.index);
        RRArgs a{};
        a.arena = arena; a.w = n.base + wAt[i];
        a.ex = int32_t(r.width); a.ey = int32_t(r.height);
        a.inOff = uint32_t(r.arena[0]); a.skOff = uint32_t(r.arena[1]); a.outOff = uint32_t(r.arena[2]); a.mrgOff = uint32_t(r.arena[3]);
        a.iw = 1.0f / float(r.width); a.ih = 1.0f / float(r.height);
        a.biasF = l.biasF;
        if (l.index == 0) {   // encoder 0: the end of its gather's scratch (rrvk_swin.hip's Args)
            if (!r.scratchEnd || r.scratchEnd > UINT32_MAX) { error = name + ": no gather scratch below 4 GiB"; return false; }
            a.inOff = uint32_t(r.scratchEnd);
        }
        image(a.t88, r.texture88, false); image(a.t96, r.texture96, false); image(a.t104, r.texture104, false);
        image(a.s112, r.surface112, true); image(a.s120, r.surface120, true); image(a.s128, r.surface128, true);
        VkDescriptorGetInfoEXT gi{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
        gi.type = VK_DESCRIPTOR_TYPE_SAMPLER; gi.data.pSampler = &linear;
        getDescriptor(device, &gi, sizeof(a.linear), a.linear);
        Stage st;
        st.index = l.index;
        st.pipeline = n.pipe(l.module, gAt[i], name, {32, 1, l.layout[4]}, error);
        if (!st.pipeline) return false;
        for (int s = 0; s < 2; ++s) {
            a.shift = l.index == 5 ? 0u : RRShifts[s];   // enc5 never shifts
            if (l.index == 0) a.skOff = (r.width + (a.shift & 0xffffu) + 7) / 8;   // the window grid's width
            // decoder 0's surface 144 (in s120's place) is the frame parity's: a decoder takes shift 0 on odd frames
            if (l.index == 10) image(a.s120, r.surface144[s == 0 ? 1 : 0], true);
            const uint64_t at = n.argsAt + n.args.size();
            n.args.resize(n.args.size() + 512);
            std::memcpy(n.args.data() + n.args.size() - 512, &a, sizeof(a));
            st.args[s] = n.base + at;
        }
        stages.push_back(st);
    }
    for (size_t i = 0; i < loaded.size(); ++i) n.contents.push_back({wAt[i], loaded[i].w});
    std::vector<const Module*> all;
    for (const auto& l : loaded) all.push_back(&l.module);
    n.finishContents(all, gAt);
    return true;
}

bool RRStages::records(unsigned stage) const {
    return std::any_of(stages.begin(), stages.end(), [&](const Stage& s) { return s.index == stage; });
}

bool RRStages::recordStage(VkCommandBuffer cb, unsigned stage, uint32_t groupsX, uint32_t groupsY, uint32_t shift) {
    auto& n = *impl;
    auto it = std::find_if(stages.begin(), stages.end(), [&](const Stage& s) { return s.index == stage; });
    if (it == stages.end()) return false;
    n.upload(cb);
    const uint64_t args = it->args[shift == RRShifts[1] ? 1 : 0];
    n.cmdBind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, it->pipeline);
    if (it->set) n.cmdBindSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, n.imageLayout, 0, 1, &it->set, 0, nullptr);
    n.cmdPush(cb, it->set ? n.imageLayout : n.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &args);
    if (it->groups[0]) { groupsX = it->groups[0]; groupsY = it->groups[1]; }
    n.cmdDispatch(cb, groupsX, groupsY, 1);
    return true;
}

} // namespace vkhip
