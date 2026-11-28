#include "fg_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>

#define VKCALL(name) PFN_vk##name name
struct FgFn {
    VKCALL(GetDeviceProcAddr); VKCALL(GetPhysicalDeviceMemoryProperties);
    VKCALL(CreateImage); VKCALL(DestroyImage); VKCALL(GetImageMemoryRequirements); VKCALL(BindImageMemory);
    VKCALL(CreateBuffer); VKCALL(DestroyBuffer); VKCALL(GetBufferMemoryRequirements); VKCALL(BindBufferMemory);
    VKCALL(AllocateMemory); VKCALL(FreeMemory); VKCALL(MapMemory); VKCALL(UnmapMemory);
    VKCALL(CreateImageView); VKCALL(DestroyImageView); VKCALL(CreateSampler); VKCALL(DestroySampler);
    VKCALL(GetBufferDeviceAddress);
    VKCALL(CreateCommandPool); VKCALL(DestroyCommandPool); VKCALL(AllocateCommandBuffers); VKCALL(BeginCommandBuffer);
    VKCALL(EndCommandBuffer); VKCALL(QueueSubmit); VKCALL(QueueWaitIdle);
    VKCALL(CmdPipelineBarrier); VKCALL(CmdFillBuffer); VKCALL(CmdClearColorImage); VKCALL(CmdCopyBuffer);
    PFN_vkCreateCuModuleNVX CreateCuModuleNVX; PFN_vkCreateCuFunctionNVX CreateCuFunctionNVX;
    PFN_vkCmdCuLaunchKernelNVX CmdCuLaunchKernelNVX; PFN_vkGetImageViewHandleNVX GetImageViewHandleNVX;
};

struct Img { VkImage img = 0; VkDeviceMemory mem = 0; VkFormat fmt; uint32_t w, h; };
struct Buf { VkBuffer buf = 0; VkDeviceMemory mem = 0; VkDeviceSize size = 0; VkDeviceAddress addr = 0; };
struct View { VkImageView view = 0; std::string image; };
struct Hdl { std::string view; int type; std::string sampler; uint64_t value = 0; };
struct Op {
    enum Kind { Launch, Fill, Clear } kind;
    // launch
    int func = 0; uint32_t grid[3], block[3], smem = 0, psize = 0; std::vector<uint8_t> params;
    struct Patch { char kind; uint32_t off; uint64_t hv; std::string buf; uint64_t boff; };
    std::vector<Patch> patches; bool matrix = false;
    // fill / clear
    std::string target; uint64_t off = 0, size = 0; uint32_t val = 0; uint32_t color[4] = {};
};
struct Variant { std::vector<Op> ops; std::map<std::string, std::string> inputs; std::vector<std::string> outputs; };

struct FgEngine {
    FgVk vk; FgFn f; VkPhysicalDeviceMemoryProperties mp;
    std::unordered_map<std::string, Img> images; std::unordered_map<std::string, Buf> buffers;
    std::unordered_map<std::string, View> views; std::unordered_map<std::string, VkSampler> samplers;
    std::map<uint64_t, Hdl> handles;           // recorded handle value -> descriptor
    std::unordered_map<uint64_t, uint64_t> hmap; // recorded value -> value in this process
    std::vector<VkCuFunctionNVX> funcs; std::vector<std::string> funcNames;
    Variant variants[3]; int nGen = 0;
};

static uint32_t memtype(FgEngine *e, uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < e->mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (e->mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return ~0u;
}

static bool readfile(const std::string &p, std::vector<uint8_t> &o)
{
    std::ifstream f(p, std::ios::binary); if (!f) return false;
    f.seekg(0, std::ios::end); size_t n = f.tellg(); f.seekg(0); o.resize(n); f.read((char *)o.data(), n); return true;
}

FgEngine *fg_create(const FgVk &vk, const std::string &dir, const std::string &cache, std::string &err)
{
    auto e = new FgEngine; e->vk = vk; FgFn &f = e->f;
#define GI(n) f.n = (PFN_vk##n)vk.gipa(vk.instance, "vk" #n)
    GI(GetDeviceProcAddr); GI(GetPhysicalDeviceMemoryProperties);
#undef GI
#define GD(n) f.n = (PFN_vk##n)f.GetDeviceProcAddr(vk.device, "vk" #n)
    GD(CreateImage); GD(DestroyImage); GD(GetImageMemoryRequirements); GD(BindImageMemory); GD(CreateBuffer); GD(DestroyBuffer);
    GD(GetBufferMemoryRequirements); GD(BindBufferMemory); GD(AllocateMemory); GD(FreeMemory); GD(MapMemory); GD(UnmapMemory);
    GD(CreateImageView); GD(DestroyImageView); GD(CreateSampler); GD(DestroySampler); GD(GetBufferDeviceAddress);
    GD(CreateCommandPool); GD(DestroyCommandPool); GD(AllocateCommandBuffers); GD(BeginCommandBuffer); GD(EndCommandBuffer);
    GD(QueueSubmit); GD(QueueWaitIdle); GD(CmdPipelineBarrier); GD(CmdFillBuffer); GD(CmdClearColorImage); GD(CmdCopyBuffer);
    GD(CreateCuModuleNVX); GD(CreateCuFunctionNVX); GD(CmdCuLaunchKernelNVX); GD(GetImageViewHandleNVX);
#undef GD
    if (!f.CreateCuModuleNVX || !f.CmdCuLaunchKernelNVX || !f.GetImageViewHandleNVX) {
        err = "nvxrt layer not active (VK_NVX_binary_import entry points missing): set VK_INSTANCE_LAYERS=VK_LAYER_NVXRT_nvx and VK_ADD_LAYER_PATH";
        delete e; return nullptr;
    }
    f.GetPhysicalDeviceMemoryProperties(vk.phys, &e->mp);
    std::ifstream sf(dir + "/script.txt");
    if (!sf) { err = "no script.txt in " + dir; delete e; return nullptr; }
    std::string line; int cur = -1; std::string initBuf, initFile; struct IC { std::string img; uint32_t c[4]; }; std::vector<IC> initClears;
    struct FuncDef { std::string hash, name; }; std::map<int, FuncDef> fdefs;
    while (std::getline(sf, line)) {
        std::istringstream is(line); std::string k; is >> k;
        if (k == "image") {
            std::string id; unsigned fmt, w, h, d, layers, mips, type; std::string us, fl; is >> id >> fmt >> w >> h >> d >> layers >> mips >> us >> fl >> type;
            VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ci.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; ci.imageType = VK_IMAGE_TYPE_2D; ci.format = (VkFormat)fmt;
            ci.extent = {w, h, d}; ci.mipLevels = mips; ci.arrayLayers = layers; ci.samples = VK_SAMPLE_COUNT_1_BIT;
            ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = (VkImageUsageFlags)strtoul(us.c_str(), 0, 16) | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
            ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            uint32_t fams[2] = {vk.queueFamily, vk.computeFamily};
            if (vk.computeFamily != ~0u && vk.computeFamily != vk.queueFamily) { ci.sharingMode = VK_SHARING_MODE_CONCURRENT; ci.queueFamilyIndexCount = 2; ci.pQueueFamilyIndices = fams; }
            Img im; im.fmt = ci.format; im.w = w; im.h = h;
            if (f.CreateImage(vk.device, &ci, nullptr, &im.img) != VK_SUCCESS) { err = "CreateImage " + id; return nullptr; }
            VkMemoryRequirements mr; f.GetImageMemoryRequirements(vk.device, im.img, &mr);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = memtype(e, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            f.AllocateMemory(vk.device, &ai, nullptr, &im.mem); f.BindImageMemory(vk.device, im.img, im.mem, 0);
            e->images[id] = im;
        } else if (k == "view") {
            std::string id, img; unsigned fmt, type, mip, mips, layer, layers; is >> id >> img >> fmt >> type >> mip >> mips >> layer >> layers;
            VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            ci.image = e->images[img].img; ci.viewType = (VkImageViewType)type; ci.format = (VkFormat)fmt;
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, mips, layer, layers};
            View v; v.image = img;
            if (f.CreateImageView(vk.device, &ci, nullptr, &v.view) != VK_SUCCESS) { err = "CreateImageView " + id; return nullptr; }
            e->views[id] = v;
        } else if (k == "sampler") {
            std::string id; int mag, mn, mip, u, v, w, border, unnorm; is >> id >> mag >> mn >> mip >> u >> v >> w >> border >> unnorm;
            VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            ci.magFilter = (VkFilter)mag; ci.minFilter = (VkFilter)mn; ci.mipmapMode = (VkSamplerMipmapMode)mip;
            ci.addressModeU = (VkSamplerAddressMode)u; ci.addressModeV = (VkSamplerAddressMode)v; ci.addressModeW = (VkSamplerAddressMode)w;
            ci.borderColor = (VkBorderColor)border; ci.unnormalizedCoordinates = unnorm; ci.maxLod = unnorm ? 0.f : VK_LOD_CLAMP_NONE;
            VkSampler s; f.CreateSampler(vk.device, &ci, nullptr, &s); e->samplers[id] = s;
        } else if (k == "buffer") {
            std::string id; uint64_t size; is >> id >> size;
            Buf b; b.size = size;
            VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = size;
            ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            uint32_t bfams[2] = {vk.queueFamily, vk.computeFamily};
            if (vk.computeFamily != ~0u && vk.computeFamily != vk.queueFamily) { ci.sharingMode = VK_SHARING_MODE_CONCURRENT; ci.queueFamilyIndexCount = 2; ci.pQueueFamilyIndices = bfams; }
            f.CreateBuffer(vk.device, &ci, nullptr, &b.buf);
            VkMemoryRequirements mr; f.GetBufferMemoryRequirements(vk.device, b.buf, &mr);
            VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi}; ai.allocationSize = mr.size; ai.memoryTypeIndex = memtype(e, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            f.AllocateMemory(vk.device, &ai, nullptr, &b.mem); f.BindBufferMemory(vk.device, b.buf, b.mem, 0);
            VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bi.buffer = b.buf;
            b.addr = f.GetBufferDeviceAddress(vk.device, &bi);
            e->buffers[id] = b;
        } else if (k == "handle") {
            std::string v, s; std::string hv; int type; is >> hv >> v >> type >> s;
            Hdl h; h.view = v; h.type = type; h.sampler = s == "-" ? "" : s; e->handles[strtoull(hv.c_str(), 0, 16)] = h;
        } else if (k == "func") {
            int i; FuncDef d; is >> i >> d.hash >> d.name; fdefs[i] = d;
        } else if (k == "initclear") { IC c; is >> c.img >> c.c[0] >> c.c[1] >> c.c[2] >> c.c[3]; initClears.push_back(c);
        } else if (k == "initdata") { is >> initBuf >> initFile;
        } else if (k == "frame") { is >> cur;
        } else if (k == "endframe") { cur = -1;
        } else if (k == "input" && cur >= 0) {
            std::string lab, img; unsigned w, h; is >> lab >> img >> w >> h; e->variants[cur].inputs[lab] = img;
        } else if (k == "output" && cur >= 0) {
            int g; std::string img; is >> g >> img; e->variants[cur].outputs.push_back(img); if (g + 1 > e->nGen) e->nGen = g + 1;
        } else if (k == "fill" && cur >= 0) {
            Op o; o.kind = Op::Fill; is >> o.target >> o.off >> o.size >> o.val; e->variants[cur].ops.push_back(o);
        } else if (k == "clear" && cur >= 0) {
            Op o; o.kind = Op::Clear; is >> o.target >> o.color[0] >> o.color[1] >> o.color[2] >> o.color[3]; e->variants[cur].ops.push_back(o);
        } else if (k == "launch" && cur >= 0) {
            Op o; o.kind = Op::Launch; std::string hex, ps;
            is >> o.func >> o.grid[0] >> o.grid[1] >> o.grid[2] >> o.block[0] >> o.block[1] >> o.block[2] >> o.smem >> o.psize >> hex >> ps;
            o.params.resize(hex.size() / 2);
            for (size_t i = 0; i < o.params.size(); i++) o.params[i] = (uint8_t)strtoul(hex.substr(2 * i, 2).c_str(), 0, 16);
            if (ps != "-") {
                std::istringstream pss(ps); std::string tok;
                while (std::getline(pss, tok, ',')) {
                    Op::Patch p{}; std::vector<std::string> t; std::stringstream ts(tok); std::string x;
                    while (std::getline(ts, x, ':')) t.push_back(x);
                    p.kind = t[0][0]; p.off = atoi(t[1].c_str());
                    if (p.kind == 'H') p.hv = strtoull(t[2].c_str(), 0, 16); else { p.buf = t[2]; p.boff = strtoull(t[3].c_str(), 0, 10); }
                    o.patches.push_back(p);
                }
            }
            e->variants[cur].ops.push_back(o);
        }
    }
    // kernels
    e->funcs.resize(fdefs.size()); e->funcNames.resize(fdefs.size());
    std::map<std::string, VkCuModuleNVX> mods;
    for (auto &kv : fdefs) {
        auto &d = kv.second;
        if (!mods.count(d.hash)) {
            std::vector<uint8_t> ptx;
            if (!readfile(cache + "/kernels/" + d.hash + "/module.ptx", ptx)) { err = "missing module.ptx for " + d.name + " in " + cache; return nullptr; }
            VkCuModuleCreateInfoNVX mi{VK_STRUCTURE_TYPE_CU_MODULE_CREATE_INFO_NVX}; mi.dataSize = ptx.size(); mi.pData = ptx.data();
            VkCuModuleTexturingModeCreateInfoNVX tm{VK_STRUCTURE_TYPE_CU_MODULE_TEXTURING_MODE_CREATE_INFO_NVX}; tm.use64bitTexturing = VK_TRUE; mi.pNext = &tm;
            VkCuModuleNVX m; if (f.CreateCuModuleNVX(vk.device, &mi, nullptr, &m) != VK_SUCCESS) { err = "CreateCuModule " + d.name; return nullptr; }
            mods[d.hash] = m;
        }
        VkCuFunctionCreateInfoNVX fi{VK_STRUCTURE_TYPE_CU_FUNCTION_CREATE_INFO_NVX}; fi.module = mods[d.hash]; fi.pName = d.name.c_str();
        if (f.CreateCuFunctionNVX(vk.device, &fi, nullptr, &e->funcs[kv.first]) != VK_SUCCESS) { err = "CreateCuFunction " + d.name; return nullptr; }
        e->funcNames[kv.first] = d.name;
    }
    // recorded handle values -> handles here
    for (auto &kv : e->handles) {
        auto &h = kv.second;
        VkImageViewHandleInfoNVX hi{VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX};
        hi.imageView = e->views[h.view].view; hi.descriptorType = (VkDescriptorType)h.type; hi.sampler = h.sampler.empty() ? VK_NULL_HANDLE : e->samplers[h.sampler];
        e->hmap[kv.first] = f.GetImageViewHandleNVX(vk.device, &hi);
    }
    // init: layout GENERAL + zero all images, weights
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = vk.queueFamily; pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool pool; f.CreateCommandPool(vk.device, &pi, nullptr, &pool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb; f.AllocateCommandBuffers(vk.device, &cai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    f.BeginCommandBuffer(cb, &bi);
    std::vector<VkImageMemoryBarrier> bars;
    for (auto &kv : e->images) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.newLayout = VK_IMAGE_LAYOUT_GENERAL; b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = kv.second.img; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}; bars.push_back(b);
    }
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 0, 0, (uint32_t)bars.size(), bars.data());
    VkClearColorValue zero{}; VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    for (auto &kv : e->images) f.CmdClearColorImage(cb, kv.second.img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rng);
    for (auto &kv : e->buffers) f.CmdFillBuffer(cb, kv.second.buf, 0, VK_WHOLE_SIZE, 0);
    if (!initClears.empty()) {
        VkMemoryBarrier cmb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; cmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; cmb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &cmb, 0, 0, 0, 0);
        for (auto &c : initClears) { VkClearColorValue cv; memcpy(cv.uint32, c.c, 16); f.CmdClearColorImage(cb, e->images[c.img].img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rng); }
    }
    VkBuffer stg = 0; VkDeviceMemory stgm = 0;
    if (!initBuf.empty()) {
        std::vector<uint8_t> w;
        if (!readfile(dir + "/" + initFile, w)) { err = "missing " + initFile; return nullptr; }
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = w.size(); ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        f.CreateBuffer(vk.device, &ci, nullptr, &stg); VkMemoryRequirements mr; f.GetBufferMemoryRequirements(vk.device, stg, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memtype(e, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        f.AllocateMemory(vk.device, &ai, nullptr, &stgm); f.BindBufferMemory(vk.device, stg, stgm, 0);
        void *p; f.MapMemory(vk.device, stgm, 0, w.size(), 0, &p); memcpy(p, w.data(), w.size()); f.UnmapMemory(vk.device, stgm);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, 0, 0, 0);
        VkBufferCopy bc{0, 0, w.size()}; f.CmdCopyBuffer(cb, stg, e->buffers[initBuf].buf, 1, &bc);
    }
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, 0, 0, 0);
    f.EndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    f.QueueSubmit(vk.queue, 1, &si, VK_NULL_HANDLE); f.QueueWaitIdle(vk.queue);
    f.DestroyCommandPool(vk.device, pool, nullptr);
    if (stg) { f.DestroyBuffer(vk.device, stg, nullptr); f.FreeMemory(vk.device, stgm, nullptr); }
    return e;
}

// The device must be idle with respect to everything the engine recorded.
void fg_destroy(FgEngine *e)
{
    FgFn &f = e->f; VkDevice d = e->vk.device;
    for (auto &kv : e->views) f.DestroyImageView(d, kv.second.view, nullptr);
    for (auto &kv : e->samplers) f.DestroySampler(d, kv.second, nullptr);
    for (auto &kv : e->images) { f.DestroyImage(d, kv.second.img, nullptr); f.FreeMemory(d, kv.second.mem, nullptr); }
    for (auto &kv : e->buffers) { f.DestroyBuffer(d, kv.second.buf, nullptr); f.FreeMemory(d, kv.second.mem, nullptr); }
    delete e;
}
int fg_generated_count(const FgEngine *e) { return e->nGen; }
VkImage fg_input_image(const FgEngine *e, const char *label)
{
    auto it = e->variants[0].inputs.find(label); return it == e->variants[0].inputs.end() ? VK_NULL_HANDLE : e->images.at(it->second).img;
}
void fg_input_size(const FgEngine *e, const char *label, uint32_t &w, uint32_t &h, VkFormat &fmt)
{
    auto it = e->variants[0].inputs.find(label); if (it == e->variants[0].inputs.end()) { w = h = 0; return; }
    auto &im = e->images.at(it->second); w = im.w; h = im.h; fmt = im.fmt;
}
VkImage fg_output_image(const FgEngine *e, int variant, int k) { return e->images.at(e->variants[variant].outputs.at(k)).img; }

void fg_record(FgEngine *e, VkCommandBuffer cmd, int variant, const float *m)
{
    FgFn &f = e->f; Variant &v = e->variants[variant];
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT; mb.dstAccessMask = mb.srcAccessMask;
    auto barrier = [&] { f.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, 0, 0, 0); };
    barrier();
    uint8_t pb[1024];
    for (auto &o : v.ops) {
        if (o.kind == Op::Fill) {
            f.CmdFillBuffer(cmd, e->buffers[o.target].buf, o.off, o.size, o.val); barrier();
        } else if (o.kind == Op::Clear) {
            VkClearColorValue c; memcpy(c.uint32, o.color, 16); VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
            f.CmdClearColorImage(cmd, e->images[o.target].img, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r); barrier();
        } else {
            memcpy(pb, o.params.data(), o.params.size());
            for (auto &p : o.patches) {
                uint64_t val;
                if (p.kind == 'H') val = e->hmap[p.hv]; else val = e->buffers[p.buf].addr + p.boff;
                memcpy(pb + p.off, &val, 8);
            }
            if (m && e->funcNames[o.func] == "Kernel_EstimatePrev2CurrScatter") memcpy(pb + 80, m, 64);
            uint32_t psz = o.psize;
            const void *extras[5] = {(const void *)1, pb, (const void *)2, &psz, (const void *)0};
            VkCuLaunchInfoNVX li{VK_STRUCTURE_TYPE_CU_LAUNCH_INFO_NVX};
            li.function = e->funcs[o.func]; li.gridDimX = o.grid[0]; li.gridDimY = o.grid[1]; li.gridDimZ = o.grid[2];
            li.blockDimX = o.block[0]; li.blockDimY = o.block[1]; li.blockDimZ = o.block[2]; li.sharedMemBytes = o.smem;
            li.extraCount = 5; li.pExtras = extras;
            f.CmdCuLaunchKernelNVX(cmd, &li);
        }
    }
    barrier();
}
