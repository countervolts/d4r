// Runs one translated DLSS texture kernel as a Vulkan compute shader on captured inputs.
//   kvk SHADER.spv MANIFEST [repeats] [subgroup]
// MANIFEST lines (bindings are assigned in the order ptx2glsl.py uses: parameter block, textures by
// offset, surfaces by offset):
//   param FILE
//   grid X Y
//   tex OFF FILE W H CHANNELS BITS linear|nearest
//   surf OFF W H CHANNELS BITS INITFILE|- OUTFILE
//   buf OFF FILE DELTA          (8-byte pointer at OFF = buffer address + DELTA)
//   lut FILE                    (storage buffer bound after the surfaces)
//   bufout OFF FILE             (after the dispatch, write the pointer buffer of parameter offset OFF to FILE)
//   outbuf FILE BYTES           (storage buffer bound after those, written to FILE after the dispatch)
//   bindings N0 N1 ...          (optional: the binding number of each of the above, in that order, for shaders
//                                whose bindings are not ptx2glsl.py's; e.g. engine/layer_m.comp's L stages)
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)
static std::vector<char> slurp(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}
static VkDevice dev; static VkPhysicalDeviceMemoryProperties mp;
static VkDeviceMemory alloc(VkMemoryRequirements mr, VkMemoryPropertyFlags want, bool bda)
{
    uint32_t mt = ~0u;
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) if ((mr.memoryTypeBits >> t & 1) && (mp.memoryTypes[t].propertyFlags & want) == want) { mt = t; break; }
    VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, bda ? &fi : nullptr}; ai.allocationSize = mr.size; ai.memoryTypeIndex = mt;
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, nullptr, &m)); return m;
}
struct Buf { VkBuffer b; VkDeviceMemory m; void* map; VkDeviceSize size; uint64_t addr; };
static Buf make_buf(VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buf r{}; r.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = size; ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    CK(vkCreateBuffer(dev, &ci, nullptr, &r.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, r.b, &mr);
    // shader-visible data must live in VRAM: the first host-visible type on RADV is system memory, which the GPU reads over PCIe
    const VkMemoryPropertyFlags vram = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    bool has_vram = false;
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) has_vram |= (mr.memoryTypeBits >> t & 1) && (mp.memoryTypes[t].propertyFlags & vram) == vram;
    r.m = alloc(mr, has_vram ? vram : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true);
    CK(vkBindBufferMemory(dev, r.b, r.m, 0)); CK(vkMapMemory(dev, r.m, 0, size, 0, &r.map));
    VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; ai.buffer = r.b; r.addr = vkGetBufferDeviceAddress(dev, &ai);
    return r;
}
struct Img { VkImage i; VkImageView v; VkFormat f; uint32_t w, h, bpp; std::string init, out; bool surf, linear; int off; };
static VkFormat fmt_of(int ch, int bits)
{
    // 8: UNORM, 9: SNORM, one byte per channel (CUDA's UNORM_INT8Xn / SNORM_INT8Xn arrays)
    if (bits == 8) return ch == 1 ? VK_FORMAT_R8_UNORM : ch == 2 ? VK_FORMAT_R8G8_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
    if (bits == 9) return ch == 1 ? VK_FORMAT_R8_SNORM : ch == 2 ? VK_FORMAT_R8G8_SNORM : VK_FORMAT_R8G8B8A8_SNORM;
    if (bits == 10 && ch == 4) return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    if (bits == 16) return ch == 1 ? VK_FORMAT_R16_SFLOAT : ch == 2 ? VK_FORMAT_R16G16_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
    return ch == 1 ? VK_FORMAT_R32_SFLOAT : ch == 2 ? VK_FORMAT_R32G32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
}
int main(int argc, char** argv)
{
    if (argc < 3) return 1;
    const int repeats = argc > 3 ? atoi(argv[3]) : 1; const uint32_t subgroup = argc > 4 ? atoi(argv[4]) : 32;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t np = 8; VkPhysicalDevice pds[8]; vkEnumeratePhysicalDevices(inst, &np, pds);
    VkPhysicalDevice pd = pds[0];
    for (uint32_t i = 0; i < np; ++i) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p); if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pd = pds[i]; }
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t nq = 8; VkQueueFamilyProperties qf[8]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qi = 0; for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi = i; break; }
    float prio = 1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    CK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = cp; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    auto run = [&](auto&& record) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi)); record(); CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));
    };
    // ---------------------------------------------------------------- manifest
    std::vector<char> param; uint32_t gx = 1, gy = 1;
    std::vector<Img> imgs; std::map<int, std::pair<Buf, int64_t>> bufs; std::vector<Buf> luts; std::vector<std::string> lutOut; std::map<int, std::string> bufOut;
    std::vector<uint32_t> remap;
    std::ifstream mf(argv[2]); std::string line;
    while (std::getline(mf, line))
    {
        std::istringstream ls(line); std::string k; ls >> k;
        if (k == "param") { std::string f; ls >> f; param = slurp(f); }
        else if (k == "grid") ls >> gx >> gy;
        else if (k == "bindings") { uint32_t b; while (ls >> b) remap.push_back(b); }
        else if (k == "tex" || k == "surf")
        {
            Img im{}; int ch, bits; std::string mode; im.surf = k == "surf";
            if (im.surf) ls >> im.off >> im.w >> im.h >> ch >> bits >> im.init >> im.out;
            else { ls >> im.off >> im.init >> im.w >> im.h >> ch >> bits >> mode; im.linear = mode == "linear"; }
            im.f = fmt_of(ch, bits); im.bpp = bits < 16 ? ch : ch * bits / 8;
            imgs.push_back(im);
        }
        else if (k == "lut")
        {
            std::string f; ls >> f; auto data = slurp(f);
            Buf b = make_buf(data.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); memcpy(b.map, data.data(), data.size()); luts.push_back(b); lutOut.push_back("");
        }
        else if (k == "bufout") { int off; std::string f; ls >> off >> f; bufOut[off] = f; }
        else if (k == "outbuf")
        {
            std::string f; VkDeviceSize n; ls >> f >> n;
            Buf b = make_buf(n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); memset(b.map, 0, n); luts.push_back(b); lutOut.push_back(f);
        }
        else if (k == "buf")
        {
            int off; std::string f; int64_t delta; ls >> off >> f >> delta;
            auto data = slurp(f); Buf b = make_buf(data.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); memcpy(b.map, data.data(), data.size());
            bufs[off] = {b, delta};
        }
    }
    for (auto& [off, b] : bufs) { uint64_t a = b.first.addr + b.second; memcpy(param.data() + off, &a, 8); }
    std::sort(imgs.begin(), imgs.end(), [](const Img& a, const Img& b) { return a.surf != b.surf ? !a.surf : a.off < b.off; });
    Buf pbuf = make_buf((param.size() + 3) & ~3ull, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); memcpy(pbuf.map, param.data(), param.size());
    // ---------------------------------------------------------------- images
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; sci.maxLod = 0;
    VkSampler nearest, linear; CK(vkCreateSampler(dev, &sci, nullptr, &nearest));
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR; CK(vkCreateSampler(dev, &sci, nullptr, &linear));
    std::vector<Buf> staging;
    for (Img& im : imgs)
    {
        const bool r16u = im.surf && im.f == VK_FORMAT_R16_SFLOAT;     // byte-addressed surface: bound as R16_UINT
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ci.imageType = VK_IMAGE_TYPE_2D; ci.format = im.f; ci.extent = {im.w, im.h, 1};
        ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (r16u) ci.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        CK(vkCreateImage(dev, &ci, nullptr, &im.i));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, im.i, &mr);
        CK(vkBindImageMemory(dev, im.i, alloc(mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false), 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = im.i; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = r16u ? VK_FORMAT_R16_UINT : im.f; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        CK(vkCreateImageView(dev, &vi, nullptr, &im.v));
        Buf st = make_buf((VkDeviceSize)im.w * im.h * im.bpp, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        memset(st.map, 0, st.size);
        if (im.init != "-") { auto d = slurp(im.init); memcpy(st.map, d.data(), std::min<size_t>(d.size(), st.size)); }
        staging.push_back(st);
    }
    auto barrier = [&](VkImage i, VkImageLayout a, VkImageLayout b) {
        VkImageMemoryBarrier mb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; mb.oldLayout = a; mb.newLayout = b; mb.image = i;
        mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        mb.srcQueueFamilyIndex = mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; mb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &mb);
    };
    auto upload = [&]() {
        for (size_t i = 0; i < imgs.size(); ++i)
        {
            barrier(imgs[i].i, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {imgs[i].w, imgs[i].h, 1};
            vkCmdCopyBufferToImage(cb, staging[i].b, imgs[i].i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            barrier(imgs[i].i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        }
    };
    run(upload);
    // ---------------------------------------------------------------- pipeline
    auto code = slurp(argv[1]);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size(); smi.pCode = (const uint32_t*)code.data();
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    auto B = [&](size_t positional) {
        if (remap.empty()) return (uint32_t)positional;
        if (positional >= remap.size()) { fprintf(stderr, "bindings: no number for resource %zu\n", positional); exit(1); }
        return remap[positional];
    };
    std::vector<VkDescriptorSetLayoutBinding> lb;
    lb.push_back({B(0), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    for (size_t i = 0; i < imgs.size(); ++i)
        lb.push_back({B(i + 1), imgs[i].surf ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    for (size_t i = 0; i < luts.size(); ++i)
        lb.push_back({B(imgs.size() + 1 + i), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = lb.size(); dli.pBindings = lb.data();
    VkDescriptorSetLayout dl; CK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dl));
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dl;
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO}; rs.requiredSubgroupSize = subgroup;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &rs, VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
    cpi.layout = pl;
    VkPipeline pipe;
    auto c0 = std::chrono::steady_clock::now();
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));
    fprintf(stderr, "pipeline compile: %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count());
    VkDescriptorPoolSize dps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 3; dpi.pPoolSizes = dps;
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dl;
    VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo pbi{pbuf.b, 0, VK_WHOLE_SIZE};
    std::vector<VkDescriptorImageInfo> ii(imgs.size()); std::vector<VkWriteDescriptorSet> wr;
    wr.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, B(0), 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pbi, nullptr});
    for (size_t i = 0; i < imgs.size(); ++i)
    {
        ii[i] = {imgs[i].linear ? linear : nearest, imgs[i].v, VK_IMAGE_LAYOUT_GENERAL};
        wr.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, B(i + 1), 0, 1, imgs[i].surf ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii[i], nullptr, nullptr});
    }
    std::vector<VkDescriptorBufferInfo> li(luts.size());
    for (size_t i = 0; i < luts.size(); ++i)
    {
        li[i] = {luts[i].b, 0, VK_WHOLE_SIZE};
        wr.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, B(imgs.size() + 1 + i), 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &li[i], nullptr});
    }
    vkUpdateDescriptorSets(dev, wr.size(), wr.data(), 0, nullptr);
    auto dispatch = [&](int n) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        for (int r = 0; r < n; ++r) vkCmdDispatch(cb, gx, gy, 1);
    };
    run([&] { dispatch(1); });
    // ---------------------------------------------------------------- read back
    run([&] {
        for (size_t i = 0; i < imgs.size(); ++i) if (imgs[i].surf)
        {
            VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {imgs[i].w, imgs[i].h, 1};
            vkCmdCopyImageToBuffer(cb, imgs[i].i, VK_IMAGE_LAYOUT_GENERAL, staging[i].b, 1, &r);
        }
    });
    for (size_t i = 0; i < imgs.size(); ++i) if (imgs[i].surf)
        std::ofstream(imgs[i].out, std::ios::binary).write((const char*)staging[i].map, staging[i].size);
    for (auto& [off, f] : bufOut)
        std::ofstream(f, std::ios::binary).write((const char*)bufs[off].first.map, bufs[off].first.size);
    for (size_t i = 0; i < luts.size(); ++i) if (!lutOut[i].empty())
        std::ofstream(lutOut[i], std::ios::binary).write((const char*)luts[i].map, luts[i].size);
    if (const char* gap = getenv("KVK_GAP_US"))
    {
        // one dispatch per submit with an idle gap, like one kernel inside a frame
        std::vector<double> t;
        for (int it = 0; it < 600; ++it)
        {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi)); dispatch(1); CK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            auto t0 = std::chrono::steady_clock::now();
            CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));
            if (it >= 100) t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(atoi(gap));
            while (std::chrono::steady_clock::now() < until) {}
        }
        std::sort(t.begin(), t.end());
        printf("%s: %.1f us per submitted dispatch with %s us gaps (median of %zu, min %.1f)\n", argv[1], t[t.size() / 2], gap, t.size(), t[0]);
    }
    else if (repeats > 1)
    {
        std::vector<double> t;
        for (int it = 0; it < 12; ++it)
        {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi)); dispatch(repeats); CK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            auto t0 = std::chrono::steady_clock::now();
            CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));
            if (it >= 3) t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / repeats);
        }
        std::sort(t.begin(), t.end());
        printf("%s: %.1f us per dispatch (median of %zu submits of %d, min %.1f)\n", argv[1], t[t.size() / 2], t.size(), repeats, t[0]);
    }
    return 0;
}
