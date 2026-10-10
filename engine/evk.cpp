// Minimal Vulkan compute runner for the engine's kernels.
//   evk SHADER.spv GX GY REPEATS in:BINDING=FILE ... out:BINDING=FILE:BYTES[:INITFILE] ...
// Every binding is a storage buffer in VRAM. Outputs are zero-filled (or INITFILE), and written back after one dispatch.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)
static std::vector<char> slurp(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}
struct B { uint32_t binding; std::string file, out; VkDeviceSize size; VkBuffer b; void* map; };
int main(int argc, char** argv)
{
    if (argc < 6) return 1;
    const uint32_t gx = atoi(argv[2]), gy = atoi(argv[3]); const int repeats = atoi(argv[4]);
    std::vector<B> bs;
    for (int i = 5; i < argc; ++i)
    {
        std::string a = argv[i]; B b{};
        const bool out = a.rfind("out:", 0) == 0; a = a.substr(a.find(':') + 1);
        b.binding = atoi(a.substr(0, a.find('=')).c_str()); a = a.substr(a.find('=') + 1);
        if (out)
        {
            b.out = a.substr(0, a.find(':')); a = a.substr(a.find(':') + 1);
            b.size = strtoull(a.c_str(), nullptr, 10);
            if (a.find(':') != std::string::npos) b.file = a.substr(a.find(':') + 1);
        }
        else b.file = a;
        bs.push_back(b);
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t np = 8; VkPhysicalDevice pds[8]; vkEnumeratePhysicalDevices(inst, &np, pds);
    VkPhysicalDevice pd = pds[0];
    for (uint32_t i = 0; i < np; ++i) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p); if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pd = pds[i]; }
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, &f8};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cm};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t nq = 8; VkQueueFamilyProperties qf[8]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qi = 0; for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi = i; break; }
    float prio = 1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    const char* exts[] = {"VK_KHR_cooperative_matrix", "VK_EXT_shader_float8"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.enabledExtensionCount = 2; dci.ppEnabledExtensionNames = exts;
    VkDevice dev; CK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qi, 0, &q);
    std::vector<VkDescriptorSetLayoutBinding> lb;
    for (B& b : bs)
    {
        std::vector<char> data;
        if (!b.file.empty()) data = slurp(b.file);
        if (b.out.empty()) b.size = data.size();
        b.size = std::max<VkDeviceSize>((b.size + 3) & ~3ull, 16);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size = b.size; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        CK(vkCreateBuffer(dev, &bci, nullptr, &b.b));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.b, &mr);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;   // VRAM, not system memory
        uint32_t mt = ~0u;
        for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) if ((mr.memoryTypeBits >> t & 1) && (mp.memoryTypes[t].propertyFlags & want) == want) { mt = t; break; }
        if (mt == ~0u) { fprintf(stderr, "no host-visible VRAM type\n"); return 1; }
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
        VkDeviceMemory mem; CK(vkAllocateMemory(dev, &mai, nullptr, &mem)); CK(vkBindBufferMemory(dev, b.b, mem, 0));
        CK(vkMapMemory(dev, mem, 0, b.size, 0, &b.map));
        memset(b.map, 0, b.size); memcpy(b.map, data.data(), std::min<size_t>(data.size(), b.size));
        lb.push_back({b.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    }
    auto code = slurp(argv[1]);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size(); smi.pCode = (const uint32_t*)code.data();
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = lb.size(); dli.pBindings = lb.data();
    VkDescriptorSetLayout dl; CK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dl));
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dl;
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO}; rs.requiredSubgroupSize = 32;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &rs, VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
    cpi.layout = pl;
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &dps;
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dl;
    VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    std::vector<VkDescriptorBufferInfo> bi(bs.size()); std::vector<VkWriteDescriptorSet> wr;
    for (size_t i = 0; i < bs.size(); ++i)
    {
        bi[i] = {bs[i].b, 0, VK_WHOLE_SIZE};
        wr.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, bs[i].binding, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[i], nullptr});
    }
    vkUpdateDescriptorSets(dev, wr.size(), wr.data(), 0, nullptr);
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = cp; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    auto submit = [&](int n) {
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &cbi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        for (int r = 0; r < n; ++r) vkCmdDispatch(cb, gx, gy, 1);
        CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        auto t0 = std::chrono::steady_clock::now();
        CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    };
    submit(1);
    for (B& b : bs) if (!b.out.empty()) std::ofstream(b.out, std::ios::binary).write((const char*)b.map, b.size);
    if (repeats > 1)
    {
        std::vector<double> t;
        for (int it = 0; it < 12; ++it) { double us = submit(repeats) / repeats; if (it >= 3) t.push_back(us); }
        std::sort(t.begin(), t.end());
        printf("%s: %.1f us per dispatch (median, min %.1f)\n", argv[1], t[t.size() / 2], t[0]);
    }
    return 0;
}
