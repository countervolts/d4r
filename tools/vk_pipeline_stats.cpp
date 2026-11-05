// vk_pipeline_stats SHADER.spv [-ir] [push=N] [bN=ssbo|sampler|image|ubo|texel|stexel ...]
// Build (Git Bash, MSYS2 MinGW + Vulkan SDK): x86_64-w64-mingw32-g++ -O1 -std=c++17 -I$VULKAN_SDK/Include tools/vk_pipeline_stats.cpp -o vk_pipeline_stats.exe -static
// VKSTATS_WAVE=32|64 requests a subgroup size; VKSTATS_NO_FP8=1 skips VK_EXT_shader_float8.
// Builds the compute pipeline on the first GPU with VK_KHR_pipeline_executable_properties and prints the driver's
// statistics (and, with -ir, its internal representations). Bindings are all in set 0.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)
#define FN(name) auto name = (PFN_##name)gipa(inst, #name)
#define DFN(name) auto name = (PFN_##name)gdpa(dev, #name)
int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: vk_pipeline_stats SHADER.spv [-ir] [push=N] [bN=ssbo|sampler|image|ubo|texel|stexel ...]\n"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    if (code.empty()) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    bool ir = false; uint32_t push = 0;
    std::vector<VkDescriptorSetLayoutBinding> binds;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-ir") { ir = true; continue; }
        if (a.rfind("push=", 0) == 0) { push = atoi(a.c_str() + 5); continue; }
        auto eq = a.find('=');
        VkDescriptorSetLayoutBinding b{};
        b.binding = atoi(a.c_str() + 1); b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        std::string t = a.substr(eq + 1);
        b.descriptorType = t == "ssbo" ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : t == "sampler" ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
                           t == "image" ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : t == "ubo" ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER :
                           t == "texel" ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        binds.push_back(b);
    }
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    auto gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(lib, "vkGetInstanceProcAddr");
    VkInstance inst = VK_NULL_HANDLE;
    FN(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    CK(vkCreateInstance(&ici, nullptr, &inst));
    FN(vkEnumeratePhysicalDevices); FN(vkGetDeviceProcAddr); FN(vkCreateDevice); FN(vkGetPhysicalDeviceProperties);
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = pds[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    fprintf(stderr, "GPU: %s\n", props.deviceName);
    float prio = 1;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; q.queueCount = 1; q.pQueuePriorities = &prio;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pe{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    pe.pipelineExecutableInfo = VK_TRUE;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, &pe};
    cm.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, &cm};
    f8.shaderFloat8 = VK_TRUE; f8.shaderFloat8CooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, getenv("VKSTATS_NO_FP8") ? (void*)&cm : (void*)&f8};
    v13.subgroupSizeControl = v13.computeFullSubgroups = v13.maintenance4 = v13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    v12.shaderFloat16 = v12.shaderInt8 = v12.storageBuffer8BitAccess = v12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    v12.bufferDeviceAddress = v12.vulkanMemoryModel = v12.vulkanMemoryModelDeviceScope = v12.scalarBlockLayout = VK_TRUE;
    v12.shaderSubgroupExtendedTypes = VK_TRUE;
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    v11.storageBuffer16BitAccess = v11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    feats.features.shaderInt16 = feats.features.shaderInt64 = feats.features.shaderFloat64 = VK_TRUE;
    std::vector<const char*> ext{"VK_KHR_pipeline_executable_properties", "VK_KHR_cooperative_matrix"};
    if (!getenv("VKSTATS_NO_FP8")) ext.push_back("VK_EXT_shader_float8");
    if (getenv("VKSTATS_EXT")) ext.push_back(getenv("VKSTATS_EXT"));
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &feats}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &q;
    dci.enabledExtensionCount = uint32_t(ext.size()); dci.ppEnabledExtensionNames = ext.data();
    VkDevice dev; CK(vkCreateDevice(pd, &dci, nullptr, &dev));
    auto gdpa = vkGetDeviceProcAddr;
    DFN(vkCreateShaderModule); DFN(vkCreateDescriptorSetLayout); DFN(vkCreatePipelineLayout); DFN(vkCreateComputePipelines);
    DFN(vkGetPipelineExecutablePropertiesKHR); DFN(vkGetPipelineExecutableStatisticsKHR); DFN(vkGetPipelineExecutableInternalRepresentationsKHR);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = code.size(); smi.pCode = (const uint32_t*)code.data();
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount = uint32_t(binds.size()); dl.pBindings = binds.data();
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &dsl));
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount = 1; pl.pSetLayouts = &dsl;
    pl.pushConstantRangeCount = push ? 1 : 0; pl.pPushConstantRanges = &pr;
    VkPipelineLayout lay; CK(vkCreatePipelineLayout(dev, &pl, nullptr, &lay));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | (ir ? VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR : 0);
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm; cpi.stage.pName = "main"; cpi.layout = lay;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if (const char* w = getenv("VKSTATS_WAVE")) { rs.requiredSubgroupSize = atoi(w); cpi.stage.pNext = &rs; }
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));
    VkPipelineInfoKHR pinfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR}; pinfo.pipeline = pipe;
    uint32_t ne = 0; vkGetPipelineExecutablePropertiesKHR(dev, &pinfo, &ne, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> ep(ne, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    vkGetPipelineExecutablePropertiesKHR(dev, &pinfo, &ne, ep.data());
    for (uint32_t e = 0; e < ne; ++e) {
        printf("executable %u: %s (subgroup %u)\n", e, ep[e].name, ep[e].subgroupSize);
        VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR}; ei.pipeline = pipe; ei.executableIndex = e;
        uint32_t ns = 0; vkGetPipelineExecutableStatisticsKHR(dev, &ei, &ns, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        vkGetPipelineExecutableStatisticsKHR(dev, &ei, &ns, st.data());
        for (auto& s : st) {
            printf("  %-40s ", s.name);
            switch (s.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: printf("%u\n", s.value.b32); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: printf("%lld\n", (long long)s.value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: printf("%llu\n", (unsigned long long)s.value.u64); break;
            default: printf("%g\n", s.value.f64);
            }
        }
        if (ir) {
            uint32_t nr = 0; vkGetPipelineExecutableInternalRepresentationsKHR(dev, &ei, &nr, nullptr);
            std::vector<VkPipelineExecutableInternalRepresentationKHR> r(nr, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
            vkGetPipelineExecutableInternalRepresentationsKHR(dev, &ei, &nr, r.data());
            std::vector<std::vector<char>> data(nr);
            for (uint32_t k = 0; k < nr; ++k) { data[k].resize(r[k].dataSize); r[k].pData = data[k].data(); }
            vkGetPipelineExecutableInternalRepresentationsKHR(dev, &ei, &nr, r.data());
            for (uint32_t k = 0; k < nr; ++k) printf("---- %s (%s)\n%.*s\n", r[k].name, r[k].description, int(r[k].dataSize), (const char*)r[k].pData);
        }
    }
    return 0;
}
