// Full-frame native K validation and GPU timing. No CUDA/NGX/PTX execution.
// With the optional trailing `hip` argument (and a model built by compile_k.py --hip) the eleven layers run on
// the native HIP kernels (hip_net.h) between the engine's Vulkan stages, whose command buffer is split around
// them through two buffers shared by file descriptor. The engine's own Vulkan network is the default.
#include "runtime.h"
#ifdef D4R_ENGINE_HIP
#include "hip_net.h"
#endif
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
void check(VkResult r, const char* operation) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(r));
}
#define CK(call) check((call), #call)
std::vector<char> read(const std::string& file) {
    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot read " + file);
    auto bytes = stream.tellg();
    if (bytes <= 0 || bytes > 1024ll * 1024 * 1024) throw std::runtime_error("invalid file size: " + file);
    std::vector<char> data(static_cast<size_t>(bytes)); stream.seekg(0);
    if (!stream.read(data.data(), bytes)) throw std::runtime_error("truncated file: " + file);
    return data;
}
template<class T> T structure(const std::string& file) {
    auto data = read(file);
    if (data.size() != sizeof(T)) throw std::runtime_error("invalid structure size: " + file);
    T result; std::memcpy(&result, data.data(), sizeof(result)); return result;
}
struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool timestamps = VK_NULL_HANDLE;
    VkPipelineCache cache = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory{};
    uint32_t timestampBits = 64;
    ~Context() {
        if (device) vkDeviceWaitIdle(device);
        if (cache) vkDestroyPipelineCache(device, cache, nullptr);
        if (timestamps) vkDestroyQueryPool(device, timestamps, nullptr);
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
    void create(uint32_t queryCount = 2, bool fp8 = false, bool externalMemory = false) {
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "d4r native inference"; application.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; info.pApplicationInfo = &application;
        CK(vkCreateInstance(&info, nullptr, &instance));
        uint32_t count = 0; CK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
        std::vector<VkPhysicalDevice> devices(count); CK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
        if (devices.empty()) throw std::runtime_error("no Vulkan GPU");
        for (auto gpu : devices) {
            VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(gpu, &p);
            if (!physical || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) physical = gpu;
        }
        vkGetPhysicalDeviceProperties(physical, &properties); vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        if (properties.apiVersion < VK_API_VERSION_1_3) throw std::runtime_error("Vulkan 1.3 required");
        uint32_t extensionCount = 0; CK(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, nullptr));
        std::vector<VkExtensionProperties> extensions(extensionCount);
        CK(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, extensions.data()));
        if (std::none_of(extensions.begin(), extensions.end(), [](const auto& e) { return !std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME); }))
            throw std::runtime_error("VK_KHR_cooperative_matrix required");
        if (fp8 && std::none_of(extensions.begin(), extensions.end(), [](const auto& e) { return !std::strcmp(e.extensionName, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME); }))
            throw std::runtime_error("VK_EXT_shader_float8 required for M");
        if (externalMemory && std::none_of(extensions.begin(), extensions.end(), [](const auto& e) { return !std::strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME); }))
            throw std::runtime_error("VK_KHR_external_memory_fd required for the native HIP network");
        VkPhysicalDeviceShaderFloat8FeaturesEXT float8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, fp8 ? &float8 : nullptr};
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &matrix};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
        vkGetPhysicalDeviceFeatures2(physical, &features);
        if (fp8 && (!float8.shaderFloat8 || !float8.shaderFloat8CooperativeMatrix))
            throw std::runtime_error("GPU lacks FP8 cooperative matrices");
        if (!matrix.cooperativeMatrix || !features.features.shaderInt16 || !features.features.shaderStorageImageExtendedFormats || !f11.storageBuffer16BitAccess || !f12.shaderFloat16 || !f13.subgroupSizeControl || !f13.computeFullSubgroups)
            throw std::runtime_error("GPU lacks f16 storage/arithmetic, cooperative matrix or subgroup control");
        auto matrixProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (!matrixProperties) throw std::runtime_error("cooperative matrix query unavailable");
        uint32_t matrixCount = 0; CK(matrixProperties(physical, &matrixCount, nullptr));
        std::vector<VkCooperativeMatrixPropertiesKHR> matrices(matrixCount, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        CK(matrixProperties(physical, &matrixCount, matrices.data()));
        if (std::none_of(matrices.begin(), matrices.end(), [](const auto& p) {
            return p.MSize == 16 && p.NSize == 16 && p.KSize == 16 && p.scope == VK_SCOPE_SUBGROUP_KHR &&
                   p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                   p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR;
        })) throw std::runtime_error("16x16x16 f16 -> f32 cooperative matrices required");
        VkPhysicalDeviceSubgroupSizeControlProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
        vkGetPhysicalDeviceProperties2(physical, &p2);
        if (subgroup.minSubgroupSize > 32 || subgroup.maxSubgroupSize < 64 || !(subgroup.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
            throw std::runtime_error("32/64-lane compute subgroups required");
        // Enable only the features this engine needs, rather than every supported feature.
        features.features = {}; features.features.shaderInt16 = VK_TRUE;
        features.features.shaderStorageImageExtendedFormats = VK_TRUE;
        f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12}; f11.storageBuffer16BitAccess = VK_TRUE;
        f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13}; f12.shaderFloat16 = VK_TRUE;
        f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &matrix}; f13.subgroupSizeControl = f13.computeFullSubgroups = VK_TRUE;
        matrix = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, fp8 ? &float8 : nullptr}; matrix.cooperativeMatrix = VK_TRUE;
        if (fp8) {
            float8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
            float8.shaderFloat8 = float8.shaderFloat8CooperativeMatrix = VK_TRUE;
        }
        uint32_t queueCount = 0; vkGetPhysicalDeviceQueueFamilyProperties(physical, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(queueCount); vkGetPhysicalDeviceQueueFamilyProperties(physical, &queueCount, families.data());
        uint32_t family = queueCount;
        for (uint32_t i = 0; i < queueCount; ++i) if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && families[i].timestampValidBits) { family = i; break; }
        if (family == queueCount) throw std::runtime_error("no timestamp-capable compute queue");
        timestampBits = families[family].timestampValidBits;
        float priority = 1;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
        const char* enabled[3] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
        uint32_t enabledCount = 1;
        if (fp8) enabled[enabledCount++] = VK_EXT_SHADER_FLOAT8_EXTENSION_NAME;
        if (externalMemory) enabled[enabledCount++] = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
        di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi; di.enabledExtensionCount = enabledCount;
        di.ppEnabledExtensionNames = enabled;
        // Diagnostic: what a host device that never asked for cooperative matrices would give the shaders.
        if (std::getenv("D4R_ENGINE_TEST_NO_COOPMAT")) { f13.pNext = nullptr; di.enabledExtensionCount = 0; }
        CK(vkCreateDevice(physical, &di, nullptr, &device)); vkGetDeviceQueue(device, family, 0, &queue);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = family; pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        CK(vkCreateCommandPool(device, &pi, nullptr, &pool));
        VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ci.commandPool = pool; ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ci.commandBufferCount = 1;
        CK(vkAllocateCommandBuffers(device, &ci, &command));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; CK(vkCreateFence(device, &fi, nullptr, &fence));
        VkQueryPoolCreateInfo ti{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; ti.queryType = VK_QUERY_TYPE_TIMESTAMP; ti.queryCount = queryCount;
        CK(vkCreateQueryPool(device, &ti, nullptr, &timestamps));
        VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO}; CK(vkCreatePipelineCache(device, &pci, nullptr, &cache));
        std::cerr << "GPU: " << properties.deviceName << '\n';
    }
    VkDeviceMemory allocate(VkMemoryRequirements requirements, VkMemoryPropertyFlags flags) {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) if ((requirements.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) {
            VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; info.allocationSize = requirements.size; info.memoryTypeIndex = i;
            VkDeviceMemory result; CK(vkAllocateMemory(device, &info, nullptr, &result)); return result;
        }
        throw std::runtime_error("no compatible memory type");
    }
    void begin() {
        CK(vkResetCommandBuffer(command, 0));
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CK(vkBeginCommandBuffer(command, &info));
    }
    void submit() {
        CK(vkEndCommandBuffer(command)); CK(vkResetFences(device, 1, &fence));
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO}; info.commandBufferCount = 1; info.pCommandBuffers = &command;
        CK(vkQueueSubmit(queue, 1, &info, fence)); CK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    }
    double elapsed(uint64_t begin, uint64_t end) const {
        const uint64_t mask = timestampBits == 64 ? UINT64_MAX : (uint64_t(1) << timestampBits) - 1;
        return ((end - begin) & mask) * double(properties.limits.timestampPeriod) / 1000.0;
    }
    double gpuMicros() {
        uint64_t times[2]; CK(vkGetQueryPoolResults(device, timestamps, 0, 2, sizeof(times), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        return elapsed(times[0], times[1]);
    }
};
struct Buffer {
    Context& context;
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* map = nullptr;
    VkDeviceSize size;
    Buffer(Context& c, VkDeviceSize bytes) : context(c), size(bytes) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; info.size = bytes; info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        CK(vkCreateBuffer(c.device, &info, nullptr, &handle));
        VkMemoryRequirements requirements; vkGetBufferMemoryRequirements(c.device, handle, &requirements);
        memory = c.allocate(requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        CK(vkBindBufferMemory(c.device, handle, memory, 0)); CK(vkMapMemory(c.device, memory, 0, VK_WHOLE_SIZE, 0, &map));
    }
    ~Buffer() {
        if (map) vkUnmapMemory(context.device, memory);
        if (handle) vkDestroyBuffer(context.device, handle, nullptr);
        if (memory) vkFreeMemory(context.device, memory, nullptr);
    }
    void load(const std::string& file) {
        auto bytes = read(file); if (bytes.size() != size) throw std::runtime_error("unexpected image bytes: " + file);
        std::memcpy(map, bytes.data(), bytes.size());
    }
    void save(const std::string& file) {
        std::ofstream stream(file, std::ios::binary); if (!stream.write(static_cast<const char*>(map), size)) throw std::runtime_error("cannot write " + file);
    }
};
#ifdef D4R_ENGINE_HIP
// A storage buffer whose memory another API (HIP) imports through a file descriptor, for the native network.
struct Shared { VkBuffer buffer{}; VkDeviceMemory memory{}; VkDeviceSize allocation = 0; int fd = -1; };
static Shared shared(const Context& c, VkDeviceSize size) {
    Shared s;
    VkExternalMemoryBufferCreateInfo ei{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ei.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ei}; bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    CK(vkCreateBuffer(c.device, &bi, nullptr, &s.buffer));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(c.device, s.buffer, &r);
    VkExportMemoryAllocateInfo xi{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    xi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &xi}; ai.allocationSize = s.allocation = r.size;
    uint32_t index = c.memory.memoryTypeCount;
    for (uint32_t i = 0; i < c.memory.memoryTypeCount; ++i)
        if ((r.memoryTypeBits & (1u << i)) && (c.memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { index = i; break; }
    if (index == c.memory.memoryTypeCount) throw std::runtime_error("no device-local memory type");
    ai.memoryTypeIndex = index;
    CK(vkAllocateMemory(c.device, &ai, nullptr, &s.memory)); CK(vkBindBufferMemory(c.device, s.buffer, s.memory, 0));
    VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    gi.memory = s.memory; gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    const auto getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(c.device, "vkGetMemoryFdKHR"));
    if (!getFd) throw std::runtime_error("vkGetMemoryFdKHR is unavailable");
    CK(getFd(c.device, &gi, &s.fd));
    return s;
}
#endif
void barrier(VkCommandBuffer cb, VkPipelineStageFlags source, VkPipelineStageFlags dest, VkAccessFlags src, VkAccessFlags dst) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask = src; b.dstAccessMask = dst;
    vkCmdPipelineBarrier(cb, source, dest, 0, 1, &b, 0, nullptr, 0, nullptr);
}
VkBufferImageCopy region(uint32_t w, uint32_t h) {
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {w, h, 1}; return r;
}
struct Image {
    Context& context;
    VkImage handle = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t w, h;
    Buffer upload;
    Image(Context& c, const std::string& file, uint32_t width, uint32_t height, int channels, int bits)
        : context(c), w(width), h(height), upload(c, VkDeviceSize(width) * height * channels * bits / 8) {
        if (!w || !h || w > c.properties.limits.maxImageDimension2D || h > c.properties.limits.maxImageDimension2D ||
            (channels != 1 && channels != 2 && channels != 4) || (bits != 16 && bits != 32)) throw std::runtime_error("unsupported frame image format/dimensions");
        VkFormat format = bits == 16 ? (channels == 1 ? VK_FORMAT_R16_SFLOAT : channels == 2 ? VK_FORMAT_R16G16_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT)
                                     : (channels == 1 ? VK_FORMAT_R32_SFLOAT : channels == 2 ? VK_FORMAT_R32G32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT);
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; info.imageType = VK_IMAGE_TYPE_2D; info.format = format;
        info.extent = {w,h,1}; info.mipLevels = info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        CK(vkCreateImage(c.device, &info, nullptr, &handle));
        VkMemoryRequirements requirements; vkGetImageMemoryRequirements(c.device, handle, &requirements);
        memory = c.allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); CK(vkBindImageMemory(c.device, handle, memory, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = handle; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; CK(vkCreateImageView(c.device, &vi, nullptr, &view));
        upload.load(file);
    }
    ~Image() {
        if (view) vkDestroyImageView(context.device, view, nullptr);
        if (handle) vkDestroyImage(context.device, handle, nullptr);
        if (memory) vkFreeMemory(context.device, memory, nullptr);
    }
    void recordUpload() {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.image = handle;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        vkCmdPipelineBarrier(context.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        auto r = region(w,h); vkCmdCopyBufferToImage(context.command, upload.handle, handle, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    }
};
struct Frame { d4r::InputParams input; d4r::OutputParams output; std::array<d4r::LayerShape,d4r::KLayers> windows; unsigned color, motion, depth; };

int run(const std::string& modelDir, const std::string& caseFile, const std::string& outputDir, int repeats, bool hip) {
#ifndef D4R_ENGINE_HIP
    if (hip) throw std::runtime_error("this build has no HIP support (rebuild with ROCm installed)");
#endif
    const bool readback = outputDir != "-";
    const bool profile = std::getenv("D4R_ENGINE_PROFILE") && std::string(std::getenv("D4R_ENGINE_PROFILE")) == "1";
    constexpr unsigned boundaries = d4r::KTimingStages + 1;
    const uint32_t queryCount = profile ? 2 + boundaries * repeats : 2;
    Context context; context.create(queryCount, false, hip);
    const std::array<const char*, d4r::KTimingStages> stageNames = {"setup", "exposure", "input",
        "enc0", "enc1", "enc2", "enc3", "enc4", "dec5", "dec4", "dec3", "dec2", "dec1", "dec0", "output", "finish"};
    struct Marks { VkQueryPool pool; uint32_t first; } marks{context.timestamps, 2};
    d4r::FrameProfiler profiler{[](VkCommandBuffer cb, uint32_t boundary, void* user) {
        auto& m = *static_cast<Marks*>(user);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m.pool, m.first + boundary);
    }, &marks};

    d4r::Shape shape{};
    std::array<std::string,3> historyFiles;
    std::vector<std::unique_ptr<Image>> images;
    std::vector<Frame> frames;
    std::ifstream manifest(caseFile);
    if (!manifest) throw std::runtime_error("cannot open case manifest");
    std::string line;
    while (std::getline(manifest,line)) {
        std::istringstream s(line); std::string kind; s >> kind;
        if (kind == "shape") {
            std::string file; s >> std::quoted(file); shape = structure<d4r::Shape>(file);
        } else if (kind == "output") {
            uint32_t w,h; s >> w >> h;
            if (!s) throw std::runtime_error("malformed output dimensions");
            shape = d4r::makeKShape(w,h);
        } else if (kind == "history") {
            for (auto& file : historyFiles) s >> std::quoted(file);
        } else if (kind == "image") {
            unsigned id, w, h; int channels, bits; std::string file;
            s >> id >> std::quoted(file) >> w >> h >> channels >> bits;
            if (!s || id != images.size()) throw std::runtime_error("malformed image declaration");
            images.push_back(std::make_unique<Image>(context,file,w,h,channels,bits));
        } else if (kind == "frame") {
            Frame f{}; std::string input,output,windows; s >> std::quoted(input) >> std::quoted(output) >> std::quoted(windows) >> f.color >> f.motion >> f.depth;
            if (!s) throw std::runtime_error("malformed frame declaration");
            f.input = structure<d4r::InputParams>(input); f.output = structure<d4r::OutputParams>(output);
            f.windows = structure<std::array<d4r::LayerShape,d4r::KLayers>>(windows); frames.push_back(f);
        } else if (kind == "frame_auto") {
            Frame f{}; d4r::FrameSettings settings{}; unsigned reset;
            s >> f.color >> f.motion >> f.depth >> settings.colorExposure >> settings.historyExposureRatio >> settings.networkExposureScale >> reset
              >> settings.jitter[0] >> settings.jitter[1] >> settings.motionScale[0] >> settings.motionScale[1]
              >> settings.motionOffset[0] >> settings.motionOffset[1] >> settings.sharpness;
            if (!s || f.color >= images.size() || reset > 1) throw std::runtime_error("malformed standard frame declaration");
            settings.renderWidth = images[f.color]->w; settings.renderHeight = images[f.color]->h; settings.reset = reset != 0;
            auto params = d4r::makeKFrameParams(shape,settings); f.input = params.input; f.output = params.output; f.windows = shape.layers;
            frames.push_back(f);
        } else if (!kind.empty()) throw std::runtime_error("unknown case declaration " + kind);
    }
    if (frames.empty() || !shape.outputWidth) throw std::runtime_error("incomplete case manifest");
    const bool seedHistory = !historyFiles[0].empty();
    if (!seedHistory && !(frames[0].input.flags & d4r::FrameFlagReset)) throw std::runtime_error("the first frame must reset or provide history");
    for (auto& f : frames) if (f.color >= images.size() || f.motion >= images.size() || f.depth >= images.size()) throw std::runtime_error("invalid frame image id");
    if (hip) {
        for (size_t i = 0; i < frames.size(); ++i) {
            const auto expected = d4r::makeKWindows(shape, uint32_t(i));
            for (unsigned layer = 0; layer < d4r::KLayers; ++layer) {
                const auto& a = frames[i].windows[layer];
                const auto& b = expected[layer];
                if (a.w != b.w || a.h != b.h || a.sx != b.sx || a.sy != b.sy)
                    throw std::runtime_error("HIP replay requires the native window cycle starting at evaluation zero; use Vulkan for custom or offset windows");
            }
        }
    }
    const auto start = std::chrono::steady_clock::now();
    d4r::Model model(context.physical,context.device,modelDir,context.cache);
    d4r::ExternalNetwork external{};
#ifdef D4R_ENGINE_HIP
    std::unique_ptr<d4r::HipNetwork> network;
    if (hip) {
        // The native K network reads the tokens and writes the head at the output-derived token grid.
        Shared tokens = shared(context, d4r::Engine::networkTokenBytes(shape.outputWidth, shape.outputHeight));
        Shared head = shared(context, d4r::Engine::networkHeadBytes(shape.outputWidth, shape.outputHeight));
        external = {tokens.buffer, head.buffer};
        network = std::make_unique<d4r::HipNetwork>(modelDir, shape.outputWidth, shape.outputHeight, tokens.fd, head.fd,
                                                    tokens.allocation, head.allocation);
    }
#endif
    d4r::Engine engine(model,shape,hip ? &external : nullptr);
    auto recordFrame = [&](uint32_t frame) {
        marks.first = 2 + frame * boundaries;
        if (profile) engine.recordFrame(context.command, profiler);
        else engine.recordFrame(context.command);
    };
    auto stageMicros = [&](uint32_t count) {
        std::vector<uint64_t> times(boundaries * count);
        CK(vkGetQueryPoolResults(context.device, context.timestamps, 2, times.size(), times.size() * sizeof(uint64_t),
            times.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        std::array<double, d4r::KTimingStages> result{};
        for (uint32_t frame = 0; frame < count; ++frame)
            for (unsigned stage = 0; stage < result.size(); ++stage)
                result[stage] += context.elapsed(times[frame * boundaries + stage], times[frame * boundaries + stage + 1]) / count;
        return result;
    };
    std::cerr << "Model/engine setup: " << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()
              << " ms; activation buffers " << engine.activationBytes() / double(1024*1024) << " MiB\n";
    uint32_t w = shape.outputWidth, h = shape.outputHeight;
    auto base = shape.layers[0];
    Buffer seedColor(context,seedHistory ? VkDeviceSize(w)*h*8 : 16), seedLuma(context,seedHistory ? VkDeviceSize(w)*h*8 : 16), seedFeature(context,seedHistory ? VkDeviceSize(base.w)*base.h*8 : 16);
    if (seedHistory) { seedColor.load(historyFiles[0]); seedLuma.load(historyFiles[1]); seedFeature.load(historyFiles[2]); }
    Buffer result(context,readback ? VkDeviceSize(w)*h*8 : 16), color(context,readback ? VkDeviceSize(w)*h*8 : 16),
           luma(context,readback ? VkDeviceSize(w)*h*8 : 16), feature(context,readback ? VkDeviceSize(base.w)*base.h*8 : 16),
           head(context,readback ? engine.headBytes() : 16);
    context.begin(); model.recordUpload(context.command); engine.recordInitialize(context.command);
    for (auto& image : images) image->recordUpload();
    // Seed copies overwrite the initialization clears, so order transfer writes explicitly.
    barrier(context.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
    auto r = region(w,h);
    if (seedHistory) {
        auto seed = engine.initialHistoryImages();
        vkCmdCopyBufferToImage(context.command,seedColor.handle,seed.color,VK_IMAGE_LAYOUT_GENERAL,1,&r);
        r = region(w*2,h*2); vkCmdCopyBufferToImage(context.command,seedLuma.handle,seed.luma,VK_IMAGE_LAYOUT_GENERAL,1,&r);
        r = region(base.w,base.h); vkCmdCopyBufferToImage(context.command,seedFeature.handle,seed.feature,VK_IMAGE_LAYOUT_GENERAL,1,&r);
    }
    barrier(context.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
    context.submit();
    std::ofstream timings, stages;
    if (profile && readback) {
        std::filesystem::create_directories(outputDir); stages.open(outputDir + "/stages.csv");
        stages << "frame"; for (auto name : stageNames) stages << "," << name << "_us"; stages << '\n';
    }
    if (readback) { std::filesystem::create_directories(outputDir); timings.open(outputDir+"/timings.csv"); timings << "frame,gpu_us\n"; }
    auto setFrame = [&](const Frame& f) { engine.setFrame({images[f.color]->view,images[f.motion]->view,images[f.depth]->view}, f.input, f.output, &f.windows); };
    // With the native network the frame is two command buffers: the front, the HIP layers on the shared buffers,
    // then the back. The tokens and head are complete before each submission returns.
    auto recordSplit = [&] {
        engine.recordFront(context.command);
        context.submit();
#ifdef D4R_ENGINE_HIP
        network->launch(); network->wait();
#else
        throw std::runtime_error("this build has no HIP support");
#endif
        context.begin();
        engine.recordBack(context.command);
    };
    for (unsigned i = 0; i < frames.size(); ++i) {
        setFrame(frames[i]); context.begin();
        vkCmdResetQueryPool(context.command,context.timestamps,0,queryCount);
        vkCmdWriteTimestamp(context.command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,context.timestamps,0);
        if (hip) recordSplit(); else recordFrame(0);
        vkCmdWriteTimestamp(context.command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,context.timestamps,1);
        if (readback) {
            auto history = engine.historyImages();
            r = region(w,h); vkCmdCopyImageToBuffer(context.command,engine.outputImage(),VK_IMAGE_LAYOUT_GENERAL,result.handle,1,&r);
            vkCmdCopyImageToBuffer(context.command,history.color,VK_IMAGE_LAYOUT_GENERAL,color.handle,1,&r);
            r = region(w*2,h*2); vkCmdCopyImageToBuffer(context.command,history.luma,VK_IMAGE_LAYOUT_GENERAL,luma.handle,1,&r);
            r = region(base.w,base.h); vkCmdCopyImageToBuffer(context.command,history.feature,VK_IMAGE_LAYOUT_GENERAL,feature.handle,1,&r);
            VkBufferCopy copy{0,0,head.size}; vkCmdCopyBuffer(context.command,engine.headBuffer(),head.handle,1,&copy);
            barrier(context.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        }
        context.submit();
        double us = context.gpuMicros(); timings << i << ',' << us << '\n';
        if (profile && !hip) {
            auto values = stageMicros(1);
            if (readback) { stages << i; for (auto value : values) stages << ',' << value; stages << '\n'; }
            if (repeats == 1) for (unsigned stage = 0; stage < values.size(); ++stage)
                std::cout << "  " << stageNames[stage] << ": " << std::fixed << std::setprecision(1) << values[stage] << " us\n";
        }
        std::cout << "frame " << i << ": " << std::fixed << std::setprecision(1) << us << " us GPU"
                  << (hip ? " (Vulkan front + HIP network + Vulkan back)" : "") << '\n';
        if (readback) {
            const std::string prefix = outputDir+"/frame-"+std::to_string(i);
            result.save(prefix+".312.bin"); color.save(prefix+".288.bin"); luma.save(prefix+".296.bin"); feature.save(prefix+".304.bin"); head.save(prefix+".head.bin");
        }
    }
    std::array<std::vector<double>, d4r::KTimingStages> stageTrials;
    if (repeats > 1) {
        // Repeat the last input as successive frames: both history sets and all data dependencies remain active.
        if (!hip) setFrame(frames.back());
        std::vector<double> gpu, wall;
        for (unsigned trial = 0; trial < 12; ++trial) {
            context.begin(); vkCmdResetQueryPool(context.command,context.timestamps,0,queryCount);
            vkCmdWriteTimestamp(context.command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,context.timestamps,0);
            for (int i = 0; i < repeats; ++i) {
                if (hip) { setFrame(frames.back()); recordSplit(); }
                else recordFrame(i);
            }
            vkCmdWriteTimestamp(context.command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,context.timestamps,1);
            auto t0 = std::chrono::steady_clock::now(); context.submit();
            double cpu = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/repeats;
            if (trial >= 3) {
                gpu.push_back(context.gpuMicros()/repeats); wall.push_back(cpu);
                if (profile && !hip) { auto values = stageMicros(repeats);
                    for (unsigned stage = 0; stage < values.size(); ++stage) stageTrials[stage].push_back(values[stage]); }
            }
        }
        std::sort(gpu.begin(),gpu.end()); std::sort(wall.begin(),wall.end());
        std::cout << "full native frame: " << gpu[gpu.size()/2] << " us GPU median (min " << gpu.front()
                  << "); " << wall[wall.size()/2] << " us amortized submit+wait; " << repeats << " frames/submit\n";
    }
    if (profile && !hip && repeats > 1) {
        std::cout << "stage GPU medians (us/frame, includes preceding dependencies; timestamp instrumentation enabled):\n";
        for (unsigned stage = 0; stage < stageTrials.size(); ++stage) {
            auto& values = stageTrials[stage]; std::sort(values.begin(), values.end());
            std::cout << "  " << stageNames[stage] << ": " << values[values.size()/2] << '\n';
        }
    }
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5 && argc != 6) { std::cerr << "usage: d4r-k MODEL_DIR CASE.txt OUTPUT_DIR|- [TIMING_REPEATS [hip]]\n"
            "The trailing 'hip' runs the eleven layers on the native HIP kernels (compile_k.py --hip) through two\n"
            "buffers shared with the engine; without it the engine's own Vulkan network is used.\n"
            "Set D4R_ENGINE_PROFILE=1 for stage timings (stages.csv with readback; Vulkan network only).\n"; return 2; }
        int repeats = argc >= 5 ? std::stoi(argv[4]) : 1;
        if (repeats < 1 || repeats > 1000) throw std::runtime_error("timing repeats must be 1..1000");
        const bool hip = argc == 6 && std::string(argv[5]) == "hip";
        if (argc == 6 && !hip) throw std::runtime_error("the optional sixth argument must be 'hip'");
        return run(argv[1],argv[2],argv[3],repeats,hip);
    } catch (const std::exception& e) { std::cerr << "d4r-k: " << e.what() << '\n'; return 1; }
}
