// Native Ray Reconstruction replay: runs REngine (engine/runtime_rr.inc) on a capture's game inputs, frame after frame,
// and writes every image the engine produces (REngine::debugImages) after each frame, to compare with the capture's
// per-stage surfaces. No CUDA, NGX or Wine. The inputs come from engine/rr/rr_inputs.py.
//
// usage: d4r-rr MODEL_DIR INPUT_DIR OUT_DIR
//   INPUT_DIR: frames.txt ("size RW RH OW OH", then one line per frame: "frame F jx jy mvx mvy preExposure reset
//   depthInverted" followed by 16 worldToView and 16 viewToClip floats) and F-color.bin (RGBA16F), F-motion.bin
//   (RG16F), F-depth.bin (R32F), F-normals.bin and F-roughness.bin (RGBA16F), F-diffuse.bin and F-specular.bin
//   (RGBA8 UNORM), all at the render size.
//   OUT_DIR receives F-NAME.bin for every debug image of frame F.
#include "runtime.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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
    std::vector<char> data(static_cast<size_t>(stream.tellg()));
    stream.seekg(0);
    stream.read(data.data(), static_cast<std::streamsize>(data.size()));
    return data;
}
uint32_t texelBytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM: return 4;
    case VK_FORMAT_R16_SFLOAT: return 2;
    default: throw std::runtime_error("texelBytes: format");
    }
}
struct Ctx {
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties memory;
    uint32_t type(uint32_t bits, VkMemoryPropertyFlags flags) const {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
        throw std::runtime_error("memory type");
    }
    template <class F> void submit(F&& record) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        VkCommandBuffer cb; CK(vkAllocateCommandBuffers(device, &ai, &cb));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CK(vkBeginCommandBuffer(cb, &bi)); record(cb); CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        CK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(queue));
        vkFreeCommandBuffers(device, pool, 1, &cb);
    }
};
struct HostBuffer {
    VkBuffer buffer; VkDeviceMemory memory; void* mapped;
    HostBuffer(Ctx& c, VkDeviceSize size) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        CK(vkCreateBuffer(c.device, &bi, nullptr, &buffer));
        VkMemoryRequirements r; vkGetBufferMemoryRequirements(c.device, buffer, &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = r.size;
        ai.memoryTypeIndex = c.type(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        CK(vkAllocateMemory(c.device, &ai, nullptr, &memory)); CK(vkBindBufferMemory(c.device, buffer, memory, 0));
        CK(vkMapMemory(c.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    }
};
void copyRegion(VkBufferImageCopy& r, uint32_t w, uint32_t h) {
    r = {}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {w, h, 1};
}
struct Input {
    VkImage image; VkImageView view; VkFormat format;
    Input(Ctx& c, VkFormat f, uint32_t w, uint32_t h) : format(f) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = f; ii.extent = {w, h, 1}; ii.mipLevels = ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        CK(vkCreateImage(c.device, &ii, nullptr, &image));
        VkMemoryRequirements r; vkGetImageMemoryRequirements(c.device, image, &r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = r.size;
        ai.memoryTypeIndex = c.type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m; CK(vkAllocateMemory(c.device, &ai, nullptr, &m)); CK(vkBindImageMemory(c.device, image, m, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = f; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        CK(vkCreateImageView(c.device, &vi, nullptr, &view));
    }
};
void barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.image = image; b.oldLayout = from; b.newLayout = to;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 4) { std::cerr << "usage: d4r-rr MODEL_DIR INPUT_DIR OUT_DIR\n"; return 2; }
    const std::string modelDir = argv[1], inputDir = argv[2], outDir = argv[3];
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance instance; CK(vkCreateInstance(&ici, nullptr, &instance));
    uint32_t count = 8; VkPhysicalDevice devices[8]; vkEnumeratePhysicalDevices(instance, &count, devices);
    Ctx c{};
    c.physical = devices[0];
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devices[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) c.physical = devices[i];
    }
    vkGetPhysicalDeviceMemoryProperties(c.physical, &c.memory);
    VkPhysicalDeviceRobustness2FeaturesEXT rb2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, &rb2};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, &f8};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cm};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    vkGetPhysicalDeviceFeatures2(c.physical, &f2);
    uint32_t ne = 0; CK(vkEnumerateDeviceExtensionProperties(c.physical, nullptr, &ne, nullptr));
    std::vector<VkExtensionProperties> ep(ne); CK(vkEnumerateDeviceExtensionProperties(c.physical, nullptr, &ne, ep.data()));
    auto hasExtension = [&](const char* name) {
        return std::any_of(ep.begin(), ep.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
    };
    if (!hasExtension(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME))
        f8.shaderFloat8 = f8.shaderFloat8CooperativeMatrix = VK_FALSE;
    std::vector<const char*> extensions;
    if (cm.cooperativeMatrix) extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    if (f8.shaderFloat8) extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    uint32_t families = 8; VkQueueFamilyProperties qf[8]; vkGetPhysicalDeviceQueueFamilyProperties(c.physical, &families, qf);
    uint32_t family = 0;
    for (uint32_t i = 0; i < families; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
    float priority = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2}; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    // D4R_RR_TEST_ROBUST2=1: robustBufferAccess2 / robustImageAccess2 as vkd3d-proton enables them (out-of-bounds
    // reads return zero)
    const bool robust2 = std::getenv("D4R_RR_TEST_ROBUST2") != nullptr;
    if (robust2 && hasExtension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)) extensions.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    else { rb2.robustBufferAccess2 = rb2.robustImageAccess2 = rb2.nullDescriptor = VK_FALSE; }
    // Omit unsupported extension feature structures as well as their extension names. gfx11 RR uses f16 matrices.
    v13.pNext = cm.cooperativeMatrix ? static_cast<void*>(&cm) : nullptr;
    cm.pNext = f8.shaderFloat8 ? static_cast<void*>(&f8) : nullptr;
    f8.pNext = robust2 && hasExtension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME) ? &rb2 : nullptr;
    if (robust2 && hasExtension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME) && !f8.shaderFloat8) {
        if (cm.cooperativeMatrix) cm.pNext = &rb2; else v13.pNext = &rb2;
    }
    di.enabledExtensionCount = static_cast<uint32_t>(extensions.size()); di.ppEnabledExtensionNames = extensions.data();
    // D4R_RR_TEST_NO_ROBUST=1: no robust buffer or image access (out-of-bounds reads are undefined)
    if (std::getenv("D4R_RR_TEST_NO_ROBUST")) { f2.features.robustBufferAccess = VK_FALSE; v13.robustImageAccess = VK_FALSE; }
    // D4R_RR_TEST_NO_COOPMAT=1: a device like vkd3d-proton's, without the cooperative-matrix and float8 features
    if (std::getenv("D4R_RR_TEST_NO_COOPMAT")) { di.enabledExtensionCount = 0; v13.pNext = nullptr; }
    // D4R_RR_TEST_NO_INT64_ATOMICS=1: no 64-bit buffer atomics (preset F's exposure then takes its two passes)
    if (std::getenv("D4R_RR_TEST_NO_INT64_ATOMICS")) v12.shaderBufferInt64Atomics = v12.shaderSharedInt64Atomics = VK_FALSE;
    CK(vkCreateDevice(c.physical, &di, nullptr, &c.device));
    vkGetDeviceQueue(c.device, family, 0, &c.queue);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = family;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CK(vkCreateCommandPool(c.device, &pi, nullptr, &c.pool));

    auto framesText = read(inputDir + "/frames.txt");
    std::istringstream frames(std::string(framesText.begin(), framesText.end()));
    std::string key;
    uint32_t rw, rh, ow, oh;
    frames >> key >> rw >> rh >> ow >> oh;
    d4r::RModel model(c.physical, c.device, modelDir, VK_NULL_HANDLE, v12.shaderBufferInt64Atomics == VK_TRUE);
    d4r::REngine engine(model, rw, rh, ow, oh);
    const char* names[] = {"color", "motion", "depth", "normals", "roughness", "diffuse", "specular"};
    const VkFormat formats[] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32_SFLOAT,
                                VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM,
                                VK_FORMAT_R8G8B8A8_UNORM};
    std::vector<Input> inputs;
    for (VkFormat f : formats) inputs.emplace_back(c, f, rw, rh);
    const char* optionalNames[] = {"hitdistance", "disocclusion", "sss", "particles"};
    const VkFormat optionalFormats[] = {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT};
    std::vector<Input> optional;
    for (VkFormat f : optionalFormats) optional.emplace_back(c, f, rw, rh);
    // large enough for every debug image: preset F's intermediate history is 1.5 times the output size on both axes
    HostBuffer staging(c, VkDeviceSize(ow) * oh * 8 * 4);
    // GPU time of every frame: REngine's four stamps (start, after exposure, after the network, after the output)
    VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qp.queryType = VK_QUERY_TYPE_TIMESTAMP; qp.queryCount = 32;
    // D4R_ENGINE_RR_STAGE_STAMPS=1: the time of every network stage and output group as well (best frame of the repeats)
    const bool stageTimes = std::getenv("D4R_ENGINE_RR_STAGE_STAMPS") != nullptr;
    VkQueryPool stamps; CK(vkCreateQueryPool(c.device, &qp, nullptr, &stamps));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c.physical, &props);
    const int repeat = std::getenv("D4R_RR_TIME_REPEAT") ? std::atoi(std::getenv("D4R_RR_TIME_REPEAT")) : 1;
    bool first = true;
    while (frames >> key) {
        uint32_t f;
        d4r::RFrameSettings s;
        int reset, inverted;
        frames >> f >> s.jitter[0] >> s.jitter[1] >> s.motionScale[0] >> s.motionScale[1] >> s.preExposure >> reset >> inverted;
        for (float& v : s.worldToView) frames >> v;
        for (float& v : s.viewToClip) frames >> v;
        s.reset = reset != 0; s.depthInverted = (inverted & 1) != 0; s.hardwareDepth = (inverted & 2) != 0;   // bit 1: DLSS.Use.HW.Depth
        s.roughnessPacked = (inverted & 4) != 0;   // bit 2: DLSS.Roughness.Mode (preset F inputs)
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto data = read(inputDir + "/" + std::to_string(f) + "-" + names[i] + ".bin");
            if (data.size() != size_t(rw) * rh * texelBytes(formats[i])) throw std::runtime_error(std::string("input size: ") + names[i]);
            std::memcpy(staging.mapped, data.data(), data.size());
            c.submit([&](VkCommandBuffer cb) {
                barrier(cb, inputs[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                VkBufferImageCopy r; copyRegion(r, rw, rh);
                vkCmdCopyBufferToImage(cb, staging.buffer, inputs[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
                barrier(cb, inputs[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
            });
        }
        d4r::RFrameImages images{};
        images.color = inputs[0].view; images.motion = inputs[1].view; images.depth = inputs[2].view;
        images.normals = inputs[3].view; images.roughness = inputs[4].view;
        images.diffuseAlbedo = inputs[5].view; images.specularAlbedo = inputs[6].view;
        // optional inputs, present when the capture had them (rr_inputs.py)
        for (int k = 0; k < 4; ++k) {
            std::ifstream probe(inputDir + "/" + std::to_string(f) + "-" + optionalNames[k] + ".bin", std::ios::binary);
            if (!probe) continue;
            auto data = read(inputDir + "/" + std::to_string(f) + "-" + optionalNames[k] + ".bin");
            if (data.size() != size_t(rw) * rh * texelBytes(optionalFormats[k])) throw std::runtime_error(std::string("input size: ") + optionalNames[k]);
            std::memcpy(staging.mapped, data.data(), data.size());
            c.submit([&](VkCommandBuffer cb) {
                barrier(cb, optional[k].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                VkBufferImageCopy region; copyRegion(region, rw, rh);
                vkCmdCopyBufferToImage(cb, staging.buffer, optional[k].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                barrier(cb, optional[k].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
            });
            VkImageView* slots[] = {&images.specularHitDistance, &images.disocclusionMask, &images.subsurfaceGuide, &images.colorBeforeParticles};
            *slots[k] = optional[k].view;
        }
        engine.setFrame(images, s);
        if (first) c.submit([&](VkCommandBuffer cb) { model.recordUpload(cb); engine.recordInitialize(cb); first = false; });
        double best = 1e30, network = 0, output = 0, exposure = 0;
        std::vector<double> stageBest(28, 1e30);
        for (int r = 0; r < repeat; ++r) {
            if (r) engine.setFrame(images, s);
            c.submit([&](VkCommandBuffer cb) {
                vkCmdResetQueryPool(cb, stamps, 0, 32);
                engine.recordFrame(cb, stamps, 0);
            });
            uint64_t t[4];
            CK(vkGetQueryPoolResults(c.device, stamps, 0, 4, sizeof(t), t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const double ms = props.limits.timestampPeriod * 1e-6;
            if (stageTimes) {   // per stage: the minimum over the repeats (stamps 1, then 4 ... as written)
                uint64_t all[32][2];
                vkGetQueryPoolResults(c.device, stamps, 0, 32, sizeof(all), all, 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
                uint64_t prev = all[1][0];
                for (uint32_t q = 4; q < 32 && all[q][1]; ++q) {
                    stageBest[q - 4] = std::min(stageBest[q - 4], (all[q][0] - prev) * ms);
                    prev = all[q][0];
                }
            }
            if ((t[3] - t[0]) * ms < best) {
                best = (t[3] - t[0]) * ms; exposure = (t[1] - t[0]) * ms; network = (t[2] - t[1]) * ms; output = (t[3] - t[2]) * ms;
            }
        }
        std::cout << "frame " << f << ": " << best << " ms (exposure " << exposure << ", network " << network
                  << ", output " << output << ")\n";
        if (stageTimes) {
            std::cout << "stages (us):";
            for (double t : stageBest) if (t < 1e29) std::cout << " " << static_cast<int>(t * 1000 + 0.5);
            std::cout << "\n";
        }
        for (const auto& d : engine.debugImages()) {
            const size_t bytes = size_t(d.width) * d.height * texelBytes(d.format);
            c.submit([&](VkCommandBuffer cb) {
                VkBufferImageCopy r; copyRegion(r, d.width, d.height);
                vkCmdCopyImageToBuffer(cb, d.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &r);
            });
            std::ofstream(outDir + "/" + std::to_string(f) + "-" + d.name + ".bin", std::ios::binary)
                .write(static_cast<const char*>(staging.mapped), static_cast<std::streamsize>(bytes));
        }
        std::cout << "frame " << f << " done\n";
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "d4r-rr: " << e.what() << "\n";
    return 1;
}
