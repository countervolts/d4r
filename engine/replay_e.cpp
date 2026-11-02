// Native Super Resolution preset E/F replay: runs EEngine (engine/runtime_e.inc) on a capture's game inputs, frame
// after frame, and writes the output (and with D4R_E_DUMP_TENSORS=1 the network's tensors) after each frame.
//
// usage: d4r-e MODEL_DIR INPUT_DIR OUT_DIR
//   INPUT_DIR: frames.txt ("size RW RH OW OH HDR LOWRES MW MH [AUTOEXPOSURE]", then per frame
//   "frame F jx jy mvsx mvsy reset [preExposure exposureScale]") and
//   F-color.bin (RGBA16F, render size), F-motion.bin (RG16F, MW x MH),
//   F-depth.bin (R32F, render size, with LOWRES), optionally F-exposure.bin (R32F 1x1: the game's exposure, HDR). D4R_E_TIME_REPEAT=N: best of N GPU timings per frame.
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
    if (argc != 4) { std::cerr << "usage: d4r-e MODEL_DIR INPUT_DIR OUT_DIR\n"; return 2; }
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
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    vkGetPhysicalDeviceFeatures2(c.physical, &f2);
    uint32_t families = 8; VkQueueFamilyProperties qf[8]; vkGetPhysicalDeviceQueueFamilyProperties(c.physical, &families, qf);
    uint32_t family = 0;
    for (uint32_t i = 0; i < families; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
    float priority = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2}; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    CK(vkCreateDevice(c.physical, &di, nullptr, &c.device));
    vkGetDeviceQueue(c.device, family, 0, &c.queue);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = family;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CK(vkCreateCommandPool(c.device, &pi, nullptr, &c.pool));

    auto framesText = read(inputDir + "/frames.txt");
    std::istringstream frames(std::string(framesText.begin(), framesText.end()));
    std::string key;
    uint32_t rw, rh, ow, oh, hdr, lowRes, mw, mh, autoExposure = 0;
    std::string sizeLine;
    std::getline(frames, sizeLine);
    std::istringstream size(sizeLine);
    size >> key >> rw >> rh >> ow >> oh >> hdr >> lowRes >> mw >> mh;
    size >> autoExposure;
    d4r::EModel model(c.physical, c.device, modelDir);
    d4r::EEngine engine(model, rw, rh, ow, oh, hdr != 0, lowRes != 0, autoExposure != 0);
    Input color(c, VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh), motion(c, VK_FORMAT_R16G16_SFLOAT, mw, mh), depth(c, VK_FORMAT_R32_SFLOAT, rw, rh);
    Input exposure(c, VK_FORMAT_R32_SFLOAT, 1, 1);
    HostBuffer staging(c, std::max<VkDeviceSize>(VkDeviceSize(ow) * oh * 8, 64u << 20));
    VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qp.queryType = VK_QUERY_TYPE_TIMESTAMP; qp.queryCount = 4;
    VkQueryPool stamps; CK(vkCreateQueryPool(c.device, &qp, nullptr, &stamps));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c.physical, &props);
    const int repeat = std::getenv("D4R_E_TIME_REPEAT") ? std::atoi(std::getenv("D4R_E_TIME_REPEAT")) : 1;
    bool first = true;
    auto upload = [&](Input& in, const std::string& name, uint32_t f, uint32_t w, uint32_t h, uint32_t texel) {
        auto data = read(inputDir + "/" + std::to_string(f) + "-" + name + ".bin");
        if (data.size() != size_t(w) * h * texel) throw std::runtime_error("input size: " + name);
        std::memcpy(staging.mapped, data.data(), data.size());
        c.submit([&](VkCommandBuffer cb) {
            barrier(cb, in.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy r; copyRegion(r, w, h);
            vkCmdCopyBufferToImage(cb, staging.buffer, in.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            barrier(cb, in.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        });
    };
    for (std::string line; std::getline(frames, line);) {
        std::istringstream fields(line);
        if (!(fields >> key)) continue;
        uint32_t f; int reset;
        d4r::EFrameSettings s;
        fields >> f >> s.jitter[0] >> s.jitter[1] >> s.motionScale[0] >> s.motionScale[1] >> reset;
        if (!(fields >> s.preExposure >> s.exposureScale)) s.preExposure = s.exposureScale = 1;
        s.reset = reset != 0;
        upload(color, "color", f, rw, rh, 8);
        upload(motion, "motion", f, mw, mh, 4);
        if (lowRes) upload(depth, "depth", f, rw, rh, 4);
        const bool gameExposure = std::ifstream(inputDir + "/" + std::to_string(f) + "-exposure.bin").good();
        if (gameExposure) upload(exposure, "exposure", f, 1, 1, 4);
        d4r::FrameImages images{};
        images.color = color.view; images.motion = motion.view; images.depth = depth.view;
        if (gameExposure) images.exposure = exposure.view;
        engine.setFrame(images, s);
        if (first) c.submit([&](VkCommandBuffer cb) { model.recordUpload(cb); engine.recordInitialize(cb); first = false; });
        double best = 1e30, in = 0, net = 0, fin = 0;
        for (int r = 0; r < repeat; ++r) {
            if (r) engine.setFrame(images, s);
            c.submit([&](VkCommandBuffer cb) {
                vkCmdResetQueryPool(cb, stamps, 0, 4);
                engine.recordFrame(cb, stamps, 0);
            });
            uint64_t t[4];
            CK(vkGetQueryPoolResults(c.device, stamps, 0, 4, sizeof(t), t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const double ms = props.limits.timestampPeriod * 1e-6;
            if ((t[3] - t[0]) * ms < best) { best = (t[3] - t[0]) * ms; in = (t[1] - t[0]) * ms; net = (t[2] - t[1]) * ms; fin = (t[3] - t[2]) * ms; }
        }
        std::cout << "frame " << f << ": " << best << " ms (input " << in << ", network " << net << ", reconstruction " << fin << ")\n";
        c.submit([&](VkCommandBuffer cb) {
            VkBufferImageCopy r; copyRegion(r, ow, oh);
            vkCmdCopyImageToBuffer(cb, engine.outputImage(), VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &r);
        });
        std::ofstream(outDir + "/" + std::to_string(f) + "-out.bin", std::ios::binary)
            .write(static_cast<const char*>(staging.mapped), static_cast<std::streamsize>(size_t(ow) * oh * 8));
        if (std::getenv("D4R_E_DUMP_TENSORS"))
            for (const auto& d : engine.debugBuffers()) {
                c.submit([&](VkCommandBuffer cb) {
                    VkBufferCopy r{d.offset, 0, d.bytes};
                    vkCmdCopyBuffer(cb, d.buffer, staging.buffer, 1, &r);
                });
                std::ofstream(outDir + "/" + std::to_string(f) + "-" + d.name + ".bin", std::ios::binary)
                    .write(static_cast<const char*>(staging.mapped), static_cast<std::streamsize>(d.bytes));
            }
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "d4r-e: " << e.what() << "\n";
    return 1;
}
