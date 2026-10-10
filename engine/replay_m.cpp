// Preset M on the engine's runtime: runs raw frames and writes the output images. No CUDA/NGX.
//   d4r-m MODEL_DIR RENDER_W RENDER_H OUT_W OUT_H OUTPUT_PREFIX [repeats [hip]] -- COLOR MOTION DEPTH JX JY ...
// "hip" runs the network on the native HIP layer kernels (hip_net.h; the model needs compile_m.py --hip) between the
// engine's Vulkan stages, through two buffers shared by file descriptor.
// COLOR: RGBA16F, MOTION: RG16F, DEPTH: R32F, all at render size, tightly packed. The first frame resets.
#include "runtime.h"
#ifdef D4R_ENGINE_HIP
#include "hip_net.h"
#endif
#include <chrono>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) throw std::runtime_error(std::string(#x) + " failed: " + std::to_string(r_)); } while (0)
static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}
struct Ctx {
    VkInstance instance{}; VkPhysicalDevice physical{}; VkDevice device{}; VkQueue queue{}; VkCommandPool pool{}; VkPhysicalDeviceMemoryProperties memory{};
    uint32_t type(uint32_t bits, VkMemoryPropertyFlags flags) const {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
        throw std::runtime_error("no memory type");
    }
};
struct Staging { VkBuffer buffer{}; VkDeviceMemory memory{}; void* map{}; };
static Staging staging(const Ctx& c, VkDeviceSize size) {
    Staging s; VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CK(vkCreateBuffer(c.device, &bi, nullptr, &s.buffer));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(c.device, s.buffer, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = r.size; ai.memoryTypeIndex = c.type(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CK(vkAllocateMemory(c.device, &ai, nullptr, &s.memory)); CK(vkBindBufferMemory(c.device, s.buffer, s.memory, 0)); CK(vkMapMemory(c.device, s.memory, 0, VK_WHOLE_SIZE, 0, &s.map));
    return s;
}
// a storage buffer whose memory another API imports
struct Shared { VkBuffer buffer{}; VkDeviceMemory memory{}; VkDeviceSize allocation = 0; int fd = -1; };
static Shared shared(const Ctx& c, VkDeviceSize size) {
    Shared s; VkExternalMemoryBufferCreateInfo ei{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO}; ei.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ei}; bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    CK(vkCreateBuffer(c.device, &bi, nullptr, &s.buffer));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(c.device, s.buffer, &r);
    VkExportMemoryAllocateInfo xi{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO}; xi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &xi}; ai.allocationSize = s.allocation = r.size;
    ai.memoryTypeIndex = c.type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CK(vkAllocateMemory(c.device, &ai, nullptr, &s.memory)); CK(vkBindBufferMemory(c.device, s.buffer, s.memory, 0));
    VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR}; gi.memory = s.memory; gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    const auto getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(c.device, "vkGetMemoryFdKHR"));
    if (!getFd) throw std::runtime_error("vkGetMemoryFdKHR is unavailable");
    CK(getFd(c.device, &gi, &s.fd));
    return s;
}
struct Image { VkImage image{}; VkDeviceMemory memory{}; VkImageView view{}; };
static Image image(const Ctx& c, uint32_t w, uint32_t h, VkFormat f) {
    Image im; VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ci.imageType = VK_IMAGE_TYPE_2D; ci.format = f; ci.extent = {w, h, 1}; ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    CK(vkCreateImage(c.device, &ci, nullptr, &im.image));
    VkMemoryRequirements r; vkGetImageMemoryRequirements(c.device, im.image, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = r.size; ai.memoryTypeIndex = c.type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CK(vkAllocateMemory(c.device, &ai, nullptr, &im.memory)); CK(vkBindImageMemory(c.device, im.image, im.memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = im.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = f; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CK(vkCreateImageView(c.device, &vi, nullptr, &im.view));
    return im;
}
int main(int argc, char** argv) try {
    int sep = 0;
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--")) sep = i;
    if (sep < 7 || (argc - sep - 1) % 5 || argc - sep - 1 == 0) { std::fprintf(stderr, "usage: d4r-m MODEL_DIR RW RH OW OH OUTPUT_PREFIX [repeats [hip]] -- COLOR MOTION DEPTH JX JY ...\n"); return 2; }
    const bool hip = sep > 8 && !std::strcmp(argv[8], "hip");
#ifndef D4R_ENGINE_HIP
    if (hip) throw std::runtime_error("built without the HIP network (engine/build.sh needs ROCm)");
#endif
    const std::string model = argv[1], prefix = argv[6];
    const uint32_t rw = std::atoi(argv[2]), rh = std::atoi(argv[3]), ow = std::atoi(argv[4]), oh = std::atoi(argv[5]);
    const int repeats = sep > 7 ? std::atoi(argv[7]) : 0;
    const int frames = (argc - sep - 1) / 5;
    Ctx c;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    CK(vkCreateInstance(&ici, nullptr, &c.instance));
    uint32_t n = 8; VkPhysicalDevice pds[8]; vkEnumeratePhysicalDevices(c.instance, &n, pds);
    if (!n) throw std::runtime_error("no Vulkan GPU");
    c.physical = pds[0];
    for (uint32_t i = 0; i < n; ++i) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p); if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) c.physical = pds[i]; }
    vkGetPhysicalDeviceMemoryProperties(c.physical, &c.memory);
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, &f8};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cm};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    vkGetPhysicalDeviceFeatures2(c.physical, &f2);
    if (!cm.cooperativeMatrix || !f8.shaderFloat8 || !f8.shaderFloat8CooperativeMatrix) throw std::runtime_error("GPU lacks FP8 cooperative matrices (VK_EXT_shader_float8)");
    uint32_t nq = 8; VkQueueFamilyProperties qf[8]; vkGetPhysicalDeviceQueueFamilyProperties(c.physical, &nq, qf);
    uint32_t qi = 0; for (uint32_t i = 0; i < nq; ++i) if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits) { qi = i; break; }
    float prio = 1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    const char* exts[] = {"VK_KHR_cooperative_matrix", "VK_EXT_shader_float8", "VK_KHR_external_memory_fd"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.enabledExtensionCount = hip ? 3 : 2; dci.ppEnabledExtensionNames = exts;
    CK(vkCreateDevice(c.physical, &dci, nullptr, &c.device)); vkGetDeviceQueue(c.device, qi, 0, &c.queue);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = qi; pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CK(vkCreateCommandPool(c.device, &pi, nullptr, &c.pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = c.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(c.device, &cai, &cb));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 7;
    VkQueryPool qp; CK(vkCreateQueryPool(c.device, &qpi, nullptr, &qp));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c.physical, &props);
    auto submit = [&] {
        CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        CK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(c.queue));
    };
    auto begin = [&] { CK(vkResetCommandBuffer(cb, 0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi)); };
    {
        d4r::MModel m(c.physical, c.device, model);
        d4r::ExternalNetwork external;
#ifdef D4R_ENGINE_HIP
        std::unique_ptr<d4r::HipNetwork> network;
        if (hip) {
            Shared tokens = shared(c, d4r::MEngine::networkBufferBytes(rw, rh)), result = shared(c, d4r::MEngine::networkBufferBytes(rw, rh));
            external = {tokens.buffer, result.buffer};
            network = std::make_unique<d4r::HipNetwork>(model, rw, rh, tokens.fd, result.fd, tokens.allocation, result.allocation);
        }
#endif
        d4r::MEngine engine(m, rw, rh, ow, oh, hip ? &external : nullptr);
        Image color = image(c, rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT), motion = image(c, rw, rh, VK_FORMAT_R16G16_SFLOAT), depth = image(c, rw, rh, VK_FORMAT_R32_SFLOAT);
        const VkDeviceSize sizes[3] = {VkDeviceSize(rw) * rh * 8, VkDeviceSize(rw) * rh * 4, VkDeviceSize(rw) * rh * 4};
        Staging in[3] = {staging(c, sizes[0]), staging(c, sizes[1]), staging(c, sizes[2])}, out = staging(c, VkDeviceSize(ow) * oh * 8);
        Image* images[3] = {&color, &motion, &depth};
        begin(); m.recordUpload(cb); engine.recordInitialize(cb);
        for (Image* im : images) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.newLayout = VK_IMAGE_LAYOUT_GENERAL; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = im->image; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        }
        submit();
        std::vector<std::vector<char>> frameInputs;
        frameInputs.reserve(size_t(frames) * 3);
        for (int f = 0; f < frames; ++f)
            for (int k = 0; k < 3; ++k) {
                auto data = slurp(argv[sep + 1 + 5 * f + k]);
                if (data.size() != sizes[k]) throw std::runtime_error("wrong input size");
                frameInputs.push_back(std::move(data));
            }
        std::vector<double> times, walls; std::vector<std::vector<double>> stage(6);
        const auto now = [] { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        for (int pass = 0; pass <= repeats; ++pass)
            for (int f = 0; f < frames; ++f) {
                char** a = argv + sep + 1 + 5 * f;
                if (pass == 0 || frames > 1)
                    for (int k = 0; k < 3; ++k)
                        std::memcpy(in[k].map, frameInputs[3 * f + k].data(), sizes[k]);
                begin();
                VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; region.imageExtent = {rw, rh, 1};
                if (pass == 0 || frames > 1)
                    for (int k = 0; k < 3; ++k) vkCmdCopyBufferToImage(cb, in[k].buffer, images[k]->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
                d4r::MFrameSettings s; s.jitter[0] = std::atof(a[3]); s.jitter[1] = std::atof(a[4]); s.reset = pass == 0 && f == 0;
                const double t0 = now();
                engine.setFrame({color.view, motion.view, depth.view}, s);
                vkCmdResetQueryPool(cb, qp, 0, 7);
                double networkUs = 0;
                if (hip) {
#ifdef D4R_ENGINE_HIP
                    engine.recordFront(cb, qp, 0);
                    submit();
                    network->launch();
                    network->wait();
                    networkUs = network->lastMilliseconds() * 1000.0;
                    begin();
                    engine.recordBack(cb, qp, 3);
#endif
                } else engine.recordFrame(cb, qp, 0);
                region.imageExtent = {ow, oh, 1};
                if (pass == 0) vkCmdCopyImageToBuffer(cb, engine.outputImage(), VK_IMAGE_LAYOUT_GENERAL, out.buffer, 1, &region);
                submit();
                if (pass) walls.push_back(now() - t0);
                uint64_t t[7]; CK(vkGetQueryPoolResults(c.device, qp, 0, 7, sizeof(t), t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                const double tick = props.limits.timestampPeriod / 1000.0;
                double d[6]; for (int k = 0; k < 6; ++k) d[k] = double(t[k + 1] - t[k]) * tick;
                if (hip) d[2] = networkUs; // measured HIP network time; the complete span includes host handoffs
                const double us = double(t[6] - t[0]) * tick;
                if (pass) for (int k = 0; k < 6; ++k) stage[k].push_back(d[k]);
                if (pass == 0 && std::getenv("D4R_M_DUMP")) {
                    // development: every intermediate of this frame, to PREFIX-NAME-FRAME.bin
                    for (const auto& d : engine.debugResources()) {
                        Staging st = staging(c, d.bytes);
                        begin();
                        if (d.buffer) { VkBufferCopy bc{0, 0, d.bytes}; vkCmdCopyBuffer(cb, d.buffer, st.buffer, 1, &bc); }
                        else { VkBufferImageCopy ic{}; ic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; ic.imageExtent = {d.width, d.height, 1};
                               vkCmdCopyImageToBuffer(cb, d.image, VK_IMAGE_LAYOUT_GENERAL, st.buffer, 1, &ic); }
                        submit();
                        std::ofstream(std::string(std::getenv("D4R_M_DUMP")) + "-" + d.name + "-" + std::to_string(f) + ".bin", std::ios::binary)
                            .write(static_cast<const char*>(st.map), std::streamsize(d.bytes));
                        vkDestroyBuffer(c.device, st.buffer, nullptr); vkFreeMemory(c.device, st.memory, nullptr);
                    }
                }
                if (pass == 0) {
                    std::ofstream(prefix + "-" + std::to_string(f) + ".rgba16f", std::ios::binary).write(static_cast<const char*>(out.map), std::streamsize(ow) * oh * 8);
                    std::printf("frame %d: %.1f us GPU\n", f, us);
                } else times.push_back(us);
            }
        if (!times.empty()) {
            std::sort(times.begin(), times.end()); std::printf("%zu timed frames: median %.1f us, min %.1f us GPU\n", times.size(), times[times.size() / 2], times[0]);
            std::sort(walls.begin(), walls.end()); std::printf("  wall clock, record to completion: median %.1f us, min %.1f us\n", walls[walls.size() / 2], walls[0]);
            const char* names[6] = {"exposure", "input", "network", "expansion", "reconstruction", "downsample"};
            for (int k = 0; k < 6; ++k) { std::sort(stage[k].begin(), stage[k].end()); std::printf("  %-14s median %.1f us\n", names[k], stage[k][stage[k].size() / 2]); }
        }
        CK(vkDeviceWaitIdle(c.device));
    }
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "d4r-m: %s\n", e.what()); return 1; }
