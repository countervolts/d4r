// Preset M on the engine's runtime: runs raw frames and writes the output images. No CUDA/NGX.
//   d4r-m MODEL_DIR RENDER_W RENDER_H OUT_W OUT_H OUTPUT_PREFIX [repeats [hip]] -- COLOR MOTION DEPTH JX JY ...
// "hip" runs the network on the native HIP layer kernels (hip_net.h; the model needs compile_m.py --hip) between the
// engine's Vulkan stages, through two buffers shared by file descriptor (an NT handle on Windows). "vkhip" runs the
// same code objects as Vulkan pipelines in the engine's own command buffer (vkhip/vkhip_net.h, D4R_ENGINE_VKHIP builds).
// COLOR: RGBA16F, MOTION: RG16F, DEPTH: R32F, all at render size, tightly packed. The first frame resets.
#include "runtime.h"
#ifdef D4R_ENGINE_HIP
#include "hip_net.h"
#include "vk_external.h"
#endif
#ifdef D4R_ENGINE_VKHIP
#include "vkhip/vkhip_net.h"
#endif
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _MSC_VER
#include <immintrin.h>
#endif
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) throw std::runtime_error(std::string(#x) + " failed: " + std::to_string(r_)); } while (0)
// half <-> float, rounding to nearest even (MSVC has no _Float16: the F16C conversions, which round the same way)
static float halfToFloat(uint16_t h) {
#ifdef _MSC_VER
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#else
    _Float16 v; std::memcpy(&v, &h, 2); return float(v);
#endif
}
static uint16_t floatToHalf(float f) {
#ifdef _MSC_VER
    return uint16_t(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT)));
#else
    const _Float16 v = _Float16(f); uint16_t h; std::memcpy(&h, &v, 2); return h;
#endif
}
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
#ifdef D4R_ENGINE_HIP
// a storage buffer whose memory another API imports
struct Shared { VkBuffer buffer{}; VkDeviceMemory memory{}; VkDeviceSize allocation = 0; d4r::ExternalMemoryHandle fd{}; };
static Shared shared(const Ctx& c, VkDeviceSize size) {
    Shared s; VkExternalMemoryBufferCreateInfo ei{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO}; ei.handleTypes = d4r::ExternalMemoryHandleType;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ei}; bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    CK(vkCreateBuffer(c.device, &bi, nullptr, &s.buffer));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(c.device, s.buffer, &r);
    VkExportMemoryAllocateInfo xi{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO}; xi.handleTypes = d4r::ExternalMemoryHandleType;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &xi}; ai.allocationSize = s.allocation = r.size;
    ai.memoryTypeIndex = c.type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CK(vkAllocateMemory(c.device, &ai, nullptr, &s.memory)); CK(vkBindBufferMemory(c.device, s.buffer, s.memory, 0));
    s.fd = d4r::exportMemory(c.device, s.memory);
    return s;
}
#endif
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
    const bool vkhip = sep > 8 && !std::strcmp(argv[8], "vkhip");
#ifndef D4R_ENGINE_HIP
    if (hip) throw std::runtime_error("built without the HIP network (engine/build.sh needs ROCm)");
#endif
#ifndef D4R_ENGINE_VKHIP
    if (vkhip) throw std::runtime_error("built without vkhip (engine/build.sh needs glslangValidator)");
#endif
    const std::string model = argv[1], prefix = argv[6];
    const uint32_t rw = std::atoi(argv[2]), rh = std::atoi(argv[3]), ow = std::atoi(argv[4]), oh = std::atoi(argv[5]);
    const int repeats = sep > 7 ? std::atoi(argv[7]) : 0;
    const int frames = (argc - sep - 1) / 5;
    Ctx c;
#ifdef D4R_ENGINE_VOLK
    if (volkInitialize() != VK_SUCCESS) throw std::runtime_error("Vulkan loader initialization failed");
#endif
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    CK(vkCreateInstance(&ici, nullptr, &c.instance));
#ifdef D4R_ENGINE_VOLK
    volkLoadInstance(c.instance);
#endif
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
    // without FP8 cooperative matrices (RDNA2) the model's v_dot2 stages run (MModel dot2)
    // Querying an unsupported extension's feature structure is not a support
    // check: some drivers populate it even when the extension is unavailable.
    uint32_t extensionCount = 0;
    CK(vkEnumerateDeviceExtensionProperties(c.physical, nullptr, &extensionCount, nullptr));
    std::vector<VkExtensionProperties> extensions(extensionCount);
    CK(vkEnumerateDeviceExtensionProperties(c.physical, nullptr, &extensionCount, extensions.data()));
    bool hasMatrix = false, hasFloat8 = false, hasBinary = false;
    for (const auto& extension : extensions) {
        hasMatrix |= !std::strcmp(extension.extensionName, "VK_KHR_cooperative_matrix");
        hasFloat8 |= !std::strcmp(extension.extensionName, "VK_EXT_shader_float8");
        hasBinary |= !std::strcmp(extension.extensionName, "VK_KHR_pipeline_binary");
    }
    // vkhip on AMD's own driver (Windows) injects through pipeline binaries (vkhip.h); RADV through its pipeline cache
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 driverProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
    vkGetPhysicalDeviceProperties2(c.physical, &driverProps);
    const bool binaries = vkhip && driver.driverID != VK_DRIVER_ID_MESA_RADV;
    if (binaries && !hasBinary) throw std::runtime_error("vkhip needs VK_KHR_pipeline_binary outside RADV");
    const bool dot2 = !hasMatrix || !hasFloat8 || !cm.cooperativeMatrix || !f8.shaderFloat8 || !f8.shaderFloat8CooperativeMatrix;
    if (dot2 && (hip || vkhip)) throw std::runtime_error("GPU lacks FP8 cooperative matrices (VK_EXT_shader_float8)");
    if (dot2) { v13.pNext = nullptr; vkGetPhysicalDeviceFeatures2(c.physical, &f2); }
    uint32_t nq = 8; VkQueueFamilyProperties qf[8]; vkGetPhysicalDeviceQueueFamilyProperties(c.physical, &nq, qf);
    uint32_t qi = 0; for (uint32_t i = 0; i < nq; ++i) if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits) { qi = i; break; }
    float prio = 1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    std::vector<const char*> exts;
    if (!dot2) exts = {"VK_KHR_cooperative_matrix", "VK_EXT_shader_float8"};
#ifdef _WIN32
    if (hip) exts.push_back("VK_KHR_external_memory_win32");
#else
    if (hip) exts.push_back("VK_KHR_external_memory_fd");
#endif
    VkPhysicalDeviceMaintenance5FeaturesKHR m5{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR};
    VkPhysicalDevicePipelineBinaryFeaturesKHR pb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR, &m5};
    if (binaries) {
        exts.push_back("VK_KHR_pipeline_binary"); exts.push_back("VK_KHR_maintenance5");
        m5.maintenance5 = pb.pipelineBinaries = VK_TRUE;
        m5.pNext = f2.pNext; f2.pNext = &pb;
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.enabledExtensionCount = uint32_t(exts.size()); dci.ppEnabledExtensionNames = exts.data();
    CK(vkCreateDevice(c.physical, &dci, nullptr, &c.device));
#ifdef D4R_ENGINE_VOLK
    volkLoadDevice(c.device);
#endif
    vkGetDeviceQueue(c.device, qi, 0, &c.queue);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = qi; pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CK(vkCreateCommandPool(c.device, &pi, nullptr, &c.pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = c.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(c.device, &cai, &cb));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 17;
    VkQueryPool qp; CK(vkCreateQueryPool(c.device, &qpi, nullptr, &qp));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c.physical, &props);
    auto submit = [&] {
        CK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        CK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(c.queue));
    };
    auto begin = [&] { CK(vkResetCommandBuffer(cb, 0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi)); };
    {
        d4r::MModel m(c.physical, c.device, model, VK_NULL_HANDLE, dot2);
        d4r::ExternalNetwork external;
#ifdef D4R_ENGINE_VKHIP
        // the render GPU's machine from its device ID would need sysfs; the replay takes whatever the model was built for
        vkhip::MNetwork native;
        if (vkhip) {
            vkhip::NetFiles files;
            files.pipelineBinary = binaries;
            auto bytes = slurp(model + "/hipnet.bin"); files.hipnet.assign(bytes.begin(), bytes.end());
            for (const char* name : {"rrlite_enc1_4x4", "rrlite_enc2_4x4", "rrlite_enc3_tube_4x4", "rrlite_dec2_4x4", "rrlite_dec1_4x4"}) {
                bytes = slurp(model + "/hip/" + name + ".hsaco"); files.code[name].assign(bytes.begin(), bytes.end());
            }
            std::string error;
            if (!native.create(c.device, vkGetDeviceProcAddr, c.memory, files, rw, rh, d4r::MEngine::networkBufferBytes(rw, rh), error))
                throw std::runtime_error("vkhip: " + error);
            external = {native.tokens(), native.head()};
            if (std::getenv("D4R_VKHIP_STAMPS"))
                native.enableStamps([](const char* line) { std::printf("%s\n", line); }, props.limits.timestampPeriod);
        }
#endif
#ifdef D4R_ENGINE_HIP
        std::unique_ptr<d4r::HipNetwork> network;
        if (hip) {
            Shared tokens = shared(c, d4r::MEngine::networkBufferBytes(rw, rh)), result = shared(c, d4r::MEngine::networkBufferBytes(rw, rh));
            external = {tokens.buffer, result.buffer};
            network = std::make_unique<d4r::HipNetwork>(model, rw, rh, tokens.fd, result.fd, tokens.allocation, result.allocation);
        }
#endif
        // D4R_M_CAPACITY=WxH: the engine reserves a larger render size than the frames use (dynamic resolution, as a
        // shim reserving more than the feature's size would)
        uint32_t cw = rw, ch = rh;
        if (const char* capacity = std::getenv("D4R_M_CAPACITY")) std::sscanf(capacity, "%ux%u", &cw, &ch);
        d4r::MEngine engine(m, cw, ch, ow, oh, hip || vkhip ? &external : nullptr);
        // D4R_M_DISPLAY_MV=1: the frames' motion is resampled to display resolution (nearest, in output pixels) and
        // passed as display-resolution motion (MFrameSettings::displayMotion)
        const bool displayMv = std::getenv("D4R_M_DISPLAY_MV") != nullptr;
        const uint32_t mw = displayMv ? ow : rw, mh = displayMv ? oh : rh;
        Image color = image(c, rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT), motion = image(c, mw, mh, VK_FORMAT_R16G16_SFLOAT), depth = image(c, rw, rh, VK_FORMAT_R32_SFLOAT);
        const VkDeviceSize sizes[3] = {VkDeviceSize(rw) * rh * 8, VkDeviceSize(mw) * mh * 4, VkDeviceSize(rw) * rh * 4};
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
                if (k == 1 && displayMv) {
                    if (data.size() != size_t(rw) * rh * 4) throw std::runtime_error("wrong input size");
                    std::vector<char> hi(sizes[1]);
                    const uint16_t* lo = reinterpret_cast<const uint16_t*>(data.data());
                    uint16_t* d = reinterpret_cast<uint16_t*>(hi.data());
                    for (uint32_t y = 0; y < oh; ++y)
                        for (uint32_t x = 0; x < ow; ++x) {
                            const size_t t = size_t(uint64_t(y) * rh / oh) * rw + uint64_t(x) * rw / ow;
                            d[2 * (size_t(y) * ow + x)] = floatToHalf(halfToFloat(lo[2 * t]) * float(ow) / float(rw));
                            d[2 * (size_t(y) * ow + x) + 1] = floatToHalf(halfToFloat(lo[2 * t + 1]) * float(oh) / float(rh));
                        }
                    data = std::move(hi);
                }
                if (data.size() != sizes[k]) throw std::runtime_error("wrong input size");
                frameInputs.push_back(std::move(data));
            }
        std::vector<double> times, walls; std::vector<std::vector<double>> stage(6), layer(10);
        const auto now = [] { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        // D4R_M_RESET_ALL=1: every frame is a reset frame, as the D3D12 harness passes them without its temporal option
        const bool resetAll = std::getenv("D4R_M_RESET_ALL") != nullptr;
        // D4R_M_WARMUP_PASSES=N: leave the first N passes untimed, so a case gets the official probe's warmed
        // interval (N * frames evaluations) before the measured ones. D4R_M_TIME_EACH=1 prints every timed frame.
        const int warmupPasses = std::getenv("D4R_M_WARMUP_PASSES") ? std::atoi(std::getenv("D4R_M_WARMUP_PASSES")) : 0;
        const bool timeEach = std::getenv("D4R_M_TIME_EACH") != nullptr;
        for (int pass = 0; pass <= repeats; ++pass)
            for (int f = 0; f < frames; ++f) {
                char** a = argv + sep + 1 + 5 * f;
                if (pass == 0 || frames > 1)
                    for (int k = 0; k < 3; ++k)
                        std::memcpy(in[k].map, frameInputs[3 * f + k].data(), sizes[k]);
                begin();
                VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                if (pass == 0 || frames > 1)
                    for (int k = 0; k < 3; ++k) {
                        region.imageExtent = k == 1 ? VkExtent3D{mw, mh, 1} : VkExtent3D{rw, rh, 1};
                        vkCmdCopyBufferToImage(cb, in[k].buffer, images[k]->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
                    }
                d4r::MFrameSettings s; s.jitter[0] = std::atof(a[3]); s.jitter[1] = std::atof(a[4]); s.reset = (pass == 0 && f == 0) || resetAll;
                s.renderWidth = rw; s.renderHeight = rh; s.displayMotion = displayMv;
                const double t0 = now();
                engine.setFrame({color.view, motion.view, depth.view}, s);
                vkCmdResetQueryPool(cb, qp, 0, 17);
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
                } else if (vkhip) {
#ifdef D4R_ENGINE_VKHIP
                    engine.recordFront(cb, qp, 0);
                    native.record(cb);
                    engine.recordBack(cb, qp, 3);
#endif
                } else engine.recordFrame(cb, qp, 0);
                region.imageExtent = {ow, oh, 1};
                if (pass == 0) vkCmdCopyImageToBuffer(cb, engine.outputImage(), VK_IMAGE_LAYOUT_GENERAL, out.buffer, 1, &region);
                submit();
                if (pass > warmupPasses) walls.push_back(now() - t0);
                uint64_t t[17] = {};
                // D4R_ENGINE_M_LAYER_STAMPS=1: the ten network layers' stamps as well (queries 7 - 16)
                static const bool layerStamps = std::getenv("D4R_ENGINE_M_LAYER_STAMPS") != nullptr && !hip;
                CK(vkGetQueryPoolResults(c.device, qp, 0, layerStamps ? 17 : 7, sizeof(t), t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                const double tick = props.limits.timestampPeriod / 1000.0;
                double d[6]; for (int k = 0; k < 6; ++k) d[k] = double(t[k + 1] - t[k]) * tick;
                if (hip) d[2] = networkUs; // measured HIP network time; the complete span includes host handoffs
                const double us = double(t[6] - t[0]) * tick;
                if (pass > warmupPasses) for (int k = 0; k < 6; ++k) stage[k].push_back(d[k]);
                if (pass > warmupPasses && layerStamps) for (int l = 0; l < 10; l++) layer[l].push_back(double(t[7 + l] - (l ? t[6 + l] : t[2])) * tick);
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
                } else if (pass > warmupPasses) {
                    times.push_back(us);
                    if (timeEach) std::printf("TIMED pass=%d frame=%d %.3f us\n", pass, f, us);
                }
            }
        if (!times.empty()) {
            std::sort(times.begin(), times.end()); std::printf("%zu timed frames: median %.1f us, min %.1f us GPU\n", times.size(), times[times.size() / 2], times[0]);
            std::sort(walls.begin(), walls.end()); std::printf("  wall clock, record to completion: median %.1f us, min %.1f us\n", walls[walls.size() / 2], walls[0]);
            const char* names[6] = {"exposure", "input", "network", "expansion", "reconstruction", "downsample"};
            for (int k = 0; k < 6; ++k) { std::sort(stage[k].begin(), stage[k].end()); std::printf("  %-14s median %.1f us\n", names[k], stage[k][stage[k].size() / 2]); }
            if (!layer[0].empty()) {
                std::printf("  layers (us):");
                for (auto& v : layer) { std::sort(v.begin(), v.end()); std::printf(" %.1f", v[v.size() / 2]); }
                std::printf("\n");
            }
        }
        CK(vkDeviceWaitIdle(c.device));
    }
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "d4r-m: %s\n", e.what()); return 1; }
