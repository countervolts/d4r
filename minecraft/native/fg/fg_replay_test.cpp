// Offline correctness/perf test for the native Lite engine: replays a DLSS-G harness capture directory
// (res_rRR_*.bin + params_rRR_eK.txt, as written by fg/dlssg/harness --replay) and writes rRR_gK.ppm.
// usage: fg_replay_test SCRIPT_DIR CACHE CAPTURE_DIR OUT_DIR [frames]
#include "fg_engine.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <chrono>
#include <sys/stat.h>
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <algorithm>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

static std::vector<uint8_t> slurp(const std::string &p) { std::ifstream f(p, std::ios::binary); std::vector<uint8_t> v((std::istreambuf_iterator<char>(f)), {}); return v; }

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: %s SCRIPT_DIR CACHE CAPTURE_DIR OUT_DIR [frames]\n", argv[0]); return 2; }
    std::string sdir = argv[1], cache = argv[2], cap = argv[3], out = argv[4]; int frames = argc > 5 ? atoi(argv[5]) : 8;
    mkdir(out.c_str(), 0755);
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW); auto gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    auto ci_ = (PFN_vkCreateInstance)gipa(nullptr, "vkCreateInstance");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst; CK(ci_(&ici, nullptr, &inst));
#define GI(n) auto n = (PFN_vk##n)gipa(inst, "vk" #n)
    GI(EnumeratePhysicalDevices); GI(GetPhysicalDeviceProperties); GI(GetPhysicalDeviceQueueFamilyProperties); GI(CreateDevice); GI(GetDeviceProcAddr);
    GI(GetPhysicalDeviceMemoryProperties);
    uint32_t n = 8; VkPhysicalDevice pds[8]; EnumeratePhysicalDevices(inst, &n, pds);
    VkPhysicalDevice pd = 0;
    for (uint32_t i = 0; i < n; i++) { VkPhysicalDeviceProperties pp; GetPhysicalDeviceProperties(pds[i], &pp); if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = pds[i]; fprintf(stderr, "gpu: %s\n", pp.deviceName); break; } }
    if (!pd) return 1;
    uint32_t qn = 16; VkQueueFamilyProperties qp[16]; GetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp);
    uint32_t qf = 0; for (uint32_t i = 0; i < qn; i++) if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { qf = i; break; }
    float prio = 1; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = qf; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    const char *exts[] = {"VK_NVX_binary_import", "VK_NVX_image_view_handle"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.enabledExtensionCount = 2; dci.ppEnabledExtensionNames = exts; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qi;
    VkDevice dev; CK(CreateDevice(pd, &dci, nullptr, &dev));
    auto GD = (PFN_vkGetDeviceProcAddr)gipa(inst, "vkGetDeviceProcAddr");
#define FD(n) auto n = (PFN_vk##n)GD(dev, "vk" #n)
    FD(GetDeviceQueue); FD(CreateBuffer); FD(GetBufferMemoryRequirements); FD(AllocateMemory); FD(BindBufferMemory); FD(MapMemory);
    FD(CreateCommandPool); FD(AllocateCommandBuffers); FD(BeginCommandBuffer); FD(EndCommandBuffer); FD(QueueSubmit); FD(QueueWaitIdle);
    FD(CmdCopyBufferToImage); FD(CmdCopyImageToBuffer); FD(CmdPipelineBarrier); FD(ResetCommandBuffer);
    FD(CreateQueryPool); FD(CmdWriteTimestamp); FD(CmdResetQueryPool); FD(GetQueryPoolResults); FD(ResetQueryPool);
    VkQueue q; GetDeviceQueue(dev, qf, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; GetPhysicalDeviceMemoryProperties(pd, &mp);
    VkPhysicalDeviceProperties pp; GetPhysicalDeviceProperties(pd, &pp);

    FgVk vk{inst, pd, dev, qf, q, gipa}; std::string err;
    FgEngine *e = fg_create(vk, sdir, cache, err);
    if (!e) { fprintf(stderr, "fg_create: %s\n", err.c_str()); return 1; }
    int ng = fg_generated_count(e); fprintf(stderr, "engine ready, %d generated frames per real frame\n", ng);

    auto mkhost = [&](VkDeviceSize sz, VkBuffer &b, void *&p) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = sz; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        CK(CreateBuffer(dev, &bi, nullptr, &b)); VkMemoryRequirements mr; GetBufferMemoryRequirements(dev, b, &mr);
        uint32_t t = ~0u; for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((mr.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & 6) == 6) { t = i; break; }
        VkDeviceMemory m; VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = t;
        CK(AllocateMemory(dev, &ai, nullptr, &m)); BindBufferMemory(dev, b, m, 0); MapMemory(dev, m, 0, sz, 0, &p);
    };
    VkBuffer host; void *hp; mkhost(256u << 20, host, hp);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = qf; pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool; CK(CreateCommandPool(dev, &pi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb; CK(AllocateCommandBuffers(dev, &cai, &cb));
    VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qci.queryType = VK_QUERY_TYPE_TIMESTAMP; qci.queryCount = 2;
    VkQueryPool qpool; CK(CreateQueryPool(dev, &qci, nullptr, &qpool));

    // capture resource files: res_rRR_NAME_WxH_fmtF_pitchP.bin
    std::vector<std::string> files; { DIR *d = opendir(cap.c_str()); while (auto *de = readdir(d)) files.push_back(de->d_name); closedir(d); }
    auto findres = [&](int r, const char *name, int &w, int &h, int &pitch) -> std::string {
        char pre[64]; snprintf(pre, sizeof pre, "res_r%02d_%s_", r, name);
        for (auto &f : files) if (f.rfind(pre, 0) == 0) { int fm; sscanf(f.c_str() + strlen(pre), "%dx%d_fmt%d_pitch%d", &w, &h, &fm, &pitch); return cap + "/" + f; }
        return "";
    };
    auto parse_matrix = [&](int r, float *m) {
        char fn[256]; snprintf(fn, sizeof fn, "%s/params_r%02d_e1.txt", cap.c_str(), r); std::ifstream f(fn); std::string l;
        while (std::getline(f, l)) if (l.rfind("m DLSSG.ClipToPrevClip", 0) == 0) { sscanf(l.c_str() + 23, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", m, m+1, m+2, m+3, m+4, m+5, m+6, m+7, m+8, m+9, m+10, m+11, m+12, m+13, m+14, m+15); return true; }
        return false;
    };
    const char *labels[4] = {"backbuffer", "depth", "hudless", "mvec"}; const char *resn[4] = {"Backbuffer", "Depth", "HUDLess", "MVecs"};
    std::vector<double> times;
    for (int r = 0; r < frames; r++) {
        // upload inputs
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CK(ResetCommandBuffer(cb, 0)); CK(BeginCommandBuffer(cb, &bi));
        size_t off = 0;
        for (int i = 0; i < 4; i++) {
            int w, h, pitch; std::string fn = findres(r, resn[i], w, h, pitch);
            if (fn.empty()) { fprintf(stderr, "frame %d: no %s\n", r, resn[i]); return 1; }
            auto data = slurp(fn); uint32_t iw, ih; VkFormat fmt; fg_input_size(e, labels[i], iw, ih, fmt);
            if ((int)iw != w || (int)ih != h) { fprintf(stderr, "input %s size %dx%d != script %ux%u\n", labels[i], w, h, iw, ih); return 1; }
            int bpp = pitch / w; memcpy((uint8_t *)hp + off, data.data(), (size_t)pitch * h);
            VkBufferImageCopy bc{}; bc.bufferOffset = off; bc.bufferRowLength = pitch / bpp; bc.bufferImageHeight = h; bc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bc.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
            CmdCopyBufferToImage(cb, host, fg_input_image(e, labels[i]), VK_IMAGE_LAYOUT_GENERAL, 1, &bc);
            off += ((size_t)pitch * h + 255) & ~255ull;
        }
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, 0, 0, 0);
        float m[16]; bool hm = parse_matrix(r, m);
        ResetQueryPool(dev, qpool, 0, 2);
        CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, qpool, 0);
        int var = r == 0 ? 2 : (r & 1); fg_record(e, cb, var, hm ? m : nullptr);
        CmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, qpool, 1);
        // read back generated frames
        size_t roff = 128u << 20; std::vector<size_t> offs;
        for (int k = 0; k < ng; k++) {
            uint32_t w, h; VkFormat f; fg_input_size(e, "backbuffer", w, h, f);
            VkBufferImageCopy bc{}; bc.bufferOffset = roff; bc.bufferRowLength = w; bc.bufferImageHeight = h; bc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bc.imageExtent = {w, h, 1};
            CmdCopyImageToBuffer(cb, fg_output_image(e, var, k), VK_IMAGE_LAYOUT_GENERAL, host, 1, &bc); offs.push_back(roff); roff += (size_t)w * h * 4;
        }
        CK(EndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        CK(QueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(QueueWaitIdle(q));
        uint64_t ts[2]; GetQueryPoolResults(dev, qpool, 0, 2, 16, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        double ms = (ts[1] - ts[0]) * pp.limits.timestampPeriod / 1e6; times.push_back(ms);
        fprintf(stderr, "frame %d: GPU %.3f ms\n", r, ms);
        uint32_t w, h; VkFormat f; fg_input_size(e, "backbuffer", w, h, f);
        for (int k = 0; k < ng; k++) {
            char fn[256]; snprintf(fn, sizeof fn, "%s/r%02d_g%d.ppm", out.c_str(), r, k + 1); FILE *fp = fopen(fn, "wb"); fprintf(fp, "P6\n%u %u\n255\n", w, h);
            const uint8_t *src = (const uint8_t *)hp + offs[k]; std::vector<uint8_t> rgb((size_t)w * h * 3);
            for (size_t i = 0; i < (size_t)w * h; i++) { rgb[3*i] = src[4*i]; rgb[3*i+1] = src[4*i+1]; rgb[3*i+2] = src[4*i+2]; }
            fwrite(rgb.data(), 1, rgb.size(), fp); fclose(fp);
        }
    }
    std::vector<double> t2(times.begin() + std::min<size_t>(2, times.size() - 1), times.end()); std::sort(t2.begin(), t2.end());
    fprintf(stderr, "median GPU per real frame (%d gen): %.3f ms\n", ng, t2[t2.size() / 2]);
    return 0;
}
