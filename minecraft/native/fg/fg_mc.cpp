// JNI bridge for dev.d4r.minecraft.NativeBridge: D4R Lite frame generation on Minecraft's Vulkan device.
#include "fg_engine.h"
#include "prep_spv.h"
#include <jni.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
void fail(JNIEnv *env, const std::string &m) { env->ThrowNew(env->FindClass("java/lang/RuntimeException"), m.c_str()); }
std::string str(JNIEnv *env, jstring s) { if (!s) return {}; const char *c = env->GetStringUTFChars(s, nullptr); std::string r(c); env->ReleaseStringUTFChars(s, c); return r; }

struct Fn {
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
#define F(n) PFN_vk##n n
    F(CreateShaderModule); F(DestroyShaderModule); F(CreateDescriptorSetLayout); F(CreatePipelineLayout); F(CreateComputePipelines);
    F(CreateDescriptorPool); F(AllocateDescriptorSets); F(UpdateDescriptorSets); F(CmdBindPipeline); F(CmdBindDescriptorSets);
    F(CmdPushConstants); F(CmdDispatch); F(CmdCopyImage); F(CmdBlitImage); F(CmdPipelineBarrier); F(CreateImageView); F(DestroyImageView);
    F(CreateSampler); F(CreateImage); F(DestroyImage); F(FreeMemory); F(DestroyPipeline); F(DestroyPipelineLayout); F(DestroyDescriptorSetLayout); F(DestroyDescriptorPool); F(GetImageMemoryRequirements); F(AllocateMemory); F(BindImageMemory); F(GetPhysicalDeviceMemoryProperties);
#undef F
};

struct Ctx {
    FgVk vk; FgEngine *e = nullptr; Fn f; uint32_t w = 0, h = 0, factor = 0;
    VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; VkDescriptorPool pool; VkDescriptorSet sets[4]; int setIdx = 0;
    VkImageView depthOutView, mvOutView; std::unordered_map<uint64_t, VkImageView> depthViews;
    static const int kRing = 4;
    std::vector<VkDeviceMemory> mems; VkShaderModule shader = 0;
    VkImage real[kRing] = {}; bool realInit[kRing] = {};
    std::vector<VkImage> gen; bool genInit[kRing] = {}; int nGen = 0;
    std::string error;
};
Ctx *C(jlong h) { return reinterpret_cast<Ctx *>(h); }

void fullBarrier(Ctx *c, VkCommandBuffer cb)
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT; mb.dstAccessMask = mb.srcAccessMask;
    c->f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, 0, 0, 0);
}
void copyColor(Ctx *c, VkCommandBuffer cb, VkImage src, VkImage dst)
{
    VkImageCopy r{}; r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.extent = {c->w, c->h, 1};
    c->f.CmdCopyImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
}
}

extern "C" {
// Sets the environment the nvxrt layer needs; must run before Minecraft creates its VkInstance (Fabric preLaunch).
JNIEXPORT jstring JNICALL Java_dev_d4r_minecraft_NativeBridge_setupEnvironment(JNIEnv *env, jclass, jstring layerDir, jstring cacheDir, jboolean exact, jstring logFile)
{
    std::string ld = str(env, layerDir), cd = str(env, cacheDir), lf = str(env, logFile);
    const char *old = getenv("VK_ADD_LAYER_PATH"); std::string lp = ld; if (old && *old && std::string(old).find(ld) == std::string::npos) lp += std::string(":") + old;
    setenv("VK_ADD_LAYER_PATH", lp.c_str(), 1);
    const char *ol = getenv("VK_INSTANCE_LAYERS"); std::string il = "VK_LAYER_NVXRT_nvx";
    if (ol && *ol && std::string(ol).find("VK_LAYER_NVXRT_nvx") == std::string::npos) il += std::string(":") + ol; else if (ol && *ol) il = ol;
    setenv("VK_INSTANCE_LAYERS", il.c_str(), 1);
    setenv("NVXRT_ARCH", "gfx11", 1); setenv("NVXRT_CACHE", cd.c_str(), 1);
    if (!exact) { setenv("NVXRT_TEX", "hw", 1); setenv("NVXRT_ACC", "fast", 1); setenv("NVXRT_SHADOW", "b,c,r", 1); }
    if (!lf.empty()) setenv("NVXRT_LOG", lf.c_str(), 1);
    return env->NewStringUTF("ok");
}

JNIEXPORT jlong JNICALL Java_dev_d4r_minecraft_NativeBridge_create(JNIEnv *env, jclass, jlong instance, jlong phys, jlong device, jlong queue, jint qfam, jint computeFam,
                                                                    jstring scriptDir, jstring cache, jint w, jint h)
{
    auto c = new Ctx; c->w = w; c->h = h;
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD); if (!lib) lib = dlopen("libvulkan.so.1", RTLD_NOW);
    auto gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    c->vk = {(VkInstance)instance, (VkPhysicalDevice)phys, (VkDevice)device, (uint32_t)qfam, (VkQueue)queue, gipa, (uint32_t)computeFam};
    std::string err; c->e = fg_create(c->vk, str(env, scriptDir), str(env, cache), err);
    if (!c->e) { delete c; fail(env, err); return 0; }
    uint32_t iw, ih; VkFormat fm; fg_input_size(c->e, "backbuffer", iw, ih, fm);
    if ((int)iw != w || (int)ih != h) { fail(env, "script resolution " + std::to_string(iw) + "x" + std::to_string(ih) + " != " + std::to_string(w) + "x" + std::to_string(h)); return 0; }
    c->factor = fg_generated_count(c->e) + 1;
    Fn &f = c->f; f.GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)gipa((VkInstance)instance, "vkGetDeviceProcAddr");
#define GD(n) f.n = (PFN_vk##n)f.GetDeviceProcAddr((VkDevice)device, "vk" #n)
    GD(CreateShaderModule); GD(DestroyShaderModule); GD(CreateDescriptorSetLayout); GD(CreatePipelineLayout); GD(CreateComputePipelines); GD(CreateDescriptorPool);
    GD(AllocateDescriptorSets); GD(UpdateDescriptorSets); GD(CmdBindPipeline); GD(CmdBindDescriptorSets); GD(CmdPushConstants); GD(CmdDispatch); GD(CmdCopyImage);
    GD(CmdBlitImage); GD(CmdPipelineBarrier); GD(CreateImageView); GD(DestroyImageView);
    GD(CreateImage); GD(DestroyImage); GD(FreeMemory); GD(DestroyPipeline); GD(DestroyPipelineLayout); GD(DestroyDescriptorSetLayout); GD(DestroyDescriptorPool); GD(GetImageMemoryRequirements); GD(AllocateMemory); GD(BindImageMemory);
#undef GD
    f.GetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)gipa((VkInstance)instance, "vkGetPhysicalDeviceMemoryProperties");
    VkPhysicalDeviceMemoryProperties mp; f.GetPhysicalDeviceMemoryProperties((VkPhysicalDevice)phys, &mp);
    c->nGen = fg_generated_count(c->e);
    auto mkimage = [&](VkImage &out) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ci.imageType = VK_IMAGE_TYPE_2D; ci.format = VK_FORMAT_R8G8B8A8_UNORM; ci.extent = {(uint32_t)w, (uint32_t)h, 1};
        ci.mipLevels = ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT; ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        f.CreateImage((VkDevice)device, &ci, nullptr, &out);
        VkMemoryRequirements mr; f.GetImageMemoryRequirements((VkDevice)device, out, &mr);
        uint32_t t = 0; for (uint32_t k = 0; k < mp.memoryTypeCount; k++) if ((mr.memoryTypeBits >> k & 1) && (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { t = k; break; }
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = t;
        VkDeviceMemory mem; f.AllocateMemory((VkDevice)device, &ai, nullptr, &mem); f.BindImageMemory((VkDevice)device, out, mem, 0);
        c->mems.push_back(mem);
    };
    // snapshots of the real frame and copies of the generated frames, kRing slots deep, so the presenter never reads an image the engine is about to rewrite
    for (int i = 0; i < Ctx::kRing; i++) mkimage(c->real[i]);
    c->gen.resize((size_t)Ctx::kRing * c->nGen);
    for (auto &im : c->gen) mkimage(im);
    VkDevice dev = (VkDevice)device;
    VkDescriptorSetLayoutBinding b[3] = {{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}, {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 3; dli.pBindings = b;
    f.CreateDescriptorSetLayout(dev, &dli, nullptr, &c->dsl);
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 80};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &c->dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pr;
    f.CreatePipelineLayout(dev, &pli, nullptr, &c->pl);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = sizeof(kPrepSpv); smi.pCode = kPrepSpv; VkShaderModule sm; f.CreateShaderModule(dev, &smi, nullptr, &sm); c->shader = sm;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", 0}; cpi.layout = c->pl;
    f.CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &c->pipe);
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 4; dpi.poolSizeCount = 2; dpi.pPoolSizes = ps;
    f.CreateDescriptorPool(dev, &dpi, nullptr, &c->pool);
    VkDescriptorSetLayout ls[4] = {c->dsl, c->dsl, c->dsl, c->dsl}; VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = c->pool; dai.descriptorSetCount = 4; dai.pSetLayouts = ls;
    f.AllocateDescriptorSets(dev, &dai, c->sets);
    auto mkview = [&](VkImage img, VkFormat fmt, VkImageAspectFlags asp) { VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt; vi.subresourceRange = {asp, 0, 1, 0, 1}; VkImageView v; f.CreateImageView(dev, &vi, nullptr, &v); return v; };
    c->depthOutView = mkview(fg_input_image(c->e, "depth"), VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    c->mvOutView = mkview(fg_input_image(c->e, "mvec"), VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    return reinterpret_cast<jlong>(c);
}

JNIEXPORT jint JNICALL Java_dev_d4r_minecraft_NativeBridge_factor(JNIEnv *, jclass, jlong h) { return (jint)C(h)->factor; }

// At the HUD-less point: copy the world colour, derive depth + motion vectors. matrix: previous clip from current clip, column-major.
JNIEXPORT void JNICALL Java_dev_d4r_minecraft_NativeBridge_captureWorld(JNIEnv *env, jclass, jlong h, jlong cbh, jlong color, jlong depth, jfloatArray matrix, jfloat sign)
{
    Ctx *c = C(h); VkCommandBuffer cb = (VkCommandBuffer)cbh; Fn &f = c->f;
    fullBarrier(c, cb);
    copyColor(c, cb, (VkImage)color, fg_input_image(c->e, "hudless"));
    auto it = c->depthViews.find((uint64_t)depth); VkImageView dv;
    if (it == c->depthViews.end()) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = (VkImage)depth; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_D32_SFLOAT; vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        f.CreateImageView(c->vk.device, &vi, nullptr, &dv); c->depthViews[(uint64_t)depth] = dv;
    } else dv = it->second;
    VkDescriptorSet set = c->sets[c->setIdx++ & 3];
    VkDescriptorImageInfo ii[3] = {{VK_NULL_HANDLE, dv, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, c->depthOutView, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, c->mvOutView, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[3]{}; VkDescriptorType ty[3] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    for (int i = 0; i < 3; i++) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = ty[i]; w[i].pImageInfo = &ii[i]; }
    f.UpdateDescriptorSets(c->vk.device, 3, w, 0, nullptr);
    struct { float m[16]; int32_t size[2]; float sign, pad; } pc{};
    env->GetFloatArrayRegion(matrix, 0, 16, pc.m); pc.size[0] = c->w; pc.size[1] = c->h; pc.sign = sign;
    fullBarrier(c, cb);
    f.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipe);
    f.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl, 0, 1, &set, 0, nullptr);
    f.CmdPushConstants(cb, c->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    f.CmdDispatch(cb, (c->w + 7) / 8, (c->h + 7) / 8, 1);
    fullBarrier(c, cb);
}

// End of frame, graphics queue: copy the final colour into the engine's backbuffer input and into the real-frame snapshot slot.
JNIEXPORT void JNICALL Java_dev_d4r_minecraft_NativeBridge_generateGfx(JNIEnv *, jclass, jlong h, jlong cbh, jlong color, jint slot)
{
    Ctx *c = C(h); VkCommandBuffer cb = (VkCommandBuffer)cbh; slot &= Ctx::kRing - 1;
    fullBarrier(c, cb);
    copyColor(c, cb, (VkImage)color, fg_input_image(c->e, "backbuffer"));
    if (!c->realInit[slot]) {
        c->realInit[slot] = true;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = c->real[slot]; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        c->f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 0, 0, 1, &b);
    }
    copyColor(c, cb, (VkImage)color, c->real[slot]);
    fullBarrier(c, cb);
}

// Compute queue: the generation script (engine-owned resources only).
JNIEXPORT void JNICALL Java_dev_d4r_minecraft_NativeBridge_generateCompute(JNIEnv *env, jclass, jlong h, jlong cbh, jint variant, jint ringSlot, jfloatArray matrix)
{
    Ctx *c = C(h); VkCommandBuffer cb = (VkCommandBuffer)cbh; float m[16]; ringSlot &= Ctx::kRing - 1;
    env->GetFloatArrayRegion(matrix, 0, 16, m);
    fg_record(c->e, cb, variant, m);
    fullBarrier(c, cb);
    if (!c->genInit[ringSlot]) {
        c->genInit[ringSlot] = true;
        std::vector<VkImageMemoryBarrier> bs;
        for (int k = 0; k < c->nGen; k++) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = c->gen[(size_t)ringSlot * c->nGen + k]; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; bs.push_back(b);
        }
        c->f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 0, 0, (uint32_t)bs.size(), bs.data());
    }
    for (int k = 0; k < c->nGen; k++) copyColor(c, cb, fg_output_image(c->e, variant, k), c->gen[(size_t)ringSlot * c->nGen + k]);
    fullBarrier(c, cb);
}

static void blitToSwap(Ctx *c, VkCommandBuffer cb, VkImage src, VkImage swap, int dw, int dh)
{
    Fn &f = c->f;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = swap; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, 0, 1, &b);
    int cw = (int)c->w < dw ? (int)c->w : dw, ch = (int)c->h < dh ? (int)c->h : dh;
    VkImageBlit r{}; r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.srcOffsets[1] = {cw, ch, 1};
    r.dstOffsets[0] = {0, ch, 0}; r.dstOffsets[1] = {cw, 0, 1};      // Minecraft's framebuffer rows are GL-oriented: flip like VulkanGpuSurface does
    f.CmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, swap, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r, VK_FILTER_NEAREST);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = (VkImageLayout)1000001002; b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = 0;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 0, 0, 1, &b);
}

// Generated frame k (0-based) of the given variant into a swapchain image.
JNIEXPORT void JNICALL Java_dev_d4r_minecraft_NativeBridge_blitGenerated(JNIEnv *, jclass, jlong h, jlong cb, jint ringSlot, jint k, jlong swapImage, jint dw, jint dh)
{
    Ctx *c = C(h); blitToSwap(c, (VkCommandBuffer)cb, c->gen[(size_t)(ringSlot & (Ctx::kRing - 1)) * c->nGen + k], (VkImage)swapImage, dw, dh);
}
// The snapshot of the real frame taken by generate() into a swapchain image.
JNIEXPORT void JNICALL Java_dev_d4r_minecraft_NativeBridge_blitReal(JNIEnv *, jclass, jlong h, jlong cb, jint slot, jlong swapImage, jint dw, jint dh)
{
    Ctx *c = C(h); blitToSwap(c, (VkCommandBuffer)cb, c->real[slot & (Ctx::kRing - 1)], (VkImage)swapImage, dw, dh);
}

// Only after every recorded frame has completed on the GPU. Engine resources die with the device.
JNIEXPORT void JNICALL Java_dev_d4r_minecraft_NativeBridge_destroy(JNIEnv *, jclass, jlong h)
{
    Ctx *c = C(h); Fn &f = c->f; VkDevice d = c->vk.device;
    fg_destroy(c->e);
    for (auto &kv : c->depthViews) f.DestroyImageView(d, kv.second, nullptr);
    f.DestroyImageView(d, c->depthOutView, nullptr); f.DestroyImageView(d, c->mvOutView, nullptr);
    f.DestroyPipeline(d, c->pipe, nullptr); f.DestroyPipelineLayout(d, c->pl, nullptr); f.DestroyDescriptorSetLayout(d, c->dsl, nullptr); f.DestroyDescriptorPool(d, c->pool, nullptr);
    if (c->shader) f.DestroyShaderModule(d, c->shader, nullptr);
    for (auto im : c->real) f.DestroyImage(d, im, nullptr);
    for (auto im : c->gen) f.DestroyImage(d, im, nullptr);
    for (auto mem : c->mems) f.FreeMemory(d, mem, nullptr);
    delete c;
}
}
