// nvxrt: a Vulkan layer that implements VK_NVX_binary_import and VK_NVX_image_view_handle on top of a
// driver that lacks them (RADV). CUDA modules handed to vkCreateCuModuleNVX are identified by the
// SHA-256 of their normalised PTX; the kernels run as SPIR-V compute shaders that the host-side
// translator (translator/) produced ahead of time into the kernel cache. Texture/surface handles are
// indices into a bindless descriptor set owned by the layer; CUDA pointers are buffer device addresses.
//
// Env:
//   NVXRT_CACHE   kernel cache directory (default ~/.cache/nvxrt)
//   NVXRT_LOG     log file (default stderr); NVXRT_TRACE=1 logs every launch with decoded parameters
//   NVXRT_EXEC=0  record/trace only, do not dispatch
//   NVXRT_TIMING=1 GPU timestamps around every launch, summarised per kernel in the log
//   NVXRT_TEX=hw  fast filtering kernels (ENTRY.hw.spv);  NVXRT_ACC=fast  fp32-accumulation kernels (ENTRY.fa.spv)
//   NVXRT_SHADOW=b,c  RGBA8 shadow images for DLSS-G's BCF candidates (b) and hudless/UI copies (c), see "shadows";
//                 NVXRT_SHADOW_LOG=1 logs shadow events, =2 also every launch parameter that names a full-res
//                 RGBA16F image (role discovery)
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <chrono>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "util.h"

#define EXPORT extern "C" __attribute__((visibility("default")))

namespace nvxrt {

// ---------------------------------------------------------------------------------------------- logging
static FILE *g_log = nullptr;
static std::mutex g_log_mutex;
static bool g_trace = false, g_exec = true, g_timing = false, g_serialize = true;
// Kernels whose SPIR-V uses nothing that depends on the subgroup (KernelMeta::lane_free) may run as wave64, which on
// RDNA3 issues each VALU instruction for 64 lanes (both ALUs) and halves the scalar work per thread. NVXRT_WAVE64: unset
// = the kernels the cache marks "wave 64" (measured faster), 1 = every eligible kernel, 0 = none
static int g_wave64 = 2;
static std::string g_dump_dir; // NVXRT_DUMP=<dir>: copy every surface after each launch (harness debugging)
static std::set<uint64_t> g_cap_launches; // NVXRT_CAPTURE=<n,n,...>: params + buffers before/after these launches
static std::string g_cache_dir;
static int g_shadow = 0;       // NVXRT_SHADOW: bit 1 = b (BCF candidates), bit 2 = c (hudless/UI copies)
static int g_shadow_log = 0;   // NVXRT_SHADOW_LOG
static bool g_legacy_slots = false;     // NVXRT_SLOTS=legacy: old image slot numbering (see "Image handles")
static uint32_t g_slot_base = 32768;   // unified numbering: [32768, 65536); small integers never look like handles

static void init_globals()
{
    static std::once_flag once;
    std::call_once(once, [] {
        const char *lp = getenv("NVXRT_LOG");
        g_log = lp && *lp ? fopen(lp, "a") : nullptr;
        if (!g_log)
            g_log = stderr;
        g_trace = getenv("NVXRT_TRACE") && atoi(getenv("NVXRT_TRACE")) != 0;
        g_exec = !(getenv("NVXRT_EXEC") && atoi(getenv("NVXRT_EXEC")) == 0);
        g_timing = getenv("NVXRT_TIMING") && atoi(getenv("NVXRT_TIMING")) != 0;
        // CUDA orders kernels launched into one stream; D3D12/Vulkan dispatches are not ordered without a barrier.
        g_serialize = !(getenv("NVXRT_SERIALIZE") && atoi(getenv("NVXRT_SERIALIZE")) == 0);
        if (getenv("NVXRT_WAVE64") && *getenv("NVXRT_WAVE64"))
            g_wave64 = atoi(getenv("NVXRT_WAVE64")) != 0;
        if (getenv("NVXRT_DUMP"))
            g_dump_dir = getenv("NVXRT_DUMP");
        if (const char *cl = getenv("NVXRT_CAPTURE")) {
            std::string s(cl);
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                if (q > p) g_cap_launches.insert(std::stoull(s.substr(p, q - p)));
                p = q + 1;
            }
        }
        if (const char *sh = getenv("NVXRT_SHADOW")) {
            if (strchr(sh, 'b') || strchr(sh, 'B')) g_shadow |= 1;
            if (strchr(sh, 'c') || strchr(sh, 'C')) g_shadow |= 2;
            // r: UI-copy shadows are read without the fp16 fix-up (k/255, < 1 fp16 step off; not exact), which also
            // enables the roles of DLLs whose readers have no fix-up compiled in
            if (strchr(sh, 'r') || strchr(sh, 'R')) g_shadow |= 4;
        }
        g_shadow_log = getenv("NVXRT_SHADOW_LOG") ? atoi(getenv("NVXRT_SHADOW_LOG")) : 0;
        g_legacy_slots = getenv("NVXRT_SLOTS") && !strcmp(getenv("NVXRT_SLOTS"), "legacy");
        if (g_legacy_slots)
            g_slot_base = 1;
        // NVXRT_SLOT_BASE=n: first slot number (tests: the output must not depend on it)
        if (getenv("NVXRT_SLOT_BASE") && atoi(getenv("NVXRT_SLOT_BASE")) > 0)
            g_slot_base = (uint32_t)atoi(getenv("NVXRT_SLOT_BASE"));
        const char *c = getenv("NVXRT_CACHE");
        if (c && *c)
            g_cache_dir = c;
        else {
            const char *home = getenv("HOME");
            g_cache_dir = std::string(home ? home : "/tmp") + "/.cache/nvxrt";
        }
    });
}

static void logf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf(const char *fmt, ...)
{
    std::lock_guard<std::mutex> lk(g_log_mutex);
    va_list ap;
    va_start(ap, fmt);
    fprintf(g_log, "nvxrt: ");
    vfprintf(g_log, fmt, ap);
    fputc('\n', g_log);
    fflush(g_log);
    va_end(ap);
}

static void mkdirs(const std::string &path)
{
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/' && cur.size() > 1)
            mkdir(cur.c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

static bool read_file(const std::string &path, std::vector<uint8_t> &out)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
        return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? n : 0);
    bool ok = n <= 0 || fread(out.data(), 1, n, f) == (size_t)n;
    fclose(f);
    return ok;
}

static void write_file_if_absent(const std::string &path, const void *data, size_t n)
{
    struct stat st;
    if (stat(path.c_str(), &st) == 0)
        return;
    std::string tmp = path + ".tmp" + std::to_string(getpid());
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f)
        return;
    fwrite(data, 1, n, f);
    fclose(f);
    rename(tmp.c_str(), path.c_str());
}

// ---------------------------------------------------------------------------------------------- dispatch
static inline void *dkey(const void *h) { return *(void **)h; }

struct InstanceDispatch {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties;
};

struct Instance {
    VkInstance handle;
    InstanceDispatch d;
};

static std::mutex g_map_mutex;
static std::unordered_map<void *, std::shared_ptr<Instance>> g_instances; // by dispatch key
static std::unordered_map<VkPhysicalDevice, std::shared_ptr<Instance>> g_pd_instance;

struct Device;
static std::unordered_map<void *, std::shared_ptr<Device>> g_devices; // by dispatch key

static std::shared_ptr<Instance> instance_of(const void *h)
{
    std::lock_guard<std::mutex> lk(g_map_mutex);
    auto it = g_instances.find(dkey(h));
    return it == g_instances.end() ? nullptr : it->second;
}

static std::shared_ptr<Instance> instance_of_pd(VkPhysicalDevice pd)
{
    std::lock_guard<std::mutex> lk(g_map_mutex);
    auto it = g_pd_instance.find(pd);
    if (it != g_pd_instance.end())
        return it->second;
    auto it2 = g_instances.find(dkey(pd));
    return it2 == g_instances.end() ? nullptr : it2->second;
}

#define DEVICE_FUNCS(X)                                                                                            \
    X(CmdCopyImageToBuffer) X(CmdCopyBuffer) X(WaitForFences) X(ResetQueryPool) X(DeviceWaitIdle) X(QueueWaitIdle)                                      \
    X(GetDeviceProcAddr) X(DestroyDevice) X(CreateImageView) X(DestroyImageView) X(CreateSampler) X(DestroySampler) \
    X(CreateBuffer) X(DestroyBuffer) X(GetBufferDeviceAddress) X(CreateDescriptorSetLayout)                          \
    X(DestroyDescriptorSetLayout) X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateDescriptorPool)          \
    X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CreateShaderModule)                \
    X(DestroyShaderModule) X(CreateComputePipelines) X(DestroyPipeline) X(CmdBindPipeline) X(CmdBindDescriptorSets) \
    X(CmdPushConstants) X(CmdDispatch) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory)               \
    X(BindBufferMemory) X(MapMemory) X(CreatePipelineCache) X(DestroyPipelineCache) X(GetPipelineCacheData)         \
    X(CreateQueryPool) X(DestroyQueryPool) X(CmdWriteTimestamp) X(CmdResetQueryPool) X(GetQueryPoolResults)          \
    X(CmdPipelineBarrier) X(QueueSubmit) X(QueueSubmit2) X(GetDeviceQueue) X(CreateImage) X(DestroyImage)          \
    X(GetImageMemoryRequirements) X(BindImageMemory) X(CmdCopyImage) X(CmdCopyImage2) X(CmdBlitImage)             \
    X(CmdBlitImage2) X(CmdResolveImage) X(CmdResolveImage2) X(CmdClearColorImage) X(CmdCopyImageToBuffer2)        \
    X(CmdCopyBufferToImage) X(CmdCopyBufferToImage2) X(CmdCopyBuffer2) X(CmdFillBuffer) X(CmdUpdateBuffer)          \
    X(CmdClearDepthStencilImage) X(CmdCopyQueryPoolResults) X(CmdBeginRenderPass) X(CmdBeginRenderPass2)          \
    X(CmdBeginRendering) X(CmdExecuteCommands) X(CmdBuildAccelerationStructuresKHR)                                \
    X(CmdBuildAccelerationStructuresIndirectKHR) X(CmdCopyAccelerationStructureKHR)                                \
    X(CmdCopyAccelerationStructureToMemoryKHR) X(CmdCopyMemoryToAccelerationStructureKHR)                          \
    X(AllocateCommandBuffers) X(FreeCommandBuffers) X(BeginCommandBuffer) X(ResetCommandBuffer) X(ResetCommandPool) \
    X(DestroyCommandPool) X(EndCommandBuffer)

struct DeviceDispatch {
#define X(n) PFN_vk##n n = nullptr;
    DEVICE_FUNCS(X)
#undef X
    PFN_vkGetBufferDeviceAddressKHR GetBufferDeviceAddressKHR = nullptr;
    PFN_vkGetBufferDeviceAddressEXT GetBufferDeviceAddressEXT = nullptr;
};

// ---------------------------------------------------------------------------------------------- objects
struct ViewInfo {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    VkImageUsageFlags usage = 0;          // the view's own usage (VkImageViewUsageCreateInfo), 0 = the image's
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
};

struct ImageInfo {
    VkFormat format;
    VkExtent3D extent;
    VkImageUsageFlags usage;
    VkImageType type = VK_IMAGE_TYPE_2D;
    uint32_t mips = 1, layers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};

struct SamplerInfo {
    VkFilter mag, min;
    VkSamplerAddressMode u, v;
    VkBorderColor border;
    VkBool32 unnormalized;
};

struct KernelMeta {
    bool ok = false;
    std::string name;
    std::string hash;
    std::vector<uint32_t> spirv;
    uint32_t param_size = 0;
    std::vector<std::pair<uint32_t, uint32_t>> params; // (offset, size) per declared .param
    std::string features;
    // T1 kernels may use their own workgroup shape and grid (meta lines "t1_block X Y Z" and
    // "t1_grid NX DX NY DY NZ DZ": dispatch grid = ceil(grid * N / D) per dimension)
    bool t1_dispatch = false;
    uint32_t t1_block[3] = {0, 0, 0};
    uint32_t t1_num[3] = {1, 1, 1}, t1_den[3] = {1, 1, 1};
    uint32_t t1_fixed[3] = {0, 0, 0};   // "t1_fixed X Y Z": persistent kernels, grid independent of the launch
    // "huse OFF tex|surf|other" (from the translated source): how the kernel uses the 8-byte parameter at OFF
    std::unordered_map<uint32_t, uint8_t> huse;
    bool huse_info = false, huse_dynamic = false;
    bool shadow8 = false;   // "shadow8 1": built with the class-3 read fix-up and checked stores (UI-copy shadows)
    bool big_params = false; // "big_params 1": parameter block > 248 bytes, read through the address in pc words 60..61
    bool lane_free = false;  // no subgroup operation, cooperative matrix or subgroup built-in: wave size does not matter
    bool wave64 = false;     // "wave 64" (t1/wave64-*.txt): measured faster as wave64 on this GPU generation
};
enum { kUseTex = 1, kUseSurf = 2, kUseOther = 4 };

struct CuModule {
    std::string hash;     // sha256 of normalised PTX (empty if SASS only)
    std::string ptx;      // chosen PTX text
    uint32_t sm = 0;
    std::vector<std::string> entries;
    std::vector<std::string> sass_names; // names in SASS-only modules
    bool sass_only = false;
};

struct PipeKey {
    uint32_t bx, by, bz, smem;
    bool operator<(const PipeKey &o) const
    {
        return std::tie(bx, by, bz, smem) < std::tie(o.bx, o.by, o.bz, o.smem);
    }
};

struct CuFunction {
    CuModule *module;
    std::string name;
    std::string gkey;   // "hash:entry" (launch groups)
    KernelMeta meta;
    std::map<PipeKey, VkPipeline> pipelines;
    VkShaderModule shader = VK_NULL_HANDLE;
    uint64_t launches = 0;
    bool warned_skip = false;
    bool hw_tex = false;   // loaded the fast-filtering variant (NVXRT_TEX=hw)
    bool fast_acc = false; // loaded the fp32-accumulation variant (NVXRT_ACC=fast)
    std::vector<const struct ShadowRole *> shadow_roles; // NVXRT_SHADOW parameter roles of this kernel (by module + entry)
    // timing
    double gpu_ms_total = 0;
    uint64_t gpu_samples = 0;
};

struct BufferRange {
    VkBuffer buffer;
    VkDeviceSize size;
};

static const uint32_t kMaxImages = 1u << 16; // per binding
// Image handles are slot | sampler slot << 20. One slot number per image view, for both bindings (NVIDIA's
// model: one descriptor-heap index per view, usable by texture and surface instructions). The binding the handle was
// requested for gets the view's descriptor; the other binding gets it too when the image allows that usage (a
// layer-owned companion view when vkd3d restricted the view's usage), because DLSS-G uses some handles with the other
// kind of instruction (Kernel_Scatter3d writes through a combined image+sampler handle). NVXRT_SLOTS=legacy: the old
// numbering (a counter per binding, no second descriptor), for comparisons and rollback.
// (g_legacy_slots / g_slot_base: see the globals at the top)
static const uint32_t kMaxSamplers = 4096;
static const uint32_t kPushBytes = 256;
static const uint32_t kQueries = 32768;
static const uint32_t kInfoBytes = kMaxSamplers * 16 + kMaxImages * 4 * 2;
// big parameter ring (kernels with blocks above 248 bytes); a slot is free again once the command buffer that
// recorded it can no longer execute (see Device::cmd_slots)
static const uint32_t kBpSlot = 4096;
static const uint32_t kBpSlots = 1024;

// raw layout of a format for suld.b (rt.glsl nvx_suld_b): kind (1 unorm, 2 snorm, 3 float, 4 uint, 5 sint) | bytes per
// component << 4 | components << 8 | BGRA order << 12; 0 = unknown
static uint32_t format_raw(VkFormat f)
{
    auto r = [](uint32_t kind, uint32_t bytes, uint32_t n, uint32_t bgra = 0) { return kind | bytes << 4 | n << 8 | bgra << 12; };
    switch (f) {
    case VK_FORMAT_R8_UNORM: return r(1, 1, 1);
    case VK_FORMAT_R8G8_UNORM: return r(1, 1, 2);
    case VK_FORMAT_R8G8B8A8_UNORM: return r(1, 1, 4);
    case VK_FORMAT_B8G8R8A8_UNORM: return r(1, 1, 4, 1);
    case VK_FORMAT_R8_SNORM: return r(2, 1, 1);
    case VK_FORMAT_R8G8_SNORM: return r(2, 1, 2);
    case VK_FORMAT_R8G8B8A8_SNORM: return r(2, 1, 4);
    case VK_FORMAT_R8_UINT: return r(4, 1, 1);
    case VK_FORMAT_R8G8_UINT: return r(4, 1, 2);
    case VK_FORMAT_R8G8B8A8_UINT: return r(4, 1, 4);
    case VK_FORMAT_R8_SINT: return r(5, 1, 1);
    case VK_FORMAT_R8G8_SINT: return r(5, 1, 2);
    case VK_FORMAT_R8G8B8A8_SINT: return r(5, 1, 4);
    case VK_FORMAT_R16_UNORM: return r(1, 2, 1);
    case VK_FORMAT_R16G16_UNORM: return r(1, 2, 2);
    case VK_FORMAT_R16G16B16A16_UNORM: return r(1, 2, 4);
    case VK_FORMAT_R16_SNORM: return r(2, 2, 1);
    case VK_FORMAT_R16G16_SNORM: return r(2, 2, 2);
    case VK_FORMAT_R16G16B16A16_SNORM: return r(2, 2, 4);
    case VK_FORMAT_R16_SFLOAT: return r(3, 2, 1);
    case VK_FORMAT_R16G16_SFLOAT: return r(3, 2, 2);
    case VK_FORMAT_R16G16B16A16_SFLOAT: return r(3, 2, 4);
    case VK_FORMAT_R16_UINT: return r(4, 2, 1);
    case VK_FORMAT_R16G16_UINT: return r(4, 2, 2);
    case VK_FORMAT_R16G16B16A16_UINT: return r(4, 2, 4);
    case VK_FORMAT_R16_SINT: return r(5, 2, 1);
    case VK_FORMAT_R16G16_SINT: return r(5, 2, 2);
    case VK_FORMAT_R16G16B16A16_SINT: return r(5, 2, 4);
    case VK_FORMAT_R32_SFLOAT: return r(3, 4, 1);
    case VK_FORMAT_R32G32_SFLOAT: return r(3, 4, 2);
    case VK_FORMAT_R32G32B32A32_SFLOAT: return r(3, 4, 4);
    case VK_FORMAT_R32_UINT: return r(4, 4, 1);
    case VK_FORMAT_R32G32_UINT: return r(4, 4, 2);
    case VK_FORMAT_R32G32B32A32_UINT: return r(4, 4, 4);
    case VK_FORMAT_R32_SINT: return r(5, 4, 1);
    case VK_FORMAT_R32G32_SINT: return r(5, 4, 2);
    case VK_FORMAT_R32G32B32A32_SINT: return r(5, 4, 4);
    default: return 0;
    }
}

static uint32_t format_class(VkFormat f)
{
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_B8G8R8A8_UINT:
    case VK_FORMAT_A8B8G8R8_UINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R64_UINT:
    case VK_FORMAT_R64G64_UINT: case VK_FORMAT_S8_UINT:
        return 1;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_SINT:
    case VK_FORMAT_A8B8G8R8_SINT_PACK32: case VK_FORMAT_A2R10G10B10_SINT_PACK32: case VK_FORMAT_A2B10G10R10_SINT_PACK32:
    case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R64_SINT:
    case VK_FORMAT_R64G64_SINT:
        return 2;
    default:
        return 0;
    }
}

// Launch groups (T1): a fixed sequence of kernel launches recorded back to back into one command buffer is replaced
// by ONE dispatch of a fused kernel (groups/NAME.spv). The dispatch is recorded at the first member's launch; every
// later member only writes its parameter block into the dispatch's region of a host-visible ring before submission.
// The fused kernel executes the first `count` members (count = members seen), so a sequence that breaks off early
// still runs exactly the launches that were absorbed, in their original order.
// region layout: [0] count, [128] grid-barrier counter, [256 + 256k] member k params, [kGrpGridOff + 16k] k's grid,
// [kGrpInfoOff + 16k] k's block (x, y, z) and which alternative of member k it was (a member may list several
// "hash:entry" alternatives that run the same code, e.g. one 1x1 conv layer compiled twice by NVIDIA)
static const uint32_t kGrpMaxMembers = 48;
static const uint32_t kGrpGridOff = 256 + kGrpMaxMembers * 256;
static const uint32_t kGrpInfoOff = kGrpGridOff + kGrpMaxMembers * 16;
static const uint32_t kGrpRegion = 16384;
static const uint32_t kGrpProfOff = 15360;  // 128 x u64 profile stamps
static const uint32_t kGrpRegions = 512;
struct Group {
    std::string name;
    std::vector<std::string> members; // "hash:entry", or alternatives "hash:entry|hash:entry"
    uint32_t block[3] = {0, 0, 0};
    uint32_t grid[3] = {1, 1, 1};
    CuFunction *fn = nullptr;         // pseudo function: shader, pipelines, timing
};
struct PendingGroup {
    Group *g = nullptr;
    uint32_t next = 0;
    uint8_t *region = nullptr;
    uint64_t last_launch = 0;   // dev.launch_count of the last member taken: the next member must be the next launch
};

// ---------------------------------------------------------------------------------------------- shadows (NVXRT_SHADOW b/c)
// DLSS-G keeps some full-resolution intermediates as RGBA16F images although their content needs less: the two
// BlendCandidatesFused candidates (NVXRT_SHADOW=b: stored as RGBA8, max error 1/255, not exact) and the per-frame
// hudless/UI copy that Downsample writes as fp16_rz(8-bit value) (NVXRT_SHADOW=c: RGBA8 + an exact read fix-up). For such an
// original image the layer keeps an RGBA8 "shadow" (same size, own memory, GENERAL layout) and rewrites handles in the
// launch parameters (the push-constant copy; DLSS-G's own memory is never touched):
//   - a kernel parameter with a shadow-write role (ShadowRole, keyed by module hash + entry + parameter block size +
//     offset, i.e. one DLL's exact layout) gets the shadow's storage handle: the image's content is now in the shadow;
//   - any sampled handle of an image whose content is in the shadow gets the shadow's sampled handle (sampler bits kept):
//     every reader follows the content, whatever kernel it is;
//   - any other storage access, or a vkd3d copy/clear/blit naming the original, moves the content back to the original;
//     if that access could be a read of content that is in the shadow, the image is marked bad (never shadowed again)
//     and the event is logged.
// Reads are taken from the state before the launch, writes applied after. Recording order = execution order (one
// DLSS-G command stream). A DLL or parameter layout without roles here never creates a shadow: unmodified path.
// UI-copy shadows have class 3 for both bindings: sampled reads re-apply fp16 round-toward-zero (exact); stores check
// that every value is an 8-bit quantity and raise a GPU flag otherwise, after which the layer stops C (kept exact).
enum { kShadowB = 1, kShadowC = 2 };
enum { kRoleShadowWrite = 1, kRoleOrigWrite = 2 };
struct ShadowRole {
    const char *gkey;   // module hash : entry
    uint32_t psize, off;
    int kind, what;
    bool raw = false;   // only with NVXRT_SHADOW=r: this DLL's readers of the image have no fix-up (not exact)
};
static const ShadowRole kShadowRoles[] = {
    // nvngx_dlssg.dll 310.9.1 (The Witcher 3): BlendCandidatesFused writes its two full-res candidates at 48/56
    // (read only by OutputPull at 16/24); DownsampleRGBAndEstimateUIFused writes the hudless/UI copy at 120 (read by
    // the next frame's Downsample at 80 and by BCF at 0/8) and another full-res image of the same pool at 160.
    {"d80209427f2fccdbeec472f11a91583c2eeadde15e93741ac3579582c73e54f1:Kernel_BlendCandidatesFused", 240, 48, kShadowB, kRoleShadowWrite},
    {"d80209427f2fccdbeec472f11a91583c2eeadde15e93741ac3579582c73e54f1:Kernel_BlendCandidatesFused", 240, 56, kShadowB, kRoleShadowWrite},
    {"f68d675507dfd4efd74ad484a52eb0f84241190d5dc2598ff65fc8b5f90558b2:Kernel_DownsampleRGBAndEstimateUIFused", 168, 120, kShadowC, kRoleShadowWrite},
    {"f68d675507dfd4efd74ad484a52eb0f84241190d5dc2598ff65fc8b5f90558b2:Kernel_DownsampleRGBAndEstimateUIFused", 168, 160, kShadowC, kRoleOrigWrite},
    // nvngx_dlssg.dll 310.9.16 (Streamline 2.14.16, FG Lite and full): Downsample writes the hudless/UI copy at 120 (read by
    // BlendCandidatesFGLite at 0/8 and by the next frame's Downsample at 80) and another full-res 8-bit-valued image at 128
    // (read by BlendCandidatesFGLite at 176/208). Their readers are compiled without the fix-up, so these apply with
    // NVXRT_SHADOW=r only: k/255 instead of fp16_rz(k/255). The 128 shadow: Downsample 0.289 -> 0.276 ms (gfx11 1440p),
    // presented frames byte-identical to the 120-only shadow on the harness and the Ghost of Tsushima capture.
    {"7947c805931c02ebbd468b4a8f27a7fce1b2a5cd55c1cfa5ffe067581aa14049:Kernel_DownsampleRGBAndEstimateUIFused", 184, 120, kShadowC, kRoleShadowWrite, true},
    {"7947c805931c02ebbd468b4a8f27a7fce1b2a5cd55c1cfa5ffe067581aa14049:Kernel_DownsampleRGBAndEstimateUIFused", 184, 128, kShadowC, kRoleShadowWrite, true},
};
struct Shadow {
    VkImage img = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t tslot = 0, sslot = 0;   // sampled / storage slot (equal with unified numbering)
    int kind = 0;
    bool in_shadow = false;   // the original's current content is in the shadow
    bool bad = false;         // an access that may read shadowed content through the original: no more shadow writes
    uint64_t writes = 0, reads = 0;
};

struct Device {
    VkDevice handle;
    VkPhysicalDevice pd;
    std::shared_ptr<Instance> inst;
    DeviceDispatch d;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memprops{};

    std::mutex mutex;
    std::unordered_map<VkImageView, ViewInfo> views;
    std::unordered_map<VkImage, ImageInfo> images;
    std::unordered_map<VkSampler, SamplerInfo> samplers;
    std::map<VkDeviceAddress, BufferRange> buffer_ranges; // by address
    std::unordered_map<VkBuffer, VkDeviceSize> buffer_sizes;

    // bindless
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkBuffer smp_buf = VK_NULL_HANDLE;
    VkDeviceMemory smp_mem = VK_NULL_HANDLE;
    uint32_t *smp_map = nullptr;
    bool bindless_ok = false;
    // slot allocation: key = (view, binding) -> slot whose descriptor of that binding holds the view
    std::map<std::pair<VkImageView, int>, uint32_t> view_slot;
    std::unordered_map<VkImageView, uint32_t> view_num;      // unified numbering: view -> slot
    std::unordered_map<uint32_t, VkImageView> slot_view;     // unified numbering: slot -> view
    std::unordered_map<VkImageView, VkImageView> companion;  // view -> layer-owned view with the other usage
    std::unordered_map<int, VkFormatFeatureFlags> fmt_features;
    std::unordered_map<VkSampler, uint32_t> sampler_slot;
    uint32_t next_slot[2] = {g_slot_base, g_slot_base};
    uint32_t next_shadow_slot[2] = {61440, 61440};   // legacy numbering: shadow views (DLSS-G's numbers stay as before)
    bool alloc_shadow = false;
    std::deque<uint32_t> free_slots[2];
    uint32_t next_sampler = 1;
    std::deque<uint32_t> free_samplers;
    // binding 4: per sampler slot an unnormalized-coordinate NEAREST companion with the same address mode / border
    // (exact texel fetches with hardware clamping: tex.base / filtered taps sample texel centres)
    std::map<uint32_t, VkSampler> nearest_samplers;
    std::map<uint32_t, std::string> handle_desc; // for trace decoding

    VkPipelineCache pcache = VK_NULL_HANDLE;
    std::vector<std::unique_ptr<CuModule>> modules;
    std::vector<std::unique_ptr<CuFunction>> functions;

    // dumps
    VkBuffer dump_buf = VK_NULL_HANDLE;
    VkDeviceMemory dump_mem = VK_NULL_HANDLE;
    uint8_t *dump_map = nullptr;
    VkDeviceSize dump_size = 0, dump_used = 0;
    struct DumpRec { uint64_t launch; std::string kernel; uint32_t poff; VkFormat fmt; uint32_t w, h, bpp; VkDeviceSize off; std::string tag; VkCommandBuffer cmd = VK_NULL_HANDLE; };
    std::vector<DumpRec> dumps;
    std::set<VkCommandBuffer> dump_submitted;   // command buffers submitted since the last flush (capture mode)
    std::unordered_map<uint32_t, VkImageView> storage_slot_view;
    std::unordered_map<uint32_t, VkImageView> tex_slot_view;

    // shadows (NVXRT_SHADOW b/c)
    std::unordered_map<VkImage, Shadow> shadows;   // by original image
    VkBuffer flag_buf = VK_NULL_HANDLE;
    VkDeviceMemory flag_mem = VK_NULL_HANDLE;
    volatile uint32_t *flag_map = nullptr;        // binding 5: flags raised by kernels
    bool c_off = false;                            // UI-copy shadows stopped (a non-8-bit value reached a shadow)
    uint64_t shadow_events = 0;

    // timing
    VkQueryPool qpool = VK_NULL_HANDLE;
    uint32_t qnext = 0;
    std::vector<std::pair<CuFunction *, uint32_t>> qpending;
    float ts_period = 1.0f;
    uint64_t launch_count = 0;

    // launch groups
    std::vector<std::unique_ptr<Group>> groups;
    std::unordered_map<std::string, Group *> group_first;          // "hash:entry" of member 0 -> group
    std::unordered_map<VkCommandBuffer, PendingGroup> pending;
    std::atomic<uint32_t> n_pending{0};   // pending.size(), read without the mutex by the command hooks
    std::deque<std::pair<uint32_t, Group *>> grp_recent;               // regions to check for a poisoned barrier
    std::set<VkCommandBuffer> cubin_cmds;                               // NVXRT_QUEUE_LOG: recorded launches
    std::map<VkQueue, uint64_t> cubin_queue_submits;
    std::map<VkQueue, std::pair<uint32_t, uint32_t>> queue_family; // NVXRT_QUEUE_LOG: native queue -> (family, index)
    VkBuffer grp_buf = VK_NULL_HANDLE;
    VkDeviceMemory grp_mem = VK_NULL_HANDLE;
    uint8_t *grp_map = nullptr;
    VkDeviceAddress grp_addr = 0;
    std::unique_ptr<CuModule> grp_module;

    // parameter blocks above 248 bytes (meta big_params): copied into this host-visible ring at record time
    VkBuffer bp_buf = VK_NULL_HANDLE;
    VkDeviceMemory bp_mem = VK_NULL_HANDLE;
    uint8_t *bp_map = nullptr;
    VkDeviceAddress bp_addr = 0;
    bool bp_failed = false;

    // ring slots (group regions, big parameter slots) are owned by the command buffer that recorded them until it is
    // begun again, reset or freed (or its pool is reset or destroyed): Vulkan only allows that once the GPU is done
    // with it, so a pending or resubmitted command buffer always reads its own parameters
    std::vector<uint32_t> grp_free, bp_free;
    struct CmdSlots { std::vector<uint32_t> grp, bp; };
    std::unordered_map<VkCommandBuffer, CmdSlots> cmd_slots;
    std::atomic<uint32_t> n_cmd_slots{0};   // cmd_slots.size(), read without the mutex by the command buffer hooks
    std::unordered_map<VkCommandBuffer, VkCommandPool> cmd_pool;
    bool grp_full_logged = false, bp_full_logged = false;
};

static uint32_t format_bpp(VkFormat f);

static std::shared_ptr<Device> device_of(const void *h)
{
    std::lock_guard<std::mutex> lk(g_map_mutex);
    auto it = g_devices.find(dkey(h));
    return it == g_devices.end() ? nullptr : it->second;
}

// device_of without the global lock or a reference count, for hooks on frequent commands: a per-thread cache that is
// valid while no device was created or destroyed (a device is not destroyed while its command buffers are recorded)
static std::atomic<uint64_t> g_dev_gen{0};
static Device *device_fast(const void *h)
{
    thread_local void *key = nullptr;
    thread_local uint64_t gen = ~0ull;
    thread_local Device *dev = nullptr;
    uint64_t g = g_dev_gen.load(std::memory_order_acquire);
    if (dkey(h) != key || g != gen) {
        dev = device_of(h).get();
        key = dkey(h);
        gen = g;
    }
    return dev;
}

// ---------------------------------------------------------------------------------------------- bindless set
static uint32_t find_memtype(Device &dev, uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < dev.memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (dev.memprops.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

// NEAREST sampler with unnormalized coordinates (texel centre = integer + 0.5): an exact texel fetch whose
// out-of-range handling (clamp to edge / border colour) the texture unit does
static VkSampler nearest_companion(Device &dev, VkSamplerAddressMode u, VkSamplerAddressMode v, VkBorderColor border)
{
    auto norm = [](VkSamplerAddressMode m) {
        // unnormalized coordinates allow only CLAMP_TO_EDGE / CLAMP_TO_BORDER; the translator treats the others as
        // clamp to edge as well
        return m == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ? m : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    };
    u = norm(u);
    v = norm(v);
    if (u != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER && v != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)
        border = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    border = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;   // CUDA border mode reads 0 whatever the API sampler says
    uint32_t key = (uint32_t)u | ((uint32_t)v << 8);
    auto it = dev.nearest_samplers.find(key);
    if (it != dev.nearest_samplers.end())
        return it->second;
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = ci.minFilter = VK_FILTER_NEAREST;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = u;
    ci.addressModeV = v;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.borderColor = border;
    ci.unnormalizedCoordinates = VK_TRUE;
    VkSampler s = VK_NULL_HANDLE;
    if (dev.d.CreateSampler(dev.handle, &ci, nullptr, &s) != VK_SUCCESS) {
        logf("nearest companion sampler creation failed");
        s = VK_NULL_HANDLE;
    }
    dev.nearest_samplers[key] = s;
    return s;
}

static void write_companion(Device &dev, uint32_t first, uint32_t count, VkSampler s)
{
    std::vector<VkDescriptorImageInfo> ii(count, VkDescriptorImageInfo{s, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = dev.set;
    w.dstBinding = 4;
    w.dstArrayElement = first;
    w.descriptorCount = count;
    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w.pImageInfo = ii.data();
    dev.d.UpdateDescriptorSets(dev.handle, 1, &w, 0, nullptr);
}

static bool init_bindless(Device &dev)
{
    VkDescriptorSetLayoutBinding b[6] = {};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxImages, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxImages, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER, kMaxSamplers, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_SAMPLER, kMaxSamplers, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorBindingFlags bf[6] = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT};
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfi.bindingCount = 6;
    bfi.pBindingFlags = bf;
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &bfi};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    li.bindingCount = 6;
    li.pBindings = b;
    if (dev.d.CreateDescriptorSetLayout(dev.handle, &li, nullptr, &dev.set_layout) != VK_SUCCESS)
        return false;
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dev.set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    if (dev.d.CreatePipelineLayout(dev.handle, &pli, nullptr, &dev.pipe_layout) != VK_SUCCESS)
        return false;
    VkDescriptorPoolSize ps[4] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxImages},
                                  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxImages},
                                  {VK_DESCRIPTOR_TYPE_SAMPLER, 2 * kMaxSamplers},
                                  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pci.maxSets = 1;
    pci.poolSizeCount = 4;
    pci.pPoolSizes = ps;
    if (dev.d.CreateDescriptorPool(dev.handle, &pci, nullptr, &dev.pool) != VK_SUCCESS)
        return false;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dev.pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &dev.set_layout;
    if (dev.d.AllocateDescriptorSets(dev.handle, &ai, &dev.set) != VK_SUCCESS)
        return false;
    // info buffer: uvec4 per sampler slot {filter, addrU, addrV, border}, then uint format class per storage-image
    // slot and per sampled-image slot (0 float/unorm, 1 uint, 2 sint)
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = kInfoBytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (dev.d.CreateBuffer(dev.handle, &bci, nullptr, &dev.smp_buf) != VK_SUCCESS)
        return false;
    VkMemoryRequirements mr;
    dev.d.GetBufferMemoryRequirements(dev.handle, dev.smp_buf, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memtype(dev, mr.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX)
        mai.memoryTypeIndex = find_memtype(dev, mr.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (dev.d.AllocateMemory(dev.handle, &mai, nullptr, &dev.smp_mem) != VK_SUCCESS)
        return false;
    dev.d.BindBufferMemory(dev.handle, dev.smp_buf, dev.smp_mem, 0);
    void *ptr = nullptr;
    dev.d.MapMemory(dev.handle, dev.smp_mem, 0, VK_WHOLE_SIZE, 0, &ptr);
    dev.smp_map = (uint32_t *)ptr;
    memset(dev.smp_map, 0, kInfoBytes);
    VkDescriptorBufferInfo dbi{dev.smp_buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = dev.set;
    w.dstBinding = 3;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &dbi;
    dev.d.UpdateDescriptorSets(dev.handle, 1, &w, 0, nullptr);
    // binding 5: flags the kernels raise (host-visible, read by the layer)
    {
        VkBufferCreateInfo fci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        fci.size = 256;
        fci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (dev.d.CreateBuffer(dev.handle, &fci, nullptr, &dev.flag_buf) != VK_SUCCESS)
            return false;
        VkMemoryRequirements fmr;
        dev.d.GetBufferMemoryRequirements(dev.handle, dev.flag_buf, &fmr);
        VkMemoryAllocateInfo fai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        fai.allocationSize = fmr.size;
        fai.memoryTypeIndex = find_memtype(dev, fmr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (fai.memoryTypeIndex == UINT32_MAX || dev.d.AllocateMemory(dev.handle, &fai, nullptr, &dev.flag_mem) != VK_SUCCESS)
            return false;
        dev.d.BindBufferMemory(dev.handle, dev.flag_buf, dev.flag_mem, 0);
        void *fp = nullptr;
        dev.d.MapMemory(dev.handle, dev.flag_mem, 0, VK_WHOLE_SIZE, 0, &fp);
        dev.flag_map = (volatile uint32_t *)fp;
        memset(fp, 0, 256);
        VkDescriptorBufferInfo fbi{dev.flag_buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet fw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        fw.dstSet = dev.set;
        fw.dstBinding = 5;
        fw.descriptorCount = 1;
        fw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        fw.pBufferInfo = &fbi;
        dev.d.UpdateDescriptorSets(dev.handle, 1, &fw, 0, nullptr);
    }
    // every companion slot valid from the start (a handle without a sampler decodes to slot 0)
    VkSampler edge = nearest_companion(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                       VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
    if (!edge)
        return false;
    write_companion(dev, 0, kMaxSamplers, edge);

    if (!g_dump_dir.empty()) {
        VkBufferCreateInfo dbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        dbci.size = dev.dump_size = 1536ull << 20;
        dbci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        dev.d.CreateBuffer(dev.handle, &dbci, nullptr, &dev.dump_buf);
        VkMemoryRequirements dmr;
        dev.d.GetBufferMemoryRequirements(dev.handle, dev.dump_buf, &dmr);
        VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        dai.allocationSize = dmr.size;
        dai.memoryTypeIndex = find_memtype(dev, dmr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (dai.memoryTypeIndex != UINT32_MAX && dev.d.AllocateMemory(dev.handle, &dai, nullptr, &dev.dump_mem) == VK_SUCCESS) {
            dev.d.BindBufferMemory(dev.handle, dev.dump_buf, dev.dump_mem, 0);
            void *p = nullptr;
            dev.d.MapMemory(dev.handle, dev.dump_mem, 0, VK_WHOLE_SIZE, 0, &p);
            dev.dump_map = (uint8_t *)p;
        }
        logf("dump buffer: %llu MB, memory type %u, %s", (unsigned long long)(dev.dump_size >> 20), dai.memoryTypeIndex,
             dev.dump_map ? "mapped" : "FAILED (no dumps/captures)");
    }
    if (g_timing) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kQueries;
        VkResult qr = dev.d.CreateQueryPool(dev.handle, &qi, nullptr, &dev.qpool);
        if (qr != VK_SUCCESS)
            logf("timestamp query pool creation failed (%d)", qr);
        else if (dev.d.ResetQueryPool)
            dev.d.ResetQueryPool(dev.handle, dev.qpool, 0, kQueries);
    }
    dev.ts_period = dev.props.limits.timestampPeriod;

    std::vector<uint8_t> cache;
    read_file(g_cache_dir + "/pipeline-cache.bin", cache);
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pcci.initialDataSize = cache.size();
    pcci.pInitialData = cache.empty() ? nullptr : cache.data();
    if (dev.d.CreatePipelineCache(dev.handle, &pcci, nullptr, &dev.pcache) != VK_SUCCESS) {
        pcci.initialDataSize = 0;
        pcci.pInitialData = nullptr;
        dev.d.CreatePipelineCache(dev.handle, &pcci, nullptr, &dev.pcache);
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- launch groups
static bool g_groups_on = true;
static bool g_queue_log = false;

// a host-visible (VRAM if possible), coherent, mapped buffer with a device address: parameter rings written at record time
static bool host_ring(Device &dev, VkDeviceSize size, VkBuffer &buf, VkDeviceMemory &mem, void *&ptr,
                      VkDeviceAddress &addr, uint32_t &memtype)
{
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkMemoryRequirements mr;
    VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
    bool ok = dev.d.CreateBuffer(dev.handle, &bci, nullptr, &buf) == VK_SUCCESS;
    if (ok) {
        dev.d.GetBufferMemoryRequirements(dev.handle, buf, &mr);
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_memtype(dev, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
            mai.memoryTypeIndex = find_memtype(dev, mr.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        ok = mai.memoryTypeIndex != UINT32_MAX &&
             dev.d.AllocateMemory(dev.handle, &mai, nullptr, &mem) == VK_SUCCESS &&
             dev.d.BindBufferMemory(dev.handle, buf, mem, 0) == VK_SUCCESS &&
             dev.d.MapMemory(dev.handle, mem, 0, VK_WHOLE_SIZE, 0, &ptr) == VK_SUCCESS;
    }
    if (ok) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = buf;
        addr = dev.d.GetBufferDeviceAddress ? dev.d.GetBufferDeviceAddress(dev.handle, &ai) : 0;
        ok = addr != 0;
    }
    memtype = mai.memoryTypeIndex;
    return ok;
}

static void ring_init(std::vector<uint32_t> &free, uint32_t n)
{
    free.clear();
    for (uint32_t i = n; i-- > 0;)
        free.push_back(i);   // taken from the back: slot 0 first
}

// a free ring slot, owned by cmd from now on; false when every slot is held by a recorded command buffer.
// caller holds dev.mutex
static bool ring_take(Device &dev, std::vector<uint32_t> Device::CmdSlots::*which, VkCommandBuffer cmd,
                      uint32_t &slot)
{
    std::vector<uint32_t> &free = which == &Device::CmdSlots::grp ? dev.grp_free : dev.bp_free;
    if (free.empty())
        return false;
    slot = free.back();
    free.pop_back();
    (dev.cmd_slots[cmd].*which).push_back(slot);
    dev.n_cmd_slots.store((uint32_t)dev.cmd_slots.size(), std::memory_order_relaxed);
    return true;
}

static void pending_changed(Device &dev)
{
    dev.n_pending.store((uint32_t)dev.pending.size(), std::memory_order_relaxed);
}

// cmd is no longer pending on the GPU and its recorded contents are gone: its ring slots are free again, and a group
// it left open cannot continue. caller holds dev.mutex
static void cmd_release(Device &dev, VkCommandBuffer cmd)
{
    auto it = dev.cmd_slots.find(cmd);
    if (it != dev.cmd_slots.end()) {
        dev.grp_free.insert(dev.grp_free.end(), it->second.grp.begin(), it->second.grp.end());
        dev.bp_free.insert(dev.bp_free.end(), it->second.bp.begin(), it->second.bp.end());
        dev.cmd_slots.erase(it);
        dev.n_cmd_slots.store((uint32_t)dev.cmd_slots.size(), std::memory_order_relaxed);
    }
    if (dev.pending.erase(cmd))
        pending_changed(dev);
}

// work other than a cubin launch recorded into cmd: an open launch group there must end, since its remaining members
// would otherwise run (inside the fused dispatch) before this work instead of after it
static void group_break(Device &dev, VkCommandBuffer cmd, const char *what)
{
    if (!dev.n_pending.load(std::memory_order_relaxed))
        return;
    std::lock_guard<std::mutex> lk(dev.mutex);
    auto it = dev.pending.find(cmd);
    if (it == dev.pending.end())
        return;
    logf("group %s: sequence broken after %u of %zu members by %s", it->second.g->name.c_str(), it->second.next,
         it->second.g->members.size(), what);
    dev.pending.erase(it);
    pending_changed(dev);
}

static void init_groups(Device &dev)
{
    g_queue_log = getenv("NVXRT_QUEUE_LOG") && atoi(getenv("NVXRT_QUEUE_LOG")) != 0;
    const char *ge = getenv("NVXRT_GROUPS");
    g_groups_on = !(ge && atoi(ge) == 0) && g_dump_dir.empty() && g_cap_launches.empty();
    if (!g_groups_on)
        return;
    std::string dir = g_cache_dir + "/groups";
    DIR *d = opendir(dir.c_str());
    if (!d)
        return;
    std::vector<std::string> names;
    while (struct dirent *e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".meta") == 0)
            names.push_back(n.substr(0, n.size() - 5));
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    if (names.empty())
        return;
    dev.grp_module = std::make_unique<CuModule>();
    dev.grp_module->hash = "group";
    static const bool fast_acc = getenv("NVXRT_ACC") && !strcmp(getenv("NVXRT_ACC"), "fast");
    for (auto &n : names) {
        std::vector<uint8_t> meta, spv;
        bool fa = fast_acc && read_file(dir + "/" + n + ".fa.spv", spv) && spv.size() >= 20;
        if (!read_file(dir + "/" + n + ".meta", meta) || (!fa && (!read_file(dir + "/" + n + ".spv", spv) || spv.size() < 20)))
            continue;
        if (fa)
            logf("group %s: fast accumulation variant (NVXRT_ACC=fast)", n.c_str());
        auto g = std::make_unique<Group>();
        g->name = n;
        std::string text(meta.begin(), meta.end());
        size_t pos = 0;
        while (pos < text.size()) {
            size_t nl = text.find('\n', pos);
            if (nl == std::string::npos)
                nl = text.size();
            std::string line = text.substr(pos, nl - pos);
            pos = nl + 1;
            char key[64] = {}, a[160] = {}, b[160] = {};
            unsigned x = 0, y = 0, z = 0;
            if (sscanf(line.c_str(), "%63s", key) != 1)
                continue;
            const char *rest = line.c_str() + strlen(key);
            if (!strcmp(key, "member") && sscanf(rest, "%159s %159s", a, b) == 2) {
                // "member H E [H E ...]": alternatives
                std::string m, h;
                std::vector<std::string> words;
                for (const char *w = rest; *w;) { // by hand: stream extraction reads locale facets
                    size_t skip = strspn(w, " \t\r\n"), len = strcspn(w + skip, " \t\r\n");
                    if (len) words.emplace_back(w + skip, len);
                    w += skip + len;
                }
                for (size_t i = 0; i + 1 < words.size(); i += 2)
                    m += (m.empty() ? "" : "|") + words[i] + ":" + words[i + 1];
                g->members.push_back(m);
            }
            else if (!strcmp(key, "block") && sscanf(rest, "%u %u %u", &x, &y, &z) == 3)
                g->block[0] = x, g->block[1] = y, g->block[2] = z;
            else if (!strcmp(key, "grid") && sscanf(rest, "%u %u %u", &x, &y, &z) == 3)
                g->grid[0] = x, g->grid[1] = y, g->grid[2] = z;
        }
        if (g->members.empty() || g->members.size() > kGrpMaxMembers || !g->block[0] || !g->block[1] || !g->block[2]) {
            logf("group %s: bad meta (%zu members)", n.c_str(), g->members.size());
            continue;
        }
        auto fn = std::make_unique<CuFunction>();
        fn->module = dev.grp_module.get();
        fn->name = "group:" + n;
        fn->meta.name = fn->name;
        fn->meta.hash = "group";
        fn->meta.spirv.resize(spv.size() / 4);
        memcpy(fn->meta.spirv.data(), spv.data(), fn->meta.spirv.size() * 4);
        VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sci.codeSize = fn->meta.spirv.size() * 4;
        sci.pCode = fn->meta.spirv.data();
        if (dev.d.CreateShaderModule(dev.handle, &sci, nullptr, &fn->shader) != VK_SUCCESS) {
            logf("group %s: vkCreateShaderModule failed", n.c_str());
            continue;
        }
        fn->meta.ok = true;
        g->fn = fn.get();
        dev.functions.push_back(std::move(fn));
        std::vector<std::string> firsts;
        for (size_t p0 = 0;;) {
            size_t q0 = g->members[0].find('|', p0);
            firsts.push_back(g->members[0].substr(p0, q0 == std::string::npos ? std::string::npos : q0 - p0));
            if (q0 == std::string::npos) break;
            p0 = q0 + 1;
        }
        bool clash = false;
        for (auto &f0 : firsts)
            if (dev.group_first.count(f0)) {
                logf("group %s: first member %s already starts group %s; ignored", n.c_str(), f0.c_str(),
                     dev.group_first[f0]->name.c_str());
                clash = true;
            }
        if (clash)
            continue;
        for (auto &f0 : firsts)
            dev.group_first[f0] = g.get();
        logf("group %s: %zu members, block %ux%ux%u grid %ux%ux%u", n.c_str(), g->members.size(), g->block[0],
             g->block[1], g->block[2], g->grid[0], g->grid[1], g->grid[2]);
        dev.groups.push_back(std::move(g));
    }
    if (dev.groups.empty())
        return;
    // parameter ring: host-visible VRAM (written at record time, read by the fused kernels)
    uint32_t memtype = 0;
    void *ptr = nullptr;
    bool ok = host_ring(dev, (VkDeviceSize)kGrpRegions * kGrpRegion, dev.grp_buf, dev.grp_mem, ptr, dev.grp_addr, memtype);
    if (!ok) {
        logf("launch groups: parameter ring allocation failed; groups disabled");
        dev.groups.clear();
        dev.group_first.clear();
        return;
    }
    dev.grp_map = (uint8_t *)ptr;
    memset(dev.grp_map, 0, (size_t)kGrpRegions * kGrpRegion);
    ring_init(dev.grp_free, kGrpRegions);
    logf("launch groups: %zu active, ring %u x %u B (memory type %u)", dev.groups.size(), kGrpRegions, kGrpRegion,
         memtype);
}

static const char *desc_type_name(VkDescriptorType t)
{
    switch (t) {
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return "tex";
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "texsmp";
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "surf";
    default: return "?";
    }
}

static VkFormatFeatureFlags format_features(Device &dev, VkFormat f)
{
    auto it = dev.fmt_features.find((int)f);
    if (it != dev.fmt_features.end())
        return it->second;
    VkFormatProperties fp{};
    if (dev.inst->d.GetPhysicalDeviceFormatProperties)
        dev.inst->d.GetPhysicalDeviceFormatProperties(dev.pd, f, &fp);
    dev.fmt_features[(int)f] = fp.optimalTilingFeatures;
    return fp.optimalTilingFeatures;
}

// the view to put into a descriptor of `binding` for `view` (the view itself, or a companion with the needed usage);
// null when the image does not allow that usage. caller holds dev.mutex
static VkImageView view_for_binding(Device &dev, VkImageView view, int binding)
{
    auto vi = dev.views.find(view);
    if (vi == dev.views.end())
        return VK_NULL_HANDLE;
    VkImageUsageFlags need = binding == 1 ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT;
    VkFormatFeatureFlags feat = binding == 1 ? VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT : VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    auto ii = dev.images.find(vi->second.image);
    if (ii == dev.images.end() || !(ii->second.usage & need) || !(format_features(dev, vi->second.format) & feat))
        return VK_NULL_HANDLE;
    if (!vi->second.usage || (vi->second.usage & need))
        return view;
    auto c = dev.companion.find(view);
    if (c != dev.companion.end())
        return c->second;
    if (binding == 1 && vi->second.range.levelCount != 1)
        return VK_NULL_HANDLE;
    VkImageViewUsageCreateInfo uci{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    uci.usage = need;
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &uci};
    ci.image = vi->second.image;
    ci.viewType = vi->second.type;
    ci.format = vi->second.format;
    ci.subresourceRange = vi->second.range;
    VkImageView cv = VK_NULL_HANDLE;
    if (dev.d.CreateImageView(dev.handle, &ci, nullptr, &cv) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    dev.companion[view] = cv;
    return cv;
}

// returns slot index for (view, binding) or 0 on failure. caller holds dev.mutex
static uint32_t view_slot(Device &dev, VkImageView view, int binding)
{
    auto key = std::make_pair(view, binding);
    auto it = dev.view_slot.find(key);
    if (it != dev.view_slot.end())
        return it->second;
    uint32_t slot = 0;
    int pool = g_legacy_slots ? binding : 0;
    auto vn = g_legacy_slots ? dev.view_num.end() : dev.view_num.find(view);
    if (vn != dev.view_num.end()) {
        slot = vn->second;
    } else if (g_legacy_slots && dev.alloc_shadow) {
        if (dev.next_shadow_slot[pool] >= kMaxImages) {
            logf("out of shadow slots (binding %d)", binding);
            return 0;
        }
        slot = dev.next_shadow_slot[pool]++;
    } else {
        // keep freed slots out of circulation for a while (in-flight command buffers may still reference them)
        if (dev.free_slots[pool].size() > 4096) {
            slot = dev.free_slots[pool].front();
            dev.free_slots[pool].pop_front();
        } else {
            if (dev.next_slot[pool] >= kMaxImages) {
                logf("out of bindless image slots (binding %d)", binding);
                return 0;
            }
            slot = dev.next_slot[pool]++;
        }
        if (!g_legacy_slots) {
            dev.view_num[view] = slot;
            dev.slot_view[slot] = view;
        }
    }
    VkImageView dv = g_legacy_slots ? view : view_for_binding(dev, view, binding);
    if (!dv)
        dv = view;   // the requested binding: as before (vkd3d created the view for it)
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, dv, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = dev.set;
    w.dstBinding = binding;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType = binding == 0 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.pImageInfo = &ii;
    dev.d.UpdateDescriptorSets(dev.handle, 1, &w, 0, nullptr);
    dev.view_slot[key] = slot;
    auto vi = dev.views.find(view);
    uint32_t cls = vi != dev.views.end() ? format_class(vi->second.format) | format_raw(vi->second.format) << 8 : 0;
    uint32_t *classes = dev.smp_map + kMaxSamplers * 4 + (binding == 1 ? 0 : kMaxImages);
    classes[slot] = cls;
    return slot;
}

// unified numbering: the other binding's descriptor at the view's slot, when the image allows that usage
static void view_slot_cross(Device &dev, VkImageView view, int binding)
{
    if (g_legacy_slots || dev.view_slot.count({view, 1 - binding}))
        return;
    if (view_for_binding(dev, view, 1 - binding))
        view_slot(dev, view, 1 - binding);
}

// frees a view's slots (and companion). caller holds dev.mutex
static void release_view_slots(Device &dev, VkImageView view)
{
    bool freed = false;
    for (int b = 0; b < 2; b++) {
        auto it = dev.view_slot.find({view, b});
        if (it == dev.view_slot.end())
            continue;
        auto &sm = b == 1 ? dev.storage_slot_view : dev.tex_slot_view;
        auto si = sm.find(it->second);
        if (si != sm.end() && si->second == view)
            sm.erase(si);
        if (g_legacy_slots) {
            if (it->second < 61440)   // (shadow slots of the legacy numbering are not recycled)
                dev.free_slots[b].push_back(it->second);
        }
        else if (!freed) {
            dev.free_slots[0].push_back(it->second);
            freed = true;
        }
        dev.view_slot.erase(it);
    }
    auto vn = dev.view_num.find(view);
    if (vn != dev.view_num.end()) {
        if (!freed && !g_legacy_slots)
            dev.free_slots[0].push_back(vn->second);
        dev.slot_view.erase(vn->second);
        dev.view_num.erase(vn);
    }
    auto c = dev.companion.find(view);
    if (c != dev.companion.end()) {
        dev.d.DestroyImageView(dev.handle, c->second, nullptr);
        dev.companion.erase(c);
    }
}

static uint32_t sampler_slot(Device &dev, VkSampler s)
{
    auto it = dev.sampler_slot.find(s);
    if (it != dev.sampler_slot.end())
        return it->second;
    uint32_t slot;
    if (dev.free_samplers.size() > 256) {
        slot = dev.free_samplers.front();
        dev.free_samplers.pop_front();
    } else {
        if (dev.next_sampler >= kMaxSamplers) {
            logf("out of sampler slots");
            return 0;
        }
        slot = dev.next_sampler++;
    }
    VkDescriptorImageInfo ii{s, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = dev.set;
    w.dstBinding = 2;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w.pImageInfo = &ii;
    dev.d.UpdateDescriptorSets(dev.handle, 1, &w, 0, nullptr);
    auto si = dev.samplers.find(s);
    uint32_t *e = dev.smp_map + 4 * slot;
    if (si != dev.samplers.end()) {
        e[0] = (uint32_t)si->second.min | ((uint32_t)si->second.mag << 4) | (si->second.unnormalized ? 0x100u : 0u);
        e[1] = (uint32_t)si->second.u;
        e[2] = (uint32_t)si->second.v;
        e[3] = (uint32_t)si->second.border;
        if (VkSampler c = nearest_companion(dev, si->second.u, si->second.v, si->second.border))
            write_companion(dev, slot, 1, c);
    } else {
        e[0] = e[1] = e[2] = e[3] = 0;
        if (VkSampler c = nearest_companion(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                            VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK))
            write_companion(dev, slot, 1, c);
    }
    dev.sampler_slot[s] = slot;
    return slot;
}


// ---------------------------------------------------------------------------------------------- shadows (B/C)
static const char *shadow_kind_name(int k) { return k == kShadowB ? "B" : "C"; }

static void shadow_logf(Device &dev, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void shadow_logf(Device &dev, const char *fmt, ...)
{
    // shadow events are rare (creation, conflicts); cap them anyway
    if (dev.shadow_events++ > 2000)
        return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    logf("shadow: %s", buf);
}

// caller holds dev.mutex
static void destroy_shadow(Device &dev, VkImage orig)
{
    auto it = dev.shadows.find(orig);
    if (it == dev.shadows.end())
        return;
    Shadow &sh = it->second;
    if (g_shadow_log)
        shadow_logf(dev, "%s image %p destroyed: shadow released (%llu shadow writes, %llu redirected reads)",
                    shadow_kind_name(sh.kind), (void *)orig, (unsigned long long)sh.writes, (unsigned long long)sh.reads);
    if (sh.view) {
        release_view_slots(dev, sh.view);
        dev.views.erase(sh.view);
        dev.d.DestroyImageView(dev.handle, sh.view, nullptr);
    }
    if (sh.img) {
        dev.images.erase(sh.img);
        dev.d.DestroyImage(dev.handle, sh.img, nullptr);
    }
    if (sh.mem)
        dev.d.FreeMemory(dev.handle, sh.mem, nullptr);
    dev.shadows.erase(it);
}

// the shadow of an original image, created on its first shadow write (null when the image is not eligible: a 2D,
// single-mip, single-layer RGBA16F storage+sampled image named through an RGBA16F view; vkd3d's one-layer 2D_ARRAY
// views of it count). Records the shadow's initial
// layout transition into cmd (the launch that first writes it). caller holds dev.mutex
static Shadow *shadow_for(Device &dev, VkCommandBuffer cmd, VkImage orig, const ViewInfo &vi, int kind)
{
    auto it = dev.shadows.find(orig);
    if (it != dev.shadows.end())
        return &it->second;
    auto ii = dev.images.find(orig);
    if (ii == dev.images.end())
        return nullptr;
    const ImageInfo &im = ii->second;
    const VkImageUsageFlags need = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (im.format != VK_FORMAT_R16G16B16A16_SFLOAT || vi.format != VK_FORMAT_R16G16B16A16_SFLOAT ||
        im.type != VK_IMAGE_TYPE_2D || (vi.type != VK_IMAGE_VIEW_TYPE_2D && vi.type != VK_IMAGE_VIEW_TYPE_2D_ARRAY) ||
        im.mips != 1 || im.layers != 1 ||
        im.samples != VK_SAMPLE_COUNT_1_BIT || (im.usage & need) != need) {
        if (g_shadow_log)
            shadow_logf(dev, "%s image %p not eligible (format %d/%d, %ux%u, mips %u layers %u usage 0x%x)",
                        shadow_kind_name(kind), (void *)orig, (int)im.format, (int)vi.format, im.extent.width,
                        im.extent.height, im.mips, im.layers, im.usage);
        Shadow &bad = dev.shadows[orig];   // remember: never try again for this image
        bad.kind = kind;
        bad.bad = true;
        return &bad;
    }
    Shadow sh;
    sh.kind = kind;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {im.extent.width, im.extent.height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool ok = dev.d.CreateImage(dev.handle, &ci, nullptr, &sh.img) == VK_SUCCESS;
    if (ok) {
        VkMemoryRequirements mr;
        dev.d.GetImageMemoryRequirements(dev.handle, sh.img, &mr);
        VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        ded.image = sh.img;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &ded};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_memtype(dev, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        ok = mai.memoryTypeIndex != UINT32_MAX && dev.d.AllocateMemory(dev.handle, &mai, nullptr, &sh.mem) == VK_SUCCESS &&
             dev.d.BindImageMemory(dev.handle, sh.img, sh.mem, 0) == VK_SUCCESS;
    }
    if (ok) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = sh.img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ok = dev.d.CreateImageView(dev.handle, &vci, nullptr, &sh.view) == VK_SUCCESS;
    }
    if (ok) {
        dev.views[sh.view] = {sh.img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_VIEW_TYPE_2D};
        dev.alloc_shadow = true;
        sh.tslot = view_slot(dev, sh.view, 0);
        sh.sslot = view_slot(dev, sh.view, 1);
        dev.alloc_shadow = false;
        ok = sh.tslot && sh.sslot;
    }
    if (!ok) {
        logf("shadow: %s image %p: shadow creation failed; original used", shadow_kind_name(kind), (void *)orig);
        if (sh.view) { dev.views.erase(sh.view); dev.d.DestroyImageView(dev.handle, sh.view, nullptr); }
        if (sh.img) dev.d.DestroyImage(dev.handle, sh.img, nullptr);
        if (sh.mem) dev.d.FreeMemory(dev.handle, sh.mem, nullptr);
        Shadow &bad = dev.shadows[orig];
        bad.kind = kind;
        bad.bad = true;
        return &bad;
    }
    dev.storage_slot_view[sh.sslot] = sh.view;   // (dumps, legacy-numbering lookups)
    {
        ImageInfo si{VK_FORMAT_R8G8B8A8_UNORM, ci.extent, ci.usage};
        dev.images[sh.img] = si;
    }
    dev.tex_slot_view[sh.tslot] = sh.view;
    if (kind == kShadowC) {
        // class 3 on both bindings: exact read fix-up (sampled), checked stores (storage)
        dev.smp_map[kMaxSamplers * 4 + sh.sslot] = 3;
        dev.smp_map[kMaxSamplers * 4 + kMaxImages + sh.tslot] = (g_shadow & 4) ? 0 : 3;
    }
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = 0;
    ib.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = sh.img;
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dev.d.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &ib);
    logf("shadow: %s: RGBA8 shadow %p for RGBA16F image %p (%ux%u), slots %u/%u", shadow_kind_name(kind),
         (void *)sh.img, (void *)orig, im.extent.width, im.extent.height, sh.tslot, sh.sslot);
    auto res = dev.shadows.emplace(orig, sh);
    return &res.first->second;
}

static bool shadow_active(const Device &dev) { return g_shadow != 0 && dev.bindless_ok; }

// the image a launch parameter names: true if the value is a handle of a known view. Unified numbering: one view per
// slot. Legacy numbering: the binding follows the kernel's use (texture reads: sampled, surface: storage; other: both
// are checked, the storage one first)
static bool param_image(Device &dev, uint64_t v, uint8_t use, VkImage &img, const ViewInfo *&vi)
{
    if (!v || (v >> 32))
        return false;
    uint32_t slot = (uint32_t)v & 0xfffffu, smp = (uint32_t)(v >> 20) & 0xfffu;
    if (slot < g_slot_base || (smp && smp >= dev.next_sampler))
        return false;
    VkImageView view = VK_NULL_HANDLE;
    if (!g_legacy_slots) {
        auto sv = dev.slot_view.find(slot);
        if (sv != dev.slot_view.end())
            view = sv->second;
    } else {
        if (use != kUseTex) {
            auto sv = dev.storage_slot_view.find(slot);
            if (sv != dev.storage_slot_view.end())
                view = sv->second;
        }
        if (!view && use != kUseSurf) {
            auto sv = dev.tex_slot_view.find(slot);
            if (sv != dev.tex_slot_view.end())
                view = sv->second;
        }
    }
    if (!view)
        return false;
    auto it = dev.views.find(view);
    if (it == dev.views.end())
        return false;
    img = it->second.image;
    vi = &it->second;
    return true;
}

static uint8_t param_use(const CuFunction &fn, uint32_t o)
{
    const KernelMeta &m = fn.meta;
    if (!m.huse_info || m.huse_dynamic)
        return kUseOther;
    auto it = m.huse.find(o);
    return it == m.huse.end() ? 0 : it->second;   // 0: the kernel never reads this parameter
}

// NVXRT_SHADOW_LOG=2: every parameter of a launch that names a full-resolution RGBA16F image (first launches only)
static void shadow_reflog(Device &dev, CuFunction &fn, const uint8_t *pbuf, uint32_t psize)
{
    static uint64_t n = 0;
    if (n > 6000)
        return;
    std::string line;
    for (uint32_t o = 0; o + 8 <= psize; o += 8) {
        uint64_t v;
        memcpy(&v, pbuf + o, 8);
        VkImage img;
        const ViewInfo *vi;
        uint8_t u = param_use(fn, o);
        if (!param_image(dev, v, u, img, vi))
            continue;
        auto ii = dev.images.find(img);
        if (ii == dev.images.end() || ii->second.format != VK_FORMAT_R16G16B16A16_SFLOAT ||
            (uint64_t)ii->second.extent.width * ii->second.extent.height < 2000000ull)
            continue;
        auto sh = dev.shadows.find(img);
        char b2[96];
        snprintf(b2, sizeof b2, " %s%u:%p%s", u == kUseTex ? "t" : (u == kUseSurf ? "s" : (u ? "?" : "-")), o, (void *)img,
                 sh == dev.shadows.end() ? "" : (sh->second.in_shadow ? "[S]" : "[O]"));
        line += b2;
    }
    if (!line.empty()) {
        n++;
        logf("shadow refs #%" PRIu64 " %s psize %u:%s", dev.launch_count, fn.name.c_str(), psize, line.c_str());
    }
}

// rewrites the handles of one launch's parameter block (pbuf -> out, psize bytes) per the shadow rules above; returns
// true if anything changed. caller holds dev.mutex
static bool shadow_params(Device &dev, VkCommandBuffer cmd, CuFunction &fn, const uint8_t *pbuf, uint32_t psize,
                          uint8_t *out)
{
    if (g_shadow_log >= 2)
        shadow_reflog(dev, fn, pbuf, psize);
    // UI-copy shadows stop when a kernel flagged a value that the RGBA8 shadow cannot hold exactly
    if ((g_shadow & kShadowC) && !dev.c_off && dev.flag_map && dev.flag_map[0]) {
        dev.c_off = true;
        logf("shadow: C: a value stored into an RGBA8 shadow was not an 8-bit quantity: UI-copy shadows off from now on "
             "(images already in shadows are read from them until overwritten)");
    }
    if (dev.shadows.empty() && fn.shadow_roles.empty())
        return false;
    memcpy(out, pbuf, psize);
    bool changed = false;
    struct Pend { uint32_t o; VkImage img; const ViewInfo *vi; const ShadowRole *role; uint8_t use; };
    Pend writes[64];
    uint32_t nw = 0;
    // pass 1: texture reads follow the content as it is before this launch
    for (uint32_t o = 0; o + 8 <= psize; o += 8) {
        uint64_t v;
        memcpy(&v, pbuf + o, 8);
        uint8_t use = param_use(fn, o);
        if (!use)
            continue;   // the kernel never reads this parameter
        VkImage img;
        const ViewInfo *vi;
        if (!param_image(dev, v, use, img, vi))
            continue;
        const ShadowRole *role = nullptr;
        for (auto *r : fn.shadow_roles)
            if (r->off == o && r->psize == psize && use == kUseSurf)
                role = r;
        if (use != kUseTex) {
            if (nw < 64)
                writes[nw++] = {o, img, vi, role, use};
            continue;
        }
        auto it = dev.shadows.find(img);
        if (it == dev.shadows.end() || !it->second.in_shadow)
            continue;
        Shadow &sh = it->second;
        if (sh.kind == kShadowC && !fn.meta.shadow8 && !(g_shadow & 4)) {   // (r: raw reads are the point)
            // a kernel without the exact read fix-up cannot read a UI-copy shadow: original read, no more shadowing
            if (!sh.bad)
                shadow_logf(dev, "C image %p (content in shadow) read by %s+%u, which has no shadow8 fix-up: image no "
                            "longer shadowed", (void *)img, fn.name.c_str(), o);
            sh.bad = true;
            continue;
        }
        if (vi->format != VK_FORMAT_R16G16B16A16_SFLOAT) {
            // another view format of shadowed content: cannot be redirected
            if (!sh.bad)
                shadow_logf(dev, "%s image %p read through a format-%d view by %s+%u: original read, image no longer shadowed",
                            shadow_kind_name(sh.kind), (void *)img, (int)vi->format, fn.name.c_str(), o);
            sh.bad = true;
            continue;
        }
        uint64_t nv = (v & ~(uint64_t)0xfffffu) | sh.tslot;
        memcpy(out + o, &nv, 8);
        sh.reads++;
        changed = true;
        if (g_trace)
            logf("   shadow %s: %s+%u reads image %p from its shadow", shadow_kind_name(sh.kind), fn.name.c_str(), o,
                 (void *)img);
    }
    // pass 2: surface accesses (writes, or reads we cannot tell apart) and other uses
    for (uint32_t i = 0; i < nw; i++) {
        const Pend &w = writes[i];
        const ShadowRole *role = w.role;
        if (role && role->what == kRoleShadowWrite && (g_shadow & role->kind) &&
            !(role->kind == kShadowC && (dev.c_off || !fn.meta.shadow8))) {
            Shadow *sh = shadow_for(dev, cmd, w.img, *w.vi, role->kind);
            if (sh && !sh->bad && sh->kind == role->kind) {
                uint64_t v;
                memcpy(&v, pbuf + w.o, 8);
                uint64_t nv = (v & ~(uint64_t)0xfffffu) | sh->sslot;
                memcpy(out + w.o, &nv, 8);
                sh->in_shadow = true;
                sh->writes++;
                changed = true;
                if (g_trace)
                    logf("   shadow %s: %s+%u writes image %p into its shadow", shadow_kind_name(sh->kind),
                         fn.name.c_str(), w.o, (void *)w.img);
                continue;
            }
        }
        auto it = dev.shadows.find(w.img);
        if (it == dev.shadows.end())
            continue;
        Shadow &sh = it->second;
        if (w.use == kUseOther) {
            // a use we cannot follow (not a texture read, not a surface access)
            if (sh.in_shadow && !sh.bad)
                shadow_logf(dev, "%s image %p (content in shadow) used by %s+%u in a way the layer cannot follow: image "
                            "no longer shadowed", shadow_kind_name(sh.kind), (void *)w.img, fn.name.c_str(), w.o);
            if (sh.in_shadow)
                sh.bad = true;
            continue;
        }
        // surface access: a known writer (any role) moves the content to the original; an unknown one might also read
        if (sh.in_shadow && !role && !sh.bad) {
            shadow_logf(dev, "%s image %p (content in shadow) accessed as surface by %s+%u: image no longer shadowed",
                        shadow_kind_name(sh.kind), (void *)w.img, fn.name.c_str(), w.o);
            sh.bad = true;
        }
        sh.in_shadow = false;
    }
    return changed;
}

// vkd3d copies / clears / blits that name an original: the content moves to the original (write), or a read of
// content that is in the shadow cannot be served (logged, the image is no longer shadowed)
static void shadow_touch(Device &dev, VkImage img, bool write, const char *what)
{
    if (!img)
        return;
    auto it = dev.shadows.find(img);
    if (it == dev.shadows.end())
        return;
    Shadow &sh = it->second;
    if (!write && sh.in_shadow && !sh.bad) {
        shadow_logf(dev, "%s image %p (content in shadow) read by %s: image no longer shadowed", shadow_kind_name(sh.kind),
                    (void *)img, what);
        sh.bad = true;
    }
    if (write && sh.in_shadow && g_shadow_log)
        shadow_logf(dev, "%s image %p written by %s: content back in the original", shadow_kind_name(sh.kind),
                    (void *)img, what);
    if (write)
        sh.in_shadow = false;
}

#define SHADOW_TOUCH(cmd, img, write, what)                                                                              \
    do {                                                                                                               \
        if (dev)                                                                                                       \
            group_break(*dev, cmd, what);                                                                              \
        if (dev && !dev->shadows.empty()) {                                                                            \
            std::lock_guard<std::mutex> lk_(dev->mutex);                                                               \
            shadow_touch(*dev, img, write, what);                                                                      \
        }                                                                                                              \
    } while (0)

static VKAPI_ATTR void VKAPI_CALL CmdCopyImage(VkCommandBuffer cmd, VkImage src, VkImageLayout sl, VkImage dst,
                                               VkImageLayout dl, uint32_t n, const VkImageCopy *r)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, src, false, "vkCmdCopyImage");
    SHADOW_TOUCH(cmd, dst, true, "vkCmdCopyImage");
    dev->d.CmdCopyImage(cmd, src, sl, dst, dl, n, r);
}
static VKAPI_ATTR void VKAPI_CALL CmdCopyImage2(VkCommandBuffer cmd, const VkCopyImageInfo2 *i)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, i->srcImage, false, "vkCmdCopyImage2");
    SHADOW_TOUCH(cmd, i->dstImage, true, "vkCmdCopyImage2");
    dev->d.CmdCopyImage2(cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL CmdBlitImage(VkCommandBuffer cmd, VkImage src, VkImageLayout sl, VkImage dst,
                                               VkImageLayout dl, uint32_t n, const VkImageBlit *r, VkFilter f)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, src, false, "vkCmdBlitImage");
    SHADOW_TOUCH(cmd, dst, true, "vkCmdBlitImage");
    dev->d.CmdBlitImage(cmd, src, sl, dst, dl, n, r, f);
}
static VKAPI_ATTR void VKAPI_CALL CmdBlitImage2(VkCommandBuffer cmd, const VkBlitImageInfo2 *i)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, i->srcImage, false, "vkCmdBlitImage2");
    SHADOW_TOUCH(cmd, i->dstImage, true, "vkCmdBlitImage2");
    dev->d.CmdBlitImage2(cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL CmdResolveImage(VkCommandBuffer cmd, VkImage src, VkImageLayout sl, VkImage dst,
                                                  VkImageLayout dl, uint32_t n, const VkImageResolve *r)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, src, false, "vkCmdResolveImage");
    SHADOW_TOUCH(cmd, dst, true, "vkCmdResolveImage");
    dev->d.CmdResolveImage(cmd, src, sl, dst, dl, n, r);
}
static VKAPI_ATTR void VKAPI_CALL CmdResolveImage2(VkCommandBuffer cmd, const VkResolveImageInfo2 *i)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, i->srcImage, false, "vkCmdResolveImage2");
    SHADOW_TOUCH(cmd, i->dstImage, true, "vkCmdResolveImage2");
    dev->d.CmdResolveImage2(cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(VkCommandBuffer cmd, VkImage img, VkImageLayout l,
                                                     const VkClearColorValue *c, uint32_t n, const VkImageSubresourceRange *r)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, img, true, "vkCmdClearColorImage");
    dev->d.CmdClearColorImage(cmd, img, l, c, n, r);
}
static VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBufferS(VkCommandBuffer cmd, VkImage src, VkImageLayout sl, VkBuffer dst,
                                                        uint32_t n, const VkBufferImageCopy *r)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, src, false, "vkCmdCopyImageToBuffer");
    dev->d.CmdCopyImageToBuffer(cmd, src, sl, dst, n, r);
}
static VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer2(VkCommandBuffer cmd, const VkCopyImageToBufferInfo2 *i)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, i->srcImage, false, "vkCmdCopyImageToBuffer2");
    dev->d.CmdCopyImageToBuffer2(cmd, i);
}
static VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer cmd, VkBuffer src, VkImage dst, VkImageLayout dl,
                                                       uint32_t n, const VkBufferImageCopy *r)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, dst, true, "vkCmdCopyBufferToImage");
    dev->d.CmdCopyBufferToImage(cmd, src, dst, dl, n, r);
}
static VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage2(VkCommandBuffer cmd, const VkCopyBufferToImageInfo2 *i)
{
    auto dev = device_of(cmd);
    SHADOW_TOUCH(cmd, i->dstImage, true, "vkCmdCopyBufferToImage2");
    dev->d.CmdCopyBufferToImage2(cmd, i);
}

// ---------------------------------------------------------------------------------------------- kernels
// true when nothing in the module depends on which lanes share a subgroup: no GroupNonUniform*, SubgroupBallotKHR or
// CooperativeMatrixKHR capability and no subgroup built-in (size, id, invocation id, masks, count)
static bool spirv_lane_free(const std::vector<uint32_t> &w)
{
    for (size_t i = 5; i < w.size();) {
        uint32_t op = w[i] & 0xffffu, wc = w[i] >> 16;
        if (!wc || i + wc > w.size())
            return false;
        if (op == 17 /* OpCapability */) {
            uint32_t c = w[i + 1];
            if ((c >= 61 && c <= 68) || c == 4423 || c == 6022)
                return false;
        }
        if (op == 71 /* OpDecorate */ && wc >= 4 && w[i + 2] == 11 /* BuiltIn */) {
            uint32_t b = w[i + 3];
            if (b == 36 || b == 38 || b == 40 || b == 41 || (b >= 4416 && b <= 4420))
                return false;
        }
        i += wc;
    }
    return true;
}

static bool load_kernel(Device &dev, CuFunction &fn)
{
    KernelMeta &m = fn.meta;
    m.name = fn.name;
    m.hash = fn.module->hash;
    if (m.hash.empty())
        return false;
    std::string base = g_cache_dir + "/kernels/" + m.hash + "/" + fn.name;
    std::vector<uint8_t> spv, meta;
    // NVXRT_TEX=hw: fast mode, linear-filtered tex.2d through the texture unit (ENTRY.hw.spv, built by dlssg-setup
    // for kernels with filtered fetches); anything else (default): the exact CUDA bilinear model
    static const bool hw_tex = getenv("NVXRT_TEX") && !strcmp(getenv("NVXRT_TEX"), "hw");
    // NVXRT_ACC=fast (not exact): fp32 accumulation in the T1 WMMA chains (ENTRY.fa[.hw].spv,
    // written by dlssg-setup only for kernels whose source differs); other kernels load as before
    static const bool fast_acc = getenv("NVXRT_ACC") && !strcmp(getenv("NVXRT_ACC"), "fast");
    if (fast_acc && hw_tex && read_file(base + ".fa.hw.spv", spv) && spv.size() >= 20) {
        fn.hw_tex = fn.fast_acc = true;
    } else if (fast_acc && read_file(base + ".fa.spv", spv) && spv.size() >= 20) {
        fn.fast_acc = true;
    } else if (hw_tex && read_file(base + ".hw.spv", spv) && spv.size() >= 20) {
        fn.hw_tex = true;
    } else if (!read_file(base + ".spv", spv) || spv.size() < 20) {
        spv.clear();
    }
    if (spv.size() < 20 || !read_file(base + ".meta", meta)) {
        logf("kernel %s (%s): no translation in cache (%s.spv) - run dlssg-setup", fn.name.c_str(),
             m.hash.substr(0, 16).c_str(), base.c_str());
        return false;
    }
    m.spirv.resize(spv.size() / 4);
    memcpy(m.spirv.data(), spv.data(), m.spirv.size() * 4);
    m.lane_free = spirv_lane_free(m.spirv);
    // plain sscanf parsing: iostream extraction of integers failed inside Wine processes (locale)
    std::string text(meta.begin(), meta.end());
    size_t pos = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos)
            nl = text.size();
        std::string line = text.substr(pos, nl - pos);
        pos = nl + 1;
        char key[64] = {};
        if (sscanf(line.c_str(), "%63s", key) != 1)
            continue;
        const char *rest = line.c_str() + strlen(key);
        unsigned a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
        if (!strcmp(key, "param_size") && sscanf(rest, "%u", &a) == 1)
            m.param_size = a;
        else if (!strcmp(key, "param") && sscanf(rest, "%u %u", &a, &b) == 2)
            m.params.push_back({a, b});
        else if (!strcmp(key, "features"))
            m.features = rest;
        else if (!strcmp(key, "t1_block") && sscanf(rest, "%u %u %u", &a, &b, &c) == 3) {
            m.t1_block[0] = a, m.t1_block[1] = b, m.t1_block[2] = c;
            m.t1_dispatch = a && b && c;
        } else if (!strcmp(key, "huse") && sscanf(rest, "%u", &a) == 1) {
            char kind[16] = {};
            sscanf(rest, "%*u %15s", kind);
            m.huse[a] = !strcmp(kind, "tex") ? kUseTex : (!strcmp(kind, "surf") ? kUseSurf : kUseOther);
            m.huse_info = true;
        } else if (!strcmp(key, "shadow8")) {
            m.shadow8 = true;
        } else if (!strcmp(key, "big_params")) {
            m.big_params = true;
        } else if (!strcmp(key, "wave") && sscanf(rest, "%u", &a) == 1) {
            m.wave64 = a == 64;
        } else if (!strcmp(key, "huse_dynamic")) {
            m.huse_dynamic = m.huse_info = true;
        } else if (!strcmp(key, "t1_fixed") && sscanf(rest, "%u %u %u", &a, &b, &c) == 3) {
            m.t1_fixed[0] = a, m.t1_fixed[1] = b, m.t1_fixed[2] = c;
        } else if (!strcmp(key, "t1_grid") && sscanf(rest, "%u %u %u %u %u %u", &a, &b, &c, &d, &e, &f) == 6) {
            m.t1_num[0] = a, m.t1_den[0] = b ? b : 1, m.t1_num[1] = c, m.t1_den[1] = d ? d : 1;
            m.t1_num[2] = e, m.t1_den[2] = f ? f : 1;
        }
    }
    if (getenv("NVXRT_META_DEBUG"))
        logf("meta %s: %zu bytes, t1 %d", base.c_str(), meta.size(), (int)m.t1_dispatch);
    if (fn.hw_tex || fn.fast_acc)
        logf("kernel %s:%s%s", fn.name.c_str(), fn.hw_tex ? " fast filtering variant (NVXRT_TEX=hw)" : "",
             fn.fast_acc ? " fast accumulation variant (NVXRT_ACC=fast)" : "");
    if (m.t1_dispatch)
        logf("kernel %s: T1 dispatch block %ux%ux%u grid x%u/%u x%u/%u x%u/%u", fn.name.c_str(), m.t1_block[0],
             m.t1_block[1], m.t1_block[2], m.t1_num[0], m.t1_den[0], m.t1_num[1], m.t1_den[1], m.t1_num[2], m.t1_den[2]);
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize = m.spirv.size() * 4;
    sci.pCode = m.spirv.data();
    if (dev.d.CreateShaderModule(dev.handle, &sci, nullptr, &fn.shader) != VK_SUCCESS) {
        logf("kernel %s: vkCreateShaderModule failed", fn.name.c_str());
        return false;
    }
    m.ok = true;
    return true;
}

static VkPipeline get_pipeline(Device &dev, CuFunction &fn, const PipeKey &k)
{
    auto it = fn.pipelines.find(k);
    if (it != fn.pipelines.end())
        return it->second;
    VkPipeline p = VK_NULL_HANDLE;
    if (fn.meta.ok) {
        uint32_t spec[5] = {k.bx, k.by, k.bz, k.smem, (k.bx * k.by * k.bz + 31) / 32};
        VkSpecializationMapEntry me[5] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}};
        VkSpecializationInfo si{5, me, sizeof(spec), spec};
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        const bool w64 = fn.meta.lane_free && (g_wave64 == 1 || (g_wave64 == 2 && fn.meta.wave64));
        rss.requiredSubgroupSize = w64 ? 64 : 32;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.pNext = &rss;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = fn.shader;
        ci.stage.pName = "main";
        ci.stage.pSpecializationInfo = &si;
        ci.layout = dev.pipe_layout;
        VkResult r = dev.d.CreateComputePipelines(dev.handle, dev.pcache, 1, &ci, nullptr, &p);
        if (r != VK_SUCCESS) {
            logf("kernel %s: pipeline creation failed (%d) for block %ux%ux%u", fn.name.c_str(), r, k.bx, k.by, k.bz);
            p = VK_NULL_HANDLE;
        } else {
            logf("kernel %s: pipeline for block %ux%ux%u smem %u (%s)%s", fn.name.c_str(), k.bx, k.by, k.bz, k.smem,
                 fn.module ? fn.module->hash.substr(0, 12).c_str() : "group", w64 ? " wave64" : "");
        }
    }
    fn.pipelines[k] = p;
    return p;
}

static void save_pipeline_cache(Device &dev)
{
    if (!dev.pcache)
        return;
    size_t n = 0;
    if (dev.d.GetPipelineCacheData(dev.handle, dev.pcache, &n, nullptr) != VK_SUCCESS || !n)
        return;
    std::vector<uint8_t> buf(n);
    if (dev.d.GetPipelineCacheData(dev.handle, dev.pcache, &n, buf.data()) != VK_SUCCESS)
        return;
    mkdirs(g_cache_dir);
    std::string path = g_cache_dir + "/pipeline-cache.bin";
    std::string tmp = path + ".tmp" + std::to_string(getpid());
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f)
        return;
    fwrite(buf.data(), 1, n, f);
    fclose(f);
    rename(tmp.c_str(), path.c_str());
}

// ---------------------------------------------------------------------------------------------- NVX entry points
static VKAPI_ATTR VkResult VKAPI_CALL CreateCuModuleNVX(VkDevice device, const VkCuModuleCreateInfoNVX *ci,
                                                         const VkAllocationCallbacks *, VkCuModuleNVX *out)
{
    auto dev = device_of(device);
    if (!dev || !ci || !ci->pData)
        return VK_ERROR_INITIALIZATION_FAILED;
    bool tex64 = false;
    for (auto *s = (const VkBaseInStructure *)ci->pNext; s; s = s->pNext)
        if (s->sType == VK_STRUCTURE_TYPE_CU_MODULE_TEXTURING_MODE_CREATE_INFO_NVX)
            tex64 = ((const VkCuModuleTexturingModeCreateInfoNVX *)s)->use64bitTexturing;
    std::vector<FatbinEntry> entries;
    if (!parse_cuda_module(ci->pData, ci->dataSize, entries)) {
        logf("CreateCuModule: unrecognised module (%zu bytes)", ci->dataSize);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto mod = std::make_unique<CuModule>();
    // pick the newest PTX target <= 120 (what a Blackwell driver would JIT); else SASS only
    const FatbinEntry *best = nullptr;
    for (auto &e : entries)
        if (e.kind == 1 && e.sm <= 120 && (!best || e.sm > best->sm))
            best = &e;
    std::string kinds;
    for (auto &e : entries)
        kinds += std::string(e.kind == 1 ? " ptx" : " sass") + "_sm" + std::to_string(e.sm);
    if (best) {
        mod->ptx.assign(best->data.begin(), best->data.end());
        mod->sm = best->sm;
        std::string norm = normalize_ptx(mod->ptx);
        mod->hash = sha256_hex(norm.data(), norm.size());
        mod->entries = ptx_entries(mod->ptx);
        std::string dir = g_cache_dir + "/kernels/" + mod->hash;
        mkdirs(dir);
        write_file_if_absent(dir + "/module.ptx", mod->ptx.data(), mod->ptx.size());
    } else {
        mod->sass_only = true;
        for (auto &e : entries)
            if (e.kind == 2) {
                auto n = elf_kernel_names(e.data.data(), e.data.size());
                mod->sass_names.insert(mod->sass_names.end(), n.begin(), n.end());
                std::string h = sha256_hex(e.data.data(), e.data.size());
                std::string dir = g_cache_dir + "/sass";
                mkdirs(dir);
                write_file_if_absent(dir + "/" + h + ".cubin", e.data.data(), e.data.size());
                mod->hash = "sass-" + h;
            }
    }
    std::string names;
    for (auto &n : (best ? mod->entries : mod->sass_names))
        names += " " + n;
    logf("CreateCuModule: %zu bytes,%s tex64=%d -> %s [%s ]", ci->dataSize, kinds.c_str(), tex64,
         mod->hash.substr(0, 16).c_str(), names.c_str());
    std::lock_guard<std::mutex> lk(dev->mutex);
    *out = (VkCuModuleNVX)(uintptr_t)mod.get();
    dev->modules.push_back(std::move(mod));
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL CreateCuFunctionNVX(VkDevice device, const VkCuFunctionCreateInfoNVX *ci,
                                                           const VkAllocationCallbacks *, VkCuFunctionNVX *out)
{
    auto dev = device_of(device);
    if (!dev || !ci || !ci->module || !ci->pName)
        return VK_ERROR_INITIALIZATION_FAILED;
    auto fn = std::make_unique<CuFunction>();
    fn->module = (CuModule *)(uintptr_t)ci->module;
    fn->name = ci->pName;
    fn->gkey = fn->module->hash + ":" + fn->name;
    bool found = fn->module->sass_only
                     ? std::find(fn->module->sass_names.begin(), fn->module->sass_names.end(), fn->name) !=
                           fn->module->sass_names.end()
                     : std::find(fn->module->entries.begin(), fn->module->entries.end(), fn->name) !=
                           fn->module->entries.end();
    if (!found) {
        logf("CreateCuFunction: %s not in module %s", ci->pName, fn->module->hash.substr(0, 16).c_str());
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (g_shadow)
        for (auto &r : kShadowRoles)
            if (fn->gkey == r.gkey && (g_shadow & r.kind) && (!r.raw || (g_shadow & 4)))
                fn->shadow_roles.push_back(&r);
    std::lock_guard<std::mutex> lk(dev->mutex);
    bool ok = load_kernel(*dev, *fn);
    logf("CreateCuFunction: %s (%s) %s", ci->pName, fn->module->hash.substr(0, 16).c_str(),
         ok ? "translated" : "NOT AVAILABLE");
    *out = (VkCuFunctionNVX)(uintptr_t)fn.get();
    dev->functions.push_back(std::move(fn));
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL DestroyCuModuleNVX(VkDevice, VkCuModuleNVX, const VkAllocationCallbacks *)
{
    // modules are kept until device destruction (cheap; NGX creates them once per feature)
}

static VKAPI_ATTR void VKAPI_CALL DestroyCuFunctionNVX(VkDevice, VkCuFunctionNVX, const VkAllocationCallbacks *) {}

#define CU_LAUNCH_PARAM_BUFFER_POINTER ((const void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((const void *)0x02)
#define CU_LAUNCH_PARAM_END ((const void *)0x00)

static std::string decode_params(Device &dev, const uint8_t *p, uint32_t n)
{
    std::string s;
    char buf[160];
    for (uint32_t o = 0; o + 8 <= n; o += 8) {
        uint64_t v;
        memcpy(&v, p + o, 8);
        if (!v)
            continue;
        auto it = dev.buffer_ranges.upper_bound(v);
        if (it != dev.buffer_ranges.begin()) {
            --it;
            if (v >= it->first && v < it->first + it->second.size) {
                snprintf(buf, sizeof buf, " [+%u]=buf%" PRIx64 "+%" PRIu64 "(%" PRIu64 "B)", o, (uint64_t)it->first,
                         v - it->first, (uint64_t)it->second.size);
                s += buf;
                continue;
            }
        }
        auto h = dev.handle_desc.find((uint32_t)v);
        if ((v >> 32) == 0 && h != dev.handle_desc.end()) {
            snprintf(buf, sizeof buf, " [+%u]=%s", o, h->second.c_str());
            s += buf;
        }
    }
    return s;
}

static void collect_timing(Device &dev)
{
    if (!dev.qpool || dev.qpending.empty())
        return;
    // results are read back lazily (WAIT would stall); query availability per pair
    std::vector<std::pair<CuFunction *, uint32_t>> still;
    for (auto &q : dev.qpending) {
        uint64_t r[4];
        VkResult res = dev.d.GetQueryPoolResults(dev.handle, dev.qpool, q.second, 2, sizeof(r), r, 16,
                                                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (res == VK_SUCCESS && r[1] && r[3]) {
            double ms = (double)(r[2] - r[0]) * dev.ts_period * 1e-6;
            static int dbg = 0;
            if (getenv("NVXRT_TIMING_DEBUG") && dbg++ < 5)
                logf("timing %s: %llu -> %llu period %f ms %f", q.first->name.c_str(), (unsigned long long)r[0],
                     (unsigned long long)r[2], dev.ts_period, ms);
            q.first->gpu_ms_total += ms;
            q.first->gpu_samples++;
        } else if (res == VK_NOT_READY) {
            still.push_back(q);
            continue;
            if (getenv("NVXRT_TIMING_DEBUG") && still.size() < 3)
                logf("timing query %u not ready: avail %llu %llu ts %llu %llu", q.second, (unsigned long long)r[1],
                     (unsigned long long)r[3], (unsigned long long)r[0], (unsigned long long)r[2]);
        } else if (getenv("NVXRT_TIMING_DEBUG")) {
            logf("timing query %u: res %d avail %llu %llu", q.second, res, (unsigned long long)r[1], (unsigned long long)r[3]);
        }
    }
    dev.qpending.swap(still);
}

static void print_summary(Device &dev, bool reset, const char *what);

// copy every buffer a launch's pointer parameters reference (whole buffer), plus the parameter block
static void capture_buffers(Device &dev, VkCommandBuffer cmd, CuFunction &fn, const uint8_t *pbuf, uint32_t psize,
                            const VkCuLaunchInfoNVX *li, const char *phase)
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    dev.d.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    std::string dir = g_dump_dir + "/L" + std::to_string(dev.launch_count);
    mkdirs(dir);
    if (!strcmp(phase, "pre")) {
        FILE *f = fopen((dir + "/params.bin").c_str(), "wb");
        if (f) { fwrite(pbuf, 1, psize, f); fclose(f); }
        f = fopen((dir + "/meta.txt").c_str(), "w");
        if (f) {
            fprintf(f, "kernel %s\nhash %s\ngrid %u %u %u\nblock %u %u %u\nsmem %u\npsize %u\n", fn.name.c_str(),
                    fn.module->hash.c_str(), li->gridDimX, li->gridDimY, li->gridDimZ, li->blockDimX, li->blockDimY,
                    li->blockDimZ, li->sharedMemBytes, psize);
            for (uint32_t o = 0; o + 8 <= psize; o += 8) {
                uint64_t v;
                memcpy(&v, pbuf + o, 8);
                auto it = dev.buffer_ranges.upper_bound(v);
                if (!v || it == dev.buffer_ranges.begin()) continue;
                --it;
                if (v >= it->first && v < it->first + it->second.size)
                    fprintf(f, "ptr %u %llx %llu %llu\n", o, (unsigned long long)it->first,
                            (unsigned long long)(v - it->first), (unsigned long long)it->second.size);
            }
            fclose(f);
        }
    }
    std::set<VkDeviceAddress> done;
    for (uint32_t o = 0; o + 8 <= psize; o += 8) {
        uint64_t v;
        memcpy(&v, pbuf + o, 8);
        auto it = dev.buffer_ranges.upper_bound(v);
        if (!v || it == dev.buffer_ranges.begin()) continue;
        --it;
        if (v < it->first || v >= it->first + it->second.size || done.count(it->first)) continue;
        done.insert(it->first);
        VkDeviceSize len = it->second.size;
        if (dev.dump_used + len > dev.dump_size) continue;
        VkBufferCopy bc{0, dev.dump_used, len};
        dev.d.CmdCopyBuffer(cmd, it->second.buffer, dev.dump_buf, 1, &bc);
        char tag[128];
        snprintf(tag, sizeof tag, "%s_%llx", phase, (unsigned long long)it->first);
        dev.dumps.push_back({dev.launch_count, fn.name, o, VK_FORMAT_UNDEFINED, (uint32_t)len, 1, 1, dev.dump_used, tag, cmd});
        dev.dump_used += (len + 255) & ~(VkDeviceSize)255;
    }
    mb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    dev.d.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
}

static void record_timed_dispatch_begin(Device &dev, VkCommandBuffer cmd, uint32_t &q0, bool &timed);

static void write_member(uint8_t *reg, uint32_t k, const uint8_t *pbuf, uint32_t psize, const VkCuLaunchInfoNVX *li,
                         uint32_t alt)
{
    uint8_t slot[256] = {};
    memcpy(slot, pbuf, std::min<uint32_t>(psize, 256));
    memcpy(reg + 256 + (size_t)k * 256, slot, 256);
    uint32_t gr[4] = {li->gridDimX, li->gridDimY, li->gridDimZ, 0};
    memcpy(reg + kGrpGridOff + (size_t)k * 16, gr, 16);
    uint32_t bi[4] = {li->blockDimX, li->blockDimY, li->blockDimZ, alt};
    memcpy(reg + kGrpInfoOff + (size_t)k * 16, bi, 16);
}

// index of the alternative of a group member ("a|b|...") that is gkey, or -1
static int member_alt(const std::string &member, const std::string &gkey)
{
    size_t p = 0;
    for (int i = 0;; i++) {
        size_t q = member.find('|', p);
        if (member.compare(p, (q == std::string::npos ? member.size() : q) - p, gkey) == 0 &&
            (q == std::string::npos ? member.size() : q) - p == gkey.size())
            return i;
        if (q == std::string::npos)
            return -1;
        p = q + 1;
    }
}

// returns true when the launch was taken by a launch group (recorded as the fused dispatch or absorbed into it)
static bool launch_group(Device &dev, VkCommandBuffer cmd, CuFunction &fn, const VkCuLaunchInfoNVX *li,
                         const uint8_t *pbuf, uint32_t psize)
{
    auto pit = dev.pending.find(cmd);
    if (pit != dev.pending.end()) {
        PendingGroup &pg = pit->second;
        int alt = pbuf && psize <= 256 ? member_alt(pg.g->members[pg.next], fn.gkey) : -1;
        if (alt >= 0 && dev.launch_count == pg.last_launch + 1) {
            write_member(pg.region, pg.next, pbuf, psize, li, (uint32_t)alt);
            pg.next++;
            pg.last_launch = dev.launch_count;
            *(volatile uint32_t *)pg.region = pg.next;           // members the fused kernel executes
            if (g_trace)
                logf("   group %s: member %u absorbed", pg.g->name.c_str(), pg.next - 1);
            if (pg.next == pg.g->members.size()) {
                dev.pending.erase(pit);
                pending_changed(dev);
            }
            return true;
        }
        logf("group %s: sequence broken after %u of %zu members by %s", pg.g->name.c_str(), pg.next,
             pg.g->members.size(), fn.name.c_str());
        dev.pending.erase(pit);
        pending_changed(dev);
    }
    auto git = dev.group_first.find(fn.gkey);
    if (git == dev.group_first.end() || !pbuf || psize > 256)
        return false;
    // (only when a group starts: these are uncached reads of host-visible VRAM)
    // a fused kernel whose grid barrier timed out (workgroups not co-resident) flags its region: disable the group
    for (auto it = dev.grp_recent.begin(); it != dev.grp_recent.end();) {
        volatile uint32_t *poison = (volatile uint32_t *)(dev.grp_map + (size_t)it->first * kGrpRegion + 64);
        if (*poison) {
            *poison = 0;
            Group *pg = it->second;
            bool was = false;
            for (auto fit = dev.group_first.begin(); fit != dev.group_first.end();) {
                if (fit->second == pg) { fit = dev.group_first.erase(fit); was = true; }
                else ++fit;
            }
            if (was)
                logf("group %s: grid barrier timed out or a fused kernel gave up: disabled, members launch singly",
                     pg->name.c_str());
        }
        ++it;
    }
    git = dev.group_first.find(fn.gkey);
    if (git == dev.group_first.end())
        return false;
    Group *g = git->second;
    VkPipeline pipe = get_pipeline(dev, *g->fn, PipeKey{g->block[0], g->block[1], g->block[2], 0});
    if (!pipe)
        return false;
    uint32_t ri = 0;
    if (!ring_take(dev, &Device::CmdSlots::grp, cmd, ri)) {
        if (!dev.grp_full_logged)
            logf("launch groups: all %u regions held by recorded command buffers; members launch singly", kGrpRegions);
        dev.grp_full_logged = true;
        return false;
    }
    uint8_t *reg = dev.grp_map + (size_t)ri * kGrpRegion;
    // NVXRT_GROUP_PROFILE=1: fused kernels may write 64-bit realtime stamps (100 MHz) at kGrpProfOff; the region's
    // previous use has finished (its command buffer was begun again, reset or freed): log them as deltas
    static const bool gprof = getenv("NVXRT_GROUP_PROFILE") && atoi(getenv("NVXRT_GROUP_PROFILE")) != 0;
    if (gprof) {
        const uint64_t *ts = (const uint64_t *)(reg + kGrpProfOff);
        std::string line;
        for (uint32_t i = 1; i < 128 && ts[0]; i++) {
            if (!ts[i])
                continue;
            char b[32];
            snprintf(b, sizeof b, " %u:%.2f", i, (double)(int64_t)(ts[i] - ts[0]) / 100.0);
            line += b;
        }
        if (!line.empty())
            logf("group profile %s (us):%s", g->name.c_str(), line.c_str());
        memset(reg + kGrpProfOff, 0, 1024);
    }
    memset(reg, 0, 256);                                          // count, grid-barrier counter
    write_member(reg, 0, pbuf, psize, li, (uint32_t)std::max(0, member_alt(g->members[0], fn.gkey)));
    *(volatile uint32_t *)reg = 1;
    if (g_serialize) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dev.d.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                                 0, nullptr, 0, nullptr);
    }
    uint32_t q0 = 0;
    bool timed = false;
    record_timed_dispatch_begin(dev, cmd, q0, timed);
    uint32_t pc[kPushBytes / 4] = {};
    VkDeviceAddress a = dev.grp_addr + (VkDeviceAddress)ri * kGrpRegion;
    pc[0] = (uint32_t)a;
    pc[1] = (uint32_t)(a >> 32);
    dev.d.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    dev.d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dev.pipe_layout, 0, 1, &dev.set, 0, nullptr);
    dev.d.CmdPushConstants(cmd, dev.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes, pc);
    dev.d.CmdDispatch(cmd, g->grid[0], g->grid[1], g->grid[2]);
    g->fn->launches++;
    if (timed) {
        dev.d.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, dev.qpool, q0 + 1);
        dev.qpending.push_back({g->fn, q0});
    }
    if (g_trace)
        logf("   group %s: fused dispatch (region %u), member 0 = %s", g->name.c_str(), ri, fn.name.c_str());
    dev.grp_recent.push_back({ri, g});
    if (dev.grp_recent.size() > 16)
        dev.grp_recent.pop_front();
    if (g->members.size() > 1) {
        dev.pending[cmd] = PendingGroup{g, 1, reg, dev.launch_count};
        pending_changed(dev);
    }
    return true;
}

// timing bookkeeping shared by plain launches and fused group dispatches: periodic summaries, query slots
static void record_timed_dispatch_begin(Device &dev, VkCommandBuffer cmd, uint32_t &q0, bool &timed)
{
    timed = dev.qpool != VK_NULL_HANDLE;
    q0 = 0;
    if (!timed)
        return;
    if ((dev.launch_count & 511) == 0)
        collect_timing(dev);
    static const double every = getenv("NVXRT_TIMING_EVERY") ? atof(getenv("NVXRT_TIMING_EVERY")) : 0;
    if (every > 0) {
        static auto last = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last).count() >= every) {
            print_summary(dev, true, "timing period");
            last = now;
        }
    }
    if (dev.qnext + 2 > kQueries)
        dev.qnext = 0;
    q0 = dev.qnext;
    dev.qnext += 2;
    dev.d.CmdResetQueryPool(cmd, dev.qpool, q0, 2);
    dev.d.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dev.qpool, q0);
}

static VKAPI_ATTR void VKAPI_CALL CmdCuLaunchKernelNVX(VkCommandBuffer cmd, const VkCuLaunchInfoNVX *li)
{
    auto dev = device_of(cmd);
    if (!dev || !li || !li->function)
        return;
    CuFunction &fn = *(CuFunction *)(uintptr_t)li->function;
    const uint8_t *pbuf = nullptr;
    uint32_t psize = 0;
    std::vector<uint8_t> packed;
    if (li->extraCount && li->pExtras) {
        const void *const *e = li->pExtras;
        while (*e != CU_LAUNCH_PARAM_END) {
            if (*e == CU_LAUNCH_PARAM_BUFFER_POINTER) {
                e++;
                pbuf = (const uint8_t *)*e;
            } else if (*e == CU_LAUNCH_PARAM_BUFFER_SIZE) {
                e++;
                psize = *(const uint32_t *)*e;
            }
            e++;
        }
    }
    if (!pbuf && li->paramCount && li->pParams) {
        // individual kernel parameters: lay them out per the translator's metadata
        packed.assign(std::max(kPushBytes, fn.meta.param_size), 0);
        for (size_t i = 0; i < li->paramCount && i < fn.meta.params.size(); i++)
            if ((uint64_t)fn.meta.params[i].first + fn.meta.params[i].second <= packed.size())
                memcpy(packed.data() + fn.meta.params[i].first, li->pParams[i], fn.meta.params[i].second);
        pbuf = packed.data();
        psize = fn.meta.param_size;
    }
    std::lock_guard<std::mutex> lk(dev->mutex);
    fn.launches++;
    dev->launch_count++;
    if (g_trace) {
        logf("launch #%" PRIu64 " %s[%.12s] grid %ux%ux%u block %ux%ux%u smem %u params %u%s", dev->launch_count,
             fn.name.c_str(), fn.module->hash.c_str(), li->gridDimX, li->gridDimY, li->gridDimZ, li->blockDimX, li->blockDimY, li->blockDimZ,
             li->sharedMemBytes, psize, pbuf ? decode_params(*dev, pbuf, psize).c_str() : "");
        if (pbuf && getenv("NVXRT_TRACE_RAW"))
            logf("   raw %s", hex(pbuf, psize).c_str());
    }
    if (!g_exec || !dev->bindless_ok)
        return;
    // NVXRT_SHADOW b/c: handles of shadowed images rewritten in a copy of the parameter block (any size the launch
    // path accepts: a launch the shadow tracking does not see could write an original or read a stale one)
    uint8_t shadow_pbuf[kBpSlot];
    if (pbuf && psize <= sizeof shadow_pbuf && (shadow_active(*dev) || g_shadow_log >= 2) &&
        shadow_params(*dev, cmd, fn, pbuf, psize, shadow_pbuf))
        pbuf = shadow_pbuf;
    if (g_queue_log)
        dev->cubin_cmds.insert(cmd);
    // NVXRT_SKIP=name[@hashprefix],...: drop these launches (cost experiments only; output is wrong)
    static const std::string skip = getenv("NVXRT_SKIP") ? std::string(",") + getenv("NVXRT_SKIP") + "," : "";
    if (!skip.empty() && (skip.find("," + fn.name + ",") != std::string::npos ||
                          skip.find("," + fn.name + "@" + fn.module->hash.substr(0, 12) + ",") != std::string::npos))
        return;
    if (!dev->groups.empty() && launch_group(*dev, cmd, fn, li, pbuf, psize))
        return;
    PipeKey k{li->blockDimX, li->blockDimY, li->blockDimZ, li->sharedMemBytes};
    uint32_t gx = li->gridDimX, gy = li->gridDimY, gz = li->gridDimZ;
    if (fn.meta.ok && fn.meta.t1_dispatch) {
        const KernelMeta &m = fn.meta;
        k = PipeKey{m.t1_block[0], m.t1_block[1], m.t1_block[2], 0};
        gx = (uint32_t)(((uint64_t)gx * m.t1_num[0] + m.t1_den[0] - 1) / m.t1_den[0]);
        gy = (uint32_t)(((uint64_t)gy * m.t1_num[1] + m.t1_den[1] - 1) / m.t1_den[1]);
        gz = (uint32_t)(((uint64_t)gz * m.t1_num[2] + m.t1_den[2] - 1) / m.t1_den[2]);
        if (m.t1_fixed[0] && m.t1_fixed[1] && m.t1_fixed[2])
            gx = m.t1_fixed[0], gy = m.t1_fixed[1], gz = m.t1_fixed[2];
    }
    VkPipeline pipe = get_pipeline(*dev, fn, k);
    if (!pipe) {
        if (!fn.warned_skip) {
            fn.warned_skip = true;
            logf("launch of %s skipped: no translated kernel/pipeline (module %s)", fn.name.c_str(),
                 fn.module->hash.substr(0, 16).c_str());
        }
        return;
    }
    if (g_serialize) {
        // stream order: everything earlier in this command buffer (any stage) is visible to this kernel
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dev->d.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                                  0, nullptr, 0, nullptr);
    }
    uint8_t pc[kPushBytes] = {};
    if (pbuf && fn.meta.big_params) {
        // the whole block into a ring slot; the kernel reads it through the address in pc words 60..61
        if (!dev->bp_map && !dev->bp_failed) {
            void *ptr = nullptr;
            uint32_t memtype = 0;
            dev->bp_failed = !host_ring(*dev, (VkDeviceSize)kBpSlots * kBpSlot, dev->bp_buf, dev->bp_mem, ptr,
                                        dev->bp_addr, memtype);
            dev->bp_map = dev->bp_failed ? nullptr : (uint8_t *)ptr;
            ring_init(dev->bp_free, kBpSlots);
            logf("big parameter ring: %s (%u x %u B, memory type %u)", dev->bp_failed ? "allocation FAILED" : "ok",
                 kBpSlots, kBpSlot, memtype);
        }
        if (!dev->bp_map || psize > kBpSlot)
            return;
        uint32_t si = 0;
        if (!ring_take(*dev, &Device::CmdSlots::bp, cmd, si)) {
            if (!dev->bp_full_logged)
                logf("big parameter ring: all %u slots held by recorded command buffers; launch of %s skipped",
                     kBpSlots, fn.name.c_str());
            dev->bp_full_logged = true;
            return;
        }
        memcpy(dev->bp_map + (size_t)si * kBpSlot, pbuf, psize);
        VkDeviceAddress a = dev->bp_addr + (VkDeviceAddress)si * kBpSlot;
        memcpy(pc + 240, &a, sizeof a);
    } else if (pbuf) {
        memcpy(pc, pbuf, std::min<uint32_t>(psize, 248));
    }
    // pc[252..255] stays zero: the opaque zero the translated code XORs into values the compiler must not fold
    bool timed = false;
    uint32_t q0 = 0;
    // read finished queries back now and then (non-blocking) so the pool can wrap without losing results
    record_timed_dispatch_begin(*dev, cmd, q0, timed);
    bool cap = !g_dump_dir.empty() && dev->dump_map && g_cap_launches.count(dev->launch_count);
    if (cap)
        capture_buffers(*dev, cmd, fn, pbuf, psize, li, "pre");
    dev->d.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    dev->d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dev->pipe_layout, 0, 1, &dev->set, 0, nullptr);
    dev->d.CmdPushConstants(cmd, dev->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes, pc);
    dev->d.CmdDispatch(cmd, gx, gy, gz);
    if (cap)
        capture_buffers(*dev, cmd, fn, pbuf, psize, li, "post");
    // NVXRT_DUMP_KERNELS=name,name limits the dumps to these kernels
    static const std::string dump_only = getenv("NVXRT_DUMP_KERNELS") ? std::string(",") + getenv("NVXRT_DUMP_KERNELS") + "," : "";
    bool dump_this = dump_only.empty() || dump_only.find("," + fn.name + ",") != std::string::npos;
    if (!g_dump_dir.empty() && pbuf && dev->dump_map && !cap && g_cap_launches.empty() && dump_this) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        dev->d.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                                  nullptr, 0, nullptr);
        for (uint32_t o = 0; o + 8 <= psize; o += 8) {
            uint64_t v;
            memcpy(&v, pbuf + o, 8);
            if (v >> 32) {
                auto it = dev->buffer_ranges.upper_bound(v);
                if (it == dev->buffer_ranges.begin())
                    continue;
                --it;
                if (v < it->first || v >= it->first + it->second.size)
                    continue;
                VkDeviceSize boff = v - it->first;
                VkDeviceSize len = std::min<VkDeviceSize>(it->second.size - boff, 64ull << 20);
                if (dev->dump_used + len > dev->dump_size)
                    continue;
                VkBufferCopy bc{boff, dev->dump_used, len};
                dev->d.CmdCopyBuffer(cmd, it->second.buffer, dev->dump_buf, 1, &bc);
                dev->dumps.push_back({dev->launch_count, fn.name, o, VK_FORMAT_UNDEFINED, (uint32_t)len, 1, 1,
                                      dev->dump_used, "", cmd});
                dev->dump_used += (len + 255) & ~(VkDeviceSize)255;
                continue;
            }
            if (!v || (v >> 20))
                continue;
            auto sv = dev->storage_slot_view.find((uint32_t)v);
            if (sv == dev->storage_slot_view.end())
                continue;
            auto vi = dev->views.find(sv->second);
            if (vi == dev->views.end())
                continue;
            auto ii = dev->images.find(vi->second.image);
            if (ii == dev->images.end())
                continue;
            uint32_t bpp = format_bpp(vi->second.format);
            VkDeviceSize need = (VkDeviceSize)ii->second.extent.width * ii->second.extent.height * bpp;
            if (!bpp || dev->dump_used + need > dev->dump_size)
                continue;
            VkBufferImageCopy r{};
            r.bufferOffset = dev->dump_used;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.imageExtent = {ii->second.extent.width, ii->second.extent.height, 1};
            dev->d.CmdCopyImageToBuffer(cmd, vi->second.image, VK_IMAGE_LAYOUT_GENERAL, dev->dump_buf, 1, &r);
            dev->dumps.push_back({dev->launch_count, fn.name, o, vi->second.format, ii->second.extent.width,
                                  ii->second.extent.height, bpp, dev->dump_used, "", cmd});
            dev->dump_used += (need + 255) & ~(VkDeviceSize)255;
        }
        mb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        dev->d.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                                  nullptr, 0, nullptr);
    }
    if (timed) {
        dev->d.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, dev->qpool, q0 + 1);
        dev->qpending.push_back({&fn, q0});

    }
}

static VKAPI_ATTR uint32_t VKAPI_CALL GetImageViewHandleNVX(VkDevice device, const VkImageViewHandleInfoNVX *info)
{
    auto dev = device_of(device);
    if (!dev || !info || !dev->bindless_ok)
        return 0;
    std::lock_guard<std::mutex> lk(dev->mutex);
    int binding = info->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? 1 : 0;
    uint32_t slot = view_slot(*dev, info->imageView, binding);
    if (slot)
        view_slot_cross(*dev, info->imageView, binding);
    uint32_t sslot = 0;
    if (info->sampler && (info->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
                          info->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER))
        sslot = sampler_slot(*dev, info->sampler);
    uint32_t h = slot | (sslot << 20);
    if (binding == 1)
        dev->storage_slot_view[slot] = info->imageView;
    else
        dev->tex_slot_view[slot] = info->imageView;
    if (g_trace) {
        auto vi = dev->views.find(info->imageView);
        unsigned ew = 0, eh = 0;
        if (vi != dev->views.end()) {
            auto ii = dev->images.find(vi->second.image);
            if (ii != dev->images.end())
                ew = ii->second.extent.width, eh = ii->second.extent.height;
        }
        char buf[128];
        snprintf(buf, sizeof buf, "%s%u%s%u:fmt%d:%ux%u", desc_type_name(info->descriptorType), slot, sslot ? "/s" : "",
                 sslot, vi != dev->views.end() ? (int)vi->second.format : -1, ew, eh);
        dev->handle_desc[h] = buf;
        logf("ImageViewHandle view %p %s -> 0x%x", (void *)info->imageView, buf, h);
    }
    return h;
}

static VKAPI_ATTR uint64_t VKAPI_CALL GetImageViewHandle64NVX(VkDevice device, const VkImageViewHandleInfoNVX *info)
{
    return GetImageViewHandleNVX(device, info);
}

static VKAPI_ATTR VkResult VKAPI_CALL GetImageViewAddressNVX(VkDevice, VkImageView view,
                                                             VkImageViewAddressPropertiesNVX *props)
{
    // vkd3d-proton asks for this for every UAV created after NvAPI_D3D12_CaptureUAVInfo (DLSS-G's surface capture);
    // translated kernels never address image memory directly, so report "no address" instead of failing (a failure
    // leaves vkd3d's capture pointer armed and every later UAV creation writes into the caller's stale struct).
    static std::atomic<int> once{0};
    if (!once.fetch_add(1))
        logf("GetImageViewAddressNVX(%p): images have no device address here, reporting 0", (void *)view);
    if (props) {
        props->deviceAddress = 0;
        props->size = 0;
    }
    return VK_SUCCESS;
}

static VKAPI_ATTR uint64_t VKAPI_CALL GetDeviceCombinedImageSamplerIndexNVX(VkDevice, uint64_t image, uint64_t smp)
{
    logf("GetDeviceCombinedImageSamplerIndexNVX(%" PRIu64 ", %" PRIu64 ") - descriptor-heap mode unsupported", image,
         smp);
    return 0;
}

// ---------------------------------------------------------------------------------------------- tracked core calls
static VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(VkDevice device, const VkImageViewCreateInfo *ci,
                                                      const VkAllocationCallbacks *a, VkImageView *out)
{
    auto dev = device_of(device);
    VkResult r = dev->d.CreateImageView(device, ci, a, out);
    if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        ViewInfo v{ci->image, ci->format, ci->viewType};
        for (auto *st = (const VkBaseInStructure *)ci->pNext; st; st = st->pNext)
            if (st->sType == VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO)
                v.usage = ((const VkImageViewUsageCreateInfo *)st)->usage;
        v.range = ci->subresourceRange;
        if (v.range.levelCount == VK_REMAINING_MIP_LEVELS || v.range.layerCount == VK_REMAINING_ARRAY_LAYERS) {
            auto ii = dev->images.find(ci->image);
            if (ii != dev->images.end()) {
                if (v.range.levelCount == VK_REMAINING_MIP_LEVELS) v.range.levelCount = ii->second.mips - v.range.baseMipLevel;
                if (v.range.layerCount == VK_REMAINING_ARRAY_LAYERS) v.range.layerCount = ii->second.layers - v.range.baseArrayLayer;
            }
        }
        dev->views[*out] = v;
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks *a)
{
    auto dev = device_of(device);
    if (view) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        release_view_slots(*dev, view);
        dev->views.erase(view);
    }
    dev->d.DestroyImageView(device, view, a);
}

static VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice device, const VkImageCreateInfo *ci,
                                                  const VkAllocationCallbacks *a, VkImage *out)
{
    auto dev = device_of(device);
    VkResult r = dev->d.CreateImage(device, ci, a, out);
    if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        ImageInfo ii{ci->format, ci->extent, ci->usage};
        ii.type = ci->imageType;
        ii.mips = ci->mipLevels;
        ii.layers = ci->arrayLayers;
        ii.samples = ci->samples;
        dev->images[*out] = ii;
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice device, VkImage img, const VkAllocationCallbacks *a)
{
    auto dev = device_of(device);
    if (img) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        dev->images.erase(img);
        destroy_shadow(*dev, img);
    }
    dev->d.DestroyImage(device, img, a);
}

static uint32_t format_bpp(VkFormat f)
{
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: return 1;
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R8G8_UNORM: return 2;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R16G16_UINT: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
    default: return 0;
    }
}

static void flush_dumps(Device &dev)
{
    if (dev.dumps.empty())
        return;
    mkdirs(g_dump_dir);
    FILE *man = fopen((g_dump_dir + "/manifest.txt").c_str(), "a");
    for (auto &d : dev.dumps) {
        char name[512];
        if (!d.tag.empty()) {
            std::string dir = g_dump_dir + "/L" + std::to_string(d.launch);
            mkdirs(dir);
            snprintf(name, sizeof name, "%s/%s.bin", dir.c_str(), d.tag.c_str());
            FILE *f = fopen(name, "wb");
            if (f) {
                fwrite(dev.dump_map + d.off, 1, (size_t)d.w * d.h * d.bpp, f);
                fclose(f);
            }
            continue;
        }
        snprintf(name, sizeof name, "%s/L%05llu_%s_p%u_f%d_%ux%u.bin", g_dump_dir.c_str(), (unsigned long long)d.launch,
                 d.kernel.c_str(), d.poff, (int)d.fmt, d.w, d.h);
        FILE *f = fopen(name, "wb");
        if (f) {
            fwrite(dev.dump_map + d.off, 1, (size_t)d.w * d.h * d.bpp, f);
            fclose(f);
        }
        if (man)
            fprintf(man, "%llu %s %u %d %u %u %u %s\n", (unsigned long long)d.launch, d.kernel.c_str(), d.poff, (int)d.fmt,
                    d.w, d.h, d.bpp, name);
    }
    if (man)
        fclose(man);
    dev.dumps.clear();
    dev.dump_used = 0;
}

static VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice device, uint32_t n, const VkFence *f, VkBool32 all,
                                                    uint64_t timeout)
{
    auto dev = device_of(device);
    VkResult r = dev->d.WaitForFences(device, n, f, all, timeout);
    if (r == VK_SUCCESS && (!g_dump_dir.empty() || dev->qpool)) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        flush_dumps(*dev);
        collect_timing(*dev);
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(VkDevice device, const VkSamplerCreateInfo *ci,
                                                    const VkAllocationCallbacks *a, VkSampler *out)
{
    auto dev = device_of(device);
    VkResult r = dev->d.CreateSampler(device, ci, a, out);
    if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        dev->samplers[*out] = {ci->magFilter, ci->minFilter, ci->addressModeU, ci->addressModeV, ci->borderColor,
                               ci->unnormalizedCoordinates};
        if (g_trace)
            logf("CreateSampler %p: mag %d min %d mip %d addr %d/%d/%d border %d unnorm %d maxlod %g aniso %d", (void *)*out,
                 ci->magFilter, ci->minFilter, ci->mipmapMode, ci->addressModeU, ci->addressModeV, ci->addressModeW,
                 ci->borderColor, ci->unnormalizedCoordinates, ci->maxLod, ci->anisotropyEnable);
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL DestroySampler(VkDevice device, VkSampler s, const VkAllocationCallbacks *a)
{
    auto dev = device_of(device);
    if (s) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        dev->samplers.erase(s);
        auto it = dev->sampler_slot.find(s);
        if (it != dev->sampler_slot.end()) {
            dev->free_samplers.push_back(it->second);
            dev->sampler_slot.erase(it);
        }
    }
    dev->d.DestroySampler(device, s, a);
}

static VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(VkDevice device, const VkBufferCreateInfo *ci,
                                                   const VkAllocationCallbacks *a, VkBuffer *out)
{
    auto dev = device_of(device);
    VkResult r = dev->d.CreateBuffer(device, ci, a, out);
    if (r == VK_SUCCESS && (ci->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) && (g_trace || !g_dump_dir.empty())) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        dev->buffer_sizes[*out] = ci->size;
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL DestroyBuffer(VkDevice device, VkBuffer b, const VkAllocationCallbacks *a)
{
    auto dev = device_of(device);
    if (b && (g_trace || !g_dump_dir.empty())) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        dev->buffer_sizes.erase(b);
        for (auto it = dev->buffer_ranges.begin(); it != dev->buffer_ranges.end();) {
            if (it->second.buffer == b)
                it = dev->buffer_ranges.erase(it);
            else
                ++it;
        }
    }
    dev->d.DestroyBuffer(device, b, a);
}

template <class F>
static VkDeviceAddress track_bda(Device &dev, const VkBufferDeviceAddressInfo *info, VkDeviceAddress a)
{
    if ((g_trace || !g_dump_dir.empty()) && a) {
        std::lock_guard<std::mutex> lk(dev.mutex);
        auto it = dev.buffer_sizes.find(info->buffer);
        if (it != dev.buffer_sizes.end())
            dev.buffer_ranges[a] = {info->buffer, it->second};
    }
    return a;
}

static VKAPI_ATTR VkDeviceAddress VKAPI_CALL GetBufferDeviceAddress(VkDevice device,
                                                                    const VkBufferDeviceAddressInfo *info)
{
    auto dev = device_of(device);
    return track_bda<int>(*dev, info, dev->d.GetBufferDeviceAddress(device, info));
}
static VKAPI_ATTR VkDeviceAddress VKAPI_CALL GetBufferDeviceAddressKHR(VkDevice device,
                                                                       const VkBufferDeviceAddressInfo *info)
{
    auto dev = device_of(device);
    auto f = dev->d.GetBufferDeviceAddressKHR ? dev->d.GetBufferDeviceAddressKHR : dev->d.GetBufferDeviceAddress;
    return track_bda<int>(*dev, info, f(device, info));
}
static VKAPI_ATTR VkDeviceAddress VKAPI_CALL GetBufferDeviceAddressEXT(VkDevice device,
                                                                       const VkBufferDeviceAddressInfo *info)
{
    auto dev = device_of(device);
    auto f = dev->d.GetBufferDeviceAddressEXT ? dev->d.GetBufferDeviceAddressEXT : dev->d.GetBufferDeviceAddress;
    return track_bda<int>(*dev, info, f(device, info));
}

// capture mode under vkd3d (which waits on timeline semaphores, not fences): once every command buffer holding pending
// dump copies has been submitted, wait for the queue and write the dumps out
static void flush_after_submit(Device &dev, VkQueue q, const std::vector<VkCommandBuffer> &cmds)
{
    if (g_dump_dir.empty())
        return;
    std::lock_guard<std::mutex> lk(dev.mutex);
    if (dev.dumps.empty())
        return;
    for (auto c : cmds)
        dev.dump_submitted.insert(c);
    for (auto &d : dev.dumps)
        if (d.cmd != VK_NULL_HANDLE && !dev.dump_submitted.count(d.cmd))
            return;
    dev.d.QueueWaitIdle(q);
    flush_dumps(dev);
    dev.dump_submitted.clear();
}

static VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo *s, VkFence f)
{
    auto dev = device_of(q);
    if (g_queue_log) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        for (uint32_t i = 0; i < n; i++) {
            bool cubin = false;
            for (uint32_t j = 0; j < s[i].commandBufferCount; j++)
                cubin |= dev->cubin_cmds.erase(s[i].pCommandBuffers[j]) > 0;
            uint64_t &c = dev->cubin_queue_submits[q];
            if (cubin && (c++ % 2000) == 0)
                logf("queue log: cubin submit (v1) #%llu on VkQueue %p, %u waits", (unsigned long long)c, (void *)q,
                     s[i].waitSemaphoreCount);
        }
    }
    VkResult r = dev->d.QueueSubmit(q, n, s, f);
    if (r == VK_SUCCESS && !g_dump_dir.empty()) {
        std::vector<VkCommandBuffer> cmds;
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < s[i].commandBufferCount; j++)
                cmds.push_back(s[i].pCommandBuffers[j]);
        flush_after_submit(*dev, q, cmds);
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2 *s, VkFence f)
{
    auto dev = device_of(q);
    if (g_queue_log) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        for (uint32_t i = 0; i < n; i++) {
            bool cubin = false;
            for (uint32_t j = 0; j < s[i].commandBufferInfoCount; j++)
                cubin |= dev->cubin_cmds.erase(s[i].pCommandBufferInfos[j].commandBuffer) > 0;
            uint64_t &c = dev->cubin_queue_submits[q];
            if (cubin && (c++ % 500) == 0) {
                std::string w;
                for (uint32_t k = 0; k < s[i].waitSemaphoreInfoCount; k++) {
                    char b[96];
                    snprintf(b, sizeof b, " [sem %p val %llu stage 0x%llx]", (void *)s[i].pWaitSemaphoreInfos[k].semaphore,
                             (unsigned long long)s[i].pWaitSemaphoreInfos[k].value,
                             (unsigned long long)s[i].pWaitSemaphoreInfos[k].stageMask);
                    w += b;
                }
                auto qf = dev->queue_family.find(q);
                logf("queue log: cubin submit #%llu on VkQueue %p (family %d index %d), %u command buffers, waits:%s",
                     (unsigned long long)c, (void *)q, qf != dev->queue_family.end() ? (int)qf->second.first : -1,
                     qf != dev->queue_family.end() ? (int)qf->second.second : -1, s[i].commandBufferInfoCount, w.c_str());
            }
        }
    }
    VkResult r = dev->d.QueueSubmit2(q, n, s, f);
    if (r == VK_SUCCESS && !g_dump_dir.empty()) {
        std::vector<VkCommandBuffer> cmds;
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < s[i].commandBufferInfoCount; j++)
                cmds.push_back(s[i].pCommandBufferInfos[j].commandBuffer);
        flush_after_submit(*dev, q, cmds);
    }
    return r;
}

// ---------------------------------------------------------------------------------------------- instance level
static const VkExtensionProperties kOurExts[] = {
    {VK_NVX_BINARY_IMPORT_EXTENSION_NAME, 2},
    {VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME, 3},
};

static VkLayerInstanceCreateInfo *find_instance_link(const VkInstanceCreateInfo *ci)
{
    auto *l = (VkLayerInstanceCreateInfo *)ci->pNext;
    while (l && !(l->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && l->function == VK_LAYER_LINK_INFO))
        l = (VkLayerInstanceCreateInfo *)l->pNext;
    return l;
}

static VkLayerDeviceCreateInfo *find_device_link(const VkDeviceCreateInfo *ci)
{
    auto *l = (VkLayerDeviceCreateInfo *)ci->pNext;
    while (l && !(l->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && l->function == VK_LAYER_LINK_INFO))
        l = (VkLayerDeviceCreateInfo *)l->pNext;
    return l;
}

static VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *a,
                                                     VkInstance *out)
{
    init_globals();
    VkLayerInstanceCreateInfo *link = find_instance_link(ci);
    if (!link)
        return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    auto create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult r = create(ci, a, out);
    if (r != VK_SUCCESS)
        return r;
    auto inst = std::make_shared<Instance>();
    inst->handle = *out;
    inst->d.GetInstanceProcAddr = gipa;
#define GI(n) inst->d.n = (PFN_vk##n)gipa(*out, "vk" #n)
    GI(DestroyInstance);
    GI(EnumerateDeviceExtensionProperties);
    GI(GetPhysicalDeviceProperties);
    GI(GetPhysicalDeviceProperties2);
    GI(GetPhysicalDeviceFeatures2);
    GI(GetPhysicalDeviceMemoryProperties);
    GI(GetPhysicalDeviceFormatProperties);
#undef GI
    std::lock_guard<std::mutex> lk(g_map_mutex);
    g_instances[dkey(*out)] = inst;
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks *a)
{
    auto inst = instance_of(instance);
    if (!inst)
        return;
    inst->d.DestroyInstance(instance, a);
    std::lock_guard<std::mutex> lk(g_map_mutex);
    g_instances.erase(dkey(instance));
}

// The translated kernels build their WMMA operands in one GPU generation's register layout (wave32): gfx12 (RDNA4)
// or gfx11 (RDNA3), as the kernel cache's "arch" file records (dlssg-setup NVXRT_ARCH; no file: gfx12). On any other
// GPU, or with a cache built for the other generation, the layer stays inert: it advertises nothing, so vkd3d-proton
// keeps its cubin path off and DLSS-G reports itself unavailable instead of running kernels with the wrong layout.
// NVXRT_ANY_GPU=1 skips the check (development only).
static const char *gpu_generation(const char *name)
{
    // RADV names: "... (RADV GFX1201)", "(RADV NAVI32)", "(RADV GFX1103_R1)", "(RADV GFX1150)"
    if (strstr(name, "RADV GFX12"))
        return "gfx12";
    if (strstr(name, "RADV NAVI3") || strstr(name, "RADV GFX110") || strstr(name, "RADV GFX115"))
        return "gfx11";
    return nullptr;
}

static std::string cache_generation()
{
    init_globals();
    std::ifstream f(g_cache_dir + "/arch");
    std::string a;
    if (!(f >> a))
        a = "gfx12";
    return a;
}

static bool gpu_supported(Instance &inst, VkPhysicalDevice pd)
{
    static std::mutex m;
    static std::map<VkPhysicalDevice, bool> seen;
    std::lock_guard<std::mutex> lk(m);
    auto it = seen.find(pd);
    if (it != seen.end())
        return it->second;
    VkPhysicalDeviceProperties props{};
    inst.d.GetPhysicalDeviceProperties(pd, &props);
    bool any = getenv("NVXRT_ANY_GPU") && atoi(getenv("NVXRT_ANY_GPU")) != 0;
    const char *gen = props.vendorID == 0x1002 ? gpu_generation(props.deviceName) : nullptr;
    std::string cache = cache_generation();
    bool ok = any || (gen && cache == gen);
    if (props.vendorID == 0x1002)
        logf("%s: %s", props.deviceName,
             ok ? "VK_NVX_binary_import emulated" :
             !gen ? "not an RDNA3/RDNA4 RADV device: NVX emulation off (DLSS frame generation unavailable)" :
                    "the kernel cache was built for another GPU generation: NVX emulation off (run d4r-features.py "
                    "--fg dlssg again)");
    if (ok && gen)
        logf("kernel cache %s: %s kernels", g_cache_dir.c_str(), cache.c_str());
    seen[pd] = ok;
    return ok;
}

static VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice pd, const char *layer,
                                                                         uint32_t *count, VkExtensionProperties *props)
{
    auto inst = instance_of_pd(pd);
    if (!layer && inst && !gpu_supported(*inst, pd))
        return inst->d.EnumerateDeviceExtensionProperties(pd, layer, count, props);
    if (layer && strcmp(layer, "VK_LAYER_NVXRT_nvx") == 0) {
        uint32_t n = 2;
        if (!props) {
            *count = n;
            return VK_SUCCESS;
        }
        uint32_t c = std::min(*count, n);
        memcpy(props, kOurExts, c * sizeof(VkExtensionProperties));
        *count = c;
        return c < n ? VK_INCOMPLETE : VK_SUCCESS;
    }
    if (!inst)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (layer)
        return inst->d.EnumerateDeviceExtensionProperties(pd, layer, count, props);
    uint32_t n = 0;
    VkResult r = inst->d.EnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    if (r != VK_SUCCESS)
        return r;
    std::vector<VkExtensionProperties> all(n);
    inst->d.EnumerateDeviceExtensionProperties(pd, nullptr, &n, all.data());
    all.resize(n);
    for (auto &e : kOurExts) {
        bool have = false;
        for (auto &x : all)
            if (strcmp(x.extensionName, e.extensionName) == 0) {
                have = true;
                x.specVersion = std::max(x.specVersion, e.specVersion);
            }
        if (!have)
            all.push_back(e);
    }
    if (!props) {
        *count = (uint32_t)all.size();
        return VK_SUCCESS;
    }
    uint32_t c = std::min<uint32_t>(*count, (uint32_t)all.size());
    memcpy(props, all.data(), c * sizeof(VkExtensionProperties));
    *count = c;
    return c < all.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

// Feature structs we need enabled for translated kernels. If the app chained a struct we flip the bits in
// place; otherwise our own copy is prepended to the chain.
struct FeatureAdds {
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR wg{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
};

static void want_v11(VkPhysicalDeviceVulkan11Features &f) { f.storageBuffer16BitAccess = VK_TRUE; }
static void want_v12(VkPhysicalDeviceVulkan12Features &f)
{
    f.shaderFloat16 = VK_TRUE;
    f.shaderInt8 = VK_TRUE;
    f.storageBuffer8BitAccess = VK_TRUE;
    f.bufferDeviceAddress = VK_TRUE;
    f.descriptorIndexing = VK_TRUE;
    f.runtimeDescriptorArray = VK_TRUE;
    f.descriptorBindingPartiallyBound = VK_TRUE;
    f.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    f.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    f.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    f.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    f.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    f.shaderBufferInt64Atomics = VK_TRUE;
    f.shaderSharedInt64Atomics = VK_TRUE;
    f.scalarBlockLayout = VK_TRUE;
    f.vulkanMemoryModel = VK_TRUE;
    f.vulkanMemoryModelDeviceScope = VK_TRUE;
    f.hostQueryReset = VK_TRUE;
}
static void want_v13(VkPhysicalDeviceVulkan13Features &f)
{
    f.subgroupSizeControl = VK_TRUE;
    f.computeFullSubgroups = VK_TRUE;
    f.shaderIntegerDotProduct = VK_TRUE;
}
static void want_core(VkPhysicalDeviceFeatures &f)
{
    f.shaderInt64 = VK_TRUE;
    f.shaderInt16 = VK_TRUE;
    f.shaderStorageImageReadWithoutFormat = VK_TRUE;
    f.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    f.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    f.shaderStorageImageArrayDynamicIndexing = VK_TRUE;
}

static VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci,
                                                   const VkAllocationCallbacks *a, VkDevice *out)
{
    init_globals();
    auto inst = instance_of_pd(pd);
    VkLayerDeviceCreateInfo *link = find_device_link(ci);
    if (!inst || !link)
        return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    auto create = (PFN_vkCreateDevice)gipa(inst->handle, "vkCreateDevice");

    // extensions: drop ours, add what translated kernels need
    bool want_nvx = false;
    std::vector<const char *> exts;
    for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) {
        const char *e = ci->ppEnabledExtensionNames[i];
        if (!strcmp(e, VK_NVX_BINARY_IMPORT_EXTENSION_NAME) || !strcmp(e, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME)) {
            want_nvx = true;
            continue;
        }
        exts.push_back(e);
    }
    uint32_t n = 0;
    inst->d.EnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> avail(n);
    inst->d.EnumerateDeviceExtensionProperties(pd, nullptr, &n, avail.data());
    auto supported = [&](const char *name) {
        for (auto &x : avail)
            if (!strcmp(x.extensionName, name))
                return true;
        return false;
    };
    auto add_ext = [&](const char *name) {
        for (auto *e : exts)
            if (!strcmp(e, name))
                return;
        if (supported(name))
            exts.push_back(name);
    };
    const uint32_t napp = (uint32_t)exts.size();   // the app's extensions without the NVX ones
    VkPhysicalDeviceProperties props;
    inst->d.GetPhysicalDeviceProperties(pd, &props);
    bool v13 = props.apiVersion >= VK_API_VERSION_1_3 &&
               (!ci->pNext || true); // the app's requested API version is not visible here; RADV is 1.4
    if (want_nvx) {
        add_ext(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        add_ext(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
        add_ext(VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME);
        if (!v13)
            add_ext(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }

    VkDeviceCreateInfo nci = *ci;
    nci.enabledExtensionCount = (uint32_t)exts.size();
    nci.ppEnabledExtensionNames = exts.data();
    auto adds = std::make_unique<FeatureAdds>();
    if (want_nvx) {
        bool has11 = false, has12 = false, has13 = false, hascm = false, haswg = false, hasf2 = false;
        // The app's own feature structs get the bits we need flipped in place for the driver call and are put back
        // afterwards, so the app (vkd3d keeps these structs as its record of what it enabled) sees no change.
        std::vector<std::pair<VkBaseOutStructure *, std::vector<uint8_t>>> saved;
        auto save = [&](VkBaseOutStructure *st, size_t size) {
            const uint8_t *b = (const uint8_t *)st;
            saved.push_back({st, std::vector<uint8_t>(b, b + size)});
        };
        for (auto *s = (VkBaseOutStructure *)ci->pNext; s; s = s->pNext) {
            switch (s->sType) {
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: save(s, sizeof(VkPhysicalDeviceVulkan11Features)); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: save(s, sizeof(VkPhysicalDeviceVulkan12Features)); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: save(s, sizeof(VkPhysicalDeviceVulkan13Features)); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR:
                save(s, sizeof(VkPhysicalDeviceCooperativeMatrixFeaturesKHR));
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR:
                save(s, sizeof(VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR));
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: save(s, sizeof(VkPhysicalDeviceFeatures2)); break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES:
                save(s, sizeof(VkPhysicalDeviceSubgroupSizeControlFeatures));
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES:
                save(s, sizeof(VkPhysicalDeviceBufferDeviceAddressFeatures));
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES:
                save(s, sizeof(VkPhysicalDevice16BitStorageFeatures));
                break;
            default: break;
            }
        }
        auto restore = [&]() {
            for (auto &sv : saved) {
                void *next = sv.first->pNext; // keep the chain as it is now
                memcpy(sv.first, sv.second.data(), sv.second.size());
                sv.first->pNext = (VkBaseOutStructure *)next;
            }
        };
        for (auto *s = (VkBaseOutStructure *)ci->pNext; s; s = s->pNext) {
            switch (s->sType) {
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
                want_v11(*(VkPhysicalDeviceVulkan11Features *)s);
                has11 = true;
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
                want_v12(*(VkPhysicalDeviceVulkan12Features *)s);
                has12 = true;
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
                want_v13(*(VkPhysicalDeviceVulkan13Features *)s);
                has13 = true;
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR:
                ((VkPhysicalDeviceCooperativeMatrixFeaturesKHR *)s)->cooperativeMatrix = VK_TRUE;
                hascm = true;
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR: {
                auto *w = (VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR *)s;
                w->workgroupMemoryExplicitLayout = w->workgroupMemoryExplicitLayout8BitAccess =
                    w->workgroupMemoryExplicitLayout16BitAccess = w->workgroupMemoryExplicitLayoutScalarBlockLayout =
                        VK_TRUE;
                haswg = true;
                break;
            }
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:
                want_core(((VkPhysicalDeviceFeatures2 *)s)->features);
                hasf2 = true;
                break;
            // individual (pre-1.2 style) structs: flip their bits if present
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES:
                ((VkPhysicalDeviceSubgroupSizeControlFeatures *)s)->subgroupSizeControl = VK_TRUE;
                ((VkPhysicalDeviceSubgroupSizeControlFeatures *)s)->computeFullSubgroups = VK_TRUE;
                has13 = true; // promoted struct present: cannot also chain 1.3
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES:
                ((VkPhysicalDeviceBufferDeviceAddressFeatures *)s)->bufferDeviceAddress = VK_TRUE;
                has12 = true;
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES:
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES:
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES:
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES:
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES:
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES:
                has12 = true; // individual struct: chaining Vulkan12Features too would be invalid
                logf("CreateDevice: app uses individual 1.2 feature structs; not all needed bits may be enabled");
                break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES:
                ((VkPhysicalDevice16BitStorageFeatures *)s)->storageBuffer16BitAccess = VK_TRUE;
                has11 = true;
                break;
            default:
                break;
            }
        }
        const void *head = nci.pNext;
        auto prepend = [&](void *s) {
            ((VkBaseOutStructure *)s)->pNext = (VkBaseOutStructure *)head;
            head = s;
        };
        if (!has11) {
            want_v11(adds->v11);
            prepend(&adds->v11);
        }
        if (!has12) {
            want_v12(adds->v12);
            prepend(&adds->v12);
        }
        if (!has13) {
            want_v13(adds->v13);
            prepend(&adds->v13);
        }
        if (!hascm && supported(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
            adds->cm.cooperativeMatrix = VK_TRUE;
            prepend(&adds->cm);
        }
        if (!haswg && supported(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME)) {
            adds->wg.workgroupMemoryExplicitLayout = adds->wg.workgroupMemoryExplicitLayout8BitAccess =
                adds->wg.workgroupMemoryExplicitLayout16BitAccess = adds->wg.workgroupMemoryExplicitLayoutScalarBlockLayout =
                    VK_TRUE;
            prepend(&adds->wg);
        }
        VkPhysicalDeviceFeatures core{};
        if (!hasf2) {
            if (ci->pEnabledFeatures) {
                core = *ci->pEnabledFeatures;
                want_core(core);
                nci.pEnabledFeatures = &core;
            } else {
                want_core(adds->f2.features);
                prepend(&adds->f2);
            }
        }
        nci.pNext = head;
        VkResult r = create(pd, &nci, a, out);
        // our prepended structs point into the app's chain; the app's chain itself was not relinked
        restore();
        if (r != VK_SUCCESS) {
            logf("CreateDevice with NVX support failed (%d); retrying without extras", r);
            VkDeviceCreateInfo bci = *ci;   // the driver does not know the NVX extensions
            bci.enabledExtensionCount = napp;
            bci.ppEnabledExtensionNames = exts.data();
            r = create(pd, &bci, a, out);
            if (r != VK_SUCCESS)
                return r;
            // still registered below: GetDeviceProcAddr needs the Device to pass calls through
            want_nvx = false;
        }
    } else {
        VkResult r = create(pd, &nci, a, out);
        if (r != VK_SUCCESS)
            return r;
    }

    auto dev = std::make_shared<Device>();
    dev->handle = *out;
    dev->pd = pd;
    dev->inst = inst;
    dev->props = props;
    inst->d.GetPhysicalDeviceMemoryProperties(pd, &dev->memprops);
#define X(n) dev->d.n = (PFN_vk##n)gdpa(*out, "vk" #n);
    DEVICE_FUNCS(X)
#undef X
    dev->d.GetDeviceProcAddr = gdpa;
    dev->d.GetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)gdpa(*out, "vkGetBufferDeviceAddressKHR");
    dev->d.GetBufferDeviceAddressEXT = (PFN_vkGetBufferDeviceAddressEXT)gdpa(*out, "vkGetBufferDeviceAddressEXT");
    if (getenv("NVXRT_QUEUE_LOG") && atoi(getenv("NVXRT_QUEUE_LOG")) != 0 && dev->d.GetDeviceQueue)
        for (uint32_t i = 0; i < ci->queueCreateInfoCount; i++)
            for (uint32_t j = 0; j < ci->pQueueCreateInfos[i].queueCount; j++) {
                if (ci->pQueueCreateInfos[i].flags)
                    continue;
                VkQueue q = VK_NULL_HANDLE;
                dev->d.GetDeviceQueue(*out, ci->pQueueCreateInfos[i].queueFamilyIndex, j, &q);
                if (q)
                    dev->queue_family[q] = {ci->pQueueCreateInfos[i].queueFamilyIndex, j};
            }
    if (want_nvx) {
        dev->bindless_ok = init_bindless(*dev);
        if (dev->bindless_ok)
            init_groups(*dev);
        logf("device %s: NVX emulation %s (cache %s, exec %d, trace %d)", props.deviceName,
             dev->bindless_ok ? "ready" : "FAILED", g_cache_dir.c_str(), g_exec, g_trace);
    }
    std::lock_guard<std::mutex> lk(g_map_mutex);
    g_devices[dkey(*out)] = dev;
    g_dev_gen.fetch_add(1, std::memory_order_release);
    return VK_SUCCESS;
}

// per-kernel GPU time summary (NVXRT_TIMING=1); reset=true starts a new period (NVXRT_TIMING_EVERY=seconds)
static void print_summary(Device &dev, bool reset, const char *what)
{
    collect_timing(dev);
    std::vector<CuFunction *> fs;
    for (auto &f : dev.functions)
        fs.push_back(f.get());
    std::sort(fs.begin(), fs.end(), [](CuFunction *x, CuFunction *y) { return x->gpu_ms_total > y->gpu_ms_total; });
    double tot = 0;
    for (auto *f : fs)
        tot += f->gpu_ms_total;
    logf("%s: %.3f ms GPU in launches", what, tot);
    for (auto *f : fs)
        if (f->gpu_samples)
            logf("summary %-44s launches %8" PRIu64 " gpu %9.3f ms total %7.4f ms avg %.12s", f->name.c_str(),
                 f->gpu_samples, f->gpu_ms_total, f->gpu_ms_total / f->gpu_samples, f->module->hash.c_str());
    if (reset)
        for (auto *f : fs)
            f->gpu_ms_total = 0, f->gpu_samples = 0;
}

static VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks *a)
{
    auto dev = device_of(device);
    if (!dev)
        return;
    {
        std::lock_guard<std::mutex> lk(dev->mutex);
        if (dev->bindless_ok) {
            if (getenv("NVXRT_TIMING_DEBUG"))
                logf("destroy: qpool %p pending %zu", (void *)dev->qpool, dev->qpending.size());
            if (dev->qpool)
                print_summary(*dev, false, "timing at device destroy");
            save_pipeline_cache(*dev);
            for (auto &f : dev->functions) {
                for (auto &p : f->pipelines)
                    if (p.second)
                        dev->d.DestroyPipeline(device, p.second, nullptr);
                if (f->shader)
                    dev->d.DestroyShaderModule(device, f->shader, nullptr);
            }
            if (dev->pcache)
                dev->d.DestroyPipelineCache(device, dev->pcache, nullptr);
            while (!dev->shadows.empty())
                destroy_shadow(*dev, dev->shadows.begin()->first);
            if (dev->flag_buf)
                dev->d.DestroyBuffer(device, dev->flag_buf, nullptr);
            if (dev->flag_mem)
                dev->d.FreeMemory(device, dev->flag_mem, nullptr);
            if (dev->qpool)
                dev->d.DestroyQueryPool(device, dev->qpool, nullptr);
            dev->d.DestroyDescriptorPool(device, dev->pool, nullptr);
            for (auto &ns : dev->nearest_samplers)
                if (ns.second)
                    dev->d.DestroySampler(device, ns.second, nullptr);
            dev->d.DestroyPipelineLayout(device, dev->pipe_layout, nullptr);
            dev->d.DestroyDescriptorSetLayout(device, dev->set_layout, nullptr);
            dev->d.DestroyBuffer(device, dev->smp_buf, nullptr);
            dev->d.FreeMemory(device, dev->smp_mem, nullptr);
            // host-visible rings and the dump buffer (freeing the memory unmaps it)
            for (VkBuffer b : {dev->grp_buf, dev->bp_buf, dev->dump_buf})
                if (b)
                    dev->d.DestroyBuffer(device, b, nullptr);
            for (VkDeviceMemory m : {dev->grp_mem, dev->bp_mem, dev->dump_mem})
                if (m)
                    dev->d.FreeMemory(device, m, nullptr);
            // companions of views the app never destroyed
            for (auto &c : dev->companion)
                dev->d.DestroyImageView(device, c.second, nullptr);
            dev->companion.clear();
        }
    }
    dev->d.DestroyDevice(device, a);
    std::lock_guard<std::mutex> lk(g_map_mutex);
    g_devices.erase(dkey(device));
    g_dev_gen.fetch_add(1, std::memory_order_release);
}

// ---------------------------------------------------------------------------------------------- command buffers
// Ring slot ownership (Device::cmd_slots): these are the calls after which a command buffer can no longer execute
// what it recorded. The slots are released before the call goes down, so a handle the driver hands out again
// at once starts clean.
static VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo *ai,
                                                             VkCommandBuffer *out)
{
    auto dev = device_of(device);
    VkResult r = dev->d.AllocateCommandBuffers(device, ai, out);
    if (r == VK_SUCCESS) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        for (uint32_t i = 0; i < ai->commandBufferCount; i++)
            dev->cmd_pool[out[i]] = ai->commandPool;
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t n,
                                                     const VkCommandBuffer *cmds)
{
    auto dev = device_of(device);
    {
        std::lock_guard<std::mutex> lk(dev->mutex);
        for (uint32_t i = 0; i < n; i++)
            if (cmds[i]) {
                cmd_release(*dev, cmds[i]);
                dev->cmd_pool.erase(cmds[i]);
            }
    }
    dev->d.FreeCommandBuffers(device, pool, n, cmds);
}

// cmd_release for every command buffer of pool that holds slots or an open group. caller holds dev.mutex
static void pool_release(Device &dev, VkCommandPool pool)
{
    std::vector<VkCommandBuffer> rel;
    for (auto &c : dev.cmd_slots)
        rel.push_back(c.first);
    for (auto &c : dev.pending)
        rel.push_back(c.first);
    for (VkCommandBuffer c : rel) {
        auto it = dev.cmd_pool.find(c);
        if (it != dev.cmd_pool.end() && it->second == pool)
            cmd_release(dev, c);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags f)
{
    auto dev = device_of(device);
    if (dev->n_cmd_slots.load(std::memory_order_relaxed) || dev->n_pending.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        pool_release(*dev, pool);
    }
    return dev->d.ResetCommandPool(device, pool, f);
}

static VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice device, VkCommandPool pool, const VkAllocationCallbacks *a)
{
    auto dev = device_of(device);
    {
        std::lock_guard<std::mutex> lk(dev->mutex);
        pool_release(*dev, pool);
        for (auto it = dev->cmd_pool.begin(); it != dev->cmd_pool.end();)
            it = it->second == pool ? dev->cmd_pool.erase(it) : std::next(it);
    }
    dev->d.DestroyCommandPool(device, pool, a);
}

static void cmd_reset(Device &dev, VkCommandBuffer cmd)
{
    if (!dev.n_cmd_slots.load(std::memory_order_relaxed) && !dev.n_pending.load(std::memory_order_relaxed))
        return;
    std::lock_guard<std::mutex> lk(dev.mutex);
    cmd_release(dev, cmd);
}

static VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer cmd, const VkCommandBufferBeginInfo *bi)
{
    Device *dev = device_fast(cmd);
    cmd_reset(*dev, cmd);
    return dev->d.BeginCommandBuffer(cmd, bi);
}

// a launch group cannot continue past the end of its command buffer: closing it here keeps the hooks below on their
// lock-free path while no group is being recorded
static VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer cmd)
{
    Device *dev = device_fast(cmd);
    if (dev->n_pending.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lk(dev->mutex);
        if (dev->pending.erase(cmd))
            pending_changed(*dev);
    }
    return dev->d.EndCommandBuffer(cmd);
}

static VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer cmd, VkCommandBufferResetFlags f)
{
    Device *dev = device_fast(cmd);
    cmd_reset(*dev, cmd);
    return dev->d.ResetCommandBuffer(cmd, f);
}

// Commands that record work other than a cubin launch end an open launch group in their command buffer
// (group_break). The patched vkd3d (0002) invalidates all of its state after a cubin launch, so whatever it records
// next binds a pipeline first; render pass begins, transfers, clears and acceleration structure commands need none.
template <typename P, P DeviceDispatch::*M, const char *Name> struct Brk;
template <typename... A, void (VKAPI_PTR *DeviceDispatch::*M)(VkCommandBuffer, A...), const char *Name>
struct Brk<void (VKAPI_PTR *)(VkCommandBuffer, A...), M, Name> {
    static VKAPI_ATTR void VKAPI_CALL f(VkCommandBuffer cmd, A... a)
    {
        Device *dev = device_fast(cmd);
        group_break(*dev, cmd, Name);
        (dev->d.*M)(cmd, a...);
    }
};
#define BRK_FUNCS(X)                                                                                               \
    X(CmdBindPipeline) X(CmdBeginRenderPass) X(CmdBeginRenderPass2) X(CmdBeginRendering) X(CmdCopyBuffer)           \
    X(CmdCopyBuffer2) X(CmdFillBuffer) X(CmdUpdateBuffer) X(CmdClearDepthStencilImage) X(CmdCopyQueryPoolResults)   \
    X(CmdExecuteCommands) X(CmdBuildAccelerationStructuresKHR) X(CmdBuildAccelerationStructuresIndirectKHR)          \
    X(CmdCopyAccelerationStructureKHR) X(CmdCopyAccelerationStructureToMemoryKHR)                                  \
    X(CmdCopyMemoryToAccelerationStructureKHR)
// the image commands NVXRT_SHADOW hooks anyway (those hooks call group_break themselves)
#define BRK_IMAGE_FUNCS(X)                                                                                         \
    X(CmdCopyImage) X(CmdCopyImage2) X(CmdBlitImage) X(CmdBlitImage2) X(CmdResolveImage) X(CmdResolveImage2)        \
    X(CmdClearColorImage) X(CmdCopyImageToBuffer) X(CmdCopyImageToBuffer2) X(CmdCopyBufferToImage)                 \
    X(CmdCopyBufferToImage2)
#define X(n) static constexpr char kBrk##n[] = "vk" #n;
BRK_FUNCS(X)
BRK_IMAGE_FUNCS(X)
#undef X

static PFN_vkVoidFunction brk_func(Device &dev, const char *name)
{
    // core 1.2/1.3 commands under their extension names
    static const char *const kAlias[][2] = {
        {"vkCmdBeginRenderPass2KHR", "vkCmdBeginRenderPass2"}, {"vkCmdBeginRenderingKHR", "vkCmdBeginRendering"},
        {"vkCmdCopyBuffer2KHR", "vkCmdCopyBuffer2"}, {"vkCmdCopyImage2KHR", "vkCmdCopyImage2"},
        {"vkCmdBlitImage2KHR", "vkCmdBlitImage2"}, {"vkCmdResolveImage2KHR", "vkCmdResolveImage2"},
        {"vkCmdCopyImageToBuffer2KHR", "vkCmdCopyImageToBuffer2"},
        {"vkCmdCopyBufferToImage2KHR", "vkCmdCopyBufferToImage2"},
    };
    for (auto &al : kAlias)
        if (!strcmp(name, al[0]))
            name = al[1];
#define X(n)                                                                                                       \
    if (!strcmp(name, "vk" #n))                                                                                    \
        return dev.d.n ? (PFN_vkVoidFunction)Brk<PFN_vk##n, &DeviceDispatch::n, kBrk##n>::f : nullptr;
    BRK_FUNCS(X)
    if (!g_shadow) {
        BRK_IMAGE_FUNCS(X)
    }
#undef X
    return nullptr;
}

// ---------------------------------------------------------------------------------------------- proc addr
static PFN_vkVoidFunction device_func(const char *name)
{
#define F(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)n;
    F(CreateCuModuleNVX) F(CreateCuFunctionNVX) F(DestroyCuModuleNVX) F(DestroyCuFunctionNVX) F(CmdCuLaunchKernelNVX)
    F(GetImageViewHandleNVX) F(GetImageViewHandle64NVX) F(GetImageViewAddressNVX)
    F(GetDeviceCombinedImageSamplerIndexNVX) F(CreateImageView) F(DestroyImageView) F(CreateSampler) F(DestroySampler)
    F(CreateBuffer) F(DestroyBuffer) F(GetBufferDeviceAddress) F(GetBufferDeviceAddressKHR) F(GetBufferDeviceAddressEXT)
    F(DestroyDevice) F(QueueSubmit) F(QueueSubmit2) F(WaitForFences) F(CreateImage) F(DestroyImage)
#undef F
    if (!strcmp(name, "vkQueueSubmit2KHR"))
        return (PFN_vkVoidFunction)QueueSubmit2;
    if (g_shadow) {
        // NVXRT_SHADOW b/c: copies / clears / blits that may name a shadowed original
#define FS(n, f) if (!strcmp(name, n)) return (PFN_vkVoidFunction)f;
        FS("vkCmdCopyImage", CmdCopyImage) FS("vkCmdCopyImage2", CmdCopyImage2) FS("vkCmdCopyImage2KHR", CmdCopyImage2)
        FS("vkCmdBlitImage", CmdBlitImage) FS("vkCmdBlitImage2", CmdBlitImage2) FS("vkCmdBlitImage2KHR", CmdBlitImage2)
        FS("vkCmdResolveImage", CmdResolveImage) FS("vkCmdResolveImage2", CmdResolveImage2)
        FS("vkCmdResolveImage2KHR", CmdResolveImage2) FS("vkCmdClearColorImage", CmdClearColorImage)
        FS("vkCmdCopyImageToBuffer", CmdCopyImageToBufferS) FS("vkCmdCopyImageToBuffer2", CmdCopyImageToBuffer2)
        FS("vkCmdCopyImageToBuffer2KHR", CmdCopyImageToBuffer2) FS("vkCmdCopyBufferToImage", CmdCopyBufferToImage)
        FS("vkCmdCopyBufferToImage2", CmdCopyBufferToImage2) FS("vkCmdCopyBufferToImage2KHR", CmdCopyBufferToImage2)
#undef FS
    }
    return nullptr;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char *name);

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char *name)
{
    if (!strcmp(name, "vkGetInstanceProcAddr"))
        return (PFN_vkVoidFunction)GetInstanceProcAddr;
    if (!strcmp(name, "vkCreateInstance"))
        return (PFN_vkVoidFunction)CreateInstance;
    if (!strcmp(name, "vkDestroyInstance"))
        return (PFN_vkVoidFunction)DestroyInstance;
    if (!strcmp(name, "vkCreateDevice"))
        return (PFN_vkVoidFunction)CreateDevice;
    if (!strcmp(name, "vkEnumerateDeviceExtensionProperties"))
        return (PFN_vkVoidFunction)EnumerateDeviceExtensionProperties;
    if (!strcmp(name, "vkGetDeviceProcAddr"))
        return (PFN_vkVoidFunction)GetDeviceProcAddr;
    if (auto f = device_func(name))
        return f;
    if (!instance)
        return nullptr;
    auto inst = instance_of(instance);
    return inst ? inst->d.GetInstanceProcAddr(instance, name) : nullptr;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char *name)
{
    if (!strcmp(name, "vkGetDeviceProcAddr"))
        return (PFN_vkVoidFunction)GetDeviceProcAddr;
    auto dev = device_of(device);
    if (!dev)
        return nullptr;
    if (dev->bindless_ok)
        if (auto f = device_func(name))
            return f;
    if (dev->bindless_ok) {
#define F(n) if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)n;
        F(AllocateCommandBuffers) F(FreeCommandBuffers) F(ResetCommandPool) F(DestroyCommandPool)
        F(BeginCommandBuffer) F(EndCommandBuffer) F(ResetCommandBuffer)
#undef F
        if (!dev->groups.empty())
            if (auto f = brk_func(*dev, name))
                return f;
    }
    // tracked core calls are needed even before NVX use (views created early)
    if (!strcmp(name, "vkCreateImageView")) return (PFN_vkVoidFunction)CreateImageView;
    if (!strcmp(name, "vkDestroyImageView")) return (PFN_vkVoidFunction)DestroyImageView;
    if (!strcmp(name, "vkCreateSampler")) return (PFN_vkVoidFunction)CreateSampler;
    if (!strcmp(name, "vkDestroySampler")) return (PFN_vkVoidFunction)DestroySampler;
    if (!strcmp(name, "vkDestroyDevice")) return (PFN_vkVoidFunction)DestroyDevice;
    return dev->d.GetDeviceProcAddr(device, name);
}

} // namespace nvxrt

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance i, const char *n)
{
    return nvxrt::GetInstanceProcAddr(i, n);
}
EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice d, const char *n)
{
    return nvxrt::GetDeviceProcAddr(d, n);
}
EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *v)
{
    if (v->loaderLayerInterfaceVersion > 2)
        v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr = nvxrt::GetInstanceProcAddr;
    v->pfnGetDeviceProcAddr = nvxrt::GetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}
