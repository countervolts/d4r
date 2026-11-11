// cubin_d3d12: NVIDIA's D3D12 cubin interface for nvngx_dlssg.dll on native Windows AMD drivers.
//
// Under Proton, DLSS-G's CUDA kernels reach the GPU through dxvk-nvapi -> vkd3d-proton -> the nvxrt Vulkan layer. On
// native Windows there is no vkd3d: the game's D3D12 device is AMD's own. This file answers the NVAPI calls DLSS-G
// makes (NvAPI_D3D12_CreateCubinComputeShader*, LaunchCubinShader, GetCudaTextureObject, ...) itself and runs the
// translated kernels (the same SPIR-V kernel cache the layer uses) on a private Vulkan device on the same adapter:
//
//  - CUDA pointers are D3D12 GPU virtual addresses. WDDM gives a process one GPU address space per adapter, so a
//    Vulkan shader dereferences them as buffer device addresses unchanged.
//  - Texture / surface objects are slots of a VK_EXT_descriptor_buffer set that hold the raw bytes of the D3D12
//    descriptors (AMD's D3D12 CPU descriptors are the hardware descriptors; the Vulkan driver uses the same sizes).
//    Handles keep the layer's encoding: slot | sampler slot << 20, slots from 32768.
//  - A launch is recorded into a Vulkan command buffer. The game's D3D12 command list is cut there: what is recorded
//    into it afterwards goes into a further list this file owns (the vtable redirection of tools/d4r_d3d12_split.h,
//    generalised to any number of cuts), and ExecuteCommandLists runs the parts in order with a D3D12 fence (shared,
//    imported as a Vulkan timeline semaphore) between each D3D12 part and the kernels after it. Launches with nothing
//    recorded in between share one command buffer; shader-only barriers, UAV clears (run as Vulkan dispatches,
//    clear_uav.comp), timestamps and markers in between do not end it (each hand-off idles the GPU ~0.15 ms on RDNA3,
//    docs/hillclimb/journal.md "FG"). The descriptor heaps, root signatures, pipeline state and root
//    arguments set during DLSS-G's evaluate are replayed into every new part (a cubin launch keeps D3D12 state).
//
// Kernels, T1 launch groups and big parameter blocks follow nvxrt.cpp (nvxrt/), which documents them.
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <d3d12.h>
#include <vulkan/vulkan.h>

#include "cubin_d3d12.h"
#include "../nvxrt/util.h"
#include "clear_uav_spv.h"
#include "copy_res_spv.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using nvxrt::FatbinEntry;

namespace cux {
namespace {

// ------------------------------------------------------------------------------------------------ config, logging
Config g_cfg;
std::string g_cache;   // UTF-8, '/' separators
HANDLE g_log = INVALID_HANDLE_VALUE;
std::mutex g_logLock;

std::string utf8(const std::wstring& w)
{
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
std::wstring wide(const std::string& s)
{
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void logf(const char* fmt, ...)
{
    if (g_log == INVALID_HANDLE_VALUE) return;
    char buf[4096];
    SYSTEMTIME t; GetLocalTime(&t);
    int n = snprintf(buf, sizeof buf, "[%02d:%02d:%02d.%03d tid %lu] ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds,
                     GetCurrentThreadId());
    va_list ap; va_start(ap, fmt); n += vsnprintf(buf + n, sizeof buf - n - 2, fmt, ap); va_end(ap);
    if (n > (int)sizeof buf - 2) n = sizeof buf - 2;
    buf[n++] = '\n';
    std::lock_guard<std::mutex> l(g_logLock);
    DWORD w; WriteFile(g_log, buf, n, &w, nullptr);
}

bool readFile(const std::string& path, std::vector<uint8_t>& out)
{
    HANDLE f = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz; GetFileSizeEx(f, &sz);
    out.resize((size_t)sz.QuadPart);
    DWORD got = 0;
    bool ok = out.empty() || (ReadFile(f, out.data(), (DWORD)out.size(), &got, nullptr) && got == out.size());
    CloseHandle(f);
    return ok;
}

// ------------------------------------------------------------------------------------------------ Vulkan entry points
#define VK_INSTANCE_FUNCS(X)                                                                                           \
    X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) X(GetPhysicalDeviceProperties2)      \
    X(GetPhysicalDeviceFeatures2) X(GetPhysicalDeviceMemoryProperties) X(GetPhysicalDeviceQueueFamilyProperties)       \
    X(EnumerateDeviceExtensionProperties) X(CreateDevice) X(GetDeviceProcAddr)
struct Vk {
    HMODULE lib = nullptr;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance CreateInstance = nullptr;
#define X(n) PFN_vk##n n = nullptr;
    VK_INSTANCE_FUNCS(X)
#undef X
#define X(n) PFN_vk##n n = nullptr;
    X(GetDeviceQueue) X(DeviceWaitIdle) X(QueueSubmit) X(CreateBuffer) X(GetBufferMemoryRequirements)
    X(AllocateMemory) X(BindBufferMemory) X(MapMemory) X(GetBufferDeviceAddress) X(CreateDescriptorSetLayout)
    X(CreatePipelineLayout) X(CreateShaderModule) X(CreateComputePipelines) X(CreatePipelineCache)
    X(GetPipelineCacheData) X(CreateSampler) X(CreateCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer)
    X(EndCommandBuffer) X(ResetCommandBuffer) X(CmdBindPipeline) X(CmdPushConstants) X(CmdDispatch)
    X(CmdPipelineBarrier) X(CreateSemaphore) X(ImportSemaphoreWin32HandleKHR) X(CreateFence) X(WaitForFences)
    X(ResetFences) X(GetDescriptorSetLayoutSizeEXT) X(GetDescriptorSetLayoutBindingOffsetEXT) X(GetDescriptorEXT)
    X(CmdBindDescriptorBuffersEXT) X(CmdSetDescriptorBufferOffsetsEXT) X(GetSemaphoreCounterValue)                    \
    X(CreateQueryPool) X(CmdResetQueryPool) X(CmdWriteTimestamp) X(GetQueryPoolResults) X(GetCalibratedTimestampsEXT)
#undef X
} vk;

// ------------------------------------------------------------------------------------------------ formats
// DXGI -> VkFormat for the formats the translated kernels distinguish (format class, raw layout for suld.b)
VkFormat vkFormat(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
    case DXGI_FORMAT_R8G8_UNORM: return VK_FORMAT_R8G8_UNORM;
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R8_SNORM: return VK_FORMAT_R8_SNORM;
    case DXGI_FORMAT_R8G8_SNORM: return VK_FORMAT_R8G8_SNORM;
    case DXGI_FORMAT_R8G8B8A8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case DXGI_FORMAT_R8_UINT: return VK_FORMAT_R8_UINT;
    case DXGI_FORMAT_R8G8_UINT: return VK_FORMAT_R8G8_UINT;
    case DXGI_FORMAT_R8G8B8A8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case DXGI_FORMAT_R8_SINT: return VK_FORMAT_R8_SINT;
    case DXGI_FORMAT_R8G8_SINT: return VK_FORMAT_R8G8_SINT;
    case DXGI_FORMAT_R8G8B8A8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case DXGI_FORMAT_R16_UNORM: return VK_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16G16_UNORM: return VK_FORMAT_R16G16_UNORM;
    case DXGI_FORMAT_R16G16B16A16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case DXGI_FORMAT_R16_SNORM: return VK_FORMAT_R16_SNORM;
    case DXGI_FORMAT_R16G16_SNORM: return VK_FORMAT_R16G16_SNORM;
    case DXGI_FORMAT_R16G16B16A16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case DXGI_FORMAT_R16_FLOAT: return VK_FORMAT_R16_SFLOAT;
    case DXGI_FORMAT_R16G16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case DXGI_FORMAT_R16_UINT: return VK_FORMAT_R16_UINT;
    case DXGI_FORMAT_R16G16_UINT: return VK_FORMAT_R16G16_UINT;
    case DXGI_FORMAT_R16G16B16A16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case DXGI_FORMAT_R16_SINT: return VK_FORMAT_R16_SINT;
    case DXGI_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_SINT;
    case DXGI_FORMAT_R16G16B16A16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        return VK_FORMAT_R32_SFLOAT;
    case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case DXGI_FORMAT_R32_UINT: return VK_FORMAT_R32_UINT;
    case DXGI_FORMAT_R32G32_UINT: return VK_FORMAT_R32G32_UINT;
    case DXGI_FORMAT_R32G32B32A32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
    case DXGI_FORMAT_R32_SINT: return VK_FORMAT_R32_SINT;
    case DXGI_FORMAT_R32G32_SINT: return VK_FORMAT_R32G32_SINT;
    case DXGI_FORMAT_R32G32B32A32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case DXGI_FORMAT_R10G10B10A2_UINT: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case DXGI_FORMAT_R11G11B10_FLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return VK_FORMAT_X8_D24_UNORM_PACK32;
    default: return VK_FORMAT_UNDEFINED;
    }
}

// as nvxrt.cpp: raw layout for suld.b (kind | bytes per component << 4 | components << 8 | BGRA << 12)
uint32_t formatRaw(VkFormat f)
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

uint32_t formatClass(VkFormat f)
{
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32A32_UINT:
        return 1;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT:
        return 2;
    default:
        return 0;
    }
}

// ------------------------------------------------------------------------------------------------ kernels (as nvxrt)
struct KernelMeta {
    bool ok = false;
    uint32_t param_size = 0;
    std::vector<std::pair<uint32_t, uint32_t>> params;
    bool t1_dispatch = false;
    uint32_t t1_block[3] = {0, 0, 0};
    uint32_t t1_num[3] = {1, 1, 1}, t1_den[3] = {1, 1, 1};
    uint32_t t1_fixed[3] = {0, 0, 0};
    bool big_params = false;
};

struct CuModule {
    std::string hash;
    std::vector<std::string> entries;
};

struct PipeKey {
    uint32_t bx, by, bz, smem;
    bool operator<(const PipeKey& o) const { return std::tie(bx, by, bz, smem) < std::tie(o.bx, o.by, o.bz, o.smem); }
};

struct CuFunction {
    CuModule* module = nullptr;
    std::string name, gkey;
    KernelMeta meta;
    VkShaderModule shader = VK_NULL_HANDLE;
    std::map<PipeKey, VkPipeline> pipelines;
    uint64_t launches = 0;
    bool warned = false;
};

// an NvAPI_D3D12_CreateCubinComputeShader* object: a function with its block shape
struct Shader {
    CuFunction* fn = nullptr;
    uint32_t block[3] = {1, 1, 1};
    uint32_t smem = 0;
};

static const uint32_t kMaxImages = 1u << 16, kMaxSamplers = 4096, kSlotBase = 32768;
static const uint32_t kPushBytes = 256;
static const uint32_t kTsQueries = 16384;   // [Host] KernelTiming
static const uint32_t kInfoBytes = kMaxSamplers * 16 + kMaxImages * 4 * 2;
static const uint32_t kBpSlot = 4096, kBpSlots = 1024;
static const uint32_t kGrpMaxMembers = 48;
static const uint32_t kGrpGridOff = 256 + kGrpMaxMembers * 256;
static const uint32_t kGrpInfoOff = kGrpGridOff + kGrpMaxMembers * 16;
static const uint32_t kGrpRegion = 16384, kGrpRegions = 512;

struct Group {
    std::string name;
    std::vector<std::string> members;
    uint32_t block[3] = {0, 0, 0}, grid[3] = {1, 1, 1};
    CuFunction* fn = nullptr;
};
struct PendingGroup {
    Group* g = nullptr;
    uint32_t next = 0;
    uint8_t* region = nullptr;
    uint64_t last_launch = 0;
};

// what a CPU descriptor handle that DLSS-G created holds (recorded by the device hooks)
struct ViewRec {
    ID3D12Resource* res = nullptr;
    bool uav = false, hasDesc = false;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
};

// one D3D12 queue's hand-off fence (shared, imported into Vulkan as a timeline semaphore)
struct QueueSync {
    ID3D12Fence* fence = nullptr;
    VkSemaphore sem = VK_NULL_HANDLE;
    uint64_t value = 0;
    bool gpu = false;
    HANDLE event = nullptr;
};

// the kernels launched between two D3D12 parts of one command list
struct Batch {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool ended = false;
    uint32_t launches = 0;
    QueueSync* qs = nullptr;   // set once submitted
    uint64_t done = 0;         // qs->fence value after which the command buffer may be reused
    std::vector<uint32_t> ts;  // [Host] KernelTiming: timestamp queries, the first after the batch's entry barrier
    std::vector<const std::string*> tsLabel;   // what ran before each further timestamp
    int d3dTs = -1;            // [Host] KernelTiming: D3D12 timestamps (pair) around the hand-off on the game's queue
    ID3D12CommandQueue* d3dQ = nullptr;
};

struct Part {
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    QueueSync* qs = nullptr;
    uint64_t done = 0;
};

// ------------------------------------------------------------------------------------------------ executor state
struct Exec {
    bool tried = false, ok = false;
    ID3D12Device* d3d = nullptr;
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties memprops{};
    bool semImport = false;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipelineCache pcache = VK_NULL_HANDLE;
    VkBuffer dbuf = VK_NULL_HANDLE;
    VkDeviceMemory dmem = VK_NULL_HANDLE;
    uint8_t* dmap = nullptr;
    VkDeviceAddress daddr = 0;
    VkDeviceSize boff[6] = {};
    size_t imgSize = 32, smpSize = 16;
    uint32_t* info = nullptr;
    VkDeviceAddress infoAddr = 0, flagAddr = 0;

    ID3D12DescriptorHeap* scratchHeap = nullptr;   // companion SRV/UAV descriptors
    UINT scratchInc = 0, scratchNext = 0;

    std::map<std::string, uint32_t> viewSlots;     // descriptor bytes -> slot
    uint32_t nextSlot = kSlotBase;
    std::map<std::string, uint32_t> samplerSlots;
    uint32_t nextSampler = 1;
    std::map<uint32_t, VkSampler> nearest;

    std::vector<std::unique_ptr<CuModule>> modules;
    std::vector<std::unique_ptr<CuFunction>> functions;
    std::vector<std::unique_ptr<Shader>> shaders;
    uint64_t launchCount = 0;

    // launch groups / big parameters (nvxrt)
    std::vector<std::unique_ptr<Group>> groups;
    std::unordered_map<std::string, Group*> groupFirst;
    std::unordered_map<VkCommandBuffer, PendingGroup> pending;
    std::deque<std::pair<uint32_t, Group*>> grpRecent;
    uint8_t* grpMap = nullptr;
    VkDeviceAddress grpAddr = 0;
    uint8_t* bpMap = nullptr;
    VkDeviceAddress bpAddr = 0;
    bool bpFailed = false;
    std::vector<uint32_t> grpFree, bpFree;
    struct CmdSlots { std::vector<uint32_t> grp, bp; };
    std::unordered_map<VkCommandBuffer, CmdSlots> cmdSlots;
    std::unique_ptr<CuModule> grpModule;

    VkCommandPool pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> freeCmds;
    std::vector<Batch*> inflightBatches;
    std::vector<Part> freeParts, inflightParts;
    std::map<ID3D12CommandQueue*, QueueSync> syncs;
    VkFence cpuFence = VK_NULL_HANDLE;
    uint64_t batchesSubmitted = 0, partsCreated = 0;
    VkShaderModule clearShader = VK_NULL_HANDLE;
    VkPipeline clearPipe = VK_NULL_HANDLE;
    bool clearFailed = false;
    VkShaderModule copyShader = VK_NULL_HANDLE;
    VkPipeline copyPipe = VK_NULL_HANDLE;
    bool copyFailed = false;
    uint64_t vkCopies = 0;
    uint64_t vkClears = 0;
    // [Host] KernelTiming
    VkQueryPool tsPool = VK_NULL_HANDLE;
    uint32_t tsNext = 0;
    double tsPeriodMs = 0;
    std::map<std::string, std::pair<double, uint64_t>> ktime;   // label -> (ms, dispatches)
    double batchMs = 0, gapMs = 0;
    uint64_t timedBatches = 0;
    Batch* cur = nullptr;      // the batch launch() records into (launchGroup's timestamps)
    ID3D12QueryHeap* d3dTsHeap = nullptr;
    ID3D12Resource* d3dTsRb = nullptr;
    uint64_t* d3dTsMap = nullptr;
    uint32_t d3dTsNext = 0;
    double d3dSpanMs = 0, vkSpanMs = 0, lat1Ms = 0, lat2Ms = 0;
    uint64_t latN = 0;
    uint64_t d3dTimed = 0;
    std::map<ID3D12CommandQueue*, double> d3dTsMsPerTick;
};
Exec g;
std::recursive_mutex g_lock;

std::mutex g_viewLock;
std::unordered_map<SIZE_T, ViewRec> g_views;
std::unordered_map<SIZE_T, D3D12_SAMPLER_DESC> g_samplers;
thread_local int t_fg = 0;

uint32_t findMemtype(uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < g.memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (g.memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// a host-visible (VRAM if possible), coherent, mapped buffer with a device address
bool hostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, void*& ptr, VkDeviceAddress& addr, VkBuffer* out = nullptr)
{
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer buf = VK_NULL_HANDLE;
    if (vk.CreateBuffer(g.dev, &bci, nullptr, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    vk.GetBufferMemoryRequirements(g.dev, buf, &mr);
    VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = findMemtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX)
        mai.memoryTypeIndex = findMemtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (mai.memoryTypeIndex == UINT32_MAX || vk.AllocateMemory(g.dev, &mai, nullptr, &mem) != VK_SUCCESS ||
        vk.BindBufferMemory(g.dev, buf, mem, 0) != VK_SUCCESS || vk.MapMemory(g.dev, mem, 0, VK_WHOLE_SIZE, 0, &ptr) != VK_SUCCESS)
        return false;
    VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    ai.buffer = buf;
    addr = vk.GetBufferDeviceAddress(g.dev, &ai);
    if (out) *out = buf;
    return addr != 0;
}

void writeDescriptor(VkDescriptorType type, const void* data, uint32_t binding, uint32_t index)
{
    VkDescriptorGetInfoEXT gi{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    gi.type = type;
    size_t size = 16;
    VkDescriptorAddressInfoEXT ai{VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT};
    if (type == VK_DESCRIPTOR_TYPE_SAMPLER) {
        gi.data.pSampler = (const VkSampler*)data;
        size = g.smpSize;
    } else {
        const VkDeviceSize* a = (const VkDeviceSize*)data;
        ai.address = a[0];
        ai.range = a[1];
        gi.data.pStorageBuffer = &ai;
    }
    vk.GetDescriptorEXT(g.dev, &gi, size, g.dmap + g.boff[binding] + (size_t)index * size);
}

VkSampler nearestCompanion(VkSamplerAddressMode u, VkSamplerAddressMode v)
{
    auto norm = [](VkSamplerAddressMode m) {
        return m == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ? m : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    };
    u = norm(u);
    v = norm(v);
    uint32_t key = (uint32_t)u | ((uint32_t)v << 8);
    auto it = g.nearest.find(key);
    if (it != g.nearest.end()) return it->second;
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = ci.minFilter = VK_FILTER_NEAREST;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = u;
    ci.addressModeV = v;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    ci.unnormalizedCoordinates = VK_TRUE;
    VkSampler s = VK_NULL_HANDLE;
    if (vk.CreateSampler(g.dev, &ci, nullptr, &s) != VK_SUCCESS) s = VK_NULL_HANDLE;
    g.nearest[key] = s;
    return s;
}

// ------------------------------------------------------------------------------------------------ kernel loading
bool loadKernel(CuFunction& fn)
{
    KernelMeta& m = fn.meta;
    std::string base = g_cache + "/kernels/" + fn.module->hash + "/" + fn.name;
    std::vector<uint8_t> spv, meta;
    bool fa = false, hw = false;
    if (g_cfg.accFast && g_cfg.texHw && readFile(base + ".fa.hw.spv", spv) && spv.size() >= 20) fa = hw = true;
    else if (g_cfg.accFast && readFile(base + ".fa.spv", spv) && spv.size() >= 20) fa = true;
    else if (g_cfg.texHw && readFile(base + ".hw.spv", spv) && spv.size() >= 20) hw = true;
    else if (!readFile(base + ".spv", spv) || spv.size() < 20) spv.clear();
    if (spv.size() < 20 || !readFile(base + ".meta", meta)) {
        logf("kernel %s (%.16s): no translation in the cache (%s.spv)", fn.name.c_str(), fn.module->hash.c_str(),
             base.c_str());
        return false;
    }
    std::string text(meta.begin(), meta.end());
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        char key[64] = {};
        if (sscanf(line.c_str(), "%63s", key) != 1) continue;
        const char* rest = line.c_str() + strlen(key);
        unsigned a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
        if (!strcmp(key, "param_size") && sscanf(rest, "%u", &a) == 1) m.param_size = a;
        else if (!strcmp(key, "param") && sscanf(rest, "%u %u", &a, &b) == 2) m.params.push_back({a, b});
        else if (!strcmp(key, "t1_block") && sscanf(rest, "%u %u %u", &a, &b, &c) == 3) {
            m.t1_block[0] = a, m.t1_block[1] = b, m.t1_block[2] = c;
            m.t1_dispatch = a && b && c;
        } else if (!strcmp(key, "big_params")) m.big_params = true;
        else if (!strcmp(key, "t1_fixed") && sscanf(rest, "%u %u %u", &a, &b, &c) == 3)
            m.t1_fixed[0] = a, m.t1_fixed[1] = b, m.t1_fixed[2] = c;
        else if (!strcmp(key, "t1_grid") && sscanf(rest, "%u %u %u %u %u %u", &a, &b, &c, &d, &e, &f) == 6) {
            m.t1_num[0] = a, m.t1_den[0] = b ? b : 1, m.t1_num[1] = c, m.t1_den[1] = d ? d : 1;
            m.t1_num[2] = e, m.t1_den[2] = f ? f : 1;
        }
    }
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize = spv.size() & ~(size_t)3;
    sci.pCode = (const uint32_t*)spv.data();
    if (vk.CreateShaderModule(g.dev, &sci, nullptr, &fn.shader) != VK_SUCCESS) {
        logf("kernel %s: vkCreateShaderModule failed", fn.name.c_str());
        return false;
    }
    if (fa || hw || m.t1_dispatch)
        logf("kernel %s:%s%s%s", fn.name.c_str(), hw ? " texture-unit filtering" : "", fa ? " fp32 accumulation" : "",
             m.t1_dispatch ? " T1" : "");
    m.ok = true;
    return true;
}

VkPipeline getPipeline(CuFunction& fn, const PipeKey& k)
{
    auto it = fn.pipelines.find(k);
    if (it != fn.pipelines.end()) return it->second;
    VkPipeline p = VK_NULL_HANDLE;
    if (fn.meta.ok) {
        uint32_t spec[5] = {k.bx, k.by, k.bz, k.smem, (k.bx * k.by * k.bz + 31) / 32};
        VkSpecializationMapEntry me[5] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}};
        VkSpecializationInfo si{5, me, sizeof spec, spec};
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        rss.requiredSubgroupSize = 32;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.pNext = &rss;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = fn.shader;
        ci.stage.pName = "main";
        ci.stage.pSpecializationInfo = &si;
        ci.layout = g.pipeLayout;
        VkResult r = vk.CreateComputePipelines(g.dev, g.pcache, 1, &ci, nullptr, &p);
        if (r != VK_SUCCESS) {
            logf("kernel %s: pipeline creation failed (%d) for block %ux%ux%u", fn.name.c_str(), r, k.bx, k.by, k.bz);
            p = VK_NULL_HANDLE;
        } else {
            logf("kernel %s: pipeline for block %ux%ux%u smem %u", fn.name.c_str(), k.bx, k.by, k.bz, k.smem);
        }
    }
    fn.pipelines[k] = p;
    return p;
}

void savePipelineCache()
{
    size_t n = 0;
    if (!g.pcache || vk.GetPipelineCacheData(g.dev, g.pcache, &n, nullptr) != VK_SUCCESS || !n) return;
    std::vector<uint8_t> buf(n);
    if (vk.GetPipelineCacheData(g.dev, g.pcache, &n, buf.data()) != VK_SUCCESS) return;
    std::wstring path = wide(g_cache + "/pipeline-cache-win.bin"), tmp = path + L".tmp";
    HANDLE f = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return;
    DWORD w; WriteFile(f, buf.data(), (DWORD)n, &w, nullptr);
    CloseHandle(f);
    MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
}

void initGroups()
{
    if (!g_cfg.groups) return;
    std::string dir = g_cache + "/groups";
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(wide(dir + "/*.meta").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    std::vector<std::string> names;
    do {
        std::string n = utf8(fd.cFileName);
        names.push_back(n.substr(0, n.size() - 5));
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    std::sort(names.begin(), names.end());
    g.grpModule = std::make_unique<CuModule>();
    g.grpModule->hash = "group";
    for (auto& n : names) {
        std::vector<uint8_t> meta, spv;
        bool fa = g_cfg.accFast && readFile(dir + "/" + n + ".fa.spv", spv) && spv.size() >= 20;
        if (!readFile(dir + "/" + n + ".meta", meta) || (!fa && (!readFile(dir + "/" + n + ".spv", spv) || spv.size() < 20)))
            continue;
        auto grp = std::make_unique<Group>();
        grp->name = n;
        std::istringstream lines(std::string(meta.begin(), meta.end()));
        std::string line;
        while (std::getline(lines, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ws(line);
            std::string key;
            ws >> key;
            unsigned x = 0, y = 0, z = 0;
            if (key == "member") {
                std::vector<std::string> words;
                std::string word, m;
                while (ws >> word) words.push_back(word);
                for (size_t i = 0; i + 1 < words.size(); i += 2) m += (m.empty() ? "" : "|") + words[i] + ":" + words[i + 1];
                if (!m.empty()) grp->members.push_back(m);
            } else if (key == "block" && (ws >> x >> y >> z)) {
                grp->block[0] = x, grp->block[1] = y, grp->block[2] = z;
            } else if (key == "grid" && (ws >> x >> y >> z)) {
                grp->grid[0] = x, grp->grid[1] = y, grp->grid[2] = z;
            }
        }
        if (grp->members.empty() || grp->members.size() > kGrpMaxMembers || !grp->block[0]) {
            logf("group %s: bad meta", n.c_str());
            continue;
        }
        auto fn = std::make_unique<CuFunction>();
        fn->module = g.grpModule.get();
        fn->name = "group:" + n;
        VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sci.codeSize = spv.size() & ~(size_t)3;
        sci.pCode = (const uint32_t*)spv.data();
        if (vk.CreateShaderModule(g.dev, &sci, nullptr, &fn->shader) != VK_SUCCESS) continue;
        fn->meta.ok = true;
        grp->fn = fn.get();
        g.functions.push_back(std::move(fn));
        std::string m0 = grp->members[0];
        size_t p0 = 0;
        for (;;) {
            size_t q0 = m0.find('|', p0);
            g.groupFirst[m0.substr(p0, q0 == std::string::npos ? std::string::npos : q0 - p0)] = grp.get();
            if (q0 == std::string::npos) break;
            p0 = q0 + 1;
        }
        logf("group %s: %zu members, block %ux%ux%u grid %ux%ux%u%s", n.c_str(), grp->members.size(), grp->block[0],
             grp->block[1], grp->block[2], grp->grid[0], grp->grid[1], grp->grid[2], fa ? " (fp32 accumulation)" : "");
        g.groups.push_back(std::move(grp));
    }
    if (g.groups.empty()) return;
    void* ptr = nullptr;
    if (!hostBuffer((VkDeviceSize)kGrpRegions * kGrpRegion, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ptr, g.grpAddr)) {
        logf("launch groups: ring allocation failed; groups off");
        g.groups.clear();
        g.groupFirst.clear();
        return;
    }
    g.grpMap = (uint8_t*)ptr;
    memset(g.grpMap, 0, (size_t)kGrpRegions * kGrpRegion);
    for (uint32_t i = kGrpRegions; i-- > 0;) g.grpFree.push_back(i);
}

// ------------------------------------------------------------------------------------------------ device
template <typename F> void loadInst(F& f, const char* n) { f = (F)vk.GetInstanceProcAddr(g.inst, n); }

bool initDevice(ID3D12Device* d3d)
{
    g.tried = true;
    g.d3d = d3d;
    // the loader without implicit layers (overlays have no business in this device)
    wchar_t old[256] = {};
    DWORD hadOld = GetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", old, 256);
    SetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", L"~implicit~");
    vk.lib = LoadLibraryW(L"vulkan-1.dll");
    if (!vk.lib) { logf("vulkan-1.dll not found"); return false; }
    vk.GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(vk.lib, "vkGetInstanceProcAddr");
    vk.CreateInstance = (PFN_vkCreateInstance)vk.GetInstanceProcAddr(nullptr, "vkCreateInstance");
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "d4r dlssg";
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VkResult r = vk.CreateInstance(&ici, nullptr, &g.inst);
    SetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", hadOld ? old : nullptr);
    if (r != VK_SUCCESS) { logf("vkCreateInstance failed (%d)", r); return false; }
#define X(n) loadInst(vk.n, "vk" #n);
    VK_INSTANCE_FUNCS(X)
#undef X
    LUID luid = d3d->GetAdapterLuid();
    uint32_t n = 0;
    vk.EnumeratePhysicalDevices(g.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vk.EnumeratePhysicalDevices(g.inst, &n, pds.data());
    VkPhysicalDeviceProperties props{};
    for (auto p : pds) {
        VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp};
        vk.GetPhysicalDeviceProperties2(p, &pp);
        if (idp.deviceLUIDValid && !memcmp(idp.deviceLUID, &luid, sizeof luid)) { g.pd = p; props = pp.properties; }
    }
    if (!g.pd) { logf("no Vulkan device for the D3D12 adapter"); return false; }
    if (props.vendorID != 0x1002) { logf("%s is not an AMD GPU", props.deviceName); return false; }
    vk.GetPhysicalDeviceMemoryProperties(g.pd, &g.memprops);

    // extensions
    vk.EnumerateDeviceExtensionProperties(g.pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> avail(n);
    vk.EnumerateDeviceExtensionProperties(g.pd, nullptr, &n, avail.data());
    auto has = [&](const char* e) {
        for (auto& x : avail) if (!strcmp(x.extensionName, e)) return true;
        return false;
    };
    std::vector<const char*> exts;
    for (const char* e : {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
                          VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME,
                          VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME})
        if (has(e)) exts.push_back(e);
    if (g_cfg.kernelTiming && has(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) exts.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    // [Host] QueuePriority: the kernels' queue's global priority (VK_KHR_global_priority; 1 low, 2 medium, 3 high,
    // 4 realtime). In a game the game's D3D12 queues, d3dhip's HIP device (the upscaler) and this device take turns on
    // one GPU engine; the priority decides who goes first when several are ready. No effect without contention
    // (harness: within +-0.01 ms).
    const bool gprio = g_cfg.queuePriority && has(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
    if (gprio) exts.push_back(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
    if (!has(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) { logf("no VK_EXT_descriptor_buffer"); return false; }
    bool cm = has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME), wg = has(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME),
         fc2 = has(VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME);

    // features: what the translated kernels need, where the driver has it
    VkPhysicalDeviceDescriptorBufferFeaturesEXT dbf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR wgf{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
    VkPhysicalDeviceShaderFloatControls2FeaturesKHR fcf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    auto chain = [&]() {
        void* head = nullptr;
        auto push = [&](void* s) { ((VkBaseOutStructure*)s)->pNext = (VkBaseOutStructure*)head; head = s; };
        push(&dbf);
        if (cm) push(&cmf);
        if (wg) push(&wgf);
        if (fc2) push(&fcf);
        push(&v13);
        push(&v12);
        push(&v11);
        f2.pNext = head;
    };
    chain();
    vk.GetPhysicalDeviceFeatures2(g.pd, &f2);
    VkPhysicalDeviceFeatures2 have = f2;
    VkPhysicalDeviceVulkan11Features h11 = v11;
    VkPhysicalDeviceVulkan12Features h12 = v12;
    VkPhysicalDeviceVulkan13Features h13 = v13;
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR hwg = wgf;
    VkPhysicalDeviceShaderFloatControls2FeaturesKHR hfc = fcf;
    memset(&f2.features, 0, sizeof f2.features);
    auto clear = [](auto& s) { void* p = s.pNext; VkStructureType t = s.sType; memset(&s, 0, sizeof s); s.sType = t; s.pNext = p; };
    clear(v11); clear(v12); clear(v13); clear(wgf); clear(fcf); clear(cmf); clear(dbf);
#define WANT(dst, src, field) dst.field = src.field
    WANT(f2.features, have.features, shaderInt64); WANT(f2.features, have.features, shaderInt16);
    WANT(f2.features, have.features, shaderStorageImageReadWithoutFormat);
    WANT(f2.features, have.features, shaderStorageImageWriteWithoutFormat);
    WANT(f2.features, have.features, shaderSampledImageArrayDynamicIndexing);
    WANT(f2.features, have.features, shaderStorageImageArrayDynamicIndexing);
    WANT(v11, h11, storageBuffer16BitAccess);
    WANT(v12, h12, shaderFloat16); WANT(v12, h12, shaderInt8); WANT(v12, h12, storageBuffer8BitAccess);
    WANT(v12, h12, bufferDeviceAddress); WANT(v12, h12, descriptorIndexing); WANT(v12, h12, runtimeDescriptorArray);
    WANT(v12, h12, shaderSampledImageArrayNonUniformIndexing); WANT(v12, h12, shaderStorageImageArrayNonUniformIndexing);
    WANT(v12, h12, shaderBufferInt64Atomics); WANT(v12, h12, shaderSharedInt64Atomics); WANT(v12, h12, scalarBlockLayout);
    WANT(v12, h12, vulkanMemoryModel); WANT(v12, h12, vulkanMemoryModelDeviceScope); WANT(v12, h12, timelineSemaphore);
    WANT(v12, h12, hostQueryReset);
    WANT(v13, h13, subgroupSizeControl); WANT(v13, h13, computeFullSubgroups); WANT(v13, h13, shaderIntegerDotProduct);
    WANT(v13, h13, maintenance4); WANT(v13, h13, synchronization2);
    WANT(wgf, hwg, workgroupMemoryExplicitLayout); WANT(wgf, hwg, workgroupMemoryExplicitLayout8BitAccess);
    WANT(wgf, hwg, workgroupMemoryExplicitLayout16BitAccess); WANT(wgf, hwg, workgroupMemoryExplicitLayoutScalarBlockLayout);
    WANT(fcf, hfc, shaderFloatControls2);
#undef WANT
    cmf.cooperativeMatrix = cm;
    dbf.descriptorBuffer = VK_TRUE;
    if (!v12.bufferDeviceAddress || !v12.timelineSemaphore) { logf("missing buffer device address / timeline semaphores"); return false; }

    uint32_t nq = 0;
    vk.GetPhysicalDeviceQueueFamilyProperties(g.pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vk.GetPhysicalDeviceQueueFamilyProperties(g.pd, &nq, qf.data());
    g.family = UINT32_MAX;
    for (uint32_t i = 0; i < nq && g.family == UINT32_MAX; i++)
        if ((qf[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) == (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT))
            g.family = i;
    if (g.family == UINT32_MAX) g.family = 0;
    for (uint32_t i = 0; i < nq; i++)
        logf("queue family %u: flags 0x%x, %u queues", i, qf[i].queueFlags, qf[i].queueCount);
    if (g_cfg.computeQueue)
        for (uint32_t i = 0; i < nq; i++)
            if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { g.family = i; break; }
    logf("kernels' queue: family %u", g.family);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dq{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dq.queueFamilyIndex = g.family;
    dq.queueCount = 1;
    dq.pQueuePriorities = &prio;
    VkDeviceQueueGlobalPriorityCreateInfoKHR gp{VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR};
    static const VkQueueGlobalPriorityKHR kPrio[] = {VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR, VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
                                                     VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR, VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
                                                     VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR};
    static const char* kPrioName[] = {"default", "low", "medium", "high", "realtime"};
    gp.globalPriority = kPrio[std::min(g_cfg.queuePriority, 4u)];
    if (gprio) dq.pNext = &gp;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dq;
    dci.enabledExtensionCount = (uint32_t)exts.size();
    dci.ppEnabledExtensionNames = exts.data();
    r = vk.CreateDevice(g.pd, &dci, nullptr, &g.dev);
    if (r != VK_SUCCESS && gprio) {
        logf("vkCreateDevice with global priority %u failed (%d): default priority", g_cfg.queuePriority, r);
        dq.pNext = nullptr;
        exts.pop_back();
        dci.enabledExtensionCount = (uint32_t)exts.size();
        r = vk.CreateDevice(g.pd, &dci, nullptr, &g.dev);
    } else if (gprio) {
        logf("kernels' queue: global priority %s", kPrioName[std::min(g_cfg.queuePriority, 4u)]);
    }
    if (r != VK_SUCCESS) { logf("vkCreateDevice failed (%d)", r); return false; }
#define X(n) vk.n = (PFN_vk##n)vk.GetDeviceProcAddr(g.dev, "vk" #n);
    X(GetDeviceQueue) X(DeviceWaitIdle) X(QueueSubmit) X(CreateBuffer) X(GetBufferMemoryRequirements)
    X(AllocateMemory) X(BindBufferMemory) X(MapMemory) X(GetBufferDeviceAddress) X(CreateDescriptorSetLayout)
    X(CreatePipelineLayout) X(CreateShaderModule) X(CreateComputePipelines) X(CreatePipelineCache)
    X(GetPipelineCacheData) X(CreateSampler) X(CreateCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer)
    X(EndCommandBuffer) X(ResetCommandBuffer) X(CmdBindPipeline) X(CmdPushConstants) X(CmdDispatch)
    X(CmdPipelineBarrier) X(CreateSemaphore) X(ImportSemaphoreWin32HandleKHR) X(CreateFence) X(WaitForFences)
    X(ResetFences) X(GetDescriptorSetLayoutSizeEXT) X(GetDescriptorSetLayoutBindingOffsetEXT) X(GetDescriptorEXT)
    X(CmdBindDescriptorBuffersEXT) X(CmdSetDescriptorBufferOffsetsEXT) X(GetSemaphoreCounterValue)                    \
    X(CreateQueryPool) X(CmdResetQueryPool) X(CmdWriteTimestamp) X(GetQueryPoolResults) X(GetCalibratedTimestampsEXT)
#undef X
    vk.GetDeviceQueue(g.dev, g.family, 0, &g.queue);
    g.semImport = vk.ImportSemaphoreWin32HandleKHR != nullptr && g_cfg.gpuSync;

    VkPhysicalDeviceDescriptorBufferPropertiesEXT dbp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &dbp};
    vk.GetPhysicalDeviceProperties2(g.pd, &pp);
    g.imgSize = dbp.sampledImageDescriptorSize;
    g.smpSize = dbp.samplerDescriptorSize;
    UINT d3dInc = d3d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    UINT d3dSmp = d3d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    if (dbp.sampledImageDescriptorSize != dbp.storageImageDescriptorSize || g.imgSize > d3dInc || g.smpSize > d3dSmp) {
        logf("descriptor sizes differ: Vulkan sampled %zu storage %zu sampler %zu, D3D12 %u/%u", g.imgSize,
             (size_t)dbp.storageImageDescriptorSize, g.smpSize, d3dInc, d3dSmp);
        return false;
    }

    // the bindless set (rt.glsl's bindings), held in a descriptor buffer
    VkDescriptorSetLayoutBinding b[6] = {};
    b[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxImages, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxImages, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[2] = {2, VK_DESCRIPTOR_TYPE_SAMPLER, kMaxSamplers, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[4] = {4, VK_DESCRIPTOR_TYPE_SAMPLER, kMaxSamplers, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    b[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    li.bindingCount = 6;
    li.pBindings = b;
    if (vk.CreateDescriptorSetLayout(g.dev, &li, nullptr, &g.setLayout) != VK_SUCCESS) { logf("set layout failed"); return false; }
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g.setLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    if (vk.CreatePipelineLayout(g.dev, &pli, nullptr, &g.pipeLayout) != VK_SUCCESS) return false;
    VkDeviceSize size = 0;
    vk.GetDescriptorSetLayoutSizeEXT(g.dev, g.setLayout, &size);
    for (uint32_t i = 0; i < 6; i++) vk.GetDescriptorSetLayoutBindingOffsetEXT(g.dev, g.setLayout, i, &g.boff[i]);
    void* ptr = nullptr;
    if (!hostBuffer(size, VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                    ptr, g.daddr)) {
        logf("descriptor buffer allocation failed (%llu bytes)", (unsigned long long)size);
        return false;
    }
    g.dmap = (uint8_t*)ptr;
    memset(g.dmap, 0, (size_t)size);
    // binding 3: sampler info and format classes; binding 5: flags
    if (!hostBuffer(kInfoBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ptr, g.infoAddr)) return false;
    g.info = (uint32_t*)ptr;
    memset(g.info, 0, kInfoBytes);
    VkDeviceSize ia[2] = {g.infoAddr, kInfoBytes};
    writeDescriptor(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ia, 3, 0);
    if (!hostBuffer(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ptr, g.flagAddr)) return false;
    memset(ptr, 0, 256);
    VkDeviceSize fa[2] = {g.flagAddr, 256};
    writeDescriptor(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, fa, 5, 0);
    VkSampler edge = nearestCompanion(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    for (uint32_t i = 0; i < kMaxSamplers; i++) writeDescriptor(VK_DESCRIPTOR_TYPE_SAMPLER, &edge, 4, i);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(d3d->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&g.scratchHeap))) g.scratchHeap = nullptr;
    g.scratchInc = d3dInc;

    std::vector<uint8_t> cache;
    readFile(g_cache + "/pipeline-cache-win.bin", cache);
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pcci.initialDataSize = cache.size();
    pcci.pInitialData = cache.empty() ? nullptr : cache.data();
    if (vk.CreatePipelineCache(g.dev, &pcci, nullptr, &g.pcache) != VK_SUCCESS) {
        pcci.initialDataSize = 0;
        pcci.pInitialData = nullptr;
        vk.CreatePipelineCache(g.dev, &pcci, nullptr, &g.pcache);
    }
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = g.family;
    if (vk.CreateCommandPool(g.dev, &cpi, nullptr, &g.pool) != VK_SUCCESS) return false;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vk.CreateFence(g.dev, &fci, nullptr, &g.cpuFence);
    if (g_cfg.kernelTiming) {
        VkPhysicalDeviceProperties dp{};
        vk.GetPhysicalDeviceProperties(g.pd, &dp);
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kTsQueries;
        if (vk.CreateQueryPool && vk.CreateQueryPool(g.dev, &qi, nullptr, &g.tsPool) != VK_SUCCESS) g.tsPool = VK_NULL_HANDLE;
        g.tsPeriodMs = dp.limits.timestampPeriod * 1e-6;
        logf("kernel timing %s (timestamp period %.3f ns)", g.tsPool ? "on" : "unavailable", dp.limits.timestampPeriod);
    }
    initGroups();
    std::vector<uint8_t> arch;
    readFile(g_cache + "/arch", arch);
    logf("executor ready on %s (driver 0x%x, Vulkan device on the D3D12 adapter): descriptor buffer %llu KB, "
         "cooperative matrix %d, kernel cache %s (%.*s), hand-offs %s",
         props.deviceName, props.driverVersion, (unsigned long long)(size >> 10), (int)cm, g_cache.c_str(),
         (int)std::min<size_t>(arch.size(), 5), (const char*)arch.data(), g.semImport ? "on the GPU" : "on the CPU");
    return true;
}

bool ready(ID3D12Device* d3d)
{
    if (!g.tried && d3d) g.ok = initDevice(d3d);
    return g.ok;
}

// ------------------------------------------------------------------------------------------------ descriptors
std::string bytesKey(char tag, const void* a, size_t n, const void* b = nullptr, size_t m = 0)
{
    std::string k(1, tag);
    k.append((const char*)a, n);
    if (b) k.append((const char*)b, m);
    return k;
}

bool lookupView(SIZE_T h, ViewRec& out)
{
    std::lock_guard<std::mutex> l(g_viewLock);
    auto it = g_views.find(h);
    if (it == g_views.end()) return false;
    out = it->second;
    return true;
}

// a companion descriptor of the other kind for the same image (texture <-> surface), or false
bool companion(const ViewRec& v, std::vector<uint8_t>& bytes, DXGI_FORMAT& fmt)
{
    if (!v.res || !g.scratchHeap) return false;
    D3D12_RESOURCE_DESC rd = v.res->GetDesc();
    if (rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || rd.SampleDesc.Count != 1) return false;
    D3D12_CPU_DESCRIPTOR_HANDLE h = g.scratchHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += (SIZE_T)(g.scratchNext++ % 4096) * g.scratchInc;
    fmt = v.uav ? (v.hasDesc ? v.uavDesc.Format : rd.Format) : (v.hasDesc ? v.srv.Format : rd.Format);
    if (v.uav) {
        if (v.hasDesc && v.uavDesc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2D) return false;
        D3D12_SHADER_RESOURCE_VIEW_DESC s{};
        s.Format = fmt;
        s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Texture2D.MostDetailedMip = v.hasDesc ? v.uavDesc.Texture2D.MipSlice : 0;
        s.Texture2D.MipLevels = 1;
        s.Texture2D.PlaneSlice = v.hasDesc ? v.uavDesc.Texture2D.PlaneSlice : 0;
        g.d3d->CreateShaderResourceView(v.res, &s, h);
    } else {
        if (!(rd.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return false;
        if (v.hasDesc && v.srv.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D) return false;
        D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{fmt};
        if (FAILED(g.d3d->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof fs)) ||
            !(fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE))
            return false;
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = fmt;
        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        u.Texture2D.MipSlice = v.hasDesc ? v.srv.Texture2D.MostDetailedMip : 0;
        u.Texture2D.PlaneSlice = v.hasDesc ? v.srv.Texture2D.PlaneSlice : 0;
        g.d3d->CreateUnorderedAccessView(v.res, nullptr, &u, h);
    }
    bytes.assign((const uint8_t*)h.ptr, (const uint8_t*)h.ptr + g.imgSize);
    return true;
}

// the slot of a texture (uav = false) or surface descriptor; the other binding gets a companion when possible
uint32_t viewSlot(D3D12_CPU_DESCRIPTOR_HANDLE h, bool uav)
{
    if (!h.ptr) return 0;
    std::string key = bytesKey(uav ? 'u' : 't', (const void*)h.ptr, g.imgSize);
    auto it = g.viewSlots.find(key);
    if (it != g.viewSlots.end()) return it->second;
    if (g.nextSlot >= kMaxImages) {
        static bool once = false;
        if (!once) logf("out of image slots");
        once = true;
        return 0;
    }
    uint32_t slot = g.nextSlot++;
    ViewRec v;
    bool known = lookupView(h.ptr, v);
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    if (known) {
        D3D12_RESOURCE_DESC rd = v.res ? v.res->GetDesc() : D3D12_RESOURCE_DESC{};
        fmt = v.uav ? (v.hasDesc ? v.uavDesc.Format : rd.Format) : (v.hasDesc ? v.srv.Format : rd.Format);
    }
    VkFormat vf = vkFormat(fmt);
    uint32_t cls = formatClass(vf) | formatRaw(vf) << 8;
    memcpy(g.dmap + g.boff[uav ? 1 : 0] + (size_t)slot * g.imgSize, (const void*)h.ptr, g.imgSize);
    g.info[kMaxSamplers * 4 + (uav ? 0 : kMaxImages) + slot] = cls;
    std::vector<uint8_t> other;
    DXGI_FORMAT ofmt = DXGI_FORMAT_UNKNOWN;
    bool comp = known && companion(v, other, ofmt);
    if (comp) {
        memcpy(g.dmap + g.boff[uav ? 0 : 1] + (size_t)slot * g.imgSize, other.data(), g.imgSize);
        VkFormat of = vkFormat(ofmt);
        g.info[kMaxSamplers * 4 + (uav ? kMaxImages : 0) + slot] = formatClass(of) | formatRaw(of) << 8;
    }
    g.viewSlots[key] = slot;
    D3D12_RESOURCE_DESC rd = known && v.res ? v.res->GetDesc() : D3D12_RESOURCE_DESC{};
    logf("%s object slot %u: %s, format %d (%llux%u)%s", uav ? "surface" : "texture", slot,
         known ? "descriptor created by DLSS-G" : "UNKNOWN descriptor (no format)", (int)fmt,
         (unsigned long long)rd.Width, rd.Height, comp ? ", companion for the other binding" : "");
    return slot;
}

uint32_t samplerSlot(D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    if (!h.ptr) return 0;
    std::string key = bytesKey('s', (const void*)h.ptr, g.smpSize);
    auto it = g.samplerSlots.find(key);
    if (it != g.samplerSlots.end()) return it->second;
    if (g.nextSampler >= kMaxSamplers) return 0;
    uint32_t slot = g.nextSampler++;
    memcpy(g.dmap + g.boff[2] + (size_t)slot * g.smpSize, (const void*)h.ptr, g.smpSize);
    D3D12_SAMPLER_DESC sd{};
    bool known;
    {
        std::lock_guard<std::mutex> l(g_viewLock);
        auto si = g_samplers.find(h.ptr);
        known = si != g_samplers.end();
        if (known) sd = si->second;
    }
    uint32_t* e = g.info + 4 * slot;
    auto addr = [](D3D12_TEXTURE_ADDRESS_MODE m) {
        switch (m) {
        case D3D12_TEXTURE_ADDRESS_MODE_WRAP: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case D3D12_TEXTURE_ADDRESS_MODE_MIRROR: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case D3D12_TEXTURE_ADDRESS_MODE_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }
    };
    VkSamplerAddressMode u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, v = u;
    if (known) {
        uint32_t minf = (sd.Filter >> 4) & 1, magf = (sd.Filter >> 2) & 1;
        u = addr(sd.AddressU);
        v = addr(sd.AddressV);
        VkBorderColor border = sd.BorderColor[3] == 0 ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
                               : sd.BorderColor[0] == 0 ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK
                                                        : VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        e[0] = minf | (magf << 4);
        e[1] = (uint32_t)u;
        e[2] = (uint32_t)v;
        e[3] = (uint32_t)border;
    }
    VkSampler c = nearestCompanion(u, v);
    writeDescriptor(VK_DESCRIPTOR_TYPE_SAMPLER, &c, 4, slot);
    g.samplerSlots[key] = slot;
    logf("sampler slot %u: %s filter 0x%x address %d/%d", slot, known ? "created by DLSS-G" : "UNKNOWN", known ? sd.Filter : 0,
         known ? (int)sd.AddressU : 0, known ? (int)sd.AddressV : 0);
    return slot;
}

// ------------------------------------------------------------------------------------------------ device hooks
// ID3D12Device slots: CreateShaderResourceView 18, CreateUnorderedAccessView 19, CreateSampler 22,
// CopyDescriptorsSimple 24
using CreateSrvFn = void(STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Resource*, const D3D12_SHADER_RESOURCE_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
using CreateUavFn = void(STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Resource*, ID3D12Resource*, const D3D12_UNORDERED_ACCESS_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
using CreateSamplerFn = void(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_SAMPLER_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
using CopySimpleFn = void(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_TYPE);
CreateSrvFn o_createSrv;
CreateUavFn o_createUav;
CreateSamplerFn o_createSampler;
CopySimpleFn o_copySimple;
UINT g_srvInc = 32, g_smpInc = 16;

void STDMETHODCALLTYPE hCreateSrv(ID3D12Device* d, ID3D12Resource* r, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    o_createSrv(d, r, desc, h);
    if (!t_fg) return;
    ViewRec v;
    v.res = r;
    v.hasDesc = desc != nullptr;
    if (desc) v.srv = *desc;
    std::lock_guard<std::mutex> l(g_viewLock);
    g_views[h.ptr] = v;
}
void STDMETHODCALLTYPE hCreateUav(ID3D12Device* d, ID3D12Resource* r, ID3D12Resource* c, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
                                  D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    o_createUav(d, r, c, desc, h);
    if (!t_fg) return;
    ViewRec v;
    v.res = r;
    v.uav = true;
    v.hasDesc = desc != nullptr;
    if (desc) v.uavDesc = *desc;
    std::lock_guard<std::mutex> l(g_viewLock);
    g_views[h.ptr] = v;
}
void STDMETHODCALLTYPE hCreateSampler(ID3D12Device* d, const D3D12_SAMPLER_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    o_createSampler(d, desc, h);
    if (!t_fg || !desc) return;
    std::lock_guard<std::mutex> l(g_viewLock);
    g_samplers[h.ptr] = *desc;
}
void STDMETHODCALLTYPE hCopySimple(ID3D12Device* d, UINT n, D3D12_CPU_DESCRIPTOR_HANDLE dst, D3D12_CPU_DESCRIPTOR_HANDLE src,
                                   D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    o_copySimple(d, n, dst, src, type);
    if (!t_fg || (type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && type != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)) return;
    std::lock_guard<std::mutex> l(g_viewLock);
    UINT inc = type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ? g_smpInc : g_srvInc;
    for (UINT i = 0; i < n; i++) {
        SIZE_T s = src.ptr + (SIZE_T)i * inc, t = dst.ptr + (SIZE_T)i * inc;
        if (type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) {
            auto it = g_samplers.find(s);
            if (it != g_samplers.end()) g_samplers[t] = it->second;
        } else {
            auto it = g_views.find(s);
            if (it != g_views.end()) g_views[t] = it->second;
        }
    }
}

bool patchSlot(void** vtable, size_t slot, void* hook, void** original)
{
    DWORD prot = 0;
    if (!VirtualProtect(&vtable[slot], sizeof(void*), PAGE_EXECUTE_READWRITE, &prot)) return false;
    if (vtable[slot] != hook) {
        *original = vtable[slot];
        vtable[slot] = hook;
    }
    VirtualProtect(&vtable[slot], sizeof(void*), prot, &prot);
    FlushInstructionCache(GetCurrentProcess(), &vtable[slot], sizeof(void*));
    return true;
}

void hookDevice(ID3D12Device* d)
{
    static bool done = false;
    if (done || !d) return;
    done = true;
    g_srvInc = d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    g_smpInc = d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    void** vt = *reinterpret_cast<void***>(d);
    bool ok = patchSlot(vt, 18, (void*)&hCreateSrv, (void**)&o_createSrv) &&
              patchSlot(vt, 19, (void*)&hCreateUav, (void**)&o_createUav) &&
              patchSlot(vt, 22, (void*)&hCreateSampler, (void**)&o_createSampler) &&
              patchSlot(vt, 24, (void*)&hCopySimple, (void**)&o_copySimple);
    logf("device %p: descriptor creation hooks %s", (void*)d, ok ? "installed" : "FAILED");
}

} // namespace
} // namespace cux

// ------------------------------------------------------------------------------------------------ list redirection
// As tools/d4r_d3d12_split.h: every recording method of ID3D12GraphicsCommandList from ClearState (slot 11) on goes
// through a stub that asks cux_split_target which list the call records into (the stub passes its slot), then
// cux_split_fn for the original method of that list's class: list classes differ (a compute list's vtable is not a
// direct list's), and each hooked class keeps its own originals.
extern "C" {
extern void* cux_split_thunks[86];
__attribute__((used)) void* cux_split_target(void* self, unsigned slot);
__attribute__((used)) void* cux_split_fn(void* list, unsigned slot);
}
namespace cux {
namespace {
struct HookedVt { std::atomic<void**> vt{nullptr}; void* orig[86] = {}; };
HookedVt g_listVts[16], g_queueVts[16];
std::atomic<int> g_nListVts{0}, g_nQueueVts{0};
HookedVt* hookedVt(HookedVt* table, int n, void* obj)
{
    void** vt = *reinterpret_cast<void***>(obj);
    for (int i = 0; i < n; i++)
        if (table[i].vt.load(std::memory_order_acquire) == vt) return &table[i];
    return nullptr;
}
} // namespace
} // namespace cux
extern "C" void* cux_split_fn(void* list, unsigned slot)
{
    cux::HookedVt* h = cux::hookedVt(cux::g_listVts, cux::g_nListVts.load(std::memory_order_acquire), list);
    return h ? h->orig[slot] : nullptr;
}

asm(R"(
    .text
    .altmacro
    .macro cux_split_stub n
    .globl cux_split_thunk_\n
cux_split_thunk_\n:
    push %rcx
    push %rdx
    push %r8
    push %r9
    sub $0x68, %rsp
    movdqu %xmm0, 0x20(%rsp)
    movdqu %xmm1, 0x30(%rsp)
    movdqu %xmm2, 0x40(%rsp)
    movdqu %xmm3, 0x50(%rsp)
    mov $\n, %edx
    call cux_split_target
    mov %rax, 0x60(%rsp)
    mov %rax, %rcx
    mov $\n, %edx
    call cux_split_fn
    mov %rax, %r11
    mov 0x60(%rsp), %rax
    movdqu 0x20(%rsp), %xmm0
    movdqu 0x30(%rsp), %xmm1
    movdqu 0x40(%rsp), %xmm2
    movdqu 0x50(%rsp), %xmm3
    add $0x68, %rsp
    pop %r9
    pop %r8
    pop %rdx
    pop %rcx
    mov %rax, %rcx
    jmp *%r11
    .endm
    .macro cux_split_entry n
    .quad cux_split_thunk_\n
    .endm
    .set cux_split_i, 11
    .rept 75
    cux_split_stub %cux_split_i
    .set cux_split_i, cux_split_i + 1
    .endr
    .section .rdata,"dr"
    .p2align 3
    .globl cux_split_thunks
cux_split_thunks:
    .rept 11
    .quad 0
    .endr
    .set cux_split_i, 11
    .rept 75
    cux_split_entry %cux_split_i
    .set cux_split_i, cux_split_i + 1
    .endr
    .noaltmacro
    .text
)");

namespace cux {
namespace {

// bound state of a list during DLSS-G's evaluate, replayed into each new part
struct RootArg {
    uint8_t kind = 0;   // 1 table, 2 constants, 3 CBV, 4 SRV, 5 UAV
    uint64_t value = 0;
    std::vector<uint32_t> consts;
};
struct BoundState {
    ID3D12DescriptorHeap* heaps[2] = {};
    UINT heapCount = 0;
    ID3D12RootSignature* rs[2] = {};   // compute, graphics
    ID3D12PipelineState* pso = nullptr;
    std::vector<RootArg> args[2];
};

struct Chain {
    std::atomic<ID3D12GraphicsCommandList*> list{nullptr};
    std::atomic<ID3D12GraphicsCommandList*> target{nullptr};   // null: the list itself
    std::atomic<bool> dirty{false};
    bool closed = false;
    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    BoundState state;
    std::vector<Batch*> batches;   // batches[k] runs after part k-1 (part -1 = the list itself)
    std::vector<Part> parts;
    uint32_t slotCalls[86] = {};
    std::string trace;   // [Host] SplitLog
};
constexpr int kMaxChains = 32;
bool g_splitLog = false;   // [Host] SplitLog: what ends each kernel batch, for a few lists (nvxrt.log)
Chain g_chains[kMaxChains];
std::atomic<int> g_activeChains{0};
bool g_listHooks = false;

// The list a pointer records into. Streamline hands DLSS-G its own command list proxies, which forward every call
// to the native list (the one our vtable hooks, Close, Reset and ExecuteCommandLists see): a PIX marker recorded
// through the pointer shows which `this` arrives at the hooks.
thread_local bool t_probe = false;
thread_local void* t_probed = nullptr;

Chain* findChain(void* list)
{
    for (auto& c : g_chains)
        if (c.list.load(std::memory_order_acquire) == list) return &c;
    return nullptr;
}

} // namespace
} // namespace cux

extern "C" void* cux_split_target(void* self, unsigned slot)
{
    using namespace cux;
    if (t_probe && slot == 56) { t_probed = self; return self; }   // SetMarker from nativeList()
    if (g_activeChains.load(std::memory_order_acquire) == 0) return self;
    Chain* c = findChain(self);
    if (!c) return self;
    if (slot < 86) c->slotCalls[slot]++;
    if (g_splitLog) c->trace += " s" + std::to_string(slot);
    // EndQuery (53), SetMarker (56), BeginEvent (57), EndEvent (58) touch nothing the kernels read or write: they go
    // into the next D3D12 part without ending the open kernel batch (a timestamp then lands after the batch)
    if (!(slot == 53 || slot == 56 || slot == 57 || slot == 58) || g_cfg.splitQueries)
        c->dirty.store(true, std::memory_order_relaxed);
    ID3D12GraphicsCommandList* t = c->target.load(std::memory_order_acquire);
    return t ? t : self;
}

namespace cux {
namespace {

using List = ID3D12GraphicsCommandList;
template <typename F> F orig(void* list, size_t slot) { return reinterpret_cast<F>(cux_split_fn(list, (unsigned)slot)); }

Chain* stateChain(List* self, List*& target)
{
    target = self;
    if (g_activeChains.load(std::memory_order_acquire) == 0) return nullptr;
    Chain* c = findChain(self);
    if (c) {
        if (List* t = c->target.load(std::memory_order_acquire)) target = t;
    }
    return c;
}

RootArg& arg(Chain* c, int gfx, UINT i)
{
    auto& v = c->state.args[gfx];
    if (v.size() <= i) v.resize(i + 1);
    return v[i];
}

// Barriers between launches. UAV barriers and transitions among shader-access states change nothing the kernels see
// (they read and write through the raw descriptors, and each launch is behind a full Vulkan memory barrier), so they
// do not end the open kernel batch: they go into the next D3D12 part. Anything else (copy, render target, depth,
// present, aliasing) may make the driver touch the memory and ends the batch as any other recording does.
// Exact=1 / Barriers=split restores splitting at every barrier.
bool g_splitBarriers = false;
bool shaderOnly(D3D12_RESOURCE_STATES s)
{
    const D3D12_RESOURCE_STATES ok = D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                     D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    return s != 0 && (s & ~ok) == 0;
}
// A transition between a shader state and COPY_SOURCE / COPY_DEST around a copy that runs on the kernels' queue
// (recordCopy): it changes nothing the kernels see either, and goes into the next D3D12 part with the other barriers.
// A copy that cannot run there ends the batch itself, after these barriers in D3D12 order.
bool copyOrShader(D3D12_RESOURCE_STATES s)
{
    return shaderOnly(s) || s == D3D12_RESOURCE_STATE_COPY_SOURCE || s == D3D12_RESOURCE_STATE_COPY_DEST;
}
void STDMETHODCALLTYPE sResourceBarrier(List* self, UINT n, const D3D12_RESOURCE_BARRIER* b)
{
    List* t;
    if (Chain* c = stateChain(self, t)) {
        c->slotCalls[26]++;
        bool harmless = !g_splitBarriers;
        for (UINT i = 0; i < n && harmless; i++) {
            if (b[i].Type == D3D12_RESOURCE_BARRIER_TYPE_UAV) continue;
            const D3D12_RESOURCE_STATES from = b[i].Transition.StateBefore, to = b[i].Transition.StateAfter;
            harmless = b[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION &&
                       ((shaderOnly(from) && shaderOnly(to)) ||
                        (g_cfg.vkCopies && copyOrShader(from) && copyOrShader(to) && (shaderOnly(from) || shaderOnly(to))));
        }
        if (!harmless) c->dirty.store(true, std::memory_order_relaxed);
        if (g_splitLog) {
            char t[64];
            for (UINT i = 0; i < n; i++) {
                if (b[i].Type == D3D12_RESOURCE_BARRIER_TYPE_UAV) snprintf(t, sizeof t, " u%p", (void*)b[i].UAV.pResource);
                else if (b[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
                    snprintf(t, sizeof t, " t%p:%x>%x", (void*)b[i].Transition.pResource, (unsigned)b[i].Transition.StateBefore,
                             (unsigned)b[i].Transition.StateAfter);
                else snprintf(t, sizeof t, " a");
                c->trace += t;
            }
        }
    }
    orig<void(STDMETHODCALLTYPE*)(List*, UINT, const D3D12_RESOURCE_BARRIER*)>(t, 26)(t, n, b);
}

void STDMETHODCALLTYPE sSetPipelineState(List* self, ID3D12PipelineState* p)
{
    List* t; if (Chain* c = stateChain(self, t)) c->state.pso = p;
    orig<void(STDMETHODCALLTYPE*)(List*, ID3D12PipelineState*)>(t, 25)(t, p);
}
void STDMETHODCALLTYPE sSetDescriptorHeaps(List* self, UINT n, ID3D12DescriptorHeap* const* h)
{
    List* t;
    if (Chain* c = stateChain(self, t)) {
        c->state.heapCount = std::min<UINT>(n, 2);
        for (UINT i = 0; i < c->state.heapCount; i++) c->state.heaps[i] = h[i];
    }
    orig<void(STDMETHODCALLTYPE*)(List*, UINT, ID3D12DescriptorHeap* const*)>(t, 28)(t, n, h);
}
template <int Gfx> void STDMETHODCALLTYPE sSetRootSignature(List* self, ID3D12RootSignature* rs)
{
    List* t;
    if (Chain* c = stateChain(self, t)) { c->state.rs[Gfx] = rs; c->state.args[Gfx].clear(); }
    orig<void(STDMETHODCALLTYPE*)(List*, ID3D12RootSignature*)>(t, 29 + Gfx)(t, rs);
}
template <int Gfx> void STDMETHODCALLTYPE sSetTable(List* self, UINT i, D3D12_GPU_DESCRIPTOR_HANDLE h)
{
    List* t;
    if (Chain* c = stateChain(self, t)) { auto& a = arg(c, Gfx, i); a.kind = 1; a.value = h.ptr; }
    orig<void(STDMETHODCALLTYPE*)(List*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE)>(t, 31 + Gfx)(t, i, h);
}
template <int Gfx> void STDMETHODCALLTYPE sSetConstant(List* self, UINT i, UINT v, UINT off)
{
    List* t;
    if (Chain* c = stateChain(self, t)) {
        auto& a = arg(c, Gfx, i);
        a.kind = 2;
        if (a.consts.size() <= off) a.consts.resize(off + 1);
        a.consts[off] = v;
    }
    orig<void(STDMETHODCALLTYPE*)(List*, UINT, UINT, UINT)>(t, 33 + Gfx)(t, i, v, off);
}
template <int Gfx> void STDMETHODCALLTYPE sSetConstants(List* self, UINT i, UINT n, const void* p, UINT off)
{
    List* t;
    if (Chain* c = stateChain(self, t)) {
        auto& a = arg(c, Gfx, i);
        a.kind = 2;
        if (a.consts.size() < off + n) a.consts.resize(off + n);
        memcpy(a.consts.data() + off, p, n * 4);
    }
    orig<void(STDMETHODCALLTYPE*)(List*, UINT, UINT, const void*, UINT)>(t, 35 + Gfx)(t, i, n, p, off);
}
template <int Gfx, int Kind> void STDMETHODCALLTYPE sSetView(List* self, UINT i, D3D12_GPU_VIRTUAL_ADDRESS va)
{
    List* t;
    if (Chain* c = stateChain(self, t)) { auto& a = arg(c, Gfx, i); a.kind = Kind; a.value = va; }
    orig<void(STDMETHODCALLTYPE*)(List*, UINT, D3D12_GPU_VIRTUAL_ADDRESS)>(t, 37 + 2 * (Kind - 3) + Gfx)(t, i, va);
}

void replay(Chain* c, List* l)
{
    BoundState& s = c->state;
    if (s.heapCount) orig<void(STDMETHODCALLTYPE*)(List*, UINT, ID3D12DescriptorHeap* const*)>(l, 28)(l, s.heapCount, s.heaps);
    if (s.rs[0]) orig<void(STDMETHODCALLTYPE*)(List*, ID3D12RootSignature*)>(l, 29)(l, s.rs[0]);
    if (s.rs[1] && c->type == D3D12_COMMAND_LIST_TYPE_DIRECT) orig<void(STDMETHODCALLTYPE*)(List*, ID3D12RootSignature*)>(l, 30)(l, s.rs[1]);
    if (s.pso) orig<void(STDMETHODCALLTYPE*)(List*, ID3D12PipelineState*)>(l, 25)(l, s.pso);
    for (int gfx = 0; gfx < 2; gfx++) {
        if (!s.rs[gfx]) continue;
        for (UINT i = 0; i < s.args[gfx].size(); i++) {
            const RootArg& a = s.args[gfx][i];
            switch (a.kind) {
            case 1: orig<void(STDMETHODCALLTYPE*)(List*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE)>(l, 31 + gfx)(l, i, D3D12_GPU_DESCRIPTOR_HANDLE{a.value}); break;
            case 2: orig<void(STDMETHODCALLTYPE*)(List*, UINT, UINT, const void*, UINT)>(l, 35 + gfx)(l, i, (UINT)a.consts.size(), a.consts.data(), 0); break;
            case 3: case 4: case 5:
                orig<void(STDMETHODCALLTYPE*)(List*, UINT, D3D12_GPU_VIRTUAL_ADDRESS)>(l, 37 + 2 * (a.kind - 3) + gfx)(l, i, a.value);
                break;
            default: break;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ batches, parts
void cmdRelease(VkCommandBuffer cmd)
{
    auto it = g.cmdSlots.find(cmd);
    if (it != g.cmdSlots.end()) {
        g.grpFree.insert(g.grpFree.end(), it->second.grp.begin(), it->second.grp.end());
        g.bpFree.insert(g.bpFree.end(), it->second.bp.begin(), it->second.bp.end());
        g.cmdSlots.erase(it);
    }
    g.pending.erase(cmd);
}

const std::string kLabelClear = "(clear)";
void stamp(Batch* b, const std::string* label)
{
    if (!g.tsPool || !b) return;
    uint32_t q = g.tsNext++ % kTsQueries;
    vk.CmdResetQueryPool(b->cmd, g.tsPool, q, 1);
    vk.CmdWriteTimestamp(b->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.tsPool, q);
    if (!b->ts.empty()) b->tsLabel.push_back(label);
    b->ts.push_back(q);
}

// a finished batch's timestamps into the per-kernel totals; a summary every 1000 batches
// [Host] KernelTiming: GPU timestamps of both APIs on the CPU's QueryPerformanceCounter timeline (ms)
struct Calib { double base = 0, msPerTick = 0; uint64_t tick0 = 0; bool ok = false; };
Calib vkCalib()
{
    Calib c;
    if (!vk.GetCalibratedTimestampsEXT) return c;
    VkCalibratedTimestampInfoEXT ci[2] = {{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT},
                                          {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT}};
    ci[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
    ci[1].timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
    uint64_t ts[2] = {}, dev = 0;
    if (vk.GetCalibratedTimestampsEXT(g.dev, 2, ci, ts, &dev) != VK_SUCCESS) return c;
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    c.base = (double)ts[1] * 1000.0 / (double)f.QuadPart;
    c.tick0 = ts[0];
    c.msPerTick = g.tsPeriodMs;
    c.ok = true;
    return c;
}
Calib d3dCalib(ID3D12CommandQueue* q)
{
    Calib c;
    UINT64 gpu = 0, cpu = 0, f = 0;
    if (FAILED(q->GetClockCalibration(&gpu, &cpu)) || FAILED(q->GetTimestampFrequency(&f)) || !f) return c;
    LARGE_INTEGER qf;
    QueryPerformanceFrequency(&qf);
    c.base = (double)cpu * 1000.0 / (double)qf.QuadPart;
    c.tick0 = gpu;
    c.msPerTick = 1000.0 / (double)f;
    c.ok = true;
    return c;
}
double onCpu(const Calib& c, uint64_t t) { return c.base + ((double)(int64_t)(t - c.tick0)) * c.msPerTick; }

void harvest(Batch* b)
{
    if (!g.tsPool || b->ts.size() < 2) return;
    std::vector<uint64_t> t(b->ts.size());
    for (size_t i = 0; i < b->ts.size(); i++)
        if (vk.GetQueryPoolResults(g.dev, g.tsPool, b->ts[i], 1, 8, &t[i], 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
    for (size_t i = 1; i < t.size(); i++) {
        auto& e = g.ktime[*b->tsLabel[i - 1]];
        e.first += (double)(t[i] - t[i - 1]) * g.tsPeriodMs;
        e.second++;
    }
    g.batchMs += (double)(t.back() - t.front()) * g.tsPeriodMs;
    if (b->d3dTs >= 0 && g.d3dTsMap) {
        uint64_t d0 = g.d3dTsMap[b->d3dTs], d1 = g.d3dTsMap[b->d3dTs + 1];
        if (d1 > d0) {
            g.d3dSpanMs += (double)(d1 - d0) * g.d3dTsMsPerTick[b->d3dQ];
            g.vkSpanMs += (double)(t.back() - t.front()) * g.tsPeriodMs;
            g.d3dTimed++;
            static Calib vc, dc;
            static uint64_t calibAt = 0;
            if (!vc.ok || g.d3dTimed - calibAt > 200 || calibAt > g.d3dTimed) {
                vc = vkCalib();
                dc = d3dCalib(b->d3dQ);
                calibAt = g.d3dTimed;
            }
            if (vc.ok && dc.ok) {
                double l1 = onCpu(vc, t.front()) - onCpu(dc, d0), l2 = onCpu(dc, d1) - onCpu(vc, t.back());
                if (l1 > -1 && l2 > -1 && l1 < 50 && l2 < 50) {
                    g.lat1Ms += l1;
                    g.lat2Ms += l2;
                    g.latN++;
                }
            }
        }
    }
    if (++g.timedBatches % 1000) return;
    if (g.d3dTimed) {
        double n = (double)g.d3dTimed;
        logf("hand-off timing, last %llu batches: game queue idle %.4f ms per batch (from the D3D12 work before the kernels "
             "to the work after them), of which the kernels' batch %.4f ms (its start to its end): %.4f ms per round trip",
             (unsigned long long)g.d3dTimed, g.d3dSpanMs / n, g.vkSpanMs / n, (g.d3dSpanMs - g.vkSpanMs) / n);
        if (g.latN)
            logf("hand-off timing: D3D12 done -> kernels start %.4f ms, kernels done -> D3D12 resumes %.4f ms (calibrated "
                 "clocks, %llu batches)", g.lat1Ms / g.latN, g.lat2Ms / g.latN, (unsigned long long)g.latN);
        g.lat1Ms = g.lat2Ms = 0;
        g.latN = 0;
        g.d3dSpanMs = g.vkSpanMs = 0;
        g.d3dTimed = 0;
    }
    std::vector<std::pair<double, std::string>> v;
    for (auto& k : g.ktime) v.push_back({k.second.first, k.first});
    std::sort(v.rbegin(), v.rend());
    double n = 1000.0;
    std::string out;
    char buf[256];
    for (auto& e : v) {
        snprintf(buf, sizeof buf, "\n  %8.4f ms/batch  %6.2f/batch  %s", e.first / n, g.ktime[e.second].second / n, e.second.c_str());
        out += buf;
    }
    logf("kernel timing, last 1000 batches: %.4f ms per batch on the kernels' queue (entry barrier to last dispatch)%s",
         g.batchMs / n, out.c_str());
    g.ktime.clear();
    g.batchMs = 0;
}

// [Host] KernelTiming: a full barrier and a timestamp at the batch's end, so the cache flush the hand-off does
// implicitly shows as its own line ("exit barrier")
void endBatch(Batch* b)
{
    if (b->ended) return;
    if (g.tsPool) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vk.CmdPipelineBarrier(b->cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                              nullptr, 0, nullptr);
        static const std::string kLabelExit = "(exit barrier)";
        stamp(b, &kLabelExit);
    }
    vk.EndCommandBuffer(b->cmd);
    b->ended = true;
}

void freeBatch(Batch* b)
{
    if (b->cmd) {
        if (!b->ended) vk.EndCommandBuffer(b->cmd);
        vk.ResetCommandBuffer(b->cmd, 0);
        cmdRelease(b->cmd);
        g.freeCmds.push_back(b->cmd);
    }
    delete b;
}

void recycle()
{
    for (size_t i = 0; i < g.inflightBatches.size();) {
        Batch* b = g.inflightBatches[i];
        if (b->qs->fence->GetCompletedValue() >= b->done) {
            harvest(b);
            freeBatch(b);
            g.inflightBatches[i] = g.inflightBatches.back();
            g.inflightBatches.pop_back();
        } else {
            i++;
        }
    }
    for (size_t i = 0; i < g.inflightParts.size();) {
        Part& p = g.inflightParts[i];
        if (p.qs->fence->GetCompletedValue() >= p.done) {
            g.freeParts.push_back(p);
            g.inflightParts[i] = g.inflightParts.back();
            g.inflightParts.pop_back();
        } else {
            i++;
        }
    }
}

Batch* newBatch()
{
    recycle();
    auto* b = new Batch;
    if (!g.freeCmds.empty()) {
        b->cmd = g.freeCmds.back();
        g.freeCmds.pop_back();
    } else {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = g.pool;
        ai.commandBufferCount = 1;
        if (vk.AllocateCommandBuffers(g.dev, &ai, &b->cmd) != VK_SUCCESS) { delete b; return nullptr; }
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.BeginCommandBuffer(b->cmd, &bi);
    VkDescriptorBufferBindingInfoEXT dbi{VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT};
    dbi.address = g.daddr;
    dbi.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
    vk.CmdBindDescriptorBuffersEXT(b->cmd, 1, &dbi);
    uint32_t idx = 0;
    VkDeviceSize off = 0;
    vk.CmdSetDescriptorBufferOffsetsEXT(b->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipeLayout, 0, 1, &idx, &off);
    stamp(b, nullptr);   // [Host] KernelTiming: the batch's start, before its entry barrier
    // everything the D3D12 queue did before the hand-off is visible (the semaphore wait covers execution; this the caches)
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vk.CmdPipelineBarrier(b->cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                          nullptr, 0, nullptr);
    static const std::string kLabelEntry = "(entry barrier)";
    stamp(b, &kLabelEntry);
    return b;
}

bool newPart(Chain* c, ID3D12Device* d3d, Part& out)
{
    recycle();
    for (size_t i = 0; i < g.freeParts.size(); i++) {
        if (g.freeParts[i].type != c->type) continue;
        out = g.freeParts[i];
        g.freeParts.erase(g.freeParts.begin() + i);
        out.alloc->Reset();
        out.list->Reset(out.alloc, nullptr);
        return true;
    }
    out = Part{};
    out.type = c->type;
    if (FAILED(d3d->CreateCommandAllocator(c->type, __uuidof(ID3D12CommandAllocator), (void**)&out.alloc)) ||
        FAILED(d3d->CreateCommandList(0, c->type, out.alloc, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&out.list))) {
        logf("part list creation failed");
        return false;
    }
    g.partsCreated++;
    return true;
}

// the chain of a list DLSS-G records into (created on demand); g_lock held

bool hookListClass(List* list);

List* nativeList(List* list)
{
    // Streamline's proxies hand out the object they wrap for this IID (StreamlineRetrieveBaseInterface)
    static const GUID kSlBase = {0xadec44e2, 0x61f0, 0x45c3, {0xad, 0x9f, 0x1b, 0x37, 0x37, 0x92, 0x84, 0xff}};
    List* native = list;
    for (int depth = 0; depth < 4; depth++) {
        IUnknown* base = nullptr;
        if (FAILED(native->QueryInterface(kSlBase, (void**)&base)) || !base || base == native) {
            if (base) base->Release();
            break;
        }
        base->Release();   // the proxy keeps it alive
        native = static_cast<List*>(static_cast<void*>(base));
    }
    if (native == list) {
        static const char kMarker[] = "d4r";
        t_probe = true;
        t_probed = nullptr;
        list->SetMarker(1, kMarker, sizeof kMarker);
        t_probe = false;
        if (t_probed) native = static_cast<List*>(t_probed);
    }
    bool known = hookedVt(g_listVts, g_nListVts.load(), native) != nullptr;
    static std::set<void*> seen;
    if (seen.insert(list).second)
        logf("list %p: native %p, class %p%s", (void*)list, (void*)native, *(void**)native,
             known ? "" : " (new class: hooking it)");
    if (!known) hookListClass(native);
    return native;
}

Chain* chainFor(List* list, bool create)
{
    if (Chain* c = findChain(list)) return c;
    list = nativeList(list);
    if (Chain* c = findChain(list)) return c;
    if (!create) return nullptr;
    for (auto& c : g_chains) {
        if (c.list.load() != nullptr) continue;
        c.target.store(nullptr);
        c.dirty.store(false);
        c.closed = false;
        c.type = list->GetType();
        c.state = BoundState{};
        c.batches.clear();
        c.parts.clear();
        memset(c.slotCalls, 0, sizeof c.slotCalls);
        c.trace.clear();
        c.list.store(list, std::memory_order_release);
        g_activeChains.fetch_add(1);
        return &c;
    }
    logf("no free chain slot for list %p", (void*)list);
    return nullptr;
}

void finishChain(Chain* c)
{
    c->list.store(nullptr, std::memory_order_release);
    c->target.store(nullptr);
    c->batches.clear();
    c->parts.clear();
    g_activeChains.fetch_sub(1);
}

// the batch a launch recorded into `list` goes to; g_lock held
Batch* batchFor(List* list)
{
    Chain* c = chainFor(list, true);
    if (!c) return nullptr;
    if (c->closed) {
        logf("launch into a closed list %p", (void*)list);
        return nullptr;
    }
    if (!c->batches.empty() && !c->dirty.load()) return c->batches.back();
    // nothing to append to: the kernels run after what was recorded so far, later D3D12 work goes into a new part
    ID3D12Device* d3d = nullptr;
    list->GetDevice(__uuidof(ID3D12Device), (void**)&d3d);
    Part p;
    bool ok = d3d && newPart(c, d3d, p);
    if (d3d) d3d->Release();
    if (!ok) return nullptr;
    Batch* b = newBatch();
    if (!b) { g.freeParts.push_back(p); return nullptr; }
    replay(c, p.list);
    // a launch group open in the previous batch cannot continue across D3D12 work
    if (!c->batches.empty()) g.pending.erase(c->batches.back()->cmd);
    if (g_splitLog) c->trace += " ||";
    c->batches.push_back(b);
    c->parts.push_back(p);
    c->target.store(p.list, std::memory_order_release);
    c->dirty.store(false);
    return b;
}

// ------------------------------------------------------------------------------------------------ list hooks
using CloseFn = HRESULT(STDMETHODCALLTYPE*)(List*);
using ResetFn = HRESULT(STDMETHODCALLTYPE*)(List*, ID3D12CommandAllocator*, ID3D12PipelineState*);
using ExecuteFn = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
HRESULT o_close(List* l) { return orig<CloseFn>(l, 9)(l); }
HRESULT o_reset(List* l, ID3D12CommandAllocator* a, ID3D12PipelineState* p) { return orig<ResetFn>(l, 10)(l, a, p); }
void o_execute(ID3D12CommandQueue* q, UINT n, ID3D12CommandList* const* lists)
{
    reinterpret_cast<ExecuteFn>(hookedVt(g_queueVts, g_nQueueVts.load(std::memory_order_acquire), q)->orig[10])(q, n, lists);
}

HRESULT STDMETHODCALLTYPE hClose(List* list)
{
    if (g_activeChains.load() != 0) {
        std::lock_guard<std::recursive_mutex> l(g_lock);
        if (Chain* c = findChain(list)) {
            if (c->parts.empty()) {
                finishChain(c);   // tracked during an evaluate without launches: nothing to split
            } else if (!c->closed) {
                HRESULT worst = S_OK;
                for (auto& p : c->parts) {
                    HRESULT r = o_close(p.list);
                    if (FAILED(r)) worst = r;
                }
                for (Batch* b : c->batches)
                    endBatch(b);
                c->closed = true;
                c->target.store(nullptr);
                HRESULT r = o_close(list);
                return FAILED(worst) ? worst : r;
            }
        }
    }
    return o_close(list);
}

void discardChain(Chain* c)
{
    for (auto& p : c->parts) {
        if (!c->closed) o_close(p.list);
        g.freeParts.push_back(p);
    }
    for (Batch* b : c->batches) freeBatch(b);
    finishChain(c);
}

HRESULT STDMETHODCALLTYPE hReset(List* list, ID3D12CommandAllocator* a, ID3D12PipelineState* p)
{
    if (g_activeChains.load() != 0) {
        std::lock_guard<std::recursive_mutex> l(g_lock);
        if (Chain* c = findChain(list)) discardChain(c);
    }
    return o_reset(list, a, p);
}

QueueSync* syncFor(ID3D12CommandQueue* q)
{
    auto it = g.syncs.find(q);
    if (it != g.syncs.end()) return &it->second;
    QueueSync& s = g.syncs[q];
    ID3D12Device* d3d = nullptr;
    q->GetDevice(__uuidof(ID3D12Device), (void**)&d3d);
    if (!d3d || FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&s.fence))) {
        if (d3d) d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&s.fence);
    }
    s.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (g.semImport && s.fence) {
        HANDLE h = nullptr;
        if (SUCCEEDED(d3d->CreateSharedHandle(s.fence, nullptr, GENERIC_ALL, nullptr, &h))) {
            VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
            if (vk.CreateSemaphore(g.dev, &sci, nullptr, &s.sem) == VK_SUCCESS) {
                VkImportSemaphoreWin32HandleInfoKHR ii{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
                ii.semaphore = s.sem;
                ii.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
                ii.handle = h;
                VkResult r = vk.ImportSemaphoreWin32HandleKHR(g.dev, &ii);
                s.gpu = r == VK_SUCCESS;
                if (!s.gpu) logf("D3D12 fence import failed (%d): CPU hand-offs", r);
            }
            CloseHandle(h);
        }
    }
    logf("queue %p (type %d): hand-off fence %p, %s", (void*)q, (int)q->GetDesc().Type, (void*)s.fence,
         s.gpu ? "GPU waits" : "CPU waits");
    if (d3d) d3d->Release();
    return &s;
}

// [Host] KernelTiming: a one-command D3D12 list writing timestamp `idx` (and resolving the pair when `resolve`)
bool d3dStamp(ID3D12CommandQueue* q, uint32_t idx, bool resolve, Part& out)
{
    if (!g.d3dTsHeap) {
        D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 4096};
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 4096 * 8;
        rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(g.d3d->CreateQueryHeap(&qd, __uuidof(ID3D12QueryHeap), (void**)&g.d3dTsHeap)) ||
            FAILED(g.d3d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                  __uuidof(ID3D12Resource), (void**)&g.d3dTsRb)) ||
            FAILED(g.d3dTsRb->Map(0, nullptr, (void**)&g.d3dTsMap))) {
            g.d3dTsMap = nullptr;
            return false;
        }
    }
    if (!g.d3dTsMap) return false;
    if (!g.d3dTsMsPerTick.count(q)) {
        UINT64 f = 0;
        q->GetTimestampFrequency(&f);
        g.d3dTsMsPerTick[q] = f ? 1000.0 / (double)f : 0;
    }
    Chain tmp;
    tmp.type = q->GetDesc().Type;
    if (!newPart(&tmp, g.d3d, out)) return false;
    out.list->EndQuery(g.d3dTsHeap, D3D12_QUERY_TYPE_TIMESTAMP, idx);
    if (resolve) out.list->ResolveQueryData(g.d3dTsHeap, D3D12_QUERY_TYPE_TIMESTAMP, idx - 1, 2, g.d3dTsRb, (idx - 1) * 8ull);
    return SUCCEEDED(out.list->Close());
}

// runs batch b between the D3D12 work submitted so far and what follows on queue q; g_lock held
void runBatch(ID3D12CommandQueue* q, QueueSync* s, Batch* b)
{
    endBatch(b);
    uint64_t a = ++s->value;
    q->Signal(s->fence, a);
    uint64_t c = ++s->value;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &b->cmd;
    if (s->gpu) {
        VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ts.waitSemaphoreValueCount = 1;
        ts.pWaitSemaphoreValues = &a;
        ts.signalSemaphoreValueCount = 1;
        ts.pSignalSemaphoreValues = &c;
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        si.pNext = &ts;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &s->sem;
        si.pWaitDstStageMask = &stage;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &s->sem;
        VkResult r = vk.QueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) { logf("vkQueueSubmit failed (%d)", r); s->fence->Signal(c); }
    } else {
        s->fence->SetEventOnCompletion(a, s->event);
        WaitForSingleObject(s->event, 5000);
        vk.ResetFences(g.dev, 1, &g.cpuFence);
        VkResult r = vk.QueueSubmit(g.queue, 1, &si, g.cpuFence);
        if (r == VK_SUCCESS) vk.WaitForFences(g.dev, 1, &g.cpuFence, VK_TRUE, 5000000000ull);
        else logf("vkQueueSubmit failed (%d)", r);
        s->fence->Signal(c);
    }
    q->Wait(s->fence, c);
    b->qs = s;
    g.batchesSubmitted++;
}

void STDMETHODCALLTYPE hExecute(ID3D12CommandQueue* q, UINT n, ID3D12CommandList* const* lists)
{
    if (g_activeChains.load() == 0) return o_execute(q, n, lists);
    std::lock_guard<std::recursive_mutex> l(g_lock);
    bool any = false;
    for (UINT i = 0; i < n && !any; i++) {
        Chain* c = findChain(lists[i]);
        any = c && c->closed && !c->parts.empty();
    }
    if (!any) return o_execute(q, n, lists);
    QueueSync* s = syncFor(q);
    std::vector<ID3D12CommandList*> chunk;
    std::vector<Batch*> batches;
    std::vector<Part> parts;
    for (UINT i = 0; i < n; i++) {
        chunk.push_back(lists[i]);
        Chain* c = findChain(lists[i]);
        if (!c || !c->closed || c->parts.empty()) continue;
        for (size_t k = 0; k < c->parts.size(); k++) {
            o_execute(q, (UINT)chunk.size(), chunk.data());
            Batch* b = c->batches[k];
            Part t0{}, t1{};
            bool timed = g.tsPool && d3dStamp(q, (g.d3dTsNext % 2048) * 2, false, t0);
            if (timed) {
                ID3D12CommandList* l0 = t0.list;
                o_execute(q, 1, &l0);
                parts.push_back(t0);
            }
            runBatch(q, s, b);
            if (timed && d3dStamp(q, (g.d3dTsNext % 2048) * 2 + 1, true, t1)) {
                ID3D12CommandList* l1 = t1.list;
                o_execute(q, 1, &l1);
                parts.push_back(t1);
                b->d3dTs = (int)((g.d3dTsNext % 2048) * 2);
                b->d3dQ = q;
                g.d3dTsNext++;
            }
            chunk.assign(1, c->parts[k].list);
        }
        if (g_splitLog) {
            static int tl = 0;
            if (g.launchCount > 3000 && tl < 8) { tl++; logf("SPLITLOG list %p: %s", (void*)lists[i], c->trace.c_str()); }
        }
        static int logged = 0;
        if (logged < 3) {
            logged++;
            std::string calls;
            for (int k = 11; k < 86; k++)
                if (c->slotCalls[k]) calls += " " + std::to_string(k) + "x" + std::to_string(c->slotCalls[k]);
            logf("list %p executed in %zu parts with %zu kernel batches; recording slots:%s", (void*)lists[i],
                 c->parts.size() + 1, c->batches.size(), calls.c_str());
        }
        batches.insert(batches.end(), c->batches.begin(), c->batches.end());
        parts.insert(parts.end(), c->parts.begin(), c->parts.end());
        finishChain(c);
    }
    o_execute(q, (UINT)chunk.size(), chunk.data());
    uint64_t done = ++s->value;
    q->Signal(s->fence, done);
    for (Batch* b : batches) { b->done = done; g.inflightBatches.push_back(b); }
    for (Part& p : parts) { p.qs = s; p.done = done; g.inflightParts.push_back(p); }
}

template <bool Float> void STDMETHODCALLTYPE sClearUav(List* self, D3D12_GPU_DESCRIPTOR_HANDLE gpu, D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                                                      ID3D12Resource* res, const void* values, UINT nrects, const D3D12_RECT* rects);
void STDMETHODCALLTYPE sResolveQueryData(List* self, ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT start, UINT n,
                                         ID3D12Resource* dst, UINT64 off);
void STDMETHODCALLTYPE sCopyResource(List* self, ID3D12Resource* dst, ID3D12Resource* src);

std::mutex g_hookLock;

// hooks the class of `list` (once per class): Close, Reset and every recording method
bool hookListClass(List* list)
{
    std::lock_guard<std::mutex> l(g_hookLock);
    void** vt = *reinterpret_cast<void***>(list);
    if (hookedVt(g_listVts, g_nListVts.load(), list)) return true;
    int n = g_nListVts.load();
    if (n >= 16) { logf("too many command list classes"); return false; }
    static const GUID list10 = {0x7013c015, 0xd161, 0x4b63, {0xa0, 0x8c, 0x23, 0x85, 0x52, 0xdd, 0x8a, 0xcc}};
    static const GUID list9 = {0x34ed2808, 0xffe6, 0x4c2b, {0xb1, 0x1a, 0xca, 0xbd, 0x2b, 0x0c, 0x59, 0xe1}};
    static const GUID list8 = {0xee936ef9, 0x599d, 0x4d28, {0x93, 0x8e, 0x23, 0xc4, 0xad, 0x05, 0xce, 0x51}};
    const struct { const GUID* iid; size_t slots; } versions[] = {
        {&list10, 86}, {&list9, 84}, {&list8, 82}, {&__uuidof(ID3D12GraphicsCommandList7), 81},
        {&__uuidof(ID3D12GraphicsCommandList6), 80}, {&__uuidof(ID3D12GraphicsCommandList5), 79},
        {&__uuidof(ID3D12GraphicsCommandList4), 77}, {&__uuidof(ID3D12GraphicsCommandList3), 68},
        {&__uuidof(ID3D12GraphicsCommandList2), 67}, {&__uuidof(ID3D12GraphicsCommandList1), 66}};
    size_t slots = 60;
    for (const auto& v : versions) {
        IUnknown* newer = nullptr;
        if (SUCCEEDED(list->QueryInterface(*v.iid, (void**)&newer))) { newer->Release(); slots = v.slots; break; }
    }
    HookedVt& h = g_listVts[n];
    // the originals are published before any slot points at a stub
    for (size_t s = 9; s < slots; s++) h.orig[s] = vt[s];
    h.vt.store(vt, std::memory_order_release);
    g_nListVts.store(n + 1, std::memory_order_release);
    bool ok = true;
    for (size_t s = 9; ok && s < slots; s++) {
        void* hook = s >= 11 ? cux_split_thunks[s] : nullptr;
        switch (s) {
        case 9: hook = (void*)&hClose; break;
        case 10: hook = (void*)&hReset; break;
        case 25: hook = (void*)&sSetPipelineState; break;
        case 26: hook = (void*)&sResourceBarrier; break;
        case 28: hook = (void*)&sSetDescriptorHeaps; break;
        case 29: hook = (void*)&sSetRootSignature<0>; break;
        case 30: hook = (void*)&sSetRootSignature<1>; break;
        case 31: hook = (void*)&sSetTable<0>; break;
        case 32: hook = (void*)&sSetTable<1>; break;
        case 33: hook = (void*)&sSetConstant<0>; break;
        case 34: hook = (void*)&sSetConstant<1>; break;
        case 35: hook = (void*)&sSetConstants<0>; break;
        case 36: hook = (void*)&sSetConstants<1>; break;
        case 37: hook = (void*)&sSetView<0, 3>; break;
        case 38: hook = (void*)&sSetView<1, 3>; break;
        case 39: hook = (void*)&sSetView<0, 4>; break;
        case 40: hook = (void*)&sSetView<1, 4>; break;
        case 41: hook = (void*)&sSetView<0, 5>; break;
        case 42: hook = (void*)&sSetView<1, 5>; break;
        case 17: hook = (void*)&sCopyResource; break;
        case 49: hook = (void*)&sClearUav<false>; break;
        case 50: hook = (void*)&sClearUav<true>; break;
        case 54: hook = (void*)&sResolveQueryData; break;
        default: break;
        }
        void* unused = nullptr;
        ok = patchSlot(vt, s, hook, &unused);
    }
    logf("command list class %p (type %d): hooks %s (%zu slots)", (void*)vt, (int)list->GetType(),
         ok ? "installed" : "FAILED", slots);
    return ok;
}

bool hookQueueClass(ID3D12CommandQueue* q)
{
    std::lock_guard<std::mutex> l(g_hookLock);
    if (hookedVt(g_queueVts, g_nQueueVts.load(), q)) return true;
    int n = g_nQueueVts.load();
    if (n >= 16) return false;
    void** vt = *reinterpret_cast<void***>(q);
    HookedVt& h = g_queueVts[n];
    h.orig[10] = vt[10];
    h.vt.store(vt, std::memory_order_release);
    g_nQueueVts.store(n + 1, std::memory_order_release);
    void* unused = nullptr;
    bool ok = patchSlot(vt, 10, (void*)&hExecute, &unused);
    logf("command queue class %p (type %d): ExecuteCommandLists hook %s", (void*)vt, (int)q->GetDesc().Type,
         ok ? "installed" : "FAILED");
    return ok;
}

// direct and compute lists and queues of `d3d` (the classes DLSS-G records into and submits on)
bool installListHooks(ID3D12Device* d3d)
{
    if (g_listHooks) return true;
    bool ok = true;
    for (D3D12_COMMAND_LIST_TYPE type : {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE}) {
        ID3D12CommandQueue* queue = nullptr;
        ID3D12CommandAllocator* alloc = nullptr;
        List* list = nullptr;
        D3D12_COMMAND_QUEUE_DESC qd{type};
        bool made = SUCCEEDED(d3d->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue)) &&
                    SUCCEEDED(d3d->CreateCommandAllocator(type, __uuidof(ID3D12CommandAllocator), (void**)&alloc)) &&
                    SUCCEEDED(d3d->CreateCommandList(0, type, alloc, nullptr, __uuidof(List), (void**)&list));
        ok = ok && made && hookListClass(list) && hookQueueClass(queue);
        if (list) { list->Close(); list->Release(); }
        if (alloc) alloc->Release();
        if (queue) queue->Release();
    }
    g_listHooks = ok;
    return ok;
}

// ------------------------------------------------------------------------------------------------ launches
bool ringTake(std::vector<uint32_t>& free, std::vector<uint32_t> Exec::CmdSlots::*which, VkCommandBuffer cmd, uint32_t& slot)
{
    if (free.empty()) return false;
    slot = free.back();
    free.pop_back();
    (g.cmdSlots[cmd].*which).push_back(slot);
    return true;
}

void writeMember(uint8_t* reg, uint32_t k, const uint8_t* pbuf, uint32_t psize, const uint32_t grid[3], const uint32_t block[3],
                 uint32_t alt)
{
    uint8_t slot[256] = {};
    memcpy(slot, pbuf, std::min<uint32_t>(psize, 256));
    memcpy(reg + 256 + (size_t)k * 256, slot, 256);
    uint32_t gr[4] = {grid[0], grid[1], grid[2], 0};
    memcpy(reg + kGrpGridOff + (size_t)k * 16, gr, 16);
    uint32_t bi[4] = {block[0], block[1], block[2], alt};
    memcpy(reg + kGrpInfoOff + (size_t)k * 16, bi, 16);
}

int memberAlt(const std::string& member, const std::string& gkey)
{
    size_t p = 0;
    for (int i = 0;; i++) {
        size_t q = member.find('|', p);
        size_t e = q == std::string::npos ? member.size() : q;
        if (e - p == gkey.size() && member.compare(p, e - p, gkey) == 0) return i;
        if (q == std::string::npos) return -1;
        p = q + 1;
    }
}

void serialize(VkCommandBuffer cmd)
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                          nullptr, 0, nullptr);
}

bool launchGroup(VkCommandBuffer cmd, CuFunction& fn, const uint32_t grid[3], const uint32_t block[3], const uint8_t* pbuf,
                 uint32_t psize)
{
    auto pit = g.pending.find(cmd);
    if (pit != g.pending.end()) {
        PendingGroup& pg = pit->second;
        int alt = pbuf && psize <= 256 ? memberAlt(pg.g->members[pg.next], fn.gkey) : -1;
        if (alt >= 0 && g.launchCount == pg.last_launch + 1) {
            writeMember(pg.region, pg.next, pbuf, psize, grid, block, (uint32_t)alt);
            pg.next++;
            pg.last_launch = g.launchCount;
            *(volatile uint32_t*)pg.region = pg.next;
            if (pg.next == pg.g->members.size()) g.pending.erase(pit);
            return true;
        }
        logf("group %s: sequence broken after %u of %zu members by %s", pg.g->name.c_str(), pg.next,
             pg.g->members.size(), fn.name.c_str());
        g.pending.erase(pit);
    }
    if (!pbuf || psize > 256) return false;
    for (auto& rp : g.grpRecent) {
        volatile uint32_t* poison = (volatile uint32_t*)(g.grpMap + (size_t)rp.first * kGrpRegion + 64);
        if (!*poison) continue;
        *poison = 0;
        for (auto it = g.groupFirst.begin(); it != g.groupFirst.end();)
            it = it->second == rp.second ? g.groupFirst.erase(it) : std::next(it);
        logf("group %s: grid barrier timed out; members launch singly", rp.second->name.c_str());
    }
    auto git = g.groupFirst.find(fn.gkey);
    if (git == g.groupFirst.end()) return false;
    Group* grp = git->second;
    VkPipeline pipe = getPipeline(*grp->fn, PipeKey{grp->block[0], grp->block[1], grp->block[2], 0});
    uint32_t ri = 0;
    if (!pipe || !ringTake(g.grpFree, &Exec::CmdSlots::grp, cmd, ri)) return false;
    uint8_t* reg = g.grpMap + (size_t)ri * kGrpRegion;
    memset(reg, 0, 256);
    writeMember(reg, 0, pbuf, psize, grid, block, (uint32_t)std::max(0, memberAlt(grp->members[0], fn.gkey)));
    *(volatile uint32_t*)reg = 1;
    serialize(cmd);
    uint32_t pc[kPushBytes / 4] = {};
    VkDeviceAddress a = g.grpAddr + (VkDeviceAddress)ri * kGrpRegion;
    pc[0] = (uint32_t)a;
    pc[1] = (uint32_t)(a >> 32);
    vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk.CmdPushConstants(cmd, g.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes, pc);
    vk.CmdDispatch(cmd, grp->grid[0], grp->grid[1], grp->grid[2]);
    stamp(g.cur, &grp->name);
    grp->fn->launches++;
    g.grpRecent.push_back({ri, grp});
    if (g.grpRecent.size() > 16) g.grpRecent.pop_front();
    if (grp->members.size() > 1) g.pending[cmd] = PendingGroup{grp, 1, reg, g.launchCount};
    return true;
}

void launch(List* list, CuFunction& fn, const uint32_t grid[3], const uint32_t block[3], uint32_t smem, const uint8_t* pbuf,
            uint32_t psize)
{
    std::lock_guard<std::recursive_mutex> l(g_lock);
    fn.launches++;
    g.launchCount++;
    if (g_cfg.trace || g.launchCount <= 3)
        logf("launch #%llu %s[%.12s] grid %ux%ux%u block %ux%ux%u smem %u params %u", (unsigned long long)g.launchCount,
             fn.name.c_str(), fn.module->hash.c_str(), grid[0], grid[1], grid[2], block[0], block[1], block[2], smem, psize);
    if (!fn.meta.ok) {
        if (!fn.warned) logf("launch of %s skipped: no translated kernel", fn.name.c_str());
        fn.warned = true;
        return;
    }
    Batch* b = batchFor(list);
    if (!b) return;
    if (g_splitLog) if (Chain* tc = findChain(nativeList(list))) tc->trace += " K:" + fn.name.substr(0, 40);
    VkCommandBuffer cmd = b->cmd;
    b->launches++;
    g.cur = b;
    if (!g.groups.empty() && launchGroup(cmd, fn, grid, block, pbuf, psize)) return;
    PipeKey k{block[0], block[1], block[2], smem};
    uint32_t gx = grid[0], gy = grid[1], gz = grid[2];
    const KernelMeta& m = fn.meta;
    if (m.t1_dispatch) {
        k = PipeKey{m.t1_block[0], m.t1_block[1], m.t1_block[2], 0};
        gx = (uint32_t)(((uint64_t)gx * m.t1_num[0] + m.t1_den[0] - 1) / m.t1_den[0]);
        gy = (uint32_t)(((uint64_t)gy * m.t1_num[1] + m.t1_den[1] - 1) / m.t1_den[1]);
        gz = (uint32_t)(((uint64_t)gz * m.t1_num[2] + m.t1_den[2] - 1) / m.t1_den[2]);
        if (m.t1_fixed[0] && m.t1_fixed[1] && m.t1_fixed[2]) gx = m.t1_fixed[0], gy = m.t1_fixed[1], gz = m.t1_fixed[2];
    }
    VkPipeline pipe = getPipeline(fn, k);
    if (!pipe) return;
    serialize(cmd);
    uint8_t pc[kPushBytes] = {};
    if (pbuf && m.big_params) {
        if (!g.bpMap && !g.bpFailed) {
            void* ptr = nullptr;
            g.bpFailed = !hostBuffer((VkDeviceSize)kBpSlots * kBpSlot, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ptr, g.bpAddr);
            g.bpMap = g.bpFailed ? nullptr : (uint8_t*)ptr;
            for (uint32_t i = kBpSlots; i-- > 0;) g.bpFree.push_back(i);
        }
        uint32_t si = 0;
        if (!g.bpMap || psize > kBpSlot || !ringTake(g.bpFree, &Exec::CmdSlots::bp, cmd, si)) {
            logf("launch of %s skipped: no big parameter slot", fn.name.c_str());
            return;
        }
        memcpy(g.bpMap + (size_t)si * kBpSlot, pbuf, psize);
        VkDeviceAddress a = g.bpAddr + (VkDeviceAddress)si * kBpSlot;
        memcpy(pc + 240, &a, sizeof a);
    } else if (pbuf) {
        memcpy(pc, pbuf, std::min<uint32_t>(psize, 248));
    }
    vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk.CmdPushConstants(cmd, g.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes, pc);
    vk.CmdDispatch(cmd, gx, gy, gz);
    stamp(b, &fn.name);
}

// ------------------------------------------------------------------------------------------------ clears, resolves
// A ClearUnorderedAccessView* between two launches used to end the kernel batch: the clear ran on the D3D12 queue
// between two hand-offs (each ~0.15 ms of idle GPU on RDNA3). When a batch is open and the clear covers one whole mip
// of a 2D surface DLSS-G created, it runs instead as a dispatch in the batch (clear_uav.comp), through the same raw
// D3D12 descriptor the kernels write through, at the same point in the order. Anything else stays on D3D12.
// 0: float / unorm / snorm, 1: uint, 2: sint, -1: not handled here
int clearKind(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R11G11B10_FLOAT: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16G16_SNORM: case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_SNORM: case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_SNORM: case DXGI_FORMAT_B8G8R8A8_UNORM:
        return 0;
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R10G10B10A2_UINT: case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R8G8_UINT: case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R8_UINT:
        return 1;
    case DXGI_FORMAT_R32G32B32A32_SINT: case DXGI_FORMAT_R16G16B16A16_SINT: case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R8G8B8A8_SINT: case DXGI_FORMAT_R16G16_SINT: case DXGI_FORMAT_R32_SINT: case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_SINT: case DXGI_FORMAT_R8_SINT:
        return 2;
    default:
        return -1;
    }
}

VkPipeline clearPipeline()
{
    if (g.clearPipe || g.clearFailed) return g.clearPipe;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = sizeof kClearUavSpv;
    mi.pCode = kClearUavSpv;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.pName = "main";
    ci.layout = g.pipeLayout;
    VkResult r = vk.CreateShaderModule(g.dev, &mi, nullptr, &g.clearShader);
    if (r == VK_SUCCESS) {
        ci.stage.module = g.clearShader;
        r = vk.CreateComputePipelines(g.dev, g.pcache, 1, &ci, nullptr, &g.clearPipe);
    }
    if (r != VK_SUCCESS) {
        logf("clear pipeline creation failed (%d): clears stay on D3D12", r);
        g.clearPipe = VK_NULL_HANDLE;
        g.clearFailed = true;
    }
    return g.clearPipe;
}

// a raw (R32_TYPELESS) or 32-bit typed buffer view: its elements written through the GPU virtual address
bool recordBufferClear(Chain* c, const ViewRec& v, ID3D12Resource* res, const D3D12_RESOURCE_DESC& rd, const uint32_t vals[4],
                       bool isFloat)
{
    if (!v.hasDesc || v.uavDesc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER || v.uavDesc.Buffer.StructureByteStride) return false;
    DXGI_FORMAT f = v.uavDesc.Format;
    bool raw = (v.uavDesc.Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW) != 0;
    if (raw ? f != DXGI_FORMAT_R32_TYPELESS : !(f == DXGI_FORMAT_R32_UINT || f == DXGI_FORMAT_R32_SINT || f == DXGI_FORMAT_R32_FLOAT))
        return false;
    if (isFloat ? f != DXGI_FORMAT_R32_FLOAT : (f == DXGI_FORMAT_R32_FLOAT && vals[0] != 0)) return false;
    uint64_t first = v.uavDesc.Buffer.FirstElement, n = v.uavDesc.Buffer.NumElements;
    if (!n || n > 0xffffffffull || (first + n) * 4 > rd.Width) return false;
    VkPipeline pipe = clearPipeline();
    if (!pipe) return false;
    uint64_t addr = res->GetGPUVirtualAddress() + first * 4;
    Batch* b = c->batches.back();
    g.pending.erase(b->cmd);   // a launch group open in this batch cannot take members after the clear
    serialize(b->cmd);
    uint32_t pc[8] = {(uint32_t)addr, (uint32_t)(addr >> 32), (uint32_t)n, 3, vals[0], 0, 0, 0};
    vk.CmdBindPipeline(b->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk.CmdPushConstants(b->cmd, g.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    uint32_t groups = (uint32_t)((n + 63) / 64), gx = std::min<uint32_t>(groups, 65535), gy = (groups + gx - 1) / gx;
    vk.CmdDispatch(b->cmd, gx, gy, 1);
    stamp(b, &kLabelClear);
    if (g.vkClears++ < 8)
        logf("clear (%s) of buffer %p, %llu elements from %llu, on the kernels' queue", isFloat ? "float" : "uint",
             (void*)res, (unsigned long long)n, (unsigned long long)first);
    return true;
}

// records the clear into the open kernel batch of c, or false (then it goes to D3D12 and ends the batch); g_lock held
bool recordClear(Chain* c, D3D12_CPU_DESCRIPTOR_HANDLE cpu, ID3D12Resource* res, const void* values, bool isFloat)
{
    if (c->closed || c->batches.empty() || c->dirty.load() || !res) return false;
    ViewRec v;
    if (!lookupView(cpu.ptr, v) || !v.uav || v.res != res) return false;
    D3D12_RESOURCE_DESC rd = res->GetDesc();
    uint32_t vals[4];
    memcpy(vals, values, sizeof vals);
    if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return recordBufferClear(c, v, res, rd, vals, isFloat);
    if (rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || rd.SampleDesc.Count != 1) return false;
    if (v.hasDesc && (v.uavDesc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2D || v.uavDesc.Texture2D.PlaneSlice != 0))
        return false;
    if (!v.hasDesc && rd.DepthOrArraySize != 1) return false;
    DXGI_FORMAT fmt = v.hasDesc && v.uavDesc.Format != DXGI_FORMAT_UNKNOWN ? v.uavDesc.Format : rd.Format;
    int kind = clearKind(fmt);
    if (kind < 0 || vkFormat(fmt) == VK_FORMAT_UNDEFINED) return false;
    uint32_t mode;
    if (isFloat) {
        if (kind != 0) return false;   // float values converted to integers: left to D3D12
        mode = 0;
    } else if (kind == 0) {
        // raw bits into a non-integer format: only zero, which is 0.0 in every float / unorm / snorm format
        if (vals[0] | vals[1] | vals[2] | vals[3]) return false;
        mode = 0;
    } else {
        mode = (uint32_t)kind;
    }
    static std::map<DXGI_FORMAT, bool> storable;
    auto st = storable.find(fmt);
    if (st == storable.end()) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{fmt};
        bool ok = SUCCEEDED(g.d3d->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof fs)) &&
                  (fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
        st = storable.emplace(fmt, ok).first;
    }
    if (!st->second) return false;
    VkPipeline pipe = clearPipeline();
    if (!pipe) return false;
    uint32_t slot = viewSlot(cpu, true);
    if (!slot) return false;
    uint32_t mip = v.hasDesc ? v.uavDesc.Texture2D.MipSlice : 0;
    uint32_t w = std::max<uint32_t>(1, (uint32_t)(rd.Width >> mip)), h = std::max<uint32_t>(1, rd.Height >> mip);
    Batch* b = c->batches.back();
    g.pending.erase(b->cmd);   // a launch group open in this batch cannot take members after the clear
    serialize(b->cmd);
    uint32_t pc[8] = {slot, w, h, mode, vals[0], vals[1], vals[2], vals[3]};
    vk.CmdBindPipeline(b->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk.CmdPushConstants(b->cmd, g.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vk.CmdDispatch(b->cmd, (w + 7) / 8, (h + 7) / 8, 1);
    stamp(b, &kLabelClear);
    if (g.vkClears++ < 4)
        logf("clear (%s) of %p %ux%u format %d on the kernels' queue, slot %u", isFloat ? "float" : "uint", (void*)res, w,
             h, (int)fmt, slot);
    return true;
}

template <bool Float>
void STDMETHODCALLTYPE sClearUav(List* self, D3D12_GPU_DESCRIPTOR_HANDLE gpu, D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                                 ID3D12Resource* res, const void* values, UINT nrects, const D3D12_RECT* rects)
{
    constexpr unsigned kSlot = Float ? 50 : 49;
    List* t;
    if (Chain* c = stateChain(self, t)) {
        c->slotCalls[kSlot]++;
        if (g_cfg.vkClears && nrects == 0 && values) {
            std::lock_guard<std::recursive_mutex> l(g_lock);
            if (recordClear(c, cpu, res, values, Float)) {
                if (g_splitLog) c->trace += Float ? " C50" : " C49";
                return;
            }
        }
        if (g_splitLog) c->trace += Float ? " s50" : " s49";
        c->dirty.store(true, std::memory_order_relaxed);
    }
    using Fn = void(STDMETHODCALLTYPE*)(List*, D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE, ID3D12Resource*,
                                        const void*, UINT, const D3D12_RECT*);
    orig<Fn>(t, kSlot)(t, gpu, cpu, res, values, nrects, rects);
}

// a resolve into CPU-read memory (a timer's readback buffer) touches nothing a kernel reads: it does not end the batch
void STDMETHODCALLTYPE sResolveQueryData(List* self, ID3D12QueryHeap* heap, D3D12_QUERY_TYPE type, UINT start, UINT n,
                                         ID3D12Resource* dst, UINT64 off)
{
    List* t;
    if (Chain* c = stateChain(self, t)) {
        c->slotCalls[54]++;
        bool readback = false;
        D3D12_HEAP_PROPERTIES hp{};
        if (!g_cfg.splitQueries && dst && SUCCEEDED(dst->GetHeapProperties(&hp, nullptr)))
            readback = hp.Type == D3D12_HEAP_TYPE_READBACK ||
                       (hp.Type == D3D12_HEAP_TYPE_CUSTOM && hp.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK);
        if (g_splitLog) c->trace += readback ? " R54" : " s54";
        if (!readback) c->dirty.store(true, std::memory_order_relaxed);
    }
    using Fn = void(STDMETHODCALLTYPE*)(List*, ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT, UINT, ID3D12Resource*, UINT64);
    orig<Fn>(t, 54)(t, heap, type, start, n, dst, off);
}

// ------------------------------------------------------------------------------------------------ copies
// DLSS-G copies its real frame into OutputReal when the caller provides one (Streamline does, at MultiFrameIndex 1):
// a CopyResource between two launches, which ended the batch (one more D3D12 <-> Vulkan round trip per real frame).
// With a batch open it runs as a dispatch in the batch instead (copy_res.comp), for buffers (32-bit words through
// the GPU virtual addresses) and for 2D textures in formats whose texel round trip is exact: UNORM, UINT, SINT.
// 0 unorm, 1 uint, 2 sint, -1 not here
int copyKind(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        return 0;
    default:
        break;
    }
    int k = clearKind(f);
    return k == 1 || k == 2 ? k : -1;
}

VkPipeline copyPipeline()
{
    if (g.copyPipe || g.copyFailed) return g.copyPipe;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = sizeof kCopyResSpv;
    mi.pCode = kCopyResSpv;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.pName = "main";
    ci.layout = g.pipeLayout;
    VkResult r = vk.CreateShaderModule(g.dev, &mi, nullptr, &g.copyShader);
    if (r == VK_SUCCESS) {
        ci.stage.module = g.copyShader;
        r = vk.CreateComputePipelines(g.dev, g.pcache, 1, &ci, nullptr, &g.copyPipe);
    }
    if (r != VK_SUCCESS) {
        logf("copy pipeline creation failed (%d): copies stay on D3D12", r);
        g.copyPipe = VK_NULL_HANDLE;
        g.copyFailed = true;
    }
    return g.copyPipe;
}

// a descriptor of `res` in the scratch heap, as a slot of the bindless set
uint32_t scratchView(ID3D12Resource* res, DXGI_FORMAT fmt, bool uav)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h = g.scratchHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += (SIZE_T)(g.scratchNext++ % 4096) * g.scratchInc;
    if (uav) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = fmt;
        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        g.d3d->CreateUnorderedAccessView(res, nullptr, &u, h);
    } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC v{};
        v.Format = fmt;
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        v.Texture2D.MipLevels = 1;
        g.d3d->CreateShaderResourceView(res, &v, h);
    }
    return viewSlot(h, uav);
}

// records CopyResource(dst, src) into the open kernel batch of c, or false (D3D12 then; it ends the batch); g_lock held
bool recordCopy(Chain* c, ID3D12Resource* dst, ID3D12Resource* src)
{
    if (c->closed || c->batches.empty() || c->dirty.load() || !dst || !src || dst == src || !g.scratchHeap) return false;
    D3D12_RESOURCE_DESC d = dst->GetDesc(), s = src->GetDesc();
    if (d.Dimension != s.Dimension || d.Width != s.Width || d.Height != s.Height || d.DepthOrArraySize != s.DepthOrArraySize ||
        d.MipLevels != s.MipLevels || d.Format != s.Format || d.SampleDesc.Count != 1 || s.SampleDesc.Count != 1)
        return false;
    VkPipeline pipe = copyPipeline();
    if (!pipe) return false;
    uint32_t pc[7] = {};
    uint32_t gx, gy;
    if (d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        if (d.Width % 4 || d.Width / 4 > 0xffffffffull) return false;
        uint64_t da = dst->GetGPUVirtualAddress(), sa = src->GetGPUVirtualAddress(), n = d.Width / 4;
        uint32_t groups = (uint32_t)((n + 63) / 64);
        gx = std::min<uint32_t>(groups, 65535);
        gy = (groups + gx - 1) / gx;
        uint32_t v[7] = {(uint32_t)da, (uint32_t)(da >> 32), 0, (uint32_t)n, 3, (uint32_t)sa, (uint32_t)(sa >> 32)};
        memcpy(pc, v, sizeof pc);
    } else {
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.DepthOrArraySize != 1 || d.MipLevels != 1) return false;
        if (!(d.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || (s.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
            return false;
        int kind = copyKind(d.Format);
        if (kind < 0) return false;
        static std::map<DXGI_FORMAT, bool> storable;
        auto st = storable.find(d.Format);
        if (st == storable.end()) {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{d.Format};
            bool ok = SUCCEEDED(g.d3d->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof fs)) &&
                      (fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
            st = storable.emplace(d.Format, ok).first;
        }
        if (!st->second) return false;
        uint32_t ss = scratchView(src, s.Format, false), ds = scratchView(dst, d.Format, true);
        if (!ss || !ds) return false;
        uint32_t w = (uint32_t)d.Width, h = d.Height;
        gx = (w + 7) / 8;
        gy = (h + 7) / 8;
        uint32_t v[7] = {ss, ds, w, h, (uint32_t)kind, 0, 0};
        memcpy(pc, v, sizeof pc);
    }
    Batch* b = c->batches.back();
    g.pending.erase(b->cmd);   // a launch group open in this batch cannot take members after the copy
    serialize(b->cmd);
    vk.CmdBindPipeline(b->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk.CmdPushConstants(b->cmd, g.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vk.CmdDispatch(b->cmd, gx, gy, 1);
    static const std::string kLabelCopy = "(copy)";
    stamp(b, &kLabelCopy);
    if (g.vkCopies++ < 4)
        logf("copy of %p into %p (%llux%u format %d) on the kernels' queue", (void*)src, (void*)dst,
             (unsigned long long)d.Width, d.Height, (int)d.Format);
    return true;
}

void STDMETHODCALLTYPE sCopyResource(List* self, ID3D12Resource* dst, ID3D12Resource* src)
{
    List* t;
    if (Chain* c = stateChain(self, t)) {
        c->slotCalls[17]++;
        if (g_cfg.vkCopies) {
            std::lock_guard<std::recursive_mutex> l(g_lock);
            if (recordCopy(c, dst, src)) {
                if (g_splitLog) c->trace += " C17";
                return;
            }
        }
        if (g_splitLog) c->trace += " s17";
        c->dirty.store(true, std::memory_order_relaxed);
    }
    orig<void(STDMETHODCALLTYPE*)(List*, ID3D12Resource*, ID3D12Resource*)>(t, 17)(t, dst, src);
}

// ------------------------------------------------------------------------------------------------ modules
CuModule* parseModule(const void* data, size_t size)
{
    std::vector<FatbinEntry> entries;
    if (!nvxrt::parse_cuda_module(data, size, entries)) {
        logf("module: unrecognised (%zu bytes)", size);
        return nullptr;
    }
    const FatbinEntry* best = nullptr;
    for (auto& e : entries)
        if (e.kind == 1 && e.sm <= 120 && (!best || e.sm > best->sm)) best = &e;
    if (!best) {
        logf("module: no PTX (%zu bytes, SASS only)", size);
        return nullptr;
    }
    std::string ptx(best->data.begin(), best->data.end());
    std::string norm = nvxrt::normalize_ptx(ptx);
    std::string hash = nvxrt::sha256_hex(norm.data(), norm.size());
    for (auto& m : g.modules)
        if (m->hash == hash) return m.get();
    auto mod = std::make_unique<CuModule>();
    mod->hash = hash;
    mod->entries = nvxrt::ptx_entries(ptx);
    std::string names;
    for (auto& n : mod->entries) names += " " + n;
    logf("module %.16s (sm_%u):%s", hash.c_str(), best->sm, names.c_str());
    g.modules.push_back(std::move(mod));
    return g.modules.back().get();
}

CuFunction* function(CuModule* mod, const char* name)
{
    std::string n = name ? name : (mod->entries.empty() ? "" : mod->entries[0]);
    for (auto& f : g.functions)
        if (f->module == mod && f->name == n) return f.get();
    if (std::find(mod->entries.begin(), mod->entries.end(), n) == mod->entries.end()) {
        logf("function %s not in module %.16s", n.c_str(), mod->hash.c_str());
        return nullptr;
    }
    auto fn = std::make_unique<CuFunction>();
    fn->module = mod;
    fn->name = n;
    fn->gkey = mod->hash + ":" + n;
    bool ok = loadKernel(*fn);
    logf("function %s (%.16s): %s", n.c_str(), mod->hash.c_str(), ok ? "translated" : "NOT AVAILABLE");
    g.functions.push_back(std::move(fn));
    return g.functions.back().get();
}

// ------------------------------------------------------------------------------------------------ NVAPI
using NvStatus = int;
const NvStatus kNvOk = 0, kNvError = -1, kNvInvalid = -5;
using NvHandle = void*;

NvStatus createShader(ID3D12Device* dev, const void* cubin, uint32_t size, uint32_t bx, uint32_t by, uint32_t bz,
                      uint32_t smem, const char* name, NvHandle* out)
{
    if (!out || !cubin) return kNvInvalid;
    std::lock_guard<std::recursive_mutex> l(g_lock);
    if (!ready(dev)) return kNvError;
    installListHooks(dev);
    CuModule* mod = parseModule(cubin, size);
    CuFunction* fn = mod ? function(mod, name) : nullptr;
    if (!fn) return kNvError;
    auto sh = std::make_unique<Shader>();
    sh->fn = fn;
    sh->block[0] = bx ? bx : 1, sh->block[1] = by ? by : 1, sh->block[2] = bz ? bz : 1;
    sh->smem = smem;
    if (fn->meta.ok) {
        // compile now, not at the first launch inside a frame
        const KernelMeta& m = fn->meta;
        getPipeline(*fn, m.t1_dispatch ? PipeKey{m.t1_block[0], m.t1_block[1], m.t1_block[2], 0}
                                       : PipeKey{sh->block[0], sh->block[1], sh->block[2], smem});
        auto git = g.groupFirst.find(fn->gkey);
        if (git != g.groupFirst.end())
            getPipeline(*git->second->fn, PipeKey{git->second->block[0], git->second->block[1], git->second->block[2], 0});
    }
    *out = sh.get();
    g.shaders.push_back(std::move(sh));
    return kNvOk;
}

NvStatus __cdecl nvCreateCubinComputeShader(ID3D12Device* d, const void* c, uint32_t n, uint32_t bx, uint32_t by, uint32_t bz, NvHandle* o)
{
    return createShader(d, c, n, bx, by, bz, 0, nullptr, o);
}
NvStatus __cdecl nvCreateCubinComputeShaderEx(ID3D12Device* d, const void* c, uint32_t n, uint32_t bx, uint32_t by, uint32_t bz,
                                              uint32_t smem, const char* name, NvHandle* o)
{
    return createShader(d, c, n, bx, by, bz, smem, name, o);
}
NvStatus __cdecl nvCreateCubinComputeShaderWithName(ID3D12Device* d, const void* c, uint32_t n, uint32_t bx, uint32_t by,
                                                    uint32_t bz, const char* name, NvHandle* o)
{
    return createShader(d, c, n, bx, by, bz, 0, name, o);
}
struct CreateCubinParams {
    size_t structSizeIn, structSizeOut;
    ID3D12Device* pDevice;
    const void* pCubin;
    uint32_t size, blockX, blockY, blockZ, dynSharedMemBytes;
    const char* pShaderName;
    uint32_t flags;
    NvHandle hShader;
};
NvStatus __cdecl nvCreateCubinComputeShaderExV2(CreateCubinParams* p)
{
    if (!p) return kNvInvalid;
    return createShader(p->pDevice, p->pCubin, p->size, p->blockX, p->blockY, p->blockZ, p->dynSharedMemBytes,
                        p->pShaderName, &p->hShader);
}
NvStatus __cdecl nvDestroyCubinComputeShader(ID3D12Device*, NvHandle) { return kNvOk; }   // kept (cheap)
NvStatus __cdecl nvLaunchCubinShader(List* list, NvHandle h, uint32_t gx, uint32_t gy, uint32_t gz, const void* params,
                                     uint32_t size)
{
    if (!list || !h) return kNvInvalid;
    Shader* sh = (Shader*)h;
    uint32_t grid[3] = {gx, gy, gz};
    launch(list, *sh->fn, grid, sh->block, sh->smem, (const uint8_t*)params, size);
    return kNvOk;
}
NvStatus __cdecl nvIsFatbinPTXSupported(ID3D12Device*, bool* s)
{
    if (s) *s = true;
    return kNvOk;
}
NvStatus __cdecl nvGetCudaTextureObject(ID3D12Device* d, D3D12_CPU_DESCRIPTOR_HANDLE tex, D3D12_CPU_DESCRIPTOR_HANDLE smp, uint32_t* out)
{
    if (!out) return kNvInvalid;
    std::lock_guard<std::recursive_mutex> l(g_lock);
    if (!ready(d)) return kNvError;
    *out = viewSlot(tex, false) | (samplerSlot(smp) << 20);
    return kNvOk;
}
NvStatus __cdecl nvGetCudaSurfaceObject(ID3D12Device* d, D3D12_CPU_DESCRIPTOR_HANDLE uav, uint32_t* out)
{
    if (!out) return kNvInvalid;
    std::lock_guard<std::recursive_mutex> l(g_lock);
    if (!ready(d)) return kNvError;
    *out = viewSlot(uav, true);
    return kNvOk;
}
struct MergedParams {
    size_t structSizeIn, structSizeOut;
    ID3D12Device* pDevice;
    D3D12_CPU_DESCRIPTOR_HANDLE texDesc, smpDesc;
    uint64_t textureHandle;
};
NvStatus __cdecl nvGetCudaMergedTextureSamplerObject(MergedParams* p)
{
    if (!p) return kNvInvalid;
    uint32_t h = 0;
    NvStatus s = nvGetCudaTextureObject(p->pDevice, p->texDesc, p->smpDesc, &h);
    p->textureHandle = h;
    return s;
}
struct IndependentParams {
    size_t structSizeIn, structSizeOut;
    ID3D12Device* pDevice;
    int type;
    D3D12_CPU_DESCRIPTOR_HANDLE desc;
    uint64_t handle;
};
NvStatus __cdecl nvGetCudaIndependentDescriptorObject(IndependentParams* p)
{
    if (!p) return kNvInvalid;
    std::lock_guard<std::recursive_mutex> l(g_lock);
    if (!ready(p->pDevice)) return kNvError;
    p->handle = p->type == 0 ? viewSlot(p->desc, true) : p->type == 1 ? viewSlot(p->desc, false) : samplerSlot(p->desc) << 20;
    static bool once = false;
    if (!once) logf("GetCudaIndependentDescriptorObject used (independent texturing is untested)");
    once = true;
    return kNvOk;
}
NvStatus __cdecl nvCreateCuModule(ID3D12Device* d, const void* blob, uint32_t size, NvHandle* out)
{
    if (!out) return kNvInvalid;
    std::lock_guard<std::recursive_mutex> l(g_lock);
    if (!ready(d)) return kNvError;
    installListHooks(d);
    CuModule* m = parseModule(blob, size);
    *out = m;
    return m ? kNvOk : kNvError;
}
NvStatus __cdecl nvEnumFunctionsInModule(ID3D12Device*, NvHandle m, uint32_t* n, const char** names)
{
    if (!m || !n) return kNvInvalid;
    CuModule* mod = (CuModule*)m;
    if (names)
        for (uint32_t i = 0; i < *n && i < mod->entries.size(); i++) names[i] = mod->entries[i].c_str();
    *n = (uint32_t)mod->entries.size();
    return kNvOk;
}
NvStatus __cdecl nvCreateCuFunction(ID3D12Device*, NvHandle m, const char* name, NvHandle* out)
{
    if (!m || !out) return kNvInvalid;
    std::lock_guard<std::recursive_mutex> l(g_lock);
    CuFunction* f = function((CuModule*)m, name);
    *out = f;
    return f ? kNvOk : kNvError;
}
struct Dim3 { uint32_t x, y, z; };
struct KernelLaunch { NvHandle hFunction; Dim3 grid, block; uint32_t smem; const void* pParams; uint32_t paramSize; };
struct KernelLaunchEx { NvHandle hFunction; Dim3 grid, block; uint32_t smem; const void* pParams; uint32_t paramSize; void** kernelParams; };
NvStatus __cdecl nvLaunchCuKernelChain(List* list, const KernelLaunch* k, uint32_t n)
{
    if (!list || !k) return kNvInvalid;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t grid[3] = {k[i].grid.x, k[i].grid.y, k[i].grid.z}, block[3] = {k[i].block.x, k[i].block.y, k[i].block.z};
        launch(list, *(CuFunction*)k[i].hFunction, grid, block, k[i].smem, (const uint8_t*)k[i].pParams, k[i].paramSize);
    }
    return kNvOk;
}
NvStatus __cdecl nvLaunchCuKernelChainEx(List* list, const KernelLaunchEx* k, uint32_t n)
{
    if (!list || !k) return kNvInvalid;
    for (uint32_t i = 0; i < n; i++) {
        CuFunction& fn = *(CuFunction*)k[i].hFunction;
        uint32_t grid[3] = {k[i].grid.x, k[i].grid.y, k[i].grid.z}, block[3] = {k[i].block.x, k[i].block.y, k[i].block.z};
        std::vector<uint8_t> packed;
        const uint8_t* p = (const uint8_t*)k[i].pParams;
        uint32_t size = k[i].paramSize;
        if (!p && k[i].kernelParams) {
            packed.assign(std::max(kPushBytes, fn.meta.param_size), 0);
            for (size_t j = 0; j < fn.meta.params.size(); j++)
                if ((uint64_t)fn.meta.params[j].first + fn.meta.params[j].second <= packed.size())
                    memcpy(packed.data() + fn.meta.params[j].first, k[i].kernelParams[j], fn.meta.params[j].second);
            p = packed.data();
            size = fn.meta.param_size;
        }
        launch(list, fn, grid, block, k[i].smem, p, size);
    }
    return kNvOk;
}
NvStatus __cdecl nvDestroyCuModule(ID3D12Device*, NvHandle) { return kNvOk; }
NvStatus __cdecl nvDestroyCuFunction(ID3D12Device*, NvHandle) { return kNvOk; }
struct UavInfo { uint32_t version, surfaceHandle; uint64_t gpuVAStart, gpuVASize; };
NvStatus __cdecl nvCaptureUAVInfo(ID3D12Device*, UavInfo* info)
{
    // the next UAV's address: translated kernels never address image memory directly (as nvxrt reports 0)
    if (info) { info->surfaceHandle = 0; info->gpuVAStart = 0; info->gpuVASize = 0; }
    return kNvOk;
}

const struct { unsigned id; void* fn; const char* name; } kFuncs[] = {
    {0x2a2c79e8u, (void*)&nvCreateCubinComputeShader, "D3D12_CreateCubinComputeShader"},
    {0x3151211bu, (void*)&nvCreateCubinComputeShaderEx, "D3D12_CreateCubinComputeShaderEx"},
    {0x1dc7261fu, (void*)&nvCreateCubinComputeShaderWithName, "D3D12_CreateCubinComputeShaderWithName"},
    {0x299f5fdcu, (void*)&nvCreateCubinComputeShaderExV2, "D3D12_CreateCubinComputeShaderExV2"},
    {0x5c52bb86u, (void*)&nvLaunchCubinShader, "D3D12_LaunchCubinShader"},
    {0x7fb785bau, (void*)&nvDestroyCubinComputeShader, "D3D12_DestroyCubinComputeShader"},
    {0x80403fc9u, (void*)&nvGetCudaTextureObject, "D3D12_GetCudaTextureObject"},
    {0x48f5b2eeu, (void*)&nvGetCudaSurfaceObject, "D3D12_GetCudaSurfaceObject"},
    {0x329fe6e0u, (void*)&nvGetCudaMergedTextureSamplerObject, "D3D12_GetCudaMergedTextureSamplerObject"},
    {0x0ddac234u, (void*)&nvGetCudaIndependentDescriptorObject, "D3D12_GetCudaIndependentDescriptorObject"},
    {0x70c07832u, (void*)&nvIsFatbinPTXSupported, "D3D12_IsFatbinPTXSupported"},
    {0xad1a677du, (void*)&nvCreateCuModule, "D3D12_CreateCuModule"},
    {0x7ab88d88u, (void*)&nvEnumFunctionsInModule, "D3D12_EnumFunctionsInModule"},
    {0xe2436e22u, (void*)&nvCreateCuFunction, "D3D12_CreateCuFunction"},
    {0x24973538u, (void*)&nvLaunchCuKernelChain, "D3D12_LaunchCuKernelChain"},
    {0x846a9bf0u, (void*)&nvLaunchCuKernelChainEx, "D3D12_LaunchCuKernelChainEx"},
    {0x41c65285u, (void*)&nvDestroyCuModule, "D3D12_DestroyCuModule"},
    {0xdf295ea6u, (void*)&nvDestroyCuFunction, "D3D12_DestroyCuFunction"},
    {0x6e5ea9dbu, (void*)&nvCaptureUAVInfo, "D3D12_CaptureUAVInfo"},
};

} // namespace

void configure(const Config& config)
{
    g_cfg = config;
    g_splitBarriers = config.splitBarriers;
    g_splitLog = config.splitLog;
    g_cache = utf8(config.cacheDir);
    for (char& c : g_cache) if (c == '\\') c = '/';
    while (!g_cache.empty() && g_cache.back() == '/') g_cache.pop_back();
    if (g_log == INVALID_HANDLE_VALUE && !config.logPath.empty())
        g_log = CreateFileW(config.logPath.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    logf("cubin executor configured: cache %s, texture-unit filtering %d, fp32 accumulation %d, groups %d",
         g_cache.c_str(), (int)config.texHw, (int)config.accFast, (int)config.groups);
}

void* queryInterface(unsigned id)
{
    for (auto& f : kFuncs)
        if (f.id == id) {
            static std::set<unsigned> seen;
            static std::mutex m;
            std::lock_guard<std::mutex> l(m);
            if (seen.insert(id).second) logf("NvAPI_%s served by the executor", f.name);
            return f.fn;
        }
    return nullptr;
}

void attach(void* device)
{
    if (!device) return;
    hookDevice(static_cast<ID3D12Device*>(device));
}

FgScope::FgScope(void* list) : outer(t_fg == 0)
{
    t_fg++;
    if (list && g_listHooks) {
        std::lock_guard<std::recursive_mutex> l(g_lock);
        chainFor(static_cast<List*>(list), true);
    }
}
FgScope::~FgScope()
{
    t_fg--;
    if (outer && g.ok && g.launchCount && (g.launchCount % 20000) < 200) {
        std::lock_guard<std::recursive_mutex> l(g_lock);
        static uint64_t saved = 0;
        if (g.launchCount - saved > 5000) { saved = g.launchCount; savePipelineCache(); }
    }
}

std::string summary()
{
    std::lock_guard<std::recursive_mutex> l(g_lock);
    char b[256];
    snprintf(b, sizeof b, "executor: %llu launches, %llu kernel batches, %llu part lists, %zu image slots, %zu samplers",
             (unsigned long long)g.launchCount, (unsigned long long)g.batchesSubmitted, (unsigned long long)g.partsCreated,
             g.viewSlots.size(), g.samplerSlots.size());
    return b;
}

} // namespace cux
