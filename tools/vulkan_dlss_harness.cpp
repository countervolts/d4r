// Synthetic native-Vulkan NGX integration test. No SDK or Vulkan import library.
// usage: vulkan_dlss_harness.exe SHIM_DLL OUTPUT_RAW [FRAMES [discard|invalid|dynamic|recreate|exit]]
// exit: the last frame is submitted and the process exits at once, without NGX shutdown.
#define VK_NO_PROTOTYPES
#include <windows.h>
#include "d4r_vulkan_ngx.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
void d4r_ngx_set_uint(void*, const char*, unsigned int);
void d4r_ngx_set_int(void*, const char*, int);
void d4r_ngx_set_float(void*, const char*, float);
void d4r_ngx_set_void(void*, const char*, void*);
}
using Result = unsigned int;
struct Handle { unsigned int id; };
using Init = Result (*)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice,
                       PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, const void*, unsigned int);
using Params = Result (*)(void**);
using Destroy = Result (*)(void*);
using Create = Result (*)(VkCommandBuffer, unsigned int, void*, Handle**);
using Eval = Result (*)(VkCommandBuffer, const Handle*, void*, void*);
using Release = Result (*)(Handle*);
using Shutdown = Result (*)();

#define VK_FUNCTIONS(X) \
X(CreateInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) X(EnumerateDeviceExtensionProperties) \
X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceMemoryProperties) X(CreateDevice) \
X(GetDeviceProcAddr) X(GetDeviceQueue) X(CreateCommandPool) X(AllocateCommandBuffers) \
X(BeginCommandBuffer) X(EndCommandBuffer) X(ResetCommandPool) X(QueueSubmit) X(QueueWaitIdle) X(CreateFence) X(WaitForFences) X(ResetFences) X(DestroyFence) \
X(CreateImage) X(GetImageMemoryRequirements) X(AllocateMemory) X(BindImageMemory) X(CreateImageView) \
X(CreateBuffer) X(GetBufferMemoryRequirements) X(BindBufferMemory) X(MapMemory) X(UnmapMemory) \
X(CmdPipelineBarrier) X(CmdClearColorImage) X(CmdCopyImageToBuffer) X(DestroyImageView) X(DestroyImage) \
X(DestroyBuffer) X(FreeMemory) X(DestroyCommandPool) X(DestroyDevice) X(DestroyInstance)
#define DECLARE(name) static PFN_vk##name vk##name;
VK_FUNCTIONS(DECLARE)
#undef DECLARE
// -DD4R_HARNESS_SCALE=2 runs DOOM's Performance size (1280x720 -> 2560x1440) for timing.
#ifndef D4R_HARNESS_SCALE
#define D4R_HARNESS_SCALE 1
#endif
constexpr uint32_t W = 640 * D4R_HARNESS_SCALE, H = 360 * D4R_HARNESS_SCALE, OW = 2 * W, OH = 2 * H;
static VkDevice device;
static VkPhysicalDeviceMemoryProperties memory;
static void check(VkResult r, const char* name) { if (r != VK_SUCCESS) { std::fprintf(stderr, "%s: %d\n", name, r); std::exit(1); } }
static void ngx(Result r, const char* name) { if (r != 1) { std::fprintf(stderr, "%s: 0x%08x\n", name, r); std::exit(1); } }
static VkDeviceMemory allocate(VkMemoryRequirements requirements, VkMemoryPropertyFlags flags)
{
    uint32_t type = UINT32_MAX;
    for (int cached = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? 1 : 0; cached >= 0 && type == UINT32_MAX; --cached)
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags &&
                (!cached || (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))) { type = i; break; }
    if (type == UINT32_MAX) { std::fprintf(stderr, "no memory type\n"); std::exit(1); }
    VkMemoryAllocateInfo info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    info.allocationSize = requirements.size; info.memoryTypeIndex = type;
    VkDeviceMemory result;
    check(vkAllocateMemory(device, &info, nullptr, &result), "allocate"); return result;
}
static NgxVkResource make_image(uint32_t w, uint32_t h, VkFormat format, VkDeviceMemory& allocation)
{
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D; info.format = format; info.extent = {w,h,1};
    info.mipLevels = info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = w == OW && h == OH ? VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT :
                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    NgxVkResource resource = {};
    auto& image = resource.resource.image;
    check(vkCreateImage(device, &info, nullptr, &image.image), "create image");
    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(device, image.image, &requirements);
    allocation = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkBindImageMemory(device, image.image, allocation, 0), "bind image");
    image.range = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}; image.format = format; image.width = w; image.height = h;
    VkImageViewCreateInfo view = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = image.image; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = format; view.subresourceRange = image.range;
    check(vkCreateImageView(device, &view, nullptr, &image.view), "image view");
    resource.readWrite = true; return resource;
}
int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    if (const char* log = std::getenv("D4R_HARNESS_LOG")) {
        if (!std::freopen(log, "w", stdout)) return 2;
        if (!std::freopen(log, "a", stderr)) return 2;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::puts("loading Vulkan and NGX shim");
    const unsigned frames = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 3;
    const bool discard = argc > 4 && !std::strcmp(argv[4], "discard");
    const bool invalid = argc > 4 && !std::strcmp(argv[4], "invalid");
    const bool dynamic = argc > 4 && !std::strcmp(argv[4], "dynamic");
    const bool recreate = argc > 4 && !std::strcmp(argv[4], "recreate");
    const bool exitEarly = argc > 4 && !std::strcmp(argv[4], "exit");
    HMODULE vulkan = LoadLibraryA("vulkan-1.dll"), shim = LoadLibraryA(argv[1]);
    if (!vulkan || !shim) { std::fprintf(stderr,"DLL load: %lu\n",GetLastError()); return 1; }
    auto ip = reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(vulkan,"vkGetInstanceProcAddr")));
#define LOAD(name) vk##name = reinterpret_cast<PFN_vk##name>(ip(VK_NULL_HANDLE, "vk" #name));
    LOAD(CreateInstance)
#undef LOAD
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo = &app;
    VkInstance instance;
    check(vkCreateInstance(&ci,nullptr,&instance),"instance");
#define LOAD(name) vk##name = reinterpret_cast<PFN_vk##name>(ip(instance, "vk" #name)); if (!vk##name) return 1;
    VK_FUNCTIONS(LOAD)
#undef LOAD
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance,&count,nullptr),"devices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance,&count,devices.data()),"devices");
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    for (auto d : devices) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d,&p); std::printf("Adapter: %s vendor=%04x\n",p.deviceName,p.vendorID); if (p.vendorID == 0x1002 || p.vendorID == 0x10de) { physical=d; std::printf("GPU: %s\n",p.deviceName); break; } }
    if (!physical) return 1;
    vkGetPhysicalDeviceMemoryProperties(physical,&memory);
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());
    uint32_t family = UINT32_MAX;
    // D4R_HARNESS_COMPUTE_QUEUE=1 records on a compute-only family, as games with async compute DLSS do.
    const bool computeQueue = std::getenv("D4R_HARNESS_COMPUTE_QUEUE") != nullptr;
    for (uint32_t i=0;i<count;++i)
        if (computeQueue ? (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                         : (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) { family=i; break; }
    std::printf("queue family %u (%s)\n", family, computeQueue ? "compute" : "graphics");
    if (family == UINT32_MAX) return 1;
    float priority = 1;
    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; queueInfo.queueFamilyIndex=family; queueInfo.queueCount=1; queueInfo.pQueuePriorities=&priority;
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount=1; di.pQueueCreateInfos=&queueInfo;
    const char* externalExtension = "VK_KHR_external_memory_win32";
    if (std::getenv("D4R_HARNESS_EXTERNAL_MEMORY")) {
        check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&count,nullptr),"extensions");
        std::vector<VkExtensionProperties> extensions(count);
        check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&count,extensions.data()),"extensions");
        bool supported = false;
        for (const auto& e : extensions) supported |= std::strcmp(e.extensionName,externalExtension)==0;
        if (!supported) { std::fputs("external memory extension unavailable\n",stderr); return 1; }
        di.enabledExtensionCount=1; di.ppEnabledExtensionNames=&externalExtension;
        std::puts("enabling external-memory VRAM interop");
    }
    check(vkCreateDevice(physical,&di,nullptr,&device),"device");
    auto dp = vkGetDeviceProcAddr;
#define LOAD(name) vk##name = reinterpret_cast<PFN_vk##name>(dp(device, "vk" #name));
    // Instance entry points above remain valid; only load device entry points here.
    LOAD(GetDeviceQueue) LOAD(QueueSubmit) LOAD(QueueWaitIdle)
#undef LOAD
#define NGX(name, type) auto name = reinterpret_cast<type>(reinterpret_cast<void*>(GetProcAddress(shim,"NVSDK_NGX_VULKAN_" #name))); if (!name) return 1;
    NGX(Init, ::Init) NGX(GetCapabilityParameters, Params) NGX(DestroyParameters, Destroy)
    NGX(CreateFeature, Create) NGX(EvaluateFeature, Eval) NGX(ReleaseFeature, Release) NGX(Shutdown, ::Shutdown)
#undef NGX
    std::puts("initializing CUDA NGX through Vulkan frontend");
    ngx(Init(100899111ULL,L"Z:\\tmp",instance,physical,device,ip,dp,nullptr,0x14),"init");
    void* params = nullptr; ngx(GetCapabilityParameters(&params),"capability parameters");
    d4r_ngx_set_uint(params,"Width",W); d4r_ngx_set_uint(params,"Height",H);
    d4r_ngx_set_uint(params,"OutWidth",OW); d4r_ngx_set_uint(params,"OutHeight",OH);
    d4r_ngx_set_int(params,"PerfQualityValue",2); d4r_ngx_set_int(params,"DLSS.Feature.Create.Flags",0x6b);
    Handle* handle = nullptr; ngx(CreateFeature(VK_NULL_HANDLE,1,params,&handle),"create");
    VkDeviceMemory allocations[4];
    NgxVkResource images[] = {make_image(W,H,VK_FORMAT_R16G16B16A16_SFLOAT,allocations[0]),
        make_image(W,H,VK_FORMAT_R32_SFLOAT,allocations[1]), make_image(W,H,VK_FORMAT_R16G16_SFLOAT,allocations[2]),
        make_image(OW,OH,VK_FORMAT_R16G16B16A16_SFLOAT,allocations[3])};
    const char* names[] = {"Color","Depth","MotionVectors","Output"};
    // D4R_HARNESS_DEPTH_ASPECT=1: label the R32F depth image with the depth aspect, as Indiana Jones
    // (through Streamline/OptiScaler) does; the harness keeps the real colour range for its own barriers.
    NgxVkResource described[4] = {images[0], images[1], images[2], images[3]};
    if (std::getenv("D4R_HARNESS_DEPTH_ASPECT")) described[1].resource.image.range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    for (int i=0;i<4;++i) d4r_ngx_set_void(params,names[i],&described[i]);
    // D4R_HARNESS_IN_FLIGHT=N: up to N submitted frames at once (own pool, command buffer and fence each),
    // as a game does; only the final frame is read back.
    const unsigned inFlight = std::max(1u, std::min(8u, static_cast<unsigned>(std::getenv("D4R_HARNESS_IN_FLIGHT") ? std::atoi(std::getenv("D4R_HARNESS_IN_FLIGHT")) : 1)));
    VkCommandPool pools[8]; VkCommandBuffer commands[8]; VkFence fences[8];
    for (unsigned i=0;i<inFlight;++i) {
        VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; poolInfo.queueFamilyIndex=family;
        check(vkCreateCommandPool(device,&poolInfo,nullptr,&pools[i]),"pool");
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=pools[i]; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
        check(vkAllocateCommandBuffers(device,&ai,&commands[i]),"command buffer");
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        check(vkCreateFence(device,&fi,nullptr,&fences[i]),"fence");
    }
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size=OW*OH*8; bi.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer readback; check(vkCreateBuffer(device,&bi,nullptr,&readback),"readback");
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,readback,&req);
    VkDeviceMemory readMemory=allocate(req,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkBindBufferMemory(device,readback,readMemory,0),"bind readback");
    VkQueue queue; vkGetDeviceQueue(device,family,0,&queue);
    for (unsigned n=0;n<frames;++n)
    {
        if (dynamic) {
            const unsigned phase=(n/4)%3;
            d4r_ngx_set_uint(params,"DLSS.Render.Subrect.Dimensions.Width",W*(8-phase)/8);
            d4r_ngx_set_uint(params,"DLSS.Render.Subrect.Dimensions.Height",H*(8-phase)/8);
        }
        const unsigned slot = n % inFlight; VkCommandBuffer cmd = commands[slot];
        check(vkWaitForFences(device,1,&fences[slot],VK_TRUE,UINT64_MAX),"fence wait"); check(vkResetFences(device,1,&fences[slot]),"fence reset");
        check(vkResetCommandPool(device,pools[slot],0),"reset pool");
        if (recreate && n && n % 4 == 0) {
            ngx(ReleaseFeature(handle),"release before recreation");
            const unsigned phase=(n/4)%3;
            d4r_ngx_set_uint(params,"Width",W*(8-phase)/8);
            d4r_ngx_set_uint(params,"Height",H*(8-phase)/8);
            d4r_ngx_set_int(params,"PerfQualityValue",phase);
            ngx(CreateFeature(VK_NULL_HANDLE,1,params,&handle),"recreate");
            std::printf("recreated feature: quality=%u render=%ux%u\n",phase,W*(8-phase)/8,H*(8-phase)/8);
        }
        VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd,&begin),"begin");
        for (int i=0;i<4;++i)
        {
            VkImageMemoryBarrier b={VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout=i==3 ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=images[i].resource.image.image; b.subresourceRange=images[i].resource.image.range;
            b.dstAccessMask=i==3 ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,i==3 ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
            if(i==3) continue;
            VkClearColorValue clear={};
            if (i==0) { clear.float32[0]=0.25f; clear.float32[1]=0.5f; clear.float32[2]=0.75f; clear.float32[3]=1; }
            if (i==1) clear.float32[0]=0.5f;
            vkCmdClearColorImage(cmd,b.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&clear,1,&b.subresourceRange);
            b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,1,&b);
        }
        if (invalid) described[2].type=1;
        const Result evaluated=EvaluateFeature(cmd,handle,params,nullptr);
        if (invalid) { if (evaluated != 0xBAD0000e) return 1; std::puts("invalid resource rejected"); break; }
        ngx(evaluated,"evaluate");
        if (discard) { std::puts("discarded command buffer; releasing pending feature"); break; }
        auto& image=images[3].resource.image;
        VkBufferImageCopy copy={}; copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent={OW,OH,1};
        VkMemoryBarrier output={VK_STRUCTURE_TYPE_MEMORY_BARRIER}; output.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT; output.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&output,0,nullptr,0,nullptr);
        vkCmdCopyImageToBuffer(cmd,image.image,VK_IMAGE_LAYOUT_GENERAL,readback,1,&copy);
        VkMemoryBarrier host={VK_STRUCTURE_TYPE_MEMORY_BARRIER}; host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host,0,nullptr,0,nullptr);
        check(vkEndCommandBuffer(cmd),"end");
        VkSubmitInfo submit={VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&cmd;
        check(vkQueueSubmit(queue,1,&submit,fences[slot]),"submit");
        if (exitEarly && n+1==frames) { std::puts("exiting with the last frame in flight"); std::fflush(stdout); ExitProcess(0); }
        if (inFlight > 1 && n+1 < frames) continue; // pipelined: only the last frame is checked
        check(vkQueueWaitIdle(queue),"wait");
        void* data; check(vkMapMemory(device,readMemory,0,VK_WHOLE_SIZE,0,&data),"map");
        const auto* values=static_cast<const uint16_t*>(data);
        unsigned nonzero=0, nonfinite=0;
        for (unsigned j=0;j<OW*OH*4;++j) { nonzero += (values[j]&0x7fff)!=0; nonfinite += (values[j]&0x7c00)==0x7c00; }
        std::printf("frame %u: nonzero=%u nonfinite=%u center=%04x,%04x,%04x,%04x\n",n,nonzero,nonfinite,values[((OH/2)*OW+OW/2)*4],values[((OH/2)*OW+OW/2)*4+1],values[((OH/2)*OW+OW/2)*4+2],values[((OH/2)*OW+OW/2)*4+3]);
        if (!nonzero || nonfinite) return 1;
        if (n+1==frames) { FILE* out=std::fopen(argv[2],"wb"); if (!out) return 1; const auto written=std::fwrite(data,1,OW*OH*8,out); std::fclose(out); if (written!=OW*OH*8) return 1; }
        vkUnmapMemory(device,readMemory);
    }
    ngx(ReleaseFeature(handle),"release"); ngx(DestroyParameters(params),"destroy params"); ngx(Shutdown(),"shutdown");
    check(vkQueueWaitIdle(queue),"final wait");
    for (unsigned i=0;i<inFlight;++i) { vkDestroyCommandPool(device,pools[i],nullptr); vkDestroyFence(device,fences[i],nullptr); } vkDestroyBuffer(device,readback,nullptr); vkFreeMemory(device,readMemory,nullptr);
    for (int i=0;i<4;++i) { vkDestroyImageView(device,images[i].resource.image.view,nullptr); vkDestroyImage(device,images[i].resource.image.image,nullptr); vkFreeMemory(device,allocations[i],nullptr); }
    vkDestroyDevice(device,nullptr); vkDestroyInstance(instance,nullptr);
    return 0;
}
