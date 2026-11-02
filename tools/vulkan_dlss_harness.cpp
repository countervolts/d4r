// Synthetic native-Vulkan NGX integration test. No SDK or Vulkan import library.
// usage: vulkan_dlss_harness.exe SHIM_DLL OUTPUT_RAW [FRAMES [discard|invalid|dynamic|recreate|exit]]
// exit: the last frame is submitted and the process exits at once, without NGX shutdown.
// D4R_HARNESS_PAN=N: a textured scene scrolling N output pixels right per frame, with matching motion vectors
// (render-resolution, NGX convention: the offset to the pixel's previous position), in place of flat colours.
// D4R_HARNESS_RR=1: Ray Reconstruction (feature 13) with flat normals, roughness and albedo guides and a camera.
#define VK_NO_PROTOTYPES
#include <windows.h>
#include "d4r_vulkan_ngx.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
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
X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceMemoryProperties) X(GetPhysicalDeviceFeatures2) X(CreateDevice) \
X(GetDeviceProcAddr) X(GetDeviceQueue) X(CreateCommandPool) X(AllocateCommandBuffers) \
X(BeginCommandBuffer) X(EndCommandBuffer) X(ResetCommandPool) X(QueueSubmit) X(QueueWaitIdle) X(CreateFence) X(WaitForFences) X(ResetFences) X(DestroyFence) \
X(CreateImage) X(GetImageMemoryRequirements) X(AllocateMemory) X(BindImageMemory) X(CreateImageView) \
X(CreateBuffer) X(GetBufferMemoryRequirements) X(BindBufferMemory) X(MapMemory) X(UnmapMemory) \
X(CmdPipelineBarrier) X(CmdClearColorImage) X(CmdCopyImageToBuffer) X(CmdCopyBufferToImage) X(DestroyImageView) X(DestroyImage) \
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
    // D4R_HARNESS_ENGINE_DEVICE=1: a Vulkan 1.3 device with every supported 1.1-1.3 feature plus cooperative matrix
    // and shader float8, as d4r's engine (and a layer in a game) needs.
    const bool engineDevice = std::getenv("D4R_HARNESS_ENGINE_DEVICE") != nullptr;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = engineDevice ? VK_API_VERSION_1_3 : VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo = &app;
    VkInstance instance;
    check(vkCreateInstance(&ci,nullptr,&instance),"instance");
    // Vulkan 1.2+ loaders return no global commands (vkCreateInstance) for an instance: keep the one loaded above.
#define LOAD(name) if (auto f = reinterpret_cast<PFN_vk##name>(ip(instance, "vk" #name))) vk##name = f; if (!vk##name) return 1;
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
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, &f8};
    VkPhysicalDeviceVulkan13Features v13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cm};
    VkPhysicalDeviceVulkan12Features v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    std::vector<const char*> engineExtensions;
    if (engineDevice) {
        vkGetPhysicalDeviceFeatures2(physical,&f2);
        check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&count,nullptr),"extensions");
        std::vector<VkExtensionProperties> extensions(count);
        check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&count,extensions.data()),"extensions");
        bool float8 = false;
        for (const auto& e : extensions) float8 |= std::strcmp(e.extensionName,"VK_EXT_shader_float8")==0;
        if (!float8) { f8.shaderFloat8 = f8.shaderFloat8CooperativeMatrix = VK_FALSE; cm.pNext = nullptr; }
        if (cm.cooperativeMatrix) engineExtensions.push_back("VK_KHR_cooperative_matrix"); else v13.pNext = nullptr;
        if (float8) engineExtensions.push_back("VK_EXT_shader_float8");
        for (uint32_t i = 0; i < di.enabledExtensionCount; ++i) engineExtensions.push_back(di.ppEnabledExtensionNames[i]);
        // and what NGX asks a game for (vkhip's VK_KHR_pipeline_binary and VK_KHR_maintenance5, with their features)
        using Requirements = int (*)(VkInstance, VkPhysicalDevice, const void*, uint32_t*, VkExtensionProperties**);
        if (auto requirements = reinterpret_cast<Requirements>(reinterpret_cast<void*>(
                GetProcAddress(shim, "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements")))) {
            uint32_t wanted = 0; VkExtensionProperties* list = nullptr;
            if (requirements(instance, physical, nullptr, &wanted, &list) == 1 && list)
                for (uint32_t i = 0; i < wanted; ++i)
                    if (std::find_if(engineExtensions.begin(), engineExtensions.end(), [&](const char* e) {
                            return std::strcmp(e, list[i].extensionName) == 0; }) == engineExtensions.end())
                        engineExtensions.push_back(list[i].extensionName);
        }
        // D4R_HARNESS_NO_BINARY_FEATURES=1: the extensions without their feature structs, as a game that only appends
        // NGX's extension names to its device (Streamline's interposer) creates it
        for (const char* e : engineExtensions)
            if (!std::strcmp(e, "VK_KHR_pipeline_binary") && !std::getenv("D4R_HARNESS_NO_BINARY_FEATURES")) {
                static VkPhysicalDeviceMaintenance5FeaturesKHR m5 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR};
                static VkPhysicalDevicePipelineBinaryFeaturesKHR pb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR, &m5};
                m5.maintenance5 = pb.pipelineBinaries = VK_TRUE;
                m5.pNext = f2.pNext; f2.pNext = &pb;
                std::puts("engine device: pipeline binaries");
            }
        di.pNext = &f2; di.enabledExtensionCount = static_cast<uint32_t>(engineExtensions.size());
        di.ppEnabledExtensionNames = engineExtensions.data();
        std::printf("engine device: cooperative matrix %u, float8 %u\n", cm.cooperativeMatrix, f8.shaderFloat8);
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
    wchar_t dataPath[MAX_PATH] = L"."; GetTempPathW(MAX_PATH, dataPath); // NGX's application data folder
    ngx(Init(100899111ULL,dataPath,instance,physical,device,ip,dp,nullptr,0x14),"init");
    void* params = nullptr; ngx(GetCapabilityParameters(&params),"capability parameters");
    d4r_ngx_set_uint(params,"Width",W); d4r_ngx_set_uint(params,"Height",H);
    d4r_ngx_set_uint(params,"OutWidth",OW); d4r_ngx_set_uint(params,"OutHeight",OH);
    // D4R_HARNESS_CREATE_FLAGS: the NVSDK_NGX_DLSS_Feature_Flags, as in the D3D12 test app (default 0x6b)
    const char* createFlags = std::getenv("D4R_HARNESS_CREATE_FLAGS");
    d4r_ngx_set_int(params,"PerfQualityValue",2);
    d4r_ngx_set_int(params,"DLSS.Feature.Create.Flags",createFlags ? static_cast<int>(std::strtol(createFlags,nullptr,0)) : 0x6b);
    const bool rr = std::getenv("D4R_HARNESS_RR") != nullptr;
    Handle* handle = nullptr; ngx(CreateFeature(VK_NULL_HANDLE,rr ? 13 : 1,params,&handle),"create");
    VkDeviceMemory allocations[4];
    // D4R_HARNESS_R11_OUTPUT=1: an R11G11B10F output, as Indiana Jones's; read back widened to RGBA16F
    const bool r11Output = std::getenv("D4R_HARNESS_R11_OUTPUT") != nullptr;
    NgxVkResource images[] = {make_image(W,H,VK_FORMAT_R16G16B16A16_SFLOAT,allocations[0]),
        make_image(W,H,VK_FORMAT_R32_SFLOAT,allocations[1]), make_image(W,H,VK_FORMAT_R16G16_SFLOAT,allocations[2]),
        make_image(OW,OH,r11Output ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : VK_FORMAT_R16G16B16A16_SFLOAT,allocations[3])};
    const char* names[] = {"Color","Depth","MotionVectors","Output"};
    // D4R_HARNESS_DEPTH_ASPECT=1: label the R32F depth image with the depth aspect, as Indiana Jones
    // (through Streamline/OptiScaler) does; the harness keeps the real colour range for its own barriers.
    NgxVkResource described[4] = {images[0], images[1], images[2], images[3]};
    if (std::getenv("D4R_HARNESS_DEPTH_ASPECT")) described[1].resource.image.range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    for (int i=0;i<4;++i) d4r_ngx_set_void(params,names[i],&described[i]);
    // the denoiser's guides (normals, roughness, diffuse and specular albedo) and its camera, row-major
    VkDeviceMemory guideMemory[4] = {};
    NgxVkResource guides[4] = {};
    const char* guideNames[] = {"GBuffer.Normals","GBuffer.Roughness","GBuffer.DiffuseAlbedo","GBuffer.SpecularAlbedo"};
    const float guideValues[4][4] = {{0,0,1,0.5f},{0.5f,0,0,0},{0.5f,0.5f,0.5f,1},{0.04f,0.04f,0.04f,1}};
    static float worldToView[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    static float viewToClip[16] = {1.0f/1.7778f,0,0,0, 0,1,0,0, 0,0,0,1, 0,0,0.1f,0};
    if (rr) {
        for (int i=0;i<4;++i) {
            guides[i] = make_image(W,H,i==1 ? VK_FORMAT_R16_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT,guideMemory[i]);
            d4r_ngx_set_void(params,guideNames[i],&guides[i]);
        }
        d4r_ngx_set_void(params,"WorldToViewMatrix",worldToView); d4r_ngx_set_void(params,"ViewToClipMatrix",viewToClip);
    }
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
    const int pan = std::getenv("D4R_HARNESS_PAN") ? std::atoi(std::getenv("D4R_HARNESS_PAN")) : 0;
    // the scene's colour (RGBA16F) and motion (RG16F) uploads, one pair per frame in flight
    const VkDeviceSize sceneBytes = VkDeviceSize(W)*H*8 + VkDeviceSize(W)*H*4;
    VkBuffer scene = VK_NULL_HANDLE; VkDeviceMemory sceneMemory = VK_NULL_HANDLE; uint8_t* sceneData = nullptr;
    if (pan) {
        VkBufferCreateInfo si = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; si.size=sceneBytes*inFlight; si.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        check(vkCreateBuffer(device,&si,nullptr,&scene),"scene buffer");
        VkMemoryRequirements sr; vkGetBufferMemoryRequirements(device,scene,&sr);
        sceneMemory=allocate(sr,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindBufferMemory(device,scene,sceneMemory,0),"bind scene");
        void* mapped; check(vkMapMemory(device,sceneMemory,0,VK_WHOLE_SIZE,0,&mapped),"map scene"); sceneData=static_cast<uint8_t*>(mapped);
    }
    // a pattern with edges and gradients at several scales, defined in output pixels so the pan stays exact
    auto half = [](float v) { _Float16 h = static_cast<_Float16>(v); uint16_t bits; std::memcpy(&bits,&h,2); return bits; };
    auto pattern = [](float x, float y, int channel) {
        const float checker = (int(std::floor(x / 24)) + int(std::floor(y / 24))) & 1 ? 0.8f : 0.2f;
        const float waves = 0.5f + 0.5f * std::sin(x * 0.045f + channel * 1.7f) * std::cos(y * 0.031f - channel);
        const float ring = std::fmod(std::sqrt((x - 640) * (x - 640) + (y - 360) * (y - 360)) / 9.0f, 1.0f);
        return 0.45f * checker + 0.35f * waves + 0.2f * ring;
    };
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
            if (pan && i != 1) {
                uint8_t* base = sceneData + sceneBytes*slot;
                const float shift = float(n) * pan;
                if (i == 0) {
                    auto* color = reinterpret_cast<uint16_t*>(base);
                    for (uint32_t y=0;y<H;++y) for (uint32_t x=0;x<W;++x) for (int c=0;c<4;++c)
                        color[(y*W+x)*4+c] = half(c==3 ? 1.0f : pattern((x+0.5f)*OW/W - shift, (y+0.5f)*OH/H, c));
                } else {
                    auto* motion = reinterpret_cast<uint16_t*>(base + VkDeviceSize(W)*H*8);
                    for (uint32_t j=0;j<W*H;++j) { motion[j*2] = half(n ? -float(pan)*W/OW : 0.0f); motion[j*2+1] = 0; }
                }
                VkBufferImageCopy up={}; up.bufferOffset = sceneBytes*slot + (i == 2 ? VkDeviceSize(W)*H*8 : 0);
                up.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; up.imageExtent={W,H,1};
                vkCmdCopyBufferToImage(cmd,scene,b.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&up);
            } else {
            VkClearColorValue clear={};
            if (i==0) { clear.float32[0]=0.25f; clear.float32[1]=0.5f; clear.float32[2]=0.75f; clear.float32[3]=1; }
            if (i==1) clear.float32[0]=0.5f;
            vkCmdClearColorImage(cmd,b.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&clear,1,&b.subresourceRange);
            }
            b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,1,&b);
        }
        for (int i=0;i<4 && rr;++i) {
            VkImageMemoryBarrier b={VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; b.image=guides[i].resource.image.image; b.subresourceRange=guides[i].resource.image.range;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkClearColorValue clear={}; std::memcpy(clear.float32,guideValues[i],sizeof(clear.float32));
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
        std::vector<uint16_t> widened;
        if (r11Output) { // the packed floats' exponents and mantissas are binary16's, shortened; alpha is 1
            widened.resize(size_t(OW)*OH*4);
            for (size_t j=0;j<size_t(OW)*OH;++j) {
                const uint32_t v=static_cast<const uint32_t*>(data)[j];
                widened[j*4]=uint16_t((v&0x7ffu)<<4); widened[j*4+1]=uint16_t(((v>>11)&0x7ffu)<<4);
                widened[j*4+2]=uint16_t(((v>>22)&0x3ffu)<<5); widened[j*4+3]=0x3c00;
            }
            values=widened.data();
        }
        unsigned nonzero=0, nonfinite=0;
        for (unsigned j=0;j<OW*OH*4;++j) { nonzero += (values[j]&0x7fff)!=0; nonfinite += (values[j]&0x7c00)==0x7c00; }
        std::printf("frame %u: nonzero=%u nonfinite=%u center=%04x,%04x,%04x,%04x\n",n,nonzero,nonfinite,values[((OH/2)*OW+OW/2)*4],values[((OH/2)*OW+OW/2)*4+1],values[((OH/2)*OW+OW/2)*4+2],values[((OH/2)*OW+OW/2)*4+3]);
        // D4R_HARNESS_ALLOW_BLACK=1: a black frame is reported but not fatal (NGX's K starts black on the reset frame here).
        if (nonfinite || (!nonzero && !std::getenv("D4R_HARNESS_ALLOW_BLACK"))) return 1;
        if (n+1==frames) { FILE* out=std::fopen(argv[2],"wb"); if (!out) return 1; const auto written=std::fwrite(values,1,OW*OH*8,out); std::fclose(out); if (written!=OW*OH*8) return 1; }
        vkUnmapMemory(device,readMemory);
    }
    ngx(ReleaseFeature(handle),"release"); ngx(DestroyParameters(params),"destroy params"); ngx(Shutdown(),"shutdown");
    check(vkQueueWaitIdle(queue),"final wait");
    for (unsigned i=0;i<inFlight;++i) { vkDestroyCommandPool(device,pools[i],nullptr); vkDestroyFence(device,fences[i],nullptr); } vkDestroyBuffer(device,readback,nullptr); vkFreeMemory(device,readMemory,nullptr);
    if (scene) { vkDestroyBuffer(device,scene,nullptr); vkFreeMemory(device,sceneMemory,nullptr); }
    for (int i=0;i<4;++i) { vkDestroyImageView(device,images[i].resource.image.view,nullptr); vkDestroyImage(device,images[i].resource.image.image,nullptr); vkFreeMemory(device,allocations[i],nullptr); }
    for (int i=0;i<4 && rr;++i) { vkDestroyImageView(device,guides[i].resource.image.view,nullptr); vkDestroyImage(device,guides[i].resource.image.image,nullptr); vkFreeMemory(device,guideMemory[i],nullptr); }
    vkDestroyDevice(device,nullptr); vkDestroyInstance(instance,nullptr);
    return 0;
}
