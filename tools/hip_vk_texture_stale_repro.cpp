// Standalone check (no Wine, no DLSS): does a HIP texture object created on memory imported from Vulkan
// see what Vulkan writes to that memory afterwards?
//
//   1. Vulkan creates a linear-tiled RGBA16F image (or, with --buffer, a plain buffer) on exportable memory
//      and uploads pattern A.
//   2. HIP imports the memory, creates a pitch-linear texture object on it and samples it in a kernel.
//   3. Vulkan uploads pattern B. HIP samples again, and also reads the memory with hipMemcpy.
//   4. HIP copies the texels out and back (a rewrite with their own bytes) and samples once more.
//
// The d4r NGX shim sees step 3's sample return pattern A for game textures while hipMemcpy returns B.
// usage: hip_vk_texture_stale_repro [--buffer] [--hip-first] [--prewrites N] [--game-usage] [WIDTH HEIGHT]
//   --buffer     export a VkBuffer (written with vkCmdCopyBuffer) instead of a linear VkImage
//   --hip-first  initialise HIP before the Vulkan object exists (default: after pattern A is uploaded)
//   --prewrites N  Vulkan uploads N times (ending on A) before HIP imports the memory (default 1)
//   --game-usage  create the image as vkd3d-proton would a render target: more usages, mutable format
#include <hip/hip_runtime.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

__global__ void sample_texture(hipTextureObject_t texture, float* out, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const float4 texel = tex2D<float4>(texture, x + 0.5f, y + 0.5f);
    float* pixel = out + (static_cast<size_t>(y) * width + x) * 4;
    pixel[0] = texel.x, pixel[1] = texel.y, pixel[2] = texel.z, pixel[3] = texel.w;
}

#define VK_CHECK(call)                                                                                          \
    do                                                                                                          \
    {                                                                                                           \
        const VkResult vkResult_ = (call);                                                                      \
        if (vkResult_ != VK_SUCCESS)                                                                            \
        {                                                                                                       \
            std::fprintf(stderr, "%s failed: %d\n", #call, vkResult_);                                          \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    } while (0)
#define HIP_CHECK(call)                                                                                         \
    do                                                                                                          \
    {                                                                                                           \
        const hipError_t hipResult_ = (call);                                                                   \
        if (hipResult_ != hipSuccess)                                                                           \
        {                                                                                                       \
            std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorName(hipResult_));                        \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    } while (0)

// Values k/256 only: exact as halves.
static uint16_t half_of(unsigned k)
{
    if (k == 0)
        return 0;
    int exponent = 0;
    while ((k >> exponent) > 1)
        ++exponent; // k = 1.m * 2^exponent
    const unsigned mantissa = (k << (10 - exponent)) & 0x3ffu;
    return static_cast<uint16_t>((exponent - 8 + 15) << 10 | mantissa);
}

// Pattern 0 ("A") and 1 ("B") differ in every texel.
static unsigned pattern_value(int pattern, int x, int y, int channel)
{
    const unsigned base = (x * 7 + y * 13 + channel * 29) % 200;
    return pattern == 0 ? base + 1 : 256 - base;
}

static void fill_pattern(uint16_t* texels, int width, int height, size_t rowPitch, int pattern)
{
    for (int y = 0; y < height; ++y)
    {
        uint16_t* row = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(texels) + y * rowPitch);
        for (int x = 0; x < width; ++x)
            for (int channel = 0; channel < 4; ++channel)
                row[x * 4 + channel] = half_of(pattern_value(pattern, x, y, channel));
    }
}

// Which pattern the floats hold: 'A', 'B', or '?' with the number of texels matching each.
static char classify(const float* values, int width, int height, size_t* matchA, size_t* matchB)
{
    *matchA = *matchB = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            bool a = true, b = true;
            for (int channel = 0; channel < 4; ++channel)
            {
                const float value = values[(static_cast<size_t>(y) * width + x) * 4 + channel];
                a &= value == pattern_value(0, x, y, channel) / 256.0f;
                b &= value == pattern_value(1, x, y, channel) / 256.0f;
            }
            *matchA += a, *matchB += b;
        }
    const size_t texels = static_cast<size_t>(width) * height;
    return *matchA == texels ? 'A' : *matchB == texels ? 'B' : '?';
}

static float half_to_float(uint16_t half)
{
    const int exponent = (half >> 10) & 31, mantissa = half & 0x3ff;
    if (exponent == 0)
        return mantissa / 1024.0f / 16384.0f;
    float value = 1.0f + mantissa / 1024.0f;
    for (int shift = exponent - 15; shift != 0; shift += shift < 0 ? 1 : -1)
        value = shift < 0 ? value / 2 : value * 2;
    return value;
}

struct Vulkan
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory = {};
};

static uint32_t memory_type(const Vulkan& vk, uint32_t typeBits, VkMemoryPropertyFlags wanted)
{
    for (uint32_t index = 0; index < vk.memory.memoryTypeCount; ++index)
        if ((typeBits & (1u << index)) && (vk.memory.memoryTypes[index].propertyFlags & wanted) == wanted)
            return index;
    std::fprintf(stderr, "no memory type with flags 0x%x\n", wanted);
    std::exit(1);
}

static void submit(const Vulkan& vk)
{
    VK_CHECK(vkEndCommandBuffer(vk.cmd));
    VkSubmitInfo info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = 1;
    info.pCommandBuffers = &vk.cmd;
    VK_CHECK(vkQueueSubmit(vk.queue, 1, &info, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(vk.queue));
}

static void begin(const Vulkan& vk)
{
    VkCommandBufferBeginInfo info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkResetCommandBuffer(vk.cmd, 0));
    VK_CHECK(vkBeginCommandBuffer(vk.cmd, &info));
}

int main(int argc, char** argv)
{
    bool useBuffer = false, hipFirst = false, gameUsage = false, zeroInit = false;
    int prewrites = 1, forcedType = -1;
    int width = 1280, height = 720, positional = 0;
    for (int index = 1; index < argc; ++index)
    {
        if (!std::strcmp(argv[index], "--buffer"))
            useBuffer = true;
        else if (!std::strcmp(argv[index], "--hip-first"))
            hipFirst = true;
        else if (!std::strcmp(argv[index], "--zero-init"))
            zeroInit = true; // allocate with VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT, as vkd3d-proton does
        else if (!std::strcmp(argv[index], "--game-usage"))
            gameUsage = true;
        else if (!std::strcmp(argv[index], "--memory-type") && index + 1 < argc)
            forcedType = std::atoi(argv[++index]); // Vulkan memory type index for the shared object
        else if (!std::strcmp(argv[index], "--prewrites") && index + 1 < argc)
            prewrites = std::atoi(argv[++index]);
        else
            (positional++ == 0 ? width : height) = std::atoi(argv[index]);
    }
    if (hipFirst)
        HIP_CHECK(hipInit(0));

    // --- Vulkan device ---
    Vulkan vk;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instanceInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &vk.instance));
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(vk.instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(vk.instance, &count, devices.data()));
    VkPhysicalDeviceProperties properties = {};
    for (VkPhysicalDevice candidate : devices)
    {
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.vendorID == 0x1002 && properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            vk.physical = candidate;
            break;
        }
    }
    if (vk.physical == VK_NULL_HANDLE)
    {
        std::fprintf(stderr, "no discrete AMD Vulkan device\n");
        return 1;
    }
    std::printf("Vulkan device: %s (driver version 0x%x)\n", properties.deviceName, properties.driverVersion);
    vkGetPhysicalDeviceMemoryProperties(vk.physical, &vk.memory);
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk.physical, &families, nullptr);
    std::vector<VkQueueFamilyProperties> familyProperties(families);
    vkGetPhysicalDeviceQueueFamilyProperties(vk.physical, &families, familyProperties.data());
    uint32_t family = 0;
    while (family < families && !(familyProperties[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        ++family;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = family;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    const char* extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, "VK_EXT_zero_initialize_device_memory"};
    VkPhysicalDeviceZeroInitializeDeviceMemoryFeaturesEXT zeroFeature = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_DEVICE_MEMORY_FEATURES_EXT};
    zeroFeature.zeroInitializeDeviceMemory = VK_TRUE;
    VkDeviceCreateInfo deviceInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, zeroInit ? &zeroFeature : nullptr};
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = zeroInit ? 2 : 1;
    deviceInfo.ppEnabledExtensionNames = extensions;
    VK_CHECK(vkCreateDevice(vk.physical, &deviceInfo, nullptr, &vk.device));
    vkGetDeviceQueue(vk.device, family, 0, &vk.queue);
    VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = family;
    VK_CHECK(vkCreateCommandPool(vk.device, &poolInfo, nullptr, &vk.pool));
    VkCommandBufferAllocateInfo cmdInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdInfo.commandPool = vk.pool;
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(vk.device, &cmdInfo, &vk.cmd));

    // --- the shared object: a linear image or a buffer on exportable device-local memory ---
    VkExternalMemoryImageCreateInfo externalImage = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    externalImage.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkExternalMemoryBufferCreateInfo externalBuffer = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    externalBuffer.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkImage image = VK_NULL_HANDLE;
    VkBuffer shared = VK_NULL_HANDLE;
    VkMemoryRequirements requirements = {};
    size_t rowPitch = static_cast<size_t>(width) * 8, texelOffset = 0;
    VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    if (useBuffer)
    {
        rowPitch = (rowPitch + 255) & ~static_cast<size_t>(255);
        VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &externalBuffer};
        info.size = rowPitch * height;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VK_CHECK(vkCreateBuffer(vk.device, &info, nullptr, &shared));
        vkGetBufferMemoryRequirements(vk.device, shared, &requirements);
        dedicated.buffer = shared;
    }
    else
    {
        VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &externalImage};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        info.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_LINEAR;
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (gameUsage)
        {
            info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
            info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        }
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(vk.device, &info, nullptr, &image));
        vkGetImageMemoryRequirements(vk.device, image, &requirements);
        dedicated.image = image;
    }
    VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dedicated};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateFlagsInfo allocateFlags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &exportInfo};
    allocateFlags.flags = VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT;
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                       zeroInit ? static_cast<const void*>(&allocateFlags) : &exportInfo};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(vk, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (forcedType >= 0)
        allocation.memoryTypeIndex = static_cast<uint32_t>(forcedType);
    VkDeviceMemory sharedMemory = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(vk.device, &allocation, nullptr, &sharedMemory));
    if (useBuffer)
        VK_CHECK(vkBindBufferMemory(vk.device, shared, sharedMemory, 0));
    else
    {
        VK_CHECK(vkBindImageMemory(vk.device, image, sharedMemory, 0));
        const VkImageSubresource subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
        VkSubresourceLayout layout = {};
        vkGetImageSubresourceLayout(vk.device, image, &subresource, &layout);
        rowPitch = layout.rowPitch;
        texelOffset = layout.offset;
    }
    std::printf("shared %s: %dx%d RGBA16F, row pitch %zu, offset %zu, allocation %llu bytes, memory type %u\n",
                useBuffer ? "VkBuffer" : "linear VkImage", width, height, rowPitch, texelOffset,
                static_cast<unsigned long long>(requirements.size), allocation.memoryTypeIndex);

    // --- host-visible staging buffer for uploads and Vulkan-side readback ---
    const size_t stagingBytes = rowPitch * height;
    VkBufferCreateInfo stagingInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagingInfo.size = stagingBytes;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer staging = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(vk.device, &stagingInfo, nullptr, &staging));
    VkMemoryRequirements stagingRequirements = {};
    vkGetBufferMemoryRequirements(vk.device, staging, &stagingRequirements);
    VkMemoryAllocateInfo stagingAllocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    stagingAllocation.allocationSize = stagingRequirements.size;
    stagingAllocation.memoryTypeIndex = memory_type(
        vk, stagingRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(vk.device, &stagingAllocation, nullptr, &stagingMemory));
    VK_CHECK(vkBindBufferMemory(vk.device, staging, stagingMemory, 0));
    void* stagingMap = nullptr;
    VK_CHECK(vkMapMemory(vk.device, stagingMemory, 0, VK_WHOLE_SIZE, 0, &stagingMap));

    VkBufferImageCopy region = {};
    region.bufferRowLength = static_cast<uint32_t>(rowPitch / 8);
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    const VkBufferCopy bufferRegion = {0, 0, stagingBytes};
    bool imageInitialised = false;
    const auto vulkan_upload = [&](int pattern) {
        std::memset(stagingMap, 0, stagingBytes);
        fill_pattern(static_cast<uint16_t*>(stagingMap), width, height, rowPitch, pattern);
        begin(vk);
        if (useBuffer)
            vkCmdCopyBuffer(vk.cmd, staging, shared, 1, &bufferRegion);
        else
        {
            if (!imageInitialised)
            {
                VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image;
                barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(vk.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                     nullptr, 0, nullptr, 1, &barrier);
                imageInitialised = true;
            }
            vkCmdCopyBufferToImage(vk.cmd, staging, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        }
        submit(vk);
    };
    std::vector<float> floats(static_cast<size_t>(width) * height * 4);
    size_t matchA = 0, matchB = 0;
    const auto report = [&](const char* what, char expected, const char* source) {
        const char seen = classify(floats.data(), width, height, &matchA, &matchB);
        std::printf("  %-34s %-22s pattern %c  (texels matching A: %zu, B: %zu)  %s\n", what, source, seen, matchA, matchB,
                    seen == expected ? "ok" : "MISMATCH");
        return seen == expected;
    };
    const auto halves_to_floats = [&](const uint16_t* texels) {
        for (int y = 0; y < height; ++y)
        {
            const uint16_t* row = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(texels) + y * rowPitch);
            for (int x = 0; x < width * 4; ++x)
                floats[static_cast<size_t>(y) * width * 4 + x] = half_to_float(row[x]);
        }
    };
    const auto vulkan_read = [&](const char* what, char expected) {
        std::memset(stagingMap, 0, stagingBytes);
        begin(vk);
        if (useBuffer)
            vkCmdCopyBuffer(vk.cmd, shared, staging, 1, &bufferRegion);
        else
            vkCmdCopyImageToBuffer(vk.cmd, image, VK_IMAGE_LAYOUT_GENERAL, staging, 1, &region);
        submit(vk);
        halves_to_floats(static_cast<const uint16_t*>(stagingMap));
        return report(what, expected, "Vulkan copy");
    };

    for (int write = prewrites - 1; write >= 0; --write)
        vulkan_upload(write & 1); // alternating, ending on A
    std::printf("Vulkan uploaded %d time(s) before the import, ending on pattern A\n", prewrites);

    // --- HIP: import, texture object, sampling kernel ---
    if (!hipFirst)
        HIP_CHECK(hipInit(0));
    int fd = -1;
    VkMemoryGetFdInfoKHR fdInfo = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    fdInfo.memory = sharedMemory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    const auto getMemoryFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(vk.device, "vkGetMemoryFdKHR"));
    VK_CHECK(getMemoryFd(vk.device, &fdInfo, &fd));
    hipExternalMemoryHandleDesc handle = {};
    handle.type = hipExternalMemoryHandleTypeOpaqueFd;
    handle.handle.fd = fd;
    handle.size = requirements.size;
    handle.flags = hipExternalMemoryDedicated;
    hipExternalMemory_t external = nullptr;
    HIP_CHECK(hipImportExternalMemory(&external, &handle));
    close(fd);
    hipExternalMemoryBufferDesc mapping = {};
    mapping.size = requirements.size;
    void* base = nullptr;
    HIP_CHECK(hipExternalMemoryGetMappedBuffer(&base, external, &mapping));
    uint8_t* texels = static_cast<uint8_t*>(base) + texelOffset;
    std::printf("HIP imported the memory at %p\n", base);

    hipResourceDesc resource = {};
    resource.resType = hipResourceTypePitch2D;
    resource.res.pitch2D.devPtr = texels;
    resource.res.pitch2D.desc = hipCreateChannelDesc(16, 16, 16, 16, hipChannelFormatKindFloat);
    resource.res.pitch2D.width = width;
    resource.res.pitch2D.height = height;
    resource.res.pitch2D.pitchInBytes = rowPitch;
    hipTextureDesc sampler = {};
    sampler.addressMode[0] = sampler.addressMode[1] = hipAddressModeClamp;
    sampler.filterMode = hipFilterModePoint;
    sampler.readMode = hipReadModeElementType;
    sampler.normalizedCoords = 0;
    hipTextureObject_t texture = nullptr;
    HIP_CHECK(hipCreateTextureObject(&texture, &resource, &sampler, nullptr));
    float* deviceOut = nullptr;
    HIP_CHECK(hipMalloc(&deviceOut, floats.size() * sizeof(float)));
    const auto hip_sample = [&](const char* what, char expected) {
        HIP_CHECK(hipMemset(deviceOut, 0, floats.size() * sizeof(float)));
        const dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
        hipLaunchKernelGGL(sample_texture, grid, block, 0, 0, texture, deviceOut, width, height);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(floats.data(), deviceOut, floats.size() * sizeof(float), hipMemcpyDeviceToHost));
        return report(what, expected, "HIP texture sample");
    };
    std::vector<uint16_t> hostTexels(stagingBytes / 2);
    const auto hip_memcpy = [&](const char* what, char expected) {
        HIP_CHECK(hipMemcpy(hostTexels.data(), texels, stagingBytes, hipMemcpyDeviceToHost));
        halves_to_floats(hostTexels.data());
        return report(what, expected, "hipMemcpy to host");
    };

    bool ok = true;
    std::printf("after import (Vulkan last wrote A):\n");
    ok &= hip_sample("texture created after A", 'A');
    ok &= hip_memcpy("", 'A');

    vulkan_upload(1);
    std::printf("Vulkan uploaded pattern B:\n");
    ok &= vulkan_read("", 'B');
    ok &= hip_memcpy("", 'B');
    ok &= hip_sample("same texture object", 'B');
    hipTextureObject_t second = nullptr;
    HIP_CHECK(hipCreateTextureObject(&second, &resource, &sampler, nullptr));
    std::swap(texture, second);
    ok &= hip_sample("new texture object", 'B');
    std::swap(texture, second);

    // HIP rewrites the texels with their own bytes.
    void* scratch = nullptr;
    HIP_CHECK(hipMalloc(&scratch, stagingBytes));
    HIP_CHECK(hipMemcpy(scratch, texels, stagingBytes, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(texels, scratch, stagingBytes, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("HIP copied the texels out and back:\n");
    ok &= hip_sample("same texture object", 'B');
    ok &= vulkan_read("", 'B');

    vulkan_upload(0);
    std::printf("Vulkan uploaded pattern A again:\n");
    ok &= hip_memcpy("", 'A');
    ok &= hip_sample("same texture object", 'A');

    std::printf("%s\n", ok ? "RESULT: consistent - HIP texture sampling follows Vulkan's writes"
                           : "RESULT: INCONSISTENT - see MISMATCH lines");
    return ok ? 0 : 3;
}
