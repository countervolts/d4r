#include <hip/hip_runtime_api.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <vector>

template <typename T> static T vk_proc(VkDevice device, const char* name)
{
    return reinterpret_cast<T>(vkGetDeviceProcAddr(device, name));
}

static void print_hip(const char* stage, hipError_t result)
{
    std::printf("%s: %s (%d)\n", stage, hipGetErrorName(result), static_cast<int>(result));
}

static uint32_t choose_memory_type(VkPhysicalDevice physicalDevice, uint32_t typeBits)
{
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
    {
        if ((typeBits & (1u << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
            return i;
    }
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
    {
        if ((typeBits & (1u << i)) != 0)
            return i;
    }
    return UINT32_MAX;
}

int main()
{
    constexpr VkDeviceSize testSize = 4096;
    constexpr unsigned char testByte = 0x5a;

    hipError_t hipResult = hipInit(0);
    print_hip("hipInit", hipResult);
    if (hipResult != hipSuccess)
        return 1;

    int hipDeviceCount = 0;
    hipResult = hipGetDeviceCount(&hipDeviceCount);
    print_hip("hipGetDeviceCount", hipResult);
    if (hipResult != hipSuccess || hipDeviceCount < 1)
    {
        std::fprintf(stderr, "No HIP device is available\n");
        return 1;
    }

    int hipDevice = 0;
    hipResult = hipSetDevice(hipDevice);
    print_hip("hipSetDevice(0)", hipResult);
    if (hipResult != hipSuccess)
        return 1;

    hipDeviceProp_t hipProperties{};
    hipResult = hipGetDeviceProperties(&hipProperties, hipDevice);
    print_hip("hipGetDeviceProperties", hipResult);
    if (hipResult != hipSuccess)
        return 1;
    std::printf("HIP device: %s (%s)\n", hipProperties.name, hipProperties.gcnArchName);

    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "d4r HIP/Vulkan external-memory probe";
    application.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &application;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult vkResult = vkCreateInstance(&instanceInfo, nullptr, &instance);
    std::printf("vkCreateInstance: %d\n", static_cast<int>(vkResult));
    if (vkResult != VK_SUCCESS)
        return 1;

    uint32_t physicalDeviceCount = 0;
    vkResult = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    if (vkResult == VK_SUCCESS && physicalDeviceCount > 0)
        vkResult = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());
    if (vkResult != VK_SUCCESS)
    {
        std::fprintf(stderr, "vkEnumeratePhysicalDevices failed: %d\n", static_cast<int>(vkResult));
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalProperties{};
    for (VkPhysicalDevice candidate : physicalDevices)
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        std::printf("Vulkan device: vendor=0x%04x device=0x%04x name=%s\n", properties.vendorID,
                    properties.deviceID, properties.deviceName);
        if (properties.vendorID == 0x1002 && physicalDevice == VK_NULL_HANDLE)
        {
            physicalDevice = candidate;
            physicalProperties = properties;
        }
    }
    if (physicalDevice == VK_NULL_HANDLE)
    {
        std::fprintf(stderr, "No AMD Vulkan device was found\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    std::printf("Target: HIP=%s, Vulkan=%s (device 0x%04x)\n", hipProperties.name,
                physicalProperties.deviceName, physicalProperties.deviceID);

    VkPhysicalDeviceExternalBufferInfo externalBufferInfo{};
    externalBufferInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    externalBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    externalBufferInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkExternalBufferProperties externalBufferProperties{};
    externalBufferProperties.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(physicalDevice, &externalBufferInfo, &externalBufferProperties);
    const VkExternalMemoryProperties& externalMemory = externalBufferProperties.externalMemoryProperties;
    std::printf("RADV OPAQUE_FD buffer: features=0x%x compatible=0x%x\n",
                externalMemory.externalMemoryFeatures, externalMemory.compatibleHandleTypes);
    if ((externalMemory.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) == 0 ||
        (externalMemory.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0)
    {
        std::fprintf(stderr, "RADV does not support importing and exporting OPAQUE_FD buffers\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    uint32_t queueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i)
    {
        if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 && queueFamilies[i].queueCount > 0)
        {
            queueFamilyIndex = i;
            break;
        }
    }
    if (queueFamilyIndex == UINT32_MAX)
    {
        std::fprintf(stderr, "No Vulkan compute queue is available\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    const char* deviceExtensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = 1;
    deviceInfo.ppEnabledExtensionNames = deviceExtensions;

    VkDevice device = VK_NULL_HANDLE;
    vkResult = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
    std::printf("vkCreateDevice: %d\n", static_cast<int>(vkResult));
    if (vkResult != VK_SUCCESS)
    {
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    auto getMemoryFd = vk_proc<PFN_vkGetMemoryFdKHR>(device, "vkGetMemoryFdKHR");
    if (getMemoryFd == nullptr)
    {
        std::fprintf(stderr, "vkGetMemoryFdKHR is unavailable\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkExternalMemoryBufferCreateInfo externalCreateInfo{};
    externalCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = &externalCreateInfo;
    bufferInfo.size = testSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    vkResult = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
    std::printf("vkCreateBuffer(exportable): %d\n", static_cast<int>(vkResult));
    if (vkResult != VK_SUCCESS)
    {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
    const uint32_t memoryTypeIndex = choose_memory_type(physicalDevice, memoryRequirements.memoryTypeBits);
    if (memoryTypeIndex == UINT32_MAX)
    {
        std::fprintf(stderr, "No compatible Vulkan memory type was found\n");
        vkDestroyBuffer(device, buffer, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryAllocateInfo allocationInfo{};
    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.pNext = &exportInfo;
    allocationInfo.allocationSize = memoryRequirements.size;
    allocationInfo.memoryTypeIndex = memoryTypeIndex;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    vkResult = vkAllocateMemory(device, &allocationInfo, nullptr, &memory);
    std::printf("vkAllocateMemory(exportable): %d size=%llu\n", static_cast<int>(vkResult),
                static_cast<unsigned long long>(allocationInfo.allocationSize));
    if (vkResult == VK_SUCCESS)
        vkResult = vkBindBufferMemory(device, buffer, memory, 0);
    if (vkResult != VK_SUCCESS)
    {
        std::fprintf(stderr, "Vulkan memory allocation/bind failed: %d\n", static_cast<int>(vkResult));
        if (memory != VK_NULL_HANDLE)
            vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkMemoryGetFdInfoKHR fdInfo{};
    fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory = memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    vkResult = getMemoryFd(device, &fdInfo, &fd);
    std::printf("vkGetMemoryFdKHR: %d fd=%d\n", static_cast<int>(vkResult), fd);
    if (vkResult != VK_SUCCESS)
    {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    hipExternalMemoryHandleDesc hipHandle{};
    hipHandle.type = hipExternalMemoryHandleTypeOpaqueFd;
    hipHandle.handle.fd = fd;
    hipHandle.size = allocationInfo.allocationSize;
    hipExternalMemory_t hipMemory = nullptr;
    hipResult = hipImportExternalMemory(&hipMemory, &hipHandle);
    print_hip("hipImportExternalMemory(Vulkan OPAQUE_FD)", hipResult);
    if (hipResult != hipSuccess)
    {
        std::fprintf(stderr, "HIP could not import RADV's exported Vulkan memory\n");
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    hipExternalMemoryBufferDesc mapInfo{};
    mapInfo.size = testSize;
    void* hipPointer = nullptr;
    hipResult = hipExternalMemoryGetMappedBuffer(&hipPointer, hipMemory, &mapInfo);
    print_hip("hipExternalMemoryGetMappedBuffer", hipResult);
    if (hipResult == hipSuccess)
    {
        hipResult = hipMemset(hipPointer, testByte, testSize);
        print_hip("hipMemset(imported Vulkan buffer)", hipResult);
    }
    if (hipResult == hipSuccess)
        hipResult = hipDeviceSynchronize();
    print_hip("hipDeviceSynchronize", hipResult);

    std::array<unsigned char, 64> readback{};
    if (hipResult == hipSuccess)
    {
        hipResult = hipMemcpy(readback.data(), hipPointer, readback.size(), hipMemcpyDeviceToHost);
        print_hip("hipMemcpy(imported buffer to host)", hipResult);
    }
    const bool patternMatches = hipResult == hipSuccess &&
        std::all_of(readback.begin(), readback.end(), [](unsigned char value) { return value == testByte; });
    std::printf("HIP readback pattern: %s\n", patternMatches ? "PASS" : "FAIL");

    const bool hipImageMapAvailable = dlsym(RTLD_DEFAULT, "hipExternalMemoryGetMappedMipmappedArray") != nullptr;
    const bool hipSemaphoreImportAvailable = dlsym(RTLD_DEFAULT, "hipImportExternalSemaphore") != nullptr;
    std::printf("HIP external-image mapping export: %s\n", hipImageMapAvailable ? "present" : "absent");
    std::printf("HIP external-semaphore import export: %s\n", hipSemaphoreImportAvailable ? "present" : "absent");

    hipError_t destroyResult = hipDestroyExternalMemory(hipMemory);
    print_hip("hipDestroyExternalMemory(buffer)", destroyResult);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return patternMatches ? 0 : 1;
}
