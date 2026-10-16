#define WIN32_LEAN_AND_MEAN
#include <wchar.h>
#include <windows.h>

#include <vulkan/vulkan.h>
#include "nvsdk_ngx_vk.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

template <typename Target, typename Source> Target copy_function_pointer(Source source)
{
    static_assert(sizeof(Target) == sizeof(Source), "function pointer sizes differ");
    Target target{};
    std::memcpy(&target, &source, sizeof(target));
    return target;
}

template <typename Function> Function load_export(HMODULE module, const char* name)
{
    FARPROC raw = GetProcAddress(module, name);
    return raw ? copy_function_pointer<Function>(raw) : nullptr;
}

static const char* ngx_result_name(NVSDK_NGX_Result result)
{
    switch (result)
    {
    case NVSDK_NGX_Result_Success:
        return "Success";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported:
        return "FeatureNotSupported";
    case NVSDK_NGX_Result_FAIL_PlatformError:
        return "PlatformError";
    case NVSDK_NGX_Result_FAIL_InvalidParameter:
        return "InvalidParameter";
    case NVSDK_NGX_Result_FAIL_NotInitialized:
        return "NotInitialized";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:
        return "UnableToInitializeFeature";
    case NVSDK_NGX_Result_FAIL_OutOfDate:
        return "OutOfDate";
    default:
        return "Other";
    }
}

static void print_ngx_result(const char* stage, NVSDK_NGX_Result result)
{
    std::printf("%s: result=0x%08x (%s)\n", stage, static_cast<unsigned int>(result), ngx_result_name(result));
}

static std::wstring utf8_to_wide(const char* input)
{
    const int required = MultiByteToWideChar(CP_UTF8, 0, input, -1, nullptr, 0);
    if (required <= 1)
        return {};
    std::wstring output(static_cast<size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, input, -1, output.data(), required);
    output.resize(static_cast<size_t>(required - 1));
    return output;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::fprintf(stderr, "usage: ngx_vulkan_probe.exe PATH_TO_NGX_CORE_DLL PATH_TO_NVNGX_DLSS_DLL PATH_TO_WRITABLE_APPDATA\n");
        return 2;
    }

    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);

    const char* ngxCorePath = argv[1];
    const char* featureDllPath = argv[2];
    std::wstring appData = utf8_to_wide(argv[3]);
    std::wstring dllSearchDirectory = utf8_to_wide(featureDllPath);
    if (appData.empty() || dllSearchDirectory.empty())
    {
        std::fprintf(stderr, "Invalid UTF-8 feature-DLL path or appdata path argument\n");
        return 2;
    }
    const size_t separator = dllSearchDirectory.find_last_of(L"\\/");
    if (separator == std::wstring::npos)
        dllSearchDirectory = L".";
    else
        dllSearchDirectory.resize(separator);

    const wchar_t* featurePaths[] = {dllSearchDirectory.c_str()};
    NVSDK_NGX_FeatureCommonInfo featureCommonInfo{};
    featureCommonInfo.PathListInfo.Path = featurePaths;
    featureCommonInfo.PathListInfo.Length = 1;

    HMODULE ngxCore = LoadLibraryA(ngxCorePath);
    if (ngxCore == nullptr)
    {
        std::fprintf(stderr, "LoadLibraryA(%s) failed: GetLastError=%lu\n", ngxCorePath, GetLastError());
        return 1;
    }
    HMODULE ngxFeature = LoadLibraryA(featureDllPath);
    if (ngxFeature == nullptr)
    {
        std::fprintf(stderr, "LoadLibraryA(%s) failed: GetLastError=%lu\n", featureDllPath, GetLastError());
        FreeLibrary(ngxCore);
        return 1;
    }
    std::printf("Loaded NGX core: %s\n", ngxCorePath);
    std::printf("DLSS SR feature search directory: %ls\n", dllSearchDirectory.c_str());

    auto getRequirements = load_export<decltype(&NVSDK_NGX_VULKAN_GetFeatureRequirements)>(
        ngxFeature, "NVSDK_NGX_VULKAN_GetFeatureRequirements");
    auto getInstanceExtensions = load_export<decltype(&NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements)>(
        ngxFeature, "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements");
    auto getDeviceExtensions = load_export<decltype(&NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements)>(
        ngxFeature, "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements");
    if (getRequirements == nullptr || getInstanceExtensions == nullptr || getDeviceExtensions == nullptr)
    {
        std::fprintf(stderr, "The DLL does not expose the required Vulkan discovery exports\n");
        FreeLibrary(ngxFeature);
        FreeLibrary(ngxCore);
        return 1;
    }

    NVSDK_NGX_FeatureDiscoveryInfo discovery{};
    discovery.SDKVersion = static_cast<NVSDK_NGX_Version>(0x13); // Version resource in tested DLLs reports API 0x13.
    discovery.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    discovery.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    discovery.Identifier.v.ApplicationId = 241534723ULL; // NGXAppId value in this DLL's VERSIONINFO resource.
    discovery.ApplicationDataPath = appData.c_str();
    discovery.FeatureInfo = &featureCommonInfo;
    std::printf("Feature discovery: SDK=0x13 appId=%llu searchPath=%ls appData=%ls\n",
                discovery.Identifier.v.ApplicationId, dllSearchDirectory.c_str(), appData.c_str());

    uint32_t instanceExtensionCount = 0;
    VkExtensionProperties* instanceExtensions = nullptr;
    NVSDK_NGX_Result result = getInstanceExtensions(&discovery, &instanceExtensionCount, &instanceExtensions);
    print_ngx_result("VULKAN_GetFeatureInstanceExtensionRequirements", result);
    std::printf("required instance extensions: %u\n", instanceExtensionCount);
    for (uint32_t i = 0; i < instanceExtensionCount && instanceExtensions != nullptr; ++i)
        std::printf("  %s (%u)\n", instanceExtensions[i].extensionName, instanceExtensions[i].specVersion);

    HMODULE vulkan = LoadLibraryA("vulkan-1.dll");
    if (vulkan == nullptr)
    {
        std::fprintf(stderr, "LoadLibraryA(vulkan-1.dll) failed: GetLastError=%lu\n", GetLastError());
        FreeLibrary(ngxCore);
        FreeLibrary(ngxFeature);
        return 1;
    }
    auto getInstanceProcAddr = load_export<PFN_vkGetInstanceProcAddr>(vulkan, "vkGetInstanceProcAddr");
    if (getInstanceProcAddr == nullptr)
    {
        std::fprintf(stderr, "vulkan-1.dll has no vkGetInstanceProcAddr export\n");
        FreeLibrary(vulkan);
        FreeLibrary(ngxFeature);
        FreeLibrary(ngxCore);
        return 1;
    }

    auto createInstance = copy_function_pointer<PFN_vkCreateInstance>(
        getInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (createInstance == nullptr)
    {
        std::fprintf(stderr, "vkCreateInstance is unavailable\n");
        FreeLibrary(vulkan);
        FreeLibrary(ngxFeature);
        FreeLibrary(ngxCore);
        return 1;
    }

    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "d4r NGX Vulkan discovery probe";
    application.applicationVersion = 1;
    application.pEngineName = "d4r";
    application.engineVersion = 1;
    application.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &application;
    std::vector<const char*> enabledInstanceExtensions;
    for (uint32_t i = 0; i < instanceExtensionCount && instanceExtensions != nullptr; ++i)
        enabledInstanceExtensions.push_back(instanceExtensions[i].extensionName);
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledInstanceExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data();

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult vkCreateResult = createInstance(&createInfo, nullptr, &instance);
    std::printf("vkCreateInstance: result=%d\n", static_cast<int>(vkCreateResult));
    if (vkCreateResult != VK_SUCCESS)
    {
        FreeLibrary(vulkan);
        FreeLibrary(ngxFeature);
        FreeLibrary(ngxCore);
        return 1;
    }

    auto enumeratePhysicalDevices = copy_function_pointer<PFN_vkEnumeratePhysicalDevices>(
        getInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
    auto enumerateDeviceExtensions = copy_function_pointer<PFN_vkEnumerateDeviceExtensionProperties>(
        getInstanceProcAddr(instance, "vkEnumerateDeviceExtensionProperties"));
    auto getPhysicalDeviceProperties = copy_function_pointer<PFN_vkGetPhysicalDeviceProperties>(
        getInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties"));
    auto destroyInstance = copy_function_pointer<PFN_vkDestroyInstance>(
        getInstanceProcAddr(instance, "vkDestroyInstance"));
    auto getQueueFamilyProperties = copy_function_pointer<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        getInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
    auto createDevice = copy_function_pointer<PFN_vkCreateDevice>(getInstanceProcAddr(instance, "vkCreateDevice"));
    auto getDeviceProcAddr = copy_function_pointer<PFN_vkGetDeviceProcAddr>(
        getInstanceProcAddr(instance, "vkGetDeviceProcAddr"));
    if (enumeratePhysicalDevices == nullptr || enumerateDeviceExtensions == nullptr ||
        getPhysicalDeviceProperties == nullptr || destroyInstance == nullptr || getQueueFamilyProperties == nullptr ||
        createDevice == nullptr || getDeviceProcAddr == nullptr)
    {
        std::fprintf(stderr, "Required Vulkan instance functions are unavailable\n");
        FreeLibrary(vulkan);
        FreeLibrary(ngxFeature);
        FreeLibrary(ngxCore);
        return 1;
    }

    uint32_t deviceCount = 0;
    VkResult enumerateResult = enumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (enumerateResult != VK_SUCCESS || deviceCount == 0)
    {
        std::fprintf(stderr, "vkEnumeratePhysicalDevices(count) failed: result=%d count=%u\n",
                     static_cast<int>(enumerateResult), deviceCount);
        destroyInstance(instance, nullptr);
        FreeLibrary(vulkan);
        FreeLibrary(ngxFeature);
        FreeLibrary(ngxCore);
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    enumerateResult = enumeratePhysicalDevices(instance, &deviceCount, devices.data());
    if (enumerateResult != VK_SUCCESS)
    {
        std::fprintf(stderr, "vkEnumeratePhysicalDevices(list) failed: result=%d\n", static_cast<int>(enumerateResult));
        destroyInstance(instance, nullptr);
        FreeLibrary(vulkan);
        FreeLibrary(ngxCore);
        return 1;
    }

    VkPhysicalDevice amdDevice = VK_NULL_HANDLE;
    for (VkPhysicalDevice device : devices)
    {
        VkPhysicalDeviceProperties properties{};
        getPhysicalDeviceProperties(device, &properties);
        std::printf("Vulkan device: vendor=0x%04x device=0x%04x type=%d name=%s\n", properties.vendorID,
                    properties.deviceID, static_cast<int>(properties.deviceType), properties.deviceName);
        if (properties.vendorID == 0x1002 && amdDevice == VK_NULL_HANDLE)
            amdDevice = device;
    }

    if (amdDevice == VK_NULL_HANDLE)
    {
        std::fprintf(stderr, "No AMD Vulkan physical device was enumerated\n");
        destroyInstance(instance, nullptr);
        FreeLibrary(vulkan);
        FreeLibrary(ngxCore);
        return 1;
    }

    NVSDK_NGX_FeatureRequirement requirement{};
    result = getRequirements(instance, amdDevice, &discovery, &requirement);
    print_ngx_result("VULKAN_GetFeatureRequirements(DLSS SR)", result);
    if (result == NVSDK_NGX_Result_Success)
    {
        std::printf("feature support flags: 0x%08x\n", static_cast<unsigned int>(requirement.FeatureSupported));
        std::printf("minimum hardware architecture: %u\n", requirement.MinHWArchitecture);
        std::printf("minimum OS version: %s\n", requirement.MinOSVersion);
    }

    uint32_t deviceExtensionCount = 0;
    VkExtensionProperties* deviceExtensions = nullptr;
    VkExtensionProperties deviceExtensionStorage[32]{};
    deviceExtensionCount = 32;
    deviceExtensions = deviceExtensionStorage;
    result = getDeviceExtensions(instance, amdDevice, &discovery, &deviceExtensionCount, &deviceExtensions);
    print_ngx_result("VULKAN_GetFeatureDeviceExtensionRequirements", result);
    if (result != NVSDK_NGX_Result_Success)
        deviceExtensionCount = 0;
    std::printf("required device extensions: %u, properties pointer=%p\n", deviceExtensionCount,
                static_cast<void*>(deviceExtensions));
    for (uint32_t i = 0; i < deviceExtensionCount && i < 32 && deviceExtensions != nullptr; ++i)
        std::printf("  %s (%u)\n", deviceExtensions[i].extensionName, deviceExtensions[i].specVersion);

    uint32_t availableExtensionCount = 0;
    VkResult extensionResult = enumerateDeviceExtensions(amdDevice, nullptr, &availableExtensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
    if (extensionResult == VK_SUCCESS && availableExtensionCount > 0)
        extensionResult = enumerateDeviceExtensions(amdDevice, nullptr, &availableExtensionCount, availableExtensions.data());
    std::printf("AMD device-extension enumeration: result=%d count=%u\n", static_cast<int>(extensionResult),
                availableExtensionCount);
    for (uint32_t i = 0; i < deviceExtensionCount && i < 32 && deviceExtensions != nullptr; ++i)
    {
        bool supported = false;
        for (const auto& extension : availableExtensions)
        {
            if (std::strcmp(extension.extensionName, deviceExtensions[i].extensionName) == 0)
            {
                supported = true;
                break;
            }
        }
        std::printf("  requirement availability: %-42s %s\n", deviceExtensions[i].extensionName,
                    supported ? "supported" : "missing");
    }

    std::vector<const char*> enabledDeviceExtensions;
    for (uint32_t i = 0; i < deviceExtensionCount && i < 32 && deviceExtensions != nullptr; ++i)
    {
        for (const auto& extension : availableExtensions)
        {
            if (std::strcmp(extension.extensionName, deviceExtensions[i].extensionName) == 0)
            {
                enabledDeviceExtensions.push_back(deviceExtensions[i].extensionName);
                break;
            }
        }
    }

    uint32_t queueFamilyCount = 0;
    getQueueFamilyProperties(amdDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    getQueueFamilyProperties(amdDevice, &queueFamilyCount, queueFamilies.data());
    uint32_t queueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i)
    {
        const VkQueueFlags requiredQueueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if ((queueFamilies[i].queueFlags & requiredQueueFlags) == requiredQueueFlags && queueFamilies[i].queueCount > 0)
        {
            queueFamilyIndex = i;
            break;
        }
    }

    if (queueFamilyIndex == UINT32_MAX)
    {
        std::fprintf(stderr, "No graphics+compute Vulkan queue family was found\n");
        destroyInstance(instance, nullptr);
        FreeLibrary(vulkan);
        FreeLibrary(ngxFeature);
        FreeLibrary(ngxCore);
        return 1;
    }

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();

    VkDevice device = VK_NULL_HANDLE;
    const VkResult vkDeviceResult = createDevice(amdDevice, &deviceInfo, nullptr, &device);
    std::printf("vkCreateDevice (only available NGX-required extensions): result=%d enabled-extension-count=%u\n",
                static_cast<int>(vkDeviceResult), deviceInfo.enabledExtensionCount);
    if (vkDeviceResult != VK_SUCCESS)
    {
        destroyInstance(instance, nullptr);
        FreeLibrary(vulkan);
        FreeLibrary(ngxFeature);
        FreeLibrary(ngxCore);
        return 1;
    }

    auto destroyDevice = copy_function_pointer<PFN_vkDestroyDevice>(getDeviceProcAddr(device, "vkDestroyDevice"));
    auto init = load_export<decltype(&NVSDK_NGX_VULKAN_Init)>(ngxCore, "NVSDK_NGX_VULKAN_Init");
    auto shutdown = load_export<decltype(&NVSDK_NGX_VULKAN_Shutdown1)>(ngxCore, "NVSDK_NGX_VULKAN_Shutdown1");
    if (init == nullptr || destroyDevice == nullptr)
    {
        std::fprintf(stderr, "NGX Vulkan init or Vulkan device cleanup export is unavailable\n");
        if (destroyDevice != nullptr)
            destroyDevice(device, nullptr);
        destroyInstance(instance, nullptr);
        FreeLibrary(vulkan);
        FreeLibrary(ngxFeature);
        FreeLibrary(ngxCore);
        return 1;
    }

    result = init(discovery.Identifier.v.ApplicationId, discovery.ApplicationDataPath, instance, amdDevice, device,
                  getInstanceProcAddr, getDeviceProcAddr, &featureCommonInfo, discovery.SDKVersion);
    print_ngx_result("VULKAN_Init(DLSS SR)", result);
    if (result == NVSDK_NGX_Result_Success && shutdown != nullptr)
    {
        NVSDK_NGX_Result shutdownResult = shutdown(device);
        print_ngx_result("VULKAN_Shutdown1", shutdownResult);
    }

    destroyDevice(device, nullptr);
    destroyInstance(instance, nullptr);
    FreeLibrary(vulkan);
    FreeLibrary(ngxFeature);
    FreeLibrary(ngxCore);
    return 0;
}
