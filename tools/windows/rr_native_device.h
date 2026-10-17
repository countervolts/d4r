#pragma once
#include <windows.h>
#include <volk.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rr_bridge {
// Owns only Vulkan objects. The caller supplies the actual D3D12 adapter LUID;
// production callers must obtain it from the borrowed game's device.
struct NativeDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    NativeDevice() = default;
    NativeDevice(const NativeDevice&) = delete;
    NativeDevice& operator=(const NativeDevice&) = delete;
    static void check(VkResult result, const char* operation) {
        if (result != VK_SUCCESS)
            throw std::runtime_error(std::string(operation) + " VkResult=" + std::to_string(result));
    }
    void initialize(const LUID& luid, bool inference) {
        if (instance) throw std::runtime_error("RR Vulkan device already initialized");
        check(volkInitialize(), "Vulkan loader");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &instance), "Vulkan instance"); volkLoadInstance(instance);
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "Vulkan device count");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "Vulkan devices");
        for (auto candidate : devices) {
            VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
            vkGetPhysicalDeviceProperties2(candidate, &properties);
            if (id.deviceLUIDValid && !std::memcmp(id.deviceLUID, &luid, VK_LUID_SIZE)) physical = candidate;
        }
        if (!physical) throw std::runtime_error("D3D12/Vulkan adapter LUID mismatch");
        uint32_t countFamilies = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &countFamilies, nullptr);
        std::vector<VkQueueFamilyProperties> properties(countFamilies);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &countFamilies, properties.data());
        family = countFamilies;
        for (uint32_t i = 0; i < countFamilies; ++i)
            if (properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
        if (family == countFamilies) throw std::runtime_error("No Vulkan compute queue");
        const bool fp8 = inference && std::string(D4R_TARGET_ARCH).rfind("gfx12", 0) == 0;
        if (fp8) {
            uint32_t extensionCount = 0;
            check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, nullptr), "RR extension count");
            std::vector<VkExtensionProperties> available(extensionCount);
            check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, available.data()), "RR extensions");
            bool supported = false;
            for (const auto& extension : available)
                supported |= std::strcmp(extension.extensionName, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME) == 0;
            if (!supported) throw std::runtime_error("RDNA4 native RR requires VK_EXT_shader_float8");
        }
        VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
            fp8 ? &f8 : nullptr};
        VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &cm};
        VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
        VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
        vkGetPhysicalDeviceFeatures2(physical, &features);
        if (!v12.timelineSemaphore) throw std::runtime_error("Timeline semaphore unsupported");
        if (fp8 && (!f8.shaderFloat8 || !f8.shaderFloat8CooperativeMatrix))
            throw std::runtime_error("RDNA4 native RR requires FP8 cooperative matrices");
        if (inference && (!cm.cooperativeMatrix || !v12.shaderFloat16 || !v11.storageBuffer16BitAccess || !v13.subgroupSizeControl))
            throw std::runtime_error("Missing RR cooperative-matrix/fp16/subgroup feature");
        if (!inference) v13.pNext = nullptr;
        float priority = 1;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
        std::vector<const char*> extensions{"VK_KHR_external_memory_win32", "VK_KHR_external_semaphore_win32"};
        if (inference) extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        if (fp8) extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
        di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = uint32_t(extensions.size()); di.ppEnabledExtensionNames = extensions.data();
        check(vkCreateDevice(physical, &di, nullptr, &device), "Vulkan device"); volkLoadDevice(device);
        vkGetDeviceQueue(device, family, 0, &queue);
    }
    // Callers must retire submitted work before destroying this owner. A failed
    // submission may instead quarantine the owner and its dependent resources.
    ~NativeDevice() {
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};
}
