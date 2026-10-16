#pragma once
#include "rr_native_device.h"
#include <vulkan/vulkan_win32.h>

namespace rr_bridge {
// Imports the caller's D3D12 shared packet and fence. Exported Win32 handles
// remain caller-owned; Vulkan keeps its own references after successful import.
struct NativePacket {
    NativeDevice& native;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    explicit NativePacket(NativeDevice& owner) : native(owner) {}
    NativePacket(const NativePacket&) = delete;
    NativePacket& operator=(const NativePacket&) = delete;
    void initialize(VkDeviceSize bytes, HANDLE resource, HANDLE fence) {
        if (buffer || !bytes || !resource || !fence)
            throw std::runtime_error("Invalid RR shared packet import");
        auto memoryProperties = reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            vkGetDeviceProcAddr(native.device, "vkGetMemoryWin32HandlePropertiesKHR"));
        auto importSemaphore = reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(native.device, "vkImportSemaphoreWin32HandleKHR"));
        if (!memoryProperties || !importSemaphore) throw std::runtime_error("Missing Win32 import entry points");
        VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external}; bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        NativeDevice::check(vkCreateBuffer(native.device, &bi, nullptr, &buffer), "External Vulkan buffer");
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(native.device, buffer, &req);
        VkMemoryWin32HandlePropertiesKHR handleProperties{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
        NativeDevice::check(memoryProperties(native.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
            resource, &handleProperties), "D3D12 resource memory types");
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(native.physical, &properties);
        const uint32_t bits = req.memoryTypeBits & handleProperties.memoryTypeBits;
        uint32_t type = properties.memoryTypeCount;
        for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                type = i; break;
            }
        if (type == properties.memoryTypeCount) throw std::runtime_error("No compatible RR packet memory type");
        VkImportMemoryWin32HandleInfoKHR imported{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
        imported.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT; imported.handle = resource;
        VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, &imported};
        dedicated.buffer = buffer;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &dedicated};
        ai.allocationSize = req.size; ai.memoryTypeIndex = type;
        NativeDevice::check(vkAllocateMemory(native.device, &ai, nullptr, &memory), "Dedicated D3D12 memory import");
        NativeDevice::check(vkBindBufferMemory(native.device, buffer, memory, 0), "Bind imported buffer");
        VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline};
        NativeDevice::check(vkCreateSemaphore(native.device, &sci, nullptr, &semaphore), "Vulkan timeline semaphore");
        VkImportSemaphoreWin32HandleInfoKHR si{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
        si.semaphore = semaphore; si.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT; si.handle = fence;
        NativeDevice::check(importSemaphore(native.device, &si), "D3D12 fence import");
    }
    // The owner must retire GPU submissions before destroying an import.
    ~NativePacket() {
        if (semaphore) vkDestroySemaphore(native.device, semaphore, nullptr);
        if (buffer) vkDestroyBuffer(native.device, buffer, nullptr);
        if (memory) vkFreeMemory(native.device, memory, nullptr);
    }
};
}
