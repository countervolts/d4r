// d4r engine: exporting a Vulkan allocation for HIP (hip_net.h) - an opaque file descriptor on Linux, an opaque
// NT handle on Windows. Header-only; the replay runners use it when built with D4R_ENGINE_HIP.
#pragma once
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#else
#include <vulkan/vulkan.h>
#endif
#include <stdexcept>
#include "hip_net.h"
namespace d4r {
#ifdef _WIN32
constexpr VkExternalMemoryHandleTypeFlagBits ExternalMemoryHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr const char* ExternalMemoryExtension = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
#else
constexpr VkExternalMemoryHandleTypeFlagBits ExternalMemoryHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr const char* ExternalMemoryExtension = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
#endif
// The handle of an allocation made with VkExportMemoryAllocateInfo{ExternalMemoryHandleType}.
inline ExternalMemoryHandle exportMemory(VkDevice device, VkDeviceMemory memory) {
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    info.memory = memory; info.handleType = ExternalMemoryHandleType;
    const auto get = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    if (!get) throw std::runtime_error("vkGetMemoryWin32HandleKHR is unavailable");
    HANDLE handle = nullptr;
    if (get(device, &info, &handle) != VK_SUCCESS) throw std::runtime_error("vkGetMemoryWin32HandleKHR failed");
    return handle;
#else
    VkMemoryGetFdInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    info.memory = memory; info.handleType = ExternalMemoryHandleType;
    const auto get = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
    if (!get) throw std::runtime_error("vkGetMemoryFdKHR is unavailable");
    int fd = -1;
    if (get(device, &info, &fd) != VK_SUCCESS) throw std::runtime_error("vkGetMemoryFdKHR failed");
    return fd;
#endif
}
} // namespace d4r
