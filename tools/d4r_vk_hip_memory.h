// Exporting Vulkan memory to HIP: an opaque fd on Linux, an opaque NT handle on Windows.
// Include after <hip/hip_runtime_api.h> (or hip_runtime.h) and <vulkan/vulkan.h>.
#pragma once
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#define D4R_VK_MEMORY_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
#define D4R_VK_EXTERNAL_MEMORY_EXTENSION VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
#define D4R_VK_MEMORY_HANDLE_NAME "OPAQUE_WIN32"
#else
#include <unistd.h>
#define D4R_VK_MEMORY_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
#define D4R_VK_EXTERNAL_MEMORY_EXTENSION VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
#define D4R_VK_MEMORY_HANDLE_NAME "OPAQUE_FD"
#endif

// Exports `memory` and fills the HIP import description (type, handle, size).
inline VkResult d4r_export_vk_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize size,
                                     hipExternalMemoryHandleDesc* desc)
{
    *desc = {};
    desc->size = size;
#ifdef _WIN32
    const auto getHandle =
        reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    if (getHandle == nullptr)
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    VkMemoryGetWin32HandleInfoKHR info = {VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    info.memory = memory;
    info.handleType = D4R_VK_MEMORY_HANDLE_TYPE;
    HANDLE handle = nullptr;
    const VkResult result = getHandle(device, &info, &handle);
    desc->type = hipExternalMemoryHandleTypeOpaqueWin32;
    desc->handle.win32.handle = handle;
#else
    const auto getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
    if (getFd == nullptr)
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    VkMemoryGetFdInfoKHR info = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    info.memory = memory;
    info.handleType = D4R_VK_MEMORY_HANDLE_TYPE;
    int fd = -1;
    const VkResult result = getFd(device, &info, &fd);
    desc->type = hipExternalMemoryHandleTypeOpaqueFd;
    desc->handle.fd = fd;
#endif
    return result;
}

// HIP keeps its own reference once the import returns, so the exported handle can be closed.
inline void d4r_close_exported_vk_memory(const hipExternalMemoryHandleDesc& desc)
{
#ifdef _WIN32
    if (desc.handle.win32.handle != nullptr)
        CloseHandle(desc.handle.win32.handle);
#else
    if (desc.handle.fd >= 0)
        close(desc.handle.fd);
#endif
}
