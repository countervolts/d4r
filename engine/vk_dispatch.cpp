#include "vk_dispatch.h"
#ifdef D4R_ENGINE_VK_TABLE
namespace d4r {
VkTable vkTable;
bool loadVulkan(PFN_vkGetInstanceProcAddr instanceProc, VkInstance instance, VkDevice device) {
    if (!instanceProc) return false;
    auto deviceProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(instanceProc(instance, "vkGetDeviceProcAddr"));
    if (!deviceProc) return false;
    bool ok = true;
    ok &= (vkTable.GetPhysicalDeviceFormatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(instanceProc(instance, "vkGetPhysicalDeviceFormatProperties"))) != nullptr;
    ok &= (vkTable.GetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(instanceProc(instance, "vkGetPhysicalDeviceMemoryProperties"))) != nullptr;
    ok &= (vkTable.GetPhysicalDeviceProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(instanceProc(instance, "vkGetPhysicalDeviceProperties"))) != nullptr;
    ok &= (vkTable.AllocateDescriptorSets = reinterpret_cast<PFN_vkAllocateDescriptorSets>(deviceProc(device, "vkAllocateDescriptorSets"))) != nullptr;
    ok &= (vkTable.AllocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(deviceProc(device, "vkAllocateMemory"))) != nullptr;
    ok &= (vkTable.BindBufferMemory = reinterpret_cast<PFN_vkBindBufferMemory>(deviceProc(device, "vkBindBufferMemory"))) != nullptr;
    ok &= (vkTable.BindImageMemory = reinterpret_cast<PFN_vkBindImageMemory>(deviceProc(device, "vkBindImageMemory"))) != nullptr;
    ok &= (vkTable.CmdBindDescriptorSets = reinterpret_cast<PFN_vkCmdBindDescriptorSets>(deviceProc(device, "vkCmdBindDescriptorSets"))) != nullptr;
    ok &= (vkTable.CmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(deviceProc(device, "vkCmdBindPipeline"))) != nullptr;
    ok &= (vkTable.CmdBlitImage = reinterpret_cast<PFN_vkCmdBlitImage>(deviceProc(device, "vkCmdBlitImage"))) != nullptr;
    ok &= (vkTable.CmdClearColorImage = reinterpret_cast<PFN_vkCmdClearColorImage>(deviceProc(device, "vkCmdClearColorImage"))) != nullptr;
    ok &= (vkTable.CmdCopyBuffer = reinterpret_cast<PFN_vkCmdCopyBuffer>(deviceProc(device, "vkCmdCopyBuffer"))) != nullptr;
    ok &= (vkTable.CmdCopyBufferToImage = reinterpret_cast<PFN_vkCmdCopyBufferToImage>(deviceProc(device, "vkCmdCopyBufferToImage"))) != nullptr;
    ok &= (vkTable.CmdCopyImageToBuffer = reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(deviceProc(device, "vkCmdCopyImageToBuffer"))) != nullptr;
    ok &= (vkTable.CmdDispatch = reinterpret_cast<PFN_vkCmdDispatch>(deviceProc(device, "vkCmdDispatch"))) != nullptr;
    ok &= (vkTable.CmdPipelineBarrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(deviceProc(device, "vkCmdPipelineBarrier"))) != nullptr;
    ok &= (vkTable.CmdUpdateBuffer = reinterpret_cast<PFN_vkCmdUpdateBuffer>(deviceProc(device, "vkCmdUpdateBuffer"))) != nullptr;
    ok &= (vkTable.CmdWriteTimestamp = reinterpret_cast<PFN_vkCmdWriteTimestamp>(deviceProc(device, "vkCmdWriteTimestamp"))) != nullptr;
    ok &= (vkTable.CreateBuffer = reinterpret_cast<PFN_vkCreateBuffer>(deviceProc(device, "vkCreateBuffer"))) != nullptr;
    ok &= (vkTable.CreateComputePipelines = reinterpret_cast<PFN_vkCreateComputePipelines>(deviceProc(device, "vkCreateComputePipelines"))) != nullptr;
    ok &= (vkTable.CreateDescriptorPool = reinterpret_cast<PFN_vkCreateDescriptorPool>(deviceProc(device, "vkCreateDescriptorPool"))) != nullptr;
    ok &= (vkTable.CreateDescriptorSetLayout = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(deviceProc(device, "vkCreateDescriptorSetLayout"))) != nullptr;
    ok &= (vkTable.CreateImage = reinterpret_cast<PFN_vkCreateImage>(deviceProc(device, "vkCreateImage"))) != nullptr;
    ok &= (vkTable.CreateImageView = reinterpret_cast<PFN_vkCreateImageView>(deviceProc(device, "vkCreateImageView"))) != nullptr;
    ok &= (vkTable.CreatePipelineLayout = reinterpret_cast<PFN_vkCreatePipelineLayout>(deviceProc(device, "vkCreatePipelineLayout"))) != nullptr;
    ok &= (vkTable.CreateSampler = reinterpret_cast<PFN_vkCreateSampler>(deviceProc(device, "vkCreateSampler"))) != nullptr;
    ok &= (vkTable.CreateShaderModule = reinterpret_cast<PFN_vkCreateShaderModule>(deviceProc(device, "vkCreateShaderModule"))) != nullptr;
    ok &= (vkTable.DestroyBuffer = reinterpret_cast<PFN_vkDestroyBuffer>(deviceProc(device, "vkDestroyBuffer"))) != nullptr;
    ok &= (vkTable.DestroyDescriptorPool = reinterpret_cast<PFN_vkDestroyDescriptorPool>(deviceProc(device, "vkDestroyDescriptorPool"))) != nullptr;
    ok &= (vkTable.DestroyDescriptorSetLayout = reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(deviceProc(device, "vkDestroyDescriptorSetLayout"))) != nullptr;
    ok &= (vkTable.DestroyImage = reinterpret_cast<PFN_vkDestroyImage>(deviceProc(device, "vkDestroyImage"))) != nullptr;
    ok &= (vkTable.DestroyImageView = reinterpret_cast<PFN_vkDestroyImageView>(deviceProc(device, "vkDestroyImageView"))) != nullptr;
    ok &= (vkTable.DestroyPipeline = reinterpret_cast<PFN_vkDestroyPipeline>(deviceProc(device, "vkDestroyPipeline"))) != nullptr;
    ok &= (vkTable.DestroyPipelineLayout = reinterpret_cast<PFN_vkDestroyPipelineLayout>(deviceProc(device, "vkDestroyPipelineLayout"))) != nullptr;
    ok &= (vkTable.DestroySampler = reinterpret_cast<PFN_vkDestroySampler>(deviceProc(device, "vkDestroySampler"))) != nullptr;
    ok &= (vkTable.DestroyShaderModule = reinterpret_cast<PFN_vkDestroyShaderModule>(deviceProc(device, "vkDestroyShaderModule"))) != nullptr;
    ok &= (vkTable.FreeMemory = reinterpret_cast<PFN_vkFreeMemory>(deviceProc(device, "vkFreeMemory"))) != nullptr;
    ok &= (vkTable.GetBufferMemoryRequirements = reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(deviceProc(device, "vkGetBufferMemoryRequirements"))) != nullptr;
    ok &= (vkTable.GetImageMemoryRequirements = reinterpret_cast<PFN_vkGetImageMemoryRequirements>(deviceProc(device, "vkGetImageMemoryRequirements"))) != nullptr;
    ok &= (vkTable.MapMemory = reinterpret_cast<PFN_vkMapMemory>(deviceProc(device, "vkMapMemory"))) != nullptr;
    ok &= (vkTable.UnmapMemory = reinterpret_cast<PFN_vkUnmapMemory>(deviceProc(device, "vkUnmapMemory"))) != nullptr;
    ok &= (vkTable.UpdateDescriptorSets = reinterpret_cast<PFN_vkUpdateDescriptorSets>(deviceProc(device, "vkUpdateDescriptorSets"))) != nullptr;
    return ok;
}
} // namespace d4r
#endif
