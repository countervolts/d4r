// Vulkan entry points for the engine. A native build links the loader. A build without a Vulkan import
// library (the MinGW NGX shim) defines D4R_ENGINE_VK_TABLE and fills this table once with loadVulkan().
#pragma once
#include <vulkan/vulkan.h>
#ifdef D4R_ENGINE_VK_TABLE
namespace d4r {
struct VkTable {
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
    PFN_vkAllocateMemory AllocateMemory = nullptr;
    PFN_vkBindBufferMemory BindBufferMemory = nullptr;
    PFN_vkBindImageMemory BindImageMemory = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
    PFN_vkCmdBlitImage CmdBlitImage = nullptr;
    PFN_vkCmdClearColorImage CmdClearColorImage = nullptr;
    PFN_vkCmdCopyBuffer CmdCopyBuffer = nullptr;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage = nullptr;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer = nullptr;
    PFN_vkCmdDispatch CmdDispatch = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdUpdateBuffer CmdUpdateBuffer = nullptr;
    PFN_vkCmdWriteTimestamp CmdWriteTimestamp = nullptr;
    PFN_vkCreateBuffer CreateBuffer = nullptr;
    PFN_vkCreateComputePipelines CreateComputePipelines = nullptr;
    PFN_vkCreateDescriptorPool CreateDescriptorPool = nullptr;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout = nullptr;
    PFN_vkCreateImage CreateImage = nullptr;
    PFN_vkCreateImageView CreateImageView = nullptr;
    PFN_vkCreatePipelineLayout CreatePipelineLayout = nullptr;
    PFN_vkCreateSampler CreateSampler = nullptr;
    PFN_vkCreateShaderModule CreateShaderModule = nullptr;
    PFN_vkDestroyBuffer DestroyBuffer = nullptr;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout = nullptr;
    PFN_vkDestroyImage DestroyImage = nullptr;
    PFN_vkDestroyImageView DestroyImageView = nullptr;
    PFN_vkDestroyPipeline DestroyPipeline = nullptr;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout = nullptr;
    PFN_vkDestroySampler DestroySampler = nullptr;
    PFN_vkDestroyShaderModule DestroyShaderModule = nullptr;
    PFN_vkFreeMemory FreeMemory = nullptr;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements = nullptr;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements = nullptr;
    PFN_vkMapMemory MapMemory = nullptr;
    PFN_vkUnmapMemory UnmapMemory = nullptr;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets = nullptr;
};
extern VkTable vkTable;
// False when an entry point is missing. Call before constructing a Model.
bool loadVulkan(PFN_vkGetInstanceProcAddr instanceProc, VkInstance instance, VkDevice device);
} // namespace d4r
#define vkGetPhysicalDeviceFormatProperties ::d4r::vkTable.GetPhysicalDeviceFormatProperties
#define vkGetPhysicalDeviceMemoryProperties ::d4r::vkTable.GetPhysicalDeviceMemoryProperties
#define vkGetPhysicalDeviceProperties ::d4r::vkTable.GetPhysicalDeviceProperties
#define vkAllocateDescriptorSets ::d4r::vkTable.AllocateDescriptorSets
#define vkAllocateMemory ::d4r::vkTable.AllocateMemory
#define vkBindBufferMemory ::d4r::vkTable.BindBufferMemory
#define vkBindImageMemory ::d4r::vkTable.BindImageMemory
#define vkCmdBindDescriptorSets ::d4r::vkTable.CmdBindDescriptorSets
#define vkCmdBindPipeline ::d4r::vkTable.CmdBindPipeline
#define vkCmdBlitImage ::d4r::vkTable.CmdBlitImage
#define vkCmdClearColorImage ::d4r::vkTable.CmdClearColorImage
#define vkCmdCopyBuffer ::d4r::vkTable.CmdCopyBuffer
#define vkCmdCopyBufferToImage ::d4r::vkTable.CmdCopyBufferToImage
#define vkCmdCopyImageToBuffer ::d4r::vkTable.CmdCopyImageToBuffer
#define vkCmdDispatch ::d4r::vkTable.CmdDispatch
#define vkCmdPipelineBarrier ::d4r::vkTable.CmdPipelineBarrier
#define vkCmdUpdateBuffer ::d4r::vkTable.CmdUpdateBuffer
#define vkCmdWriteTimestamp ::d4r::vkTable.CmdWriteTimestamp
#define vkCreateBuffer ::d4r::vkTable.CreateBuffer
#define vkCreateComputePipelines ::d4r::vkTable.CreateComputePipelines
#define vkCreateDescriptorPool ::d4r::vkTable.CreateDescriptorPool
#define vkCreateDescriptorSetLayout ::d4r::vkTable.CreateDescriptorSetLayout
#define vkCreateImage ::d4r::vkTable.CreateImage
#define vkCreateImageView ::d4r::vkTable.CreateImageView
#define vkCreatePipelineLayout ::d4r::vkTable.CreatePipelineLayout
#define vkCreateSampler ::d4r::vkTable.CreateSampler
#define vkCreateShaderModule ::d4r::vkTable.CreateShaderModule
#define vkDestroyBuffer ::d4r::vkTable.DestroyBuffer
#define vkDestroyDescriptorPool ::d4r::vkTable.DestroyDescriptorPool
#define vkDestroyDescriptorSetLayout ::d4r::vkTable.DestroyDescriptorSetLayout
#define vkDestroyImage ::d4r::vkTable.DestroyImage
#define vkDestroyImageView ::d4r::vkTable.DestroyImageView
#define vkDestroyPipeline ::d4r::vkTable.DestroyPipeline
#define vkDestroyPipelineLayout ::d4r::vkTable.DestroyPipelineLayout
#define vkDestroySampler ::d4r::vkTable.DestroySampler
#define vkDestroyShaderModule ::d4r::vkTable.DestroyShaderModule
#define vkFreeMemory ::d4r::vkTable.FreeMemory
#define vkGetBufferMemoryRequirements ::d4r::vkTable.GetBufferMemoryRequirements
#define vkGetImageMemoryRequirements ::d4r::vkTable.GetImageMemoryRequirements
#define vkMapMemory ::d4r::vkTable.MapMemory
#define vkUnmapMemory ::d4r::vkTable.UnmapMemory
#define vkUpdateDescriptorSets ::d4r::vkTable.UpdateDescriptorSets
#endif
