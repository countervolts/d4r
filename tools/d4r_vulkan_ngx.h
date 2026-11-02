#pragma once
#include <vulkan/vulkan_core.h>

// ABI-compatible subset of NVIDIA nvsdk_ngx_defs_vk.h.
struct NgxVkImage
{
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned int width, height;
};
struct NgxVkResource
{
    union { NgxVkImage image; struct { VkBuffer buffer; unsigned int size; } buffer; } resource;
    int type; // 0 = image view, 1 = buffer
    bool readWrite;
};

