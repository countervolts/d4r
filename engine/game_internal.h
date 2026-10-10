#pragma once
#include "game.h"
#include "vk_dispatch.h"
#include <stdexcept>

namespace d4r::game_detail {
inline void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(result));
}
inline uint32_t memoryType(VkPhysicalDevice physical, uint32_t bits) {
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return i;
    throw std::runtime_error("no device-local Vulkan memory type");
}
// Canonical fallback plane; direct sampled inputs do not use it.
struct Plane {
    VkDevice device = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    ~Plane() {
        if (view) vkDestroyImageView(device, view, nullptr);
        if (image) vkDestroyImage(device, image, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
    void create(VkPhysicalDevice physical, VkDevice dev, VkFormat format, uint32_t width, uint32_t height) {
        device = dev;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D; info.format = format; info.extent = {width, height, 1};
        info.mipLevels = info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        check(vkCreateImage(dev, &info, nullptr, &image), "vkCreateImage");
        VkMemoryRequirements requirements; vkGetImageMemoryRequirements(dev, image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size; allocation.memoryTypeIndex = memoryType(physical, requirements.memoryTypeBits);
        check(vkAllocateMemory(dev, &allocation, nullptr, &memory), "vkAllocateMemory");
        check(vkBindImageMemory(dev, image, memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(dev, &vi, nullptr, &view), "vkCreateImageView");
    }
};
inline void barrier(VkCommandBuffer cb, VkPipelineStageFlags source, VkPipelineStageFlags destination, VkAccessFlags src, VkAccessFlags dst) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = src; b.dstAccessMask = dst;
    vkCmdPipelineBarrier(cb, source, destination, 0, 1, &b, 0, nullptr, 0, nullptr);
}
inline bool directOutput(const GameImage& output, const GameImage& color, const GameImage& motion, const GameImage& depth, bool rgb10) {
    return output.view != VK_NULL_HANDLE && (output.format == VK_FORMAT_R16G16B16A16_SFLOAT ||
        (rgb10 && output.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32)) &&
        output.layout == VK_IMAGE_LAYOUT_GENERAL && output.image != color.image &&
        output.image != motion.image && output.image != depth.image;
}
inline void copyInput(VkCommandBuffer cb, const GameImage& source, VkImage destination, uint32_t width, uint32_t height,
                      VkBuffer depthStaging = VK_NULL_HANDLE) {
    if (source.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        region.imageOffset = {int32_t(source.x), int32_t(source.y), 0}; region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cb, source.image, source.layout, depthStaging, 1, &region);
        barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; region.imageOffset = {0, 0, 0};
        vkCmdCopyBufferToImage(cb, depthStaging, destination, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    } else {
        VkImageBlit region{};
        region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[0] = {int32_t(source.x), int32_t(source.y), 0};
        region.srcOffsets[1] = {int32_t(source.x + width), int32_t(source.y + height), 1};
        region.dstOffsets[1] = {int32_t(width), int32_t(height), 1};
        vkCmdBlitImage(cb, source.image, source.layout, destination, VK_IMAGE_LAYOUT_GENERAL, 1, &region, VK_FILTER_NEAREST);
    }
}
// Both adapters allocate conversion resources only when an input cannot be borrowed.
struct CanonicalInputs {
    VkPhysicalDevice physical;
    VkDevice device;
    uint32_t width, height, motionWidth, motionHeight;
    Plane color, motion, depth;
    VkBuffer depthStaging = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    CanonicalInputs(VkPhysicalDevice p, VkDevice d, uint32_t w, uint32_t h, uint32_t mw, uint32_t mh)
        : physical(p), device(d), width(w), height(h), motionWidth(mw), motionHeight(mh) {}
    ~CanonicalInputs() {
        if (depthStaging) vkDestroyBuffer(device, depthStaging, nullptr);
        if (depthMemory) vkFreeMemory(device, depthMemory, nullptr);
    }
    void ensure(VkCommandBuffer cb, Plane& plane, VkFormat format, uint32_t w, uint32_t h) {
        if (plane.image) return;
        plane.create(physical, device, format, w, h);
        VkImageMemoryBarrier layout{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        layout.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        layout.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; layout.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        layout.srcQueueFamilyIndex = layout.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        layout.image = plane.image; layout.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &layout);
    }
    void record(VkCommandBuffer cb, const GameImage& col, const GameImage& mv, const GameImage& dep,
                uint32_t w, uint32_t h, uint32_t mw, uint32_t mh, bool directColor, bool directMotion, bool directDepth) {
        if (directColor && directMotion && directDepth) return;
        if (!directColor) ensure(cb, color, VK_FORMAT_R16G16B16A16_SFLOAT, width, height);
        if (!directMotion) ensure(cb, motion, VK_FORMAT_R16G16_SFLOAT, motionWidth, motionHeight);
        if (!directDepth) {
            ensure(cb, depth, VK_FORMAT_R32_SFLOAT, width, height);
            if ((dep.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !depthStaging) {
                VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                info.size = VkDeviceSize(width) * height * 4;
                info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                check(vkCreateBuffer(device, &info, nullptr, &depthStaging), "vkCreateBuffer");
                VkMemoryRequirements requirements; vkGetBufferMemoryRequirements(device, depthStaging, &requirements);
                VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex = memoryType(physical, requirements.memoryTypeBits);
                check(vkAllocateMemory(device, &allocation, nullptr, &depthMemory), "vkAllocateMemory");
                check(vkBindBufferMemory(device, depthStaging, depthMemory, 0), "vkBindBufferMemory");
            }
        }
        barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        if (!directColor) copyInput(cb, col, color.image, w, h);
        if (!directMotion) copyInput(cb, mv, motion.image, mw, mh);
        if (!directDepth) copyInput(cb, dep, depth.image, w, h, depthStaging);
    }
};
inline void copyOutput(VkCommandBuffer cb, VkImage source, const GameImage& destination, uint32_t width, uint32_t height) {
    VkImageBlit region{};
    region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[1] = {int32_t(width), int32_t(height), 1};
    region.dstOffsets[0] = {int32_t(destination.x), int32_t(destination.y), 0};
    region.dstOffsets[1] = {int32_t(destination.x + width), int32_t(destination.y + height), 1};
    vkCmdBlitImage(cb, source, VK_IMAGE_LAYOUT_GENERAL, destination.image, destination.layout, 1, &region, VK_FILTER_NEAREST);
    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}
} // namespace d4r::game_detail
