#pragma once
#include <vulkan/vulkan_core.h>
#include <cstddef>

// Image-to-buffer depth copies use the depth aspect's scalar representation,
// not the size of a packed depth+stencil texel.
constexpr size_t d4r_vk_depth_bytes(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D16_UNORM_S8_UINT: return 2;
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return 4;
    default: return 0;
    }
}
constexpr bool d4r_vk_has_stencil(VkFormat format)
{
    return format == VK_FORMAT_D16_UNORM_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT ||
           format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}
constexpr VkImageAspectFlags d4r_vk_copy_aspect(VkFormat format, VkImageAspectFlags requested)
{
    if (d4r_vk_depth_bytes(format))
    {
        const auto allowed = VK_IMAGE_ASPECT_DEPTH_BIT |
                             (d4r_vk_has_stencil(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        return (requested & VK_IMAGE_ASPECT_DEPTH_BIT) && !(requested & ~allowed)
                   ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT) : 0;
    }
    // Depth kept in a colour image (R32_SFLOAT in Indiana Jones) arrives labelled with the depth
    // aspect by Streamline/OptiScaler; the caller's view is still a colour view, so sample it as one.
    return requested == VK_IMAGE_ASPECT_COLOR_BIT || requested == VK_IMAGE_ASPECT_DEPTH_BIT
               ? static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_COLOR_BIT) : 0;
}
constexpr VkImageAspectFlags d4r_vk_barrier_aspect(VkFormat format, VkImageAspectFlags requested)
{
    return d4r_vk_has_stencil(format) ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : requested;
}
