// Real-GPU check of the native NGX depth-aspect copy contract (no CUDA/NGX).
#include "d4r_vulkan_depth.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void check(VkResult r) { if (r != VK_SUCCESS) { std::fprintf(stderr,"Vulkan error %d\n",r); std::abort(); } }
int main()
{
    assert(d4r_vk_copy_aspect(VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_ASPECT_STENCIL_BIT)==0);
    assert(d4r_vk_copy_aspect(VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_COLOR_BIT)==0);
    VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ic={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ic.pApplicationInfo=&app;
    VkInstance instance; check(vkCreateInstance(&ic,nullptr,&instance));
    uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr));
    std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()));
    VkPhysicalDevice physical=VK_NULL_HANDLE;
    for(auto candidate:devices) { VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(candidate,&p);if(p.vendorID==0x1002){physical=candidate;std::printf("GPU: %s\n",p.deviceName);break;} }
    assert(physical);
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);
    std::vector<VkQueueFamilyProperties> families(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());
    uint32_t family=0;while(family<count && !(families[family].queueFlags&VK_QUEUE_GRAPHICS_BIT))++family;assert(family<count);
    float priority=1;
    VkDeviceQueueCreateInfo qi={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
    VkDeviceCreateInfo dc={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dc.queueCreateInfoCount=1;dc.pQueueCreateInfos=&qi;
    VkDevice device;check(vkCreateDevice(physical,&dc,nullptr,&device));
    VkQueue queue;vkGetDeviceQueue(device,family,0,&queue);
    VkPhysicalDeviceMemoryProperties memory;vkGetPhysicalDeviceMemoryProperties(physical,&memory);
    auto allocate=[&](VkMemoryRequirements req,VkMemoryPropertyFlags flags){
        uint32_t type=0;while(type<memory.memoryTypeCount && (!(req.memoryTypeBits&(1u<<type)) || (memory.memoryTypes[type].propertyFlags&flags)!=flags))++type;assert(type<memory.memoryTypeCount);
        VkMemoryAllocateInfo info={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};info.allocationSize=req.size;info.memoryTypeIndex=type;
        VkDeviceMemory result;check(vkAllocateMemory(device,&info,nullptr,&result));return result;
    };
    VkCommandPoolCreateInfo pi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=family;
    VkCommandPool pool;check(vkCreateCommandPool(device,&pi,nullptr,&pool));
    VkCommandBufferAllocateInfo ca={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;
    VkCommandBuffer cmd;check(vkAllocateCommandBuffers(device,&ca,&cmd));
    unsigned verified=0;
    for(auto format:{VK_FORMAT_D16_UNORM_S8_UINT,VK_FORMAT_D24_UNORM_S8_UINT,VK_FORMAT_D32_SFLOAT_S8_UINT})
    {
        VkImageFormatProperties properties;
        VkResult support=vkGetPhysicalDeviceImageFormatProperties(physical,format,VK_IMAGE_TYPE_2D,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,0,&properties);
        if(support==VK_ERROR_FORMAT_NOT_SUPPORTED){std::printf("format %d: unsupported by GPU (skipped)\n",format);continue;}check(support);
        for(auto viewAspect:{static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT),static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT)})
        {
            constexpr unsigned W=7,H=5; // odd widths detect accidental packed-depth strides
            const size_t bytes=d4r_vk_depth_bytes(format);assert(bytes==2 || bytes==4);
            VkImageCreateInfo ii={VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};ii.imageType=VK_IMAGE_TYPE_2D;ii.format=format;ii.extent={W,H,1};ii.mipLevels=ii.arrayLayers=1;ii.samples=VK_SAMPLE_COUNT_1_BIT;ii.tiling=VK_IMAGE_TILING_OPTIMAL;ii.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            VkImage image;check(vkCreateImage(device,&ii,nullptr,&image));VkMemoryRequirements req;vkGetImageMemoryRequirements(device,image,&req);auto im=allocate(req,0);check(vkBindImageMemory(device,image,im,0));
            VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=W*H*bytes;bi.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VkBuffer buffer;check(vkCreateBuffer(device,&bi,nullptr,&buffer));vkGetBufferMemoryRequirements(device,buffer,&req);auto bm=allocate(req,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);check(vkBindBufferMemory(device,buffer,bm,0));
            check(vkResetCommandPool(device,pool,0));VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(cmd,&begin));
            VkImageMemoryBarrier barrier={VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};barrier.image=image;barrier.subresourceRange={d4r_vk_barrier_aspect(format,viewAspect),0,1,0,1};barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;barrier.newLayout=VK_IMAGE_LAYOUT_GENERAL;barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
            VkClearDepthStencilValue clear={0.25f,0xab};vkCmdClearDepthStencilImage(cmd,image,VK_IMAGE_LAYOUT_GENERAL,&clear,1,&barrier.subresourceRange);
            barrier.oldLayout=VK_IMAGE_LAYOUT_GENERAL;barrier.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
            VkBufferImageCopy copy={};copy.imageSubresource={d4r_vk_copy_aspect(format,viewAspect),0,0,1};copy.imageExtent={W,H,1};assert(copy.imageSubresource.aspectMask==VK_IMAGE_ASPECT_DEPTH_BIT);
            vkCmdCopyImageToBuffer(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);
            barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;barrier.newLayout=VK_IMAGE_LAYOUT_GENERAL;barrier.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT;barrier.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&barrier);
            VkMemoryBarrier host={VK_STRUCTURE_TYPE_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host,0,nullptr,0,nullptr);
            check(vkEndCommandBuffer(cmd));VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cmd;check(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE));check(vkQueueWaitIdle(queue));
            void* mapped;check(vkMapMemory(device,bm,0,VK_WHOLE_SIZE,0,&mapped));
            for(unsigned n=0;n<W*H;++n)
            {
                float value=0;
                const auto* source=static_cast<const uint8_t*>(mapped)+n*bytes;
                if(format==VK_FORMAT_D32_SFLOAT_S8_UINT)std::memcpy(&value,source,4);
                else if(bytes==2){uint16_t v;std::memcpy(&v,source,2);value=static_cast<float>(v)/65535.f;}
                else{uint32_t v;std::memcpy(&v,source,4);value=static_cast<float>(v&0xffffffu)/16777215.f;}
                assert(std::isfinite(value) && std::abs(value-0.25f)<0.00002f);
            }
            std::printf("format %d viewAspect=0x%x: %zu-byte depth copy, 35 correct samples\n",format,viewAspect,bytes);++verified;
            vkUnmapMemory(device,bm);vkDestroyBuffer(device,buffer,nullptr);vkFreeMemory(device,bm,nullptr);vkDestroyImage(device,image,nullptr);vkFreeMemory(device,im,nullptr);
        }
    }
    assert(verified>=4); // at least two supported combined formats, each with both view masks
    vkDestroyCommandPool(device,pool,nullptr);vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);
}
