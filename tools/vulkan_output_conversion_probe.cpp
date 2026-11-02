// Compare fused storage output with the original RGBA16F copy + R11G11B10 blit.
#include "d4r_vulkan_staging.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void check(VkResult r) { if (r != VK_SUCCESS) { std::fprintf(stderr,"Vulkan error %d\n",r); std::exit(1); } }
struct Image { VkImage image; VkImageView view; VkDeviceMemory memory; VkFormat format; bool depth; };
int main() {
    constexpr uint32_t W=257, H=257;
    constexpr size_t BYTES=W*H*16;
    VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ic={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ic.pApplicationInfo=&app;
    VkInstance instance;check(vkCreateInstance(&ic,nullptr,&instance));
    uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr));
    std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()));
    VkPhysicalDevice physical=VK_NULL_HANDLE;
    for(auto d:devices){VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(d,&p);if(p.vendorID==0x1002){physical=d;std::printf("GPU: %s\n",p.deviceName);break;}}
    assert(physical);
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());
    uint32_t family=0;while(family<count && !(families[family].queueFlags&VK_QUEUE_COMPUTE_BIT))++family;assert(family<count);
    float priority=1;VkDeviceQueueCreateInfo qi={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
    VkPhysicalDeviceFeatures supported, enabled={};vkGetPhysicalDeviceFeatures(physical,&supported);
    enabled.shaderStorageImageExtendedFormats=supported.shaderStorageImageExtendedFormats;
    VkDeviceCreateInfo dc={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dc.queueCreateInfoCount=1;dc.pQueueCreateInfos=&qi;dc.pEnabledFeatures=&enabled;
    VkDevice device;check(vkCreateDevice(physical,&dc,nullptr,&device));VkQueue queue;vkGetDeviceQueue(device,family,0,&queue);
    VkPhysicalDeviceMemoryProperties memory;vkGetPhysicalDeviceMemoryProperties(physical,&memory);
    auto allocate=[&](VkMemoryRequirements req,VkMemoryPropertyFlags flags){
        uint32_t type=0;while(type<memory.memoryTypeCount && (!(req.memoryTypeBits&(1u<<type)) || (memory.memoryTypes[type].propertyFlags&flags)!=flags))++type;assert(type<memory.memoryTypeCount);
        VkMemoryAllocateInfo a={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};a.allocationSize=req.size;a.memoryTypeIndex=type;VkDeviceMemory m;check(vkAllocateMemory(device,&a,nullptr,&m));return m;
    };
    auto image=[&](VkFormat format,bool output){
        Image result={};result.format=format;result.depth=format==VK_FORMAT_D32_SFLOAT_S8_UINT;
        VkImageCreateInfo info={VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};info.imageType=VK_IMAGE_TYPE_2D;info.format=format;info.extent={W,H,1};info.mipLevels=info.arrayLayers=1;info.samples=VK_SAMPLE_COUNT_1_BIT;info.tiling=VK_IMAGE_TILING_OPTIMAL;
        // Transfers build the reference and read back packed R11G11B10 words.
        info.usage=VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (output ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
        check(vkCreateImage(device,&info,nullptr,&result.image));VkMemoryRequirements req;vkGetImageMemoryRequirements(device,result.image,&req);
        result.memory=allocate(req,0);check(vkBindImageMemory(device,result.image,result.memory,0));
        VkImageViewCreateInfo view={VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};view.image=result.image;view.viewType=VK_IMAGE_VIEW_TYPE_2D;view.format=format;
        view.subresourceRange={result.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        check(vkCreateImageView(device,&view,nullptr,&result.view));return result;
    };
    auto destroyImage=[&](Image& i){vkDestroyImageView(device,i.view,nullptr);vkDestroyImage(device,i.image,nullptr);vkFreeMemory(device,i.memory,nullptr);};
    VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=BYTES;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buffers[2];VkDeviceMemory allocations[2];float* mapped[2];
    for(int i=0;i<2;++i){check(vkCreateBuffer(device,&bi,nullptr,&buffers[i]));VkMemoryRequirements req;vkGetBufferMemoryRequirements(device,buffers[i],&req);allocations[i]=allocate(req,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);check(vkBindBufferMemory(device,buffers[i],allocations[i],0));check(vkMapMemory(device,allocations[i],0,VK_WHOLE_SIZE,0,reinterpret_cast<void**>(&mapped[i])));}
    D4rVkStaging staging;assert(staging.init(device,vkGetDeviceProcAddr));
    VkCommandPoolCreateInfo pi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=family;VkCommandPool pool;check(vkCreateCommandPool(device,&pi,nullptr,&pool));
    VkCommandBufferAllocateInfo ca={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;VkCommandBuffer cmd;check(vkAllocateCommandBuffers(device,&ca,&cmd));
    auto begin=[&](){check(vkResetCommandPool(device,pool,0));VkCommandBufferBeginInfo b={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};b.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;check(vkBeginCommandBuffer(cmd,&b));};
    auto submit=[&](){VkMemoryBarrier host={VK_STRUCTURE_TYPE_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host,0,nullptr,0,nullptr);check(vkEndCommandBuffer(cmd));VkSubmitInfo s={VK_STRUCTURE_TYPE_SUBMIT_INFO};s.commandBufferCount=1;s.pCommandBuffers=&cmd;check(vkQueueSubmit(queue,1,&s,VK_NULL_HANDLE));check(vkQueueWaitIdle(queue));};
    auto barrier=[&](const Image& i,VkImageLayout from,VkImageLayout to,VkAccessFlags src,VkAccessFlags dst){
        VkImageMemoryBarrier b={VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};b.image=i.image;b.oldLayout=from;b.newLayout=to;b.srcAccessMask=src;b.dstAccessMask=dst;b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange={static_cast<VkImageAspectFlags>(i.depth ? VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT),0,1,0,1};vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
    };
    auto source=image(VK_FORMAT_R16G16B16A16_SFLOAT,false);
    auto reference=image(VK_FORMAT_B10G11R11_UFLOAT_PACK32,true);
    auto candidate=image(VK_FORMAT_B10G11R11_UFLOAT_PACK32,true);
    D4rVkStagingSet write;assert(staging.set(write,candidate.view,candidate.format,buffers[0],W*H*8,true));
    auto* halves=reinterpret_cast<uint16_t*>(mapped[0]);
    for(unsigned i=0;i<W*H;++i){halves[i*4]=i;halves[i*4+1]=i*73+31;halves[i*4+2]=i*193+57;halves[i*4+3]=0x3c00;}
    std::memset(mapped[1],0xa5,BYTES);
    begin();barrier(source,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region={};region.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};region.imageExtent={W,H,1};
    vkCmdCopyBufferToImage(cmd,buffers[0],source.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);
    barrier(source,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
    barrier(reference,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageBlit blit={};blit.srcSubresource=blit.dstSubresource=region.imageSubresource;
    blit.srcOffsets[1]=blit.dstOffsets[1]={(int)W,(int)H,1};
    vkCmdBlitImage(cmd,source.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,reference.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&blit,VK_FILTER_NEAREST);
    barrier(reference,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cmd,reference.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffers[1],1,&region);
    barrier(candidate,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,0,VK_ACCESS_SHADER_WRITE_BIT);
    staging.record(cmd,write,W,H);
    barrier(candidate,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
    region.bufferOffset=W*H*4;
    vkCmdCopyImageToBuffer(cmd,candidate.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffers[1],1,&region);submit();
    auto* words=reinterpret_cast<uint32_t*>(mapped[1]);unsigned mismatches=0,finite=0,nan=0;
    for(unsigned i=0;i<W*H;++i)for(unsigned c=0;c<3;++c) {
        const auto h=halves[i*4+c];const unsigned shift=c*11,bits=c==2 ? 10 : 11,mask=(1u<<bits)-1;
        unsigned old=(words[i]>>shift)&mask,now=(words[W*H+i]>>shift)&mask;
        if((h&0x7c00)==0x7c00 && (h&0x3ff)) {
            ++nan;const unsigned m=c==2 ? 5 : 6;
            assert((old>>m)==31 && (now>>m)==31 && (old&((1u<<m)-1)) && (now&((1u<<m)-1)));
        } else {++finite;if(old!=now){if(mismatches<8)std::printf("DIFF pixel=%u channel=%u half=%04x blit=%x compute=%x\n",i,c,h,old,now);++mismatches;}}
    }
    std::printf("R11 output blit vs compute: finite_or_inf_channels=%u mismatches=%u nan_channels=%u\n",finite,mismatches,nan);
    assert(mismatches==0);
    staging.free(write);destroyImage(source);destroyImage(reference);destroyImage(candidate);
    // Input conversion: every unsigned R11/R10 channel encoding, padded rows.
    auto input=image(VK_FORMAT_B10G11R11_UFLOAT_PACK32,false);
    auto input_reference=image(VK_FORMAT_R16G16B16A16_SFLOAT,true);
    D4rVkStagingSet read;assert(staging.set(read,input.view,input.format,buffers[0],BYTES,false));
    auto* raw=reinterpret_cast<uint32_t*>(mapped[0]);
    for(unsigned i=0;i<BYTES/4;++i)raw[i]=0xa5a5a5a5;
    for(unsigned i=0;i<W*H;++i)raw[i]=(i&2047u)|((i*73u+31u)&2047u)<<11|((i*193u+57u)&1023u)<<22;
    std::vector<uint32_t> initial(raw,raw+BYTES/4);
    begin();barrier(input,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT);
    region.bufferOffset=0;
    vkCmdCopyBufferToImage(cmd,buffers[0],input.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);
    barrier(input,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
    barrier(input_reference,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdBlitImage(cmd,input.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,input_reference.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&blit,VK_FILTER_NEAREST);
    barrier(input_reference,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
    vkCmdCopyImageToBuffer(cmd,input_reference.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffers[1],1,&region);
    barrier(input,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT);
    VkMemoryBarrier reuse={VK_STRUCTURE_TYPE_MEMORY_BARRIER};reuse.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT;reuse.dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&reuse,0,nullptr,0,nullptr);
    constexpr unsigned stride=W+7;
    staging.record(cmd,read,W,H,1,stride);submit();
    const auto* actual=reinterpret_cast<const uint16_t*>(mapped[0]);
    const auto* expected=reinterpret_cast<const uint16_t*>(mapped[1]);
    unsigned input_bad=0,input_nan=0;
    for(unsigned y=0;y<H;++y) {
        for(unsigned x=0;x<W;++x)for(unsigned c=0;c<4;++c) {
            const auto got=actual[(size_t(y)*stride+x)*4+c],want=expected[(size_t(y)*W+x)*4+c];
            if((want&0x7c00)==0x7c00 && (want&0x3ff)) {
                ++input_nan;assert((got&0x7c00)==0x7c00 && (got&0x3ff));
            } else if(got!=want){if(input_bad<8)std::printf("INPUT DIFF x=%u y=%u channel=%u blit=%04x compute=%04x\n",x,y,c,want,got);++input_bad;}
        }
        for(unsigned word=(y*stride+W)*2;word<(y+1)*stride*2;++word)assert(raw[word]==initial[word]);
    }
    for(unsigned word=H*stride*2;word<BYTES/4;++word)assert(raw[word]==initial[word]);
    std::printf("R11 input blit vs compute: mismatches=%u nan_channels=%u padded_rows_unchanged=1\n",input_bad,input_nan);
    assert(input_bad==0);
    staging.free(read);destroyImage(input);destroyImage(input_reference);
    staging.destroy();vkDestroyCommandPool(device,pool,nullptr);
    for(int i=0;i<2;++i){vkUnmapMemory(device,allocations[i]);vkDestroyBuffer(device,buffers[i],nullptr);vkFreeMemory(device,allocations[i],nullptr);}
    vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);std::puts("PASS: exhaustive half inputs, all RGB channels, partial workgroups");
}
