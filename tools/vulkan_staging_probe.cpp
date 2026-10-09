// Real GPU regression for sampled-only inputs and storage-only output writes.
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
    constexpr uint32_t W=7, H=5;
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
        // Input initialization uses only TRANSFER_DST. Outputs have neither
        // TRANSFER bit: all writes and verification reads use the new shaders.
        info.usage=VK_IMAGE_USAGE_SAMPLED_BIT | (output ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        check(vkCreateImage(device,&info,nullptr,&result.image));VkMemoryRequirements req;vkGetImageMemoryRequirements(device,result.image,&req);
        result.memory=allocate(req,0);check(vkBindImageMemory(device,result.image,result.memory,0));
        VkImageViewCreateInfo view={VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};view.image=result.image;view.viewType=VK_IMAGE_VIEW_TYPE_2D;view.format=format;
        view.subresourceRange={result.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        check(vkCreateImageView(device,&view,nullptr,&result.view));return result;
    };
    auto destroyImage=[&](Image& i){vkDestroyImageView(device,i.view,nullptr);vkDestroyImage(device,i.image,nullptr);vkFreeMemory(device,i.memory,nullptr);};
    VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=BYTES;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer buffers[2];VkDeviceMemory allocations[2];float* mapped[2];
    for(int i=0;i<2;++i){check(vkCreateBuffer(device,&bi,nullptr,&buffers[i]));VkMemoryRequirements req;vkGetBufferMemoryRequirements(device,buffers[i],&req);allocations[i]=allocate(req,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);check(vkBindBufferMemory(device,buffers[i],allocations[i],0));check(vkMapMemory(device,allocations[i],0,VK_WHOLE_SIZE,0,reinterpret_cast<void**>(&mapped[i])));}
    D4rVkStaging staging;assert(staging.init(device,vkGetDeviceProcAddr));
    VkCommandPoolCreateInfo pi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=family;VkCommandPool pool;check(vkCreateCommandPool(device,&pi,nullptr,&pool));
    VkCommandBufferAllocateInfo ca={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;VkCommandBuffer cmd;check(vkAllocateCommandBuffers(device,&ca,&cmd));
    auto begin=[&](){check(vkResetCommandPool(device,pool,0));VkCommandBufferBeginInfo b={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};b.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;check(vkBeginCommandBuffer(cmd,&b));};
    auto submit=[&](){VkMemoryBarrier host={VK_STRUCTURE_TYPE_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host,0,nullptr,0,nullptr);check(vkEndCommandBuffer(cmd));VkSubmitInfo s={VK_STRUCTURE_TYPE_SUBMIT_INFO};s.commandBufferCount=1;s.pCommandBuffers=&cmd;check(vkQueueSubmit(queue,1,&s,VK_NULL_HANDLE));check(vkQueueWaitIdle(queue));};
    auto barrier=[&](const Image& i,VkImageLayout from,VkImageLayout to,VkAccessFlags src,VkAccessFlags dst){
        VkImageMemoryBarrier b={VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};b.image=i.image;b.oldLayout=from;b.newLayout=to;b.srcAccessMask=src;b.dstAccessMask=dst;b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange={i.depth ? VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);
    };
    unsigned tests=0;
    for(auto format:{VK_FORMAT_R16G16B16A16_SFLOAT,VK_FORMAT_B10G11R11_UFLOAT_PACK32,VK_FORMAT_R16G16_SFLOAT,VK_FORMAT_R16_SFLOAT,VK_FORMAT_R32_SFLOAT,VK_FORMAT_D32_SFLOAT_S8_UINT}) {
        auto source=image(format,false);D4rVkStagingSet read;assert(staging.set(read,source.view,format,buffers[0],BYTES,false));
        std::memset(mapped[0],0xa5,BYTES);begin();barrier(source,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageSubresourceRange range={source.depth ? VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        if(source.depth){VkClearDepthStencilValue c={0.25f,0xab};vkCmdClearDepthStencilImage(cmd,source.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&c,1,&range);}
        else{VkClearColorValue c={{0.25f,0.5f,0.75f,1.f}};vkCmdClearColorImage(cmd,source.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&c,1,&range);}
        barrier(source,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
        staging.record(cmd,read,W,H);submit();
        const bool two=format==VK_FORMAT_R16G16_SFLOAT, scalar=source.depth||format==VK_FORMAT_R32_SFLOAT||format==VK_FORMAT_R16_SFLOAT;
        for(unsigned p=0;p<W*H;++p){assert(mapped[0][p*4]==0.25f);assert(mapped[0][p*4+1]==(scalar?0.f:0.5f));assert(mapped[0][p*4+2]==(scalar||two?0.f:0.75f));assert(mapped[0][p*4+3]==1.f);}
        for(unsigned mode=1;mode<=4;++mode) {
            std::memset(mapped[0],0xa5,BYTES);begin();staging.record(cmd,read,W,H,mode);submit();
            auto* words=reinterpret_cast<uint32_t*>(mapped[0]);
            const unsigned stride=mode==1?2:1;
            for(unsigned p=0;p<W*H;++p) {
                const uint32_t expected=mode==1||mode==3 ? 0x3400u | ((scalar?0u:0x3800u)<<16) : 0x3e800000u;
                assert(words[p*stride]==expected);
                if(mode==1)assert(words[p*stride+1]==(scalar||two?0u:0x3a00u) + (0x3c00u<<16));
            }
            for(unsigned j=W*H*stride;j<BYTES/4;++j)assert(words[j]==0xa5a5a5a5u);
        }
        std::puts("canonical RGBA16F/R32F/RG16F input modes: values and bounds correct");
        std::printf("sample input format %d: 35 correct pixels; no TRANSFER_SRC\n",format);++tests;
        check(vkResetCommandPool(device,pool,0));staging.free(read);destroyImage(source);
    }
    for(auto format:{VK_FORMAT_R16G16B16A16_SFLOAT,VK_FORMAT_R32G32B32A32_SFLOAT,VK_FORMAT_B10G11R11_UFLOAT_PACK32,VK_FORMAT_A2B10G10R10_UNORM_PACK32,VK_FORMAT_R8G8B8A8_UNORM}) {
        auto output=image(format,true);D4rVkStagingSet write,verify;assert(staging.set(write,output.view,format,buffers[0],W*H*8,true));assert(staging.set(verify,output.view,format,buffers[1],BYTES,false,VK_IMAGE_LAYOUT_GENERAL));
        auto* packed=reinterpret_cast<uint16_t*>(mapped[0]);
        const uint16_t half[]={0x3400,0x3800,0x3a00,0x3c00};
        const float expected[]={0.25f,0.5f,0.75f,1.f};
        for(unsigned p=0;p<W*H;++p)for(unsigned c=0;c<4;++c)packed[p*4+c]=half[c];
        std::memset(mapped[1],0xa5,BYTES);begin();barrier(output,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,0,VK_ACCESS_SHADER_WRITE_BIT);
        staging.record(cmd,write,W,H);barrier(output,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);staging.record(cmd,verify,W,H);submit();
        const float epsilon=format==VK_FORMAT_R8G8B8A8_UNORM ? 1.f/255 : format==VK_FORMAT_A2B10G10R10_UNORM_PACK32 ? 1.f/1023 : 0.00001f;
        for(unsigned p=0;p<W*H;++p)for(unsigned c=0;c<4;++c)assert(std::isfinite(mapped[1][p*4+c]) && std::abs(mapped[1][p*4+c]-expected[c])<=epsilon);
        std::printf("storage output format %d: 35 correct pixels; no TRANSFER_DST\n",format);++tests;
        if(format==VK_FORMAT_R32G32B32A32_SFLOAT) {
            unsigned verified=0;
            for(unsigned base=0;base<65536;base+=W*H*4) {
                for(unsigned j=0;j<W*H*4;++j)packed[j]=static_cast<uint16_t>(base+j);
                begin();barrier(output,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_SHADER_WRITE_BIT);
                staging.record(cmd,write,W,H);barrier(output,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
                staging.record(cmd,verify,W,H);submit();
                for(unsigned j=0;j<W*H*4 && base+j<65536;++j) {
                    const uint16_t h=packed[j];const unsigned e=(h>>10)&31, m=h&1023;
                    if(e==31)continue;
                    float value=e ? std::ldexp(1.f+static_cast<float>(m)/1024.f,static_cast<int>(e)-15) : std::ldexp(static_cast<float>(m),-24);
                    if(h&0x8000)value=-value;
                    uint32_t wanted,actual;std::memcpy(&wanted,&value,4);std::memcpy(&actual,&mapped[1][j],4);
                    assert(actual==wanted);++verified;
                }
                // Repack the RGBA32F sampled result using the new canonical
                // input path. Every finite half must round-trip bit exactly.
                begin();staging.record(cmd,verify,W,H,1);submit();
                const auto* encoded=reinterpret_cast<const uint16_t*>(mapped[1]);
                for(unsigned j=0;j<W*H*4 && base+j<65536;++j)
                    if(((packed[j]>>10)&31)!=31)assert(encoded[j]==packed[j]);
            }
            assert(verified==63488);std::printf("PASS: all %u finite half encodings and input round trips, including signed zeros and subnormals\n",verified);
        }
        check(vkResetCommandPool(device,pool,0));staging.free(write);staging.free(verify);destroyImage(output);
    }
    staging.destroy();vkDestroyCommandPool(device,pool,nullptr);
    for(int i=0;i<2;++i){vkUnmapMemory(device,allocations[i]);vkDestroyBuffer(device,buffers[i],nullptr);vkFreeMemory(device,allocations[i],nullptr);}
    vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);std::printf("PASS: %u GPU format checks\n",tests);
}
