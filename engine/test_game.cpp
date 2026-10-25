// Native regression test for direct game textures, rectangle origins and queued descriptor rotation.
// Reuse the standalone replay runner's Vulkan context/readback helpers, without another device harness.
#define main d4r_replay_main
#include "replay.cpp"
#undef main
#include "game.h"
#include "game_m.h"
#include <cmath>
#ifdef D4R_TEST_BUILD_VKHIP
#include "vkhip/vkhip_net.h"
#endif

namespace {
bool stageTimingEnabled = false;
std::string vkhipModelDirectory;
unsigned stageTimingRows = 0;
void stageTimingLog(const char* line) { std::cout << line << '\n'; ++stageTimingRows; }
constexpr unsigned rotatingMasks[] = {15u, 14u, 13u, 11u, 15u, 8u};
struct TestImage {
    Context& context;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format;
    VkImageAspectFlags aspect;
    uint32_t w, h, channels, bits;
    Buffer upload;
    TestImage(Context& c, VkFormat f, uint32_t width, uint32_t height, uint32_t nc, uint32_t nb,
              VkImageAspectFlags a = VK_IMAGE_ASPECT_COLOR_BIT)
        : context(c), format(f), aspect(a), w(width), h(height), channels(nc), bits(nb),
          upload(c, VkDeviceSize(w) * h * nc * nb / 8) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D; info.format = f; info.extent = {w,h,1};
        info.mipLevels = info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (f == VK_FORMAT_R16G16B16A16_SFLOAT || f == VK_FORMAT_R32G32B32A32_SFLOAT ||
            f == VK_FORMAT_A2B10G10R10_UNORM_PACK32) info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        CK(vkCreateImage(c.device, &info, nullptr, &image));
        VkMemoryRequirements requirements; vkGetImageMemoryRequirements(c.device, image, &requirements);
        memory = c.allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CK(vkBindImageMemory(c.device, image, memory, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = f;
        vi.subresourceRange = {a,0,1,0,1}; CK(vkCreateImageView(c.device, &vi, nullptr, &view));
    }
    ~TestImage() {
        if (view) vkDestroyImageView(context.device, view, nullptr);
        if (image) vkDestroyImage(context.device, image, nullptr);
        if (memory) vkFreeMemory(context.device, memory, nullptr);
    }
    void put(uint32_t x, uint32_t y, uint32_t channel, float value) {
        if (format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
            if (channel >= 3) return;
            auto& word=static_cast<uint32_t*>(upload.map)[size_t(y)*w+x];
            const uint32_t shift=channel==0 ? 0 : channel==1 ? 11 : 22;
            const uint32_t width=channel==2 ? 10 : 11;
            uint16_t half;_Float16 v=static_cast<_Float16>(std::max(0.f,value));std::memcpy(&half,&v,2);
            const uint32_t mask=(1u<<width)-1;
            word=(word & ~(mask<<shift)) | ((uint32_t(half) >> (15-width) & mask)<<shift);
            return;
        }
        if (format == VK_FORMAT_D16_UNORM) {
            static_cast<uint16_t*>(upload.map)[size_t(y)*w+x]=uint16_t(std::lround(std::clamp(value,0.f,1.f)*65535.f));
            return;
        }
        if (format == VK_FORMAT_D24_UNORM_S8_UINT) {
            static_cast<uint32_t*>(upload.map)[size_t(y)*w+x]=uint32_t(std::lround(std::clamp(value,0.f,1.f)*16777215.f));
            return;
        }
        const size_t i = (size_t(y)*w+x)*channels+channel;
        if (bits == 32) static_cast<float*>(upload.map)[i] = value;
        else static_cast<_Float16*>(upload.map)[i] = static_cast<_Float16>(value);
    }
    void initialize(VkImageLayout layout) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.image = image;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.subresourceRange = {aspect,0,1,0,1};
        vkCmdPipelineBarrier(context.command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,0,nullptr,0,nullptr,1,&b);
        auto r = region(w,h); r.imageSubresource.aspectMask = aspect;
        vkCmdCopyBufferToImage(context.command, upload.handle, image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        if (layout != VK_IMAGE_LAYOUT_GENERAL) {
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL; b.newLayout = layout;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(context.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0,0,nullptr,0,nullptr,1,&b);
        }
    }
    d4r::GameImage game(uint32_t x, uint32_t y, bool direct, VkImageLayout layout) const {
        return {image,format,layout,aspect,x,y,direct ? view : VK_NULL_HANDLE};
    }
};
struct Case {
    uint32_t ow = 1283, oh = 723;
    bool depthAspect = false, wideColor = false, wideMotion = false, wideOutput = false, readonly = false;
    bool rgb10Output = false, rotateOutputFormats = false;
    bool regularDepth = false, swapDepth = false; // K: depth convention; near and far values exchanged
    float exposureScale = 1, preExposure = 1;     // K: the game's factors on the measured exposure
    bool dynamicM = false, largerCapacityM = false, explicitSizeResetM = false;
    bool displayMotion = false;                   // K: motion vectors at output resolution, in output pixels
    bool gameExposure = false; float exposureValue = 1; // K: the game's 1x1 exposure texture instead of measurement
    bool packedColor = false;
    bool packedFloatOutput = false;
    VkFormat packedDepth = VK_FORMAT_UNDEFINED;
};
void validateFrames(const Buffer& results, VkDeviceSize stride, const std::array<std::unique_ptr<TestImage>, 3>& output, unsigned count) {
    for (unsigned frame = 0; frame < count; ++frame) {
        const auto& image = *output[frame % output.size()];
        if (image.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) continue;
        if (image.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
            const auto* words=reinterpret_cast<const uint32_t*>(static_cast<const char*>(results.map)+frame*stride);
            bool nonzero=false;
            for(uint32_t y=7;y<image.h;++y) for(uint32_t x=5;x<image.w;++x) {
                uint32_t word=words[size_t(y)*image.w+x];nonzero|=word!=0;
                if (((word>>6)&31)==31 || ((word>>17)&31)==31 || ((word>>27)&31)==31)
                    throw std::runtime_error("nonfinite R11G11B10 output");
            }
            if(!nonzero)throw std::runtime_error("empty R11G11B10 output");
            continue;
        }
        const auto* data = static_cast<const char*>(results.map) + frame * stride;
        for (VkDeviceSize i = 0; i < image.upload.size / (image.bits / 8); ++i)
            if (!std::isfinite(image.bits == 32 ? reinterpret_cast<const float*>(data)[i] : float(reinterpret_cast<const _Float16*>(data)[i])))
                throw std::runtime_error("nonfinite adapter output");
    }
}
std::vector<char> evaluate(Context& c, d4r::Model& model, const Case& test, unsigned mask,
                           bool queued, bool bareEngine = false, bool rotateInputs = false, unsigned count = 6) {
    constexpr uint32_t rw = 640, rh = 360, slots = 3;
    std::array<std::unique_ptr<TestImage>, slots> color, motion, depth, output;
    TestImage exposureTexture(c, VK_FORMAT_R32_SFLOAT, 1, 1, 1, 32);
    exposureTexture.put(0,0,0,test.exposureValue);
    const auto inputLayout = test.readonly ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    for (unsigned slot = 0; slot < slots; ++slot) {
        color[slot] = std::make_unique<TestImage>(c, test.packedColor ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 :
            test.wideColor ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT,
            rw+16,rh+16,test.packedColor ? 1 : 4,test.packedColor || test.wideColor ? 32 : 16);
        const uint32_t mw = test.displayMotion ? test.ow+16 : rw+16, mh = test.displayMotion ? test.oh+16 : rh+16;
        motion[slot] = std::make_unique<TestImage>(c, test.wideMotion ? VK_FORMAT_R32G32_SFLOAT : VK_FORMAT_R16G16_SFLOAT, mw,mh,2,test.wideMotion ? 32 : 16);
        depth[slot] = std::make_unique<TestImage>(c, test.packedDepth != VK_FORMAT_UNDEFINED ? test.packedDepth :
            test.depthAspect ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R32_SFLOAT, rw+16,rh+16,1,
                                                test.packedDepth==VK_FORMAT_D16_UNORM ? 16 : 32,
                                                test.depthAspect ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
        const bool packed = test.packedFloatOutput || (test.rgb10Output && (!test.rotateOutputFormats || slot != 1));
        output[slot] = std::make_unique<TestImage>(c, test.packedFloatOutput ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 :
            packed ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 :
            test.wideOutput ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT,
            test.ow+16,test.oh+16,packed ? 1 : 4,packed || test.wideOutput ? 32 : 16);
        for (uint32_t y = 0; y < rh+16; ++y) for (uint32_t x = 0; x < rw+16; ++x) {
            const int cx = int(x)-7, cy = int(y)-9;
            const bool inside = cx >= 0 && cy >= 0 && cx < int(rw) && cy < int(rh);
            for (unsigned ch = 0; ch < 4; ++ch) color[slot]->put(x,y,ch, ch == 3 ? 1.f :
                inside ? .1f + ((cx + 3*cy + int(slot)*13 + int(ch)*7)%19)*.08f + (test.wideColor ? .00012345f : 0) : 20.f);
            depth[slot]->put(x,y,0, (((int(x)-11)/13)%2 != 0) != test.swapDepth ? .2f : .8f);
        }
        for (uint32_t y = 0; y < mh; ++y) for (uint32_t x = 0; x < mw; ++x) {
            const int mx = int(x)-3, my = int(y)-5;
            motion[slot]->put(x,y,0, .25f * ((mx+int(slot))%7-3) + (test.wideMotion ? .00012345f : 0));
            motion[slot]->put(x,y,1, .25f * ((my+int(slot))%9-4) + (test.wideMotion ? .00032109f : 0));
        }
        std::memset(output[slot]->upload.map,0,output[slot]->upload.size);
    }
    // The borrowed views above outlive both the adapter/engine and their immutable descriptor cache.
    d4r::ExternalNetwork external{};
#ifdef D4R_TEST_BUILD_VKHIP
    std::unique_ptr<vkhip::KNetwork> native;
    if (std::getenv("D4R_TEST_VKHIP")) {
        vkhip::NetFiles files;
        auto bytes=read(vkhipModelDirectory+"/hipnet.bin");files.hipnet.assign(bytes.begin(),bytes.end());
        for (const char* layer : {"enc0","enc1","enc2","enc3","enc4","dec5","dec4","dec3","dec2","dec1","dec0"}) {
            std::string name=std::string("dltss_pwin_")+layer+"_layer";
            bytes=read(vkhipModelDirectory+"/hip/"+name+".hsaco");files.code[name].assign(bytes.begin(),bytes.end());
        }
        native=std::make_unique<vkhip::KNetwork>();std::string error;
        if(!native->create(c.device,vkGetDeviceProcAddr,c.memory,files,test.ow,test.oh,
             d4r::Engine::networkTokenBytes(test.ow,test.oh),d4r::Engine::networkHeadBytes(test.ow,test.oh),error))
            throw std::runtime_error(error);
        external={native->tokens(),native->head()};
    }
#endif
    d4r::GameUpscaler adapter(model,c.physical,c.device,test.ow,test.oh,rw,rh,external.tokens ? &external : nullptr);
    auto shape = d4r::makeKShape(test.ow,test.oh);
    std::unique_ptr<d4r::Engine> engine;
    if (bareEngine) engine = std::make_unique<d4r::Engine>(model,shape,external.tokens ? &external : nullptr);
    const auto bytes = std::max({output[0]->upload.size, output[1]->upload.size, output[2]->upload.size});
    Buffer results(c,bytes*count);
    std::memset(results.map,0,results.size);
    c.begin();
    for (unsigned slot = 0; slot < slots; ++slot) {
        color[slot]->initialize(inputLayout); motion[slot]->initialize(inputLayout);
        depth[slot]->initialize(inputLayout); output[slot]->initialize(VK_IMAGE_LAYOUT_GENERAL);
    }
    exposureTexture.initialize(inputLayout);
    if (engine) engine->recordInitialize(c.command);
    c.submit();
    if (queued) c.begin();
    for (unsigned frame = 0; frame < count; ++frame) {
        unsigned slot = frame%slots;
        if (!queued) c.begin();
        const unsigned frameMask = rotateInputs ? rotatingMasks[frame % std::size(rotatingMasks)] : mask;
        d4r::GameFrame f;
        if (stageTimingEnabled) f.profile = {stageTimingLog, "test", frame + 1, true};
        f.color = color[slot]->game(7,9,frameMask&1,inputLayout);
        f.motion = motion[slot]->game(3,5,frameMask&2,inputLayout);
        f.depth = depth[slot]->game(11,13,frameMask&4,inputLayout);
        f.output = output[slot]->game(5,7,frameMask&8,VK_IMAGE_LAYOUT_GENERAL);
        if (test.gameExposure) f.exposure = exposureTexture.game(0,0,true,inputLayout);
        f.settings.renderWidth = rw-frame%2*8; f.settings.renderHeight = rh-frame%2*4;
        f.settings.reset = frame == 0 || frame == 4; f.settings.autoExposure = !test.gameExposure;
        f.settings.depthInverted = !test.regularDepth; f.settings.exposureScale = test.exposureScale; f.settings.preExposure = test.preExposure;
        f.settings.displayMotion = test.displayMotion;
        f.settings.jitter[0] = frame%2 ? .25f : -.125f; f.settings.jitter[1] = frame%3 ? .375f : -.25f;
        if (engine) {
            auto p = d4r::makeKFrameParams(shape,f.settings);
            const int origins[3][2] = {{7,9},{3,5},{11,13}};
            int32_t* in[] = {p.input.colOrigin,p.input.mvOrigin,p.input.depOrigin};
            int32_t* out[] = {p.output.colOrigin,p.output.mvOrigin,p.output.depOrigin};
            int32_t* hi[] = {p.output.colMax,p.output.mvMax,p.output.depMax};
            for (unsigned k=0;k<3;++k) for(unsigned a=0;a<2;++a) {in[k][a]=out[k][a]=origins[k][a];hi[k][a]+=origins[k][a];}
            p.output.outOrigin[0]=5;p.output.outOrigin[1]=7;
            d4r::ExposureControl exposure; exposure.origin[0]=7;exposure.origin[1]=9;
            exposure.size[0]=int32_t(f.settings.renderWidth);exposure.size[1]=int32_t(f.settings.renderHeight);
            exposure.measure=frame%2==0||f.settings.reset;exposure.reset=f.settings.reset;
            exposure.exposureScale=test.exposureScale;exposure.preExposure=test.preExposure;
            exposure.game=test.gameExposure;
            auto windows=d4r::makeKWindows(shape,frame);
            engine->setFrame({f.color.view,f.motion.view,f.depth.view,inputLayout,inputLayout,inputLayout,f.exposure.view,inputLayout},p.input,p.output,&windows,&exposure,{f.output.view,f.output.format});
#ifdef D4R_TEST_BUILD_VKHIP
            if (native) {engine->recordFront(c.command);native->record(c.command,f.settings.reset);engine->recordBack(c.command);}
            else
#endif
            engine->recordFrame(c.command);
        } else {
#ifdef D4R_TEST_BUILD_VKHIP
            if (native) {adapter.recordFront(c.command,f);native->record(c.command,f.settings.reset);adapter.recordBack(c.command,f);}
            else
#endif
            adapter.record(c.command,f);
        }
        auto r=region(output[slot]->w,output[slot]->h);r.bufferOffset=frame*bytes;
        vkCmdCopyImageToBuffer(c.command,output[slot]->image,VK_IMAGE_LAYOUT_GENERAL,results.handle,1,&r);
        if (!queued) {
            barrier(c.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
            c.submit();
        }
    }
    if (queued) {
        barrier(c.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        c.submit();
    }
    validateFrames(results,bytes,output,count);
    return {static_cast<char*>(results.map),static_cast<char*>(results.map)+results.size};
}
std::vector<char> evaluateM(Context& c, d4r::MModel& model, const Case& test, unsigned mask, bool queued, bool rotateInputs = false) {
    constexpr uint32_t rw = 640, rh = 360, ow = 960, oh = 540, count = 6, slots = 3;
    std::array<std::unique_ptr<TestImage>, slots> color, motion, depth, output;
    const auto layout = test.readonly ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    for (unsigned slot = 0; slot < slots; ++slot) {
        color[slot] = std::make_unique<TestImage>(c, VK_FORMAT_R16G16B16A16_SFLOAT, rw,rh,4,16);
        motion[slot] = std::make_unique<TestImage>(c, VK_FORMAT_R16G16_SFLOAT, rw,rh,2,16);
        depth[slot] = std::make_unique<TestImage>(c, test.depthAspect ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R32_SFLOAT, rw,rh,1,32,
            test.depthAspect ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
        const bool packed = test.rgb10Output && (!test.rotateOutputFormats || slot != 1);
        output[slot] = std::make_unique<TestImage>(c, packed ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 :
            test.wideOutput ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT,
            ow,oh,packed ? 1 : 4,packed || test.wideOutput ? 32 : 16);
        for (uint32_t y = 0; y < rh; ++y) for (uint32_t x = 0; x < rw; ++x) {
            for (unsigned ch = 0; ch < 4; ++ch)
                color[slot]->put(x,y,ch,ch == 3 ? 1.f : .1f + ((x+3*y+slot*13+ch*7)%19)*.08f);
            motion[slot]->put(x,y,0,.25f*(int((x+slot)%7)-3));
            motion[slot]->put(x,y,1,.25f*(int((y+slot)%9)-4));
            depth[slot]->put(x,y,0,(x/13)%2 ? .2f : .8f);
        }
        std::memset(output[slot]->upload.map,0,output[slot]->upload.size);
    }
    d4r::GameUpscalerM adapter(model,c.physical,c.device,ow,oh,test.largerCapacityM ? ow : rw,test.largerCapacityM ? oh : rh);
    const auto bytes = std::max({output[0]->upload.size, output[1]->upload.size, output[2]->upload.size});
    Buffer results(c,bytes*count);
    std::memset(results.map,0,results.size);
    c.begin();
    for (unsigned slot = 0; slot < slots; ++slot) {
        color[slot]->initialize(layout); motion[slot]->initialize(layout);
        depth[slot]->initialize(layout); output[slot]->initialize(VK_IMAGE_LAYOUT_GENERAL);
    }
    c.submit();
    if (queued) c.begin();
    for (unsigned frame = 0; frame < count; ++frame) {
        const unsigned slot = frame%slots;
        if (!queued) c.begin();
        d4r::GameFrameM f;
        if (stageTimingEnabled) f.profile = {stageTimingLog, "test", frame + 1, true};
        const unsigned frameMask = rotateInputs ? rotatingMasks[frame % std::size(rotatingMasks)] : mask;
        f.color=color[slot]->game(0,0,frameMask&1,layout); f.motion=motion[slot]->game(0,0,frameMask&2,layout);
        f.depth=depth[slot]->game(0,0,frameMask&4,layout); f.output=output[slot]->game(0,0,frameMask&8,VK_IMAGE_LAYOUT_GENERAL);
        if (test.dynamicM) {
            const uint32_t widths[count] = {640, 608, 448, 448, 640, 640};
            const uint32_t heights[count] = {360, 344, 256, 256, 360, 360};
            f.renderWidth = widths[frame]; f.renderHeight = heights[frame];
        }
        f.reset=frame==0||frame==4||(test.explicitSizeResetM && (frame==1||frame==2));
        f.jitter[0]=frame%2 ? .25f : -.125f; f.jitter[1]=frame%3 ? .375f : -.25f;
        adapter.record(c.command,f);
        auto r=region(ow,oh); r.bufferOffset=frame*bytes;
        vkCmdCopyImageToBuffer(c.command,output[slot]->image,VK_IMAGE_LAYOUT_GENERAL,results.handle,1,&r);
        if (!queued) {
            barrier(c.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
            c.submit();
        }
    }
    if (queued) {
        barrier(c.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        c.submit();
    }
    validateFrames(results,bytes,output,count);
    return {static_cast<char*>(results.map),static_cast<char*>(results.map)+results.size};
}
void expectEqual(const std::vector<char>& expected,const std::vector<char>& actual,const std::string& name) {
    if(expected!=actual) throw std::runtime_error(name+": output mismatch");
    std::cout<<name<<": byte-identical (full temporal sequence)\n";
}
} // namespace
// Normalized bilinear sampling can round differently when the allocation extent changes.
// FP8 quantization can amplify a rounding difference in a few recurrent pixels.
// Require less than 1% per-channel error and at least 86 dB PSNR at peak 1.
void expectCapacityEquivalent(const std::vector<char>& expected, const std::vector<char>& actual, const std::string& name) {
    if (expected.size() != actual.size()) throw std::runtime_error(name + ": size mismatch");
    double squared = 0; float relative = 0;
    for (size_t i = 0; i < expected.size()/2; ++i) {
        const float a = float(reinterpret_cast<const _Float16*>(expected.data())[i]);
        const float b = float(reinterpret_cast<const _Float16*>(actual.data())[i]);
        const float error = std::abs(a-b);
        if (!std::isfinite(error)) throw std::runtime_error(name + ": nonfinite output");
        squared += double(error)*error;
        relative = std::max(relative,error/std::max(1.f,std::abs(a)));
    }
    const double rmse = std::sqrt(squared/(expected.size()/2));
    if (relative > .01f || rmse > 5e-5)
        throw std::runtime_error(name + ": capacity changed the result (relative " + std::to_string(relative) + ", RMSE " + std::to_string(rmse) + ")");
    std::cout<<name<<": RMSE "<<rmse<<", max relative error "<<relative<<"\n";
}
void checkDynamicM(Context& c, d4r::MModel& m) {
    Case dynamic; dynamic.dynamicM = true;
    const auto reference = evaluateM(c,m,dynamic,0,false);
    expectEqual(reference,evaluateM(c,m,dynamic,15,true),"M/L dynamic sizes with queued direct inputs");
    dynamic.explicitSizeResetM = true;
    expectEqual(reference,evaluateM(c,m,dynamic,0,true),"M/L size changes reset history automatically");
    dynamic.explicitSizeResetM = false;
    dynamic.largerCapacityM = true;
    expectCapacityEquivalent(reference,evaluateM(c,m,dynamic,0,true),"M/L dynamic sizes within output-sized capacity");
    expectCapacityEquivalent(reference,evaluateM(c,m,dynamic,15,true),"M/L dynamic sizes with larger capacity and direct inputs");
}
int main(int argc,char** argv) {
    try {
        if(argc >= 2) vkhipModelDirectory=argv[1];
        // Six temporal frames per model: read five completed profiles without waiting in the profiler,
        // and compare the entire output sequence against the same adapter with instrumentation disabled.
        if (argc == 5 && !std::strcmp(argv[1], "--profile")) {
            Context c; c.create(2, true);
            d4r::Model k(c.physical, c.device, argv[2], c.cache);
            Case test;
            auto reference = evaluate(c, k, test, 15, false);
            stageTimingEnabled = true;
            expectEqual(reference, evaluate(c, k, test, 15, false), "K profiling preserves output");
            for (int i = 3; i <= 4; ++i) {
                d4r::MModel model(c.physical, c.device, argv[i], c.cache);
                stageTimingEnabled = false;
                reference = evaluateM(c, model, test, 15, false);
                stageTimingEnabled = true;
                expectEqual(reference, evaluateM(c, model, test, 15, false), "M/L profiling preserves output");
            }
            if (stageTimingRows != 15) throw std::runtime_error("expected five completed stage profiles per model");
            return 0;
        }
        if (argc == 3 && !std::strcmp(argv[1], "--dynamic-ml")) {
            Context c; c.create(2,true);
            d4r::MModel m(c.physical,c.device,argv[2],c.cache);
            c.begin(); m.recordUpload(c.command); c.submit();
            checkDynamicM(c,m);
            return 0;
        }
        if(argc!=2 && argc!=3) throw std::runtime_error("usage: test-game K_MODEL_DIR [M_MODEL_DIR] | --dynamic-ml M_OR_L_MODEL_DIR | --profile K_MODEL_DIR M_MODEL_DIR L_MODEL_DIR");
        // The reconstruction table NGX itself selected at these sizes (measured through its launch arguments),
        // on both sides of every threshold and for unequal axis ratios: {render w, h, output w, h, table}.
        {
            const uint32_t probes[][5] = {{1280,720,1280,720,4},{1238,696,1280,720,4},{1232,693,1280,720,6},{1030,580,1280,720,6},
                {1024,576,1280,720,8},{880,495,1280,720,8},{872,491,1280,720,10},{764,430,1280,720,10},{760,428,1280,720,12},
                {680,383,1280,720,12},{675,380,1280,720,14},{426,240,1280,720,14},{640,720,1280,720,8},{640,648,1280,720,10},
                {640,432,1280,720,12},{1706,960,2560,1440,10},{1280,720,3840,2160,14}};
            for (const auto& q : probes)
                if (d4r::kReconstructionTable(q[2],q[3],q[0],q[1]) != q[4])
                    throw std::runtime_error("reconstruction table selection differs from NGX at render "+std::to_string(q[0])+"x"+std::to_string(q[1]));
            std::cout<<"reconstruction table selection matches NGX at "<<std::size(probes)<<" measured sizes\n";
        }
        Context c;c.create(2,argc==3);d4r::Model model(c.physical,c.device,argv[1],c.cache);
        if (!model.hasDirectOrigins()) throw std::runtime_error("recompile the model with compile_k.py before testing direct rectangles");
        c.begin();model.recordUpload(c.command);c.submit();
        if (std::getenv("D4R_TEST_VKHIP_FORMATS_ONLY")) {
            for(bool regular : {false,true}) for(VkFormat depth :
                {VK_FORMAT_UNDEFINED,VK_FORMAT_D16_UNORM,VK_FORMAT_D24_UNORM_S8_UINT}) {
                Case packed;packed.packedColor=packed.packedFloatOutput=true;
                packed.regularDepth=regular;packed.packedDepth=depth;packed.depthAspect=depth!=VK_FORMAT_UNDEFINED;
                packed.wideMotion=true;packed.readonly=true;
                auto reference=evaluate(c,model,packed,15,false,false,false,16);
                expectEqual(reference,evaluate(c,model,packed,15,true,false,false,16),
                            "R11 colour/output, RG32 motion, packed/regular depth, rotating descriptors, 16 frames");
            }
            std::cout<<"All vkhip packed-format checks passed\n";
            return 0;
        }
        for(bool depthAspect:{false,true}) {
            Case test;test.depthAspect=depthAspect;
            auto reference=evaluate(c,model,test,0,false);
            for(unsigned mask:{1u,2u,4u,8u,7u,15u}) expectEqual(reference,evaluate(c,model,test,mask,true),"direct mask "+std::to_string(mask)+(depthAspect?" D32":" R32"));
            for (bool queued : {false, true})
                expectEqual(reference,evaluate(c,model,test,15,queued,false,true),"late fallback allocation "+std::string(queued ? "queued" : "submitted")+(depthAspect ? " D32" : " R32"));
            if (!model.hasRgb10Output()) throw std::runtime_error("recompile the K model for RGB10A2 checks");
            for (bool mixed : {false, true}) {
                test.rgb10Output=true;test.rotateOutputFormats=mixed;
                const auto packedReference=evaluate(c,model,test,0,false);
                for (bool queued : {false,true})
                    expectEqual(packedReference,evaluate(c,model,test,15,queued,false,true),
                        std::string(mixed ? "mixed RGBA16F/RGB10A2" : "RGB10A2")+ " late fallback "+(queued ? "queued" : "submitted")+(depthAspect ? " D32" : " R32"));
            }
            test.rgb10Output=false;test.rotateOutputFormats=false;
            test.readonly=true;
            expectEqual(reference,evaluate(c,model,test,15,true),"read-only input layouts");
            test.readonly=false;test.wideOutput=true;
            expectEqual(evaluate(c,model,test,0,false),evaluate(c,model,test,15,true),"RGBA32F output conversion fallback");
        }
        // Regular depth picks the nearest neighbour as the smallest depth. With near and far exchanged it must
        // choose exactly the texels inverted depth chose; inverted depth on the exchanged values must not.
        if (!model.hasRegularDepth()) throw std::runtime_error("recompile the K model for regular-depth checks");
        {
            Case test;
            const auto reference=evaluate(c,model,test,0,false);
            test.swapDepth=true;
            if (evaluate(c,model,test,0,false)==reference) throw std::runtime_error("depth does not affect the output; the regular-depth check is vacuous");
            test.regularDepth=true;
            for (unsigned mask:{0u,15u}) expectEqual(reference,evaluate(c,model,test,mask,mask!=0),"regular depth mirrors inverted depth, direct mask "+std::to_string(mask));
        }
        // The game's exposure scale multiplies the measured exposure: the adapter must equal an Engine given
        // the scaled key, and differ from the neutral result.
        {
            Case test;test.wideColor=true;test.wideMotion=true;test.readonly=true;
            const auto neutral=evaluate(c,model,test,15,true);
            test.exposureScale=.7f;test.preExposure=3.f;
            const auto scaled=evaluate(c,model,test,15,true);
            if (scaled==neutral) throw std::runtime_error("the exposure scale and pre-exposure do not affect the output");
            expectEqual(evaluate(c,model,test,15,false,true),scaled,"exposure scale 0.7 and pre-exposure 3 match the Engine");
        }
        // A game exposure texture replaces the measurement (AutoExposure off): the adapter must equal an Engine given
        // the same value, and the value must reach the output. A non-positive texel is read as 1.
        if (!model.hasGameExposure()) throw std::runtime_error("recompile the K model for game exposure checks");
        {
            Case test;test.wideColor=true;test.wideMotion=true;test.readonly=true;test.gameExposure=true;test.exposureValue=.6f;
            const auto game=evaluate(c,model,test,15,true);
            expectEqual(evaluate(c,model,test,15,false,true),game,"game exposure 0.6 matches the Engine");
            test.exposureValue=1.f;
            const auto unit=evaluate(c,model,test,15,true);
            if (game==unit) throw std::runtime_error("the game exposure does not affect the output");
            test.exposureValue=0.f;
            expectEqual(evaluate(c,model,test,15,true),unit,"non-positive game exposure reads as 1");
        }
        Case precise;precise.wideColor=true;precise.wideMotion=true;precise.readonly=true;
        expectEqual(evaluate(c,model,precise,15,false,true),evaluate(c,model,precise,15,true),"RGBA32F color/RG32F motion match unquantized Engine");
        // Display-resolution motion (flag 4): output-sized vectors in output pixels. The render-resolution
        // result must differ, or the check is vacuous. The direct/copy comparisons use the same input formats on
        // both sides (a wide source may only be borrowed, not copied: the fallback copy quantizes to RGBA16F/RG16F).
        if (!model.hasDisplayMotion()) throw std::runtime_error("recompile the K model with compile_k.py for display-resolution motion checks");
        {
            Case display;display.displayMotion=true;
            const auto reference=evaluate(c,model,display,0,false);
            Case render;
            if (evaluate(c,model,render,0,false)==reference) throw std::runtime_error("display-resolution motion does not affect the output; the check is vacuous");
            for (unsigned mask:{2u,15u}) expectEqual(reference,evaluate(c,model,display,mask,true),"display-resolution motion direct mask "+std::to_string(mask));
            expectEqual(reference,evaluate(c,model,display,15,false,true),"display-resolution motion matches the Engine");
            Case wide;wide.displayMotion=true;wide.wideColor=true;wide.wideMotion=true;wide.readonly=true;
            expectEqual(evaluate(c,model,wide,15,false,true),evaluate(c,model,wide,15,true),"display-resolution motion RGBA32F/RG32F match unquantized Engine");
        }
        for (bool regular : {false,true}) {
            Case packed;packed.regularDepth=regular;packed.packedColor=true;
            expectEqual(evaluate(c,model,packed,0,false),evaluate(c,model,packed,15,true),
                        "R11G11B10 colour, direct and converted, regular/inverted depth");
            packed.packedFloatOutput=true;
            expectEqual(evaluate(c,model,packed,15,false),evaluate(c,model,packed,15,true),
                        "R11G11B10 colour/output, regular/inverted depth");
            for(VkFormat depth : {VK_FORMAT_D16_UNORM,VK_FORMAT_D24_UNORM_S8_UINT}) {
                packed.depthAspect=true;packed.packedDepth=depth;
                expectEqual(evaluate(c,model,packed,15,false),evaluate(c,model,packed,15,true),
                            "packed D16/D24 depth, queued descriptors, regular/inverted depth");
            }
        }
        if (argc==3) {
            d4r::MModel m(c.physical,c.device,argv[2],c.cache);
            c.begin();m.recordUpload(c.command);c.submit();
            checkDynamicM(c, m);
            for (bool depthAspect:{false,true}) {
                Case test;test.depthAspect=depthAspect;
                auto reference=evaluateM(c,m,test,0,false);
                for (unsigned mask:{1u,2u,4u,8u,7u,15u})
                    expectEqual(reference,evaluateM(c,m,test,mask,true),"M Quality direct mask "+std::to_string(mask)+(depthAspect?" D32":" R32"));
                for (bool queued : {false, true})
                    expectEqual(reference,evaluateM(c,m,test,15,queued,true),"M late fallback allocation "+std::string(queued ? "queued" : "submitted")+(depthAspect ? " D32" : " R32"));
                if (!m.hasRgb10Output()) throw std::runtime_error("recompile the M model for RGB10A2 checks");
                for (bool mixed : {false,true}) {
                    test.rgb10Output=true;test.rotateOutputFormats=mixed;
                    const auto packedReference=evaluateM(c,m,test,0,false);
                    for (bool queued : {false,true})
                        expectEqual(packedReference,evaluateM(c,m,test,15,queued,true),
                            std::string(mixed ? "M mixed RGBA16F/RGB10A2" : "M RGB10A2")+" late fallback "+(queued ? "queued" : "submitted")+(depthAspect ? " D32" : " R32"));
                }
                test.rgb10Output=false;test.rotateOutputFormats=false;
                test.readonly=true;
                expectEqual(reference,evaluateM(c,m,test,15,true),"M read-only input layouts");
                test.readonly=false;test.wideOutput=true;
                expectEqual(evaluateM(c,m,test,0,false),evaluateM(c,m,test,15,true),"M RGBA32F output conversion fallback");
            }
        }
        std::cout<<"All direct-texture checks passed\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"test-game: "<<e.what()<<'\n';return 1;}
}
