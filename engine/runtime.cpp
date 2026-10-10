#include "runtime.h"
#include "vk_dispatch.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace d4r {
static_assert(sizeof(InputParams) == 128);
static_assert(sizeof(OutputParams) == 188);
static_assert(offsetof(OutputParams, expo) == 148);
static_assert(sizeof(LayerShape) == 16);
static_assert(sizeof(Shape) == 184);
namespace {
void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(result));
}
#define VKCHECK(call) check((call), #call)
std::vector<char> read(const std::string& file) {
    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + file);
    auto size = stream.tellg();
    if (size <= 0 || size > 256 * 1024 * 1024) throw std::runtime_error("invalid size: " + file);
    std::vector<char> data(static_cast<size_t>(size));
    stream.seekg(0);
    if (!stream.read(data.data(), size)) throw std::runtime_error("cannot read " + file);
    return data;
}
struct Device {
    VkDevice handle;
    VkPhysicalDeviceMemoryProperties memory;
    VkPhysicalDeviceProperties properties;
    Device(VkPhysicalDevice physical, VkDevice device) : handle(device) {
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        vkGetPhysicalDeviceProperties(physical, &properties);
    }
    VkDeviceMemory allocate(VkMemoryRequirements requirements, VkMemoryPropertyFlags flags) const {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) {
                VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                info.allocationSize = requirements.size; info.memoryTypeIndex = i;
                VkDeviceMemory result;
                VKCHECK(vkAllocateMemory(handle, &info, nullptr, &result));
                return result;
            }
        }
        throw std::runtime_error("no compatible Vulkan memory type");
    }
};
struct Buffer {
    const Device* device = nullptr;
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (!device) return;
        if (mapped) vkUnmapMemory(device->handle, memory);
        if (handle) vkDestroyBuffer(device->handle, handle, nullptr);
        if (memory) vkFreeMemory(device->handle, memory, nullptr);
    }
    void create(const Device& d, VkDeviceSize bytes, VkBufferUsageFlags usage, bool host = false) {
        device = &d; size = std::max<VkDeviceSize>(16, (bytes + 3) & ~VkDeviceSize(3));
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size; info.usage = usage;
        VKCHECK(vkCreateBuffer(d.handle, &info, nullptr, &handle));
        VkMemoryRequirements requirements; vkGetBufferMemoryRequirements(d.handle, handle, &requirements);
        memory = d.allocate(requirements, host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                             : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VKCHECK(vkBindBufferMemory(d.handle, handle, memory, 0));
        if (host) VKCHECK(vkMapMemory(d.handle, memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    }
};
struct Image {
    const Device* device = nullptr;
    VkImage handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView sample = VK_NULL_HANDLE, storage = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    ~Image() {
        if (!device) return;
        if (storage && storage != sample) vkDestroyImageView(device->handle, storage, nullptr);
        if (sample) vkDestroyImageView(device->handle, sample, nullptr);
        if (handle) vkDestroyImage(device->handle, handle, nullptr);
        if (memory) vkFreeMemory(device->handle, memory, nullptr);
    }
    void create(const Device& d, uint32_t width, uint32_t height, bool luma = false) {
        device = &d; w = width; h = height;
        if (!w || !h || w > d.properties.limits.maxImageDimension2D || h > d.properties.limits.maxImageDimension2D)
            throw std::runtime_error("image dimensions exceed device limits");
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.flags = luma ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
        info.imageType = VK_IMAGE_TYPE_2D; info.format = luma ? VK_FORMAT_R16_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        info.extent = {w, h, 1}; info.mipLevels = info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VKCHECK(vkCreateImage(d.handle, &info, nullptr, &handle));
        VkMemoryRequirements requirements; vkGetImageMemoryRequirements(d.handle, handle, &requirements);
        memory = d.allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VKCHECK(vkBindImageMemory(d.handle, handle, memory, 0));
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = handle; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = info.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VKCHECK(vkCreateImageView(d.handle, &view, nullptr, &sample));
        storage = sample;
        if (luma) {
            view.format = VK_FORMAT_R16_UINT;
            VKCHECK(vkCreateImageView(d.handle, &view, nullptr, &storage));
        }
    }
};
void imageBarrier(VkCommandBuffer cb, VkImage image, VkImageLayout before, VkImageLayout after,
                  VkPipelineStageFlags source, VkPipelineStageFlags destination, VkAccessFlags src, VkAccessFlags dst) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.image = image; barrier.oldLayout = before; barrier.newLayout = after;
    barrier.srcAccessMask = src; barrier.dstAccessMask = dst;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, source, destination, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}
void memoryBarrier(VkCommandBuffer cb, VkPipelineStageFlags source, VkPipelineStageFlags destination,
                   VkAccessFlags src, VkAccessFlags dst) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = src; barrier.dstAccessMask = dst;
    vkCmdPipelineBarrier(cb, source, destination, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}
struct Pipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;
    ~Pipeline() {
        if (handle) vkDestroyPipeline(device, handle, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
    }
    void create(VkDevice dev, const std::string& file, const std::vector<VkDescriptorType>& types,
                uint32_t subgroup, VkPipelineCache cache) {
        device = dev;
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < types.size(); ++i)
            bindings.push_back({i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        di.bindingCount = static_cast<uint32_t>(bindings.size()); di.pBindings = bindings.data();
        VKCHECK(vkCreateDescriptorSetLayout(dev, &di, nullptr, &descriptors));
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1; li.pSetLayouts = &descriptors;
        VKCHECK(vkCreatePipelineLayout(dev, &li, nullptr, &layout));
        auto bytes = read(file);
        if (bytes.size() % 4 || bytes.size() < 20) throw std::runtime_error("invalid SPIR-V: " + file);
        std::vector<uint32_t> code(bytes.size() / 4); std::memcpy(code.data(), bytes.data(), bytes.size());
        if (code[0] != 0x07230203) throw std::runtime_error("invalid SPIR-V magic: " + file);
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = bytes.size(); si.pCode = code.data();
        VkShaderModule shader;
        VKCHECK(vkCreateShaderModule(dev, &si, nullptr, &shader));
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        required.requiredSubgroupSize = subgroup;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.layout = layout;
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &required,
                    VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", nullptr};
        VkResult result = vkCreateComputePipelines(dev, cache, 1, &ci, nullptr, &handle);
        vkDestroyShaderModule(dev, shader, nullptr);
        check(result, "vkCreateComputePipelines");
    }
};
VkDescriptorSet descriptorSet(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    info.descriptorPool = pool; info.descriptorSetCount = 1; info.pSetLayouts = &layout;
    VkDescriptorSet result;
    VKCHECK(vkAllocateDescriptorSets(device, &info, &result));
    return result;
}
void bindBuffer(VkDevice device, VkDescriptorSet set, uint32_t binding, const Buffer& buffer, VkDeviceSize size = VK_WHOLE_SIZE) {
    VkDescriptorBufferInfo info{buffer.handle, 0, size};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set; write.dstBinding = binding; write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}
void bindImage(VkDevice device, VkDescriptorSet set, uint32_t binding, VkImageView view, VkSampler sampler = VK_NULL_HANDLE,
               VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) {
    VkDescriptorImageInfo info{sampler, view, layout};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set; write.dstBinding = binding; write.descriptorCount = 1;
    write.descriptorType = sampler ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}
void dispatch(VkCommandBuffer cb, const Pipeline& pipeline, VkDescriptorSet set, uint32_t x, uint32_t y) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cb, x, y, 1);
}
constexpr const char* Names[KLayers] = {"enc0", "enc1", "enc2", "enc3", "enc4", "dec5", "dec4", "dec3", "dec2", "dec1", "dec0"};
constexpr uint32_t Channels[KLayers] = {32, 64, 64, 96, 128, 160, 128, 96, 64, 64, 32};
constexpr uint32_t Heads[KLayers] = {2, 2, 2, 4, 4, 8, 4, 4, 2, 2, 2};
constexpr uint32_t Merged[5] = {64, 64, 96, 128, 160};
struct LayerParams { LayerShape shape; std::array<uint32_t, 108> offsets; };
struct ExposureParams { int32_t origin[2], size[2]; float invCount, scale, key; uint32_t flags, blocks, blocksX; int32_t sampleSize[2];
                        float gameScale = 1, invPre = 1; }; // the last two: preset K, shader revision 2
static_assert(sizeof(ExposureParams) == 56);
bool exists(const std::string& file) { return bool(std::ifstream(file, std::ios::binary)); }
static_assert(sizeof(LayerParams) == 448);
} // namespace

Shape makeKShape(uint32_t width, uint32_t height) {
    if (width < 256 || height < 256 || width > 8192 || height > 8192)
        throw std::runtime_error("standard K output dimensions must be 256..8192");
    Shape shape{}; shape.outputWidth = width; shape.outputHeight = height;
    uint32_t tw = std::max(256u, ((width + 3) / 4 + 31) / 32 * 32);
    uint32_t th = std::max(256u, ((height + 3) / 4 + 31) / 32 * 32);
    for (unsigned i = 0; i < KLayers; ++i) {
        unsigned scale = i <= 5 ? i : 10 - i;
        shape.layers[i] = {int32_t(tw >> scale), int32_t(th >> scale), 4, 4};
    }
    return shape;
}
std::array<LayerShape, KLayers> makeKWindows(const Shape& shape, uint32_t frameIndex) {
    auto windows = shape.layers;
    // Observed over 64 consecutive NGX evaluations: an eight-frame cycle that visits every shift once per axis.
    static constexpr int32_t Cycle[8][2] = {{0, 2}, {5, 6}, {4, 0}, {1, 4}, {7, 5}, {2, 1}, {3, 7}, {6, 3}};
    int32_t x = Cycle[frameIndex % 8][0], y = Cycle[frameIndex % 8][1];
    for (unsigned depth = 0; depth < 3; ++depth) { // dec5, then enc4/dec4, then enc3/dec3
        windows[5 - depth].sx = windows[5 + depth].sx = x; windows[5 - depth].sy = windows[5 + depth].sy = y;
        x = (2 * x + 4) & 7; y = (2 * y + 4) & 7;
    }
    return windows;
}
uint32_t kReconstructionTable(uint32_t outputWidth, uint32_t outputHeight, uint32_t renderWidth, uint32_t renderHeight) {
    const double ratio = std::sqrt(double(outputWidth) * outputHeight / (double(renderWidth) * renderHeight));
    const double position = (ratio - 0.5) * 28.0 / 3.0;                 // table index on NGX's ratio grid
    const double even = 2.0 * std::floor(position * 0.5 + 0.5);         // nearest even table; a tie takes the upper
    return uint32_t(std::clamp(even, 0.0, 14.0));
}
FrameParams makeKFrameParams(const Shape& shape, const FrameSettings& settings) {
    if (!settings.renderWidth || !settings.renderHeight || settings.renderWidth > shape.outputWidth || settings.renderHeight > shape.outputHeight)
        throw std::runtime_error("render dimensions must be positive and no larger than the output");
    FrameParams params{}; auto& in = params.input; auto& out = params.output;
    auto base = shape.layers[0];
    in.tokW = out.tokW = base.w; out.tokH = base.h;
    in.flags = out.flags = (settings.reset ? FrameFlagReset : 0) | (settings.depthInverted ? 0 : FrameFlagRegularDepth) | (settings.displayMotion ? FrameFlagDisplayMotion : 0);
    out.flags |= kReconstructionTable(shape.outputWidth, shape.outputHeight, settings.renderWidth, settings.renderHeight) << FrameFlagTableShift;
    in.expo = out.expo = settings.colorExposure;
    in.expoRatio = out.expoRatio = settings.historyExposureRatio;
    in.expoIn = settings.networkExposureScale; in.minW = 1e-4f;
    out.sharp = settings.sharpness; out.minAlpha = .02f; out.alphaThr = .0200042724609375f;
    out.lutScale = 8; out.c184 = .040008544921875f; out.c186 = 1.0419921875f;
    for (unsigned axis = 0; axis < 2; ++axis) {
        float render = axis ? settings.renderHeight : settings.renderWidth, output = axis ? shape.outputHeight : shape.outputWidth;
        float tokens = axis ? base.h : base.w;
        (axis ? out.kB : out.kA) = -16.f * output / render;
        in.out2rnd[axis] = out.out2rnd[axis] = out.near2rnd[axis] = render / output;
        out.sub2rnd[axis] = .5f * render / output; out.out2tok[axis] = .5f;
        in.rndOffs[axis] = settings.jitter[axis]; out.rndOffs[axis] = settings.jitter[axis] - .5f;
        in.mvScale[axis] = out.mvScale[axis] = settings.motionScale[0] == 0 && settings.motionScale[1] == 0 ? (settings.displayMotion ? 1.f : output / render) : settings.motionScale[axis];
        in.mvOffs[axis] = out.mvOffs[axis] = settings.motionOffset[axis];
        in.invHist[axis] = in.invValid[axis] = out.invOut[axis] = out.invOutValid[axis] = 1.f / output;
        in.tok2uv[axis] = 1.f / (tokens * 4.f);
        in.colSize[axis] = in.depSize[axis] = out.colMax[axis] = out.depMax[axis] = int32_t(render) - 1;
        in.mvSize[axis] = out.mvMax[axis] = int32_t(settings.displayMotion ? output : render) - 1;
        out.outSize[axis] = int32_t(output);
    }
    return params;
}

struct Model::Impl {
    Device device;
    Buffer weights, lut, weightUpload, lutUpload;
    std::array<std::array<uint32_t, 108>, KLayers> offsets{};
    std::array<Pipeline, KLayers> network;
    Pipeline input, output, outputRgb10;
    std::array<Pipeline, 2> exposure;
    bool uploaded = false, hasExposure = false, hasDirectOrigins = false, hasRegularDepth = false, hasDisplayMotion = false,
         hasGameExposure = false, hasRatioTables = false;
    Impl(VkPhysicalDevice physical, VkDevice dev) : device(physical, dev) {}
};
Model::Model(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache)
    : impl(std::make_unique<Impl>(physical, device)) {
    auto& m = *impl;
    if (exists(directory + "/direct_origins.bin")) {
        auto marker = read(directory + "/direct_origins.bin");
        // Shader revision: 1 has the corrected motion origins; 2 also reads the frame flags (regular depth);
        // 3 also reads the display-motion flag and the game's exposure texture.
        if (marker.size() != 8 || std::memcmp(marker.data(), "D4RO000", 7) || marker[7] < '1' || marker[7] > '3')
            throw std::runtime_error("incompatible direct-origin shader marker");
        m.hasDirectOrigins = true;
        m.hasRegularDepth = marker[7] >= '2';
        m.hasDisplayMotion = m.hasGameExposure = marker[7] == '3';
    }
    auto offsets = read(directory + "/offsets.bin");
    uint32_t count = 0, weightSize = 0;
    if (offsets.size() != 16 + KLayers * 108 * 4 || std::memcmp(offsets.data(), "D4RK0002", 8))
        throw std::runtime_error("incompatible K model metadata");
    std::memcpy(&count, offsets.data() + 8, 4); std::memcpy(&weightSize, offsets.data() + 12, 4);
    auto weights = read(directory + "/weights.bin"), lut = read(directory + "/lut.bin");
    // One reconstruction table (older packages: the ratio it was captured at serves every ratio) or all sixteen.
    if (count != KLayers || weights.size() != weightSize || weights.size() % 2 ||
        (lut.size() != KTableBytes && lut.size() != size_t(KTableBytes) * KTableCount))
        throw std::runtime_error("invalid K model sizes");
    m.hasRatioTables = lut.size() != KTableBytes;
    std::memcpy(m.offsets.data(), offsets.data() + 16, sizeof(m.offsets));
    for (unsigned i = 0; i < KLayers; ++i) {
        const auto& layer = m.offsets[i]; uint32_t c = Channels[i];
        auto span = [&](unsigned slot, uint64_t elements) {
            uint32_t offset = layer[slot];
            if (offset % 16 || (uint64_t(offset) + elements) * 2 > weights.size())
                throw std::runtime_error(std::string("invalid tensor span in ") + Names[i]);
        };
        for (unsigned j = 0; j < 4; ++j) span(j, 16 * c);
        for (unsigned h = 0; h < Heads[i]; ++h) {
            for (unsigned j = 0; j < 3; ++j) span(4 + 5 * h + j, c * 32);
            span(7 + 5 * h, 4096); span(8 + 5 * h, 32 * c);
        }
        for (unsigned ch = 0; ch < c / 8; ++ch) {
            span(44 + 3 * ch, c * 32); span(45 + 3 * ch, 512); span(46 + 3 * ch, 32 * c);
        }
        if (i < 5) { span(104, 4 * c * Merged[i]); span(105, 16 * Merged[i]); }
        if (i >= 6) { span(104, Channels[i - 1] * 4 * c); span(105, 16 * 4 * c); }
        if (i == 0) { span(106, 16 * c); span(107, 16 * c); }
        if (i == 10) { span(106, c * 48); span(107, 16 * 48); }
    }
    if (weights.size() > m.device.properties.limits.maxStorageBufferRange)
        throw std::runtime_error("model weights exceed the device's storage buffer range");
    m.weights.create(m.device, weights.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m.lut.create(m.device, lut.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m.weightUpload.create(m.device, weights.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    m.lutUpload.create(m.device, lut.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    std::memcpy(m.weightUpload.mapped, weights.data(), weights.size()); std::memcpy(m.lutUpload.mapped, lut.data(), lut.size());
    const auto b = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, t = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    for (unsigned i = 0; i < KLayers; ++i) m.network[i].create(device, directory + "/" + Names[i] + ".spv", {b,b,b,b,b,b}, 32, cache);
    m.input.create(device, directory + "/input_k.spv", {b,t,t,t,t,t,b}, 64, cache);
    m.output.create(device, directory + "/output_k.spv", {b,t,t,t,t,t,s,s,s,s,b,b}, 64, cache);
    if (exists(directory + "/output_k_rgb10.spv"))
        m.outputRgb10.create(device, directory + "/output_k_rgb10.spv", {b,t,t,t,t,t,s,s,s,s,b,b}, 64, cache);
    if (exists(directory + "/exposure_k0.spv") && exists(directory + "/exposure_k1.spv")) {
        m.exposure[0].create(device, directory + "/exposure_k0.spv", {b,t,b}, 64, cache);
        m.exposure[1].create(device, directory + "/exposure_k1.spv", {b,b,b,b,b,t}, 64, cache);
        m.hasExposure = true;
    }
}
bool Model::hasAutoExposure() const { return impl->hasExposure; }
bool Model::hasDirectOrigins() const { return impl->hasDirectOrigins; }
bool Model::hasRegularDepth() const { return impl->hasRegularDepth; }
bool Model::hasDisplayMotion() const { return impl->hasDisplayMotion; }
bool Model::hasGameExposure() const { return impl->hasGameExposure; }
bool Model::hasRatioTables() const { return impl->hasRatioTables; }
bool Model::hasRgb10Output() const { return impl->outputRgb10.handle != VK_NULL_HANDLE; }
Model::~Model() = default;
bool Model::uploadRecorded() const { return impl->uploaded; }
void Model::recordUpload(VkCommandBuffer cb) {
    auto& m = *impl;
    if (m.uploaded) throw std::runtime_error("model upload already recorded");
    VkBufferCopy copy{0, 0, m.weightUpload.size}; vkCmdCopyBuffer(cb, m.weightUpload.handle, m.weights.handle, 1, &copy);
    copy.size = m.lutUpload.size; vkCmdCopyBuffer(cb, m.lutUpload.handle, m.lut.handle, 1, &copy);
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    m.uploaded = true;
}

struct Engine::Impl {
    Model::Impl& model;
    Shape shape;
    Buffer inParams, outParams, dummy;
    Buffer exposureParams, exposureSums, exposureState;
    std::array<VkDescriptorSet, 2> exposureSets{};
    ExposureParams exposure{};
    bool exposureOn = false;
    std::array<Buffer, KLayers> layerParams;
    std::array<Buffer, 2> main;
    std::array<Buffer, 5> skip;
    struct History { Image color, luma, feature; };
    std::array<History, 2> history;
    Image final;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, KLayers> layerSets{};
    std::array<VkDescriptorSet, 2> inputSets{}, outputSets{};
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    unsigned previous = 0;
    bool initialized = false, frameSet = false, reset = false, windowsChanged = false, rgb10Output = false;
    bool external = false;   // the caller owns the token and head buffers; the layers run outside
    bool frontDone = false;  // recordFront has run and recordBack is due
    // Recorded into each frame's command buffer, so a frame still queued on the GPU keeps its own values.
    InputParams input{}; OutputParams output{};
    struct Bindings {
        FrameImages images{};
        FrameOutput finalOutput{};
        VkDescriptorPool pool = VK_NULL_HANDLE; // the initial binding uses Impl::pool
        std::array<VkDescriptorSet, 2> input{}, output{};
        VkDescriptorSet exposure = VK_NULL_HANDLE;
        VkDescriptorSet exposureUpdate = VK_NULL_HANDLE; // the exposure state update (pass 1), per binding
    };
    std::vector<Bindings> bindings;
    VkDeviceSize activations = 0;
    Impl(Model::Impl& m, const Shape& s) : model(m), shape(s) {}
    ~Impl() {
        auto dev = model.device.handle;
        for (auto& binding : bindings) if (binding.pool) vkDestroyDescriptorPool(dev, binding.pool, nullptr);
        if (pool) vkDestroyDescriptorPool(dev, pool, nullptr);
        if (nearest) vkDestroySampler(dev, nearest, nullptr);
        if (linear) vkDestroySampler(dev, linear, nullptr);
    }
};
Engine::Engine(Model& model, const Shape& shape, const ExternalNetwork* external) : impl(std::make_unique<Impl>(*model.impl, shape)) {
    auto& e = *impl; const auto& d = e.model.device;
    auto dev = d.handle;
    if (!shape.outputWidth || !shape.outputHeight || shape.outputWidth > d.properties.limits.maxImageDimension2D / 2 ||
        shape.outputHeight > d.properties.limits.maxImageDimension2D / 2) throw std::runtime_error("unsupported output dimensions");
    auto base = shape.layers[0];
    if (base.w < 256 || base.h < 256 || base.w % 32 || base.h % 32 || uint64_t(base.w) * 4 < shape.outputWidth || uint64_t(base.h) * 4 < shape.outputHeight ||
        uint32_t(base.w) > shape.outputWidth || uint32_t(base.h) > shape.outputHeight)
        throw std::runtime_error("K token grid must be a multiple of 32, at least 256x256, and cover the output");
    e.external = external != nullptr;
    if (e.external && (!external->tokens || !external->head)) throw std::runtime_error("external network without its buffers");
    for (unsigned i = 0; i < KLayers; ++i) {
        auto layer = shape.layers[i]; unsigned scale = i <= 5 ? i : 10 - i;
        if (layer.w != base.w / (1 << scale) || layer.h != base.h / (1 << scale) || layer.sx < 0 || layer.sx > 7 || layer.sy < 0 || layer.sy > 7 || (i != 5 && (layer.sx % 2 || layer.sy % 2)))
            throw std::runtime_error(std::string("invalid geometry for ") + Names[i]);
        if (e.external) continue; // the layers run outside; their buffer geometry is not this engine's concern
        uint32_t gx = (layer.w + layer.sx + 7) / 8, gy = (layer.h + layer.sy + 7) / 8;
        if (gx > d.properties.limits.maxComputeWorkGroupCount[0] || gy > d.properties.limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error("dispatch exceeds device limits");
    }
    e.inParams.create(d, sizeof(InputParams), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    e.outParams.create(d, sizeof(OutputParams), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!e.external) e.dummy.create(d, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::array<VkDeviceSize, 2> capacity{VkDeviceSize(base.w) * base.h * 16 * 2, 16};
    for (unsigned i = 0; i < KLayers && !e.external; ++i) {
        const auto& ls = shape.layers[i];
        VkDeviceSize bytes = VkDeviceSize(ls.w) * ls.h * (i < 5 ? Merged[i] / 4 : i == 10 ? 40 : Channels[i]) * 2;
        capacity[(i + 1) % 2] = std::max(capacity[(i + 1) % 2], bytes);
    }
    for (unsigned i = 0; i < 2; ++i) {
        if (e.external) { // the caller's buffers: main[0] holds the tokens, main[1] the head the final store reads
            e.main[i].handle = i ? external->head : external->tokens;
            e.main[i].size = i ? networkHeadBytes(shape.outputWidth, shape.outputHeight) : networkTokenBytes(shape.outputWidth, shape.outputHeight);
            continue;
        }
        if (capacity[i] > d.properties.limits.maxStorageBufferRange) throw std::runtime_error("activation buffer exceeds device range");
        e.main[i].create(d, capacity[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        e.activations += e.main[i].size;
    }
    for (unsigned i = 0; i < 5 && !e.external; ++i) {
        auto ls = shape.layers[i]; VkDeviceSize bytes = VkDeviceSize(ls.w) * ls.h * Channels[i] * 2;
        if (bytes > d.properties.limits.maxStorageBufferRange) throw std::runtime_error("skip buffer exceeds device range");
        e.skip[i].create(d, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        e.activations += e.skip[i].size;
    }
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, e.external ? 16u : 96u}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 24}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = e.external ? 6u : 17u; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
    VKCHECK(vkCreateDescriptorPool(dev, &pi, nullptr, &e.pool));
    for (unsigned i = 0; i < KLayers && !e.external; ++i) {
        e.layerParams[i].create(d, sizeof(LayerParams), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        LayerParams params{shape.layers[i], e.model.offsets[i]};
        std::memcpy(e.layerParams[i].mapped, &params, sizeof(params));
        auto set = e.layerSets[i] = descriptorSet(dev, e.pool, e.model.network[i].descriptors);
        bindBuffer(dev, set, 0, e.layerParams[i]); bindBuffer(dev, set, 1, e.model.weights);
        bindBuffer(dev, set, 2, e.main[i % 2]);
        bindBuffer(dev, set, 3, i >= 6 ? e.skip[10 - i] : e.dummy);
        bindBuffer(dev, set, 4, e.main[(i + 1) % 2]);
        bindBuffer(dev, set, 5, i < 5 ? e.skip[i] : e.dummy);
    }
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VKCHECK(vkCreateSampler(dev, &si, nullptr, &e.nearest));
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    VKCHECK(vkCreateSampler(dev, &si, nullptr, &e.linear));
    e.final.create(d, shape.outputWidth, shape.outputHeight);
    for (unsigned i = 0; i < 2; ++i) {
        auto& h = e.history[i];
        h.color.create(d, shape.outputWidth, shape.outputHeight);
        h.luma.create(d, shape.outputWidth * 2, shape.outputHeight * 2, true);
        h.feature.create(d, base.w, base.h);
    }
    if (e.model.hasExposure) {
        // One partial sum per 32x32 block of the native output/2 exposure sample grid.
        VkDeviceSize blocks = VkDeviceSize((shape.outputWidth / 2 + 31) / 32) * ((shape.outputHeight / 2 + 31) / 32);
        e.exposureParams.create(d, sizeof(ExposureParams), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        e.exposureSums.create(d, blocks * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        e.exposureState.create(d, 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        for (unsigned i = 0; i < 2; ++i) {
            e.exposureSets[i] = descriptorSet(dev, e.pool, e.model.exposure[i].descriptors);
            bindBuffer(dev, e.exposureSets[i], 0, e.exposureParams);
        }
        bindBuffer(dev, e.exposureSets[0], 2, e.exposureSums);
    }
    for (unsigned i = 0; i < 2; ++i) {
        auto& h = e.history[i];
        e.inputSets[i] = descriptorSet(dev, e.pool, e.model.input.descriptors);
        e.outputSets[i] = descriptorSet(dev, e.pool, e.model.output.descriptors);
        bindBuffer(dev, e.inputSets[i], 0, e.inParams); bindBuffer(dev, e.inputSets[i], 6, e.main[0]);
        // NGX creates both of these texture objects with linear filtering.
        bindImage(dev, e.inputSets[i], 2, h.color.sample, e.linear);
        bindImage(dev, e.inputSets[i], 5, h.feature.sample, e.linear);
        auto out = e.outputSets[i]; auto& dest = e.history[1 - i];
        bindBuffer(dev, out, 0, e.outParams); bindBuffer(dev, out, 10, e.main[1]); bindBuffer(dev, out, 11, e.model.lut);
        bindImage(dev, out, 2, h.color.sample, e.linear); bindImage(dev, out, 3, h.luma.sample, e.linear);
        bindImage(dev, out, 6, dest.color.storage); bindImage(dev, out, 7, dest.luma.storage);
        bindImage(dev, out, 8, dest.feature.storage); bindImage(dev, out, 9, e.final.storage);
    }
}
Engine::~Engine() = default;
void Engine::recordInitialize(VkCommandBuffer cb) {
    auto& e = *impl;
    if (e.initialized) throw std::runtime_error("engine initialization already recorded");
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    auto init = [&](Image& image) {
        imageBarrier(cb, image.handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdClearColorImage(cb, image.handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    };
    for (auto& history : e.history) { init(history.color); init(history.luma); init(history.feature); }
    init(e.final);
    if (e.model.hasExposure) {
        const float one[2] = {1, 1};
        vkCmdUpdateBuffer(cb, e.exposureState.handle, 0, sizeof(one), one);
    }
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    e.initialized = true;
}
void Engine::setFrame(const FrameImages& images, const InputParams& input, const OutputParams& output,
                      const std::array<LayerShape, KLayers>* windows, const ExposureControl* exposure,
                      FrameOutput finalOutput) {
    auto& e = *impl;
    if (finalOutput.view && finalOutput.format != VK_FORMAT_R16G16B16A16_SFLOAT &&
        (finalOutput.format != VK_FORMAT_A2B10G10R10_UNORM_PACK32 || !e.model.outputRgb10.handle))
        throw std::runtime_error("unsupported borrowed output format or missing RGB10A2 shader");
    if (exposure) {
        if (!e.model.hasExposure) throw std::runtime_error("the model has no exposure shaders (recompile it with compile_k.py)");
        if (exposure->game && (!e.model.hasGameExposure || !images.exposure)) throw std::runtime_error("game exposure needs recompiled K shaders and an exposure view");
        if (exposure->size[0] <= 0 || exposure->size[1] <= 0 || exposure->origin[0] < 0 || exposure->origin[1] < 0 ||
            uint32_t(exposure->size[0]) > e.shape.outputWidth || uint32_t(exposure->size[1]) > e.shape.outputHeight ||
            !(exposure->scale > 0) || !(exposure->key > 0) || !std::isfinite(exposure->scale) || !std::isfinite(exposure->key) ||
            !(exposure->exposureScale > 0) || !(exposure->preExposure > 0) || !std::isfinite(exposure->exposureScale) ||
            !std::isfinite(exposure->preExposure))
            throw std::runtime_error("unsupported exposure rectangle or scale");
    }
    if (!images.color || !images.motion || !images.depth) throw std::runtime_error("frame has a null image view");
    if (input.tokW != e.shape.layers[0].w || output.tokW != input.tokW || output.tokH != e.shape.layers[0].h ||
        input.flags != (output.flags & ~FrameFlagTableMask) || (input.flags & ~(FrameFlagReset | FrameFlagRegularDepth | FrameFlagDisplayMotion)) || output.outSize[0] != int32_t(e.shape.outputWidth) || output.outSize[1] != int32_t(e.shape.outputHeight) ||
        (finalOutput.view ? output.outOrigin[0] < 0 || output.outOrigin[1] < 0 : output.outOrigin[0] || output.outOrigin[1]) || output.out2tok[0] != .5f || output.out2tok[1] != .5f)
        throw std::runtime_error("frame parameters do not match engine dimensions");
    if ((input.flags & FrameFlagRegularDepth) && !e.model.hasRegularDepth)
        throw std::runtime_error("regular depth needs a model compiled by the current compile_k.py");
    if ((input.flags & FrameFlagDisplayMotion) && !e.model.hasDisplayMotion)
        throw std::runtime_error("display-resolution motion needs a model compiled by the current compile_k.py");
    if (!(output.out2rnd[0] > 0 && output.out2rnd[0] <= 1 && output.out2rnd[1] > 0 && output.out2rnd[1] <= 1) ||
        output.near2rnd[0] != output.out2rnd[0] || output.near2rnd[1] != output.out2rnd[1] ||
        !std::isfinite(output.sharp) ||
        !(std::isfinite(input.expo) && input.expo > 0 && std::isfinite(input.expoIn) && input.expoIn > 0 &&
          std::isfinite(output.expo) && output.expo > 0 && std::isfinite(input.expoRatio) && input.expoRatio > 0 &&
          std::isfinite(output.expoRatio) && output.expoRatio > 0)) throw std::runtime_error("unsupported scale or exposure");
    if (windows) {
        for (unsigned i = 0; i < KLayers; ++i) {
            auto old = e.shape.layers[i], ls = (*windows)[i];
            if (ls.w != old.w || ls.h != old.h || ls.sx < 0 || ls.sx > 7 || ls.sy < 0 || ls.sy > 7 || (i != 5 && (ls.sx % 2 || ls.sy % 2)))
                throw std::runtime_error(std::string("unsupported frame window geometry: ") + Names[i]);
        }
        e.shape.layers = *windows; e.windowsChanged = true;
    }
    e.input = input; e.output = output;
    // A single-table package has only table 0, and its shaders may predate the table bits.
    if (!e.model.hasRatioTables) e.output.flags &= ~FrameFlagTableMask;
    e.exposureOn = exposure != nullptr;
    if (exposure) {
        const uint32_t sw = e.shape.outputWidth / 2, sh = e.shape.outputHeight / 2;
        const uint32_t bx = (sw + 31) / 32, by = (sh + 31) / 32;
        e.exposure = {{exposure->origin[0], exposure->origin[1]}, {exposure->size[0], exposure->size[1]},
                      1.f / (float(sw) * float(sh)), exposure->scale / exposure->preExposure, exposure->key,
                      (exposure->measure && !exposure->game ? 1u : 0u) | (exposure->reset ? 2u : 0u) | (exposure->game ? 4u : 0u), bx * by, bx,
                      {int32_t(sw), int32_t(sh)}, exposure->exposureScale, 1.f / exposure->preExposure};
    }
    auto dev = e.model.device.handle;
    auto sameImages = [&](const FrameImages& b) {
        return images.color == b.color && images.motion == b.motion && images.depth == b.depth &&
               images.colorLayout == b.colorLayout && images.motionLayout == b.motionLayout && images.depthLayout == b.depthLayout &&
               images.exposure == b.exposure && images.exposureLayout == b.exposureLayout;
    };
    auto found = std::find_if(e.bindings.begin(), e.bindings.end(), [&](const auto& b) {
        return sameImages(b.images) && finalOutput.view == b.finalOutput.view && finalOutput.format == b.finalOutput.format;
    });
    if (found == e.bindings.end()) {
        Impl::Bindings binding;
        binding.images = images; binding.finalOutput = finalOutput;
        if (e.bindings.empty()) {
            binding.input = e.inputSets; binding.output = e.outputSets; binding.exposure = e.exposureSets[0];
            binding.exposureUpdate = e.exposureSets[1];
        } else {
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 24}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8}};
            VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            info.maxSets = 6; info.poolSizeCount = 3; info.pPoolSizes = sizes;
            VKCHECK(vkCreateDescriptorPool(dev, &info, nullptr, &binding.pool));
        }
        // Only new descriptor sets are written. Existing command buffers retain their original textures.
        try {
            if (e.model.hasExposure) {
                if (binding.pool) {
                    binding.exposure = descriptorSet(dev, binding.pool, e.model.exposure[0].descriptors);
                    binding.exposureUpdate = descriptorSet(dev, binding.pool, e.model.exposure[1].descriptors);
                }
                bindBuffer(dev, binding.exposure, 0, e.exposureParams);
                bindImage(dev, binding.exposure, 1, images.color, e.nearest, images.colorLayout);
                bindBuffer(dev, binding.exposure, 2, e.exposureSums);
                // The sampled exposure view is read only in game mode; the colour image stands in otherwise.
                const bool game = images.exposure != VK_NULL_HANDLE;
                bindBuffer(dev, binding.exposureUpdate, 0, e.exposureParams);
                bindBuffer(dev, binding.exposureUpdate, 1, e.exposureSums);
                bindBuffer(dev, binding.exposureUpdate, 2, e.exposureState);
                bindBuffer(dev, binding.exposureUpdate, 3, e.inParams);
                bindBuffer(dev, binding.exposureUpdate, 4, e.outParams);
                bindImage(dev, binding.exposureUpdate, 5, game ? images.exposure : images.color, e.nearest, game ? images.exposureLayout : images.colorLayout);
            }
            for (unsigned i = 0; i < 2; ++i) {
                if (binding.pool) {
                    binding.input[i] = descriptorSet(dev, binding.pool, e.model.input.descriptors);
                    binding.output[i] = descriptorSet(dev, binding.pool, e.model.output.descriptors);
                }
                auto in = binding.input[i], out = binding.output[i];
                auto& h = e.history[i]; auto& dest = e.history[1 - i];
                bindBuffer(dev, in, 0, e.inParams); bindBuffer(dev, in, 6, e.main[0]);
                bindImage(dev, in, 1, images.color, e.nearest, images.colorLayout);
                bindImage(dev, in, 2, h.color.sample, e.linear);
                bindImage(dev, in, 3, images.motion, e.nearest, images.motionLayout);
                bindImage(dev, in, 4, images.depth, e.nearest, images.depthLayout);
                bindImage(dev, in, 5, h.feature.sample, e.linear);
                bindBuffer(dev, out, 0, e.outParams); bindBuffer(dev, out, 10, e.main[1]); bindBuffer(dev, out, 11, e.model.lut);
                bindImage(dev, out, 1, images.color, e.nearest, images.colorLayout);
                bindImage(dev, out, 2, h.color.sample, e.linear); bindImage(dev, out, 3, h.luma.sample, e.linear);
                bindImage(dev, out, 4, images.motion, e.nearest, images.motionLayout);
                bindImage(dev, out, 5, images.depth, e.nearest, images.depthLayout);
                bindImage(dev, out, 6, dest.color.storage); bindImage(dev, out, 7, dest.luma.storage);
                bindImage(dev, out, 8, dest.feature.storage);
                bindImage(dev, out, 9, finalOutput.view ? finalOutput.view : e.final.storage);
            }
            e.bindings.push_back(binding);
        } catch (...) {
            if (binding.pool) vkDestroyDescriptorPool(dev, binding.pool, nullptr);
            throw;
        }
        found = e.bindings.end() - 1;
    }
    e.inputSets = found->input; e.outputSets = found->output;
    e.exposureSets[0] = found->exposure;
    e.exposureSets[1] = found->exposureUpdate;
    e.rgb10Output = finalOutput.view && finalOutput.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    e.reset = (input.flags & FrameFlagReset) != 0; e.frameSet = true;
}
void Engine::recordFrame(VkCommandBuffer cb) {
    recordFrame(cb, {});
}
void Engine::recordFrame(VkCommandBuffer cb, const FrameProfiler& profiler) {
    auto& e = *impl;
    if (e.external) throw std::runtime_error("this engine's network is external: record the front and the back");
    if (!e.initialized || !e.model.uploaded || !e.frameSet) throw std::runtime_error("initialize/upload/setFrame before recording inference");
    auto mark = [&](uint32_t boundary) { if (profiler.mark) profiler.mark(cb, boundary, profiler.user); };
    mark(0);
    // Includes reads-to-writes when reusing scratch and histories, and copies made by the caller after the last frame.
    memoryBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdUpdateBuffer(cb, e.inParams.handle, 0, sizeof(InputParams), &e.input);
    vkCmdUpdateBuffer(cb, e.outParams.handle, 0, sizeof(OutputParams), &e.output);
    if (e.exposureOn) vkCmdUpdateBuffer(cb, e.exposureParams.handle, 0, sizeof(ExposureParams), &e.exposure);
    if (e.windowsChanged) {
        for (unsigned i = 0; i < KLayers; ++i) vkCmdUpdateBuffer(cb, e.layerParams[i].handle, 0, sizeof(LayerShape), &e.shape.layers[i]);
        e.windowsChanged = false;
    }
    if (e.reset) {
        VkClearColorValue zero{}; VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (auto& h : e.history) for (auto* im : {&h.color, &h.luma, &h.feature})
            vkCmdClearColorImage(cb, im->handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    mark(1);
    if (e.exposureOn) {
        const auto compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        if (e.exposure.flags & 1u) {
            dispatch(cb, e.model.exposure[0], e.exposureSets[0], e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX);
            memoryBarrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
        dispatch(cb, e.model.exposure[1], e.exposureSets[1], 1, 1);
        memoryBarrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    mark(2);
    auto base = e.shape.layers[0];
    dispatch(cb, e.model.input, e.inputSets[e.previous], uint32_t(base.w) / 8, uint32_t(base.h) / 8);
    mark(3);
    for (unsigned i = 0; i < KLayers; ++i) {
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        auto ls = e.shape.layers[i];
        dispatch(cb, e.model.network[i], e.layerSets[i], (ls.w + ls.sx + 7) / 8, (ls.h + ls.sy + 7) / 8);
        mark(4 + i);
    }
    memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    dispatch(cb, e.rgb10Output ? e.model.outputRgb10 : e.model.output, e.outputSets[e.previous],
             (e.shape.outputWidth + 15) / 16, (e.shape.outputHeight + 15) / 16);
    mark(4 + KLayers);
    memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    mark(KTimingStages);
    e.previous = 1 - e.previous;
}
// The front: setup, exposure and the input stage, which writes the tokens the external network reads.
void Engine::recordFront(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) {
    auto& e = *impl;
    if (!e.external) throw std::runtime_error("this engine's network is internal: record the whole frame");
    if (!e.initialized || !e.model.uploaded || !e.frameSet || e.frontDone) throw std::runtime_error("initialize/upload/setFrame before recording inference");
    const auto compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    memoryBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, compute | VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdUpdateBuffer(cb, e.inParams.handle, 0, sizeof(InputParams), &e.input);
    vkCmdUpdateBuffer(cb, e.outParams.handle, 0, sizeof(OutputParams), &e.output);
    if (e.exposureOn) vkCmdUpdateBuffer(cb, e.exposureParams.handle, 0, sizeof(ExposureParams), &e.exposure);
    if (e.reset) {
        VkClearColorValue zero{}; VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (auto& h : e.history) for (auto* im : {&h.color, &h.luma, &h.feature})
            vkCmdClearColorImage(cb, im->handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }
    memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, compute, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    uint32_t query = first;
    auto stamp = [&] { if (stamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, stamps, query++); };
    stamp(); // setup
    if (e.exposureOn) {
        if (e.exposure.flags & 1u) {
            dispatch(cb, e.model.exposure[0], e.exposureSets[0], e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX);
            memoryBarrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
        dispatch(cb, e.model.exposure[1], e.exposureSets[1], 1, 1);
        memoryBarrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    stamp(); // exposure
    auto base = e.shape.layers[0];
    dispatch(cb, e.model.input, e.inputSets[e.previous], uint32_t(base.w) / 8, uint32_t(base.h) / 8);
    stamp(); // input
    memoryBarrier(cb, compute, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    e.frontDone = true;
}
// The back: the final store, which reads the head the external network wrote.
void Engine::recordBack(VkCommandBuffer cb, VkQueryPool stamps, uint32_t first) {
    auto& e = *impl;
    if (!e.frontDone) throw std::runtime_error("recordFront before recordBack");
    const auto compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    memoryBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, compute, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    uint32_t query = first;
    auto stamp = [&] { if (stamps) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, stamps, query++); };
    stamp(); // before the final store
    dispatch(cb, e.rgb10Output ? e.model.outputRgb10 : e.model.output, e.outputSets[e.previous],
             (e.shape.outputWidth + 15) / 16, (e.shape.outputHeight + 15) / 16);
    memoryBarrier(cb, compute, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    stamp(); // after the final store
    e.previous = 1 - e.previous; e.frameSet = false; e.frontDone = false;
}
VkImage Engine::outputImage() const { return impl->final.handle; }
HistoryImages Engine::historyImages() const {
    auto& h = impl->history[impl->previous]; return {h.color.handle, h.luma.handle, h.feature.handle};
}
HistoryImages Engine::initialHistoryImages() const {
    auto& h = impl->history[0]; return {h.color.handle, h.luma.handle, h.feature.handle};
}
VkBuffer Engine::headBuffer() const { return impl->main[1].handle; }
VkDeviceSize Engine::headBytes() const { auto s = impl->shape.layers[0]; return VkDeviceSize(s.w) * s.h * 40 * 2; }
// The external network's buffers: the input stage's 16 f16 per token, the head's 40 f16 per token, both at the
// token grid makeKShape derives from the output.
VkDeviceSize Engine::networkTokenBytes(uint32_t outputWidth, uint32_t outputHeight) {
    auto base = makeKShape(outputWidth, outputHeight).layers[0];
    return VkDeviceSize(base.w) * base.h * 16 * 2;
}
VkDeviceSize Engine::networkHeadBytes(uint32_t outputWidth, uint32_t outputHeight) {
    auto base = makeKShape(outputWidth, outputHeight).layers[0];
    return VkDeviceSize(base.w) * base.h * 40 * 2;
}
VkDeviceSize Engine::activationBytes() const { return impl->activations; }
#include "runtime_m.inc"
} // namespace d4r
