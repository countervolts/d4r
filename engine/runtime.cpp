#include "runtime.h"
#include "rr_binding_cache.h"
#include "m_shared.h"
#include "vk_dispatch.h"
#include "rr_camera.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace d4r {
namespace {
bool noBufferRobustness = false;
void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": VkResult " + std::to_string(result));
}
#define VKCHECK(call) check((call), #call)
std::vector<char> read(const std::string& file) { return readFile(file); }
bool exists(const std::string& file) { return fileExists(file); }
// The driver, from VkPhysicalDeviceVulkan12Properties on a Vulkan 1.2 device: OptiScaler's Vulkan spoofing rewrites
// VkPhysicalDeviceDriverProperties to NVIDIA's for every caller in the game's process, the 1.2 core struct it leaves.
VkDriverId driverId(VkPhysicalDevice physical) {
    VkPhysicalDeviceProperties base{};
    vkGetPhysicalDeviceProperties(physical, &base);
    VkPhysicalDeviceVulkan12Properties v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    const bool core = base.apiVersion >= VK_API_VERSION_1_2;
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = core ? static_cast<void*>(&v12) : static_cast<void*>(&driver);
    vkGetPhysicalDeviceProperties2(physical, &properties);
    return core ? v12.driverID : driver.driverID;
}
using LayerParams = KLayerParams;
constexpr const auto& Names = KLayerNames;
constexpr const auto& Channels = KChannels;
struct Device {
    VkDevice handle;
    VkPhysicalDeviceMemoryProperties memory;
    VkPhysicalDeviceProperties properties;
    bool coopmat = false;
    Device(VkPhysicalDevice physical, VkDevice device) : handle(device) {
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        vkGetPhysicalDeviceProperties(physical, &properties);
#ifndef D4R_ENGINE_VK_TABLE
        uint32_t count = 0;
        VKCHECK(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr));
        std::vector<VkExtensionProperties> extensions(count);
        VKCHECK(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()));
        coopmat = std::any_of(extensions.begin(), extensions.end(), [](const auto& extension) {
            return std::strcmp(extension.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0;
        });
        if (const char* force = std::getenv("D4R_RR_TEST_NO_COOPMAT"))
            if (std::strcmp(force, "1") == 0) coopmat = false;
#endif
        // The shim borrows a device whose enabled feature chain is unavailable here.
        // CNN stages use the extension-free flavor instead of assuming physical support
        // means VK_KHR_cooperative_matrix was enabled by the application.
    }
    VkDeviceMemory allocate(VkMemoryRequirements requirements, VkMemoryPropertyFlags flags, bool address = false) const {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) {
                VkMemoryAllocateFlagsInfo addressFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
                addressFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, address ? &addressFlags : nullptr};
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
    // address: the buffer has a device address (VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
    void create(const Device& d, VkDeviceSize bytes, VkBufferUsageFlags usage, bool host = false, bool address = false) {
        device = &d; size = std::max<VkDeviceSize>(16, (bytes + 3) & ~VkDeviceSize(3));
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size; info.usage = usage | (address ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
        VKCHECK(vkCreateBuffer(d.handle, &info, nullptr, &handle));
        VkMemoryRequirements requirements; vkGetBufferMemoryRequirements(d.handle, handle, &requirements);
        memory = d.allocate(requirements, host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                             : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, address);
        VKCHECK(vkBindBufferMemory(d.handle, handle, memory, 0));
        if (host) VKCHECK(vkMapMemory(d.handle, memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    }
};
// A buffer's contents written into the command buffer itself (vkCmdUpdateBuffer, at most 65536 bytes a call), so no
// host-visible staging copy stays allocated; the host copy (padded to the buffer's size) is dropped once recorded
void recordContents(VkCommandBuffer cb, const Buffer& buffer, std::vector<char>& bytes) {
    bytes.resize(buffer.size);
    for (VkDeviceSize at = 0; at < bytes.size(); at += 65536)
        vkCmdUpdateBuffer(cb, buffer.handle, at, std::min<VkDeviceSize>(65536, bytes.size() - at), bytes.data() + at);
    std::vector<char>().swap(bytes);
}
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
// An image created after recordInitialize (an output only some frames need): GENERAL and zero, as recordInitialize
// leaves its images; the caller's next barrier covers the clear
void clearNewImage(VkCommandBuffer cb, VkImage image) {
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    imageBarrier(cb, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
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
    // K's input, output and exposure stages: the same shader with CANONICAL_INPUT (specialization constant 7)
    // false, for frames whose colour and motion are already exact in binary16 (no conversion code at all)
    VkPipeline f16Inputs = VK_NULL_HANDLE;
    ~Pipeline() {
        if (f16Inputs) vkDestroyPipeline(device, f16Inputs, nullptr);
        if (handle) vkDestroyPipeline(device, handle, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
    }
    void create(VkDevice dev, const std::string& file, const std::vector<VkDescriptorType>& types,
                uint32_t subgroup, VkPipelineCache cache, VkPipelineCreateFlags flags = 0,
                const VkSpecializationInfo* specialization = nullptr, bool f16InputVariant = false) {
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
        VkPipelineRobustnessCreateInfoEXT robustness{VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT};
        robustness.storageBuffers = robustness.uniformBuffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, noBufferRobustness ? &robustness : nullptr};
        ci.layout = layout; ci.flags = flags;
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &required,
                    VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", specialization};
        VkResult result = vkCreateComputePipelines(dev, cache, 1, &ci, nullptr, &handle);
        if (result == VK_SUCCESS && f16InputVariant) {
            const VkBool32 canonical = VK_FALSE;
            const VkSpecializationMapEntry entry{7, 0, sizeof(canonical)};
            const VkSpecializationInfo variant{1, &entry, sizeof(canonical), &canonical};
            ci.stage.pSpecializationInfo = &variant;
            result = vkCreateComputePipelines(dev, cache, 1, &ci, nullptr, &f16Inputs);
        }
        vkDestroyShaderModule(dev, shader, nullptr);
        check(result, "vkCreateComputePipelines");
        if (const char* isaDir = std::getenv("D4R_TMP_ISA_DIR")) {   // TEMP diagnostic
            auto info = reinterpret_cast<PFN_vkGetShaderInfoAMD>(vkGetDeviceProcAddr(dev, "vkGetShaderInfoAMD"));
            size_t size = 0;
            if (info && info(dev, handle, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &size, nullptr) == VK_SUCCESS) {
                std::vector<char> text(size);
                info(dev, handle, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &size, text.data());
                std::string name = file.substr(file.find_last_of("/\\") + 1);
                std::ofstream(std::string(isaDir) + "/" + name + ".isa", std::ios::binary).write(text.data(), size);
            }
            VkShaderStatisticsInfoAMD st{};
            size = sizeof(st);
            if (info && info(dev, handle, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_INFO_TYPE_STATISTICS_AMD, &size, &st) == VK_SUCCESS)
                std::fprintf(stderr, "ISA %s vgpr %u sgpr %u lds %u isa %zu\n", file.c_str(), st.resourceUsage.numUsedVgprs,
                             st.resourceUsage.numUsedSgprs, st.resourceUsage.ldsUsageSizeInBytes, st.resourceUsage.scratchMemUsageInBytes);
        }
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
void dispatch(VkCommandBuffer cb, const Pipeline& pipeline, VkDescriptorSet set, uint32_t x, uint32_t y, bool f16Inputs = false) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, f16Inputs && pipeline.f16Inputs ? pipeline.f16Inputs : pipeline.handle);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cb, x, y, 1);
}
} // namespace
void setBufferRobustnessDisabled(bool disabled) { noBufferRobustness = disabled; }


struct Model::Impl {
    Device device;
    Buffer weights, lut;
    std::vector<char> weightBytes, lutBytes;   // until recordUpload
    std::array<std::array<uint32_t, 108>, KLayers> offsets{};
    std::array<Pipeline, KLayers> network;
    Pipeline input, output, outputRgb10, outputRgba8, outputR11;
    // the final-store pipeline for a borrowed output texture of this format; null when none is packaged
    const Pipeline* outputFor(VkFormat format) const {
        const Pipeline* p = format == VK_FORMAT_R16G16B16A16_SFLOAT ? &output :
            format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 ? &outputRgb10 : format == VK_FORMAT_R8G8B8A8_UNORM ? &outputRgba8 :
            format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ? &outputR11 : nullptr;
        return p && p->handle ? p : nullptr;
    }
    std::array<Pipeline, 2> exposure;
    bool uploaded = false, hasExposure = false, hasDirectOrigins = false, hasRegularDepth = false, hasDisplayMotion = false,
         hasGameExposure = false, hasRatioTables = false, accuracyStages = false, dot2 = false;
    Impl(VkPhysicalDevice physical, VkDevice dev) : device(physical, dev) {}
};
Model::Model(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache, KNetwork network,
             bool accuracy)
    : impl(std::make_unique<Impl>(physical, device)) {
    auto& m = *impl;
    if (const char* force = std::getenv("D4R_ENGINE_K_NETWORK")) {
        if (!std::strcmp(force, "dot2")) network = KNetwork::Dot2;
        else if (!std::strcmp(force, "matrix")) network = KNetwork::Matrix;
    }
#ifdef D4R_ENGINE_VK_TABLE
    if (network == KNetwork::Auto) network = KNetwork::Matrix;
#else
    if (network == KNetwork::Auto) network = m.device.coopmat ? KNetwork::Matrix : KNetwork::Dot2;
#endif
    m.dot2 = network == KNetwork::Dot2;
    auto data = loadKModel(directory, m.dot2);
    m.hasDirectOrigins = data.hasDirectOrigins; m.hasRegularDepth = data.hasRegularDepth;
    m.hasDisplayMotion = data.hasDisplayMotion; m.hasGameExposure = data.hasGameExposure;
    m.hasRatioTables = data.hasRatioTables; m.offsets = data.offsets;
    auto& weights = data.weights; auto& lut = data.lut;
    // AMD's Windows driver (LLPC) loads a row-major cooperative-matrix B operand one 16-bit element at a time (1168
    // buffer_load_u16 in dec5): there the network takes the model's <layer>_llpc.spv, which read weights_llpc.bin
    // (every 16x16 B tile transposed, compile_k.py d3d12_weight_tiles) with column-major loads - the same logical
    // operands, so the same output. Every other driver keeps weights.bin.
    bool llpcTiles = !m.dot2 && driverId(physical) == VK_DRIVER_ID_AMD_PROPRIETARY &&
                     !std::getenv("D4R_ENGINE_K_ROW_MAJOR_WEIGHTS");
    for (unsigned i = 0; i < KLayers && llpcTiles; ++i) llpcTiles = exists(directory + "/" + Names[i] + "_llpc.spv");
    if (llpcTiles && exists(directory + "/weights_llpc.bin")) {
        auto tiles = read(directory + "/weights_llpc.bin");
        llpcTiles = tiles.size() == weights.size() + 8 && !std::memcmp(tiles.data(), "D4RWTL01", 8);
        if (llpcTiles) std::memcpy(weights.data(), tiles.data() + 8, weights.size());
    } else llpcTiles = false;
    if (weights.size() > m.device.properties.limits.maxStorageBufferRange)
        throw std::runtime_error("model weights exceed the device's storage buffer range");
    m.weights.create(m.device, weights.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m.lut.create(m.device, lut.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m.weightBytes = std::move(weights); m.lutBytes = std::move(lut);
    const auto b = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, t = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    for (unsigned i = 0; i < KLayers; ++i) {
        std::string spv = directory + "/" + Names[i] + (m.dot2 ? "_dot2.spv" : llpcTiles ? "_llpc.spv" : ".spv");
        // AMD's Windows driver (LLPC) rejects the v_dot2 layers' capability 6912: their _dot2_llpc.spv there
        const std::string llpc = directory + "/" + Names[i] + "_dot2_llpc.spv";
        if (m.dot2 && driverId(physical) == VK_DRIVER_ID_AMD_PROPRIETARY && exists(llpc)) spv = llpc;
        if (m.dot2 && !exists(spv))
            throw std::runtime_error("the model has no network shaders for GPUs without cooperative matrices (" + spv +
                                     "): rebuild it with the current compile_k.py");
        m.network[i].create(device, spv, {b,b,b,b,b,b}, 32, cache);
    }
    // The stages around the network run as wave64 (D4R_ENGINE_K_STAGE_WAVE=32|64 forces one size; the outputs are
    // identical). Without cooperative matrices (RDNA2, which issues wave64 in two passes) the input stage and the
    // reconstruction run as wave32: Steam Deck, 640x360 -> 1280x720, input 615 -> 494 us, reconstruction
    // 1805 -> 1729 us (with the single-pass fallback of output_k.comp; before it wave64 was faster, 1898 vs 2021 us).
    const char* waveEnv = std::getenv("D4R_ENGINE_K_STAGE_WAVE");
    const uint32_t wave = waveEnv && !std::strcmp(waveEnv, "32") ? 32 : 64;
    const uint32_t stageWave = waveEnv ? wave : m.dot2 ? 32 : 64;
    m.input.create(device, directory + "/input_k.spv", {b,t,t,t,t,t,b}, stageWave, cache, 0, nullptr, true);
    // Older packages retain their existing stage. New accuracy packages provide a separate shader for each
    // final-store format; the fast paths continue to load exactly their original files.
    m.accuracyStages = accuracy && exists(directory + "/output_k_accuracy.spv");
    const auto outputFile = [&](const char* name) {
        return directory + "/" + name + (m.accuracyStages ? "_accuracy.spv" : ".spv");
    };
    m.output.create(device, outputFile("output_k"), {b,t,t,t,t,t,s,s,s,s,b,b}, stageWave, cache, 0, nullptr, true);
    if (exists(outputFile("output_k_rgb10")))
        m.outputRgb10.create(device, outputFile("output_k_rgb10"), {b,t,t,t,t,t,s,s,s,s,b,b}, stageWave, cache, 0, nullptr, true);
    if (exists(outputFile("output_k_store_rgba8")))
        m.outputRgba8.create(device, outputFile("output_k_store_rgba8"), {b,t,t,t,t,t,s,s,s,s,b,b}, stageWave, cache, 0, nullptr, true);
    if (exists(outputFile("output_k_store_r11g11b10")))
        m.outputR11.create(device, outputFile("output_k_store_r11g11b10"), {b,t,t,t,t,t,s,s,s,s,b,b}, stageWave, cache, 0, nullptr, true);
    if (exists(directory + "/exposure_k0.spv") && exists(directory + "/exposure_k1.spv")) {
        m.exposure[0].create(device, directory + "/exposure_k0.spv", {b,t,b}, wave, cache, 0, nullptr, true);
        const std::string update = directory + (m.accuracyStages && exists(directory + "/exposure_k1_accuracy.spv")
                                                    ? "/exposure_k1_accuracy.spv" : "/exposure_k1.spv");
        m.exposure[1].create(device, update, {b,b,b,b,b,t}, wave, cache);
        m.hasExposure = true;
    }
}
bool Model::hasAutoExposure() const { return impl->hasExposure; }
bool Model::hasDirectOrigins() const { return impl->hasDirectOrigins; }
bool Model::hasRegularDepth() const { return impl->hasRegularDepth; }
bool Model::hasDisplayMotion() const { return impl->hasDisplayMotion; }
bool Model::hasGameExposure() const { return impl->hasGameExposure; }
bool Model::hasRatioTables() const { return impl->hasRatioTables; }
bool Model::hasAccuracyStages() const { return impl->accuracyStages; }
bool Model::hasRgb10Output() const { return impl->outputRgb10.handle != VK_NULL_HANDLE; }
bool Model::hasDirectOutput(VkFormat format) const { return impl->outputFor(format) != nullptr; }
bool Model::dot2Network() const { return impl->dot2; }
Model::~Model() = default;
bool Model::uploadRecorded() const { return impl->uploaded; }
void Model::recordUpload(VkCommandBuffer cb) {
    auto& m = *impl;
    if (m.uploaded) throw std::runtime_error("model upload already recorded");
    recordContents(cb, m.weights, m.weightBytes); recordContents(cb, m.lut, m.lutBytes);
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
    Image final;   // the engine's own output: only for frames that do not write the caller's image (created on first use)
    bool finalNew = false;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, KLayers> layerSets{};
    std::array<VkDescriptorSet, 2> inputSets{}, outputSets{};
    VkSampler nearest = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    unsigned previous = 0;
    bool initialized = false, frameSet = false, reset = false, windowsChanged = false;
    const Pipeline* finalStore = nullptr; // setFrame: the final-store pipeline for the frame's output
    bool f16Inputs = false;  // the frame's colour and motion are exact in binary16: the stages without the conversion
    bool external = false;   // the caller owns the token and head buffers; the layers run outside
    bool trim = true;        // only the windows the results depend on (kLayerGroups)
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
    validateKShape(shape);
    e.external = external != nullptr;
    if (const char* trim = std::getenv("D4R_ENGINE_K_TRIM")) e.trim = std::strcmp(trim, "0") != 0;
    if (e.external && (!external->tokens || !external->head)) throw std::runtime_error("external network without its buffers");
    for (unsigned i = 0; i < KLayers; ++i) {
        auto layer = shape.layers[i];
        if (e.external) continue; // the layers run outside; their buffer geometry is not this engine's concern
        uint32_t gx = (layer.w + layer.sx + 7) / 8, gy = (layer.h + layer.sy + 7) / 8;
        if (gx > d.properties.limits.maxComputeWorkGroupCount[0] || gy > d.properties.limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error("dispatch exceeds device limits");
    }
    e.inParams.create(d, sizeof(InputParams), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    e.outParams.create(d, sizeof(OutputParams), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!e.external) e.dummy.create(d, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto capacity = kActivationBytes(shape);
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
        VkDeviceSize bytes = kSkipBytes(shape, i);
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
    for (unsigned i = 0; i < 2; ++i) {
        auto& h = e.history[i];
        // NGX's histories cover the output padded to 16 pixels (its output kernel's 16x16 blocks write the padding,
        // and the bilinear history fetches near the last row and column read it)
        const uint32_t hw = (shape.outputWidth + 15) / 16 * 16, hh = (shape.outputHeight + 15) / 16 * 16;
        h.color.create(d, hw, hh);
        h.luma.create(d, hw * 2, hh * 2, true);
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
        bindImage(dev, out, 8, dest.feature.storage);   // binding 9, the output: setFrame
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
    if (finalOutput.view && !e.model.outputFor(finalOutput.format))
        throw std::runtime_error("unsupported borrowed output format or missing final-store shader for it");
    ExposureParams exposureParams{};
    if (exposure) {
        if (!e.model.hasExposure) throw std::runtime_error("the model has no exposure shaders (recompile it with compile_k.py)");
        if (exposure->game && (!e.model.hasGameExposure || !images.exposure)) throw std::runtime_error("game exposure needs recompiled K shaders and an exposure view");
        exposureParams = makeKExposureParams(e.shape, *exposure);
    }
    if (!images.color || !images.motion || !images.depth) throw std::runtime_error("frame has a null image view");
    KModelData capabilities;
    capabilities.hasRegularDepth = e.model.hasRegularDepth; capabilities.hasDisplayMotion = e.model.hasDisplayMotion;
    validateKFrame(e.shape, capabilities, input, output, windows, finalOutput.view != VK_NULL_HANDLE);
    if (windows) {
        e.shape.layers = *windows; e.windowsChanged = true;
    }
    e.input = input; e.output = output;
    // A single-table package has only table 0, and its shaders may predate the table bits.
    if (!e.model.hasRatioTables) e.output.flags &= ~FrameFlagTableMask;
    e.exposureOn = exposure != nullptr;
    if (exposure) e.exposure = exposureParams;
    auto dev = e.model.device.handle;
    auto sameImages = [&](const FrameImages& b) {
        return images.color == b.color && images.motion == b.motion && images.depth == b.depth &&
               images.colorLayout == b.colorLayout && images.motionLayout == b.motionLayout && images.depthLayout == b.depthLayout &&
               images.exposure == b.exposure && images.exposureLayout == b.exposureLayout;
    };
    if (!finalOutput.view && !e.final.handle) {
        e.final.create(e.model.device, e.shape.outputWidth, e.shape.outputHeight);
        e.finalNew = true;
    }
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
    e.finalStore = finalOutput.view ? e.model.outputFor(finalOutput.format) : &e.model.output;
    e.reset = (input.flags & FrameFlagReset) != 0; e.frameSet = true;
    e.f16Inputs = (input.flags & (FrameFlagCanonicalColor | FrameFlagCanonicalMotion)) == 0;
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
    if (e.finalNew) { clearNewImage(cb, e.final.handle); e.finalNew = false; }
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
            dispatch(cb, e.model.exposure[0], e.exposureSets[0], e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX, e.f16Inputs);
            memoryBarrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
        dispatch(cb, e.model.exposure[1], e.exposureSets[1], 1, 1);
        memoryBarrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    mark(2);
    auto base = e.shape.layers[0];
    dispatch(cb, e.model.input, e.inputSets[e.previous], uint32_t(base.w) / 8, uint32_t(base.h) / 8, e.f16Inputs);
    mark(3);
    const auto groups = kLayerGroups(e.shape, e.trim);
    const unsigned layers = kRecordedLayers();
    for (unsigned i = 0; i < layers; ++i) {
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        dispatch(cb, e.model.network[i], e.layerSets[i], groups[i][0], groups[i][1]);
        mark(4 + i);
    }
    memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    dispatch(cb, *e.finalStore, e.outputSets[e.previous],
             (e.shape.outputWidth + 15) / 16, (e.shape.outputHeight + 15) / 16, e.f16Inputs);
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
    if (e.finalNew) { clearNewImage(cb, e.final.handle); e.finalNew = false; }
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
            dispatch(cb, e.model.exposure[0], e.exposureSets[0], e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX, e.f16Inputs);
            memoryBarrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
        dispatch(cb, e.model.exposure[1], e.exposureSets[1], 1, 1);
        memoryBarrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    stamp(); // exposure
    auto base = e.shape.layers[0];
    dispatch(cb, e.model.input, e.inputSets[e.previous], uint32_t(base.w) / 8, uint32_t(base.h) / 8, e.f16Inputs);
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
    dispatch(cb, *e.finalStore, e.outputSets[e.previous],
             (e.shape.outputWidth + 15) / 16, (e.shape.outputHeight + 15) / 16, e.f16Inputs);
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
VkBuffer Engine::tokenBuffer() const { return impl->main[0].handle; }
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
#include "runtime_rr.inc"
#include "runtime_e.inc"
} // namespace d4r
