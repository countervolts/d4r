#pragma once
#include <vulkan/vulkan_core.h>
#include "d4r_vulkan_staging_spv.h"

// Uses the actual NGX contract: sampled input views and storage output views.
// The game need not create its images with either TRANSFER usage bit.
struct D4rVkStagingSet {
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool output = false;
};
struct D4rVkStaging {
    VkDevice device = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout readSet = VK_NULL_HANDLE, writeSet = VK_NULL_HANDLE;
    VkPipelineLayout readLayout = VK_NULL_HANDLE, writeLayout = VK_NULL_HANDLE;
    VkPipeline readPipeline = VK_NULL_HANDLE;
    struct OutputPipeline { VkFormat format; VkPipeline pipeline; } outputs[5] = {};
    PFN_vkCreateSampler createSampler = nullptr;
    PFN_vkDestroySampler destroySampler = nullptr;
    PFN_vkCreateDescriptorSetLayout createSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroySetLayout = nullptr;
    PFN_vkCreatePipelineLayout createLayout = nullptr;
    PFN_vkDestroyPipelineLayout destroyLayout = nullptr;
    PFN_vkCreateShaderModule createShader = nullptr;
    PFN_vkDestroyShaderModule destroyShader = nullptr;
    PFN_vkCreateComputePipelines createPipelines = nullptr;
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool createPool = nullptr;
    PFN_vkDestroyDescriptorPool destroyPool = nullptr;
    PFN_vkAllocateDescriptorSets allocateSets = nullptr;
    PFN_vkUpdateDescriptorSets updateSets = nullptr;
    PFN_vkCmdBindPipeline bindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets bindSets = nullptr;
    PFN_vkCmdPushConstants push = nullptr;
    PFN_vkCmdDispatch dispatch = nullptr;

    static bool supported_output(VkFormat format) {
        return format == VK_FORMAT_R16G16B16A16_SFLOAT || format == VK_FORMAT_R32G32B32A32_SFLOAT ||
               format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 ||
               format == VK_FORMAT_R8G8B8A8_UNORM;
    }
    bool pipeline(const uint32_t* code, size_t bytes, VkPipelineLayout layout, VkPipeline& result) {
        VkShaderModuleCreateInfo shader = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = bytes; shader.pCode = code;
        VkShaderModule module = VK_NULL_HANDLE;
        if (createShader(device, &shader, nullptr, &module) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; info.stage.module = module; info.stage.pName = "main";
        info.layout = layout;
        const VkResult status = createPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &result);
        destroyShader(device, module, nullptr);
        return status == VK_SUCCESS;
    }
    bool init(VkDevice d, PFN_vkGetDeviceProcAddr proc) {
        device = d;
#define D4R_LOAD(member, name) member = reinterpret_cast<decltype(member)>(proc(d, "vk" #name)); if (!member) return false
        D4R_LOAD(createSampler, CreateSampler); D4R_LOAD(destroySampler, DestroySampler);
        D4R_LOAD(createSetLayout, CreateDescriptorSetLayout); D4R_LOAD(destroySetLayout, DestroyDescriptorSetLayout);
        D4R_LOAD(createLayout, CreatePipelineLayout); D4R_LOAD(destroyLayout, DestroyPipelineLayout);
        D4R_LOAD(createShader, CreateShaderModule); D4R_LOAD(destroyShader, DestroyShaderModule);
        D4R_LOAD(createPipelines, CreateComputePipelines); D4R_LOAD(destroyPipeline, DestroyPipeline);
        D4R_LOAD(createPool, CreateDescriptorPool); D4R_LOAD(destroyPool, DestroyDescriptorPool);
        D4R_LOAD(allocateSets, AllocateDescriptorSets); D4R_LOAD(updateSets, UpdateDescriptorSets);
        D4R_LOAD(bindPipeline, CmdBindPipeline); D4R_LOAD(bindSets, CmdBindDescriptorSets);
        D4R_LOAD(push, CmdPushConstants); D4R_LOAD(dispatch, CmdDispatch);
#undef D4R_LOAD
        VkSamplerCreateInfo sample = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sample.magFilter = sample.minFilter = VK_FILTER_NEAREST;
        sample.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sample.addressModeU = sample.addressModeV = sample.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (createSampler(d, &sample, nullptr, &sampler) != VK_SUCCESS) return false;
        for (int output = 0; output < 2; ++output) {
            VkDescriptorSetLayoutBinding bindings[2] = {};
            bindings[0] = {0, output ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                           1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo set = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            set.bindingCount = 2; set.pBindings = bindings;
            auto& setLayout = output ? writeSet : readSet;
            if (createSetLayout(d, &set, nullptr, &setLayout) != VK_SUCCESS) return false;
            VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, output ? 8u : 12u};
            VkPipelineLayoutCreateInfo layout = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout.setLayoutCount = 1; layout.pSetLayouts = &setLayout;
            layout.pushConstantRangeCount = 1; layout.pPushConstantRanges = &range;
            if (createLayout(d, &layout, nullptr, output ? &writeLayout : &readLayout) != VK_SUCCESS) return false;
        }
        return pipeline(d4r_vk_read, sizeof(d4r_vk_read), readLayout, readPipeline);
    }
    VkPipeline output_pipeline(VkFormat format) {
        for (auto& item : outputs) if (item.pipeline && item.format == format) return item.pipeline;
        const uint32_t* code = nullptr; size_t bytes = 0;
#define D4R_CODE(name) code = d4r_vk_write_##name; bytes = sizeof(d4r_vk_write_##name)
        switch (format) {
        case VK_FORMAT_R16G16B16A16_SFLOAT: D4R_CODE(rgba16f); break;
        case VK_FORMAT_R32G32B32A32_SFLOAT: D4R_CODE(rgba32f); break;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: D4R_CODE(r11f_g11f_b10f); break;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: D4R_CODE(rgb10_a2); break;
        case VK_FORMAT_R8G8B8A8_UNORM: D4R_CODE(rgba8); break;
        default: return VK_NULL_HANDLE;
        }
#undef D4R_CODE
        for (auto& item : outputs) if (!item.pipeline) {
            if (!pipeline(code, bytes, writeLayout, item.pipeline)) return VK_NULL_HANDLE;
            item.format = format; return item.pipeline;
        }
        return VK_NULL_HANDLE;
    }
    bool set(D4rVkStagingSet& result, VkImageView image, VkFormat format, VkBuffer buffer,
             VkDeviceSize bytes, bool output, VkImageLayout inputLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        result.output = output;
        result.pipeline = output ? output_pipeline(format) : readPipeline;
        if (!result.pipeline) return false;
        VkDescriptorPoolSize sizes[] = {{output ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
        VkDescriptorPoolCreateInfo pool = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1; pool.poolSizeCount = 2; pool.pPoolSizes = sizes;
        if (!result.pool && createPool(device, &pool, nullptr, &result.pool) != VK_SUCCESS) return false;
        const auto layout = output ? writeSet : readSet;
        VkDescriptorSetAllocateInfo alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = result.pool; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &layout;
        if (!result.set && allocateSets(device, &alloc, &result.set) != VK_SUCCESS) return false;
        VkDescriptorImageInfo texture = {output ? VK_NULL_HANDLE : sampler, image,
                                        output ? VK_IMAGE_LAYOUT_GENERAL : inputLayout};
        VkDescriptorBufferInfo storage = {buffer, 0, bytes};
        VkWriteDescriptorSet writes[2] = {};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = result.set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
        writes[0].descriptorType = sizes[0].type; writes[0].pImageInfo = &texture;
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet = result.set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[1].pBufferInfo = &storage;
        updateSets(device, 2, writes, 0, nullptr);
        return true;
    }
    void record(VkCommandBuffer cmd, const D4rVkStagingSet& set, uint32_t width, uint32_t height, uint32_t inputMode = 0) {
        const auto layout = set.output ? writeLayout : readLayout;
        bindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, set.pipeline);
        bindSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set.set, 0, nullptr);
        const uint32_t dimensions[] = {width, height, inputMode};
        push(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, set.output ? 8 : 12, dimensions);
        dispatch(cmd, (width+7)/8, (height+7)/8, 1);
    }
    void free(D4rVkStagingSet& set) {
        if (set.pool) destroyPool(device, set.pool, nullptr);
        set = {};
    }
    void destroy() {
        if (readPipeline) destroyPipeline(device, readPipeline, nullptr);
        for (auto& item : outputs) if (item.pipeline) destroyPipeline(device, item.pipeline, nullptr);
        if (readLayout) destroyLayout(device, readLayout, nullptr);
        if (writeLayout) destroyLayout(device, writeLayout, nullptr);
        if (readSet) destroySetLayout(device, readSet, nullptr);
        if (writeSet) destroySetLayout(device, writeSet, nullptr);
        if (sampler) destroySampler(device, sampler, nullptr);
        *this = {};
    }
};
