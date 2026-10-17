// The FSR Ray Regeneration backend's own two compute passes (tools/windows/fsr_rr/fsr_rr_prepare.comp and
// fsr_rr_compose.comp), recorded into the game's D3D12 command list through vkd3d-proton's Vulkan
// interop. Descriptors are pushed (VK_KHR_push_descriptor, which vkd3d-proton requires), so nothing
// here outlives the command buffer that recorded it except the pipelines.
#pragma once
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fsr_rr/fsr_rr_shaders_spv.h"

#include "fsr_rr_constants.h"

// One image a pass reads or writes: the view and the layout the image is in while the pass runs.
struct D4rFsrRrImage
{
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
};

struct D4rFsrRrPasses
{
    static constexpr uint32_t kPrepareInputs = 10, kPrepareOutputs = 8, kComposeImages = 8;

    VkDevice device = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout prepareSet = VK_NULL_HANDLE, composeSet = VK_NULL_HANDLE;
    VkPipelineLayout prepareLayout = VK_NULL_HANDLE, composeLayout = VK_NULL_HANDLE;
    VkPipeline preparePipeline = VK_NULL_HANDLE, composePipeline = VK_NULL_HANDLE;
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
    PFN_vkCmdBindPipeline bindPipeline = nullptr;
    PFN_vkCmdPushConstants push = nullptr;
    PFN_vkCmdPushDescriptorSetKHR pushDescriptors = nullptr;
    PFN_vkCmdDispatch dispatch = nullptr;
    PFN_vkCmdPipelineBarrier barrier = nullptr;
    PFN_vkCreateImageView createView = nullptr;
    PFN_vkDestroyImageView destroyView = nullptr;

    bool pipeline(const uint32_t* code, size_t bytes, VkPipelineLayout layout, VkPipeline& result)
    {
        VkShaderModuleCreateInfo shader = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = bytes;
        shader.pCode = code;
        VkShaderModule module = VK_NULL_HANDLE;
        if (createShader(device, &shader, nullptr, &module) != VK_SUCCESS)
            return false;
        VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = layout;
        const VkResult status = createPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &result);
        destroyShader(device, module, nullptr);
        return status == VK_SUCCESS;
    }

    bool set_layout(const VkDescriptorType* types, uint32_t count, uint32_t pushBytes, VkDescriptorSetLayout& set,
                    VkPipelineLayout& layout)
    {
        VkDescriptorSetLayoutBinding bindings[kPrepareInputs + kPrepareOutputs] = {};
        for (uint32_t i = 0; i < count; ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = types[i];
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = types[i] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? &sampler : nullptr;
        }
        VkDescriptorSetLayoutCreateInfo info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        info.bindingCount = count;
        info.pBindings = bindings;
        if (createSetLayout(device, &info, nullptr, &set) != VK_SUCCESS)
            return false;
        VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
        VkPipelineLayoutCreateInfo pipelineLayout = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayout.setLayoutCount = 1;
        pipelineLayout.pSetLayouts = &set;
        pipelineLayout.pushConstantRangeCount = 1;
        pipelineLayout.pPushConstantRanges = &range;
        return createLayout(device, &pipelineLayout, nullptr, &layout) == VK_SUCCESS;
    }

    // `maxPushConstants`: VkPhysicalDeviceLimits::maxPushConstantsSize (the prepare pass needs 224).
    bool init(VkDevice d, PFN_vkGetDeviceProcAddr proc, uint32_t maxPushConstants)
    {
        device = d;
        if (maxPushConstants < sizeof(D4rFsrRrPrepareConstants))
            return false;
#define D4R_LOAD(member, name)                                                                                     \
    member = reinterpret_cast<decltype(member)>(proc(d, "vk" #name));                                              \
    if (!member)                                                                                                   \
    return false
        D4R_LOAD(createSampler, CreateSampler);
        D4R_LOAD(destroySampler, DestroySampler);
        D4R_LOAD(createSetLayout, CreateDescriptorSetLayout);
        D4R_LOAD(destroySetLayout, DestroyDescriptorSetLayout);
        D4R_LOAD(createLayout, CreatePipelineLayout);
        D4R_LOAD(destroyLayout, DestroyPipelineLayout);
        D4R_LOAD(createShader, CreateShaderModule);
        D4R_LOAD(destroyShader, DestroyShaderModule);
        D4R_LOAD(createPipelines, CreateComputePipelines);
        D4R_LOAD(destroyPipeline, DestroyPipeline);
        D4R_LOAD(bindPipeline, CmdBindPipeline);
        D4R_LOAD(push, CmdPushConstants);
        D4R_LOAD(pushDescriptors, CmdPushDescriptorSetKHR);
        D4R_LOAD(dispatch, CmdDispatch);
        D4R_LOAD(barrier, CmdPipelineBarrier);
        D4R_LOAD(createView, CreateImageView);
        D4R_LOAD(destroyView, DestroyImageView);
#undef D4R_LOAD
        VkSamplerCreateInfo nearest = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        nearest.magFilter = nearest.minFilter = VK_FILTER_NEAREST;
        nearest.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        nearest.addressModeU = nearest.addressModeV = nearest.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (createSampler(d, &nearest, nullptr, &sampler) != VK_SUCCESS)
            return false;
        VkDescriptorType prepare[kPrepareInputs + kPrepareOutputs];
        for (uint32_t i = 0; i < kPrepareInputs + kPrepareOutputs; ++i)
            prepare[i] = i < kPrepareInputs ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        VkDescriptorType compose[kComposeImages];
        for (VkDescriptorType& type : compose)
            type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        return set_layout(prepare, kPrepareInputs + kPrepareOutputs, sizeof(D4rFsrRrPrepareConstants), prepareSet,
                          prepareLayout) &&
               set_layout(compose, kComposeImages, 16, composeSet, composeLayout) &&
               pipeline(d4r_vk_fsr_rr_prepare, sizeof(d4r_vk_fsr_rr_prepare), prepareLayout, preparePipeline) &&
               pipeline(d4r_vk_fsr_rr_compose, sizeof(d4r_vk_fsr_rr_compose), composeLayout, composePipeline);
    }

    void destroy()
    {
        if (device == VK_NULL_HANDLE || destroyPipeline == nullptr)
            return;
        destroyPipeline(device, preparePipeline, nullptr);
        destroyPipeline(device, composePipeline, nullptr);
        destroyLayout(device, prepareLayout, nullptr);
        destroyLayout(device, composeLayout, nullptr);
        destroySetLayout(device, prepareSet, nullptr);
        destroySetLayout(device, composeSet, nullptr);
        destroySampler(device, sampler, nullptr);
        *this = D4rFsrRrPasses{};
    }

    // Everything the D3D12 side wrote before is visible to the pass, and the pass's writes to what
    // the D3D12 side records after.
    void full_barrier(VkCommandBuffer cmd)
    {
        VkMemoryBarrier memory = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        memory.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &memory, 0, nullptr,
                0, nullptr);
    }

    void record(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout, const D4rFsrRrImage* images,
                uint32_t count, uint32_t sampledCount, const void* constants, uint32_t constantBytes, uint32_t width,
                uint32_t height)
    {
        VkDescriptorImageInfo infos[kPrepareInputs + kPrepareOutputs] = {};
        VkWriteDescriptorSet writes[kPrepareInputs + kPrepareOutputs] = {};
        for (uint32_t i = 0; i < count; ++i)
        {
            infos[i].sampler = i < sampledCount ? sampler : VK_NULL_HANDLE;
            infos[i].imageView = images[i].view;
            infos[i].imageLayout = images[i].layout;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType =
                i < sampledCount ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &infos[i];
        }
        bindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        pushDescriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, count, writes);
        push(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constantBytes, constants);
        dispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
    }

    // images: the 10 sampled inputs, then the 8 storage outputs (fsr_rr_prepare.comp's bindings)
    void record_prepare(VkCommandBuffer cmd, const D4rFsrRrImage* images, const D4rFsrRrPrepareConstants& constants,
                        uint32_t width, uint32_t height)
    {
        full_barrier(cmd);
        record(cmd, preparePipeline, prepareLayout, images, kPrepareInputs + kPrepareOutputs, kPrepareInputs,
               &constants, sizeof(constants), width, height);
        full_barrier(cmd);
    }

    // images: denoised diffuse, denoised specular, diffuse albedo, specular albedo, residual, composed, noisy diffuse,
    // noisy specular. haloLimit: fsr_rr_compose.comp's limit (0: off)
    void record_compose(VkCommandBuffer cmd, const D4rFsrRrImage* images, uint32_t width, uint32_t height,
                        float haloLimit = 0.0f)
    {
        uint32_t limitBits;
        std::memcpy(&limitBits, &haloLimit, sizeof(limitBits));
        const uint32_t constants[4] = {width, height, limitBits, 0};
        full_barrier(cmd);
        record(cmd, composePipeline, composeLayout, images, kComposeImages, 0, constants, sizeof(constants), width,
               height);
        full_barrier(cmd);
    }
};
