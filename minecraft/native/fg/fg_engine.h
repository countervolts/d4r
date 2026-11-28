// d4r native Lite frame generation engine: replays a recorded DLSS-G launch script (tools/mkscript.py) through the
// nvxrt Vulkan layer (VK_NVX_binary_import emulation). Needs the layer enabled on the instance (VK_INSTANCE_LAYERS).
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

struct FgEngine;
struct FgVk {
    VkInstance instance; VkPhysicalDevice phys; VkDevice device; uint32_t queueFamily; VkQueue queue;
    PFN_vkGetInstanceProcAddr gipa;
    uint32_t computeFamily = ~0u;   // != queueFamily: engine images/buffers are CONCURRENT between both families (async compute)
};
// scriptDir holds script.txt (+ weights.bin); cache is the nvxrt kernel cache (module PTX by hash).
FgEngine *fg_create(const FgVk &vk, const std::string &scriptDir, const std::string &cache, std::string &err);
void fg_destroy(FgEngine *e);
int fg_generated_count(const FgEngine *e);
// DLL-side images the caller fills before fg_record: label is backbuffer, hudless, depth or mvec (GENERAL layout)
VkImage fg_input_image(const FgEngine *e, const char *label);
void fg_input_size(const FgEngine *e, const char *label, uint32_t &w, uint32_t &h, VkFormat &fmt);
// generated frame k (0-based) of the given variant (real frame counter & 1) once fg_record has run
VkImage fg_output_image(const FgEngine *e, int variant, int k);
// records the ops of one real frame (all evaluates) into cmd. clipToPrevClip: 16 floats as NGX passes them (or null)
void fg_record(FgEngine *e, VkCommandBuffer cmd, int variant, const float *clipToPrevClip);
// optional GPU timestamps around fg_record: returns microseconds of the last completed pair, 0 if none
