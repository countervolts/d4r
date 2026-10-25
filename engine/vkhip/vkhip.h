// vkhip: run native AMDGPU (HIP code object) kernels as Vulkan compute pipelines on RADV, unchanged.
//
// A HIP kernel on gfx11 starts with its kernarg pointer in s[0:1] (two user SGPRs), the workgroup ids after
// them, packed thread ids in v0, and reaches its globals PC-relative (s_getpc_b64 + s_add_u32 + s_addc_u32).
// A RADV compute shader whose only interface is one 64-bit push constant gets exactly that SGPR layout
// (gfx11 compute declares no ring offsets; the push constant is inlined into the first user SGPRs). So:
//   1. compile a donor pipeline with the kernel's workgroup size and LDS size,
//   2. read it back through the pipeline cache (core Vulkan; RADV stores the shader as radv_shader_binary_legacy:
//      config + info + code, no checksum),
//   3. replace its code with the kernel's .text (behind one s_branch to the entry) and its register config
//      (rsrc1/rsrc2/rsrc3, VGPRs, scratch) with the kernel descriptor's,
//   4. create the pipeline again from the patched cache data.
// The globals live in a Vulkan buffer: each s_getpc/s_add/s_addc triple that targets .data/.bss becomes two
// s_mov_b32 of the buffer address plus s_nop (same size, same instruction count). Every other instruction is
// the code object's own, so the arithmetic is bit-identical to the HIP launch.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace vkhip {

struct Kernel {
    std::string name;
    uint64_t entry = 0;  // virtual address
    uint32_t lds = 0, scratch = 0, kernarg = 0, rsrc1 = 0, rsrc2 = 0, rsrc3 = 0, props = 0;
};

struct Global { uint64_t va = 0, size = 0; };

// A parsed AMDGPU code object: its .text, its writable data image and its kernels.
struct Module {
    std::vector<uint8_t> text;
    uint64_t textVa = 0;
    uint64_t dataLo = 0, dataHi = 0;  // the .data/.bss range
    std::vector<uint8_t> dataInit;    // initial contents of [dataLo, dataHi)
    std::map<std::string, Kernel> kernels;
    std::map<std::string, Global> globals;
    struct Site { uint32_t word = 0, sdst = 0; uint64_t target = 0; };
    std::vector<Site> sites;           // getpc triples that address data
    uint32_t textGetpc = 0;            // getpc triples that address code (left alone)
    std::string error;

    bool parse(const std::vector<uint8_t>& elf);
    // .text with every data site redirected to dataAddress + (target - dataLo)
    std::vector<uint32_t> patchedText(uint64_t dataAddress) const;
};

// Vulkan entry points (device-level) the injector needs.
struct Device {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkCreatePipelineCache createCache = nullptr;
    PFN_vkDestroyPipelineCache destroyCache = nullptr;
    PFN_vkGetPipelineCacheData cacheData = nullptr;
    PFN_vkCreateComputePipelines createCompute = nullptr;
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkCreateShaderModule createModule = nullptr;
    PFN_vkDestroyShaderModule destroyModule = nullptr;
    bool load(VkDevice d, PFN_vkGetDeviceProcAddr proc);
};

struct Launch { uint32_t bx = 32, by = 1, bz = 1; };

// Creates the pipeline running `kernel` with workgroup size `launch`. `layout` must have exactly one push
// constant range of 8 bytes (compute stage) and no descriptor sets. `id` must differ between all pipelines
// created on one device (it keys the donor). Returns VK_NULL_HANDLE and sets `error` on failure.
VkPipeline createPipeline(const Device& vk, VkPipelineLayout layout, const Module& module, const Kernel& kernel,
                          Launch launch, uint64_t dataAddress, uint32_t id, std::string& error);

} // namespace vkhip
