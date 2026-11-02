// vkhip: run native AMDGPU (HIP code object) kernels as Vulkan compute pipelines on RADV, unchanged.
//
// A HIP kernel on gfx11 starts with its kernarg pointer in s[0:1] (two user SGPRs; on gfx1100/gfx1102 LLVM pads
// them to 16 user + system SGPRs for a wave32 init bug, which the kernel's own rsrc2 keeps), the workgroup ids after
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
//
// AMD's own drivers (the Windows driver, AMDVLK: PAL with LLPC) keep their shaders out of the pipeline cache data, so
// there step 2 and 4 go through VK_KHR_pipeline_binary instead (Device::pipelineBinary): the donor's binary is an
// AMDPAL ELF (.text with _amdgpu_cs_main, msgpack metadata in a note), rebuilt with the kernel's code and the kernel
// descriptor's register settings written into the metadata. Its user data map puts the push constant at s[0:1].
// The driver compiles the donor for the GPU it runs on, so its ELF machine authenticates the code objects' target.
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
    uint32_t lds = 0, scratch = 0, kernarg = 0, rsrc1 = 0, rsrc2 = 0, rsrc3 = 0, props = 0, preload = 0;
};

struct Global { uint64_t va = 0, size = 0; };

// A parsed AMDGPU code object: its .text, its writable data image and its kernels.
struct Module {
    std::vector<uint8_t> text;
    uint64_t textVa = 0;
    uint32_t mach = 0;                // EF_AMDGPU_MACH_* (gfx1100 0x41, gfx1101 0x46, gfx1102 0x47, gfx1103 0x44)
    uint64_t dataLo = 0, dataHi = 0;  // the .data/.bss range
    std::vector<uint8_t> dataInit;    // initial contents of [dataLo, dataHi)
    std::map<std::string, Kernel> kernels;
    std::map<std::string, Global> globals;
    // words: the span replaced (two s_mov_b32 with literals, then s_nop): 5 on gfx11, more with gfx12's s_sext/s_wait_alu
    struct Site { uint32_t word = 0, sdst = 0; uint64_t target = 0; uint32_t words = 5; };
    std::vector<Site> sites;           // getpc triples that address data
    uint32_t textGetpc = 0;            // getpc triples that address code (left alone)
    std::string error;

    bool parse(const std::vector<uint8_t>& elf);
    // .text with every data site redirected to dataAddress + (target - dataLo)
    std::vector<uint32_t> patchedText(uint64_t dataAddress) const;
};

// The ELF machine of code objects for a KFD gfx_target_version (110001 = gfx1101); 0 outside RDNA3.
uint32_t gfx11Mach(uint32_t gfxTargetVersion);
// Likewise for RDNA4 (120001 = gfx1201: 0x4e); 0 outside gfx1200/gfx1201. gfx12 code addresses its globals with an
// extra s_sext_i32_i16 after s_getpc_b64 and its workgroup ids arrive in ttmp7/ttmp9 (as RADV's own shaders' do).
uint32_t gfx12Mach(uint32_t gfxTargetVersion);

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
    // pipelineBinary: the device enabled VK_KHR_pipeline_binary (and VK_KHR_maintenance5); inject through it
    bool pipelineBinary = false;
    PFN_vkCreatePipelineBinariesKHR createBinaries = nullptr;
    PFN_vkGetPipelineBinaryDataKHR binaryData = nullptr;
    PFN_vkDestroyPipelineBinaryKHR destroyBinary = nullptr;
    bool load(VkDevice d, PFN_vkGetDeviceProcAddr proc, bool binaries = false);
};

struct Launch { uint32_t bx = 32, by = 1, bz = 1; };

// Creates the pipeline running `kernel` with workgroup size `launch`. `layout` must have exactly one push
// constant range of 8 bytes (compute stage) and no descriptor sets. `id` must differ between all pipelines
// created on one device (it keys the donor). Returns VK_NULL_HANDLE and sets `error` on failure. With
// vk.pipelineBinary the driver must be AMD's (an AMDPAL donor binary for the module's GPU), otherwise RADV.
VkPipeline createPipeline(const Device& vk, VkPipelineLayout layout, const Module& module, const Kernel& kernel,
                          Launch launch, uint64_t dataAddress, uint32_t id, std::string& error);

} // namespace vkhip
