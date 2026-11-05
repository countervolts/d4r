// Experimental native K and M networks recorded entirely on the Vulkan command list.
#pragma once
#include "vkhip.h"
#include "../runtime.h"
#include <memory>
namespace vkhip {
struct NetFiles {
    std::vector<uint8_t> hipnet;
    std::map<std::string, std::vector<uint8_t>> code;
    uint32_t mach = 0;  // the render GPU's EF_AMDGPU_MACH_* (gfx11Mach / gfx12Mach); 0: any, as long as all layers agree
    // the device enabled VK_KHR_pipeline_binary and VK_KHR_maintenance5: AMD's own drivers (Windows), where the
    // pipelines are injected through pipeline binaries (vkhip.h); RADV otherwise
    bool pipelineBinary = false;
};
namespace detail { struct Net; }
class KNetwork {
public:
    KNetwork();
    ~KNetwork();
    KNetwork(const KNetwork&) = delete;
    KNetwork& operator=(const KNetwork&) = delete;
    bool create(VkDevice device, PFN_vkGetDeviceProcAddr proc,
                const VkPhysicalDeviceMemoryProperties& memory, const NetFiles& files,
                uint32_t width, uint32_t height, uint64_t tokenBytes, uint64_t headBytes,
                std::string& error);
    VkBuffer tokens() const;
    VkBuffer head() const;
    uint32_t dispatchesPerFrame() const;
    void record(VkCommandBuffer command, bool reset);
    // D4R_VKHIP_STAMPS: GPU timestamps around every network dispatch, one "D4R_VKHIP_STAMPS" line per sampled frame
    // through `log` (busy = dispatch start to end, gap = previous end to this start: the barrier and launch).
    // Never waits: a frame whose slot is still in flight is not sampled.
    void enableStamps(void (*log)(const char*), float timestampPeriod);
    // Debug (D4R_VKHIP_DUMP_DIR): copies the tokens and the head of the first `frames` evaluations (at most 8) into
    // host memory and writes them as DIR/tokens-N.bin and DIR/head-N.bin when the network is destroyed (which then
    // waits for the device).
    void enableDump(const std::string& directory, uint32_t frames);
private:
    std::unique_ptr<detail::Net> impl;
};
// Presets M and L (D4RMHIP1, compile_m.py --hip): the ten kernels/m layers, identical for both presets, between
// MEngine's plane-layout input and expansion stages (ExternalNetwork{tokens(), head()}). width/height are the render
// dimensions; the windows alternate between two horizontal alignments, one per evaluation (hip_net.cpp's plan). The
// frame counter is never reset: a reset evaluation is still an evaluation.
class MNetwork {
public:
    MNetwork();
    ~MNetwork();
    MNetwork(const MNetwork&) = delete;
    MNetwork& operator=(const MNetwork&) = delete;
    bool create(VkDevice device, PFN_vkGetDeviceProcAddr proc,
                const VkPhysicalDeviceMemoryProperties& memory, const NetFiles& files,
                uint32_t renderWidth, uint32_t renderHeight, uint64_t bufferBytes, std::string& error);
    VkBuffer tokens() const;
    VkBuffer head() const;
    uint32_t dispatchesPerFrame() const;
    void record(VkCommandBuffer command);
    void enableStamps(void (*log)(const char*), float timestampPeriod);  // as KNetwork's
    void enableDump(const std::string& directory, uint32_t frames);     // as KNetwork's
private:
    std::unique_ptr<detail::Net> impl;
};
// Ray Reconstruction presets D and E (RDNA4): encoder 0 (after the engine's gather of its features, RModel's
// enc0_g.spv), the Swin stages enc1 - enc5, dec4 - dec1 and decoder 0 as
// kernels/rr/rrvk_swin.hip code objects (files.code "rr_<stage>": MODEL/hip/rr_<stage>.hsaco, kernels/rr/build_vkhip.sh), recorded by the
// engine in place of its shaders (REngine::setStageRecorder). They work in the engine's arena (by device address:
// REngine with arenaAddress) and on its images through raw descriptors, which needs VK_EXT_descriptor_buffer
// (vkGetDescriptorEXT) enabled on the device; the stages' weights are the model's (weights-<stage>m.bin, `weights`),
// rearranged for the kernels. Stages without a code object stay on the engine's shaders.
// Preset F on RDNA4: files.code["rr2_<stage>"], weights["<stage>"] replace network stages 0-14.
// Uses the live frame parameters and arena guide planes; input/output texture stages stay on Vulkan.
class RRStages final : public d4r::RStageRecorder {
public:
    RRStages();
    ~RRStages() override;
    RRStages(const RRStages&) = delete;
    RRStages& operator=(const RRStages&) = delete;
    bool create(VkPhysicalDevice physical, PFN_vkGetInstanceProcAddr instanceProc, VkInstance instance, VkDevice device,
                PFN_vkGetDeviceProcAddr proc, const VkPhysicalDeviceMemoryProperties& memory, const NetFiles& files,
                const std::map<std::string, std::vector<uint8_t>>& weights, d4r::REngine& engine, std::string& error);
    uint32_t stageCount() const;   // the stages that run on vkhip
    bool recordStage(VkCommandBuffer command, unsigned stage, uint32_t groupsX, uint32_t groupsY, uint32_t shift) override;
    bool records(unsigned stage) const override;
private:
    std::unique_ptr<detail::Net> impl;
    struct Stage;
    std::vector<Stage> stages;
    VkSampler linear = VK_NULL_HANDLE;
    PFN_vkDestroySampler destroySampler = nullptr;
};
} // namespace vkhip
