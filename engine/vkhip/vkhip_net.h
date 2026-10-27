// Experimental native K network recorded entirely on the Vulkan command list.
#pragma once
#include "vkhip.h"
#include <memory>
namespace vkhip {
struct NetFiles {
    std::vector<uint8_t> hipnet;
    std::map<std::string, std::vector<uint8_t>> code;
    uint32_t mach = 0;  // the render GPU's EF_AMDGPU_MACH_* (gfx11Mach); 0: any, as long as all layers agree
};
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
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace vkhip
