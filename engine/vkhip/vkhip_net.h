// Experimental native K network recorded entirely on the Vulkan command list.
#pragma once
#include "vkhip.h"
#include <memory>
namespace vkhip {
struct NetFiles {
    std::vector<uint8_t> hipnet;
    std::map<std::string, std::vector<uint8_t>> code;
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
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace vkhip
