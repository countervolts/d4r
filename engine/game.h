// Adapter between a game's own textures and the preset K engine. Everything is recorded into the
// game's command buffer: direct texture access (or format conversion), inference and output.
#pragma once
#include "runtime.h"

namespace d4r {
struct GameImage {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL; // at record time; must match the supplied sampled views or allow transfers
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t x = 0, y = 0; // origin of the used rectangle
    // Optional compatible sampled view (inputs) or RGBA16F/RGB10A2 storage view (output, GENERAL).
    // The caller guarantees usage/format compatibility and keeps image + view alive until adapter destruction.
    // Null selects the existing copy/conversion path. Sampled views use layout above.
    VkImageView view = VK_NULL_HANDLE;
};
struct GameFrame {
    GameImage color, depth, motion, output;
    // Optional 1x1 float exposure (texel 0 is read): the game's value replaces measurement. Exclusive with settings.autoExposure.
    GameImage exposure;
    FrameSettings settings; // render size, jitter, motion scale, reset
};
// Formats record() converts. Depth may be a depth-aspect D32 image or a one-channel colour image.
bool gameColorFormat(VkFormat format);
bool gameMotionFormat(VkFormat format);
bool gameDepthFormat(VkFormat format, VkImageAspectFlags aspect);
bool gameOutputFormat(VkFormat format);

class GameUpscaler {
public:
    // The model must outlive this object. Inputs up to maxRenderWidth x maxRenderHeight. With an external
    // network (ExternalNetwork) the eleven layers run outside the engine and the game's command list is split
    // around them (recordFront/recordBack); without one record() runs the whole network.
    GameUpscaler(Model& model, VkPhysicalDevice physical, VkDevice device, uint32_t outputWidth, uint32_t outputHeight,
                 uint32_t maxRenderWidth, uint32_t maxRenderHeight, const ExternalNetwork* external = nullptr);
    ~GameUpscaler(); // only once every recorded frame has completed on the GPU
    GameUpscaler(const GameUpscaler&) = delete;
    GameUpscaler& operator=(const GameUpscaler&) = delete;
    // Frames may be recorded while earlier ones are still queued; they must execute in recording order.
    void record(VkCommandBuffer command, const GameFrame& frame);
    // With an external network: the part before it (input copies, exposure, input stage; writes the tokens) and
    // the part after it (final store; reads the head). The network runs between the two on the GPU and the
    // caller orders them (the shim splits the game's command list there).
    void recordFront(VkCommandBuffer command, const GameFrame& frame);
    void recordBack(VkCommandBuffer command, const GameFrame& frame);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r
