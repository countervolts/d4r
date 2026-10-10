// Adapter between a game's own textures and the preset M engine: input conversion, the frame and the copy into
// the game's output texture, all recorded into the game's command buffer. Separate from GameUpscaler (preset K).
#pragma once
#include "game.h"

namespace d4r {
struct GameFrameM {
    GameImage color, depth, motion, output;     // as for GameUpscaler; views are not used (inputs are always copied)
    float jitter[2] = {0, 0};
    float motionToRender[2] = {1, 1};           // (motion texel + motionOffset) * this = render pixels
    float motionOffset[2] = {0, 0};             // jittered motion vectors: jitter - previous jitter, motion texel units
    bool reset = false;
    bool depthInverted = true;
};
class GameUpscalerM {
public:
    // The model must outlive this object. The render size is fixed and at most half the output size.
    GameUpscalerM(MModel& model, VkPhysicalDevice physical, VkDevice device, uint32_t outputWidth, uint32_t outputHeight,
                  uint32_t renderWidth, uint32_t renderHeight, const ExternalNetwork* external = nullptr);
    ~GameUpscalerM(); // only once every recorded frame has completed on the GPU
    GameUpscalerM(const GameUpscalerM&) = delete;
    GameUpscalerM& operator=(const GameUpscalerM&) = delete;
    void record(VkCommandBuffer command, const GameFrameM& frame);
    // With an external network (ExternalNetwork): the part before it (input copies, exposure, input stage) and the
    // part after it (expansion, reconstruction, downsample, the copy into the game's output). The network must run
    // between the two on the GPU; the caller orders them (the shim splits the game's command list there).
    void recordFront(VkCommandBuffer command, const GameFrameM& frame);
    void recordBack(VkCommandBuffer command, const GameFrameM& frame);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r
