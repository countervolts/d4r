// Adapter between a game's own textures and the preset E/F engine (DLSS 3's CNN, engine/runtime_e.inc): input
// conversion, the frame and the copy into the game's output texture, all recorded into the game's command buffer.
#pragma once
#include "game.h"

namespace d4r {
struct GameFrameE {
    GameImage color, depth, motion, output;     // as for GameUpscaler; compatible views at origin zero may be borrowed
    GameImage exposure;                         // HDR without autoExposure: the game's 1x1 float exposure (a sampled view), else 1
    float jitter[2] = {0, 0};
    float motionScale[2] = {1, 1};              // NGX's MV scale
    float preExposure = 1, exposureScale = 1;   // the game's (DLSS.Pre.Exposure, DLSS.Exposure.Scale)
    bool reset = false;
};
class GameUpscalerE {
public:
    // The model must outlive this object. Render size fixed for its lifetime (a new size needs a new adapter).
    // lowResMotion: motion at the render size (NGX's MVLowRes), otherwise at the output size. autoExposure: NGX's
    // AutoExposure (HDR), the exposure measured from the colour (EModel::hasAutoExposure).
    GameUpscalerE(EModel& model, VkPhysicalDevice physical, VkDevice device, uint32_t outputWidth, uint32_t outputHeight,
                  uint32_t renderWidth, uint32_t renderHeight, bool hdr, bool lowResMotion, bool autoExposure = false);
    ~GameUpscalerE(); // only once every recorded frame has completed on the GPU
    GameUpscalerE(const GameUpscalerE&) = delete;
    GameUpscalerE& operator=(const GameUpscalerE&) = delete;
    void record(VkCommandBuffer command, const GameFrameE& frame);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r
