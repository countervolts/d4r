// Adapter between a game's own textures and the Ray Reconstruction engine (REngine): input conversion where a
// texture cannot be sampled directly, the frame and the copy into the game's output texture, all recorded into the
// game's command buffer.
#pragma once
#include "game.h"

namespace d4r {
struct GameFrameR {
    GameImage color, depth, motion, normals, roughness, diffuseAlbedo, specularAlbedo, output;
    // optional (image null: the game passes none): DLSSD.SpecularHitDistance, DLSS.DisocclusionMask
    GameImage specularHitDistance, disocclusionMask;
    // optional too: DLSSD.ScreenSpaceSubsurfaceScatteringGuide, DLSSD.ColorBeforeParticles
    GameImage subsurfaceGuide, colorBeforeParticles;
    RFrameSettings settings;
};
class GameDenoiser {
public:
    // The model must outlive this object. The render size is fixed.
    GameDenoiser(RModel& model, VkPhysicalDevice physical, VkDevice device, uint32_t outputWidth, uint32_t outputHeight,
                 uint32_t renderWidth, uint32_t renderHeight);
    ~GameDenoiser(); // only once every recorded frame has completed on the GPU
    GameDenoiser(const GameDenoiser&) = delete;
    GameDenoiser& operator=(const GameDenoiser&) = delete;
    void record(VkCommandBuffer command, const GameFrameR& frame);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r
