#include "game_e.h"
#include "game_internal.h"

namespace d4r {

struct GameUpscalerE::Impl {
    EModel& model;
    EEngine engine;
    uint32_t width, height, outputWidth, outputHeight, motionWidth, motionHeight;
    game_detail::CanonicalInputs fallback;
    bool started = false;
    Impl(EModel& m, VkPhysicalDevice p, VkDevice d, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh, bool hdr, bool lowRes,
         bool autoExposure)
        : model(m), engine(m, rw, rh, ow, oh, hdr, lowRes, autoExposure), width(rw), height(rh), outputWidth(ow), outputHeight(oh),
          motionWidth(lowRes ? rw : ow), motionHeight(lowRes ? rh : oh), fallback(p, d, rw, rh, motionWidth, motionHeight) {}
};
GameUpscalerE::GameUpscalerE(EModel& model, VkPhysicalDevice physical, VkDevice device, uint32_t outputWidth, uint32_t outputHeight,
                             uint32_t renderWidth, uint32_t renderHeight, bool hdr, bool lowResMotion, bool autoExposure)
    : impl(std::make_unique<Impl>(model, physical, device, outputWidth, outputHeight, renderWidth, renderHeight, hdr, lowResMotion,
                                  autoExposure)) {}
GameUpscalerE::~GameUpscalerE() = default;

void GameUpscalerE::record(VkCommandBuffer cb, const GameFrameE& frame) {
    auto& g = *impl;
    if (!gameColorFormat(frame.color.format) || !gameMotionFormat(frame.motion.format) ||
        !gameDepthFormat(frame.depth.format, frame.depth.aspect) || !gameOutputFormat(frame.output.format))
        throw std::runtime_error("unsupported game texture format");
    const bool directColor = frame.color.view != VK_NULL_HANDLE && !frame.color.x && !frame.color.y;
    const bool directMotion = frame.motion.view != VK_NULL_HANDLE && !frame.motion.x && !frame.motion.y;
    const bool directDepth = frame.depth.view != VK_NULL_HANDLE && !frame.depth.x && !frame.depth.y;
    const bool directOutput = game_detail::directOutput(frame.output, frame.color, frame.motion, frame.depth, false) &&
        !frame.output.x && !frame.output.y;
    if (!g.started) {
        if (!g.model.uploadRecorded()) g.model.recordUpload(cb);
        g.engine.recordInitialize(cb);
        g.started = true;
    }
    g.fallback.record(cb, frame.color, frame.motion, frame.depth, g.width, g.height, g.motionWidth, g.motionHeight,
                      directColor, directMotion, directDepth);
    EFrameSettings s;
    s.jitter[0] = frame.jitter[0]; s.jitter[1] = frame.jitter[1];
    s.motionScale[0] = frame.motionScale[0]; s.motionScale[1] = frame.motionScale[1];
    s.reset = frame.reset;
    s.preExposure = frame.preExposure; s.exposureScale = frame.exposureScale;
    FrameImages images{};
    images.color = directColor ? frame.color.view : g.fallback.color.view;
    images.motion = directMotion ? frame.motion.view : g.fallback.motion.view;
    images.depth = directDepth ? frame.depth.view : g.fallback.depth.view;
    images.colorLayout = directColor ? frame.color.layout : VK_IMAGE_LAYOUT_GENERAL;
    images.motionLayout = directMotion ? frame.motion.layout : VK_IMAGE_LAYOUT_GENERAL;
    images.depthLayout = directDepth ? frame.depth.layout : VK_IMAGE_LAYOUT_GENERAL;
    images.exposure = frame.exposure.view;
    images.exposureLayout = frame.exposure.layout;
    g.engine.setFrame(images, s, directOutput ? FrameOutput{frame.output.view, frame.output.format} : FrameOutput{});
    g.engine.recordFrame(cb);
    if (!directOutput) game_detail::copyOutput(cb, g.engine.outputImage(), frame.output, g.outputWidth, g.outputHeight);
}
} // namespace d4r
