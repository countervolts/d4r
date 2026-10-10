#include "game_m.h"
#include "game_internal.h"

namespace d4r {

struct GameUpscalerM::Impl {
    MModel& model;
    MEngine engine;
    uint32_t width, height, outputWidth, outputHeight;
    game_detail::CanonicalInputs fallback;
    bool started = false;
    Impl(MModel& m, VkPhysicalDevice p, VkDevice d, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh, const ExternalNetwork* external)
        : model(m), engine(m, rw, rh, ow, oh, external), width(rw), height(rh), outputWidth(ow), outputHeight(oh),
          fallback(p, d, rw, rh, rw, rh) {}
    void inputs(VkCommandBuffer cb, const GameFrameM& frame);
    void result(VkCommandBuffer cb, const GameFrameM& frame);
};
GameUpscalerM::GameUpscalerM(MModel& model, VkPhysicalDevice physical, VkDevice device, uint32_t outputWidth, uint32_t outputHeight,
                             uint32_t renderWidth, uint32_t renderHeight, const ExternalNetwork* external)
    : impl(std::make_unique<Impl>(model, physical, device, outputWidth, outputHeight, renderWidth, renderHeight, external)) {}
GameUpscalerM::~GameUpscalerM() = default;

// the game's inputs into the engine's images, and the frame's settings
void GameUpscalerM::Impl::inputs(VkCommandBuffer cb, const GameFrameM& frame) {
    auto& g = *this;
    const uint32_t w = g.width, h = g.height;
    if (!gameColorFormat(frame.color.format) || !gameMotionFormat(frame.motion.format) ||
        !gameDepthFormat(frame.depth.format, frame.depth.aspect) || !gameOutputFormat(frame.output.format))
        throw std::runtime_error("unsupported game texture format");
    // Nonzero input origins retain conversion copies until M's shader parameter blocks carry origins.
    const bool directColor = frame.color.view != VK_NULL_HANDLE && !frame.color.x && !frame.color.y;
    const bool directMotion = frame.motion.view != VK_NULL_HANDLE && !frame.motion.x && !frame.motion.y;
    const bool directDepth = frame.depth.view != VK_NULL_HANDLE && !frame.depth.x && !frame.depth.y;
    const bool directOutput = game_detail::directOutput(frame.output, frame.color, frame.motion, frame.depth, g.model.hasRgb10Output()) &&
        !frame.output.x && !frame.output.y;
    if (!g.started) {
        if (!g.model.uploadRecorded()) g.model.recordUpload(cb);
        g.engine.recordInitialize(cb);
        g.started = true;
    }
    g.fallback.record(cb, frame.color, frame.motion, frame.depth, w, h, w, h, directColor, directMotion, directDepth);
    MFrameSettings s;
    s.jitter[0] = frame.jitter[0]; s.jitter[1] = frame.jitter[1];
    s.motionScale[0] = frame.motionToRender[0]; s.motionScale[1] = frame.motionToRender[1];
    s.motionOffset[0] = frame.motionOffset[0]; s.motionOffset[1] = frame.motionOffset[1];
    s.reset = frame.reset; s.depthInverted = frame.depthInverted;
    FrameImages images{};
    images.color = directColor ? frame.color.view : g.fallback.color.view;
    images.motion = directMotion ? frame.motion.view : g.fallback.motion.view;
    images.depth = directDepth ? frame.depth.view : g.fallback.depth.view;
    images.colorLayout = directColor ? frame.color.layout : VK_IMAGE_LAYOUT_GENERAL;
    images.motionLayout = directMotion ? frame.motion.layout : VK_IMAGE_LAYOUT_GENERAL;
    images.depthLayout = directDepth ? frame.depth.layout : VK_IMAGE_LAYOUT_GENERAL;
    g.engine.setFrame(images, s, directOutput ? FrameOutput{frame.output.view, frame.output.format} : FrameOutput{});
}
void GameUpscalerM::Impl::result(VkCommandBuffer cb, const GameFrameM& frame) {
    auto& g = *this;
    if (game_detail::directOutput(frame.output, frame.color, frame.motion, frame.depth, g.model.hasRgb10Output()) && !frame.output.x && !frame.output.y) return;
    game_detail::copyOutput(cb, g.engine.outputImage(), frame.output, g.outputWidth, g.outputHeight);
}
void GameUpscalerM::record(VkCommandBuffer cb, const GameFrameM& frame) {
    impl->inputs(cb, frame);
    impl->engine.recordFrame(cb); // begins and ends with barriers against all other work
    impl->result(cb, frame);
}
void GameUpscalerM::recordFront(VkCommandBuffer cb, const GameFrameM& frame) {
    impl->inputs(cb, frame);
    impl->engine.recordFront(cb);
}
void GameUpscalerM::recordBack(VkCommandBuffer cb, const GameFrameM& frame) {
    impl->engine.recordBack(cb);
    impl->result(cb, frame);
}
} // namespace d4r
