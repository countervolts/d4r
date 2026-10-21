#include "game_e.h"
#include "game_internal.h"

#include <algorithm>
#include <array>
#include <vector>

namespace d4r {

struct GameUpscalerE::Impl {
    EModel& model;
    EEngine engine;
    uint32_t width, height, outputWidth, outputHeight, motionWidth, motionHeight;
    game_detail::CanonicalInputs fallback;
    bool started = false;
    // The engine keeps one descriptor set per distinct combination of input and output views (runtime_e.inc, at most
    // 16) and gives up on the feature beyond that. Games that rotate or keep reallocating these textures (Unreal's
    // transient textures: Townfall reached the limit after ~6800 frames) would then fall back to the CUDA backend for
    // the rest of the feature. As in game_rr.cpp: an input (colour, motion, depth, output) that shows more than
    // kRotating distinct views goes through the adapter's own plane from then on, and once kCombinations
    // combinations exist, all four do, which leaves a single stable combination.
    static constexpr size_t kRotating = 4, kCombinations = 12;
    std::array<std::vector<VkImageView>, 4> seenViews;
    std::array<bool, 4> needsCopy = {};
    std::vector<std::array<VkImageView, 5>> combinations;   // colour, motion, depth, exposure, output as bound
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
    const GameImage* const tracked[4] = {&frame.color, &frame.motion, &frame.depth, &frame.output};
    for (size_t i = 0; i < 4; ++i) {
        const VkImageView view = tracked[i]->view;
        auto& seen = g.seenViews[i];
        if (view != VK_NULL_HANDLE && std::find(seen.begin(), seen.end(), view) == seen.end()) {
            seen.push_back(view);
            if (seen.size() > Impl::kRotating) g.needsCopy[i] = true;
        }
    }
    bool directColor, directMotion, directDepth, directOutput;
    auto decide = [&] {
        directColor = !g.needsCopy[0] && frame.color.view != VK_NULL_HANDLE && !frame.color.x && !frame.color.y;
        directMotion = !g.needsCopy[1] && frame.motion.view != VK_NULL_HANDLE && !frame.motion.x && !frame.motion.y;
        directDepth = !g.needsCopy[2] && frame.depth.view != VK_NULL_HANDLE && !frame.depth.x && !frame.depth.y;
        directOutput = !g.needsCopy[3] && game_detail::directOutput(frame.output, frame.color, frame.motion, frame.depth, false) &&
            !frame.output.x && !frame.output.y;
        // the adapter's planes stand for every copied input; the engine's own image (null here) for a copied output
        return std::array<VkImageView, 5>{directColor ? frame.color.view : VK_NULL_HANDLE,
                                          directMotion ? frame.motion.view : VK_NULL_HANDLE,
                                          directDepth ? frame.depth.view : VK_NULL_HANDLE, frame.exposure.view,
                                          directOutput ? frame.output.view : VK_NULL_HANDLE};
    };
    auto combination = decide();
    if (std::find(g.combinations.begin(), g.combinations.end(), combination) == g.combinations.end()) {
        if (g.combinations.size() >= Impl::kCombinations) {
            g.needsCopy = {true, true, true, true};
            combination = decide();
        }
        if (std::find(g.combinations.begin(), g.combinations.end(), combination) == g.combinations.end())
            g.combinations.push_back(combination);
    }
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
