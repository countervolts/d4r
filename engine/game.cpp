#include "game.h"
#include "game_internal.h"

namespace d4r {
namespace {
bool depthAspect(VkFormat format) { return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT; }
} // namespace

bool gameColorFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM:
        return true;
    default: return false;
    }
}
bool gameMotionFormat(VkFormat format) {
    return format == VK_FORMAT_R16G16_SFLOAT || format == VK_FORMAT_R32G32_SFLOAT ||
           format == VK_FORMAT_R16G16B16A16_SFLOAT || format == VK_FORMAT_R32G32B32A32_SFLOAT;
}
bool gameDepthFormat(VkFormat format, VkImageAspectFlags aspect) {
    if (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) return depthAspect(format);
    return format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_R16_SFLOAT || format == VK_FORMAT_R16_UNORM;
}
bool gameOutputFormat(VkFormat format) { return gameColorFormat(format); }

struct GameUpscaler::Impl {
    Model& model;
    Shape shape;
    Engine engine;
    uint32_t maxWidth, maxHeight;
    game_detail::CanonicalInputs fallback;
    bool started = false;
    uint32_t frames = 0;
    Impl(Model& m, VkPhysicalDevice p, VkDevice d, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh, const ExternalNetwork* external)
        : model(m), shape(makeKShape(ow, oh)), engine(m, shape, external), maxWidth(rw), maxHeight(rh),
          fallback(p, d, rw, rh, ow, oh) {}
    void inputs(VkCommandBuffer cb, const GameFrame& frame);
    void result(VkCommandBuffer cb, const GameFrame& frame);
};
GameUpscaler::GameUpscaler(Model& model, VkPhysicalDevice physical, VkDevice device, uint32_t outputWidth, uint32_t outputHeight,
                           uint32_t maxRenderWidth, uint32_t maxRenderHeight, const ExternalNetwork* external)
    : impl(std::make_unique<Impl>(model, physical, device, outputWidth, outputHeight, maxRenderWidth, maxRenderHeight, external)) {
    if (!maxRenderWidth || !maxRenderHeight || maxRenderWidth > outputWidth || maxRenderHeight > outputHeight)
        throw std::runtime_error("render dimensions must be positive and no larger than the output");
}
GameUpscaler::~GameUpscaler() = default;

// Everything before the network: input conversion, exposure setup and the input stage (and a reset's clears).
void GameUpscaler::Impl::inputs(VkCommandBuffer cb, const GameFrame& frame) {
    auto& g = *this;
    const uint32_t w = frame.settings.renderWidth, h = frame.settings.renderHeight;
    if (!w || !h || w > g.maxWidth || h > g.maxHeight) throw std::runtime_error("render size exceeds the adapter's inputs");
    if (!gameColorFormat(frame.color.format) || !gameMotionFormat(frame.motion.format) ||
        !gameDepthFormat(frame.depth.format, frame.depth.aspect) || !gameOutputFormat(frame.output.format))
        throw std::runtime_error("unsupported game texture format");
    const bool gameExposure = frame.exposure.view != VK_NULL_HANDLE;
    if (gameExposure && frame.settings.autoExposure) throw std::runtime_error("game exposure and automatic exposure are exclusive");
    // Validates the settings before anything is recorded.
    FrameParams parameters = makeKFrameParams(g.shape, frame.settings);
    const bool directColor = frame.color.view != VK_NULL_HANDLE;
    const bool directMotion = frame.motion.view != VK_NULL_HANDLE;
    const bool directDepth = frame.depth.view != VK_NULL_HANDLE;
    const bool directOutput = game_detail::directOutput(frame.output, frame.color, frame.motion, frame.depth, g.model.hasRgb10Output());
    auto origin = [&](const GameImage& image, bool direct, int32_t* inOrigin, int32_t* outOrigin, int32_t* outMax) {
        if (!direct) return;
        if ((image.x || image.y) && !g.model.hasDirectOrigins())
            throw std::runtime_error("recompile the model for direct texture rectangles");
        if (uint64_t(image.x) + w > INT32_MAX || uint64_t(image.y) + h > INT32_MAX)
            throw std::runtime_error("texture rectangle exceeds shader coordinate range");
        inOrigin[0] = outOrigin[0] = int32_t(image.x); inOrigin[1] = outOrigin[1] = int32_t(image.y);
        outMax[0] += int32_t(image.x); outMax[1] += int32_t(image.y);
    };
    origin(frame.color, directColor, parameters.input.colOrigin, parameters.output.colOrigin, parameters.output.colMax);
    origin(frame.motion, directMotion, parameters.input.mvOrigin, parameters.output.mvOrigin, parameters.output.mvMax);
    origin(frame.depth, directDepth, parameters.input.depOrigin, parameters.output.depOrigin, parameters.output.depMax);
    if (directOutput) {
        if (uint64_t(frame.output.x) + g.shape.outputWidth > INT32_MAX || uint64_t(frame.output.y) + g.shape.outputHeight > INT32_MAX)
            throw std::runtime_error("output rectangle exceeds shader coordinate range");
        parameters.output.outOrigin[0] = int32_t(frame.output.x); parameters.output.outOrigin[1] = int32_t(frame.output.y);
    }
    if (!g.started) {
        if (!g.model.uploadRecorded()) g.model.recordUpload(cb);
        g.engine.recordInitialize(cb);
        g.started = true;
    }
    const uint32_t mw = frame.settings.displayMotion ? g.shape.outputWidth : w, mh = frame.settings.displayMotion ? g.shape.outputHeight : h;
    g.fallback.record(cb, frame.color, frame.motion, frame.depth, w, h, mw, mh, directColor, directMotion, directDepth);
    ExposureControl exposure;
    exposure.size[0] = int32_t(w); exposure.size[1] = int32_t(h);
    if (directColor) { exposure.origin[0] = int32_t(frame.color.x); exposure.origin[1] = int32_t(frame.color.y); }
    exposure.game = gameExposure;
    exposure.measure = !gameExposure && (g.frames % 2 == 0 || frame.settings.reset);
    exposure.reset = frame.settings.reset;
    exposure.exposureScale = frame.settings.exposureScale; exposure.preExposure = frame.settings.preExposure;
    // The window schedule follows the evaluation counter, which advances every frame and is not restarted by a
    // reset; the external network's phase counter must advance in lockstep, so this is the only place it moves.
    auto windows = makeKWindows(g.shape, g.frames++);
    FrameImages images{directColor ? frame.color.view : g.fallback.color.view, directMotion ? frame.motion.view : g.fallback.motion.view,
        directDepth ? frame.depth.view : g.fallback.depth.view, directColor ? frame.color.layout : VK_IMAGE_LAYOUT_GENERAL,
        directMotion ? frame.motion.layout : VK_IMAGE_LAYOUT_GENERAL, directDepth ? frame.depth.layout : VK_IMAGE_LAYOUT_GENERAL,
        gameExposure ? frame.exposure.view : VK_NULL_HANDLE, frame.exposure.layout};
    g.engine.setFrame(images, parameters.input, parameters.output, &windows,
                      (frame.settings.autoExposure || gameExposure) ? &exposure : nullptr,
                      directOutput ? FrameOutput{frame.output.view, frame.output.format} : FrameOutput{});
}
// Everything after the network: the copy into the game's output texture when the store was not direct.
void GameUpscaler::Impl::result(VkCommandBuffer cb, const GameFrame& frame) {
    auto& g = *this;
    if (game_detail::directOutput(frame.output, frame.color, frame.motion, frame.depth, g.model.hasRgb10Output())) return;
    game_detail::copyOutput(cb, g.engine.outputImage(), frame.output, g.shape.outputWidth, g.shape.outputHeight);
}
void GameUpscaler::record(VkCommandBuffer cb, const GameFrame& frame) {
    impl->inputs(cb, frame);
    impl->engine.recordFrame(cb); // begins and ends with barriers against all other work
    impl->result(cb, frame);
}
void GameUpscaler::recordFront(VkCommandBuffer cb, const GameFrame& frame) {
    impl->inputs(cb, frame);
    impl->engine.recordFront(cb);
}
void GameUpscaler::recordBack(VkCommandBuffer cb, const GameFrame& frame) {
    impl->engine.recordBack(cb);
    impl->result(cb, frame);
}
} // namespace d4r
