#include "game.h"
#include "game_internal.h"
#include "game_profile.h"

namespace d4r {
namespace {
bool depthAspect(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D16_UNORM_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_X8_D24_UNORM_PACK32;
}
} // namespace

bool gameColorFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:   // R9G9B9E5_SHAREDEXP (RE Requiem's colour and output): sampled and blitted
        return true;
    default: return false;
    }
}
bool gameMotionFormat(VkFormat format) {
    // R16G16_SNORM (Enshrouded): sampled as floats in [-1, 1]; MV.Scale carries them to pixels as for float vectors.
    return format == VK_FORMAT_R16G16_SFLOAT || format == VK_FORMAT_R32G32_SFLOAT || format == VK_FORMAT_R16G16_SNORM ||
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
    game_detail::StageProfile profile;
    game_detail::CanonicalInputs fallback;
    bool started = false;
    uint32_t frames = 0;
    VkPhysicalDevice physical; VkDevice device;
    // the optional history dump (debug)
    std::string dumpDir; uint32_t dumpFrames = 0, dumpDone = 0; VkDeviceSize dumpBytes[3] = {};
    VkBuffer dumpBuffer = VK_NULL_HANDLE; VkDeviceMemory dumpMemory = VK_NULL_HANDLE;
    Impl(Model& m, VkPhysicalDevice p, VkDevice d, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh, const ExternalNetwork* external)
        : model(m), shape(makeKShape(ow, oh)), engine(m, shape, external), maxWidth(rw), maxHeight(rh),
          profile(p, d), fallback(p, d, rw, rh, ow, oh), physical(p), device(d) {}
    ~Impl();
    void dumpHistories(VkCommandBuffer cb);
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
GameUpscaler::Impl::~Impl() {
    if (!dumpBuffer) return;
    // the destructor's contract: every recorded frame has completed on the GPU
    void* mapped = nullptr;
    if (vkMapMemory(device, dumpMemory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
        const char* names[3] = {"color", "luma", "feature"};
        VkDeviceSize at = 0;
        for (uint32_t f = 0; f < dumpDone; ++f)
            for (int k = 0; k < 3; ++k) {
                const std::string path = dumpDir + "/hist-" + names[k] + "-" + std::to_string(f) + ".bin";
                if (FILE* out = std::fopen(path.c_str(), "wb")) {
                    std::fwrite(static_cast<const char*>(mapped) + at, 1, dumpBytes[k], out);
                    std::fclose(out);
                }
                at += dumpBytes[k];
            }
    }
    vkDestroyBuffer(device, dumpBuffer, nullptr);
    vkFreeMemory(device, dumpMemory, nullptr);
}
void GameUpscaler::enableHistoryDump(const std::string& directory, uint32_t frames) {
    auto& g = *impl;
    if (g.dumpBuffer || directory.empty() || !frames) return;
    frames = std::min(frames, 8u);
    const VkDeviceSize hw = (g.shape.outputWidth + 15) / 16 * 16, hh = (g.shape.outputHeight + 15) / 16 * 16;
    g.dumpBytes[0] = hw * hh * 8; g.dumpBytes[1] = hw * hh * 4 * 2; g.dumpBytes[2] = VkDeviceSize(g.shape.layers[0].w) * g.shape.layers[0].h * 8;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = frames * (g.dumpBytes[0] + g.dumpBytes[1] + g.dumpBytes[2]); info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(g.device, &info, nullptr, &g.dumpBuffer) != VK_SUCCESS) { g.dumpBuffer = VK_NULL_HANDLE; return; }
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(g.device, g.dumpBuffer, &r);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g.physical, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = ~0u;
    for (uint32_t t = 0; t < mp.memoryTypeCount && type == ~0u; ++t)
        if ((r.memoryTypeBits & (1u << t)) && (mp.memoryTypes[t].propertyFlags & want) == want) type = t;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = r.size; mai.memoryTypeIndex = type;
    if (type == ~0u || vkAllocateMemory(g.device, &mai, nullptr, &g.dumpMemory) != VK_SUCCESS ||
        vkBindBufferMemory(g.device, g.dumpBuffer, g.dumpMemory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(g.device, g.dumpBuffer, nullptr); g.dumpBuffer = VK_NULL_HANDLE;
        if (g.dumpMemory) vkFreeMemory(g.device, g.dumpMemory, nullptr);
        g.dumpMemory = VK_NULL_HANDLE;
        return;
    }
    g.dumpDir = directory; g.dumpFrames = frames;
}
void GameUpscaler::Impl::dumpHistories(VkCommandBuffer cb) {
    if (!dumpBuffer || dumpDone >= dumpFrames) return;
    const HistoryImages h = engine.historyImages();
    const VkImage images[3] = {h.color, h.luma, h.feature};
    const uint32_t hw = (shape.outputWidth + 15) / 16 * 16, hh = (shape.outputHeight + 15) / 16 * 16;
    const VkExtent3D extents[3] = {{hw, hh, 1}, {2 * hw, 2 * hh, 1}, {uint32_t(shape.layers[0].w), uint32_t(shape.layers[0].h), 1}};
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkDeviceSize at = dumpDone * (dumpBytes[0] + dumpBytes[1] + dumpBytes[2]);
    for (int k = 0; k < 3; ++k) {
        VkBufferImageCopy copy{};
        copy.bufferOffset = at; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = extents[k];
        vkCmdCopyImageToBuffer(cb, images[k], VK_IMAGE_LAYOUT_GENERAL, dumpBuffer, 1, &copy);
        at += dumpBytes[k];
    }
    VkMemoryBarrier back{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &back, 0, nullptr, 0, nullptr);
    ++dumpDone;
}

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
    // D16/D24 must be sampled as normalized depth. The legacy D32 fallback
    // copies raw depth bits into R32F and cannot reinterpret packed UNORM.
    if (!directDepth && (frame.depth.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) &&
        frame.depth.format != VK_FORMAT_D32_SFLOAT && frame.depth.format != VK_FORMAT_D32_SFLOAT_S8_UINT)
        throw std::runtime_error("packed depth requires a sampleable depth view");
    const bool directOutput = game_detail::directOutput(frame.output, frame.color, frame.motion, frame.depth, g.model.hasDirectOutput(frame.output.format));
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
    // The built-in network keeps its evaluation-counter schedule. The inline native accuracy set advertises
    // its reference reset schedule separately; its loader owns the external network's window counter.
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
    if (game_detail::directOutput(frame.output, frame.color, frame.motion, frame.depth, g.model.hasDirectOutput(frame.output.format))) return;
    game_detail::copyOutput(cb, g.engine.outputImage(), frame.output, g.shape.outputWidth, g.shape.outputHeight);
}
void GameUpscaler::record(VkCommandBuffer cb, const GameFrame& frame) {
    auto& g = *impl;
    g.profile.begin(cb, frame.profile, frame.settings.renderWidth, frame.settings.renderHeight,
                    g.shape.outputWidth, g.shape.outputHeight,
                    {"prepare", "setup", "exposure", "input", "enc0", "enc1", "enc2", "enc3", "enc4",
                     "dec5", "dec4", "dec3", "dec2", "dec1", "dec0", "reconstruction", "sync", "output"});
    g.inputs(cb, frame);
    g.engine.recordFrame(cb, g.profile.callback());
    g.result(cb, frame);
    g.profile.end(cb);
}
void GameUpscaler::recordFront(VkCommandBuffer cb, const GameFrame& frame) {
    auto& g = *impl;
    g.profile.begin(cb, frame.profile, frame.settings.renderWidth, frame.settings.renderHeight,
                    g.shape.outputWidth, g.shape.outputHeight,
                    {"prepare", "setup", "exposure", "input", "external_network_wait", "reconstruction", "output"});
    g.inputs(cb, frame);
    g.profile.mark(cb, 1);
    g.engine.recordFront(cb, g.profile.pool(), g.profile.first(2));
}
void GameUpscaler::recordBack(VkCommandBuffer cb, const GameFrame& frame) {
    auto& g = *impl;
    g.engine.recordBack(cb, g.profile.pool(), g.profile.first(5));
    g.dumpHistories(cb);
    g.result(cb, frame);
    g.profile.end(cb);
}
} // namespace d4r
