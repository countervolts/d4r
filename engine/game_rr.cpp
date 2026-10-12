#include "game_rr.h"
#include "game_internal.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace d4r {

struct GameDenoiser::Impl {
    RModel& model;
    REngine engine;
    uint32_t width, height, outputWidth, outputHeight;
    game_detail::CanonicalInputs fallback;                 // colour, motion, depth
    std::array<game_detail::Plane, 8> guides;              // the guides and optional inputs, when one cannot be sampled directly
    bool started = false;
    // debugging: D4R_ENGINE_RR_DUMP=DIR copies every engine image after each frame into a host-visible buffer and
    // writes it at the next frame's recording (by then the game has waited for the frame), as DIR/F-NAME.bin
    VkPhysicalDevice physical;
    VkDevice device;
    VkBuffer dumpBuffer = VK_NULL_HANDLE;
    VkDeviceMemory dumpMemory = VK_NULL_HANDLE;
    void* dumpMapped = nullptr;
    std::vector<std::pair<std::string, size_t>> dumpPending;   // name, bytes, in buffer order
    uint32_t dumpFrame = 0;
    Impl(RModel& m, VkPhysicalDevice p, VkDevice d, uint32_t ow, uint32_t oh, uint32_t rw, uint32_t rh)
        : model(m), engine(m, rw, rh, ow, oh), width(rw), height(rh), outputWidth(ow), outputHeight(oh),
          fallback(p, d, rw, rh, rw, rh), physical(p), device(d) {}
    ~Impl() {
        if (dumpBuffer) vkDestroyBuffer(device, dumpBuffer, nullptr);
        if (dumpMemory) vkFreeMemory(device, dumpMemory, nullptr);
    }
    static size_t texel(VkFormat f) {
        return f == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : f == VK_FORMAT_R16_SFLOAT ? 2 : 4;
    }
    void writeDump(const char* dir) {
        size_t offset = 0;
        for (const auto& [name, bytes] : dumpPending) {
            std::ofstream(std::string(dir) + "/" + std::to_string(dumpFrame) + "-" + name + ".bin", std::ios::binary)
                .write(static_cast<const char*>(dumpMapped) + offset, static_cast<std::streamsize>(bytes));
            offset += (bytes + 255) & ~size_t(255);
        }
        dumpPending.clear();
    }
    void recordDump(VkCommandBuffer cb, uint32_t frame) {
        const auto images = engine.debugImages();
        size_t total = 0;
        for (const auto& d : images) total += (size_t(d.width) * d.height * texel(d.format) + 255) & ~size_t(255);
        if (!dumpBuffer) {
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = total; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            game_detail::check(vkCreateBuffer(device, &bi, nullptr, &dumpBuffer), "vkCreateBuffer");
            VkMemoryRequirements r; vkGetBufferMemoryRequirements(device, dumpBuffer, &r);
            VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(physical, &mp);
            const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            uint32_t type = 0;
            while (type < mp.memoryTypeCount && !((r.memoryTypeBits >> type & 1u) && (mp.memoryTypes[type].propertyFlags & want) == want)) ++type;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = r.size; ai.memoryTypeIndex = type;
            game_detail::check(vkAllocateMemory(device, &ai, nullptr, &dumpMemory), "vkAllocateMemory");
            game_detail::check(vkBindBufferMemory(device, dumpBuffer, dumpMemory, 0), "vkBindBufferMemory");
            game_detail::check(vkMapMemory(device, dumpMemory, 0, VK_WHOLE_SIZE, 0, &dumpMapped), "vkMapMemory");
        }
        game_detail::barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                             VK_ACCESS_TRANSFER_READ_BIT);
        size_t offset = 0;
        for (const auto& d : images) {
            VkBufferImageCopy region{};
            region.bufferOffset = offset;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {d.width, d.height, 1};
            vkCmdCopyImageToBuffer(cb, d.image, VK_IMAGE_LAYOUT_GENERAL, dumpBuffer, 1, &region);
            const size_t bytes = size_t(d.width) * d.height * texel(d.format);
            dumpPending.emplace_back(d.name, bytes);
            offset += (bytes + 255) & ~size_t(255);
        }
        game_detail::barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_ACCESS_HOST_READ_BIT);
        dumpFrame = frame;
    }
};
GameDenoiser::GameDenoiser(RModel& model, VkPhysicalDevice physical, VkDevice device, uint32_t outputWidth,
                           uint32_t outputHeight, uint32_t renderWidth, uint32_t renderHeight)
    : impl(std::make_unique<Impl>(model, physical, device, outputWidth, outputHeight, renderWidth, renderHeight)) {}
GameDenoiser::~GameDenoiser() = default;

void GameDenoiser::record(VkCommandBuffer cb, const GameFrameR& frame) {
    auto& g = *impl;
    const uint32_t w = g.width, h = g.height;
    const char* dump = std::getenv("D4R_ENGINE_RR_DUMP");
    static uint32_t frameCount = 0;
    ++frameCount;
    if (dump && !g.dumpPending.empty()) g.writeDump(dump);
    if (!gameColorFormat(frame.color.format) || !gameMotionFormat(frame.motion.format) ||
        !gameDepthFormat(frame.depth.format, frame.depth.aspect) || !gameOutputFormat(frame.output.format))
        throw std::runtime_error("unsupported game texture format");
    // The stages sample every input at its texel coordinates from the origin, so only views at origin zero are
    // borrowed; anything else is copied first. Every input must be in GENERAL when borrowed (one layout for all).
    auto direct = [](const GameImage& image) {
        return image.view != VK_NULL_HANDLE && !image.x && !image.y && image.layout == VK_IMAGE_LAYOUT_GENERAL;
    };
    const bool directColor = direct(frame.color), directMotion = direct(frame.motion), directDepth = direct(frame.depth);
    // the four guides, then the two optional inputs (kept in their own one-channel formats when copied)
    const GameImage* guides[8] = {&frame.normals, &frame.roughness, &frame.diffuseAlbedo, &frame.specularAlbedo,
                                  &frame.specularHitDistance, &frame.disocclusionMask, &frame.subsurfaceGuide,
                                  &frame.colorBeforeParticles};
    const VkFormat guideFormats[8] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                      VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16_SFLOAT,
                                      VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT};
    auto present = [&](int i) { return i < 4 || guides[i]->image != VK_NULL_HANDLE; };
    if (!g.started) {
        if (!g.model.uploadRecorded()) g.model.recordUpload(cb);
        g.engine.recordInitialize(cb);
        g.started = true;
    }
    g.fallback.record(cb, frame.color, frame.motion, frame.depth, w, h, w, h, directColor, directMotion, directDepth);
    for (int i = 0; i < 8; ++i) {
        if (!present(i) || direct(*guides[i])) continue;
        if (!g.guides[i].image) {
            g.fallback.ensure(cb, g.guides[i], guideFormats[i], w, h);
        }
        game_detail::barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        game_detail::copyInput(cb, *guides[i], g.guides[i].image, w, h);
    }
    game_detail::barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    RFrameImages images{};
    images.color = directColor ? frame.color.view : g.fallback.color.view;
    images.motion = directMotion ? frame.motion.view : g.fallback.motion.view;
    images.depth = directDepth ? frame.depth.view : g.fallback.depth.view;
    VkImageView* guideViews[8] = {&images.normals, &images.roughness, &images.diffuseAlbedo, &images.specularAlbedo,
                                  &images.specularHitDistance, &images.disocclusionMask, &images.subsurfaceGuide,
                                  &images.colorBeforeParticles};
    for (int i = 0; i < 8; ++i)
        if (present(i)) *guideViews[i] = direct(*guides[i]) ? guides[i]->view : g.guides[i].view;
    const bool directOutput = frame.output.view != VK_NULL_HANDLE && frame.output.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
        frame.output.layout == VK_IMAGE_LAYOUT_GENERAL && !frame.output.x && !frame.output.y &&
        frame.output.image != frame.color.image && frame.output.image != frame.motion.image && frame.output.image != frame.depth.image;
    g.engine.setFrame(images, frame.settings, directOutput ? FrameOutput{frame.output.view, frame.output.format} : FrameOutput{});
    g.engine.recordFrame(cb);
    if (dump) g.recordDump(cb, frameCount);
    if (directOutput)   // the output stage wrote the game's texture: make it visible to whatever the game does next
        game_detail::barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    else
        game_detail::copyOutput(cb, g.engine.outputImage(), frame.output, g.outputWidth, g.outputHeight);
}
} // namespace d4r
