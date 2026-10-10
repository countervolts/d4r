// GPU-resident preset K inference. All Vulkan handles belong to the caller's device.
#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace d4r {
constexpr unsigned KLayers = 11;
// These are the shader ABI, not NGX/CUDA parameter blocks. Sizes are checked in runtime.cpp.
struct InputParams {
    float out2rnd[2], rndOffs[2], mvScale[2], mvOffs[2], invHist[2], invValid[2], tok2uv[2];
    int32_t colOrigin[2], colSize[2], depOrigin[2], depSize[2], mvOrigin[2], mvSize[2];
    int32_t tokW; uint32_t flags; // FrameFlagReset, FrameFlagRegularDepth, FrameFlagDisplayMotion
    float expo, expoIn, expoRatio, minW;
};
struct OutputParams {
    float out2rnd[2], rndOffs[2], near2rnd[2], sub2rnd[2], out2tok[2], mvScale[2], mvOffs[2], invOut[2], invOutValid[2];
    int32_t colOrigin[2], colMax[2], depOrigin[2], depMax[2], mvOrigin[2], mvMax[2], outOrigin[2], outSize[2];
    int32_t tokW, tokH; uint32_t flags; // as InputParams::flags (both must agree), plus the table bits
    float expo, expoRatio, sharp, minAlpha, alphaThr, lutScale, kA, kB, c184, c186;
};
constexpr uint32_t FrameFlagReset = 1, FrameFlagRegularDepth = 2, FrameFlagDisplayMotion = 4;
// OutputParams::flags only: the reconstruction table of this frame, 0..15 (kReconstructionTable). A model
// holding a single table ignores it.
constexpr uint32_t FrameFlagTableShift = 8, FrameFlagTableMask = 15u << FrameFlagTableShift;
constexpr uint32_t KTableBytes = 32768, KTableCount = 16;
struct LayerShape { int32_t w, h, sx, sy; };
struct Shape {
    uint32_t outputWidth, outputHeight;
    std::array<LayerShape, KLayers> layers;
};
struct FrameImages {
    VkImageView color, motion, depth;
    VkImageLayout colorLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkImageLayout motionLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkImageLayout depthLayout = VK_IMAGE_LAYOUT_GENERAL;
    // The game's 1x1 exposure texture (a float view; texel 0 is read), used when ExposureControl::game is set.
    VkImageView exposure = VK_NULL_HANDLE;
    VkImageLayout exposureLayout = VK_IMAGE_LAYOUT_GENERAL;
};
// A borrowed storage view in GENERAL. Its format selects the final store pipeline.
struct FrameOutput {
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
};
struct HistoryImages { VkImage color, luma, feature; };
// Standard linear HDR, render-resolution motion vectors. Coordinates start at (0,0).
// Motion is in render pixels by default; provide a scale to convert another convention to output pixels.
// Exposure is supplied by the application. Ratios refer to the compressed history already in this Engine.
struct FrameSettings {
    uint32_t renderWidth, renderHeight;
    float jitter[2] = {0, 0};
    float motionScale[2] = {0, 0}; // (0,0) chooses output/render, or 1 with displayMotion
    float motionOffset[2] = {0, 0};
    float colorExposure = 1, historyExposureRatio = 1, networkExposureScale = 1;
    float sharpness = 1.37f;
    bool reset = false;
    bool depthInverted = true; // false: regular depth (larger is farther); needs Model::hasRegularDepth
    // Motion vectors at output resolution (one texel per output pixel, in output pixels); needs Model::hasDisplayMotion.
    bool displayMotion = false;
    bool autoExposure = false; // GameUpscaler only: measure the exposure instead of using the three fields above; exclusive with GameFrame::exposure
    // GameUpscaler with autoExposure: the game's exposure scale and the factor it pre-multiplied the colour by.
    // Values other than 1 need Model::hasRegularDepth (the same shader revision).
    float exposureScale = 1, preExposure = 1;
};
struct FrameParams { InputParams input; OutputParams output; };
// The network run by something else (the native HIP layers, engine/hip_net.h and the CUDA bridge's
// d4rEngineNet*): the caller owns two storage buffers of the model's own size, shared with that network and
// written/read by the engine's stages as row-major f16. Preset M: `tokens` and `head` are each
// MEngine::networkBufferBytes; the input stage writes the tokens and the expansion reads the head. Preset K:
// `tokens` is Engine::networkTokenBytes and `head` is Engine::networkHeadBytes; the input stage writes the
// tokens and the final store reads the head.
struct ExternalNetwork { VkBuffer tokens = VK_NULL_HANDLE, head = VK_NULL_HANDLE; };

// Automatic K exposure: key / geometric mean of luma on an output/2 grid sampled across the colour rectangle,
// kept as a half float inside the Engine. When passed to setFrame, the frame's exposure fields (expo, expoIn,
// expoRatio) are written on the GPU and the values in InputParams/OutputParams are ignored. NGX measures on
// every second evaluation, starting with the first; a frame that does not measure keeps the last value.
struct ExposureControl {
    int32_t origin[2] = {0, 0}, size[2] = {0, 0};
    float scale = 1;            // applied to the luma before the logarithm, together with 1 / preExposure
    float key = 0.18f / 0.82f;
    // The game's. NGX scales the luma it measures by 1 / pre (cuda_luma_convert_kernel), so the kept measurement is
    // pre times the unscaled one, and the frame's exposure is exposureScale * measured / pre.
    float exposureScale = 1, preExposure = 1;
    bool measure = true;
    bool reset = false;         // the history was produced with no earlier exposure: previous = current
    // The game supplies the exposure in FrameImages::exposure instead of measuring it (measure is then unused).
    bool game = false;
};
Shape makeKShape(uint32_t outputWidth, uint32_t outputHeight);
// NGX keeps sixteen reconstruction tables on a uniform grid of the upscaling ratio (0.5 + index * 3/28) and
// uses the nearest even one for the geometric mean of the two axes' ratios, at most index 14 (ratio 2).
uint32_t kReconstructionTable(uint32_t outputWidth, uint32_t outputHeight, uint32_t renderWidth, uint32_t renderHeight);
FrameParams makeKFrameParams(const Shape& shape, const FrameSettings& settings);
// NGX's window schedule: the bottleneck layer's shift follows an eight-frame cycle, counted from the feature's
// first evaluation, and every shallower layer follows from the one below it as (2 * shift + 4) mod 8, which
// leaves the three outer layers on each side at (4,4).
std::array<LayerShape, KLayers> makeKWindows(const Shape& shape, uint32_t frameIndex);

// Device must enable shaderInt16, shaderStorageImageExtendedFormats,
// storageBuffer16BitAccess, shaderFloat16, subgroupSizeControl,
// computeFullSubgroups and VK_KHR_cooperative_matrix (16x16x16 f16 inputs / f32 accumulation).
// Caller serializes recording and waits for the GPU before destruction. Frames execute in recording order.
class Model {
public:
    Model(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache = VK_NULL_HANDLE);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    // Once, before any Engine using this model. The upload staging memory lives until Model destruction.
    void recordUpload(VkCommandBuffer command);
    bool uploadRecorded() const;
    bool hasAutoExposure() const; // the model directory holds the exposure shaders
    bool hasDirectOrigins() const; // packaged shaders have the corrected nonzero motion-origin handling
    bool hasRegularDepth() const; // packaged shaders read the frame flags, so regular depth can be used
    bool hasDisplayMotion() const; // packaged shaders read FrameFlagDisplayMotion (render-resolution motion otherwise)
    bool hasGameExposure() const; // packaged shaders read FrameImages::exposure (exposure flag 4)
    bool hasRatioTables() const; // lut.bin holds all sixteen reconstruction tables, not only one ratio's
    bool hasRgb10Output() const; // a packaged RGB10A2 final-store variant is available
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class Engine;
};

// Optional diagnostic markers: 0 at entry, then setup, exposure, input, eleven layers,
// reconstruction, and final synchronization (16 intervals). The callback records commands only.
constexpr unsigned KTimingStages = KLayers + 5;
struct FrameProfiler {
    void (*mark)(VkCommandBuffer, uint32_t boundary, void* user) = nullptr;
    void* user = nullptr;
};

class Engine {
public:
    // With an external network the caller owns the token and head buffers (ExternalNetwork) and the eleven
    // layers run outside the engine: recordFront writes the tokens, recordBack reads the head. Without one the
    // engine allocates its own activation buffers and recordFrame runs the whole network internally.
    Engine(Model& model, const Shape& shape, const ExternalNetwork* external = nullptr);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    // The two buffers of an external network, in the row-major f16 layout the kernels read and write:
    // 16 values per token for the input stage's tokens, 40 per token for the head the final store reads.
    static VkDeviceSize networkTokenBytes(uint32_t outputWidth, uint32_t outputHeight);
    static VkDeviceSize networkHeadBytes(uint32_t outputWidth, uint32_t outputHeight);
    // Once, before recordFrame. Clears both sets of temporal history on the GPU.
    void recordInitialize(VkCommandBuffer command);
    // Parameters and window shifts are recorded into the command buffer by recordFrame, so frames may be recorded
    // while earlier ones are still queued. Each image/view/layout combination gets immutable cached descriptors.
    // Borrowed images and views must stay alive until Engine destruction and be on the recording queue family.
    // Images must be sampleable and in the supplied descriptor layouts (GENERAL by default).
    // Window shifts may change each frame; dimensions stay fixed for the lifetime of this Engine.
    // Optional borrowed RGBA16F or RGB10A2 storage view in GENERAL, covering outOrigin + outSize.
    // Views must outlive the Engine: immutable bindings allow rotation while earlier frames remain queued.
    void setFrame(const FrameImages& images, const InputParams& input, const OutputParams& output,
                  const std::array<LayerShape, KLayers>* windows = nullptr, const ExposureControl* exposure = nullptr,
                  FrameOutput finalOutput = {});
    // Thirteen dispatches (up to two more with automatic exposure), with GPU dependencies. No submission, allocation, host wait or readback.
    // A reset frame clears history first. Records are stateful: submit each recorded frame exactly once in order.
    void recordFrame(VkCommandBuffer command);
    void recordFrame(VkCommandBuffer command, const FrameProfiler& profiler);
    // The same frame in two parts, for an external network that runs between them: the front (exposure and the
    // input stage, which writes the tokens; three stamps) and the back (the final store, which reads the head;
    // two stamps, the first before the store). The caller runs the network on the GPU between the two parts.
    void recordFront(VkCommandBuffer command, VkQueryPool stamps = VK_NULL_HANDLE, uint32_t first = 0);
    void recordBack(VkCommandBuffer command, VkQueryPool stamps = VK_NULL_HANDLE, uint32_t first = 0);
    VkImage outputImage() const; // RGBA16F, GENERAL, ready for a transfer read after recordFrame
    HistoryImages historyImages() const; // most recently recorded frame, GENERAL
    VkBuffer headBuffer() const;
    VkDeviceSize headBytes() const;
    VkDeviceSize activationBytes() const;
    // Seed initial history for replay/validation. Caller records copies after recordInitialize, before the first frame.
    HistoryImages initialHistoryImages() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// ---------------------------------------------------------------- preset M (DLSS 4.5)
// The same conventions as the preset K classes above: the caller owns the device and the command buffer, frames
// are recorded with their parameters and execute in recording order. Reconstruction works at 1.5 x the output
// size, selecting NGX's post_3_1/post_3_2 tile class from the render ratio. Render dimensions must not exceed
// the output. Exposure is measured on the GPU every frame, as NGX does for this preset.
struct MFrameSettings {
    float jitter[2] = {0, 0};     // render-pixel offsets in [-0.5, 0.5], as NGX's reconstruction tiles require
    float motionScale[2] = {1, 1};  // (motion texel + offset) * scale = render pixels
    float motionOffset[2] = {0, 0}; // jittered motion vectors: jitter - previous jitter, in motion texel units
    bool reset = false;
    bool depthInverted = true;
};
class MModel {
public:
    MModel(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache = VK_NULL_HANDLE);
    ~MModel();
    MModel(const MModel&) = delete;
    MModel& operator=(const MModel&) = delete;
    void recordUpload(VkCommandBuffer command);
    bool uploadRecorded() const;
    bool hasExternalNetworkStages() const; // the model directory has the plane-layout input and expansion stages
    bool hasRgb10Output() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class MEngine;
};
// The network run by something else (the native HIP layers, engine/hip_net.h): the caller owns the two
// ExternalNetwork buffers, each of MEngine::networkBufferBytes shared with it. The input stage writes the
// tokens, the expansion reads the head, both in NGX's layout ([2][H][W][32] FP8, H x W = the render size
// rounded up to 32, halved).
class MEngine {
public:
    MEngine(MModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight, const ExternalNetwork* external = nullptr);
    static VkDeviceSize networkBufferBytes(uint32_t renderWidth, uint32_t renderHeight);
    ~MEngine();
    MEngine(const MEngine&) = delete;
    MEngine& operator=(const MEngine&) = delete;
    void recordInitialize(VkCommandBuffer command);
    // Sampleable render-resolution views with the supplied layouts. A borrowed finalOutput writes
    // RGBA16F or RGB10A2 directly in GENERAL at origin zero. Views must outlive this engine.
    void setFrame(const FrameImages& images, const MFrameSettings& settings, FrameOutput finalOutput = {});
    // With a timestamp query pool, seven stamps are written from query `first`: start, after exposure, input,
    // network, expansion, reconstruction and downsample.
    void recordFrame(VkCommandBuffer command, VkQueryPool stamps = VK_NULL_HANDLE, uint32_t first = 0);
    // The same frame in two parts, for an external network that runs between them: the front (exposure, input stage;
    // three stamps) and the back (expansion, reconstruction, downsample; four stamps, the first before the expansion).
    void recordFront(VkCommandBuffer command, VkQueryPool stamps = VK_NULL_HANDLE, uint32_t first = 0);
    void recordBack(VkCommandBuffer command, VkQueryPool stamps = VK_NULL_HANDLE, uint32_t first = 0);
    VkImage outputImage() const; // RGBA16F, GENERAL
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r
