// GPU-resident preset K inference. All Vulkan handles belong to the caller's device.
#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

#include "k_frame.h"
#include "m_shared.h"

namespace d4r {
// Pipelines created after this call opt out of storage and uniform buffer robustness (VK_EXT_pipeline_robustness);
// the device must enable pipelineRobustness. For a host device with robustBufferAccess2 (vkd3d-proton): RADV then
// cannot use scalar loads for storage buffers, which costs preset K's v_dot2 network about 1.6x on a Steam Deck.
// The engine's shaders stay in bounds without it.
void setBufferRobustnessDisabled(bool disabled);
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
// The network run by something else (the native HIP layers, engine/hip_net.h and the CUDA bridge's
// d4rEngineNet*): the caller owns two storage buffers of the model's own size, shared with that network and
// written/read by the engine's stages as row-major f16. Preset M: `tokens` and `head` are each
// MEngine::networkBufferBytes; the input stage writes the tokens and the expansion reads the head. Preset K:
// `tokens` is Engine::networkTokenBytes and `head` is Engine::networkHeadBytes; the input stage writes the
// tokens and the final store reads the head.
struct ExternalNetwork { VkBuffer tokens = VK_NULL_HANDLE, head = VK_NULL_HANDLE; };

// The network's shaders: cooperative matrices (layer.comp, VK_KHR_cooperative_matrix with 16x16x16 f16 inputs / f32
// accumulation) or v_dot2_f32_f16 without them (layer_dot2.comp, the model's <layer>_dot2.spv: RDNA2). Auto takes
// the matrices when the device exposes the extension (D4R_ENGINE_K_NETWORK=dot2|matrix overrides); a build with the
// Vulkan table (the shim) cannot see the device's extensions, so there Auto means Matrix and the caller decides.
enum class KNetwork { Auto, Matrix, Dot2 };
// Device must enable shaderInt16, shaderStorageImageExtendedFormats,
// storageBuffer16BitAccess, shaderFloat16, subgroupSizeControl,
// computeFullSubgroups and, for KNetwork::Matrix, VK_KHR_cooperative_matrix.
// Caller serializes recording and waits for the GPU before destruction. Frames execute in recording order.
class Model {
public:
    Model(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache = VK_NULL_HANDLE,
          KNetwork network = KNetwork::Auto, bool accuracy = false);
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
    bool hasAccuracyStages() const; // reconstruction compiled with D4R_ACCURACY
    bool hasRgb10Output() const; // a packaged RGB10A2 final-store variant is available
    bool hasDirectOutput(VkFormat format) const; // the final store can write a borrowed output of this format
    bool dot2Network() const; // the network runs layer_dot2.comp (no cooperative matrices)
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class Engine;
};

// Optional diagnostic markers: 0 at entry, then setup, exposure, input, eleven layers,
// reconstruction, and final synchronization (16 intervals). The callback records commands only.
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
    // The input stage's tokens (16 f16 per token; the layers reuse the buffer: see kRecordedLayers).
    VkBuffer tokenBuffer() const;
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
class MModel {
public:
    // dot2: the network, input and expansion stages for GPUs without cooperative matrices or FP8 (RDNA2), the
    // model directory's *_dot2.spv (D4R_ENGINE_M_NETWORK=dot2|matrix overrides); presets M and L
    MModel(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache = VK_NULL_HANDLE,
           bool dot2 = false);
    ~MModel();
    MModel(const MModel&) = delete;
    MModel& operator=(const MModel&) = delete;
    void recordUpload(VkCommandBuffer command);
    bool uploadRecorded() const;
    bool hasExternalNetworkStages() const; // the model directory has the plane-layout input and expansion stages
    bool hasRgb10Output() const;
    bool hasR11G11B10Output() const;   // a packaged R11G11B10F final-store variant (down_m_r11g11b10.spv)
    bool hasDynamicResolution() const; // active-rectangle sampling: D4RM0004 / D4RL0002
    bool hasDisplayMotion() const;     // display-resolution motion vectors (MFrameSettings::displayMotion): D4RM0005, L's dot2
    bool isPresetL() const;    // a preset L model directory (offsets.bin D4RL0002): MEngine then runs preset L
    bool dot2Network() const;  // the v_dot2 stages (no cooperative matrices, no FP8)
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
    // Constructor render dimensions allocate capacity. Vulkan frames may use smaller active dimensions;
    // external networks retain their constructor grid. A size change resets temporal history.
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
    // Development: the engine's intermediate buffers and render-resolution images (GENERAL), for replays that compare
    // them with a capture. Valid while the engine lives; read them only after a recorded frame has completed.
    struct DebugResource { const char* name; VkBuffer buffer; VkImage image; VkDeviceSize bytes; uint32_t width, height; };
    std::vector<DebugResource> debugResources() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// ---------------------------------------------------------------- Ray Reconstruction (DLSS-D, presets D and E)
// NVIDIA's denoiser frame as d4r's own shaders (engine/rr): exposure, six encoders, five decoders and the output
// stage, each reading and writing NGX's layouts so it can be checked against a captured launch. The network runs at
// the render size, the output stage at the output size. Same conventions as the classes above.
struct RFrameImages {
    // Render-resolution inputs, sampled: colour, motion (two channels, before MV.Scale), depth (one channel), and the
    // guides the denoiser reads (GBuffer.Normals, GBuffer.Roughness, the diffuse and specular albedo).
    VkImageView color, motion, depth, normals, roughness, diffuseAlbedo, specularAlbedo;
    // Optional inputs (null: the game passes none): DLSSD.SpecularHitDistance and DLSS.DisocclusionMask, one channel each
    VkImageView specularHitDistance = VK_NULL_HANDLE, disocclusionMask = VK_NULL_HANDLE;
    // DLSSD.ScreenSpaceSubsurfaceScatteringGuide (one channel) and DLSSD.ColorBeforeParticles (colour), optional too
    VkImageView subsurfaceGuide = VK_NULL_HANDLE, colorBeforeParticles = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL; // of every view
};
struct RFrameSettings {
    float jitter[2] = {0, 0};         // Jitter.Offset, render pixels
    float motionScale[2] = {1, 1};    // MV.Scale
    float preExposure = 1;            // DLSS.Pre.Exposure
    float frameTimeMs = 16.666667f;   // FrameTimeDeltaInMsec
    bool reset = false;
    bool depthInverted = false;       // creation flag NVSDK_NGX_DLSS_Feature_Flags_DepthInverted
    bool hardwareDepth = false;       // DLSS.Use.HW.Depth: the depth input is the nonlinear depth buffer (Cyberpunk 2077)
    bool roughnessPacked = false;     // DLSS.Roughness.Mode 1: the roughness is the normals' w (no roughness texture)
    float worldToView[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; // row-major, as the SDK passes them
    float viewToClip[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};
class RModel {
public:
    // int64Atomics: the device has shaderBufferInt64Atomics enabled; preset F then measures the exposure in one pass
    // (engine/rr/exposure_f.comp PASS 2) instead of two. robustImages: the device has robustImageAccess2 enabled; the
    // translated stages then leave the zero of an image load outside the image to the device (engine/rr/ptx2rr.py fetch0)
    RModel(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache = VK_NULL_HANDLE,
           bool int64Atomics = false, bool robustImages = false);
    ~RModel();
    RModel(const RModel&) = delete;
    RModel& operator=(const RModel&) = delete;
    void recordUpload(VkCommandBuffer command);
    bool uploadRecorded() const;
    uint32_t preset() const; // NVSDK_NGX_RayReconstruction_Hint_Render_Preset: 4 = D, 5 = E
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class REngine;
};
// Experimental (engine/vkhip/vkhip_net.h): Ray Reconstruction network stages recorded by the caller in place of the
// engine's shaders. D/E stages: 0 enc0, 1 - 5 enc1 - enc5, 6 - 9 dec4 - dec1, 10 dec0.
// F stages: 0 - 5 enc0 - enc5, then two stages per decoder (6 dec4_0 ... 15 dec0_1).
class RStageRecorder {
public:
    virtual ~RStageRecorder() = default;
    // records `stage` over groupsX x groupsY windows with this frame's window shift (parameter block byte 80);
    // false: the engine's shader runs instead. The engine places its barrier after the stage either way.
    virtual bool recordStage(VkCommandBuffer command, unsigned stage, uint32_t groupsX, uint32_t groupsY, uint32_t shift) = 0;
    // whether recordStage records `stage`, asked for enc0 only: when the model has enc0's gather (enc0_g.spv) and the
    // arena an address, the engine dispatches the gather before recordStage(0), which then runs the rest of the stage
    virtual bool records(unsigned stage) const = 0;
};
// enc0's gather (RModel's enc0_g.spv): per window its shared buffer sh, the 10 x 10 noise grid (4 x f16, 800 bytes) and
// the two 16-byte feature planes of the 10 x 10 pixels; window i (row-major in the frame's grid) at scratchEnd - (i + 1) x
// RGatherBytes
constexpr uint32_t RGatherBytes = 4000;
// What a network stage (enc0 0: with its gather, Swin stages 1 - 9, decoder 0 10) reads and writes besides the arena: its parameter block's fixed fields
// (the level's token grid, the arena offsets at block bytes 8, 24, 48 and 56) and its images (null where the stage has
// none; decoder 0's surface 144 alternates with the frame's parity).
struct RStageResources {
    uint32_t width = 0, height = 0;
    uint64_t arena[4] = {};              // input, skip, output, merge
    VkImageView texture88 = VK_NULL_HANDLE, texture96 = VK_NULL_HANDLE, texture104 = VK_NULL_HANDLE;
    VkImageView surface112 = VK_NULL_HANDLE, surface120 = VK_NULL_HANDLE, surface128 = VK_NULL_HANDLE;
    VkImageView surface144[2] = {};      // per frame parity
    uint64_t scratchEnd = 0;             // enc0: the end of its gather's scratch (RGatherBytes); 0: no gather
};
class REngine {
public:
    // arenaAddress: the arena buffer gets a device address (the device has bufferDeviceAddress enabled), for a
    // stage recorder that reaches it by address
    REngine(RModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight,
            bool arenaAddress = false);
    ~REngine();
    REngine(const REngine&) = delete;
    REngine& operator=(const REngine&) = delete;
    void recordInitialize(VkCommandBuffer command);
    // A borrowed RGBA16F finalOutput (GENERAL, origin zero) receives the presented image as a copy after the frame;
    // otherwise read outputImage(). Views must outlive this engine.
    void setFrame(const RFrameImages& images, const RFrameSettings& settings, FrameOutput finalOutput = {});
    // With a timestamp query pool, four stamps are written from query `first`: start, after the exposure, after the
    // network (enc0 to dec0) and after the output stage.
    void recordFrame(VkCommandBuffer command, VkQueryPool stamps = VK_NULL_HANDLE, uint32_t first = 0);
    VkImage outputImage() const; // RGBA16F, GENERAL: the presented image of the last recorded frame
    // Debugging (engine/replay_rr.cpp): every image the frame writes, named after the stage and argument offset that
    // writes it ("enc1-112", "dec0-144", "q0", ...), GENERAL layout
    struct DebugImage { std::string name; VkImage image; VkFormat format; uint32_t width, height; };
    std::vector<DebugImage> debugImages() const;
    // The stage recorder (not owned; null: none). stageResources is D/E only (presets A - C: the stage's size);
    // stageParameterAddress is F and A - C only.
    void setStageRecorder(RStageRecorder* recorder);
    VkBuffer arenaBuffer() const;
    uint64_t stageParameterAddress(unsigned stage) const; // presets F, A - C; F requires arenaAddress
    RStageResources stageResources(unsigned stage) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
// ---------------------------------------------------------------- Super Resolution presets E and F (the DLSS 3 CNN)
// NGX 310.7's CNN upscaler: its input kernel and its reconstruction kernel translated from the user's DLL, the
// network between them d4r's own packed-f16 shaders (engine/sr/e_net.comp), so it runs on any GPU with binary16
// arithmetic, cooperative matrices not required (engine/runtime_e.inc, built by engine/compile_sr_e.py). Same
// conventions as the classes above. The device needs shaderFloat16, storageBuffer16BitAccess, shaderInt16,
// subgroupSizeControl and computeFullSubgroups with 32-wide subgroups.
struct EFrameSettings {
    float jitter[2] = {0, 0};       // render pixels, as the game passes them to NGX
    float motionScale[2] = {1, 1};  // NGX's MV scale: motion texels times this are the motion's pixels
    bool reset = false;
    float preExposure = 1, exposureScale = 1;   // the game's (DLSS.Pre.Exposure, DLSS.Exposure.Scale)
};
class EModel {
public:
    EModel(VkPhysicalDevice physical, VkDevice device, const std::string& directory, VkPipelineCache cache = VK_NULL_HANDLE);
    ~EModel();
    EModel(const EModel&) = delete;
    EModel& operator=(const EModel&) = delete;
    void recordUpload(VkCommandBuffer command);
    bool uploadRecorded() const;
    bool hasAutoExposure() const;   // the model directory holds the exposure shaders (exposure_e0/1.spv)
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend class EEngine;
};
class EEngine {
public:
    // hdr: NGX's IsHDR (linear colour, the exposure textures); lowResMotion: MVLowRes (render-size motion vectors,
    // in render pixels; otherwise output-size motion in output pixels). Render size at most the output size.
    // autoExposure (HDR): NGX's AutoExposure, the exposure measured from the colour on every second frame instead of
    // read from images.exposure (needs EModel::hasAutoExposure).
    EEngine(EModel& model, uint32_t renderWidth, uint32_t renderHeight, uint32_t outputWidth, uint32_t outputHeight,
            bool hdr, bool lowResMotion, bool autoExposure = false);
    ~EEngine();
    EEngine(const EEngine&) = delete;
    EEngine& operator=(const EEngine&) = delete;
    void recordInitialize(VkCommandBuffer command);
    // color (and depth with lowResMotion) at the render size, motion at the render or output size; images.exposure
    // the game's 1x1 exposure (HDR without autoExposure; 1 when absent). A borrowed RGBA16F finalOutput (GENERAL) is written directly.
    void setFrame(const FrameImages& images, const EFrameSettings& settings, FrameOutput finalOutput = {});
    // With a timestamp query pool, four stamps are written from query `first`: start, after the input kernel, after
    // the network and after the reconstruction.
    void recordFrame(VkCommandBuffer command, VkQueryPool stamps = VK_NULL_HANDLE, uint32_t first = 0);
    VkImage outputImage() const; // RGBA16F, GENERAL, when no finalOutput was bound
    struct DebugBuffer { std::string name; VkBuffer buffer; VkDeviceSize offset, bytes; };
    std::vector<DebugBuffer> debugBuffers() const; // the network's tensors in the arena (engine/replay_e.cpp)
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace d4r
