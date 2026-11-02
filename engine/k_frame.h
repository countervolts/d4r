// Preset K's frame, independent of the graphics API: the shader parameter blocks, the token grid and its window
// schedule, the frame parameters, the dispatch sizes, and the model package's files. The Vulkan runtime
// (runtime.h) and the D3D12 runtime (d3d12/runtime_d3d12.h) both build on it.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace d4r {
constexpr unsigned KLayers = 11;
// These are the shader ABI, not NGX/CUDA parameter blocks. Sizes are checked in k_frame.cpp.
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
// The colour / motion texels go through the CUDA backend's binary16 canonical conversion (input_quant.glsl): set
// for textures whose format is not exact in binary16 (FrameSettings::canonicalColor / canonicalMotion).
constexpr uint32_t FrameFlagCanonicalColor = 16, FrameFlagCanonicalMotion = 32;
// OutputParams::flags only: the reconstruction table of this frame, 0..15 (kReconstructionTable). A model
// holding a single table ignores it.
constexpr uint32_t FrameFlagTableShift = 8, FrameFlagTableMask = 15u << FrameFlagTableShift;
constexpr uint32_t KTableBytes = 32768, KTableCount = 16;
struct LayerShape { int32_t w, h, sx, sy; };
struct Shape {
    uint32_t outputWidth, outputHeight;
    std::array<LayerShape, KLayers> layers;
};
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
    // Round the colour / motion texels to binary16 toward zero, as the CUDA backend's conversion of the game's
    // textures to its canonical RGBA16F / RG16F planes does; a no-op for textures already in binary16, which the
    // caller can declare by clearing both (the stages then run pipelines without the conversion code; the
    // automatic exposure measures the same colour). The D3D12 backend's shaders read the textures as they are.
    bool canonicalColor = true, canonicalMotion = true;
};
struct FrameParams { InputParams input; OutputParams output; };

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
// ---------------------------------------------------------------- shared by the backends
constexpr const char* KLayerNames[KLayers] = {"enc0", "enc1", "enc2", "enc3", "enc4", "dec5", "dec4", "dec3", "dec2", "dec1", "dec0"};
constexpr uint32_t KChannels[KLayers] = {32, 64, 64, 96, 128, 160, 128, 96, 64, 64, 32};
constexpr uint32_t KHeads[KLayers] = {2, 2, 2, 4, 4, 8, 4, 4, 2, 2, 2};
constexpr uint32_t KMerged[5] = {64, 64, 96, 128, 160};
// The stages a frame profiler times: setup, exposure, input, the eleven layers, output and the finishing barrier.
constexpr unsigned KTimingStages = KLayers + 5;
// layer.comp's parameter block
struct KLayerParams { LayerShape shape; std::array<uint32_t, 108> offsets; };
// exposure_k.comp's parameter block (presets K and M)
struct ExposureParams { int32_t origin[2], size[2]; float invCount, scale, key; uint32_t flags, blocks, blocksX; int32_t sampleSize[2];
                        float gameScale = 1, invPre = 1; }; // the last two: preset K, shader revision 2
// The windows of each layer (x, y group counts) that the frame's results depend on (trim), or all of them.
std::array<std::array<uint32_t, 2>, KLayers> kLayerGroups(const Shape& shape, bool trim);
// Diagnostics: D4R_ENGINE_K_LAYERS=n records only the first n layers (default: all), so a backend's activations
// after layer n - 1 can be read back (the head buffer for odd n, the token buffer for even n; 0: the input tokens).
unsigned kRecordedLayers();
// Bytes of the two alternating activation buffers and of each encoder's skip buffer, for a shape.
std::array<uint64_t, 2> kActivationBytes(const Shape& shape);
uint64_t kSkipBytes(const Shape& shape, unsigned encoder);
// A model package (compile_k.py): offsets.bin, weights.bin and lut.bin, validated, plus the shader revision marker.
// With dot2, every matrix W[K][N] is interleaved in pairs of rows for layer_dot2.comp (the Vulkan RDNA2 network).
struct KModelData {
    std::vector<char> weights, lut;
    std::array<std::array<uint32_t, 108>, KLayers> offsets{};
    bool hasDirectOrigins = false, hasRegularDepth = false, hasDisplayMotion = false, hasGameExposure = false,
         hasRatioTables = false;
};
KModelData loadKModel(const std::string& directory, bool dot2);
bool fileExists(const std::string& file);
std::vector<char> readFile(const std::string& file);
// The exposure pass parameters of a frame (ExposureControl validated against the output size).
ExposureParams makeKExposureParams(const Shape& shape, const ExposureControl& exposure);
// Checks a frame's parameters against the engine's shape and the model's capabilities; throws std::runtime_error.
// borrowedOutput: the frame writes the caller's output image (outOrigin may then be nonzero).
void validateKFrame(const Shape& shape, const KModelData& model, const InputParams& input, const OutputParams& output,
                    const std::array<LayerShape, KLayers>* windows, bool borrowedOutput);
// Checks an engine shape (the token grid and every layer's geometry).
void validateKShape(const Shape& shape);

// A replay case (d4r-k and the D3D12 runner; test_frame.py and prepare_frames.py write them): one declaration per
// line. "output W H" or "shape FILE"; "history COLOR LUMA FEATURE" (seed images); "image ID FILE W H CHANNELS BITS"
// (raw float texels, ids in order); "frame INPUT OUTPUT WINDOWS COLOR MOTION DEPTH" (captured parameter blocks) or
// "frame_auto COLOR MOTION DEPTH EXPOSURE RATIO SCALE RESET JX JY MSX MSY MOX MOY SHARPNESS" (makeKFrameParams).
struct KCaseImage { std::string file; uint32_t w = 0, h = 0; int channels = 0, bits = 0; };
struct KCaseFrame {
    InputParams input{}; OutputParams output{};
    std::array<LayerShape, KLayers> windows{};
    unsigned color = 0, motion = 0, depth = 0;
    bool autoWindows = false;     // frame_auto: the engine's own windows
};
struct KCase {
    Shape shape{};
    std::array<std::string, 3> history;     // empty: the first frame resets
    std::vector<KCaseImage> images;
    std::vector<KCaseFrame> frames;
};
KCase loadKCase(const std::string& file);

// A game frame's plan, shared by the Vulkan and D3D12 game adapters: the stage parameters (makeKFrameParams plus the
// borrowed textures' rectangles), the exposure control and the frame's windows. A borrowed (direct) texture is read
// at its rectangle's origin; a converted one starts at (0,0). `evaluation` is the adapter's frame counter: the window
// schedule follows it (it advances every frame, and a reset does not restart it), so it is incremented here only.
struct KGameRect { bool direct = false; uint32_t x = 0, y = 0; };
struct KGamePlan {
    FrameParams params;
    ExposureControl exposure;
    bool exposureOn = false;            // pass exposure to setFrame (automatic or the game's texture)
    std::array<LayerShape, KLayers> windows;
};
KGamePlan planKGameFrame(const Shape& shape, const FrameSettings& settings, bool hasDirectOrigins, KGameRect color,
                         KGameRect motion, KGameRect depth, KGameRect output, bool gameExposure, uint32_t& evaluation);
} // namespace d4r
