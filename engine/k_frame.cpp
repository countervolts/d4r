#include "k_frame.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace d4r {
static_assert(sizeof(InputParams) == 128);
static_assert(sizeof(OutputParams) == 188);
static_assert(offsetof(OutputParams, expo) == 148);
static_assert(sizeof(LayerShape) == 16);
static_assert(sizeof(Shape) == 184);
static_assert(sizeof(ExposureParams) == 56);
static_assert(sizeof(KLayerParams) == 448);

std::vector<char> readFile(const std::string& file) {
    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + file);
    auto size = stream.tellg();
    if (size <= 0 || size > 256 * 1024 * 1024) throw std::runtime_error("invalid size: " + file);
    std::vector<char> data(static_cast<size_t>(size));
    stream.seekg(0);
    if (!stream.read(data.data(), size)) throw std::runtime_error("cannot read " + file);
    return data;
}

bool fileExists(const std::string& file) { return bool(std::ifstream(file, std::ios::binary)); }

std::array<uint64_t, 2> kActivationBytes(const Shape& shape) {
    auto base = shape.layers[0];
    std::array<uint64_t, 2> capacity{uint64_t(base.w) * base.h * 16 * 2, 16};
    for (unsigned i = 0; i < KLayers; ++i) {
        const auto& ls = shape.layers[i];
        uint64_t bytes = uint64_t(ls.w) * ls.h * (i < 5 ? KMerged[i] / 4 : i == 10 ? 40 : KChannels[i]) * 2;
        capacity[(i + 1) % 2] = std::max(capacity[(i + 1) % 2], bytes);
    }
    return capacity;
}
uint64_t kSkipBytes(const Shape& shape, unsigned encoder) {
    auto ls = shape.layers[encoder]; return uint64_t(ls.w) * ls.h * KChannels[encoder] * 2;
}

KModelData loadKModel(const std::string& directory, bool dot2) {
    KModelData m;
    if (fileExists(directory + "/direct_origins.bin")) {
        auto marker = readFile(directory + "/direct_origins.bin");
        // Shader revision: 1 has the corrected motion origins; 2 also reads the frame flags (regular depth);
        // 3 also reads the display-motion flag and the game's exposure texture.
        if (marker.size() != 8 || std::memcmp(marker.data(), "D4RO000", 7) || marker[7] < '1' || marker[7] > '3')
            throw std::runtime_error("incompatible direct-origin shader marker");
        m.hasDirectOrigins = true;
        m.hasRegularDepth = marker[7] >= '2';
        m.hasDisplayMotion = m.hasGameExposure = marker[7] == '3';
    }
    auto offsets = readFile(directory + "/offsets.bin");
    uint32_t count = 0, weightSize = 0;
    if (offsets.size() != 16 + KLayers * 108 * 4 || std::memcmp(offsets.data(), "D4RK0002", 8))
        throw std::runtime_error("incompatible K model metadata");
    std::memcpy(&count, offsets.data() + 8, 4); std::memcpy(&weightSize, offsets.data() + 12, 4);
    auto weights = readFile(directory + "/weights.bin"), lut = readFile(directory + "/lut.bin");
    // One reconstruction table (older packages: the ratio it was captured at serves every ratio) or all sixteen.
    if (count != KLayers || weights.size() != weightSize || weights.size() % 2 ||
        (lut.size() != KTableBytes && lut.size() != size_t(KTableBytes) * KTableCount))
        throw std::runtime_error("invalid K model sizes");
    m.hasRatioTables = lut.size() != KTableBytes;
    std::memcpy(m.offsets.data(), offsets.data() + 16, sizeof(m.offsets));
    // layer_dot2.comp reads every matrix W[K][N] with each pair of rows interleaved, so one 32-bit word holds
    // W[2i][n] and W[2i+1][n] (test_k.py dot2_weights): done here, on the validated spans, in place
    std::vector<char> pairs;
    for (unsigned i = 0; i < KLayers; ++i) {
        const auto& layer = m.offsets[i]; uint32_t c = KChannels[i];
        auto span = [&](unsigned slot, uint64_t elements) {
            uint32_t offset = layer[slot];
            if (offset % 16 || (uint64_t(offset) + elements) * 2 > weights.size())
                throw std::runtime_error(std::string("invalid tensor span in ") + KLayerNames[i]);
        };
        auto matrix = [&](unsigned slot, uint32_t k, uint32_t n) {
            span(slot, uint64_t(k) * n);
            if (!dot2) return;
            uint16_t* w = reinterpret_cast<uint16_t*>(weights.data()) + layer[slot];
            pairs.resize(size_t(k) * n * 2);
            uint16_t* t = reinterpret_cast<uint16_t*>(pairs.data());
            for (uint32_t r = 0; r < k; ++r)
                for (uint32_t col = 0; col < n; ++col) t[(r / 2 * n + col) * 2 + r % 2] = w[r * n + col];
            std::memcpy(w, t, pairs.size());
        };
        for (unsigned j = 0; j < 4; ++j) span(j, 16 * c);
        for (unsigned h = 0; h < KHeads[i]; ++h) {
            for (unsigned j = 0; j < 3; ++j) matrix(4 + 5 * h + j, c, 32);
            span(7 + 5 * h, 4096); matrix(8 + 5 * h, 32, c);
        }
        for (unsigned ch = 0; ch < c / 8; ++ch) {
            matrix(44 + 3 * ch, c, 32); span(45 + 3 * ch, 512); matrix(46 + 3 * ch, 32, c);
        }
        if (i < 5) { matrix(104, 4 * c, KMerged[i]); span(105, 16 * KMerged[i]); }
        if (i >= 6) { matrix(104, KChannels[i - 1], 4 * c); span(105, 16 * 4 * c); }
        if (i == 0) { matrix(106, 16, c); span(107, 16 * c); }
        if (i == 10) { matrix(106, c, 48); span(107, 16 * 48); }
    }
    m.weights = std::move(weights); m.lut = std::move(lut);
    return m;
}

ExposureParams makeKExposureParams(const Shape& shape, const ExposureControl& exposure) {
    if (exposure.size[0] <= 0 || exposure.size[1] <= 0 || exposure.origin[0] < 0 || exposure.origin[1] < 0 ||
        uint32_t(exposure.size[0]) > shape.outputWidth || uint32_t(exposure.size[1]) > shape.outputHeight ||
        !(exposure.scale > 0) || !(exposure.key > 0) || !std::isfinite(exposure.scale) || !std::isfinite(exposure.key) ||
        !(exposure.exposureScale > 0) || !(exposure.preExposure > 0) || !std::isfinite(exposure.exposureScale) ||
        !std::isfinite(exposure.preExposure))
        throw std::runtime_error("unsupported exposure rectangle or scale");
    const uint32_t sw = shape.outputWidth / 2, sh = shape.outputHeight / 2;
    const uint32_t bx = (sw + 31) / 32, by = (sh + 31) / 32;
    return {{exposure.origin[0], exposure.origin[1]}, {exposure.size[0], exposure.size[1]},
            1.f / (float(sw) * float(sh)), exposure.scale / exposure.preExposure, exposure.key,
            (exposure.measure && !exposure.game ? 1u : 0u) | (exposure.reset ? 2u : 0u) | (exposure.game ? 4u : 0u), bx * by, bx,
            {int32_t(sw), int32_t(sh)}, exposure.exposureScale, 1.f / exposure.preExposure};
}

void validateKFrame(const Shape& shape, const KModelData& model, const InputParams& input, const OutputParams& output,
                    const std::array<LayerShape, KLayers>* windows, bool borrowedOutput) {
    if (input.tokW != shape.layers[0].w || output.tokW != input.tokW || output.tokH != shape.layers[0].h ||
        input.flags != (output.flags & ~FrameFlagTableMask) || (input.flags & ~(FrameFlagReset | FrameFlagRegularDepth | FrameFlagDisplayMotion | FrameFlagCanonicalColor | FrameFlagCanonicalMotion)) || output.outSize[0] != int32_t(shape.outputWidth) || output.outSize[1] != int32_t(shape.outputHeight) ||
        (borrowedOutput ? output.outOrigin[0] < 0 || output.outOrigin[1] < 0 : output.outOrigin[0] || output.outOrigin[1]) || output.out2tok[0] != .5f || output.out2tok[1] != .5f)
        throw std::runtime_error("frame parameters do not match engine dimensions");
    if ((input.flags & FrameFlagRegularDepth) && !model.hasRegularDepth)
        throw std::runtime_error("regular depth needs a model compiled by the current compile_k.py");
    if ((input.flags & FrameFlagDisplayMotion) && !model.hasDisplayMotion)
        throw std::runtime_error("display-resolution motion needs a model compiled by the current compile_k.py");
    if (!(output.out2rnd[0] > 0 && output.out2rnd[0] <= 1 && output.out2rnd[1] > 0 && output.out2rnd[1] <= 1) ||
        // Display-resolution motion bypasses the depth-neighbour lookup. NGX
        // uses (1,1) for its unused nearest scale in that variant; standard
        // frame construction uses the render ratio. Both are valid there.
        (!(input.flags & FrameFlagDisplayMotion) &&
            (output.near2rnd[0] != output.out2rnd[0] || output.near2rnd[1] != output.out2rnd[1])) ||
        !(output.near2rnd[0] > 0 && output.near2rnd[0] <= 1 && output.near2rnd[1] > 0 && output.near2rnd[1] <= 1) ||
        !std::isfinite(output.sharp) ||
        !(std::isfinite(input.expo) && input.expo > 0 && std::isfinite(input.expoIn) && input.expoIn > 0 &&
          std::isfinite(output.expo) && output.expo > 0 && std::isfinite(input.expoRatio) && input.expoRatio > 0 &&
          std::isfinite(output.expoRatio) && output.expoRatio > 0)) throw std::runtime_error("unsupported scale or exposure");
    if (windows) {
        for (unsigned i = 0; i < KLayers; ++i) {
            auto old = shape.layers[i], ls = (*windows)[i];
            if (ls.w != old.w || ls.h != old.h || ls.sx < 0 || ls.sx > 7 || ls.sy < 0 || ls.sy > 7 || (i != 5 && (ls.sx % 2 || ls.sy % 2)))
                throw std::runtime_error(std::string("unsupported frame window geometry: ") + KLayerNames[i]);
        }
    }
}

void validateKShape(const Shape& shape) {
    auto base = shape.layers[0];
    if (base.w < 256 || base.h < 256 || base.w % 32 || base.h % 32 || uint64_t(base.w) * 4 < shape.outputWidth || uint64_t(base.h) * 4 < shape.outputHeight ||
        uint32_t(base.w) > shape.outputWidth || uint32_t(base.h) > shape.outputHeight)
        throw std::runtime_error("K token grid must be a multiple of 32, at least 256x256, and cover the output");
    for (unsigned i = 0; i < KLayers; ++i) {
        auto layer = shape.layers[i]; unsigned scale = i <= 5 ? i : 10 - i;
        if (layer.w != base.w / (1 << scale) || layer.h != base.h / (1 << scale) || layer.sx < 0 || layer.sx > 7 || layer.sy < 0 || layer.sy > 7 || (i != 5 && (layer.sx % 2 || layer.sy % 2)))
            throw std::runtime_error(std::string("invalid geometry for ") + KLayerNames[i]);
    }
}

// The windows of each layer (x, y group counts) that the frame's results depend on. A window reads only its own
// tokens (mirrored at the grid's edges; a decoder also the 4x4 low-resolution tokens below it and the skip tokens
// of its encoder), so windows whose tokens nothing reads later can be left out without changing any result. The
// token grid is padded (a multiple of 32, at least 256: 1280x720 has 180 of 256 rows), and nothing reads the
// decoders' padded tokens: the reconstruction reads the head of the output's own tokens, the next frame's input
// stage the token feature at most one token past them (bilinear, only for on-screen history). The encoders all
// run: the deep layers' few windows mirror at the grid's edge, so they read every token above them.
// D4R_ENGINE_K_TRIM=0 runs every window (the padded tokens then match NGX's).
unsigned kRecordedLayers() {
    const char* value = std::getenv("D4R_ENGINE_K_LAYERS");
    if (!value || !*value) return KLayers;
    return unsigned(std::min<unsigned long>(std::strtoul(value, nullptr, 10), KLayers));
}
std::array<std::array<uint32_t, 2>, KLayers> kLayerGroups(const Shape& shape, bool trim) {
    std::array<std::array<uint32_t, 2>, KLayers> groups{};
    auto mirror = [](int v, int n) { v = std::abs(v); return std::min(v, 2 * n - 2 - v); };
    auto tdiv2 = [](int v) { return v < 0 ? -(-v / 2) : v / 2; };
    for (unsigned a = 0; a < 2; ++a) {
        auto size = [&](unsigned i) { return a ? shape.layers[i].h : shape.layers[i].w; };
        auto shift = [&](unsigned i) { return a ? shape.layers[i].sy : shape.layers[i].sx; };
        auto full = [&](unsigned i) { return uint32_t(size(i) + shift(i) + 7) / 8; };
        for (unsigned i = 0; i < KLayers; ++i) groups[i][a] = full(i);
        if (!trim) continue;
        // the windows that write tokens [0, n) of layer i
        auto windows = [&](unsigned i, int n) { return std::clamp(uint32_t(n + shift(i) + 7) / 8, 1u, full(i)); };
        // one past the last token those windows read at the layer's resolution, or at half of it (low-resolution)
        auto reads = [&](unsigned i, bool half) {
            int last = 0;
            for (int w = 0; w < int(groups[i][a]); ++w)
                for (int t = 0; t < (half ? 4 : 8); ++t) {
                    const int origin = 8 * w - shift(i);
                    last = std::max(last, half ? mirror(tdiv2(origin) + t, size(i) / 2) : mirror(origin + t, size(i)));
                }
            return last + 1;
        };
        const int out = int(a ? shape.outputHeight : shape.outputWidth);
        int need = std::min(size(10), (out + 3) / 4 + 2), skip[5] = {};
        for (unsigned i = 10; i >= 6; --i) {
            groups[i][a] = windows(i, need);
            skip[10 - i] = reads(i, false);
            need = reads(i, true);
        }
        groups[5][a] = windows(5, need);
        int merged = reads(5, false);
        for (int i = 4; i >= 0; --i) {
            groups[i][a] = windows(unsigned(i), std::max(skip[i], 2 * merged));
            merged = reads(unsigned(i), false);
        }
    }
    return groups;
}

Shape makeKShape(uint32_t width, uint32_t height) {
    if (width < 256 || height < 256 || width > 8192 || height > 8192)
        throw std::runtime_error("standard K output dimensions must be 256..8192");
    Shape shape{}; shape.outputWidth = width; shape.outputHeight = height;
    uint32_t tw = std::max(256u, ((width + 3) / 4 + 31) / 32 * 32);
    uint32_t th = std::max(256u, ((height + 3) / 4 + 31) / 32 * 32);
    for (unsigned i = 0; i < KLayers; ++i) {
        unsigned scale = i <= 5 ? i : 10 - i;
        shape.layers[i] = {int32_t(tw >> scale), int32_t(th >> scale), 4, 4};
    }
    return shape;
}
std::array<LayerShape, KLayers> makeKWindows(const Shape& shape, uint32_t frameIndex) {
    auto windows = shape.layers;
    // Observed over 64 consecutive NGX evaluations: an eight-frame cycle that visits every shift once per axis.
    static constexpr int32_t Cycle[8][2] = {{0, 2}, {5, 6}, {4, 0}, {1, 4}, {7, 5}, {2, 1}, {3, 7}, {6, 3}};
    int32_t x = Cycle[frameIndex % 8][0], y = Cycle[frameIndex % 8][1];
    for (unsigned depth = 0; depth < 3; ++depth) { // dec5, then enc4/dec4, then enc3/dec3
        windows[5 - depth].sx = windows[5 + depth].sx = x; windows[5 - depth].sy = windows[5 + depth].sy = y;
        x = (2 * x + 4) & 7; y = (2 * y + 4) & 7;
    }
    return windows;
}
uint32_t kReconstructionTable(uint32_t outputWidth, uint32_t outputHeight, uint32_t renderWidth, uint32_t renderHeight) {
    const double ratio = std::sqrt(double(outputWidth) * outputHeight / (double(renderWidth) * renderHeight));
    const double position = (ratio - 0.5) * 28.0 / 3.0;                 // table index on NGX's ratio grid
    const double even = 2.0 * std::floor(position * 0.5 + 0.5);         // nearest even table; a tie takes the upper
    return uint32_t(std::clamp(even, 0.0, 14.0));
}
FrameParams makeKFrameParams(const Shape& shape, const FrameSettings& settings) {
    if (!settings.renderWidth || !settings.renderHeight || settings.renderWidth > shape.outputWidth || settings.renderHeight > shape.outputHeight)
        throw std::runtime_error("render dimensions must be positive and no larger than the output");
    FrameParams params{}; auto& in = params.input; auto& out = params.output;
    auto base = shape.layers[0];
    in.tokW = out.tokW = base.w; out.tokH = base.h;
    in.flags = out.flags = (settings.reset ? FrameFlagReset : 0) | (settings.depthInverted ? 0 : FrameFlagRegularDepth) | (settings.displayMotion ? FrameFlagDisplayMotion : 0) |
        (settings.canonicalColor ? FrameFlagCanonicalColor : 0) | (settings.canonicalMotion ? FrameFlagCanonicalMotion : 0);
    out.flags |= kReconstructionTable(shape.outputWidth, shape.outputHeight, settings.renderWidth, settings.renderHeight) << FrameFlagTableShift;
    in.expo = out.expo = settings.colorExposure;
    in.expoRatio = out.expoRatio = settings.historyExposureRatio;
    in.expoIn = settings.networkExposureScale; in.minW = 1e-4f;
    out.sharp = settings.sharpness; out.minAlpha = .02f; out.alphaThr = .0200042724609375f;
    out.lutScale = 8; out.c184 = .040008544921875f; out.c186 = 1.0419921875f;
    for (unsigned axis = 0; axis < 2; ++axis) {
        float render = axis ? settings.renderHeight : settings.renderWidth, output = axis ? shape.outputHeight : shape.outputWidth;
        float tokens = axis ? base.h : base.w;
        (axis ? out.kB : out.kA) = -16.f * output / render;
        // NGX's f32 associations: render * (1 / output) (one ulp below render / output at 720/1080), the motion scale
        // as its reciprocal, and the history uv over the 16-padded history (on-screen tests use the output itself)
        const float out2rnd = render * (1.f / output), padded = float((uint32_t(output) + 15) / 16 * 16);
        in.out2rnd[axis] = out.out2rnd[axis] = out.near2rnd[axis] = out2rnd;
        out.sub2rnd[axis] = .5f * out2rnd; out.out2tok[axis] = .5f;
        in.rndOffs[axis] = settings.jitter[axis]; out.rndOffs[axis] = settings.jitter[axis] - .5f;
        in.mvScale[axis] = out.mvScale[axis] = settings.motionScale[0] == 0 && settings.motionScale[1] == 0 ? (settings.displayMotion ? 1.f : 1.f / out2rnd) : settings.motionScale[axis];
        in.mvOffs[axis] = out.mvOffs[axis] = settings.motionOffset[axis];
        in.invHist[axis] = out.invOut[axis] = 1.f / padded;
        in.invValid[axis] = out.invOutValid[axis] = 1.f / output;
        in.tok2uv[axis] = 1.f / (tokens * 4.f);
        in.colSize[axis] = in.depSize[axis] = out.colMax[axis] = out.depMax[axis] = int32_t(render) - 1;
        in.mvSize[axis] = out.mvMax[axis] = int32_t(settings.displayMotion ? output : render) - 1;
        out.outSize[axis] = int32_t(output);
    }
    return params;
}
namespace {
template<class T> T structure(const std::string& file) {
    auto data = readFile(file);
    if (data.size() != sizeof(T)) throw std::runtime_error("invalid structure size: " + file);
    T result; std::memcpy(&result, data.data(), sizeof(result)); return result;
}
} // namespace

KGamePlan planKGameFrame(const Shape& shape, const FrameSettings& settings, bool hasDirectOrigins, KGameRect color,
                         KGameRect motion, KGameRect depth, KGameRect output, bool gameExposure, uint32_t& evaluation) {
    const uint32_t w = settings.renderWidth, h = settings.renderHeight;
    if (gameExposure && settings.autoExposure) throw std::runtime_error("game exposure and automatic exposure are exclusive");
    KGamePlan plan;
    // validates the settings before anything is recorded
    plan.params = makeKFrameParams(shape, settings);
    auto& in = plan.params.input; auto& out = plan.params.output;
    auto origin = [&](const KGameRect& rect, int32_t* inOrigin, int32_t* outOrigin, int32_t* outMax) {
        if (!rect.direct) return;
        if ((rect.x || rect.y) && !hasDirectOrigins) throw std::runtime_error("recompile the model for direct texture rectangles");
        if (uint64_t(rect.x) + w > INT32_MAX || uint64_t(rect.y) + h > INT32_MAX)
            throw std::runtime_error("texture rectangle exceeds shader coordinate range");
        inOrigin[0] = outOrigin[0] = int32_t(rect.x); inOrigin[1] = outOrigin[1] = int32_t(rect.y);
        outMax[0] += int32_t(rect.x); outMax[1] += int32_t(rect.y);
    };
    origin(color, in.colOrigin, out.colOrigin, out.colMax);
    origin(motion, in.mvOrigin, out.mvOrigin, out.mvMax);
    origin(depth, in.depOrigin, out.depOrigin, out.depMax);
    if (output.direct) {
        if (uint64_t(output.x) + shape.outputWidth > INT32_MAX || uint64_t(output.y) + shape.outputHeight > INT32_MAX)
            throw std::runtime_error("output rectangle exceeds shader coordinate range");
        out.outOrigin[0] = int32_t(output.x); out.outOrigin[1] = int32_t(output.y);
    }
    auto& exposure = plan.exposure;
    exposure.size[0] = int32_t(w); exposure.size[1] = int32_t(h);
    if (color.direct) { exposure.origin[0] = int32_t(color.x); exposure.origin[1] = int32_t(color.y); }
    exposure.game = gameExposure;
    exposure.measure = !gameExposure && (evaluation % 2 == 0 || settings.reset);
    exposure.reset = settings.reset;
    exposure.exposureScale = settings.exposureScale; exposure.preExposure = settings.preExposure;
    plan.exposureOn = settings.autoExposure || gameExposure;
    plan.windows = makeKWindows(shape, evaluation++);
    return plan;
}

KCase loadKCase(const std::string& file) {
    KCase result;
    std::ifstream manifest(file);
    if (!manifest) throw std::runtime_error("cannot open case manifest");
    std::string line;
    while (std::getline(manifest, line)) {
        std::istringstream s(line); std::string kind; s >> kind;
        if (kind == "shape") {
            std::string name; s >> std::quoted(name); result.shape = structure<Shape>(name);
        } else if (kind == "output") {
            uint32_t w, h; s >> w >> h;
            if (!s) throw std::runtime_error("malformed output dimensions");
            result.shape = makeKShape(w, h);
        } else if (kind == "history") {
            for (auto& name : result.history) s >> std::quoted(name);
        } else if (kind == "image") {
            unsigned id; KCaseImage image;
            s >> id >> std::quoted(image.file) >> image.w >> image.h >> image.channels >> image.bits;
            if (!s || id != result.images.size()) throw std::runtime_error("malformed image declaration");
            if (!image.w || !image.h || (image.channels != 1 && image.channels != 2 && image.channels != 4) ||
                (image.bits != 16 && image.bits != 32)) throw std::runtime_error("unsupported frame image format/dimensions");
            result.images.push_back(image);
        } else if (kind == "frame") {
            KCaseFrame f; std::string input, output, windows;
            s >> std::quoted(input) >> std::quoted(output) >> std::quoted(windows) >> f.color >> f.motion >> f.depth;
            if (!s) throw std::runtime_error("malformed frame declaration");
            f.input = structure<InputParams>(input); f.output = structure<OutputParams>(output);
            f.windows = structure<std::array<LayerShape, KLayers>>(windows);
            result.frames.push_back(f);
        } else if (kind == "frame_auto") {
            KCaseFrame f; FrameSettings settings{}; unsigned reset;
            s >> f.color >> f.motion >> f.depth >> settings.colorExposure >> settings.historyExposureRatio >> settings.networkExposureScale >> reset
              >> settings.jitter[0] >> settings.jitter[1] >> settings.motionScale[0] >> settings.motionScale[1]
              >> settings.motionOffset[0] >> settings.motionOffset[1] >> settings.sharpness;
            if (!s || f.color >= result.images.size() || reset > 1) throw std::runtime_error("malformed standard frame declaration");
            settings.renderWidth = result.images[f.color].w; settings.renderHeight = result.images[f.color].h; settings.reset = reset != 0;
            auto params = makeKFrameParams(result.shape, settings);
            f.input = params.input; f.output = params.output; f.windows = result.shape.layers; f.autoWindows = true;
            result.frames.push_back(f);
        } else if (!kind.empty()) throw std::runtime_error("unknown case declaration " + kind);
    }
    if (result.frames.empty() || !result.shape.outputWidth) throw std::runtime_error("incomplete case manifest");
    if (result.history[0].empty() && !(result.frames[0].input.flags & FrameFlagReset))
        throw std::runtime_error("the first frame must reset or provide history");
    for (auto& f : result.frames)
        if (f.color >= result.images.size() || f.motion >= result.images.size() || f.depth >= result.images.size())
            throw std::runtime_error("invalid frame image id");
    return result;
}
} // namespace d4r
