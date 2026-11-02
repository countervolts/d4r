// Shared M/L parameter ABI and weight lowering for Vulkan and native D3D12.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace d4r {
struct MFrameSettings {
    float jitter[2] = {0, 0};     // render-pixel offsets in [-0.5, 0.5], as NGX's reconstruction tiles require
    float motionScale[2] = {1, 1};  // (motion texel + offset) * scale = render pixels
    float motionOffset[2] = {0, 0}; // jittered motion vectors: jitter - previous jitter, in motion texel units
    // Motion vectors at output resolution (one texel per output pixel; motionScale then gives output pixels); needs
    // MModel::hasDisplayMotion
    bool displayMotion = false;
    bool reset = false;
    bool depthInverted = true;
    uint32_t renderWidth = 0, renderHeight = 0; // zero uses the constructor size; otherwise within its capacity
};
} // namespace d4r
namespace d4r::m_detail {
constexpr unsigned MLayers = 10;
constexpr const char* MShader[MLayers] = {"m_enc1", "m_enc2", "m_tube", "m_tube", "m_tube", "m_tube", "m_tube", "m_tube", "m_dec2", "m_dec1"};
constexpr int32_t MShift[MLayers] = {2, 2, 2, 0, 2, 0, 2, 0, 0, 0};
constexpr unsigned MScale[MLayers] = {1, 2, 4, 4, 4, 4, 4, 4, 2, 1};       // token grid divisor
constexpr unsigned MChannels[MLayers] = {64, 96, 128, 128, 128, 128, 128, 128, 96, 64};
constexpr unsigned MHeads[MLayers] = {2, 4, 4, 4, 4, 4, 4, 4, 4, 2};
constexpr unsigned MMerged[MLayers] = {96, 128, 0, 0, 0, 0, 0, 0, 0, 0};    // encoders: channels after the patch merge
constexpr unsigned MExpanded[MLayers] = {0, 0, 0, 0, 0, 0, 0, 0, 128, 96};  // decoders: channels before the patch expand
struct MEnc0Params {
    int32_t size[2], shift[2], tok[2];
    float jitter[2], prevJitter[2], mvScale[2], mvOffs[2], rnd2hi[2], hi2rnd[2], hiSize[2];
    float changeScale, changeGain;
    uint32_t depthFlag, reset, frame;
    float expo, expoPrev;
    float halfInv[2], renderInv[2]; // CUDA texture coordinates use host-rounded reciprocals
    uint32_t displayMv; int32_t mvMax[2]; float mvPix[2]; // display-resolution motion (enc0_m.comp, D4RM0005)
};
struct MDec0Params { int32_t size[2], tokW; uint32_t reset; float rndOffs[2], rnd2hi[2], hiSize[2], histOffs[2], sigma, rho, renderInv[2]; };
struct MPostParams {
    int32_t colOrigin[2], hiSize[2], rndSize[2]; float hi2rnd[2], rndOffs[2], invRnd[2], gain; uint32_t reset; float expo, expoPrev;
    int32_t mvSize[2]; float mvScale[2], mvOffs[2]; // display-resolution motion (post_m.comp, D4RM0005): zero size otherwise
};
struct MDownParams { int32_t srcSize[2], outOrigin[2], colOrigin[2], colSize[2]; float out2rnd[2], rndOffs[2], expo; };
struct MLayerParams { int32_t w, h, sx, sy; uint32_t o[72]; };
// The grid NGX measures the exposure on (cuda_dldn_engine_luma_convert_kernel's launch, presets M and L alike,
// harness captures 2026-10-11): half the output, unless the output is more than twice the render size on that axis
// (Ultra Performance), then half the render size but at least 256 (640x360 -> 1920x1080: 320x256, 854x480 ->
// 2560x1440: 427x256, 1280x720 -> 3840x2160: 640x360).
inline uint32_t exposureGrid(uint32_t render, uint32_t output) { return output / 2 <= render ? output / 2 : std::max(render / 2, 256u); }
static_assert(sizeof(MEnc0Params) == 144 && offsetof(MEnc0Params, expo) == 100 && offsetof(MEnc0Params, halfInv) == 108 &&
              offsetof(MEnc0Params, mvMax) == 128, "enc0_m.comp and exposure_k.comp (words 25, 26)");
static_assert(sizeof(MDec0Params) == 64, "dec0_m.comp");
static_assert(sizeof(MPostParams) == 88 && offsetof(MPostParams, expo) == 56 && offsetof(MPostParams, mvSize) == 64,
              "post_m.comp and exposure_k.comp (words 14, 15)");
static_assert(sizeof(MDownParams) == 52 && offsetof(MDownParams, expo) == 48, "down_m.comp and exposure_k.comp (word 12)");
static_assert(sizeof(MLayerParams) == 304, "layer_m.comp");
// The v_dot2 stages' weights: every FP8 matrix W[K][N] at byte offset o of weights8.bin as binary16 pairs [K/2][N]
// (W[2k][n] low, W[2k + 1][n] high) at dword o / 2, in a buffer twice the size (the matrices do not overlap). The
// layers' matrices are blocked as test_m.pack8 ([K/16][N/16][16 n][16 k]); the input and expansion stages' are plain.
// The attention output projection, MLP and patch merge matrices (v_dot4_i32_i8) are int8 instead: [K/4][N] words of four (W[4k + i][n] in byte i), each column
// scaled by its largest magnitude / 127 (rounded to nearest even), followed by the N column scales as f32
// (test_m.widen does the same).
// qvInt8 (D4RM0006): the heads' q and v matrices are int8 as well (layer_m_dot2.comp gemmHeadI8) and the MLP's w2
// blocks share their column scales in pairs (MLP_PAIR_SCALE); older packages' shaders read q and v as binary16 pairs
// stageInt8 (D4RM0007): the expansion stage's matrix (dec0_m.comp INT8, plain [K/4][N] words) and the decoders'
// expansion matrices (layer_m_dot2.comp EXPAND_INT8) are int8 as well
// Preset L (lEnc0 / lDec0 set; its packages have no older dot2 form): the network as D4RM0007, the input stage's block
// (C 32) with its patch merge (merge_l_dot2.comp) and its embedding (W1 as binary16 pairs, W2 int8), the
// expansion stage's block with its patch expand, all int8 as the network's.
inline std::vector<char> widenM(const std::vector<char>& b8, const std::array<std::array<uint32_t, 74>, MLayers>& layers,
                         const uint32_t enc0[3], const uint32_t dec0[2], bool qvInt8, bool stageInt8,
                         const std::array<uint32_t, 74>* lEnc0 = nullptr, const std::array<uint32_t, 74>* lDec0 = nullptr) {
    const bool pairScale = qvInt8;
    uint16_t half[256];
    for (unsigned c = 0; c < 256; ++c) {
        // E4M3 -> binary16: the magnitude bits shifted into binary16's exponent and mantissa are the value times 2^-8;
        // adding 8 to the exponent field undoes that for normal codes, subnormal codes become normal binary16 values
        const unsigned mag = c & 0x7f, sign = (c & 0x80) << 8;
        unsigned bits;
        if (mag >= 8) bits = (mag << 7) + (8u << 10);
        else if (mag == 0) bits = 0;
        else { unsigned e = 1; while (!((mag << e) & 8)) ++e; bits = (((mag << e) & 7) << 7) | ((9 - e) << 10); }  // mag 2^-9
        half[c] = uint16_t(sign | bits);
    }
    std::vector<char> out(b8.size() * 2, 0);
    auto byte = [&](size_t at) { if (at >= b8.size()) throw std::runtime_error("M weights out of range"); return uint8_t(b8[at]); };
    auto put = [&](size_t dword, uint32_t v) { std::memcpy(out.data() + 4 * dword, &v, 4); };
    auto blocked = [&](size_t off, unsigned K, unsigned N) {
        for (unsigned k = 0; k < K; k += 2) for (unsigned n = 0; n < N; ++n) {
            auto at = [&](unsigned kk) { return off + 256 * ((kk / 16) * (N / 16) + n / 16) + (n % 16) * 16 + kk % 16; };
            put(off / 2 + (k / 2) * N + n, half[byte(at(k))] | (uint32_t(half[byte(at(k + 1))]) << 16));
        }
    };
    auto value = [&](uint8_t c) {
        const unsigned mag = c & 0x7f;
        const double v = mag >= 8 ? std::ldexp(1.0 + (mag & 7) / 8.0, int(mag >> 3) - 7) : std::ldexp(double(mag), -9);
        return c & 0x80 ? -v : v;
    };
    // int8 matrices: [K/4][N] words and N column scales at dword off / 2; with off2 the matrix at off2 shares the
    // column scales (the largest magnitude of both)
    auto int8 = [&](size_t off, unsigned K, unsigned N, size_t off2 = SIZE_MAX, bool plainLayout = false) {
        auto at = [&](size_t o, unsigned k, unsigned n) {
            return plainLayout ? o + size_t(k) * N + n : o + 256 * ((k / 16) * (N / 16) + n / 16) + (n % 16) * 16 + k % 16;
        };
        for (unsigned n = 0; n < N; ++n) {
            double largest = 0;
            for (unsigned k = 0; k < K; ++k) largest = std::max(largest, std::abs(value(byte(at(off, k, n)))));
            if (off2 != SIZE_MAX)
                for (unsigned k = 0; k < K; ++k) largest = std::max(largest, std::abs(value(byte(at(off2, k, n)))));
            const double scale = largest > 0 ? largest / 127.0 : 1.0;
            const float f = float(scale);
            uint32_t bits; std::memcpy(&bits, &f, 4);
            for (size_t o : {off, off2}) {
                if (o == SIZE_MAX) continue;
                for (unsigned k = 0; k < K; k += 4) {
                    uint32_t word = 0;
                    for (unsigned i = 0; i < 4; ++i) {
                        const double q = std::clamp(std::nearbyint(value(byte(at(o, k + i, n))) / scale), -127.0, 127.0);
                        word |= uint32_t(uint8_t(int8_t(q))) << (8 * i);
                    }
                    put(o / 2 + (k / 4) * N + n, word);
                }
                put(o / 2 + (K / 4) * N + n, bits);
            }
        }
    };
    auto plain = [&](size_t off, unsigned K, unsigned N) {
        for (unsigned k = 0; k < K; k += 2) for (unsigned n = 0; n < N; ++n)
            put(off / 2 + (k / 2) * N + n, half[byte(off + k * N + n)] | (uint32_t(half[byte(off + (k + 1) * N + n)]) << 16));
    };
    auto block = [&](size_t base, const uint32_t* o, unsigned C, unsigned NH, unsigned merged, unsigned expanded) {
        for (unsigned h = 0; h < NH; ++h) {
            if (qvInt8) { int8(base + o[4 + 4 * h], C, 32); int8(base + o[5 + 4 * h], C, 32); }
            else { blocked(base + o[4 + 4 * h], C, 32); blocked(base + o[5 + 4 * h], C, 32); }
            // the heads' output projections are one [NA][C] matrix (consecutive [32][C] blocks)
            if (o[7 + 4 * h] != o[7] + h * 32 * C)
                throw std::runtime_error("M weights: the output projections are not consecutive");
        }
        int8(base + o[7], 32 * NH, C);
        for (unsigned c = 0; c < C / 8; ++c) int8(base + o[20 + 3 * c], C, 32);
        // D4RM0006 (layer_m_dot2.comp MLP_PAIR_SCALE): a chunk pair's w2 blocks share their column scales
        for (unsigned c = 0; c < C / 8; c += pairScale ? 2 : 1) {
            if (pairScale) int8(base + o[22 + 3 * c], 32, C, base + o[22 + 3 * (c + 1)]);
            else int8(base + o[22 + 3 * c], 32, C);
        }
        if (merged) int8(base + o[68], 4 * C, merged);
        if (expanded && stageInt8) int8(base + o[68], expanded, 4 * C);
        else if (expanded) blocked(base + o[68], expanded, 4 * C);
    };
    for (unsigned i = 0; i < MLayers; ++i) block(layers[i][0], layers[i].data() + 2, MChannels[i], MHeads[i], MMerged[i], MExpanded[i]);
    if (lEnc0 && lDec0) {
        const uint32_t* o = lEnc0->data() + 2;
        block((*lEnc0)[0], o, 32, 2, 64, 0);
        blocked((*lEnc0)[0] + o[60], 32, 32); int8((*lEnc0)[0] + o[62], 32, 32);
        block((*lDec0)[0], lDec0->data() + 2, 32, 2, 0, 64);
        return out;
    }
    plain(enc0[0], 128, 64);
    if (stageInt8) int8(dec0[0], 64, 80, SIZE_MAX, true); else plain(dec0[0], 64, 80);
    return out;
}
} // namespace d4r::m_detail
