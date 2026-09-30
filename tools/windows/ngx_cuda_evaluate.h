#pragma once
#include "cuda_image_api.h"
#include <cmath>
#include <chrono>
#include <fstream>

extern "C" void d4r_ngx_set_uint(void*, const char*, unsigned);
extern "C" void d4r_ngx_set_int(void*, const char*, int);
extern "C" void d4r_ngx_set_float(void*, const char*, float);
extern "C" void d4r_ngx_set_ull(void*, const char*, unsigned long long);
extern "C" void d4r_ngx_set_void(void*, const char*, void*);

namespace d4r::diag {
inline float half_value(uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    const float sign = bits & 0x8000 ? -1.0f : 1.0f;
    if (exponent == 31) return mantissa ? NAN : sign * INFINITY;
    if (!exponent) return sign * std::ldexp(float(mantissa), -24);
    return sign * std::ldexp(float(1024 + mantissa), int(exponent) - 25);
}
inline void save_output(const std::filesystem::path& directory, const std::vector<uint16_t>& output,
                        unsigned width, unsigned height, unsigned frame) {
    const auto stem = "output-" + std::to_string(frame);
    std::ofstream raw(directory / (stem + ".rgba16f"), std::ios::binary);
    raw.write(reinterpret_cast<const char*>(output.data()), std::streamsize(output.size() * 2));
    // BMP is only a diagnostic preview; the raw file preserves unmodified HDR values.
    const unsigned stride = (width * 3 + 3) & ~3u;
    std::vector<unsigned char> bytes(54 + size_t(stride) * height, 0);
    auto u32 = [&](size_t offset, uint32_t value) {
        for (int j = 0; j < 4; ++j) bytes[offset + j] = static_cast<unsigned char>(value >> (8 * j));
    };
    bytes[0] = 'B'; bytes[1] = 'M'; u32(2, static_cast<uint32_t>(bytes.size())); u32(10, 54);
    u32(14, 40); u32(18, width); u32(22, height); bytes[26] = 1; bytes[28] = 24;
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x)
        for (unsigned c = 0; c < 3; ++c) {
            const float value = half_value(output[(size_t(y) * width + x) * 4 + c]);
            bytes[54 + size_t(height - y - 1) * stride + x * 3 + (2 - c)] =
                static_cast<unsigned char>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
    std::ofstream bmp(directory / (stem + ".bmp"), std::ios::binary);
    bmp.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!raw || !bmp) throw std::runtime_error("Saving DLSS output failed");
}

inline void run_synthetic_dlss(CudaApi& cuda, Library& core, void* parameters,
                              unsigned preset, unsigned frames, const std::filesystem::path& directory) {
    // CPU upload/readback is confined to this synthetic diagnostic. The game
    // backend uses external D3D12 memory and GPU copies instead.
    constexpr unsigned width = 256, height = 144, outWidth = 512, outHeight = 288;
    using Create = unsigned(*)(unsigned, void*, void**);
    using Evaluate = unsigned(*)(void*, void*, void*);
    using Release = unsigned(*)(void*);
    const auto create = core.symbol<Create>("NVSDK_NGX_CUDA_CreateFeature");
    const auto evaluate = core.symbol<Evaluate>("NVSDK_NGX_CUDA_EvaluateFeature");
    const auto release = core.symbol<Release>("NVSDK_NGX_CUDA_ReleaseFeature");
    auto ngx_check = [](unsigned result, const char* operation) {
        std::printf("NGX %s -> 0x%08x\n", operation, result);
        if (result != 1) throw std::runtime_error(std::string(operation) + " result=" + std::to_string(result));
    };
    d4r_ngx_set_uint(parameters, "Width", width);
    d4r_ngx_set_uint(parameters, "Height", height);
    d4r_ngx_set_uint(parameters, "OutWidth", outWidth);
    d4r_ngx_set_uint(parameters, "OutHeight", outHeight);
    d4r_ngx_set_int(parameters, "PerfQualityValue", 2);
    d4r_ngx_set_int(parameters, "DLSS.Feature.Create.Flags", 0);
    d4r_ngx_set_int(parameters, "DLSS.Enable.Output.Subrects", 0);
    d4r_ngx_set_uint(parameters, "CreationNodeMask", 1);
    d4r_ngx_set_uint(parameters, "VisibilityNodeMask", 1);
    for (const char* name : {"DLSS.Hint.Render.Preset.DLAA", "DLSS.Hint.Render.Preset.Quality",
         "DLSS.Hint.Render.Preset.Balanced", "DLSS.Hint.Render.Preset.Performance",
         "DLSS.Hint.Render.Preset.UltraPerformance", "DLSS.Hint.Render.Preset.UltraQuality"})
        d4r_ngx_set_uint(parameters, name, preset);
    CUdeviceptr scratch = 0;
    cuda.check(cuda.cuMemAlloc_v2(&scratch, 64ull * 1024 * 1024), "cuMemAlloc_v2(NGX scratch)");
    struct ScratchCleanup { CudaApi& api; CUdeviceptr pointer; ~ScratchCleanup() { (void)api.cuMemFree_v2(pointer); } } sc{cuda, scratch};
    d4r_ngx_set_void(parameters, "Scratch", reinterpret_cast<void*>(uintptr_t(scratch)));
    d4r_ngx_set_ull(parameters, "Scratch.SizeInBytes", 64ull * 1024 * 1024);
    void* handle = nullptr;
    std::printf("STAGE NGX_CREATE preset=%u dimensions=%ux%u->%ux%u\n", preset, width, height, outWidth, outHeight);
    ngx_check(create(1, parameters, &handle), "NVSDK_NGX_CUDA_CreateFeature");
    if (!handle) throw std::runtime_error("NGX CreateFeature returned a null handle");
    struct FeatureCleanup { Release release; void*& handle; ~FeatureCleanup() { if (handle) (void)release(handle); } } fc{release, handle};
    cuda::ImageApi images(cuda);
    cuda::Image color(images, width, height, 16, 4, false, 1);
    cuda::Image depth(images, width, height, 32, 1, false);
    cuda::Image motion(images, width, height, 16, 2, false);
    cuda::Image exposure(images, 1, 1, 32, 1, false);
    cuda::Image output(images, outWidth, outHeight, 16, 4, true);
    std::vector<uint16_t> input(size_t(width) * height * 4), zeroMotion(size_t(width) * height * 2, 0);
    std::vector<float> inputDepth(size_t(width) * height, 0.5f);
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const size_t index = (size_t(y) * width + x) * 4;
        input[index] = ((x / 16 + y / 16) & 1) ? 0x3a00 : 0x3400; // .75 / .25
        input[index + 1] = (x & 32) ? 0x3800 : 0x3400;
        input[index + 2] = (y & 32) ? 0x3a00 : 0x3800;
        input[index + 3] = 0x3c00;
    }
    const float one = 1.0f;
    color.upload(input.data()); depth.upload(inputDepth.data()); motion.upload(zeroMotion.data()); exposure.upload(&one);
    d4r_ngx_set_void(parameters, "Color", &color.object);
    d4r_ngx_set_void(parameters, "Depth", &depth.object);
    d4r_ngx_set_void(parameters, "MotionVectors", &motion.object);
    d4r_ngx_set_void(parameters, "ExposureTexture", &exposure.object);
    d4r_ngx_set_void(parameters, "Output", &output.object);
    for (const char* name : {"TransparencyMask", "DLSS.Input.Bias.Current.Color.Mask"})
        d4r_ngx_set_void(parameters, name, nullptr);
    for (const char* name : {"Jitter.Offset.X", "Jitter.Offset.Y", "Sharpness"})
        d4r_ngx_set_float(parameters, name, 0.0f);
    for (const char* name : {"MV.Scale.X", "MV.Scale.Y", "DLSS.Pre.Exposure", "DLSS.Exposure.Scale"})
        d4r_ngx_set_float(parameters, name, 1.0f);
    d4r_ngx_set_float(parameters, "FrameTimeDeltaInMsec", 16.666667f);
    d4r_ngx_set_int(parameters, "Disable.Watermark", 1);
    d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Width", width);
    d4r_ngx_set_uint(parameters, "DLSS.Render.Subrect.Dimensions.Height", height);
    for (const char* name : {"DLSS.Input.Color.Subrect.Base.X", "DLSS.Input.Color.Subrect.Base.Y",
         "DLSS.Input.Depth.Subrect.Base.X", "DLSS.Input.Depth.Subrect.Base.Y", "DLSS.Input.MV.Subrect.Base.X",
         "DLSS.Input.MV.Subrect.Base.Y", "DLSS.Output.Subrect.Base.X", "DLSS.Output.Subrect.Base.Y"})
        d4r_ngx_set_uint(parameters, name, 0);
    std::vector<uint16_t> pixels(size_t(outWidth) * outHeight * 4, 0x7e00);
    for (unsigned frame = 0; frame < frames; ++frame) {
        std::fill(pixels.begin(), pixels.end(), 0x7e00); // unwritten output is an explicit failure
        output.upload(pixels.data());
        d4r_ngx_set_int(parameters, "Reset", frame == 0 ? 1 : 0);
        std::printf("STAGE NGX_EVALUATE preset=%u frame=%u\n", preset, frame);
        const auto begin = std::chrono::steady_clock::now();
        ngx_check(evaluate(handle, parameters, nullptr), "NVSDK_NGX_CUDA_EvaluateFeature");
        cuda.check(cuda.cuCtxSynchronize(), "cuCtxSynchronize(NGX Evaluate)");
        const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        output.download(pixels.data());
        size_t nonfinite = 0; double sum = 0, square = 0;
        for (size_t i = 0; i < pixels.size(); ++i) {
            const float value = half_value(pixels[i]);
            if (!std::isfinite(value)) ++nonfinite;
            if ((i & 3) < 3) { sum += value; square += double(value) * value; }
        }
        const double count = outWidth * outHeight * 3;
        const double variance = square / count - (sum / count) * (sum / count);
        std::printf("NGX_OUTPUT preset=%u frame=%u nonfinite=%zu mean=%.9g variance=%.9g evaluate_ms=%.3f\n",
            preset, frame, nonfinite, sum / count, variance, milliseconds);
        if (nonfinite || !std::isfinite(variance) || variance < 1e-6)
            throw std::runtime_error("DLSS output has NaN/Inf, unwritten pixels or no synthetic pattern");
        save_output(directory, pixels, outWidth, outHeight, frame);
    }
    ngx_check(release(handle), "NVSDK_NGX_CUDA_ReleaseFeature"); handle = nullptr;
    std::printf("PASS NGX_EVALUATE preset=%u frames=%u finite=1 network_validation=pending\n", preset, frames);
}
}
