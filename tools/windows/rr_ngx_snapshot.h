#pragma once
#include "ngx_parameters.h"
#include "../../engine/runtime.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace d4r::win::rr {
inline unsigned number(void* parameters, const char* name, unsigned fallback = 0) {
    unsigned value = 0;
    if (d4r_ngx_get_uint(parameters, name, &value) == 1) return value;
    int signedValue = 0;
    if (d4r_ngx_get_int(parameters, name, &signedValue) == 1 && signedValue >= 0) return unsigned(signedValue);
    return fallback;
}
struct Creation {
    unsigned width = 0, height = 0, outWidth = 0, outHeight = 0;
    unsigned flags = 0, outputSubrects = 0, hardwareDepth = 0, roughnessMode = 0;
    unsigned depthType = 0, normalsType = 0, roughnessType = 0, diffuseType = 0, specularType = 0;
    int quality = 0;
    std::array<unsigned, 6> presets{};
    explicit Creation(void* parameters) {
        if (!parameters) throw std::runtime_error("Null RR creation parameters");
        width = number(parameters, "Width"); height = number(parameters, "Height");
        outWidth = number(parameters, "OutWidth"); outHeight = number(parameters, "OutHeight");
        if (!width || !height || !outWidth || !outHeight || width > 16384 || height > 16384 || outWidth > 16384 || outHeight > 16384)
            throw std::runtime_error("Invalid RR creation geometry");
        flags = number(parameters, "DLSS.Feature.Create.Flags");
        outputSubrects = number(parameters, "DLSS.Enable.Output.Subrects");
        hardwareDepth = number(parameters, "DLSS.Use.HW.Depth");
        roughnessMode = number(parameters, "DLSS.Roughness.Mode");
        depthType = number(parameters, "DLSS.Depth.Type"); normalsType = number(parameters, "DLSS.Normals.Type");
        roughnessType = number(parameters, "DLSS.Roughness.Type"); diffuseType = number(parameters, "DLSS.Diffuse.Type");
        specularType = number(parameters, "DLSS.Specular.Type");
        quality = ngx::int_value(parameters, "PerfQualityValue");
        const char* names[] = {"RayReconstruction.Hint.Render.Preset.DLAA", "RayReconstruction.Hint.Render.Preset.Quality",
            "RayReconstruction.Hint.Render.Preset.Balanced", "RayReconstruction.Hint.Render.Preset.Performance",
            "RayReconstruction.Hint.Render.Preset.UltraPerformance", "RayReconstruction.Hint.Render.Preset.UltraQuality"};
        for (size_t i = 0; i < presets.size(); ++i) presets[i] = number(parameters, names[i]);
    }
};
// All data that may outlive EvaluateFeature is owned here. No matrix pointer
// or resource pointer depends on the caller's parameter object's lifetime.
struct Snapshot {
    enum Plane : size_t { Color, Motion, Depth, Normals, Roughness, Diffuse, Specular, Output,
        Exposure, HitDistance, Disocclusion, Subsurface, BeforeParticles, Count };
    std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, Count> resources;
    std::array<std::array<unsigned, 2>, Count> origins{};
    RFrameSettings settings;
    unsigned renderWidth = 0, renderHeight = 0;
    unsigned depthType = 0, normalsType = 0, roughnessType = 0, diffuseType = 0, specularType = 0;
    float exposureScale = 1;
    static ID3D12Resource* resource(void* parameters, const char* name, const char* alias = nullptr) {
        ID3D12Resource* value = nullptr;
        (void)d4r_ngx_get_d3d12_resource(parameters, name, &value);
        if (!value && alias) (void)d4r_ngx_get_d3d12_resource(parameters, alias, &value);
        return value;
    }
    Snapshot(const Creation& creation, void* parameters) {
        if (!parameters) throw std::runtime_error("Null RR frame parameters");
        const char* names[] = {"Color", "MotionVectors", "Depth", "GBuffer.Normals", "GBuffer.Roughness",
            "DLSS.Input.DiffuseAlbedo", "DLSS.Input.SpecularAlbedo", "Output", "ExposureTexture",
            "DLSSD.SpecularHitDistance", "DLSS.DisocclusionMask", "DLSSD.ScreenSpaceSubsurfaceScatteringGuide", "DLSSD.ColorBeforeParticles"};
        for (size_t i = 0; i < Count; ++i)
            resources[i] = resource(parameters, names[i], i == Diffuse ? "GBuffer.DiffuseAlbedo" : i == Specular ? "GBuffer.SpecularAlbedo" : nullptr);
        renderWidth = number(parameters, "DLSS.Render.Subrect.Dimensions.Width", creation.width);
        renderHeight = number(parameters, "DLSS.Render.Subrect.Dimensions.Height", creation.height);
        if (!renderWidth) renderWidth = creation.width;
        if (!renderHeight) renderHeight = creation.height;
        const char* prefixes[] = {"DLSS.Input.Color", "DLSS.Input.MV", "DLSS.Input.Depth", "DLSS.Input.Normals",
            "DLSS.Input.Roughness", "DLSS.Input.DiffuseAlbedo", "DLSS.Input.SpecularAlbedo", "DLSS.Output"};
        for (size_t i = 0; i < 8; ++i) {
            origins[i][0] = number(parameters, (std::string(prefixes[i]) + ".Subrect.Base.X").c_str());
            origins[i][1] = number(parameters, (std::string(prefixes[i]) + ".Subrect.Base.Y").c_str());
        }
        const char* optionalPrefixes[] = {"DLSSD.SpecularHitDistance", "DLSS.DisocclusionMask",
            "DLSSD.ScreenSpaceSubsurfaceScatteringGuide", "DLSSD.ColorBeforeParticles"};
        for (size_t i = HitDistance; i < Count; ++i) {
            origins[i][0] = number(parameters, (std::string(optionalPrefixes[i - HitDistance]) + ".Subrect.Base.X").c_str());
            origins[i][1] = number(parameters, (std::string(optionalPrefixes[i - HitDistance]) + ".Subrect.Base.Y").c_str());
        }
        settings.jitter[0] = ngx::float_value(parameters, "Jitter.Offset.X");
        settings.jitter[1] = ngx::float_value(parameters, "Jitter.Offset.Y");
        settings.motionScale[0] = ngx::float_value(parameters, "MV.Scale.X", 1);
        settings.motionScale[1] = ngx::float_value(parameters, "MV.Scale.Y", 1);
        settings.preExposure = ngx::float_value(parameters, "DLSS.Pre.Exposure", 1);
        settings.frameTimeMs = ngx::float_value(parameters, "FrameTimeDeltaInMsec", 16.666667f);
        exposureScale = ngx::float_value(parameters, "DLSS.Exposure.Scale", 1);
        settings.reset = number(parameters, "Reset") != 0;
        settings.depthInverted = (creation.flags & 8) != 0;
        settings.hardwareDepth = number(parameters, "DLSS.Use.HW.Depth", creation.hardwareDepth) != 0;
        // D/E consume a bound roughness texture even in packed mode. Only the
        // absent-texture case uses normals.w; preset F has a different contract.
        settings.roughnessPacked = number(parameters, "DLSS.Roughness.Mode", creation.roughnessMode) == 1 && !resources[Roughness];
        if (settings.roughnessPacked) { resources[Roughness] = resources[Normals]; origins[Roughness] = origins[Normals]; }
        depthType = number(parameters, "DLSS.Depth.Type", creation.depthType);
        normalsType = number(parameters, "DLSS.Normals.Type", creation.normalsType);
        roughnessType = number(parameters, "DLSS.Roughness.Type", creation.roughnessType);
        diffuseType = number(parameters, "DLSS.Diffuse.Type", creation.diffuseType);
        specularType = number(parameters, "DLSS.Specular.Type", creation.specularType);
        const char* matrixNames[] = {"WorldToViewMatrix", "ViewToClipMatrix"};
        float* matrices[] = {settings.worldToView, settings.viewToClip};
        for (unsigned i = 0; i < 2; ++i) {
            void* source = nullptr;
            if (d4r_ngx_get_void(parameters, matrixNames[i], &source) != 1 || !source)
                throw std::runtime_error(std::string("Missing RR matrix: ") + matrixNames[i]);
            std::memcpy(matrices[i], source, sizeof(float) * 16);
            for (unsigned n = 0; n < 16; ++n)
                if (!std::isfinite(matrices[i][n])) throw std::runtime_error("Nonfinite RR matrix");
        }
        for (float value : {settings.jitter[0], settings.jitter[1], settings.motionScale[0], settings.motionScale[1],
             settings.preExposure, settings.frameTimeMs, exposureScale})
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite RR frame scalar");
        if (settings.preExposure <= 0 || exposureScale <= 0 || settings.frameTimeMs < 0)
            throw std::runtime_error("Invalid RR exposure or frame time");
    }
};
}
