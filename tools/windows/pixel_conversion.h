#pragma once
#include "d3d12_external.h"
#include "pixel_format.h"

namespace d4r::win {
struct PixelSpec {
    pixel::Storage storage;
    unsigned bytes;
    bool direct;
};
inline PixelSpec pixel_spec(unsigned plane, DXGI_FORMAT format) {
    using namespace pixel;
    // Typeless resources use the corresponding floating-point/UNORM view,
    // as NGX's resource-only ABI provides no SRV descriptor to disambiguate.
    switch (format) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            return {rgba16f, 8, plane == color || plane == output};
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return {rgba32f, 16, false};
        case DXGI_FORMAT_R11G11B10_FLOAT: return {r11g11b10f, 4, false};
        case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_TYPELESS: return {rgb10a2, 4, false};
        case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_TYPELESS: return {rgba8, 4, false};
        case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_TYPELESS: return {bgra8, 4, false};
        case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_TYPELESS: return {rg16f, 4, plane == motion};
        case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G32_TYPELESS: return {rg32f, 8, false};
        case DXGI_FORMAT_R16G16_SNORM: return {rg16snorm, 4, false};
        case DXGI_FORMAT_R16G16_UNORM: return {rg16unorm, 4, false};
        case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT:
            return {r32f, 4, plane == depth || plane == exposure};
        case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
            if (plane != depth) throw std::runtime_error("Depth/stencil resources are supported only for the depth input");
            return {r32f, 4, true}; // D3D12 subresource 0 is the R32 depth plane.
        case DXGI_FORMAT_R16_FLOAT: return {r16f, 2, false};
        case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
            return {depth24, 4, false};
        case DXGI_FORMAT_R16_TYPELESS: return {plane == exposure ? r16f : r16unorm, 2, false};
        case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_UNORM:
            return {r16unorm, 2, false};
        default: throw std::runtime_error("No validated GPU conversion for DXGI format=" + std::to_string(format));
    }
}
inline bool pixel_supported(unsigned plane, pixel::Storage storage) {
    using namespace pixel;
    switch (plane) {
        case color: case output: return storage <= rgb10a2;
        case motion: return storage == rgba16f || storage == rgba32f || (storage >= rg16f && storage <= rg16unorm);
        case depth: return storage == r32f || storage == depth24 || storage == r16unorm;
        case exposure: return storage == r32f || storage == r16f || storage == rgba16f || storage == rgba32f;
        default: return false;
    }
}
class PixelProgram {
    diag::HipApi& hip_;
    hipModule_t module_ = nullptr;
    hipFunction_t decode_ = nullptr, encode_ = nullptr;
public:
    PixelProgram(diag::HipApi& hip, const std::filesystem::path& module) : hip_(hip) {
        try {
            hip.check(hip.hipModuleLoad(&module_, diag::utf8(module.c_str()).c_str()), "Load native GPU format conversion");
            hip.check(hip.hipModuleGetFunction(&decode_, module_, "d4r_pixel_decode"), "Resolve GPU pixel decode");
            hip.check(hip.hipModuleGetFunction(&encode_, module_, "d4r_pixel_encode"), "Resolve GPU pixel encode");
        } catch (...) { if (module_) (void)hip.hipModuleUnload(module_); throw; }
    }
    ~PixelProgram() { if (module_) (void)hip_.hipModuleUnload(module_); }
    PixelProgram(const PixelProgram&) = delete;
    void convert(bool encode, const void* source, uint64_t sourcePitch, void* dest, uint64_t destPitch,
        unsigned width, unsigned height, pixel::Storage format, unsigned plane) {
        if (!width || !height || uint64_t(width) * height > UINT32_MAX) throw std::runtime_error("Invalid GPU conversion dimensions");
        unsigned storage = unsigned(format);
        void* args[] = {&source, &sourcePitch, &dest, &destPitch, &width, &height, &storage, &plane};
        hip_.check(hip_.hipModuleLaunchKernel(encode ? encode_ : decode_, unsigned((uint64_t(width) * height + 255) / 256), 1, 1,
            256, 1, 1, 0, nullptr, args, nullptr), encode ? "GPU pixel encode" : "GPU pixel decode");
        // CUDA may use other nonblocking streams. Establish completion before
        // giving the converted allocation to that API; images remain in VRAM.
        hip_.check(hip_.hipDeviceSynchronize(), "GPU conversion completion");
    }
};
class PixelAllocation {
    diag::HipApi& hip_;
public:
    void* data = nullptr;
    uint64_t pitch;
    PixelAllocation(diag::HipApi& hip, unsigned width, unsigned height, unsigned canonicalBytes) : hip_(hip) {
        pitch = (uint64_t(width) * canonicalBytes + 255) & ~uint64_t(255);
        hip.check(hip.hipMalloc(&data, size_t(pitch * height)), "Allocate canonical GPU pixels");
    }
    ~PixelAllocation() { if (data) (void)hip_.hipFree(data); }
    PixelAllocation(const PixelAllocation&) = delete;
};
}
