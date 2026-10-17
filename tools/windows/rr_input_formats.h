#pragma once
// RR guide formats shared by the native and FSR RR backends. Include after volk.h and rr_bridge_replay.h.
#include <string>

namespace d4r::win::rr {
// The Linux shim's RR guide formats (supported_input and aux_array_layout): every UNORM/SNORM/float
// layout samples verbatim, as a D3D12 shader reads it; no CPU conversion or reinterpretation is involved.
// Typeless resources use the float/UNORM view and sRGB is read as stored UNORM, like Linux.
inline rr_bridge::Plane guide_format(DXGI_FORMAT format, size_t plane, const char* backend) {
    if (plane == Snapshot::Depth) {
        switch (format) {
            case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT:
            case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
                return {nullptr, VK_FORMAT_R32_SFLOAT, 4};
            case DXGI_FORMAT_R16_FLOAT: return {nullptr, VK_FORMAT_R16_SFLOAT, 2};
            case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_TYPELESS: return {nullptr, VK_FORMAT_R16_UNORM, 2};
            default: break;
        }
    } else {
        switch (format) {
            case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_TYPELESS: return {nullptr, VK_FORMAT_R8_UNORM, 1};
            case DXGI_FORMAT_R8_SNORM: return {nullptr, VK_FORMAT_R8_SNORM, 1};
            case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_TYPELESS: return {nullptr, VK_FORMAT_R8G8_UNORM, 2};
            case DXGI_FORMAT_R8G8_SNORM: return {nullptr, VK_FORMAT_R8G8_SNORM, 2};
            case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
                return {nullptr, VK_FORMAT_R8G8B8A8_UNORM, 4};
            case DXGI_FORMAT_R8G8B8A8_SNORM: return {nullptr, VK_FORMAT_R8G8B8A8_SNORM, 4};
            case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
                return {nullptr, VK_FORMAT_B8G8R8A8_UNORM, 4};
            case DXGI_FORMAT_R16_UNORM: return {nullptr, VK_FORMAT_R16_UNORM, 2};
            case DXGI_FORMAT_R16_SNORM: return {nullptr, VK_FORMAT_R16_SNORM, 2};
            case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_TYPELESS: return {nullptr, VK_FORMAT_R16_SFLOAT, 2};
            case DXGI_FORMAT_R16G16_UNORM: return {nullptr, VK_FORMAT_R16G16_UNORM, 4};
            case DXGI_FORMAT_R16G16_SNORM: return {nullptr, VK_FORMAT_R16G16_SNORM, 4};
            case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_TYPELESS: return {nullptr, VK_FORMAT_R16G16_SFLOAT, 4};
            case DXGI_FORMAT_R16G16B16A16_UNORM: return {nullptr, VK_FORMAT_R16G16B16A16_UNORM, 8};
            case DXGI_FORMAT_R16G16B16A16_SNORM: return {nullptr, VK_FORMAT_R16G16B16A16_SNORM, 8};
            case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_TYPELESS: return {nullptr, VK_FORMAT_R16G16B16A16_SFLOAT, 8};
            case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: return {nullptr, VK_FORMAT_R32_SFLOAT, 4};
            case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G32_TYPELESS: return {nullptr, VK_FORMAT_R32G32_SFLOAT, 8};
            case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_TYPELESS: return {nullptr, VK_FORMAT_R32G32B32A32_SFLOAT, 16};
            case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_TYPELESS: return {nullptr, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4};
            // Same bit layouts: R11G11B10 red in bits 0-10; R9G9B9E5 red in bits 0-8, exponent 27-31.
            case DXGI_FORMAT_R11G11B10_FLOAT: return {nullptr, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4};
            case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return {nullptr, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 4};
            default: break;
        }
    }
    // Linux also accepts UINT/SINT guides by converting them to float on the CPU; this
    // sampled-image route has no conversion pass, so those still fail here, by name.
    throw std::runtime_error(std::string("Unsupported ") + backend + " sampled input format: plane " + std::to_string(plane) +
        " DXGI format " + std::to_string(unsigned(format)));
}
// UNORM guides store [0, 1]; normals in them are remapped to [-1, 1] before denoising.
inline bool unorm_guide(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16G16_UNORM:
            return true;
        default: return false;
    }
}
}
