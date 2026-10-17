// Exercises native D3D12 guide conversion, composition and exact half widening.
// AMD's denoiser is deliberately absent: this tests our adapter on RDNA3 or WARP.
#include "fsr_rr_native_passes.h"
#include "fsr_rr_route.h"
#include "binary16.h"
#include <array>
#include <vector>
#include <cmath>
#include <cstdio>

using namespace d4r::win;
using namespace d4r::win::rr;
namespace diag = d4r::diag;
int main(int argc, char** argv) try {
    if (native_fsr_route("gfx1101") || !native_fsr_route("gfx1200") || !native_fsr_route("gfx1201") ||
        native_fsr_route("gfx1201", "vkd3d") || !native_fsr_route("gfx1101", "native"))
        throw std::runtime_error("FSR route selection failed");
    bool rejected = false;
    try { native_fsr_route("gfx1201", "invalid"); } catch (...) { rejected = true; }
    if (!rejected) throw std::runtime_error("Invalid FSR route accepted");
    ComPtr<IDXGIFactory4> factory; dx(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "Factory");
    ComPtr<IDXGIAdapter1> adapter;
    auto option = [&](const char* name) { for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == name) return true; return false; };
    const bool warp = option("--warp"), sky = option("--sky"), depthStencil = option("--depth-stencil");
    const bool hardwareDepth = option("--hardware-depth") || depthStencil, subrect = option("--subrect");
    if (depthStencil && subrect) throw std::runtime_error("Use a full depth-stencil rectangle");
    if (warp) dx(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "WARP");
    else {
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapters1(i, &candidate) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{}; candidate->GetDesc1(&desc);
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { adapter = candidate; break; }
        }
    }
    if (!adapter) throw std::runtime_error("No test adapter");
    ComPtr<ID3D12Device> device; dx(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "Device");
    NativeFsrPasses passes; passes.initialize(device.Get());
    ComPtr<ID3D12CommandQueue> queue; D3D12_COMMAND_QUEUE_DESC qd{};
    dx(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "Queue");
    ComPtr<ID3D12CommandAllocator> allocator;
    dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "Allocator");
    ComPtr<ID3D12GraphicsCommandList> list;
    dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "List");
    constexpr UINT width = 65, height = 67;
    auto texture = [&](DXGI_FORMAT format, D3D12_RESOURCE_STATES state, bool uav, UINT tw = 65, UINT th = 67) {
        D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = tw; desc.Height = th; desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.Format = format; desc.SampleDesc.Count = 1;
        if (uav) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT)
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        ComPtr<ID3D12Resource> result;
        dx(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&result)), "Texture");
        return result;
    };
    auto desc_of = [](ID3D12Resource* resource) {
        D3D12_RESOURCE_DESC desc{};
#if defined(__MINGW32__)
        resource->GetDesc(&desc);
#else
        desc = resource->GetDesc();
#endif
        return desc;
    };
    auto location = [](ID3D12Resource* resource) {
        D3D12_TEXTURE_COPY_LOCATION result{}; result.pResource = resource;
        result.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; return result;
    };
    std::array<ComPtr<ID3D12Resource>, 10> inputs;
    std::vector<ComPtr<ID3D12Resource>> uploads;
    float values[10][4] = {{2,1,.5f,1}, {2,0,0,0}, {.25f,-.5f,0,0}, {0,0,1,.75f}, {.5f,0,0,0},
        {.5f,.5f,.5f,0}, {.25f,.25f,.25f,0}, {3,0,0,0}, {5,0,0,0}, {1,.5f,.25f,1}};
    if (sky) values[3][2] = 0;
    if (hardwareDepth) values[1][0] = .25f;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const bool depth = depthStencil && i == 1;
        inputs[i] = texture(depth ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT : DXGI_FORMAT_R32G32B32A32_FLOAT,
            D3D12_RESOURCE_STATE_COPY_DEST, false, width + (subrect ? 6 : 0), height + (subrect ? 8 : 0));
        auto desc = desc_of(inputs[i].Get());
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        auto upload = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr; D3D12_RANGE empty{}; dx(upload->Map(0, &empty, &mapped), "Upload map");
        const float padding[4] = {-10,-10,-10,-10};
        const UINT bytesPerTexel = depth ? 4 : 16;
        for (UINT y = 0; y < desc.Height; ++y) for (UINT x = 0; x < desc.Width; ++x) {
            const bool inside = !subrect || (x >= 3 && x < width + 3 && y >= 4 && y < height + 4);
            std::memcpy(static_cast<char*>(mapped) + y * footprint.Footprint.RowPitch + x * bytesPerTexel,
                inside ? values[i] : padding, bytesPerTexel);
        }
        upload->Unmap(0, nullptr);
        auto dst = location(inputs[i].Get()); D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = upload.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(list.Get(), inputs[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            depth ? D3D12_RESOURCE_STATE_DEPTH_READ : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        uploads.push_back(upload);
    }
    std::array<ComPtr<ID3D12Resource>, 8> prepared;
    std::array<NativeFsrPasses::Guide, 10> guides;
    ID3D12Resource* resources[18]{}; DXGI_FORMAT formats[18]{};
    for (size_t i = 0; i < 10; ++i) {
        const ResourceAccess access(depthStencil && i == 1 ? D3D12_RESOURCE_STATE_DEPTH_READ : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        passes.copy_guide(list.Get(), inputs[i].Get(), access, i == 1, width, height, width, height,
            subrect ? 3 : 0, subrect ? 4 : 0, guides[i]);
        resources[i] = guides[i].resource.Get(); formats[i] = guides[i].format;
    }
    for (size_t i = 0; i < 8; ++i) {
        formats[10+i] = i == 0 ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
        prepared[i] = texture(formats[10+i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true); resources[10+i] = prepared[i].Get();
    }
    D4rFsrRrPrepareConstants constants{};
    for (int i : {0,5,10,15}) constants.clipToView[i] = constants.viewToPreviousView[i] = 1;
    constants.viewToPreviousView[14] = .5f;
    constants.renderSize[0] = width; constants.renderSize[1] = height;
    constants.renderSize[2] = 1.0f/width; constants.renderSize[3] = 1.0f/height;
    constants.motion[0] = 1.0f/width; constants.motion[1] = 1.0f/height;
    constants.motion[2] = constants.motion[3] = 1;
    constants.motionBaseFlags[2] = kFsrRrSpecularHit | kFsrRrDiffuseHit | kFsrRrBeforeParticles | kFsrRrInvalidNormalPassthrough;
    if (hardwareDepth) constants.motionBaseFlags[2] |= kFsrRrHardwareDepth;
    constants.depthParams[0] = constants.depthParams[1] = 1; constants.depthParams[2] = 1e6f;
    constants.detail[1] = .1f;
    passes.record(list.Get(), passes.prepare, &constants, resources, formats, width, height);
    auto composed = texture(DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
    ID3D12Resource* composition[] = {prepared[5].Get(), prepared[6].Get(), prepared[3].Get(), prepared[4].Get(),
        prepared[7].Get(), composed.Get(), prepared[5].Get(), prepared[6].Get()};
    DXGI_FORMAT halfFormats[8]; for (auto& format : halfFormats) format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    struct { UINT width, height; float halo; UINT unused; } composeConstants{width, height, 0, 0};
    passes.record(list.Get(), passes.compose, &composeConstants, composition, halfFormats, width, height);
    transition(list.Get(), composed.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto widened = texture(DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
    ID3D12Resource* widening[] = {composed.Get(), widened.Get()};
    DXGI_FORMAT wideFormats[] = {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT};
    const UINT extent[] = {width, height};
    passes.record(list.Get(), passes.widen, extent, widening, wideFormats, width, height);
    struct Readback { ComPtr<ID3D12Resource> resource; D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint; };
    std::vector<Readback> reads;
    for (size_t i = 0; i < 10; ++i) {
        auto* source = i < 8 ? prepared[i].Get() : i == 8 ? composed.Get() : widened.Get();
        transition(list.Get(), source, i == 8 ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto desc = desc_of(source); Readback r{}; UINT64 bytes = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &r.footprint, nullptr, nullptr, &bytes);
        r.resource = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        auto src = location(source); D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = r.resource.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = r.footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr); reads.push_back(r);
    }
    dx(list->Close(), "Close"); ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence; dx(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Fence");
    Handle event; event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    dx(queue->Signal(fence.Get(), 1), "Signal"); dx(fence->SetEventOnCompletion(1, event.value), "Completion");
    if (WaitForSingleObject(event.value, 30000) != WAIT_OBJECT_0 || fence->GetCompletedValue() == UINT64_MAX)
        throw std::runtime_error("GPU test failed to retire");
    double expected[10][4] = {{2}, {.25/width,-.5/height,.5,0}, {.5,.5,.5,0}, {.5,.5,.5,0},
        {.25,.25,.25,0}, {4.0/3,2.0/3,1.0/3,5}, {4.0/3,2.0/3,1.0/3,3}, {1,.5,.25,0}, {2,1,.5,1}, {2,1,.5,1}};
    if (hardwareDepth) expected[0][0] = .25;
    if (sky) {
        for (int i = 0; i < 8; ++i) for (double& value : expected[i]) value = 0;
        expected[0][0] = 1e7;
        expected[2][0] = expected[2][1] = .5; expected[2][2] = 1;
        expected[7][0] = 2; expected[7][1] = 1; expected[7][2] = .5;
    }
    // D3D's typed float32-to-float16 store truncates toward zero. Compute the
    // independent expected composition from the specified half signal values.
    for (int c = 0; c < 3; ++c) {
        const double signal = diag::binary16::half_value(diag::binary16::half_bits(expected[5][c]));
        const double color = signal * (sky ? 0 : .75) + expected[7][c];
        uint16_t bits = diag::binary16::half_bits(color);
        if (diag::binary16::half_value(bits) > color) --bits;
        expected[8][c] = expected[9][c] = diag::binary16::half_value(bits);
    }
    std::vector<std::vector<uint16_t>> halfData(10);
    for (size_t i = 0; i < reads.size(); ++i) {
        void* mapped = nullptr; dx(reads[i].resource->Map(0, nullptr, &mapped), "Readback map");
        const bool wide = i == 0 || i == 9; const UINT channels = i == 0 ? 1 : 4;
        for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) for (UINT c = 0; c < channels; ++c) {
            const char* p = static_cast<const char*>(mapped) + y * reads[i].footprint.Footprint.RowPitch + (x * channels + c) * (wide ? 4 : 2);
            float value; uint16_t half = 0;
            if (wide) std::memcpy(&value, p, 4);
            else { std::memcpy(&half, p, 2); value = float(diag::binary16::half_value(half)); }
            const double wanted = wide ? expected[i][c] : diag::binary16::half_value(diag::binary16::half_bits(expected[i][c]));
            if (!std::isfinite(value) || std::abs(double(value) - wanted) > 1e-6)
                throw std::runtime_error("Native conversion mismatch plane=" + std::to_string(i) + " channel=" + std::to_string(c) + " value=" + std::to_string(value));
            if (i == 8) halfData[8].push_back(half);
            if (i == 9 && value != diag::binary16::half_value(halfData[8][(y * width + x) * 4 + c]))
                throw std::runtime_error("Native output widening is not exact");
        }
        reads[i].resource->Unmap(0, nullptr);
    }
    std::printf("PASS FSR_NATIVE_CONVERSIONS adapter=%s sky=%d hardware_depth=%d depth_stencil=%d subrect=%d pixels=%u prepared_planes=8 composed=1 exact_widening=1 route_contract=1 denoiser_executed=0\n",
        warp ? "WARP" : "hardware", sky, hardwareDepth, depthStencil, subrect, width*height);
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "FAIL FSR_NATIVE_CONVERSIONS %s\n", error.what()); return 1; }
