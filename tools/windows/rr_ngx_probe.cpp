// Synthetic NGX feature-13 lifecycle and delayed-submission test. No game access.
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <volk.h>
#include "d3d12_external.h"
#include "rr_bridge_replay.h"
#include "ngx_parameters.h"
#include "ngx_public_abi.h"
#include <dxgi1_4.h>

using namespace d4r::win;
int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 6 && argc != 7) throw std::runtime_error("Usage: rr_ngx_probe SHIM INPUTS REFERENCE OUTPUT MODEL [contracts]");
    const std::string mode = argc == 7 ? argv[6] : "";
    const bool contracts = mode.find("contracts") != std::string::npos;
    const bool inherited = mode.find("inherited") != std::string::npos;
    const bool enhanced = mode.find("enhanced") != std::string::npos;
    const bool split = mode.find("split") != std::string::npos;
    const bool unavailable = mode.find("unavailable") != std::string::npos;
    const bool withoutSr = mode.find("without-sr") != std::string::npos;
    const bool floatGuides = mode.find("float-guides") != std::string::npos;
    const bool switchGuides = mode.find("switch-guides") != std::string::npos;
    const bool autoExposure = mode.find("auto-exposure") != std::string::npos;
    DXGI_FORMAT depthFormat = DXGI_FORMAT_UNKNOWN;
    if (mode.find("depth-d32s8-typeless") != std::string::npos) depthFormat = DXGI_FORMAT_R32G8X24_TYPELESS;
    else if (mode.find("depth-d32s8") != std::string::npos) depthFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    else if (mode.find("depth-d32") != std::string::npos) depthFormat = DXGI_FORMAT_D32_FLOAT;
    else if (mode.find("depth-r32-typeless") != std::string::npos) depthFormat = DXGI_FORMAT_R32_TYPELESS;
    const bool stencil = depthFormat == DXGI_FORMAT_R32G8X24_TYPELESS || depthFormat == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    if (enhanced && split) throw std::runtime_error("The split fixture currently tests legacy split barriers only");
    rr_bridge::Layout layout(argv[2], argv[4], argv[3]);
    const auto canonical = layout;
    if (floatGuides) {
        for (size_t i : {size_t(0), size_t(3)}) {
            layout.planes[i].format = VK_FORMAT_R32G32B32A32_SFLOAT; layout.planes[i].bytes = 16;
        }
        layout.planes[1].format = VK_FORMAT_R32G32_SFLOAT; layout.planes[1].bytes = 8;
        layout.planes[4].format = VK_FORMAT_R16_SFLOAT; layout.planes[4].bytes = 2;
        layout.initializePacket();
    }
    const auto floatLayout = layout;
    auto uploadFrame = [&](void* destination, unsigned index) {
        if (!floatGuides) { layout.upload(destination, index); return; }
        // CPU conversion belongs only to fixture initialization. The production
        // backend transfers and samples the actual D3D12 formats on the GPU.
        std::memset(destination, 0, size_t(layout.bytes));
        for (size_t i = 0; i < layout.planes.size(); ++i) {
            const auto& plane = layout.planes[i];
            if (!plane.enabled || (i >= 7 && !layout.frames[index].optional[i - 7])) continue;
            auto data = rr_bridge::read(layout.inputs / (std::to_string(layout.frames[index].id) + "-" + plane.name + ".bin"));
            if (data.size() != size_t(layout.rw) * layout.rh * canonical.planes[i].bytes)
                throw std::runtime_error("Invalid canonical input for float-guide fixture");
            for (uint32_t y = 0; y < layout.rh; ++y) for (uint32_t x = 0; x < layout.rw; ++x) {
                const auto* source = data.data() + (size_t(y) * layout.rw + x) * canonical.planes[i].bytes;
                auto* dest = static_cast<char*>(destination) + plane.offset + y * plane.pitch + x * plane.bytes;
                if ((i == 0 || i == 1 || i == 3) && plane.bytes != canonical.planes[i].bytes) {
                    for (uint32_t c = 0; c < plane.bytes / 4; ++c) {
                        uint16_t half; std::memcpy(&half, source + c * 2, 2);
                        const float value = float(d4r::diag::binary16::half_value(half)); std::memcpy(dest + c * 4, &value, 4);
                    }
                } else std::memcpy(dest, source, plane.bytes);
            }
        }
    };
    ComPtr<IDXGIFactory4> factory; dx(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "NGX RR factory");
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; ; ++i) {
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) throw std::runtime_error("No AMD adapter");
        DXGI_ADAPTER_DESC1 desc{}; dx(adapter->GetDesc1(&desc), "NGX RR adapter");
        if (desc.VendorId == 0x1002 && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
        adapter.Reset();
    }
    ComPtr<ID3D12Device> device; dx(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "NGX RR device");
    if (enhanced) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 support{};
        dx(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &support, sizeof(support)), "RR enhanced barrier support");
        if (!support.EnhancedBarriersSupported) throw std::runtime_error("RR enhanced barriers unavailable");
    }
    auto enhancedAccess = [](bool output) {
        ResourceAccess access; access.enhanced = true;
        access.layout = output ? D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS : D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE;
        access.access = output ? D3D12_BARRIER_ACCESS_UNORDERED_ACCESS : D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
        return access;
    };
    auto switchBarrierModel = [&](ID3D12GraphicsCommandList* list, ID3D12Resource* texture, bool output, bool toEnhanced) {
        // Microsoft requires COMMON at each legacy/enhanced API boundary.
        // https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html
        const auto legacy = output ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if (toEnhanced) transition(list, texture, legacy, D3D12_RESOURCE_STATE_COMMON);
        ComPtr<ID3D12GraphicsCommandList7> newer; dx(list->QueryInterface(IID_PPV_ARGS(&newer)), "RR enhanced fixture list");
        const auto access = enhancedAccess(output);
        D3D12_TEXTURE_BARRIER barrier{}; barrier.pResource = texture;
        barrier.LayoutBefore = toEnhanced ? D3D12_BARRIER_LAYOUT_COMMON : access.layout;
        barrier.LayoutAfter = toEnhanced ? access.layout : D3D12_BARRIER_LAYOUT_COMMON;
        barrier.AccessBefore = toEnhanced ? D3D12_BARRIER_ACCESS_COMMON : access.access;
        barrier.AccessAfter = toEnhanced ? access.access : D3D12_BARRIER_ACCESS_COMMON;
        barrier.SyncBefore = barrier.SyncAfter = D3D12_BARRIER_SYNC_ALL;
        D3D12_BARRIER_GROUP group{}; group.Type = D3D12_BARRIER_TYPE_TEXTURE; group.NumBarriers = 1; group.pTextureBarriers = &barrier;
        newer->Barrier(1, &group);
        if (!toEnhanced) transition(list, texture, D3D12_RESOURCE_STATE_COMMON, legacy);
    };
    ComPtr<ID3D12CommandQueue> queue; D3D12_COMMAND_QUEUE_DESC q{}; q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    dx(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)), "NGX RR queue");
    d4r::diag::Library shim(d4r::diag::wide(std::filesystem::absolute(argv[1]).string()));
    auto init = shim.symbol<unsigned(*)(const char*, int, const char*, const wchar_t*, ID3D12Device*, unsigned, const void*)>("NVSDK_NGX_D3D12_Init_ProjectID");
    auto allocate = shim.symbol<unsigned(*)(void**)>("NVSDK_NGX_D3D12_AllocateParameters");
    auto destroy = shim.symbol<unsigned(*)(void*)>("NVSDK_NGX_D3D12_DestroyParameters");
    auto create = shim.symbol<unsigned(*)(ID3D12GraphicsCommandList*, unsigned, void*, void**)>("NVSDK_NGX_D3D12_CreateFeature");
    auto evaluate = shim.symbol<unsigned(*)(ID3D12GraphicsCommandList*, const void*, void*, void*)>("NVSDK_NGX_D3D12_EvaluateFeature");
    auto release = shim.symbol<unsigned(*)(void*)>("NVSDK_NGX_D3D12_ReleaseFeature");
    auto shutdown = shim.symbol<unsigned(*)()>("NVSDK_NGX_D3D12_Shutdown");
    auto check = [](unsigned r, const char* operation) { if (r != 1) throw std::runtime_error(std::string(operation) + " NGX=" + std::to_string(r)); };
    auto require = [](bool condition, const char* operation) { if (!condition) throw std::runtime_error(operation); };
    if (contracts || unavailable) {
        auto requirements = shim.symbol<unsigned(*)(IDXGIAdapter*, const d4r::ngx::FeatureDiscoveryInfo*, d4r::ngx::FeatureRequirement*)>("NVSDK_NGX_D3D12_GetFeatureRequirements");
        d4r::ngx::FeatureDiscoveryInfo discovery{}; discovery.FeatureID = 13;
        d4r::ngx::FeatureRequirement result{};
        check(requirements(adapter.Get(), &discovery, &result), "RR requirements");
        require((result.FeatureSupported == 0) != unavailable, "RR requirement readiness disagrees with configured backend");
        ComPtr<IDXGIAdapter> warp; dx(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "RR WARP adapter");
        check(requirements(warp.Get(), &discovery, &result), "RR WARP requirements");
        require(result.FeatureSupported != 0, "RR advertised on unsupported WARP adapter");
        discovery.FeatureID = 14;
        check(requirements(adapter.Get(), &discovery, &result), "Unknown feature requirements");
        require(result.FeatureSupported != 0, "Unknown NGX feature advertised");
    }
    std::string projectId = "24480451-f00d-face-1304-0308dabad187", projectVersion = "1.0";
    std::wstring featurePath = std::filesystem::absolute(argv[2]).wstring();
    const wchar_t* featurePaths[] = {featurePath.c_str()};
    d4r::ngx::FeatureCommonInfo common{}; common.PathListInfo = {featurePaths, 1};
    check(init(projectId.c_str(), 0, projectVersion.c_str(), L"", device.Get(), 0x15, &common), "RR Init");
    // Native RR defers SR startup. Its Init snapshot must survive caller-owned
    // project/path strings and path-list descriptors changing before SR use.
    projectId.assign("mutated-project"); projectVersion.assign("mutated-version");
    featurePath.assign(L"mutated-feature-path"); featurePaths[0] = nullptr;
    common.PathListInfo = {}; common.LoggingInfo.DisableOtherLoggingSinks = true;
    if (contracts || unavailable) {
        auto capabilities = shim.symbol<unsigned(*)(void**)>("NVSDK_NGX_D3D12_GetCapabilityParameters");
        void* caps = nullptr; check(capabilities(&caps), "RR capabilities");
        require(d4r::ngx::int_value(caps, "SuperSamplingDenoising.Available") == (unavailable ? 0 : 1) &&
            (d4r::ngx::int_value(caps, "SuperSamplingDenoising.FeatureInitResult") == 1) != unavailable, "RR capability readiness failed");
        require(d4r::ngx::int_value(caps, "SuperSampling.Available") == (withoutSr ? 0 : 1), "SR capability disagrees with dependency readiness");
        check(destroy(caps), "RR destroy capabilities");
    }
    void* p = nullptr; check(allocate(&p), "RR Allocate");
    d4r_ngx_set_uint(p, "Width", layout.rw); d4r_ngx_set_uint(p, "Height", layout.rh);
    d4r_ngx_set_uint(p, "OutWidth", layout.ow); d4r_ngx_set_uint(p, "OutHeight", layout.oh);
    d4r_ngx_set_int(p, "PerfQualityValue", 2);
    const unsigned creationFlags = 3u | (layout.frames.front().settings.depthInverted ? 8u : 0u) | (autoExposure ? 64u : 0u);
    d4r_ngx_set_uint(p, "DLSS.Feature.Create.Flags", creationFlags);
    std::printf("RR_NGX_CREATION flags=%u\n", creationFlags);
    if (unavailable) {
        void* rejected = reinterpret_cast<void*>(uintptr_t(1));
        require(create(nullptr, 13, p, &rejected) != 1 && !rejected, "Unavailable RR backend accepted creation");
        check(destroy(p), "Unavailable RR destroy parameters");
        check(shutdown(), "Unavailable RR shutdown");
        std::puts("PASS RR_NGX_UNAVAILABLE capabilities=0 requirements=unsupported creation=rejected sr_available=1");
        return 0;
    }
    if (contracts) {
        void* rejected = reinterpret_cast<void*>(uintptr_t(1));
        require(create(nullptr, 14, p, &rejected) != 1 && !rejected, "Unknown feature creation accepted");
        d4r_ngx_set_uint(p, "Width", 0); rejected = reinterpret_cast<void*>(uintptr_t(1));
        require(create(nullptr, 13, p, &rejected) != 1 && !rejected, "Invalid RR creation accepted");
        d4r_ngx_set_uint(p, "Width", layout.rw);
        for (unsigned flags : {0u, 1u, 2u, 7u, 131u}) {
            d4r_ngx_set_uint(p, "DLSS.Feature.Create.Flags", flags); rejected = reinterpret_cast<void*>(uintptr_t(1));
            require(create(nullptr, 13, p, &rejected) != 1 && !rejected, "Unsupported RR creation flags accepted");
        }
        d4r_ngx_set_uint(p, "DLSS.Feature.Create.Flags", creationFlags);
    }
    void* feature = nullptr; check(create(nullptr, 13, p, &feature), "RR CreateFeature");
    auto textures = std::make_unique<rr_bridge::D3D12Frames>(layout, device.Get(), depthFormat);
    ComPtr<ID3D12Resource> stencilUpload, stencilReadback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT stencilFootprint{}; UINT64 stencilBytes = 0;
    if (stencil) {
        const auto desc = resource_desc(textures->inputs[2].Get()); UINT rows = 0; UINT64 rowBytes = 0;
        device->GetCopyableFootprints(&desc, 1, 1, 0, &stencilFootprint, &rows, &rowBytes, &stencilBytes);
        require(rows == layout.rh && rowBytes == layout.rw && stencilBytes && stencilBytes != UINT64_MAX, "Unexpected stencil-plane footprint");
        stencilUpload = make_buffer(device.Get(), stencilBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        stencilReadback = make_buffer(device.Get(), stencilBytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    auto stencilTransition = [](ID3D12GraphicsCommandList* list, ID3D12Resource* image, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {image, 1, from, to}; list->ResourceBarrier(1, &barrier);
    };
    auto stencilCopy = [&](ID3D12GraphicsCommandList* list, ID3D12Resource* image, bool upload) {
        D3D12_TEXTURE_COPY_LOCATION buffer{}; buffer.pResource = upload ? stencilUpload.Get() : stencilReadback.Get();
        buffer.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; buffer.PlacedFootprint = stencilFootprint;
        auto texture = rr_bridge::D3D12Frames::texture(image); texture.SubresourceIndex = 1;
        const auto copy = upload ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
        stencilTransition(list, image, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, copy);
        list->CopyTextureRegion(upload ? &texture : &buffer, 0, 0, 0, upload ? &buffer : &texture, nullptr);
        stencilTransition(list, image, copy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    };
    auto upload = make_buffer(device.Get(), layout.bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto scratch = make_buffer(device.Get(), layout.bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto readback = make_buffer(device.Get(), layout.bytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ComPtr<ID3D12Fence> fence; dx(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "NGX RR fixture fence");
    Handle event; event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event.value) throw std::runtime_error("NGX RR fixture event");
    for (unsigned index = 0; index < layout.frames.size(); ++index) {
        if (switchGuides) {
            layout = index % 2 ? floatLayout : canonical;
            textures = std::make_unique<rr_bridge::D3D12Frames>(layout, device.Get(), depthFormat);
        }
        void* mapped = nullptr; D3D12_RANGE noRead{};
        dx(upload->Map(0, &noRead, &mapped), "NGX RR upload map"); uploadFrame(mapped, index); upload->Unmap(0, nullptr);
        ComPtr<ID3D12CommandAllocator> allocator; ComPtr<ID3D12GraphicsCommandList> list;
        dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "NGX RR fixture allocator");
        dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "NGX RR fixture list");
        if (stencil) {
            dx(stencilUpload->Map(0, &noRead, &mapped), "RR stencil upload map");
            std::memset(mapped, 0, size_t(stencilBytes));
            for (uint32_t y = 0; y < layout.rh; ++y) for (uint32_t x = 0; x < layout.rw; ++x)
                static_cast<uint8_t*>(mapped)[stencilFootprint.Offset + y * stencilFootprint.Footprint.RowPitch + x] = uint8_t(x * 13 + y * 17 + index * 29);
            stencilUpload->Unmap(0, nullptr); stencilCopy(list.Get(), textures->inputs[2].Get(), true);
        }
        uint32_t optionalMask = 0;
        for (unsigned i = 0; i < 4; ++i) if (layout.frames[index].optional[i]) optionalMask |= 1u << i;
        textures->recordInputs(list.Get(), upload.Get(), scratch.Get(), optionalMask);
        if (enhanced) {
            for (size_t i = 0; i < textures->inputs.size(); ++i)
                if (textures->inputs[i] && (i < 7 || (optionalMask & (1u << (i - 7)))))
                    switchBarrierModel(list.Get(), textures->inputs[i].Get(), false, true);
            switchBarrierModel(list.Get(), textures->output.Get(), true, true);
        }
        ComPtr<ID3D12CommandAllocator> prefixAllocator;
        ComPtr<ID3D12GraphicsCommandList> prefix;
        if (inherited) {
            // The preceding list publishes every resource state, including an
            // output which has not otherwise been used by a fixture upload.
            if (!enhanced) {
                transition(list.Get(), textures->output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
                transition(list.Get(), textures->output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            dx(list->Close(), "NGX RR prefix close"); prefix = list; prefixAllocator = allocator;
            list.Reset(); allocator.Reset();
            dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "NGX RR inherited allocator");
            dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "NGX RR inherited list");
        }
        const char* names[] = {"Color", "MotionVectors", "Depth", "GBuffer.Normals", "GBuffer.Roughness", "DLSS.Input.DiffuseAlbedo", "DLSS.Input.SpecularAlbedo"};
        for (size_t i = 0; i < 7; ++i) d4r_ngx_set_d3d12_resource(p, names[i], textures->inputs[i].Get());
        d4r_ngx_set_d3d12_resource(p, "Output", textures->output.Get());
        const char* optionalNames[] = {"DLSSD.SpecularHitDistance", "DLSS.DisocclusionMask", "DLSSD.ScreenSpaceSubsurfaceScatteringGuide", "DLSSD.ColorBeforeParticles"};
        for (size_t i = 0; i < 4; ++i)
            d4r_ngx_set_d3d12_resource(p, optionalNames[i], layout.frames[index].optional[i] ? textures->inputs[i + 7].Get() : nullptr);
        auto settings = layout.frames[index].settings;
        d4r_ngx_set_float(p, "Jitter.Offset.X", settings.jitter[0]); d4r_ngx_set_float(p, "Jitter.Offset.Y", settings.jitter[1]);
        d4r_ngx_set_float(p, "MV.Scale.X", settings.motionScale[0]); d4r_ngx_set_float(p, "MV.Scale.Y", settings.motionScale[1]);
        d4r_ngx_set_float(p, "DLSS.Pre.Exposure", settings.preExposure);
        d4r_ngx_set_uint(p, "Reset", settings.reset); d4r_ngx_set_uint(p, "DLSS.Use.HW.Depth", settings.hardwareDepth);
        d4r_ngx_set_uint(p, "DLSS.Roughness.Mode", settings.roughnessPacked);
        d4r_ngx_set_void(p, "WorldToViewMatrix", settings.worldToView); d4r_ngx_set_void(p, "ViewToClipMatrix", settings.viewToClip);
        const char* states[] = {"D4R.Color.State", "D4R.Motion.State", "D4R.Depth.State", "D4R.Normals.State", "D4R.Roughness.State", "D4R.Diffuse.State", "D4R.Specular.State"};
        if (!inherited) {
            for (const auto* state : states) d4r_ngx_set_uint(p, state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            d4r_ngx_set_uint(p, "D4R.Output.State", D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (contracts && index == 0) {
            auto rejects = [&](const char* name) { require(evaluate(list.Get(), feature, p, nullptr) != 1, name); };
            if (depthFormat == DXGI_FORMAT_D32_FLOAT || stencil) {
                d4r_ngx_set_uint(p, "DLSS.Input.Depth.Subrect.Base.X", 1); rejects("Cropped depth-stencil accepted");
                d4r_ngx_set_uint(p, "DLSS.Input.Depth.Subrect.Base.X", 0);
                std::puts("RR_NGX_DEPTH_SUBRECT cropped_rejected=1");
            }
            d4r_ngx_set_void(p, "ViewToClipMatrix", nullptr); rejects("Missing RR matrix accepted");
            d4r_ngx_set_void(p, "ViewToClipMatrix", settings.viewToClip);
            d4r_ngx_set_d3d12_resource(p, "DLSSD.SpecularHitDistance", textures->inputs[0].Get()); rejects("Invalid RR scalar guide format accepted");
            d4r_ngx_set_d3d12_resource(p, "DLSSD.SpecularHitDistance", layout.frames[index].optional[0] ? textures->inputs[7].Get() : nullptr);
            d4r_ngx_set_uint(p, "DLSS.Render.Subrect.Dimensions.Width", layout.rw + 1); rejects("Unsupported RR dynamic extent accepted");
            d4r_ngx_set_uint(p, "DLSS.Render.Subrect.Dimensions.Width", layout.rw);
            d4r_ngx_set_uint(p, "DLSS.Depth.Type", 1); rejects("Unsupported RR encoding accepted");
            d4r_ngx_set_uint(p, "DLSS.Depth.Type", 0);
            d4r_ngx_set_uint(p, "DLSS.Input.Color.Subrect.Base.X", layout.rw); rejects("Invalid RR subrect accepted");
            d4r_ngx_set_uint(p, "DLSS.Input.Color.Subrect.Base.X", 0);
            d4r_ngx_set_d3d12_resource(p, "Output", textures->inputs[0].Get()); rejects("Aliased or invalid RR output accepted");
            d4r_ngx_set_d3d12_resource(p, "Output", textures->output.Get());
            std::puts("RR_NGX_NEGATIVE rejected=13 command_list_preserved=1");
        }
        if (split && index == 0) {
            D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition = {textures->inputs[0].Get(), 0, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE};
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY; list->ResourceBarrier(1, &barrier);
            require(evaluate(list.Get(), feature, p, nullptr) != 1, "NGX RR accepted unfinished split barrier");
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY; list->ResourceBarrier(1, &barrier);
            transition(list.Get(), textures->inputs[0].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            std::puts("RR_NGX_SPLIT unfinished_rejected=1 paired_before_submit=1");
        }
        check(evaluate(list.Get(), feature, p, nullptr), "RR EvaluateFeature");
        // Both pointer payloads and parameter scalars change before submission.
        settings.worldToView[0] = 999; settings.viewToClip[5] = 999;
        d4r_ngx_set_float(p, "DLSS.Pre.Exposure", 999);
        d4r_ngx_set_d3d12_resource(p, "Color", nullptr);
        auto output = textures->output;
        if (enhanced) texture_copy_barrier(list.Get(), output.Get(), enhancedAccess(true), false, false);
        else transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto source = rr_bridge::D3D12Frames::texture(output.Get());
        auto dest = rr_bridge::D3D12Frames::placed(readback.Get(), layout.outputOffset, layout.outputPitch,
            DXGI_FORMAT_R16G16B16A16_FLOAT, layout.ow, layout.oh);
        list->CopyTextureRegion(&dest, 0, 0, 0, &source, nullptr);
        if (enhanced) {
            texture_copy_barrier(list.Get(), output.Get(), enhancedAccess(true), false, true);
            for (size_t i = 0; i < textures->inputs.size(); ++i)
                if (textures->inputs[i] && (i < 7 || (optionalMask & (1u << (i - 7)))))
                    switchBarrierModel(list.Get(), textures->inputs[i].Get(), false, false);
            switchBarrierModel(list.Get(), output.Get(), true, false);
        } else transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (stencil) stencilCopy(list.Get(), textures->inputs[2].Get(), false);
        if (index + 1 == layout.frames.size()) {
            check(release(feature), "RR Release before submission"); feature = nullptr;
            check(destroy(p), "RR Destroy parameters before submission"); p = nullptr;
            textures.reset(); output.Reset();
        }
        dx(list->Close(), "NGX RR fixture close");
        ID3D12CommandList* lists[] = {prefix ? prefix.Get() : list.Get(), list.Get()};
        queue->ExecuteCommandLists(inherited ? 2 : 1, lists);
        dx(queue->Signal(fence.Get(), index + 1), "NGX RR fixture signal");
        dx(fence->SetEventOnCompletion(index + 1, event.value), "NGX RR fixture completion");
        if (WaitForSingleObject(event.value, 10000) != WAIT_OBJECT_0) throw std::runtime_error("NGX RR fixture timeout");
        D3D12_RANGE range{0, SIZE_T(layout.bytes)}; dx(readback->Map(0, &range, &mapped), "NGX RR readback map");
        try { layout.validate(mapped, index); } catch (...) { readback->Unmap(0, nullptr); throw; }
        readback->Unmap(0, nullptr);
        if (stencil) {
            D3D12_RANGE stencilRange{0, SIZE_T(stencilBytes)};
            dx(stencilReadback->Map(0, &stencilRange, &mapped), "RR stencil verification map");
            bool exact = true;
            for (uint32_t y = 0; y < layout.rh; ++y) for (uint32_t x = 0; x < layout.rw; ++x)
                exact &= static_cast<const uint8_t*>(mapped)[stencilFootprint.Offset + y * stencilFootprint.Footprint.RowPitch + x] == uint8_t(x * 13 + y * 17 + index * 29);
            stencilReadback->Unmap(0, nullptr); require(exact, "RR bridge changed stencil values");
        }
        std::printf("RR_NGX_FRAME index=%u snapshot_mutated=1 exact=1 optional_mask=%u hardware_depth=%u depth_inverted=%u\n",
            index, optionalMask, settings.hardwareDepth, settings.depthInverted);
    }
    check(shutdown(), "RR Shutdown");
    if (contracts) std::puts("PASS RR_NGX_CONTRACTS capabilities=1 requirements=1 warp_rejected=1 invalid_inputs_rejected=1");
    if (inherited) std::puts("PASS RR_NGX_INHERITED prefix_list=1 caller_states=0");
    if (enhanced) std::puts("PASS RR_NGX_ENHANCED tracked_access=1 restored=1");
    if (split) std::puts("PASS RR_NGX_SPLIT unfinished_rejected=1");
    if (withoutSr) std::puts("PASS RR_NGX_WITHOUT_SR rr_executed=1 sr_available=0");
    if (floatGuides) std::puts("PASS RR_NGX_FLOAT_GUIDES color_rgba32=1 motion_rg32=1 normals_rgba32=1 roughness_r16=1");
    if (switchGuides) std::puts("PASS RR_NGX_SWITCH_GUIDES alternate_each_frame=1 feature_recreated=0");
    if (autoExposure) std::puts("PASS RR_NGX_FLAGS hdr=1 render_motion=1 auto_exposure=1");
    if (depthFormat != DXGI_FORMAT_UNKNOWN) std::printf("PASS RR_NGX_DEPTH format=%u stencil_preserved=%u\n", unsigned(depthFormat), stencil);
    std::puts("PASS RR_NGX_LIFECYCLE feature=13 snapshot_mutation=1 release_before_submit=1 parameter_destroy_before_submit=1");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "FAIL RR_NGX %s\n", error.what()); return 1; }
