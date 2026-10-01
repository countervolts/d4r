#include "d3d12_external.h"
#include "ngx_parameters.h"
#include "ngx_cuda_evaluate.h"
#include "ngx_public_abi.h"
#include <dxgi1_6.h>
#include <memory>
#include <array>

int main(int argc, char** argv) {
    using namespace d4r::diag;
    using namespace d4r::win;
    start();
    try {
        Args args(argc, argv);
        if (args.cuda_dll.empty() || (args.interop_mode != "images" && (args.module.empty() || args.ngx_core.empty() || args.dlss_dll.empty())))
            throw std::runtime_error("--module shim DLL, --cuda-dll, --ngx-core and --dlss-dll required");
        const char* diagDir = std::getenv("D4R_DIAG_DIR");
        if (!diagDir && args.interop_mode != "images") throw std::runtime_error("Set D4R_DIAG_DIR");
        const std::filesystem::path directory(diagDir ? wide(diagDir) : std::filesystem::current_path().wstring());
        std::filesystem::create_directories(directory);
        const bool commandBackend = args.interop_mode == "command-list";
        const bool inherited = args.barrier_mode.rfind("inherited-", 0) == 0;
        const bool enhanced = args.barrier_mode == "enhanced" || args.barrier_mode == "inherited-enhanced";
        if (inherited && !commandBackend) throw std::runtime_error("Inherited state diagnostic requires the command-list backend");
        if (args.interop_mode == "images" && args.pixel_profile != "baseline") throw std::runtime_error("Raw CUDA image diagnostic requires baseline formats");
        std::printf("D3D12_PIXEL_PROFILE %s\n", args.pixel_profile.c_str());
        if (commandBackend && _putenv_s("D4R_D3D12_COMMAND_BACKEND", "1")) throw std::runtime_error("Cannot enable command backend");
        for (const auto& pair : {std::make_pair("D4R_HIP_ROOT", args.hip_root),
             std::make_pair("D4R_NVCUDA_DLL", args.cuda_dll), std::make_pair("D4R_NVAPI_DLL", args.nvapi_dll),
             std::make_pair("D4R_NGX_CORE", args.ngx_core), std::make_pair("D4R_DLSS_DLL", args.dlss_dll)})
            if (_putenv_s(pair.first, pair.second.c_str())) throw std::runtime_error("Cannot set runtime environment");
        HipApi hip(args.hip_root); hipDeviceProp_t props{}; hip.select_gfx1201(args.device, props);
        ComPtr<IDXGIFactory4> factory; dx(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> adapter;
        bool found = false;
        for (UINT i = 0; ; ++i) {
            const HRESULT result = factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf());
            if (result == DXGI_ERROR_NOT_FOUND) break;
            dx(result, "EnumAdapters"); DXGI_ADAPTER_DESC1 desc{}; dx(adapter->GetDesc1(&desc), "GetDesc1");
            if (!std::memcmp(&desc.AdapterLuid, props.luid, sizeof(LUID))) { found = true; break; }
        }
        if (!found) throw std::runtime_error("HIP adapter missing from DXGI");
        ComPtr<ID3D12Device> device;
        dx(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.GetAddressOf())), "D3D12CreateDevice");
        if (enhanced) { D3D12_FEATURE_DATA_D3D12_OPTIONS12 support{}; dx(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &support, sizeof(support)), "Enhanced barrier support"); if (!support.EnhancedBarriersSupported) throw std::runtime_error("Enhanced barriers unavailable"); }
        auto enhancedBarrier = [&](ID3D12GraphicsCommandList* command, ID3D12Resource* texture, unsigned plane, bool initial) {
            ComPtr<ID3D12GraphicsCommandList7> newer; dx(command->QueryInterface(IID_PPV_ARGS(newer.GetAddressOf())), "Harness enhanced list");
            D3D12_TEXTURE_BARRIER barrier{}; barrier.pResource = texture;
            barrier.LayoutAfter = plane == 3 ? D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS : D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE;
            barrier.LayoutBefore = initial ? D3D12_BARRIER_LAYOUT_COMMON : barrier.LayoutAfter;
            barrier.AccessAfter = plane == 3 ? D3D12_BARRIER_ACCESS_UNORDERED_ACCESS : D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
            barrier.AccessBefore = initial ? D3D12_BARRIER_ACCESS_COMMON : barrier.AccessAfter;
            barrier.SyncBefore = barrier.SyncAfter = D3D12_BARRIER_SYNC_ALL;
            D3D12_BARRIER_GROUP group{}; group.Type = D3D12_BARRIER_TYPE_TEXTURE; group.NumBarriers = 1; group.pTextureBarriers = &barrier; newer->Barrier(1, &group);
        };
        ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        dx(device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.GetAddressOf())), "CreateCommandQueue");
        ComPtr<ID3D12Fence> completion;
        dx(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(completion.GetAddressOf())), "CreateFence(harness)");
        Handle event; event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event.value) throw std::runtime_error("CreateEvent(harness)");
        uint64_t fenceValue = 0;
        auto drain = [&] {
            dx(queue->Signal(completion.Get(), ++fenceValue), "Queue Signal(harness)");
            if (completion->GetCompletedValue() < fenceValue) {
                dx(completion->SetEventOnCompletion(fenceValue, event.value), "SetEventOnCompletion(harness)");
                if (WaitForSingleObject(event.value, 30000) != WAIT_OBJECT_0) throw std::runtime_error("Harness GPU timeout");
            }
        };
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf())), "CreateCommandAllocator");
        dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(list.GetAddressOf())), "CreateCommandList");
        constexpr unsigned width = 256, height = 144, outWidth = 512, outHeight = 288;
        struct Texture { ComPtr<ID3D12Resource> image, upload; D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes; } textures[5];
        for (unsigned i = 0; i < 5; ++i) {
            auto& texture = textures[i];
            D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = i == 4 ? 1 : i == 3 ? outWidth : width;
            desc.Height = i == 4 ? 1 : i == 3 ? outHeight : height;
            desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
            desc.Format = i == 0 || i == 3 ? DXGI_FORMAT_R16G16B16A16_FLOAT : i == 2 ? DXGI_FORMAT_R16G16_FLOAT : DXGI_FORMAT_R32_FLOAT;
            if (args.pixel_profile == "packed") desc.Format = i == 0 || i == 3 ? DXGI_FORMAT_R11G11B10_FLOAT :
                i == 2 ? DXGI_FORMAT_R32G32B32A32_FLOAT : i == 4 ? DXGI_FORMAT_R16_FLOAT : DXGI_FORMAT_R32_FLOAT;
            if (args.pixel_profile == "unorm") desc.Format = i == 3 ? DXGI_FORMAT_B8G8R8A8_UNORM :
                i == 2 ? DXGI_FORMAT_R16G16_SNORM : i == 4 ? DXGI_FORMAT_R32G32B32A32_FLOAT : desc.Format;
            if (i == 3) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            heap.CreationNodeMask = heap.VisibleNodeMask = 1;
            dx(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr, IID_PPV_ARGS(texture.image.GetAddressOf())), "CreateCommittedResource(texture)");
            UINT rows; UINT64 rowBytes;
            device->GetCopyableFootprints(&desc, 0, 1, 0, &texture.footprint, &rows, &rowBytes, &texture.bytes);
            texture.upload = make_buffer(device.Get(), texture.bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            void* raw = nullptr; D3D12_RANGE noRead{};
            dx(texture.upload->Map(0, &noRead, &raw), "Map(synthetic initialization)");
            std::memset(raw, 0, size_t(texture.bytes));
            for (unsigned y = 0; y < desc.Height; ++y) for (unsigned x = 0; x < desc.Width; ++x) {
                auto* row = static_cast<uint8_t*>(raw) + y * texture.footprint.Footprint.RowPitch;
                if (i == 0) {
                    uint16_t pixel[] = {uint16_t(((x / 16 + y / 16) & 1) ? 0x3a00 : 0x3400),
                        uint16_t((x & 32) ? 0x3800 : 0x3400), uint16_t((y & 32) ? 0x3a00 : 0x3800), 0x3c00};
                    if (desc.Format == DXGI_FORMAT_R11G11B10_FLOAT) {
                        const uint32_t packed = uint32_t(pixel[0] >> 4) | (uint32_t(pixel[1] >> 4) << 11) | (uint32_t(pixel[2] >> 5) << 22);
                        std::memcpy(row + x * 4, &packed, 4);
                    } else std::memcpy(row + x * 8, pixel, sizeof(pixel));
                } else if (i == 1 || i == 4) {
                    if (desc.Format == DXGI_FORMAT_R16_FLOAT) { const uint16_t one = 0x3c00; std::memcpy(row + x * 2, &one, 2); }
                    else { const float value = i == 1 ? .5f : 1.f; std::memcpy(row + x * 4, &value, 4); }
                } else if (i == 3) {
                    if (desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) { const uint16_t pixel[] = {0x7e00, 0x7e00, 0x7e00, 0x7e00}; std::memcpy(row + x * 8, pixel, 8); }
                    else { const uint32_t sentinel = 0xffffffff; std::memcpy(row + x * 4, &sentinel, 4); }
                }
            }
            texture.upload->Unmap(0, nullptr);
            D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
            src.pResource = texture.upload.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = texture.footprint;
            dst.pResource = texture.image.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            transition(list.Get(), texture.image.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                (enhanced || inherited) ? D3D12_RESOURCE_STATE_COMMON : i == 3 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (enhanced && !inherited) enhancedBarrier(list.Get(), texture.image.Get(), i, true);
        }
        if (!inherited) dx(list->Close(), "Close(synthetic producers)"); ID3D12CommandList* producer[] = {list.Get()};
        if (!commandBackend) { queue->ExecuteCommandLists(1, producer); drain(); }
        if (args.interop_mode == "images") {
            CudaApi cuda(args.cuda_dll); cuda.check(cuda.cuInit(0), "cuInit(image interop)");
            CUdevice ordinal = 0; CUcontext context = nullptr;
            cuda.check(cuda.cuDeviceGet(&ordinal, 0), "cuDeviceGet(image interop)");
            cuda.check(cuda.cuDevicePrimaryCtxRetain(&context, ordinal), "cuDevicePrimaryCtxRetain(image interop)");
            struct ContextCleanup { CudaApi& cuda; CUdevice ordinal; ~ContextCleanup() { (void)cuda.cuCtxSetCurrent(nullptr); (void)cuda.cuDevicePrimaryCtxRelease_v2(ordinal); } } contextCleanup{cuda, ordinal};
            cuda.check(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent(image interop)");
            ExternalApi external{hip}; SharedTimeline timeline(external, device.Get());
            d4r::cuda::ImageApi images(cuda);
            std::unique_ptr<SharedPlane> planes[5];
            std::unique_ptr<d4r::cuda::Image> arrays[5];
            dx(allocator->Reset(), "Reset(image interop allocator)"); dx(list->Reset(allocator.Get(), nullptr), "Reset(image interop list)");
            for (unsigned i = 0; i < 5; ++i) {
                const auto desc = resource_desc(textures[i].image.Get());
                const unsigned channels = i == 0 || i == 3 ? 4 : i == 2 ? 2 : 1;
                const bool half = i == 0 || i == 2 || i == 3;
                planes[i] = std::make_unique<SharedPlane>(external, device.Get(), desc);
                arrays[i] = std::make_unique<d4r::cuda::Image>(images, unsigned(desc.Width), desc.Height, half ? 16 : 32, channels, i == 3, i == 0 ? 1 : 0);
                planes[i]->copy_input(list.Get(), textures[i].image.Get(), i == 3 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            dx(list->Close(), "Close(image interop input)"); ID3D12CommandList* copies[] = {list.Get()}; queue->ExecuteCommandLists(1, copies);
            timeline.wait_input(queue.Get());
            for (unsigned i = 0; i < 5; ++i) arrays[i]->upload_device(reinterpret_cast<uintptr_t>(planes[i]->mapped), planes[i]->footprint.Footprint.RowPitch);
            for (unsigned i = 0; i < 5; ++i) {
                const auto desc = resource_desc(textures[i].image.Get());
                const unsigned channels = i == 0 || i == 3 ? 4 : i == 2 ? 2 : 1;
                const bool half = i == 0 || i == 2 || i == 3;
                const size_t rowBytes = size_t(desc.Width) * channels * (half ? 2 : 4);
                std::vector<uint8_t> actual(rowBytes * desc.Height);
                arrays[i]->download(actual.data());
                void* expected = nullptr; D3D12_RANGE read{0, size_t(textures[i].bytes)};
                dx(textures[i].upload->Map(0, &read, &expected), "Map(expected diagnostic image)");
                bool matches = true;
                for (unsigned y = 0; y < desc.Height; ++y) if (std::memcmp(actual.data() + y * rowBytes,
                    static_cast<uint8_t*>(expected) + y * textures[i].footprint.Footprint.RowPitch, rowBytes)) { matches = false; break; }
                D3D12_RANGE noWrite{}; textures[i].upload->Unmap(0, &noWrite);
                std::printf("D3D12_ARRAY_COPY plane=%u format=%u matches=%u first=%02x%02x%02x%02x\n", i, desc.Format, matches, actual[0], actual[1], actual[2], actual[3]);
                if (!matches) throw std::runtime_error("Imported D3D12 VRAM to CUDA array mismatch on plane " + std::to_string(i));
                timeline.drain(queue.Get());
            }
            std::printf("PASS D3D12_ARRAY_COPY architecture=gfx1201 planes=5 cpu_copies_between_apis=0\n");
            return 0;
        }
        Library shim(wide(args.module));
        using Init = unsigned(*)(unsigned long long, const wchar_t*, ID3D12Device*, unsigned, const void*);
        using Allocate = unsigned(*)(void**); using Destroy = unsigned(*)(void*);
        using Create = unsigned(*)(ID3D12GraphicsCommandList*, unsigned, void*, void**);
        using Evaluate = unsigned(*)(ID3D12CommandQueue*, void*, void*);
        using Release = unsigned(*)(void*); using Shutdown = unsigned(*)();
        auto check = [](unsigned result, const char* call) { if (result != 1) throw std::runtime_error(std::string(call) + " NGX=" + std::to_string(result)); };
        d4r::ngx::FeatureDiscoveryInfo discovery{}; discovery.SDKVersion = 0x15; discovery.FeatureID = 1;
        d4r::ngx::FeatureRequirement requirements{};
        using Requirements = unsigned(*)(IDXGIAdapter*,const d4r::ngx::FeatureDiscoveryInfo*,d4r::ngx::FeatureRequirement*);
        check(shim.symbol<Requirements>("NVSDK_NGX_D3D12_GetFeatureRequirements")(adapter.Get(),&discovery,&requirements),"D3D12 feature requirements");
        if (requirements.FeatureSupported != 0) throw std::runtime_error("D3D12 DLSS requirements unsupported");
        if (args.ngx_abi == "project") {
            using ProjectInit = unsigned(*)(const char*,int,const char*,const wchar_t*,ID3D12Device*,unsigned,const void*);
            check(shim.symbol<ProjectInit>("NVSDK_NGX_D3D12_Init_ProjectID")("24480451-f00d-face-1304-0308dabad187",0,"1.0",directory.c_str(),device.Get(),0x15,nullptr),"D3D12 project Init");
        } else check(shim.symbol<Init>("NVSDK_NGX_D3D12_Init_Ext")(241534723ull, directory.c_str(), device.Get(), 0x15, nullptr), "D3D12 Init");
        auto shutdown = shim.symbol<Shutdown>("NVSDK_NGX_D3D12_Shutdown");
        struct ShutdownCleanup { Shutdown fn; ~ShutdownCleanup() { (void)fn(); } } shutdownCleanup{shutdown};
        void* parameters = nullptr; check(shim.symbol<Allocate>("NVSDK_NGX_D3D12_AllocateParameters")(&parameters), "AllocateParameters");
        auto destroy = shim.symbol<Destroy>("NVSDK_NGX_D3D12_DestroyParameters");
        struct ParameterCleanup { Destroy fn; void* p; ~ParameterCleanup() { (void)fn(p); } } parameterCleanup{destroy, parameters};
        for (const auto& pair : {std::make_pair("Width", width), std::make_pair("Height", height),
             std::make_pair("OutWidth", outWidth), std::make_pair("OutHeight", outHeight),
             std::make_pair("CreationNodeMask", 1u), std::make_pair("VisibilityNodeMask", 1u)}) d4r_ngx_set_uint(parameters, pair.first, pair.second);
        d4r_ngx_set_int(parameters, "PerfQualityValue", 2); d4r_ngx_set_int(parameters, "DLSS.Feature.Create.Flags", 0);
        for (const char* name : {"DLSS.Hint.Render.Preset.DLAA", "DLSS.Hint.Render.Preset.Quality", "DLSS.Hint.Render.Preset.Balanced",
             "DLSS.Hint.Render.Preset.Performance", "DLSS.Hint.Render.Preset.UltraPerformance", "DLSS.Hint.Render.Preset.UltraQuality"}) d4r_ngx_set_uint(parameters, name, args.preset);
        void* handle = nullptr;
        // Public NGX and OptiScaler require an open creation command list.
        // Submit it independently from the pending synthetic input producer.
        ComPtr<ID3D12CommandAllocator> creationAllocator;
        ComPtr<ID3D12GraphicsCommandList> creationList;
        dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(creationAllocator.GetAddressOf())), "Create feature allocator");
        dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, creationAllocator.Get(), nullptr,
            IID_PPV_ARGS(creationList.GetAddressOf())), "Create feature command list");
        check(shim.symbol<Create>("NVSDK_NGX_D3D12_CreateFeature")(creationList.Get(), 1, parameters, &handle), "D3D12 CreateFeature");
        dx(creationList->Close(), "Close feature creation list");
        ID3D12CommandList* creationBatch[] = {creationList.Get()}; queue->ExecuteCommandLists(1, creationBatch); drain();
        auto release = shim.symbol<Release>("NVSDK_NGX_D3D12_ReleaseFeature");
        struct FeatureCleanup { Release fn; void* h; ~FeatureCleanup() { (void)fn(h); } } featureCleanup{release, handle};
        static const char* names[] = {"Color", "Depth", "MotionVectors", "Output", "ExposureTexture"};
        for (unsigned i = 0; i < 5; ++i) d4r_ngx_set_d3d12_resource(parameters, names[i], textures[i].image.Get());
        if (enhanced && !commandBackend) {
            const char* states[] = {"D4R.Color.State", "D4R.Depth.State", "D4R.Motion.State", "D4R.Output.State", "D4R.Exposure.State"};
            for (unsigned i = 0; i < 5; ++i) { const std::string prefix(states[i]); d4r_ngx_set_uint(parameters,(prefix+".Enhanced").c_str(),1);
                d4r_ngx_set_uint(parameters,(prefix+".Layout").c_str(), i == 3 ? D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS : D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE);
                d4r_ngx_set_uint(parameters,(prefix+".Access").c_str(), i == 3 ? D3D12_BARRIER_ACCESS_UNORDERED_ACCESS : D3D12_BARRIER_ACCESS_SHADER_RESOURCE); }
        }
        d4r_ngx_set_int(parameters, "Disable.Watermark", 1);
        auto readback = make_buffer(device.Get(), textures[3].bytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        ComPtr<ID3D12GraphicsCommandList> predecessor;
        ComPtr<ID3D12CommandAllocator> predecessorAllocator;
        if (commandBackend) {
            if (inherited) {
                // Record state only in predecessor A after the backend has
                // initialized. List B's Evaluate must resolve it at submission,
                // after A, and retain it across subsequent frames/resets.
                for (unsigned i = 0; i < 5; ++i) {
                    if (enhanced) enhancedBarrier(list.Get(), textures[i].image.Get(), i, true);
                    else transition(list.Get(), textures[i].image.Get(), D3D12_RESOURCE_STATE_COMMON,
                        i == 3 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                }
                dx(list->Close(), "Close tracked predecessor A");
            }
            // Keep initial synthetic producers unsubmitted until the first
            // batch, before the list which contains the NGX evaluation.
            predecessor = list; predecessorAllocator = allocator;
            dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.ReleaseAndGetAddressOf())), "Create game-style allocator");
            dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                IID_PPV_ARGS(list.ReleaseAndGetAddressOf())), "Create game-style command list");
            dx(list->Close(), "Close initial game-style list");
        }
        for (unsigned frame = 0; frame < args.iterations; ++frame) {
            d4r_ngx_set_int(parameters, "Reset", frame == 0 ? 1 : 0);
            if (commandBackend) {
                dx(allocator->Reset(), "Reset(game-style allocator)"); dx(list->Reset(allocator.Get(), nullptr), "Reset(game-style list)");
                if (enhanced && !inherited) for (unsigned i = 0; i < 5; ++i) enhancedBarrier(list.Get(), textures[i].image.Get(), i, false);
                using RecordedEvaluate = unsigned(*)(ID3D12GraphicsCommandList*, void*, void*, void*);
                check(shim.symbol<RecordedEvaluate>("NVSDK_NGX_D3D12_EvaluateFeature")(list.Get(), handle, parameters, nullptr), "D3D12 recorded same-frame Evaluate");
                // The caller may reuse/mutate its parameter object as soon as
                // Evaluate returns. The queued snapshot must remain immutable.
                d4r_ngx_set_d3d12_resource(parameters, "Color", nullptr);
                d4r_ngx_set_int(parameters, "Reset", 0);
            } else {
                check(shim.symbol<Evaluate>("d4r_D3D12_EvaluateAtBoundary")(queue.Get(), handle, parameters), "D3D12 same-frame Evaluate");
                dx(allocator->Reset(), "Reset(harness allocator)"); dx(list->Reset(allocator.Get(), nullptr), "Reset(harness list)");
            }
            ResourceAccess outputAccess(D3D12_RESOURCE_STATE_UNORDERED_ACCESS); outputAccess.enhanced = enhanced;
            outputAccess.layout = D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS; outputAccess.access = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
            texture_copy_barrier(list.Get(), textures[3].image.Get(), outputAccess, false, false);
            D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
            src.pResource = textures[3].image.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.pResource = readback.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = textures[3].footprint;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            texture_copy_barrier(list.Get(), textures[3].image.Get(), outputAccess, false, true);
            dx(list->Close(), "Close(verification)");
            if (commandBackend && frame == 0) {
                ID3D12CommandList* batch[] = {predecessor.Get(), list.Get()}; queue->ExecuteCommandLists(2, batch);
            } else { ID3D12CommandList* verify[] = {list.Get()}; queue->ExecuteCommandLists(1, verify); }
            drain();
            if (commandBackend) d4r_ngx_set_d3d12_resource(parameters, "Color", textures[0].image.Get());
            void* raw = nullptr; D3D12_RANGE range{0, size_t(textures[3].bytes)};
            dx(readback->Map(0, &range, &raw), "Map(verification only)");
            std::vector<uint16_t> pixels(size_t(outWidth) * outHeight * 4);
            const auto format = resource_desc(textures[3].image.Get()).Format;
            if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                for (unsigned y = 0; y < outHeight; ++y) std::memcpy(pixels.data() + size_t(y) * outWidth * 4,
                    static_cast<uint8_t*>(raw) + y * textures[3].footprint.Footprint.RowPitch, outWidth * 8);
            } else {
                std::ofstream native(directory / ("output-" + std::to_string(frame) + (args.pixel_profile == "packed" ? ".r11g11b10f" : ".bgra8")), std::ios::binary);
                // Independently decode the diagnostic readback for previews and
                // frame comparisons. No CPU image copies are in the backend.
                static const auto unormHalf = [] {
                    std::array<uint16_t, 256> lut{};
                    for (unsigned b = 0; b < 256; ++b) { double best = INFINITY; for (unsigned h = 0; h <= 0x3c00; ++h) { double error = std::abs(double(half_value(uint16_t(h))) - double(b)/255.); if (error < best || (error == best && !(h & 1))) { best = error; lut[b] = uint16_t(h); } } }
                    return lut;
                }();
                for (unsigned y = 0; y < outHeight; ++y) {
                    const auto* row = static_cast<uint8_t*>(raw) + y * textures[3].footprint.Footprint.RowPitch;
                    native.write(reinterpret_cast<const char*>(row), outWidth * 4);
                    for (unsigned x = 0; x < outWidth; ++x) {
                        auto* p = pixels.data() + (size_t(y) * outWidth + x) * 4;
                        if (format == DXGI_FORMAT_R11G11B10_FLOAT) { uint32_t value = 0; std::memcpy(&value,row + x * 4,4); p[0] = uint16_t((value & 2047) << 4); p[1] = uint16_t(((value >> 11) & 2047) << 4); p[2] = uint16_t((value >> 22) << 5); p[3] = 0x3c00; }
                        else for (unsigned c = 0; c < 4; ++c) p[c] = unormHalf[row[x * 4 + (c == 0 ? 2 : c == 2 ? 0 : c)]];
                    }
                }
                if (!native) throw std::runtime_error("Saving native output format failed");
            }
            D3D12_RANGE noWrite{}; readback->Unmap(0, &noWrite);
            double sum = 0, square = 0;
            for (size_t i = 0; i < pixels.size(); ++i) {
                const float value = half_value(pixels[i]);
                if (!std::isfinite(value)) throw std::runtime_error("D3D12 output contains NaN/Inf or unwritten pixels");
                if ((i & 3) < 3) { sum += value; square += double(value) * value; }
            }
            const double count = outWidth * outHeight * 3, variance = square / count - (sum / count) * (sum / count);
            save_output(directory, pixels, outWidth, outHeight, frame);
            std::printf("D3D12_OUTPUT preset=%u frame=%u finite=1 mean=%.9g variance=%.9g frame_age=0\n", args.preset, frame, sum/count, variance);
            if (variance < 1e-6) throw std::runtime_error("D3D12 output lost the synthetic pattern");
        }
        loaded_modules();
        std::printf("PASS D3D12_DLSS architecture=gfx1201 preset=%u frames=%u fast_path_cpu_copies=0 frame_age=0 enhanced=%u queue_integration=%s\n",
            args.preset, args.iterations, enhanced, commandBackend ? "recorded_command_list" : "explicit_harness");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL D3D12_DLSS %s\n", error.what()); loaded_modules(); return 4; }
}
