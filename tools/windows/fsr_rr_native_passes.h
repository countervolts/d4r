#pragma once
#include "d3d12_external.h"
#include "fsr_rr_constants.h"
#include "fsr_rr_shaders_hlsl.generated.h"
#include <d3dcompiler.h>
#include <utility>

namespace d4r::win::rr {
// All descriptors belong to one synchronous evaluation. The owner retires the
// submission before rewriting this heap or releasing any referenced resource.
struct NativeFsrPasses {
    struct Guide {
        ComPtr<ID3D12Resource> resource;
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    };
    struct Pipeline {
        ComPtr<ID3D12RootSignature> root;
        ComPtr<ID3D12PipelineState> state;
        UINT constants = 0, srvs = 0, uavs = 0, offset = 0;
    } prepare, compose, widen;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12DescriptorHeap> heap;
    UINT stride = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu{};

    void pipeline(Pipeline& p, const char* source, UINT constants, UINT srvs, UINT uavs, UINT offset) {
        p.constants = constants; p.srvs = srvs; p.uavs = uavs; p.offset = offset;
        // Load the OS compiler by absolute system search, never from the game directory.
        static HMODULE compiler = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        auto compile = compiler ? reinterpret_cast<decltype(&D3DCompile)>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
        if (!compile) throw std::runtime_error("Native FSR RR requires Windows d3dcompiler_47.dll");
        ComPtr<ID3DBlob> code, errors;
        HRESULT result = compile(source, std::strlen(source), "d4r-fsr-rr", nullptr, nullptr, "main", "cs_5_1",
            D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS, 0, &code, &errors);
        if (FAILED(result)) throw std::runtime_error(std::string("FSR RR native shader: ") +
            (errors ? static_cast<const char*>(errors->GetBufferPointer()) : "compilation failed"));
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, constants};
        UINT count = 1, range = 0;
        for (auto binding : {std::pair<D3D12_DESCRIPTOR_RANGE_TYPE, UINT>{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, srvs},
                            {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, uavs}}) {
            if (!binding.second) continue;
            ranges[range] = {binding.first, binding.second, 0, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
            params[count].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            params[count].DescriptorTable = {1, &ranges[range]};
            ++count; ++range;
        }
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters = count; desc.pParameters = params;
        ComPtr<ID3DBlob> serialized;
        dx(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors), "FSR RR root serialization");
        dx(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&p.root)), "FSR RR root signature");
        D3D12_COMPUTE_PIPELINE_STATE_DESC state{}; state.pRootSignature = p.root.Get();
        state.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        dx(device->CreateComputePipelineState(&state, IID_PPV_ARGS(&p.state)), "FSR RR native pipeline");
    }
    void initialize(ID3D12Device* owner) {
        device = owner;
        for (auto format : {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT}) {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{}; support.Format = format;
            dx(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)), "FSR RR native format support");
            const UINT wanted = D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
            if ((support.Support2 & wanted) != wanted) throw std::runtime_error("FSR RR native typed UAV formats are unsupported");
        }
        D3D12_DESCRIPTOR_HEAP_DESC desc{}; desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = 28; desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        dx(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)), "FSR RR native descriptor heap");
        stride = device->GetDescriptorHandleIncrementSize(desc.Type);
#if defined(__MINGW32__)
        heap->GetCPUDescriptorHandleForHeapStart(&cpu); heap->GetGPUDescriptorHandleForHeapStart(&gpu);
#else
        cpu = heap->GetCPUDescriptorHandleForHeapStart(); gpu = heap->GetGPUDescriptorHandleForHeapStart();
#endif
        pipeline(prepare, d4r_hlsl_fsr_rr_prepare, sizeof(D4rFsrRrPrepareConstants) / 4, 10, 8, 0);
        pipeline(compose, d4r_hlsl_fsr_rr_compose, 4, 0, 8, 18);
        const char* widening = R"HLSL(
            Texture2D<float4> source : register(t0);
            RWTexture2D<float4> destination : register(u0);
            cbuffer Constants : register(b0) { uint2 extent; }
            [numthreads(8,8,1)] void main(uint3 p : SV_DispatchThreadID) {
                if (all(p.xy < extent)) destination[p.xy] = source.Load(int3(p.xy, 0));
            })HLSL";
        pipeline(widen, widening, 2, 1, 1, 26);
    }
    static DXGI_FORMAT guide_format(DXGI_FORMAT format, bool depth) {
        if (depth) switch (format) {
            case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
            case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
            case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
            case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
            default: return format;
        }
        switch (format) {
            case DXGI_FORMAT_R8_TYPELESS: return DXGI_FORMAT_R8_UNORM;
            case DXGI_FORMAT_R8G8_TYPELESS: return DXGI_FORMAT_R8G8_UNORM;
            case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM;
            case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM;
            case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_FLOAT;
            case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
            case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
            case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
            case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
            case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
            default: return format;
        }
    }
    void copy_guide(ID3D12GraphicsCommandList* list, ID3D12Resource* source, const ResourceAccess& access,
                    bool depth, UINT width, UINT height, UINT maxWidth, UINT maxHeight, UINT x, UINT y, Guide& guide) {
        // Copy into typed, shader-readable textures on the same device. This also
        // supports game depth resources created with DENY_SHADER_RESOURCE.
        const auto desc = resource_desc(source);
        const auto format = guide_format(desc.Format, depth);
        if (!guide.resource || guide.format != format) {
            guide.resource.Reset(); guide.format = format;
            D3D12_RESOURCE_DESC target{}; target.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            target.Width = maxWidth; target.Height = maxHeight; target.DepthOrArraySize = target.MipLevels = 1;
            target.Format = format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS ? DXGI_FORMAT_R32G8X24_TYPELESS : format;
            target.SampleDesc.Count = 1;
            D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            dx(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &target,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&guide.resource)), "FSR RR native guide texture");
        }
        transition(list, guide.resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        texture_copy_barrier(list, source, access, false, false);
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = source; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.pResource = guide.resource.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_BOX box{x, y, 0, x + width, y + height, 1};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) ? nullptr : &box);
        texture_copy_barrier(list, source, access, false, true);
        transition(list, guide.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    void record(ID3D12GraphicsCommandList* list, const Pipeline& p, const void* constants,
                ID3D12Resource* const* resources, const DXGI_FORMAT* formats, UINT width, UINT height) {
        for (UINT i = 0; i < p.srvs + p.uavs; ++i) {
            auto handle = cpu; handle.ptr += SIZE_T(p.offset + i) * stride;
            if (i < p.srvs) {
                D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format = formats[i];
                view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels = 1;
                device->CreateShaderResourceView(resources[i], &view, handle);
            } else {
                D3D12_UNORDERED_ACCESS_VIEW_DESC view{}; view.Format = formats[i]; view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                device->CreateUnorderedAccessView(resources[i], nullptr, &view, handle);
            }
        }
        ID3D12DescriptorHeap* heaps[] = {heap.Get()}; list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(p.root.Get()); list->SetPipelineState(p.state.Get());
        list->SetComputeRoot32BitConstants(0, p.constants, constants, 0);
        UINT table = 1; auto handle = gpu; handle.ptr += UINT64(p.offset) * stride;
        if (p.srvs) { list->SetComputeRootDescriptorTable(table++, handle); handle.ptr += UINT64(p.srvs) * stride; }
        list->SetComputeRootDescriptorTable(table, handle);
        list->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        list->ResourceBarrier(1, &barrier);
    }
};
}
