#include "d3d12_external.h"
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <array>

using namespace d4r::diag;
using namespace d4r::win;
namespace {
constexpr unsigned count = 256, bytes = count * sizeof(uint32_t), seed = 0x12345;
void drain(ID3D12Device* device, ID3D12CommandQueue* queue) {
    ComPtr<ID3D12Fence> fence; dx(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())), "Probe fence");
    Handle event; event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event.value) throw std::runtime_error("Probe event");
    dx(queue->Signal(fence.Get(), 1), "Probe signal");
    if (fence->GetCompletedValue() < 1) {
        dx(fence->SetEventOnCompletion(1, event.value), "Probe fence event");
        if (WaitForSingleObject(event.value, 30000) != WAIT_OBJECT_0) throw std::runtime_error("Probe GPU timeout");
    }
}
struct NativeList {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    explicit NativeList(ID3D12Device* device) {
        dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf())), "Probe allocator");
        dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(list.GetAddressOf())), "Probe command list");
    }
    void submit(ID3D12Device* device, ID3D12CommandQueue* queue) {
        dx(list->Close(), "Close probe list"); ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists); drain(device, queue);
    }
};
ComPtr<ID3D12Resource> output_buffer(ID3D12Device* device) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT; heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> result;
    dx(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr, IID_PPV_ARGS(result.GetAddressOf())), "Probe UAV");
    return result;
}
void upload(ID3D12Resource* buffer, uint32_t multiplier, uint32_t addend) {
    void* data = nullptr; D3D12_RANGE noRead{}; dx(buffer->Map(0, &noRead, &data), "Probe upload map");
    auto* values = static_cast<uint32_t*>(data);
    for (unsigned i = 0; i < count; ++i) values[i] = i * multiplier + addend;
    buffer->Unmap(0, nullptr);
}
bool verify(ID3D12Resource* buffer, uint32_t multiplier, uint32_t addend) {
    void* data = nullptr; D3D12_RANGE range{0, bytes}; dx(buffer->Map(0, &range, &data), "Probe verification map");
    const auto* values = static_cast<const uint32_t*>(data);
    bool passed = true;
    for (unsigned i = 0; i < count; ++i) if (values[i] != i * multiplier + addend + seed) {
        std::fprintf(stderr, "COMMAND_DIFFERENT index=%u expected=%u actual=%u\n", i, i * multiplier + addend + seed, values[i]);
        passed = false; break;
    }
    D3D12_RANGE noWrite{}; buffer->Unmap(0, &noWrite); return passed;
}
struct Callback {
    ID3D12Device* device;
    ID3D12Resource* input;
    ID3D12Resource* output;
    ID3D12Resource* replacement;
    ID3D12Resource* readback;
    bool prefix_verified = false;
};
void WINAPI boundary(ID3D12CommandQueue* queue, void* context) {
    auto& callback = *static_cast<Callback*>(context);
    NativeList list(callback.device);
    transition(list.list.Get(), callback.output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list.list->CopyResource(callback.readback, callback.output);
    transition(list.list.Get(), callback.output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(list.list.Get(), callback.input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    list.list->CopyResource(callback.input, callback.replacement);
    transition(list.list.Get(), callback.input, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list.submit(callback.device, queue);
    callback.prefix_verified = verify(callback.readback, 17, 10);
}
}

int main(int argc, char** argv) {
    start();
    try {
        Args args(argc, argv);
        if (args.module.empty()) throw std::runtime_error("--module d4r_nvngx.dll is required; NVIDIA DLLs are not needed");
        HipApi hip(args.hip_root); hipDeviceProp_t props{}; hip.select_gfx1201(args.device, props);
        ComPtr<IDXGIFactory4> factory; dx(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())), "Probe DXGI");
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; ; ++i) {
            dx(factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()), "Probe adapter");
            DXGI_ADAPTER_DESC1 desc{}; dx(adapter->GetDesc1(&desc), "Probe adapter desc");
            if (!std::memcmp(&desc.AdapterLuid, props.luid, sizeof(LUID))) break;
        }
        ComPtr<ID3D12Device> device;
        dx(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.GetAddressOf())), "Probe D3D12");
        ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        dx(device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.GetAddressOf())), "Probe queue");
        wchar_t system[32768]{}; if (!GetSystemDirectoryW(system, 32768)) throw std::runtime_error("System directory");
        Library compiler(std::filesystem::path(system) / L"d3dcompiler_47.dll");
        auto compile = compiler.symbol<decltype(&D3DCompile)>("D3DCompile");
        static constexpr char shader[] = "RWStructuredBuffer<uint> Output : register(u0); StructuredBuffer<uint> Input : register(t0); cbuffer Constants : register(b0) { uint Seed; }; [numthreads(64,1,1)] void main(uint3 t:SV_DispatchThreadID) { Output[t.x] = Input[t.x] + Seed; }";
        ComPtr<ID3DBlob> code, errors;
        dx(compile(shader, sizeof(shader)-1, nullptr, nullptr, nullptr, "main", "cs_5_1", D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0, code.GetAddressOf(), errors.GetAddressOf()), "Probe HLSL");
        D3D12_ROOT_PARAMETER roots[3]{};
        roots[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; roots[0].Descriptor.ShaderRegister = 0;
        roots[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; roots[1].Descriptor.ShaderRegister = 0;
        roots[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; roots[2].Constants.Num32BitValues = 1;
        D3D12_ROOT_SIGNATURE_DESC signature{}; signature.NumParameters = 3; signature.pParameters = roots;
        ComPtr<ID3DBlob> serialized;
        dx(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, serialized.GetAddressOf(), errors.ReleaseAndGetAddressOf()), "Probe root signature serialize");
        ComPtr<ID3D12RootSignature> root;
        dx(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(root.GetAddressOf())), "Probe root signature");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = root.Get(); pipeline.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        ComPtr<ID3D12PipelineState> pso; dx(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(pso.GetAddressOf())), "Probe PSO");
        Library shim(wide(args.module));
        using Install = unsigned(*)(ID3D12Device*);
        using Record = unsigned(*)(ID3D12GraphicsCommandList*, void(WINAPI*)(ID3D12CommandQueue*, void*), void*);
        using Live = unsigned long long(*)();
        if (shim.symbol<Install>("d4r_D3D12_InstallCommandBackend")(device.Get()) != 1) throw std::runtime_error("Command backend installation failed");
        auto record = shim.symbol<Record>("d4r_D3D12_RecordDiagnosticBoundary");
        auto live = shim.symbol<Live>("d4r_D3D12_LiveRecordings");
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            auto input = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            auto output = output_buffer(device.Get());
            auto first = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            auto second = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            auto readback = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            auto prefix = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            upload(first.Get(), 17, 10); upload(second.Get(), 25, 100);
            {
                NativeList a(device.Get()), b(device.Get()), c(device.Get());
                a.list->CopyResource(input.Get(), first.Get());
                transition(a.list.Get(), input.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                dx(a.list->Close(), "Close predecessor A");
                b.list->SetComputeRootSignature(root.Get()); b.list->SetPipelineState(pso.Get());
                b.list->SetComputeRootUnorderedAccessView(0, output->GetGPUVirtualAddress());
                b.list->SetComputeRootShaderResourceView(1, input->GetGPUVirtualAddress());
                uint32_t temporary = seed;
                b.list->SetComputeRoot32BitConstants(2, 1, &temporary, 0);
                temporary = 0xdeadbeef; // Deep-copy check: replay must retain Seed.
                b.list->Dispatch(count / 64, 1, 1);
                Callback callback{device.Get(), input.Get(), output.Get(), second.Get(), prefix.Get()};
                if (record(b.list.Get(), boundary, &callback) != 1) throw std::runtime_error("Record split failed");
                b.list->Dispatch(count / 64, 1, 1); // No rebinding after the split.
                dx(b.list->Close(), "Close logical list B");
                transition(c.list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                c.list->CopyResource(readback.Get(), output.Get());
                transition(c.list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                dx(c.list->Close(), "Close consumer C");
                ID3D12CommandList* batch[] = {a.list.Get(), b.list.Get(), c.list.Get()};
                queue->ExecuteCommandLists(3, batch); drain(device.Get(), queue.Get());
                if (!callback.prefix_verified || !verify(readback.Get(), 25, 100)) throw std::runtime_error("Queue order or suffix root state mismatch");
            }
            if (live() != 0) throw std::runtime_error("D3D12 recording metadata leaked after command-list Release");
        }
        std::printf("PASS D3D12_COMMAND_BACKEND architecture=gfx1201 iterations=%u batch_order=1 root_state=1 deep_copy=1 live_recordings=0 frame_age=0\n", args.iterations);
        return 0;
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "FAIL D3D12_COMMAND_BACKEND %s\n", failure.what()); loaded_modules(); return 4;
    }
}
