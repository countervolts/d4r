#include "d3d12_external.h"
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <d3d12sdklayers.h>
#include <array>

using namespace d4r::diag;
using namespace d4r::win;
#if defined(D4R_DIAGNOSTIC_AGILITY)
extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = 619;
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\debug-d3d12\\";
}
#endif
namespace {
constexpr unsigned count = 256, bytes = count * sizeof(uint32_t), seed = 0x12345;
void messages(ID3D12Device* device) {
    ComPtr<ID3D12InfoQueue> info;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(info.GetAddressOf())))) return;
    for (UINT64 i = 0; i < info->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
        SIZE_T size = 0; (void)info->GetMessage(i, nullptr, &size);
        std::vector<uint8_t> buffer(size);
        auto* value = reinterpret_cast<D3D12_MESSAGE*>(buffer.data());
        if (SUCCEEDED(info->GetMessage(i, value, &size)))
            std::fprintf(stderr, "D3D12_VALIDATION severity=%u id=%u %s\n", unsigned(value->Severity), unsigned(value->ID), value->pDescription);
    }
    info->ClearStoredMessages();
}
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
bool verify(ID3D12Resource* buffer, uint32_t multiplier, uint32_t addend, uint32_t root_seed = seed) {
    void* data = nullptr; D3D12_RANGE range{0, bytes}; dx(buffer->Map(0, &range, &data), "Probe verification map");
    const auto* values = static_cast<const uint32_t*>(data);
    bool passed = true;
    for (unsigned i = 0; i < count; ++i) if (values[i] != i * multiplier + addend + root_seed) {
        std::fprintf(stderr, "COMMAND_DIFFERENT index=%u expected=%u actual=%u\n", i, i * multiplier + addend + root_seed, values[i]);
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
    uint32_t expected_seed = seed;
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
    callback.prefix_verified = verify(callback.readback, 17, 10, callback.expected_seed);
}
}

int main(int argc, char** argv) {
    start();
    ComPtr<ID3D12Device> device;
    std::unique_ptr<Library> debug_layers;
    try {
        Args args(argc, argv);
        if (args.module.empty()) throw std::runtime_error("--module d4r_nvngx.dll is required; NVIDIA DLLs are not needed");
        if (std::getenv("D4R_COMMAND_DEBUG_SDK")) {
            wchar_t exe[32768]{}; GetModuleFileNameW(nullptr, exe, 32768);
            debug_layers = std::make_unique<Library>(std::filesystem::path(exe).parent_path() / L"debug-d3d12/d3d12SDKLayers.dll");
            ComPtr<ID3D12Debug> debug;
            auto get_interface = reinterpret_cast<decltype(&D3D12GetInterface)>(GetProcAddress(GetModuleHandleW(L"d3d12.dll"), "D3D12GetInterface"));
            if (!get_interface) throw std::runtime_error("D3D12GetInterface is unavailable");
            dx(get_interface(CLSID_D3D12Debug, IID_PPV_ARGS(debug.GetAddressOf())), "Diagnostic debug layer(public CLSID)");
            debug->EnableDebugLayer();
        }
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred_settings;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(dred_settings.GetAddressOf())))) {
            dred_settings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred_settings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        }
        std::unique_ptr<HipApi> hip;
        hipDeviceProp_t props{};
        const bool no_hip = std::getenv("D4R_COMMAND_PROBE_NO_HIP") != nullptr;
        if (!no_hip) { hip = std::make_unique<HipApi>(args.hip_root); hip->select_gfx1201(args.device, props); }
        ComPtr<IDXGIFactory4> factory; dx(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())), "Probe DXGI");
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; ; ++i) {
            dx(factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()), "Probe adapter");
            DXGI_ADAPTER_DESC1 desc{}; dx(adapter->GetDesc1(&desc), "Probe adapter desc");
            if (no_hip ? desc.VendorId == 0x1002 : !std::memcmp(&desc.AdapterLuid, props.luid, sizeof(LUID))) break;
        }
        if (std::getenv("D4R_COMMAND_PROBE_WARP")) {
            dx(factory->EnumWarpAdapter(IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf())), "Probe WARP adapter");
        }
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
        const bool no_hooks = std::getenv("D4R_COMMAND_PROBE_NO_HOOKS") != nullptr;
        if (!no_hooks && shim.symbol<Install>("d4r_D3D12_InstallCommandBackend")(device.Get()) != 1) throw std::runtime_error("Command backend installation failed");
        auto record = shim.symbol<Record>("d4r_D3D12_RecordDiagnosticBoundary");
        auto live = shim.symbol<Live>("d4r_D3D12_LiveRecordings");
        const bool root_indirect = args.interop_mode == "indirect-root-reset";
        const bool indirect_enabled = args.interop_mode != "basic";
        if (args.interop_mode != "direct" && args.interop_mode != "basic" && args.interop_mode != "indirect" && !root_indirect)
            throw std::runtime_error("Command probe mode must be basic, indirect, or indirect-root-reset");
        D3D12_INDIRECT_ARGUMENT_DESC arguments[2]{};
        arguments[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        arguments[0].Constant.RootParameterIndex = 2;
        arguments[0].Constant.Num32BitValuesToSet = 1;
        arguments[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        D3D12_COMMAND_SIGNATURE_DESC indirect_desc{};
        indirect_desc.ByteStride = (root_indirect ? 4 : 3) * sizeof(uint32_t);
        indirect_desc.NumArgumentDescs = root_indirect ? 2 : 1;
        indirect_desc.pArgumentDescs = arguments + (root_indirect ? 0 : 1);
        ComPtr<ID3D12CommandSignature> indirect;
        dx(device->CreateCommandSignature(&indirect_desc, root_indirect ? root.Get() : nullptr, IID_PPV_ARGS(indirect.GetAddressOf())), "Probe indirect signature");
        auto indirect_upload = make_buffer(device.Get(), 4 * sizeof(uint32_t), D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto indirect_buffer = make_buffer(device.Get(), 4 * sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        void* mapped = nullptr; D3D12_RANGE no_read{};
        dx(indirect_upload->Map(0, &no_read, &mapped), "Probe indirect arguments");
        const uint32_t indirect_seed = std::getenv("D4R_COMMAND_PROBE_SMALL_SEED") ? 1u : seed;
        const uint32_t dispatch[] = {indirect_seed, count / 64, 1, 1};
        std::memcpy(mapped, dispatch, sizeof(dispatch)); indirect_upload->Unmap(0, nullptr);
        {
            NativeList copy(device.Get()); copy.list->CopyResource(indirect_buffer.Get(), indirect_upload.Get());
            transition(copy.list.Get(), indirect_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            copy.submit(device.Get(), queue.Get());
        }
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) {
            auto input = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            auto output = output_buffer(device.Get());
            auto first = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            auto second = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            auto readback = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            auto prefix = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            input->SetName(L"Probe input"); output->SetName(L"Probe output"); indirect_buffer->SetName(L"Probe indirect");
            std::printf("PROBE_GPU iteration=%u input=0x%llx output=0x%llx indirect=0x%llx\n", iteration,
                input->GetGPUVirtualAddress(), output->GetGPUVirtualAddress(), indirect_buffer->GetGPUVirtualAddress());
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
                const uint32_t expected_seed = root_indirect && iteration % 2 ? 0 : seed;
                if (indirect_enabled && iteration % 2) {
                    b.list->ExecuteIndirect(indirect.Get(), 1, indirect_buffer.Get(), root_indirect ? 0 : sizeof(uint32_t), nullptr, 0);
                    D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; uav.UAV.pResource = output.Get();
                    b.list->ResourceBarrier(1, &uav);
                    // Only the root-mutating signature resets Seed to zero.
                    // A dispatch-only signature must preserve every binding.
                }
                b.list->Dispatch(count / 64, 1, 1);
                Callback callback{device.Get(), input.Get(), output.Get(), second.Get(), prefix.Get(), expected_seed};
                if (no_hooks) {
                    dx(b.list->Close(), "Close native prefix");
                    messages(device.Get());
                    ID3D12CommandList* prefix_batch[] = {a.list.Get(), b.list.Get()};
                    queue->ExecuteCommandLists(2, prefix_batch); drain(device.Get(), queue.Get());
                    boundary(queue.Get(), &callback);
                    dx(b.allocator->Reset(), "Reset native prefix allocator");
                    dx(b.list->Reset(b.allocator.Get(), pso.Get()), "Reset native suffix");
                    b.list->SetComputeRootSignature(root.Get());
                    b.list->SetComputeRootUnorderedAccessView(0, output->GetGPUVirtualAddress());
                    b.list->SetComputeRootShaderResourceView(1, input->GetGPUVirtualAddress());
                    b.list->SetComputeRoot32BitConstant(2, expected_seed, 0);
                } else if (record(b.list.Get(), boundary, &callback) != 1) throw std::runtime_error("Record split failed");
                b.list->Dispatch(count / 64, 1, 1); // No rebinding after the split.
                dx(b.list->Close(), "Close logical list B");
                transition(c.list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                c.list->CopyResource(readback.Get(), output.Get());
                transition(c.list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                dx(c.list->Close(), "Close consumer C");
                messages(device.Get());
                ID3D12CommandList* batch[] = {a.list.Get(), b.list.Get(), c.list.Get()};
                queue->ExecuteCommandLists(no_hooks ? 2 : 3, batch + (no_hooks ? 1 : 0)); drain(device.Get(), queue.Get());
                if (!callback.prefix_verified || !verify(readback.Get(), 25, 100, expected_seed)) throw std::runtime_error("Queue order or suffix root state mismatch");
            }
            if (live() != 0) throw std::runtime_error("D3D12 recording metadata leaked after command-list Release");
        }
        const bool warp = std::getenv("D4R_COMMAND_PROBE_WARP") != nullptr;
        std::printf("PASS D3D12_COMMAND_BACKEND adapter=%s architecture=%s backend=%u iterations=%u batch_order=1 root_state=1 deep_copy=1 indirect=%u indirect_reset=%u live_recordings=0 frame_age=0\n",
            warp ? "WARP" : "AMD", warp ? "software" : no_hip ? "not_queried" : "gfx1201", !no_hooks,
            args.iterations, indirect_enabled, root_indirect);
        return 0;
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "FAIL D3D12_COMMAND_BACKEND %s\n", failure.what());
        if (device) {
            messages(device.Get());
            std::fprintf(stderr, "DEVICE_REMOVED result=0x%08x\n", unsigned(device->GetDeviceRemovedReason()));
            ComPtr<ID3D12DeviceRemovedExtendedData> dred;
            if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(dred.GetAddressOf())))) {
                D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT breadcrumbs{};
                if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&breadcrumbs)))
                    for (auto* node = breadcrumbs.pHeadAutoBreadcrumbNode; node; node = node->pNext) {
                        UINT completed = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0;
                        std::fprintf(stderr, "DRED list=%p completed=%u total=%u\n", node->pCommandList, completed, node->BreadcrumbCount);
                        for (UINT i = completed > 2 ? completed - 2 : 0; i < std::min(node->BreadcrumbCount, completed + 3); ++i)
                            std::fprintf(stderr, "DRED command=%u operation=%u\n", i, unsigned(node->pCommandHistory[i]));
                    }
                D3D12_DRED_PAGE_FAULT_OUTPUT fault{};
                if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&fault)))
                    std::fprintf(stderr, "DRED fault_address=0x%llx\n", fault.PageFaultVA);
                for (auto* allocation = fault.pHeadExistingAllocationNode; allocation; allocation = allocation->pNext)
                    std::fprintf(stderr, "DRED existing=%ls type=%u\n", allocation->ObjectNameW ? allocation->ObjectNameW : L"(unnamed)", unsigned(allocation->AllocationType));
                for (auto* allocation = fault.pHeadRecentFreedAllocationNode; allocation; allocation = allocation->pNext)
                    std::fprintf(stderr, "DRED freed=%ls type=%u\n", allocation->ObjectNameW ? allocation->ObjectNameW : L"(unnamed)", unsigned(allocation->AllocationType));
            }
        }
        loaded_modules(); return 4;
    }
}
