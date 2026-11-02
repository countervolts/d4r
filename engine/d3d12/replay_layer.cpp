// Standalone plain-K layer replay. The same executable can run on Windows or
// through Wine/vkd3d-proton. GPU timestamps exclude upload, readback and CPU waits.
// d4r-k-layer SHADER PARAMS WEIGHTS INPUT OUTPUT GX GY REPEATS [--debug] [--wmma]
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
static void check(HRESULT hr, const char* operation)
{
    if (FAILED(hr)) {
        char error[256];
        std::snprintf(error, sizeof(error), "%s: HRESULT 0x%08lx", operation, static_cast<unsigned long>(hr));
        throw std::runtime_error(error);
    }
}
#define CK(call) check((call), #call)
static std::vector<char> read(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot read " + path.string());
    const auto size = file.tellg();
    if (size <= 0 || size > 512 * 1024 * 1024) throw std::runtime_error("invalid file size: " + path.string());
    std::vector<char> result(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(result.data(), size)) throw std::runtime_error("short read: " + path.string());
    return result;
}
static unsigned number(const char* text, unsigned maximum)
{
    size_t end = 0;
    const std::string s(text);
    const auto n = std::stoul(s, &end);
    if (end != s.size() || n == 0 || n > maximum) throw std::runtime_error("invalid positive integer: " + s);
    return static_cast<unsigned>(n);
}
struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Event() { if (!handle) throw std::runtime_error("CreateEvent failed"); }
    ~Event() { CloseHandle(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct Runner {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> validation;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    ComPtr<ID3D12QueryHeap> queries;
    ComPtr<ID3D12Resource> timing;
    std::array<ComPtr<ID3D12Resource>, 4> buffers, uploads;
    ComPtr<ID3D12Resource> output;
    Event event;
    UINT64 fenceValue = 0, frequency = 0;
    UINT64 outputBytes = 0;
    bool wmmaMode = false;

    ComPtr<ID3D12Resource> buffer(UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                                 D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
    {
        D3D12_HEAP_PROPERTIES properties{}; properties.Type = heap;
        properties.CreationNodeMask = properties.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = bytes;
        desc.Height = 1; desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = flags;
        ComPtr<ID3D12Resource> result;
        CK(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&result)));
        return result;
    }
    void begin()
    {
        CK(allocator->Reset()); CK(list->Reset(allocator.Get(), nullptr));
    }
    void submit()
    {
        CK(list->Close());
        ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
        CK(queue->Signal(fence.Get(), ++fenceValue));
        CK(fence->SetEventOnCompletion(fenceValue, event.handle));
        if (WaitForSingleObject(event.handle, 30000) != WAIT_OBJECT_0) {
            check(device->GetDeviceRemovedReason(), "GPU wait / device removed");
            throw std::runtime_error("GPU completion timed out");
        }
        CK(device->GetDeviceRemovedReason());
        if (validation) {
            const auto count = validation->GetNumStoredMessagesAllowedByRetrievalFilter();
            for (UINT64 i = 0; i < count; ++i) {
                SIZE_T bytes = 0; CK(validation->GetMessage(i, nullptr, &bytes));
                std::vector<char> storage(bytes);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                CK(validation->GetMessage(i, message, &bytes));
                if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR || message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION)
                    throw std::runtime_error(std::string("D3D12 validation: ") + message->pDescription);
            }
            validation->ClearStoredMessages();
        }
    }
    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        list->ResourceBarrier(1, &barrier);
    }
    void create(const std::vector<char>& shader, const std::array<std::vector<char>, 3>& data, bool debug, bool wmma)
    {
        wmmaMode = wmma;
        if (debug) {
            ComPtr<ID3D12Debug> layer;
            CK(D3D12GetDebugInterface(IID_PPV_ARGS(&layer))); layer->EnableDebugLayer();
        }
        ComPtr<IDXGIFactory4> factory; CK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT result = factory->EnumAdapters1(i, &candidate);
            if (result == DXGI_ERROR_NOT_FOUND) break;
            CK(result);
            DXGI_ADAPTER_DESC1 desc{}; CK(candidate->GetDesc1(&desc));
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (wmma && desc.VendorId != 0x1002) continue;
            if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
                adapter = candidate;
                std::printf("Adapter: %ls (vendor %04x, device %04x)\n", desc.Description, desc.VendorId, desc.DeviceId);
                break;
            }
        }
        if (!adapter) throw std::runtime_error("no hardware D3D12 adapter (software adapters are excluded)");
        if (debug) CK(device.As(&validation));
        D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_6};
        CK(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm)));
        if (sm.HighestShaderModel < D3D_SHADER_MODEL_6_6) throw std::runtime_error("Shader Model 6.6 required");
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        CK(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)));
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
        CK(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)));
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4{};
        CK(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4)));
        if (!options1.WaveOps || options1.WaveLaneCountMin > 32 || options1.WaveLaneCountMax < 32 ||
            !options4.Native16BitShaderOpsSupported || !(options.MinPrecisionSupport & D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT))
            throw std::runtime_error("wave32 and native f16 shader operations required");
        D3D12_COMMAND_QUEUE_DESC q{}; q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CK(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
        CK(queue->GetTimestampFrequency(&frequency));
        CK(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&allocator)));
        CK(device->CreateCommandList(0, q.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
        CK(list->Close()); CK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0};
        ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 3};
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable = {2, ranges}; parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_PARAMETER rootParameters[2] = {parameter, {}};
        rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParameters[1].Descriptor = {0, 0x7fff0ade};
        rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rootDesc{};
        rootDesc.NumParameters = wmma ? 2 : 1; rootDesc.pParameters = rootParameters;
        ComPtr<ID3DBlob> blob, error;
        const HRESULT serialized = D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
        if (FAILED(serialized) && error) std::fprintf(stderr, "%s\n", static_cast<const char*>(error->GetBufferPointer()));
        CK(serialized); CK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        D3D12_COMPUTE_PIPELINE_STATE_DESC ps{}; ps.pRootSignature = root.Get(); ps.CS = {shader.data(), shader.size()};
        CK(device->CreateComputePipelineState(&ps, IID_PPV_ARGS(&pipeline)));
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 4; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CK(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&descriptors)));
        const UINT increment = device->GetDescriptorHandleIncrementSize(hd.Type);
        auto handle = descriptors->GetCPUDescriptorHandleForHeapStart();
        outputBytes = data[2].size();
        begin();
        for (size_t i = 0; i < buffers.size(); ++i) {
            const UINT64 bytes = i == 3 ? outputBytes : data[i].size();
            if (bytes == 0 || bytes % 4 != 0 || bytes / 4 > std::numeric_limits<UINT>::max())
                throw std::runtime_error("buffer sizes must be positive DWORD multiples");
            buffers[i] = buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                                i == 3 ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE);
            uploads[i] = buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            void* mapped = nullptr; const D3D12_RANGE empty{0, 0};
            CK(uploads[i]->Map(0, &empty, &mapped));
            if (i == 3) std::memset(mapped, 0, static_cast<size_t>(bytes));
            else std::memcpy(mapped, data[i].data(), static_cast<size_t>(bytes));
            uploads[i]->Unmap(0, nullptr);
            list->CopyBufferRegion(buffers[i].Get(), 0, uploads[i].Get(), 0, bytes);
            transition(buffers[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                       i == 3 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (i == 3) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC u{}; u.Format = DXGI_FORMAT_R32_UINT;
                if (wmma) { u.Format = DXGI_FORMAT_R32_TYPELESS; u.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW; }
                u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; u.Buffer.NumElements = static_cast<UINT>(bytes / 4);
                device->CreateUnorderedAccessView(buffers[i].Get(), nullptr, &u, handle);
            } else {
                D3D12_SHADER_RESOURCE_VIEW_DESC s{}; s.Format = DXGI_FORMAT_R32_UINT;
                if (wmma) { s.Format = DXGI_FORMAT_R32_TYPELESS; s.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW; }
                s.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                s.Buffer.NumElements = static_cast<UINT>(bytes / 4);
                device->CreateShaderResourceView(buffers[i].Get(), &s, handle);
            }
            handle.ptr += increment;
        }
        submit();
        for (auto& upload : uploads) upload.Reset();
        D3D12_QUERY_HEAP_DESC qh{}; qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qh.Count = 2;
        CK(device->CreateQueryHeap(&qh, IID_PPV_ARGS(&queries)));
        timing = buffer(16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        output = buffer(outputBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    double run(UINT gx, UINT gy, UINT repeats, bool capture)
    {
        begin(); list->SetPipelineState(pipeline.Get()); list->SetComputeRootSignature(root.Get());
        ID3D12DescriptorHeap* heaps[] = {descriptors.Get()}; list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootDescriptorTable(0, descriptors->GetGPUDescriptorHandleForHeapStart());
        if (wmmaMode) list->SetComputeRootUnorderedAccessView(1, buffers[3]->GetGPUVirtualAddress());
        list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        for (UINT i = 0; i < repeats; ++i) {
            list->Dispatch(gx, gy, 1);
            D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = buffers[3].Get(); list->ResourceBarrier(1, &barrier);
        }
        list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        list->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, timing.Get(), 0);
        if (capture) {
            transition(buffers[3].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyBufferRegion(output.Get(), 0, buffers[3].Get(), 0, outputBytes);
            transition(buffers[3].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        submit();
        void* mapped = nullptr; const D3D12_RANGE readRange{0, 16}, empty{0, 0};
        CK(timing->Map(0, &readRange, &mapped));
        UINT64 ticks[2]; std::memcpy(ticks, mapped, sizeof(ticks)); timing->Unmap(0, &empty);
        if (ticks[1] <= ticks[0] || !frequency) throw std::runtime_error("invalid GPU timestamps");
        return double(ticks[1] - ticks[0]) * 1e6 / double(frequency) / repeats;
    }
    void save(const std::filesystem::path& path)
    {
        void* mapped = nullptr; const D3D12_RANGE range{0, static_cast<SIZE_T>(outputBytes)}, empty{0, 0};
        CK(output->Map(0, &range, &mapped));
        std::ofstream file(path, std::ios::binary);
        file.write(static_cast<const char*>(mapped), static_cast<std::streamsize>(outputBytes));
        output->Unmap(0, &empty);
        if (!file) throw std::runtime_error("cannot write " + path.string());
    }
};
int main(int argc, char** argv)
{
    try {
        if (argc < 9 || argc > 11) {
            std::fprintf(stderr, "usage: d4r-k-layer SHADER PARAMS WEIGHTS INPUT OUTPUT GX GY REPEATS [--debug] [--wmma]\n");
            return 2;
        }
        bool debug = false, wmma = false;
        for (int i = 9; i < argc; ++i) {
            const std::string option(argv[i]);
            if (option == "--debug" && !debug) debug = true;
            else if (option == "--wmma" && !wmma) wmma = true;
            else throw std::runtime_error("unknown or duplicate option: " + option);
        }
        const UINT gx = number(argv[6], 65535), gy = number(argv[7], 65535), repeats = number(argv[8], 10000);
        auto params = read(argv[2]);
        if (params.size() != 448) throw std::runtime_error("parameters must use the existing 448-byte K layer ABI");
        Runner runner;
        runner.create(read(argv[1]), {params, read(argv[3]), read(argv[4])}, debug, wmma);
        runner.run(gx, gy, 1, true); runner.save(argv[5]);
        std::vector<double> samples;
        for (int i = 0; i < 7; ++i) {
            const double us = runner.run(gx, gy, repeats, false);
            if (i >= 2) samples.push_back(us);
        }
        std::sort(samples.begin(), samples.end());
        std::printf("%.3f us per dispatch (GPU timestamps, median; min %.3f; %u repeats; includes UAV ordering)\n",
                    samples[samples.size() / 2], samples.front(), repeats);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "d4r-k-layer: %s\n", e.what()); return 1; }
}
