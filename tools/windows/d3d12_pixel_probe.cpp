// Independent reference: D3D12 typed SRV loads and typed UAV stores.
#include "pixel_conversion.h"
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <array>
#include <cmath>
using namespace d4r::diag;
using namespace d4r::win;
namespace {
constexpr unsigned width = 259, height = 17, count = width * height;
struct Case { DXGI_FORMAT resource, view; unsigned plane; const char* name; };
const Case cases[] = {
    {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 0, "rgba16f"},
    {DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, "rgba32f"},
    {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, 0, "rgba8"},
    {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, 0, "bgra8"},
    {DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R11G11B10_FLOAT, 0, "r11g11b10f"},
    {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM, 0, "rgb10a2"},
    {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, 2, "rg16f"},
    {DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, 2, "rg32f"},
    {DXGI_FORMAT_R16G16_SNORM, DXGI_FORMAT_R16G16_SNORM, 2, "rg16snorm"},
    {DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_R16G16_UNORM, 2, "rg16unorm"},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 2, "motion-rgba16f"},
    {DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT, 2, "motion-rgba32f"},
    {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_FLOAT, 1, "depth-r32f"},
    {DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM, 1, "depth-r16unorm"},
    {DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, 1, "depth24"},
    {DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_FLOAT, 4, "exposure-r16f"},
    {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_FLOAT, 4, "exposure-r32f"},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 4, "exposure-rgba16f"},
    {DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT, 4, "exposure-rgba32f"},
};
struct List {
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    explicit List(ID3D12Device* device) {
        dx(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(alloc.GetAddressOf())), "Pixel allocator");
        dx(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(list.GetAddressOf())), "Pixel command list");
    }
    void submit(ID3D12CommandQueue* queue, SharedTimeline& timeline) {
        dx(list->Close(), "Close pixel list"); ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists); timeline.drain(queue);
    }
};
ComPtr<ID3D12Resource> make_texture(ID3D12Device* device, DXGI_FORMAT fmt, bool uav = false) {
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height; desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Format = fmt; desc.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT; heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    ComPtr<ID3D12Resource> result;
    dx(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(result.GetAddressOf())), "Pixel texture"); return result;
}
ComPtr<ID3D12Resource> uav_buffer(ID3D12Device* device) {
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = count * 16;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT; heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    ComPtr<ID3D12Resource> result;
    dx(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr, IID_PPV_ARGS(result.GetAddressOf())), "Pixel reference buffer"); return result;
}
ResourceAccess shader_access() {
    ResourceAccess result; result.enhanced = true; result.layout = D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE;
    result.access = D3D12_BARRIER_ACCESS_SHADER_RESOURCE; return result;
}
void native_layout(ID3D12GraphicsCommandList* list, ID3D12Resource* texture, D3D12_BARRIER_LAYOUT before, D3D12_BARRIER_LAYOUT after,
    D3D12_BARRIER_ACCESS accessBefore, D3D12_BARRIER_ACCESS accessAfter) {
    ComPtr<ID3D12GraphicsCommandList7> newer; dx(list->QueryInterface(IID_PPV_ARGS(newer.GetAddressOf())), "Probe enhanced list");
    D3D12_TEXTURE_BARRIER barrier{}; barrier.pResource = texture; barrier.LayoutBefore = before; barrier.LayoutAfter = after;
    barrier.AccessBefore = accessBefore; barrier.AccessAfter = accessAfter; barrier.SyncBefore = barrier.SyncAfter = D3D12_BARRIER_SYNC_ALL;
    D3D12_BARRIER_GROUP group{}; group.Type = D3D12_BARRIER_TYPE_TEXTURE; group.NumBarriers = 1; group.pTextureBarriers = &barrier; newer->Barrier(1,&group);
}
void fixture(uint8_t* row, unsigned y, PixelSpec spec) {
    for (unsigned x = 0; x < width; ++x) {
        unsigned i = x + y * width; auto* p = row + x * spec.bytes;
        const uint16_t half[] = {0, 1, 0x3ff, 0x400, 0x1400, 0x3555, 0x3800, 0x3bff, 0x3c00, 0x3c01, 0x5c07, 0x7bff, 0x8001, 0xbc00};
        switch (spec.storage) {
            case d4r::pixel::rgba16f: case d4r::pixel::rg16f: case d4r::pixel::r16f:
                for (unsigned c = 0; c < spec.bytes / 2; ++c) reinterpret_cast<uint16_t*>(p)[c] = half[(i + c * 3) % std::size(half)]; break;
            case d4r::pixel::rgba32f: case d4r::pixel::rg32f: case d4r::pixel::r32f:
                for (unsigned c = 0; c < spec.bytes / 4; ++c) reinterpret_cast<float*>(p)[c] = float(int((i * 23 + c * 13) % 4097) - 2048) / 4096.f; break;
            case d4r::pixel::r11g11b10f:
                *reinterpret_cast<uint32_t*>(p) = (i % 1984) | (((i * 13) % 1984) << 11) | (((i * 7) % 992) << 22); break;
            case d4r::pixel::rgb10a2:
                *reinterpret_cast<uint32_t*>(p) = (i % 1024) | (((i * 13) % 1024) << 10) | (((i * 7) % 1024) << 20) | ((i % 4) << 30); break;
            case d4r::pixel::rgba8: case d4r::pixel::bgra8:
                for (unsigned c = 0; c < 4; ++c) p[c] = uint8_t(i * (2 * c + 1) + c * 43); break;
            case d4r::pixel::rg16snorm: case d4r::pixel::rg16unorm: case d4r::pixel::r16unorm:
                for (unsigned c = 0; c < spec.bytes / 2; ++c) reinterpret_cast<uint16_t*>(p)[c] = uint16_t(i < 4 ? i * 32767 : i * 193 + c * 37); break;
            case d4r::pixel::depth24: *reinterpret_cast<uint32_t*>(p) = (i * 3791) | ((i % 256) << 24); break;
            default: throw std::runtime_error("Missing diagnostic fixture");
        }
    }
}
uint16_t reference_half(uint32_t bits) {
    // Independent nearest-value search over the binary16 number line. HLSL
    // f32tof16 may truncate; it is not the CUDA RNE conversion being tested.
    static const std::vector<double> values = [] {
        std::vector<double> v(31744);
        for (unsigned h = 0; h < v.size(); ++h) v[h] = h < 1024 ? std::ldexp(double(h), -24) :
            std::ldexp(1. + double(h & 1023) / 1024., int(h >> 10) - 15);
        return v;
    }();
    float value = 0; std::memcpy(&value, &bits, 4);
    double magnitude = std::abs(double(value));
    if (!std::isfinite(magnitude) || magnitude > values.back()) throw std::runtime_error("Unexpected non-finite decode reference");
    auto upper = std::lower_bound(values.begin(), values.end(), magnitude);
    unsigned h = unsigned(upper - values.begin());
    if (h && (magnitude - values[h - 1] < values[h] - magnitude ||
        (magnitude - values[h - 1] == values[h] - magnitude && (h & 1)))) --h;
    return uint16_t(h | ((bits >> 16) & 0x8000));
}
struct Shaders {
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> decode, encode;
    explicit Shaders(ID3D12Device* device) {
        wchar_t system[32768]{}; if (!GetSystemDirectoryW(system, 32768)) throw std::runtime_error("System path");
        Library compiler(std::filesystem::path(system) / L"d3dcompiler_47.dll");
        auto compile = compiler.symbol<decltype(&D3DCompile)>("D3DCompile");
        const char* source = R"(
Texture2D<float4> Input : register(t0);
StructuredBuffer<uint4> Canonical : register(t1);
RWStructuredBuffer<uint4> Output : register(u0);
RWTexture2D<float4> Expected : register(u1);
cbuffer Constants : register(b0) { uint Width, Height, Plane; };
[numthreads(64,1,1)] void read(uint3 t:SV_DispatchThreadID) {
    if (t.x >= Width*Height) return;
    float4 v = Input.Load(int3(t.x % Width,t.x / Width,0));
    Output[t.x] = asuint(v);
}
[numthreads(64,1,1)] void write(uint3 t:SV_DispatchThreadID) {
    if (t.x < Width*Height) Expected[uint2(t.x % Width,t.x / Width)] = f16tof32(Canonical[t.x]);
})";
        D3D12_DESCRIPTOR_RANGE range[3]{};
        for (unsigned i = 0; i < 3; ++i) { range[i].RangeType = i == 2 ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range[i].NumDescriptors = 1; range[i].BaseShaderRegister = i == 2 ? 1 : i; }
        D3D12_ROOT_PARAMETER parameters[5]{};
        for (unsigned i = 0; i < 3; ++i) { parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[i].DescriptorTable.NumDescriptorRanges = 1; parameters[i].DescriptorTable.pDescriptorRanges = &range[i]; }
        parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[4].Constants.Num32BitValues = 3;
        D3D12_ROOT_SIGNATURE_DESC sd{}; sd.NumParameters = 5; sd.pParameters = parameters;
        ComPtr<ID3DBlob> blob, errors; dx(D3D12SerializeRootSignature(&sd, D3D_ROOT_SIGNATURE_VERSION_1, blob.GetAddressOf(), errors.GetAddressOf()), "Pixel root serialize");
        dx(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(root.GetAddressOf())), "Pixel root");
        for (unsigned i = 0; i < 2; ++i) {
            if (FAILED(compile(source, std::strlen(source), nullptr, nullptr, nullptr, i ? "write" : "read", "cs_5_1", D3DCOMPILE_OPTIMIZATION_LEVEL3,
                0, blob.ReleaseAndGetAddressOf(), errors.ReleaseAndGetAddressOf()))) throw std::runtime_error(errors ? std::string(static_cast<char*>(errors->GetBufferPointer())) : "Pixel HLSL compilation");
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = root.Get(); pd.CS = {blob->GetBufferPointer(), blob->GetBufferSize()};
            dx(device->CreateComputePipelineState(&pd, IID_PPV_ARGS((i ? encode : decode).GetAddressOf())), "Pixel PSO");
        }
    }
};
std::vector<uint32_t> read_reference(ID3D12Device* device, ID3D12CommandQueue* queue, SharedTimeline& timeline,
    Shaders& shader, ID3D12DescriptorHeap* heap, D3D12_GPU_DESCRIPTOR_HANDLE srv, ID3D12Resource* reference, unsigned plane) {
    auto readback = make_buffer(device, count * 16, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    List list(device); list.list->SetPipelineState(shader.decode.Get()); list.list->SetComputeRootSignature(shader.root.Get());
    list.list->SetDescriptorHeaps(1, &heap); list.list->SetComputeRootDescriptorTable(0, srv);
    list.list->SetComputeRootUnorderedAccessView(3, reference->GetGPUVirtualAddress());
    const unsigned constants[] = {width, height, plane}; list.list->SetComputeRoot32BitConstants(4, 3, constants, 0);
    list.list->Dispatch((count + 63) / 64, 1, 1);
    transition(list.list.Get(), reference, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list.list->CopyResource(readback.Get(), reference);
    transition(list.list.Get(), reference, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list.submit(queue, timeline);
    void* data = nullptr; D3D12_RANGE range{0, count * 16}; dx(readback->Map(0, &range, &data), "Pixel readback");
    std::vector<uint32_t> result(count * 4); std::memcpy(result.data(), data, count * 16); D3D12_RANGE noWrite{}; readback->Unmap(0, &noWrite); return result;
}
}
int main(int argc, char** argv) {
    start();
    try {
        Args args(argc, argv); HipApi hip(args.hip_root); hipDeviceProp_t props{}; hip.select_gfx1201(args.device, props);
        ExternalApi external{hip}; PixelProgram program(hip, wide(args.module));
        ComPtr<IDXGIFactory4> factory; dx(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())), "Pixel DXGI");
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; ; ++i) { dx(factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()), "Pixel adapter"); DXGI_ADAPTER_DESC1 d{}; dx(adapter->GetDesc1(&d), "Pixel adapter desc"); if (!std::memcmp(&d.AdapterLuid, props.luid, sizeof(LUID))) break; }
        ComPtr<ID3D12Device> device; dx(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.GetAddressOf())), "Pixel D3D12");
        const bool enhanced = args.interop_mode == "enhanced";
        if (enhanced) { D3D12_FEATURE_DATA_D3D12_OPTIONS12 support{}; dx(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &support, sizeof(support)), "Enhanced barrier support"); if (!support.EnhancedBarriersSupported) throw std::runtime_error("Enhanced barriers unsupported"); }
        ComPtr<ID3D12CommandQueue> queue; D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT; dx(device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.GetAddressOf())), "Pixel queue");
        SharedTimeline timeline(external, device.Get()); Shaders shader(device.Get()); auto reference = uav_buffer(device.Get());
        unsigned verified = 0;
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration) for (const auto& item : cases) {
            PixelSpec spec = pixel_spec(item.plane, item.resource); if (!pixel_supported(item.plane, spec.storage)) throw std::runtime_error("Unsupported fixture plane");
            auto input = make_texture(device.Get(), item.resource); SharedPlane shared(external, device.Get(), resource_desc(input.Get()));
            auto upload = make_buffer(device.Get(), shared.bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            void* data = nullptr; D3D12_RANGE noRead{}; dx(upload->Map(0, &noRead, &data), "Pixel fixture upload");
            std::memset(data, 0x5a, size_t(shared.bytes)); for (unsigned y = 0; y < height; ++y) fixture(static_cast<uint8_t*>(data) + y * shared.footprint.Footprint.RowPitch, y, spec); upload->Unmap(0, nullptr);
            { List list(device.Get()); D3D12_TEXTURE_COPY_LOCATION src{}, dst{}; src.pResource = upload.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = shared.footprint; dst.pResource = input.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                transition(list.list.Get(), input.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST); list.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                transition(list.list.Get(), input.Get(), D3D12_RESOURCE_STATE_COPY_DEST, enhanced ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if (enhanced) native_layout(list.list.Get(),input.Get(),D3D12_BARRIER_LAYOUT_COMMON,D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE,D3D12_BARRIER_ACCESS_COMMON,D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
                shared.copy_input(list.list.Get(), input.Get(), enhanced ? shader_access() : ResourceAccess(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)); list.submit(queue.Get(), timeline); }
            D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 4; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap; dx(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(heap.GetAddressOf())), "Pixel descriptors");
            D3D12_CPU_DESCRIPTOR_HANDLE cpu{}, initial{}; D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
#if defined(__MINGW32__)
            heap->GetCPUDescriptorHandleForHeapStart(&cpu); heap->GetGPUDescriptorHandleForHeapStart(&gpu);
#else
            cpu = heap->GetCPUDescriptorHandleForHeapStart(); gpu = heap->GetGPUDescriptorHandleForHeapStart();
#endif
            initial = cpu; unsigned stride = device->GetDescriptorHandleIncrementSize(hd.Type);
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format = item.view; srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; srv.Texture2D.MipLevels = 1; device->CreateShaderResourceView(input.Get(), &srv, cpu);
            auto expected = read_reference(device.Get(), queue.Get(), timeline, shader, heap.Get(), gpu, reference.Get(), item.plane);
            unsigned channels = item.plane == 2 ? 2 : item.plane == 0 ? 4 : 1; unsigned element = item.plane == 0 || item.plane == 2 ? 2 : 4;
            PixelAllocation canonical(hip, width, height, channels * element);
            std::vector<uint8_t> guard(size_t(canonical.pitch * height), 0xcd);
            hip.check(hip.hipMemcpy(canonical.data, guard.data(), guard.size(), hipMemcpyHostToDevice), "Diagnostic pitch guard initialization");
            program.convert(false, shared.mapped, shared.footprint.Footprint.RowPitch, canonical.data, canonical.pitch, width, height, spec.storage, item.plane);
            std::vector<uint8_t> actual(size_t(canonical.pitch * height)); hip.check(hip.hipMemcpy(actual.data(), canonical.data, actual.size(), hipMemcpyDeviceToHost), "Diagnostic canonical readback");
            unsigned differs = 0; double maxAbs = 0;
            for (unsigned y = 0; y < height; ++y) for (uint64_t offset = width * channels * element; offset < canonical.pitch; ++offset)
                if (actual[size_t(y * canonical.pitch + offset)] != 0xcd) throw std::runtime_error("GPU decode wrote past the row/pixel range");
            for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) for (unsigned c = 0; c < channels; ++c) {
                unsigned i = (y * width + x) * 4 + c; const auto* p = actual.data() + y * canonical.pitch + (x * channels + c) * element; uint32_t got = 0; std::memcpy(&got, p, element);
                if (element == 2) expected[i] = reference_half(expected[i]);
                bool mismatch = got != expected[i];
                if (element == 4) { float a = 0, b = 0; std::memcpy(&a, &got, 4); std::memcpy(&b, &expected[i], 4); double error = std::abs(double(a) - b); maxAbs = std::max(maxAbs, error); mismatch = !std::isfinite(a) || !std::isfinite(b) || error > 1.2e-7 * std::max(1., std::abs(double(b))); }
                if (mismatch && differs++ < 5) std::fprintf(stderr, "PIXEL_DIFFERENT format=%s pixel=%u channel=%u expected=%08x actual=%08x\n", item.name, y * width + x, c, expected[i], got);
            }
            if (differs) throw std::runtime_error(std::string("D3D12 typed decode mismatch: ") + item.name + " count=" + std::to_string(differs));
            if (item.plane == 0) {
                // BGRA uses the identical UNORM converter through an RGBA UAV;
                // the consumer SRV independently tests the channel permutation.
                DXGI_FORMAT encodedFormat = spec.storage == d4r::pixel::bgra8 ? DXGI_FORMAT_R8G8B8A8_UNORM : item.view;
                auto encoded = make_texture(device.Get(), encodedFormat, true); auto actualTexture = make_texture(device.Get(), item.resource);
                auto canonicalUpload = make_buffer(device.Get(), count * 16, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
                dx(canonicalUpload->Map(0, &noRead, &data), "Encode fixture upload");
                auto* values = static_cast<uint32_t*>(data); std::vector<uint16_t> halves(count * 4);
                // Cover finite half values, rounding ties, negatives and overflow.
                for (unsigned i = 0; i < count; ++i) for (unsigned c = 0; c < 4; ++c) { uint16_t v = uint16_t(((i * 37 + c * 7919) % 31744) | (i % 11 == 0 ? 0x8000 : 0)); values[i * 4 + c] = v; halves[i * 4 + c] = v; }
                canonicalUpload->Unmap(0, nullptr);
                std::vector<uint8_t> padded(size_t(canonical.pitch * height), 0x5a); for (unsigned y = 0; y < height; ++y) std::memcpy(padded.data() + y * canonical.pitch, halves.data() + y * width * 4, width * 8);
                hip.check(hip.hipMemcpy(canonical.data, padded.data(), padded.size(), hipMemcpyHostToDevice), "Diagnostic encode input");
                program.convert(true, canonical.data, canonical.pitch, shared.mapped, shared.footprint.Footprint.RowPitch, width, height, spec.storage, 3);
                cpu.ptr = initial.ptr + stride; srv = {}; srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; srv.Buffer.NumElements = count; srv.Buffer.StructureByteStride = 16; device->CreateShaderResourceView(canonicalUpload.Get(), &srv, cpu);
                cpu.ptr += stride; D3D12_UNORDERED_ACCESS_VIEW_DESC uv{}; uv.Format = encodedFormat; uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D; device->CreateUnorderedAccessView(encoded.Get(), nullptr, &uv, cpu);
                { List list(device.Get()); transition(list.list.Get(), encoded.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); list.list->SetPipelineState(shader.encode.Get()); list.list->SetComputeRootSignature(shader.root.Get()); ID3D12DescriptorHeap* heaps[] = {heap.Get()}; list.list->SetDescriptorHeaps(1, heaps); auto handle = gpu; handle.ptr += stride; list.list->SetComputeRootDescriptorTable(1, handle); handle.ptr += stride; list.list->SetComputeRootDescriptorTable(2, handle); unsigned constants[] = {width,height,0}; list.list->SetComputeRoot32BitConstants(4,3,constants,0); list.list->Dispatch((count+63)/64,1,1); transition(list.list.Get(), encoded.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    ResourceAccess state; state.enhanced = enhanced; shared.copy_output(list.list.Get(), actualTexture.Get(), state);
                    if (enhanced) native_layout(list.list.Get(),actualTexture.Get(),D3D12_BARRIER_LAYOUT_COMMON,D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE,D3D12_BARRIER_ACCESS_COMMON,D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
                    else transition(list.list.Get(), actualTexture.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    list.submit(queue.Get(),timeline); }
                srv = {}; srv.Format = item.view; srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; srv.Texture2D.MipLevels = 1; device->CreateShaderResourceView(actualTexture.Get(), &srv, initial);
                auto encodedActual = read_reference(device.Get(), queue.Get(), timeline, shader, heap.Get(), gpu, reference.Get(), 4);
                srv.Format = encodedFormat; device->CreateShaderResourceView(encoded.Get(), &srv, initial);
                auto encodedExpected = read_reference(device.Get(), queue.Get(), timeline, shader, heap.Get(), gpu, reference.Get(), 4);
                for (unsigned i = 0; i < count * 4; ++i) if (encodedActual[i] != encodedExpected[i]) { if (differs++ < 5) std::fprintf(stderr, "ENCODE_DIFFERENT format=%s pixel=%u channel=%u expected=%08x actual=%08x\n", item.name, i/4,i%4,encodedExpected[i],encodedActual[i]); }
                if (differs) throw std::runtime_error(std::string("D3D12 typed encode mismatch: ") + item.name + " count=" + std::to_string(differs));
            }
            ++verified; std::printf("PASS PIXEL_FORMAT name=%s plane=%u pixels=%u decode_max_abs=%.9g encode=%u\n", item.name,item.plane,count,maxAbs,item.plane==0);
        }
        std::printf("PASS D3D12_GPU_FORMATS architecture=gfx1201 cases=%u iterations=%u width=%u height=%u enhanced=%u typed_srv_reference=1 typed_uav_reference=1\n", verified,args.iterations,width,height,enhanced); return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"FAIL D3D12_GPU_FORMATS %s\n",error.what()); loaded_modules(); return 1; }
}
