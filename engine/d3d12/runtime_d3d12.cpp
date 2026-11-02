#include "runtime_d3d12.h"
#include "runtime_m_rdna3.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace d4r::dx {
namespace {
void check(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        char text[32]; std::snprintf(text, sizeof(text), "0x%08lx", static_cast<unsigned long>(result));
        throw std::runtime_error(std::string(operation) + ": HRESULT " + text);
    }
}
#define DXCHECK(call) check((call), #call)

// The reserved register space AMD's driver reads the WMMA intrinsics from (ags_wmma.hlsli).
constexpr UINT AgsSpace = 0x7fff0ade;
// Engine-owned per-frame parameter blocks, one buffer: written by WriteBufferImmediate at the start of every
// recording, read by the stages as structured views (and the exposure pass writes its fields into the first two).
// Structured views use multiples of their block size. CBV layers instead use
// 256-byte-aligned addresses and a 512-byte stride for their 448-byte blocks.
constexpr UINT64 InputOffset = 0, OutputOffset = 188, ExposureOffset = 392, LayerOffset = 448;
constexpr UINT64 CbvLayerOffset = 512, CbvLayerStride = 512;
constexpr UINT64 ParamsBytes = CbvLayerOffset + uint64_t(KLayers) * CbvLayerStride;
static_assert(CbvLayerOffset % D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT == 0 &&
              CbvLayerStride % D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT == 0 && CbvLayerStride >= sizeof(KLayerParams));
static_assert(OutputOffset % sizeof(OutputParams) == 0 && OutputOffset >= sizeof(InputParams));
static_assert(ExposureOffset % sizeof(ExposureParams) == 0 && ExposureOffset >= OutputOffset + sizeof(OutputParams));
static_assert(LayerOffset % sizeof(KLayerParams) == 0 && LayerOffset >= ExposureOffset + sizeof(ExposureParams));

ComPtr<ID3D12Resource> buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                              D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_HEAP_PROPERTIES properties{heap};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = std::max<UINT64>(bytes, 16);
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    DXCHECK(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource)));
    return resource;
}
ComPtr<ID3D12Resource> texture(ID3D12Device* device, UINT width, UINT height, DXGI_FORMAT format) {
    D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = width; desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1; desc.Format = format; desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> resource;
    DXCHECK(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                            nullptr, IID_PPV_ARGS(&resource)));
    return resource;
}
D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    return barrier;
}
void uavBarrier(ID3D12GraphicsCommandList* list) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &barrier);
}
// The shader-readable format of a game texture
DXGI_FORMAT readFormat(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    default: return format;
    }
}
DXGI_FORMAT textureFormat(const FrameTexture& t) {
    return t.format != DXGI_FORMAT_UNKNOWN ? t.format : readFormat(t.resource->GetDesc().Format);
}

#include "capture_d3d12.h"

// ---------------------------------------------------------------- pipelines
// A stage's bindings, in the shaders' numbering: binding N is register tN or uN (one descriptor table, slot N).
enum class Bind { None, Texture, Raw, Structured, RwStructured, RwRaw, RwTexture, Typed, RwTyped, Typed4, RwTyped4 };
struct Binding { Bind kind = Bind::None; UINT stride = 0; bool linear = false; DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN; };
struct Pipeline {
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> state;
    std::vector<Binding> bindings;
    bool ags = false;     // root parameter 1: the AGS intrinsic UAV
    bool layerCbv = false;
    void create(ID3D12Device* device, const std::vector<char>& dxil, std::vector<Binding> table, bool wmma, bool cbv = false, const char* name = nullptr) {
        bindings = std::move(table); ags = wmma; layerCbv = cbv;
        std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
        std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
        for (UINT i = 0; i < bindings.size(); ++i) {
            const auto kind = bindings[i].kind;
            if (kind == Bind::None) continue;
            const bool srv = kind == Bind::Texture || kind == Bind::Raw || kind == Bind::Structured || kind == Bind::Typed || kind == Bind::Typed4;
            ranges.push_back({srv ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, i, 0, i});
            if (kind == Bind::Texture) {
                D3D12_STATIC_SAMPLER_DESC s{};
                s.Filter = bindings[i].linear ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
                s.AddressU = s.AddressV = s.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
                s.MaxLOD = D3D12_FLOAT32_MAX; s.ShaderRegister = i; s.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                s.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
                samplers.push_back(s);
            }
        }
        D3D12_ROOT_PARAMETER parameters[3]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[0].DescriptorTable = {UINT(ranges.size()), ranges.data()};
        parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[1].Descriptor = {0, AgsSpace};
        parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        if (layerCbv) {
            auto& cb = parameters[ags ? 2 : 1];
            cb.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            cb.Descriptor = {0, 0}; cb.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        D3D12_ROOT_SIGNATURE_DESC desc{(ags ? 2u : 1u) + (layerCbv ? 1u : 0u), parameters, UINT(samplers.size()), samplers.data(), D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob, error;
        if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error)))
            throw std::runtime_error(std::string("root signature: ") + (error ? static_cast<const char*>(error->GetBufferPointer()) : "?"));
        DXCHECK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get(); pso.CS = {dxil.data(), dxil.size()};
        DXCHECK(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&state)));
        if (const char* directory = std::getenv("D4R_D3D12_DUMP_DIR"); directory && *directory && name) {
            std::filesystem::create_directories(directory);
            auto save = [&](const char* suffix, const void* data, size_t bytes) {
                const auto path = std::filesystem::path(directory) / (std::string(name) + suffix);
                std::ofstream file(path, std::ios::binary);
                file.write(static_cast<const char*>(data), std::streamsize(bytes));
                if (!file) throw std::runtime_error("cannot write pipeline diagnostic: " + path.string());
            };
            save(".dxil", dxil.data(), dxil.size());
            save(".rootsig", blob->GetBufferPointer(), blob->GetBufferSize());
            ComPtr<ID3DBlob> cached;
            if (SUCCEEDED(state->GetCachedBlob(&cached)))
                save(".cached-pso", cached->GetBufferPointer(), cached->GetBufferSize());
        }
    }
    explicit operator bool() const { return state != nullptr; }
};
const Binding T{Bind::Texture}, TL{Bind::Texture, 0, true}, N{};
Binding rw(UINT stride) { return {Bind::RwStructured, stride}; }
std::vector<Binding> layerBindings(bool cbv = false) { return {cbv ? N : rw(sizeof(KLayerParams)), {Bind::Raw}, rw(8), rw(8), rw(8), rw(8)}; }
std::vector<Binding> inputBindings() { return {rw(sizeof(InputParams)), T, TL, T, T, TL, rw(8)}; }
std::vector<Binding> outputBindings() {
    const Binding image{Bind::RwTexture};
    return {rw(sizeof(OutputParams)), T, TL, TL, T, T, image, image, image, image, {Bind::RwRaw}, {Bind::Structured, 16}};
}
std::vector<Binding> exposure0Bindings() { return {rw(sizeof(ExposureParams)), T, rw(4)}; }
std::vector<Binding> exposure1Bindings() { return {rw(sizeof(ExposureParams)), rw(4), rw(8), rw(4), rw(4), T}; }

const char* const Stages[] = {"input_k", "output_k", "exposure_k0", "exposure_k1"};
} // namespace

// ---------------------------------------------------------------- descriptors
struct Heap {
    ComPtr<ID3D12DescriptorHeap> gpu, cpu;     // shader-visible, and its CPU-only twin for UAV clears
    UINT increment = 0, size = 0;
    void create(ID3D12Device* device, UINT count) {
        size = count;
        D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        DXCHECK(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&gpu)));
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        DXCHECK(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&cpu)));
        increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle(UINT i) const { auto h = gpu->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * increment; return h; }
    D3D12_CPU_DESCRIPTOR_HANDLE cpuOnly(UINT i) const { auto h = cpu->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * increment; return h; }
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle(UINT i) const { auto h = gpu->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(i) * increment; return h; }
};

struct Model::Impl {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12Resource> weights, lut, upload;
    KModelData data;
    std::array<Pipeline, KLayers> network;
    Pipeline input, output, outputRgb10;
    std::array<Pipeline, 2> exposure;
    bool uploaded = false, hasExposure = false, layerCbv = false;
    UINT64 layerParamOffset(UINT i) const { return layerCbv ? CbvLayerOffset + i * CbvLayerStride : LayerOffset + i * sizeof(KLayerParams); }
    std::mutex setup; // execute: one setup list at a time
};

std::string Model::missingShaders(const std::string& directory) {
    std::string missing;
    auto need = [&](const std::string& name) { if (!fileExists(directory + "/" + name + ".dxil")) missing += (missing.empty() ? "" : ", ") + name + ".dxil"; };
    for (auto name : KLayerNames) need(name);
    need("input_k"); need("output_k");
    return missing;
}

Model::Model(ID3D12Device* device, const std::string& directory) : impl(std::make_unique<Impl>()) {
    auto& m = *impl;
    m.device = device;
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) || model.HighestShaderModel < D3D_SHADER_MODEL_6_6)
        throw std::runtime_error("the device does not support Shader Model 6.6");
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1))) ||
        !options1.WaveOps || options1.WaveLaneCountMin > 32 || options1.WaveLaneCountMax < 64)
        throw std::runtime_error("the device does not run both wave32 and wave64 compute shaders");
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4))) || !options4.Native16BitShaderOpsSupported)
        throw std::runtime_error("the device has no native 16-bit shader operations");
    if (auto missing = missingShaders(directory); !missing.empty())
        throw std::runtime_error("the model has no D3D12 shaders (" + missing + "): rebuild it with compile_k.py on Windows");
    m.data = loadKModel(directory, false);
    m.layerCbv = fileExists(directory + "/d3d12_layer_cbv.bin");
    bool tiledWeights = false;
    if (m.layerCbv) {
        const auto marker = readFile(directory + "/d3d12_layer_cbv.bin");
        if (marker.size() != 8 || (std::memcmp(marker.data(), "D4RCBV01", 8) && std::memcmp(marker.data(), "D4RCBV02", 8)))
            throw std::runtime_error("incompatible D3D12 layer constant-buffer marker");
        tiledWeights = !std::memcmp(marker.data(), "D4RCBV02", 8);
    }
    const std::string packedPath = directory + "/weights_d3d12.bin";
    if (tiledWeights) {
        const auto packed = readFile(packedPath);
        if (packed.size() != m.data.weights.size() + 8 || std::memcmp(packed.data(), "D4RWTB01", 8))
            throw std::runtime_error("incompatible D3D12 tiled weights");
        m.data.weights.assign(packed.begin() + 8, packed.end());
    } else if (fileExists(packedPath)) {
        throw std::runtime_error("D3D12 tiled weights require the D4RCBV02 package marker");
    }
    m.weights = buffer(device, m.data.weights.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    m.lut = buffer(device, m.data.lut.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    auto shader = [&](const std::string& name) { return readFile(directory + "/" + name + ".dxil"); };
    // Pipeline creation fails on a driver without AGS's WMMA: the network then cannot run on this device.
    for (unsigned i = 0; i < KLayers; ++i) {
        try { m.network[i].create(device, shader(KLayerNames[i]), layerBindings(m.layerCbv), true, m.layerCbv, KLayerNames[i]); }
        catch (const std::exception& e) { throw std::runtime_error(std::string(KLayerNames[i]) + " (AMD WMMA): " + e.what()); }
    }
    m.input.create(device, shader("input_k"), inputBindings(), false, false, "input_k");
    m.output.create(device, shader("output_k"), outputBindings(), false, false, "output_k");
    if (fileExists(directory + "/output_k_rgb10.dxil")) m.outputRgb10.create(device, shader("output_k_rgb10"), outputBindings(), false, false, "output_k_rgb10");
    if (fileExists(directory + "/exposure_k0.dxil") && fileExists(directory + "/exposure_k1.dxil")) {
        m.exposure[0].create(device, shader("exposure_k0"), exposure0Bindings(), false, false, "exposure_k0");
        m.exposure[1].create(device, shader("exposure_k1"), exposure1Bindings(), false, false, "exposure_k1");
        m.hasExposure = true;
    }
    (void)Stages;
}
Model::~Model() = default;
bool Model::uploadRecorded() const { return impl->uploaded; }
void Model::execute(const std::function<void(ID3D12GraphicsCommandList*)>& commands) {
    auto& m = *impl;
    std::lock_guard<std::mutex> lock(m.setup);
    D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    DXCHECK(m.device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)));
    DXCHECK(m.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
    DXCHECK(m.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
    DXCHECK(m.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    commands(list.Get());
    DXCHECK(list->Close());
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    DXCHECK(queue->Signal(fence.Get(), 1));
    if (fence->GetCompletedValue() < 1) {
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("CreateEvent failed");
        const HRESULT hr = fence->SetEventOnCompletion(1, event);
        if (SUCCEEDED(hr)) WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
        DXCHECK(hr);
    }
    if (FAILED(m.device->GetDeviceRemovedReason())) throw std::runtime_error("the device was removed during setup");
}
bool Model::hasAutoExposure() const { return impl->hasExposure; }
bool Model::hasRegularDepth() const { return impl->data.hasRegularDepth; }
bool Model::hasDisplayMotion() const { return impl->data.hasDisplayMotion; }
bool Model::hasGameExposure() const { return impl->data.hasGameExposure; }
bool Model::hasRgb10Output() const { return bool(impl->outputRgb10); }
bool Model::hasDirectOrigins() const { return impl->data.hasDirectOrigins; }
void Model::recordUpload(ID3D12GraphicsCommandList* list) {
    auto& m = *impl;
    if (m.uploaded) throw std::runtime_error("model upload already recorded");
    const UINT64 lutAt = (m.data.weights.size() + 255) / 256 * 256;
    m.upload = buffer(m.device.Get(), lutAt + m.data.lut.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    D3D12_RANGE none{0, 0};
    DXCHECK(m.upload->Map(0, &none, &mapped));
    std::memcpy(mapped, m.data.weights.data(), m.data.weights.size());
    std::memcpy(static_cast<char*>(mapped) + lutAt, m.data.lut.data(), m.data.lut.size());
    m.upload->Unmap(0, nullptr);
    list->CopyBufferRegion(m.weights.Get(), 0, m.upload.Get(), 0, m.data.weights.size());
    list->CopyBufferRegion(m.lut.Get(), 0, m.upload.Get(), lutAt, m.data.lut.size());
    const D3D12_RESOURCE_BARRIER barriers[] = {
        transition(m.weights.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transition(m.lut.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
    list->ResourceBarrier(2, barriers);
    m.uploaded = true;
    std::vector<char>().swap(m.data.weights); std::vector<char>().swap(m.data.lut);   // only the sizes are kept below
}

// ---------------------------------------------------------------- engine
struct Engine::Impl {
    Model::Impl& model;
    Shape shape;
    ID3D12Device* device;
    ComPtr<ID3D12Resource> params, dummy, exposureSums, exposureState;
    std::array<ComPtr<ID3D12Resource>, 2> main;
    std::array<ComPtr<ID3D12Resource>, 5> skip;
    struct History { ComPtr<ID3D12Resource> color, luma, feature; };
    std::array<History, 2> history;
    ComPtr<ID3D12Resource> final;           // the engine's own output, created on first use
    Heap heap;
    // Static descriptors: the layers' tables, then the six history UAVs and the own output's UAV (for clears).
    static constexpr UINT LayerSlots = 6, ClearBase = KLayers * LayerSlots, ClearCount = 7, BindingBase = ClearBase + ClearCount;
    // One block per distinct set of borrowed resources (the Vulkan runtime's cached descriptor sets): exposure
    // passes 0 and 1, then input and output for each history parity.
    static constexpr UINT BlockSlots = 64, ExposureSlot0 = 0, ExposureSlot1 = 8, InputSlot = 16, OutputSlot = 40, Blocks = 32;
    struct Bindings {
        FrameTextures textures{};
        FrameOutput output{};
        ComPtr<ID3D12Resource> retained[5];
        ComPtr<ID3D12Fence> fence;          // null: retired by frame count (submitted without a fence)
        uint64_t value = 0, lastUse = 0;
        bool pending = false, inFlight = false;
        UINT block = 0;
    };
    std::vector<Bindings> bindings;
    size_t current = 0;                     // the frame's bindings (setFrame)
    size_t recorded = SIZE_MAX;             // the bindings of the recording awaiting submitted/discarded
    uint64_t uses = 0;
    std::string captureDirectory;
    std::vector<std::unique_ptr<FrameCapture>> captures;
    FrameCapture* recordingCapture = nullptr;
    InputParams input{}; OutputParams output{};
    ExposureParams exposure{};
    bool exposureOn = false, initialized = false, frameSet = false, reset = false, rgb10 = false, borrowedOutput = false;
    bool trim = true;
    bool external = false, frontDone = false;
    ID3D12Resource* externalTokens = nullptr;   // the caller's (an external network's), in COMMON, never transitioned
    ID3D12Resource* externalHead = nullptr;
    unsigned previous = 0;
    Impl(Model::Impl& m, const Shape& s) : model(m), shape(s), device(m.device.Get()) {
        if (const char* dir = std::getenv("D4R_D3D12_CAPTURE_DIR")) captureDirectory = dir;
    }

    void bufferView(UINT slot, ID3D12Resource* resource, const Binding& b, UINT64 offset = 0, UINT64 bytes = 0) {
        const auto handle = heap.cpuHandle(slot);
        if (!bytes && resource) bytes = resource->GetDesc().Width - offset;
        switch (b.kind) {
        case Bind::Raw: case Bind::Structured: {
            D3D12_SHADER_RESOURCE_VIEW_DESC v{};
            v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if (b.kind == Bind::Raw) { v.Format = DXGI_FORMAT_R32_TYPELESS; v.Buffer = {offset / 4, UINT(bytes / 4), 0, D3D12_BUFFER_SRV_FLAG_RAW}; }
            else v.Buffer = {offset / b.stride, UINT(bytes / b.stride), b.stride, D3D12_BUFFER_SRV_FLAG_NONE};
            device->CreateShaderResourceView(resource, &v, handle);
            break;
        }
        case Bind::RwRaw: case Bind::RwStructured: {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
            v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            if (b.kind == Bind::RwRaw) { v.Format = DXGI_FORMAT_R32_TYPELESS; v.Buffer = {offset / 4, UINT(bytes / 4), 0, 0, D3D12_BUFFER_UAV_FLAG_RAW}; }
            else v.Buffer = {offset / b.stride, UINT(bytes / b.stride), b.stride, 0, D3D12_BUFFER_UAV_FLAG_NONE};
            device->CreateUnorderedAccessView(resource, nullptr, &v, handle);
            break;
        }
        default: throw std::logic_error("buffer view of a texture binding");
        }
    }
    void textureView(UINT slot, ID3D12Resource* resource, DXGI_FORMAT format) {
        D3D12_SHADER_RESOURCE_VIEW_DESC v{};
        v.Format = format; v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; v.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(resource, &v, heap.cpuHandle(slot));
    }
    void imageView(D3D12_CPU_DESCRIPTOR_HANDLE handle, ID3D12Resource* resource, DXGI_FORMAT format) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
        v.Format = format; v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(resource, nullptr, &v, handle);
    }
    void writeBindings(Bindings& b);
    void dispatch(ID3D12GraphicsCommandList* list, const Pipeline& p, UINT table, UINT x, UINT y) {
        list->SetComputeRootSignature(p.root.Get());
        list->SetPipelineState(p.state.Get());
        list->SetComputeRootDescriptorTable(0, heap.gpuHandle(table));
        if (p.ags) list->SetComputeRootUnorderedAccessView(1, dummy->GetGPUVirtualAddress());
        if (p.layerCbv) list->SetComputeRootConstantBufferView(2, params->GetGPUVirtualAddress() + model.layerParamOffset(table / LayerSlots));
        list->Dispatch(x, y, 1);
    }
    // A host that cannot name a fence (submitted(nullptr)) gets its frames retired this many frames later.
    static constexpr uint64_t UnfencedFrames = 16;
    void retire() {
        for (auto it = captures.begin(); it != captures.end();) {
            auto& c = **it;
            if (!c.pending && (c.fence ? c.fence->GetCompletedValue() >= c.value : uses >= c.value + UnfencedFrames)) {
                for (auto& data : c.data) data.save();
                if (!c.data.empty()) {
                    std::ofstream complete(std::filesystem::path(c.data[0].path).parent_path() / "complete");
                    complete << "GPU submission retired\n";
                }
                it = captures.erase(it);
            } else ++it;
        }
        for (auto& b : bindings)
            if (b.inFlight && (b.fence ? b.fence->GetCompletedValue() >= b.value : uses >= b.value + UnfencedFrames)) {
                b.inFlight = false; b.fence.Reset();
            }
    }
};

void Engine::Impl::writeBindings(Bindings& b) {
    const UINT base = BindingBase + b.block * BlockSlots;
    const auto& t = b.textures;
    const DXGI_FORMAT color = textureFormat(t.color), motion = textureFormat(t.motion), depth = textureFormat(t.depth);
    auto out = b.output.resource ? b.output.resource : final.Get();
    const DXGI_FORMAT outFormat = b.output.resource ? (b.output.format != DXGI_FORMAT_UNKNOWN ? b.output.format : readFormat(b.output.resource->GetDesc().Format))
                                                    : DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (model.hasExposure) {
        const auto e0 = exposure0Bindings(), e1 = exposure1Bindings();
        bufferView(base + ExposureSlot0 + 0, params.Get(), e0[0], ExposureOffset, sizeof(ExposureParams));
        textureView(base + ExposureSlot0 + 1, t.color.resource, color);
        bufferView(base + ExposureSlot0 + 2, exposureSums.Get(), e0[2]);
        bufferView(base + ExposureSlot1 + 0, params.Get(), e1[0], ExposureOffset, sizeof(ExposureParams));
        bufferView(base + ExposureSlot1 + 1, exposureSums.Get(), e1[1]);
        bufferView(base + ExposureSlot1 + 2, exposureState.Get(), e1[2]);
        bufferView(base + ExposureSlot1 + 3, params.Get(), e1[3], InputOffset, sizeof(InputParams));
        bufferView(base + ExposureSlot1 + 4, params.Get(), e1[4], OutputOffset, sizeof(OutputParams));
        // the exposure texture is read only in game mode; the colour stands in otherwise
        if (t.exposure.resource) textureView(base + ExposureSlot1 + 5, t.exposure.resource, textureFormat(t.exposure));
        else textureView(base + ExposureSlot1 + 5, t.color.resource, color);
    }
    const auto in = inputBindings(), ou = outputBindings();
    for (unsigned i = 0; i < 2; ++i) {
        const UINT si = base + InputSlot + i * 12, so = base + OutputSlot + i * 12;
        auto& h = history[i]; auto& dest = history[1 - i];
        bufferView(si + 0, params.Get(), in[0], InputOffset, sizeof(InputParams));
        textureView(si + 1, t.color.resource, color);
        textureView(si + 2, h.color.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
        textureView(si + 3, t.motion.resource, motion);
        textureView(si + 4, t.depth.resource, depth);
        textureView(si + 5, h.feature.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
        bufferView(si + 6, external ? externalTokens : main[0].Get(), in[6]);
        bufferView(so + 0, params.Get(), ou[0], OutputOffset, sizeof(OutputParams));
        textureView(so + 1, t.color.resource, color);
        textureView(so + 2, h.color.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
        textureView(so + 3, h.luma.Get(), DXGI_FORMAT_R16_FLOAT);
        textureView(so + 4, t.motion.resource, motion);
        textureView(so + 5, t.depth.resource, depth);
        imageView(heap.cpuHandle(so + 6), dest.color.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
        imageView(heap.cpuHandle(so + 7), dest.luma.Get(), DXGI_FORMAT_R16_UINT);
        imageView(heap.cpuHandle(so + 8), dest.feature.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
        imageView(heap.cpuHandle(so + 9), out, outFormat);
        bufferView(so + 10, external ? externalHead : main[1].Get(), ou[10]);
        bufferView(so + 11, model.lut.Get(), ou[11]);
    }
}

Engine::Engine(Model& model, const Shape& shape) : Engine(model, shape, nullptr) {}

Engine::Engine(Model& model, const Shape& shape, const ExternalNetworkDx* external)
    : impl(std::make_unique<Impl>(*model.impl, shape)) {
    auto& e = *impl; auto* d = e.device;
    validateKShape(shape);
    if (external) {
        if (!external->tokens || !external->head) throw std::runtime_error("an external network needs its tokens and head resources");
        e.external = true; e.externalTokens = external->tokens; e.externalHead = external->head;
    }
    if (const char* trim = std::getenv("D4R_ENGINE_K_TRIM")) e.trim = std::strcmp(trim, "0") != 0;
    const auto base = shape.layers[0];
    e.params = buffer(d, ParamsBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    e.dummy = buffer(d, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto capacity = kActivationBytes(shape);
    if (!e.external) {
        for (unsigned i = 0; i < 2; ++i)
            e.main[i] = buffer(d, capacity[i], D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        for (unsigned i = 0; i < 5; ++i)
            e.skip[i] = buffer(d, kSkipBytes(shape, i), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
    for (auto& h : e.history) {
        h.color = texture(d, shape.outputWidth, shape.outputHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
        h.luma = texture(d, shape.outputWidth * 2, shape.outputHeight * 2, DXGI_FORMAT_R16_TYPELESS);
        h.feature = texture(d, base.w, base.h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    if (e.model.hasExposure) {
        const UINT64 blocks = UINT64((shape.outputWidth / 2 + 31) / 32) * ((shape.outputHeight / 2 + 31) / 32);
        e.exposureSums = buffer(d, blocks * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        e.exposureState = buffer(d, 8, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
    e.heap.create(d, Impl::BindingBase + Impl::Blocks * Impl::BlockSlots);
    const auto lb = layerBindings(e.model.layerCbv);
    if (!e.external)     // an external network runs the layers itself: their tables would have no buffers
    for (unsigned i = 0; i < KLayers; ++i) {
        const UINT s = i * Impl::LayerSlots;
        if (!e.model.layerCbv) e.bufferView(s + 0, e.params.Get(), lb[0], e.model.layerParamOffset(i), sizeof(KLayerParams));
        e.bufferView(s + 1, e.model.weights.Get(), lb[1]);
        e.bufferView(s + 2, e.main[i % 2].Get(), lb[2]);
        e.bufferView(s + 3, i >= 6 ? e.skip[10 - i].Get() : e.dummy.Get(), lb[3]);
        e.bufferView(s + 4, e.main[(i + 1) % 2].Get(), lb[4]);
        e.bufferView(s + 5, i < 5 ? e.skip[i].Get() : e.dummy.Get(), lb[5]);
    }
    for (unsigned i = 0; i < 2; ++i) {
        auto& h = e.history[i];
        const UINT c = Impl::ClearBase + i * 3;
        for (auto handle : {e.heap.cpuHandle(c), e.heap.cpuOnly(c)}) e.imageView(handle, h.color.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
        for (auto handle : {e.heap.cpuHandle(c + 1), e.heap.cpuOnly(c + 1)}) e.imageView(handle, h.luma.Get(), DXGI_FORMAT_R16_UINT);
        for (auto handle : {e.heap.cpuHandle(c + 2), e.heap.cpuOnly(c + 2)}) e.imageView(handle, h.feature.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
}
Engine::~Engine() = default;

void Engine::recordInitialize(ID3D12GraphicsCommandList* list) {
    auto& e = *impl;
    if (e.initialized) throw std::runtime_error("engine initialization already recorded");
    ComPtr<ID3D12GraphicsCommandList2> list2;
    DXCHECK(list->QueryInterface(IID_PPV_ARGS(&list2)));
    ID3D12DescriptorHeap* heaps[] = {e.heap.gpu.Get()};
    list->SetDescriptorHeaps(1, heaps);
    std::vector<D3D12_RESOURCE_BARRIER> toUav, toSrv;
    for (auto& h : e.history)
        for (auto* r : {h.color.Get(), h.luma.Get(), h.feature.Get()}) {
            toUav.push_back(transition(r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            toSrv.push_back(transition(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
        }
    list->ResourceBarrier(UINT(toUav.size()), toUav.data());
    const float zero[4] = {}; const UINT zeroes[4] = {};
    for (UINT i = 0; i < 6; ++i) {
        const UINT c = Impl::ClearBase + i;
        auto* r = (i % 3 == 0) ? e.history[i / 3].color.Get() : (i % 3 == 1) ? e.history[i / 3].luma.Get() : e.history[i / 3].feature.Get();
        if (i % 3 == 1) list->ClearUnorderedAccessViewUint(e.heap.gpuHandle(c), e.heap.cpuOnly(c), r, zeroes, 0, nullptr);
        else list->ClearUnorderedAccessViewFloat(e.heap.gpuHandle(c), e.heap.cpuOnly(c), r, zero, 0, nullptr);
    }
    list->ResourceBarrier(UINT(toSrv.size()), toSrv.data());
    if (e.model.hasExposure) {
        const D3D12_RESOURCE_BARRIER b = transition(e.exposureState.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        list->ResourceBarrier(1, &b);
        const float one = 1;
        UINT32 bits; std::memcpy(&bits, &one, 4);
        const auto at = e.exposureState->GetGPUVirtualAddress();
        D3D12_WRITEBUFFERIMMEDIATE_PARAMETER values[] = {{at, bits}, {at + 4, bits}};
        list2->WriteBufferImmediate(2, values, nullptr);
        const D3D12_RESOURCE_BARRIER back = transition(e.exposureState.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->ResourceBarrier(1, &back);
    }
    // Model tensor offsets are immutable. Upload them once; each frame updates
    // only the layer shapes/window shifts, in its own command list.
    auto writableParams = transition(e.params.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    list->ResourceBarrier(1, &writableParams);
    std::vector<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> constants;
    for (unsigned i = 0; i < KLayers; ++i) {
        const auto address = e.params->GetGPUVirtualAddress() + e.model.layerParamOffset(i) + sizeof(LayerShape);
        for (unsigned j = 0; j < e.model.data.offsets[i].size(); ++j)
            constants.push_back({address + j * sizeof(UINT32), e.model.data.offsets[i][j]});
    }
    list2->WriteBufferImmediate(UINT(constants.size()), constants.data(), nullptr);
    auto readableParams = transition(e.params.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(1, &readableParams);
    e.initialized = true;
}

void Engine::setFrame(const FrameTextures& textures, const InputParams& input, const OutputParams& output,
                      const std::array<LayerShape, KLayers>* windows, const ExposureControl* exposure, FrameOutput finalOutput) {
    auto& e = *impl;
    if (e.recorded != SIZE_MAX) throw std::runtime_error("the previous recording was neither submitted nor discarded");
    if (!e.captures.empty()) e.retire();
    if (finalOutput.resource) {
        const DXGI_FORMAT f = finalOutput.format != DXGI_FORMAT_UNKNOWN ? finalOutput.format : readFormat(finalOutput.resource->GetDesc().Format);
        if (f == DXGI_FORMAT_R10G10B10A2_UNORM && !e.model.outputRgb10)
            throw std::runtime_error("RGB10A2 output needs the packaged output_k_rgb10 shader");
    }
    ExposureParams exposureParams{};
    if (exposure) {
        if (!e.model.hasExposure) throw std::runtime_error("the model has no exposure shaders (recompile it with compile_k.py)");
        if (exposure->game && (!e.model.data.hasGameExposure || !textures.exposure.resource))
            throw std::runtime_error("game exposure needs recompiled K shaders and an exposure texture");
        exposureParams = makeKExposureParams(e.shape, *exposure);
    }
    if (!textures.color.resource || !textures.motion.resource || !textures.depth.resource) throw std::runtime_error("frame has a null texture");
    validateKFrame(e.shape, e.model.data, input, output, windows, finalOutput.resource != nullptr);
    if (windows) e.shape.layers = *windows;
    e.input = input; e.output = output;
    // makeKFrameParams sets the history uv over the 16-padded history of the Vulkan backend (as NGX's); this
    // backend's histories cover the output itself
    for (unsigned axis = 0; axis < 2; ++axis)
        e.input.invHist[axis] = e.output.invOut[axis] = 1.f / float(axis ? e.shape.outputHeight : e.shape.outputWidth);
    if (!e.model.data.hasRatioTables) e.output.flags &= ~FrameFlagTableMask;
    e.exposureOn = exposure != nullptr;
    if (exposure) e.exposure = exposureParams;
    if (!finalOutput.resource && !e.final) e.final = texture(e.device, e.shape.outputWidth, e.shape.outputHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
    auto same = [](const FrameTexture& a, const FrameTexture& b) { return a.resource == b.resource && a.format == b.format; };
    auto found = std::find_if(e.bindings.begin(), e.bindings.end(), [&](const Impl::Bindings& b) {
        return same(b.textures.color, textures.color) && same(b.textures.motion, textures.motion) && same(b.textures.depth, textures.depth) &&
               same(b.textures.exposure, textures.exposure) && b.output.resource == finalOutput.resource && b.output.format == finalOutput.format;
    });
    if (found == e.bindings.end()) {
        e.retire();
        UINT block = UINT(e.bindings.size());
        if (e.bindings.size() >= Impl::Blocks) {
            // reuse the least recently used block whose frames have all retired
            auto victim = e.bindings.end();
            for (auto it = e.bindings.begin(); it != e.bindings.end(); ++it)
                if (!it->inFlight && !it->pending && (victim == e.bindings.end() || it->lastUse < victim->lastUse)) victim = it;
            if (victim == e.bindings.end()) throw std::runtime_error("too many distinct frame resources in flight");
            block = victim->block;
            e.bindings.erase(victim);
        }
        Impl::Bindings b;
        b.textures = textures; b.output = finalOutput; b.block = block;
        ID3D12Resource* keep[] = {textures.color.resource, textures.motion.resource, textures.depth.resource, textures.exposure.resource, finalOutput.resource};
        for (unsigned i = 0; i < 5; ++i) b.retained[i] = keep[i];
        e.writeBindings(b);
        e.bindings.push_back(std::move(b));
        found = e.bindings.end() - 1;
    }
    e.current = size_t(found - e.bindings.begin());
    e.borrowedOutput = finalOutput.resource != nullptr;
    e.rgb10 = e.borrowedOutput && (finalOutput.format != DXGI_FORMAT_UNKNOWN ? finalOutput.format : readFormat(finalOutput.resource->GetDesc().Format)) == DXGI_FORMAT_R10G10B10A2_UNORM;
    e.reset = (input.flags & FrameFlagReset) != 0; e.frameSet = true;
}

void Engine::recordFrame(ID3D12GraphicsCommandList* list, const FrameProfiler& profiler) {
    auto& e = *impl;
    if (e.external) throw std::runtime_error("this engine's network is external: record the front and the back");
    if (!e.initialized || !e.model.uploaded || !e.frameSet) throw std::runtime_error("initialize/upload/setFrame before recording inference");
    if (e.recorded != SIZE_MAX) throw std::runtime_error("the previous recording was neither submitted nor discarded");
    ComPtr<ID3D12GraphicsCommandList2> list2;
    DXCHECK(list->QueryInterface(IID_PPV_ARGS(&list2)));
    auto& b = e.bindings[e.current];
    const UINT base = Impl::BindingBase + b.block * Impl::BlockSlots;
    const auto& src = e.history[e.previous]; const auto& dest = e.history[1 - e.previous];
    auto mark = [&](uint32_t boundary) { if (profiler.mark) profiler.mark(list, boundary, profiler.user); };
    mark(0);
    ID3D12DescriptorHeap* heaps[] = {e.heap.gpu.Get()};
    list->SetDescriptorHeaps(1, heaps);
    const auto srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    // entry: the parameters become writable, this frame's history destinations and the own output storage
    std::vector<D3D12_RESOURCE_BARRIER> entry, exit;
    entry.push_back(transition(e.params.Get(), uav, D3D12_RESOURCE_STATE_COPY_DEST));
    for (auto* r : {dest.color.Get(), dest.luma.Get(), dest.feature.Get()}) { entry.push_back(transition(r, srv, uav)); exit.push_back(transition(r, uav, srv)); }
    if (!e.borrowedOutput) { entry.push_back(transition(e.final.Get(), srv, uav)); exit.push_back(transition(e.final.Get(), uav, srv)); }
    if (e.reset) for (auto* r : {src.color.Get(), src.luma.Get(), src.feature.Get()}) entry.push_back(transition(r, srv, uav));
    list->ResourceBarrier(UINT(entry.size()), entry.data());
    // the frame's parameters live in the command list, so frames may be recorded while earlier ones are queued
    std::vector<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> values;
    const auto at = e.params->GetGPUVirtualAddress();
    auto put = [&](UINT64 offset, const void* data, size_t bytes) {
        for (size_t i = 0; i < bytes; i += 4) { UINT32 v; std::memcpy(&v, static_cast<const char*>(data) + i, 4); values.push_back({at + offset + i, v}); }
    };
    put(InputOffset, &e.input, sizeof(InputParams));
    put(OutputOffset, &e.output, sizeof(OutputParams));
    if (e.exposureOn) put(ExposureOffset, &e.exposure, sizeof(ExposureParams));
    // The offsets that follow the shape were initialized once in recordInitialize.
    for (unsigned i = 0; i < KLayers; ++i) {
        put(e.model.layerParamOffset(i), &e.shape.layers[i], sizeof(LayerShape));
    }
    list2->WriteBufferImmediate(UINT(values.size()), values.data(), nullptr);
    D3D12_RESOURCE_BARRIER written = transition(e.params.Get(), D3D12_RESOURCE_STATE_COPY_DEST, uav);
    list->ResourceBarrier(1, &written);
    if (e.reset) {
        const float zero[4] = {}; const UINT zeroes[4] = {};
        for (UINT i = 0; i < 6; ++i) {
            const UINT c = Impl::ClearBase + i;
            auto& h = e.history[i / 3];
            auto* r = (i % 3 == 0) ? h.color.Get() : (i % 3 == 1) ? h.luma.Get() : h.feature.Get();
            if (i % 3 == 1) list->ClearUnorderedAccessViewUint(e.heap.gpuHandle(c), e.heap.cpuOnly(c), r, zeroes, 0, nullptr);
            else list->ClearUnorderedAccessViewFloat(e.heap.gpuHandle(c), e.heap.cpuOnly(c), r, zero, 0, nullptr);
        }
        uavBarrier(list);
        std::vector<D3D12_RESOURCE_BARRIER> back;
        for (auto* r : {src.color.Get(), src.luma.Get(), src.feature.Get()}) back.push_back(transition(r, uav, srv));
        list->ResourceBarrier(UINT(back.size()), back.data());
    }
    uavBarrier(list);    // earlier writes to the borrowed output and the engine's buffers
    mark(1);
    if (e.exposureOn) {
        if (e.exposure.flags & 1u) {
            e.dispatch(list, e.model.exposure[0], base + Impl::ExposureSlot0, e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX);
            uavBarrier(list);
        }
        e.dispatch(list, e.model.exposure[1], base + Impl::ExposureSlot1, 1, 1);
        uavBarrier(list);
    }
    mark(2);
    std::string capturePath;
    if (!e.captureDirectory.empty() && std::filesystem::exists(e.captureDirectory + "/capture-now")) {
        std::filesystem::remove(e.captureDirectory + "/capture-now");
        capturePath = e.captureDirectory + "/frame-" + std::to_string(e.uses);
        const auto captureBase = capturePath;
        for (unsigned suffix = 1; std::filesystem::exists(capturePath); ++suffix)
            capturePath = captureBase + "-" + std::to_string(suffix);
        std::filesystem::create_directories(capturePath);
        e.captures.push_back(std::make_unique<FrameCapture>());
        e.recordingCapture = e.captures.back().get();
        auto add = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES state, const char* name) {
            e.recordingCapture->add(e.device, list, resource, state, capturePath + "/" + name);
        };
        add(e.params.Get(), uav, "params.bin");
        add(e.model.lut.Get(), srv, "lut.bin");
        add(b.textures.color.resource, srv, "color.bin");
        add(b.textures.motion.resource, srv, "motion.bin");
        add(b.textures.depth.resource, srv, "depth.bin");
        add(src.color.Get(), srv, "history-color.bin");
        add(src.luma.Get(), srv, "history-luma.bin");
        add(src.feature.Get(), srv, "history-feature.bin");
        std::ofstream shapeFile(capturePath + "/shape.bin", std::ios::binary);
        shapeFile.write(reinterpret_cast<const char*>(&e.shape), sizeof(e.shape));
        std::ofstream metadata(capturePath + "/textures.txt");
        for (const auto* texture : {&b.textures.color, &b.textures.motion, &b.textures.depth}) {
            const auto desc = texture->resource->GetDesc();
            metadata << desc.Width << ' ' << desc.Height << ' ' << unsigned(textureFormat(*texture)) << '\n';
        }
    }
    const auto lb = e.shape.layers[0];
    e.dispatch(list, e.model.input, base + Impl::InputSlot + e.previous * 12, UINT(lb.w) / 8, UINT(lb.h) / 8);
    mark(3);
    if (e.model.layerCbv) {
        auto readable = transition(e.params.Get(), uav, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        list->ResourceBarrier(1, &readable);
    }
    const auto groups = kLayerGroups(e.shape, e.trim);
    const unsigned layers = kRecordedLayers();
    for (unsigned i = 0; i < layers; ++i) {
        uavBarrier(list);
        e.dispatch(list, e.model.network[i], i * Impl::LayerSlots, groups[i][0], groups[i][1]);
        mark(4 + i);
    }
    if (e.model.layerCbv) {
        auto writable = transition(e.params.Get(), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, uav);
        list->ResourceBarrier(1, &writable);
    }
    uavBarrier(list);
    e.dispatch(list, e.rgb10 ? e.model.outputRgb10 : e.model.output, base + Impl::OutputSlot + e.previous * 12,
               (e.shape.outputWidth + 15) / 16, (e.shape.outputHeight + 15) / 16);
    mark(4 + KLayers);
    uavBarrier(list);
    if (e.recordingCapture) {
        e.recordingCapture->add(e.device, list, e.main[1].Get(), uav, capturePath + "/head.bin");
        e.recordingCapture->add(e.device, list, b.output.resource ? b.output.resource : e.final.Get(), uav, capturePath + "/output.bin");
        e.recordingCapture->add(e.device, list, dest.color.Get(), uav, capturePath + "/next-color.bin");
        e.recordingCapture->add(e.device, list, dest.luma.Get(), uav, capturePath + "/next-luma.bin");
        e.recordingCapture->add(e.device, list, dest.feature.Get(), uav, capturePath + "/next-feature.bin");
    }
    list->ResourceBarrier(UINT(exit.size()), exit.data());
    mark(KTimingStages);
    b.pending = true; b.lastUse = ++e.uses;
    e.recorded = e.current;
    e.previous = 1 - e.previous;
}


// The front of an external network's frame (the Vulkan Engine::recordFront): setup, exposure and the input stage, which
// writes the caller's tokens. The engine's own resources are back in their resting states when this returns; the
// tokens stay in COMMON and are never transitioned (hip_net_d3d12.h).
void Engine::recordFront(ID3D12GraphicsCommandList* list) {
    auto& e = *impl;
    if (!e.external) throw std::runtime_error("recordFront needs an engine with an external network");
    if (!e.initialized || !e.model.uploaded || !e.frameSet) throw std::runtime_error("initialize/upload/setFrame before recording inference");
    if (e.recorded != SIZE_MAX) throw std::runtime_error("the previous recording was neither submitted nor discarded");
    if (e.frontDone) throw std::runtime_error("recordFront before recordBack");
    ComPtr<ID3D12GraphicsCommandList2> list2;
    DXCHECK(list->QueryInterface(IID_PPV_ARGS(&list2)));
    auto& b = e.bindings[e.current];
    const UINT base = Impl::BindingBase + b.block * Impl::BlockSlots;
    const auto& src = e.history[e.previous]; const auto& dest = e.history[1 - e.previous];
    ID3D12DescriptorHeap* heaps[] = {e.heap.gpu.Get()};
    list->SetDescriptorHeaps(1, heaps);
    const auto srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    std::vector<D3D12_RESOURCE_BARRIER> entry, exit;
    entry.push_back(transition(e.params.Get(), uav, D3D12_RESOURCE_STATE_COPY_DEST));
    for (auto* r : {dest.color.Get(), dest.luma.Get(), dest.feature.Get()}) { entry.push_back(transition(r, srv, uav)); exit.push_back(transition(r, uav, srv)); }
    if (!e.borrowedOutput) { entry.push_back(transition(e.final.Get(), srv, uav)); exit.push_back(transition(e.final.Get(), uav, srv)); }
    if (e.reset) for (auto* r : {src.color.Get(), src.luma.Get(), src.feature.Get()}) entry.push_back(transition(r, srv, uav));
    list->ResourceBarrier(UINT(entry.size()), entry.data());
    std::vector<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> values;
    const auto at = e.params->GetGPUVirtualAddress();
    auto put = [&](UINT64 offset, const void* data, size_t bytes) {
        for (size_t i = 0; i < bytes; i += 4) { UINT32 v; std::memcpy(&v, static_cast<const char*>(data) + i, 4); values.push_back({at + offset + i, v}); }
    };
    put(InputOffset, &e.input, sizeof(InputParams));
    put(OutputOffset, &e.output, sizeof(OutputParams));
    if (e.exposureOn) put(ExposureOffset, &e.exposure, sizeof(ExposureParams));
    list2->WriteBufferImmediate(UINT(values.size()), values.data(), nullptr);
    D3D12_RESOURCE_BARRIER written = transition(e.params.Get(), D3D12_RESOURCE_STATE_COPY_DEST, uav);
    list->ResourceBarrier(1, &written);
    if (e.reset) {
        const float zero[4] = {}; const UINT zeroes[4] = {};
        for (UINT i = 0; i < 6; ++i) {
            const UINT c = Impl::ClearBase + i;
            auto& h = e.history[i / 3];
            auto* r = (i % 3 == 0) ? h.color.Get() : (i % 3 == 1) ? h.luma.Get() : h.feature.Get();
            if (i % 3 == 1) list->ClearUnorderedAccessViewUint(e.heap.gpuHandle(c), e.heap.cpuOnly(c), r, zeroes, 0, nullptr);
            else list->ClearUnorderedAccessViewFloat(e.heap.gpuHandle(c), e.heap.cpuOnly(c), r, zero, 0, nullptr);
        }
        uavBarrier(list);
        std::vector<D3D12_RESOURCE_BARRIER> back;
        for (auto* r : {src.color.Get(), src.luma.Get(), src.feature.Get()}) back.push_back(transition(r, uav, srv));
        list->ResourceBarrier(UINT(back.size()), back.data());
    }
    uavBarrier(list);    // earlier writes to the borrowed output and the engine's buffers
    if (e.exposureOn) {
        if (e.exposure.flags & 1u) {
            e.dispatch(list, e.model.exposure[0], base + Impl::ExposureSlot0, e.exposure.blocksX, e.exposure.blocks / e.exposure.blocksX);
            uavBarrier(list);
        }
        e.dispatch(list, e.model.exposure[1], base + Impl::ExposureSlot1, 1, 1);
        uavBarrier(list);
    }
    const auto lb = e.shape.layers[0];
    e.dispatch(list, e.model.input, base + Impl::InputSlot + e.previous * 12, UINT(lb.w) / 8, UINT(lb.h) / 8);
    uavBarrier(list);
    list->ResourceBarrier(UINT(exit.size()), exit.data());
    e.frontDone = true;
}

// The back of an external network's frame (the Vulkan Engine::recordBack): the final store, which reads the caller's
// head. Record and submit it only after the external network has finished; the head stays in COMMON.
void Engine::recordBack(ID3D12GraphicsCommandList* list) {
    auto& e = *impl;
    if (!e.external) throw std::runtime_error("recordBack needs an engine with an external network");
    if (!e.frontDone) throw std::runtime_error("recordFront before recordBack");
    auto& b = e.bindings[e.current];
    const UINT base = Impl::BindingBase + b.block * Impl::BlockSlots;
    const auto& dest = e.history[1 - e.previous];
    ID3D12DescriptorHeap* heaps[] = {e.heap.gpu.Get()};
    list->SetDescriptorHeaps(1, heaps);
    const auto srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    std::vector<D3D12_RESOURCE_BARRIER> entry, exit;
    for (auto* r : {dest.color.Get(), dest.luma.Get(), dest.feature.Get()}) { entry.push_back(transition(r, srv, uav)); exit.push_back(transition(r, uav, srv)); }
    if (!e.borrowedOutput) { entry.push_back(transition(e.final.Get(), srv, uav)); exit.push_back(transition(e.final.Get(), uav, srv)); }
    list->ResourceBarrier(UINT(entry.size()), entry.data());
    // the external network wrote the head after the front retired on the CPU: this UAV barrier invalidates the caches
    // the final store reads it through
    uavBarrier(list);
    e.dispatch(list, e.rgb10 ? e.model.outputRgb10 : e.model.output, base + Impl::OutputSlot + e.previous * 12,
               (e.shape.outputWidth + 15) / 16, (e.shape.outputHeight + 15) / 16);
    uavBarrier(list);
    list->ResourceBarrier(UINT(exit.size()), exit.data());
    b.pending = true; b.lastUse = ++e.uses;
    e.recorded = e.current;
    e.previous = 1 - e.previous;
    e.frontDone = false;
}

void Engine::submitted(ID3D12Fence* fence, uint64_t value) {
    auto& e = *impl;
    if (e.recorded == SIZE_MAX) throw std::runtime_error("no recording to report");
    auto& b = e.bindings[e.recorded];
    b.pending = false; b.inFlight = true; b.fence = fence; b.value = fence ? value : e.uses;
    if (e.recordingCapture) {
        e.recordingCapture->pending = false; e.recordingCapture->fence = fence;
        e.recordingCapture->value = fence ? value : e.uses;
        e.recordingCapture = nullptr;
    }
    e.recorded = SIZE_MAX; e.frameSet = false;
}
bool Engine::idle() const {
    auto& e = *impl;
    if (e.recorded != SIZE_MAX) return false;
    e.retire();
    return std::none_of(e.bindings.begin(), e.bindings.end(), [](const Impl::Bindings& b) { return b.inFlight; });
}
void Engine::discarded() {
    auto& e = *impl;
    if (e.recorded == SIZE_MAX) throw std::runtime_error("no recording to report");
    e.bindings[e.recorded].pending = false;
    if (e.recordingCapture) {
        e.captures.erase(std::remove_if(e.captures.begin(), e.captures.end(), [&](const auto& c) {
            return c.get() == e.recordingCapture;
        }), e.captures.end());
        e.recordingCapture = nullptr;
    }
    e.recorded = SIZE_MAX; e.frameSet = false;
    e.previous = 1 - e.previous;     // the history stays at the last submitted frame
    e.frontDone = false;
}
ID3D12Resource* Engine::outputImage() const { return impl->final.Get(); }
ID3D12Resource* Engine::headBuffer() const { return impl->external ? impl->externalHead : impl->main[1].Get(); }
ID3D12Resource* Engine::tokenBuffer() const { return impl->external ? impl->externalTokens : impl->main[0].Get(); }
uint64_t Engine::headBytes() const { auto s = impl->shape.layers[0]; return uint64_t(s.w) * s.h * 40 * 2; }
uint64_t Engine::networkTokenBytes(const Shape& shape) { const auto s = shape.layers[0]; return uint64_t(s.w) * s.h * 16 * 2; }
uint64_t Engine::networkHeadBytes(const Shape& shape) { const auto s = shape.layers[0]; return uint64_t(s.w) * s.h * 40 * 2; }
std::array<ID3D12Resource*, 3> Engine::historyImages() const {
    auto& h = impl->history[impl->previous]; return {h.color.Get(), h.luma.Get(), h.feature.Get()};
}
#include "runtime_m_rdna3.inc"
} // namespace d4r::dx
