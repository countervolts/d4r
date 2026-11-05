// d4r-k-d3d12: d4r-k (../replay.cpp) on the D3D12 backend. The same case files (loadKCase), and the same outputs
// (frame-N.312.bin: the output, .288/.296/.304: the history, .head: the network's head; timings.csv), so the
// two backends' results compare file by file (compare_k.py).
#include "runtime_d3d12.h"
#ifdef D4R_ENGINE_D3DHIP
#include "hip_net_d3d12.h"
#endif
#include <dxgi1_6.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using d4r::dx::ComPtr;
namespace {
void check(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        char text[32]; std::snprintf(text, sizeof(text), "0x%08lx", static_cast<unsigned long>(result));
        throw std::runtime_error(std::string(operation) + ": HRESULT " + text);
    }
}
#define CK(call) check((call), #call)

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    return barrier;
}
ComPtr<ID3D12Resource> buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES properties{heap};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = std::max<UINT64>(bytes, 256);
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> resource;
    CK(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource)));
    return resource;
}

struct Context {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12QueryHeap> timestamps;
    ComPtr<ID3D12Resource> stampReadback;
    uint64_t value = 0, frequency = 1;
    uint32_t queryCount = 0;
    HANDLE event = nullptr;
    ~Context() { if (fence && queue) wait(); if (event) CloseHandle(event); }
    void create(uint32_t queries) {
        if (const char* debug = std::getenv("D4R_D3D12_DEBUG"); debug && std::strcmp(debug, "1") == 0) {
            ComPtr<ID3D12Debug> layer;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&layer)))) layer->EnableDebugLayer();
        }
        ComPtr<IDXGIFactory6> factory;
        CK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        CK(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
        DXGI_ADAPTER_DESC1 desc; adapter->GetDesc1(&desc);
        CK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        std::wcerr << L"GPU: " << desc.Description << L'\n';
        D3D12_COMMAND_QUEUE_DESC q{D3D12_COMMAND_LIST_TYPE_DIRECT};
        CK(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
        CK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        CK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
        CK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        queryCount = queries;
        D3D12_QUERY_HEAP_DESC h{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, queries, 0};
        CK(device->CreateQueryHeap(&h, IID_PPV_ARGS(&timestamps)));
        stampReadback = buffer(device.Get(), uint64_t(queries) * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        CK(queue->GetTimestampFrequency(&frequency));
    }
    void wait() {
        if (fence->GetCompletedValue() < value) { CK(fence->SetEventOnCompletion(value, event)); WaitForSingleObject(event, INFINITE); }
    }
    // Closes, executes and waits; the list is open again afterwards.
    void submit() {
        CK(list->Close());
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        CK(queue->Signal(fence.Get(), ++value));
        wait();
        if (FAILED(device->GetDeviceRemovedReason())) check(device->GetDeviceRemovedReason(), "device removed");
        CK(allocator->Reset()); CK(list->Reset(allocator.Get(), nullptr));
    }
    void stamp(uint32_t index) { list->EndQuery(timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index); }
    void resolve(uint32_t count) { list->ResolveQueryData(timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, count, stampReadback.Get(), 0); }
    std::vector<uint64_t> stamps(uint32_t count) {
        std::vector<uint64_t> result(count);
        void* mapped = nullptr; D3D12_RANGE range{0, size_t(count) * 8};
        CK(stampReadback->Map(0, &range, &mapped));
        std::memcpy(result.data(), mapped, size_t(count) * 8);
        D3D12_RANGE none{0, 0}; stampReadback->Unmap(0, &none);
        return result;
    }
    double micros(uint64_t begin, uint64_t end) const { return double(end - begin) * 1e6 / double(frequency); }
};

DXGI_FORMAT imageFormat(int channels, int bits) {
    if (bits == 16) return channels == 1 ? DXGI_FORMAT_R16_FLOAT : channels == 2 ? DXGI_FORMAT_R16G16_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
    return channels == 1 ? DXGI_FORMAT_R32_FLOAT : channels == 2 ? DXGI_FORMAT_R32G32_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT;
}
// A texture and its copy footprint: uploads from and reads back through buffers with 256-byte row pitches.
struct Footprint {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT rows = 0; UINT64 rowBytes = 0, total = 0;
    Footprint(ID3D12Device* device, ID3D12Resource* texture) {
        const auto desc = texture->GetDesc();
        device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, &rows, &rowBytes, &total);
    }
};
// D4R_REPLAY_FORMATS=A,B,C: the case's images (in case order) as these DXGI formats instead of the float format
// their channels/bits name, the file's bytes uploaded unchanged: a live snapshot replayed with the game's own
// textures (prepare_live_k.py writes float planes; point a case at the raw color.bin / depth.bin for this). 20 or 19
// (D32_FLOAT_S8X24 / R32G8X24_TYPELESS) is a depth-stencil resource whose depth plane takes the file; 0 keeps the
// default for that image.
DXGI_FORMAT replayFormat(size_t index) {
    const char* list = std::getenv("D4R_REPLAY_FORMATS");
    if (!list) return DXGI_FORMAT_UNKNOWN;
    std::string s(list);
    for (size_t i = 0, at = 0; at <= s.size(); ++i) {
        const size_t comma = s.find(',', at);
        const std::string item = s.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        if (i == index) return item.empty() ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT(std::stoi(item));
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return DXGI_FORMAT_UNKNOWN;
}
struct Image {
    ComPtr<ID3D12Resource> texture, upload;
    d4r::dx::FrameTexture frame;
    Image(ID3D12Device* device, const d4r::KCaseImage& image, DXGI_FORMAT forced = DXGI_FORMAT_UNKNOWN) {
        D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = image.w; desc.Height = image.h;
        desc.DepthOrArraySize = desc.MipLevels = 1; desc.SampleDesc.Count = 1;
        desc.Format = forced != DXGI_FORMAT_UNKNOWN ? forced : imageFormat(image.channels, image.bits);
        if (desc.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT || desc.Format == DXGI_FORMAT_R32G8X24_TYPELESS) {
            desc.Format = DXGI_FORMAT_R32G8X24_TYPELESS;     // a game's depth buffer: typeless, depth-stencil capable
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        }
        CK(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture)));
        const auto data = d4r::readFile(image.file);
        Footprint f(device, texture.Get());
        if (data.size() != f.rowBytes * f.rows) throw std::runtime_error("unexpected image bytes: " + image.file);
        upload = buffer(device, f.total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        char* mapped = nullptr; D3D12_RANGE none{0, 0};
        CK(upload->Map(0, &none, reinterpret_cast<void**>(&mapped)));
        for (UINT y = 0; y < f.rows; ++y)
            std::memcpy(mapped + f.layout.Offset + UINT64(y) * f.layout.Footprint.RowPitch, data.data() + y * f.rowBytes, f.rowBytes);
        upload->Unmap(0, nullptr);
        frame = {texture.Get(), DXGI_FORMAT_UNKNOWN};
    }
    void recordUpload(ID3D12GraphicsCommandList* list, ID3D12Device* device) {
        Footprint f(device, texture.Get());
        D3D12_TEXTURE_COPY_LOCATION dst{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}; dst.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}; src.PlacedFootprint = f.layout;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        const auto b = transition(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &b);
    }
};
// A capture's history file (the case's `history` line) copied into one of the engine's history images, which rest in
// NON_PIXEL_SHADER_RESOURCE. NGX may pad its history surfaces below the image: the image's rows are taken from the
// top of the file.
struct Seed {
    ComPtr<ID3D12Resource> upload;
    ID3D12Resource* texture;
    Footprint f;
    Seed(ID3D12Device* device, ID3D12Resource* image, const std::string& file) : texture(image), f(device, image) {
        const auto data = d4r::readFile(file);
        if (data.size() < f.rowBytes * f.rows || data.size() % f.rowBytes != 0)
            throw std::runtime_error("history file does not match its image: " + file);
        upload = buffer(device, f.total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        char* mapped = nullptr; D3D12_RANGE none{0, 0};
        CK(upload->Map(0, &none, reinterpret_cast<void**>(&mapped)));
        for (UINT y = 0; y < f.rows; ++y)
            std::memcpy(mapped + f.layout.Offset + UINT64(y) * f.layout.Footprint.RowPitch, data.data() + y * f.rowBytes, f.rowBytes);
        upload->Unmap(0, nullptr);
    }
    void record(ID3D12GraphicsCommandList* list) {
        const D3D12_RESOURCE_BARRIER before[] = {
            transition(texture, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST)};
        list->ResourceBarrier(1, before);
        D3D12_TEXTURE_COPY_LOCATION dst{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}; dst.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}; src.PlacedFootprint = f.layout;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        const auto after = transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &after);
    }
};
// Reads a texture (in state) or a buffer (in state) back to a file after the submission.
struct Readback {
    ComPtr<ID3D12Resource> staging;
    ID3D12Resource* source = nullptr;
    Footprint* footprint = nullptr;
    std::unique_ptr<Footprint> owned;
    UINT64 bytes = 0;
    D3D12_RESOURCE_STATES state{};
    Readback(ID3D12Device* device, ID3D12Resource* resource, D3D12_RESOURCE_STATES resting, UINT64 bufferBytes = 0)
        : source(resource), state(resting) {
        if (resource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) bytes = bufferBytes;
        else { owned = std::make_unique<Footprint>(device, resource); footprint = owned.get(); bytes = owned->total; }
        staging = buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void record(ID3D12GraphicsCommandList* list) {
        auto before = transition(source, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &before);
        if (footprint) {
            D3D12_TEXTURE_COPY_LOCATION src{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}; src.SubresourceIndex = 0;
            D3D12_TEXTURE_COPY_LOCATION dst{staging.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}; dst.PlacedFootprint = footprint->layout;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        } else list->CopyBufferRegion(staging.Get(), 0, source, 0, bytes);
        auto after = transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        list->ResourceBarrier(1, &after);
    }
    void save(const std::string& file) {
        std::ofstream stream(file, std::ios::binary);
        char* mapped = nullptr; D3D12_RANGE range{0, size_t(bytes)};
        CK(staging->Map(0, &range, reinterpret_cast<void**>(&mapped)));
        if (footprint) {
            for (UINT y = 0; y < footprint->rows; ++y)
                stream.write(mapped + footprint->layout.Offset + UINT64(y) * footprint->layout.Footprint.RowPitch, std::streamsize(footprint->rowBytes));
        } else stream.write(mapped, std::streamsize(bytes));
        D3D12_RANGE none{0, 0}; staging->Unmap(0, &none);
        if (!stream) throw std::runtime_error("cannot write " + file);
    }
};

int run(const std::string& modelDir, const std::string& caseFile, const std::string& outputDir, int repeats, bool hip) {
    const bool readback = outputDir != "-";
    const bool profile = std::getenv("D4R_ENGINE_PROFILE") && std::string(std::getenv("D4R_ENGINE_PROFILE")) == "1";
    constexpr unsigned boundaries = d4r::KTimingStages + 1;
    // D4R_REPLAY_PHASES=1: four timestamps per timed frame (front start/end, back start/end; the whole frame for the
    // DXIL network) and per-phase medians over the eight-frame window cycle
    const bool phases = std::getenv("D4R_REPLAY_PHASES") && std::string(std::getenv("D4R_REPLAY_PHASES")) == "1";
    const uint32_t queryCount = std::max<uint32_t>(2 + (profile ? boundaries * repeats : 0), phases ? 2 + 4 * repeats : 0);
    Context context; context.create(std::max<uint32_t>(queryCount, 2 + boundaries));
    auto* device = context.device.Get();
    auto* list = context.list.Get();
    const std::array<const char*, d4r::KTimingStages> stageNames = {"setup", "exposure", "input",
        "enc0", "enc1", "enc2", "enc3", "enc4", "dec5", "dec4", "dec3", "dec2", "dec1", "dec0", "output", "finish"};
    struct Marks { Context* context; uint32_t first; } marks{&context, 2};
    d4r::dx::FrameProfiler profiler{[](ID3D12GraphicsCommandList*, uint32_t boundary, void* user) {
        auto& m = *static_cast<Marks*>(user);
        m.context->stamp(m.first + boundary);
    }, &marks};
    auto kcase = d4r::loadKCase(caseFile);
    const auto shape = kcase.shape;
    std::vector<std::unique_ptr<Image>> images;
    for (auto& image : kcase.images) images.push_back(std::make_unique<Image>(device, image, replayFormat(images.size())));

    const auto start = std::chrono::steady_clock::now();
    d4r::dx::Model model(device, modelDir);
    // The native HIP network between the front and the back (the trailing `hip` argument), as d4r-k's `hip` route.
    std::unique_ptr<d4r::dx::HipNetworkDx> hipNetwork;
    d4r::dx::ExternalNetworkDx external;
    if (hip) {
#ifdef D4R_ENGINE_D3DHIP
        try {
            hipNetwork = std::make_unique<d4r::dx::HipNetworkDx>(device, modelDir, shape.outputWidth, shape.outputHeight,
                                                                 d4r::dx::Engine::networkTokenBytes(shape),
                                                                 d4r::dx::Engine::networkHeadBytes(shape));
            external = {hipNetwork->tokens(), hipNetwork->head()};
            std::cout << "network: hip (native layers)\n";
        } catch (const std::exception& error) {
            std::cout << "network: hip unavailable (" << error.what() << "); using the DXIL network\n";
            hip = false;
        }
#else
        std::cout << "network: hip unavailable (built without the HIP SDK: set D4R_ROCM_DIR); using the DXIL network\n";
        hip = false;
#endif
    }
    d4r::dx::Engine engine(model, shape, hip ? &external : nullptr);
#ifdef D4R_ENGINE_D3DHIP
    if (hipNetwork && std::getenv("D4R_REPLAY_HIP_PHASES")) {
        std::cout << "HIP network per window phase (us): graph, kernels one by one\n";
        const auto per = hipNetwork->debugPhaseMicros(25);
        for (size_t k = 0; k < per.size(); ++k) std::printf("  phase %zu: %8.1f %8.1f\n", k, per[k].first, per[k].second);
        const auto nodes = hipNetwork->debugNodeMicros(0, 25);
        std::cout << "phase 0 launches (us, network order):";
        for (size_t k = 0; k < nodes.size(); ++k) std::printf("%s%zu:%.1f", k % 8 ? "  " : "\n  ", k, nodes[k]);
        std::cout << '\n';
    }
#endif
    model.recordUpload(list); engine.recordInitialize(list);
    for (auto& image : images) image->recordUpload(list, device);
    // A case captured mid-sequence starts from NGX's history (color, luma, feature): the first frame reads the
    // images historyImages() names before any frame has been recorded.
    std::vector<std::unique_ptr<Seed>> seeds;
    if (!kcase.history[0].empty()) {
        const auto history = engine.historyImages();
        for (int i = 0; i < 3; ++i) seeds.push_back(std::make_unique<Seed>(device, history[i], kcase.history[i]));
        for (auto& seed : seeds) seed->record(list);
    }
    context.submit();
    std::cerr << "Model/engine setup: " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << " ms\n";

    // D4R_REPLAY_OUTPUT_FORMAT=N: the final store goes to a borrowed output of DXGI format N (24: R10G10B10A2, a
    // game's swap-chain-style target, which runs output_k_rgb10), as in a game, instead of the engine's own RGBA16F
    ComPtr<ID3D12Resource> borrowedOutput;
    d4r::dx::FrameOutput finalOutput{};
    if (const char* format = std::getenv("D4R_REPLAY_OUTPUT_FORMAT"); format && *format) {
        D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = shape.outputWidth; desc.Height = shape.outputHeight;
        desc.DepthOrArraySize = desc.MipLevels = 1; desc.Format = DXGI_FORMAT(std::stoi(format)); desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        CK(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           nullptr, IID_PPV_ARGS(&borrowedOutput)));
        finalOutput = {borrowedOutput.Get(), desc.Format};
        std::cout << "final output: borrowed, DXGI format " << unsigned(desc.Format) << "\n";
    }
    uint32_t evaluation = 0;
    auto setFrame = [&](const d4r::KCaseFrame& f) {
        if (f.input.flags & d4r::FrameFlagReset) evaluation = 0;
        const auto windows = f.autoWindows ? d4r::makeKWindows(shape, evaluation) : f.windows;
        engine.setFrame({images[f.color]->frame, images[f.motion]->frame, images[f.depth]->frame, {}}, f.input, f.output, &windows,
                        nullptr, finalOutput);
        ++evaluation;
    };
    // d3dhip's handoff: GPU-side through a shared fence (the default), or CPU-serialized (D4R_D3DHIP_SYNC=cpu)
    const char* syncMode = std::getenv("D4R_D3DHIP_SYNC");
    const bool fenced = hipNetwork && !(syncMode && std::string(syncMode) == "cpu");
    ComPtr<ID3D12Fence> shared;
    uint64_t sharedValue = 0;
    if (fenced) {
        CK(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&shared)));
        hipNetwork->useFence(shared.Get());
        std::cout << "d3dhip sync: GPU fence (shared D3D12 fence imported into HIP)\n";
    }
    auto recordFrameSplit = [&](uint32_t frame) {
        marks.first = 2 + frame * boundaries;
        engine.recordFront(list);
        if (fenced) {
            // the front runs, HIP waits for it on the GPU, the rest of `list` waits for HIP: no CPU wait
            CK(list->Close());
            ID3D12CommandList* lists[] = {list};
            context.queue->ExecuteCommandLists(1, lists);
            CK(context.queue->Signal(shared.Get(), ++sharedValue));
            hipNetwork->launchAfter(sharedValue, sharedValue + 1);
            CK(context.queue->Wait(shared.Get(), ++sharedValue));
            CK(context.queue->Signal(context.fence.Get(), ++context.value));
            context.wait();                     // only to reuse the one allocator; the GPU order is the fence's
            CK(context.allocator->Reset()); CK(list->Reset(context.allocator.Get(), nullptr));
        } else {
            context.submit();                   // the front retires on the CPU first
            hipNetwork->run();
        }
        engine.recordBack(list);
        engine.submitted(context.fence.Get(), context.value + 1);
    };
    std::unique_ptr<Readback> result, color, luma, feature, head, tokens;
    std::ofstream timings, stages;
    if (readback) {
        std::filesystem::create_directories(outputDir);
        timings.open(outputDir + "/timings.csv"); timings << "frame,gpu_us\n";
        if (profile) { stages.open(outputDir + "/stages.csv"); stages << "frame"; for (auto name : stageNames) stages << "," << name << "_us"; stages << '\n'; }
    }
    auto stageMicros = [&](uint32_t count) {
        auto times = context.stamps(2 + boundaries * count);
        std::array<double, d4r::KTimingStages> values{};
        for (uint32_t frame = 0; frame < count; ++frame)
            for (unsigned stage = 0; stage < values.size(); ++stage)
                values[stage] += context.micros(times[2 + frame * boundaries + stage], times[2 + frame * boundaries + stage + 1]) / count;
        return values;
    };
    const auto srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    for (unsigned i = 0; i < kcase.frames.size(); ++i) {
        setFrame(kcase.frames[i]);
        context.stamp(0);
        if (hipNetwork) { recordFrameSplit(0); context.stamp(1); }
        else { engine.recordFrame(list, profile ? profiler : d4r::dx::FrameProfiler{}); engine.submitted(context.fence.Get(), context.value + 1); context.stamp(1); }
        if (readback) {
            // the history images alternate: the readbacks follow this frame's destination set
            auto history = engine.historyImages();
            result = borrowedOutput ? std::make_unique<Readback>(device, borrowedOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                                    : std::make_unique<Readback>(device, engine.outputImage(), srv);
            color = std::make_unique<Readback>(device, history[0], srv);
            luma = std::make_unique<Readback>(device, history[1], srv);
            feature = std::make_unique<Readback>(device, history[2], srv);
            // an external network's buffers rest in COMMON; the engine's own in UNORDERED_ACCESS
            const auto networkState = hipNetwork ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            head = std::make_unique<Readback>(device, engine.headBuffer(), networkState, engine.headBytes());
            const auto base = shape.layers[0];
            tokens = std::make_unique<Readback>(device, engine.tokenBuffer(), networkState, uint64_t(base.w) * base.h * 16 * 2);
            for (auto* r : {result.get(), color.get(), luma.get(), feature.get(), head.get(), tokens.get()}) r->record(list);
        }
        context.resolve(profile ? 2 + boundaries : 2);
        context.submit();
        const auto times = context.stamps(2);
        const double us = context.micros(times[0], times[1]);
        if (readback) timings << i << ',' << us << '\n';
        if (profile) {
            auto values = stageMicros(1);
            if (readback) { stages << i; for (auto value : values) stages << ',' << value; stages << '\n'; }
            if (repeats == 1) for (unsigned stage = 0; stage < values.size(); ++stage)
                std::cout << "  " << stageNames[stage] << ": " << std::fixed << std::setprecision(1) << values[stage] << " us\n";
        }
        if (hipNetwork)
            std::cout << "frame " << i << ": " << std::fixed << std::setprecision(1) << us << " us GPU (front+network+back; network "
                      << hipNetwork->lastMilliseconds() << " ms hip)\n";
        else
            std::cout << "frame " << i << ": " << std::fixed << std::setprecision(1) << us << " us GPU\n";
        if (readback) {
            const std::string prefix = outputDir + "/frame-" + std::to_string(i);
            result->save(prefix + ".312.bin"); color->save(prefix + ".288.bin"); luma->save(prefix + ".296.bin");
            feature->save(prefix + ".304.bin"); head->save(prefix + ".head.bin"); tokens->save(prefix + ".tokens.bin");
        }
    }
    if (repeats > 1) {
        // the last input as successive frames, as d4r-k
        std::vector<double> gpu, wall, networkMilliseconds;
        std::array<std::vector<double>, d4r::KTimingStages> stageTrials;
        // phase -> {front, network gap, back, whole frame} samples (D4R_REPLAY_PHASES)
        std::array<std::array<std::vector<double>, 4>, 8> phaseSamples;
        std::vector<uint32_t> phaseOf(repeats);
        auto collectPhases = [&](bool split) {
            const auto t = context.stamps(2 + 4 * uint32_t(repeats));
            for (int i = 0; i < repeats; ++i) {
                const uint64_t* q = &t[2 + 4 * i];
                auto& ph = phaseSamples[phaseOf[i] % 8];
                if (split) {
                    ph[0].push_back(context.micros(q[0], q[1])); ph[1].push_back(context.micros(q[1], q[2]));
                    ph[2].push_back(context.micros(q[2], q[3]));
                }
                ph[3].push_back(context.micros(q[0], q[3]));
            }
        };
        // d3dhip with GPU-side ordering (`fenced`, above): one shared fence orders front -> HIP network -> back on the
        // GPU, so the CPU only waits once per batch; each frame records into its own pair of lists
        std::vector<ComPtr<ID3D12CommandAllocator>> allocators;
        std::vector<ComPtr<ID3D12GraphicsCommandList>> lists;
        if (fenced) {
            for (int i = 0; i < 2 * repeats; ++i) {
                ComPtr<ID3D12CommandAllocator> allocator; ComPtr<ID3D12GraphicsCommandList> commands;
                CK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
                CK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)));
                CK(commands->Close());
                allocators.push_back(allocator); lists.push_back(commands);
            }
        }
        for (unsigned trial = 0; trial < 12; ++trial) {
            double hipTotal = 0;
            if (fenced) {
                const auto begin = std::chrono::steady_clock::now();
                auto* queue = context.queue.Get();
                for (int i = 0; i < repeats; ++i) {
                    setFrame(kcase.frames.back());
                    auto* front = lists[2 * i].Get(); auto* back = lists[2 * i + 1].Get();
                    CK(allocators[2 * i]->Reset()); CK(front->Reset(allocators[2 * i].Get(), nullptr));
                    CK(allocators[2 * i + 1]->Reset()); CK(back->Reset(allocators[2 * i + 1].Get(), nullptr));
                    phaseOf[i] = evaluation - 1;
                    if (i == 0) front->EndQuery(context.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    if (phases) front->EndQuery(context.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 + 4 * i);
                    engine.recordFront(front);
                    if (phases) front->EndQuery(context.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 + 4 * i + 1);
                    CK(front->Close());
                    ID3D12CommandList* first[] = {front};
                    queue->ExecuteCommandLists(1, first);
                    CK(queue->Signal(shared.Get(), ++sharedValue));            // the tokens are written
                    hipNetwork->launchAfter(sharedValue, sharedValue + 1);    // HIP: wait, network, signal
                    CK(queue->Wait(shared.Get(), ++sharedValue));              // the head is written
                    if (phases) back->EndQuery(context.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 + 4 * i + 2);
                    engine.recordBack(back);
                    if (phases) back->EndQuery(context.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 + 4 * i + 3);
                    if (i == repeats - 1) {
                        back->EndQuery(context.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        back->ResolveQueryData(context.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                                               phases ? 2 + 4 * repeats : 2, context.stampReadback.Get(), 0);
                    }
                    CK(back->Close());
                    ID3D12CommandList* second[] = {back};
                    queue->ExecuteCommandLists(1, second);
                    engine.submitted(context.fence.Get(), context.value + 1);
                }
                CK(queue->Signal(context.fence.Get(), ++context.value));
                context.wait();
                hipNetwork->finish();
                if (FAILED(device->GetDeviceRemovedReason())) check(device->GetDeviceRemovedReason(), "device removed");
                const double cpu = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / repeats;
                const auto times = context.stamps(2);
                if (trial >= 3) { gpu.push_back(context.micros(times[0], times[1]) / repeats); wall.push_back(cpu);
                                  networkMilliseconds.push_back(hipNetwork->lastMilliseconds());
                                  if (phases) collectPhases(true); }
                continue;
            }
            if (hipNetwork) {
                // CPU-serialized: each frame is a front submission and wait, the HIP graph, then the back
                double total = 0; const auto begin = std::chrono::steady_clock::now();
                for (int i = 0; i < repeats; ++i) {
                    setFrame(kcase.frames.back());
                    context.stamp(0);
                    recordFrameSplit(uint32_t(i));
                    context.stamp(1);
                    context.resolve(2);
                    context.submit();
                    const auto times = context.stamps(2);
                    total += context.micros(times[0], times[1]);
                    hipTotal += hipNetwork->lastMilliseconds();
                }
                const double cpu = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / repeats;
                if (trial >= 3) { gpu.push_back(total / repeats); wall.push_back(cpu); networkMilliseconds.push_back(hipTotal / repeats); }
                continue;
            }
            context.stamp(0);
            for (int i = 0; i < repeats; ++i) {
                setFrame(kcase.frames.back()); phaseOf[i] = evaluation - 1;
                if (phases) context.stamp(2 + 4 * i);
                engine.recordFrame(list, profile ? profiler : d4r::dx::FrameProfiler{});
                if (phases) context.stamp(2 + 4 * i + 3);
                engine.submitted(context.fence.Get(), context.value + 1);
            }
            context.stamp(1);
            context.resolve(phases ? 2 + 4 * repeats : profile ? 2 + boundaries * repeats : 2);
            const auto t0 = std::chrono::steady_clock::now();
            context.submit();
            const double cpu = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / repeats;
            if (trial >= 3) {
                const auto times = context.stamps(2);
                gpu.push_back(context.micros(times[0], times[1]) / repeats); wall.push_back(cpu);
                if (phases) collectPhases(false);
                if (profile) { auto values = stageMicros(uint32_t(repeats)); for (unsigned s = 0; s < values.size(); ++s) stageTrials[s].push_back(values[s]); }
            }
        }
        std::sort(gpu.begin(), gpu.end()); std::sort(wall.begin(), wall.end());
        if (hipNetwork) {
            std::sort(networkMilliseconds.begin(), networkMilliseconds.end());
            std::cout << "full native frame (hip" << (fenced ? ", GPU fence" : ", CPU-serialized") << "): " << gpu[gpu.size() / 2]
                      << " us GPU median (min " << gpu.front() << "); "
                      << wall[wall.size() / 2] << " us submit+wait per frame; network "
                      << networkMilliseconds[networkMilliseconds.size() / 2] << " ms hip median\n";
        } else
            std::cout << "full native frame: " << gpu[gpu.size() / 2] << " us GPU median (min " << gpu.front() << "); "
                      << wall[wall.size() / 2] << " us amortized submit+wait; " << repeats << " frames/submit\n";
        if (phases) {
            auto median = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            auto maxOf = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
            std::cout << "per window phase (median us; max of the whole frame):\n";
            std::cout << "  phase   front  network    back   frame   max\n";
            for (unsigned k = 0; k < 8; ++k) {
                const auto& ph = phaseSamples[k];
                std::printf("  %5u %7.1f %8.1f %7.1f %7.1f %7.1f\n", k, median(ph[0]), median(ph[1]), median(ph[2]),
                            median(ph[3]), maxOf(ph[3]));
            }
        }
        if (profile) {
            std::cout << "stage GPU medians (us/frame, includes preceding dependencies; timestamp instrumentation enabled):\n";
            for (unsigned s = 0; s < stageTrials.size(); ++s) {
                auto& values = stageTrials[s]; std::sort(values.begin(), values.end());
                std::cout << "  " << stageNames[s] << ": " << values[values.size() / 2] << '\n';
            }
        }
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        // [TIMING_REPEATS] [hip]: `hip` runs the network on the model's native HIP layers
        const bool hip = argc >= 2 && !std::strcmp(argv[argc - 1], "hip");
        const int used = argc - int(hip);
        if (used != 4 && used != 5) {
            std::cerr << "usage: d4r-k-d3d12 MODEL_DIR CASE.txt OUTPUT_DIR|- [TIMING_REPEATS] [hip]\n"
                         "As d4r-k, on the D3D12 backend (the model needs its DXIL: compile_k.py on Windows).\n"
                         "hip: the native HIP network between the front and the back (the model needs hipnet.bin and hip/).\n"
                         "D4R_ENGINE_PROFILE=1: stage timings; D4R_D3D12_DEBUG=1: the D3D12 debug layer.\n";
            return 2;
        }
        const int repeats = used == 5 ? std::stoi(argv[4]) : 1;
        if (repeats < 1 || repeats > 1000) throw std::runtime_error("timing repeats must be 1..1000");
        return run(argv[1], argv[2], argv[3], repeats, hip);
    } catch (const std::exception& e) { std::cerr << "d4r-k-d3d12: " << e.what() << '\n'; return 1; }
}
