// Native M/L raw-frame replay. Same CLI and raw formats as engine/replay_m.cpp.
// Upload/readback and CPU waits are outside the inference GPU timestamps.
#include "runtime_m_rdna3.h"
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
struct Image {
    ComPtr<ID3D12Resource> texture, upload;
    d4r::dx::FrameTexture frame;
    Image(ID3D12Device* device, const d4r::KCaseImage& image) {
        D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = image.w; desc.Height = image.h;
        desc.DepthOrArraySize = desc.MipLevels = 1; desc.Format = imageFormat(image.channels, image.bits); desc.SampleDesc.Count = 1;
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

} // namespace

int main(int argc, char** argv) try {
    int separator = 1; while (separator < argc && std::strcmp(argv[separator], "--")) ++separator;
    if (separator < 7 || separator > 8 || (argc - separator - 1) % 5 || argc - separator - 1 == 0) {
        std::cerr << "usage: d4r-m-d3d12 MODEL RW RH OW OH OUTPUT_PREFIX [REPEATS] -- COLOR MOTION DEPTH JX JY ...\n";
        return 2;
    }
    auto dimension = [&](int index) { const int v = std::stoi(argv[index]); if (v < 64 || v > 16384) throw std::runtime_error("invalid dimension"); return uint32_t(v); };
    const uint32_t rw = dimension(2), rh = dimension(3), ow = dimension(4), oh = dimension(5);
    const unsigned repeats = separator == 8 ? unsigned(std::stoul(argv[7])) : 0;
    if (repeats > 10000) throw std::runtime_error("repeat count exceeds 10000");
    const std::string prefix = argv[6];
    if (const auto parent = std::filesystem::path(prefix).parent_path(); !parent.empty()) std::filesystem::create_directories(parent);
    const bool profile = std::getenv("D4R_ENGINE_PROFILE") && std::string(std::getenv("D4R_ENGINE_PROFILE")) == "1";
    Context c; c.create(profile ? 2 + 8 * std::max(1u, repeats) : 2);
    struct Profile { Context* context; uint32_t first; } profileState{&c, 2};
    d4r::dx::FrameProfiler profiler{[](ID3D12GraphicsCommandList*, uint32_t boundary, void* user) {
        auto& p = *static_cast<Profile*>(user); p.context->stamp(p.first + boundary);
    }, &profileState};
    const char* stageNames[] = {"setup", "exposure", "input", "network", "expansion", "reconstruction", "downsample"};
    d4r::dx::rdna3::MModel model(c.device.Get(), argv[1]);
    d4r::dx::rdna3::GameUpscalerM engine(model, ow, oh, rw, rh);
    D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = ow; desc.Height = oh;
    desc.DepthOrArraySize = desc.MipLevels = 1; desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> output;
    CK(c.device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&output)));
    Readback readback(c.device.Get(), output.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const bool display = std::getenv("D4R_M_DISPLAY_MV") && std::string(std::getenv("D4R_M_DISPLAY_MV")) == "1";
    const bool testDiscard = std::getenv("D4R_M_TEST_DISCARD") && std::string(std::getenv("D4R_M_TEST_DISCARD")) == "1";
    const int resetAt = std::getenv("D4R_M_RESET_AT") ? std::stoi(std::getenv("D4R_M_RESET_AT")) : -1;
    std::ofstream timings(prefix + "-timings.csv"); timings << "frame,gpu_us\n";
    const unsigned count = unsigned((argc - separator - 1) / 5);
    for (unsigned f = 0; f < count; ++f) {
        const int at = separator + 1 + 5 * f;
        Image color(c.device.Get(), {argv[at], rw, rh, 4, 16});
        Image motion(c.device.Get(), {argv[at + 1], display ? ow : rw, display ? oh : rh, 2, 16});
        Image depth(c.device.Get(), {argv[at + 2], rw, rh, 1, 32});
        color.recordUpload(c.list.Get(), c.device.Get()); motion.recordUpload(c.list.Get(), c.device.Get()); depth.recordUpload(c.list.Get(), c.device.Get()); c.submit();
        d4r::dx::rdna3::MGameFrame frame;
        frame.color.resource = color.texture.Get(); frame.motion.resource = motion.texture.Get(); frame.depth.resource = depth.texture.Get(); frame.output.resource = output.Get();
        frame.output.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        frame.settings.jitter[0] = std::stof(argv[at + 3]); frame.settings.jitter[1] = std::stof(argv[at + 4]);
        frame.settings.reset = f == 0 || int(f) == resetAt; frame.settings.displayMotion = display;
        if (testDiscard) {
            auto discarded = frame; discarded.settings.jitter[0] = .25f; discarded.settings.jitter[1] = -.25f;
            engine.record(c.list.Get(), discarded); engine.discarded();
            CK(c.list->Close()); CK(c.allocator->Reset()); CK(c.list->Reset(c.allocator.Get(), nullptr));
        }
        c.stamp(0); profileState.first = 2; engine.record(c.list.Get(), frame, profile ? profiler : d4r::dx::FrameProfiler{});
        // The fence value belongs to this batch; every recording in a repeated
        // batch is guaranteed the same future completion value before reuse.
        engine.submitted(c.fence.Get(), c.value + 1);
        c.stamp(1); c.resolve(profile ? 10 : 2); readback.record(c.list.Get()); c.submit();
        readback.save(prefix + "-" + std::to_string(f) + ".rgba16f");
        const auto stamps = c.stamps(2); const double us = c.micros(stamps[0], stamps[1]);
        timings << f << ',' << std::setprecision(12) << us << '\n'; std::cout << "frame " << f << ": " << us << " us GPU\n";
        if (profile) {
            const auto stages = c.stamps(10);
            for (unsigned i = 0; i < 7; ++i) std::cout << "  " << stageNames[i] << ": " << c.micros(stages[2 + i], stages[3 + i]) << " us\n";
        }
        if (repeats && f + 1 == count) {
            frame.settings.reset = false;
            std::vector<double> trials;
            std::array<std::vector<double>, 7> stageTrials;
            for (unsigned trial = 0; trial < 12; ++trial) {
                c.stamp(0);
                for (unsigned n = 0; n < repeats; ++n) { profileState.first = 2 + 8 * n; engine.record(c.list.Get(), frame, profile ? profiler : d4r::dx::FrameProfiler{}); engine.submitted(c.fence.Get(), c.value + 1); }
                c.stamp(1); c.resolve(profile ? 2 + 8 * repeats : 2); c.submit(); const auto values = c.stamps(profile ? 2 + 8 * repeats : 2);
                const double value = c.micros(values[0], values[1]) / repeats;
                if (profile && trial >= 3) {
                    for (unsigned stage = 0; stage < 7; ++stage) {
                        double total = 0;
                        for (unsigned n = 0; n < repeats; ++n) total += c.micros(values[2 + 8 * n + stage], values[3 + 8 * n + stage]);
                        stageTrials[stage].push_back(total / repeats);
                    }
                }
                if (trial >= 3) { trials.push_back(value); timings << "batch-" << trial << ',' << value << '\n'; }
            }
            std::sort(trials.begin(), trials.end());
            if (profile) for (unsigned i = 0; i < 7; ++i) {
                auto& t = stageTrials[i]; std::sort(t.begin(), t.end());
                std::cout << "  " << stageNames[i] << " median: " << t[t.size() / 2] << " us\n";
            }
            std::cout << "Complete native " << (model.isPresetL() ? 'L' : 'M') << ": " << trials[trials.size() / 2] << " us GPU median; min " << trials.front() << "; " << repeats << " frames per batch\n";
        }
    }
    if (!engine.idle()) throw std::runtime_error("native M/L retained an unretired frame after the GPU wait");
    return 0;
} catch (const std::exception& error) { std::cerr << "d4r-m-d3d12: " << error.what() << '\n'; return 1; }
