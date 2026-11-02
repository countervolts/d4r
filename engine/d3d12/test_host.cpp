// Milestone 1 host-integration checks for the record-only backend (host.h). This program plays the game: it
// creates the device, queue, command lists and textures, and lends them to d4r. No model is involved; the
// diagnostic shader stands in for inference.
//   d4r-host-test [--debug] [--gpu-validation]
#include "host.h"
#include <windows.h>
#include <dxgi1_6.h>
#include <DirectXPackedVector.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using d4r::dx::Status;
using Microsoft::WRL::ComPtr;
namespace {
void check(HRESULT hr, const char* operation)
{
    if (FAILED(hr)) {
        char error[256];
        std::snprintf(error, sizeof(error), "%s: HRESULT 0x%08lx", operation, static_cast<unsigned long>(hr));
        throw std::runtime_error(error);
    }
}
#define CK(call) check((call), #call)
void expect(bool condition, const std::string& what)
{
    if (!condition) throw std::runtime_error(what);
}
void expectStatus(Status got, Status want, const char* what)
{
    if (got != want)
        throw std::runtime_error(std::string(what) + ": " + d4r::dx::statusName(got) + ", expected " + d4r::dx::statusName(want));
}

constexpr DXGI_FORMAT Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
struct Pixel { float v[4]; };
using Image = std::vector<Pixel>;      // row-major, width * height

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Event() { if (!handle) throw std::runtime_error("CreateEvent failed"); }
    ~Event() { CloseHandle(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

bool debugLayer = false, gpuValidation = false;

// The application side: its own device, one direct queue, a ring of allocators and one command list.
struct Game {
    static constexpr int Allocators = 8;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> validation;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence, gate;
    ComPtr<ID3D12CommandAllocator> allocators[Allocators];
    UINT64 allocatorValue[Allocators]{};
    ComPtr<ID3D12GraphicsCommandList> list;
    std::vector<ComPtr<ID3D12Resource>> garbage;   // upload buffers, released by idle()
    UINT64 value = 0, gateValue = 0;
    int next = 0;
    Event event;

    Game()
    {
        if (debugLayer) {
            ComPtr<ID3D12Debug> layer; CK(D3D12GetDebugInterface(IID_PPV_ARGS(&layer))); layer->EnableDebugLayer();
            ComPtr<ID3D12Debug1> layer1;
            if (gpuValidation && SUCCEEDED(layer.As(&layer1))) layer1->SetEnableGPUBasedValidation(TRUE);
        }
        ComPtr<IDXGIFactory4> factory; CK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        for (UINT i = 0; !device; ++i) {
            ComPtr<IDXGIAdapter1> adapter;
            const HRESULT result = factory->EnumAdapters1(i, &adapter);
            if (result == DXGI_ERROR_NOT_FOUND) throw std::runtime_error("no hardware D3D12 adapter");
            CK(result);
            DXGI_ADAPTER_DESC1 desc{}; CK(adapter->GetDesc1(&desc));
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
        }
        if (debugLayer) CK(device.As(&validation));
        D3D12_COMMAND_QUEUE_DESC q{}; q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CK(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
        for (auto& a : allocators) CK(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&a)));
        CK(device->CreateCommandList(0, q.Type, allocators[0].Get(), nullptr, IID_PPV_ARGS(&list)));
        CK(list->Close());
        CK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        CK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)));
    }
    ~Game() { try { idle(); } catch (...) {} }

    ID3D12GraphicsCommandList* begin()
    {
        const int a = next; next = (next + 1) % Allocators;
        wait(allocatorValue[a]);
        CK(allocators[a]->Reset()); CK(list->Reset(allocators[a].Get(), nullptr));
        current = a;
        return list.Get();
    }
    UINT64 submit()
    {
        CK(list->Close());
        ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
        CK(queue->Signal(fence.Get(), ++value));
        allocatorValue[current] = value;
        return value;
    }
    // Close the list without executing it, as an application that abandons a frame would.
    void abandon() { CK(list->Close()); allocatorValue[current] = 0; }
    // Queue work behind a CPU-controlled gate, so it cannot complete until open() is called.
    void closeGate() { CK(queue->Wait(gate.Get(), ++gateValue)); }
    void open() { CK(gate->Signal(gateValue)); }
    void wait(UINT64 v)
    {
        if (fence->GetCompletedValue() < v) {
            CK(fence->SetEventOnCompletion(v, event.handle));
            if (WaitForSingleObject(event.handle, 10000) != WAIT_OBJECT_0) throw std::runtime_error("GPU wait timed out");
        }
        CK(device->GetDeviceRemovedReason());
        checkValidation();
    }
    void idle() { wait(value); garbage.clear(); }
    void checkValidation()
    {
        if (!validation) return;
        const auto count = validation->GetNumStoredMessagesAllowedByRetrievalFilter();
        for (UINT64 i = 0; i < count; ++i) {
            SIZE_T bytes = 0; CK(validation->GetMessage(i, nullptr, &bytes));
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            CK(validation->GetMessage(i, message, &bytes));
            if (message->Severity == D3D12_MESSAGE_SEVERITY_ERROR || message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
                validation->ClearStoredMessages();
                throw std::runtime_error(std::string("D3D12 validation: ") + message->pDescription);
            }
        }
        validation->ClearStoredMessages();
    }

    ComPtr<ID3D12Resource> texture(UINT width, UINT height, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC t{};
        t.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; t.Width = width; t.Height = height;
        t.DepthOrArraySize = t.MipLevels = 1; t.Format = Format; t.SampleDesc.Count = 1;
        t.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> result;
        CK(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &t, state, nullptr, IID_PPV_ARGS(&result)));
        return result;
    }
    ComPtr<ID3D12Resource> buffer(UINT64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = type;
        D3D12_RESOURCE_DESC b{};
        b.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; b.Width = bytes; b.Height = b.DepthOrArraySize = b.MipLevels = 1;
        b.SampleDesc.Count = 1; b.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> result;
        CK(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &b, state, nullptr, IID_PPV_ARGS(&result)));
        return result;
    }
    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        list->ResourceBarrier(1, &barrier);
    }
    // Records an upload of `image` into a texture in COPY_DEST state.
    void upload(ID3D12Resource* target, const Image& image)
    {
        const auto desc = target->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{}; UINT64 total = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, nullptr, nullptr, &total);
        auto staging = buffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        uint8_t* mapped = nullptr; const D3D12_RANGE none{0, 0};
        CK(staging->Map(0, &none, reinterpret_cast<void**>(&mapped)));
        for (UINT y = 0; y < desc.Height; ++y)
            for (UINT x = 0; x < desc.Width; ++x)
                for (int c = 0; c < 4; ++c)
                    reinterpret_cast<uint16_t*>(mapped + layout.Offset + y * layout.Footprint.RowPitch)[x * 4 + c] =
                        DirectX::PackedVector::XMConvertFloatToHalf(image[y * desc.Width + x].v[c]);
        staging->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst{target, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}}, src{};
        src.pResource = staging.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = layout;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        garbage.push_back(staging);
    }
    // A texture prepared for d4r: uploaded, then left in the state the contract requires.
    ComPtr<ID3D12Resource> input(const Image& image, UINT width, UINT height, D3D12_RESOURCE_STATES state)
    {
        auto t = texture(width, height, D3D12_RESOURCE_STATE_COPY_DEST);
        upload(t.Get(), image);
        transition(t.Get(), D3D12_RESOURCE_STATE_COPY_DEST, state);
        return t;
    }
    // Synchronous readback of a texture in `state` (test-side only).
    Image read(ID3D12Resource* source, D3D12_RESOURCE_STATES state)
    {
        const auto desc = source->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{}; UINT64 total = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, nullptr, nullptr, &total);
        auto staging = buffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        begin();
        transition(source, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}}, dst{};
        dst.pResource = staging.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = layout;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        wait(submit());
        Image image(size_t(desc.Width) * desc.Height);
        uint8_t* mapped = nullptr; const D3D12_RANGE range{0, size_t(total)}, none{0, 0};
        CK(staging->Map(0, &range, reinterpret_cast<void**>(&mapped)));
        for (UINT y = 0; y < desc.Height; ++y)
            for (UINT x = 0; x < desc.Width; ++x)
                for (int c = 0; c < 4; ++c)
                    image[y * desc.Width + x].v[c] = DirectX::PackedVector::XMConvertHalfToFloat(
                        reinterpret_cast<const uint16_t*>(mapped + layout.Offset + y * layout.Footprint.RowPitch)[x * 4 + c]);
        staging->Unmap(0, &none);
        return image;
    }

private:
    int current = 0;
};

// Values stay small multiples of 1/2, so f16 results are exact.
Image pattern(UINT width, UINT height, int seed)
{
    Image image(size_t(width) * height);
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x)
            image[y * width + x] = {{float((x + seed) % 5), float((y + 2 * seed) % 7), float(seed % 3), 1}};
    return image;
}
Image filled(UINT width, UINT height, float value) { return Image(size_t(width) * height, Pixel{{value, value, value, value}}); }

// CPU model of the diagnostic stage: the expected output texture and committed history.
struct Reference {
    UINT width, height;
    Image history;
    bool valid = false;
    void frame(Image& output, UINT outputWidth, const Image& color, UINT colorWidth, UINT x0, UINT y0, float scale, bool reset)
    {
        if (history.empty()) history = filled(width, height, 0);
        for (UINT y = 0; y < height; ++y)
            for (UINT x = 0; x < width; ++x) {
                Pixel p{};
                for (int c = 0; c < 4; ++c)
                    p.v[c] = color[y * colorWidth + x].v[c] * scale + (reset || !valid ? 0.f : history[y * width + x].v[c]);
                history[y * width + x] = p;
                output[(y0 + y) * outputWidth + x0 + x] = p;
            }
        valid = true;
    }
};
void compare(const Image& got, const Image& want, const char* what)
{
    expect(got.size() == want.size(), std::string(what) + ": size");
    for (size_t i = 0; i < got.size(); ++i)
        for (int c = 0; c < 4; ++c)
            if (got[i].v[c] != want[i].v[c])
                throw std::runtime_error(std::string(what) + ": pixel " + std::to_string(i) + " channel " + std::to_string(c) +
                                         " is " + std::to_string(got[i].v[c]) + ", expected " + std::to_string(want[i].v[c]));
}

constexpr UINT OW = 64, OH = 48;           // output texture
constexpr UINT RX = 5, RY = 3, RW = 40, RH = 30;   // output rectangle (= render size)
constexpr auto SRV = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
constexpr auto UAV = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

std::unique_ptr<d4r::dx::Backend> backend(Game& game)
{
    std::string reason;
    auto b = d4r::dx::Backend::create(game.device.Get(), &reason);
    expect(b != nullptr, "Backend::create: " + reason);
    return b;
}
d4r::dx::Feature* feature(d4r::dx::Backend& b, uint32_t slots = 3)
{
    d4r::dx::Feature* f = nullptr;
    expectStatus(b.createFeature({RW, RH, slots}, &f), Status::ok, "createFeature");
    return f;
}
d4r::dx::FrameDesc frame(ID3D12Resource* color, ID3D12Resource* output, float scale, bool reset)
{
    return {color, output, RX, RY, RW, RH, scale, reset};
}

// One recording inside the application's own list, between its own work, with the output subrectangle.
void basic()
{
    Game game; auto b = backend(game); auto* f = feature(*b);
    const Image color = pattern(RW, RH, 1);
    Image want = filled(OW, OH, 7);
    game.begin();
    auto c = game.input(color, RW, RH, SRV);
    auto out = game.input(want, OW, OH, UAV);
    d4r::dx::Ticket t;
    expectStatus(f->record(game.list.Get(), frame(c.Get(), out.Get(), 0.5f, true), &t), Status::ok, "record");
    const UINT64 v = game.submit();
    expectStatus(f->submitted(t, game.fence.Get(), v), Status::ok, "submitted");
    Reference ref{RW, RH};
    ref.frame(want, OW, color, RW, RX, RY, 0.5f, true);
    compare(game.read(out.Get(), UAV), want, "output (pixels outside the rectangle must be unchanged)");
}

// A frame-paced sequence: no CPU wait except allocator reuse, a reset (cut) mid-sequence, varying inputs.
void sequence()
{
    Game game; auto b = backend(game); auto* f = feature(*b, 3);
    Image want = filled(OW, OH, -1);
    game.begin();
    auto out = game.input(want, OW, OH, UAV);
    game.wait(game.submit());
    Reference ref{RW, RH};
    for (int n = 0; n < 24; ++n) {      // the game keeps each frame's input for that frame only
        const Image color = pattern(RW, RH, n);
        const bool reset = n == 0 || n == 13;
        const float scale = n % 2 ? 0.5f : 1.f;
        game.begin();
        auto c = game.input(color, RW, RH, SRV);
        d4r::dx::Ticket t;
        Status s = f->record(game.list.Get(), frame(c.Get(), out.Get(), scale, reset), &t);
        if (s == Status::busy) {        // three frames in flight: wait for the oldest, as the game's pacing would
            game.wait(game.value - 2);
            s = f->record(game.list.Get(), frame(c.Get(), out.Get(), scale, reset), &t);
        }
        expectStatus(s, Status::ok, "record");
        expectStatus(f->submitted(t, game.fence.Get(), game.submit()), Status::ok, "submitted");
        ref.frame(want, OW, color, RW, RX, RY, scale, reset);
        if (n % 8 == 0) game.garbage.clear();   // the game's uploads; d4r keeps its own references
    }
    compare(game.read(out.Get(), UAV), want, "output after 24 frames");
}

// Per-recording memory is not reused before the fence completes, and no recording waits on the CPU.
void retirement()
{
    Game game; auto b = backend(game); auto* f = feature(*b, 3);
    Image want = filled(OW, OH, 0);
    game.begin();
    auto out = game.input(want, OW, OH, UAV);
    game.wait(game.submit());
    game.closeGate();
    Reference ref{RW, RH};
    std::vector<ComPtr<ID3D12Resource>> colors;
    for (int n = 0; n < 3; ++n) {
        const Image color = pattern(RW, RH, n + 3);
        game.begin();
        colors.push_back(game.input(color, RW, RH, SRV));
        d4r::dx::Ticket t;
        expectStatus(f->record(game.list.Get(), frame(colors.back().Get(), out.Get(), 1, n == 0), &t), Status::ok, "record");
        expectStatus(f->submitted(t, game.fence.Get(), game.submit()), Status::ok, "submitted");
        ref.frame(want, OW, color, RW, RX, RY, 1, n == 0);
    }
    expect(game.fence->GetCompletedValue() < game.value, "the gate must hold the queue");
    const Image fourth = pattern(RW, RH, 9);
    game.begin();
    auto c = game.input(fourth, RW, RH, SRV);
    d4r::dx::Ticket t;
    expectStatus(f->record(game.list.Get(), frame(c.Get(), out.Get(), 1, false), &t), Status::busy, "record with every slot queued");
    expect(f->inFlight() == 3, "three recordings in flight");
    colors.clear();                         // the game drops its inputs; queued recordings must keep them alive
    game.open();
    game.wait(game.value);
    expectStatus(f->record(game.list.Get(), frame(c.Get(), out.Get(), 1, false), &t), Status::ok, "record after retirement");
    expectStatus(f->submitted(t, game.fence.Get(), game.submit()), Status::ok, "submitted");
    ref.frame(want, OW, fourth, RW, RX, RY, 1, false);
    compare(game.read(out.Get(), UAV), want, "output after retirement");
}

// A recording abandoned before submission leaves committed history untouched; misuse is rejected.
void discarded()
{
    Game game; auto b = backend(game); auto* f = feature(*b);
    Image want = filled(OW, OH, 0);
    game.begin();
    auto out = game.input(want, OW, OH, UAV);
    game.wait(game.submit());
    Reference ref{RW, RH};
    const Image a = pattern(RW, RH, 1), skipped = pattern(RW, RH, 4), c = pattern(RW, RH, 2);

    game.begin();
    auto ta = game.input(a, RW, RH, SRV);
    d4r::dx::Ticket t;
    expectStatus(f->record(game.list.Get(), frame(ta.Get(), out.Get(), 1, true), &t), Status::ok, "record a");
    expectStatus(f->submitted(t, game.fence.Get(), game.submit()), Status::ok, "submitted a");
    ref.frame(want, OW, a, RW, RX, RY, 1, true);

    game.begin();
    auto tb = game.input(skipped, RW, RH, SRV);
    d4r::dx::Ticket abandoned, second;
    expectStatus(f->record(game.list.Get(), frame(tb.Get(), out.Get(), 1, false), &abandoned), Status::ok, "record b");
    expectStatus(f->record(game.list.Get(), frame(tb.Get(), out.Get(), 1, false), &second), Status::pending,
                 "second recording before b is resolved");
    game.abandon();
    expectStatus(f->discarded(abandoned), Status::ok, "discarded b");
    expectStatus(f->discarded(abandoned), Status::invalid, "discarding b twice");
    expectStatus(f->submitted(abandoned, game.fence.Get(), game.value + 1), Status::invalid, "submitting a discarded recording");

    game.begin();
    auto tc = game.input(c, RW, RH, SRV);
    expectStatus(f->record(game.list.Get(), frame(tc.Get(), out.Get(), 0.5f, false), &t), Status::ok, "record c");
    const UINT64 v = game.submit();
    expectStatus(f->submitted(t, game.fence.Get(), game.value - 1), Status::invalid, "a fence value that does not increase");
    expectStatus(f->submitted(t, game.gate.Get(), v), Status::invalid, "a different fence");
    expectStatus(f->submitted(t, game.fence.Get(), v), Status::ok, "submitted c");
    ref.frame(want, OW, c, RW, RX, RY, 0.5f, false);
    compare(game.read(out.Get(), UAV), want, "output after a discarded recording");

    game.begin();
    auto big = game.texture(OW, OH, SRV);
    expectStatus(f->record(game.list.Get(), {big.Get(), out.Get(), 30, 0, RW, RH, 1, false}, &t), Status::invalid,
                 "rectangle outside the output");
    expectStatus(f->record(game.list.Get(), {big.Get(), out.Get(), 0, 0, RW + 1, RH, 1, false}, &t), Status::invalid,
                 "size above the feature maximum");
    game.abandon();
}

// Releasing a feature with queued work defers destruction until the work retires.
void releaseQueued()
{
    Game game; auto b = backend(game); auto* f = feature(*b);
    Image want = filled(OW, OH, 3);
    game.begin();
    auto out = game.input(want, OW, OH, UAV);
    game.wait(game.submit());
    game.closeGate();
    const Image color = pattern(RW, RH, 5);
    game.begin();
    auto c = game.input(color, RW, RH, SRV);
    d4r::dx::Ticket t;
    expectStatus(f->record(game.list.Get(), frame(c.Get(), out.Get(), 2, true), &t), Status::ok, "record");
    expectStatus(f->submitted(t, game.fence.Get(), game.submit()), Status::ok, "submitted");
    c.Reset();
    b->release(f);
    expect(b->features() == 1 && b->pendingReleases() == 1, "a released feature with queued work stays alive");
    game.open();
    game.wait(game.value);
    b->collect();
    expect(b->features() == 0, "the feature is destroyed once its work retires");
    Reference ref{RW, RH};
    ref.frame(want, OW, color, RW, RX, RY, 2, true);
    compare(game.read(out.Get(), UAV), want, "output of a released feature");
}

// Two features share the backend's pipeline and keep separate history, recorded into one list.
void sharedBackend()
{
    Game game; auto b = backend(game); auto* f1 = feature(*b); auto* f2 = feature(*b);
    Image want1 = filled(OW, OH, 0), want2 = filled(OW, OH, 1);
    game.begin();
    auto out1 = game.input(want1, OW, OH, UAV), out2 = game.input(want2, OW, OH, UAV);
    game.wait(game.submit());
    Reference r1{RW, RH}, r2{RW, RH};
    for (int n = 0; n < 5; ++n) {
        const Image c1 = pattern(RW, RH, n), c2 = pattern(RW, RH, 10 - n);
        game.begin();
        auto t1 = game.input(c1, RW, RH, SRV), t2 = game.input(c2, RW, RH, SRV);
        d4r::dx::Ticket k1, k2;
        expectStatus(f1->record(game.list.Get(), frame(t1.Get(), out1.Get(), 1, n == 0), &k1), Status::ok, "record 1");
        expectStatus(f2->record(game.list.Get(), frame(t2.Get(), out2.Get(), 0.5f, n == 0), &k2), Status::ok, "record 2");
        const UINT64 v = game.submit();
        expectStatus(f1->submitted(k1, game.fence.Get(), v), Status::ok, "submitted 1");
        expectStatus(f2->submitted(k2, game.fence.Get(), v), Status::ok, "submitted 2");
        r1.frame(want1, OW, c1, RW, RX, RY, 1, n == 0);
        r2.frame(want2, OW, c2, RW, RX, RY, 0.5f, n == 0);
        game.wait(v);
    }
    compare(game.read(out1.Get(), UAV), want1, "feature 1");
    compare(game.read(out2.Get(), UAV), want2, "feature 2");
    b->release(f1); b->release(f2);
    expect(b->features() == 0, "both features released");
}

// After device removal, calls fail with deviceLost, released features are freed, and shutdown does not hang.
void deviceRemoval()
{
    Game game; auto b = backend(game); auto* f = feature(*b);
    game.begin();
    auto out = game.input(filled(OW, OH, 0), OW, OH, UAV);
    auto c = game.input(pattern(RW, RH, 1), RW, RH, SRV);
    game.wait(game.submit());
    game.closeGate();
    game.begin();
    d4r::dx::Ticket t;
    expectStatus(f->record(game.list.Get(), frame(c.Get(), out.Get(), 1, true), &t), Status::ok, "record");
    expectStatus(f->submitted(t, game.fence.Get(), game.submit()), Status::ok, "submitted");
    ComPtr<ID3D12Device5> device5;
    expect(SUCCEEDED(game.device.As(&device5)), "ID3D12Device5 is required for this test");
    if (game.validation) game.validation->SetMuteDebugOutput(TRUE);
    device5->RemoveDevice();
    expect(b->lost(), "removal is detected");
    // The list is closed: a lost backend must return before touching it.
    expectStatus(f->record(game.list.Get(), frame(c.Get(), out.Get(), 1, false), &t), Status::deviceLost, "record after removal");
    d4r::dx::Feature* other = nullptr;
    expectStatus(b->createFeature({RW, RH, 3}, &other), Status::deviceLost, "createFeature after removal");
    b->release(f);
    expect(b->features() == 0, "released features are freed on a removed device");
    b.reset();
    game.validation.Reset();
    game.value = 0;     // the game's own teardown does not wait on a removed device
}

// The native path must not load NVIDIA/CUDA/ZLUDA components or a translated D3D12.
void modules()
{
    for (const wchar_t* name : {L"nvcuda.dll", L"nvngx_dlss.dll", L"_nvngx.dll", L"nvngx.dll", L"zluda.dll", L"nvapi64.dll"})
        expect(!GetModuleHandleW(name), "unexpected module loaded");
    HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
    expect(d3d12 != nullptr, "d3d12.dll is loaded");
    wchar_t path[MAX_PATH]{}, system[MAX_PATH]{};
    GetModuleFileNameW(d3d12, path, MAX_PATH);
    GetSystemDirectoryW(system, MAX_PATH);
    expect(_wcsnicmp(path, system, wcslen(system)) == 0, "d3d12.dll comes from the system directory, not vkd3d");
}
}

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--debug") debugLayer = true;
        else if (option == "--gpu-validation") debugLayer = gpuValidation = true;
        else { std::fprintf(stderr, "usage: d4r-host-test [--debug] [--gpu-validation]\n"); return 2; }
    }
    const std::pair<const char*, std::function<void()>> tests[] = {
        {"basic recording and output rectangle", basic},
        {"frame-paced sequence with a cut", sequence},
        {"retirement-gated slot reuse", retirement},
        {"discarded recording and misuse", discarded},
        {"release with queued work", releaseQueued},
        {"two features, one backend", sharedBackend},
        {"no NVIDIA, ZLUDA or vkd3d modules", modules},
        {"device removal", deviceRemoval},      // last: removal can affect later devices in the process
    };
    int failures = 0;
    for (auto& [name, run] : tests) {
        try { run(); std::printf("PASS  %s\n", name); }
        catch (const std::exception& e) { ++failures; std::printf("FAIL  %s: %s\n", name, e.what()); }
    }
    std::printf("%d of %zu passed\n", int(std::size(tests)) - failures, std::size(tests));
    return failures ? 1 : 0;
}
