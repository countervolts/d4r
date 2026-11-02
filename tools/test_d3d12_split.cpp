// Exercise the native command-list split (d4r_d3d12_split.h) on a real device: recording after begin() goes into
// the second part, the second part waits for the fence while the first part has run, a method with a float
// register argument and stack arguments (ClearDepthStencilView) survives the forwarding stub, and a list reset
// before it was executed drops its split.
// usage: test_d3d12_split.exe (MinGW: g++ -std=c++20 -O2 -Itools tools/test_d3d12_split.cpp -ld3d12 -static)
#include "d4r_d3d12_split.h"
#include <wrl/client.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>

using Microsoft::WRL::ComPtr;
static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct Done { int executed = 0, discarded = 0; };
static void done(void* owner, bool executed) { ++(executed ? static_cast<Done*>(owner)->executed : static_cast<Done*>(owner)->discarded); }

int main() {
    try {
        ComPtr<ID3D12Device> device;
        check(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))), "device");
        const char* failure = "";
        check(d4r_split::install(device.Get(), &failure), failure);
        ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
        check(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))), "queue");
        ComPtr<ID3D12CommandAllocator> alloc[2];
        ComPtr<ID3D12GraphicsCommandList> lists[2];
        for (int i = 0; i < 2; ++i) {
            check(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc[i]))), "allocator");
            check(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[i].Get(), nullptr,
                                                      IID_PPV_ARGS(&lists[i]))), "list");
        }
        auto buffer = [&](D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out) {
            D3D12_HEAP_PROPERTIES hp{type};
            D3D12_RESOURCE_DESC rd{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 1024, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
            check(SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&out))), "buffer");
        };
        ComPtr<ID3D12Resource> upload, readback;
        buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload);
        buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, readback);
        uint32_t* up = nullptr;
        check(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))), "map");
        for (int i = 0; i < 256; ++i) up[i] = 0x1000 + i;
        // a 64x1 D32 depth target: cleared in the second part with depth 0.25 (xmm3) and stencil/rects on the stack
        D3D12_HEAP_PROPERTIES dh{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC td{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 1, 1, 1, DXGI_FORMAT_D32_FLOAT, {1, 0},
                               D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
        ComPtr<ID3D12Resource> depth;
        check(SUCCEEDED(device->CreateCommittedResource(&dh, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                        nullptr, IID_PPV_ARGS(&depth))), "depth");
        ComPtr<ID3D12DescriptorHeap> dsvHeap;
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
        check(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsvHeap))), "dsv heap");
        const D3D12_CPU_DESCRIPTOR_HANDLE dsv = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        device->CreateDepthStencilView(depth.Get(), nullptr, dsv);
        ComPtr<ID3D12Fence> fence, retire, idle;
        check(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))), "fence");
        check(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&retire))), "fence");
        check(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&idle))), "fence");
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        auto wait = [&](ID3D12Fence* f, uint64_t v, DWORD ms) {
            f->SetEventOnCompletion(v, event);
            return WaitForSingleObject(event, ms) == WAIT_OBJECT_0;
        };
        Done d;

        // the game's list: copy words 0..63, split, copy words 64..127 and clear the depth target
        lists[0]->CopyBufferRegion(readback.Get(), 0, upload.Get(), 0, 256);
        check(d4r_split::begin(lists[0].Get(), lists[1].Get(), fence.Get(), 1, retire.Get(), 1, done, &d), "begin");
        lists[0]->CopyBufferRegion(readback.Get(), 256, upload.Get(), 256, 256);
        lists[0]->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.25f, 0, 0, nullptr);
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
        b.Transition = {depth.Get(), 0, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE};
        lists[0]->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION src{depth.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION dst{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = {512, {DXGI_FORMAT_R32_TYPELESS, 64, 1, 1, 256}};
        lists[0]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        check(SUCCEEDED(lists[0]->Close()), "close");
        ID3D12CommandList* batch = lists[0].Get();
        queue->ExecuteCommandLists(1, &batch);
        queue->Signal(idle.Get(), 1);
        check(d.executed == 1, "executed callback");
        check(!wait(retire.Get(), 1, 300), "the second part ran before the fence was signalled");
        uint32_t* rb = nullptr;
        fence->Signal(1);
        check(wait(retire.Get(), 1, 10000) && wait(idle.Get(), 1, 10000), "second part never ran");
        check(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&rb))), "map readback");
        bool first = true, second = true, cleared = true;
        for (int i = 0; i < 64; ++i) first &= rb[i] == 0x1000u + i;
        for (int i = 64; i < 128; ++i) second &= rb[i] == 0x1000u + i;
        for (int i = 0; i < 64; ++i) { float v; std::memcpy(&v, &rb[128 + i], 4); cleared &= v == 0.25f; }
        readback->Unmap(0, nullptr);
        check(first, "first part's copy");
        check(second, "second part's copy (redirected recording)");
        check(cleared, "ClearDepthStencilView arguments through the stub");

        // discarded: reset before execution drops the split
        check(SUCCEEDED(alloc[0]->Reset()) && SUCCEEDED(lists[0]->Reset(alloc[0].Get(), nullptr)), "reset");
        check(SUCCEEDED(alloc[1]->Reset()) && SUCCEEDED(lists[1]->Reset(alloc[1].Get(), nullptr)), "reset second");
        check(d4r_split::begin(lists[0].Get(), lists[1].Get(), fence.Get(), 2, nullptr, 0, done, &d), "begin again");
        lists[0]->Close();
        lists[0]->Reset(alloc[0].Get(), nullptr);
        check(d.discarded == 1 && d.executed == 1, "discard callback");
        lists[0]->Close();
        batch = lists[0].Get();
        queue->ExecuteCommandLists(1, &batch);
        queue->Signal(idle.Get(), 2);
        check(wait(idle.Get(), 2, 10000), "a list no longer split runs without waiting");

        // the between-parts hook (d3dhip runs its network there): called once, and the second part runs even when the
        // hook signals nothing itself (the split always satisfies its own wait)
        check(SUCCEEDED(alloc[0]->Reset()) && SUCCEEDED(lists[0]->Reset(alloc[0].Get(), nullptr)), "reset");
        check(SUCCEEDED(alloc[1]->Reset()) && SUCCEEDED(lists[1]->Reset(alloc[1].Get(), nullptr)), "reset second");
        int between = 0;
        check(d4r_split::begin(lists[0].Get(), lists[1].Get(), fence.Get(), 3, retire.Get(), 2, done, &d,
                               [](void* owner, ID3D12CommandQueue*) { ++*static_cast<int*>(owner); }, &between),
              "begin with the between-parts hook");
        lists[0]->CopyBufferRegion(readback.Get(), 0, upload.Get(), 0, 256);
        check(SUCCEEDED(lists[0]->Close()), "close");
        batch = lists[0].Get();
        queue->ExecuteCommandLists(1, &batch);
        queue->Signal(idle.Get(), 3);
        check(wait(retire.Get(), 2, 10000) && wait(idle.Get(), 3, 10000), "the second part waits for the hook's return");
        check(between == 1, "the between-parts hook ran exactly once");
        check(d.executed == 2, "executed callback for the hook's split");
        std::printf("d4r_d3d12_split: all checks passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("d4r_d3d12_split: FAILED: %s\n", e.what());
        return 1;
    }
}
