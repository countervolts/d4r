// Exercise the actual game submission hooks, including delayed execution,
// discarded lists, multiple lists in a batch and per-submission timestamp rate.
#include "d4r_d3d12_submit.h"
#include <wrl/client.h>
#include <cstdio>
#include <stdexcept>

using Microsoft::WRL::ComPtr;
static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Owner {
    unsigned submitted = 0, discarded = 0;
    ComPtr<ID3D12Fence> fence;
    uint64_t value = 0, frequency = 0;
};
static d4r_submit::Pending pending(Owner& owner) {
    return {&owner, [](void* raw, ID3D12Fence* fence, uint64_t value, uint64_t frequency) {
        auto& o = *static_cast<Owner*>(raw); ++o.submitted;
        o.fence = fence; o.value = value; o.frequency = frequency;
    }, [](void* raw) { ++static_cast<Owner*>(raw)->discarded; }};
}
static void wait(Owner& owner) {
    check(owner.fence && owner.value && owner.frequency, "submission fence/timestamp frequency missing");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    check(event != nullptr, "event");
    check(SUCCEEDED(owner.fence->SetEventOnCompletion(owner.value, event)), "fence wait");
    const auto result = WaitForSingleObject(event, 10000); CloseHandle(event);
    check(result == WAIT_OBJECT_0, "GPU timeout");
}
int main() {
    try {
        ComPtr<ID3D12Device> device;
        check(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))), "device");
        ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        check(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))), "queue");
        ComPtr<ID3D12CommandAllocator> alloc[2];
        ComPtr<ID3D12GraphicsCommandList> lists[2];
        for (int i = 0; i < 2; ++i) {
            check(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc[i]))), "allocator");
            check(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[i].Get(), nullptr,
                                                    IID_PPV_ARGS(&lists[i]))), "list");
        }
        check(d4r_submit::install(device.Get()), "install submission hooks");
        Owner a, b;
        d4r_submit::track(lists[0].Get(), pending(a));
        check(a.submitted == 0 && d4r_submit::pending(&a), "unsubmitted list completed early");
        check(SUCCEEDED(lists[0]->Close()), "close");
        ID3D12CommandList* batch[] = {lists[0].Get(), lists[1].Get()};
        queue->ExecuteCommandLists(1, batch); wait(a);
        check(a.submitted == 1 && !d4r_submit::pending(&a), "submission callback");
        check(SUCCEEDED(alloc[0]->Reset()) && SUCCEEDED(lists[0]->Reset(alloc[0].Get(), nullptr)), "reset executed list");
        d4r_submit::track(lists[0].Get(), pending(a));
        check(SUCCEEDED(lists[0]->Close()) && SUCCEEDED(lists[0]->Reset(alloc[0].Get(), nullptr)), "discard list");
        check(a.discarded == 1 && a.submitted == 1 && !d4r_submit::pending(&a), "discard callback");
        d4r_submit::track(lists[0].Get(), pending(a));
        d4r_submit::track(lists[1].Get(), pending(b));
        check(SUCCEEDED(lists[0]->Close()) && SUCCEEDED(lists[1]->Close()), "close batch");
        queue->ExecuteCommandLists(2, batch); wait(a); wait(b);
        check(a.submitted == 2 && b.submitted == 1 && a.value == b.value && a.frequency == b.frequency, "batch callbacks");
        uint64_t expected = 0; queue->GetTimestampFrequency(&expected);
        check(a.frequency == expected, "timestamp frequency was not from the submitting queue");
        check(SUCCEEDED(alloc[0]->Reset()) && SUCCEEDED(lists[0]->Reset(alloc[0].Get(), nullptr)), "reset batch");
        d4r_submit::track(lists[0].Get(), pending(a));
        check(d4r_submit::forget(&a) == 1 && !d4r_submit::pending(&a), "forget pending owner");
        lists[0]->Close();
        check(SUCCEEDED(device->GetDeviceRemovedReason()), "device removed");
        std::puts("delayed submission, discard, batched lists, timestamp frequency and owner retirement passed");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
