#pragma once
#include "d3d12_external.h"
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace d4r::win {
// GPU-completion ownership, independent of command-list Reset and queue COM
// lifetime. The command backend pins its DLL. One process-lifetime thread
// releases completed batches outside its lock, including the final batch;
// queue -> recording -> feature -> queue cannot become a reference cycle.
class GpuRetirement {
    struct Batch { ComPtr<ID3D12Fence> fence; uint64_t value; std::shared_ptr<void> owner; };
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Batch> pending_;
    GpuRetirement() {
        std::thread([this] {
            Handle event; event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            for (;;) {
                Batch batch;
                { std::unique_lock<std::mutex> lock(mutex_);
                  changed_.wait(lock, [&] { return !pending_.empty(); });
                  batch = std::move(pending_.front()); pending_.pop_front(); }
                if (batch.fence->GetCompletedValue() < batch.value) {
                    if (!event.value || FAILED(batch.fence->SetEventOnCompletion(batch.value, event.value)) ||
                        WaitForSingleObject(event.value, 30000) != WAIT_OBJECT_0) {
                        std::fprintf(stderr, "D4R_RETIREMENT_FAILURE GPU completion timeout; retaining resources\n");
                        // A hung GPU must not make us free live command memory.
                        auto* retained = new Batch(std::move(batch)); (void)retained;
                    }
                }
                batch.owner.reset();
            }
        }).detach();
    }
public:
    static GpuRetirement& instance() { static auto* service = new GpuRetirement; return *service; }
    void submit(ID3D12Fence* fence, uint64_t value, std::shared_ptr<void> owner) {
        { std::lock_guard<std::mutex> lock(mutex_); pending_.push_back({fence, value, std::move(owner)}); }
        changed_.notify_one();
    }
};
}
