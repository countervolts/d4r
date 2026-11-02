#pragma once
#include <cstdint>
#include <deque>

// A timeline signal releases every lower value too. Prep can cancel a later
// frame while the CUDA worker is still producing an earlier one, so only the
// completed prefix of admitted frames may be signalled. The caller serializes
// admission and retirement with the same mutex used for the timeline signal.
class D4rFrameCompletion
{
public:
    void admit(uint32_t frame) { pending_.push_back({frame, false}); }

    // A native completion callback can release this value without crossing an unfinished producer
    // only when it is the first admitted frame still outstanding. Retirement itself stays on the
    // worker, which also folds any later discarded frames into the completed prefix.
    bool can_signal(uint32_t frame) const { return !pending_.empty() && pending_.front().frame == frame; }

    uint64_t retire(uint32_t frame)
    {
        for (auto& pending : pending_)
            if (pending.frame == frame)
            {
                pending.done = true;
                break;
            }
        while (!pending_.empty() && pending_.front().done)
        {
            completedThrough_ = pending_.front().frame;
            pending_.pop_front();
        }
        return completedThrough_;
    }

private:
    struct Pending { uint32_t frame; bool done; };
    std::deque<Pending> pending_;
    uint64_t completedThrough_ = 0;
};
