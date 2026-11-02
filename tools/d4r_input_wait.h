#pragma once
#include <cstdint>

// The original marker remains the authority for whether a command list ran. A CPU release of the
// separate, coherent GPU wait word must never make discarded input look ready to the worker.
enum class D4rInputWaitAbort { None, Retired, Discarded, Timeout };

inline D4rInputWaitAbort d4r_input_wait_abort(uint32_t frame, uint32_t slotMarker, uint32_t aggregate,
                                            bool retiring, double elapsedUs)
{
    if (retiring)
        return D4rInputWaitAbort::Retired;
    if (slotMarker != frame && static_cast<int32_t>(aggregate - frame) > 0)
        return D4rInputWaitAbort::Discarded;
    // Also covers a stream wait that failed to observe an input word already written by Vulkan.
    if (elapsedUs > 5000000.0)
        return D4rInputWaitAbort::Timeout;
    return D4rInputWaitAbort::None;
}

inline unsigned d4r_marker_poll_delay_us(double predictedUs, double elapsedUs)
{
    const double remaining = predictedUs - 150.0 - elapsedUs;
    if (remaining > 20.0)
        return static_cast<unsigned>(remaining < 200.0 ? remaining : 200.0);
    if (elapsedUs < predictedUs + 150.0)
        return 0; // yield during the final 150 us and a short late window
    const double late = 20.0 * (1.0 + (elapsedUs - predictedUs - 150.0) / 160.0);
    return static_cast<unsigned>(late < 200.0 ? late : 200.0);
}
