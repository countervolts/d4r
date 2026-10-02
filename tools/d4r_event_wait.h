#pragma once

// Query completion rather than trusting HIP event flags to put the CPU to
// sleep. Never treat NOT_READY as completion, including during retirement.
template <typename Query, typename Sleep>
int d4r_wait_event(Query query, Sleep sleep)
{
    constexpr int cudaNotReady = 600;
    for (;;)
    {
        const int result = query();
        if (result != cudaNotReady)
            return result;
        sleep();
    }
}
