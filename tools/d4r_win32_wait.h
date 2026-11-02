#pragma once
#include <windows.h>
#include <cstdint>

// MinGW's Win32 std::sleep_for can truncate submillisecond durations to zero.
// NtDelayExecution retains 100 ns precision and actually sleeps under Wine.
inline void d4r_sleep_us(unsigned int microseconds)
{
    using Delay = LONG(WINAPI*)(BOOLEAN, LARGE_INTEGER*);
    static const auto delay = reinterpret_cast<Delay>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtDelayExecution")));
    // D4R_SHIM_SPIN_POLL=1 (measurement only): every poll yields instead of sleeping, so frame timings are
    // not quantised to the ~250 us a 200 us sleep takes. Costs a CPU core per waiting thread.
    static const bool spin = GetEnvironmentVariableA("D4R_SHIM_SPIN_POLL", nullptr, 0) > 1;
    if (microseconds == 0 || spin)
    {
        SwitchToThread();
        return;
    }
    LARGE_INTEGER interval;
    interval.QuadPart = -static_cast<LONGLONG>(microseconds) * 10;
    if (delay != nullptr && delay(FALSE, &interval) >= 0)
        return;
    // A rounded-up millisecond wait still releases the CPU if ntdll is absent.
    Sleep(static_cast<DWORD>((static_cast<uint64_t>(microseconds) + 999) / 1000));
}
