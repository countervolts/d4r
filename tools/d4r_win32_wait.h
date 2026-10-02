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
    if (microseconds == 0)
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
