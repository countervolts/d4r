// Force-included into ZLUDA's PTX helpers when they are built with the Windows HIP SDK
// (scripts/build_zluda_ptx_helpers.sh). Windows HIP declares only _wassert for device code, while ZLUDA's
// __assertfail calls glibc's __assert_fail. Like HIP's Windows assertion, it traps; only a failing
// device-side assertion behaves differently from the Linux build (no message).
extern "C" __attribute__((device)) __attribute__((noinline)) __attribute__((weak)) void
__assert_fail(const char*, const char*, unsigned int, const char*)
{
    __builtin_trap();
}
