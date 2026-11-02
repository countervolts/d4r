/* LD_PRELOAD for zluda_precompile.py: report D4R_SPOOF_GFX (e.g. gfx1201) as the device's gcnArchName, so
 * ZLUDA compiles (and caches) code objects for a GPU that is not installed. Loading them then fails, which
 * the precompile treats as expected; the cache entry is written before the load. Never use it for running. */
#define _GNU_SOURCE
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* props, int device)
{
    static hipError_t (*real)(hipDeviceProp_tR0600*, int);
    if (!real)
        real = (hipError_t (*)(hipDeviceProp_tR0600*, int))dlsym(RTLD_NEXT, "hipGetDevicePropertiesR0600");
    if (!real)
    {
        /* ZLUDA loaded RTLD_LOCAL (ctypes): its HIP is not in the global scope RTLD_NEXT searches */
        void* hip = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_NOLOAD);
        if (!hip)
            hip = dlopen("libamdhip64.so", RTLD_NOW | RTLD_NOLOAD);
        if (hip)
            real = (hipError_t (*)(hipDeviceProp_tR0600*, int))dlsym(hip, "hipGetDevicePropertiesR0600");
    }
    if (!real)
        return hipErrorNotInitialized;
    hipError_t result = real(props, device);
    const char* arch = getenv("D4R_SPOOF_GFX");
    if (result == hipSuccess && props && arch && *arch)
    {
        memset(props->gcnArchName, 0, sizeof(props->gcnArchName));
        strncpy(props->gcnArchName, arch, sizeof(props->gcnArchName) - 1);
    }
    return result;
}
