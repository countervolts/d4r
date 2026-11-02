/* dlopen/dlsym for the diagnostic tools (C and C++) on Linux and native Windows.
   On Windows the Linux library names the tools use map to their Windows counterparts:
   libcuda.so -> nvcuda.dll (ZLUDA's, found through PATH or next to the executable),
   libamdhip64.so* -> amdhip64_7.dll (the AMD driver's or the HIP SDK's). */
#pragma once
#ifndef _WIN32
#include <dlfcn.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <stdio.h>
#include <string.h>

#define RTLD_LAZY 0x1
#define RTLD_NOW 0x2
#define RTLD_LOCAL 0
#define RTLD_GLOBAL 0x100
#define RTLD_NOLOAD 0x4
#define RTLD_DEFAULT NULL

static char d4r_dl_error_message[512];

static inline void d4r_dl_set_error(const char* what, const char* name)
{
    const DWORD code = GetLastError();
    char system[384] = {0};
    size_t n;
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, code, 0, system,
                   sizeof(system), NULL);
    for (n = strlen(system); n > 0 && (system[n - 1] == '\n' || system[n - 1] == '\r'); --n)
        system[n - 1] = 0;
    snprintf(d4r_dl_error_message, sizeof(d4r_dl_error_message), "%s(%s): error %lu: %s", what, name ? name : "",
             (unsigned long)code, system);
}

static inline const char* d4r_dl_windows_name(const char* name)
{
    const char* base = name;
    const char* p;
    for (p = name; *p; ++p)
        if (*p == '/' || *p == '\\')
            base = p + 1;
    if (base != name)
        return name; /* an explicit path is used as given */
    if (strcmp(name, "libcuda.so") == 0 || strncmp(name, "libcuda.so.", 11) == 0 || strcmp(name, "libnvcuda.so") == 0)
        return "nvcuda.dll";
    if (strncmp(name, "libamdhip64.so", 14) == 0)
        return "amdhip64_7.dll";
    return name;
}

static inline void* dlopen(const char* name, int flags)
{
    const char* file;
    HMODULE module;
    d4r_dl_error_message[0] = 0;
    if (name == NULL)
        return (void*)GetModuleHandleA(NULL);
    file = d4r_dl_windows_name(name);
    module = (flags & RTLD_NOLOAD) ? GetModuleHandleA(file) : LoadLibraryA(file);
    if (module == NULL)
        d4r_dl_set_error("LoadLibrary", file);
    return (void*)module;
}

/* RTLD_DEFAULT searches the executable and then the libraries the tools load by name. */
static inline void* dlsym(void* library, const char* symbol)
{
    static const char* const searched[] = {NULL, "amdhip64_7.dll", "nvcuda.dll"};
    FARPROC address = NULL;
    void* result;
    size_t i;
    d4r_dl_error_message[0] = 0;
    if (library != NULL)
        address = GetProcAddress((HMODULE)library, symbol);
    else
        for (i = 0; i < sizeof(searched) / sizeof(searched[0]) && address == NULL; ++i)
        {
            HMODULE module = GetModuleHandleA(searched[i]);
            if (module != NULL)
                address = GetProcAddress(module, symbol);
        }
    if (address == NULL)
    {
        d4r_dl_set_error("GetProcAddress", symbol);
        return NULL;
    }
    memcpy(&result, &address, sizeof(result)); /* function and data pointers are both 64 bits here */
    return result;
}

static inline int dlclose(void* library)
{
    return library != NULL && library != (void*)GetModuleHandleA(NULL) && !FreeLibrary((HMODULE)library);
}

static inline const char* dlerror(void)
{
    return d4r_dl_error_message[0] ? d4r_dl_error_message : NULL;
}
#endif
