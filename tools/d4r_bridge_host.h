/* The CUDA bridge's host side. The Linux build (winegcc) is a Wine DLL whose code runs on the Unix side and calls
   ZLUDA's libcuda.so: System V calls, dlopen and the host's library paths. The native Windows build (MinGW,
   D4R_NATIVE_WINDOWS) is an ordinary nvcuda.dll that forwards to ZLUDA's Windows nvcuda.dll in a subdirectory:
   Windows calls, LoadLibrary, and HIP from HIP_PATH (ZLUDA loads it). Everything else in the bridge is shared. */
#pragma once
#ifdef D4R_NATIVE_WINDOWS
#include <windows.h>
#include <direct.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D4R_HOST_ABI
#define RTLD_NOW 0
#define RTLD_LOCAL 0
#define RTLD_NOLOAD 0x10000
/* MinGW's mkdir takes no mode */
#define mkdir(path, mode) _mkdir(path)

static char d4r_dl_error[512];
static int d4r_dl_error_set;

static void* dlopen(const char* path, int flags)
{
    const int qualified = strchr(path, '\\') != NULL || strchr(path, '/') != NULL;
    HMODULE module = (flags & RTLD_NOLOAD) ? GetModuleHandleA(path)
                                           : LoadLibraryExA(path, NULL, qualified ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
    if (module == NULL)
    {
        snprintf(d4r_dl_error, sizeof(d4r_dl_error), "%s: Windows error %lu", path, GetLastError());
        d4r_dl_error_set = 1;
    }
    return module;
}

static void* dlsym(void* library, const char* name)
{
    FARPROC symbol = GetProcAddress((HMODULE)library, name);
    void* result;
    memcpy(&result, &symbol, sizeof(result));
    return result;
}

static const char* dlerror(void)
{
    if (!d4r_dl_error_set)
        return NULL;
    d4r_dl_error_set = 0;
    return d4r_dl_error;
}

/* _putenv_s updates the C runtime's copy (getenv) and the process environment ZLUDA reads */
static int setenv(const char* name, const char* value, int overwrite)
{
    if (!overwrite && getenv(name) != NULL)
        return 0;
    return _putenv_s(name, value) == 0 ? 0 : -1;
}

static int unsetenv(const char* name) { return _putenv_s(name, "") == 0 ? 0 : -1; }

/* kill(pid, 0): whether a process exists (the native-kernel cache removes directories of exited ones) */
static int kill(int pid, int signal)
{
    (void)signal;
    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (process == NULL)
    {
        errno = GetLastError() == ERROR_INVALID_PARAMETER ? ESRCH : EPERM;
        return -1;
    }
    DWORD code = 0;
    const BOOL running = GetExitCodeProcess(process, &code) && code == STILL_ACTIVE;
    CloseHandle(process);
    if (!running)
    {
        errno = ESRCH;
        return -1;
    }
    return 0;
}

static char* realpath(const char* path, char* resolved) { return _fullpath(resolved, path, MAX_PATH); }

/* the native kernels are served as copies: links need privileges on Windows */
static int symlink(const char* target, const char* path)
{
    if (CopyFileA(target, path, TRUE))
        return 0;
    errno = GetLastError() == ERROR_FILE_EXISTS ? EEXIST : EIO;
    return -1;
}

/* ZLUDA's nvcuda.dll beside this one, in zluda\ (D4R_ZLUDA_LIBCUDA overrides it) */
static const char* d4r_default_zluda(void)
{
    static char path[MAX_PATH];
    HMODULE self = NULL;
    if (path[0] == '\0' &&
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)(void*)d4r_default_zluda, &self))
    {
        const DWORD length = GetModuleFileNameA(self, path, MAX_PATH);
        char* slash = length > 0 && length < MAX_PATH ? strrchr(path, '\\') : NULL;
        if (slash == NULL || (size_t)(slash - path) + sizeof("\\zluda\\nvcuda.dll") > MAX_PATH)
            path[0] = '\0';
        else
            strcpy(slash, "\\zluda\\nvcuda.dll");
    }
    return path[0] != '\0' ? path : "nvcuda_zluda.dll";
}

/* bytes readable from address to the end of its committed region */
static size_t d4r_readable_span(const void* address)
{
    MEMORY_BASIC_INFORMATION info;
    if (VirtualQuery(address, &info, sizeof(info)) == 0 || info.State != MEM_COMMIT ||
        (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0)
        return 0;
    return (size_t)((const char*)info.BaseAddress + info.RegionSize - (const char*)address);
}

/* where CUDA modules are saved without D4R_CUDA_CAPTURE_DIR: %TEMP%\d4r-dlss-cuda-modules */
static const char* d4r_default_capture_dir(void)
{
    static char path[MAX_PATH];
    if (path[0] == '\0')
    {
        const DWORD length = GetTempPathA(MAX_PATH, path);
        if (length == 0 || length + sizeof("d4r-dlss-cuda-modules") > MAX_PATH)
            strcpy(path, "d4r-dlss-cuda-modules");
        else
            strcat(path, "d4r-dlss-cuda-modules");
    }
    return path;
}

#define D4R_HIP_LIBRARY "amdhip64_7.dll"
#define D4R_HIP_LIBRARY_FALLBACK "amdhip64.dll"
#else
#include <dlfcn.h>
#define D4R_HOST_ABI __attribute__((sysv_abi))
static const char* d4r_default_capture_dir(void) { return "/tmp/d4r-dlss-cuda-modules"; }
#define D4R_HIP_LIBRARY "libamdhip64.so.7"
#define D4R_HIP_LIBRARY_FALLBACK "libamdhip64.so"
#endif
