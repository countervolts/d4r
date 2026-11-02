// CPU-only smoke probe for the ASI loader contract. No NGX/CUDA/GPU initialization.
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <string>

int wmain(int argc, wchar_t** argv)
{
    if (argc != 3 && argc != 4)
        return 2; // ASI absolute path, expected runtime directory
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE optiscaler = nullptr;
    if (argc == 4)
    {
        optiscaler = LoadLibraryW(argv[3]);
        if (!optiscaler || !GetModuleHandleW(argv[1]) ||
            GetEnvironmentVariableW(L"D4R_NVCUDA_BRIDGE", nullptr, 0) == 0)
        {
            std::fprintf(stderr, "OptiScaler did not initialize the ASI: %lu\n", GetLastError());
            return 8;
        }
    }
    HMODULE plugin = LoadLibraryW(argv[1]);
    if (!plugin)
    {
        std::fprintf(stderr, "LoadLibrary failed: %lu\n", GetLastError());
        return 3;
    }
    for (const char* name : {"InitializeASI", "NVSDK_NGX_D3D12_Init", "NVSDK_NGX_VULKAN_Init",
                             "NVSDK_NGX_D3D12_CreateFeature", "NVSDK_NGX_VULKAN_CreateFeature"})
        if (!GetProcAddress(plugin, name))
        {
            std::fprintf(stderr, "Missing export: %s\n", name);
            return 4;
        }
    auto initialize = reinterpret_cast<void (*)()>(GetProcAddress(plugin, "InitializeASI"));
    initialize();
    initialize(); // ASI initialization is idempotent.
    struct Expected { const wchar_t* name; const wchar_t* suffix; };
    for (const auto& expected : {Expected{L"D4R_NGX_FEATURE_DIR", L""},
                                Expected{L"D4R_NVCUDA_BRIDGE", L"\\nvcuda.dll"},
                                Expected{L"D4R_NGX_CORE", L"\\ngx\\_nvngx.dll"},
                                Expected{L"D4R_SHIM_LOG", L"\\d4r_nvngx.log"}})
    {
        wchar_t value[MAX_PATH] = {};
        GetEnvironmentVariableW(expected.name, value, MAX_PATH);
        if (std::wstring(value) != std::wstring(argv[2]) + expected.suffix)
        {
            std::fwprintf(stderr, L"Wrong %ls: %ls\n", expected.name, value);
            return 5;
        }
    }
    HMODULE backend = LoadLibraryW(argv[1]);
    if (backend != plugin)
        return 6;
    if (GetModuleHandleA("nvcuda.dll") || GetModuleHandleA("_nvngx.dll"))
        return 7; // The ASI loader must not trigger GPU initialization.
    char delay[16] = {};
    if (GetEnvironmentVariableA("D4R_ASI_PROBE_PAUSE_MS", delay, sizeof(delay)))
        Sleep(static_cast<DWORD>(std::strtoul(delay, nullptr, 10)));
    FreeLibrary(backend);
    FreeLibrary(plugin);
    if (optiscaler) FreeLibrary(optiscaler);
    std::puts("ASI loader contract passed; backend reused, runtime located, GPU initialization deferred");
    return 0;
}
