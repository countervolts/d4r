#include <windows.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: ngx_load_probe.exe PATH_TO_NVNGX_DLSS_DLL\n");
        return 2;
    }

    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);

    const char* path = argv[1];
    HMODULE module = LoadLibraryA(path);
    if (module == nullptr)
    {
        std::fprintf(stderr, "LoadLibraryA failed for '%s': GetLastError=%lu\n", path, GetLastError());
        return 1;
    }

    std::printf("LoadLibraryA succeeded: %s\n", path);

    constexpr const char* symbols[] = {
        "NVSDK_NGX_GetAPIVersion",
        "NVSDK_NGX_D3D12_GetFeatureRequirements",
        "NVSDK_NGX_D3D12_Init",
        "NVSDK_NGX_D3D12_CreateFeature",
        "NVSDK_NGX_D3D12_EvaluateFeature",
        "NVSDK_NGX_VULKAN_GetFeatureRequirements",
        "NVSDK_NGX_VULKAN_Init",
        "NVSDK_NGX_VULKAN_CreateFeature",
        "NVSDK_NGX_VULKAN_EvaluateFeature",
        "NVSDK_NGX_CUDA_GetFeatureRequirements",
        "NVSDK_NGX_CUDA_Init",
        "NVSDK_NGX_CUDA_CreateFeature",
        "NVSDK_NGX_CUDA_EvaluateFeature",
    };

    for (const char* symbol : symbols)
    {
        std::printf("export %-52s %s\n", symbol, GetProcAddress(module, symbol) ? "present" : "absent");
    }

    FARPROC getApiVersion = GetProcAddress(module, "NVSDK_NGX_GetAPIVersion");
    if (getApiVersion == nullptr)
    {
        std::fprintf(stderr, "NVSDK_NGX_GetAPIVersion export is unavailable\n");
        FreeLibrary(module);
        return 1;
    }

    const unsigned int apiVersion = static_cast<unsigned int>(getApiVersion());
    std::printf("NVSDK_NGX_GetAPIVersion returned 0x%08x\n", apiVersion);

    FreeLibrary(module);
    return 0;
}
