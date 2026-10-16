// Calls the drag-in loader the way OptiScaler does: the GPU-free discovery
// marker, then NGX feature requirements for every DXGI adapter.
#include <windows.h>
#include <dxgi.h>
#include <cstdio>

struct Info { unsigned SDKVersion, FeatureID; char reserved[48]; };
struct Requirement { unsigned FeatureSupported, MinHWArchitecture; char MinOSVersion[255]; };
using Requirements = unsigned (*)(IDXGIAdapter*, const Info*, Requirement*);
using Version = unsigned (*)();

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) { std::printf("usage: loader_selection_probe LOADER_DLL\n"); return 2; }
    HMODULE loader = LoadLibraryW(argv[1]);
    if (!loader) { std::printf("LOAD_FAILED error=%lu\n", GetLastError()); return 2; }
    auto version = reinterpret_cast<Version>(reinterpret_cast<void*>(GetProcAddress(loader, "d4r_WindowsBackendVersion")));
    auto requirements = reinterpret_cast<Requirements>(reinterpret_cast<void*>(GetProcAddress(loader, "NVSDK_NGX_D3D12_GetFeatureRequirements")));
    if (!version || !requirements) { std::printf("EXPORTS_MISSING\n"); return 2; }
    std::printf("MARKER version=%u hip_loaded=%d\n", version(), GetModuleHandleW(L"amdhip64_7.dll") != nullptr);
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return 2;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
        DXGI_ADAPTER_DESC1 desc{}; adapter->GetDesc1(&desc);
        Info info{0x15, 1, {}}; Requirement result{};
        const unsigned status = requirements(adapter, &info, &result);
        std::printf("ADAPTER vendor=0x%04x status=0x%x supported=%u marker=0x%x name=%ls\n", desc.VendorId, status,
            result.FeatureSupported, result.MinHWArchitecture, desc.Description);
        adapter->Release();
    }
    factory->Release();
    return 0;
}
