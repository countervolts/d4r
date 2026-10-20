// Opt-in latency NVAPI for OptiScaler. Never replace NGX's topology/driver APIs
// with fakenvapi's synthetic GPU handles. This DLL is selected by NvapiDllPath,
// not installed as nvapi64.dll in the prefix or game directory.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>

using Query = void* (__cdecl*)(unsigned int);
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
static Query realQuery, latencyQuery;

static HMODULE load_direct(const std::wstring& path)
{
    // Bypass OptiScaler's LoadLibrary NVAPI redirection, avoiding recursion.
    struct UnicodeString { USHORT length, capacity; PWSTR data; };
    using Load = LONG (NTAPI*)(PWSTR, PULONG, UnicodeString*, PHANDLE);
    auto load = reinterpret_cast<Load>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "LdrLoadDll")));
    if (!load || path.size() > 32766) return nullptr;
    UnicodeString name{static_cast<USHORT>(path.size() * sizeof(wchar_t)),
                       static_cast<USHORT>(path.size() * sizeof(wchar_t)), const_cast<PWSTR>(path.c_str())};
    HANDLE module = nullptr;
    return load(nullptr, nullptr, &name, &module) >= 0 ? static_cast<HMODULE>(module) : nullptr;
}

static Query query_from(const std::wstring& path)
{
    HMODULE module = load_direct(path);
    return module ? reinterpret_cast<Query>(reinterpret_cast<void*>(GetProcAddress(module, "nvapi_QueryInterface"))) : nullptr;
}

static int __cdecl initialize_nvapi()
{
    using Fn = int (__cdecl*)();
    auto real = realQuery ? reinterpret_cast<Fn>(realQuery(0x0150e828)) : nullptr;
    if (!real) return -3; // NVAPI_NO_IMPLEMENTATION
    const int status = real();
    if (status == 0 && latencyQuery) {
        auto latency = reinterpret_cast<Fn>(latencyQuery(0x0150e828));
        if (latency) latency();
    }
    return status;
}

static int __cdecl unload_nvapi()
{
    using Fn = int (__cdecl*)();
    auto real = realQuery ? reinterpret_cast<Fn>(realQuery(0xd22bdd7e)) : nullptr;
    if (!real) return -3;
    if (latencyQuery) {
        auto latency = reinterpret_cast<Fn>(latencyQuery(0xd22bdd7e));
        if (latency) latency();
    }
    return real();
}

static BOOL CALLBACK initialize(PINIT_ONCE, PVOID, PVOID*)
{
    wchar_t system[MAX_PATH] = {};
    if (GetSystemDirectoryW(system, MAX_PATH)) {
        // Developer Proton launchers keep the original under this name.
        std::wstring original = std::wstring(system) + L"\\d4r_nvapi64_real.dll";
        if (GetFileAttributesW(original.c_str()) != INVALID_FILE_ATTRIBUTES)
            realQuery = query_from(original);
        if (!realQuery)
            realQuery = query_from(std::wstring(system) + L"\\nvapi64.dll");
    }
    HMODULE self = nullptr;
    wchar_t path[32768] = {};
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&once), &self)) {
        DWORD length = GetModuleFileNameW(self, path, 32768);
        if (length && length < 32768) {
            std::wstring directory(path, length);
            const auto slash = directory.find_last_of(L"\\/");
            if (slash != std::wstring::npos) {
                directory.resize(slash + 1);
                // Fakenvapi loads XeLL by basename. Preload our colocated DLL;
                // never change the process-wide DLL search path.
                load_direct(directory + L"libxell.dll");
                latencyQuery = query_from(directory + L"fakenvapi.dll");
            }
        }
    }
    OutputDebugStringA(latencyQuery ? "d4r NVAPI: latency -> fakenvapi; other APIs -> Proton\n"
                                   : "d4r NVAPI: fakenvapi missing; using Proton\n");
    return TRUE;
}

static bool latency_interface(unsigned int id)
{
    switch (id) {
    case 0xaef96ca1: // NvAPI_D3D_GetSleepStatus
    case 0xac1ca9e0: // NvAPI_D3D_SetSleepMode
    case 0x852cd1d2: // NvAPI_D3D_Sleep
    case 0x1a587f9c: // NvAPI_D3D_GetLatency
    case 0xd9984c05: // NvAPI_D3D_SetLatencyMarker
    case 0x13c98f73: // NvAPI_D3D12_SetAsyncFrameMarker
    case 0x21372137: // Fake_GetLatency (legacy)
    case 0x21382138: // Fake_InformFGState
    case 0x21392139: // Fake_InformPresentFG
    case 0x21402140: // Fake_GetAntiLagCtx
    case 0x21412141: // Fake_GetLowLatencyCtx
    case 0x21422142: // Fake_SetLowLatencyCtx
        return true;
    default:
        return false;
    }
}

extern "C" __declspec(dllexport) void* __cdecl nvapi_QueryInterface(unsigned int id)
{
    InitOnceExecuteOnce(&once, initialize, nullptr, nullptr);
    if (id == 0x0150e828 && realQuery && realQuery(id)) return reinterpret_cast<void*>(&initialize_nvapi);
    if (id == 0xd22bdd7e && realQuery && realQuery(id)) return reinterpret_cast<void*>(&unload_nvapi);
    if (latency_interface(id) && latencyQuery) {
        if (void* result = latencyQuery(id)) return result;
    }
    return realQuery ? realQuery(id) : nullptr;
}
