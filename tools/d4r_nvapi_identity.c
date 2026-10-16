// Opt-in NVAPI identity bridge for OptiScaler's NVIDIA-only DLSS backend.
// All NVAPI interfaces except the version string are forwarded to the
// prefix-local DXVK-NVAPI DLL. The underlying GPU and DLSS execution remain
// AMD/ZLUDA; this only lets OptiScaler select its official-DLSS route.
#include <windows.h>
#include <string.h>

#define NVAPI_GET_INTERFACE_VERSION_STRING 0x01053fa5u
#define NVAPI_INITIALIZE 0x0150e828u
#define NVAPI_UNLOAD 0xd22bdd7eu

typedef void *(__cdecl *NvapiQueryInterface)(unsigned int);
static NvapiQueryInterface real_query;
static volatile LONG spoof_probe_armed;
static volatile LONG spoof_probe_used;
static volatile LONG spoof_unload_armed;

static NvapiQueryInterface load_real_query(void)
{
    if (real_query != NULL)
        return real_query;

    wchar_t path[MAX_PATH];
    UINT length = GetSystemDirectoryW(path, MAX_PATH);
    static const wchar_t suffix[] = L"\\d4r_nvapi64_real.dll";
    if (length == 0 || length + (sizeof(suffix) / sizeof(suffix[0])) > MAX_PATH)
        return NULL;
    memcpy(path + length, suffix, sizeof(suffix));

    HMODULE module = LoadLibraryW(path);
    if (module == NULL)
        return NULL;
    FARPROC symbol = GetProcAddress(module, "nvapi_QueryInterface");
    memcpy(&real_query, &symbol, sizeof(real_query));
    return real_query;
}

static int __cdecl d4r_get_interface_version_string(char *description)
{
    if (description == NULL)
        return -5; // NVAPI_INVALID_ARGUMENT
    // OptiScaler takes the real-NVIDIA branch for any non-DXVK version string.
    // NVAPI's short string has 64 bytes.
    static const char identity[] = "NVIDIA NVAPI (d4r compatibility)";
    memcpy(description, identity, sizeof(identity));
    if (InterlockedCompareExchange(&spoof_probe_used, 0, 0) == 0)
        InterlockedExchange(&spoof_probe_armed, 1);
    return 0; // NVAPI_OK
}

static int __cdecl d4r_initialize_identity_probe(void)
{
    // OptiScaler calls this inside DLL_PROCESS_ATTACH when it believes it is
    // loading NVIDIA's own NVAPI. DXVK-NVAPI initializes DXGI here, which is
    // unsafe under the loader lock. The later, ordinary init calls are passed
    // through to DXVK-NVAPI.
    InterlockedExchange(&spoof_probe_used, 1);
    InterlockedExchange(&spoof_unload_armed, 1);
    return 0;
}

static int __cdecl d4r_unload_identity_probe(void)
{
    return 0;
}

__declspec(dllexport) void *__cdecl nvapi_QueryInterface(unsigned int interface_id)
{
    NvapiQueryInterface query = load_real_query();
    if (query == NULL)
        return NULL;
    if (interface_id == NVAPI_GET_INTERFACE_VERSION_STRING)
        return (void *)&d4r_get_interface_version_string;
    if (interface_id == NVAPI_INITIALIZE &&
        InterlockedCompareExchange(&spoof_probe_armed, 0, 1) == 1)
        return (void *)&d4r_initialize_identity_probe;
    if (interface_id == NVAPI_UNLOAD &&
        InterlockedCompareExchange(&spoof_unload_armed, 0, 1) == 1)
        return (void *)&d4r_unload_identity_probe;
    return query(interface_id);
}
