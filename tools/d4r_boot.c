/* d4r_boot.dll: the NVIDIA identity that the Steam launch options PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100
 * used to provide, set up inside the game process instead.
 *
 * The release patches OptiScaler (the game folder's dxgi.dll) to import this DLL first (scripts/pe_add_import.py),
 * and d4r's d3d12core.dll imports dxgi.dll, so DllMain below runs before OptiScaler's DllMain and before anything
 * creates a DXGI factory, a D3D12 device or calls NvAPI_Initialize.
 *
 * What PROTON_FORCE_NVAPI=1 does, and what replaces it here:
 * - DXVK_NVAPI_ALLOW_OTHER_DRIVERS=1, DXVK_NVAPI_DRIVER_VERSION=99999 (and DXVK_NVAPI_GPU_ARCH from the launch
 *   options): dxvk-nvapi reads them from the Windows environment when the game initializes NVAPI, so they are set
 *   here (unless already set).
 * - WINE_HIDE_AMD_GPU=1: Proton's win32u then reports AMD GPUs as PCI 10de:2487 in EnumDisplayDevices and in
 *   vkGetPhysicalDeviceProperties(2) (driver id and name unchanged). win32u reads it from the Unix environment at
 *   startup, which a running process cannot change, so the same rewrite is applied to those entry points here:
 *   the export tables of user32, vulkan-1 and winevulkan point at small stubs (allocated above each module, as
 *   export RVAs must be), and already-bound import tables of loaded modules (OptiScaler's) are redirected the same
 *   way. vkGetInstanceProcAddr is wrapped so DXVK, vkd3d-proton and dxvk-nvapi receive the wrapped property
 *   queries. No code of Wine's modules is modified. WINE_HIDE_AMD_GPU=1 is also put into the Windows environment,
 *   so processes the game starts get Proton's own rewrite.
 *
 * Nothing is done when an NVIDIA GPU is already visible (a real one, or the launch options are still set), apart
 * from defaulting DXVK_NVAPI_GPU_ARCH when WINE_HIDE_AMD_GPU says the visible one is a hidden AMD GPU.
 * d4r\d4r.ini [Identity] SpoofGpu = false (env D4R_SPOOF_GPU=0) turns this off; GpuArch selects the architecture.
 *
 * DLSS frame generation (fg/dlssg, d4r.ini [FrameGeneration] Backend = dlssg, set by d4r-features.py --fg dlssg)
 * needs its Vulkan layer in the game's Unix environment before the first VkInstance: VK_ADD_LAYER_PATH,
 * VK_INSTANCE_LAYERS and the layer's NVXRT_* options are set through the Wine CUDA bridge's d4rSetEnv (the only
 * Unix-side code in the folder), which a launch wrapper did before. The host loads d4r's shim as
 * d4r\d4r_dlssg_next.dll (OptiScaler answers loads of names ending in nvngx.dll itself); that copy is refreshed
 * here from d4r\nvngx.dll, before OptiScaler's file hooks exist.
 * The outcome is written to d4r\d4r_boot.log. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#define AMD_VENDOR 0x1002u
#define NV_VENDOR 0x10deu
#define NV_DEVICE 0x2487u /* what Proton's WINE_HIDE_AMD_GPU reports */

static wchar_t g_dir[MAX_PATH]; /* folder of this DLL (the game executable's folder) */
static FILE *g_log;

static void logf_(const char *fmt, ...)
{
    if (!g_log)
        return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_log);
    fflush(g_log);
}

/* ---------------------------------------------------------------- configuration */

static int env_set(const wchar_t *name)
{
    return GetEnvironmentVariableW(name, NULL, 0) != 0;
}

/* 1 on, 0 off, -1 auto */
static int ini_bool(const wchar_t *value)
{
    if (!_wcsicmp(value, L"true") || !_wcsicmp(value, L"1") || !_wcsicmp(value, L"on"))
        return 1;
    if (!_wcsicmp(value, L"false") || !_wcsicmp(value, L"0") || !_wcsicmp(value, L"off"))
        return 0;
    return -1;
}

static void ini_read(const wchar_t *key, wchar_t *out, DWORD size)
{
    wchar_t path[MAX_PATH + 32];
    _snwprintf(path, MAX_PATH + 32, L"%ls\\d4r\\d4r.ini", g_dir);
    path[MAX_PATH + 31] = 0;
    GetPrivateProfileStringW(L"Identity", key, L"auto", out, size, path);
}

/* sets NAME=VALUE in the Windows environment unless it is set, and in the C runtimes' copies of it for getenv()
 * readers (dxvk-nvapi links msvcrt, this DLL ucrt) */
static void env_default(const wchar_t *name, const wchar_t *value)
{
    if (env_set(name))
        return;
    SetEnvironmentVariableW(name, value);
    wchar_t both[128];
    _snwprintf(both, 128, L"%ls=%ls", name, value);
    both[127] = 0;
    _wputenv(both);
    HMODULE msvcrt = LoadLibraryW(L"msvcrt.dll");
    int(__cdecl * msvcrt_wputenv)(const wchar_t *) =
        msvcrt ? (int(__cdecl *)(const wchar_t *))(void *)GetProcAddress(msvcrt, "_wputenv") : NULL;
    if (msvcrt_wputenv)
        msvcrt_wputenv(both);
    logf_("set %ls=%ls", name, value);
}

/* ---------------------------------------------------------------- PE helpers */

static IMAGE_NT_HEADERS64 *nt_headers(HMODULE module)
{
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)module;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return NULL;
    IMAGE_NT_HEADERS64 *nt = (IMAGE_NT_HEADERS64 *)((BYTE *)module + dos->e_lfanew);
    return nt->Signature == IMAGE_NT_SIGNATURE ? nt : NULL;
}

/* A 14-byte absolute jump to TARGET placed above MODULE within 4 GiB of its base, so an export RVA can name it. */
static void *stub_near(HMODULE module, void *target)
{
    static BYTE *page;
    static BYTE *page_module_end;
    static size_t used;
    IMAGE_NT_HEADERS64 *nt = nt_headers(module);
    if (!nt)
        return NULL;
    BYTE *base = (BYTE *)module;
    BYTE *end = base + nt->OptionalHeader.SizeOfImage;
    if (!page || page_module_end != end || used + 16 > 4096) {
        page = NULL;
        used = 0;
        for (uintptr_t at = ((uintptr_t)end + 0xffff) & ~(uintptr_t)0xffff;
             at < (uintptr_t)base + 0xf0000000u; at += 0x10000) {
            page = VirtualAlloc((void *)at, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
            if (page)
                break;
        }
        if (!page)
            return NULL;
        page_module_end = end;
    }
    BYTE *stub = page + used;
    stub[0] = 0xff;
    stub[1] = 0x25;
    memset(stub + 2, 0, 4); /* jmp [rip+0] */
    memcpy(stub + 6, &target, 8);
    used += 16;
    FlushInstructionCache(GetCurrentProcess(), stub, 16);
    return stub;
}

static int write_protected(void *at, const void *value, size_t size)
{
    DWORD old;
    if (!VirtualProtect(at, size, PAGE_READWRITE, &old))
        return 0;
    memcpy(at, value, size);
    VirtualProtect(at, size, old, &old);
    return 1;
}

/* Points MODULE's export NAME at WRAPPER; returns the original address (NULL if not exported, a forwarder - then
 * the target module's export is the one to patch - or not patched). */
static int g_forwarder;
static void *patch_export(HMODULE module, const char *name, void *wrapper)
{
    g_forwarder = 0;
    IMAGE_NT_HEADERS64 *nt = nt_headers(module);
    if (!nt)
        return NULL;
    IMAGE_DATA_DIRECTORY dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!dir.VirtualAddress)
        return NULL;
    BYTE *base = (BYTE *)module;
    IMAGE_EXPORT_DIRECTORY *exp = (IMAGE_EXPORT_DIRECTORY *)(base + dir.VirtualAddress);
    DWORD *functions = (DWORD *)(base + exp->AddressOfFunctions);
    DWORD *names = (DWORD *)(base + exp->AddressOfNames);
    WORD *ordinals = (WORD *)(base + exp->AddressOfNameOrdinals);
    for (DWORD i = 0; i < exp->NumberOfNames; i++) {
        if (strcmp((const char *)(base + names[i]), name))
            continue;
        DWORD *slot = &functions[ordinals[i]];
        if (*slot >= dir.VirtualAddress && *slot < dir.VirtualAddress + dir.Size) {
            g_forwarder = 1;
            return NULL;
        }
        void *original = base + *slot;
        BYTE *stub = stub_near(module, wrapper);
        if (!stub)
            return NULL;
        DWORD rva = (DWORD)(stub - base);
        return write_protected(slot, &rva, sizeof(rva)) ? original : NULL;
    }
    return NULL;
}

/* Redirects every bound import of ORIGINAL in the loaded modules to WRAPPER. */
static void patch_imports(void *original, void *wrapper)
{
    HMODULE self = NULL;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCWSTR)(void *)patch_imports, &self);
    /* the loader's module list, through the PEB */
    BYTE *peb = (BYTE *)__readgsqword(0x60);
    LIST_ENTRY *head = (LIST_ENTRY *)(*(BYTE **)(peb + 0x18) + 0x10); /* PEB_LDR_DATA.InLoadOrderModuleList */
    for (LIST_ENTRY *e = head->Flink; e != head; e = e->Flink) {
        HMODULE module = *(HMODULE *)((BYTE *)e + 0x30); /* LDR_DATA_TABLE_ENTRY.DllBase */
        if (!module || module == self)
            continue;
        IMAGE_NT_HEADERS64 *nt = nt_headers(module);
        if (!nt)
            continue;
        IMAGE_DATA_DIRECTORY dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        if (!dir.VirtualAddress)
            continue;
        BYTE *base = (BYTE *)module;
        for (IMAGE_IMPORT_DESCRIPTOR *d = (IMAGE_IMPORT_DESCRIPTOR *)(base + dir.VirtualAddress); d->Name; d++) {
            for (void **slot = (void **)(base + d->FirstThunk); *slot; slot++) {
                if (*slot == original)
                    write_protected(slot, &wrapper, sizeof(wrapper));
            }
        }
    }
}

/* ---------------------------------------------------------------- EnumDisplayDevices */

typedef BOOL(WINAPI *EnumDisplayDevicesA_fn)(LPCSTR, DWORD, PDISPLAY_DEVICEA, DWORD);
typedef BOOL(WINAPI *EnumDisplayDevicesW_fn)(LPCWSTR, DWORD, PDISPLAY_DEVICEW, DWORD);
static EnumDisplayDevicesA_fn real_EnumDisplayDevicesA;
static EnumDisplayDevicesW_fn real_EnumDisplayDevicesW;

/* "PCI\VEN_1002&DEV_xxxx..." -> "PCI\VEN_10DE&DEV_2487...", as win32u does */
#define REWRITE_PCI_ID(id)                                                                                         \
    do {                                                                                                           \
        if (!memcmp((id), PCI_PREFIX, 8 * sizeof((id)[0])) && (id)[8] == '1' && (id)[9] == '0' &&                  \
            (id)[10] == '0' && (id)[11] == '2' && !memcmp((id) + 12, DEV_PREFIX, 5 * sizeof((id)[0]))) {             \
            (id)[10] = 'D', (id)[11] = 'E';                                                                        \
            (id)[17] = '2', (id)[18] = '4', (id)[19] = '8', (id)[20] = '7';                                       \
        }                                                                                                          \
    } while (0)

static BOOL WINAPI wrap_EnumDisplayDevicesA(LPCSTR device, DWORD index, PDISPLAY_DEVICEA info, DWORD flags)
{
    BOOL ok = real_EnumDisplayDevicesA(device, index, info, flags);
    if (ok && info && info->cb >= offsetof(DISPLAY_DEVICEA, DeviceID) + 21) {
#define PCI_PREFIX "PCI\\VEN_"
#define DEV_PREFIX "&DEV_"
        REWRITE_PCI_ID(info->DeviceID);
#undef PCI_PREFIX
#undef DEV_PREFIX
    }
    return ok;
}

static BOOL WINAPI wrap_EnumDisplayDevicesW(LPCWSTR device, DWORD index, PDISPLAY_DEVICEW info, DWORD flags)
{
    BOOL ok = real_EnumDisplayDevicesW(device, index, info, flags);
    if (ok && info && info->cb >= offsetof(DISPLAY_DEVICEW, DeviceID) + 21 * sizeof(WCHAR)) {
#define PCI_PREFIX L"PCI\\VEN_"
#define DEV_PREFIX L"&DEV_"
        REWRITE_PCI_ID(info->DeviceID);
#undef PCI_PREFIX
#undef DEV_PREFIX
    }
    return ok;
}

/* ---------------------------------------------------------------- Vulkan */

/* VkPhysicalDeviceProperties begins apiVersion, driverVersion, vendorID, deviceID; VkPhysicalDeviceProperties2 has
 * sType and pNext in front of it. */
typedef void(WINAPI *props_fn)(void *physical_device, void *properties);
typedef void(WINAPI *(WINAPI *gipa_fn)(void *instance, const char *name))(void);

struct vk_module {
    props_fn props, props2, props2khr;
    gipa_fn gipa;
};
static struct vk_module g_vk[2]; /* vulkan-1.dll, winevulkan.dll */

static void hide_amd(uint32_t *vendor_device)
{
    if (vendor_device[0] == AMD_VENDOR) {
        vendor_device[0] = NV_VENDOR;
        vendor_device[1] = NV_DEVICE;
    }
}

#define VK_WRAPPERS(n)                                                                                             \
    static void WINAPI wrap_props_##n(void *pd, void *p)                                                          \
    {                                                                                                              \
        g_vk[n].props(pd, p);                                                                                      \
        hide_amd((uint32_t *)p + 2);                                                                               \
    }                                                                                                              \
    static void WINAPI wrap_props2_##n(void *pd, void *p)                                                         \
    {                                                                                                              \
        g_vk[n].props2(pd, p);                                                                                     \
        hide_amd((uint32_t *)((BYTE *)p + 16) + 2);                                                                \
    }                                                                                                              \
    static void WINAPI wrap_props2khr_##n(void *pd, void *p)                                                      \
    {                                                                                                              \
        g_vk[n].props2khr(pd, p);                                                                                  \
        hide_amd((uint32_t *)((BYTE *)p + 16) + 2);                                                                \
    }                                                                                                              \
    static void(WINAPI *WINAPI wrap_gipa_##n(void *instance, const char *name))(void)                             \
    {                                                                                                              \
        void(WINAPI * f)(void) = g_vk[n].gipa(instance, name);                                                    \
        if (!f || !name)                                                                                           \
            return f;                                                                                              \
        if (!strcmp(name, "vkGetInstanceProcAddr"))                                                                \
            return (void(WINAPI *)(void))wrap_gipa_##n;                                                            \
        if (!strcmp(name, "vkGetPhysicalDeviceProperties")) {                                                      \
            g_vk[n].props = (props_fn)f;                                                                           \
            return (void(WINAPI *)(void))wrap_props_##n;                                                           \
        }                                                                                                          \
        if (!strcmp(name, "vkGetPhysicalDeviceProperties2")) {                                                     \
            g_vk[n].props2 = (props_fn)f;                                                                          \
            return (void(WINAPI *)(void))wrap_props2_##n;                                                          \
        }                                                                                                          \
        if (!strcmp(name, "vkGetPhysicalDeviceProperties2KHR")) {                                                  \
            g_vk[n].props2khr = (props_fn)f;                                                                       \
            return (void(WINAPI *)(void))wrap_props2khr_##n;                                                       \
        }                                                                                                          \
        return f;                                                                                                  \
    }
VK_WRAPPERS(0)
VK_WRAPPERS(1)

static int hook_vulkan(int n, const wchar_t *dll, void *gipa_w, void *props_w, void *props2_w)
{
    HMODULE module = LoadLibraryW(dll);
    if (!module)
        return 0;
    void *gipa = patch_export(module, "vkGetInstanceProcAddr", gipa_w);
    if (!gipa) {
        logf_(g_forwarder ? "%ls: forwards to winevulkan.dll" : "%ls: vkGetInstanceProcAddr not patched", dll);
        return 0;
    }
    g_vk[n].gipa = (gipa_fn)gipa;
    patch_imports(gipa, gipa_w);
    void *p = patch_export(module, "vkGetPhysicalDeviceProperties", props_w);
    if (p) {
        g_vk[n].props = (props_fn)p;
        patch_imports(p, props_w);
    }
    p = patch_export(module, "vkGetPhysicalDeviceProperties2", props2_w);
    if (p) {
        g_vk[n].props2 = (props_fn)p;
        patch_imports(p, props2_w);
    }
    logf_("%ls: Vulkan GPU identity wrapped", dll);
    return 1;
}

/* ---------------------------------------------------------------- setup */

static int visible_vendor(int *amd, int *nvidia)
{
    DISPLAY_DEVICEA dd;
    *amd = *nvidia = 0;
    for (DWORD i = 0;; i++) {
        memset(&dd, 0, sizeof(dd));
        dd.cb = sizeof(dd);
        if (!EnumDisplayDevicesA(NULL, i, &dd, 0))
            break;
        if (!(dd.StateFlags & DISPLAY_DEVICE_ACTIVE))
            continue;
        if (strstr(dd.DeviceID, "VEN_1002"))
            *amd = 1;
        if (strstr(dd.DeviceID, "VEN_10DE"))
            *nvidia = 1;
    }
    return *amd || *nvidia;
}

static void setup(void)
{
    wchar_t value[64];
    int mode = -1;
    if (GetEnvironmentVariableW(L"D4R_SPOOF_GPU", value, 64))
        mode = ini_bool(value);
    else {
        ini_read(L"SpoofGpu", value, 64);
        mode = ini_bool(value);
    }
    wchar_t arch[32];
    ini_read(L"GpuArch", arch, 32);
    if (!_wcsicmp(arch, L"auto") || !arch[0])
        wcscpy(arch, L"AD100");

    if (mode == 0) {
        logf_("SpoofGpu = false: identity left to the launch options");
        return;
    }
    int amd, nvidia;
    visible_vendor(&amd, &nvidia);
    if (nvidia) {
        if (env_set(L"WINE_HIDE_AMD_GPU")) {
            /* the launch options still hide the AMD GPU; only the architecture may be missing */
            env_default(L"DXVK_NVAPI_GPU_ARCH", arch);
            logf_("an NVIDIA identity is already visible (WINE_HIDE_AMD_GPU)");
        } else
            logf_("an NVIDIA GPU is visible: nothing to do");
        return;
    }
    if (!amd && mode != 1) {
        logf_("no AMD display adapter found: nothing to do");
        return;
    }

    env_default(L"DXVK_ENABLE_NVAPI", L"1");
    env_default(L"DXVK_NVAPI_ALLOW_OTHER_DRIVERS", L"1");
    env_default(L"DXVK_NVAPI_DRIVER_VERSION", L"99999");
    env_default(L"DXVK_NVAPI_GPU_ARCH", arch);
    env_default(L"WINE_HIDE_AMD_GPU", L"1");

    HMODULE user32 = LoadLibraryW(L"user32.dll");
    real_EnumDisplayDevicesA = (EnumDisplayDevicesA_fn)patch_export(user32, "EnumDisplayDevicesA",
                                                                    (void *)wrap_EnumDisplayDevicesA);
    real_EnumDisplayDevicesW = (EnumDisplayDevicesW_fn)patch_export(user32, "EnumDisplayDevicesW",
                                                                    (void *)wrap_EnumDisplayDevicesW);
    if (real_EnumDisplayDevicesA)
        patch_imports((void *)real_EnumDisplayDevicesA, (void *)wrap_EnumDisplayDevicesA);
    if (real_EnumDisplayDevicesW)
        patch_imports((void *)real_EnumDisplayDevicesW, (void *)wrap_EnumDisplayDevicesW);
    logf_("EnumDisplayDevices wrapped: A %s, W %s", real_EnumDisplayDevicesA ? "yes" : "no",
          real_EnumDisplayDevicesW ? "yes" : "no");

    /* Proton 11's vulkan-1.dll forwards to winevulkan.dll; DXVK loads winevulkan.dll itself */
    hook_vulkan(1, L"winevulkan.dll", (void *)wrap_gipa_1, (void *)wrap_props_1, (void *)wrap_props2_1);
    hook_vulkan(0, L"vulkan-1.dll", (void *)wrap_gipa_0, (void *)wrap_props_0, (void *)wrap_props2_0);
    logf_("AMD GPU shown as NVIDIA %04x:%04x, architecture %ls", NV_VENDOR, NV_DEVICE, arch);
}

/* ---------------------------------------------------------------- DLSS frame generation */

static void ini_read_section(const wchar_t *section, const wchar_t *key, const wchar_t *def, wchar_t *out, DWORD size)
{
    wchar_t path[MAX_PATH + 32];
    _snwprintf(path, MAX_PATH + 32, L"%ls\\d4r\\d4r.ini", g_dir);
    path[MAX_PATH + 31] = 0;
    GetPrivateProfileStringW(section, key, def, out, size, path);
}

static int same_file_contents(const wchar_t *a, const wchar_t *b)
{
    WIN32_FILE_ATTRIBUTE_DATA x, y;
    if (!GetFileAttributesExW(a, GetFileExInfoStandard, &x) || !GetFileAttributesExW(b, GetFileExInfoStandard, &y))
        return 0;
    return x.nFileSizeHigh == y.nFileSizeHigh && x.nFileSizeLow == y.nFileSizeLow &&
           CompareFileTime(&x.ftLastWriteTime, &y.ftLastWriteTime) == 0;
}

typedef int(WINAPI *d4rSetEnv_fn)(const char *, const char *, int);
typedef char *(CDECL *wine_get_unix_file_name_fn)(const wchar_t *);

static void setup_dlssg(void)
{
    wchar_t backend[32];
    if (!GetEnvironmentVariableW(L"D4R_FG_BACKEND", backend, 32))
        ini_read_section(L"FrameGeneration", L"Backend", L"off", backend, 32);
    if (_wcsicmp(backend, L"dlssg"))
        return;
    wchar_t layer[MAX_PATH + 64], shim[MAX_PATH + 32], next[MAX_PATH + 32], bridge[MAX_PATH + 32];
    _snwprintf(layer, MAX_PATH + 64, L"%ls\\d4r\\fg\\dlssg\\layer\\nvxrt_layer.json", g_dir);
    _snwprintf(shim, MAX_PATH + 32, L"%ls\\d4r\\nvngx.dll", g_dir);
    _snwprintf(next, MAX_PATH + 32, L"%ls\\d4r\\d4r_dlssg_next.dll", g_dir);
    _snwprintf(bridge, MAX_PATH + 32, L"%ls\\d4r\\nvcuda.dll", g_dir);
    if (GetFileAttributesW(layer) == INVALID_FILE_ATTRIBUTES) {
        logf_("frame generation: %ls missing (re-extract the d4r zip)", layer);
        return;
    }
    if (!same_file_contents(shim, next)) {
        BOOL ok = CopyFileW(shim, next, FALSE);
        logf_("frame generation: d4r\\d4r_dlssg_next.dll %s from d4r\\nvngx.dll", ok ? "refreshed" : "NOT refreshed");
    }
    wine_get_unix_file_name_fn unix_name = (wine_get_unix_file_name_fn)(void *)GetProcAddress(
        GetModuleHandleW(L"kernel32.dll"), "wine_get_unix_file_name");
    HMODULE bridge_module = LoadLibraryW(bridge);
    d4rSetEnv_fn set_env = bridge_module ? (d4rSetEnv_fn)(void *)GetProcAddress(bridge_module, "d4rSetEnv") : NULL;
    wchar_t layer_dir[MAX_PATH + 64];
    _snwprintf(layer_dir, MAX_PATH + 64, L"%ls\\d4r\\fg\\dlssg\\layer", g_dir);
    char *unix_layer = unix_name ? unix_name(layer_dir) : NULL;
    if (!set_env || !unix_layer) {
        logf_("frame generation: cannot reach the Linux environment (bridge %p, d4rSetEnv %p, path %s)",
              (void *)bridge_module, (void *)set_env, unix_layer ? unix_layer : "-");
        return;
    }
    char value[4096];
    const char *old = getenv("VK_ADD_LAYER_PATH"); /* the Windows copy mirrors the start-up environment */
    _snprintf(value, sizeof(value), "%s%s%s", unix_layer, old && *old ? ":" : "", old ? old : "");
    value[sizeof(value) - 1] = 0;
    set_env("VK_ADD_LAYER_PATH", value, 1);
    old = getenv("VK_INSTANCE_LAYERS");
    if (!old || !strstr(old, "VK_LAYER_NVXRT_nvx")) {
        _snprintf(value, sizeof(value), "%s%sVK_LAYER_NVXRT_nvx", old && *old ? old : "", old && *old ? ":" : "");
        value[sizeof(value) - 1] = 0;
        set_env("VK_INSTANCE_LAYERS", value, 1);
    }
    /* [FrameGeneration] KernelCache: the translated kernels for this GPU generation (d4r-features.py writes it) */
    wchar_t cache[MAX_PATH];
    ini_read_section(L"FrameGeneration", L"KernelCache", L"", cache, MAX_PATH);
    if (cache[0]) {
        WideCharToMultiByte(CP_UTF8, 0, cache, -1, value, sizeof(value), NULL, NULL);
        set_env("NVXRT_CACHE", value, 0);
    }
    _snprintf(value, sizeof(value), "%s/../nvxrt.log", unix_layer);
    value[sizeof(value) - 1] = 0;
    set_env("NVXRT_LOG", value, 0);
    /* [FrameGeneration] Exact = false: texture-unit filtering, fp32 accumulation and RGBA8 storage for two
     * intermediate images, about 10% less GPU time (PSNR ~55 dB against the exact kernels) */
    wchar_t exact[16];
    ini_read_section(L"FrameGeneration", L"Exact", L"false", exact, 16);
    if (ini_bool(exact) != 1) {
        set_env("NVXRT_TEX", "hw", 0);
        set_env("NVXRT_ACC", "fast", 0);
        set_env("NVXRT_SHADOW", "b,c", 0);
    }
    logf_("frame generation: DLSS-G layer enabled from %s (%s kernels)", unix_layer, ini_bool(exact) == 1 ? "exact" : "fast");
    HeapFree(GetProcessHeap(), 0, unix_layer);
}

/* imported by name by the patched OptiScaler DLL */
/* ---------------------------------------------------------------- vkd3d-proton from d4r\vkd3d */

/* Installs outside the game folder (GE-Proton's system32\umu, which its ntdll loads as dxgi.dll when
 * WINE_OPTISCALER_NAME is set; anti-cheat games such as The Outlast Trials crash on unsigned DLLs loaded from the
 * game folder) cannot put d4r's d3d12.dll next to the executable, and Proton rewrites system32's on every launch.
 * d4r\vkd3d\d3d12core.dll and d3d12.dll are loaded here by full path instead: later loads by name (the game's
 * d3d12.dll, vkd3d-proton's d3d12core.dll, an Agility SDK D3D12Core.dll in the game folder) find them loaded. */
static void preload_vkd3d(void)
{
    static const wchar_t *const names[] = {L"d3d12core.dll", L"d3d12.dll"};
    wchar_t path[MAX_PATH + 32];
    for (int i = 0; i < 2; i++) {
        _snwprintf(path, MAX_PATH + 32, L"%ls\\d4r\\vkd3d\\%ls", g_dir, names[i]);
        path[MAX_PATH + 31] = 0;
        if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES)
            return;
        HMODULE loaded = GetModuleHandleW(names[i]);
        if (loaded) {
            wchar_t other[MAX_PATH];
            other[0] = 0;
            GetModuleFileNameW(loaded, other, MAX_PATH);
            logf_("vkd3d: %ls already loaded from %ls; d4r\\vkd3d not used", names[i], other);
            return;
        }
        HMODULE module = LoadLibraryW(path);
        logf_("vkd3d: d4r\\vkd3d\\%ls %s (error %lu)", names[i], module ? "loaded" : "NOT loaded",
              module ? 0ul : GetLastError());
        if (!module)
            return;
    }
}

__declspec(dllexport) int d4r_boot(void)
{
    return 1;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
    (void)reserved;
    if (reason != DLL_PROCESS_ATTACH)
        return TRUE;
    DisableThreadLibraryCalls(instance);
    DWORD n = GetModuleFileNameW(instance, g_dir, MAX_PATH);
    if (n && n < MAX_PATH) {
        wchar_t *slash = wcsrchr(g_dir, L'\\');
        if (slash)
            *slash = 0;
        /* next to d4r_nvngx.log, when the d4r folder is there */
        wchar_t path[MAX_PATH + 32];
        _snwprintf(path, MAX_PATH + 32, L"%ls\\d4r", g_dir);
        path[MAX_PATH + 31] = 0;
        DWORD attributes = GetFileAttributesW(path);
        /* GE-Proton's umu loader: OptiScaler is system32\umu\dxgi.dll, and its imports are searched in system32,
         * so this DLL sits there; d4r is next to the loaded dxgi.dll */
        HMODULE dxgi = GetModuleHandleW(L"dxgi.dll");
        if ((attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) && dxgi) {
            wchar_t other[MAX_PATH];
            DWORD m = GetModuleFileNameW(dxgi, other, MAX_PATH);
            wchar_t *end = m && m < MAX_PATH ? wcsrchr(other, L'\\') : NULL;
            if (end) {
                *end = 0;
                _snwprintf(path, MAX_PATH + 32, L"%ls\\d4r", other);
                path[MAX_PATH + 31] = 0;
                attributes = GetFileAttributesW(path);
                if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY))
                    wcscpy(g_dir, other);
            }
        }
        if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
            wcscat(path, L"\\d4r_boot.log");
            g_log = _wfopen(path, L"w");
        }
    }
    setup();
    setup_dlssg();
    preload_vkd3d();
    if (g_log) {
        fclose(g_log);
        g_log = NULL;
    }
    return TRUE;
}
