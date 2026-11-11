// nvngx.dll_dlssg_host.dll: an NGX core layer that runs NVIDIA's own DLSS Frame Generation DLL (the game's
// nvngx_dlssg.dll, unmodified) for NGX feature 11 and forwards everything else to the next NGX core.
//
// The chain:  game -> Streamline (sl.dlss_g) -> OptiScaler (dxgi.dll, NvngxPath = this DLL)
//             -> feature 11: nvngx_dlssg.dll here;  everything else: the next core (d4r: DLSS SR / RR), or none
//
// What an NVIDIA NGX core does for a feature DLL ("snippet") and therefore this DLL does for nvngx_dlssg.dll:
//  - D3D12_Init_Ext(appId, dataPath, device, sdkVersion, initParams) once per device; initParams carries
//    Context.GPU.ArchID (the snippet then skips its own NVAPI GPU query; 0x1b0 = Blackwell: multi-frame generation and
//    the PTX kernel path), the logging level and a log callback (into this DLL's log).
//  - capability parameters: FrameGeneration.Available etc. plus the snippet's PopulateParameters_Impl
//    (DLSSG.MultiFrameCountMax, DLSSG.ModelVersion, ...).
//  - Create/Evaluate/Release with the application's parameter maps; handles are wrapped so their ids cannot clash
//    with the other cores' ids (OptiScaler keys its bookkeeping by handle id).
// nvngx_dlssg.dll accepts calls only from a module whose path contains "nvngx.dll" (it checks the caller with
// GetModuleHandleEx), hence this DLL's name. The name must not end in "nvngx.dll": OptiScaler answers such loads.
//
// Config: dlssg_host.ini next to this DLL, section [Host]: Next=<next core | none> (default: d4r\d4r_dlssg_next.dll
// when present, else none), FG=<DLSS-G DLL> (default nvngx_dlssg.dll; relative paths are taken from the game's
// executable folder), ArchID=0x1b0,
// FGEnabled=1, LogLevel=1 (0 off, 1 on, 2 verbose), Timing=1, Log=1, Lite=0 (1: FG Lite, DLSS-G >= 310.9.16, UI
// recomposition off), RrCreateMatrices=1 (see rrCreateMatrices), DumpDir/DumpAt/DumpReal (test captures).
// Log: dlssg_host.log next to this DLL.
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <windows.h>
#include <d3d12.h>
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "cubin_d3d12.h"

using NgxResult = unsigned int;
struct NgxHandle { unsigned int Id; };
static const NgxResult kOk = 0x1, kFail = 0xBAD00000u, kNotInit = 0xBAD00007u, kNotFound = 0xBAD00004u;
static const unsigned int kFeatureFG = 11;
static bool failed(NgxResult r) { return (r & 0xFFF00000u) == 0xBAD00000u; }

// ------------------------------------------------------------------------------------------------ MSVC-ABI parameters
// NVSDK_NGX_Parameter is a C++ interface with overloaded virtuals. MSVC (which built NVIDIA's DLLs, Streamline and
// OptiScaler) groups overloads and lays them out in reverse declaration order; this table is that layout (checked with
// clang-cl -fdump-vtable-layouts). On x64 Windows `this` is simply the first argument, so plain functions work.
enum : int { kSetPtr = 0, kSetD3D12 = 1, kSetD3D11 = 2, kSetI = 3, kSetU = 4, kSetD = 5, kSetF = 6, kSetULL = 7,
             kGetPtr = 8, kGetD3D12 = 9, kGetD3D11 = 10, kGetI = 11, kGetU = 12, kGetD = 13, kGetF = 14, kGetULL = 15,
             kReset = 16 };

namespace ms {
template <typename F> F slot(void* obj, int i) { return reinterpret_cast<F>((*reinterpret_cast<void***>(obj))[i]); }
void setU(void* p, const char* n, unsigned v) { slot<void (*)(void*, const char*, unsigned)>(p, kSetU)(p, n, v); }
void setI(void* p, const char* n, int v) { slot<void (*)(void*, const char*, int)>(p, kSetI)(p, n, v); }
void setPtr(void* p, const char* n, void* v) { slot<void (*)(void*, const char*, void*)>(p, kSetPtr)(p, n, v); }
NgxResult getU(void* p, const char* n, unsigned* v) { return slot<NgxResult (*)(void*, const char*, unsigned*)>(p, kGetU)(p, n, v); }
NgxResult getI(void* p, const char* n, int* v) { return slot<NgxResult (*)(void*, const char*, int*)>(p, kGetI)(p, n, v); }
NgxResult getULL(void* p, const char* n, unsigned long long* v)
{
    return slot<NgxResult (*)(void*, const char*, unsigned long long*)>(p, kGetULL)(p, n, v);
}
NgxResult getPtr(void* p, const char* n, void** v) { return slot<NgxResult (*)(void*, const char*, void**)>(p, kGetPtr)(p, n, v); }
NgxResult getF(void* p, const char* n, float* v) { return slot<NgxResult (*)(void*, const char*, float*)>(p, kGetF)(p, n, v); }
NgxResult getD3D12(void* p, const char* n, ID3D12Resource** v)
{
    return slot<NgxResult (*)(void*, const char*, ID3D12Resource**)>(p, kGetD3D12)(p, n, v);
}
} // namespace ms

// A small parameter map with that layout, for the Init call (the snippet only reads from it).
struct Param {
    const void* const* vtbl;
    struct V { int type; unsigned long long u; double d; void* p; };
    std::map<std::string, V> m;
    enum { TU = 1, TI, TF, TD, TP };
    void put(const char* n, V v) { if (n) m[n] = v; }
    const V* find(const char* n) const { auto it = n ? m.find(n) : m.end(); return it == m.end() ? nullptr : &it->second; }
};
namespace pm {
void setPtr(Param* s, const char* n, void* v) { s->put(n, {Param::TP, 0, 0, v}); }
void setI(Param* s, const char* n, int v) { s->put(n, {Param::TI, (unsigned long long)(long long)v, (double)v, nullptr}); }
void setU(Param* s, const char* n, unsigned v) { s->put(n, {Param::TU, v, (double)v, nullptr}); }
void setD(Param* s, const char* n, double v) { s->put(n, {Param::TD, (unsigned long long)v, v, nullptr}); }
void setF(Param* s, const char* n, float v) { s->put(n, {Param::TF, (unsigned long long)v, v, nullptr}); }
void setULL(Param* s, const char* n, unsigned long long v) { s->put(n, {Param::TU, v, (double)v, nullptr}); }
template <typename T> NgxResult num(const Param* s, const char* n, T* o)
{
    auto v = s->find(n);
    if (!v || v->type == Param::TP || !o) return 0xBAD00010u; // unsupported parameter
    *o = (v->type == Param::TF || v->type == Param::TD) ? (T)v->d : (T)v->u;
    return kOk;
}
NgxResult getPtr(const Param* s, const char* n, void** o)
{
    auto v = s->find(n);
    if (!v || !o) return 0xBAD00010u;
    *o = v->type == Param::TP ? v->p : (void*)(uintptr_t)v->u;
    return kOk;
}
NgxResult getI(const Param* s, const char* n, int* o) { return num(s, n, o); }
NgxResult getU(const Param* s, const char* n, unsigned* o) { return num(s, n, o); }
NgxResult getD(const Param* s, const char* n, double* o) { return num(s, n, o); }
NgxResult getF(const Param* s, const char* n, float* o) { return num(s, n, o); }
NgxResult getULL(const Param* s, const char* n, unsigned long long* o) { return num(s, n, o); }
void reset(Param* s) { s->m.clear(); }
} // namespace pm
static const void* const kParamVtbl[17] = {
    (void*)pm::setPtr, (void*)pm::setPtr, (void*)pm::setPtr, (void*)pm::setI, (void*)pm::setU, (void*)pm::setD,
    (void*)pm::setF, (void*)pm::setULL, (void*)pm::getPtr, (void*)pm::getPtr, (void*)pm::getPtr, (void*)pm::getI,
    (void*)pm::getU, (void*)pm::getD, (void*)pm::getF, (void*)pm::getULL, (void*)pm::reset};

// ------------------------------------------------------------------------------------------------ state
namespace {

std::mutex g_lock, g_logLock;
HMODULE g_next = nullptr, g_fg = nullptr;
bool g_loaded = false, g_fgEnabled = true, g_fgInited = false, g_standalone = false;
unsigned g_arch = 0x1b0, g_logLevel = 1, g_timing = 1;
unsigned g_rrCreateMatrices = 1; // RrCreateMatrices=1: camera matrices on Ray Reconstruction's CreateFeature (see rrCreateMatrices)
unsigned g_lite = 0;             // Lite=1: create DLSS-G with DLSSG.LiteCandidateBlendingEnabled (DLSS-G >= 310.9.16)
std::wstring g_dumpDir;          // DumpDir=<dir>: write the inputs + parameters of DumpReal real frames (test captures)
unsigned g_dumpAt = 600, g_dumpReal = 0;
void* g_device = nullptr;
HANDLE g_log = INVALID_HANDLE_VALUE;
std::wstring g_base; // this DLL's path without ".dll"
unsigned g_nextId = 0x7d000;
struct FgHandle { NgxHandle pub; NgxHandle* inner; };
std::map<const void*, FgHandle*> g_handles; // keyed by the public handle pointer

void logf(const char* fmt, ...)
{
    if (g_log == INVALID_HANDLE_VALUE) return;
    char buf[2048];
    SYSTEMTIME t; GetLocalTime(&t);
    int n = snprintf(buf, sizeof(buf), "[%02d:%02d:%02d.%03d tid %lu] ", t.wHour, t.wMinute, t.wSecond,
                     t.wMilliseconds, GetCurrentThreadId());
    va_list ap; va_start(ap, fmt); n += vsnprintf(buf + n, sizeof(buf) - n - 2, fmt, ap); va_end(ap);
    if (n > (int)sizeof(buf) - 2) n = sizeof(buf) - 2;
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) n--;
    buf[n++] = '\n';
    std::lock_guard<std::mutex> l(g_logLock);
    DWORD w; WriteFile(g_log, buf, n, &w, nullptr);
}

void __cdecl snippetLog(const char* message, int level, int feature)
{
    logf("dlssg[%d/%d]: %s", level, feature, message ? message : "(null)");
}

std::wstring iniString(const wchar_t* key, const wchar_t* def)
{
    wchar_t buf[1024] = {};
    GetPrivateProfileStringW(L"Host", key, def, buf, 1024, (g_base + L".ini").c_str());
    return buf;
}
unsigned iniUInt(const wchar_t* key, unsigned def)
{
    std::wstring s = iniString(key, L"");
    return s.empty() ? def : (unsigned)wcstoul(s.c_str(), nullptr, 0);
}

// The player's settings: d4r.ini [FrameGeneration] (Backend, Model, Exact, KernelCache; env D4R_FG_BACKEND,
// D4R_FG_MODEL). dlssg_host.ini [Host] keys are developer overrides and win when present.
std::wstring g_d4rIni;
std::wstring d4rString(const wchar_t* key, const wchar_t* env, const wchar_t* def)
{
    wchar_t buf[1024] = {};
    if (env && GetEnvironmentVariableW(env, buf, 1024)) return buf;
    if (g_d4rIni.empty()) return def;
    GetPrivateProfileStringW(L"FrameGeneration", key, def, buf, 1024, g_d4rIni.c_str());
    std::wstring s = buf;
    size_t c = s.find(L';');   // inline comments
    if (c != std::wstring::npos) s.resize(c);
    while (!s.empty() && iswspace(s.back())) s.pop_back();
    while (!s.empty() && iswspace(s.front())) s.erase(s.begin());
    return s.empty() ? def : s;
}
bool d4rBool(const wchar_t* key, bool def)
{
    std::wstring s = d4rString(key, nullptr, def ? L"true" : L"false");
    return !_wcsicmp(s.c_str(), L"true") || !_wcsicmp(s.c_str(), L"1") || !_wcsicmp(s.c_str(), L"on") || !_wcsicmp(s.c_str(), L"yes");
}

bool wrapGetProcAddress(HMODULE module);
bool g_cubin = false; // NVIDIA's cubin interface is served here (native Windows; under Proton dxvk-nvapi does it)

// Native Windows: no vkd3d-proton, so nothing implements the D3D12 cubin calls DLSS-G makes; cubin_d3d12.cpp does.
// Cubin=auto (default): on unless running under Wine; 0 / 1 force it.
void setupCubin(const std::wstring& dir)
{
    unsigned mode = iniUInt(L"Cubin", 2);
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    bool wine = ntdll && GetProcAddress(ntdll, "wine_get_version");
    g_cubin = mode == 1 || (mode == 2 && !wine);
    logf("cubin interface: %s", g_cubin ? "served by this DLL (native D3D12)" : wine ? "left to dxvk-nvapi (Wine)" : "off");
    if (!g_cubin) return;
    cux::Config c;
    wchar_t local[MAX_PATH] = {};
    GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH);
    std::wstring cache = d4rString(L"KernelCache", nullptr, L"");
    if (cache.empty() || !_wcsicmp(cache.c_str(), L"auto")) cache = std::wstring(local) + L"\\d4r\\nvxrt-gfx11";
    c.cacheDir = iniString(L"Cache", cache.c_str());
    c.logPath = dir + L"nvxrt.log";
    bool exact = iniUInt(L"Exact", d4rBool(L"Exact", false) ? 1 : 0) != 0;
    c.accFast = !exact && iniUInt(L"AccFast", 1) != 0;
    c.texHw = !exact && iniUInt(L"TexHw", 1) != 0;
    c.groups = iniUInt(L"Groups", 1) != 0;
    c.trace = iniUInt(L"Trace", 0) != 0;
    c.gpuSync = iniUInt(L"GpuSync", 1) != 0;
    c.splitBarriers = iniUInt(L"SplitBarriers", 0) != 0;
    c.vkClears = iniUInt(L"VkClears", 1) != 0;
    c.vkCopies = iniUInt(L"VkCopies", 1) != 0;
    c.queuePriority = iniUInt(L"QueuePriority", 0);
    c.dxil = !_wcsicmp(iniString(L"Route", L"vulkan").c_str(), L"dxil");
    c.dxc = iniString(L"Dxc", (dir + L"dxc\dxc.exe").c_str());
    c.splitQueries = iniUInt(L"SplitQueries", 0) != 0;
    c.splitLog = iniUInt(L"SplitLog", 0) != 0;
    c.kernelTiming = iniUInt(L"KernelTiming", 0) != 0;
    c.computeQueue = iniUInt(L"ComputeQueue", 1) != 0;
    cux::configure(c);
}

void load()
{
    std::lock_guard<std::mutex> l(g_lock);
    if (g_loaded) return;
    g_loaded = true;
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&load), &self);
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(self, path, MAX_PATH);
    // Companion files must not have "nvngx.dll" in their names: OptiScaler's CreateFileW hook redirects any such
    // path (outside the Windows directory) to the game's nvngx_dlss.dll (its signature-check trick).
    std::wstring self_path = path;
    std::wstring dir = self_path.substr(0, self_path.find_last_of(L"\\/") + 1);
    g_base = dir + L"dlssg_host";
    if (iniUInt(L"Log", 1))
        g_log = CreateFileW((g_base + L".log").c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    // relative Next/FG paths (and the defaults) are taken from the game's executable folder
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring exe_path = exe;
    std::wstring exe_dir = exe_path.substr(0, exe_path.find_last_of(L"\\/") + 1);
    auto resolve = [&](std::wstring p) {
        bool abs = (p.size() > 1 && p[1] == L':') || (!p.empty() && (p[0] == L'\\' || p[0] == L'/'));
        return abs || _wcsicmp(p.c_str(), L"none") == 0 ? p : exe_dir + p;
    };
    // default next core: d4r's shim under a name OptiScaler's nvngx.dll load hook leaves alone (dlssg-install
    // keeps that copy in sync), else none
    std::wstring d4r_core = exe_dir + L"d4r\\d4r_dlssg_next.dll";
    std::wstring next = resolve(iniString(L"Next",
        GetFileAttributesW(d4r_core.c_str()) != INVALID_FILE_ATTRIBUTES ? d4r_core.c_str() : L"none"));
    // d4r.ini: the game's d4r folder, or two levels above this DLL (d4r\fg\dlssg)
    for (std::wstring p : {exe_dir + L"d4r\\d4r.ini", dir + L"..\\..\\d4r.ini"})
        if (g_d4rIni.empty() && GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES) g_d4rIni = p;
    // Model: full (DLSS-G's full model) or lite (NVIDIA FG Lite, DLSS-G >= 310.9.16); auto: Lite (RDNA3, the only
    // generation the native Windows executor targets so far). Each model has its DLL: fg\dlssg\<model>\nvngx_dlssg.dll
    // (install.ps1 downloads them), else the game's own nvngx_dlssg.dll.
    std::wstring model = d4rString(L"Model", L"D4R_FG_MODEL", L"auto");
    bool lite = !_wcsicmp(model.c_str(), L"lite") || !_wcsicmp(model.c_str(), L"auto");
    std::wstring modelDll = dir + (lite ? L"lite\\nvngx_dlssg.dll" : L"full\\nvngx_dlssg.dll");
    if (GetFileAttributesW(modelDll.c_str()) == INVALID_FILE_ATTRIBUTES) modelDll = L"nvngx_dlssg.dll";
    std::wstring fg = resolve(iniString(L"FG", modelDll.c_str()));
    std::wstring backend = d4rString(L"Backend", L"D4R_FG_BACKEND", L"dlssg");
    g_fgEnabled = iniUInt(L"FGEnabled", _wcsicmp(backend.c_str(), L"off") ? 1 : 0) != 0;
    g_arch = iniUInt(L"ArchID", 0x1b0);
    g_logLevel = iniUInt(L"LogLevel", 1);
    g_timing = iniUInt(L"Timing", 1);
    g_lite = iniUInt(L"Lite", lite ? 1 : 0);
    logf("settings: %ls (Backend %ls, Model %ls -> %s)", g_d4rIni.empty() ? L"no d4r.ini" : g_d4rIni.c_str(),
         backend.c_str(), model.c_str(), g_lite ? "lite" : "full");
    g_rrCreateMatrices = iniUInt(L"RrCreateMatrices", 1);
    g_dumpDir = iniString(L"DumpDir", L"");
    g_dumpAt = iniUInt(L"DumpAt", 600);
    g_dumpReal = iniUInt(L"DumpReal", 0);
    setupCubin(dir);
    // Next=none: no next core (test harness); this DLL then hands out its own parameter maps
    g_standalone = _wcsicmp(next.c_str(), L"none") == 0;
    if (!g_standalone) {
        g_next = LoadLibraryW(next.c_str());
        logf("next core %ls: %s (error %lu)", next.c_str(), g_next ? "loaded" : "FAILED",
             g_next ? 0 : GetLastError());
    } else {
        logf("standalone (no next core)");
    }
    if (g_fgEnabled) {
        g_fg = LoadLibraryW(fg.c_str());
        logf("frame generation %ls: %s (error %lu), ArchID 0x%x", fg.c_str(), g_fg ? "loaded" : "FAILED",
             g_fg ? 0 : GetLastError(), g_arch);
        if (g_fg && ((g_arch && iniUInt(L"NvapiArch", 1)) || g_cubin))
            logf("NVAPI architecture query of DLSS-G %s", wrapGetProcAddress(g_fg) ? "wrapped" : "NOT wrapped");
    } else {
        logf("frame generation disabled (FGEnabled=0)");
    }
}

template <typename F> F fn(HMODULE m, const char* name) { return m ? reinterpret_cast<F>(GetProcAddress(m, name)) : nullptr; }

// DLSS-G 310.7 ignores Context.GPU.ArchID and asks NVAPI (NvAPI_GPU_GetArchInfo) for the architecture. The process
// reports Ada (DXVK_NVAPI_GPU_ARCH=AD100), which DLSS SR's network weights need; DLSS-G would then pick its sm_89 SASS
// kernels, which nvxrt cannot run. So nvngx_dlssg.dll alone gets ArchID from that query: its imported GetProcAddress
// hands out a wrapped nvapi_QueryInterface (NvapiArch=0 turns this off).
using QueryInterfaceFn = void*(__cdecl*)(unsigned);
using GetArchInfoFn = int(__cdecl*)(void*, unsigned*);
QueryInterfaceFn g_realQueryInterface = nullptr;
GetArchInfoFn g_realGetArchInfo = nullptr;
const unsigned kNvAPI_GPU_GetArchInfo = 0xD8265D24u;

int __cdecl fgGetArchInfo(void* gpu, unsigned* info)
{
    int status = g_realGetArchInfo(gpu, info);
    // NV_GPU_ARCH_INFO: version, architecture, implementation, revision
    if (status == 0 && info && info[1] != g_arch) {
        logf("NvAPI_GPU_GetArchInfo for DLSS-G: architecture 0x%x reported as 0x%x", info[1], g_arch);
        info[1] = g_arch;
    }
    return status;
}

void* __cdecl fgQueryInterface(unsigned id)
{
    if (g_cubin)
        if (void* own = cux::queryInterface(id)) return own;
    void* f = g_realQueryInterface ? g_realQueryInterface(id) : nullptr;
    if (g_logLevel >= 2) logf("nvapi_QueryInterface(0x%08x) -> %p", id, f);
    if (id == kNvAPI_GPU_GetArchInfo && f) {
        g_realGetArchInfo = reinterpret_cast<GetArchInfoFn>(f);
        return reinterpret_cast<void*>(fgGetArchInfo);
    }
    return f;
}

FARPROC WINAPI fgGetProcAddress(HMODULE module, LPCSTR name)
{
    FARPROC f = GetProcAddress(module, name);
    bool named = reinterpret_cast<uintptr_t>(name) > 0xffff;
    if (named && strcmp(name, "nvapi_QueryInterface") == 0 && (f || g_cubin)) {
        g_realQueryInterface = reinterpret_cast<QueryInterfaceFn>(f);
        logf("DLSS-G resolves nvapi_QueryInterface (real %p)", reinterpret_cast<void*>(f));
        return reinterpret_cast<FARPROC>(fgQueryInterface);
    }
    return f;
}

// DLSS-G loads nvapi64.dll itself (from System32 only, which AMD systems lack). Native Windows: the copy already in
// the process (OptiScaler's / d4r's), else one next to this DLL or the game's d4r\nvapi folder.
HMODULE nvapiModule(HMODULE loaded)
{
    if (loaded) return loaded;
    if (HMODULE m = GetModuleHandleW(L"nvapi64.dll")) return m;
    wchar_t self[MAX_PATH] = {}, exe[MAX_PATH] = {};
    HMODULE me = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&nvapiModule), &me);
    GetModuleFileNameW(me, self, MAX_PATH);
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring s = self, e = exe;
    for (std::wstring p : {s.substr(0, s.find_last_of(L"\\/") + 1) + L"nvapi64.dll",
                           e.substr(0, e.find_last_of(L"\\/") + 1) + L"nvapi64.dll",
                           e.substr(0, e.find_last_of(L"\\/") + 1) + L"d4r\\nvapi\\nvapi64.dll"})
        if (HMODULE m = LoadLibraryW(p.c_str())) { logf("nvapi64.dll for DLSS-G: %ls", p.c_str()); return m; }
    return nullptr;
}
bool isNvapi(const wchar_t* n) { return n && wcsstr(_wcslwr(std::wstring(n).data()), L"nvapi64") != nullptr; }
bool isNvapiA(const char* n) { std::string s = n ? n : ""; for (char& c : s) c = (char)tolower(c); return s.find("nvapi64") != std::string::npos; }

HMODULE WINAPI fgLoadLibraryExW(LPCWSTR name, HANDLE file, DWORD flags)
{
    HMODULE m = LoadLibraryExW(name, file, flags);
    if (g_cubin && isNvapi(name)) { m = nvapiModule(m); logf("DLSS-G LoadLibraryExW(%ls, 0x%lx) -> %p", name, flags, (void*)m); }
    return m;
}
HMODULE WINAPI fgLoadLibraryW(LPCWSTR name)
{
    HMODULE m = LoadLibraryW(name);
    if (g_cubin && isNvapi(name)) { m = nvapiModule(m); logf("DLSS-G LoadLibraryW(%ls) -> %p", name, (void*)m); }
    return m;
}
HMODULE WINAPI fgLoadLibraryExA(LPCSTR name, HANDLE file, DWORD flags)
{
    HMODULE m = LoadLibraryExA(name, file, flags);
    if (g_cubin && isNvapiA(name)) { m = nvapiModule(m); logf("DLSS-G LoadLibraryExA(%s) -> %p", name, (void*)m); }
    return m;
}
HMODULE WINAPI fgLoadLibraryA(LPCSTR name)
{
    HMODULE m = LoadLibraryA(name);
    if (g_cubin && isNvapiA(name)) { m = nvapiModule(m); logf("DLSS-G LoadLibraryA(%s) -> %p", name, (void*)m); }
    return m;
}

// redirects MODULE's bound imports of kernel32!GetProcAddress (and LoadLibrary* with the cubin executor)
bool wrapGetProcAddress(HMODULE module)
{
    auto base = reinterpret_cast<BYTE*>(module);
    auto dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    IMAGE_DATA_DIRECTORY dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) return false;
    const struct { const char* name; void* hook; bool always; } hooks[] = {
        {"GetProcAddress", reinterpret_cast<void*>(fgGetProcAddress), true},
        {"LoadLibraryW", reinterpret_cast<void*>(fgLoadLibraryW), false},
        {"LoadLibraryExW", reinterpret_cast<void*>(fgLoadLibraryExW), false},
        {"LoadLibraryA", reinterpret_cast<void*>(fgLoadLibraryA), false},
        {"LoadLibraryExA", reinterpret_cast<void*>(fgLoadLibraryExA), false}};
    bool wrapped = false;
    for (auto d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); d->Name; d++) {
        if (_stricmp(reinterpret_cast<char*>(base + d->Name), "kernel32.dll") != 0 || !d->OriginalFirstThunk) continue;
        auto names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + d->OriginalFirstThunk);
        auto slots = reinterpret_cast<IMAGE_THUNK_DATA*>(base + d->FirstThunk);
        for (; names->u1.AddressOfData; names++, slots++) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
            auto ibn = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            for (auto& h : hooks) {
                if (strcmp(reinterpret_cast<char*>(ibn->Name), h.name) != 0 || !(h.always || g_cubin)) continue;
                DWORD old;
                VirtualProtect(&slots->u1.Function, sizeof(void*), PAGE_READWRITE, &old);
                slots->u1.Function = reinterpret_cast<ULONG_PTR>(h.hook);
                VirtualProtect(&slots->u1.Function, sizeof(void*), old, &old);
                wrapped |= h.always;
            }
        }
    }
    return wrapped;
}

FgHandle* fgHandle(const void* h)
{
    std::lock_guard<std::mutex> l(g_lock);
    auto it = g_handles.find(h);
    return it == g_handles.end() ? nullptr : it->second;
}

// Initialise nvngx_dlssg.dll on this device (once). Called after the next core's Init succeeded.
void initFg(unsigned long long appId, const wchar_t* dataPath, void* device, unsigned sdkVersion)
{
    if (!g_fg || g_fgInited) return;
    using T = NgxResult (*)(unsigned long long, const wchar_t*, void*, unsigned, const void*);
    auto init = fn<T>(g_fg, "NVSDK_NGX_D3D12_Init_Ext");
    if (!init) { logf("nvngx_dlssg.dll has no NVSDK_NGX_D3D12_Init_Ext"); return; }
    Param p; p.vtbl = kParamVtbl;
    pm::setU(&p, "Minimum.Logging.Level", g_logLevel);
    pm::setPtr(&p, "Log.Callback", reinterpret_cast<void*>(&snippetLog));
    pm::setU(&p, "Disable.Other.Logging.Sinks", 0);
    if (g_arch) pm::setU(&p, "Context.GPU.ArchID", g_arch);
    if (g_cubin) cux::attach(device);
    cux::FgScope scope;
    NgxResult r = init(appId, dataPath, device, sdkVersion, &p);
    g_fgInited = !failed(r);
    g_device = device;
    logf("dlssg Init_Ext(app %llu, device %p, sdk 0x%x) -> 0x%08X", appId, device, sdkVersion, r);
}

void addFgCaps(void* params)
{
    if (!params || !g_fg) return;
    ms::setI(params, "FrameGeneration.Available", g_fgInited ? 1 : 0);
    ms::setI(params, "FrameGeneration.NeedsUpdatedDriver", 0);
    ms::setI(params, "FrameGeneration.FeatureInitResult", g_fgInited ? 1 : 0);
    ms::setU(params, "FrameGeneration.MinDriverVersionMajor", 0);
    ms::setU(params, "FrameGeneration.MinDriverVersionMinor", 0);
    ms::setI(params, "FrameInterpolation.Available", g_fgInited ? 1 : 0);
    ms::setI(params, "FrameInterpolation.NeedsUpdatedDriver", 0);
    ms::setI(params, "FrameInterpolation.FeatureInitResult", g_fgInited ? 1 : 0);
    ms::setU(params, "FrameInterpolation.MinDriverVersionMajor", 0);
    ms::setU(params, "FrameInterpolation.MinDriverVersionMinor", 0);
    if (!g_fgInited) return;
    using T = NgxResult (*)(void*);
    auto pop = fn<T>(g_fg, "NVSDK_NGX_D3D12_PopulateParameters_Impl");
    NgxResult r = pop ? pop(params) : kFail;
    unsigned mfc = 0, model = 0;
    ms::getU(params, "DLSSG.MultiFrameCountMax", &mfc);
    ms::getU(params, "DLSSG.ModelVersion", &model);
    static bool once = false;
    if (!once || failed(r)) {
        once = true;
        logf("capabilities: dlssg PopulateParameters_Impl -> 0x%08X, MultiFrameCountMax %u, ModelVersion %u", r, mfc,
             model);
    }
}

} // namespace

#define EXPORT extern "C" __declspec(dllexport)

#define NEXT(type, name, ...)                                                                        \
    load();                                                                                          \
    auto f = fn<type>(g_next, name);                                                                 \
    if (!f) { logf("%s: missing in the next core", name); return kFail; }                            \
    return f(__VA_ARGS__)

EXPORT NgxResult NVSDK_NGX_D3D12_Init(unsigned long long appId, const wchar_t* dataPath, void* device,
                                      const void* featureInfo, unsigned int sdkVersion)
{
    using T = NgxResult (*)(unsigned long long, const wchar_t*, void*, const void*, unsigned int);
    load();
    auto f = fn<T>(g_next, "NVSDK_NGX_D3D12_Init");
    NgxResult r = f ? f(appId, dataPath, device, featureInfo, sdkVersion) : g_standalone ? kOk : kFail;
    logf("D3D12_Init -> next 0x%08X", r);
    if (!failed(r)) initFg(appId, dataPath, device, sdkVersion);
    return r;
}

EXPORT NgxResult NVSDK_NGX_D3D12_Init_Ext(unsigned long long appId, const wchar_t* dataPath, void* device,
                                          unsigned int sdkVersion, const void* featureInfo)
{
    using T = NgxResult (*)(unsigned long long, const wchar_t*, void*, unsigned int, const void*);
    load();
    auto f = fn<T>(g_next, "NVSDK_NGX_D3D12_Init_Ext");
    NgxResult r = f ? f(appId, dataPath, device, sdkVersion, featureInfo) : g_standalone ? kOk : kFail;
    logf("D3D12_Init_Ext(app %llu) -> next 0x%08X", appId, r);
    if (!failed(r)) initFg(appId, dataPath, device, sdkVersion);
    return r;
}

EXPORT NgxResult NVSDK_NGX_D3D12_Init_ProjectID(const char* projectId, int engineType, const char* engineVersion,
                                                const wchar_t* dataPath, void* device, unsigned int sdkVersion,
                                                const void* featureInfo)
{
    using T = NgxResult (*)(const char*, int, const char*, const wchar_t*, void*, unsigned int, const void*);
    load();
    auto f = fn<T>(g_next, "NVSDK_NGX_D3D12_Init_ProjectID");
    NgxResult r = f ? f(projectId, engineType, engineVersion, dataPath, device, sdkVersion, featureInfo)
                    : g_standalone ? kOk : kFail;
    logf("D3D12_Init_ProjectID(%s) -> next 0x%08X", projectId ? projectId : "", r);
    if (!failed(r)) initFg(0, dataPath, device, sdkVersion);
    return r;
}

static void shutdownFg(void* device)
{
    if (!g_fgInited) return;
    using T1 = NgxResult (*)(void*);
    using T0 = NgxResult (*)();
    NgxResult r = kFail;
    if (auto s1 = fn<T1>(g_fg, "NVSDK_NGX_D3D12_Shutdown1")) r = s1(device ? device : g_device);
    else if (auto s0 = fn<T0>(g_fg, "NVSDK_NGX_D3D12_Shutdown")) r = s0();
    g_fgInited = false;
    logf("dlssg shutdown -> 0x%08X", r);
}

EXPORT NgxResult NVSDK_NGX_D3D12_Shutdown()
{
    using T = NgxResult (*)();
    load();
    shutdownFg(nullptr);
    if (g_standalone) return kOk;
    NEXT(T, "NVSDK_NGX_D3D12_Shutdown");
}

EXPORT NgxResult NVSDK_NGX_D3D12_Shutdown1(void* device)
{
    using T = NgxResult (*)(void*);
    load();
    shutdownFg(device);
    if (g_standalone) return kOk;
    NEXT(T, "NVSDK_NGX_D3D12_Shutdown1", device);
}

static NgxResult ownParams(void** params, bool caps)
{
    if (!params) return 0xBAD00005u;
    auto* p = new Param;
    p->vtbl = kParamVtbl;
    *params = p;
    if (caps) addFgCaps(p);
    return kOk;
}

EXPORT NgxResult NVSDK_NGX_D3D12_GetParameters(void** params)
{
    using T = NgxResult (*)(void**);
    load();
    if (g_standalone) return ownParams(params, true);
    auto f = fn<T>(g_next, "NVSDK_NGX_D3D12_GetParameters");
    NgxResult r = f ? f(params) : kFail;
    if (!failed(r) && params) addFgCaps(*params);
    return r;
}

EXPORT NgxResult NVSDK_NGX_D3D12_AllocateParameters(void** params)
{
    using T = NgxResult (*)(void**);
    load();
    if (g_standalone) return ownParams(params, false);
    NEXT(T, "NVSDK_NGX_D3D12_AllocateParameters", params);
}

EXPORT NgxResult NVSDK_NGX_D3D12_GetCapabilityParameters(void** params)
{
    using T = NgxResult (*)(void**);
    load();
    if (g_standalone) return ownParams(params, true);
    auto f = fn<T>(g_next, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    NgxResult r = f ? f(params) : kFail;
    if (!failed(r) && params) addFgCaps(*params);
    return r;
}

EXPORT NgxResult NVSDK_NGX_D3D12_DestroyParameters(void* params)
{
    using T = NgxResult (*)(void*);
    load();
    if (g_standalone) { delete static_cast<Param*>(params); return kOk; }
    NEXT(T, "NVSDK_NGX_D3D12_DestroyParameters", params);
}

EXPORT NgxResult NVSDK_NGX_D3D12_PopulateParameters_Impl(void* params)
{
    using T = NgxResult (*)(void*);
    load();
    auto f = fn<T>(g_next, "NVSDK_NGX_D3D12_PopulateParameters_Impl");
    NgxResult r = f ? f(params) : kOk;
    addFgCaps(params);
    return r;
}

EXPORT NgxResult NVSDK_NGX_D3D12_GetScratchBufferSize(unsigned int feature, const void* params, size_t* size)
{
    using T = NgxResult (*)(unsigned int, const void*, size_t*);
    load();
    if (feature == kFeatureFG && g_fgInited) {
        auto f = fn<T>(g_fg, "NVSDK_NGX_D3D12_GetScratchBufferSize");
        return f ? f(feature, params, size) : kFail;
    }
    NEXT(T, "NVSDK_NGX_D3D12_GetScratchBufferSize", feature, params, size);
}

// NVSDK_NGX_FeatureDiscoveryInfo: SDKVersion (4 bytes), then FeatureID.
EXPORT NgxResult NVSDK_NGX_D3D12_GetFeatureRequirements(void* adapter, const void* discovery, void* requirement)
{
    using T = NgxResult (*)(void*, const void*, void*);
    load();
    unsigned int feature = discovery ? reinterpret_cast<const unsigned int*>(discovery)[1] : 0;
    if (feature == kFeatureFG && g_fg) {
        auto f = fn<T>(g_fg, "NVSDK_NGX_D3D12_GetFeatureRequirements");
        NgxResult r = f ? f(adapter, discovery, requirement) : kFail;
        struct Req { int supported; unsigned minArch; char os[255]; };
        auto* q = static_cast<Req*>(requirement);
        if (q && failed(r)) {
            // the snippet asks NVAPI for the adapter (Ada identity here); answer what a Blackwell driver would
            q->supported = 0;
            q->minArch = 0x190;
            strcpy(q->os, "10.0.0");
            logf("GetFeatureRequirements(FG): dlssg 0x%08X -> answered supported", r);
            r = kOk;
        } else if (q) {
            logf("GetFeatureRequirements(FG): dlssg 0x%08X supported %d minArch 0x%x os %s", r, q->supported,
                 q->minArch, q->os);
        }
        return r;
    }
    NEXT(T, "NVSDK_NGX_D3D12_GetFeatureRequirements", adapter, discovery, requirement);
}

EXPORT NgxResult NVSDK_NGX_UpdateFeature(const void* appId, unsigned int feature)
{
    using T = NgxResult (*)(const void*, unsigned int);
    load();
    if (feature == kFeatureFG && g_fg) return kOk;
    NEXT(T, "NVSDK_NGX_UpdateFeature", appId, feature);
}

// Ray Reconstruction (feature 13) camera matrices at creation. In The Witcher 3 with DLSS-G on, Streamline creates the
// denoiser before the game's first camera constants exist, so the parameter block carries no WorldToViewMatrix or
// ViewToClipMatrix yet. d4r checks for them once, on the feature's first evaluate, and without them serves Ray
// Reconstruction from its CUDA backend for the whole session (RX 9070 XT, 1280x720 -> 1440p: about 10 ms more per
// frame). A missing matrix gets an identity placeholder in the block at creation; the host does not touch evaluates,
// and the game's own matrices replace the placeholder once it registers them (The Witcher 3 does on every evaluate).
static const unsigned int kFeatureRR = 13;

static void rrCreateMatrices(void* params)
{
    static float identity[2][16];
    static const char* const names[2] = {"WorldToViewMatrix", "ViewToClipMatrix"};
    if (!g_rrCreateMatrices || !params) return;
    for (int i = 0; i < 2; ++i) {
        void* v = nullptr;
        if (!failed(ms::getPtr(params, names[i], &v)) && v) continue;
        std::memset(identity[i], 0, sizeof identity[i]);
        for (int k = 0; k < 4; ++k) identity[i][k * 5] = 1.0f;
        ms::setPtr(params, names[i], identity[i]);
        logf("CreateFeature(Ray Reconstruction): no %s yet, identity placeholder for creation", names[i]);
    }
}

EXPORT NgxResult NVSDK_NGX_D3D12_CreateFeature(void* list, unsigned int feature, void* params, NgxHandle** handle)
{
    using T = NgxResult (*)(void*, unsigned int, void*, NgxHandle**);
    load();
    if (feature == kFeatureFG && g_fg) {
        if (!g_fgInited) { logf("CreateFeature(FG) before dlssg init"); return kNotInit; }
        if (params && g_lite) {
            // FG Lite (DLSS-G >= 310.9.16): lighter candidate blending, no DL2Net refinement pass. DLSS-G uses it only
            // when UI recomposition is off (measured: with UserInterfaceRecompositionEnabled = 1 at create it silently
            // runs the full model), so Lite=1 also switches UI recomposition off: the HUD is interpolated with the scene.
            ms::setU(params, "DLSSG.LiteCandidateBlendingEnabled", 1);
            ms::setU(params, "DLSSG.UserInterfaceRecompositionEnabled", 0);
            logf("CreateFeature(FG): Lite=1: DLSSG.LiteCandidateBlendingEnabled = 1, UserInterfaceRecompositionEnabled = 0");
        }
        auto f = fn<T>(g_fg, "NVSDK_NGX_D3D12_CreateFeature");
        NgxHandle* inner = nullptr;
        NgxResult r;
        {
            cux::FgScope scope;
            r = f ? f(list, feature, params, &inner) : kFail;
        }
        if (!failed(r) && inner && handle) {
            auto* h = new FgHandle{{0}, inner};
            std::lock_guard<std::mutex> l(g_lock);
            h->pub.Id = g_nextId++;
            g_handles[&h->pub] = h;
            *handle = &h->pub;
        }
        unsigned w = 0, hh = 0;
        if (params) { ms::getU(params, "DLSSG.Width", &w); ms::getU(params, "DLSSG.Height", &hh); }
        logf("CreateFeature(FG %ux%u) -> dlssg 0x%08X, handle id %u (inner id %u)", w, hh, r,
             (handle && *handle) ? (*handle)->Id : 0, inner ? inner->Id : 0);
        return r;
    }
    auto f = fn<T>(g_next, "NVSDK_NGX_D3D12_CreateFeature");
    if (feature == kFeatureRR) rrCreateMatrices(params);
    NgxResult r = f ? f(list, feature, params, handle) : kFail;
    logf("CreateFeature(feature %u) -> next 0x%08X", feature, r);
    return r;
}

static uint64_t g_evals = 0;

// GPU time of every frame-generation evaluate: timestamps around the snippet's work in the caller's command list,
// resolved into a readback buffer and read back ~100 evaluates later. Logged as medians per MultiFrameIndex.
struct GpuTimer {
    static const unsigned kSlots = 512;
    bool tried = false, ok = false;
    ID3D12QueryHeap* heap = nullptr;
    ID3D12Resource* rb = nullptr;
    double ms_per_tick = 0;
    struct Slot { uint64_t eval = 0; unsigned idx = 0, mfc = 0; };
    Slot slots[kSlots];
    uint64_t readUpTo = 0;
    std::vector<double> samples[8];
    uint64_t lastReport = 0;
    uint64_t realFrames = 0;      // evaluates with MultiFrameIndex 1 since the last report
    ULONGLONG lastReportMs = 0;

    void init(ID3D12Device* dev)
    {
        tried = true;
        D3D12_QUERY_HEAP_DESC qd{};
        qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qd.Count = 2 * kSlots;
        if (FAILED(dev->CreateQueryHeap(&qd, IID_ID3D12QueryHeap, (void**)&heap))) return;
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = 16 * kSlots;
        d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.Format = DXGI_FORMAT_UNKNOWN;
        d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_ID3D12Resource, (void**)&rb))) return;
        ID3D12CommandQueue* q = nullptr;
        D3D12_COMMAND_QUEUE_DESC cq{};
        cq.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        UINT64 freq = 0;
        if (SUCCEEDED(dev->CreateCommandQueue(&cq, IID_ID3D12CommandQueue, (void**)&q))) {
            q->GetTimestampFrequency(&freq);
            q->Release();
        }
        if (!freq) return;
        ms_per_tick = 1000.0 / (double)freq;
        ok = true;
    }
    void before(ID3D12GraphicsCommandList* cl, uint64_t n)
    {
        cl->EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * (unsigned)(n % kSlots));
    }
    void after(ID3D12GraphicsCommandList* cl, uint64_t n, unsigned idx, unsigned mfc)
    {
        unsigned s = (unsigned)(n % kSlots);
        cl->EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * s + 1);
        cl->ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * s, 2, rb, 16 * (UINT64)s);
        slots[s] = {n, idx, mfc};
        if (idx <= 1) realFrames++;
        collect(n);
    }
    void collect(uint64_t n)
    {
        // results of evaluates older than 128 are certainly on the GPU by now
        if (n < readUpTo + 192) return;
        uint64_t upto = n - 128;
        D3D12_RANGE r{0, 16 * kSlots};
        uint64_t* m = nullptr;
        if (FAILED(rb->Map(0, &r, (void**)&m)) || !m) return;
        for (uint64_t e = std::max<uint64_t>(readUpTo + 1, n > kSlots ? n - kSlots + 1 : 1); e <= upto; e++) {
            const Slot& sl = slots[e % kSlots];
            if (sl.eval != e) continue;
            uint64_t t0 = m[2 * (e % kSlots)], t1 = m[2 * (e % kSlots) + 1];
            if (t1 > t0 && sl.idx < 8) samples[sl.idx].push_back((double)(t1 - t0) * ms_per_tick);
        }
        D3D12_RANGE w{0, 0};
        rb->Unmap(0, &w);
        readUpTo = upto;
        if (n - lastReport >= 2000) report(n);
    }
    void report(uint64_t n)
    {
        lastReport = n;
        char buf[512];
        int o = snprintf(buf, sizeof buf, "GPU ms per evaluate (median/p90 by MultiFrameIndex, last %llu evals):",
                         (unsigned long long)(n - (n >= 2000 ? n - 2000 : 0)));
        double total = 0;
        for (unsigned i = 0; i < 8; i++) {
            auto& v = samples[i];
            if (v.empty()) continue;
            std::sort(v.begin(), v.end());
            double med = v[v.size() / 2], p90 = v[(v.size() * 9) / 10];
            total += med;
            o += snprintf(buf + o, sizeof buf - o, " [%u] %.3f/%.3f (n=%zu)", i, med, p90, v.size());
            v.clear();
        }
        ULONGLONG now = GetTickCount64();
        double secs = lastReportMs ? (now - lastReportMs) / 1000.0 : 0;
        snprintf(buf + o, sizeof buf - o, "; per real frame %.3f ms; %.1f real frames/s", total,
                 secs > 0 ? realFrames / secs : 0.0);
        lastReportMs = now;
        realFrames = 0;
        logf("%s", buf);
    }
};
static GpuTimer g_timer;

// Test captures (DumpDir/DumpAt/DumpReal in dlssg_host.ini): for DumpReal consecutive real frames starting at real
// frame DumpAt, every evaluate's parameters (text) and, at the first evaluate of each real frame, the input resources
// (copied into readback buffers after the evaluate in the same command list, written ~128 evaluates later). Used to
// replay real game frames through the D3D12 harness (--replay). Off unless DumpReal > 0.
struct Dumper {
    struct Pending { uint64_t eval; std::string file; ID3D12Resource* rb; D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp; UINT rows; UINT64 rowBytes; };
    std::vector<Pending> pend;
    uint64_t real = 0, first = 0;
    unsigned done = 0;
    unsigned polls = 0;
    bool active(unsigned idx)
    {
        if (!g_dumpReal || g_dumpDir.empty()) return false;
        if (idx <= 1) real++;
        if (real < g_dumpAt) return false;
        if (!first) {
            // starts at the first real frame after DumpDir\trigger appears (the test script creates it)
            if (idx > 1 || (polls++ & 15)) return false;
            std::wstring t = g_dumpDir + L"\\trigger";
            if (GetFileAttributesW(t.c_str()) == INVALID_FILE_ATTRIBUTES) return false;
            DeleteFileW(t.c_str());
            first = real;
            logf("dump: triggered at real frame %llu", (unsigned long long)real);
        }
        return real < first + g_dumpReal;
    }
    static std::string narrow(const std::wstring& w) { std::string r; for (wchar_t c : w) r += (char)c; return r; }
    void writeFile(const std::string& name, const void* data, size_t n)
    {
        std::wstring p = g_dumpDir + L"\\";
        for (char c : name) p += (wchar_t)c;
        HANDLE f = CreateFileW(p.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) { logf("dump: cannot write %s", name.c_str()); return; }
        DWORD w; WriteFile(f, data, (DWORD)n, &w, nullptr); CloseHandle(f);
    }
    void params(void* p, uint64_t eval, unsigned rf, unsigned idx)
    {
        static const char* fl[] = {"MvecScaleX", "MvecScaleY", "JitterOffsetX", "JitterOffsetY", "CameraPinholeOffsetX",
            "CameraPinholeOffsetY", "CameraPosX", "CameraPosY", "CameraPosZ", "CameraUpX", "CameraUpY", "CameraUpZ",
            "CameraRightX", "CameraRightY", "CameraRightZ", "CameraFwdX", "CameraFwdY", "CameraFwdZ", "CameraNear",
            "CameraFar", "CameraFOV", "CameraAspectRatio", "LinearizedDepth_Scale", "LinearizedDepth_NearFarPartition",
            "MinRelativeLinearDepthObjectSeparation", "BidirectionalDistortionFieldLowPrecision.Bias",
            "BidirectionalDistortionFieldLowPrecision.Scale", "MvecInvalidValue", "TargetFrameRate"};
        static const char* ui[] = {"Width", "Height", "BackbufferFormat", "ColorBuffersHDR", "DepthInverted",
            "CameraMotionIncluded", "Reset", "AutomodeOverrideReset", "NotRenderingGameFrames", "OrthoProjection",
            "InvertXAxis", "InvertYAxis", "MvecDilated", "MvecJittered", "MenuDetectionEnabled", "MultiFrameCount",
            "MultiFrameIndex", "EvalFlags", "FullscreenMode", "DynamicResolution", "InternalWidth", "InternalHeight",
            "UserInterfaceRecompositionEnabled", "ResourceAlwaysProvidedFlags", "ResourceNeverProvidedFlags",
            "BidirectionalDistortionFieldLowPrecision.IsLowPrecision", "OutputDisableInterpolation",
            "InputBackbufferSubrectBaseX", "InputBackbufferSubrectBaseY", "InputBackbufferSubrectWidth",
            "InputBackbufferSubrectHeight", "MVecsSubrectBaseX", "MVecsSubrectBaseY", "MVecsSubrectWidth",
            "MVecsSubrectHeight", "DepthSubrectBaseX", "DepthSubrectBaseY", "DepthSubrectWidth", "DepthSubrectHeight",
            "HUDLessSubrectBaseX", "HUDLessSubrectBaseY", "HUDLessSubrectWidth", "HUDLessSubrectHeight",
            "UISubrectBaseX", "UISubrectBaseY", "UISubrectWidth", "UISubrectHeight", "UIAlphaSubrectBaseX",
            "UIAlphaSubrectBaseY", "UIAlphaSubrectWidth", "UIAlphaSubrectHeight", "OutputInterpolatedSubrectBaseX",
            "OutputInterpolatedSubrectBaseY", "OutputInterpolatedSubrectWidth", "OutputInterpolatedSubrectHeight",
            "OutputRealSubrectBaseX", "OutputRealSubrectBaseY", "OutputRealSubrectWidth", "OutputRealSubrectHeight",
            "BackbufferSubrectBaseX", "BackbufferSubrectBaseY", "BackbufferSubrectWidth", "BackbufferSubrectHeight"};
        static const char* mats[] = {"CameraViewToClip", "ClipToCameraView", "ClipToLensClip", "ClipToPrevClip", "PrevClipToClip"};
        static const char* res[] = {"Backbuffer", "HUDLess", "Depth", "MVecs", "UI", "UIAlpha",
            "BidirectionalDistortionField", "OutputInterpolated", "OutputReal"};
        std::string t;
        char b[512];
        snprintf(b, sizeof b, "eval %llu real %u index %u\n", (unsigned long long)eval, rf, idx); t += b;
        for (auto n : fl) {
            float v; std::string k = std::string("DLSSG.") + n;
            if (!failed(ms::getF(p, k.c_str(), &v))) { snprintf(b, sizeof b, "f %s %.9g\n", k.c_str(), v); t += b; }
        }
        for (auto n : ui) {
            unsigned v; std::string k = std::string("DLSSG.") + n;
            if (!failed(ms::getU(p, k.c_str(), &v))) { snprintf(b, sizeof b, "u %s %u\n", k.c_str(), v); t += b; }
        }
        unsigned long long id = 0;
        if (!failed(ms::getULL(p, "DLSSG.BackbufferFrameID", &id))) { snprintf(b, sizeof b, "ull DLSSG.BackbufferFrameID %llu\n", id); t += b; }
        for (auto n : mats) {
            void* m = nullptr; std::string k = std::string("DLSSG.") + n;
            if (!failed(ms::getPtr(p, k.c_str(), &m)) && m) {
                const float* f = (const float*)m;
                int o = snprintf(b, sizeof b, "m %s", k.c_str());
                for (int i = 0; i < 16; i++) o += snprintf(b + o, sizeof b - o, " %.9g", f[i]);
                snprintf(b + o, sizeof b - o, "\n"); t += b;
            }
        }
        for (auto n : res) {
            ID3D12Resource* r = nullptr; std::string k = std::string("DLSSG.") + n;
            if (!failed(ms::getD3D12(p, k.c_str(), &r)) && r) {
                D3D12_RESOURCE_DESC d = r->GetDesc();
                snprintf(b, sizeof b, "r %s %p %llux%u fmt %u flags 0x%x\n", k.c_str(), (void*)r, (unsigned long long)d.Width,
                         d.Height, (unsigned)d.Format, (unsigned)d.Flags);
                t += b;
            }
        }
        snprintf(b, sizeof b, "params_r%02u_e%u.txt", rf, idx);
        writeFile(b, t.data(), t.size());
    }
    void resources(ID3D12GraphicsCommandList* cl, void* p, uint64_t eval, unsigned rf)
    {
        static const char* res[] = {"Backbuffer", "HUDLess", "Depth", "MVecs", "UI", "UIAlpha", "BidirectionalDistortionField"};
        auto* dev = static_cast<ID3D12Device*>(g_device);
        for (auto n : res) {
            ID3D12Resource* r = nullptr; std::string k = std::string("DLSSG.") + n;
            if (failed(ms::getD3D12(p, k.c_str(), &r)) || !r) continue;
            D3D12_RESOURCE_DESC d = r->GetDesc();
            Pending pe{}; UINT64 total = 0;
            dev->GetCopyableFootprints(&d, 0, 1, 0, &pe.fp, &pe.rows, &pe.rowBytes, &total);
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC bd{}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1;
            bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.Format = DXGI_FORMAT_UNKNOWN; bd.SampleDesc.Count = 1;
            bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_ID3D12Resource, (void**)&pe.rb))) { logf("dump: readback alloc failed"); continue; }
            D3D12_RESOURCE_BARRIER br{}; br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            br.Transition.pResource = r; br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            br.Transition.StateBefore = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
            br.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            cl->ResourceBarrier(1, &br);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            dst.pResource = pe.rb; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = pe.fp;
            src.pResource = r; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            std::swap(br.Transition.StateBefore, br.Transition.StateAfter);
            cl->ResourceBarrier(1, &br);
            char b[256];
            snprintf(b, sizeof b, "res_r%02u_%s_%llux%u_fmt%u_pitch%u.bin", rf, n, (unsigned long long)d.Width, d.Height,
                     (unsigned)d.Format, pe.fp.Footprint.RowPitch);
            pe.file = b; pe.eval = eval;
            pend.push_back(pe);
        }
    }
    void flush(uint64_t eval, bool all = false)
    {
        for (size_t i = 0; i < pend.size();) {
            Pending& pe = pend[i];
            if (!all && eval < pe.eval + 128) { i++; continue; }
            void* m = nullptr; D3D12_RANGE rr{0, (SIZE_T)(pe.fp.Footprint.RowPitch * pe.rows)};
            if (SUCCEEDED(pe.rb->Map(0, &rr, &m)) && m) {
                writeFile(pe.file, m, (size_t)pe.fp.Footprint.RowPitch * pe.rows);
                D3D12_RANGE w{0, 0}; pe.rb->Unmap(0, &w);
                logf("dump: wrote %s", pe.file.c_str());
            }
            pe.rb->Release();
            pend.erase(pend.begin() + i);
        }
    }
};
static Dumper g_dump;

static NgxResult evalFg(const char* name, void* list, FgHandle* h, void* params, void* cb)
{
    using T = NgxResult (*)(void*, const NgxHandle*, void*, void*);
    auto f = fn<T>(g_fg, name);
    if (!f) f = fn<T>(g_fg, "NVSDK_NGX_D3D12_EvaluateFeature");
    uint64_t n = ++g_evals;
    unsigned mfc = 0, idx = 0;
    if (params) { ms::getU(params, "DLSSG.MultiFrameCount", &mfc); ms::getU(params, "DLSSG.MultiFrameIndex", &idx); }
    auto* cl = static_cast<ID3D12GraphicsCommandList*>(list);
    bool timed = false;
    if (g_timing && cl && g_device) {
        std::lock_guard<std::mutex> l(g_lock);
        if (!g_timer.tried) g_timer.init(static_cast<ID3D12Device*>(g_device));
        timed = g_timer.ok;
        if (timed) g_timer.before(cl, n);
    }
    NgxResult r;
    {
        cux::FgScope scope(g_cubin ? list : nullptr);
        r = f ? f(list, h->inner, params, cb) : kFail;
    }
    if (timed) {
        std::lock_guard<std::mutex> l(g_lock);
        g_timer.after(cl, n, idx, mfc);
    }
    if (g_dumpReal && params && cl && g_device) {
        std::lock_guard<std::mutex> l(g_lock);
        if (g_dump.active(idx)) {
            unsigned rf = (unsigned)(g_dump.real - g_dump.first);
            g_dump.params(params, n, rf, idx);
            if (idx <= 1) g_dump.resources(cl, params, n, rf);
        }
        g_dump.flush(n);
    }
    if (failed(r) || n <= 3 || (n % 10000) == 0)
        logf("Evaluate(FG) #%llu (%u/%u) -> 0x%08X%s%s", (unsigned long long)n, idx, mfc, r, g_cubin ? "; " : "",
             g_cubin ? cux::summary().c_str() : "");
    return r;
}

EXPORT NgxResult NVSDK_NGX_D3D12_EvaluateFeature(void* list, const NgxHandle* handle, void* params, void* cb)
{
    using T = NgxResult (*)(void*, const NgxHandle*, void*, void*);
    load();
    if (FgHandle* h = fgHandle(handle)) return evalFg("NVSDK_NGX_D3D12_EvaluateFeature", list, h, params, cb);
    auto f = fn<T>(g_next, "NVSDK_NGX_D3D12_EvaluateFeature");
    return f ? f(list, handle, params, cb) : kFail;
}

EXPORT NgxResult NVSDK_NGX_D3D12_EvaluateFeature_C(void* list, const NgxHandle* handle, void* params, void* cb)
{
    using T = NgxResult (*)(void*, const NgxHandle*, void*, void*);
    load();
    if (FgHandle* h = fgHandle(handle)) return evalFg("NVSDK_NGX_D3D12_EvaluateFeature_C", list, h, params, cb);
    auto f = fn<T>(g_next, "NVSDK_NGX_D3D12_EvaluateFeature_C");
    if (!f) f = fn<T>(g_next, "NVSDK_NGX_D3D12_EvaluateFeature");
    return f ? f(list, handle, params, cb) : kFail;
}

EXPORT NgxResult NVSDK_NGX_D3D12_ReleaseFeature(NgxHandle* handle)
{
    using T = NgxResult (*)(NgxHandle*);
    load();
    if (FgHandle* h = fgHandle(handle)) {
        {
            std::lock_guard<std::mutex> l(g_lock);
            g_handles.erase(handle);
            // pending test captures: the feature is released, its frames are done (the harness waits per frame)
            g_dump.flush(g_evals, true);
        }
        auto f = fn<T>(g_fg, "NVSDK_NGX_D3D12_ReleaseFeature");
        NgxResult r = f ? f(h->inner) : kFail;
        logf("ReleaseFeature(FG id %u) -> dlssg 0x%08X", h->pub.Id, r);
        delete h;
        return r;
    }
    NEXT(T, "NVSDK_NGX_D3D12_ReleaseFeature", handle);
}

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
