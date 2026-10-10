// Experimental native Vulkan NGX frontend: d4r_vulkan_frontend.inl.
// d4r_nvngx.dll: an NGX core replacement that implements the D3D12 DLSS
// entry points on top of the official NGX core's CUDA path (which runs on AMD
// through the Wine nvcuda bridge and ZLUDA). OptiScaler loads it through its
// NvngxPath setting.
//
// A D3D12 EvaluateFeature call only records into the game's still-open command
// list, while DLSS runs on a separate CUDA context. Each evaluation therefore:
//   1. records copies of the inputs into readback buffers plus a GPU-written
//      frame marker (WriteBufferImmediate) on the game's command list,
//   2. records a copy of the most recent finished DLSS result from an upload
//      buffer into the output texture,
//   3. queues a CUDA job that waits for the marker, uploads the inputs into
//      CUDA arrays, runs the official DLSS evaluation, and stores the result
//      into an upload buffer for a later frame.
// The output is therefore one or more frames behind the game's other passes.
//
// Environment:
//   D4R_NGX_CORE        Windows path of the official _nvngx.dll (required)
//   D4R_NGX_FEATURE_DIR directory holding nvngx_dlss.dll (optional, added to
//                       the feature search paths)
//   D4R_SHIM_LOG        log file path (default: d4r_nvngx.log next to the DLL)
//   D4R_PROFILE=1       log per-frame CPU stages and default-stream GPU event time
//   D4R_SHIM_OUTPUT_HASH=1 log an FNV-1a hash of every DLSS output as produced
//   D4R_SHIM_OUTPUT_DUMP_DIR=<Windows path> save selected raw RGBA16F DLSS frames
//   D4R_SHIM_OUTPUT_DUMP_START / _COUNT / _EVERY select frames (defaults 1/1/1)
//   D4R_SHIM_INPUT_DUMP_DIR=<Windows path> save selected canonical input planes
//                       (use D4R_SHIM_VRAM_VERIFY=1 with VRAM interop)
//   D4R_SHIM_CAPTURE_TRIGGER=<Windows path> select the next frame when this file appears
//                       (D4R_SHIM_CAPTURE_COUNT=N: that frame and the N-1 after it;
//                       D4R_SHIM_CAPTURE_MEMORY=1 buffers them in memory, written afterwards)
//   D4R_SHIM_INPUT_STATE / D4R_SHIM_DEPTH_STATE / D4R_SHIM_OUTPUT_STATE
//                       D3D12 resource states the inputs/output are in when
//                       EvaluateFeature is called (defaults 0x40, 0x40, 0x8)
//   D4R_SHIM_VRAM_INTEROP=1 keep inputs and output in VRAM (see "VRAM interop")
//   D4R_SHIM_SPLIT_FRAME=1 with VRAM interop and the d4r vkd3d-proton patch,
//                       present each frame's own DLSS result (see "Split frames")
//   D4R_SHIM_INPLACE=1  experimental, with VKD3D_D4R_LINEAR_TEXTURES=1 in the patched vkd3d-proton: the output
//                       kernel writes the game's output texture itself (no game-side output copy).
//                       D4R_SHIM_OUTPUT_PACKED=1 lets M's output kernel store R10G10B10A2 texels itself.
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#include <windows.h>
#include <d3d12.h>
#include <vulkan/vulkan_core.h>
#include "d4r_motion_dilation.h"

#include <algorithm>
#include <bit>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "d4r_event_wait.h"
#include "d4r_frame_completion.h"
#include "d4r_win32_wait.h"
#include "d4r_vkd3d_interop.h"
#include "../engine/game.h"
#include "../engine/game_m.h"
#include "../engine/vk_dispatch.h"

// --- NGX types (declared locally: the core's exports differ from the SDK) --

using NgxResult = unsigned int;
constexpr NgxResult NGX_SUCCESS = 0x1;
constexpr NgxResult NGX_FAIL = 0xBAD00000;
constexpr NgxResult NGX_FAIL_FEATURE_NOT_SUPPORTED = NGX_FAIL | 1;
constexpr NgxResult NGX_FAIL_PLATFORM_ERROR = NGX_FAIL | 2;
constexpr NgxResult NGX_FAIL_INVALID_PARAMETER = NGX_FAIL | 5;
constexpr NgxResult NGX_FAIL_NOT_INITIALIZED = NGX_FAIL | 7;
constexpr NgxResult NGX_FAIL_UNSUPPORTED_FORMAT = NGX_FAIL | 14;
constexpr unsigned int NGX_FEATURE_SUPER_SAMPLING = 1;

struct NgxHandle
{
    unsigned int Id;
};

struct NgxPathListInfo
{
    const wchar_t* const* Path;
    unsigned int Length;
};

struct NgxLoggingInfo
{
    void* LoggingCallback;
    int MinimumLoggingLevel;
    bool DisableOtherLoggingSinks;
};

struct NgxFeatureCommonInfo
{
    NgxPathListInfo PathListInfo;
    void* InternalData;
    NgxLoggingInfo LoggingInfo;
};

struct NgxFeatureRequirement
{
    unsigned int FeatureSupported;
    unsigned int MinHWArchitecture;
    char MinOSVersion[255];
};

// MSVC-ABI parameter accessors (d4r_ngx_param_msvc.cpp, built with clang-cl).
extern "C"
{
    void d4r_ngx_set_ull(void* parameters, const char* name, unsigned long long value);
    void d4r_ngx_set_float(void* parameters, const char* name, float value);
    void d4r_ngx_set_uint(void* parameters, const char* name, unsigned int value);
    void d4r_ngx_set_int(void* parameters, const char* name, int value);
    void d4r_ngx_set_void(void* parameters, const char* name, void* value);
    NgxResult d4r_ngx_get_ull(void* parameters, const char* name, unsigned long long* value);
    NgxResult d4r_ngx_get_float(void* parameters, const char* name, float* value);
    NgxResult d4r_ngx_get_uint(void* parameters, const char* name, unsigned int* value);
    NgxResult d4r_ngx_get_int(void* parameters, const char* name, int* value);
    NgxResult d4r_ngx_get_void(void* parameters, const char* name, void** value);
    NgxResult d4r_ngx_get_d3d12_resource(void* parameters, const char* name, ID3D12Resource** value);
    void* d4r_ngx_parameters_create();
    void d4r_ngx_parameters_destroy(void* parameters);
}

// --- CUDA driver API (through nvcuda.dll, the Wine bridge to ZLUDA) --------

using CudaDevicePtr = unsigned long long;
using CudaArray = void*;
using CudaObject = unsigned long long;
using CudaEvent = void*;

struct CudaArrayDescriptor
{
    size_t Width;
    size_t Height;
    uint32_t Format;
    uint32_t NumChannels;
};

struct CudaArray3DDescriptor
{
    size_t Width;
    size_t Height;
    size_t Depth;
    uint32_t Format;
    uint32_t NumChannels;
    uint32_t Flags;
};

struct CudaMemcpy2D
{
    size_t srcXInBytes;
    size_t srcY;
    uint32_t srcMemoryType;
    uint32_t srcAlignment;
    const void* srcHost;
    CudaDevicePtr srcDevice;
    CudaArray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    uint32_t dstMemoryType;
    uint32_t dstAlignment;
    void* dstHost;
    CudaDevicePtr dstDevice;
    CudaArray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};
static_assert(sizeof(CudaMemcpy2D) == 128);

struct CudaResourceDesc
{
    uint32_t resType;
    uint32_t alignment;
    union
    {
        struct
        {
            CudaArray hArray;
        } array;
        int reserved[32];
    } res;
    uint32_t flags;
    uint32_t reserved;
};
static_assert(sizeof(CudaResourceDesc) == 144);

struct CudaTextureDesc
{
    uint32_t addressMode[3];
    uint32_t filterMode;
    uint32_t flags;
    uint32_t maxAnisotropy;
    uint32_t mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int32_t reserved[12];
};
static_assert(sizeof(CudaTextureDesc) == 104);

constexpr uint32_t CUDA_FORMAT_HALF = 0x10;
constexpr uint32_t CUDA_FORMAT_FLOAT = 0x20;
constexpr uint32_t CUDA_MEMORY_HOST = 1;
constexpr uint32_t CUDA_MEMORY_DEVICE = 2;
constexpr uint32_t CUDA_MEMORY_ARRAY = 3;

struct CudaApi
{
    int(WINAPI* memAlloc)(CudaDevicePtr*, size_t) = nullptr;
    int(WINAPI* memFree)(CudaDevicePtr) = nullptr;
    int(WINAPI* arrayCreate)(CudaArray*, const CudaArrayDescriptor*) = nullptr;
    int(WINAPI* array3DCreate)(CudaArray*, const CudaArray3DDescriptor*) = nullptr;
    int(WINAPI* arrayDestroy)(CudaArray) = nullptr;
    int(WINAPI* memcpy2D)(const CudaMemcpy2D*) = nullptr;
    int(WINAPI* texObjectCreate)(CudaObject*, const CudaResourceDesc*, const CudaTextureDesc*, const void*) = nullptr;
    int(WINAPI* texObjectDestroy)(CudaObject) = nullptr;
    int(WINAPI* surfObjectCreate)(CudaObject*, const CudaResourceDesc*) = nullptr;
    int(WINAPI* surfObjectDestroy)(CudaObject) = nullptr;
    int(WINAPI* ctxSynchronize)() = nullptr;
    int(WINAPI* eventCreate)(CudaEvent*, unsigned int) = nullptr;
    int(WINAPI* eventRecord)(CudaEvent, void*) = nullptr;
    int(WINAPI* eventQuery)(CudaEvent) = nullptr;
    int(WINAPI* eventSynchronize)(CudaEvent) = nullptr;
    int(WINAPI* eventElapsedTime)(float*, CudaEvent, CudaEvent) = nullptr;
    int(WINAPI* eventDestroy)(CudaEvent) = nullptr;
    // Optional: overlapped staging (see async_copies()).
    int(WINAPI* memAllocHost)(void**, size_t) = nullptr;
    int(WINAPI* memFreeHost)(void*) = nullptr;
    int(WINAPI* memcpyHtoDAsync)(CudaDevicePtr, const void*, size_t, void*) = nullptr;
    int(WINAPI* memcpyDtoHAsync)(void*, CudaDevicePtr, size_t, void*) = nullptr;
    int(WINAPI* streamCreate)(void**, unsigned int) = nullptr;
    int(WINAPI* streamSynchronize)(void*) = nullptr;
    // Optional (d4r nvcuda bridge): device <-> array copies queued on the null stream.
    int(WINAPI* memcpy2DAsync)(const CudaMemcpy2D*, void*) = nullptr;
    // Optional (d4r nvcuda bridge): null-stream wait for a device u32, and a write from another stream.
    int(WINAPI* streamWaitValue32)(CudaDevicePtr, uint32_t) = nullptr;
    int(WINAPI* writeValue32)(CudaDevicePtr, uint32_t) = nullptr;
    // Optional (d4r nvcuda bridge): surfaces of an array store to linear memory instead (d4r native kernels).
    int(WINAPI* setArrayRedirect)(CudaArray, CudaDevicePtr, uint32_t) = nullptr;
    int(WINAPI* outputKernelNative)() = nullptr;
    // Optional (d4r nvcuda bridge): report a pitch-linear texture to NGX as an array of that size/format.
    int(WINAPI* registerLinearTexture)(CudaObject, size_t, size_t, uint32_t, uint32_t) = nullptr;
    int(WINAPI* dilateMotion)(const D4rMotionDilationParams*) = nullptr;
};

// --- official NGX core CUDA API --------------------------------------------

struct CoreApi
{
    NgxResult (*init)(unsigned long long, const wchar_t*, const NgxFeatureCommonInfo*, unsigned int) = nullptr;
    NgxResult (*initProjectId)(const char*, int, const char*, const wchar_t*, unsigned int,
                               const NgxFeatureCommonInfo*) = nullptr;
    NgxResult (*shutdown)() = nullptr;
    NgxResult (*getParameters)(void**) = nullptr;
    NgxResult (*allocateParameters)(void**) = nullptr;
    NgxResult (*getCapabilityParameters)(void**) = nullptr;
    NgxResult (*destroyParameters)(void*) = nullptr;
    NgxResult (*createFeature)(unsigned int, void*, NgxHandle**) = nullptr;
    NgxResult (*releaseFeature)(NgxHandle*) = nullptr;
    NgxResult (*evaluateFeature)(const NgxHandle*, void*, void*) = nullptr;
};

// --- logging -----------------------------------------------------------------

static std::mutex g_logMutex;
static FILE* g_log = nullptr;
static HMODULE g_selfModule = nullptr;

// --- portable install ----------------------------------------------------------
// The drag-in release keeps this DLL (as nvngx.dll, explicitly selected by OptiScaler's NvngxPath),
// the nvcuda bridge, ZLUDA and the native kernels in an d4r folder next to the game's executable,
// with that game's settings in d4r\d4r.ini. The file's settings become the environment variables the
// developer launcher sets (config/d4r.ini.default documents them), and a variable that is already set,
// for example in the game's launch options, wins. Settings for the Linux side (ZLUDA, the bridge) are
// handed to the bridge right after it is loaded, before its first CUDA call. Without an d4r.ini next
// to the DLL nothing changes.

struct PortableInstall
{
    bool active = false;
    std::wstring dir;                                         // the d4r folder
    std::string unixDir;                                      // the same folder as a Linux path
    std::vector<std::pair<std::string, std::string>> unixEnv; // for the bridge's d4rSetEnv
    std::vector<std::string> notes;                           // logged once the log is open
};
static PortableInstall g_portable;
static std::once_flag g_portableOnce;
static bool g_logFresh = false; // a portable install starts a new log for every launch

struct IniEntry
{
    std::string section, key, value;
};

static std::string ascii_lower(std::string text)
{
    for (char& c : text)
        c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return text;
}

static std::string trim(const std::string& text)
{
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return std::string();
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

// INI in the configparser dialect d4r_config.py reads: [Section], key = value, full-line comments
// with ; or #, inline comments after whitespace.
static std::vector<IniEntry> parse_ini(const std::string& text)
{
    std::vector<IniEntry> entries;
    std::string section;
    size_t start = text.compare(0, 3, "\xEF\xBB\xBF") == 0 ? 3 : 0;
    while (start < text.size())
    {
        size_t end = text.find('\n', start);
        if (end == std::string::npos)
            end = text.size();
        const std::string line = trim(text.substr(start, end - start));
        start = end + 1;
        if (line.empty() || line[0] == ';' || line[0] == '#')
            continue;
        if (line[0] == '[')
        {
            section = ascii_lower(trim(line.substr(1, line.find(']') - 1)));
            continue;
        }
        const size_t equals = line.find('=');
        if (equals == std::string::npos)
            continue;
        std::string value = line.substr(equals + 1);
        for (size_t i = 1; i < value.size(); ++i)
            if ((value[i] == ';' || value[i] == '#') && (value[i - 1] == ' ' || value[i - 1] == '\t'))
            {
                value.resize(i);
                break;
            }
        entries.push_back({section, trim(line.substr(0, equals)), trim(value)});
    }
    return entries;
}

// The last value for [section] key; "" when it is missing, empty or "auto".
static std::string ini_value(const std::vector<IniEntry>& ini, const char* section, const char* key)
{
    std::string result;
    for (const IniEntry& entry : ini)
        if (entry.section == section && ascii_lower(entry.key) == ascii_lower(key))
            result = entry.value;
    return ascii_lower(result) == "auto" ? std::string() : result;
}

// 1 or 0 for a boolean setting, fallback when it is unset or not a boolean.
static int ini_flag(const std::vector<IniEntry>& ini, const char* section, const char* key, int fallback)
{
    const std::string value = ascii_lower(ini_value(ini, section, key));
    if (value.empty())
        return fallback;
    if (value == "1" || value == "true" || value == "yes" || value == "on")
        return 1;
    if (value == "0" || value == "false" || value == "no" || value == "off")
        return 0;
    g_portable.notes.push_back("d4r.ini: [" + std::string(section) + "] " + key + " must be true or false, not '" +
                               value + "'; using " + (fallback ? "true" : "false"));
    return fallback;
}

static bool read_whole_file(const std::wstring& path, std::string& text)
{
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return false;
    LARGE_INTEGER size = {};
    bool ok = GetFileSizeEx(file, &size) && size.QuadPart < (1 << 20);
    if (ok)
    {
        text.resize(static_cast<size_t>(size.QuadPart));
        DWORD read = 0;
        ok = text.empty() || (ReadFile(file, text.data(), static_cast<DWORD>(text.size()), &read, nullptr) &&
                              read == text.size());
    }
    CloseHandle(file);
    return ok;
}

static bool file_exists(const std::wstring& path)
{
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

// Wine's kernel32 maps a Windows path to the Linux one.
static std::string unix_path(const std::wstring& path)
{
    using UnixFileNameFn = char*(CDECL*)(const wchar_t*);
    const auto function = reinterpret_cast<UnixFileNameFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "wine_get_unix_file_name")));
    char* result = function != nullptr ? function(path.c_str()) : nullptr;
    if (result == nullptr)
        return std::string();
    std::string text(result);
    HeapFree(GetProcessHeap(), 0, result);
    return text;
}

static std::wstring widen(const std::string& text)
{
    std::wstring wide(text.size() + 1, L'\0');
    const int length = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), static_cast<int>(wide.size()));
    wide.resize(length > 0 ? static_cast<size_t>(length - 1) : 0);
    return wide;
}

// A Windows path from d4r.ini: absolute (C:\..., \\...) or relative to the d4r folder.
static std::wstring portable_path(const std::string& value)
{
    const std::wstring path = widen(value);
    if ((path.size() > 2 && path[1] == L':') || (path.size() > 1 && path[0] == L'\\' && path[1] == L'\\'))
        return path;
    return g_portable.dir + L"\\" + path;
}

static void portable_set(const char* name, const std::string& value)
{
    if (GetEnvironmentVariableA(name, nullptr, 0) == 0)
        SetEnvironmentVariableA(name, value.c_str());
}

static void portable_set(const wchar_t* name, const std::wstring& value)
{
    if (GetEnvironmentVariableW(name, nullptr, 0) == 0)
        SetEnvironmentVariableW(name, value.c_str());
}

static void portable_set_unix(const char* name, const std::string& value)
{
    g_portable.unixEnv.emplace_back(name, value);
}

// [DLSS] Model: a preset letter, a friendly name or a render-preset number; "" leaves the choice to the game.
static std::string preset_number(const std::string& model)
{
    std::string name;
    for (char c : model)
        name += static_cast<char>(toupper(static_cast<unsigned char>(c)));
    if (name.empty() || name == "GAME" || name == "DEFAULT")
        return std::string();
    if (name.find_first_not_of("0123456789") == std::string::npos)
        return name;
    static const std::pair<const char*, const char*> aliases[] = {
        {"CNN", "E"}, {"DLSS3", "E"}, {"DLSS4", "K"}, {"DLSS4.5", "M"}, {"TRANSFORMER", "K"}};
    for (const auto& [alias, letter] : aliases)
        if (name == alias)
            name = letter;
    static const std::pair<const char*, int> letters[] = {{"A", 1}, {"B", 2}, {"C", 3}, {"D", 4},  {"E", 5},  {"F", 6},
                                                          {"G", 7}, {"J", 10}, {"K", 11}, {"L", 12}, {"M", 13}};
    for (const auto& [letter, number] : letters)
        if (name == letter)
            return std::to_string(number);
    g_portable.notes.push_back("d4r.ini: [DLSS] Model '" + model + "' is not a preset (use K, E, L, M or a number); "
                               "leaving the preset to the game");
    return std::string();
}

static void load_portable_config()
{
    wchar_t module[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(g_selfModule, module, MAX_PATH);
    if (length == 0 || length >= MAX_PATH)
        return;
    std::wstring dir(module, length);
    dir.resize(dir.find_last_of(L"\\/"));
    g_portable.dir = dir; // also without d4r.ini: the default log goes here
    std::string text;
    if (!read_whole_file(dir + L"\\d4r.ini", text))
        return; // developer layout: the launcher's environment has everything
    const std::vector<IniEntry> ini = parse_ini(text);
    g_portable.active = true;
    g_portable.unixDir = unix_path(dir);
    if (g_portable.unixDir.empty())
        g_portable.notes.push_back("cannot map the d4r folder to a Linux path (not running under Wine?)");

    // Files of the install; NGX looks for nvngx_dlss.dll next to this DLL.
    portable_set(L"D4R_NVCUDA_BRIDGE", dir + L"\\nvcuda.dll");
    const std::string core = ini_value(ini, "paths", "NgxCore");
    portable_set(L"D4R_NGX_CORE", core.empty() ? dir + L"\\ngx\\_nvngx.dll" : portable_path(core));
    portable_set(L"D4R_NGX_FEATURE_DIR", dir);
    const std::string log = ini_value(ini, "debug", "Log");
    g_logFresh = log.empty();
    portable_set(L"D4R_SHIM_LOG", log.empty() ? dir + L"\\d4r_nvngx.log" : portable_path(log));
    portable_set("D4R_CUDA_CAPTURE", "0"); // the bridge would otherwise save every DLSS module it loads

    const std::string zluda = ini_value(ini, "paths", "ZludaDir");
    portable_set_unix("D4R_ZLUDA_LIBCUDA", (zluda.empty() ? g_portable.unixDir + "/zluda" : zluda) + "/libcuda.so");
    // The release's bundled ROCm runtime unless d4r.ini names another
    const std::string rocm = ini_value(ini, "paths", "RocmDir");
    if (!rocm.empty())
        portable_set_unix("D4R_ROCM_DIR", rocm);
    else if (!g_portable.unixDir.empty() && GetFileAttributesW((dir + L"\\rocm\\lib").c_str()) != INVALID_FILE_ATTRIBUTES)
        portable_set_unix("D4R_ROCM_DIR", g_portable.unixDir + "/rocm");
    const std::string cache = ini_value(ini, "paths", "CacheDir");
    portable_set_unix("D4R_ZLUDA_CACHE_HOME", cache.empty() ? "~/.cache/d4r" : cache);

    std::string model = ini_value(ini, "dlss", "Model");
    if (ascii_lower(model) != "game")
        model = preset_number(model.empty() ? "K" : model);
    else
        model.clear();
    if (!model.empty())
        portable_set("D4R_DLSS_PRESET", model);

    const std::string age = ini_value(ini, "latency", "FrameAge");
    if (age.empty() || age == "0")
    {
        portable_set("D4R_SHIM_SPLIT_FRAME", "1");
        portable_set("D4R_SHIM_MAX_IN_FLIGHT", "3");
    }
    else if (age == "1" || age == "2" || age == "3")
    {
        portable_set("D4R_SHIM_SPLIT_FRAME", "0");
        portable_set("D4R_SHIM_MAX_IN_FLIGHT", age);
    }
    else
        g_portable.notes.push_back("d4r.ini: [Latency] FrameAge must be 0-3, not '" + age + "'");

    // Native kernels: d4r\kernels, each used only when DLSS's PTX for it matches (see the bridge).
    const std::string native = ascii_lower(ini_value(ini, "kernels", "NativeKernels"));
    // The bridge selects accuracy variants and enforces their compiler/sync policy before cuInit.
    portable_set_unix("D4R_PREFER_ACCURACY", ini_flag(ini, "kernels", "PreferAccuracy", 0) ? "1" : "0");
    portable_set_unix("D4R_NATIVE_SWIN_ENCODERS", ini_flag(ini, "kernels", "NativeSwinEncoders", 1) ? "1" : "0");
    const bool nativeOn = native.empty() || native == "on" || native == "true" || native == "1" || native == "fast";
    if (!nativeOn && native != "off" && native != "false" && native != "0")
        g_portable.notes.push_back("d4r.ini: [Kernels] NativeKernels must be on or off, not '" + native + "'");
    if (nativeOn && !g_portable.unixDir.empty())
        portable_set_unix("D4R_ZLUDA_NATIVE_DIR", g_portable.unixDir + "/kernels");
    portable_set_unix("D4R_ZLUDA_WMMA", ini_flag(ini, "kernels", "Wmma", 1) ? "1" : "0");
    portable_set_unix("D4R_ZLUDA_WMMA_FP8", ini_flag(ini, "kernels", "Fp8Wmma", 1) ? "1" : "0");
    // RDNA4's native FP8 WMMA; ZLUDA and the bridge ignore it on other GPUs
    portable_set_unix("D4R_ZLUDA_WMMA_FP8_NATIVE", ini_flag(ini, "kernels", "NativeFp8", 1) ? "1" : "0");
    portable_set_unix("D4R_ZLUDA_IGNORE_DENORMAL", ini_flag(ini, "kernels", "IgnoreDenormals", 1) ? "1" : "0");
    const std::string maxBlock = ini_value(ini, "kernels", "ImplicitMaxBlock");
    portable_set_unix("D4R_ZLUDA_IMPLICIT_MAX_BLOCK", maxBlock.empty() ? "256" : maxBlock);

    static const struct
    {
        const char* key;
        const char* variable;
        int fallback;
    } interop[] = {{"VramInterop", "D4R_SHIM_VRAM_INTEROP", 1}, {"InputSync", "D4R_SHIM_INPUT_SYNC", 1},
                   {"LinearInputs", "D4R_SHIM_LINEAR_INPUTS", 1},
                   {"EvalSync", "D4R_SHIM_EVAL_SYNC", 0}};
    for (const auto& setting : interop)
        portable_set(setting.variable, ini_flag(ini, "interop", setting.key, setting.fallback) ? "1" : "0");
    portable_set_unix("D4R_ELIDE_NGX_SYNC", ini_flag(ini, "interop", "ElideNgxSync", 1) ? "1" : "0");
    const std::string poll = ini_value(ini, "interop", "MarkerPollUs");
    portable_set("D4R_SHIM_MARKER_POLL_US", poll.empty() ? "200" : poll);
    // Direct output takes effect only while NGX runs the native output kernel (the bridge reports it); the
    // release cannot ship that kernel, which is built from NVIDIA's PTX.
    portable_set("D4R_SHIM_OUTPUT_DIRECT", ini_flag(ini, "interop", "DirectOutput", nativeOn) ? "1" : "0");

    // [Engine]: d4r's own Vulkan inference for DLSS 4 (K); the model is looked up in d4r\engine\k
    portable_set("D4R_ENGINE", ini_flag(ini, "engine", "Enabled", 0) ? "1" : "0");
    if (const std::string engineModel = ini_value(ini, "engine", "ModelDir"); !engineModel.empty())
        portable_set(L"D4R_ENGINE_MODEL_DIR", portable_path(engineModel));
    // Network: preset M's network on the native HIP layers ("hip", the default when the model has them: faster) or
    // on the engine's Vulkan shader ("vulkan")
    if (const std::string network = ini_value(ini, "engine", "Network"); network == "hip" || network == "vulkan")
        portable_set("D4R_ENGINE_NETWORK", network);
    else if (!network.empty() && network != "auto")
        g_portable.notes.push_back("[Engine] Network must be auto, hip or vulkan; using auto");

    portable_set("D4R_SHIM_WATERMARK", ini_flag(ini, "dlss", "ShowWatermark", 0) ? "1" : "0");
    const std::string dilation = ini_value(ini, "interop", "MotionVectorDilation");
    if (dilation.empty() || dilation == "0" || dilation == "1" || dilation == "2")
        portable_set("D4R_MOTION_DILATION", dilation.empty() ? "0" : dilation);
    else
        g_portable.notes.push_back("MotionVectorDilation must be 0, 1 or 2; using original motion vectors");
    if (ini_flag(ini, "debug", "Profile", 0))
        portable_set("D4R_PROFILE", "1");
    if (const std::string level = ini_value(ini, "debug", "NgxLogLevel"); !level.empty())
        portable_set("D4R_NGX_LOG_LEVEL", level);

    // [Env]: any other variable, for both sides
    for (const IniEntry& entry : ini)
        if (entry.section == "env" && !entry.key.empty())
        {
            portable_set(entry.key.c_str(), entry.value);
            portable_set_unix(entry.key.c_str(), entry.value);
        }

    if (!file_exists(dir + L"\\nvngx_dlss.dll"))
        g_portable.notes.push_back("nvngx_dlss.dll is missing: copy NVIDIA's DLSS library (310.7 or 310.9 recommended) "
                                   "into the d4r folder");
    wchar_t corePath[MAX_PATH] = {};
    const DWORD coreLength = GetEnvironmentVariableW(L"D4R_NGX_CORE", corePath, MAX_PATH);
    if (coreLength > 0 && coreLength < MAX_PATH && !file_exists(corePath))
        g_portable.notes.push_back("the NGX core is missing: copy NVIDIA's _nvngx.dll into d4r\\ngx");
}

static void ensure_portable_config()
{
    std::call_once(g_portableOnce, load_portable_config);
}

static void log_open()
{
    ensure_portable_config();
    wchar_t path[MAX_PATH] = {};
    DWORD length = GetEnvironmentVariableW(L"D4R_SHIM_LOG", path, MAX_PATH);
    if (length > 0 && length < MAX_PATH)
        g_log = _wfopen(path, g_logFresh ? L"w" : L"a");
    else if (!g_portable.dir.empty()) // no log when the DLL's folder is unknown
        g_log = _wfopen((g_portable.dir + L"\\d4r_nvngx.log").c_str(), g_logFresh ? L"w" : L"a");
}

static void logf(const char* format, ...)
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (g_log == nullptr)
        log_open();
    if (g_log == nullptr)
        return;
    std::fprintf(g_log, "[%10.3f] ", GetTickCount64() / 1000.0);
    va_list arguments;
    va_start(arguments, format);
    std::vfprintf(g_log, format, arguments);
    va_end(arguments);
    std::fputc('\n', g_log);
    std::fflush(g_log);
}

// NGX core and feature diagnostics, routed into the shim log.
static void ngx_log_callback(const char* message, int level, unsigned int component)
{
    if (message == nullptr)
        return;
    std::string line(message);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        line.pop_back();
    logf("NGX[%d/%u] %s", level, component, line.c_str());
}

static unsigned int env_uint(const char* name, unsigned int fallback)
{
    char value[64] = {};
    DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
    if (length == 0 || length >= sizeof(value) || value[0] < '0' || value[0] > '9')
        return fallback; // also signs and text: strtoul turns "-1" into 4294967295
    return static_cast<unsigned int>(std::strtoul(value, nullptr, 0));
}

static std::string env_string(const char* name)
{
    char value[1024] = {};
    const DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
    return length > 0 && length < sizeof(value) ? std::string(value, length) : std::string();
}

static bool profile_enabled()
{
    static const bool enabled = env_uint("D4R_PROFILE", 0) != 0;
    return enabled;
}

using ProfileClock = std::chrono::steady_clock;
static double profile_ms(ProfileClock::time_point start, ProfileClock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static double profile_thread_cpu_ms()
{
    FILETIME created = {}, exited = {}, kernel = {}, user = {};
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user))
        return -1.0;
    const uint64_t kernelTicks = (static_cast<uint64_t>(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime;
    const uint64_t userTicks = (static_cast<uint64_t>(user.dwHighDateTime) << 32) | user.dwLowDateTime;
    return static_cast<double>(kernelTicks + userTicks) / 10000.0;
}

// --- CUDA worker thread --------------------------------------------------------
// Every NGX-core and CUDA call runs on this thread so the ZLUDA context that
// the nvcuda bridge creates on first use stays current.

class Worker
{
public:
    void start()
    {
        // Detached: the process may exit without shutting NGX down.
        std::thread([this] { run(); }).detach();
    }

    void post(std::function<void()> job)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        condition_.notify_one();
    }

    size_t pending()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return jobs_.size();
    }

    template <typename Function> auto call(Function function) -> decltype(function())
    {
        using Result = decltype(function());
        auto task = std::make_shared<std::packaged_task<Result()>>(std::move(function));
        auto future = task->get_future();
        post([task] { (*task)(); });
        return future.get();
    }

private:
    void run()
    {
        for (;;)
        {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return !jobs_.empty(); });
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::function<void()>> jobs_;
};

// --- half/float and packed format conversions --------------------------------

static uint16_t float_to_half(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xff)
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    int halfExponent = static_cast<int>(exponent) - 112;
    if (halfExponent >= 31)
        return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t shift = 13;
    if (halfExponent <= 0)
    {
        if (halfExponent < -10)
            return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        shift = static_cast<uint32_t>(14 - halfExponent);
        halfExponent = 0;
    }
    const uint32_t halfMantissa = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    uint32_t result = sign | (static_cast<uint32_t>(halfExponent) << 10);
    result += halfMantissa & (halfExponent == 0 ? 0x7ffu : 0x3ffu);
    if (remainder > halfway || (remainder == halfway && (halfMantissa & 1u)))
        ++result;
    return static_cast<uint16_t>(result);
}

static float half_to_float(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t result;
    if (exponent == 0)
    {
        float magnitude = static_cast<float>(mantissa) * (1.0f / 16777216.0f);
        std::memcpy(&result, &magnitude, sizeof(result));
        result |= sign;
    }
    else if (exponent == 0x1f)
        result = sign | 0x7f800000u | (mantissa << 13);
    else
        result = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

// Unsigned small floats (R11G11B10): 5-bit exponent, 6 or 5 bit mantissa.
static float small_float_to_float(uint32_t bits, int mantissaBits)
{
    const uint32_t exponent = (bits >> mantissaBits) & 0x1fu;
    const uint32_t mantissa = bits & ((1u << mantissaBits) - 1u);
    if (exponent == 0)
        return std::ldexp(static_cast<float>(mantissa), -14 - mantissaBits);
    if (exponent == 0x1f)
        return mantissa ? NAN : INFINITY;
    return std::ldexp(1.0f + static_cast<float>(mantissa) / static_cast<float>(1u << mantissaBits),
                      static_cast<int>(exponent) - 15);
}

static uint32_t float_to_small_float(float value, int mantissaBits)
{
    if (!(value > 0.0f))
        return 0; // negatives and NaN clamp to zero
    const float maximum = std::ldexp(2.0f - std::ldexp(1.0f, -mantissaBits), 15);
    if (value >= maximum)
        value = maximum;
    int exponent;
    const float fraction = std::frexp(value, &exponent); // value = fraction * 2^exponent
    int biased = exponent - 1 + 15;
    if (biased <= 0)
    {
        const uint32_t mantissa = static_cast<uint32_t>(std::lround(std::ldexp(value, 14 + mantissaBits)));
        return mantissa;
    }
    uint32_t mantissa = static_cast<uint32_t>(std::lround((fraction * 2.0f - 1.0f) * static_cast<float>(1u << mantissaBits)));
    if (mantissa >= (1u << mantissaBits))
    {
        mantissa = 0;
        ++biased;
    }
    if (biased >= 31)
        return (30u << mantissaBits) | ((1u << mantissaBits) - 1u);
    return (static_cast<uint32_t>(biased) << mantissaBits) | mantissa;
}

// Canonical CUDA-side layouts: color RGBA16F, depth R32F, motion RG16F,
// exposure R32F, output RGBA16F.
enum class Plane
{
    Color,
    Depth,
    Motion,
    Exposure
};

static bool supported_input(Plane plane, DXGI_FORMAT format)
{
    switch (plane)
    {
    case Plane::Color:
        return format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R16G16B16A16_TYPELESS ||
               format == DXGI_FORMAT_R32G32B32A32_FLOAT || format == DXGI_FORMAT_R11G11B10_FLOAT ||
               format == DXGI_FORMAT_R10G10B10A2_UNORM || format == DXGI_FORMAT_R10G10B10A2_TYPELESS ||
               format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ||
               format == DXGI_FORMAT_R8G8B8A8_TYPELESS || format == DXGI_FORMAT_B8G8R8A8_UNORM ||
               format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB || format == DXGI_FORMAT_B8G8R8A8_TYPELESS ||
               format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    case Plane::Depth:
        return format == DXGI_FORMAT_D32_FLOAT || format == DXGI_FORMAT_R32_FLOAT ||
               format == DXGI_FORMAT_R32_TYPELESS || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ||
               format == DXGI_FORMAT_R32G8X24_TYPELESS || format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS ||
               format == DXGI_FORMAT_D24_UNORM_S8_UINT || format == DXGI_FORMAT_R24G8_TYPELESS ||
               format == DXGI_FORMAT_R24_UNORM_X8_TYPELESS || format == DXGI_FORMAT_D16_UNORM ||
               format == DXGI_FORMAT_R16_UNORM || format == DXGI_FORMAT_R16_TYPELESS ||
               format == DXGI_FORMAT_R16_FLOAT;
    case Plane::Motion:
        return format == DXGI_FORMAT_R16G16_FLOAT || format == DXGI_FORMAT_R16G16_TYPELESS ||
               format == DXGI_FORMAT_R32G32_FLOAT || format == DXGI_FORMAT_R32G32_TYPELESS ||
               format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R16G16B16A16_TYPELESS ||
               format == DXGI_FORMAT_R32G32B32A32_FLOAT || format == DXGI_FORMAT_R8G8B8A8_UNORM;
    case Plane::Exposure:
        return format == DXGI_FORMAT_R32_FLOAT || format == DXGI_FORMAT_R32_TYPELESS ||
               format == DXGI_FORMAT_R16_FLOAT || format == DXGI_FORMAT_R16_TYPELESS ||
               format == DXGI_FORMAT_R32G32B32A32_FLOAT || format == DXGI_FORMAT_R16G16B16A16_FLOAT;
    }
    return false;
}

static bool supported_output(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R16G16B16A16_TYPELESS ||
           format == DXGI_FORMAT_R32G32B32A32_FLOAT || format == DXGI_FORMAT_R11G11B10_FLOAT ||
           format == DXGI_FORMAT_R10G10B10A2_UNORM || format == DXGI_FORMAT_R10G10B10A2_TYPELESS ||
           format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_R8G8B8A8_TYPELESS ||
           format == DXGI_FORMAT_B8G8R8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_TYPELESS ||
           format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
}

static float unorm(uint32_t value, int bits)
{
    return static_cast<float>(value) / static_cast<float>((1u << bits) - 1u);
}

static uint32_t to_unorm(float value, int bits)
{
    const float clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    return static_cast<uint32_t>(std::lround(clamped * static_cast<float>((1u << bits) - 1u)));
}

// Converts one row of a D3D12 texel format into the canonical layout.
static void convert_row_in(Plane plane, DXGI_FORMAT format, const uint8_t* source, void* destination, UINT width)
{
    uint16_t* half = static_cast<uint16_t*>(destination);
    float* single = static_cast<float*>(destination);
    for (UINT x = 0; x < width; ++x)
    {
        switch (plane)
        {
        case Plane::Color:
        {
            float rgba[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            switch (format)
            {
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
            case DXGI_FORMAT_R16G16B16A16_TYPELESS:
                std::memcpy(half + x * 4, source + x * 8, 8);
                continue;
            case DXGI_FORMAT_R32G32B32A32_FLOAT:
                std::memcpy(rgba, source + x * 16, 16);
                break;
            case DXGI_FORMAT_R11G11B10_FLOAT:
            {
                uint32_t packed;
                std::memcpy(&packed, source + x * 4, 4);
                rgba[0] = small_float_to_float(packed & 0x7ffu, 6);
                rgba[1] = small_float_to_float((packed >> 11) & 0x7ffu, 6);
                rgba[2] = small_float_to_float((packed >> 22) & 0x3ffu, 5);
                break;
            }
            case DXGI_FORMAT_R10G10B10A2_UNORM:
            case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            {
                uint32_t packed;
                std::memcpy(&packed, source + x * 4, 4);
                rgba[0] = unorm(packed & 0x3ffu, 10);
                rgba[1] = unorm((packed >> 10) & 0x3ffu, 10);
                rgba[2] = unorm((packed >> 20) & 0x3ffu, 10);
                rgba[3] = unorm(packed >> 30, 2);
                break;
            }
            case DXGI_FORMAT_B8G8R8A8_UNORM:
            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            case DXGI_FORMAT_B8G8R8A8_TYPELESS:
                rgba[0] = unorm(source[x * 4 + 2], 8);
                rgba[1] = unorm(source[x * 4 + 1], 8);
                rgba[2] = unorm(source[x * 4 + 0], 8);
                rgba[3] = unorm(source[x * 4 + 3], 8);
                break;
            case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
            {
                // Microsoft spec: c = (p >> offset) & 0x1FF, E = (p >> 27) & 0x1F,
                // v = c * 2^(E - 24) for every E: there is no implicit leading bit.
                uint32_t packed;
                std::memcpy(&packed, source + x * 4, 4);
                uint32_t e = (packed >> 27) & 0x1Fu;
                float scale = std::ldexp(1.0f, (int)e - 24);
                rgba[0] = static_cast<float>((packed >> 0) & 0x1FFu) * scale;
                rgba[1] = static_cast<float>((packed >> 9) & 0x1FFu) * scale;
                rgba[2] = static_cast<float>((packed >> 18) & 0x1FFu) * scale;
                rgba[3] = 1.0f;
                break;
            }
            default: // RGBA8 variants
                for (int c = 0; c < 4; ++c)
                    rgba[c] = unorm(source[x * 4 + c], 8);
                break;
            }
            for (int c = 0; c < 4; ++c)
                half[x * 4 + c] = float_to_half(rgba[c]);
            break;
        }
        case Plane::Depth:
        {
            switch (format)
            {
            case DXGI_FORMAT_D24_UNORM_S8_UINT:
            case DXGI_FORMAT_R24G8_TYPELESS:
            case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
            {
                uint32_t packed;
                std::memcpy(&packed, source + x * 4, 4);
                single[x] = unorm(packed & 0xffffffu, 24);
                break;
            }
            case DXGI_FORMAT_D16_UNORM:
            case DXGI_FORMAT_R16_UNORM:
            case DXGI_FORMAT_R16_TYPELESS:
            {
                uint16_t value;
                std::memcpy(&value, source + x * 2, 2);
                single[x] = unorm(value, 16);
                break;
            }
            case DXGI_FORMAT_R16_FLOAT:
            {
                uint16_t value;
                std::memcpy(&value, source + x * 2, 2);
                single[x] = half_to_float(value);
                break;
            }
            default: // 32-bit float depth (the depth plane of D32S8 copies as 4 bytes)
                std::memcpy(single + x, source + x * 4, 4);
                break;
            }
            break;
        }
        case Plane::Motion:
            switch (format)
            {
            case DXGI_FORMAT_R16G16_FLOAT:
            case DXGI_FORMAT_R16G16_TYPELESS:
                std::memcpy(half + x * 2, source + x * 4, 4);
                break;
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
            case DXGI_FORMAT_R16G16B16A16_TYPELESS:
                std::memcpy(half + x * 2, source + x * 8, 4);
                break;
            case DXGI_FORMAT_R32G32B32A32_FLOAT:
            case DXGI_FORMAT_R32G32_FLOAT:
            case DXGI_FORMAT_R32G32_TYPELESS:
            {
                const size_t stride = format == DXGI_FORMAT_R32G32B32A32_FLOAT ? 16 : 8;
                float mv[2];
                std::memcpy(mv, source + x * stride, 8);
                half[x * 2] = float_to_half(mv[0]);
                half[x * 2 + 1] = float_to_half(mv[1]);
                break;
            }
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            {
                const uint8_t* rgba = source + x * 4;
                half[x * 2] = float_to_half(static_cast<float>(rgba[0]) / 255.0f);
                half[x * 2 + 1] = float_to_half(static_cast<float>(rgba[1]) / 255.0f);
                break;
            }
            default:
                break;
            }
            break;
        case Plane::Exposure:
            switch (format)
            {
            case DXGI_FORMAT_R16_FLOAT:
            case DXGI_FORMAT_R16_TYPELESS:
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
            {
                uint16_t value;
                std::memcpy(&value, source + x * (format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 2), 2);
                single[x] = half_to_float(value);
                break;
            }
            default:
                std::memcpy(single + x, source + x * (format == DXGI_FORMAT_R32G32B32A32_FLOAT ? 16 : 4), 4);
                break;
            }
            break;
        }
    }
}

// Formats whose texels are bit-identical to the canonical CUDA layout (the
// cases convert_row_in handles as a plain copy), so the mapped staging rows can
// be copied straight into the CUDA array with their D3D12 row pitch.
static bool canonical_input(Plane plane, DXGI_FORMAT format)
{
    switch (plane)
    {
    case Plane::Color:
        return format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R16G16B16A16_TYPELESS;
    case Plane::Depth:
        return format == DXGI_FORMAT_D32_FLOAT || format == DXGI_FORMAT_R32_FLOAT ||
               format == DXGI_FORMAT_R32_TYPELESS || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ||
               format == DXGI_FORMAT_R32G8X24_TYPELESS || format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case Plane::Motion:
        return format == DXGI_FORMAT_R16G16_FLOAT || format == DXGI_FORMAT_R16G16_TYPELESS;
    case Plane::Exposure:
        return format == DXGI_FORMAT_R32_FLOAT || format == DXGI_FORMAT_R32_TYPELESS;
    }
    return false;
}

// R11G11B10 floats share binary16's 5-bit exponent and bias, so each channel
// widens to half exactly by shifting its mantissa; NaN becomes the quiet NaN
// float_to_half produces. Matches convert_row_in bit for bit, without libm.
static uint16_t small_float_to_half(uint32_t bits, int mantissaBits)
{
    const uint32_t exponent = (bits >> mantissaBits) & 0x1fu;
    const uint32_t mantissa = bits & ((1u << mantissaBits) - 1u);
    if (exponent == 0x1f && mantissa != 0)
        return 0x7e00u;
    return static_cast<uint16_t>((exponent << 10) | (mantissa << (10 - mantissaBits)));
}

static void convert_r11g11b10_row(const uint8_t* source, uint16_t* half, UINT width)
{
    for (UINT x = 0; x < width; ++x)
    {
        uint32_t packed;
        std::memcpy(&packed, source + x * 4, 4);
        half[x * 4 + 0] = small_float_to_half(packed & 0x7ffu, 6);
        half[x * 4 + 1] = small_float_to_half((packed >> 11) & 0x7ffu, 6);
        half[x * 4 + 2] = small_float_to_half((packed >> 22) & 0x3ffu, 5);
        half[x * 4 + 3] = 0x3c00u; // 1.0
    }
}

// Packs one RGBA16F DLSS texel into the 9-9-9-5 shared-exponent layout.
// Microsoft spec: v = c * 2^(E - 24), c in [0, 511]. The exponent is chosen as
// the smallest E such that all channels fit in 9 bits, then each channel is
// rounded to the nearest representable value (round-half-even, matching
// D3D's conversion semantics). Values above the format's max (65408.0) are
// clamped, like D3D12 does.
static void rgbe_pack(const float rgba[4], uint8_t* destination)
{
    const float MAX_VALUE = 65408.0f; // 511 * 2^(31 - 24)
    float r = rgba[0] < 0.0f ? 0.0f : (rgba[0] > MAX_VALUE ? MAX_VALUE : rgba[0]);
    float g = rgba[1] < 0.0f ? 0.0f : (rgba[1] > MAX_VALUE ? MAX_VALUE : rgba[1]);
    float b = rgba[2] < 0.0f ? 0.0f : (rgba[2] > MAX_VALUE ? MAX_VALUE : rgba[2]);

    float mx = fmaxf(fmaxf(r, g), b);
    // Smallest scale (i.e. smallest e) such that mx / scale rounds to at most 511. This puts
    // the largest channel in the 9-bit range [256, 511] (lower only at e == 0,
    // which also encodes zero), the maximum precision the format allows.
    int e = 31;
    for (int et = 0; et <= 31; ++et)
    {
        if (mx / std::ldexp(1.0f, et - 24) < 511.5f) { e = et; break; }
    }
    const float scale = std::ldexp(1.0f, e - 24);
    const float inv = 1.0f / scale;

    // Round to nearest, ties to even (D3D round-to-nearest-even semantics).
    auto to_c = [](float v, float inv) -> uint32_t
    {
        double d = std::nearbyint(v * inv); // round-half-even
        if (d < 0.0) d = 0.0;
        if (d > 511.0) d = 511.0;
        return static_cast<uint32_t>(d);
    };

    uint32_t packed = 0;
    packed |= to_c(r, inv) << 0;
    packed |= to_c(g, inv) << 9;
    packed |= to_c(b, inv) << 18;
    packed |= (e & 0x1Fu) << 27;
    std::memcpy(destination, &packed, 4);
}

// Converts one row of RGBA16F DLSS output into the game's output format.
static void convert_row_out(DXGI_FORMAT format, const uint16_t* source, uint8_t* destination, UINT width)
{
    for (UINT x = 0; x < width; ++x)
    {
        const uint16_t* texel = source + x * 4;
        switch (format)
        {
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            std::memcpy(destination + x * 8, texel, 8);
            break;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            for (int c = 0; c < 4; ++c)
            {
                const float value = half_to_float(texel[c]);
                std::memcpy(destination + x * 16 + c * 4, &value, 4);
            }
            break;
        case DXGI_FORMAT_R11G11B10_FLOAT:
        {
            const uint32_t packed = float_to_small_float(half_to_float(texel[0]), 6) |
                                    (float_to_small_float(half_to_float(texel[1]), 6) << 11) |
                                    (float_to_small_float(half_to_float(texel[2]), 5) << 22);
            std::memcpy(destination + x * 4, &packed, 4);
            break;
        }
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        {
            const uint32_t packed = to_unorm(half_to_float(texel[0]), 10) |
                                    (to_unorm(half_to_float(texel[1]), 10) << 10) |
                                    (to_unorm(half_to_float(texel[2]), 10) << 20) |
                                    (to_unorm(half_to_float(texel[3]), 2) << 30);
            std::memcpy(destination + x * 4, &packed, 4);
            break;
        }
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            destination[x * 4 + 0] = static_cast<uint8_t>(to_unorm(half_to_float(texel[2]), 8));
            destination[x * 4 + 1] = static_cast<uint8_t>(to_unorm(half_to_float(texel[1]), 8));
            destination[x * 4 + 2] = static_cast<uint8_t>(to_unorm(half_to_float(texel[0]), 8));
            destination[x * 4 + 3] = static_cast<uint8_t>(to_unorm(half_to_float(texel[3]), 8));
            break;
        case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
        {
            float rgba[4] = {half_to_float(texel[0]), half_to_float(texel[1]), half_to_float(texel[2]), 1.0f};
            rgbe_pack(rgba, destination + x * 4);
            break;
        }
        default: // RGBA8
            for (int c = 0; c < 4; ++c)
                destination[x * 4 + c] = static_cast<uint8_t>(to_unorm(half_to_float(texel[c]), 8));
            break;
        }
    }
}

// --- global state ----------------------------------------------------------------

struct Global
{
    std::mutex mutex;
    bool started = false;
    Worker worker;        // every NGX-core and CUDA call
    std::thread::id workerThread;
    Worker prep, finish;  // CPU-only pipeline stages around it
    Worker cleanup;       // deferred GPU resource destruction, never under allocator locks
    std::atomic<unsigned int> resourceOwners{0};
    bool shutdownPending = false; // under mutex
    HMODULE core = nullptr;
    CoreApi ngx;
    HMODULE cuda = nullptr;
    CudaApi cu;
    bool ngxInitialized = false;
    ID3D12Device* device = nullptr;
    std::vector<std::wstring> paths;
    std::vector<const wchar_t*> pathPointers;
    NgxFeatureCommonInfo featureInfo = {};
    std::wstring dataPath;
    std::atomic<unsigned int> nextHandleId{0x7200};
};

static Global g;
static std::mutex g_featureApiMutex;

template <typename Function> static bool load_export(HMODULE module, const char* name, Function& function)
{
    function = reinterpret_cast<Function>(reinterpret_cast<void*>(GetProcAddress(module, name)));
    if (function == nullptr)
        logf("missing export %s", name);
    return function != nullptr;
}

// OptiScaler's DLSS-input hook answers kernel32 LoadLibrary* calls for any
// *nvngx*.dll outside the game directory with OptiScaler itself, which would
// hide the official core's CUDA exports. ntdll's LdrLoadDll sits below those
// hooks (OptiScaler's own proxies load libraries the same way).
static HMODULE load_library_below_hooks(const wchar_t* path)
{
    struct UnicodeString
    {
        USHORT Length;
        USHORT MaximumLength;
        PWSTR Buffer;
    };
    using LdrLoadDllFn = LONG(NTAPI*)(PWSTR, PULONG, UnicodeString*, PHANDLE);
    auto ldrLoadDll = reinterpret_cast<LdrLoadDllFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "LdrLoadDll")));
    if (ldrLoadDll == nullptr)
        return LoadLibraryW(path);
    std::wstring copy(path);
    UnicodeString name;
    name.Length = static_cast<USHORT>(copy.size() * sizeof(wchar_t));
    name.MaximumLength = static_cast<USHORT>(name.Length + sizeof(wchar_t));
    name.Buffer = copy.data();
    HANDLE module = nullptr;
    const LONG status = ldrLoadDll(nullptr, nullptr, &name, &module);
    if (status < 0)
    {
        logf("LdrLoadDll(%ls) failed: NTSTATUS 0x%08lx", path, static_cast<unsigned long>(status));
        return nullptr;
    }
    return static_cast<HMODULE>(module);
}

// With DLSS inputs enabled, OptiScaler also hooks ntdll-level loads and
// answers every *nvngx*.dll request with itself unless the requested path
// contains the game EXE's directory in lowercase (its test for "the game's own
// NGX core"). The official core must keep its _nvngx.dll name, so request it
// as <lowercase exe dir>\..\..\<core path>: the hook sees the game
// directory, and Wine collapses the ".." segments to the staged core without
// anything being written into the game folder.
static std::wstring core_path_inside_exe_dir(const wchar_t* corePath)
{
    std::wstring core(corePath);
    wchar_t exe[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (length == 0 || length >= MAX_PATH || core.size() < 3 || core[1] != L':' || exe[1] != L':' ||
        towlower(core[0]) != towlower(exe[0]))
        return core;
    std::wstring directory(exe);
    directory.resize(directory.find_last_of(L"\\/"));
    for (wchar_t& c : directory)
        c = static_cast<wchar_t>(towlower(c));
    std::wstring path = directory;
    size_t depth = 0;
    for (size_t index = 3; index <= directory.size(); ++index)
        if (index == directory.size() || directory[index] == L'\\' || directory[index] == L'/')
            ++depth;
    for (size_t level = 0; level < depth; ++level)
        path += L"\\..";
    path += L"\\";
    path += core.substr(3);
    logf("requesting NGX core as %ls", path.c_str());
    return path;
}

static bool load_libraries()
{
    DWORD length = 0;
    // Under Proton, nvcuda.dll resolves to Proton's own builtin, which skips the
    // bridge's fixups. Loading the bridge by path first makes every later
    // LoadLibrary("nvcuda.dll"), including the NGX core's, reuse it.
    wchar_t bridgePath[MAX_PATH] = {};
    length = GetEnvironmentVariableW(L"D4R_NVCUDA_BRIDGE", bridgePath, MAX_PATH);
    if (length > 0 && length < MAX_PATH)
    {
        HMODULE bridge = LoadLibraryW(bridgePath);
        logf("preloaded nvcuda bridge %ls: %p (error %lu)", bridgePath, static_cast<void*>(bridge),
             bridge != nullptr ? 0ul : GetLastError());
        // d4r.ini's Linux-side settings, before the bridge's first CUDA call loads ZLUDA
        if (bridge != nullptr && g_portable.active)
        {
            using SetEnvFn = int(WINAPI*)(const char*, const char*, int);
            const auto setEnv = reinterpret_cast<SetEnvFn>(reinterpret_cast<void*>(GetProcAddress(bridge, "d4rSetEnv")));
            if (setEnv == nullptr)
                logf("the nvcuda bridge has no d4rSetEnv; d4r.ini's ZLUDA settings are not applied");
            for (const auto& [name, value] : g_portable.unixEnv)
                if (setEnv != nullptr && setEnv(name.c_str(), value.c_str(), 0))
                    logf("  %s=%s%s", name.c_str(), value.c_str(),
                         getenv(name.c_str()) != nullptr && value != getenv(name.c_str()) ? " (set by the environment)" : "");
        }
    }
    wchar_t corePath[MAX_PATH] = {};
    length = GetEnvironmentVariableW(L"D4R_NGX_CORE", corePath, MAX_PATH);
    if (length == 0 || length >= MAX_PATH)
    {
        logf("D4R_NGX_CORE is not set; cannot locate the official NGX core");
        return false;
    }
    g.core = load_library_below_hooks(core_path_inside_exe_dir(corePath).c_str());
    if (g.core == nullptr)
    {
        logf("loading D4R_NGX_CORE=%ls failed: %lu", corePath, GetLastError());
        return false;
    }
    // A successful LoadLibrary can still be OptiScaler's interception proxy.
    // Diagnose that before trying to bind the rest of the CUDA NGX API.
    if (GetProcAddress(g.core, "NVSDK_NGX_CUDA_Init") == nullptr)
    {
        wchar_t loadedPath[MAX_PATH] = {};
        GetModuleFileNameW(g.core, loadedPath, MAX_PATH);
        logf("NGX core load returned an incompatible/intercepted module: %ls; "
             "expected %ls. Set [Hooks] HookOriginalNvngxOnly=true in OptiScaler.ini",
             loadedPath, corePath);
        g.core = nullptr;
        return false;
    }
    bool ok = true;
    ok &= load_export(g.core, "NVSDK_NGX_CUDA_Init", g.ngx.init);
    if (!load_export(g.core, "NVSDK_NGX_CUDA_Init_ProjectID", g.ngx.initProjectId))
        g.ngx.initProjectId = nullptr;
    ok &= load_export(g.core, "NVSDK_NGX_CUDA_Shutdown", g.ngx.shutdown);
    ok &= load_export(g.core, "NVSDK_NGX_CUDA_GetParameters", g.ngx.getParameters);
    ok &= load_export(g.core, "NVSDK_NGX_CUDA_AllocateParameters", g.ngx.allocateParameters);
    ok &= load_export(g.core, "NVSDK_NGX_CUDA_GetCapabilityParameters", g.ngx.getCapabilityParameters);
    ok &= load_export(g.core, "NVSDK_NGX_CUDA_DestroyParameters", g.ngx.destroyParameters);
    ok &= load_export(g.core, "NVSDK_NGX_CUDA_CreateFeature", g.ngx.createFeature);
    ok &= load_export(g.core, "NVSDK_NGX_CUDA_ReleaseFeature", g.ngx.releaseFeature);
    ok &= load_export(g.core, "NVSDK_NGX_CUDA_EvaluateFeature", g.ngx.evaluateFeature);

    g.cuda = LoadLibraryA("nvcuda.dll");
    if (g.cuda == nullptr)
    {
        logf("LoadLibraryA(nvcuda.dll) failed: %lu", GetLastError());
        return false;
    }
    ok &= load_export(g.cuda, "cuMemAlloc", g.cu.memAlloc);
    ok &= load_export(g.cuda, "cuMemFree", g.cu.memFree);
    ok &= load_export(g.cuda, "cuArrayCreate", g.cu.arrayCreate);
    ok &= load_export(g.cuda, "cuArray3DCreate", g.cu.array3DCreate);
    ok &= load_export(g.cuda, "cuArrayDestroy", g.cu.arrayDestroy);
    ok &= load_export(g.cuda, "cuMemcpy2D", g.cu.memcpy2D);
    ok &= load_export(g.cuda, "cuTexObjectCreate", g.cu.texObjectCreate);
    ok &= load_export(g.cuda, "cuTexObjectDestroy", g.cu.texObjectDestroy);
    ok &= load_export(g.cuda, "cuSurfObjectCreate", g.cu.surfObjectCreate);
    ok &= load_export(g.cuda, "cuSurfObjectDestroy", g.cu.surfObjectDestroy);
    ok &= load_export(g.cuda, "cuCtxSynchronize", g.cu.ctxSynchronize);
    // The d4r bridge's own export keeps waiting when the application's syncs are elided
    // (D4R_ELIDE_NGX_SYNC); cuCtxSynchronize above stays the fallback.
    {
        decltype(g.cu.ctxSynchronize) ownSync = nullptr;
        if (load_export(g.cuda, "d4rCtxSynchronize", ownSync))
            g.cu.ctxSynchronize = ownSync;
    }
    if (!load_export(g.cuda, "cuMemAllocHost", g.cu.memAllocHost) ||
        !load_export(g.cuda, "cuMemFreeHost", g.cu.memFreeHost) ||
        !load_export(g.cuda, "cuMemcpyHtoDAsync", g.cu.memcpyHtoDAsync) ||
        !load_export(g.cuda, "cuMemcpyDtoHAsync", g.cu.memcpyDtoHAsync) ||
        !load_export(g.cuda, "cuStreamCreate", g.cu.streamCreate) ||
        !load_export(g.cuda, "cuStreamSynchronize", g.cu.streamSynchronize))
    {
        g.cu.memAllocHost = nullptr;
        logf("overlapped staging unavailable; using synchronous copies");
    }
    if (!load_export(g.cuda, "d4rMemcpy2DAsync", g.cu.memcpy2DAsync))
        g.cu.memcpy2DAsync = nullptr;
    if (!load_export(g.cuda, "d4rStreamWaitValue32", g.cu.streamWaitValue32) ||
        !load_export(g.cuda, "d4rWriteValue32", g.cu.writeValue32))
        g.cu.streamWaitValue32 = nullptr, g.cu.writeValue32 = nullptr;
    if (!load_export(g.cuda, "d4rSetArrayRedirect", g.cu.setArrayRedirect))
        g.cu.setArrayRedirect = nullptr;
    if (!load_export(g.cuda, "d4rOutputKernelNative", g.cu.outputKernelNative))
        g.cu.outputKernelNative = nullptr;
    if (!load_export(g.cuda, "d4rRegisterLinearTexture", g.cu.registerLinearTexture))
        g.cu.registerLinearTexture = nullptr;
    g.cu.dilateMotion = reinterpret_cast<decltype(g.cu.dilateMotion)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rDilateMotionVectors")));
    if (profile_enabled() || env_uint("D4R_SHIM_BLOCKING_SYNC", 1) != 0)
    {
        const bool create = load_export(g.cuda, "cuEventCreate", g.cu.eventCreate);
        const bool record = load_export(g.cuda, "cuEventRecord", g.cu.eventRecord);
        const bool wait = env_uint("D4R_SHIM_BLOCKING_SYNC", 1) != 0
                              ? load_export(g.cuda, "d4rEventSynchronize", g.cu.eventSynchronize) : true;
        const bool query = load_export(g.cuda, "cuEventQuery", g.cu.eventQuery);
        const bool destroy = load_export(g.cuda, "cuEventDestroy", g.cu.eventDestroy);
        const bool elapsed = profile_enabled() ? load_export(g.cuda, "cuEventElapsedTime", g.cu.eventElapsedTime) : true;
        if (profile_enabled() && !(create && record && destroy && elapsed))
            logf("D4R_PROFILE: CUDA event timing unavailable; CPU stage timings remain enabled");
        if (env_uint("D4R_SHIM_BLOCKING_SYNC", 1) != 0 && !(create && record && wait && query && destroy))
            logf("blocking output wait unavailable; using context sync");
    }
    logf("loaded NGX core %ls and nvcuda.dll: %s", corePath, ok ? "all exports present" : "exports missing");
    return ok;
}

// A game that initialises NGX with its project ID (engine integrations such as Unreal's) is identified
// to the NGX core the same way.
struct ProjectIdentity
{
    std::string id;
    int engineType = 0;
    std::string engineVersion;
};

static NgxResult initialize(unsigned long long applicationId, const wchar_t* dataPath, ID3D12Device* device,
                            const NgxFeatureCommonInfo* featureInfo, unsigned int sdkVersion,
                            const ProjectIdentity* project = nullptr)
{
    std::lock_guard<std::mutex> lock(g.mutex);
    g.shutdownPending = false;
    if (g.ngxInitialized)
        return NGX_SUCCESS;
    if (!g.started)
    {
        // Allocators can retain COM callbacks and detached workers past NGX
        // shutdown. Keep their code loaded until process exit.
        HMODULE pinned = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                               reinterpret_cast<LPCWSTR>(g_selfModule), &pinned))
            return NGX_FAIL_PLATFORM_ERROR;
        g.worker.start();
        g.workerThread = g.worker.call([] { return std::this_thread::get_id(); });
        g.prep.start();
        g.finish.start();
        g.cleanup.start();
        g.started = true;
    }
    ensure_portable_config();
    if (g_portable.active && !g.core)
    {
        logf("portable install in %ls (settings from d4r.ini, Linux path %s)", g_portable.dir.c_str(),
             g_portable.unixDir.c_str());
        for (const std::string& note : g_portable.notes)
            logf("d4r: %s", note.c_str());
    }
    if (g.core == nullptr && !load_libraries())
    {
        g.core = nullptr;
        return NGX_FAIL_PLATFORM_ERROR;
    }
    if (device != nullptr)
    {
        device->AddRef();
        g.device = device;
        // NGX only accepts a CUDA device whose LUID matches the D3D12 adapter;
        // the nvcuda bridge reports this LUID from cuDeviceGetLuid.
        LUID luid;
        device->GetAdapterLuid(&luid);
        char low[24], high[24];
        std::snprintf(low, sizeof(low), "0x%08lx", static_cast<unsigned long>(luid.LowPart));
        std::snprintf(high, sizeof(high), "0x%08lx", static_cast<unsigned long>(luid.HighPart));
        SetEnvironmentVariableA("D4R_CUDA_LUID_LOW", low);
        SetEnvironmentVariableA("D4R_CUDA_LUID_HIGH", high);
        SetEnvironmentVariableA("D4R_CUDA_NODE_MASK", "1");
        logf("CUDA device LUID set to the D3D12 adapter LUID %s:%s", high, low);
    }

    g.paths.clear();
    wchar_t featureDirectory[MAX_PATH] = {};
    DWORD length = GetEnvironmentVariableW(L"D4R_NGX_FEATURE_DIR", featureDirectory, MAX_PATH);
    if (length > 0 && length < MAX_PATH)
        g.paths.emplace_back(featureDirectory);
    if (featureInfo != nullptr && featureInfo->PathListInfo.Path != nullptr)
        for (unsigned int index = 0; index < featureInfo->PathListInfo.Length; ++index)
            if (featureInfo->PathListInfo.Path[index] != nullptr)
                g.paths.emplace_back(featureInfo->PathListInfo.Path[index]);
    g.pathPointers.clear();
    for (const std::wstring& path : g.paths)
    {
        g.pathPointers.push_back(path.c_str());
        logf("feature search path: %ls", path.c_str());
    }
    g.featureInfo = {};
    g.featureInfo.PathListInfo.Path = g.pathPointers.data();
    g.featureInfo.PathListInfo.Length = static_cast<unsigned int>(g.pathPointers.size());
    g.featureInfo.LoggingInfo.LoggingCallback = reinterpret_cast<void*>(&ngx_log_callback);
    g.featureInfo.LoggingInfo.MinimumLoggingLevel = static_cast<int>(env_uint("D4R_NGX_LOG_LEVEL", 1)); // 1=on, 2=verbose
    g.featureInfo.LoggingInfo.DisableOtherLoggingSinks = false;
    g.dataPath = dataPath != nullptr ? dataPath : L".";

    // The CUDA path takes the same application identity as the game's D3D12 init. (The SDK sample's id
    // is only a last resort: NGX stamps "DLSS SDK - DO NOT DISTRIBUTE" over the output for it.)
    const unsigned long long id = applicationId != 0 ? applicationId : 241534723ULL;
    const unsigned int version = sdkVersion != 0 ? sdkVersion : 0x15;
    // Logging info is only read for API versions 0x14 and later.
    const unsigned int apiVersion = version < 0x14 ? 0x14 : version;
    NgxResult result;
    if (project != nullptr && !project->id.empty() && g.ngx.initProjectId != nullptr)
    {
        result = g.worker.call([&] {
            return g.ngx.initProjectId(project->id.c_str(), project->engineType, project->engineVersion.c_str(),
                                       g.dataPath.c_str(), apiVersion, &g.featureInfo);
        });
        logf("NVSDK_NGX_CUDA_Init_ProjectID(project=%s, engine=%d %s, sdk=0x%x) -> 0x%08x", project->id.c_str(),
             project->engineType, project->engineVersion.c_str(), version, result);
    }
    else
    {
        result = g.worker.call([&] { return g.ngx.init(id, g.dataPath.c_str(), &g.featureInfo, apiVersion); });
        logf("NVSDK_NGX_CUDA_Init(app=%llu, sdk=0x%x) -> 0x%08x", id, version, result);
    }
    using LoadErrorFn = const char*(WINAPI*)();
    const auto loadError = reinterpret_cast<LoadErrorFn>(reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rLoadError")));
    if (loadError != nullptr && loadError()[0] != '\0')
        logf("nvcuda bridge: %s", loadError());
    g.ngxInitialized = result == NGX_SUCCESS;
    return result;
}

// --- per-feature state ---------------------------------------------------------

constexpr int kSlots = 3;
// One more result slot than input slots: a slot is reused only once the GPU
// is past every command list that copies from it (see claim_output_slot).
constexpr int kOutputSlots = kSlots + 1;

// A VkBuffer in device-local memory on vkd3d-proton's VkDevice, exported to
// the CUDA side (see "VRAM interop").
struct VramBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    CudaDevicePtr device = 0;
    void* external = nullptr;
    size_t bytes = 0;
    void* mapped = nullptr;       // engine readiness word: CPU-coherent, kept until HIP and Vulkan retire
    // What a pooled buffer can be reused as: 0 the interop transfer buffers (create_vram_buffer), 1 and 2 the
    // engine's device-local and host-coherent storage buffers (create_engine_shared_buffer).
    uint8_t kind = 0;
};

// A temporary image for colour or exposure format conversion.
struct VramImage
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    UINT width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct Staging
{
    ID3D12Resource* buffer = nullptr;
    uint8_t* mapped = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout = {};
    UINT rows = 0;
    UINT64 total = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT width = 0;
    UINT height = 0;

    std::vector<ID3D12Resource*> retired;

    void retire()
    {
        if (buffer != nullptr)
            retired.push_back(buffer);
        buffer = nullptr;
        mapped = nullptr;
        width = height = 0;
        format = DXGI_FORMAT_UNKNOWN;
    }

    void release()
    {
        if (buffer != nullptr)
            buffer->Release();
        for (ID3D12Resource* old : retired)
            old->Release();
        *this = Staging{};
    }
};

struct CudaImage
{
    CudaArray array = nullptr;
    CudaObject object = 0; // texture or surface object
    UINT width = 0;
    UINT height = 0;
};

struct FrameParams
{
    float jitterX = 0, jitterY = 0, mvScaleX = 1, mvScaleY = 1, sharpness = 0;
    float preExposure = 1, exposureScale = 1, frameTime = 16.6f;
    int reset = 0, invertX = 0, invertY = 0;
    unsigned int renderWidth = 0, renderHeight = 0;
    unsigned int colorBaseX = 0, colorBaseY = 0, depthBaseX = 0, depthBaseY = 0;
    unsigned int mvBaseX = 0, mvBaseY = 0, outputBaseX = 0, outputBaseY = 0;
    bool hasExposure = false;
    bool vram = false; // inputs and output stay in VRAM (VRAM interop)
    bool split = false; // this frame presents its own result (split frames)
    unsigned motionDilation = 0; // actually applied, worker only
    // D4R_SHIM_INPLACE: the game's linear output texture, written without a game-side copy (0 = none)
    CudaDevicePtr outputInPlace = 0;
    uint32_t outputInPlacePitch = 0;
    // D4R_SHIM_OUTPUT_PACKED: the output kernel stores the game's R10G10B10A2 texels itself, so the game-side
    // copy (or the in-place texture) takes them without the RGBA16F conversion image and blit
    bool packedOutput = false;
};

struct FrameTiming
{
    bool enabled = false;
    ProfileClock::time_point gameStart, queuedAt;
    double slotWait = 0, inputRecord = 0, outputRecord = 0, gameCall = 0, gameInterval = 0;
    double convert[4] = {}, upload[4] = {};
    double download = 0, outputConvert = 0;
    double prep = 0, worker = 0, workerWait = 0, finishWait = 0, markerWait = 0;
    double ngxHost = 0, ctxSync = 0, gpuEval = -1;
    double motionDilationWall = 0;
    bool markerDeferred = false; // the worker waits for the input marker (D4R_SHIM_DEFER_MARKER)
    bool gpuEventsRecorded = false;
    double outputSyncWall = 0, outputSyncCpu = -1;
    bool outputSyncBlocking = false;
    double h2dTotal = 0, d2hIssue = 0, d2hWait = 0;
    ProfileClock::time_point prepStart, workerQueued, finishQueued;
    size_t queueDepth = 0;
    uint32_t presentedFrame = 0;
};

// A plane in the canonical CUDA layout, tightly packed, in ordinary host
// memory: filled from the D3D12 staging by one pipeline stage and consumed by
// the next (see prepare_inputs / run_evaluation / publish_output).
struct HostPlane
{
    uint8_t* bytes = nullptr; // page-locked when overlapped staging is on
    size_t capacity = 0;
    bool pinned = false;
    std::vector<uint8_t> fallback;
    UINT width = 0, height = 0;
    size_t rowBytes = 0;
    size_t size() const { return rowBytes * height; }
};

struct InputSlot
{
    Staging color, depth, motion, exposure;
    VramBuffer vram[4]; // VRAM interop: the planes in canonical layout
    // D4R_SHIM_LINEAR_INPUTS: pitch-linear texture objects on vram[] handed to NGX (worker only)
    CudaObject linearTexture[4] = {};
    CudaDevicePtr linearPointer[4] = {};
    UINT linearWidth[4] = {}, linearHeight[4] = {};
    size_t linearPitch[4] = {};
    CudaDevicePtr dilatedMotion = 0;
    size_t dilatedBytes = 0;
    CudaObject dilatedTexture = 0;
    UINT dilatedWidth = 0, dilatedHeight = 0;
    size_t dilatedPitch = 0;
    std::atomic<bool> busy{false}; // readback staging owned by the pipeline
    HostPlane host[4];
    std::atomic<bool> hostBusy{false}; // host planes not yet uploaded by the worker
};

struct OutputSlot
{
    Staging staging;
    VramBuffer vram; // VRAM interop: the result in RGBA16F
    std::atomic<uint32_t> lastReadFrame{0}; // last frame whose command list copies from it
    std::atomic<uint32_t> producedFrame{0};
};

// Views and their D3D12 image owners survive every command list recorded by this feature.
struct EngineBorrowedView
{
    ID3D12Resource* resource = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
    VkImageUsageFlags usage = 0;
    ~EngineBorrowedView()
    {
        if (view) d4r::vkTable.DestroyImageView(device, view, nullptr);
        if (resource) resource->Release();
    }
};

struct Feature
{
    NgxHandle handle = {}; // returned to the caller
    bool nativeVulkan = false; // distinct resource/lifetime contract
    std::atomic<bool> retiring{false};
    IUnknown* resources = nullptr; // main reference plus command-allocator references
    NgxHandle* cudaHandle = nullptr;
    void* cudaParams = nullptr;
    CudaDevicePtr scratch = 0;
    unsigned int width = 0, height = 0, outWidth = 0, outHeight = 0;
    unsigned int cudaWidth = 0, cudaHeight = 0; // render size the NGX feature was created with (worker only)
    unsigned int refusedWidth = 0, refusedHeight = 0; // render size NGX would not create a feature for (worker only)
    int quality = 0, flags = 0, subrects = 0;
    std::array<unsigned int, 6> creationPresets{};
    bool cudaCreationDeferred = false;
    unsigned int preset = 0;
    CudaEvent profileStart = nullptr, profileEnd = nullptr;
    CudaEvent outputReadyEvent = nullptr;
    // output wait pacing (D4R_SHIM_OUTPUT_POLL_US): smoothed time from the event record to its completion, kept
    // separately for each wait site (they wait for very different amounts of GPU work)
    double publishWaitEmaUs = 0.0, evalWaitEmaUs = 0.0;
    // native Vulkan input wait: smoothed time from recording to the input event (waiters may overlap)
    std::atomic<double> vkInputWaitEmaUs{0.0};
    bool profileEventsReady = false;

    CudaImage color, depth, motion, exposure, output;
    CudaObject colorHandle = 0, depthHandle = 0, motionHandle = 0, exposureHandle = 0, outputHandle = 0;

    InputSlot inputs[kSlots];
    OutputSlot outputs[kOutputSlots];
    HostPlane outputHost[2]; // downloaded results waiting for publish_output
    std::atomic<bool> outputHostBusy[2] = {false, false};
    unsigned int nextOutputHost = 0; // worker only
    // Overlapped staging (worker only): linear device buffers the async copies
    // target, moved to/from the CUDA arrays with device-to-device copies.
    CudaDevicePtr planeLinear[4] = {}, outputLinear = 0;
    size_t planeLinearBytes[4] = {}, outputLinearBytes = 0;
    // Frames queued but not yet published (or dropped). evaluate() blocks the
    // game at D4R_SHIM_MAX_IN_FLIGHT (default 3) so a backlog cannot build up
    // between pipeline stages and age the presented result.
    std::atomic<int> inFlight{0};
    DXGI_FORMAT outputFormat = DXGI_FORMAT_UNKNOWN;
    UINT outputWidth = 0, outputHeight = 0;
    // VRAM interop: decided on the first evaluation, dropped for good if the
    // formats later stop qualifying.
    bool vram = false, vramDecided = false;
    VramImage colorConversion;
    VramImage depthConversion;
    VramImage motionConversion;
    VramImage exposureConversion;
    VramImage outputConversion;
    // Split frames: the rest of frame N's command list is submitted only once
    // splitSemaphore reaches N (all admitted frames through N have retired).
    std::atomic<bool> split{false};
    VkSemaphore splitSemaphore = VK_NULL_HANDLE;
    std::mutex splitMutex;
    uint64_t splitSignalled = 0; // under splitMutex
    D4rFrameCompletion splitCompletion; // under splitMutex; cancellation can finish out of order
    // Inputs become ready on the CPU before CUDA work is queued. GPU-side waits
    // for recorded-but-discarded command lists can otherwise wedge the device.
    // D4R_SHIM_OUTPUT_DIRECT: this frame's result is stored by the native output kernel straight into its
    // destination buffer (worker only), so the array -> buffer copy is skipped.
    bool outputRedirected = false;
    // The previous frame's output kernel was the native one and honoured the redirect (worker writes, game
    // thread reads): only then is a frame recorded for packed output.
    std::atomic<bool> packedReady{false};
    bool outputDirectAllowed = false; // the forced preset's output kernel honours the redirect
    // D4R_SHIM_LINEAR_INPUTS: NGX samples the interop buffers directly through pitch-linear
    // texture objects (rows padded to 256 bytes), so the buffer -> array copies disappear.
    bool linearInputs = false;
    bool dilationReported = false;
    std::vector<VramBuffer> retiredBuffers; // may still be read by queued command lists
    std::vector<VramImage> retiredImages;
    ID3D12Resource* marker = nullptr;
    volatile uint32_t* markerValue = nullptr; // [0] latest input; [1 + slot] exact input frame
    uint32_t frame = 0;
    uint32_t rejectedFormatCount = 0; // game thread; rate-limits format diagnostics
    std::atomic<int> latestOutput{-1};
    std::atomic<uint32_t> completedFrames{0};
    uint32_t evaluatedFrames = 0; // worker only
    // D4R_ENGINE: d4r's own inference, recorded into the game's command lists (game thread only). Recorded
    // lists retain `resources`, so this is destroyed only once the GPU has finished with them.
    uint64_t engineViewBytes = 0;
    bool engineViewBudgetReported = false;
    std::vector<std::unique_ptr<EngineBorrowedView>> engineViews; // destroyed after engine descriptor pools
    std::unique_ptr<d4r::GameUpscaler> engine;
    std::unique_ptr<d4r::GameUpscalerM> engineM; // preset M
    unsigned int engineMRenderWidth = 0, engineMRenderHeight = 0; // first evaluation's effective extent
    // Preset M and preset K with the network on the native HIP layers ([Engine] Network): the buffers shared
    // with HIP and the bridge's network. The game's list is split around the network, as for the CUDA backend's
    // split frames. `engineHead` is the network's result (M's expansion / K's final store reads it).
    VramBuffer engineTokens, engineHead;
    // Input readiness for the HIP network: engineSync holds the frame number the engine's front writes (a Vulkan
    // fill) and the network's stream waits for it on the GPU (engineGpuWait), so the layers start without a
    // CPU round trip. engineWaitFrame: the frame a queued stream wait is waiting for (0: none).
    VramBuffer engineSync;
    std::atomic<bool> engineGpuWait{false};
    bool engineNetTimeline = false; // native HIP callback and dropped-frame signals share one timeline lock
    std::atomic<uint32_t> engineWaitFrame{0};
    double engineNetWaitEmaUs = 0.0; // worker only: smoothed launch-to-completion wait, for adaptive polling
    void* engineNet = nullptr;
    bool engineRefused = false;
    float engineJitter[2] = {0, 0}; // of the previous engine frame
    // Serializes picking the slot to present (game thread) against picking
    // the slot to overwrite (worker).
    std::mutex outputMutex;
};

static std::mutex g_featuresMutex;
static std::vector<Feature*> g_features;

static bool create_buffer(D3D12_HEAP_TYPE heapType, UINT64 size, ID3D12Resource** buffer, uint8_t** mapped)
{
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = heapType;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_RESOURCE_STATES state =
        heapType == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ;
    HRESULT hr = g.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                   __uuidof(ID3D12Resource), reinterpret_cast<void**>(buffer));
    if (FAILED(hr))
    {
        logf("CreateCommittedResource(heap=%d, %llu bytes) failed: 0x%08lx", heapType,
             static_cast<unsigned long long>(size), hr);
        return false;
    }
    D3D12_RANGE readRange = {0, heapType == D3D12_HEAP_TYPE_READBACK ? static_cast<SIZE_T>(size) : 0};
    hr = (*buffer)->Map(0, &readRange, reinterpret_cast<void**>(mapped));
    if (FAILED(hr))
    {
        logf("Map failed: 0x%08lx", hr);
        (*buffer)->Release();
        *buffer = nullptr;
        return false;
    }
    return true;
}

// Ensures staging matches the resource's first subresource; heap selects
// readback (inputs) or upload (output).
static bool ensure_staging(Staging& staging, ID3D12Resource* resource, D3D12_HEAP_TYPE heapType)
{
    D3D12_RESOURCE_DESC desc;
    resource->GetDesc(&desc);
    if (staging.buffer != nullptr && staging.format == desc.Format && staging.width == desc.Width &&
        staging.height == desc.Height)
        return true;
    staging.retire();
    UINT64 rowSize = 0;
    g.device->GetCopyableFootprints(&desc, 0, 1, 0, &staging.layout, &staging.rows, &rowSize, &staging.total);
    if (!create_buffer(heapType, staging.total, &staging.buffer, &staging.mapped))
        return false;
    staging.format = desc.Format;
    staging.width = static_cast<UINT>(desc.Width);
    staging.height = desc.Height;
    logf("staging %s %ux%u format=%d rowPitch=%u total=%llu", heapType == D3D12_HEAP_TYPE_READBACK ? "readback" : "upload",
         staging.width, staging.height, staging.format, staging.layout.Footprint.RowPitch,
         static_cast<unsigned long long>(staging.total));
    return true;
}

static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                       D3D12_RESOURCE_STATES after)
{
    if (before == after)
        return;
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

static void copy_to_staging(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, Staging& staging,
                            D3D12_RESOURCE_STATES state)
{
    transition(list, resource, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION destination = {};
    destination.pResource = staging.buffer;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = staging.layout;
    D3D12_TEXTURE_COPY_LOCATION source = {};
    source.pResource = resource;
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source.SubresourceIndex = 0;
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    transition(list, resource, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
}

// --- CUDA-side resources (worker thread) -----------------------------------------

// Sampler of an input texture: clamp addressing, normalized coordinates (as NGX's own textures); bilinear
// for the colour input (D4R_SHIM_COLOR_LINEAR), optionally for motion / depth (experiments).
static CudaTextureDesc input_sampler(UINT width, uint32_t channels, bool linear)
{
    CudaTextureDesc sampler = {};
    sampler.addressMode[0] = 1;
    sampler.addressMode[1] = 1;
    static const bool colorLinear = env_uint("D4R_SHIM_COLOR_LINEAR", 1) != 0;
    sampler.filterMode = linear && colorLinear ? 1 : 0;
    static const bool motionLinear = env_uint("D4R_SHIM_MOTION_LINEAR", 0) != 0;
    static const bool depthLinear = env_uint("D4R_SHIM_DEPTH_LINEAR", 0) != 0;
    if ((motionLinear && channels == 2) || (depthLinear && channels == 1 && width > 1))
        sampler.filterMode = 1;
    sampler.flags = 2; // CU_TRSF_NORMALIZED_COORDINATES
    return sampler;
}

static bool ensure_cuda_image(CudaImage& image, UINT width, UINT height, uint32_t format, uint32_t channels,
                              bool surface, bool linear)
{
    if (image.array != nullptr && image.width == width && image.height == height)
        return true;
    if (image.object != 0)
        surface ? g.cu.surfObjectDestroy(image.object) : g.cu.texObjectDestroy(image.object);
    if (image.array != nullptr)
        g.cu.arrayDestroy(image.array);
    image = CudaImage{};
    int result;
    if (surface)
    {
        const CudaArray3DDescriptor descriptor{width, height, 0, format, channels, 2u /* SURFACE_LDST */};
        result = g.cu.array3DCreate(&image.array, &descriptor);
    }
    else
    {
        const CudaArrayDescriptor descriptor{width, height, format, channels};
        result = g.cu.arrayCreate(&image.array, &descriptor);
    }
    if (result != 0)
    {
        logf("CUDA array create %ux%u fmt=0x%x ch=%u failed: %d", width, height, format, channels, result);
        return false;
    }
    CudaResourceDesc resource = {};
    resource.resType = 0; // CU_RESOURCE_TYPE_ARRAY
    resource.res.array.hArray = image.array;
    if (surface)
        result = g.cu.surfObjectCreate(&image.object, &resource);
    else
    {
        // Matches the texture objects NGX itself creates: clamp addressing,
        // normalized coordinates.
        CudaTextureDesc sampler = {};
        sampler.addressMode[0] = 1;
        sampler.addressMode[1] = 1;
        // D4R_SHIM_COLOR_LINEAR=0 samples the colour input with point filtering (experiment).
        static const bool colorLinear = env_uint("D4R_SHIM_COLOR_LINEAR", 1) != 0;
        sampler.filterMode = linear && colorLinear ? 1 : 0;
        // D4R_SHIM_MOTION_LINEAR / D4R_SHIM_DEPTH_LINEAR=1: bilinear filtering for
        // those planes (identified by format; experiment).
        static const bool motionLinear = env_uint("D4R_SHIM_MOTION_LINEAR", 0) != 0;
        static const bool depthLinear = env_uint("D4R_SHIM_DEPTH_LINEAR", 0) != 0;
        if ((motionLinear && channels == 2) || (depthLinear && channels == 1 && width > 1))
            sampler.filterMode = 1;
        sampler.flags = 2; // CU_TRSF_NORMALIZED_COORDINATES
        result = g.cu.texObjectCreate(&image.object, &resource, &sampler, nullptr);
    }
    if (result != 0)
    {
        logf("CUDA %s object create failed: %d", surface ? "surface" : "texture", result);
        return false;
    }
    image.width = width;
    image.height = height;
    return true;
}

// Device <-> array copy. Queued on the null stream, where NGX evaluates, when the bridge provides
// it (D4R_SHIM_ASYNC_2D=0: ZLUDA's cuMemcpy2D); callers synchronize before the data is used.
static int copy_2d(const CudaMemcpy2D& copy)
{
    static const bool async = env_uint("D4R_SHIM_ASYNC_2D", 1) != 0;
    if (async && g.cu.memcpy2DAsync != nullptr && g.cu.memcpy2DAsync(&copy, nullptr) == 0)
        return 0;
    return g.cu.memcpy2D(&copy);
}

// Splits row-wise staging work (tens of MB per frame at 1440p, which one core
// moves at ~11 GB/s) across helper threads plus the calling worker thread.
// D4R_SHIM_COPY_THREADS sets the total thread count (default 4, 1 = serial).
class RowPool
{
public:
    void run(UINT rows, size_t bytes, const std::function<void(UINT, UINT)>& task)
    {
        const unsigned int parts = threads();
        if (parts <= 1 || bytes < (2u << 20) || rows < parts * 8)
        {
            task(0, rows);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = &task;
            rows_ = rows;
            parts_ = parts;
            remaining_ = parts - 1;
            ++generation_;
        }
        start_.notify_all();
        task(0, rows / parts);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return remaining_ == 0; });
        task_ = nullptr;
    }

private:
    unsigned int threads()
    {
        std::call_once(once_, [this] {
            count_ = env_uint("D4R_SHIM_COPY_THREADS", 4);
            count_ = count_ < 1 ? 1 : (count_ > 16 ? 16 : count_);
            for (unsigned int index = 1; index < count_; ++index)
                std::thread([this, index] { helper(index); }).detach();
        });
        return count_;
    }

    void helper(unsigned int index)
    {
        uint64_t seen = 0;
        for (;;)
        {
            const std::function<void(UINT, UINT)>* task;
            UINT begin, end;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_.wait(lock, [&] { return generation_ != seen; });
                seen = generation_;
                task = task_;
                begin = static_cast<UINT>(static_cast<uint64_t>(rows_) * index / parts_);
                end = static_cast<UINT>(static_cast<uint64_t>(rows_) * (index + 1) / parts_);
            }
            (*task)(begin, end);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --remaining_;
            }
            done_.notify_one();
        }
    }

    std::once_flag once_;
    unsigned int count_ = 1;
    std::mutex mutex_;
    std::condition_variable start_, done_;
    const std::function<void(UINT, UINT)>* task_ = nullptr;
    UINT rows_ = 0;
    unsigned int parts_ = 1, remaining_ = 0;
    uint64_t generation_ = 0;
};

// One pool per CPU pipeline stage; a pool serves one caller at a time.
static RowPool g_inputRows, g_outputRows;

// Copies rows between pitched buffers.
static void copy_rows(RowPool& pool, uint8_t* destination, size_t destinationPitch, const uint8_t* source,
                      size_t sourcePitch, size_t rowBytes, UINT rows)
{
    pool.run(rows, rowBytes * rows, [&](UINT begin, UINT end) {
        if (destinationPitch == rowBytes && sourcePitch == rowBytes)
            std::memcpy(destination + begin * rowBytes, source + begin * rowBytes, rowBytes * (end - begin));
        else
            for (UINT y = begin; y < end; ++y)
                std::memcpy(destination + y * destinationPitch, source + y * sourcePitch, rowBytes);
    });
}

// Overlapped staging: frame N's uploads run while frame N-1's result downloads
// (PCIe is full duplex), as async copies between page-locked host memory and
// linear device buffers on two non-blocking streams. D4R_SHIM_ASYNC_COPIES=0
// keeps the synchronous host<->array copies.
static bool async_copies()
{
    static const bool enabled = env_uint("D4R_SHIM_ASYNC_COPIES", 1) != 0;
    return enabled && g.cu.memAllocHost != nullptr;
}

struct CopyStreams
{
    void* up = nullptr;
    void* down = nullptr;
};

// Worker thread only.
static CopyStreams* copy_streams()
{
    static CopyStreams streams;
    static bool attempted = false;
    if (!attempted)
    {
        attempted = true;
        constexpr unsigned int kNonBlocking = 1; // CU_STREAM_NON_BLOCKING
        if (g.cu.streamCreate(&streams.up, kNonBlocking) != 0 || g.cu.streamCreate(&streams.down, kNonBlocking) != 0)
        {
            logf("cuStreamCreate failed; using synchronous copies");
            streams = {};
        }
    }
    return streams.up != nullptr ? &streams : nullptr;
}

// Grows a host plane's buffer. Page-locked memory is allocated on the CUDA
// worker thread (the only one with a current context).
static void reserve_host(HostPlane& host, size_t bytes)
{
    if (bytes <= host.capacity)
        return;
    auto allocate = [&host, bytes] {
        if (host.pinned)
            g.cu.memFreeHost(host.bytes);
        host = HostPlane{};
        void* pinned = nullptr;
        if (async_copies() && g.cu.memAllocHost(&pinned, bytes) == 0)
        {
            host.bytes = static_cast<uint8_t*>(pinned);
            host.pinned = true;
        }
        else
        {
            host.fallback.resize(bytes);
            host.bytes = host.fallback.data();
        }
        host.capacity = bytes;
        return 0;
    };
    if (async_copies() && std::this_thread::get_id() != g.workerThread)
        g.worker.call(allocate);
    else
        allocate();
}

static void release_host(HostPlane& host)
{
    if (host.pinned)
        g.cu.memFreeHost(host.bytes);
    host = HostPlane{};
}

// Worker thread only.
static bool ensure_linear(CudaDevicePtr& buffer, size_t& capacity, size_t bytes)
{
    if (bytes <= capacity)
        return true;
    if (buffer != 0)
        g.cu.memFree(buffer);
    buffer = 0;
    capacity = 0;
    if (g.cu.memAlloc(&buffer, bytes) != 0)
    {
        buffer = 0;
        logf("cuMemAlloc(%zu) for staging failed", bytes);
        return false;
    }
    capacity = bytes;
    return true;
}

static uint32_t plane_format(Plane plane)
{
    return (plane == Plane::Color || plane == Plane::Motion) ? CUDA_FORMAT_HALF : CUDA_FORMAT_FLOAT;
}

static uint32_t plane_channels(Plane plane)
{
    return plane == Plane::Color ? 4 : (plane == Plane::Motion ? 2 : 1);
}

// --- VRAM interop ------------------------------------------------------------------
//
// With D4R_SHIM_VRAM_INTEROP=1 the inputs and the result never leave VRAM.
// vkd3d-proton's interop interface hands out the game's VkDevice, the VkImage
// behind each D3D12 texture, and the raw VkCommandBuffer behind the game's
// D3D12 command list. evaluate() records Vulkan copies of the inputs into
// exportable device-local buffers (colour formats other than RGBA16F are
// converted by a blit first), and of the latest result out of one; the Wine
// nvcuda bridge (d4rImportVulkanMemory) maps those buffers into CUDA. The
// worker then only moves data between them and the CUDA arrays on the GPU.
// Frames whose formats do not qualify use the readback/upload path.

struct VulkanInterop
{
    std::once_flag once;
    bool ready = false;
    ID3D12DXVKInteropDevice1* interop = nullptr;
    ID3D12DXVKInteropDeviceD4R2* lifetime = nullptr;
    ID3D12DXVKInteropDeviceD4R* split = nullptr; // only with the d4r vkd3d-proton patch
    ID3D12DXVKInteropDeviceD4R3* linear = nullptr; // experimental: linear exportable game textures
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory = {};
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    PFN_vkGetPhysicalDeviceFormatProperties formatProperties = nullptr;
    PFN_vkCreateBuffer createBuffer = nullptr;
    PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements bufferRequirements = nullptr;
    PFN_vkBindBufferMemory bindBuffer = nullptr;
    PFN_vkCreateImage createImage = nullptr;
    PFN_vkDestroyImage destroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements imageRequirements = nullptr;
    PFN_vkBindImageMemory bindImage = nullptr;
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    PFN_vkMapMemory map = nullptr;
    PFN_vkUnmapMemory unmap = nullptr;
    PFN_vkCmdPipelineBarrier barrier = nullptr;
    PFN_vkCmdCopyImageToBuffer copyImageToBuffer = nullptr;
    PFN_vkCmdCopyBufferToImage copyBufferToImage = nullptr;
    PFN_vkCmdBlitImage blit = nullptr;
    PFN_vkCmdFillBuffer fill = nullptr;
    PFN_vkCreateSemaphore createSemaphore = nullptr;
    PFN_vkDestroySemaphore destroySemaphore = nullptr;
    PFN_vkSignalSemaphore signalSemaphore = nullptr;
    int(WINAPI* import)(VkDevice, uint64_t, uint64_t, CudaDevicePtr*, void**) = nullptr;
    int(WINAPI* release)(void*) = nullptr;
};
static VulkanInterop g_vk;

static bool vram_interop_requested()
{
    static const bool requested = env_uint("D4R_SHIM_VRAM_INTEROP", 0) != 0;
    return requested;
}

// Game thread, once.
static void init_vram_interop()
{
    if (FAILED(g.device->QueryInterface(__uuidof(ID3D12DXVKInteropDevice1), reinterpret_cast<void**>(&g_vk.interop))))
    {
        logf("VRAM interop: ID3D12DXVKInteropDevice1 unavailable (not vkd3d-proton?)");
        return;
    }
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->GetVulkanHandles(&instance, &physical, &g_vk.device)))
    {
        logf("VRAM interop: GetVulkanHandles failed");
        return;
    }
    UINT count = 0;
    g_vk.interop->GetDeviceExtensions(&count, nullptr);
    std::vector<const char*> extensions(count);
    g_vk.interop->GetDeviceExtensions(&count, extensions.data());
    bool exportable = false;
    for (const char* name : extensions)
        exportable |= std::strcmp(name, "VK_KHR_external_memory_win32") == 0;
    HMODULE vulkan = GetModuleHandleA("vulkan-1.dll");
    if (vulkan == nullptr)
        vulkan = LoadLibraryA("vulkan-1.dll");
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    if (vulkan != nullptr)
        load_export(vulkan, "vkGetInstanceProcAddr", instanceProc);
    load_export(g.cuda, "d4rImportVulkanMemory", g_vk.import);
    load_export(g.cuda, "d4rReleaseVulkanMemory", g_vk.release);
    if (!exportable || instanceProc == nullptr || g_vk.import == nullptr || g_vk.release == nullptr)
    {
        logf("VRAM interop: unavailable (external_memory_win32 %d, vulkan-1 %p, bridge exports %d)", exportable,
             reinterpret_cast<void*>(instanceProc), g_vk.import != nullptr && g_vk.release != nullptr);
        return;
    }
    auto deviceProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(instanceProc(instance, "vkGetDeviceProcAddr"));
    auto memoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        instanceProc(instance, "vkGetPhysicalDeviceMemoryProperties"));
    if (deviceProc == nullptr || memoryProperties == nullptr)
        return;
    memoryProperties(physical, &g_vk.memory);
    g_vk.physical = physical;
    g_vk.formatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
        instanceProc(instance, "vkGetPhysicalDeviceFormatProperties"));
    bool ok = g_vk.formatProperties != nullptr;
    auto load = [&](auto& function, const char* name) {
        function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(deviceProc(g_vk.device, name));
        ok &= function != nullptr;
    };
    load(g_vk.createBuffer, "vkCreateBuffer");
    load(g_vk.destroyBuffer, "vkDestroyBuffer");
    load(g_vk.bufferRequirements, "vkGetBufferMemoryRequirements");
    load(g_vk.bindBuffer, "vkBindBufferMemory");
    load(g_vk.createImage, "vkCreateImage");
    load(g_vk.destroyImage, "vkDestroyImage");
    load(g_vk.imageRequirements, "vkGetImageMemoryRequirements");
    load(g_vk.bindImage, "vkBindImageMemory");
    load(g_vk.allocate, "vkAllocateMemory");
    load(g_vk.free, "vkFreeMemory");
    load(g_vk.map, "vkMapMemory");
    load(g_vk.unmap, "vkUnmapMemory");
    load(g_vk.barrier, "vkCmdPipelineBarrier");
    load(g_vk.copyImageToBuffer, "vkCmdCopyImageToBuffer");
    load(g_vk.copyBufferToImage, "vkCmdCopyBufferToImage");
    load(g_vk.blit, "vkCmdBlitImage");
    g_vk.fill = reinterpret_cast<PFN_vkCmdFillBuffer>(deviceProc(g_vk.device, "vkCmdFillBuffer"));
    load(g_vk.createSemaphore, "vkCreateSemaphore");
    load(g_vk.destroySemaphore, "vkDestroySemaphore");
    load(g_vk.signalSemaphore, "vkSignalSemaphore");
    if (FAILED(g.device->QueryInterface(__uuidof(ID3D12DXVKInteropDeviceD4R), reinterpret_cast<void**>(&g_vk.split))))
        g_vk.split = nullptr;
    if (FAILED(g.device->QueryInterface(__uuidof(ID3D12DXVKInteropDeviceD4R2), reinterpret_cast<void**>(&g_vk.lifetime))))
        g_vk.lifetime = nullptr;
    if (env_uint("D4R_SHIM_INPLACE", 0) == 0 ||
        FAILED(g.device->QueryInterface(__uuidof(ID3D12DXVKInteropDeviceD4R3), reinterpret_cast<void**>(&g_vk.linear))))
        g_vk.linear = nullptr;
    g_vk.ready = ok;
    logf("VRAM interop: %s (VkDevice %p), split frames %s", ok ? "ready" : "missing Vulkan entry points",
         static_cast<void*>(g_vk.device), g_vk.split != nullptr ? "available" : "unavailable (stock vkd3d-proton)");
}

static bool vram_interop_available()
{
    if (!vram_interop_requested())
        return false;
    std::call_once(g_vk.once, init_vram_interop);
    return g_vk.ready;
}

// Device-local memory; prefers memory the CPU cannot map (no BAR window).
static int device_local_type(uint32_t typeBits)
{
    for (int pass = 0; pass < 2; ++pass)
        for (uint32_t index = 0; index < g_vk.memory.memoryTypeCount; ++index)
        {
            const VkMemoryPropertyFlags flags = g_vk.memory.memoryTypes[index].propertyFlags;
            if ((typeBits & (1u << index)) && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (pass == 1 || !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)))
                return static_cast<int>(index);
        }
    return -1;
}

// Buffers of released features, reused by later ones. ROCm up to at least 7.2.4 never frees memory mapped with
// hipExternalMemoryGetMappedBuffer (the mapping's buffer view is retained twice but released once by hipFree), so
// destroying a buffer would leak its memory: each DLSS quality change (a new feature) would add ~100 MB of VRAM.
static std::mutex g_vramPoolMutex;
static std::vector<VramBuffer> g_vramPool;
constexpr size_t kVramPoolLimit = 64;

// The smallest pooled buffer of `kind` that holds `bytes`.
static bool take_pooled_vram_buffer(VramBuffer& target, size_t bytes, uint8_t kind = 0)
{
    std::lock_guard<std::mutex> lock(g_vramPoolMutex);
    auto best = g_vramPool.end();
    for (auto it = g_vramPool.begin(); it != g_vramPool.end(); ++it)
        if (it->kind == kind && it->bytes >= bytes && (best == g_vramPool.end() || it->bytes < best->bytes))
            best = it;
    if (best == g_vramPool.end())
        return false;
    target = *best;
    g_vramPool.erase(best);
    return true;
}

// Game thread. The CUDA import runs on the worker, which owns the context.
static bool create_vram_buffer(VramBuffer& target, size_t bytes)
{
    if (take_pooled_vram_buffer(target, bytes))
        return true;
    VramBuffer buffer;
    VkExternalMemoryBufferCreateInfo external = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = g_vk.createBuffer(g_vk.device, &info, nullptr, &buffer.buffer);
    VkMemoryRequirements requirements = {};
    if (result == VK_SUCCESS)
        g_vk.bufferRequirements(g_vk.device, buffer.buffer, &requirements);
    const int type = device_local_type(requirements.memoryTypeBits);
    VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = buffer.buffer;
    VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dedicated};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exportInfo};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = static_cast<uint32_t>(type);
    if (result == VK_SUCCESS && type < 0)
        result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (result == VK_SUCCESS)
        result = g_vk.allocate(g_vk.device, &allocation, nullptr, &buffer.memory);
    if (result == VK_SUCCESS)
        result = g_vk.bindBuffer(g_vk.device, buffer.buffer, buffer.memory, 0);
    int imported = -1;
    if (result == VK_SUCCESS)
        imported = g.worker.call([&] {
            return g_vk.import(g_vk.device, reinterpret_cast<uint64_t>(buffer.memory), requirements.size,
                               &buffer.device, &buffer.external);
        });
    if (result != VK_SUCCESS || imported != 0)
    {
        logf("VRAM interop: exportable buffer of %zu bytes failed (VkResult %d, import %d)", bytes, result, imported);
        if (buffer.memory != VK_NULL_HANDLE)
            g_vk.free(g_vk.device, buffer.memory, nullptr);
        if (buffer.buffer != VK_NULL_HANDLE)
            g_vk.destroyBuffer(g_vk.device, buffer.buffer, nullptr);
        return false;
    }
    buffer.bytes = bytes;
    target = buffer;
    return true;
}

// Only once no command list or CUDA work uses the buffer.
static void destroy_vram_buffer(VramBuffer& buffer)
{
    if (buffer.external != nullptr)
        g.worker.call([&buffer] {
            const int freed = g.cu.memFree(buffer.device);
            const int released = g_vk.release(buffer.external);
            if (freed != 0 || released != 0)
                logf("VRAM interop: freeing %zu-byte buffer: cuMemFree %d, release %d", buffer.bytes, freed, released);
            return released;
        });
    if (buffer.mapped != nullptr)
        g_vk.unmap(g_vk.device, buffer.memory);
    if (buffer.memory != VK_NULL_HANDLE)
        g_vk.free(g_vk.device, buffer.memory, nullptr);
    if (buffer.buffer != VK_NULL_HANDLE)
        g_vk.destroyBuffer(g_vk.device, buffer.buffer, nullptr);
    buffer = VramBuffer{};
}

// Only once no command list or CUDA work uses the buffer: keeps it for a later feature (see g_vramPool).
static void recycle_vram_buffer(VramBuffer& buffer)
{
    if (buffer.buffer == VK_NULL_HANDLE)
        return;
    {
        std::lock_guard<std::mutex> lock(g_vramPoolMutex);
        if (g_vramPool.size() < kVramPoolLimit)
        {
            g_vramPool.push_back(buffer);
            buffer = VramBuffer{};
            return;
        }
    }
    destroy_vram_buffer(buffer);
}

static void free_vram_pool()
{
    std::vector<VramBuffer> pool;
    {
        std::lock_guard<std::mutex> lock(g_vramPoolMutex);
        pool.swap(g_vramPool);
    }
    for (VramBuffer& buffer : pool)
        destroy_vram_buffer(buffer);
}

static void destroy_vram_image(VramImage& image)
{
    if (image.memory != VK_NULL_HANDLE)
        g_vk.free(g_vk.device, image.memory, nullptr);
    if (image.image != VK_NULL_HANDLE)
        g_vk.destroyImage(g_vk.device, image.image, nullptr);
    image = VramImage{};
}

static bool create_vram_image(VramImage& image, UINT width, UINT height, VkFormat format)
{
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkResult result = g_vk.createImage(g_vk.device, &info, nullptr, &image.image);
    VkMemoryRequirements requirements = {};
    if (result == VK_SUCCESS)
        g_vk.imageRequirements(g_vk.device, image.image, &requirements);
    const int type = device_local_type(requirements.memoryTypeBits);
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = static_cast<uint32_t>(type);
    if (result == VK_SUCCESS && type < 0)
        result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (result == VK_SUCCESS)
        result = g_vk.allocate(g_vk.device, &allocation, nullptr, &image.memory);
    if (result == VK_SUCCESS)
        result = g_vk.bindImage(g_vk.device, image.image, image.memory, 0);
    if (result != VK_SUCCESS)
    {
        logf("VRAM interop: %ux%u conversion image failed: %d", width, height, result);
        destroy_vram_image(image);
        return false;
    }
    image.width = width;
    image.height = height;
    image.format = format;
    return true;
}

static size_t canonical_texel_bytes(Plane plane)
{
    return (plane_format(plane) == CUDA_FORMAT_HALF ? 2 : 4) * plane_channels(plane);
}

static size_t vk_texel_bytes(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_UNORM:
        return 8;
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: // the depth aspect alone
        return 4;
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_UINT:
        return 2;
    default:
        return 0;
    }
}

// D4R_SHIM_INPLACE (experimental): a game texture that vkd3d-proton created linear and exportable is imported
// into CUDA once, so DLSS writes its texels without a game-side copy. Imports and their source resources
// stay alive for the process lifetime (ROCm does not free mapped external memory, see g_vramPool).
struct InPlaceTexture
{
    CudaDevicePtr texels = 0;
    size_t rowPitch = 0;
};
struct InPlaceImport
{
    uint64_t memory;
    CudaDevicePtr base;
    ID3D12Resource* owner; // retained: prevents reuse of the cached VkDeviceMemory handle
};
static std::mutex g_inPlaceMutex;
static std::vector<InPlaceImport> g_inPlaceImports;

// Game thread.
static bool describe_in_place(ID3D12Resource* resource, InPlaceTexture& texture)
{
    UINT64 memory = 0, size = 0, offset = 0, pitch = 0;
    if (g_vk.linear == nullptr || FAILED(g_vk.linear->GetVulkanLinearImageInfo(resource, &memory, &size, &offset, &pitch)))
        return false;
    CudaDevicePtr base = 0;
    {
        std::lock_guard<std::mutex> lock(g_inPlaceMutex);
        for (const auto& entry : g_inPlaceImports)
            if (entry.memory == memory)
                base = entry.base;
    }
    if (base == 0)
    {
        void* external = nullptr;
        const int imported = g.worker.call([&] {
            return g_vk.import(g_vk.device, memory, size, &base, &external);
        });
        if (imported != 0 || base == 0)
        {
            logf("in-place: importing %llu bytes of texture memory failed (%d)", static_cast<unsigned long long>(size), imported);
            return false;
        }
        logf("in-place: imported %llu bytes of game texture memory (offset %llu, row pitch %llu)",
             static_cast<unsigned long long>(size), static_cast<unsigned long long>(offset),
             static_cast<unsigned long long>(pitch));
        std::lock_guard<std::mutex> lock(g_inPlaceMutex);
        resource->AddRef();
        g_inPlaceImports.push_back({memory, base, resource});
    }
    texture.texels = base + offset;
    texture.rowPitch = static_cast<size_t>(pitch);
    return true;
}

// How one input or the output moves between its D3D12 texture and a buffer.
struct VramCopy
{
    ID3D12Resource* resource = nullptr;
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    bool convert = false; // blit to/from the plane's canonical format
    UINT width = 0, height = 0;
};

static bool vram_blit_supported(VkFormat source, VkFormat destination)
{
    VkFormatProperties sourceProperties = {}, destinationProperties = {};
    g_vk.formatProperties(g_vk.physical, source, &sourceProperties);
    g_vk.formatProperties(g_vk.physical, destination, &destinationProperties);
    return (sourceProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0 &&
           (destinationProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
}

// sRGB is deliberately excluded: the host staging path treats its bytes as
// UNORM and a Vulkan blit would apply an sRGB transfer function instead.
static bool vram_color_blit_supported(VkFormat format, bool output)
{
    switch (format)
    {
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        break;
    default:
        return false;
    }
    return output ? vram_blit_supported(VK_FORMAT_R16G16B16A16_SFLOAT, format)
                  : vram_blit_supported(format, VK_FORMAT_R16G16B16A16_SFLOAT);
}

// The host path takes only the first (R) component of an exposure texel.
static bool vram_exposure_blit_supported(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return vram_blit_supported(format, VK_FORMAT_R32_SFLOAT);
    default:
        return false;
    }
}

// Depth in a 16-bit format is widened to R32F by a blit (UNORM normalizes, SFLOAT converts).
static bool vram_depth_blit_supported(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
        return vram_blit_supported(format, VK_FORMAT_R32_SFLOAT);
    default:
        return false;
    }
}

// Motion vectors in a wider float format keep their first two components (as the host path does).
static bool vram_motion_blit_supported(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R8G8B8A8_UNORM:
        return vram_blit_supported(format, VK_FORMAT_R16G16_SFLOAT);
    default:
        return false;
    }
}

// Whether `resource` can be copied raw (or, for colour/motion/exposure, blitted)
// to/from the canonical layout of `plane`.
static bool describe_vram_copy(ID3D12Resource* resource, Plane plane, VramCopy& copy, bool output = false)
{
    UINT64 handle = 0, offset = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    if (FAILED(g_vk.interop->GetVulkanResourceInfo1(resource, &handle, &offset, &format)) || handle == 0)
        return false;
    D3D12_RESOURCE_DESC desc;
    resource->GetDesc(&desc);
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1)
        return false;
    copy.resource = resource;
    copy.image = reinterpret_cast<VkImage>(handle);
    copy.width = static_cast<UINT>(desc.Width);
    copy.height = desc.Height;
    copy.aspect = (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT)
                      ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    if (plane == Plane::Color)
    {
        copy.convert = vram_color_blit_supported(format, output);
        return copy.convert || format == VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    if (plane == Plane::Exposure)
        copy.convert = vram_exposure_blit_supported(format);
    else if (plane == Plane::Motion)
        copy.convert = vram_motion_blit_supported(format);
    else if (plane == Plane::Depth)
        copy.convert = vram_depth_blit_supported(format);
    return copy.convert || vk_texel_bytes(format) == canonical_texel_bytes(plane);
}

static const VkMemoryBarrier kBeforeTransfer = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
                                                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
static const VkMemoryBarrier kAfterTransfer = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                               VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};

static VkBufferImageCopy full_region(const VramCopy& copy)
{
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = copy.aspect;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {copy.width, copy.height, 1};
    return region;
}

// Records the input copies into the slot's buffers. The resources are in
// COPY_SOURCE state already.
static bool record_vram_inputs(Feature& feature, ID3D12GraphicsCommandList* list, InputSlot& slot,
                               const VramCopy* copies, int count)
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &cmd)))
        return false;
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &kBeforeTransfer, 0,
                 nullptr, 0, nullptr);
    for (int index = 0; index < count; ++index)
    {
        const VramCopy& copy = copies[index];
        VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
        g_vk.interop->GetVulkanImageLayout(copy.resource, D3D12_RESOURCE_STATE_COPY_SOURCE, &layout);
        VkBufferImageCopy region = full_region(copy);
        static const Plane kPlanes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
        if (feature.linearInputs)
            region.bufferRowLength = static_cast<uint32_t>(slot.host[index].rowBytes / canonical_texel_bytes(kPlanes[index]));
        if (!copy.convert)
        {
            g_vk.copyImageToBuffer(cmd, copy.image, layout, slot.vram[index].buffer, 1, &region);
            continue;
        }
        // Convert colour to RGBA16F, motion to RG16F or exposure's first component
        // to R32F, then copy from the canonical image into the shared buffer.
        VkImage conversion = kPlanes[index] == Plane::Exposure ? feature.exposureConversion.image
                             : kPlanes[index] == Plane::Motion ? feature.motionConversion.image
                             : kPlanes[index] == Plane::Depth ? feature.depthConversion.image
                                                              : feature.colorConversion.image;
        VkImageMemoryBarrier toDestination = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toDestination.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toDestination.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDestination.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDestination.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDestination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDestination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDestination.image = conversion;
        toDestination.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                     &toDestination);
        VkImageBlit blit = {};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {static_cast<int32_t>(copy.width), static_cast<int32_t>(copy.height), 1};
        blit.dstSubresource = blit.srcSubresource;
        blit.dstOffsets[1] = blit.srcOffsets[1];
        g_vk.blit(cmd, copy.image, layout, conversion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                  VK_FILTER_NEAREST);
        VkImageMemoryBarrier toSource = toDestination;
        toSource.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSource.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                     &toSource);
        g_vk.copyImageToBuffer(cmd, conversion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.vram[index].buffer, 1,
                               &region);
    }
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &kAfterTransfer, 0,
                 nullptr, 0, nullptr);
    return SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
}

// Records the copy of a finished result into the output texture, which is in
// COPY_DEST state already.
static bool record_vram_output(ID3D12GraphicsCommandList* list, const VramBuffer& buffer, const VramCopy& copy,
                               const VramImage& conversion)
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &cmd)))
        return false;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    g_vk.interop->GetVulkanImageLayout(copy.resource, D3D12_RESOURCE_STATE_COPY_DEST, &layout);
    const VkBufferImageCopy region = full_region(copy);
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &kBeforeTransfer, 0,
                 nullptr, 0, nullptr);
    if (!copy.convert)
        g_vk.copyBufferToImage(cmd, buffer.buffer, copy.image, layout, 1, &region);
    else
    {
        VkImageMemoryBarrier toDestination = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toDestination.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toDestination.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDestination.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDestination.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDestination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDestination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDestination.image = conversion.image;
        toDestination.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                     &toDestination);
        g_vk.copyBufferToImage(cmd, buffer.buffer, conversion.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier toSource = toDestination;
        toSource.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSource.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                     &toSource);
        VkImageBlit blit = {};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {static_cast<int32_t>(copy.width), static_cast<int32_t>(copy.height), 1};
        blit.dstSubresource = blit.srcSubresource;
        blit.dstOffsets[1] = blit.srcOffsets[1];
        g_vk.blit(cmd, conversion.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copy.image, layout, 1, &blit,
                  VK_FILTER_NEAREST);
    }
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &kAfterTransfer, 0,
                 nullptr, 0, nullptr);
    return SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
}

// --- Split frames -------------------------------------------------------------------
//
// To present frame N's own result, everything the game records after its DLSS
// call must wait for DLSS. With the d4r vkd3d-proton patch, evaluate() splits
// the game's command list right after the input copies and the frame marker;
// vkd3d-proton submits the second half separately, waiting (at queue submission
// level, so no GPU ring is ever blocked) for splitSemaphore to reach N. The
// worker writes N's result into output slot N % kOutputSlots. Completed and
// dropped frames advance the signal only past earlier retired frames. The
// watchdog reports delays without releasing an unfinished producer.

static VkResult signal_split_semaphore(Feature& feature, uint64_t value);

static void release_split_frame(Feature* feature, uint32_t frame)
{
    std::lock_guard<std::mutex> lock(feature->splitMutex);
    const uint64_t value = feature->splitCompletion.retire(frame);
    if (value <= feature->splitSignalled)
        return;
    const VkResult result = signal_split_semaphore(*feature, value);
    if (result == VK_SUCCESS)
        feature->splitSignalled = value;
    if (result != VK_SUCCESS)
        logf("split frame %llu signal failed (VkResult %d)", static_cast<unsigned long long>(value), result);
}

static void split_watchdog()
{
    const auto timeout = std::chrono::milliseconds(env_uint("D4R_SHIM_SPLIT_TIMEOUT_MS", 200));
    struct Seen
    {
        uint32_t marker = 0;
        std::chrono::steady_clock::time_point at;
    };
    std::vector<std::pair<Feature*, Seen>> seen;
    for (;;)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::lock_guard<std::mutex> lock(g_featuresMutex);
        const auto now = std::chrono::steady_clock::now();
        for (Feature* feature : g_features)
        {
            if (!feature->split || feature->markerValue == nullptr)
                continue;
            auto it = std::find_if(seen.begin(), seen.end(), [feature](const auto& entry) { return entry.first == feature; });
            if (it == seen.end())
                it = seen.insert(seen.end(), {feature, Seen{}});
            const uint32_t marker = *feature->markerValue;
            uint64_t signalled;
            {
                std::lock_guard<std::mutex> splitLock(feature->splitMutex);
                signalled = feature->splitSignalled;
            }
            if (marker <= signalled)
                continue; // nothing on the GPU is waiting on us
            if (it->second.marker != marker)
                it->second = Seen{marker, now};
            else if (now - it->second.at > timeout)
            {
                // A slow producer is not a completed producer. Releasing this
                // wait would race the game's output copy with a late CUDA write.
                logf("split frame %u still waiting for DLSS; preserving output synchronization", marker);
                it->second.at = now;
            }
        }
        // Forget released features.
        seen.erase(std::remove_if(seen.begin(), seen.end(),
                                  [](const auto& entry) {
                                      return std::find(g_features.begin(), g_features.end(), entry.first) ==
                                             g_features.end();
                                  }),
                   seen.end());
    }
}

static bool create_split_semaphore(Feature& feature)
{
    VkSemaphoreTypeCreateInfo type = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0;
    const VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    if (g_vk.createSemaphore(g_vk.device, &info, nullptr, &feature.splitSemaphore) != VK_SUCCESS)
    {
        logf("split frames: vkCreateSemaphore failed");
        return false;
    }
    static std::once_flag watchdog;
    std::call_once(watchdog, [] { std::thread(split_watchdog).detach(); });
    return true;
}

// Game thread: grows a buffer; an outgrown one is kept until the feature is
// released since queued command lists may still reference it.
static bool ensure_vram_buffer(Feature& feature, VramBuffer& buffer, size_t bytes)
{
    if (buffer.buffer != VK_NULL_HANDLE && buffer.bytes >= bytes)
        return true;
    if (buffer.buffer != VK_NULL_HANDLE)
        feature.retiredBuffers.push_back(buffer);
    buffer = VramBuffer{};
    return create_vram_buffer(buffer, bytes);
}

static bool ensure_conversion_image(Feature& feature, VramImage& image, UINT width, UINT height, VkFormat format)
{
    if (image.image != VK_NULL_HANDLE && image.width == width && image.height == height && image.format == format)
        return true;
    if (image.image != VK_NULL_HANDLE)
        feature.retiredImages.push_back(image);
    image = VramImage{};
    return create_vram_image(image, width, height, format);
}

static void release_vram(Feature& feature)
{
    if (feature.splitSemaphore != VK_NULL_HANDLE)
        g_vk.destroySemaphore(g_vk.device, feature.splitSemaphore, nullptr);
    for (InputSlot& slot : feature.inputs)
        for (VramBuffer& buffer : slot.vram)
            recycle_vram_buffer(buffer);
    for (OutputSlot& slot : feature.outputs)
        recycle_vram_buffer(slot.vram);
    for (VramBuffer& buffer : feature.retiredBuffers)
        recycle_vram_buffer(buffer);
    feature.retiredBuffers.clear();
    destroy_vram_image(feature.colorConversion);
    destroy_vram_image(feature.depthConversion);
    destroy_vram_image(feature.motionConversion);
    destroy_vram_image(feature.exposureConversion);
    destroy_vram_image(feature.outputConversion);
    for (VramImage& image : feature.retiredImages)
        destroy_vram_image(image);
    feature.retiredImages.clear();
}

// Prep stage: converts one readback staging buffer into the canonical layout.
static void stage_plane(Plane plane, const Staging& staging, HostPlane& host, FrameTiming* timing, int timingIndex)
{
    const size_t texelBytes = (plane_format(plane) == CUDA_FORMAT_HALF ? 2 : 4) * plane_channels(plane);
    reserve_host(host, texelBytes * staging.width * staging.height);
    host.width = staging.width;
    host.height = staging.height;
    host.rowBytes = texelBytes * staging.width;
    uint8_t* rows = host.bytes;
    const size_t rowBytes = host.rowBytes;
    const uint8_t* source = staging.mapped + staging.layout.Offset;
    const size_t sourcePitch = staging.layout.Footprint.RowPitch;
    const auto conversionStart = timing != nullptr ? ProfileClock::now() : ProfileClock::time_point{};
    if (canonical_input(plane, staging.format))
        copy_rows(g_inputRows, rows, rowBytes, source, sourcePitch, rowBytes, staging.height);
    else
        g_inputRows.run(staging.height, rowBytes * staging.height, [&](UINT begin, UINT end) {
            for (UINT y = begin; y < end; ++y)
                if (plane == Plane::Color && staging.format == DXGI_FORMAT_R11G11B10_FLOAT)
                    convert_r11g11b10_row(source + y * sourcePitch,
                                          reinterpret_cast<uint16_t*>(rows + rowBytes * y), staging.width);
                else
                    convert_row_in(plane, staging.format, source + y * sourcePitch, rows + rowBytes * y,
                                   staging.width);
        });
    if (timing != nullptr)
        timing->convert[timingIndex] = profile_ms(conversionStart, ProfileClock::now());
}

// Worker stage: uploads one staged plane into its CUDA array.
// D4R_SHIM_MV_STATS=N logs, every N frames, what the game's motion vectors
// look like (staged RG16F plane): how much of the texture is populated, the
// populated bounding box (a render-size sub-rectangle means render-resolution
// vectors in a display-size texture), magnitudes in texture units, and the
// global image shift between the previous and current colour frames measured
// in render pixels, so the vectors' unit can be read off their ratio.
static std::vector<float> downsampled_luma(const HostPlane& color, UINT& width, UINT& height)
{
    // RGBA16F colour, 2x2 box-downsampled luma.
    width = color.width / 2, height = color.height / 2;
    std::vector<float> luma(static_cast<size_t>(width) * height);
    const uint8_t* bytes = color.bytes != nullptr ? color.bytes : color.fallback.data();
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x)
        {
            float sum = 0.0f;
            for (UINT dy = 0; dy < 2; ++dy)
            {
                const uint16_t* row = reinterpret_cast<const uint16_t*>(bytes + (2 * y + dy) * color.rowBytes);
                for (UINT dx = 0; dx < 2; ++dx)
                    for (UINT c = 0; c < 3; ++c)
                        sum += static_cast<float>(std::bit_cast<_Float16>(row[(2 * x + dx) * 4 + c]));
            }
            luma[static_cast<size_t>(y) * width + x] = sum / (sum + 12.0f); // tone-mapped (HDR input)
        }
    return luma;
}

static void log_motion_stats(const HostPlane& host, const HostPlane& color, uint32_t frame)
{
    static const uint32_t every = env_uint("D4R_SHIM_MV_STATS", 0);
    static std::vector<float> previousLuma;
    static uint32_t previousFrame = 0;
    if (every == 0 || host.width == 0 || color.width == 0)
        return;
    const bool report = frame % every == 0;
    if (!report && (frame + 1) % every != 0)
        return;
    UINT lumaWidth = 0, lumaHeight = 0;
    std::vector<float> luma = downsampled_luma(color, lumaWidth, lumaHeight);
    if (!report)
    {
        previousLuma = std::move(luma);
        previousFrame = frame;
        return;
    }
    const uint8_t* bytes = host.bytes != nullptr ? host.bytes : host.fallback.data();
    size_t nonzero = 0;
    double sumMagnitude = 0.0;
    float maxMagnitude = 0.0f;
    UINT minX = host.width, minY = host.height, maxX = 0, maxY = 0;
    std::vector<float> xs, ys;
    for (UINT y = 0; y < host.height; ++y)
    {
        const uint16_t* row = reinterpret_cast<const uint16_t*>(bytes + y * host.rowBytes);
        for (UINT x = 0; x < host.width; ++x)
        {
            const float mx = static_cast<float>(std::bit_cast<_Float16>(row[x * 2]));
            const float my = static_cast<float>(std::bit_cast<_Float16>(row[x * 2 + 1]));
            if ((x % 8) == 0 && (y % 8) == 0)
            {
                xs.push_back(mx);
                ys.push_back(my);
            }
            if (mx == 0.0f && my == 0.0f)
                continue;
            ++nonzero;
            const float magnitude = std::sqrt(mx * mx + my * my);
            sumMagnitude += magnitude;
            maxMagnitude = std::max(maxMagnitude, magnitude);
            minX = std::min(minX, x), maxX = std::max(maxX, x);
            minY = std::min(minY, y), maxY = std::max(maxY, y);
        }
    }
    auto median = [](std::vector<float>& values) {
        if (values.empty())
            return 0.0f;
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        return values[values.size() / 2];
    };
    const float medianX = median(xs), medianY = median(ys);
    // Global shift d (render pixels, current -> previous, the vectors'
    // convention) minimising |cur(p) - prev(p + d)| over the central region.
    int bestX = 0, bestY = 0;
    double bestCost = -1.0, zeroCost = 0.0;
    if (previousFrame + 1 == frame && previousLuma.size() == luma.size())
    {
        const int range = 80, rangeY = 40; // in downsampled pixels
        const int x0 = range, x1 = static_cast<int>(lumaWidth) - range;
        const int y0 = rangeY, y1 = static_cast<int>(lumaHeight) - rangeY;
        for (int dy = -rangeY; dy <= rangeY; ++dy)
            for (int dx = -range; dx <= range; ++dx)
            {
                double cost = 0.0;
                for (int y = y0; y < y1; y += 2)
                    for (int x = x0; x < x1; x += 2)
                        cost += std::fabs(luma[static_cast<size_t>(y) * lumaWidth + x] -
                                          previousLuma[static_cast<size_t>(y + dy) * lumaWidth + (x + dx)]);
                if (dx == 0 && dy == 0)
                    zeroCost = cost;
                if (bestCost < 0.0 || cost < bestCost)
                    bestCost = cost, bestX = dx, bestY = dy;
            }
    }
    const double pixels = static_cast<double>(host.width) * host.height;
    logf("MV_STATS frame=%u size=%ux%u nonzero=%.4f bbox=%u,%u-%u,%u median=(%.3f,%.3f) mean_abs=%.3f max=%.3f "
         "image_shift_render_px=(%d,%d) match=%.3f",
         frame, host.width, host.height, nonzero / pixels, nonzero ? minX : 0, nonzero ? minY : 0, maxX, maxY,
         medianX, medianY, nonzero ? sumMagnitude / nonzero : 0.0, maxMagnitude, 2 * bestX, 2 * bestY,
         zeroCost > 0.0 ? bestCost / zeroCost : 1.0);
}

static bool upload_plane(Plane plane, const HostPlane& host, CudaImage& image, uint32_t frame, FrameTiming* timing,
                         int timingIndex)
{
    if (!ensure_cuda_image(image, host.width, host.height, plane_format(plane), plane_channels(plane), false,
                           plane == Plane::Color))
        return false;
    if ((frame <= 3 || env_uint("D4R_SHIM_INPUT_HASH_ALL", 0) != 0) && GetEnvironmentVariableA("D4R_SHIM_INPUT_HASH", nullptr, 0) != 0)
    {
        uint64_t hash = 1469598103934665603ull;
        for (size_t index = 0; index < host.size(); ++index)
            hash = (hash ^ host.bytes[index]) * 1099511628211ull;
        logf("frame %u upload plane=%u array=%p object=0x%llx bytes=%zu hash=%016llx", frame,
             static_cast<unsigned int>(plane), image.array,
             static_cast<unsigned long long>(image.object), host.size(),
             static_cast<unsigned long long>(hash));
    }
    CudaMemcpy2D copy = {};
    copy.srcMemoryType = CUDA_MEMORY_HOST;
    copy.srcHost = host.bytes;
    copy.srcPitch = host.rowBytes;
    copy.dstMemoryType = CUDA_MEMORY_ARRAY;
    copy.dstArray = image.array;
    copy.WidthInBytes = host.rowBytes;
    copy.Height = host.height;
    const auto copyStart = timing != nullptr ? ProfileClock::now() : ProfileClock::time_point{};
    const int result = g.cu.memcpy2D(&copy);
    if (timing != nullptr)
        timing->upload[timingIndex] = profile_ms(copyStart, ProfileClock::now());
    if (result != 0)
        logf("cuMemcpy2D upload failed: %d", result);
    if (result != 0)
        return false;
    if (GetEnvironmentVariableA("D4R_SHIM_SYNC_UPLOAD", nullptr, 0) != 0)
    {
        const int syncResult = g.cu.ctxSynchronize();
        if (syncResult != 0)
            logf("cuCtxSynchronize after upload failed: %d", syncResult);
        return syncResult == 0;
    }
    return true;
}

static void log_output_hash(const HostPlane& host, uint32_t frame)
{
    if (GetEnvironmentVariableA("D4R_SHIM_OUTPUT_HASH", nullptr, 0) == 0)
        return;
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < host.size(); ++index)
        hash = (hash ^ host.bytes[index]) * 1099511628211ull;
    logf("frame %u output hash=%016llx", frame, static_cast<unsigned long long>(hash));
}

static std::atomic<uint32_t> g_captureFirst{0}; // first frame of a triggered capture

static bool dump_frame_selected(uint32_t frame, const char* startName, const char* countName, const char* everyName)
{
    char trigger[MAX_PATH];
    const DWORD triggerLength = GetEnvironmentVariableA("D4R_SHIM_CAPTURE_TRIGGER", trigger, MAX_PATH);
    if (triggerLength > 0 && triggerLength < MAX_PATH)
    {
        uint32_t captured = g_captureFirst.load();
        if (captured == 0 && GetFileAttributesA(trigger) != INVALID_FILE_ATTRIBUTES)
        {
            uint32_t expected = 0;
            if (g_captureFirst.compare_exchange_strong(expected, frame))
                logf("raw input/output capture triggered on frame %u", frame);
            captured = g_captureFirst.load();
        }
        // D4R_SHIM_CAPTURE_COUNT=N keeps capturing N consecutive frames (replay sequences).
        static const uint32_t count = std::max(1u, env_uint("D4R_SHIM_CAPTURE_COUNT", 1));
        return captured != 0 && frame >= captured && frame - captured < count;
    }
    const uint32_t start = env_uint(startName, 1);
    const uint32_t count = env_uint(countName, 1);
    const uint32_t every = std::max(1u, env_uint(everyName, 1));
    return frame >= start && count != 0 && (frame - start) / every < count && (frame - start) % every == 0;
}

static bool output_dump_path(uint32_t frame, char (&path)[MAX_PATH])
{
    char directory[MAX_PATH];
    const DWORD length = GetEnvironmentVariableA("D4R_SHIM_OUTPUT_DUMP_DIR", directory, MAX_PATH);
    if (length == 0 || length >= MAX_PATH)
        return false;
    if (!dump_frame_selected(frame, "D4R_SHIM_OUTPUT_DUMP_START", "D4R_SHIM_OUTPUT_DUMP_COUNT",
                             "D4R_SHIM_OUTPUT_DUMP_EVERY"))
        return false;
    const int written = std::snprintf(path, MAX_PATH, "%s\\frame-%06u.rgba16f", directory, frame);
    return written > 0 && written < MAX_PATH;
}

// Capture files. With D4R_SHIM_CAPTURE_MEMORY=1 they are kept in memory and
// written by a background thread once the triggered capture is complete, so a
// sequence can be recorded at the game's normal frame rate.
static std::mutex g_captureMutex;
static std::vector<std::pair<std::string, std::vector<uint8_t>>> g_captureFiles;

// Pinned staging for memory captures (D4R_SHIM_CAPTURE_POOL_MB, default 3072):
// device readbacks land in it directly, so a capture costs a few ms per frame.
struct CapturePool
{
    uint8_t* base = nullptr;
    size_t size = 0, used = 0;
    bool tried = false;
};
static CapturePool g_capturePool;
static std::vector<std::pair<std::string, std::pair<const uint8_t*, size_t>>> g_capturePinned;

// Worker thread (a CUDA context is current). Returns nullptr when memory
// captures are off or the pool is exhausted; the caller then uses ordinary memory.
static uint8_t* capture_pinned(const char* path, size_t bytes)
{
    if (env_uint("D4R_SHIM_CAPTURE_MEMORY", 0) == 0 || g.cu.memAllocHost == nullptr)
        return nullptr;
    std::lock_guard<std::mutex> lock(g_captureMutex);
    CapturePool& pool = g_capturePool;
    if (!pool.tried)
    {
        pool.tried = true;
        for (size_t mb = env_uint("D4R_SHIM_CAPTURE_POOL_MB", 3072); mb >= 256 && pool.base == nullptr; mb /= 2)
        {
            void* memory = nullptr;
            if (g.cu.memAllocHost(&memory, mb << 20) == 0)
                pool.base = static_cast<uint8_t*>(memory), pool.size = mb << 20;
        }
        logf("capture pool: %zu MB pinned", pool.size >> 20);
    }
    const size_t aligned = (bytes + 4095) & ~size_t(4095);
    if (pool.base == nullptr || pool.used + aligned > pool.size)
        return nullptr;
    uint8_t* block = pool.base + pool.used;
    pool.used += aligned;
    g_capturePinned.emplace_back(path, std::make_pair(block, bytes));
    return block;
}

static void write_capture_file(uint32_t frame, const char* what, const char* path, const void* data, size_t bytes)
{
    static const bool memory = env_uint("D4R_SHIM_CAPTURE_MEMORY", 0) != 0;
    if (memory)
    {
        if (uint8_t* pinned = capture_pinned(path, bytes))
        {
            std::memcpy(pinned, data, bytes);
            return;
        }
        const auto* first = static_cast<const uint8_t*>(data);
        std::lock_guard<std::mutex> lock(g_captureMutex);
        g_captureFiles.emplace_back(path, std::vector<uint8_t>(first, first + bytes));
        return;
    }
    FILE* file = std::fopen(path, "wb");
    if (file == nullptr || std::fwrite(data, 1, bytes, file) != bytes)
        logf("frame %u %s dump failed: %s", frame, what, path);
    else
        logf("frame %u %s saved: %s", frame, what, path);
    if (file != nullptr)
        std::fclose(file);
}

// Called after each frame's output dump: once the last triggered frame is in,
// the buffered files go to disk.
static void flush_capture_if_complete(uint32_t frame)
{
    char trigger[MAX_PATH];
    if (env_uint("D4R_SHIM_CAPTURE_MEMORY", 0) == 0 ||
        GetEnvironmentVariableA("D4R_SHIM_CAPTURE_TRIGGER", trigger, MAX_PATH) == 0)
        return;
    const uint32_t count = std::max(1u, env_uint("D4R_SHIM_CAPTURE_COUNT", 1));
    const uint32_t first = g_captureFirst.load();
    if (first == 0 || frame != first + count - 1)
        return;
    std::vector<std::pair<std::string, std::vector<uint8_t>>> files;
    std::vector<std::pair<std::string, std::pair<const uint8_t*, size_t>>> pinned;
    {
        std::lock_guard<std::mutex> lock(g_captureMutex);
        files.swap(g_captureFiles);
        pinned.swap(g_capturePinned);
    }
    logf("capture of frames %u-%u complete in memory; writing %zu files", first, frame, files.size() + pinned.size());
    // The pinned pool is not reused, so its blocks stay valid for the writer.
    std::thread([files = std::move(files), pinned = std::move(pinned), first, frame] {
        size_t failed = 0;
        auto write = [&failed](const std::string& path, const uint8_t* data, size_t bytes) {
            FILE* file = std::fopen(path.c_str(), "wb");
            if (file == nullptr || std::fwrite(data, 1, bytes, file) != bytes)
                ++failed;
            if (file != nullptr)
                std::fclose(file);
        };
        for (const auto& [path, block] : pinned)
            write(path, block.first, block.second);
        for (const auto& [path, bytes] : files)
            write(path, bytes.data(), bytes.size());
        logf("capture of frames %u-%u written (%zu files, %zu failed)", first, frame, files.size() + pinned.size(),
             failed);
    }).detach();
}

static void dump_output(const HostPlane& host, uint32_t frame, const char* path)
{
    write_capture_file(frame, "raw DLSS output", path, host.bytes, host.size());
    flush_capture_if_complete(frame);
}

static bool input_dump_directory(uint32_t frame, char (&directory)[MAX_PATH])
{
    const DWORD length = GetEnvironmentVariableA("D4R_SHIM_INPUT_DUMP_DIR", directory, MAX_PATH);
    return length != 0 && length < MAX_PATH &&
           dump_frame_selected(frame, "D4R_SHIM_INPUT_DUMP_START", "D4R_SHIM_INPUT_DUMP_COUNT",
                               "D4R_SHIM_INPUT_DUMP_EVERY");
}

static void dump_input_planes(const HostPlane (&planes)[4], uint32_t frame, const FrameParams& p,
                              const char* directory)
{
    constexpr const char* names[4] = {"color.rgba16f", "depth.r32f", "motion.rg16f", "exposure.r32f"};
    for (int index = 0; index < (planes[3].width != 0 ? 4 : 3); ++index)
    {
        const HostPlane& host = planes[index];
        const uint8_t* bytes = host.bytes != nullptr ? host.bytes : host.fallback.data();
        if (bytes == nullptr || host.width == 0 || host.height == 0)
            continue;
        char path[MAX_PATH];
        const int written = std::snprintf(path, MAX_PATH, "%s\\frame-%06u-%s", directory, frame, names[index]);
        if (written > 0 && written < MAX_PATH)
            write_capture_file(frame, names[index], path, bytes, host.size());
    }
    // The evaluation parameters, so the harness can replay the sequence.
    char path[MAX_PATH];
    const int written = std::snprintf(path, MAX_PATH, "%s\\frame-%06u-params.txt", directory, frame);
    if (written <= 0 || written >= MAX_PATH)
        return;
    char text[1024];
    const int length = std::snprintf(
        text, sizeof(text),
        "jitter %.9g %.9g\nmv_scale %.9g %.9g\nsharpness %.9g\npre_exposure %.9g\n"
        "exposure_scale %.9g\nframe_time %.9g\nreset %d\ninvert %d %d\nrender %u %u\n"
        "color_base %u %u\ndepth_base %u %u\nmv_base %u %u\noutput_base %u %u\nhas_exposure %d\n",
        p.jitterX, p.jitterY, p.mvScaleX, p.mvScaleY, p.sharpness, p.preExposure, p.exposureScale, p.frameTime,
        p.reset, p.invertX, p.invertY, p.renderWidth, p.renderHeight, p.colorBaseX, p.colorBaseY, p.depthBaseX,
        p.depthBaseY, p.mvBaseX, p.mvBaseY, p.outputBaseX, p.outputBaseY, p.hasExposure ? 1 : 0);
    if (length > 0)
        write_capture_file(frame, "params", path, text, static_cast<size_t>(length));
}

// VRAM interop without D4R_SHIM_VRAM_VERIFY: reads the selected frame's input
// planes straight from the imported buffers DLSS is about to consume.
static void dump_vram_inputs(const InputSlot& slot, uint32_t frame, const FrameParams& p, int count)
{
    char directory[MAX_PATH];
    if (!input_dump_directory(frame, directory))
        return;
    HostPlane planes[4];
    constexpr const char* names[4] = {"color.rgba16f", "depth.r32f", "motion.rg16f", "exposure.r32f"};
    bool allPinned = true;
    for (int index = 0; index < count; ++index)
    {
        const HostPlane& geometry = slot.host[index];
        HostPlane& plane = planes[index];
        plane.width = geometry.width, plane.height = geometry.height, plane.rowBytes = geometry.rowBytes;
        char path[MAX_PATH];
        std::snprintf(path, MAX_PATH, "%s\\frame-%06u-%s", directory, frame, names[index]);
        plane.bytes = capture_pinned(path, geometry.size());
        if (plane.bytes == nullptr)
        {
            allPinned = false;
            plane.fallback.resize(geometry.size());
        }
        CudaMemcpy2D readback = {};
        readback.srcMemoryType = CUDA_MEMORY_DEVICE;
        readback.srcDevice = slot.vram[index].device;
        readback.srcPitch = geometry.rowBytes;
        readback.dstMemoryType = CUDA_MEMORY_HOST;
        readback.dstHost = plane.bytes != nullptr ? plane.bytes : plane.fallback.data();
        readback.dstPitch = geometry.rowBytes;
        readback.WidthInBytes = geometry.rowBytes;
        readback.Height = geometry.height;
        if (g.cu.memcpy2D(&readback) != 0)
        {
            logf("frame %u: VRAM input readback for the dump failed", frame);
            return;
        }
    }
    // Pinned planes are already registered for writing; only the rest (and the
    // parameters) go through dump_input_planes.
    for (HostPlane& plane : planes)
        if (plane.bytes != nullptr)
            plane.width = 0;
    (void)allPinned;
    dump_input_planes(planes, frame, p, directory);
}

// Worker stage: downloads the DLSS output into host memory.
static bool download_output(Feature& feature, HostPlane& host, uint32_t frame, FrameTiming* timing)
{
    reserve_host(host, static_cast<size_t>(feature.output.width) * 8 * feature.output.height);
    host.width = feature.output.width;
    host.height = feature.output.height;
    host.rowBytes = static_cast<size_t>(feature.output.width) * 8;
    CudaMemcpy2D copy = {};
    copy.srcMemoryType = CUDA_MEMORY_ARRAY;
    copy.srcArray = feature.output.array;
    copy.dstMemoryType = CUDA_MEMORY_HOST;
    copy.dstHost = host.bytes;
    copy.dstPitch = host.rowBytes;
    copy.WidthInBytes = host.rowBytes;
    copy.Height = host.height;
    const auto copyStart = timing != nullptr ? ProfileClock::now() : ProfileClock::time_point{};
    const int result = g.cu.memcpy2D(&copy);
    if (timing != nullptr)
        timing->download = profile_ms(copyStart, ProfileClock::now());
    if (result != 0)
    {
        logf("cuMemcpy2D download failed: %d", result);
        return false;
    }
    log_output_hash(host, frame);
    char dumpPath[MAX_PATH];
    if (output_dump_path(frame, dumpPath))
        dump_output(host, frame, dumpPath);
    return true;
}

// Finish stage: converts a downloaded result into an upload staging buffer.
static void write_output(Staging& staging, const HostPlane& host, FrameTiming* timing)
{
    const auto conversionStart = timing != nullptr ? ProfileClock::now() : ProfileClock::time_point{};
    uint8_t* destination = staging.mapped + staging.layout.Offset;
    const size_t destinationPitch = staging.layout.Footprint.RowPitch;
    const UINT width = staging.width < host.width ? staging.width : host.width;
    const UINT height = staging.height < host.height ? staging.height : host.height;
    const uint8_t* rows = host.bytes;
    if (staging.format == DXGI_FORMAT_R16G16B16A16_FLOAT || staging.format == DXGI_FORMAT_R16G16B16A16_TYPELESS)
        copy_rows(g_outputRows, destination, destinationPitch, rows, host.rowBytes, static_cast<size_t>(width) * 8,
                  height);
    else
        g_outputRows.run(height, host.rowBytes * height, [&](UINT begin, UINT end) {
            for (UINT y = begin; y < end; ++y)
                convert_row_out(staging.format, reinterpret_cast<const uint16_t*>(rows + host.rowBytes * y),
                                destination + y * destinationPitch, width);
        });
    if (timing != nullptr)
        timing->outputConvert = profile_ms(conversionStart, ProfileClock::now());
}

static void set_evaluation_parameters(Feature& feature, const FrameParams& p)
{
    void* params = feature.cudaParams;
    feature.colorHandle = feature.color.object;
    feature.depthHandle = feature.depth.object;
    feature.motionHandle = feature.motion.object;
    feature.exposureHandle = feature.exposure.object;
    feature.outputHandle = feature.output.object;
    // CUDA resource parameters point to variables holding the object handles.
    d4r_ngx_set_void(params, "Color", &feature.colorHandle);
    d4r_ngx_set_void(params, "Depth", &feature.depthHandle);
    d4r_ngx_set_void(params, "MotionVectors", &feature.motionHandle);
    d4r_ngx_set_void(params, "Output", &feature.outputHandle);
    d4r_ngx_set_void(params, "ExposureTexture", p.hasExposure ? &feature.exposureHandle : nullptr);
    d4r_ngx_set_void(params, "TransparencyMask", nullptr);
    d4r_ngx_set_void(params, "DLSS.Input.Bias.Current.Color.Mask", nullptr);
    d4r_ngx_set_float(params, "Jitter.Offset.X", p.jitterX);
    d4r_ngx_set_float(params, "Jitter.Offset.Y", p.jitterY);
    d4r_ngx_set_float(params, "MV.Scale.X", p.mvScaleX);
    d4r_ngx_set_float(params, "MV.Scale.Y", p.mvScaleY);
    d4r_ngx_set_float(params, "Sharpness", p.sharpness);
    // The DLSS library stamps "DLSS SDK - DO NOT DISTRIBUTE" over the output in this setup; the SDK's
    // Disable.Watermark parameter turns that off. D4R_SHIM_WATERMARK=1 keeps it.
    static const bool watermark = env_uint("D4R_SHIM_WATERMARK", 0) != 0;
    d4r_ngx_set_int(params, "Disable.Watermark", watermark ? 0 : 1);
    d4r_ngx_set_int(params, "Reset", p.reset);
    d4r_ngx_set_float(params, "DLSS.Pre.Exposure", p.preExposure);
    d4r_ngx_set_float(params, "DLSS.Exposure.Scale", p.exposureScale);
    d4r_ngx_set_float(params, "FrameTimeDeltaInMsec", p.frameTime);
    d4r_ngx_set_int(params, "DLSS.Indicator.Invert.X.Axis", p.invertX);
    d4r_ngx_set_int(params, "DLSS.Indicator.Invert.Y.Axis", p.invertY);
    d4r_ngx_set_uint(params, "DLSS.Render.Subrect.Dimensions.Width", p.renderWidth);
    d4r_ngx_set_uint(params, "DLSS.Render.Subrect.Dimensions.Height", p.renderHeight);
    d4r_ngx_set_uint(params, "DLSS.Input.Color.Subrect.Base.X", p.colorBaseX);
    d4r_ngx_set_uint(params, "DLSS.Input.Color.Subrect.Base.Y", p.colorBaseY);
    d4r_ngx_set_uint(params, "DLSS.Input.Depth.Subrect.Base.X", p.depthBaseX);
    d4r_ngx_set_uint(params, "DLSS.Input.Depth.Subrect.Base.Y", p.depthBaseY);
    d4r_ngx_set_uint(params, "DLSS.Input.MV.Subrect.Base.X", p.mvBaseX);
    d4r_ngx_set_uint(params, "DLSS.Input.MV.Subrect.Base.Y", p.mvBaseY);
    d4r_ngx_set_uint(params, "DLSS.Output.Subrect.Base.X", p.outputBaseX);
    d4r_ngx_set_uint(params, "DLSS.Output.Subrect.Base.Y", p.outputBaseY);
    if (feature.scratch != 0)
    {
        d4r_ngx_set_void(params, "Scratch", reinterpret_cast<void*>(static_cast<uintptr_t>(feature.scratch)));
        d4r_ngx_set_ull(params, "Scratch.SizeInBytes", 64ull * 1024 * 1024);
    }
}

// DLSS evaluations run as a three-stage pipeline so the CPU copies of one
// frame overlap the GPU work of another:
//   prepare_inputs (prep thread)  waits for the game's input copies and
//                                 converts the readback staging into host planes;
//   run_evaluation (CUDA worker)  uploads them, evaluates, downloads the output;
//   publish_output (finish thread) writes it into an upload staging buffer and
//                                 makes it the result evaluate() presents.
// Every NGX-core and CUDA call stays on the worker. A stage only reuses a
// buffer once the next stage has released it (hostBusy / outputHostBusy).
static void publish_output(Feature* feature, int hostIndex, uint32_t frame, FrameParams params, FrameTiming timing);
static int claim_output_slot(Feature* feature);
static void finish_publish(Feature* feature, int target, uint32_t frame, const FrameParams& params,
                           const FrameTiming& timing, ProfileClock::time_point start);

// Spins until `flag` clears; false after two seconds.
static bool wait_for_release(const std::atomic<bool>& flag)
{
    const auto start = std::chrono::steady_clock::now();
    while (flag.load())
    {
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(2))
            return false;
        d4r_sleep_us(100);
    }
    return true;
}

static void release_split_frame(Feature* feature, uint32_t frame);

// Every queued frame ends in exactly one call of this (published or dropped).
static void frame_retired(Feature* feature, uint32_t frame)
{
    if (feature->split)
        release_split_frame(feature, frame);
    feature->inFlight.fetch_sub(1);
}

// Overlapped staging, worker thread only. Uploads the staged planes with async
// copies into linear buffers on the "up" stream, then moves them into the CUDA
// arrays device-to-device (on the null stream, ahead of NGX's kernels).
static bool upload_inputs_async(Feature& feature, InputSlot& slot, bool hasExposure, uint32_t frame,
                                FrameTiming* timing)
{
    CopyStreams* streams = copy_streams();
    const Plane planes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
    CudaImage* images[4] = {&feature.color, &feature.depth, &feature.motion, &feature.exposure};
    const int count = hasExposure ? 4 : 3;
    const auto start = ProfileClock::now();
    for (int index = 0; index < count; ++index)
    {
        const HostPlane& host = slot.host[index];
        if (!ensure_cuda_image(*images[index], host.width, host.height, plane_format(planes[index]),
                               plane_channels(planes[index]), false, planes[index] == Plane::Color) ||
            !ensure_linear(feature.planeLinear[index], feature.planeLinearBytes[index], host.size()))
            return false;
        if ((frame <= 3 || env_uint("D4R_SHIM_INPUT_HASH_ALL", 0) != 0) && GetEnvironmentVariableA("D4R_SHIM_INPUT_HASH", nullptr, 0) != 0)
        {
            uint64_t hash = 1469598103934665603ull;
            for (size_t byte = 0; byte < host.size(); ++byte)
                hash = (hash ^ host.bytes[byte]) * 1099511628211ull;
            logf("frame %u upload plane=%d bytes=%zu hash=%016llx (async)", frame, index, host.size(),
                 static_cast<unsigned long long>(hash));
        }
        const auto issue = ProfileClock::now();
        const int result = g.cu.memcpyHtoDAsync(feature.planeLinear[index], host.bytes, host.size(), streams->up);
        if (timing != nullptr)
            timing->upload[index] = profile_ms(issue, ProfileClock::now());
        if (result != 0)
        {
            logf("cuMemcpyHtoDAsync failed: %d", result);
            return false;
        }
    }
    int result = g.cu.streamSynchronize(streams->up);
    for (int index = 0; index < count && result == 0; ++index)
    {
        const HostPlane& host = slot.host[index];
        CudaMemcpy2D copy = {};
        copy.srcMemoryType = CUDA_MEMORY_DEVICE;
        copy.srcDevice = feature.planeLinear[index];
        copy.srcPitch = host.rowBytes;
        copy.dstMemoryType = CUDA_MEMORY_ARRAY;
        copy.dstArray = images[index]->array;
        copy.WidthInBytes = host.rowBytes;
        copy.Height = host.height;
        result = copy_2d(copy);
    }
    if (timing != nullptr)
        timing->h2dTotal = profile_ms(start, ProfileClock::now());
    if (result != 0)
        logf("staged upload failed: %d", result);
    return result == 0;
}

// Starts the output download: array -> linear buffer, then an async copy on
// the "down" stream that the next frame's uploads overlap.
static bool start_download(Feature& feature, HostPlane& host, FrameTiming* timing)
{
    const size_t rowBytes = static_cast<size_t>(feature.output.width) * 8;
    const size_t bytes = rowBytes * feature.output.height;
    reserve_host(host, bytes);
    host.width = feature.output.width;
    host.height = feature.output.height;
    host.rowBytes = rowBytes;
    if (!ensure_linear(feature.outputLinear, feature.outputLinearBytes, bytes))
        return false;
    const auto start = ProfileClock::now();
    CudaMemcpy2D copy = {};
    copy.srcMemoryType = CUDA_MEMORY_ARRAY;
    copy.srcArray = feature.output.array;
    copy.dstMemoryType = CUDA_MEMORY_DEVICE;
    copy.dstDevice = feature.outputLinear;
    copy.dstPitch = rowBytes;
    copy.WidthInBytes = rowBytes;
    copy.Height = host.height;
    int result = feature.outputRedirected ? 0 : copy_2d(copy);
    // The null-stream copy must land before the non-blocking stream reads it.
    if (result == 0)
        result = g.cu.ctxSynchronize();
    if (result == 0)
        result = g.cu.memcpyDtoHAsync(host.bytes, feature.outputLinear, bytes, copy_streams()->down);
    if (timing != nullptr)
        timing->d2hIssue = profile_ms(start, ProfileClock::now());
    if (result != 0)
        logf("staged download failed: %d", result);
    return result == 0;
}

// A result whose download is still in flight so the next frame's uploads can
// overlap it. Worker thread only.
struct PendingDownload
{
    Feature* feature = nullptr;
    int hostIndex = 0;
    uint32_t frame = 0;
    FrameParams params;
    FrameTiming timing;
};
static PendingDownload g_pendingDownload;

static void post_publish(Feature* feature, int hostIndex, uint32_t frame, const FrameParams& params,
                         FrameTiming timing)
{
    if (timing.enabled)
        timing.finishQueued = ProfileClock::now();
    g.finish.post([feature, hostIndex, frame, params, timing] {
        publish_output(feature, hostIndex, frame, params, timing);
    });
}

// Waits for the in-flight download, if any, and hands it to publish_output.
// `inline_completion` marks the case without a next frame to overlap: the wait
// then belongs to this frame's worker time.
static void complete_pending_download(bool inline_completion = false)
{
    PendingDownload pending = g_pendingDownload;
    if (pending.feature == nullptr)
        return;
    g_pendingDownload = {};
    const auto waitStart = ProfileClock::now();
    const int result = g.cu.streamSynchronize(copy_streams()->down);
    if (pending.timing.enabled)
    {
        pending.timing.d2hWait = profile_ms(waitStart, ProfileClock::now());
        pending.timing.download = pending.timing.d2hIssue + pending.timing.d2hWait;
        if (inline_completion)
            pending.timing.worker += pending.timing.d2hWait;
    }
    if (result != 0)
    {
        logf("frame %u: output download failed: %d", pending.frame, result);
        pending.feature->outputHostBusy[pending.hostIndex] = false;
        frame_retired(pending.feature, pending.frame);
        return;
    }
    log_output_hash(pending.feature->outputHost[pending.hostIndex], pending.frame);
    char dumpPath[MAX_PATH];
    if (output_dump_path(pending.frame, dumpPath))
        dump_output(pending.feature->outputHost[pending.hostIndex], pending.frame, dumpPath);
    post_publish(pending.feature, pending.hostIndex, pending.frame, pending.params, pending.timing);
}

// Linear inputs, worker thread: pitch-linear texture objects on the slot's interop buffers (recreated
// when a buffer or its geometry changes), reported to NGX as arrays by the bridge.
static bool ensure_linear_inputs(InputSlot& slot, bool hasExposure)
{
    const Plane planes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
    const int count = hasExposure ? 4 : 3;
    for (int index = 0; index < count; ++index)
    {
        const HostPlane& geometry = slot.host[index];
        const VramBuffer& buffer = slot.vram[index];
        if (slot.linearTexture[index] != 0 && slot.linearPointer[index] == buffer.device &&
            slot.linearWidth[index] == geometry.width && slot.linearHeight[index] == geometry.height &&
            slot.linearPitch[index] == geometry.rowBytes)
            continue;
        if (slot.linearTexture[index] != 0)
            g.cu.texObjectDestroy(slot.linearTexture[index]);
        slot.linearTexture[index] = 0;
        struct Pitch2D
        {
            CudaDevicePtr pointer;
            uint32_t format, channels;
            size_t width, height, pitch;
        } pitch2D = {buffer.device, plane_format(planes[index]), plane_channels(planes[index]), geometry.width,
                     geometry.height, geometry.rowBytes};
        static_assert(sizeof(Pitch2D) == 40, "CUDA_RESOURCE_DESC pitch2D");
        CudaResourceDesc resource = {};
        resource.resType = 3; // CU_RESOURCE_TYPE_PITCH2D
        memcpy(&resource.res, &pitch2D, sizeof(pitch2D));
        const CudaTextureDesc sampler = input_sampler(geometry.width, pitch2D.channels, planes[index] == Plane::Color);
        CudaObject object = 0;
        int result = g.cu.texObjectCreate(&object, &resource, &sampler, nullptr);
        if (result == 0)
            result = g.cu.registerLinearTexture(object, geometry.width, geometry.height, pitch2D.format, pitch2D.channels);
        if (result != 0)
        {
            logf("linear input texture %d (%ux%u, pitch %zu) failed: %d", index, geometry.width, geometry.height,
                 geometry.rowBytes, result);
            if (object != 0)
                g.cu.texObjectDestroy(object);
            return false;
        }
        slot.linearTexture[index] = object;
        slot.linearPointer[index] = buffer.device;
        slot.linearWidth[index] = geometry.width;
        slot.linearHeight[index] = geometry.height;
        slot.linearPitch[index] = geometry.rowBytes;
    }
    return true;
}

// VRAM interop, worker thread: moves the planes from the imported buffers
// Opt-in thin-feature motion coverage. Keep the original imported velocity buffer
// immutable; the new texture is owned by this input slot until its GPU work retires.
static CudaObject prepare_dilated_motion(Feature& feature, InputSlot& slot, const FrameParams& p)
{
    const unsigned radius = env_uint("D4R_MOTION_DILATION", 0);
    if (radius == 0)
        return 0;
    const auto& depth = slot.host[1];
    const auto& motion = slot.host[2];
    D4rMotionDilationParams args = {slot.vram[1].device, slot.vram[2].device, slot.dilatedMotion,
        depth.rowBytes, motion.rowBytes, p.renderWidth, p.renderHeight, motion.width, motion.height,
        p.jitterX, p.jitterY, radius, (feature.flags & 8u) != 0};
    const bool motionGeometry = d4r_motion_dilation_geometry(&args, feature.flags,
                                                            feature.outWidth, feature.outHeight);
    const bool supported = radius <= 2 && p.vram && feature.linearInputs && g.cu.dilateMotion != nullptr &&
        motionGeometry && p.colorBaseX == 0 && p.colorBaseY == 0 && p.depthBaseX == 0 &&
        p.depthBaseY == 0 && p.mvBaseX == 0 && p.mvBaseY == 0 && p.outputBaseX == 0 && p.outputBaseY == 0 &&
        p.renderWidth > 0 && p.renderHeight > 0 && std::isfinite(p.jitterX) && std::isfinite(p.jitterY) &&
        p.renderWidth <= depth.width && p.renderHeight <= depth.height;
    if (!supported)
    {
        if (!feature.dilationReported)
            logf("motion-vector dilation unavailable for this feature; original motion vectors retained");
        feature.dilationReported = true;
        return 0;
    }
    const size_t bytes = motion.size();
    const bool recreate = slot.dilatedBytes < bytes || slot.dilatedWidth != motion.width ||
        slot.dilatedHeight != motion.height || slot.dilatedPitch != motion.rowBytes;
    if (recreate)
    {
        if (slot.dilatedTexture != 0)
            g.cu.texObjectDestroy(slot.dilatedTexture), slot.dilatedTexture = 0;
        if (slot.dilatedMotion != 0)
            g.cu.memFree(slot.dilatedMotion), slot.dilatedMotion = 0;
        slot.dilatedBytes = 0;
        if (g.cu.memAlloc(&slot.dilatedMotion, bytes) != 0)
            return 0;
        slot.dilatedBytes = bytes;
        struct Pitch2D { CudaDevicePtr pointer; uint32_t format, channels; size_t width, height, pitch; };
        const Pitch2D pitch = {slot.dilatedMotion, plane_format(Plane::Motion), 2, motion.width, motion.height, motion.rowBytes};
        CudaResourceDesc resource = {};
        resource.resType = 3;
        static_assert(sizeof(pitch) == 40);
        std::memcpy(&resource.res, &pitch, sizeof(pitch));
        const CudaTextureDesc sampler = input_sampler(motion.width, 2, false);
        int result = g.cu.texObjectCreate(&slot.dilatedTexture, &resource, &sampler, nullptr);
        if (result == 0)
            result = g.cu.registerLinearTexture(slot.dilatedTexture, motion.width, motion.height, pitch.format, 2);
        if (result != 0)
        {
            if (slot.dilatedTexture != 0)
                g.cu.texObjectDestroy(slot.dilatedTexture), slot.dilatedTexture = 0;
            return 0;
        }
        slot.dilatedWidth = motion.width, slot.dilatedHeight = motion.height, slot.dilatedPitch = motion.rowBytes;
    }
    if (slot.dilatedTexture == 0)
        return 0;
    args.output = slot.dilatedMotion;
    // Preserve allocation padding, which DLSS may sample at the active edge.
    // The dilation kernel only writes the active render grid for low-res MVs.
    if (args.motion_width != motion.width || args.motion_height != motion.height)
    {
        CudaMemcpy2D copy = {};
        copy.srcMemoryType = copy.dstMemoryType = CUDA_MEMORY_DEVICE;
        copy.srcDevice = args.motion;
        copy.dstDevice = args.output;
        copy.srcPitch = copy.dstPitch = motion.rowBytes;
        copy.WidthInBytes = motion.rowBytes;
        copy.Height = motion.height;
        if (copy_2d(copy) != 0)
            return 0;
    }
    const int result = g.cu.dilateMotion(&args);
    if (!feature.dilationReported)
        logf("motion-vector dilation %s: radius=%u render pixels, GPU status=%d, %s active=%ux%u allocation=%ux%u",
             result == 0 ? "enabled" : "unavailable", radius, result,
             (feature.flags & 2u) != 0 ? "low-resolution" : "display-resolution",
             args.motion_width, args.motion_height, motion.width, motion.height);
    feature.dilationReported = true;
    return result == 0 ? slot.dilatedTexture : 0;
}

// VRAM interop, worker thread: moves the planes from the imported buffers
// into the CUDA arrays on the GPU. With D4R_SHIM_VRAM_VERIFY also compares
// them with the host-staged planes of the same frame.
static bool upload_inputs_vram(Feature& feature, InputSlot& slot, bool hasExposure, uint32_t frame,
                               FrameTiming* timing)
{
    const Plane planes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
    CudaImage* images[4] = {&feature.color, &feature.depth, &feature.motion, &feature.exposure};
    const int count = hasExposure ? 4 : 3;
    const bool verify = env_uint("D4R_SHIM_VRAM_VERIFY", 0) != 0;
    const auto start = ProfileClock::now();
    int result = 0;
    for (int index = 0; index < count && result == 0; ++index)
    {
        const HostPlane& geometry = slot.host[index];
        if (!ensure_cuda_image(*images[index], geometry.width, geometry.height, plane_format(planes[index]),
                               plane_channels(planes[index]), false, planes[index] == Plane::Color))
            return false;
        if (verify)
        {
            std::vector<uint8_t> vram(geometry.size());
            CudaMemcpy2D readback = {};
            readback.srcMemoryType = CUDA_MEMORY_DEVICE;
            readback.srcDevice = slot.vram[index].device;
            readback.srcPitch = geometry.rowBytes;
            readback.dstMemoryType = CUDA_MEMORY_HOST;
            readback.dstHost = vram.data();
            readback.dstPitch = geometry.rowBytes;
            readback.WidthInBytes = geometry.rowBytes;
            readback.Height = geometry.height;
            size_t differing = 0;
            if (g.cu.memcpy2D(&readback) == 0)
                for (size_t byte = 0; byte < vram.size(); ++byte)
                    differing += vram[byte] != geometry.bytes[byte];
            if (differing != 0 || frame <= 3 || frame % 120 == 0)
                logf("frame %u VRAM verify plane=%d: %zu of %zu bytes differ from host staging", frame, index,
                     differing, vram.size());
        }
        CudaMemcpy2D copy = {};
        copy.srcMemoryType = CUDA_MEMORY_DEVICE;
        copy.srcDevice = slot.vram[index].device;
        copy.srcPitch = geometry.rowBytes;
        copy.dstMemoryType = CUDA_MEMORY_ARRAY;
        copy.dstArray = images[index]->array;
        copy.WidthInBytes = geometry.rowBytes;
        copy.Height = geometry.height;
        result = copy_2d(copy);
        // D4R_SHIM_VRAM_ARRAY_VERIFY=1: read the CUDA array DLSS samples back
        // and compare it with the imported buffer it was filled from.
        static const bool arrayVerify = env_uint("D4R_SHIM_VRAM_ARRAY_VERIFY", 0) != 0;
        if (arrayVerify && result == 0 && g.cu.ctxSynchronize() == 0)
        {
            std::vector<uint8_t> fromBuffer(geometry.size()), fromArray(geometry.size());
            CudaMemcpy2D readBuffer = {};
            readBuffer.srcMemoryType = CUDA_MEMORY_DEVICE;
            readBuffer.srcDevice = slot.vram[index].device;
            readBuffer.srcPitch = geometry.rowBytes;
            readBuffer.dstMemoryType = CUDA_MEMORY_HOST;
            readBuffer.dstHost = fromBuffer.data();
            readBuffer.dstPitch = geometry.rowBytes;
            readBuffer.WidthInBytes = geometry.rowBytes;
            readBuffer.Height = geometry.height;
            CudaMemcpy2D readArray = {};
            readArray.srcMemoryType = CUDA_MEMORY_ARRAY;
            readArray.srcArray = images[index]->array;
            readArray.dstMemoryType = CUDA_MEMORY_HOST;
            readArray.dstHost = fromArray.data();
            readArray.dstPitch = geometry.rowBytes;
            readArray.WidthInBytes = geometry.rowBytes;
            readArray.Height = geometry.height;
            if (g.cu.memcpy2D(&readBuffer) == 0 && g.cu.memcpy2D(&readArray) == 0)
            {
                size_t differing = 0, firstRow = geometry.height;
                for (size_t byte = 0; byte < fromArray.size(); ++byte)
                    if (fromArray[byte] != fromBuffer[byte])
                    {
                        if (differing++ == 0)
                            firstRow = byte / geometry.rowBytes;
                    }
                if (differing != 0 || frame <= 3 || frame % 120 == 0)
                    logf("frame %u VRAM array verify plane=%d: %zu of %zu bytes differ (first row %zu)", frame, index,
                         differing, fromArray.size(), firstRow);
            }
        }
    }
    if (timing != nullptr)
        timing->h2dTotal = profile_ms(start, ProfileClock::now());
    if (result != 0)
        logf("VRAM input copy failed: %d", result);
    return result == 0;
}

// Only called on the CUDA worker. A completion event covers queued work on
// the default stream; do not rely on HIP's blocking flag to release the CPU.
static int synchronize_default_stream(Feature& feature, bool* querySleep = nullptr, double* waitEmaUs = nullptr)
{
    if (feature.outputReadyEvent != nullptr)
    {
        const int recorded = g.cu.eventRecord(feature.outputReadyEvent, nullptr);
        if (recorded == 0)
        {
            // D4R_SHIM_OUTPUT_POLL_US: N > 0 polls every N us (200 was the fixed interval before), 0 yields between
            // queries (spins). Unset: adaptive - sleep until shortly before the predicted completion (smoothed
            // wait of earlier frames), then poll every 20 us, backing off to 200 us when the GPU runs late.
            static const int fixedPoll = [] {
                const char* v = std::getenv("D4R_SHIM_OUTPUT_POLL_US");
                return v != nullptr && *v != '\0' ? static_cast<int>(env_uint("D4R_SHIM_OUTPUT_POLL_US", 200)) : -1;
            }();
            const auto waitStart = std::chrono::steady_clock::now();
            unsigned polls = 0;
            double lastNotReadyUs = -1.0; // elapsed time of the last query that found the work unfinished
            const int result = d4r_wait_event(
                [&] { return g.cu.eventQuery(feature.outputReadyEvent); },
                [&] {
                    if (fixedPoll >= 0)
                    {
                        d4r_sleep_us(static_cast<unsigned>(fixedPoll));
                        return;
                    }
                    constexpr double kMarginUs = 120.0, kStepUs = 20.0, kChunkUs = 200.0, kSpinAfterUs = 150.0;
                    const double elapsed = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - waitStart).count();
                    lastNotReadyUs = elapsed; // this sleep follows a query that returned NOT_READY
                    ++polls;
                    const double predicted = waitEmaUs != nullptr ? *waitEmaUs : 0.0;
                    const double remaining = predicted - kMarginUs - elapsed;
                    // far from the predicted completion: sleep in chunks of at most 200 us (a single long sleep
                    // can overshoot by far more under Wine); near it: 20 us polls; late: back off to 200 us
                    if (remaining > kStepUs)
                        d4r_sleep_us(static_cast<unsigned>(std::min(remaining, kChunkUs)));
                    else if (elapsed < predicted + kSpinAfterUs)
                        d4r_sleep_us(0); // final stretch around the predicted completion: spin (SwitchToThread);
                                         // a Wine sleep rounds up to ~50 us, which would be the detection latency
                    else
                        d4r_sleep_us(static_cast<unsigned>(std::min(
                            kStepUs * (1.0 + (elapsed - predicted - kSpinAfterUs) / 160.0), kChunkUs)));
                });
            if (result == 0 && fixedPoll < 0 && waitEmaUs != nullptr)
            {
                const double waited = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - waitStart).count();
                // Learn the completion time, not our own wait: the work finished between the last NOT_READY query
                // and the successful one (at most one 200 us chunk apart), so the midpoint cannot ratchet upwards
                double sample = lastNotReadyUs >= 0.0 ? 0.5 * (lastNotReadyUs + waited) : waited;
                sample = std::min(sample, 20000.0);
                *waitEmaUs = *waitEmaUs == 0.0 ? sample : 0.8 * *waitEmaUs + 0.2 * sample;
            }
            if (result == 0)
            {
                if (querySleep != nullptr) *querySleep = true;
                // The event is already complete; this also flushes bridge kernel profiles.
                return g.cu.eventSynchronize(feature.outputReadyEvent);
            }
            logf("output event query failed (%d); using context synchronization", result);
        }
        else
            logf("output event record failed (%d); using context synchronization", recorded);
    }
    return g.cu.ctxSynchronize();
}

// A frame recorded for packed output whose output kernel then did not pack it (it was not the native one
// after all): the RGBA16F result is converted on the host, rounding as the kernel store and the game-side
// blit do. Slow, and only expected around a kernel change.
static int pack_output_on_host(Feature& feature, CudaMemcpy2D copy, size_t packedPitch, uint32_t frame)
{
    const size_t width = feature.output.width, height = feature.output.height;
    std::vector<uint16_t> halves(width * height * 4);
    std::vector<uint32_t> packed(width * height);
    const CudaDevicePtr target = copy.dstDevice;
    copy.dstMemoryType = CUDA_MEMORY_HOST;
    copy.dstHost = halves.data();
    copy.dstPitch = width * 8;
    int result = g.cu.memcpy2D(&copy);
    if (result != 0)
        return result;
    const auto unorm = [](uint16_t half, float scale) {
        float value = half_to_float(half);
        value = value > 0.0f ? value : 0.0f;
        value = value < 1.0f ? value : 1.0f;
        return static_cast<uint32_t>(value * scale + 0.5f);
    };
    for (size_t texel = 0; texel < packed.size(); ++texel)
        packed[texel] = unorm(halves[texel * 4], 1023.0f) | unorm(halves[texel * 4 + 1], 1023.0f) << 10 |
                        unorm(halves[texel * 4 + 2], 1023.0f) << 20 | unorm(halves[texel * 4 + 3], 3.0f) << 30;
    CudaMemcpy2D upload = {};
    upload.srcMemoryType = CUDA_MEMORY_HOST;
    upload.srcHost = packed.data();
    upload.srcPitch = width * 4;
    upload.dstMemoryType = CUDA_MEMORY_DEVICE;
    upload.dstDevice = target;
    upload.dstPitch = packedPitch;
    upload.WidthInBytes = width * 4;
    upload.Height = height;
    result = g.cu.memcpy2D(&upload);
    logf("frame %u: output kernel did not pack its result; converted on the host (%d)", frame, result);
    return result;
}

// VRAM interop, worker thread: copies the finished result into a free output
// slot's imported buffer and publishes it; evaluate() copies it on to the game.
static void publish_vram(Feature* feature, uint32_t frame, const FrameParams& params, FrameTiming& timing,
                         ProfileClock::time_point workerStart)
{
    const auto start = ProfileClock::now();
    // A split frame's second half reads slot frame % kOutputSlots; its previous
    // reader, frame - kOutputSlots, ran before this frame's marker.
    const int target = params.split ? static_cast<int>(frame % kOutputSlots) : claim_output_slot(feature);
    if (target < 0)
    {
        logf("frame %u: no free output slot; dropping result", frame);
        frame_retired(feature, frame);
        return;
    }
    VramBuffer buffer = feature->outputs[target].vram;
    const size_t rowBytes = static_cast<size_t>(feature->output.width) * 8;
    CudaMemcpy2D copy = {};
    copy.srcMemoryType = CUDA_MEMORY_ARRAY;
    copy.srcArray = feature->output.array;
    copy.dstMemoryType = CUDA_MEMORY_DEVICE;
    copy.dstDevice = buffer.device;
    copy.dstPitch = rowBytes;
    copy.WidthInBytes = rowBytes;
    copy.Height = feature->output.height;
    if (params.outputInPlace != 0)
    {
        // no game-side copy was recorded: a result the output kernel did not redirect goes to the texture here
        buffer.device = copy.dstDevice = params.outputInPlace;
        copy.dstPitch = params.outputInPlacePitch;
        buffer.bytes = static_cast<size_t>(params.outputInPlacePitch) * feature->output.height;
    }
    int result = rowBytes / (params.packedOutput ? 2 : 1) * feature->output.height > buffer.bytes ? -1
                 : feature->outputRedirected ? 0
                 : params.packedOutput ? pack_output_on_host(*feature, copy, params.outputInPlace != 0 ? copy.dstPitch
                                                                               : rowBytes / 2, frame)
                 : copy_2d(copy);
    if (result == 0)
    {
        const auto syncStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
        const double cpuStart = timing.enabled ? profile_thread_cpu_ms() : -1.0;
        result = synchronize_default_stream(*feature, &timing.outputSyncBlocking, &feature->publishWaitEmaUs);
        if (timing.enabled)
        {
            timing.outputSyncWall = profile_ms(syncStart, ProfileClock::now());
            const double cpuEnd = profile_thread_cpu_ms();
            if (cpuStart >= 0.0 && cpuEnd >= 0.0)
                timing.outputSyncCpu = std::max(0.0, cpuEnd - cpuStart);
        }
    }
    if (result == 0 && timing.gpuEventsRecorded)
    {
        float gpuEvalMs = -1.0f;
        if (g.cu.eventElapsedTime(&gpuEvalMs, feature->profileStart, feature->profileEnd) == 0)
            timing.gpuEval = gpuEvalMs;
    }
    if (result != 0)
    {
        logf("frame %u: VRAM output copy failed: %d", frame, result);
        std::lock_guard<std::mutex> lock(feature->outputMutex);
        feature->outputs[target].lastReadFrame = 0;
        frame_retired(feature, frame);
        return;
    }
    static const bool arrayVerify = env_uint("D4R_SHIM_VRAM_ARRAY_VERIFY", 0) != 0;
    if (arrayVerify)
    {
        // The result as DLSS left it in its array vs the buffer the game copies from.
        std::vector<uint8_t> fromArray(rowBytes * feature->output.height), fromBuffer(fromArray.size());
        CudaMemcpy2D readArray = copy;
        readArray.dstMemoryType = CUDA_MEMORY_HOST;
        readArray.dstHost = fromArray.data();
        CudaMemcpy2D readBuffer = {};
        readBuffer.srcMemoryType = CUDA_MEMORY_DEVICE;
        readBuffer.srcDevice = buffer.device;
        readBuffer.srcPitch = rowBytes;
        readBuffer.dstMemoryType = CUDA_MEMORY_HOST;
        readBuffer.dstHost = fromBuffer.data();
        readBuffer.dstPitch = rowBytes;
        readBuffer.WidthInBytes = rowBytes;
        readBuffer.Height = feature->output.height;
        if (g.cu.memcpy2D(&readArray) == 0 && g.cu.memcpy2D(&readBuffer) == 0)
        {
            size_t differing = 0;
            for (size_t byte = 0; byte < fromArray.size(); ++byte)
                differing += fromArray[byte] != fromBuffer[byte];
            if (differing != 0 || frame <= 3 || frame % 120 == 0)
                logf("frame %u VRAM output verify: %zu of %zu bytes differ between result array and buffer", frame,
                     differing, fromArray.size());
        }
    }
    char dumpPath[MAX_PATH];
    const bool saveRaw = output_dump_path(frame, dumpPath);
    if (GetEnvironmentVariableA("D4R_SHIM_OUTPUT_HASH", nullptr, 0) != 0 || saveRaw)
    {
        HostPlane host;
        uint8_t* pinned = saveRaw ? capture_pinned(dumpPath, rowBytes * feature->output.height) : nullptr;
        if (pinned == nullptr)
            host.fallback.resize(rowBytes * feature->output.height);
        host.bytes = pinned != nullptr ? pinned : host.fallback.data();
        host.width = feature->output.width;
        host.height = feature->output.height;
        host.rowBytes = rowBytes;
        copy.dstMemoryType = CUDA_MEMORY_HOST;
        copy.dstHost = host.bytes;
        if (g.cu.memcpy2D(&copy) == 0)
        {
            log_output_hash(host, frame);
            if (saveRaw && pinned != nullptr)
                flush_capture_if_complete(frame); // already registered for writing
            else if (saveRaw)
                dump_output(host, frame, dumpPath);
        }
        else if (saveRaw)
            logf("frame %u raw DLSS output readback failed", frame);
    }
    if (timing.enabled)
    {
        timing.d2hIssue = profile_ms(start, ProfileClock::now());
        timing.download = timing.d2hIssue;
        timing.worker = profile_ms(workerStart, ProfileClock::now());
    }
    finish_publish(feature, target, frame, params, timing, start);
}

// Worker. Streamline can create the feature at one quality's render size while the game still renders at
// another's (PRAGMATA starts with an Ultra Performance 854x480 frame on a Performance 1280x720 feature), and NGX
// rejects a subrect outside the feature's dynamic range on every frame until the game recreates it: a black
// screen. Recreate the NGX feature at the size actually rendered; the game's handle and output stay the same.
static bool recreate_for_render_size(Feature& feature, FrameParams& params, uint32_t frame)
{
    if ((params.renderWidth == feature.cudaWidth && params.renderHeight == feature.cudaHeight) ||
        (params.renderWidth == feature.refusedWidth && params.renderHeight == feature.refusedHeight))
        return false;
    logf("frame %u: NGX rejected render %ux%u on a %ux%u feature; recreating it at %ux%u", frame, params.renderWidth,
         params.renderHeight, feature.cudaWidth, feature.cudaHeight, params.renderWidth, params.renderHeight);
    // earlier frames' kernels may still be running on the old feature
    synchronize_default_stream(feature);
    if (feature.cudaHandle != nullptr)
        g.ngx.releaseFeature(feature.cudaHandle);
    feature.cudaHandle = nullptr;
    void* p = feature.cudaParams;
    d4r_ngx_set_uint(p, "Width", params.renderWidth);
    d4r_ngx_set_uint(p, "Height", params.renderHeight);
    NgxResult created = g.ngx.createFeature(NGX_FEATURE_SUPER_SAMPLING, p, &feature.cudaHandle);
    unsigned int width = params.renderWidth, height = params.renderHeight;
    if (created != NGX_SUCCESS)
    {
        logf("frame %u: recreating the NGX feature at %ux%u failed (0x%08x); restoring %ux%u", frame, width, height,
             created, feature.cudaWidth, feature.cudaHeight);
        width = feature.cudaWidth;
        height = feature.cudaHeight;
        d4r_ngx_set_uint(p, "Width", width);
        d4r_ngx_set_uint(p, "Height", height);
        if (width != 0 && height != 0)
            created = g.ngx.createFeature(NGX_FEATURE_SUPER_SAMPLING, p, &feature.cudaHandle);
        feature.refusedWidth = created == NGX_SUCCESS ? params.renderWidth : 0;
        feature.refusedHeight = created == NGX_SUCCESS ? params.renderHeight : 0;
        if (created != NGX_SUCCESS)
        {
            logf("frame %u: restoring the NGX feature failed (0x%08x)", frame, created);
            feature.cudaHandle = nullptr;
            width = height = 0;
        }
    }
    // a new feature has no history
    feature.cudaWidth = width;
    feature.cudaHeight = height;
    params.reset = 1;
    d4r_ngx_set_int(p, "Reset", 1);
    return feature.cudaHandle != nullptr && width == params.renderWidth && height == params.renderHeight;
}

static bool wait_for_input_marker(Feature* feature, int slotIndex, uint32_t frame);

static void run_evaluation(Feature* feature, int slotIndex, uint32_t frame, FrameParams params, FrameTiming timing)
{
    InputSlot& slot = feature->inputs[slotIndex];
    const auto start = ProfileClock::now();
    if (timing.enabled)
        timing.workerWait = profile_ms(timing.workerQueued, start);
    FrameTiming* stages = timing.enabled ? &timing : nullptr;
    const bool overlapped = async_copies() && copy_streams() != nullptr;
    if (timing.markerDeferred)
    {
        const auto markerStart = ProfileClock::now();
        if (!wait_for_input_marker(feature, slotIndex, frame))
            return;
        if (timing.enabled)
            timing.markerWait = profile_ms(markerStart, ProfileClock::now());
    }
    if (params.vram && env_uint("D4R_SHIM_VRAM_VERIFY", 0) == 0)
        dump_vram_inputs(slot, frame, params, params.hasExposure ? 4 : 3);
    const bool linearInputs = params.vram && feature->linearInputs;
    bool ok = linearInputs ? ensure_linear_inputs(slot, params.hasExposure)
              : params.vram ? upload_inputs_vram(*feature, slot, params.hasExposure, frame, stages)
              : overlapped
                  ? upload_inputs_async(*feature, slot, params.hasExposure, frame, stages)
                  : upload_plane(Plane::Color, slot.host[0], feature->color, frame, stages, 0) &&
                        upload_plane(Plane::Depth, slot.host[1], feature->depth, frame, stages, 1) &&
                        upload_plane(Plane::Motion, slot.host[2], feature->motion, frame, stages, 2) &&
                        (!params.hasExposure ||
                         upload_plane(Plane::Exposure, slot.host[3], feature->exposure, frame, stages, 3));
    ok = ok && ensure_cuda_image(feature->output, feature->outputWidth, feature->outputHeight, CUDA_FORMAT_HALF, 4,
                                 true, false);
    const auto dilationStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    const CudaObject dilatedMotion = ok ? prepare_dilated_motion(*feature, slot, params) : 0;
    // Device-to-array copies (VRAM interop, staged uploads) return before they
    // finish, and NGX does not evaluate on the stream they were issued on, so
    // without this DLSS can sample a mix of this frame's and the previous
    // frame's colour/depth/motion: invisible when the camera is still, beaded
    // thin geometry in motion. D4R_SHIM_INPUT_SYNC=0 restores the old behaviour.
    static const bool inputSync = env_uint("D4R_SHIM_INPUT_SYNC", 1) != 0;
    // Linear inputs are already ready; other copies need completion before NGX samples them.
    if (ok && (dilatedMotion != 0 || (inputSync && !linearInputs)))
    {
        const int syncResult = synchronize_default_stream(*feature);
        if (syncResult != 0)
        {
            logf("frame %u: input copy synchronize failed: %d", frame, syncResult);
            ok = false;
        }
    }
    if (dilatedMotion != 0 && ok)
    {
        params.motionDilation = env_uint("D4R_MOTION_DILATION", 0);
        if (timing.enabled)
            timing.motionDilationWall = profile_ms(dilationStart, ProfileClock::now());
    }
    slot.hostBusy = false; // the host planes are no longer needed
    // The previous frame's download ran alongside these uploads.
    complete_pending_download();
    if (!ok)
    {
        slot.busy = false;
        frame_retired(feature, frame);
        return;
    }

    // The first frame NGX evaluates for a feature starts its history. Decided
    // here, in evaluation order, so it does not depend on thread timing.
    if (feature->evaluatedFrames++ == 0)
        params.reset = 1;
    set_evaluation_parameters(*feature, params);
    if (linearInputs)
    {
        // the parameters point at these variables
        feature->colorHandle = slot.linearTexture[0];
        feature->depthHandle = slot.linearTexture[1];
        feature->motionHandle = dilatedMotion != 0 ? dilatedMotion : slot.linearTexture[2];
        feature->exposureHandle = params.hasExposure ? slot.linearTexture[3] : feature->exposureHandle;
    }
    static const bool outputDirect = env_uint("D4R_SHIM_OUTPUT_DIRECT", 0) != 0;
    // The redirect is carried out by the native output kernel's stores (ZLUDA's compile of NVIDIA's kernel
    // writes the array), so it is set while the previous frame's output kernel was the native one and
    // undone below when this frame's was not.
    const bool nativeOutput = g.cu.outputKernelNative == nullptr || g.cu.outputKernelNative() == 1;
    if (outputDirect && feature->outputDirectAllowed && g.cu.setArrayRedirect != nullptr && feature->output.array != nullptr &&
        (nativeOutput || feature->outputRedirected))
    {
        // split frames write their own slot; the staged path downloads from outputLinear
        size_t rowBytes = static_cast<size_t>(feature->output.width) * 8;
        const size_t bytes = rowBytes * feature->output.height;
        CudaDevicePtr target = 0;
        if (params.packedOutput)
            rowBytes = static_cast<size_t>(feature->output.width) * 4;
        if (params.vram && params.split && params.outputInPlace != 0)
        {
            target = params.outputInPlace;
            rowBytes = params.outputInPlacePitch;
        }
        else if (params.vram && params.split && feature->outputs[frame % kOutputSlots].vram.bytes >= bytes)
            target = feature->outputs[frame % kOutputSlots].vram.device;
        if (params.packedOutput)
            rowBytes |= 1; // the bridge's flag for R10G10B10A2 texels (row pitches are multiples of 8)
        else if (!params.vram && overlapped && ensure_linear(feature->outputLinear, feature->outputLinearBytes, bytes))
            target = feature->outputLinear;
        const bool redirect = nativeOutput && target != 0 &&
                              g.cu.setArrayRedirect(feature->output.array, target, static_cast<uint32_t>(rowBytes)) == 0;
        if (!redirect && feature->outputRedirected)
            g.cu.setArrayRedirect(feature->output.array, 0, 0);
        static bool logged = false;
        if (redirect && !logged)
        {
            logf("frame %u: direct output (the native output kernel writes the shared buffer; no array copy)", frame);
            logged = true;
        }
        feature->outputRedirected = redirect;
    }
    const int eventStartResult = timing.enabled && feature->profileEventsReady
                                     ? g.cu.eventRecord(feature->profileStart, nullptr) : -1;
    const auto evalStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    NgxResult result = g.ngx.evaluateFeature(feature->cudaHandle, feature->cudaParams, nullptr);
    if (result == NGX_FAIL_INVALID_PARAMETER && recreate_for_render_size(*feature, params, frame))
        result = g.ngx.evaluateFeature(feature->cudaHandle, feature->cudaParams, nullptr);
    if (feature->outputRedirected && g.cu.outputKernelNative != nullptr && g.cu.outputKernelNative() != 1)
    {
        // this frame's output kernel ignored the redirect: publish from the array as usual
        g.cu.setArrayRedirect(feature->output.array, 0, 0);
        feature->outputRedirected = false;
    }
    feature->packedReady = feature->outputRedirected;
    const auto evalReturned = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    const int eventEndResult = eventStartResult == 0 ? g.cu.eventRecord(feature->profileEnd, nullptr) : -1;
    const auto syncStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    // D4R_SHIM_EVAL_SYNC=0 (VRAM interop): no wait between NGX's kernels and the output copy queued
    // behind them on the same stream; publish_vram synchronises before the result is released.
    static const bool evalSync = env_uint("D4R_SHIM_EVAL_SYNC", 1) != 0;
    const int syncResult = params.vram && !evalSync ? 0 : synchronize_default_stream(*feature, nullptr, &feature->evalWaitEmaUs);
    const auto evaluated = ProfileClock::now();
    const bool gpuEventsRecorded = eventStartResult == 0 && eventEndResult == 0;
    float gpuEvalMs = -1.0f;
    if (gpuEventsRecorded && syncResult == 0 && (!params.vram || evalSync) &&
        g.cu.eventElapsedTime(&gpuEvalMs, feature->profileStart, feature->profileEnd) != 0)
        gpuEvalMs = -1.0f;
    // Keep the input slot owned until output completion, including linear reads.
    if (result != NGX_SUCCESS || syncResult != 0)
    {
        logf("frame %u: CUDA_EvaluateFeature -> 0x%08x, sync %d", frame, result, syncResult);
        slot.busy = false;
        frame_retired(feature, frame);
        return;
    }
    if (timing.enabled)
    {
        timing.ngxHost = profile_ms(evalStart, evalReturned);
        timing.ctxSync = profile_ms(syncStart, evaluated);
        timing.gpuEval = gpuEvalMs;
        timing.gpuEventsRecorded = gpuEventsRecorded && params.vram && !evalSync;
    }
    if (params.vram)
    {
        publish_vram(feature, frame, params, timing, start);
        slot.busy = false; // output synchronization completed all reads of this input slot
        return;
    }

    const int hostIndex = static_cast<int>(feature->nextOutputHost++ % 2);
    if (!wait_for_release(feature->outputHostBusy[hostIndex]))
    {
        logf("frame %u: output host buffer still busy; dropping result", frame);
        frame_retired(feature, frame);
        return;
    }
    if (overlapped)
    {
        if (!start_download(*feature, feature->outputHost[hostIndex], stages))
        {
            frame_retired(feature, frame);
            return;
        }
        feature->outputHostBusy[hostIndex] = true;
        if (timing.enabled)
            timing.worker = profile_ms(start, ProfileClock::now());
        g_pendingDownload = {feature, hostIndex, frame, params, timing};
        // Without a queued frame to overlap, finish it now rather than add latency.
        if (g.worker.pending() == 0)
            complete_pending_download(true);
        return;
    }
    if (!download_output(*feature, feature->outputHost[hostIndex], frame, stages))
    {
        frame_retired(feature, frame);
        return;
    }
    feature->outputHostBusy[hostIndex] = true;
    if (timing.enabled)
        timing.worker = profile_ms(start, ProfileClock::now());
    post_publish(feature, hostIndex, frame, params, timing);
}

// Waits for this frame's own input marker (the GPU reached the game's input copies). false: the frame was
// dropped (retired feature, skipped submission, timeout) and has been retired here.
static bool wait_for_input_marker(Feature* feature, int slotIndex, uint32_t frame)
{
    InputSlot& slot = feature->inputs[slotIndex];
    const auto start = std::chrono::steady_clock::now();
    // A later input marker does not prove this slot's copies were submitted:
    // the game may have discarded this frame's command list. Require its own
    // tag, otherwise stale input data can enter DLSS's temporal history.
    const volatile uint32_t* inputMarker = feature->markerValue + 1 + slotIndex;
    while (*inputMarker != frame)
    {
        std::atomic_thread_fence(std::memory_order_acquire);
        const bool skipped = static_cast<int32_t>(*feature->markerValue - frame) > 0;
        // The GPU may have filled our slot between the loop condition and
        // the aggregate-marker read. Recheck before deciding it was skipped.
        if (*inputMarker == frame)
            break;
        if (feature->retiring.load() || skipped || std::chrono::steady_clock::now() - start > std::chrono::seconds(5))
        {
            logf("frame %u: %s waiting for GPU marker (at %u); dropping", frame,
                 feature->retiring.load() ? "feature retired while" : skipped ? "input submission skipped while" : "timed out",
                 *inputMarker);
            slot.busy = false;
            frame_retired(feature, frame);
            return false;
        }
        // D4R_SHIM_MARKER_POLL_US: poll interval (0 = yield); detection latency is on the critical path
        static const unsigned markerPollUs = env_uint("D4R_SHIM_MARKER_POLL_US", 200);
        if (markerPollUs == 0)
            std::this_thread::yield();
        else
            d4r_sleep_us(markerPollUs);
    }
    return true;
}

static void prepare_inputs(Feature* feature, int slotIndex, uint32_t frame, FrameParams params, FrameTiming timing)
{
    InputSlot& slot = feature->inputs[slotIndex];
    const auto start = ProfileClock::now();
    // D4R_SHIM_DEFER_MARKER (default on): the worker waits for the game's copies itself, avoiding a thread
    // wake-up between readiness and evaluation. The wait must precede every input read, including CUDA
    // uploads and diagnostic readbacks; waiting only before NGX would copy stale data into its arrays.
    static const bool deferMarker = env_uint("D4R_SHIM_DEFER_MARKER", 1) != 0 &&
                                    env_uint("D4R_MOTION_DILATION", 0) == 0;
    if (params.vram && deferMarker && env_uint("D4R_SHIM_VRAM_VERIFY", 0) == 0)
    {
        timing.markerDeferred = true;
        if (timing.enabled)
        {
            timing.prepStart = start;
            timing.prep = profile_ms(start, ProfileClock::now());
            timing.workerQueued = ProfileClock::now();
        }
        g.worker.post([feature, slotIndex, frame, params, timing] {
            run_evaluation(feature, slotIndex, frame, params, timing);
        });
        return;
    }
    if (!wait_for_input_marker(feature, slotIndex, frame))
        return;
    const auto ready = ProfileClock::now();
    // With VRAM interop the inputs are already on the GPU; host staging runs
    // only to cross-check them (D4R_SHIM_VRAM_VERIFY).
    if (!params.vram || env_uint("D4R_SHIM_VRAM_VERIFY", 0) != 0)
    {
        if (!wait_for_release(slot.hostBusy))
        {
            logf("frame %u: host planes of slot %d still busy; dropping", frame, slotIndex);
            slot.busy = false;
            frame_retired(feature, frame);
            return;
        }
        FrameTiming* stages = timing.enabled ? &timing : nullptr;
        stage_plane(Plane::Color, slot.color, slot.host[0], stages, 0);
        stage_plane(Plane::Depth, slot.depth, slot.host[1], stages, 1);
        stage_plane(Plane::Motion, slot.motion, slot.host[2], stages, 2);
        log_motion_stats(slot.host[2], slot.host[0], frame);
        if (params.hasExposure)
            stage_plane(Plane::Exposure, slot.exposure, slot.host[3], stages, 3);
        char dumpDirectory[MAX_PATH];
        if (input_dump_directory(frame, dumpDirectory))
            dump_input_planes(slot.host, frame, params, dumpDirectory);
        slot.hostBusy = true;
    }
    if (!params.vram)
        slot.busy = false; // the readback buffers are no longer needed
    if (timing.enabled)
    {
        timing.prepStart = start;
        timing.markerWait = profile_ms(start, ready);
        timing.prep = profile_ms(start, ProfileClock::now());
        timing.workerQueued = ProfileClock::now();
    }
    g.worker.post([feature, slotIndex, frame, params, timing] {
        run_evaluation(feature, slotIndex, frame, params, timing);
    });
}

// Picks a result slot that no queued command list still copies from, and
// marks it so evaluate() does not present it while it is rewritten. A reader
// frame's output copy is recorded after its marker, so the GPU is past it
// only once a later frame's marker has landed. Waits briefly for that.
static int claim_output_slot(Feature* feature)
{
    const auto start = std::chrono::steady_clock::now();
    for (;;)
    {
        {
            std::lock_guard<std::mutex> lock(feature->outputMutex);
            const uint32_t gpuFrame = *feature->markerValue;
            for (int index = 0; index < kOutputSlots; ++index)
            {
                if (index == feature->latestOutput.load())
                    continue;
                if (static_cast<int32_t>(gpuFrame - feature->outputs[index].lastReadFrame.load()) > 0)
                {
                    feature->outputs[index].lastReadFrame = 0x7fffffffu + gpuFrame;
                    return index;
                }
            }
        }
        if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(50))
            return -1;
        d4r_sleep_us(200);
    }
}

// Makes `target` the result evaluate() presents and retires the frame.
static void finish_publish(Feature* feature, int target, uint32_t frame, const FrameParams& params,
                           const FrameTiming& timing, ProfileClock::time_point start)
{
    {
        std::lock_guard<std::mutex> lock(feature->outputMutex);
        feature->outputs[target].lastReadFrame = 0;
        feature->outputs[target].producedFrame = frame;
        feature->latestOutput = target;
    }
    const uint32_t completed = ++feature->completedFrames;
    frame_retired(feature, frame);
    const auto finished = ProfileClock::now();
    if (timing.enabled)
    {
        static thread_local ProfileClock::time_point lastFinished;
        const double interval = lastFinished.time_since_epoch().count() != 0
                                    ? profile_ms(lastFinished, finished) : -1.0;
        lastFinished = finished;
        const InputSlot& slot = feature->inputs[frame % kSlots];
        logf("D4R_PROFILE frame=%u render=%ux%u output=%ux%u color=%ux%u depth=%ux%u "
             "motion=%ux%u exposure=%d quality=%d flags=0x%x preset=%u "
             "jitter=%.6f,%.6f mv_scale=%.3f,%.3f reset=%d pre_exposure=%.6f exposure_scale=%.6f "
             "frame_dt=%.3f game_interval=%.3f "
             "queue_depth=%zu queue_wait=%.3f marker_wait=%.3f slot_wait=%.3f "
             "d3d_input_record=%.3f d3d_output_record=%.3f game_call=%.3f "
             "color_convert=%.3f depth_convert=%.3f motion_convert=%.3f exposure_convert=%.3f "
             "color_h2d=%.3f depth_h2d=%.3f motion_h2d=%.3f exposure_h2d=%.3f "
             "ngx_host=%.3f ctx_sync=%.3f gpu_eval=%.3f output_sync_wall=%.3f "
             "output_sync_cpu=%.3f output_sync_blocking=%d output_sync_method=%s d2h=%.3f output_convert=%.3f "
             "h2d_total=%.3f d2h_issue=%.3f d2h_wait=%.3f "
             "prep=%.3f worker_wait=%.3f worker=%.3f finish_wait=%.3f finish=%.3f pipeline=%.3f "
             "throughput_interval=%.3f presented=%u age=%d motion_dilation=%u motion_dilation_wall=%.3f",
             frame, params.renderWidth, params.renderHeight, feature->outWidth, feature->outHeight,
             slot.host[0].width, slot.host[0].height, slot.host[1].width, slot.host[1].height,
             slot.host[2].width, slot.host[2].height, params.hasExposure, feature->quality, feature->flags,
             feature->preset, params.jitterX, params.jitterY, params.mvScaleX, params.mvScaleY,
             params.reset, params.preExposure, params.exposureScale, params.frameTime, timing.gameInterval,
             timing.queueDepth, profile_ms(timing.queuedAt, timing.prepStart), timing.markerWait, timing.slotWait,
             timing.inputRecord, timing.outputRecord, timing.gameCall,
             timing.convert[0], timing.convert[1], timing.convert[2], timing.convert[3],
             timing.upload[0], timing.upload[1], timing.upload[2], timing.upload[3],
             timing.ngxHost, timing.ctxSync, timing.gpuEval, timing.outputSyncWall, timing.outputSyncCpu,
             timing.outputSyncBlocking, timing.outputSyncBlocking ? "event_query_sleep" : "context",
             timing.download, timing.outputConvert,
             timing.h2dTotal, timing.d2hIssue, timing.d2hWait,
             timing.prep, timing.workerWait, timing.worker, timing.finishWait, profile_ms(start, finished),
             profile_ms(timing.prepStart, finished), interval,
             timing.presentedFrame, timing.presentedFrame != 0 ? static_cast<int>(frame - timing.presentedFrame) : -1,
             params.motionDilation, timing.motionDilationWall);
    }
    if (completed <= 5 || completed % 120 == 0)
        logf("frame %u done (slot %d)", frame, target);
}

static void publish_output(Feature* feature, int hostIndex, uint32_t frame, FrameParams params, FrameTiming timing)
{
    const auto start = ProfileClock::now();
    if (timing.enabled)
        timing.finishWait = profile_ms(timing.finishQueued, start);
    const int target = claim_output_slot(feature);
    if (target < 0)
    {
        feature->outputHostBusy[hostIndex] = false;
        logf("frame %u: no free output slot; dropping result", frame);
        frame_retired(feature, frame);
        return;
    }
    write_output(feature->outputs[target].staging, feature->outputHost[hostIndex], timing.enabled ? &timing : nullptr);
    feature->outputHostBusy[hostIndex] = false;
    finish_publish(feature, target, frame, params, timing, start);
}

// Waits until every frame queued so far has passed through all three stages.
static void drain_pipeline()
{
    g.prep.call([] { return 0; });
    g.worker.call([] {
        complete_pending_download();
        return 0;
    });
    g.finish.call([] { return 0; });
}

static void release_feature(Feature* feature);
static void destroy_recorded_resources(Feature* feature);
static void finish_shutdown();

class DeferredResources final : public IUnknown
{
public:
    explicit DeferredResources(Feature* feature) : feature_(feature) { ++g.resourceOwners; }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override
    {
        if (out == nullptr) return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown)) return E_NOINTERFACE;
        *out = static_cast<IUnknown*>(this);
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG remaining = --refs_;
        if (remaining == 0)
            g.cleanup.post([this] {
                destroy_recorded_resources(feature_);
                delete this;
                --g.resourceOwners;
                finish_shutdown();
            });
        return remaining;
    }
private:
    std::atomic<ULONG> refs_{1};
    Feature* feature_;
};

// --- exported NGX D3D12 API ------------------------------------------------------

#define D4R_EXPORT extern "C" __declspec(dllexport)

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_Init(unsigned long long applicationId, const wchar_t* dataPath,
                                          ID3D12Device* device, const NgxFeatureCommonInfo* featureInfo,
                                          unsigned int sdkVersion)
{
    logf("NVSDK_NGX_D3D12_Init(app=%llu, device=%p, sdk=0x%x)", applicationId, device, sdkVersion);
    return initialize(applicationId, dataPath, device, featureInfo, sdkVersion);
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_Init_Ext(unsigned long long applicationId, const wchar_t* dataPath,
                                              ID3D12Device* device, unsigned int sdkVersion,
                                              const NgxFeatureCommonInfo* featureInfo)
{
    logf("NVSDK_NGX_D3D12_Init_Ext(app=%llu, device=%p, sdk=0x%x)", applicationId, device, sdkVersion);
    return initialize(applicationId, dataPath, device, featureInfo, sdkVersion);
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_Init_ProjectID(const char* projectId, int engineType, const char* engineVersion,
                                                    const wchar_t* dataPath, ID3D12Device* device,
                                                    unsigned int sdkVersion, const NgxFeatureCommonInfo* featureInfo)
{
    logf("NVSDK_NGX_D3D12_Init_ProjectID(project=%s, engine=%d %s, device=%p, sdk=0x%x)",
         projectId != nullptr ? projectId : "", engineType, engineVersion != nullptr ? engineVersion : "", device,
         sdkVersion);
    ProjectIdentity project;
    project.id = projectId != nullptr ? projectId : "";
    project.engineType = engineType;
    project.engineVersion = engineVersion != nullptr ? engineVersion : "";
    return initialize(0, dataPath, device, featureInfo, sdkVersion, &project);
}

// Parameter objects handed to callers are the shim's own implementation: the
// CUDA-path objects of the official core cannot hold D3D12 resources.
static std::mutex g_parametersMutex;
static std::vector<void*> g_ownedParameters;

static void* new_parameters()
{
    void* parameters = d4r_ngx_parameters_create();
    std::lock_guard<std::mutex> lock(g_parametersMutex);
    g_ownedParameters.push_back(parameters);
    return parameters;
}

// Copies the official capability values (availability, driver requirements,
// and the optimal-settings/stats callbacks, which work on any parameter
// implementation) into one of our parameter objects.
static NgxResult fill_capabilities(void* parameters)
{
    return g.worker.call([&]() -> NgxResult {
        void* official = nullptr;
        const NgxResult result = g.ngx.getCapabilityParameters(&official);
        if (result != NGX_SUCCESS || official == nullptr)
            return result;
        const char* integers[] = {"SuperSampling.Available", "SuperSampling.NeedsUpdatedDriver",
                                  "SuperSampling.FeatureInitResult"};
        for (const char* name : integers)
        {
            int value = 0;
            if (d4r_ngx_get_int(official, name, &value) == NGX_SUCCESS)
                d4r_ngx_set_int(parameters, name, value);
        }
        const char* unsignedValues[] = {"SuperSampling.MinDriverVersionMajor", "SuperSampling.MinDriverVersionMinor"};
        for (const char* name : unsignedValues)
        {
            unsigned int value = 0;
            if (d4r_ngx_get_uint(official, name, &value) == NGX_SUCCESS)
                d4r_ngx_set_uint(parameters, name, value);
        }
        const char* pointers[] = {"DLSSOptimalSettingsCallback", "DLSSGetStatsCallback"};
        for (const char* name : pointers)
        {
            void* value = nullptr;
            if (d4r_ngx_get_void(official, name, &value) == NGX_SUCCESS && value != nullptr)
                d4r_ngx_set_void(parameters, name, value);
        }
        g.ngx.destroyParameters(official);
        return NGX_SUCCESS;
    });
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_GetParameters(void** parameters)
{
    if (!g.ngxInitialized)
        return NGX_FAIL_NOT_INITIALIZED;
    *parameters = new_parameters();
    return fill_capabilities(*parameters);
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_AllocateParameters(void** parameters)
{
    if (!g.ngxInitialized)
        return NGX_FAIL_NOT_INITIALIZED;
    *parameters = new_parameters();
    return NGX_SUCCESS;
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_GetCapabilityParameters(void** parameters)
{
    if (!g.ngxInitialized)
        return NGX_FAIL_NOT_INITIALIZED;
    *parameters = new_parameters();
    const NgxResult result = fill_capabilities(*parameters);
    int available = -1, initResult = -1, needsDriver = -1;
    d4r_ngx_get_int(*parameters, "SuperSampling.Available", &available);
    d4r_ngx_get_int(*parameters, "SuperSampling.FeatureInitResult", &initResult);
    d4r_ngx_get_int(*parameters, "SuperSampling.NeedsUpdatedDriver", &needsDriver);
    logf("NVSDK_NGX_D3D12_GetCapabilityParameters -> 0x%08x, SuperSampling.Available=%d FeatureInitResult=0x%08x "
         "NeedsUpdatedDriver=%d",
         result, available, static_cast<unsigned int>(initResult), needsDriver);
    return result;
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_DestroyParameters(void* parameters)
{
    {
        std::lock_guard<std::mutex> lock(g_parametersMutex);
        for (auto it = g_ownedParameters.begin(); it != g_ownedParameters.end(); ++it)
            if (*it == parameters)
            {
                g_ownedParameters.erase(it);
                d4r_ngx_parameters_destroy(parameters);
                return NGX_SUCCESS;
            }
    }
    return NGX_FAIL_INVALID_PARAMETER;
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_GetScratchBufferSize(unsigned int, const void*, size_t* size)
{
    // Scratch memory for the CUDA path is allocated internally.
    if (size != nullptr)
        *size = 0;
    return NGX_SUCCESS;
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_GetFeatureRequirements(IUnknown*, const void*, NgxFeatureRequirement* requirement)
{
    if (requirement != nullptr)
    {
        std::memset(requirement, 0, sizeof(*requirement));
        requirement->FeatureSupported = 0; // supported
        requirement->MinHWArchitecture = 0x160;
    }
    return NGX_SUCCESS;
}

static unsigned int get_uint_or(void* parameters, const char* name, unsigned int fallback)
{
    unsigned int value = fallback;
    if (d4r_ngx_get_uint(parameters, name, &value) != NGX_SUCCESS)
    {
        int signedValue = 0;
        if (d4r_ngx_get_int(parameters, name, &signedValue) == NGX_SUCCESS)
            return static_cast<unsigned int>(signedValue);
        return fallback;
    }
    return value;
}

static int get_int_or(void* parameters, const char* name, int fallback)
{
    int value = fallback;
    if (d4r_ngx_get_int(parameters, name, &value) != NGX_SUCCESS)
    {
        unsigned int unsignedValue = 0;
        if (d4r_ngx_get_uint(parameters, name, &unsignedValue) == NGX_SUCCESS)
            return static_cast<int>(unsignedValue);
        return fallback;
    }
    return value;
}

static float get_float_or(void* parameters, const char* name, float fallback)
{
    float value = fallback;
    return d4r_ngx_get_float(parameters, name, &value) == NGX_SUCCESS ? value : fallback;
}

static bool engine_requested();

// Called on the game thread with the feature API lock held. All CUDA resources are made on its worker.
// Vulkan-only K features retain just their creation settings until a frame actually needs CUDA.
static NgxResult create_cuda_feature(Feature& feature)
{
    const char* presetNames[] = {"DLSS.Hint.Render.Preset.DLAA", "DLSS.Hint.Render.Preset.Quality",
                                "DLSS.Hint.Render.Preset.Balanced", "DLSS.Hint.Render.Preset.Performance",
                                "DLSS.Hint.Render.Preset.UltraPerformance", "DLSS.Hint.Render.Preset.UltraQuality"};
    const NgxResult result = g.worker.call([&]() -> NgxResult {
        NgxResult allocation = g.ngx.allocateParameters(&feature.cudaParams);
        if (allocation != NGX_SUCCESS)
            return allocation;
        void* p = feature.cudaParams;
        d4r_ngx_set_uint(p, "Width", feature.width);
        d4r_ngx_set_uint(p, "Height", feature.height);
        d4r_ngx_set_uint(p, "OutWidth", feature.outWidth);
        d4r_ngx_set_uint(p, "OutHeight", feature.outHeight);
        d4r_ngx_set_int(p, "PerfQualityValue", feature.quality);
        d4r_ngx_set_int(p, "DLSS.Feature.Create.Flags", feature.flags);
        d4r_ngx_set_int(p, "DLSS.Enable.Output.Subrects", feature.subrects);
        d4r_ngx_set_uint(p, "CreationNodeMask", 1);
        d4r_ngx_set_uint(p, "VisibilityNodeMask", 1);
        for (int index = 0; index < 6; ++index)
            d4r_ngx_set_uint(p, presetNames[index], feature.creationPresets[index]);
        if (g.cu.memAlloc(&feature.scratch, 64ull * 1024 * 1024) == 0)
        {
            d4r_ngx_set_void(p, "Scratch", reinterpret_cast<void*>(static_cast<uintptr_t>(feature.scratch)));
            d4r_ngx_set_ull(p, "Scratch.SizeInBytes", 64ull * 1024 * 1024);
        }
        else
            feature.scratch = 0;
        const NgxResult created = g.ngx.createFeature(NGX_FEATURE_SUPER_SAMPLING, p, &feature.cudaHandle);
        if (created != NGX_SUCCESS)
        {
            if (feature.cudaHandle != nullptr) g.ngx.releaseFeature(feature.cudaHandle);
            feature.cudaHandle = nullptr;
            g.ngx.destroyParameters(feature.cudaParams); feature.cudaParams = nullptr;
            if (feature.scratch != 0) g.cu.memFree(feature.scratch);
            feature.scratch = 0;
            return created;
        }
        feature.cudaWidth = feature.width;
        feature.cudaHeight = feature.height;
        if (created == NGX_SUCCESS && env_uint("D4R_SHIM_BLOCKING_SYNC", 1) != 0 &&
            g.cu.eventCreate != nullptr && g.cu.eventRecord != nullptr &&
            g.cu.eventSynchronize != nullptr && g.cu.eventQuery != nullptr && g.cu.eventDestroy != nullptr)
        {
            // Explicit query/sleep waits avoid HIP's spinning event-sync path.
            // Timing is unnecessary for this completion event.
            constexpr unsigned int kBlockingSyncNoTiming = 0x1u | 0x2u;
            const int eventResult = g.cu.eventCreate(&feature.outputReadyEvent, kBlockingSyncNoTiming);
            if (eventResult != 0)
            {
                logf("blocking output wait unavailable (cuEventCreate %d); using context sync", eventResult);
                feature.outputReadyEvent = nullptr;
            }
            else
                logf("output event query/sleep wait enabled for this feature");
        }
        if (created == NGX_SUCCESS && profile_enabled() && g.cu.eventCreate != nullptr &&
            g.cu.eventRecord != nullptr && g.cu.eventElapsedTime != nullptr && g.cu.eventDestroy != nullptr)
        {
            const int startResult = g.cu.eventCreate(&feature.profileStart, 0);
            const int endResult = startResult == 0 ? g.cu.eventCreate(&feature.profileEnd, 0) : -1;
            feature.profileEventsReady = startResult == 0 && endResult == 0;
            if (!feature.profileEventsReady)
            {
                logf("D4R_PROFILE: CUDA event creation failed (%d, %d)", startResult, endResult);
                if (feature.profileStart != nullptr)
                    g.cu.eventDestroy(feature.profileStart);
                if (feature.profileEnd != nullptr)
                    g.cu.eventDestroy(feature.profileEnd);
                feature.profileStart = feature.profileEnd = nullptr;
            }
        }
        return created;
    });
    logf("NVSDK_NGX_CUDA_CreateFeature -> 0x%08x", result);
    if (result == NGX_SUCCESS) feature.cudaCreationDeferred = false;
    return result;
}

static NgxResult create_feature(unsigned int featureId, void* parameters, NgxHandle** handle, bool nativeVulkan)
{
    std::lock_guard<std::mutex> apiLock(g_featureApiMutex);
    std::lock_guard<std::mutex> stateLock(g.mutex);
    if (!g.ngxInitialized)
        return NGX_FAIL_NOT_INITIALIZED;
    if (!nativeVulkan && g.device == nullptr)
        return NGX_FAIL_NOT_INITIALIZED;
    if (!nativeVulkan)
        std::call_once(g_vk.once, init_vram_interop);
    if (!nativeVulkan && g_vk.lifetime == nullptr)
    {
        logf("DLSS requires lifetime-aware d4r d3d12.dll/d3d12core.dll; update both files with the shim");
        return NGX_FAIL_PLATFORM_ERROR;
    }
    if (featureId != NGX_FEATURE_SUPER_SAMPLING || parameters == nullptr || handle == nullptr)
    {
        logf("NVSDK_NGX_D3D12_CreateFeature(feature=%u) unsupported", featureId);
        return NGX_FAIL_FEATURE_NOT_SUPPORTED;
    }
    auto* feature = new Feature();
    feature->nativeVulkan = nativeVulkan;
    feature->width = get_uint_or(parameters, "Width", 0);
    feature->height = get_uint_or(parameters, "Height", 0);
    feature->outWidth = get_uint_or(parameters, "OutWidth", 0);
    feature->outHeight = get_uint_or(parameters, "OutHeight", 0);
    const int quality = get_int_or(parameters, "PerfQualityValue", 0);
    const int flags = get_int_or(parameters, "DLSS.Feature.Create.Flags", 0);
    feature->quality = quality;
    feature->flags = flags;
    const int subrects = get_int_or(parameters, "DLSS.Enable.Output.Subrects", 0);
    const char* presetNames[] = {"DLSS.Hint.Render.Preset.DLAA", "DLSS.Hint.Render.Preset.Quality",
                                 "DLSS.Hint.Render.Preset.Balanced", "DLSS.Hint.Render.Preset.Performance",
                                 "DLSS.Hint.Render.Preset.UltraPerformance", "DLSS.Hint.Render.Preset.UltraQuality"};
    unsigned int presets[6];
    for (int index = 0; index < 6; ++index)
        presets[index] = get_uint_or(parameters, presetNames[index], 0);
    // D4R_DLSS_PRESET forces one render preset (NVSDK_NGX_DLSS_Hint_Render_Preset
    // value, e.g. 5 = E) for every quality mode.
    const std::string forced = env_string("D4R_DLSS_PRESET");
    if (!forced.empty())
    {
        const unsigned int value = static_cast<unsigned int>(strtoul(forced.c_str(), nullptr, 0));
        const char* model = value == 5 ? "E" : value == 11 ? "K" : value == 12 ? "L" : value == 13 ? "M" : "custom";
        logf("DLSS model override: D4R_DLSS_PRESET=%s (model %s) overrides game/OptiScaler presets "
             "[DLAA=%u Quality=%u Balanced=%u Performance=%u UltraPerformance=%u UltraQuality=%u]; "
             "using preset=%u for all quality modes",
             forced.c_str(), model, presets[0], presets[1], presets[2], presets[3], presets[4], presets[5], value);
        for (unsigned int& preset : presets)
            preset = value;
    }
    feature->preset = presets[1];
    // Direct output needs the native output kernel of the preset in use (verified: K = 11 through
    // hiluma_engine_output, M = 13 through rrlite_downsample_kernel, its last kernel). Only a forced
    // preset qualifies.
    if (!forced.empty())
    {
        char allowed[128] = "11,13";
        if (const std::string list = env_string("D4R_SHIM_OUTPUT_DIRECT_PRESETS"); !list.empty())
            snprintf(allowed, sizeof(allowed), "%s", list.c_str());
        for (char* token = strtok(allowed, ","); token != nullptr; token = strtok(nullptr, ","))
            feature->outputDirectAllowed |= strtoul(token, nullptr, 0) == presets[1];
    }
    logf("NVSDK_NGX_D3D12_CreateFeature: %ux%u -> %ux%u quality=%d flags=0x%x subrects=%d preset=%u", feature->width,
         feature->height, feature->outWidth, feature->outHeight, quality, flags, subrects, presets[1]);

    feature->subrects = subrects;
    std::copy(std::begin(presets), std::end(presets), feature->creationPresets.begin());
    // The engine records D3D12 evaluations only; the native Vulkan frontend keeps eager CUDA creation.
    feature->cudaCreationDeferred = !nativeVulkan && engine_requested() && feature->preset == 11;
    const NgxResult result = feature->cudaCreationDeferred ? NGX_SUCCESS : create_cuda_feature(*feature);
    if (feature->cudaCreationDeferred)
        logf("engine backend: CUDA feature creation deferred until fallback is needed");
    if (result != NGX_SUCCESS)
    {
        release_feature(feature);
        return result;
    }
    if (!nativeVulkan && !create_buffer(D3D12_HEAP_TYPE_READBACK, 256, &feature->marker,
                       reinterpret_cast<uint8_t**>(const_cast<uint32_t**>(&feature->markerValue))))
    {
        release_feature(feature);
        return NGX_FAIL_PLATFORM_ERROR;
    }
    if (!nativeVulkan)
    {
        for (int index = 0; index <= kSlots; ++index)
            feature->markerValue[index] = 0;
        feature->resources = new DeferredResources(feature);
    }
    feature->handle.Id = g.nextHandleId++;
    {
        std::lock_guard<std::mutex> lock(g_featuresMutex);
        g_features.push_back(feature);
    }
    *handle = &feature->handle;
    return NGX_SUCCESS;
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList*, unsigned int featureId,
                                                   void* parameters, NgxHandle** handle)
{
    return create_feature(featureId, parameters, handle, false);
}

static Feature* find_feature(const NgxHandle* handle)
{
    std::lock_guard<std::mutex> lock(g_featuresMutex);
    for (Feature* feature : g_features)
        if (&feature->handle == handle && !feature->retiring.load())
            return feature;
    return nullptr;
}

static ID3D12Resource* get_resource(void* parameters, const char* name)
{
    ID3D12Resource* resource = nullptr;
    if (d4r_ngx_get_d3d12_resource(parameters, name, &resource) == NGX_SUCCESS && resource != nullptr)
        return resource;
    void* raw = nullptr;
    if (d4r_ngx_get_void(parameters, name, &raw) == NGX_SUCCESS)
        return static_cast<ID3D12Resource*>(raw);
    return nullptr;
}

// --- Native engine backend (engine/): no NGX evaluation, CUDA, worker or frame split -------------
// [Engine] Enabled in d4r.ini (D4R_ENGINE=1) runs presets K and M with d4r's own shaders inside the game's command
// list. The models (the output of engine/compile_k.py / compile_m.py) are the folders engine\k, engine\k-ldr,
// engine\m and engine\m-ldr next to this DLL, or [Engine] ModelDir (D4R_ENGINE_MODEL_DIR) and the
// D4R_ENGINE_MODEL_{LDR,M,M_LDR}_DIR variables. Anything the engine does not cover falls back to the CUDA backend.
struct EngineBackend
{
    std::once_flag vulkanOnce; // the game's device and the engine's Vulkan entry points, for every preset
    std::once_flag once;
    std::unique_ptr<d4r::Model> model; // shared by all features, kept for the life of the process
    // The same network with NGX's LDR input and output stages, for games that do not set the HDR flag
    // (engine\\k-ldr next to this DLL, or D4R_ENGINE_MODEL_LDR_DIR); loaded on first use.
    std::once_flag ldrOnce;
    std::unique_ptr<d4r::Model> modelLdr;
    std::string directory, directoryLdr; // kept for the native HIP network's hipnet.bin and hip/*.hsaco
};
static EngineBackend g_engine;

static bool engine_requested()
{
    static const bool requested = env_uint("D4R_ENGINE", 0) != 0;
    return requested;
}

static void init_engine_vulkan()
{
    VkInstance instance = VK_NULL_HANDLE;
    if (g_vk.interop == nullptr || g_vk.lifetime == nullptr ||
        FAILED(g_vk.interop->GetVulkanHandles(&instance, &g_vk.physical, &g_vk.device)))
    {
        logf("engine backend: the vkd3d-proton interop interfaces are unavailable");
        return;
    }
    HMODULE vulkan = GetModuleHandleA("vulkan-1.dll");
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    if (vulkan == nullptr || !load_export(vulkan, "vkGetInstanceProcAddr", instanceProc) ||
        !d4r::loadVulkan(instanceProc, instance, g_vk.device))
    {
        logf("engine backend: Vulkan entry points are unavailable");
        return;
    }
}

static void init_engine_backend()
{
    // SetEnvironmentVariable (d4r.ini) does not reach the C runtime's getenv
    std::string directory = env_string("D4R_ENGINE_MODEL_DIR");
    if (directory.empty())
    {
        char folder[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (g_portable.dir + L"\\engine\\k").c_str(), -1, folder, sizeof(folder), nullptr, nullptr);
        directory = folder;
    }
    if (g_vk.interop == nullptr || g_vk.lifetime == nullptr || g_vk.device == VK_NULL_HANDLE || d4r::vkTable.CreateBuffer == nullptr)
        return;
    // A Unix path names the same file through Wine's Z: drive.
    const std::string path = directory[0] == '/' ? "Z:" + directory : directory;
    const auto start = std::chrono::steady_clock::now();
    try
    {
        g_engine.model = std::make_unique<d4r::Model>(g_vk.physical, g_vk.device, path);
        g_engine.directory = path;
    }
    catch (const std::exception& error)
    {
        logf("engine backend: cannot load the model from %s: %s (see D4R_README.txt / engine/README.md for how to "
             "create it)", path.c_str(), error.what());
        return;
    }
    logf("engine backend: model %s loaded and pipelines compiled in %.0f ms", path.c_str(),
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

static void init_engine_backend_ldr()
{
    std::string directory = env_string("D4R_ENGINE_MODEL_LDR_DIR");
    if (directory.empty())
    {
        char folder[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (g_portable.dir + L"\\engine\\k-ldr").c_str(), -1, folder, sizeof(folder), nullptr, nullptr);
        directory = folder;
    }
    if (g_vk.interop == nullptr || g_vk.lifetime == nullptr || g_vk.device == VK_NULL_HANDLE || d4r::vkTable.CreateBuffer == nullptr)
        return;
    const std::string path = directory[0] == '/' ? "Z:" + directory : directory;
    try
    {
        g_engine.modelLdr = std::make_unique<d4r::Model>(g_vk.physical, g_vk.device, path);
        g_engine.directoryLdr = path;
        logf("engine backend: LDR model %s loaded", path.c_str());
    }
    catch (const std::exception& error)
    {
        logf("engine backend: cannot load the LDR model from %s: %s", path.c_str(), error.what());
    }
}

// K and M borrow compatible views through the same bounded, resource-owning cache.
static void prepare_engine_view(Feature& feature, ID3D12Resource* resource, const D3D12_RESOURCE_DESC& desc,
                                d4r::GameImage& image, bool isOutput, bool originSupported,
                                D3D12_RESOURCE_STATES& state)
{
    if (image.format == VK_FORMAT_D32_SFLOAT || image.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        image.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    const VkImageUsageFlags usage = isOutput ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT;
    VkFormatProperties properties{};
    d4r::vkTable.GetPhysicalDeviceFormatProperties(g_vk.physical, image.format, &properties);
    const bool enabled = env_uint(isOutput ? "D4R_ENGINE_DIRECT_OUTPUT" : "D4R_ENGINE_DIRECT_INPUTS", 1) != 0;
    const bool compatible = desc.DepthOrArraySize == 1 && originSupported && (isOutput ?
        (image.format == VK_FORMAT_R16G16B16A16_SFLOAT || image.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) &&
            (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) &&
            (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) :
        !(desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) &&
            (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT));
    if (!enabled || !compatible)
        return;
    auto found = std::find_if(feature.engineViews.begin(), feature.engineViews.end(), [&](const auto& view) {
        return view->resource == resource && view->format == image.format && view->aspect == image.aspect &&
            view->usage == usage;
    });
    if (found == feature.engineViews.end())
    {
        VkMemoryRequirements memory{};
        d4r::vkTable.GetImageMemoryRequirements(g_vk.device, image.image, &memory);
        constexpr uint64_t cacheBudget = 512ull * 1024 * 1024;
        if (feature.engineViews.size() < 64 && memory.size <= cacheBudget - feature.engineViewBytes)
        {
            auto view = std::make_unique<EngineBorrowedView>();
            view->device = g_vk.device; view->format = image.format;
            view->aspect = image.aspect; view->usage = usage;
            VkImageViewUsageCreateInfo restricted{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO}; restricted.usage = usage;
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &restricted};
            info.image = image.image; info.viewType = VK_IMAGE_VIEW_TYPE_2D; info.format = view->format;
            info.subresourceRange = {view->aspect, 0, 1, 0, 1};
            if (d4r::vkTable.CreateImageView(g_vk.device, &info, nullptr, &view->view) == VK_SUCCESS)
            {
                resource->AddRef(); view->resource = resource;
                feature.engineViews.push_back(std::move(view));
                feature.engineViewBytes += memory.size;
                found = feature.engineViews.end() - 1;
            }
        }
        else if (!feature.engineViewBudgetReported)
        {
            feature.engineViewBudgetReported = true;
            logf("engine view cache budget reached; new textures use copies until feature retirement");
        }
    }
    if (found != feature.engineViews.end())
    {
        image.view = (*found)->view;
        state = isOutput ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
}

// Preset M on the engine: its own model (engine\\m next to this DLL, or D4R_ENGINE_MODEL_M_DIR) and adapter.
struct EngineBackendM
{
    std::once_flag once, ldrOnce;
    std::unique_ptr<d4r::MModel> model, modelLdr; // LDR: engine\\m-ldr or D4R_ENGINE_MODEL_M_LDR_DIR
    std::string directory, directoryLdr;
};
static EngineBackendM g_engineM;

static void init_engine_backend_m_variant(bool ldr)
{
    std::unique_ptr<d4r::MModel>& slot = ldr ? g_engineM.modelLdr : g_engineM.model;
    std::string directory = env_string(ldr ? "D4R_ENGINE_MODEL_M_LDR_DIR" : "D4R_ENGINE_MODEL_M_DIR");
    if (directory.empty())
    {
        char folder[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (g_portable.dir + (ldr ? L"\\engine\\m-ldr" : L"\\engine\\m")).c_str(), -1, folder, sizeof(folder), nullptr, nullptr);
        directory = folder;
    }
    if (g_vk.interop == nullptr || g_vk.lifetime == nullptr || g_vk.device == VK_NULL_HANDLE || d4r::vkTable.CreateBuffer == nullptr)
    {
        logf("engine backend (M): Vulkan interop is unavailable");
        return;
    }
    const std::string path = directory[0] == '/' ? "Z:" + directory : directory;
    const auto start = std::chrono::steady_clock::now();
    try
    {
        slot = std::make_unique<d4r::MModel>(g_vk.physical, g_vk.device, path);
        (ldr ? g_engineM.directoryLdr : g_engineM.directory) = path;
    }
    catch (const std::exception& error)
    {
        logf("engine backend (M): cannot load the model from %s: %s", path.c_str(), error.what());
        return;
    }
    logf("engine backend (M): model %s loaded and pipelines compiled in %.0f ms", path.c_str(),
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

// Preset M's network on d4r's native HIP layer kernels (bridge: d4rEngineNet*; engine/hip_net.h). HIP runs the
// network about 1.7x faster than the engine's cooperative-matrix shader; the stages around it stay on Vulkan in
// the game's command list, which is split at the network like a CUDA split frame. The ten layers wait on the GPU
// for the front's fill of the frame number (d4rEngineNetInputWaitReady/d4rEngineNetLaunchAt, D4R_SHIM_ENGINE_GPU_WAIT)
// when the bridge supports it; otherwise the worker waits for the CPU-visible marker before launching.
struct EngineNetCode
{
    const char* name;
    const void* data;
    uint64_t bytes;
};
struct EngineNetExports
{
    std::once_flag once;
    int(WINAPI* create)(const uint8_t*, uint64_t, const EngineNetCode*, uint32_t, uint32_t, uint32_t, CudaDevicePtr,
                        CudaDevicePtr, void**) = nullptr;
    int(WINAPI* launch)(void*, uint32_t) = nullptr;
    int(WINAPI* done)(void*, float*) = nullptr;
    void(WINAPI* destroy)(void*) = nullptr;
    // GPU-side readiness: the front fills a u32; the HIP stream waits on it instead of a CPU-visible marker.
    int(WINAPI* inputWaitReady)(void*, CudaDevicePtr, uint32_t*) = nullptr;
    int(WINAPI* launchAt)(void*, uint32_t) = nullptr;
    int(WINAPI* releaseInput)(void*, uint32_t) = nullptr;
    int(WINAPI* setTimeline)(void*, VkDevice, uint64_t) = nullptr;
    int(WINAPI* signalTimeline)(void*, uint64_t) = nullptr;
    bool ready = false;
};
static EngineNetExports g_engineNet;

static VkResult signal_split_semaphore(Feature& feature, uint64_t value)
{
    if (feature.engineNetTimeline)
        return static_cast<VkResult>(g_engineNet.signalTimeline(feature.engineNet, value));
    VkSemaphoreSignalInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    info.semaphore = feature.splitSemaphore;
    info.value = value;
    return g_vk.signalSemaphore(g_vk.device, &info);
}

static bool read_file(const std::string& path, std::vector<char>& bytes)
{
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr)
        return false;
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    bytes.resize(size > 0 ? static_cast<size_t>(size) : 0);
    const bool ok = size > 0 && fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
    fclose(file);
    return ok;
}

// A storage buffer the engine's shaders and HIP both use. Pooled like the interop buffers, for the same ROCm leak.
static bool create_engine_shared_buffer(VramBuffer& target, size_t bytes, bool host = false)
{
    const uint8_t kind = host ? 2 : 1;
    if (take_pooled_vram_buffer(target, bytes, kind))
    {
        if (target.mapped != nullptr)
        {
            std::memset(target.mapped, 0, target.bytes);
            MemoryBarrier();
        }
        return true;
    }
    VramBuffer buffer;
    buffer.kind = kind;
    VkExternalMemoryBufferCreateInfo external = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = g_vk.createBuffer(g_vk.device, &info, nullptr, &buffer.buffer);
    VkMemoryRequirements requirements = {};
    if (result == VK_SUCCESS)
        g_vk.bufferRequirements(g_vk.device, buffer.buffer, &requirements);
    int type = device_local_type(requirements.memoryTypeBits);
    if (host)
    {
        type = -1;
        const auto coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (int pass = 0; pass < 2 && type < 0; ++pass)
            for (uint32_t index = 0; index < g_vk.memory.memoryTypeCount; ++index)
            {
                const auto flags = g_vk.memory.memoryTypes[index].propertyFlags;
                if ((requirements.memoryTypeBits & (1u << index)) && (flags & coherent) == coherent &&
                    (pass != 0 || !(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
                {
                    type = static_cast<int>(index);
                    break;
                }
            }
    }
    VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = buffer.buffer;
    VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dedicated};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exportInfo};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = static_cast<uint32_t>(type);
    if (result == VK_SUCCESS && type < 0)
        result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (result == VK_SUCCESS)
        result = g_vk.allocate(g_vk.device, &allocation, nullptr, &buffer.memory);
    if (result == VK_SUCCESS)
        result = g_vk.bindBuffer(g_vk.device, buffer.buffer, buffer.memory, 0);
    if (result == VK_SUCCESS && host)
        result = g_vk.map(g_vk.device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped);
    int imported = -1;
    if (result == VK_SUCCESS)
        imported = g.worker.call([&] {
            return g_vk.import(g_vk.device, reinterpret_cast<uint64_t>(buffer.memory), requirements.size, &buffer.device,
                               &buffer.external);
        });
    if (result != VK_SUCCESS || imported != 0)
    {
        logf("engine backend: shared buffer of %zu bytes failed (VkResult %d, import %d)", bytes, result, imported);
        if (buffer.mapped != nullptr)
            g_vk.unmap(g_vk.device, buffer.memory);
        if (buffer.memory != VK_NULL_HANDLE)
            g_vk.free(g_vk.device, buffer.memory, nullptr);
        if (buffer.buffer != VK_NULL_HANDLE)
            g_vk.destroyBuffer(g_vk.device, buffer.buffer, nullptr);
        return false;
    }
    if (buffer.mapped != nullptr)
    {
        std::memset(buffer.mapped, 0, bytes);
        MemoryBarrier();
    }
    buffer.bytes = bytes;
    target = buffer;
    return true;
}

// Only once no command list uses the engine and no network launch is pending.
static void release_engine_network(Feature& feature)
{
    if (feature.engineNet != nullptr && g_engineNet.destroy != nullptr)
        g.worker.call([&feature] {
            g_engineNet.destroy(feature.engineNet);
            return 0;
        });
    feature.engineNet = nullptr;
    feature.engineNetTimeline = false;
    for (VramBuffer* buffer : {&feature.engineTokens, &feature.engineHead, &feature.engineSync})
        recycle_vram_buffer(*buffer);
}

// The HIP network of a feature: the buffers shared with HIP and the bridge's network over them. False: the
// engine's own Vulkan network is used instead (reason logged). `preset` is "M" or "K"; `width`
// and `height` are the dimensions the bridge derives the token grid from (M: render, K: output); `tokenBytes`
// and `headBytes` size the two shared buffers (M: both MEngine::networkBufferBytes, K: Engine::networkTokenBytes
// and Engine::networkHeadBytes). The layer kernel names come from hipnet.bin, so no preset list is repeated here.
static bool create_engine_network(Feature& feature, const std::string& directory, const char* preset,
                                  bool stagesAvailable, unsigned int width, unsigned int height, size_t tokenBytes,
                                  size_t headBytes)
{
    const std::string choice = env_string("D4R_ENGINE_NETWORK");
    if (choice == "vulkan")
        return false;
    auto unavailable = [&](const char* reason) {
        logf("engine backend (%s): HIP network unavailable (%s); using the Vulkan network", preset, reason);
        release_engine_network(feature);
        return false;
    };
    std::call_once(g_engineNet.once, [] {
        g_engineNet.ready = load_export(g.cuda, "d4rEngineNetCreate", g_engineNet.create) &&
                            load_export(g.cuda, "d4rEngineNetLaunch", g_engineNet.launch) &&
                            load_export(g.cuda, "d4rEngineNetDone", g_engineNet.done) &&
                            load_export(g.cuda, "d4rEngineNetDestroy", g_engineNet.destroy) &&
                            load_export(g.cuda, "d4rEngineNetInputWaitReady", g_engineNet.inputWaitReady) &&
                            load_export(g.cuda, "d4rEngineNetLaunchAt", g_engineNet.launchAt) &&
                            load_export(g.cuda, "d4rEngineNetReleaseInput", g_engineNet.releaseInput) &&
                            load_export(g.cuda, "d4rEngineNetSetTimeline", g_engineNet.setTimeline) &&
                            load_export(g.cuda, "d4rEngineNetSignalTimeline", g_engineNet.signalTimeline);
    });
    if (!g_engineNet.ready)
        return unavailable("the CUDA bridge lacks d4rEngineNet*");
    if (!stagesAvailable)
        return unavailable("the model has no stages for the native network; rebuild it with --hip");
    if (g_vk.split == nullptr || g_vk.import == nullptr || feature.marker == nullptr || feature.markerValue == nullptr)
        return unavailable("split frames need the d4r vkd3d-proton and VRAM interop");
    std::vector<char> net;
    if (!read_file(directory + "\\hipnet.bin", net))
        return unavailable("no hipnet.bin in the model directory");
    // The bridge dispatches on the magic alone, and the shared buffers are sized for `preset`.
    const char* const magic = std::strcmp(preset, "K") == 0 ? "D4RKHIP1" : "D4RMHIP1";
    if (net.size() < 16 || std::memcmp(net.data(), magic, 8) != 0)
        return unavailable("hipnet.bin is not a native network of this preset");
    uint32_t count = 0;
    std::memcpy(&count, net.data() + 8, sizeof(count));
    if (count == 0 || net.size() < 16 + 64ull * count)
        return unavailable("invalid hipnet.bin");
    // The kernel names live in the layer table; load the code object of each distinct one (the M tube repeats).
    std::vector<std::string> names;
    names.reserve(count);
    for (uint32_t index = 0; index < count; ++index)
    {
        char name[33] = {};
        std::memcpy(name, net.data() + 16 + 64 * index, 32);
        if (name[0] == '\0')
            return unavailable("invalid kernel name in hipnet.bin");
        if (std::find(names.begin(), names.end(), name) == names.end())
            names.emplace_back(name);
    }
    std::vector<std::vector<char>> code(names.size());
    std::vector<EngineNetCode> codes(names.size());
    for (size_t index = 0; index < names.size(); ++index)
    {
        if (!read_file(directory + "\\hip\\" + names[index] + ".hsaco", code[index]))
            return unavailable("a layer kernel is missing in the model's hip folder");
        codes[index] = {names[index].c_str(), code[index].data(), code[index].size()};
    }
    if (!create_engine_shared_buffer(feature.engineTokens, tokenBytes) ||
        !create_engine_shared_buffer(feature.engineHead, headBytes) ||
        !create_engine_shared_buffer(feature.engineSync, 256, true))
        return unavailable("shared buffers");
    const int created = g.worker.call([&] {
        return g_engineNet.create(reinterpret_cast<const uint8_t*>(net.data()), net.size(), codes.data(),
                                  static_cast<uint32_t>(codes.size()), width, height, feature.engineTokens.device,
                                  feature.engineHead.device, &feature.engineNet);
    });
    if (created != 0 || feature.engineNet == nullptr)
    {
        feature.engineNet = nullptr;
        logf("engine backend (%s): d4rEngineNetCreate failed (%d)", preset, created);
        return unavailable("the kernels could not be loaded for this GPU");
    }
    // Input readiness on the GPU: the engine's front fills engineSync with the frame number (the shim records the
    // fill right after recordFront) and the network's stream waits for it, so the layers start the moment the
    // front is complete, without the CPU round trip through the readback marker. The bridge checks HIP's
    // capability and a round trip on this buffer. Without wait-value support (or D4R_SHIM_ENGINE_GPU_WAIT=0)
    // the worker keeps waiting for the CPU-visible marker before launching.
    const bool gpuWaitEnabled = env_uint("D4R_SHIM_ENGINE_GPU_WAIT", 1) != 0;
    const int waitProbe = gpuWaitEnabled ? g.worker.call([&] {
        return g_engineNet.inputWaitReady(feature.engineNet, feature.engineSync.device,
                                         static_cast<uint32_t*>(feature.engineSync.mapped));
    }) : 0;
    if (waitProbe < 0)
        return unavailable("the HIP stream cannot complete a wait-value test");
    feature.engineGpuWait = waitProbe == 1;
    logf("engine backend (%s): HIP network input readiness on %s", preset, feature.engineGpuWait.load()
             ? "the GPU (the front publishes the frame number; the layers wait on it)"
             : "the CPU (marker polling)");
    if (feature.splitSemaphore == VK_NULL_HANDLE && !create_split_semaphore(feature))
        return unavailable("timeline semaphore");
    if (g.worker.call([&] {
            return g_engineNet.setTimeline(feature.engineNet, g_vk.device,
                                          reinterpret_cast<uint64_t>(feature.splitSemaphore));
        }) != 0)
        return unavailable("native HIP completion timeline");
    feature.engineNetTimeline = true;
    logf("engine backend (%s): network completion releases the split timeline from the native HIP callback", preset);
    return true;
}

// Records the end of the front: a fill of this frame's number into the buffer the HIP stream waits on
// (d4rEngineNetLaunchAt). The engine's recordFront ends the input stage with its own ALL_COMMANDS barrier (the
// tokens are then available); the fill is ordered after it and, once it lands, the ten layers may read them.
static void record_engine_input_ready(Feature& feature, VkCommandBuffer command, uint32_t frame)
{
    const VkMemoryBarrier before = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                    VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.barrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr,
                 0, nullptr);
    g_vk.fill(command, feature.engineSync.buffer, 0, sizeof(uint32_t), frame);
    const VkMemoryBarrier after = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    g_vk.barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr,
                 0, nullptr);
}

static bool wait_for_input_marker(Feature* feature, int slotIndex, uint32_t frame);
static void frame_retired(Feature* feature, uint32_t frame);

// The network's inputs are written by the front of the game's list, so its stream waits for the frame's fill of
// the shared u32 (d4rEngineNetLaunchAt). A wait for a frame whose front never reached the GPU (the game recorded
// its list but did not execute it, or the game stalled) can only be satisfied by a later frame's fill: writing
// the value it waits for releases it, so the stream keeps moving. The frame's output is undefined either way.
// `disable`: the GPU-side wait itself looks broken for this feature, so later frames wait for the CPU marker.
static void release_engine_input_wait(Feature& feature, uint32_t frame, const char* reason, bool disable)
{
    if (feature.engineNet == nullptr || g_engineNet.releaseInput == nullptr)
        return;
    const int result = g_engineNet.releaseInput(feature.engineNet, frame);
    if (disable)
        feature.engineGpuWait.store(false);
    logf("engine frame %u: %s; releasing the GPU-side input wait (%d)", frame, reason, result);
}

// CUDA worker: runs the network of `frame` (behind the GPU-side input wait, or once the CPU marker shows the
// front has run), then lets the second part of the game's command list (the engine's reconstruction) run.
static void run_engine_network(Feature* feature, int slotIndex, uint32_t frame)
{
    const bool gpuWait = feature->engineGpuWait.load() && g_engineNet.launchAt != nullptr;
    if (!gpuWait && !wait_for_input_marker(feature, slotIndex, frame))
        return; // dropped and retired there
    const auto start = std::chrono::steady_clock::now();
    int launched;
    if (gpuWait)
    {
        feature->engineWaitFrame.store(frame);
        launched = g_engineNet.launchAt(feature->engineNet, frame);
        if (launched == 801 /* CUDA_ERROR_NOT_SUPPORTED: no wait was queued */)
        {
            feature->engineWaitFrame.store(0);
            // The bridge could not queue the wait (no hipStreamWaitValue32, or the stream rejected it): wait
            // for the input marker on the CPU, as without the GPU-side wait.
            feature->engineGpuWait.store(false);
            logf("engine frame %u: GPU-side input wait unavailable (%d); waiting for the input marker on the CPU", frame, launched);
            if (!wait_for_input_marker(feature, slotIndex, frame))
                return;
            launched = g_engineNet.launch(feature->engineNet, frame);
        }
    }
    else
        launched = g_engineNet.launch(feature->engineNet, frame);
    float gpuMs = 0.0f;
    int done = launched == 0 ? 0 : -1;
    bool released = false, discarded = false;
    // Completion polling (as the CUDA path's output event): sleep while far from the predicted completion,
    // yield around it (a Wine sleep rounds up to ~50 us), back off when the GPU runs late. The prediction
    // learns from the earlier frames; with the GPU-side wait it also covers the front's remaining GPU time.
    double lastNotReadyUs = -1.0;
    while (done == 0)
    {
        done = g_engineNet.done(feature->engineNet, &gpuMs);
        if (done != 0)
            break;
        const double elapsedUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        if (gpuWait && !released)
        {
            // This frame's own marker (written with the aggregate one, at the end of its front) tells whether
            // its front ran at all. A later frame's front running while this one's never did means the game
            // discarded this list: its wait can only be satisfied by that later front, which would make the
            // network read another frame's tokens. Release it and drop the frame, as the CPU marker path does.
            const uint32_t slotMarker = feature->markerValue[1 + slotIndex];
            if (slotMarker != frame && static_cast<int32_t>(*feature->markerValue - frame) > 0)
            {
                release_engine_input_wait(*feature, frame, "its command list was discarded", false);
                discarded = true;
                break;
            }
            // The front ran but the stream's wait did not start the network (a broken wait would leave every
            // later frame queued behind it): release it and wait for the CPU marker from now on.
            if (slotMarker == frame && elapsedUs > 1000000.0)
            {
                release_engine_input_wait(*feature, frame, "the GPU-side wait did not start the network", true);
                released = true;
            }
        }
        if (elapsedUs > 2000000.0)
        {
            done = -2;
            break;
        }
        lastNotReadyUs = elapsedUs;
        constexpr double kMarginUs = 120.0, kStepUs = 20.0, kChunkUs = 200.0, kSpinAfterUs = 150.0;
        const double predictedUs = feature->engineNetWaitEmaUs;
        const double remaining = predictedUs - kMarginUs - elapsedUs;
        if (remaining > kStepUs)
            d4r_sleep_us(static_cast<unsigned>(std::min(remaining, kChunkUs)));
        else if (elapsedUs < predictedUs + kSpinAfterUs)
            d4r_sleep_us(0); // polling retires resources; the native callback releases the GPU's split wait
        else
            d4r_sleep_us(static_cast<unsigned>(std::min(
                kStepUs * (1.0 + (elapsedUs - predictedUs - kSpinAfterUs) / 160.0), kChunkUs)));
    }
    feature->engineWaitFrame.store(0);
    const double waitedUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    if (done == 1)
    {
        // Learn the completion time, not our own wait: the work finished between the last NOT_READY poll and the
        // successful one (at most one chunk apart), so the midpoint cannot ratchet upwards.
        double sample = lastNotReadyUs >= 0.0 ? 0.5 * (lastNotReadyUs + waitedUs) : waitedUs;
        sample = std::min(sample, 20000.0);
        feature->engineNetWaitEmaUs = feature->engineNetWaitEmaUs == 0.0
                                          ? sample : 0.8 * feature->engineNetWaitEmaUs + 0.2 * sample;
    }
    if (discarded)
        logf("engine frame %u: the game discarded its command list; the frame is dropped", frame);
    else if (done != 1)
        logf("engine frame %u: HIP network failed (launch %d, state %d); the frame's output is undefined", frame, launched, done);
    else if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u: HIP network %.2f ms GPU, %.2f ms %s", frame, gpuMs, waitedUs / 1000.0,
             gpuWait ? "from launch to completion (GPU-side input wait)" : "from input readiness to completion");
    if (gpuWait && !discarded && done != 1)
        release_engine_input_wait(*feature, frame, "the network failed before completing its input wait", true);
    feature->inputs[slotIndex].busy = false;
    frame_retired(feature, frame);
}

// Preset K and preset M record a native-network frame the same way: the front half, this frame's marker, the
// split of the game's list, the back half, and the worker that runs the network between the two halves.
// `recordPart(front)` records one half into the game's list; `sources` is {colour, depth, motion, output}, the
// first three sampled through the front. `refuse` (nothing usable recorded; the CUDA backend takes over) and
// `abandon` (recorded but the frame is unusable) belong to the caller so its logging and flags stay preset-specific.
struct EngineFrameSource
{
    ID3D12Resource* resource;
    d4r::GameImage* image;
    unsigned int x, y, width, height;
    D3D12_RESOURCE_STATES state, copyState;
};

template <typename RecordPart, typename Refuse, typename Abandon>
static bool record_engine_network_frame(Feature& feature, ID3D12GraphicsCommandList* list, EngineFrameSource* sources,
                                        uint32_t frame, const char* preset, int reset, float mvScaleX, float mvScaleY,
                                        float jitterX, float jitterY, RecordPart&& recordPart, Refuse&& refuse,
                                        Abandon&& abandon)
{
    // HIP waits on the GPU-published input value between the two Vulkan list parts. Its native completion
    // callback releases the second part. The Wine worker retires resources, not frame timing.
    const int slotIndex = static_cast<int>(frame % kSlots);
    InputSlot& slot = feature.inputs[slotIndex];
    const auto waitStart = std::chrono::steady_clock::now();
    while (slot.busy.load() || feature.inFlight.load() >= 2)
    {
        if (std::chrono::steady_clock::now() - waitStart > std::chrono::seconds(2))
        {
            logf("engine frame %u: %d frames in flight; skipping evaluation", frame, feature.inFlight.load());
            return true;
        }
        d4r_sleep_us(200);
    }
    if (FAILED(g_vk.lifetime->RetainExternalResources(list, feature.resources)))
        return refuse("the command list cannot retain the engine's resources");
    ID3D12GraphicsCommandList2* list2 = nullptr;
    if (FAILED(list->QueryInterface(__uuidof(ID3D12GraphicsCommandList2), reinterpret_cast<void**>(&list2))))
        return refuse("ID3D12GraphicsCommandList2 is unavailable");
    for (int index = 0; index < 3; ++index)
        transition(list, sources[index].resource, sources[index].state, sources[index].copyState);
    const char* failure = recordPart(true);
    if (failure != nullptr)
    {
        for (int index = 0; index < 3; ++index)
            transition(list, sources[index].resource, sources[index].copyState, sources[index].state);
        list2->Release();
        return refuse(failure);
    }
    const D3D12_GPU_VIRTUAL_ADDRESS markerAddress = feature.marker->GetGPUVirtualAddress();
    D3D12_WRITEBUFFERIMMEDIATE_PARAMETER markers[] = {
        {markerAddress + sizeof(uint32_t) * (1 + slotIndex), frame},
        {markerAddress, frame},
    };
    D3D12_WRITEBUFFERIMMEDIATE_MODE modes[] = {
        D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT,
    };
    list2->WriteBufferImmediate(2, markers, modes);
    list2->Release();
    // From here on this call owns the frame: a failure leaves this frame's output undefined and hands the next
    // frame to the CUDA backend (which must not record its own marker and split for this frame).
    if (FAILED(g_vk.split->SplitCommandListForExternalWait(list, reinterpret_cast<UINT64>(feature.splitSemaphore), frame)))
    {
        for (int index = 0; index < 3; ++index)
            transition(list, sources[index].resource, sources[index].copyState, sources[index].state);
        return abandon("the command list could not be split");
    }
    transition(list, sources[3].resource, sources[3].state, sources[3].copyState);
    failure = recordPart(false);
    transition(list, sources[3].resource, sources[3].copyState, sources[3].state);
    // The reconstruction still samples borrowed colour; keep its sampled layout through the back.
    for (int index = 0; index < 3; ++index)
        transition(list, sources[index].resource, sources[index].copyState, sources[index].state);
    feature.split = true;
    slot.busy = true;
    feature.inFlight.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(feature.splitMutex);
        feature.splitCompletion.admit(frame);
    }
    Feature* const owner = &feature;
    g.worker.post([owner, slotIndex, frame] { run_engine_network(owner, slotIndex, frame); });
    if (failure != nullptr)
        return abandon(failure);
    feature.engineJitter[0] = jitterX;
    feature.engineJitter[1] = jitterY;
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded, preset %s with the HIP network (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f)",
             frame, preset, jitterX, jitterY, reset, mvScaleX, mvScaleY);
    return true;
}

// Preset M: record the evaluation into `list`, borrowing compatible inputs and copying only fallbacks.
static bool engine_evaluate_m(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                              ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                              const FrameParams& p, uint32_t frame)
{
    auto refuse = [&](const char* reason) {
        feature.engineRefused = true;
        logf("engine backend (M): %s; feature %u uses the CUDA backend from frame %u", reason, feature.handle.Id, frame);
        return false;
    };
    // NVSDK_NGX_DLSS_Feature_Flags: IsHDR 1, MVLowRes 2, MVJittered 4, DepthInverted 8, AutoExposure 0x40.
    const bool ldr = (feature.flags & 0x1) == 0; // NGX's LDR stages: a model directory of its own, no exposure
    if (ldr)
        std::call_once(g_engineM.ldrOnce, [] { init_engine_backend_m_variant(true); });
    else
        std::call_once(g_engineM.once, [] { init_engine_backend_m_variant(false); });
    d4r::MModel* const model = ldr ? g_engineM.modelLdr.get() : g_engineM.model.get();
    if (model == nullptr)
        return refuse(ldr ? "no LDR model (engine\\m-ldr) for a game without the HDR flag" : "not available");
    if ((feature.flags & 0x2) == 0)
        return refuse("needs render-resolution motion vectors");
    // NGX measures the exposure for this preset, so the AutoExposure flag, the game's exposure texture, the
    // pre-exposure and the exposure scale are all ignored here (CUDA output is identical for each of them).
    if (feature.engineM != nullptr &&
        (p.renderWidth != feature.engineMRenderWidth || p.renderHeight != feature.engineMRenderHeight))
        return refuse("changing the effective render resolution after initialization is not implemented");
    if (color == output || depth == output || motion == output)
        return refuse("input/output texture aliasing is not supported");
    d4r::GameFrameM game;
    using Source = EngineFrameSource;
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    Source sources[4] = {
        {color, &game.color, p.colorBaseX, p.colorBaseY, p.renderWidth, p.renderHeight, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {depth, &game.depth, p.depthBaseX, p.depthBaseY, p.renderWidth, p.renderHeight, depthState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {motion, &game.motion, p.mvBaseX, p.mvBaseY, p.renderWidth, p.renderHeight, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {output, &game.output, p.outputBaseX, p.outputBaseY, feature.outWidth, feature.outHeight, outputState,
         D3D12_RESOURCE_STATE_COPY_DEST},
    };
    for (Source& source : sources)
    {
        UINT64 handle = 0, offset = 0;
        D3D12_RESOURCE_DESC desc;
        source.resource->GetDesc(&desc);
        if (FAILED(g_vk.interop->GetVulkanResourceInfo1(source.resource, &handle, &offset, &source.image->format)) ||
            handle == 0 || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
            source.x + source.width > desc.Width || source.y + source.height > desc.Height)
            return refuse("a texture is not a single-sample 2D image covering its rectangle");
        source.image->image = reinterpret_cast<VkImage>(handle);
        source.image->x = source.x;
        source.image->y = source.y;
        prepare_engine_view(feature, source.resource, desc, *source.image, source.image == &game.output,
                            source.x == 0 && source.y == 0, source.copyState);
        g_vk.interop->GetVulkanImageLayout(source.resource, source.copyState, &source.image->layout);
    }
    if (game.depth.format == VK_FORMAT_D32_SFLOAT || game.depth.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        game.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!d4r::gameColorFormat(game.color.format) || !d4r::gameMotionFormat(game.motion.format) ||
        !d4r::gameDepthFormat(game.depth.format, game.depth.aspect) || !d4r::gameOutputFormat(game.output.format))
    {
        logf("engine backend (M): Vulkan formats colour %d, depth %d, motion %d, output %d", game.color.format,
             game.depth.format, game.motion.format, game.output.format);
        return refuse("a texture format is not implemented");
    }
    game.jitter[0] = p.jitterX;
    game.jitter[1] = p.jitterY;
    game.motionToRender[0] = p.mvScaleX;
    game.motionToRender[1] = p.mvScaleY;
    if ((feature.flags & 0x4) != 0 && p.mvScaleX != 0.0f && p.mvScaleY != 0.0f)
    {
        game.motionOffset[0] = (p.jitterX - feature.engineJitter[0]) / p.mvScaleX;
        game.motionOffset[1] = (p.jitterY - feature.engineJitter[1]) / p.mvScaleY;
    }
    game.reset = p.reset != 0 || feature.engineM == nullptr;
    game.depthInverted = (feature.flags & 0x8) != 0;
    try
    {
        if (feature.engineM == nullptr)
        {
            const size_t bytes = static_cast<size_t>(d4r::MEngine::networkBufferBytes(p.renderWidth, p.renderHeight));
            const bool hip = create_engine_network(feature, ldr ? g_engineM.directoryLdr : g_engineM.directory, "M",
                                                   model->hasExternalNetworkStages(), p.renderWidth, p.renderHeight,
                                                   bytes, bytes);
            const d4r::ExternalNetwork external = {feature.engineTokens.buffer, feature.engineHead.buffer};
            feature.engineM = std::make_unique<d4r::GameUpscalerM>(*model, g_vk.physical, g_vk.device, feature.outWidth,
                                                                   feature.outHeight, p.renderWidth, p.renderHeight,
                                                                   hip ? &external : nullptr);
            feature.engineMRenderWidth = p.renderWidth;
            feature.engineMRenderHeight = p.renderHeight;
            logf("engine backend (M): feature %u runs preset M in the game's command list (%ux%u -> %ux%u), network on %s, RGB10A2 direct output %d",
                 feature.handle.Id, p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight,
                 hip ? "the native HIP layers (list split around it)" : "Vulkan",
                 model->hasRgb10Output() && game.output.view != VK_NULL_HANDLE &&
                     game.output.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
                     game.output.layout == VK_IMAGE_LAYOUT_GENERAL && game.output.image != game.color.image &&
                     game.output.image != game.motion.image && game.output.image != game.depth.image);
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend (M): %s", error.what());
        release_engine_network(feature);
        return refuse("the engine could not be created");
    }
    if (feature.engineNet != nullptr)
    {
        // The native HIP network: the game's list is split around it. Its native completion callback releases
        // the second part, which records expansion, reconstruction and the final store (or the conversion blit
        // for incompatible outputs). The Wine worker retires resources, not frame timing.
        auto recordPart = [&](bool front) -> const char* {
            VkCommandBuffer command = VK_NULL_HANDLE;
            if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &command)))
                return "BeginVkCommandBufferInterop failed";
            const char* failure = nullptr;
            try
            {
                if (front)
                {
                    feature.engineM->recordFront(command, game);
                    // The HIP network's stream waits for this frame's number (record_engine_input_ready): the
                    // fill is part of the first half, so the network starts the moment the front's GPU work ends.
                    if (feature.engineGpuWait.load())
                        record_engine_input_ready(feature, command, frame);
                }
                else
                    feature.engineM->recordBack(command, game);
            }
            catch (const std::exception& error)
            {
                logf("engine backend (M): %s", error.what());
                failure = "the frame could not be recorded";
            }
            if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && failure == nullptr)
                failure = "EndVkCommandBufferInterop failed";
            return failure;
        };
        auto abandon = [&](const char* reason) {
            feature.engineRefused = true;
            logf("engine backend (M): %s; feature %u uses the CUDA backend after frame %u", reason, feature.handle.Id, frame);
            return true;
        };
        return record_engine_network_frame(feature, list, sources, frame, "M", game.reset, p.mvScaleX, p.mvScaleY,
                                           p.jitterX, p.jitterY, recordPart, refuse, abandon);
    }
    if (FAILED(g_vk.lifetime->RetainExternalResources(list, feature.resources)))
        return refuse("the command list cannot retain the engine's resources");
    for (const Source& source : sources)
        transition(list, source.resource, source.state, source.copyState);
    VkCommandBuffer command = VK_NULL_HANDLE;
    bool recorded = SUCCEEDED(g_vk.interop->BeginVkCommandBufferInterop(list, &command));
    const char* failure = recorded ? nullptr : "BeginVkCommandBufferInterop failed";
    if (recorded)
    {
        try
        {
            feature.engineM->record(command, game);
        }
        catch (const std::exception& error)
        {
            logf("engine backend (M): %s", error.what());
            failure = "the frame could not be recorded";
            recorded = false;
        }
        if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && recorded)
        {
            failure = "EndVkCommandBufferInterop failed";
            recorded = false;
        }
    }
    for (const Source& source : sources)
        transition(list, source.resource, source.copyState, source.state);
    if (!recorded)
        return refuse(failure);
    feature.engineJitter[0] = p.jitterX;
    feature.engineJitter[1] = p.jitterY;
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded, preset M (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f)", frame, p.jitterX, p.jitterY,
             game.reset, p.mvScaleX, p.mvScaleY);
    return true;
}

// Records the whole evaluation of `frame` into `list`. False: nothing usable was recorded and the
// CUDA backend takes this feature over.
static bool engine_evaluate(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                            ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                            ID3D12Resource* exposure, const FrameParams& p, uint32_t frame)
{
    if (feature.engineRefused)
        return false;
    auto refuse = [&](const char* reason) {
        feature.engineRefused = true;
        logf("engine backend: %s; feature %u uses the CUDA backend from frame %u", reason, feature.handle.Id, frame);
        if (feature.engine != nullptr)
            logf("engine backend: the temporal history restarts in the CUDA backend");
        return false;
    };
    std::call_once(g_engine.vulkanOnce, init_engine_vulkan);
    if (feature.preset == 13)
        return engine_evaluate_m(feature, list, color, depth, motion, output, p, frame);
    // Without the HDR flag NGX runs its LDR input and output kernels: a model directory of its own.
    const bool ldr = (feature.flags & 0x1) == 0;
    if (ldr)
        std::call_once(g_engine.ldrOnce, init_engine_backend_ldr);
    else
        std::call_once(g_engine.once, init_engine_backend);
    d4r::Model* const model = ldr ? g_engine.modelLdr.get() : g_engine.model.get();
    if (model == nullptr)
        return refuse(ldr ? "no LDR model (engine\\k-ldr) for a game without the HDR flag" : "not available");
    // NVSDK_NGX_DLSS_Feature_Flags: IsHDR 1, MVLowRes 2, MVJittered 4, DepthInverted 8.
    if (feature.preset != 11)
        return refuse("only preset K is implemented");
    // MVLowRes (0x2) clear: vectors are at output resolution (engine revision D4RO0003).
    const bool displayMotion = (feature.flags & 0x2) == 0;
    if (displayMotion && !model->hasDisplayMotion())
        return refuse("display-resolution motion needs a model compiled by the current compile_k.py");
    if ((feature.flags & 0x8) == 0 && !model->hasRegularDepth())
        return refuse("regular (non-inverted) depth needs a model compiled by the current compile_k.py");
    // DLSS.Indicator.Invert.X/Y.Axis (p.invertX/Y) only place NGX's debug indicator: NGX's K and M outputs are
    // byte-identical with them set (harness D4R_HARNESS_INDICATOR_INVERT=1), so neither engine path reads them.
    // Two exposure sources. AutoExposure (0x40) measures the frame's luma scaled by 1 / pre-exposure and multiplies
    // the result by the exposure scale. Without it the game's ExposureTexture is the exposure: texel x scale /
    // pre-exposure (read directly, not measured).
    const bool autoExposure = (feature.flags & 0x40) != 0;
    const bool gameExposure = !ldr && !autoExposure;
    if (!ldr && autoExposure && !model->hasAutoExposure())
        return refuse("automatic exposure needs a model with exposure shaders");
    if (gameExposure && (!p.hasExposure || !model->hasGameExposure()))
        return refuse("exposure without AutoExposure needs an exposure texture and a model compiled by the current compile_k.py");
    if (!ldr && !(std::isfinite(p.exposureScale) && p.exposureScale > 0.0f))
        return refuse("the exposure scale is not a positive number");
    if (!ldr && !(std::isfinite(p.preExposure) && p.preExposure > 0.0f))
        return refuse("the pre-exposure is not a positive number");

    if (output == color || output == motion || output == depth)
        return refuse("input/output texture aliasing is not supported");

    d4r::GameFrame game;
    using Source = EngineFrameSource;
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    Source sources[5] = {
        {color, &game.color, p.colorBaseX, p.colorBaseY, p.renderWidth, p.renderHeight, inputState,
         D3D12_RESOURCE_STATE_COPY_SOURCE},
        {depth, &game.depth, p.depthBaseX, p.depthBaseY, p.renderWidth, p.renderHeight, depthState,
         D3D12_RESOURCE_STATE_COPY_SOURCE},
        {motion, &game.motion, p.mvBaseX, p.mvBaseY, displayMotion ? feature.outWidth : p.renderWidth,
         displayMotion ? feature.outHeight : p.renderHeight, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {output, &game.output, p.outputBaseX, p.outputBaseY, feature.outWidth, feature.outHeight, outputState,
         D3D12_RESOURCE_STATE_COPY_DEST},
        {exposure, &game.exposure, 0, 0, 1, 1, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
    };
    const size_t sourceCount = gameExposure ? 5 : 4;
    for (size_t index = 0; index < sourceCount; ++index)
    {
        Source& source = sources[index];
        UINT64 handle = 0, offset = 0;
        D3D12_RESOURCE_DESC desc;
        source.resource->GetDesc(&desc);
        if (FAILED(g_vk.interop->GetVulkanResourceInfo1(source.resource, &handle, &offset, &source.image->format)) ||
            handle == 0 || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
            source.x + source.width > desc.Width || source.y + source.height > desc.Height)
            return refuse("a texture is not a single-sample 2D image covering its rectangle");
        source.image->image = reinterpret_cast<VkImage>(handle);
        source.image->x = source.x;
        source.image->y = source.y;
        const bool isOutput = source.image == &game.output;
        prepare_engine_view(feature, source.resource, desc, *source.image, isOutput,
                            (source.x == 0 && source.y == 0) || model->hasDirectOrigins(), source.copyState);
        g_vk.interop->GetVulkanImageLayout(source.resource, source.copyState, &source.image->layout);
    }
    if (gameExposure && (game.exposure.view == VK_NULL_HANDLE || (game.exposure.format != VK_FORMAT_R32_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R16_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R32G32B32A32_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R16G16B16A16_SFLOAT)))
        return refuse("the exposure texture must be a float texture the engine samples directly");
    if (game.depth.format == VK_FORMAT_D32_SFLOAT || game.depth.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        game.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!d4r::gameColorFormat(game.color.format) || !d4r::gameMotionFormat(game.motion.format) ||
        !d4r::gameDepthFormat(game.depth.format, game.depth.aspect) || !d4r::gameOutputFormat(game.output.format))
    {
        logf("engine backend: Vulkan formats colour %d, depth %d, motion %d, output %d", game.color.format,
             game.depth.format, game.motion.format, game.output.format);
        return refuse("a texture format is not implemented");
    }
    if (frame <= 2)
        logf("engine direct textures: color %d, motion %d, depth %d, output %d (format %d, RGB10A2 pipeline %d)",
             game.color.view != VK_NULL_HANDLE, game.motion.view != VK_NULL_HANDLE,
             game.depth.view != VK_NULL_HANDLE, game.output.view != VK_NULL_HANDLE,
             game.output.format, model->hasRgb10Output());
    game.settings.renderWidth = p.renderWidth;
    game.settings.renderHeight = p.renderHeight;
    game.settings.jitter[0] = p.jitterX;
    game.settings.jitter[1] = p.jitterY;
    game.settings.displayMotion = displayMotion;
    // The engine takes motion in output pixels; MV.Scale gives render pixels (display-resolution vectors are output pixels already).
    const float toOutput = displayMotion ? 1.0f : static_cast<float>(feature.outWidth) / static_cast<float>(p.renderWidth);
    const float toOutputY = displayMotion ? 1.0f : static_cast<float>(feature.outHeight) / static_cast<float>(p.renderHeight);
    game.settings.motionScale[0] = p.mvScaleX * toOutput;
    game.settings.motionScale[1] = p.mvScaleY * toOutputY;
    // MVJittered: NGX adds the change of the jitter (render pixels) to the vectors before scaling them
    // (observed with MV.Scale 1: offset = jitter - previous jitter, the first frame's previous being 0).
    // The offset is added to the raw vector, before motionScale's own output-pixel conversion, so it carries
    // no extra output/render factor in either motion mode.
    if ((feature.flags & 0x4) != 0 && p.mvScaleX != 0.0f && p.mvScaleY != 0.0f)
    {
        game.settings.motionOffset[0] = (p.jitterX - feature.engineJitter[0]) / p.mvScaleX;
        game.settings.motionOffset[1] = (p.jitterY - feature.engineJitter[1]) / p.mvScaleY;
    }
    game.settings.reset = p.reset != 0 || feature.engine == nullptr;
    game.settings.depthInverted = (feature.flags & 0x8) != 0;
    game.settings.autoExposure = !ldr && autoExposure; // LDR: no exposure at all
    game.settings.exposureScale = ldr ? 1.0f : p.exposureScale;
    // NGX's automatic K measurement scales the luma by 1 / pre-exposure before the logarithm
    // (cuda_luma_convert_kernel), so the stored exposure is pre times the raw measurement and the stage's
    // 1 / pre factor recovers it; the game texture path applies the same stage factor.
    game.settings.preExposure = ldr ? 1.0f : p.preExposure;
    try
    {
        if (feature.engine == nullptr)
        {
            const size_t tokenBytes = static_cast<size_t>(d4r::Engine::networkTokenBytes(feature.outWidth, feature.outHeight));
            const size_t headBytes = static_cast<size_t>(d4r::Engine::networkHeadBytes(feature.outWidth, feature.outHeight));
            const bool hip = create_engine_network(feature, ldr ? g_engine.directoryLdr : g_engine.directory, "K", true,
                                                   feature.outWidth, feature.outHeight, tokenBytes, headBytes);
            const d4r::ExternalNetwork external = {feature.engineTokens.buffer, feature.engineHead.buffer};
            feature.engine = std::make_unique<d4r::GameUpscaler>(*model, g_vk.physical, g_vk.device,
                                                                 feature.outWidth, feature.outHeight,
                                                                 std::max(feature.width, p.renderWidth),
                                                                 std::max(feature.height, p.renderHeight),
                                                                 hip ? &external : nullptr);
            logf("engine backend: feature %u runs preset K in the game's command list (%ux%u -> %ux%u), network on %s",
                 feature.handle.Id, p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight,
                 hip ? "the native HIP layers (list split around it)" : "Vulkan");
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend: %s", error.what());
        release_engine_network(feature);
        return refuse("the engine could not be created");
    }
    if (feature.engineNet != nullptr)
    {
        // The native HIP network: the game's list is split around it. Its native completion callback releases
        // the second part, which records the final store (or the conversion blit for incompatible outputs).
        auto recordPart = [&](bool front) -> const char* {
            VkCommandBuffer command = VK_NULL_HANDLE;
            if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &command)))
                return "BeginVkCommandBufferInterop failed";
            const char* failure = nullptr;
            try
            {
                if (front)
                {
                    feature.engine->recordFront(command, game);
                    // The HIP network's stream waits for this frame's number (record_engine_input_ready): the
                    // fill is part of the first half, so the network starts the moment the front's GPU work ends.
                    if (feature.engineGpuWait.load())
                        record_engine_input_ready(feature, command, frame);
                }
                else
                    feature.engine->recordBack(command, game);
            }
            catch (const std::exception& error)
            {
                logf("engine backend (K): %s", error.what());
                failure = "the frame could not be recorded";
            }
            if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && failure == nullptr)
                failure = "EndVkCommandBufferInterop failed";
            return failure;
        };
        auto abandon = [&](const char* reason) {
            feature.engineRefused = true;
            logf("engine backend (K): %s; feature %u uses the CUDA backend after frame %u", reason, feature.handle.Id, frame);
            return true;
        };
        return record_engine_network_frame(feature, list, sources, frame, "K", game.settings.reset,
                                           game.settings.motionScale[0], game.settings.motionScale[1], p.jitterX,
                                           p.jitterY, recordPart, refuse, abandon);
    }
    if (FAILED(g_vk.lifetime->RetainExternalResources(list, feature.resources)))
        return refuse("the command list cannot retain the engine's resources");
    for (const Source& source : sources)
        transition(list, source.resource, source.state, source.copyState);
    VkCommandBuffer command = VK_NULL_HANDLE;
    bool recorded = SUCCEEDED(g_vk.interop->BeginVkCommandBufferInterop(list, &command));
    const char* failure = recorded ? nullptr : "BeginVkCommandBufferInterop failed";
    if (recorded)
    {
        try
        {
            feature.engine->record(command, game);
        }
        catch (const std::exception& error)
        {
            logf("engine backend: %s", error.what());
            failure = "the frame could not be recorded";
            recorded = false;
        }
        if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && recorded)
        {
            failure = "EndVkCommandBufferInterop failed";
            recorded = false;
        }
    }
    for (const Source& source : sources)
        transition(list, source.resource, source.copyState, source.state);
    if (!recorded)
        return refuse(failure);
    feature.engineJitter[0] = p.jitterX;
    feature.engineJitter[1] = p.jitterY;
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f, render %ux%u)", frame,
             p.jitterX, p.jitterY, game.settings.reset, p.mvScaleX, p.mvScaleY, p.renderWidth, p.renderHeight);
    return true;
}

static NgxResult evaluate(ID3D12GraphicsCommandList* list, const NgxHandle* handle, void* parameters)
{
    std::lock_guard<std::mutex> apiLock(g_featureApiMutex);
    FrameTiming timing;
    timing.enabled = profile_enabled();
    if (timing.enabled)
    {
        timing.gameStart = ProfileClock::now();
        static thread_local ProfileClock::time_point lastGameStart;
        if (lastGameStart.time_since_epoch().count() != 0)
            timing.gameInterval = profile_ms(lastGameStart, timing.gameStart);
        lastGameStart = timing.gameStart;
    }
    Feature* feature = find_feature(handle);
    if (feature == nullptr || feature->nativeVulkan || list == nullptr || parameters == nullptr)
        return NGX_FAIL_INVALID_PARAMETER;
    ID3D12Resource* color = get_resource(parameters, "Color");
    ID3D12Resource* depth = get_resource(parameters, "Depth");
    ID3D12Resource* motion = get_resource(parameters, "MotionVectors");
    ID3D12Resource* output = get_resource(parameters, "Output");
    ID3D12Resource* exposure = get_resource(parameters, "ExposureTexture");
    const uint32_t frame = ++feature->frame;
    if (color == nullptr || depth == nullptr || motion == nullptr || output == nullptr)
    {
        if (frame <= 3)
            logf("evaluate frame %u: missing resource color=%p depth=%p mv=%p output=%p", frame, color, depth, motion,
                 output);
        return NGX_FAIL_INVALID_PARAMETER;
    }

    D3D12_RESOURCE_DESC colorDesc, depthDesc, motionDesc, outputDesc;
    color->GetDesc(&colorDesc);
    depth->GetDesc(&depthDesc);
    motion->GetDesc(&motionDesc);
    output->GetDesc(&outputDesc);
    if (frame == 1)
        logf("evaluate: color %llux%u fmt=%d, depth %llux%u fmt=%d, mv %llux%u fmt=%d, output %llux%u fmt=%d, "
             "exposure=%p",
             colorDesc.Width, colorDesc.Height, colorDesc.Format, depthDesc.Width, depthDesc.Height, depthDesc.Format,
             motionDesc.Width, motionDesc.Height, motionDesc.Format, outputDesc.Width, outputDesc.Height,
             outputDesc.Format, exposure);
    const bool colorSupported = supported_input(Plane::Color, colorDesc.Format);
    const bool depthSupported = supported_input(Plane::Depth, depthDesc.Format);
    const bool motionSupported = supported_input(Plane::Motion, motionDesc.Format);
    const bool outputSupported = supported_output(outputDesc.Format);
    if (!colorSupported || !depthSupported || !motionSupported || !outputSupported)
    {
        // Log the first failures even if they occur well after startup, then periodically.
        const uint32_t rejected = ++feature->rejectedFormatCount;
        if (rejected <= 3 || rejected % 120 == 0)
            logf("evaluate feature %u frame %u: unsupported format (rejection %u): "
                 "color %llux%u fmt=%d%s, depth %llux%u fmt=%d%s, "
                 "mv %llux%u fmt=%d%s, output %llux%u fmt=%d%s",
                 feature->handle.Id, frame, rejected,
                 colorDesc.Width, colorDesc.Height, colorDesc.Format, colorSupported ? "" : " (unsupported)",
                 depthDesc.Width, depthDesc.Height, depthDesc.Format, depthSupported ? "" : " (unsupported)",
                 motionDesc.Width, motionDesc.Height, motionDesc.Format, motionSupported ? "" : " (unsupported)",
                 outputDesc.Width, outputDesc.Height, outputDesc.Format, outputSupported ? "" : " (unsupported)");
        return NGX_FAIL_UNSUPPORTED_FORMAT;
    }
    if (exposure != nullptr)
    {
        D3D12_RESOURCE_DESC exposureDesc;
        exposure->GetDesc(&exposureDesc);
        if (!supported_input(Plane::Exposure, exposureDesc.Format))
            exposure = nullptr;
    }

    FrameParams p;
    p.jitterX = get_float_or(parameters, "Jitter.Offset.X", 0.0f);
    p.jitterY = get_float_or(parameters, "Jitter.Offset.Y", 0.0f);
    p.mvScaleX = get_float_or(parameters, "MV.Scale.X", 1.0f);
    p.mvScaleY = get_float_or(parameters, "MV.Scale.Y", 1.0f);
    p.sharpness = get_float_or(parameters, "Sharpness", 0.0f);
    p.preExposure = get_float_or(parameters, "DLSS.Pre.Exposure", 1.0f);
    p.exposureScale = get_float_or(parameters, "DLSS.Exposure.Scale", 1.0f);
    p.frameTime = get_float_or(parameters, "FrameTimeDeltaInMsec", 16.6f);
    p.reset = get_int_or(parameters, "Reset", 0);
    p.invertX = get_int_or(parameters, "DLSS.Indicator.Invert.X.Axis", 0);
    p.invertY = get_int_or(parameters, "DLSS.Indicator.Invert.Y.Axis", 0);
    p.renderWidth = get_uint_or(parameters, "DLSS.Render.Subrect.Dimensions.Width", 0);
    p.renderHeight = get_uint_or(parameters, "DLSS.Render.Subrect.Dimensions.Height", 0);
    if (p.renderWidth == 0 || p.renderHeight == 0)
    {
        p.renderWidth = feature->width;
        p.renderHeight = feature->height;
    }
    p.colorBaseX = get_uint_or(parameters, "DLSS.Input.Color.Subrect.Base.X", 0);
    p.colorBaseY = get_uint_or(parameters, "DLSS.Input.Color.Subrect.Base.Y", 0);
    p.depthBaseX = get_uint_or(parameters, "DLSS.Input.Depth.Subrect.Base.X", 0);
    p.depthBaseY = get_uint_or(parameters, "DLSS.Input.Depth.Subrect.Base.Y", 0);
    p.mvBaseX = get_uint_or(parameters, "DLSS.Input.MV.Subrect.Base.X", 0);
    p.mvBaseY = get_uint_or(parameters, "DLSS.Input.MV.Subrect.Base.Y", 0);
    p.outputBaseX = get_uint_or(parameters, "DLSS.Output.Subrect.Base.X", 0);
    p.outputBaseY = get_uint_or(parameters, "DLSS.Output.Subrect.Base.Y", 0);
    p.hasExposure = exposure != nullptr;

    if (engine_requested() && engine_evaluate(*feature, list, color, depth, motion, output, exposure, p, frame))
        return NGX_SUCCESS;

    if (feature->cudaCreationDeferred)
    {
        logf("engine backend: creating CUDA fallback for feature %u at frame %u", feature->handle.Id, frame);
        const NgxResult created = create_cuda_feature(*feature);
        if (created != NGX_SUCCESS) return created;
        p.reset = 1; // the freshly created CUDA feature has none of the Vulkan engine's history
        logf("engine backend: CUDA fallback initialized; temporal history reset");
    }

    const int slotIndex = static_cast<int>(frame % kSlots);
    InputSlot& slot = feature->inputs[slotIndex];
    // Throttle the game to DLSS speed when the pipeline still owns this slot
    // or already holds the maximum number of frames.
    static const int maxInFlight = static_cast<int>(env_uint("D4R_SHIM_MAX_IN_FLIGHT", 3)) > 0
                                       ? static_cast<int>(env_uint("D4R_SHIM_MAX_IN_FLIGHT", 3)) : 3;
    // Split frames reuse frame N's result slot for frame N + kOutputSlots.
    const int cap = feature->split && maxInFlight > kOutputSlots - 1 ? kOutputSlots - 1 : maxInFlight;
    const auto waitStart = std::chrono::steady_clock::now();
    while (slot.busy.load() || feature->inFlight.load() >= cap)
    {
        if (std::chrono::steady_clock::now() - waitStart > std::chrono::seconds(2))
        {
            logf("frame %u: input slot %d busy or %d frames in flight; skipping evaluation", frame, slotIndex, feature->inFlight.load());
            return NGX_SUCCESS;
        }
        d4r_sleep_us(200);
    }
    if (timing.enabled)
        timing.slotWait = profile_ms(waitStart, ProfileClock::now());

    if (g_vk.lifetime == nullptr ||
        FAILED(g_vk.lifetime->RetainExternalResources(list, feature->resources)))
    {
        logf("frame %u: unable to retain command-list resources; evaluation not recorded", frame);
        return NGX_FAIL_PLATFORM_ERROR;
    }

    // VRAM interop when every resource qualifies (decided once per feature).
    ID3D12Resource* const inputs[4] = {color, depth, motion, exposure};
    const Plane planes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
    const int inputCount = exposure != nullptr ? 4 : 3;
    VramCopy vramInputs[4], vramOutput;
    if ((!feature->vramDecided || feature->vram) && vram_interop_available())
    {
        p.vram = describe_vram_copy(output, Plane::Color, vramOutput, true);
        for (int index = 0; index < inputCount && p.vram; ++index)
            p.vram = describe_vram_copy(inputs[index], planes[index], vramInputs[index]);
    }
    if (!feature->vramDecided)
    {
        feature->vramDecided = true;
        feature->vram = p.vram;
        feature->split = p.vram && env_uint("D4R_SHIM_SPLIT_FRAME", 0) != 0 && g_vk.split != nullptr &&
                         (feature->splitSemaphore != VK_NULL_HANDLE || create_split_semaphore(*feature));
        feature->linearInputs = feature->split && env_uint("D4R_SHIM_LINEAR_INPUTS", 0) != 0 &&
                                g.cu.registerLinearTexture != nullptr;
        if (feature->linearInputs)
            logf("linear inputs: NGX samples the interop buffers directly");
        logf("VRAM interop %s for this feature%s%s", p.vram ? "on" : "off",
             feature->split ? ", presenting each frame's own result (split frames)" : "",
             p.vram ? ", waiting for input readiness with CPU sleeps" : "");
    }
    else if (feature->vram && !p.vram)
    {
        logf("frame %u: resources no longer qualify for VRAM interop; staging through host memory from now on", frame);
        drain_pipeline();
        {
            std::lock_guard<std::mutex> lock(feature->splitMutex);
            feature->split = false;
        }
        feature->vram = false;
        feature->linearInputs = false;
        feature->latestOutput = -1;
        feature->outputFormat = DXGI_FORMAT_UNKNOWN; // recreates the output staging below
    }
    const bool verify = p.vram && env_uint("D4R_SHIM_VRAM_VERIFY", 0) != 0;
    if (p.vram)
    {
        for (int index = 0; index < inputCount; ++index)
        {
            HostPlane& geometry = slot.host[index]; // geometry only; the bytes stay in VRAM
            geometry.width = vramInputs[index].width;
            geometry.height = vramInputs[index].height;
            geometry.rowBytes = canonical_texel_bytes(planes[index]) * geometry.width;
            if (feature->linearInputs)
                geometry.rowBytes = (geometry.rowBytes + 255) & ~static_cast<size_t>(255); // texture pitch alignment
            if (!ensure_vram_buffer(*feature, slot.vram[index], geometry.size()))
                return NGX_FAIL_PLATFORM_ERROR;
        }
        // M's downsample kernel has only formatted stores, which the native store can pack (kernels/tex/tex_common.h).
        p.packedOutput = feature->split && feature->outputDirectAllowed && feature->preset == 13 && vramOutput.convert &&
                         outputDesc.Format == DXGI_FORMAT_R10G10B10A2_UNORM && vramOutput.width % 2 == 0 &&
                         env_uint("D4R_SHIM_OUTPUT_DIRECT", 0) != 0 && env_uint("D4R_SHIM_OUTPUT_PACKED", 0) != 0 &&
                         feature->packedReady.load();
        static bool packedLogged = false;
        if (p.packedOutput && !packedLogged)
        {
            packedLogged = true;
            logf("frame %u: packed output (the output kernel stores R10G10B10A2 texels; no conversion blit)", frame);
        }
        InPlaceTexture outputTexture;
        const size_t outputTexelBytes = p.packedOutput ? 4 : 8;
        if (feature->split && feature->outputDirectAllowed && (!vramOutput.convert || p.packedOutput) &&
            env_uint("D4R_SHIM_OUTPUT_DIRECT", 0) != 0 &&
            describe_in_place(output, outputTexture) && outputTexture.rowPitch % 8 == 0 &&
            outputTexture.rowPitch / 8 <= 0xffffu &&
            outputTexture.rowPitch >= static_cast<size_t>(vramOutput.width) * outputTexelBytes)
        {
            p.outputInPlace = outputTexture.texels;
            p.outputInPlacePitch = static_cast<uint32_t>(outputTexture.rowPitch);
        }
        static bool outputInPlaceLogged = false;
        if (p.outputInPlace != 0 && !outputInPlaceLogged)
        {
            outputInPlaceLogged = true;
            logf("frame %u: output in place (the output kernel writes the game's %s texture; no game-side copy)", frame,
                 p.packedOutput ? "R10G10B10A2" : "RGBA16F");
        }
        static bool inPlaceLogged = false;
        if (g_vk.linear != nullptr && !inPlaceLogged)
        {
            inPlaceLogged = true;
            if (p.outputInPlace == 0)
            {
                UINT64 memory = 0, size = 0, offset = 0, pitch = 0;
                const HRESULT linear = g_vk.linear->GetVulkanLinearImageInfo(output, &memory, &size, &offset, &pitch);
                logf("in-place: output copied on the first frame (split %d, direct output allowed %d, format %d, needs conversion %d, "
                     "linear exportable texture %d, row pitch %llu)", static_cast<int>(feature->split), feature->outputDirectAllowed,
                     static_cast<int>(outputDesc.Format), vramOutput.convert, SUCCEEDED(linear),
                     static_cast<unsigned long long>(pitch));
            }
        }
        if (vramInputs[0].convert && !ensure_conversion_image(*feature, feature->colorConversion,
                                                               vramInputs[0].width, vramInputs[0].height,
                                                               VK_FORMAT_R16G16B16A16_SFLOAT))
            return NGX_FAIL_PLATFORM_ERROR;
        if (vramInputs[1].convert && !ensure_conversion_image(*feature, feature->depthConversion,
                                                               vramInputs[1].width, vramInputs[1].height,
                                                               VK_FORMAT_R32_SFLOAT))
            return NGX_FAIL_PLATFORM_ERROR;
        if (vramInputs[2].convert && !ensure_conversion_image(*feature, feature->motionConversion,
                                                               vramInputs[2].width, vramInputs[2].height,
                                                               VK_FORMAT_R16G16_SFLOAT))
            return NGX_FAIL_PLATFORM_ERROR;
        if (inputCount == 4 && vramInputs[3].convert &&
            !ensure_conversion_image(*feature, feature->exposureConversion,
                                     vramInputs[3].width, vramInputs[3].height, VK_FORMAT_R32_SFLOAT))
            return NGX_FAIL_PLATFORM_ERROR;
        if (vramOutput.convert && !ensure_conversion_image(*feature, feature->outputConversion,
                                                             vramOutput.width, vramOutput.height,
                                                             VK_FORMAT_R16G16B16A16_SFLOAT))
            return NGX_FAIL_PLATFORM_ERROR;
    }
    if ((!p.vram || verify) &&
        (!ensure_staging(slot.color, color, D3D12_HEAP_TYPE_READBACK) ||
         !ensure_staging(slot.depth, depth, D3D12_HEAP_TYPE_READBACK) ||
         !ensure_staging(slot.motion, motion, D3D12_HEAP_TYPE_READBACK) ||
         (exposure != nullptr && !ensure_staging(slot.exposure, exposure, D3D12_HEAP_TYPE_READBACK))))
        return NGX_FAIL_PLATFORM_ERROR;
    if (outputDesc.Format != feature->outputFormat || outputDesc.Width != feature->outputWidth ||
        outputDesc.Height != feature->outputHeight)
    {
        // A new output size invalidates earlier results; let queued work finish
        // before the output buffers go away.
        drain_pipeline();
        feature->latestOutput = -1;
        for (OutputSlot& outputSlot : feature->outputs)
        {
            outputSlot.staging.retire();
            if (p.vram ? !ensure_vram_buffer(*feature, outputSlot.vram, static_cast<size_t>(outputDesc.Width) * 8 *
                                                                            outputDesc.Height)
                       : !ensure_staging(outputSlot.staging, output, D3D12_HEAP_TYPE_UPLOAD))
                return NGX_FAIL_PLATFORM_ERROR;
        }
        feature->outputFormat = outputDesc.Format;
        feature->outputWidth = static_cast<UINT>(outputDesc.Width);
        feature->outputHeight = outputDesc.Height;
    }

    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    const auto inputRecordStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    if (p.vram)
    {
        const D3D12_RESOURCE_STATES states[4] = {inputState, depthState, inputState, inputState};
        for (int index = 0; index < inputCount; ++index)
            transition(list, inputs[index], states[index], D3D12_RESOURCE_STATE_COPY_SOURCE);
        const bool recorded = record_vram_inputs(*feature, list, slot, vramInputs, inputCount);
        for (int index = 0; index < inputCount; ++index)
            transition(list, inputs[index], D3D12_RESOURCE_STATE_COPY_SOURCE, states[index]);
        if (!recorded)
        {
            logf("frame %u: BeginVkCommandBufferInterop failed", frame);
            return NGX_FAIL_PLATFORM_ERROR;
        }
    }
    if (!p.vram || verify)
    {
        copy_to_staging(list, color, slot.color, inputState);
        copy_to_staging(list, depth, slot.depth, depthState);
        copy_to_staging(list, motion, slot.motion, inputState);
        if (exposure != nullptr)
            copy_to_staging(list, exposure, slot.exposure, inputState);
    }
    if (timing.enabled)
        timing.inputRecord = profile_ms(inputRecordStart, ProfileClock::now());

    // Frame marker, written once the copies above have completed.
    ID3D12GraphicsCommandList2* list2 = nullptr;
    if (FAILED(list->QueryInterface(__uuidof(ID3D12GraphicsCommandList2), reinterpret_cast<void**>(&list2))))
    {
        logf("ID3D12GraphicsCommandList2 unavailable; cannot place frame marker");
        return NGX_FAIL_PLATFORM_ERROR;
    }
    const D3D12_GPU_VIRTUAL_ADDRESS markerAddress = feature->marker->GetGPUVirtualAddress();
    D3D12_WRITEBUFFERIMMEDIATE_PARAMETER markers[] = {
        {markerAddress + sizeof(uint32_t) * (1 + slotIndex), frame},
        {markerAddress, frame},
    };
    D3D12_WRITEBUFFERIMMEDIATE_MODE modes[] = {
        D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT,
    };
    list2->WriteBufferImmediate(2, markers, modes);
    list2->Release();

    // Present the most recent finished DLSS result.
    int latest;
    {
        std::lock_guard<std::mutex> lock(feature->outputMutex);
        latest = feature->latestOutput.load();
        if (latest >= 0)
        {
            feature->outputs[latest].lastReadFrame = frame;
            if (timing.enabled)
                timing.presentedFrame = feature->outputs[latest].producedFrame.load();
        }
    }
    const auto outputRecordStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    if (p.vram && feature->split)
        p.split = SUCCEEDED(g_vk.split->SplitCommandListForExternalWait(
            list, reinterpret_cast<UINT64>(feature->splitSemaphore), frame));
    if (!p.split)
    {
        p.outputInPlace = 0; // only a split frame waits for the result before the game reads its texture
        p.packedOutput = false;
    }
    if (p.split)
    {
        // Runs in the second half of the list, once frame's result is in place.
        const int target = static_cast<int>(frame % kOutputSlots);
        if (p.outputInPlace == 0)
        {
            if (p.packedOutput)
                vramOutput.convert = false; // the buffer already holds the texture's own texels
            transition(list, output, outputState, D3D12_RESOURCE_STATE_COPY_DEST);
            if (!record_vram_output(list, feature->outputs[target].vram, vramOutput, feature->outputConversion))
                logf("frame %u: BeginVkCommandBufferInterop failed for the output", frame);
            transition(list, output, D3D12_RESOURCE_STATE_COPY_DEST, outputState);
        }
        if (timing.enabled)
            timing.presentedFrame = frame;
    }
    else if (latest >= 0 && p.vram)
    {
        transition(list, output, outputState, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!record_vram_output(list, feature->outputs[latest].vram, vramOutput, feature->outputConversion))
            logf("frame %u: BeginVkCommandBufferInterop failed for the output", frame);
        transition(list, output, D3D12_RESOURCE_STATE_COPY_DEST, outputState);
    }
    else if (latest >= 0)
    {
        OutputSlot& outputSlot = feature->outputs[latest];
        transition(list, output, outputState, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION destination = {};
        destination.pResource = output;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION source = {};
        source.pResource = outputSlot.staging.buffer;
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = outputSlot.staging.layout;
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        transition(list, output, D3D12_RESOURCE_STATE_COPY_DEST, outputState);
    }
    if (timing.enabled)
    {
        timing.outputRecord = profile_ms(outputRecordStart, ProfileClock::now());
        timing.queueDepth = g.prep.pending() + g.worker.pending() + g.finish.pending() + 1;
        timing.queuedAt = ProfileClock::now();
        timing.gameCall = profile_ms(timing.gameStart, timing.queuedAt);
    }

    slot.busy = true;
    feature->inFlight.fetch_add(1);
    if (feature->split)
    {
        std::lock_guard<std::mutex> lock(feature->splitMutex);
        feature->splitCompletion.admit(frame);
    }
    g.prep.post([feature, slotIndex, frame, p, timing] { prepare_inputs(feature, slotIndex, frame, p, timing); });
    if (frame <= 3 || frame % 600 == 0 || GetEnvironmentVariableA("D4R_SHIM_TRACE_FRAMES", nullptr, 0) != 0)
        logf("evaluate frame %u queued (slot %d, showing result %d, jitter %.6f,%.6f, reset %d, "
             "mvScale %.3f,%.3f, preExposure %.3f, exposureScale %.3f, frameTime %.3f, sharpness %.3f)",
             frame, slotIndex, latest, p.jitterX, p.jitterY, p.reset, p.mvScaleX, p.mvScaleY,
             p.preExposure, p.exposureScale, p.frameTime, p.sharpness);
    return NGX_SUCCESS;
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList* list, const NgxHandle* handle,
                                                     void* parameters, void*)
{
    return evaluate(list, handle, parameters);
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_EvaluateFeature_C(ID3D12GraphicsCommandList* list, const NgxHandle* handle,
                                                       void* parameters, void*)
{
    return evaluate(list, handle, parameters);
}

static void release_feature(Feature* feature)
{
    // A queued GPU-side input wait for a frame that never ran would stall the network's stream: release it
    // before the drain below waits for the worker (the worker also releases waits it sees stuck).
    if (feature->engineNet != nullptr)
    {
        const uint32_t waiting = feature->engineWaitFrame.load();
        if (waiting != 0)
            release_engine_input_wait(*feature, waiting, "the feature is being released", false);
    }
    // Wait for outstanding evaluations, then free CUDA resources on the worker.
    drain_pipeline();
    g.worker.call([feature] {
        if (feature->cudaHandle != nullptr)
            g.ngx.releaseFeature(feature->cudaHandle);
        if (feature->cudaParams != nullptr)
            g.ngx.destroyParameters(feature->cudaParams);
        for (InputSlot& slot : feature->inputs)
        {
            for (CudaObject& texture : slot.linearTexture)
                if (texture != 0)
                    g.cu.texObjectDestroy(texture), texture = 0;
            if (slot.dilatedTexture != 0)
                g.cu.texObjectDestroy(slot.dilatedTexture), slot.dilatedTexture = 0;
            if (slot.dilatedMotion != 0)
                g.cu.memFree(slot.dilatedMotion), slot.dilatedMotion = 0;
        }
        for (CudaImage* image : {&feature->color, &feature->depth, &feature->motion, &feature->exposure})
        {
            if (image->object != 0)
                g.cu.texObjectDestroy(image->object);
            if (image->array != nullptr)
                g.cu.arrayDestroy(image->array);
        }
        if (feature->output.object != 0)
            g.cu.surfObjectDestroy(feature->output.object);
        if (feature->output.array != nullptr)
            g.cu.arrayDestroy(feature->output.array);
        if (feature->profileStart != nullptr)
            g.cu.eventDestroy(feature->profileStart);
        if (feature->profileEnd != nullptr)
            g.cu.eventDestroy(feature->profileEnd);
        if (feature->outputReadyEvent != nullptr)
            g.cu.eventDestroy(feature->outputReadyEvent);
        if (feature->scratch != 0)
            g.cu.memFree(feature->scratch);
        for (CudaDevicePtr buffer : feature->planeLinear)
            if (buffer != 0)
                g.cu.memFree(buffer);
        if (feature->outputLinear != 0)
            g.cu.memFree(feature->outputLinear);
        for (InputSlot& slot : feature->inputs)
            for (HostPlane& host : slot.host)
                release_host(host);
        for (HostPlane& host : feature->outputHost)
            release_host(host);
        return 0;
    });
    {
        std::lock_guard<std::mutex> lock(g_featuresMutex);
        g_features.erase(std::remove(g_features.begin(), g_features.end(), feature), g_features.end());
    }
    // Recorded lists retain this owner through their allocator. Reset/discard
    // or GPU completion followed by allocator destruction releases those refs.
    if (feature->resources != nullptr)
        feature->resources->Release();
    else
    {
        release_engine_network(*feature);
        delete feature;
    }
}

static void destroy_recorded_resources(Feature* feature)
{
    for (InputSlot& slot : feature->inputs)
        for (Staging* staging : {&slot.color, &slot.depth, &slot.motion, &slot.exposure})
            staging->release();
    for (OutputSlot& slot : feature->outputs)
        slot.staging.release();
    release_engine_network(*feature);
    const bool engineSemaphore = !feature->vramDecided && feature->splitSemaphore != VK_NULL_HANDLE;
    if (g_vk.ready && feature->vramDecided)
        release_vram(*feature);
    else if (engineSemaphore) // created for the engine's HIP network; release_vram destroys it otherwise
        g_vk.destroySemaphore(g_vk.device, feature->splitSemaphore, nullptr);
    if (feature->marker != nullptr)
        feature->marker->Release();
    logf("feature %u recorded resources retired after allocator completion", feature->handle.Id);
    delete feature;
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_ReleaseFeature(NgxHandle* handle)
{
    std::lock_guard<std::mutex> apiLock(g_featureApiMutex);
    Feature* feature = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_featuresMutex);
        for (Feature* candidate : g_features)
            if (&candidate->handle == handle && !candidate->nativeVulkan && !candidate->retiring.exchange(true))
            {
                feature = candidate;
                break;
            }
    }
    if (feature == nullptr || feature->nativeVulkan)
        return NGX_FAIL_INVALID_PARAMETER;
    logf("NVSDK_NGX_D3D12_ReleaseFeature(id=%u) after %u completed frames", feature->handle.Id,
         feature->completedFrames.load());
    release_feature(feature);
    return NGX_SUCCESS;
}

static void finish_shutdown()
{
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!g.shutdownPending || g.resourceOwners.load() != 0)
        return;
    if (g_vk.ready)
        free_vram_pool();
    if (g.ngxInitialized)
    {
        drain_pipeline();
        g.worker.call([] { return g.ngx.shutdown(); });
    }
    g.ngxInitialized = false;
    g.shutdownPending = false;
    if (g.device != nullptr)
    {
        g.device->Release();
        g.device = nullptr;
    }
    logf("NGX shutdown completed after recorded resources retired");
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_Shutdown()
{
    std::lock_guard<std::mutex> apiLock(g_featureApiMutex);
    logf("NVSDK_NGX_D3D12_Shutdown");
    std::vector<Feature*> features;
    {
        std::lock_guard<std::mutex> lock(g_featuresMutex);
        features = g_features;
        for (Feature* feature : features)
            if (feature->nativeVulkan)
                return NGX_FAIL_INVALID_PARAMETER;
        for (Feature* feature : features)
            feature->retiring = true;
    }
    for (Feature* feature : features)
        release_feature(feature);
    {
        std::lock_guard<std::mutex> lock(g.mutex);
        if (!g.started)
            return NGX_SUCCESS;
        g.shutdownPending = true;
    }
    // Drain already queued cleanup before returning. If command allocators
    // still own resources, finish_shutdown leaves NGX alive until their last
    // reference retires; otherwise shutdown completes synchronously.
    g.cleanup.call([] { finish_shutdown(); return 0; });
    return NGX_SUCCESS;
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_Shutdown1(ID3D12Device*)
{
    return NVSDK_NGX_D3D12_Shutdown();
}

D4R_EXPORT NgxResult NVSDK_NGX_UpdateFeature(const void*, unsigned int)
{
    return NGX_SUCCESS;
}

#include "d4r_vulkan_frontend.inl"

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_selfModule = instance;
        DisableThreadLibraryCalls(instance);
    }
    else if (reason == DLL_PROCESS_DETACH)
        native_vk_process_exit();
    return TRUE;
}

// The DLL entry point (linked with --entry). At process termination (reserved != NULL) ExitProcess
// has already killed the worker and waiter threads, possibly inside winpthreads' global condition
// variable spinlock, so the CRT's static destructors (mutexes, condition variables) can spin forever
// and the game never exits. Release the GPU waits and skip all CRT teardown then, as Windows advises;
// the system reclaims everything. Every other notification goes through the CRT as usual.
extern "C" BOOL WINAPI DllMainCRTStartup(HANDLE, DWORD, LPVOID);
extern "C" BOOL WINAPI d4r_dll_entry(HANDLE instance, DWORD reason, LPVOID reserved)
{
    if (reason == DLL_PROCESS_DETACH && reserved != nullptr)
    {
        native_vk_process_exit();
        return TRUE;
    }
    return DllMainCRTStartup(instance, reason, reserved);
}
