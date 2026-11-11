// Experimental native Vulkan NGX frontend: d4r_vulkan_frontend.inl.
// Modified in this fork for CUDA Ray Reconstruction support and validation (2026).
// d4r_nvngx.dll: an NGX core replacement that implements the D3D12 DLSS
// entry points on top of the official NGX core's CUDA path (which runs on AMD
// through the Wine nvcuda bridge and ZLUDA). OptiScaler loads it through its
// NvngxPath setting.
//
// A D3D12 EvaluateFeature call only records into the game's still-open command
// list, while DLSS runs on a separate CUDA context. Each evaluation therefore:
//   1. records input copies into shared VRAM buffers (host readback fallback)
//      and a GPU-written frame marker on the game's command list,
//   2. queues CUDA work after that marker, evaluates NGX, and writes a result slot,
//   3. copies that slot into the output texture. Split command lists present
//      the current frame; without splitting the latest finished frame is used.
//
// Environment:
//   D4R_NGX_CORE        Windows path of the official _nvngx.dll (required)
//   D4R_NGX_FEATURE_DIR directory holding nvngx_dlss.dll (optional, added to
//                       the feature search paths)
//   D4R_SHIM_LOG        log file path (default: d4r_nvngx.log next to the DLL)
//   D4R_PROFILE=1       measure CPU stages and default-stream GPU event time
//   D4R_PROFILE_LOG_EVERY=N log the first 5 frames, then every Nth frame (default 60; 1 logs all)
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
//   GPU hand-off settings, all default 1 (=0 turns one off; each falls back by itself, and logs why, where its
//   prerequisites are missing). Ghost of Tsushima 1440p Quality K: game-queue DLSS span 3.05 -> 2.79 ms with
//   them, 2.43 ms together with the native LDR output kernels (kernels/build.sh); output frames unchanged.
//   D4R_SHIM_GPU_INPUT_WAIT NGX null-stream input wait (split VRAM, same-frame Super Sampling only)
//   D4R_SHIM_OUTPUT_RGBA8 validated native K output stores RGBA8 directly (needs the d4r-rgba8.txt sidecar)
//   D4R_SHIM_NATIVE_COMPLETION native null-stream callback signals/wakes split completion
//   D4R_SHIM_FUSE_RGBA8_INPUT combine RGBA8 color conversion and linear-buffer copy
//   D4R_SHIM_GPU_COMPLETION (needs GPU_INPUT_WAIT) the game's queue polls DLSS's completion word on the GPU
//                       instead of a CPU-signalled split (bounded poll)
//   D4R_PROFILE_GPU_EVENTS / _TIMESTAMPS / _LOG / _CPU_TIME=0 drop one part of D4R_PROFILE (bisection)
//   D4R_NGX_STREAM_PRIORITY=high|normal (bridge) run NGX on one priority HIP stream (default off; no gain measured)
//   D4R_SHIM_MARKER_POLL_US=N fixed CPU marker pacing; unset: adaptive (0 yields)
//   D4R_SHIM_VRAM_INTEROP=1 keep inputs and output in VRAM (see "VRAM interop")
//   D4R_ENGINE_NATIVE_D3D12=1 preset K on the native D3D12 engine (engine/d3d12) even on vkd3d-proton
//   D4R_SHIM_SPLIT_FRAME=1 with VRAM interop and the d4r vkd3d-proton patch,
//                       present each frame's own DLSS result (see "Split frames")
//   D4R_SHIM_INPLACE=1  experimental, with VKD3D_D4R_LINEAR_TEXTURES=1 in the patched vkd3d-proton: the output
//                       kernel writes the game's output texture itself (no game-side output copy).
//                       D4R_SHIM_OUTPUT_PACKED=1 lets M's output kernel store R10G10B10A2 texels itself.
//   (the DLSS-Denoiser library nvngx_dlssd.dll, which serves Ray Reconstruction, must sit
//                       next to this DLL: that is where the NGX core looks for it. It needs
//                       a build exporting NVSDK_NGX_CUDA_* (3.10.7); without it Ray
//                       Reconstruction reports itself unavailable and Super Sampling is
//                       unaffected.)
//   D4R_RR_ENABLE=1/0    Ask for Ray Reconstruction / forbid it. Unset (auto) offers it
//                       exactly when the denoiser loaded and initialised.
//   D4R_RR_BACKEND=dlss|fsr Which denoiser serves Ray Reconstruction: NVIDIA's (dlss, the default)
//                       or AMD's FSR Ray Regeneration 1.2 followed by Super Sampling (fsr; see
//                       d4r_fsr_rr_backend.inl). D4R_FSR_RR_DLL: the AMD denoiser DLL (default:
//                       amd_fidelityfx_denoiser_dx12.dll in the d4r folder).
//   D4R_RR_PRESET        Force the Ray Reconstruction render preset
//                       (NVSDK_NGX_RayReconstruction_Hint_Render_Preset: 4 = D, 5 = E, 6 = F).
//                       Independent of D4R_DLSS_PRESET, which only affects Super Sampling.

#include "d4r_nvngx_shim_internal.h"

// --- logging -----------------------------------------------------------------

static std::mutex g_logMutex;

static HANDLE g_log = INVALID_HANDLE_VALUE; // FILE_APPEND_DATA: other processes' lines land whole, in order

static void ensure_portable_config()
{
    std::call_once(g_portableOnce, load_portable_config);
}

// The log can be shared by more than one process: a launcher and the game it starts both load the DLL, and
// CP2077's benchmark relaunches itself. With a private file position (fopen "a" or "w") the second process
// truncated the first one's log and each wrote at its own offset, which left the tail out of timestamp order.
// Every line is written by one append (FILE_APPEND_DATA puts each WriteFile at the end of the file) and tagged
// with the writer's process id; a fresh log only starts when no other process has the file open for writing.

static HANDLE log_open_path(const wchar_t* path)
{
    if (g_logFresh)
    {
        // Without FILE_SHARE_WRITE this fails while another process is writing to the log: keep its lines
        const HANDLE fresh = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                                         FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fresh != INVALID_HANDLE_VALUE)
            CloseHandle(fresh);
    }
    return CreateFileW(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                       OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}

static void log_open()
{
    ensure_portable_config();
    wchar_t path[MAX_PATH] = {};
    DWORD length = GetEnvironmentVariableW(L"D4R_SHIM_LOG", path, MAX_PATH);
    if (length > 0 && length < MAX_PATH)
        g_log = log_open_path(path);
    else if (!g_portable.dir.empty()) // no log when the DLL's folder is unknown
        g_log = log_open_path((g_portable.dir + L"\\d4r_nvngx.log").c_str());
}

void logf(const char* format, ...)
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    static bool opened = false;
    if (!opened)
    {
        opened = true; // one attempt: a log that cannot be opened stays off rather than retrying every line
        log_open();
    }
    if (g_log == INVALID_HANDLE_VALUE)
        return;
    char line[4096];
    int used = std::snprintf(line, sizeof(line), "[%10.3f] [pid %lu] ", GetTickCount64() / 1000.0,
                             static_cast<unsigned long>(GetCurrentProcessId()));
    if (used < 0)
        return;
    va_list arguments;
    va_start(arguments, format);
    const int body = std::vsnprintf(line + used, sizeof(line) - used - 1, format, arguments);
    va_end(arguments);
    if (body > 0)
        used = std::min<int>(used + body, int(sizeof(line)) - 2); // a longer line is cut, not split
    line[used++] = '\n';
    DWORD written = 0;
    WriteFile(g_log, line, DWORD(used), &written, nullptr);
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

unsigned int env_uint(const char* name, unsigned int fallback)
{
    char value[64] = {};
    DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
    if (length == 0 || length >= sizeof(value) || value[0] < '0' || value[0] > '9')
        return fallback; // also signs and text: strtoul turns "-1" into 4294967295
    return static_cast<unsigned int>(std::strtoul(value, nullptr, 0));
}

std::string env_string(const char* name)
{
    char value[1024] = {};
    const DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
    return length > 0 && length < sizeof(value) ? std::string(value, length) : std::string();
}

bool profile_enabled()
{
    static const bool enabled = env_uint("D4R_PROFILE", 0) != 0;
    return enabled;
}

bool profile_log_enabled()
{
    static const bool enabled = env_uint("D4R_PROFILE_LOG", 1) != 0;
    return enabled;
}

static unsigned int profile_log_every()
{
    static const unsigned int every = []() {
        const std::string text = env_string("D4R_PROFILE_LOG_EVERY");
        char* end = nullptr;
        const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
        return !text.empty() && text[0] >= '0' && text[0] <= '9' && end != text.c_str() && *end == '\0' &&
                       value >= 1 && value <= UINT32_MAX
                   ? static_cast<unsigned int>(value) : 60u;
    }();
    return every;
}

bool profile_log_frame(uint32_t frame)
{
    return profile_log_enabled() && (frame <= 5 || frame % profile_log_every() == 0);
}

d4r::GameProfile engine_stage_profile(uint32_t frame, const char* name)
{
    if (!profile_enabled() || env_uint("D4R_PROFILE_GPU_TIMESTAMPS", 1) == 0)
        return {};
    return {[](const char* line) { logf("%s", line); }, name, frame, profile_log_frame(frame)};
}

double profile_ms(ProfileClock::time_point start, ProfileClock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// nvngx_dlssd.dll 310.9 and newer hold preset F (DLSS 4.5 Ray Reconstruction); unknown versions count as without.

bool denoiser_has_preset_f()
{
    return g.denoiserVersion[0] > 310 || (g.denoiserVersion[0] == 310 && g.denoiserVersion[1] >= 9);
}

double profile_thread_cpu_ms()
{
    // Wine's GetThreadTimes is tick-accounted (10 ms on this host), despite FILETIME's 100 ns units.
    // The Unix bridge reads the current worker's actual thread CPU clock. Older bridges report n/a.
    static const bool wanted = env_uint("D4R_PROFILE_CPU_TIME", 1) != 0; // 0: skip the Unix call (bisection)
    return wanted && g.cu.threadCpuTimeMs != nullptr ? g.cu.threadCpuTimeMs() : -1.0;
}

static std::mutex g_featureApiMutex;

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
    if (length == 0 || length >= MAX_PATH || core.size() < 3 || core[1] != L':' || exe[1] != L':')
        return core;
    // A core on another drive (an install in the Proton prefix's C: with the game on Steam's S: or on Z:) is
    // named on the game's drive first, through the Linux path both drives map to.
    if (towlower(core[0]) != towlower(exe[0]))
    {
        const std::string coreUnix = unix_path(core);
        std::string root = unix_path(std::wstring{exe[0], L':', L'\\'});
        if (!root.empty() && root.back() != '/')
            root += '/';
        if (coreUnix.empty() || root.empty() || coreUnix.compare(0, root.size(), root) != 0)
            return core;
        std::wstring rest = widen(coreUnix.substr(root.size()));
        for (wchar_t& c : rest)
            if (c == L'/')
                c = L'\\';
        core = std::wstring{exe[0], L':', L'\\'} + rest;
    }
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
    // The NGX core loads nvapi64.dll from System32 only (LOAD_LIBRARY_SEARCH_SYSTEM32), which on a Windows
    // AMD system has none; a module of that name already loaded is reused, as with the bridge. Under Proton
    // the prefix's DXVK-NVAPI is in System32 and this stays unset.
    wchar_t nvapiPath[MAX_PATH] = {};
    length = GetEnvironmentVariableW(L"D4R_NVAPI", nvapiPath, MAX_PATH);
    if (length > 0 && length < MAX_PATH && GetModuleHandleW(L"nvapi64.dll") == nullptr)
    {
        HMODULE nvapi = LoadLibraryW(nvapiPath);
        logf("preloaded NVAPI %ls: %p (error %lu)", nvapiPath, static_cast<void*>(nvapi),
             nvapi != nullptr ? 0ul : GetLastError());
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
    if (!load_export(g.cuda, "cuCtxGetDevice", g.cu.ctxGetDevice))
        g.cu.ctxGetDevice = nullptr;
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
    g.cu.streamWaitValue32Eq = reinterpret_cast<decltype(g.cu.streamWaitValue32Eq)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rStreamWaitValue32Eq")));
    g.cu.ngxWriteValue32 = reinterpret_cast<decltype(g.cu.ngxWriteValue32)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxWriteValue32")));
    g.cu.ngxArmCompletion = reinterpret_cast<decltype(g.cu.ngxArmCompletion)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxArmCompletion")));
    g.cu.ngxCompletionFired = reinterpret_cast<decltype(g.cu.ngxCompletionFired)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxCompletionFired")));
    if (g.cu.ngxArmCompletion == nullptr || g.cu.ngxCompletionFired == nullptr)
        g.cu.ngxArmCompletion = nullptr, g.cu.ngxCompletionFired = nullptr;
    g.cu.threadCpuTimeMs = reinterpret_cast<decltype(g.cu.threadCpuTimeMs)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rThreadCpuTimeMs")));
    if (!load_export(g.cuda, "d4rSetArrayRedirect", g.cu.setArrayRedirect))
        g.cu.setArrayRedirect = nullptr;
    if (!load_export(g.cuda, "d4rOutputKernelNative", g.cu.outputKernelNative))
        g.cu.outputKernelNative = nullptr;
    g.cu.outputKernelRgba8 = reinterpret_cast<decltype(g.cu.outputKernelRgba8)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rOutputKernelRgba8")));
    g.cu.timelineCreate = reinterpret_cast<decltype(g.cu.timelineCreate)>(reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxTimelineCreate")));
    g.cu.timelinePrepare = reinterpret_cast<decltype(g.cu.timelinePrepare)>(reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxTimelinePrepare")));
    g.cu.timelineCancel = reinterpret_cast<decltype(g.cu.timelineCancel)>(reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxTimelineCancel")));
    g.cu.timelineQueue = reinterpret_cast<decltype(g.cu.timelineQueue)>(reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxTimelineQueue")));
    g.cu.timelineSignal = reinterpret_cast<decltype(g.cu.timelineSignal)>(reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxTimelineSignal")));
    g.cu.timelineDestroy = reinterpret_cast<decltype(g.cu.timelineDestroy)>(reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxTimelineDestroy")));
    g.cu.timelineWait = reinterpret_cast<decltype(g.cu.timelineWait)>(reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rNgxTimelineWait")));
    if (!load_export(g.cuda, "d4rRegisterLinearTexture", g.cu.registerLinearTexture))
        g.cu.registerLinearTexture = nullptr;
    g.cu.dilateMotion = reinterpret_cast<decltype(g.cu.dilateMotion)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rDilateMotionVectors")));
    g.cu.dilateMotionPadded = reinterpret_cast<decltype(g.cu.dilateMotionPadded)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rDilateMotionVectorsPadded")));
    g.cu.dilateMotionThin = reinterpret_cast<decltype(g.cu.dilateMotionThin)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rDilateMotionVectorsThin")));
    g.cu.composeNativeMotion = reinterpret_cast<decltype(g.cu.composeNativeMotion)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rComposeNativeMotion")));
    g.cu.composeNativeMotionThin = reinterpret_cast<decltype(g.cu.composeNativeMotionThin)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rComposeNativeMotionThin")));
    g.cu.composeNativeMotionAligned = reinterpret_cast<decltype(g.cu.composeNativeMotionAligned)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rComposeNativeMotionAligned")));
    g.cu.composeNativeMotionHybrid = reinterpret_cast<decltype(g.cu.composeNativeMotionHybrid)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rComposeNativeMotionHybrid")));
    g.cu.composeNativeMotionPrepared = reinterpret_cast<decltype(g.cu.composeNativeMotionPrepared)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rComposeNativeMotionPrepared")));
    g.cu.stabilizeThinWiresPrepared = reinterpret_cast<decltype(g.cu.stabilizeThinWiresPrepared)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rStabilizeThinWiresPrepared")));
    g.cu.stabilizeThinWires = reinterpret_cast<decltype(g.cu.stabilizeThinWires)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rStabilizeThinWires")));
    g.cu.stabilizeThinWiresSparse = reinterpret_cast<decltype(g.cu.stabilizeThinWiresSparse)>(
        reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rStabilizeThinWiresSparse")));
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

// Loads nvngx_dlssd.dll and looks for the CUDA entry points that mark a build able to serve
// Ray Reconstruction. Nothing here calls into the denoiser: the NGX core loads and
// dispatches it, so this is a presence-and-capability check, not a second lifecycle.
// Without it Super Sampling is unaffected and Ray Reconstruction reports itself unavailable.
//
// The 3.10.7 build of this library carries the whole snippet stack and exports
// NVSDK_NGX_CUDA_*. Other builds - notably the DLAA-only "rel_tot" one, which ships D3D11,
// D3D12, Vulkan and DirectSR and nothing else - do not, and are rejected here rather than
// half-supported: a missing entry point means there is no CUDA path to bridge to.

static bool load_denoiser_library()
{
    if (g.denoiserAttempted)
        return g.denoiser.cudaCapable;
    g.denoiserAttempted = true;
    ensure_portable_config();

    // The core finds nvngx_dlssd.dll beside the DLL that called into it and ignores the
    // feature search paths, so the only place worth looking is here: offering a path that
    // could not redirect the core's own load would be claiming a selection it does not have.
    wchar_t self[MAX_PATH] = {};
    const DWORD selfLength = GetModuleFileNameW(g_selfModule, self, MAX_PATH);
    if (selfLength == 0 || selfLength >= MAX_PATH)
    {
        logf("Ray Reconstruction: cannot locate this DLL, so the denoiser cannot be found");
        return false;
    }
    std::wstring directory(self, selfLength);
    directory.resize(directory.find_last_of(L"\\/"));
    g.denoiserPath = directory + L"\\nvngx_dlssd.dll";
    if (!file_exists(g.denoiserPath))
    {
        logf("Ray Reconstruction: no denoiser at %ls; the NGX core loads it from beside this DLL, so copy "
             "NVIDIA's nvngx_dlssd.dll (3.10.7) there",
             g.denoiserPath.c_str());
        return false;
    }
    g.denoiserModule = load_library_below_hooks(g.denoiserPath.c_str());
    if (g.denoiserModule == nullptr)
    {
        logf("Ray Reconstruction: loading %ls failed: %lu", g.denoiserPath.c_str(), GetLastError());
        return false;
    }
    // Which entry points exist is only ever evidence - none of them is called from here.
    // Create/Evaluate/Release is what "can serve Ray Reconstruction" means. NVSDK_NGX_CUDA_Init
    // is deliberately not required and is never called: in the 310.7 build it is a stub that
    // always reports 0xBAD00001.
    void* create = nullptr, *evaluate = nullptr, *release = nullptr;
    load_export(g.denoiserModule, "NVSDK_NGX_CUDA_CreateFeature", create);
    load_export(g.denoiserModule, "NVSDK_NGX_CUDA_EvaluateFeature", evaluate);
    load_export(g.denoiserModule, "NVSDK_NGX_CUDA_ReleaseFeature", release);
    g.denoiser.cudaCapable = create != nullptr && evaluate != nullptr && release != nullptr;
    wchar_t mapped[MAX_PATH] = {};
    GetModuleFileNameW(g.denoiserModule, mapped, MAX_PATH);
    logf("found the DLSS-Denoiser library %ls (mapped as %ls): %s", g.denoiserPath.c_str(), mapped,
         g.denoiser.cudaCapable ? "CUDA entry points present"
                                : "no CUDA entry points - not a CUDA-capable denoiser build");
    // The file version, from the module's version resource (VS_FIXEDFILEINFO follows its signature 0xFEEF04BD).
    // Preset F (DLSS 4.5, the rr2_* kernels) exists from 310.9 on: an older library runs preset D for that hint.
    if (HRSRC found = FindResourceW(g.denoiserModule, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(16)))
    {
        const DWORD size = SizeofResource(g.denoiserModule, found);
        HGLOBAL loaded = LoadResource(g.denoiserModule, found);
        const unsigned char* bytes = loaded != nullptr ? static_cast<const unsigned char*>(LockResource(loaded)) : nullptr;
        for (DWORD at = 0; bytes != nullptr && at + 16 <= size; at += 4)
        {
            uint32_t words[4];
            std::memcpy(words, bytes + at, sizeof(words));
            if (words[0] != 0xFEEF04BDu)
                continue;
            g.denoiserVersion[0] = words[2] >> 16;
            g.denoiserVersion[1] = words[2] & 0xffffu;
            g.denoiserVersion[2] = words[3] >> 16;
            g.denoiserVersion[3] = words[3] & 0xffffu;
            break;
        }
    }
    if (g.denoiserVersion[0] == 0)
        logf("Ray Reconstruction: the DLSS-Denoiser library has no file version; whether its CUDA path has preset F "
             "(DLSS 4.5, 310.9 or newer) is unknown");
    else
        logf("Ray Reconstruction: DLSS-Denoiser library version %u.%u.%u.%u: its CUDA path %s", g.denoiserVersion[0],
             g.denoiserVersion[1], g.denoiserVersion[2], g.denoiserVersion[3],
             denoiser_has_preset_f() ? "has preset F (DLSS 4.5)"
                                     : "has NO preset F (DLSS 4.5 needs 310.9 or newer) - preset F is shown only by "
                                       "frames the engine serves; frames the CUDA backend serves show preset D");
    return g.denoiser.cudaCapable;
}

// D4R_RR_ENABLE: 1 asks for Ray Reconstruction, 0 forbids it, and unset (auto) offers it
// exactly when the denoiser really loaded. Auto is the default because the only honest answer
// to a game asking whether Ray Reconstruction is available is what this process can actually
// run - never a fixed yes.

bool rr_enabled()
{
    return env_uint("D4R_RR_ENABLE", 1) != 0;
}

static DenoiserCapability g_denoiserCapability;

static std::mutex g_denoiserCapabilityMutex;

static DenoiserCapability denoiser_capability()
{
    std::lock_guard<std::mutex> lock(g_denoiserCapabilityMutex);
    return g_denoiserCapability;
}

// The native Vulkan frontend stages only Super Sampling's inputs, so its Ray Reconstruction is the engine's alone: on
// offer with the engine on and a preset D, E or F package installed (nvngx_dlssd.dll is not used)

static bool native_vk_rr_package()
{
    if (!engine_requested() || env_uint("D4R_VULKAN_ENGINE", 1) == 0) return false;
    for (int slot = 0; slot < 3; ++slot)
        if (GetFileAttributesA((rr_model_directory(slot) + "\\rr.txt").c_str()) != INVALID_FILE_ATTRIBUTES) return true;
    return false;
}

static bool ray_reconstruction_available()
{
    const RrBackend backend = rr_backend();
    if (g.nativeD3d12Only && backend == RrBackend::Dlss) return rr_enabled() && (native_engine_rr(6) || native_engine_rr(4) || native_engine_rr(5));
    if (g.nativeVulkan && backend == RrBackend::Dlss) return rr_enabled() && native_vk_rr_package();
    if (backend == RrBackend::Fsr)
        return rr_enabled() && g.ngxInitialized && fsr_rr_ready();
    const DenoiserCapability capability = denoiser_capability();
    return backend == RrBackend::Dlss && rr_enabled() && g.ngxInitialized && g.denoiser.cudaCapable &&
           capability.known && capability.available != 0;
}

// Why the route is not on offer, for the log and for the capability answer.

static const char* rr_denial_reason()
{
    if (!rr_enabled())
        return "disabled by D4R_RR_ENABLE=0";
    if (g.nativeD3d12Only && rr_backend() == RrBackend::Dlss) return ray_reconstruction_available() ? "available (native D3D12 RR D/E/F)" : "no native RR D/E/F package";
    if (g.nativeVulkan && rr_backend() == RrBackend::Dlss)
        return ray_reconstruction_available() ? "available (native Vulkan engine RR D/E/F)"
                                              : "the native Vulkan frontend needs the engine and an RR D/E/F package (engine\\rr-f)";
    if (rr_backend() == RrBackend::Invalid)
        return "D4R_RR_BACKEND must be dlss or fsr";
    if (!g.ngxInitialized)
        return "the NGX core is not initialised (Super Sampling, which every backend needs)";
    if (rr_backend() == RrBackend::Fsr)
        return fsr_rr_ready() ? "available (FSR Ray Regeneration)" : fsr_rr_denial_reason();
    if (!g.denoiser.cudaCapable)
        return g.denoiserAttempted ? "nvngx_dlssd.dll has no NVSDK_NGX_CUDA_* entry points (needs a 3.10.7 build)"
                                   : "no nvngx_dlssd.dll was found next to this DLL";
    const DenoiserCapability capability = denoiser_capability();
    if (!capability.known)
        return "the NGX core has not reported the denoiser's state yet";
    if (capability.available == 0)
    {
        thread_local char text[160];
        snprintf(text, sizeof(text), "the NGX core reports the denoiser unavailable (init 0x%08x)",
                 static_cast<unsigned int>(capability.initResult));
        return text;
    }
    return "available";
}

// Discovery is legal before NGX Init; use the same pinned worker for both entry points.
// The caller holds g.mutex.

static NgxResult ensure_workers_started()
{
    if (!g.started)
    {
        // Allocators can retain COM callbacks and detached workers past NGX shutdown.
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
    return NGX_SUCCESS;
}

static void set_cuda_adapter_luid(const LUID& luid)
{
    char low[24], high[24];
    std::snprintf(low, sizeof(low), "0x%08lx", static_cast<unsigned long>(luid.LowPart));
    std::snprintf(high, sizeof(high), "0x%08lx", static_cast<unsigned long>(luid.HighPart));
    SetEnvironmentVariableA("D4R_CUDA_LUID_LOW", low);
    SetEnvironmentVariableA("D4R_CUDA_LUID_HIGH", high);
    SetEnvironmentVariableA("D4R_CUDA_NODE_MASK", "1");
    logf("CUDA device LUID set to the D3D12 adapter LUID %s:%s", high, low);
}

static bool native_d3d12_requested()
{
    ensure_portable_config();
    return env_uint("D4R_ENGINE", 0) != 0 &&
        (GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") == nullptr ||
         env_uint("D4R_ENGINE_NATIVE_D3D12", 0) != 0);
}

static NgxResult initialize(unsigned long long applicationId, const wchar_t* dataPath, ID3D12Device* device,
                            const NgxFeatureCommonInfo* featureInfo, unsigned int sdkVersion,
                            const ProjectIdentity* project = nullptr)
{
    std::lock_guard<std::mutex> lock(g.mutex);
    g.shutdownPending = false;
    if (g.ngxInitialized)
        return NGX_SUCCESS;
    if (ensure_workers_started() != NGX_SUCCESS)
        return NGX_FAIL_PLATFORM_ERROR;
    ensure_portable_config();
    // A game on vkd3d-proton under native Windows (its d3d12.dll in the game's folder) takes the Proton route instead:
    // the Vulkan engine through vkd3d-proton's interop, with vkhip, so K's native kernels run in the game's own
    // command list without a second queue. D4R_ENGINE_NATIVE_D3D12=1 keeps the native D3D12 engine there too.
    bool vkd3d = false;
    if (device != nullptr)
    {
        ID3D12DXVKInteropDevice1* probe = nullptr;
        vkd3d = SUCCEEDED(device->QueryInterface(__uuidof(ID3D12DXVKInteropDevice1), reinterpret_cast<void**>(&probe)));
        if (probe != nullptr) probe->Release();
        if (vkd3d && env_uint("D4R_ENGINE_NATIVE_D3D12", 0) != 0) vkd3d = false;
        if (vkd3d) logf("the game's D3D12 device is vkd3d-proton's: the Vulkan engine (vkhip) instead of the native D3D12 engine");
    }
    // Every native D3D12 shader uses WMMA (RDNA 3 and later); creating one on an older AMD GPU removes the game's
    // device, so such a GPU takes the CUDA route. An unidentified GPU keeps the engine; D4R_ENGINE_GPU_CHECK=0 skips this.
    bool nativeGpu = true;
    if (device != nullptr && native_d3d12_requested() && !vkd3d && env_uint("D4R_ENGINE_GPU_CHECK", 1) != 0 &&
        amd_render_generation(device) == AmdGeneration::Older)
    {
        nativeGpu = false;
        g_engineGpuOff = true;
        logf("native D3D12 engine: not on this GPU (its shaders need RDNA 3 or later); the CUDA route instead, as with "
             "[Engine] Enabled = false");
    }
    if (device != nullptr && native_d3d12_requested() && !vkd3d && nativeGpu)
    {
        device->AddRef();
        g.device = device;
        // Provisioned K packages contain everything inference needs. Do not
        // load NGX/CUDA merely to initialize an ordinary Windows D3D12 device.
        const bool hdr = native_engine_available(false), ldr = native_engine_available(true);
        // presets M and L alone suffice too (CreateFeature refuses the presets without a package)
        bool presetML[2][2] = {};
        for (int l = 0; l < 2; ++l)
            for (int d = 0; d < 2; ++d)
                presetML[l][d] = native_engine_m(d != 0, l != 0) != nullptr || native_engine_m_rdna3(d != 0, l != 0) != nullptr;
        if (!hdr && !ldr && !presetML[0][0] && !presetML[0][1] && !presetML[1][0] && !presetML[1][1] && !native_engine_rr(4) && !native_engine_rr(5) && !native_engine_rr(6))
        {
            g.device->Release(); g.device = nullptr;
            return NGX_FAIL_FEATURE_NOT_SUPPORTED;
        }
        g.nativeD3d12Only = g.ngxInitialized = true;
        logf("native D3D12 initialized on game device %p; K HDR=%d LDR=%d, M HDR=%d LDR=%d, L HDR=%d LDR=%d, RR D=%d E=%d F=%d; no CUDA/NGX runtime",
             device, hdr, ldr, presetML[0][0], presetML[0][1], presetML[1][0], presetML[1][1], native_engine_rr(4)!=nullptr, native_engine_rr(5)!=nullptr, native_engine_rr(6)!=nullptr);
        return NGX_SUCCESS;
    }
    if (g_portable.active && !g.core)
    {
        logf("d4r runtime build %s", D4R_RUNTIME_BUILD_ID);
        if (g_portable.nativeWindows)
            logf("portable install in %ls (settings from d4r.ini, native Windows)", g_portable.dir.c_str());
        else
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
        set_cuda_adapter_luid(luid);
        std::string adapterDetail, adapterPci;
        d4r_true_identity(device, adapterDetail, &adapterPci);
        SetEnvironmentVariableA("D4R_RENDER_PCI", adapterPci.empty() ? nullptr : adapterPci.c_str());
        logf("CUDA adapter mapping: %s; PCI match %s", adapterDetail.c_str(),
             adapterPci.empty() ? "unavailable (multiple compute devices will be refused)" : adapterPci.c_str());
        // OptiScaler's FSR 4 option needs the real AMD identity for OptiScaler and AMD's modules (d4r_fsr4_identity.h)
        if (env_string("D4R_FSR4_IDENTITY") != "0")
        {
            std::string detail;
            d4r_fsr4_identity::install(device, detail);
            logf("FSR 4 identity: %s", detail.c_str());
        }
    }

    g.paths.clear();
    logf("engine selection: D4R_ENGINE=%s D4R_ENGINE_K=%s network=%s (environment overrides d4r.ini)",
         env_string("D4R_ENGINE").c_str(), env_string("D4R_ENGINE_K").c_str(), env_string("D4R_ENGINE_NETWORK").c_str());
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
    const bool hasProject = project != nullptr && !project->id.empty();
    bool unavailableAfterInit = false;
    if (result == NGX_SUCCESS && !hasProject && g.ngx.initProjectId != nullptr)
    {
        // Init may succeed while NGX rejects the application identity at feature
        // initialization. Streamline caches this first answer, before a later
        // project-ID Init. Match the Windows runtime's capability check too.
        unavailableAfterInit = g.worker.call([&] {
            void* caps = nullptr;
            if (g.ngx.getCapabilityParameters(&caps) != NGX_SUCCESS || caps == nullptr)
                return false;
            int available = -1, initResult = NGX_SUCCESS, needsDriver = -1;
            const bool known = d4r_ngx_get_int(caps, "SuperSampling.Available", &available) == NGX_SUCCESS &&
                               d4r_ngx_get_int(caps, "SuperSampling.FeatureInitResult", &initResult) == NGX_SUCCESS &&
                               d4r_ngx_get_int(caps, "SuperSampling.NeedsUpdatedDriver", &needsDriver) == NGX_SUCCESS;
            g.ngx.destroyParameters(caps);
            const bool retry = known && needsDriver == 0 && (available == 0 || initResult != NGX_SUCCESS);
            if (retry)
                logf("application-ID capabilities unavailable: available=%d init=0x%08x needsDriver=%d",
                     available, static_cast<unsigned int>(initResult), needsDriver);
            return retry;
        });
    }
    if ((result != NGX_SUCCESS || unavailableAfterInit) && !hasProject && g.ngx.initProjectId != nullptr)
    {
        // NGX 617.14 rejects application-ID initialization on this path (Cyberpunk 2077:
        // 0xBAD0000C, or SR unavailable with 0xBAD00004). Streamline's RR plugin reads
        // capabilities right after that first Init, so a failure there hides RR for the
        // session even though the game's later project-ID Init works. Retry with d4r's
        // own project identity (the diagnostics' default) so the first Init succeeds.
        logf("D4R_NGX_IDENTITY_FALLBACK application=%llu: 0x%08x", id, result);
        // A successful Init owns a session even when its SR feature is unavailable.
        // Release it before retrying, so the fallback gets a fresh identity.
        if (unavailableAfterInit)
            g.worker.call([&] { return g.ngx.shutdown(); });
        result = g.worker.call([&] {
            return g.ngx.initProjectId("24480451-f00d-face-1304-0308dabad187", 0, "1.0", g.dataPath.c_str(),
                                       apiVersion, &g.featureInfo);
        });
        logf("NVSDK_NGX_CUDA_Init_ProjectID(d4r fallback project=24480451-f00d-face-1304-0308dabad187, engine=0 1.0, sdk=0x%x) -> 0x%08x",
             version, result);
    }
    using LoadErrorFn = const char*(WINAPI*)();
    const auto loadError = reinterpret_cast<LoadErrorFn>(reinterpret_cast<void*>(GetProcAddress(g.cuda, "d4rLoadError")));
    if (loadError != nullptr && loadError()[0] != '\0')
        logf("nvcuda bridge: %s", loadError());
    g.ngxInitialized = result == NGX_SUCCESS;

    // The denoiser is not initialised here. The NGX core loads and dispatches
    // nvngx_dlssd.dll itself, so a second init would duplicate that state rather than add
    // anything; this only looks for the library and checks it is a build that can serve
    // Ray Reconstruction. Its own NVSDK_NGX_CUDA_Init is a stub that always fails, so
    // calling it would report "unavailable" for a denoiser that is in fact working.
    if (rr_enabled() && rr_backend() == RrBackend::Fsr)
    {
        // nvngx_dlssd.dll plays no part: the denoiser is AMD's, then Super Sampling upscales.
        if (result == NGX_SUCCESS)
            fsr_rr_prepare_runtime();
        logf("Ray Reconstruction: requested backend fsr: %s", rr_denial_reason());
    }
    else if (rr_enabled())
    {
        load_denoiser_library();
        logf("Ray Reconstruction: requested backend %s: %s",
             rr_backend() == RrBackend::Dlss ? "dlss" : "invalid (D4R_RR_BACKEND)", rr_denial_reason());
    }
    else
        logf("Ray Reconstruction: disabled by D4R_RR_ENABLE=0");
    if (result == NGX_SUCCESS && device != nullptr)
        start_engine_preload();
    return result;
}

// --- exported NGX D3D12 API ------------------------------------------------------

#define D4R_EXPORT extern "C" __declspec(dllexport)

// Windows OptiScaler's external-backend handshake. Inference uses the regular
// NGX command-list API; early installation observes submissions from lists the
// game creates before its NGX Init call.

D4R_EXPORT unsigned d4r_WindowsBackendVersion() { return 1; }

D4R_EXPORT unsigned d4r_D3D12_InstallCommandBackend(ID3D12Device* device)
{
    return device && d4r_submit::install(device) ? NGX_SUCCESS : NGX_FAIL_PLATFORM_ERROR;
}

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
// Reads the denoising family out of the core's own capability parameters. Worker only.

void refresh_denoiser_capability()
{
    void* official = nullptr;
    if (g.ngx.getCapabilityParameters == nullptr || g.ngx.getCapabilityParameters(&official) != NGX_SUCCESS ||
        official == nullptr)
        return;
    DenoiserCapability capability;
    int available = 0;
    capability.available =
        d4r_ngx_get_int(official, "SuperSamplingDenoising.Available", &available) == NGX_SUCCESS ? available : 0;
    int initResult = 0;
    capability.initResult =
        d4r_ngx_get_int(official, "SuperSamplingDenoising.FeatureInitResult", &initResult) == NGX_SUCCESS
            ? initResult
            : static_cast<int>(NGX_FAIL_NOT_INITIALIZED);
    capability.known = true;
    g.ngx.destroyParameters(official);
    std::lock_guard<std::mutex> lock(g_denoiserCapabilityMutex);
    g_denoiserCapability = capability;
}

// NVSDK's NGX_DLSS_GET_OPTIMAL_SETTINGS passes display Width/Height and reads
// render OutWidth/OutHeight from this single-parameter callback.

static NgxResult native_optimal_settings(void* parameters)
{
    unsigned int width = 0, height = 0; int quality = 2;
    if (d4r_ngx_get_uint(parameters, "Width", &width) != NGX_SUCCESS ||
        d4r_ngx_get_uint(parameters, "Height", &height) != NGX_SUCCESS || !width || !height)
        return NGX_FAIL_INVALID_PARAMETER;
    d4r_ngx_get_int(parameters, "PerfQualityValue", &quality);
    const double ratios[] = {0.5, 0.58, 2.0 / 3.0, 1.0 / 3.0, 0.77, 1.0};
    if (quality < 0 || quality > 5) return NGX_FAIL_INVALID_PARAMETER;
    const unsigned int rw = std::max(1u, unsigned(std::lround(width * ratios[quality])));
    const unsigned int rh = std::max(1u, unsigned(std::lround(height * ratios[quality])));
    d4r_ngx_set_uint(parameters, "OutWidth", rw); d4r_ngx_set_uint(parameters, "OutHeight", rh);
    d4r_ngx_set_uint(parameters, "DLSS.Get.Dynamic.Max.Render.Width", width);
    d4r_ngx_set_uint(parameters, "DLSS.Get.Dynamic.Max.Render.Height", height);
    d4r_ngx_set_uint(parameters, "DLSS.Get.Dynamic.Min.Render.Width", rw);
    d4r_ngx_set_uint(parameters, "DLSS.Get.Dynamic.Min.Render.Height", rh);
    d4r_ngx_set_float(parameters, "Sharpness", 0.0f);
    return NGX_SUCCESS;
}

static NgxResult fill_capabilities(void* parameters)
{
    if (g.nativeD3d12Only)
    {
        d4r_ngx_set_int(parameters, "SuperSampling.Available", 1);
        d4r_ngx_set_int(parameters, "SuperSampling.NeedsUpdatedDriver", 0);
        d4r_ngx_set_int(parameters, "SuperSampling.FeatureInitResult", NGX_SUCCESS);
        const bool rr = ray_reconstruction_available();
        d4r_ngx_set_int(parameters, "SuperSamplingDenoising.Available", rr ? 1 : 0);
        d4r_ngx_set_int(parameters, "SuperSamplingDenoising.FeatureInitResult", rr ? NGX_SUCCESS : NGX_FAIL_FEATURE_NOT_SUPPORTED);
        d4r_ngx_set_int(parameters, "SuperSamplingDenoising.NeedsUpdatedDriver", 0);
        d4r_ngx_set_void(parameters, "DLSSDOptimalSettingsCallback",
                         rr ? reinterpret_cast<void*>(&native_optimal_settings) : nullptr);
        d4r_ngx_set_void(parameters, "DLSSOptimalSettingsCallback", reinterpret_cast<void*>(&native_optimal_settings));
        return NGX_SUCCESS;
    }
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
        // DLSSDOptimalSettingsCallback is not optional decoration: the SDK's helpers call it to
        // work out the render dimensions, so a Ray Reconstruction route that dropped it would
        // misbehave in a way that looks like a game bug. Forwarded exactly as the Super
        // Sampling pair already is - no new ABI is invented here.
        const char* pointers[] = {"DLSSOptimalSettingsCallback", "DLSSGetStatsCallback",
                                  "DLSSDOptimalSettingsCallback", "DLSSDGetStatsCallback"};
        for (const char* name : pointers)
        {
            void* value = nullptr;
            if (d4r_ngx_get_void(official, name, &value) == NGX_SUCCESS && value != nullptr)
                d4r_ngx_set_void(parameters, name, value);
        }
        // The denoising family is the core's own account of the denoiser it loaded, so it is
        // passed through rather than re-derived here - and cached, because it is also what
        // decides whether Ray Reconstruction may be created at all.
        refresh_denoiser_capability();
        const char* denoisingInts[] = {"SuperSamplingDenoising.Available",
                                       "SuperSamplingDenoising.NeedsUpdatedDriver",
                                       "SuperSamplingDenoising.FeatureInitResult"};
        for (const char* name : denoisingInts)
        {
            int value = 0;
            if (d4r_ngx_get_int(official, name, &value) == NGX_SUCCESS)
                d4r_ngx_set_int(parameters, name, value);
        }
        const char* denoisingUnsigned[] = {"SuperSamplingDenoising.MinDriverVersionMajor",
                                           "SuperSamplingDenoising.MinDriverVersionMinor"};
        for (const char* name : denoisingUnsigned)
        {
            unsigned int value = 0;
            if (d4r_ngx_get_uint(official, name, &value) == NGX_SUCCESS)
                d4r_ngx_set_uint(parameters, name, value);
        }
        // Backend = fsr: the core's account of nvngx_dlssd.dll does not apply (that library is
        // not used, and need not exist). Availability is this process's FSR Ray Regeneration
        // provider plus Super Sampling. Without nvngx_dlssd.dll the core offers no DLSSD helper
        // callbacks; feature 13's optimal render sizes are Super Sampling's (same parameter
        // names, same quality modes), so its callbacks answer them.
        if (rr_enabled() && rr_backend() == RrBackend::Fsr)
        {
            const bool ready = ray_reconstruction_available();
            d4r_ngx_set_int(parameters, "SuperSamplingDenoising.Available", ready ? 1 : 0);
            d4r_ngx_set_int(parameters, "SuperSamplingDenoising.NeedsUpdatedDriver", 0);
            d4r_ngx_set_int(parameters, "SuperSamplingDenoising.FeatureInitResult",
                            static_cast<int>(ready ? NGX_SUCCESS : NGX_FAIL_FEATURE_NOT_SUPPORTED));
            const char* pairs[2][2] = {{"DLSSDOptimalSettingsCallback", "DLSSOptimalSettingsCallback"},
                                       {"DLSSDGetStatsCallback", "DLSSGetStatsCallback"}};
            for (const auto& pair : pairs)
            {
                void* superSampling = nullptr;
                d4r_ngx_get_void(official, pair[1], &superSampling);
                d4r_ngx_set_void(parameters, pair[0], ready ? superSampling : nullptr);
            }
        }
        // The native Vulkan frontend: the engine's packages decide, as for native D3D12; feature 13's optimal render
        // sizes are Super Sampling's
        if (g.nativeVulkan && rr_enabled() && rr_backend() == RrBackend::Dlss)
        {
            const bool ready = ray_reconstruction_available();
            d4r_ngx_set_int(parameters, "SuperSamplingDenoising.Available", ready ? 1 : 0);
            d4r_ngx_set_int(parameters, "SuperSamplingDenoising.NeedsUpdatedDriver", 0);
            d4r_ngx_set_int(parameters, "SuperSamplingDenoising.FeatureInitResult",
                            static_cast<int>(ready ? NGX_SUCCESS : NGX_FAIL_FEATURE_NOT_SUPPORTED));
            void* stats = nullptr;
            d4r_ngx_get_void(official, "DLSSGetStatsCallback", &stats);
            d4r_ngx_set_void(parameters, "DLSSDOptimalSettingsCallback",
                             ready ? reinterpret_cast<void*>(&native_optimal_settings) : nullptr);
            d4r_ngx_set_void(parameters, "DLSSDGetStatsCallback", ready ? stats : nullptr);
        }
        // D4R_RR_ENABLE=0 is this file's decision and outranks what the core reports: with the
        // route switched off, availability reads 0 and the callbacks are cleared, so a game
        // cannot reach a denoiser route that is not being offered.
        if (!rr_enabled() || rr_backend() == RrBackend::Invalid)
        {
            d4r_ngx_set_int(parameters, "SuperSamplingDenoising.Available", 0);
            d4r_ngx_set_int(parameters, "SuperSamplingDenoising.FeatureInitResult",
                            static_cast<int>(NGX_FAIL_FEATURE_NOT_SUPPORTED));
            d4r_ngx_set_void(parameters, "DLSSDOptimalSettingsCallback", nullptr);
            d4r_ngx_set_void(parameters, "DLSSDGetStatsCallback", nullptr);
        }
        const DenoiserCapability capability = denoiser_capability();
        if (rr_backend() == RrBackend::Fsr)
            logf("Ray Reconstruction capability (backend fsr, nvngx_dlssd.dll not used): %s", rr_denial_reason());
        else if (g.nativeVulkan)
            logf("Ray Reconstruction capability (native Vulkan, nvngx_dlssd.dll not used): %s", rr_denial_reason());
        else
            logf("denoiser capabilities from the core: available=%d init=0x%08x (Ray Reconstruction %s)",
                 capability.available, static_cast<unsigned int>(capability.initResult),
                 rr_denial_reason());
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
    int available = -1, initResult = -1, needsDriver = -1, denoising = -1, denoisingInit = -1;
    d4r_ngx_get_int(*parameters, "SuperSampling.Available", &available);
    d4r_ngx_get_int(*parameters, "SuperSampling.FeatureInitResult", &initResult);
    d4r_ngx_get_int(*parameters, "SuperSampling.NeedsUpdatedDriver", &needsDriver);
    d4r_ngx_get_int(*parameters, "SuperSamplingDenoising.Available", &denoising);
    d4r_ngx_get_int(*parameters, "SuperSamplingDenoising.FeatureInitResult", &denoisingInit);
    logf("NVSDK_NGX_D3D12_GetCapabilityParameters -> 0x%08x, SuperSampling.Available=%d FeatureInitResult=0x%08x "
         "NeedsUpdatedDriver=%d, SuperSamplingDenoising.Available=%d FeatureInitResult=0x%08x",
         result, available, static_cast<unsigned int>(initResult), needsDriver, denoising,
         static_cast<unsigned int>(denoisingInit));
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

// nvsdk_ngx.h: NVSDK_NGX_D3D12_GetFeatureRequirements(IDXGIAdapter *Adapter,
//   const NVSDK_NGX_FeatureDiscoveryInfo *FeatureDiscoveryInfo,
//   NVSDK_NGX_FeatureRequirement *OutSupported). The feature is NOT a separate argument: it is
// FeatureDiscoveryInfo->FeatureID (defs.h, the field after SDKVersion), and the first argument
// is the display adapter, which must never be read as a feature id.
//
// This answer is derived from d4r's own state alone. The earlier version forwarded the query
// to the NGX core's CUDA entry point on the shim's worker thread and, for feature 13, called
// the denoiser's discovery export before Init. Those calls turn a requirements query that
// OptiScaler hooks into a cross-binary handoff: inside a Steam runtime that reorders module
// loads, OptiScaler's follow-up scan of its own support lists then read memory the other
// binary had already released (deterministic EXCEPTION_ACCESS_VIOLATION inside OptiScaler's
// dxgi.dll; see the repo's Cyberpunk notes). The requirements answer carries no state that is
// not already authoritative elsewhere: Ray Reconstruction's real gate is the CreateFeature
// check plus the core's denoiser account reported through GetCapabilityParameters, both of
// which run after their CUDA context exists.

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_GetFeatureRequirements(void* adapter, const void* discovery,
                                                          NgxFeatureRequirement* requirement)
{
    if (requirement == nullptr)
        return NGX_SUCCESS;
    // NVSDK_NGX_FeatureDiscoveryInfo: { NVSDK_NGX_Version SDKVersion; NVSDK_NGX_Feature FeatureID; ... },
    // both enums, so FeatureID is the unsigned int at offset 4.
    unsigned int feature = 0;
    if (discovery != nullptr)
        std::memcpy(&feature, static_cast<const char*>(discovery) + sizeof(unsigned int), sizeof(feature));
    std::memset(requirement, 0, sizeof(*requirement));
    if (discovery != nullptr && native_d3d12_requested())
    {
        requirement->FeatureSupported = (feature == NGX_FEATURE_SUPER_SAMPLING || (feature == NGX_FEATURE_RAY_RECONSTRUCTION && rr_enabled() && rr_backend() == RrBackend::Dlss)) ? 0 : 1;
        return NGX_SUCCESS;
    }
    if (feature != NGX_FEATURE_SUPER_SAMPLING && feature != NGX_FEATURE_RAY_RECONSTRUCTION)
    {
        requirement->FeatureSupported = 1; // not supported
        return NGX_SUCCESS;
    }
    // The local switch is checked before any core appears: with D4R_RR_ENABLE=0 this process
    // is not offering Ray Reconstruction, and no core answer may override that.
    if (feature == NGX_FEATURE_RAY_RECONSTRUCTION && (!rr_enabled() || rr_backend() == RrBackend::Invalid))
    {
        requirement->FeatureSupported = 1; // not supported
        logf("Ray Reconstruction is not supported: %s", rr_denial_reason());
        return NGX_SUCCESS;
    }
    // CPU-side adapter identity only; safe before Init and without any CUDA context.
    if (adapter != nullptr && !g.ngxInitialized)
    {
        DXGI_ADAPTER_DESC desc = {};
        if (SUCCEEDED(static_cast<IDXGIAdapter*>(adapter)->GetDesc(&desc)))
            set_cuda_adapter_luid(desc.AdapterLuid);
    }
    {
        std::lock_guard<std::mutex> lock(g.mutex);
        // Load the provider without initializing NGX: creation keeps the game's identity, and
        // feature 13's acceptance is decided later from ray_reconstruction_available().
        if (ensure_workers_started() != NGX_SUCCESS || (g.core == nullptr && !load_libraries()))
            return NGX_FAIL_PLATFORM_ERROR;
        if (feature == NGX_FEATURE_RAY_RECONSTRUCTION && rr_backend() == RrBackend::Dlss)
            load_denoiser_library();
    }
    // Before Init there is no device to ask AMD's runtime about: the FSR backend is offered when its DLL is in
    // place, and CreateFeature makes the final decision on the real device.
    const bool supported = feature == NGX_FEATURE_SUPER_SAMPLING ? true
                           : g.ngxInitialized                    ? ray_reconstruction_available()
                           : rr_backend() == RrBackend::Fsr      ? fsr_rr_present()
                                                                 : g.denoiser.cudaCapable;
    if (!supported)
    {
        requirement->FeatureSupported = 1; // not supported
        if (feature == NGX_FEATURE_RAY_RECONSTRUCTION)
            logf("Ray Reconstruction is not supported: %s",
                 g.ngxInitialized                 ? rr_denial_reason()
                 : rr_backend() == RrBackend::Fsr ? "backend fsr: amd_fidelityfx_denoiser_dx12.dll is missing"
                                                  : "no CUDA-capable DLSS-Denoiser for pre-Init discovery");
        return NGX_SUCCESS;
    }
    // supported (NVSDK_NGX_FeatureSupportResult bitfield). No minimum architecture, as on the native route:
    // Streamline drops sl.dlss_d (adapter mask 0) when the GPU it sees is below a nonzero MinHWArchitecture.
    requirement->FeatureSupported = 0;
    return NGX_SUCCESS;
}

unsigned int get_uint_or(void* parameters, const char* name, unsigned int fallback)
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

// Called on the game thread with the feature API lock held. All CUDA resources are made on its worker.
// Vulkan-only K features retain just their creation settings until a frame actually needs CUDA.

static NgxResult create_cuda_feature(Feature& feature)
{
    const auto createStarted = std::chrono::steady_clock::now();
    logf("CUDA_CREATE_BEGIN preset=%u render=%ux%u output=%ux%u flags=0x%x", feature.preset,
         feature.width, feature.height, feature.outWidth, feature.outHeight, feature.flags);
    static const char* const kSuperSamplingPresets[] = {
        "DLSS.Hint.Render.Preset.DLAA", "DLSS.Hint.Render.Preset.Quality", "DLSS.Hint.Render.Preset.Balanced",
        "DLSS.Hint.Render.Preset.Performance", "DLSS.Hint.Render.Preset.UltraPerformance",
        "DLSS.Hint.Render.Preset.UltraQuality"};
    static const char* const kRayReconstructionPresets[] = {
        "RayReconstruction.Hint.Render.Preset.DLAA", "RayReconstruction.Hint.Render.Preset.Quality",
        "RayReconstruction.Hint.Render.Preset.Balanced", "RayReconstruction.Hint.Render.Preset.Performance",
        "RayReconstruction.Hint.Render.Preset.UltraPerformance", "RayReconstruction.Hint.Render.Preset.UltraQuality"};
    const bool rayReconstruction = feature.rayReconstruction;
    const char* const* presetNames = rayReconstruction ? kRayReconstructionPresets : kSuperSamplingPresets;
    const NgxResult result = g.worker.call([&]() -> NgxResult {
        logf("CUDA_CREATE_WORKER_BEGIN preset=%u", feature.preset);
        NgxResult allocation = g.ngx.allocateParameters(&feature.cudaParams);
        if (allocation != NGX_SUCCESS)
            return allocation;
        void* p = feature.cudaParams;
        if (rayReconstruction)
        {
            // The CUDA denoiser reads creation dimensions through Get(int*),
            // unlike Super Sampling's Get(unsigned int*) contract.
            d4r_ngx_set_int(p, "Width", static_cast<int>(feature.width));
            d4r_ngx_set_int(p, "Height", static_cast<int>(feature.height));
            d4r_ngx_set_int(p, "OutWidth", static_cast<int>(feature.outWidth));
            d4r_ngx_set_int(p, "OutHeight", static_cast<int>(feature.outHeight));
        }
        else
        {
            d4r_ngx_set_uint(p, "Width", feature.width);
            d4r_ngx_set_uint(p, "Height", feature.height);
            d4r_ngx_set_uint(p, "OutWidth", feature.outWidth);
            d4r_ngx_set_uint(p, "OutHeight", feature.outHeight);
        }
        d4r_ngx_set_int(p, "PerfQualityValue", feature.quality);
        d4r_ngx_set_int(p, "DLSS.Feature.Create.Flags", feature.flags);
        d4r_ngx_set_int(p, "DLSS.Enable.Output.Subrects", feature.subrects);
        d4r_ngx_set_uint(p, "CreationNodeMask", 1);
        d4r_ngx_set_uint(p, "VisibilityNodeMask", 1);
        for (int index = 0; index < 6; ++index)
            d4r_ngx_set_uint(p, presetNames[index], feature.creationPresets[index]);
        if (rayReconstruction)
            for (size_t index = 0; index < sizeof(kAuxScalars) / sizeof(kAuxScalars[0]); ++index)
                d4r_ngx_set_int(p, kAuxScalars[index].name, feature.auxScalarBaseline[index]);
        if (g.cu.memAlloc(&feature.scratch, 64ull * 1024 * 1024) == 0)
        {
            d4r_ngx_set_void(p, "Scratch", reinterpret_cast<void*>(static_cast<uintptr_t>(feature.scratch)));
            d4r_ngx_set_ull(p, "Scratch.SizeInBytes", 64ull * 1024 * 1024);
        }
        else
            feature.scratch = 0;
        // The core owns the denoiser: it loads nvngx_dlssd.dll, knows its kernel map, and
        // dispatches feature 13 to it. Only the feature id differs from Super Sampling.
        logf("CUDA_CREATE_NGX_BEGIN preset=%u scratch=0x%llx", feature.preset, feature.scratch);
        const NgxResult created = g.ngx.createFeature(
            rayReconstruction ? NGX_FEATURE_RAY_RECONSTRUCTION : NGX_FEATURE_SUPER_SAMPLING, p, &feature.cudaHandle);
        logf("CUDA_CREATE_NGX_END preset=%u result=0x%08x", feature.preset, created);
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
        feature.cudaFlags = feature.flags;
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
        // D4R_PROFILE_GPU_EVENTS=0 keeps D4R_PROFILE's CPU timings without the CUDA timing events
        if (created == NGX_SUCCESS && profile_enabled() && env_uint("D4R_PROFILE_GPU_EVENTS", 1) != 0 &&
            g.cu.eventCreate != nullptr &&
            g.cu.eventRecord != nullptr && g.cu.eventElapsedTime != nullptr && g.cu.eventDestroy != nullptr)
        {
            const int startResult = g.cu.eventCreate(&feature.profileStart, 0);
            const int endResult = startResult == 0 ? g.cu.eventCreate(&feature.profileEnd, 0) : -1;
            feature.profileEventsReady = startResult == 0 && endResult == 0;
            if(feature.profileEventsReady && (!g_portable.active || g_portable.nativeMotionVectors) &&
                env_uint("D4R_NATIVE_MV_LIVE",0)==1)
            {
                bool ready=true;
                for(CudaEvent& event:feature.nativePassEvents)
                    if(g.cu.eventCreate(&event,0)!=0) { ready=false;break; }
                if(!ready) for(CudaEvent& event:feature.nativePassEvents)
                { if(event) g.cu.eventDestroy(event);event=nullptr; }
            }
            // optional: without it GPU-input-wait frames report dlss_gpu=-1 and keep gpu_eval
            if (feature.profileEventsReady && g.cu.eventCreate(&feature.profileInputsReleased, 0) != 0)
                feature.profileInputsReleased = nullptr;
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
    }, "CUDA feature creation");
    logf("NVSDK_NGX_CUDA_CreateFeature(feature=%u) -> 0x%08x",
         rayReconstruction ? NGX_FEATURE_RAY_RECONSTRUCTION : NGX_FEATURE_SUPER_SAMPLING, result);
    logf("CUDA_CREATE_END preset=%u result=0x%08x elapsed_ms=%.1f", feature.preset, result,
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - createStarted).count());
    if (result == NGX_SUCCESS) feature.cudaCreationDeferred = false;
    return result;
}

// Whether the presets of this creation were picked in OptiScaler (d4r_render_presets.h): its override is on in
// OptiScaler.ini, or the presets changed since the game's first creation with the same creation flags. (OptiScaler
// forwards the game's AllocateParameters to this DLL, so the table itself does not say who filled it.) D4R_FOLLOW_OPTISCALER=0 always
// keeps d4r's model. g.mutex is held.

static bool optiscaler_chose_presets(bool rayReconstruction, const unsigned int (&presets)[6], int flags)
{
    static D4rPresetFollower followers[2];
    static bool iniOverride[2] = {};
    static std::once_flag iniOnce;
    if (env_string("D4R_FOLLOW_OPTISCALER") == "0")
        return false;
    std::call_once(iniOnce, [] {
        // OptiScaler reads OptiScaler.ini beside its own DLL, which d4r's install puts beside the game's EXE
        wchar_t exe[MAX_PATH] = {};
        const DWORD length = GetModuleFileNameW(nullptr, exe, MAX_PATH);
        if (length == 0 || length >= MAX_PATH)
            return;
        std::wstring path(exe, length);
        path.resize(path.find_last_of(L"\\/") + 1);
        std::string text;
        if (!read_whole_file(path + L"OptiScaler.ini", text))
            return;
        const std::vector<IniEntry> ini = parse_ini(text);
        iniOverride[0] = ini_flag(ini, "dlss", "RenderPresetOverride", 0) == 1;
        iniOverride[1] = ini_flag(ini, "dlssd", "RenderPresetOverride", 0) == 1;
        logf("OptiScaler.ini: RenderPresetOverride DLSS=%d DLSSD=%d", iniOverride[0], iniOverride[1]);
    });
    return followers[rayReconstruction].follow(presets, iniOverride[rayReconstruction], flags);
}

static NgxResult create_feature_locked(unsigned int featureId, void* parameters, NgxHandle** handle,
                                       bool nativeVulkan);

static NgxResult create_feature(unsigned int featureId, void* parameters, NgxHandle** handle, bool nativeVulkan)
{
    std::lock_guard<std::mutex> apiLock(g_featureApiMutex);
    std::lock_guard<std::mutex> stateLock(g.mutex);
    return create_feature_locked(featureId, parameters, handle, nativeVulkan);
}

// The API lock and g.mutex are held. The FSR Ray Regeneration backend creates its internal Super Sampling
// feature through here as well, without taking either lock again.

static NgxResult create_feature_locked(unsigned int featureId, void* parameters, NgxHandle** handle,
                                       bool nativeVulkan)
{
    if (!g.ngxInitialized)
        return NGX_FAIL_NOT_INITIALIZED;
    if (!nativeVulkan && g.device == nullptr)
        return NGX_FAIL_NOT_INITIALIZED;
    if (!nativeVulkan)
        std::call_once(g_vk.once, init_vram_interop);
    // A device initialized for the native D3D12 engine (native Windows with [Engine] on) runs Super Sampling on it
    // alone. D4R_ENGINE_NATIVE_D3D12=1 takes that path on vkd3d-proton too (comparing the D3D12 kernels on Linux).
    static const bool forceNativeD3d12 = env_uint("D4R_ENGINE_NATIVE_D3D12", 0) != 0;
    const bool nativeD3d12 = !nativeVulkan && (g.nativeD3d12Only || (forceNativeD3d12 && !g_engineGpuOff.load()));
    // Without vkd3d-proton (native Windows) the CUDA backend stages through host memory and the submission hooks
    // keep each frame's resources alive (retain_native).
    if (!nativeVulkan && !nativeD3d12 && g_vk.interop == nullptr)
    {
        const char* failure = "unknown";
        if (!d4r_submit::install(g.device, &failure))
        {
            logf("DLSS on the CUDA backend without vkd3d-proton needs the submission hooks (%s)", failure);
            return NGX_FAIL_PLATFORM_ERROR;
        }
    }
    else if (!nativeVulkan && !nativeD3d12 && g_vk.lifetime == nullptr)
    {
        logf("DLSS requires lifetime-aware d4r d3d12.dll/d3d12core.dll; update both files with the shim");
        return NGX_FAIL_PLATFORM_ERROR;
    }
    if (nativeD3d12 && ((featureId != NGX_FEATURE_SUPER_SAMPLING && featureId != NGX_FEATURE_RAY_RECONSTRUCTION) || !engine_requested()))
    {
        logf("NVSDK_NGX_D3D12_CreateFeature(feature=%u): without vkd3d-proton only Super Sampling on d4r's D3D12 "
             "engine is available%s", featureId, engine_requested() ? "" : " ([Engine] Enabled is off)");
        return NGX_FAIL_FEATURE_NOT_SUPPORTED;
    }
    // Two features are served here. Super Sampling is feature 1 and Ray Reconstruction
    // feature 13; the NGX core dispatches the second to the denoiser it loaded, and is
    // offered feature 13 only once it has reported that denoiser working.
    const bool rayReconstruction = featureId == NGX_FEATURE_RAY_RECONSTRUCTION;
    // The native Vulkan frontend stages only Super Sampling's four inputs, not the denoiser's guides: its Ray
    // Reconstruction runs on the engine alone (ray_reconstruction_available)
    if ((featureId != NGX_FEATURE_SUPER_SAMPLING && !rayReconstruction) ||
        (rayReconstruction && nativeVulkan && !g.nativeVulkan))
    {
        logf("NVSDK_NGX_D3D12_CreateFeature(feature=%u) unsupported", featureId);
        return NGX_FAIL_FEATURE_NOT_SUPPORTED;
    }
    if (rayReconstruction && rr_backend() == RrBackend::Dlss)
    {
        // A game may create feature 13 without ever asking for capabilities, so the core's
        // account of the denoiser is fetched here if it has not been already.
        g.worker.call([] { refresh_denoiser_capability(); return 0; });
    }
    if (rayReconstruction && !ray_reconstruction_available())
    {
        logf("NVSDK_NGX_D3D12_CreateFeature(feature=%u) unavailable: %s", featureId, rr_denial_reason());
        return NGX_FAIL_FEATURE_NOT_SUPPORTED;
    }
    if (parameters == nullptr || handle == nullptr)
    {
        logf("NVSDK_NGX_D3D12_CreateFeature(feature=%u): no parameters or handle", featureId);
        return NGX_FAIL_INVALID_PARAMETER;
    }
    auto* feature = new Feature();
    feature->nativeVulkan = nativeVulkan;
    feature->nativeD3d12 = nativeD3d12;
    feature->width = get_uint_or(parameters, "Width", 0);
    feature->height = get_uint_or(parameters, "Height", 0);
    feature->outWidth = get_uint_or(parameters, "OutWidth", 0);
    feature->outHeight = get_uint_or(parameters, "OutHeight", 0);
    const int quality = get_int_or(parameters, "PerfQualityValue", 0);
    const int flags = get_int_or(parameters, "DLSS.Feature.Create.Flags", 0);
    feature->quality = quality;
    feature->flags = flags;
    const int subrects = get_int_or(parameters, "DLSS.Enable.Output.Subrects", 0);
    feature->outputSubrects = subrects;
    // The render preset is chosen by two independent knobs. D4R_DLSS_PRESET forces the
    // Super Sampling model (NVSDK_NGX_DLSS_Hint_Render_Preset, e.g. 5 = E) and touches only
    // Super Sampling features; D4R_RR_PRESET forces the Ray Reconstruction model
    // (NVSDK_NGX_RayReconstruction_Hint_Render_Preset, D = 4, E = 5, F = 6) and touches only
    // Ray Reconstruction features. Neither one silently becomes the other.
    static const char* const kSuperSamplingPresets[] = {
        "DLSS.Hint.Render.Preset.DLAA", "DLSS.Hint.Render.Preset.Quality", "DLSS.Hint.Render.Preset.Balanced",
        "DLSS.Hint.Render.Preset.Performance", "DLSS.Hint.Render.Preset.UltraPerformance",
        "DLSS.Hint.Render.Preset.UltraQuality"};
    static const char* const kRayReconstructionPresets[] = {
        "RayReconstruction.Hint.Render.Preset.DLAA", "RayReconstruction.Hint.Render.Preset.Quality",
        "RayReconstruction.Hint.Render.Preset.Balanced", "RayReconstruction.Hint.Render.Preset.Performance",
        "RayReconstruction.Hint.Render.Preset.UltraPerformance", "RayReconstruction.Hint.Render.Preset.UltraQuality"};
    const char* const* presetNames = rayReconstruction ? kRayReconstructionPresets : kSuperSamplingPresets;
    unsigned int presets[6];
    for (int index = 0; index < 6; ++index)
        presets[index] = get_uint_or(parameters, presetNames[index], 0);
    const std::string forced = env_string(d4r_render_preset_variable(rayReconstruction));
    const auto presetOverride = d4r_parse_render_preset_override(forced.c_str());
    // Flagged, retired or unknown values count as Default, both for spotting OptiScaler's change and in what is followed
    if (presetOverride.enabled)
        d4r_sanitize_presets(presets, rayReconstruction);
    const bool followed = presetOverride.enabled && optiscaler_chose_presets(rayReconstruction, presets, flags);
    if (followed)
    {
        logf("DLSS model: OptiScaler's preset override [DLAA=%u Quality=%u Balanced=%u Performance=%u "
             "UltraPerformance=%u UltraQuality=%u] wins over %s=%s (which still fills Default slots)",
             presets[0], presets[1], presets[2], presets[3], presets[4], presets[5],
             d4r_render_preset_variable(rayReconstruction), forced.c_str());
        d4r_apply_followed_presets(presets, presetOverride);
    }
    else if (presetOverride.enabled)
    {
        const unsigned int value = presetOverride.value;
        const char* model = value == 5 ? "E" : value == 11 ? "K" : value == 12 ? "L" : value == 13 ? "M" : "custom";
        logf("DLSS model override: D4R_DLSS_PRESET=%s (model %s) overrides game presets "
             "[DLAA=%u Quality=%u Balanced=%u Performance=%u UltraPerformance=%u UltraQuality=%u]; "
             "using preset=%u for all quality modes",
             forced.c_str(), model, presets[0], presets[1], presets[2], presets[3], presets[4], presets[5], value);
        presetOverride.apply(presets);
    }
    // NGX runs the preset of the feature's quality mode.
    feature->preset = presets[d4r_preset_slot(quality)];
    // the preset the engine's model slot runs (rr_engine_slot: F for every hint but D and E when F is installed)
    if (rayReconstruction && nativeVulkan)
    {
        static const uint32_t slotPresets[6] = {4, 5, 6, 1, 2, 3}; // g_rrPreset
        feature->preset = slotPresets[rr_engine_slot(feature->preset)];
    }
    if (nativeD3d12)
    {
        // DLSSD 310.9 uses F for every hint except explicit D/E when F is installed.
        if (rayReconstruction && feature->preset != 4 && feature->preset != 5 && native_engine_rr(6))
            feature->preset = 6;
        if (feature->preset == 0) feature->preset = rayReconstruction ? 4 : 11;
        const bool ldrFeature = (feature->flags & 1u) == 0;
        bool presetM = (feature->preset == 13 || feature->preset == 12) && !rayReconstruction &&
                       (native_engine_m(ldrFeature, feature->preset == 12) != nullptr ||
                        native_engine_m_rdna3(ldrFeature, feature->preset == 12) != nullptr);
        bool presetRR = rayReconstruction && native_engine_rr(feature->preset) != nullptr;
        // Ray Reconstruction likewise: a hint without its package (D or E asked for with only F installed, F without
        // its package, a flagged or retired value) runs the newest installed one instead of being refused
        if (rayReconstruction && !presetRR)
            for (const unsigned candidate : {6u, 5u, 4u})
                if (native_engine_rr(candidate) != nullptr)
                {
                    logf("native D3D12: Ray Reconstruction preset %u has no native package; the feature runs %c",
                         feature->preset, char('A' + candidate - 1));
                    feature->preset = candidate;
                    presetRR = true;
                    break;
                }
        // The native engine has Super Resolution's K, L and M only: another preset (the CNN presets A-F and J, which
        // an OptiScaler override can ask for) or an M/L without its package runs as K rather than leaving the game
        // without DLSS (there is no CUDA route to fall back to here)
        if (!rayReconstruction && !presetM && feature->preset != 11 && native_engine_available(ldrFeature))
        {
            logf("native D3D12: preset %u has no native package (the engine runs K, L and M); the feature runs K",
                 feature->preset);
            feature->preset = 11;
        }
        // ... and without K's package, the M or L that is installed
        if (!rayReconstruction && !presetM && !native_engine_available(ldrFeature))
            for (const unsigned candidate : {13u, 12u})
                if (native_engine_m(ldrFeature, candidate == 12) != nullptr ||
                    native_engine_m_rdna3(ldrFeature, candidate == 12) != nullptr)
                {
                    logf("native D3D12: preset %u has no native package and K is not installed; the feature runs %c",
                         feature->preset, candidate == 13 ? 'M' : 'L');
                    feature->preset = candidate;
                    presetM = true;
                    break;
                }
        if (!presetRR && !presetM && (rayReconstruction || feature->preset != 11 || !native_engine_available(ldrFeature)))
        {
            logf("native D3D12 CreateFeature refused: preset=%u flags=0x%x (%s)", feature->preset, feature->flags,
                 rayReconstruction ? "no Ray Reconstruction D/E/F package for it"
                                   : "no Super Resolution K, L or M package for it; the native engine has no other presets");
            delete feature;
            return NGX_FAIL_FEATURE_NOT_SUPPORTED;
        }
    }
    // Direct output needs the native output kernel of the preset in use (verified: K = 11 through
    // hiluma_engine_output, M = 13 through rrlite_downsample_kernel, its last kernel). Only a forced or
    // OptiScaler-chosen Super Sampling preset qualifies; a Ray Reconstruction preset never does here.
    if (!forced.empty() && !rayReconstruction)
    {
        char allowed[128] = "11,13";
        if (const std::string list = env_string("D4R_SHIM_OUTPUT_DIRECT_PRESETS"); !list.empty())
            snprintf(allowed, sizeof(allowed), "%s", list.c_str());
        for (char* token = strtok(allowed, ","); token != nullptr; token = strtok(nullptr, ","))
            feature->outputDirectAllowed |= strtoul(token, nullptr, 0) == feature->preset;
    }
    feature->rayReconstruction = rayReconstruction;
    if (rayReconstruction)
        for (size_t index = 0; index < sizeof(kAuxScalars) / sizeof(kAuxScalars[0]); ++index)
            feature->auxScalarBaseline[index] = get_int_or(parameters, kAuxScalars[index].name, 0);
    logf("NVSDK_NGX_D3D12_CreateFeature(feature=%u%s): %ux%u -> %ux%u quality=%d flags=0x%x subrects=%d preset=%u",
         featureId, rayReconstruction ? ", Ray Reconstruction" : "", feature->width, feature->height,
         feature->outWidth, feature->outHeight, quality, flags, subrects, feature->preset);

    feature->subrects = subrects;
    std::copy(std::begin(presets), std::end(presets), feature->creationPresets.begin());
    // With the engine on, every feature creates its CUDA feature only when a frame falls back: NGX's L kernels take
    // ZLUDA about 20 s to compile on a cold cache, which stalled the game inside CreateFeature (Overwatch,
    // 2026-10-11), and the denoiser's 310.9 kernels 44 s (RE Requiem, 2026-10-14); M's 12 minutes in the Vulkan
    // harness on a loaded machine. A feature the engine refuses pays that at its first frame instead. The native
    // Vulkan frontend's engine (D4R_VULKAN_ENGINE, Super Sampling only) does the same; its staging path creates the
    // CUDA feature at the first frame it takes.
    const bool fsrBackend = rayReconstruction && rr_backend() == RrBackend::Fsr;
    feature->cudaCreationDeferred = nativeVulkan ? engine_requested() && env_uint("D4R_VULKAN_ENGINE", 1) != 0
                                                 : nativeD3d12 || (!fsrBackend && engine_requested());
    // Backend = fsr: no CUDA denoiser feature at all; AMD's denoiser and an internal Super Sampling feature.
    const NgxResult result = fsrBackend                      ? fsr_rr_create(*feature, parameters)
                             : feature->cudaCreationDeferred ? NGX_SUCCESS
                                                             : create_cuda_feature(*feature);
    if (feature->cudaCreationDeferred)
    {
        if (nativeD3d12)
            logf("engine backend: native D3D12 feature, %s preset %c (this route has no CUDA fallback: a frame the "
                 "engine cannot record is logged and the game keeps its own output)",
                 rayReconstruction ? "Ray Reconstruction" : "Super Resolution",
                 feature->preset >= 1 && feature->preset <= 26 ? char('A' + feature->preset - 1) : '?');
        else
            logf("engine backend: CUDA feature creation deferred until fallback is needed");
        // the denoiser's model loads on a thread of its own: start it now, so it is usually ready by the first frame
        // (the native Vulkan frontend starts it once its engine is initialised, NVSDK_NGX_VULKAN_CreateFeature)
        if (rayReconstruction && !nativeVulkan)
            start_engine_model_rr(*feature);
    }
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
        for (int index = 0; index <= kSlots; ++index)
            feature->markerValue[index] = 0;
    // vkd3d-proton's lifetime interface retains these; the native D3D12 engine keeps its own resources alive
    // (the submission hooks)
    if (!nativeVulkan && !nativeD3d12)
        feature->resources = new DeferredResources(feature);
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

bool profile_timestamps_ready(Feature& feature)
{
    if (feature.timestampHeap != nullptr)
        return true;
    if (feature.timestampFailed || g.device == nullptr)
        return false;
    feature.timestampFailed = true;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    UINT64 frequency = 0;
    if (SUCCEEDED(g.device->CreateCommandQueue(&queueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue))))
    {
        queue->GetTimestampFrequency(&frequency);
        queue->Release();
    }
    D3D12_QUERY_HEAP_DESC heapDesc = {};
    heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    heapDesc.Count = 4 * kOutputSlots;
    uint8_t* mapped = nullptr;
    if (frequency == 0 ||
        FAILED(g.device->CreateQueryHeap(&heapDesc, __uuidof(ID3D12QueryHeap), reinterpret_cast<void**>(&feature.timestampHeap))))
    {
        logf("D4R_PROFILE: game-queue timestamps unavailable (frequency %llu)", static_cast<unsigned long long>(frequency));
        feature.timestampHeap = nullptr;
        return false;
    }
    if (!create_buffer(D3D12_HEAP_TYPE_READBACK, 8 * 4 * kOutputSlots, &feature.timestampReadback, &mapped))
    {
        feature.timestampHeap->Release();
        feature.timestampHeap = nullptr;
        return false;
    }
    feature.timestampValues = reinterpret_cast<volatile uint64_t*>(mapped);
    for (int index = 0; index < 4 * kOutputSlots; ++index)
        feature.timestampValues[index] = 0;
    feature.timestampTicksPerMs = static_cast<double>(frequency) / 1000.0;
    feature.timestampFailed = false;
    logf("D4R_PROFILE: game-queue timestamps enabled (%llu Hz)", static_cast<unsigned long long>(frequency));
    return true;
}

void profile_timestamp(Feature& feature, ID3D12GraphicsCommandList* list, uint32_t frame, int point)
{
    const UINT base = 4 * (frame % kOutputSlots);
    list->EndQuery(feature.timestampHeap, D3D12_QUERY_TYPE_TIMESTAMP, base + point);
    if (point == 3)
    {
        feature.timestampOutputFused[frame % kOutputSlots]=feature.recordOutputFused;
        feature.timestampInputFused[frame % kOutputSlots]=feature.recordInputFused;
        list->ResolveQueryData(feature.timestampHeap, D3D12_QUERY_TYPE_TIMESTAMP, base, 4, feature.timestampReadback,
                               8ull * base);
    }
}

// The evaluation kOutputSlots frames earlier, if its resolve has landed; clears the slot for this frame.

void read_profile_timestamps(Feature& feature, uint32_t frame, FrameTiming& timing)
{
    feature.recordOutputFused=feature.recordInputFused=false;
    volatile uint64_t* t = feature.timestampValues + 4 * (frame % kOutputSlots);
    const uint64_t t0 = t[0], t1 = t[1], t2 = t[2], t3 = t[3];
    if (t0 != 0 && t0 <= t1 && t1 <= t2 && t2 <= t3)
    {
        const double perMs = feature.timestampTicksPerMs;
        timing.vkInput = (t1 - t0) / perMs;
        timing.vkDlssWait = (t2 - t1) / perMs;
        timing.vkOutput = (t3 - t2) / perMs;
        timing.vkTotal = (t3 - t0) / perMs;
        timing.vkOutputFused=feature.timestampOutputFused[frame % kOutputSlots];
        timing.vkInputFused=feature.timestampInputFused[frame % kOutputSlots];
    }
    for (int index = 0; index < 4; ++index)
        t[index] = 0;
}

static NgxResult fsr_rr_evaluate(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                                 ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                                 ID3D12Resource* exposure, void* parameters, const FrameParams& p, uint32_t frame);

static NgxResult evaluate(ID3D12GraphicsCommandList* list, const NgxHandle* handle, void* parameters)
{
    std::lock_guard<std::mutex> apiLock(g_featureApiMutex);
    return evaluate_unlocked(list, handle, parameters);
}

// Ray Reconstruction: the denoiser's own per-frame settings, guide origins and camera matrices, read on the game
// thread while the caller's parameter object is certainly alive and carried on by value.

static void read_rr_params(const Feature& feature, void* parameters, FrameParams& p)
{
    for (size_t index = 0; index < sizeof(kAuxScalars) / sizeof(kAuxScalars[0]); ++index)
    {
        int value = 0;
        const bool given = d4r_ngx_get_int(parameters, kAuxScalars[index].name, &value) == NGX_SUCCESS;
        // Not given this frame: back to what the caller asked for at creation, so a
        // setting it has stopped varying does not keep an old per-frame override.
        p.auxScalar[index] = given ? value : feature.auxScalarBaseline[index];
        p.auxScalarSet[index] = true;
    }
    // Each guide's Subrect origin, read here while the caller's object is alive. An origin
    // the caller stops registering resets to zero rather than lingering.
    for (size_t index = 0; index < sizeof(kAuxSubrectNames) / sizeof(kAuxSubrectNames[0]); ++index)
    {
        static std::string nameX, nameY;
        nameX.assign(kAuxSubrectNames[index]).append(".Subrect.Base.X");
        nameY.assign(kAuxSubrectNames[index]).append(".Subrect.Base.Y");
        p.auxSubrect[index][0] = get_uint_or(parameters, nameX.c_str(), 0);
        p.auxSubrect[index][1] = get_uint_or(parameters, nameY.c_str(), 0);
    }
    for (size_t index = 0; index < sizeof(kAuxMatrices) / sizeof(kAuxMatrices[0]); ++index)
    {
        // A matrix is registered as a pointer to float[16] (nvsdk: the SDK's helpers
        // set these with SetVoidPointer), never as sixteen separate scalars.
        void* value = nullptr;
        p.auxMatrixSet[index] =
            d4r_ngx_get_void(parameters, kAuxMatrices[index], &value) == NGX_SUCCESS && value != nullptr;
        if (p.auxMatrixSet[index])
            std::memcpy(p.auxMatrix[index], value, sizeof(p.auxMatrix[index]));
    }
}

// The API lock is held. The FSR Ray Regeneration backend evaluates its internal Super Sampling feature
// through here, in the same command list, without taking the lock again.

NgxResult evaluate_unlocked(ID3D12GraphicsCommandList* list, const NgxHandle* handle, void* parameters)
{
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
    {
        D3D12_RESOURCE_DESC exposureDesc = {};
        if (exposure != nullptr)
            exposure->GetDesc(&exposureDesc);
        logf("evaluate: color %llux%u fmt=%d, depth %llux%u fmt=%d, mv %llux%u fmt=%d, output %llux%u fmt=%d, "
             "exposure=%p %llux%u fmt=%d",
             colorDesc.Width, colorDesc.Height, colorDesc.Format, depthDesc.Width, depthDesc.Height, depthDesc.Format,
             motionDesc.Width, motionDesc.Height, motionDesc.Format, outputDesc.Width, outputDesc.Height,
             outputDesc.Format, exposure, exposureDesc.Width, exposureDesc.Height, exposureDesc.Format);
    }
    if (env_uint("D4R_SHIM_TRACE_MOTION", 0) != 0)
    {
        static const GUID tag = {0xbaecd840,0x42b4,0x4b60,{0x8b,0x44,0x81,0x9a,0x7f,0xa1,0x51,0xa4}};
        const uint32_t info[] = {frame, feature->handle.Id, static_cast<uint32_t>(outputDesc.Width), outputDesc.Height};
        const HRESULT tagged = motion->SetPrivateData(tag, sizeof(info), info);
        if (frame <= 10 || frame % 120 == 0)
        logf("motion producer trace: feature=%u frame=%u resource=%p size=%llux%u format=%u status=0x%08x",
             feature->handle.Id, frame, motion, motionDesc.Width, motionDesc.Height, motionDesc.Format, tagged);
    }
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
    unsigned int exposureWidth = 0, exposureHeight = 0;
    if (exposure != nullptr)
    {
        D3D12_RESOURCE_DESC exposureDesc;
        exposure->GetDesc(&exposureDesc);
        // Recorded before the format test: a capture that kept the plane but not its geometry
        // could not say whether the game passed a scalar or a per-pixel exposure buffer.
        exposureWidth = static_cast<unsigned int>(exposureDesc.Width);
        exposureHeight = static_cast<unsigned int>(exposureDesc.Height);
        if (!supported_input(Plane::Exposure, exposureDesc.Format))
        {
            exposure = nullptr;
            exposureWidth = 0;
            exposureHeight = 0;
        }
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
    p.exposureWidth = exposureWidth;
    p.exposureHeight = exposureHeight;
    // Creation state, so a capture records the model and geometry the game asked for rather
    // than what a replay would otherwise have to guess.
    p.createWidth = feature->width;
    p.createHeight = feature->height;
    p.createOutWidth = feature->outWidth;
    p.createOutHeight = feature->outHeight;
    p.createQuality = feature->quality;
    p.createFlags = feature->flags;
    p.createSubrects = feature->outputSubrects;
    p.createPreset = feature->preset;

    // Ray Reconstruction: the denoiser's own per-frame settings and camera matrices are read
    // here, on the game thread, while the caller's parameter object is certainly alive, and
    // carried to the worker by value.
    if (feature->rayReconstruction)
        read_rr_params(*feature, parameters, p);

    // Backend = fsr: AMD's denoiser, then the internal Super Sampling feature; never NVIDIA's denoiser.
    if (feature->fsr != nullptr)
        return fsr_rr_evaluate(*feature, list, color, depth, motion, output, exposure, parameters, p, frame);

    if (engine_requested() && engine_evaluate(*feature, list, color, depth, motion, output, exposure, parameters, p, frame))
    {
        ++feature->engineFrames;
        return NGX_SUCCESS;
    }
    if (feature->nativeD3d12)
    {
        // no CUDA backend without vkd3d-proton: say so rather than leaving the game's output unwritten in silence
        if (++feature->engineMisses <= 3 || feature->engineMisses % 600 == 0)
            logf("native D3D12 feature %u: frame %u not recorded by the engine (%u so far; %u recorded); the game "
                 "keeps its own output for it",
                 feature->handle.Id, frame, feature->engineMisses, feature->engineFrames);
        return NGX_FAIL_FEATURE_NOT_SUPPORTED;
    }

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

    if (g_vk.lifetime != nullptr ? FAILED(g_vk.lifetime->RetainExternalResources(list, feature->resources))
                                 : !retain_native(list, feature->resources))
    {
        logf("frame %u: unable to retain command-list resources; evaluation not recorded", frame);
        return NGX_FAIL_PLATFORM_ERROR;
    }

    // VRAM interop when every resource qualifies (decided once per feature).
    ID3D12Resource* const inputs[4] = {color, depth, motion, exposure};
    const Plane planes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
    const int inputCount = exposure != nullptr ? 4 : 3;
    VramCopy vramInputs[4], vramOutput;
    // A separate alpha output no longer holds the feature off VRAM interop: the float32 result
    // stays on the GPU and is blitted into the caller's texture when that texture's format
    // supports the conversion (see prepare_alpha_vram). A format that does not - or a texture Vulkan
    // cannot hand over - keeps the alpha on the host route for that frame alone, without the
    // colour output having to fall back with it. The existing cutover below still drains VRAM
    // work if the colour output itself stops qualifying.
    // Native Windows (no vkd3d-proton): D3D12 shared buffers instead (Super Sampling only; Ray Reconstruction's
    // guides and alpha still need the Vulkan route).
    const bool nativeVram = g_vk.interop == nullptr && !feature->rayReconstruction;
    if ((!feature->vramDecided || feature->vram) && nativeVram && native_vram_available())
    {
        p.vram = describe_native_vram(output, vramOutput, true);
        for (int index = 0; index < inputCount && p.vram; ++index)
            p.vram = describe_native_vram(inputs[index], vramInputs[index]);
    }
    else if ((!feature->vramDecided || feature->vram) && vram_interop_available())
    {
        p.vram = describe_vram_copy(output, Plane::Color, vramOutput, true);
        for (int index = 0; index < inputCount && p.vram; ++index)
            p.vram = describe_vram_copy(inputs[index], planes[index], vramInputs[index]);
    }
    if (!feature->vramDecided)
    {
        feature->vramDecided = true;
        feature->vram = p.vram;
        feature->split = p.vram && env_uint("D4R_SHIM_SPLIT_FRAME", 0) != 0 &&
                         (nativeVram ? g_dxv.split && (feature->splitFence != nullptr || create_native_split(*feature))
                                     : g_vk.split != nullptr && (feature->splitSemaphore != VK_NULL_HANDLE ||
                                                                 create_split_semaphore(*feature)));
        feature->linearInputs = feature->split && !nativeVram && env_uint("D4R_SHIM_LINEAR_INPUTS", 0) != 0 &&
                                g.cu.registerLinearTexture != nullptr;
        if (feature->linearInputs)
            logf("linear inputs: NGX samples the interop buffers directly");
        const bool gpuWait = p.vram && initialize_ngx_gpu_wait(*feature);
        if (feature->split && !nativeVram && !feature->rayReconstruction && !feature->engineNetTimeline &&
            env_uint("D4R_SHIM_NATIVE_COMPLETION", 1) != 0 && g.cu.timelineCreate != nullptr &&
            g.cu.timelinePrepare != nullptr && g.cu.timelineCancel != nullptr && g.cu.timelineQueue != nullptr &&
            g.cu.timelineSignal != nullptr && g.cu.timelineDestroy != nullptr && g.cu.timelineWait != nullptr)
        {
            feature->ngxNativeCompletion = g.worker.call([&] {
                return g.cu.timelineCreate(g_vk.device, reinterpret_cast<uint64_t>(feature->splitSemaphore),
                                          &feature->ngxTimeline);
            }) == 0;
            logf("NGX output completion for this feature: %s", feature->ngxNativeCompletion
                ? "native HIP callback signals the split timeline" : "CPU event polling (native callback unavailable)");
        }
        logf("VRAM interop %s for this feature%s%s", p.vram ? "on" : "off",
             feature->split ? ", presenting each frame's own result (split frames)" : "",
             gpuWait ? ", GPU-side input wait on NGX's null stream (CPU waits for diagnostics/host input reads)" :
             p.vram ? ", waiting for input readiness with adaptive CPU polling (or explicit MarkerPollUs)" : "");
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
        p.packedOutput = feature->split && !nativeVram && feature->outputDirectAllowed && feature->preset == 13 && vramOutput.convert &&
                         outputDesc.Format == DXGI_FORMAT_R10G10B10A2_UNORM && vramOutput.width % 2 == 0 &&
                         env_uint("D4R_SHIM_OUTPUT_DIRECT", 0) != 0 && env_uint("D4R_SHIM_OUTPUT_PACKED", 0) != 0 &&
                         feature->packedReady.load();
        p.packedRgba8 = feature->split && !nativeVram && feature->outputDirectAllowed && feature->preset == 11 &&
                       outputDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM && vramOutput.width % 2 == 0 &&
                       env_uint("D4R_SHIM_OUTPUT_DIRECT", 0) != 0 && env_uint("D4R_SHIM_OUTPUT_RGBA8", 1) != 0 &&
                       feature->rgba8Ready.load();
        p.packedOutput |= p.packedRgba8;
        static bool packedLogged = false;
        if (p.packedOutput && !packedLogged)
        {
            packedLogged = true;
            logf("frame %u: packed output (the output kernel stores %s texels; no conversion blit)", frame,
                 p.packedRgba8 ? "RGBA8" : "R10G10B10A2");
        }
        InPlaceTexture outputTexture;
        const size_t outputTexelBytes = p.packedOutput ? 4 : 8;
        if (feature->split && !nativeVram && feature->outputDirectAllowed && (!vramOutput.convert || p.packedOutput) &&
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
    static const bool timestampsWanted = env_uint("D4R_PROFILE_TIMESTAMPS", 1) != 0;
    const bool timestamps = timing.enabled && timestampsWanted && p.vram && feature->split &&
                            profile_timestamps_ready(*feature);
    if (timestamps)
    {
        read_profile_timestamps(*feature, frame, timing);
        profile_timestamp(*feature, list, frame, 0);
    }
    if (p.vram && nativeVram)
    {
        const D3D12_RESOURCE_STATES states[4] = {inputState, depthState, inputState, inputState};
        feature->recordInputFused = false;
        record_native_inputs(list, slot, inputs, planes, states, inputCount);
    }
    else if (p.vram)
    {
        const D3D12_RESOURCE_STATES states[4] = {inputState, depthState, inputState, inputState};
        for (int index = 0; index < inputCount; ++index)
            transition(list, inputs[index], states[index], D3D12_RESOURCE_STATE_COPY_SOURCE);
        const bool fused = ensure_fused_color(*feature, slotIndex, color, vramInputs[0], frame);
        feature->recordInputFused=fused;
        const bool recorded = record_vram_inputs(*feature, list, slot, vramInputs, inputCount, fused);
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

    record_native_motion_capture(list, slot, motion, frame);
    record_native_gpu_check(*feature,list,slot,motion);

    // Frame marker, written once the copies above have completed.
    ID3D12GraphicsCommandList2* list2 = nullptr;
    if (FAILED(list->QueryInterface(__uuidof(ID3D12GraphicsCommandList2), reinterpret_cast<void**>(&list2))))
    {
        logf("ID3D12GraphicsCommandList2 unavailable; cannot place frame marker");
        return NGX_FAIL_PLATFORM_ERROR;
    }
    // Ray Reconstruction's auxiliary surfaces go into their own readback buffers here, still
    // ahead of the frame marker so the worker can read them the moment that marker lands.
    if (feature->rayReconstruction)
    {
        if (!capture_alpha_output(*feature, parameters, slotIndex, frame) ||
            !gather_aux_inputs(*feature, list, parameters, slotIndex, inputState, frame))
        {
            // capture_alpha_output may already have taken a reference on the caller's texture,
            // and this frame is never queued, so nothing else would release it.
            InputSlot& failed = feature->inputs[slotIndex];
            if (failed.alphaResource != nullptr)
            {
                failed.alphaResource->Release();
                failed.alphaResource = nullptr;
                failed.alphaRequested = false;
            }
            list2->Release();
            return NGX_FAIL_PLATFORM_ERROR;
        }
        // The alpha's own route is decided here, on the game thread, next to the capture that
        // named the texture: allocating on the worker is not an option (the CUDA import runs on
        // the worker) and this is the same place that already knows the region and the format.
        prepare_alpha_vram(*feature, slot, p.vram);
    }
    if (timing.enabled)
        timing.inputRecord = profile_ms(inputRecordStart, ProfileClock::now());

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

    if (feature->ngxGpuWait.load())
    {
        p.gpuInputMarker = record_ngx_input_ready(*feature, list, slotIndex, frame);
        if (!p.gpuInputMarker)
        {
            feature->ngxGpuWait.store(false);
            logf("NGX feature %u: recording the GPU input marker failed; using CPU marker polling from now on",
                 feature->handle.Id);
        }
    }

    if (timestamps)
        profile_timestamp(*feature, list, frame, 1);

    // Present the most recent finished DLSS result.
    int latest;
    uint32_t colorFrame = 0; // the frame whose colour is presented below, 0 if none
    {
        std::lock_guard<std::mutex> lock(feature->outputMutex);
        latest = feature->latestOutput.load();
        if (latest >= 0)
        {
            feature->outputs[latest].lastReadFrame = frame;
            colorFrame = feature->outputs[latest].producedFrame.load();
            if (timing.enabled)
                timing.presentedFrame = colorFrame;
        }
    }
    const auto outputRecordStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    if (p.vram && feature->split && p.gpuInputMarker && feature->gpuCompletion)
        p.split = p.gpuCompletion = record_gpu_completion_wait(*feature, list, frame);
    if (p.vram && feature->split && !p.gpuCompletion)
        p.split = nativeVram ? native_split_frame(*feature, list, frame)
                             : SUCCEEDED(g_vk.split->SplitCommandListForExternalWait(
                                   list, reinterpret_cast<UINT64>(feature->splitSemaphore), frame));
    if (!p.split)
    {
        p.outputInPlace = 0; // only a split frame waits for the result before the game reads its texture
        p.packedOutput = false;
    }
    if (p.split && timestamps)
        profile_timestamp(*feature, list, frame, 2);
    if (p.split)
    {
        // Runs in the second half of the list, once frame's result is in place. Split frames
        // present their own result rather than the newest one, so the alpha must pair with
        // that one instead.
        const int target = static_cast<int>(frame % kOutputSlots);
        {
            std::lock_guard<std::mutex> lock(feature->outputMutex);
            colorFrame = feature->outputs[target].producedFrame.load();
        }
        if (p.outputInPlace == 0)
        {
            if (p.packedOutput)
                vramOutput.convert = false; // the buffer already holds the texture's own texels
            // a native split list records this into its second part (d4r_d3d12_split.h)
            if (nativeVram)
                record_native_output(*feature, target, list, output, outputState);
            else if (!record_feature_vram_output(*feature,target,list,output,outputState,vramOutput,frame))
                logf("frame %u: BeginVkCommandBufferInterop failed for the output", frame);
        }
        if (timing.enabled)
            timing.presentedFrame = frame;
    }
    else if (latest >= 0 && p.vram && nativeVram)
        record_native_output(*feature, latest, list, output, outputState);
    else if (latest >= 0 && p.vram)
    {
        if (!record_feature_vram_output(*feature,latest,list,output,outputState,vramOutput,frame))
            logf("frame %u: BeginVkCommandBufferInterop failed for the output", frame);
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
    if (feature->rayReconstruction &&
        !record_alpha_copyback(*feature, list, slot, outputState, colorFrame, p.split))
        return NGX_FAIL_PLATFORM_ERROR;
    if (p.split && timestamps)
        profile_timestamp(*feature, list, frame, 3);
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

// The command list an evaluation records into, as vkd3d-proton's own object: its interop calls (Vulkan recording,
// RetainExternalResources) accept nothing else. Streamline's proxy lists (sl.interposer.dll) hand out the list they
// wrap for this interface; games whose FSR inputs OptiScaler forwards to NGX can pass such a proxy. The proxy keeps
// the wrapped list alive for the call.

static const GUID kStreamlineRetrieveBaseInterface = {0xADEC44E2, 0x61F0, 0x45C3, {0xAD, 0x9F, 0x1B, 0x37, 0x37, 0x92, 0x84, 0xFF}};

static ID3D12GraphicsCommandList* native_command_list(ID3D12GraphicsCommandList* list)
{
    if (!list)
        return list;
    static std::atomic<bool> logged{false};

    // The d4r-patched vkd3d-proton lists implement ID3D12DXVKInteropCommandListD4R. A proxy layer that
    // forwards unknown interfaces to the wrapped object (ReShade) resolves to the native list through it.
    ID3D12DXVKInteropCommandListD4R* d4r = nullptr;
    if (SUCCEEDED(list->QueryInterface(__uuidof(ID3D12DXVKInteropCommandListD4R), reinterpret_cast<void**>(&d4r))) && d4r)
    {
        ID3D12GraphicsCommandList* native = nullptr;
        const HRESULT hr = d4r->GetCommandList(&native);
        d4r->Release();
        if (SUCCEEDED(hr) && native)
        {
            // GetCommandList hands out its own reference (COM out-parameter convention); drop it
            // now, the returned pointer stays valid while the proxy that wraps the list is alive.
            const bool proxy = native != list;
            native->Release();
            if (proxy)
            {
                if (!logged.exchange(true))
                    logf("evaluate: the command list is a proxy; recording into the native list it wraps");
                return native;
            }
        }
    }

    IUnknown* base = nullptr;
    if (SUCCEEDED(list->QueryInterface(kStreamlineRetrieveBaseInterface, reinterpret_cast<void**>(&base))) && base)
    {
        ID3D12GraphicsCommandList* native = nullptr;
        const HRESULT hr = base->QueryInterface(__uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&native));
        base->Release();
        if (SUCCEEDED(hr) && native)
        {
            native->Release();
            if (!logged.exchange(true))
                logf("evaluate: the command list is a Streamline proxy; recording into the list it wraps");
            return native;
        }
    }
    if (!logged.exchange(true))
    {
        // which module implements the list (vkd3d-proton's d3d12core.dll unless something wraps it)
        HMODULE module = nullptr;
        char path[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>((*reinterpret_cast<void***>(list))[0]), &module))
            GetModuleFileNameA(module, path, sizeof(path));
        logf("evaluate: command list implemented by %s", path);
    }
    return list;
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList* list, const NgxHandle* handle,
                                                     void* parameters, void*)
{
    return evaluate(native_command_list(list), handle, parameters);
}

D4R_EXPORT NgxResult NVSDK_NGX_D3D12_EvaluateFeature_C(ID3D12GraphicsCommandList* list, const NgxHandle* handle,
                                                       void* parameters, void*)
{
    return evaluate(native_command_list(list), handle, parameters);
}

void release_feature(Feature* feature)
{
    // The FSR backend's internal Super Sampling feature first, through this same function.
    fsr_rr_release(*feature);
    {
        std::lock_guard<std::mutex> lock(feature->ngxWaitMutex);
        release_ngx_input_wait_locked(*feature, "the feature is being released", false);
    }
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
        // Every denoiser input dies with its feature, drains to nothing first.
        for (AuxSurface& surface : feature->aux)
            destroy_aux_surface(surface);
        feature->aux.clear();
        if (feature->alphaImage.object != 0)
            g.cu.surfObjectDestroy(feature->alphaImage.object);
        if (feature->alphaImage.array != nullptr)
            g.cu.arrayDestroy(feature->alphaImage.array);
        release_host(feature->alphaHost);
        for (AlphaSlot& slot : feature->alpha)
            slot.staging.release();
        for (InputSlot& slot : feature->inputs)
            if (slot.alphaResource != nullptr)
                slot.alphaResource->Release();
        for (InputSlot& slot : feature->inputs)
        {
            for (CudaObject& texture : slot.linearTexture)
                if (texture != 0)
                    g.cu.texObjectDestroy(texture), texture = 0;
            if (slot.dilatedTexture != 0)
                g.cu.texObjectDestroy(slot.dilatedTexture), slot.dilatedTexture = 0;
            if (slot.dilatedMotion != 0)
                g.cu.memFree(slot.dilatedMotion), slot.dilatedMotion = 0;
            if(slot.nativeGpuTexture) g.cu.texObjectDestroy(slot.nativeGpuTexture),slot.nativeGpuTexture=0;
            if(slot.nativeGpuResult) g.cu.memFree(slot.nativeGpuResult),slot.nativeGpuResult=0;
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
        for(CudaEvent event:feature->nativePassEvents) if(event) g.cu.eventDestroy(event);
        if (feature->profileInputsReleased != nullptr)
            g.cu.eventDestroy(feature->profileInputsReleased);
        if (feature->timestampHeap != nullptr)
            feature->timestampHeap->Release();
        if (feature->timestampReadback != nullptr)
            feature->timestampReadback->Release();
        if (feature->outputReadyEvent != nullptr)
            g.cu.eventDestroy(feature->outputReadyEvent);
        if (feature->scratch != 0)
            g.cu.memFree(feature->scratch);
        for (CudaDevicePtr buffer : feature->planeLinear)
            if (buffer != 0)
                g.cu.memFree(buffer);
        if (feature->outputLinear != 0)
            g.cu.memFree(feature->outputLinear);
        if(feature->nativeThinMask) g.cu.memFree(feature->nativeThinMask);
        if(feature->preparedWireAxes) g.cu.memFree(feature->preparedWireAxes);
        if(feature->wirePatchMask) g.cu.memFree(feature->wirePatchMask);
        for(CudaDevicePtr buffer:feature->wireBuffers) if(buffer) g.cu.memFree(buffer);
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
    release_engine_dx(*feature);
    // Recorded lists retain this owner through their allocator. Reset/discard
    // or GPU completion followed by allocator destruction releases those refs.
    if (feature->resources != nullptr)
        feature->resources->Release();
    else
    {
        release_engine_network(*feature);
        fsr_rr_destroy(*feature);
        delete feature;
    }
}

void destroy_recorded_resources(Feature* feature)
{
    for(InputSlot& slot : feature->inputs)
        if(slot.nativeValidityOwner) { slot.nativeValidityOwner->Release();slot.nativeValidityOwner=nullptr; }
    for (InputSlot& slot : feature->inputs)
        for (Staging* staging : {&slot.color, &slot.depth, &slot.motion, &slot.exposure, &slot.nativeMotionCapture, &slot.nativeDepthCapture, &slot.nativeSceneCapture, &slot.nativeOpaqueCapture, &slot.ownedSceneCapture, &slot.ownedMatrixCapture, &slot.ownedDepthCapture, &slot.originalOpaqueCopyCapture, &slot.glassMaskCapture, &slot.materialSceneCapture, &slot.ownedValidityCapture, &slot.glassValidityCapture, &slot.nativePrimaryDepthCapture, &slot.opaqueSceneCapture, &slot.opaqueMatrixCapture, &slot.originalPrimaryDepthCapture, &slot.originalGeometryCapture, &slot.originalHardwareDepthCapture})
            staging->release();
    for (OutputSlot& slot : feature->outputs)
        slot.staging.release();
    release_engine_network(*feature);
    const bool engineSemaphore = !feature->vramDecided && feature->splitSemaphore != VK_NULL_HANDLE;
    if ((g_vk.ready || g_dxv.ready) && feature->vramDecided)
        release_vram(*feature);
    else if (engineSemaphore) // created for the engine's HIP network; release_vram destroys it otherwise
        g_vk.destroySemaphore(g_vk.device, feature->splitSemaphore, nullptr);
    if (feature->marker != nullptr)
        feature->marker->Release();
    fsr_rr_destroy(*feature);
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
            if (&candidate->handle == handle && !candidate->nativeVulkan && candidate->owner == nullptr)
            {
                // The API mutex serializes releases and release_feature removes the registry entry. An
                // internal drain failure may already have marked this feature retiring; it still needs
                // the caller's normal release to clean up, rather than becoming an unreleasable handle.
                candidate->retiring.store(true);
                feature = candidate;
                break;
            }
    }
    if (feature == nullptr || feature->nativeVulkan)
        return NGX_FAIL_INVALID_PARAMETER;
    if (feature->engineFrames != 0)
        logf("NVSDK_NGX_D3D12_ReleaseFeature(id=%u) after %u engine frames and %u completed CUDA frames",
             feature->handle.Id, feature->engineFrames, feature->completedFrames.load());
    else
        logf("NVSDK_NGX_D3D12_ReleaseFeature(id=%u) after %u completed frames", feature->handle.Id,
             feature->completedFrames.load());
    release_feature(feature);
    return NGX_SUCCESS;
}

void finish_shutdown()
{
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!g.shutdownPending || g.resourceOwners.load() != 0)
        return;
    if (g_vk.ready)
        free_vram_pool();
    if (g_fusedColorReady) { g_fusedColorStage.destroy(); g_fusedColorReady = false; }
    if (g.ngxInitialized)
    {
        drain_pipeline();
        if (!g.nativeD3d12Only) g.worker.call([] { return g.ngx.shutdown(); });
    }
    // No separate denoiser shutdown: the core loaded it and the core's shutdown releases it.
    g.ngxInitialized = false;
    g.nativeD3d12Only = false;
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
        // Internal features (the FSR backend's Super Sampling stage) go with their owner.
        for (Feature* feature : g_features)
            if (feature->owner == nullptr)
                features.push_back(feature);
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

#include "d4r_fsr_rr_backend.inl"

#include "d4r_vulkan_frontend.inl"

// Proxy filenames vary (dxgi.dll, winmm.dll, ...). Identify OptiScaler by its
// PE export name rather than guessing from the EXE or plugin folder location.

static std::wstring optiscaler_config_path()
{
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) return {};
    MODULEENTRY32W module = {};
    module.dwSize = sizeof(module);
    std::wstring result;
    if (Module32FirstW(snapshot, &module)) do
    {
        const BYTE* base = module.modBaseAddr;
        const size_t size = module.modBaseSize;
        if (size < sizeof(IMAGE_NT_HEADERS64)) continue;
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0 ||
            static_cast<size_t>(dos->e_lfanew) > size - sizeof(IMAGE_NT_HEADERS64)) continue;
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
        if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) continue;
        const DWORD rva = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress;
        if (!rva || rva > size - sizeof(IMAGE_EXPORT_DIRECTORY)) continue;
        const auto* exports = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + rva);
        if (exports->Name >= size) continue;
        const char* name = reinterpret_cast<const char*>(base + exports->Name);
        if (!std::memchr(name, 0, size - exports->Name) || _stricmp(name, "OptiScaler.dll") != 0) continue;
        result = module.szExePath;
        result.resize(result.find_last_of(L"\\/"));
        result += L"\\OptiScaler.ini";
        break;
    } while (Module32NextW(snapshot, &module));
    CloseHandle(snapshot);
    return result;
}

// OptiScaler calls this after LoadLibrary returns, outside our DllMain.
// Launch first-time setup in a separate native process; never do network/GPU
// work under the loader lock. CUDA/NGX work remains lazy until NGX Init.
// NvngxPath must name this ASI, so backend calls reuse this module's state.

D4R_EXPORT void InitializeASI()
{
    static std::once_flag once;
    std::call_once(once, [] {
        wchar_t exe[MAX_PATH] = {}, plugin[MAX_PATH] = {};
        if (!GetModuleFileNameW(nullptr, exe, MAX_PATH) ||
            !GetModuleFileNameW(g_selfModule, plugin, MAX_PATH))
            return;
        if (!d4r_asi_game_process(exe)) return;
        logf("InitializeASI: d4r ASI backend ready (GPU initialization deferred)");
        if (env_uint("D4R_ASI_AUTOINSTALL", 1) == 0) return;
        std::wstring game(exe);
        game.resize(game.find_last_of(L"\\/"));
        std::wstring pluginDir(plugin);
        pluginDir.resize(pluginDir.find_last_of(L"\\/"));
        const std::wstring helper = pluginDir + L"\\d4r\\d4r-setup.exe";
        const std::wstring optiscalerConfig = optiscaler_config_path();
        std::wstring command = L"\"" + helper + L"\" \"" + game + L"\" \"" + plugin + L"\" " +
                               std::to_wstring(GetCurrentProcessId()) + L" \"" + optiscalerConfig + L"\"";
        STARTUPINFOW startup = {};
        startup.cb = sizeof(startup);
        PROCESS_INFORMATION process = {};
        if (CreateProcessW(helper.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                           nullptr, game.c_str(), &startup, &process))
        {
            CloseHandle(process.hThread);
            CloseHandle(process.hProcess);
        }
        else
            logf("InitializeASI: automatic setup could not start (%lu); extract the complete ASI archive", GetLastError());
    });
}

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

extern "C" BOOL WINAPI d4r_dll_entry(HANDLE instance, DWORD reason, LPVOID reserved)
{
    if (reason == DLL_PROCESS_DETACH && reserved != nullptr)
    {
        native_vk_process_exit();
        return TRUE;
    }
    return DllMainCRTStartup(instance, reason, reserved);
}
