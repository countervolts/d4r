// d4r_nvngx_shim_config.cpp: the portable install's settings file. Reads the INI beside this DLL,
// turns it into the environment and search paths the rest of the shim reads, and resolves the model
// directories. No CUDA, NGX or D3D12 yet.

// the command-list stubs live in d4r_nvngx_shim.cpp, which assembles them once
#define D4R_D3D12_SPLIT_DECL 1
#include "d4r_nvngx_shim_internal.h"

std::string ascii_lower(std::string text)
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

std::vector<IniEntry> parse_ini(const std::string& text)
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

int ini_flag(const std::vector<IniEntry>& ini, const char* section, const char* key, int fallback)
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

bool read_whole_file(const std::wstring& path, std::string& text)
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

bool file_exists(const std::wstring& path)
{
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

// Wine's kernel32 maps a Windows path to the Linux one.

std::string unix_path(const std::wstring& path)
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

std::wstring widen(const std::string& text)
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

// The C runtime keeps its own copy of the environment, taken when it started: SetEnvironmentVariable alone is never
// seen by std::getenv, which the engine runtimes (engine/d3d12) and a few shim checks (D4R_DX_NET) use, so a d4r.ini
// setting such as [Env] D4R_D3D12_CAPTURE_DIR had no effect there. Set both.

static void portable_set(const char* name, const std::string& value)
{
    if (GetEnvironmentVariableA(name, nullptr, 0) == 0)
    {
        SetEnvironmentVariableA(name, value.c_str());
        _putenv_s(name, value.c_str());
    }
}

static void portable_set(const wchar_t* name, const std::wstring& value)
{
    if (GetEnvironmentVariableW(name, nullptr, 0) == 0)
    {
        SetEnvironmentVariableW(name, value.c_str());
        _wputenv_s(name, value.c_str());
    }
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

void load_portable_config()
{
    wchar_t module[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(g_selfModule, module, MAX_PATH);
    if (length == 0 || length >= MAX_PATH)
        return;
    std::wstring dir(module, length);
    // ASI delivery keeps its runtime in a companion d4r/ folder beside the ASI.
    // DLL delivery stays self-relative.
    const auto dot = dir.find_last_of(L'.');
    if (dot != std::wstring::npos && _wcsicmp(dir.c_str() + dot, L".asi") == 0)
    {
        dir.resize(dir.find_last_of(L"\\/"));
        dir += L"\\d4r";
    }
    else
        dir.resize(dir.find_last_of(L"\\/"));
    g_portable.dir = dir; // also without d4r.ini: the default log goes here
    std::string text;
    if (!read_whole_file(dir + L"\\d4r.ini", text))
        return; // developer layout: the launcher's environment has everything
    const std::vector<IniEntry> ini = parse_ini(text);
    g_portable.active = true;
    // Native Windows (no Wine): ZLUDA's Windows nvcuda.dll and d4r's NVAPI in the folder, HIP from the AMD driver
    g_portable.nativeWindows = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") == nullptr;
    g_portable.unixDir = g_portable.nativeWindows ? std::string() : unix_path(dir);
    if (!g_portable.nativeWindows && g_portable.unixDir.empty())
        g_portable.notes.push_back("cannot map the d4r folder to a Linux path");

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
    if (g_portable.nativeWindows)
    {
        // The NGX core takes nvapi64.dll from System32 only; load_libraries preloads this one. ZLUDA keeps its
        // compiled kernels in %LOCALAPPDATA%\zluda and finds HIP (amdhip64_7.dll) where the AMD driver put it.
        portable_set(L"D4R_NVAPI", dir + L"\\nvapi\\nvapi64.dll");
        const std::wstring zludaDir = zluda.empty() ? dir + L"\\zluda" : portable_path(zluda);
        char path[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (zludaDir + L"\\nvcuda.dll").c_str(), -1, path, sizeof(path), nullptr, nullptr);
        portable_set_unix("D4R_ZLUDA_LIBCUDA", path);
    }
    else
    {
        portable_set_unix("D4R_ZLUDA_LIBCUDA", (zluda.empty() ? g_portable.unixDir + "/zluda" : zluda) + "/libcuda.so");
        // The release's bundled ROCm runtime unless d4r.ini names another
        const std::string rocm = ini_value(ini, "paths", "RocmDir");
        if (!rocm.empty())
            portable_set_unix("D4R_ROCM_DIR", rocm);
        else if (!g_portable.unixDir.empty() && GetFileAttributesW((dir + L"\\rocm\\lib").c_str()) != INVALID_FILE_ATTRIBUTES)
            portable_set_unix("D4R_ROCM_DIR", g_portable.unixDir + "/rocm");
        const std::string cache = ini_value(ini, "paths", "CacheDir");
        portable_set_unix("D4R_ZLUDA_CACHE_HOME", cache.empty() ? "~/.cache/d4r" : cache);
    }

    std::string model = ini_value(ini, "dlss", "Model");
    if (ascii_lower(model) != "game")
        model = preset_number(model.empty() ? "K" : model);
    else
        model.clear();
    if (!model.empty())
        portable_set("D4R_DLSS_PRESET", model);
    // [OptiScaler] FollowPresets: its Render Presets Override picks the model in game (d4r_render_presets.h);
    // Fsr4: OptiScaler and AMD's FidelityFX modules see the real GPU, for its FSR 4 option (d4r_fsr4_identity.h).
    if (ini_flag(ini, "optiscaler", "FollowPresets", 1) == 0)
        portable_set("D4R_FOLLOW_OPTISCALER", "0");
    if (ini_flag(ini, "optiscaler", "Fsr4", 1) == 0)
        portable_set("D4R_FSR4_IDENTITY", "0");

    // Ray Reconstruction is configured on its own section and drives its own feature, so
    // nothing here reaches [DLSS] Model and nothing there reaches this.
    //
    // Enable: true -> D4R_RR_ENABLE=1, false -> 0, auto/unset -> the variable is left alone,
    // which is the same thing as auto here: Ray Reconstruction is offered exactly when a
    // CUDA-capable denoiser really loaded and initialised.
    const std::string rrEnable = ascii_lower(ini_value(ini, "rayreconstruction", "Enable"));
    if (rrEnable == "true" || rrEnable == "on" || rrEnable == "1")
        portable_set("D4R_RR_ENABLE", "1");
    else if (rrEnable == "false" || rrEnable == "off" || rrEnable == "0")
        portable_set("D4R_RR_ENABLE", "0");
    else if (!rrEnable.empty() && rrEnable != "auto")
        g_portable.notes.push_back("d4r.ini: [RayReconstruction] Enable must be true, false or auto, not '" +
                                   rrEnable + "'; treating it as auto");

    // Model: auto/unset leaves the choice to the game; "default" is the SDK's Default preset
    // (0); D (4) and E (5) are the two the SDK gives a meaning to. F and above are
    // documented as reverting to default behaviour, so they are not offered. E is the model
    // to pick when the depth-of-field guide is in use. These are the same values
    // scripts/d4r_config.py accepts, so both ways of reading d4r.ini agree.
    const std::string rrModel = ascii_lower(ini_value(ini, "rayreconstruction", "Model"));
    if (!rrModel.empty() && rrModel != "auto" && rrModel != "game")
    {
        std::string upper = rrModel;
        for (char& c : upper)
            c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
        if (upper == "DEFAULT")
            portable_set("D4R_RR_PRESET", "0");
        else if (upper == "D")
            portable_set("D4R_RR_PRESET", "4");
        else if (upper == "E")
            portable_set("D4R_RR_PRESET", "5");
        else if (upper == "F")   // DLSS 4.5 Ray Reconstruction: nvngx_dlssd.dll 310.9 or later
            portable_set("D4R_RR_PRESET", "6");
        else if (upper == "A" || upper == "B" || upper == "C")   // the retired CNN (engine\rr-a, rr-b, rr-c): engine only
            portable_set("D4R_RR_PRESET", std::to_string(upper[0] - 'A' + 1).c_str());
        else if (upper == "0" || upper == "4" || upper == "5" || upper == "6")
            portable_set("D4R_RR_PRESET", upper);
        else
            g_portable.notes.push_back("d4r.ini: [RayReconstruction] Model must be auto, default, A, B, C, D, E or F, not '" +
                                       rrModel + "'; leaving the model to the game");
    }
    // Backend: dlss (NVIDIA's denoiser, the default) or fsr (AMD FSR Ray Regeneration, then Super
    // Sampling; d4r_fsr_rr_backend.inl). FsrDenoiser: the AMD DLL, auto = in the d4r folder.
    const std::string rrBackend = ascii_lower(ini_value(ini, "rayreconstruction", "Backend"));
    if (rrBackend == "dlss" || rrBackend == "fsr")
        portable_set("D4R_RR_BACKEND", rrBackend);
    else if (!rrBackend.empty() && rrBackend != "auto")
        g_portable.notes.push_back("d4r.ini: [RayReconstruction] Backend must be dlss or fsr, not '" + rrBackend +
                                   "'; using dlss");
    // FSR denoiser tuning (d4r_fsr_rr_backend.inl): FsrStabilityBias = auto (AMD's default) or a number;
    // FsrDetail = the share (0-1) of the colour kept out of the denoiser, 0 = off.
    const std::string fsrStability = ini_value(ini, "rayreconstruction", "FsrStabilityBias");
    if (!fsrStability.empty() && ascii_lower(fsrStability) != "auto")
        portable_set("D4R_FSR_RR_STABILITY_BIAS", fsrStability);
    const std::string fsrDetail = ini_value(ini, "rayreconstruction", "FsrDetail");
    if (!fsrDetail.empty())
        portable_set("D4R_FSR_RR_DETAIL", fsrDetail);
    const std::string fsrDenoiser = ini_value(ini, "rayreconstruction", "FsrDenoiser");
    portable_set(L"D4R_FSR_RR_DLL", fsrDenoiser.empty() || ascii_lower(fsrDenoiser) == "auto"
                                        ? dir + L"\\amd_fidelityfx_denoiser_dx12.dll"
                                        : portable_path(fsrDenoiser));

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
    // ... and for the shim: vkhip's K network takes the accuracy kernel set (create_engine_vkhip)
    portable_set("D4R_PREFER_ACCURACY", ini_flag(ini, "kernels", "PreferAccuracy", 0) ? "1" : "0");
    portable_set_unix("D4R_NATIVE_SWIN_ENCODERS", ini_flag(ini, "kernels", "NativeSwinEncoders", 1) ? "1" : "0");
    const bool nativeOn = native.empty() || native == "on" || native == "true" || native == "1" || native == "fast";
    if (!nativeOn && native != "off" && native != "false" && native != "0")
        g_portable.notes.push_back("d4r.ini: [Kernels] NativeKernels must be on or off, not '" + native + "'");
    if (nativeOn && !g_portable.unixDir.empty())
        portable_set_unix("D4R_ZLUDA_NATIVE_DIR", g_portable.unixDir + "/kernels");
    else if (nativeOn && g_portable.nativeWindows)
    {
        char path[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (dir + L"\\kernels").c_str(), -1, path, sizeof(path), nullptr, nullptr);
        portable_set_unix("D4R_ZLUDA_NATIVE_DIR", path);
    }
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
    if (!poll.empty() && ascii_lower(poll) != "auto")
        portable_set("D4R_SHIM_MARKER_POLL_US", poll);
    // Direct output takes effect only while NGX runs the native output kernel (the bridge reports it); the
    // release cannot ship that kernel, which is built from NVIDIA's PTX.
    portable_set("D4R_SHIM_OUTPUT_DIRECT", ini_flag(ini, "interop", "DirectOutput", nativeOn) ? "1" : "0");

    // [Engine]: d4r's own Vulkan inference (K, L, M and Ray Reconstruction); the models are looked up in d4r\engine.
    // Enabled = auto (the default) runs it when install.sh built a model there: it is the faster path, and a model
    // the device cannot run, or a feature no model covers, still goes to the CUDA backend (d4r_nvngx.log says why).
    {
        const std::string enabled = ascii_lower(ini_value(ini, "engine", "Enabled"));
        bool on = false;
        if (enabled.empty() || enabled == "auto")
        {
            for (const wchar_t* model : {L"k", L"k-ldr", L"m", L"m-ldr", L"l", L"l-ldr", L"rr-d", L"rr-e", L"rr-f", L"rr-d-gfx11", L"rr-e-gfx11"})
            {
                const DWORD attributes = GetFileAttributesW((dir + L"\\engine\\" + model).c_str());
                if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY))
                    on = true;
            }
        }
        else
            on = ini_flag(ini, "engine", "Enabled", 0) != 0;
        portable_set("D4R_ENGINE", on ? "1" : "0");
        // auto leaves K on the native kernels on RDNA3: K's Vulkan network is slower there (2026-10-17, 7700 XT,
        // 1706x960 -> 1440p: 3.59 ms vs 2.35 ms dlss_gpu). RDNA4 keeps K on the engine (engine_k_refused).
        // Enabled = true, or Network = hip with a model that has the HIP layers, keeps it on every GPU; vkhip
        // (the native layers inside the game's command list, faster than both) takes K there when it loads.
        if (enabled.empty() || enabled == "auto")
            portable_set("D4R_ENGINE_K", "0");
    }
    portable_set("D4R_ENGINE_VKHIP", ini_flag(ini, "engine", "ExperimentalVkHip", 1) ? "1" : "0");
    // Ray Reconstruction's native stages follow ExperimentalVkHip (on by default for A - F on AMD's driver, A - E on
    // RADV); an explicit true or false overrides that ("auto", d4r.ini.default's value, keeps the default)
    if (const std::string rr = ascii_lower(ini_value(ini, "engine", "ExperimentalRrVkHip")); !rr.empty() && rr != "auto")
        portable_set("D4R_ENGINE_RR_VKHIP", ini_flag(ini, "engine", "ExperimentalRrVkHip", 1) ? "1" : "0");
    else if (!ini_flag(ini, "engine", "ExperimentalVkHip", 1))
        portable_set("D4R_ENGINE_RR_VKHIP", "0");
    if (const std::string engineModel = ini_value(ini, "engine", "ModelDir"); !engineModel.empty())
        portable_set(L"D4R_ENGINE_MODEL_DIR", portable_path(engineModel));
    // Network: the network on the engine's Vulkan shader ("vulkan", and the default: "auto") or on the native HIP
    // layers ("hip", when the model has them)
    if (const std::string network = ini_value(ini, "engine", "Network"); network == "hip" || network == "vulkan")
        portable_set("D4R_ENGINE_NETWORK", network);
    else if (!network.empty() && network != "auto")
        g_portable.notes.push_back("[Engine] Network must be auto, hip or vulkan; using auto");

    portable_set("D4R_SHIM_WATERMARK", ini_flag(ini, "dlss", "ShowWatermark", 0) ? "1" : "0");
    g_portable.nativeMotionVectors = ini_flag(ini, "interop", "NativeMotionVectors", 0) != 0;
    portable_set("D4R_NATIVE_MV_LIVE", g_portable.nativeMotionVectors ? "1" : "0");
    portable_set("D4R_MOTION_THIN_SKY", ini_flag(ini, "interop", "MotionDilationThinSky", 0) ? "1" : "0");
    const std::string dilation = ini_value(ini, "interop", "MotionVectorDilation");
    if (dilation.empty() || dilation == "0" || dilation == "1" || dilation == "2")
        portable_set("D4R_MOTION_DILATION", dilation.empty() ? "0" : dilation);
    else
        g_portable.notes.push_back("MotionVectorDilation must be 0, 1 or 2; using original motion vectors");
    if (ini_flag(ini, "debug", "Profile", 0))
        portable_set("D4R_PROFILE", "1");
    if (const std::string every = ini_value(ini, "debug", "ProfileLogEvery");
        !every.empty() && ascii_lower(every) != "auto")
    {
        char* end = nullptr;
        const unsigned long long value = std::strtoull(every.c_str(), &end, 10);
        if (every[0] >= '0' && every[0] <= '9' && end != every.c_str() && *end == '\0' &&
            value >= 1 && value <= UINT32_MAX)
            portable_set("D4R_PROFILE_LOG_EVERY", std::to_string(value));
        else
            g_portable.notes.push_back("d4r.ini: [Debug] ProfileLogEvery must be 1-4294967295; using default 60");
    }
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
    // Only a note: without the denoiser Super Sampling is unaffected and Ray Reconstruction
    // simply reports itself unavailable, which is the same thing the capability answer says.
    char backend[8] = {};
    const DWORD backendLength = GetEnvironmentVariableA("D4R_RR_BACKEND", backend, sizeof(backend));
    const bool fsrBackend = backendLength > 0 && backendLength < sizeof(backend) && ascii_lower(backend) == "fsr";
    if (ini_flag(ini, "rayreconstruction", "Enable", 1) && fsrBackend)
    {
        wchar_t fsrPath[MAX_PATH] = {};
        const DWORD fsrLength = GetEnvironmentVariableW(L"D4R_FSR_RR_DLL", fsrPath, MAX_PATH);
        if (fsrLength > 0 && fsrLength < MAX_PATH && !file_exists(fsrPath))
            g_portable.notes.push_back("[RayReconstruction] Backend = fsr, but AMD's amd_fidelityfx_denoiser_dx12.dll "
                                       "(FidelityFX SDK 2.3, FSR Ray Regeneration 1.2) is missing from the d4r folder");
    }
    else if (ini_flag(ini, "rayreconstruction", "Enable", 1) && !file_exists(dir + L"\\nvngx_dlssd.dll"))
        g_portable.notes.push_back("nvngx_dlssd.dll is missing: copy NVIDIA's DLSS-Denoiser library (3.10.7, the "
                                   "build that exports NVSDK_NGX_CUDA_*) into the d4r folder to get Ray "
                                   "Reconstruction");
    wchar_t corePath[MAX_PATH] = {};
    const DWORD coreLength = GetEnvironmentVariableW(L"D4R_NGX_CORE", corePath, MAX_PATH);
    if (coreLength > 0 && coreLength < MAX_PATH && !file_exists(corePath))
        g_portable.notes.push_back("the NGX core is missing: copy NVIDIA's _nvngx.dll into d4r\\ngx");
}
