#pragma once
// Drag-in install, as on Linux: the game folder holds dxgi.dll (OptiScaler),
// OptiScaler.ini and a d4r folder containing this shim and d4r.ini. When the
// launcher has not set D4R_HIP_ROOT and d4r.ini exists beside this DLL, the
// shim derives the backend's environment itself, so the game can be started
// from Steam, Epic or a shortcut. Without d4r.ini nothing changes: diagnostic
// runs keep using the launcher's environment.
//
// Codegen switches are fixed to the Windows-validated values; d4r.ini only
// selects the model (K/M), diagnostics and the cache folder. NVIDIA DLLs must
// match an identity in dll-pins.txt before anything is loaded.
#include "hip_api.h"
#include "ngx_parameters.h"
#include <bcrypt.h>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>

namespace d4r::win::player {
struct IniEntry { std::string section, key, value; };
struct Pin { std::string kind, sha256, version, status; };
struct Settings {
    unsigned preset = 11; // K
    bool asyncInterop = false, validateOutput = false, verbose = false;
    std::wstring cacheDir;
};
struct State {
    bool active = false;
    unsigned preset = 0;
    std::filesystem::path dir;
    std::vector<std::string> notes;
};
inline State& state() { static State value; return value; }

inline std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}
inline std::string lower(std::string value) {
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}
inline std::string upper(std::string value) {
    for (char& c : value) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return value;
}
inline std::filesystem::path module_directory() {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&module_directory), &module)) throw std::runtime_error("Locate the d4r shim directory");
    wchar_t path[32768]{}; const DWORD length = GetModuleFileNameW(module, path, 32768);
    if (!length || length == 32768) throw std::runtime_error("d4r shim path is invalid");
    return std::filesystem::path(path).parent_path();
}
inline std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read " + diag::utf8(path.c_str()));
    std::stringstream text; text << file.rdbuf(); return text.str();
}

// The configparser dialect of packaging/d4r.ini and the Linux shim: [Section],
// key = value, full-line ; or # comments, inline comments after whitespace.
inline std::vector<IniEntry> parse_ini(const std::string& text) {
    std::vector<IniEntry> entries; std::string section;
    size_t start = text.compare(0, 3, "\xEF\xBB\xBF") == 0 ? 3 : 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        const std::string line = trim(text.substr(start, end - start));
        start = end + 1;
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line[0] == '[') { section = trim(line.substr(1, line.find(']') - 1)); continue; }
        const size_t equals = line.find('=');
        if (equals == std::string::npos) continue;
        std::string value = line.substr(equals + 1);
        for (size_t i = 1; i < value.size(); ++i)
            if ((value[i] == ';' || value[i] == '#') && (value[i - 1] == ' ' || value[i - 1] == '\t')) { value.resize(i); break; }
        entries.push_back({section, trim(line.substr(0, equals)), trim(value)});
    }
    return entries;
}

// Invalid values fall back to the safe default and are reported in the log
// (and by d4r-check.ps1), rather than silently changing behaviour.
inline Settings read_settings(const std::vector<IniEntry>& ini, std::vector<std::string>& notes) {
    Settings settings;
    const auto flag = [&](const IniEntry& entry, bool fallback) {
        const auto value = lower(entry.value);
        if (value.empty() || value == "auto") return fallback;
        if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
        if (value == "0" || value == "false" || value == "no" || value == "off") return false;
        notes.push_back("d4r.ini: [" + entry.section + "] " + entry.key + " must be true or false, not '" + entry.value +
            "'; using " + (fallback ? "true" : "false"));
        return fallback;
    };
    for (const auto& entry : ini) {
        const auto name = lower(entry.section + "." + entry.key); // Matched case-insensitively, reported as written.
        if (name == "dlss.model") {
            const auto value = upper(entry.value);
            if (value.empty() || value == "AUTO" || value == "K" || value == "DLSS4" || value == "11") settings.preset = 11;
            else if (value == "M" || value == "DLSS4.5" || value == "13") settings.preset = 13;
            else notes.push_back("d4r.ini: [DLSS] Model '" + entry.value + "' is not available on Windows (use K or M); using K");
        } else if (name == "interop.asyncinterop") settings.asyncInterop = flag(entry, false);
        else if (name == "debug.validateoutput") settings.validateOutput = flag(entry, false);
        else if (name == "debug.log") {
            const auto value = lower(entry.value);
            if (value == "verbose") settings.verbose = true;
            else if (!value.empty() && value != "normal" && value != "auto")
                notes.push_back("d4r.ini: [Debug] Log must be normal or verbose, not '" + entry.value + "'; using normal");
        } else if (name == "paths.cachedir") {
            if (!entry.value.empty() && lower(entry.value) != "auto") settings.cacheDir = diag::wide(entry.value);
        } else notes.push_back("d4r.ini: [" + entry.section + "] " + entry.key + " is not a Windows setting; ignored");
    }
    return settings;
}

// "kind sha256 version status" lines; kind is dlss or ngx.
inline std::vector<Pin> parse_pins(const std::string& text) {
    std::vector<Pin> pins; std::istringstream lines(text); std::string line;
    while (std::getline(lines, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line); Pin pin;
        if (!(fields >> pin.kind >> pin.sha256 >> pin.version >> pin.status) || pin.sha256.size() != 64 ||
            (pin.kind != "dlss" && pin.kind != "ngx")) throw std::runtime_error("dll-pins.txt is damaged; extract the d4r ZIP again");
        pin.sha256 = upper(pin.sha256); pins.push_back(pin);
    }
    return pins;
}

inline std::string sha256(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    BCRYPT_ALG_HANDLE algorithm = nullptr; BCRYPT_HASH_HANDLE hash = nullptr;
    struct Cleanup { BCRYPT_ALG_HANDLE& a; BCRYPT_HASH_HANDLE& h;
        ~Cleanup() { if (h) BCryptDestroyHash(h); if (a) BCryptCloseAlgorithmProvider(a, 0); } } cleanup{algorithm, hash};
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
        BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) throw std::runtime_error("SHA256 unavailable");
    std::vector<char> buffer(1 << 20);
    while (file) {
        file.read(buffer.data(), std::streamsize(buffer.size()));
        if (file.gcount() > 0 && BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer.data()), ULONG(file.gcount()), 0) < 0)
            throw std::runtime_error("SHA256 failed");
    }
    unsigned char digest[32]{};
    if (BCryptFinishHash(hash, digest, sizeof(digest), 0) < 0) throw std::runtime_error("SHA256 failed");
    static const char hex[] = "0123456789ABCDEF"; std::string text;
    for (unsigned char byte : digest) { text += hex[byte >> 4]; text += hex[byte & 15]; }
    return text;
}
inline std::string file_version(const std::filesystem::path& path) {
    DWORD ignored = 0; const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &ignored);
    std::vector<char> data(size);
    VS_FIXEDFILEINFO* info = nullptr; UINT length = 0;
    if (!size || !GetFileVersionInfoW(path.c_str(), 0, size, data.data()) ||
        !VerQueryValueW(data.data(), L"\\", reinterpret_cast<void**>(&info), &length) || !info) return "unknown";
    return std::to_string(HIWORD(info->dwFileVersionMS)) + "." + std::to_string(LOWORD(info->dwFileVersionMS)) + "." +
        std::to_string(HIWORD(info->dwFileVersionLS)) + "." + std::to_string(LOWORD(info->dwFileVersionLS));
}
// Exact identity check with the fix spelled out: what was found, what is needed.
inline void verify_dll(const std::filesystem::path& dir, const std::vector<Pin>& pins, const char* kind,
    const wchar_t* relative, const char* label, std::vector<std::string>& notes) {
    const auto path = dir / relative;
    std::string wanted;
    for (const auto& pin : pins) if (pin.kind == kind) wanted += (wanted.empty() ? "" : " or ") + pin.version;
    const auto hash = sha256(path);
    if (hash.empty())
        throw std::runtime_error(std::string(label) + " is missing. Copy NVIDIA's " + label + " (version " + wanted +
            ") to d4r\\" + diag::utf8(relative) + " and restart the game.");
    for (const auto& pin : pins) if (pin.kind == kind && pin.sha256 == hash) {
        if (pin.status != "validated")
            notes.push_back(std::string(label) + " " + pin.version + " is accepted but not yet validated on Windows");
        return;
    }
    throw std::runtime_error("Wrong " + std::string(label) + ": you have version " + file_version(path) +
        " (or a modified file); d4r on Windows needs exactly " + wanted + ". Replace d4r\\" + diag::utf8(relative) +
        " and restart the game.");
}

// The installer used to check these before touching the game; a drag-in
// install checks them at startup instead.
inline void verify_code_objects(const std::filesystem::path& dir) {
    const std::string arch = diag::HipApi::target_arch();
    std::vector<std::filesystem::path> objects{dir / (L"pixel_convert_" + diag::wide(arch) + L".hsaco")};
    const auto native = dir / L"native" / diag::wide(arch);
    if (!std::filesystem::is_directory(native)) throw std::runtime_error("d4r\\native\\" + arch + " is missing; extract the d4r ZIP again");
    for (const auto& entry : std::filesystem::directory_iterator(native))
        if (entry.path().extension() == L".hsaco") objects.push_back(entry.path());
    for (const auto& object : objects) {
        try { diag::require_code_object_target(object, arch); }
        catch (const std::exception&) {
            throw std::runtime_error(diag::utf8(object.filename().c_str()) + " is not built for " + arch +
                " or is damaged; extract the d4r ZIP again (do not rename or mix files from other GPUs)");
        }
    }
}
inline void reject_translation_layer() {
    wchar_t exe[32768]{}; GetModuleFileNameW(nullptr, exe, 32768);
    const auto proxy = std::filesystem::path(exe).parent_path() / L"d3d12.dll";
    std::error_code error;
    const auto size = std::filesystem::file_size(proxy, error);
    if (error || size > 64ull << 20) return;
    auto text = lower(read_text(proxy));
    if (text.find("vkd3d") != std::string::npos || text.find("wined3d") != std::string::npos)
        throw std::runtime_error("the game folder contains a d3d12.dll from vkd3d-proton/Wine (left over from a Linux "
            "install?). Move d3d12.dll and d3d12core.dll out of the game folder; d4r on Windows uses native D3D12");
}

// Update the OS block (ZLUDA/Rust, later DLLs) and this CRT's copy (getenv).
inline void set_variable(const wchar_t* name, const std::optional<std::wstring>& value) {
    SetEnvironmentVariableW(name, value ? value->c_str() : nullptr);
    _wputenv_s(name, value ? value->c_str() : L"");
}
// Only when the game has no console/pipe: never steal a launcher's streams.
inline bool unattached(DWORD stream) {
    const HANDLE current = GetStdHandle(stream);
    return !current || current == INVALID_HANDLE_VALUE;
}
inline void redirect(FILE* crt, DWORD stream, const std::filesystem::path& path) {
    FILE* reopened = nullptr;
    if (_wfreopen_s(&reopened, path.c_str(), L"a", crt) || !reopened) return;
    std::setvbuf(crt, nullptr, _IONBF, 0);
    SetStdHandle(stream, reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(crt))));
}

inline void apply(const std::filesystem::path& dir, const Settings& settings) {
    std::filesystem::path cache = settings.cacheDir;
    if (cache.empty()) {
        const wchar_t* localAppData = _wgetenv(L"LOCALAPPDATA");
        if (!localAppData || !*localAppData) throw std::runtime_error("LOCALAPPDATA is not set; set [Paths] CacheDir in d4r.ini");
        cache = std::filesystem::path(localAppData) / L"d4r" / L"cache";
    } else if (cache.is_relative()) cache = dir / cache;
    std::filesystem::create_directories(cache);
    const auto p = [&](const std::wstring& relative) { return std::optional<std::wstring>((dir / relative).wstring()); };
    const std::optional<std::wstring> on = std::wstring(L"1"), off = std::wstring(L"0"), unset;
    const wchar_t* path = _wgetenv(L"PATH");
    // Same codegen and paths as windows-game.ps1; only diagnostics differ.
    const std::pair<const wchar_t*, std::optional<std::wstring>> values[] = {
        {L"D4R_HIP_ROOT", p(L"hip")}, {L"HIP_PATH", p(L"hip")},
        {L"D4R_NVCUDA_DLL", p(L"zluda\\nvcuda.dll")}, {L"ZLUDA_CUDA_LIB", p(L"zluda\\nvcuda.dll")},
        {L"D4R_NVAPI_DLL", p(L"nvapi\\" + diag::wide(diag::HipApi::target_arch()) + L"\\nvapi64.dll")}, {L"D4R_NVAPI_BACKEND", p(L"zluda\\nvapi64.dll")},
        {L"D4R_NGX_CORE", p(L"ngx\\_nvngx.dll")}, {L"D4R_DLSS_DLL", p(L"nvngx_dlss.dll")},
        {L"D4R_FORMAT_MODULE", p(L"pixel_convert_" + diag::wide(diag::HipApi::target_arch()) + L".hsaco")},
        {L"D4R_ZLUDA_NATIVE_DIR", p(L"native\\" + diag::wide(diag::HipApi::target_arch()))},{L"D4R_D3D12_COMMAND_BACKEND", on},
        {L"D4R_ZLUDA_WMMA", on}, {L"D4R_ZLUDA_WMMA_FP8", on}, {L"D4R_ZLUDA_WMMA_FP8_NATIVE", off},
        {L"D4R_ZLUDA_WMMA_F16_REFERENCE", on},
        {L"D4R_ZLUDA_IGNORE_DENORMAL", unset}, {L"D4R_ZLUDA_FAST_MATH", unset}, {L"D4R_ZLUDA_WMMA_F32ACC", unset},
        {L"D4R_ZLUDA_WAVE64", unset}, {L"D4R_ZLUDA_WMMA_LAYOUT", unset},
        // Per-launch kernel lines are diagnostics; ZLUDA excludes this flag from its cache key.
        {L"D4R_ZLUDA_VERBOSE", settings.verbose ? on : unset},
        {L"D4R_QUIET_API", settings.verbose ? unset : on},
        {L"D4R_VALIDATE_OUTPUT", settings.validateOutput ? on : unset},
        {L"D4R_ASYNC_INTEROP", settings.asyncInterop ? on : unset},
        {L"D4R_PROFILE_STAGES", unset}, {L"D4R_ZLUDA_PROFILE", unset}, {L"D4R_ZLUDA_PROFILE_DEFERRED", unset},
        {L"D4R_ZLUDA_PROFILE_ALLOW_LEGACY", unset}, {L"D4R_ZLUDA_PROFILE_EVERY", unset},
        {L"D4R_PROFILE_COMMAND_HOOKS", unset}, {L"D4R_ZLUDA_PROFILE_API", unset}, {L"D4R_PROFILE_GPU_BOUNDARY", unset},
        {L"D4R_DISABLE_INTEROP_LIST_CACHE", unset}, {L"D4R_BATCH_INPUT_COPIES", unset}, {L"D4R_INTEROP_VERIFY", unset},
        {L"D4R_DIAG_DIR", unset}, {L"ZLUDA_LOG_DIR", unset},
        {L"ZLUDA_CACHE_DIR", std::optional<std::wstring>(cache.wstring())},
        {L"PATH", std::optional<std::wstring>((dir / L"hip\\bin").wstring() + L";" + (dir / L"zluda").wstring() +
            (path ? std::wstring(L";") + path : std::wstring()))},
    };
    for (const auto& [name, value] : values) set_variable(name, value);
    const wchar_t* wgp = _wgetenv(L"D4R_ZLUDA_WGP");
    if (wgp && std::wstring(wgp) != L"1") set_variable(L"D4R_ZLUDA_WGP", unset);
}

// Called before any path variable is read. A failure leaves DLSS unavailable
// (the game keeps its own upscaler) and is written to d4r\d4r_nvngx.log.
inline void ensure_environment() {
    static std::once_flag once; static std::string failure;
    std::call_once(once, [] {
        try {
            if (const char* root = std::getenv("D4R_HIP_ROOT"); root && *root) return; // Launcher/diagnostic run.
            const auto dir = module_directory();
            if (!std::filesystem::exists(dir / L"d4r.ini")) return; // Developer layout: keep the original requirement.
            auto& current = state(); current.dir = dir;
            const auto log = dir / L"d4r_nvngx.log";
            if (unattached(STD_ERROR_HANDLE)) {
                { std::ofstream truncate(log, std::ios::trunc); } // Rewritten at every launch, as on Linux.
                redirect(stderr, STD_ERROR_HANDLE, log);
            }
            const auto settings = read_settings(parse_ini(read_text(dir / L"d4r.ini")), current.notes);
            if (settings.verbose && unattached(STD_OUTPUT_HANDLE)) redirect(stdout, STD_OUTPUT_HANDLE, log);
            const auto pins = parse_pins(read_text(dir / L"dll-pins.txt"));
            verify_dll(dir, pins, "dlss", L"nvngx_dlss.dll", "nvngx_dlss.dll", current.notes);
            verify_dll(dir, pins, "ngx", L"ngx\\_nvngx.dll", "_nvngx.dll", current.notes);
            reject_translation_layer();
            verify_code_objects(dir);
            apply(dir, settings);
            current.active = true; current.preset = settings.preset;
            std::fprintf(stderr, "d4r: player mode, model=%s validate=%u async=%u log=%s, architecture=%s\n",
                settings.preset == 13 ? "M" : "K", unsigned(settings.validateOutput), unsigned(settings.asyncInterop),
                settings.verbose ? "verbose" : "normal", diag::HipApi::target_arch());
        } catch (const std::exception& error) { failure = error.what(); }
        for (const auto& note : state().notes) std::fprintf(stderr, "d4r: %s\n", note.c_str());
        if (!failure.empty()) std::fprintf(stderr, "d4r: DLSS disabled: %s\n", failure.c_str());
    });
    if (!failure.empty()) throw std::runtime_error(failure);
}

// [DLSS] Model forces one render preset for every quality mode, as the Linux
// shim does. Applied at create and evaluate so a recreated feature cannot fall
// back to the game's preset. Not used by launcher runs (OptiScaler sets it).
inline void apply_preset(void* parameters) {
    const auto& current = state();
    if (!current.active || !current.preset || !parameters) return;
    for (const char* name : {"DLSS.Hint.Render.Preset.DLAA", "DLSS.Hint.Render.Preset.Quality", "DLSS.Hint.Render.Preset.Balanced",
         "DLSS.Hint.Render.Preset.Performance", "DLSS.Hint.Render.Preset.UltraPerformance", "DLSS.Hint.Render.Preset.UltraQuality"})
        d4r_ngx_set_uint(parameters, name, current.preset);
}
}
