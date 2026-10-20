// Native first-run setup, launched by InitializeASI. No Python, shell or system install.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "d4r_asi_process.h"

namespace fs = std::filesystem;

static std::string utf8(const std::wstring& value)
{
    int length = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    std::string result(length, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), result.data(), length, nullptr, nullptr);
    return result;
}

static std::string lower(std::string value)
{
    for (char& c : value)
        if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return value;
}

static std::string trim(const std::string& value)
{
    auto start = value.find_first_not_of(" \t\r\n");
    return start == std::string::npos ? "" : value.substr(start, value.find_last_not_of(" \t\r\n") - start + 1);
}

static std::string read(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read " + utf8(path.wstring()));
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

static void write(const fs::path& path, const std::string& data)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
    file.close();
    if (!file) throw std::runtime_error("Cannot write " + utf8(path.wstring()));
}

static std::wstring unique_suffix()
{
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    return L"." + std::to_wstring(GetCurrentProcessId()) + L"." + std::to_wstring(
        (static_cast<unsigned long long>(now.dwHighDateTime) << 32) | now.dwLowDateTime);
}

struct Replacement
{
    fs::path staged, destination, backup;
};

static void commit(std::vector<Replacement>& files)
{
    // Prepare all backups before replacing any file. Reject aliases/directories.
    for (auto& file : files) {
        if (fs::is_symlink(file.destination) || (fs::exists(file.destination) && !fs::is_regular_file(file.destination)))
            throw std::runtime_error("Destination must be a regular file: " + utf8(file.destination.wstring()));
        if (fs::exists(file.destination)) {
            file.backup = file.destination.wstring() + L".d4r-backup" + unique_suffix();
            if (!CopyFileW(file.destination.c_str(), file.backup.c_str(), TRUE))
                throw std::runtime_error("Cannot back up " + utf8(file.destination.wstring()));
        }
    }
    size_t applied = 0;
    for (; applied < files.size(); ++applied) {
        auto& file = files[applied];
        if (MoveFileExW(file.staged.c_str(), file.destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            continue;
        bool restored = true;
        while (applied > 0) {
            auto& previous = files[--applied];
            if (previous.backup.empty()) restored &= DeleteFileW(previous.destination.c_str()) != FALSE;
            else restored &= CopyFileW(previous.backup.c_str(), previous.destination.c_str(), FALSE) != FALSE;
        }
        throw std::runtime_error("Cannot install " + utf8(file.destination.wstring()) +
                                 (restored ? "; earlier changes were restored" : "; restore the .d4r-backup files before retrying"));
    }
}

static bool same_file(const fs::path& source, const fs::path& destination)
{
    return fs::is_regular_file(destination) && fs::file_size(source) == fs::file_size(destination) && read(source) == read(destination);
}

static std::string configure(std::string source, const fs::path& plugin)
{
    const std::string bom = source.starts_with("\xef\xbb\xbf") ? "\xef\xbb\xbf" : "";
    if (!bom.empty()) source.erase(0, 3);
    std::string newline = source.find("\r\n") == std::string::npos ? "\n" : "\r\n";
    std::vector<std::string> lines;
    for (size_t pos = 0; pos < source.size();) {
        auto end = source.find('\n', pos);
        end = end == std::string::npos ? source.size() : end + 1;
        lines.push_back(source.substr(pos, end - pos));
        pos = end;
    }
    struct Setting { std::string section, key, value; };
    const std::array<Setting, 10> settings = {{{"Upscalers", "Dx12Upscaler", "dlss"}, {"DLSS", "Enabled", "true"},
        {"Inputs", "EnableDlssInputs", "true"}, {"Hooks", "HookOriginalNvngxOnly", "true"},
        {"Hooks", "UseNtdllHooks", "false"},
        {"Spoofing", "Dxgi", "true"},
        {"FSR", "FsrAgilitySDKUpgrade", "false"},
        {"Plugins", "LoadAsiPlugins", "true"}, {"Plugins", "Path", utf8(plugin.parent_path().wstring())},
        {"Libraries", "NvngxPath", utf8(plugin.wstring())}}};
    for (const auto& setting : settings) {
        size_t start = lines.size(), end = lines.size(), match = lines.size();
        for (size_t i = 0; i < lines.size(); ++i)
            if (lower(trim(lines[i])) == "[" + lower(setting.section) + "]") {
                if (start != lines.size()) throw std::runtime_error("Duplicate [" + setting.section + "] section");
                start = i;
            }
        if (start == lines.size()) {
            if (!lines.empty() && !lines.back().ends_with('\n')) lines.back() += newline;
            lines.push_back("[" + setting.section + "]" + newline);
            lines.push_back(setting.key + "=" + setting.value + newline);
            continue;
        }
        for (size_t i = start + 1; i < lines.size(); ++i) {
            auto line = trim(lines[i]);
            if (line.starts_with('[')) { end = i; break; }
            auto equals = line.find('=');
            if (equals != std::string::npos && lower(trim(line.substr(0, equals))) == lower(setting.key)) {
                if (match != lines.size()) throw std::runtime_error("Duplicate " + setting.key + " setting");
                match = i;
            }
        }
        if (match != lines.size()) {
            std::string ending = lines[match].ends_with("\r\n") ? "\r\n" : lines[match].ends_with('\n') ? "\n" : "";
            // Keep original spelling, spacing and comments if the active value already matches.
            auto line = trim(lines[match]);
            auto value = trim(line.substr(line.find('=') + 1));
            if (lower(value) != lower(setting.value))
                lines[match] = setting.key + "=" + setting.value + ending;
        } else {
            if (end && !lines[end - 1].ends_with('\n')) lines[end - 1] += newline;
            lines.insert(lines.begin() + end, setting.key + "=" + setting.value + newline);
        }
    }
    std::string result = bom;
    for (const auto& line : lines) result += line;
    return result;
}

int wmain(int argc, wchar_t** argv)
{
    if (argc < 3 || argc > 5) return 2;
    fs::path game = fs::absolute(argv[1]), plugin = fs::absolute(argv[2]);
    fs::path runtime = plugin.parent_path() / "d4r", config = game / "OptiScaler.ini";
    // Also reject background callers from older ASIs before taking locks or
    // touching files. A copied helper must not continue an xalia installation.
    if (argc >= 4) {
        HANDLE caller = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, wcstoul(argv[3], nullptr, 10));
        if (caller) {
            wchar_t executable[32768] = {}; DWORD size = 32768;
            bool queried = QueryFullProcessImageNameW(caller, 0, executable, &size) != FALSE;
            CloseHandle(caller);
            if (!queried || !d4r_asi_game_process(executable)) return 0;
        }
    }
    // The ASI supplies the active loader's location when available. Older ASIs
    // still work with the usual plugins/ folder under OptiScaler's directory.
    wchar_t quiet[2] = {};
    bool silent = GetEnvironmentVariableW(L"D4R_ASI_SETUP_QUIET", quiet, 2) && quiet[0] == L'1';
    // One setup process per game, even if two copies/launches call InitializeASI.
    unsigned long long key = 14695981039346656037ull;
    for (auto c : runtime.wstring()) { key ^= static_cast<unsigned short>(std::towlower(c)); key *= 1099511628211ull; }
    HANDLE mutex = CreateMutexW(nullptr, FALSE, (L"Local\\d4r-asi-setup-" + std::to_wstring(key)).c_str());
    if (!mutex) return 3;
    DWORD wait = WaitForSingleObject(mutex, 0);
    if (wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED) { CloseHandle(mutex); return 0; }
    struct Lock { HANDLE handle; ~Lock() { ReleaseMutex(handle); CloseHandle(handle); } } lock{mutex};
    fs::path temporary = game / (L".d4r-setup" + unique_suffix());
    try {
        if (argc == 5 && *argv[4]) config = fs::absolute(argv[4]);
        else if (fs::is_regular_file(plugin.parent_path().parent_path() / "OptiScaler.ini"))
            config = plugin.parent_path().parent_path() / "OptiScaler.ini";
        if (!fs::is_regular_file(plugin))
            throw std::runtime_error("ASI file not found: " + utf8(plugin.wstring()));
        if (!fs::is_regular_file(runtime / "d4r.ini"))
            throw std::runtime_error("Companion d4r.ini not found: " + utf8((runtime / "d4r.ini").wstring()));
        if (!fs::is_regular_file(config))
            throw std::runtime_error("OptiScaler.ini not found: " + utf8(config.wstring()) +
                                     "; keep the INI beside the OptiScaler loader");
        const std::string original = read(config);
        const std::string updated = configure(original, plugin); // reject ambiguity before mutations
        if (!fs::is_regular_file(runtime / "nvngx_dlss.dll") || !fs::is_regular_file(runtime / "ngx/_nvngx.dll"))
            throw std::runtime_error("Run bash install.sh inside the plugin's d4r folder to download and verify the NVIDIA DLLs, then launch the game again");
        for (const char* name : {"d3d12.dll", "d4r12core.dll"})
            if (!fs::is_regular_file(runtime / name))
                throw std::runtime_error("The bundled D3D12 files are missing; extract the complete ASI archive");
        if (fs::is_regular_file(game / "d3d12.dll") &&
            lower(read(game / "d3d12.dll")).find("optiscaler.dll") != std::string::npos)
            throw std::runtime_error("OptiScaler is installed as d3d12.dll. Use a supported loader such as dxgi.dll or winmm.dll before installing d4r, so setup can preserve your OptiScaler loader");
        bool pairChanged = !same_file(runtime / "d3d12.dll", game / "d3d12.dll") ||
                           !same_file(runtime / "d4r12core.dll", game / "d4r12core.dll");
        if (original == updated && !pairChanged) return 0;
        // A running game may hold these DLLs mapped or save its old INI on exit.
        // The ASI gives us its Windows PID; finish only after that game closes.
        if (argc >= 4) {
            wchar_t* end = nullptr;
            unsigned long pid = wcstoul(argv[3], &end, 10);
            if (!pid || *end) throw std::runtime_error("Invalid game process ID");
            HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, pid);
            if (!parent && GetLastError() != ERROR_INVALID_PARAMETER)
                throw std::runtime_error("Cannot wait for the game to close");
            if (parent) {
                write(runtime / "installation-status.txt", "Close the game to finish d4r setup. Files will be backed up and installed automatically after it exits.\r\n");
                if (!silent) MessageBoxW(nullptr, L"Close the game to finish d4r setup.\n\nSetup will back up your existing D3D12 files and OptiScaler settings after the game exits, then show a ready message.",
                                         L"d4r setup", MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND);
                DWORD waited = WaitForSingleObject(parent, INFINITE);
                CloseHandle(parent);
                if (waited != WAIT_OBJECT_0) throw std::runtime_error("Could not wait for game shutdown");
            }
        }
        fs::create_directories(temporary);
        // The loader's INI can be in the prefix on another filesystem. Stage it
        // beside its own destination, so its replacement remains atomic too.
        fs::path configStage = config.parent_path() / (L".d4r-config" + unique_suffix());
        struct Cleanup {
            fs::path path, config;
            ~Cleanup() { std::error_code error; fs::remove_all(path, error); fs::remove(config, error); }
        } cleanup{temporary, configStage};
        // Merge the latest settings in case OptiScaler changed them during setup.
        const std::string current = read(config);
        const std::string finalConfig = configure(current, plugin);
        std::vector<Replacement> replacements;
        for (const char* name : {"d3d12.dll", "d4r12core.dll"}) {
            fs::path source = runtime / name, destination = game / name;
            if (!same_file(source, destination)) {
                fs::copy_file(source, temporary / name);
                replacements.push_back({temporary / name, destination, {}});
            }
        }
        if (current != finalConfig) {
            write(configStage, finalConfig);
            replacements.push_back({configStage, config, {}});
        }
        commit(replacements);
        write(runtime / "installation-status.txt", "d4r setup complete. Launch the game again to activate DLSS.\r\n");
        if (!silent) MessageBoxW(nullptr, L"d4r is ready. Launch the game again to activate DLSS.\n\nChanged D3D12 files and OptiScaler settings have been backed up.",
                                 L"d4r ready", MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND);
        return 0;
    } catch (const std::exception& error) {
        std::string message = "d4r setup failed: " + std::string(error.what()) + "\r\nLaunch the game again to retry.\r\n";
        try { write(runtime / "installation-status.txt", message); } catch (...) {}
        std::fprintf(stderr, "%s", message.c_str());
        if (!silent) {
            int count = MultiByteToWideChar(CP_UTF8, 0, message.data(), static_cast<int>(message.size()), nullptr, 0);
            std::wstring wide(count, '\0');
            MultiByteToWideChar(CP_UTF8, 0, message.data(), static_cast<int>(message.size()), wide.data(), count);
            MessageBoxW(nullptr, wide.c_str(), L"d4r setup", MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
        }
        return 1;
    }
}
