#pragma once
#include <windows.h>
#include <tlhelp32.h>
#include <dbghelp.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace d4r::diag {
inline std::wstring wide(const std::string& s)
{
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(), -1, nullptr, 0);
    if (!n) throw std::runtime_error("Invalid UTF-8 path");
    std::wstring out(n, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(), -1, out.data(), n);
    out.pop_back();
    return out;
}
inline std::string utf8(const wchar_t* s)
{
    const int n = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
    std::string out(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, s, -1, out.data(), n, nullptr, nullptr);
    if (n) out.pop_back();
    return out;
}
inline void loaded_modules()
{
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) return;
    MODULEENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    if (Module32FirstW(snapshot, &entry)) do {
        DWORD unused = 0;
        DWORD size = GetFileVersionInfoSizeW(entry.szExePath, &unused);
        std::vector<unsigned char> bytes(size);
        VS_FIXEDFILEINFO* version = nullptr;
        UINT length = 0;
        std::printf("DLL path=%s", utf8(entry.szExePath).c_str());
        if (size && GetFileVersionInfoW(entry.szExePath, 0, size, bytes.data()) &&
            VerQueryValueW(bytes.data(), L"\\", reinterpret_cast<void**>(&version), &length) &&
            length >= sizeof(*version))
            std::printf(" version=%u.%u.%u.%u", HIWORD(version->dwFileVersionMS),
                LOWORD(version->dwFileVersionMS), HIWORD(version->dwFileVersionLS),
                LOWORD(version->dwFileVersionLS));
        std::printf("\n");
    } while (Module32NextW(snapshot, &entry));
    CloseHandle(snapshot);
}
inline LONG WINAPI unhandled(EXCEPTION_POINTERS* info)
{
    std::fprintf(stderr, "EXCEPTION code=0x%08lx address=%p\n",
        info->ExceptionRecord->ExceptionCode, info->ExceptionRecord->ExceptionAddress);
    if (info->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION &&
        info->ExceptionRecord->NumberParameters >= 2)
        std::fprintf(stderr, "EXCEPTION_ACCESS operation=%llu target=0x%llx\n",
            static_cast<unsigned long long>(info->ExceptionRecord->ExceptionInformation[0]),
            static_cast<unsigned long long>(info->ExceptionRecord->ExceptionInformation[1]));
    std::fprintf(stderr, "CONTEXT rip=0x%llx rsp=0x%llx rcx=0x%llx rdx=0x%llx r8=0x%llx r9=0x%llx\n",
        static_cast<unsigned long long>(info->ContextRecord->Rip),
        static_cast<unsigned long long>(info->ContextRecord->Rsp),
        static_cast<unsigned long long>(info->ContextRecord->Rcx),
        static_cast<unsigned long long>(info->ContextRecord->Rdx),
        static_cast<unsigned long long>(info->ContextRecord->R8),
        static_cast<unsigned long long>(info->ContextRecord->R9));
    loaded_modules();
    wchar_t directory[32768]{};
    const DWORD n = GetEnvironmentVariableW(L"D4R_DIAG_DIR", directory, 32768);
    if (n && n < 32768) {
        const std::wstring path = std::wstring(directory) + L"\\crash-" +
            std::to_wstring(GetCurrentProcessId()) + L".dmp";
        HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            MINIDUMP_EXCEPTION_INFORMATION exception{GetCurrentThreadId(), info, FALSE};
            const BOOL ok = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file,
                MiniDumpNormal, &exception, nullptr, nullptr);
            std::fprintf(stderr, "MINIDUMP success=%d error=%lu\n", ok, ok ? 0 : GetLastError());
            CloseHandle(file);
        }
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
inline LONG WINAPI first_chance(EXCEPTION_POINTERS* info)
{
    // NGX installs its own final exception filter. Capture a diagnostic dump
    // before that filter can replace ours; do not swallow or retry the fault.
    const auto code = info->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_ILLEGAL_INSTRUCTION) {
        static LONG captured = 0;
        if (InterlockedCompareExchange(&captured, 1, 0) == 0) (void)unhandled(info);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
inline void start()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    SetUnhandledExceptionFilter(unhandled);
    AddVectoredExceptionHandler(1, first_chance);
    std::printf("d4r native Windows diagnostic x64 pid=%lu\n", GetCurrentProcessId());
}
struct Args {
    std::string hip_root, cuda_dll, module, context = "primary";
    std::string stream_lifetime = "persistent";
    std::string interop_mode = "roundtrip";
    std::string pixel_profile = "baseline";
    std::string ngx_core, dlss_dll;
    std::string nvapi_dll;
    std::string ngx_abi = "driver";
    std::string ngx_mode = "init";
    unsigned preset = 11;
    std::string fixture_dir;
    std::string output_dir;
    std::string kernel_name = "enc1";
    unsigned iterations = 32;
    int device = -1;
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + key);
            const std::string value = argv[++i];
            if (key == "--hip-root") hip_root = value;
            else if (key == "--cuda-dll") cuda_dll = value;
            else if (key == "--module") module = value;
            else if (key == "--context") context = value;
            else if (key == "--stream-lifetime") stream_lifetime = value;
            else if (key == "--interop-mode") interop_mode = value;
            else if (key == "--pixel-profile") {
                if (value != "baseline" && value != "packed" && value != "unorm") throw std::runtime_error("pixel-profile must be baseline, packed or unorm");
                pixel_profile = value;
            }
            else if (key == "--ngx-core") ngx_core = value;
            else if (key == "--dlss-dll") dlss_dll = value;
            else if (key == "--nvapi-dll") nvapi_dll = value;
            else if (key == "--ngx-abi") ngx_abi = value;
            else if (key == "--ngx-mode") ngx_mode = value;
            else if (key == "--preset") {
                size_t end = 0;
                preset = static_cast<unsigned>(std::stoul(value, &end));
                if (end != value.size() || (preset != 5 && preset != 11 && preset != 13))
                    throw std::runtime_error("preset must be 5 (E), 11 (K) or 13 (M)");
            }
            else if (key == "--fixture-dir") fixture_dir = value;
            else if (key == "--output-dir") output_dir = value;
            else if (key == "--kernel-name") kernel_name = value;
            else if (key == "--iterations") {
                size_t end = 0;
                iterations = static_cast<unsigned>(std::stoul(value, &end));
                if (end != value.size() || iterations < 1 || iterations > 10000)
                    throw std::runtime_error("iterations must be 1..10000");
            } else if (key == "--device") {
                size_t end = 0;
                device = std::stoi(value, &end);
                if (end != value.size() || device < 0) throw std::runtime_error("Invalid device ordinal");
            } else throw std::runtime_error("Unknown option " + key);
        }
        if (hip_root.empty()) throw std::runtime_error("--hip-root is required");
    }
};
class Library {
    HMODULE handle_ = nullptr;
public:
    explicit Library(const std::filesystem::path& path) {
        if (!path.is_absolute()) throw std::runtime_error("DLL path must be absolute");
        handle_ = LoadLibraryExW(path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        if (!handle_) throw std::runtime_error("LoadLibrary " + utf8(path.c_str()) +
            " Win32=" + std::to_string(GetLastError()));
        std::printf("LOAD requested=%s\n", utf8(path.c_str()).c_str());
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    ~Library() { if (handle_) FreeLibrary(handle_); }
    template<class T> T symbol(const char* name) const {
        FARPROC raw = GetProcAddress(handle_, name);
        if (!raw) throw std::runtime_error(std::string("Missing export ") + name);
        static_assert(sizeof(T) == sizeof(raw));
        T out;
        std::memcpy(&out, &raw, sizeof(out));
        return out;
    }
};
class SearchDirectory {
    DLL_DIRECTORY_COOKIE cookie_ = nullptr;
public:
    explicit SearchDirectory(const std::filesystem::path& path) {
        if (!path.is_absolute()) throw std::runtime_error("HIP root must be absolute");
        cookie_ = AddDllDirectory(path.c_str());
        if (!cookie_) throw std::runtime_error("AddDllDirectory failed");
    }
    ~SearchDirectory() { if (cookie_) RemoveDllDirectory(cookie_); }
};
inline uint32_t pattern(uint32_t i, uint32_t seed) { return (i * 1664525u + seed) ^ 0xa5a55a5au; }
inline bool verify(const std::vector<uint32_t>& result, uint32_t count, uint32_t seed) {
    for (size_t i = 0; i < result.size(); ++i) {
        const uint32_t expected = i < count ? pattern(static_cast<uint32_t>(i), seed) : 0xdeadbeefu;
        if (result[i] != expected) {
            std::fprintf(stderr, "MISMATCH index=%zu expected=0x%08x actual=0x%08x\n", i, expected, result[i]);
            return false;
        }
    }
    return true;
}
}
