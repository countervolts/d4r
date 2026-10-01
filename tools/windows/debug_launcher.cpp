// Out-of-process crash capture: unaffected by the frontend's loader hooks.
#include "diagnostic.h"
#include <map>
#include <fstream>
#include <tlhelp32.h>

namespace {
std::wstring quote(const std::wstring& value) {
    std::wstring result = L"\""; unsigned slashes = 0;
    for (wchar_t c : value) {
        if (c == L'\\') { ++slashes; continue; }
        result.append(slashes * (c == L'\"' ? 2 : 1), L'\\'); slashes = 0;
        if (c == L'\"') result += L'\\';
        result += c;
    }
    result.append(slashes * 2, L'\\'); return result + L'\"';
}
std::string file_path(HANDLE file) {
    wchar_t path[32768]{};
    const auto length = GetFinalPathNameByHandleW(file, path, 32768, FILE_NAME_NORMALIZED);
    return length && length < 32768 ? d4r::diag::utf8(path) : "unknown";
}
struct Module { std::string path; uint32_t size; };
uint32_t module_size(HANDLE process, uintptr_t base) {
    IMAGE_DOS_HEADER dos{}; IMAGE_NT_HEADERS64 nt{}; SIZE_T read = 0;
    if (!ReadProcessMemory(process, reinterpret_cast<void*>(base), &dos, sizeof(dos), &read) ||
        dos.e_magic != IMAGE_DOS_SIGNATURE || dos.e_lfanew < 0 || dos.e_lfanew > 65536 ||
        !ReadProcessMemory(process, reinterpret_cast<void*>(base + dos.e_lfanew), &nt, sizeof(nt), &read) ||
        nt.Signature != IMAGE_NT_SIGNATURE) return 0;
    return nt.OptionalHeader.SizeOfImage;
}
}
int main(int argc, char** argv) {
    try {
        const bool debug = argc < 4 || std::strcmp(argv[3], "--no-debug");
        const int separator = debug ? 3 : 4;
        if (argc <= separator + 1 || std::strcmp(argv[1], "--output-directory") || std::strcmp(argv[separator], "--"))
            throw std::runtime_error("Usage: d4r_debug_launcher --output-directory PATH [--no-debug] -- EXE [arguments]");
        const std::filesystem::path directory(d4r::diag::wide(argv[2]));
        std::filesystem::create_directories(directory);
        std::ofstream log(directory / L"debugger.log", std::ios::app);
        const auto executable = d4r::diag::wide(argv[separator + 1]);
        std::wstring command;
        for (int i = separator + 1; i < argc; ++i) { if (i > separator + 1) command += L' '; command += quote(d4r::diag::wide(argv[i])); }
        STARTUPINFOW startup{sizeof(startup)}; PROCESS_INFORMATION child{};
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        HANDLE job = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
            throw std::runtime_error("Cannot create diagnostic process lifetime job");
        if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
            (debug ? DEBUG_ONLY_THIS_PROCESS : 0) | CREATE_SUSPENDED | CREATE_NO_WINDOW, nullptr, nullptr, &startup, &child))
            throw std::runtime_error("CreateProcess debugger error=" + std::to_string(GetLastError()));
        if (!AssignProcessToJobObject(job, child.hProcess)) {
            TerminateProcess(child.hProcess, 1); CloseHandle(job);
            throw std::runtime_error("AssignProcessToJobObject debugger failed");
        }
        log << "LAUNCH pid=" << child.dwProcessId << " attached_debugger=" << debug << std::endl;
        if (ResumeThread(child.hThread) == DWORD(-1)) {
            CloseHandle(job); throw std::runtime_error("Cannot resume diagnostic child");
        }
        if (!debug) {
            // Retain bounded process ownership and raw inherited output without
            // debugger stops on DLL/thread/exception/debug-string events.
            std::map<uintptr_t, Module> inventory;
            while (WaitForSingleObject(child.hProcess, 1000) == WAIT_TIMEOUT) {
                HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, child.dwProcessId);
                if (snapshot == INVALID_HANDLE_VALUE) continue;
                MODULEENTRY32W module{sizeof(module)};
                if (Module32FirstW(snapshot, &module)) do {
                    const auto base = reinterpret_cast<uintptr_t>(module.modBaseAddr);
                    if (inventory.count(base)) continue;
                    inventory[base] = {d4r::diag::utf8(module.szExePath), module.modBaseSize};
                    log << "LOAD base=0x" << std::hex << base << " size=0x" << module.modBaseSize
                        << " path=" << inventory[base].path << '\n';
                } while (Module32NextW(snapshot, &module));
                CloseHandle(snapshot); log.flush();
            }
            DWORD result = 1;
            GetExitCodeProcess(child.hProcess, &result);
            log << "EXIT code=0x" << std::hex << result << std::endl;
            CloseHandle(child.hThread); CloseHandle(child.hProcess); CloseHandle(job);
            return static_cast<int>(result);
        }
        std::map<uintptr_t, Module> modules;
        bool initialBreakpoint = true, dumped = false; DWORD exitCode = 1;
        auto add_module = [&](void* base, HANDLE file) {
            const auto address = reinterpret_cast<uintptr_t>(base);
            modules[address] = {file_path(file), module_size(child.hProcess, address)};
            log << "LOAD base=0x" << std::hex << address << " size=0x" << modules[address].size << " path=" << modules[address].path << std::endl;
            if (file && file != INVALID_HANDLE_VALUE) CloseHandle(file);
        };
        auto address_info = [&](uintptr_t address) {
            auto it = modules.upper_bound(address);
            if (it != modules.begin()) {
                --it;
                if (address - it->first < it->second.size)
                    log << " module=" << it->second.path << "+0x" << std::hex << address - it->first;
            }
        };
        DEBUG_EVENT event{};
        uint64_t debugEvents = 0, debugStrings = 0, exceptions = 0;
        ULONGLONG lastSample = GetTickCount64();
        while (WaitForDebugEvent(&event, INFINITE)) {
            ++debugEvents;
            if (event.dwDebugEventCode == OUTPUT_DEBUG_STRING_EVENT) ++debugStrings;
            if (event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT) ++exceptions;
            const auto now = GetTickCount64();
            if (now - lastSample >= 5000) {
                log << "DEBUG_EVENTS events=" << std::dec << debugEvents << " debug_strings=" << debugStrings << " exceptions=" << exceptions << std::endl;
                lastSample = now;
            }
            DWORD disposition = DBG_CONTINUE;
            if (event.dwDebugEventCode == CREATE_PROCESS_DEBUG_EVENT) {
                add_module(event.u.CreateProcessInfo.lpBaseOfImage, event.u.CreateProcessInfo.hFile);
                CloseHandle(event.u.CreateProcessInfo.hProcess); CloseHandle(event.u.CreateProcessInfo.hThread);
            } else if (event.dwDebugEventCode == CREATE_THREAD_DEBUG_EVENT) CloseHandle(event.u.CreateThread.hThread);
            else if (event.dwDebugEventCode == LOAD_DLL_DEBUG_EVENT) add_module(event.u.LoadDll.lpBaseOfDll, event.u.LoadDll.hFile);
            else if (event.dwDebugEventCode == UNLOAD_DLL_DEBUG_EVENT) {
                const auto base = reinterpret_cast<uintptr_t>(event.u.UnloadDll.lpBaseOfDll);
                log << "UNLOAD base=0x" << std::hex << base << std::endl; modules.erase(base);
            } else if (event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT) {
                const auto& exception = event.u.Exception;
                const DWORD code = exception.ExceptionRecord.ExceptionCode;
                log << "EXCEPTION code=0x" << std::hex << code << " firstChance=" << exception.dwFirstChance
                    << " address=0x" << reinterpret_cast<uintptr_t>(exception.ExceptionRecord.ExceptionAddress) << std::endl;
                disposition = DBG_EXCEPTION_NOT_HANDLED;
                if (initialBreakpoint && code == EXCEPTION_BREAKPOINT) { initialBreakpoint = false; disposition = DBG_CONTINUE; }
                if (!dumped && (!exception.dwFirstChance || code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_ILLEGAL_INSTRUCTION || code == EXCEPTION_STACK_OVERFLOW || code == EXCEPTION_IN_PAGE_ERROR)) {
                    dumped = true;
                    log << "FAULT code=0x" << std::hex << code << " address=0x" << reinterpret_cast<uintptr_t>(exception.ExceptionRecord.ExceptionAddress)
                        << " thread=" << std::dec << event.dwThreadId << " firstChance=" << exception.dwFirstChance;
                    address_info(reinterpret_cast<uintptr_t>(exception.ExceptionRecord.ExceptionAddress)); log << std::endl;
                    HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, event.dwThreadId);
                    CONTEXT context{}; context.ContextFlags = CONTEXT_ALL;
                    if (thread && GetThreadContext(thread, &context)) {
                        log << "CONTEXT rip=0x" << std::hex << context.Rip << " rsp=0x" << context.Rsp << " rcx=0x" << context.Rcx << " rdx=0x" << context.Rdx << std::endl;
                        EXCEPTION_RECORD record = exception.ExceptionRecord;
                        EXCEPTION_POINTERS pointers{&record, &context};
                        MINIDUMP_EXCEPTION_INFORMATION info{event.dwThreadId, &pointers, FALSE};
                        const auto path = directory / (L"crash-" + std::to_wstring(child.dwProcessId) + L".dmp");
                        HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
                        if (file != INVALID_HANDLE_VALUE) {
                            const BOOL ok = MiniDumpWriteDump(child.hProcess, child.dwProcessId, file,
                                static_cast<MINIDUMP_TYPE>(MiniDumpNormal | MiniDumpWithThreadInfo), &info, nullptr, nullptr);
                            log << "MINIDUMP success=" << ok << " error=" << (ok ? 0 : GetLastError()) << std::endl; CloseHandle(file);
                        }
                        uintptr_t stack[256]{}; SIZE_T read = 0;
                        if (ReadProcessMemory(child.hProcess, reinterpret_cast<void*>(context.Rsp), stack, sizeof(stack), &read))
                            for (unsigned i = 0; i < read / sizeof(uintptr_t); ++i) {
                                log << "STACK slot=" << std::dec << i << " address=0x" << std::hex << stack[i]; address_info(stack[i]); log << '\n';
                            }
                    }
                    if (thread) CloseHandle(thread); log.flush();
                }
            } else if (event.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT) {
                exitCode = event.u.ExitProcess.dwExitCode;
                log << "EXIT code=0x" << std::hex << exitCode << std::endl;
                ContinueDebugEvent(event.dwProcessId, event.dwThreadId, disposition); break;
            }
            ContinueDebugEvent(event.dwProcessId, event.dwThreadId, disposition);
        }
        CloseHandle(child.hThread); CloseHandle(child.hProcess); CloseHandle(job);
        return static_cast<int>(exitCode);
    } catch (const std::exception& error) { std::fprintf(stderr, "DEBUG_LAUNCH_FAILURE %s\n", error.what()); return 1; }
}
