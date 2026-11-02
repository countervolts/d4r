#pragma once
#include <cwctype>
#include <string>

// GE-Proton redirects DXGI in background utilities too. They share the ASI
// directory with the game but must never install files or show setup dialogs.
inline bool d4r_asi_game_process(std::wstring executable)
{
    for (auto& c : executable) { c = std::towlower(c); if (c == L'/') c = L'\\'; }
    const auto slash = executable.find_last_of(L'\\');
    const std::wstring name = executable.substr(slash == std::wstring::npos ? 0 : slash + 1);
    for (const wchar_t* utility : {L"xalia.exe", L"steam.exe", L"d4r-setup.exe", L"start_protected_game.exe",
                                   L"crashreportclient.exe", L"unrealcefsubprocess.exe", L"unitycrashhandler64.exe",
                                   L"unitycrashhandler32.exe", L"winedevice.exe", L"services.exe", L"explorer.exe",
                                   L"rundll32.exe", L"conhost.exe", L"wineboot.exe", L"wineserver.exe"})
        if (name == utility) return false;
    return executable.find(L"\\windows\\system32\\") == std::wstring::npos &&
           executable.find(L"\\windows\\syswow64\\") == std::wstring::npos;
}
