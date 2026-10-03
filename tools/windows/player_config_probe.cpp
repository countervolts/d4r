// Checks the drag-in d4r.ini parser and NVIDIA DLL pin decisions with
// fixtures. No GPU work, no NVIDIA binaries.
#include "player_config.h"
#include <functional>

namespace player = d4r::win::player;
namespace {
int failures = 0;
void expect(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "ok  " : "FAIL", what.c_str());
    if (!condition) ++failures;
}
bool has_note(const std::vector<std::string>& notes, const std::string& text) {
    for (const auto& note : notes) if (note.find(text) != std::string::npos) return true;
    return false;
}
player::Settings settings(const std::string& ini, std::vector<std::string>& notes) {
    notes.clear(); return player::read_settings(player::parse_ini(ini), notes);
}
std::string error_of(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception& error) { return error.what(); }
    return {};
}
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) { std::printf("usage: player_config_probe FIXTURE_DIR\n"); return 2; }
    const std::filesystem::path dir(argv[1]);
    std::vector<std::string> notes;

    auto s = settings("", notes);
    expect(s.preset == 11 && !s.asyncInterop && !s.validateOutput && !s.verbose && notes.empty(), "empty d4r.ini uses K, sync, no validation");
    s = settings("\xEF\xBB\xBF[DLSS]\nModel = M ; DLSS 4.5\n", notes);
    expect(s.preset == 13 && notes.empty(), "BOM and inline comment; Model = M");
    s = settings("[dlss]\nmodel = dlss4.5\n", notes);
    expect(s.preset == 13, "case-insensitive section/key and alias DLSS4.5");
    for (const char* model : {"E", "L", "game", "7"}) {
        s = settings(std::string("[DLSS]\nModel = ") + model + "\n", notes);
        expect(s.preset == 11 && has_note(notes, "not available on Windows"), std::string("Model = ") + model + " is rejected with a note and K");
    }
    s = settings("[Interop]\nAsyncInterop = yes\n[Debug]\nValidateOutput = on\nLog = verbose\n", notes);
    expect(s.asyncInterop && s.validateOutput && s.verbose && notes.empty(), "boolean spellings and Log = verbose");
    s = settings("[Debug]\nValidateOutput = maybe\nLog = loud\n", notes);
    expect(!s.validateOutput && !s.verbose && has_note(notes, "must be true or false, not 'maybe'") &&
        has_note(notes, "must be normal or verbose"), "invalid values fall back with plain notes");
    s = settings("[Kernels]\nNativeFp8 = on\n[Latency]\nFrameAge = 1\n", notes);
    expect(has_note(notes, "[Kernels] NativeFp8 is not a Windows setting") && has_note(notes, "[Latency] FrameAge"),
        "Linux-only settings are reported as ignored");
    s = settings("", notes);
    expect(!s.showWatermark, "watermark is hidden by default");
    s = settings("[DLSS]\nShowWatermark = true\n", notes);
    expect(s.showWatermark && notes.empty(), "[DLSS] ShowWatermark = true keeps NVIDIA's stamp");
    s = settings("[DLSS]\nShowWatermark = sometimes\n", notes);
    expect(!s.showWatermark && has_note(notes, "ShowWatermark must be true or false"), "invalid ShowWatermark falls back to hidden with a note");
    s = settings("[Paths]\nCacheDir = D:\\d4r-cache\n", notes);
    expect(s.cacheDir == L"D:\\d4r-cache" && notes.empty(), "CacheDir path");

    // Pins: fixture DLL contents are ordinary text; their hashes are computed here.
    std::filesystem::create_directories(dir / L"ngx");
    { std::ofstream(dir / L"nvngx_dlss.dll", std::ios::binary) << "validated dlss fixture"; }
    { std::ofstream(dir / L"ngx" / L"_nvngx.dll", std::ios::binary) << "unvalidated core fixture"; }
    const auto dlss = player::sha256(dir / L"nvngx_dlss.dll"), core = player::sha256(dir / L"ngx" / L"_nvngx.dll");
    const auto pins = player::parse_pins("# comment\ndlss " + dlss + " 310.9.1 validated\nngx " + core + " 32.0.15.9636 unvalidated\n"
        "dlss " + std::string(64, 'A') + " 310.7.0 unvalidated\n");
    notes.clear();
    expect(error_of([&] { player::verify_dll(dir, pins, "dlss", L"nvngx_dlss.dll", "nvngx_dlss.dll", notes); }).empty() && notes.empty(),
        "validated DLSS pin accepted silently");
    expect(error_of([&] { player::verify_dll(dir, pins, "ngx", L"ngx\\_nvngx.dll", "_nvngx.dll", notes); }).empty() &&
        has_note(notes, "not yet validated on Windows"), "unvalidated pin accepted with a note");
    { std::ofstream(dir / L"nvngx_dlss.dll", std::ios::binary) << "changed"; }
    auto error = error_of([&] { player::verify_dll(dir, pins, "dlss", L"nvngx_dlss.dll", "nvngx_dlss.dll", notes); });
    expect(error.find("Wrong nvngx_dlss.dll: you have version") != std::string::npos && error.find("310.9.1 or 310.7.0") != std::string::npos,
        "changed DLL rejected with you-have/you-need: " + error);
    std::filesystem::remove(dir / L"nvngx_dlss.dll");
    error = error_of([&] { player::verify_dll(dir, pins, "dlss", L"nvngx_dlss.dll", "nvngx_dlss.dll", notes); });
    expect(error.find("is missing. Copy NVIDIA's nvngx_dlss.dll") != std::string::npos, "missing DLL explains where to copy it");
    expect(!error_of([] { player::parse_pins("dlss short 1 validated\n"); }).empty(), "damaged pins file rejected");
    expect(!error_of([] { player::parse_pins("other " + std::string(64, 'A') + " 1 validated\n"); }).empty(), "unknown pin kind rejected");

    std::printf("%s PLAYER_CONFIG failures=%d\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
