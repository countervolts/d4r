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
    for (const char* model : {"game", "7"}) {
        s = settings(std::string("[DLSS]\nModel = ") + model + "\n", notes);
        expect(s.preset == 11 && has_note(notes, "not available on Windows"), std::string("Model = ") + model + " is rejected with a note and K");
    }
    s = settings("[DLSS]\nModel = E\n", notes);
    expect(s.preset == 5 && has_note(notes, "no standalone Windows engine"), "Model = E selects NGX preset 5 with limitation note");
    s = settings("[DLSS]\nModel = L\n", notes);
    expect(s.preset == 12 && has_note(notes, "not validated"), "Model = L selects NGX preset 12 with limitation note");
    s = settings("[DLSS]\nModel = 12\n", notes);
    expect(s.preset == 12 && has_note(notes, "not validated"), "Model = 12 selects NGX preset 12 with limitation note");
    s = settings("[Interop]\nAsyncInterop = yes\n[Debug]\nValidateOutput = on\nLog = verbose\n", notes);
    expect(s.asyncInterop && s.validateOutput && s.verbose && notes.empty(), "boolean spellings and Log = verbose");
    s = settings("[Debug]\nValidateOutput = maybe\nLog = loud\n", notes);
    expect(!s.validateOutput && !s.verbose && has_note(notes, "must be true or false, not 'maybe'") &&
        has_note(notes, "must be normal or verbose"), "invalid values fall back with plain notes");
    s = settings("[Kernels]\nNativeFp8 = on\n[Latency]\nFrameAge = 1\n", notes);
    expect(s.nativeFp8 && !has_note(notes, "[Kernels] NativeFp8") && has_note(notes, "[Latency] FrameAge"),
        "native FP8 is parsed while Linux-only settings are reported as ignored");
    s = settings("", notes);
    expect(!s.showWatermark, "watermark is hidden by default");
    s = settings("[DLSS]\nShowWatermark = true\n", notes);
    expect(s.showWatermark && notes.empty(), "[DLSS] ShowWatermark = true keeps NVIDIA's stamp");
    s = settings("[DLSS]\nShowWatermark = sometimes\n", notes);
    expect(!s.showWatermark && has_note(notes, "ShowWatermark must be true or false"), "invalid ShowWatermark falls back to hidden with a note");
    s = settings("[Paths]\nCacheDir = D:\\d4r-cache\n", notes);
    expect(s.cacheDir == L"D:\\d4r-cache" && notes.empty(), "CacheDir path");
    s = settings("[Paths]\nNativeDir = private-native\n", notes);
    expect(s.nativeDir == L"private-native" && notes.empty(), "NativeDir path");
    expect(player::native_code_object_dir(dir, &s) == dir / L"private-native", "relative NativeDir resolves beside d4r.ini");
    expect(!error_of([&] { player::verify_code_objects(dir, &s); }).empty(), "incomplete private native directory rejected");
    s.cacheDir = (dir / L"cache").wstring();
    player::apply(dir, s);
    expect(std::wstring(_wgetenv(L"D4R_ZLUDA_NATIVE_DIR")) == (dir / L"private-native").wstring(),
        "player environment selects configured private native directory");
    s = settings("[Paths]\nNativeDir = auto\n", notes);
    expect(s.nativeDir.empty() && notes.empty(), "NativeDir auto preserves packaged kernels");
    s = settings("[RayReconstruction]\nBackend = NATIVE\nModelDir = local-rr\n", notes);
    expect(s.rrBackend == "native" && s.rrModelDir == L"local-rr" && notes.empty(), "native RR settings parse independently of SR");
    expect(player::rr_model_directory(dir, s) == std::filesystem::absolute(dir / L"local-rr").lexically_normal(), "relative RR model resolves beside d4r.ini");
    s.cacheDir = (dir / L"cache").wstring(); player::apply(dir, s);
    expect(std::string(std::getenv("D4R_RR_BACKEND")) == "native" &&
        std::filesystem::path(_wgetenv(L"D4R_RR_MODEL_DIR")) == player::rr_model_directory(dir, s), "native RR environment selected without a launcher");
    s = settings("[RayReconstruction]\nBackend = fsr\n", notes);
    expect(s.rrBackend == "fsr" && has_note(notes, "FSR RR is unavailable"), "unavailable FSR selection remains explicit");
    s = settings("[RayReconstruction]\nBackend = typo\n", notes);
    expect(s.rrBackend == "off" && has_note(notes, "Backend must be"), "invalid RR backend stays off");
    s = settings("", notes); s.cacheDir = (dir / L"cache").wstring(); player::apply(dir, s);
    expect(!std::getenv("D4R_RR_BACKEND") && !std::getenv("D4R_RR_MODEL_DIR"), "default RR off clears stale selection");

    std::filesystem::create_directories(dir / L"nvapi");
    { std::ofstream(dir / L"nvapi" / L"nvapi64.dll", std::ios::binary) << "generic nvapi fixture"; }
    expect(player::compat_nvapi_path(dir) == dir / L"nvapi" / L"nvapi64.dll", "player mode accepts quick-test NVAPI layout");
    const auto archNvapi = dir / L"nvapi" / d4r::diag::wide(d4r::diag::HipApi::target_arch()) / L"nvapi64.dll";
    std::filesystem::create_directories(archNvapi.parent_path());
    { std::ofstream(archNvapi, std::ios::binary) << "architecture NVAPI fixture"; }
    expect(player::compat_nvapi_path(dir) == archNvapi, "player mode prefers architecture-specific release NVAPI");
    std::filesystem::create_directories(dir / L"zluda");
    const auto colocatedNvapi = dir / L"zluda" / L"nvapi64.dll";
    { std::ofstream(colocatedNvapi, std::ios::binary) << "colocated compatibility NVAPI fixture"; }
    expect(player::compat_nvapi_path(dir) == colocatedNvapi, "player mode reuses NVAPI colocated with nvcuda");
    std::filesystem::remove(colocatedNvapi);
    std::filesystem::remove(archNvapi);
    expect(player::raw_nvapi_backend_path(dir) == dir / L"nvapi" / L"backend" / L"nvapi64.dll",
        "raw ZLUDA NVAPI backend has a separate explicit path");
    const auto archNative = dir / L"native" / d4r::diag::wide(d4r::diag::HipApi::target_arch());
    std::filesystem::create_directories(archNative);
    expect(player::native_code_object_dir(dir) == archNative, "player mode prefers architecture-specific release kernels");
    std::filesystem::remove_all(archNative);
    std::filesystem::create_directories(dir / L"native");
    expect(player::native_code_object_dir(dir) == dir / L"native", "player mode accepts target-specific flat kernel package");

    // Pins: fixture DLL contents are ordinary text; their hashes are computed here.
    std::filesystem::create_directories(dir / L"ngx");
    std::filesystem::create_directories(dir / L"dlss");
    { std::ofstream(dir / L"dlss" / L"nvngx_dlss.dll", std::ios::binary) << "validated dlss fixture"; }
    { std::ofstream(dir / L"ngx" / L"_nvngx.dll", std::ios::binary) << "unvalidated core fixture"; }
    const auto dlss = player::sha256(dir / L"dlss" / L"nvngx_dlss.dll"), core = player::sha256(dir / L"ngx" / L"_nvngx.dll");
    const auto pins = player::parse_pins("# comment\ndlss " + dlss + " 310.9.1 validated\ndlssd " + std::string(64, 'B') +
        " 310.9.1 unvalidated\nngx " + core + " 32.0.15.9636 unvalidated\n" +
        "dlss " + std::string(64, 'A') + " 310.7.0 unvalidated\n");
    expect(pins.size() == 4, "installer-only dlssd pin is accepted");
    notes.clear();
    expect(error_of([&] { player::verify_dll(dir, pins, "dlss", L"dlss\\nvngx_dlss.dll", "nvngx_dlss.dll", notes); }).empty() && notes.empty(),
        "validated DLSS pin accepted silently");
    expect(error_of([&] { player::verify_dll(dir, pins, "ngx", L"ngx\\_nvngx.dll", "_nvngx.dll", notes); }).empty() &&
        has_note(notes, "not yet validated on Windows"), "unvalidated pin accepted with a note");
    { std::ofstream(dir / L"dlss" / L"nvngx_dlss.dll", std::ios::binary) << "changed"; }
    auto error = error_of([&] { player::verify_dll(dir, pins, "dlss", L"dlss\\nvngx_dlss.dll", "nvngx_dlss.dll", notes); });
    expect(error.find("Wrong nvngx_dlss.dll: you have version") != std::string::npos && error.find("310.9.1 or 310.7.0") != std::string::npos,
        "changed DLL rejected with you-have/you-need: " + error);
    std::filesystem::remove(dir / L"dlss" / L"nvngx_dlss.dll");
    error = error_of([&] { player::verify_dll(dir, pins, "dlss", L"dlss\\nvngx_dlss.dll", "nvngx_dlss.dll", notes); });
    expect(error.find("is missing. Copy NVIDIA's nvngx_dlss.dll") != std::string::npos, "missing DLL explains where to copy it");
    expect(!error_of([] { player::parse_pins("dlss short 1 validated\n"); }).empty(), "damaged pins file rejected");
    expect(!error_of([] { player::parse_pins("other " + std::string(64, 'A') + " 1 validated\n"); }).empty(), "unknown pin kind rejected");

    std::printf("%s PLAYER_CONFIG failures=%d\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
