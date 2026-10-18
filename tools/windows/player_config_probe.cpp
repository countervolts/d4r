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

using d4r::diag::AdapterLuid;
using d4r::diag::GpuCandidate;
AdapterLuid luid_of(unsigned low, unsigned high = 0) {
    AdapterLuid luid{};
    for (int i = 0; i < 4; ++i) { luid[i] = static_cast<unsigned char>(low >> (8 * i)); luid[4 + i] = static_cast<unsigned char>(high >> (8 * i)); }
    return luid;
}
void write_file(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}
bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// Architecture selection rules that decide which NVAPI, shim and kernels a GPU gets. Pure logic, so every
// registered target is covered here without the hardware; `builtNvapi` (optional) holds real nvapi64.dll
// builds, one per target, to check the compiled-in target marker the same way the shim and packager do.
void architecture_selection(const std::filesystem::path& root, const std::filesystem::path& builtNvapi) {
    namespace diag = d4r::diag;
    std::filesystem::remove_all(root);
    std::vector<std::string> archs;
    for (const auto& target : diag::gpu_targets) archs.push_back(target.architecture);
    expect(archs.size() == 11, "the registry lists the 11 Windows GPU targets");

    // D4R_GPU_ARCH: valid, suffixed, missing, empty and unsupported values.
    for (const auto& arch : archs) {
        auto r = diag::resolve_target_arch(arch.c_str(), "gfx1101");
        expect(r.architecture == arch && r.source == "D4R_GPU_ARCH" && r.warning.empty(), "D4R_GPU_ARCH=" + arch + " selects " + arch);
        r = diag::resolve_target_arch((arch + ":xnack-").c_str(), "gfx1201");
        expect(r.architecture == arch, "feature-suffixed D4R_GPU_ARCH is normalized for " + arch);
        r = diag::resolve_target_arch(nullptr, arch.c_str());
        expect(r.architecture == arch && r.source == "build" && r.warning.empty(), "missing D4R_GPU_ARCH uses the build target " + arch);
        r = diag::resolve_target_arch("", arch.c_str());
        expect(r.architecture == arch && r.warning.empty(), "empty D4R_GPU_ARCH uses the build target " + arch);
    }
    for (const char* bad : {"gfx1030", "gfx9999", "GFX1101", "1101", "gfx1101 ", "rdna3"}) {
        const auto r = diag::resolve_target_arch(bad, "gfx1151");
        expect(r.architecture == "gfx1151" && r.source == "build" && contains(r.warning, std::string("'") + bad + "' is not a supported target"),
            std::string("unsupported D4R_GPU_ARCH='") + bad + "' is reported and the build target is kept");
    }

    // LUID text used to hand the game's adapter to NVAPI.
    for (const auto luid : {luid_of(0), luid_of(0x1a2b3c4d, 0), luid_of(0xdeadbeef, 0x7), luid_of(0xffffffff, 0xffffffff)}) {
        AdapterLuid parsed{};
        expect(diag::parse_luid(diag::format_luid(luid).c_str(), parsed) && parsed == luid, "LUID round trip " + diag::format_luid(luid));
    }
    AdapterLuid scratch{};
    for (const char* bad : {"", "1234", "0000000000000000", "0000000:0000000a", "0000000g:00000001", "00000000:0000000100", "0x000000:00000001", "00000000:+0000001"})
        expect(!diag::parse_luid(bad, scratch), std::string("malformed LUID '") + bad + "' is rejected");

    // Release layout: nvapi\<arch>\nvapi64.dll for all targets plus the 0.1.5-style shared copy beside nvcuda.
    const auto release = root / L"release";
    for (const auto& arch : archs) write_file(release / L"nvapi" / diag::wide(arch) / L"nvapi64.dll", "nvapi " + arch);
    write_file(release / L"zluda" / L"nvapi64.dll", "nvapi gfx1101 (shared)");
    write_file(release / L"nvapi" / L"backend" / L"nvapi64.dll", "raw backend");
    for (const auto& arch : archs) {
        const auto choice = player::select_compat_nvapi(release, arch);
        expect(choice.error.empty() && choice.path == release / L"nvapi" / diag::wide(arch) / L"nvapi64.dll" && player::read_text(choice.path) == "nvapi " + arch,
            "NVAPI for " + arch + " is its own nvapi\\" + arch + " build, not the shared gfx1101 copy");
    }
    std::vector<std::string> notes;
    player::note_stale_shared_nvapi(release, notes);
    expect(has_note(notes, "left over from an older d4r release"), "a leftover shared NVAPI is called out");

    // A missing per-target build is an error that names the fix; no other file may stand in for it.
    for (const auto& arch : archs) {
        const auto folder = root / (L"missing-" + diag::wide(arch));
        std::filesystem::remove_all(folder);
        std::filesystem::copy(release, folder, std::filesystem::copy_options::recursive);
        std::filesystem::remove(folder / L"nvapi" / diag::wide(arch) / L"nvapi64.dll");
        const auto choice = player::select_compat_nvapi(folder, arch);
        expect(choice.path.empty() && contains(choice.error, "no NVAPI build for " + arch) && contains(choice.error, "nvapi\\" + arch + "\\nvapi64.dll"),
            "missing nvapi\\" + arch + " fails with a clear message instead of using another DLL");
        expect(!error_of([&] { player::compat_nvapi_path(folder, arch); }).empty(), "compat_nvapi_path(" + arch + ") throws when its build is missing");
        std::filesystem::remove_all(folder);
    }

    // Single-target layouts (quick test, game package, development folders).
    const auto single = root / L"single";
    expect(contains(player::select_compat_nvapi(single, "gfx1151").error, "nvapi64.dll is missing"), "an empty folder has no NVAPI");
    write_file(single / L"zluda" / L"nvapi64.dll", "single zluda");
    expect(player::select_compat_nvapi(single, "gfx1151").path == single / L"zluda" / L"nvapi64.dll", "a single-target package may keep NVAPI beside nvcuda");
    write_file(single / L"nvapi" / L"nvapi64.dll", "single nvapi");
    expect(player::select_compat_nvapi(single, "gfx1151").path == single / L"nvapi" / L"nvapi64.dll", "nvapi\\nvapi64.dll outranks the copy beside nvcuda");
    notes.clear(); player::note_stale_shared_nvapi(single, notes);
    expect(notes.empty(), "single-target layouts are not flagged as stale");

    // The environment the shim hands to ZLUDA, NVAPI and NGX.
    {   // target_arch() is fixed per process; the per-target matrix above covers the other architectures.
        const auto folder = root / L"apply";
        for (const auto& other : archs) write_file(folder / L"nvapi" / diag::wide(other) / L"nvapi64.dll", "nvapi " + other);
        auto s = settings("", notes); s.cacheDir = (folder / L"cache").wstring();
        player::apply(folder, s);
        const auto current = std::string(diag::HipApi::target_arch());
        expect(_wgetenv(L"D4R_GPU_ARCH") && std::wstring(_wgetenv(L"D4R_GPU_ARCH")) == diag::wide(current), "player environment publishes D4R_GPU_ARCH=" + current);
        expect(std::filesystem::path(_wgetenv(L"D4R_NVAPI_DLL")) == folder / L"nvapi" / diag::wide(current) / L"nvapi64.dll",
            "player environment points D4R_NVAPI_DLL at the build for " + current);
        std::filesystem::remove(folder / L"nvapi" / diag::wide(current) / L"nvapi64.dll");
        player::apply(folder, s);
        expect(!std::getenv("D4R_NVAPI_DLL"), "a missing build clears D4R_NVAPI_DLL instead of keeping a stale path");
    }

    // Several GPUs: iGPU + two dGPUs of one architecture + every registered target on one machine.
    const std::vector<GpuCandidate> machine{{"AMD Radeon 780M", "gfx1103", luid_of(10)}, {"AMD Radeon RX 7700 XT", "gfx1101", luid_of(11)},
        {"AMD Radeon RX 7700 XT #2", "gfx1101", luid_of(12)}, {"Radeon 8060S", "gfx1151", luid_of(13)}};
    expect(diag::find_gpu_device(machine, "gfx1101") == 1, "without a LUID the first device of the architecture is used");
    const auto second = luid_of(12), igpu = luid_of(10), unknown = luid_of(99);
    expect(diag::find_gpu_device(machine, "gfx1101", -1, &second) == 2, "the LUID picks the second of two same-architecture GPUs");
    expect(diag::find_gpu_device(machine, "gfx1103", -1, &igpu) == 0, "the iGPU is found by LUID");
    expect(diag::find_gpu_device(machine, "gfx1101", -1, &igpu) == -1, "a LUID of another architecture does not match");
    expect(diag::find_gpu_device(machine, "gfx1101", 2) == 2 && diag::find_gpu_device(machine, "gfx1101", 3) == -1, "an explicit ordinal must also be of the target architecture");
    expect(diag::find_gpu_device(machine, "gfx1151", -1, &unknown) == -1, "an unknown adapter LUID matches nothing");
    std::vector<GpuCandidate> everything;
    for (size_t i = 0; i < archs.size(); ++i) everything.push_back({"GPU " + archs[i], archs[i], luid_of(100 + unsigned(i))});
    for (size_t i = 0; i < archs.size(); ++i) {
        const auto luid = luid_of(100 + unsigned(i));
        bool exact = diag::find_gpu_device(everything, archs[i], -1, &luid) == int(i) && diag::find_gpu_device(everything, archs[i]) == int(i);
        for (size_t j = 0; j < archs.size(); ++j) exact = exact && (j == i || diag::find_gpu_device(everything, archs[j], -1, &luid) == -1);
        expect(exact, "among all 11 architectures " + archs[i] + " resolves only to its own adapter");
    }

    // Diagnostics for genuinely unsupported or mismatched GPUs (no crash, an actionable message).
    auto message = diag::no_device_message(machine, "gfx1101", &unknown);
    expect(contains(message, "No HIP device has the game's adapter LUID") && contains(message, "gfx1151 (Radeon 8060S)"), "unknown adapter: " + message);
    const auto strix = luid_of(13);
    message = diag::no_device_message(machine, "gfx1101", &strix);
    expect(contains(message, "The GPU is gfx1151") && contains(message, "built for gfx1101") && contains(message, "extract the d4r ZIP again"),
        "a gfx1101 file on a gfx1151 adapter is named as a mixed install: " + message);
    const std::vector<GpuCandidate> rdna2{{"AMD Radeon RX 6800", "gfx1030", luid_of(5)}};
    const auto six = luid_of(5);
    message = diag::no_device_message(rdna2, "gfx1101", &six);
    expect(contains(message, "gfx1030") && contains(message, "does not support") && contains(message, "gfx1154"), "unsupported RDNA2 GPU: " + message);
    message = diag::no_device_message(rdna2, "gfx1201");
    expect(contains(message, "none supported by d4r"), "no supported device at all: " + message);
    expect(contains(diag::no_device_message({}, "gfx1100"), "reports no GPU"), "an empty HIP device list asks for a driver update");

    // Loader: pick the shim for the game's adapter, never a different GPU's.
    const auto has_all = [](const std::string& arch) { return diag::supported_gpu_target(arch); }; // A full release: one shim per target.
    for (size_t i = 0; i < archs.size(); ++i) {
        const auto luid = luid_of(100 + unsigned(i));
        const auto choice = diag::choose_loader_architecture(everything, &luid, has_all, "all");
        expect(choice.architecture == archs[i] && choice.error.empty(), "loader picks _nvngx_" + archs[i] + ".dll for that adapter");
    }
    auto choice = diag::choose_loader_architecture(machine, &igpu, has_all, "all");
    expect(choice.architecture == "gfx1103", "loader follows the iGPU when the game renders on it");
    choice = diag::choose_loader_architecture(machine, &second, has_all, "all");
    expect(choice.architecture == "gfx1101", "loader follows the game's dGPU among several GPUs");
    choice = diag::choose_loader_architecture(machine, nullptr, has_all, "all");
    expect(choice.architecture.empty() && contains(choice.error, "several GPUs with different architectures"), "before an adapter is known, several architectures wait for Init");
    choice = diag::choose_loader_architecture(machine, nullptr, [](const std::string& a) { return a == "gfx1151"; }, "gfx1151");
    expect(choice.architecture == "gfx1151", "before an adapter is known, the only architecture with a build is used");
    choice = diag::choose_loader_architecture(machine, &strix, [](const std::string& a) { return a == "gfx1101"; }, "gfx1101");
    expect(choice.architecture.empty() && contains(choice.error, "no build for your GPU (gfx1151)") && contains(choice.error, "supports: gfx1101"), "loader: " + choice.error);
    choice = diag::choose_loader_architecture(rdna2, &six, has_all, "all");
    expect(choice.architecture.empty() && contains(choice.error, "gfx1030") && contains(choice.error, "not supported by d4r"), "loader: " + choice.error);
    choice = diag::choose_loader_architecture(rdna2, nullptr, has_all, "all");
    expect(choice.architecture.empty() && contains(choice.error, "no supported AMD GPU") && contains(choice.error, "gfx1030"), "loader: " + choice.error);
    expect(diag::nvapi_target_error("gfx1151", "gfx1151", "p").empty() && contains(diag::nvapi_target_error("gfx1101", "gfx1151", "p"), "built for gfx1101 but the selected GPU is gfx1151"),
        "an NVAPI built for another target is refused");

    // Real nvapi64.dll builds, one per target: the file in nvapi\<arch> names <arch> and only <arch>.
    if (builtNvapi.empty()) { std::printf("skip real NVAPI builds (no directory given)\n"); return; }
    for (const auto& arch : archs) {
        const auto layout = root / L"built";
        const auto source = builtNvapi / diag::wide(arch) / L"nvapi64.dll";
        std::filesystem::create_directories(layout / L"nvapi" / diag::wide(arch));
        std::filesystem::copy_file(source, layout / L"nvapi" / diag::wide(arch) / L"nvapi64.dll", std::filesystem::copy_options::overwrite_existing);
    }
    for (const auto& arch : archs) {
        const auto choice = player::select_compat_nvapi(root / L"built", arch);
        expect(choice.error.empty(), "built layout has nvapi\\" + arch);
        if (!choice.error.empty()) continue;
        const HMODULE module = LoadLibraryExW(choice.path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        auto target = module ? reinterpret_cast<const char*(__cdecl*)()>(reinterpret_cast<void*>(GetProcAddress(module, "d4r_NvapiTargetArchitecture"))) : nullptr;
        const std::string built = target ? target() : "";
        expect(built == arch, "nvapi\\" + arch + "\\nvapi64.dll names " + arch + " (reports '" + built + "')");
        const auto bytes = player::read_text(choice.path);
        std::string others;
        for (const auto& other : archs) if (other != arch && contains(bytes, "D4R_NVAPI_TARGET=" + other)) others += other + " ";
        expect(contains(bytes, "D4R_NVAPI_TARGET=" + arch + '\0') && others.empty(), "nvapi\\" + arch + " carries only its own marker for the packager (" + others + ")");
        expect(diag::nvapi_target_error(built, arch, "p").empty(), "the shim accepts nvapi\\" + arch + " for " + arch);
        if (module) FreeLibrary(module);
    }
}
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2 && argc != 3) { std::printf("usage: player_config_probe FIXTURE_DIR [BUILT_NVAPI_DIR]\n"); return 2; }
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
    s = settings("[Interop]\nBatchInputCopies = yes\n", notes);
    expect(s.batchInputCopies && notes.empty(), "input batching selected through player INI");
    s.cacheDir = (dir / L"cache").wstring(); player::apply(dir, s);
    expect(std::getenv("D4R_BATCH_INPUT_COPIES") != nullptr, "player environment enables input batching");
    s = settings("[Interop]\nBatchInputCopies = invalid\n", notes);
    expect(!s.batchInputCopies && has_note(notes, "BatchInputCopies must be true or false"), "invalid batching falls back to unbatched with a note");
    s.cacheDir = (dir / L"cache").wstring(); player::apply(dir, s);
    expect(std::getenv("D4R_BATCH_INPUT_COPIES") == nullptr, "player environment clears earlier input batching");
    s = settings("", notes);
    expect(!s.batchInputCopies && notes.empty(), "older INIs retain unbatched behavior");
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
    expect(s.rrBackend == "fsr" && has_note(notes, "FSR RR is experimental"), "FSR selection is reported as experimental");
    s = settings("[RayReconstruction]\nBackend = fsr\nFsrDenoiser = amd\\denoiser.dll\n", notes);
    s.cacheDir = (dir / L"cache").wstring(); player::apply(dir, s);
    expect(std::string(std::getenv("D4R_RR_BACKEND")) == "fsr" &&
        std::filesystem::path(_wgetenv(L"D4R_FSR_RR_DLL")) == std::filesystem::absolute(dir / L"amd" / L"denoiser.dll").lexically_normal() &&
        std::filesystem::path(_wgetenv(L"D4R_FSR_VKD3D_CORE")) == dir / L"fsr" / L"d4r_vkd3d_core.dll" &&
        std::filesystem::path(_wgetenv(L"D4R_FSR_AMDXC_SHIM")) == dir / L"fsr" / L"amdxc64.dll" &&
        std::filesystem::path(_wgetenv(L"D4R_FSR_CACHE_DIR")) == dir / L"cache" / L"vkd3d" && !std::getenv("D4R_RR_MODEL_DIR"),
        "FSR environment selected from d4r.ini without a launcher");
    s = settings("[RayReconstruction]\nBackend = fsr\nFsrStabilityBias = 0\nFsrDetail = 0.25\n", notes);
    s.cacheDir = (dir / L"cache").wstring(); player::apply(dir, s);
    expect(notes.empty() || has_note(notes, "experimental"), "FSR tuning keys are recognized");
    expect(std::string(std::getenv("D4R_FSR_RR_STABILITY_BIAS")) == "0" && std::string(std::getenv("D4R_FSR_RR_DETAIL")) == "0.25",
        "FSR tuning reaches the environment");
    s = settings("[RayReconstruction]\nBackend = fsr\nFsrStabilityBias = auto\nFsrDetail = 2\n", notes);
    s.cacheDir = (dir / L"cache").wstring(); player::apply(dir, s);
    expect(has_note(notes, "FsrDetail must be a number from 0 to 1") && !std::getenv("D4R_FSR_RR_STABILITY_BIAS") && !std::getenv("D4R_FSR_RR_DETAIL"),
        "auto bias leaves AMD's default and an out-of-range detail is rejected");
    s = settings("[RayReconstruction]\nBackend = typo\n", notes);
    expect(s.rrBackend == "off" && has_note(notes, "Backend must be"), "invalid RR backend stays off");
    s = settings("", notes); s.cacheDir = (dir / L"cache").wstring(); player::apply(dir, s);
    expect(!std::getenv("D4R_RR_BACKEND") && !std::getenv("D4R_RR_MODEL_DIR") && !std::getenv("D4R_FSR_RR_DLL") &&
        !std::getenv("D4R_FSR_VKD3D_CORE") && !std::getenv("D4R_FSR_AMDXC_SHIM") && !std::getenv("D4R_FSR_CACHE_DIR") &&
        !std::getenv("D4R_FSR_RR_STABILITY_BIAS") && !std::getenv("D4R_FSR_RR_DETAIL"), "default RR off clears stale selection");

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
    expect(player::compat_nvapi_path(dir) == archNvapi, "a shared NVAPI beside nvcuda never outranks the architecture-specific one");
    std::filesystem::remove(colocatedNvapi);
    std::filesystem::remove(archNvapi);
    std::filesystem::remove(dir / L"nvapi" / L"nvapi64.dll");
    architecture_selection(dir / L"architecture-fixture", argc > 2 ? std::filesystem::path(argv[2]) : std::filesystem::path());
    std::filesystem::create_directories(dir / L"nvapi");
    { std::ofstream(dir / L"nvapi" / L"nvapi64.dll", std::ios::binary) << "generic nvapi fixture"; }
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
