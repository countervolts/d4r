// d4r_nvngx_shim_native_d3d12.cpp: the native D3D12 engine (engine/d3d12). Preset K on Windows
// without vkd3d-proton, driven through d4r's submission hooks and, where the package has it,
// the d3dhip network behind d4r_hip_net.dll.

// the command-list stubs live in d4r_nvngx_shim.cpp, which assembles them once
#define D4R_D3D12_SPLIT_DECL 1
#include "d4r_nvngx_shim_internal.h"

#ifdef D4R_ENGINE_D3DHIP

static bool g_hipLoaded = false;

static bool load_hip_net()
{
    if (g_hipLoaded)
        return g_hipApi.create != nullptr;
    g_hipLoaded = true;
    // d4r_hip_net.dll sits beside this DLL (the game's d4r folder)
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&load_hip_net), &self);
    std::wstring path;
    if (self != nullptr)
    {
        wchar_t file[MAX_PATH * 3] = {};
        const DWORD length = GetModuleFileNameW(self, file, DWORD(std::size(file)));
        if (length > 0 && length < std::size(file))
        {
            path.assign(file, length);
            const size_t slash = path.find_last_of(L"\\/");
            path.resize(slash == std::wstring::npos ? 0 : slash + 1);
            path += L"d4r_hip_net.dll";
        }
    }
    HMODULE module = path.empty() ? nullptr : LoadLibraryW(path.c_str());
    const DWORD pathError = GetLastError();
    if (module == nullptr)
        module = LoadLibraryW(L"d4r_hip_net.dll");
    if (module == nullptr)
    {
        logf("engine backend (D3D12, d3dhip): d4r_hip_net.dll could not be loaded (error %lu); using the DXIL network",
             pathError);
        return false;
    }
    g_hipApi.create = reinterpret_cast<decltype(g_hipApi.create)>(GetProcAddress(module, "d4rHipNetCreate"));
    g_hipApi.tokens = reinterpret_cast<decltype(g_hipApi.tokens)>(GetProcAddress(module, "d4rHipNetTokens"));
    g_hipApi.head = reinterpret_cast<decltype(g_hipApi.head)>(GetProcAddress(module, "d4rHipNetHead"));
    g_hipApi.run = reinterpret_cast<decltype(g_hipApi.run)>(GetProcAddress(module, "d4rHipNetRun"));
    g_hipApi.destroy = reinterpret_cast<decltype(g_hipApi.destroy)>(GetProcAddress(module, "d4rHipNetDestroy"));
    if (g_hipApi.create == nullptr || g_hipApi.tokens == nullptr || g_hipApi.head == nullptr || g_hipApi.run == nullptr ||
        g_hipApi.destroy == nullptr)
    {
        logf("engine backend (D3D12, d3dhip): d4r_hip_net.dll lacks the network's exports; using the DXIL network");
        g_hipApi = HipNetApi{};
        return false;
    }
    g_hipApi.useFence = reinterpret_cast<decltype(g_hipApi.useFence)>(GetProcAddress(module, "d4rHipNetUseFence"));
    g_hipApi.launchAfter = reinterpret_cast<decltype(g_hipApi.launchAfter)>(GetProcAddress(module, "d4rHipNetLaunchAfter"));
    g_hipApi.rrCreate = reinterpret_cast<decltype(g_hipApi.rrCreate)>(GetProcAddress(module, "d4rHipRrCreate"));
    g_hipApi.rrExternal = reinterpret_cast<decltype(g_hipApi.rrExternal)>(GetProcAddress(module, "d4rHipRrExternal"));
    g_hipApi.rrRun = reinterpret_cast<decltype(g_hipApi.rrRun)>(GetProcAddress(module, "d4rHipRrRun"));
    g_hipApi.rrUseFence = reinterpret_cast<decltype(g_hipApi.rrUseFence)>(GetProcAddress(module, "d4rHipRrUseFence"));
    g_hipApi.rrLaunchAfter = reinterpret_cast<decltype(g_hipApi.rrLaunchAfter)>(GetProcAddress(module, "d4rHipRrLaunchAfter"));
    g_hipApi.rrDestroy = reinterpret_cast<decltype(g_hipApi.rrDestroy)>(GetProcAddress(module, "d4rHipRrDestroy"));
    if (!g_hipApi.rrCreate || !g_hipApi.rrExternal || !g_hipApi.rrRun || !g_hipApi.rrUseFence || !g_hipApi.rrLaunchAfter ||
        !g_hipApi.rrDestroy)
        g_hipApi.rrCreate = nullptr;   // all or nothing: Ray Reconstruction keeps its DXIL network
    logf("engine backend (D3D12, d3dhip): d4r_hip_net.dll loaded (%s)",
         g_hipApi.useFence && g_hipApi.launchAfter ? "GPU-fence handoff available" : "CPU-serialized handoff only");
    return true;
}

#endif

static EngineBackendDx g_engineDx;

static std::string engine_model_directory(const char* variable, const wchar_t* folder)
{
    std::string directory = env_string(variable);
    if (directory.empty())
    {
        char path[MAX_PATH * 3] = {};
        WideCharToMultiByte(CP_ACP, 0, (g_portable.dir + folder).c_str(), -1, path, sizeof(path), nullptr, nullptr);
        directory = path;
    }
    return directory;
}

static void load_engine_dx_model(std::unique_ptr<d4r::dx::Model>& model, const char* variable, const wchar_t* folder)
{
    const std::string directory = engine_model_directory(variable, folder);
    try
    {
        model = std::make_unique<d4r::dx::Model>(g.device, directory);
        logf("engine backend (D3D12): model %s loaded", directory.c_str());
    }
    catch (const std::exception& error)
    {
        logf("engine backend (D3D12): model %s: %s", directory.c_str(), error.what());
    }
}

// Preset M's package with its DXIL (compile_m.py --dxil on Windows): D4R_ENGINE_M_MODEL_DIR or d4r\engine\m
// (D4R_ENGINE_M_MODEL_LDR_DIR or d4r\engine\m-ldr for games without the HDR flag); preset L's likewise from
// D4R_ENGINE_L_MODEL_DIR or d4r\engine\l (D4R_ENGINE_L_MODEL_LDR_DIR, d4r\engine\l-ldr).

static std::string engine_m_directory(bool ldr, bool presetL)
{
    return presetL ? (ldr ? engine_model_directory("D4R_ENGINE_L_MODEL_LDR_DIR", L"\\engine\\l-ldr")
                          : engine_model_directory("D4R_ENGINE_L_MODEL_DIR", L"\\engine\\l"))
                   : (ldr ? engine_model_directory("D4R_ENGINE_M_MODEL_LDR_DIR", L"\\engine\\m-ldr")
                          : engine_model_directory("D4R_ENGINE_M_MODEL_DIR", L"\\engine\\m"));
}

// The AMD architecture of a PCI device id, for the shaders the native D3D12 engine may create: all of them use WMMA
// (RDNA 3 and later), and M's, L's and RR's FP8 WMMA ones RDNA 4 only. Creating a pipeline the GPU cannot run does
// not just fail: AMD's driver removes the game's device (FP8 shaders on an RX 7700 XT in Ghost of Tsushima; RDNA 3
// shaders on an RX 6800 in Delta Force, at the first pipeline). An id not listed counts as older.

static AmdGeneration amd_generation(unsigned device)
{
    // Navi 48 (RX 9070 XT / 9070 / 9070 GRE 0x7550, Radeon AI PRO R9700 0x7551), Navi 44 (RX 9060 XT / 9060)
    if (device == 0x7550 || device == 0x7551 || device == 0x7590)
        return AmdGeneration::Rdna4;
    // Navi 31/32/33 (RX 7000, Radeon PRO W7000, mobile), and the RDNA 3/3.5 APUs: Phoenix, Hawk Point, Strix Point,
    // Strix Halo, Krackan Point
    if ((device >= 0x7440 && device <= 0x749f) || device == 0x15bf || device == 0x15c8 || device == 0x1900 ||
        device == 0x1901 || device == 0x150e || device == 0x1586 || device == 0x1114)
        return AmdGeneration::Rdna3;
    return AmdGeneration::Older;
}

// The render adapter's PCI address through the kernel thunks (D3DKMTOpenAdapterFromLuid, KMTQAITYPE_ADAPTERADDRESS):
// DXGI reports OptiScaler's NVIDIA identity to every caller in the game, but not a different LUID or bus address.

static bool render_adapter_address(ID3D12Device* device, DWORD& bus, DWORD& address)
{
    struct OpenFromLuid { LUID luid; UINT adapter; };
    struct Query { UINT adapter; UINT type; void* data; UINT size; };
    struct Address { UINT bus, device, function; };
    struct Close { UINT adapter; };
    const HMODULE gdi = GetModuleHandleW(L"gdi32.dll") != nullptr ? GetModuleHandleW(L"gdi32.dll") : LoadLibraryW(L"gdi32.dll");
    if (device == nullptr || gdi == nullptr)
        return false;
    const auto open = reinterpret_cast<LONG(WINAPI*)(OpenFromLuid*)>(
        reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid")));
    const auto query = reinterpret_cast<LONG(WINAPI*)(Query*)>(
        reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTQueryAdapterInfo")));
    const auto close = reinterpret_cast<LONG(WINAPI*)(Close*)>(
        reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTCloseAdapter")));
    if (open == nullptr || query == nullptr || close == nullptr)
        return false;
    OpenFromLuid adapter{};
    device->GetAdapterLuid(&adapter.luid);
    if (open(&adapter) != 0)
        return false;
    Address location{};
    Query request{adapter.adapter, 6 /* KMTQAITYPE_ADAPTERADDRESS */, &location, sizeof(location)};
    const bool found = query(&request) == 0;
    Close closing{adapter.adapter};
    close(&closing);
    bus = location.bus;
    address = (location.device << 16) | location.function; // SPDRP_ADDRESS's encoding for PCI
    return found;
}

// The architecture of the AMD GPU the game renders on, from the PnP tree's PCI ids. When the render adapter's
// address cannot be read, the newest AMD GPU installed (an iGPU beside a Radeon card is the older one).
// Unknown: no AMD GPU found, or the PnP tree could not be read.

AmdGeneration amd_render_generation(ID3D12Device* renderDevice)
{
    static const AmdGeneration generation = [renderDevice] {
        DWORD renderBus = 0, renderAddress = 0;
        const bool located = render_adapter_address(renderDevice, renderBus, renderAddress);
        const HDEVINFO set = SetupDiGetClassDevsW(nullptr, L"PCI", nullptr, DIGCF_PRESENT | DIGCF_ALLCLASSES);
        if (set == INVALID_HANDLE_VALUE) return AmdGeneration::Unknown;
        AmdGeneration newest = AmdGeneration::Unknown, render = AmdGeneration::Unknown;
        unsigned renderId = 0;
        SP_DEVINFO_DATA data{};
        data.cbSize = sizeof(data);
        for (DWORD index = 0; SetupDiEnumDeviceInfo(set, index, &data); ++index)
        {
            wchar_t ids[512] = {};
            unsigned vendor = 0, device = 0;
            if (!SetupDiGetDeviceRegistryPropertyW(set, &data, SPDRP_HARDWAREID, nullptr, reinterpret_cast<PBYTE>(ids),
                                                   sizeof(ids) - sizeof(wchar_t), nullptr) ||
                std::swscanf(ids, L"PCI\\VEN_%4x&DEV_%4x", &vendor, &device) != 2 || vendor != 0x1002)
                continue;
            const AmdGeneration found = amd_generation(device);
            newest = std::max(newest, found);
            DWORD bus = 0, address = 0;
            if (located &&
                SetupDiGetDeviceRegistryPropertyW(set, &data, SPDRP_BUSNUMBER, nullptr, reinterpret_cast<PBYTE>(&bus), sizeof(bus), nullptr) &&
                SetupDiGetDeviceRegistryPropertyW(set, &data, SPDRP_ADDRESS, nullptr, reinterpret_cast<PBYTE>(&address), sizeof(address), nullptr) &&
                bus == renderBus && address == renderAddress)
            {
                render = found;
                renderId = device;
            }
        }
        SetupDiDestroyDeviceInfoList(set);
        const char* names[] = {"unknown", "older than RDNA 3", "RDNA 3", "RDNA 4"};
        if (render != AmdGeneration::Unknown)
            logf("render GPU: AMD 1002:%04x (%s)", renderId, names[int(render)]);
        else
            logf("render GPU: not identified (%s); the newest AMD GPU installed is %s",
                 located ? "no PnP device at its PCI address" : "no kernel adapter address", names[int(newest)]);
        return render != AmdGeneration::Unknown ? render : newest;
    }();
    return generation;
}

// Whether the game renders on an RDNA4 GPU (FP8 WMMA). Only asked once the game's device is known (g.device).

static bool rdna4_installed()
{
    return amd_render_generation(g.device) == AmdGeneration::Rdna4;
}

d4r::dx::MModel* native_engine_m(bool ldr, bool presetL)
{
    std::call_once(g_engineDx.mOnce[presetL][ldr], [ldr, presetL] {
        const std::string directory = engine_m_directory(ldr, presetL);
        const char* name = presetL ? "L" : "M";
        if (GetFileAttributesA((directory + "/offsets.bin").c_str()) == INVALID_FILE_ATTRIBUTES)
            return;
        // The FP8 WMMA shaders (RDNA 4) only with an RDNA 4 GPU installed (rdna4_installed): elsewhere the package's
        // rdna3/ ones. D4R_ENGINE_M_BACKEND=rdna3 or fp8 forces either.
        const std::string backend = env_string("D4R_ENGINE_M_BACKEND");
        if (backend == "fp8" || (backend != "rdna3" && rdna4_installed()))
        {
            try
            {
                auto model = std::make_unique<d4r::dx::MModel>(g.device, directory);
                if (model->isPresetL() != presetL)
                {
                    logf("engine backend (D3D12): %s holds a preset %s package, not %s", directory.c_str(), presetL ? "M" : "L", name);
                    return;
                }
                g_engineDx.modelM[presetL][ldr] = std::move(model);
                logf("engine backend (D3D12): preset %s model %s loaded", name, directory.c_str());
                return;
            }
            catch (const std::exception& error)
            {
                logf("engine backend (D3D12): preset %s model %s: %s", name, directory.c_str(), error.what());
            }
        }
        // GPUs without FP8 WMMA (RDNA 3): FP16 WMMA with software FP8 storage, from compile_m.py's rdna3/ shaders
        if (GetFileAttributesA((directory + "/rdna3/d3d12_ml.bin").c_str()) == INVALID_FILE_ATTRIBUTES)
            return;
        try
        {
            auto model = std::make_unique<d4r::dx::rdna3::MModel>(g.device, directory);
            if (model->isPresetL() != presetL)
            {
                logf("engine backend (D3D12): %s holds a preset %s package, not %s", directory.c_str(), presetL ? "M" : "L", name);
                return;
            }
            g_engineDx.modelMRdna3[presetL][ldr] = std::move(model);
            logf("engine backend (D3D12): preset %s model %s loaded (RDNA 3 shaders)", name, directory.c_str());
        }
        catch (const std::exception& error)
        {
            logf("engine backend (D3D12): preset %s model %s (RDNA 3 shaders): %s", name, directory.c_str(), error.what());
        }
    });
    return g_engineDx.modelM[presetL][ldr].get();
}

d4r::dx::RModel* native_engine_rr(unsigned preset)
{
    if(preset<4 || preset>6) return nullptr;
    const unsigned i=preset-4;
    std::call_once(g_engineDx.rrOnce[i],[i,preset]{
        static const char* variables[] = {"D4R_ENGINE_MODEL_RR_D_DIR", "D4R_ENGINE_MODEL_RR_E_DIR", "D4R_ENGINE_MODEL_RR_F_DIR"};
        static const wchar_t* folders[] = {L"\\engine\\rr-d", L"\\engine\\rr-e", L"\\engine\\rr-f"};
        const auto directory=engine_model_directory(variables[i],folders[i]);
        if(GetFileAttributesA((directory+"/rr-d3d12.txt").c_str())==INVALID_FILE_ATTRIBUTES)return;
        // every D3D12 RR package has FP8 WMMA stages (RDNA 4; rr-e's enc0_m too): their pipelines remove the game's
        // device elsewhere, as M's do (Ghost of Tsushima, RX 7700 XT)
        if(!rdna4_installed() && env_string("D4R_ENGINE_M_BACKEND")!="fp8"){
            logf("engine backend (D3D12 RR %c): needs an RDNA 4 GPU (FP8 WMMA); not loaded",char('D'+i));return;}
        try { auto model=std::make_unique<d4r::dx::RModel>(g.device,directory);
            if(model->preset()!=preset)throw std::runtime_error("RR model preset mismatch");
            g_engineDx.rrModel[i]=std::move(model);logf("engine backend (D3D12 RR %c): loaded %s",char('D'+i),directory.c_str());
        } catch(const std::exception& e){logf("engine backend (D3D12 RR): %s",e.what());}
    });
    return g_engineDx.rrModel[i].get();
}

// The RDNA 3 package of a preset (see native_engine_m, which loads both); null when the FP8 WMMA one is in use.

d4r::dx::rdna3::MModel* native_engine_m_rdna3(bool ldr, bool presetL)
{
    native_engine_m(ldr, presetL);
    return g_engineDx.modelMRdna3[presetL][ldr].get();
}

bool native_engine_available(bool ldr)
{
    if (ldr)
        std::call_once(g_engineDx.ldrOnce, [] { load_engine_dx_model(g_engineDx.modelLdr, "D4R_ENGINE_MODEL_LDR_DIR", L"\\engine\\k-ldr"); });
    else
        std::call_once(g_engineDx.once, [] { load_engine_dx_model(g_engineDx.model, "D4R_ENGINE_MODEL_DIR", L"\\engine\\k"); });
    return (ldr ? g_engineDx.modelLdr : g_engineDx.model) != nullptr;
}

static void engine_dx_submitted(void* owner, ID3D12Fence* fence, uint64_t value, uint64_t frequency)
{
    auto* state = static_cast<EngineDxFeature*>(owner);
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->pending)
        state->submitted(fence, value);
    if (state->pendingTiming >= 0)
    {
        auto& sample = state->timings[state->pendingTiming];
        sample.fence = fence; sample.value = value;
        sample.frequency = frequency;
        state->pendingTiming = -1;
    }
    state->pending = false;
}

static void engine_dx_discarded(void* owner)
{
    auto* state = static_cast<EngineDxFeature*>(owner);
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->pending)
        state->discarded();
    if (state->pendingTiming >= 0)
    {
        state->timings[state->pendingTiming].busy = false;
        state->pendingTiming = -1;
    }
    state->pending = false;
}

// A recording no hook will report (taken back from the tracker, or recorded without hooks): kept as executed
// without a fence, so its resources stay alive for the engine's unfenced frame count.

static void engine_dx_settle(EngineDxFeature& state)
{
    const bool forgotten = d4r_submit::forget(&state) != 0;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.pending && (forgotten || !g_engineDx.hooks))
    {
        state.submitted(nullptr, 0);
        state.pending = false;
    }
}

// Destroys released features once their frames have retired. Frames reported without a fence never retire on
// their own once the feature stops recording, so those features go after a few seconds.

static void reap_engine_dx()
{
    std::lock_guard<std::mutex> lock(g_engineDx.retiredMutex);
    const auto now = std::chrono::steady_clock::now();
    auto& retired = g_engineDx.retired;
    retired.erase(std::remove_if(retired.begin(), retired.end(), [&](const std::unique_ptr<EngineDxFeature>& state) {
        std::lock_guard<std::mutex> stateLock(state->mutex);
        return !state->pending && (state->idle() || now - state->released > std::chrono::seconds(5));
    }), retired.end());
}

void release_engine_dx(Feature& feature)
{
    if (feature.engineDx == nullptr)
        return;
    engine_dx_settle(*feature.engineDx);
    feature.engineDx->released = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(g_engineDx.retiredMutex);
        g_engineDx.retired.push_back(std::move(feature.engineDx));
    }
    reap_engine_dx();
}

// D4R_PROFILE stage names per D3D12 path: boundary 0 is the shim's, before the adapter's entry barriers, and the last
// one follows the adapter's exit barriers (restore); the engines' own boundaries lie between (FrameProfiler)

static const char* const kEngineDxStagesK[] = {"prepare", "setup", "exposure", "input", "enc0", "enc1", "enc2", "enc3",
    "enc4", "dec5", "dec4", "dec3", "dec2", "dec1", "dec0", "output", "finish", "restore"};

static const char* const kEngineDxStagesKHip[] = {"prepare", "setup", "exposure", "input", "external_network_wait",
    "output", "finish", "restore"};

static const char* const kEngineDxStagesM[] = {"prepare", "setup", "exposure", "input", "network", "expansion",
    "reconstruction", "downsample", "restore"};

static const char* const kEngineDxStagesMHip[] = {"prepare", "setup", "exposure", "input", "external_network_wait",
    "expansion", "reconstruction", "downsample", "restore"};

static const char* const kEngineDxStagesMRdna3[] = {"prepare", "setup", "exposure", "input", "network", "expansion",
    "reconstruction", "downsample", "finish"};

static const char* const kEngineDxStagesRR[] = {"prepare", "setup", "enc0", "enc1", "enc2", "enc3", "enc4", "enc5",
    "dec4", "dec3", "dec2", "dec1", "dec0", "output", "restore"};

static const char* const kEngineDxStagesRRHip[] = {"front", "external_network_wait", "back"};

// Resolve only completed submissions, without a CPU wait or queue flush. A
// discarded list never contributes a sample and an occupied slot is not reused.
// Returns the slot for this evaluation's boundaries (-1: not profiled); labels[i] names the stage ending at boundary
// i + 1 and must outlive the sample (string literals).

static int engine_dx_timing(EngineDxFeature& state, uint32_t frame, const char* preset, const char* const* labels,
                            uint32_t stages, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh)
{
    if (!profile_enabled() || !g_engineDx.hooks || stages >= EngineDxFeature::TimingQueries) return -1;
    constexpr uint32_t Q = EngineDxFeature::TimingQueries;
    if (!state.queries)
    {
        D3D12_QUERY_HEAP_DESC q{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, EngineDxFeature::TimingSlots * Q, 0};
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = uint64_t(EngineDxFeature::TimingSlots) * Q * 8;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(g.device->CreateQueryHeap(&q, IID_PPV_ARGS(&state.queries))) ||
            FAILED(g.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&state.readback))))
        {
            state.queries.Reset(); state.readback.Reset();
            logf("D4R_ENGINE_STAGE_PROFILE: GPU timestamps unavailable; inference continues without stage profiling");
            return -1;
        }
    }
    for (unsigned i = 0; i < state.timings.size(); ++i)
    {
        auto& t = state.timings[i];
        if (!t.busy || !t.fence || t.fence->GetCompletedValue() < t.value) continue;
        void* mapped = nullptr; D3D12_RANGE range{size_t(i) * Q * 8, size_t(i + 1) * Q * 8};
        if (t.frequency && profile_log_frame(t.frame) && SUCCEEDED(state.readback->Map(0, &range, &mapped)))
        {
            uint64_t stamps[Q]; std::memcpy(stamps, static_cast<char*>(mapped) + range.Begin, sizeof(stamps));
            D3D12_RANGE none{0, 0}; state.readback->Unmap(0, &none);
            auto ms = [&](uint32_t a, uint32_t b) { return double(stamps[b] - stamps[a]) * 1000.0 / double(t.frequency); };
            // a boundary without a timestamp (a stage that did not run) folds its time into the next stage
            std::string stagesText;
            uint32_t previous = 0;
            char field[96];
            for (uint32_t boundary = 1; boundary <= t.stages; ++boundary)
            {
                if (!(t.written & (1u << boundary))) continue;
                std::snprintf(field, sizeof(field), " %s=%.3f", t.labels[boundary - 1], ms(previous, boundary));
                stagesText += field;
                previous = boundary;
            }
            if ((t.written & 1u) && previous > 0)
                logf("D4R_ENGINE_STAGE_PROFILE preset=%s gpu_frame=%u render=%ux%u output=%ux%u gpu_eval=%.3f%s "
                     "cpu_record=%.3f game_interval=%.3f", t.preset, t.frame, t.rw, t.rh, t.ow, t.oh, ms(0, previous),
                     stagesText.c_str(), t.cpuRecord, t.gameInterval);
        }
        t.busy = false; t.fence.Reset();
    }
    const auto now = ProfileClock::now();
    const double interval = state.profileAt.time_since_epoch().count() ? profile_ms(state.profileAt, now) : -1.0;
    state.profileAt = now;
    for (unsigned i = 0; i < state.timings.size(); ++i)
        if (!state.timings[i].busy)
        {
            auto& t = state.timings[i];
            t.busy = true; t.frame = frame; t.preset = preset; t.labels = labels; t.stages = stages; t.written = 0;
            t.rw = rw; t.rh = rh; t.ow = ow; t.oh = oh;
            t.cpuRecord = -1; t.gameInterval = interval;
            state.profileRecordAt = now;
            return int(i);
        }
    return -1;
}

void engine_dx_mark(EngineDxFeature& state, int slot, ID3D12GraphicsCommandList* list, uint32_t boundary)
{
    if (slot < 0) return;
    auto& t = state.timings[slot];
    if (boundary > t.stages) return;
    list->EndQuery(state.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, uint32_t(slot) * EngineDxFeature::TimingQueries + boundary);
    t.written |= 1u << boundary;
}

// The last boundary, then the timestamps into the readback buffer from the list that records the frame's last work

static void engine_dx_end_timing(EngineDxFeature& state, int slot, ID3D12GraphicsCommandList* list)
{
    if (slot < 0) return;
    auto& t = state.timings[slot];
    engine_dx_mark(state, slot, list, t.stages);
    const uint32_t first = uint32_t(slot) * EngineDxFeature::TimingQueries;
    for (uint32_t boundary = 0; boundary <= t.stages; ++boundary)
        if (t.written & (1u << boundary))
            list->ResolveQueryData(state.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, first + boundary, 1,
                                   state.readback.Get(), uint64_t(first + boundary) * 8);
    t.cpuRecord = profile_ms(state.profileRecordAt, ProfileClock::now());
    state.pendingTiming = slot;
}

// Presets M and L on GPUs without FP8 WMMA (engine/d3d12/runtime_m_rdna3.h): the same frame, recorded into the game's
// command list with FP16 WMMA and software FP8 storage. Same contract as engine_evaluate_d3d12_m.

static bool engine_evaluate_d3d12_m_rdna3(Feature& feature, d4r::dx::rdna3::MModel& model, ID3D12GraphicsCommandList* list,
                                          ID3D12Resource* color, ID3D12Resource* depth, ID3D12Resource* motion,
                                          ID3D12Resource* output, const FrameParams& p, uint32_t frame)
{
    auto refuse = [&](const char* reason) {
        feature.engineRefused = true;
        logf("engine backend (D3D12, RDNA 3 M/L): %s; native feature %u rejected at frame %u", reason, feature.handle.Id, frame);
        release_engine_dx(feature);
        return false;
    };
    const bool ldr = (feature.flags & 0x1) == 0, presetL = feature.preset == 12;
    if (output == color || output == motion || output == depth)
        return refuse("input/output texture aliasing is not supported");
    if (p.colorBaseX || p.colorBaseY || p.depthBaseX || p.depthBaseY || p.mvBaseX || p.mvBaseY)
        return refuse("nonzero input texture origins are not implemented");
    std::call_once(g_engineDx.hooksOnce, [] {
        const char* failure = "";
        g_engineDx.hooks = d4r_submit::install(g.device, &failure);
        if (g_engineDx.hooks)
            logf("engine backend (D3D12): submission hooks installed");
        else
            logf("engine backend (D3D12): submission hooks unavailable (%s): frames retire by count", failure);
    });
    reap_engine_dx();
    const uint32_t capacityWidth = std::min(feature.outWidth, std::max(feature.width, p.renderWidth));
    const uint32_t capacityHeight = std::min(feature.outHeight, std::max(feature.height, p.renderHeight));
    bool reset = p.reset != 0 || feature.engineDx == nullptr;
    if (feature.engineDx != nullptr && feature.engineDx->upscalerRdna3 &&
        (p.renderWidth > feature.engineDx->upscalerRdna3->maxRenderWidth() || p.renderHeight > feature.engineDx->upscalerRdna3->maxRenderHeight()))
    {
        logf("engine backend (D3D12, RDNA 3 M/L): render size %ux%u beyond the engine's %ux%u: a new engine (history restarts)",
             p.renderWidth, p.renderHeight, feature.engineDx->upscalerRdna3->maxRenderWidth(), feature.engineDx->upscalerRdna3->maxRenderHeight());
        release_engine_dx(feature);
        reset = true;
    }
    try
    {
        if (feature.engineDx == nullptr)
        {
            auto state = std::make_unique<EngineDxFeature>();
            state->upscalerRdna3 = std::make_unique<d4r::dx::rdna3::GameUpscalerM>(model, feature.outWidth, feature.outHeight,
                                                                                    capacityWidth, capacityHeight);
            feature.engineDx = std::move(state);
            logf("engine backend (D3D12): feature %u runs preset %c in the game's command list (%ux%u -> %ux%u, capacity %ux%u%s), FP16 WMMA, no CUDA",
                 feature.handle.Id, presetL ? 'L' : 'M', p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight, capacityWidth,
                 capacityHeight, ldr ? ", LDR" : "");
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend (D3D12, RDNA 3 M/L): %s", error.what());
        return refuse("the engine could not be created");
    }
    EngineDxFeature& state = *feature.engineDx;
    engine_dx_settle(state); // the previous frame, when no hook reported it
    d4r::dx::rdna3::MGameFrame game;
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    game.color = {color, inputState}; game.motion = {motion, inputState}; game.depth = {depth, depthState};
    game.output = {output, outputState, DXGI_FORMAT_UNKNOWN, p.outputBaseX, p.outputBaseY};
    auto& s = game.settings;
    s.jitter[0] = p.jitterX; s.jitter[1] = p.jitterY;
    s.motionScale[0] = p.mvScaleX; s.motionScale[1] = p.mvScaleY;
    s.renderWidth = p.renderWidth; s.renderHeight = p.renderHeight;
    s.displayMotion = (feature.flags & 0x2) == 0; s.depthInverted = (feature.flags & 0x8) != 0;
    s.reset = reset;
    try
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if ((feature.flags & 0x4) != 0 && p.mvScaleX != 0.0f && p.mvScaleY != 0.0f)
        {
            s.motionOffset[0] = (p.jitterX - state.committedJitter[0]) / p.mvScaleX;
            s.motionOffset[1] = (p.jitterY - state.committedJitter[1]) / p.mvScaleY;
        }
        const int timing = engine_dx_timing(state, frame, presetL ? "L" : "M", kEngineDxStagesMRdna3,
                                            stage_count(kEngineDxStagesMRdna3), p.renderWidth, p.renderHeight,
                                            feature.outWidth, feature.outHeight);
        EngineDxMarks marks{&state, timing, 0};
        engine_dx_mark(state, timing, list, 0);
        state.upscalerRdna3->record(list, game, marks.profiler(1));
        engine_dx_end_timing(state, timing, list);
        state.pendingJitter[0] = p.jitterX; state.pendingJitter[1] = p.jitterY;
        state.pending = true;
    }
    catch (const std::exception& error)
    {
        logf("engine backend (D3D12, RDNA 3 M/L): %s", error.what());
        return refuse("the frame could not be recorded");
    }
    if (g_engineDx.hooks)
        d4r_submit::track(list, {&state, engine_dx_submitted, engine_dx_discarded});
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded on D3D12 (RDNA 3), preset %c (reset %d, render %ux%u)", frame,
             presetL ? 'L' : 'M', s.reset, p.renderWidth, p.renderHeight);
    return true;
}

#ifdef D4R_ENGINE_D3DHIP

// d3dhip for presets M and L (engine/d3d12/hip_net_bridge.cpp): the model's native HIP layers run the network between
// the D3D12 engine's front and back. The game's command list is split there (d4r_d3d12_split.h): the front is
// submitted, its fence retires on the CPU, the HIP graph runs, and the back follows on the same queue. HDR and LDR
// packages use the same network; LDR only skips the exposure stages in front of it. L's front and back are its own
// plane-layout stages (enc0_l_planes / dec0_l_planes).

static bool engine_m_d3dhip_package(const std::string& directory, const d4r::dx::MModel& model)
{
    const char* net = std::getenv("D4R_DX_NET");
    if (net != nullptr && !std::strcmp(net, "dxil")) return false;
    const bool presetL = model.isPresetL();
    auto exists = [&](const char* name) {
        return GetFileAttributesA((directory + "/" + name).c_str()) != INVALID_FILE_ATTRIBUTES;
    };
    return exists("hipnet.bin") && exists("hip") && exists(presetL ? "enc0_l_planes.dxil" : "enc0_m_planes.dxil") &&
           exists(presetL ? "dec0_l_planes.dxil" : "dec0_m_planes.dxil");
}

// The split hooks patch the command lists' vtable: installed once, only when the HIP route is used.

static bool engine_dx_split_ready()
{
    static std::once_flag once;
    static bool ready = false;
    std::call_once(once, [] {
        const char* failure = "";
        ready = d4r_split::install(g.device, &failure);
        if (ready)
            logf("engine backend (D3D12, d3dhip): command-list split available");
        else
            logf("engine backend (D3D12, d3dhip): command-list split unavailable (%s)", failure);
    });
    return ready;
}

// The front fence of a d3dhip feature: shared with HIP for the GPU-side handoff unless D4R_D3DHIP_SYNC=cpu, the bridge
// is older or the import fails (then the CPU-serialized one). Throws when no fence can be created at all.

static void engine_dx_hip_fence(EngineDxFeature& state)
{
    const bool gpu = env_string("D4R_D3DHIP_SYNC") != "cpu" &&
                     (state.hipRr != nullptr ? g_hipApi.rrUseFence && g_hipApi.rrLaunchAfter
                                             : g_hipApi.useFence && g_hipApi.launchAfter);
    if (FAILED(g.device->CreateFence(0, gpu ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE,
                                     IID_PPV_ARGS(state.front.GetAddressOf()))))
        throw std::runtime_error("the front fence could not be created");
    if (!gpu)
        return;
    char error[256] = {};
    state.hipFenced = (state.hipRr != nullptr ? g_hipApi.rrUseFence(state.hipRr, state.front.Get(), error, sizeof(error))
                                              : g_hipApi.useFence(state.hip, state.front.Get(), error, sizeof(error))) == 0;
    logf("engine backend (D3D12, d3dhip): %s", state.hipFenced ? "GPU-fence handoff (no CPU wait between the parts)"
                                                                : (std::string("CPU-serialized handoff: ") + error).c_str());
}

// d4r_split's between-parts hook: the front has been submitted on `queue`. Wait for it on the CPU, run the network's
// graph (blocking, the documented d3dhip round trip), and return; the split then runs the back.

static void engine_dx_hip_network(void* owner, ID3D12CommandQueue* queue)
{
    auto& state = *static_cast<EngineDxFeature*>(owner);
    std::lock_guard<std::mutex> lock(state.mutex);
    if ((!state.hip && !state.hipRr) || !state.front)
        return;
    if (state.hipFenced)
    {
        // GPU-side: the queue signals the front, HIP waits for it, runs the network and signals; the queue's Wait
        // keeps the back (the split's second part) behind the network
        const uint64_t front = ++state.frontValue, network = ++state.frontValue;
        char error[256] = {};
        if (FAILED(queue->Signal(state.front.Get(), front)) ||
            (state.hipRr != nullptr ? g_hipApi.rrLaunchAfter(state.hipRr, front, network, error, sizeof(error))
                                    : g_hipApi.launchAfter(state.hip, front, network, error, sizeof(error))) != 0 ||
            FAILED(queue->Wait(state.front.Get(), network)))
        {
            // the back must not wait for a signal that never comes: release it, then fall back to the CPU handoff
            state.front->Signal(network);
            state.hipFenced = false;
            logf("engine backend (D3D12, d3dhip): the GPU-fence handoff failed (%s); CPU-serialized from the next frame",
                 error[0] != '\0' ? error : "a fence operation failed");
        }
        return;
    }
    const uint64_t value = ++state.frontValue;
    if (FAILED(queue->Signal(state.front.Get(), value)))
    {
        logf("engine backend (D3D12, d3dhip): signalling the front fence failed; the network did not run");
        return;
    }
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool done = event != nullptr && SUCCEEDED(state.front->SetEventOnCompletion(value, event)) &&
                      WaitForSingleObject(event, 2000) == WAIT_OBJECT_0;
    if (event != nullptr) CloseHandle(event);
    if (!done)
    {
        logf("engine backend (D3D12, d3dhip): the front did not complete within 2 s; the network did not run");
        return;
    }
    char error[256] = {};
    if ((state.hipRr != nullptr ? g_hipApi.rrRun(state.hipRr, error, sizeof(error))
                                : g_hipApi.run(state.hip, error, sizeof(error))) != 0)
        logf("engine backend (D3D12, d3dhip): %s", error[0] != '\0' ? error : "the network failed");
}

#endif

// Preset M on the native D3D12 engine (engine/d3d12/runtime_m_d3d12.h). NGX measures the exposure for this preset, so
// the exposure flags and texture are ignored, as on the Vulkan engine (engine_evaluate_m).

static bool engine_evaluate_d3d12_m(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                                    ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                                    const FrameParams& p, uint32_t frame)
{
    auto refuse = [&](const char* reason) {
        feature.engineRefused = true;
        logf("engine backend (D3D12, M): %s; native feature %u rejected at frame %u", reason, feature.handle.Id, frame);
        release_engine_dx(feature);
        return false;
    };
    const bool ldr = (feature.flags & 0x1) == 0, presetL = feature.preset == 12;
    d4r::dx::MModel* const model = native_engine_m(ldr, presetL);
    if (model == nullptr)
        if (d4r::dx::rdna3::MModel* const rdna3 = native_engine_m_rdna3(ldr, presetL))
            return engine_evaluate_d3d12_m_rdna3(feature, *rdna3, list, color, depth, motion, output, p, frame);
    if (model == nullptr)
        return refuse(presetL ? (ldr ? "no D3D12 L model (engine\\l-ldr) for a game without the HDR flag" : "no D3D12 L model (engine\\l)")
                              : (ldr ? "no D3D12 M model (engine\\m-ldr) for a game without the HDR flag" : "no D3D12 M model (engine\\m)"));
    const bool displayMotion = (feature.flags & 0x2) == 0;
    if (displayMotion && !model->hasDisplayMotion())
        return refuse(presetL ? "display-resolution motion vectors need an L model rebuilt by the current compile_m.py"
                              : "display-resolution motion vectors need an M model rebuilt by the current compile_m.py");
    if (output == color || output == motion || output == depth)
        return refuse("input/output texture aliasing is not supported");
    if (p.colorBaseX || p.colorBaseY || p.depthBaseX || p.depthBaseY || p.mvBaseX || p.mvBaseY || p.outputBaseX || p.outputBaseY)
        return refuse("texture rectangles with nonzero origins are not implemented for preset M");
    std::call_once(g_engineDx.hooksOnce, [] {
        const char* failure = "";
        g_engineDx.hooks = d4r_submit::install(g.device, &failure);
        if (g_engineDx.hooks)
            logf("engine backend (D3D12): submission hooks installed");
        else
            logf("engine backend (D3D12): submission hooks unavailable (%s): frames retire by count", failure);
    });
    reap_engine_dx();

    d4r::dx::GameFrameM game;
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    struct Source
    {
        ID3D12Resource* resource;
        d4r::dx::GameTexture* texture;
        uint32_t width, height;
        D3D12_RESOURCE_STATES state;
    };
    const Source sources[4] = {
        {color, &game.color, p.renderWidth, p.renderHeight, inputState},
        {depth, &game.depth, p.renderWidth, p.renderHeight, depthState},
        {motion, &game.motion, displayMotion ? feature.outWidth : p.renderWidth, displayMotion ? feature.outHeight : p.renderHeight, inputState},
        {output, &game.output, feature.outWidth, feature.outHeight, outputState},
    };
    for (const Source& source : sources)
    {
        if (source.resource == nullptr)
            return refuse("a texture is missing");
        *source.texture = {source.resource, source.state, DXGI_FORMAT_UNKNOWN, 0, 0};
        const char* reason = nullptr;
        if (source.texture != &game.output && !d4r::dx::gameInputTexture(g.device, *source.texture, &reason))
            return refuse(reason);
        D3D12_RESOURCE_DESC desc;
        source.resource->GetDesc(&desc);
        if (source.texture == &game.output && !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            return refuse("the output does not allow unordered access");
        if (source.width > desc.Width || source.height > desc.Height)
            return refuse("a texture does not cover its rectangle");
    }
    auto& s = game.settings;
    s.renderWidth = p.renderWidth; s.renderHeight = p.renderHeight;
    s.jitter[0] = p.jitterX; s.jitter[1] = p.jitterY;
    s.motionScale[0] = p.mvScaleX; s.motionScale[1] = p.mvScaleY;
    s.displayMotion = displayMotion;
    if ((feature.flags & 0x4) != 0 && p.mvScaleX != 0.0f && p.mvScaleY != 0.0f)
    {
        s.motionOffset[0] = (p.jitterX - feature.engineJitter[0]) / p.mvScaleX;
        s.motionOffset[1] = (p.jitterY - feature.engineJitter[1]) / p.mvScaleY;
    }
    s.reset = p.reset != 0 || feature.engineDx == nullptr;
    s.depthInverted = (feature.flags & 0x8) != 0;
    // capacity: the render size the feature was created with (NGX's maximum), at least this frame's; a larger frame
    // starts a new engine at its size
    if (feature.engineDx != nullptr && feature.engineDx->upscalerM &&
        (p.renderWidth > feature.engineDx->upscalerM->maxRenderWidth() || p.renderHeight > feature.engineDx->upscalerM->maxRenderHeight()))
    {
        logf("engine backend (D3D12, M): render size %ux%u beyond the engine's %ux%u: a new engine (history restarts)",
             p.renderWidth, p.renderHeight, feature.engineDx->upscalerM->maxRenderWidth(), feature.engineDx->upscalerM->maxRenderHeight());
        release_engine_dx(feature);
        s.reset = true;
    }
#ifdef D4R_ENGINE_D3DHIP
    // the HIP network's code objects are built for one render size: a different one starts a new engine at its size
    if (feature.engineDx != nullptr && feature.engineDx->hip &&
        (p.renderWidth != feature.engineDx->hipWidth || p.renderHeight != feature.engineDx->hipHeight))
    {
        logf("engine backend (D3D12, d3dhip): render size %ux%u differs from the network's %ux%u: a new engine (history restarts)",
             p.renderWidth, p.renderHeight, feature.engineDx->hipWidth, feature.engineDx->hipHeight);
        release_engine_dx(feature);
        s.reset = true;
    }
    // a failed split leaves this frame's engine external: the next frame is created with the DXIL network instead
    if (feature.engineMHipFailed && feature.engineDx != nullptr && feature.engineDx->hip)
    {
        release_engine_dx(feature);
        s.reset = true;
    }
#endif
    try
    {
        if (feature.engineDx == nullptr)
        {
            const uint32_t capacityWidth = std::min(feature.outWidth, std::max(feature.width, p.renderWidth));
            const uint32_t capacityHeight = std::min(feature.outHeight, std::max(feature.height, p.renderHeight));
#ifdef D4R_ENGINE_D3DHIP
            if (!feature.engineMHipFailed)
            {
                const std::string directory = engine_m_directory(ldr, presetL);
                if (engine_m_d3dhip_package(directory, *model) && engine_dx_split_ready() &&
                    (feature.splitFence != nullptr || create_native_split(feature)) && load_hip_net())
                {
                    try
                    {
                        // the input stage writes the token grid of this render size; the network's two buffers both hold it
                        const uint64_t bytes = d4r::dx::MEngine::networkBufferBytes(p.renderWidth, p.renderHeight);
                        auto state = std::make_unique<EngineDxFeature>();
                        char error[256] = {};
                        state->hip = g_hipApi.create(g.device, directory.c_str(), p.renderWidth, p.renderHeight, bytes,
                                                     bytes, error, sizeof(error));
                        if (state->hip == nullptr)
                            throw std::runtime_error(error[0] != '\0' ? error : "the HIP network could not be created");
                        // the state destroys the network if the rest of the creation throws
                        engine_dx_hip_fence(*state);
                        state->hipTokens = g_hipApi.tokens(state->hip);
                        state->hipHead = g_hipApi.head(state->hip);
                        const d4r::dx::ExternalNetworkDx external{state->hipTokens.Get(), state->hipHead.Get()};
                        state->upscalerM = std::make_unique<d4r::dx::GameUpscalerM>(*model, feature.outWidth, feature.outHeight,
                                                                                    p.renderWidth, p.renderHeight, &external);
                        state->hipWidth = p.renderWidth;
                        state->hipHeight = p.renderHeight;
                        feature.engineDx = std::move(state);
                        logf("engine backend (D3D12): feature %u runs preset %c in the game's command list (%ux%u -> %ux%u) with "
                             "its network on the model's HIP layers (d3dhip)",
                             feature.handle.Id, presetL ? 'L' : 'M', p.renderWidth, p.renderHeight, feature.outWidth,
                             feature.outHeight);
                    }
                    catch (const std::exception& error)
                    {
                        logf("engine backend (D3D12, d3dhip): %s; using the DXIL network", error.what());
                    }
                }
            }
#endif
            if (feature.engineDx == nullptr)
            {
                auto state = std::make_unique<EngineDxFeature>();
                state->upscalerM = std::make_unique<d4r::dx::GameUpscalerM>(*model, feature.outWidth, feature.outHeight,
                                                                            capacityWidth, capacityHeight);
                feature.engineDx = std::move(state);
                logf("engine backend (D3D12): feature %u runs preset %c in the game's command list (%ux%u -> %ux%u, capacity %ux%u%s)",
                     feature.handle.Id, presetL ? 'L' : 'M', p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight, capacityWidth,
                     capacityHeight, ldr ? ", LDR" : "");
            }
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend (D3D12, M): %s", error.what());
        return refuse("the engine could not be created");
    }
    EngineDxFeature& state = *feature.engineDx;
    engine_dx_settle(state); // the previous frame, when no hook reported it
    try
    {
        int timing = -1;
#ifdef D4R_ENGINE_D3DHIP
        const bool hip = state.hip != nullptr;
#else
        const bool hip = false;
#endif
        EngineDxMarks marks{&state, -1, 0};
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            timing = engine_dx_timing(state, frame, presetL ? "L" : "M", hip ? kEngineDxStagesMHip : kEngineDxStagesM,
                                      hip ? stage_count(kEngineDxStagesMHip) : stage_count(kEngineDxStagesM),
                                      p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight);
            marks.slot = timing;
            engine_dx_mark(state, timing, list, 0);
#ifdef D4R_ENGINE_D3DHIP
            if (hip) state.upscalerM->recordFront(list, game, marks.profiler(1));
            else
#endif
            state.upscalerM->record(list, game, marks.profiler(1));
        }
        // the end timestamp goes into the list that records the frame's last work (the split's second part, if any)
        auto endTiming = [&](ID3D12GraphicsCommandList* recorded) { engine_dx_end_timing(state, timing, recorded); };
#ifdef D4R_ENGINE_D3DHIP
        if (hip)
        {
            // the split's mutex is taken here without state.mutex: the between-parts hook takes them the other way
            ID3D12GraphicsCommandList* back = nullptr;
            const bool split = native_split_frame(feature, list, frame, engine_dx_hip_network, &state, &back);
            std::lock_guard<std::mutex> lock(state.mutex);
            if (split)
            {
                state.upscalerM->recordBack(back, marks.profiler(5));
                endTiming(back);
            }
            else
            {
                // no second part: complete the frame in the game's list (the network's head is the previous frame's)
                // and have the next frame's engine run the DXIL network instead
                logf("engine backend (D3D12, d3dhip): feature %u could not split its list at frame %u; the DXIL engine replaces it",
                     feature.handle.Id, frame);
                feature.engineMHipFailed = true;
                state.upscalerM->recordBack(list, marks.profiler(5));
                endTiming(list);
            }
            state.pending = true;
        }
        else
#endif
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            endTiming(list);
            state.pending = true;
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend (D3D12, M): %s", error.what());
        return refuse("the frame could not be recorded");
    }
    if (g_engineDx.hooks)
        d4r_submit::track(list, {&state, engine_dx_submitted, engine_dx_discarded});
    feature.engineJitter[0] = p.jitterX;
    feature.engineJitter[1] = p.jitterY;
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded on D3D12, preset %c (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f, render %ux%u)", frame,
             presetL ? 'L' : 'M', p.jitterX, p.jitterY, s.reset, p.mvScaleX, p.mvScaleY, p.renderWidth, p.renderHeight);
    return true;
}

static bool engine_evaluate_d3d12_rr(Feature &feature, ID3D12GraphicsCommandList *list, ID3D12Resource *color,
                                     ID3D12Resource *depth, ID3D12Resource *motion, ID3D12Resource *output,
                                     void *parameters, const FrameParams &p, uint32_t frame)
{
    auto refuse = [&](const char *why) {
        feature.engineRefused = true;
        logf("engine backend (D3D12 RR): feature %u frame %u refused: %s", feature.handle.Id, frame, why);
        release_engine_dx(feature);
        return false;
    };
    auto *model = native_engine_rr(feature.preset);
    if (!model)
        return refuse("a matching native D/E/F model is missing");
    if ((feature.flags & 3) != 3 || (feature.flags & 4))
        return refuse("requires HDR and unjittered render-resolution motion");
    if (get_resource(parameters, "DLSSD.OutputAlpha"))
        return refuse("RR output alpha is unsupported");
    if (!p.auxMatrixSet[0] || !p.auxMatrixSet[1])
        return refuse("camera matrices missing");
    if (p.colorBaseX || p.colorBaseY || p.depthBaseX || p.depthBaseY || p.mvBaseX || p.mvBaseY || p.outputBaseX ||
        p.outputBaseY)
        return refuse("nonzero texture origins are unsupported");
    for (auto &origin : p.auxSubrect)
        if (origin[0] || origin[1])
            return refuse("nonzero guide origins are unsupported");
    d4r::dx::GameFrameR game;
    ID3D12Resource *normals = get_resource(parameters, "GBuffer.Normals");
    ID3D12Resource *roughness = get_resource(parameters, "GBuffer.Roughness");
    const bool packed = p.auxScalar[AuxRoughnessMode] == 1 &&
                        (model->preset() == 6 || !roughness);
    if (packed)
        roughness = normals;
    ID3D12Resource *diffuse = get_resource(parameters, "DLSS.Input.DiffuseAlbedo");
    if (!diffuse)
        diffuse = get_resource(parameters, "GBuffer.DiffuseAlbedo");
    ID3D12Resource *specular = get_resource(parameters, "DLSS.Input.SpecularAlbedo");
    if (!specular)
        specular = get_resource(parameters, "GBuffer.SpecularAlbedo");
    if (!normals || !roughness || !diffuse || !specular)
        return refuse("missing required RR guides");
    auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    struct Source
    {
        ID3D12Resource *resource;
        d4r::dx::GameTexture *texture;
        D3D12_RESOURCE_STATES state;
    };
    Source sources[] = {
        {color, &game.color, inputState},
        {depth, &game.depth, depthState},
        {motion, &game.motion, inputState},
        {normals, &game.normals, inputState},
        {roughness, &game.roughness, inputState},
        {diffuse, &game.diffuseAlbedo, inputState},
        {specular, &game.specularAlbedo, inputState},
        {get_resource(parameters, "DLSSD.SpecularHitDistance"), &game.specularHitDistance, inputState},
        {get_resource(parameters, "DLSS.DisocclusionMask"), &game.disocclusionMask, inputState},
        {get_resource(parameters, "DLSSD.ScreenSpaceSubsurfaceScatteringGuide"), &game.subsurfaceGuide, inputState},
        {get_resource(parameters, "DLSSD.ColorBeforeParticles"), &game.colorBeforeParticles, inputState},
        {output, &game.output, outputState}};
    for (auto &source : sources)
    {
        if (!source.resource)
            continue;
        *source.texture = {source.resource, source.state, DXGI_FORMAT_UNKNOWN, 0, 0};
        const char *why = nullptr;
        if (source.texture != &game.output && !d4r::dx::gameInputTexture(g.device, *source.texture, &why))
            return refuse(why);
    }
    auto &s = game.settings;
    s.jitter[0] = p.jitterX;
    s.jitter[1] = p.jitterY;
    s.motionScale[0] = p.mvScaleX;
    s.motionScale[1] = p.mvScaleY;
    s.preExposure = p.preExposure;
    s.frameTimeMs = p.frameTime;
    s.reset = p.reset != 0 || !feature.engineDx;
    s.depthInverted = (feature.flags & 8) != 0;
    s.hardwareDepth = p.auxScalar[AuxHardwareDepth] != 0;
    s.roughnessPacked = packed;
    std::copy_n(p.auxMatrix[0], 16, s.worldToView);
    std::copy_n(p.auxMatrix[1], 16, s.viewToClip);
    std::call_once(g_engineDx.hooksOnce, [] {
        const char *why = "";
        g_engineDx.hooks = d4r_submit::install(g.device, &why);
        if (!g_engineDx.hooks)
            logf("D3D12 RR submission hooks: %s", why);
    });
    if (!g_engineDx.hooks)
        return refuse("RR requires submission hooks to retire resources");
    reap_engine_dx();
    if (feature.engineDx &&
        (feature.engineDx->rrWidth != p.renderWidth || feature.engineDx->rrHeight != p.renderHeight))
    {
        release_engine_dx(feature);
        s.reset = true;
    }
#ifdef D4R_ENGINE_D3DHIP
    // a failed split leaves this frame's engine external: the next frame is created with the DXIL network instead
    if (feature.engineMHipFailed && feature.engineDx && feature.engineDx->hipRr)
    {
        release_engine_dx(feature);
        s.reset = true;
    }
#endif
    try
    {
#ifdef D4R_ENGINE_D3DHIP
        // d3dhip for preset F: the network's fourteen stages as the package's HIP code objects (hip/rr2_*.hsaco,
        // engine/d3d12/hip_net_rr.h) between the engine's front and back, which run in the game's list split around
        // them, as presets M and L do. D4R_DX_NET=dxil or D4R_RR_DX_NET=dxil keeps the DXIL network.
        if (!feature.engineDx && model->preset() == 6 && !feature.engineMHipFailed && env_string("D4R_DX_NET") != "dxil" &&
            env_string("D4R_RR_DX_NET") != "dxil")
        {
            const std::string directory = engine_model_directory("D4R_ENGINE_MODEL_RR_F_DIR", L"\\engine\\rr-f");
            if (GetFileAttributesA((directory + "/hip/rr2_dec0_0.hsaco").c_str()) != INVALID_FILE_ATTRIBUTES &&
                engine_dx_split_ready() && (feature.splitFence != nullptr || create_native_split(feature)) &&
                load_hip_net() && g_hipApi.rrCreate != nullptr)
            {
                try
                {
                    auto state = std::make_unique<EngineDxFeature>();
                    state->denoiser = std::make_unique<d4r::dx::GameDenoiser>(*model, feature.outWidth, feature.outHeight,
                                                                              p.renderWidth, p.renderHeight, true);
                    char error[256] = {};
                    state->hipRr = g_hipApi.rrCreate(g.device, directory.c_str(), 1, p.renderWidth, p.renderHeight,
                                                     feature.outWidth, feature.outHeight, state->denoiser->arena(),
                                                     state->denoiser->arenaBytes(), error, sizeof(error));
                    if (state->hipRr == nullptr)
                        throw std::runtime_error(error[0] != '\0' ? error : "the HIP network could not be created");
                    engine_dx_hip_fence(*state);
                    state->denoiser->setExternalNetwork(
                        static_cast<const d4r::dx::ExternalNetworkRr *>(g_hipApi.rrExternal(state->hipRr)));
                    state->rrWidth = p.renderWidth;
                    state->rrHeight = p.renderHeight;
                    feature.engineDx = std::move(state);
                    logf("engine backend (D3D12 RR F): feature %u runs in the game's command list (%ux%u -> %ux%u) with "
                         "its network on the model's HIP layers (d3dhip)",
                         feature.handle.Id, p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight);
                }
                catch (const std::exception &error)
                {
                    logf("engine backend (D3D12 RR F, d3dhip): %s; using the DXIL network", error.what());
                }
            }
        }
#endif
        if (!feature.engineDx)
        {
            auto state = std::make_unique<EngineDxFeature>();
            state->denoiser = std::make_unique<d4r::dx::GameDenoiser>(*model, feature.outWidth, feature.outHeight,
                                                                      p.renderWidth, p.renderHeight);
            state->rrWidth = p.renderWidth;
            state->rrHeight = p.renderHeight;
            feature.engineDx = std::move(state);
        }
        auto &state = *feature.engineDx;
        engine_dx_settle(state);
#ifdef D4R_ENGINE_D3DHIP
        const bool hip = state.hipRr != nullptr;
#else
        const bool hip = false;
#endif
        int timing = -1;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            const char *name = feature.preset == 6 ? "RR_F" : (feature.preset == 4 ? "RR_D" : "RR_E");
            timing = hip ? engine_dx_timing(state, frame, name, kEngineDxStagesRRHip, stage_count(kEngineDxStagesRRHip),
                                            p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight)
                         : engine_dx_timing(state, frame, name, kEngineDxStagesRR, stage_count(kEngineDxStagesRR),
                                            p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight);
            EngineDxMarks marks{&state, timing, 0};
            engine_dx_mark(state, timing, list, 0);
            if (hip)
            {
                state.denoiser->record(list, game);
                engine_dx_mark(state, timing, list, 1);
            }
            else
            {
                state.denoiser->record(list, game, marks.profiler(1));
                engine_dx_end_timing(state, timing, list);
                state.pending = true;
            }
        }
#ifdef D4R_ENGINE_D3DHIP
        if (hip)
        {
            // the split's mutex is taken here without state.mutex: the between-parts hook takes them the other way
            ID3D12GraphicsCommandList *back = nullptr;
            const bool split = native_split_frame(feature, list, frame, engine_dx_hip_network, &state, &back);
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!split)
            {
                // no second part: complete the frame in the game's list (the network's results are an earlier
                // frame's) and have the next frame's engine run the DXIL network instead
                logf("engine backend (D3D12 RR F, d3dhip): feature %u could not split its list at frame %u; the DXIL "
                     "engine replaces it", feature.handle.Id, frame);
                feature.engineMHipFailed = true;
                back = list;
            }
            engine_dx_mark(state, timing, back, 2);
            state.denoiser->recordBack(back);
            engine_dx_end_timing(state, timing, back);
            state.pending = true;
        }
#endif
        d4r_submit::track(list, {&state, engine_dx_submitted, engine_dx_discarded});
    }
    catch (const std::exception &e)
    {
        return refuse(e.what());
    }
    if (frame <= 3)
        logf("engine frame %u recorded on D3D12, RR preset %c", frame, char('D' + feature.preset - 4));
    return true;
}

static bool engine_evaluate_d3d12(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                                  ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                                  ID3D12Resource* exposure, const FrameParams& p, uint32_t frame)
{
    auto refuse = [&](const char* reason) {
        feature.engineRefused = true;
        if (feature.nativeD3d12)
            logf("engine backend (D3D12): %s; native feature %u rejected at frame %u", reason, feature.handle.Id, frame);
        else
            logf("engine backend (D3D12): %s; feature %u uses the CUDA backend from frame %u", reason, feature.handle.Id, frame);
        if (feature.engineDx != nullptr)
            logf("engine backend (D3D12): the temporal history restarts in the CUDA backend");
        release_engine_dx(feature);
        return false;
    };
    if (!feature.rayReconstruction && (feature.preset == 13 || feature.preset == 12))
        return engine_evaluate_d3d12_m(feature, list, color, depth, motion, output, p, frame);
    if (feature.rayReconstruction || feature.preset != 11)
        return refuse("the native D3D12 engine runs presets K, L and M Super Resolution only");
    // Without the HDR flag NGX runs its LDR input and output kernels: a model directory of its own.
    const bool ldr = (feature.flags & 0x1) == 0;
    if (ldr)
        std::call_once(g_engineDx.ldrOnce, [] { load_engine_dx_model(g_engineDx.modelLdr, "D4R_ENGINE_MODEL_LDR_DIR", L"\\engine\\k-ldr"); });
    else
        std::call_once(g_engineDx.once, [] { load_engine_dx_model(g_engineDx.model, "D4R_ENGINE_MODEL_DIR", L"\\engine\\k"); });
    d4r::dx::Model* const model = ldr ? g_engineDx.modelLdr.get() : g_engineDx.model.get();
    if (model == nullptr)
        return refuse(ldr ? "no D3D12 LDR model (engine\\k-ldr) for a game without the HDR flag" : "no D3D12 model (engine\\k)");
    const bool displayMotion = (feature.flags & 0x2) == 0;
    const bool autoExposure = (feature.flags & 0x40) != 0;
    const bool gameExposure = !ldr && !autoExposure;
    if (const char* reason = engine_k_refusal(*model, p, ldr, displayMotion, autoExposure, (feature.flags & 0x8) != 0))
        return refuse(reason);
    if (output == color || output == motion || output == depth)
        return refuse("input/output texture aliasing is not supported");
    std::call_once(g_engineDx.hooksOnce, [] {
        const char* failure = "";
        g_engineDx.hooks = d4r_submit::install(g.device, &failure);
        if (g_engineDx.hooks)
            logf("engine backend (D3D12): submission hooks installed");
        else
            logf("engine backend (D3D12): submission hooks unavailable (%s): frames retire by count", failure);
    });
    reap_engine_dx();

    d4r::dx::GameFrame game;
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    struct Source
    {
        ID3D12Resource* resource;
        d4r::dx::GameTexture* texture;
        uint32_t x, y, width, height;
        D3D12_RESOURCE_STATES state;
    };
    const Source sources[5] = {
        {color, &game.color, p.colorBaseX, p.colorBaseY, p.renderWidth, p.renderHeight, inputState},
        {depth, &game.depth, p.depthBaseX, p.depthBaseY, p.renderWidth, p.renderHeight, depthState},
        {motion, &game.motion, p.mvBaseX, p.mvBaseY, displayMotion ? feature.outWidth : p.renderWidth,
         displayMotion ? feature.outHeight : p.renderHeight, inputState},
        {output, &game.output, p.outputBaseX, p.outputBaseY, feature.outWidth, feature.outHeight, outputState},
        {exposure, &game.exposure, 0, 0, 1, 1, inputState},
    };
    for (size_t index = 0; index < (gameExposure ? 5u : 4u); ++index) // sources[4] (exposure): game exposure only
    {
        const Source& source = sources[index];
        if (source.resource == nullptr)
            return refuse("a texture is missing");
        *source.texture = {source.resource, source.state, DXGI_FORMAT_UNKNOWN, source.x, source.y};
        const char* reason = nullptr;
        const bool usable = source.texture == &game.output
            ? d4r::dx::gameOutputTexture(g.device, *source.texture, *model, &reason)
            : d4r::dx::gameInputTexture(g.device, *source.texture, &reason);
        if (!usable)
            return refuse(reason);
        D3D12_RESOURCE_DESC desc;
        source.resource->GetDesc(&desc);
        if (source.x + source.width > desc.Width || source.y + source.height > desc.Height)
            return refuse("a texture does not cover its rectangle");
    }
#ifdef D4R_ENGINE_D3DHIP
    // a failed split leaves this frame's engine external: the next frame is created with the DXIL network instead
    if (feature.engineKHipFailed && feature.engineDx != nullptr && feature.engineDx->hip)
        release_engine_dx(feature);
#endif
    game.settings = engine_k_settings(feature, p, ldr, displayMotion, autoExposure, feature.engineDx == nullptr);
    try
    {
#ifdef D4R_ENGINE_D3DHIP
        // D4R_D3DHIP_K=1: the network on the model's native HIP layers (compile_k.py --hip; engine\k\hipnet.bin and
        // hip\), between the engine's front and back in the game's split list. Faster than the DXIL network where
        // AMD's driver compiles that network poorly (26.10.1 on RDNA3) and on RDNA4 (RX 9070 XT, 26.10.1: 1.30
        // against 1.38 ms at 720p -> 1440p, 2.70 against 2.77 at 1080p -> 4K), slower on others. Default (2): on
        // with an RDNA4 GPU in the fast set ([Kernels] PreferAccuracy = false) when the package has the HIP layers;
        // 0 never, 1 on any GPU.
        const unsigned d3dhipK = env_uint("D4R_D3DHIP_K", 2);
        if (feature.engineDx == nullptr && !feature.engineKHipFailed &&
            (d3dhipK == 1 || (d3dhipK == 2 && env_uint("D4R_PREFER_ACCURACY", 0) == 0 && rdna4_installed())))
        {
            const std::string directory = ldr ? engine_model_directory("D4R_ENGINE_MODEL_LDR_DIR", L"\\engine\\k-ldr")
                                              : engine_model_directory("D4R_ENGINE_MODEL_DIR", L"\\engine\\k");
            if (GetFileAttributesA((directory + "/hipnet.bin").c_str()) == INVALID_FILE_ATTRIBUTES)
                logf("engine backend (D3D12, d3dhip): %s has no HIP layers (hipnet.bin); preset K keeps the DXIL network",
                     directory.c_str());
            else if (engine_dx_split_ready() && (feature.splitFence != nullptr || create_native_split(feature)) &&
                     load_hip_net())
            {
                try
                {
                    auto state = std::make_unique<EngineDxFeature>();
                    char error[256] = {};
                    state->hip = g_hipApi.create(g.device, directory.c_str(), feature.outWidth, feature.outHeight,
                                                 d4r::dx::GameUpscaler::networkTokenBytes(feature.outWidth, feature.outHeight),
                                                 d4r::dx::GameUpscaler::networkHeadBytes(feature.outWidth, feature.outHeight),
                                                 error, sizeof(error));
                    if (state->hip == nullptr)
                        throw std::runtime_error(error[0] != '\0' ? error : "the HIP network could not be created");
                    engine_dx_hip_fence(*state);
                    state->hipTokens = g_hipApi.tokens(state->hip);
                    state->hipHead = g_hipApi.head(state->hip);
                    const d4r::dx::ExternalNetworkDx external{state->hipTokens.Get(), state->hipHead.Get()};
                    state->upscaler = std::make_unique<d4r::dx::GameUpscaler>(*model, feature.outWidth, feature.outHeight,
                                                                              feature.outWidth, feature.outHeight, &external);
                    feature.engineDx = std::move(state);
                    logf("engine backend (D3D12): feature %u runs preset K in the game's command list (%ux%u -> %ux%u%s) "
                         "with its network on the model's HIP layers (d3dhip)",
                         feature.handle.Id, p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight,
                         ldr ? ", LDR" : "");
                }
                catch (const std::exception& error)
                {
                    logf("engine backend (D3D12, d3dhip): %s; preset K keeps the DXIL network", error.what());
                    feature.engineKHipFailed = true;
                }
            }
        }
#endif
        if (feature.engineDx == nullptr)
        {
            auto state = std::make_unique<EngineDxFeature>();
            state->upscaler = std::make_unique<d4r::dx::GameUpscaler>(*model, feature.outWidth, feature.outHeight,
                                                                      feature.outWidth, feature.outHeight);
            feature.engineDx = std::move(state);
            logf("engine backend (D3D12): feature %u runs preset K in the game's command list (%ux%u -> %ux%u%s)",
                 feature.handle.Id, p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight, ldr ? ", LDR" : "");
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend (D3D12): %s", error.what());
        return refuse("the engine could not be created");
    }
    EngineDxFeature& state = *feature.engineDx;
    engine_dx_settle(state); // the previous frame, when no hook reported it
    try
    {
        int timing = -1;
        EngineDxMarks marks{&state, -1, 0};
        auto startTiming = [&](const char* const* stages, uint32_t count) {
            timing = engine_dx_timing(state, frame, "K", stages, count, p.renderWidth, p.renderHeight,
                                      feature.outWidth, feature.outHeight);
            marks.slot = timing;
            engine_dx_mark(state, timing, list, 0);
        };
        // the end timestamp goes into the list that records the frame's last work (the split's second part, if any)
        auto endTiming = [&](ID3D12GraphicsCommandList* recorded) { engine_dx_end_timing(state, timing, recorded); };
#ifdef D4R_ENGINE_D3DHIP
        if (state.hip != nullptr)
        {
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                startTiming(kEngineDxStagesKHip, stage_count(kEngineDxStagesKHip));
                state.upscaler->recordFront(list, game, marks.profiler(1));
            }
            // the split's mutex is taken here without state.mutex: the between-parts hook takes them the other way
            ID3D12GraphicsCommandList* back = nullptr;
            const bool split = native_split_frame(feature, list, frame, engine_dx_hip_network, &state, &back);
            std::lock_guard<std::mutex> lock(state.mutex);
            if (split)
            {
                state.upscaler->recordBack(back, marks.profiler(5));
                endTiming(back);
            }
            else
            {
                // no second part: complete the frame in the game's list (the head is the previous frame's) and have
                // the next frame's engine run the DXIL network instead
                logf("engine backend (D3D12, d3dhip): feature %u could not split its list at frame %u; the DXIL engine replaces it",
                     feature.handle.Id, frame);
                feature.engineKHipFailed = true;
                state.upscaler->recordBack(list, marks.profiler(5));
                endTiming(list);
            }
            state.pending = true;
        }
        else
#endif
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            startTiming(kEngineDxStagesK, stage_count(kEngineDxStagesK));
            state.upscaler->record(list, game, marks.profiler(1));
            endTiming(list);
            state.pending = true;
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend (D3D12): %s", error.what());
        return refuse("the frame could not be recorded");
    }
    if (g_engineDx.hooks)
        d4r_submit::track(list, {&state, engine_dx_submitted, engine_dx_discarded});
    feature.engineJitter[0] = p.jitterX;
    feature.engineJitter[1] = p.jitterY;
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded on D3D12 (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f, render %ux%u, "
             "exposure %s, preExposure %.6g, exposureScale %.6g)", frame,
             p.jitterX, p.jitterY, game.settings.reset, p.mvScaleX, p.mvScaleY, p.renderWidth, p.renderHeight,
             ldr ? "none (LDR)" : (gameExposure ? "game texture" : "automatic"), p.preExposure, p.exposureScale);
    return true;
}

// Records the whole evaluation of `frame` into `list`. False: nothing usable was recorded and the
// CUDA backend takes this feature over.

bool engine_evaluate(Feature& feature, ID3D12GraphicsCommandList* list, ID3D12Resource* color,
                            ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                            ID3D12Resource* exposure, void* parameters, const FrameParams& p, uint32_t frame)
{
    if (feature.engineRefused)
        return false;
    auto refuse = [&](const char* reason) {
        feature.engineRefused = true;
        logf("engine backend: %s; feature %u uses the CUDA backend from frame %u", reason, feature.handle.Id, frame);
        if (feature.engine != nullptr)
            logf("engine backend: the temporal history restarts in the CUDA backend");
        return false;
    };
    // Windows without vkd3d-proton: the native D3D12 engine (preset K)
    if (feature.nativeD3d12 && feature.rayReconstruction)
        return engine_evaluate_d3d12_rr(feature, list, color, depth, motion, output, parameters, p, frame);
    if (feature.nativeD3d12)
        return engine_evaluate_d3d12(feature, list, color, depth, motion, output, exposure, p, frame);
    std::call_once(g_engine.vulkanOnce, init_engine_vulkan);
    if (feature.rayReconstruction)
        return engine_evaluate_rr(feature, list, color, depth, motion, output, parameters, p, frame);
    if (feature.preset == 13 || feature.preset == 12)
        return engine_evaluate_m(feature, list, color, depth, motion, output, p, frame);
    if (feature.preset == 5 || feature.preset == 6)
        return engine_evaluate_e(feature, list, color, depth, motion, output, exposure, p, frame);
    if (feature.preset == 11 && engine_k_refused())
        return refuse("[Engine] Enabled = auto keeps K on the native kernels on RDNA3 (faster than the Vulkan network)");
    // Without the HDR flag NGX runs its LDR input and output kernels: a model directory of its own.
    const bool ldr = (feature.flags & 0x1) == 0;
    if (ldr)
        std::call_once(g_engine.ldrOnce, init_engine_backend_ldr);
    else
        std::call_once(g_engine.once, init_engine_backend);
    d4r::Model* const model = ldr ? g_engine.modelLdr.get() : g_engine.model.get();
    if (model == nullptr)
        return refuse(ldr ? "no LDR model (engine\\k-ldr) for a game without the HDR flag" : "not available");
    // NVSDK_NGX_DLSS_Feature_Flags: IsHDR 1, MVLowRes 2, MVJittered 4, DepthInverted 8.
    if (feature.preset != 11)
        return refuse("only presets E, F, K, L and M are implemented");
    // MVLowRes (0x2) clear: vectors are at output resolution (engine revision D4RO0003).
    const bool displayMotion = (feature.flags & 0x2) == 0;
    const bool autoExposure = (feature.flags & 0x40) != 0;
    const bool gameExposure = !ldr && !autoExposure;
    if (const char* reason = engine_k_refusal(*model, p, ldr, displayMotion, autoExposure, (feature.flags & 0x8) != 0))
        return refuse(reason);
    if (output == color || output == motion || output == depth)
        return refuse("input/output texture aliasing is not supported");

    d4r::GameFrame game;
    game.profile = engine_stage_profile(frame, "K");
    using Source = EngineFrameSource;
    const auto inputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_INPUT_STATE", 0x40));
    const auto depthState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_DEPTH_STATE", 0x40));
    const auto outputState = static_cast<D3D12_RESOURCE_STATES>(env_uint("D4R_SHIM_OUTPUT_STATE", 0x8));
    Source sources[5] = {
        {color, &game.color, p.colorBaseX, p.colorBaseY, p.renderWidth, p.renderHeight, inputState,
         D3D12_RESOURCE_STATE_COPY_SOURCE},
        {depth, &game.depth, p.depthBaseX, p.depthBaseY, p.renderWidth, p.renderHeight, depthState,
         D3D12_RESOURCE_STATE_COPY_SOURCE},
        {motion, &game.motion, p.mvBaseX, p.mvBaseY, displayMotion ? feature.outWidth : p.renderWidth,
         displayMotion ? feature.outHeight : p.renderHeight, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
        {output, &game.output, p.outputBaseX, p.outputBaseY, feature.outWidth, feature.outHeight, outputState,
         D3D12_RESOURCE_STATE_COPY_DEST},
        {exposure, &game.exposure, 0, 0, 1, 1, inputState, D3D12_RESOURCE_STATE_COPY_SOURCE},
    };
    const size_t sourceCount = gameExposure ? 5 : 4;
    for (size_t index = 0; index < sourceCount; ++index)
    {
        Source& source = sources[index];
        UINT64 handle = 0, offset = 0;
        D3D12_RESOURCE_DESC desc;
        source.resource->GetDesc(&desc);
        if (FAILED(g_vk.interop->GetVulkanResourceInfo1(source.resource, &handle, &offset, &source.image->format)) ||
            handle == 0 || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
            source.x + source.width > desc.Width || source.y + source.height > desc.Height)
            return refuse("a texture is not a single-sample 2D image covering its rectangle");
        source.image->image = reinterpret_cast<VkImage>(handle);
        source.image->x = source.x;
        source.image->y = source.y;
        const bool isOutput = source.image == &game.output;
        prepare_engine_view(feature, source.resource, desc, *source.image, isOutput,
                            (source.x == 0 && source.y == 0) || model->hasDirectOrigins(), source.copyState);
        g_vk.interop->GetVulkanImageLayout(source.resource, source.copyState, &source.image->layout);
    }
    if (gameExposure && (game.exposure.view == VK_NULL_HANDLE || (game.exposure.format != VK_FORMAT_R32G32_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R16G16_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R32_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R16_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R32G32_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R16G16_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R32G32B32A32_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R16G16B16A16_SFLOAT &&
                                                                  game.exposure.format != VK_FORMAT_R8G8B8A8_UNORM)))
        return refuse("the exposure texture must be a float or unorm texture the engine samples directly");
    if (game.depth.format == VK_FORMAT_D32_SFLOAT || game.depth.format == VK_FORMAT_D32_SFLOAT_S8_UINT)
        game.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!d4r::gameColorFormat(game.color.format) || !d4r::gameMotionFormat(game.motion.format) ||
        !d4r::gameDepthFormat(game.depth.format, game.depth.aspect) || !d4r::gameOutputFormat(game.output.format))
    {
        logf("engine backend: Vulkan formats colour %d, depth %d, motion %d, output %d", game.color.format,
             game.depth.format, game.motion.format, game.output.format);
        return refuse("a texture format is not implemented");
    }
    if (frame <= 2)
        logf("engine direct textures: color %d, motion %d, depth %d, output %d (format %d, final store for it %d)",
             game.color.view != VK_NULL_HANDLE, game.motion.view != VK_NULL_HANDLE,
             game.depth.view != VK_NULL_HANDLE, game.output.view != VK_NULL_HANDLE,
             game.output.format, model->hasDirectOutput(game.output.format));
    game.settings = engine_k_settings(feature, p, ldr, displayMotion, autoExposure, feature.engine == nullptr);
    // The CUDA backend converts other colour and motion formats to binary16 (toward zero) before NGX reads them;
    // the engine repeats that conversion, which these formats do not need (exact in binary16).
    game.settings.canonicalColor = game.color.format != VK_FORMAT_R16G16B16A16_SFLOAT &&
                                   game.color.format != VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    game.settings.canonicalMotion = game.motion.format != VK_FORMAT_R16G16_SFLOAT &&
                                    game.motion.format != VK_FORMAT_R16G16B16A16_SFLOAT;
    try
    {
        if (feature.engine == nullptr)
        {
            const size_t tokenBytes = static_cast<size_t>(d4r::Engine::networkTokenBytes(feature.outWidth, feature.outHeight));
            const size_t headBytes = static_cast<size_t>(d4r::Engine::networkHeadBytes(feature.outWidth, feature.outHeight));
            const std::string& directory = ldr ? g_engine.directoryLdr : g_engine.directory;
            const bool vkhip = create_engine_vkhip(feature, directory, tokenBytes, headBytes);
            if (!vkhip && engine_k_native_rdna3())
                return refuse("vkhip is unavailable, and [Engine] Enabled = auto keeps K on the native kernels on "
                              "RDNA3 (faster than the Vulkan network)");
            const bool hip = !vkhip &&
                create_engine_network(feature, directory, "K", true,
                                      feature.outWidth, feature.outHeight, tokenBytes, headBytes);
            const d4r::ExternalNetwork external = vkhip ?
                d4r::ExternalNetwork{feature.engineVkHip->tokens(), feature.engineVkHip->head()} :
                d4r::ExternalNetwork{feature.engineTokens.buffer, feature.engineHead.buffer};
            feature.engine = std::make_unique<d4r::GameUpscaler>(*model, g_vk.physical, g_vk.device,
                                                                 feature.outWidth, feature.outHeight,
                                                                 feature.outWidth, feature.outHeight,
                                                                 (hip || vkhip) ? &external : nullptr);
            if (const std::string dump = env_string("D4R_ENGINE_HISTORY_DUMP_DIR"); !dump.empty())
                feature.engine->enableHistoryDump(dump, env_uint("D4R_VKHIP_DUMP_FRAMES", 4));
            logf("engine backend: feature %u runs preset K in the game's command list (%ux%u -> %ux%u), network on %s",
                 feature.handle.Id, p.renderWidth, p.renderHeight, feature.outWidth, feature.outHeight,
                 vkhip ? "experimental vkhip (one Vulkan command list)" :
                 hip ? "the native HIP layers (list split around it)" : "Vulkan");
        }
    }
    catch (const std::exception& error)
    {
        logf("engine backend: %s", error.what());
        release_engine_network(feature);
        return refuse("the engine could not be created");
    }
    if (feature.engineNet != nullptr)
    {
        // The native HIP network: the game's list is split around it. Its native completion callback releases
        // the second part, which records the final store (or the conversion blit for incompatible outputs).
        auto recordPart = [&](bool front) -> const char* {
            VkCommandBuffer command = VK_NULL_HANDLE;
            if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &command)))
                return "BeginVkCommandBufferInterop failed";
            const char* failure = nullptr;
            try
            {
                if (front)
                {
                    feature.engine->recordFront(command, game);
                    // The HIP network's stream waits for this frame's number (record_engine_input_ready): the
                    // fill is part of the first half, so the network starts the moment the front's GPU work ends.
                    if (feature.engineGpuWait.load())
                        record_engine_input_ready(feature, command, frame);
                }
                else
                    feature.engine->recordBack(command, game);
            }
            catch (const std::exception& error)
            {
                logf("engine backend (K): %s", error.what());
                failure = "the frame could not be recorded";
            }
            if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && failure == nullptr)
                failure = "EndVkCommandBufferInterop failed";
            return failure;
        };
        auto abandon = [&](const char* reason) {
            feature.engineRefused = true;
            logf("engine backend (K): %s; feature %u uses the CUDA backend after frame %u", reason, feature.handle.Id, frame);
            return true;
        };
        return record_engine_network_frame(feature, list, sources, frame, "K", game.settings.reset,
                                           game.settings.motionScale[0], game.settings.motionScale[1], p.jitterX,
                                           p.jitterY, recordPart, refuse, abandon);
    }
    if (FAILED(g_vk.lifetime->RetainExternalResources(list, feature.resources)))
        return refuse("the command list cannot retain the engine's resources");
    // D4R_PROFILE: the Vulkan network's GPU time on the game's queue (timestamps around the whole recording)
    const bool profile = profile_enabled() && env_uint("D4R_PROFILE_GPU_TIMESTAMPS", 1) != 0 &&
        profile_timestamps_ready(feature);
    if (profile)
    {
        FrameTiming measured;
        read_profile_timestamps(feature, frame, measured);
        const auto now = ProfileClock::now();
        const double interval = feature.engineRProfileAt.time_since_epoch().count() ?
            profile_ms(feature.engineRProfileAt, now) : 0.0;
        feature.engineRProfileAt = now;
        if (profile_log_frame(frame) && measured.vkTotal >= 0)
            logf("D4R_ENGINE_K_PROFILE frame=%u gpu_frame=%u render=%ux%u output=%ux%u gpu_eval=%.3f "
                 "game_interval=%.3f age=0", frame, frame - kOutputSlots, p.renderWidth, p.renderHeight,
                 feature.outWidth, feature.outHeight, measured.vkTotal, interval);
        profile_timestamp(feature, list, frame, 0);
        profile_timestamp(feature, list, frame, 1);
    }
    for (size_t index = 0; index < sourceCount; ++index) // sources[4] (exposure) is null under AutoExposure
        transition(list, sources[index].resource, sources[index].state, sources[index].copyState);
    VkCommandBuffer command = VK_NULL_HANDLE;
    bool recorded = SUCCEEDED(g_vk.interop->BeginVkCommandBufferInterop(list, &command));
    const char* failure = recorded ? nullptr : "BeginVkCommandBufferInterop failed";
    if (recorded)
    {
        try
        {
            if (feature.engineVkHip) {
                feature.engine->recordFront(command, game);
                feature.engineVkHip->record(command, game.settings.reset);
                feature.engine->recordBack(command, game);
            } else feature.engine->record(command, game);
        }
        catch (const std::exception& error)
        {
            logf("engine backend: %s", error.what());
            failure = "the frame could not be recorded";
            recorded = false;
        }
        if (FAILED(g_vk.interop->EndVkCommandBufferInterop(list)) && recorded)
        {
            failure = "EndVkCommandBufferInterop failed";
            recorded = false;
        }
    }
    for (size_t index = 0; index < sourceCount; ++index) // sources[4] (exposure) is null under AutoExposure
        transition(list, sources[index].resource, sources[index].copyState, sources[index].state);
    if (!recorded)
        return refuse(failure);
    if (profile)
    {
        profile_timestamp(feature, list, frame, 2);
        profile_timestamp(feature, list, frame, 3);
    }
    feature.engineJitter[0] = p.jitterX;
    feature.engineJitter[1] = p.jitterY;
    if (frame <= 3 || frame % 600 == 0)
        logf("engine frame %u recorded (jitter %.6f,%.6f, reset %d, mvScale %.3f,%.3f, render %ux%u)", frame,
             p.jitterX, p.jitterY, game.settings.reset, p.mvScaleX, p.mvScaleY, p.renderWidth, p.renderHeight);
    return true;
}
