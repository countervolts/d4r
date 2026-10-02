#include "hip_api.h"
#include "cuda_api.h"
#include "ngx_cuda_evaluate.h"

// Same NGX C ABI and freestanding MSVC parameter accessors as the upstream shim.
// NVIDIA SDK headers and proprietary binaries are not build dependencies.
struct NgxPathListInfo { const wchar_t* const* Path; unsigned Length; };
struct NgxLoggingInfo { void* LoggingCallback; int MinimumLoggingLevel; bool DisableOtherLoggingSinks; };
struct NgxFeatureCommonInfo { NgxPathListInfo PathListInfo; void* InternalData; NgxLoggingInfo LoggingInfo; };
static_assert(sizeof(NgxFeatureCommonInfo) == 40);
extern "C" void d4r_ngx_set_uint(void*, const char*, unsigned);
extern "C" unsigned d4r_ngx_get_uint(void*, const char*, unsigned*);
extern "C" unsigned d4r_ngx_get_int(void*, const char*, int*);

static void ngx_log(const char* message, int level, unsigned component)
{
    if (message) std::printf("NGX_LOG level=%d component=%u %s\n", level, component, message);
}
static void check(unsigned result, const char* call)
{
    std::printf("NGX %s -> 0x%08x\n", call, result);
    if (result != 1) throw std::runtime_error(std::string(call) + " NGX result=" + std::to_string(result));
}

int main(int argc, char** argv)
{
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        if (args.cuda_dll.empty() || args.ngx_core.empty() || args.dlss_dll.empty())
            throw std::runtime_error("--cuda-dll, --ngx-core and --dlss-dll absolute paths required");
        const std::filesystem::path core_path(wide(args.ngx_core)), dlss_path(wide(args.dlss_dll));
        if (dlss_path.filename() != L"nvngx_dlss.dll") throw std::runtime_error("DLSS file must be named nvngx_dlss.dll");
        HipApi hip(args.hip_root);
        hipDeviceProp_t props{};
        hip.select_gfx1201(args.device, props);
        SearchDirectory cuda_search(std::filesystem::path(wide(args.cuda_dll)).parent_path());
        CudaApi cuda(args.cuda_dll);
        cuda.check(cuda.cuInit(0), "cuInit");
        int count = 0;
        cuda.check(cuda.cuDeviceGetCount(&count), "cuDeviceGetCount");
        CUdevice device = -1;
        for (int i = 0; i < count; ++i) {
            CUdevice candidate = -1;
            int bus = -1, pci_device = -1, domain = -1;
            cuda.check(cuda.cuDeviceGet(&candidate, i), "cuDeviceGet");
            cuda.check(cuda.cuDeviceGetAttribute(&bus, 33, candidate), "cuDeviceGetAttribute(PCI_BUS_ID)");
            cuda.check(cuda.cuDeviceGetAttribute(&pci_device, 34, candidate), "cuDeviceGetAttribute(PCI_DEVICE_ID)");
            cuda.check(cuda.cuDeviceGetAttribute(&domain, 50, candidate), "cuDeviceGetAttribute(PCI_DOMAIN_ID)");
            if (bus == props.pciBusID && pci_device == props.pciDeviceID && domain == props.pciDomainID) device = candidate;
        }
        if (device < 0) throw std::runtime_error("NGX CUDA/HIP PCI identity mismatch");
        CUcontext context = nullptr;
        cuda.check(cuda.cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
        struct ContextCleanup {
            CudaApi& cuda; CUcontext& context; CUdevice device;
            ~ContextCleanup() { if (context) { (void)cuda.cuCtxSetCurrent(nullptr); (void)cuda.cuDevicePrimaryCtxRelease_v2(device); } }
        } cc{cuda, context, device};
        cuda.check(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent");
        auto nvapi_path = std::filesystem::path(wide(args.cuda_dll)).parent_path() / L"nvapi64.dll";
        if (!std::filesystem::exists(nvapi_path)) nvapi_path = nvapi_path.parent_path().parent_path() / L"nvapi64.dll";
        if (!args.nvapi_dll.empty()) nvapi_path = wide(args.nvapi_dll);
        Library nvapi(nvapi_path);
        SearchDirectory feature_search(dlss_path.parent_path());
        // Preload the supplied absolute file so a system copy cannot replace it.
        Library dlss(dlss_path), core(core_path);
        using Init = unsigned(*)(unsigned long long, const wchar_t*, const NgxFeatureCommonInfo*, unsigned);
        using DriverInit = unsigned(*)(unsigned long long, const wchar_t*, unsigned);
        using ProjectInit = unsigned(*)(const char*, int, const char*, const wchar_t*, unsigned, const NgxFeatureCommonInfo*);
        using Shutdown = unsigned(*)();
        using Parameters = unsigned(*)(void**);
        using Destroy = unsigned(*)(void*);
        const auto init = core.symbol<Init>("NVSDK_NGX_CUDA_Init");
        const auto driver_init = core.symbol<DriverInit>("NVSDK_NGX_CUDA_Init");
        const auto shutdown = core.symbol<Shutdown>("NVSDK_NGX_CUDA_Shutdown");
        const auto capabilities = core.symbol<Parameters>("NVSDK_NGX_CUDA_GetCapabilityParameters");
        const auto allocate = core.symbol<Parameters>("NVSDK_NGX_CUDA_AllocateParameters");
        const auto destroy = core.symbol<Destroy>("NVSDK_NGX_CUDA_DestroyParameters");
        const std::wstring directory = dlss_path.parent_path().wstring();
        const wchar_t* paths[] = {directory.c_str()};
        const char* diagnostic_dir = std::getenv("D4R_DIAG_DIR");
        const auto data_path = diagnostic_dir ? std::filesystem::path(wide(diagnostic_dir)) :
            std::filesystem::current_path() / L"test-results" / L"ngx-init";
        std::filesystem::create_directories(data_path);
        NgxFeatureCommonInfo common{};
        common.PathListInfo = {paths, 1};
        common.LoggingInfo = {reinterpret_cast<void*>(&ngx_log), 2, false};
        loaded_modules();
        std::printf("STAGE NVSDK_NGX_CUDA_Init sdk=0x15 core=%s dlss=%s\n", args.ngx_core.c_str(), args.dlss_dll.c_str());
        if (args.ngx_abi == "driver") check(driver_init(241534723ull, data_path.c_str(), 0x15), "NVSDK_NGX_CUDA_Init(driver ABI)");
        else if (args.ngx_abi == "sdk") check(init(241534723ull, data_path.c_str(), &common, 0x15), "NVSDK_NGX_CUDA_Init(SDK ABI)");
        else if (args.ngx_abi == "project") check(core.symbol<ProjectInit>("NVSDK_NGX_CUDA_Init_ProjectID")(
            "24480451-f00d-face-1304-0308dabad187", 0, "1.0", data_path.c_str(), 0x15, &common), "NVSDK_NGX_CUDA_Init_ProjectID");
        else throw std::runtime_error("--ngx-abi must be driver, sdk or project");
        bool initialized = true;
        struct NgxCleanup { Shutdown shutdown; bool& active; ~NgxCleanup() { if (active) (void)shutdown(); } } nc{shutdown, initialized};
        void* caps = nullptr;
        check(capabilities(&caps), "NVSDK_NGX_CUDA_GetCapabilityParameters");
        if (!caps) throw std::runtime_error("NGX returned null capability parameters");
        struct ParameterCleanup { Destroy destroy; void*& p; ~ParameterCleanup() { if (p) (void)destroy(p); } } pc{destroy, caps};
        int available = -1, init_result = -1, needs_driver = -1;
        check(d4r_ngx_get_int(caps, "SuperSampling.Available", &available), "Get(SuperSampling.Available)");
        check(d4r_ngx_get_int(caps, "SuperSampling.FeatureInitResult", &init_result), "Get(SuperSampling.FeatureInitResult)");
        check(d4r_ngx_get_int(caps, "SuperSampling.NeedsUpdatedDriver", &needs_driver), "Get(SuperSampling.NeedsUpdatedDriver)");
        std::printf("CAPABILITIES available=%d feature_init=0x%08x needs_driver=%d\n", available, static_cast<unsigned>(init_result), needs_driver);
        if (available != 1 || init_result != 1 || needs_driver != 0)
            throw std::runtime_error("DLSS CUDA feature is unavailable; initialization alone is insufficient");
        void* parameters = nullptr;
        check(allocate(&parameters), "NVSDK_NGX_CUDA_AllocateParameters");
        if (!parameters) throw std::runtime_error("Null NGX parameters");
        ParameterCleanup ac{destroy, parameters};
        d4r_ngx_set_uint(parameters, "Width", 640);
        unsigned roundtrip = 0;
        check(d4r_ngx_get_uint(parameters, "Width", &roundtrip), "MSVC parameter ABI roundtrip");
        if (roundtrip != 640) throw std::runtime_error("NGX parameter ABI mismatch");
        if (args.ngx_mode == "evaluate")
            run_synthetic_dlss(cuda, core, parameters, args.preset, args.iterations, data_path);
        else if (args.ngx_mode != "init") throw std::runtime_error("--ngx-mode must be init or evaluate");
        check(destroy(parameters), "DestroyParameters(allocated)"); parameters = nullptr;
        check(destroy(caps), "DestroyParameters(capabilities)"); caps = nullptr;
        loaded_modules();
        check(shutdown(), "NVSDK_NGX_CUDA_Shutdown"); initialized = false;
        cuda.check(cuda.cuCtxSynchronize(), "cuCtxSynchronize");
        cuda.check(cuda.cuCtxSetCurrent(nullptr), "cuCtxSetCurrent(NULL)");
        cuda.check(cuda.cuDevicePrimaryCtxRelease_v2(device), "cuDevicePrimaryCtxRelease_v2"); context = nullptr;
        std::printf("PASS NGX_INIT architecture=gfx1201 sr_available=1 transformer_validation=pending\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL NGX_INIT %s\n", e.what());
        loaded_modules();
        return 4;
    }
}
