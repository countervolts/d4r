#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_4.h>
#include <d3d12.h>
#include "nvsdk_ngx.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

template <typename Target, typename Source> Target copy_function_pointer(Source source)
{
    static_assert(sizeof(Target) == sizeof(Source), "function pointer sizes differ");
    Target target{};
    std::memcpy(&target, &source, sizeof(target));
    return target;
}

template <typename Function> Function load_export(HMODULE module, const char* name)
{
    FARPROC raw = GetProcAddress(module, name);
    return raw ? copy_function_pointer<Function>(raw) : nullptr;
}

static std::wstring utf8_to_wide(const char* input)
{
    const int required = MultiByteToWideChar(CP_UTF8, 0, input, -1, nullptr, 0);
    if (required <= 1)
        return {};
    std::wstring output(static_cast<size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, input, -1, output.data(), required);
    output.resize(static_cast<size_t>(required - 1));
    return output;
}

static const char* ngx_result_name(NVSDK_NGX_Result result)
{
    switch (result)
    {
    case NVSDK_NGX_Result_Success:
        return "Success";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported:
        return "FeatureNotSupported";
    case NVSDK_NGX_Result_FAIL_PlatformError:
        return "PlatformError";
    case NVSDK_NGX_Result_FAIL_InvalidParameter:
        return "InvalidParameter";
    case NVSDK_NGX_Result_FAIL_NotInitialized:
        return "NotInitialized";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:
        return "UnableToInitializeFeature";
    case NVSDK_NGX_Result_FAIL_OutOfDate:
        return "OutOfDate";
    default:
        return "Other";
    }
}

static void print_ngx_result(const char* stage, NVSDK_NGX_Result result)
{
    std::printf("%s: result=0x%08x (%s)\n", stage, static_cast<unsigned int>(result), ngx_result_name(result));
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::fprintf(stderr,
                     "usage: ngx_d3d12_probe.exe PATH_TO_NGX_CORE_DLL PATH_TO_NVNGX_DLSS_DLL WRITABLE_APPDATA_DIRECTORY\n");
        return 2;
    }

    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);

    const char* ngxCorePath = argv[1];
    const char* featureDllPath = argv[2];
    std::wstring appData = utf8_to_wide(argv[3]);
    std::wstring featureDirectory = utf8_to_wide(featureDllPath);
    if (appData.empty() || featureDirectory.empty())
    {
        std::fprintf(stderr, "Invalid UTF-8 feature-DLL path or appdata path argument\n");
        return 2;
    }
    const size_t separator = featureDirectory.find_last_of(L"\\/");
    if (separator == std::wstring::npos)
        featureDirectory = L".";
    else
        featureDirectory.resize(separator);

    const wchar_t* featurePaths[] = {featureDirectory.c_str()};
    NVSDK_NGX_FeatureCommonInfo featureCommonInfo{};
    featureCommonInfo.PathListInfo.Path = featurePaths;
    featureCommonInfo.PathListInfo.Length = 1;

    HMODULE ngxCore = LoadLibraryA(ngxCorePath);
    if (ngxCore == nullptr)
    {
        std::fprintf(stderr, "LoadLibraryA(%s) failed: GetLastError=%lu\n", ngxCorePath, GetLastError());
        return 1;
    }
    std::printf("Loaded NGX core: %s\n", ngxCorePath);
    std::printf("DLSS SR feature search directory: %ls\n", featureDirectory.c_str());

    HMODULE dxgiModule = LoadLibraryA("dxgi.dll");
    HMODULE d3d12Module = LoadLibraryA("d3d12.dll");
    if (dxgiModule == nullptr || d3d12Module == nullptr)
    {
        std::fprintf(stderr, "Could not load dxgi.dll or d3d12.dll: GetLastError=%lu\n", GetLastError());
        if (d3d12Module != nullptr)
            FreeLibrary(d3d12Module);
        if (dxgiModule != nullptr)
            FreeLibrary(dxgiModule);
        FreeLibrary(ngxCore);
        return 1;
    }

    using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
    using D3D12CreateDeviceFn = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    auto createFactory = load_export<CreateFactoryFn>(dxgiModule, "CreateDXGIFactory1");
    auto createD3D12Device = load_export<D3D12CreateDeviceFn>(d3d12Module, "D3D12CreateDevice");
    auto getRequirements = load_export<decltype(&NVSDK_NGX_D3D12_GetFeatureRequirements)>(
        ngxCore, "NVSDK_NGX_D3D12_GetFeatureRequirements");
    auto init = load_export<decltype(&NVSDK_NGX_D3D12_Init)>(ngxCore, "NVSDK_NGX_D3D12_Init");
    auto shutdown = load_export<decltype(&NVSDK_NGX_D3D12_Shutdown1)>(ngxCore, "NVSDK_NGX_D3D12_Shutdown1");
    auto getCapabilityParameters = load_export<decltype(&NVSDK_NGX_D3D12_GetCapabilityParameters)>(
        ngxCore, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    auto destroyParameters = load_export<decltype(&NVSDK_NGX_D3D12_DestroyParameters)>(
        ngxCore, "NVSDK_NGX_D3D12_DestroyParameters");
    auto getScratchBufferSize = load_export<decltype(&NVSDK_NGX_D3D12_GetScratchBufferSize)>(
        ngxCore, "NVSDK_NGX_D3D12_GetScratchBufferSize");
    auto createFeature = load_export<decltype(&NVSDK_NGX_D3D12_CreateFeature)>(
        ngxCore, "NVSDK_NGX_D3D12_CreateFeature");
    auto releaseFeature = load_export<decltype(&NVSDK_NGX_D3D12_ReleaseFeature)>(
        ngxCore, "NVSDK_NGX_D3D12_ReleaseFeature");
    if (createFactory == nullptr || createD3D12Device == nullptr || getRequirements == nullptr || init == nullptr ||
        getCapabilityParameters == nullptr || destroyParameters == nullptr || getScratchBufferSize == nullptr ||
        createFeature == nullptr || releaseFeature == nullptr)
    {
        std::fprintf(stderr, "Missing DXGI, D3D12, or NGX exports\n");
        FreeLibrary(d3d12Module);
        FreeLibrary(dxgiModule);
        FreeLibrary(ngxCore);
        return 1;
    }

    constexpr GUID iidFactory1 = {0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
    constexpr GUID iidD3D12Device = {0x189819f1, 0x1db6, 0x4b57, {0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7}};
    constexpr GUID iidD3D12CommandAllocator = {0x6102dee4, 0xaf59, 0x4b09, {0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24}};
    constexpr GUID iidD3D12GraphicsCommandList = {0x5b160d0f, 0xac1b, 0x4185, {0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55}};
    constexpr GUID iidD3D12Resource = {0x696442be, 0xa72e, 0x4059, {0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad}};
    IDXGIFactory1* factory = nullptr;
    HRESULT hr = createFactory(iidFactory1, reinterpret_cast<void**>(&factory));
    std::printf("CreateDXGIFactory1: HRESULT=0x%08lx\n", static_cast<unsigned long>(hr));
    if (FAILED(hr) || factory == nullptr)
    {
        FreeLibrary(d3d12Module);
        FreeLibrary(dxgiModule);
        FreeLibrary(ngxCore);
        return 1;
    }

    NVSDK_NGX_FeatureDiscoveryInfo discovery{};
    discovery.SDKVersion = static_cast<NVSDK_NGX_Version>(0x13);
    if (const char* apiVersion = std::getenv("D4R_NGX_API_VERSION"))
        discovery.SDKVersion = static_cast<NVSDK_NGX_Version>(std::strtoul(apiVersion, nullptr, 0));
    discovery.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    discovery.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    discovery.Identifier.v.ApplicationId = 241534723ULL;
    discovery.ApplicationDataPath = appData.c_str();
    discovery.FeatureInfo = &featureCommonInfo;

    IDXGIAdapter1* adapter = nullptr;
    IDXGIAdapter1* amdAdapter = nullptr;
    const char* selectFirstAdapterSetting = std::getenv("D4R_SELECT_FIRST_DXGI_ADAPTER");
    const bool selectFirstAdapter = selectFirstAdapterSetting != nullptr && std::strcmp(selectFirstAdapterSetting, "1") == 0;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index)
    {
        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(adapter->GetDesc1(&description)))
        {
            std::printf("DXGI adapter: vendor=0x%04x device=0x%04x flags=0x%08x luid=%08lx:%08lx name=%ls\n",
                        description.VendorId, description.DeviceId, description.Flags,
                        static_cast<unsigned long>(description.AdapterLuid.HighPart),
                        static_cast<unsigned long>(description.AdapterLuid.LowPart), description.Description);
            if ((description.VendorId == 0x1002 || selectFirstAdapter) && amdAdapter == nullptr)
            {
                amdAdapter = adapter;
                adapter = nullptr;
            }
        }
        if (adapter != nullptr)
        {
            adapter->Release();
            adapter = nullptr;
        }
    }

    if (amdAdapter == nullptr)
    {
        std::fprintf(stderr, "No AMD DXGI adapter was enumerated\n");
        factory->Release();
        FreeLibrary(d3d12Module);
        FreeLibrary(dxgiModule);
        FreeLibrary(ngxCore);
        return 1;
    }

    NVSDK_NGX_FeatureRequirement requirement{};
    NVSDK_NGX_Result ngxResult = getRequirements(amdAdapter, &discovery, &requirement);
    print_ngx_result("D3D12_GetFeatureRequirements(DLSS SR)", ngxResult);
    if (ngxResult == NVSDK_NGX_Result_Success)
    {
        std::printf("feature support flags: 0x%08x\n", static_cast<unsigned int>(requirement.FeatureSupported));
        std::printf("minimum hardware architecture: %u\n", requirement.MinHWArchitecture);
        std::printf("minimum OS version: %s\n", requirement.MinOSVersion);
    }

    ID3D12Device* device = nullptr;
    hr = createD3D12Device(amdAdapter, D3D_FEATURE_LEVEL_12_0, iidD3D12Device,
                           reinterpret_cast<void**>(&device));
    std::printf("D3D12CreateDevice: HRESULT=0x%08lx\n", static_cast<unsigned long>(hr));
    if (FAILED(hr) || device == nullptr)
    {
        amdAdapter->Release();
        factory->Release();
        FreeLibrary(d3d12Module);
        FreeLibrary(dxgiModule);
        FreeLibrary(ngxCore);
        return 1;
    }

    std::printf("NGX API version passed to core: 0x%08x\n", static_cast<unsigned int>(discovery.SDKVersion));
    ngxResult = init(discovery.Identifier.v.ApplicationId, discovery.ApplicationDataPath, device,
                     &featureCommonInfo, discovery.SDKVersion);
    print_ngx_result("D3D12_Init(DLSS SR)", ngxResult);

    if (ngxResult == NVSDK_NGX_Result_Success)
    {
        NVSDK_NGX_Parameter* parameters = nullptr;
        NVSDK_NGX_Result parameterResult = getCapabilityParameters(&parameters);
        print_ngx_result("D3D12_GetCapabilityParameters", parameterResult);
        if (parameterResult == NVSDK_NGX_Result_Success && parameters != nullptr)
        {
            int available = -1;
            NVSDK_NGX_Result availabilityResult = parameters->Get(
                NVSDK_NGX_Parameter_SuperSampling_Available, &available);
            print_ngx_result("Parameters.Get(SuperSampling.Available)", availabilityResult);
            std::printf("DLSS SR availability value: %d\n", available);

            int featureInitResult = -1;
            NVSDK_NGX_Result initStateResult = parameters->Get(
                NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &featureInitResult);
            print_ngx_result("Parameters.Get(SuperSampling.FeatureInitResult)", initStateResult);
            std::printf("DLSS SR recorded feature-init result: 0x%08x\n",
                        static_cast<unsigned int>(featureInitResult));
            int needsUpdatedDriver = -1;
            NVSDK_NGX_Result driverStateResult = parameters->Get(
                NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver);
            print_ngx_result("Parameters.Get(SuperSampling.NeedsUpdatedDriver)", driverStateResult);
            std::printf("DLSS SR needs updated driver: %d\n", needsUpdatedDriver);

            constexpr unsigned int renderWidth = 640;
            constexpr unsigned int renderHeight = 360;
            constexpr unsigned int outputWidth = 1280;
            constexpr unsigned int outputHeight = 720;
            parameters->Set(NVSDK_NGX_Parameter_Width, renderWidth);
            parameters->Set(NVSDK_NGX_Parameter_Height, renderHeight);
            parameters->Set(NVSDK_NGX_Parameter_OutWidth, outputWidth);
            parameters->Set(NVSDK_NGX_Parameter_OutHeight, outputHeight);
            parameters->Set(NVSDK_NGX_Parameter_PerfQualityValue,
                            static_cast<int>(NVSDK_NGX_PerfQuality_Value_MaxQuality));
            parameters->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                            static_cast<int>(NVSDK_NGX_DLSS_Feature_Flags_None));

            size_t scratchSize = 0;
            NVSDK_NGX_Result scratchResult = getScratchBufferSize(
                NVSDK_NGX_Feature_SuperSampling, parameters, &scratchSize);
            print_ngx_result("D3D12_GetScratchBufferSize(DLSS SR)", scratchResult);
            std::printf("DLSS SR scratch size: %llu bytes\n", static_cast<unsigned long long>(scratchSize));

            ID3D12CommandAllocator* commandAllocator = nullptr;
            ID3D12GraphicsCommandList* commandList = nullptr;
            HRESULT commandResult = device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, iidD3D12CommandAllocator,
                reinterpret_cast<void**>(&commandAllocator));
            std::printf("CreateCommandAllocator: HRESULT=0x%08lx\n", static_cast<unsigned long>(commandResult));
            if (SUCCEEDED(commandResult) && commandAllocator != nullptr)
            {
                commandResult = device->CreateCommandList(
                    0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator, nullptr,
                    iidD3D12GraphicsCommandList, reinterpret_cast<void**>(&commandList));
                std::printf("CreateCommandList: HRESULT=0x%08lx\n", static_cast<unsigned long>(commandResult));
            }

            ID3D12Resource* scratchBuffer = nullptr;
            if (scratchResult == NVSDK_NGX_Result_Success && scratchSize > 0 && commandList != nullptr)
            {
                D3D12_HEAP_PROPERTIES heap{};
                heap.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC buffer{};
                buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                buffer.Width = scratchSize;
                buffer.Height = 1;
                buffer.DepthOrArraySize = 1;
                buffer.MipLevels = 1;
                buffer.SampleDesc.Count = 1;
                buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                HRESULT scratchCreateResult = device->CreateCommittedResource(
                    &heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
                    iidD3D12Resource, reinterpret_cast<void**>(&scratchBuffer));
                std::printf("CreateScratchBuffer: HRESULT=0x%08lx\n", static_cast<unsigned long>(scratchCreateResult));
                if (SUCCEEDED(scratchCreateResult) && scratchBuffer != nullptr)
                {
                    parameters->Set(NVSDK_NGX_Parameter_Scratch, scratchBuffer);
                    parameters->Set(NVSDK_NGX_Parameter_Scratch_SizeInBytes,
                                    static_cast<unsigned long long>(scratchSize));
                }
            }

            if (commandList != nullptr)
            {
                NVSDK_NGX_Handle* featureHandle = nullptr;
                NVSDK_NGX_Result createFeatureResult = createFeature(
                    commandList, NVSDK_NGX_Feature_SuperSampling, parameters, &featureHandle);
                print_ngx_result("D3D12_CreateFeature(DLSS SR)", createFeatureResult);
                std::printf("DLSS SR feature handle: %s\n", featureHandle != nullptr ? "created" : "null");
                if (createFeatureResult == NVSDK_NGX_Result_Success && featureHandle != nullptr)
                {
                    NVSDK_NGX_Result releaseResult = releaseFeature(featureHandle);
                    print_ngx_result("D3D12_ReleaseFeature", releaseResult);
                }
            }
            else
            {
                std::printf("Skipping DLSS SR CreateFeature because D3D12 command-list creation failed\n");
            }

            if (scratchBuffer != nullptr)
                scratchBuffer->Release();
            if (commandList != nullptr)
                commandList->Release();
            if (commandAllocator != nullptr)
                commandAllocator->Release();
            NVSDK_NGX_Result destroyResult = destroyParameters(parameters);
            print_ngx_result("D3D12_DestroyParameters", destroyResult);
        }
    }

    if (ngxResult == NVSDK_NGX_Result_Success && shutdown != nullptr)
    {
        NVSDK_NGX_Result shutdownResult = shutdown(device);
        print_ngx_result("D3D12_Shutdown1", shutdownResult);
    }

    device->Release();
    amdAdapter->Release();
    factory->Release();
    FreeLibrary(d3d12Module);
    FreeLibrary(dxgiModule);
    FreeLibrary(ngxCore);
    return 0;
}
