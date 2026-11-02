// NVAPI for the NGX core on native Windows with an AMD GPU (the role DXVK-NVAPI has under Proton).
// The NGX core loads nvapi64.dll from System32 only; the shim preloads this one by path (D4R_NVAPI) so that
// load returns it. It describes one GPU: the system's AMD adapter, reported as an Ada (AD100) part because NGX
// picks its weights by architecture (DXVK_NVAPI_GPU_ARCH=AD100 under Proton), and as CUDA capable through
// ZLUDA. Everything else returns no interface; D4R_NVAPI_LOG=path lists what was asked for.
#define COBJMACROS
#include <windows.h>
#include <dxgi.h>
#include <stdio.h>
#include <string.h>

typedef unsigned int NvU32;
typedef void* NvHandle;
#define NVAPI_OK 0
#define NVAPI_ERROR (-1)
#define NVAPI_INVALID_ARGUMENT (-5)
#define NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE (-101)
#define NVAPI_INCOMPATIBLE_STRUCT_VERSION (-9)
#define NVAPI_SETTING_NOT_FOUND (-160)
#define NVAPI_EXECUTABLE_NOT_FOUND (-166)
#define NVAPI_MAX_PHYSICAL_GPUS 64
#define NV_VERSION(type, v) ((NvU32)(sizeof(type) | ((v) << 16)))

// The GPU d4r reports: an RTX 4090's identity (AD102), which is what DLSS's model selection sees
#define D4R_NV_ARCH_AD100 0x190u
#define D4R_NV_IMPL_AD102 0x2u
#define D4R_NV_REV_A01 0x11u
#define D4R_NV_DEVICE_ID 0x268410deu
#define D4R_NV_DRIVER 99999u

static NvHandle const physical_gpu = (NvHandle)(ULONG_PTR)1;
static NvHandle const logical_gpu = (NvHandle)(ULONG_PTR)2;
static NvHandle const drs_session = (NvHandle)(ULONG_PTR)3;
static NvHandle const drs_profile = (NvHandle)(ULONG_PTR)4;

static void trace(const char* format, ...)
{
    static FILE* file;
    static int opened;
    if (!opened)
    {
        char path[MAX_PATH];
        const DWORD length = GetEnvironmentVariableA("D4R_NVAPI_LOG", path, MAX_PATH);
        if (length > 0 && length < MAX_PATH)
            file = fopen(path, "a");
        opened = 1;
    }
    if (file == NULL)
        return;
    va_list args;
    va_start(args, format);
    vfprintf(file, format, args);
    va_end(args);
    fputc('\n', file);
    fflush(file);
}

// The adapter CUDA (ZLUDA) runs on: the AMD adapter with the most dedicated memory, as the bridge picks it
static int adapter_luid(LUID* luid)
{
    static LUID cached;
    static int state; // 0 unknown, 1 found, -1 none
    if (state == 0)
    {
        state = -1;
        IDXGIFactory1* factory = NULL;
        if (SUCCEEDED(CreateDXGIFactory1(&IID_IDXGIFactory1, (void**)&factory)))
        {
            SIZE_T best = 0;
            IDXGIAdapter1* adapter = NULL;
            for (UINT i = 0; IDXGIFactory1_EnumAdapters1(factory, i, &adapter) != DXGI_ERROR_NOT_FOUND; i++)
            {
                DXGI_ADAPTER_DESC1 desc;
                if (SUCCEEDED(IDXGIAdapter1_GetDesc1(adapter, &desc)) && desc.VendorId == 0x1002 &&
                    !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && desc.DedicatedVideoMemory > best)
                {
                    best = desc.DedicatedVideoMemory;
                    cached = desc.AdapterLuid;
                    state = 1;
                }
                IDXGIAdapter1_Release(adapter);
            }
            IDXGIFactory1_Release(factory);
        }
    }
    if (state == 1)
        *luid = cached;
    return state == 1;
}

static void copy_short_string(char* destination, const char* source)
{
    strncpy(destination, source, 63);
    destination[63] = '\0';
}

static int __cdecl initialize(void) { return NVAPI_OK; }
static int __cdecl unload(void) { return NVAPI_OK; }

static int __cdecl get_error_message(int status, char* text)
{
    if (text == NULL)
        return NVAPI_INVALID_ARGUMENT;
    snprintf(text, 64, "NVAPI status %d", status);
    return NVAPI_OK;
}

static int __cdecl get_interface_version_string(char* text)
{
    if (text == NULL)
        return NVAPI_INVALID_ARGUMENT;
    copy_short_string(text, "d4r NVAPI (ZLUDA)");
    return NVAPI_OK;
}

static int __cdecl enum_physical_gpus(NvHandle handles[NVAPI_MAX_PHYSICAL_GPUS], NvU32* count)
{
    if (handles == NULL || count == NULL)
        return NVAPI_INVALID_ARGUMENT;
    handles[0] = physical_gpu;
    *count = 1;
    return NVAPI_OK;
}

static int __cdecl enum_logical_gpus(NvHandle handles[NVAPI_MAX_PHYSICAL_GPUS], NvU32* count)
{
    if (handles == NULL || count == NULL)
        return NVAPI_INVALID_ARGUMENT;
    handles[0] = logical_gpu;
    *count = 1;
    return NVAPI_OK;
}

static int __cdecl get_physical_gpus_from_logical_gpu(NvHandle logical, NvHandle handles[NVAPI_MAX_PHYSICAL_GPUS],
                                                       NvU32* count)
{
    if (logical != logical_gpu || handles == NULL || count == NULL)
        return NVAPI_INVALID_ARGUMENT;
    handles[0] = physical_gpu;
    *count = 1;
    return NVAPI_OK;
}

static int __cdecl get_logical_gpu_from_physical_gpu(NvHandle physical, NvHandle* logical)
{
    if (physical != physical_gpu)
        return NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE;
    if (logical == NULL)
        return NVAPI_INVALID_ARGUMENT;
    *logical = logical_gpu;
    return NVAPI_OK;
}

typedef struct
{
    NvU32 version;
    NvU32 architecture;
    NvU32 implementation;
    NvU32 revision;
} ArchInfo; // NV_GPU_ARCH_INFO V1 and V2 share this layout

static int __cdecl gpu_get_arch_info(NvHandle gpu, ArchInfo* info)
{
    if (gpu != physical_gpu)
        return NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE;
    if (info == NULL)
        return NVAPI_INVALID_ARGUMENT;
    if (info->version != NV_VERSION(ArchInfo, 1) && info->version != NV_VERSION(ArchInfo, 2))
        return NVAPI_INCOMPATIBLE_STRUCT_VERSION;
    info->architecture = D4R_NV_ARCH_AD100;
    info->implementation = D4R_NV_IMPL_AD102;
    info->revision = D4R_NV_REV_A01;
    return NVAPI_OK;
}

static int __cdecl gpu_get_pci_identifiers(NvHandle gpu, NvU32* device, NvU32* subsystem, NvU32* revision,
                                           NvU32* external_device)
{
    if (gpu != physical_gpu)
        return NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE;
    if (device == NULL || subsystem == NULL || revision == NULL || external_device == NULL)
        return NVAPI_INVALID_ARGUMENT;
    *device = D4R_NV_DEVICE_ID;
    *subsystem = 0;
    *revision = 0xa1;
    *external_device = D4R_NV_DEVICE_ID >> 16;
    return NVAPI_OK;
}

static int __cdecl gpu_get_full_name(NvHandle gpu, char* name)
{
    if (gpu != physical_gpu)
        return NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE;
    if (name == NULL)
        return NVAPI_INVALID_ARGUMENT;
    copy_short_string(name, "d4r ZLUDA GPU (AMD)");
    return NVAPI_OK;
}

static int __cdecl gpu_get_gpu_type(NvHandle gpu, NvU32* type)
{
    if (gpu != physical_gpu)
        return NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE;
    if (type == NULL)
        return NVAPI_INVALID_ARGUMENT;
    *type = 2; // NV_SYSTEM_TYPE_DGPU
    return NVAPI_OK;
}

static int __cdecl gpu_get_bus_type(NvHandle gpu, NvU32* type)
{
    if (gpu != physical_gpu)
        return NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE;
    if (type == NULL)
        return NVAPI_INVALID_ARGUMENT;
    *type = 3; // NVAPI_GPU_BUS_TYPE_PCI_EXPRESS
    return NVAPI_OK;
}

static int __cdecl gpu_get_adapter_id_from_physical_gpu(NvHandle gpu, void* luid)
{
    if (gpu != physical_gpu)
        return NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE;
    LUID adapter;
    if (luid == NULL)
        return NVAPI_INVALID_ARGUMENT;
    if (!adapter_luid(&adapter))
        return NVAPI_ERROR;
    memcpy(luid, &adapter, sizeof(adapter));
    return NVAPI_OK;
}

typedef struct
{
    NvU32 version;
    void* os_adapter_id;
    NvU32 physical_gpu_count;
    NvHandle physical_gpus[NVAPI_MAX_PHYSICAL_GPUS];
    NvU32 reserved[8];
} LogicalGpuData; // NV_LOGICAL_GPU_DATA_V1

static int __cdecl gpu_get_logical_gpu_info(NvHandle logical, LogicalGpuData* data)
{
    if (logical != logical_gpu)
        return NVAPI_INVALID_ARGUMENT;
    if (data == NULL)
        return NVAPI_INVALID_ARGUMENT;
    if (data->version != NV_VERSION(LogicalGpuData, 1))
        return NVAPI_INCOMPATIBLE_STRUCT_VERSION;
    LUID adapter;
    if (data->os_adapter_id != NULL)
    {
        if (!adapter_luid(&adapter))
            return NVAPI_ERROR;
        memcpy(data->os_adapter_id, &adapter, sizeof(adapter));
    }
    data->physical_gpu_count = 1;
    data->physical_gpus[0] = physical_gpu;
    return NVAPI_OK;
}

typedef struct
{
    NvU32 version;
    NvU32 count;
    struct
    {
        NvHandle gpu;
        NvU32 flags;
    } gpus[NVAPI_MAX_PHYSICAL_GPUS];
} ComputeTopologyV1;

typedef struct
{
    NvHandle gpu;
    NvU32 flags;
} ComputeGpu;

typedef struct
{
    NvU32 version;
    NvU32 count;
    ComputeGpu* gpus;
} ComputeTopologyV2;

// CUDA capable and available, recommended for PhysX: what NVIDIA's own reports for a GeForce card
#define D4R_NV_COMPUTE_FLAGS 0x3fu

static int __cdecl gpu_cuda_enum_compute_capable_gpus(NvU32* version)
{
    if (version == NULL)
        return NVAPI_INVALID_ARGUMENT;
    if (*version == NV_VERSION(ComputeTopologyV1, 1))
    {
        ComputeTopologyV1* topology = (ComputeTopologyV1*)version;
        topology->count = 1;
        topology->gpus[0].gpu = physical_gpu;
        topology->gpus[0].flags = D4R_NV_COMPUTE_FLAGS;
        return NVAPI_OK;
    }
    if (*version == NV_VERSION(ComputeTopologyV2, 2))
    {
        static ComputeGpu gpu;
        ComputeTopologyV2* topology = (ComputeTopologyV2*)version;
        gpu.gpu = physical_gpu;
        gpu.flags = D4R_NV_COMPUTE_FLAGS;
        topology->count = 1;
        topology->gpus = &gpu;
        return NVAPI_OK;
    }
    return NVAPI_INCOMPATIBLE_STRUCT_VERSION;
}

static int __cdecl get_gpu_id_from_physical_gpu(NvHandle gpu, NvU32* id)
{
    if (gpu != physical_gpu)
        return NVAPI_EXPECTED_PHYSICAL_GPU_HANDLE;
    if (id == NULL)
        return NVAPI_INVALID_ARGUMENT;
    *id = 1;
    return NVAPI_OK;
}

static int __cdecl get_physical_gpu_from_gpu_id(NvU32 id, NvHandle* gpu)
{
    if (id != 1 || gpu == NULL)
        return NVAPI_INVALID_ARGUMENT;
    *gpu = physical_gpu;
    return NVAPI_OK;
}

static int __cdecl sys_get_driver_and_branch_version(NvU32* driver, char* branch)
{
    if (driver == NULL || branch == NULL)
        return NVAPI_INVALID_ARGUMENT;
    *driver = D4R_NV_DRIVER;
    copy_short_string(branch, "r999_99");
    return NVAPI_OK;
}

typedef struct
{
    NvU32 version;
    NvU32 driver;
    NvU32 changelist;
    char branch[64];
    char adapter[64];
} DisplayDriverVersion;

static int __cdecl get_display_driver_version(NvHandle display, DisplayDriverVersion* version)
{
    (void)display;
    if (version == NULL)
        return NVAPI_INVALID_ARGUMENT;
    if (version->version != NV_VERSION(DisplayDriverVersion, 1))
        return NVAPI_INCOMPATIBLE_STRUCT_VERSION;
    version->driver = D4R_NV_DRIVER;
    version->changelist = 0;
    copy_short_string(version->branch, "r999_99");
    copy_short_string(version->adapter, "d4r ZLUDA GPU (AMD)");
    return NVAPI_OK;
}

// Driver settings (DRS): an empty database, so NGX sees no per-application overrides
static int __cdecl drs_create_session(NvHandle* session)
{
    if (session == NULL)
        return NVAPI_INVALID_ARGUMENT;
    *session = drs_session;
    return NVAPI_OK;
}

static int __cdecl drs_session_only(NvHandle session) { return session == drs_session ? NVAPI_OK : NVAPI_INVALID_ARGUMENT; }

static int __cdecl drs_get_profile(NvHandle session, NvHandle* profile)
{
    if (session != drs_session || profile == NULL)
        return NVAPI_INVALID_ARGUMENT;
    *profile = drs_profile;
    return NVAPI_OK;
}

static int __cdecl drs_find_application_by_name(NvHandle session, const void* name, NvHandle* profile, void* application)
{
    (void)name;
    (void)profile;
    (void)application;
    return session == drs_session ? NVAPI_EXECUTABLE_NOT_FOUND : NVAPI_INVALID_ARGUMENT;
}

static int __cdecl drs_get_setting(NvHandle session, NvHandle profile, NvU32 id, void* setting)
{
    (void)profile;
    (void)setting;
    trace("  DRS_GetSetting 0x%08x: not set", id);
    return session == drs_session ? NVAPI_SETTING_NOT_FOUND : NVAPI_INVALID_ARGUMENT;
}

static const struct
{
    NvU32 id;
    void* function;
    const char* name;
} interfaces[] = {
    {0x0150e828u, (void*)initialize, "Initialize"},
    {0xd22bdd7eu, (void*)unload, "Unload"},
    {0x6c2d048cu, (void*)get_error_message, "GetErrorMessage"},
    {0x01053fa5u, (void*)get_interface_version_string, "GetInterfaceVersionString"},
    {0xe5ac921fu, (void*)enum_physical_gpus, "EnumPhysicalGPUs"},
    {0x48b3ea59u, (void*)enum_logical_gpus, "EnumLogicalGPUs"},
    {0xaea3fa32u, (void*)get_physical_gpus_from_logical_gpu, "GetPhysicalGPUsFromLogicalGPU"},
    {0xadd604d1u, (void*)get_logical_gpu_from_physical_gpu, "GetLogicalGPUFromPhysicalGPU"},
    {0xd8265d24u, (void*)gpu_get_arch_info, "GPU_GetArchInfo"},
    {0x2ddfb66eu, (void*)gpu_get_pci_identifiers, "GPU_GetPCIIdentifiers"},
    {0xceee8e9fu, (void*)gpu_get_full_name, "GPU_GetFullName"},
    {0xc33baeb1u, (void*)gpu_get_gpu_type, "GPU_GetGPUType"},
    {0x1bb18724u, (void*)gpu_get_bus_type, "GPU_GetBusType"},
    {0x0ff07fdeu, (void*)gpu_get_adapter_id_from_physical_gpu, "GPU_GetAdapterIdFromPhysicalGpu"},
    {0x842b066eu, (void*)gpu_get_logical_gpu_info, "GPU_GetLogicalGpuInfo"},
    {0x5786cc6eu, (void*)gpu_cuda_enum_compute_capable_gpus, "GPU_CudaEnumComputeCapableGpus"},
    {0x36e39e6bu, (void*)gpu_cuda_enum_compute_capable_gpus, "GPU_CudaEnumComputeCapableGpus (internal)"},
    {0x6533ea3eu, (void*)get_gpu_id_from_physical_gpu, "GetGPUIDfromPhysicalGPU"},
    {0x5380ad1au, (void*)get_physical_gpu_from_gpu_id, "GetPhysicalGPUFromGPUID"},
    {0x2926aaadu, (void*)sys_get_driver_and_branch_version, "SYS_GetDriverAndBranchVersion"},
    {0xf951a4d1u, (void*)get_display_driver_version, "GetDisplayDriverVersion"},
    {0x0694d52eu, (void*)drs_create_session, "DRS_CreateSession"},
    {0xdad9cff8u, (void*)drs_session_only, "DRS_DestroySession"},
    {0x375dbd6bu, (void*)drs_session_only, "DRS_LoadSettings"},
    {0xda8466a0u, (void*)drs_get_profile, "DRS_GetBaseProfile"},
    {0x617bff9fu, (void*)drs_get_profile, "DRS_GetCurrentGlobalProfile"},
    {0xeee566b2u, (void*)drs_find_application_by_name, "DRS_FindApplicationByName"},
    {0x73bf8338u, (void*)drs_get_setting, "DRS_GetSetting"},
};

__declspec(dllexport) void* __cdecl nvapi_QueryInterface(NvU32 id)
{
    for (size_t i = 0; i < sizeof(interfaces) / sizeof(interfaces[0]); i++)
        if (interfaces[i].id == id)
        {
            trace("0x%08x %s", id, interfaces[i].name);
            return interfaces[i].function;
        }
    trace("0x%08x (not provided)", id);
    return NULL;
}
