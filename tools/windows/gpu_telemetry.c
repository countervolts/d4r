// Read-only AMD driver telemetry, separate from the game and HIP runtime.
// Uses the ADLX SDK's C interface; no SDK/sample implementation is vendored.
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <ADLX.h>
#include <IPerformanceMonitoring.h>

static int number(const char* text, unsigned* value) {
    char* end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 10);
    if (!text[0] || *end || text[0] == '-' || errno == ERANGE) return 0;
    *value = (unsigned)parsed;
    return 1;
}

static void real_value(ADLX_RESULT status, double value) {
    if (ADLX_SUCCEEDED(status) && isfinite(value) && value >= 0) printf("%.6f", value);
}

static void int_value(ADLX_RESULT status, int value) {
    if (ADLX_SUCCEEDED(status) && value >= 0) printf("%d", value);
}

int main(int argc, char** argv) {
    unsigned seconds = 30, interval = 500, target_pid = 0, delay = 0;
    const char* gpu_name = "RX 9070 XT";
    for (int i = 1; i < argc; ++i) {
        if (i + 1 == argc) return 2;
        const char* key = argv[i];
        const char* value = argv[++i];
        unsigned* dest = !strcmp(key, "--seconds") ? &seconds :
            !strcmp(key, "--interval-ms") ? &interval :
            !strcmp(key, "--target-pid") ? &target_pid :
            !strcmp(key, "--delay-ms") ? &delay : NULL;
        if (dest) { if (!number(value, dest)) return 2; }
        else if (!strcmp(key, "--gpu-name")) gpu_name = value;
        else return 2;
    }
    if (seconds < 1 || seconds > 300 || interval < 100 || interval > 5000 || delay > 300000) return 2;
    int exit_code = 1, initialized = 0;
    HMODULE dll = NULL;
    HANDLE target = NULL;
    IADLXSystem* system = NULL; // Owned by ADLXInitialize/ADLXTerminate.
    IADLXGPUList* gpus = NULL;
    IADLXGPU* selected = NULL;
    IADLXPerformanceMonitoringServices* service = NULL;
    ADLXTerminate_Fn terminate = NULL;
    if (target_pid) {
        target = OpenProcess(SYNCHRONIZE, FALSE, target_pid);
        if (!target) { fprintf(stderr, "OpenProcess error=%lu\n", GetLastError()); goto done; }
    }
    // Load the installed driver library, never a DLL from the game/current dir.
    dll = LoadLibraryExW(L"amdadlx64.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!dll) { fprintf(stderr, "ADLX driver DLL unavailable error=%lu\n", GetLastError()); goto done; }
    ADLXQueryFullVersion_Fn query_version = (ADLXQueryFullVersion_Fn)GetProcAddress(dll, ADLX_QUERY_FULL_VERSION_FUNCTION_NAME);
    ADLXInitialize_Fn initialize = (ADLXInitialize_Fn)GetProcAddress(dll, ADLX_INIT_FUNCTION_NAME);
    terminate = (ADLXTerminate_Fn)GetProcAddress(dll, ADLX_TERMINATE_FUNCTION_NAME);
    adlx_uint64 runtime_version = 0;
    if (!query_version || !initialize || !terminate || ADLX_FAILED(query_version(&runtime_version))) goto done;
    // Only use the base interfaces. Request no version newer than either the
    // installed runtime or these headers (e.g. SDK 2.0 with driver ADLX 1.5).
    adlx_uint64 requested = runtime_version < ADLX_FULL_VERSION ? runtime_version : ADLX_FULL_VERSION;
    ADLX_RESULT result = initialize(requested, &system);
    fprintf(stderr, "ADLX runtime=%llu requested=%llu initialize=%d interval_ms=%u target_pid=%u\n",
        (unsigned long long)runtime_version, (unsigned long long)requested, result, interval, target_pid);
    if (ADLX_FAILED(result) || !system) goto done;
    initialized = 1;
    result = system->pVtbl->GetGPUs(system, &gpus);
    if (ADLX_FAILED(result) || !gpus) goto done;
    unsigned matches = 0;
    for (adlx_uint i = gpus->pVtbl->Begin(gpus); i < gpus->pVtbl->End(gpus); ++i) {
        IADLXGPU* gpu = NULL;
        if (ADLX_FAILED(gpus->pVtbl->At_GPUList(gpus, i, &gpu)) || !gpu) goto done;
        const char* name = NULL; const char* pnp = NULL; adlx_int id = -1;
        gpu->pVtbl->Name(gpu, &name);
        gpu->pVtbl->PNPString(gpu, &pnp);
        gpu->pVtbl->UniqueId(gpu, &id);
        fprintf(stderr, "ADLX GPU index=%u id=%d name=%s pnp=%s\n", i, id, name ? name : "unknown", pnp ? pnp : "unknown");
        if (name && strstr(name, gpu_name)) {
            ++matches;
            if (!selected) { selected = gpu; gpu = NULL; }
        }
        if (gpu) gpu->pVtbl->Release(gpu);
    }
    if (matches != 1) { fprintf(stderr, "Expected one GPU matching '%s', found %u\n", gpu_name, matches); goto done; }
    result = system->pVtbl->GetPerformanceMonitoringServices(system, &service);
    if (ADLX_FAILED(result) || !service) goto done;
    LARGE_INTEGER frequency, qpc;
    QueryPerformanceFrequency(&frequency);
    // Preserve the driver's timestamp verbatim: the installed ADLX 1.5 reports
    // uptime here despite SDK documentation describing epoch time. Use our
    // independent QPC/UTC stamps for correlation with PresentMon.
    printf("qpc_ms,host_unix_ms,sensor_timestamp_ms,gpu_usage_pct,gpu_clock_mhz,vram_clock_mhz,gpu_power_w,board_power_w,temperature_c,hotspot_c,poll_ms\n");
    fflush(stdout);
    if (delay) Sleep(delay);
    ULONGLONG start = GetTickCount64();
    unsigned samples = 0;
    while (GetTickCount64() - start < (ULONGLONG)seconds * 1000) {
        if (target && WaitForSingleObject(target, 0) == WAIT_OBJECT_0) break;
        LARGE_INTEGER before, after;
        QueryPerformanceCounter(&before);
        IADLXGPUMetrics* metrics = NULL;
        result = service->pVtbl->GetCurrentGPUMetrics(service, selected, &metrics);
        QueryPerformanceCounter(&after);
        if (ADLX_FAILED(result) || !metrics) { fprintf(stderr, "GetCurrentGPUMetrics result=%d\n", result); goto done; }
        FILETIME time; ULARGE_INTEGER unix_time;
        GetSystemTimePreciseAsFileTime(&time);
        unix_time.LowPart = time.dwLowDateTime; unix_time.HighPart = time.dwHighDateTime;
        QueryPerformanceCounter(&qpc);
        adlx_int64 sensor_time = 0;
        printf("%.6f,%llu,", qpc.QuadPart * 1000. / frequency.QuadPart,
            (unsigned long long)((unix_time.QuadPart - 116444736000000000ULL) / 10000));
        if (ADLX_SUCCEEDED(metrics->pVtbl->TimeStamp(metrics, &sensor_time))) printf("%lld", (long long)sensor_time);
        double real = -1; int integer = -1;
        printf(","); result = metrics->pVtbl->GPUUsage(metrics, &real); real_value(result, real);
        printf(","); result = metrics->pVtbl->GPUClockSpeed(metrics, &integer); int_value(result, integer);
        printf(","); result = metrics->pVtbl->GPUVRAMClockSpeed(metrics, &integer); int_value(result, integer);
        printf(","); result = metrics->pVtbl->GPUPower(metrics, &real); real_value(result, real);
        printf(","); result = metrics->pVtbl->GPUTotalBoardPower(metrics, &real); real_value(result, real);
        printf(","); result = metrics->pVtbl->GPUTemperature(metrics, &real); real_value(result, real);
        printf(","); result = metrics->pVtbl->GPUHotspotTemperature(metrics, &real); real_value(result, real);
        printf(",%.6f\n", (after.QuadPart - before.QuadPart) * 1000. / frequency.QuadPart);
        metrics->pVtbl->Release(metrics);
        ++samples; fflush(stdout);
        ULONGLONG elapsed = GetTickCount64() - start;
        ULONGLONG next = (ULONGLONG)samples * interval;
        if (next > elapsed) Sleep((DWORD)(next - elapsed));
    }
    fprintf(stderr, "ADLX collected=%u read_only=1\n", samples);
    exit_code = samples ? 0 : 1;
done:
    // All acquired interfaces must be released before unloading ADLX.
    if (service) service->pVtbl->Release(service);
    if (selected) selected->pVtbl->Release(selected);
    if (gpus) gpus->pVtbl->Release(gpus);
    if (initialized) {
        ADLX_RESULT status = terminate();
        fprintf(stderr, "ADLX terminate=%d\n", status);
        if (ADLX_FAILED(status)) exit_code = 1;
    }
    if (dll) FreeLibrary(dll);
    if (target) CloseHandle(target);
    return exit_code;
}
