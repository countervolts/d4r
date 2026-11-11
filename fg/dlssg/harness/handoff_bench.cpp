// handoff_bench: GPU cost of one D3D12 -> other device -> D3D12 round trip on the same adapter, as the FG executor
// (Vulkan) and d3dhip (HIP) do it: D3D12 Signal(a), the other device waits a, runs a tiny fill, signals b, D3D12 Wait(b).
// Measured with D3D12 timestamps around the round trip; "none" is the same two timestamps with no hand-off between.
// Usage: handoff_bench [iterations] [load_mb] [fill_mb]   (load_mb: a D3D12 copy before each round trip, GPU busy;
// fill_mb: what the other device writes, default 4 KB). Build (Git Bash, llvm-mingw, Vulkan SDK, ROCm headers):
//   clang++ -std=c++17 -O2 -I$VULKAN_SDK/Include -I"C:/Program Files/AMD/ROCm/7.2/include" -static
//     -o handoff_bench.exe handoff_bench.cpp -ld3d12 -ldxgi -ldxguid
#define WIDL_EXPLICIT_AGGREGATE_RETURNS
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#define __HIP_PLATFORM_AMD__
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <vulkan/vulkan.h>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { std::printf("FAILED 0x%08lx: %s (line %d)\n", (unsigned long)hr_, #x, __LINE__); std::exit(1); } } while (0)
#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { std::printf("Vulkan %d: %s (line %d)\n", (int)r_, #x, __LINE__); std::exit(1); } } while (0)
#define HIPC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %d: %s (line %d)\n", (int)e_, #x, __LINE__); std::exit(1); } } while (0)

static ID3D12Device* dev;
static ID3D12CommandQueue* q;
static ID3D12Fence* fence;      // shared: the hand-off fence
static ID3D12Fence* pace;       // CPU pacing
static uint64_t fv = 0, pv = 0;
static LUID luid;
static HANDLE fenceHandle;
static uint64_t fillBytes = 4096;

// ---------------------------------------------------------------- Vulkan
struct Vk {
    PFN_vkGetInstanceProcAddr gipa;
#define F(n) PFN_vk##n n;
    F(CreateInstance) F(EnumeratePhysicalDevices) F(GetPhysicalDeviceProperties2) F(GetPhysicalDeviceQueueFamilyProperties)
    F(CreateDevice) F(GetDeviceProcAddr) F(GetDeviceQueue) F(CreateSemaphore) F(ImportSemaphoreWin32HandleKHR)
    F(CreateCommandPool) F(AllocateCommandBuffers) F(BeginCommandBuffer) F(EndCommandBuffer) F(CmdFillBuffer)
    F(QueueSubmit) F(CreateBuffer) F(GetBufferMemoryRequirements) F(AllocateMemory) F(BindBufferMemory)
    F(GetPhysicalDeviceMemoryProperties) F(DeviceWaitIdle)
#undef F
    VkInstance inst; VkPhysicalDevice pd; VkDevice d; VkQueue q; VkSemaphore sem; VkCommandBuffer cmd;
    bool ok = false;
} vk;

static bool vkInit(bool computeFamily)
{
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) return false;
    vk.gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(lib, "vkGetInstanceProcAddr");
    vk.CreateInstance = (PFN_vkCreateInstance)vk.gipa(nullptr, "vkCreateInstance");
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VKC(vk.CreateInstance(&ici, nullptr, &vk.inst));
#define I(n) vk.n = (PFN_vk##n)vk.gipa(vk.inst, "vk" #n);
    I(EnumeratePhysicalDevices) I(GetPhysicalDeviceProperties2) I(GetPhysicalDeviceQueueFamilyProperties) I(CreateDevice)
    I(GetDeviceProcAddr) I(GetPhysicalDeviceMemoryProperties)
#undef I
    uint32_t n = 0;
    vk.EnumeratePhysicalDevices(vk.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vk.EnumeratePhysicalDevices(vk.inst, &n, pds.data());
    vk.pd = VK_NULL_HANDLE;
    for (auto p : pds) {
        VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &idp};
        vk.GetPhysicalDeviceProperties2(p, &pp);
        if (idp.deviceLUIDValid && !memcmp(idp.deviceLUID, &luid, 8)) vk.pd = p;
    }
    if (!vk.pd) { std::printf("Vulkan: no device with the D3D12 adapter's LUID\n"); return false; }
    vk.GetPhysicalDeviceQueueFamilyProperties(vk.pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qf(n);
    vk.GetPhysicalDeviceQueueFamilyProperties(vk.pd, &n, qf.data());
    uint32_t fam = 0;
    for (uint32_t i = 0; i < n; i++) {
        bool gfx = qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT, cmp = qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT;
        if (computeFamily ? (cmp && !gfx) : gfx) { fam = i; break; }
    }
    float prio = 1;
    VkDeviceQueueCreateInfo dq{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dq.queueFamilyIndex = fam;
    dq.queueCount = 1;
    dq.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.timelineSemaphore = VK_TRUE;
    const char* ext[] = {VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &v12};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dq;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = ext;
    VKC(vk.CreateDevice(vk.pd, &dci, nullptr, &vk.d));
#define D(n) vk.n = (PFN_vk##n)vk.GetDeviceProcAddr(vk.d, "vk" #n);
    D(GetDeviceQueue) D(CreateSemaphore) D(ImportSemaphoreWin32HandleKHR) D(CreateCommandPool) D(AllocateCommandBuffers)
    D(BeginCommandBuffer) D(EndCommandBuffer) D(CmdFillBuffer) D(QueueSubmit) D(CreateBuffer) D(GetBufferMemoryRequirements)
    D(AllocateMemory) D(BindBufferMemory) D(DeviceWaitIdle)
#undef D
    vk.GetDeviceQueue(vk.d, fam, 0, &vk.q);
    VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
    VKC(vk.CreateSemaphore(vk.d, &sci, nullptr, &vk.sem));
    VkImportSemaphoreWin32HandleInfoKHR ii{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
    ii.semaphore = vk.sem;
    ii.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
    ii.handle = fenceHandle;
    VKC(vk.ImportSemaphoreWin32HandleKHR(vk.d, &ii));
    // a 4 KB device-local buffer and one command buffer that fills it (the "kernels")
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = fillBytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf;
    VKC(vk.CreateBuffer(vk.d, &bci, nullptr, &buf));
    VkMemoryRequirements mr;
    vk.GetBufferMemoryRequirements(vk.d, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vk.GetPhysicalDeviceMemoryProperties(vk.pd, &mp);
    uint32_t mt = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((mr.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mt;
    VkDeviceMemory mem;
    VKC(vk.AllocateMemory(vk.d, &mai, nullptr, &mem));
    VKC(vk.BindBufferMemory(vk.d, buf, mem, 0));
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = fam;
    VkCommandPool pool;
    VKC(vk.CreateCommandPool(vk.d, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.commandBufferCount = 1;
    VKC(vk.AllocateCommandBuffers(vk.d, &cai, &vk.cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vk.BeginCommandBuffer(vk.cmd, &bi);   // reusable (no ONE_TIME_SUBMIT): submitted once per round trip
    vk.CmdFillBuffer(vk.cmd, buf, 0, fillBytes, 0x12345678);
    VKC(vk.EndCommandBuffer(vk.cmd));
    std::printf("Vulkan: queue family %u (%s)\n", fam, computeFamily ? "compute-only" : "graphics");
    return vk.ok = true;
}

static void vkRoundTrip(uint64_t a, uint64_t b)
{
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.waitSemaphoreValueCount = 1;
    ts.pWaitSemaphoreValues = &a;
    ts.signalSemaphoreValueCount = 1;
    ts.pSignalSemaphoreValues = &b;
    VkPipelineStageFlags st = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &vk.sem;
    si.pWaitDstStageMask = &st;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &vk.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &vk.sem;
    VKC(vk.QueueSubmit(vk.q, 1, &si, VK_NULL_HANDLE));
}

// ---------------------------------------------------------------- HIP
struct Hip {
#define F(n) decltype(&::n) n;
    F(hipInit) F(hipGetDeviceCount) F(hipSetDevice) F(hipStreamCreate) F(hipMemsetAsync)
    F(hipImportExternalSemaphore) F(hipWaitExternalSemaphoresAsync) F(hipSignalExternalSemaphoresAsync)
    F(hipDeviceSynchronize)
#undef F
    decltype(&::hipGetDeviceProperties) props;
    hipError_t (*hipMalloc)(void**, size_t);
    hipStream_t s; hipExternalSemaphore_t sem; void* buf;
    bool ok = false;
} hip;

static bool hipInitAll()
{
    HMODULE lib = LoadLibraryA("amdhip64_7.dll");
    if (!lib) { std::printf("HIP: no amdhip64_7.dll\n"); return false; }
#define F(n) hip.n = (decltype(&::n))GetProcAddress(lib, #n);
    F(hipInit) F(hipGetDeviceCount) F(hipSetDevice) F(hipStreamCreate) F(hipMemsetAsync)
    F(hipImportExternalSemaphore) F(hipWaitExternalSemaphoresAsync) F(hipSignalExternalSemaphoresAsync)
    F(hipDeviceSynchronize)
#undef F
    hip.props = (decltype(&::hipGetDeviceProperties))GetProcAddress(lib, "hipGetDevicePropertiesR0600");
    hip.hipMalloc = (hipError_t (*)(void**, size_t))GetProcAddress(lib, "hipMalloc");
    HIPC(hip.hipInit(0));
    int n = 0;
    HIPC(hip.hipGetDeviceCount(&n));
    int pick = -1;
    for (int i = 0; i < n; i++) {
        hipDeviceProp_t p{};
        HIPC(hip.props(&p, i));
        if (!memcmp(p.luid, &luid, 8)) pick = i;
        std::printf("HIP device %d: %s (%s)\n", i, p.name, p.gcnArchName);
    }
    if (pick < 0) { std::printf("HIP: no device with the D3D12 adapter's LUID\n"); return false; }
    HIPC(hip.hipSetDevice(pick));
    HIPC(hip.hipStreamCreate(&hip.s));
    HIPC(hip.hipMalloc(&hip.buf, fillBytes));
    hipExternalSemaphoreHandleDesc d{};
    d.type = hipExternalSemaphoreHandleTypeD3D12Fence;
    d.handle.win32.handle = fenceHandle;
    HIPC(hip.hipImportExternalSemaphore(&hip.sem, &d));
    return hip.ok = true;
}

static void hipRoundTrip(uint64_t a, uint64_t b)
{
    hipExternalSemaphoreWaitParams w{};
    w.params.fence.value = a;
    HIPC(hip.hipWaitExternalSemaphoresAsync(&hip.sem, &w, 1, hip.s));
    HIPC(hip.hipMemsetAsync(hip.buf, 0x5a, fillBytes, hip.s));
    hipExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = b;
    HIPC(hip.hipSignalExternalSemaphoresAsync(&hip.sem, &sp, 1, hip.s));
}

// ---------------------------------------------------------------- D3D12 loop
struct Slot { ID3D12CommandAllocator* a; ID3D12GraphicsCommandList* l0; ID3D12GraphicsCommandList* l1; uint64_t done = 0; };

int main(int argc, char** argv)
{
    int iters = argc > 1 ? std::atoi(argv[1]) : 1000;
    int loadMb = argc > 2 ? std::atoi(argv[2]) : 0;
    if (argc > 3) fillBytes = (uint64_t)std::atoi(argv[3]) << 20;
    IDXGIFactory6* fac;
    CHK(CreateDXGIFactory2(0, __uuidof(IDXGIFactory6), (void**)&fac));
    IDXGIAdapter1* ad;
    CHK(fac->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, __uuidof(IDXGIAdapter1), (void**)&ad));
    DXGI_ADAPTER_DESC1 ds;
    ad->GetDesc1(&ds);
    luid = ds.AdapterLuid;
    std::printf("adapter: %ls\n", ds.Description);
    CHK(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), (void**)&dev));
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CHK(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&q));
    CHK(dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&fence));
    CHK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&pace));
    CHK(dev->CreateSharedHandle(fence, nullptr, GENERIC_ALL, nullptr, &fenceHandle));
    UINT64 freq;
    q->GetTimestampFrequency(&freq);

    const int kSlots = 3;
    D3D12_QUERY_HEAP_DESC qhd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, (UINT)(2 * iters)};
    ID3D12QueryHeap* qh;
    CHK(dev->CreateQueryHeap(&qhd, __uuidof(ID3D12QueryHeap), (void**)&qh));
    auto buffer = [](D3D12_HEAP_TYPE t, uint64_t size, D3D12_RESOURCE_STATES st) {
        D3D12_HEAP_PROPERTIES hp{t};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = size;
        rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* r;
        CHK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, __uuidof(ID3D12Resource), (void**)&r));
        return r;
    };
    ID3D12Resource* rb = buffer(D3D12_HEAP_TYPE_READBACK, 16ull * iters, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource *ls = nullptr, *ld = nullptr;
    if (loadMb) {
        ls = buffer(D3D12_HEAP_TYPE_DEFAULT, (uint64_t)loadMb << 20, D3D12_RESOURCE_STATE_COMMON);
        ld = buffer(D3D12_HEAP_TYPE_DEFAULT, (uint64_t)loadMb << 20, D3D12_RESOURCE_STATE_COMMON);
    }
    Slot slots[kSlots];
    for (auto& s : slots) {
        CHK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&s.a));
        CHK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.a, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&s.l0));
        CHK(s.l0->Close());
        CHK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.a, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&s.l1));
        CHK(s.l1->Close());
    }

    bool haveVk = vkInit(false);
    bool haveHip = hipInitAll();
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    auto run = [&](const char* name, void (*other)(uint64_t, uint64_t)) {
        for (int i = 0; i < iters; i++) {
            Slot& s = slots[i % kSlots];
            if (s.done && pace->GetCompletedValue() < s.done) {
                pace->SetEventOnCompletion(s.done, ev);
                WaitForSingleObject(ev, 5000);
            }
            CHK(s.a->Reset());
            CHK(s.l0->Reset(s.a, nullptr));
            if (loadMb) s.l0->CopyBufferRegion(ld, 0, ls, 0, (uint64_t)loadMb << 20);
            s.l0->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i);
            CHK(s.l0->Close());
            CHK(s.l1->Reset(s.a, nullptr));
            s.l1->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i + 1);
            s.l1->ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i, 2, rb, 16ull * i);
            CHK(s.l1->Close());
            ID3D12CommandList* l0 = s.l0;
            ID3D12CommandList* l1 = s.l1;
            q->ExecuteCommandLists(1, &l0);
            if (other) {
                uint64_t a = ++fv, b = ++fv;
                q->Signal(fence, a);
                other(a, b);
                q->Wait(fence, b);
            }
            q->ExecuteCommandLists(1, &l1);
            s.done = ++pv;
            q->Signal(pace, s.done);
        }
        pace->SetEventOnCompletion(pv, ev);
        WaitForSingleObject(ev, 10000);
        uint64_t* m;
        D3D12_RANGE r{0, 16ull * iters};
        CHK(rb->Map(0, &r, (void**)&m));
        std::vector<double> us;
        for (int i = iters / 10; i < iters; i++) us.push_back((double)(m[2 * i + 1] - m[2 * i]) * 1e6 / (double)freq);
        D3D12_RANGE w{0, 0};
        rb->Unmap(0, &w);
        std::sort(us.begin(), us.end());
        std::printf("%-22s median %8.1f us   p10 %8.1f   p90 %8.1f\n", name, us[us.size() / 2], us[us.size() / 10],
                    us[us.size() * 9 / 10]);
    };
    for (int round = 0; round < 3; round++) {
        std::printf("round %d (load %d MB)\n", round, loadMb);
        run("none", nullptr);
        if (haveVk) run("vulkan (graphics q)", vkRoundTrip);
        if (haveHip) run("hip", hipRoundTrip);
    }
    if (haveHip) hip.hipDeviceSynchronize();
    if (haveVk) vk.DeviceWaitIdle(vk.d);
    return 0;
}
