// Native D3D12/Vulkan prerequisite query for the RR engine adapter. No dispatch.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <volk.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 1;
    IDXGIAdapter1* selected = nullptr;
    DXGI_ADAPTER_DESC1 adapter{};
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* next = nullptr;
        if (factory->EnumAdapters1(i, &next) == DXGI_ERROR_NOT_FOUND) break;
        if (!next) break;
        DXGI_ADAPTER_DESC1 desc{};
        next->GetDesc1(&desc);
        if (desc.VendorId == 0x1002 && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            selected = next; adapter = desc; break;
        }
        next->Release();
    }
    factory->Release();
    if (!selected) { std::puts("FAIL no native AMD D3D12 adapter"); return 1; }
    ID3D12Device* device = nullptr;
    const HRESULT created = D3D12CreateDevice(selected, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    selected->Release();
    if (FAILED(created)) { std::puts("FAIL D3D12 device creation"); return 1; }
    device->Release();
    if (volkInitialize() != VK_SUCCESS) return 1;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) return 1;
    volkLoadInstance(instance);
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    bool ready = false;
    for (auto physical : devices) {
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
        vkGetPhysicalDeviceProperties2(physical, &properties);
        if (!id.deviceLUIDValid || std::memcmp(id.deviceLUID, &adapter.AdapterLuid, VK_LUID_SIZE)) continue;
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, extensions.data());
        auto has = [&](const char* name) {
            return std::any_of(extensions.begin(), extensions.end(), [&](const auto& e) { return !std::strcmp(name, e.extensionName); });
        };
        VkPhysicalDeviceExternalBufferInfo buffer{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
        VkExternalBufferProperties memory{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        vkGetPhysicalDeviceExternalBufferProperties(physical, &buffer, &memory);
        VkPhysicalDeviceExternalSemaphoreInfo semaphore{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
        semaphore.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
        VkExternalSemaphoreProperties fence{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
        vkGetPhysicalDeviceExternalSemaphoreProperties(physical, &semaphore, &fence);
        const bool memoryImport = (memory.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        const bool fenceImport = (fence.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
        const bool memoryExtension = has("VK_KHR_external_memory_win32");
        const bool fenceExtension = has("VK_KHR_external_semaphore_win32");
        std::printf("RR_INTEROP device=%s luid_match=1 memory_win32=%u d3d12_buffer_import=%u dedicated_only=%u semaphore_win32=%u d3d12_fence_import=%u\n",
            properties.properties.deviceName, memoryExtension, memoryImport,
            unsigned((memory.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0),
            fenceExtension, fenceImport);
        ready = memoryExtension && memoryImport && fenceExtension && fenceImport;
    }
    vkDestroyInstance(instance, nullptr);
    std::puts(ready ? "PASS RR_INTEROP_CAPABILITIES import_or_dispatch_executed=0" : "FAIL RR_INTEROP_CAPABILITIES");
    return ready ? 0 : 1;
}
