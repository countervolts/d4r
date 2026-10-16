#pragma once
#include <d3d12.h>
#include <vulkan/vulkan_core.h>

MIDL_INTERFACE("39da4e09-bd1c-4198-9fae-86bbe3be41fd")
ID3D12DXVKInteropDevice : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetDXGIAdapter(REFIID iid, void** object) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetInstanceExtensions(UINT* count, const char** extensions) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceExtensions(UINT* count, const char** extensions) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceFeatures(const VkPhysicalDeviceFeatures2** features) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanHandles(VkInstance* instance, VkPhysicalDevice* physical,
                                                       VkDevice* device) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanQueueInfo(ID3D12CommandQueue* queue, VkQueue* vkQueue,
                                                         UINT32* family) = 0;
    virtual void STDMETHODCALLTYPE GetVulkanImageLayout(ID3D12Resource* resource, D3D12_RESOURCE_STATES state,
                                                        VkImageLayout* layout) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo(ID3D12Resource* resource, UINT64* handle,
                                                            UINT64* offset) = 0;
    virtual HRESULT STDMETHODCALLTYPE LockCommandQueue(ID3D12CommandQueue* queue) = 0;
    virtual HRESULT STDMETHODCALLTYPE UnlockCommandQueue(ID3D12CommandQueue* queue) = 0;
};

MIDL_INTERFACE("902d8115-59eb-4406-9518-fe00f991ee65")
ID3D12DXVKInteropDevice1 : public ID3D12DXVKInteropDevice
{
    virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo1(ID3D12Resource* resource, UINT64* handle, UINT64* offset,
                                                             VkFormat* format) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandQueue(const D3D12_COMMAND_QUEUE_DESC* desc, UINT32 family,
                                                                ID3D12CommandQueue** queue) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateInteropCommandAllocator(D3D12_COMMAND_LIST_TYPE type, UINT32 family,
                                                                    ID3D12CommandAllocator** allocator) = 0;
    virtual HRESULT STDMETHODCALLTYPE BeginVkCommandBufferInterop(ID3D12CommandList* list, VkCommandBuffer* buffer) = 0;
    virtual HRESULT STDMETHODCALLTYPE EndVkCommandBufferInterop(ID3D12CommandList* list) = 0;
};
__CRT_UUID_DECL(ID3D12DXVKInteropDevice1, 0x902d8115, 0x59eb, 0x4406, 0x95, 0x18, 0xfe, 0x00, 0xf9, 0x91, 0xee, 0x65)

// Added by the d4r vkd3d-proton patch (patches/vkd3d-proton/): ends the list's
// current Vulkan command buffer; what is recorded afterwards is submitted
// separately, with a queue-level wait for the timeline semaphore value.
MIDL_INTERFACE("5a7c8b3e-2f61-4d0e-9c1a-7e3b52d4a901")
ID3D12DXVKInteropDeviceD4R : public ID3D12DXVKInteropDevice1
{
    virtual HRESULT STDMETHODCALLTYPE LockVulkanQueue(ID3D12CommandQueue* queue) = 0;
    virtual HRESULT STDMETHODCALLTYPE UnlockVulkanQueue(ID3D12CommandQueue* queue) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanHeapInfo(ID3D12Heap* heap, UINT64* memory, UINT64* offset,
                                                        UINT32* type) = 0;
    virtual HRESULT STDMETHODCALLTYPE SplitCommandListForExternalWait(ID3D12CommandList* list, UINT64 semaphore,
                                                                      UINT64 value) = 0;
};
__CRT_UUID_DECL(ID3D12DXVKInteropDeviceD4R, 0x5a7c8b3e, 0x2f61, 0x4d0e, 0x9c, 0x1a, 0x7e, 0x3b, 0x52, 0xd4, 0xa9, 0x01)

MIDL_INTERFACE("4f8c9462-8197-4dc1-a584-cf44d9706db2")
ID3D12DXVKInteropDeviceD4R2 : public ID3D12DXVKInteropDeviceD4R
{
    virtual HRESULT STDMETHODCALLTYPE RetainExternalResources(ID3D12CommandList* list, IUnknown* resources) = 0;
};
__CRT_UUID_DECL(ID3D12DXVKInteropDeviceD4R2, 0x4f8c9462, 0x8197, 0x4dc1, 0xa5, 0x84, 0xcf, 0x44, 0xd9, 0x70, 0x6d, 0xb2)

