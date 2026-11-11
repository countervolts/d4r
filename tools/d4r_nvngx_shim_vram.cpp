// d4r_nvngx_shim_vram.cpp: inputs and output kept in VRAM. Both ways of getting there: vkd3d-proton's
// VkDevice, whose VkBuffers are imported into CUDA, and native Windows, where D3D12 resources are shared
// with HIP instead. The pools, the descriptor table, the blit capability probes and the copies.

// the command-list stubs live in d4r_nvngx_shim.cpp, which assembles them once
#define D4R_D3D12_SPLIT_DECL 1
#include "d4r_nvngx_shim_internal.h"

static bool record_motion_constants(ID3D12GraphicsCommandList* list, Staging& target,
        ID3D12Resource* motion,uint32_t frame,uint32_t guidValue,uint64_t shader,uint64_t expectedBytes)
{
    const GUID guid = {guidValue,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
    NativeCameraSource source = {};
    UINT bytes = sizeof(source);
    if (FAILED(motion->GetPrivateData(guid, &bytes, &source)) || bytes != sizeof(source) ||
        (source.shader != shader && !(shader==UINT64_C(0xc5d8688e46fd5c66) && source.shader==UINT64_C(0xafe29e382bf4bdb4))) || source.bytes != expectedBytes || !source.buffer)
    {
        logf("native motion capture frame %u: composition constants unavailable", frame);
        return false;
    }
    if (target.total && target.total!=source.bytes) target.retire();
    if (target.buffer == nullptr &&
        !create_buffer(D3D12_HEAP_TYPE_READBACK, source.bytes, &target.buffer, &target.mapped))
        return false;
    UINT64 handle = 0, offset = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    if (g_vk.copyBuffer == nullptr ||
        FAILED(g_vk.interop->GetVulkanResourceInfo1(target.buffer, &handle, &offset, &format)) || !handle)
        return false;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &cmd)))
        return false;
    const VkMemoryBarrier before = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        1, &before, 0, nullptr, 0, nullptr);
    const VkBufferCopy region = {source.offset, offset, source.bytes};
    g_vk.copyBuffer(cmd, reinterpret_cast<VkBuffer>(source.buffer), reinterpret_cast<VkBuffer>(handle), 1, &region);
    const VkMemoryBarrier after = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
        1, &after, 0, nullptr, 0, nullptr);
    const bool recorded = SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
    if (recorded)
    {
        target.total = source.bytes;
        logf("native motion capture frame %u: %llu composition constant bytes for %ux%u", frame,
            static_cast<unsigned long long>(source.bytes), source.width, source.height);
    }
    return recorded;
}

// Diagnostic of GPU-owned camera storage, captured alongside the old raw-source
// path to verify the ownership change without changing NGX inputs.

bool record_owned_motion_constants(ID3D12GraphicsCommandList* list, Staging& target,
        ID3D12Resource* motion,uint32_t guidValue,uint64_t expectedBytes)
{
    const GUID guid={guidValue,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
    ID3D12Resource* source=nullptr;
    UINT size=sizeof(source);
    if(FAILED(motion->GetPrivateData(guid,&size,&source)) || !source) return false;
    D3D12_RESOURCE_DESC desc;source->GetDesc(&desc);
    bool recorded=false;
    if(desc.Dimension==D3D12_RESOURCE_DIMENSION_BUFFER && desc.Width==expectedBytes &&
        g_vk.lifetime && SUCCEEDED(g_vk.lifetime->RetainExternalResources(list,source)))
    {
        if(target.total && target.total!=expectedBytes) target.retire();
        if(target.buffer || create_buffer(D3D12_HEAP_TYPE_READBACK,expectedBytes,&target.buffer,&target.mapped))
        {
            list->CopyBufferRegion(target.buffer,0,source,0,expectedBytes);
            target.total=expectedBytes;recorded=true;
        }
    }
    source->Release();return recorded;
}

static bool record_original_motion_source(ID3D12GraphicsCommandList* list, ID3D12Resource* motion,
                                          Staging& target, bool depth,uint32_t guidValue=0,uint64_t shader=UINT64_C(0xc5d8688e46fd5c66),uint32_t texelBytes=4,uint64_t expectedImage=0)
{
    const GUID guid = {guidValue ? guidValue : depth ? 0x7536783au : 0x75367839u,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
    NativeImageSource source = {};
    UINT bytes = sizeof(source);
    const uint32_t aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    if (FAILED(motion->GetPrivateData(guid, &bytes, &source)) || bytes != sizeof(source) ||
        (source.shader != shader && !(shader==UINT64_C(0xc5d8688e46fd5c66) && source.shader==UINT64_C(0xafe29e382bf4bdb4))) || !source.image || source.aspect != aspect ||
        !source.width || !source.height || source.width > 4096 || source.height > 4096 ||
        source.layout == VK_IMAGE_LAYOUT_UNDEFINED || !texelBytes || texelBytes>16 || (expectedImage && source.image!=expectedImage))
        return false;
    const uint64_t total = static_cast<uint64_t>(source.width) * source.height * texelBytes;
    if (target.total != total) target.retire();
    if (target.buffer == nullptr && !create_buffer(D3D12_HEAP_TYPE_READBACK,total,&target.buffer,&target.mapped))
        return false;
    target.total = total; target.width = source.width; target.height = source.height;
    UINT64 handle = 0, offset = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    if (FAILED(g_vk.interop->GetVulkanResourceInfo1(target.buffer,&handle,&offset,&format)) || !handle)
        return false;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list,&cmd))) return false;
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = static_cast<VkImageLayout>(source.layout);
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = reinterpret_cast<VkImage>(source.image);
    barrier.subresourceRange = {aspect,0,1,0,1};
    g_vk.barrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
    VkBufferImageCopy region = {};
    region.bufferOffset = offset;
    region.imageSubresource = {aspect,0,0,1};
    region.imageExtent = {source.width,source.height,1};
    g_vk.copyImageToBuffer(cmd,barrier.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,reinterpret_cast<VkBuffer>(handle),1,&region);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = static_cast<VkImageLayout>(source.layout);
    g_vk.barrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&barrier);
    const VkMemoryBarrier host = {VK_STRUCTURE_TYPE_MEMORY_BARRIER,nullptr,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT};
    g_vk.barrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host,0,nullptr,0,nullptr);
    return SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
}

void record_native_motion_capture(ID3D12GraphicsCommandList* list, InputSlot& slot,
                                         ID3D12Resource* motion, uint32_t frame)
{
    slot.nativeMotionCaptureFrame = 0;
    slot.nativeDepthCaptureReady = slot.nativeSceneCaptureReady = false;
    slot.nativeOpaqueCaptureReady = slot.nativeGeometryCaptureReady = false;
    if(slot.nativeValidityOwner) { slot.nativeValidityOwner->Release();slot.nativeValidityOwner=nullptr; }
    slot.materialSceneCaptureReady=false;
    slot.glassMaskCaptureReady=false;
    slot.originalOpaqueCopyCaptureReady=false;
    slot.nativeGpuStageValid=false;
    slot.ownedValidityCaptureReady=slot.glassValidityCaptureReady=false;
    slot.ownedDepthCaptureReady=false;
    slot.ownedSceneCaptureReady=slot.ownedMatrixCaptureReady=false;
    slot.nativePrimaryDepthCaptureReady=slot.opaqueSceneCaptureReady=false;
    slot.opaqueMatrixCaptureReady=slot.originalPrimaryDepthCaptureReady=false;
    slot.originalGeometryCaptureReady = slot.originalHardwareDepthCaptureReady = false;
    char directory[MAX_PATH];
    const DWORD length = GetEnvironmentVariableA("D4R_NATIVE_MV_CAPTURE_DIR", directory, MAX_PATH);
    if(!length || length>=MAX_PATH) return;
    // Freeze native capture selection before recording any per-frame copies.
    // A worker noticing the trigger later cannot retroactively capture them.
    arm_native_capture(frame);
    if(!dump_frame_selected(frame,"D4R_NATIVE_MV_CAPTURE_START",
        "D4R_NATIVE_MV_CAPTURE_COUNT","D4R_NATIVE_MV_CAPTURE_EVERY")) return;
    {
        const GUID maskGuid={0x75367847,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
        const GUID formatGuid={0x75367849,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
        ID3D12Resource* mask=nullptr;UINT size=sizeof(mask),formatSize=sizeof(slot.glassMaskFormat);
        if(SUCCEEDED(motion->GetPrivateData(maskGuid,&size,&mask)) && mask)
        {
            UINT64 handle=0,offset=0;VkFormat format=VK_FORMAT_UNDEFINED;uint32_t bytes=0;
            D3D12_RESOURCE_DESC maskDesc;mask->GetDesc(&maskDesc);
            if(SUCCEEDED(motion->GetPrivateData(formatGuid,&formatSize,&slot.glassMaskFormat)))
                switch(static_cast<DXGI_FORMAT>(slot.glassMaskFormat))
                {
                    case DXGI_FORMAT_R8_UNORM:bytes=1;break;
                    case DXGI_FORMAT_R16_FLOAT:case DXGI_FORMAT_R16_UNORM:bytes=2;break;
                    case DXGI_FORMAT_R32_FLOAT:case DXGI_FORMAT_D32_FLOAT:case DXGI_FORMAT_R16G16_FLOAT:case DXGI_FORMAT_R8G8B8A8_UNORM:bytes=4;break;
                    case DXGI_FORMAT_R16G16B16A16_FLOAT:case DXGI_FORMAT_R32G32_FLOAT:bytes=8;break;
                    case DXGI_FORMAT_R32G32B32A32_FLOAT:bytes=16;break;
                    default:break;
                }
            if(bytes && maskDesc.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D && maskDesc.SampleDesc.Count==1 &&
                maskDesc.DepthOrArraySize==1 && g_vk.lifetime && SUCCEEDED(g_vk.lifetime->RetainExternalResources(list,mask)) &&
                SUCCEEDED(g_vk.interop->GetVulkanResourceInfo1(mask,&handle,&offset,&format)))
                slot.glassMaskCaptureReady=record_original_motion_source(list,motion,slot.glassMaskCapture,
                    format==VK_FORMAT_D32_SFLOAT,0x75367848,UINT64_C(0xc5d8688e46fd5c66),bytes,handle);
            logf("native material mask capture frame %u: format=%u ready=%d",frame,slot.glassMaskFormat,slot.glassMaskCaptureReady);
            mask->Release();
        }
    }
    slot.materialSceneCaptureReady=record_owned_motion_constants(list,slot.materialSceneCapture,motion,0x7536784a,1136);
    // Opaque capture is independent of whether this scene runs the glass composition pass.
    static const GUID opaqueGuid = {0x7536783b,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
    ID3D12Resource* opaque = nullptr;
    UINT bytes = sizeof(opaque);
    if (SUCCEEDED(motion->GetPrivateData(opaqueGuid,&bytes,&opaque)) && opaque != nullptr)
    {
        D3D12_RESOURCE_DESC opaqueDesc;
        opaque->GetDesc(&opaqueDesc);
        if (opaqueDesc.Format == DXGI_FORMAT_R16G16_FLOAT && g_vk.lifetime != nullptr &&
            ensure_staging(slot.nativeOpaqueCapture,opaque,D3D12_HEAP_TYPE_READBACK) &&
            SUCCEEDED(g_vk.lifetime->RetainExternalResources(list,opaque)))
        {
            copy_to_staging(list,opaque,slot.nativeOpaqueCapture,D3D12_RESOURCE_STATE_COMMON);
            char opaquePath[MAX_PATH];
            const int size=std::snprintf(opaquePath,MAX_PATH,"%s\\frame-%06u-%ux%u.geometry.rg16f",directory,
                frame,static_cast<UINT>(opaqueDesc.Width),opaqueDesc.Height);
            if (size>0 && size<MAX_PATH)
            {
                slot.nativeMotionCapturePath=opaquePath;
                slot.nativeMotionCaptureFrame=frame;
                slot.nativeOpaqueCaptureReady = true;
            }
            static const GUID primaryGuid={0x7536783c,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
            ID3D12Resource* primary=nullptr;
            bytes=sizeof(primary);
            if(SUCCEEDED(motion->GetPrivateData(primaryGuid,&bytes,&primary)) && primary)
            {
                D3D12_RESOURCE_DESC pd;primary->GetDesc(&pd);
                if((pd.Format==DXGI_FORMAT_R32_TYPELESS || pd.Format==DXGI_FORMAT_D32_FLOAT) &&
                    pd.Width==opaqueDesc.Width && pd.Height==opaqueDesc.Height &&
                    ensure_staging(slot.nativePrimaryDepthCapture,primary,D3D12_HEAP_TYPE_READBACK) &&
                    SUCCEEDED(g_vk.lifetime->RetainExternalResources(list,primary)))
                {
                    copy_to_staging(list,primary,slot.nativePrimaryDepthCapture,D3D12_RESOURCE_STATE_COMMON);
                    slot.nativePrimaryDepthCaptureReady=true;
                }
                primary->Release();
            }
            const GUID ownedDepthGuid={0x75367842,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
            ID3D12Resource* ownedDepth=nullptr;bytes=sizeof(ownedDepth);
            if(SUCCEEDED(motion->GetPrivateData(ownedDepthGuid,&bytes,&ownedDepth)) && ownedDepth)
            {
                D3D12_RESOURCE_DESC pd;ownedDepth->GetDesc(&pd);
                if((pd.Format==DXGI_FORMAT_R32_TYPELESS || pd.Format==DXGI_FORMAT_D32_FLOAT) &&
                    pd.Width==opaqueDesc.Width && pd.Height==opaqueDesc.Height &&
                    ensure_staging(slot.ownedDepthCapture,ownedDepth,D3D12_HEAP_TYPE_READBACK) &&
                    SUCCEEDED(g_vk.lifetime->RetainExternalResources(list,ownedDepth)))
                {
                    copy_to_staging(list,ownedDepth,slot.ownedDepthCapture,D3D12_RESOURCE_STATE_COMMON);
                    slot.ownedDepthCaptureReady=true;
                }
                ownedDepth->Release();
            }
            slot.opaqueSceneCaptureReady=record_motion_constants(list,slot.opaqueSceneCapture,motion,frame,
                0x7536783e,UINT64_C(0xc539803a4bac94ac),71*16);
            slot.opaqueMatrixCaptureReady=record_motion_constants(list,slot.opaqueMatrixCapture,motion,frame,
                0x7536783d,UINT64_C(0xc539803a4bac94ac),4*16);
            {
                const GUID originalGuid={0x75367846,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
                ID3D12Resource* original=nullptr;UINT originalBytes=sizeof(original);
                if(SUCCEEDED(motion->GetPrivateData(originalGuid,&originalBytes,&original)) && original)
                {
                    D3D12_RESOURCE_DESC od;original->GetDesc(&od);
                    if(od.Format==DXGI_FORMAT_R16G16_FLOAT && ensure_staging(slot.originalOpaqueCopyCapture,original,D3D12_HEAP_TYPE_READBACK) &&
                        SUCCEEDED(g_vk.lifetime->RetainExternalResources(list,original)))
                    {
                        copy_to_staging(list,original,slot.originalOpaqueCopyCapture,D3D12_RESOURCE_STATE_COMMON);
                        slot.originalOpaqueCopyCaptureReady=true;
                    }
                    original->Release();
                }
            }
            slot.ownedValidityCaptureReady=record_owned_motion_constants(list,slot.ownedValidityCapture,motion,0x75367843,8);
            if(slot.ownedValidityCaptureReady) { slot.nativeValidityOwner=motion;motion->AddRef(); }
            slot.ownedSceneCaptureReady=record_owned_motion_constants(list,slot.ownedSceneCapture,motion,0x75367840,1136);
            slot.ownedMatrixCaptureReady=record_owned_motion_constants(list,slot.ownedMatrixCapture,motion,0x75367841,64);
            logf("native motion capture frame %u: owned scene/matrix %d/%d",frame,
                slot.ownedSceneCaptureReady,slot.ownedMatrixCaptureReady);
            slot.originalPrimaryDepthCaptureReady=record_original_motion_source(list,motion,slot.originalPrimaryDepthCapture,
                true,0x7536783f,UINT64_C(0xc539803a4bac94ac));
            logf("native motion capture frame %u: primary depth/scene/matrix/original %d/%d/%d/%d",frame,
                slot.nativePrimaryDepthCaptureReady,slot.opaqueSceneCaptureReady,slot.opaqueMatrixCaptureReady,
                slot.originalPrimaryDepthCaptureReady);
            logf("native motion capture frame %u: opaque per-object source %p %llux%u",frame,opaque,
                static_cast<unsigned long long>(opaqueDesc.Width),opaqueDesc.Height);
        }
        opaque->Release();
    }
    static const GUID guid = {0x75367836,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
    ID3D12Resource* native = nullptr;
    bytes = sizeof(native);
    if (FAILED(motion->GetPrivateData(guid, &bytes, &native)) || native == nullptr)
    {
        logf("native motion capture frame %u: no geometry sidecar linked", frame);
        return;
    }
    D3D12_RESOURCE_DESC desc;
    native->GetDesc(&desc);
    const bool ready = desc.Format == DXGI_FORMAT_R16G16_FLOAT &&
        ensure_staging(slot.nativeMotionCapture, native, D3D12_HEAP_TYPE_READBACK) &&
        g_vk.lifetime != nullptr && SUCCEEDED(g_vk.lifetime->RetainExternalResources(list, native));
    if (ready)
    {
        copy_to_staging(list, native, slot.nativeMotionCapture, D3D12_RESOURCE_STATE_COMMON);
        char path[MAX_PATH];
        const int size = std::snprintf(path, MAX_PATH, "%s\\frame-%06u-%ux%u.geometry.rg16f", directory,
            frame, static_cast<UINT>(desc.Width), desc.Height);
        if (size > 0 && size < MAX_PATH)
        {
            slot.nativeMotionCapturePath = path;
            slot.nativeMotionCaptureFrame = frame;
            slot.nativeGeometryCaptureReady = true;
            logf("native motion capture frame %u: geometry sidecar %p %llux%u (original NGX input retained)",
                frame, native, static_cast<unsigned long long>(desc.Width), desc.Height);
        }
    }
    if (slot.nativeMotionCaptureFrame == frame)
    {
        static const GUID depthGuid = {0x75367837,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
        ID3D12Resource* nativeDepth = nullptr;
        bytes = sizeof(nativeDepth);
        if (SUCCEEDED(motion->GetPrivateData(depthGuid, &bytes, &nativeDepth)) && nativeDepth != nullptr)
        {
            D3D12_RESOURCE_DESC depthDesc;
            nativeDepth->GetDesc(&depthDesc);
            if ((depthDesc.Format == DXGI_FORMAT_R32_TYPELESS || depthDesc.Format == DXGI_FORMAT_D32_FLOAT) &&
                depthDesc.Width == desc.Width && depthDesc.Height == desc.Height &&
                ensure_staging(slot.nativeDepthCapture, nativeDepth, D3D12_HEAP_TYPE_READBACK) &&
                SUCCEEDED(g_vk.lifetime->RetainExternalResources(list, nativeDepth)))
            {
                copy_to_staging(list, nativeDepth, slot.nativeDepthCapture, D3D12_RESOURCE_STATE_COMMON);
                slot.glassValidityCaptureReady=record_owned_motion_constants(list,slot.glassValidityCapture,nativeDepth,0x75367845,4);
                slot.nativeDepthCaptureReady = true;
            }
            nativeDepth->Release();
        }
        slot.nativeSceneCaptureReady = record_motion_constants(list,slot.nativeSceneCapture,motion,frame,0x75367838,UINT64_C(0xc5d8688e46fd5c66),71*16);
        slot.originalGeometryCaptureReady = record_original_motion_source(list,motion,slot.originalGeometryCapture,false);
        slot.originalHardwareDepthCaptureReady = record_original_motion_source(list,motion,slot.originalHardwareDepthCapture,true);
        logf("native motion capture frame %u: original geometry/depth readbacks %d/%d",frame,
            slot.originalGeometryCaptureReady,slot.originalHardwareDepthCaptureReady);
    }
    native->Release(); // the allocator retains it through completion
}

static bool vram_interop_requested()
{
    static const bool requested = env_uint("D4R_SHIM_VRAM_INTEROP", 0) != 0;
    return requested;
}

// Game thread, once.

void init_vram_interop()
{
    if (FAILED(g.device->QueryInterface(__uuidof(ID3D12DXVKInteropDevice1), reinterpret_cast<void**>(&g_vk.interop))))
    {
        // On native Windows (D3D12Core, no vkd3d-proton) this interface never exists: VRAM interop there is the
        // native route (native_vram_available), so only a Wine/Proton device that lacks it is worth a line
        static const bool underWine =
            GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
        if (underWine)
            logf("VRAM interop: ID3D12DXVKInteropDevice1 unavailable (not vkd3d-proton?)");
        return;
    }
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->GetVulkanHandles(&instance, &physical, &g_vk.device)))
    {
        logf("VRAM interop: GetVulkanHandles failed");
        return;
    }
    UINT count = 0;
    g_vk.interop->GetDeviceExtensions(&count, nullptr);
    std::vector<const char*> extensions(count);
    g_vk.interop->GetDeviceExtensions(&count, extensions.data());
    bool exportable = false;
    for (const char* name : extensions)
        exportable |= std::strcmp(name, "VK_KHR_external_memory_win32") == 0;
    // d4r's vkd3d-proton enables VK_EXT_pipeline_robustness (patches/vkd3d-proton/0005): the engine's pipelines then
    // opt out of robustBufferAccess2, which keeps RADV from scalar storage-buffer loads (K's network: about 1.6x).
    bool pipelineRobustness = false;
    for (const char* name : extensions)
        pipelineRobustness |= std::strcmp(name, "VK_EXT_pipeline_robustness") == 0;
    if (pipelineRobustness && env_uint("D4R_ENGINE_BUFFER_ROBUSTNESS", 0) == 0)
        d4r::setBufferRobustnessDisabled(true);
    // d4r's vkd3d-proton also enables VK_KHR_pipeline_binary (patches/vkd3d-proton/0013), with which vkhip injects its
    // native layers on AMD's own driver (native Windows); RADV does not need it
    bool binaries = false, maintenance5 = false;
    for (const char* name : extensions)
    {
        binaries |= std::strcmp(name, "VK_KHR_pipeline_binary") == 0;
        maintenance5 |= std::strcmp(name, "VK_KHR_maintenance5") == 0;
    }
    logf("engine backend: buffer robustness %s", pipelineRobustness && env_uint("D4R_ENGINE_BUFFER_ROBUSTNESS", 0) == 0 ?
         "off for the engine's pipelines (VK_EXT_pipeline_robustness)" : "on (the device's default)");
    HMODULE vulkan = GetModuleHandleA("vulkan-1.dll");
    if (vulkan == nullptr)
        vulkan = LoadLibraryA("vulkan-1.dll");
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    if (vulkan != nullptr)
        load_export(vulkan, "vkGetInstanceProcAddr", instanceProc);
    load_export(g.cuda, "d4rImportVulkanMemory", g_vk.import);
    load_export(g.cuda, "d4rReleaseVulkanMemory", g_vk.release);
    if (!exportable || instanceProc == nullptr || g_vk.import == nullptr || g_vk.release == nullptr)
    {
        logf("VRAM interop: unavailable (external_memory_win32 %d, vulkan-1 %p, bridge exports %d)", exportable,
             reinterpret_cast<void*>(instanceProc), g_vk.import != nullptr && g_vk.release != nullptr);
        return;
    }
    auto deviceProc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(instanceProc(instance, "vkGetDeviceProcAddr"));
    auto memoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        instanceProc(instance, "vkGetPhysicalDeviceMemoryProperties"));
    if (deviceProc == nullptr || memoryProperties == nullptr)
        return;
    g_vk.deviceProc = deviceProc;
    g_vk.instance = instance;
    g_vk.instanceProc = instanceProc;
    g_vk.pipelineBinary = binaries && maintenance5 && deviceProc(g_vk.device, "vkCreatePipelineBinariesKHR") != nullptr;
    memoryProperties(physical, &g_vk.memory);
    g_vk.physical = physical;
    g_vk.formatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
        instanceProc(instance, "vkGetPhysicalDeviceFormatProperties"));
    bool ok = g_vk.formatProperties != nullptr;
    auto load = [&](auto& function, const char* name) {
        function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(deviceProc(g_vk.device, name));
        ok &= function != nullptr;
    };
    load(g_vk.createBuffer, "vkCreateBuffer");
    load(g_vk.destroyBuffer, "vkDestroyBuffer");
    load(g_vk.bufferRequirements, "vkGetBufferMemoryRequirements");
    load(g_vk.bindBuffer, "vkBindBufferMemory");
    load(g_vk.createView, "vkCreateImageView");
    load(g_vk.destroyView, "vkDestroyImageView");
    load(g_vk.createImage, "vkCreateImage");
    load(g_vk.destroyImage, "vkDestroyImage");
    load(g_vk.imageRequirements, "vkGetImageMemoryRequirements");
    load(g_vk.bindImage, "vkBindImageMemory");
    load(g_vk.allocate, "vkAllocateMemory");
    load(g_vk.free, "vkFreeMemory");
    load(g_vk.map, "vkMapMemory");
    load(g_vk.unmap, "vkUnmapMemory");
    load(g_vk.barrier, "vkCmdPipelineBarrier");
    load(g_vk.copyImageToBuffer, "vkCmdCopyImageToBuffer");
    load(g_vk.copyBuffer, "vkCmdCopyBuffer");
    load(g_vk.copyBufferToImage, "vkCmdCopyBufferToImage");
    load(g_vk.blit, "vkCmdBlitImage");
    g_vk.fill = reinterpret_cast<PFN_vkCmdFillBuffer>(deviceProc(g_vk.device, "vkCmdFillBuffer"));
    load(g_vk.createSemaphore, "vkCreateSemaphore");
    load(g_vk.destroySemaphore, "vkDestroySemaphore");
    load(g_vk.signalSemaphore, "vkSignalSemaphore");
    if (FAILED(g.device->QueryInterface(__uuidof(ID3D12DXVKInteropDeviceD4R), reinterpret_cast<void**>(&g_vk.split))))
        g_vk.split = nullptr;
    if (FAILED(g.device->QueryInterface(__uuidof(ID3D12DXVKInteropDeviceD4R2), reinterpret_cast<void**>(&g_vk.lifetime))))
        g_vk.lifetime = nullptr;
    if (env_uint("D4R_SHIM_INPLACE", 0) == 0 ||
        FAILED(g.device->QueryInterface(__uuidof(ID3D12DXVKInteropDeviceD4R3), reinterpret_cast<void**>(&g_vk.linear))))
        g_vk.linear = nullptr;
    g_vk.ready = ok;
    if (ok && (env_uint("D4R_SHIM_FUSE_RGBA8_INPUT", 1) != 0 ||
               env_uint("D4R_SHIM_FUSE_R11_OUTPUT", 0) != 0 || env_uint("D4R_SHIM_FUSE_R11_INPUT", 0) != 0))
    {
        g_fusedColorReady = g_fusedColorStage.init(g_vk.device, deviceProc);
        if (!g_fusedColorReady) g_fusedColorStage.destroy();
    }
    if (ok && env_uint("D4R_SHIM_GPU_COMPLETION", 1) != 0)
    {
        g_waitReady = g_waitStage.init(g_vk.device, deviceProc) && g_waitStage.wait_ready();
        if (!g_waitReady) g_waitStage.destroy();
    }
    logf("VRAM interop: %s (VkDevice %p), split frames %s", ok ? "ready" : "missing Vulkan entry points",
         static_cast<void*>(g_vk.device), g_vk.split != nullptr ? "available" : "unavailable (stock vkd3d-proton)");
}

bool vram_interop_available()
{
    if (!vram_interop_requested())
        return false;
    std::call_once(g_vk.once, init_vram_interop);
    return g_vk.ready;
}

static void init_native_vram()
{
#ifdef D4R_NO_SPLIT_SHADERS
    logf("native VRAM: unavailable (the shim was built without dxc)");
#else
    if (g.device == nullptr || g.cuda == nullptr)
        return;
    load_export(g.cuda, "d4rImportD3D12Memory", g_dxv.import);
    if (g_vk.release == nullptr)
        load_export(g.cuda, "d4rReleaseVulkanMemory", g_vk.release);
    if (g_dxv.import == nullptr || g_vk.release == nullptr)
    {
        logf("native VRAM: unavailable (the nvcuda bridge lacks d4rImportD3D12Memory)");
        return;
    }
    auto pipeline = [](const unsigned char* code, size_t size, ID3D12RootSignature** root, ID3D12PipelineState** state) {
        if (FAILED(g.device->CreateRootSignature(0, code, size, IID_PPV_ARGS(root))))
            return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = *root;
        desc.CS = {code, size};
        return SUCCEEDED(g.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(state)));
    };
    D3D12_DESCRIPTOR_HEAP_DESC heap = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kNativeDescriptors,
                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    if (!pipeline(g_d4r_to_canonical, sizeof(g_d4r_to_canonical), &g_dxv.inputRoot, &g_dxv.inputPipeline) ||
        !pipeline(g_d4r_from_canonical, sizeof(g_d4r_from_canonical), &g_dxv.outputRoot, &g_dxv.outputPipeline) ||
        FAILED(g.device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&g_dxv.heap))))
    {
        logf("native VRAM: unavailable (its shader pipelines or descriptor heap could not be created)");
        return;
    }
    g_dxv.increment = g.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const char* failure = "unknown";
    g_dxv.split = d4r_split::install(g.device, &failure);
    logf("native VRAM: on (D3D12 shared buffers imported into HIP); split frames %s",
         g_dxv.split ? "available" : failure);
    g_dxv.ready = true;
#endif
}

// [Interop] VramInterop on a device that is not vkd3d-proton's.

bool native_vram_available()
{
    if (!vram_interop_requested() || g_vk.interop != nullptr)
        return false;
    std::call_once(g_dxv.once, init_native_vram);
    return g_dxv.ready && !g_dxv.importFailed.load();
}

// The view format a shader reads `format` through: typeless and depth formats as their colour-readable equivalent,
// sRGB as UNORM (the host route treats those bytes as UNORM too). UNKNOWN: not supported here.

static DXGI_FORMAT native_view_format(DXGI_FORMAT format, bool uav)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT: return uav ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return uav ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return uav ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_D16_UNORM: return uav ? DXGI_FORMAT_UNKNOWN : DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_UNORM:
        return format;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

// Game thread. The CUDA import runs on the worker, which owns the context.

bool create_native_vram_buffer(VramBuffer& target, size_t bytes)
{
    D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    VramBuffer buffer;
    HANDLE handle = nullptr;
    HRESULT hr = g.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&buffer.d3d));
    if (SUCCEEDED(hr))
        hr = g.device->CreateSharedHandle(buffer.d3d, nullptr, GENERIC_ALL, nullptr, &handle);
    int imported = -1;
    if (SUCCEEDED(hr))
    {
        const uint64_t allocation = g.device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
        imported = g.worker.call([&] { return g_dxv.import(handle, allocation, &buffer.device, &buffer.external); });
        CloseHandle(handle); // HIP holds its own reference
    }
    if (FAILED(hr) || imported != 0)
    {
        logf("native VRAM: shared buffer of %zu bytes failed (HRESULT 0x%08lx, import %d)", bytes,
             static_cast<unsigned long>(hr), imported);
        if (SUCCEEDED(hr) && !g_dxv.importFailed.exchange(true))
            logf("native VRAM: off for this session (HIP cannot import D3D12 shared buffers here); features stage "
                 "through host memory from their next frame");
        if (buffer.d3d != nullptr)
            buffer.d3d->Release();
        return false;
    }
    buffer.bytes = bytes;
    target = buffer;
    return true;
}

static D3D12_GPU_DESCRIPTOR_HANDLE native_descriptor(D3D12_CPU_DESCRIPTOR_HANDLE& cpu)
{
    const uint32_t index = g_dxv.next.fetch_add(1) % kNativeDescriptors;
    cpu = g_dxv.heap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += static_cast<SIZE_T>(index) * g_dxv.increment;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_dxv.heap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += static_cast<UINT64>(index) * g_dxv.increment;
    return gpu;
}

// record_vram_inputs' native counterpart: each input into its plane's canonical layout in slot.vram. NGX's evaluate
// leaves the list's state undefined for the caller, so binding d4r's heap and pipeline here is allowed.

void record_native_inputs(ID3D12GraphicsCommandList* list, InputSlot& slot, ID3D12Resource* const* inputs,
                                 const Plane* planes, const D3D12_RESOURCE_STATES* states, int count)
{
    list->SetDescriptorHeaps(1, &g_dxv.heap);
    list->SetComputeRootSignature(g_dxv.inputRoot);
    list->SetPipelineState(g_dxv.inputPipeline);
    for (int index = 0; index < count; ++index)
    {
        const D3D12_RESOURCE_DESC desc = inputs[index]->GetDesc();
        const bool readable = (states[index] & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) != 0;
        if (!readable)
            transition(list, inputs[index], states[index], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        D3D12_SHADER_RESOURCE_VIEW_DESC view = {};
        view.Format = native_view_format(desc.Format, false);
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Texture2D.MipLevels = 1;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu;
        const D3D12_GPU_DESCRIPTOR_HANDLE gpu = native_descriptor(cpu);
        g.device->CreateShaderResourceView(inputs[index], &view, cpu);
        const HostPlane& geometry = slot.host[index];
        const uint32_t layout = planes[index] == Plane::Color ? 0 : planes[index] == Plane::Motion ? 1 : 2;
        const uint32_t constants[4] = {geometry.width, geometry.height, static_cast<uint32_t>(geometry.rowBytes), layout};
        list->SetComputeRoot32BitConstants(0, 4, constants, 0);
        list->SetComputeRootDescriptorTable(1, gpu);
        list->SetComputeRootUnorderedAccessView(2, slot.vram[index].d3d->GetGPUVirtualAddress());
        list->Dispatch((geometry.width + 7) / 8, (geometry.height + 7) / 8, 1);
        if (!readable)
            transition(list, inputs[index], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, states[index]);
    }
    D3D12_RESOURCE_BARRIER barrier = {D3D12_RESOURCE_BARRIER_TYPE_UAV};
    list->ResourceBarrier(1, &barrier); // the buffers are complete before the frame marker that follows
}

// record_feature_vram_output's native counterpart: DLSS's result in outputs[slotIndex] into the game's texture.

void record_native_output(Feature& feature, unsigned slotIndex, ID3D12GraphicsCommandList* list,
                                 ID3D12Resource* output, D3D12_RESOURCE_STATES outputState)
{
    const D3D12_RESOURCE_DESC desc = output->GetDesc();
    const bool writable = (outputState & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) != 0;
    if (!writable)
        transition(list, output, outputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetDescriptorHeaps(1, &g_dxv.heap);
    list->SetComputeRootSignature(g_dxv.outputRoot);
    list->SetPipelineState(g_dxv.outputPipeline);
    D3D12_UNORDERED_ACCESS_VIEW_DESC view = {};
    view.Format = native_view_format(desc.Format, true);
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu;
    const D3D12_GPU_DESCRIPTOR_HANDLE gpu = native_descriptor(cpu);
    g.device->CreateUnorderedAccessView(output, nullptr, &view, cpu);
    const uint32_t width = std::min(feature.output.width, static_cast<uint32_t>(desc.Width));
    const uint32_t height = std::min(feature.output.height, static_cast<uint32_t>(desc.Height));
    const uint32_t constants[4] = {width, height, feature.output.width * 8, 0};
    list->SetComputeRoot32BitConstants(0, 4, constants, 0);
    list->SetComputeRootShaderResourceView(1, feature.outputs[slotIndex].vram.d3d->GetGPUVirtualAddress());
    list->SetComputeRootDescriptorTable(2, gpu);
    list->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    D3D12_RESOURCE_BARRIER barrier = {D3D12_RESOURCE_BARRIER_TYPE_UAV};
    barrier.UAV.pResource = output;
    list->ResourceBarrier(1, &barrier);
    if (!writable)
        transition(list, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outputState);
}

bool create_native_split(Feature& feature)
{
    return SUCCEEDED(g.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&feature.splitFence)));
}

// SplitCommandListForExternalWait's native counterpart: the rest of `list` goes into this output slot's second
// list, which runs once splitFence reaches `frame`. `beforeSecond` runs between the two parts (d3dhip: the network);
// `back`, when not null, receives that second list, which is open and must be recorded into before the game closes
// `list`.

bool native_split_frame(Feature& feature,  ID3D12GraphicsCommandList* list,  uint32_t frame,  void (*beforeSecond)(void*, ID3D12CommandQueue*),  void* beforeOwner,  ID3D12GraphicsCommandList** back)
{
    Feature::SplitPart& part = feature.splitParts[frame % kOutputSlots];
    if (part.list == nullptr)
    {
        const D3D12_COMMAND_LIST_TYPE type = list->GetType();
        if (FAILED(g.device->CreateCommandAllocator(type, IID_PPV_ARGS(&part.allocator))) ||
            FAILED(g.device->CreateCommandList(0, type, part.allocator, nullptr, IID_PPV_ARGS(&part.list))) ||
            FAILED(g.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&part.retire))))
        {
            logf("frame %u: the second part of a split list could not be created", frame);
            return false;
        }
        part.list->Close();
    }
    // this slot's previous second part, kOutputSlots frames ago, must have run before its allocator is reused
    if (part.retire->GetCompletedValue() < part.retireValue)
    {
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool done = event != nullptr && SUCCEEDED(part.retire->SetEventOnCompletion(part.retireValue, event)) &&
                          WaitForSingleObject(event, 2000) == WAIT_OBJECT_0;
        if (event != nullptr)
            CloseHandle(event);
        if (!done)
        {
            logf("frame %u: the split list of frame %u has not run yet; not splitting", frame, frame - kOutputSlots);
            return false;
        }
    }
    if (FAILED(part.allocator->Reset()) || FAILED(part.list->Reset(part.allocator, nullptr)))
        return false;
    ++part.retireValue;
    auto done = [](void* owner, bool executed) {
        // a discarded second part never runs: its allocator is free again at once
        auto* discarded = static_cast<Feature::SplitPart*>(owner);
        if (!executed)
            discarded->retire->Signal(discarded->retireValue);
    };
    if (!d4r_split::begin(list, part.list, feature.splitFence, frame, part.retire, part.retireValue, done, &part,
                          beforeSecond, beforeOwner))
    {
        part.list->Close();
        part.retire->Signal(part.retireValue);
        return false;
    }
    if (back != nullptr) *back = part.list;
    return true;
}

void release_native_split(Feature& feature)
{
    for (Feature::SplitPart& part : feature.splitParts)
    {
        if (part.list != nullptr) part.list->Release();
        if (part.allocator != nullptr) part.allocator->Release();
        if (part.retire != nullptr) part.retire->Release();
        part = Feature::SplitPart{};
    }
    if (feature.splitFence != nullptr)
        feature.splitFence->Release();
    feature.splitFence = nullptr;
}

// Device-local memory; prefers memory the CPU cannot map (no BAR window).

int device_local_type(uint32_t typeBits)
{
    for (int pass = 0; pass < 2; ++pass)
        for (uint32_t index = 0; index < g_vk.memory.memoryTypeCount; ++index)
        {
            const VkMemoryPropertyFlags flags = g_vk.memory.memoryTypes[index].propertyFlags;
            if ((typeBits & (1u << index)) && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (pass == 1 || !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)))
                return static_cast<int>(index);
        }
    return -1;
}

// Buffers of released features, reused by later ones. ROCm up to at least 7.2.4 never frees memory mapped with
// hipExternalMemoryGetMappedBuffer (the mapping's buffer view is retained twice but released once by hipFree), so
// destroying a buffer would leak its memory: each DLSS quality change (a new feature) would add ~100 MB of VRAM.

static std::mutex g_vramPoolMutex;

static std::vector<VramBuffer> g_vramPool;

// The smallest pooled buffer of `kind` that holds `bytes`.

bool take_pooled_vram_buffer(VramBuffer& target,  size_t bytes,  uint8_t kind)
{
    std::lock_guard<std::mutex> lock(g_vramPoolMutex);
    auto best = g_vramPool.end();
    for (auto it = g_vramPool.begin(); it != g_vramPool.end(); ++it)
        if (it->kind == kind && it->bytes >= bytes && (best == g_vramPool.end() || it->bytes < best->bytes))
            best = it;
    if (best == g_vramPool.end())
        return false;
    target = *best;
    g_vramPool.erase(best);
    return true;
}

// Game thread. The CUDA import runs on the worker, which owns the context.

bool create_vram_buffer(VramBuffer& target,  size_t bytes,  bool importNow)
{
    if (take_pooled_vram_buffer(target, bytes))
        return true;
    VramBuffer buffer;
    VkExternalMemoryBufferCreateInfo external = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (g_fusedColorReady) info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = g_vk.createBuffer(g_vk.device, &info, nullptr, &buffer.buffer);
    VkMemoryRequirements requirements = {};
    if (result == VK_SUCCESS)
        g_vk.bufferRequirements(g_vk.device, buffer.buffer, &requirements);
    const int type = device_local_type(requirements.memoryTypeBits);
    VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = buffer.buffer;
    VkExportMemoryAllocateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &dedicated};
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exportInfo};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = static_cast<uint32_t>(type);
    if (result == VK_SUCCESS && type < 0)
        result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (result == VK_SUCCESS)
        result = g_vk.allocate(g_vk.device, &allocation, nullptr, &buffer.memory);
    if (result == VK_SUCCESS)
        result = g_vk.bindBuffer(g_vk.device, buffer.buffer, buffer.memory, 0);
    int imported = 0;
    if (result == VK_SUCCESS && importNow)
        imported = g.worker.call([&] {
            return g_vk.import(g_vk.device, reinterpret_cast<uint64_t>(buffer.memory), requirements.size,
                               &buffer.device, &buffer.external);
        });
    if (result != VK_SUCCESS || imported != 0)
    {
        logf("VRAM interop: exportable buffer of %zu bytes failed (VkResult %d, import %d)", bytes, result, imported);
        if (buffer.memory != VK_NULL_HANDLE)
            g_vk.free(g_vk.device, buffer.memory, nullptr);
        if (buffer.buffer != VK_NULL_HANDLE)
            g_vk.destroyBuffer(g_vk.device, buffer.buffer, nullptr);
        return false;
    }
    buffer.bytes = bytes;
    target = buffer;
    return true;
}

// Only once no command list or CUDA work uses the buffer.

static void destroy_vram_buffer(VramBuffer& buffer)
{
    if (buffer.d3d != nullptr)
    {
        if (buffer.external != nullptr)
            g.worker.call([&buffer] {
                const int freed = g.cu.memFree(buffer.device);
                const int released = g_vk.release(buffer.external);
                if (freed != 0 || released != 0)
                    logf("native VRAM: freeing %zu-byte buffer: cuMemFree %d, release %d", buffer.bytes, freed, released);
                return released;
            });
        buffer.d3d->Release();
        buffer = VramBuffer{};
        return;
    }
    if (buffer.external != nullptr)
        g.worker.call([&buffer] {
            const int freed = g.cu.memFree(buffer.device);
            const int released = g_vk.release(buffer.external);
            if (freed != 0 || released != 0)
                logf("VRAM interop: freeing %zu-byte buffer: cuMemFree %d, release %d", buffer.bytes, freed, released);
            return released;
        });
    if (buffer.mapped != nullptr)
        g_vk.unmap(g_vk.device, buffer.memory);
    if (buffer.memory != VK_NULL_HANDLE)
        g_vk.free(g_vk.device, buffer.memory, nullptr);
    if (buffer.buffer != VK_NULL_HANDLE)
        g_vk.destroyBuffer(g_vk.device, buffer.buffer, nullptr);
    buffer = VramBuffer{};
}

// Only once no command list or CUDA work uses the buffer: keeps it for a later feature (see g_vramPool).

void recycle_vram_buffer(VramBuffer& buffer)
{
    if (buffer.d3d != nullptr)
    {
        destroy_vram_buffer(buffer); // the pool holds Vulkan buffers only
        return;
    }
    if (buffer.buffer == VK_NULL_HANDLE)
        return;
    {
        std::lock_guard<std::mutex> lock(g_vramPoolMutex);
        if (g_vramPool.size() < kVramPoolLimit)
        {
            g_vramPool.push_back(buffer);
            buffer = VramBuffer{};
            return;
        }
    }
    destroy_vram_buffer(buffer);
}

void free_vram_pool()
{
    std::vector<VramBuffer> pool;
    {
        std::lock_guard<std::mutex> lock(g_vramPoolMutex);
        pool.swap(g_vramPool);
    }
    for (VramBuffer& buffer : pool)
        destroy_vram_buffer(buffer);
}

void destroy_vram_image(VramImage& image)
{
    if (image.memory != VK_NULL_HANDLE)
        g_vk.free(g_vk.device, image.memory, nullptr);
    if (image.image != VK_NULL_HANDLE)
        g_vk.destroyImage(g_vk.device, image.image, nullptr);
    image = VramImage{};
}

bool create_vram_image(VramImage& image, UINT width, UINT height, VkFormat format)
{
    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkResult result = g_vk.createImage(g_vk.device, &info, nullptr, &image.image);
    VkMemoryRequirements requirements = {};
    if (result == VK_SUCCESS)
        g_vk.imageRequirements(g_vk.device, image.image, &requirements);
    const int type = device_local_type(requirements.memoryTypeBits);
    VkMemoryAllocateInfo allocation = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = static_cast<uint32_t>(type);
    if (result == VK_SUCCESS && type < 0)
        result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (result == VK_SUCCESS)
        result = g_vk.allocate(g_vk.device, &allocation, nullptr, &image.memory);
    if (result == VK_SUCCESS)
        result = g_vk.bindImage(g_vk.device, image.image, image.memory, 0);
    if (result != VK_SUCCESS)
    {
        logf("VRAM interop: %ux%u conversion image failed: %d", width, height, result);
        destroy_vram_image(image);
        return false;
    }
    image.width = width;
    image.height = height;
    image.format = format;
    return true;
}

size_t canonical_texel_bytes(Plane plane)
{
    return (plane_format(plane) == CUDA_FORMAT_HALF ? 2 : 4) * plane_channels(plane);
}

static size_t vk_texel_bytes(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_UNORM:
        return 8;
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: // the depth aspect alone
        return 4;
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_UINT:
        return 2;
    default:
        return 0;
    }
}

static std::mutex g_inPlaceMutex;

static std::vector<InPlaceImport> g_inPlaceImports;

// Game thread.

bool describe_in_place(ID3D12Resource* resource, InPlaceTexture& texture)
{
    UINT64 memory = 0, size = 0, offset = 0, pitch = 0;
    if (g_vk.linear == nullptr || FAILED(g_vk.linear->GetVulkanLinearImageInfo(resource, &memory, &size, &offset, &pitch)))
        return false;
    CudaDevicePtr base = 0;
    {
        std::lock_guard<std::mutex> lock(g_inPlaceMutex);
        for (const auto& entry : g_inPlaceImports)
            if (entry.memory == memory)
                base = entry.base;
    }
    if (base == 0)
    {
        void* external = nullptr;
        const int imported = g.worker.call([&] {
            return g_vk.import(g_vk.device, memory, size, &base, &external);
        });
        if (imported != 0 || base == 0)
        {
            logf("in-place: importing %llu bytes of texture memory failed (%d)", static_cast<unsigned long long>(size), imported);
            return false;
        }
        logf("in-place: imported %llu bytes of game texture memory (offset %llu, row pitch %llu)",
             static_cast<unsigned long long>(size), static_cast<unsigned long long>(offset),
             static_cast<unsigned long long>(pitch));
        std::lock_guard<std::mutex> lock(g_inPlaceMutex);
        resource->AddRef();
        g_inPlaceImports.push_back({memory, base, resource});
    }
    texture.texels = base + offset;
    texture.rowPitch = static_cast<size_t>(pitch);
    return true;
}

// describe_vram_copy's native counterpart: a single-sample 2D texture a shader can read (or, for the output, write).

bool describe_native_vram(ID3D12Resource* resource,  VramCopy& copy,  bool output)
{
    D3D12_RESOURCE_DESC desc;
    resource->GetDesc(&desc);
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
        native_view_format(desc.Format, output) == DXGI_FORMAT_UNKNOWN ||
        (output ? (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0
                : (desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) != 0))
        return false;
    copy.resource = resource;
    copy.convert = false;
    copy.width = static_cast<UINT>(desc.Width);
    copy.height = desc.Height;
    return true;
}

bool vram_blit_supported(VkFormat source, VkFormat destination)
{
    VkFormatProperties sourceProperties = {}, destinationProperties = {};
    g_vk.formatProperties(g_vk.physical, source, &sourceProperties);
    g_vk.formatProperties(g_vk.physical, destination, &destinationProperties);
    return (sourceProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0 &&
           (destinationProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
}

// sRGB is deliberately excluded: the host staging path treats its bytes as
// UNORM and a Vulkan blit would apply an sRGB transfer function instead.

static bool vram_color_blit_supported(VkFormat format, bool output)
{
    switch (format)
    {
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        break;
    default:
        return false;
    }
    return output ? vram_blit_supported(VK_FORMAT_R16G16B16A16_SFLOAT, format)
                  : vram_blit_supported(format, VK_FORMAT_R16G16B16A16_SFLOAT);
}

// The host path takes only the first (R) component of an exposure texel.

static bool vram_exposure_blit_supported(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R8G8B8A8_UNORM:
        return vram_blit_supported(format, VK_FORMAT_R32_SFLOAT);
    default:
        return false;
    }
}

// Depth in a 16-bit format is widened to R32F by a blit (UNORM normalizes, SFLOAT converts).

static bool vram_depth_blit_supported(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UNORM:
        return vram_blit_supported(format, VK_FORMAT_R32_SFLOAT);
    default:
        return false;
    }
}

// Motion vectors in a wider float format keep their first two components (as the host path does).

static bool vram_motion_blit_supported(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R8G8B8A8_UNORM:
        return vram_blit_supported(format, VK_FORMAT_R16G16_SFLOAT);
    default:
        return false;
    }
}

// Whether `resource` can be copied raw (or, for colour/motion/exposure, blitted)
// to/from the canonical layout of `plane`.

bool describe_vram_copy(ID3D12Resource* resource,  Plane plane,  VramCopy& copy,  bool output)
{
    UINT64 handle = 0, offset = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    if (FAILED(g_vk.interop->GetVulkanResourceInfo1(resource, &handle, &offset, &format)) || handle == 0)
        return false;
    D3D12_RESOURCE_DESC desc;
    resource->GetDesc(&desc);
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1)
        return false;
    copy.resource = resource;
    copy.image = reinterpret_cast<VkImage>(handle);
    copy.width = static_cast<UINT>(desc.Width);
    copy.height = desc.Height;
    copy.aspect = (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT)
                      ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    if (plane == Plane::Color)
    {
        copy.convert = vram_color_blit_supported(format, output);
        return copy.convert || format == VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    if (plane == Plane::Exposure)
        copy.convert = vram_exposure_blit_supported(format);
    else if (plane == Plane::Motion)
        copy.convert = vram_motion_blit_supported(format);
    else if (plane == Plane::Depth)
        copy.convert = vram_depth_blit_supported(format);
    return copy.convert || vk_texel_bytes(format) == canonical_texel_bytes(plane);
}

VkBufferImageCopy full_region(const VramCopy& copy)
{
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = copy.aspect;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {copy.width, copy.height, 1};
    return region;
}

bool ensure_fused_color(Feature& feature,  int slotIndex,  ID3D12Resource* source,  const VramCopy& copy,  uint32_t frame)
{
    const auto desc=source->GetDesc();
    const bool rgba8=desc.Format==DXGI_FORMAT_R8G8B8A8_UNORM && env_uint("D4R_SHIM_FUSE_RGBA8_INPUT", 1)!=0;
    const unsigned requested=env_uint("D4R_SHIM_FUSE_R11_INPUT",0);
    const bool r11=desc.Format==DXGI_FORMAT_R11G11B10_FLOAT &&
        (requested==1 || (requested==2 && (frame/300)%4>=2));
    if (!g_fusedColorReady || !feature.fusedColorAllowed || (!rgba8 && !r11) ||
        (desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) != 0) return false;
    // Descriptor/image lifetimes extend through allocator retirement, including discarded lists.
    // Bound that retention if a game creates a new transient input resource every frame; fallback
    // to the original blit/copy path without invalidating descriptors already recorded on the GPU.
    if (feature.retiredFusedColor.size() >= 64)
    {
        feature.fusedColorAllowed = false;
        logf("fused color input descriptor retention reached its bound; using blit/copy for this feature");
        return false;
    }
    auto& slot = feature.fusedColor[slotIndex];
    auto& target = feature.inputs[slotIndex].vram[0];
    if (slot.image == copy.image && slot.buffer == target.buffer && slot.owner == source) return true;
    if (slot.owner != nullptr) feature.retiredFusedColor.push_back(slot);
    slot = {};
    VkImageViewCreateInfo view = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = copy.image; view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = rgba8 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (g_vk.createView(g_vk.device, &view, nullptr, &slot.view) != VK_SUCCESS) return false;
    if (!g_fusedColorStage.set(slot.set, slot.view, view.format, target.buffer, feature.inputs[slotIndex].host[0].size(), false))
    {
        g_fusedColorStage.free(slot.set); g_vk.destroyView(g_vk.device, slot.view, nullptr); slot = {}; return false;
    }
    slot.image = copy.image; slot.buffer = target.buffer; slot.owner = source; source->AddRef();
    slot.inputMode=rgba8 ? 5 : 1;
    logf("fused %s input conversion active for slot %d", rgba8 ? "RGBA8" : "R11G11B10", slotIndex);
    return true;
}

// Records the input copies into the slot's buffers. The resources are in
// COPY_SOURCE state already.

bool record_vram_inputs(Feature& feature,  ID3D12GraphicsCommandList* list,  InputSlot& slot,  const VramCopy* copies,  int count,  bool fusedColor)
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &cmd)))
        return false;
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &kBeforeTransfer, 0,
                 nullptr, 0, nullptr);
    for (int index = 0; index < count; ++index)
    {
        const VramCopy& copy = copies[index];
        VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
        g_vk.interop->GetVulkanImageLayout(copy.resource, D3D12_RESOURCE_STATE_COPY_SOURCE, &layout);
        if (index == 0 && fusedColor)
        {
            const VkMemoryBarrier writable = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
            g_vk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &writable, 0, nullptr, 0, nullptr);
            VkImageMemoryBarrier sampled = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            sampled.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sampled.oldLayout = layout; sampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sampled.srcQueueFamilyIndex = sampled.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sampled.image = copy.image; sampled.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            g_vk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &sampled);
            auto& stage = feature.fusedColor[&slot - feature.inputs];
            g_fusedColorStage.record(cmd, stage.set, copy.width, copy.height, stage.inputMode, slot.host[0].rowBytes / 8);
            sampled.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; sampled.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sampled.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; sampled.newLayout = layout;
            g_vk.barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &sampled);
            continue;
        }
        VkBufferImageCopy region = full_region(copy);
        static const Plane kPlanes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
        if (feature.linearInputs)
            region.bufferRowLength = static_cast<uint32_t>(slot.host[index].rowBytes / canonical_texel_bytes(kPlanes[index]));
        if (!copy.convert)
        {
            g_vk.copyImageToBuffer(cmd, copy.image, layout, slot.vram[index].buffer, 1, &region);
            continue;
        }
        // Convert colour to RGBA16F, motion to RG16F or exposure's first component
        // to R32F, then copy from the canonical image into the shared buffer.
        VkImage conversion = kPlanes[index] == Plane::Exposure ? feature.exposureConversion.image
                             : kPlanes[index] == Plane::Motion ? feature.motionConversion.image
                             : kPlanes[index] == Plane::Depth ? feature.depthConversion.image
                                                              : feature.colorConversion.image;
        VkImageMemoryBarrier toDestination = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toDestination.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toDestination.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDestination.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDestination.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDestination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDestination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDestination.image = conversion;
        toDestination.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                     &toDestination);
        VkImageBlit blit = {};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {static_cast<int32_t>(copy.width), static_cast<int32_t>(copy.height), 1};
        blit.dstSubresource = blit.srcSubresource;
        blit.dstOffsets[1] = blit.srcOffsets[1];
        g_vk.blit(cmd, copy.image, layout, conversion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                  VK_FILTER_NEAREST);
        VkImageMemoryBarrier toSource = toDestination;
        toSource.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSource.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                     &toSource);
        g_vk.copyImageToBuffer(cmd, conversion, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.vram[index].buffer, 1,
                               &region);
    }
    const VkMemoryBarrier completed = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    g_vk.barrier(cmd, fusedColor ? VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT :
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                 fusedColor ? &completed : &kAfterTransfer, 0,
                 nullptr, 0, nullptr);
    return SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
}

// Records the copy of a finished result into the output texture, which is in
// COPY_DEST state already.

static bool record_vram_output(ID3D12GraphicsCommandList* list, const VramBuffer& buffer, const VramCopy& copy,
                               const VramImage& conversion)
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &cmd)))
        return false;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    g_vk.interop->GetVulkanImageLayout(copy.resource, D3D12_RESOURCE_STATE_COPY_DEST, &layout);
    const VkBufferImageCopy region = full_region(copy);
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &kBeforeTransfer, 0,
                 nullptr, 0, nullptr);
    if (!copy.convert)
        g_vk.copyBufferToImage(cmd, buffer.buffer, copy.image, layout, 1, &region);
    else
    {
        VkImageMemoryBarrier toDestination = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toDestination.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toDestination.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDestination.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDestination.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDestination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDestination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDestination.image = conversion.image;
        toDestination.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                     &toDestination);
        g_vk.copyBufferToImage(cmd, buffer.buffer, conversion.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier toSource = toDestination;
        toSource.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSource.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                     &toSource);
        VkImageBlit blit = {};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {static_cast<int32_t>(copy.width), static_cast<int32_t>(copy.height), 1};
        blit.dstSubresource = blit.srcSubresource;
        blit.dstOffsets[1] = blit.srcOffsets[1];
        g_vk.blit(cmd, conversion.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copy.image, layout, 1, &blit,
                  VK_FILTER_NEAREST);
    }
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &kAfterTransfer, 0,
                 nullptr, 0, nullptr);
    return SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
}

// Reuse the storage-output shader to combine half decoding and the game's
// R11G11B10 conversion. Descriptors retain their images through allocator retirement.

bool record_feature_vram_output(Feature& feature,  unsigned slotIndex,  ID3D12GraphicsCommandList* list,  ID3D12Resource* output,  D3D12_RESOURCE_STATES outputState,  const VramCopy& copy,  uint32_t frame)
{
    feature.recordOutputFused=false;
    const auto& buffer=feature.outputs[slotIndex].vram;
    const auto desc=output->GetDesc();
    const unsigned requested=env_uint("D4R_SHIM_FUSE_R11_OUTPUT",0);
    // Mode 2 alternates warmed 300-frame blocks for same-launch total-cost comparisons.
    const bool selected=requested==1 || (requested==2 && (frame/300)%2==1);
    bool fused=g_fusedColorReady && feature.fusedOutputAllowed && copy.convert &&
        selected && desc.Format==DXGI_FORMAT_R11G11B10_FLOAT &&
        (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)!=0 &&
        buffer.bytes>=size_t(copy.width)*copy.height*8;
    auto& slot=feature.fusedOutput[slotIndex];
    if(fused && (slot.owner!=output || slot.image!=copy.image || slot.buffer!=buffer.buffer))
    {
        if(feature.retiredFusedOutput.size()>=64)
            feature.fusedOutputAllowed=fused=false;
        else
        {
            if(slot.owner) feature.retiredFusedOutput.push_back(slot);
            slot={};
            VkImageViewCreateInfo view={VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image=copy.image;view.viewType=VK_IMAGE_VIEW_TYPE_2D;view.format=VK_FORMAT_B10G11R11_UFLOAT_PACK32;
            view.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            fused=g_vk.createView(g_vk.device,&view,nullptr,&slot.view)==VK_SUCCESS;
            if(fused) fused=g_fusedColorStage.set(slot.set,slot.view,view.format,buffer.buffer,
                                                size_t(copy.width)*copy.height*8,true);
            if(fused)
            {
                slot.owner=output;output->AddRef();slot.image=copy.image;slot.buffer=buffer.buffer;
                logf("fused R11G11B10 output conversion active for slot %u",slotIndex);
            }
            else
            {
                g_fusedColorStage.free(slot.set);
                if(slot.view) g_vk.destroyView(g_vk.device,slot.view,nullptr);
                slot={};feature.fusedOutputAllowed=false;
            }
        }
    }
    if(fused)
    {
        transition(list,output,outputState,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        VkCommandBuffer cmd=VK_NULL_HANDLE;
        bool recorded=SUCCEEDED(g_vk.interop->BeginVkCommandBufferInterop(list,&cmd));
        if(recorded)
        {
            const VkMemoryBarrier before={VK_STRUCTURE_TYPE_MEMORY_BARRIER,nullptr,VK_ACCESS_MEMORY_WRITE_BIT,
                                          VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
            g_vk.barrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,
                         1,&before,0,nullptr,0,nullptr);
            g_fusedColorStage.record(cmd,slot.set,copy.width,copy.height);
            const VkMemoryBarrier after={VK_STRUCTURE_TYPE_MEMORY_BARRIER,nullptr,VK_ACCESS_SHADER_WRITE_BIT,
                                         VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT};
            g_vk.barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,
                         1,&after,0,nullptr,0,nullptr);
            recorded=SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
        }
        transition(list,output,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,outputState);
        if(recorded) {feature.recordOutputFused=true;return true;}
        feature.fusedOutputAllowed=false;
    }
    transition(list,output,outputState,D3D12_RESOURCE_STATE_COPY_DEST);
    const bool recorded=record_vram_output(list,buffer,copy,feature.outputConversion);
    transition(list,output,D3D12_RESOURCE_STATE_COPY_DEST,outputState);
    return recorded;
}
