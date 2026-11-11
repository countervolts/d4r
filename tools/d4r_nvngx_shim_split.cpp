// d4r_nvngx_shim_split.cpp: presenting each frame its own DLSS result (Split frames), and the
// denoiser's own alpha surface (DLSSD.OutputAlpha), including the GPU-resident path that keeps it on
// the GPU and the frame dumps that check the results.

// the command-list stubs live in d4r_nvngx_shim.cpp, which assembles them once
#define D4R_D3D12_SPLIT_DECL 1
#include "d4r_nvngx_shim_internal.h"

// Only the private wait word is released; never write the game's original readback marker. No HIP call
// belongs here: HIP may hold its queue lock while NGX submits, and this runs independently of that worker.
// The coherent allocation stays alive through CUDA drain and through recorded command-list retirement.

void release_ngx_input_wait_locked(Feature& feature, const char* reason, bool disable)
{
    if (!feature.ngxWaitActive || feature.ngxWaitCancelled)
        return;
    feature.ngxWaitCancelled = true;
    if (feature.ngxTimeline != nullptr) g.cu.timelineCancel(feature.ngxTimeline);
    if (disable)
        feature.ngxGpuWait.store(false);
    auto* word = static_cast<volatile uint32_t*>(feature.ngxInputSync.mapped) + 1 + feature.ngxWaitSlot;
    *word = feature.ngxWaitFrame;
    MemoryBarrier();
    logf("NGX frame %u: %s; CPU released the coherent GPU-side input wait; dropping after stream drain",
         feature.ngxWaitFrame, reason);
}

void ngx_input_watchdog()
{
    for (;;)
    {
        d4r_sleep_us(1000);
        // Registry ownership prevents feature destruction while we inspect or release its wait. The
        // worker never holds ngxWaitMutex across a CUDA call, including the wait enqueue itself.
        std::lock_guard<std::mutex> registryLock(g_featuresMutex);
        for (Feature* feature : g_features)
        {
            std::lock_guard<std::mutex> lock(feature->ngxWaitMutex);
            if (!feature->ngxWaitActive || feature->ngxWaitCancelled)
                continue;
            const double elapsedUs = std::chrono::duration<double, std::micro>(
                ProfileClock::now() - feature->ngxWaitStart).count();
            const uint32_t aggregate = feature->markerValue[0];
            MemoryBarrier();
            // Read the exact slot after the aggregate, as the CPU path's recheck does: a marker that
            // landed between the reads must not be mistaken for a discarded command list.
            const auto reason = d4r_input_wait_abort(feature->ngxWaitFrame,
                feature->markerValue[1 + feature->ngxWaitSlot], aggregate, feature->retiring.load(), elapsedUs);
            if (reason != D4rInputWaitAbort::None)
                release_ngx_input_wait_locked(*feature,
                    reason == D4rInputWaitAbort::Retired ? "the feature retired" :
                    reason == D4rInputWaitAbort::Discarded ? "its command list was discarded" :
                    "the input wait or evaluation timed out (5 s)", reason == D4rInputWaitAbort::Timeout);
        }
    }
}

void release_split_frame(Feature* feature, uint32_t frame)
{
    std::lock_guard<std::mutex> lock(feature->splitMutex);
    const uint64_t value = feature->splitCompletion.retire(frame);
    if (value <= feature->splitSignalled)
        return;
    const VkResult result = signal_split_semaphore(*feature, value);
    if (result == VK_SUCCESS)
        feature->splitSignalled = value;
    if (result != VK_SUCCESS)
        logf("split frame %llu signal failed (VkResult %d)", static_cast<unsigned long long>(value), result);
}

static void split_watchdog()
{
    const auto timeout = std::chrono::milliseconds(env_uint("D4R_SHIM_SPLIT_TIMEOUT_MS", 200));
    struct Seen
    {
        uint32_t marker = 0;
        std::chrono::steady_clock::time_point at;
    };
    std::vector<std::pair<Feature*, Seen>> seen;
    for (;;)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::lock_guard<std::mutex> lock(g_featuresMutex);
        const auto now = std::chrono::steady_clock::now();
        for (Feature* feature : g_features)
        {
            if (!feature->split || feature->markerValue == nullptr)
                continue;
            auto it = std::find_if(seen.begin(), seen.end(), [feature](const auto& entry) { return entry.first == feature; });
            if (it == seen.end())
                it = seen.insert(seen.end(), {feature, Seen{}});
            const uint32_t marker = *feature->markerValue;
            uint64_t signalled;
            {
                std::lock_guard<std::mutex> splitLock(feature->splitMutex);
                signalled = feature->splitSignalled;
            }
            if (marker <= signalled)
                continue; // nothing on the GPU is waiting on us
            if (it->second.marker != marker)
                it->second = Seen{marker, now};
            else if (now - it->second.at > timeout)
            {
                // A slow producer is not a completed producer. Releasing this
                // wait would race the game's output copy with a late CUDA write.
                logf("split frame %u still waiting for DLSS; preserving output synchronization", marker);
                it->second.at = now;
            }
        }
        // Forget released features.
        seen.erase(std::remove_if(seen.begin(), seen.end(),
                                  [](const auto& entry) {
                                      return std::find(g_features.begin(), g_features.end(), entry.first) ==
                                             g_features.end();
                                  }),
                   seen.end());
    }
}

bool create_split_semaphore(Feature& feature)
{
    VkSemaphoreTypeCreateInfo type = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0;
    const VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    if (g_vk.createSemaphore(g_vk.device, &info, nullptr, &feature.splitSemaphore) != VK_SUCCESS)
    {
        logf("split frames: vkCreateSemaphore failed");
        return false;
    }
    static std::once_flag watchdog;
    std::call_once(watchdog, [] { std::thread(split_watchdog).detach(); });
    return true;
}

// Game thread: grows a buffer; an outgrown one is kept until the feature is
// released since queued command lists may still reference it.

bool ensure_vram_buffer(Feature& feature, VramBuffer& buffer, size_t bytes)
{
    const bool allocated = buffer.buffer != VK_NULL_HANDLE || buffer.d3d != nullptr;
    if (allocated && buffer.bytes >= bytes)
        return true;
    if (allocated)
        feature.retiredBuffers.push_back(buffer);
    buffer = VramBuffer{};
    return g_vk.interop == nullptr ? create_native_vram_buffer(buffer, bytes) : create_vram_buffer(buffer, bytes);
}

bool ensure_conversion_image(Feature& feature, VramImage& image, UINT width, UINT height, VkFormat format)
{
    if (image.image != VK_NULL_HANDLE && image.width == width && image.height == height && image.format == format)
        return true;
    if (image.image != VK_NULL_HANDLE)
        feature.retiredImages.push_back(image);
    image = VramImage{};
    return create_vram_image(image, width, height, format);
}

// Diagnostic first: native inputs travel GPU image/buffer -> per-slot imported
// VRAM -> HIP. Only selected capture frames pay for readback/reference checks.
// This composes opaque+camera motion; it does not substitute live NGX inputs.

void record_native_gpu_check(Feature& feature, ID3D12GraphicsCommandList* list,
        InputSlot& slot, ID3D12Resource* motion)
{
    slot.nativeGpuCheckReady=false;
    slot.nativeLive=false;
    const bool live=(!g_portable.active || g_portable.nativeMotionVectors) &&
        env_uint("D4R_NATIVE_MV_LIVE",0)==1 && !feature.rayReconstruction && (feature.flags&2)!=0;
    const bool check=env_uint("D4R_NATIVE_MV_GPU_COMPOSE_CHECK",0)==1 && slot.nativeOpaqueCaptureReady &&
        slot.ownedSceneCaptureReady && slot.ownedMatrixCaptureReady && slot.ownedDepthCaptureReady &&
        slot.ownedValidityCaptureReady;
    if((!live && !check) || !g.cu.composeNativeMotion || !g_vk.lifetime) return;
    if(!slot.ownedValidityCaptureReady)
    {
        // Live frames have no diagnostic capture; the GPU validity words are read after the input marker.
        slot.ownedValidityCaptureReady=record_owned_motion_constants(list,slot.ownedValidityCapture,motion,0x75367843,8);
        if(!slot.ownedValidityCaptureReady) return;
        if(slot.nativeValidityOwner) slot.nativeValidityOwner->Release();
        slot.nativeValidityOwner=motion;motion->AddRef();
    }
    ID3D12Resource* sources[4]={};
    const uint32_t ids[4]={0x75367842,0x7536783b,0x75367840,0x75367841};
    D3D12_RESOURCE_DESC desc[4]={};
    VramCopy copies[2]={};UINT64 handles[2]={},offsets[2]={};
    bool ready=true;
    for(int i=0;i<4;i++)
    {
        const GUID guid={ids[i],0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
        UINT size=sizeof(sources[i]);
        if(FAILED(motion->GetPrivateData(guid,&size,&sources[i])) || size!=sizeof(sources[i]) || !sources[i]) { ready=false;break; }
        sources[i]->GetDesc(&desc[i]);
        if(FAILED(g_vk.lifetime->RetainExternalResources(list,sources[i]))) { ready=false;break; }
    }
    if(ready)
    {
        ready=(desc[0].Format==DXGI_FORMAT_R32_TYPELESS || desc[0].Format==DXGI_FORMAT_D32_FLOAT) &&
            desc[1].Format==DXGI_FORMAT_R16G16_FLOAT && desc[0].Width==desc[1].Width && desc[0].Height==desc[1].Height &&
            desc[0].Width==feature.outputWidth && desc[0].Height==feature.outputHeight && desc[0].Width<=4096 && desc[0].Height<=4096 &&
            desc[2].Dimension==D3D12_RESOURCE_DIMENSION_BUFFER && desc[2].Width==1136 &&
            desc[3].Dimension==D3D12_RESOURCE_DIMENSION_BUFFER && desc[3].Width==64;
        for(int i=0;i<2 && ready;i++)
            ready=desc[i].Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc[i].SampleDesc.Count==1 &&
                desc[i].MipLevels==1 && desc[i].DepthOrArraySize==1 &&
                describe_vram_copy(sources[i],i==0 ? Plane::Depth : Plane::Motion,copies[i]) && !copies[i].convert;
        for(int i=0;i<2 && ready;i++)
        {
            VkFormat format=VK_FORMAT_UNDEFINED;
            ready=SUCCEEDED(g_vk.interop->GetVulkanResourceInfo1(sources[i+2],&handles[i],&offsets[i],&format)) && handles[i];
        }
        const size_t pixels=static_cast<size_t>(desc[0].Width)*desc[0].Height;
        for(int i=0;i<4 && ready;i++)
            ready=ensure_vram_buffer(feature,slot.nativeGpuInputs[i],i<2 ? pixels*4 : i==2 ? 1136 : 64);
    }
    if(ready)
    {
        for(int i=0;i<2;i++) transition(list,sources[i],D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        VkCommandBuffer cmd=VK_NULL_HANDLE;
        ready=SUCCEEDED(g_vk.interop->BeginVkCommandBufferInterop(list,&cmd));
        if(ready)
        {
            g_vk.barrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&kBeforeTransfer,0,nullptr,0,nullptr);
            for(int i=0;i<2;i++)
            {
                VkImageLayout layout=VK_IMAGE_LAYOUT_GENERAL;
                g_vk.interop->GetVulkanImageLayout(sources[i],D3D12_RESOURCE_STATE_COPY_SOURCE,&layout);
                const VkBufferImageCopy region=full_region(copies[i]);
                g_vk.copyImageToBuffer(cmd,copies[i].image,layout,slot.nativeGpuInputs[i].buffer,1,&region);
                const VkBufferCopy regionBuffer={offsets[i],0,i==0 ? 1136u : 64u};
                g_vk.copyBuffer(cmd,reinterpret_cast<VkBuffer>(handles[i]),slot.nativeGpuInputs[i+2].buffer,1,&regionBuffer);
            }
            g_vk.barrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&kAfterTransfer,0,nullptr,0,nullptr);
            ready=SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
        }
        for(int i=0;i<2;i++) transition(list,sources[i],D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
    }
    if(ready) { slot.nativeGpuWidth=static_cast<UINT>(desc[0].Width);slot.nativeGpuHeight=desc[0].Height;slot.nativeGpuCheckReady=true;slot.nativeLive=live; }
    for(auto* source : sources) if(source) source->Release();
    {
        static std::atomic<unsigned> logged{0};
        if(!live || logged.fetch_add(1)<16) logf("native GPU composition inputs recorded=%d live=%d",ready,live);
    }
}

void release_vram(Feature& feature)
{
    for (auto& stage : feature.fusedOutput) feature.retiredFusedOutput.push_back(stage);
    for (auto& stage : feature.retiredFusedOutput)
    {
        g_fusedColorStage.free(stage.set);
        if (stage.view != VK_NULL_HANDLE) g_vk.destroyView(g_vk.device, stage.view, nullptr);
        if (stage.owner != nullptr) stage.owner->Release();
    }
    feature.retiredFusedOutput.clear();
    for (auto& stage : feature.fusedColor) feature.retiredFusedColor.push_back(stage);
    for (auto& stage : feature.retiredFusedColor)
    {
        g_fusedColorStage.free(stage.set);
        if (stage.view != VK_NULL_HANDLE) g_vk.destroyView(g_vk.device, stage.view, nullptr);
        if (stage.owner != nullptr) stage.owner->Release();
    }
    feature.retiredFusedColor.clear();
    if (feature.ngxTimeline != nullptr)
    {
        g.cu.timelineDestroy(feature.ngxTimeline);
        feature.ngxTimeline = nullptr;
    }
    recycle_vram_buffer(feature.ngxInputSync);
    if (feature.splitSemaphore != VK_NULL_HANDLE)
        g_vk.destroySemaphore(g_vk.device, feature.splitSemaphore, nullptr);
    release_native_split(feature);
    for (InputSlot& slot : feature.inputs)
        for (VramBuffer& buffer : slot.vram)
            recycle_vram_buffer(buffer);
    for (InputSlot& slot : feature.inputs)
        for (VramBuffer& buffer : slot.nativeGpuInputs) recycle_vram_buffer(buffer);
    for (OutputSlot& slot : feature.outputs)
        recycle_vram_buffer(slot.vram);
    for (AlphaSlot& slot : feature.alpha)
    {
        recycle_vram_buffer(slot.vram);
        destroy_vram_image(slot.conversion);
    }
    for (VramBuffer& buffer : feature.retiredBuffers)
        recycle_vram_buffer(buffer);
    feature.retiredBuffers.clear();
    destroy_vram_image(feature.colorConversion);
    destroy_vram_image(feature.depthConversion);
    destroy_vram_image(feature.motionConversion);
    destroy_vram_image(feature.exposureConversion);
    destroy_vram_image(feature.outputConversion);
    for (VramImage& image : feature.retiredImages)
        destroy_vram_image(image);
    feature.retiredImages.clear();
}

// Prep stage: converts one readback staging buffer into the canonical layout.

void stage_plane(Plane plane, const Staging& staging, HostPlane& host, FrameTiming* timing, int timingIndex,
                        size_t rowPitch)
{
    const size_t texelBytes = (plane_format(plane) == CUDA_FORMAT_HALF ? 2 : 4) * plane_channels(plane);
    const size_t packedRowBytes = texelBytes * staging.width;
    host.rowBytes = std::max(rowPitch, packedRowBytes);
    reserve_host(host, host.rowBytes * staging.height);
    host.width = staging.width;
    host.height = staging.height;
    uint8_t* rows = host.bytes;
    const size_t rowBytes = host.rowBytes;
    const uint8_t* source = staging.mapped + staging.layout.Offset;
    const size_t sourcePitch = staging.layout.Footprint.RowPitch;
    const auto conversionStart = timing != nullptr ? ProfileClock::now() : ProfileClock::time_point{};
    if (canonical_input(plane, staging.format))
        copy_rows(g_inputRows, rows, rowBytes, source, sourcePitch, packedRowBytes, staging.height);
    else
        g_inputRows.run(staging.height, rowBytes * staging.height, [&](UINT begin, UINT end) {
            for (UINT y = begin; y < end; ++y)
                if (plane == Plane::Color && staging.format == DXGI_FORMAT_R11G11B10_FLOAT)
                    convert_r11g11b10_row(source + y * sourcePitch,
                                          reinterpret_cast<uint16_t*>(rows + rowBytes * y), staging.width);
                else
                    convert_row_in(plane, staging.format, source + y * sourcePitch, rows + rowBytes * y,
                                   staging.width);
        });
    if (timing != nullptr)
        timing->convert[timingIndex] = profile_ms(conversionStart, ProfileClock::now());
}

// Worker stage: uploads one staged plane into its CUDA array.
// D4R_SHIM_MV_STATS=N logs, every N frames, what the game's motion vectors
// look like (staged RG16F plane): how much of the texture is populated, the
// populated bounding box (a render-size sub-rectangle means render-resolution
// vectors in a display-size texture), magnitudes in texture units, and the
// global image shift between the previous and current colour frames measured
// in render pixels, so the vectors' unit can be read off their ratio.

static std::vector<float> downsampled_luma(const HostPlane& color, UINT& width, UINT& height)
{
    // RGBA16F colour, 2x2 box-downsampled luma.
    width = color.width / 2, height = color.height / 2;
    std::vector<float> luma(static_cast<size_t>(width) * height);
    const uint8_t* bytes = color.bytes != nullptr ? color.bytes : color.fallback.data();
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x)
        {
            float sum = 0.0f;
            for (UINT dy = 0; dy < 2; ++dy)
            {
                const uint16_t* row = reinterpret_cast<const uint16_t*>(bytes + (2 * y + dy) * color.rowBytes);
                for (UINT dx = 0; dx < 2; ++dx)
                    for (UINT c = 0; c < 3; ++c)
                        sum += static_cast<float>(std::bit_cast<_Float16>(row[(2 * x + dx) * 4 + c]));
            }
            luma[static_cast<size_t>(y) * width + x] = sum / (sum + 12.0f); // tone-mapped (HDR input)
        }
    return luma;
}

void log_motion_stats(const HostPlane& host, const HostPlane& color, uint32_t frame)
{
    static const uint32_t every = env_uint("D4R_SHIM_MV_STATS", 0);
    static std::vector<float> previousLuma;
    static uint32_t previousFrame = 0;
    if (every == 0 || host.width == 0 || color.width == 0)
        return;
    const bool report = frame % every == 0;
    if (!report && (frame + 1) % every != 0)
        return;
    UINT lumaWidth = 0, lumaHeight = 0;
    std::vector<float> luma = downsampled_luma(color, lumaWidth, lumaHeight);
    if (!report)
    {
        previousLuma = std::move(luma);
        previousFrame = frame;
        return;
    }
    const uint8_t* bytes = host.bytes != nullptr ? host.bytes : host.fallback.data();
    size_t nonzero = 0;
    double sumMagnitude = 0.0;
    float maxMagnitude = 0.0f;
    UINT minX = host.width, minY = host.height, maxX = 0, maxY = 0;
    std::vector<float> xs, ys;
    for (UINT y = 0; y < host.height; ++y)
    {
        const uint16_t* row = reinterpret_cast<const uint16_t*>(bytes + y * host.rowBytes);
        for (UINT x = 0; x < host.width; ++x)
        {
            const float mx = static_cast<float>(std::bit_cast<_Float16>(row[x * 2]));
            const float my = static_cast<float>(std::bit_cast<_Float16>(row[x * 2 + 1]));
            if ((x % 8) == 0 && (y % 8) == 0)
            {
                xs.push_back(mx);
                ys.push_back(my);
            }
            if (mx == 0.0f && my == 0.0f)
                continue;
            ++nonzero;
            const float magnitude = std::sqrt(mx * mx + my * my);
            sumMagnitude += magnitude;
            maxMagnitude = std::max(maxMagnitude, magnitude);
            minX = std::min(minX, x), maxX = std::max(maxX, x);
            minY = std::min(minY, y), maxY = std::max(maxY, y);
        }
    }
    auto median = [](std::vector<float>& values) {
        if (values.empty())
            return 0.0f;
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        return values[values.size() / 2];
    };
    const float medianX = median(xs), medianY = median(ys);
    // Global shift d (render pixels, current -> previous, the vectors'
    // convention) minimising |cur(p) - prev(p + d)| over the central region.
    int bestX = 0, bestY = 0;
    double bestCost = -1.0, zeroCost = 0.0;
    if (previousFrame + 1 == frame && previousLuma.size() == luma.size())
    {
        const int range = 80, rangeY = 40; // in downsampled pixels
        const int x0 = range, x1 = static_cast<int>(lumaWidth) - range;
        const int y0 = rangeY, y1 = static_cast<int>(lumaHeight) - rangeY;
        for (int dy = -rangeY; dy <= rangeY; ++dy)
            for (int dx = -range; dx <= range; ++dx)
            {
                double cost = 0.0;
                for (int y = y0; y < y1; y += 2)
                    for (int x = x0; x < x1; x += 2)
                        cost += std::fabs(luma[static_cast<size_t>(y) * lumaWidth + x] -
                                          previousLuma[static_cast<size_t>(y + dy) * lumaWidth + (x + dx)]);
                if (dx == 0 && dy == 0)
                    zeroCost = cost;
                if (bestCost < 0.0 || cost < bestCost)
                    bestCost = cost, bestX = dx, bestY = dy;
            }
    }
    const double pixels = static_cast<double>(host.width) * host.height;
    logf("MV_STATS frame=%u size=%ux%u nonzero=%.4f bbox=%u,%u-%u,%u median=(%.3f,%.3f) mean_abs=%.3f max=%.3f "
         "image_shift_render_px=(%d,%d) match=%.3f",
         frame, host.width, host.height, nonzero / pixels, nonzero ? minX : 0, nonzero ? minY : 0, maxX, maxY,
         medianX, medianY, nonzero ? sumMagnitude / nonzero : 0.0, maxMagnitude, 2 * bestX, 2 * bestY,
         zeroCost > 0.0 ? bestCost / zeroCost : 1.0);
}

bool upload_plane(Plane plane, const HostPlane& host, CudaImage& image, uint32_t frame, FrameTiming* timing,
                         int timingIndex)
{
    if (!ensure_cuda_image(image, host.width, host.height, plane_format(plane), plane_channels(plane), false,
                           plane == Plane::Color))
        return false;
    if ((frame <= 3 || env_uint("D4R_SHIM_INPUT_HASH_ALL", 0) != 0) && GetEnvironmentVariableA("D4R_SHIM_INPUT_HASH", nullptr, 0) != 0)
    {
        uint64_t hash = 1469598103934665603ull;
        for (size_t index = 0; index < host.size(); ++index)
            hash = (hash ^ host.bytes[index]) * 1099511628211ull;
        logf("frame %u upload plane=%u array=%p object=0x%llx bytes=%zu hash=%016llx", frame,
             static_cast<unsigned int>(plane), image.array,
             static_cast<unsigned long long>(image.object), host.size(),
             static_cast<unsigned long long>(hash));
    }
    CudaMemcpy2D copy = {};
    copy.srcMemoryType = CUDA_MEMORY_HOST;
    copy.srcHost = host.bytes;
    copy.srcPitch = host.rowBytes;
    copy.dstMemoryType = CUDA_MEMORY_ARRAY;
    copy.dstArray = image.array;
    copy.WidthInBytes = host.rowBytes;
    copy.Height = host.height;
    const auto copyStart = timing != nullptr ? ProfileClock::now() : ProfileClock::time_point{};
    const int result = g.cu.memcpy2D(&copy);
    if (timing != nullptr)
        timing->upload[timingIndex] = profile_ms(copyStart, ProfileClock::now());
    if (result != 0)
        logf("cuMemcpy2D upload failed: %d", result);
    if (result != 0)
        return false;
    if (GetEnvironmentVariableA("D4R_SHIM_SYNC_UPLOAD", nullptr, 0) != 0)
    {
        const int syncResult = g.cu.ctxSynchronize();
        if (syncResult != 0)
            logf("cuCtxSynchronize after upload failed: %d", syncResult);
        return syncResult == 0;
    }
    return true;
}

void log_output_hash(const HostPlane& host, uint32_t frame)
{
    if (GetEnvironmentVariableA("D4R_SHIM_OUTPUT_HASH", nullptr, 0) == 0)
        return;
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < host.size(); ++index)
        hash = (hash ^ host.bytes[index]) * 1099511628211ull;
    logf("frame %u output hash=%016llx", frame, static_cast<unsigned long long>(hash));
}

static std::atomic<uint32_t> g_captureFirst{0}; // first frame of a triggered capture

void arm_native_capture(uint32_t frame)
{
    char trigger[MAX_PATH];
    const DWORD length=GetEnvironmentVariableA("D4R_SHIM_CAPTURE_TRIGGER",trigger,MAX_PATH);
    if(length && length<MAX_PATH && GetFileAttributesA(trigger)!=INVALID_FILE_ATTRIBUTES)
    {
        uint32_t expected=0;
        if(g_captureFirst.compare_exchange_strong(expected,frame))
            logf("native capture armed before frame %u input copies",frame);
    }
}

bool dump_frame_selected(uint32_t frame, const char* startName, const char* countName, const char* everyName)
{
    char trigger[MAX_PATH];
    const DWORD triggerLength = GetEnvironmentVariableA("D4R_SHIM_CAPTURE_TRIGGER", trigger, MAX_PATH);
    if (triggerLength > 0 && triggerLength < MAX_PATH)
    {
        uint32_t captured = g_captureFirst.load();
        // Native D3D12 captures arm only on the recording thread, before its
        // copies. Preserve the existing trigger behavior for other captures.
        const bool nativeCapture=GetEnvironmentVariableA("D4R_NATIVE_MV_CAPTURE_DIR",nullptr,0)!=0;
        if (captured == 0 && !nativeCapture && GetFileAttributesA(trigger) != INVALID_FILE_ATTRIBUTES)
        {
            uint32_t expected = 0;
            if (g_captureFirst.compare_exchange_strong(expected, frame))
                logf("raw input/output capture triggered on frame %u", frame);
            captured = g_captureFirst.load();
        }
        // D4R_SHIM_CAPTURE_COUNT=N keeps capturing N consecutive frames (replay sequences).
        static const uint32_t count = std::max(1u, env_uint("D4R_SHIM_CAPTURE_COUNT", 1));
        return captured != 0 && frame >= captured && frame - captured < count;
    }
    const uint32_t start = env_uint(startName, 1);
    const uint32_t count = env_uint(countName, 1);
    const uint32_t every = std::max(1u, env_uint(everyName, 1));
    return frame >= start && count != 0 && (frame - start) / every < count && (frame - start) % every == 0;
}

bool output_dump_path(uint32_t frame, char (&path)[MAX_PATH])
{
    char directory[MAX_PATH];
    const DWORD length = GetEnvironmentVariableA("D4R_SHIM_OUTPUT_DUMP_DIR", directory, MAX_PATH);
    if (length == 0 || length >= MAX_PATH)
        return false;
    if (!dump_frame_selected(frame, "D4R_SHIM_OUTPUT_DUMP_START", "D4R_SHIM_OUTPUT_DUMP_COUNT",
                             "D4R_SHIM_OUTPUT_DUMP_EVERY"))
        return false;
    const int written = std::snprintf(path, MAX_PATH, "%s\\frame-%06u.rgba16f", directory, frame);
    return written > 0 && written < MAX_PATH;
}

// Capture files. With D4R_SHIM_CAPTURE_MEMORY=1 they are kept in memory and
// written by a background thread once the triggered capture is complete, so a
// sequence can be recorded at the game's normal frame rate.

static std::mutex g_captureMutex;

static std::vector<std::pair<std::string, std::vector<uint8_t>>> g_captureFiles;

static CapturePool g_capturePool;

static std::vector<std::pair<std::string, std::pair<const uint8_t*, size_t>>> g_capturePinned;

// Worker thread (a CUDA context is current). Returns nullptr when memory
// captures are off or the pool is exhausted; the caller then uses ordinary memory.

uint8_t* capture_pinned(const char* path, size_t bytes)
{
    if (env_uint("D4R_SHIM_CAPTURE_MEMORY", 0) == 0 || g.cu.memAllocHost == nullptr)
        return nullptr;
    std::lock_guard<std::mutex> lock(g_captureMutex);
    CapturePool& pool = g_capturePool;
    if (!pool.tried)
    {
        pool.tried = true;
        for (size_t mb = env_uint("D4R_SHIM_CAPTURE_POOL_MB", 3072); mb >= 256 && pool.base == nullptr; mb /= 2)
        {
            void* memory = nullptr;
            if (g.cu.memAllocHost(&memory, mb << 20) == 0)
                pool.base = static_cast<uint8_t*>(memory), pool.size = mb << 20;
        }
        logf("capture pool: %zu MB pinned", pool.size >> 20);
    }
    const size_t aligned = (bytes + 4095) & ~size_t(4095);
    if (pool.base == nullptr || pool.used + aligned > pool.size)
        return nullptr;
    uint8_t* block = pool.base + pool.used;
    pool.used += aligned;
    g_capturePinned.emplace_back(path, std::make_pair(block, bytes));
    return block;
}

void write_capture_file(uint32_t frame, const char* what, const char* path, const void* data, size_t bytes)
{
    static const bool memory = env_uint("D4R_SHIM_CAPTURE_MEMORY", 0) != 0;
    if (memory)
    {
        if (uint8_t* pinned = capture_pinned(path, bytes))
        {
            std::memcpy(pinned, data, bytes);
            return;
        }
        const auto* first = static_cast<const uint8_t*>(data);
        std::lock_guard<std::mutex> lock(g_captureMutex);
        g_captureFiles.emplace_back(path, std::vector<uint8_t>(first, first + bytes));
        return;
    }
    FILE* file = std::fopen(path, "wb");
    if (file == nullptr || std::fwrite(data, 1, bytes, file) != bytes)
        logf("frame %u %s dump failed: %s", frame, what, path);
    else
        logf("frame %u %s saved: %s", frame, what, path);
    if (file != nullptr)
        std::fclose(file);
}

// Called after each frame's output dump: once the last triggered frame is in,
// the buffered files go to disk.

void flush_capture_if_complete(uint32_t frame)
{
    char trigger[MAX_PATH];
    if (env_uint("D4R_SHIM_CAPTURE_MEMORY", 0) == 0 ||
        GetEnvironmentVariableA("D4R_SHIM_CAPTURE_TRIGGER", trigger, MAX_PATH) == 0)
        return;
    const uint32_t count = std::max(1u, env_uint("D4R_SHIM_CAPTURE_COUNT", 1));
    const uint32_t first = g_captureFirst.load();
    if (first == 0 || frame != first + count - 1)
        return;
    std::vector<std::pair<std::string, std::vector<uint8_t>>> files;
    std::vector<std::pair<std::string, std::pair<const uint8_t*, size_t>>> pinned;
    {
        std::lock_guard<std::mutex> lock(g_captureMutex);
        files.swap(g_captureFiles);
        pinned.swap(g_capturePinned);
    }
    logf("capture of frames %u-%u complete in memory; writing %zu files", first, frame, files.size() + pinned.size());
    // The pinned pool is not reused, so its blocks stay valid for the writer.
    std::thread([files = std::move(files), pinned = std::move(pinned), first, frame] {
        size_t failed = 0;
        auto write = [&failed](const std::string& path, const uint8_t* data, size_t bytes) {
            FILE* file = std::fopen(path.c_str(), "wb");
            if (file == nullptr || std::fwrite(data, 1, bytes, file) != bytes)
                ++failed;
            if (file != nullptr)
                std::fclose(file);
        };
        for (const auto& [path, block] : pinned)
            write(path, block.first, block.second);
        for (const auto& [path, bytes] : files)
            write(path, bytes.data(), bytes.size());
        logf("capture of frames %u-%u written (%zu files, %zu failed)", first, frame, files.size() + pinned.size(),
             failed);
    }).detach();
}

void dump_output(const HostPlane& host, uint32_t frame, const char* path)
{
    write_capture_file(frame, "raw DLSS output", path, host.bytes, host.size());
    flush_capture_if_complete(frame);
}

bool input_dump_directory(uint32_t frame, char (&directory)[MAX_PATH])
{
    const DWORD length = GetEnvironmentVariableA("D4R_SHIM_INPUT_DUMP_DIR", directory, MAX_PATH);
    return length != 0 && length < MAX_PATH &&
           dump_frame_selected(frame, "D4R_SHIM_INPUT_DUMP_START", "D4R_SHIM_INPUT_DUMP_COUNT",
                               "D4R_SHIM_INPUT_DUMP_EVERY");
}

void dump_input_planes(const HostPlane (&planes)[4], uint32_t frame, const FrameParams& p,
                              const char* directory)
{
    constexpr const char* names[4] = {"color.rgba16f", "depth.r32f", "motion.rg16f", "exposure.r32f"};
    for (int index = 0; index < (planes[3].width != 0 ? 4 : 3); ++index)
    {
        const HostPlane& host = planes[index];
        const uint8_t* bytes = host.bytes != nullptr ? host.bytes : host.fallback.data();
        if (bytes == nullptr || host.width == 0 || host.height == 0)
            continue;
        char path[MAX_PATH];
        const int written = std::snprintf(path, MAX_PATH, "%s\\frame-%06u-%s", directory, frame, names[index]);
        if (written > 0 && written < MAX_PATH)
            write_capture_file(frame, names[index], path, bytes, host.size());
    }
    // The evaluation parameters, so the harness can replay the sequence. The exposure plane's
    // own geometry and the feature's creation state are recorded here too: a capture that kept a
    // plane without its size, or a sequence without the model it ran under, would leave the
    // replay guessing at both.
    char path[MAX_PATH];
    const int written = std::snprintf(path, MAX_PATH, "%s\\frame-%06u-params.txt", directory, frame);
    if (written <= 0 || written >= MAX_PATH)
        return;
    char text[1536];
    const int length = std::snprintf(
        text, sizeof(text),
        "jitter %.9g %.9g\nmv_scale %.9g %.9g\nsharpness %.9g\npre_exposure %.9g\n"
        "exposure_scale %.9g\nframe_time %.9g\nreset %d\ninvert %d %d\nrender %u %u\n"
        "color_base %u %u\ndepth_base %u %u\nmv_base %u %u\noutput_base %u %u\nhas_exposure %d\n"
        "exposure_size %u %u\ncreate %u %u %u %u %d %d %d %u\n",
        p.jitterX, p.jitterY, p.mvScaleX, p.mvScaleY, p.sharpness, p.preExposure, p.exposureScale, p.frameTime,
        p.reset, p.invertX, p.invertY, p.renderWidth, p.renderHeight, p.colorBaseX, p.colorBaseY, p.depthBaseX,
        p.depthBaseY, p.mvBaseX, p.mvBaseY, p.outputBaseX, p.outputBaseY, p.hasExposure ? 1 : 0, p.exposureWidth,
        p.exposureHeight, p.createWidth, p.createHeight, p.createOutWidth, p.createOutHeight, p.createQuality,
        p.createFlags, p.createSubrects, p.createPreset);
    if (length > 0)
        write_capture_file(frame, "params", path, text, static_cast<size_t>(length));
}

// VRAM interop without D4R_SHIM_VRAM_VERIFY: reads the selected frame's input
// planes straight from the imported buffers DLSS is about to consume.

void dump_vram_inputs(const InputSlot& slot, uint32_t frame, const FrameParams& p, int count)
{
    char directory[MAX_PATH];
    if (!input_dump_directory(frame, directory))
        return;
    HostPlane planes[4];
    constexpr const char* names[4] = {"color.rgba16f", "depth.r32f", "motion.rg16f", "exposure.r32f"};
    bool allPinned = true;
    for (int index = 0; index < count; ++index)
    {
        const HostPlane& geometry = slot.host[index];
        HostPlane& plane = planes[index];
        plane.width = geometry.width, plane.height = geometry.height, plane.rowBytes = geometry.rowBytes;
        char path[MAX_PATH];
        std::snprintf(path, MAX_PATH, "%s\\frame-%06u-%s", directory, frame, names[index]);
        plane.bytes = capture_pinned(path, geometry.size());
        if (plane.bytes == nullptr)
        {
            allPinned = false;
            plane.fallback.resize(geometry.size());
        }
        CudaMemcpy2D readback = {};
        readback.srcMemoryType = CUDA_MEMORY_DEVICE;
        readback.srcDevice = slot.vram[index].device;
        readback.srcPitch = geometry.rowBytes;
        readback.dstMemoryType = CUDA_MEMORY_HOST;
        readback.dstHost = plane.bytes != nullptr ? plane.bytes : plane.fallback.data();
        readback.dstPitch = geometry.rowBytes;
        readback.WidthInBytes = geometry.rowBytes;
        readback.Height = geometry.height;
        if (g.cu.memcpy2D(&readback) != 0)
        {
            logf("frame %u: VRAM input readback for the dump failed", frame);
            return;
        }
    }
    // Pinned planes are already registered for writing; only the rest (and the
    // parameters) go through dump_input_planes.
    for (HostPlane& plane : planes)
        if (plane.bytes != nullptr)
            plane.width = 0;
    (void)allPinned;
    dump_input_planes(planes, frame, p, directory);
}

// Worker stage: downloads the DLSS output into host memory.

bool download_output(Feature& feature, HostPlane& host, uint32_t frame, FrameTiming* timing)
{
    reserve_host(host, static_cast<size_t>(feature.output.width) * 8 * feature.output.height);
    host.width = feature.output.width;
    host.height = feature.output.height;
    host.rowBytes = static_cast<size_t>(feature.output.width) * 8;
    CudaMemcpy2D copy = {};
    copy.srcMemoryType = CUDA_MEMORY_ARRAY;
    copy.srcArray = feature.output.array;
    copy.dstMemoryType = CUDA_MEMORY_HOST;
    copy.dstHost = host.bytes;
    copy.dstPitch = host.rowBytes;
    copy.WidthInBytes = host.rowBytes;
    copy.Height = host.height;
    const auto copyStart = timing != nullptr ? ProfileClock::now() : ProfileClock::time_point{};
    const int result = g.cu.memcpy2D(&copy);
    if (timing != nullptr)
        timing->download = profile_ms(copyStart, ProfileClock::now());
    if (result != 0)
    {
        logf("cuMemcpy2D download failed: %d", result);
        return false;
    }
    log_output_hash(host, frame);
    char dumpPath[MAX_PATH];
    if (output_dump_path(frame, dumpPath))
        dump_output(host, frame, dumpPath);
    return true;
}

// Finish stage: converts a downloaded result into an upload staging buffer.

void write_output(Staging& staging, const HostPlane& host, FrameTiming* timing)
{
    const auto conversionStart = timing != nullptr ? ProfileClock::now() : ProfileClock::time_point{};
    uint8_t* destination = staging.mapped + staging.layout.Offset;
    const size_t destinationPitch = staging.layout.Footprint.RowPitch;
    const UINT width = staging.width < host.width ? staging.width : host.width;
    const UINT height = staging.height < host.height ? staging.height : host.height;
    const uint8_t* rows = host.bytes;
    if (staging.format == DXGI_FORMAT_R16G16B16A16_FLOAT || staging.format == DXGI_FORMAT_R16G16B16A16_TYPELESS)
        copy_rows(g_outputRows, destination, destinationPitch, rows, host.rowBytes, static_cast<size_t>(width) * 8,
                  height);
    else
        g_outputRows.run(height, host.rowBytes * height, [&](UINT begin, UINT end) {
            for (UINT y = begin; y < end; ++y)
                convert_row_out(staging.format, reinterpret_cast<const uint16_t*>(rows + host.rowBytes * y),
                                destination + y * destinationPitch, width);
        });
    if (timing != nullptr)
        timing->outputConvert = profile_ms(conversionStart, ProfileClock::now());
}

ID3D12Resource* get_resource(void* parameters, const char* name)
{
    ID3D12Resource* resource = nullptr;
    if (d4r_ngx_get_d3d12_resource(parameters, name, &resource) == NGX_SUCCESS && resource != nullptr)
        return resource;
    void* raw = nullptr;
    if (d4r_ngx_get_void(parameters, name, &raw) == NGX_SUCCESS)
        return static_cast<ID3D12Resource*>(raw);
    return nullptr;
}

// --- DLSSD.OutputAlpha: the denoiser's own output ---------------------------------
//
// nvsdk_ngx_defs_dlssd.h:98 makes this an OUTPUT the denoiser writes, not another G-buffer
// input, and it carries its own Subrect base (101/102), so it has to come back to the caller
// rather than be read from it.
//
// The surface is float32, single channel: the vendor's transformer output kernel stores this
// value as a nullable scalar float32 pixel (sust.p.2d.b32.zero against param+248), so the
// surface object is created over a 1-channel float array and converted once on the way out.
//
// Everything else is borrowed from the colour output's proven lifecycle - a ring of slots, a
// claim so a staged result is not overwritten while a recorded command list still reads it,
// and the result presented a frame later - rather than being a second, asymmetric mechanism.

// The single-channel formats a caller may register, and how to write each back.

static bool alpha_output_layout(DXGI_FORMAT format, uint32_t& writeback)
{
    switch (format)
    {
    case DXGI_FORMAT_R8_UNORM: writeback = 1; return true;
    case DXGI_FORMAT_R8_UINT: writeback = 2; return true;
    case DXGI_FORMAT_R16_FLOAT: writeback = 3; return true;
    case DXGI_FORMAT_R16_UNORM: writeback = 4; return true;
    case DXGI_FORMAT_R16_UINT: writeback = 5; return true;
    case DXGI_FORMAT_R32_FLOAT: writeback = 6; return true;
    default: return false;
    }
}

static size_t alpha_texel_bytes(uint32_t writeback)
{
    return writeback <= 2 ? 1 : (writeback <= 5 ? 2 : 4);
}

static void write_alpha_texel(uint32_t writeback, uint8_t* destination, float value)
{
    switch (writeback)
    {
    case 1:
    {
        const uint32_t v = to_unorm(value, 8);
        std::memcpy(destination, &v, 1);
        return;
    }
    case 2:
    {
        const uint8_t v = value <= 0.0f ? 0u : (value >= 255.0f ? 255u : static_cast<uint8_t>(value));
        std::memcpy(destination, &v, sizeof(v));
        return;
    }
    case 3:
    {
        const uint16_t v = float_to_half(value);
        std::memcpy(destination, &v, sizeof(v));
        return;
    }
    case 4:
    {
        const uint16_t v = static_cast<uint16_t>(to_unorm(value, 16));
        std::memcpy(destination, &v, sizeof(v));
        return;
    }
    case 5:
    {
        const uint32_t v = value <= 0.0f ? 0u : (value >= 65535.0f ? 65535u : static_cast<uint32_t>(value));
        const uint16_t narrow = static_cast<uint16_t>(v);
        std::memcpy(destination, &narrow, sizeof(narrow));
        return;
    }
    default:
    {
        std::memcpy(destination, &value, sizeof(value));
        return;
    }
    }
}

// --- GPU-resident alpha -----------------------------------------------------------
//
// Separate alpha no longer forces RGB through host memory. R32F is copied
// verbatim through an imported result buffer; half/UNORM destinations use a
// Vulkan blit. Integer destinations retain the existing host conversion.
// Blit rounding is hardware-dependent: gfx1201 rounds half toward zero, and
// half/UNORM output can differ from the CPU quantizer by one destination step.

// The destination image's Vulkan format, and whether reaching it from the float32 result needs a
// conversion at all (an R32F destination is copied verbatim).

static bool alpha_vram_format(DXGI_FORMAT format, VkFormat& vulkan, bool& convert)
{
    switch (format)
    {
    case DXGI_FORMAT_R32_FLOAT:
        vulkan = VK_FORMAT_R32_SFLOAT;
        convert = false;
        return true;
    case DXGI_FORMAT_R16_FLOAT:
        vulkan = VK_FORMAT_R16_SFLOAT;
        convert = true;
        return true;
    case DXGI_FORMAT_R16_UNORM:
        vulkan = VK_FORMAT_R16_UNORM;
        convert = true;
        return true;
    case DXGI_FORMAT_R8_UNORM:
        vulkan = VK_FORMAT_R8_UNORM;
        convert = true;
        return true;
    // R8_UINT and R16_UINT deliberately do not qualify. Vulkan will not blit a float image into an
    // integer one, and the host conversion of those is not a value conversion at all: it truncates
    // the float toward zero (write_alpha_texel, cases 2 and 5), so an alpha of 3.9 must land on 3,
    // not on the 0 a normalized blit would give. Only the host route can express that.
    default:
        return false;
    }
}

// The R32F image a converted destination is blitted from is an ordinary conversion image, shared
// with the rest of the file only in kind: one per alpha result slot, since a recorded copy reads
// it a frame or more later.

// Game thread. Decides whether this frame's alpha result stays on the GPU: the colour output's
// VRAM interop has to be on, the caller's texture has to support the conversion,
// and every result slot needs a buffer (and, for a converted format, an R32F image)
// big enough for the evaluated region. Anything short of that uses the host route,
// which needs none of it, so a caller with an exotic alpha format still gets its alpha.

void prepare_alpha_vram(Feature& feature, InputSlot& slot, bool vram)
{
    slot.alphaImageHandle = VK_NULL_HANDLE;
    slot.alphaConvert = false;
    slot.alphaGpu = false;
    if (!slot.alphaRequested || !vram || g_vk.interop == nullptr)
        return;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool convert = false;
    if (!alpha_vram_format(slot.alphaFormat, format, convert) ||
        !vram_blit_supported(VK_FORMAT_R32_SFLOAT, format))
        return;
    UINT64 handle = 0, offset = 0;
    VkFormat actual = VK_FORMAT_UNDEFINED;
    if (FAILED(g_vk.interop->GetVulkanResourceInfo1(slot.alphaResource, &handle, &offset, &actual)) ||
        handle == 0 || actual != format)
        return;
    // Only the evaluated region travels, not the whole extent the denoiser stores into: the result
    // is read back as the caller's rectangle and blitted into the caller's rectangle.
    const size_t bytes = sizeof(float) * static_cast<size_t>(slot.alphaExtentW) * slot.alphaExtentH;
    for (AlphaSlot& alpha : feature.alpha)
    {
        if (!ensure_vram_buffer(feature, alpha.vram, bytes))
        {
            logf("DLSSD.OutputAlpha: no VRAM result buffer of %zu bytes; this frame's alpha is read back "
                 "through host memory",
                 bytes);
            return;
        }
        if (convert &&
            !ensure_conversion_image(feature, alpha.conversion, slot.alphaExtentW, slot.alphaExtentH,
                                     VK_FORMAT_R32_SFLOAT))
        {
            logf("DLSSD.OutputAlpha: no R32F conversion image of %ux%u; this frame's alpha is read back "
                 "through host memory",
                 slot.alphaExtentW, slot.alphaExtentH);
            return;
        }
    }
    slot.alphaImageHandle = reinterpret_cast<VkImage>(handle);
    slot.alphaConvert = convert;
    slot.alphaGpu = true;
}

// Records the copy of a GPU-resident alpha result into the caller's texture, which is in COPY_DEST
// state already. The result buffer holds the evaluated region as float32, so an R32F destination is
// copied verbatim at the caller's subrect origin and a converted one is blitted there.

static bool record_alpha_vram(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, VkImage image,
                              const VramBuffer& buffer, const VramImage& conversion, const AlphaRegion& region,
                              bool convert)
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &cmd)))
        return false;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    g_vk.interop->GetVulkanImageLayout(resource, D3D12_RESOURCE_STATE_COPY_DEST, &layout);
    VkBufferImageCopy regionCopy = {};
    regionCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    regionCopy.imageSubresource.layerCount = 1;
    regionCopy.imageOffset = {static_cast<int32_t>(region.baseX), static_cast<int32_t>(region.baseY), 0};
    regionCopy.imageExtent = {region.width, region.height, 1};
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &kBeforeTransfer, 0,
                 nullptr, 0, nullptr);
    if (!convert)
        g_vk.copyBufferToImage(cmd, buffer.buffer, image, layout, 1, &regionCopy);
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
        regionCopy.imageOffset = {0, 0, 0};
        g_vk.copyBufferToImage(cmd, buffer.buffer, conversion.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                              &regionCopy);
        VkImageMemoryBarrier toSource = toDestination;
        toSource.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSource.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                     &toSource);
        // Source offsets are the region's own origin, destination offsets the caller's subrect, so
        // every texel of the caller's texture outside the rectangle is left as it was.
        VkImageBlit blit = {};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstSubresource = blit.srcSubresource;
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {static_cast<int32_t>(region.width), static_cast<int32_t>(region.height), 1};
        blit.dstOffsets[0] = {static_cast<int32_t>(region.baseX), static_cast<int32_t>(region.baseY), 0};
        blit.dstOffsets[1] = {static_cast<int32_t>(region.baseX + region.width),
                               static_cast<int32_t>(region.baseY + region.height), 1};
        g_vk.blit(cmd, conversion.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, layout, 1, &blit,
                  VK_FILTER_NEAREST);
    }
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &kAfterTransfer, 0,
                 nullptr, 0, nullptr);
    return SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
}

// Game thread. Records which DLSSD.OutputAlpha texture this frame wants written, and holds a
// reference on it until the frame retires. Refuses loudly a texture this file cannot fill:
// accepting the frame and leaving the caller's surface untouched is exactly the silent wrong
// answer this whole route exists to avoid. Sets nothing on the CUDA parameter object - that is
// the worker's, and this runs on the game thread.

bool capture_alpha_output(Feature& feature, void* parameters, int slotIndex, uint32_t frame)
{
    InputSlot& slot = feature.inputs[slotIndex];
    ID3D12Resource* const resource = get_resource(parameters, "DLSSD.OutputAlpha");
    if (resource == nullptr)
    {
        slot.alphaRequested = false;
        return true;
    }
    D3D12_RESOURCE_DESC desc;
    resource->GetDesc(&desc);
    uint32_t writeback = 0;
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.DepthOrArraySize != 1 ||
        desc.Width == 0 || desc.Height == 0 || desc.Width > 0xFFFFFFFFull || desc.SampleDesc.Count != 1 ||
        !alpha_output_layout(desc.Format, writeback))
    {
        logf("Ray Reconstruction: DLSSD.OutputAlpha is %s %llux%u array=%u samples=%u fmt=0x%x, which this "
             "file cannot write back; this frame's evaluation is refused rather than accepted with the "
             "texture left untouched",
             desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D ? "a texture" : "not a texture", desc.Width,
             desc.Height, desc.DepthOrArraySize, desc.SampleDesc.Count, desc.Format);
        return false;
    }
    // The caller's Subrect. The surface is allocated at the texture's FULL extent and the base
    // is forwarded, so the denoiser stores where the caller expects; only the evaluated region
    // is copied back, leaving the caller's pixels outside it untouched.
    const UINT baseX = get_uint_or(parameters, "DLSSD.OutputAlpha.Subrect.Base.X", 0);
    const UINT baseY = get_uint_or(parameters, "DLSSD.OutputAlpha.Subrect.Base.Y", 0);
    // Guard the subtractions themselves: a base outside the texture underflows below.
    if (baseX > desc.Width || baseY > desc.Height)
    {
        logf("Ray Reconstruction: DLSSD.OutputAlpha subrect origin (%u,%u) is outside its %llux%u texture; "
             "this frame's evaluation is refused",
             baseX, baseY, desc.Width, desc.Height);
        return false;
    }
    // The evaluated region is the output size placed at that origin, and it must fit; it is not
    // clamped silently into whatever happens to fit.
    const UINT remainingW = static_cast<UINT>(desc.Width) - baseX;
    const UINT remainingH = desc.Height - baseY;
    UINT extentW = get_uint_or(parameters, "OutWidth", feature.outWidth);
    UINT extentH = get_uint_or(parameters, "OutHeight", feature.outHeight);
    if (extentW == 0 || extentW > remainingW || extentH == 0 || extentH > remainingH)
    {
        logf("Ray Reconstruction: DLSSD.OutputAlpha needs a %ux%u region at (%u,%u) inside a %llux%u texture, "
             "which does not fit; this frame's evaluation is refused",
             extentW, extentH, baseX, baseY, desc.Width, desc.Height);
        return false;
    }
    if (slot.alphaResource != nullptr && slot.alphaResource != resource)
    {
        slot.alphaResource->Release();
        slot.alphaResource = nullptr; // so the new resource is always AddRef'd below
    }
    if (slot.alphaResource == nullptr)
        resource->AddRef();
    slot.alphaResource = resource;
    slot.alphaFormat = desc.Format;
    slot.alphaWidth = static_cast<UINT>(desc.Width); // full extent: where the vendor stores
    slot.alphaHeight = desc.Height;
    slot.alphaBaseX = baseX;
    slot.alphaBaseY = baseY;
    slot.alphaExtentW = extentW;                    // what is actually copied back
    slot.alphaExtentH = extentH;
    slot.alphaWriteback = writeback;
    slot.alphaRequested = true;
    if (frame <= 2)
        logf("Ray Reconstruction: DLSSD.OutputAlpha %ux%u fmt=0x%x, subrect (%u,%u) region %ux%u", desc.Width,
             desc.Height, desc.Format, baseX, baseY, extentW, extentH);
    return true;
}

// Worker. Gives the denoiser a float32 single-channel CUDA surface object for this frame's
// alpha and forwards the subrect origin, so its stores land where the caller expects. Clearing
// the parameter when no texture was registered keeps the denoiser from writing into a surface
// belonging to a frame that no longer wants one; the image itself is kept, since dropping it
// would mean a fresh GPU allocation on the next frame that does ask.

bool publish_alpha_output(Feature& feature, const InputSlot& slot)
{
    if (!slot.alphaRequested)
    {
        d4r_ngx_set_void(feature.cudaParams, "DLSSD.OutputAlpha", nullptr);
        return true;
    }
    if (!ensure_cuda_image(feature.alphaImage, slot.alphaWidth, slot.alphaHeight, CUDA_FORMAT_FLOAT, 1, true, false))
        return false;
    d4r_ngx_set_void(feature.cudaParams, "DLSSD.OutputAlpha", &feature.alphaImage.object);
    d4r_ngx_set_uint(feature.cudaParams, "DLSSD.OutputAlpha.Subrect.Base.X", slot.alphaBaseX);
    d4r_ngx_set_uint(feature.cudaParams, "DLSSD.OutputAlpha.Subrect.Base.Y", slot.alphaBaseY);
    return true;
}

// Worker. The colour output's exact claim: the GPU marker is the only thing that proves a
// recorded copy has run, so a slot is reusable only once the GPU is past the frame that last
// read it, and the newest result is never taken from under a reader.

static int claim_alpha_slot(Feature& feature)
{
    const auto start = std::chrono::steady_clock::now();
    for (;;)
    {
        {
            std::lock_guard<std::mutex> lock(feature.outputMutex);
            const uint32_t gpuFrame = *feature.markerValue;
            for (int index = 0; index < kOutputSlots; ++index)
            {
                if (index == feature.latestAlpha.load())
                    continue;
                if (static_cast<int32_t>(gpuFrame - feature.alpha[index].lastReadFrame.load()) > 0)
                {
                    feature.alpha[index].lastReadFrame = 0x7fffffffu + gpuFrame;
                    return index;
                }
            }
        }
        if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(50))
            return -1;
        d4r_sleep_us(200);
    }
}

// D4R_SHIM_EVAL_SYNC: whether the worker waits for NGX's kernels before it reads a result back.
// With VRAM interop the wait is deferred to publish_vram, which releases the colour result only
// after it - so anything that reads the arrays earlier, such as the alpha copy below, has to
// bring the wait forward for itself.

bool eval_sync_enabled()
{
    static const bool sync = env_uint("D4R_SHIM_EVAL_SYNC", 1) != 0;
    return sync;
}

// Worker. Reads the surface the denoiser just wrote and puts the evaluated region in the claimed
// slot: straight into that slot's imported buffer when the game thread qualified the GPU route,
// otherwise into an upload buffer as converted texels. A failure fails the frame: an accepted
// frame with a missing alpha would leave the caller believing it was written.

bool download_alpha_output(Feature& feature, uint32_t frame, const FrameParams& params)
{
    const InputSlot& slot = feature.inputs[frame % kSlots];
    if (!slot.alphaRequested)
        return true;
    if (feature.alphaImage.array == nullptr)
        return false;
    // A split frame presents its own result in the second half of its own command list, so it takes
    // the slot that frame's colour takes rather than claiming one: a claim would be free to pick a
    // slot whose reader - frame N's own second half - the GPU marker has passed but whose copy has
    // not run yet, and overwrite it underneath that copy. Frames in flight are capped below
    // kOutputSlots, so frame N's slot is not reachable again until its second half has run.
    const int index = params.split ? static_cast<int>(frame % kOutputSlots) : claim_alpha_slot(feature);
    if (index < 0)
    {
        logf("frame %u: no free DLSSD.OutputAlpha result slot; the evaluation is dropped", frame);
        return false;
    }
    AlphaSlot& target = feature.alpha[index];
    const size_t regionBytes = sizeof(float) * static_cast<size_t>(slot.alphaExtentW) * slot.alphaExtentH;
    // Only the evaluated region moves, from either route: the denoiser stores into the whole
    // extent its surface was created over, and the rest of the caller's texture is not its to
    // touch.
    // `params.vram` is the worker's copy of this frame's decision, because the colour output's
    // interop can be cut back to host staging between the capture and the evaluation.
    // With VRAM interop run_evaluation skipped its own wait (publish_vram takes it before releasing
    // the colour result), but the alpha is read out of that array here, on either route: the GPU
    // one copies it, the host one converts the bytes the moment they land. Bring the wait forward
    // or the denoiser's alpha would be read while it is still being written.
    if (params.vram && !eval_sync_enabled())
    {
        const int synced = synchronize_default_stream(feature, nullptr);
        if (synced != 0)
        {
            logf("frame %u: DLSSD.OutputAlpha copy synchronize failed: %d; the evaluation is dropped", frame,
                 synced);
            return false;
        }
    }
    if (slot.alphaGpu && params.vram && target.vram.device != 0 && target.vram.bytes >= regionBytes)
    {
        CudaMemcpy2D copy = {};
        // The source origin is a byte offset: the surface is float32 single channel, so the
        // caller's subrect origin in pixels is that many bytes into each row. The result buffer
        // holds the region alone, which is why the destination offsets stay at zero.
        copy.srcXInBytes = slot.alphaBaseX * sizeof(float);
        copy.srcY = slot.alphaBaseY;
        copy.srcMemoryType = CUDA_MEMORY_ARRAY;
        copy.srcArray = feature.alphaImage.array;
        copy.srcPitch = sizeof(float) * slot.alphaWidth;
        copy.dstMemoryType = CUDA_MEMORY_DEVICE;
        copy.dstDevice = target.vram.device;
        copy.dstPitch = sizeof(float) * slot.alphaExtentW;
        copy.WidthInBytes = copy.dstPitch;
        copy.Height = slot.alphaExtentH;
        // Queued on the null stream, where the colour result's copy is queued too, and picked up by
        // the Vulkan transfer that blits this buffer into the caller's texture.
        if (copy_2d(copy) != 0)
        {
            logf("frame %u: copying the DLSSD.OutputAlpha result into VRAM failed; the evaluation is dropped",
                 frame);
            return false;
        }
        if (env_uint("D4R_SHIM_ALPHA_VERIFY", 0) != 0)
        {
            // The region as the denoiser left it in its array, against the buffer the Vulkan blit
            // reads it from: the two must be the same float32 the host route would have converted.
            std::vector<uint8_t> fromArray(regionBytes), fromBuffer(regionBytes);
            CudaMemcpy2D readArray = copy;
            readArray.dstMemoryType = CUDA_MEMORY_HOST;
            readArray.dstHost = fromArray.data();
            CudaMemcpy2D readBuffer = {};
            readBuffer.srcMemoryType = CUDA_MEMORY_DEVICE;
            readBuffer.srcDevice = target.vram.device;
            readBuffer.srcPitch = copy.dstPitch;
            readBuffer.dstMemoryType = CUDA_MEMORY_HOST;
            readBuffer.dstHost = fromBuffer.data();
            readBuffer.dstPitch = copy.dstPitch;
            readBuffer.WidthInBytes = copy.WidthInBytes;
            readBuffer.Height = copy.Height;
            if (g.cu.memcpy2D(&readArray) == 0 && g.cu.memcpy2D(&readBuffer) == 0)
            {
                size_t differing = 0;
                for (size_t byte = 0; byte < fromArray.size(); ++byte)
                    differing += fromArray[byte] != fromBuffer[byte];
                if (differing != 0 || frame <= 3 || frame % 120 == 0)
                    logf("frame %u VRAM alpha verify: %zu of %zu bytes differ between the result array and "
                         "the buffer",
                         frame, differing, fromArray.size());
                if (differing != 0)
                    return false;
                char dumpPath[MAX_PATH], alphaPath[MAX_PATH];
                if (output_dump_path(frame, dumpPath))
                {
                    const int written = std::snprintf(alphaPath, sizeof(alphaPath), "%s.alpha.f32", dumpPath);
                    if (written > 0 && written < static_cast<int>(sizeof(alphaPath)))
                        write_capture_file(frame, "raw DLSS alpha region", alphaPath, fromArray.data(),
                                           fromArray.size());
                }
            }
            else
            {
                logf("frame %u: reading DLSSD.OutputAlpha verification data failed", frame);
                return false;
            }
        }
        target.gpu = true;
    }
    else
    {
        target.gpu = false;
        if (!ensure_staging(target.staging, slot.alphaResource, D3D12_HEAP_TYPE_UPLOAD))
        {
            logf("frame %u: no upload buffer for the DLSSD.OutputAlpha result; the evaluation is dropped", frame);
            return false;
        }
        reserve_host(feature.alphaHost, sizeof(float) * slot.alphaWidth * slot.alphaHeight);
        if (feature.alphaHost.bytes == nullptr ||
            feature.alphaHost.capacity < sizeof(float) * slot.alphaWidth * slot.alphaHeight)
        {
            logf("frame %u: no room to read the DLSSD.OutputAlpha surface back; the evaluation is dropped", frame);
            return false;
        }
        CudaMemcpy2D copy = {};
        copy.srcMemoryType = CUDA_MEMORY_ARRAY;
        copy.srcArray = feature.alphaImage.array;
        copy.dstMemoryType = CUDA_MEMORY_HOST;
        copy.dstHost = feature.alphaHost.bytes;
        copy.dstPitch = sizeof(float) * slot.alphaWidth;
        copy.WidthInBytes = sizeof(float) * slot.alphaWidth;
        copy.Height = slot.alphaHeight;
        // Conversion reads the host bytes immediately, so this transfer must complete on return.
        if (g.cu.memcpy2D(&copy) != 0)
        {
            logf("frame %u: reading the DLSSD.OutputAlpha surface back failed; the evaluation is dropped", frame);
            return false;
        }
        // Only the evaluated rectangle is written, at its position in the buffer. The footprint
        // offset from GetCopyableFootprints is left alone - it must stay 512-byte aligned - and the
        // region is instead selected by the source box of the copy below.
        const float* const source = reinterpret_cast<const float*>(feature.alphaHost.bytes);
        uint8_t* const destination = target.staging.mapped + target.staging.layout.Offset;
        const size_t rowPitch = target.staging.layout.Footprint.RowPitch;
        const size_t texel = alpha_texel_bytes(slot.alphaWriteback);
        for (UINT row = 0; row < slot.alphaExtentH; ++row)
            for (UINT x = 0; x < slot.alphaExtentW; ++x)
                write_alpha_texel(
                    slot.alphaWriteback,
                    destination + static_cast<size_t>(slot.alphaBaseY + row) * rowPitch +
                        static_cast<size_t>(slot.alphaBaseX) * texel + x * texel,
                    source[static_cast<size_t>(slot.alphaBaseY + row) * slot.alphaWidth + slot.alphaBaseX + x]);
    }
    {
        // Published under the same lock the claim uses, exactly as the colour output publishes.
        std::lock_guard<std::mutex> lock(feature.outputMutex);
        target.format = slot.alphaFormat;
        target.baseX = slot.alphaBaseX;
        target.baseY = slot.alphaBaseY;
        target.width = slot.alphaExtentW;
        target.height = slot.alphaExtentH;
        target.writeback = slot.alphaWriteback;
        target.lastReadFrame = 0;
        target.producedFrame = frame;
        feature.latestAlpha = index;
    }
    return true;
}

// Game thread. Presents the alpha result that goes with the colour result about to be presented.
// The slot's lastReadFrame is stamped before the copy is recorded and producedFrame is NOT
// cleared: the colour output's claim relies on both, and clearing here would let the worker
// overwrite a result whose command list has not run yet.

bool record_alpha_copyback(Feature& feature, ID3D12GraphicsCommandList* list, InputSlot& slot,
                                  D3D12_RESOURCE_STATES outputState, uint32_t colorFrame, bool split)
{
    if (!slot.alphaRequested || slot.alphaResource == nullptr)
        return true;
    AlphaRegion region = {slot.alphaBaseX, slot.alphaBaseY, slot.alphaExtentW, slot.alphaExtentH};
    int index = -1;
    if (split)
    {
        // A split frame presents its own colour result in the second half of its own command list,
        // so it presents its own alpha result: the very slot the worker is about to fill for this
        // frame, which does not exist yet. The copy is recorded now and runs after the split
        // semaphore, by which time the worker has written it - the same ordering the colour copy
        // above relies on, and the reason the region comes from this frame's capture rather than
        // from a published result.
        index = static_cast<int>(feature.frame % kOutputSlots);
        AlphaSlot& chosen = feature.alpha[index];
        if (slot.alphaGpu)
        {
            if (chosen.vram.buffer == VK_NULL_HANDLE)
                return false;
        }
        else if (!ensure_staging(chosen.staging, slot.alphaResource, D3D12_HEAP_TYPE_UPLOAD))
        {
            logf("frame %u: no upload buffer for the DLSSD.OutputAlpha result", feature.frame);
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(feature.outputMutex);
            chosen.lastReadFrame = feature.frame;
        }
    }
    else
    {
        if (colorFrame == 0)
            return true; // no colour result to pair with this frame
        {
            std::lock_guard<std::mutex> lock(feature.outputMutex);
            // The frame whose colour is about to be presented is `colorFrame`; only an alpha result
            // produced by that very same frame may go with it. Picking "the newest alpha" instead
            // would pair alpha N with colour N-1, since the alpha read-back completes before the
            // colour's, and the caller would composite two different frames. No match, no copy: the
            // colour is presented on its own and the alpha waits for its own frame to come round.
            for (int candidate = 0; candidate < kOutputSlots; ++candidate)
                if (feature.alpha[candidate].producedFrame.load() == colorFrame)
                {
                    index = candidate;
                    break;
                }
            if (index < 0)
                return true; // the matching frame has not been read back yet; colour goes alone
            AlphaSlot& chosen = feature.alpha[index];
            if ((chosen.gpu && chosen.vram.buffer == VK_NULL_HANDLE) ||
                (!chosen.gpu && chosen.staging.buffer == nullptr) || chosen.producedFrame.load() == 0)
                return true;
            // A result the worker left on the GPU can only be presented by a frame that qualified
            // the GPU route itself: that frame's capture is what recorded the caller's image and
            // whether its format needs the conversion blit. Anything else discards the result,
            // exactly as a result whose rectangle no longer matches is discarded below.
            if (chosen.gpu && !slot.alphaGpu)
            {
                chosen.producedFrame = 0;
                return true;
            }
            if (chosen.width != slot.alphaExtentW || chosen.height != slot.alphaExtentH ||
                chosen.format != slot.alphaFormat || chosen.baseX != slot.alphaBaseX ||
                chosen.baseY != slot.alphaBaseY)
            {
                // The caller's rectangle changed under a result already in flight; presenting it
                // would write where the caller no longer expects. Leave it for the next result.
                chosen.producedFrame = 0;
                return true;
            }
            chosen.lastReadFrame = feature.frame;
            region = {chosen.baseX, chosen.baseY, chosen.width, chosen.height};
        }
    }
    AlphaSlot& chosen = feature.alpha[index];
    const bool gpu = split ? slot.alphaGpu : chosen.gpu;
    transition(list, slot.alphaResource, outputState, D3D12_RESOURCE_STATE_COPY_DEST);
    if (gpu)
    {
        // The result is still the float32 the denoiser wrote, so it becomes the caller's texels
        // here rather than on the CPU. The region lands at the caller's subrect, and the
        // transitions around it are the same ones the host copy below records.
        if (!record_alpha_vram(list, slot.alphaResource, slot.alphaImageHandle, chosen.vram, chosen.conversion,
                               region, slot.alphaConvert))
        {
            transition(list, slot.alphaResource, D3D12_RESOURCE_STATE_COPY_DEST, outputState);
            logf("frame %u: recording DLSSD.OutputAlpha GPU copy failed", feature.frame);
            return false;
        }
    }
    else
    {
        D3D12_TEXTURE_COPY_LOCATION destination = {};
        destination.pResource = slot.alphaResource;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION source = {};
        source.pResource = chosen.staging.buffer;
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = chosen.staging.layout;
        // The source box picks out the evaluated rectangle inside the full-extent staging buffer,
        // and the destination origin places it at the caller's subrect. Together they leave every
        // pixel of the caller's texture outside the rectangle untouched.
        D3D12_BOX box = {};
        box.left = region.baseX;
        box.top = region.baseY;
        box.right = region.baseX + region.width;
        box.bottom = region.baseY + region.height;
        box.back = 1;
        list->CopyTextureRegion(&destination, region.baseX, region.baseY, 0, &source, &box);
    }
    transition(list, slot.alphaResource, D3D12_RESOURCE_STATE_COPY_DEST, outputState);
    return true;
}
