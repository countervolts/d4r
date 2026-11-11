// d4r_nvngx_shim_pipeline.cpp: one evaluation from end to end. Moves the auxiliary surfaces to CUDA,
// runs NGX on the worker thread, and publishes the result back to the game's command list.

// the command-list stubs live in d4r_nvngx_shim.cpp, which assembles them once
#define D4R_D3D12_SPLIT_DECL 1
#include "d4r_nvngx_shim_internal.h"

static AuxSurface* find_aux_surface(Feature& feature, ID3D12Resource* resource)
{
    for (AuxSurface& surface : feature.aux)
        if (surface.resource == resource)
            return &surface;
    return nullptr;
}

static std::string join_aux_names(const std::vector<const char*>& names)
{
    std::string joined;
    for (const char* name : names)
        joined += std::string(joined.empty() ? "" : ", ") + name;
    return joined;
}

// Worker. Frees the CUDA objects of a surface and the caller's reference to its resource.
// Only ever called once no queued frame names it any more.

void destroy_aux_surface(AuxSurface& surface)
{
    if (surface.image.object != 0)
        g.cu.texObjectDestroy(surface.image.object);
    if (surface.image.array != nullptr)
        g.cu.arrayDestroy(surface.image.array);
    for (Staging& staging : surface.staging)
        staging.release();
    for (HostPlane& host : surface.host)
        release_host(host);
    for (VramBuffer& buffer : surface.vram)
    {
        if (buffer.buffer == VK_NULL_HANDLE)
            continue;
        // Retirement can run on the finish thread. Release imports on the CUDA worker
        // without a nested worker.call when retirement itself runs there.
        g.worker.post([buffer] {
            if (buffer.external != nullptr)
            {
                g.cu.memFree(buffer.device);
                g_vk.release(buffer.external);
            }
            g_vk.destroyBuffer(g_vk.device, buffer.buffer, nullptr);
            g_vk.free(g_vk.device, buffer.memory, nullptr);
        });
        buffer = {};
    }
    destroy_vram_image(surface.conversion);
    if (surface.resource != nullptr)
        surface.resource->Release();
    logf("Ray Reconstruction: released denoiser input %s (%ux%u fmt=0x%x)", join_aux_names(surface.publish).c_str(),
         surface.width, surface.height, surface.format);
}

// Vulkan blits cannot convert integer images to float. Raw UINT/SINT guides
// retain host staging; other formats preserve the host path's sampled values.

static bool describe_aux_vram(AuxSurface& surface, VramCopy& copy)
{
    if (!vram_interop_available() || env_uint("D4R_SHIM_RR_VRAM_GUIDES", 1) == 0)
        return false;
    UINT64 handle = 0, offset = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    if (FAILED(g_vk.interop->GetVulkanResourceInfo1(surface.resource, &handle, &offset, &format)) || handle == 0)
        return false;
    copy.resource = surface.resource;
    copy.image = reinterpret_cast<VkImage>(handle);
    copy.width = surface.width;
    copy.height = surface.height;
    if (surface.how == AuxLayout::Verbatim)
        return true;
    if (surface.format != DXGI_FORMAT_B8G8R8A8_UNORM &&
        surface.format != DXGI_FORMAT_R10G10B10A2_UNORM &&
        surface.format != DXGI_FORMAT_R11G11B10_FLOAT)
        return false;
    copy.convert = true;
    return vram_blit_supported(format, VK_FORMAT_R32G32B32A32_SFLOAT);
}

static bool record_aux_vram(ID3D12GraphicsCommandList* list, AuxSurface& surface, int slotIndex,
                            const VramCopy& copy, D3D12_RESOURCE_STATES state)
{
    const size_t bytes = aux_texel_bytes(surface.arrayFormat, surface.channels) * surface.width * surface.height;
    VramBuffer& buffer = surface.vram[slotIndex];
    // Resource geometry is immutable. Never worker.call under auxMutex: the
    // worker could be waiting for this lock to upload an earlier frame.
    if (buffer.buffer == VK_NULL_HANDLE && !create_vram_buffer(buffer, bytes, false))
        return false;
    if (copy.convert && surface.conversion.image == VK_NULL_HANDLE &&
        !create_vram_image(surface.conversion, surface.width, surface.height, VK_FORMAT_R32G32B32A32_SFLOAT))
        return false;
    transition(list, copy.resource, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (FAILED(g_vk.interop->BeginVkCommandBufferInterop(list, &cmd)))
    {
        transition(list, copy.resource, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        return false;
    }
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                 &kBeforeTransfer, 0, nullptr, 0, nullptr);
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    g_vk.interop->GetVulkanImageLayout(copy.resource, D3D12_RESOURCE_STATE_COPY_SOURCE, &layout);
    VkImage image = copy.image;
    if (copy.convert)
    {
        image = surface.conversion.image;
        VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                     0, nullptr, 1, &barrier);
        VkImageBlit blit = {};
        blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = blit.dstOffsets[1] =
            {static_cast<int32_t>(copy.width), static_cast<int32_t>(copy.height), 1};
        g_vk.blit(cmd, copy.image, layout, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                     0, nullptr, 1, &barrier);
        layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    const VkBufferImageCopy region = full_region(copy);
    g_vk.copyImageToBuffer(cmd, image, layout, buffer.buffer, 1, &region);
    g_vk.barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                 &kAfterTransfer, 0, nullptr, 0, nullptr);
    const bool recorded = SUCCEEDED(g_vk.interop->EndVkCommandBufferInterop(list));
    transition(list, copy.resource, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    return recorded;
}

// Game thread. Finds the auxiliary surfaces the caller registered this frame and records a
// readback copy of each into this frame's slot, ahead of the frame marker so the worker can
// read them as soon as that marker is in. The result is this slot's binding snapshot: the
// worker publishes exactly these, so a surface the caller did not register this frame is
// neither sampled nor named, whatever it did last frame.
//
// A texture the caller did supply but this file cannot hand to the denoiser (a shape or a
// format with no CUDA layout) fails the evaluation. Silently omitting a guide the denoiser
// expects is exactly the failure that produces a plausible-looking wrong image, so the frame
// is refused and the reason named instead.

bool gather_aux_inputs(Feature& feature, ID3D12GraphicsCommandList* list, void* parameters, int slotIndex,
                              D3D12_RESOURCE_STATES inputState, uint32_t frame)
{
    InputSlot& slot = feature.inputs[slotIndex];
    std::lock_guard<std::mutex> lock(feature.auxMutex);
    // A frame the input dump selected records this frame's guides too, so a replay of an RR
    // sequence feeds the denoiser the game's own guides rather than the harness's synthetic
    // ones - without which the replayed image cannot be compared with the game's.
    char captureDirectory[MAX_PATH];
    const bool capture = input_dump_directory(frame, captureDirectory);
    std::vector<AuxBinding> bindings;
    for (const AuxInput& input : kAuxInputs)
    {
        ID3D12Resource* const resource = get_resource(parameters, input.source);
        if (resource == nullptr)
            continue;
        AuxSurface* surface = find_aux_surface(feature, resource);
        if (surface == nullptr)
        {
            feature.aux.emplace_back();
            surface = &feature.aux.back();
            surface->resource = resource;
            resource->AddRef(); // held until every frame naming it has retired
            logf("Ray Reconstruction: denoiser input %s is resource %p", input.source, static_cast<void*>(resource));
        }
        surface->dropping = false;
        surface->lastSeen = frame;
        // One binding per name: two names for the same texture are two denoiser parameters.
        AuxBinding* binding = nullptr;
        for (AuxBinding& candidate : bindings)
            if (candidate.surface == surface)
            {
                binding = &candidate;
                break;
            }
        const bool firstBinding = binding == nullptr;
        if (binding == nullptr)
        {
            bindings.emplace_back();
            binding = &bindings.back();
            binding->surface = surface;
            binding->frame = frame;
        }
        bool listed = false;
        for (const char* name : surface->publish)
            listed = listed || strcmp(name, input.publish) == 0;
        if (!listed)
            surface->publish.push_back(input.publish);
        bool bound = false;
        for (const char* name : binding->names)
            bound = bound || strcmp(name, input.publish) == 0;
        if (!bound)
            binding->names.push_back(input.publish);
        bool sourceBound = false;
        for (const char* name : binding->sources)
            sourceBound = sourceBound || strcmp(name, input.source) == 0;
        if (!sourceBound)
            binding->sources.push_back(input.source);
        if (!firstBinding)
            continue; // aliases share one captured plane and one upload

        D3D12_RESOURCE_DESC desc;
        resource->GetDesc(&desc);
        uint32_t arrayFormat = 0, channels = 0, texelBytes = 0;
        AuxLayout how = AuxLayout::Verbatim;
        if (desc.DepthOrArraySize != 1 || desc.Width > 0xFFFFFFFFull ||
            !aux_array_layout(desc.Format, arrayFormat, channels, how, texelBytes))
        {
            logf("Ray Reconstruction: denoiser input %s is %llux%u array=%u fmt=0x%x, which has no single-plane "
                 "CUDA layout; this frame's evaluation is refused rather than run without it",
                 join_aux_names(binding->names).c_str(), desc.Width, desc.Height, desc.DepthOrArraySize, desc.Format);
            return false;
        }
        surface->format = desc.Format;
        surface->arrayFormat = arrayFormat;
        surface->channels = channels;
        surface->how = how;
        surface->texelBytes = texelBytes;
        surface->width = static_cast<UINT>(desc.Width);
        surface->height = static_cast<UINT>(desc.Height);
        VramCopy gpuCopy;
        surface->gpu[slotIndex] = describe_aux_vram(*surface, gpuCopy);
        if (surface->gpu[slotIndex])
        {
            if (!record_aux_vram(list, *surface, slotIndex, gpuCopy, inputState))
                return false;
        }
        // A captured frame reads its guides back even with VRAM interop on: the capture
        // records the caller's own bytes in the caller's own format, which is what a replay
        // has to feed back for the same conversion to run over it.
        if (!surface->gpu[slotIndex] || env_uint("D4R_SHIM_VRAM_VERIFY", 0) != 0 || capture)
        {
            if (!ensure_staging(surface->staging[slotIndex], resource, D3D12_HEAP_TYPE_READBACK))
                return false;
            copy_to_staging(list, resource, surface->staging[slotIndex], inputState);
        }
    }
    // Drop surfaces the caller stopped registering, but only once no queued frame names one:
    // erasing a surface whose handle is still in a parameter object would leave the denoiser
    // sampling a freed texture object.
    for (AuxSurface& surface : feature.aux)
    {
        if (surface.holds != 0 || frame - surface.lastSeen < kAuxStaleFrames)
            continue;
        surface.dropping = true;
        logf("Ray Reconstruction: denoiser input %s was not registered for %u frames",
             join_aux_names(surface.publish).c_str(), kAuxStaleFrames);
    }
    // Committed only now. An earlier failure path returns from this function with `bindings`
    // half-built, and a hold taken there would never be given back - the surface could then
    // never be dropped, and its reference to the caller's resource would never be released.
    for (const AuxBinding& binding : bindings)
        ++binding.surface->holds; // given back by retire_aux_bindings when this frame passes
    slot.auxBindings = std::move(bindings);
    return true;
}

// Worker. Copies this frame's bound surfaces into their CUDA arrays verbatim - same bytes,
// same row order, only the destination is the GPU - and mints the texture objects the
// denoiser samples. A failure fails the evaluation: a denoiser handed fewer guides than the
// caller registered produces a wrong image, not a visible error.

static bool upload_aux_inputs(Feature& feature, int slotIndex, uint32_t frame)
{
    for (const AuxBinding& binding : feature.inputs[slotIndex].auxBindings)
    {
        AuxSurface& surface = *binding.surface;
        const Staging& staging = surface.staging[slotIndex];
        if (!ensure_cuda_image(surface.image, surface.width, surface.height, surface.arrayFormat,
                               surface.channels, false, false))
        {
            logf("frame %u: Ray Reconstruction input %s: a CUDA texture object over %ux%u fmt=0x%x could not be "
                 "created",
                 frame, join_aux_names(binding.names).c_str(), surface.width, surface.height, surface.format);
            return false;
        }
        CudaMemcpy2D copy = {};
        if (surface.gpu[slotIndex])
        {
            VramBuffer& buffer = surface.vram[slotIndex];
            if (buffer.external == nullptr)
            {
                VkMemoryRequirements requirements = {};
                g_vk.bufferRequirements(g_vk.device, buffer.buffer, &requirements);
                const int result = g_vk.import(g_vk.device, reinterpret_cast<uint64_t>(buffer.memory),
                                               requirements.size, &buffer.device, &buffer.external);
                if (result != 0)
                {
                    logf("frame %u: Ray Reconstruction guide import failed: %d", frame, result);
                    return false;
                }
            }
            copy.srcMemoryType = CUDA_MEMORY_DEVICE;
            copy.srcDevice = buffer.device;
            copy.srcPitch = aux_texel_bytes(surface.arrayFormat, surface.channels) * surface.width;
        }
        else if (surface.how == AuxLayout::Verbatim)
        {
            copy.srcMemoryType = CUDA_MEMORY_HOST;
            // Use HostPlane staging: direct ROCm image copies from Vulkan-mapped memory
            // truncate guide rows on gfx1201, even when the source footprint is tightly packed.
            HostPlane& host = surface.host[slotIndex];
            const size_t rowBytes = aux_texel_bytes(surface.arrayFormat, surface.channels) * surface.width;
            reserve_host(host, rowBytes * surface.height);
            if (host.bytes == nullptr || host.capacity < rowBytes * surface.height)
                return false;
            host.width = surface.width;
            host.height = surface.height;
            host.rowBytes = rowBytes;
            for (UINT row = 0; row < surface.height; ++row)
                memcpy(host.bytes + static_cast<size_t>(row) * rowBytes,
                       staging.mapped + staging.layout.Offset +
                           static_cast<size_t>(row) * staging.layout.Footprint.RowPitch, rowBytes);
            copy.srcHost = host.bytes;
            copy.srcPitch = rowBytes;
        }
        else
        {
            copy.srcMemoryType = CUDA_MEMORY_HOST;
            // Packed, integer and BGRA inputs are unpacked into floats first: a CUDA array
            // has no three-channel layout, a uint array is sampled normalized, and BGRA bytes
            // are in the wrong order for a read as RGBA.
            HostPlane& host = surface.host[slotIndex];
            const size_t rowBytes = sizeof(float) * surface.channels * surface.width;
            reserve_host(host, rowBytes * surface.height);
            if (host.bytes == nullptr || host.capacity < rowBytes * surface.height)
            {
                logf("frame %u: Ray Reconstruction input %s: no room for %zu unpacked bytes",
                     frame, join_aux_names(binding.names).c_str(), rowBytes * surface.height);
                return false;
            }
            host.width = surface.width;
            host.height = surface.height;
            host.rowBytes = rowBytes;
            const uint8_t* const base = staging.mapped + staging.layout.Offset;
            for (UINT row = 0; row < surface.height; ++row)
                convert_aux_row(surface.format, base + static_cast<size_t>(row) * staging.layout.Footprint.RowPitch,
                                reinterpret_cast<float*>(host.bytes + static_cast<size_t>(row) * rowBytes),
                                surface.width, surface.channels);
            copy.srcHost = host.bytes;
            copy.srcPitch = rowBytes;
        }
        copy.dstMemoryType = CUDA_MEMORY_ARRAY;
        copy.dstArray = surface.image.array;
        copy.WidthInBytes = aux_texel_bytes(surface.arrayFormat, surface.channels) * surface.width;
        copy.Height = surface.height;
        if (copy_2d(copy) != 0)
        {
            logf("frame %u: Ray Reconstruction input %s: the upload of %ux%u failed", frame,
                 join_aux_names(binding.names).c_str(), surface.width, surface.height);
            return false;
        }
        if (surface.gpu[slotIndex] && env_uint("D4R_SHIM_VRAM_VERIFY", 0) != 0)
        {
            // Compare the array NGX actually samples, not just the imported buffer.
            const size_t rowBytes = copy.WidthInBytes;
            std::vector<uint8_t> expected(rowBytes * surface.height), actual(expected.size());
            for (UINT row = 0; row < surface.height; ++row)
            {
                const uint8_t* source = staging.mapped + staging.layout.Offset +
                                        static_cast<size_t>(row) * staging.layout.Footprint.RowPitch;
                uint8_t* destination = expected.data() + static_cast<size_t>(row) * rowBytes;
                if (surface.how == AuxLayout::Verbatim)
                    memcpy(destination, source, rowBytes);
                else
                    convert_aux_row(surface.format, source, reinterpret_cast<float*>(destination),
                                    surface.width, surface.channels);
            }
            CudaMemcpy2D read = {};
            read.srcMemoryType = CUDA_MEMORY_ARRAY;
            read.srcArray = surface.image.array;
            read.dstMemoryType = CUDA_MEMORY_HOST;
            read.dstHost = actual.data();
            read.dstPitch = read.WidthInBytes = rowBytes;
            read.Height = surface.height;
            if (g.cu.ctxSynchronize() != 0 || g.cu.memcpy2D(&read) != 0)
                return false;
            size_t differing = 0;
            if (surface.how == AuxLayout::Verbatim)
            {
                for (size_t byte = 0; byte < expected.size(); ++byte)
                    differing += actual[byte] != expected[byte];
            }
            else
            {
                const float* a = reinterpret_cast<const float*>(actual.data());
                const float* e = reinterpret_cast<const float*>(expected.data());
                for (size_t i = 0; i < expected.size() / sizeof(float); ++i)
                    differing += !(a[i] == e[i] || (std::isnan(a[i]) && std::isnan(e[i])) ||
                                   std::abs(a[i] - e[i]) <= 1e-6f * std::max(1.0f, std::abs(e[i])));
            }
            logf("frame %u RR VRAM verify %s: %zu differing %s", frame,
                 join_aux_names(binding.names).c_str(), differing,
                 surface.how == AuxLayout::Verbatim ? "bytes" : "components");
            if (differing != 0)
                return false;
        }
        surface.handle = surface.image.object;
    }
    return true;
}

// Worker. Writes this frame's Ray Reconstruction guides next to the canonical input planes:
// the caller's own bytes in the caller's own format, the descriptor each was read with, and
// the per-frame settings, subrect origins and camera matrices the denoiser was given. A replay
// rebuilds the textures from those descriptors and uploads the recorded bytes, so the same
// conversion runs over the same data and the replayed image is comparable with the game's -
// which synthetic guides can never be.
//
// A guide whose bytes are not readable fails the whole capture: a descriptor without its
// plane, or a plane the replay silently skipped, is an image the denoiser would produce from
// fewer guides than the game gave it, and nothing downstream could tell.

static void dump_aux_guides(Feature& feature, int slotIndex, uint32_t frame, const FrameParams& p)
{
    char directory[MAX_PATH];
    if (!input_dump_directory(frame, directory))
        return;
    const InputSlot& slot = feature.inputs[slotIndex];
    for (const AuxBinding& binding : slot.auxBindings)
        if (binding.surface->staging[slotIndex].mapped == nullptr)
        {
            logf("frame %u: Ray Reconstruction guide %s is not readable, so this frame's capture is "
                 "incomplete and is not written",
                 frame, join_aux_names(binding.names).c_str());
            return;
        }
    std::string text = "guides " + std::to_string(slot.auxBindings.size()) + "\n";
    char path[MAX_PATH];
    for (size_t index = 0; index < slot.auxBindings.size(); ++index)
    {
        const AuxBinding& binding = slot.auxBindings[index];
        const AuxSurface& surface = *binding.surface;
        // Both spellings are recorded: the names the caller registered the texture under, and
        // the names the denoiser reads it as, which for the two albedos differ.
        std::string sources, names;
        for (const char* name : binding.sources)
            sources += sources.empty() ? name : std::string(",") + name;
        for (const char* name : binding.names)
            names += names.empty() ? name : std::string(",") + name;
        const int written =
            std::snprintf(path, MAX_PATH, "%s\\frame-%06u-guide-%02zu.bin", directory, frame, index);
        if (written <= 0 || written >= MAX_PATH)
        {
            logf("frame %u: the path for Ray Reconstruction guide %s does not fit; this frame's capture is "
                 "incomplete and is not written",
                 frame, join_aux_names(binding.names).c_str());
            return;
        }
        const Staging& staging = surface.staging[slotIndex];
        const size_t rowBytes = static_cast<size_t>(surface.texelBytes) * surface.width;
        const size_t bytes = rowBytes * surface.height;
        uint8_t* const pinned = capture_pinned(path, bytes);
        std::vector<uint8_t> fallback;
        if (pinned == nullptr)
            fallback.resize(bytes);
        uint8_t* const destination = pinned != nullptr ? pinned : fallback.data();
        for (UINT row = 0; row < surface.height; ++row)
            std::memcpy(destination + static_cast<size_t>(row) * rowBytes,
                        staging.mapped + staging.layout.Offset +
                            static_cast<size_t>(row) * staging.layout.Footprint.RowPitch,
                        rowBytes);
        if (pinned == nullptr)
            write_capture_file(frame, "Ray Reconstruction guide", path, destination, bytes);
        char line[512];
        const int length = std::snprintf(line, sizeof(line),
                                         "guide %zu %s %s %u %u %u %u %d\n", index, sources.c_str(), names.c_str(),
                                         static_cast<unsigned>(surface.format), surface.width, surface.height,
                                         surface.texelBytes, surface.how == AuxLayout::Verbatim ? 0 : 1);
        if (length <= 0)
            return;
        text += line;
    }
    // The scalar settings, the Subrect origins and the camera matrices, under the very
    // parameter names the denoiser reads: a replay sets them by name, so a table that drifted
    // here and there could not misroute them.
    for (size_t index = 0; index < sizeof(kAuxScalars) / sizeof(kAuxScalars[0]); ++index)
    {
        char line[256];
        const int length = std::snprintf(line, sizeof(line), "scalar %s %d\n", kAuxScalars[index].name,
                                         p.auxScalar[index]);
        if (length <= 0)
            return;
        text += line;
    }
    for (size_t index = 0; index < sizeof(kAuxSubrectNames) / sizeof(kAuxSubrectNames[0]); ++index)
    {
        for (int axis = 0; axis < 2; ++axis)
        {
            char line[256];
            const int length = std::snprintf(line, sizeof(line), "subrect %s.Subrect.Base.%c %u\n",
                                             kAuxSubrectNames[index], axis == 0 ? 'X' : 'Y',
                                             p.auxSubrect[index][axis]);
            if (length <= 0)
                return;
            text += line;
        }
    }
    for (size_t index = 0; index < sizeof(kAuxMatrices) / sizeof(kAuxMatrices[0]); ++index)
    {
        char line[1024];
        int length = std::snprintf(line, sizeof(line), "matrix %s %d", kAuxMatrices[index],
                                   p.auxMatrixSet[index] ? 1 : 0);
        if (length <= 0)
            return;
        for (size_t element = 0; element < 16 && p.auxMatrixSet[index]; ++element)
        {
            const int written = std::snprintf(line + length, sizeof(line) - static_cast<size_t>(length), " %.9g",
                                              static_cast<double>(p.auxMatrix[index][element]));
            if (written <= 0 || static_cast<size_t>(written) >= sizeof(line) - static_cast<size_t>(length))
                return;
            length += written;
        }
        text += line;
        text += "\n";
    }
    const int written = std::snprintf(path, MAX_PATH, "%s\\frame-%06u-guides.txt", directory, frame);
    if (written <= 0 || written >= MAX_PATH)
        return;
    write_capture_file(frame, "Ray Reconstruction guide parameters", path, text.data(), text.size());
}

// Worker. Points the denoiser's parameters at this frame's texture objects and forwards the
// settings and camera matrices the caller registered for this frame. Every name this file
// ever published is cleared first: the parameter object outlives the frame, so a name left
// holding last frame's handle would be read as a live texture.

static void publish_aux_parameters(Feature& feature, FrameParams& p, const InputSlot& slot)
{
    void* params = feature.cudaParams;
    for (size_t index = 0; index < sizeof(kAuxScalars) / sizeof(kAuxScalars[0]); ++index)
        if (p.auxScalarSet[index])
            d4r_ngx_set_int(params, kAuxScalars[index].name, p.auxScalar[index]);
    // Always written, never conditionally: the parameter object outlives this frame, so a
    // matrix pointer left over from the previous one would dangle the moment this frame's
    // FrameParams goes out of scope. Absent, it is cleared - the denoiser then knows the
    // caller supplied none, rather than reading a stale one.
    for (size_t index = 0; index < sizeof(kAuxMatrices) / sizeof(kAuxMatrices[0]); ++index)
        d4r_ngx_set_void(params, kAuxMatrices[index],
                         p.auxMatrixSet[index] ? static_cast<void*>(p.auxMatrix[index]) : nullptr);
    for (size_t index = 0; index < sizeof(kAuxSubrectNames) / sizeof(kAuxSubrectNames[0]); ++index)
    {
        static std::string nameX, nameY;
        nameX.assign(kAuxSubrectNames[index]).append(".Subrect.Base.X");
        nameY.assign(kAuxSubrectNames[index]).append(".Subrect.Base.Y");
        d4r_ngx_set_uint(params, nameX.c_str(), p.auxSubrect[index][0]);
        d4r_ngx_set_uint(params, nameY.c_str(), p.auxSubrect[index][1]);
    }
    for (const AuxSurface& surface : feature.aux)
        for (const char* name : surface.publish)
            d4r_ngx_set_void(params, name, nullptr);
    // Pointers to the handles, which is what a CUDA resource parameter is: a variable holding
    // the object, not the object itself. Two surfaces naming one denoiser input (a caller that
    // binds GBuffer.DiffuseAlbedo and DLSS.Input.DiffuseAlbedo to different textures) leave the
    // later one winning, which is the caller's ambiguity to resolve, not this file's.
    for (const AuxBinding& binding : slot.auxBindings)
        if (binding.surface->handle != 0)
            for (const char* name : binding.names)
                d4r_ngx_set_void(params, name, &binding.surface->handle);
}

// Worker, once the frame this slot was gathering for has passed. Each bound surface keeps the
// caller's resource alive until here, and a dropped surface is freed only once its last frame
// is gone.

static void retire_aux_bindings(Feature& feature, int slotIndex)
{
    std::lock_guard<std::mutex> lock(feature.auxMutex);
    for (AuxBinding& binding : feature.inputs[slotIndex].auxBindings)
        if (binding.surface->holds > 0)
            --binding.surface->holds;
    feature.inputs[slotIndex].auxBindings.clear();
    for (auto it = feature.aux.begin(); it != feature.aux.end();)
    {
        if (!it->dropping || it->holds != 0)
        {
            ++it;
            continue;
        }
        destroy_aux_surface(*it);
        it = feature.aux.erase(it);
    }
}

void set_evaluation_parameters(Feature& feature, const FrameParams& p)
{
    void* params = feature.cudaParams;
    feature.colorHandle = feature.color.object;
    feature.depthHandle = feature.depth.object;
    feature.motionHandle = feature.motion.object;
    feature.exposureHandle = feature.exposure.object;
    feature.outputHandle = feature.output.object;
    // CUDA resource parameters point to variables holding the object handles.
    d4r_ngx_set_void(params, "Color", &feature.colorHandle);
    d4r_ngx_set_void(params, "Depth", &feature.depthHandle);
    d4r_ngx_set_void(params, "MotionVectors", &feature.motionHandle);
    d4r_ngx_set_void(params, "Output", &feature.outputHandle);
    d4r_ngx_set_void(params, "ExposureTexture", p.hasExposure ? &feature.exposureHandle : nullptr);
    d4r_ngx_set_void(params, "TransparencyMask", nullptr);
    d4r_ngx_set_void(params, "DLSS.Input.Bias.Current.Color.Mask", nullptr);
    d4r_ngx_set_float(params, "Jitter.Offset.X", p.jitterX);
    d4r_ngx_set_float(params, "Jitter.Offset.Y", p.jitterY);
    d4r_ngx_set_float(params, "MV.Scale.X", p.mvScaleX);
    d4r_ngx_set_float(params, "MV.Scale.Y", p.mvScaleY);
    d4r_ngx_set_float(params, "Sharpness", p.sharpness);
    // The DLSS library stamps "DLSS SDK - DO NOT DISTRIBUTE" over the output in this setup; the SDK's
    // Disable.Watermark parameter turns that off. D4R_SHIM_WATERMARK=1 keeps it.
    static const bool watermark = env_uint("D4R_SHIM_WATERMARK", 0) != 0;
    d4r_ngx_set_int(params, "Disable.Watermark", watermark ? 0 : 1);
    d4r_ngx_set_int(params, "Reset", p.reset);
    d4r_ngx_set_float(params, "DLSS.Pre.Exposure", p.preExposure);
    d4r_ngx_set_float(params, "DLSS.Exposure.Scale", p.exposureScale);
    d4r_ngx_set_float(params, "FrameTimeDeltaInMsec", p.frameTime);
    d4r_ngx_set_int(params, "DLSS.Indicator.Invert.X.Axis", p.invertX);
    d4r_ngx_set_int(params, "DLSS.Indicator.Invert.Y.Axis", p.invertY);
    d4r_ngx_set_uint(params, "DLSS.Render.Subrect.Dimensions.Width", p.renderWidth);
    d4r_ngx_set_uint(params, "DLSS.Render.Subrect.Dimensions.Height", p.renderHeight);
    d4r_ngx_set_uint(params, "DLSS.Input.Color.Subrect.Base.X", p.colorBaseX);
    d4r_ngx_set_uint(params, "DLSS.Input.Color.Subrect.Base.Y", p.colorBaseY);
    d4r_ngx_set_uint(params, "DLSS.Input.Depth.Subrect.Base.X", p.depthBaseX);
    d4r_ngx_set_uint(params, "DLSS.Input.Depth.Subrect.Base.Y", p.depthBaseY);
    d4r_ngx_set_uint(params, "DLSS.Input.MV.Subrect.Base.X", p.mvBaseX);
    d4r_ngx_set_uint(params, "DLSS.Input.MV.Subrect.Base.Y", p.mvBaseY);
    d4r_ngx_set_uint(params, "DLSS.Output.Subrect.Base.X", p.outputBaseX);
    d4r_ngx_set_uint(params, "DLSS.Output.Subrect.Base.Y", p.outputBaseY);
    if (feature.scratch != 0)
    {
        d4r_ngx_set_void(params, "Scratch", reinterpret_cast<void*>(static_cast<uintptr_t>(feature.scratch)));
        d4r_ngx_set_ull(params, "Scratch.SizeInBytes", 64ull * 1024 * 1024);
    }
}

static void finish_publish(Feature* feature, int target, uint32_t frame, const FrameParams& params,
                           const FrameTiming& timing, ProfileClock::time_point start);

// Spins until `flag` clears; false after two seconds.

static bool wait_for_release(const std::atomic<bool>& flag)
{
    const auto start = std::chrono::steady_clock::now();
    while (flag.load())
    {
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(2))
            return false;
        d4r_sleep_us(100);
    }
    return true;
}

// Every queued frame ends in exactly one call of this (published or dropped).

void frame_retired(Feature* feature, uint32_t frame)
{
    // Covers every early error exit, even failures after the wait but before an output event was recorded.
    // Never release the split timeline or input ownership while queued NGX writes can still be running.
    complete_ngx_input_wait(*feature, frame, true);
    if (feature->gpuCompletion)
    {
        // Every queued frame ends here after its GPU work completed or was drained: release a game queue that may
        // be polling for it (success paths wrote the same value from the GPU already).
        static_cast<volatile uint32_t*>(feature->ngxInputSync.mapped)[kCompletionWord + frame % kOutputSlots] = frame;
        MemoryBarrier();
    }
    if (feature->split)
        release_split_frame(feature, frame);
    // The denoiser inputs this frame bound belong to the slot it used, and the caller stops
    // owing them a reference only now that its evaluate has returned.
    if (feature->rayReconstruction)
    {
        retire_aux_bindings(*feature, static_cast<int>(frame % kSlots));
    }
    feature->inFlight.fetch_sub(1);
}

// Overlapped staging, worker thread only. Uploads the staged planes with async
// copies into linear buffers on the "up" stream, then moves them into the CUDA
// arrays device-to-device (on the null stream, ahead of NGX's kernels).

static bool upload_inputs_async(Feature& feature, InputSlot& slot, bool hasExposure, uint32_t frame,
                                FrameTiming* timing)
{
    CopyStreams* streams = copy_streams();
    const Plane planes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
    CudaImage* images[4] = {&feature.color, &feature.depth, &feature.motion, &feature.exposure};
    const int count = hasExposure ? 4 : 3;
    const auto start = ProfileClock::now();
    for (int index = 0; index < count; ++index)
    {
        const HostPlane& host = slot.host[index];
        if (!ensure_cuda_image(*images[index], host.width, host.height, plane_format(planes[index]),
                               plane_channels(planes[index]), false, planes[index] == Plane::Color) ||
            !ensure_linear(feature.planeLinear[index], feature.planeLinearBytes[index], host.size()))
            return false;
        if ((frame <= 3 || env_uint("D4R_SHIM_INPUT_HASH_ALL", 0) != 0) && GetEnvironmentVariableA("D4R_SHIM_INPUT_HASH", nullptr, 0) != 0)
        {
            uint64_t hash = 1469598103934665603ull;
            for (size_t byte = 0; byte < host.size(); ++byte)
                hash = (hash ^ host.bytes[byte]) * 1099511628211ull;
            logf("frame %u upload plane=%d bytes=%zu hash=%016llx (async)", frame, index, host.size(),
                 static_cast<unsigned long long>(hash));
        }
        const auto issue = ProfileClock::now();
        const int result = g.cu.memcpyHtoDAsync(feature.planeLinear[index], host.bytes, host.size(), streams->up);
        if (timing != nullptr)
            timing->upload[index] = profile_ms(issue, ProfileClock::now());
        if (result != 0)
        {
            logf("cuMemcpyHtoDAsync failed: %d", result);
            return false;
        }
    }
    int result = g.cu.streamSynchronize(streams->up);
    for (int index = 0; index < count && result == 0; ++index)
    {
        const HostPlane& host = slot.host[index];
        CudaMemcpy2D copy = {};
        copy.srcMemoryType = CUDA_MEMORY_DEVICE;
        copy.srcDevice = feature.planeLinear[index];
        copy.srcPitch = host.rowBytes;
        copy.dstMemoryType = CUDA_MEMORY_ARRAY;
        copy.dstArray = images[index]->array;
        copy.WidthInBytes = host.rowBytes;
        copy.Height = host.height;
        result = copy_2d(copy);
    }
    if (timing != nullptr)
        timing->h2dTotal = profile_ms(start, ProfileClock::now());
    if (result != 0)
        logf("staged upload failed: %d", result);
    return result == 0;
}

// Starts the output download: array -> linear buffer, then an async copy on
// the "down" stream that the next frame's uploads overlap.

static bool start_download(Feature& feature, HostPlane& host, FrameTiming* timing)
{
    const size_t rowBytes = static_cast<size_t>(feature.output.width) * 8;
    const size_t bytes = rowBytes * feature.output.height;
    reserve_host(host, bytes);
    host.width = feature.output.width;
    host.height = feature.output.height;
    host.rowBytes = rowBytes;
    if (!ensure_linear(feature.outputLinear, feature.outputLinearBytes, bytes))
        return false;
    const auto start = ProfileClock::now();
    CudaMemcpy2D copy = {};
    copy.srcMemoryType = CUDA_MEMORY_ARRAY;
    copy.srcArray = feature.output.array;
    copy.dstMemoryType = CUDA_MEMORY_DEVICE;
    copy.dstDevice = feature.outputLinear;
    copy.dstPitch = rowBytes;
    copy.WidthInBytes = rowBytes;
    copy.Height = host.height;
    int result = feature.outputRedirected ? 0 : copy_2d(copy);
    // The null-stream copy must land before the non-blocking stream reads it.
    if (result == 0)
        result = g.cu.ctxSynchronize();
    if (result == 0)
        result = g.cu.memcpyDtoHAsync(host.bytes, feature.outputLinear, bytes, copy_streams()->down);
    if (timing != nullptr)
        timing->d2hIssue = profile_ms(start, ProfileClock::now());
    if (result != 0)
        logf("staged download failed: %d", result);
    return result == 0;
}

static PendingDownload g_pendingDownload;

static void post_publish(Feature* feature, int hostIndex, uint32_t frame, const FrameParams& params,
                         FrameTiming timing)
{
    if (timing.enabled)
        timing.finishQueued = ProfileClock::now();
    g.finish.post([feature, hostIndex, frame, params, timing] {
        publish_output(feature, hostIndex, frame, params, timing);
    });
}

// Waits for the in-flight download, if any, and hands it to publish_output.
// `inline_completion` marks the case without a next frame to overlap: the wait
// then belongs to this frame's worker time.

static void complete_pending_download(bool inline_completion = false)
{
    PendingDownload pending = g_pendingDownload;
    if (pending.feature == nullptr)
        return;
    g_pendingDownload = {};
    const auto waitStart = ProfileClock::now();
    const int result = g.cu.streamSynchronize(copy_streams()->down);
    if (pending.timing.enabled)
    {
        pending.timing.d2hWait = profile_ms(waitStart, ProfileClock::now());
        pending.timing.download = pending.timing.d2hIssue + pending.timing.d2hWait;
        if (inline_completion)
            pending.timing.worker += pending.timing.d2hWait;
    }
    if (result != 0)
    {
        logf("frame %u: output download failed: %d", pending.frame, result);
        pending.feature->outputHostBusy[pending.hostIndex] = false;
        frame_retired(pending.feature, pending.frame);
        return;
    }
    log_output_hash(pending.feature->outputHost[pending.hostIndex], pending.frame);
    char dumpPath[MAX_PATH];
    if (output_dump_path(pending.frame, dumpPath))
        dump_output(pending.feature->outputHost[pending.hostIndex], pending.frame, dumpPath);
    post_publish(pending.feature, pending.hostIndex, pending.frame, pending.params, pending.timing);
}

// Linear inputs, worker thread: pitch-linear texture objects on the slot's interop buffers (recreated
// when a buffer or its geometry changes), reported to NGX as arrays by the bridge.

static bool ensure_linear_inputs(InputSlot& slot, bool hasExposure)
{
    const Plane planes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
    const int count = hasExposure ? 4 : 3;
    for (int index = 0; index < count; ++index)
    {
        const HostPlane& geometry = slot.host[index];
        const VramBuffer& buffer = slot.vram[index];
        if (slot.linearTexture[index] != 0 && slot.linearPointer[index] == buffer.device &&
            slot.linearWidth[index] == geometry.width && slot.linearHeight[index] == geometry.height &&
            slot.linearPitch[index] == geometry.rowBytes)
            continue;
        if (slot.linearTexture[index] != 0)
            g.cu.texObjectDestroy(slot.linearTexture[index]);
        slot.linearTexture[index] = 0;
        struct Pitch2D
        {
            CudaDevicePtr pointer;
            uint32_t format, channels;
            size_t width, height, pitch;
        } pitch2D = {buffer.device, plane_format(planes[index]), plane_channels(planes[index]), geometry.width,
                     geometry.height, geometry.rowBytes};
        static_assert(sizeof(Pitch2D) == 40, "CUDA_RESOURCE_DESC pitch2D");
        CudaResourceDesc resource = {};
        resource.resType = 3; // CU_RESOURCE_TYPE_PITCH2D
        memcpy(&resource.res, &pitch2D, sizeof(pitch2D));
        const CudaTextureDesc sampler = input_sampler(geometry.width, pitch2D.channels, planes[index] == Plane::Color);
        CudaObject object = 0;
        int result = g.cu.texObjectCreate(&object, &resource, &sampler, nullptr);
        if (result == 0)
            result = g.cu.registerLinearTexture(object, geometry.width, geometry.height, pitch2D.format, pitch2D.channels);
        if (result != 0)
        {
            logf("linear input texture %d (%ux%u, pitch %zu) failed: %d", index, geometry.width, geometry.height,
                 geometry.rowBytes, result);
            if (object != 0)
                g.cu.texObjectDestroy(object);
            return false;
        }
        slot.linearTexture[index] = object;
        slot.linearPointer[index] = buffer.device;
        slot.linearWidth[index] = geometry.width;
        slot.linearHeight[index] = geometry.height;
        slot.linearPitch[index] = geometry.rowBytes;
    }
    return true;
}

// VRAM interop, worker thread: moves the planes from the imported buffers
// Opt-in thin-feature motion coverage. Keep the original imported velocity buffer
// immutable; the new texture is owned by this input slot until its GPU work retires.

static bool motion_dilation_parameters(Feature& feature, const InputSlot& slot, const FrameParams& p,
                                       D4rMotionDilationParams& args)
{
    const unsigned radius = env_uint("D4R_MOTION_DILATION", 0);
    const auto& depth = slot.host[1];
    const auto& motion = slot.host[2];
    args = {slot.vram[1].device, slot.vram[2].device, slot.dilatedMotion,
        depth.rowBytes, motion.rowBytes, p.renderWidth, p.renderHeight, motion.width, motion.height,
        p.jitterX, p.jitterY, radius, (feature.flags & 8u) != 0};
    const bool geometry = d4r_motion_dilation_geometry(&args, feature.flags, feature.outWidth, feature.outHeight);
    return radius >= 1 && radius <= 2 && p.vram && feature.linearInputs && g.cu.dilateMotion != nullptr &&
        geometry && p.colorBaseX == 0 && p.colorBaseY == 0 && p.depthBaseX == 0 &&
        p.depthBaseY == 0 && p.mvBaseX == 0 && p.mvBaseY == 0 && p.outputBaseX == 0 && p.outputBaseY == 0 &&
        p.renderWidth > 0 && p.renderHeight > 0 && std::isfinite(p.jitterX) && std::isfinite(p.jitterY) &&
        p.renderWidth <= depth.width && p.renderHeight <= depth.height;
}

// All lazy allocations/module initialization must precede a GPU-side input wait.
// Warm each slot through CPU marker polling first; geometry changes return it to
// that route. The generic fallback can copy synchronously, so keep it CPU-polled.

static bool motion_dilation_gpu_wait_ready(Feature& feature, const InputSlot& slot, const FrameParams& p)
{
    if (env_uint("D4R_MOTION_DILATION", 0) == 0)
        return true;
    D4rMotionDilationParams args = {};
    const auto& motion = slot.host[2];
    return g.cu.dilateMotionPadded != nullptr && slot.dilationEnqueued && slot.dilatedTexture != 0 &&
        slot.dilatedBytes >= motion.size() && slot.dilatedWidth == motion.width &&
        slot.dilatedHeight == motion.height && slot.dilatedPitch == motion.rowBytes &&
        motion_dilation_parameters(feature, slot, p, args) && d4r_motion_dilation_lds_eligible(&args);
}

CudaObject prepare_dilated_motion(Feature& feature, InputSlot& slot, const FrameParams& p)
{
    const unsigned radius = env_uint("D4R_MOTION_DILATION", 0);
    if (radius == 0)
        return 0;
    const auto& motion = slot.host[2];
    D4rMotionDilationParams args = {};
    const bool supported = motion_dilation_parameters(feature, slot, p, args);
    if (!supported)
    {
        if (!feature.dilationReported)
            logf("motion-vector dilation unavailable for this feature; original motion vectors retained");
        feature.dilationReported = true;
        return 0;
    }
    const size_t bytes = motion.size();
    const bool recreate = slot.dilatedBytes < bytes || slot.dilatedWidth != motion.width ||
        slot.dilatedHeight != motion.height || slot.dilatedPitch != motion.rowBytes;
    if (recreate)
    {
        slot.dilationEnqueued = false;
        if (slot.dilatedTexture != 0)
            g.cu.texObjectDestroy(slot.dilatedTexture), slot.dilatedTexture = 0;
        if (slot.dilatedMotion != 0)
            g.cu.memFree(slot.dilatedMotion), slot.dilatedMotion = 0;
        slot.dilatedBytes = 0;
        if (g.cu.memAlloc(&slot.dilatedMotion, bytes) != 0)
            return 0;
        slot.dilatedBytes = bytes;
        struct Pitch2D { CudaDevicePtr pointer; uint32_t format, channels; size_t width, height, pitch; };
        const Pitch2D pitch = {slot.dilatedMotion, plane_format(Plane::Motion), 2, motion.width, motion.height, motion.rowBytes};
        CudaResourceDesc resource = {};
        resource.resType = 3;
        static_assert(sizeof(pitch) == 40);
        std::memcpy(&resource.res, &pitch, sizeof(pitch));
        const CudaTextureDesc sampler = input_sampler(motion.width, 2, false);
        int result = g.cu.texObjectCreate(&slot.dilatedTexture, &resource, &sampler, nullptr);
        if (result == 0)
            result = g.cu.registerLinearTexture(slot.dilatedTexture, motion.width, motion.height, pitch.format, 2);
        if (result != 0)
        {
            if (slot.dilatedTexture != 0)
                g.cu.texObjectDestroy(slot.dilatedTexture), slot.dilatedTexture = 0;
            return 0;
        }
        slot.dilatedWidth = motion.width, slot.dilatedHeight = motion.height, slot.dilatedPitch = motion.rowBytes;
    }
    if (slot.dilatedTexture == 0)
        return 0;
    args.output = slot.dilatedMotion;
    // LDS copies padding in the same launch. Older bridges and the display-res
    // fallback retain the full-plane copy when their active grid is smaller.
    const bool paddedLds = g.cu.dilateMotionPadded != nullptr && d4r_motion_dilation_lds_eligible(&args);
    if (!paddedLds && (args.motion_width != motion.width || args.motion_height != motion.height))
    {
        CudaMemcpy2D copy = {};
        copy.srcMemoryType = copy.dstMemoryType = CUDA_MEMORY_DEVICE;
        copy.srcDevice = args.motion;
        copy.dstDevice = args.output;
        copy.srcPitch = copy.dstPitch = motion.rowBytes;
        copy.WidthInBytes = motion.rowBytes;
        copy.Height = motion.height;
        if (copy_2d(copy) != 0)
            return 0;
    }
    const bool thinRequested = env_uint("D4R_MOTION_THIN_SKY", 0) != 0;
    const bool thin = thinRequested && paddedLds && g.cu.dilateMotionThin != nullptr;
    const int result = thin ? g.cu.dilateMotionThin(&args, motion.height) :
        paddedLds ? g.cu.dilateMotionPadded(&args, motion.height) : g.cu.dilateMotion(&args);
    slot.dilationEnqueued = paddedLds && result == 0;
    if (!feature.dilationReported)
        logf("motion-vector dilation %s: radius=%u render pixels, GPU status=%d, sky_fill=%s, %s active=%ux%u allocation=%ux%u",
             result == 0 ? "enabled" : "unavailable", radius, result,
             thin ? "thin" : thinRequested ? "nearest (thin unsupported)" : "nearest",
             (feature.flags & 2u) != 0 ? "low-resolution" : "display-resolution",
             args.motion_width, args.motion_height, motion.width, motion.height);
    feature.dilationReported = true;
    return result == 0 ? slot.dilatedTexture : 0;
}

// VRAM interop, worker thread: moves the planes from the imported buffers
// into the CUDA arrays on the GPU. With D4R_SHIM_VRAM_VERIFY also compares
// them with the host-staged planes of the same frame.

static bool upload_inputs_vram(Feature& feature, InputSlot& slot, bool hasExposure, uint32_t frame,
                               FrameTiming* timing, bool gpuWait = false)
{
    const Plane planes[4] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
    CudaImage* images[4] = {&feature.color, &feature.depth, &feature.motion, &feature.exposure};
    const int count = hasExposure ? 4 : 3;
    const bool verify = !gpuWait && env_uint("D4R_SHIM_VRAM_VERIFY", 0) != 0;
    const auto start = ProfileClock::now();
    int result = 0;
    for (int index = 0; index < count && result == 0; ++index)
    {
        const HostPlane& geometry = slot.host[index];
        if (!ensure_cuda_image(*images[index], geometry.width, geometry.height, plane_format(planes[index]),
                               plane_channels(planes[index]), false, planes[index] == Plane::Color))
            return false;
        if (verify)
        {
            std::vector<uint8_t> vram(geometry.size());
            CudaMemcpy2D readback = {};
            readback.srcMemoryType = CUDA_MEMORY_DEVICE;
            readback.srcDevice = slot.vram[index].device;
            readback.srcPitch = geometry.rowBytes;
            readback.dstMemoryType = CUDA_MEMORY_HOST;
            readback.dstHost = vram.data();
            readback.dstPitch = geometry.rowBytes;
            readback.WidthInBytes = geometry.rowBytes;
            readback.Height = geometry.height;
            size_t differing = 0;
            if (g.cu.memcpy2D(&readback) == 0)
                for (size_t byte = 0; byte < vram.size(); ++byte)
                    differing += vram[byte] != geometry.bytes[byte];
            if (differing != 0 || frame <= 3 || frame % 120 == 0)
                logf("frame %u VRAM verify plane=%d: %zu of %zu bytes differ from host staging", frame, index,
                     differing, vram.size());
        }
        CudaMemcpy2D copy = {};
        copy.srcMemoryType = CUDA_MEMORY_DEVICE;
        copy.srcDevice = slot.vram[index].device;
        copy.srcPitch = geometry.rowBytes;
        copy.dstMemoryType = CUDA_MEMORY_ARRAY;
        copy.dstArray = images[index]->array;
        copy.WidthInBytes = geometry.rowBytes;
        copy.Height = geometry.height;
        // An active GPU wait requires the known null-stream async route. Do not silently fall back to
        // a synchronous or differently ordered copy while this frame's input boundary is outstanding.
        result = gpuWait ? g.cu.memcpy2DAsync(&copy, nullptr) : copy_2d(copy);
        // D4R_SHIM_VRAM_ARRAY_VERIFY=1: read the CUDA array DLSS samples back
        // and compare it with the imported buffer it was filled from.
        static const bool arrayVerify = env_uint("D4R_SHIM_VRAM_ARRAY_VERIFY", 0) != 0;
        if (!gpuWait && arrayVerify && result == 0 && g.cu.ctxSynchronize() == 0)
        {
            std::vector<uint8_t> fromBuffer(geometry.size()), fromArray(geometry.size());
            CudaMemcpy2D readBuffer = {};
            readBuffer.srcMemoryType = CUDA_MEMORY_DEVICE;
            readBuffer.srcDevice = slot.vram[index].device;
            readBuffer.srcPitch = geometry.rowBytes;
            readBuffer.dstMemoryType = CUDA_MEMORY_HOST;
            readBuffer.dstHost = fromBuffer.data();
            readBuffer.dstPitch = geometry.rowBytes;
            readBuffer.WidthInBytes = geometry.rowBytes;
            readBuffer.Height = geometry.height;
            CudaMemcpy2D readArray = {};
            readArray.srcMemoryType = CUDA_MEMORY_ARRAY;
            readArray.srcArray = images[index]->array;
            readArray.dstMemoryType = CUDA_MEMORY_HOST;
            readArray.dstHost = fromArray.data();
            readArray.dstPitch = geometry.rowBytes;
            readArray.WidthInBytes = geometry.rowBytes;
            readArray.Height = geometry.height;
            if (g.cu.memcpy2D(&readBuffer) == 0 && g.cu.memcpy2D(&readArray) == 0)
            {
                size_t differing = 0, firstRow = geometry.height;
                for (size_t byte = 0; byte < fromArray.size(); ++byte)
                    if (fromArray[byte] != fromBuffer[byte])
                    {
                        if (differing++ == 0)
                            firstRow = byte / geometry.rowBytes;
                    }
                if (differing != 0 || frame <= 3 || frame % 120 == 0)
                    logf("frame %u VRAM array verify plane=%d: %zu of %zu bytes differ (first row %zu)", frame, index,
                         differing, fromArray.size(), firstRow);
            }
        }
    }
    if (timing != nullptr)
        timing->h2dTotal = profile_ms(start, ProfileClock::now());
    if (result != 0)
    {
        logf("VRAM input copy failed: %d", result);
        if (gpuWait && feature.ngxGpuWait.exchange(false))
            logf("NGX feature %u: null-stream async input copy failed; using CPU marker polling from now on",
                 feature.handle.Id);
    }
    return result == 0;
}

// Only called on the CUDA worker. A completion event covers queued work on
// the default stream; do not rely on HIP's blocking flag to release the CPU.

int synchronize_default_stream(Feature& feature, bool* querySleep, double* waitEmaUs)
{
    if (feature.outputReadyEvent != nullptr)
    {
        const int recorded = g.cu.eventRecord(feature.outputReadyEvent, nullptr);
        if (recorded == 0)
        {
            // D4R_SHIM_OUTPUT_POLL_US: N > 0 polls every N us (200 was the fixed interval before), 0 yields between
            // queries (spins). Unset: adaptive - sleep until shortly before the predicted completion (smoothed
            // wait of earlier frames), then poll every 20 us, backing off to 200 us when the GPU runs late.
            static const int fixedPoll = [] {
                const char* v = std::getenv("D4R_SHIM_OUTPUT_POLL_US");
                return v != nullptr && *v != '\0' ? static_cast<int>(env_uint("D4R_SHIM_OUTPUT_POLL_US", 200)) : -1;
            }();
            const auto waitStart = std::chrono::steady_clock::now();
            unsigned polls = 0;
            double lastNotReadyUs = -1.0; // elapsed time of the last query that found the work unfinished
            const int result = d4r_wait_event(
                [&] { return g.cu.eventQuery(feature.outputReadyEvent); },
                [&] {
                    if (fixedPoll >= 0)
                    {
                        d4r_sleep_us(static_cast<unsigned>(fixedPoll));
                        return;
                    }
                    constexpr double kMarginUs = 120.0, kStepUs = 20.0, kChunkUs = 200.0, kSpinAfterUs = 150.0;
                    const double elapsed = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - waitStart).count();
                    lastNotReadyUs = elapsed; // this sleep follows a query that returned NOT_READY
                    ++polls;
                    const double predicted = waitEmaUs != nullptr ? *waitEmaUs : 0.0;
                    const double remaining = predicted - kMarginUs - elapsed;
                    // far from the predicted completion: sleep in chunks of at most 200 us (a single long sleep
                    // can overshoot by far more under Wine); near it: 20 us polls; late: back off to 200 us
                    if (remaining > kStepUs)
                        d4r_sleep_us(static_cast<unsigned>(std::min(remaining, kChunkUs)));
                    else if (elapsed < predicted + kSpinAfterUs)
                        d4r_sleep_us(0); // final stretch around the predicted completion: spin (SwitchToThread);
                                         // a Wine sleep rounds up to ~50 us, which would be the detection latency
                    else
                        d4r_sleep_us(static_cast<unsigned>(std::min(
                            kStepUs * (1.0 + (elapsed - predicted - kSpinAfterUs) / 160.0), kChunkUs)));
                });
            if (result == 0 && fixedPoll < 0 && waitEmaUs != nullptr)
            {
                const double waited = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - waitStart).count();
                // Learn the completion time, not our own wait: the work finished between the last NOT_READY query
                // and the successful one (at most one 200 us chunk apart), so the midpoint cannot ratchet upwards
                double sample = lastNotReadyUs >= 0.0 ? 0.5 * (lastNotReadyUs + waited) : waited;
                sample = std::min(sample, 20000.0);
                *waitEmaUs = *waitEmaUs == 0.0 ? sample : 0.8 * *waitEmaUs + 0.2 * sample;
            }
            if (result == 0)
            {
                if (querySleep != nullptr) *querySleep = true;
                // The event is already complete; this also flushes bridge kernel profiles.
                return g.cu.eventSynchronize(feature.outputReadyEvent);
            }
            logf("output event query failed (%d); using context synchronization", result);
        }
        else
            logf("output event record failed (%d); using context synchronization", recorded);
    }
    return g.cu.ctxSynchronize();
}

// A frame recorded for packed output whose output kernel then did not pack it (it was not the native one
// after all): the RGBA16F result is converted on the host, rounding as the kernel store and the game-side
// blit do. Slow, and only expected around a kernel change.

static int pack_output_on_host(Feature& feature, CudaMemcpy2D copy, size_t packedPitch, uint32_t frame, bool rgba8 = false)
{
    const size_t width = feature.output.width, height = feature.output.height;
    std::vector<uint16_t> halves(width * height * 4);
    std::vector<uint32_t> packed(width * height);
    const CudaDevicePtr target = copy.dstDevice;
    copy.dstMemoryType = CUDA_MEMORY_HOST;
    copy.dstHost = halves.data();
    copy.dstPitch = width * 8;
    int result = g.cu.memcpy2D(&copy);
    if (result != 0)
        return result;
    const auto unorm = [](uint16_t half, float scale) {
        float value = half_to_float(half);
        value = value > 0.0f ? value : 0.0f;
        value = value < 1.0f ? value : 1.0f;
        return static_cast<uint32_t>(value * scale + 0.5f);
    };
    for (size_t texel = 0; texel < packed.size(); ++texel)
        packed[texel] = rgba8
            ? unorm(halves[texel * 4], 255.0f) | unorm(halves[texel * 4 + 1], 255.0f) << 8 |
              unorm(halves[texel * 4 + 2], 255.0f) << 16 | unorm(halves[texel * 4 + 3], 255.0f) << 24
            : unorm(halves[texel * 4], 1023.0f) | unorm(halves[texel * 4 + 1], 1023.0f) << 10 |
              unorm(halves[texel * 4 + 2], 1023.0f) << 20 | unorm(halves[texel * 4 + 3], 3.0f) << 30;
    CudaMemcpy2D upload = {};
    upload.srcMemoryType = CUDA_MEMORY_HOST;
    upload.srcHost = packed.data();
    upload.srcPitch = width * 4;
    upload.dstMemoryType = CUDA_MEMORY_DEVICE;
    upload.dstDevice = target;
    upload.dstPitch = packedPitch;
    upload.WidthInBytes = width * 4;
    upload.Height = height;
    result = g.cu.memcpy2D(&upload);
    logf("frame %u: output kernel did not pack its result; converted on the host (%d)", frame, result);
    return result;
}

// Worker only: finish the tracked wait once output completed, or release and drain an aborted submission.

bool complete_ngx_input_wait(Feature& feature, uint32_t frame, bool abort)
{
    {
        std::lock_guard<std::mutex> lock(feature.ngxWaitMutex);
        if (!feature.ngxWaitActive || feature.ngxWaitFrame != frame)
            return true;
        if (abort)
            release_ngx_input_wait_locked(feature, "submission or output completion failed", false);
    }
    // Success callers already synchronized the output; error callers may have no completion event, so
    // drain the real context (the bridge export bypasses NGX sync elision) after releasing the wait. Keep
    // the watchdog armed during this call, and the slot/resources owned until it returns.
    if (abort && g.cu.ctxSynchronize() != 0)
    {
        feature.ngxGpuWait.store(false);
        feature.retiring.store(true);
        logf("NGX frame %u: context drain failed after input-wait release; retiring feature", frame);
    }
    std::lock_guard<std::mutex> lock(feature.ngxWaitMutex);
    const bool valid = !feature.ngxWaitCancelled;
    feature.ngxWaitActive = false;
    if (!valid)
        feature.evaluatedFrames = 0; // a discarded evaluation must not supply temporal history to the next one
    return valid;
}

// VRAM interop, worker thread: copies the finished result into a free output
// slot's imported buffer and publishes it; evaluate() copies it on to the game.

static void publish_vram(Feature* feature, uint32_t frame, const FrameParams& params, FrameTiming& timing,
                         ProfileClock::time_point workerStart)
{
    const auto start = ProfileClock::now();
    // A split frame's second half reads slot frame % kOutputSlots; its previous
    // reader, frame - kOutputSlots, ran before this frame's marker.
    const int target = params.split ? static_cast<int>(frame % kOutputSlots) : claim_output_slot(feature);
    if (target < 0)
    {
        logf("frame %u: no free output slot; dropping result", frame);
        frame_retired(feature, frame);
        return;
    }
    VramBuffer buffer = feature->outputs[target].vram;
    const size_t rowBytes = static_cast<size_t>(feature->output.width) * 8;
    CudaMemcpy2D copy = {};
    copy.srcMemoryType = CUDA_MEMORY_ARRAY;
    copy.srcArray = feature->output.array;
    copy.dstMemoryType = CUDA_MEMORY_DEVICE;
    copy.dstDevice = buffer.device;
    copy.dstPitch = rowBytes;
    copy.WidthInBytes = rowBytes;
    copy.Height = feature->output.height;
    if(feature->wireOutputReady)
    {
        copy.srcMemoryType=CUDA_MEMORY_DEVICE;
        copy.srcArray=nullptr;
        copy.srcDevice=feature->wirePublishBuffer;
        copy.srcPitch=rowBytes;
    }
    if (params.outputInPlace != 0)
    {
        // no game-side copy was recorded: a result the output kernel did not redirect goes to the texture here
        buffer.device = copy.dstDevice = params.outputInPlace;
        copy.dstPitch = params.outputInPlacePitch;
        buffer.bytes = static_cast<size_t>(params.outputInPlacePitch) * feature->output.height;
    }
    int result = rowBytes / (params.packedOutput ? 2 : 1) * feature->output.height > buffer.bytes ? -1
                 : feature->outputRedirected || feature->wirePublishedDirect ? 0
                 : params.packedOutput ? pack_output_on_host(*feature, copy, params.outputInPlace != 0 ? copy.dstPitch
                                                                               : rowBytes / 2, frame, params.packedRgba8)
                 : copy_2d(copy);
    // GPU-side completion: the word is written behind every output write on NGX's stream. A failed enqueue is
    // covered by frame_retired's CPU write once this frame has synchronized below.
    if (result == 0 && params.gpuCompletion && !feature->completionFired &&
        g.cu.ngxWriteValue32(feature->ngxInputSync.device + sizeof(uint32_t) * (kCompletionWord + frame % kOutputSlots),
                             frame) != 0)
    {
        static bool logged = false;
        if (!logged)
            logf("frame %u: GPU completion write could not be queued; the CPU releases the game's queue", frame);
        logged = true;
    }
    bool nativeQueued = false;
    if (result == 0 && params.split && feature->ngxNativeCompletion && !feature->retiring.load())
    {
        bool head;
        {
            std::lock_guard<std::mutex> lock(feature->splitMutex);
            head = feature->splitCompletion.can_signal(frame);
        }
        // The callback is ordered after every output write. It releases this frame on the GPU's
        // completion boundary and wakes the native condition; the worker retains resources until
        // that condition completes. Future discarded frames cannot cross an earlier producer.
        if (head)
        {
            nativeQueued = g.cu.timelineQueue(feature->ngxTimeline, frame) == 0;
            if (!nativeQueued)
            {
                feature->ngxNativeCompletion = false;
                logf("NGX native completion enqueue failed; using CPU output completion from now on");
            }
        }
    }
    if (result == 0)
    {
        const auto syncStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
        const double cpuStart = timing.enabled ? profile_thread_cpu_ms() : -1.0;
        if (nativeQueued)
        {
            result = g.cu.timelineWait(feature->ngxTimeline);
            timing.outputSyncNative = result == 0;
            if (result != 0)
            {
                g.cu.timelineCancel(feature->ngxTimeline);
                feature->ngxNativeCompletion = false;
                logf("NGX native completion wait failed (%d); using CPU output completion from now on", result);
            }
        }
        if (!nativeQueued || result != 0)
            result = synchronize_default_stream(*feature, &timing.outputSyncBlocking, &feature->publishWaitEmaUs);
        if (timing.enabled)
        {
            timing.outputSyncWall = profile_ms(syncStart, ProfileClock::now());
            const double cpuEnd = profile_thread_cpu_ms();
            if (cpuStart >= 0.0 && cpuEnd >= 0.0)
                timing.outputSyncCpu = std::max(0.0, cpuEnd - cpuStart);
        }
    }
    if (result == 0 && !complete_ngx_input_wait(*feature, frame, false))
    {
        frame_retired(feature, frame);
        return; // queued work drained, but watchdog-released input is not a publishable frame
    }
    if (result == 0 && timing.gpuEventsRecorded)
    {
        float gpuEvalMs = -1.0f;
        if (g.cu.eventElapsedTime(&gpuEvalMs, feature->profileStart, feature->profileEnd) == 0)
            timing.gpuEval = gpuEvalMs;
        float dlssMs = -1.0f, inputMs = -1.0f;
        if (timing.inputEventRecorded &&
            g.cu.eventElapsedTime(&dlssMs, feature->profileInputsReleased, feature->profileEnd) == 0 &&
            g.cu.eventElapsedTime(&inputMs, feature->profileStart, feature->profileInputsReleased) == 0)
        {
            timing.dlssGpu = dlssMs;
            timing.inputGpuWait = inputMs;
        }
        else if (!timing.gpuInputWait && timing.gpuEval >= 0.0)
        {
            timing.dlssGpu = timing.gpuEval;
            timing.inputGpuWait = 0.0;
        }
    }
    if(result==0)
    {
        float ms=-1;
        if(timing.nativeMotionTimed && g.cu.eventElapsedTime(&ms,feature->nativePassEvents[0],feature->nativePassEvents[1])==0)
            timing.nativeMotionGpu=ms;
        if(timing.wireRepairTimed && g.cu.eventElapsedTime(&ms,feature->nativePassEvents[2],feature->nativePassEvents[3])==0)
            timing.wireRepairGpu=ms;
    }
    if (result != 0)
    {
        logf("frame %u: VRAM output copy failed: %d", frame, result);
        std::lock_guard<std::mutex> lock(feature->outputMutex);
        feature->outputs[target].lastReadFrame = 0;
        frame_retired(feature, frame);
        return;
    }
    static const bool arrayVerify = env_uint("D4R_SHIM_VRAM_ARRAY_VERIFY", 0) != 0;
    if (arrayVerify)
    {
        // The result as DLSS left it in its array vs the buffer the game copies from.
        std::vector<uint8_t> fromArray(rowBytes * feature->output.height), fromBuffer(fromArray.size());
        CudaMemcpy2D readArray = copy;
        readArray.dstMemoryType = CUDA_MEMORY_HOST;
        readArray.dstHost = fromArray.data();
        CudaMemcpy2D readBuffer = {};
        readBuffer.srcMemoryType = CUDA_MEMORY_DEVICE;
        readBuffer.srcDevice = buffer.device;
        readBuffer.srcPitch = rowBytes;
        readBuffer.dstMemoryType = CUDA_MEMORY_HOST;
        readBuffer.dstHost = fromBuffer.data();
        readBuffer.dstPitch = rowBytes;
        readBuffer.WidthInBytes = rowBytes;
        readBuffer.Height = feature->output.height;
        if (g.cu.memcpy2D(&readArray) == 0 && g.cu.memcpy2D(&readBuffer) == 0)
        {
            size_t differing = 0;
            for (size_t byte = 0; byte < fromArray.size(); ++byte)
                differing += fromArray[byte] != fromBuffer[byte];
            if (differing != 0 || frame <= 3 || frame % 120 == 0)
                logf("frame %u VRAM output verify: %zu of %zu bytes differ between result array and buffer", frame,
                     differing, fromArray.size());
        }
    }
    char dumpPath[MAX_PATH];
    const bool saveRaw = output_dump_path(frame, dumpPath);
    if (GetEnvironmentVariableA("D4R_SHIM_OUTPUT_HASH", nullptr, 0) != 0 || saveRaw)
    {
        HostPlane host;
        uint8_t* pinned = saveRaw ? capture_pinned(dumpPath, rowBytes * feature->output.height) : nullptr;
        if (pinned == nullptr)
            host.fallback.resize(rowBytes * feature->output.height);
        host.bytes = pinned != nullptr ? pinned : host.fallback.data();
        host.width = feature->output.width;
        host.height = feature->output.height;
        host.rowBytes = rowBytes;
        copy.dstMemoryType = CUDA_MEMORY_HOST;
        copy.dstHost = host.bytes;
        if (g.cu.memcpy2D(&copy) == 0)
        {
            log_output_hash(host, frame);
            if (saveRaw && pinned != nullptr)
                flush_capture_if_complete(frame); // already registered for writing
            else if (saveRaw)
                dump_output(host, frame, dumpPath);
        }
        else if (saveRaw)
            logf("frame %u raw DLSS output readback failed", frame);
    }
    if (timing.enabled)
    {
        timing.d2hIssue = profile_ms(start, ProfileClock::now());
        timing.download = timing.d2hIssue;
        timing.worker = profile_ms(workerStart, ProfileClock::now());
    }
    finish_publish(feature, target, frame, params, timing, start);
}

// Worker. Streamline can create the feature at one quality's render size while the game still renders at
// another's (PRAGMATA starts with an Ultra Performance 854x480 frame on a Performance 1280x720 feature), and NGX
// rejects a subrect outside the feature's dynamic range on every frame until the game recreates it: a black
// screen. Recreate the NGX feature at the size actually rendered; the game's handle and output stay the same.

bool recreate_for_render_size(Feature& feature, FrameParams& params, uint32_t frame)
{
    if ((params.renderWidth == feature.cudaWidth && params.renderHeight == feature.cudaHeight) ||
        (params.renderWidth == feature.refusedWidth && params.renderHeight == feature.refusedHeight))
        return false;
    logf("frame %u: NGX rejected render %ux%u on a %ux%u feature; recreating it at %ux%u", frame, params.renderWidth,
         params.renderHeight, feature.cudaWidth, feature.cudaHeight, params.renderWidth, params.renderHeight);
    // earlier frames' kernels may still be running on the old feature
    synchronize_default_stream(feature);
    if (feature.cudaHandle != nullptr)
        g.ngx.releaseFeature(feature.cudaHandle);
    feature.cudaHandle = nullptr;
    void* p = feature.cudaParams;
    d4r_ngx_set_uint(p, "Width", params.renderWidth);
    d4r_ngx_set_uint(p, "Height", params.renderHeight);
    const unsigned int featureId =
        feature.rayReconstruction ? NGX_FEATURE_RAY_RECONSTRUCTION : NGX_FEATURE_SUPER_SAMPLING;
    NgxResult created = g.ngx.createFeature(featureId, p, &feature.cudaHandle);
    unsigned int width = params.renderWidth, height = params.renderHeight;
    if (created != NGX_SUCCESS)
    {
        logf("frame %u: recreating the NGX feature at %ux%u failed (0x%08x); restoring %ux%u", frame, width, height,
             created, feature.cudaWidth, feature.cudaHeight);
        width = feature.cudaWidth;
        height = feature.cudaHeight;
        d4r_ngx_set_uint(p, "Width", width);
        d4r_ngx_set_uint(p, "Height", height);
        if (width != 0 && height != 0)
            created = g.ngx.createFeature(featureId, p, &feature.cudaHandle);
        feature.refusedWidth = created == NGX_SUCCESS ? params.renderWidth : 0;
        feature.refusedHeight = created == NGX_SUCCESS ? params.renderHeight : 0;
        if (created != NGX_SUCCESS)
        {
            logf("frame %u: restoring the NGX feature failed (0x%08x)", frame, created);
            feature.cudaHandle = nullptr;
            width = height = 0;
        }
    }
    // a new feature has no history
    feature.cudaWidth = width;
    feature.cudaHeight = height;
    params.reset = 1;
    d4r_ngx_set_int(p, "Reset", 1);
    return feature.cudaHandle != nullptr && width == params.renderWidth && height == params.renderHeight;
}

static bool ngx_gpu_wait_eligible(Feature& feature, const InputSlot& slot, const FrameParams& params, uint32_t frame)
{
    char directory[MAX_PATH];
    // The denoiser can mix GPU guides with CPU-transcoded auxiliary guides and alpha readbacks. Keep
    // that mixed route on the CPU readiness contract; never infer that main-plane VRAM means all guides
    // are GPU-only. Likewise verification, captures and host staging must see real input first.
    return feature.ngxGpuWait.load() && params.gpuInputMarker && params.vram && params.split && !feature.rayReconstruction &&
           !slot.nativeLive &&
           env_uint("D4R_SHIM_VRAM_VERIFY", 0) == 0 && env_uint("D4R_SHIM_VRAM_ARRAY_VERIFY", 0) == 0 &&
           motion_dilation_gpu_wait_ready(feature, slot, params) && !input_dump_directory(frame, directory) &&
           (feature.linearInputs || (env_uint("D4R_SHIM_ASYNC_2D", 1) != 0 && g.cu.memcpy2DAsync != nullptr));
}

static void run_native_gpu_check(InputSlot& slot, const FrameParams& p, uint32_t frame)
{
    if(!slot.nativeGpuCheckReady || !slot.nativeGpuStageValid) return;
    for(auto& buffer : slot.nativeGpuInputs)
    {
        // Pooled kind-0 buffers can originate from a guide that never needed
        // a CUDA import. Import on the context-owning worker before use.
        if(!buffer.external)
        {
            VkMemoryRequirements requirements={};g_vk.bufferRequirements(g_vk.device,buffer.buffer,&requirements);
            const int imported=g_vk.import(g_vk.device,reinterpret_cast<uint64_t>(buffer.memory),requirements.size,&buffer.device,&buffer.external);
            if(imported!=0) { logf("native GPU check frame %u: pooled buffer import failed %d",frame,imported);return; }
        }
        if(!buffer.device) { logf("native GPU check frame %u: missing input device address",frame);return; }
    }
    const size_t row=static_cast<size_t>(slot.nativeGpuWidth)*4;
    const size_t bytes=row*slot.nativeGpuHeight;
    if(slot.nativeGpuResultBytes!=bytes)
    {
        if(slot.nativeGpuResult) g.cu.memFree(slot.nativeGpuResult),slot.nativeGpuResult=0;
        slot.nativeGpuResultBytes=0;
        if(g.cu.memAlloc(&slot.nativeGpuResult,bytes)!=0) { logf("native GPU check frame %u: allocation failed",frame);return; }
        slot.nativeGpuResultBytes=bytes;
    }
    const D4rNativeMotionParams args={slot.nativeGpuInputs[0].device,slot.nativeGpuInputs[1].device,
        slot.nativeGpuInputs[2].device,slot.nativeGpuInputs[3].device,slot.nativeGpuResult,
        row,row,row,slot.nativeGpuWidth,slot.nativeGpuHeight,p.renderWidth,p.renderHeight};
    int result=g.cu.composeNativeMotion(&args);
    if(result==0) result=g.cu.ctxSynchronize(); // diagnostic readback only; not a live pipeline timing path
    std::vector<uint8_t> output(bytes);
    CudaMemcpy2D copy={};copy.srcMemoryType=CUDA_MEMORY_DEVICE;copy.srcDevice=slot.nativeGpuResult;copy.srcPitch=row;
    copy.dstMemoryType=CUDA_MEMORY_HOST;copy.dstHost=output.data();copy.dstPitch=row;copy.WidthInBytes=row;copy.Height=slot.nativeGpuHeight;
    if(result==0) result=g.cu.memcpy2D(&copy);
    const std::string base=slot.nativeMotionCapturePath.substr(0,slot.nativeMotionCapturePath.find(".geometry.rg16f"));
    if(result==0) write_capture_file(frame,"GPU-only imported native composition",(base+".gpu-owned-dilated.rg16f").c_str(),output.data(),output.size());
    logf("native GPU check frame %u: compose/sync/readback=%d size=%ux%u",frame,result,slot.nativeGpuWidth,slot.nativeGpuHeight);
}

// These events only bracket sampled live-native dispatches. Elapsed times are
// read after the existing publication synchronization; no extra wait is added.

static bool begin_native_pass_profile(Feature& feature,unsigned pair,uint32_t frame,const FrameTiming* timing)
{
    return timing && timing->enabled && profile_log_frame(frame) && feature.nativePassEvents[pair] &&
        feature.nativePassEvents[pair+1] && g.cu.eventRecord(feature.nativePassEvents[pair],nullptr)==0;
}

static uint32_t wire_history_flags()
{
    return (env_uint("D4R_NATIVE_MV_WIRE_CONSERVATIVE",0)==1 ? D4R_WIRE_HISTORY_CONSERVATIVE : 0u) |
        (env_uint("D4R_NATIVE_MV_WIRE_SKY_ONLY",0)==1 ? D4R_WIRE_HISTORY_SKY_ONLY : 0u);
}

// D4R_NATIVE_MV_LIVE, worker: composes the owned native inputs into an output-resolution field and exposes it as a
// pitch-linear texture on the NGX stream. No CPU wait: the inputs were complete at the observed input marker and
// the kernel runs on the same stream as NGX. Returns the texture, or 0 to keep the game's own vectors.

static CudaObject compose_native_live(Feature& feature, InputSlot& slot, const FrameParams& p, uint32_t frame, FrameTiming* timing)
{
    feature.preparedWireAxesReady=false;
    if(!slot.nativeLive || !slot.nativeGpuCheckReady || !slot.nativeGpuStageValid || !feature.linearInputs ||
        !p.vram || p.colorBaseX || p.colorBaseY || p.depthBaseX || p.depthBaseY || p.mvBaseX || p.mvBaseY ||
        p.outputBaseX || p.outputBaseY || p.renderWidth==0 || p.renderHeight==0 ||
        slot.nativeGpuWidth!=feature.outWidth || slot.nativeGpuHeight!=feature.outHeight ||
        !g.cu.composeNativeMotion || !g.cu.texObjectCreate || !g.cu.registerLinearTexture) return 0;
    for(auto& buffer : slot.nativeGpuInputs)
    {
        if(!buffer.external)
        {
            VkMemoryRequirements requirements={};g_vk.bufferRequirements(g_vk.device,buffer.buffer,&requirements);
            const int imported=g_vk.import(g_vk.device,reinterpret_cast<uint64_t>(buffer.memory),requirements.size,&buffer.device,&buffer.external);
            if(imported!=0) { logf("native live frame %u: pooled buffer import failed %d",frame,imported);return 0; }
        }
        if(!buffer.device) { logf("native live frame %u: missing input device address",frame);return 0; }
    }
    const size_t rowIn=static_cast<size_t>(slot.nativeGpuWidth)*4;
    const size_t rowOut=(rowIn+255)&~size_t(255);
    const size_t bytes=rowOut*slot.nativeGpuHeight;
    if(slot.nativeGpuResultBytes!=bytes)
    {
        if(slot.nativeGpuTexture) g.cu.texObjectDestroy(slot.nativeGpuTexture),slot.nativeGpuTexture=0;
        if(slot.nativeGpuResult) g.cu.memFree(slot.nativeGpuResult),slot.nativeGpuResult=0;
        slot.nativeGpuResultBytes=0;
        if(g.cu.memAlloc(&slot.nativeGpuResult,bytes)!=0) { logf("native live frame %u: allocation failed",frame);return 0; }
        slot.nativeGpuResultBytes=bytes;
    }
    if(slot.nativeGpuTexture==0 || slot.nativeTexPointer!=slot.nativeGpuResult || slot.nativeTexWidth!=slot.nativeGpuWidth ||
        slot.nativeTexHeight!=slot.nativeGpuHeight || slot.nativeTexPitch!=rowOut)
    {
        if(slot.nativeGpuTexture) g.cu.texObjectDestroy(slot.nativeGpuTexture),slot.nativeGpuTexture=0;
        struct Pitch2D { CudaDevicePtr pointer; uint32_t format, channels; size_t width, height, pitch; };
        const Pitch2D pitch={slot.nativeGpuResult,plane_format(Plane::Motion),2,slot.nativeGpuWidth,slot.nativeGpuHeight,rowOut};
        static_assert(sizeof(Pitch2D)==40);
        CudaResourceDesc resource={};resource.resType=3;std::memcpy(&resource.res,&pitch,sizeof(pitch));
        const CudaTextureDesc sampler=input_sampler(slot.nativeGpuWidth,2,false);
        int result=g.cu.texObjectCreate(&slot.nativeGpuTexture,&resource,&sampler,nullptr);
        if(result==0) result=g.cu.registerLinearTexture(slot.nativeGpuTexture,slot.nativeGpuWidth,slot.nativeGpuHeight,pitch.format,2);
        if(result!=0)
        {
            logf("native live frame %u: motion texture failed %d",frame,result);
            if(slot.nativeGpuTexture) g.cu.texObjectDestroy(slot.nativeGpuTexture),slot.nativeGpuTexture=0;
            return 0;
        }
        slot.nativeTexPointer=slot.nativeGpuResult;slot.nativeTexWidth=slot.nativeGpuWidth;
        slot.nativeTexHeight=slot.nativeGpuHeight;slot.nativeTexPitch=rowOut;
    }
    const D4rNativeMotionParams args={slot.nativeGpuInputs[0].device,slot.nativeGpuInputs[1].device,
        slot.nativeGpuInputs[2].device,slot.nativeGpuInputs[3].device,slot.nativeGpuResult,
        rowIn,rowIn,rowOut,slot.nativeGpuWidth,slot.nativeGpuHeight,p.renderWidth,p.renderHeight};
    const bool thinHalo=env_uint("D4R_NATIVE_MV_THIN_HALO",1)==1;
    if(thinHalo && !g.cu.composeNativeMotionThin) return 0;
    const D4rNativeMotionThinParams thinArgs={args,p.jitterX,p.jitterY};
    const bool hybrid=thinHalo && env_uint("D4R_NATIVE_MV_HYBRID",0)==1 && g.cu.composeNativeMotionHybrid &&
        slot.linearPointer[2] && slot.host[2].width==p.renderWidth && slot.host[2].height==p.renderHeight;
    const D4rNativeMotionHybridParams hybridArgs={thinArgs,slot.linearPointer[2],slot.linearPitch[2]};
    const bool prepared=thinHalo && !hybrid && env_uint("D4R_NATIVE_MV_PREPARED",0)==1 &&
        g.cu.composeNativeMotionPrepared && g.cu.stabilizeThinWiresPrepared &&
        ensure_linear(feature.nativeThinMask,feature.nativeThinMaskBytes,size_t((feature.outWidth+31)/32)*feature.outHeight*4) &&
        ensure_linear(feature.preparedWireAxes,feature.preparedWireAxesBytes,size_t(feature.outWidth)*feature.outHeight);
    const D4rNativeMotionPreparedParams preparedArgs={thinArgs,feature.nativeThinMask,feature.preparedWireAxes,wire_history_flags(),0};
    const bool measured=begin_native_pass_profile(feature,0,frame,timing);
    const int result=prepared ? g.cu.composeNativeMotionPrepared(&preparedArgs) : hybrid ? g.cu.composeNativeMotionHybrid(&hybridArgs) : thinHalo ? g.cu.composeNativeMotionThin(&thinArgs) :
        g.cu.composeNativeMotionAligned ? g.cu.composeNativeMotionAligned(&thinArgs) : g.cu.composeNativeMotion(&args);
    if(result!=0) { logf("native live frame %u: compose failed %d",frame,result);return 0; }
    feature.preparedWireAxesReady=prepared;feature.preparedWireFlags=preparedArgs.wire_flags;
    if(measured && g.cu.eventRecord(feature.nativePassEvents[1],nullptr)==0) timing->nativeMotionTimed=true;
    if(frame<=3 || frame%300==1) logf("frame %u: native motion field=%s",frame,prepared ? "prepared full with halo" : hybrid ? "wire-local hybrid" : thinHalo ? "full with halo" : "aligned 3x3");
    return slot.nativeGpuTexture;
}

// Worker-only output history. Copies and full kernel share NGX's stream. Only
// current native sky silhouettes with recent depth-matched color can be repaired.

static bool ensure_wire_buffers(Feature& feature)
{
    const uint64_t pixels=uint64_t(feature.outWidth)*feature.outHeight;
    const size_t sizes[8]={size_t(pixels*8),size_t(pixels*8),size_t(pixels*8),size_t(pixels*8),size_t(pixels),size_t(pixels),size_t(pixels*4),size_t(pixels*4)};
    if(feature.wireWidth!=feature.outWidth || feature.wireHeight!=feature.outHeight)
    {
        if(synchronize_default_stream(feature)!=0) return false;
        for(auto& buffer:feature.wireBuffers) { if(buffer) g.cu.memFree(buffer);buffer=0; }
        if(feature.wirePatchMask) g.cu.memFree(feature.wirePatchMask),feature.wirePatchMask=0;
        feature.wireWidth=feature.outWidth;feature.wireHeight=feature.outHeight;feature.wireValid=false;
    }
    for(unsigned i=0;i<8;i++) if(!feature.wireBuffers[i] && g.cu.memAlloc(&feature.wireBuffers[i],sizes[i])!=0)
    { feature.wireValid=false;return false; }
    return true;
}

// Split-slot reuse is already ordered behind this frame's input marker.
// Different pitches and packed outputs use the ordinary filtered-buffer path.

static CudaDevicePtr wire_direct_target(Feature& feature,const FrameParams& p,uint32_t frame)
{
    const size_t pitch=size_t(feature.outWidth)*8,bytes=pitch*feature.outHeight;
    if(!p.vram || !p.split || p.packedOutput) return 0;
    if(p.outputInPlace) return p.outputInPlacePitch==pitch ? p.outputInPlace : 0;
    const VramBuffer& target=feature.outputs[frame%kOutputSlots].vram;
    return target.bytes>=bytes ? target.device : 0;
}

static bool stabilize_wire_output(Feature& feature, InputSlot& slot, const FrameParams& p, uint32_t frame, bool inputRedirected, CudaDevicePtr sparseTarget, FrameTiming* timing)
{
    if(!g.cu.stabilizeThinWires || !slot.linearPointer[0] || !slot.nativeGpuInputs[0].device ||
        !slot.nativeGpuResult || feature.outWidth<33 || feature.outHeight<33) return false;
    if(!ensure_wire_buffers(feature)) return false;
    const unsigned current=feature.wireIndex,previous=1-current;
    const size_t pitch=size_t(feature.outWidth)*8;
    CudaMemcpy2D copy={};copy.srcMemoryType=CUDA_MEMORY_ARRAY;copy.srcArray=feature.output.array;
    copy.dstMemoryType=CUDA_MEMORY_DEVICE;copy.dstDevice=feature.wireBuffers[0];copy.dstPitch=pitch;
    copy.WidthInBytes=pitch;copy.Height=feature.outHeight;
    if(!inputRedirected && copy_2d(copy)!=0) { feature.wireValid=false;return false; }
    const D4rWireHistoryParams args={sparseTarget ? sparseTarget : feature.wireBuffers[0],slot.linearPointer[0],slot.nativeGpuInputs[0].device,slot.nativeGpuResult,
        feature.wireBuffers[2+previous],feature.wireBuffers[4+previous],feature.wireBuffers[6+previous],
        feature.wireBuffers[1],feature.wireBuffers[4+current],feature.wireBuffers[6+current],slot.linearPointer[1],feature.wireBuffers[2+current],
        slot.linearPitch[0],uint64_t(feature.outWidth)*4,slot.nativeTexPitch,pitch,slot.linearPitch[1],
        feature.outWidth,feature.outHeight,p.renderWidth,p.renderHeight,
        feature.wireValid && feature.wireLastFrame+1==frame && !p.reset && feature.wireRenderWidth==p.renderWidth &&
        feature.wireRenderHeight==p.renderHeight && feature.wirePreExposure==p.preExposure && feature.wireExposureScale==p.exposureScale ? 1u : 0u,p.jitterX,p.jitterY,wire_history_flags()};
    const D4rWireHistorySparseParams sparse={args,feature.wirePatchMask};
    const D4rWireHistoryPreparedParams cached={sparse,feature.preparedWireAxes};
    const bool preclassified=sparseTarget && feature.preparedWireAxesReady && feature.preparedWireFlags==args.reserved && g.cu.stabilizeThinWiresPrepared;
    const bool measured=begin_native_pass_profile(feature,2,frame,timing);
    const int status=preclassified ? g.cu.stabilizeThinWiresPrepared(&cached) : sparseTarget ? g.cu.stabilizeThinWiresSparse(&sparse) : g.cu.stabilizeThinWires(&args);
    if(status!=0) { feature.wireValid=false;return false; }
    if(measured && g.cu.eventRecord(feature.nativePassEvents[3],nullptr)==0) timing->wireRepairTimed=true;
    feature.wirePublishedDirect=sparseTarget!=0;
    feature.wirePublishBuffer=sparseTarget ? sparseTarget : feature.wireBuffers[1];
    if(p.vram && !p.packedOutput && env_uint("D4R_NATIVE_MV_WIRE_DIRECT",1)==1)
        feature.wireOutputReady=true; // publish directly, without copying through the NGX array
    else
    {
        copy={};copy.srcMemoryType=CUDA_MEMORY_DEVICE;copy.srcDevice=feature.wireBuffers[1];copy.srcPitch=pitch;
        copy.dstMemoryType=CUDA_MEMORY_ARRAY;copy.dstArray=feature.output.array;copy.WidthInBytes=pitch;copy.Height=feature.outHeight;
        if(copy_2d(copy)!=0) { feature.wireValid=false;return false; }
    }
    feature.wireValid=true;feature.wireLastFrame=frame;feature.wireIndex=previous;
    feature.wireRenderWidth=p.renderWidth;feature.wireRenderHeight=p.renderHeight;
    feature.wirePreExposure=p.preExposure;feature.wireExposureScale=p.exposureScale;
    if(frame<=3 || frame%300==1) logf("frame %u: native wire color history applied, valid=%u input=%s publish=%s geometry=%s",
        frame,args.history_valid,inputRedirected ? "direct" : "array-copy",feature.wirePublishedDirect ? "sparse-direct" : feature.wireOutputReady ? "linear" : "array-copy",preclassified ? "prepared" : "recomputed");
    return true;
}

// Worker. MVLowRes (flag 2) describes the motion plane's resolution, and NGX fixes it at creation. A live native
// field is output-sized, so the feature is recreated without the flag (history is lost: reset) and recreated with
// the game's own flags when the game's vectors are used again. Returns true if the feature now matches the request.

static bool set_native_motion_flags(Feature& feature, FrameParams& params, bool native, uint32_t frame)
{
    const int desired=native ? (feature.flags & ~2) : feature.flags;
    if(desired==feature.cudaFlags) return feature.cudaHandle!=nullptr;
    logf("frame %u: native motion %s: recreating the NGX feature, flags 0x%x -> 0x%x",frame,native ? "active" : "unavailable",
         feature.cudaFlags,desired);
    synchronize_default_stream(feature);
    if(feature.cudaHandle!=nullptr) g.ngx.releaseFeature(feature.cudaHandle);
    feature.cudaHandle=nullptr;
    void* p=feature.cudaParams;
    d4r_ngx_set_uint(p,"Width",feature.cudaWidth);
    d4r_ngx_set_uint(p,"Height",feature.cudaHeight);
    d4r_ngx_set_int(p,"DLSS.Feature.Create.Flags",desired);
    NgxResult created=g.ngx.createFeature(NGX_FEATURE_SUPER_SAMPLING,p,&feature.cudaHandle);
    if(created!=NGX_SUCCESS)
    {
        logf("frame %u: recreating the NGX feature with flags 0x%x failed (0x%08x); restoring 0x%x",frame,desired,created,feature.cudaFlags);
        feature.cudaHandle=nullptr;
        d4r_ngx_set_int(p,"DLSS.Feature.Create.Flags",feature.cudaFlags);
        created=g.ngx.createFeature(NGX_FEATURE_SUPER_SAMPLING,p,&feature.cudaHandle);
        if(created!=NGX_SUCCESS) { feature.cudaHandle=nullptr;logf("frame %u: restoring the NGX feature failed (0x%08x)",frame,created);return false; }
        params.reset=1;d4r_ngx_set_int(p,"Reset",1);
        return false; // the previous flags are back; this frame cannot use the new field
    }
    feature.cudaFlags=desired;
    params.reset=1;
    d4r_ngx_set_int(p,"Reset",1);
    return true;
}

static void run_evaluation(Feature* feature, int slotIndex, uint32_t frame, FrameParams params, FrameTiming timing)
{
    InputSlot& slot = feature->inputs[slotIndex];
    const auto start = ProfileClock::now();
    if (timing.enabled)
        timing.workerWait = profile_ms(timing.workerQueued, start);
    FrameTiming* stages = timing.enabled ? &timing : nullptr;
    const bool overlapped = async_copies() && copy_streams() != nullptr;
    if (feature->ngxNativeCompletion && g.cu.timelinePrepare(feature->ngxTimeline) != 0)
    {
        feature->ngxNativeCompletion = false;
        logf("NGX native completion still busy; using CPU output completion from now on");
    }
    int eventStartResult = -1;
    if (timing.markerDeferred)
    {
        const auto markerStart = ProfileClock::now();
        if (ngx_gpu_wait_eligible(*feature, slot, params, frame))
        {
            complete_pending_download(); // earlier host staging must not synchronize against our new wait
            // Mint textures/arrays before blocking the null stream. Lazy allocation can synchronize
            // internally; it must not be part of the steady-state input-readiness submission window.
            bool ready = feature->linearInputs ? ensure_linear_inputs(slot, params.hasExposure) : true;
            if (!feature->linearInputs)
            {
                const Plane planes[] = {Plane::Color, Plane::Depth, Plane::Motion, Plane::Exposure};
                CudaImage* images[] = {&feature->color, &feature->depth, &feature->motion, &feature->exposure};
                for (int index = 0; index < (params.hasExposure ? 4 : 3) && ready; ++index)
                    ready = ensure_cuda_image(*images[index], slot.host[index].width, slot.host[index].height,
                        plane_format(planes[index]), plane_channels(planes[index]), false, index == 0);
            }
            ready = ready && ensure_cuda_image(feature->output, feature->outputWidth, feature->outputHeight,
                                               CUDA_FORMAT_HALF, 4, true, false);
            if (!ready)
            {
                slot.busy = false;
                frame_retired(feature, frame);
                return;
            }
            if (!feature->ngxWaitEventsWarmed)
            {
                if (feature->outputReadyEvent != nullptr)
                    g.cu.eventRecord(feature->outputReadyEvent, nullptr);
                if (timing.enabled && feature->profileEventsReady)
                    g.cu.eventRecord(feature->profileEnd, nullptr);
                feature->ngxWaitEventsWarmed = true;
            }
            eventStartResult = timing.enabled && feature->profileEventsReady
                                   ? g.cu.eventRecord(feature->profileStart, nullptr) : -1;
            {
                std::lock_guard<std::mutex> lock(feature->ngxWaitMutex);
                feature->ngxWaitActive = true;
                feature->ngxWaitCancelled = false;
                feature->ngxWaitFrame = frame;
                feature->ngxWaitSlot = slotIndex;
                feature->ngxWaitStart = ProfileClock::now();
            }
            // Arm before the call: a blocked HIP enqueue/NGX evaluate must never prevent timeout release.
            const int queued = g.cu.streamWaitValue32Eq(
                feature->ngxInputSync.device + sizeof(uint32_t) * (1 + slotIndex), frame);
            if (queued == 0)
            {
                timing.gpuInputWait = true;
                // Profiling only: a timing event behind the wait completes when the game's input word is released,
                // so it splits gpu_eval into input readiness and DLSS's own work. Recording queues one marker on
                // the null stream and never waits on the CPU; elapsed times are read after output completion.
                timing.inputEventRecorded = eventStartResult == 0 && feature->profileInputsReleased != nullptr &&
                                            g.cu.eventRecord(feature->profileInputsReleased, nullptr) == 0;
                bool cancelled;
                {
                    std::lock_guard<std::mutex> lock(feature->ngxWaitMutex);
                    cancelled = feature->ngxWaitCancelled;
                }
                if (cancelled)
                {
                    complete_ngx_input_wait(*feature, frame, true);
                    slot.busy = false;
                    frame_retired(feature, frame);
                    return;
                }
            }
            else
            {
                // Release and drain even on rejection: a partially submitted driver operation must
                // never survive a switch back to CPU polling. Do not hold the ticket lock over HIP.
                complete_ngx_input_wait(*feature, frame, true);
                feature->ngxGpuWait.store(false);
                logf("NGX feature %u: GPU-side input wait unavailable (%d); using CPU marker polling from now on",
                     feature->handle.Id, queued);
            }
        }
        if (!timing.gpuInputWait)
        {
            if (!wait_for_input_marker(feature, slotIndex, frame))
                return;
            if (timing.enabled)
                timing.markerWait = profile_ms(markerStart, ProfileClock::now());
        }
    }
    if (slot.nativeMotionCaptureFrame == frame)
    {
        // A CPU diagnostic readback needs an observed marker even when NGX uses a GPU wait.
        if (!wait_for_input_marker(feature, slotIndex, frame))
            return;
        if (slot.nativeGeometryCaptureReady)
        {
            const Staging& staging = slot.nativeMotionCapture;
            const size_t rowBytes = static_cast<size_t>(staging.width) * 4;
            std::vector<uint8_t> tight(rowBytes * staging.height);
            for (UINT row = 0; row < staging.height; ++row)
                std::memcpy(tight.data() + row * rowBytes,
                    staging.mapped + staging.layout.Offset + row * staging.layout.Footprint.RowPitch, rowBytes);
            write_capture_file(frame, "native geometry motion", slot.nativeMotionCapturePath.c_str(), tight.data(), tight.size());
        }
        const std::string base = slot.nativeMotionCapturePath.substr(0, slot.nativeMotionCapturePath.find(".geometry.rg16f"));
        if(slot.nativePrimaryDepthCaptureReady)
        {
            const Staging& z=slot.nativePrimaryDepthCapture;
            std::vector<uint8_t> data(static_cast<size_t>(z.width)*z.height*4);
            for(UINT y=0;y<z.height;y++) std::memcpy(data.data()+static_cast<size_t>(y)*z.width*4,
                z.mapped+z.layout.Offset+y*z.layout.Footprint.RowPitch,z.width*4);
            write_capture_file(frame,"native primary hardware depth",(base+".primary-depth.r32f").c_str(),data.data(),data.size());
        }
        if(slot.opaqueSceneCaptureReady)
            write_capture_file(frame,"opaque scene constants",(base+".opaque-scene.f32").c_str(),
                slot.opaqueSceneCapture.mapped,static_cast<size_t>(slot.opaqueSceneCapture.total));
        if(slot.opaqueMatrixCaptureReady)
            write_capture_file(frame,"opaque pixel reprojection matrix",(base+".opaque-matrix.f32").c_str(),
                slot.opaqueMatrixCapture.mapped,static_cast<size_t>(slot.opaqueMatrixCapture.total));
        if(slot.materialSceneCaptureReady)
            write_capture_file(frame,"owned material scene at its draw",(base+".material-owned-scene.f32").c_str(),slot.materialSceneCapture.mapped,1136);
        if(slot.glassMaskCaptureReady)
        {
            const Staging& mask=slot.glassMaskCapture;
            write_capture_file(frame,"material motion branch mask",(base+".glass-mask.raw").c_str(),mask.mapped,static_cast<size_t>(mask.total));
            char metadata[128];const int size=std::snprintf(metadata,sizeof(metadata),"%u %u %u\n",mask.width,mask.height,slot.glassMaskFormat);
            if(size>0 && size<static_cast<int>(sizeof(metadata))) write_capture_file(frame,"material mask geometry/format",(base+".glass-mask.txt").c_str(),metadata,size);
        }
        if(slot.originalOpaqueCopyCaptureReady)
        {
            const Staging& st=slot.originalOpaqueCopyCapture;
            std::vector<uint8_t> data(static_cast<size_t>(st.width)*st.height*4);
            for(UINT y=0;y<st.height;y++) std::memcpy(data.data()+static_cast<size_t>(y)*st.width*4,
                st.mapped+st.layout.Offset+y*st.layout.Footprint.RowPitch,st.width*4);
            write_capture_file(frame,"original opaque copy before material overlay",(base+".original-opaque-copy.rg16f").c_str(),data.data(),data.size());
        }
        if(slot.ownedValidityCaptureReady)
        {
            const GUID availableGuid={0x75367844,0x817a,0x4c96,{0x97,0x35,0x2c,0xa1,0x7a,0x99,0x48,0x32}};
            uint32_t available=0;UINT bytes=sizeof(available);
            if(!slot.nativeValidityOwner || FAILED(slot.nativeValidityOwner->GetPrivateData(availableGuid,&bytes,&available)) || bytes!=sizeof(available)) available=0;
            write_capture_file(frame,"GPU-owned validity",(base+".owned-validity.u32").c_str(),slot.ownedValidityCapture.mapped,8);
            write_capture_file(frame,"validity availability after input marker",(base+".owned-validity-status.u32").c_str(),&available,4);
            const uint32_t* words=reinterpret_cast<const uint32_t*>(slot.ownedValidityCapture.mapped);
            slot.nativeGpuStageValid=available==1 && words[0]==1 && words[1]==1;
            logf("native motion capture frame %u: GPU validity opaque/depth=%u/%u post-marker availability=%u",frame,words[0],words[1],available);
        }
        if(slot.nativeValidityOwner) { slot.nativeValidityOwner->Release();slot.nativeValidityOwner=nullptr; }
        if(slot.glassValidityCaptureReady)
            write_capture_file(frame,"glass depth GPU validity",(base+".glass-depth-validity.u32").c_str(),slot.glassValidityCapture.mapped,4);
        if(slot.ownedDepthCaptureReady)
        {
            const Staging& z=slot.ownedDepthCapture;
            std::vector<uint8_t> data(static_cast<size_t>(z.width)*z.height*4);
            for(UINT y=0;y<z.height;y++) std::memcpy(data.data()+static_cast<size_t>(y)*z.width*4,
                z.mapped+z.layout.Offset+y*z.layout.Footprint.RowPitch,z.width*4);
            write_capture_file(frame,"owned primary depth",(base+".owned-depth.r32f").c_str(),data.data(),data.size());
        }
        if(slot.ownedSceneCaptureReady)
            write_capture_file(frame,"owned opaque scene",(base+".owned-scene.f32").c_str(),
                slot.ownedSceneCapture.mapped,static_cast<size_t>(slot.ownedSceneCapture.total));
        if(slot.ownedMatrixCaptureReady)
            write_capture_file(frame,"owned opaque matrix",(base+".owned-matrix.f32").c_str(),
                slot.ownedMatrixCapture.mapped,static_cast<size_t>(slot.ownedMatrixCapture.total));
        if(slot.originalPrimaryDepthCaptureReady)
            write_capture_file(frame,"original primary hardware depth",(base+".original-primary-depth.r32f").c_str(),
                slot.originalPrimaryDepthCapture.mapped,static_cast<size_t>(slot.originalPrimaryDepthCapture.total));
        if (slot.nativeDepthCaptureReady)
        {
            const Staging& depth = slot.nativeDepthCapture;
            std::vector<uint8_t> z(static_cast<size_t>(depth.width) * depth.height * 4);
            for (UINT row = 0; row < depth.height; ++row)
                std::memcpy(z.data() + static_cast<size_t>(row) * depth.width * 4,
                    depth.mapped + depth.layout.Offset + row * depth.layout.Footprint.RowPitch, depth.width * 4);
            write_capture_file(frame, "native hardware depth", (base + ".depth.r32f").c_str(), z.data(), z.size());
        }
        if (slot.nativeSceneCaptureReady)
            write_capture_file(frame, "motion composition constants", (base + ".scene.f32").c_str(),
                slot.nativeSceneCapture.mapped, static_cast<size_t>(slot.nativeSceneCapture.total));

        if (slot.nativeOpaqueCaptureReady)
        {
            const Staging& opaque = slot.nativeOpaqueCapture;
            const size_t row = static_cast<size_t>(opaque.width) * 4;
            std::vector<uint8_t> data(row * opaque.height);
            for (UINT y = 0; y < opaque.height; ++y)
                std::memcpy(data.data() + y * row,opaque.mapped + opaque.layout.Offset + y * opaque.layout.Footprint.RowPitch,row);
            write_capture_file(frame,"native opaque per-object motion",(base+".opaque-geometry.rg16f").c_str(),data.data(),data.size());
        }
        if (slot.originalGeometryCaptureReady)
            write_capture_file(frame,"original geometry motion",(base+".original-geometry.rg16f").c_str(),
                slot.originalGeometryCapture.mapped,static_cast<size_t>(slot.originalGeometryCapture.total));
        if (slot.originalHardwareDepthCaptureReady)
            write_capture_file(frame,"original hardware depth",(base+".original-hardware-depth.r32f").c_str(),
                slot.originalHardwareDepthCapture.mapped,static_cast<size_t>(slot.originalHardwareDepthCapture.total));
    }
    if (slot.nativeMotionCaptureFrame != frame && slot.ownedValidityCaptureReady && slot.nativeValidityOwner)
    {
        // Live frame (no diagnostic capture): the validity words and the post-marker availability are only
        // meaningful once this frame's own input marker has been observed.
        bool markerSeen = true;
        if (slot.nativeLive)
        {
            markerSeen = wait_for_input_marker(feature, slotIndex, frame);
            if (markerSeen)
            {
                const GUID availableGuid = {0x75367844, 0x817a, 0x4c96, {0x97, 0x35, 0x2c, 0xa1, 0x7a, 0x99, 0x48, 0x32}};
                uint32_t available = 0;
                UINT bytes = sizeof(available);
                if (FAILED(slot.nativeValidityOwner->GetPrivateData(availableGuid, &bytes, &available)) || bytes != sizeof(available))
                    available = 0;
                const uint32_t* words = reinterpret_cast<const uint32_t*>(slot.nativeValidityOwner ? slot.ownedValidityCapture.mapped : nullptr);
                slot.nativeGpuStageValid = words != nullptr && available == 1 && words[0] == 1 && words[1] == 1;
                if (!slot.nativeGpuStageValid && frame % 300 == 1)
                    logf("native motion validity frame %u: available=%u opaque=%u depth=%u", frame, available, words ? words[0] : 0, words ? words[1] : 0);
            }
        }
        slot.nativeValidityOwner->Release();
        slot.nativeValidityOwner = nullptr;
        if (!markerSeen)
            return;
    }
    if (slot.nativeLive)
        ; // composed below, after any feature flag change, on NGX's stream
    else
        run_native_gpu_check(slot,params,frame);
    if (!timing.gpuInputWait && params.vram && env_uint("D4R_SHIM_VRAM_VERIFY", 0) == 0)
        dump_vram_inputs(slot, frame, params, params.hasExposure ? 4 : 3);
    const bool linearInputs = params.vram && feature->linearInputs;
    bool ok = linearInputs ? ensure_linear_inputs(slot, params.hasExposure)
              : params.vram ? upload_inputs_vram(*feature, slot, params.hasExposure, frame, stages, timing.gpuInputWait)
              : overlapped
                  ? upload_inputs_async(*feature, slot, params.hasExposure, frame, stages)
                  : upload_plane(Plane::Color, slot.host[0], feature->color, frame, stages, 0) &&
                        upload_plane(Plane::Depth, slot.host[1], feature->depth, frame, stages, 1) &&
                        upload_plane(Plane::Motion, slot.host[2], feature->motion, frame, stages, 2) &&
                        (!params.hasExposure ||
                         upload_plane(Plane::Exposure, slot.host[3], feature->exposure, frame, stages, 3));
    ok = ok && ensure_cuda_image(feature->output, feature->outputWidth, feature->outputHeight, CUDA_FORMAT_HALF, 4,
                                 true, false);
    const auto dilationStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    // Keep the native field available to repair independently of the field fed
    // to NGX. The opt-in repair-only mode retains the game's flags and MV scale.
    CudaObject nativeMotion = 0;
    CudaObject liveMotion = 0;
    if (ok && (slot.nativeLive || feature->cudaFlags != feature->flags))
    {
        nativeMotion = compose_native_live(*feature, slot, params, frame, stages);
        liveMotion = env_uint("D4R_NATIVE_MV_SUBSTITUTE",1)==1 ? nativeMotion : 0;
        const bool matched = set_native_motion_flags(*feature, params, liveMotion != 0, frame);
        if (liveMotion != 0 && !matched)
        {
            liveMotion = 0;
            nativeMotion = 0; // preserve the existing raw-output fallback when substitution fails
        }
        if (feature->cudaHandle == nullptr || (liveMotion == 0 && feature->cudaFlags != feature->flags))
            ok = false; // Never evaluate low-res vectors using a restored high-res-only feature.
        if (liveMotion != 0)
        {
            // The field is in output pixels: MV.Scale (render pixels) follows it.
            params.mvScaleX = static_cast<float>(static_cast<double>(params.mvScaleX) * feature->outWidth / params.renderWidth);
            params.mvScaleY = static_cast<float>(static_cast<double>(params.mvScaleY) * feature->outHeight / params.renderHeight);
        }
        static std::atomic<unsigned> liveFrames{0}, fallbackFrames{0};
        const unsigned live = liveMotion != 0 ? ++liveFrames : liveFrames.load();
        const unsigned fallback = liveMotion == 0 ? ++fallbackFrames : fallbackFrames.load();
        if (frame <= 3 || (liveMotion != 0 ? live : fallback) <= 4 || (live + fallback) % 300 == 0)
            logf("frame %u: native motion %s (live %u, fallback %u, mvScale %.3f,%.3f, flags 0x%x)", frame,
                 liveMotion != 0 ? "SUBSTITUTED" : "not used", live, fallback, params.mvScaleX, params.mvScaleY,
                 feature->cudaFlags);
    }
    const CudaObject dilatedMotion = ok && liveMotion == 0 ? prepare_dilated_motion(*feature, slot, params) : 0;
    // Ray Reconstruction: the auxiliary surfaces are copied in the same window as the four
    // main planes, so the one synchronize below covers them as well. The lock is taken only
    // around the copies and, later, around the evaluate - never across the synchronize, which
    // is the one long wait here and would otherwise stall the game thread's next gather.
    bool auxUploaded = false;
    if (ok && feature->rayReconstruction)
    {
        {
            std::lock_guard<std::mutex> lock(feature->auxMutex);
            auxUploaded = upload_aux_inputs(*feature, slotIndex, frame);
            if (auxUploaded)
                dump_aux_guides(*feature, slotIndex, frame, params);
        }
        if (!auxUploaded)
        {
            complete_ngx_input_wait(*feature, frame, true);
            slot.busy = false;
            frame_retired(feature, frame);
            return;
        }
    }
    // Device-to-array copies (VRAM interop, staged uploads) return before they
    // finish. Staged copies can use another stream, so without their completion
    // DLSS can sample a mix of this frame's and the previous
    // frame's colour/depth/motion: invisible when the camera is still, beaded
    // thin geometry in motion. D4R_SHIM_INPUT_SYNC=0 restores the old behaviour.
    static const bool inputSync = env_uint("D4R_SHIM_INPUT_SYNC", 1) != 0;
    // Linear inputs are ready after the marker. The GPU-wait route orders all copies and NGX on
    // the null stream itself, so its input-sync CPU wait would only put readiness back on the critical path.
    // Dilation is submitted after the CPU marker or GPU input wait and before
    // NGX, on the same bridge-selected stream. It needs no CPU completion wait.
    if (ok && !timing.gpuInputWait && (auxUploaded || (inputSync && !linearInputs)))
    {
        const int syncResult = synchronize_default_stream(*feature);
        if (syncResult != 0)
        {
            logf("frame %u: input copy synchronize failed: %d", frame, syncResult);
            ok = false;
        }
    }
    if (dilatedMotion != 0 && ok)
    {
        params.motionDilation = env_uint("D4R_MOTION_DILATION", 0);
        if (timing.enabled)
            timing.motionDilationWall = profile_ms(dilationStart, ProfileClock::now());
    }
    slot.hostBusy = false; // the host planes are no longer needed
    // The previous frame's download ran alongside these uploads.
    complete_pending_download();
    if (!ok)
    {
        complete_ngx_input_wait(*feature, frame, true);
        slot.busy = false;
        frame_retired(feature, frame);
        return;
    }

    // The first frame NGX evaluates for a feature starts its history. Decided
    // here, in evaluation order, so it does not depend on thread timing.
    if (feature->evaluatedFrames++ == 0)
        params.reset = 1;
    set_evaluation_parameters(*feature, params);
    // Held across the denoiser's evaluate too: the parameter object it is reading names
    // &auxSurface.handle, which the game thread must not be able to move or erase meanwhile.
    std::unique_lock<std::mutex> auxLock(feature->auxMutex, std::defer_lock);
    if (feature->rayReconstruction)
    {
        auxLock.lock();
        publish_aux_parameters(*feature, params, slot);
        if (!publish_alpha_output(*feature, slot))
        {
            auxLock.unlock();
            complete_ngx_input_wait(*feature, frame, true);
            slot.busy = false;
            frame_retired(feature, frame);
            return;
        }
    }
    if (linearInputs)
    {
        // the parameters point at these variables
        feature->colorHandle = slot.linearTexture[0];
        feature->depthHandle = slot.linearTexture[1];
        feature->motionHandle = liveMotion != 0 ? liveMotion : dilatedMotion != 0 ? dilatedMotion : slot.linearTexture[2];
        feature->exposureHandle = params.hasExposure ? slot.linearTexture[3] : feature->exposureHandle;
    }
    const bool wireHistoryRequested=nativeMotion!=0 && g.cu.stabilizeThinWires &&
        env_uint("D4R_NATIVE_MV_WIRE_HISTORY",1)==1;
    timing.nativeFieldReady=nativeMotion!=0;
    timing.nativeSubstituted=liveMotion!=0;
    timing.wireRequested=wireHistoryRequested;
    feature->wireOutputReady=false;feature->wirePublishedDirect=false;feature->wirePublishBuffer=0;
    if(frame<=3 || frame%300==1)
        logf("frame %u: motion routing native_ready=%u ngx=%s wire_requested=%u halo=%u",
             frame,nativeMotion!=0,liveMotion!=0 ? "native" : "game",wireHistoryRequested,
             env_uint("D4R_NATIVE_MV_THIN_HALO",1)==1);
    if(!wireHistoryRequested) feature->wireValid=false;
    if(wireHistoryRequested && feature->outputRedirected)
    { g.cu.setArrayRedirect(feature->output.array,0,0);feature->outputRedirected=false; }
    static const bool outputDirect = env_uint("D4R_SHIM_OUTPUT_DIRECT", 0) != 0;
    // The redirect is carried out by the native output kernel's stores (ZLUDA's compile of NVIDIA's kernel
    // writes the array), so it is set while the previous frame's output kernel was the native one and
    // undone below when this frame's was not.
    const bool nativeOutput = g.cu.outputKernelNative == nullptr || g.cu.outputKernelNative() == 1;
    if (!wireHistoryRequested && outputDirect && feature->outputDirectAllowed && g.cu.setArrayRedirect != nullptr && feature->output.array != nullptr &&
        (nativeOutput || feature->outputRedirected))
    {
        // split frames write their own slot; the staged path downloads from outputLinear
        size_t rowBytes = static_cast<size_t>(feature->output.width) * 8;
        const size_t bytes = rowBytes * feature->output.height;
        CudaDevicePtr target = 0;
        if (params.packedOutput)
            rowBytes = static_cast<size_t>(feature->output.width) * 4;
        if (params.vram && params.split && params.outputInPlace != 0)
        {
            target = params.outputInPlace;
            rowBytes = params.outputInPlacePitch;
        }
        else if (params.vram && params.split && feature->outputs[frame % kOutputSlots].vram.bytes >= bytes)
            target = feature->outputs[frame % kOutputSlots].vram.device;
        if (params.packedOutput)
            rowBytes |= params.packedRgba8 ? 2 : 1; // R4: RGBA8, R3: RGB10A2; row pitches are multiples of 8
        else if (!params.vram && overlapped && ensure_linear(feature->outputLinear, feature->outputLinearBytes, bytes))
            target = feature->outputLinear;
        const bool redirect = nativeOutput && target != 0 &&
                              g.cu.setArrayRedirect(feature->output.array, target, static_cast<uint32_t>(rowBytes)) == 0;
        if (!redirect && feature->outputRedirected)
            g.cu.setArrayRedirect(feature->output.array, 0, 0);
        static bool logged = false;
        if (redirect && !logged)
        {
            logf("frame %u: direct output (the native output kernel writes the shared buffer; no array copy)", frame);
            logged = true;
        }
        feature->outputRedirected = redirect;
    }
    // Keep full RGBA16F for the repair pass. Completion is signalled by publish_vram,
    // after the repair and final copy, never by NGX's earlier raw-output write.
    CudaDevicePtr wireSparseTarget=0;
    if(wireHistoryRequested && g.cu.stabilizeThinWiresSparse && env_uint("D4R_NATIVE_MV_WIRE_SPARSE",1)==1 &&
        env_uint("D4R_NATIVE_MV_WIRE_DIRECT",1)==1 && ensure_wire_buffers(*feature))
    {
        if(!feature->wirePatchMask && g.cu.memAlloc(&feature->wirePatchMask,size_t(feature->outWidth)*feature->outHeight)!=0)
            feature->wirePatchMask=0;
        if(feature->wirePatchMask) wireSparseTarget=wire_direct_target(*feature,params,frame);
    }
    const bool wireInputRedirected=wireHistoryRequested && env_uint("D4R_NATIVE_MV_WIRE_DIRECT",1)==1 &&
        params.vram && !params.packedOutput &&
        outputDirect && feature->outputDirectAllowed && g.cu.outputKernelNative && nativeOutput && g.cu.setArrayRedirect &&
        ensure_wire_buffers(*feature) &&
        g.cu.setArrayRedirect(feature->output.array,wireSparseTarget ? wireSparseTarget : feature->wireBuffers[0],feature->outWidth*8)==0;
    if (!timing.gpuInputWait)
        eventStartResult = timing.enabled && feature->profileEventsReady
                              ? g.cu.eventRecord(feature->profileStart, nullptr) : -1;
    // GPU completion: with a redirected output the native output kernel is the last writer the game reads, so
    // the bridge may queue the completion word right behind it (it checks the kernel actually launched).
    if (params.gpuCompletion && feature->outputRedirected && g.cu.ngxArmCompletion != nullptr)
        g.cu.ngxArmCompletion(feature->ngxInputSync.device + sizeof(uint32_t) * (kCompletionWord + frame % kOutputSlots),
                              frame, params.packedRgba8 ? 1 : 0);
    const auto evalStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    // The core dispatches feature 13 to the denoiser it loaded, so this call is the same
    // either way; only the feature id that created it differs.
    NgxResult result = g.ngx.evaluateFeature(feature->cudaHandle, feature->cudaParams, nullptr);
    if (result == NGX_FAIL_INVALID_PARAMETER && recreate_for_render_size(*feature, params, frame))
        result = g.ngx.evaluateFeature(feature->cudaHandle, feature->cudaParams, nullptr);
    if (auxLock.owns_lock())
        auxLock.unlock();
    const bool wireInputWritten=wireInputRedirected && g.cu.outputKernelNative && g.cu.outputKernelNative()==1;
    if(wireInputRedirected) g.cu.setArrayRedirect(feature->output.array,0,0);
    feature->completionFired = !wireHistoryRequested && g.cu.ngxCompletionFired != nullptr && g.cu.ngxCompletionFired() == 1;
    if (feature->outputRedirected && ((g.cu.outputKernelNative != nullptr && g.cu.outputKernelNative() != 1) ||
        (params.packedRgba8 && (g.cu.outputKernelRgba8 == nullptr || g.cu.outputKernelRgba8() != 1))))
    {
        // this frame's output kernel ignored the redirect: publish from the array as usual
        g.cu.setArrayRedirect(feature->output.array, 0, 0);
        feature->outputRedirected = false;
    }
    feature->packedReady = feature->outputRedirected;
    feature->rgba8Ready = feature->outputRedirected && g.cu.outputKernelRgba8 != nullptr && g.cu.outputKernelRgba8() == 1;
    const auto evalReturned = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    const int eventEndResult = eventStartResult == 0 ? g.cu.eventRecord(feature->profileEnd, nullptr) : -1;
    const auto syncStart = timing.enabled ? ProfileClock::now() : ProfileClock::time_point{};
    // D4R_SHIM_EVAL_SYNC=0 (VRAM interop): no wait between NGX's kernels and the output copy queued
    // behind them on the same stream; publish_vram synchronises before the result is released.
    const int syncResult =
        params.vram && !eval_sync_enabled() ? 0 : synchronize_default_stream(*feature, nullptr, &feature->evalWaitEmaUs);
    const auto evaluated = ProfileClock::now();
    const bool gpuEventsRecorded = eventStartResult == 0 && eventEndResult == 0;
    float gpuEvalMs = -1.0f;
    if (gpuEventsRecorded && syncResult == 0 && (!params.vram || eval_sync_enabled()) &&
        g.cu.eventElapsedTime(&gpuEvalMs, feature->profileStart, feature->profileEnd) != 0)
        gpuEvalMs = -1.0f;
    // Keep the input slot owned until output completion, including linear reads.
    if (result != NGX_SUCCESS || syncResult != 0)
    {
        logf("frame %u: CUDA_EvaluateFeature -> 0x%08x, sync %d", frame, result, syncResult);
        complete_ngx_input_wait(*feature, frame, true);
        slot.busy = false;
        frame_retired(feature, frame);
        return;
    }
    if(wireHistoryRequested && !stabilize_wire_output(*feature,slot,params,frame,wireInputWritten,wireInputWritten ? wireSparseTarget : 0,stages))
    {
        feature->wireValid=false;
        // A redirected raw frame left the NGX array untouched. Restore the raw output
        // if repair could not be queued, so fallback never exposes an older array.
        feature->wireOutputReady=false;
        if(wireInputWritten && wireSparseTarget)
        {
            // Both sparse dispatches preserve raw current until apply; an apply
            // enqueue failure therefore leaves a valid raw frame in the game slot.
            feature->wireOutputReady=true;feature->wirePublishedDirect=true;feature->wirePublishBuffer=wireSparseTarget;
        }
        else if(wireInputWritten)
        {
            CudaMemcpy2D raw={};raw.srcMemoryType=CUDA_MEMORY_DEVICE;raw.srcDevice=feature->wireBuffers[0];
            raw.srcPitch=feature->outWidth*8;raw.dstMemoryType=CUDA_MEMORY_ARRAY;raw.dstArray=feature->output.array;
            raw.WidthInBytes=raw.srcPitch;raw.Height=feature->outHeight;
            if(copy_2d(raw)!=0)
            {
                complete_ngx_input_wait(*feature,frame,true);slot.busy=false;frame_retired(feature,frame);return;
            }
        }
        logf("frame %u: native wire history unavailable; original NGX output retained",frame);
    }
    // The alpha surface is complete once the synchronize above has passed. A failure here drops
    // the whole frame, exactly as a failed synchronize does: an accepted frame with an unwritten
    // alpha would leave the caller believing it was written.
    if (feature->rayReconstruction && !download_alpha_output(*feature, frame, params))
    {
        complete_ngx_input_wait(*feature, frame, true);
        slot.busy = false;
        frame_retired(feature, frame);
        return;
    }
    if (timing.enabled)
    {
        timing.ngxHost = profile_ms(evalStart, evalReturned);
        timing.ctxSync = profile_ms(syncStart, evaluated);
        timing.gpuEval = gpuEvalMs;
        if (gpuEvalMs >= 0.0f && !timing.gpuInputWait)
        {
            timing.dlssGpu = gpuEvalMs;
            timing.inputGpuWait = 0.0;
        }
        timing.gpuEventsRecorded = gpuEventsRecorded && params.vram && !eval_sync_enabled();
    }
    if (params.vram)
    {
        publish_vram(feature, frame, params, timing, start);
        slot.busy = false; // output synchronization completed all reads of this input slot
        return;
    }

    const int hostIndex = static_cast<int>(feature->nextOutputHost++ % 2);
    if (!wait_for_release(feature->outputHostBusy[hostIndex]))
    {
        logf("frame %u: output host buffer still busy; dropping result", frame);
        frame_retired(feature, frame);
        return;
    }
    if (overlapped)
    {
        if (!start_download(*feature, feature->outputHost[hostIndex], stages))
        {
            frame_retired(feature, frame);
            return;
        }
        feature->outputHostBusy[hostIndex] = true;
        if (timing.enabled)
            timing.worker = profile_ms(start, ProfileClock::now());
        g_pendingDownload = {feature, hostIndex, frame, params, timing};
        // Without a queued frame to overlap, finish it now rather than add latency.
        if (g.worker.pending() == 0)
            complete_pending_download(true);
        return;
    }
    if (!download_output(*feature, feature->outputHost[hostIndex], frame, stages))
    {
        frame_retired(feature, frame);
        return;
    }
    feature->outputHostBusy[hostIndex] = true;
    if (timing.enabled)
        timing.worker = profile_ms(start, ProfileClock::now());
    post_publish(feature, hostIndex, frame, params, timing);
}

// Waits for this frame's own input marker (the GPU reached the game's input copies). false: the frame was
// dropped (retired feature, skipped submission, timeout) and has been retired here.

bool wait_for_input_marker(Feature* feature, int slotIndex, uint32_t frame)
{
    InputSlot& slot = feature->inputs[slotIndex];
    const auto start = std::chrono::steady_clock::now();
    static const bool fixedOverride = GetEnvironmentVariableA("D4R_SHIM_MARKER_POLL_US", nullptr, 0) != 0;
    static const unsigned fixedPoll = env_uint("D4R_SHIM_MARKER_POLL_US", 200);
    const double predictedUs = feature->markerWaitEmaUs.load();
    double lastNotReadyUs = -1.0;
    // A later input marker does not prove this slot's copies were submitted:
    // the game may have discarded this frame's command list. Require its own
    // tag, otherwise stale input data can enter DLSS's temporal history.
    const volatile uint32_t* inputMarker = feature->markerValue + 1 + slotIndex;
    while (*inputMarker != frame || feature->retiring.load())
    {
        std::atomic_thread_fence(std::memory_order_acquire);
        const bool skipped = static_cast<int32_t>(*feature->markerValue - frame) > 0;
        // The GPU may have filled our slot between the loop condition and
        // the aggregate-marker read. Recheck before deciding it was skipped.
        if (*inputMarker == frame && !feature->retiring.load())
            break;
        if (feature->retiring.load() || skipped || std::chrono::steady_clock::now() - start > std::chrono::seconds(5))
        {
            logf("frame %u: %s waiting for GPU marker (at %u); dropping", frame,
                 feature->retiring.load() ? "feature retired while" : skipped ? "input submission skipped while" : "timed out",
                 *inputMarker);
            slot.busy = false;
            frame_retired(feature, frame);
            return false;
        }
        lastNotReadyUs = std::chrono::duration<double, std::micro>(ProfileClock::now() - start).count();
        // An explicit N retains fixed pacing (0 yields). Otherwise learn readiness per feature, sleep
        // in <=200 us chunks, yield near prediction, and back off when late, as the output wait does.
        d4r_sleep_us(fixedOverride ? fixedPoll
                                  : d4r_marker_poll_delay_us(predictedUs, lastNotReadyUs));
    }
    if (!fixedOverride)
    {
        const double waitedUs = std::chrono::duration<double, std::micro>(ProfileClock::now() - start).count();
        // Estimate readiness between the last not-ready read and this one, not the end of our sleep.
        const double sample = std::min(lastNotReadyUs >= 0.0 ? 0.5 * (lastNotReadyUs + waitedUs) : waitedUs, 20000.0);
        feature->markerWaitEmaUs.store(predictedUs == 0.0 ? sample : 0.8 * predictedUs + 0.2 * sample);
    }
    return true;
}

void prepare_inputs(Feature* feature, int slotIndex, uint32_t frame, FrameParams params, FrameTiming timing)
{
    InputSlot& slot = feature->inputs[slotIndex];
    const auto start = ProfileClock::now();
    // D4R_SHIM_DEFER_MARKER (default on): the worker waits for the game's copies itself, avoiding a thread
    // wake-up between readiness and evaluation. The wait must precede every input read, including CUDA
    // uploads and diagnostic readbacks; waiting only before NGX would copy stale data into its arrays.
    static const bool deferMarker = env_uint("D4R_SHIM_DEFER_MARKER", 1) != 0;
    if (params.vram && (deferMarker || ngx_gpu_wait_eligible(*feature, slot, params, frame)) &&
        env_uint("D4R_SHIM_VRAM_VERIFY", 0) == 0)
    {
        timing.markerDeferred = true;
        if (timing.enabled)
        {
            timing.prepStart = start;
            timing.prep = profile_ms(start, ProfileClock::now());
            timing.workerQueued = ProfileClock::now();
        }
        g.worker.post([feature, slotIndex, frame, params, timing] {
            run_evaluation(feature, slotIndex, frame, params, timing);
        });
        return;
    }
    if (!wait_for_input_marker(feature, slotIndex, frame))
        return;
    const auto ready = ProfileClock::now();
    // With VRAM interop the inputs are already on the GPU; host staging runs
    // only to cross-check them (D4R_SHIM_VRAM_VERIFY).
    if (!params.vram || env_uint("D4R_SHIM_VRAM_VERIFY", 0) != 0)
    {
        if (!wait_for_release(slot.hostBusy))
        {
            logf("frame %u: host planes of slot %d still busy; dropping", frame, slotIndex);
            slot.busy = false;
            frame_retired(feature, frame);
            return;
        }
        FrameTiming* stages = timing.enabled ? &timing : nullptr;
        // Verification must not replace a pitch-linear texture's padded GPU stride
        // with the tightly packed host stride.
        stage_plane(Plane::Color, slot.color, slot.host[0], stages, 0, params.vram ? slot.host[0].rowBytes : 0);
        stage_plane(Plane::Depth, slot.depth, slot.host[1], stages, 1, params.vram ? slot.host[1].rowBytes : 0);
        stage_plane(Plane::Motion, slot.motion, slot.host[2], stages, 2, params.vram ? slot.host[2].rowBytes : 0);
        log_motion_stats(slot.host[2], slot.host[0], frame);
        if (params.hasExposure)
            stage_plane(Plane::Exposure, slot.exposure, slot.host[3], stages, 3,
                        params.vram ? slot.host[3].rowBytes : 0);
        char dumpDirectory[MAX_PATH];
        if (input_dump_directory(frame, dumpDirectory))
            dump_input_planes(slot.host, frame, params, dumpDirectory);
        slot.hostBusy = true;
    }
    if (!params.vram)
        slot.busy = false; // the readback buffers are no longer needed
    if (timing.enabled)
    {
        timing.prepStart = start;
        timing.markerWait = profile_ms(start, ready);
        timing.prep = profile_ms(start, ProfileClock::now());
        timing.workerQueued = ProfileClock::now();
    }
    g.worker.post([feature, slotIndex, frame, params, timing] {
        run_evaluation(feature, slotIndex, frame, params, timing);
    });
}

// Picks a result slot that no queued command list still copies from, and
// marks it so evaluate() does not present it while it is rewritten. A reader
// frame's output copy is recorded after its marker, so the GPU is past it
// only once a later frame's marker has landed. Waits briefly for that.

int claim_output_slot(Feature* feature)
{
    const auto start = std::chrono::steady_clock::now();
    for (;;)
    {
        {
            std::lock_guard<std::mutex> lock(feature->outputMutex);
            const uint32_t gpuFrame = *feature->markerValue;
            for (int index = 0; index < kOutputSlots; ++index)
            {
                if (index == feature->latestOutput.load())
                    continue;
                if (static_cast<int32_t>(gpuFrame - feature->outputs[index].lastReadFrame.load()) > 0)
                {
                    feature->outputs[index].lastReadFrame = 0x7fffffffu + gpuFrame;
                    return index;
                }
            }
        }
        if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(50))
            return -1;
        d4r_sleep_us(200);
    }
}

// Makes `target` the result evaluate() presents and retires the frame.

static void finish_publish(Feature* feature, int target, uint32_t frame, const FrameParams& params,
                           const FrameTiming& timing, ProfileClock::time_point start)
{
    {
        std::lock_guard<std::mutex> lock(feature->outputMutex);
        feature->outputs[target].lastReadFrame = 0;
        feature->outputs[target].producedFrame = frame;
        feature->latestOutput = target;
    }
    const uint32_t completed = ++feature->completedFrames;
    frame_retired(feature, frame);
    const auto finished = ProfileClock::now();
    if (timing.enabled && profile_log_enabled())
    {
        static thread_local ProfileClock::time_point lastFinished;
        const double interval = lastFinished.time_since_epoch().count() != 0
                                    ? profile_ms(lastFinished, finished) : -1.0;
        lastFinished = finished;
        // Update lastFinished on every measured frame: a sampled row still reports the
        // interval from the previous frame, not the span since the previous logged row.
        if (profile_log_frame(frame))
        {
            const InputSlot& slot = feature->inputs[frame % kSlots];
            logf("D4R_PROFILE frame=%u render=%ux%u output=%ux%u color=%ux%u depth=%ux%u "
                 "motion=%ux%u exposure=%d quality=%d flags=0x%x preset=%u "
                 "jitter=%.6f,%.6f mv_scale=%.3f,%.3f reset=%d pre_exposure=%.6f exposure_scale=%.6f "
                 "frame_dt=%.3f game_interval=%.3f "
                 "queue_depth=%zu queue_wait=%.3f marker_wait=%.3f slot_wait=%.3f "
                 "d3d_input_record=%.3f d3d_output_record=%.3f game_call=%.3f "
                 "color_convert=%.3f depth_convert=%.3f motion_convert=%.3f exposure_convert=%.3f "
                 "color_h2d=%.3f depth_h2d=%.3f motion_h2d=%.3f exposure_h2d=%.3f "
                 "ngx_host=%.3f ctx_sync=%.3f gpu_eval=%.3f gpu_eval_scope=%s input_wait=%s output_sync_wall=%.3f "
                 "output_sync_cpu=%.3f output_sync_blocking=%d output_sync_method=%s d2h=%.3f output_convert=%.3f "
                 "h2d_total=%.3f d2h_issue=%.3f d2h_wait=%.3f "
                 "prep=%.3f worker_wait=%.3f worker=%.3f finish_wait=%.3f finish=%.3f pipeline=%.3f "
                 "throughput_interval=%.3f presented=%u age=%d motion_dilation=%u motion_dilation_wall=%.3f "
                 "dlss_gpu=%.3f input_gpu_wait=%.3f vk_input=%.3f vk_dlss_wait=%.3f vk_output=%.3f vk_total=%.3f "
                 "native_motion_gpu=%.3f wire_repair_gpu=%.3f "
                 "feature=%u native_field=%u native_substitute=%u wire_requested=%u gpu_completion=%u vk_output_mode=%s vk_input_mode=%s",
                 frame, params.renderWidth, params.renderHeight, feature->outWidth, feature->outHeight,
                 slot.host[0].width, slot.host[0].height, slot.host[1].width, slot.host[1].height,
                 slot.host[2].width, slot.host[2].height, params.hasExposure, feature->quality, feature->flags,
                 feature->preset, params.jitterX, params.jitterY, params.mvScaleX, params.mvScaleY,
                 params.reset, params.preExposure, params.exposureScale, params.frameTime, timing.gameInterval,
                 timing.queueDepth, profile_ms(timing.queuedAt, timing.prepStart), timing.markerWait, timing.slotWait,
                 timing.inputRecord, timing.outputRecord, timing.gameCall,
                 timing.convert[0], timing.convert[1], timing.convert[2], timing.convert[3],
                 timing.upload[0], timing.upload[1], timing.upload[2], timing.upload[3],
                 timing.ngxHost, timing.ctxSync, timing.gpuEval,
                 timing.gpuInputWait ? "launch_to_completion" : "evaluation", timing.gpuInputWait ? "gpu" : "cpu",
                 timing.outputSyncWall, timing.outputSyncCpu,
                 timing.outputSyncBlocking, timing.outputSyncNative ? "native_callback_wait" :
                                            timing.outputSyncBlocking ? "event_query_sleep" : "context",
                 timing.download, timing.outputConvert,
                 timing.h2dTotal, timing.d2hIssue, timing.d2hWait,
                 timing.prep, timing.workerWait, timing.worker, timing.finishWait, profile_ms(start, finished),
                 profile_ms(timing.prepStart, finished), interval,
                 timing.presentedFrame, timing.presentedFrame != 0 ? static_cast<int>(frame - timing.presentedFrame) : -1,
                 params.motionDilation, timing.motionDilationWall, timing.dlssGpu, timing.inputGpuWait,
                 timing.vkInput, timing.vkDlssWait, timing.vkOutput, timing.vkTotal, timing.nativeMotionGpu, timing.wireRepairGpu,
                 feature->handle.Id, timing.nativeFieldReady, timing.nativeSubstituted, timing.wireRequested, params.gpuCompletion,
                 timing.vkOutputFused<0 ? "unknown" : timing.vkOutputFused ? "compute" : "transfer",
                 timing.vkInputFused<0 ? "unknown" : timing.vkInputFused ? "compute" : "transfer");
        }
    }
    if (completed <= 5 || completed % 120 == 0)
        logf("frame %u done (slot %d)", frame, target);
}

void publish_output(Feature* feature, int hostIndex, uint32_t frame, FrameParams params, FrameTiming timing)
{
    const auto start = ProfileClock::now();
    if (timing.enabled)
        timing.finishWait = profile_ms(timing.finishQueued, start);
    const int target = claim_output_slot(feature);
    if (target < 0)
    {
        feature->outputHostBusy[hostIndex] = false;
        logf("frame %u: no free output slot; dropping result", frame);
        frame_retired(feature, frame);
        return;
    }
    write_output(feature->outputs[target].staging, feature->outputHost[hostIndex], timing.enabled ? &timing : nullptr);
    feature->outputHostBusy[hostIndex] = false;
    finish_publish(feature, target, frame, params, timing, start);
}

// Waits until every frame queued so far has passed through all three stages.

void drain_pipeline()
{
    g.prep.call([] { return 0; });
    g.worker.call([] {
        complete_pending_download();
        return 0;
    });
    g.finish.call([] { return 0; });
}

static NativeRetirer g_nativeRetirer;

bool retain_native(ID3D12GraphicsCommandList* list, IUnknown* resources)
{
    if (g_vk.interop != nullptr || !d4r_submit::installed())
        return false;
    resources->AddRef();
    d4r_submit::track(list, {resources,
        [](void* owner, ID3D12Fence* fence, uint64_t value, uint64_t) {
            g_nativeRetirer.retire(static_cast<IUnknown*>(owner), fence, fence != nullptr ? value : 0);
        },
        [](void* owner) { static_cast<IUnknown*>(owner)->Release(); }});
    return true;
}
