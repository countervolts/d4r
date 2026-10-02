#pragma once
#include "d3d12_external.h"
#include "d3d12_boundary_timing.h"
#include "pixel_conversion.h"
#include "cuda_image_api.h"
#include "ngx_parameters.h"
#include "ngx_public_abi.h"
#include "d3d12_command_hooks.h"
#include "d3d12_retirement.h"
#include <condition_variable>
#include <deque>
#include <thread>
#include <chrono>
#include <memory>
#include <mutex>

namespace d4r::win {
inline std::string env_path(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) throw std::runtime_error(std::string("Set the absolute path in ") + name);
    if (!std::filesystem::path(diag::wide(value)).is_absolute()) throw std::runtime_error(std::string(name) + " must be absolute");
    return value;
}
inline void ngx_check(unsigned result, const char* operation) {
    if (result != 1 || !std::getenv("D4R_QUIET_API")) std::printf("D4R_NGX %s result=0x%08x\n", operation, result);
    if (result != 1) throw std::runtime_error(std::string(operation) + " NGX=" + std::to_string(result));
}
inline ResourceAccess ngx_resource_access(void* parameters, const char* name, D3D12_RESOURCE_STATES fallback) {
    ResourceAccess result(D3D12_RESOURCE_STATES(ngx::uint_value(parameters, name, unsigned(fallback))));
    const std::string prefix(name);
    result.enhanced = ngx::uint_value(parameters, (prefix + ".Enhanced").c_str()) != 0;
    if (result.enhanced) {
        result.layout = D3D12_BARRIER_LAYOUT(ngx::uint_value(parameters, (prefix + ".Layout").c_str()));
        result.access = D3D12_BARRIER_ACCESS(ngx::uint_value(parameters, (prefix + ".Access").c_str()));
    }
    return result;
}
inline std::filesystem::path pixel_module_path() {
    if (const char* path = std::getenv("D4R_FORMAT_MODULE")) return diag::wide(env_path("D4R_FORMAT_MODULE"));
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&pixel_module_path), &module)) throw std::runtime_error("Locate GPU conversion module directory");
    wchar_t path[32768]{}; const DWORD length = GetModuleFileNameW(module, path, 32768);
    if (!length || length == 32768) throw std::runtime_error("GPU conversion module path is invalid");
    return std::filesystem::path(path).parent_path() / L"pixel_convert_gfx1201.hsaco";
}

struct Runtime {
    diag::HipApi hip;
    CudaApi cuda;
    std::unique_ptr<diag::Library> nvapi;
    diag::SearchDirectory featureSearch;
    std::unique_ptr<diag::Library> dlss, core;
    ExternalApi external;
    cuda::ImageApi images;
    ComPtr<ID3D12Device> device;
    CUdevice ordinal = -1;
    CUcontext context = nullptr;
    bool initialized = false;
    std::unique_ptr<SharedTimeline> timeline;
    std::unique_ptr<PixelProgram> pixels;
    std::mutex mutex;
    // Features share one CUDA context. Follow submission order even when
    // separate feature workers wake in a different order; otherwise a worker
    // could hold the context while waiting for an earlier worker's output.
    std::mutex async_mutex;
    std::condition_variable async_changed;
    std::deque<uint64_t> async_tickets;
    std::string async_failure;
    ComPtr<ID3D12CommandQueue> async_queue;
    ngx::ProjectIdentity project;
    ngx::FeatureCommonInfo common{};
    std::vector<std::wstring> featurePaths;
    std::vector<const wchar_t*> featurePathPointers;
    using Init = unsigned(*)(unsigned long long, const wchar_t*, unsigned);
    using Shutdown = unsigned(*)();
    using Allocate = unsigned(*)(void**);
    using Destroy = unsigned(*)(void*);
    using Create = unsigned(*)(unsigned, void*, void**);
    using Evaluate = unsigned(*)(void*, void*, void*);
    using Release = unsigned(*)(void*);
    Allocate allocate = nullptr, capabilities = nullptr;
    Destroy destroy = nullptr;
    Create create = nullptr;
    Evaluate evaluate = nullptr;
    Release release = nullptr;
    Shutdown shutdown = nullptr;

    Runtime(ID3D12Device* d3d, unsigned long long app, const wchar_t* data, unsigned sdk,
        const ngx::FeatureCommonInfo* info = nullptr, const ngx::ProjectIdentity* identity = nullptr)
        : hip(env_path("D4R_HIP_ROOT")), cuda(env_path("D4R_NVCUDA_DLL")),
          featureSearch(std::filesystem::path(diag::wide(env_path("D4R_DLSS_DLL"))).parent_path()),
          external{hip}, images(cuda), device(d3d) {
        if (!d3d) throw std::runtime_error("Null D3D12 device");
        try {
            hipDeviceProp_t props{};
            hip.select_gfx1201(-1, props);
            hip.verbose = cuda.verbose = !std::getenv("D4R_QUIET_API");
            const auto luid = adapter_luid(d3d);
            if (std::memcmp(&luid, props.luid, sizeof(luid))) throw std::runtime_error("D3D12/HIP LUID mismatch");
            cuda.check(cuda.cuInit(0), "cuInit(Windows runtime)");
            int count = 0; cuda.check(cuda.cuDeviceGetCount(&count), "cuDeviceGetCount");
            for (int i = 0; i < count; ++i) {
                CUdevice candidate = -1; cuda.check(cuda.cuDeviceGet(&candidate, i), "cuDeviceGet");
                int bus = -1, pciDevice = -1, domain = -1;
                cuda.check(cuda.cuDeviceGetAttribute(&bus, 33, candidate), "PCI bus");
                cuda.check(cuda.cuDeviceGetAttribute(&pciDevice, 34, candidate), "PCI device");
                cuda.check(cuda.cuDeviceGetAttribute(&domain, 50, candidate), "PCI domain");
                if (bus == props.pciBusID && pciDevice == props.pciDeviceID && domain == props.pciDomainID) ordinal = candidate;
            }
            if (ordinal < 0) throw std::runtime_error("CUDA/HIP PCI identity mismatch");
            cuda.check(cuda.cuDevicePrimaryCtxRetain(&context, ordinal), "cuDevicePrimaryCtxRetain");
            current();
            timeline = std::make_unique<SharedTimeline>(external, d3d);
            // NGX's loader can probe CUDA during DLL initialization: establish
            // the correct primary context before loading either NVIDIA DLL.
            nvapi = std::make_unique<diag::Library>(diag::wide(env_path("D4R_NVAPI_DLL")));
            dlss = std::make_unique<diag::Library>(diag::wide(env_path("D4R_DLSS_DLL")));
            core = std::make_unique<diag::Library>(diag::wide(env_path("D4R_NGX_CORE")));
            allocate = core->symbol<Allocate>("NVSDK_NGX_CUDA_AllocateParameters");
            capabilities = core->symbol<Allocate>("NVSDK_NGX_CUDA_GetCapabilityParameters");
            destroy = core->symbol<Destroy>("NVSDK_NGX_CUDA_DestroyParameters");
            create = core->symbol<Create>("NVSDK_NGX_CUDA_CreateFeature");
            evaluate = core->symbol<Evaluate>("NVSDK_NGX_CUDA_EvaluateFeature");
            release = core->symbol<Release>("NVSDK_NGX_CUDA_ReleaseFeature");
            shutdown = core->symbol<Shutdown>("NVSDK_NGX_CUDA_Shutdown");
            if (identity) {
                if (identity->id.empty()) throw std::runtime_error("Empty NGX project identity");
                project = *identity;
                featurePaths.push_back(std::filesystem::path(diag::wide(env_path("D4R_DLSS_DLL"))).parent_path().wstring());
                if (info) {
                    if (info->PathListInfo.Length > 1024 || (info->PathListInfo.Length && !info->PathListInfo.Path)) throw std::runtime_error("Invalid NGX feature path list");
                    for (unsigned i = 0; i < info->PathListInfo.Length; ++i) if (info->PathListInfo.Path[i]) featurePaths.emplace_back(info->PathListInfo.Path[i]);
                    if (sdk >= 0x14) common.LoggingInfo = info->LoggingInfo;
                }
                for (const auto& path : featurePaths) featurePathPointers.push_back(path.c_str());
                common.PathListInfo = {featurePathPointers.data(), unsigned(featurePathPointers.size())};
                using ProjectInit = unsigned(*)(const char*, int, const char*, const wchar_t*, unsigned, const ngx::FeatureCommonInfo*);
                std::printf("D4R_IDENTITY project=%s engine=%d version=%s sdk=0x%x\n", project.id.c_str(),project.engine,project.version.c_str(),sdk);
                ngx_check(core->symbol<ProjectInit>("NVSDK_NGX_CUDA_Init_ProjectID")(project.id.c_str(),project.engine,project.version.c_str(),data,sdk,&common), "CUDA Init_ProjectID");
            } else {
                std::printf("D4R_IDENTITY application=%llu sdk=0x%x\n", app,sdk);
                ngx_check(core->symbol<Init>("NVSDK_NGX_CUDA_Init")(app, data, sdk), "CUDA Init(driver ABI)");
            }
            initialized = true;
            void* caps = nullptr;
            ngx_check(capabilities(&caps), "CUDA feature capabilities");
            if (!caps) throw std::runtime_error("Null CUDA capability parameters");
            const int available = ngx::int_value(caps, "SuperSampling.Available", -1);
            const int initResult = ngx::int_value(caps, "SuperSampling.FeatureInitResult", -1);
            const int needsDriver = ngx::int_value(caps, "SuperSampling.NeedsUpdatedDriver", -1);
            std::printf("D4R_CAPABILITIES available=%d feature_init=0x%08x needs_driver=%d\n", available, unsigned(initResult), needsDriver);
            ngx_check(destroy(caps), "Destroy capability parameters");
            if (available != 1 || initResult != 1 || needsDriver != 0)
                throw std::runtime_error("CUDA DLSS is unavailable after capability initialization");
            std::printf("D4R_RUNTIME platform=Windows architecture=gfx1201 cpu_frame_copies=0 frame_age=0\n");
        } catch (...) { cleanup(); throw; }
    }
    Runtime(const Runtime&) = delete;
    ~Runtime() { cleanup(); }
    void current() { cuda.check(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent(runtime)"); }
    void drain_async_queue() {
        ComPtr<ID3D12CommandQueue> queue;
        { std::lock_guard<std::mutex> lock(async_mutex); queue = async_queue; }
        if (!queue) return;
        // Creation/free/import may synchronize the AMD device internally.
        // Drain queued D3D12 consumers before taking the CUDA mutex: workers
        // must remain able to satisfy any already-published output waits.
        ComPtr<ID3D12Fence> completion;
        dx(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(completion.GetAddressOf())), "Reconfigure fence");
        Handle event; event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event.value) throw std::runtime_error("Reconfigure event");
        dx(queue->Signal(completion.Get(), 1), "Reconfigure queue Signal");
        if (completion->GetCompletedValue() < 1) {
            dx(completion->SetEventOnCompletion(1, event.value), "Reconfigure completion event");
            if (WaitForSingleObject(event.value, 30000) != WAIT_OBJECT_0) throw std::runtime_error("Reconfigure GPU timeout");
        }
        dx(device->GetDeviceRemovedReason(), "Reconfigure device status");
    }
    void cleanup() noexcept {
        if (context) (void)cuda.cuCtxSetCurrent(context);
        if (initialized) { (void)shutdown(); initialized = false; }
        timeline.reset();
        pixels.reset();
        if (context) {
            (void)cuda.cuCtxSynchronize(); (void)cuda.cuCtxSetCurrent(nullptr);
            (void)cuda.cuDevicePrimaryCtxRelease_v2(ordinal); context = nullptr;
        }
    }
};

class Feature {
    std::shared_ptr<Runtime> rt_;
    void* parameters_ = nullptr;
    void* handle_ = nullptr;
    CUdeviceptr scratch_ = 0;
    ComPtr<ID3D12CommandQueue> queue_;
    // A boundary drains both copies before returning. Reuse their allocators
    // only after that completion; no game allocator is retained here.
    ComPtr<ID3D12CommandAllocator> input_allocator_, output_allocator_;
    ComPtr<ID3D12GraphicsCommandList> input_list_, output_list_;
    struct CopySlot {
        ComPtr<ID3D12CommandAllocator> input_allocator, output_allocator;
        ComPtr<ID3D12GraphicsCommandList> input, output;
        std::unique_ptr<BoundaryTiming> timing;
        uint64_t done = 0;
    };
    struct AsyncFrame {
        void* parameters = d4r_ngx_parameters_create();
        std::vector<ComPtr<ID3D12Resource>> resources;
        SharedTimeline::Ticket ticket{};
        ~AsyncFrame() { if (parameters) d4r_ngx_parameters_destroy(parameters); }
    };
    std::vector<std::unique_ptr<CopySlot>> copy_slots_;
    ComPtr<ID3D12Fence> copy_completion_;
    Handle copy_event_;
    uint64_t copy_value_ = 0;
    std::mutex jobs_mutex_;
    std::mutex submit_mutex_;
    std::condition_variable jobs_changed_;
    std::deque<std::shared_ptr<AsyncFrame>> jobs_;
    std::thread worker_;
    bool worker_stop_ = false, worker_active_ = false;
    std::string worker_failure_;
    struct Plane {
        ComPtr<ID3D12Resource> texture;
        D3D12_RESOURCE_DESC desc{};
        std::unique_ptr<SharedPlane> shared;
        std::unique_ptr<cuda::Image> image;
        std::unique_ptr<PixelAllocation> canonical;
        PixelSpec spec{};
    } planes_[5];
    void cleanup() noexcept {
        (void)rt_->cuda.cuCtxSetCurrent(rt_->context);
        (void)rt_->cuda.cuCtxSynchronize();
        if (handle_) { (void)rt_->release(handle_); handle_ = nullptr; }
        for (auto& plane : planes_) {
            plane.image.reset(); plane.canonical.reset(); plane.shared.reset(); plane.texture.Reset();
        }
        if (scratch_) { (void)rt_->cuda.cuMemFree_v2(scratch_); scratch_ = 0; }
        if (parameters_) { (void)rt_->destroy(parameters_); parameters_ = nullptr; }
    }
public:
    unsigned width = 0, height = 0;
    explicit Feature(std::shared_ptr<Runtime> runtime, void* parameters) : rt_(std::move(runtime)) {
        if (std::getenv("D4R_ASYNC_INTEROP")) rt_->drain_async_queue();
        std::lock_guard<std::mutex> lock(rt_->mutex);
        rt_->current();
        try {
            ngx_check(rt_->allocate(&parameters_), "CUDA AllocateParameters");
            ngx::copy_create(parameters, parameters_);
            d4r_ngx_set_int(parameters_, "DLSS.Enable.Output.Subrects", ngx::int_value(parameters, "DLSS.Enable.Output.Subrects"));
            width = ngx::uint_value(parameters, "Width"); height = ngx::uint_value(parameters, "Height");
            std::printf("D4R_CREATE source=%ux%u->%ux%u cuda=%ux%u->%ux%u\n", width, height,
                ngx::uint_value(parameters, "OutWidth"), ngx::uint_value(parameters, "OutHeight"),
                ngx::uint_value(parameters_, "Width"), ngx::uint_value(parameters_, "Height"),
                ngx::uint_value(parameters_, "OutWidth"), ngx::uint_value(parameters_, "OutHeight"));
            if (!width || !height) throw std::runtime_error("Zero NGX input dimensions");
            rt_->cuda.check(rt_->cuda.cuMemAlloc_v2(&scratch_, 64ull * 1024 * 1024), "NGX VRAM scratch");
            d4r_ngx_set_void(parameters_, "Scratch", reinterpret_cast<void*>(uintptr_t(scratch_)));
            d4r_ngx_set_ull(parameters_, "Scratch.SizeInBytes", 64ull * 1024 * 1024);
            ngx_check(rt_->create(1, parameters_, &handle_), "CUDA CreateFeature");
        } catch (...) { cleanup(); throw; }
    }
    ~Feature() {
        { std::lock_guard<std::mutex> lock(jobs_mutex_); worker_stop_ = true; }
        jobs_changed_.notify_all();
        if (worker_.joinable()) worker_.join();
        if (copy_completion_) {
            try { wait_copies(copy_value_); report_completed_copies(); }
            catch (const std::exception& e) { std::fprintf(stderr, "D4R_ASYNC_CLEANUP_FAILURE %s\n", e.what()); }
        }
        std::lock_guard<std::mutex> lock(rt_->mutex); cleanup();
    }
    Feature(const Feature&) = delete;
    void drain() {
        if (!std::getenv("D4R_ASYNC_INTEROP")) return;
        wait_worker();
        if (copy_value_) wait_copies(copy_value_);
        report_completed_copies();
        rt_->drain_async_queue();
    }

    // Explicit harness boundary: all earlier producers must already be queued.
    // The forthcoming command-list backend calls the same stages at queue submit.
    void evaluate_boundary(ID3D12CommandQueue* queue, void* parameters) {
        if (std::getenv("D4R_ASYNC_INTEROP")) { evaluate_async(queue, parameters); return; }
        const auto boundaryBegin = std::chrono::steady_clock::now();
        auto boundaryStage = boundaryBegin;
        const bool profile = std::getenv("D4R_PROFILE_STAGES") != nullptr;
        auto mark = [&](const char* name) {
            if (!profile) return;
            const auto now = std::chrono::steady_clock::now();
            std::printf("D4R_STAGE name=%s cpu_ms=%.6f\n", name,
                std::chrono::duration<double, std::milli>(now - boundaryStage).count());
            boundaryStage = std::chrono::steady_clock::now();
        };
        commands::InternalScope internal;
        std::lock_guard<std::mutex> lock(rt_->mutex);
        if (!queue) throw std::runtime_error("Null D3D12 queue");
        if (queue_ && queue_.Get() != queue) throw std::runtime_error("A feature cannot switch queues without an explicit drain");
        queue_ = queue;
        ComPtr<ID3D12Device> queueDevice;
        dx(queue->GetDevice(IID_PPV_ARGS(queueDevice.GetAddressOf())), "GetDevice(queue)");
        ComPtr<IUnknown> queueIdentity, runtimeIdentity;
        dx(queueDevice->QueryInterface(IID_PPV_ARGS(queueIdentity.GetAddressOf())), "Query queue device identity");
        dx(rt_->device->QueryInterface(IID_PPV_ARGS(runtimeIdentity.GetAddressOf())), "Query runtime device identity");
        if (queueIdentity.Get() != runtimeIdentity.Get()) throw std::runtime_error("Queue belongs to a different D3D12 device");
        rt_->current();
        prepare(parameters);
        mark("boundary_prepare");
        ComPtr<ID3D12CommandAllocator> inputAllocator, outputAllocator;
        ComPtr<ID3D12GraphicsCommandList> inputList, outputList;
        auto list = [&](ComPtr<ID3D12CommandAllocator>& allocator, ComPtr<ID3D12GraphicsCommandList>& commandList) {
            dx(rt_->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf())), "CreateCommandAllocator(interop)");
            dx(rt_->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                IID_PPV_ARGS(commandList.GetAddressOf())), "CreateCommandList(interop)");
        };
        const bool cached = !std::getenv("D4R_DISABLE_INTEROP_LIST_CACHE");
        if (cached) {
            if (!input_list_) {
                // Publish the pair only after both creations succeed.
                list(inputAllocator, inputList); list(outputAllocator, outputList);
                input_allocator_ = inputAllocator; output_allocator_ = outputAllocator;
                input_list_ = inputList; output_list_ = outputList;
            } else {
                dx(input_allocator_->Reset(), "Reset(input interop allocator)");
                dx(output_allocator_->Reset(), "Reset(output interop allocator)");
                dx(input_list_->Reset(input_allocator_.Get(), nullptr), "Reset(input interop list)");
                dx(output_list_->Reset(output_allocator_.Get(), nullptr), "Reset(output interop list)");
            }
            inputAllocator = input_allocator_; outputAllocator = output_allocator_;
            inputList = input_list_; outputList = output_list_;
        } else { list(inputAllocator, inputList); list(outputAllocator, outputList); }
        mark("boundary_copy_list_setup");
        record_inputs(inputList.Get(), parameters); dx(inputList->Close(), "Close(input copies)");
        ID3D12CommandList* before[] = {inputList.Get()}; queue->ExecuteCommandLists(1, before);
        mark("boundary_input_copy_submit");
        try {
            execute_cuda(queue);
            mark("boundary_cuda_transaction");
            record_output(outputList.Get(), parameters); dx(outputList->Close(), "Close(output copy)");
            ID3D12CommandList* after[] = {outputList.Get()}; queue->ExecuteCommandLists(1, after);
            mark("boundary_output_copy_submit");
            rt_->timeline->drain(queue);
            mark("boundary_output_copy_drain");
        } catch (...) {
            // Keep all recorded resources alive until submitted work is complete.
            rt_->timeline->drain(queue); throw;
        }
    }
private:
    void report_completed_copies() {
        if (!copy_completion_ || !queue_) return;
        const auto completed = copy_completion_->GetCompletedValue();
        if (completed == UINT64_MAX) throw std::runtime_error("Device removed during boundary timing");
        for (auto& slot : copy_slots_) if (slot->timing && slot->done && slot->done <= completed)
            slot->timing->report(queue_.Get(), this);
    }
    void wait_copies(uint64_t value) {
        if (copy_completion_->GetCompletedValue() >= value) return;
        dx(copy_completion_->SetEventOnCompletion(value, copy_event_.value), "Async copy completion event");
        if (WaitForSingleObject(copy_event_.value, 30000) != WAIT_OBJECT_0)
            throw std::runtime_error("Async interop copy completion timeout");
    }
    void wait_worker() {
        std::unique_lock<std::mutex> lock(jobs_mutex_);
        jobs_changed_.wait(lock, [&] { return jobs_.empty() && !worker_active_; });
        if (!worker_failure_.empty()) throw std::runtime_error(worker_failure_);
    }
    CopySlot& copy_slot() {
        report_completed_copies();
        if (!copy_completion_) {
            dx(rt_->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(copy_completion_.GetAddressOf())), "Async copy fence");
            copy_event_.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!copy_event_.value) throw std::runtime_error("Async copy event");
        }
        CopySlot* selected = nullptr;
        const auto completed = copy_completion_->GetCompletedValue();
        for (auto& slot : copy_slots_) if (slot->done <= completed) { selected = slot.get(); break; }
        if (!selected && copy_slots_.size() == 3) {
            selected = copy_slots_.front().get();
            for (auto& slot : copy_slots_) if (slot->done < selected->done) selected = slot.get();
            // Bounded ownership; this wait is outside the runtime mutex.
            wait_copies(selected->done);
            report_completed_copies();
        }
        if (selected) {
            dx(selected->input_allocator->Reset(), "Async input allocator Reset");
            dx(selected->output_allocator->Reset(), "Async output allocator Reset");
            dx(selected->input->Reset(selected->input_allocator.Get(), nullptr), "Async input list Reset");
            dx(selected->output->Reset(selected->output_allocator.Get(), nullptr), "Async output list Reset");
            return *selected;
        }
        auto slot = std::make_unique<CopySlot>();
        auto create = [&](auto& allocator, auto& list) {
            dx(rt_->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.GetAddressOf())), "Async copy allocator");
            dx(rt_->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(list.GetAddressOf())), "Async copy list");
        };
        create(slot->input_allocator, slot->input); create(slot->output_allocator, slot->output);
        if (std::getenv("D4R_PROFILE_GPU_BOUNDARY")) slot->timing = std::make_unique<BoundaryTiming>(rt_->device.Get());
        copy_slots_.push_back(std::move(slot)); return *copy_slots_.back();
    }
    bool resize_needed(void* parameters) {
        const char* names[] = {"Color", "Depth", "MotionVectors", "Output", "ExposureTexture"};
        for (unsigned i = 0; i < 5; ++i) {
            ID3D12Resource* texture = nullptr; (void)d4r_ngx_get_d3d12_resource(parameters, names[i], &texture);
            if (!texture) {
                if (i < 4) throw std::runtime_error(std::string("Missing async resource ") + names[i]);
                continue;
            }
            const auto desc = resource_desc(texture);
            if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 || desc.DepthOrArraySize != 1)
                throw std::runtime_error("Async interop requires single-sample 2D textures");
            if (!planes_[i].shared || planes_[i].desc.Width != desc.Width || planes_[i].desc.Height != desc.Height || planes_[i].desc.Format != desc.Format) return true;
        }
        return false;
    }
    void evaluate_async(ID3D12CommandQueue* queue, void* parameters) {
        commands::InternalScope internal;
        std::lock_guard<std::mutex> submitted(submit_mutex_);
        if (!queue || (queue_ && queue_.Get() != queue)) throw std::runtime_error("Async interop requires one feature queue");
        ComPtr<ID3D12Device> queueDevice; dx(queue->GetDevice(IID_PPV_ARGS(queueDevice.GetAddressOf())), "Async queue device");
        ComPtr<IUnknown> a, b;
        dx(queueDevice->QueryInterface(IID_PPV_ARGS(a.GetAddressOf())), "Async queue identity");
        dx(rt_->device->QueryInterface(IID_PPV_ARGS(b.GetAddressOf())), "Async runtime identity");
        if (a.Get() != b.Get()) throw std::runtime_error("Async queue device mismatch");
        { std::lock_guard<std::mutex> lock(jobs_mutex_);
          if (!worker_failure_.empty()) throw std::runtime_error(worker_failure_); }
        { std::lock_guard<std::mutex> lock(rt_->async_mutex);
          if (!rt_->async_failure.empty()) throw std::runtime_error(rt_->async_failure);
          if (rt_->async_queue && rt_->async_queue.Get() != queue) throw std::runtime_error("Async runtime features must share one D3D12 queue");
          rt_->async_queue = queue; }
        auto& slot = copy_slot();
        if (resize_needed(parameters)) {
            wait_worker(); if (copy_value_) wait_copies(copy_value_);
            rt_->drain_async_queue();
            std::lock_guard<std::mutex> lock(rt_->mutex);
            rt_->current(); prepare(parameters);
        }
        // Steady-state recording touches immutable allocation metadata and the
        // caller's resource snapshot. It does not contend with CUDA or mutate
        // the worker's NGX parameters / texture references.
        queue_ = queue;
        auto frame = std::make_shared<AsyncFrame>();
        if (!frame->parameters) throw std::runtime_error("Async frame parameters allocation");
        ngx::copy_create(parameters, frame->parameters); ngx::copy_frame(parameters, frame->parameters);
        const char* names[] = {"Color", "Depth", "MotionVectors", "Output", "ExposureTexture"};
        for (unsigned i = 0; i < 5; ++i) {
            ID3D12Resource* resource = nullptr; (void)d4r_ngx_get_d3d12_resource(parameters, names[i], &resource);
            frame->resources.emplace_back(resource);
            d4r_ngx_set_d3d12_resource(frame->parameters, names[i], resource);
        }
        if (slot.timing) slot.timing->input_begin(slot.input.Get());
        record_inputs(slot.input.Get(), parameters);
        if (slot.timing) slot.timing->input_end(slot.input.Get());
        dx(slot.input->Close(), "Async input Close");
        if (slot.timing) slot.timing->output_begin(slot.output.Get());
        record_output(slot.output.Get(), parameters);
        if (slot.timing) slot.timing->output_end(slot.output.Get());
        dx(slot.output->Close(), "Async output Close");
        if (!worker_.joinable()) worker_ = std::thread([this] { work(); });
        ID3D12CommandList* input[] = {slot.input.Get()}, *output[] = {slot.output.Get()};
        // Allocate the job before any GPU wait is published. Worker cannot
        // acquire the runtime mutex until every queue dependency is enqueued.
        std::lock_guard<std::mutex> pending(jobs_mutex_);
        { std::lock_guard<std::mutex> order(rt_->async_mutex);
          frame->ticket = rt_->timeline->reserve(); rt_->async_tickets.push_back(frame->ticket.output); }
        try { jobs_.push_back(frame); }
        catch (...) {
            std::lock_guard<std::mutex> order(rt_->async_mutex); rt_->async_tickets.pop_back(); throw;
        }
        try {
            queue->ExecuteCommandLists(1, input);
            rt_->timeline->enqueue(queue, frame->ticket);
            queue->ExecuteCommandLists(1, output);
            slot.done = ++copy_value_;
            dx(queue->Signal(copy_completion_.Get(), slot.done), "Async output-copy retirement Signal");
            if (slot.timing) slot.timing->submitted(frame->ticket.output);
        } catch (const std::exception& e) {
            jobs_.pop_back(); worker_failure_ = e.what();
            { std::lock_guard<std::mutex> order(rt_->async_mutex);
              rt_->async_tickets.pop_back(); rt_->async_failure = e.what(); }
            (void)rt_->timeline->fence->Signal(frame->ticket.output);
            jobs_changed_.notify_all(); rt_->async_changed.notify_all(); throw;
        }
        jobs_changed_.notify_one();
        std::printf("D4R_ASYNC_SUBMIT ticket=%llu copy_slots=%zu cpu_frame_copies=0 frame_age=0\n",
            (unsigned long long)frame->ticket.output, copy_slots_.size());
    }
    void work() noexcept {
        commands::InternalScope internal;
        for (;;) {
            std::shared_ptr<AsyncFrame> frame;
            { std::unique_lock<std::mutex> lock(jobs_mutex_);
              jobs_changed_.wait(lock, [&] { return worker_stop_ || !jobs_.empty(); });
              if (jobs_.empty()) return;
              frame = std::move(jobs_.front()); jobs_.pop_front(); worker_active_ = true; }
            { std::unique_lock<std::mutex> order(rt_->async_mutex);
              rt_->async_changed.wait(order, [&] { return !rt_->async_tickets.empty() && rt_->async_tickets.front() == frame->ticket.output; }); }
            try {
                std::lock_guard<std::mutex> lock(rt_->mutex);
                { std::lock_guard<std::mutex> pending(jobs_mutex_);
                  if (!worker_failure_.empty()) throw std::runtime_error(worker_failure_); }
                rt_->current(); prepare(frame->parameters); execute_cuda(nullptr, &frame->ticket);
            } catch (const std::exception& e) {
                { std::lock_guard<std::mutex> lock(jobs_mutex_); worker_failure_ = e.what(); }
                { std::lock_guard<std::mutex> order(rt_->async_mutex); rt_->async_failure = e.what(); }
                std::fprintf(stderr, "D4R_ASYNC_INTEROP_FAILURE %s\n", e.what());
                // Poison the feature and release the queue for error handling;
                // subsequent evaluates fail explicitly, never reuse an old frame.
                (void)rt_->timeline->fence->Signal(frame->ticket.output);
            }
            { std::lock_guard<std::mutex> order(rt_->async_mutex); rt_->async_tickets.pop_front(); }
            rt_->async_changed.notify_all();
            frame.reset();
            { std::lock_guard<std::mutex> lock(jobs_mutex_); worker_active_ = false; }
            jobs_changed_.notify_all();
        }
    }
    void prepare(void* parameters) {
        static const char* names[] = {"Color", "Depth", "MotionVectors", "Output", "ExposureTexture"};
        for (unsigned i = 0; i < 5; ++i) {
            ID3D12Resource* texture = nullptr;
            (void)d4r_ngx_get_d3d12_resource(parameters, names[i], &texture);
            if (!texture) {
                if (i < 4) throw std::runtime_error(std::string("Missing resource ") + names[i]);
                planes_[i].texture.Reset(); d4r_ngx_set_void(parameters_, names[i], nullptr); continue;
            }
            auto desc = resource_desc(texture);
            if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 || desc.DepthOrArraySize != 1)
                throw std::runtime_error("Interop baseline requires single-sample 2D textures");
            const bool half = i == 0 || i == 2 || i == 3;
            const unsigned channels = i == 0 || i == 3 ? 4 : i == 2 ? 2 : 1;
            if (desc.Width > UINT32_MAX) throw std::runtime_error("Texture width exceeds GPU conversion ABI");
            const PixelSpec spec = pixel_spec(i, desc.Format);
            if (!pixel_supported(i, spec.storage)) throw std::runtime_error(std::string("Unsupported ") + names[i] + " DXGI format=" + std::to_string(desc.Format));
            auto& plane = planes_[i];
            if (!plane.shared || plane.desc.Width != desc.Width || plane.desc.Height != desc.Height || plane.desc.Format != desc.Format) {
                rt_->cuda.check(rt_->cuda.cuCtxSynchronize(), "Synchronize(resize interop)");
                plane.image.reset(); plane.canonical.reset(); plane.shared.reset();
                plane.shared = std::make_unique<SharedPlane>(rt_->external, rt_->device.Get(), desc);
                if (!spec.direct) {
                    if (!rt_->pixels) rt_->pixels = std::make_unique<PixelProgram>(rt_->hip, pixel_module_path());
                    plane.canonical = std::make_unique<PixelAllocation>(rt_->hip, unsigned(desc.Width), desc.Height, channels * (half ? 2 : 4));
                }
                plane.image = std::make_unique<cuda::Image>(rt_->images, unsigned(desc.Width), desc.Height, half ? 16 : 32, channels, i == 3, i == 0 ? 1 : 0);
                plane.desc = desc; plane.spec = spec;
                std::printf("D4R_FORMAT plane=%s dxgi=%u canonical_channels=%u canonical_bits=%u gpu_conversion=%u cpu_copy=0\n",
                    names[i], unsigned(desc.Format), channels, half ? 16u : 32u, !spec.direct);
            }
            plane.texture = texture;
            d4r_ngx_set_void(parameters_, names[i], &plane.image->object);
        }
        ngx::copy_frame(parameters, parameters_);
        if (!ngx::uint_value(parameters_, "DLSS.Render.Subrect.Dimensions.Width")) d4r_ngx_set_uint(parameters_, "DLSS.Render.Subrect.Dimensions.Width", width);
        if (!ngx::uint_value(parameters_, "DLSS.Render.Subrect.Dimensions.Height")) d4r_ngx_set_uint(parameters_, "DLSS.Render.Subrect.Dimensions.Height", height);
        for (const char* name : {"TransparencyMask", "DLSS.Input.Bias.Current.Color.Mask"}) d4r_ngx_set_void(parameters_, name, nullptr);
    }
    void record_inputs(ID3D12GraphicsCommandList* list, void* parameters) {
        const char* names[] = {"Color", "Depth", "MotionVectors", "Output", "ExposureTexture"};
        for (unsigned i : {0u, 1u, 2u, 4u}) {
            ID3D12Resource* texture = nullptr; (void)d4r_ngx_get_d3d12_resource(parameters, names[i], &texture);
            if (!texture) continue;
            static const char* states[] = {"D4R.Color.State", "D4R.Depth.State", "D4R.Motion.State", "D4R.Output.State", "D4R.Exposure.State"};
            planes_[i].shared->copy_input(list, texture, ngx_resource_access(parameters, states[i], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
        }
    }
    void execute_cuda(ID3D12CommandQueue* queue, const SharedTimeline::Ticket* ticket = nullptr) {
        const auto begin = std::chrono::steady_clock::now();
        auto stage = begin;
        const bool profileStages = std::getenv("D4R_PROFILE_STAGES") != nullptr;
        auto mark = [&](const char* name) {
            if (!profileStages) return;
            const auto now = std::chrono::steady_clock::now();
            std::printf("D4R_STAGE name=%s cpu_ms=%.6f\n", name,
                std::chrono::duration<double, std::milli>(now - stage).count());
            stage = std::chrono::steady_clock::now();
        };
        if (ticket) rt_->timeline->wait_ready(ticket->input); else rt_->timeline->wait_input(queue);
        rt_->cuda.check(rt_->cuda.cuCtxSynchronize(), "Interop input producer completion");
        mark("input_fence_wait");
        for (unsigned i : {0u, 1u, 2u, 4u}) if (planes_[i].texture) {
            auto& plane = planes_[i];
            void* data = plane.shared->mapped; uint64_t pitch = plane.shared->footprint.Footprint.RowPitch;
            if (plane.canonical) {
                rt_->pixels->convert(false, data, pitch, plane.canonical->data, plane.canonical->pitch,
                    unsigned(plane.desc.Width), plane.desc.Height, plane.spec.storage, i);
                data = plane.canonical->data; pitch = plane.canonical->pitch;
            }
            plane.image->upload_device(reinterpret_cast<uintptr_t>(data), pitch);
        }
        mark("input_conversion_array_upload");
        if (std::getenv("D4R_INTEROP_VERIFY")) {
            for (unsigned i : {0u, 1u, 2u, 4u}) if (planes_[i].texture) {
                uint32_t sample[8]{};
                rt_->hip.check(rt_->hip.hipMemcpy(sample, planes_[i].shared->mapped,
                    std::min<size_t>(sizeof(sample), planes_[i].shared->bytes), hipMemcpyDeviceToHost), "DIAGNOSTIC external-memory sample");
                std::printf("D4R_VRAM_VERIFY plane=%u words=%08x,%08x,%08x,%08x\n", i, sample[0], sample[1], sample[2], sample[3]);
                const unsigned bytesPerPixel = i == 0 ? 8 : i == 2 ? 4 : 4;
                std::vector<uint8_t> array(size_t(planes_[i].desc.Width) * planes_[i].desc.Height * bytesPerPixel);
                planes_[i].image->download(array.data());
                uint32_t first = 0; std::memcpy(&first, array.data(), 4);
                std::printf("D4R_ARRAY_VERIFY plane=%u first=%08x\n", i, first);
            }
        }
        ngx_check(rt_->evaluate(handle_, parameters_, nullptr), "CUDA EvaluateFeature");
        mark("ngx_evaluate_host");
        // NGX can use internal nonblocking streams. The default stream alone
        // does not establish completion of those streams.
        rt_->cuda.check(rt_->cuda.cuCtxSynchronize(), "NGX all-stream completion");
        mark("ngx_all_stream_completion");
        auto& output = planes_[3];
        output.image->download_device(reinterpret_cast<uintptr_t>(output.canonical ? output.canonical->data : output.shared->mapped),
            output.canonical ? output.canonical->pitch : output.shared->footprint.Footprint.RowPitch);
        rt_->cuda.check(rt_->cuda.cuCtxSynchronize(), "VRAM output completion");
        mark("output_array_download");
        if (std::getenv("D4R_VALIDATE_OUTPUT")) {
            if (!rt_->pixels) rt_->pixels = std::make_unique<PixelProgram>(rt_->hip, pixel_module_path());
            const auto counts = rt_->pixels->validate(output.canonical ? output.canonical->data : output.shared->mapped,
                output.canonical ? output.canonical->pitch : output.shared->footprint.Footprint.RowPitch, unsigned(output.desc.Width), output.desc.Height);
            std::printf("D4R_OUTPUT_VALIDATION elements=%llu nan=%u inf=%u diagnostics_cpu_bytes=8\n", output.desc.Width * output.desc.Height * 4, counts[0], counts[1]);
            if (counts[0] || counts[1]) throw std::runtime_error("DLSS output contains NaN/Inf");
        }
        mark("output_diagnostic");
        if (output.canonical) rt_->pixels->convert(true, output.canonical->data, output.canonical->pitch, output.shared->mapped,
            output.shared->footprint.Footprint.RowPitch, unsigned(output.desc.Width), output.desc.Height, output.spec.storage, 3);
        if (ticket) rt_->timeline->signal_ready(ticket->output); else rt_->timeline->signal_output(queue);
        mark("output_conversion_fence_signal");
        std::printf("D4R_FRAME cpu_frame_copies=%u frame_age=0 interop_ngx_ms=%.3f\n", std::getenv("D4R_INTEROP_VERIFY") ? 1u : 0u,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
    }
    void record_output(ID3D12GraphicsCommandList* list, void* parameters) {
        ID3D12Resource* texture = nullptr; (void)d4r_ngx_get_d3d12_resource(parameters, "Output", &texture);
        if (!texture) throw std::runtime_error("Missing output resource");
        planes_[3].shared->copy_output(list, texture, ngx_resource_access(parameters, "D4R.Output.State", D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    }
};
}
