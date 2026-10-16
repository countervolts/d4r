#include "hip_api.h"
#include <fstream>
#include <sstream>
#include <memory>
#include <chrono>
#include <cmath>
#include <limits>

namespace {
std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read replay file " + d4r::diag::utf8(path.c_str()));
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
struct Allocation {
    d4r::diag::HipApi& hip;
    int id;
    size_t size;
    void* device = nullptr;
    std::vector<uint8_t> initial;
    Allocation(d4r::diag::HipApi& api, int identifier, const std::vector<uint8_t>& bytes)
        : hip(api), id(identifier), size(bytes.size()), initial(bytes) {
        hip.check(hip.hipMalloc(&device, size), "hipMalloc(replay)");
        try { hip.check(hip.hipMemcpy(device, bytes.data(), size, hipMemcpyHostToDevice), "hipMemcpy(replay upload)"); }
        catch (...) { (void)hip.hipFree(device); device = nullptr; throw; }
    }
    ~Allocation() { if (device) (void)hip.hipFree(device); }
};
}

int main(int argc, char** argv) {
    using namespace d4r::diag;
    start();
    try {
        Args args(argc, argv);
        if (args.module.empty() || args.fixture_dir.empty() || args.output_dir.empty())
            throw std::runtime_error("--module, --fixture-dir and --output-dir are required");
        HipApi hip(args.hip_root);
        hipDeviceProp_t props{};
        hip.select_architecture(args.device, props);
        const std::filesystem::path input(wide(args.fixture_dir)), output(wide(args.output_dir));
        auto parameters = read_file(input / L"args.bin");
        if (parameters.empty() || parameters.size() > 4096) throw std::runtime_error("Invalid replay parameter size");
        std::ifstream manifest(input / L"manifest.txt");
        if (!manifest) throw std::runtime_error("Replay manifest is missing");
        std::vector<std::string> lines;
        std::string line, kernel;
        unsigned grid[3] = {1, 1, 1}, block[3] = {32, 1, 4}, shared = 0;
        std::vector<std::unique_ptr<Allocation>> allocations;
        while (std::getline(manifest, line)) {
            lines.push_back(line);
            std::istringstream stream(line);
            std::string kind;
            stream >> kind;
            if (kind == "kernel") stream >> kernel;
            else if (kind == "launch") stream >> grid[0] >> grid[1] >> grid[2] >> block[0] >> block[1] >> block[2] >> shared;
            else if (kind == "texture" || kind == "surface") throw std::runtime_error("This replay requires a texture/surface recreation backend");
            else if (kind == "alloc") {
                int id = -1; std::string base; size_t size = 0;
                stream >> id >> base >> size;
                if (!stream || id < 0 || size == 0 || size > props.totalGlobalMem / 2)
                    throw std::runtime_error("Invalid replay allocation record");
                for (const auto& allocation : allocations)
                    if (allocation->id == id) throw std::runtime_error("Duplicate replay allocation ID");
                auto bytes = read_file(input / ("alloc-" + std::to_string(id) + ".bin"));
                if (bytes.size() != size) throw std::runtime_error("Replay allocation size mismatch");
                allocations.push_back(std::make_unique<Allocation>(hip, id, bytes));
            }
        }
        if (kernel.empty() || allocations.empty() || !grid[0] || !grid[1] || !grid[2])
            throw std::runtime_error("Incomplete replay manifest");
        for (const auto& record : lines) {
            std::istringstream stream(record); std::string kind;
            stream >> kind;
            if (kind != "pointer") continue;
            size_t offset = 0, delta = 0; int id = -1;
            stream >> offset >> id >> delta;
            if (!stream || offset > parameters.size() || parameters.size() - offset < 8)
                throw std::runtime_error("Invalid replay pointer offset");
            auto allocation = std::find_if(allocations.begin(), allocations.end(), [&](const auto& a) { return a->id == id; });
            if (allocation == allocations.end() || delta >= (*allocation)->size)
                throw std::runtime_error("Invalid replay pointer relocation");
            const uint64_t pointer = reinterpret_cast<uintptr_t>((*allocation)->device) + delta;
            std::memcpy(parameters.data() + offset, &pointer, 8);
        }
        hipModule_t module = nullptr;
        hip.check(hip.hipModuleLoad(&module, args.module.c_str()), "hipModuleLoad(replay)");
        struct Cleanup { HipApi& hip; hipModule_t module; ~Cleanup() { (void)hip.hipModuleUnload(module); } } cleanup{hip, module};
        if (const char* debugBlock = std::getenv("D4R_SWIN_DEBUG_BLOCK")) {
            int coordinates[2]{};
            if (std::sscanf(debugBlock, "%d,%d", &coordinates[0], &coordinates[1]) != 2) throw std::runtime_error("D4R_SWIN_DEBUG_BLOCK must be x,y");
            hipDeviceptr_t address = nullptr; size_t size = 0;
            hip.check(hip.hipModuleGetGlobal(&address, &size, module, "d4r_swin_debug_block"), "Find Swin diagnostic block");
            if (size != sizeof(coordinates)) throw std::runtime_error("Swin diagnostic coordinates ABI");
            hip.check(hip.hipMemcpy(address, coordinates, size, hipMemcpyHostToDevice), "Set Swin diagnostic block");
        }
        auto module_scalar = [&](hipModule_t owner, const char* name, unsigned fallback) {
            hipDeviceptr_t address = nullptr; size_t size = 0;
            if (hip.hipModuleGetGlobal(&address, &size, owner, name) != hipSuccess) return fallback;
            if (size != 4) throw std::runtime_error(std::string("Invalid native metadata ") + name);
            unsigned value = 0;
            hip.check(hip.hipMemcpy(&value, address, 4, hipMemcpyDeviceToHost), name);
            return value ? value : fallback;
        };
        auto scalar = [&](const char* name, unsigned fallback) { return module_scalar(module, name, fallback); };
        const std::vector<unsigned> captured_grid{grid[0], grid[1], grid[2]};
        const unsigned captured_block_z = block[2];
        block[2] = scalar("d4r_block_z", block[2]);
        const unsigned persistentGrid = scalar("d4r_grid_x", 0);
        if (persistentGrid) { grid[0] = persistentGrid; grid[1] = grid[2] = 1; }
        hipFunction_t function = nullptr, prep = nullptr;
        hip.check(hip.hipModuleGetFunction(&function, module, kernel.c_str()), "hipModuleGetFunction(replay)");
        void* launchArguments[] = {parameters.data()};
        if (hip.hipModuleGetFunction(&prep, module, (kernel + "_prep").c_str()) == hipSuccess) {
            const unsigned prepBlocks = scalar("d4r_prep_blocks", 0);
            if (!prepBlocks) throw std::runtime_error("Native prep has no valid d4r_prep_blocks");
            hip.check(hip.hipModuleLaunchKernel(prep, prepBlocks, 1, 1, 128, 1, 1, 0, nullptr, launchArguments, nullptr), "hipModuleLaunchKernel(replay prep)");
        }
        std::printf("REPLAY kernel=%s grid=%u,%u,%u block=%u,%u,%u allocations=%zu arguments=%zu\n",
            kernel.c_str(), grid[0], grid[1], grid[2], block[0], block[1], block[2], allocations.size(), parameters.size());
        if (!args.benchmark_module.empty()) {
            hipModule_t candidate = nullptr;
            hip.check(hip.hipModuleLoad(&candidate, args.benchmark_module.c_str()), "Load paired benchmark candidate");
            Cleanup candidate_cleanup{hip, candidate};
            unsigned grids[2][3] = {{grid[0], grid[1], grid[2]},
                {captured_grid[0], captured_grid[1], captured_grid[2]}};
            unsigned blocks[2][3] = {{block[0], block[1], block[2]},
                {block[0], block[1], module_scalar(candidate, "d4r_block_z", captured_block_z)}};
            const unsigned candidate_persistent = module_scalar(candidate, "d4r_grid_x", 0);
            if (candidate_persistent) { grids[1][0] = candidate_persistent; grids[1][1] = grids[1][2] = 1; }
            for (unsigned variant = 0; variant < 2; ++variant)
                std::printf("D4R_REPLAY_MODULE variant=%s grid=%u,%u,%u block=%u,%u,%u geometry_source=native_metadata\n",
                    variant ? "candidate" : "control", grids[variant][0], grids[variant][1], grids[variant][2],
                    blocks[variant][0], blocks[variant][1], blocks[variant][2]);
            hipFunction_t other = nullptr, other_prep = nullptr;
            hip.check(hip.hipModuleGetFunction(&other, candidate, kernel.c_str()), "Find paired benchmark function");
            if (hip.hipModuleGetFunction(&other_prep, candidate, (kernel + "_prep").c_str()) == hipSuccess) {
                const auto blocks = module_scalar(candidate, "d4r_prep_blocks", 0);
                if (!blocks) throw std::runtime_error("Paired candidate prep metadata is missing");
                hip.check(hip.hipModuleLaunchKernel(other_prep, blocks, 1, 1, 128, 1, 1, 0, nullptr,
                    launchArguments, nullptr), "Paired candidate prep");
            }
            auto create = hip.library.symbol<decltype(&::hipEventCreate)>("hipEventCreate");
            auto destroy = hip.library.symbol<decltype(&::hipEventDestroy)>("hipEventDestroy");
            auto record = hip.library.symbol<decltype(&::hipEventRecord)>("hipEventRecord");
            auto synchronize = hip.library.symbol<decltype(&::hipEventSynchronize)>("hipEventSynchronize");
            auto elapsed = hip.library.symbol<decltype(&::hipEventElapsedTime)>("hipEventElapsedTime");
            // Documented hip/hip_ext.h ABI. Resolve dynamically like the other
            // HIP APIs so the host probe does not depend on HIP host linking.
            using DispatchFn = hipError_t (*)(hipFunction_t, uint32_t, uint32_t, uint32_t,
                uint32_t, uint32_t, uint32_t, size_t, hipStream_t, void**, void**,
                hipEvent_t, hipEvent_t, uint32_t);
            DispatchFn dispatch = nullptr;
            uint32_t global[2][3]{};
            if (args.benchmark_timing == "dispatch") {
                dispatch = hip.library.symbol<DispatchFn>("hipExtModuleLaunchKernel");
                for (unsigned variant = 0; variant < 2; ++variant) for (unsigned axis = 0; axis < 3; ++axis) {
                    const auto items = uint64_t(grids[variant][axis]) * blocks[variant][axis];
                    if (!items || items > std::numeric_limits<uint32_t>::max())
                        throw std::runtime_error("Dispatch profiling work-item dimension overflow");
                    global[variant][axis] = static_cast<uint32_t>(items);
                }
            }
            struct Event {
                decltype(destroy) release; hipEvent_t value = nullptr;
                ~Event() { if (value) (void)release(value); }
            } begin{destroy}, end{destroy};
            hip.check(create(&begin.value), "Create paired start event");
            hip.check(create(&end.value), "Create paired end event");
            auto launch = [&](hipFunction_t fn) {
                const unsigned variant = fn == other;
                hip.check(hip.hipModuleLaunchKernel(fn, grids[variant][0], grids[variant][1], grids[variant][2],
                    blocks[variant][0], blocks[variant][1], blocks[variant][2],
                    shared, nullptr, launchArguments, nullptr), "Paired benchmark launch");
            };
            // Both modules and weight images stay resident on the same GPU.
            // Warm continuous work first, then alternate AB/BA order. Restore
            // captured inputs outside the event so neither module consumes
            // modified temporal inputs from a preceding sample.
            const auto verbose = hip.verbose; hip.verbose = false;
            for (unsigned warm = 0; warm < 16; ++warm)
                for (auto fn : {function, other}) for (unsigned batch = 0; batch < 16; ++batch) launch(fn);
            hip.check(hip.hipDeviceSynchronize(), "Paired benchmark warmup");
            for (unsigned pair = 0; pair < args.iterations; ++pair) for (unsigned order = 0; order < 2; ++order) {
                const unsigned variant = order ^ (pair & 1u);
                for (const auto& allocation : allocations)
                    hip.check(hip.hipMemcpy(allocation->device, allocation->initial.data(), allocation->size,
                        hipMemcpyHostToDevice), "Restore paired captured input");
                const auto host_begin = std::chrono::steady_clock::now();
                if (!dispatch) hip.check(record(begin.value, nullptr), "Record paired start");
                for (unsigned batch = 0; batch < args.benchmark_batch; ++batch) {
                    if (dispatch) {
                        // Flags zero preserve stream order. Global sizes are
                        // work-items, unlike hipModuleLaunchKernel's blocks.
                        hip.check(dispatch(variant ? other : function, global[variant][0], global[variant][1], global[variant][2],
                            blocks[variant][0], blocks[variant][1], blocks[variant][2], shared, nullptr, launchArguments, nullptr,
                            batch == 0 ? begin.value : nullptr,
                            batch + 1 == args.benchmark_batch ? end.value : nullptr, 0),
                            "Paired dispatch-profiled launch");
                    } else launch(variant ? other : function);
                }
                if (!dispatch) hip.check(record(end.value, nullptr), "Record paired end");
                hip.check(synchronize(end.value), "Complete paired sample");
                const auto host_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - host_begin).count();
                float ms = 0; hip.check(elapsed(&ms, begin.value, end.value), "Paired GPU elapsed time");
                const bool valid = std::isfinite(ms) && ms > 0 && double(ms) <= host_ms;
                std::printf("D4R_REPLAY_PROFILE%s kernel=%s variant=%s pair=%u gpu_ms=%.6f "
                    "host_ms=%.6f timing_source=%s launches=%u warmup_launches=512 paired=1 "
                    "input_restore_outside_event=1\n", valid ? "" : "_INVALID",
                    kernel.c_str(), variant ? "candidate" : "control", pair,
                    double(ms) / args.benchmark_batch, host_ms / args.benchmark_batch,
                    args.benchmark_timing.c_str(), args.benchmark_batch);
            }
            hip.verbose = verbose;
            // Leave a single ordinary control replay for numerical checking.
            for (const auto& allocation : allocations)
                hip.check(hip.hipMemcpy(allocation->device, allocation->initial.data(), allocation->size,
                    hipMemcpyHostToDevice), "Restore final reference input");
            launch(function);
        }
        for (unsigned iteration = 0; args.benchmark_module.empty() && iteration < args.iterations; ++iteration)
            hip.check(hip.hipModuleLaunchKernel(function, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared,
                nullptr, launchArguments, nullptr), "hipModuleLaunchKernel(replay)");
        hip.check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize(replay)");
        std::filesystem::create_directories(output);
        if (std::getenv("D4R_SWIN_DEBUG_BLOCK")) {
            hipDeviceptr_t address = nullptr; size_t size = 0;
            hip.check(hip.hipModuleGetGlobal(&address, &size, module, "d4r_swin_debug_values"), "Find Swin diagnostic stages");
            std::vector<uint8_t> bytes(size);
            hip.check(hip.hipMemcpy(bytes.data(), address, size, hipMemcpyDeviceToHost), "Download Swin diagnostic stages");
            std::ofstream file(output / "swin-stages.f16", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
            if (!file) throw std::runtime_error("Cannot save Swin diagnostic stages");
        }
        for (const auto& allocation : allocations) {
            std::vector<uint8_t> bytes(allocation->size);
            hip.check(hip.hipMemcpy(bytes.data(), allocation->device, bytes.size(), hipMemcpyDeviceToHost), "hipMemcpy(replay result)");
            std::ofstream file(output / ("alloc-" + std::to_string(allocation->id) + ".bin"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
            if (!file) throw std::runtime_error("Cannot save replay allocation");
        }
        std::printf("PASS NATIVE_REPLAY architecture=%s kernel=%s iterations=%u benchmark_pairs=%u numerical_reference=pending\n", d4r::diag::HipApi::target_arch(),
            kernel.c_str(), args.benchmark_module.empty() ? args.iterations : 1, args.benchmark_module.empty() ? 0 : args.iterations);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL NATIVE_REPLAY %s\n", error.what()); loaded_modules(); return 4;
    }
}
