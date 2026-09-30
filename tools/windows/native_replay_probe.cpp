#include "hip_api.h"
#include <fstream>
#include <sstream>
#include <memory>

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
    Allocation(d4r::diag::HipApi& api, int identifier, const std::vector<uint8_t>& bytes)
        : hip(api), id(identifier), size(bytes.size()) {
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
        hip.select_gfx1201(args.device, props);
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
        hip.check(hip.hipModuleLoad(&module, args.module.c_str()), "hipModuleLoad(replay gfx1201)");
        struct Cleanup { HipApi& hip; hipModule_t module; ~Cleanup() { (void)hip.hipModuleUnload(module); } } cleanup{hip, module};
        auto scalar = [&](const char* name, unsigned fallback) {
            hipDeviceptr_t address = nullptr; size_t size = 0;
            if (hip.hipModuleGetGlobal(&address, &size, module, name) != hipSuccess) return fallback;
            if (size != 4) throw std::runtime_error(std::string("Invalid native metadata ") + name);
            unsigned value = 0;
            hip.check(hip.hipMemcpy(&value, address, 4, hipMemcpyDeviceToHost), name);
            return value ? value : fallback;
        };
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
        for (unsigned iteration = 0; iteration < args.iterations; ++iteration)
            hip.check(hip.hipModuleLaunchKernel(function, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared,
                nullptr, launchArguments, nullptr), "hipModuleLaunchKernel(replay)");
        hip.check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize(replay)");
        std::filesystem::create_directories(output);
        for (const auto& allocation : allocations) {
            std::vector<uint8_t> bytes(allocation->size);
            hip.check(hip.hipMemcpy(bytes.data(), allocation->device, bytes.size(), hipMemcpyDeviceToHost), "hipMemcpy(replay result)");
            std::ofstream file(output / ("alloc-" + std::to_string(allocation->id) + ".bin"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
            if (!file) throw std::runtime_error("Cannot save replay allocation");
        }
        std::printf("PASS NATIVE_REPLAY architecture=gfx1201 kernel=%s iterations=%u numerical_reference=pending\n", kernel.c_str(), args.iterations);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL NATIVE_REPLAY %s\n", error.what()); loaded_modules(); return 4;
    }
}
