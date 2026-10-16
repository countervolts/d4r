// Launch a native kernel on a replay dump (manifest.txt, args.bin, alloc-N.bin), save every allocation
// after the first launch, then time repeated launches.
// usage: dump_runner <hsaco> <kernel> <dump dir> <out dir> [iterations]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(x)                                                                                                        \
    do                                                                                                                  \
    {                                                                                                                   \
        hipError_t e_ = (x);                                                                                            \
        if (e_ != hipSuccess)                                                                                           \
        {                                                                                                               \
            fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_));                                             \
            return 1;                                                                                                   \
        }                                                                                                               \
    } while (0)

static std::vector<char> read_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        fprintf(stderr, "usage: %s hsaco kernel dump out [iters]\n", argv[0]);
        return 2;
    }
    const std::string dump = argv[3], out = argv[4];
    const int iters = argc > 5 ? atoi(argv[5]) : 20;
    hipModule_t mod;
    hipFunction_t fn;
    CHECK(hipModuleLoad(&mod, argv[1]));
    CHECK(hipModuleGetFunction(&fn, mod, argv[2]));
    hipFunction_t prep = nullptr;
    if (hipModuleGetFunction(&prep, mod, (std::string(argv[2]) + "_prep").c_str()) != hipSuccess)
        prep = nullptr;
    // the prep grid: PREP_GRID, else the module's d4r_prep_blocks (what the ZLUDA hook launches), else 88
    unsigned prepGrid = 88;
    {
        hipDeviceptr_t pb;
        size_t pbs = 0;
        uint32_t v = 0;
        if (hipModuleGetGlobal(&pb, &pbs, mod, "d4r_prep_blocks") == hipSuccess && pbs == 4 &&
            hipMemcpyDtoH(&v, pb, 4) == hipSuccess && v != 0)
            prepGrid = v;
    }
    if (getenv("PREP_GRID"))
        prepGrid = atoi(getenv("PREP_GRID"));

    std::vector<char> args = read_file(dump + "/args.bin");
    std::ifstream man(dump + "/manifest.txt");
    std::string line;
    unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1}, shared = 0;
    std::vector<int> ids;
    std::vector<void*> dev;
    std::vector<size_t> sizes;
    while (std::getline(man, line))
    {
        std::istringstream s(line);
        std::string kind;
        s >> kind;
        if (kind == "texture" || kind == "surface")
        {
            // texture/surface object handles in the arguments would point at nothing here: the kernel faults
            fprintf(stderr, "%s: the dump uses texture or surface objects, which dump_runner cannot recreate\n",
                    dump.c_str());
            return 3;
        }
        if (kind == "launch")
            s >> grid[0] >> grid[1] >> grid[2] >> block[0] >> block[1] >> block[2] >> shared;
        else if (kind == "alloc")
        {
            int id;
            std::string base;
            size_t size;
            s >> id >> base >> size;
            std::vector<char> data = read_file(dump + "/alloc-" + std::to_string(id) + ".bin");
            if (data.size() != size)
            {
                fprintf(stderr, "alloc %d size mismatch\n", id);
                return 1;
            }
            void* d;
            CHECK(hipMalloc(&d, size));
            CHECK(hipMemcpy(d, data.data(), size, hipMemcpyHostToDevice));
            ids.push_back(id);
            dev.push_back(d);
            sizes.push_back(size);
        }
        else if (kind == "pointer")
        {
            size_t off, delta;
            int id;
            s >> off >> id >> delta;
            size_t i = std::find(ids.begin(), ids.end(), id) - ids.begin();
            uint64_t v = (uint64_t)dev[i] + delta;
            memcpy(args.data() + off, &v, 8);
        }
    }
    {
        hipDeviceptr_t bz;
        size_t bzs = 0;
        uint32_t v = 0;
        if (hipModuleGetGlobal(&bz, &bzs, mod, "d4r_block_z") == hipSuccess && bzs == 4 &&
            hipMemcpyDtoH(&v, bz, 4) == hipSuccess && v != 0)
            block[2] = v;
    }
    {
        hipDeviceptr_t gp;
        size_t gs = 0;
        uint32_t v = 0;
        if (hipModuleGetGlobal(&gp, &gs, mod, "d4r_grid_x") == hipSuccess && gs == 4 &&
            hipMemcpyDtoH(&v, gp, 4) == hipSuccess && v != 0)
        {
            grid[0] = v;
            grid[1] = grid[2] = 1;
        }
    }
    if (getenv("GRID_X"))
        grid[0] = atoi(getenv("GRID_X"));
    if (getenv("GRID_Y"))
        grid[1] = atoi(getenv("GRID_Y"));
    size_t argSize = args.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                   HIP_LAUNCH_PARAM_END};
    const bool prepOnce = getenv("PREP_ONCE") != nullptr;
    int launches = 0;
    auto launch = [&]() {
        if (prep != nullptr && !(prepOnce && launches++ > 0))
        {
            hipError_t e = hipModuleLaunchKernel(prep, prepGrid, 1, 1, 128, 1, 1, 0, nullptr, nullptr, cfg);
            if (e != hipSuccess)
                return e;
        }
        return hipModuleLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, nullptr,
                                     nullptr, cfg);
    };
    CHECK(launch());
    CHECK(hipDeviceSynchronize());
    for (size_t i = 0; i < ids.size(); ++i)
    {
        std::vector<char> h(sizes[i]);
        CHECK(hipMemcpy(h.data(), dev[i], sizes[i], hipMemcpyDeviceToHost));
        std::ofstream(out + "/alloc-" + std::to_string(ids[i]) + ".bin", std::ios::binary).write(h.data(), h.size());
    }
    if (const char* dbg = getenv("DBG_GLOBAL"))
    {
        hipDeviceptr_t ptr;
        size_t size = 0;
        if (hipModuleGetGlobal(&ptr, &size, mod, dbg) == hipSuccess)
        {
            std::vector<char> h(size);
            CHECK(hipMemcpyDtoH(h.data(), ptr, size));
            std::ofstream(out + "/dbg.bin", std::ios::binary).write(h.data(), h.size());
        }
    }
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    std::vector<float> ms;
    for (int i = 0; i < iters; ++i)
    {
        CHECK(hipEventRecord(a));
        CHECK(launch());
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float t;
        CHECK(hipEventElapsedTime(&t, a, b));
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    if (!ms.empty())
        printf("%s: median %.4f ms, min %.4f ms over %d launches\n", argv[2], ms[ms.size() / 2], ms[0], iters);
    return 0;
}
