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
    // phase kernels NAME_post1..NAME_post8, launched after the main kernel with the same parameters, exactly as
    // the ZLUDA hook and MicroCUDA do: u32 globals d4r_postK_grid_x/_grid_y (0 = the launch's grid; if either is
    // set, grid z is 1) and d4r_postK_block_x/_y/_z (0 = 32/1/1)
    struct Post { hipFunction_t fn; unsigned gx, gy, gz, bx, by, bz; };
    std::vector<Post> posts;
    auto global_u32 = [&](const std::string& name) -> uint32_t {
        hipDeviceptr_t p;
        size_t n = 0;
        uint32_t v = 0;
        if (hipModuleGetGlobal(&p, &n, mod, name.c_str()) == hipSuccess && n == 4 && hipMemcpyDtoH(&v, p, 4) != hipSuccess)
            v = 0;
        return v;
    };
    for (int k = 1; k <= 8; ++k)
    {
        Post q{};
        const std::string name = std::string(argv[2]) + "_post" + std::to_string(k);
        if (hipModuleGetFunction(&q.fn, mod, name.c_str()) != hipSuccess)
            break;
        const std::string g = "d4r_post" + std::to_string(k) + "_";
        const uint32_t px = global_u32(g + "grid_x"), py = global_u32(g + "grid_y");
        q.gx = px ? px : grid[0];
        q.gy = py ? py : grid[1];
        q.gz = (px || py) ? 1 : grid[2];
        q.bx = global_u32(g + "block_x");
        q.by = global_u32(g + "block_y");
        q.bz = global_u32(g + "block_z");
        if (!q.bx) q.bx = 32;
        if (!q.by) q.by = 1;
        if (!q.bz) q.bz = 1;
        posts.push_back(q);
    }
    if (!posts.empty())
        printf("%s: %zu post phase(s)\n", argv[2], posts.size());
    const bool prepOnce = getenv("PREP_ONCE") != nullptr;
    int launches = 0;
    // PHASE_TIMES=1: an event before and after every phase (prep, main, post1..) for per-phase GPU times
    const bool phaseTimes = getenv("PHASE_TIMES") != nullptr;
    std::vector<hipEvent_t> marks(posts.size() + 3);
    for (auto& m : marks)
        CHECK(hipEventCreate(&m));
    std::vector<std::vector<float>> phaseMs(posts.size() + 2);
    bool recordPhases = false;
    auto launch = [&]() -> hipError_t {
        bool prepped = false;
        if (recordPhases) (void)hipEventRecord(marks[0]);
        if (prep != nullptr && !(prepOnce && launches++ > 0))
        {
            hipError_t e = hipModuleLaunchKernel(prep, prepGrid, 1, 1, 128, 1, 1, 0, nullptr, nullptr, cfg);
            if (e != hipSuccess)
                return e;
            prepped = true;
        }
        if (recordPhases) (void)hipEventRecord(marks[1]);
        hipError_t e = hipModuleLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared,
                                             nullptr, nullptr, cfg);
        if (e != hipSuccess)
            return e;
        if (recordPhases) (void)hipEventRecord(marks[2]);
        for (size_t k = 0; k < posts.size(); ++k)
        {
            const Post& q = posts[k];
            e = hipModuleLaunchKernel(q.fn, q.gx, q.gy, q.gz, q.bx, q.by, q.bz, 0, nullptr, nullptr, cfg);
            if (e != hipSuccess)
                return e;
            if (recordPhases) (void)hipEventRecord(marks[3 + k]);
        }
        if (recordPhases)
        {
            if (hipError_t w = hipEventSynchronize(marks.back()); w != hipSuccess)
                return w;
            for (size_t k = 0; k < phaseMs.size(); ++k)
            {
                if (k == 0 && !prepped)
                    continue;
                float t;
                if (hipEventElapsedTime(&t, marks[k], marks[k + 1]) == hipSuccess)
                    phaseMs[k].push_back(t);
            }
        }
        return hipSuccess;
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
    if (phaseTimes)
    {
        recordPhases = true;
        for (int i = 0; i < iters; ++i)
            CHECK(launch());
        CHECK(hipDeviceSynchronize());
        for (size_t k = 0; k < phaseMs.size(); ++k)
        {
            auto& v = phaseMs[k];
            if (v.empty())
                continue;
            std::sort(v.begin(), v.end());
            const std::string phase = k == 0 ? "prep" : k == 1 ? "main" : "post" + std::to_string(k - 1);
            printf("%s %s: median %.4f ms, min %.4f ms\n", argv[2], phase.c_str(), v[v.size() / 2], v[0]);
        }
    }
    std::sort(ms.begin(), ms.end());
    if (!ms.empty())
        printf("%s: median %.4f ms, min %.4f ms over %d launches\n", argv[2], ms[ms.size() / 2], ms[0], iters);
    return 0;
}
