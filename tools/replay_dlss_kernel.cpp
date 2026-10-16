#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = uint64_t;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;

template <typename Function> static Function load_function(void* library, const char* name)
{
    void* raw = dlsym(library, name);
    Function function{};
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
}

static void print_cuda_error(const char* stage, CUresult result,
                             CUresult (*get_error_string)(CUresult, const char**) = nullptr)
{
    const char* message = nullptr;
    if (get_error_string != nullptr)
        get_error_string(result, &message);
    std::printf("%s: CUDA result=%d", stage, result);
    if (message != nullptr)
        std::printf(" (%s)", message);
    std::putchar('\n');
}

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3)
    {
        std::fprintf(stderr,
                     "usage: replay_dlss_kernel PATH_TO_LOCALLY_CAPTURED_DLSS_FATBIN [KERNEL_NAME]\n");
        return 2;
    }
    const char* kernel_name = argc == 3 ? argv[2] : "cuda_clear_buffer_kernel";

    std::ifstream input(argv[1], std::ios::binary);
    if (!input)
    {
        std::perror("Could not open captured fatbin");
        return 2;
    }
    std::vector<unsigned char> image((std::istreambuf_iterator<char>(input)),
                                     std::istreambuf_iterator<char>());
    if (image.size() < 16 || std::memcmp(image.data(), "\x50\xed\x55\xba", 4) != 0)
    {
        std::fprintf(stderr, "Input is not a CUDA FATBIN_MAGIC v1 payload\n");
        return 2;
    }

    void* library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
    {
        std::fprintf(stderr, "dlopen(libcuda.so) failed: %s\n", dlerror());
        return 1;
    }

    using InitFn = CUresult (*)(unsigned int);
    using DeviceGetFn = CUresult (*)(CUdevice*, int);
    using ContextCreateFn = CUresult (*)(CUcontext*, unsigned int, CUdevice);
    using ContextDestroyFn = CUresult (*)(CUcontext);
    using MemAllocFn = CUresult (*)(CUdeviceptr*, size_t);
    using MemFreeFn = CUresult (*)(CUdeviceptr);
    using CopyHtoDFn = CUresult (*)(CUdeviceptr, const void*, size_t);
    using CopyDtoHFn = CUresult (*)(void*, CUdeviceptr, size_t);
    using ModuleLoadDataFn = CUresult (*)(CUmodule*, const void*);
    using ModuleGetFunctionFn = CUresult (*)(CUfunction*, CUmodule, const char*);
    using LaunchKernelFn = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int, unsigned int,
                                        CUstream, void**, void**);
    using ContextSynchronizeFn = CUresult (*)(void);
    using ModuleUnloadFn = CUresult (*)(CUmodule);
    using GetErrorStringFn = CUresult (*)(CUresult, const char**);

    const InitFn init = load_function<InitFn>(library, "cuInit");
    const DeviceGetFn device_get = load_function<DeviceGetFn>(library, "cuDeviceGet");
    const ContextCreateFn context_create = load_function<ContextCreateFn>(library, "cuCtxCreate_v2");
    const ContextDestroyFn context_destroy = load_function<ContextDestroyFn>(library, "cuCtxDestroy_v2");
    const MemAllocFn mem_alloc = load_function<MemAllocFn>(library, "cuMemAlloc_v2");
    const MemFreeFn mem_free = load_function<MemFreeFn>(library, "cuMemFree_v2");
    const CopyHtoDFn copy_htod = load_function<CopyHtoDFn>(library, "cuMemcpyHtoD_v2");
    const CopyDtoHFn copy_dtoh = load_function<CopyDtoHFn>(library, "cuMemcpyDtoH_v2");
    const ModuleLoadDataFn module_load = load_function<ModuleLoadDataFn>(library, "cuModuleLoadData");
    const ModuleGetFunctionFn module_get_function =
        load_function<ModuleGetFunctionFn>(library, "cuModuleGetFunction");
    const LaunchKernelFn launch_kernel = load_function<LaunchKernelFn>(library, "cuLaunchKernel");
    const ContextSynchronizeFn context_synchronize =
        load_function<ContextSynchronizeFn>(library, "cuCtxSynchronize");
    const ModuleUnloadFn module_unload = load_function<ModuleUnloadFn>(library, "cuModuleUnload");
    const GetErrorStringFn get_error_string = load_function<GetErrorStringFn>(library, "cuGetErrorString");
    if (init == nullptr || device_get == nullptr || context_create == nullptr || context_destroy == nullptr ||
        mem_alloc == nullptr || mem_free == nullptr || copy_htod == nullptr || copy_dtoh == nullptr ||
        module_load == nullptr || module_get_function == nullptr || launch_kernel == nullptr ||
        context_synchronize == nullptr || module_unload == nullptr)
    {
        std::fprintf(stderr, "ZLUDA libcuda.so is missing one or more driver exports\n");
        dlclose(library);
        return 1;
    }

    CUresult result = init(0);
    if (result != 0)
    {
        print_cuda_error("cuInit", result, get_error_string);
        dlclose(library);
        return 1;
    }
    CUdevice device = 0;
    result = device_get(&device, 0);
    if (result != 0)
    {
        print_cuda_error("cuDeviceGet", result, get_error_string);
        dlclose(library);
        return 1;
    }
    CUcontext context = nullptr;
    result = context_create(&context, 0, device);
    if (result != 0)
    {
        print_cuda_error("cuCtxCreate_v2", result, get_error_string);
        dlclose(library);
        return 1;
    }

    CUmodule module = nullptr;
    result = module_load(&module, image.data());
    print_cuda_error("cuModuleLoadData(official DLSS fatbin)", result, get_error_string);
    if (result != 0)
    {
        context_destroy(context);
        dlclose(library);
        return 1;
    }

    CUfunction kernel = nullptr;
    result = module_get_function(&kernel, module, kernel_name);
    char lookup_stage[256];
    std::snprintf(lookup_stage, sizeof(lookup_stage), "cuModuleGetFunction(%s)", kernel_name);
    print_cuda_error(lookup_stage, result, get_error_string);
    if (result != 0)
    {
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }
    if (std::strcmp(kernel_name, "cuda_clear_buffer_kernel") != 0)
    {
        std::printf("Kernel-resolution-only probe: PASS\n");
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 0;
    }

    unsigned int buffer_width = 64;
    unsigned int buffer_height = 32;
    unsigned int clear_width = 32;
    unsigned int clear_height = 16;
    unsigned int clear_value = 0x5a;
    unsigned int debug_checks = 0;
    const size_t byte_count = buffer_width * buffer_height;
    std::vector<unsigned char> host(byte_count, 0x11);
    CUdeviceptr device_buffer = 0;
    result = mem_alloc(&device_buffer, byte_count);
    if (result == 0)
        result = copy_htod(device_buffer, host.data(), byte_count);
    if (result != 0)
    {
        print_cuda_error("allocate and initialize test buffer", result, get_error_string);
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }

    unsigned int x = 0;
    unsigned int y = 0;
    uint64_t pointer_argument = device_buffer;
    void* parameters[] = {&buffer_width, &buffer_height, &x, &y, &clear_width, &clear_height,
                          &clear_value, &debug_checks, &pointer_argument};
    result = launch_kernel(kernel, 2, 1, 1, 16, 16, 1, 0, nullptr, parameters, nullptr);
    print_cuda_error("cuLaunchKernel(cuda_clear_buffer_kernel)", result, get_error_string);
    if (result == 0)
        result = context_synchronize();
    if (result == 0)
        result = copy_dtoh(host.data(), device_buffer, byte_count);
    if (result != 0)
    {
        print_cuda_error("synchronize and read back kernel output", result, get_error_string);
        mem_free(device_buffer);
        module_unload(module);
        context_destroy(context);
        dlclose(library);
        return 1;
    }

    bool valid = true;
    size_t changed = 0;
    for (unsigned int row = 0; row < buffer_height; ++row)
    {
        for (unsigned int column = 0; column < buffer_width; ++column)
        {
            const unsigned char expected = row < clear_height && column < clear_width ? clear_value : 0x11;
            const unsigned char actual = host[row * buffer_width + column];
            if (actual != expected)
                valid = false;
            if (actual != 0x11)
                ++changed;
        }
    }
    std::printf("Official DLSS kernel test: %s (changed %zu bytes in a %ux%u region; surrounding bytes preserved)\n",
                valid ? "PASS" : "FAIL", changed, clear_width, clear_height);

    mem_free(device_buffer);
    module_unload(module);
    context_destroy(context);
    dlclose(library);
    return valid ? 0 : 1;
}
