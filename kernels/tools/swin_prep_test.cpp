// GPU validation of content-checked preparation against an uncached module.
// Usage: swin_prep_test CONTROL.hsaco CACHED.hsaco WEIGHTS.bin [kernel]
// Tests source mutation at the same address, zero/NaN codes, slot overflow,
// and allocation recycling. No NVIDIA weights are embedded in this test.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iterator>
#include <string>
#include <vector>
#define HIP(call)                                                              \
  do {                                                                         \
    auto e = (call);                                                           \
    if (e != hipSuccess) {                                                     \
      std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e));           \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
struct Module {
  hipModule_t module;
  hipFunction_t prep;
  unsigned blocks;
};
static Module load(const char *path, const std::string &kernel) {
  Module m{};
  HIP(hipModuleLoad(&m.module, path));
  HIP(hipModuleGetFunction(&m.prep, m.module, (kernel + "_prep").c_str()));
  hipDeviceptr_t p;
  size_t n;
  HIP(hipModuleGetGlobal(&p, &n, m.module, "d4r_prep_blocks"));
  HIP(hipMemcpyDtoH(&m.blocks, p, sizeof(m.blocks)));
  return m;
}
static std::vector<unsigned char> global(Module &m, const char *name) {
  hipDeviceptr_t p;
  size_t n;
  if (hipModuleGetGlobal(&p, &n, m.module, name) != hipSuccess)
    return {};
  std::vector<unsigned char> data(n);
  HIP(hipMemcpyDtoH(data.data(), p, n));
  return data;
}
int main(int argc, char **argv) {
  if (argc < 4)
    return 2;
  const std::string kernel = argc > 4 ? argv[4] : "rrlite_enc3_tube_4x4";
  Module control = load(argv[1], kernel), cached = load(argv[2], kernel);
  std::ifstream file(argv[3], std::ios::binary);
  std::vector<unsigned char> original((std::istreambuf_iterator<char>(file)),
                                      {});
  if (original.empty())
    return 2;
  std::vector<void *> weights(11);
  for (auto &p : weights)
    HIP(hipMalloc(&p, original.size()));
  unsigned cases = 0;
  for (unsigned step = 0; step < 40; ++step) {
    const unsigned slot = step < 10 ? 0 : (step - 10) % weights.size();
    auto data = original;
    if (step == 0 || step == 4)
      std::fill(data.begin(), data.end(), 0);
    else if (step == 5)
      std::fill(data.begin(), data.end(), 0x7f);
    else if (step & 1)
      for (size_t i = 0; i < data.size(); i += 71)
        data[i] ^= 0x80;
    if (step == 28) {
      HIP(hipFree(weights[slot]));
      HIP(hipMalloc(&weights[slot], data.size()));
    }
    HIP(hipMemcpy(weights[slot], data.data(), data.size(),
                  hipMemcpyHostToDevice));
    struct {
      void *weights;
      unsigned char rest[80];
    } args{weights[slot], {}};
    size_t bytes = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &bytes,
                      HIP_LAUNCH_PARAM_END};
    for (auto *m : {&control, &cached})
      HIP(hipModuleLaunchKernel(m->prep, m->blocks, 1, 1, 128, 1, 1, 0, nullptr,
                                nullptr, config));
    HIP(hipDeviceSynchronize());
    for (const char *symbol : {"g_w16", "g_c16", "g_prep_keys"}) {
      if (global(control, symbol) != global(cached, symbol)) {
        std::fprintf(stderr, "step %u slot %u: %s mismatch\n", step, slot,
                     symbol);
        return 1;
      }
    }
    ++cases;
  }
  for (auto p : weights)
    HIP(hipFree(p));
  HIP(hipModuleUnload(control.module));
  HIP(hipModuleUnload(cached.module));
  std::printf("%u preparation cases: all prepared weights, constants and keys "
              "byte-identical\n",
              cases);
}
