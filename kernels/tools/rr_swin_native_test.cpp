// Runs kernels/rr/swin_native.h (one Ray Reconstruction encoder stage as a native HIP kernel) on a stage work
// directory written by engine/rr/test_layer.py, compares it with that directory's reference outputs and times it.
//
//   hipcc --offload-arch=gfx1201 -O3 -I STAGE_DIR -o rr_swin_native_test kernels/tools/rr_swin_native_test.cpp
//   rr_swin_native_test WORK WEIGHTS.bin TEXTURE88.bin [REPEATS]
//
// STAGE_DIR holds the stage's swin_stage.h (engine/rr/swin_pack.py), WEIGHTS.bin is the blob the same run wrote, WORK
// is test_layer.py's (params.bin, arena.bin; arena.out.bin and surf112 / 120 / 128.bin from a run of the Vulkan stage
// that was identical to the capture). Channel 3 of the surfaces is left out, as test_layer.py --ignore does.
#include "swin_stage.h"
#include "../rr/swin_native.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(expr)                                                                                          \
    do {                                                                                                     \
        hipError_t e_ = (expr);                                                                              \
        if (e_ != hipSuccess) { std::printf("HIP error at line %d: %s -> %s\n", __LINE__, #expr, hipGetErrorString(e_)); std::exit(2); } \
    } while (0)

static std::vector<uint8_t> slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("cannot open %s\n", path.c_str()); std::exit(2); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
template <typename T> static T* upload(const std::vector<uint8_t>& host, size_t bytes = 0)
{
    void* p = nullptr;
    bytes = std::max(bytes, host.size());
    CHECK(hipMalloc(&p, bytes));
    CHECK(hipMemset(p, 0, bytes));
    if (!host.empty()) CHECK(hipMemcpy(p, host.data(), host.size(), hipMemcpyHostToDevice));
    return (T*)p;
}
static size_t differing(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, bool skipAlpha)
{
    size_t n = a.size() != b.size();
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        if (a[i] != b[i] && !(skipAlpha && (i & 7) >= 6)) ++n;
    return n;
}

int main(int argc, char** argv)
{
    if (argc < 4) { std::printf("usage: %s WORK WEIGHTS.bin TEXTURE88.bin [REPEATS]\n", argv[0]); return 2; }
    const std::string work = argv[1];
    const int repeats = argc > 4 ? atoi(argv[4]) : 20;
    const auto params = slurp(work + "/params.bin"), arena0 = slurp(work + "/arena.bin"), tex = slurp(argv[3]);
    auto weights = slurp(argv[2]);
    const uint32_t* p = (const uint32_t*)params.data();
    const int ex = (int)p[0], ey = (int)p[1], hx = ex / 2, hy = ey / 2;
    const uint32_t shift = p[20];
    const int gx = (ex + (int)(shift & 0xffffu) + 7) / 8, gy = (ey + (int)(shift >> 16) + 7) / 8;
    if (tex.size() != (size_t)ex * ey * 8) { std::printf("texture 88 is not %dx%d RGBA16F\n", ex, ey); return 2; }

    // the layout changes swin_native.h documents: V tiles [k16][n16] -> [n16][k16], bias [key][query] -> [query][key]
    for (uint32_t hd = 0; hd < NH; ++hd) {
        uint8_t* wv = weights.data() + W_WV0 + SN_HEAD * hd;
        for (uint32_t t = 0; t < (C / 16) * 2; ++t)
            for (int k = 0; k < 16; ++k)
                for (int n = 0; n < k; ++n) std::swap(wv[256 * t + 16 * k + n], wv[256 * t + 16 * n + k]);
        uint16_t* b = (uint16_t*)(weights.data() + W_BIAS0 + SN_HEAD * hd);
        std::vector<uint16_t> o(96 * 64);
        for (int key = 0; key < 96; ++key)
            for (int q = 0; q < 64; ++q) o[q * 96 + key] = b[key * 64 + q];
        memcpy(b, o.data(), o.size() * 2);
    }

    auto* dParam = upload<uint32_t>(params);
    auto* dW = upload<uint8_t>(weights);
    auto* dArena = upload<uint8_t>(arena0);
    auto* dTex = upload<swin_native::hf>(tex);
    const size_t fullBytes = (size_t)ex * ey * 8, halfBytes = (size_t)hx * hy * 8;
    auto* d112 = upload<swin_native::hf>({}, fullBytes);
    auto* d120 = upload<swin_native::hf>({}, fullBytes);
    auto* d128 = upload<swin_native::hf>({}, halfBytes);
    const dim3 grid(gx, gy), block(32, 1, 4);
    std::printf("grid %d x %d, %d x %d tokens, C %d\n", gx, gy, ex, ey, C);

    swin_native::swin_stage<<<grid, block>>>(dParam, dW, dArena, dTex, d112, d120, d128);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    bool same = true;
    {
        std::vector<uint8_t> got(arena0.size());
        CHECK(hipMemcpy(got.data(), dArena, got.size(), hipMemcpyDeviceToHost));
        const size_t d = differing(got, slurp(work + "/arena.out.bin"), false);
        std::printf("  arena: %zu bytes differ\n", d);
        same &= d == 0;
    }
    const struct { const char* name; swin_native::hf* dev; size_t bytes; } surfaces[] = {
        {"surf112.bin", d112, fullBytes}, {"surf120.bin", d120, fullBytes}, {"surf128.bin", d128, halfBytes}};
    for (const auto& s : surfaces) {
        std::vector<uint8_t> got(s.bytes);
        CHECK(hipMemcpy(got.data(), s.dev, got.size(), hipMemcpyDeviceToHost));
        const size_t d = differing(got, slurp(work + "/" + s.name), true);
        std::printf("  %s: %zu bytes differ\n", s.name, d);
        same &= d == 0;
    }
    std::printf("%s\n", same ? "IDENTICAL" : "DIFFERENT");

    // like vulkan/kvk: the median over submits of REPEATS launches each
    std::vector<double> t;
    for (int it = 0; it < 12; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeats; ++r) swin_native::swin_stage<<<grid, block>>>(dParam, dW, dArena, dTex, d112, d120, d128);
        CHECK(hipDeviceSynchronize());
        if (it >= 3) t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / repeats);
    }
    std::sort(t.begin(), t.end());
    std::printf("%.1f us per launch (median of %zu rounds of %d, min %.1f)\n", t[t.size() / 2], t.size(), repeats, t[0]);
    return same ? 0 : 1;
}
