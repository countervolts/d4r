// Compare enc0 phase 8's native staged-image GEMM with a combined p8c oracle
// capture: 24 A fragments, 8 B fragments and 48 D registers per lane.
// The engine produces D[weight][token]; the oracle stores D[token][weight].
// Every HIP call is checked, and a mismatch produces a nonzero exit status.
//
// hipcc --offload-arch=gfx1201 -DD4R_TEX_FP8 -DD4R_WMMA_LAYOUT=12 -O2 \
//   -o /tmp/pilot kernels/tools/rrswin_pilot_enc0_p8.cpp
// /tmp/pilot <oracle-p8c-dir>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>

#include "../rr/rrswin_engine.h"

#define CHECK(expr)                                                                              \
    do                                                                                           \
    {                                                                                            \
        hipError_t e_ = (expr);                                                                  \
        if (e_ != hipSuccess)                                                                    \
        {                                                                                        \
            std::printf("HIP error at %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,              \
                        hipGetErrorString(e_));                                                  \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

static const int kDebugSlot = 16384;           // swin_debug_slot(80) for the combined stage
static const int kDebugBase = 46000000;

static std::vector<uint8_t> slurp(const std::string& dir, const char* what, size_t* size_out)
{
    // the dump names are launch-<n>-<kernel>-<arg>-<kind>-<base>-fmt<n>.bin; matching the pieces
    // is more robust than a glob over a name that contains '+0x0'
    DIR* d = opendir(dir.c_str());
    if (d == nullptr)
    {
        std::printf("cannot open %s\n", dir.c_str());
        std::exit(2);
    }
    std::string found;
    for (dirent* e = readdir(d); e != nullptr; e = readdir(d))
    {
        const std::string name = e->d_name;
        if (name.find("swin_enc0_kernel") != std::string::npos &&
            name.find(std::string("-") + what + "-") != std::string::npos)
        {
            found = dir + "/" + name;
            break;
        }
    }
    closedir(d);
    if (found.empty())
    {
        std::printf("no dump with %s in %s\n", what, dir.c_str());
        std::exit(2);
    }
    FILE* f = std::fopen(found.c_str(), "rb");
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data((size_t)n);
    if (std::fread(data.data(), 1, data.size(), f) != data.size())
    {
        std::printf("short read of %s\n", found.c_str());
        std::exit(2);
    }
    std::fclose(f);
    if (size_out)
        *size_out = data.size();
    return data;
}

// One phase over one debug slot: stage six m-tiles of A fragments, prep the weights, run the
// phase, write D[weight][token] out.  The weight preparation reads the image through the
// documented enc0 layout, offset by the phase's byte base.
__global__ void pilot_kernel(const uint32_t* frag, const uint8_t* wimg, int weight_base,
                             float* out)
{
    __shared__ uint8_t img[96 * 32];
    __shared__ uint8_t weight_slots[32 * 2 * 16];
    for (int i = threadIdx.x; i < 96 * 32; i += blockDim.x)
        img[i] = 0;
    __syncthreads();
    rrswin::swin_stage_fragment(img, 32, frag + (size_t)24 * (blockIdx.x * 32u + threadIdx.x), 6);
    if (threadIdx.x == 0)
        rrswin::swin_prep_phase(weight_slots, 0, 32, 1, wimg + weight_base, rrswin::swin_waddr_enc0);
    __syncthreads();
    const rrswin::Phase phase = {2, 6, 1, 0, 1, 0};
    f8v acc[12];
    rrswin::swin_phase(phase, (const kslot*)weight_slots, 32, img, 32, nullptr, 0, acc);
    for (int tile = 0; tile < 12; ++tile)
        for (int i = 0; i < 8; ++i)
            out[(size_t)blockIdx.x * 3072 + (tile * 8 + i) * 32 + threadIdx.x] = acc[tile][i];
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s <oracle-p8c-dir>\n", argv[0]);
        return 2;
    }
    size_t no = 0, nw = 0;
    const std::vector<uint8_t> p8o = slurp(argv[1], "arg048", &no);   // combined: 32 operands + 48 accumulators
    const std::vector<uint8_t> p8 = p8o;
    const std::vector<uint8_t> wimg = slurp(argv[1], "arg040", &nw);
    std::printf("arg048 (combined p8c) %zu bytes, arg040 %zu bytes\n", no, nw);
    if (no < (size_t)kDebugBase + 64 * kDebugSlot)
    {
        std::printf("dumps are too small for the payload window\n");
        return 2;
    }

    // the phase's A fragments: 24 registers per lane, 32 lanes, per block
    std::vector<uint32_t> frag(64 * 32 * 24);
    for (int b = 0; b < 64; ++b)
        for (int lane = 0; lane < 32; ++lane)
        {
            const uint8_t* p = p8o.data() + kDebugBase + (size_t)b * kDebugSlot + (size_t)lane * 80 * 4;
            for (int r = 0; r < 24; ++r)
                std::memcpy(&frag[((size_t)b * 32 + lane) * 24 + r], p + 4 * r, 4);
        }

    uint32_t* dfrag;
    uint8_t* dwimg;
    float* dout;
    CHECK(hipMalloc(&dfrag, frag.size() * 4));
    CHECK(hipMalloc(&dwimg, wimg.size()));
    CHECK(hipMalloc(&dout, 64 * 3072 * sizeof(float)));
    CHECK(hipMemcpy(dfrag, frag.data(), frag.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dwimg, wimg.data(), wimg.size(), hipMemcpyHostToDevice));
    CHECK(hipMemset(dout, 0, 64 * 3072 * sizeof(float)));
    pilot_kernel<<<64, 32>>>(dfrag, dwimg, 21696, dout);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    std::vector<float> got(64 * 3072);
    CHECK(hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost));

    // Compare the engine's D[weight][token] with the oracle's D[token][weight].
    //
    // The two sides do not use the same lane for the same element, which is what earlier attempts
    // got wrong by reusing one lane for both:
    //   engine: acc[mt][nt][i] of lane L holds D[row = 16*mt + i + 8*(L>>4)][col = 16*nt + (L&15)]
    //   oracle: the p8 payload's register (mi, widx, rh) of lane M holds D[row = 16*mi + 8*rh + g]
    //           [col = 8*widx + 2*t + c], with g = M>>2 and t = M&3
    // so an element identified by (weight w, token k) lives in lane 16*((w%16)/8) + (k%16) on the
    // engine side and in lane 4*((k%16)%8) + (w%8)/2 on the oracle side.
    int bad = 0, checked = 0;
    for (int b = 0; b < 64; ++b)
    {
        const uint8_t* d = p8.data() + kDebugBase + (size_t)b * kDebugSlot;
        for (int wt = 0; wt < 32; ++wt)
            for (int tok = 0; tok < 96; ++tok)
            {
                const int mt = wt / 16, i = (wt % 16) & 7, h = (wt % 16) / 8, nt = tok / 16;
                const int eLane = 16 * h + (tok % 16);
                const float gotv = got[(size_t)b * 3072 + ((mt * 6 + nt) * 8 + i) * 32 + eLane];

                const int mi = tok / 16, rh = (tok % 16) / 8, g = (tok % 16) % 8;
                const int widx = wt / 8, t = (wt % 8) / 2, c = wt % 2;
                const int oLane = 4 * g + t;
                const uint16_t* regs = (const uint16_t*)(d + (size_t)oLane * 80 * 4 + 32 * 4);
                const int reg = (mi * 4 + widx) * 2 + rh;
                const uint32_t u = (uint32_t)regs[2 * reg] | ((uint32_t)regs[2 * reg + 1] << 16);
                const uint16_t bits = (uint16_t)(c ? (u >> 16) : (u & 0xffff));
                const _Float16 hb = *reinterpret_cast<const _Float16*>(&bits);
                const float oracle_f = (float)hb;
                // the engine's accumulator is f32 (the WMMA's), the oracle's mma accumulates in
                // f16, so the comparison is at f16: round the engine's value to f16 first
                const _Float16 got_h = (_Float16)gotv;
                const float got_r = (float)got_h;
                ++checked;
                if (got_r != oracle_f)
                {
                    if (bad < 4)
                        std::printf("  block %2d weight %2d token %2d: engine %+.6f oracle %+.6f\n",
                                    b, wt, tok, got_r, oracle_f);
                    ++bad;
                }
            }
    }
    std::printf("phase-8 pilot: checked %d, mismatched %d\n", checked, bad);
    CHECK(hipFree(dfrag));
    CHECK(hipFree(dwimg));
    CHECK(hipFree(dout));
    return bad != 0;
}
