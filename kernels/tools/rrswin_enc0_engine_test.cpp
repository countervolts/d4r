// Steps 1-5 of the staged-image engine's enc0-port validation: each new primitive in
// kernels/rr/rrswin_engine.h run on the GPU with checked HIP calls and compared against
// an independently written host model.
//
//   step 1  swin_slots_from_wimg   -- the slot bytes directly, then through a m16n16
//           swin_gemm_tile against a random staged image (the "from wimg" part only
//           reshuffles layout, the GEMM proves the (row, k) map as the k16 test's step 4
//           does); 32 rows, two m-tiles, so slot_step = n_rows is exercised
//   step 2  the dcell exchange (writer, wave barrier, reader): bit-exact against the D
//           cells the host model derives from the engine-tile values; a twelve-tile
//           phase (n_mt = 2, n_nt = 6), also driven in the body's (2,2)-tile chunks, and
//           an eight-tile phase (n_mt = 2, n_nt = 4)
//   step 3  swin_stage_operand_p modes 0/1/2: the staged image bytes compared bit for
//           byte against the host's own requantisation (an exhaustive u16->u8 e4m3 table
//           built at startup from the documented satfinite semantics), the activation
//           function at a one-ulp f16 bar per half (the k16 test's step 6 style)
//   step 4  an integrated engine phase: per-column f16 C seeds for a 32-row phase in
//           the column-major f16 blob form (§2), two m-tiles x six n-tiles, k32, pairs
//           rounded once to f16 like the engine loop does
//   step 5  the phase1->2 chain: D1 (per-column C), relu-staged (mode 1; the staged
//           image is dumped and compared bit for byte), phase 2 on the staged image with
//           its own per-column C, pairs rounded once to f16
//
// The host models are written from the contract (kernels/rr/rrswin_enc0_engine.md) and
// the translated body's documented layouts, not from the device code: the enc0 weight
// waddr, the fragment->(row, k) map, the dcell exchange's lds addresses and the
// satfinite-e4m3 rounding.  They share no code with the device side.
//
// Build and run:
//   hipcc --offload-arch=gfx1201 -DD4R_TEX_FP8 -DD4R_WMMA_LAYOUT=12 -O2 \
//         -o /tmp/enc0_engine_test kernels/tools/rrswin_enc0_engine_test.cpp \
//         && /tmp/enc0_engine_test
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>

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

// ---------------------------------------------------------------------------
// independent host models -----------------------------------------------------------

// the enc0 weight-image layout, from the contract (rrswin_enc0.hip's "byte j of a
// lane's 16-byte weight tile holds (k, n) at addr(n, k) = ...")
static int waddr_host(int k, int n)
{
    return 512 * (n >> 4) + 64 * (n & 7) + 16 * ((k >> 1) & 3) + 8 * ((n >> 3) & 1)
         + 4 * ((k >> 3) & 1) + 2 * ((k >> 4) & 1) + (k & 1);
}

// exact e4m3 decode: full f32 (subnormals and the NaN included), from the format's
// definition (3 mantissa bits, bias 7, subnormals 2^-9 apart)
static double dec(uint8_t c)
{
    const uint32_t sign = c & 0x80u, e = (c >> 3) & 0xfu, m = c & 7u;
    const double mag = e == 0 ? (m / 8.0) * std::pow(2.0, -6.0)
                              : (1.0 + m / 8.0) * std::pow(2.0, (int)e - 7.0);
    return sign ? -mag : mag;
}

// f16 bits <-> value, host-side
__attribute__((device, host)) static inline _Float16 h_of(uint16_t b)
{
    _Float16 h;
    std::memcpy(&h, &b, sizeof(h));
    return h;
}
__attribute__((device, host)) static inline uint16_t b_of(_Float16 h)
{
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

// ---------------------------------------------------------------------------
// the RNE-satfinite e4m3 rounding table ---------------------------------------------

// For f16 bits: the exact f64 value, then, at every binary step up 2^-24..2^25, bits
// 0..7: bit 7 into code (the merged exponent covers normal and subnormal), the bit
// below the cut -> a half, anything below -> sticky.  At the top of the walk the code
// saturates at 0x7e; at the bottom subnormals emit 2^-9-spaced codes.  Overflow and
// NaN are the doc's two cases (0x7e and the whole-tab 0x7f).
static uint8_t tab[0x10000];
static void tab_build()
{
    for (uint32_t b = 0; b < 0x10000u; ++b)
    {
        const uint16_t bits = (uint16_t)b;
        const uint32_t mag = bits & 0x7fffu, sign = (bits >> 8) & 0x80u;
        if (mag > 0x7c00u)
        {
            tab[b] = (uint8_t)(sign | 0x7fu);           // NaN keeps its sign
            continue;
        }
        if (mag >= 0x7c00u)
        {
            tab[b] = (uint8_t)(sign | 0x7eu);           // inf: out of range -> max finite
            continue;
        }
        if (mag == 0)
        {
            tab[b] = (uint8_t)sign;                     // +/-0 preserved
            continue;
        }
        const int e = (int)(bits >> 10) & 0x1f, m = (int)(bits & 0x3ffu);
        // the exact f64 of the f16 magnitude
        double v;
        if (e > 0)
            v = (1.0 + m / 1024.0) * std::pow(2.0, e - 15.0);
        else
            v = (m / 1024.0) * std::pow(2.0, -14.0);
        // the walk: the leading power of two (2^9 .. 2^-6) fixes the code's merged
        // exponent, the three steps below it pick the mantissa bits, and the first
        // dropped power of two is the rounding bit, the rest sticky.  Under the whole
        // subnormal grid (v < 2^-9) the same walk on the 2^-9-spaced subnormal codes
        // decides, rounding on 2^-10.
        uint32_t code = 0;
        uint32_t round = 0, sticky = 0;
        int base = 0;
        int hit = 0;
        for (int q = 9; q >= -6; --q)               // the merged exponent, q >= -6
        {
            const double step = std::pow(2.0, q);
            if (v >= step)
            {
                code = (uint32_t)((q + 7) << 3);
                base = q;
                v -= step;
                hit = 1;
                break;
            }
        }
        if (hit)
        {
            for (int q = base - 1; q >= base - 3; --q)
            {
                const double step = std::pow(2.0, q);
                if (v >= step)
                {
                    v -= step;
                    code |= 1u << (q - (base - 3));
                }
            }
            // the first dropped power of two is the rounding bit, the rest sticky
            const double halfstep = std::pow(2.0, (double)(base - 4));
            if (v >= halfstep)
                round = 1;
            if (v > halfstep)
                sticky = 1;
        }
        else
        {
            // subnormal: codes m * 2^-9; walk the three steps 2^-7, 2^-8, 2^-9 (m bits
            // 2,1,0), then the half (2^-10) and sticky
            for (int q = -7; q >= -9; --q)
            {
                const double step = std::pow(2.0, q);
                if (v >= step)
                {
                    v -= step;
                    code |= 1u << (q + 9);
                }
            }
            const double halfstep = std::pow(2.0, -10.0);
            if (v >= halfstep)
                round = 1;
            if (v > halfstep)
                sticky = 1;
        }
        if (round && (sticky || (code & 1u)))
            ++code;                                     // RNE: tie -> even
        if (code > 0x7eu)
            code = 0x7eu;
        tab[b] = (uint8_t)(sign | code);
    }
}

// RNE-satfinite e4m3 from f16 bits, through the table (the device's
// cvt.rn.satfinite.e4m3x2.f16x2, low half -> low byte)
static uint8_t requant_code(uint16_t bits) { return tab[bits]; }
static uint8_t requant_code_bytes(double v) { return requant_code(b_of((_Float16)v)); }

// The requantisation the device performs (dec->f16->satfinite->dec, the k16 test's
// requant with the table behind it)
static double requant(double v) { return dec(requant_code_bytes(v)); }

// relu_keep_nan_halves per half, host model: negative or -0 becomes +0, NaN kept
static uint16_t relu_half(uint16_t bits)
{
    return (bits & 0x8000u) && (bits & 0x7fffu) <= 0x7c00u ? 0u : bits;
}

// The clamped cubic activation per half, host model from the spec formula in
// rrswin_enc0.hip's comment, f16 throughout (each intermediate cast back), one-ulp bar
// for the packed-f16x2 device ops
static uint16_t act_model(uint16_t bits)
{
    const _Float16 x = h_of(bits);
    const _Float16 two = h_of(0x4000), slope = h_of(0x2d30), mid = h_of(0x3698), half0 = h_of(0x3800);
    _Float16 y = x > -two ? x : -two;
    y = y < two ? y : two;
    y = y > -y ? y : -y;
    const _Float16 g = (_Float16)(half0 + y * (mid - slope * y));
    return b_of((_Float16)(x * g));
}

// ---------------------------------------------------------------------------
// step 1 kernels: the slots (dumped for the byte check), and one m16n16 GEMM per
// m-tile against a random staged image (tile (mt, nt) of a 32-row x 32-token block)
__global__ void step1_kernel(const uint8_t* wimg, uint8_t* slots_dump, const uint8_t* ximg,
                             float* out)
{
    __shared__ uint8_t slots[2048];
    __shared__ uint8_t img[32 * 32];
    for (int i = threadIdx.x; i < 32 * 32; i += blockDim.x)
        img[i] = ximg[i];
    __syncthreads();
    rrswin::swin_slots_from_wimg((lds_u8*)slots, wimg, 0, 32);
    __syncthreads();
    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
        slots_dump[i] = slots[i];
    for (int mt = 0; mt < 2; ++mt)
    {
        f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
        c = rrswin::swin_gemm_tile(c, (const lds_u8*)slots, 16 * mt, 32, (const lds_u8*)img, 32, 0);
        for (int i = 0; i < 8; ++i)
            out[256 * mt + i * 32 + threadIdx.x] = c[i];
    }
}

// ---------------------------------------------------------------------------
// step 2 kernel: the dcell exchange.  The value source is random f32 D values, shaped
// as the lane's own [tiles][8] VGPRs (pair j covers VGPRs 2j, 2j + 1).  The kernel packs
// pairs per tile, writes, barriers, reads, and dumps d flat as u32 words per lane.
// `chunked`: 0 = the whole twelve-tile buffer in one call, 1 = the body's (2,2)-tile
// chunk driver over the same [12][8] source (d's [6][2][4] accumulated through three
// chunks), 2 = the eight-tile phase (its own [8][8] source buffer).
__global__ void step2_kernel(const float* vin, uint32_t* out, int chunked)
{
    __shared__ __attribute__((aligned(1024))) unsigned char sraw[12 * 128 * 4];
    rrswin::lds_u32* const lds = (rrswin::lds_u32*)(uintptr_t)sraw;
    const uint32_t l = threadIdx.x;
    const uint32_t ntiles = chunked ? 12u : (uint32_t)(chunked == 0 ? 12u : 8u);
    (void)ntiles;
    // pack pairs per tile
    if (chunked == 0 || chunked == 1)
    {
        const int tiles = chunked == 0 ? 12 : 12;
        uint32_t pairs[12 * 4];
        for (int tl = 0; tl < tiles; ++tl)
            for (int j = 0; j < 4; ++j)
            {
                const _Float16 lo = (_Float16)vin[((size_t)l * tiles + tl) * 8 + 2 * j];
                const _Float16 hi = (_Float16)vin[((size_t)l * tiles + tl) * 8 + 2 * j + 1];
                uint16_t bl, bh;
                std::memcpy(&bl, &lo, sizeof(bl));
                std::memcpy(&bh, &hi, sizeof(bh));
                pairs[tl * 4 + j] = (uint32_t)bl | ((uint32_t)bh << 16);
            }
        for (int i = 0; i < 12 * 128; ++i)
            lds[i] = 0u;
        __builtin_amdgcn_wave_barrier();
        if (chunked == 0)
        {
            rrswin::swin_dcell_exchange_write(lds, pairs, 2, 6);
            __builtin_amdgcn_wave_barrier();
            uint32_t d[6 * 2 * 4];
            rrswin::swin_dcell_exchange_read(lds, d, 6, 4, 6);
            for (int i = 0; i < 48; ++i)
                out[i * 32 + threadIdx.x] = d[i];
        }
        else
        {
            // the body's chunk shape: value source [2][6][4], d + 16c per chunk
            uint32_t xp[16], d[6 * 2 * 4];
            for (int c = 0; c < 3; ++c)
            {
                for (int mt = 0; mt < 2; ++mt)
                    for (int l2 = 0; l2 < 2; ++l2)
                        for (int p = 0; p < 4; ++p)
                            xp[(mt * 2 + l2) * 4 + p] = pairs[(mt * 6 + 2 * c + l2) * 4 + p];
                rrswin::swin_dcell_exchange_write(lds, xp, 2, 2);
                __builtin_amdgcn_wave_barrier();
                rrswin::swin_dcell_exchange_read(lds, d + 16 * c, 2, 4, 2);
                __builtin_amdgcn_wave_barrier();
            }
            for (int i = 0; i < 48; ++i)
                out[i * 32 + threadIdx.x] = d[i];
        }
    }
    else
    {
        // eight-tile phase: [2][4][4] pairs -> d[4][2][4]; the value source is laid out
        // as [8][8] per lane, at the same (lane, tile, vgpr) offsets as the 12-tile one
        uint32_t pairs[8 * 4];
        for (int tl = 0; tl < 8; ++tl)
            for (int j = 0; j < 4; ++j)
            {
                const _Float16 lo = (_Float16)vin[((size_t)l * 8 + tl) * 8 + 2 * j];
                const _Float16 hi = (_Float16)vin[((size_t)l * 8 + tl) * 8 + 2 * j + 1];
                uint16_t bl, bh;
                std::memcpy(&bl, &lo, sizeof(bl));
                std::memcpy(&bh, &hi, sizeof(bh));
                pairs[tl * 4 + j] = (uint32_t)bl | ((uint32_t)bh << 16);
            }
        for (int i = 0; i < 8 * 128; ++i)
            lds[i] = 0u;
        __builtin_amdgcn_wave_barrier();
        rrswin::swin_dcell_exchange_write(lds, pairs, 2, 4);
        __builtin_amdgcn_wave_barrier();
        uint32_t d[4 * 2 * 4];
        rrswin::swin_dcell_exchange_read(lds, d, 4, 4, 4);
        for (int i = 0; i < 32; ++i)
            out[i * 32 + threadIdx.x] = d[i];
    }
}

// ---------------------------------------------------------------------------
// step 3 kernel: swin_stage_operand_p's three modes into three images.
__global__ void step3_kernel(const uint32_t* pairs, uint8_t* img0, uint8_t* img1, uint8_t* img2)
{
    __shared__ uint8_t s0[32 * 32], s1[32 * 32], s2[32 * 32];
    for (int i = threadIdx.x; i < 32 * 32; i += blockDim.x)
    {
        s0[i] = 0;
        s1[i] = 0;
        s2[i] = 0;
    }
    __syncthreads();
    const uint32_t* mine = pairs + threadIdx.x * 32;
    for (int tile = 0; tile < 2; ++tile)
    {
        const uint32_t* p = mine + tile * 4;
        rrswin::swin_stage_operand_p((lds_u8*)s0, 32, 16 * tile, 0, p, 0);
        rrswin::swin_stage_operand_p((lds_u8*)s1, 32, 16 * tile, 0, p, 1);
        rrswin::swin_stage_operand_p((lds_u8*)s2, 32, 16 * tile, 0, p, 2);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 32; i += blockDim.x)
    {
        img0[i] = s0[i];
        img1[i] = s1[i];
        img2[i] = s2[i];
    }
}

// ---------------------------------------------------------------------------
// step 4 kernel: the integrated engine phase.  Per-column f16 C seeds at w+5120, two
// m-tiles x six n-tiles, k32, each tile's pairs rounded once to f16 (enc0_pack).
__global__ void step4_kernel(const uint8_t* wimg, uint8_t* slots_dump, const uint8_t* ximg,
                             const uint32_t* cseed, uint32_t* out)
{
    __shared__ uint8_t slots[2048];
    __shared__ uint8_t img[96 * 32];
    for (int i = threadIdx.x; i < 96 * 32; i += blockDim.x)
        img[i] = ximg[i];
    __syncthreads();
    rrswin::swin_slots_from_wimg((lds_u8*)slots, wimg, 0, 32);
    __syncthreads();
    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
        slots_dump[i] = slots[i];
    const uint32_t h = threadIdx.x >> 4;
    for (int mt = 0; mt < 2; ++mt)
        for (int nt = 0; nt < 6; ++nt)
        {
            // the per-column C seed: c[i] = f32(f16 at w + 2*(i + 8h + 16mt)), §2
            f8v c;
            for (int i = 0; i < 8; ++i)
            {
                // the per-column blob's column 2*(i + 8h) + 32mt sits at byte 2*(...):
                // two adjacent f16, in one u32 when the byte offset is even
                const uint32_t w32 = cseed[(2 * (i + 8 * h + 16 * mt)) >> 2];
                const uint16_t bits = (uint16_t)(((2 * (i + 8 * h + 16 * mt)) & 2) ? w32 >> 16
                                                                                  : w32 & 0xffffu);
                c[i] = (float)h_of(bits);
            }
            c = rrswin::swin_gemm_tile(c, (const lds_u8*)slots, 16 * mt, 32, (const lds_u8*)img, 32, 16 * nt);
            uint32_t pairs[4];
            for (int p = 0; p < 4; ++p)
            {
                const _Float16 lo = (_Float16)c[2 * p], hi = (_Float16)c[2 * p + 1];
                uint16_t bl, bh;
                std::memcpy(&bl, &lo, sizeof(bl));
                std::memcpy(&bh, &hi, sizeof(bh));
                pairs[p] = (uint32_t)bl | ((uint32_t)bh << 16);
            }
            for (int p = 0; p < 4; ++p)
                out[((mt * 6 + nt) * 4 + p) * 32 + threadIdx.x] = pairs[p];
        }
}

// ---------------------------------------------------------------------------
// step 5 kernel: the phase1->2 chain.  Phase 1: per-column C at w+2048, weights at
// w+4096; its pairs relu-staged (mode 1) into a fresh image; phase 2: per-column C at
// w+5120, weights at w+6144, on the staged image.  The C seeds are blobs this test
// generates at w+2048 (phase 1) and w+5120 (phase 2), read column-major (§2).
__global__ void step5_kernel(const uint8_t* wimg, const uint8_t* ximg, const uint32_t* cseed1,
                             const uint32_t* cseed2, uint8_t* staged, uint32_t* out)
{
    __shared__ uint8_t slots[2048];
    __shared__ uint8_t img[96 * 32];
    __shared__ uint8_t img2[96 * 32];
    for (int i = threadIdx.x; i < 96 * 32; i += blockDim.x)
        img[i] = ximg[i];
    for (int i = threadIdx.x; i < 96 * 32; i += blockDim.x)
        img2[i] = 0;
    __syncthreads();
    rrswin::swin_slots_from_wimg((lds_u8*)slots, wimg, 4096, 32);
    __syncthreads();
    uint32_t pairs1[2][6][4];
    const uint32_t h = threadIdx.x >> 4;
    for (int mt = 0; mt < 2; ++mt)
        for (int nt = 0; nt < 6; ++nt)
        {
            f8v c;
            for (int i = 0; i < 8; ++i)
            {
                const uint32_t w32 = cseed1[(2 * (i + 8 * h + 16 * mt)) >> 2];
                const uint16_t bits = (uint16_t)(((2 * (i + 8 * h + 16 * mt)) & 2) ? w32 >> 16
                                                                                  : w32 & 0xffffu);
                c[i] = (float)h_of(bits);
            }
            c = rrswin::swin_gemm_tile(c, (const lds_u8*)slots, 16 * mt, 32, (const lds_u8*)img, 32, 16 * nt);
            for (int p = 0; p < 4; ++p)
            {
                const _Float16 lo = (_Float16)c[2 * p], hi = (_Float16)c[2 * p + 1];
                uint16_t bl, bh;
                std::memcpy(&bl, &lo, sizeof(bl));
                std::memcpy(&bh, &hi, sizeof(bh));
                pairs1[mt][nt][p] = (uint32_t)bl | ((uint32_t)bh << 16);
            }
        }
    __syncthreads();
    for (int mt = 0; mt < 2; ++mt)
        for (int nt = 0; nt < 6; ++nt)
            rrswin::swin_stage_operand_p((lds_u8*)img2, 32, 16 * nt, mt, pairs1[mt][nt], 1);
    __syncthreads();
    for (int i = threadIdx.x; i < 96 * 32; i += blockDim.x)
        staged[i] = img2[i];
    __syncthreads();
    rrswin::swin_slots_from_wimg((lds_u8*)slots, wimg, 6144, 32);
    __syncthreads();
    for (int mt = 0; mt < 2; ++mt)
        for (int nt = 0; nt < 6; ++nt)
        {
            f8v c;
            for (int i = 0; i < 8; ++i)
            {
                const uint32_t w32 = cseed2[(2 * (i + 8 * h + 16 * mt)) >> 2];
                const uint16_t bits = (uint16_t)(((2 * (i + 8 * h + 16 * mt)) & 2) ? w32 >> 16
                                                                                  : w32 & 0xffffu);
                c[i] = (float)h_of(bits);
            }
            c = rrswin::swin_gemm_tile(c, (const lds_u8*)slots, 16 * mt, 32, (const lds_u8*)img2, 32, 16 * nt);
            uint32_t pairs[4];
            for (int p = 0; p < 4; ++p)
            {
                const _Float16 lo = (_Float16)c[2 * p], hi = (_Float16)c[2 * p + 1];
                uint16_t bl, bh;
                std::memcpy(&bl, &lo, sizeof(bl));
                std::memcpy(&bh, &hi, sizeof(bh));
                pairs[p] = (uint32_t)bl | ((uint32_t)bh << 16);
            }
            for (int p = 0; p < 4; ++p)
                out[((mt * 6 + nt) * 4 + p) * 32 + threadIdx.x] = pairs[p];
        }
}

// pair q's halves are VGPRs 2q, 2q + 1; VGPR v of a lane's half h is row v + 8h.
static int i_vgpr(int q, int half, int h) { return 2 * q + half + 8 * h; }

// ---------------------------------------------------------------------------
// main ---------------------------------------------------------------------------

// well-spread e4m3 codes (the k16 test's next_code: exponents 0..7 and mantissas, no
// NaN codes), so a wrong (row, column) mapping shows up as a real mismatch
static uint32_t lcg_main = 0x12345678u;
static uint8_t next_code()
{
    lcg_main = lcg_main * 1664525u + 1013904223u;
    uint32_t mag = (lcg_main >> 13) & 0x3fu;      // exponents 0..7 and mantissas
    if (mag == 0)
        mag = 0x21;
    const uint32_t sign = (lcg_main >> 7) & 1u;
    return (uint8_t)((sign << 7) | mag);
}

int main()
{
    tab_build();

    // ---------------------------------------------------------------- step 1
    {
        std::vector<uint8_t> wimg(4096, 0);
        for (auto& v : wimg)
            v = next_code();
        std::vector<uint8_t> ximg(32 * 32, 0);
        for (auto& v : ximg)
            v = next_code();

        uint8_t *dwimg, *dslots, *dximg;
        float* dout;
        CHECK(hipMalloc(&dwimg, wimg.size()));
        CHECK(hipMalloc(&dslots, 2048));
        CHECK(hipMalloc(&dximg, ximg.size()));
        CHECK(hipMalloc(&dout, 512 * sizeof(float)));
        CHECK(hipMemset(dslots, 0, 2048));
        CHECK(hipMemcpy(dwimg, wimg.data(), wimg.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dximg, ximg.data(), ximg.size(), hipMemcpyHostToDevice));
        step1_kernel<<<1, 32>>>(dwimg, dslots, dximg, dout);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint8_t> slots(2048, 0);
        CHECK(hipMemcpy(slots.data(), dslots, slots.size(), hipMemcpyDeviceToHost));
        std::vector<float> got(512);
        CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));

        // 1a: every slot byte = img[waddr(k, n)]
        int bad = 0, checked = 0;
        for (int bank = 0; bank < 64; ++bank)
        {
            // the slot layout is n-indexed: bank n holds k 0..15 (byte j = element
            // (j, n), one full waddr each — the image's records do NOT store
            // consecutive k contiguously) and bank 32 + n holds k 16..31
            const int n = bank & 31;
            const int k0 = bank < 32 ? 0 : 16;
            for (int j = 0; j < 16; ++j)
            {
                const uint8_t have = slots[(size_t)bank * 16 + j];
                const uint8_t want = wimg[waddr_host(k0 + j, n)];
                ++checked;
                if (have != want)
                {
                    if (bad < 4)
                        std::printf("  slot[h%d n%2d j%2d] = 0x%02x, expected 0x%02x\n",
                                    bank >= 32, n, j, have, want);
                    ++bad;
                }
            }
        }
        std::printf("step 1a slot bytes vs waddr: checked %d, mismatched %d\n", checked, bad);

        // 1b: the GEMM through the slots, against the double-precision m16n16 model
        // the device mma's accumulator layout: lane l, VGPR i holds D row i + 8h,
        // column l & 15; the model's ref is the contract's own double sum for D[wr][tk]
        int gbad = 0, gchecked = 0;
        for (int mt = 0; mt < 2; ++mt)
            for (int l = 0; l < 32; ++l)
            {
                const int hh = l >> 4, col = l & 15;
                for (int i = 0; i < 8; ++i)
                {
                    const int wr = 16 * mt + i + 8 * hh;
                    double ref = 0.0;
                    for (int k = 0; k < 32; ++k)
                        ref += dec(wimg[waddr_host(k, wr)]) * dec(ximg[(size_t)col * 32 + k]);
                    const float g = got[256 * mt + i * 32 + l];
                    ++gchecked;
                    if (std::fabs((double)g - ref) > 2e-3 * (1.0 + std::fabs(ref)))
                    {
                        if (gbad < 4)
                            std::printf("  step1 GEMM tile %d lane %2d vgpr %d: got %+.6f ref %+.6f\n",
                                        mt, l, i, g, ref);
                        ++gbad;
                    }
                }
            }
        std::printf("step 1b GEMM through the slots: checked %d, mismatched %d\n", gchecked, gbad);
        CHECK(hipFree(dwimg));
        CHECK(hipFree(dslots));
        CHECK(hipFree(dximg));
        CHECK(hipFree(dout));
        if (bad || gbad)
            std::exit(1);
    }

    // ---------------------------------------------------------------- step 2
    {
        // 12-tile phase: per-lane [12][8] random f32 D values; 8-tile: [8][8]
        std::vector<float> v12(32 * 12 * 8), v8(32 * 8 * 8);
        uint32_t lcg = 0x0f0f0f0fu;
        auto nextr = [&lcg]() {
            lcg = lcg * 1664525u + 1013904223u;
            return (float)((int)(lcg >> 8) - 8388608) * (3.0f / 8388608.0f);
        };
        for (auto& v : v12)
            v = nextr() * 60.0f;
        for (auto& v : v8)
            v = nextr() * 60.0f;
        float *dv, *dv8;
        uint32_t* dout;
        CHECK(hipMalloc(&dv, v12.size() * 4));
        CHECK(hipMalloc(&dv8, v8.size() * 4));
        CHECK(hipMalloc(&dout, 48 * 32 * sizeof(uint32_t)));
        CHECK(hipMemcpy(dv, v12.data(), v12.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dv8, v8.data(), v8.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemset(dout, 0, 48 * 32 * sizeof(uint32_t)));

        for (int pass = 0; pass < 3; ++pass)
        {
            const int chunked = pass == 0 ? 0 : (pass == 1 ? 1 : 2);
            step2_kernel<<<1, 32>>>(chunked == 2 ? dv8 : dv, dout, chunked);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            std::vector<uint32_t> got(48 * 32);
            CHECK(hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost));

            const int tiles = chunked == 2 ? 8 : 12;
            const float* src = chunked == 2 ? v8.data() : v12.data();
            const int n_nt = chunked == 2 ? 4 : 6;
            int bad = 0, checked = 0;
            for (int lane = 0; lane < 32; ++lane)
            {
                const int g = (lane >> 2) & 7, t = lane & 3;
                for (int mi = 0; mi < tiles / 2; ++mi)     // n_mi
                    for (int rh = 0; rh < 2; ++rh)
                        for (int nt = 0; nt < 4; ++nt)     // n_nt4
                        {
                            // the wanted cell is D[wr = 8nt + 2t + cb][tk = 16mi + 8rh + g]:
                            // tile (nt>>1, mi), lane 8rh + g + 16*(nt&1), VGPRs 2t, 2t+1
                            const int src_lane = 8 * rh + g + 16 * (nt & 1);
                            const int src_tile = (nt >> 1) * n_nt + mi;
                            const float vlo = src[((size_t)src_lane * tiles + src_tile) * 8 + 2 * t];
                            const float vhi = src[((size_t)src_lane * tiles + src_tile) * 8 + 2 * t + 1];
                            const uint32_t want = (uint32_t)b_of((_Float16)vlo)
                                                | ((uint32_t)b_of((_Float16)vhi) << 16);
                            const uint32_t have = got[((mi * 2 + rh) * 4 + nt) * 32 + lane];
                            ++checked;
                            if (have != want)
                            {
                                if (bad < 4)
                                    std::printf("  dcell lane %2d mi %d rh %d nt %d: got %08x want %08x\n",
                                                lane, mi, rh, nt, have, want);
                                ++bad;
                            }
                        }
            }
            std::printf("step 2  %s exchange: checked %d, mismatched %d\n",
                        chunked == 0 ? "twelve-tile whole-buffer"
                                     : (chunked == 1 ? "twelve-tile (2,2)-chunk" : "eight-tile"),
                        checked, bad);
            if (bad)
                std::exit(1);
        }
        CHECK(hipFree(dv));
        CHECK(hipFree(dv8));
        CHECK(hipFree(dout));
    }

    // ---------------------------------------------------------------- step 3
    {
        // every f16 of the whole 16-bit space (all 65536, incl. -0.0, negatives, NaN
        // halves, values beyond +-2 and subnormals), shuffled into the 64 pairs per lane
        // x 4 tiles the kernel stages (mode 0/1's bar is bit exact, so the whole space)
        uint16_t allbits[65536];
        for (uint32_t b = 0; b < 0x10000u; ++b)
            allbits[b] = (uint16_t)b;
        uint32_t lcg = 0x00c0ffeeu;
        auto nextr = [&lcg]() {
            lcg = lcg * 1664525u + 1013904223u;
            return lcg >> 16;
        };
        for (uint32_t b = 0xffffu; b >= 1; --b)
        {
            const uint32_t j = nextr() % (b + 1);
            const uint16_t t = allbits[b];
            allbits[b] = allbits[j];
            allbits[j] = t;
        }
        std::vector<uint32_t> pairs(32 * 32);
        for (size_t i = 0; i < pairs.size(); ++i)
        {
            const uint16_t lo = allbits[2 * i];
            const uint16_t hi = allbits[2 * i + 1];
            pairs[i] = (uint32_t)lo | ((uint32_t)hi << 16);
        }
        uint32_t* dp;
        uint8_t *di0, *di1, *di2;
        CHECK(hipMalloc(&dp, pairs.size() * 4));
        CHECK(hipMalloc(&di0, 32 * 32));
        CHECK(hipMalloc(&di1, 32 * 32));
        CHECK(hipMalloc(&di2, 32 * 32));
        CHECK(hipMemcpy(dp, pairs.data(), pairs.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemset(di0, 0, 32 * 32));
        CHECK(hipMemset(di1, 0, 32 * 32));
        CHECK(hipMemset(di2, 0, 32 * 32));
        step3_kernel<<<1, 32>>>(dp, di0, di1, di2);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint8_t> i0(32 * 32), i1(32 * 32), i2(32 * 32);
        CHECK(hipMemcpy(i0.data(), di0, i0.size(), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(i1.data(), di1, i1.size(), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(i2.data(), di2, i2.size(), hipMemcpyDeviceToHost));

        auto within_ulp = [](uint16_t a, uint16_t b) {
            // compare as f16 by ordering the sign-magnitude representation
            const auto key = [](uint16_t v) -> int {
                return (v & 0x8000) ? (int)(0x8000 - (v & 0x7fff)) : (int)(v & 0x7fff);
            };
            const int d = key(a) - key(b);
            return d >= -1 && d <= 1;
        };

        // mode 0 and mode 1: bit exact through the table
        int bad0 = 0, bad1 = 0, checked = 0;
        for (int tile = 0; tile < 2; ++tile)
            for (int lane = 0; lane < 32; ++lane)
                {
                    const int h = lane >> 4, col = lane & 15;
                    const uint32_t* p = pairs.data() + (size_t)(lane * 32) + tile * 4;
                    for (int q = 0; q < 4; ++q)
                        for (int half = 0; half < 2; ++half)
                        {
                            const uint16_t bits = (uint16_t)(half ? p[q] >> 16 : p[q] & 0xffffu);
                            const uint16_t re = relu_half(bits);
                            const uint8_t want0 = requant_code(bits);
                            const uint8_t want1 = requant_code(re);
                            // the staged byte sits at image row = row_base + (lane & 15),
                            // byte = 16*chunk + 8h + (VGPR = 2q + half)
                            const int rowi = 16 * tile + col;
                            const size_t off = (size_t)rowi * 32 + i_vgpr(q, half, h);
                            ++checked;
                            if (i0[off] != want0)
                            {
                                if (bad0 < 4)
                                    std::printf("  mode0 img row %2d byte %2d (tile %d lane %2d q %d h%d):"
                                                " got 0x%02x want 0x%02x\n",
                                                rowi, (int)off % 32, tile, lane, q, half,
                                                i0[off], want0);
                                ++bad0;
                            }
                            ++checked;
                            if (i1[off] != want1)
                            {
                                if (bad1 < 4)
                                    std::printf("  mode1 img row %2d byte %2d (tile %d lane %2d q %d h%d):"
                                                " got 0x%02x want 0x%02x\n",
                                                rowi, (int)off % 32, tile, lane, q, half,
                                                i1[off], want1);
                                ++bad1;
                            }
                        }
                }
        std::printf("step 3a stage modes 0/1 vs the table: checked %d, mode0 %d, mode1 %d mismatched\n",
                    checked, bad0, bad1);

        // mode 2: the clamped cubic then encode, one ulp per half (the device applies
        // the formula with packed f16x2 ops, the scalar _Float16 model can land one ulp
        // away -- the k16 test's step 6 bar; a logic error moves whole clamps)
        int bad2 = 0, checked2 = 0;
        for (int tile = 0; tile < 2; ++tile)
            for (int lane = 0; lane < 32; ++lane)
                {
                    const int h = lane >> 4, col = lane & 15;
                    const uint32_t* p = pairs.data() + (size_t)(lane * 32) + tile * 4;
                    for (int q = 0; q < 4; ++q)
                        for (int half = 0; half < 2; ++half)
                        {
                            const uint16_t bits = (uint16_t)(half ? p[q] >> 16 : p[q] & 0xffffu);
                            const uint16_t acted = act_model(bits);
                            const uint8_t want2 = requant_code(acted);
                            const int rowi = 16 * tile + col;
                            const size_t off = (size_t)rowi * 32 + i_vgpr(q, half, h);
                            ++checked2;
                            if (!within_ulp(i2[off], want2))
                            {
                                if (bad2 < 4)
                                    std::printf("  mode2 img row %2d byte %2d: got 0x%02x want 0x%02x (act %04x)\n",
                                                rowi, (int)off % 32, i2[off], want2, acted);
                                ++bad2;
                            }
                        }
                }
        std::printf("step 3b stage mode 2 (1 ulp): checked %d, mismatched %d\n", checked2, bad2);
        CHECK(hipFree(dp));
        CHECK(hipFree(di0));
        CHECK(hipFree(di1));
        CHECK(hipFree(di2));
        if (bad0 || bad1 || bad2)
            std::exit(1);
    }
    // ---------------------------------------------------------------- step 4
    {
        // random weights in the waddr layout (32 rows x 32 k), random staged image
        // (96 rows x 32 bytes), per-column f16 C seeds (32 columns, column c at byte 2c)
        std::vector<uint8_t> wimg(4096, 0), ximg(96 * 32, 0);
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 32; ++n)
                wimg[waddr_host(k, n)] = next_code();
        for (auto& v : ximg)
            v = next_code();
        std::vector<uint8_t> cseed(64, 0);
        for (int c = 0; c < 32; ++c)
        {
            const uint16_t b = (uint16_t)(0x3c00u - (next_code() & 0xffu));   // +-small f16
            cseed[2 * c] = (uint8_t)(b & 0xffu);
            cseed[2 * c + 1] = (uint8_t)(b >> 8);
        }
        uint8_t *dw, *dx, *dsl;
        uint32_t* dseed;
        uint32_t* dout;
        CHECK(hipMalloc(&dw, wimg.size()));
        CHECK(hipMalloc(&dx, ximg.size()));
        CHECK(hipMalloc(&dseed, cseed.size()));
        CHECK(hipMalloc(&dsl, 2048));
        CHECK(hipMalloc(&dout, 48 * 4 * 32 * 4));
        CHECK(hipMemcpy(dw, wimg.data(), wimg.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dx, ximg.data(), ximg.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dseed, cseed.data(), cseed.size(), hipMemcpyHostToDevice));
        CHECK(hipMemset(dout, 0, 48 * 4 * 32 * 4));
        step4_kernel<<<1, 32>>>(dw, dsl, dx, dseed, dout);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint32_t> got(48 * 4 * 32);
        CHECK(hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost));

        int bad = 0, checked = 0;
        for (int mt = 0; mt < 2; ++mt)
            for (int nt = 0; nt < 6; ++nt)
                for (int lane = 0; lane < 32; ++lane)
                {
                    const int h = lane >> 4, col = lane & 15;
                    for (int p = 0; p < 4; ++p)
                        for (int half = 0; half < 2; ++half)
                        {
                            // the engine cell D[wr = 16mt + 2p + half + 8h][tk = 16nt + col]:
                            // double sum over k of dec(w[waddr(k, wr)]) * dec(img[tk*32 + k])
                            // + f32(C seed f16), rounded once to f16
                            const int wr = 16 * mt + 2 * p + half + 8 * h;
                            const int tk = 16 * nt + col;
                            double ref = 0.0;
                            for (int k = 0; k < 32; ++k)
                                ref += dec(wimg[waddr_host(k, wr)]) * dec(ximg[(size_t)tk * 32 + k]);
                            const uint16_t seedbits =
                                (uint16_t)(cseed[2 * wr] | (cseed[2 * wr + 1] << 8));
                            ref += (double)h_of(seedbits);
                            const uint16_t want = b_of((_Float16)ref);
                            const uint16_t bits = (uint16_t)(half ? got[((mt * 6 + nt) * 4 + p) * 32 + lane] >> 16
                                                                  : got[((mt * 6 + nt) * 4 + p) * 32 + lane] & 0xffffu);
                            ++checked;
                            if (bits != want)
                            {
                                if (bad < 6)
                                    std::printf("  D[wr %2d][tk %2d] (mt %d nt %d lane %2d p %d h%d):"
                                                " got %04x want %04x\n",
                                                wr, tk, mt, nt, lane, p, half, bits, want);
                                ++bad;
                            }
                        }
                }
        std::printf("step 4  integrated phase with C seeds: checked %d, mismatched %d\n",
                    checked, bad);
        CHECK(hipFree(dw));
        CHECK(hipFree(dx));
        CHECK(hipFree(dseed));
        CHECK(hipFree(dsl));
        CHECK(hipFree(dout));
        if (bad)
            std::exit(1);
    }

    // ---------------------------------------------------------------- step 5
    {
        // phase 1: per-column C at w+2048, weights at w+4096; phase 2: per-column C at
        // w+5120, weights at w+6144, off the relu-staged image
        std::vector<uint8_t> wimg(10240, 0), ximg(96 * 32, 0);
        for (auto& v : ximg)
            v = next_code();
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 32; ++n)
                wimg[4096 + waddr_host(k, n)] = next_code();
        for (int k = 0; k < 32; ++k)
            for (int n = 0; n < 32; ++n)
                wimg[6144 + waddr_host(k, n)] = next_code();
        std::vector<uint8_t> cs1(64, 0), cs2(64, 0);
        for (int c = 0; c < 32; ++c)
        {
            const uint16_t b = (uint16_t)(0x3c00u - (next_code() & 0xffu));
            cs1[2 * c] = (uint8_t)(b & 0xffu);
            cs1[2 * c + 1] = (uint8_t)(b >> 8);
            const uint16_t b2 = (uint16_t)(0x3c00u - (next_code() & 0xffu));
            cs2[2 * c] = (uint8_t)(b2 & 0xffu);
            cs2[2 * c + 1] = (uint8_t)(b2 >> 8);
        }
        uint8_t *dw, *dx, *dst;
        uint32_t *ds1, *ds2;
        uint32_t* dout;
        CHECK(hipMalloc(&dw, wimg.size()));
        CHECK(hipMalloc(&dx, ximg.size()));
        CHECK(hipMalloc(&ds1, cs1.size()));
        CHECK(hipMalloc(&ds2, cs2.size()));
        CHECK(hipMalloc(&dst, 96 * 32));
        CHECK(hipMalloc(&dout, 48 * 4 * 32 * 4));
        CHECK(hipMemcpy(dw, wimg.data(), wimg.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dx, ximg.data(), ximg.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(ds1, cs1.data(), cs1.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(ds2, cs2.data(), cs2.size(), hipMemcpyHostToDevice));
        CHECK(hipMemset(dst, 0, 96 * 32));
        CHECK(hipMemset(dout, 0, 48 * 4 * 32 * 4));
        step5_kernel<<<1, 32>>>(dw, dx, ds1, ds2, dst, dout);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::vector<uint8_t> st(96 * 32, 0);
        CHECK(hipMemcpy(st.data(), dst, st.size(), hipMemcpyDeviceToHost));
        std::vector<uint32_t> got(48 * 4 * 32);
        CHECK(hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost));

        // 5a: the staged image BIT-exactly: D1 cells -> relu per element -> RNE-satfinite
        // e4m3 code per element
        int sbad = 0, schecked = 0;
        for (int mt = 0; mt < 2; ++mt)
            for (int nt = 0; nt < 6; ++nt)
                for (int lane = 0; lane < 32; ++lane)
                {
                    const int h = lane >> 4, col = lane & 15;
                    for (int p = 0; p < 4; ++p)
                        for (int half = 0; half < 2; ++half)
                        {
                            const int wr = 16 * mt + 2 * p + half + 8 * h;
                            const int tk = 16 * nt + col;
                            double ref = 0.0;
                            for (int k = 0; k < 32; ++k)
                                ref += dec(wimg[4096 + waddr_host(k, wr)]) * dec(ximg[(size_t)tk * 32 + k]);
                            const uint16_t seedbits = (uint16_t)(cs1[2 * wr] | (cs1[2 * wr + 1] << 8));
                            ref += (double)h_of(seedbits);
                            uint16_t d1bits = b_of((_Float16)ref);
                            const uint16_t re = relu_half(d1bits);
                            const uint8_t want = requant_code(re);
                            const size_t off = (size_t)tk * 32 + (16 * mt + 2 * p + half + 8 * h);
                            ++schecked;
                            if (st[off] != want)
                            {
                                if (sbad < 4)
                                    std::printf("  staged[col %2d][k %2d] = 0x%02x, expected 0x%02x\n",
                                                tk, wr, st[off], want);
                                ++sbad;
                            }
                        }
                }
        std::printf("step 5a staged image (relu requant) bit exact: checked %d, mismatched %d\n",
                    schecked, sbad);

        // 5b: phase 2's pairs off the staged image, with phase 2's own per-column C
        int bad = 0, checked = 0;
        for (int mt = 0; mt < 2; ++mt)
            for (int nt = 0; nt < 6; ++nt)
                for (int lane = 0; lane < 32; ++lane)
                {
                    const int h = lane >> 4, col = lane & 15;
                    for (int p = 0; p < 4; ++p)
                        for (int half = 0; half < 2; ++half)
                        {
                            const int wr = 16 * mt + 2 * p + half + 8 * h;
                            const int tk = 16 * nt + col;
                            // phase 2's GEMM: sum over k of dec(w2[k][wr]) * the staged
                            // byte's decode (relu(D1 cell (k, tk)) requantised to e4m3,
                            // the operand the device's GEMM reads) + f32(C seed f16),
                            // rounded once to f16
                            double ref = 0.0;
                            for (int k = 0; k < 32; ++k)
                            {
                                double d1 = 0.0;
                                for (int j = 0; j < 32; ++j)
                                    d1 += dec(wimg[4096 + waddr_host(j, k)]) * dec(ximg[(size_t)tk * 32 + j]);
                                const uint16_t seedbits = (uint16_t)(cs1[2 * k] | (cs1[2 * k + 1] << 8));
                                d1 += (double)h_of(seedbits);
                                const uint16_t re = relu_half(b_of((_Float16)d1));
                                ref += dec(wimg[6144 + waddr_host(k, wr)]) * dec(requant_code(re));
                            }
                            const uint16_t seed2bits = (uint16_t)(cs2[2 * wr] | (cs2[2 * wr + 1] << 8));
                            ref += (double)h_of(seed2bits);
                            const uint16_t want = b_of((_Float16)ref);
                            const uint16_t bits = (uint16_t)(half ? got[((mt * 6 + nt) * 4 + p) * 32 + lane] >> 16
                                                                  : got[((mt * 6 + nt) * 4 + p) * 32 + lane] & 0xffffu);
                            ++checked;
                            if (bits != want)
                            {
                                if (bad < 4)
                                    std::printf("  D2[wr %2d][tk %2d] (mt %d nt %d lane %2d p %d h%d):"
                                                " got %04x want %04x\n",
                                                wr, tk, mt, nt, lane, p, half, bits, want);
                                ++bad;
                            }
                        }
                }
        std::printf("step 5b phase 2 pairs off the staged image: checked %d, mismatched %d\n",
                    checked, bad);
        CHECK(hipFree(dw));
        CHECK(hipFree(dx));
        CHECK(hipFree(ds1));
        CHECK(hipFree(ds2));
        CHECK(hipFree(dst));
        CHECK(hipFree(dout));
        if (sbad || bad)
            std::exit(1);
    }
    return 0;
}
