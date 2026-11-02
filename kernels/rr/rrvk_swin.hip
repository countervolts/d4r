// One Ray Reconstruction Swin stage (presets D and E: enc1 - enc5, dec4 - dec1) as a native kernel for vkhip
// (engine/vkhip/vkhip_rr.h): engine/rr/swin_m.comp in the arithmetic the engine models ship (D4R_RR_RELAXED=3:
// RR_RELAXED=2 with RR_NATIVE), with the window's work spread over more waves.
//
// swin_m.comp runs one 8x8-token window per workgroup of four waves, wave w owning query tile w (16 tokens) for the
// whole stage: the heads one after another, then the MLP branches one after another. The deep stages have few
// windows (enc5 at 1280x720: 18), so the GPU idles while each workgroup walks that chain. Here a window has
// NW = 4 * HS waves (HS = min(NH, 4) heads at a time) and every phase is split by what is independent in it:
//   - K / V projections: HS heads x 12 (token tile, channel tile) jobs;
//   - attention: one (head, query tile) per wave; the head's output O^T goes to LDS;
//   - out-projection, MLP output and residual: wave (q, g) owns query tile q and channel tiles m = g, g + HS, ...
//     and adds the heads' (then the branches') products in swin_m.comp's order, so every element sees the same
//     sequence of 32-deep steps as there;
//   - MLP hidden layer: (branch, query tile) jobs, written to LDS;
//   - expand (decoders) and merge (encoders): their (row, channel tile) jobs.
// The arithmetic is swin_m.comp's line for line (f32 accumulators, one saturating conversion to FP8, the cubic
// softmax table, the f32 MLP activation), so a stage's output equals the shader's.
//
// Inputs: the arena by device address (NGX's plane layout), the stage's packed weights (engine/rr/swin_pack.py, with
// the V tiles stored [n][k] and the bias [query][key]: vkhip_rr.cpp rearranges them), and the stage's images as raw
// descriptors (VK_EXT_descriptor_buffer). The including build defines the stage (swin_stage.h) and RRVK_NAME.
#include <hip/hip_runtime.h>
#include <stdint.h>
#ifdef RRVK_DEC0
// decoder 0 (engine/rr/dec0_m.comp): a decoder with C = 32 whose outputs are the output stage's eight FP8 codes per
// pixel, the tanh of four channels (surface 144) and the filter (surface 112); engine/rr/dec0_pack.py's weights
#include "dec0_m_w.h"
#define KIND 1
#define C 32
#define CX 64
#define NH 2
#define DEC0 1
#define W_TABLE7 W_TABLE0
#define W_GAIN14 W_GAIN1
#define W_TABLE16 W_TABLE1
#define ENC0 0
#elif defined(RRVK_ENC0)
// encoder 0 (engine/rr/enc0_m_tail.glsl): an encoder with C = 32 whose window tokens are embedded from the 10 x 10
// pixels' features that enc0's translated gather (enc0_g.spv) leaves in scratch, and whose filter runs over its
// noise grid; engine/rr/enc0_pack.py's weights
#include "enc0_m_w.h"
#define KIND 0
#define C 32
#define CX 64
#define NH 2
#define DEC0 0
#define ENC0 1
#define W_GAIN14 W_GAIN1
#define W_TABLE16 W_TABLE1
#else
#include "swin_stage.h"
#define DEC0 0
#define ENC0 0
#endif

#pragma clang fp contract(off)

#define KC (C / 32)
#define MT (C / 16)
#define NB (C / 8)
#define HEAD (W_WK1 - W_WK0)
#define BRANCH (W_W1_1 - W_W1_0)
#ifndef RRVK_HS
#define HS (NH < 4 ? NH : 4)
#else
#define HS RRVK_HS
#endif
// the residual phases: unit (channel tile m, query group g of QB query tiles), one per wave, so that a wave loads a
// weight tile once for QB products
#ifndef RRVK_QB
#define QB 2
#else
#define QB RRVK_QB
#endif
#define NU (MT * 4 / QB)
#ifndef RRVK_NW
#define NW (4 * HS > NU ? 4 * HS : NU)
#else
#define NW RRVK_NW
#endif
static_assert(NW >= NU && NW >= 4 * HS && NW <= 32, "waves");

namespace rrvk
{
typedef _Float16 hf;
typedef float f8v __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef int i2v __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef hf h2 __attribute__((ext_vector_type(2)));   // packed binary16 arithmetic (v_pk_*), rounded per element
typedef hf h8 __attribute__((ext_vector_type(8)));
#define RD __device__ __forceinline__

// the image instructions (LLVM's intrinsics; this clang has no builtins for them)
__device__ v4f image_load(int dmask, int x, int y, v8i rsrc, int tfe, int aux) __asm("llvm.amdgcn.image.load.2d.v4f32.i32");
__device__ v4f image_sample(int dmask, float u, float v, v8i rsrc, v4i samp, bool unorm, int tfe, int aux)
    __asm("llvm.amdgcn.image.sample.lz.2d.v4f32.f32");
__device__ void image_store(v4f d, int dmask, int x, int y, v8i rsrc, int tfe, int aux) __asm("llvm.amdgcn.image.store.2d.v4f32.i32");

// the kernel arguments (by value: vkhip's push constant points at them): vkhip_rr.cpp's RRArgs. Encoder 0 has no input
// tokens: inOff is the end of its gather's scratch (window i's 4000 bytes end (i + 1) x 4000 before it), skOff the
// width of the dispatch's window grid
struct Args {
    uint64_t arena, w;
    int32_t ex, ey;
    uint32_t inOff, skOff, outOff, mrgOff, shift;
    float iw, ih;
    uint32_t biasF;   // the attention bias as f32, [head][query][key], after the packed weights (vkhip_net.cpp)
    v8i t88, t96, t104, s112, s120, s128;
    v4i linear;
};

RD uint32_t inter(uint32_t j) { return (j & ~31u) | (((j & 31u) >> 2) << 1) | (j & 1u) | (((j >> 1) & 1u) << 4); }
RD int mirror(int v, int n) { v = v < 0 ? -v : v; const int m = 2 * n - 2 - v; return v < m ? v : m; }
RD hf hb(uint32_t b) { return __builtin_bit_cast(hf, (uint16_t)b); }
RD uint32_t bh(hf h) { return __builtin_bit_cast(uint16_t, h); }
RD uint32_t other_half(uint32_t v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false); }
// the four e4m3 codes of a word as binary16 (the conversion's byte select, no shifts)
RD void e4x4(uint32_t w, hf* o)
{
    o[0] = (hf)__builtin_amdgcn_cvt_f32_fp8((int)w, 0); o[1] = (hf)__builtin_amdgcn_cvt_f32_fp8((int)w, 1);
    o[2] = (hf)__builtin_amdgcn_cvt_f32_fp8((int)w, 2); o[3] = (hf)__builtin_amdgcn_cvt_f32_fp8((int)w, 3);
}
// four values -> four e4m3 codes (MODE.FP16_OVFL is set in the prologue: a finite value beyond 448 saturates, as the
// shaders' saturatedConvertEXT does)
RD uint32_t pk4(float a, float b, float c, float d)
{
    const int w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    return (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
}
// MODE.FP16_OVFL also clamps every binary16 overflow to 65504 where the shaders (and NVIDIA) get infinity. That
// matters for a token norm's sum of squares (the softmax sums stay below 2^16: the cubic numerator is at most 2^9),
// which then scales the token by 1 / 256 instead of 0 (preset E's encoder 5); an f32 residual beyond the binary16
// range, converted for the MLP's norm, is left at 65504 (handling it costs 4% of the network for no visible case). A sum of non-negative terms that
// overflowed stays at 65504 under the mode, so 65504 is taken as the infinity it stands for (an exact 65504 without
// overflow would be taken too; that sum needs every rounding to land on the largest binary16 number).
RD hf sumsq(hf s) { return s >= (hf)65504.0f ? hb(0x7c00u) : s; }
RD i2v q8(f8v a) { return (i2v){(int)pk4(a[0], a[1], a[2], a[3]), (int)pk4(a[4], a[5], a[6], a[7])}; }
RD f8v zero8() { return (f8v){0, 0, 0, 0, 0, 0, 0, 0}; }
// one 32-deep step: the f32 accumulator through both products (swin_m.comp's K32 at RR_RELAXED)
RD void k32(f8v& c, i2v a0, i2v b0, i2v a1, i2v b1)
{
#ifdef RRVK_HALFMMA   // profiling: one product per step (wrong results)
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0 ^ a1, b0 ^ b1, c);
#else
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, b0, c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, b1, c);
#endif
}
RD i2v op(const uint8_t* p) { return *(const i2v*)p; }
// a weight tile's A operand (row col, k 8hl .. 8hl + 7), row-major 16 x 16 tiles of 256 bytes
#if defined(RRVK_WZERO)   // profiling: no weight-tile loads at all
RD i2v tile(const uint8_t* w, uint32_t base, uint32_t t, uint32_t col, uint32_t hl) { return (i2v){(int)(base + t), (int)(col + hl)}; }
#elif defined(RRVK_WFAKE)   // profiling: every weight tile read from one place (wrong results, the weight traffic gone)
RD i2v tile(const uint8_t* w, uint32_t base, uint32_t t, uint32_t col, uint32_t hl) { return op(w + ((base + 256u * t) & 255u) + 16u * col + 8u * hl); }
#else
// the tile's start (uniform: a scalar base) plus the lane's offset (one 32-bit VGPR), so that the load takes the
// scalar-base form instead of a 64-bit address per lane
RD i2v tile(const uint8_t* w, uint32_t base, uint32_t t, uint32_t col, uint32_t hl)
{
    const uint8_t* p = w + (base + 256u * t);
    return op(p + (16u * col + 8u * hl));
}
#endif
RD hf h16at(const uint8_t* w, uint32_t byte) { return *(const hf*)(w + byte); }
RD h2 h2at(const uint8_t* w, uint32_t byte) { return *(const h2*)(w + byte); }   // 4-byte aligned
// eight binary16 values at a 16-byte aligned byte offset (every bias row: the packed offsets are multiples of 256)
RD f8v bias8(const uint8_t* w, uint32_t byte)
{
    const h8 h = *(const h8*)(w + byte);
    f8v r;
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = (float)h[i];
    return r;
}
// the cubic softmax numerator of two binary16 scores, computed (packed) as engine/rr/enc1_pack.cubic_table tabulates
// it: m = clamp(f16(x * 0.017181396484375), +-0.55615234375) (min first, so that a NaN score takes the upper bound
// as the table's does), u = fma(m, fma(m, -m, 0.92724609375), 1.375), then u's mantissa as the exponent's bits
RD h2 cubic(h2 x)
{
    const h2 lim = {(hf)0.55615234375f, (hf)0.55615234375f};
    h2 m = x * h2{(hf)0.017181396484375f, (hf)0.017181396484375f};
    m = __builtin_elementwise_max(__builtin_elementwise_min(m, lim), -lim);
    const h2 t = __builtin_elementwise_fma(m, -m, h2{(hf)0.92724609375f, (hf)0.92724609375f});
    const h2 u = __builtin_elementwise_fma(m, t, h2{(hf)1.375f, (hf)1.375f});
    const uint32_t b = __builtin_bit_cast(uint32_t, u);
    return __builtin_bit_cast(h2, (b << 5) & 0x7fe07fe0u);
}
// two f32 values rounded to binary16 into one register (v_fma_mixlo / mixhi: x * 1 + 0, rounded once as a conversion;
// a zero comes out positive, which the cubic does not distinguish), one instruction fewer than two conversions and a pack
RD h2 pack16(float a, float b)
{
    uint32_t r;
    __asm("v_fma_mixlo_f16 %0, %1, 1.0, 0\n\tv_fma_mixhi_f16 %0, %2, 1.0, 0" : "=&v"(r) : "v"(a), "v"(b));
    return __builtin_bit_cast(h2, r);
}
// the high binary16 element as f32 (exact; a negative zero comes out positive), without the shift a conversion needs
RD float hi32(h2 v)
{
    float r;
    __asm("v_fma_mix_f32 %0, %1, 1.0, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]" : "=v"(r) : "v"(__builtin_bit_cast(uint32_t, v)));
    return r;
}
RD hf rcp16(float x) { return (hf)__builtin_amdgcn_rcpf(x); }
RD hf sig(hf v) { return rcp16(__builtin_amdgcn_exp2f((float)v * -1.4426950216293335f) + 1.0f); }
template <typename T> RD T tree(const T* v)
{
#if KC == 1
    return v[0];
#elif KC == 2
    return v[0] + v[1];
#elif KC == 3
    return (v[0] + v[1]) + v[2];
#elif KC == 4
    return (v[0] + v[1]) + (v[2] + v[3]);
#else
    return ((v[0] + v[1]) + (v[2] + v[3])) + v[4];
#endif
}
RD void key_xy(uint32_t t, int x0, int y0, int& x, int& y)
{
    if (t < 64u) { x = x0 + (int)(t & 7u); y = y0 + (int)(t >> 3); return; }
    const int i = (int)(t & 7u), side = (int)((t - 64u) >> 3);
    x = side < 2 ? x0 + i : (side == 2 ? x0 - 1 : x0 + 8);
    y = side == 0 ? y0 - 1 : (side == 1 ? y0 + 8 : y0 + i);
}
// dec0_m.comp's tanhNV (NVIDIA's tanh.approx sequence)
RD float tanhNV(float t)
{
    const float m = __builtin_fabsf(t), s = t * t;
    const float big = m >= 9.0109138f ? 1.0f : __builtin_fmaf(__builtin_amdgcn_rcpf(__builtin_amdgcn_exp2f(m * 2.885390043258667f) + 1.0f), -2.0f, 1.0f);
    return m >= 0.6f ? (t < 0.0f ? -big : big) :
        __builtin_fmaf(__builtin_fmaf(__builtin_fmaf(__builtin_fmaf(__builtin_fmaf(0.01573968306183815f, s, -0.05230396240949631f), s, 0.13315297663211823f), s, -0.33332768082618713f), s, 0.0f), t, t);
}
RD int clampTap(int v, int n) { return v < 0 ? 1 : (v >= n ? n - 2 : v); }
// the waves share only LDS: a barrier with an LDS fence (__syncthreads' global one invalidates the cache each time)
RD void sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// ---------------------------------------------------------------- LDS
// One pool, laid out per phase (byte offsets):
//   expand (decoders): EX after XT (48 low-res rows x 4 sub-positions x C, slot order), LOW after EX (48 x CX)
//   prologue / residual: XT (96 key tokens x C, slot order), EX
//   attention: XT; KV (HS slots of K 96 x 32 and V^T 32 x 96) and OB (NH x 4 query tiles x 512) over EX
//   MLP: NBUF (the normalised tokens, 64 x C), then RV (the residual as binary16, 64 x C) or HB (NB x 4 x 512)
//   outputs: MGR (encoders: 16 merge rows x 4C), TB (the output tokens, 64 x C), then OUTF / TAP / O112
// The rows that WMMA operands are read from (16 lanes, one row each) are padded by 8 bytes, so that the lanes of a
// load fall in different banks (unpadded, a row is a multiple of 256 bytes or of 32: up to 16 lanes on one bank).
#define XS (C + 8)        // XT row
#define ES (4 * C + 8)    // EX and MGR row
#define LS (CX + 8)       // LOW row
#define KS_S 40           // K row (32 channels)
#define VS_S 104          // V^T row (96 key slots)
#define KV_SLOT (96 * KS_S + 32 * VS_S)
#define XT_AT 0
#define XT_B (96 * XS)
#define XF_AT XT_B                      // encoder 0's embedding: the key pixels' 32 FP8 features (plane 0 | plane 1)
#define RX_AT (XF_AT + 96 * XS)         // and the queries' residual (RV's layout), both read before the attention
#define EX_AT XT_B
#define EX_B (KIND == 1 ? 48 * ES : 0)
#define LOW_AT (EX_AT + EX_B)
#define LOW_B (KIND == 1 ? 48 * LS : 0)
#define KV_AT XT_B
#define KV_B (HS * KV_SLOT)
#define OB_AT (KV_AT + KV_B)
#define OB_B (HS * 2048)   // the current head group's O^T, projected before the next group
#define NBUF_AT 0
#define NBUF_B (64 * C)
#define RV_AT NBUF_B
#define HB_AT NBUF_B
#define HB_B (NB * 2048)
#define MLP_END (NBUF_B + (HB_B > 128 * C ? HB_B : 128 * C))
#define MGR_AT 0
#define TB_AT (16 * ES)
#define TAIL_AT (TB_AT + 64 * C)
#define OUTF_C (DEC0 ? 32 : 16)   // OUTF: the head per window token (decoder 0: the final tokens, natural channels)
#define TAIL_B (64 * OUTF_C * 2 + 100 * 4 * 2 + 64 * 4 * 2)
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define POOL_B MAX2(MAX2(LOW_AT + LOW_B, OB_AT + OB_B), MAX2(MLP_END, TAIL_AT + TAIL_B))
static_assert(POOL_B <= 65536, "LDS");
} // namespace rrvk

using namespace rrvk;

// for the host (engine/vkhip/vkhip_net.cpp RRStages): the stage's shape, the launch and the weight offsets it rearranges
extern "C" __device__ __attribute__((used)) uint32_t rrvk_layout[8] = {KIND, C, CX, NH, NW, W_WV0, W_BIAS0, HEAD};


#ifdef RRVK_STOP   // profiling: stop after phase RRVK_STOP (the work so far is kept alive by an impossible store)
#define STOP(k) if (RRVK_STOP == k) { if (A.shift == 0xdeadbeefu) for (uint32_t qi = 0; qi < QB; ++qi) for (int i = 0; i < 8; ++i) arena[tid * 64u + 8u * qi + i] = __builtin_bit_cast(uint32_t, R[qi][i]) ^ *(const uint32_t*)(pool + 4u * tid); return; }
#else
#define STOP(k)
#endif

extern "C" __global__ void __launch_bounds__(32 * NW) RRVK_NAME(const Args A)
{
    __shared__ __attribute__((aligned(16))) uint8_t pool[POOL_B];
#ifdef RRVK_EXIT   // profiling: nothing (the dispatch alone)
    if (A.shift != 0xdeadbeefu) return;
#endif
    // global, not flat: the generic pointers below are cast from these, so LLVM keeps every access a global one
    const uint8_t* __restrict__ w = (const uint8_t*)(const __attribute__((address_space(1))) uint8_t*)A.w;
    uint32_t* __restrict__ arena = (uint32_t*)(__attribute__((address_space(1))) uint32_t*)A.arena;

    // the wave index in an SGPR: the compiler cannot know that threadIdx.z is uniform per wave, and everything that
    // depends on it (jobs, tiles, weight addresses, branches) would otherwise be computed per lane
    __builtin_amdgcn_s_setreg(1 | (23 << 6), 1);   // MODE.FP16_OVFL: saturating FP8 conversions (and sumsq)
    const uint32_t lane = threadIdx.x, wv = __builtin_amdgcn_readfirstlane(threadIdx.z), tid = 32u * wv + lane;
    const uint32_t hl = lane >> 4, col = lane & 15u;
    const uint32_t q = wv & 3u;                   // the attention: query tile q of head slot wv / 4
    const bool unit = wv < NU;                     // the residual phases: channel tile um, query tiles QB ug, ...
    const uint32_t um = wv % MT, ug = wv / MT;
    const int ex = A.ex, ey = A.ey, hx = ex / 2, hy = ey / 2;
    const uint32_t inOff = A.inOff, outOff = A.outOff;
    const int x0 = 8 * (int)blockIdx.x - (int)(A.shift & 0xffffu), y0 = 8 * (int)blockIdx.y - (int)(A.shift >> 16);
    const uint32_t plane = (uint32_t)(ex * ey) * 32u;
#if KIND == 1
    const uint32_t skOff = A.skOff;
    const uint32_t lplane = (uint32_t)(hx * hy) * 32u;
#define TOKENS skOff
#else
#define TOKENS inOff
#endif

#if KIND == 1
    // ---------------------------------------------------------------- expand: the CX channels one level down, 2x2
    const int X0 = (x0 >> 1) - 1, Y0 = (y0 >> 1) - 1;
    {
        uint32_t* low = (uint32_t*)(pool + LOW_AT);
        for (uint32_t i = tid; i < 48u * (CX / 4u); i += 32u * NW) {
            const uint32_t r = i / (CX / 4u), wd = i % (CX / 4u);
            uint32_t u = 0u;
            if (r < 36u) {
                const int X = mirror(X0 + (int)(r % 6u), hx), Y = mirror(Y0 + (int)(r / 6u), hy);
                u = arena[(inOff + (wd >> 3) * lplane + (uint32_t)(Y * hx + X) * 32u) / 4u + (wd & 7u)];
            }
            low[r * (LS / 4u) + wd] = u;
        }
    }
    sync();
    // jobs: channel tile m of the 4C over the three row tiles n (the A operand the expansion weights, loaded once; B
    // the low-res tokens)
    for (uint32_t m = wv; m < 4u * MT; m += NW) {
        i2v we[CX / 16];
#pragma unroll
        for (uint32_t kt = 0; kt < CX / 16; ++kt) we[kt] = tile(w, W_WE, m * (CX / 16) + kt, col, hl);
        const f8v b = bias8(w, W_BE + 2u * (16u * m + 8u * hl));
#pragma unroll
        for (uint32_t n = 0; n < 3; ++n) {
            f8v a = b;
            const uint8_t* lb = pool + LOW_AT + (16u * n + col) * LS + 8u * hl;
#pragma unroll
            for (uint32_t kt = 0; kt < CX / 16; kt += 2) k32(a, we[kt], op(lb + 16u * kt), we[kt + 1], op(lb + 16u * kt + 16u));
            *(i2v*)(pool + EX_AT + (16u * n + col) * ES + 16u * m + 8u * hl) = q8(a);
        }
    }
    sync();
#endif

#if ENC0
    // ---------------------------------------------------------------- prologue: the embedding of the gathered features
    const uint8_t* const G = (const uint8_t*)arena + (A.inOff - (blockIdx.y * A.skOff + blockIdx.x + 1u) * 4000u);
    if (tid < 96u) {
        int kx, ky;
        key_xy(inter(tid), x0, y0, kx, ky);
        const uint32_t g = (uint32_t)((ky - y0 + 1) * 10 + (kx - x0 + 1));
        uint8_t* xf = pool + XF_AT + tid * XS;
        *(v4i*)xf = *(const v4i*)(G + 800u + g * 16u);
        *(v4i*)(xf + 16) = *(const v4i*)(G + 2400u + g * 16u);
    }
    sync();
#if defined(RRVK_STOP) && RRVK_STOP == 0
    if (A.shift == 0xdeadbeefu) arena[tid] = *(const uint32_t*)(pool + 4u * tid);
    return;
#endif
    // jobs: token tile n; x = W2 q8(relu(W1 f + b1)) + b2, its norm to XT and, for the queries, x + table 0
    for (uint32_t n = wv; n < 6u; n += NW) {
        const uint8_t* xf = pool + XF_AT + (16u * n + col) * XS + 8u * hl;
        const i2v f0 = op(xf), f1 = op(xf + 16);
        i2v hbv[2];
#pragma unroll
        for (uint32_t m = 0; m < 2; ++m) {
            f8v a = bias8(w, W_B1 + 2u * (16u * m + 8u * hl));
            k32(a, tile(w, W_W1, 2u * m, col, hl), f0, tile(w, W_W1, 2u * m + 1u, col, hl), f1);
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = __builtin_fmaxf(a[i], 0.0f);
            hbv[m] = q8(a);
        }
        hf xv[2][8];
#pragma unroll
        for (uint32_t m = 0; m < 2; ++m) {
            f8v a = bias8(w, W_B2 + 2u * (16u * m + 8u * hl));
            k32(a, tile(w, W_W2, 2u * m, col, hl), hbv[0], tile(w, W_W2, 2u * m + 1u, col, hl), hbv[1]);
#pragma unroll
            for (int i = 0; i < 8; ++i) xv[m][i] = (hf)a[i];
            if (n < 4u) {
                hf* rx = (hf*)(pool + RX_AT + ((n * MT + m) * 32u + lane) * 16u);
#pragma unroll
                for (int i = 0; i < 8; ++i) rx[i] = xv[m][i] + h16at(w, W_TABLE0 + 2u * (16u * m + 8u * hl + (uint32_t)i));
            }
        }
        hf h1[2];
#pragma unroll
        for (uint32_t c = 0; c < 2; ++c) {
            hf ht[2];
#pragma unroll
            for (uint32_t tq = 0; tq < 2; ++tq) {
                const uint32_t i0 = 4u * tq + c, i1 = i0 + 2u;
                const hf a0 = xv[0][i0] * xv[0][i0], a1 = xv[0][i1] * xv[0][i1], b0 = xv[1][i0] * xv[1][i0], b1 = xv[1][i1] * xv[1][i1];
                const hf p0 = a0 + a1, p1 = b0 + b1;
                ht[tq] = p0 + p1;
            }
            h1[c] = ht[0] + ht[1];
        }
        const uint32_t other = other_half(bh(h1[0]) | (bh(h1[1]) << 16));
        const hf g0 = h1[0] + hb(other & 0xffffu), g1 = h1[1] + hb(other >> 16);
        const hf sm = sumsq(g0 + g1);
        const hf se = sm + (hf)0.0001220703125f;
        const hf rr = (hf)__builtin_amdgcn_rsqf((float)se);
#pragma unroll
        for (uint32_t m = 0; m < 2; ++m) {
            float nv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const hf g = rr * h16at(w, W_GAIN0 + 2u * inter(16u * m + 8u * hl + (uint32_t)i));
                nv[i] = (float)(hf)(xv[m][i] * g);
            }
            *(i2v*)(pool + XT_AT + (16u * n + col) * XS + 16u * m + 8u * hl) =
                (i2v){(int)pk4(nv[0], nv[1], nv[2], nv[3]), (int)pk4(nv[4], nv[5], nv[6], nv[7])};
        }
    }
    sync();
    f8v R[QB];
    if (unit)
#pragma unroll
        for (uint32_t qi = 0; qi < QB; ++qi) {
            const hf* rx = (const hf*)(pool + RX_AT + (((QB * ug + qi) * MT + um) * 32u + lane) * 16u);
#pragma unroll
            for (int i = 0; i < 8; ++i) R[qi][i] = (float)rx[i];
        }
#else
    // ---------------------------------------------------------------- prologue: one key token per thread
    if (tid < 96u) {
        int kx, ky;
        key_xy(inter(tid), x0, y0, kx, ky);
        const int xx = mirror(kx, ex), yy = mirror(ky, ey);
        const uint32_t tb = (TOKENS + (uint32_t)(yy * ex + xx) * 32u) / 4u;
#if KIND == 1
        const uint32_t r = (uint32_t)(((ky >> 1) - Y0) * 6 + ((kx >> 1) - X0)), s = (uint32_t)(2 * (ky & 1) + (kx & 1));
        const uint32_t* exw = (const uint32_t*)(pool + EX_AT);
#endif
        hf v[C];
#pragma unroll
        for (uint32_t wd = 0; wd < C / 4; ++wd) {
            const uint32_t t = arena[tb + (wd >> 3) * (plane >> 2) + (wd & 7u)];
#if KIND == 1
            const uint32_t e = exw[r * (ES / 4u) + s * (C / 4u) + wd];
            {
                hf ee[4], tt[4];
                e4x4(e, ee); e4x4(t, tt);
#pragma unroll
                for (int b = 0; b < 4; ++b) v[4u * wd + b] = ee[b] + tt[b];
            }
#else
            e4x4(t, v + 4u * wd);
#endif
        }
        h2 h[4];
#pragma unroll
        for (uint32_t qq = 0; qq < 4; ++qq) {
            h2 hh[2];
#pragma unroll
            for (uint32_t half_ = 0; half_ < 2; ++half_) {
                h2 pr[KC];
#pragma unroll
                for (uint32_t p = 0; p < KC; ++p) {
                    const uint32_t o = 32u * p + 4u * (qq + 4u * half_);
                    const h2 a = h2{v[o], v[o + 1]} * h2{v[o], v[o + 1]};
                    const h2 c = h2{v[o + 2], v[o + 3]} * h2{v[o + 2], v[o + 3]};
                    pr[p] = a + c;
                }
                hh[half_] = tree(pr);
            }
            h[qq] = hh[0] + hh[1];
        }
        const h2 h0 = h[0] + h[1], h2_ = h[2] + h[3];
        const h2 hs = h0 + h2_;
        const hf sm = sumsq(hs.x + hs.y);
        const hf se = sm + (hf)0.0001220703125f;
        const hf rr = (hf)__builtin_amdgcn_rsqf((float)se);
#pragma unroll
        for (uint32_t wd = 0; wd < C / 4; ++wd) {
            float n[4];
#pragma unroll
            for (uint32_t b = 0; b < 4; b += 2) {
                const uint32_t j = 4u * wd + b;
                const h2 g = h2{rr, rr} * h2at(w, W_GAIN0 + 2u * inter(j));
                const h2 nn = h2{v[j], v[j + 1]} * g;
                n[b] = (float)nn.x; n[b + 1] = (float)nn.y;
            }
            *(uint32_t*)(pool + XT_AT + tid * XS + 4u * wd) = pk4(n[0], n[1], n[2], n[3]);
        }
    }

    // ---------------------------------------------------------------- the residual: the window token + table 7
    f8v R[QB];
    if (unit) {
#pragma unroll
        for (uint32_t qi = 0; qi < QB; ++qi) {
            const uint32_t tq = inter(16u * (QB * ug + qi) + col), m = um;
            const int qx = x0 + (int)(tq & 7u), qy = y0 + (int)(tq >> 3);
            const uint32_t tb = (TOKENS + (uint32_t)(mirror(qy, ey) * ex + mirror(qx, ex)) * 32u) / 4u;
#if KIND == 1
            const uint32_t r = (uint32_t)(((qy >> 1) - Y0) * 6 + ((qx >> 1) - X0)), s = (uint32_t)(2 * (qy & 1) + (qx & 1));
#endif
            const uint32_t ta = tb + (m >> 1) * (plane >> 2) + 4u * (m & 1u) + 2u * hl;
            const uint32_t t[2] = {arena[ta], arena[ta + 1]};
#if KIND == 1
            const uint32_t* e = (const uint32_t*)(pool + EX_AT + r * ES + s * C + 16u * m + 8u * hl);
#endif
            hf tv[8];
            e4x4(t[0], tv); e4x4(t[1], tv + 4);
#if KIND == 1
            hf ev[8];
            e4x4(e[0], ev); e4x4(e[1], ev + 4);
#endif
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#if KIND == 1
                const hf x = ev[i] + tv[i];
#else
                const hf x = tv[i];
#endif
                const hf c = x + h16at(w, W_TABLE7 + 2u * (16u * m + 8u * hl + (uint32_t)i));
                R[qi][i] = (float)c;
            }
        }
    }
#endif

    STOP(1)
    // ---------------------------------------------------------------- attention, HS heads at a time
#pragma nounroll
    for (uint32_t h0 = 0; h0 < NH; h0 += HS) {
        sync();   // XT written; the previous heads' K / V and (first) the expansion's EX are done with
        // jobs (head slot, channel tile c, token tiles 3nh .. 3nh + 2): the weight tiles loaded once for three products
#ifdef RRVK_XKV   // profiling: no K / V projections
        for (uint32_t job = wv; job < 0u; job += NW) {
#else
        for (uint32_t job = wv; job < 4u * HS; job += NW) {
#endif
            const uint32_t hh = job >> 2, c = job & 1u, nh = (job >> 1) & 1u;
            const uint32_t hd = h0 + hh;
            if (hd >= NH) continue;
            const uint32_t wk = W_WK0 + HEAD * hd, wvb = W_WV0 + HEAD * hd;
            uint8_t* KS = pool + KV_AT + KV_SLOT * hh;
            uint8_t* VS = KS + 96 * KS_S;
            i2v ka[2 * KC], vb[2 * KC];
#pragma unroll
            for (uint32_t kt = 0; kt < 2 * KC; ++kt) { ka[kt] = tile(w, wk, c * (C / 16) + kt, col, hl); vb[kt] = tile(w, wvb, kt * 2 + c, col, hl); }
#pragma unroll
            for (uint32_t n = 3u * nh; n < 3u * nh + 3u; ++n) {
                f8v a = zero8(), v = zero8();
                const uint8_t* xt = pool + XT_AT + (16u * n + col) * XS + 8u * hl;
#pragma unroll
                for (uint32_t kt = 0; kt < 2 * KC; kt += 2) {
                    const i2v x0_ = op(xt + 16u * kt), x1_ = op(xt + 16u * kt + 16u);
                    k32(a, ka[kt], x0_, ka[kt + 1], x1_);
                    k32(v, x0_, vb[kt], x1_, vb[kt + 1]);
                }
                *(i2v*)(KS + (16u * n + col) * KS_S + 16u * c + 8u * hl) = q8(a);
                *(i2v*)(VS + (16u * c + col) * VS_S + 16u * n + 8u * hl) = q8(v);
            }
        }
        sync();
#ifdef RRVK_XATT   // profiling: no attention proper
        if (A.shift == 0xdeadbeefu) {
#else
        if (wv < 4u * HS && h0 + (wv >> 2) < NH) {   // one (head slot, query tile) per wave
#endif
            const uint32_t hh = wv >> 2, hd = h0 + hh;
            const uint8_t* KS = pool + KV_AT + KV_SLOT * hh;
            const uint8_t* VS = KS + 96 * KS_S;
            i2v Qb[2], Pb[6];
#pragma unroll
            for (uint32_t c = 0; c < 2; ++c) Qb[c] = op(KS + (16u * q + col) * KS_S + 16u * c + 8u * hl);
            {
                hf e[6][8];
                const float* bias = (const float*)(w + A.biasF + hd * (96u * 64u * 4u) + (16u * q + col) * 384u + 32u * hl);
#pragma unroll
                for (uint32_t kt = 0; kt < 6; ++kt) {
#ifdef RRVK_XBIAS   // profiling: no bias
                    f8v s = zero8();
#elif defined(RRVK_BFAKE)   // profiling: the bias from one place (loaded and converted, no traffic)
                    f8v s = bias8(w, W_BIAS0 + 32u * kt);
#else
                    f8v s = *(const f8v*)(bias + 16u * kt);
#endif
                    k32(s, op(KS + (16u * kt + col) * KS_S + 8u * hl), Qb[0], op(KS + (16u * kt + col) * KS_S + 16u + 8u * hl), Qb[1]);
#pragma unroll
                    for (int i = 0; i < 8; i += 2) {
#ifdef RRVK_XCUBIC   // profiling: no cubic
                        const h2 c = h2{(hf)s[i], (hf)s[i + 1]};
#else
                        const h2 c = cubic(pack16(s[i], s[i + 1]));
#endif
                        e[kt][i] = c.x; e[kt][i + 1] = c.y;
                    }
                }
                h2 ep[6][4];
#pragma unroll
                for (uint32_t kt = 0; kt < 6; ++kt)
#pragma unroll
                    for (uint32_t qq = 0; qq < 4; ++qq) ep[kt][qq] = h2{e[kt][2 * qq], e[kt][2 * qq + 1]};
                h2 st[2];
#pragma unroll
                for (uint32_t tq = 0; tq < 2; ++tq) {
                    const uint32_t q0 = 2u * tq, q1 = q0 + 1u;
                    const h2 ev = ((ep[0][q0] + ep[0][q1]) + (ep[2][q0] + ep[2][q1])) + (ep[4][q0] + ep[4][q1]);
                    const h2 od = ((ep[1][q0] + ep[1][q1]) + (ep[3][q0] + ep[3][q1])) + (ep[5][q0] + ep[5][q1]);
                    st[tq] = ev + od;
                }
                const h2 loc = st[0] + st[1];
                const uint32_t other = other_half(bh(loc.x) | (bh(loc.y) << 16));
                const h2 o = loc + h2{hb(other & 0xffffu), hb(other >> 16)};
                const hf rs = o.x + o.y;
                const hf r = (hf)__builtin_amdgcn_rcpf((float)rs);
#pragma unroll
                for (uint32_t kt = 0; kt < 6; ++kt) {
                    float p[8];
#pragma unroll
#ifdef RRVK_XPROB   // profiling: the probabilities from f32 products, no binary16 rounding
                    for (int qq = 0; qq < 4; ++qq) { p[2 * qq] = (float)ep[kt][qq].x * (float)r; p[2 * qq + 1] = (float)ep[kt][qq].y * (float)r; }
#else
                    for (int qq = 0; qq < 4; ++qq) { const h2 pv = ep[kt][qq] * h2{r, r}; p[2 * qq] = (float)pv.x; p[2 * qq + 1] = hi32(pv); }
#endif
                    Pb[kt] = (i2v){(int)pk4(p[0], p[1], p[2], p[3]), (int)pk4(p[4], p[5], p[6], p[7])};
                }
            }
#pragma unroll
            for (uint32_t c = 0; c < 2; ++c) {
                f8v o = zero8();
#pragma unroll
                for (uint32_t ks = 0; ks < 3; ++ks) {
                    const uint8_t* vp = VS + (16u * c + col) * VS_S + 32u * ks + 8u * hl;
                    k32(o, op(vp), Pb[2 * ks], op(vp + 16), Pb[2 * ks + 1]);
                }
                *(i2v*)(pool + OB_AT + ((hh * 4u + q) * 2u + c) * 256u + lane * 8u) = q8(o);
            }
        }
        sync();
        // the group's out-projections onto the residual, head after head
#pragma nounroll
        for (uint32_t hh = 0; hh < HS && h0 + hh < NH && unit; ++hh) {
            const uint32_t wo = W_WO0 + HEAD * (h0 + hh);
            const i2v a0 = tile(w, wo, 2 * um, col, hl), a1 = tile(w, wo, 2 * um + 1, col, hl);
#pragma unroll
            for (uint32_t qi = 0; qi < QB; ++qi) {
                const uint32_t qq = QB * ug + qi;
                const i2v o0 = op(pool + OB_AT + ((hh * 4u + qq) * 2u) * 256u + lane * 8u), o1 = op(pool + OB_AT + ((hh * 4u + qq) * 2u + 1u) * 256u + lane * 8u);
                k32(R[qi], a0, o0, a1, o1);
            }
        }
    }
    STOP(2)
    sync();   // OB read
    STOP(3)

    // ---------------------------------------------------------------- MLP
    if (unit)
#pragma unroll
        for (uint32_t qi = 0; qi < QB; ++qi) {
            hf* rv = (hf*)(pool + RV_AT + (((QB * ug + qi) * MT + um) * 32u + lane) * 16u);
#pragma unroll
            for (int i = 0; i < 8; ++i) rv[i] = (hf)R[qi][i];
        }
    sync();
#pragma unroll
    for (uint32_t qi = 0; qi < QB; ++qi) {
        if (!unit) break;
        const uint32_t q = QB * ug + qi;
        // swin_m.comp's rowsum_sq tree over all channel tiles of the token, read back from RV
        hf h1[2];
#pragma unroll
        for (uint32_t c = 0; c < 2; ++c) {
            hf ht[2];
#pragma unroll
            for (uint32_t tq = 0; tq < 2; ++tq) {
                const uint32_t i0 = 4u * tq + c, i1 = i0 + 2u;
                hf par[2];
#pragma unroll
                for (uint32_t pa = 0; pa < 2; ++pa) {
                    hf pr[KC];
#pragma unroll
                    for (uint32_t b = 0; b < KC; ++b) {
                        const hf* rv = (const hf*)(pool + RV_AT + ((q * MT + 2u * b + pa) * 32u + lane) * 16u);
                        const hf s0 = rv[i0] * rv[i0], s1 = rv[i1] * rv[i1];
                        pr[b] = s0 + s1;
                    }
                    par[pa] = tree(pr);
                }
                ht[tq] = par[0] + par[1];
            }
            h1[c] = ht[0] + ht[1];
        }
        const uint32_t other = other_half(bh(h1[0]) | (bh(h1[1]) << 16));
        const hf g0 = h1[0] + hb(other & 0xffffu), g1 = h1[1] + hb(other >> 16);
        const hf s = sumsq(g0 + g1);
        const hf se = s + (hf)0.0001220703125f;
        const hf r = (hf)__builtin_amdgcn_rsqf((float)se);
        {
            const uint32_t m = um;
            const hf* rv = (const hf*)(pool + RV_AT + ((q * MT + m) * 32u + lane) * 16u);
            float n[8];
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const uint32_t k = 16u * m + 8u * hl + (uint32_t)i;
                const h2 x = h2{rv[i], rv[i + 1]};
                const h2 g = h2{r, r} * h2at(w, W_GAIN14 + 2u * k);
                const h2 nn = x * g;
                n[i] = (float)nn.x; n[i + 1] = (float)nn.y;
                const h2 a = x + h2at(w, W_TABLE16 + 2u * k);
                R[qi][i] = (float)a.x; R[qi][i + 1] = (float)a.y;
            }
            *(i2v*)(pool + NBUF_AT + ((q * MT + m) * 32u + lane) * 8u) = (i2v){(int)pk4(n[0], n[1], n[2], n[3]), (int)pk4(n[4], n[5], n[6], n[7])};
        }
    }
    sync();   // NBUF written, RV read (HB reuses it)
    // the hidden layer: (branch, channel tile c) jobs over the four query tiles, the weight tiles loaded once
    for (uint32_t job = wv; job < 2u * NB; job += NW) {
        const uint32_t br = job >> 1, c = job & 1u;
        const uint32_t w1 = W_W1_0 + BRANCH * br, b1 = W_B1_0 + BRANCH * br;
        i2v wa[MT];
#pragma unroll
        for (uint32_t kt = 0; kt < MT; ++kt) wa[kt] = tile(w, w1, c * (C / 16) + kt, col, hl);
        const f8v bias = bias8(w, b1 + 2u * (16u * c + 8u * hl));
#pragma unroll
        for (uint32_t jq = 0; jq < 4; ++jq) {
            const uint8_t* nb = pool + NBUF_AT + (jq * MT * 32u + lane) * 8u;
            f8v h = bias;
#pragma unroll
            for (uint32_t kt = 0; kt < MT; kt += 2) k32(h, wa[kt], op(nb + kt * 256u), wa[kt + 1], op(nb + (kt + 1) * 256u));
            // RR_NATIVE's activation: in f32, straight into FP8
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float x = h[i], y = __builtin_fminf(__builtin_fmaxf(x, -2.0f), 2.0f);
                const float g = __builtin_fmaf(y, __builtin_fmaf(-__builtin_fabsf(y), 0.0810546875f, 0.412109375f), 0.5f);
                a[i] = x * g;
            }
            *(i2v*)(pool + HB_AT + (((br * 4u + jq) * 2u + c) * 32u + lane) * 8u) = (i2v){(int)pk4(a[0], a[1], a[2], a[3]), (int)pk4(a[4], a[5], a[6], a[7])};
        }
    }
    sync();
    STOP(4)
#pragma nounroll
    for (uint32_t br = 0; br < NB && unit; ++br) {
        const uint32_t w2 = W_W2_0 + BRANCH * br;
        const i2v a0 = tile(w, w2, 2 * um, col, hl), a1 = tile(w, w2, 2 * um + 1, col, hl);
#pragma unroll
        for (uint32_t qi = 0; qi < QB; ++qi) {
            const uint32_t qq = QB * ug + qi;
            const i2v hb0 = op(pool + HB_AT + (((br * 4u + qq) * 2u) * 32u + lane) * 8u), hb1 = op(pool + HB_AT + (((br * 4u + qq) * 2u + 1u) * 32u + lane) * 8u);
            k32(R[qi], a0, hb0, a1, hb1);
        }
    }
    sync();   // HB read (the outputs reuse the pool)
    STOP(5)

    // ---------------------------------------------------------------- outputs
    hf* const OUTF = (hf*)(pool + TAIL_AT);                              // raster order, OUTF_C per token
    hf* const TAP = (hf*)(pool + TAIL_AT + 64 * OUTF_C * 2);             // texture 88 at the 10 x 10 tap positions, decoders
    hf* const O112 = (hf*)(pool + TAIL_AT + 64 * OUTF_C * 2 + 100 * 4 * 2);   // surface 112 per token, encoders
#pragma unroll
    for (uint32_t qi = 0; qi < QB; ++qi) {
        if (!unit) break;
        const uint32_t q = QB * ug + qi, m = um;
        const uint32_t t = inter(16u * q + col);
        const int tx = x0 + (int)(t & 7u), ty = y0 + (int)(t >> 3);
        const bool inside = tx >= 0 && ty >= 0 && tx < ex && ty < ey;
        const uint32_t o = (outOff + (uint32_t)(ty * ex + tx) * 32u) >> 2;
#if KIND == 0 && CX
        const uint32_t ty2 = t >> 3, tx2 = t & 7u;
        const uint32_t mr = (ty2 >> 1) < 2u ? 2u * (tx2 >> 1) + (ty2 >> 1) : 8u + 2u * (tx2 >> 1) + (ty2 >> 1) - 2u;
        const uint32_t blk = 2u * (ty2 & 1u) + (tx2 & 1u);
#endif
#if DEC0
#pragma unroll
        for (int i = 0; i < 8; ++i) OUTF[t * 32u + inter(16u * m + 8u * hl + (uint32_t)i)] = (hf)R[qi][i];
        (void)inside; (void)o;
#else
        {
            const i2v tb = q8(R[qi]);
            const uint32_t at = 4u * (m & 1u) + 2u * hl;
            if (inside) { arena[o + (m >> 1) * (plane >> 2) + at] = (uint32_t)tb[0]; arena[o + (m >> 1) * (plane >> 2) + at + 1u] = (uint32_t)tb[1]; }
#if KIND == 0 && CX
            *(i2v*)(pool + MGR_AT + mr * ES + blk * C + 16u * m + 8u * hl) = tb;
#endif
            *(i2v*)(pool + TB_AT + ((q * MT + m) * 32u + lane) * 8u) = tb;
        }
#endif
    }
    sync();
    STOP(6)
#if !DEC0
    if (wv < 4u) {   // the C -> 16 head, wave q (= wv) for query tile q
        const uint32_t t = inter(16u * q + col);
        const int tx = x0 + (int)(t & 7u), ty = y0 + (int)(t >> 3);
        const bool inside = tx >= 0 && ty >= 0 && tx < ex && ty < ey;
        f8v hv = bias8(w, W_BH + 16u * hl);
        const uint8_t* tbp = pool + TB_AT + (q * MT * 32u + lane) * 8u;
#pragma unroll
        for (uint32_t kt = 0; kt < MT; kt += 2)
            k32(hv, tile(w, W_WH, kt, col, hl), op(tbp + kt * 256u), tile(w, W_WH, kt + 1, col, hl), op(tbp + (kt + 1) * 256u));
#if KIND == 0
        {   // the kernel-prediction filter (lanes 0-15; 16-31 hold head channels 8-15)
            hf l[12];
#pragma unroll
            for (int i = 0; i < 8; ++i) l[i] = (hf)hv[i];
            const uint32_t u0 = other_half(bh(l[0]) | (bh(l[1]) << 16)), u1 = other_half(bh(l[2]) | (bh(l[3]) << 16));
            l[8] = hb(u0 & 0xffffu); l[9] = hb(u0 >> 16); l[10] = hb(u1 & 0xffffu); l[11] = hb(u1 >> 16);
#define RRVK_MAX(a, b) __builtin_fmaxf16((a), (b))
            const hf ma = RRVK_MAX(RRVK_MAX(RRVK_MAX(l[0], l[2]), l[4]), l[6]), mb = RRVK_MAX(RRVK_MAX(RRVK_MAX(l[1], l[3]), l[5]), l[7]);
            const hf mx = RRVK_MAX(RRVK_MAX(ma, mb), l[8]);
            float e[9];
#pragma unroll
            for (int i = 0; i < 9; ++i) { const hf d = l[i] - mx; e[i] = __builtin_amdgcn_exp2f((float)d * 1.4426950216293335f); }
            float s = (e[0] + e[1]) + 0.0f;
            s = s + (e[2] + e[3]);
            s = s + (e[4] + e[5]);
            s = s + (e[6] + e[7]);
            s = s + e[8];
            const hf rr = rcp16(s);
#if ENC0
            // over the gathered noise grid: two binary16 fma chains (taps even / odd), the token's taps at
            // (t & 7) .. + 2, (t >> 3) .. + 2
            hf wg[9];
#pragma unroll
            for (int i = 0; i < 9; ++i) wg[i] = (hf)(e[i] * (float)rr);
            const uint32_t gx = t & 7u, gy = t >> 3;
            const uint8_t* nz = G + 80u * gy + 8u * gx;
            hf o112[3], o120[3];
#pragma unroll
            for (uint32_t c = 0; c < 3; ++c) {
                hf lo = (hf)0.0f, hi = (hf)0.0f;
#pragma unroll
                for (uint32_t k = 0; k < 8; k += 2) {
                    lo = (hf)__builtin_fmaf((float)h16at(nz, 80u * (k / 3u) + 8u * (k % 3u) + 2u * c), (float)wg[k], (float)lo);
                    hi = (hf)__builtin_fmaf((float)h16at(nz, 80u * ((k + 1u) / 3u) + 8u * ((k + 1u) % 3u) + 2u * c), (float)wg[k + 1], (float)hi);
                }
                const hf sum = lo + hi;
                const hf f = (hf)__builtin_fmaf((float)h16at(nz, 160u + 16u + 2u * c), (float)wg[8], (float)sum);
                const hf a = sig(l[9 + c]) * f;
                o120[c] = a; o112[c] = f - a;
            }
            if (hl == 0u && inside) {
                image_store((v4f){(float)o112[0], (float)o112[1], (float)o112[2], 0.0f}, 15, tx, ty, A.s112, 0, 0);
                image_store((v4f){(float)o120[0], (float)o120[1], (float)o120[2], 0.0f}, 15, tx, ty, A.s120, 0, 0);
            }
            if (hl == 0u)
#pragma unroll
                for (int c = 0; c < 3; ++c) O112[t * 4u + c] = o112[c];
#else
            float wg[9];
#pragma unroll
            for (int i = 0; i < 9; ++i) wg[i] = e[i] * (float)rr;
            const int x = tx, y = ty;
            const int xm = x - 1 < 0 ? 1 : x - 1, xc = x < 0 ? 1 : (x >= ex ? ex - 2 : x), xp = x + 1 >= ex ? ex - 2 : x + 1;
            const int ym = y - 1 < 0 ? 1 : y - 1, yc = y < 0 ? 1 : (y >= ey ? ey - 2 : y), yp = y + 1 >= ey ? ey - 2 : y + 1;
            const int px[9] = {xm, xc, xp, xm, xc, xp, xm, xc, xp}, py[9] = {ym, ym, ym, yc, yc, yc, yp, yp, yp};
            float acc[3][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const v4f tv = image_load(15, px[k], py[k], A.t88, 0, 0);
#pragma unroll
                for (int c = 0; c < 3; ++c) acc[c][k & 1] = tv[c] * wg[k] + acc[c][k & 1];
            }
            const v4f t8 = image_load(15, px[8], py[8], A.t88, 0, 0);
            float o112[3];
#if CX
            float o120[3];
#endif
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                const float f0 = acc[c][0] + acc[c][1];
                const float f = t8[c] * wg[8] + f0;
#if CX
                const float a = (float)sig(l[9 + c]) * f;
                o120[c] = a; o112[c] = f - a;
#else
                o112[c] = f;   // the bottleneck: the filtered value itself
#endif
            }
            if (hl == 0u && inside) {
                image_store((v4f){o112[0], o112[1], o112[2], 0.0f}, 15, x, y, A.s112, 0, 0);
#if CX
                image_store((v4f){o120[0], o120[1], o120[2], 0.0f}, 15, x, y, A.s120, 0, 0);
#endif
            }
#if CX
            if (hl == 0u)
#pragma unroll
                for (int c = 0; c < 3; ++c) O112[t * 4u + c] = (hf)o112[c];
#endif
#endif
        }
#else
#pragma unroll
        for (int i = 0; i < 8; ++i) OUTF[t * 16u + 8u * hl + (uint32_t)i] = (hf)hv[i];
#endif
    }
#endif
    STOP(7)
#if KIND == 0 && CX
    sync();
    {   // the 2x2 merge: merged^T (CX x 16 rows), channel tiles m = wv, wv + NW, ...
        const uint32_t mxx = col < 8u ? col >> 1 : (col - 8u) >> 1, mxy = col < 8u ? col & 1u : 2u + (col & 1u);
        const int X = x0 / 2 + (int)mxx, Y = y0 / 2 + (int)mxy;
        const uint32_t lplane = (uint32_t)(hx * hy) * 32u;
        const uint32_t o = (A.mrgOff + (uint32_t)(Y * hx + X) * 32u) >> 2;
        for (uint32_t m = wv; m < CX / 16u; m += NW) {
            f8v a = bias8(w, W_BM + 2u * (16u * m + 8u * hl));
            for (uint32_t ks = 0; ks < C / 8u; ++ks) {
                const uint8_t* g = pool + MGR_AT + col * ES + 32u * ks + 8u * hl;
                k32(a, tile(w, W_WM, m * (4u * C / 16u) + 2u * ks, col, hl), op(g), tile(w, W_WM, m * (4u * C / 16u) + 2u * ks + 1u, col, hl), op(g + 16));
            }
            const i2v qv = q8(a);
            if (X >= 0 && Y >= 0 && X < hx && Y < hy) {
                const uint32_t at = o + (m >> 1) * (lplane >> 2) + 4u * (m & 1u) + 2u * hl;
                arena[at] = (uint32_t)qv[0];
                arena[at + 1u] = (uint32_t)qv[1];
            }
        }
        if (wv == NW - 1u && lane < 16u) {   // surface 128: 2x2 sums of 0.25 * surface 112, rightwards first, then downwards
            const uint32_t mx_ = lane & 3u, my_ = lane >> 2;
            const int x = x0 + 2 * (int)mx_, y = y0 + 2 * (int)my_;
            if (x >= 0 && y >= 0 && x < ex && y < ey) {
                const uint32_t t00 = (2u * my_) * 8u + 2u * mx_;
                float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (uint32_t c = 0; c < 3; ++c) {
                    const hf a00 = O112[t00 * 4u + c] * (hf)0.25f, a01 = O112[(t00 + 1u) * 4u + c] * (hf)0.25f;
                    const hf a10 = O112[(t00 + 8u) * 4u + c] * (hf)0.25f, a11 = O112[(t00 + 9u) * 4u + c] * (hf)0.25f;
                    const hf b0 = a00 + a01, b1 = a10 + a11;
                    out[c] = (float)(hf)(b0 + b1);
                }
                image_store((v4f){out[0], out[1], out[2], out[3]}, 15, x >> 1, y >> 1, A.s128, 0, 0);
            }
        }
    }
#endif
#if KIND == 1
    sync();
    // texture 88 (one level down) at the window's 10 x 10 full-resolution tap positions, through the bilinear sampler
    // at NVIDIA's coordinates (engine/rr/rr_decfilter.glsl)
    if (tid < 100u) {
        const int xi = clampTap(x0 - 1 + (int)(tid % 10u), ex), yi = clampTap(y0 - 1 + (int)(tid / 10u), ey);
        const float iw = A.iw, ih = A.ih;
        const float cu = __builtin_fmaf(iw, (float)xi, iw * 0.5f), cv = __builtin_fmaf(ih, (float)yi, ih * 0.5f);
        const v4f bl = image_sample(15, cu, cv, A.t88, A.linear, false, 0, 0);
#pragma unroll
        for (uint32_t c = 0; c < 3; ++c) TAP[tid * 4u + c] = (hf)bl[c];
    }
    sync();
    if (tid < 128u) {   // two threads per token: half 0 filters channels 0 and 1, half 1 channel 2
        const uint32_t t = tid >> 1, hf_ = tid & 1u;
        const int x = x0 + (int)(t & 7u), y = y0 + (int)(t >> 3);
        const bool inside = !(x < 0 || y < 0 || x >= ex || y >= ey);
        hf a[10];
#pragma unroll
        for (uint32_t c = 0; c < 10; ++c) a[c] = OUTF[t * OUTF_C + c];
#if DEC0
        if (hf_ == 1u && inside) {   // the output stage's eight codes and surface 144
            hf b[12];
#pragma unroll
            for (uint32_t c = 0; c < 12; ++c) b[c] = OUTF[t * 32u + 12u + c];
            const uint32_t o = (A.outOff + (uint32_t)(y * ex + x) * 8u) / 4u;
#ifdef RR_PRESET_D
            // cuda_dldn_engine_swin_dec0_diamond_wallaby_kernel: channels 12, 13, 0, 18, 19, 20, 21, 0
            arena[o] = pk4((float)b[0], (float)b[1], 0.0f, (float)b[6]) & 0xff00ffffu;
            arena[o + 1u] = pk4((float)b[7], (float)b[8], (float)b[9], 0.0f) & 0x00ffffffu;
#else
            arena[o] = pk4((float)b[0], (float)b[1], (float)b[6], (float)b[7]);
            arena[o + 1u] = pk4((float)b[8], (float)b[9], (float)b[10], (float)b[11]);
#endif
            image_store((v4f){tanhNV((float)b[2]), tanhNV((float)b[3]), tanhNV((float)b[4]), tanhNV((float)b[5])}, 15, x, y, A.s120, 0, 0);
        }
#endif
#define RRVK_MAX(a, b) __builtin_fmaxf16((a), (b))
        const hf ma = RRVK_MAX(RRVK_MAX(RRVK_MAX(a[0], a[2]), a[4]), a[6]), mb = RRVK_MAX(RRVK_MAX(RRVK_MAX(a[1], a[3]), a[5]), a[7]);
        const hf mx = RRVK_MAX(RRVK_MAX(ma, mb), a[8]);
        float e[9];
#pragma unroll
        for (uint32_t i = 0; i < 9; ++i) { const hf d = a[i] - mx; e[i] = __builtin_amdgcn_exp2f((float)d * 1.4426950216293335f); }
        float s = (e[0] + e[1]) + 0.0f;
        s = s + (e[2] + e[3]);
        s = s + (e[4] + e[5]);
        s = s + (e[6] + e[7]);
        s = s + e[8];
        const hf rr = rcp16(s);
        hf wg[9];
#pragma unroll
        for (uint32_t i = 0; i < 9; ++i) wg[i] = (hf)(e[i] * (float)rr);
        const uint32_t tx = (uint32_t)(x - x0), ty = (uint32_t)(y - y0), c0 = 2u * hf_;
        hf f[2] = {(hf)0.0f, (hf)0.0f};
#pragma unroll
        for (uint32_t k = 0; k < 9; ++k) {
            const uint32_t at = ((ty + k / 3u) * 10u + tx + k % 3u) * 4u + c0;
#pragma unroll
            for (uint32_t c = 0; c < 2; ++c) f[c] = (hf)__builtin_fmaf((float)TAP[at + c], (float)wg[k], (float)f[c]);
        }
        const hf g = sig(a[9]);
        const hf om = (hf)1.0f - g;
        v4f t96 = {0, 0, 0, 0}, t104 = {0, 0, 0, 0};
        if (inside) { t96 = image_load(15, x, y, A.t96, 0, 0); t104 = image_load(15, x, y, A.t104, 0, 0); }
        hf o[2];
#pragma unroll
        for (uint32_t c = 0; c < 2; ++c) {
            const hf fg = f[c] * g;
            hf v = (hf)__builtin_fmaf((float)(hf)t96[c0 + c], (float)om, (float)fg);
            const hf t4 = (hf)t104[c0 + c];
            if (!__builtin_isnan(t4)) v = v + t4;
            o[c] = v;
        }
        // the pair's lanes are neighbours: channel 2 comes from the odd lane
        const uint32_t other = (uint32_t)__builtin_amdgcn_ds_swizzle((int)bh(o[0]), 0x041f);
        if (hf_ == 0u && inside) image_store((v4f){(float)o[0], (float)o[1], (float)hb(other), 0.0f}, 15, x, y, A.s112, 0, 0);
    }
#endif
}
