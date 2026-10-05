// Native replacement for the Ray Reconstruction denoiser's third Swin decoder,
// cuda_dldn_engine_swin_dec2_kernel (nvngx_dlssd.dll 310.7).
//
// ---------------------------------------------------------------------------
// STATE: the mirror-cut frame is in place and builds/serves.  Phase 1 (the
// patch-expand GEMM) is implemented; phases 2-38 and their 37 epilogues are not,
// so the four accumulators handed back to the epilogue are not yet phase 38's
// and the layer's output is wrong.
//
// The previous revision claimed phase 1 was "bit-exact against the debug
// oracle".  That claim was void: its stage spliced the payload store next to
// phase 1's mma (line 1534), and E37's plane-arena scatter erases it before the
// dump -- see the measurement in swin_dec2_debug_stages.py.  Every stage now
// splices after E37 and phase 1 is re-verified through that channel.
//
// The kernel is one warp per 8x8 token patch: grid (41, 24, 1), block (32, 1, 4)
// -- four waves per block, which is why the decoders' weight addressing carries a
// per-tid.z stride -- and a 144-byte parameter block.  The PTX this file
// replaces is kernels/extracted/rr/0021-PREPASS_ENTRYPOINT_NAME.ptx, lines
// 1535-36892 (statements s452-s15510, 726 mma in 38 phases).
//
// kernels/rr/swin_dec2_recipe.py's recipe (make_ptx.py's per-layer hook) keeps
// NVIDIA's prologue (lines 1-1534) and its final epilogue (line 36893 to the
// end) and replaces the 726 mma and everything between them with the call
// below.
//
// Linked into ZLUDA's compile of that PTX (D4R_ZLUDA_EXTRA_BC), so
// tex_common.h's helpers and ZLUDA's own device functions are available.
// Compiled with -nogpuinc, so only plain clang builtins and attributes are used.
#include "../tex/tex_common.h"

// ---------------------------------------------------------------------------
// Parameter block. Fields are addressed by their byte offset in the PTX, so the
// raw image is kept and read through a typed accessor.
struct DldnSwinDec2Params
{
    uint8_t raw[144];
};
static_assert(sizeof(DldnSwinDec2Params) == 144, "parameter block");

enum : int
{
    kOutPlane = 48, // plane arena the epilogue reads back (arg048), also the debug arena
    kWeights = 40,  // NVIDIA's prepared weight image (arg040), read-only
};

#define DEVF __attribute__((device)) static inline
typedef float lds_f32 __attribute__((address_space(3)));
typedef uint32_t lds_u32 __attribute__((address_space(3)));
typedef uint8_t lds_u8 __attribute__((address_space(3)));

// ---------------------------------------------------------------------------
// Phase 1's weight sub-image.  Kernels/rr/swin_dec2_recipe.py's docstring has
// the derivation: the 24 `ld.weak.global.ca.v4.u32` of the prologue read
// W + 12288*tid.z + 16*laneid + 512*t for t = 0..23, and the 144 mma use them as
// 3 m-tiles x 12 n-tiles x 4 k-steps, so M = 48, N = 96, K = 128.
//
// A 512-byte tile is 16 columns x 32 k, and the chain of four mma that
// accumulates one output tile walks tiles 4*ng+0..3, i.e. one n-group (16
// columns) across all four k-slices.  Byte j of a lane's 16-byte slot holds
//   (n, k) = (16*ng + 8*(frag&1) + g, 32*ks + 2t + (j&1) + 16*((j>>1)&1)
// with the second 8-byte half of the slot at k+8 (enc0's settled layout, the
// same one rr_layer_spec.py verifies on this family), so
//   waddr1(n, k) = 12288*tid.z + 512*(4*(n>>4) + (k>>5)) + 64*(n&7)
//                  + 8*((n>>3)&1) + 16*((k>>1)&3) + 4*((k>>3)&1)
//                  + 2*((k>>4)&1) + (k&1)
// which is enc0's waddr() with the tile index widened from (n>>4) to
// 4*(n>>4) + (k>>5) and the per-window base added.  (VERIFIED on all 96*128 of a
// phase-1 tile's (n, k) pairs against the oracle's own B fragment registers,
// debug stage p1.)
DEVF uint32_t waddr1(int tidz, int n, int k)
{
    return (uint32_t)(12288 * tidz) + 512u * (uint32_t)(4 * (n >> 4) + (k >> 5))
         + 64u * (uint32_t)(n & 7) + 8u * (uint32_t)((n >> 3) & 1)
         + 16u * (uint32_t)((k >> 1) & 3) + 4u * (uint32_t)((k >> 3) & 1)
         + 2u * (uint32_t)((k >> 4) & 1) + (uint32_t)(k & 1);
}

// e4m3 code -> f32, exact. The mma's e4m3 operands reach the FMA path through
// ZLUDA's exact e4m3 -> f16 widening, so this decode is the same value.
DEVF float e4m3_to_f32(uint32_t code)
{
    const uint32_t mag = code & 0x7fu;
    const uint32_t bits = mag >= 8u
        ? ((((mag >> 3) + 120u) << 23) | ((mag & 7u) << 20))
        : __builtin_bit_cast(uint32_t, (float)mag * 0x1p-9f);
    return __builtin_bit_cast(float, bits | ((code & 0x80u) << 24));
}

// warp butterfly over the four lanes that hold a row's 32 k values (laneid ^ 1
// and laneid ^ 2): the two shfl.sync.bfly steps the PTX's reductions perform.
DEVF uint32_t bfly1_bits(uint32_t v)
{
    return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0xB1, 0xf, 0xf, false);
}

DEVF uint32_t bfly2_bits(uint32_t v)
{
    return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0x4E, 0xf, 0xf, false);
}

// f16 bits -> f32
DEVF float h2f(uint32_t bits)
{
    return (float)__builtin_bit_cast(half_t, (uint16_t)bits);
}

DEVF uint32_t pair_of(half_t a, half_t b)
{
    return (uint32_t)__builtin_bit_cast(uint16_t, a)
         | ((uint32_t)__builtin_bit_cast(uint16_t, b) << 16);
}

// ---------------------------------------------------------------------------
// Phase 1: M = 48 (three m-tiles of 16 rows), N = 96 (twelve n-tiles of eight
// columns), K = 128 (four k-slices of 32).
//
// A lane's twelve A fragments are the twelve m16n8k32 .f16.e4m3 operand quads
// in the order the 144 mma first use them: m-tile 0's four k-slices
// (as1[0..15]), m-tile 1's (as1[16..31]), m-tile 2's (as1[32..47]).  Within a
// fragment the hardware order is (enc0's, verified on this family):
//   reg 0 -> row g,  k = 2t + (j&1) + 16*((j>>1)&1)
//   reg 1 -> row g+8, same k
//   reg 2 -> row g,  k = 8 + 2t + (j&1) + 16*((j>>1)&1)
//   reg 3 -> row g+8, same k
// with g = laneid>>2, t = laneid&3 and j the byte index 0..3, so the four lanes
// that differ in t hold four disjoint eighths of the slice's 32 k and the two
// butterfly steps bring the other three in.
//
// The B operand is read from the weight image for the lane's own two columns of
// each n-tile.  The four mma of one output tile are a chain: each rounds its f32
// accumulator to f16 before the next adds to it, which is why they are four
// mma and not one -- the first takes the per-n-tile bias seed from the weight
// image at 49152 + 192*tid.z + 16*ntile + 4*t, the rest take the previous D.
//
// Every e4m3 product is exact and a 32-term k slice sums exactly in f32, so
// rounding once per slice to f16 reproduces the translated mma bit for bit.
//
// d is [n-tile][row half], each entry the packed f16 pair for columns 2t, 2t+1;
// hist, when non-null, receives the 144 chain results in mma order, i.e.
//   index 48*m-tile + 8*(n-tile>>1) + (n-tile&1) + 2*k-slice
// which is exactly where the PTX's k-th mma of that chain writes its D.
DEVF void gemm_p1(const uint32_t* as1, const uint8_t* w, const uint32_t* seeds,
                  int tidz, int t, uint32_t d[12][2], uint32_t* hist)
{
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 12; ++nt) {
            // the lane's two columns of this n-tile: 16 columns per weight tile,
            // the n-tile's parity selecting its 8-column half
            const int ncol = 16 * (nt >> 1) + 8 * (nt & 1) + 2 * t;
            const uint32_t bias = seeds[nt];
            half_t prev[2][2];
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const uint32_t* as = as1 + 16 * mt + 4 * ks;
                float bv[2][32];
#pragma unroll
                for (int c = 0; c < 2; ++c) {
#pragma unroll
                    for (int k = 0; k < 32; ++k) {
                        bv[c][k] = e4m3_to_f32(w[waddr1(tidz, ncol + c, 32 * ks + k)]);
                    }
                }
                float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll
                for (int reg = 0; reg < 2; ++reg) {
#pragma unroll
                    for (int row = 0; row < 2; ++row) {
                        // the register's value on this lane and on lanes t^1, t^2, t^3
                        const uint32_t p0 = as[2 * reg + row];
                        const uint32_t p[4] = {p0, bfly1_bits(p0), bfly2_bits(p0),
                                               bfly1_bits(bfly2_bits(p0))};
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            const int kk = 2 * (t ^ m);
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const int k = 8 * reg + kk + (j & 1) + 16 * ((j >> 1) & 1);
                                const float av = e4m3_to_f32((p[m] >> (8 * j)) & 0xffu);
                                acc[row][0] += av * bv[0][k];
                                acc[row][1] += av * bv[1][k];
                            }
                        }
                    }
                }
#pragma unroll
                for (int row = 0; row < 2; ++row) {
                    const float c0 = ks == 0 ? acc[row][0] + h2f(bias)
                                             : acc[row][0] + (float)prev[row][0];
                    const float c1 = ks == 0 ? acc[row][1] + h2f(bias >> 16)
                                             : acc[row][1] + (float)prev[row][1];
                    prev[row][0] = (half_t)c0;
                    prev[row][1] = (half_t)c1;
                }
                if (hist != nullptr) {
                    const int i = 48 * mt + 8 * (nt >> 1) + (nt & 1) + 2 * ks;
                    hist[2 * i] = pair_of(prev[0][0], prev[0][1]);
                    hist[2 * i + 1] = pair_of(prev[1][0], prev[1][1]);
                }
            }
            d[nt][0] = pair_of(prev[0][0], prev[0][1]);
            d[nt][1] = pair_of(prev[1][0], prev[1][1]);
        }
    }
}

// ---------------------------------------------------------------------------
// Ground truth for a stage comes from kernels/rr/swin_dec2_debug_stages.py: with
// D4R_RRSWIN_DEBUG=p1 (a1, or none) make_ptx.py keeps NVIDIA's own body and
// splices in a store of that stage's registers at param_0+48 + 46,000,000 +
// block*slot + lane*n*4, which the kernel never otherwise writes; the launch
// dump then holds the translated values for a block/lane.  Built with
// -DD4R_RRSWIN_DEBUG_STAGE=<n> (kernels/build.sh rrswin, D4R_TEX_CFLAGS), this
// body stores the same registers of its own phase n in the same order, so the
// two dumps compare byte for byte.  debug_store() mirrors make_ptx.py's shared
// swin_debug_block() exactly, including its ctaid.y*161 block numbering.
#ifndef D4R_RRSWIN_DEBUG_STAGE
#define D4R_RRSWIN_DEBUG_STAGE 0
#endif
// The payload base.  make_ptx.py's swin_debug_block() reads the same value from
// D4R_RRSWIN_DEBUG_BASE, so both sides of a comparison store at the same place.
// Measured for dec2 (D4R_CUDA_REPLAY_DUMP capture): param_0+48 is byte 60,293,120
// of the 65,945,600-byte block the bridge dumps as arg048's allocation, so the
// arena is 5,652,480 bytes and the default 46,000,000 of enc0 is past its end
// (the store faults).  0 leaves room for every stage's 41*slot payload.
// The stage tables splice after E37 because E37 rewrites the whole arena: a
// pre-launch replay capture diffed against the post-launch reference shows
// 5,564,929 of 5,652,480 arena bytes changed, so a payload written next to a
// phase's mma is erased before the dump.  This body therefore stores at the very
// end of the entry, which is after its own E37.
#ifndef D4R_RRSWIN_DEBUG_BASE
#define D4R_RRSWIN_DEBUG_BASE 0
#endif

DEVF void debug_store(const uint8_t* prm, const uint32_t* regs, int n)
{
    const uint32_t block = __builtin_amdgcn_workgroup_id_x()
                         + 161u * __builtin_amdgcn_workgroup_id_y();
    if (block >= 64u) {
        return;   // the oracle's per-block slots only cover the first 64 blocks
    }
    const uint64_t arena = *(const uint64_t*)(prm + kOutPlane);
    // the oracle's slot, make_ptx.py's swin_debug_slot: 32 lanes x n x 4 bytes,
    // at least 8192, rounded up to a power of two (the oracle's splice forms
    // block*slot with a shift)
    const uint64_t want = (uint64_t)n * 128ull > 8192ull ? (uint64_t)n * 128ull : 8192ull;
    uint64_t slot = 1;
    while (slot < want) {
        slot <<= 1;
    }
    uint32_t* dst = (uint32_t*)(arena + (uint64_t)D4R_RRSWIN_DEBUG_BASE
                                 + (uint64_t)block * slot
                                 + (uint64_t)lane_id() * (uint64_t)n * 4ull);
    for (int i = 0; i < n; ++i) {
        dst[i] = regs[i];
    }
}

// ---------------------------------------------------------------------------
// Mirror cut.  Phase 1's twelve A fragments arrive as 48 b32 in the order the
// 144 mma first use them, then its twelve bias seeds (%r1597 %r1607 %r1677
// ... %r2007, read from the weight image at +49152..+49328 with the per-tid.z
// stride 192), the weight image, the parameter block and the shared scratch.
//
// The four accumulators the epilogue reads cross the cut through shared memory,
// not `call.uni`'s return parameters: ZLUDA lowers those to the function's LLVM
// return type, so there is no .param-space object and no pointer is formed -- the
// callee reads 0x0 in all four slots, and a 4 x i32 C++ struct return is an sret
// pointer, a different LLVM signature that will not link.  `scratch` is the base
// of the entry's .shared array (9600 bytes), which the epilogue re-stages before
// it reads, so the body is free to use it as scratch.
extern "C" __attribute__((device)) void d4r_swin_dec2(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6,
    uint32_t a7, uint32_t a8, uint32_t a9, uint32_t a10, uint32_t a11, uint32_t a12, uint32_t a13,
    uint32_t a14, uint32_t a15, uint32_t a16, uint32_t a17, uint32_t a18, uint32_t a19, uint32_t a20,
    uint32_t a21, uint32_t a22, uint32_t a23, uint32_t a24, uint32_t a25, uint32_t a26, uint32_t a27,
    uint32_t a28, uint32_t a29, uint32_t a30, uint32_t a31, uint32_t a32, uint32_t a33, uint32_t a34,
    uint32_t a35, uint32_t a36, uint32_t a37, uint32_t a38, uint32_t a39, uint32_t a40, uint32_t a41,
    uint32_t a42, uint32_t a43, uint32_t a44, uint32_t a45, uint32_t a46, uint32_t a47,
    uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5,
    uint32_t s6, uint32_t s7, uint32_t s8, uint32_t s9, uint32_t s10, uint32_t s11,
    const uint8_t* w, const uint8_t* prm, uint32_t scratch)
{
    const uint32_t as1[48] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                              a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23,
                              a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35,
                              a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47};
    const uint32_t seeds[12] = {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11};
    const int tidz = (int)__builtin_amdgcn_workitem_id_z();
    const int t = (int)(lane_id() & 3);   // the lane's two columns of each n-tile

    // ---- phase 1 -----------------------------------------------------------
    uint32_t d1[12][2];
    uint32_t hist1[288];
#if D4R_RRSWIN_DEBUG_STAGE == 2
    gemm_p1(as1, w, seeds, tidz, t, d1, hist1);
#else
    gemm_p1(as1, w, seeds, tidz, t, d1, nullptr);
#endif

    // ---- handing the accumulators back ------------------------------------
    // Phases 2-38 and their epilogues are not implemented yet.  Until they are,
    // fill the four slots with lane-distinct values so the read-back path is
    // observable without producing wild magnitudes.
    __attribute__((address_space(3))) uint32_t* out =
        ((__attribute__((address_space(3))) uint32_t*)(uintptr_t)scratch) + 4 * lane_id();
    for (int i = 0; i < 4; ++i) {
        out[i] = (uint32_t)lane_id() + 32u * (uint32_t)i;
    }

    // ---- debug store -------------------------------------------------------
    // Last, so it follows this body's own arena scatter: E37 rewrites the whole
    // plane arena, and an earlier store would be erased before the dump.  The
    // oracle splices its block at the same point in its entry (after E37).
#if D4R_RRSWIN_DEBUG_STAGE == 1
    {
        uint32_t regs[64];
        regs[0] = *(const uint32_t*)(w + 212736 + ((lane_id() << 2) & 12));
        for (int i = 0; i < 48; ++i) {
            regs[1 + i] = as1[i];
        }
        for (int i = 0; i < 12; ++i) {
            regs[49 + i] = seeds[i];
        }
        for (int i = 61; i < 64; ++i) {
            regs[i] = regs[0];          // the sentinel padding both sides store
        }
        debug_store(prm, regs, 64);
    }
#elif D4R_RRSWIN_DEBUG_STAGE == 2
    {
        // %r19329 (the leading sentinel make_ptx's window check needs) then the
        // 288 D registers of phase 1's 144 mma in statement order
        uint32_t regs[292];
        regs[0] = *(const uint32_t*)(w + 212736 + ((lane_id() << 2) & 12));
        for (int i = 0; i < 288; ++i) {
            regs[1 + i] = hist1[i];
        }
        for (int i = 289; i < 292; ++i) {
            regs[i] = regs[0];          // the sentinel padding both sides store
        }
        debug_store(prm, regs, 292);
    }
#endif
}