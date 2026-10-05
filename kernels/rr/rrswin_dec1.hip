// Native replacement for the Ray Reconstruction denoiser's third decoder,
// cuda_dldn_engine_swin_dec1_kernel (nvngx_dlssd.dll 310.7).
//
// ---------------------------------------------------------------------------
// STATE: phase 1 (the 48x384 patch-expand GEMM, 144 mma) is implemented and
// bit-exact against the GPU oracle -- debug stage p1, 4,194,304 payload bytes
// over 64 blocks x 32 lanes x 288 registers, 0 differing bytes, identical md5
// (ae431ae0baff).  Stage a1 (the 52 staged input registers) is likewise
// 100.0000%.  E1 and phases 2-26 are not implemented yet, so the eight
// accumulators the epilogue reads back are placeholders: the served frame is
// wrong (compare_launch_dump.py: 14 arguments compared, 4 differ -- arg[8] the
// feature plane E25 writes and arg[112] the output surface), and the layer
// costs 0.036 ms/frame instead of the translated 1-2 ms, because phase 1's
// accumulators are dead after the read-back.
//
// The next piece is E1 (s547-s7095): the 48 packed patch cells per lane into a
// 10x10x4 shared grid, the 3x3-ish gather from param_0+24, the per-column RMS
// norm (raw sum of squares + 2^-13, rsqrt.approx.ftz, gain at weight +25088) and
// the restage into phases 2-5's A operands.  Its lane -> (sub-array, row, col)
// map is 438 basic blocks in the PTX and is *not* characterised in
// rrswin_dec1_epilogues.md (its U4), so it has to be lifted out of the index
// arithmetic rather than guessed.
//
// The mirror-cut splice is kernels/rr/swin_dec1_recipe.py: it keeps NVIDIA's
// texture prologue (which stages phase 1's A fragments and bias seeds) and its
// surface epilogue (the shared staging, the softmax, the three texture reads
// and the surface store), and replaces physical lines 1486-38472 -- the whole
// 26-phase mma region, 632 mma -- with one call into d4r_swin_dec1.
//
// Linked into ZLUDA's compile of that PTX (D4R_ZLUDA_EXTRA_BC), so tex_common.h's
// helpers and ZLUDA's own device functions are available. Compiled with -nogpuinc,
// so only plain clang builtins and attributes are used.
//
// ---------------------------------------------------------------------------
// Launch: grid 81x47, block 32x1x2 (two waves per block, which is why the weight
// addressing carries a per-tid.z stride), parameter block 144 bytes. The PTX this
// file replaces is ~/.cache/d4r-rr-corpus/0022-PREPASS_ENTRYPOINT_NAME.ptx.
#include "../tex/tex_common.h"

// ---------------------------------------------------------------------------
// Parameter block. Fields are addressed by their byte offset in the PTX, so the
// raw image is kept and read through a typed accessor.
struct DldnSwinDec1Params
{
    uint8_t raw[144];
};
static_assert(sizeof(DldnSwinDec1Params) == 144, "parameter block");

enum : int
{
    kExtents = 0,      // ex, ey: two u32 (s14)
    kInPlane = 8,      // input feature plane arena, read by phase 1's A loads
    kPatchGather = 24, // the skip input E1 gathers the 10x10 patch from
    kWeights = 40,     // NVIDIA's prepared weight image (arg040), read-only
    kOutPlane = 48,    // plane arena E25 scatters phase 25's output into
};

#define DEVF __attribute__((device)) static inline
typedef uint32_t lds_u32 __attribute__((address_space(3)));
typedef uint8_t lds_u8 __attribute__((address_space(3)));

// ---------------------------------------------------------------------------
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

// byte j of a lane's 16-byte weight tile holds (k, n) at
//   addr(n, k) = base + 512*(n>>4) + 64*(n&7) + 16*((k>>1)&3) + 8*((n>>3)&1)
//              + 4*((k>>3)&1) + 2*((k>>4)&1) + (k&1)
// (the settled formula, VERIFIED on enc0's 1024 (n, k) pairs per tile: the
// 512-byte block is 32 lanes of 16 contiguous bytes, and 64*(n&7) + 16*((k>>1)&3)
// is exactly 16*laneid of the lane that owns the byte, so the remaining four terms
// are that byte's index inside the lane's slot).
DEVF uint32_t waddr(int base, int k, int n)
{
    return (uint32_t)base + 512u * (uint32_t)(n >> 4) + 64u * (uint32_t)(n & 7)
         + 16u * (uint32_t)((k >> 1) & 3) + 8u * (uint32_t)((n >> 3) & 1)
         + 4u * (uint32_t)((k >> 3) & 1) + 2u * (uint32_t)((k >> 4) & 1)
         + (uint32_t)(k & 1);
}

// warp butterfly over the four lanes that hold a row's 32 columns (laneid ^ 1 and
// laneid ^ 2): the two shfl.sync.bfly steps the PTX's reductions perform.
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
// One m16n8k32 e4m3 mma, f16 accumulate, done the way the hardware does it.
//
// The lane's four A registers hold, for each of its two rows (row g and row g+8),
// the eight e4m3 values of k = 2t + (j&1) + 16*((j>>1)&1) (byte j = 0..3) and of
// k + 8; the four lanes of the group hold the four disjoint eighths of the 32 k's,
// so two butterfly steps bring the other three in.  The B operand is read from the
// weight image at waddr() for the lane's own two D columns (D register 0 is row g,
// columns 2t and 2t+1; register 1 is row g+8).  Every e4m3 product is exact and 32
// of them sum exactly in f32, so a plain f32 accumulation rounded to f16 once is
// what the translated mma computes.
//
//   a[0..3]     the A fragment as the mma reads it
//   w           the weight image
//   base        byte offset of the B tile's first 512-byte block
//   n0          the mma's first output column (a multiple of 8)
//   c0, c1      the C operand's two registers (row g, row g+8)
//   t           laneid & 3
//   d[0], d[1]  the two D registers
DEVF void mma_tile(const uint32_t* a, const uint8_t* w, int base, int n0,
                   uint32_t c0, uint32_t c1, int t, uint32_t d[2])
{
    float bv[2][32];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const int n = n0 + 2 * t + c;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            bv[c][k] = e4m3_to_f32(w[waddr(base, k, n)]);
        }
    }
#pragma unroll
    for (int row = 0; row < 2; ++row) {
        float acc[2] = {0.0f, 0.0f};
#pragma unroll
        for (int reg = 0; reg < 2; ++reg) {
            const uint32_t q = a[2 * reg + row];
            uint32_t p[4];
            p[0] = q;
            p[1] = bfly1_bits(q);
            p[2] = bfly2_bits(q);
            p[3] = bfly1_bits(p[2]);
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int kk = 2 * (t ^ m);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int k = 8 * reg + kk + (j & 1) + 16 * ((j >> 1) & 1);
                    const float av = e4m3_to_f32((p[m] >> (8 * j)) & 0xffu);
                    acc[0] += av * bv[0][k];
                    acc[1] += av * bv[1][k];
                }
            }
        }
        const uint32_t cpair = row ? c1 : c0;
        d[row] = pair_of((half_t)(acc[0] + h2f(cpair)),
                         (half_t)(acc[1] + h2f(cpair >> 16)));
    }
}

// ---------------------------------------------------------------------------
// Ground truth for a stage comes from kernels/rr/swin_dec1_recipe.py's debug
// stages: with D4R_RRSWIN_DEBUG=a1 (or p1) the same splice stores that stage's
// registers into the param_0+8 feature-plane buffer at +46,000,000, which the
// entry has finished reading by then (the prologue's last read is s324, before
// the cut at line 1486), and the launch dump then holds the translated values
// for a block/lane.  Built with -DD4R_RRSWIN_DEBUG_STAGE=<n>
// (kernels/tex/build_tex.sh, D4R_TEX_CFLAGS) this body stores the same
// registers of its own stage n in the same order, so the two dumps compare byte
// for byte.
#ifndef D4R_RRSWIN_DEBUG_STAGE
#define D4R_RRSWIN_DEBUG_STAGE 0
#endif

DEVF void debug_store(const uint8_t* prm, const uint32_t* regs, int n)
{
    const uint32_t block = __builtin_amdgcn_workgroup_id_x()
                         + 81u * __builtin_amdgcn_workgroup_id_y();
    if (block >= 64u) {
        return;   // the oracle's per-block slots only cover the first 64 blocks
    }
    const uint64_t arena = *(const uint64_t*)(prm + kInPlane);
    // the oracle's slot, swin_dec1_recipe.py's debug_slot: 32 lanes x n x 4 bytes,
    // at least 8192, rounded up to a power of two (the oracle's splice forms
    // block*slot with a shift)
    const uint64_t want = (uint64_t)n * 128ull > 8192ull ? (uint64_t)n * 128ull : 8192ull;
    uint64_t slot = 1;
    while (slot < want) {
        slot <<= 1;
    }
    uint32_t* dst = (uint32_t*)(arena + (uint64_t)block * slot
                                 + (uint64_t)lane_id() * (uint64_t)n * 4ull);
    for (int i = 0; i < n; ++i) {
        dst[i] = regs[i];
    }
}

// ---------------------------------------------------------------------------
// Mirror cut. The nine 16-row A fragments of the staged block input arrive in
// this order (the first mma's operand order, .f16.e4m3 fragments of four b32
// each):
//   f0 %r1084..  f1 %r1104..  f2 %r1124..     (m-group 0's three steps)
//   f3 %r1564..  f4 %r1584..  f5 %r1604..     (m-group 1)
//   f6 %r2044..  f7 %r2064..  f8 %r2084..     (m-group 2)
// then phase 1's sixteen bias seeds (%r1621 %r1631 ... %r2041 %r2051, read from
// W + 256*tid.z + 4*(laneid&3) + 24576..24816), the weight image and the parameter
// block.  No other C register is passed: every other mma's C operand is another
// mma's D register, which this body computes itself.
//
// The ten values the epilogue reads out of the deleted region cross the cut
// through the shared array, not `call.uni`'s return parameters: ZLUDA lowers
// those to the function's LLVM return type, so there is no .param-space object
// and no pointer is formed.  `scratch` is the base of the entry's 6400-byte
// .shared array, 40 bytes apart per lane.
extern "C" __attribute__((device)) void d4r_swin_dec1(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5,
    uint32_t a6, uint32_t a7, uint32_t a8, uint32_t a9, uint32_t a10, uint32_t a11,
    uint32_t a12, uint32_t a13, uint32_t a14, uint32_t a15, uint32_t a16, uint32_t a17,
    uint32_t a18, uint32_t a19, uint32_t a20, uint32_t a21, uint32_t a22, uint32_t a23,
    uint32_t a24, uint32_t a25, uint32_t a26, uint32_t a27, uint32_t a28, uint32_t a29,
    uint32_t a30, uint32_t a31, uint32_t a32, uint32_t a33, uint32_t a34, uint32_t a35,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t b4, uint32_t b5,
    uint32_t b6, uint32_t b7, uint32_t b8, uint32_t b9, uint32_t b10, uint32_t b11,
    uint32_t b12, uint32_t b13, uint32_t b14, uint32_t b15,
    const uint8_t* w, const uint8_t* prm, uint32_t scratch)
{
    const uint32_t as1[36] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                              a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23,
                              a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35};
    const uint32_t seeds[16] = {b0, b1, b2, b3, b4, b5, b6, b7,
                                b8, b9, b10, b11, b12, b13, b14, b15};
    // The epilogue's ten registers: phase 26's eight accumulator halves, then
    // %r18991 and %r18993.  Address space 3 (the shared scratch) explicitly: a
    // generic store lowers to flat, which faults.
    lds_u32* const out = (lds_u32*)(uintptr_t)scratch + 10 * (uint32_t)lane_id();
    const uint32_t lane = lane_id();
    const int t = (int)(lane & 3);      // the lane's two columns of each n-tile

#if D4R_RRSWIN_DEBUG_STAGE == 1
    {
        uint32_t regs[52];
#pragma unroll
        for (int i = 0; i < 36; ++i) {
            regs[i] = as1[i];
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            regs[36 + i] = seeds[i];
        }
        debug_store(prm, regs, 52);
    }
#endif

    // ---- phase 1 -----------------------------------------------------------
    // The 144 mma of s403-s546 are three m-groups of 48 tiles.  Within a group the
    // 48 tiles are eight chains of three: chain gp starts from the bias seed
    // 2*gp + h (broadcast into both C registers), takes its A operand from
    // fragment 3*mgrp + j at step j, its B operand from the 512-byte block
    // p = 3*gp + j at the output columns 16*p + 8*h, and adds to the previous
    // step's D -- so the chain is a running sum along j, not a k-accumulation.
    // The tile index is i = 48*mgrp + 2*p + h; the output is 48 rows by 384
    // columns and every element is written by exactly one tile (m-tile i/48,
    // n-tile i%48).  The weight image is per wave: the B blocks at
    // W + 12288*tid.z, the seeds at W + 256*tid.z + 4*t + 24576 + 16*seed.
    const int base1 = 12288 * (int)__builtin_amdgcn_workitem_id_z();
    uint32_t d1[3][8][3][2][2];   // [m-group][chain][step][n-tile half][row half]
#pragma unroll
    for (int mgrp = 0; mgrp < 3; ++mgrp) {
#pragma unroll
        for (int gp = 0; gp < 8; ++gp) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int p = 3 * gp + j;
                    const uint32_t seed = seeds[2 * gp + h];
                    const uint32_t c0 = j == 0 ? seed : d1[mgrp][gp][j - 1][h][0];
                    const uint32_t c1 = j == 0 ? seed : d1[mgrp][gp][j - 1][h][1];
                    mma_tile(as1 + 4 * (3 * mgrp + j), w, base1, 16 * p + 8 * h,
                             c0, c1, t, d1[mgrp][gp][j][h]);
                }
            }
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 2
    {
        // the oracle's stage p1: the 144 mma D pairs in mma order
        uint32_t regs[288];
        int i = 0;
#pragma unroll
        for (int mgrp = 0; mgrp < 3; ++mgrp) {
#pragma unroll
            for (int gp = 0; gp < 8; ++gp) {
#pragma unroll
                for (int j = 0; j < 3; ++j) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        regs[i++] = d1[mgrp][gp][j][h][0];
                        regs[i++] = d1[mgrp][gp][j][h][1];
                    }
                }
            }
        }
        debug_store(prm, regs, 288);
    }
#endif

    // ---- handing the accumulators back ------------------------------------
    // %r18991 = 32*tid.z (PTX line 38396) and %r18993 = the shared base (38394);
    // both sit inside the cut, so the epilogue cannot recompute them.  Phase 26's
    // eight accumulator halves (%r18794 %r18795 %r18804 %r18805 %r18834 %r18835
    // %r18844 %r18845) are not implemented yet: lane-distinct placeholders keep
    // the read-back path observable without producing wild magnitudes.
    out[0] = lane;
    out[1] = lane + 1u;
    out[2] = lane + 2u;
    out[3] = lane + 3u;
    out[4] = lane + 4u;
    out[5] = lane + 5u;
    out[6] = lane + 6u;
    out[7] = lane + 7u;
    out[8] = 32u * __builtin_amdgcn_workitem_id_z();
    out[9] = scratch;
}