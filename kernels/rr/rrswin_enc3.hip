// Native replacement for the Ray Reconstruction denoiser's third Swin
// encoder, cuda_dldn_engine_swin_enc3_kernel (nvngx_dlssd.dll 310.7).
//
// STATE: phases 1..10 are implemented and bit-exact against the debug oracle
// (see "Ground truth" below): stages 1,2,3,4,6,7,8 verified byte-identical with
// a proven control.  Phases 11..48 are NOT implemented -- their accumulators are
// zero -- so the layer's three output surfaces still differ from the translated
// reference and the four accumulators handed back through the arena are zero.
// The missing work is the whole block §3 of rrswin_enc3_epilogues.md describes
// after E10: E11..E13 (shared re-read), E14 (row-RMS norm), E15/E17 (clamped
// cubic activation), E16/E18 (table+residual, re-pack), the 15 repeats of the
// P15..P46 MLP pairs, then E46 (arena scatter), E47 (surface scatter) and E48
// (softmax, 3x3 texture gather, three sust.b surface writes).
//
// ---------------------------------------------------------------------------
// The kernel is one warp per (8x8 token patch, wave): grid (21, 13, 1),
// block (32, 1, 4), four waves per block.  The waves are *not* redundant:
// wave z reads its own 24576-byte slice of the prepared weight image (line 9765
// `mul.lo.s32 %r21780, %r12, 12288` -> `mul.wide.u32 %rd406, %r21780, 2`),
// while all four read the *same* staged A operands out of shared memory (the
// staging is a cooperative cross-wave shuffle, lines 9614-9763: wave z stages
// the fragments of m-tile z and z+4, the reads at 9771-9794 carry no %tid.z
// term, and the `bar.sync 0` at 9764 separates the two).
//
// The PTX this file replaces is kernels/extracted/rr/0016-PREPASS_ENTRYPOINT_NAME.ptx.
// kernels/rr/swin_enc3_recipe.py's recipe keeps NVIDIA's prologue (lines
// 1-9811: the texture/plane reads, the A staging, the C register setup) and
// the epilogue (from line 41984: E48's texture gather and surface stores) and
// replaces the 936 mma of the 48 phases with the call below.
//
// Linked into ZLUDA's compile of that PTX (D4R_ZLUDA_EXTRA_BC), so
// tex_common.h's helpers and ZLUDA's own device functions are available.
// Compiled with -nogpuinc, so only plain clang builtins and attributes are used.
#include "../tex/tex_common.h"

// ---------------------------------------------------------------------------
// Parameter block. Fields are addressed by their byte offset in the PTX, so the
// raw image is kept and read through a typed accessor.
struct DldnSwinEnc3Params
{
    uint8_t raw[144];
};
static_assert(sizeof(DldnSwinEnc3Params) == 144, "parameter block");

enum : int
{
    kWeights = 40,   // NVIDIA's prepared weight image (arg040), read-only
    kPlaneArg = 48,  // plane arena pointer, as make_ptx.py's debug splice reads it
};

#define DEVF __attribute__((device)) static inline
typedef float lds_f32 __attribute__((address_space(3)));
typedef uint32_t lds_u32 __attribute__((address_space(3)));
typedef uint8_t lds_u8 __attribute__((address_space(3)));

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
// (VERIFIED on all 1024 (n, k) pairs of a tile; the 64*(n&7) term is what keeps
// columns 16..31 from silently reading as zero.)
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

// the low f16 of a packed pair (column 2t)
DEVF half_t half_lo(uint32_t pair)
{
    return __builtin_bit_cast(half_t, (uint16_t)pair);
}

// the high f16 of a packed pair (column 2t+1)
DEVF half_t half_hi(uint32_t pair)
{
    return __builtin_bit_cast(half_t, (uint16_t)(pair >> 16));
}

// f16 bits -> f32
DEVF float h2f(uint32_t bits)
{
    return (float)__builtin_bit_cast(half_t, (uint16_t)bits);
}

// add.f16x2 on a packed pair (the PTX's reduction and requantisation steps)
DEVF uint32_t pair_add(uint32_t a, uint32_t b)
{
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(hv2, a) + __builtin_bit_cast(hv2, b));
}

DEVF uint32_t pair_of(half_t a, half_t b)
{
    return (uint32_t)__builtin_bit_cast(uint16_t, a)
         | ((uint32_t)__builtin_bit_cast(uint16_t, b) << 16);
}

// ---------------------------------------------------------------------------
// One stem GEMM: 96 rows (six m-tiles of 16) x 32 columns, K=32, f16 output.
//
// A lane's four A registers hold, for each of its two rows, the eight e4m3
// values of k = 2t + (j&1) + 16*((j>>1)&1) (byte j = 0..3 of register 0) and of
// k + 8 (register 2); the four lanes that differ in t hold the four disjoint
// eighths of the 32 k's, so two butterfly steps broadcast the other three
// eighths in.  The B operand is read from the weight image (waddr) for the lane's
// own two columns of each n-tile.  The dot product runs in f32 and is rounded to
// f16 once, which is what the translated mma does: e4m3 x e4m3 products are
// exact and 32 of them sum exactly in f32, so this reproduces the translated
// kernel bit for bit.
//
// `cd` is the phase's C operand, six m-tiles x two row halves x four n-tiles of
// packed f16 pairs (48 words), or null for the chain's first phase (whose C is
// the zero register %r10141).  `d` is written in the same layout.
DEVF void stem_gemm(const uint32_t (*as)[4], const uint8_t* w, int base,
                    const uint32_t* cd, uint32_t* d)
{
    const int t = (int)(lane_id() & 3);   // the lane's two columns of each n-tile
    for (int nt = 0; nt < 4; ++nt) {
        float bv[2][32];
        for (int c = 0; c < 2; ++c) {
            const int n = 8 * nt + 2 * t + c;
            for (int k = 0; k < 32; ++k) {
                bv[c][k] = e4m3_to_f32(w[waddr(base, k, n)]);
            }
        }
        for (int mi = 0; mi < 6; ++mi) {
            float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
            for (int reg = 0; reg < 2; ++reg) {
                for (int row = 0; row < 2; ++row) {
                    // the register's value on this lane and on lanes t^1, t^2, t^3
                    const uint32_t p0 = as[mi][2 * reg + row];
                    uint32_t p[4];
                    p[0] = p0;
                    p[1] = bfly1_bits(p0);
                    p[2] = bfly2_bits(p0);
                    p[3] = bfly1_bits(p[2]);
                    for (int m = 0; m < 4; ++m) {
                        const int kk = 2 * (t ^ m);
                        for (int j = 0; j < 4; ++j) {
                            const int k = 8 * reg + kk + (j & 1) + 16 * ((j >> 1) & 1);
                            const float av = e4m3_to_f32((p[m] >> (8 * j)) & 0xffu);
                            acc[row][0] += av * bv[0][k];
                            acc[row][1] += av * bv[1][k];
                        }
                    }
                }
            }
            for (int row = 0; row < 2; ++row) {
                // the accumulator is d[mi][row][nt] in memory order, so the
                // (mi, row, nt) entry sits at (2*mi + row)*4 + nt
                const uint32_t seed = cd ? cd[(2 * mi + row) * 4 + nt] : 0u;
                d[(2 * mi + row) * 4 + nt] = pair_of((half_t)(acc[row][0] + h2f(seed)),
                                                     (half_t)(acc[row][1] + h2f(seed >> 16)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ground truth for a stage comes from kernels/rr/swin_enc3_recipe.py's debug
// recipe: with D4R_RRSWIN_DEBUG=w<n> (n = 1..48) the splice keeps NVIDIA's
// whole body and stores that phase's accumulator registers into the plane arena
// (param_0+48) at D4R_RRSWIN_DEBUG_BASE, and the launch dump then holds the
// translated values for a block/wave/lane.  Built with
// -DD4R_RRSWIN_DEBUG_STAGE=<n> (kernels/build.sh rrswin, D4R_TEX_CFLAGS), this
// body stores the same registers of its own phase n in the same order, so the
// two payload regions compare byte for byte.
//
// Two things about that arena are measured, not assumed, and both silently
// void a comparison if you get them wrong:
//
//  * param_0+48 is a *window* inside the 65,945,600-byte block the launcher
//    dumps under the name arg008 (the replay manifest gives arg048 =
//    alloc+37,683,200), so the payload is read from the arg008 file at
//    37,683,200 + base, not at base.
//  * the base must clear the arena bytes the rest of the frame rewrites.  At
//    524,288 the payload window comes back byte-identical to the plain
//    translated run -- something later in the frame overwrites arena bytes
//    524,288..2,473,984 -- so a "MATCH" there is ambient memory.  4,194,304 is
//    above that, and 4 MiB + 64 * 16384 stays inside the arena.  The shared
//    default of 46,000,000 is past the end of enc3's allocation entirely.
//
// So every verdict below is stated with its control: the oracle's window has to
// differ from the same window of a reference run (same layer, translated, no
// D4R_RRSWIN_DEBUG) before a match means anything.
#ifndef D4R_RRSWIN_DEBUG_STAGE
#define D4R_RRSWIN_DEBUG_STAGE 0
#endif
#ifndef D4R_RRSWIN_DEBUG_BASE
#define D4R_RRSWIN_DEBUG_BASE 4194304
#endif

DEVF void debug_store(const uint8_t* prm, const uint32_t* regs, int n)
{
    // Four waves per block, and each reads a different slice of the weight image,
    // so the oracle's per-block slot carries %tid.z too: block =
    // (ctaid.x + 21*ctaid.y)*4 + tid.z, first 64 of those.
    const uint32_t block = (__builtin_amdgcn_workgroup_id_x()
                          + 21u * __builtin_amdgcn_workgroup_id_y()) * 4u
                         + (uint32_t)__builtin_amdgcn_workitem_id_z();
    if (block >= 64u) {
        return;   // the oracle's per-block slots only cover the first 64
    }
    const uint64_t arena = *(const uint64_t*)(prm + kPlaneArg);
    // the oracle's slot, swin_enc3_recipe.debug_slot: 32 lanes x n x 4 bytes, at
    // least 8192, rounded up to a power of two
    const uint64_t want = (uint64_t)n * 128ull > 8192ull ? (uint64_t)n * 128ull : 8192ull;
    uint64_t slot = 1;
    while (slot < want) {
        slot <<= 1;
    }
    uint32_t* dst = (uint32_t*)(arena + D4R_RRSWIN_DEBUG_BASE
                                 + (uint64_t)block * slot
                                 + (uint64_t)lane_id() * (uint64_t)n * 4ull);
    for (int i = 0; i < n; ++i) {
        dst[i] = regs[i];
    }
}

// stage registers in the oracle's order: the phase's mma in order, each writing
// the (row g, row g+8) f16 pair.
DEVF void stage_regs(const uint32_t* d, uint32_t regs[48])
{
    int i = 0;
    for (int mi = 0; mi < 6; ++mi) {
        for (int nt = 0; nt < 4; ++nt) {
            regs[i++] = d[(2 * mi + 0) * 4 + nt];
            regs[i++] = d[(2 * mi + 1) * 4 + nt];
        }
    }
}

// ---------------------------------------------------------------------------
// The attention window: phases 9 and 10.
//
// Phase 9 (64x96x32, the score GEMM Q·Kᵀ) takes *both* operands from P7's
// accumulators: its four A fragments are the query rows 16*mi + g + 8*row
// (mi = 0..3) and its twelve B fragments the key rows 16*(nb>>1) + 8*(nb&1) + g
// (nb = 0..11, i.e. all six m-tiles of the 96-row stem output).  That is the
// same row split enc0's phase 5 uses, and the packs are built the same way:
// word 0 of a row packs n-tiles 0 and 2, word 1 packs 1 and 3, and the four
// lanes t = 0..3 of a group hold the four disjoint eighths of the 32 k's, which
// the two butterfly steps bring together.
//
// A lane therefore owns one row group of the queries and one of the keys and
// can only dot each with itself, so every lane *publishes* its two key words
// per n-tile into the shared arena -- 12 x 2 x 32 lanes = 768 words, which
// fits the wave's 3200-byte share of the 12800-byte arena -- and then forms the
// dots its own D registers hold: query row 16*mi + g + 8*row against key rows
// 16*(nb>>1) + 8*(nb&1) + 2t + c, published by the lanes 4*(2t + c) + 0..3.
//
// The C operand is the 11792-byte f16 score bias at weight bytes 8448..20240,
// 24 tiles of 512 bytes: the seed of (m-tile, n-tile) is in tile
// 4*(nb>>1) + mi, at the lane's 16-byte record + 8*(nb&1) + 4*row.  (s6368:
// `add.s64 %rd94, %rd425, 8448` with %rd425 = the wave's slice + 16*laneid.)

// the f16 the PTX's literals carry: 0f3C8CCB50, 0dBFE1CC0000000000, 0f3F6D6000, 0f3FB00000
DEVF half_t hf16(uint16_t bits)
{
    return __builtin_bit_cast(half_t, bits);
}

DEVF hv2 hv2_of(uint32_t p)
{
    return __builtin_bit_cast(hv2, p);
}

DEVF uint32_t u32_of(hv2 v)
{
    return __builtin_bit_cast(uint32_t, v);
}

// ZLUDA's own rcp.approx.ftz.f32 helper: calling it is what makes the reciprocal bit-exact.
extern "C" __attribute__((device)) float __zluda_ptx_impl_rcp_approx_f32(float);

// the e4m3 code of one f16 (cvt.rn.satfinite.e4m3x2.f16x2 of a pair of equal halves)
DEVF uint32_t q8(half_t v)
{
    return e4m3x2(pair_of(v, v));
}

// the f16 at (row, col) of the stem's 96x32 block, col = 8*n-tile + 2t + c
DEVF half_t dcell(const uint32_t* d, int row, int col)
{
    const uint32_t p = d[(2 * (row >> 4) + ((row >> 3) & 1)) * 4 + (col >> 3)];
    return (col & 1) ? half_hi(p) : half_lo(p);
}

// the lane's two e4m3-packed words for row r: word 0 packs n-tiles 0 and 2,
// word 1 packs 1 and 3
DEVF void row_words(const uint32_t* d, int r, uint32_t w[2])
{
    w[0] = e4m3x2(pair_of(dcell(d, r, 0), dcell(d, r, 1)))
         | (e4m3x2(pair_of(dcell(d, r, 16), dcell(d, r, 17))) << 16);
    w[1] = e4m3x2(pair_of(dcell(d, r, 8), dcell(d, r, 9)))
         | (e4m3x2(pair_of(dcell(d, r, 24), dcell(d, r, 25))) << 16);
}

// out[i][m]: word i as lane t^m holds it, which is what the mma pairs across the group
DEVF void bfly2x4(const uint32_t w[2], uint32_t out[2][4])
{
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        out[i][0] = w[i];
        out[i][1] = bfly1_bits(w[i]);
        out[i][2] = bfly2_bits(w[i]);
        out[i][3] = bfly1_bits(out[i][2]);
    }
}

// word i at butterfly index m (0..3), without the dynamic index into the pair
DEVF uint32_t word_at(const uint32_t out[2][4], int i, int m)
{
    return m == 0 ? out[i][0] : (m == 1 ? out[i][1] : (m == 2 ? out[i][2] : out[i][3]));
}

// the 32-term dot product of two packed operand word pairs, in the mma's own k order
DEVF float dot32(const uint32_t a[2][4], const uint32_t b[2][4])
{
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc += e4m3_to_f32((a[i][m] >> (8 * j)) & 0xffu)
                     * e4m3_to_f32((b[i][m] >> (8 * j)) & 0xffu);
            }
        }
    }
    return acc;
}

// E9(a), one element: the scale, the clamp, the two clamped-cubic fma pairs and the trick
// that moves the mantissa into the exponent field ((u << 5) & 0x7FE07FE0 over the pair).
DEVF uint32_t expval_pair(uint32_t score)
{
    hv2 m = hv2_of(score) * __builtin_bit_cast(hv2, pair_of(hf16(0x2466), hf16(0x2466)));
    m = __builtin_elementwise_max(m, __builtin_bit_cast(hv2, pair_of(hf16(0xb873), hf16(0xb873))));
    m = __builtin_elementwise_min(m, __builtin_bit_cast(hv2, pair_of(hf16(0x3873), hf16(0x3873))));
    const hv2 tt = __builtin_elementwise_fma(m, -m,
                        __builtin_bit_cast(hv2, pair_of(hf16(0x3b6b), hf16(0x3b6b))));
    const uint32_t u = u32_of(__builtin_elementwise_fma(m, tt,
                        __builtin_bit_cast(hv2, pair_of(hf16(0x3d80), hf16(0x3d80)))));
    return (u << 5) & 0x7fe07fe0u;
}

// E9(a)-(c), one row: the twelve exponent pairs, the in-lane sum tree
// (((e0+e2)+(e4+e6))+(e8+e10)) + (((e1+e3)+(e5+e7))+(e9+e11)), the two butterfly steps
// that finish the row's 96 columns, the two halves' add, the reciprocal and the
// twelve products.  There is no row maximum in this epilogue (the `max.f16x2`
// of E9(a) is the clamp), which the oracle's stage 9 confirms.
DEVF void row_probs(const uint32_t* d9, int mi, int rh, uint32_t prob[12])
{
    hv2 ev[6], eo[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const uint32_t a = d9[(2 * mi + rh) * 12 + 2 * i];
        const uint32_t b = d9[(2 * mi + rh) * 12 + 2 * i + 1];
        ev[i] = hv2_of(expval_pair(a));
        eo[i] = hv2_of(expval_pair(b));
    }
    const hv2 even = ((ev[0] + ev[1]) + (ev[2] + ev[3])) + (ev[4] + ev[5]);
    const hv2 odd = ((eo[0] + eo[1]) + (eo[2] + eo[3])) + (eo[4] + eo[5]);
    hv2 s = even + odd;
    s = s + hv2_of(bfly1_bits(u32_of(s)));
    s = s + hv2_of(bfly2_bits(u32_of(s)));
    const uint32_t p = u32_of(s);
    // the PTX broadcasts the f16 sum of the two halves (add.f16) and takes
    // cvt.f32.f16 of it: the value, not its bits
    const half_t r = (half_t)__zluda_ptx_impl_rcp_approx_f32((float)(half_lo(p) + half_hi(p)));
    const uint32_t rcp = pair_of(r, r);
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        prob[2 * i] = u32_of(ev[i] * hv2_of(rcp));
        prob[2 * i + 1] = u32_of(eo[i] * hv2_of(rcp));
    }
}

// ---------------------------------------------------------------------------
// The native body.
//
// Nothing crosses the cut in: the 24 staged A fragments are still in the shared
// arena when the splice is taken (their last ld.shared is line 9794 and the
// arena's next writer is E10 at line 22563), so the body reads them itself at
// `smem + 16*laneid + 512*k`.  The B operands come from the weight image and
// the C seeds are zero or a previous phase's accumulator, so both are
// recomputed here.
//
// Phase 48's four accumulators cross the cut back through shared memory, not
// `call.uni`'s return parameters: ZLUDA lowers those to the function's LLVM
// return type (emit.rs's emit_call makes one value, or an
// get_or_create_struct_type struct for several, and distributes them with
// LLVMBuildExtractValue), so there is no .param-space object and no pointer is
// formed -- the callee reads 0x0 in all four slots, and a 4 x i32 C++ struct
// return is an sret pointer, a different LLVM signature that will not link.
// `scratch` is the base of the arena (`.shared 12800 bytes`), dead after the cut
// except for the window the splice itself reads back, 16 bytes apart per lane.
extern "C" __attribute__((device)) void d4r_swin_enc3(
    const uint8_t* w, const uint8_t* prm, uint32_t scratch)
{
    const uint32_t z = (uint32_t)__builtin_amdgcn_workitem_id_z();
    const uint8_t* wz = w + z * 24576u;      // the wave's own weight slice

    // the 24 staged A fragments: four shift groups of six m-tiles, read at
    // `smem + 16*laneid + 512*k` exactly as the PTX does at lines 9771-9794
    const lds_u32* sm = (const lds_u32*)(uintptr_t)scratch;
    uint32_t as[24][4];
#pragma unroll
    for (int k = 0; k < 24; ++k) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            as[k][i] = sm[4 * lane_id() + 128 * k + i];
        }
    }

    // ---- the stem: two chains of four phases, P1..P8 -------------------
    //
    // Phase p (1-based) uses chain (p-1)&1 and group (p-1)>>1.  Its B block sits
    // at 256 + 1024*group + 4096*chain from the wave's slice -- the eight
    // `add.s64 %rdNN, %rd409, <base>` of the prologue and E1..E7 give exactly
    // P1 256, P2 4352, P3 1280, P4 5376, P5 2304, P6 6400, P7 3328, P8 7424 --
    // and its m-tile mi takes A fragment 4*mi + group, i.e. the four groups are
    // four shifted views of one 96-row tile (the shift structure of note U6).
    //
    // Each chain accumulates: P3's C is P1's D pair for the same (mi, nt)
    // (s6186 `C={%r4296,%r4297}`, P1's first mma), P5's is P3's, P7's is P5's,
    // and P1's and P2's C is the zero register %r10141.
    uint32_t d[2][48] = {};
    for (int chain = 0; chain < 2; ++chain) {
        for (int g = 0; g < 4; ++g) {
            uint32_t a[6][4];
#pragma unroll
            for (int mi = 0; mi < 6; ++mi) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a[mi][i] = as[4 * mi + g][i];
                }
            }
            stem_gemm(a, wz, 256 + 1024 * g + 4096 * chain, d[chain], d[chain]);
#if D4R_RRSWIN_DEBUG_STAGE
            if (D4R_RRSWIN_DEBUG_STAGE == 2 * g + 1 + chain) {
                uint32_t regs[48];
                stage_regs(d[chain], regs);
                debug_store(prm, regs, 48);
            }
#endif
        }
    }

    // ---- phase 9: the score GEMM, Q·Kᵀ over the 96-row stem output ----
    // The arena's 12800 bytes are split four ways (3200 per wave); the stem's
    // A fragments have already been read into registers above, so the wave's
    // whole share is free from here on.
    const int g = (int)(lane_id() >> 2);
    const int t = (int)(lane_id() & 3);
    lds_u32* const pub = (lds_u32*)(uintptr_t)scratch + 800 * (int)z;
    {
        const uint32_t* qk = d[0];        // P7's output: queries and keys alike
#pragma unroll
        for (int nb = 0; nb < 12; ++nb) {
            uint32_t kw[2];
            row_words(qk, 16 * (nb >> 1) + 8 * (nb & 1) + g, kw);
            pub[64 * nb + lane_id()] = kw[0];
            pub[64 * nb + 32 + lane_id()] = kw[1];
        }
    }
    __builtin_amdgcn_s_barrier();
    uint32_t d9[48] = {};
    {
        const uint32_t* qk = d[0];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            for (int row = 0; row < 2; ++row) {
                uint32_t aw[2], a[2][4];
                row_words(qk, 16 * mi + g + 8 * row, aw);
                bfly2x4(aw, a);
                for (int nb = 0; nb < 12; ++nb) {
                    float s[2] = {0.0f, 0.0f};
                    for (int c = 0; c < 2; ++c) {
                        // the mma's eight n-index rows are the key rows
                        // 16*(nb>>1) + 8*(nb&1) + n, and the lane's two D columns
                        // are n = 2t + c: each half dots a *different* key row
                        uint32_t kb[2][4];
                        for (int i = 0; i < 2; ++i) {
#pragma unroll
                            for (int m = 0; m < 4; ++m) {
                                kb[i][m] = pub[64 * nb + 32 * i + 4 * (2 * t + c) + (t ^ m)];
                            }
                        }
                        s[c] = dot32(a, kb);
                    }
                    const uint32_t bias = *(const uint32_t*)(wz + 8448
                                                + 512 * (4 * (nb >> 1) + mi)
                                                + 16 * lane_id() + 8 * (nb & 1) + 4 * row);
                    d9[(2 * mi + row) * 12 + nb] = pair_of((half_t)(s[0] + h2f(bias)),
                                                            (half_t)(s[1] + h2f(bias >> 16)));
                }
            }
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 9
    {
        uint32_t regs[96];
        for (int mi = 0; mi < 4; ++mi) {
            for (int nb = 0; nb < 12; ++nb) {
                regs[2 * (mi * 12 + nb)] = d9[(2 * mi + 0) * 12 + nb];
                regs[2 * (mi * 12 + nb) + 1] = d9[(2 * mi + 1) * 12 + nb];
            }
        }
        debug_store(prm, regs, 96);
    }
#endif

    // ---- E9 and phase 10: the softmax and P·V --------------------------
    // E9's V transpose is a movmatrix of P8's accumulators (the even chain);
    // the scalar body reaches it by publishing the 96x32 tile as one e4m3 byte
    // per element (3072 bytes, the same window the score keys used) and letting
    // every lane walk all 96 k's.  k-step ks packs key columns {4ks, 4ks+2} of
    // the probabilities into the A fragment's first word and {4ks+1, 4ks+3}
    // into its second, which is the same packing enc0's phase 6 uses.
    {
        lds_u8* const vs = (lds_u8*)(uintptr_t)scratch + 3200 * (int)z;
        const uint32_t* v = d[1];          // P8's output
#pragma unroll
        for (int mi4 = 0; mi4 < 6; ++mi4) {
            for (int rh4 = 0; rh4 < 2; ++rh4) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    for (int c = 0; c < 2; ++c) {
                        const int r = 16 * mi4 + 8 * rh4 + g;
                        vs[r * 32 + 8 * nt + 2 * t + c] =
                            (uint8_t)q8(dcell(v, r, 8 * nt + 2 * t + c));
                    }
                }
            }
        }
    }
    __builtin_amdgcn_s_barrier();
    uint32_t d10[48] = {};
    {
        const lds_u8* const vs = (const lds_u8*)(uintptr_t)scratch + 3200 * (int)z;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            for (int rh = 0; rh < 2; ++rh) {
                uint32_t prob[12];
                row_probs(d9, mi, rh, prob);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    half_t prev0 = hf16(0x0000), prev1 = hf16(0x0000);
#pragma unroll
                    for (int ks = 0; ks < 3; ++ks) {
                        uint32_t aw[2], a[2][4];
                        aw[0] = e4m3x2(prob[4 * ks]) | (e4m3x2(prob[4 * ks + 2]) << 16);
                        aw[1] = e4m3x2(prob[4 * ks + 1]) | (e4m3x2(prob[4 * ks + 3]) << 16);
                        bfly2x4(aw, a);
                        float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
                        for (int kk = 0; kk < 32; ++kk) {
                            const float pv = e4m3_to_f32(
                                (word_at(a, (kk >> 3) & 1, t ^ ((kk >> 1) & 3))
                                 >> (8 * (2 * (kk >> 4) + (kk & 1)))) & 0xffu);
                            const int r = (32 * ks + kk) * 32 + 8 * nt + 2 * t;
                            acc0 += pv * e4m3_to_f32(vs[r]);
                            acc1 += pv * e4m3_to_f32(vs[r + 1]);
                        }
                        // each k-step rounds to f16 before the next adds to it,
                        // which is why the chain has three separate mma
                        prev0 = (half_t)(acc0 + (float)prev0);
                        prev1 = (half_t)(acc1 + (float)prev1);
                    }
                    d10[(2 * mi + rh) * 4 + nt] = pair_of(prev0, prev1);
                }
            }
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 10
    {
        uint32_t regs[48];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 4; ++nt) {
                regs[i++] = d10[(2 * mi + 0) * 4 + nt];
                regs[i++] = d10[(2 * mi + 1) * 4 + nt];
            }
        }
        debug_store(prm, regs, 48);
    }
#endif
    // ---- handing the accumulators back ----------------------------------
    // Phase 48's last two mma D pairs; the other forty-six phases are not
    // implemented yet, so the four slots carry zero.
    __attribute__((address_space(3))) uint32_t* out =
        ((__attribute__((address_space(3))) uint32_t*)(uintptr_t)scratch) + 12288 / 4
        + 4 * (uint32_t)lane_id();
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
}

// ---------------------------------------------------------------------------
// Weight image layout of the stem, as byte offsets from the wave's slice
// (wave z's slice starts at z*24576):
//   P1 256   P2 4352   P3 1280   P4 5376   P5 2304   P6 6400   P7 3328   P8 7424
// i.e. phase p (1..8) uses 256 + 1024*((p-1)/2) + 4096*((p-1) and 1); each reads
// two 512-byte tiles (n-tiles 0..1 and 2..3) of 8 bytes per n-tile.
//
// Fragment maps (VERIFIED against the debug oracle, see below; g = lane>>2,
// t = lane&3).  The hardware order of an m16n8k32 .e4m3 fragment is *not* the
// order the PTX ISA figures suggest.  NVIDIA's own epilogue settles it: it
// builds phase 2's A operand as `%r1207 = {cvt(%r877), cvt(%r897)}` where
// %r877/%r897 are the D halves of mma rows 2231/2233, i.e. (row g, n-tile 0,
// cols 2t, 2t+1) and (row g, n-tile 2, cols 2t, 2t+1).  So with the four bytes of
// an A register in memory order:
//   A reg 0 -> row g,     k = 2t + (j&1) + 16*((j>>1)&1)
//   A reg 1 -> row g+8,   same k
//   A reg 2 -> row g,     k = 8 + 2t + (j&1) + 16*((j>>1)&1)
//   A reg 3 -> row g+8,   same k
// and the B half that must pair with it (a0/a1 with b0, a2/a3 with b1) is
//   B byte j (j<4)  -> n = 16*tt + 8*(b>>3) + g, k = 2t + (j&1) + 16*((j>>1)&1)
//   B byte j (j>=4) -> n = 16*tt + 8*(b>>3) + g, k = 8 + 2t + (j&1) + 16*((j>>1)&1)
// with b the byte index within the lane's 16 and tt the 512-byte tile.  Inverted,
// that is waddr() above.  C/D:
//   C/D reg 0 -> row g,   cols t*2, t*2+1
//   C/D reg 1 -> row g+8, cols t*2, t*2+1
//
// A plain f32 dot product over K plus C, rounded to f16 once, reproduces the
// translated kernel's phase-1 output exactly, so the phases need no
// operand-layout gymnastics and no WMMA intrinsics: every GEMM here is exact in
// f32 before the f16 rounding.