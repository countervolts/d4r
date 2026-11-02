// Native replacement for the Ray Reconstruction denoiser's fifth Swin encoder,
// cuda_dldn_engine_swin_enc4_kernel (nvngx_dlssd.dll 310.7).
//
// ---------------------------------------------------------------------------
// STATE: the mirror cut is in place and builds and serves, and phases 1-11 are implemented: the
// ten-phase stem (two chains of five 96x32x32 GEMMs over the shared-staged block input, each
// phase's A operand read from the staging arena exactly where the spliced PTX read it) and the
// 64x96 position-only score GEMM with its 11792-byte f16 bias, followed by E11's cubic-exponent
// softmax.  Every stem stage and the score stage are bit-exact against the debug oracle (see
// "Ground truth" below).  Phases 12-62 -- the P*V GEMM and the 3x3-stencil merge, the sixteen
// 16x160 attention projections with their row-RMS norm, and the twenty banded MLP pairs -- are not
// implemented yet, so the two registers the cut reads back for phase 62 carry lane-distinct
// placeholders and the layer's surface output is still the translated kernel's.
//
// The kernel is one warp per 8x8 token patch with four waves per block (tid.z bounds to 0..3), and
// each wave reads its own 27648-byte slice of the weight image.  The PTX this file replaces is
// kernels/extracted/rr/0017-PREPASS_ENTRYPOINT_NAME.ptx.
//
// kernels/rr/swin_enc4_recipe.py's recipe (D4R_RRSWIN_CUT=1) keeps NVIDIA's prologue (the input
// plane read, the e4m3 requantisation, the 30-fragment shared staging and the first weight loads,
// file lines 1-11882) and the surface epilogue (from line 52672) and replaces the 1386 mma of the
// 62 phases with the call below.  Without that variable the same recipe builds the debug oracle.
//
// Linked into ZLUDA's compile of that PTX (D4R_ZLUDA_EXTRA_BC), so tex_common.h's helpers and
// ZLUDA's own device functions are available. Compiled with -nogpuinc, so only plain clang
// builtins and attributes are used.
#include "../tex/tex_common.h"

// ---------------------------------------------------------------------------
// Parameter block. Fields are addressed by their byte offset in the PTX, so the raw image is kept
// and read through a typed accessor.
struct DldnSwinEnc4Params
{
    uint8_t raw[144];
};
static_assert(sizeof(DldnSwinEnc4Params) == 144, "parameter block");

enum : int
{
    kWeights = 40,   // NVIDIA's prepared weight image (arg040), read-only
    kPlaneArg = 8,    // plane arena (arg008), the 65,945,600-byte allocation the debug store writes past
};

#define DEVF __attribute__((device)) static inline
typedef float lds_f32 __attribute__((address_space(3)));
typedef uint32_t lds_u32 __attribute__((address_space(3)));
typedef uint8_t lds_u8 __attribute__((address_space(3)));

// e4m3 code -> f32, exact. The mma's e4m3 operands reach the FMA path through ZLUDA's exact
// e4m3 -> f16 widening, so this decode is the same value.
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
// (VERIFIED on all 1024 (n, k) pairs of a tile; the 64*(n&7) term is what keeps columns 16..31
// from silently reading as zero.)
DEVF uint32_t waddr(int base, int k, int n)
{
    return (uint32_t)base + 512u * (uint32_t)(n >> 4) + 64u * (uint32_t)(n & 7)
         + 16u * (uint32_t)((k >> 1) & 3) + 8u * (uint32_t)((n >> 3) & 1)
         + 4u * (uint32_t)((k >> 3) & 1) + 2u * (uint32_t)((k >> 4) & 1)
         + (uint32_t)(k & 1);
}

// warp butterfly over the four lanes that hold a row's 32 columns (laneid ^ 1 and laneid ^ 2).
DEVF uint32_t bfly1_bits(uint32_t v)
{
    return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0xB1, 0xf, 0xf, false);
}

DEVF uint32_t bfly2_bits(uint32_t v)
{
    return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0x4E, 0xf, 0xf, false);
}

DEVF float bfly1(float v)
{
    return __builtin_bit_cast(float, bfly1_bits(__builtin_bit_cast(uint32_t, v)));
}

DEVF float bfly2(float v)
{
    return __builtin_bit_cast(float, bfly2_bits(__builtin_bit_cast(uint32_t, v)));
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

// ZLUDA's relu_keep_nan_halves: a negative (or -0) half becomes +0, NaN is kept.
DEVF uint32_t relu_pair(uint32_t p)
{
    if ((p & 0x8000u) != 0 && (p & 0x7fffu) <= 0x7c00u) {
        p &= 0xffff0000u;
    }
    if ((p & 0x80000000u) != 0 && ((p >> 16) & 0x7fffu) <= 0x7c00u) {
        p &= 0x0000ffffu;
    }
    return p;
}

// ---------------------------------------------------------------------------
// One GEMM: 96 rows (six m-tiles of 16) x 32 columns, K=32, f16 output.
//
// A lane's A registers hold, for each m-tile and each of its two rows, the eight e4m3 values of
// k = 2t + (j&1) + 16*((j>>1)&1) (byte j = 0..3) and of k + 8; the four lanes that differ in t
// hold the four disjoint eighths of the 32 k's, so two butterfly steps broadcast the other three
// eighths in. The B operand is read from the weight image (waddr) for the lane's own two columns of
// each n-tile. The dot product runs in f32 and is rounded to f16 once, which is what the
// translated mma does: e4m3 x e4m3 products are exact and 32 of them sum exactly in f32, so this
// reproduces the translated kernel bit for bit.
//
// Exactly one of the three C operands carries the seed: `wbias` (a byte pointer at the phase's bias
// offset, 16 bytes per n-tile), `seeds` (four already lane-indexed packed pairs, a zero C seed), or
// `cmat` (48 lane-indexed pairs of one chain, indexed [m-tile][row][n-tile] as
// (mi*2 + row)*4 + nt, the previous phase's D, whose seed differs per row and per n-tile).
// d is [m-tile][row half][n-tile], each entry the packed f16 pair for columns 2t, 2t+1.
DEVF void gemm(const uint32_t* as, const uint8_t* w, int base, const uint8_t* wbias,
               const uint32_t* seeds, const uint32_t* cmat, int nmi, int t,
               uint32_t d[6][2][4])
{
    for (int nt = 0; nt < 4; ++nt) {
        float bv[2][32];
        for (int c = 0; c < 2; ++c) {
            const int n = 8 * nt + 2 * t + c;
            for (int k = 0; k < 32; ++k) {
                bv[c][k] = e4m3_to_f32(w[waddr(base, k, n)]);
            }
        }
        for (int mi = 0; mi < nmi; ++mi) {
            uint32_t afrag[4];
            for (int j = 0; j < 4; ++j) {
                afrag[j] = as[4 * mi + j];
            }
            uint32_t fixed = 0;
            if (wbias != nullptr) {
                fixed = *(const uint32_t*)(wbias + 16 * nt + 4 * t);
            } else if (seeds != nullptr) {
                fixed = seeds[nt];
            }
            float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
            for (int reg = 0; reg < 2; ++reg) {
                for (int row = 0; row < 2; ++row) {
                    // the register's value on this lane and on lanes t^1, t^2, t^3
                    uint32_t p[4];
                    p[0] = afrag[2 * reg + row];
                    p[1] = bfly1_bits(p[0]);
                    p[2] = bfly2_bits(p[0]);
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
                // (mi*2 + row)*4 + nt packs a chain's 6 x 2 x 4 seeds into 48 words
                const uint32_t bias_pair = (cmat != nullptr) ? cmat[(mi * 2 + row) * 4 + nt] : fixed;
                d[mi][row][nt] = pair_of((half_t)(acc[row][0] + h2f(bias_pair)),
                                         (half_t)(acc[row][1] + h2f(bias_pair >> 16)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ground truth for a stage comes from kernels/rr/swin_enc4_recipe.py: with
// D4R_RRSWIN_DEBUG=p1 (p2, p3, ... or a) the same splice keeps NVIDIA's whole body and stores that
// stage's registers into the plane arena at +46,000,000, which the kernel never otherwise writes,
// and the launch dump then holds the translated values for a block/lane.  Built with
// -DD4R_RRSWIN_DEBUG_STAGE=<n> (kernels/build.sh rrswin, D4R_TEX_CFLAGS), this body stores the same
// registers of its own phase n in the same order, so the two dumps compare byte for byte.
#ifndef D4R_RRSWIN_DEBUG_STAGE
#define D4R_RRSWIN_DEBUG_STAGE 0
#endif

// The same geometry as the recipe's debug_block(): the slot index is
// `wgid.x + 32*wgid.y + 256*workitem_id_z`, capped at 64 slots, and a store only happens when every
// component is in range, so the mapping stays injective however the kernel is launched.  enc4 runs
// four waves per block (tid.z bounds to 0..3) and each wave computes a different result -- its own
// 27648-byte weight slice, its own staging slots -- so the wave has to be part of the index.  The
// slot is max(8192, 32*nregs*4) bytes, rounded up to a power of two.
//
// DEBUG_BASE is a build-time define so the native store and the oracle splice can be moved
// together; enc0's 46,000,000 is unusable here because enc4's arena is smaller and the launch
// dump only exposes arena offsets below 26,378,240 (see the recipe).  The 64-slot cap bounds the
// widest stage (160 registers, a 32,768-byte slot) to 2 MiB.
#ifndef D4R_RRSWIN_DEBUG_BASE
#define D4R_RRSWIN_DEBUG_BASE 20000000
#endif
DEVF void debug_store(const uint8_t* prm, const uint32_t* regs, int n)
{
    const uint32_t bx = __builtin_amdgcn_workgroup_id_x();
    const uint32_t by = __builtin_amdgcn_workgroup_id_y();
    const uint32_t bz = __builtin_amdgcn_workitem_id_z();
    const uint64_t block = bx + 32ull * by + 256ull * bz;
    if (bx >= 32u || by >= 8u || bz >= 4u || block >= 64ull) {
        return;
    }
    const uint64_t arena = *(const uint64_t*)(prm + kPlaneArg);
    const uint64_t want = (uint64_t)n * 128ull > 8192ull ? (uint64_t)n * 128ull : 8192ull;
    uint64_t slot = 1;
    while (slot < want) {
        slot <<= 1;
    }
    uint32_t* dst = (uint32_t*)(arena + (uint64_t)D4R_RRSWIN_DEBUG_BASE + block * slot
                                 + (uint64_t)lane_id() * (uint64_t)n * 4ull);
    for (int i = 0; i < n; ++i) {
        dst[i] = regs[i];
    }
}

// stage registers in the oracle's order: the phase's mma, m-tile major, n-tile minor, each as the
// (row g, row g+8) pair the mma writes.
DEVF void stage_regs(const uint32_t d[6][2][4], uint32_t regs[48])
{
    int i = 0;
    for (int mi = 0; mi < 6; ++mi) {
        for (int nt = 0; nt < 4; ++nt) {
            regs[i++] = d[mi][0][nt];
            regs[i++] = d[mi][1][nt];
        }
    }
}

// ---------------------------------------------------------------------------
// The stem, phases 1-10.  Five shift groups of six staged A fragments, two chains of five
// accumulations: phase 2g+1 and 2g+2 both read shift group g, phase 2g+3 accumulates phase 2g+1's
// result and phase 2g+4 accumulates phase 2g+2's.  Nothing is requantised between them -- the
// chain's C operand is the previous phase's f16 D pair -- so the stem is ten GEMMs over the same
// five input fragments with five different weight blocks.
//
// The staged fragments are read from the shared arena at `smem + 16*laneid + 512*k`, k = 0..29,
// with *no* tid.z term -- exactly the addresses the spliced PTX's 30 ld.shared.v4 used (the
// prologue's store side adds tid.z, so each fragment is written by the one wave that owns it and
// all four waves read all thirty; the body does not need to know that).
//
// The stem's weight block starts at byte 320 of the wave's slice, phase p's at
// 320 + 1024*((p-1)/2) + 5120*((p-1)%2) -- VERIFIED against the PTX's own add.s64 immediates
// (s7550, s7581, s7612, s7643, s7674, s7705, s7736, s7767, s7798, s7829).
//
// Phase 11 and later read the wave slice too (E10's 24 score-bias loads are at 10560 + 512*i of
// the same per-wave base); from E12 on the loads are off the image base directly.
DEVF void stem_group(const lds_u32* smem, int g, uint32_t lane, uint32_t as[24])
{
    for (int mi = 0; mi < 6; ++mi) {
        const lds_u32* src = smem + 128u * (uint32_t)(5 * mi + g) + 4u * lane;
        for (int j = 0; j < 4; ++j) {
            as[4 * mi + j] = src[j];
        }
    }
}

// The next group's C operand: the odd chain's seed is the odd phase's D, the even chain's the even
// phase's, each at the same (m-tile, row, n-tile) -- VERIFIED against the PTX, where phase p's mma
// i reads phase p-2's mma i as its C (s7616 reads %r5333/%r5334, phase 1's first D pair, and so on
// for every odd and even phase up to 10).
// The two chains' seeds live side by side in one 96-word array: the odd chain at 0..47, the even
// chain at 48..95, each packed [m-tile][row][n-tile] as (mi*2 + row)*4 + nt.
//
// The previous layout indexed (4*mi + row)*4 + nt, which strides four words per (m-tile, row)
// pair and so reaches 87 for mi = 5 -- past a 48-word array.  m-tiles 0-2 stayed inside it (their
// seed is zero, so the stale contents were invisible) and m-tiles 3-5 read past the end, which is
// exactly the range that disagreed with the oracle.  The even chain was also written at +24,
// overlapping the odd chain instead of following it.
DEVF void stem_seed(const uint32_t dodd[6][2][4], const uint32_t deven[6][2][4],
                    uint32_t cmat[96])
{
    for (int mi = 0; mi < 6; ++mi) {
        for (int row = 0; row < 2; ++row) {
            for (int nt = 0; nt < 4; ++nt) {
                cmat[(mi * 2 + row) * 4 + nt] = dodd[mi][row][nt];
                cmat[48 + (mi * 2 + row) * 4 + nt] = deven[mi][row][nt];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 11, the 64x96 score GEMM.  Both operands come from phase 9's accumulators (position-only
// attention, S = bias + Q*Q^T): A is m-tile `a` of the 96-row block, B is the same 96 rows read as
// 12 eight-row groups, `nt = 2*mb + r` being m-tile mb's row half r (so the score's 96 columns are
// the 96 rows of the block, the 8x8 token patch plus its halo).
//
// E10 packs each operand's fragment exactly as the mma consumes it (VERIFIED against the PTX's own
// mov.b32 list, s7908-s8027): fragment `a`'s register 2*reg+row is
//     e4m3x2(row_words(row 16a + g + 8*row)[reg])
// and B fragment `nt`'s words 0 and 1 are the same two words of row 16*(nt/2) + 8*(nt%2) + g.  Both
// sides therefore use one byte layout -- byte j of a word is column 2t + (j&1) + 16*((j>>1)&1) of
// word `reg`'s half -- so the dot product is the same 32-term pairing on both sides and two butterfly
// steps bring the other three eighths of the 32 k's in, exactly as enc0's phase 5 does.
//
// A lane only holds its own two rows of the group, and the mma's B fragment is indexed by
// groupID while the score's column is indexed by threadID_in_group, so every lane needs eight rows
// it does not hold.  They go through the arena as e4m3 words: 12 groups x 8 rows x 4 k-eighths x
// two words = 3072 bytes, at arena offset 0 -- shift group 0's staging slots, which nothing reads
// after phase 10.

// the f16 at (row, col) of a 96x32 block of packed pairs, col = 8*n-tile + 2t + c
DEVF half_t dcell(const uint32_t d[6][2][4], int row, int col)
{
    const uint32_t p = d[row >> 4][(row >> 3) & 1][col >> 3];
    return (col & 1) ? half_hi(p) : half_lo(p);
}

// the lane's two e4m3-packed words for row r: word 0 packs columns (2t, 2t+1) and (16+2t,
// 16+2t+1), word 1 the columns 8 and 24 -- E10's pack, which is also the mma's own A fragment order
DEVF void row_words(const uint32_t d[6][2][4], int r, uint32_t w[2])
{
    w[0] = e4m3x2(pair_of(dcell(d, r, 0), dcell(d, r, 1)))
         | (e4m3x2(pair_of(dcell(d, r, 16), dcell(d, r, 17))) << 16);
    w[1] = e4m3x2(pair_of(dcell(d, r, 8), dcell(d, r, 9)))
         | (e4m3x2(pair_of(dcell(d, r, 24), dcell(d, r, 25))) << 16);
}

// out[i][m]: word i as lane t^m holds it, which is what the dot product pairs across the group
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

// Phase 11's C seeds: the 11792-byte f16 table at wave byte 10560, read as 24 tiles of four words
// per lane at 10560 + 512*j.  Each word is one (row, n-tile) f16 pair -- columns 2t and 2t+1 -- so a
// tile holds two n-tiles x two rows, and the mma's own C operands fix the order: for the tile
// a + 4*(nt>>1) the four words are (row g, nt), (row g+8, nt), (row g, nt+1), (row g+8, nt+1).  The
// PTX confirms the tile stride: phase 11's first A-tile reads 10560 and 10568 for n-tiles 0 and 1
// (s8028, s8029), 12608 and 12616 for n-tiles 2 and 3 (s8030, s8031), and the second A-tile starts
// at 11072 (s8040), i.e. 512 bytes per A-tile.
DEVF uint32_t score_seed(const uint8_t* wv, uint32_t lane, int a, int nt, int row)
{
    return *(const uint32_t*)(wv + 10560 + 512u * (uint32_t)(a + 4 * (nt >> 1)) + 16u * lane
                              + 4u * (uint32_t)(2 * (nt & 1) + row));
}

// The key rows go through the arena at word index 64*nt + 8*row + 2*t + i: twelve groups, eight
// rows, four k-eighths and two words, 768 words.  The lane publishes the rows it holds, row g of
// every group, and then reads back the two rows (2t and 2t+1) its own two output columns need.
DEVF void score(lds_u32* smem, const uint32_t dodd[6][2][4], const uint8_t* wv,
                uint32_t lane, int t, int g, uint32_t d11[4][2][12])
{
#pragma unroll
    for (int nt = 0; nt < 12; ++nt) {
        uint32_t w[2];
        row_words(dodd, 16 * (nt >> 1) + 8 * (nt & 1) + g, w);
        smem[64 * nt + 8 * g + 2 * t] = w[0];
        smem[64 * nt + 8 * g + 2 * t + 1] = w[1];
    }
    __builtin_amdgcn_wave_barrier();
    for (int a = 0; a < 4; ++a) {
        uint32_t aw[2][2];
        row_words(dodd, 16 * a + g, aw[0]);
        row_words(dodd, 16 * a + g + 8, aw[1]);
        uint32_t av[2][2][4];
        bfly2x4(aw[0], av[0]);
        bfly2x4(aw[1], av[1]);
        for (int nt = 0; nt < 12; ++nt) {
            float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
            for (int c = 0; c < 2; ++c) {
                // the mma's B fragment is indexed by groupID, so the score's columns 2t and 2t+1
                // are the rows 2t and 2t+1 of the group -- held by four other lanes each
                const int col = 2 * t + c;
                uint32_t kb[2][4];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        kb[i][m] = smem[64 * nt + 8 * col + 2 * (t ^ m) + i];
                    }
                }
                acc[0][c] = dot32(av[0], kb);
                acc[1][c] = dot32(av[1], kb);
            }
            for (int row = 0; row < 2; ++row) {
                // the seed is per row: the mma's C pair is (row g, row g+8), each a packed f16 pair
                const float bias = h2f(score_seed(wv, lane, a, nt, row));
                d11[a][row][nt] = pair_of((half_t)(acc[row][0] + bias),
                                          (half_t)(acc[row][1] + bias));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// E11, the score softmax.  Byte-for-byte enc0's E5 with the same 64x96 shape (four m-tiles of
// sixteen rows, twelve n-tiles of eight columns): per element the scale, the clamp and the two
// clamped-cubic fma pairs that move the mantissa into the exponent field; then the in-lane sum tree,
// two butterfly steps that finish the row's 96 columns, the two halves' add, the reciprocal and
// the twelve products.  There is no row maximum in this epilogue -- every max.f16x2 belongs to the
// clamp.
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

// one element: the scale, the clamp, the two clamped-cubic fma pairs and the trick that moves the
// mantissa into the exponent field ((u << 5) & 0x7FE07FE0 over the pair)
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

// one row: the twelve exponent pairs, the in-lane sum tree
// (((e0+e2)+(e4+e6))+(e8+e10)) + (((e1+e3)+(e5+e7))+(e9+e11)), the two butterfly steps that finish
// the row's 96 columns, the two halves' add, the reciprocal and the twelve products
DEVF void row_probs(const uint32_t d11[4][2][12], int mi, int rh, uint32_t prob[12])
{
    hv2 ev[6], eo[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        ev[i] = hv2_of(expval_pair(d11[mi][rh][2 * i]));
        eo[i] = hv2_of(expval_pair(d11[mi][rh][2 * i + 1]));
    }
    const hv2 even = ((ev[0] + ev[1]) + (ev[2] + ev[3])) + (ev[4] + ev[5]);
    const hv2 odd = ((eo[0] + eo[1]) + (eo[2] + eo[3])) + (eo[4] + eo[5]);
    hv2 s = even + odd;
    s = s + hv2_of(bfly1_bits(u32_of(s)));
    s = s + hv2_of(bfly2_bits(u32_of(s)));
    const uint32_t p = u32_of(s);
    // the PTX broadcasts the f16 sum of the two halves (add.f16) and takes cvt.f32.f16 of it:
    // the value, not its bits
    const half_t r = (half_t)__zluda_ptx_impl_rcp_approx_f32((float)(half_lo(p) + half_hi(p)));
    const uint32_t rcp = pair_of(r, r);
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        prob[2 * i] = u32_of(ev[i] * hv2_of(rcp));
        prob[2 * i + 1] = u32_of(eo[i] * hv2_of(rcp));
    }
}

// phase 11's 96 registers in the oracle's order: mma, m-tile major, n-tile minor, each as the
// (row g, row g+8) pair the mma writes
DEVF void score_regs(const uint32_t d11[4][2][12], uint32_t regs[96])
{
    int i = 0;
    for (int mi = 0; mi < 4; ++mi) {
        for (int nt = 0; nt < 12; ++nt) {
            regs[i++] = d11[mi][0][nt];
            regs[i++] = d11[mi][1][nt];
        }
    }
}

// ---------------------------------------------------------------------------
// Mirror cut.  The six 16-row fragments of shift group 0 arrive in the first mma's operand order
// (.f16.e4m3 fragments of four b32 each), then phase 1's C seed (%r11676, zero), the weight image,
// the parameter block and the address of the shared arena.
//
// The five registers the region's epilogue reads -- %r29719 (the x extent, param_0+0), %r31372 and
// %r31382 (phase 62's D pair), %r31608 (%tid.z) and %r31615 (the shared arena's address) -- cross
// the cut through the arena, not `call.uni`'s return parameters: ZLUDA lowers those to the
// function's LLVM return type (emit.rs's emit_call makes one value, or a
// get_or_create_struct_type struct for several, and distributes them with LLVMBuildExtractValue),
// so there is no .param-space object and no pointer is formed -- the callee reads 0x0 in all five
// slots.  The body writes them at `scratch + 20*laneid + 4*i` as its last act and the splice reads
// them back immediately, before E62 issues a single shared store.
extern "C" __attribute__((device)) void d4r_swin_enc4(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6,
    uint32_t a7, uint32_t a8, uint32_t a9, uint32_t a10, uint32_t a11, uint32_t a12, uint32_t a13,
    uint32_t a14, uint32_t a15, uint32_t a16, uint32_t a17, uint32_t a18, uint32_t a19, uint32_t a20,
    uint32_t a21, uint32_t a22, uint32_t a23, uint32_t c0,
    const uint8_t* w, const uint8_t* prm, uint32_t scratch)
{
    const uint32_t as1[24] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                              a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23};
    // Address space 3 (the shared arena) explicitly: a generic access lowers to flat, which faults.
    lds_u32* const smem = (lds_u32*)(uintptr_t)scratch;
    const uint32_t lane = lane_id();
    const int t = (int)(lane & 3);   // the lane's two columns of each n-tile
    const int z = (int)__builtin_amdgcn_workitem_id_z();
    // the four waves each read their own 27648-byte slice of the weight image (s7507: tid.z*13824,
    // then s7508 doubles it)
    const uint8_t* const wv = w + 27648u * (uint32_t)z;

    uint32_t dodd[6][2][4];   // phases 1, 3, 5, 7, 9
    uint32_t deven[6][2][4];  // phases 2, 4, 6, 8, 10
    // Two chains' C seeds: odd at 0..47, even at 48..95.  Phase 1/2's seed is %r11676, which is
    // zero, so both start zeroed.
    uint32_t cmat[96] = {0};
    uint32_t as[24];
#if D4R_RRSWIN_DEBUG_STAGE == -1
    // Stage "a": the 24 A registers the cut hands over plus the C seed, in the oracle's order.
    // This pins the prologue handoff independently of the body's arithmetic.  28 registers, not
    // 25: the lane stride must be a multiple of 16 bytes so the oracle's st.global.v4.u32 stays
    // aligned (see the recipe's _stage_table).  The last four repeat the C seed, matching it.
    {
        uint32_t regs[28];
        for (int j = 0; j < 24; ++j) {
            regs[j] = as1[j];
        }
        for (int j = 24; j < 28; ++j) {
            regs[j] = c0;
        }
        debug_store(prm, regs, 28);
    }
#endif

    // ---- the stem: five shift groups, two chains of five accumulations -----
    for (int g = 0; g < 5; ++g) {
        stem_group(smem, g, lane, as);
        // Shift group 0 comes from the 24 registers the cut hands over: phase 1's 24 mma consume
        // exactly the fragments at arena byte offsets 0, 2560, 5120, 7680, 10240, 12800, in that
        // order (VERIFIED against the mma list at lines 11883-12044, which names %r11754, %r11774,
        // %r11794, %r11814, %r11834 and %r11854 as its six A operands).  Groups 1-4 are not
        // live-out of the cut, so they are read from the arena, where the prologue staged them per
        // wave (base `tid.z*2560`, s11638) -- see stem_group.
        if (g == 0) {
            for (int j = 0; j < 24; ++j) {
                as[j] = as1[j];
            }
        }
        gemm(as, wv, 320 + 1024 * g, nullptr, nullptr, cmat, 6, t, dodd);
#if D4R_RRSWIN_DEBUG_STAGE
        if (D4R_RRSWIN_DEBUG_STAGE == 2 * g + 1) {
            uint32_t regs[48];
            stage_regs(dodd, regs);
            debug_store(prm, regs, 48);
        }
#endif
        gemm(as, wv, 5440 + 1024 * g, nullptr, nullptr, cmat + 48, 6, t, deven);
#if D4R_RRSWIN_DEBUG_STAGE
        if (D4R_RRSWIN_DEBUG_STAGE == 2 * g + 2) {
            uint32_t regs[48];
            stage_regs(deven, regs);
            debug_store(prm, regs, 48);
        }
#endif
        // the next group accumulates: its C operand is this pair's D, per (m-tile, row, n-tile)
        stem_seed(dodd, deven, cmat);
    }

    // ---- phase 11, the score, and E11's softmax ----------------------------
    uint32_t d11[4][2][12];
    score(smem, dodd, wv, lane, t, (int)(lane >> 2), d11);
#if D4R_RRSWIN_DEBUG_STAGE
    if (D4R_RRSWIN_DEBUG_STAGE == 11) {
        uint32_t regs[96];
        score_regs(d11, regs);
        debug_store(prm, regs, 96);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE
    if (D4R_RRSWIN_DEBUG_STAGE == 12) {
        // E11's output: the softmax probabilities as phase 12 packs them into its A fragments
        uint32_t regs[96];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int ks = 0; ks < 3; ++ks) {
                for (int rh = 0; rh < 2; ++rh) {
                    uint32_t prob[12];
                    row_probs(d11, mi, rh, prob);
                    regs[i++] = e4m3x2(prob[4 * ks]) | (e4m3x2(prob[4 * ks + 2]) << 16);
                    regs[i++] = e4m3x2(prob[4 * ks + 1]) | (e4m3x2(prob[4 * ks + 3]) << 16);
                }
            }
        }
        debug_store(prm, regs, 96);
    }
#endif

    (void)c0;
    // ---- handing the five accumulators back --------------------------------
    lds_u32* const out = smem + 5u * lane_id();
    out[0] = *(const uint32_t*)prm;                 // %r29719, param_0+0
    // %r31372 / %r31382 are phase 62's D pair; phases 11-62 are not implemented yet, so the two
    // slots carry lane-distinct values that keep the read-back path observable without producing
    // wild magnitudes.
    out[1] = (uint32_t)lane_id() + 32u;
    out[2] = (uint32_t)lane_id() + 64u;
    out[3] = (uint32_t)z;                          // %r31608, %tid.z
    out[4] = scratch;                              // %r31615, the shared arena's address
}