// Native replacement for the Ray Reconstruction denoiser's third Swin encoder,
// cuda_dldn_engine_swin_enc2_kernel (nvngx_dlssd.dll 310.7).
//
// ---------------------------------------------------------------------------
// STATE: the mirror cut is in place and phases 1-7 are implemented and bit-exact
// against the debug oracle (see "Ground truth" below): each stage's payload region
// compares byte for byte, 0 differing f16 halves out of the whole payload.  Phases
// 8-38 (the attention output projection, the MLP band and the patch merge) are not
// ported yet, so the five accumulators the epilogue reads are still filled with
// lane-distinct placeholders; the read-back path is exercised, the layer's output is
// not yet correct.
//
// The cut hands over phase 1's staged A fragments, its B fragments, its zero C seed,
// the weight image, the parameter block and the base of the entry's own `.shared`
// tile, which the body reuses: the A fragments of phases 3 and 5 are read back out
// of it, phase 7's keys are published through it (32 lanes x 24 words), and the five
// live-out accumulators cross the cut through it.
//
// ---------------------------------------------------------------------------
// The kernel is one warp per 8x8 token patch, four waves deep in z: grid (161, 93, 1),
// block (32, 1, 4), parameter block 144 bytes.  The PTX this file replaces is
// ~/.cache/d4r-rr-corpus/0015-PREPASS_ENTRYPOINT_NAME.ptx, statements s4621-s13876
// (file lines 7669-33949): the 630 mma of 38 phases.
//
// kernels/rr/swin_enc2_recipe.py owns the splice (kernels/tex/make_ptx.py loads it for
// cuda_dldn_engine_swin_enc2_kernel), so NVIDIA's texture prologue (s1-s4620) and its
// surface epilogue (s13877-end, the staged softmax window, the 3x3 texture gather and
// the three surface writes) stay PTX and the region above becomes one call.uni.
//
// Linked into ZLUDA's compile of that PTX (D4R_ZLUDA_EXTRA_BC), so tex_common.h's
// helpers and ZLUDA's own device functions are available.  Compiled with -nogpuinc,
// so only plain clang builtins and attributes are used.
#include "../tex/tex_common.h"

// ---------------------------------------------------------------------------
// Parameter block.  Fields are addressed by their byte offset in the PTX, so the raw
// image is kept and read through a typed accessor.
struct DldnSwinEnc2Params
{
    uint8_t raw[144];
};
static_assert(sizeof(DldnSwinEnc2Params) == 144, "parameter block");

enum : int
{
    kExtents = 0,   // (ex, ey) v2.u32
    kWeights = 40,   // NVIDIA's prepared weight image (arg040), read-only
    kPlaneArg = 48,  // plane arena pointer (arg048), read by the debug store only
    kOutArg = 56,    // phase 37's output buffer (arg056)
    kOrigins = 80,   // patch origins v2.u16 (rs126, rs127)
};

#define DEVF __attribute__((device)) static inline
typedef float lds_f32 __attribute__((address_space(3)));
typedef uint32_t lds_u32 __attribute__((address_space(3)));
typedef uint8_t lds_u8 __attribute__((address_space(3)));

// ---------------------------------------------------------------------------
// The implemented phases, as byte offsets into arg040.  Every offset is relative to
// the slab pointer Wz = W + 21504*tid.z + 16*laneid that the entry builds at s4613,
// which is what phases 1..7 index through (rrswin_enc2_epilogues.md, section 1.1):
//   P1  weights 192     P4  weights 4288
//   P2  weights 3264    P5  weights 2240
//   P3  weights 1216    P6  weights 5312
//   P7  no weights (A and B are packs of phase 5's D), C = the score bias at 6336
//
// e4m3 code -> f32, exact.  The mma's e4m3 operands reach the FMA path through ZLUDA's
// exact e4m3 -> f16 widening, so this decode is the same value.
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
// (inverted from the hardware fragment layout: a lane's 16-byte chunk holds the eight
// k = 8*h + 2t + (j&1) + 16*((j>>1)&1) of its own t, for the columns 16*tt + 8*f + g;
// the mma hands column n out of lane n, so the native side reads n = 8*nt + 2t + c
// straight from the image instead.)
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

DEVF uint32_t pair_of(half_t a, half_t b)
{
    return (uint32_t)__builtin_bit_cast(uint16_t, a)
         | ((uint32_t)__builtin_bit_cast(uint16_t, b) << 16);
}

// ---------------------------------------------------------------------------
// One GEMM: six m-tiles of 16 rows x four n-tiles of eight columns, K=32, f16 out.
//
// A lane's A registers hold, for each m-tile and each of its two rows, the eight e4m3
// values of k = 8*reg + 2t + (j&1) + 16*((j>>1)&1) (byte j = 0..3 of register
// 2*reg + row); the four lanes that differ in t hold the four disjoint eighths of the
// 32 k's, so two butterfly steps broadcast the other three eighths in.  The B operand
// is read from the weight image (waddr) for the lane's own two columns of each n-tile.
// The dot product runs in f32 and is rounded to f16 once, which is what the translated
// mma does: e4m3 x e4m3 products are exact and 32 of them sum exactly in f32.
//
// `cprev` is the C operand as a [m-tile][row half][n-tile] array of packed pairs --
// phase 1 and phase 2 start from zero (%r8606), phases 3..6 accumulate onto phase 1's,
// 2's, 3's and 4's accumulators.  d is [m-tile][row half][n-tile].
DEVF void gemm_k32(const uint32_t* as, const uint8_t* w, int base, const uint32_t* cprev,
                   int nmi, int t, uint32_t d[6][2][4])
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
            float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
            for (int reg = 0; reg < 2; ++reg) {
                for (int row = 0; row < 2; ++row) {
                    // the register's value on this lane and on lanes t^1, t^2, t^3
                    uint32_t p[4];
                    p[0] = as[4 * mi + 2 * reg + row];
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
                const uint32_t seed = cprev != nullptr ? cprev[(2 * mi + row) * 4 + nt] : 0u;
                d[mi][row][nt] = pair_of((half_t)(acc[row][0] + h2f(seed)),
                                         (half_t)(acc[row][1] + h2f(seed >> 16)));
            }
        }
    }
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

// ---------------------------------------------------------------------------
// A [6][2][4] block of packed f16 pairs, and the e4m3 words E6 packs out of it.
//
// A D register d[mi][row][nt] holds row 16*mi + 8*row + g, columns 8*nt + 2t and
// 8*nt + 2t+1.  The two e4m3 words the PTX builds for that row are
//   w[0] = {pack(columns 2t, 2t+1), pack(columns 16+2t, 16+2t+1)}
//   w[1] = {pack(columns 8+2t, 8+2t+1), pack(columns 24+2t, 24+2t+1)}
// i.e. exactly the hardware's A register for the k-halves 0 and 1, so w[h] is both an
// A register (row r) and a B fragment (row r) -- which is what E6 exploits: phase 7's
// A fragment m is {w0(16m+g), w0(16m+g+8), w1(16m+g), w1(16m+g+8)} and its B fragment
// n-tile is {w0(8n+g), w1(8n+g)} (VERIFIED against the oracle's e6a/e6b stages).
DEVF half_t dcell(const uint32_t d[6][2][4], int row, int col)
{
    const uint32_t p = d[row >> 4][(row >> 3) & 1][col >> 3];
    return (col & 1) ? half_hi(p) : half_lo(p);
}

DEVF void row_words(const uint32_t d[6][2][4], int r, uint32_t w[2])
{
    w[0] = e4m3x2(pair_of(dcell(d, r, 0), dcell(d, r, 1)))
         | (e4m3x2(pair_of(dcell(d, r, 16), dcell(d, r, 17))) << 16);
    w[1] = e4m3x2(pair_of(dcell(d, r, 8), dcell(d, r, 9)))
         | (e4m3x2(pair_of(dcell(d, r, 24), dcell(d, r, 25))) << 16);
}

// ---------------------------------------------------------------------------
// Ground truth for a stage comes from kernels/rr/swin_enc2_recipe.py: with
// D4R_RRSWIN_DEBUG=<n> the same splice keeps NVIDIA's whole body and stores that
// stage's registers into the plane arena at +4,000,000, which the kernel never
// otherwise writes, and the launch dump then holds the translated values for a
// block/lane.  Built with -DD4R_RRSWIN_DEBUG_STAGE=<n> (kernels/build.sh rrswin,
// D4R_TEX_CFLAGS), this body stores the same registers of its own phase n in the same
// order, so the two dumps compare byte for byte.  The base is 4,000,000 rather than the
// family default of 46,000,000 because enc2's arg048 sits at allocation offset
// 30,146,560 of a 65,945,600-byte allocation, which the bridge dumps under arg008.
#ifndef D4R_RRSWIN_DEBUG_STAGE
#define D4R_RRSWIN_DEBUG_STAGE 0
#endif

DEVF void debug_store(const uint8_t* prm, const uint32_t* regs, int n)
{
    const uint32_t block = __builtin_amdgcn_workgroup_id_x()
                         + 41u * __builtin_amdgcn_workgroup_id_y();
    if (block >= 64u) {
        return;   // the oracle's per-block slots only cover the first 64 blocks
    }
    // the block is four waves deep in z and every plane reads a different weight slab
    // (the 21504-byte stride), so only plane 0 may write the slot
    if (__builtin_amdgcn_workitem_id_z() != 0u) {
        return;
    }
    const uint64_t arena = *(const uint64_t*)(prm + kPlaneArg);
    // the oracle's slot, the recipe's _debug_block: 32 lanes x n x 4 bytes, at least
    // 8192, rounded up to a power of two (its splice forms block*slot with a shift)
    const uint64_t want = (uint64_t)n * 128ull > 8192ull ? (uint64_t)n * 128ull : 8192ull;
    uint64_t slot = 1;
    while (slot < want) {
        slot <<= 1;
    }
    uint32_t* dst = (uint32_t*)(arena + 4000000ull + (uint64_t)block * slot
                                + (uint64_t)lane_id() * (uint64_t)n * 4ull);
    for (int i = 0; i < n; ++i) {
        dst[i] = regs[i];
    }
}

// stage registers in the oracle's order: 24 mma, m-tile major, n-tile minor, each as
// the (row g, row g+8) pair the mma writes
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
// Mirror cut.  Phase 1's six staged A fragments arrive in this order (the first mma's
// operand order, .f16.e4m3 fragments of 4 b32 each):
//   f0 %r16120 %r16121 %r16122 %r16123   f3 %r16156 %r16157 %r16158 %r16159
//   f1 %r16132 %r16133 %r16134 %r16135   f4 %r16168 %r16169 %r16170 %r16171
//   f2 %r16144 %r16145 %r16146 %r16147   f5 %r16180 %r16181 %r16182 %r16183
// then phase 1's four B fragments (%r3251 %r3252 %r3253 %r3254 %r3255 %r3256 %r3257
// %r3258, read from Wz+192 and Wz+704), its zero C seed (%r8606), the weight image and
// the parameter block.
//
// The entry's own `.shared` tile holds the eighteen A fragments at
// `smem + 512*(3*m-tile + slice) + 16*laneid`, three K=32 slices of the 96x96 input
// patch that the six z-planes wrote cooperatively (s4520-s4590); the body reads all
// three back rather than taking slices 1 and 2 over the cut.  It is otherwise dead
// inside the region, and E38 re-stages its own window into it after the cut.
extern "C" __attribute__((device)) void d4r_swin_enc2(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5,
    uint32_t a6, uint32_t a7, uint32_t a8, uint32_t a9, uint32_t a10, uint32_t a11,
    uint32_t a12, uint32_t a13, uint32_t a14, uint32_t a15, uint32_t a16, uint32_t a17,
    uint32_t a18, uint32_t a19, uint32_t a20, uint32_t a21, uint32_t a22, uint32_t a23,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t b4, uint32_t b5,
    uint32_t b6, uint32_t b7, uint32_t c0,
    const uint8_t* w, const uint8_t* prm, uint32_t scratch)
{
    const uint32_t as1[24] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                              a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23};
    (void)c0;
    const uint32_t lane = lane_id();
    const int t = (int)(lane & 3);   // the lane's two columns of each n-tile
    const int g = (int)(lane >> 2);  // ... and of its sixteen rows
    const uint32_t tz = __builtin_amdgcn_workitem_id_z();
    // Wz: the slab pointer phases 1..7 index through (s4589-s4590, s4613-s4615)
    const uint8_t* const wz = w + 21504u * tz + 16u * lane;
    lds_u32* const sm = (lds_u32*)(uintptr_t)scratch;

    // ---- the three K=32 slices of the staged 96x96 input patch --------------
    // as[s] is slice s's six A fragments, m-tile major, in the order the entry reads
    // them: fragment m of slice s at 512*(3*m + s) + 16*laneid.
    uint32_t as[3][24];
#pragma unroll
    for (int q = 0; q < 18; ++q) {
        const int s = q % 3;
        const int mi = q / 3;
        as[s][4 * mi + 0] = sm[128 * q + 4 * lane + 0];
        as[s][4 * mi + 1] = sm[128 * q + 4 * lane + 1];
        as[s][4 * mi + 2] = sm[128 * q + 4 * lane + 2];
        as[s][4 * mi + 3] = sm[128 * q + 4 * lane + 3];
    }
#if D4R_RRSWIN_DEBUG_STAGE == 90
    {
        uint32_t regs[72];
#pragma unroll
        for (int mi = 0; mi < 6; ++mi) {
            for (int s = 0; s < 3; ++s) {
                regs[12 * mi + 4 * s + 0] = as[s][4 * mi + 0];
                regs[12 * mi + 4 * s + 1] = as[s][4 * mi + 1];
                regs[12 * mi + 4 * s + 2] = as[s][4 * mi + 2];
                regs[12 * mi + 4 * s + 3] = as[s][4 * mi + 3];
            }
        }
        debug_store(prm, regs, 72);
    }
#endif

    // ---- phases 1-6: the 96x96 -> 96x32 residual projection ------------------
    // K=32 and N=32 throughout: two K-slices are paired per input slice, each phase
    // accumulates onto the phase two before it that shares its A fragment (s4621-s4799).
    uint32_t zero[6][2][4] = {};
    uint32_t d1[6][2][4], d2[6][2][4], d3[6][2][4], d4[6][2][4], d5[6][2][4], d6[6][2][4];
    gemm_k32(as1, wz, 192, &zero[0][0][0], 6, t, d1);
#if D4R_RRSWIN_DEBUG_STAGE == 1
    {
        uint32_t regs[48];
        stage_regs(d1, regs);
        debug_store(prm, regs, 48);
    }
#endif
    gemm_k32(&as[0][0], wz, 3264, &zero[0][0][0], 6, t, d2);
#if D4R_RRSWIN_DEBUG_STAGE == 2
    {
        uint32_t regs[48];
        stage_regs(d2, regs);
        debug_store(prm, regs, 48);
    }
#endif
    gemm_k32(&as[1][0], wz, 1216, &d1[0][0][0], 6, t, d3);
#if D4R_RRSWIN_DEBUG_STAGE == 3
    {
        uint32_t regs[48];
        stage_regs(d3, regs);
        debug_store(prm, regs, 48);
    }
#endif
    gemm_k32(&as[1][0], wz, 4288, &d2[0][0][0], 6, t, d4);
#if D4R_RRSWIN_DEBUG_STAGE == 4
    {
        uint32_t regs[48];
        stage_regs(d4, regs);
        debug_store(prm, regs, 48);
    }
#endif
    gemm_k32(&as[2][0], wz, 2240, &d3[0][0][0], 6, t, d5);
#if D4R_RRSWIN_DEBUG_STAGE == 5
    {
        uint32_t regs[48];
        stage_regs(d5, regs);
        debug_store(prm, regs, 48);
    }
#endif
    gemm_k32(&as[2][0], wz, 5312, &d4[0][0][0], 6, t, d6);
#if D4R_RRSWIN_DEBUG_STAGE == 6
    {
        uint32_t regs[48];
        stage_regs(d6, regs);
        debug_store(prm, regs, 48);
    }
#endif

    // ---- E6(b) and phase 7: the 64x96 attention score -----------------------
    // Both operands of the score GEMM are packs of phase 5's D: the four A fragments
    // are m-tiles 0..3 (rows 0..63, the queries) and the twelve B fragments are its
    // twelve n-tiles read as rows (rows 0..95, the keys) -- the score multiplies one
    // projection by its own transpose.  The A side needs only the lane's own rows, but
    // the mma hands column n out of lane n, so the B fragments go through the shared
    // tile: 32 lanes x 24 words at 1024, the first 3 KB of a tile nothing else in the
    // region touches.
    uint32_t as7[16];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        uint32_t w0[2], w8[2];
        row_words(d5, 16 * mi + g, w0);
        row_words(d5, 16 * mi + g + 8, w8);
        as7[4 * mi + 0] = w0[0];
        as7[4 * mi + 1] = w8[0];
        as7[4 * mi + 2] = w0[1];
        as7[4 * mi + 3] = w8[1];
    }
    uint32_t bf[12][2];
#pragma unroll
    for (int nt = 0; nt < 12; ++nt) {
        row_words(d5, 8 * nt + g, bf[nt]);
    }
    {
        lds_u32* const keys = sm + 256;          // 1024 bytes, 24 words per lane
#pragma unroll
        for (int nt = 0; nt < 12; ++nt) {
            keys[24 * lane + 2 * nt + 0] = bf[nt][0];
            keys[24 * lane + 2 * nt + 1] = bf[nt][1];
        }
    }
    __builtin_amdgcn_wave_barrier();
#if D4R_RRSWIN_DEBUG_STAGE == 92
    {
        uint32_t regs[24];
#pragma unroll
        for (int nt = 0; nt < 12; ++nt) {
            regs[2 * nt + 0] = bf[nt][0];
            regs[2 * nt + 1] = bf[nt][1];
        }
        debug_store(prm, regs, 24);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 91
    {
        debug_store(prm, as7, 16);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 93
    {
        // the 96 C seeds: phase 7's additive 64x96 f16 score bias, one value per
        // score element, 512 bytes per output tile, tile 4*(nt>>1) + m-tile
        uint32_t regs[96];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 12; ++nt) {
                const uint8_t* p = wz + 6336 + 512u * (4u * (uint32_t)(nt >> 1) + (uint32_t)mi)
                                        + 16u * lane + 8u * (uint32_t)(nt & 1);
                regs[2 * (mi * 12 + nt) + 0] = *(const uint32_t*)p;
                regs[2 * (mi * 12 + nt) + 1] = *(const uint32_t*)(p + 4);
            }
        }
        debug_store(prm, regs, 96);
    }
#endif
    uint32_t d7[4][2][12];
    {
        const lds_u32* const keys = sm + 256;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            for (int rh = 0; rh < 2; ++rh) {
                uint32_t aw[2], a[2][4];
                row_words(d5, 16 * mi + 8 * rh + g, aw);
                bfly2x4(aw, a);
                for (int nt = 0; nt < 12; ++nt) {
                    const uint32_t bias = *(const uint32_t*)(wz + 6336
                        + 512u * (4u * (uint32_t)(nt >> 1) + (uint32_t)mi)
                        + 16u * lane + 8u * (uint32_t)(nt & 1));
                    float s[2] = {0.0f, 0.0f};
                    for (int c = 0; c < 2; ++c) {
                        // the fragment of n = 8*nt + 2t + c lives in the four lanes of
                        // row group 2t+c; A's byte j on lane t^m pairs with B's byte j
                        // on lane 2t+c, t^m
                        uint32_t kb[2][4];
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
#pragma unroll
                            for (int m = 0; m < 4; ++m) {
                                kb[i][m] = keys[24 * (4 * (2 * t + c) + (t ^ m)) + 2 * nt + i];
                            }
                        }
                        s[c] = dot32(a, kb);
                    }
                    d7[mi][rh][nt] = pair_of((half_t)(s[0] + h2f(bias)),
                                            (half_t)(s[1] + h2f(bias >> 16)));
                }
            }
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 7
    {
        uint32_t regs[96];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 12; ++nt) {
                regs[i++] = d7[mi][0][nt];
                regs[i++] = d7[mi][1][nt];
            }
        }
        debug_store(prm, regs, 96);
    }
#endif

    // ---- handing the accumulators back ------------------------------------
    // %r16472 is param_0[0] (ld.param.v2.u32 at s13138) and %r17746 = 16*tid.z (s13139),
    // both computed inside the region; %r17758 is &smem (s13852); %r17496/%r17506 are
    // phase 38's accumulator pairs.  Phases 8-38 are not ported yet, so the two mma
    // pairs are filled with lane-distinct values: that keeps the read-back path
    // observable (E38's surface output changes from the unported baseline) without
    // producing wild magnitudes.
    __attribute__((address_space(3))) uint32_t* out =
        ((__attribute__((address_space(3))) uint32_t*)(uintptr_t)scratch)
        + 5 * (uint32_t)lane_id();
    out[0] = *(const uint32_t*)prm;              // ex
    out[1] = lane + 32u;                         // %r17496
    out[2] = lane + 64u;                         // %r17506
    out[3] = tz << 5;                            // %r17746 = 16*tid.z
    out[4] = scratch;                            // %r17758 = &smem
}
