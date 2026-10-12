// Native replacement for the Ray Reconstruction denoiser's first Swin encoder,
// cuda_dldn_engine_swin_enc0_kernel (nvngx_dlssd.dll 310.7).
//
// ---------------------------------------------------------------------------
// STATE: every phase of the kernel is implemented, in the PTX's own order.  Each one is
// checked against the debug oracle (see "Ground truth" below) with its own oracle capture,
// and a stage counts as validated only when its whole payload region matches the
// translated kernel byte for byte.  Phase 1's GEMM, its relu requantisation
// into phase 2's A operand, phase 2's GEMM, the LayerNorm tree (the sum-of-squares
// butterfly, the rsqrt and the activation gains) and the fused Q/V projection of phases 3
// and 4; the first attention window in phases 5 to 7 (the 64x96 score with its position
// bias, the softmax and P*V, the output projection with the residual and the per-column
// table); the second window in phases 8 to 12 (two more projections off the same normed
// tile, its own score table at weight bytes 23744, its P*V, and the output projection that
// folds phase 7's accumulators in as its C operand); the MLP in phases 13 to 20 (E12's RMS
// norm, four parallel branches that share its tile and each with the clamped cubic
// activation, their running residual); and the merge in phases 21 and 22 (E20's plane-arena
// scatter and eight-pair lane merge, E21's surface scatter, phase 22's projection).
//
// The epilogue E22 stays NVIDIA's: it lives outside the cut and reads the eighteen values
// this body hands back (the parameter block's first word, %tid.x and phase 22's eight mma
// D pairs).
//
// The whole body costs about 6 ms per launch on an RX 9070 XT (gfx1201) with its results
// kept alive by a debug store; as served, where the optimiser drops them again, it is the
// kernel's dominant cost.
//
// ---------------------------------------------------------------------------
// The kernel is one warp per 8x8 token patch: grid (161, 93, 1), block (32, 1, 1),
// 14973 blocks, parameter block 1032 bytes. The PTX this file replaces is
// kernels/extracted/rr/0013-PREPASS_ENTRYPOINT_NAME.ptx.
//
// kernels/tex/make_ptx.py's recipe (D4R_RRSWIN_CUT=1) keeps NVIDIA's texture
// prologue (lines 1-3963) and surface epilogue (lines 44528-end) and replaces the
// 536 mma of the 22 phases with the call below. Without that variable the recipe
// replaces the whole body instead (a served stub).
//
// Linked into ZLUDA's compile of that PTX (D4R_ZLUDA_EXTRA_BC), so tex_common.h's
// helpers and ZLUDA's own device functions are available. Compiled with -nogpuinc,
// so only plain clang builtins and attributes are used.
#include "../tex/tex_common.h"
// The staged-image engine: the body's sixteen plain GEMM phases read staged images and prepared
// slots through it, so the header is needed unconditionally (the phase-8 pilot below keeps its
// own D4R_RRSWIN_NATIVE_GEMM guard).
#include "rrswin_engine.h"

// ---------------------------------------------------------------------------
// Parameter block. Fields are addressed by their byte offset in the PTX, so the
// raw image is kept and read through a typed accessor.
struct DldnSwinEnc0Params
{
    uint8_t raw[1032];
};
static_assert(sizeof(DldnSwinEnc0Params) == 1032, "parameter block");

enum : int
{
    kOutPlane = 8,   // plane arena the encoder writes (arg048)
    kWeights = 40,   // NVIDIA's prepared weight image (arg040), read-only
    kSurfaceA = 112, // 1280x736x4 RGBA16F outputs (arg112/arg120)
    kSurfaceB = 120,
    kSurfaceHalf = 128, // 640x368x4 RGBA16F (arg128)
    kSurfaceIn = 304,   // 1280x720x4 RGBA16F, read-only (arg304)
    kPlaneArg = 48,     // plane arena pointer, as make_ptx.py's debug splice reads it
};

#define DEVF __attribute__((device)) static inline
typedef float lds_f32 __attribute__((address_space(3)));
using rrswin::lds_u32;   // the engine header's AS3 word (the body defines none of its own)
typedef uint8_t lds_u8 __attribute__((address_space(3)));

#ifdef D4R_RRSWIN_NATIVE_GEMM
// A token tile (512 bytes), two prepared weight tiles (1024 bytes), and their
// f16 result (1024 bytes) fit within the existing 3200-byte shared scratch.
DEVF void gemm_native_unbiased(const uint32_t* as, const uint32_t* bs,
                               int nmi, uint32_t scratch, lds_u32* out)
{
    lds_u8* const shared = (lds_u8*)(uintptr_t)scratch;
    lds_u8* const image = shared;
    lds_u8* const weights = shared + 512;
    __attribute__((address_space(3))) half_t* const result =
        (__attribute__((address_space(3))) half_t*)(shared + 1536);
    const uint32_t lane = lane_id();
#pragma unroll
    for (int tile = 0; tile < 2; ++tile)
#pragma unroll
        for (int word = 0; word < 4; ++word)
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const uint32_t n = 16 * tile + (lane >> 2) + 8 * (word >> 1);
                const uint32_t k = 2 * (lane & 3) + (j & 1) + 16 * (j >> 1) + 8 * (word & 1);
                weights[16 * (32 * (k >> 4) + n) + (k & 15)] =
                    (uint8_t)(bs[4 * tile + word] >> (8 * j));
            }
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int mi = 0; mi < nmi; ++mi)
    {
        rrswin::swin_stage_fragment(image, 32, as + 4 * mi, 1);
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
        {
            f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
            c = rrswin::swin_gemm_tile(c, weights, 16 * mt, 32, image, 32, 0);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                result[32 * (lane & 15) + 16 * mt + i + 8 * (lane >> 4)] = (half_t)c[i];
        }
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int rh = 0; rh < 2; ++rh)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
            {
                const int at = 32 * ((lane >> 2) + 8 * rh) + 8 * nt + 2 * (lane & 3);
                out[8 * mi + 2 * nt + rh] = rrswin::pair_of(result[at], result[at + 1]);
            }
        __builtin_amdgcn_wave_barrier();
    }
}
#endif

#ifdef D4R_RRSWIN_PHASE8
extern "C" __attribute__((device)) void d4r_swin_enc0_p8(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7, uint32_t scratch)
{
    const uint32_t as[4] = {a0, a1, a2, a3};
    const uint32_t bs[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    // All WMMA reads finish before the lane-major return overwrites operand
    // storage. The result matrix at +1536 remains disjoint from this return.
    lds_u32* const out = (lds_u32*)(uintptr_t)(scratch + 32 * lane_id());
    gemm_native_unbiased(as, bs, 1, scratch, out);
}
#endif

// ---------------------------------------------------------------------------
// The implemented phases, as byte offsets into arg040 (from the PTX's own loads;
// K=32 and N=32, so each reads two 512-byte weight tiles):
//   P1  weights 4096, bias seeds handed in as b0..b3 (loaded from +5120..+5184)
//   P2  weights 5184, f16 bias at 6208..6272, activation gains at 6272..6336
//   P3  weights 6336, no bias (the fused Q projection)
//   P4  weights 7360, no bias (V)

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

// ZLUDA's relu_keep_nan_halves (the phase-2 staging's mode 1) lives in the engine header
// as rrswin::swin_relu_pair; this body has no local copy.

// ---------------------------------------------------------------------------
// One GEMM: 96 rows (six m-tiles of 16) x 32 columns, K=32, f16 output.
//
// A lane's A registers hold, for each m-tile and each of its two rows, the eight
// e4m3 values of k = 2t + (j&1) + 16*((j>>1)&1) (byte j = 0..3) and of k + 8; the
// four lanes that differ in t hold the four disjoint eighths of the 32 k's, so two
// butterfly steps broadcast the other three eighths in. The B operand is read from
// the weight image (waddr) for the lane's own two columns of each n-tile. The dot
// product runs in f32 and is rounded to f16 once, which is what the translated mma
// does: e4m3 x e4m3 products are exact and 32 of them sum exactly in f32, so this
// reproduces the translated kernel bit for bit.
//
// Exactly one of the three C operands carries the bias: `wbias` (a byte pointer at the phase's
// bias offset, 16 bytes per n-tile), `seeds` (four already lane-indexed packed pairs, phase 1),
// or `cmat` (32 lane-indexed pairs indexed [4*m-tile + row][n-tile], phase 7, whose seed is
// or `cmat` (32 lane-indexed pairs indexed [4*m-tile + row][n-tile], phases 7 and 12-20,
// whose seed is the residual plus the per-column table and so differs per row).
// d is [m-tile][row half][n-tile], each entry the packed f16 pair for columns 2t, 2t+1;
// `nnt` is the number of n-tiles (four everywhere but phase 22's two).
DEVF void gemm(const uint32_t* as, const uint8_t* w, int base, const uint8_t* wbias,
               const uint32_t* seeds, const uint32_t* cmat, int nmi, int t,
               uint32_t d[6][2][4], int nnt = 4)
{
    for (int nt = 0; nt < nnt; ++nt) {
        float bv[2][32];
        for (int c = 0; c < 2; ++c) {
            const int n = 8 * nt + 2 * t + c;
            for (int k = 0; k < 32; ++k) {
                bv[c][k] = e4m3_to_f32(w[waddr(base, k, n)]);
            }
        }
        for (int mi = 0; mi < nmi; ++mi) {
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
                const uint32_t bias_pair = (cmat != nullptr) ? cmat[(4 * mi + row) * 4 + nt] : fixed;
                d[mi][row][nt] = pair_of((half_t)(acc[row][0] + h2f(bias_pair)),
                                         (half_t)(acc[row][1] + h2f(bias_pair >> 16)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phases 5-7: the window's self-attention and the output projection.  Both operands of
// phase 5 come from phase 3's accumulators (rows 0..63 are the queries, rows 0..95 the keys
// -- the 8x8 token patch plus its 32-row halo), so A and B are built exactly the way the
// PTX's 24 packs build them: word 0 of a row packs n-tiles 0 and 2, word 1 packs 1 and 3,
// and the four lanes t=0..3 of a group hold the four disjoint eighths of the 32 k's, which
// the two butterfly steps bring together.  Every e4m3 product is exact and a 32-term k
// slice sums exactly in f32, so the dot product is plain f32 accumulation rounded to f16
// once -- what the translated mma does.

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

// the f16 at (row, col) of a 96x32 block of packed pairs, col = 8*n-tile + 2t + c
DEVF half_t dcell(const uint32_t d[6][2][4], int row, int col)
{
    const uint32_t p = d[row >> 4][(row >> 3) & 1][col >> 3];
    return (col & 1) ? half_hi(p) : half_lo(p);
}

// the lane's two e4m3-packed words for row r: word 0 packs n-tiles 0 and 2, word 1 tiles 1 and 3
DEVF void row_words(const uint32_t d[6][2][4], int r, uint32_t w[2])
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

// E5(a), one element: the scale, the clamp, the two clamped-cubic fma pairs and the trick
// that moves the mantissa into the exponent field ((u << 5) & 0x7FE07FE0 over the pair).
DEVF uint32_t expval_pair(uint32_t score)
{
    hv2 m = hv2_of(score) * __builtin_bit_cast(hv2, pair_of(hf16(0x2466), hf16(0x2466)));
    m = __builtin_elementwise_max(m, __builtin_bit_cast(hv2, pair_of(hf16(0xb873), hf16(0xb873))));
    m = __builtin_elementwise_min(m, __builtin_bit_cast(hv2, pair_of(hf16(0x3873), hf16(0x3873))));
    const hv2 t = __builtin_elementwise_fma(m, -m,
                        __builtin_bit_cast(hv2, pair_of(hf16(0x3b6b), hf16(0x3b6b))));
    const uint32_t u = u32_of(__builtin_elementwise_fma(m, t,
                        __builtin_bit_cast(hv2, pair_of(hf16(0x3d80), hf16(0x3d80)))));
    return (u << 5) & 0x7fe07fe0u;
}

// E5(a)-(e), one row: the twelve exponent pairs, the in-lane sum tree
// (((e0+e2)+(e4+e6))+(e8+e10)) + (((e1+e3)+(e5+e7))+(e9+e11)), the two butterfly steps that
// finish the row's 96 columns, the two halves' add, the reciprocal and the twelve products.
DEVF void row_probs(const uint32_t d5[4][2][12], int mi, int rh, uint32_t prob[12])
{
    hv2 ev[6], eo[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        ev[i] = hv2_of(expval_pair(d5[mi][rh][2 * i]));
        eo[i] = hv2_of(expval_pair(d5[mi][rh][2 * i + 1]));
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

// ---------------------------------------------------------------------------
// The two GEMMs the two attention windows share: the 64x96 score (phases 5 and 10) and
// the softmax plus P·V that follow it (phases 6 and 11).  The second window repeats the
// first one's code with another weight base and another source accumulator, so each is
// one function.

// The score block.  A and B are both the one 96x32 accumulator `dq`: the lane's rows
// 16*mi + g + 8*row are the queries (the 8x8 token patch, 64 rows) and its rows
// 16*(nb>>1) + 8*(nb&1) + g the keys (the patch plus a 32-row halo, 96 rows), which is
// how the PTX's 24 packs build the two operands out of the same D.  Every lane publishes
// its two key words per n-tile into `slot` (twelve n-tiles x 64 words = 3072 bytes, inside
// the dead input_tensor the cut hands over) and then forms the dots its own D registers
// hold: query row 16*mi + g + 8*row (the A side keeps the lane's own row group g) against
// key rows 16*(nb>>1) + 8*(nb&1) + 2t + c, published by the lanes 4*(2t + c) + 0..3.  The
// C operand is the phase's 64x96 f16 position-bias table at `bias` (weight bytes 8384 or
// 23744): 24 tiles of 512 bytes and 16 per lane, the seed of (m-tile, n-tile) in tile
// mi + 4*(nb>>1) at word 2*(nb&1) + row of the lane's eight f16 (VERIFIED against the
// oracle's own C registers, debug stage 11 for phase 5 and 204 for phase 10).
// Every e4m3 product is exact and a 32-term k slice sums exactly in f32, so the dot
// product is plain f32 accumulation rounded to f16 once -- what the translated mma does.
//
// `cap`, when the debug stage asks for the published key words (203) or the C and B
// operand registers (204), receives them in the oracle's order.
DEVF void score_gemm(const uint32_t dq[6][2][4], const uint8_t* w, int bias, int t, int g,
                     int lane, lds_u32* slot, uint32_t* cap, uint32_t d[4][2][12])
{
#pragma unroll
    for (int nb = 0; nb < 12; ++nb) {
        uint32_t kw[2];
        row_words(dq, 16 * (nb >> 1) + 8 * (nb & 1) + g, kw);
        slot[64 * nb + lane] = kw[0];
        slot[64 * nb + 32 + lane] = kw[1];
    }
    __builtin_amdgcn_wave_barrier();
#if D4R_RRSWIN_DEBUG_STAGE == 203
    if (cap != nullptr) {
        for (int nb = 0; nb < 12; ++nb) {
            cap[2 * nb] = slot[64 * nb + 4 * (2 * t)];
            cap[2 * nb + 1] = slot[64 * nb + 32 + 4 * (2 * t + 1)];
        }
    }
#endif
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#if D4R_RRSWIN_DEBUG_STAGE == 204
        if (cap != nullptr) {
            // the mma's A fragment: rows g and g+8 of m-tile mi, word 0 (k = 2t + (j&1) +
            // 16*((j>>1)&1), i.e. n-tiles 0 and 2) then word 1 (n-tiles 1 and 3)
            uint32_t q0[2], q1[2];
            row_words(dq, 16 * mi + g, q0);
            row_words(dq, 16 * mi + g + 8, q1);
            cap[4 * mi + 0] = q0[0];
            cap[4 * mi + 1] = q1[0];
            cap[4 * mi + 2] = q0[1];
            cap[4 * mi + 3] = q1[1];
        }
#endif
        for (int row = 0; row < 2; ++row) {
            uint32_t aw[2], a[2][4];
            row_words(dq, 16 * mi + g + 8 * row, aw);
            bfly2x4(aw, a);
            for (int nb = 0; nb < 12; ++nb) {
                float s[2];
                for (int c = 0; c < 2; ++c) {
                    uint32_t kb[2][4];
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        // The mma's eight n-index rows are the key rows
                        // 16*(nb>>1) + 8*(nb&1) + n, and the lane's two D columns c = 0, 1
                        // are n = 2*t + c: each half dots a *different* key row, not the two
                        // halves of one.  Its B words are the four lanes 4*(2*t + c) + (t^m)
                        // of that row's group (the same butterfly the A side walks, on the
                        // row group the D column selects).
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            kb[i][m] = slot[64 * nb + 32 * i + 4 * (2 * t + c) + (t ^ m)];
                        }
                    }
                    s[c] = dot32(a, kb);
                }
                const uint32_t b =
                    *(const uint32_t*)(w + bias + 512 * (mi + 4 * (nb >> 1))
                                       + 16 * lane + 8 * (nb & 1) + 4 * row);
                d[mi][row][nb] = pair_of((half_t)(s[0] + h2f(b)),
                                         (half_t)(s[1] + h2f(b >> 16)));
#if D4R_RRSWIN_DEBUG_STAGE == 204
                if (cap != nullptr) {
                    // the position bias, the 24 tiles of four words after the A and B
                    cap[40 + 4 * (mi + 4 * (nb >> 1)) + 2 * (nb & 1) + row] = b;
                }
#elif D4R_RRSWIN_DEBUG_STAGE == 211
                if (cap != nullptr) {
                    cap[4 * (mi + 4 * (nb >> 1)) + 2 * (nb & 1) + row] = b;
                }
#endif
            }
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 204
    if (cap != nullptr) {
        // the twelve B fragment pairs, as the mma reads them: the lane's two packed words
        // of the key row 16*(nb>>1) + 8*(nb&1) + g, which is what it published in `slot`
        for (int nb = 0; nb < 12; ++nb) {
            cap[16 + 2 * nb] = slot[64 * nb + lane];
            cap[16 + 2 * nb + 1] = slot[64 * nb + 32 + lane];
        }
    }
#elif D4R_RRSWIN_DEBUG_STAGE == 211
    if (cap != nullptr) {
        for (int nb = 0; nb < 12; ++nb) {
            cap[96 + 2 * nb] = slot[64 * nb + lane];
            cap[96 + 2 * nb + 1] = slot[64 * nb + 32 + lane];
        }
    }
#endif
}

// E5 and E10 with the P·V of phases 6 and 11.  row_probs() gives the twelve probability
// pairs of one row; k-step ks packs key columns {4ks, 4ks+2} -- the columns of prob[4ks]
// and prob[4ks+2], low half first -- into the fragment's first word and {4ks+1, 4ks+3}
// into its second.  The lane's A fragment is its *own* two rows (16*mi + g and
// 16*mi + g + 8), so the D row it accumulates is its own g; what it lacks is V.  The key
// rows 32*ks + kk live in the lanes whose g is kk % 8, and every lane needs all 96 of
// them, so V goes through the dead input_tensor as one e4m3 byte per element
// (96 x 32 = 3072 bytes) and each lane then walks the full k range.  The P byte for key
// column kk is the one the lane with t = (kk%8)>>1 packs: word (kk>>3)&1, byte
// 2*(kk>>4) + (kk&1) -- the two butterfly steps of bfly2x4 bring the other three eighths
// in.  Each k-step rounds to f16 before the next one adds to it, which is why the chain
// has three separate mma.
//
// `cap`, when the debug stage asks for the A fragments (201), receives the twelve packed
// word pairs in the oracle's order.
DEVF void pv_gemm(const uint32_t ds[4][2][12], const uint32_t dv[6][2][4], int t, int g,
                  lds_u8* vs, uint32_t* cap, uint32_t d[4][2][4])
{
    __builtin_amdgcn_wave_barrier();   // the score GEMM's reads of `slot` are done
#pragma unroll
    for (int mi4 = 0; mi4 < 6; ++mi4) {
        for (int rh4 = 0; rh4 < 2; ++rh4) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                for (int c = 0; c < 2; ++c) {
                    const int r = 16 * mi4 + 8 * rh4 + g;
                    vs[r * 32 + 8 * nt + 2 * t + c] =
                        (uint8_t)q8(dcell(dv, r, 8 * nt + 2 * t + c));
                }
            }
        }
    }
    __builtin_amdgcn_wave_barrier();
    const lds_u8* const vp = vs;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            uint32_t prob[12], aw[3][2];
            row_probs(ds, mi, rh, prob);
#pragma unroll
            for (int ks = 0; ks < 3; ++ks) {
                aw[ks][0] = e4m3x2(prob[4 * ks]) | (e4m3x2(prob[4 * ks + 2]) << 16);
                aw[ks][1] = e4m3x2(prob[4 * ks + 1]) | (e4m3x2(prob[4 * ks + 3]) << 16);
#if D4R_RRSWIN_DEBUG_STAGE == 201
                if (cap != nullptr) {
                    cap[4 * (mi * 3 + ks) + rh] = aw[ks][0];
                    cap[4 * (mi * 3 + ks) + 2 + rh] = aw[ks][1];
                }
#endif
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                half_t prev0 = hf16(0x0000), prev1 = hf16(0x0000);
#pragma unroll
                for (int ks = 0; ks < 3; ++ks) {
                    uint32_t a[2][4];
                    bfly2x4(aw[ks], a);
                    float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
                    for (int kk = 0; kk < 32; ++kk) {
                        const float pv = e4m3_to_f32(
                            (word_at(a, (kk >> 3) & 1, t ^ ((kk >> 1) & 3))
                             >> (8 * (2 * (kk >> 4) + (kk & 1)))) & 0xffu);
                        const int r = (32 * ks + kk) * 32 + 8 * nt + 2 * t;
                        acc0 += pv * e4m3_to_f32(vp[r]);
                        acc1 += pv * e4m3_to_f32(vp[r + 1]);
                    }
                    prev0 = (half_t)(acc0 + (float)prev0);
                    prev1 = (half_t)(acc1 + (float)prev1);
                }
                d[mi][rh][nt] = pair_of(prev0, prev1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The MLP's epilogues and the two operand helpers every mma in the entry shares.

// The sum-of-squares tree: the LayerNorm before phases 3 and 4 (PTX stmts 2399-3701) and
// E12's RMS norm before phase 13 (s12288-s13110), which are the same code with another
// gain table.  Per row of the lane: the four n-tiles are squared and added as (n0+n2) then
// (n1+n3), both halves in f16; the butterfly finishes the sum over the row's 32 columns
// and the two halves are added with add.f16; +eps, rsqrt.approx.ftz.f32 and back to f16,
// times the activation gain of the column.  The row's value times that factor is the A
// operand of the phase (cvt.rn.satfinite.e4m3x2.f16x2, no relu), packed as {n-tile 0,
// n-tile 2} into the k-group 0 register and {n-tile 1, n-tile 3} into k-group 1.  There is
// no mean subtraction and no 1/N: the quantity is the raw sum of the 32 squares.
DEVF void norm_as(const uint32_t (*d)[2][4], const uint8_t* w, int gain, half_t eps,
                  int nmi, uint32_t as[24])
{
#pragma unroll
    for (int mi = 0; mi < nmi; ++mi) {
        for (int row = 0; row < 2; ++row) {
            // the row's partial sums, packed for the two columns 2t and 2t+1
            uint32_t x = 0;
            for (int c = 0; c < 2; ++c) {
                half_t sq[4];
                for (int nt = 0; nt < 4; ++nt) {
                    const half_t v =
                        c == 0 ? half_lo(d[mi][row][nt]) : half_hi(d[mi][row][nt]);
                    sq[nt] = v * v;
                }
                const half_t s = (sq[0] + sq[2]) + (sq[1] + sq[3]);
                x |= (uint32_t)__builtin_bit_cast(uint16_t, s) << (16 * c);
            }
            // the PTX's two shfl.sync.bfly steps over the four lanes that hold
            // this row's 32 columns
            x = pair_add(x, bfly1_bits(x));
            x = pair_add(x, bfly2_bits(x));
            const half_t sum = (half_t)(half_lo(x) + half_hi(x));
            const half_t r = (half_t)__builtin_amdgcn_rsqf((float)(sum + eps));
            const uint32_t* dr = d[mi][row];
            uint32_t p0 = 0;
            uint32_t p1 = 0;
            for (int nt = 0; nt < 4; ++nt) {
                uint32_t prod = 0;
                for (int c = 0; c < 2; ++c) {
                    const half_t g = __builtin_bit_cast(half_t,
                        *(const uint16_t*)(w + gain + 16 * nt + 4 * (lane_id() & 3) + 2 * c));
                    const half_t v = (c == 0 ? half_lo(dr[nt]) : half_hi(dr[nt])) * (r * g);
                    prod |= (uint32_t)__builtin_bit_cast(uint16_t, v) << (16 * c);
                }
                const uint32_t code = e4m3x2(prod);
                (nt & 1 ? p1 : p0) |= code << 16 * (nt >> 1);
            }
            as[4 * mi + row] = p0;
            as[4 * mi + 2 + row] = p1;
        }
    }
}

// E13/E15/E17/E19, one packed pair: the clamped cubic activation, in f16 throughout.  The
// PTX's own sequence (s13997.. / s33795..) is
//   y = max(x, -2);  y = min(y, +2);  a = abs(y)
//   g = mid - slope * a;  g = y * g;  g = 0.5 + g;  out = x * g
// so the *signed* clamped y is the multiplier and only the inner `slope * a` term takes the
// absolute value; using |y| throughout would make every negative input's gain 0.5 + |y|(...)
// instead of 0.5 - |y|(...) and shift all four branches' activations.
DEVF uint32_t act_pair(uint32_t x)
{
    const hv2 two = hv2_of(pair_of(hf16(0x4000), hf16(0x4000)));
    const hv2 slope = hv2_of(pair_of(hf16(0x2d30), hf16(0x2d30)));
    const hv2 mid = hv2_of(pair_of(hf16(0x3698), hf16(0x3698)));
    const hv2 half0 = hv2_of(pair_of(hf16(0x3800), hf16(0x3800)));
    hv2 y = __builtin_elementwise_max(hv2_of(x), -two);
    y = __builtin_elementwise_min(y, two);
    const hv2 a = __builtin_elementwise_max(y, -y);        // abs.f16x2
    const hv2 g = half0 + y * (mid - slope * a);
    return u32_of(hv2_of(x) * g);
}

// E13/E15/E17/E19's pack: the activation of `d` requantised into the next phase's A operand.
// The A operand is four m-tiles of four registers: register 4*mi+row holds n-tiles 0 and 2,
// register 4*mi+2+row holds n-tiles 1 and 3, each as one e4m3x2 of the pair the mma reads.
DEVF void act_pack(const uint32_t d[4][2][4], uint32_t as[16])
{
    for (int mi = 0; mi < 4; ++mi) {
        for (int row = 0; row < 2; ++row) {
            const uint32_t a0 = e4m3x2(act_pair(d[mi][row][0]));
            const uint32_t a1 = e4m3x2(act_pair(d[mi][row][1]));
            const uint32_t a2 = e4m3x2(act_pair(d[mi][row][2]));
            const uint32_t a3 = e4m3x2(act_pair(d[mi][row][3]));
            as[4 * mi + row] = a0 | (a2 << 16);
            as[4 * mi + 2 + row] = a1 | (a3 << 16);
        }
    }
}

// The next phase's C seeds: the block's residual plus the per-column table at `tab`
// (eight f16 per lane, the lane's 8 columns of all four n-tiles), added in f16.
// Phase 14's residual is *phase 12's* accumulators -- the MLP adds its input block back, so
// the running residual enters phase 14, not phase 13's own first GEMM (VERIFIED against the
// oracle's own C registers, debug stage p14o: 1024/1024 over four blocks).
DEVF void tab_seed(const uint32_t d[4][2][4], const uint8_t* tab, int lane, uint32_t c[64])
{
    for (int mi = 0; mi < 4; ++mi) {
        for (int row = 0; row < 2; ++row) {
            for (int nt = 0; nt < 4; ++nt) {
                const uint32_t r = d[mi][row][nt];
                const uint32_t tb = *(const uint32_t*)(tab + 16 * nt + ((lane << 2) & 12));
                c[(4 * mi + row) * 4 + nt] = pair_of(
                    (half_t)(half_lo(tb) + half_lo(r)),
                    (half_t)(half_hi(tb) + half_hi(r)));
            }
        }
    }
}

// The four B fragment pairs of a phase (a weight-image B), as the mma reads them: the
// lane's own two words of tile nt>>1 at byte offset 8*(nt&1), which is its column
// 8*nt + g (VERIFIED against the oracle's own B registers, debug stage p8o).
DEVF void bpair_regs(const uint8_t* w, int base, int lane, uint32_t regs[8])
{
    for (int nt = 0; nt < 4; ++nt) {
        const uint8_t* const p = w + base + 512 * (nt >> 1) + 16 * lane + 8 * (nt & 1);
        regs[2 * nt] = *(const uint32_t*)p;
        regs[2 * nt + 1] = *(const uint32_t*)(p + 4);
    }
}

// The four bias words of a phase, one per n-tile: the f16 pair of the lane's columns 2t
// and 2t+1, each replicated over the 16 rows of its tile and used by all four m-tiles.
DEVF void wbias_regs(const uint8_t* w, int base, int lane, uint32_t regs[4])
{
    for (int nt = 0; nt < 4; ++nt) {
        regs[nt] = *(const uint32_t*)(w + base + 16 * nt + ((lane << 2) & 12));
    }
}

// A phase's C seeds in the oracle's register order: the mma run visits (m-tile, n-tile)
// with n-tile fastest, and each of its C pairs is the (row g, row g+8) pair.
DEVF void cseed_regs(const uint32_t c[64], uint32_t regs[32])
{
    for (int mi = 0; mi < 4; ++mi) {
        for (int nt = 0; nt < 4; ++nt) {
            for (int row = 0; row < 2; ++row) {
                regs[2 * (4 * mi + nt) + row] = c[(4 * mi + row) * 4 + nt];
            }
        }
    }
}

// An accumulator block as a C operand: the mma's C pair for m-tile mi and n-tile nt is
// d[mi][row][nt] (VERIFIED for phase 12 against the oracle's own C registers, stage p12o).
DEVF void cres_regs(const uint32_t d[4][2][4], uint32_t c[64])
{
    for (int mi = 0; mi < 4; ++mi) {
        for (int nt = 0; nt < 4; ++nt) {
            for (int row = 0; row < 2; ++row) {
                c[(4 * mi + row) * 4 + nt] = d[mi][row][nt];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The staged-image engine's plumbing: the four contract-§4 primitives live in
// rrswin_engine.h (rrswin::lds_u32, swin_slots_from_wimg, swin_dcell_exchange_write/read,
// swin_stage_operand_p with modes 0/1/2); what remains here is the body's own drivers.

// The engine tile's f16 pair compaction -- the one f16 rounding of a phase -- and its C
// seeds.  Per-column f16 tables widen to f32; a residual enters as the producer's own tile
// pairs (contract §2's identity: same tile, same lane, pair i>>1 half i&1), with a
// per-column table added in f16 first where one exists (phases 7 and 14).
DEVF void enc0_pack(uint32_t pairs[4], const f8v& c)
{
#pragma unroll
    for (int p = 0; p < 4; ++p)
        pairs[p] = pair_of((half_t)c[2 * p], (half_t)c[2 * p + 1]);
}

// The lane's 16 f16 of a per-column blob = 8 u32: features 16*mt + 8h + 2j and +1 sit at
// bytes 32*mt + 16h + 4j (column c at byte 2c, contract §2).
DEVF void enc0_load_cols(uint32_t wb[2][4], const uint8_t* w, int base)
{
    const uint32_t h = lane_id() >> 4;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            wb[mt][j] = *(const uint32_t*)(w + base + 32 * mt + 16 * h + 4 * j);
}

DEVF void enc0_seed_bias(f8v& c, int mt, const uint32_t wb[2][4])
{
#pragma unroll
    for (int i = 0; i < 8; ++i)
        c[i] = h2f((i & 1) ? (wb[mt][i >> 1] >> 16) : wb[mt][i >> 1]);
}

DEVF void enc0_seed_res(f8v& c, int mt, int nt, const uint32_t* pp, int n_nt)
{
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t r = pp[(mt * n_nt + nt) * 4 + (i >> 1)];
        c[i] = (float)((i & 1) ? half_hi(r) : half_lo(r));
    }
}

DEVF void enc0_seed_res_tab(f8v& c, int mt, int nt, const uint32_t* pp, int n_nt,
                            const uint32_t* wt)
{
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t r = pp[(mt * n_nt + nt) * 4 + (i >> 1)];
        const uint32_t tb = wt[(mt << 2) + (i >> 1)];
        c[i] = (float)((i & 1) ? half_hi(r) + half_hi(tb) : half_lo(r) + half_lo(tb));
    }
}

// One phase's dcell exchange: its 2*n_nt engine tiles do not fit the 2048-byte dead image
// (128 u32 per tile block), so the exchange runs in 2x2-tile chunks of 2048 bytes, chunk c
// refilling d[2c] and d[2c+1].  pairs is the lane's [2][n_nt][4] pair array, d its
// [n_nt][2][4] accumulator view.
DEVF void enc0_dcell_exchange(const uint32_t* pairs, int n_nt, lds_u32* ex, uint32_t* d)
{
    for (int c = 0; c < n_nt / 2; ++c) {
        uint32_t xp[16];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int l = 0; l < 2; ++l)
#pragma unroll
                for (int p = 0; p < 4; ++p)
                    xp[(mt * 2 + l) * 4 + p] = pairs[(mt * n_nt + 2 * c + l) * 4 + p];
        rrswin::swin_dcell_exchange_write(ex, xp, 2, 2);
        __builtin_amdgcn_wave_barrier();
        rrswin::swin_dcell_exchange_read(ex, d + 16 * c, 2, 4, 2);
        __builtin_amdgcn_wave_barrier();
    }
}

// Phase 1: the staged block's fragments stage into the image per 48-token half (the six
// m-tiles restaged at local rows 0..47), the C seed is the per-column bias at w+5120, and
// the result stays in registers as pairs.
DEVF void enc0_phase1(const uint32_t* as1, const uint8_t* w, lds_u8* slots, lds_u8* img,
                      uint32_t pairs1[2][6][4])
{
    uint32_t wb[2][4];
    rrswin::swin_slots_from_wimg(slots, w, 4096, 32);
    rrswin::swin_stage_fragment(img, 32, as1, 3);
    __builtin_amdgcn_wave_barrier();
    enc0_load_cols(wb, w, 5120);
    for (int half = 0; half < 2; ++half) {
        if (half != 0) {
            rrswin::swin_stage_fragment(img, 32, as1 + 12, 3);
            __builtin_amdgcn_wave_barrier();
        }
#pragma unroll
        for (int nt = 3 * half; nt < 3 * half + 3; ++nt)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                f8v c;
                enc0_seed_bias(c, mt, wb);
                c = rrswin::swin_gemm_tile(c, slots, 16 * mt, 32, img, 32,
                                           16 * (nt - 3 * half));
                enc0_pack(pairs1[mt][nt], c);
            }
        __builtin_amdgcn_wave_barrier();
    }
}

// Phase 2: phase 1's pairs staged with the relu (mode 1) as the A image, the per-column
// bias at w+6208 as C; the caller keeps pairs2 for phase 7's C and exchanges the dcell view.
DEVF void enc0_phase2(const uint32_t pairs1[2][6][4], const uint8_t* w, lds_u8* slots,
                      lds_u8* img, uint32_t pairs2[2][6][4])
{
    uint32_t wb[2][4];
    rrswin::swin_slots_from_wimg(slots, w, 5184, 32);
    __builtin_amdgcn_wave_barrier();
    enc0_load_cols(wb, w, 6208);
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int nt = 3 * half; nt < 3 * half + 3; ++nt)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                rrswin::swin_stage_operand_p(img, 32, 16 * (nt - 3 * half), mt, pairs1[mt][nt], 1);
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int nt = 3 * half; nt < 3 * half + 3; ++nt)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                f8v c;
                enc0_seed_bias(c, mt, wb);
                c = rrswin::swin_gemm_tile(c, slots, 16 * mt, 32, img, 32,
                                           16 * (nt - 3 * half));
                enc0_pack(pairs2[mt][nt], c);
            }
        __builtin_amdgcn_wave_barrier();
    }
}

// Phases 3, 4, 8 and 9: the fused projections of the normed tile -- a 96-token phase whose
// A image is the as3 fragment pack staged per half and whose D is consumed only as the
// NVIDIA d[mi][row][nt] view (the score and P·V GEMMs walk it scalar).
DEVF void enc0_proj96(const uint32_t* as3, const uint8_t* w, int base, lds_u8* slots,
                      lds_u8* img, lds_u32* ex, uint32_t* d)
{
    uint32_t pr[2][6][4];
    rrswin::swin_slots_from_wimg(slots, w, base, 32);
    rrswin::swin_stage_fragment(img, 32, as3, 3);
    __builtin_amdgcn_wave_barrier();
    for (int half = 0; half < 2; ++half) {
        if (half != 0) {
            rrswin::swin_stage_fragment(img, 32, as3 + 12, 3);
            __builtin_amdgcn_wave_barrier();
        }
#pragma unroll
        for (int nt = 3 * half; nt < 3 * half + 3; ++nt)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
                c = rrswin::swin_gemm_tile(c, slots, 16 * mt, 32, img, 32,
                                           16 * (nt - 3 * half));
                enc0_pack(pr[mt][nt], c);
            }
        __builtin_amdgcn_wave_barrier();
    }
    enc0_dcell_exchange(&pr[0][0][0], 6, ex, d);
}

// The four MLP branches' first GEMMs (13, 15, 17, 19): the normed tile as the A image, the
// per-column bias as C, the result kept as pairs for the mode-2 staging and the residual.
DEVF void enc0_branch(const uint32_t* asN, const uint8_t* w, int base, int bias,
                      lds_u8* slots, lds_u8* img, uint32_t pairs[2][4][4])
{
    uint32_t wb[2][4];
    rrswin::swin_slots_from_wimg(slots, w, base, 32);
    rrswin::swin_stage_fragment(img, 32, asN, 4);
    __builtin_amdgcn_wave_barrier();
    enc0_load_cols(wb, w, bias);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            f8v c;
            enc0_seed_bias(c, mt, wb);
            c = rrswin::swin_gemm_tile(c, slots, 16 * mt, 32, img, 32, 16 * nt);
            enc0_pack(pairs[mt][nt], c);
        }
    __builtin_amdgcn_wave_barrier();
}

// The four branches' second GEMMs (14, 16, 18, 20): the previous branch's pairs staged with
// the clamped cubic transform (mode 2) as the A image (pp), the running residual as C (pc)
// -- phase 12's pairs plus the per-column table for phase 14 (wt), the producer pairs
// without one for 16, 18 and 20 (wt null).
DEVF void enc0_branch_act(const uint32_t pp[2][4][4], const uint32_t* pc, const uint32_t* wt,
                          const uint8_t* w, int base, lds_u8* slots, lds_u8* img,
                          uint32_t pairs[2][4][4])
{
    rrswin::swin_slots_from_wimg(slots, w, base, 32);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
            rrswin::swin_stage_operand_p(img, 32, 16 * nt, mt, pp[mt][nt], 2);
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            f8v c;
            if (wt != nullptr) {
                enc0_seed_res_tab(c, mt, nt, pc, 4, wt);
            } else {
                enc0_seed_res(c, mt, nt, pc, 4);
            }
            c = rrswin::swin_gemm_tile(c, slots, 16 * mt, 32, img, 32, 16 * nt);
            enc0_pack(pairs[mt][nt], c);
        }
    __builtin_amdgcn_wave_barrier();
}

// ---------------------------------------------------------------------------
// Ground truth for a stage comes from kernels/tex/make_ptx.py's debug recipe:
// with D4R_RRSWIN_DEBUG=p1 (p2, p3, p4, or a) the same splice stores that stage's
// registers into the plane arena at +46,000,000, which the kernel never otherwise
// writes, and the launch dump then holds the translated values for a block/lane.
// Built with -DD4R_RRSWIN_DEBUG_STAGE=<n> (kernels/build.sh rrswin,
// D4R_TEX_CFLAGS), this body stores the same registers of its own phase n in the
// same order, so the two dumps compare byte for byte.
#ifndef D4R_RRSWIN_DEBUG_STAGE
#define D4R_RRSWIN_DEBUG_STAGE 0
#endif

DEVF void debug_store(const uint8_t* prm, const uint32_t* regs, int n)
{
    const uint32_t block = __builtin_amdgcn_workgroup_id_x()
                         + 161u * __builtin_amdgcn_workgroup_id_y();
    if (block >= 64u) {
        return;   // the oracle's per-block slots only cover the first 64 blocks
    }
    const uint64_t arena = *(const uint64_t*)(prm + kPlaneArg);
    // the oracle's slot, make_ptx.py's swin_debug_slot: 32 lanes x n x 4 bytes, at least 8192,
    // rounded up to a power of two (the oracle's splice forms block*slot with a shift)
    const uint64_t want = (uint64_t)n * 128ull > 8192ull ? (uint64_t)n * 128ull : 8192ull;
    uint64_t slot = 1;
    while (slot < want) {
        slot <<= 1;
    }
    uint32_t* dst = (uint32_t*)(arena + 46000000ull + (uint64_t)block * slot
                                 + (uint64_t)lane_id() * (uint64_t)n * 4ull);
    for (int i = 0; i < n; ++i) {
        dst[i] = regs[i];
    }
}

// stage registers in the oracle's order: 24 mma, m-tile major, n-tile minor, each
// as the (row g, row g+8) pair the mma writes.
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

// the same for the sixteen-mma phases: m-tile major, n-tile minor, the (row g, row g+8)
// pair each mma writes.
DEVF void stage_regs4(const uint32_t d[4][2][4], uint32_t regs[32])
{
    int i = 0;
    for (int mi = 0; mi < 4; ++mi) {
        for (int nt = 0; nt < 4; ++nt) {
            regs[i++] = d[mi][0][nt];
            regs[i++] = d[mi][1][nt];
        }
    }
}

// ---------------------------------------------------------------------------
// The merge: E20's arena scatter and its eight shfl pairs, then phases 21 and 22.

// two u16 into one b32 (the PTX's mov.b32 of two cvt results)
DEVF uint32_t pair16(uint16_t a, uint16_t b)
{
    return (uint32_t)a | ((uint32_t)b << 16);
}

// E20(b): the sixteen guarded stores of phase 20's requantised output into the plane
// arena.  The index is a word index, idx = 8*ex*y + 8*x + d, where (ex, ey) are the
// extents of the parameter block's first word pair, x0 = 8*ctaid.x - param[80] and
// y0 = 8*(ctaid.y + tid.y) - param[82] are the block origins (the PTX's s6 and s16), and
// the lane's cell is x = x0 + (g & 7) with g = laneid>>2 and t = laneid&3: the eight
// stores of one half walk the eight rows y0 + ((g + 8*j)>>3) = y0 + j and the two words
// d = t and t|4 of them.  A store whose y or x falls outside the arena is skipped (the
// PTX turns its index into -1 and branches).
DEVF void arena_scatter(const uint8_t* prm, const uint32_t r[16], int lane)
{
    const uint32_t ex = *(const uint32_t*)prm;
    const uint32_t ey = *(const uint32_t*)(prm + 4);
    const int x0 = 8 * (int)__builtin_amdgcn_workgroup_id_x()
                 - (int)*(const uint16_t*)(prm + 80);
    const int y0 = 8 * (int)(__builtin_amdgcn_workgroup_id_y()
                             + __builtin_amdgcn_workitem_id_y())
                 - (int)*(const uint16_t*)(prm + 82);
    uint8_t* const s = (uint8_t*)(uintptr_t)*(const uint64_t*)(prm + kPlaneArg);
    const int x = x0 + ((lane >> 2) & 7);
    const int t = lane & 3;
    if ((unsigned)x >= ex) {
        return;
    }
    const int stride = 8 * (int)ex;
    for (int j = 0; j < 8; ++j) {
        const int y = y0 + (((lane >> 2) + 8 * j) >> 3);
        if ((unsigned)y >= ey) {
            continue;
        }
        const int idx = stride * y + 8 * x;
        *(uint32_t*)(s + 4 * (uint32_t)(idx + t)) = r[2 * j];
        *(uint32_t*)(s + 4 * (uint32_t)(idx + (t | 4))) = r[2 * j + 1];
    }
}

// E20(c): the eight shfl pairs, all with delta 4, that re-pair the eight lanes of every
// group of eight.  For a source pair (lo, hi) the lane keeps its own `lo` and its
// partner's `hi` when it is in the lower group of four (laneid & 4 == 0) and the other way
// round when it is in the upper one, so each destination ends up holding both members:
//   d1 = (lane & 4) == 0 ? lo[lane]     : hi[lane - 4]
//   d2 = (lane & 4) == 0 ? lo[lane + 4] : hi[lane]
// The eight pairs are (%r262,%r266) (%r264,%r268) (%r263,%r267) (%r265,%r269)
// (%r270,%r274) (%r272,%r276) (%r271,%r275) (%r273,%r277) and their sixteen results are
// phase 21's four A fragments: %r17490..%r17493, %r17510..%r17513, %r17530..%r17533 and
// %r17550..%r17553, in that order.
// The partner values go through the shared scratch (sixteen words per lane, 512 of the
// 3200 bytes the cut hands over and dead by now) rather than through two more warp
// shuffles: the exchange is the same read of the same lane's value at the same point.
DEVF void merge_frag(const uint32_t r[16], uint32_t scratch, int lane, uint32_t a[16])
{
    const int lo[8] = {0, 2, 1, 3, 8, 10, 9, 11};
    const int hi[8] = {4, 6, 5, 7, 12, 14, 13, 15};
    // where each pair's first and second result land, as indices into a[]
    const int d1[8] = {0, 8, 2, 10, 1, 9, 3, 11};
    const int d2[8] = {4, 12, 6, 14, 5, 13, 7, 15};
    lds_u32* const s = (lds_u32*)(uintptr_t)scratch;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        s[16 * lane + i] = r[i];
    }
    __builtin_amdgcn_wave_barrier();
    const int partner = lane ^ 4;
    const bool up = (lane & 4) != 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[d1[i]] = up ? s[16 * partner + hi[i]] : r[lo[i]];
        a[d2[i]] = up ? s[16 * lane + hi[i]] : s[16 * partner + lo[i]];
    }
}

// Phase 21: 16 rows x 64 columns contracting over K = 128 in four 32-wide steps.  All 32
// mma read the same four A fragments (one per k-step, the merged `a`), eight n-tiles' B
// fragment pairs from the weight image at 45696, and eight C words at 53888 (one per
// n-tile, replicated over the tile's 16 rows).  The A operand's row is the lane's own g
// and its column index 2*(lane & 3), so the four eighths of every k slice come from the
// four lanes of the row group and the two butterfly steps bring them in; the B operand's
// column index is the lane's own g, so the output column 2t + c is the B column of the
// lanes whose g is 2t + c -- which the weight image already holds, and each k-step rounds
// to f16 before the next one adds to it.
DEVF void band21(const uint32_t a[16], const uint8_t* w, const uint8_t* cbias, int lane,
                 uint32_t d[8][2])
{
    const int t = lane & 3;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        half_t prev[2][2] = {{hf16(0x0000), hf16(0x0000)}, {hf16(0x0000), hf16(0x0000)}};
        const uint32_t cb = *(const uint32_t*)(cbias + 16 * nt + ((lane << 2) & 12));
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // the lane's own A fragment of this k-step, as the four lanes of its row group
            // hold it: register o is row o&1, k slice o>>1
            uint32_t ar[2][2][4];      // [row][k slice][lane t^m]
#pragma unroll
            for (int slice = 0; slice < 2; ++slice) {
                for (int row = 0; row < 2; ++row) {
                    const uint32_t v = a[4 * ks + 2 * slice + row];
                    ar[row][slice][0] = v;
                    ar[row][slice][1] = bfly1_bits(v);
                    ar[row][slice][2] = bfly2_bits(v);
                    ar[row][slice][3] = bfly1_bits(ar[row][slice][2]);
                }
            }
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                // the B fragment pairs of the four lanes whose g is the D column 2t + c
                const int gp = 2 * t + c;
                uint32_t br[2][4];   // [k slice][lane m]
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const uint8_t* const p = w + 45696 + 512 * (ks + 4 * (nt >> 1))
                                         + 16 * (4 * gp + m) + 8 * (nt & 1);
                    br[0][m] = *(const uint32_t*)p;
                    br[1][m] = *(const uint32_t*)(p + 4);
                }
#pragma unroll
                for (int row = 0; row < 2; ++row) {
                    float s2 = 0.0f;
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
#pragma unroll
                        for (int slice = 0; slice < 2; ++slice) {
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                s2 += e4m3_to_f32((ar[row][slice][m] >> (8 * j)) & 0xffu)
                                    * e4m3_to_f32((br[slice][m] >> (8 * j)) & 0xffu);
                            }
                        }
                    }
                    prev[row][c] = (half_t)(s2 + (ks == 0 ? h2f(cb) : 0.0f)
                                            + (float)prev[row][c]);
                }
            }
        }
        d[nt][0] = pair_of(prev[0][0], prev[0][1]);
        d[nt][1] = pair_of(prev[1][0], prev[1][1]);
    }
}

// E21: phase 21's requantisation and its eight guarded stores into the buffer the
// parameter block holds at +56 -- not the plane arena.  The extents come from the first
// word pair halved, the origin from %r1 and %r6 halved, and the two index bases are
//   base1 = 8*(ex>>1)*y1 + 8*x1   with y1 = (%r6>>1) + ((g&1)<<1),
//                                        x1 = (%r1>>1) + ((laneid>>3) & 3)
//   base2 = 8*(ex>>1)*y2 + 8*x2   with y2 = y1 + 1,  x2 = (%r1>>1) + ((g>>1) & 3)
// (the laneid>>5 term of y1 and the tile-bit test of the guard are both zero for a
// 32-lane block), each store landing at base + t or base + (t|4) words, the last four
// of them one row-stride further on.  The sixteen payloads are phase 21's D registers in
// mma order, one e4m3x2 each: the first and the last two travel as v2.u16 and the middle
// twelve as six u32.
DEVF void surface_scatter21(const uint8_t* prm, const uint32_t d[8][2], int lane)
{
#if defined(D4R_RRSWIN_SC1) && D4R_RRSWIN_SC1 == 1
    (void)prm; (void)d; (void)lane;
    return;   // regression probe: empty call
#endif
    const int r1 = 8 * (int)__builtin_amdgcn_workgroup_id_x()
                 - (int)*(const uint16_t*)(prm + 80);
    const int r6 = 8 * (int)(__builtin_amdgcn_workgroup_id_y()
                             + __builtin_amdgcn_workitem_id_y())
                 - (int)*(const uint16_t*)(prm + 82);
    const int ex = (int)(*(const uint32_t*)prm >> 1);
    const int ey = (int)(*(const uint32_t*)(prm + 4) >> 1);
    const int g = lane >> 2;
    const int t = lane & 3;
    const int sx = 8 * ex;
    const int y1 = (r6 >> 1) + ((g & 1) << 1);
    const int y2 = y1 + 1;
    const int x1 = (r1 >> 1) + ((lane >> 3) & 3);
    const int x2 = (r1 >> 1) + ((g >> 1) & 3);
    uint8_t* const dst_raw = (uint8_t*)(uintptr_t)*(const uint64_t*)(prm + 56);
    // quadrant bisect: 1 = the four prm reads and the target load, 2 = the q fill,
    // 3 = the index arithmetic and its two gates, 4 = the store groups
#ifndef D4R_RRSWIN_Q
#define D4R_RRSWIN_Q 4
#endif
#if D4R_RRSWIN_Q == 1
    (void)d; (void)lane;
    return;
#endif
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT == 19
    uint8_t* const dst = (uint8_t*)(uintptr_t)*(const uint64_t*)(prm + 48);   // the dumped arena
#else
    uint8_t* const dst = dst_raw;
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 19
    {
        // pointer capture: prm's own words, the +56 target and the geometry, in the
        // debug-store convention (stage_regs4 layout filler is irrelevant here)
        uint32_t cap[36];
        for (int i = 0; i < 32; ++i) {
            cap[i] = *(const uint32_t*)(prm + 4 * i);
        }
        cap[32] = (uint32_t)(uintptr_t)(prm + 56);
        *(uint64_t*)(cap + 32) = *(const uint64_t*)(prm + 56);
        debug_store(prm, cap, 34);
    }
#endif
    // The sixteen payloads as the eight u32 words the four store groups write.
    //   pair16(q[2n], q[2n+1]) with q[k] = (uint16_t)e4m3x2(d[n][row]) is exactly
    //   e4m3x2(d[n][0]) | (e4m3x2(d[n][1]) << 16):
    //   e4m3x2 returns two e4m3 bytes, each in [0,255] with a zero upper half, so the
    //   uint16_t truncation in q[] is a no-op and the packing is bit for bit the same.
    //   Forming the words directly (rather than through a uint16_t q[16] that -O2/-O3
    //   widens into a sixteen-lane <16 x i16> built by a shuffle whose mask indexes
    //   past its operands -- the last two lanes come back `poison`) keeps each payload in
    //   one u32 register and restores the -O0/-O1 lowering of the stores.
    uint32_t w[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        w[nt] = e4m3x2(d[nt][0]) | (e4m3x2(d[nt][1]) << 16);
    }
#if D4R_RRSWIN_Q == 2
    return;
#endif
    const int b1 = sx * y1 + 8 * x1;
    const int b2 = sx * y2 + 8 * x2;
    const int st = sx * ey;
    const bool ok1 = (unsigned)y1 < (unsigned)ey && (unsigned)x1 < (unsigned)ex;
    const bool ok2 = (unsigned)y2 < (unsigned)ey && (unsigned)x2 < (unsigned)ex;
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT == 18
    return;   // bisect: the scatter's prm/geometry reads only, no stores
#endif
#if D4R_RRSWIN_Q == 3
    return;
#endif
#if D4R_RRSWIN_Q == 9
    // same as Q5, but the target pointer is re-read from prm inside the guarded block
    if (ok1) {
        uint8_t* const d2 = (uint8_t*)(uintptr_t)*(const uint64_t*)(prm + 56);
        *(uint32_t*)(d2 + 4 * (uint32_t)(b1 + t)) = w[0];
        *(uint32_t*)(d2 + 4 * (uint32_t)(b1 + (t | 4))) = w[1];
    }
    return;
#endif
    // Q5..Q8 keep exactly one store group; Q4 keeps all four
#if D4R_RRSWIN_Q >= 4
    if (ok1) {
        *(uint32_t*)(dst + 4 * (uint32_t)(b1 + t)) = w[0];
        *(uint32_t*)(dst + 4 * (uint32_t)(b1 + (t | 4))) = w[1];
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT == 15
        return;
#endif
    }
#endif
#if D4R_RRSWIN_Q == 4 || D4R_RRSWIN_Q >= 6
    if (ok2) {
        *(uint32_t*)(dst + 4 * (uint32_t)(b2 + t)) = w[2];
        *(uint32_t*)(dst + 4 * (uint32_t)(b2 + (t | 4))) = w[3];
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT == 16
        return;
#endif
    }
#endif
#if D4R_RRSWIN_Q == 4 || D4R_RRSWIN_Q == 7
    if (ok1) {
        *(uint32_t*)(dst + 4 * (uint32_t)(b1 + st + t)) = w[4];
        *(uint32_t*)(dst + 4 * (uint32_t)(b1 + st + (t | 4))) = w[5];
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT == 17
        return;
#endif
    }
#endif
#if D4R_RRSWIN_Q == 4 || D4R_RRSWIN_Q == 8
    if (ok2) {
        *(uint32_t*)(dst + 4 * (uint32_t)(b2 + st + t)) = w[6];
        *(uint32_t*)(dst + 4 * (uint32_t)(b2 + st + (t | 4))) = w[7];
    }
#endif
}


// ---------------------------------------------------------------------------
// Mirror cut. The six 16-row fragments of the staged block input arrive in this
// order (the first mma's operand order, .f16.e4m3 fragments of 4 b32 each):
//   f0 %r120 %r136 %r128 %r144   f3 %r216 %r232 %r224 %r240
//   f1 %r152 %r168 %r160 %r176   f4 %r246 %r257 %r253 %r1042
//   f2 %r184 %r200 %r192 %r208   f5 %r1079 %r1080 %r1081 %r1082
// then phase 1's four bias C seeds (%r1086 %r1096 %r1106 %r1116, read from
// +5120/+5136/+5152/+5168), the weight image and the parameter block.
//
// The eighteen accumulators phase 22's epilogue reads cross the cut through shared memory, not
// `call.uni`'s return parameters: ZLUDA lowers those to the function's LLVM return type (emit.rs's
// emit_call makes one value, or an get_or_create_struct_type struct for several, and distributes
// them with LLVMBuildExtractValue), so there is no .param-space object and no pointer is formed --
// the callee reads 0x0 in all eighteen slots, and an 18 x i32 C++ struct return is an sret pointer,
// a different LLVM signature that will not link. `scratch` is the base of the entry's `input_tensor`
// (.shared 3200 bytes, dead after the cut), 72 bytes apart per lane.
#ifndef D4R_RRSWIN_PAD
#define D4R_RRSWIN_PAD 0
#endif
#ifndef D4R_RRSWIN_VGPRC
#define D4R_RRSWIN_VGPRC 0
#endif
#if D4R_RRSWIN_VGPRC
__attribute__((amdgpu_max_vgpr_count(D4R_RRSWIN_VGPRC)))
#endif
extern "C" __attribute__((device)) void d4r_swin_enc0(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6,
    uint32_t a7, uint32_t a8, uint32_t a9, uint32_t a10, uint32_t a11, uint32_t a12, uint32_t a13,
    uint32_t a14, uint32_t a15, uint32_t a16, uint32_t a17, uint32_t a18, uint32_t a19, uint32_t a20,
    uint32_t a21, uint32_t a22, uint32_t a23, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    const uint8_t* w, const uint8_t* prm, uint32_t scratch)
{
    const uint32_t as1[24] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                              a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23};
    // b0..b3 (phase 1's bias C seeds, this lane's t-slice of the per-column blob) are unused
    // by the engine body: its column-major C seed reads the whole blob at w+5120 directly.
    // The epilogue's eighteen registers: %r17597, %r18203 (%tid.x) and phase 22's eight mma D pairs.
    // Address space 3 (the shared scratch) explicitly: a generic store lowers to flat, which faults.
    __attribute__((address_space(3))) uint32_t* out =
        ((__attribute__((address_space(3))) uint32_t*)(uintptr_t)scratch) + 18 * (uint32_t)lane_id();
    const int t = (int)(lane_id() & 3);   // the lane's two columns of each n-tile
    const int g = (int)(lane_id() >> 2);  // ... and of its sixteen rows
    const uint32_t lane = lane_id();
    const half_t eps = (half_t)0.0009765625f;   // the tree's sum-of-squares epsilon
#if D4R_RRSWIN_PAD
    volatile uint32_t pad[D4R_RRSWIN_PAD];
#pragma unroll
    for (int i = 0; i < D4R_RRSWIN_PAD; ++i) {
        pad[i] = (uint32_t)i;
    }
#endif

    // The staged-image engine's shared regions: the prepared slots (1024 B) at +0, the
    // phase's A image (2048 B) at +1024, and the dcell exchange overlaying the dead image
    // (3072 of the 3200 bytes; the score_gemm/pv_gemm `slot` = scratch use comes later,
    // by then both are dead).  AS3 pointers throughout: generic pointers built from the
    // raw shared-offset value lower to flat accesses at VA ~0 and fault.
    lds_u8* const slots = (lds_u8*)(uintptr_t)scratch;
    lds_u8* const img = slots + 1024;
    lds_u32* const ex = (lds_u32*)(uintptr_t)(scratch + 1024);

    // ---- phase 1 -----------------------------------------------------------
    // The staged block's fragments stage into the image per 48-token half (the six m-tiles
    // restaged at local rows 0..47); the C seed is the per-column bias at w+5120.  The
    // result stays in registers as pairs; E1's relu requantisation happens inside phase 2's
    // mode-1 staging.
    uint32_t pairs1[2][6][4];
    enc0_phase1(as1, w, slots, img, pairs1);
#if D4R_RRSWIN_DEBUG_STAGE == 1
    {
        uint32_t d1o[6][2][4];
        enc0_dcell_exchange(&pairs1[0][0][0], 6, ex, &d1o[0][0][0]);
        uint32_t regs[48];
        stage_regs(d1o, regs);
        debug_store(prm, regs, 48);
    }
#endif

    // ---- phase 2, on phase 1's relu requantisation --------------------------
    // E1's relu (cvt.rn.satfinite.relu.e4m3x2.f16x2) is a per-pair transform, so staging
    // phase 1's pairs with swin_stage_operand_p's mode 1 replaces the old as2 pack
    // bit-identically; the C seed is the per-column bias at w+6208.  The dcell exchange
    // feeds norm_as (and stage 2); the pairs themselves stay for phase 7's C.
    uint32_t pairs2[2][6][4];
    enc0_phase2(pairs1, w, slots, img, pairs2);
    uint32_t d2[6][2][4];
    enc0_dcell_exchange(&pairs2[0][0][0], 6, ex, &d2[0][0][0]);
#if D4R_RRSWIN_DEBUG_STAGE == 2
    {
        uint32_t regs[48];
        stage_regs(d2, regs);
        debug_store(prm, regs, 48);
    }
#endif

    // ---- the LayerNorm tree (PTX stmts 2399-3701) --------------------------
    // Per row: the four n-tiles are squared and added as (n0+n2) then (n1+n3),
    // both halves in f16; the butterfly finishes the sum over the row's 32
    // columns and the two halves are added with add.f16; +2^-13, then
    // rsqrt.approx.ftz.f32 and back to f16, times the activation gain of the
    // column. The row's value times that factor is the A operand of phases 3 and
    // 4 (cvt.rn.satfinite.e4m3x2.f16x2, no relu).
    uint32_t as3[24];
    norm_as(d2, w, 6272, eps, 6, as3);

#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 1
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 1) {
        return;
    }
#endif

    // ---- phases 3 and 4, the fused Q/V projection, on that A operand -------
    // Both phases read the same normed-tile image halves (as3's fragment groups restaged
    // per half); their D is consumed only as the dcell view (phases 5/6 walk it scalar).
    uint32_t d3[6][2][4];
    uint32_t d4[6][2][4];
    enc0_proj96(as3, w, 6336, slots, img, ex, &d3[0][0][0]);
    enc0_proj96(as3, w, 7360, slots, img, ex, &d4[0][0][0]);
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 2
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 2) {
        return;
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 3
    {
        uint32_t regs[48];
        stage_regs(d3, regs);
        debug_store(prm, regs, 48);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 4
    {
        uint32_t regs[48];
        stage_regs(d4, regs);
        debug_store(prm, regs, 48);
    }
#endif

    // ---- phases 5 to 7: the first window --------------------------------------
    // score_gemm() and pv_gemm() are the two GEMMs both windows run; see their comments
    // for the fragment bookkeeping.  The first window scores phase 3's accumulator against
    // itself (A and B are both D3 -- the PTX's 24 B fragments are %r2932+10*tt, m-tile
    // tt>>1, row half tt&1, VERIFIED against the oracle's own B registers) with the bias
    // at weight byte 8384, and takes its V from phase 4's accumulator.
#if D4R_RRSWIN_DEBUG_STAGE == 201 || D4R_RRSWIN_DEBUG_STAGE == 203 \
    || D4R_RRSWIN_DEBUG_STAGE == 204 || D4R_RRSWIN_DEBUG_STAGE == 211
    uint32_t cap[136];
    uint32_t* const cp = cap;
#else
    uint32_t* const cp = nullptr;
#endif
    lds_u32* const slot = (lds_u32*)(uintptr_t)scratch;   // twelve n-tiles x 64 words
    uint32_t d5[4][2][12];       // [m-tile][row half][n-tile]: f16 pair
    score_gemm(d3, w, 8384, t, g, (int)lane, slot, cp, d5);
#if D4R_RRSWIN_DEBUG_STAGE == 5
    {
        uint32_t regs[96];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nb = 0; nb < 12; ++nb) {
                regs[i++] = d5[mi][0][nb];
                regs[i++] = d5[mi][1][nb];
            }
        }
        debug_store(prm, regs, 96);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 203
    debug_store(prm, cp, 24);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 204
    debug_store(prm, cp, 136);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 211
    debug_store(prm, cp, 120);
#endif

    uint32_t d6[4][2][4];
    pv_gemm(d5, d4, t, g, (lds_u8*)(uintptr_t)scratch, cp, d6);
#if D4R_RRSWIN_DEBUG_STAGE == 6
    {
        uint32_t regs[32];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 4; ++nt) {
                regs[i++] = d6[mi][0][nt];
                regs[i++] = d6[mi][1][nt];
            }
        }
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 201
    debug_store(prm, cp, 48);
#endif

    // ---- E6 and phase 7, the first window's output projection ----------------
    // A is phase 6's D packed as before (the as7 fragment pack stages as the image); C is
    // phase 2's pairs plus the per-column table at weight bytes 37056..37120, seeded per
    // tile from the producer's own pairs (contract §2's residual identity, VERIFIED, stage
    // e6).  The pairs stay for phase 12's C; the dcell view only the stage 7 and stage 112
    // c12 rebuilds need, so its exchange is debug-only.
    uint32_t as7[16];
    for (int mi = 0; mi < 4; ++mi) {
        for (int row = 0; row < 2; ++row) {
            as7[4 * mi + row] = e4m3x2(d6[mi][row][0]) | (e4m3x2(d6[mi][row][2]) << 16);
            as7[4 * mi + 2 + row] = e4m3x2(d6[mi][row][1]) | (e4m3x2(d6[mi][row][3]) << 16);
        }
    }
    uint32_t wt7[2][4];
    enc0_load_cols(wt7, w, 37056);
    uint32_t pairs7[2][4][4];
    rrswin::swin_slots_from_wimg(slots, w, 20672, 32);
    rrswin::swin_stage_fragment(img, 32, as7, 4);
    __builtin_amdgcn_wave_barrier();
    for (int nt = 0; nt < 4; ++nt) {
        for (int mt = 0; mt < 2; ++mt) {
            f8v c;
            enc0_seed_res_tab(c, mt, nt, &pairs2[0][0][0], 6, &wt7[0][0]);
            c = rrswin::swin_gemm_tile(c, slots, 16 * mt, 32, img, 32, 16 * nt);
            enc0_pack(pairs7[mt][nt], c);
        }
    }
    __builtin_amdgcn_wave_barrier();
#if D4R_RRSWIN_DEBUG_STAGE == 7 || D4R_RRSWIN_DEBUG_STAGE == 112
    uint32_t d7o[4][2][4];
    enc0_dcell_exchange(&pairs7[0][0][0], 4, ex, &d7o[0][0][0]);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 202
    {
        // the oracle's C array, rebuilt from phase 2's dcell view (which norm_as read in
        // production) plus the per-column table
        uint32_t c7[64];
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 4; ++nt) {
                const uint32_t tab = *(const uint32_t*)(w + 37056 + 16 * nt + ((lane << 2) & 12));
                for (int row = 0; row < 2; ++row) {
                    const uint32_t r = d2[mi][row][nt];
                    c7[(4 * mi + row) * 4 + nt] = pair_of(
                        (half_t)(half_lo(tab) + half_lo(r)), (half_t)(half_hi(tab) + half_hi(r)));
                }
            }
        }
        uint32_t regs[48];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int o = 0; o < 4; ++o) {
                regs[i++] = as7[4 * mi + o];
            }
        }
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 4; ++nt) {
                regs[i++] = c7[(4 * mi + 0) * 4 + nt];
                regs[i++] = c7[(4 * mi + 1) * 4 + nt];
            }
        }
        debug_store(prm, regs, 48);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 7
    {
        uint32_t regs[32];
        stage_regs4(d7o, regs);
        debug_store(prm, regs, 32);
    }
#endif
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 3
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 3) {
        return;
    }
#endif

    // ---- phases 8 and 9: the second window's two projections -----------------
    // The same LayerNorm tile as the A operand of phases 3 and 4 (all four mma read the
    // registers %r6784+40*mi -- VERIFIED against the oracle's own A fragment registers,
    // debug stage p8o) and no bias: the second window scores D8 against itself and takes
    // its V from D9.  E7 and E8 between them only load the two weight tiles this reads.
    uint32_t d8[6][2][4];
    uint32_t d9[6][2][4];
    enc0_proj96(as3, w, 21696, slots, img, ex, &d8[0][0][0]);
    enc0_proj96(as3, w, 22720, slots, img, ex, &d9[0][0][0]);
#if D4R_RRSWIN_DEBUG_STAGE == 8
    {
        uint32_t regs[48];
        stage_regs(d8, regs);
        debug_store(prm, regs, 48);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 9
    {
        uint32_t regs[48];
        stage_regs(d9, regs);
        debug_store(prm, regs, 48);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 108
    {
        // phase 8's mma operands: the six A fragments (the LayerNorm tile) and the four
        // B fragment pairs, which are the lane's own two words of tile nt>>1 at byte
        // offset 8*(nt&1) -- its column 8*nt + g
        uint32_t regs[32];
        for (int o = 0; o < 24; ++o) {
            regs[o] = as3[o];
        }
        for (int nt = 0; nt < 4; ++nt) {
            const uint8_t* const p = w + 21696 + 512 * (nt >> 1) + 16 * lane + 8 * (nt & 1);
            regs[24 + 2 * nt] = *(const uint32_t*)p;
            regs[24 + 2 * nt + 1] = *(const uint32_t*)(p + 4);
        }
        debug_store(prm, regs, 32);
    }
#endif

    // ---- phases 10 and 11, and E10: the second window's score and P·V --------
    // Identical to phases 5 and 6 with phase 8's accumulator in place of phase 3's and the
    // bias at weight bytes 23744..35532 (the second 12288-byte table, same layout).
    uint32_t d10[4][2][12];
    score_gemm(d8, w, 23744, t, g, (int)lane, slot, cp, d10);
#if D4R_RRSWIN_DEBUG_STAGE == 10
    {
        uint32_t regs[96];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nb = 0; nb < 12; ++nb) {
                regs[i++] = d10[mi][0][nb];
                regs[i++] = d10[mi][1][nb];
            }
        }
        debug_store(prm, regs, 96);
    }
#endif
    uint32_t d11[4][2][4];
    pv_gemm(d10, d9, t, g, (lds_u8*)(uintptr_t)scratch, cp, d11);
#if D4R_RRSWIN_DEBUG_STAGE == 11
    {
        uint32_t regs[32];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 4; ++nt) {
                regs[i++] = d11[mi][0][nt];
                regs[i++] = d11[mi][1][nt];
            }
        }
        debug_store(prm, regs, 32);
    }
#endif

#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 4
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 4) {
        return;
    }
#endif
    // ---- E11 and phase 12: the second window's output projection --------------
    // Unlike E6 there is no table: the residual of the first window is phase 12's C
    // operand, seeded per tile from phase 7's own pairs (contract §2's identity, VERIFIED
    // against the oracle's own C registers, debug stage p12o).  The dcell view feeds E12's
    // RMS norm (and stage 12); the pairs stay for phase 14's C.
    uint32_t as12[16];
    for (int mi = 0; mi < 4; ++mi) {
        for (int row = 0; row < 2; ++row) {
            as12[4 * mi + row] = e4m3x2(d11[mi][row][0]) | (e4m3x2(d11[mi][row][2]) << 16);
            as12[4 * mi + 2 + row] = e4m3x2(d11[mi][row][1]) | (e4m3x2(d11[mi][row][3]) << 16);
        }
    }
    uint32_t pairs12[2][4][4];
    rrswin::swin_slots_from_wimg(slots, w, 36032, 32);
    rrswin::swin_stage_fragment(img, 32, as12, 4);
    __builtin_amdgcn_wave_barrier();
    for (int nt = 0; nt < 4; ++nt) {
        for (int mt = 0; mt < 2; ++mt) {
            f8v c;
            enc0_seed_res(c, mt, nt, &pairs7[0][0][0], 4);
            c = rrswin::swin_gemm_tile(c, slots, 16 * mt, 32, img, 32, 16 * nt);
            enc0_pack(pairs12[mt][nt], c);
        }
    }
    __builtin_amdgcn_wave_barrier();
    uint32_t d12[4][2][4];
    enc0_dcell_exchange(&pairs12[0][0][0], 4, ex, &d12[0][0][0]);
#if D4R_RRSWIN_DEBUG_STAGE == 12
    {
        uint32_t regs[32];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 4; ++nt) {
                regs[i++] = d12[mi][0][nt];
                regs[i++] = d12[mi][1][nt];
            }
        }
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 112
    {
        // phase 12's mma operands: the four A fragments, the four B fragment pairs and
        // the 32 C seeds in the oracle's (m-tile, n-tile) pair order -- the C array
        // rebuilt from phase 7's debug-only dcell view
        uint32_t c12[64];
        cres_regs(d7o, c12);
        uint32_t regs[56];
        for (int o = 0; o < 16; ++o) {
            regs[o] = as12[o];
        }
        bpair_regs(w, 36032, (int)lane, regs + 16);
        cseed_regs(c12, regs + 24);
        debug_store(prm, regs, 56);
    }
#endif

#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 5
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 5) {
        return;
    }
#endif
    // ---- E12 and phases 13 to 20: the MLP ------------------------------------
    // E12's RMS norm is the same sum-of-squares tree as the LayerNorm above, on phase 12's
    // accumulators and with the gain at weight bytes 37120..37184; its epsilon is E12's own
    // add (%fd770 = 2^-13) and there is no mean subtraction and no 1/N.  Its output is the
    // shared A operand of the four branches: phases 13, 15, 17 and 19 all read the same
    // sixteen registers (%r15389..%r15420, re-packed by E14, E16 and E18).
    uint32_t asN[16];
    norm_as(d12, w, 37120, (half_t)0.0001220703125f, 4, asN);

    // ---- phase 13 and E13, the first branch -----------------------------------
    // A = the normed tile image, B = the weights at 37184, C = the per-column bias at
    // 38208.  E13's activation stages into phase 14's image as mode 2; the d13 view only
    // stages 13 and 114 need, so its exchange is debug-only.
    uint32_t pairs13[2][4][4];
    enc0_branch(asN, w, 37184, 38208, slots, img, pairs13);
#if D4R_RRSWIN_DEBUG_STAGE == 13 || D4R_RRSWIN_DEBUG_STAGE == 114
    uint32_t d13o[4][2][4];
    enc0_dcell_exchange(&pairs13[0][0][0], 4, ex, &d13o[0][0][0]);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 13
    {
        uint32_t regs[32];
        stage_regs4(d13o, regs);
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 113
    {
        uint32_t regs[28];
        for (int o = 0; o < 16; ++o) {
            regs[o] = asN[o];
        }
        bpair_regs(w, 37184, (int)lane, regs + 16);
        wbias_regs(w, 38208, (int)lane, regs + 24);
        debug_store(prm, regs, 28);
    }
#endif
    // E13's activation is phase 14's A image (mode 2 = the clamped cubic per pair,
    // bit-identical to the old act_pair + e4m3x2 pack); C is phase 12's pairs plus the
    // per-column table at 39296, added in f16 before the widen (VERIFIED, stage p14o).
    uint32_t wt14[2][4];
    enc0_load_cols(wt14, w, 39296);
    uint32_t pairs14[2][4][4];
    enc0_branch_act(pairs13, &pairs12[0][0][0], &wt14[0][0], w, 38272, slots, img, pairs14);
#if D4R_RRSWIN_DEBUG_STAGE == 14 || D4R_RRSWIN_DEBUG_STAGE == 116
    uint32_t d14o[4][2][4];
    enc0_dcell_exchange(&pairs14[0][0][0], 4, ex, &d14o[0][0][0]);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 14
    {
        uint32_t regs[32];
        stage_regs4(d14o, regs);
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 114
    {
        // the oracle's as14 pack and C array, rebuilt from the d13/d12 dcell views
        uint32_t as14[16];
        act_pack(d13o, as14);
        uint32_t c14[64];
        tab_seed(d12, w + 39296, (int)lane, c14);
        uint32_t regs[56];
        for (int o = 0; o < 16; ++o) {
            regs[o] = as14[o];
        }
        bpair_regs(w, 38272, (int)lane, regs + 16);
        cseed_regs(c14, regs + 24);
        debug_store(prm, regs, 56);
    }
#endif

    // ---- phases 15 and 16, the second branch ---------------------------------
    // The same pair with the weights at 39360 and 40448 and the bias at 40384: no table
    // add, and phase 16's C is branch one's accumulators.
    uint32_t pairs15[2][4][4];
    enc0_branch(asN, w, 39360, 40384, slots, img, pairs15);
#if D4R_RRSWIN_DEBUG_STAGE == 15 || D4R_RRSWIN_DEBUG_STAGE == 116
    uint32_t d15o[4][2][4];
    enc0_dcell_exchange(&pairs15[0][0][0], 4, ex, &d15o[0][0][0]);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 15
    {
        uint32_t regs[32];
        stage_regs4(d15o, regs);
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 115
    {
        uint32_t regs[28];
        for (int o = 0; o < 16; ++o) {
            regs[o] = asN[o];
        }
        bpair_regs(w, 39360, (int)lane, regs + 16);
        wbias_regs(w, 40384, (int)lane, regs + 24);
        debug_store(prm, regs, 28);
    }
#endif
    uint32_t pairs16[2][4][4];
    enc0_branch_act(pairs15, &pairs14[0][0][0], nullptr, w, 40448, slots, img, pairs16);
#if D4R_RRSWIN_DEBUG_STAGE == 16 || D4R_RRSWIN_DEBUG_STAGE == 118
    uint32_t d16o[4][2][4];
    enc0_dcell_exchange(&pairs16[0][0][0], 4, ex, &d16o[0][0][0]);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 16
    {
        uint32_t regs[32];
        stage_regs4(d16o, regs);
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 116
    {
        // the oracle's as16 pack and C array, rebuilt from the d15/d14 dcell views
        uint32_t as16[16];
        act_pack(d15o, as16);
        uint32_t c16[64];
        cres_regs(d14o, c16);
        uint32_t regs[56];
        for (int o = 0; o < 16; ++o) {
            regs[o] = as16[o];
        }
        bpair_regs(w, 40448, (int)lane, regs + 16);
        cseed_regs(c16, regs + 24);
        debug_store(prm, regs, 56);
    }
#endif

#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 6
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 6) {
        return;
    }
#endif
    // ---- phases 17 and 18, the third branch ----------------------------------
    uint32_t pairs17[2][4][4];
    enc0_branch(asN, w, 41472, 42496, slots, img, pairs17);
#if D4R_RRSWIN_DEBUG_STAGE == 17 || D4R_RRSWIN_DEBUG_STAGE == 118
    uint32_t d17o[4][2][4];
    enc0_dcell_exchange(&pairs17[0][0][0], 4, ex, &d17o[0][0][0]);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 17
    {
        uint32_t regs[32];
        stage_regs4(d17o, regs);
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 117
    {
        uint32_t regs[28];
        for (int o = 0; o < 16; ++o) {
            regs[o] = asN[o];
        }
        bpair_regs(w, 41472, (int)lane, regs + 16);
        wbias_regs(w, 42496, (int)lane, regs + 24);
        debug_store(prm, regs, 28);
    }
#endif
    uint32_t pairs18[2][4][4];
    enc0_branch_act(pairs17, &pairs16[0][0][0], nullptr, w, 42560, slots, img, pairs18);
#if D4R_RRSWIN_DEBUG_STAGE == 18
    uint32_t d18o[4][2][4];
    enc0_dcell_exchange(&pairs18[0][0][0], 4, ex, &d18o[0][0][0]);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 18
    {
        uint32_t regs[32];
        stage_regs4(d18o, regs);
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 118
    {
        // the oracle's as18 pack and C array, rebuilt from the d17/d16 dcell views
        uint32_t as18[16];
        act_pack(d17o, as18);
        uint32_t c18[64];
        cres_regs(d16o, c18);
        uint32_t regs[56];
        for (int o = 0; o < 16; ++o) {
            regs[o] = as18[o];
        }
        bpair_regs(w, 42560, (int)lane, regs + 16);
        cseed_regs(c18, regs + 24);
        debug_store(prm, regs, 56);
    }
#endif

    // ---- phases 19 and 20, the fourth branch ---------------------------------
    uint32_t pairs19[2][4][4];
    enc0_branch(asN, w, 43584, 44608, slots, img, pairs19);
#if D4R_RRSWIN_DEBUG_STAGE == 19
    uint32_t d19o[4][2][4];
    enc0_dcell_exchange(&pairs19[0][0][0], 4, ex, &d19o[0][0][0]);
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 19
    {
        uint32_t regs[32];
        stage_regs4(d19o, regs);
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 119
    {
        uint32_t regs[28];
        for (int o = 0; o < 16; ++o) {
            regs[o] = asN[o];
        }
        bpair_regs(w, 43584, (int)lane, regs + 16);
        wbias_regs(w, 44608, (int)lane, regs + 24);
        debug_store(prm, regs, 28);
    }
#endif
    uint32_t pairs20[2][4][4];
    enc0_branch_act(pairs19, &pairs18[0][0][0], nullptr, w, 44672, slots, img, pairs20);
    // E20's r20 build walks the NVIDIA d[mi][row][nt] view
    uint32_t d20[4][2][4];
    enc0_dcell_exchange(&pairs20[0][0][0], 4, ex, &d20[0][0][0]);
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 8
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 8) {
        return;
    }
#endif

#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 7
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 7) {
        return;
    }
#endif
    // ---- E20 and phases 21 and 22: the merge ---------------------------------
    // E20 requantises phase 20 into sixteen packed words (r20, the four A fragments of
    // phase 22), scatters them into the plane arena, and merges the lanes of each group of
    // eight back into phase 21's four A fragments.
    uint32_t r20[16];
    for (int mi = 0; mi < 4; ++mi) {
        for (int row = 0; row < 2; ++row) {
            const uint32_t a0 = e4m3x2(d20[mi][row][0]) | (e4m3x2(d20[mi][row][2]) << 16);
            const uint32_t a1 = e4m3x2(d20[mi][row][1]) | (e4m3x2(d20[mi][row][3]) << 16);
            r20[4 * mi + row] = a0;
            r20[4 * mi + 2 + row] = a1;
        }
    }
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 9
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 9) {
        return;
    }
#endif
    arena_scatter(prm, r20, (int)lane);
    uint32_t m21[16];
    merge_frag(r20, scratch, (int)lane, m21);
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 10
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 10) {
        return;
    }
#endif
    uint32_t d21[8][2];
    band21(m21, w, w + 53888, (int)lane, d21);
#if D4R_RRSWIN_KEEP21 == 1
    // band21 live, but no E21 store: d21's only real consumer is the scatter, so without
    // this the whole phase-21 band gemm is dead-code-eliminated
    if (lane == 0x7ffffffu) {
        out[0] = d21[0][0];
        out[1] = d21[7][1];
        out[2] = d21[3][0];
        return;
    }
#endif

    // ---- E21 and phase 22 ---------------------------------------------------
    // E21 requantises phase 21 and scatters it into the buffer at parameter block +56;
    // phase 22 takes the sixteen words E20 packed as its A operand (four m-tiles, K=32)
    // with the weights at 54016 and the two bias words at 54528.
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT == 20
    // capture: the param block's words (arena/extents/buffer pointers/geometry) handed back
    // through the 18-word cut ABI, so the launch dump's arg112 surface bytes carry them
    for (int i = 0; i < 18; ++i) {
        out[i] = *(const uint32_t*)(prm + 4 * i);
    }
    return;
#endif
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 12
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 12) {
        return;
    }
#endif
    surface_scatter21(prm, d21, (int)lane);
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 14
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 14) {
        return;
    }
#endif
    uint32_t d22[4][2][4];
    gemm(r20, w, 54016, w + 54528, nullptr, nullptr, 4, t, d22, 2);
#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 11
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 11) {
        return;
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 20
    {
        uint32_t regs[32];
        stage_regs4(d20, regs);
        debug_store(prm, regs, 32);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 21
    {
        uint32_t regs[16];
        for (int nt = 0; nt < 8; ++nt) {
            regs[2 * nt] = d21[nt][0];
            regs[2 * nt + 1] = d21[nt][1];
        }
        debug_store(prm, regs, 16);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 22
    {
        uint32_t regs[16];
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 2; ++nt) {
                regs[i++] = d22[mi][0][nt];
                regs[i++] = d22[mi][1][nt];
            }
        }
        debug_store(prm, regs, 16);
    }
#endif

#if defined(D4R_RRSWIN_BISECT) && D4R_RRSWIN_BISECT >= 13
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    for (int i = 2; i < 18; ++i) out[i] = 0;
    if (D4R_RRSWIN_BISECT == 13) {
        return;
    }
#endif
    // ---- handing the accumulators back ------------------------------------
    // %r17597 is param_0[0] (ld.param.v2.u32 at line 44316) and %r18203 is %tid.x
    // (line 44456); both sit inside the cut, so the epilogue cannot recompute them.
    // The other sixteen are phase 22's eight mma D pairs in mma order.
    out[0] = *(const uint32_t*)prm;
    out[1] = (uint32_t)__builtin_amdgcn_workitem_id_x();
    int i = 2;
    for (int mi = 0; mi < 4; ++mi) {
        for (int nt = 0; nt < 2; ++nt) {
            out[i++] = d22[mi][0][nt];
            out[i++] = d22[mi][1][nt];
        }
    }
}

// ---------------------------------------------------------------------------
// Weight image layout of the 22 phases (byte ranges of arg040, from the PTX's own
// ld.weak.global loads; 512 bytes per B-fragment pair, K=32, N=32):
//   P1  [4096,5120)   bias 5120..5184      P12 [36032,37056)
//   P2  [5184,6208)   bias 6208..6272      P13 [37184,38208)  bias 38208..38272
//   P3  [6336,7360)                        P14 [38272,39296)
//   P4  [7360,8384)                        P15 [39360,40384)  bias 40384..40448
//   P5  bias [8384,20672) (64x96 f16)      P16 [40448,41472)
//   P6  none (P V, B from mma output)      P17 [41472,42496)  bias 42496..42560
//   P7  [20672,21696)                      P18 [42560,43584)
//   P8  [21696,22720)                      P19 [43584,44608)  bias 44608..44672
//   P9  [22720,23744)                      P20 [44672,45696)
//   P10 bias [23744,36032) (64x96 f16)     P21 [45696,53888)  bias 53888..54016
//   P11 none (P V)                         P22 [54016,54544)  bias 54528..54560
//
// Fragment maps (g = lane>>2, t = lane&3). The corrected full-lane phase-8
// model matches all 3072 f16 outputs of captured block 0. The checked native
// pilot matches 196608 f16 outputs over 64 captured blocks. Its former
// mismatch came from invalid accumulator bounds and device/host block strides,
// not from the fragment map. The software fallback's different fragment
// convention is not the active gfx12 FP8 WMMA reference.
// The hardware order of an m16n8k32 .e4m3 fragment pair is *not* the order the
// PTX ISA figures suggest. NVIDIA's own epilogue (stmts 2256-2327) settles it: it
// builds phase 2's A operand as
//     %r1207 = {cvt(%r877), cvt(%r897)}
// where %r877/%r897 are the D halves of mma rows 2231/2233, i.e. (row g, n-tile 0,
// cols 2t, 2t+1) and (row g, n-tile 2, cols 2t, 2t+1). So with the four bytes of
// an A register in memory order:
//   A reg 0 -> row g,     k = 2t + (j&1) + 16*((j>>1)&1)
//   A reg 1 -> row g+8,   same k
//   A reg 2 -> row g,     k = 8 + 2t + (j&1) + 16*((j>>1)&1)
//   A reg 3 -> row g+8,   same k
// and the B half that must pair with it (a0/a1 with b0, a2/a3 with b1) is
//   B byte j (j<4)  -> n = 16*tt + 8*(b>>3) + g, k = 2t + (j&1) + 16*((j>>1)&1)
//   B byte j (j>=4) -> n = 16*tt + 8*(b>>3) + g, k = 8 + 2t + (j&1) + 16*((j>>1)&1)
// with b the byte index within the lane's 16 and tt the 512-byte tile. Inverted,
// that is waddr() above. C/D:
//   C/D reg 0 -> row g,   cols t*2, t*2+1
//   C/D reg 1 -> row g+8, cols t*2, t*2+1
//
// A plain f32 dot product over K plus C, rounded to f16 once, reproduces the
// translated kernel's phase-1 output exactly (96x32 elements, 4 blocks x 32 lanes,
// 0 mismatches out of 12288), so the phases need no operand-layout gymnastics and
// no WMMA intrinsics: every GEMM here is exact in f32 before the f16 rounding.
// The textbook order (reg 0 = row g, k = 4t..4t+3) also reproduces phase 1, but
// only because it is a self-consistent relabeling of k on *both* sides; it cannot
// be used for A operands that an epilogue produced, which arrive in the hardware
// order above.
