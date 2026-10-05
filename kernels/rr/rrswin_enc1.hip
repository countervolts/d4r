// Native replacement for the Ray Reconstruction denoiser's second Swin encoder,
// cuda_dldn_engine_swin_enc1_kernel (nvngx_dlssd.dll 310.7).
//
// ---------------------------------------------------------------------------
// STATE: the recipe (kernels/rr/swin_enc1_recipe.py) cuts statements s8270-s26139
// out of the entry and calls this body instead, handing over the prologue's two
// staged A operand sets, the zero register phase 1 uses as its C seed and the
// dequantised input tile E6 needs for phase 7's residual.  The window implemented
// here is phases 1-7: the four 96x32 projections, the 64x96 score, P*V and the
// 64x64 output projection.  Phase 32's accumulators are still placeholders, so the
// epilogue's surface output differs from the translated kernel; every implemented
// stage is bit-exact against the debug oracle (see "Ground truth" below).
//
// The kernel is one warp per 8x8 token patch: grid (81, 47, 1), block (32, 1, 1)
// with one wave per block (the prologue also handles tid.y > 0, for waves > 1, but
// the shipped manifest launches one), parameter block 144 bytes.  The PTX this
// file replaces is kernels/extracted/rr/0014-PREPASS_ENTRYPOINT_NAME.ptx.
//
// Linked into ZLUDA's compile of that PTX (D4R_ZLUDA_EXTRA_BC), so tex_common.h's
// helpers and ZLUDA's own device functions are available.  Compiled with -nogpuinc,
// so only plain clang builtins and attributes are used.
#include "../tex/tex_common.h"

// ---------------------------------------------------------------------------
// Parameter block.  Fields are addressed by their byte offset in the PTX, so the
// raw image is kept and read through a typed accessor.
struct DldnSwinEnc1Params
{
    uint8_t raw[144];
};
static_assert(sizeof(DldnSwinEnc1Params) == 144, "parameter block");

enum : int
{
    kWeights = 40,   // NVIDIA's prepared weight image (arg040), read-only
    kPlaneArg = 48,  // plane arena pointer, as make_ptx.py's debug splice reads it
    kExtents = 0,    // ex, ey (v2.u32)
};

#define DEVF __attribute__((device)) static inline
typedef float lds_f32 __attribute__((address_space(3)));
typedef uint32_t lds_u32 __attribute__((address_space(3)));
typedef uint8_t lds_u8 __attribute__((address_space(3)));

// ---------------------------------------------------------------------------
// The implemented phases, as byte offsets into arg040 (from the PTX's own loads;
// K=32, 512 bytes per group of 16 columns):
//   P1  weights 128    P2  weights 2176   P3  weights 1152   P4  weights 3200
//   P5  score bias 4224..16512 (24 tiles of 512)
//   P6  none (P*V; B is the movmatrix transpose of phase 4's D)
//   P7  weights 16512..18560, per-column table 36992..37120 (8 words of 16 bytes)

// e4m3 code -> f32, exact.  The mma's e4m3 operands reach the FMA path through
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
// (VERIFIED on all 1024 (n, k) pairs of a tile for enc0; the B operand layout is
// the hardware's, so the same inversion applies here.)
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

// the low/high f16 of a packed pair (columns 2t / 2t+1)
DEVF half_t half_lo(uint32_t pair)
{
    return __builtin_bit_cast(half_t, (uint16_t)pair);
}

DEVF half_t half_hi(uint32_t pair)
{
    return __builtin_bit_cast(half_t, (uint16_t)(pair >> 16));
}

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
// One GEMM: `nmi` m-tiles of 16 rows x `nti` n-tiles of 8 columns, K=32, f16 out.
//
// A lane's A registers hold, for each m-tile and each of its two rows, the eight
// e4m3 values of k = 2t + (j&1) + 16*((j>>1)&1) (byte j = 0..3) and of k + 8; the
// four lanes that differ in t hold the four disjoint eighths of the 32 k's, so two
// butterfly steps broadcast the other three eighths in.  The B operand is read from
// the weight image (waddr) for the lane's own two columns of each n-tile.  The dot
// product runs in f32 and is rounded to f16 once, which is what the translated mma
// does: e4m3 x e4m3 products are exact and 32 of them sum exactly in f32.
//
// The C operand is the 32-bit seed the PTX's mma adds (one packed f16 pair per D
// register, indexed [m-tile][row half][n-tile] as `cmat`) or a constant (`fixed`).
// d is [m-tile][row half][n-tile], each entry the packed f16 pair 2t, 2t+1.
DEVF void gemm(const uint32_t* as, const uint8_t* w, int base, uint32_t fixed,
               const uint32_t* cmat, int nmi, int nti, int t, uint32_t d[6][2][8])
{
    for (int nt = 0; nt < nti; ++nt) {
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
                const uint32_t seed = (cmat != nullptr) ? cmat[(4 * mi + row) * nti + nt] : fixed;
                d[mi][row][nt] = pair_of((half_t)(acc[row][0] + h2f(seed)),
                                         (half_t)(acc[row][1] + h2f(seed >> 16)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the f16 the PTX's literals carry: 0x2466, 0xb873, 0x3873, 0x3b6b, 0x3d80
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

// ZLUDA's own rcp.approx.ftz.f32 helper: calling it is what makes the reciprocal
// bit-exact.
extern "C" __attribute__((device)) float __zluda_ptx_impl_rcp_approx_f32(float);

// the e4m3 code of one f16 (cvt.rn.satfinite.e4m3x2.f16x2 of a pair of equal halves)
DEVF uint32_t q8(half_t v)
{
    return e4m3x2(pair_of(v, v));
}

// the f16 at (row, col) of a 96x32 (or 64x32) block of packed pairs,
// col = 8*n-tile + 2t + c
DEVF half_t dcell(const uint32_t d[6][2][8], int row, int col)
{
    const uint32_t p = d[row >> 4][(row >> 3) & 1][col >> 3];
    return (col & 1) ? half_hi(p) : half_lo(p);
}

// the lane's two e4m3-packed words for row r: word 0 packs n-tiles 0 and 2, word 1
// packs tiles 1 and 3 (the mma's own k order)
DEVF void row_words(const uint32_t d[6][2][8], int r, uint32_t w[2])
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

// E5(a), one element: the scale, the clamp, the two clamped-cubic fma pairs and the
// trick that moves the mantissa into the exponent field ((u << 5) & 0x7FE07FE0).
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

// E5(a)-(e), one row: the twelve exponent pairs, the in-lane sum tree, the two
// butterfly steps that finish the row's 96 columns, the two halves' add, the
// reciprocal and the twelve products.
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
    // the PTX broadcasts the f16 sum of the two halves (add.f16) and takes cvt.f32.f16
    // of it: the value, not its bits
    const half_t r = (half_t)__zluda_ptx_impl_rcp_approx_f32((float)(half_lo(p) + half_hi(p)));
    const uint32_t rcp = pair_of(r, r);
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        prob[2 * i] = u32_of(ev[i] * hv2_of(rcp));
        prob[2 * i + 1] = u32_of(eo[i] * hv2_of(rcp));
    }
}

// ---------------------------------------------------------------------------
// Ground truth for a stage comes from kernels/tex/make_ptx.py's debug recipe:
// with D4R_RRSWIN_DEBUG=<stage> the NVIDIA body keeps every mma and stores that
// stage's registers into the plane arena (per block 0..63, slot by register count,
// 4 bytes per lane), and the launch dump then holds the translated values.  Built
// with -DD4R_RRSWIN_DEBUG_STAGE=<n> (kernels/build.sh rrswin, D4R_TEX_CFLAGS), this
// body stores the same registers of its own phase n in the same order, so the two
// dumps compare byte for byte.
//
// The offset into the arena is D4R_RRSWIN_DEBUG_BASE on the oracle side (read by
// make_ptx.py's swin_debug_block) and D4R_RRSWIN_DEBUG_BASE_NUM here.  Two measured
// facts fix its value for enc1:
//
//  * enc1's param_0+48 is not dumped as arg048.  The launch manifest puts it at byte
//    45,219,840 of the 65,945,600-byte block the bridge dumps as arg008
//    (`pointer 8 0 30146560`, `pointer 48 0 45219840`, `pointer 56 0 60293120`), so
//    the bridge skips arg048 as a duplicate allocation base.  The payload therefore
//    lives in the arg008 dump at 45,219,840 + base, and enc0's default of 46,000,000
//    would fall 25 MB past the end of the block.
//  * enc1's own 44 `st.global` fill that block's whole 20,725,760-byte tail (measured
//    on an isolated launch: the translated kernel changes every byte of it), so a base
//    inside the tail is overwritten by the kernel's own later stores and the comparison
//    silently compares ambient memory.  The last byte E31 reaches is arena + 16,957,436,
//    and the block ends at arena + 20,725,760.
//
// 17,006,592 is a multiple of the largest slot (16,384), clears E31, and leaves room
// for all 64 slots (1,048,576 bytes) inside the block.
#ifndef D4R_RRSWIN_DEBUG_STAGE
#define D4R_RRSWIN_DEBUG_STAGE 0
#endif
#ifndef D4R_RRSWIN_DEBUG_BASE_NUM
#define D4R_RRSWIN_DEBUG_BASE_NUM 17006592
#endif

DEVF void debug_store(const uint8_t* prm, const uint32_t* regs, int n)
{
    // make_ptx.py's swin_debug_block forms the block index as ctaid.x + 161*ctaid.y
    const uint32_t block = __builtin_amdgcn_workgroup_id_x()
                         + 161u * __builtin_amdgcn_workgroup_id_y();
    if (block >= 64u) {
        return;   // the oracle's per-block slots only cover the first 64 blocks
    }
    const uint64_t arena = *(const uint64_t*)(prm + kPlaneArg);
    // the oracle's slot: 32 lanes x n x 4 bytes, at least 8192, rounded up to a power of two
    const uint64_t want = (uint64_t)n * 128ull > 8192ull ? (uint64_t)n * 128ull : 8192ull;
    uint64_t slot = 1;
    while (slot < want) {
        slot <<= 1;
    }
    uint32_t* dst = (uint32_t*)(arena + (uint64_t)D4R_RRSWIN_DEBUG_BASE_NUM
                                 + (uint64_t)block * slot
                                 + (uint64_t)lane_id() * (uint64_t)n * 4ull);
    for (int i = 0; i < n; ++i) {
        dst[i] = regs[i];
    }
}

// stage registers in the oracle's order: m-tile major, n-tile minor, each as the
// (row g, row g+8) pair the mma writes.
DEVF void stage_regs(const uint32_t d[6][2][8], int nmi, int nti, uint32_t regs[96])
{
    int i = 0;
    for (int mi = 0; mi < nmi; ++mi) {
        for (int nt = 0; nt < nti; ++nt) {
            regs[i++] = d[mi][0][nt];
            regs[i++] = d[mi][1][nt];
        }
    }
}

// ---------------------------------------------------------------------------
// The native body.
//
// Live-ins (recipes's order): AS1[24] (phases 1/2/8/9), AS2[24] (phases 3/4/10/11),
// the phase-1 C seed (the constant zero register), RES[64] (the dequantised input
// tile in (m-tile, n-tile, row half) order), the weight image, the parameter block
// and the address of the LDS scratch the accumulators go back through.
#define AS(i)  a##i
extern "C" __attribute__((device)) void d4r_swin_enc1(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6,
    uint32_t a7, uint32_t a8, uint32_t a9, uint32_t a10, uint32_t a11, uint32_t a12, uint32_t a13,
    uint32_t a14, uint32_t a15, uint32_t a16, uint32_t a17, uint32_t a18, uint32_t a19, uint32_t a20,
    uint32_t a21, uint32_t a22, uint32_t a23, uint32_t a24, uint32_t a25, uint32_t a26, uint32_t a27,
    uint32_t a28, uint32_t a29, uint32_t a30, uint32_t a31, uint32_t a32, uint32_t a33, uint32_t a34,
    uint32_t a35, uint32_t a36, uint32_t a37, uint32_t a38, uint32_t a39, uint32_t a40, uint32_t a41,
    uint32_t a42, uint32_t a43, uint32_t a44, uint32_t a45, uint32_t a46, uint32_t a47,
    uint32_t cseed,
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4, uint32_t r5, uint32_t r6,
    uint32_t r7, uint32_t r8, uint32_t r9, uint32_t r10, uint32_t r11, uint32_t r12, uint32_t r13,
    uint32_t r14, uint32_t r15, uint32_t r16, uint32_t r17, uint32_t r18, uint32_t r19, uint32_t r20,
    uint32_t r21, uint32_t r22, uint32_t r23, uint32_t r24, uint32_t r25, uint32_t r26, uint32_t r27,
    uint32_t r28, uint32_t r29, uint32_t r30, uint32_t r31, uint32_t r32, uint32_t r33, uint32_t r34,
    uint32_t r35, uint32_t r36, uint32_t r37, uint32_t r38, uint32_t r39, uint32_t r40, uint32_t r41,
    uint32_t r42, uint32_t r43, uint32_t r44, uint32_t r45, uint32_t r46, uint32_t r47, uint32_t r48,
    uint32_t r49, uint32_t r50, uint32_t r51, uint32_t r52, uint32_t r53, uint32_t r54, uint32_t r55,
    uint32_t r56, uint32_t r57, uint32_t r58, uint32_t r59, uint32_t r60, uint32_t r61, uint32_t r62,
    uint32_t r63,
    const uint8_t* w, const uint8_t* prm, uint32_t scratch)
{
    const uint32_t as1[24] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                              a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23};
    const uint32_t as2[24] = {a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35,
                              a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47};
    const uint32_t res[64] = {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
                              r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29,
                              r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43,
                              r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57,
                              r58, r59, r60, r61, r62, r63};
#if D4R_RRSWIN_DEBUG_STAGE
    // The oracle's payload store is spliced in after E31's last arena write (see
    // swin_enc1_debug_stages.LINE), because enc1 fills its whole plane-arena tail, so a
    // store placed beside its own phase is overwritten before the dump reads it back.
    // This side therefore fills `cap` where each stage's registers are live and hands
    // the whole thing to debug_store() once, at the end of the body -- after this
    // body's own E30/E31 stores, which is the same program point.
    uint32_t cap[128];
    int capn = 0;
#if D4R_RRSWIN_DEBUG_STAGE == 10
    // the oracle's stage a: the two staged A operand sets, the phase-1 C seed and
    // the three shfl constants the PTX carries (1, 2, -1)
    for (int i = 0; i < 24; ++i) {
        cap[i] = as1[i];
        cap[24 + i] = as2[i];
    }
    cap[48] = cseed;
    cap[49] = 1u;
    cap[50] = 2u;
    cap[51] = 0xffffffffu;
    capn = 52;
#endif
#endif
    (void)cseed;   // phase 1's C operand is this register's value, the constant zero
    const int t = (int)(lane_id() & 3);   // the lane's two columns of each n-tile
    const int g = (int)(lane_id() >> 2);  // ... and of its sixteen rows

    lds_u32* const slot = (lds_u32*)(uintptr_t)scratch;   // the phase 5/6 exchange

    // ---- the four 96x32 projections ----------------------------------------
    // Phases 1 and 2 have C = 0 (the PTX hands in the zero register); phases 3 and
    // 4 add phase 1's / phase 2's D back in as their C seed.
    uint32_t d1[6][2][8];
    uint32_t d2[6][2][8];
    gemm(as1, w, 128, 0u, nullptr, 6, 4, t, d1);
    gemm(as1, w, 2176, 0u, nullptr, 6, 4, t, d2);
#if D4R_RRSWIN_DEBUG_STAGE == 1
    stage_regs(d1, 6, 4, cap);
    capn = 48;
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 2
    stage_regs(d2, 6, 4, cap);
    capn = 48;
#endif
    uint32_t d3[6][2][8];
    uint32_t d4[6][2][8];
    {
        uint32_t c1[96], c2[96];
        for (int mi = 0; mi < 6; ++mi) {
            for (int nt = 0; nt < 4; ++nt) {
                for (int rh = 0; rh < 2; ++rh) {
                    c1[(4 * mi + rh) * 4 + nt] = d1[mi][rh][nt];
                    c2[(4 * mi + rh) * 4 + nt] = d2[mi][rh][nt];
                }
            }
        }
        gemm(as2, w, 1152, 0u, c1, 6, 4, t, d3);
        gemm(as2, w, 3200, 0u, c2, 6, 4, t, d4);
    }
#if D4R_RRSWIN_DEBUG_STAGE == 3
    stage_regs(d3, 6, 4, cap);
    capn = 48;
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 4
    stage_regs(d4, 6, 4, cap);
    capn = 48;
#endif

    // ---- phase 5, the score: Q.K^T over the patch and its halo ---------------
    // A and B are both phase 3's accumulators.  The lane's rows 16*mi + g + 8*rh are the
    // queries (the 8x8 token patch) and its rows 16*(nb>>1) + 8*(nb&1) + g the keys (the
    // patch plus a 32-row halo): the PTX builds n-tile nb's B fragment from phase 3's
    // m-tile nb>>1 and row half nb&1, whose two words are the k-groups {0, 2} and {1, 3}
    // of the 32 columns.  row_words() packs a row that way.
    //
    // A lane holds one row group of the queries and one of the keys, so it can only dot
    // them with each other; every lane therefore publishes its two key words per n-tile
    // (12 x 2 x 32 lanes = 768 words = 3072 bytes, the whole scratch) and then forms the
    // dots its own D registers hold.
    //
    // The C operand is the 64x96 f16 score bias at weight bytes 4224..16512: 24 tiles of
    // 512 bytes, the seed of (m-tile, n-tile, row half) at byte
    // 4224 + 512*(mi + 4*(nb>>1)) + 16*lane + 4*(2*(nb&1) + row).
    uint32_t d5[4][2][12];
    {
#pragma unroll
        for (int nb = 0; nb < 12; ++nb) {
            uint32_t wd[2];
            row_words(d3, 16 * (nb >> 1) + 8 * (nb & 1) + g, wd);
            slot[64 * nb + lane_id()] = wd[0];
            slot[64 * nb + 32 + lane_id()] = wd[1];
        }
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int row = 0; row < 2; ++row) {
                uint32_t aw[2], a[2][4];
                row_words(d3, 16 * mi + g + 8 * row, aw);
                bfly2x4(aw, a);
#pragma unroll
                for (int nb = 0; nb < 12; ++nb) {
                    float s[2];
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        uint32_t kb[2][4];
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
#pragma unroll
                            for (int m = 0; m < 4; ++m) {
                                kb[i][m] = slot[64 * nb + 32 * i + 4 * (2 * t + c) + (t ^ m)];
                            }
                        }
                        s[c] = dot32(a, kb);
                    }
                    const uint32_t bias = *(const uint32_t*)(w + 4224 + 512 * (mi + 4 * (nb >> 1))
                                                  + 16 * lane_id() + 4 * (2 * (nb & 1) + row));
                    d5[mi][row][nb] = pair_of((half_t)(s[0] + h2f(bias)),
                                              (half_t)(s[1] + h2f(bias >> 16)));
                }
            }
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 8
    // the oracle's stage p5o: the 96 C words (the score bias, tile-major: 24 tiles
    // of four words per lane) and the 24 B words (the two e4m3 key words of each of
    // the twelve n-tiles) exactly as the mma reads them.
#pragma unroll
    for (int ti = 0; ti < 24; ++ti) {
        for (int wd = 0; wd < 4; ++wd) {
            cap[4 * ti + wd] = *(const uint32_t*)(w + 4224 + 512 * ti + 16 * lane_id() + 4 * wd);
        }
    }
#pragma unroll
    for (int nb = 0; nb < 12; ++nb) {
        cap[96 + 2 * nb] = slot[64 * nb + lane_id()];
        cap[96 + 2 * nb + 1] = slot[64 * nb + 32 + lane_id()];
    }
    capn = 120;
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 5
    {
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nb = 0; nb < 12; ++nb) {
                cap[i++] = d5[mi][0][nb];
                cap[i++] = d5[mi][1][nb];
            }
        }
        capn = 96;
    }
#endif

    // ---- E5 and phase 6: the softmax and P.V --------------------------------
    // row_probs() gives the twelve probability pairs of one row; k-step ks packs key
    // columns {4ks, 4ks+2} into the fragment's first word and {4ks+1, 4ks+3} into its
    // second.  The lane's A fragment is its own two rows (16*mi + g and 16*mi + g + 8),
    // so the D row it accumulates is its own g; what it lacks is V.  The key rows
    // 32*ks + kk live in the lanes whose g is kk % 8, and every lane needs all 96 of
    // them, so V goes through the scratch as one e4m3 byte per element (96 x 32 = 3072
    // bytes) and each lane then walks the full k range.  Each k-step rounds to f16
    // before the next one adds to it, which is why the chain has three separate mma.
    uint32_t d6[4][2][8];
    {
#pragma unroll
        for (int mi4 = 0; mi4 < 6; ++mi4) {
            for (int rh4 = 0; rh4 < 2; ++rh4) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    for (int c = 0; c < 2; ++c) {
                        const int r = 16 * mi4 + 8 * rh4 + g;
                        ((lds_u8*)(uintptr_t)scratch)[r * 32 + 8 * nt + 2 * t + c] =
                            (uint8_t)q8(dcell(d4, r, 8 * nt + 2 * t + c));
                    }
                }
            }
        }
    }
    __builtin_amdgcn_wave_barrier();
    {
        const lds_u8* const vs = (const lds_u8*)(uintptr_t)scratch;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
                uint32_t prob[12], aw[3][2];
                row_probs(d5, mi, rh, prob);
#pragma unroll
                for (int ks = 0; ks < 3; ++ks) {
                    aw[ks][0] = e4m3x2(prob[4 * ks]) | (e4m3x2(prob[4 * ks + 2]) << 16);
                    aw[ks][1] = e4m3x2(prob[4 * ks + 1]) | (e4m3x2(prob[4 * ks + 3]) << 16);
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
                            acc0 += pv * e4m3_to_f32(vs[r]);
                            acc1 += pv * e4m3_to_f32(vs[r + 1]);
                        }
                        prev0 = (half_t)(acc0 + (float)prev0);
                        prev1 = (half_t)(acc1 + (float)prev1);
                    }
                    d6[mi][rh][nt] = pair_of(prev0, prev1);
                }
            }
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 6
    // the oracle's stage p6: the third (last) k-step of each chain, m-tile major,
    // n-tile minor, the (row g, row g+8) pair
    {
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 4; ++nt) {
                cap[i++] = d6[mi][0][nt];
                cap[i++] = d6[mi][1][nt];
            }
        }
        capn = 32;
    }
#endif

    // ---- E6 and phase 7, the output projection ------------------------------
    // A is phase 6's D packed as before; C is the pipeline's residual + the
    // per-column table at weight bytes 36992..37120 (8 words of two f16 per n-tile,
    // the lane's own column pair 2t, 2t+1).
    uint32_t as7[16];
    for (int mi = 0; mi < 4; ++mi) {
        for (int row = 0; row < 2; ++row) {
            as7[4 * mi + row] = e4m3x2(d6[mi][row][0]) | (e4m3x2(d6[mi][row][2]) << 16);
            as7[4 * mi + 2 + row] = e4m3x2(d6[mi][row][1]) | (e4m3x2(d6[mi][row][3]) << 16);
        }
    }
    uint32_t c7[64];        // [(4*m-tile + row half)][n-tile]
    for (int mi = 0; mi < 4; ++mi) {
        for (int nt = 0; nt < 8; ++nt) {
            const uint32_t tab = *(const uint32_t*)(w + 36992 + 16 * nt + ((lane_id() << 2) & 12));
            for (int row = 0; row < 2; ++row) {
                const uint32_t rr = res[(mi * 8 + nt) * 2 + row];
                c7[(4 * mi + row) * 8 + nt] = pair_of(
                    (half_t)(half_lo(tab) + half_lo(rr)), (half_t)(half_hi(tab) + half_hi(rr)));
            }
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 9
    // the oracle's stage e6: phase 7's 16 A registers and its 64 C seeds
    {
        int i = 0;
        for (int mi = 0; mi < 4; ++mi) {
            for (int o = 0; o < 4; ++o) {
                cap[i++] = as7[4 * mi + o];
            }
        }
        for (int mi = 0; mi < 4; ++mi) {
            for (int nt = 0; nt < 8; ++nt) {
                cap[i++] = c7[(4 * mi + 0) * 8 + nt];
                cap[i++] = c7[(4 * mi + 1) * 8 + nt];
            }
        }
        capn = 80;
    }
#endif
    uint32_t d7[6][2][8];
    gemm(as7, w, 16512, 0u, c7, 4, 8, t, d7);
#if D4R_RRSWIN_DEBUG_STAGE == 7
    stage_regs(d7, 4, 8, cap);
    capn = 64;
#endif

    // ---- handing the accumulators back --------------------------------------
    // Phase 32's eight mma D pairs are not implemented yet.  Until they are, fill the
    // sixteen slots with lane-distinct values so the read-back path is observable
    // without producing wild magnitudes (the epilogue's surface output changes from
    // the known-bad baseline).  64 bytes apart per lane, inside the 8 KiB scratch.
    lds_u32* const out = (lds_u32*)(uintptr_t)scratch + 16 * (uint32_t)lane_id();
    for (int i = 0; i < 16; ++i) {
        out[i] = (uint32_t)lane_id() + 32u * (uint32_t)i;
    }
#if D4R_RRSWIN_DEBUG_STAGE
    // the payload goes out here, after every store this body makes: the oracle's store
    // is spliced in at the same program point (after E31's last arena write)
    debug_store(prm, cap, capn);
#endif
}

// ---------------------------------------------------------------------------
// Weight image layout of the implemented phases (byte ranges of arg040, from the
// PTX's own ld.weak.global loads; 512 bytes per group of 16 columns, K=32):
//   P1  [128,1152)    P2  [2176,3200)    P3  [1152,2176)    P4  [3200,4224)
//   P5  bias [4224,16512) (24 tiles of 512, a 64x96 f16 block)
//   P6  none (P.V: B is the movmatrix transpose of phase 4's D)
//   P7  [16512,18560)  column table [36992,37120)
//
// Fragment maps (the hardware's, identical to enc0's, VERIFIED against the debug
// oracle there; g = lane>>2, t = lane&3):
//   A reg 0 -> row g,     k = 2t + (j&1) + 16*((j>>1)&1)
//   A reg 1 -> row g+8,   same k
//   A reg 2 -> row g,     k = 8 + 2t + (j&1) + 16*((j>>1)&1)
//   A reg 3 -> row g+8,   same k
//   B byte j (j<4)  -> n = 16*tt + 8*(b>>3) + g, k = 2t + (j&1) + 16*((j>>1)&1)
//   B byte j (j>=4) -> same n, k = 8 + 2t + (j&1) + 16*((j>>1)&1)
//   C/D reg 0 -> row g,   cols t*2, t*2+1
//   C/D reg 1 -> row g+8, cols t*2, t*2+1
