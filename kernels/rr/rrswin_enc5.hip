// Native replacement for the Ray Reconstruction denoiser's last Swin encoder,
// cuda_dldn_engine_swin_enc5_kernel (nvngx_dlssd.dll 310.7).
//
// ---------------------------------------------------------------------------
// STATE: the mirror-cut frame is in place and builds/serves. Phases 1-12 are
// implemented -- the block's 96x64 input GEMM (two chains of five k-steps) and
// the self-attention (the 64x96 score and P*V) -- and each is bit-exact against
// the debug oracle; see "Ground truth" below. Phases 13-61 are not implemented
// yet: the eight 16x160 projections, the RMS norm, the twenty banded MLP GEMMs,
// their clamped-cubic activation and the plane-arena scatter, so the layer's
// result is not yet the reference's.
//
// The kernel is 8x8 token patches: grid (5,3,1) x block (32,1,8) -- one warp per
// %tid.z group, eight warps per block -- 15 blocks, parameter block 144 bytes.
// The PTX this file replaces is ~/.cache/d4r-rr-corpus/0018-PREPASS_ENTRYPOINT_NAME.ptx,
// entry lines 11883-50752 (kernels/rr/rrswin_enc5_epilogues.md).
//
// kernels/tex/make_ptx.py's recipe (kernels/rr/swin_enc5_recipe.py) keeps
// NVIDIA's input staging (lines 1-11882) and the whole surface epilogue (from
// line 50756) and replaces the 1306 mma of the 61 phases with the call below.
//
// Linked into ZLUDA's compile of that PTX (D4R_ZLUDA_EXTRA_BC), so tex_common.h's
// helpers and ZLUDA's own device functions are available. Compiled with -nogpuinc,
// so only plain clang builtins and attributes are used.
#include "../tex/tex_common.h"

// ---------------------------------------------------------------------------
// Parameter block. Fields are addressed by their byte offset in the PTX, so the
// raw image is kept and read through a typed accessor.
struct DldnSwinEnc5Params
{
    uint8_t raw[144];
};
static_assert(sizeof(DldnSwinEnc5Params) == 144, "parameter block");

// the payload's offset into the plane arena, shared with the oracle's splice
// (kernels/rr/swin_enc5_recipe.py reads the same environment variable)
#ifndef D4R_RRSWIN_DEBUG_BASE
#define D4R_RRSWIN_DEBUG_BASE 4000000
#endif

enum : int
{
    kOutPlane = 48,   // plane arena the encoder writes and the debug store uses (arg048)
    kWeights = 40,    // NVIDIA's prepared weight image (arg040), read-only
    kPlaneArg = 48,   // plane arena pointer, as the recipe's debug splice reads it
    kDebugBase = D4R_RRSWIN_DEBUG_BASE,  // bytes into the arena the kernel never writes
    kDebugSlotFloor = 8192,  // the per-block debug slot floor: 32 lanes x 64 registers x 4 bytes
};


#define DEVF __attribute__((device)) static inline
typedef uint32_t lds_u32 __attribute__((address_space(3)));
typedef uint8_t lds_u8 __attribute__((address_space(3)));

// ---------------------------------------------------------------------------
// The weight image (arg040, 431124 bytes) of phases 1-10 lives in four %tid.z
// slots of 27648 bytes each; every later phase reads the image from its base.
// Within a slot the block's input GEMM is two chains of five k-steps:
//
//   chain A (columns  0..31)  weight base 320  + 1024*k
//   chain B (columns 32..63)  weight base 5440 + 1024*k
//
// Each k-step is one m16n8k32 mma per (m-tile, n-tile), so a step reads two
// 512-byte tiles: 320 and 832 for chain A's k=0, 1344 and 1856 for its k=1, and
// so on. Chain A's k=0 is phase 1, its k=1 is phase 3 (which accumulates phase
// 1's result as its C operand), and chain B's k=0 is phase 2, its k=1 phase 4.

#define WIN_BASE_A 320
#define WIN_BASE_B 5440
#define WIN_KSTEP 1024   // bytes between consecutive k-slices of one chain
#define WIN_SLOTS  27648  // bytes of weight image per %tid.z in phases 1-10

// phase 11's C operand, the 64x96 relative-position bias, at the slot base
#define WIN_BIAS_SCORE 10560
// one k-step's 32 x 32 e4m3 V block, the shared window phase 12 needs per %tid.z
#define WIN_V_BYTES 1024

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
// (VERIFIED on enc0 against the debug oracle for all 1024 (n, k) pairs of a
// tile; the 64*(n&7) term is what keeps columns 16..31 from silently reading as
// zero, and the whole 32x32 tile is a bijection onto base + [0,1024).)
DEVF uint32_t waddr(int base, int k, int n)
{
    return (uint32_t)base + 512u * (uint32_t)(n >> 4) + 64u * (uint32_t)(n & 7)
         + 16u * (uint32_t)((k >> 1) & 3) + 8u * (uint32_t)((n >> 3) & 1)
         + 4u * (uint32_t)((k >> 3) & 1) + 2u * (uint32_t)((k >> 4) & 1)
         + (uint32_t)(k & 1);
}

// warp butterfly over the four lanes that hold a row's 32 k values (laneid ^ 1
// and laneid ^ 2): the two shfl.sync.bfly steps the PTX's operand staging does.
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
// The block's input GEMM: 96 rows (six m-tiles of 16) x 32 columns, K = 160 as
// five k-steps of 32 that each round to f16 before the next one accumulates --
// which is exactly what the mma chain does, since its C operand is the previous
// step's D pair.
//
// A lane's A registers hold, for each m-tile and each of its two rows, the eight
// e4m3 values of k = 2t + (j&1) + 16*((j>>1)&1) (byte j = 0..3) and of k + 8; the
// four lanes that differ in t hold the four disjoint eighths of the 32 k's, so two
// butterfly steps broadcast the other three eighths in. The B operand is read from
// the weight image (waddr) for the lane's own two columns of each n-tile. The dot
// product runs in f32 and is rounded to f16 once, which is what the translated mma
// does: e4m3 x e4m3 products are exact and 32 of them sum exactly in f32.
//
// The thirty A fragments are the ones NVIDIA's prologue staged in shared memory
// and the cut reads back into registers immediately before the first mma (PTX
// lines 11835-11864: `ld.shared.v4.u32 {…}, [%r11624 + 512*i]`, i = 0..29, with
// `%r11624 = smem + laneid*16`). Fragment 5*mi + k is m-tile mi's k-slice, so
// `as` is the lane's slot as u32 and fragment i starts 128 words in.
//
// d is [m-tile][row half][n-tile], each entry the packed f16 pair for columns 2t, 2t+1.
DEVF void win_zero(uint32_t d[6][2][4])
{
#pragma unroll
    for (int mi = 0; mi < 6; ++mi) {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                d[mi][row][nt] = 0;
            }
        }
    }
}

// one k-step: d <- round_f16(A[ks] * W[base + 1024*ks] + d)
DEVF void win_step(const uint8_t* w, int base, int ks, const lds_u32* as, int t,
                   uint32_t d[6][2][4], uint32_t* dbg)
{
    const int wb = base + WIN_KSTEP * ks;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        float bv[2][32];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int n = 8 * nt + 2 * t + c;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                bv[c][k] = e4m3_to_f32(w[waddr(wb, k, n)]);
            }
        }
#pragma unroll
        for (int mi = 0; mi < 6; ++mi) {
            const uint32_t* q = as + 128u * (uint32_t)(5 * mi + ks);
#pragma unroll
            for (int reg = 0; reg < 2; ++reg) {
#pragma unroll
                for (int row = 0; row < 2; ++row) {
                    // the register's value on this lane and on lanes t^1, t^2, t^3
                    uint32_t p[4];
                    p[0] = q[2 * reg + row];
                    p[1] = bfly1_bits(p[0]);
                    p[2] = bfly2_bits(p[0]);
                    p[3] = bfly1_bits(p[2]);
                    // f64 accumulators: the mma's dot product is the correctly
                    // rounded sum of the 32 exact e4m3 products, and an f32
                    // accumulator would round the partial sums in whatever
                    // order the compiler picks (measured: ~8% of the outputs
                    // one ulp off the reference).
                    double a0 = 0.0, a1 = 0.0;
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const int kk = 2 * (t ^ m);
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int k = 8 * reg + kk + (j & 1) + 16 * ((j >> 1) & 1);
                            const double av = e4m3_to_f32((p[m] >> (8 * j)) & 0xffu);
                            a0 += av * (double)bv[0][k];
                            a1 += av * (double)bv[1][k];
                        }
                    }
#if D4R_RRSWIN_DEBUG_STAGE == 16
                    // this body's own arithmetic, for the oracle comparison
                    if (dbg != nullptr && ks == 0 && mi == 0) {
                        const int s16 = (row ? 4 : 0) + nt;
                        dbg[s16] = __builtin_bit_cast(uint32_t, (float)a0);
                        dbg[8 + s16] = (uint32_t)__builtin_bit_cast(uint16_t, (half_t)(float)a0);
                    }
#endif
                    // the mma's f32 accumulator plus the C operand, rounded once
                    // to f32 and then to f16
                    const uint32_t prev = d[mi][row][nt];
                    const float f0 = (float)(a0 + (double)h2f(prev));
                    const float f1 = (float)(a1 + (double)h2f(prev >> 16));
                    d[mi][row][nt] = pair_of((half_t)f0, (half_t)f1);
                }
            }
        }
    }
}

// the whole chain: five k-steps into a zeroed accumulator
DEVF void win_chain(const uint8_t* w, int base, const lds_u32* as, int t,
                    uint32_t d[6][2][4])
{
    win_zero(d);
#pragma unroll
    for (int ks = 0; ks < 5; ++ks) {
        win_step(w, base, ks, as, t, d, nullptr);
    }
}

// ---------------------------------------------------------------------------
// The attention.  Phase 11 is the 64x96 score Q*K^T: its A operand is the query
// rows 0..63 of D9 and its B operand the key rows 0..95 of the *same* D9, so
// every lane owns both rows of each dot and no lane exchange is needed -- only
// the butterfly that brings the other three lanes' eighths of the 32 k's in.
//
// E10 packs both operands identically (PTX s7908-s8027): A quad mi = registers
// written {D9[m0,nt0,g], D9[m0,nt0,g+8]} {…nt2…} {…nt1…} {…nt3…} and B pair
// nt = the same four words of D9 rows 16*(nt>>1)+g and +8.  Read back in the
// mma's own fragment order that is
//   A reg o, byte j -> row 16*mi + g + 8*(o&1), k = 2t + (j&1) + 16*((j>>1)&1) + 8*(o>>1)
//   B pair nt, reg b, byte j -> key row 16*(nt>>1) + g + 8*(nt&1), same k
// and the mma pairs A's register o with B's register o>>1, so the four n-tiles
// of an 8-column group share the two k-steps o>>1 = 0 and 1.  `row_words` below
// builds one row's two words in exactly that order (word 0 = k-slice 0, word 1
// = k-slice 1), so a row is four `cvt.rn.satfinite.e4m3x2.f16x2` and two packs.

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

// the f16 at (row, col) of a 96x32 block of packed pairs, col = 8*n-tile + 2t + c
DEVF half_t dcell(const uint32_t d[6][2][4], int row, int col)
{
    const uint32_t p = d[row >> 4][(row >> 3) & 1][col >> 3];
    return (col & 1) ? half_hi(p) : half_lo(p);
}

// the lane's two e4m3-packed words for row r: word 0 is k-slice 0 (columns
// 2t, 2t+1 and 16+2t, 17+2t), word 1 is k-slice 1 (columns 8+2t … and 24+2t …)
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

// E11(a), one element: the scale, the clamp, the two clamped-cubic fma pairs and the
// trick that moves the mantissa into the exponent field ((u << 5) & 0x7FE07FE0 over
// the pair).  The constants are the same five the PTX materialises at s8077-s8097.
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

// E11(a)-(e), one row: the twelve exponent pairs, the in-lane sum tree
// ((e0+e1)+(e2+e3))+(e4+e5) over the even pairs and the same over the odd ones,
// the two butterfly steps that finish the row's 96 columns, the two halves'
// add, the reciprocal and the twelve products.  d11 is [m-tile][row half][n-tile].
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
// The eight 16x160x32 projections, phases 13-20.
//
// E12 (PTX s11968-s12241) requantises D12 to e4m3, packs each of its four
// 16x32 m-tiles into an A fragment and stores them at
//   smem + %tid.z*2048 + 512*m-tile + laneid*16
// so every %tid.z group keeps its own attention output.  E13-E19 then read
// back, for phase 13+j, the fragment at `smem + laneid*16 + 2048*j +
// 512*%tid.z`: m-tile %tid.z of group j's D12 -- eight groups of four tiles,
// and only the %tid.z <= 3 groups carry on (the rest branch to the epilogue's
// exit at s12027, after the stores).  The fragment is in the mma's own A
// layout: word o, byte j -> row g + 8*(o&1), k = 2t + (j&1) + 16*((j>>1)&1)
// + 8*(o>>1), the same one the prologue staged.

// the lane's four A words of one staged fragment
DEVF const lds_u32* tile_frag(uint32_t scratch, int group, int tile)
{
    return (const lds_u32*)(uintptr_t)scratch
         + 4u * lane_id() + 128u * (uint32_t)(4 * group + tile);
}

// D12's m-tile mi as its A fragment's four words.  Word o = 2*(o>>1) + (o&1)
// is row (o&1)'s k-slice (o>>1), so word 0 is (row g, k = 2t, 2t+1, 16+2t,
// 17+2t) out of n-tiles 0 and 2 and word 2 the same out of n-tiles 1 and 3.
DEVF void d12_frag(const uint32_t d12[4][2][4], int mi, uint32_t af[4])
{
#pragma unroll
    for (int o = 0; o < 4; ++o) {
        const uint32_t lo = e4m3x2(d12[mi][o & 1][2 * (o >> 1)]);
        const uint32_t hi = e4m3x2(d12[mi][o & 1][2 * (o >> 1) + 1]);
        af[o] = lo | (hi << 16);
    }
}

// one 16x160x32 mma phase: A is the staged fragment, B the image's ten
// 512-byte tiles (two n-tiles each) and C the per-column table plus the block's
// input tile x.  The dot product runs in f32 and rounds to f16 once, exactly
// as the translated mma does.
DEVF void proj160(const uint8_t* w, int base, const uint32_t af[4],
                  const uint32_t cg[20], const uint32_t ch[20], int t,
                  uint32_t d[2][20])
{
#pragma unroll
    for (int nt = 0; nt < 20; ++nt) {
        const int wb = base + 512 * (nt >> 1);
        const int n0 = 8 * (nt & 1) + 2 * t;
        float bv[2][32];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                bv[c][k] = e4m3_to_f32(w[waddr(wb, k, n0 + c)]);
            }
        }
        double a0 = 0.0, a1 = 0.0;
#pragma unroll
        for (int o = 0; o < 4; ++o) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = 2 * t + (j & 1) + 16 * ((j >> 1) & 1) + 8 * (o >> 1);
                const double av = e4m3_to_f32((af[o] >> (8 * j)) & 0xffu);
                a0 += av * (double)bv[0][k];
                a1 += av * (double)bv[1][k];
            }
        }
        d[0][nt] = pair_of((half_t)(float)(a0 + (double)h2f(cg[nt])),
                           (half_t)(float)(a1 + (double)h2f(cg[nt] >> 16)));
        d[1][nt] = pair_of((half_t)(float)(a0 + (double)h2f(ch[nt])),
                           (half_t)(float)(a1 + (double)h2f(ch[nt] >> 16)));
    }
}

// one byte of the prologue's staged A fragments: m-tile `mi`, column `col`
// (0..159, so k-slice col>>5), row half `rh`.  x is a decode of e4m3 codes, so
// the staged byte is the very value the C-seed adds use.
DEVF uint16_t xcell(const lds_u32* as, int mi, int col, int rh)
{
    const int k = col & 31;
    const uint32_t word = as[128u * (uint32_t)(5 * mi + (col >> 5))
                             + 2u * (uint32_t)((k >> 3) & 1) + (uint32_t)rh];
    const uint32_t j = (uint32_t)(k & 1) | (((uint32_t)(k >> 4) & 1u) << 1);
    return __builtin_bit_cast(uint16_t, (half_t)e4m3_to_f32((word >> (8 * j)) & 0xffu));
}

// the C seeds' x half, before the table: forty b32 per lane, n-tile major and
// the lane's two columns of each, row g in xg and row g+8 in xh
DEVF void proj_x(const lds_u32* as, int mi, int t, uint32_t xg[20], uint32_t xh[20])
{
#pragma unroll
    for (int nt = 0; nt < 20; ++nt) {
        uint32_t g = 0, h = 0;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int col = 8 * nt + 2 * t + c;
            const uint32_t sh = 16u * (uint32_t)c;
            g |= (uint32_t)xcell(as, mi, col, 0) << sh;
            h |= (uint32_t)xcell(as, mi, col, 1) << sh;
        }
        xg[nt] = g;
        xh[nt] = h;
    }
}

// E12's per-column table: twenty ld.global.v2.u16 at W+221520+16*i, the last
// one at W+221504, so n-tile nt reads chunk (nt+19) % 20 -- the lane's two
// halves are 2t and 2t+1 of it.
DEVF uint32_t tabpair(const uint8_t* w, int nt)
{
    const int t = (int)(lane_id() & 3);
    const int chunk = (nt + 19) % 20;
    return *(const uint32_t*)(w + 221504u + 16u * (uint32_t)chunk + 4u * (uint32_t)t);
}

// the C seeds: table[column] + x[row][column], f16 add as the PTX does
DEVF void proj_c(const uint8_t* w, const uint32_t xg[20], const uint32_t xh[20],
                 uint32_t cg[20], uint32_t ch[20])
{
#pragma unroll
    for (int nt = 0; nt < 20; ++nt) {
        const uint32_t tab = tabpair(w, nt);
        cg[nt] = pair_of((half_t)(h2f(tab) + h2f(xg[nt])),
                         (half_t)(h2f(tab >> 16) + h2f(xg[nt] >> 16)));
        ch[nt] = pair_of((half_t)(h2f(tab) + h2f(xh[nt])),
                         (half_t)(h2f(tab >> 16) + h2f(xh[nt] >> 16)));
    }
}


// ---------------------------------------------------------------------------
// Ground truth for a stage comes from kernels/rr/swin_enc5_recipe.py's debug
// recipe: with D4R_RRSWIN_DEBUG=w1 (w2 … w10) the splice keeps NVIDIA's whole
// body and stores that phase's forty-eight accumulator registers into the plane
// arena at +46,000,000, which the kernel never otherwise writes, and the launch
// dump then holds the translated values for a block/lane. Built with
// -DD4R_RRSWIN_DEBUG_STAGE=<n> (kernels/build.sh rrswin, D4R_TEX_CFLAGS), this
// body stores the same registers of its own phase n in the same order, so the
// two dumps compare byte for byte.
#ifndef D4R_RRSWIN_DEBUG_STAGE
#define D4R_RRSWIN_DEBUG_STAGE 0
#endif

DEVF void debug_store(const uint8_t* prm, const uint32_t* regs, int n)
{
    // the oracle's splice is predicated on %tid.z == 0: the eight waves of a block
    // share one per-block slot, so only one group may fill it
    if (__builtin_amdgcn_workitem_id_z() != 0) {
        return;
    }
    const uint32_t block = __builtin_amdgcn_workgroup_id_x()
                         + 5u * __builtin_amdgcn_workgroup_id_y();
    if (block >= 64u) {
        return;   // the oracle's per-block slots only cover the first 64 blocks
    }
    const uint64_t arena = *(const uint64_t*)(prm + kPlaneArg);
    // the oracle's slot: 32 lanes x n x 4 bytes, at least kDebugSlotFloor, rounded
    // up to a power of two (its splice forms block*slot with a shift)
    const uint64_t want = (uint64_t)n * 128ull > (uint64_t)kDebugSlotFloor
        ? (uint64_t)n * 128ull : (uint64_t)kDebugSlotFloor;
    uint64_t slot = 1;
    while (slot < want) {
        slot <<= 1;
    }
    uint32_t* dst = (uint32_t*)(arena + (uint64_t)kDebugBase + (uint64_t)block * slot
                                 + (uint64_t)lane_id() * (uint64_t)n * 4ull);
    for (int i = 0; i < n; ++i) {
        dst[i] = regs[i];
    }
}

// the stage's registers in the oracle's order: the phase's twenty-four mma,
// m-tile major and n-tile minor, each as the (row g, row g+8) pair the mma writes
DEVF void stage_regs(const uint32_t d[6][2][4], uint32_t regs[48])
{
    int i = 0;
#pragma unroll
    for (int mi = 0; mi < 6; ++mi) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            regs[i++] = d[mi][0][nt];
            regs[i++] = d[mi][1][nt];
        }
    }
}

// ---------------------------------------------------------------------------
// Mirror cut. The callee is handed the weight image (arg040), the parameter
// block and the base of the entry's 16384-byte `smem` array; the thirty staged
// A fragments are still in that array when the cut is taken, so they are read
// from `scratch` rather than passed (kernels/rr/swin_enc5_recipe.py explains the
// six registers that come back the other way).
extern "C" __attribute__((device)) void d4r_swin_enc5(
    const uint8_t* w, const uint8_t* prm, uint32_t scratch)
{
    const int t = (int)(lane_id() & 3);   // the lane's two columns of each n-tile
    const int g = (int)(lane_id() >> 2);  // ... and of its sixteen rows
    // the lane's 16-byte slot in the staging array; fragment i is 512 bytes in
    const lds_u32* const as = (const lds_u32*)(uintptr_t)scratch + 4u * lane_id();
    // phases 1-10 read their weights from this %tid.z's 27648-byte slot
    const uint8_t* const wt = w + (uintptr_t)WIN_SLOTS * __builtin_amdgcn_workitem_id_z();

    // ---- phases 1-10: the block's 96x64 input GEMM -------------------------
    // chain A is phases 1,3,5,7,9 and ends in D9 (columns 0..31); chain B is
    // phases 2,4,6,8,10 and ends in D10 (columns 32..63).  All ten read the same
    // staged A operand; only the weights and the C chain differ, and the C chain
    // is what the five k-steps of win_step accumulate.  Phase 2*n-1 is chain A's
    // k-step n and phase 2*n is chain B's, so a stage store sits after each.
    uint32_t d9[6][2][4];
    uint32_t d10[6][2][4];
    win_zero(d9);
    win_zero(d10);
#if D4R_RRSWIN_DEBUG_STAGE == 15
    {   // the six staged A fragments of k-step 0, as the mma sees them
        uint32_t regs[24];
#pragma unroll
        for (int mi = 0; mi < 6; ++mi) {
#pragma unroll
            for (int o = 0; o < 4; ++o) {
                regs[4 * mi + o] = as[128u * (uint32_t)(5 * mi) + (uint32_t)o];
            }
        }
        debug_store(prm, regs, 24);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 16
    uint32_t dbg16[16] = {0};
#endif
#pragma unroll
    for (int ks = 0; ks < 5; ++ks) {
        win_step(wt, WIN_BASE_A, ks, as, t, d9,
#if D4R_RRSWIN_DEBUG_STAGE == 16
                 dbg16
#else
                 nullptr
#endif
        );
#if D4R_RRSWIN_DEBUG_STAGE == 16
        if (ks == 0) {
            uint32_t regs[24];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                regs[i] = dbg16[i];
            }
            int q = 16;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                regs[q++] = d9[0][0][nt];
                regs[q++] = d9[0][1][nt];
            }
            debug_store(prm, regs, 24);
        }
#endif
        win_step(wt, WIN_BASE_B, ks, as, t, d10, nullptr);
#if D4R_RRSWIN_DEBUG_STAGE == 1 || D4R_RRSWIN_DEBUG_STAGE == 3 \
    || D4R_RRSWIN_DEBUG_STAGE == 5 || D4R_RRSWIN_DEBUG_STAGE == 7 \
    || D4R_RRSWIN_DEBUG_STAGE == 9
        if (ks == (D4R_RRSWIN_DEBUG_STAGE - 1) / 2) {
            uint32_t regs[48];
            stage_regs(d9, regs);
            debug_store(prm, regs, 48);
        }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 2 || D4R_RRSWIN_DEBUG_STAGE == 4 \
    || D4R_RRSWIN_DEBUG_STAGE == 6 || D4R_RRSWIN_DEBUG_STAGE == 8 \
    || D4R_RRSWIN_DEBUG_STAGE == 10
        if (ks == (D4R_RRSWIN_DEBUG_STAGE - 2) / 2) {
            uint32_t regs[48];
            stage_regs(d10, regs);
            debug_store(prm, regs, 48);
        }
#endif
    }

    // ---- phases 11 and 12: the block's self-attention -----------------------
    // Phase 11: D11[mi][rh][nt] = row 16*mi + g + 8*rh of D9 dotted with row
    // 16*(nt>>1) + g + 8*(nt&1) of D9 over the sixteen k's the two registers
    // o = 2*(nt&1) and o+1 select, plus the 64x96 relative-position bias.  The
    // bias is twenty-four 512-byte tiles at slot + WIN_BIAS_SCORE, tile
    // 4*(nt>>1) + mi, sixteen bytes per lane holding the four f16 of the lane's
    // (nt&1) pair (PTX co for mma j = 12*mi + 2*(nt>>1) + (nt&1)).
    uint32_t d11[4][2][12];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            uint32_t aw[2], a[2][4];
            row_words(d9, 16 * mi + g + 8 * rh, aw);
            bfly2x4(aw, a);
#pragma unroll
            for (int nt = 0; nt < 12; ++nt) {
                uint32_t kw[2], kb[2][4];
                row_words(d9, 16 * (nt >> 1) + 8 * (nt & 1) + g, kw);
                bfly2x4(kw, kb);
                // the mma pairs A's register o with B's register o>>1, so the
                // n-tile's parity picks the k-step both sides use
                const int b = nt & 1;
                // the same exact accumulation as win_step: the correctly rounded
                // sum of the sixteen exact products, rounded once more into the bias
                double s = 0.0;
#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        s += (double)e4m3_to_f32((a[b][m] >> (8 * j)) & 0xffu)
                           * (double)e4m3_to_f32((kb[b][m] >> (8 * j)) & 0xffu);
                    }
                }
                const uint32_t bias = *(const uint32_t*)(wt + WIN_BIAS_SCORE
                                          + 512u * (uint32_t)(4 * (nt >> 1) + mi)
                                          + 16u * lane_id() + 8u * (uint32_t)(nt & 1));
                d11[mi][rh][nt] = pair_of((half_t)(float)(s + (double)h2f(bias)),
                                          (half_t)(float)(s + (double)h2f(bias >> 16)));
            }
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 11
    {
        uint32_t regs[96];
        int i = 0;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int nt = 0; nt < 12; ++nt) {
                regs[i++] = d11[mi][0][nt];
                regs[i++] = d11[mi][1][nt];
            }
        }
        debug_store(prm, regs, 96);
    }
#endif

    // Phase 12: P*V, three k-steps of 32 over the 96 key columns, each rounded
    // to f16 before the next adds to it (the chain's C operand).  The V rows a
    // lane needs live on the other 31 lanes, so each k-step's 32 x 32 e4m3 block
    // goes through shared memory -- one kilobyte per %tid.z, because the eight
    // waves of the block share the 16384-byte array and each computes its own
    // attention from its own weight slot.
    uint32_t d12[4][2][4];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                d12[mi][rh][nt] = 0;
            }
        }
    }
#pragma unroll
    for (int ks = 0; ks < 3; ++ks) {
        lds_u8* const vs = (lds_u8*)(uintptr_t)scratch
                         + (uintptr_t)WIN_V_BYTES * __builtin_amdgcn_workitem_id_z();
        // only the k-step's own 32 key rows (m-tiles 2*ks and 2*ks+1) go into the
        // kilobyte, indexed by the row's offset inside the k-step
#pragma unroll
        for (int m4 = 2 * ks; m4 < 2 * ks + 2; ++m4) {
#pragma unroll
            for (int rh4 = 0; rh4 < 2; ++rh4) {
                const int row = 16 * m4 + 8 * rh4 + g;
                const int r = row - 32 * ks;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    for (int c = 0; c < 2; ++c) {
                        const half_t v = dcell(d10, row, 8 * nt + 2 * t + c);
                        vs[r * 32 + 8 * nt + 2 * t + c] = (uint8_t)e4m3x2(pair_of(v, v));
                    }
                }
            }
        }
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
                uint32_t prob[12];
                row_probs(d11, mi, rh, prob);
                // the k-step's A fragment: word 0 packs prob[4ks] and prob[4ks+2],
                // word 1 prob[4ks+1] and prob[4ks+3]
                uint32_t aw[2];
                aw[0] = e4m3x2(prob[4 * ks]) | (e4m3x2(prob[4 * ks + 2]) << 16);
                aw[1] = e4m3x2(prob[4 * ks + 1]) | (e4m3x2(prob[4 * ks + 3]) << 16);
                uint32_t af[2][4];
                bfly2x4(aw, af);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    double acc0 = 0.0, acc1 = 0.0;
#pragma unroll
                    for (int kk = 0; kk < 32; ++kk) {
                        // the P byte for key column kk is the one the lane with
                        // t = (kk&7)>>1 packs into word (kk>>3)&1, byte
                        // 2*(kk>>4) + (kk&1); the two butterfly steps bring the
                        // other three lanes' eighths in
                        const uint32_t pa =
                            (af[(kk >> 3) & 1][(kk >> 1) & 3]
                             >> (8 * (2 * (kk >> 4) + (kk & 1)))) & 0xffu;
                        const int r = kk * 32 + 8 * nt + 2 * t;
                        acc0 += (double)e4m3_to_f32(pa) * (double)e4m3_to_f32(vs[r]);
                        acc1 += (double)e4m3_to_f32(pa) * (double)e4m3_to_f32(vs[r + 1]);
                    }
                    // the mma rounds to f16 after every k-step
                    const uint32_t prev = d12[mi][rh][nt];
                    d12[mi][rh][nt] = pair_of((half_t)(float)(acc0 + (double)h2f(prev)),
                                              (half_t)(float)(acc1 + (double)h2f(prev >> 16)));
                }
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
#if D4R_RRSWIN_DEBUG_STAGE == 12
    {
        uint32_t regs[32];
        int i = 0;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                regs[i++] = d12[mi][0][nt];
                regs[i++] = d12[mi][1][nt];
            }
        }
        debug_store(prm, regs, 32);
    }
#endif

    // ---- E12: the C seeds' x, then D12 staged as the projections' A ---------
    // x is read first: the staging below overwrites the fragments it lives in.
    const int z = (int)__builtin_amdgcn_workitem_id_z();
    uint32_t xg[20], xh[20];
    proj_x(as, z, t, xg, xh);
    {
        lds_u32* const dst = (lds_u32*)(uintptr_t)scratch + 4u * lane_id()
                           + 128u * (uint32_t)(4 * z);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            uint32_t af[4];
            d12_frag(d12, mi, af);
#pragma unroll
            for (int o = 0; o < 4; ++o) {
                dst[128 * mi + o] = af[o];
            }
        }
    }
    __builtin_amdgcn_s_barrier();   // E12's bar.sync 0 at s12026
    if (z >= 4) {
        return;   // s12027: the upper %tid.z groups leave before phase 13
    }

    // ---- phases 13-20: eight 16x160x32 projections -------------------------
    // Every phase reads the same A-fragment slot out of every group's staging
    // (phase 13+j is group j's m-tile z) and the same C seeds; only the weights
    // differ, at the tail of the eight 27648-byte weight slots.
    uint32_t cg[20], ch[20];
    proj_c(w, xg, xh, cg, ch);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t* const af = tile_frag(scratch, j, z);
        uint32_t d[2][20];
        proj160(w, 22848 + 27648 * j, af, cg, ch, t, d);
#if D4R_RRSWIN_DEBUG_STAGE == 13
        if (j == 0) {
            uint32_t regs[40];
#pragma unroll
            for (int nt = 0; nt < 20; ++nt) {
                regs[2 * nt] = cg[nt];
                regs[2 * nt + 1] = ch[nt];
            }
            debug_store(prm, regs, 40);
        }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 14
        if (j == 0) {
            uint32_t regs[40];
#pragma unroll
            for (int nt = 0; nt < 20; ++nt) {
                regs[2 * nt] = d[0][nt];
                regs[2 * nt + 1] = d[1][nt];
            }
            debug_store(prm, regs, 40);
        }
#endif
        (void)d;
    }

    // ---- handing the accumulators back -------------------------------------
    // Phase 61's four accumulators and the two values E61 reads that are computed
    // inside the region (%r29781 = %tid.z, %r29782 = the shared base) cross the cut
    // through the last 768 bytes of the shared array, 24 bytes apart per lane.
    __attribute__((address_space(3))) uint32_t* const out =
        (__attribute__((address_space(3))) uint32_t*)(uintptr_t)scratch
        + (16384u - 24u * 32u) / 4u + 6u * lane_id();
    (void)d12;

    // Phases 13-61 are not implemented yet; until they are, the four accumulators
    // are filled with lane-distinct values so the read-back path is observable
    // without producing wild magnitudes.
    for (int i = 0; i < 4; ++i) {
        out[i] = (uint32_t)lane_id() + 32u * (uint32_t)i;
    }
    out[4] = __builtin_amdgcn_workitem_id_z();
    out[5] = scratch;
}
