// The Swin-layer engine: descriptor-driven e4m3 GEMM phases for the eleven
// cuda_dldn_engine_swin_* denoiser layers.
//
// Why: the translated kernels are ~83,373 instructions for 536 FP8 WMMAs, dominated by
// operand reassembly (9,124 selects, 7,976 LDS permutes, ~23,000 shifts/ands/perms).  An
// engine that keeps its activations as an e4m3 image and reads operands directly from it
// pays none of that.
//
// STATUS: every function here is validated by kernels/tools/rrswin_engine_k16_test.cpp, which
// runs on the GPU with checked HIP calls and compares against host models written independently:
// step 1 one k32 step, step 2 a 64-wide k chain as two k32 steps with f16 rounding between them,
// step 3 staging an accumulator as the next phase's operands (staged image dumped and compared
// byte for byte), step 4a/4b the weight preparation (slots checked directly, then through a GEMM
// against the documented image layout), step 4c a 32-row phase with two m-tiles, which is the
// case swin_gemm_tile's slot_step exists for, step 5/5a the phase loop itself, two phases chained
// through a staged image (the staged image is checked against the model as well), and step 6
// swin_act_pair, the clamped cubic activation, against a scalar model of the specification's
// formula at a one-ulp bar (the device uses packed f16x2 ops, as the PTX does; the bit-exact
// check for it is the oracle comparison).  The layout was measured, not assumed:
// kernels/tools/rrswin_operand_fit.cpp reports D's row index as VGPR i + 8*(lane>>4), its
// column index as lane & 15, and both operands' byte order as the identity.
// Step 3 of that test stages an accumulator and consumes it as the next phase's operands, and
// also dumps the staged image: every staged byte equalled the model's own e4m3 requantisation
// of D[k][col], so the accumulator-to-operand mapping ("D VGPR i is row i + 8h" is the next
// GEMM's k) is measured, not assumed.  Its earlier mismatches were all bugs in the test: a
// k32 step mislabelled k16, a weight array too small for a second GEMM's slots, and a second
// k16 chunk left holding the first GEMM's operands.
//
// Layout facts this file uses, and where each comes from:
//  1. A k16 WMMA operand is 8 e4m3 bytes = this lane's half of a 16-byte row.  "half h the K
//     values 8h .. 8h+7 in order"        -- kernels/common/wmma_layout.h
//  2. The lane/half selection is `8 * hf` with hf = lane >> 4, and the caller reads the row's
//     16 bytes from shared memory itself
//                                        -- kernels/tex/enc0_tail.hip (working native)
//  3. A weight slot is the same 16 bytes per lane, `kslot_from16` is the identity under
//     D4R_TEX_FP8 on gfx12, and `kop_image(img, idx)` picks this half
//                                        -- kernels/tex/tex_common.h
//  4. `k32_e4m3(c, a0, b0, a1, b1)` is one m16n8k32 e4m3 step as two native gfx12 FP8
//     WMMAs and is bit-identical to ZLUDA's gfx12 lowering of NVIDIA's mma
//                                        -- kernels/tex/tex_common.h (with tex_common's own
//                                           comment, and enc0_tail/dec0_head using it in
//                                           shipped, verified kernels)
#pragma once
#include <stddef.h>
#include "../tex/tex_common.h"

namespace rrswin
{

// This lane's operand for k16 chunk `chunk` of image row `row_base + (lane & 15)`; rows are
// `stride` bytes apart.  The row comes from the lane, as in enc0_tail (`m = l & 15`), so a
// caller cannot pass a row that does not match its lane.  A row holds consecutive k, 16
// bytes per k16 chunk, so a k32 image needs stride >= 32; a smaller stride silently reads
// the next row.
__attribute__((device)) static inline kop swin_operand(const uint8_t* img, int stride, int row_base, int chunk)
{
    const uint32_t l = lane_id();
    const uint32_t half_bytes = (l >= 16) ? 8u : 0u;
    const uint32_t row = (uint32_t)row_base + (l & 15u);
    const uint2_t* p = (const uint2_t*)(img + (size_t)row * (size_t)stride + 16u * (uint32_t)chunk + half_bytes);
    return kop_from8(*p);
}

// One m16n16 output tile of one k32 step.  The operand roles follow the working kernels
// (enc0_tail): the prepared weight slot is the A operand and the staged activation is the B
// operand, so the WMMA computes D^T relative to the token-major view.  Both operands are
// lane-indexed: A's row and B's column are both `lane & 15` (wmma_layout.h: "A operand lane
// l = row r of A, B operand lane l = column r of B", r = l & 15), which is why the weight
// slot is `slot_base + (lane & 15)` and the two k16 halves of that slot are 16 slots apart
// -- exactly enc0_tail's `kop_image(w, slot)` / `kop_image(w, slot + 16)` with
// `slot = base + m`.
// `slot_step` is the distance between a row's two k16 slots.  It is a parameter, not a
// constant, because a phase with more than one m-tile puts the next tile's rows in between:
// with slot_step fixed at 16 a 32-row phase would read tile 1's rows as tile 0's second k16.
// `k32_index` selects which 32-wide slice of the activations this step reads: a phase with more
// than one k32 step must advance the image's chunks too, or every step re-reads k 0..31.
__attribute__((device)) static inline f8v swin_gemm_tile(f8v c, const kslot* w, int slot_base,
                                                         int slot_step, const uint8_t* img, int stride,
                                                         int row_base, int k32_index = 0)
{
    const int slot = slot_base + (int)(lane_id() & 15u);
    const kop a0 = kop_image(w, slot);
    const kop b0 = swin_operand(img, stride, row_base, 2 * k32_index);
    const kop a1 = kop_image(w, slot + slot_step);
    const kop b1 = swin_operand(img, stride, row_base, 2 * k32_index + 1);
    return k32_e4m3(c, a0, b0, a1, b1);
}

// Stage an accumulator as the operand image of the next phase.  On gfx12 the accumulator's
// layout *is* the operand layout (wmma_layout.h: "D VGPR i is row i + 8h, which is the gfx12
// B-operand layout of the next GEMM (no exchange)"): this lane's VGPR i holds the value that
// belongs at k = i + 8*(lane >> 4) of column (lane & 15), so each lane requantises its own
// eight values to e4m3 and writes them at its own row's k-half -- no cross-lane movement.
// `row_base` is the first token column this tile covers and `chunk` the k16 chunk the tile's
// rows land in (a 16-row m-tile is one k16 chunk), so a phase with several tiles stages each
// into its own place.
__attribute__((device)) static inline void swin_stage_operand(uint8_t* img, int stride, int row_base,
                                                              int chunk, const f8v acc)
{
    const uint32_t l = lane_id();
    const uint32_t h = l >> 4, col = (uint32_t)row_base + (l & 15u);
    uint8_t* p = img + (size_t)col * (size_t)stride + 16u * (uint32_t)chunk + 8u * h;
#pragma unroll
    for (int i = 0; i < 8; i += 2)
    {
        const hv2 pair = {(half_t)acc[i], (half_t)acc[i + 1]};
        const uint32_t code = e4m3x2(__builtin_bit_cast(uint32_t, pair));
        p[i] = (uint8_t)(code & 0xffu);
        p[i + 1] = (uint8_t)(code >> 8);
    }
}

// ---------------------------------------------------------------- weight preparation
// One phase's weights as WMMA slots, so the phase loop can read them with kop_image and never
// touch the image layout.  Slot `base + 2*n_rows*s + n_rows*h + n` holds the k32 step s, k16
// half h and output row n, i.e. 16 consecutive e4m3 bytes of k 32s + 16h .. +15; the caller
// passes that same `n_rows` as swin_gemm_tile's slot_step.  The 2*n_rows per step is what keeps
// a step's two k16 halves adjacent for any row count.
//
// `waddr(k, n)` returns the byte offset of the weight image's element (k, n).  Layer bodies
// document their own; enc0's is
//   base + 512*(n>>4) + 64*(n&7) + 16*((k>>1)&3) + 8*((n>>3)&1) + 4*((k>>3)&1) + 2*((k>>4)&1) + (k&1)
// and a layer whose image differs passes its own lookup.  Run it once per launch (a `_prep`
// kernel, as kernels/tex/enc0_tail.hip does for its weights).
template <typename Lookup>
__attribute__((device)) static inline void swin_prep_phase(uint8_t* slots, int base, int n_rows,
                                                           int k_steps, const uint8_t* image,
                                                           Lookup waddr)
{
    for (int n = 0; n < n_rows; ++n)
        for (int s = 0; s < k_steps; ++s)
            for (int h = 0; h < 2; ++h)
                for (int j = 0; j < 16; ++j)
                    slots[(size_t)(16 * (base + 2 * n_rows * s + n_rows * h + n) + j)] =
                        image[waddr(32 * s + 16 * h + j, n)];
}

// ---------------------------------------------------------------- f16 pair helpers
// The packed-f16 plumbing kernels/rr/rrswin_enc0.hip defines for its epilogues.
__attribute__((device)) static inline half_t hf16(uint16_t bits) { return __builtin_bit_cast(half_t, bits); }
__attribute__((device)) static inline float h2f(uint32_t bits)
{
    return (float)__builtin_bit_cast(half_t, (uint16_t)bits);
}
__attribute__((device)) static inline hv2 hv2_of(uint32_t p) { return __builtin_bit_cast(hv2, p); }
__attribute__((device)) static inline uint32_t u32_of(hv2 v) { return __builtin_bit_cast(uint32_t, v); }
__attribute__((device)) static inline uint32_t pair_of(half_t a, half_t b)
{
    return (uint32_t)__builtin_bit_cast(uint16_t, a) | ((uint32_t)__builtin_bit_cast(uint16_t, b) << 16);
}

// ---------------------------------------------------------------- epilogues
// The clamped cubic activation every layer's MLP applies before requantising:
//   y = clamp(x, -2, +2);  out = x * (0.5 + y * (0.412109375 - 0.0810546875 * |y|))
// in f16 throughout, on the two halves of a packed pair.  (Not SiLU/GELU: g(+2) = 1 and
// g(-2) = 0.)  The constants are the f16 values of the PTX's 0f3ED306EB/0f3DA60DD6/0f3F000000/
// 0f40000000 as kernels/rr/rrswin_enc0.hip documents them.
__attribute__((device)) static inline uint32_t swin_act_pair(uint32_t x)
{
    const hv2 two = hv2_of(pair_of(hf16(0x4000), hf16(0x4000)));
    const hv2 slope = hv2_of(pair_of(hf16(0x2d30), hf16(0x2d30)));
    const hv2 mid = hv2_of(pair_of(hf16(0x3698), hf16(0x3698)));
    const hv2 half0 = hv2_of(pair_of(hf16(0x3800), hf16(0x3800)));
    hv2 y = __builtin_elementwise_max(hv2_of(x), -two);
    y = __builtin_elementwise_min(y, two);
    y = __builtin_elementwise_max(y, -y);
    const hv2 g = half0 + y * (mid - slope * y);
    return u32_of(hv2_of(x) * g);
}

// Stage NVIDIA's m16n8k32 A fragments as an image.  The fragment layout is the one
// kernels/rr/rrswin_enc0.hip documents and verified against the oracle:
//   reg 0 byte j -> row g,     k = 2t + (j&1) + 16*((j>>1)&1)
//   reg 1 byte j -> row g+8,   same k
//   reg 2 byte j -> row g,     k = 8 + 2t + (j&1) + 16*((j>>1)&1)
//   reg 3 byte j -> row g+8,   same k
// with g = lane>>2 and t = lane&3.  One m-tile's fragment is four registers per lane, so a
// phase's 4*m_tiles registers cover 16*m_tiles rows.  Each lane scatters its own 16 bytes, which
// is a shared-memory store per byte -- the whole point of the engine is that this happens once
// per phase instead of once per mma.
__attribute__((device)) static inline void swin_stage_fragment(uint8_t* img, int stride,
                                                              const uint32_t* frag, int m_tiles)
{
    const uint32_t l = lane_id();
    const uint32_t g = l >> 2, t = l & 3u;
    for (int mi = 0; mi < m_tiles; ++mi)
    {
        const uint32_t* f = frag + 4 * mi;
        uint8_t* r0 = img + (size_t)(16 * mi + g) * (size_t)stride;
        uint8_t* r1 = img + (size_t)(16 * mi + g + 8) * (size_t)stride;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const uint32_t k0 = 2 * t + (j & 1) + 16u * ((j >> 1) & 1);
            const uint32_t k2 = 8u + k0;
            r0[k0] = (uint8_t)(f[0] >> (8 * j));
            r1[k0] = (uint8_t)(f[1] >> (8 * j));
            r0[k2] = (uint8_t)(f[2] >> (8 * j));
            r1[k2] = (uint8_t)(f[3] >> (8 * j));
        }
    }
}

// ---------------------------------------------------------------- the phase loop
// One phase: m_tiles x n_tiles tiles of a k32-steps GEMM, each tile a 16x16 m16n16 WMMA block,
// with the epilogue staging each tile's accumulator as the next phase's activations.
struct Phase
{
    int m_tiles;      // 16-row tiles of the weight image
    int n_tiles;      // 16-token tiles
    int k32_steps;    // k = 32 * k32_steps
    int slot_base;    // first weight slot of this phase (see swin_prep_phase)
    int epilogue;     // 0 stage the accumulator, 1 keep it in acc
    int dst_image;    // image the epilogue stages into (epilogue 0)
};

// Run one phase from `src`, staging requantized accumulators into `dst` or retaining them.
// `weight_rows` is the output row count used to prepare each pair of k16 weight slot banks.
// When retaining results, `acc` must hold m_tiles * n_tiles entries in row-major tile order.
__attribute__((device)) static inline void swin_phase(const Phase& ph, const kslot* w, int weight_rows,
                                                      const uint8_t* src, int src_stride,
                                                      uint8_t* dst, int dst_stride, f8v* acc)
{
    for (int mt = 0; mt < ph.m_tiles; ++mt)
        for (int nt = 0; nt < ph.n_tiles; ++nt)
        {
            f8v c = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int step = 0; step < ph.k32_steps; ++step)
                c = swin_gemm_tile(c, w, ph.slot_base + 16 * mt + 2 * weight_rows * step, weight_rows,
                                   src, src_stride, 16 * nt, step);
            if (ph.epilogue == 0)
                swin_stage_operand(dst, dst_stride, 16 * nt, mt, c);
            else if (acc != nullptr)
                acc[mt * ph.n_tiles + nt] = c;
        }
}

// enc0's weight image layout, as its body documents it.
__attribute__((device)) static inline int swin_waddr_enc0(int k, int n)
{
    return 512 * (n >> 4) + 64 * (n & 7) + 16 * ((k >> 1) & 3) + 8 * ((n >> 3) & 1)
         + 4 * ((k >> 3) & 1) + 2 * ((k >> 4) & 1) + (k & 1);
}

} // namespace rrswin
