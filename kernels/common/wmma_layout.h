// WMMA register layouts of RDNA3 (gfx11) and RDNA4 (gfx12) for the native kernels (wave32, 16x16x16 f16 -> f32).
// Every layout fact the kernels depend on is here; the kernel sources are written against these helpers.
//
// Notation: lane l, half h = l >> 4, r = l & 15. A operand lane l = row r of A, B operand lane l = column r
// of B, D lane l = column r of D.
//   gfx11: an operand holds all 16 K values of its row/column (8 dwords, the same data in both halves);
//          D VGPR i is row 2i + h.
//   gfx12: an operand holds 8 of the 16 K values (4 dwords), half h the K values 8h .. 8h+7 in order;
//          D VGPR i is row i + 8h, which is the gfx12 B-operand layout of the next GEMM (no exchange).
// A and B only have to agree on which K each slot holds; "half h = K 8h .. 8h+7" makes a gfx12 operand the
// lower or upper 16 bytes of the gfx11 one, so row images and weight images serve both layouts.
//
// D4R_WMMA_LAYOUT selects the layout (default: the target's). D4R_WMMA_LAYOUT=12 on a gfx11 target is the
// layout shim: the gfx12 register layout with every WMMA done by the gfx11 instruction on operands and
// accumulators converted around it, in the gfx11 K order. Its results are bit-identical to the gfx11 build,
// so the gfx12 indexing can be validated on RDNA3 hardware.
//
// Only clang builtins are used (the texture tails compile with -nogpuinc -nogpulib).
#pragma once
#include <stdint.h>

#ifndef D4R_WMMA_LAYOUT
#if defined(__GFX12__)
#define D4R_WMMA_LAYOUT 12
#else
#define D4R_WMMA_LAYOUT 11
#endif
#endif
#if D4R_WMMA_LAYOUT == 12 && !defined(__GFX12__)
#define D4R_WMMA_SHIM 1
#endif
#if D4R_WMMA_LAYOUT == 11 && defined(__GFX12__)
#error "the gfx11 WMMA layout does not exist on gfx12"
#endif
// RDNA2 (gfx10.3): no WMMA instruction; the gfx11 layout with the products done in software (wm_mma below).
// The gfx11 true16 assembly (v_cvt_f16_f32 %0.l/.h) does not exist there either; the C conversions replace it.
#if defined(__GFX10__) && !defined(D4R_WMMA_SOFT)
#define D4R_WMMA_SOFT 1
#endif
#ifdef __GFX10__
#undef WM_CVT_ASM
#endif
#if defined(__GFX10__) && D4R_WMMA_LAYOUT != 11
#error "RDNA2 kernels use the gfx11 WMMA layout"
#endif

#define WM_FN __attribute__((device, always_inline)) static inline

typedef _Float16 wm_f16;
typedef _Float16 wm_hv2 __attribute__((ext_vector_type(2)));
typedef _Float16 wm_h8 __attribute__((ext_vector_type(8)));
typedef _Float16 wm_h16 __attribute__((ext_vector_type(16)));
typedef float wm_f8v __attribute__((ext_vector_type(8)));
typedef uint32_t wm_u8v __attribute__((ext_vector_type(8)));
typedef uint32_t wm_u4v __attribute__((ext_vector_type(4)));
typedef int wm_i2v __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------- lanes
WM_FN uint32_t wm_lane()
{
    return __builtin_amdgcn_mbcnt_lo(~0u, 0u);
}
WM_FN uint32_t wm_half()
{
    return wm_lane() >> 4;
}
// value of the lane with the same index in the other half of the wave
WM_FN uint32_t wm_other_half(uint32_t v)
{
#ifdef WM_HALF_SWIZZLE
    // Bit-mode AND=31, OR=0, XOR=16: the same half-wave partner, no address VGPR.
    return (uint32_t)__builtin_amdgcn_ds_swizzle((int)v, 0x401f);
#elif defined(WM_HALF_BPERMUTE)
    // through the LDS crossbar (no VALU issue; the same value)
    return (uint32_t)__builtin_amdgcn_ds_bpermute((int)((__builtin_amdgcn_mbcnt_lo(~0u, 0u) ^ 16u) << 2), (int)v);
#elif defined(WM_PERMLANE_ASM)
    // the builtin ties the destination to an "old" input (unused here: all lanes active, every select valid), which
    // costs a v_mov per exchange; a write-only asm output avoids it. Same instruction, same result.
    uint32_t r;
    __asm__("v_permlanex16_b32 %0, %1, %2, %3" : "=&v"(r) : "v"(v), "s"(0x76543210u), "s"(0xfedcba98u));
    return r;
#else
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
#endif
}
WM_FN float wm_other_half_f(float v)
{
    return __builtin_bit_cast(float, wm_other_half(__builtin_bit_cast(uint32_t, v)));
}
WM_FN uint32_t wm_pack2(wm_f16 lo, wm_f16 hi)
{
    return __builtin_bit_cast(uint32_t, (wm_hv2){lo, hi});
}
WM_FN wm_f16 wm_lo16(uint32_t v)
{
    return __builtin_bit_cast(wm_hv2, v)[0];
}
WM_FN wm_f16 wm_hi16(uint32_t v)
{
    return __builtin_bit_cast(wm_hv2, v)[1];
}

// ---------------------------------------------------------------- accumulator rows
// row (of the 16-row tile) that accumulator VGPR i holds in a lane of half h (prep code: logical lanes)
WM_FN int wm_acc_row_h(int i, int h)
{
#if D4R_WMMA_LAYOUT == 12
    return i + 8 * h;
#else
    return 2 * i + h;
#endif
}
// ... in this lane
WM_FN int wm_acc_row(int i)
{
    return wm_acc_row_h(i, (int)wm_half());
}
// rows are the K dimension of the next GEMM: VGPRs 2j, 2j + 1 are adjacent K values in gfx12 (and 16 bytes
// of this lane's 8 rows are contiguous), rows 2i + h are interleaved between the halves in gfx11
#if D4R_WMMA_LAYOUT == 12
#define WM_ACC_CONTIGUOUS 1
#else
#define WM_ACC_CONTIGUOUS 0
#endif

// ---------------------------------------------------------------- f16 operands
#if D4R_WMMA_LAYOUT == 12
typedef wm_u4v wm_op; // this half's 8 K values
#define WM_OP_DWORDS 4
#else
typedef wm_u8v wm_op; // all 16 K values
#define WM_OP_DWORDS 8
#endif

// operand from 16 contiguous halves (K 0..15) in memory (16-byte aligned); gfx12 reads only this half's 8
WM_FN wm_op wm_op_load(const wm_f16* p)
{
#if D4R_WMMA_LAYOUT == 12
    return *(const wm_u4v*)(p + 8 * wm_half());
#else
    const wm_u4v a = *(const wm_u4v*)p, b = *(const wm_u4v*)(p + 8);
    return (wm_u8v){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
#endif
}

// operand slot `idx` of an image of 32-byte slots (16 halves each, K in order): gfx12 reads its 16-byte half
WM_FN wm_op wm_op_image(const wm_u8v* __restrict__ img, int idx)
{
#if D4R_WMMA_LAYOUT == 12
    return ((const wm_u4v*)img)[2 * idx + (int)wm_half()];
#else
    return img[idx];
#endif
}

// element (K value) of an operand that holds accumulator row wm_acc_row(i) of the same lane
WM_FN wm_f16 wm_op_acc_elem(const wm_op& v, int i)
{
#if D4R_WMMA_LAYOUT == 12
    return (i & 1) ? wm_hi16(v[i >> 1]) : wm_lo16(v[i >> 1]);
#else
    // element 2 i + h is half h of dword i: one shift by 16 h (written as v[(2 i + h) >> 1] the compiler could not
    // see that the dword index is the constant i and indexed the operand dynamically, ~9 VALU per element)
    return __builtin_bit_cast(wm_f16, (uint16_t)(v[i] >> (wm_half() << 4)));
#endif
}

// the f16 accumulator values own[i] (row wm_acc_row(i) of this lane's column) as an operand of the next GEMM
// (the rows become K; gfx11 fetches the other parity from lane l ^ 16)
WM_FN wm_op wm_op_from_acc(const wm_f16 own[8])
{
#if D4R_WMMA_LAYOUT == 12
    return (wm_u4v){wm_pack2(own[0], own[1]), wm_pack2(own[2], own[3]), wm_pack2(own[4], own[5]), wm_pack2(own[6], own[7])};
#else
#ifdef WM_OP_VIA_LDS
    // Through a per-wave LDS row buffer (WM_OP_VIA_LDS = waves per block, wave index = work-item z): every lane
    // stores its 8 rows of its token's row, then loads the whole 16-value row. 8 VALU conversions + 10 LDS
    // instructions instead of ~20 VALU (pack, exchange, perm). The same values. Rows 48 bytes apart (16 mod 32).
    // A wave's LDS instructions complete in order, so the loads see all lanes' stores and the next call's stores
    // cannot overtake these loads.
    __shared__ __attribute__((aligned(16))) wm_f16 wm_xbuf[WM_OP_VIA_LDS][16][24];
    {
        const uint32_t ln = __builtin_amdgcn_mbcnt_lo(~0u, 0u);
        wm_f16* row = wm_xbuf[__builtin_amdgcn_workitem_id_z()][ln & 15];
        const uint32_t h = ln >> 4;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            row[2 * i + h] = own[i];
        const wm_u4v a = *(const wm_u4v*)row, b = *(const wm_u4v*)(row + 8);
        return (wm_u8v){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
    }
#endif
    const uint32_t hf = wm_half();
    uint32_t mine[4], theirs[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        mine[j] = wm_pack2(own[2 * j], own[2 * j + 1]);
        theirs[j] = wm_other_half(mine[j]);
    }
#ifndef PWIN_NO_PERM
    // v_perm_b32(theirs, mine): bytes 0-3 = mine, 4-7 = theirs. Even j: the low halves, odd j: the high
    // halves, in the order (mine, theirs) for hf = 0 and (theirs, mine) for hf = 1.
    const uint32_t sel_lo = hf ? 0x01000504u : 0x05040100u, sel_hi = hf ? 0x03020706u : 0x07060302u;
    wm_u8v r;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        r[j] = __builtin_amdgcn_perm(theirs[j >> 1], mine[j >> 1], (j & 1) ? sel_hi : sel_lo);
    return r;
#else
    wm_u8v r;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        const wm_f16 a = (j & 1) ? wm_hi16(mine[j >> 1]) : wm_lo16(mine[j >> 1]);
        const wm_f16 b = (j & 1) ? wm_hi16(theirs[j >> 1]) : wm_lo16(theirs[j >> 1]);
        r[j] = hf ? wm_pack2(b, a) : wm_pack2(a, b);
    }
    return r;
#endif
#endif
}

// two f32 -> f16 conversions (round to nearest even) packed into one register (low = a, high = b). WM_CVT_ASM: the
// same v_cvt_f16_f32 instructions as inline asm writing %0.l then %0.h; the compiler otherwise re-packs the halves
// with v_pack_b32_f16 copies
WM_FN uint32_t wm_cvt2(float a, float b)
{
#ifdef WM_CVT_ASM
    uint32_t r;
    asm("v_cvt_f16_f32 %0.l, %1" : "=v"(r) : "v"(a));       // r may reuse a's register (a is read first)
    asm("v_cvt_f16_f32 %0.h, %1" : "+v"(r) : "v"(b));       // keeps the low half
    return r;
#else
    return wm_pack2((wm_f16)a, (wm_f16)b);
#endif
}
#if D4R_WMMA_LAYOUT == 12
// gfx12: an operand is this half's 8 K values in 4 dwords, accumulator VGPR i = row (K) 8 h + i: the packed pairs
// ARE the operand (no exchange, no byte permutation)
WM_FN wm_op wm_op_from_mine(const uint32_t mine[4])
{
    return (wm_u4v){mine[0], mine[1], mine[2], mine[3]};
}
// the pair (element 2 j, 2 j + 1) of wm_op_acc_elem(v, .): operand dword j
WM_FN uint32_t wm_op_acc_pair(const wm_op& v, int j)
{
    return v[j];
}
#else
// operand from this lane's 8 accumulator-row halves already packed in pairs (mine[j] = own[2 j], own[2 j + 1]);
// the same exchange + byte permutation as wm_op_from_acc
WM_FN wm_op wm_op_from_mine(const uint32_t mine[4])
{
    const uint32_t hf = wm_half();
    uint32_t theirs[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        theirs[j] = wm_other_half(mine[j]);
    const uint32_t sel_lo = hf ? 0x01000504u : 0x05040100u, sel_hi = hf ? 0x03020706u : 0x07060302u;
    wm_u8v r;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        r[j] = __builtin_amdgcn_perm(theirs[j >> 1], mine[j >> 1], (j & 1) ? sel_hi : sel_lo);
    return r;
}
// the pair (element 2 j, 2 j + 1) of wm_op_acc_elem(v, .) as one packed register (one byte permutation)
WM_FN uint32_t wm_op_acc_pair(const wm_op& v, int j)
{
    return __builtin_amdgcn_perm(v[2 * j + 1], v[2 * j], wm_half() ? 0x07060302u : 0x05040100u);
}
#endif
#if defined(WM_CVT_ASM) && (D4R_WMMA_LAYOUT == 12 || (!defined(WM_OP_VIA_LDS) && !defined(PWIN_NO_PERM)))
#define WM_HAVE_FROM_MINE 1
WM_FN wm_op wm_op_from_f8(wm_f8v d)
{
    uint32_t mine[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        mine[j] = wm_cvt2(d[2 * j], d[2 * j + 1]);
    return wm_op_from_mine(mine);
}
#else
WM_FN wm_op wm_op_from_f8(wm_f8v d)
{
    wm_f16 h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        h[i] = (wm_f16)d[i];
    return wm_op_from_acc(h);
}
#endif

// Stores an operand back as a row of 16 halves (K 0..15). In gfx11 both halves hold the whole row and the
// lanes of half 0 store it; in gfx12 each half stores its 8 values. `count` (8 or 16) limits the row to its
// first `count` halves; `ok` is the lane's own condition.
WM_FN void wm_op_store(wm_f16* p, const wm_op& v, bool ok = true, int count = 16)
{
#if D4R_WMMA_LAYOUT == 12
    if (ok && (wm_half() == 0 || count > 8))
        *(wm_u4v*)(p + 8 * wm_half()) = v;
#else
    if (ok && wm_half() == 0)
    {
        *(wm_u4v*)p = (wm_u4v){v[0], v[1], v[2], v[3]};
        if (count > 8)
            *(wm_u4v*)(p + 8) = (wm_u4v){v[4], v[5], v[6], v[7]};
    }
#endif
}

// 8 halves of an f16 vector in accumulator row order: out[i] = v[wm_acc_row(i)], v at p (16 halves, 16-byte
// aligned)
WM_FN void wm_acc_vec_load(const wm_f16* p, wm_f16 out[8])
{
#if D4R_WMMA_LAYOUT == 12
    const wm_u4v a = *(const wm_u4v*)(p + 8 * wm_half());
#pragma unroll
    for (int i = 0; i < 8; ++i)
        out[i] = (i & 1) ? wm_hi16(a[i >> 1]) : wm_lo16(a[i >> 1]);
#else
    const uint32_t hf = wm_half();
    const wm_u4v a = *(const wm_u4v*)p, b = *(const wm_u4v*)(p + 8);
    const wm_u8v v = (wm_u8v){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
#pragma unroll
    for (int i = 0; i < 8; ++i)
        out[i] = __builtin_bit_cast(wm_f16, (uint16_t)(v[i] >> (hf << 4)));
#endif
}

// ---------------------------------------------------------------- the WMMA itself
#if D4R_WMMA_LAYOUT == 12 && defined(__GFX12__)
WM_FN wm_f8v wm_mma(const wm_op& a, const wm_op& b, wm_f8v c)
{
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(wm_h8, a), __builtin_bit_cast(wm_h8, b), c);
}
#else
#ifndef __GFX10__
WM_FN wm_f8v wm_mma11(const wm_u8v& a, const wm_u8v& b, wm_f8v c)
{
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(wm_h16, a), __builtin_bit_cast(wm_h16, b), c);
}
#endif
#if D4R_WMMA_LAYOUT == 12
// layout shim: gfx12 operand halves -> one gfx11 operand in K order (half 0's 8 K, then half 1's)
WM_FN wm_u8v wm_shim_operand(const wm_u4v& v)
{
    const bool hf = wm_half() != 0;
    wm_u8v r;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const uint32_t o = wm_other_half(v[j]);
        r[j] = hf ? o : v[j];
        r[4 + j] = hf ? v[j] : o;
    }
    return r;
}
// gfx12 accumulator (VGPR i = row i + 8h) <-> gfx11 accumulator (VGPR i = row 2i + h)
WM_FN wm_f8v wm_shim_acc_to11(const wm_f8v& c)
{
    const bool hf = wm_half() != 0;
    wm_f8v o, r;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        o[i] = wm_other_half_f(c[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        // row 2i + h lives in half (2i + h) >> 3, VGPR (2i + h) & 7
        const float h0 = i < 4 ? c[2 * i] : o[2 * i - 8];
        const float h1 = i < 4 ? o[2 * i + 1] : c[2 * i - 7];
        r[i] = hf ? h1 : h0;
    }
    return r;
}
WM_FN wm_f8v wm_shim_acc_to12(const wm_f8v& d)
{
    const bool hf = wm_half() != 0;
    wm_f8v o, r;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        o[i] = wm_other_half_f(d[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        // row i + 8h lives in half i & 1, VGPR (i >> 1) + 4h
        const float h0 = (i & 1) ? o[i >> 1] : d[i >> 1];
        const float h1 = (i & 1) ? d[(i >> 1) + 4] : o[(i >> 1) + 4];
        r[i] = hf ? h1 : h0;
    }
    return r;
}
WM_FN wm_f8v wm_mma(const wm_op& a, const wm_op& b, wm_f8v c)
{
    return wm_shim_acc_to12(wm_mma11(wm_shim_operand(a), wm_shim_operand(b), wm_shim_acc_to11(c)));
}
// ZLUDA's K order: its gfx11 helpers put K pair p (K 2p, 2p + 1) in operand dword p, its gfx12 helpers put
// in half h, dword j the pair (j & 1) * 4 + 2h + (j >> 1) (see zluda_ptx_impl.cpp)
WM_FN wm_u8v wm_shim_operand_zluda(const wm_u4v& v)
{
    const bool hf = wm_half() != 0;
    wm_u4v o;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        o[j] = wm_other_half(v[j]);
    wm_u8v r;
#pragma unroll
    for (int p = 0; p < 8; ++p)
    {
        const int hp = (p & 3) >> 1, jp = 2 * (p & 1) + (p >> 2);
        r[p] = (hp != 0) == hf ? v[jp] : o[jp];
    }
    return r;
}
WM_FN wm_f8v wm_mma_zluda(const wm_op& a, const wm_op& b, wm_f8v c)
{
    return wm_shim_acc_to12(wm_mma11(wm_shim_operand_zluda(a), wm_shim_operand_zluda(b), wm_shim_acc_to11(c)));
}
#elif defined(D4R_WMMA_SOFT)
// RDNA2 (gfx103x) has no WMMA: the same gfx11 register layout, multiplied with v_dot2_f32_f16. Lane (r, h) holds
// column r of B and needs rows 2i + h of A, which lanes 2i + h (any half) hold. Half 1's A is rotated by one row
// first (row_ror:15 on DPP row 1 only), so that one row_share:2i broadcast gives half 0 row 2i and half 1 row
// 2i + 1. 8 + 64 DPP moves and 64 dot2 per 16x16x16 product, f32 accumulation in K order (two products per step).
// D4R_WMMA_SOFT on gfx11 runs this path on RDNA3 hardware for validation: there it is bit-identical to the WMMA
// instruction (random 64-step chains with denormals, and K's whole network over a moving 24-frame sequence).
WM_FN uint32_t wm_dpp_rot_half1(uint32_t v)
{
    // DPP control 0x12f = row_ror:15 (lane j of a 16-lane row reads lane (j + 1) & 15); row mask 2 = half 1 only
    return (uint32_t)__builtin_amdgcn_update_dpp((int)v, (int)v, 0x12f, 0x2, 0xf, false);
}
// One dot2 per A row with the row_share broadcast folded into the instruction (VOP2 DPP: v_dot2c_f32_f16 on gfx10,
// v_dot2acc_f32_f16 on gfx11). LLVM's DPP combiner does not fold a v_mov_b32_dpp into these, and a separate move
// per row doubled the instruction count and kept 64 broadcast values live (spills). gfx10 and gfx11 resolve the
// VALU -> DPP read dependency in hardware, so no wait states are needed around the assembly.
#ifdef __GFX10__
#define WM_DOT2_DPP "v_dot2c_f32_f16_dpp"
#else
#define WM_DOT2_DPP "v_dot2acc_f32_f16_dpp"
#endif
#define WM_SOFT_ROW(I, LANE)                                                                                       \
    asm(WM_DOT2_DPP " %0, %1, %2 row_share:" #LANE " row_mask:0xf bank_mask:0xf" : "+v"(c[I]) : "v"(x), "v"(column))
WM_FN wm_f8v wm_mma(const wm_op& a, const wm_op& b, wm_f8v c)
{
    // Each accumulator sums its K pairs in order 0..7, as the WMMA instruction does on gfx11 (bit-identical there).
#pragma unroll
    for (int d = 0; d < 8; ++d)
    {
        const uint32_t x = wm_dpp_rot_half1(a[d]);
        const uint32_t column = b[d];
        WM_SOFT_ROW(0, 0);
        WM_SOFT_ROW(1, 2);
        WM_SOFT_ROW(2, 4);
        WM_SOFT_ROW(3, 6);
        WM_SOFT_ROW(4, 8);
        WM_SOFT_ROW(5, 10);
        WM_SOFT_ROW(6, 12);
        WM_SOFT_ROW(7, 14);
    }
    return c;
}
#undef WM_SOFT_ROW
#else
WM_FN wm_f8v wm_mma(const wm_op& a, const wm_op& b, wm_f8v c)
{
    return wm_mma11(a, b, c);
}
#endif
#endif
#if D4R_WMMA_LAYOUT == 12 && defined(__GFX12__)
// operands whose K slots follow ZLUDA's gfx12 order: the same instruction (the order only matters to the shim)
WM_FN wm_f8v wm_mma_zluda(const wm_op& a, const wm_op& b, wm_f8v c)
{
    return wm_mma(a, b, c);
}
#endif

WM_FN wm_f8v wm_splat8(float v)
{
    return (wm_f8v){v, v, v, v, v, v, v, v};
}
