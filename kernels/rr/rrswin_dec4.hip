// Native replacement for the Ray Reconstruction denoiser's last Swin decoder,
// cuda_dldn_engine_swin_dec4_kernel (nvngx_dlssd.dll 310.7).
//
// ---------------------------------------------------------------------------
// STATE: the mirror-cut frame is in place, builds, and *runs* (measured exit 0,
// after carrying the epilogue's six live registers across the cut -- see
// swin_dec4_recipe.py's RET_REGS). Phase 1 (the patch-expand GEMM, 300 mma in
// sixty independent five-step chains) is implemented, but it is NOT yet validated:
// the debug payload cannot be extracted from the launch dumps yet, because the
// bridge deduplicates param_0+48 against arg008's allocation and never writes that
// allocation out, so the oracle's store lands outside every dumped file. Phases
// 2-62 are unimplemented, so the six registers that cross the cut back are still
// lane-distinct placeholders and the layer's surface output does not match the
// translated kernel (compare_launch_dump: arg112 DIFFERS, arg120 ok).
//
// dec4 is the widest of the eleven layers: the attention/MLP width is 160 (dec3's
// is 128), so no count here is interchangeable with another layer's. It has 62
// phases and 1606 mma against enc0's 22 and 536, and its E1 alone is 44% of the
// entry.
//
// The kernel is one warp per 8x8 token patch with eight waves per block, block z
// selecting the slab (%r9 = %tid.z, guarded to 0..3). The PTX this file replaces
// is kernels/extracted/rr/0019-PREPASS_ENTRYPOINT_NAME.ptx.
//
// kernels/tex/make_ptx.py's recipe (kernels/rr/swin_dec4_recipe.py) keeps
// NVIDIA's texture prologue (file lines 1-1706) and surface epilogue (from line
// 58080) and replaces the 1606 mma of the 62 phases (lines 1707-58079) with the
// call below.
//
// Linked into ZLUDA's compile of that PTX (D4R_ZLUDA_EXTRA_BC), so tex_common.h's
// helpers and ZLUDA's own device functions are available. Compiled with -nogpuinc,
// so only plain clang builtins and attributes are used.
#include "../tex/tex_common.h"

// ---------------------------------------------------------------------------
// Parameter block. Fields are addressed by their byte offset in the PTX, so the
// raw image is kept and read through a typed accessor. dec4's block is 144 bytes.
struct DldnSwinDec4Params
{
    uint8_t raw[144];
};
static_assert(sizeof(DldnSwinDec4Params) == 144, "parameter block");

enum : int
{
    kPlaneArg = 48,   // plane arena the decoder writes (arg048), also the debug base's arena
};

// The shared scratch the six registers the surviving epilogue reads cross the cut
// through: the top 768 bytes of `SM` (16384 bytes), twenty-four bytes per lane, at
// 15616. See swin_dec4_recipe.py (RET_BASE, RET_REGS) for why the top of the array is
// free -- SM itself is *not* dead after the cut, E62 reads it back at line 58152, but
// its deepest reach is 6328. A base past 15616 is out of bounds of the array, and the
// compiler then proves the store undefined and drops the whole body, so this constant
// has to match the recipe's.
enum : int
{
    kRetBase = 15616,
    kRetStride = 24,
};

#define DEVF __attribute__((device)) static inline
typedef float lds_f32 __attribute__((address_space(3)));
typedef uint32_t lds_u32 __attribute__((address_space(3)));
typedef uint8_t lds_u8 __attribute__((address_space(3)));

// ---------------------------------------------------------------------------
// The weight image is prepared: `NAME_prep` may permute it once, so the native side
// consumes it in the fragment order its own gemm wants rather than NVIDIA's. What
// follows is the mapping the *splice's* own reading of the image implies.
//
// Phase 1's B block is 25600 bytes: fifty 512-byte tiles at
// `W + z*25600 + 16*laneid + 512*i` (file lines 1478-1674), i.e. one tile per two
// B fragments, read as `ld.weak.global.ca.v4.u32` four b32 at a time. The hundred
// B fragments (%r1017..%r1216, two b32 each) are the mma's B operands in order.
//
// Within one 512-byte tile, byte j of a lane's sixteen holds the e4m3 for the
// (k, n) pair
//   byte = 64*(n & 7) + 8*((n >> 3) & 1) + 16*((k >> 1) & 3)
//        + 4*((k >> 3) & 1) + 2*((k >> 4) & 1) + (k & 1)
// and `16*laneid` is the tile's lane stride. VERIFIED: this is a bijection from the
// 512 (k, n) pairs of a tile onto the tile's 512 bytes, and it agrees with the
// fragment layout enc0's waddr() encodes for the same mma shape (checked over all
// 1024 pairs of a two-tile block), so the two layers' readers see one image.
//
// n here is the *tile-local* column: tile t covers n = 16*t .. 16*t+15, so a phase
// whose B block starts at `base` addresses column `n` of the phase at
// waddr(base, k, n) with n already phase-relative.
DEVF uint32_t waddr(int base, int k, int n)
{
    return (uint32_t)base + 64u * (uint32_t)(n & 7) + 8u * (uint32_t)((n >> 3) & 1)
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

// warp butterfly over the four lanes that hold a row's 32 columns (laneid ^ 1 and
// laneid ^ 2): the two shfl.sync.bfly steps the mma's k-split needs.
DEVF uint32_t bfly1_bits(uint32_t v)
{
    return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0xB1, 0xf, 0xf, false);
}

DEVF uint32_t bfly2_bits(uint32_t v)
{
    return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0x4E, 0xf, 0xf, false);
}

// the low f16 of a packed pair
DEVF half_t half_lo(uint32_t pair)
{
    return __builtin_bit_cast(half_t, (uint16_t)pair);
}

// the high f16 of a packed pair
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
// One mma.m16n8k32.row.col.f16.e4m3.e4m3.f16 step, the shape every phase of dec4
// uses: sixteen rows, eight columns, K=32, e4m3 in, f16 out.
//
// `a` is the A fragment (four b32) in the hardware order enc0's comment documents:
//   A reg 0 -> row g,     k = 2t + (j&1) + 16*((j>>1)&1),  byte j = 0..3
//   A reg 1 -> row g+8,   same k
//   A reg 2 -> row g,     k = 8 + 2t + (j&1) + 16*((j>>1)&1)
//   A reg 3 -> row g+8,   same k
// and the B half that pairs with a0/a1 is b0, with a2/a3 is b1:
//   B byte j (j<4)  -> n = 16*tt + 8*(b>>3) + g, k = 2t + (j&1) + 16*((j>>1)&1)
//   B byte j (j>=4) -> same n, k = 8 + 2t + (j&1) + 16*((j>>1)&1)
// with b the byte index within the lane's sixteen and tt the 512-byte tile.
//
// `bv[2][32]` is therefore the lane's own two columns of the eight, over all 32 k,
// read straight from the weight image; the four lanes that differ in t hold the four
// disjoint eighths of each row's k, so the A side needs both butterfly steps to see
// all 32 k for its own t.
//
// `d` receives the two b32 the mma writes: d[0] is (row g, cols 2t, 2t+1) and d[1]
// is (row g+8, the same two columns), matching the PTX's D-register order.
//
// `c0`/`c1` are the C operand's two b32 (or zero for an unseeded step). e4m3 x e4m3
// products are exact and 32 of them sum exactly in f32, so the dot product is a plain
// f32 accumulation rounded to f16 once -- what the translated mma does.
DEVF void mma_step(const uint32_t* a, const float bv[2][32], uint32_t c0, uint32_t c1,
                   uint32_t d[2])
{
    const int t = (int)(lane_id() & 3);
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    for (int reg = 0; reg < 2; ++reg) {
        for (int row = 0; row < 2; ++row) {
            // the register's value on this lane and on lanes t^1, t^2, t^3
            const uint32_t p0 = a[2 * reg + row];
            const uint32_t p[4] = {p0, bfly1_bits(p0), bfly2_bits(p0), bfly1_bits(bfly2_bits(p0))};
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
    const uint32_t c[2] = {c0, c1};
    for (int row = 0; row < 2; ++row) {
        d[row] = pair_of((half_t)(acc[row][0] + h2f(c[row])),
                         (half_t)(acc[row][1] + h2f(c[row] >> 16)));
    }
}

// ---------------------------------------------------------------------------
// Ground truth for a stage comes from kernels/tex/make_ptx.py's debug recipe: with
// D4R_RRSWIN_DEBUG=<stage> (resolved from kernels/rr/swin_dec4_debug_stages.py) the
// same splice stores that stage's registers into the plane arena at +46,000,000,
// which the kernel never otherwise writes, and the launch dump then holds the
// translated values for a block/lane. Built with -DD4R_RRSWIN_DEBUG_STAGE=<n>
// (kernels/build.sh rrswin, D4R_TEX_CFLAGS), this body stores the same registers of
// its own phase n in the same order, so the two dumps compare byte for byte.
//
// The block index, the per-block slot and the lane stride all mirror
// make_ptx.py's swin_debug_block() exactly, including its 161-wide grid pitch and
// its skip of everything past the first 64 blocks.
#ifndef D4R_RRSWIN_DEBUG_STAGE
#define D4R_RRSWIN_DEBUG_STAGE 0
#endif

// The debug payload's byte offset into the plane arena. It must agree with
// D4R_RRSWIN_DEBUG_BASE, which make_ptx reads for the oracle splice. The default
// 46000000 is wrong for dec4: VERIFIED by storing param_0+48's value into a dumped
// argument, the arena is the 65,945,600-byte block at offset 39,567,360, so
// 39,567,360 + 46,000,000 runs 19.6 MB past the end of the block and the payload
// lands in memory no launch dump contains -- which is why every dec4 stage
// comparison first looked like a MATCH while comparing stale buffer content
// (the same window is byte-identical in the reference run, which stores nothing).
// 16 MiB leaves 64*slot bytes of room inside the block (the tail after the arena is
// 26,378,240) and clears arg008's own window at offset 2,282,752.
#ifndef D4R_RRSWIN_DEBUG_BASE
#define D4R_RRSWIN_DEBUG_BASE 16777216
#endif

enum : int
{
    kDebugBase = D4R_RRSWIN_DEBUG_BASE,
    kDebugBlocks = 64,
    kDebugGridPitch = 161,
};

DEVF uint32_t debug_slot(int nregs)
{
    const uint32_t want = (uint32_t)nregs * 128u > 8192u ? (uint32_t)nregs * 128u : 8192u;
    uint32_t slot = 1;
    while (slot < want) {
        slot <<= 1;
    }
    return slot;
}

DEVF void debug_store(const uint8_t* prm, const uint32_t* regs, int n)
{
    const uint32_t block = __builtin_amdgcn_workgroup_id_x()
                         + kDebugGridPitch * __builtin_amdgcn_workgroup_id_y();
    if (block >= (uint32_t)kDebugBlocks) {
        return;   // the oracle's per-block slots only cover the first 64 blocks
    }
    const uint64_t arena = *(const uint64_t*)(prm + kPlaneArg);
    const uint32_t slot = debug_slot(n);
    uint32_t* dst = (uint32_t*)(arena + (uint64_t)kDebugBase
                                 + ((uint64_t)block * slot)
                                 + (uint64_t)lane_id() * (uint64_t)n * 4ull);
    for (int i = 0; i < n; ++i) {
        dst[i] = regs[i];
    }
}

// ---------------------------------------------------------------------------
// Phase 1: the patch-expand GEMM, the entry's first 300 mma.
//
// Shape: M = 240 (fifteen 16-row A fragments), N = 800 (a hundred 8-column B
// fragments), K = 160 (five k-steps), and 300 mma rather than the 7500 a dense
// 240x800x160 would need -- so this is a band, not a dense GEMM, and the tool's
// M/N columns are its "distinct fragment" counts rather than a real matrix shape.
//
// The mma order (VERIFIED by walking all 300 statements) is: sixty five-step chains,
// each seeded once and then accumulating through C. Statement i uses
//   A fragment  5*(i/100) + (i/2)%5
//   B fragment  i % 100
//   C           the D pair of statement i-2, or the chain's seed when i%10 is 0 or 1
// so chain (g, q, p) -- g = 0..2, q = 0..9, p = 0,1 -- runs the statements
// 100g + 10q + p, +2, +4, +6, +8, carries A fragment 5*g + step, and is seeded with
// the seed register at index 10*q + p of the twenty (3227, 3237, 3327, 3337, ...),
// three chains per seed, one per A group.  The sixty chains are disjoint: their
// accumulators are %r(1218 + 10*i) with i = 100g + 10q + p + 8, so the chain's own
// statement order is also its accumulator order (g major, then q, then p) -- which is
// the order the oracle's p1 stage lists the sixty pairs in.
//
// The B block is [0, 25600) per slab: `W + z*25600 + 16*laneid + 512*i` for the
// hundred fragments, fifty 512-byte tiles, two fragments per tile. Fragment f lives
// at tile f/2 with `8*(f&1)` selecting which half, so its eight columns are
// n = 8*f .. 8*f+7 phase-relative -- which is exactly waddr()'s phase-relative n.
//
// The lane's two columns of fragment f are 2t and 2t+1, so bv[c][k] is read at
// n = 8*f + 2*t + c.
DEVF void phase1(const uint32_t* as, const uint8_t* w, const uint32_t* seeds, int z,
                const uint8_t* prm)
{
    const int t = (int)(lane_id() & 3);
    const int base = z * 25600;
    uint32_t acc[60][2];

    for (int chain = 0; chain < 60; ++chain) {
        const int g = chain / 20;      // the A group: statements 100g..100g+99
        const int sp = chain % 20;     // 10*q + p: the seed index and the B fragment base
        uint32_t c0 = seeds[sp];
        uint32_t c1 = c0;   // PTX: {%r3227, %r3227} -- one u32 in both halves
        for (int step = 0; step < 5; ++step) {
            const int af = 5 * g + step;
            const int bf = sp + 2 * step;
            float bv[2][32];
            for (int c = 0; c < 2; ++c) {
                const int n = 8 * bf + 2 * t + c;
                for (int k = 0; k < 32; ++k) {
                    bv[c][k] = e4m3_to_f32(w[waddr(base, k, n)]);
                }
            }
            mma_step(&as[4 * af], bv, c0, c1, acc[chain]);
            c0 = acc[chain][0];
            c1 = acc[chain][1];
        }
    }
#if D4R_RRSWIN_DEBUG_STAGE == 1
    {
        // the oracle's p1 stage: the sixty chain-final D pairs in mma statement
        // order, each as the (row g, row g+8) pair its mma wrote
        uint32_t regs[120];
        for (int c = 0; c < 60; ++c) {
            regs[2 * c] = acc[c][0];
            regs[2 * c + 1] = acc[c][1];
        }
        debug_store(prm, regs, 120);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 2
    {
        // the oracle's p1a stage: the fifteen A fragments, four b32 each, in the
        // first mma's operand order -- the sixty registers the splice hands over
        uint32_t regs[60];
        for (int i = 0; i < 60; ++i) {
            regs[i] = as[i];
        }
        debug_store(prm, regs, 60);
    }
#endif
#if D4R_RRSWIN_DEBUG_STAGE == 3
    {
        // the oracle's p1s stage: the twenty bias seeds in chain-root order
        uint32_t regs[20];
        for (int i = 0; i < 20; ++i) {
            regs[i] = seeds[i];
        }
        debug_store(prm, regs, 20);
    }
#endif
}

// ---------------------------------------------------------------------------
// Mirror cut. dec4's first phase reads its A operands from the e4m3 feature buffer
// in the prologue (param+8, a clamped halo gather, file lines 1063-1468), so its
// fifteen 16-row fragments arrive here already in registers: %r2130..%r2213,
// %r3130..%r3213 and %r4130..%r4213, four b32 each in the first mma's operand
// order. Then phase 1's twenty bias seeds, the weight image (arg040) and the
// parameter block; see kernels/rr/swin_dec4_recipe.py for the register lists and
// the file-line evidence behind each.
//
// The four accumulators phase 62 leaves live across the cut (%r34761, %r34762,
// %r34771, %r34772, read back out of `SM` by E62's softmax at line 58152) cross it
// through shared memory, not `call.uni`'s return parameters: ZLUDA lowers those to the
// function's LLVM return type (emit.rs's emit_call makes one value, or a struct for
// several, and distributes them with LLVMBuildExtractValue), so there is no
// .param-space object and no pointer is formed -- the callee reads 0x0 in all four
// slots. `scratch` is the base of `SM` (16384 bytes), whose top 128 bytes no
// ld.shared/st.shared in the entry can reach.
extern "C" __attribute__((device)) void d4r_swin_dec4(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6,
    uint32_t a7, uint32_t a8, uint32_t a9, uint32_t a10, uint32_t a11, uint32_t a12, uint32_t a13,
    uint32_t a14, uint32_t a15, uint32_t a16, uint32_t a17, uint32_t a18, uint32_t a19, uint32_t a20,
    uint32_t a21, uint32_t a22, uint32_t a23, uint32_t a24, uint32_t a25, uint32_t a26, uint32_t a27,
    uint32_t a28, uint32_t a29, uint32_t a30, uint32_t a31, uint32_t a32, uint32_t a33, uint32_t a34,
    uint32_t a35, uint32_t a36, uint32_t a37, uint32_t a38, uint32_t a39, uint32_t a40, uint32_t a41,
    uint32_t a42, uint32_t a43, uint32_t a44, uint32_t a45, uint32_t a46, uint32_t a47, uint32_t a48,
    uint32_t a49, uint32_t a50, uint32_t a51, uint32_t a52, uint32_t a53, uint32_t a54, uint32_t a55,
    uint32_t a56, uint32_t a57, uint32_t a58, uint32_t a59,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t b4, uint32_t b5, uint32_t b6,
    uint32_t b7, uint32_t b8, uint32_t b9, uint32_t b10, uint32_t b11, uint32_t b12, uint32_t b13,
    uint32_t b14, uint32_t b15, uint32_t b16, uint32_t b17, uint32_t b18, uint32_t b19,
    const uint8_t* w, const uint8_t* prm, uint32_t scratch)
{
    const uint32_t as1[60] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                              a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23,
                              a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35,
                              a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47,
                              a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59};
    const uint32_t seeds[20] = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9,
                                b10, b11, b12, b13, b14, b15, b16, b17, b18, b19};
    // the slab index: %r9 = %tid.z, guarded to 0..3 by the prologue's `setp.gt.u32`
    const int z = (int)__builtin_amdgcn_workitem_id_z();
    // Address space 3 (the shared scratch) explicitly: a generic store lowers to flat,
    // which faults.  The splice reads the same six words back at SM + 15616 + 24*laneid
    // (swin_dec4_recipe.py's RET_BASE/RET_STRIDE).
    __attribute__((address_space(3))) uint32_t* out =
        ((__attribute__((address_space(3))) uint32_t*)(uintptr_t)scratch)
        + kRetBase / 4 + (uint32_t)lane_id() * (kRetStride / 4);

    // ---- phase 1 -----------------------------------------------------------
    phase1(as1, w, seeds, z, prm);

    // ---- phases 2-62: not implemented yet ----------------------------------
    // The three D pairs the epilogue reads (%r34761/62, %r34771/72, %r34888/89) are
    // still placeholders. Fill them with lane-distinct values so the read-back path is
    // observable without producing wild magnitudes. Every one of the six must be
    // written: a register PTX leaves undefined is poison to ZLUDA's optimizer, and one
    // such survivor folds the whole entry to `unreachable` (measured: HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION).
    for (int i = 0; i < 6; ++i) {
        out[i] = (uint32_t)lane_id() + 32u * (uint32_t)i;
    }
}
