// Checks ZLUDA's emulation of the FP8/matrix PTX instructions used by the
// RRLite DLSS network (rrlite_enc*/dec*) against CPU references built from
// the PTX ISA fragment layouts:
//   cvt.rn.f16x2.e4m3x2                     (exhaustive over all 65536 inputs)
//   cvt.rn.satfinite.e4m3x2.f16x2           (exhaustive over all f16 lanes)
//   cvt.rn.satfinite.relu.e4m3x2.{f16x2,f32} (exhaustive over all f16 values)
//   mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
//   mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
//   movmatrix.sync.aligned.m8n8.trans.b16
// Matrix inputs are small values whose products and sums are exact in f16,
// so results must match bit for bit regardless of accumulation order.
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = uint64_t;

template <typename Function> static Function load_function(void* library, const char* name)
{
    void* raw = dlsym(library, name);
    Function function{};
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
}

static const char* ptx = R"PTX(
.version 8.7
.target sm_89
.address_size 64

.visible .entry d4r_cvt_f16x2_from_e4m3x2(.param .u64 input, .param .u64 output, .param .u32 count)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [count];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $done;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mul.wide.u32 %rd2, %r3, 2;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.u16 %rs0, [%rd3];
    cvt.rn.f16x2.e4m3x2 %r5, %rs0;
    mul.wide.u32 %rd4, %r3, 4;
    add.s64 %rd5, %rd1, %rd4;
    st.global.u32 [%rd5], %r5;
$done:
    ret;
}

.visible .entry d4r_cvt_e4m3x2_from_f16x2(.param .u64 input, .param .u64 output, .param .u32 count)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [count];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $done;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mul.wide.u32 %rd2, %r3, 4;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.u32 %r5, [%rd3];
    cvt.rn.satfinite.e4m3x2.f16x2 %rs0, %r5;
    mul.wide.u32 %rd4, %r3, 2;
    add.s64 %rd5, %rd1, %rd4;
    st.global.u16 [%rd5], %rs0;
$done:
    ret;
}

.visible .entry d4r_cvt_e4m3x2_relu_from_f16x2(.param .u64 input, .param .u64 output, .param .u32 count)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [count];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $done;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mul.wide.u32 %rd2, %r3, 4;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.u32 %r5, [%rd3];
    cvt.rn.satfinite.relu.e4m3x2.f16x2 %rs0, %r5;
    mul.wide.u32 %rd4, %r3, 2;
    add.s64 %rd5, %rd1, %rd4;
    st.global.u16 [%rd5], %rs0;
$done:
    ret;
}

// The same f16x2 inputs widened to f32; the high half is the first operand.
.visible .entry d4r_cvt_e4m3x2_relu_from_f32(.param .u64 input, .param .u64 output, .param .u32 count)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<4>;
    .reg .b32 %r<8>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<8>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [count];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $done;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mul.wide.u32 %rd2, %r3, 4;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.u32 %r5, [%rd3];
    mov.b32 {%rs1, %rs2}, %r5;
    cvt.f32.f16 %f0, %rs1;
    cvt.f32.f16 %f1, %rs2;
    cvt.rn.satfinite.relu.e4m3x2.f32 %rs0, %f1, %f0;
    mul.wide.u32 %rd4, %r3, 2;
    add.s64 %rd5, %rd1, %rd4;
    st.global.u16 [%rd5], %rs0;
$done:
    ret;
}

// Per lane: 4 A words, 2 B words, 2 C words in; 2 D words out.
.visible .entry d4r_mma_m16n8k16_f16(.param .u64 input, .param .u64 output)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 32;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd3];
    ld.global.v4.u32 {%r5, %r6, %r7, %r8}, [%rd3+16];
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r9, %r10}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r7, %r8};
    mul.wide.u32 %rd4, %r0, 8;
    add.s64 %rd5, %rd1, %rd4;
    st.global.v2.u32 [%rd5], {%r9, %r10};
    ret;
}

// Chained MMAs in a loop, two per step sharing A (the pattern of DLSS's
// convolutions): per lane 8 steps x (4 A, 2 B0, 2 B1 words), then C0 and C1
// (2 words each); out D0 and D1. Exercises accumulators carried through a
// loop and pairs of MMAs that share A.
.visible .entry d4r_mma_chain_f16(.param .u64 input, .param .u64 output)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 272;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.v4.u32 {%r9, %r10, %r11, %r12}, [%rd3+256];
    mov.u32 %r13, 0;
$L_chain:
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd3];
    ld.global.v4.u32 {%r5, %r6, %r7, %r8}, [%rd3+16];
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r9, %r10}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r9, %r10};
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r11, %r12}, {%r1, %r2, %r3, %r4}, {%r7, %r8}, {%r11, %r12};
    add.s64 %rd3, %rd3, 32;
    add.u32 %r13, %r13, 1;
    setp.lt.u32 %p0, %r13, 8;
    @%p0 bra $L_chain;
    mul.wide.u32 %rd4, %r0, 16;
    add.s64 %rd5, %rd1, %rd4;
    st.global.v4.u32 [%rd5], {%r9, %r10, %r11, %r12};
    ret;
}

// The same chain with m16n8k32 e4m3 MMAs (A 4 words, B 2 words: same sizes).
.visible .entry d4r_mma_chain_e4m3(.param .u64 input, .param .u64 output)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 272;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.v4.u32 {%r9, %r10, %r11, %r12}, [%rd3+256];
    mov.u32 %r13, 0;
$L_chain8:
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd3];
    ld.global.v4.u32 {%r5, %r6, %r7, %r8}, [%rd3+16];
    mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r9, %r10}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r9, %r10};
    mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r11, %r12}, {%r1, %r2, %r3, %r4}, {%r7, %r8}, {%r11, %r12};
    add.s64 %rd3, %rd3, 32;
    add.u32 %r13, %r13, 1;
    setp.lt.u32 %p0, %r13, 8;
    @%p0 bra $L_chain8;
    mul.wide.u32 %rd4, %r0, 16;
    add.s64 %rd5, %rd1, %rd4;
    st.global.v4.u32 [%rd5], {%r9, %r10, %r11, %r12};
    ret;
}

// Two e4m3 MMAs sharing A whose second accumulator is one zeroed register
// used for both halves ({%r20, %r20}), as DLSS's first layers do. In the
// "late" kernel that register is rewritten between the two MMAs, which must
// keep them apart. Per lane: 4 A, 2 B0, 2 B1, 2 C0 words and a value for
// %r20; out D0, D1.
.visible .entry d4r_mma_pair_alias_e4m3(.param .u64 input, .param .u64 output, .param .u32 late)
{
    .reg .b32 %r<24>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r22, [late];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 48;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd3];
    ld.global.v4.u32 {%r5, %r6, %r7, %r8}, [%rd3+16];
    ld.global.v4.u32 {%r9, %r10, %r11, %r12}, [%rd3+32];
    setp.ne.u32 %p0, %r22, 0;
    mov.u32 %r20, 0;
    mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r13, %r14}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r9, %r10};
    mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r15, %r16}, {%r1, %r2, %r3, %r4}, {%r7, %r8}, {%r20, %r20};
    mul.wide.u32 %rd4, %r0, 16;
    add.s64 %rd5, %rd1, %rd4;
    st.global.v4.u32 [%rd5], {%r13, %r14, %r15, %r16};
    ret;
}

.visible .entry d4r_mma_pair_alias_late_e4m3(.param .u64 input, .param .u64 output, .param .u32 late)
{
    .reg .b32 %r<24>;
    .reg .b64 %rd<8>;
    .reg .pred %p<2>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r22, [late];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 48;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd3];
    ld.global.v4.u32 {%r5, %r6, %r7, %r8}, [%rd3+16];
    ld.global.v4.u32 {%r9, %r10, %r11, %r12}, [%rd3+32];
    setp.ne.u32 %p0, %r22, 0;
    mov.u32 %r20, 0;
    mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r13, %r14}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r9, %r10};
    selp.b32 %r20, %r11, 0, %p0;
    mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r15, %r16}, {%r1, %r2, %r3, %r4}, {%r7, %r8}, {%r20, %r20};
    mul.wide.u32 %rd4, %r0, 16;
    add.s64 %rd5, %rd1, %rd4;
    st.global.v4.u32 [%rd5], {%r13, %r14, %r15, %r16};
    ret;
}

.visible .entry d4r_mma_m16n8k8_f16(.param .u64 input, .param .u64 output)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 32;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd3];
    ld.global.v4.u32 {%r5, %r6, %r7, %r8}, [%rd3+16];
    mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 
{%r9, %r10}, 
{%r1, %r2}, 
{%r5}, 
{%r7, %r8};
    mul.wide.u32 %rd4, %r0, 8;
    add.s64 %rd5, %rd1, %rd4;
    st.global.v2.u32 [%rd5], {%r9, %r10};
    ret;
}

// fma.rn.sat.f16x2 and fma.rn.relu.f16x2 as used by the DLTSS model kernels.
// Per thread: 3 input words, 3 output words (DLTSS writes .relu after the type).
.visible .entry d4r_f16x2_fma_modes(.param .u64 input, .param .u64 output, .param .u32 count)
{
    .reg .pred %p<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [count];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $fma_done;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mul.wide.u32 %rd2, %r3, 12;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.u32 %r5, [%rd3];
    ld.global.u32 %r6, [%rd3+4];
    ld.global.u32 %r7, [%rd3+8];
    fma.rn.sat.f16x2 %r8, %r5, %r6, %r7;
    fma.rn.relu.f16x2 %r9, %r5, %r6, %r7;
    fma.rn.f16x2.relu %r10, %r5, %r6, %r7;
    mul.wide.u32 %rd4, %r3, 12;
    add.s64 %rd5, %rd1, %rd4;
    st.global.v2.u32 [%rd5], {%r8, %r9};
    st.global.u32 [%rd5+8], %r10;
$fma_done:
    ret;
}

// Forms used only by the DLTSS model kernels: ldmatrix x4/x2 on 32-bit shared
// addresses, prmt selectors, set.equ.u32.f16x2, dp2a.lo and cp.async.cg with a
// source size. One warp; per lane 16 input words and 20 output words.
.visible .entry d4r_dltss_misc(.param .u64 input, .param .u64 output)
{
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .shared .align 16 .b8 smem_[1024];
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 64;
    add.s64 %rd3, %rd0, %rd2;
    mul.wide.u32 %rd4, %r0, 80;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd3];
    mov.u32 %r5, smem_;
    shl.b32 %r6, %r0, 4;
    add.s32 %r7, %r5, %r6;
    st.shared.v4.b32 [%r7], {%r1, %r2, %r3, %r4};
    bar.sync 0;
    // Row address: lane l points at 16-byte row (5 * l) mod 32.
    mul.lo.u32 %r8, %r0, 5;
    and.b32 %r8, %r8, 31;
    shl.b32 %r8, %r8, 4;
    add.s32 %r9, %r5, %r8;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r10, %r11, %r12, %r13}, [%r9];
    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r14, %r15}, [%r9];
    st.global.v4.u32 [%rd5], {%r10, %r11, %r12, %r13};
    st.global.v2.u32 [%rd5+16], {%r14, %r15};
    ld.global.v4.u32 {%r16, %r17, %r18, %r19}, [%rd3+16];
    prmt.b32 %r20, %r16, 0, 0x3424;
    prmt.b32 %r21, %r16, 0, 0x1404;
    prmt.b32 %r22, %r16, 0, 0x4431;
    prmt.b32 %r23, %r16, %r17, 0x7632;
    prmt.b32 %r24, %r16, %r17, 0x5410;
    prmt.b32 %r25, %r16, %r17, 0x7531;
    set.equ.u32.f16x2 %r26, %r16, %r19;
    dp2a.lo.u32.u32 %r27, %r16, %r17, %r18;
    st.global.v4.u32 [%rd5+24], {%r20, %r21, %r22, %r23};
    st.global.v4.u32 [%rd5+40], {%r24, %r25, %r26, %r27};
    // cp.async of 16 bytes with src-size = w12 (zero fill past it).
    ld.global.u32 %r28, [%rd3+48];
    add.s64 %rd6, %rd3, 32;
    add.s32 %r29, %r7, 512;
    cp.async.cg.shared.global [%r29], [%rd6], 16, %r28;
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;
    ld.shared.v4.b32 {%r30, %r31, %r32, %r33}, [%r29];
    st.global.v4.u32 [%rd5+56], {%r30, %r31, %r32, %r33};
    ret;
}

// Rounding forms from the DLTSS output kernel's source-row mapping
// (round half away from zero via add.rz + cvt.rzi). Per thread: 2 input
// floats, 6 output words.
.visible .entry d4r_round_ops(.param .u64 input, .param .u64 output, .param .u32 count)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .f32 %f<16>;
    .reg .b64 %rd<8>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [count];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $round_done;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mul.wide.u32 %rd2, %r3, 8;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.v2.f32 {%f0, %f1}, [%rd3];
    add.rz.ftz.f32 %f2, %f0, %f1;
    mov.b32 %r5, %f0;
    and.b32 %r6, %r5, -2147483648;
    or.b32 %r7, %r6, 1056964608;
    mov.b32 %f3, %r7;
    add.rz.ftz.f32 %f4, %f0, %f3;
    cvt.rzi.f32.f32 %f5, %f4;
    cvt.rzi.ftz.s32.f32 %r8, %f5;
    cvt.rzi.f32.f32 %f6, %f0;
    add.ftz.f32 %f7, %f0, %f1;
    mov.u32 %r9, 0;
    mov.u32 %r10, 31;
    mov.u32 %r11, -1;
    shfl.sync.idx.b32 %r12|%p1, %r3, %r9, %r10, %r11;
    selp.b32 %r13, 1, 0, %p1;
    // bfi as in the DLTSS output kernel's ldmatrix row addressing.
    and.b32 %r5, %r3, 8;
    and.b32 %r6, %r3, 7;
    bfi.b32 %r7, %r5, %r6, 3, 29;
    bfi.b32 %r9, %r3, %r6, 4, 8;
    mul.wide.u32 %rd4, %r3, 36;
    add.s64 %rd5, %rd1, %rd4;
    mov.b32 %r14, %f2;
    mov.b32 %r15, %f5;
    st.global.v4.u32 [%rd5], {%r14, %r15, %r8, %r12};
    mov.b32 %r14, %f6;
    mov.b32 %r15, %f7;
    st.global.v4.u32 [%rd5+16], {%r14, %r15, %r13, %r7};
    st.global.u32 [%rd5+32], %r9;
$round_done:
    ret;
}

.visible .entry d4r_mma_m16n8k32_e4m3(.param .u64 input, .param .u64 output)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 32;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd3];
    ld.global.v4.u32 {%r5, %r6, %r7, %r8}, [%rd3+16];
    mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r9, %r10}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r7, %r8};
    mul.wide.u32 %rd4, %r0, 8;
    add.s64 %rd5, %rd1, %rd4;
    st.global.v2.u32 [%rd5], {%r9, %r10};
    ret;
}

.visible .entry d4r_movmatrix_trans(.param .u64 input, .param .u64 output)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 4;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.u32 %r1, [%rd3];
    movmatrix.sync.aligned.m8n8.trans.b16 %r2, %r1;
    add.s64 %rd5, %rd1, %rd2;
    st.global.u32 [%rd5], %r2;
    ret;
}

.visible .entry d4r_global_v2_u16(.param .u64 input, .param .u64 output)
{
    .reg .b16 %rs<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r0, %laneid;
    mul.wide.u32 %rd2, %r0, 4;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.v2.u16 {%rs0, %rs1}, [%rd3];
    mov.b32 %r1, {%rs0, %rs1};
    mul.wide.u32 %rd4, %r0, 8;
    add.s64 %rd5, %rd1, %rd4;
    st.global.u32 [%rd5], %r1;
    st.global.v2.u16 [%rd5+4], {%rs0, %rs1};
    ret;
}

.visible .entry d4r_ex2_f16x2(.param .u64 input, .param .u64 output, .param .u32 count)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [count];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $done;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    mul.wide.u32 %rd2, %r3, 4;
    add.s64 %rd3, %rd0, %rd2;
    add.s64 %rd4, %rd1, %rd2;
    ld.global.u32 %r5, [%rd3];
    ex2.approx.f16x2 %r6, %r5;
    st.global.u32 [%rd4], %r6;
$done:
    ret;
}

.visible .entry d4r_multidim_lane(.param .u64 output)
.maxntid 32, 1, 8
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd0, [output];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %tid.y;
    mov.u32 %r2, %tid.z;
    mov.u32 %r3, %laneid;
    mad.lo.u32 %r4, %r2, 32, %r0;
    mul.wide.u32 %rd1, %r4, 16;
    add.s64 %rd2, %rd0, %rd1;
    st.global.v4.u32 [%rd2], {%r0, %r1, %r2, %r3};
    ret;
}
// Packed and scalar f16 arithmetic as emitted by cuda_fp16.h inline asm in
// the RRLite kernels, including nested register scopes that redeclare the
// same names. Per thread: 3 input words, 13 output words.
.visible .entry d4r_f16x2_ops(.param .u64 input, .param .u64 output, .param .u32 count)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
    .reg .f64 %fd<2>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.s32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [count];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $done;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    mul.wide.u32 %rd2, %r3, 12;
    add.s64 %rd3, %rd0, %rd2;
    ld.global.u32 %r5, [%rd3];
    ld.global.u32 %r6, [%rd3+4];
    ld.global.u32 %r7, [%rd3+8];
    {abs.f16x2 %r10,%r5;
    }
    {neg.f16x2 %r11,%r5;
    }
    {min.f16x2 %r12,%r5,%r6;
    }
    {max.f16x2 %r13,%r5,%r6;
    }
    {add.f16x2 %r14,%r5,%r6;
    }
    {sub.f16x2 %r15,%r5,%r6;
    }
    {mul.f16x2 %r16,%r5,%r6;
    }
    {fma.rn.f16x2 %r17,%r5,%r6,%r7;
    }
    mov.b32 {%rs0, %rs1}, %r5;
    mov.b32 {%rs2, %rs3}, %r6;
    {add.f16 %rs4,%rs0,%rs2;
    }
    {mul.f16 %rs5,%rs1,%rs3;
    }
    mov.b32 %r18, {%rs4, %rs5};
    mov.f32 %f0, 0f3F000000;
    {.reg .f16 low;
    cvt.rn.f16.f32 low, %f0;
    mov.b32 %r20, {low,low};}
    mov.f32 %f1, 0f40000000;
    {.reg .f16 low;
    cvt.rn.f16.f32 low, %f1;
    mov.b32 %r21, {low,low};}
    mov.f64 %fd0, 0dBFE1CC0000000000;
    { cvt.rn.f16.f64 %rs6, %fd0;}
    mov.f64 %fd1, 0d3FF0000000000000;
    { cvt.rn.f16.f64 %rs7, %fd1;}
    mov.b32 %r22, {%rs6, %rs7};
    {.reg.b16 hl, hu;
    .reg.b32 fl, fu;
    mov.b32 {hl, hu}, %r10;
    cvt.f32.f16 fl, hl;
    cvt.f32.f16 fu, hu;
    rsqrt.approx.ftz.f32 fl, fl;
    rsqrt.approx.ftz.f32 fu, fu;
    cvt.rn.f16.f32 hl, fl;
    cvt.rn.f16.f32 hu, fu;
    mov.b32 %r23, {hl, hu};
    }
    mul.wide.u32 %rd4, %r3, 52;
    add.s64 %rd5, %rd1, %rd4;
    st.global.v4.u32 [%rd5], {%r10, %r11, %r12, %r13};
    st.global.v4.u32 [%rd5+16], {%r14, %r15, %r16, %r17};
    st.global.v4.u32 [%rd5+32], {%r18, %r20, %r21, %r22};
    st.global.u32 [%rd5+48], %r23;
$done:
    ret;
}
)PTX";

// --- CPU references ---------------------------------------------------------

static float half_to_float(uint16_t bits)
{
    const uint32_t sign = (bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    if (exponent == 0)
        return std::ldexp(static_cast<float>(mantissa), -24) * (sign ? -1.0f : 1.0f);
    uint32_t result = exponent == 0x1f ? sign | 0x7f800000u | (mantissa << 13)
                                       : sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

// Exact for every value representable in f16 (all E4M3 values and all the
// matrix results used below).
static uint16_t float_to_half_exact(float value)
{
    for (uint32_t bits = 0; bits < 0x10000u; ++bits)
        if (half_to_float(static_cast<uint16_t>(bits)) == value &&
            std::signbit(half_to_float(static_cast<uint16_t>(bits))) == std::signbit(value))
            return static_cast<uint16_t>(bits);
    return 0x7e00u;
}

// OCP FP8 E4M3 (NVIDIA e4m3): bias 7, no infinities, S.1111.111 is NaN.
static float e4m3_to_float(uint8_t bits)
{
    const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
    const int exponent = (bits >> 3) & 0xf;
    const int mantissa = bits & 0x7;
    if (exponent == 0xf && mantissa == 0x7)
        return std::nanf("");
    if (exponent == 0)
        return sign * std::ldexp(static_cast<float>(mantissa), -9);
    return sign * std::ldexp(1.0f + mantissa / 8.0f, exponent - 7);
}

// cvt.rn.satfinite: round to nearest even, finite overflow and infinities
// saturate to +-448, NaN stays NaN.
static uint8_t float_to_e4m3_satfinite(float value)
{
    if (std::isnan(value))
        return 0x7f;
    const uint8_t sign = std::signbit(value) ? 0x80u : 0u;
    const float magnitude = std::fabs(value);
    if (magnitude >= 448.0f)
        return sign | 0x7e;
    uint8_t best = 0;
    float best_error = INFINITY;
    for (uint8_t code = 0; code <= 0x7e; ++code)
    {
        const float error = std::fabs(e4m3_to_float(code) - magnitude);
        if (error < best_error || (error == best_error && (code & 1u) == 0))
        {
            best = code;
            best_error = error;
        }
    }
    return sign | best;
}

// PTX ISA fragment layouts. groupID = lane / 4, threadID_in_group = lane % 4.
// mma.m16n8k16 .f16 A (16x16): a[i] for i in 0..7, packed two per register.
static void layout_a_f16_k16(int lane, int i, int& row, int& col)
{
    const int group = lane >> 2, thread = lane & 3;
    row = (i < 2 || (i >= 4 && i < 6)) ? group : group + 8;
    col = thread * 2 + (i & 1) + (i >= 4 ? 8 : 0);
}
// mma.m16n8k16 .f16 B (16x8): b[i] for i in 0..3.
static void layout_b_f16_k16(int lane, int i, int& row, int& col)
{
    const int group = lane >> 2, thread = lane & 3;
    row = thread * 2 + (i & 1) + (i >= 2 ? 8 : 0);
    col = group;
}
// f16 accumulator C/D (16x8): c[i] for i in 0..3.
static void layout_c_f16(int lane, int i, int& row, int& col)
{
    const int group = lane >> 2, thread = lane & 3;
    row = i < 2 ? group : group + 8;
    col = thread * 2 + (i & 1);
}
// mma.m16n8k8 .f16 A (16x8): a[i] for i in 0..3; B (8x8): b[i] for i in 0..1.
static void layout_a_f16_k8(int lane, int i, int& row, int& col)
{
    const int group = lane >> 2, thread = lane & 3;
    row = i < 2 ? group : group + 8;
    col = thread * 2 + (i & 1);
}
static void layout_b_f16_k8(int lane, int i, int& row, int& col)
{
    const int group = lane >> 2, thread = lane & 3;
    row = thread * 2 + (i & 1);
    col = group;
}
// mma.m16n8k32 8-bit A (16x32): a[i] for i in 0..15, four per register.
static void layout_a_8bit_k32(int lane, int i, int& row, int& col)
{
    const int group = lane >> 2, thread = lane & 3;
    row = (i < 4 || (i >= 8 && i < 12)) ? group : group + 8;
    col = thread * 4 + (i & 3) + (i >= 8 ? 16 : 0);
}
// mma.m16n8k32 8-bit B (32x8): b[i] for i in 0..7.
static void layout_b_8bit_k32(int lane, int i, int& row, int& col)
{
    const int group = lane >> 2, thread = lane & 3;
    row = thread * 4 + (i & 3) + (i >= 4 ? 16 : 0);
    col = group;
}

// --- driver plumbing --------------------------------------------------------

struct Driver
{
    using InitFn = CUresult (*)(unsigned int);
    using DeviceGetFn = CUresult (*)(CUdevice*, int);
    using ContextCreateFn = CUresult (*)(CUcontext*, unsigned int, CUdevice);
    using ModuleLoadFn = CUresult (*)(CUmodule*, const void*);
    using ModuleGetFunctionFn = CUresult (*)(CUfunction*, CUmodule, const char*);
    using AllocFn = CUresult (*)(CUdeviceptr*, size_t);
    using FreeFn = CUresult (*)(CUdeviceptr);
    using CopyHtoDFn = CUresult (*)(CUdeviceptr, const void*, size_t);
    using CopyDtoHFn = CUresult (*)(void*, CUdeviceptr, size_t);
    using LaunchKernelFn = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int, unsigned int,
                                        CUstream, void**, void**);
    using SynchronizeFn = CUresult (*)(void);

    InitFn init;
    DeviceGetFn deviceGet;
    ContextCreateFn contextCreate;
    ModuleLoadFn moduleLoad;
    ModuleGetFunctionFn moduleGetFunction;
    AllocFn alloc;
    FreeFn free;
    CopyHtoDFn copyHtoD;
    CopyDtoHFn copyDtoH;
    LaunchKernelFn launch;
    SynchronizeFn synchronize;
    CUmodule module = nullptr;

    bool load(void* library)
    {
        init = load_function<InitFn>(library, "cuInit");
        deviceGet = load_function<DeviceGetFn>(library, "cuDeviceGet");
        contextCreate = load_function<ContextCreateFn>(library, "cuCtxCreate_v2");
        moduleLoad = load_function<ModuleLoadFn>(library, "cuModuleLoadData");
        moduleGetFunction = load_function<ModuleGetFunctionFn>(library, "cuModuleGetFunction");
        alloc = load_function<AllocFn>(library, "cuMemAlloc_v2");
        free = load_function<FreeFn>(library, "cuMemFree_v2");
        copyHtoD = load_function<CopyHtoDFn>(library, "cuMemcpyHtoD_v2");
        copyDtoH = load_function<CopyDtoHFn>(library, "cuMemcpyDtoH_v2");
        launch = load_function<LaunchKernelFn>(library, "cuLaunchKernel");
        synchronize = load_function<SynchronizeFn>(library, "cuCtxSynchronize");
        return init && deviceGet && contextCreate && moduleLoad && moduleGetFunction && alloc &&
               free && copyHtoD && copyDtoH && launch && synchronize;
    }

    // Runs kernel(input, output[, count]) over one buffer each way.
    bool run(const char* name, const void* input, size_t inputBytes, void* output,
             size_t outputBytes, unsigned int blocks, unsigned int threads, int count)
    {
        CUfunction function = nullptr;
        CUdeviceptr deviceInput = 0, deviceOutput = 0;
        CUresult result = moduleGetFunction(&function, module, name);
        if (result == 0)
            result = alloc(&deviceInput, inputBytes);
        if (result == 0)
            result = alloc(&deviceOutput, outputBytes);
        if (result == 0)
            result = copyHtoD(deviceInput, input, inputBytes);
        if (result == 0)
        {
            void* withCount[] = {&deviceInput, &deviceOutput, &count};
            void* withoutCount[] = {&deviceInput, &deviceOutput};
            result = launch(function, blocks, 1, 1, threads, 1, 1, 0, nullptr,
                            count >= 0 ? withCount : withoutCount, nullptr);
        }
        if (result == 0)
            result = synchronize();
        if (result == 0)
            result = copyDtoH(output, deviceOutput, outputBytes);
        if (deviceInput != 0)
            free(deviceInput);
        if (deviceOutput != 0)
            free(deviceOutput);
        if (result != 0)
            std::printf("%s: CUDA result=%d\n", name, result);
        return result == 0;
    }
};

// --- tests ------------------------------------------------------------------

static bool test_cvt_from_e4m3(Driver& driver)
{
    std::vector<uint16_t> input(65536);
    for (uint32_t index = 0; index < input.size(); ++index)
        input[index] = static_cast<uint16_t>(index);
    std::vector<uint32_t> output(input.size());
    if (!driver.run("d4r_cvt_f16x2_from_e4m3x2", input.data(), input.size() * 2, output.data(),
                    output.size() * 4, 256, 256, static_cast<int>(input.size())))
        return false;
    size_t mismatches = 0;
    for (uint32_t index = 0; index < input.size(); ++index)
    {
        for (int lane = 0; lane < 2; ++lane)
        {
            const float expected = e4m3_to_float(static_cast<uint8_t>(index >> (8 * lane)));
            const uint16_t got_bits = static_cast<uint16_t>(output[index] >> (16 * lane));
            const float got = half_to_float(got_bits);
            const bool ok = std::isnan(expected) ? std::isnan(got)
                : got == expected && std::signbit(got) == std::signbit(expected);
            if (!ok && mismatches++ < 8)
                std::printf("  e4m3x2 0x%04x lane %d: got f16 0x%04x (%g) expected %g\n", index,
                            lane, got_bits, got, expected);
        }
    }
    std::printf("cvt.rn.f16x2.e4m3x2 (65536 inputs): %s (%zu mismatches)\n",
                mismatches == 0 ? "PASS" : "FAIL", mismatches);
    return mismatches == 0;
}

static bool test_cvt_to_e4m3(Driver& driver, const char* kernel = "d4r_cvt_e4m3x2_from_f16x2",
                             const char* label = "cvt.rn.satfinite.e4m3x2.f16x2", bool relu = false)
{
    // Lane 0 sweeps every f16 value while lane 1 sweeps them in reverse.
    std::vector<uint32_t> input(65536);
    for (uint32_t index = 0; index < input.size(); ++index)
        input[index] = index | ((0xffffu - index) << 16);
    std::vector<uint16_t> output(input.size());
    if (!driver.run(kernel, input.data(), input.size() * 4, output.data(),
                    output.size() * 2, 256, 256, static_cast<int>(input.size())))
        return false;
    size_t mismatches = 0;
    for (uint32_t index = 0; index < input.size(); ++index)
    {
        for (int lane = 0; lane < 2; ++lane)
        {
            const uint16_t half = static_cast<uint16_t>(input[index] >> (16 * lane));
            // .relu: negative results (and -0) become +0, NaN stays NaN.
            const float value = half_to_float(half);
            const uint8_t expected = relu && !std::isnan(value) && std::signbit(value)
                ? 0 : float_to_e4m3_satfinite(value);
            const uint8_t got = static_cast<uint8_t>(output[index] >> (8 * lane));
            const bool ok = (expected & 0x7f) == 0x7f ? (got & 0x7f) == 0x7f : got == expected;
            if (!ok && mismatches++ < 8)
                std::printf("  f16 0x%04x (%g) lane %d: got e4m3 0x%02x expected 0x%02x\n", half,
                            half_to_float(half), lane, got, expected);
        }
    }
    std::printf("%s (131072 lanes): %s (%zu mismatches)\n", label,
                mismatches == 0 ? "PASS" : "FAIL", mismatches);
    return mismatches == 0;
}

// Runs one MMA through the fragment layouts and compares D with A*B+C.
template <typename ALayout, typename BLayout>
static bool test_mma(Driver& driver, const char* kernel, const char* label, int k, int elementBits,
                     ALayout aLayout, BLayout bLayout, const std::vector<float>& a,
                     const std::vector<float>& b, const std::vector<float>& c,
                     const std::vector<uint8_t>& aCodes, const std::vector<uint8_t>& bCodes)
{
    uint32_t input[32][8] = {};
    // 16x k A and k x 8 B fragments spread over 32 lanes.
    const int aElements = k / 2;
    const int bElements = k / 4;
    const int perWord = 32 / elementBits;
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int i = 0; i < aElements; ++i)
        {
            int row, col;
            aLayout(lane, i, row, col);
            const uint32_t bits = elementBits == 16 ? float_to_half_exact(a[row * k + col])
                                                    : aCodes[row * k + col];
            input[lane][i / perWord] |= bits << (elementBits * (i % perWord));
        }
        for (int i = 0; i < bElements; ++i)
        {
            int row, col;
            bLayout(lane, i, row, col);
            const uint32_t bits = elementBits == 16 ? float_to_half_exact(b[row * 8 + col])
                                                    : bCodes[row * 8 + col];
            input[lane][4 + i / perWord] |= bits << (elementBits * (i % perWord));
        }
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            input[lane][6 + i / 2] |= static_cast<uint32_t>(float_to_half_exact(c[row * 8 + col]))
                                      << (16 * (i % 2));
        }
    }
    uint32_t output[32][2] = {};
    if (!driver.run(kernel, input, sizeof(input), output, sizeof(output), 1, 32, -1))
        return false;
    size_t mismatches = 0;
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            float expected = c[row * 8 + col];
            for (int kk = 0; kk < k; ++kk)
                expected += a[row * k + kk] * b[kk * 8 + col];
            const float got = half_to_float(static_cast<uint16_t>(output[lane][i / 2] >> (16 * (i % 2))));
            if (got != expected && mismatches++ < 6)
                std::printf("  %s D[%d][%d] (lane %d, d%d): got %g expected %g\n", label, row, col,
                            lane, i, got, expected);
        }
    }
    std::printf("%s: %s (%zu of 128 outputs mismatched)\n", label, mismatches == 0 ? "PASS" : "FAIL",
                mismatches);
    return mismatches == 0;
}

// Order-independent f16 distance in units in the last place.
static int half_ulp_distance(uint16_t a, uint16_t b)
{
    auto ordered = [](uint16_t bits) { return (bits & 0x8000u) ? -static_cast<int>(bits & 0x7fffu) : static_cast<int>(bits); };
    return std::abs(ordered(a) - ordered(b));
}

// Random f16 inputs through d4r_mma_chain_f16, against a reference that sums
// each MMA exactly (in double) and rounds the accumulator to f16 once per MMA.
// Tensor cores and the translations differ only in summation order inside one
// MMA, so a result may be off by a few ulp; layout errors are off by far more.
static bool test_mma_chain(Driver& driver, int trial)
{
    std::mt19937 random(1234 + trial);
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
    auto pick_half = [&]() { return static_cast<float>(static_cast<_Float16>(uniform(random))); };
    constexpr int kSteps = 8;
    std::vector<float> a(kSteps * 16 * 16), b0(kSteps * 16 * 8), b1(kSteps * 16 * 8), c0(16 * 8), c1(16 * 8);
    for (float& value : a) value = pick_half();
    for (float& value : b0) value = pick_half();
    for (float& value : b1) value = pick_half();
    for (float& value : c0) value = pick_half();
    for (float& value : c1) value = pick_half();
    static uint32_t input[32][68];
    std::memset(input, 0, sizeof(input));
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int step = 0; step < kSteps; ++step)
        {
            uint32_t* words = input[lane] + step * 8;
            for (int i = 0; i < 8; ++i)
            {
                int row, col;
                layout_a_f16_k16(lane, i, row, col);
                words[i / 2] |= static_cast<uint32_t>(float_to_half_exact(a[step * 256 + row * 16 + col])) << (16 * (i % 2));
            }
            for (int i = 0; i < 4; ++i)
            {
                int row, col;
                layout_b_f16_k16(lane, i, row, col);
                words[4 + i / 2] |= static_cast<uint32_t>(float_to_half_exact(b0[step * 128 + row * 8 + col])) << (16 * (i % 2));
                words[6 + i / 2] |= static_cast<uint32_t>(float_to_half_exact(b1[step * 128 + row * 8 + col])) << (16 * (i % 2));
            }
        }
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            input[lane][64 + i / 2] |= static_cast<uint32_t>(float_to_half_exact(c0[row * 8 + col])) << (16 * (i % 2));
            input[lane][66 + i / 2] |= static_cast<uint32_t>(float_to_half_exact(c1[row * 8 + col])) << (16 * (i % 2));
        }
    }
    uint32_t output[32][4] = {};
    if (!driver.run("d4r_mma_chain_f16", input, sizeof(input), output, sizeof(output), 1, 32, -1))
        return false;
    std::vector<uint16_t> reference0(128), reference1(128);
    std::vector<double> acc0(c0.begin(), c0.end()), acc1(c1.begin(), c1.end());
    for (int step = 0; step < kSteps; ++step)
        for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 8; ++col)
            {
                double sum0 = acc0[row * 8 + col], sum1 = acc1[row * 8 + col];
                for (int k = 0; k < 16; ++k)
                {
                    sum0 += static_cast<double>(a[step * 256 + row * 16 + k]) * b0[step * 128 + k * 8 + col];
                    sum1 += static_cast<double>(a[step * 256 + row * 16 + k]) * b1[step * 128 + k * 8 + col];
                }
                acc0[row * 8 + col] = static_cast<double>(static_cast<_Float16>(sum0));
                acc1[row * 8 + col] = static_cast<double>(static_cast<_Float16>(sum1));
            }
    int worst = 0, exact = 0;
    for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            const uint16_t got0 = static_cast<uint16_t>(output[lane][i / 2] >> (16 * (i % 2)));
            const uint16_t got1 = static_cast<uint16_t>(output[lane][2 + i / 2] >> (16 * (i % 2)));
            const _Float16 want0 = static_cast<_Float16>(acc0[row * 8 + col]);
            const _Float16 want1 = static_cast<_Float16>(acc1[row * 8 + col]);
            uint16_t bits0, bits1;
            std::memcpy(&bits0, &want0, 2);
            std::memcpy(&bits1, &want1, 2);
            const int d0 = half_ulp_distance(got0, bits0), d1 = half_ulp_distance(got1, bits1);
            worst = std::max(worst, std::max(d0, d1));
            exact += (d0 == 0) + (d1 == 0);
        }
    const bool ok = worst <= 8;
    std::printf("mma.m16n8k16.f16 chained x%d, paired, trial %d: %s (worst %d ulp, %d of 256 exact)\n", kSteps,
                trial, ok ? "PASS" : "FAIL", worst, exact);
    return ok;
}

// Full-range e4m3 codes (normals, subnormals, zeros; no NaN) through one
// m16n8k32 MMA against a double reference rounded to f16 once. The f32
// summation order may differ from the reference, so allow 2 ulp.
static bool test_mma_e4m3_codes(Driver& driver, const char* label, const std::vector<uint8_t>& aCodes,
                                const std::vector<uint8_t>& bCodes, const std::vector<float>& c)
{
    uint32_t input[32][8] = {};
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int i = 0; i < 16; ++i)
        {
            int row, col;
            layout_a_8bit_k32(lane, i, row, col);
            input[lane][i / 4] |= static_cast<uint32_t>(aCodes[row * 32 + col]) << (8 * (i % 4));
        }
        for (int i = 0; i < 8; ++i)
        {
            int row, col;
            layout_b_8bit_k32(lane, i, row, col);
            input[lane][4 + i / 4] |= static_cast<uint32_t>(bCodes[row * 8 + col]) << (8 * (i % 4));
        }
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            input[lane][6 + i / 2] |= static_cast<uint32_t>(float_to_half_exact(c[row * 8 + col])) << (16 * (i % 2));
        }
    }
    uint32_t output[32][2] = {};
    if (!driver.run("d4r_mma_m16n8k32_e4m3", input, sizeof(input), output, sizeof(output), 1, 32, -1))
        return false;
    int worst = 0, exact = 0;
    for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            double sum = c[row * 8 + col];
            for (int k = 0; k < 32; ++k)
                sum += static_cast<double>(e4m3_to_float(aCodes[row * 32 + k])) * e4m3_to_float(bCodes[k * 8 + col]);
            const _Float16 want = static_cast<_Float16>(sum);
            uint16_t bits;
            std::memcpy(&bits, &want, 2);
            const uint16_t got = static_cast<uint16_t>(output[lane][i / 2] >> (16 * (i % 2)));
            const int distance = half_ulp_distance(got, bits);
            if (distance > 2 && worst <= 2)
                std::printf("  %s D[%d][%d]: got 0x%04x expected 0x%04x (%g)\n", label, row, col, got, bits, sum);
            worst = std::max(worst, distance);
            exact += distance == 0;
        }
    const bool ok = worst <= 2;
    std::printf("%s: %s (worst %d ulp, %d of 128 exact)\n", label, ok ? "PASS" : "FAIL", worst, exact);
    return ok;
}

// Random full-range e4m3 codes through d4r_mma_chain_e4m3 (8 steps, pairs that
// share A), against a reference that rounds the accumulator to f16 per MMA.
static bool test_mma_chain_e4m3(Driver& driver, int trial)
{
    std::mt19937 random(4321 + trial);
    auto pick_code = [&]() {
        uint8_t code;
        do
            code = static_cast<uint8_t>(random());
        while ((code & 0x7f) == 0x7f || (code & 0x78) >= 0x50); // no NaN, |x| < 32
        return code;
    };
    constexpr int kSteps = 8;
    std::vector<uint8_t> a(kSteps * 16 * 32), b0(kSteps * 32 * 8), b1(kSteps * 32 * 8);
    for (uint8_t& code : a) code = pick_code();
    for (uint8_t& code : b0) code = pick_code();
    for (uint8_t& code : b1) code = pick_code();
    std::uniform_real_distribution<float> uniform(-4.0f, 4.0f);
    std::vector<float> c0(16 * 8), c1(16 * 8);
    for (float& value : c0) value = static_cast<float>(static_cast<_Float16>(uniform(random)));
    for (float& value : c1) value = static_cast<float>(static_cast<_Float16>(uniform(random)));
    static uint32_t input[32][68];
    std::memset(input, 0, sizeof(input));
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int step = 0; step < kSteps; ++step)
        {
            uint32_t* words = input[lane] + step * 8;
            for (int i = 0; i < 16; ++i)
            {
                int row, col;
                layout_a_8bit_k32(lane, i, row, col);
                words[i / 4] |= static_cast<uint32_t>(a[step * 512 + row * 32 + col]) << (8 * (i % 4));
            }
            for (int i = 0; i < 8; ++i)
            {
                int row, col;
                layout_b_8bit_k32(lane, i, row, col);
                words[4 + i / 4] |= static_cast<uint32_t>(b0[step * 256 + row * 8 + col]) << (8 * (i % 4));
                words[6 + i / 4] |= static_cast<uint32_t>(b1[step * 256 + row * 8 + col]) << (8 * (i % 4));
            }
        }
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            input[lane][64 + i / 2] |= static_cast<uint32_t>(float_to_half_exact(c0[row * 8 + col])) << (16 * (i % 2));
            input[lane][66 + i / 2] |= static_cast<uint32_t>(float_to_half_exact(c1[row * 8 + col])) << (16 * (i % 2));
        }
    }
    uint32_t output[32][4] = {};
    if (!driver.run("d4r_mma_chain_e4m3", input, sizeof(input), output, sizeof(output), 1, 32, -1))
        return false;
    std::vector<double> acc0(c0.begin(), c0.end()), acc1(c1.begin(), c1.end());
    for (int step = 0; step < kSteps; ++step)
        for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 8; ++col)
            {
                double sum0 = acc0[row * 8 + col], sum1 = acc1[row * 8 + col];
                for (int k = 0; k < 32; ++k)
                {
                    const double av = e4m3_to_float(a[step * 512 + row * 32 + k]);
                    sum0 += av * e4m3_to_float(b0[step * 256 + k * 8 + col]);
                    sum1 += av * e4m3_to_float(b1[step * 256 + k * 8 + col]);
                }
                acc0[row * 8 + col] = static_cast<double>(static_cast<_Float16>(sum0));
                acc1[row * 8 + col] = static_cast<double>(static_cast<_Float16>(sum1));
            }
    int worst = 0, exact = 0;
    for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            const uint16_t got0 = static_cast<uint16_t>(output[lane][i / 2] >> (16 * (i % 2)));
            const uint16_t got1 = static_cast<uint16_t>(output[lane][2 + i / 2] >> (16 * (i % 2)));
            const _Float16 want0 = static_cast<_Float16>(acc0[row * 8 + col]);
            const _Float16 want1 = static_cast<_Float16>(acc1[row * 8 + col]);
            uint16_t bits0, bits1;
            std::memcpy(&bits0, &want0, 2);
            std::memcpy(&bits1, &want1, 2);
            const int d0 = half_ulp_distance(got0, bits0), d1 = half_ulp_distance(got1, bits1);
            worst = std::max(worst, std::max(d0, d1));
            exact += (d0 == 0) + (d1 == 0);
        }
    const bool ok = worst <= 8;
    std::printf("mma.m16n8k32.e4m3 chained x%d, paired, trial %d: %s (worst %d ulp, %d of 256 exact)\n", kSteps,
                trial, ok ? "PASS" : "FAIL", worst, exact);
    return ok;
}

// Checks d4r_mma_pair_alias_e4m3 against a double reference (2 ulp).
static bool test_mma_pair_alias(Driver& driver, bool late)
{
    std::mt19937 random(77 + late);
    auto code = [&]() {
        uint8_t value;
        do
            value = static_cast<uint8_t>(random());
        while ((value & 0x7f) == 0x7f || (value & 0x78) >= 0x50);
        return value;
    };
    std::vector<uint8_t> a(16 * 32), b0(32 * 8), b1(32 * 8);
    for (uint8_t& value : a) value = code();
    for (uint8_t& value : b0) value = code();
    for (uint8_t& value : b1) value = code();
    std::uniform_real_distribution<float> uniform(-4.0f, 4.0f);
    std::vector<float> c0(16 * 8);
    for (float& value : c0) value = static_cast<float>(static_cast<_Float16>(uniform(random)));
    // The late value of %r20: one f16 pair per lane, so C1 = {v, v} there.
    const uint16_t lateHalf0 = 0x3c00, lateHalf1 = 0xc000; // 1.0, -2.0
    uint32_t input[32][12] = {};
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int i = 0; i < 16; ++i)
        {
            int row, col;
            layout_a_8bit_k32(lane, i, row, col);
            input[lane][i / 4] |= static_cast<uint32_t>(a[row * 32 + col]) << (8 * (i % 4));
        }
        for (int i = 0; i < 8; ++i)
        {
            int row, col;
            layout_b_8bit_k32(lane, i, row, col);
            input[lane][4 + i / 4] |= static_cast<uint32_t>(b0[row * 8 + col]) << (8 * (i % 4));
            input[lane][6 + i / 4] |= static_cast<uint32_t>(b1[row * 8 + col]) << (8 * (i % 4));
        }
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            input[lane][8 + i / 2] |= static_cast<uint32_t>(float_to_half_exact(c0[row * 8 + col])) << (16 * (i % 2));
        }
        input[lane][10] = lateHalf0 | (static_cast<uint32_t>(lateHalf1) << 16);
    }
    uint32_t output[32][4] = {};
    if (!driver.run(late ? "d4r_mma_pair_alias_late_e4m3" : "d4r_mma_pair_alias_e4m3", input, sizeof(input), output,
                    sizeof(output), 1, 32, late ? 1 : 0))
        return false;
    int worst = 0;
    for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 4; ++i)
        {
            int row, col;
            layout_c_f16(lane, i, row, col);
            double sum0 = c0[row * 8 + col];
            // C1 = {%r20, %r20}: element i reads half (i % 2) of %r20.
            double sum1 = late ? half_to_float(i % 2 == 0 ? lateHalf0 : lateHalf1) : 0.0;
            for (int k = 0; k < 32; ++k)
            {
                sum0 += static_cast<double>(e4m3_to_float(a[row * 32 + k])) * e4m3_to_float(b0[k * 8 + col]);
                sum1 += static_cast<double>(e4m3_to_float(a[row * 32 + k])) * e4m3_to_float(b1[k * 8 + col]);
            }
            const _Float16 want0 = static_cast<_Float16>(sum0), want1 = static_cast<_Float16>(sum1);
            uint16_t bits0, bits1;
            std::memcpy(&bits0, &want0, 2);
            std::memcpy(&bits1, &want1, 2);
            const uint16_t got0 = static_cast<uint16_t>(output[lane][i / 2] >> (16 * (i % 2)));
            const uint16_t got1 = static_cast<uint16_t>(output[lane][2 + i / 2] >> (16 * (i % 2)));
            worst = std::max(worst, std::max(half_ulp_distance(got0, bits0), half_ulp_distance(got1, bits1)));
        }
    const bool ok = worst <= 2;
    std::printf("mma.m16n8k32.e4m3 pair, aliased zero accumulator%s: %s (worst %d ulp)\n",
                late ? " rewritten between" : "", ok ? "PASS" : "FAIL", worst);
    return ok;
}

static bool test_movmatrix(Driver& driver)
{
    // Source fragment: lane holds row lane/4, columns 2*(lane%4) and +1.
    uint32_t input[32], output[32];
    for (int lane = 0; lane < 32; ++lane)
    {
        const int row = lane >> 2, col = (lane & 3) * 2;
        input[lane] = static_cast<uint32_t>(row * 8 + col) | (static_cast<uint32_t>(row * 8 + col + 1) << 16);
    }
    if (!driver.run("d4r_movmatrix_trans", input, sizeof(input), output, sizeof(output), 1, 32, -1))
        return false;
    size_t mismatches = 0;
    for (int lane = 0; lane < 32; ++lane)
    {
        const int row = lane >> 2, col = (lane & 3) * 2;
        // Destination holds the transpose in the same fragment layout.
        const uint32_t expected = static_cast<uint32_t>(col * 8 + row) |
                                  (static_cast<uint32_t>((col + 1) * 8 + row) << 16);
        if (output[lane] != expected && mismatches++ < 6)
            std::printf("  movmatrix lane %d: got 0x%08x expected 0x%08x\n", lane, output[lane], expected);
    }
    std::printf("movmatrix.sync.aligned.m8n8.trans.b16: %s (%zu of 32 lanes mismatched)\n",
                mismatches == 0 ? "PASS" : "FAIL", mismatches);
    return mismatches == 0;
}

static bool test_global_v2_u16(Driver& driver)
{
    uint32_t input[32], output[32][2] = {};
    for (uint32_t lane = 0; lane < 32; ++lane)
        input[lane] = (0x9137u + lane * 37u) | ((0xe5c3u - lane * 29u) << 16);
    if (!driver.run("d4r_global_v2_u16", input, sizeof(input), output, sizeof(output), 1, 32, -1))
        return false;
    size_t mismatches = 0;
    for (int lane = 0; lane < 32; ++lane)
        for (int copy = 0; copy < 2; ++copy)
            if (output[lane][copy] != input[lane] && mismatches++ < 6)
                std::printf("  global.v2.u16 lane %d copy %d: got 0x%08x expected 0x%08x\n",
                            lane, copy, output[lane][copy], input[lane]);
    std::printf("ld/st.global.v2.u16: %s (%zu of 64 words mismatched)\n",
                mismatches == 0 ? "PASS" : "FAIL", mismatches);
    return mismatches == 0;
}

static bool test_ex2_f16x2(Driver& driver)
{
    constexpr int count = 4096;
    std::vector<uint32_t> input(count), output(count);
    for (int index = 0; index < count; ++index)
    {
        const _Float16 lo = static_cast<_Float16>((index % 97 - 48) * 0.125f);
        const _Float16 hi = static_cast<_Float16>((index % 113 - 56) * 0.125f);
        uint16_t loBits, hiBits;
        std::memcpy(&loBits, &lo, sizeof(loBits));
        std::memcpy(&hiBits, &hi, sizeof(hiBits));
        input[index] = static_cast<uint32_t>(loBits) | (static_cast<uint32_t>(hiBits) << 16);
    }
    if (!driver.run("d4r_ex2_f16x2", input.data(), input.size() * 4, output.data(), output.size() * 4,
                    16, 256, count))
        return false;
    size_t mismatches = 0;
    double worstRelative = 0.0;
    for (int index = 0; index < count; ++index)
        for (int half = 0; half < 2; ++half)
        {
            const auto bits = static_cast<uint16_t>(input[index] >> (16 * half));
            const auto gotBits = static_cast<uint16_t>(output[index] >> (16 * half));
            const float expected = std::exp2(half_to_float(bits));
            const float got = half_to_float(gotBits);
            const double relative = std::fabs(got - expected) / std::max(1.0f, expected);
            worstRelative = std::max(worstRelative, relative);
            if ((!std::isfinite(got) || relative > 0.01) && mismatches++ < 6)
                std::printf("  ex2.f16x2 input %g: got %g expected %g\n", half_to_float(bits), got, expected);
        }
    std::printf("ex2.approx.f16x2: %s (%zu of %d lanes mismatched, worst relative %.6g)\n",
                mismatches == 0 ? "PASS" : "FAIL", mismatches, count * 2, worstRelative);
    return mismatches == 0;
}

static bool test_multidim_lane(Driver& driver)
{
    CUfunction function = nullptr;
    CUdeviceptr deviceOutput = 0;
    uint32_t output[256][4] = {};
    CUresult result = driver.moduleGetFunction(&function, driver.module, "d4r_multidim_lane");
    if (result == 0)
        result = driver.alloc(&deviceOutput, sizeof(output));
    if (result == 0)
    {
        void* arguments[] = {&deviceOutput};
        result = driver.launch(function, 1, 1, 1, 32, 1, 8, 0, nullptr, arguments, nullptr);
    }
    if (result == 0)
        result = driver.synchronize();
    if (result == 0)
        result = driver.copyDtoH(output, deviceOutput, sizeof(output));
    if (deviceOutput != 0)
        driver.free(deviceOutput);
    size_t mismatches = 0;
    for (int z = 0; z < 8; ++z)
        for (int x = 0; x < 32; ++x)
        {
            const uint32_t* got = output[z * 32 + x];
            if ((got[0] != static_cast<uint32_t>(x) || got[1] != 0 ||
                 got[2] != static_cast<uint32_t>(z) || got[3] != static_cast<uint32_t>(x)) &&
                mismatches++ < 6)
                std::printf("  lane block x=%d z=%d: got %u,%u,%u,%u\n", x, z, got[0], got[1], got[2], got[3]);
        }
    std::printf("block 32x1x8 tid/laneid: %s (CUDA result=%d, %zu of 256 lanes mismatched)\n",
                result == 0 && mismatches == 0 ? "PASS" : "FAIL", result, mismatches);
    return result == 0 && mismatches == 0;
}

// Rounds a double to the nearest f16 (ties to even), with overflow to
// infinity; exact for sums and products of two f16 values.
static uint16_t double_to_half_rne(double value)
{
    if (std::isnan(value))
        return 0x7e00u;
    const uint16_t sign = std::signbit(value) ? 0x8000u : 0u;
    double magnitude = std::fabs(value);
    if (magnitude >= 65520.0)
        return sign | 0x7c00u;
    if (magnitude < std::ldexp(1.0, -14))
    {
        // Subnormal: units of 2^-24.
        const double units = magnitude * 16777216.0;
        double rounded = std::nearbyint(units); // default mode is ties-to-even
        return sign | static_cast<uint16_t>(rounded);
    }
    int exponent;
    double fraction = std::frexp(magnitude, &exponent); // magnitude = fraction * 2^exponent, fraction in [0.5,1)
    double mantissa = std::nearbyint((fraction * 2.0 - 1.0) * 1024.0);
    int biased = exponent - 1 + 15;
    if (mantissa >= 1024.0)
    {
        mantissa = 0.0;
        ++biased;
    }
    if (biased >= 31)
        return sign | 0x7c00u;
    return sign | static_cast<uint16_t>((biased << 10) | static_cast<int>(mantissa));
}

static bool half_bits_equal(uint16_t got, uint16_t expected)
{
    const bool gotNan = (got & 0x7c00u) == 0x7c00u && (got & 0x3ffu) != 0;
    const bool expectedNan = (expected & 0x7c00u) == 0x7c00u && (expected & 0x3ffu) != 0;
    if (gotNan || expectedNan)
        return gotNan && expectedNan;
    // min/max of +0 and -0 may return either zero.
    if ((got & 0x7fffu) == 0 && (expected & 0x7fffu) == 0)
        return true;
    return got == expected;
}

static bool test_f16x2_fma_modes(Driver& driver)
{
    std::mt19937 random(7);
    constexpr int count = 8192;
    std::vector<uint32_t> input(count * 3);
    for (uint32_t& word : input)
    {
        auto random_half = [&]() -> uint16_t {
            for (;;)
            {
                const uint16_t bits = static_cast<uint16_t>(random());
                if ((bits & 0x7c00u) != 0x7c00u && std::fabs(half_to_float(bits)) < 4.0f)
                    return bits;
            }
        };
        word = random_half() | (static_cast<uint32_t>(random_half()) << 16);
    }
    // NaN and infinity operands.
    input[0] = 0x7e00u | (0xfc00u << 16);
    input[1] = 0x3c00u | (0x3c00u << 16);
    input[2] = 0x0000u | (0x0000u << 16);
    std::vector<uint32_t> output(count * 3);
    if (!driver.run("d4r_f16x2_fma_modes", input.data(), input.size() * 4, output.data(),
                    output.size() * 4, count / 256, 256, count))
        return false;
    const char* names[] = {"fma.rn.sat.f16x2", "fma.rn.relu.f16x2", "fma.rn.f16x2.relu"};
    size_t mismatches[3] = {};
    for (int thread = 0; thread < count; ++thread)
    {
        for (int half = 0; half < 2; ++half)
        {
            const uint16_t ah = static_cast<uint16_t>(input[thread * 3] >> (16 * half));
            const uint16_t bh = static_cast<uint16_t>(input[thread * 3 + 1] >> (16 * half));
            const uint16_t ch = static_cast<uint16_t>(input[thread * 3 + 2] >> (16 * half));
            const double result = static_cast<double>(half_to_float(ah)) * half_to_float(bh) + half_to_float(ch);
            const uint16_t rounded = double_to_half_rne(result);
            const bool nan = std::isnan(result);
            // .sat: clamp to [0, 1], NaN to +0. .relu: negative to +0, NaN stays NaN.
            const uint16_t sat = nan ? 0 : (rounded & 0x8000u) ? 0 : half_to_float(rounded) > 1.0f ? 0x3c00u : rounded;
            const uint16_t relu = nan ? 0x7fffu : (rounded & 0x8000u) ? 0 : rounded;
            const uint16_t expected[3] = {sat, relu, relu};
            for (int op = 0; op < 3; ++op)
            {
                const uint16_t got = static_cast<uint16_t>(output[thread * 3 + op] >> (16 * half));
                if (!half_bits_equal(got, expected[op]) && mismatches[op]++ < 3)
                    std::printf("  %s half %d: a=0x%04x b=0x%04x c=0x%04x got 0x%04x expected 0x%04x\n",
                                names[op], half, ah, bh, ch, got, expected[op]);
            }
        }
    }
    for (int op = 0; op < 3; ++op)
        std::printf("%s (%d lanes): %s (%zu mismatches)\n", names[op], count * 2,
                    mismatches[op] == 0 ? "PASS" : "FAIL", mismatches[op]);
    return mismatches[0] == 0 && mismatches[1] == 0 && mismatches[2] == 0;
}

static uint32_t reference_prmt(uint32_t a, uint32_t b, uint32_t selector)
{
    const uint64_t bytes = (static_cast<uint64_t>(b) << 32) | a;
    uint32_t result = 0;
    for (int index = 0; index < 4; ++index)
    {
        const uint32_t nibble = (selector >> (4 * index)) & 0xfu;
        uint8_t value = static_cast<uint8_t>(bytes >> (8 * (nibble & 7u)));
        if (nibble & 8u)
            value = (value & 0x80u) ? 0xffu : 0x00u;
        result |= static_cast<uint32_t>(value) << (8 * index);
    }
    return result;
}

static bool test_dltss_misc(Driver& driver)
{
    std::mt19937 random(31);
    uint32_t input[32][16] = {};
    for (int lane = 0; lane < 32; ++lane)
    {
        for (int word = 0; word < 12; ++word)
            input[lane][word] = static_cast<uint32_t>(random());
        // Make some f16 halves compare equal, including +0 against -0.
        if (lane % 3 == 0)
            input[lane][7] = input[lane][4];
        if (lane % 5 == 0)
            input[lane][7] = (input[lane][7] & 0xffff0000u) | (input[lane][4] & 0xffffu);
        if (lane == 1)
        {
            input[lane][4] = 0x80000000u;
            input[lane][7] = 0x00000000u;
        }
        input[lane][12] = static_cast<uint32_t>(lane % 17);
    }
    uint32_t output[32][20] = {};
    if (!driver.run("d4r_dltss_misc", input, sizeof(input), output, sizeof(output), 1, 32, -1))
        return false;
    // Shared tile: row r (16 bytes) comes from lane r's words 0..3. Matrix m row q
    // is addressed by lane 8m+q, which points at tile row (5 * (8m+q)) mod 32.
    auto tile_half = [&](int row, int column) -> uint32_t {
        const uint32_t word = input[row][column / 2];
        return (word >> (16 * (column % 2))) & 0xffffu;
    };
    const char* names[] = {"ldmatrix.x4", "ldmatrix.x2", "prmt 0x3424", "prmt 0x1404", "prmt 0x4431",
                           "prmt 0x7632", "prmt 0x5410", "prmt 0x7531", "set.equ.u32.f16x2",
                           "dp2a.lo.u32.u32", "cp.async.cg src-size"};
    size_t mismatches[11] = {};
    auto check = [&](int op, int lane, uint32_t got, uint32_t expected) {
        if (got != expected && mismatches[op]++ < 3)
            std::printf("  %s lane %d: got 0x%08x expected 0x%08x\n", names[op], lane, got, expected);
    };
    for (int lane = 0; lane < 32; ++lane)
    {
        const int group = lane / 4, pair = lane % 4;
        for (int matrix = 0; matrix < 4; ++matrix)
        {
            const int row = (5 * (8 * matrix + group)) % 32;
            const uint32_t expected = tile_half(row, 2 * pair) | (tile_half(row, 2 * pair + 1) << 16);
            check(0, lane, output[lane][matrix], expected);
            if (matrix < 2)
                check(1, lane, output[lane][4 + matrix], expected);
        }
        const uint32_t a = input[lane][4], b = input[lane][5], c = input[lane][6], e = input[lane][7];
        check(2, lane, output[lane][6], reference_prmt(a, 0, 0x3424));
        check(3, lane, output[lane][7], reference_prmt(a, 0, 0x1404));
        check(4, lane, output[lane][8], reference_prmt(a, 0, 0x4431));
        check(5, lane, output[lane][9], reference_prmt(a, b, 0x7632));
        check(6, lane, output[lane][10], reference_prmt(a, b, 0x5410));
        check(7, lane, output[lane][11], reference_prmt(a, b, 0x7531));
        uint32_t equal = 0;
        for (int half = 0; half < 2; ++half)
        {
            const float x = half_to_float(static_cast<uint16_t>(a >> (16 * half)));
            const float y = half_to_float(static_cast<uint16_t>(e >> (16 * half)));
            // equ is the unordered comparison: NaN operands compare equal.
            if (x == y || std::isnan(x) || std::isnan(y))
                equal |= 0xffffu << (16 * half);
        }
        check(8, lane, output[lane][12], equal);
        const uint32_t dot = c + (a & 0xffffu) * (b & 0xffu) + (a >> 16) * ((b >> 8) & 0xffu);
        check(9, lane, output[lane][13], dot);
        const uint32_t size = input[lane][12];
        for (int word = 0; word < 4; ++word)
        {
            uint32_t expected = 0;
            for (int byte = 0; byte < 4; ++byte)
                if (static_cast<uint32_t>(word * 4 + byte) < size)
                    expected |= input[lane][8 + word] & (0xffu << (8 * byte));
            check(10, lane, output[lane][14 + word], expected);
        }
    }
    bool passed = true;
    for (int op = 0; op < 11; ++op)
    {
        std::printf("%s: %s (%zu mismatches)\n", names[op], mismatches[op] == 0 ? "PASS" : "FAIL",
                    mismatches[op]);
        passed &= mismatches[op] == 0;
    }
    return passed;
}

static float round_toward_zero_sum(float a, float b)
{
    // Exact sum in double, then truncate toward zero to float.
    const double exact = static_cast<double>(a) + static_cast<double>(b);
    float nearest = static_cast<float>(exact);
    if (std::fabs(static_cast<double>(nearest)) > std::fabs(exact))
        nearest = std::nextafter(nearest, 0.0f);
    return nearest;
}

static bool test_round_ops(Driver& driver)
{
    std::mt19937 random(5);
    std::uniform_real_distribution<float> coordinate(-40.0f, 40.0f);
    constexpr int count = 4096;
    std::vector<float> input(count * 2);
    for (int index = 0; index < count; ++index)
    {
        // Mix exact quarter/half values (row mappings) with arbitrary ones.
        const float quarter = static_cast<float>(static_cast<int>(random() % 321) - 160) * 0.25f;
        input[index * 2] = index % 2 ? quarter : coordinate(random);
        input[index * 2 + 1] = index % 3 ? 1e-8f * static_cast<float>(random() % 7) : coordinate(random);
    }
    std::vector<uint32_t> output(count * 9);
    if (!driver.run("d4r_round_ops", input.data(), input.size() * 4, output.data(), output.size() * 4,
                    count / 256, 256, count))
        return false;
    const char* names[] = {"add.rz.ftz.f32", "round half away (add.rz+cvt.rzi.f32)", "cvt.rzi.ftz.s32.f32",
                           "shfl.sync.idx lane 0", "cvt.rzi.f32.f32", "add.ftz.f32", "shfl.sync pred",
                           "bfi.b32 pos 3 len 29", "bfi.b32 pos 4 len 8"};
    size_t mismatches[9] = {};
    auto bits = [](float value) { uint32_t word; std::memcpy(&word, &value, 4); return word; };
    for (int index = 0; index < count; ++index)
    {
        const float a = input[index * 2], b = input[index * 2 + 1];
        const float away = std::trunc(round_toward_zero_sum(a, std::copysign(0.5f, a)));
        const uint32_t expected[9] = {bits(round_toward_zero_sum(a, b)), bits(away),
                                      static_cast<uint32_t>(static_cast<int32_t>(away)),
                                      static_cast<uint32_t>(index & ~31), bits(std::trunc(a)),
                                      bits(a + b), 1u,
                                      (static_cast<uint32_t>(index) & 7u) | ((static_cast<uint32_t>(index) & 8u) << 3),
                                      ((static_cast<uint32_t>(index) & 7u) & ~0xff0u) | ((static_cast<uint32_t>(index) & 0xffu) << 4)};
        const uint32_t* row = &output[index * 9];
        const uint32_t got[9] = {row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]};
        for (int op = 0; op < 9; ++op)
        {
            // Signed zeros compare equal.
            const bool zeroes = op != 2 && op != 3 && op < 6 && (got[op] & 0x7fffffffu) == 0 &&
                                (expected[op] & 0x7fffffffu) == 0;
            if (got[op] != expected[op] && !zeroes && mismatches[op]++ < 3)
                std::printf("  %s: a=%.9g b=%.9g got 0x%08x expected 0x%08x\n", names[op], a, b, got[op],
                            expected[op]);
        }
    }
    bool passed = true;
    for (int op = 0; op < 9; ++op)
    {
        std::printf("%s: %s (%zu mismatches)\n", names[op], mismatches[op] == 0 ? "PASS" : "FAIL", mismatches[op]);
        passed &= mismatches[op] == 0;
    }
    return passed;
}

static bool test_f16x2_ops(Driver& driver)
{
    std::mt19937 random(99);
    constexpr int count = 8192;
    std::vector<uint32_t> input(count * 3);
    auto random_half = [&]() -> uint16_t {
        // Finite values across the normal and subnormal range, magnitudes below 64.
        for (;;)
        {
            const uint16_t bits = static_cast<uint16_t>(random());
            if ((bits & 0x7c00u) != 0x7c00u && std::fabs(half_to_float(bits)) < 64.0f)
                return bits;
        }
    };
    for (uint32_t& word : input)
        word = random_half() | (static_cast<uint32_t>(random_half()) << 16);
    std::vector<uint32_t> output(count * 13);
    if (!driver.run("d4r_f16x2_ops", input.data(), input.size() * 4, output.data(), output.size() * 4,
                    count / 256, 256, count))
        return false;
    const char* names[] = {"abs.f16x2", "neg.f16x2", "min.f16x2", "max.f16x2", "add.f16x2",
                           "sub.f16x2", "mul.f16x2", "fma.rn.f16x2", "add.f16/mul.f16",
                           "nested .reg cvt 0.5", "nested .reg cvt 2.0", "cvt.rn.f16.f64",
                           "h2rsqrt (nested .reg)"};
    size_t mismatches[13] = {};
    for (int thread = 0; thread < count; ++thread)
    {
        const uint32_t a = input[thread * 3], b = input[thread * 3 + 1], c = input[thread * 3 + 2];
        for (int half = 0; half < 2; ++half)
        {
            const uint16_t ah = static_cast<uint16_t>(a >> (16 * half));
            const uint16_t bh = static_cast<uint16_t>(b >> (16 * half));
            const uint16_t ch = static_cast<uint16_t>(c >> (16 * half));
            const double av = half_to_float(ah), bv = half_to_float(bh), cv = half_to_float(ch);
            const uint16_t expected[13] = {
                static_cast<uint16_t>(ah & 0x7fffu),
                static_cast<uint16_t>(ah ^ 0x8000u),
                double_to_half_rne(av < bv ? av : bv),
                double_to_half_rne(av > bv ? av : bv),
                double_to_half_rne(av + bv),
                double_to_half_rne(av - bv),
                double_to_half_rne(av * bv),
                double_to_half_rne(av * bv + cv),
                half == 0 ? double_to_half_rne(av + bv) : double_to_half_rne(av * bv),
                0x3800u,
                0x4000u,
                half == 0 ? double_to_half_rne(-0.55615234375) : 0x3c00u,
                double_to_half_rne(1.0 / std::sqrt(std::fabs(av))),
            };
            for (int op = 0; op < 13; ++op)
            {
                const uint16_t got = static_cast<uint16_t>(output[thread * 13 + op] >> (16 * half));
                bool ok = half_bits_equal(got, expected[op]);
                if (op == 12 && !ok)
                {
                    // rsqrt.approx: allow one f16 ulp; zero/subnormal inputs flush (.ftz) to +inf.
                    ok = std::abs(static_cast<int>(got) - static_cast<int>(expected[op])) <= 1 ||
                         ((ah & 0x7c00u) == 0 && got == 0x7c00u);
                }
                if (!ok && mismatches[op]++ < 3)
                    std::printf("  %s half %d: a=0x%04x b=0x%04x c=0x%04x got 0x%04x expected 0x%04x\n",
                                names[op], half, ah, bh, ch, got, expected[op]);
            }
        }
    }
    bool passed = true;
    for (int op = 0; op < 13; ++op)
    {
        std::printf("%s (%d lanes): %s (%zu mismatches)\n", names[op], count * 2,
                    mismatches[op] == 0 ? "PASS" : "FAIL", mismatches[op]);
        passed &= mismatches[op] == 0;
    }
    return passed;
}

int main()
{
    void* library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
    {
        std::fprintf(stderr, "dlopen(libcuda.so) failed: %s\n", dlerror());
        return 1;
    }
    Driver driver{};
    if (!driver.load(library))
    {
        std::fprintf(stderr, "ZLUDA libcuda.so is missing a required export\n");
        return 1;
    }
    CUdevice device = 0;
    CUcontext context = nullptr;
    if (driver.init(0) != 0 || driver.deviceGet(&device, 0) != 0 ||
        driver.contextCreate(&context, 0, device) != 0)
    {
        std::fprintf(stderr, "CUDA context setup failed\n");
        return 1;
    }
    const CUresult loadResult = driver.moduleLoad(&driver.module, ptx);
    if (loadResult != 0)
    {
        std::fprintf(stderr, "cuModuleLoadData failed: %d\n", loadResult);
        return 1;
    }

    std::mt19937 random(1234);
    const float smallValues[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    auto pick = [&]() { return smallValues[random() % 7]; };

    int failures = 0;
    failures += !test_cvt_from_e4m3(driver);
    failures += !test_cvt_to_e4m3(driver);
    failures += !test_cvt_to_e4m3(driver, "d4r_cvt_e4m3x2_relu_from_f16x2", "cvt.rn.satfinite.relu.e4m3x2.f16x2", true);
    failures += !test_cvt_to_e4m3(driver, "d4r_cvt_e4m3x2_relu_from_f32", "cvt.rn.satfinite.relu.e4m3x2.f32", true);
    failures += !test_movmatrix(driver);
    failures += !test_global_v2_u16(driver);
    failures += !test_ex2_f16x2(driver);
    failures += !test_multidim_lane(driver);
    failures += !test_f16x2_ops(driver);
    failures += !test_f16x2_fma_modes(driver);
    failures += !test_dltss_misc(driver);
    failures += !test_round_ops(driver);

    for (int trial = 0; trial < 3; ++trial)
    {
        std::vector<float> a(16 * 8), b(8 * 8), c(16 * 8);
        for (float& value : a) value = pick();
        for (float& value : b) value = pick();
        for (float& value : c) value = pick();
        char label[96];
        std::snprintf(label, sizeof(label), "mma.m16n8k8.f16.f16 trial %d", trial);
        failures += !test_mma(driver, "d4r_mma_m16n8k8_f16", label, 8, 16, layout_a_f16_k8,
                              layout_b_f16_k8, a, b, c, {}, {});
    }

    for (int trial = 0; trial < 3; ++trial)
    {
        std::vector<float> a(16 * 16), b(16 * 8), c(16 * 8);
        for (float& value : a) value = pick();
        for (float& value : b) value = pick();
        for (float& value : c) value = pick();
        char label[96];
        std::snprintf(label, sizeof(label), "mma.m16n8k16.f16.f16 trial %d", trial);
        failures += !test_mma(driver, "d4r_mma_m16n8k16_f16", label, 16, 16, layout_a_f16_k16,
                              layout_b_f16_k16, a, b, c, {}, {});
    }
    for (int trial = 0; trial < 4; ++trial)
        failures += !test_mma_chain(driver, trial);
    for (int trial = 0; trial < 3; ++trial)
    {
        std::vector<float> a(16 * 32), b(32 * 8), c(16 * 8);
        std::vector<uint8_t> aCodes(a.size()), bCodes(b.size());
        for (size_t index = 0; index < a.size(); ++index)
        {
            a[index] = pick();
            aCodes[index] = float_to_e4m3_satfinite(a[index]);
        }
        for (size_t index = 0; index < b.size(); ++index)
        {
            b[index] = pick();
            bCodes[index] = float_to_e4m3_satfinite(b[index]);
        }
        for (float& value : c) value = pick();
        char label[96];
        std::snprintf(label, sizeof(label), "mma.m16n8k32.f16.e4m3.e4m3 trial %d", trial);
        failures += !test_mma(driver, "d4r_mma_m16n8k32_e4m3", label, 32, 8, layout_a_8bit_k32,
                              layout_b_8bit_k32, a, b, c, aCodes, bCodes);
    }
    for (int trial = 0; trial < 3; ++trial)
    {
        // Any finite code with |x| < 32 (so sums stay well inside f16).
        std::mt19937 codes(99 + trial);
        auto code = [&]() {
            uint8_t value;
            do
                value = static_cast<uint8_t>(codes());
            while ((value & 0x7f) == 0x7f || (value & 0x78) >= 0x50);
            return value;
        };
        std::vector<uint8_t> aCodes(16 * 32), bCodes(32 * 8);
        for (uint8_t& value : aCodes) value = code();
        for (uint8_t& value : bCodes) value = code();
        std::vector<float> c(16 * 8);
        for (float& value : c) value = pick();
        char label[96];
        std::snprintf(label, sizeof(label), "mma.m16n8k32.e4m3 full-range codes trial %d", trial);
        failures += !test_mma_e4m3_codes(driver, label, aCodes, bCodes, c);
    }
    {
        // Subnormal A (m * 2^-9) times large B: a flushed denormal reads as 0.
        std::mt19937 codes(7);
        std::vector<uint8_t> aCodes(16 * 32), bCodes(32 * 8);
        for (uint8_t& value : aCodes) value = static_cast<uint8_t>((codes() & 0x80) | (1 + codes() % 7));
        for (uint8_t& value : bCodes) value = static_cast<uint8_t>((codes() & 0x80) | 0x70 | (codes() % 8)); // 256..480
        for (uint8_t& value : bCodes) if ((value & 0x7f) == 0x7f) value &= 0xfe;
        std::vector<float> c(16 * 8, 0.0f);
        failures += !test_mma_e4m3_codes(driver, "mma.m16n8k32.e4m3 subnormal A x large B", aCodes, bCodes, c);
    }
    for (int trial = 0; trial < 4; ++trial)
        failures += !test_mma_chain_e4m3(driver, trial);
    failures += !test_mma_pair_alias(driver, false);
    failures += !test_mma_pair_alias(driver, true);
    std::printf("FP8/matrix instruction tests: %s (%d failing)\n", failures == 0 ? "PASS" : "FAIL",
                failures);
    return failures == 0 ? 0 : 1;
}
