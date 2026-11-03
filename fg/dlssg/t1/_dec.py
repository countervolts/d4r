# T1 launch groups for DL1Net's decoder steps: k_upscale -> k_element_wise -> 1x1 k_conv_fp16_nhwc.
#
# Semantics (PTX of nvngx_dlssg 310.9.16, every variant identical up to G = 16-byte channel groups per pixel):
#   k_upscale(in, out, Hi, Wi, Ho, Wo, Hp, Wp): bilinear 2x resize with align_corners (scale = (Hi-1)/(Ho-1) by
#     div.approx), packed-f16 weights; U[(row * Wp + col) * G + c] for row < Hp, col < Wp, c < G; zero where
#     col >= Wo or row >= Ho.
#   k_element_wise(a, b, out, W, H, C, Wa, Ha, Wb, Hb): out[(y * W + x) * (C / 8) + c] = a[((y % Ha) * Wa + x % Wa) * G + c]
#     + b[((y % Hb) * Wb + x % Wb) * G + c] (add.f16x2 per 32-bit word) for x < W, y < H, c < C / 8.
#   The conv reads only that sum (E), and U and E are not read again before they are overwritten (DLSS-G's buffer
#   reuse in this DLL), so the fused kernel computes E inside the conv's input staging and writes neither U nor E.
# The element arithmetic below is the translator's output for these PTX lines, statement for statement, so the values
# are bit-identical to the three separate launches. If the sequence breaks off, the buffers do not chain as expected
# or the shapes do not line up, the fused kernel instead runs the absorbed members one after the other (grid-stride
# element loops for the two element-wise kernels, the conv's own T1 body reading E from memory) with grid barriers in
# between, like the other launch groups.
import _conv, _fuse, _dl1

UP_PARAMS = 40   # k_upscale: 0 in, 8 out, 16 Hi, 20 Wi, 24 Ho, 28 Wo, 32 Hp, 36 Wp
EW_PARAMS = 52   # k_element_wise: 0 a, 8 b, 16 out, 24 W, 28 H, 32 C, 36 Wa, 40 Ha, 44 Wb, 48 Hb

HELPERS = r'''
uint64_t nvx_reg;     // group region
bool nvx_fused;       // the three members chain as expected: E is computed in the conv's staging
uint nvx_nwg;         // workgroups taking part (all when fused, NRES member by member)
// grid barrier over workgroups 0 .. NRES-1 (the counter at region + 128, poison flag at + 64 as in _fuse's barrier:
// a workgroup that waits `spin` polls poisons the region, after which no barrier waits and nvxrt disables the group)
void nvx_dec_sync(uint spin)
{
    controlBarrier(gl_ScopeWorkgroup, gl_ScopeQueueFamily,
                   gl_StorageSemanticsBuffer | gl_StorageSemanticsImage | gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
    if (gl_LocalInvocationIndex == 0u) {
        uint old = atomicAdd(NvxU32A(nvx_reg + 128ul).v, 1u, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);
        uint target = (old / %(NRES)du + 1u) * %(NRES)du;
        for (uint i = 0u; i < spin; i++) {
            if (atomicLoad(NvxU32A(nvx_reg + 128ul).v, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsAcquire) >= target)
                break;
            if ((i & 31u) == 0u && atomicLoad(NvxU32A(nvx_reg + 64ul).v, gl_ScopeQueueFamily, 0, gl_SemanticsRelaxed) != 0u)
                break;
            if (i + 1u == spin)
                atomicStore(NvxU32A(nvx_reg + 64ul).v, 1u, gl_ScopeQueueFamily, 0, gl_SemanticsRelaxed);
        }
    }
    controlBarrier(gl_ScopeWorkgroup, gl_ScopeQueueFamily,
                   gl_StorageSemanticsBuffer | gl_StorageSemanticsImage | gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
}
uint nvx_mp32(uint k, uint off) { return NvxPW(nvx_reg + 256ul + uint64_t(k) * 256ul + uint64_t(off)).v; }
uint64_t nvx_mp64(uint k, uint off) { return packUint2x32(uvec2(nvx_mp32(k, off), nvx_mp32(k, off + 4u))); }

// k_upscale's per-pixel part at (r_r2 = col, r_r3 = row), col < Wo, row < Ho (member 0's parameters): the four
// source pixel indices and the four packed weights
struct NvxUC { uint i00, i01, i10, i11, w26, w27, w28, w29; };
NvxUC nvx_up_coords(uint r_r2, uint r_r3)
{
    uint r_r4 = nvx_mp32(0u, 16u), r_r5 = nvx_mp32(0u, 20u), r_r6 = nvx_mp32(0u, 24u), r_r7 = nvx_mp32(0u, 28u);
    precise float r_f25, r_f26;
    if ((r_r6) < (2u)) {
        r_f25 = uintBitsToFloat(0x00000000U);
    } else {
        uint r_r22 = r_r4 + 4294967295u;
        precise float r_f6 = float(r_r22);
        uint r_r23 = r_r6 + 4294967295u;
        precise float r_f7 = float(r_r23);
        r_f25 = nvx_div_approx(r_f6, r_f7);
    }
    if ((r_r7) > (1u)) {
        uint r_r24 = r_r5 + 4294967295u;
        precise float r_f9 = float(r_r24);
        uint r_r25 = r_r7 + 4294967295u;
        precise float r_f10 = float(r_r25);
        r_f26 = nvx_div_approx(r_f9, r_f10);
    } else {
        r_f26 = uintBitsToFloat(0x00000000U);
    }
    precise float r_f15 = float(r_r2);
    precise float r_f16 = r_f26 * r_f15;
    precise float r_f17 = float(r_r3);
    precise float r_f18 = r_f25 * r_f17;
    uint r_r138 = uint(trunc(r_f16));
    uint r_r139 = uint(trunc(r_f18));
    precise float r_f19 = r_f16 + uintBitsToFloat(0x3F800000U);
    uint r_r140 = uint(trunc(r_f19));
    bool r_p11 = (r_r140) >= (r_r5);
    uint r_r141 = r_r5 + 4294967295u;
    uint r_r142 = r_p11 ? r_r141 : r_r138;
    uint r_r143 = r_p11 ? r_r141 : r_r140;
    precise float r_f20 = r_f18 + uintBitsToFloat(0x3F800000U);
    uint r_r144 = uint(trunc(r_f20));
    bool r_p12 = (r_r144) >= (r_r4);
    uint r_r145 = r_r4 + 4294967295u;
    uint r_r146 = r_p12 ? r_r145 : r_r139;
    uint r_r147 = r_p12 ? r_r145 : r_r144;
    bool r_p13 = (r_f16) < (uintBitsToFloat(0x00000000U));
    uint r_r148 = r_p13 ? 0u : r_r142;
    uint r_r149 = r_p13 ? 0u : r_r143;
    bool r_p14 = (r_f18) < (uintBitsToFloat(0x00000000U));
    uint r_r150 = r_p14 ? 0u : r_r146;
    uint r_r151 = r_p14 ? 0u : r_r147;
    precise float r_f21 = float(r_r150);
    precise float r_f11 = r_f18 - r_f21;
    uint16_t r_rs1 = nvx_f2h(r_f11);
    precise float r_f22 = float(r_r148);
    precise float r_f12 = r_f16 - r_f22;
    uint16_t r_rs4 = nvx_f2h(r_f12);
    precise float r_f23 = r_f21 - r_f18;
    precise float r_f13 = r_f23 + uintBitsToFloat(0x3F800000U);
    uint16_t r_rs7 = nvx_f2h(r_f13);
    precise float r_f24 = r_f22 - r_f16;
    precise float r_f14 = r_f24 + uintBitsToFloat(0x3F800000U);
    uint16_t r_rs10 = nvx_f2h(r_f14);
    NvxUC u;
    u.w26 = uint(r_rs1) | (uint(r_rs1) << 16);
    u.w27 = uint(r_rs4) | (uint(r_rs4) << 16);
    u.w28 = uint(r_rs7) | (uint(r_rs7) << 16);
    u.w29 = uint(r_rs10) | (uint(r_rs10) << 16);
    uint r_r152 = r_r150 * r_r5, r_r167 = r_r151 * r_r5;
    u.i00 = r_r152 + r_r148;
    u.i01 = r_r152 + r_r149;
    u.i10 = r_r167 + r_r148;
    u.i11 = r_r167 + r_r149;
    return u;
}
// k_upscale's value for channel group r_r1 from its per-pixel part
uvec4 nvx_up_apply(NvxUC u, uint r_r1)
{
    uint64_t r_rd6 = nvx_mp64(0u, 0u);
    uint r_r29 = u.w29, r_r27 = u.w27, r_r28 = u.w28, r_r26 = u.w26;
    uint r_r155 = nvx_shl32(u.i00, %(SH)du) + r_r1;
    uvec4 q00 = NvxU32x4(r_rd6 + uint64_t((int64_t(int(r_r155)) * int64_t(16)))).v;
    uint r_r162 = nvx_shl32(u.i01, %(SH)du) + r_r1;
    uvec4 q01 = NvxU32x4(r_rd6 + uint64_t((int64_t(int(r_r162)) * int64_t(16)))).v;
    uint r_r170 = nvx_shl32(u.i10, %(SH)du) + r_r1;
    uvec4 q10 = NvxU32x4(r_rd6 + uint64_t((int64_t(int(r_r170)) * int64_t(16)))).v;
    uint r_r177 = nvx_shl32(u.i11, %(SH)du) + r_r1;
    uvec4 q11 = NvxU32x4(r_rd6 + uint64_t((int64_t(int(r_r177)) * int64_t(16)))).v;
    uvec4 o;
    [[unroll]] for (uint j = 0u; j < 4u; j++) {
        uint r_r30 = packFloat2x16(unpackFloat2x16(r_r29) * unpackFloat2x16(q00[j]));
        uint r_r42 = packFloat2x16(unpackFloat2x16(r_r27) * unpackFloat2x16(q01[j]));
        uint r_r54 = packFloat2x16(unpackFloat2x16(r_r30) + unpackFloat2x16(r_r42));
        uint r_r66 = packFloat2x16(unpackFloat2x16(r_r28) * unpackFloat2x16(r_r54));
        uint r_r78 = packFloat2x16(unpackFloat2x16(r_r29) * unpackFloat2x16(q10[j]));
        uint r_r90 = packFloat2x16(unpackFloat2x16(r_r27) * unpackFloat2x16(q11[j]));
        uint r_r102 = packFloat2x16(unpackFloat2x16(r_r78) + unpackFloat2x16(r_r90));
        uint r_r114 = packFloat2x16(unpackFloat2x16(r_r26) * unpackFloat2x16(r_r102));
        o[j] = packFloat2x16(unpackFloat2x16(r_r66) + unpackFloat2x16(r_r114));
    }
    return o;
}
uvec4 nvx_up_val(uint c, uint col, uint row) { return nvx_up_apply(nvx_up_coords(col, row), c); }
// what k_upscale stores at (c, col, row), col < Wp, row < Hp
uvec4 nvx_up_store(uint c, uint col, uint row)
{
    if (col >= nvx_mp32(0u, 28u) || row >= nvx_mp32(0u, 24u))
        return uvec4(0u);
    return nvx_up_val(c, col, row);
}
// k_element_wise's pixel terms at (r_r1 = x, r_r2 = y): U's (col, row) and the skip input's pixel offset
uvec3 nvx_ew_pix(uint r_r1, uint r_r2)
{
    uint r_r23 = nvx_mp32(1u, 36u), r_r24 = nvx_mp32(1u, 40u), r_r25 = nvx_mp32(1u, 44u), r_r26 = nvx_mp32(1u, 48u);
    uint r_r37 = uint((nvx_rem32(int(r_r2), int(r_r24))));
    uint r_r8 = uint((nvx_rem32(int(r_r1), int(r_r23))));
    uint r_r62 = uint((nvx_rem32(int(r_r2), int(r_r26))));
    uint r_r63 = uint((nvx_rem32(int(r_r1), int(r_r25))));
    uint r_r64 = nvx_shl32(r_r63, %(SH)du);
    uint r_r13 = nvx_shl32(r_r25, %(SH)du);
    uint r_r65 = r_r13 * r_r62 + r_r64;
    return uvec3(r_r8, r_r37, r_r65);
}
uvec4 nvx_ew_add(uvec4 a, uint r_r144, uint r_r65)
{
    uint r_r66 = r_r144 + r_r65;
    uvec4 b = NvxU32x4a4(nvx_mp64(1u, 8u) + uint64_t((int64_t(int(r_r66)) * int64_t(16)))).v;
    uvec4 o;
    [[unroll]] for (uint j = 0u; j < 4u; j++)
        o[j] = packFloat2x16(unpackFloat2x16(a[j]) + unpackFloat2x16(b[j]));
    return o;
}
// k_element_wise's value at (x, y, c), reading U from memory (fallback)
uvec4 nvx_ew_val(uint x, uint y, uint c)
{
    uvec3 t = nvx_ew_pix(x, y);
    uint r_r45 = c + nvx_shl32(t.y * nvx_mp32(1u, 36u) + t.x, %(SH)du);
    uvec4 a = NvxU32x4(nvx_mp64(1u, 0u) + uint64_t((int64_t(int(r_r45)) * int64_t(16)))).v;
    return nvx_ew_add(a, c, t.z);
}
// fused: per tile pixel p the U part and the skip offset, kept in LDS (smem[DT + 10 p ..]): weights w26..w29, source
// indices i00..i11, skip offset, U-is-zero flag (col >= Wo or row >= Ho: k_upscale stored 0 there)
void nvx_dec_setup(uint p, uint x, uint y)
{
    uvec3 t = nvx_ew_pix(x, y);
    uint b = %(DT)du + p * 10u;
    bool z = t.x >= nvx_mp32(0u, 28u) || t.y >= nvx_mp32(0u, 24u);
    NvxUC u = nvx_up_coords(z ? 0u : t.x, z ? 0u : t.y);
    smem[b] = u.i00; smem[b + 1u] = u.i01; smem[b + 2u] = u.i10; smem[b + 3u] = u.i11;
    smem[b + 4u] = u.w26; smem[b + 5u] = u.w27; smem[b + 6u] = u.w28; smem[b + 7u] = u.w29;
    smem[b + 8u] = t.z; smem[b + 9u] = z ? 1u : 0u;
}
// the conv's input staging (tile pixel p at clamped (gx, gy), 16-byte channel group cg): E computed when fused, else
// read like the plain conv
uvec4 nvx_dec_load(uint64_t ip, int gx, int gy, uint cg, int W, uint p)
{
    if (nvx_fused) {
        uint b = %(DT)du + p * 10u;
        NvxUC u;
        u.i00 = smem[b]; u.i01 = smem[b + 1u]; u.i10 = smem[b + 2u]; u.i11 = smem[b + 3u];
        u.w26 = smem[b + 4u]; u.w27 = smem[b + 5u]; u.w28 = smem[b + 6u]; u.w29 = smem[b + 7u];
        uvec4 a = smem[b + 9u] != 0u ? uvec4(0u) : nvx_up_apply(u, cg);
        return nvx_ew_add(a, cg, smem[b + 8u]);
    }
    return NvxU32x4a4(ip + uint64_t((uint(gy * W + gx) * %(CINPIX)du + cg * 8u) * 2u)).v;
}
// fallback: k_upscale over its whole output, grid-stride
void nvx_up_all()
{
    uint Hp = nvx_mp32(0u, 32u), Wp = nvx_mp32(0u, 36u);
    uint64_t op = nvx_mp64(0u, 8u);
    uint n = Hp * Wp * %(G)du, stride = nvx_nwg * gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x * gl_WorkGroupSize.y + gl_LocalInvocationIndex; i < n; i += stride) {
        uint c = i %% %(G)du, q = i / %(G)du, col = q %% Wp, row = q / Wp;
        uint r_r21 = nvx_shl32(row * Wp + col, %(SH)du) + c;
        NvxU32x4(op + uint64_t((int64_t(int(r_r21)) * int64_t(16)))).v = nvx_up_store(c, col, row);
    }
}
// fallback: k_element_wise over its whole output, grid-stride
void nvx_ew_all()
{
    uint W = nvx_mp32(1u, 24u), H = nvx_mp32(1u, 28u), C8 = uint(nvx_sar32(int(nvx_mp32(1u, 32u)), 3u));
    uint64_t op = nvx_mp64(1u, 16u);
    uint n = H * W * C8, stride = nvx_nwg * gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x * gl_WorkGroupSize.y + gl_LocalInvocationIndex; i < n; i += stride) {
        uint c = i %% C8, q = i / C8, x = q %% W, y = q / W;
        uint r_r44 = c + C8 * (y * W + x);
        NvxU32x4(op + uint64_t((int64_t(int(r_r44)) * int64_t(16)))).v = nvx_ew_val(x, y, c);
    }
}
'''

MAIN = r'''
void nvx_conv_all()
{
    nvx_pb = nvx_reg + 768ul;      // member 2's parameter block (region + 256 + 256 k)
    // the conv's tiles from its output size (16 x TH pixels, ZS channel slices): the same for every alternative of
    // member 2 (one layer compiled more than once by NVIDIA, with different launch grids)
    uvec3 vg = uvec3((nvx_mp32(2u, 32u) + 15u) / 16u, (nvx_mp32(2u, 36u) + %(TH1)du) / %(TH)du, %(ZS)du);
    uint tot = vg.x * vg.y * vg.z;
    for (uint v = gl_WorkGroupID.x; v < tot; v += nvx_nwg) {
        nvx_wg = uvec3(v %% vg.x, (v / vg.x) %% vg.y, v / (vg.x * vg.y));
        nvx_ng = vg;
        barrier();
        nvx_conv();
    }
}
// the conv's output bytes do not overlap the skip input (the conv may write in place over it: then only one workgroup
// may produce each output pixel, i.e. one z slice, so that a pixel's skip values are read before they are replaced)
bool nvx_inplace_ok()
{
    uint64_t b0 = nvx_mp64(1u, 8u), b1 = b0 + uint64_t(nvx_mp32(1u, 44u)) * uint64_t(nvx_mp32(1u, 48u)) * %(G)dul * 16ul;
    uint64_t o0 = nvx_mp64(2u, 24u), o1 = o0 + uint64_t(nvx_mp32(2u, 32u)) * uint64_t(nvx_mp32(2u, 36u)) * %(COUT2)dul * 4ul;
    return o1 <= b0 || b1 <= o0 || %(ZS)du == 1u;
}
void main()
{
    nvx_reg = packUint2x32(uvec2(pc.w[0], pc.w[1]));
    uint count = min(NvxPW(nvx_reg).v, 3u);
    nvx_fused = count == 3u
        && nvx_mp64(1u, 0u) == nvx_mp64(0u, 8u)                 // element_wise reads the upscale's output
        && nvx_mp64(2u, 8u) == nvx_mp64(1u, 16u)                // the conv reads element_wise's output
        && uint(nvx_sar32(int(nvx_mp32(1u, 32u)), 3u)) == %(G)du      // E's pixel stride = the conv's
        && nvx_mp32(2u, 16u) == nvx_mp32(1u, 24u) && nvx_mp32(2u, 20u) == nvx_mp32(1u, 28u)
        && nvx_mp32(1u, 36u) != 0u && nvx_mp32(1u, 40u) != 0u && nvx_mp32(1u, 44u) != 0u && nvx_mp32(1u, 48u) != 0u
        && nvx_mp32(1u, 36u) <= nvx_mp32(0u, 36u) && nvx_mp32(1u, 40u) <= nvx_mp32(0u, 32u)
        && int(nvx_mp32(1u, 36u)) > 0 && int(nvx_mp32(1u, 40u)) > 0 && int(nvx_mp32(1u, 44u)) > 0 && int(nvx_mp32(1u, 48u)) > 0
        && nvx_inplace_ok();
#ifdef NVX_DEC_NOFUSE
    nvx_fused = false;    // test switch: always the member-by-member path
#endif
    nvx_nwg = nvx_fused ? gl_NumWorkGroups.x : min(gl_NumWorkGroups.x, %(NRES)du);
    if (nvx_fused) {
        nvx_conv_all();
        return;
    }
    // member by member: only the first NRES workgroups (one per CU: always co-resident) take part, the others leave
    // at once; grid-stride loops and grid barriers over those NRES
    if (gl_WorkGroupID.x >= %(NRES)du)
        return;
    nvx_dec_sync(%(SPIN0)du);
    for (uint k = 0u; k < count; k++) {
        if (k != 0u)
            nvx_dec_sync(%(SPIN)du);
        if (k == 0u)
            nvx_up_all();
        else if (k == 1u)
            nvx_ew_all();
        else
            nvx_conv_all();
    }
}
'''


def decoder_group(G, CIN, TH, ZS, conv_build, nofuse=False):
    """the fused kernel for k_upscale/k_element_wise with G 16-byte channel groups per pixel, followed by the 1x1 conv
    (CIN input channels = its input pixel stride; tiles of 16 x TH pixels, ZS = NTALL / NT channel slices) whose T1
    build function (with hooks) is conv_build. Returns (glsl, block)."""
    SH = {4: 2, 8: 3, 16: 4, 32: 5}[G]
    assert CIN == 8 * G
    pre = ('\n    if (nvx_fused) {\n        for (uint p_ = lid; p_ < %du; p_ += nthr)\n'
           '            nvx_dec_setup(p_, uint(clamp(x0 + int(p_ %%%% 16u), 0, W - 1)), uint(clamp(y0 + int(p_ / 16u), 0, H - 1)));\n'
           '    }' % (16 * TH))
    glsl, meta = conv_build(None, hooks={'loadin': 'nvx_dec_load(ip, gx, gy, cl * 2u + hh, W, p)', 'pre': pre})
    words, body = _fuse.split_kernel(glsl, _conv.RT)
    DT = (words + 3) // 4 * 4                 # the fused loader's per-pixel table after the conv's own LDS
    words = (DT + 16 * TH * 10 + 3) // 4 * 4
    m = __import__('re').search(r'\* (\d+)u \+ zt \* 8u \+ pr\) \* 4u', glsl)
    assert m, 'conv output store not found'
    g = dict(G=G, SH=SH, CINPIX=G * 8, COUT2=int(m.group(1)), DT=DT, TH=TH, TH1=TH - 1, ZS=ZS,
             NRES=_conv.fit_persistent(_conv.TUNED_CUS, 1), SPIN=_fuse.SPIN, SPIN0=_fuse.SPIN0)
    src = (_conv.ptxc.HEADER +
           ('#define NVX_DEC_NOFUSE\n' if nofuse else '') +
           '#extension GL_EXT_shared_memory_block : require\n'
           '#extension GL_EXT_control_flow_attributes : require\n'
           'shared NvxSMU { uint smem[%d]; };\n'
           'shared NvxSMH { float16_t smh[%d]; };\n'
           'shared NvxSMV { uvec4 smv[%d]; };\n' % (words, words * 2, words // 4) + _fuse.group_rt(_conv.RT))
    src += HELPERS % g
    src += _fuse.member_fn(body, 'nvx_conv') + '\n'
    src += MAIN % g
    return src, _fuse.t1_block(meta)
