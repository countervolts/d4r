# T1: the 8 chained custom_block{0,1}_conv2 launches of DL2Net as ONE persistent kernel
# (an nvxrt launch group). Same arithmetic as _conv.conv2 (and so as T0): per output element
#   acc_0 = bias (f16), acc_{s+1} = RNE_f16(acc_s + sum_k A_s[p][k] B_s[k][oc]), s = tap * CB + cb, then
#   v = max(v * 0.01, v); v = min(max(v, -6), 6); out = v + (flag == 1 ? residual : +0)   (f16)
# What changes is the schedule: every workgroup owns one output block per layer for the whole chain, the weights of a
# layer are read once per warp (into registers for 'reg', streamed from the L0-cached weight buffer for 'glob'), the
# next layer's weights are fetched before the grid barrier, and there is one launch instead of eight.
# Activations written by one layer and read by the next go through device-coherent loads/stores (_fuse.group_rt).
import os
import _conv, _fuse
import ptxc

PROF = bool(os.environ.get('T1_PROF'))   # realtime stamps per layer phase (nvxrt NVXRT_GROUP_PROFILE=1)

CM = 'gl_ScopeSubgroup, 16, 16'


def conv2_chain(C, BW, BH, NTW, bmode, n_members):
    """C channels; block BW x BH output pixels x NTW of the C/16 output-channel tiles per workgroup (8 warps);
    warp w: row r = w % BH... see body. bmode 'reg': all S = 9*C/16 weight fragments of the warp's channel tile in
    registers; 'glob': streamed per step from global memory, 6 steps ahead."""
    CB = C // 16
    S = 9 * CB
    NT_ALL = C // 16
    assert NT_ALL % NTW == 0
    MT = BW // 16                     # 16-pixel M tiles per block row
    warps = 8
    assert (BH * NTW) in (8, 4, 2) and warps % (BH * NTW) == 0
    MPW = MT * BH * NTW // warps       # M tiles per warp
    assert MPW >= 1 and MT % MPW == 0
    CS = C + 8
    TW2 = BW + 2
    TIN = TW2 * (BH + 2) * CS          # halves
    BOFF = TIN                         # bias rows [NTW][16][16]
    TOT = BOFF + NTW * 256
    STG = 0                            # epilogue staging aliases the input tile (after a barrier)
    assert warps * MPW * 256 <= TIN
    rt = _fuse.group_rt(_conv.RT)
    words = (TOT + 7) // 8 * 4
    g = dict(C=C, CB=CB, S=S, CS=CS, BW=BW, BH=BH, TW2=TW2, NTW=NTW, NT_ALL=NT_ALL, MT=MT, MPW=MPW, BOFF=BOFF,
             CM=CM, NIN=TW2 * (BH + 2) * (C // 8), C8=C // 8, NB=NTW * 256, NM=n_members,
             ROUND=_conv.ROUND % {'m': 'acc[j]'}, GO=_fuse.GRIDOFF, SPIN=_fuse.SPIN, SPIN0=_fuse.SPIN0,
             NBX=0)
    g['NFR'] = NTW * S
    # 'glob' B prefetch depth (steps ahead). gfx11's B fragments take 16 halves per lane (gfx12: 8), so 6 steps
    # ahead need twice the registers there; 3 keeps the chain within 128 VGPRs, i.e. 3 workgroups per CU, which
    # the grid barrier needs to have all 120 resident on a 54-CU GPU.
    g['PF'] = int(os.environ.get('T1_CHAIN_PF') or (3 if os.environ.get('PTXC_LAYOUT') == 'gfx11' else 6))
    assert S % g['PF'] == 0
    g['NTQ'] = (TW2 * (BH + 2) * (C // 8) + 255) // 256
    g['NBQ'] = (NTW * S * 32 + 255) // 256
    g['FH'] = (NTW * S + 1) // 2 // NTW * NTW   # fragments per staging pass (whole steps)
    assert bmode != 'reg' or g['FH'] * 512 <= TIN * 2, 'B staging pass must fit in the input tile region'
    src = (ptxc.HEADER +
           '#extension GL_EXT_shared_memory_block : require\n'
           '#extension GL_EXT_control_flow_attributes : require\n'
           + ('#extension GL_EXT_shader_realtime_clock : require\n' if PROF else '') +
           'shared NvxSMU { uint smem[%d]; };\n'
           'shared NvxSMH { float16_t smh[%d]; };\n'
           'shared NvxSMV { uvec4 smv[%d]; };\n' % (words, words * 2, words // 4) + rt)
    if bmode == 'reg':
        bdecl = 'coopmat<float16_t, %(CM)s, gl_MatrixUseB> bw[%(S)d];' % g
        bload = r'''
// weights-stationary B (reg mode): the workgroup fetches its NTW x S fragments once per layer (fragment f = s*NTW + nt,
// 32 x 16-byte lane rows each, NBQ per thread) before the grid barrier; after it they go through LDS in two passes
// (aliasing the input tile, which is loaded later) and every warp coopMatLoads its channel tile's S fragments.
void fetch_bq(out uvec4 bq[%(NBQ)d], uint64_t wp)
{
    [[unroll]] for (uint i = 0u; i < %(NBQ)du; i++) {
        uint e = gl_LocalInvocationIndex + 256u * i;
        bq[i] = uvec4(0u);
        if (e < %(NFR)du * 32u) {
            // fragment order rotated per workgroup: spreads the simultaneous weight reads of all workgroups
            uint f = ((e >> 5) + gl_WorkGroupID.x * 7u) %% %(NFR)du, ls = e & 31u, s = f / %(NTW)du, nt = f %% %(NTW)du;
            uint n = ls & 15u, kh = ls >> 4, z = nt * 2u + (n >> 3);
            bq[i] = NvxW4(wp + uint64_t((z * %(S)du * 128u + s * 128u + kh * 64u + (n & 7u) * 8u) * 2u)).v;
        }
    }
}
void stage_b(out coopmat<float16_t, %(CM)s, gl_MatrixUseB> bw[%(S)d], uvec4 bq[%(NBQ)d], uint ntl)
{
    [[unroll]] for (uint h = 0u; h < 2u; h++) {
        uint f0 = h * %(FH)du;                       // first fragment of this pass
        [[unroll]] for (uint i = 0u; i < %(NBQ)du; i++) {
            uint e = gl_LocalInvocationIndex + 256u * i, f = ((e >> 5) + gl_WorkGroupID.x * 7u) %% %(NFR)du, ls = e & 31u;
            if (e < %(NFR)du * 32u && f >= f0 && f < f0 + %(FH)du)
                smv[(f - f0) * 32u + (ls & 15u) * 2u + (ls >> 4)] = bq[i];
        }
        barrier();
        [[unroll]] for (uint s = 0u; s < %(S)du; s++) {
            uint f = s * %(NTW)du + ntl;
            if (s * %(NTW)du >= f0 && s * %(NTW)du < f0 + %(FH)du)
                coopMatLoad(bw[s], smh, (f - f0) * 256u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
        }
        barrier();
    }
}
''' % g
    else:
        bdecl = ''
        bload = ''
    src += r'''
layout(buffer_reference, buffer_reference_align = 16) readonly buffer NvxW4 { uvec4 v; };   // weights: never written
// grid-wide barrier between members. Bounded: a workgroup that waits SPIN polls sets the poison flag (region + 64),
// after which no barrier of this dispatch waits any more (wrong output for that frame, never a GPU hang); nvxrt sees
// the flag and disables the group. nvx_resident(): the same at the start of the dispatch with a short bound, so a
// grid that is not fully co-resident is detected within microseconds.
void nvx_gsync_(uint64_t reg, uint spin)
{
    controlBarrier(gl_ScopeWorkgroup, gl_ScopeQueueFamily,
                   gl_StorageSemanticsBuffer | gl_StorageSemanticsImage | gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
    if (gl_LocalInvocationIndex == 0u) {
        uint n = gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_NumWorkGroups.z;
        uint old = atomicAdd(NvxU32A(reg + 128ul).v, 1u, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);
        uint target = (old / n + 1u) * n;
        for (uint i = 0u; i < spin; i++) {
            if (atomicLoad(NvxU32A(reg + 128ul).v, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsAcquire) >= target)
                break;
            if ((i & 31u) == 0u && atomicLoad(NvxU32A(reg + 64ul).v, gl_ScopeQueueFamily, 0, gl_SemanticsRelaxed) != 0u)
                break;
            if (i + 1u == spin)
                atomicStore(NvxU32A(reg + 64ul).v, 1u, gl_ScopeQueueFamily, 0, gl_SemanticsRelaxed);
        }
    }
    controlBarrier(gl_ScopeWorkgroup, gl_ScopeQueueFamily,
                   gl_StorageSemanticsBuffer | gl_StorageSemanticsImage | gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
}
void nvx_gridsync(uint64_t ctr) { nvx_gsync_(ctr - 128ul, %(SPIN)du); }
void nvx_resident(uint64_t reg) { nvx_gsync_(reg, %(SPIN0)du); }
''' % g
    src += ('void stamp(uint64_t reg, uint idx) { if (gl_LocalInvocationIndex == 0u && (gl_WorkGroupID.x == 0u || gl_WorkGroupID.x == ' + os.environ.get('T1_PROF_WG', 'gl_NumWorkGroups.x - 1') + 'u)) '
            'NvxU64(reg + 15360ul + 8ul * uint64_t(idx + (gl_WorkGroupID.x == 0u ? 0u : 64u))).v = clockRealtimeEXT(); }\n' if PROF else
            'void stamp(uint64_t reg, uint idx) {}\n')
    src += bload
    # B for one step in 'glob' mode, built in the WMMA register layout (see ptxc: gfx12 lane l = column l&15, K half
    # l>>4; gfx11 (PTXC_LAYOUT=gfx11) lane l = column l&15, all 16 K, lanes 16-31 repeating 0-15)
    if os.environ.get('PTXC_LAYOUT') == 'gfx11':
        src += r'''
coopmat<float16_t, %(CM)s, gl_MatrixUseB> fetch_b(uint64_t wp, uint nt, uint s)
{
    uint l = gl_SubgroupInvocationID, n = l & 15u;
    uint z = nt * 2u + (n >> 3);
    uint64_t a = wp + uint64_t((z * %(S)du * 128u + s * 128u + (n & 7u) * 8u) * 2u);
    uvec4 q0 = NvxW4(a).v, q1 = NvxW4(a + 128ul).v;
    coopmat<float16_t, %(CM)s, gl_MatrixUseB> b;
    [[unroll]] for (uint t = 0u; t < 4u; t++) {
        f16vec2 h0 = unpackFloat2x16(q0[t]), h1 = unpackFloat2x16(q1[t]);
        b[2u * t] = h0.x;
        b[2u * t + 1u] = h0.y;
        b[8u + 2u * t] = h1.x;
        b[9u + 2u * t] = h1.y;
    }
    return b;
}
''' % g
    else:
        src += r'''
coopmat<float16_t, %(CM)s, gl_MatrixUseB> fetch_b(uint64_t wp, uint nt, uint s)
{
    uint l = gl_SubgroupInvocationID, n = l & 15u, kh = l >> 4;
    uint z = nt * 2u + (n >> 3);
    uvec4 q = NvxW4(wp + uint64_t((z * %(S)du * 128u + s * 128u + kh * 64u + (n & 7u) * 8u) * 2u)).v;
    coopmat<float16_t, %(CM)s, gl_MatrixUseB> b;
    [[unroll]] for (uint t = 0u; t < 4u; t++) {
        f16vec2 h = unpackFloat2x16(q[t]);
        b[2u * t] = h.x;
        b[2u * t + 1u] = h.y;
    }
    return b;
}
''' % g
    if bmode == 'reg':
        mma = r'''
            [[unroll]] for (uint s = 0u; s < %(S)du; s++) {
                uint t = s / %(CB)du, cb = s %% %(CB)du, ky = t / 3u, kx = t %% 3u;
                [[unroll]] for (uint j = 0u; j < %(MPW)du; j++) {
                    coopmat<float16_t, %(CM)s, gl_MatrixUseA> a;
                    coopMatLoad(a, smh, ((r + ky) * %(TW2)du + (mx0 + j) * 16u + kx) * %(CS)du + cb * 16u, %(CS)du,
                                gl_CooperativeMatrixLayoutRowMajor);
                    acc[j] = coopMatMulAdd(a, bw[s], acc[j]);
                    %(ROUND)s
                }
            }'''
    else:
        mma = r'''
            coopmat<float16_t, %(CM)s, gl_MatrixUseB> bq[%(PF)d];
            [[unroll]] for (uint s = 0u; s < %(PF)du; s++) bq[s] = fetch_b(wp, ntg, s);
            [[unroll]] for (uint s0 = 0u; s0 < %(S)du; s0 += %(PF)du) {
                coopmat<float16_t, %(CM)s, gl_MatrixUseB> bn[%(PF)d];
                [[unroll]] for (uint s = 0u; s < %(PF)du; s++)
                    if (s0 + %(PF)du + s < %(S)du) bn[s] = fetch_b(wp, ntg, s0 + %(PF)du + s);
                [[unroll]] for (uint ss = 0u; ss < %(PF)du; ss++) {
                    uint s = s0 + ss;
                    uint t = s / %(CB)du, cb = s %% %(CB)du, ky = t / 3u, kx = t %% 3u;
                    [[unroll]] for (uint j = 0u; j < %(MPW)du; j++) {
                        coopmat<float16_t, %(CM)s, gl_MatrixUseA> a;
                        coopMatLoad(a, smh, ((r + ky) * %(TW2)du + (mx0 + j) * 16u + kx) * %(CS)du + cb * 16u, %(CS)du,
                                    gl_CooperativeMatrixLayoutRowMajor);
                        acc[j] = coopMatMulAdd(a, bq[ss], acc[j]);
                        %(ROUND)s
                    }
                }
                [[unroll]] for (uint s = 0u; s < %(PF)du; s++) bq[s] = bn[s];
            }'''
    body = r'''
void main()
{
    uint64_t reg = packUint2x32(uvec2(pc.w[0], pc.w[1]));
    uint count = min(NvxPW(reg).v, %(NM)du);
    nvx_resident(reg);
    uint lid = gl_LocalInvocationIndex, nthr = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    uint w = gl_SubgroupID;
    // warp -> (block row r, channel tile ntl of this workgroup's NTW, first M tile mx0 of its MPW)
    uint ntl = w %% %(NTW)du, rest = w / %(NTW)du;
    uint r = rest %% %(BH)du, mx0 = (rest / %(BH)du) * %(MPW)du;
    %(BDECL)s
    for (uint k = 0u; k < count; k++) {
        nvx_pb = reg + 256ul + uint64_t(k) * 256ul;
        uint64_t wp = nvx_p64(0u), bp = nvx_p64(8u), ip = nvx_p64(32u), rp = nvx_p64(40u), op = nvx_p64(64u);
        int H = int(nvx_p32(52u)), W = int(nvx_p32(56u));
        uint cpair = nvx_p32(28u) >> 1;
        uint flag = nvx_p32(88u);
        uint nbx = (uint(W) + %(BW)du - 1u) / %(BW)du, nby = (uint(H) + %(BH)du - 1u) / %(BH)du;
        uint nblk = nbx * nby * %(NGRP)du;
        stamp(reg, 6u * k);
        %(BFIRST)s
        if (k != 0u)
            nvx_gridsync(reg + 128ul);
        stamp(reg, 6u * k + 1u);
        %(BSTAGE)s
        stamp(reg, 6u * k + 2u);
        // blocks beyond one per workgroup (only at large resolutions) first, the workgroup's own block last: after its
        // MMA the next layer's weights are fetched while the epilogue runs
        for (uint v = gl_WorkGroupID.x + gl_NumWorkGroups.x; v < nblk; v += gl_NumWorkGroups.x) {
            uint bz = v %% %(NGRP)du, bxy = v / %(NGRP)du;
            int x0 = int(bxy %% nbx) * %(BW)d, y0 = int(bxy / nbx) * %(BH)d;
            uint ntg = bz * %(NTW)du + ntl;       // global channel tile of this warp
            for (uint e = lid; e < %(NIN)du; e += nthr) {
                uint c8 = e %% %(C8)du, p = e / %(C8)du;
                int gx = x0 - 1 + int(p %% %(TW2)du), gy = y0 - 1 + int(p / %(TW2)du);
                uvec4 q = uvec4(0u);
                if (gx >= 0 && gy >= 0 && gx < W && gy < H)
                    q = NvxU32x4(ip + uint64_t((uint(gy * W + gx) * %(C)du + c8 * 8u) * 2u)).v;
                smv[(p * %(CS)du + c8 * 8u) >> 3] = q;
            }
            for (uint e = lid; e < %(NB)du; e += nthr) {
                uint col = e & 15u, nt = e >> 8;
                smh[%(BOFF)du + e] = uint16BitsToFloat16(NvxU16(bp + uint64_t(((bz * %(NTW)du + nt) * 16u + col) * 2u)).v);
            }
            barrier();
            stamp(reg, 6u * k + 3u);
            coopmat<float, %(CM)s, gl_MatrixUseAccumulator> acc[%(MPW)d];
            {
                coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator> c16;
                coopMatLoad(c16, smh, %(BOFF)du + ntl * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint j = 0u; j < %(MPW)du; j++)
                    acc[j] = coopmat<float, %(CM)s, gl_MatrixUseAccumulator>(c16);
            }
%(MMA)s
            barrier();                              // input tile dead: the staging below reuses its space
            stamp(reg, 6u * k + 4u);
            uint stg = w * %(MPW)du * 256u;
            [[unroll]] for (uint j = 0u; j < %(MPW)du; j++)
                coopMatStore(coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator>(acc[j]), smh, stg + j * 256u, 16u,
                             gl_CooperativeMatrixLayoutRowMajor);
            subgroupMemoryBarrierShared(); subgroupBarrier();
            uint16_t h001 = nvx_f2h(uintBitsToFloat(0x3C23D70AU)), h6 = nvx_f2h(uintBitsToFloat(0x40C00000U));
            uint16_t h0 = nvx_f2h(uintBitsToFloat(0x00000000U));
            uint k001 = uint(h001) | (uint(h001) << 16), k6 = uint(h6) | (uint(h6) << 16), kz = uint(h0) | (uint(h0) << 16);
            [[unroll]] for (uint q = 0u; q < 4u * %(MPW)du; q++) {
                uint e = gl_SubgroupInvocationID + 32u * q;     // (M tile j, pixel i, channel pair pr)
                uint j = e >> 7, i = (e >> 3) & 15u, pr = e & 7u;
                int gx = x0 + int((mx0 + j) * 16u + i), gy = y0 + int(r);
                if (gx >= W || gy >= H)
                    continue;
                uint vv = smem[(stg >> 1) + j * 128u + i * 8u + pr];
                uint idx = uint(gy * W + gx) * cpair + ntg * 8u + pr;
                uint m1 = packFloat2x16(unpackFloat2x16(vv) * unpackFloat2x16(k001));
                f16vec2 a1 = unpackFloat2x16(m1), b1 = unpackFloat2x16(vv);
                uint m2 = packFloat2x16(max(a1, b1));
                uint n6 = k6 ^ 0x80008000u;
                f16vec2 a2 = unpackFloat2x16(m2), b2 = unpackFloat2x16(n6);
                uint m3 = packFloat2x16(max(a2, b2));
                f16vec2 a3 = unpackFloat2x16(m3), b3 = unpackFloat2x16(k6);
                uint m4 = packFloat2x16(min(a3, b3));
                uint rs = int(flag) == 1 ? NvxU32(rp + uint64_t(idx) * 4ul).v : kz;
                NvxU32(op + uint64_t(idx) * 4ul).v = packFloat2x16(unpackFloat2x16(m4) + unpackFloat2x16(rs));
            }
            barrier();                              // other warps' staging aliases the next block's input tile
                }
        if (gl_WorkGroupID.x < nblk) {
            uint v = gl_WorkGroupID.x;
            uint bz = v %% %(NGRP)du, bxy = v / %(NGRP)du;
            int x0 = int(bxy %% nbx) * %(BW)d, y0 = int(bxy / nbx) * %(BH)d;
            uint ntg = bz * %(NTW)du + ntl;       // global channel tile of this warp
            for (uint e = lid; e < %(NIN)du; e += nthr) {
                uint c8 = e %% %(C8)du, p = e / %(C8)du;
                int gx = x0 - 1 + int(p %% %(TW2)du), gy = y0 - 1 + int(p / %(TW2)du);
                uvec4 q = uvec4(0u);
                if (gx >= 0 && gy >= 0 && gx < W && gy < H)
                    q = NvxU32x4(ip + uint64_t((uint(gy * W + gx) * %(C)du + c8 * 8u) * 2u)).v;
                smv[(p * %(CS)du + c8 * 8u) >> 3] = q;
            }
            for (uint e = lid; e < %(NB)du; e += nthr) {
                uint col = e & 15u, nt = e >> 8;
                smh[%(BOFF)du + e] = uint16BitsToFloat16(NvxU16(bp + uint64_t(((bz * %(NTW)du + nt) * 16u + col) * 2u)).v);
            }
            barrier();
            stamp(reg, 6u * k + 3u);
            coopmat<float, %(CM)s, gl_MatrixUseAccumulator> acc[%(MPW)d];
            {
                coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator> c16;
                coopMatLoad(c16, smh, %(BOFF)du + ntl * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint j = 0u; j < %(MPW)du; j++)
                    acc[j] = coopmat<float, %(CM)s, gl_MatrixUseAccumulator>(c16);
            }
%(MMA)s
            barrier();                              // input tile dead: the staging below reuses its space
            stamp(reg, 6u * k + 4u);
            bool lastb = k + 1u < count;
            %(BFETCH)s
            uint stg = w * %(MPW)du * 256u;
            [[unroll]] for (uint j = 0u; j < %(MPW)du; j++)
                coopMatStore(coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator>(acc[j]), smh, stg + j * 256u, 16u,
                             gl_CooperativeMatrixLayoutRowMajor);
            subgroupMemoryBarrierShared(); subgroupBarrier();
            uint16_t h001 = nvx_f2h(uintBitsToFloat(0x3C23D70AU)), h6 = nvx_f2h(uintBitsToFloat(0x40C00000U));
            uint16_t h0 = nvx_f2h(uintBitsToFloat(0x00000000U));
            uint k001 = uint(h001) | (uint(h001) << 16), k6 = uint(h6) | (uint(h6) << 16), kz = uint(h0) | (uint(h0) << 16);
            [[unroll]] for (uint q = 0u; q < 4u * %(MPW)du; q++) {
                uint e = gl_SubgroupInvocationID + 32u * q;     // (M tile j, pixel i, channel pair pr)
                uint j = e >> 7, i = (e >> 3) & 15u, pr = e & 7u;
                int gx = x0 + int((mx0 + j) * 16u + i), gy = y0 + int(r);
                if (gx >= W || gy >= H)
                    continue;
                uint vv = smem[(stg >> 1) + j * 128u + i * 8u + pr];
                uint idx = uint(gy * W + gx) * cpair + ntg * 8u + pr;
                uint m1 = packFloat2x16(unpackFloat2x16(vv) * unpackFloat2x16(k001));
                f16vec2 a1 = unpackFloat2x16(m1), b1 = unpackFloat2x16(vv);
                uint m2 = packFloat2x16(max(a1, b1));
                uint n6 = k6 ^ 0x80008000u;
                f16vec2 a2 = unpackFloat2x16(m2), b2 = unpackFloat2x16(n6);
                uint m3 = packFloat2x16(max(a2, b2));
                f16vec2 a3 = unpackFloat2x16(m3), b3 = unpackFloat2x16(k6);
                uint m4 = packFloat2x16(min(a3, b3));
                uint rs = int(flag) == 1 ? NvxU32(rp + uint64_t(idx) * 4ul).v : kz;
                NvxU32(op + uint64_t(idx) * 4ul).v = packFloat2x16(unpackFloat2x16(m4) + unpackFloat2x16(rs));
            }
            %(BNEXTST)s
                } else if (k + 1u < count) {
            bool lastb = true;
            %(BFETCH)s
            %(BNEXTST)s
        }
        stamp(reg, 6u * k + 5u);
        %(BNEXT)s
    }
}
'''
    NGRP = NT_ALL // NTW
    g['NGRP'] = NGRP
    if bmode == 'reg':
        assert NGRP == 1, 'reg mode keeps one channel tile per warp for the whole layer'
        g['BDECL'] = bdecl + ' uvec4 bq[%d];' % g['NBQ']
        g['BFIRST'] = 'if (k == 0u) { fetch_bq(bq, wp); barrier(); stage_b(bw, bq, ntl); }'
        g['BNEXT'] = ''
        g['BSTAGE'] = ''
        g['BFETCH'] = 'if (lastb) { nvx_pb = reg + 256ul + uint64_t(k + 1u) * 256ul; fetch_bq(bq, nvx_p64(0u)); nvx_pb = reg + 256ul + uint64_t(k) * 256ul; }'
        g['BNEXTST'] = 'if (lastb) { barrier(); stage_b(bw, bq, ntl); }'
    else:
        g['BDECL'] = ''
        g['BFIRST'] = ''
        g['BNEXT'] = ''
        g['BSTAGE'] = ''
        g['BFETCH'] = ''
        g['BNEXTST'] = ''
    g['MMA'] = mma % g
    src += body % g
    return src
