# T1: exact-mode 3x3 convolution kernels for the DL2Net "custom_block*" layers of
# nvngx_dlssg 310.9.1, generated as complete GLSL kernels.
#
# Semantics (taken from the PTX, identical to the T0 translation):
#   acc_0 = bias[oc] (f16);  acc_{s+1} = RNE_f16(acc_s + sum_{k<16} A_s[pixel][k] * B_s[k][oc])  (one WMMA per step)
#   step s = tap t (ky*3+kx, row-major) * CB + channel block cb (CB = C_in/16): A_s[p][k] = in[y+ky-1][x+kx-1][16cb+k]
#   (0 outside the image); B_s from NVIDIA's per-8-channel-group fragment layout; then the layer's f16 epilogue.
# What T1 changes: one 16x16x16 WMMA serves two 8-channel groups (columns 0-7 / 8-15: every output element is still
# the same dot product of the same 16 products), A and B come from LDS with coopMatLoad (16 consecutive pixels of one
# image row per M tile), weights are staged once per workgroup, the epilogue works on 16-byte rows.
import os
import re
import ptxc

RT = open(os.path.join(os.path.dirname(os.path.abspath(ptxc.__file__)), 'rt.glsl')).read()

ROUND_EXACT = ('{ uint z_ = nvx_opaque_zero(); for (int i_ = 0; i_ < %(m)s.length(); i_ += 2) { '
               'uint w_ = packFloat2x16(f16vec2(float16_t(%(m)s[i_]), float16_t(%(m)s[i_ + 1]))) ^ z_; '
               'f16vec2 h_ = unpackFloat2x16(w_); %(m)s[i_] = float(h_.x); %(m)s[i_ + 1] = float(h_.y); } }')
# Fast accumulation (NVXRT_ACC=fast, NOT exact): fp32 accumulation over the whole chain, one rounding at the store.
# Selected for a whole t1_apply run by NVX_T1_ACC=fast (t1_apply.py --acc fast writes the ENTRY.fa*.spv variants).
ROUND_FAST = '{ /* fast accumulation: no per-step fp16 rounding of %(m)s (NVX_T1_ACC=fast) */ }'
FAST_ACC = os.environ.get('NVX_T1_ACC') == 'fast'
ROUND = ROUND_FAST if FAST_ACC else ROUND_EXACT

CM = 'gl_ScopeSubgroup, 16, 16'

# The persistent grids below were sized for RX 9070's 56 CUs (168 = 3 workgroups per CU, ...). A grid larger than the
# GPU holds at once runs its last workgroups as a second wave (on a 54-CU RX 7700 XT, 168 workgroups at 3 per CU leave
# 6 for later: almost twice the time). NVXRT_CUS (dlssg-setup reads it from KFD) rescales them; the tiles a workgroup
# computes do not depend on the grid size, so the results are the same.
TUNED_CUS = 56
# RDNA3 (gfx11): the tile shapes chosen per kernel where they differ (measured on RX 7700 XT; results are the same for
# every shape, only the work split changes). RDNA3 prefers small workgroups for these layers.
GFX11 = os.environ.get('PTXC_LAYOUT') == 'gfx11'


def fit_persistent(persistent, ngrp):
    cus = int(os.environ.get('NVXRT_CUS') or TUNED_CUS)
    if cus == TUNED_CUS:
        return persistent
    per_cu = max(1, round(persistent / TUNED_CUS))
    return max(ngrp, per_cu * cus // ngrp * ngrp)


def header(smem_halves):
    words = (smem_halves + 7) // 8 * 4
    return (ptxc.HEADER +
            '#extension GL_EXT_shared_memory_block : require\n'
            '#extension GL_EXT_control_flow_attributes : require\n'
            'shared NvxSMU { uint smem[%d]; };\n'
            'shared NvxSMH { float16_t smh[%d]; };\n'
            'shared NvxSMV { uvec4 smv[%d]; };\n' % (words, words * 2, words // 4) + RT)


def conv2(C, TH, NT, warps=8, NTALL=None):
    """custom_block{0,1}_conv2_kernel: C -> C channels, epilogue
         v = acc; v = max(v * 0.01, v); v = min(max(v, -6), 6); out = v + (flag == 1 ? residual : +0)
       params: 0 weights, 8 bias, 28 C_in, 32 input, 40 residual, 52 H, 56 W, 64 output, 88 flag.
       Workgroup: 16 x TH output pixels, all C channels; warp = (image row r, 16-channel tile nt)."""
    assert TH * NT == warps
    NTALL = NTALL or NT              # 16-channel tiles of the layer; a workgroup does NT of them (grid z splits)
    TW = 16
    CB = C // 16
    S = 9 * CB
    CS = C + 8                       # pixel stride in the LDS tile (halves): rows of a coopMatLoad hit other banks
    TIN = (TW + 2) * (TH + 2) * CS   # input tile with halo
    WOFF = TIN                       # weights: [s][nt][n 16][k 16] (column-major B)
    BOFF = WOFF + S * NT * 256       # bias rows: [nt][16][16]
    STG = BOFF + NT * 256            # per-warp epilogue staging [16 pixels][16 channels]
    TOT = STG + warps * 256
    assert TIN % 8 == 0
    g = dict(TW=TW, TH=TH, C=C, CB=CB, S=S, CS=CS, NT=NT, NTALL=NTALL, WOFF=WOFF, BOFF=BOFF, STG=STG, CM=CM,
             NIN=(TW + 2) * (TH + 2) * (C // 8), NW=S * NT * 32, NB=NT * 256,
             ROUND=ROUND % {'m': 'acc'})
    body = r'''
void main() {
    uint64_t wp = nvx_p64(0u), bp = nvx_p64(8u), ip = nvx_p64(32u), rp = nvx_p64(40u), op = nvx_p64(64u);
    int H = int(nvx_p32(52u)), W = int(nvx_p32(56u));
    uint cpair = nvx_p32(28u) >> 1;                    // u32 (channel pair) stride per pixel, as the PTX uses
    uint flag = nvx_p32(88u);
    uint lid = gl_LocalInvocationIndex;
    int x0 = int(gl_WorkGroupID.x) * %(TW)d, y0 = int(gl_WorkGroupID.y) * %(TH)d;
    // input tile with a 1-pixel halo, 8 channels (16 bytes) per item; outside the image: f16 +0
    for (uint e = lid; e < %(NIN)du; e += gl_WorkGroupSize.x * gl_WorkGroupSize.y) {
        uint c8 = e %% %(C8)du, p = e / %(C8)du;
        int gx = x0 - 1 + int(p %% %(TW2)du), gy = y0 - 1 + int(p / %(TW2)du);
        uvec4 v = uvec4(0u);
        if (gx >= 0 && gy >= 0 && gx < W && gy < H)
            v = NvxU32x4(ip + uint64_t((uint(gy * W + gx) * %(C)du + c8 * 8u) * 2u)).v;
        smv[(p * %(CS)du + c8 * 8u) >> 3] = v;
    }
    // weights: NVIDIA's layout per 8-channel group z, step s: halves z*S*128 + s*128 + kh*64 + n*8 + (0..7) hold
    // B[k = 8kh + 0..7][n]; staged as [s][nt][16 columns][16 k]
    for (uint e = lid; e < %(NW)du; e += gl_WorkGroupSize.x * gl_WorkGroupSize.y) {
        uint kh = e & 1u, n = (e >> 1) & 15u, q = e >> 5, nt = q %% %(NT)du, s = q / %(NT)du;
        uint z = (gl_WorkGroupID.z * %(NT)du + nt) * 2u + (n >> 3);
        smv[(%(WOFF)du + (s * %(NT)du + nt) * 256u + n * 16u + kh * 8u) >> 3] =
            NvxU32x4(wp + uint64_t((z * %(S)du * 128u + s * 128u + kh * 64u + (n & 7u) * 8u) * 2u)).v;
    }
    // bias rows (the C operand of the first step, every row the same)
    for (uint e = lid; e < %(NB)du; e += gl_WorkGroupSize.x * gl_WorkGroupSize.y) {
        uint col = e & 15u, nt = e >> 8;
        smh[%(BOFF)du + e] = uint16BitsToFloat16(NvxU16(bp + uint64_t(((gl_WorkGroupID.z * %(NT)du + nt) * 16u + col) * 2u)).v);
    }
    barrier();
    uint w = gl_SubgroupID, r = w %% %(TH)du, nt = w / %(TH)du;
    coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator> c16;
    coopMatLoad(c16, smh, %(BOFF)du + nt * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
    coopmat<float, %(CM)s, gl_MatrixUseAccumulator> acc = coopmat<float, %(CM)s, gl_MatrixUseAccumulator>(c16);
    [[unroll]] for (uint s = 0u; s < %(S)du; s++) {
        uint t = s / %(CB)du, cb = s %% %(CB)du;
        uint ky = t / 3u, kx = t %% 3u;
        coopmat<float16_t, %(CM)s, gl_MatrixUseA> a;
        coopMatLoad(a, smh, ((r + ky) * %(TW2)du + kx) * %(CS)du + cb * 16u, %(CS)du, gl_CooperativeMatrixLayoutRowMajor);
        coopmat<float16_t, %(CM)s, gl_MatrixUseB> b;
        coopMatLoad(b, smh, %(WOFF)du + (s * %(NT)du + nt) * 256u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
        acc = coopMatMulAdd(a, b, acc);
        %(ROUND)s
    }
    uint stg = %(STG)du + w * 256u;
    coopMatStore(coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator>(acc), smh, stg, 16u, gl_CooperativeMatrixLayoutRowMajor);
    subgroupMemoryBarrierShared(); subgroupBarrier();
    // epilogue, per channel pair exactly as the PTX: leaky 0.01, clamp to [-6, 6], then the residual (or +0)
    uint16_t h001 = nvx_f2h(uintBitsToFloat(0x3C23D70AU)), h6 = nvx_f2h(uintBitsToFloat(0x40C00000U));
    uint16_t h0 = nvx_f2h(uintBitsToFloat(0x00000000U));
    uint k001 = uint(h001) | (uint(h001) << 16), k6 = uint(h6) | (uint(h6) << 16), kz = uint(h0) | (uint(h0) << 16);
    [[unroll]] for (uint q = 0u; q < 4u; q++) {
        uint e = gl_SubgroupInvocationID + 32u * q;          // channel pair 0..127 of this warp's 16x16 tile
        uint i = e >> 3, pr = e & 7u;
        int gx = x0 + int(i), gy = y0 + int(r);
        if (gx >= W || gy >= H)
            continue;
        uint v = smem[(stg >> 1) + i * 8u + pr];
        uint idx = uint(gy * W + gx) * cpair + (gl_WorkGroupID.z * %(NT)du + nt) * 8u + pr;
        uint m1 = packFloat2x16(unpackFloat2x16(v) * unpackFloat2x16(k001));
        f16vec2 a1 = unpackFloat2x16(m1), b1 = unpackFloat2x16(v);
        uint m2 = packFloat2x16(max(a1, b1));
        uint n6 = k6 ^ 0x80008000u;
        f16vec2 a2 = unpackFloat2x16(m2), b2 = unpackFloat2x16(n6);
        uint m3 = packFloat2x16(max(a2, b2));
        f16vec2 a3 = unpackFloat2x16(m3), b3 = unpackFloat2x16(k6);
        uint m4 = packFloat2x16(min(a3, b3));
        uint res = int(flag) == 1 ? NvxU32(rp + uint64_t(idx) * 4ul).v : kz;
        NvxU32(op + uint64_t(idx) * 4ul).v = packFloat2x16(unpackFloat2x16(m4) + unpackFloat2x16(res));
    }
}
'''
    g['TW2'] = TW + 2
    g['C8'] = C // 8
    return header(TOT) + body % g, ['t1_block 32 %d 1' % warps]


def conv_pool(CB, C_out, TH, NT, NTALL=None, warps=8, wl='frag', persistent=None):
    """custom_block*_conv* kernels that end in a 2x2 average pool (convPre, block0 conv0_c8/conv1, block1 conv1):
         per conv pixel: v = acc; v = max(v * 0.01, v); v = min(max(v, -6), 6)   (stored to C_smem in the PTX)
         per pooled pixel: o = (((v00 + v10) + v01) + v11) * 0.25, out[(Y*(W/2) + X)*(C_out/2) + pair]
       params: 0 weights (fragment layout, S = 9*CB steps of 16 K), 8 bias, 16 C_out, 32 input, 44 H, 48 W,
       52 C (input pixel stride; channels >= 2*(C/2) read as 0), 56 output, 68 H2, 72 W2.
       Workgroup: 16 x TH conv pixels (8 x TH/2 pooled), NT of the NTALL 16-channel tiles (grid z picks them)."""
    assert TH * NT == warps and TH % 2 == 0
    NTALL = NTALL or NT
    TW = 16
    S = 9 * CB
    CIN = CB * 16
    CS = CIN + 8
    TIN = (TW + 2) * (TH + 2) * CS
    WOFF = TIN
    BOFF = WOFF + S * NT * 256
    CW = NT * 16                        # channels of this workgroup
    STG = BOFF + NT * 256               # conv results [TH][16][CW]
    TOT = STG + TH * TW * CW
    assert TIN % 8 == 0 and (BOFF % 8) == 0
    g = dict(TW=TW, TW2=TW + 2, TH=TH, CB=CB, CIN=CIN, C8=CIN // 8, S=S, CS=CS, NT=NT, WOFF=WOFF, BOFF=BOFF,
             STG=STG, CW=CW, CM=CM, NIN=(TW + 2) * (TH + 2) * (CIN // 8), NW=S * NT * 32, NB=NT * 256,
             NP=(TW // 2) * (TH // 2) * (CW // 2), ROUND=ROUND % {'m': 'acc'})
    body = r'''
void main() {
    uint64_t wp = nvx_p64(0u), bp = nvx_p64(8u), ip = nvx_p64(32u), op = nvx_p64(56u);
    uint cout2 = nvx_p32(16u) >> 1;
    int H = int(nvx_p32(44u)), W = int(nvx_p32(48u));
    uint cpix = nvx_p32(52u), cpair = cpix >> 1;          // input pixel stride (halves); pairs >= cpair read 0
    int H2 = int(nvx_p32(68u)), W2 = int(nvx_p32(72u));
    uint lid = gl_LocalInvocationIndex;
    uint nthr = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    int x0 = int(gl_WorkGroupID.x) * %(TW)d, y0 = int(gl_WorkGroupID.y) * %(TH)d;
    uint zt = gl_WorkGroupID.z * %(NT)du;
    // input tile with halo, per 32-bit channel pair (the input pixel stride need not be 16-byte aligned)
    for (uint e = lid; e < %(NIN)du * 4u; e += nthr) {
        uint c2 = e %% %(CP)du, p = e / %(CP)du;
        int gx = x0 - 1 + int(p %% %(TW2)du), gy = y0 - 1 + int(p / %(TW2)du);
        uint v = 0u;
        if (gx >= 0 && gy >= 0 && gx < W && gy < H && c2 < cpair)
            v = NvxU32(ip + uint64_t((uint(gy * W + gx) * cpair + c2) * 4u)).v;
        smem[((p * %(CS)du) >> 1) + c2] = v;
    }
%(WSTAGE)s
    for (uint e = lid; e < %(NB)du; e += nthr) {
        uint col = e & 15u, nt = e >> 8;
        smh[%(BOFF)du + e] = uint16BitsToFloat16(NvxU16(bp + uint64_t(((zt + nt) * 16u + col) * 2u)).v);
    }
    barrier();
    uint w = gl_SubgroupID, r = w %% %(TH)du, nt = w / %(TH)du;
    coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator> c16;
    coopMatLoad(c16, smh, %(BOFF)du + nt * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
    coopmat<float, %(CM)s, gl_MatrixUseAccumulator> acc = coopmat<float, %(CM)s, gl_MatrixUseAccumulator>(c16);
    [[unroll]] for (uint s = 0u; s < %(S)du; s++) {
        uint t = s / %(CB)du, cb = s %% %(CB)du;
        uint ky = t / 3u, kx = t %% 3u;
        coopmat<float16_t, %(CM)s, gl_MatrixUseA> a;
        coopMatLoad(a, smh, ((r + ky) * %(TW2)du + kx) * %(CS)du + cb * 16u, %(CS)du, gl_CooperativeMatrixLayoutRowMajor);
        coopmat<float16_t, %(CM)s, gl_MatrixUseB> b;
        coopMatLoad(b, smh, %(WOFF)du + (s * %(NT)du + nt) * 256u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
        acc = coopMatMulAdd(a, b, acc);
        %(ROUND)s
    }
    coopMatStore(coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator>(acc), smh,
                 %(STG)du + r * %(TW)du * %(CW)du + nt * 16u, %(CW)du, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    uint16_t h001 = nvx_f2h(uintBitsToFloat(0x3C23D70AU)), h6 = nvx_f2h(uintBitsToFloat(0x40C00000U));
    uint16_t h025 = nvx_f2h(uintBitsToFloat(0x3E800000U));
    uint k001 = uint(h001) | (uint(h001) << 16), k6 = uint(h6) | (uint(h6) << 16), k025 = uint(h025) | (uint(h025) << 16);
    uint n6 = k6 ^ 0x80008000u;
    for (uint e = lid; e < %(NP)du; e += nthr) {
        uint pr = e %% %(CW2)du, q = e / %(CW2)du, px = q %% %(TWP)du, py = q / %(TWP)du;
        int X = x0 / 2 + int(px), Y = y0 / 2 + int(py);
        if (X >= W2 || Y >= H2)
            continue;
        uint sum = 0u;
        [[unroll]] for (uint j = 0u; j < 4u; j++) {
            uint cx = 2u * px + (j & 1u), cy = 2u * py + (j >> 1);
            uint v = smem[((%(STG)du + (cy * %(TW)du + cx) * %(CW)du) >> 1) + pr];
            uint m1 = packFloat2x16(unpackFloat2x16(v) * unpackFloat2x16(k001));
            f16vec2 a1 = unpackFloat2x16(m1), b1 = unpackFloat2x16(v);
            uint m2 = packFloat2x16(max(a1, b1));
            f16vec2 a2 = unpackFloat2x16(m2), b2 = unpackFloat2x16(n6);
            uint m3 = packFloat2x16(max(a2, b2));
            f16vec2 a3 = unpackFloat2x16(m3), b3 = unpackFloat2x16(k6);
            uint m4 = packFloat2x16(min(a3, b3));
            sum = j == 0u ? m4 : packFloat2x16(unpackFloat2x16(sum) + unpackFloat2x16(m4));
        }
        uint o = packFloat2x16(unpackFloat2x16(sum) * unpackFloat2x16(k025));
        NvxU32(op + uint64_t((uint(Y * (W / 2) + X) * cout2 + zt * 8u + pr) * 4u)).v = o;
    }
}
'''
    if wl == 'frag':
        # NVIDIA's fragment layout per 8-channel group z and step s (see conv2)
        g['WSTAGE'] = ('''    for (uint e = lid; e < %(NW)du; e += nthr) {
        uint kh = e & 1u, n = (e >> 1) & 15u, q = e >> 5, nt = q %% %(NT)du, s = q / %(NT)du;
        uint z = (zt + nt) * 2u + (n >> 3);
        smv[(%(WOFF)du + (s * %(NT)du + nt) * 256u + n * 16u + kh * 8u) >> 3] =
            NvxU32x4(wp + uint64_t((z * %(S)du * 128u + s * 128u + kh * 64u + (n & 7u) * 8u) * 2u)).v;
    }''' % g)
    elif wl == 'oihw18':
        # block1 conv0: W[oc][tap][18 channels] (u32 index oc*81 + 9*tap + pair); per tap a K=16 step (channels
        # 0-15) and a K=8 step whose only nonzero K are channels 16,17 (the PTX's m16n8k8 with lanes k>=2 zero)
        g['WSTAGE'] = ('''    for (uint e = lid; e < %(S)du * %(NT)du * 128u; e += nthr) {
        uint kp = e & 7u, n = (e >> 3) & 15u, q = e >> 7, nt = q %% %(NT)du, s = q / %(NT)du;
        uint oc = (zt + nt) * 16u + n, t = s >> 1;
        uint v = 0u;
        if ((s & 1u) == 0u)
            v = NvxU32(wp + uint64_t((oc * 81u + t * 9u + kp) * 4u)).v;
        else if (kp == 0u)
            v = NvxU32(wp + uint64_t((oc * 81u + t * 9u + 8u) * 4u)).v;
        smem[((%(WOFF)du + (s * %(NT)du + nt) * 256u + n * 16u) >> 1) + kp] = v;
    }''' % g)
    else:
        raise ValueError(wl)
    g['CP'] = CIN // 2
    g['CW2'] = CW // 2
    g['TWP'] = TW // 2
    g['NIN'] = (TW + 2) * (TH + 2) * (CIN // 8)
    if persistent:
        # persistent variant: `persistent` workgroups (a multiple of the NTALL/NT channel groups) stage the weights
        # and bias once and loop over the conv tiles of their channel group
        g['NGRPP'] = NTALL // NT
        assert persistent % g['NGRPP'] == 0
        persistent = fit_persistent(persistent, g['NGRPP'])
        body = _persist(body, '    // input tile with halo', '%(WSTAGE)s', '    barrier();\n    uint w = gl_SubgroupID')
        return header(TOT) + body % g, ['t1_block 32 %d 1' % warps, 't1_fixed %d 1 1' % persistent]
    return header(TOT) + body % g, ['t1_block 32 %d 1' % warps]


def _persist(body, in_start, in_end, compute_start):
    """rewrite a tile kernel template into a persistent one: weights/bias staged once, tiles in a grid-stride loop,
    the next tile's input fetched into registers while the current tile computes (software pipelining)"""
    import re
    head = '    int x0 = int(gl_WorkGroupID.x) * %(TW)d, y0 = int(gl_WorkGroupID.y) * %(TH)d;\n    uint zt = gl_WorkGroupID.z * %(NT)du;\n'
    assert head in body
    body = body.replace(head, '    uint ngrp_ = %(NGRPP)du, zt = (gl_WorkGroupID.x %% ngrp_) * %(NT)du;\n'
                        '    uint ntx_ = (uint(W) + %(TW)du - 1u) / %(TW)du, nty_ = (uint(H) + %(TH)du - 1u) / %(TH)du;\n', 1)
    a = body.index(in_start)
    b = body.index(in_end)
    tile_in = body[a:b]
    body = body[:a] + body[b:]
    # tile_in: "for (uint e = lid; e < LIMIT; e += nthr) { decls; T v = Z; if (C) v = LOAD; DST = v; }"
    m = re.search(r'for \(uint e = lid; e < ([^;]+); e \+= nthr\) \{\n(.*?)\n(\s*)(uint|uvec4) (\w+) = ([^;]+);\n\s*if \((.*?)\)\n\s*\5 = ([^;]+);\n\s*(.+?) = \5;\n\s*\}', tile_in, re.S)
    assert m, tile_in
    limit, decls, _, typ, var, zero, cond, load, dst = m.groups()
    npf = '((%s) + 255u) / 256u' % limit
    fetch = ('    {\n    int x0 = int(ntt_ %% ntx_) * %(TW)d, y0 = int(ntt_ / ntx_) * %(TH)d;\n'
             '    [[unroll]] for (uint i_ = 0u; i_ < NPF_; i_++) {\n        uint e = lid + 256u * i_;\n' + decls + '\n'
             '        %s %s = %s;\n        if (e < %s && %s)\n            %s = %s;\n        pf_[i_] = %s;\n    }\n    }\n'
             % (typ, var, zero, limit, cond, var, load, var))
    store = ('    [[unroll]] for (uint i_ = 0u; i_ < NPF_; i_++) {\n        uint e = lid + 256u * i_;\n' + decls + '\n'
             '        if (e < %s)\n            %s = pf_[i_];\n    }\n' % (limit, dst))
    c = body.index(compute_start)
    end = body.rindex('}')
    loop = ('    const uint NPF_ = ' + npf + ';\n'
            '    %s pf_[NPF_];\n' % typ +
            '    uint step_ = gl_NumWorkGroups.x / ngrp_, ntt_ = gl_WorkGroupID.x / ngrp_;\n'
            '    if (ntt_ < ntx_ * nty_) {\n' + fetch + '    }\n'
            '    for (uint tt_ = gl_WorkGroupID.x / ngrp_; tt_ < ntx_ * nty_; tt_ += step_) {\n'
            '    int x0 = int(tt_ %% ntx_) * %(TW)d, y0 = int(tt_ / ntx_) * %(TH)d;\n'
            '    barrier();\n' + store +
            '    ntt_ = tt_ + step_;\n'
            '    if (ntt_ < ntx_ * nty_) {\n' + fetch + '    }\n' + body[c:end] + '    }\n')
    return body[:c] + loop + body[end:]


UPSAMPLE = r'''
    // bilinear 2x upsample (align_corners = false) of the half-resolution input into the tile, one invocation per
    // tile pixel, all channel pairs: per pair exactly the PTX's element arithmetic (_up2x.py)
    for (uint p = lid; p < %(NPIX)du; p += nthr) {
        uint px_ = p %% %(TW2)du, py_ = p / %(TW2)du;
        uint r45_ = uint(x0 - 1) + px_, r47_ = uint(y0 - 1) + py_;
        bool oob_ = (int(r45_ | r47_) < 0) || (int(r45_) >= Wo) || (int(r47_) >= Ho);
        uint dst = (p * %(CS)du) >> 1;
        if (oob_) {
            precise float z_ = uintBitsToFloat(0x00000000U);
            uint16_t zh_ = nvx_f2h(z_);
            uint zz_ = uint(zh_) | (uint(zh_) << 16);
            for (uint c = 0u; c < %(CP)du; c++) smem[dst + c] = zz_;
            continue;
        }
        precise float f6_ = float(int(r45_));
        precise float f7_ = f6_ + uintBitsToFloat(0xBF000000U);
        precise float f8_ = f7_ * uintBitsToFloat(0x3F000000U);
        precise float f9_ = float(int(r47_));
        precise float f10_ = f9_ + uintBitsToFloat(0xBF000000U);
        precise float f11_ = f10_ * uintBitsToFloat(0x3F000000U);
        uint r252_ = uint(trunc(f8_));
        uint r253_ = uint(trunc(f11_));
        precise float f12_ = f8_ + uintBitsToFloat(0x3F800000U);
        uint r254_ = uint(trunc(f12_));
        bool p13_ = (r254_) >= (Wi);
        uint r255_ = p13_ ? Wi - 1u : r252_;
        uint r256_ = p13_ ? Wi - 1u : r254_;
        precise float f13_ = f11_ + uintBitsToFloat(0x3F800000U);
        uint r257_ = uint(trunc(f13_));
        bool p14_ = (r257_) >= (Hi);
        uint r258_ = p14_ ? Hi - 1u : r253_;
        uint r259_ = p14_ ? Hi - 1u : r257_;
        bool p15_ = (f8_) < (uintBitsToFloat(0x00000000U));
        uint r260_ = p15_ ? 0u : r255_;
        uint r261_ = p15_ ? 0u : r256_;
        bool p16_ = (f11_) < (uintBitsToFloat(0x00000000U));
        uint r262_ = p16_ ? 0u : r258_;
        uint r263_ = p16_ ? 0u : r259_;
        precise float f14_ = float(r262_);
        precise float f2_ = f11_ - f14_;
        uint16_t s2_ = nvx_f2h(f2_);
        uint r221_ = uint(s2_) | (uint(s2_) << 16);
        precise float f15_ = float(r260_);
        precise float f3_ = f8_ - f15_;
        uint16_t s3_ = nvx_f2h(f3_);
        uint r222_ = uint(s3_) | (uint(s3_) << 16);
        precise float f16_ = f14_ - f11_;
        precise float f4_ = f16_ + uintBitsToFloat(0x3F800000U);
        uint16_t s4_ = nvx_f2h(f4_);
        uint r223_ = uint(s4_) | (uint(s4_) << 16);
        precise float f17_ = f15_ - f8_;
        precise float f5_ = f17_ + uintBitsToFloat(0x3F800000U);
        uint16_t s5_ = nvx_f2h(f5_);
        uint r224_ = uint(s5_) | (uint(s5_) << 16);
        uint r264_ = r262_ * Wi;
        uint r269_ = r263_ * Wi;
        uint b00_ = (r264_ + r260_) * cpi, b01_ = (r264_ + r261_) * cpi;
        uint b10_ = (r269_ + r260_) * cpi, b11_ = (r269_ + r261_) * cpi;
        for (uint h_ = 0u; h_ < %(CQ)du; h_++) {
            uvec4 q00_ = NvxU32x4a4(ip + uint64_t((b00_ + 4u * h_) * 4u)).v;
            uvec4 q01_ = NvxU32x4a4(ip + uint64_t((b01_ + 4u * h_) * 4u)).v;
            uvec4 q10_ = NvxU32x4a4(ip + uint64_t((b10_ + 4u * h_) * 4u)).v;
            uvec4 q11_ = NvxU32x4a4(ip + uint64_t((b11_ + 4u * h_) * 4u)).v;
            uvec4 o_;
            for (uint c_ = 0u; c_ < 4u; c_++) {
                uint r227_ = q00_[c_];
                uint r230_ = q01_[c_];
                uint r239_ = q10_[c_];
                uint r242_ = q11_[c_];
                uint r225_ = packFloat2x16(unpackFloat2x16(r224_) * unpackFloat2x16(r227_));
                uint r228_ = packFloat2x16(unpackFloat2x16(r222_) * unpackFloat2x16(r230_));
                uint r231_ = packFloat2x16(unpackFloat2x16(r225_) + unpackFloat2x16(r228_));
                uint r234_ = packFloat2x16(unpackFloat2x16(r223_) * unpackFloat2x16(r231_));
                uint r237_ = packFloat2x16(unpackFloat2x16(r224_) * unpackFloat2x16(r239_));
                uint r240_ = packFloat2x16(unpackFloat2x16(r222_) * unpackFloat2x16(r242_));
                uint r243_ = packFloat2x16(unpackFloat2x16(r237_) + unpackFloat2x16(r240_));
                uint r246_ = packFloat2x16(unpackFloat2x16(r221_) * unpackFloat2x16(r243_));
                o_[c_] = packFloat2x16(unpackFloat2x16(r234_) + unpackFloat2x16(r246_));
            }
            smv[(p * %(CS)du + 8u * h_) >> 3] = o_;
        }
    }
'''

MMA = r'''
    uint w = gl_SubgroupID, r = w %% %(TH)du, nt = w / %(TH)du;
    coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator> c16;
    coopMatLoad(c16, smh, %(BOFF)du + nt * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
    coopmat<float, %(CM)s, gl_MatrixUseAccumulator> acc = coopmat<float, %(CM)s, gl_MatrixUseAccumulator>(c16);
    [[unroll]] for (uint s = 0u; s < %(S)du; s++) {
        uint t = s / %(CB)du, cb = s %% %(CB)du;
        uint ky = t / 3u, kx = t %% 3u;
        coopmat<float16_t, %(CM)s, gl_MatrixUseA> a;
        coopMatLoad(a, smh, ((r + ky) * %(TW2)du + kx) * %(CS)du + cb * 16u, %(CS)du, gl_CooperativeMatrixLayoutRowMajor);
        coopmat<float16_t, %(CM)s, gl_MatrixUseB> b;
        coopMatLoad(b, smh, %(WOFF)du + (s * %(NT)du + nt) * 256u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
        acc = coopMatMulAdd(a, b, acc);
        %(ROUND)s
    }
    uint stg = %(STG)du + w * 256u;
    coopMatStore(coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator>(acc), smh, stg, 16u, gl_CooperativeMatrixLayoutRowMajor);
    subgroupMemoryBarrierShared(); subgroupBarrier();
'''


def _layout(CB, TH, NT, warps, TW=16):
    S = 9 * CB
    CIN = 16 * CB
    CS = CIN + 8
    TIN = (TW + 2) * (TH + 2) * CS
    WOFF = TIN
    BOFF = WOFF + S * NT * 256
    STG = BOFF + NT * 256
    TOT = STG + warps * 256
    assert TIN % 8 == 0
    return dict(TW=TW, TW2=TW + 2, TH=TH, CB=CB, CIN=CIN, CP=CIN // 2, CQ=CIN // 8, S=S, CS=CS, NT=NT, WOFF=WOFF,
                BOFF=BOFF, STG=STG, TOT=TOT, CM=CM, NPIX=(TW + 2) * (TH + 2), ROUND=ROUND % {'m': 'acc'})


def bot0(CB, CT, NTENS, TH, NT, warps, clamp_from, persistent=None):
    """custom_block{0,1}_conv_bot0_hf_kernel: 2x upsampled input, conv 16*CB -> NTENS tensors of CT channels,
         v = max(v * 0.01, v); clamp to [-6, 6] for tensors j >= clamp_from (from the PTX's ctaid.z test);
         tensor j at (y * Wo + x) * (CT/2) + pair.   Struct params: 0 w, 8 bias, 32 in, 44 Hi, 48 Wi, 52 Ci,
         56/64/72 out tensors, 84 Ho, 88 Wo.  Workgroup: NT 16-channel tiles starting at gl_WorkGroupID.z * NT."""
    g = _layout(CB, TH, NT, warps)
    g.update(NTT=CT // 16, CT2=CT // 2, CLAMPJ=clamp_from, UPSAMPLE='', NW=g['S'] * NT * 32, NB=NT * 256)
    g['UPSAMPLE'] = UPSAMPLE % g
    body = r'''
void main() {
    uint64_t wp = nvx_p64(0u), bp = nvx_p64(8u), ip = nvx_p64(32u);
    uint Hi = nvx_p32(44u), Wi = nvx_p32(48u), cpi = nvx_p32(52u) >> 1;
    int Ho = int(nvx_p32(84u)), Wo = int(nvx_p32(88u));
    uint lid = gl_LocalInvocationIndex, nthr = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    int x0 = int(gl_WorkGroupID.x) * %(TW)d, y0 = int(gl_WorkGroupID.y) * %(TH)d;
    uint zt = gl_WorkGroupID.z * %(NT)du;
%(UPSAMPLE)s
    for (uint e = lid; e < %(NW)du; e += nthr) {
        uint kh = e & 1u, n = (e >> 1) & 15u, q = e >> 5, nt = q %% %(NT)du, s = q / %(NT)du;
        uint z = (zt + nt) * 2u + (n >> 3);
        smv[(%(WOFF)du + (s * %(NT)du + nt) * 256u + n * 16u + kh * 8u) >> 3] =
            NvxU32x4(wp + uint64_t((z * %(S)du * 128u + s * 128u + kh * 64u + (n & 7u) * 8u) * 2u)).v;
    }
    for (uint e = lid; e < %(NB)du; e += nthr) {
        uint col = e & 15u, nt = e >> 8;
        smh[%(BOFF)du + e] = uint16BitsToFloat16(NvxU16(bp + uint64_t(((zt + nt) * 16u + col) * 2u)).v);
    }
    barrier();
%(MMA)s
    uint T = zt + nt, j = T / %(NTT)du, tt = T %% %(NTT)du;
    uint64_t op = nvx_p64(56u + 8u * j);
    bool doclamp = j >= %(CLAMPJ)du;
    uint16_t h001 = nvx_f2h(uintBitsToFloat(0x3C23D70AU)), h6 = nvx_f2h(uintBitsToFloat(0x40C00000U));
    uint k001 = uint(h001) | (uint(h001) << 16), k6 = uint(h6) | (uint(h6) << 16);
    [[unroll]] for (uint q = 0u; q < 4u; q++) {
        uint e = gl_SubgroupInvocationID + 32u * q;
        uint i = e >> 3, pr = e & 7u;
        int gx = x0 + int(i), gy = y0 + int(r);
        if (gx >= Wo || gy >= Ho)
            continue;
        uint v = smem[(stg >> 1) + i * 8u + pr];
        uint m1 = packFloat2x16(unpackFloat2x16(v) * unpackFloat2x16(k001));
        f16vec2 a1 = unpackFloat2x16(m1), b1 = unpackFloat2x16(v);
        // unclamped slices store the leaky result: the T0 NaN rule must hold there
        uint m2 = doclamp ? packFloat2x16(max(a1, b1)) : packFloat2x16(f16vec2(nvx_hmax(a1.x, b1.x), nvx_hmax(a1.y, b1.y)));
        if (doclamp) {
            uint n6 = k6 ^ 0x80008000u;
            f16vec2 a2 = unpackFloat2x16(m2), b2 = unpackFloat2x16(n6);
            uint m3 = packFloat2x16(max(a2, b2));
            f16vec2 a3 = unpackFloat2x16(m3), b3 = unpackFloat2x16(k6);
            m2 = packFloat2x16(min(a3, b3));
        }
        NvxU32(op + uint64_t((uint(gy * Wo + gx) * %(CT2)du + tt * 8u + pr) * 4u)).v = m2;
    }
}
'''
    g['MMA'] = MMA % g
    if persistent:
        # persistent variant (weights stationary): `persistent` workgroups, NGRP channel groups; a workgroup stages its
        # group's weights and bias once and loops over the conv tiles (same per-tile code: upsample, MMA, epilogue)
        g['NGRP'] = NTENS * CT // 16 // NT
        assert persistent % g['NGRP'] == 0
        persistent = fit_persistent(persistent, g['NGRP'])
        head = ('    int x0 = int(gl_WorkGroupID.x) * %(TW)d, y0 = int(gl_WorkGroupID.y) * %(TH)d;\n'
                '    uint zt = gl_WorkGroupID.z * %(NT)du;\n%(UPSAMPLE)s\n')
        assert head in body
        body = body.replace(head, '    uint ngrp_ = %(NGRP)du, zt = (gl_WorkGroupID.x %% ngrp_) * %(NT)du;\n'
                                  '    uint ntx_ = (uint(Wo) + %(TW)du - 1u) / %(TW)du, nty_ = (uint(Ho) + %(TH)du - 1u) / %(TH)du;\n', 1)
        mark = '    barrier();\n%(MMA)s\n'
        assert mark in body
        body = body.replace(mark, '    for (uint tt_ = gl_WorkGroupID.x / ngrp_; tt_ < ntx_ * nty_; tt_ += gl_NumWorkGroups.x / ngrp_) {\n'
                                  '    int x0 = int(tt_ %% ntx_) * %(TW)d, y0 = int(tt_ / ntx_) * %(TH)d;\n'
                                  '    barrier();\n%(UPSAMPLE)s\n    barrier();\n%(MMA)s\n', 1)
        end = body.rindex('}')
        body = body[:end] + '    }\n' + body[end:]
        return header(g['TOT']) + body % g, ['t1_block 32 %d 1' % warps, 't1_fixed %d 1 1' % persistent]
    return header(g['TOT']) + body % g, ['t1_block 32 %d 1' % warps]


def bot1(CB, TH, warps, variant, persistent=None):
    """custom_block{0,1}_conv_bot1_hf_kernel: per slice z = gl_WorkGroupID.z (input tensor z, 2x upsampled),
       conv 16*CB -> 8 channels (one 8-column group, columns 8..15 of the WMMA unused), the slice keeps channels
       [base_z, base_z + cnt_z):  variant 'b0': z == 0 ? d : clamp(d, -6, 6)  -> out_z[(y*Wo+x)*cnt + c - base]
                                  variant 'b1': z == 0 ? d + 2*res : clamp(d) + 1*res -> out[(y*Wo+x)*8 + c]"""
    g = _layout(CB, TH, 1, warps)
    g.update(NW=g['S'] * 32, NB=256)
    g['UPSAMPLE'] = UPSAMPLE % g
    if variant == 'b0':
        params = r'''
    uint64_t wp = nvx_p64(0u), bp = nvx_p64(8u), ip = nvx_p64(32u + 8u * z);
    uint Hi = nvx_p32(60u), Wi = nvx_p32(64u), cpi = nvx_p32(68u) >> 1;
    int Ho = int(nvx_p32(84u + 24u * z)), Wo = int(nvx_p32(88u + 24u * z));
    uint cnt = nvx_p32(92u + 24u * z);
    uint base = z == 0u ? 0u : (z == 1u ? nvx_p32(92u) : nvx_p32(92u) + nvx_p32(116u));
    uint64_t op = nvx_p64(72u + 24u * z);'''
        store = r'''
            uint16_t d = uint16_t((c & 1u) != 0u ? (v >> 16) : (v & 0xffffu));
            uint16_t o;
            if (z == 0u) {
                o = d;
            } else {
                uint16_t n6_ = float16BitsToUint16((-(uint16BitsToFloat16(h6))));
                uint16_t mx_ = float16BitsToUint16(max(uint16BitsToFloat16(d), uint16BitsToFloat16(n6_)));
                o = float16BitsToUint16(min(uint16BitsToFloat16(mx_), uint16BitsToFloat16(h6)));
            }
            NvxU16(op + uint64_t((uint(gy * Wo + gx) * cnt + (c - base)) * 2u)).v = o;'''
    else:
        params = r'''
    uint64_t wp = nvx_p64(0u), bp = nvx_p64(8u), ip = nvx_p64(32u + 8u * z), rp = nvx_p64(56u + 8u * z);
    uint Hi = nvx_p32(84u), Wi = nvx_p32(88u), cpi = nvx_p32(92u) >> 1;
    int Ho = int(nvx_p32(108u + 16u * z)), Wo = int(nvx_p32(112u + 16u * z));
    uint cnt = nvx_p32(116u + 16u * z);
    uint base = z == 0u ? 0u : (z == 1u ? nvx_p32(116u) : nvx_p32(116u) + nvx_p32(132u));
    uint64_t op = nvx_p64(96u);'''
        store = r'''
            uint16_t d = uint16_t((c & 1u) != 0u ? (v >> 16) : (v & 0xffffu));
            uint16_t rs = uint16_t(uint(NvxU16(rp + uint64_t((uint(gy * Wo + gx) * cnt + (c - base)) * 2u)).v));
            uint16_t o;
            if (z == 0u) {
                uint16_t k2_ = nvx_f2h(uintBitsToFloat(0x40000000U));
                uint16_t m_ = float16BitsToUint16((uint16BitsToFloat16(k2_) * uint16BitsToFloat16(rs)));
                o = float16BitsToUint16((uint16BitsToFloat16(d) + uint16BitsToFloat16(m_)));
            } else {
                uint16_t k1_ = nvx_f2h(uintBitsToFloat(0x3F800000U));
                uint16_t n6_ = float16BitsToUint16((-(uint16BitsToFloat16(h6))));
                uint16_t mx_ = float16BitsToUint16(max(uint16BitsToFloat16(d), uint16BitsToFloat16(n6_)));
                uint16_t cl_ = float16BitsToUint16(min(uint16BitsToFloat16(mx_), uint16BitsToFloat16(h6)));
                uint16_t m_ = float16BitsToUint16((uint16BitsToFloat16(k1_) * uint16BitsToFloat16(rs)));
                o = float16BitsToUint16((uint16BitsToFloat16(cl_) + uint16BitsToFloat16(m_)));
            }
            NvxU16(op + uint64_t((uint(gy * Wo + gx) * 8u + c) * 2u)).v = o;'''
    g['PARAMS'] = params
    g['STORE'] = store
    body = r'''
void main() {
    uint z = gl_WorkGroupID.z;
%(PARAMS)s
    uint lid = gl_LocalInvocationIndex, nthr = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    int x0 = int(gl_WorkGroupID.x) * %(TW)d, y0 = int(gl_WorkGroupID.y) * %(TH)d;
%(UPSAMPLE)s
    // the 8 output channels' weights (one fragment group) in columns 0..7, zeros in 8..15
    for (uint e = lid; e < %(NW)du; e += nthr) {
        uint kh = e & 1u, n = (e >> 1) & 15u, s = e >> 5;
        uvec4 v = uvec4(0u);
        if (n < 8u)
            v = NvxU32x4(wp + uint64_t((s * 128u + kh * 64u + n * 8u) * 2u)).v;
        smv[(%(WOFF)du + s * 256u + n * 16u + kh * 8u) >> 3] = v;
    }
    for (uint e = lid; e < %(NB)du; e += nthr) {
        uint col = e & 15u;
        smh[%(BOFF)du + e] = col < 8u ? uint16BitsToFloat16(NvxU16(bp + uint64_t(col * 2u)).v) : float16_t(0.0);
    }
    barrier();
    uint nt = 0u;
%(MMA)s
    uint16_t h6 = nvx_f2h(uintBitsToFloat(0x40C00000U));
    // the slice's channels of this warp's 16 pixels (one image row)
    for (uint e = gl_SubgroupInvocationID; e < 16u * cnt; e += 32u) {
        uint i = e / cnt, c = base + e %% cnt;
        int gx = x0 + int(i), gy = y0 + int(r);
        if (gx < Wo && gy < Ho) {
            uint v = smem[(stg >> 1) + i * 8u + (c >> 1)];
%(STORE)s
        }
    }
}
'''
    g['MMA'] = (MMA % g).replace('uint w = gl_SubgroupID, r = w %% %(TH)du, nt = w / %(TH)du;' % g,
                                 'uint w = gl_SubgroupID, r = w;')
    if persistent:
        # persistent variant (weights stationary): the weights (the same for the 3 slices) and bias are staged once
        # per workgroup; the workgroups loop over all (slice, tile) pairs with the same per-tile code
        persistent = fit_persistent(persistent, 1)
        head = ('void main() {\n    uint z = gl_WorkGroupID.z;\n%(PARAMS)s\n'
                '    uint lid = gl_LocalInvocationIndex, nthr = gl_WorkGroupSize.x * gl_WorkGroupSize.y;\n'
                '    int x0 = int(gl_WorkGroupID.x) * %(TW)d, y0 = int(gl_WorkGroupID.y) * %(TH)d;\n%(UPSAMPLE)s\n')
        assert head in body
        dims = re.search(r'int Ho = int\(nvx_p32\((\d+)u \+ (\d+)u \* z\)\), Wo = int\(nvx_p32\((\d+)u \+ \d+u \* z\)\);', params)
        ho0, hs, wo0 = int(dims.group(1)), int(dims.group(2)), int(dims.group(3))
        cnt = '\n'.join('    uint ntx%d_ = (nvx_p32(%du) + 15u) / 16u, nt%d_ = ntx%d_ * ((nvx_p32(%du) + %du) / %du);'
                        % (z, wo0 + hs * z, z, z, ho0 + hs * z, TH - 1, TH) for z in range(3))
        body = body.replace(head, 'void main() {\n'
                            '    uint lid = gl_LocalInvocationIndex, nthr = gl_WorkGroupSize.x * gl_WorkGroupSize.y;\n'
                            '    uint64_t wp = nvx_p64(0u), bp = nvx_p64(8u);\n' + cnt.replace('%', '%%') + '\n'
                            '    uint tot_ = nt0_ + nt1_ + nt2_;\n', 1)
        mark = '    barrier();\n    uint nt = 0u;\n%(MMA)s'
        assert mark in body
        body = body.replace(mark, '    for (uint it_ = gl_WorkGroupID.x; it_ < tot_; it_ += gl_NumWorkGroups.x) {\n'
                            '    uint z = it_ < nt0_ ? 0u : (it_ < nt0_ + nt1_ ? 1u : 2u);\n'
                            '    uint tt_ = it_ - (z == 0u ? 0u : (z == 1u ? nt0_ : nt0_ + nt1_));\n'
                            '    uint ntx_ = z == 0u ? ntx0_ : (z == 1u ? ntx1_ : ntx2_);\n'
                            '%(PARAMS)s\n'
                            '    int x0 = int(tt_ %% ntx_) * %(TW)d, y0 = int(tt_ / ntx_) * %(TH)d;\n'
                            '    barrier();\n%(UPSAMPLE)s\n    barrier();\n    uint nt = 0u;\n%(MMA)s', 1)
        end = body.rindex('}')
        body = body[:end] + '    }\n' + body[end:]
        return header(g['TOT']) + body % g, ['t1_block 32 %d 1' % warps, 't1_fixed %d 1 1' % persistent]
    return header(g['TOT']) + body % g, ['t1_block 32 %d 1' % warps]
