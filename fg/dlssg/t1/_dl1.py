# T1: exact-mode kernels for DL1Net's k_conv_fp16_nhwc instances of nvngx_dlssg
# 310.9.1 (NVIDIA's generic NHWC conv template, specialised per layer).
#
# Semantics read from the PTX of the first conv (fcda7303ddc8; the other instances are checked one by one):
#   chain per output element: acc_0 = bias[oc]; for chunk c (16 input channels): for kx: for ky: one m16n8k16 step
#   (f16 accumulate = RNE to f16 after every step);  A = input pixel (x+kx-1, y+ky-1) channels 16c..16c+15 (0 outside
#   the image), weights per 8-channel group g in 128-half blocks blk = ((g*3+ky)*3+kx)*NCH + c, element (k, n) of a
#   block at 16n + 4*((k&7)>>1) + (k&1) + 2*(k>=8);  bias of oc = 8g+j at half BIAS + 128g + j;
#   epilogue per element: m = 0.01h * v; out = max(v, m)  (leaky ReLU, NaN rules of PTX max.f16)
#   output NHWC with C_out channels.
# T1: one 16x16x16 WMMA per step for two groups (16 output channels), A/B from LDS with coopMatLoad, the input chunk and
# its weights staged per K chunk like the PTX does, 16x16-pixel tiles.
import _conv

CM = _conv.CM
ROUND = _conv.ROUND


def kconv(NCH, NT, BIAS, TH=16, R=4, warps=8, act='leaky', hooks=None):
    """params: 0 weights, 8 input, 16 W, 20 H (image), 32 output, 40 W, 44 H (output; equal here).
       NCH input chunks of 16 channels (C_in = 16*NCH, input pixel stride C_in), NT 16-channel output tiles."""
    TW = 16
    assert (TH // R) * NT == warps
    CIN = 16 * NCH
    COUT = 16 * NT
    CS = 24                              # 16 channels of the current chunk + 8 halves padding per pixel
    TIN = (TW + 2) * (TH + 2) * CS       # current input chunk with halo
    WOFF = TIN                           # weights of the chunk: [tap 9 (kx*3+ky)][nt][16 n][16 k]
    BOFF = WOFF + 9 * NT * 256
    TOT1 = BOFF + NT * 256
    STG = 0                              # after the last chunk: [TH][16][COUT] results (reuses the tile)
    TOT = max(TOT1, TH * TW * COUT)
    g = dict(TW=TW, TW2=TW + 2, TH=TH, TH2=TH + 2, R=R, NCH=NCH, NT=NT, CIN=CIN, COUT=COUT, CS=CS, WOFF=WOFF,
             BOFF=BOFF, BIAS=BIAS, CM=CM, NG=2 * NT, NROW=TH // R,
             NIN=(TW + 2) * (TH + 2) * 2, NW=9 * NT * 32, NB=NT * 256, NOUT=TH * TW * (COUT // 2),
             ROUND0=ROUND % {'m': 'acc[i]'})
    body = r'''
void main() {
    uint64_t wp = nvx_p64(0u), ip = nvx_p64(8u), op = nvx_p64(32u);
    int W = int(nvx_p32(16u)), H = int(nvx_p32(20u)), Wo = int(nvx_p32(40u)), Ho = int(nvx_p32(44u));
    uint lid = gl_LocalInvocationIndex, nthr = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    int x0 = int(gl_WorkGroupID.x) * %(TW)d, y0 = int(gl_WorkGroupID.y) * %(TH)d;
    uint w = gl_SubgroupID, rg = w %% %(NROW)du, nt = w / %(NROW)du;
    // bias rows (C of the first step): [nt][16][16]
    for (uint e = lid; e < %(NB)du; e += nthr) {
        uint col = e & 15u, t = e >> 8, gg = t * 2u + (col >> 3);
        smh[%(BOFF)du + e] = uint16BitsToFloat16(NvxU16(wp + uint64_t((%(BIAS)du + 128u * gg + (col & 7u)) * 2u)).v);
    }
    barrier();
    coopmat<float, %(CM)s, gl_MatrixUseAccumulator> acc[%(R)d];
    {
        coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator> c16;
        coopMatLoad(c16, smh, %(BOFF)du + nt * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        [[unroll]] for (uint i = 0u; i < %(R)du; i++) acc[i] = coopmat<float, %(CM)s, gl_MatrixUseAccumulator>(c16);
    }
    for (uint c = 0u; c < %(NCH)du; c++) {
        barrier();
        // input chunk with a 1-pixel halo: 16 channels = two 16-byte halves per pixel; outside the image f16 +0
        for (uint e = lid; e < %(NIN)du; e += nthr) {
            uint hh = e & 1u, p = e >> 1;
            int gx = x0 - 1 + int(p %% %(TW2)du), gy = y0 - 1 + int(p / %(TW2)du);
            uvec4 v = uvec4(0u);
            if (gx >= 0 && gy >= 0 && gx < W && gy < H)
                v = NvxU32x4a4(ip + uint64_t((uint(gy * W + gx) * %(CIN)du + c * 16u + hh * 8u) * 2u)).v;
            smv[(p * %(CS)du + hh * 8u) >> 3] = v;
        }
        // weights of this chunk, unswizzled to column-major 16x16 per (tap, nt): pairs of halves (k, k+1)
        for (uint e = lid; e < %(NW)du * 4u; e += nthr) {
            uint kp = e & 7u, n = (e >> 3) & 15u, q = e >> 7, t = q %% %(NT)du, tap = q / %(NT)du;
            uint kx = tap / 3u, ky = tap %% 3u, gg = t * 2u + (n >> 3), k = kp * 2u;
            uint blk = ((gg * 3u + ky) * 3u + kx) * %(NCH)du + c;
            uint pos = 16u * (n & 7u) + 4u * ((k & 7u) >> 1) + ((k & 8u) != 0u ? 2u : 0u);
            smem[((%(WOFF)du + (tap * %(NT)du + t) * 256u + n * 16u) >> 1) + kp] =
                NvxU32(wp + uint64_t((blk * 128u + pos) * 2u)).v;
        }
        barrier();
        [[unroll]] for (uint kx = 0u; kx < 3u; kx++) {
            [[unroll]] for (uint ky = 0u; ky < 3u; ky++) {
                coopmat<float16_t, %(CM)s, gl_MatrixUseB> b;
                coopMatLoad(b, smh, %(WOFF)du + ((kx * 3u + ky) * %(NT)du + nt) * 256u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
                [[unroll]] for (uint i = 0u; i < %(R)du; i++) {
                    uint row = rg * %(R)du + i;
                    coopmat<float16_t, %(CM)s, gl_MatrixUseA> a;
                    coopMatLoad(a, smh, ((row + ky) * %(TW2)du + kx) * %(CS)du, %(CS)du, gl_CooperativeMatrixLayoutRowMajor);
                    acc[i] = coopMatMulAdd(a, b, acc[i]);
                    %(ROUND0)s
                }
            }
        }
    }
    barrier();
    [[unroll]] for (uint i = 0u; i < %(R)du; i++) {
        uint row = rg * %(R)du + i;
        coopMatStore(coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator>(acc[i]), smh,
                     row * %(TW)du * %(COUT)du + nt * 16u, %(COUT)du, gl_CooperativeMatrixLayoutRowMajor);
    }
    barrier();
    uint16_t h001 = nvx_f2h(uintBitsToFloat(0x3C23D70AU));
    for (uint e = lid; e < %(NOUT)du; e += nthr) {
        uint pr = e %% %(COUT2)du, q = e / %(COUT2)du, px = q %% %(TW)du, py = q / %(TW)du;
        int gx = x0 + int(px), gy = y0 + int(py);
        if (gx >= Wo || gy >= Ho)
            continue;
        uint v = smem[(q * %(COUT)du >> 1) + pr];
        uint16_t lo = uint16_t(v & 0xffffu), hi = uint16_t(v >> 16);
        uint16_t ml = float16BitsToUint16((uint16BitsToFloat16(h001) * uint16BitsToFloat16(lo)));
        uint16_t mh = float16BitsToUint16((uint16BitsToFloat16(h001) * uint16BitsToFloat16(hi)));
        uint16_t ol = float16BitsToUint16((nvx_hmax(uint16BitsToFloat16(lo), uint16BitsToFloat16(ml))));
        uint16_t oh = float16BitsToUint16((nvx_hmax(uint16BitsToFloat16(hi), uint16BitsToFloat16(mh))));
        NvxU32(op + uint64_t((uint(gy * Wo + gx) * %(COUT2)du + pr) * 4u)).v = uint(ol) | (uint(oh) << 16);%(EPI_STORE)s
    }%(POST)s
}
'''
    g['COUT2'] = COUT // 2
    hk = hooks or {}
    g['EPI_STORE'] = hk.get('epi_store', '') % dict(g, LO='ol', HI='oh', CWX=COUT, ZT='0u')
    g['POST'] = hk.get('post', '') % dict(g, CWX=COUT, CW2X=COUT // 2, ZT='0u', NPOOL=(TW // 2) * (TH // 2) * (COUT // 2),
                                          TW2H=TW // 2)
    return _conv.header(TOT) + body % g, ['t1_block 32 %d 1' % warps]


def kconv_g(KS, NCH, NT, NTALL, BIAS, order, TH=16, R=4, warps=8, act='leaky', CINPIX=None, hooks=None):
    """Generalised k_conv_fp16_nhwc: KS = 3 (padding 1) or 1; order of the chain per output element:
       'ckxky' (chunk, kx, ky), 'kxcky' (kx, chunk, ky) or 'c' (1x1); NT of NTALL 16-channel output tiles per
       workgroup (grid z picks them); act 'leaky' or 'none'.  params (3x3): 0 w, 8 in, 16 W, 20 H, 32 out, 40 W, 44 H;
       (1x1): 0 w, 8 in, 16 W, 20 H, 24 out, 32 W, 36 H."""
    TW = 16
    assert (TH // R) * NT == warps
    CIN = 16 * NCH
    CINPIX = CINPIX or CIN
    COUT = 16 * NTALL
    CW = 16 * NT
    HALO = 1 if KS == 3 else 0
    TW2, TH2 = TW + 2 * HALO, TH + 2 * HALO
    CS = 24
    TIN = TW2 * TH2 * CS
    NTAPS = KS * KS
    WOFF = TIN
    BOFF = WOFF + NTAPS * NT * 256
    TOT = max(BOFF + NT * 256, TH * TW * CW)
    if KS == 3:
        params = 'uint64_t wp = nvx_p64(0u), ip = nvx_p64(8u), op = nvx_p64(32u);\n    int W = int(nvx_p32(16u)), H = int(nvx_p32(20u)), Wo = int(nvx_p32(40u)), Ho = int(nvx_p32(44u));'
        blk = '((gg * 3u + ky) * 3u + kx) * %du + c' % NCH
    else:
        params = 'uint64_t wp = nvx_p64(0u), ip = nvx_p64(8u), op = nvx_p64(24u);\n    int W = int(nvx_p32(16u)), H = int(nvx_p32(20u)), Wo = int(nvx_p32(32u)), Ho = int(nvx_p32(36u));'
        blk = 'gg * %du + c' % NCH
    # one staging = input chunk c (+ halo) and its weights for the taps listed in STAGE_TAPS
    stage = r'''
        barrier();
        for (uint e = lid; e < %(NIN)du; e += nthr) {
            uint hh = e & 1u, p = e >> 1;
            int gx = x0 - %(HALO)d + int(p %% %(TW2)du), gy = y0 - %(HALO)d + int(p / %(TW2)du);
            uvec4 v = uvec4(0u);
            if (gx >= 0 && gy >= 0 && gx < W && gy < H)
                v = %(LOADIN)s;
            smv[(p * %(CS)du + hh * 8u) >> 3] = v;
        }
        for (uint e = lid; e < %(NTAPS)du * %(NT)du * 128u; e += nthr) {
            uint kp = e & 7u, n = (e >> 3) & 15u, q = e >> 7, t = q %% %(NT)du, tap = q / %(NT)du;
            uint kx = tap / %(KS)du, ky = tap %% %(KS)du, gg = (zt + t) * 2u + (n >> 3), k = kp * 2u;
            uint blk = %(BLK)s;
            uint pos = 16u * (n & 7u) + 4u * ((k & 7u) >> 1) + ((k & 8u) != 0u ? 2u : 0u);
            smem[((%(WOFF)du + (tap * %(NT)du + t) * 256u + n * 16u) >> 1) + kp] =
                NvxU32(wp + uint64_t((blk * 128u + pos) * 2u)).v;
        }
        barrier();'''
    step = r'''
                {
                    coopmat<float16_t, %(CM)s, gl_MatrixUseB> b;
                    coopMatLoad(b, smh, %(WOFF)du + ((kx * %(KS)du + ky) * %(NT)du + nt) * 256u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
                    [[unroll]] for (uint i = 0u; i < %(R)du; i++) {
                        uint row = rg * %(R)du + i;
                        coopmat<float16_t, %(CM)s, gl_MatrixUseA> a;
                        coopMatLoad(a, smh, ((row + ky) * %(TW2)du + kx) * %(CS)du, %(CS)du, gl_CooperativeMatrixLayoutRowMajor);
                        acc[i] = coopMatMulAdd(a, b, acc[i]);
                        %(ROUND0)s
                    }
                }'''
    if order == 'ckxky':
        loops = ('    for (uint c = 0u; c < %(NCH)du; c++) {\n' + stage +
                 '\n        [[unroll]] for (uint kx = 0u; kx < 3u; kx++) {\n            [[unroll]] for (uint ky = 0u; ky < 3u; ky++)' +
                 step + '\n        }\n    }')
    elif order == 'kxcky':
        loops = ('    for (uint kx_ = 0u; kx_ < 3u; kx_++) {\n    for (uint c = 0u; c < %(NCH)du; c++) {\n' + stage +
                 '\n        uint kx = kx_;\n            [[unroll]] for (uint ky = 0u; ky < 3u; ky++)' + step + '\n    }\n    }')
    elif order == 'c':
        loops = ('    for (uint c = 0u; c < %(NCH)du; c++) {\n' + stage +
                 '\n        uint kx = 0u, ky = 0u;' + step + '\n    }')
    else:
        raise ValueError(order)
    if act == 'leaky':
        epi = r'''
        uint16_t ml = float16BitsToUint16((uint16BitsToFloat16(h001) * uint16BitsToFloat16(lo)));
        uint16_t mh = float16BitsToUint16((uint16BitsToFloat16(h001) * uint16BitsToFloat16(hi)));
        lo = float16BitsToUint16((nvx_hmax(uint16BitsToFloat16(lo), uint16BitsToFloat16(ml))));
        hi = float16BitsToUint16((nvx_hmax(uint16BitsToFloat16(hi), uint16BitsToFloat16(mh))));'''
    else:
        epi = ''
    g = dict(TW=TW, TW2=TW2, TH=TH, R=R, NCH=NCH, NT=NT, CIN=CIN, CINPIX=CINPIX, COUT=COUT, CW=CW, CS=CS,
             WOFF=WOFF, BOFF=BOFF, BIAS=BIAS, CM=CM, NROW=TH // R, HALO=HALO, KS=KS, NTAPS=NTAPS, BLK=blk,
             NIN=TW2 * TH2 * 2, NB=NT * 256, NOUT=TH * TW * (CW // 2), CW2=CW // 2, COUT2=COUT // 2,
             ROUND0=ROUND % {'m': 'acc[i]'}, PARAMS=params, EPI=epi)
    hk = hooks or {}
    g['LOADIN'] = hk.get('loadin', 'NvxU32x4a4(ip + uint64_t((uint(gy * W + gx) * %(CINPIX)du + c * 16u + hh * 8u) * 2u)).v') % g
    g['LOOPS'] = (loops % g)
    g['EPI_STORE'] = hk.get('epi_store', '') % dict(g, LO='lo', HI='hi', CWX=CW, ZT='zt')
    g['POST'] = hk.get('post', '') % dict(g, CWX=CW, CW2X=CW // 2, ZT='zt', NPOOL=(TW // 2) * (TH // 2) * (CW // 2),
                                          TW2H=TW // 2)
    body = r'''
void main() {
    %(PARAMS)s
    uint lid = gl_LocalInvocationIndex, nthr = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    int x0 = int(gl_WorkGroupID.x) * %(TW)d, y0 = int(gl_WorkGroupID.y) * %(TH)d;
    uint zt = gl_WorkGroupID.z * %(NT)du;
    uint w = gl_SubgroupID, rg = w %% %(NROW)du, nt = w / %(NROW)du;
    for (uint e = lid; e < %(NB)du; e += nthr) {
        uint col = e & 15u, t = e >> 8, gg = (zt + t) * 2u + (col >> 3);
        smh[%(BOFF)du + e] = uint16BitsToFloat16(NvxU16(wp + uint64_t((%(BIAS)du + 128u * gg + (col & 7u)) * 2u)).v);
    }
    barrier();
    coopmat<float, %(CM)s, gl_MatrixUseAccumulator> acc[%(R)d];
    {
        coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator> c16;
        coopMatLoad(c16, smh, %(BOFF)du + nt * 256u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        [[unroll]] for (uint i = 0u; i < %(R)du; i++) acc[i] = coopmat<float, %(CM)s, gl_MatrixUseAccumulator>(c16);
    }
%(LOOPS)s
    barrier();
    [[unroll]] for (uint i = 0u; i < %(R)du; i++) {
        uint row = rg * %(R)du + i;
        coopMatStore(coopmat<float16_t, %(CM)s, gl_MatrixUseAccumulator>(acc[i]), smh,
                     row * %(TW)du * %(CW)du + nt * 16u, %(CW)du, gl_CooperativeMatrixLayoutRowMajor);
    }
    barrier();
    uint16_t h001 = nvx_f2h(uintBitsToFloat(0x3C23D70AU));
    for (uint e = lid; e < %(NOUT)du; e += nthr) {
        uint pr = e %% %(CW2)du, q = e / %(CW2)du, px = q %% %(TW)du, py = q / %(TW)du;
        int gx = x0 + int(px), gy = y0 + int(py);
        if (gx >= Wo || gy >= Ho)
            continue;
        uint v = smem[(q * %(CW)du >> 1) + pr];
        uint16_t lo = uint16_t(v & 0xffffu), hi = uint16_t(v >> 16);%(EPI)s
        NvxU32(op + uint64_t((uint(gy * Wo + gx) * %(COUT2)du + zt * 8u + pr) * 4u)).v = uint(lo) | (uint(hi) << 16);%(EPI_STORE)s
    }%(POST)s
}
'''
    return _conv.header(TOT) + body % g, ['t1_block 32 %d 1' % warps]
