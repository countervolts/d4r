#!/usr/bin/env python3
"""Run the engine's own preset M layers (layer_m.comp) on a captured frame and compare with the captured outputs and
with the plain numpy model (m_model.py).

usage: test_m.py CAPTURE_DIR WORK_DIR EVK [--frame N] [--time N] [--model] [--chain] [--dot2] [layer index ...]
  CAPTURE_DIR/replay: layer launches before they ran; CAPTURE_DIR/dump: buffers after each launch.
  --model also runs the numpy model (slow) and reports the shader against it.
  --chain feeds each layer the engine's own previous outputs instead of the captured inputs.
  --dot2 runs layer_m_dot2.comp (no cooperative matrices or FP8: RDNA2) on the matrices widened to binary16 pairs
  (the output projection's, MLP's and patch merge's int8, as engine/runtime_m.inc widenM), and the encoders' patch
  merge as merge_m_dot2.comp after the block.

The layers' FP8 matrices are exported in the shader's blocked fixed-16 tile layout (pack8 below); the front-end
embedding/expansion weights that compile_m.py packages for enc0_m/dec0_m stay in NGX's own plain layout.
"""
import glob, os, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m_model as mm
import model_enc3 as me

F16 = np.float16


def codes(values):
    """e4m3 codes of exact e4m3 values"""
    v = np.asarray(values, np.float64); a = np.abs(v)
    c = np.searchsorted(me._POS, a).astype(np.uint8)
    assert np.array_equal(me._POS[c], a), 'not an FP8 value'
    return (c | np.where(np.signbit(v) & (a > 0), 128, 0)).astype(np.uint8)


class Pack:
    def __init__(self, dtype, align):
        self.parts, self.n, self.dtype, self.align = [], 0, dtype, align

    def add(self, a):
        a = np.ascontiguousarray(a, self.dtype).ravel()
        pad = (-self.n) % self.align
        if pad: self.parts.append(np.zeros(pad, self.dtype)); self.n += pad
        off = self.n; self.parts.append(a); self.n += a.size
        return off

    def data(self):
        return np.concatenate(self.parts + [np.zeros((-self.n) % 64 + 64, self.dtype)])


def pack8(values):
    """blocked FP8 bytes of a logical weight matrix W[K][N] (K, N multiples of 16), the layout layer_m.comp loads:
    [K/16][N/16][n16][k16], i.e. reshape(K//16, 16, N//16, 16).transpose(0, 2, 3, 1). contiguous (no arithmetic).
    A fixed 16x16 tile is 256 bytes and starts at 256*(kblock*(N//16) + nblock); the shader's A operand is the tile
    [N16][K16] read row-major with stride 16 (A[n][k] = W[kblock*16 + k][nblock*16 + n]), so a K16 row block of
    every output channel tile stays contiguous and the per-head slice of wo is still base + 32*h*C."""
    W = codes(values)
    K, N = W.shape
    assert K % 16 == 0 and N % 16 == 0, (K, N)
    return np.ascontiguousarray(W.reshape(K // 16, 16, N // 16, 16).transpose(0, 2, 3, 1)).ravel()


def pack_layer(t, name):
    """(o[72], FP8 bytes, f16 words) of one layer's tensors; the FP8 matrices are blocked by pack8()"""
    C, heads, NPM, CIN = mm.LAYERS[name]
    b8, b16 = Pack(np.uint8, 16), Pack(F16, 16)
    tile = lambda v: np.tile(np.asarray(v, F16)[None, :], (16, 1))
    o = [0] * 72
    o[0], o[1], o[2], o[3] = b16.add(t['g1']), b16.add(tile(t['bo'])), b16.add(t['g2']), b16.add(tile(t['b2']))
    assert t['wo'].shape == (32 * heads, C), t['wo'].shape
    wo = b8.add(pack8(t['wo']))
    for h in range(heads):
        assert t['q'][h].shape == (C, 32) and t['v'][h].shape == (C, 32), (t['q'][h].shape, t['v'][h].shape)
        o[4 + 4 * h], o[5 + 4 * h], o[6 + 4 * h], o[7 + 4 * h] = b8.add(pack8(t['q'][h])), b8.add(pack8(t['v'][h])), b16.add(t['bias'][h]), wo + 32 * h * C
    for c in range(C // 8):
        assert t['w1'][c].shape == (C, 32) and t['w2'][c].shape == (32, C), (t['w1'][c].shape, t['w2'][c].shape)
        o[20 + 3 * c], o[21 + 3 * c], o[22 + 3 * c] = b8.add(pack8(t['w1'][c])), b16.add(tile(t['b1'][c])), b8.add(pack8(t['w2'][c]))
    if NPM:
        assert t['merge_w'].shape == (4 * C, NPM), t['merge_w'].shape
        o[68], o[69] = b8.add(pack8(t['merge_w'])), b16.add(tile(t['merge_b']))
    if CIN:
        assert t['expand_w'].shape == (CIN, 4 * C), t['expand_w'].shape
        o[68], o[69] = b8.add(pack8(t['expand_w'])), b16.add(tile(t['expand_b']))
    if 'emb_w1' in t:
        # preset L's input embedding (layer_m.comp KIND 3): 32 features -> 32 hidden -> C
        assert t['emb_w1'].shape == (32, 32) and t['emb_w2'].shape == (32, C), (t['emb_w1'].shape, t['emb_w2'].shape)
        o[60], o[61] = b8.add(pack8(t['emb_w1'])), b16.add(tile(t['emb_b1']))
        o[62], o[63] = b8.add(pack8(t['emb_w2'])), b16.add(tile(t['emb_b2']))
    return o, b8.data(), b16.data()


def defines(name):
    C, heads, NPM, CIN = mm.LAYERS[name]
    kind = {'enc0l': 3, 'dec0l': 4}.get(name, 1 if NPM else 2 if CIN else 0)
    # compile_m.py's arithmetic (ZLUDA's lowering); M_TEXL=0 gives the native accuracy kernels' for the network layers.
    # M_DEFS: extra defines for every layer_m.comp stage; M_DEFS_ENC1, M_DEFS_ENC3_TUBE, ...: for one layer
    texl = ['-DTEXL=1'] if kind >= 3 or os.environ.get('M_TEXL', '1') != '0' else []
    # M_WPW=2: two 4x4-token windows per wave in the network layers (layer_m.comp WPW; L's input / expansion: M_DEFS_*)
    wpw = [f"-DWPW={os.environ['M_WPW']}"] if kind < 3 and os.environ.get('M_WPW') else []
    return [f'-DC={C}', f'-DNH={heads}', f'-DKIND={kind}', f'-DXC={NPM or CIN or 16}'] + texl + wpw + os.environ.get('M_DEFS', '').split() + os.environ.get('M_DEFS_' + name.upper(), '').split()


# layer_m_dot2.comp MLP_PAIR_SCALE: one int8 scale for a pair of MLP chunks, whose w2 blocks then share their column
# scales (widen)
DOT2_PAIR = int(os.environ.get('M_DOT2_PAIR', '1'))   # compile_m.py's D4RM0006 packages: 1
# layer_m_dot2.comp EXPAND_INT8 (D4RM0007): the decoders' expansion on int8 (widen packs its matrix so)
DOT2_EXPAND_INT8 = int(os.environ.get('M_DOT2_EXPAND_INT8', '1'))


def dot2_defines(name):
    """layer_m_dot2.comp's build of a network layer: KIND 2 expands from XC channels, 0 not (an encoder's patch merge
    is merge_m_dot2.comp's, dot2_merge_defines)"""
    C, heads, NPM, CIN = mm.LAYERS[name]
    return [f'-DC={C}', f'-DNH={heads}', f'-DKIND={2 if CIN else 0}', f'-DXC={CIN or 16}', f'-DMLP_PAIR_SCALE={DOT2_PAIR}',
            f'-DEXPAND_INT8={DOT2_EXPAND_INT8}']


def dot2_merge_defines(name):
    """merge_m_dot2.comp's build of an encoder's patch merge: channels, merged channels, waves per 8x8 merged tokens
    (the faster count on the Steam Deck)"""
    C, heads, NPM, CIN = mm.LAYERS[name]
    return [f'-DC={C}', f'-DXC={NPM}', f'-DNW={ {"enc1": 4, "enc2": 8}[name]}']


def widen(b8, o, name):
    """a layer's FP8 bytes (pack_layer) as the dot2 stages read them (engine/runtime_m.inc widenM): every blocked matrix
    W[K][N] at byte offset off as [K/2][N] binary16 pairs (W[2i][n], W[2i+1][n]) at dword off / 2; the MLP's as int8
    [K/4][N] words (W[4i + j][n] in byte j), each column scaled by its largest magnitude / 127, then N f32 scales, and
    the output projection's (all heads as one [32 NH][C] matrix), the patch merge's and (DOT2_EXPAND_INT8) the
    decoders' expansion's too"""
    C, heads, NPM, CIN = mm.LAYERS[name]
    h16 = me.E4M3.astype(F16).view(np.uint16).astype(np.uint32)
    out = np.zeros(b8.size // 2 + 64, np.uint32)
    blocked = lambda off, K, N: b8[off + 256 * ((np.arange(K)[:, None] // 16) * (N // 16) + np.arange(N)[None, :] // 16)
                                     + (np.arange(N)[None, :] % 16) * 16 + np.arange(K)[:, None] % 16]
    mats = [(o[68], CIN, 4 * C)] if CIN and not DOT2_EXPAND_INT8 else []
    for off, K, N in mats:
        vals = h16[blocked(off, K, N)]
        out[off // 2:off // 2 + K * N // 2] = (vals[0::2] | (vals[1::2] << 16)).ravel()
    i8 = [m for h in range(heads) for m in ((o[4 + 4 * h], C, 32), (o[5 + 4 * h], C, 32))]
    i8 += [m for c in range(C // 8) for m in ((o[20 + 3 * c], C, 32), (o[22 + 3 * c], 32, C))]
    i8 += [(o[68], 4 * C, NPM)] if NPM else []
    i8 += [(o[68], CIN, 4 * C)] if CIN and DOT2_EXPAND_INT8 else []
    assert all(o[7 + 4 * h] == o[7] + h * 32 * C for h in range(heads)), 'the output projections are not consecutive'
    i8.append((o[7], 32 * heads, C))     # the heads' output projections as one [NA][C] matrix
    groups = [[m] for m in i8]
    if DOT2_PAIR:   # the MLP's w2 blocks in pairs (c, c + 1) with shared column scales
        w2 = {o[22 + 3 * c] for c in range(C // 8)}
        groups = [g for g in groups if g[0][0] not in w2] + [[(o[22 + 3 * c], 32, C), (o[22 + 3 * (c + 1)], 32, C)] for c in range(0, C // 8, 2)]
    for group in groups:
        Ws = [me.E4M3.astype(np.float64)[blocked(off, K, N)] for off, K, N in group]
        scale = np.max([np.abs(W).max(0) for W in Ws], 0) / 127.0
        scale[scale == 0] = 1.0
        for (off, K, N), W in zip(group, Ws):
            q = np.clip(np.round(W / scale), -127, 127).astype(np.int8).view(np.uint8).astype(np.uint32)
            out[off // 2:off // 2 + K * N // 4] = (q[0::4] | (q[1::4] << 8) | (q[2::4] << 16) | (q[3::4] << 24)).ravel()
            out[off // 2 + K * N // 4:off // 2 + K * N // 4 + N] = scale.astype(np.float32).view(np.uint32)
    return out


def token_codes(buf, off, W, H, C):
    a = buf[off:off + (C // 32) * H * W * 32].reshape(C // 32, H, W, 32)
    return np.ascontiguousarray(a.transpose(1, 2, 0, 3).reshape(H, W, C))


def report(tag, got, ref):
    rank = lambda v: np.searchsorted(me._POS, np.abs(v)) * np.sign(v)
    g, r = me.E4M3[got].astype(np.float64), me.E4M3[ref].astype(np.float64)
    steps = np.abs(rank(g) - rank(r))
    return (f'{tag} {mm.psnr(g, r):5.1f} dB, identical {100 * (got == ref).mean():5.1f}%, within one step {100 * (steps <= 1).mean():5.1f}%')


def main():
    a = sys.argv[1:]
    def opt(flag, default):
        if flag in a:
            i = a.index(flag); v = a[i + 1]; del a[i:i + 2]; return v
        return default
    frame, reps = int(opt('--frame', 0)), opt('--time', '1')
    model = '--model' in a
    if model: a.remove('--model')
    chain = '--chain' in a
    if chain: a.remove('--chain')
    dot2 = '--dot2' in a
    if dot2: a.remove('--dot2')
    own = {}    # arena address -> the engine's own tokens there
    cap, work, evk = a[:3]
    only = [int(x) for x in a[3:]]
    os.makedirs(work, exist_ok=True)
    Ps, dirs = mm.launches(cap, frame)
    total = 0.0
    for i, (P, d, name) in enumerate(zip(Ps, dirs, mm.ORDER)):
        if only and i not in only: continue
        C, heads, NPM, CIN = mm.LAYERS[name]
        seq = int(os.path.basename(d).split('-')[1])
        after = np.fromfile(glob.glob(f'{cap}/dump/launch-{seq:03d}-*-arg024-*')[0], np.uint8)
        t = mm.plain_layer(P, name)
        o, b8, b16 = pack_layer(t, name)
        W, H = P.W, P.H
        pre = f'{work}/{i}-{name}'
        open(pre + '.params', 'wb').write(struct.pack('<4i72I', W, H, P.sx, P.sy, *o))
        b8.tofile(pre + '.w8'); b16.tofile(pre + '.w16')
        if dot2: widen(b8, o, name).tofile(pre + '.w8')
        def source(ptr, w, h_, c):
            return own[ptr] if chain and ptr in own else token_codes(P.arena, ptr - P.abase, w, h_, c)
        if CIN:
            source(P.inp, W // 2, H // 2, CIN).tofile(pre + '.in')
            source(P.p32, W, H, C).tofile(pre + '.skip')
        else:
            source(P.inp, W, H, C).tofile(pre + '.in')
            np.zeros(64, np.uint8).tofile(pre + '.skip')
        spv = pre + '.spv'
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *(dot2_defines(name) if dot2 else defines(name)),
                        f'{HERE}/layer_m_dot2.comp' if dot2 else f'{HERE}/layer_m.comp', '-o', spv], check=True, stdout=subprocess.DEVNULL)
        gx, gy = (W + P.sx + 7) // 8, (H + P.sy + 7) // 8
        msize = (H // 2) * (W // 2) * NPM if NPM else 64
        r = subprocess.run([evk, spv, str(gx), str(gy), reps, f'in:0={pre}.params', f'in:1={pre}.w8', f'in:2={pre}.w16', f'in:3={pre}.in',
                            f'in:4={pre}.skip', f'out:5={pre}.out:{H * W * C}', f'out:6={pre}.merged:{msize}'], capture_output=True, text=True)
        if r.returncode:
            print(name, 'FAILED', r.stderr[-600:]); sys.exit(1)
        us = [float(l.split(':')[1].split()[0]) for l in r.stdout.split('\n') if 'us per dispatch' in l]
        if dot2 and NPM:
            mspv = pre + '.merge.spv'
            subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *dot2_merge_defines(name), f'{HERE}/merge_m_dot2.comp',
                            '-o', mspv], check=True, stdout=subprocess.DEVNULL)
            r = subprocess.run([evk, mspv, str((W // 2 + 7) // 8), str((H // 2 + 7) // 8), reps, f'in:0={pre}.params', f'in:1={pre}.w8',
                                f'in:2={pre}.w16', f'in:3={pre}.out', f'out:4={pre}.merged:{msize}'], capture_output=True, text=True)
            if r.returncode:
                print(name, 'merge FAILED', r.stderr[-600:]); sys.exit(1)
            us = [sum(us + [float(l.split(':')[1].split()[0]) for l in r.stdout.split('\n') if 'us per dispatch' in l])]
        total += us[0] if us else 0
        got = np.fromfile(pre + '.out', np.uint8).reshape(H, W, C)
        own[P.out] = got
        ref = token_codes(after, P.out - P.abase, W, H, C)
        line = f'{i} {name:9s} {W}x{H}x{C} {us[0] if us else 0:7.1f} us: ' + report('vs capture', got, ref)
        if NPM:
            gm = np.fromfile(pre + '.merged', np.uint8).reshape(H // 2, W // 2, NPM)
            own[P.p48] = gm
            line += '; ' + report('merged', gm, token_codes(after, P.p48 - P.abase, W // 2, H // 2, NPM))
        print(line, flush=True)
        if model:
            if CIN:
                out, merged = mm.forward(t, name, mm.tokens(P.arena, P.inp - P.abase, W // 2, H // 2, CIN), mm.tokens(P.arena, P.p32 - P.abase, W, H, C), P.sx, P.sy)
            else:
                out, merged = mm.forward(t, name, mm.tokens(P.arena, P.inp - P.abase, W, H, C), None, P.sx, P.sy)
            line = '            ' + report('shader vs numpy model', got, codes(out))
            if NPM: line += '; ' + report('merged', gm, codes(merged))
            print(line, flush=True)
    print(f'total {total:.1f} us')


if __name__ == '__main__':
    main()
