#!/usr/bin/env python3
"""Extract the preset K network from one captured frame into d4r's own model file.

usage: extract_k.py REPLAY_DIR OUT_PREFIX
  REPLAY_DIR  D4R_CUDA_REPLAY_DUMP_DIR of a run with D4R_CUDA_REPLAY_DUMP_FILTER=dltss_pwin (the 11 layer launches)
Writes OUT_PREFIX.bin (f16) and OUT_PREFIX.json (per layer: shape parameters and the offset of every tensor, in f16
elements). Tensors are plain row-major matrices W[K][N] (y = x W), unlike the pre-swizzled fragments NGX keeps on
the GPU; per-channel vectors are stored 16 times (one 16-row tile) so a kernel can load them as a matrix operand.
No NVIDIA code is involved: only the weight values, read from the buffers NGX uploaded.
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kernels', 'tools'))
import pwin_model as pm

F16 = np.float16
# layer -> (kind, heads, C, COUT or CL, position-only attention)
LAYERS = {'enc0': ('enc0', 2, 32, 64, True), 'enc1': ('enc', 2, 64, 64, True), 'enc2': ('enc', 2, 64, 96, False),
          'enc3': ('enc', 4, 96, 128, False), 'enc4': ('enc', 4, 128, 160, False), 'dec5': ('plain', 8, 160, 0, False),
          'dec4': ('dec', 4, 128, 160, False), 'dec3': ('dec', 4, 96, 128, False), 'dec2': ('dec', 2, 64, 96, False),
          'dec1': ('dec', 2, 64, 64, False), 'dec0': ('dec0', 2, 32, 64, True)}


class Blob:
    def __init__(self):
        self.parts, self.n = [], 0

    def add(self, a):
        a = np.ascontiguousarray(a, F16).ravel()
        pad = (-self.n) % 16                      # keep every tensor 32-byte aligned
        if pad:
            self.parts.append(np.zeros(pad, F16)); self.n += pad
        off = self.n
        self.parts.append(a); self.n += a.size
        return off


def tile16(v):
    return np.tile(np.asarray(v, F16)[None, :], (16, 1))


def pad_cols(W, n):
    out = np.zeros((W.shape[0], n), F16); out[:, :W.shape[1]] = W
    return out


def core(P, L, blob, posattn, fold_gains=True):
    C, H = L.C, L.H
    # The two per-channel scales are folded into the rows of the weights they feed (x * g) W = x (diag(g) W):
    # norm 1's gain into Q, K, V and the MLP input gain into fc1. Products in f64, rounded once to f16.
    g1 = pm.vec(P, L.G1, C).astype(np.float64)[:, None]
    g2 = pm.vec(P, L.G2, C).astype(np.float64)[:, None]
    t = {'g1': blob.add(tile16(pm.vec(P, L.G1, C)))}
    t['qkv'], t['table'], t['wo'] = [], [], []
    for h in range(H):
        hb = L.QKV + L.HEAD * h
        for j in range(3):
            W = np.concatenate([pm.weights(P, hb + 2048 * (3 * ch + j), 32, 32, 1024) for ch in range(C // 32)], axis=0)  # [C][32]
            t['qkv'].append(blob.add((W.astype(np.float64) * g1).astype(F16) if fold_gains else W))
        t['table'].append(blob.add(pm.bias_table(P, L.BIAS + 8192 * h)))                                                 # [64][64]
        t['wo'].append(blob.add(pm.weights(P, L.WO + 64 * C * h, 32, C, 1024)))                                         # [32][C]
    t['bo'] = blob.add(tile16(pm.vec(P, L.BO, C)))
    t['g2'] = blob.add(tile16(pm.vec(P, L.G2, C)))
    t['b2'] = blob.add(tile16(pm.vec(P, L.B2, C)))
    t['w1'] = []
    for c in range(C // 8):
        W = pm.weights(P, L.W1 + 64 * C * c, C, 32, 32 * C)
        t['w1'].append(blob.add((W.astype(np.float64) * g2).astype(F16) if fold_gains else W))
    t['b1'] = [blob.add(tile16(pm.vec(P, L.B1 + 64 * c, 32))) for c in range(C // 8)]
    t['w2'] = [blob.add(pm.weights(P, L.W2 + 64 * C * c, 32, C, 1024)) for c in range(C // 8)]                          # [32][C]
    return t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay_dir')
    parser.add_argument('out_prefix')
    parser.add_argument('--unfold-gains', action='store_true', help='preserve explicit f16 gain and residual rounding instead of folding gains into weights')
    args = parser.parse_args()
    rdir, prefix = args.replay_dir, args.out_prefix
    blob, index = Blob(), {}
    for name, (kind, H, C, X, posattn) in LAYERS.items():
        matches = sorted(glob.glob(f'{rdir}/replay-*-dltss_pwin_{name}_layer'))
        if not matches:
            raise ValueError(f'missing preset K layer {name} in {rdir}')
        d = matches[0]
        P = pm.Dump(d)
        e = {'kind': kind, 'H': H, 'C': C, 'X': X, 'posattn': posattn, 'fold_gains': not args.unfold_gains}
        if kind in ('enc0', 'enc'):
            L = pm.Layout(H, C, X)
            Pw = P
            if kind == 'enc0':
                CIN = 16
                e['embed_w'] = blob.add(pm.weights(P, 2 * C, CIN, C, 512 * (CIN // 16)))        # [16][C]
                e['embed_b'] = blob.add(tile16(pm.vec(P, 0, C)))
                Pw = pm.Shifted(P, 2 * C + 2 * CIN * C)
            e.update(core(Pw, L, blob, posattn, not args.unfold_gains))
            Wp = pm.weights(Pw, L.PM, 4 * C, L.NPA * H, 128 * C)                                 # [4C][NPA*H]
            bias = pm.vec(Pw, L.PMB, L.NPA * H) if kind == 'enc0' or name != 'enc1x' else None
            cols = np.concatenate([Wp[:, L.NPA * w:L.NPA * w + L.NPW] for w in range(H)], axis=1)        # [4C][COUT]
            b = np.concatenate([bias[L.NPA * w:L.NPA * w + L.NPW] for w in range(H)])
            n16 = -(-X // 16) * 16
            e['merge_w'] = blob.add(pad_cols(cols, n16)); e['merge_b'] = blob.add(tile16(np.concatenate([b, np.zeros(n16 - X, F16)])))
            e['merge_n'] = n16
        elif kind == 'plain':
            e.update(core(P, pm.Layout(H, C, C), blob, posattn, not args.unfold_gains))
        else:
            CL = X
            EXP = 2 * CL * 4 * C
            e['expand_w'] = blob.add(pm.weights(P, 0, CL, 4 * C, 512 * (CL // 16)))              # [CL][4C]
            e['expand_b'] = blob.add(tile16(pm.vec(P, EXP, 4 * C)))
            L = pm.Layout(H, C, C)
            Pw = pm.Shifted(P, EXP + 8 * C)
            e.update(core(Pw, L, blob, posattn, not args.unfold_gains))
            if kind == 'dec0':
                NOUTA = 48
                e['head_w'] = blob.add(pm.weights(Pw, L.PM + 2 * NOUTA, C, NOUTA, 1024))         # [C][48]
                e['head_b'] = blob.add(tile16(pm.vec(Pw, L.PM, NOUTA)))
        index[name] = e
    data = np.concatenate(blob.parts)
    data.tofile(prefix + '.bin')
    with open(prefix + '.json', 'w') as f:
        json.dump(index, f, indent=1)
    print(f'{prefix}.bin: {data.size * 2} bytes, {len(index)} layers')


if __name__ == '__main__':
    main()
