#!/usr/bin/env python3
"""Compares native preset-M Swin-block outputs with the numpy model (swin_model.py) on sampled blocks of a dump.

usage: swin_check.py DUMP_DIR OUT_DIR [OUT_DIR...] [--blocks N]
  DUMP_DIR  replay dump of one rrlite_{enc1,enc2,enc3_tube,dec1,dec2}_4x4 launch
  OUT_DIR   dump_runner output of a native kernel on that dump
The stored e4m3 outputs (block output, and the patch merge of encoders that have one) are compared code by code:
the share of identical codes, of codes one e4m3 step apart, and the PSNR of the decoded values. The model is
NVIDIA's arithmetic (e4m3 activations, f16 accumulation per k32 step); the production kernels keep activations
f16 and accumulate in f32 (SWIN_NO_Q8, SWIN_F32ACC), so they are not expected to match it exactly.
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_enc3 as me  # noqa: E402
import swin_model as sm  # noqa: E402

# kernel -> (C, heads, NPM, CIN)
LAYERS = {'enc1': (64, 2, 96, 0), 'enc2': (96, 4, 128, 0), 'enc3_tube': (128, 4, 0, 0), 'dec2': (96, 4, 0, 128),
          'dec1': (64, 2, 0, 96)}


def blocks(grid, n, seed=1):
    gx, gy = grid[0], grid[1]
    b = {(0, 0), (gx - 1, 0), (0, gy - 1), (gx - 1, gy - 1), (gx // 2, 0), (0, gy // 2), (gx - 1, gy // 2), (gx // 2, gy - 1)}
    r = random.Random(seed)
    while len(b) < min(n, gx * gy):
        b.add((r.randrange(gx), r.randrange(gy)))
    return sorted(b)


def model_block(P, layer, bx, by):
    C, heads, NPM, CIN = LAYERS[layer]
    pre = 4 * C * CIN + 8 * C if CIN else 0
    x0 = sm.decoder_input(P, bx, by, C, CIN, heads, P.p32) if CIN else sm.load_tokens(P, bx, by, C // 32)
    x2, out, L = sm.swin_block(P, x0, C, heads, pre)
    merged = None
    if NPM:
        A = sm.patch_merge_input(out, C)
        PM0 = L['end']
        cols = []
        for g in range(NPM // 32):
            Wg = me.E4M3[P.wbuf[sm.woff_table(PM0 + g * 128 * C, 512, 64 * C, 4 * C, 32)]]
            bias = P.f16vec(PM0 + 4 * C * NPM + 64 * g, 32)
            cols.append(me.mma_chain(A, Wg, C0=np.broadcast_to(bias, (16, 32)).copy()))
        merged = me.q8(sm.pair_to_nat(np.concatenate(cols, axis=1)))
    return out, merged


def codes_at(buf, base, ptr, W, H, X, Y, C):
    """e4m3 codes of tokens (X, Y) (in bounds) from planes of 32 channels at ptr"""
    c = np.arange(C)
    off = ptr - base + (((c[None, :] >> 5) * H + Y[:, None]) * W + X[:, None]) * 32 + (c[None, :] & 31)
    return buf[off]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n = 8
    if '--blocks' in sys.argv:
        n = int(sys.argv[sys.argv.index('--blocks') + 1])
        args.remove(str(n))
    dump, outs = args[0], args[1:]
    P = me.Params(dump)
    layer = P.kernel.replace('rrlite_', '').replace('_4x4', '')
    C, heads, NPM, CIN = LAYERS[layer]
    bl = blocks(P.grid, n)
    models = {b: model_block(P, layer, *b) for b in bl}
    for out_dir in outs:
        got, want = [], []
        for (bx, by) in bl:
            base, _ = P.alloc_of(P.out)
            aid = next(i for i, (b0, buf) in P.allocs.items() if b0 == base)
            buf = np.fromfile(f'{out_dir}/alloc-{aid}.bin', np.uint8)
            X, Y = P.token_xy(bx, by)
            ok = (X >= 0) & (X < P.W) & (Y >= 0) & (Y < P.H)
            out, merged = models[(bx, by)]
            got.append(codes_at(buf, base, P.out, P.W, P.H, X[ok], Y[ok], C).ravel())
            want.append(out[ok].ravel())
            if NPM:
                W2, H2 = P.W // 2, P.H // 2
                r = np.arange(16)
                MX, MY = (8 * bx - P.sx) // 2 + (r & 3), (8 * by - P.sy) // 2 + (r >> 2)
                okm = (MX >= 0) & (MX < W2) & (MY >= 0) & (MY < H2)
                mbase, _ = P.alloc_of(P.p48)
                mid = next(i for i, (b0, _) in P.allocs.items() if b0 == mbase)
                mbuf = buf if mid == aid else np.fromfile(f'{out_dir}/alloc-{mid}.bin', np.uint8)
                got.append(codes_at(mbuf, mbase, P.p48, W2, H2, MX[okm], MY[okm], NPM).ravel())
                want.append(merged[okm].ravel())
        g = me.E4M3[np.concatenate(got)]
        w = np.concatenate(want).astype(np.float64)
        rank = lambda v: np.searchsorted(me._POS, np.abs(v)) * np.sign(v)  # noqa: E731
        steps = np.abs(rank(g) - rank(w))
        psnr = 10 * np.log10(np.nanmax(np.abs(w)) ** 2 / max(np.nanmean((g - w) ** 2), 1e-30))
        print(f'{layer:9s} {os.path.basename(os.path.normpath(out_dir)):10s} {len(bl)} blocks {g.size} codes: '
              f'identical {100 * (g == w).mean():5.1f}%  <=1 step {100 * (steps <= 1).mean():6.2f}%  '
              f'max {int(steps.max())} steps  psnr {psnr:.1f} dB')


if __name__ == '__main__':
    main()
