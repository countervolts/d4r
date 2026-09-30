#!/usr/bin/env python3
"""Compares native preset-K layer outputs with the numpy model (pwin_model.py) on sampled windows of a replay dump.

usage: pwin_check.py DUMP_DIR OUT_DIR [OUT_DIR...] [--windows N]
  DUMP_DIR  replay dump of one dltss_pwin_* launch (inputs)
  OUT_DIR   dump_runner output of a native kernel on that dump (alloc-N.bin after the launch)
For every sampled window (the four corners, border windows and random interior ones) the model output and the
kernel's stored values are compared per f16 element: exact matches, and the error in f16 ulps at the model
value's magnitude (values below 2^-10 count with the ulp of 2^-10, so near-zero sign flips do not dominate).
The kernels round as NVIDIA's f16 wmma does except in the PWIN_F32ACC layers, and gfx12 WMMA groups its f32
sums differently from gfx11, so neither build is expected to match the model exactly.
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pwin_model as pm  # noqa: E402

F16 = np.float16

# layer -> (kind, H, C, COUT/CL, extra)
LAYERS = {
    'enc0': ('enc0', 2, 32, 64, None), 'enc1': ('enc', 2, 64, 64, True), 'enc2': ('enc', 2, 64, 96, False),
    'enc3': ('enc', 4, 96, 128, False), 'enc4': ('enc', 4, 128, 160, False), 'dec5': ('plain', 8, 160, 16, None),
    'dec4': ('dec', 4, 128, 160, None), 'dec3': ('dec', 4, 96, 128, None), 'dec2': ('dec', 2, 64, 96, None),
    'dec1': ('dec', 2, 64, 64, None), 'dec0': ('dec0', 2, 32, 64, None),
}


class Outputs:
    """the dump's pointer map over the allocations a native run saved"""
    def __init__(self, dump, out_dir):
        self.d = dump
        self.allocs = {i: (base, np.fromfile(f'{out_dir}/alloc-{i}.bin', np.uint8)) for i, (base, _) in dump.allocs.items()}

    def rows(self, off_param, index, width):
        a = self.d.u64(off_param)
        for base, buf in self.allocs.values():
            if base <= a < base + buf.size:
                o = a - base + 2 * index * width
                return buf[o:o + 2 * width].view(F16)
        raise ValueError(hex(a))


def ulps(model, got):
    m = model.astype(np.float64)
    g = got.astype(np.float64)
    scale = np.maximum(np.abs(m), 2.0 ** -10)
    ulp = 2.0 ** (np.floor(np.log2(scale)) - 10)
    return np.abs(m - g) / ulp


def windows(grid, n, seed=1):
    gx, gy = grid[0], grid[1]
    w = {(0, 0), (gx - 1, 0), (0, gy - 1), (gx - 1, gy - 1), (gx // 2, 0), (0, gy // 2), (gx - 1, gy // 2), (gx // 2, gy - 1)}
    r = random.Random(seed)
    while len(w) < min(n, gx * gy):
        w.add((r.randrange(gx), r.randrange(gy)))
    return sorted(w)


def model_window(P, layer, bx, by):
    kind, H, C, X, extra = LAYERS[layer]
    if kind == 'enc0':
        return pm.pwin_enc0(P, bx, by)
    if kind == 'enc':
        return pm.pwin_encoder(P, bx, by, H, C, X, posattn=extra)
    if kind == 'plain':
        return pm.swin_core(P, pm.Layout(H, C, C), pm.load_window(P, bx, by, C)), None
    if kind == 'dec':
        return pm.pwin_decoder(P, bx, by, H, C, X), None
    return pm.pwin_dec0(P, bx, by)


def stored(P, O, layer, bx, by, y, extra_out):
    """(model values, kernel values) of everything the kernel stores for window (bx, by)"""
    kind, H, C, X, _ = LAYERS[layer]
    W, Hh = P.i32x2(0)
    sx, sy = P.i32x2(56)
    ms, gs = [], []
    full_off = 32 if kind in ('enc0', 'enc') else 24
    for t in range(64):
        Y0, X0 = 8 * by - sy + (t >> 3), 8 * bx - sx + (t & 7)
        if not (0 <= Y0 < Hh and 0 <= X0 < W):
            continue
        if kind == 'dec0':
            ms.append(extra_out[t]); gs.append(O.rows(24, Y0 * W + X0, 40))
        else:
            ms.append(y[t]); gs.append(O.rows(full_off, Y0 * W + X0, C))
    if kind in ('enc0', 'enc'):
        cout = extra_out.shape[1]
        for r in range(16):
            Ym, Xm = int((8 * by - sy) / 2) + (r >> 2), int((8 * bx - sx) / 2) + (r & 3)
            if 0 <= Ym < Hh // 2 and 0 <= Xm < W // 2:
                ms.append(extra_out[r]); gs.append(O.rows(24, Ym * (W // 2) + Xm, cout))
    return np.concatenate(ms), np.concatenate(gs)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n = 12
    if '--windows' in sys.argv:
        n = int(sys.argv[sys.argv.index('--windows') + 1])
        args.remove(str(n))
    dump_dir, outs = args[0], args[1:]
    P = pm.Dump(dump_dir)
    layer = P.kernel.replace('dltss_pwin_', '').replace('_layer', '')
    wins = windows(P.grid, n)
    models = {w: model_window(P, layer, *w) for w in wins}
    for out in outs:
        O = Outputs(P, out)
        m, g = zip(*(stored(P, O, layer, bx, by, *models[(bx, by)]) for bx, by in wins))
        m, g = np.concatenate(m), np.concatenate(g)
        u = ulps(m, g)
        exact = (m.view(np.uint16) == g.view(np.uint16)).mean()
        md, gd = m.astype(np.float64), g.astype(np.float64)
        psnr = 10 * np.log10(np.abs(md).max() ** 2 / max(((md - gd) ** 2).mean(), 1e-30))
        print(f'{layer:5s} {os.path.basename(os.path.normpath(out)):10s} {len(wins)} windows {m.size} values: exact {100 * exact:5.1f}%  '
              f'<=1ulp {100 * (u <= 1).mean():5.1f}%  <=4ulp {100 * (u <= 4).mean():6.2f}%  max {u.max():7.1f} ulp  '
              f'mean {u.mean():.3f} ulp  psnr {psnr:.1f} dB  finite {np.isfinite(g.astype(np.float32)).mean() * 100:.0f}%')


if __name__ == '__main__':
    main()
