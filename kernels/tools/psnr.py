#!/usr/bin/env python3
"""psnr.py DIR_A DIR_B: per-frame PSNR of rgba16f outputs (watermark box excluded, values clipped to [0, 1] after tonemap-free clamp)."""
import sys, glob, os, numpy as np
a, b = sys.argv[1:3]
vals = []
for f in sorted(glob.glob(a + '/frame-*.rgba16f')):
    g = os.path.join(b, os.path.basename(f))
    if not os.path.exists(g): continue
    x = np.fromfile(f, np.float16).reshape(1440, 2560, 4)[..., :3].astype(np.float64)
    y = np.fromfile(g, np.float16).reshape(1440, 2560, 4)[..., :3].astype(np.float64)
    m = np.ones(x.shape[:2], bool); m[1370:1392, 1850:2460] = False
    x, y = np.clip(x[m], 0, 1), np.clip(y[m], 0, 1)
    mse = np.mean((x - y) ** 2)
    vals.append(10 * np.log10(1.0 / mse) if mse > 0 else 99.0)
print('frames', len(vals), 'PSNR dB min %.2f mean %.2f' % (min(vals), np.mean(vals)), ' '.join('%.1f' % v for v in vals))
