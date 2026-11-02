#!/usr/bin/env python3
"""Recreate the official NVIDIA SR probe's input exactly, as d4r-m's raw frame files.

tools/bench_official_ngx_cuda.cpp builds its SR input on the host:
  color[x,y].r = 0x3400 + ((x/8 + y/8) % 8) * 128   (binary16 bit patterns)
  .g = 0x3800, .b = 0x3000, .a = 0x3c00
  motion: all zero (RG16F), depth: 0.5 everywhere (R32F)
It then evaluates with jitter X = +-0.25 by frame%2 and Y = +-0.25 by frame%4<2, and Reset=1 only on frame 0.
This writes four frames carrying that jitter cycle (the jitter itself is passed on d4r-m's command line).

usage: make_sr_inputs.py OUT_DIR RENDER_W RENDER_H
"""
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
w, h = int(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

x = np.arange(w, dtype=np.int32) // 8
y = np.arange(h, dtype=np.int32) // 8
pattern = (x[None, :] + y[:, None]) % 8
red = np.uint16(0x3400 + pattern * 128)
green = np.full((h, w), 0x3800, np.uint16)
blue = np.full((h, w), 0x3000, np.uint16)
alpha = np.full((h, w), 0x3C00, np.uint16)
color = np.stack([red, green, blue, alpha], axis=-1).reshape(-1)
motion = np.zeros(h * w * 2, np.uint16)
depth = np.full(h * w, 0.5, np.float32)

for f in range(4):
    (out / f'{f}-color.bin').write_bytes(color.tobytes())
    (out / f'{f}-motion.bin').write_bytes(motion.tobytes())
    (out / f'{f}-depth.bin').write_bytes(depth.tobytes())
print(f'{out}: 4 frames {w}x{h} color RGBA16F {color.size * 2} B, motion RG16F {motion.size * 2} B, depth R32F {depth.size * 4} B')
