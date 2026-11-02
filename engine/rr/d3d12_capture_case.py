#!/usr/bin/env python3
"""Turn a native D3D12 RR capture (D4R_RR_CAPTURE_DIR) into a replay case.

The capture holds NNNN-meta.txt and NNNN-NAME.raw per frame: the game's own
textures (subresource 0, rows unpadded) in their own DXGI formats, and the
engine's output.  The case uses the planes d4r-rr-d3d12 reads (frames.txt,
N-NAME.bin); formats it does not take are converted here when that is lossless
(channel drops, depth typeless views), otherwise the import stops.  The game's
output is kept as N-game.bin (RGBA16F) for comparison.
"""
import argparse
import os
from pathlib import Path

import numpy as np

# case plane -> (DXGI format the case stores, bytes per pixel)
CASE = {'color': (10, 8), 'motion': (34, 4), 'depth': (41, 4), 'normals': (10, 8), 'roughness': (10, 8),
        'diffuse': (28, 4), 'specular': (28, 4), 'hitdistance': (41, 4), 'disocclusion': (54, 2),
        'sss': (54, 2), 'particles': (10, 8)}
BPP = {2: 16, 10: 8, 11: 8, 16: 8, 24: 4, 26: 4, 27: 4, 28: 4, 29: 4, 34: 4, 35: 4, 39: 4, 40: 4, 41: 4,
       53: 2, 54: 2, 56: 2, 61: 1, 65: 1, 87: 4}
R32 = {39, 40, 41}   # R32_TYPELESS / D32_FLOAT / R32_FLOAT
RGBA16 = {9, 10}      # R16G16B16A16_TYPELESS / _FLOAT
RGBA8 = {27, 28, 29}  # R8G8B8A8_TYPELESS / _UNORM / _UNORM_SRGB (bits kept)


def convert(name, fmt, raw, w, h):
    want, bpp = CASE[name]
    if fmt == want or (want == 41 and fmt in R32) or (want == 10 and fmt in RGBA16) or (want == 28 and fmt in RGBA8):
        return raw
    if want == 34 and fmt in RGBA16:  # motion in RGBA16F: keep RG
        return np.frombuffer(raw, np.uint32).reshape(-1, 2)[:, 0].tobytes()
    if want == 54 and fmt in (41,):  # R32F guide -> R16F
        return np.frombuffer(raw, np.float32).astype(np.float16).tobytes()
    if want == 54 and fmt in RGBA16:  # first channel
        return np.frombuffer(raw, np.uint16).reshape(-1, 4)[:, 0].tobytes()
    if want == 41 and fmt == 54:
        return np.frombuffer(raw, np.float16).astype(np.float32).tobytes()
    if want == 10 and fmt == 2:  # RGBA32F -> RGBA16F
        return np.frombuffer(raw, np.float32).astype(np.float16).tobytes()
    raise ValueError(f'{name}: no lossless conversion from DXGI format {fmt}')


def place(src, data, dst, same):
    """Writes data to dst, as a hard link to src when unchanged (captures are large)."""
    dst.unlink(missing_ok=True)
    if same:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    dst.write_bytes(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('case', type=Path)
    a = p.parse_args()
    a.case.mkdir(parents=True, exist_ok=True)
    metas = sorted(a.capture.glob('*-meta.txt'))
    if not metas:
        raise SystemExit('no captured frames')
    lines, size = [], None
    for n, meta in enumerate(metas):
        index = meta.name.split('-')[0]
        rows = [l.split() for l in meta.read_text().splitlines() if l.strip()]
        st = next(r for r in rows if r[0] == 'settings')
        view = next(r for r in rows if r[0] == 'worldToView')[1:]
        clip = next(r for r in rows if r[0] == 'viewToClip')[1:]
        rw, rh = int(st[1]), int(st[2])
        jx, jy, mx, my, pre, _, reset, inverted, hw, packed = st[3:13]
        planes = {r[1]: (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in rows if r[0] == 'plane'}
        ow, oh = planes['output'][1:3]
        if size and size != (rw, rh, ow, oh):
            raise SystemExit('resize inside the capture')
        size = (rw, rh, ow, oh)
        for name, (fmt, w, h, row) in planes.items():
            src = a.capture / f'{index}-{name}.raw'
            raw = src.read_bytes()
            if name == 'output':
                if fmt not in RGBA16:
                    raise SystemExit(f'output format {fmt}')
                place(src, raw, a.case / f'{n}-game.bin', True)
                continue
            if (w, h) != (rw, rh):
                raise SystemExit(f'{name}: {w}x{h} is not the render size')
            data = convert(name, fmt, raw, w, h)
            if len(data) != rw * rh * CASE[name][1]:
                raise SystemExit(f'{name}: {len(data)} bytes after conversion')
            place(src, data, a.case / f'{n}-{name}.bin', data is raw)
        flag = (1 if int(inverted) else 0) | (2 if int(hw) else 0) | (4 if int(packed) else 0)
        lines.append(' '.join(['frame', str(n), jx, jy, mx, my, pre, reset, str(flag), *view, *clip]))
        print(f'frame {n}: ' + ', '.join(f'{k}={v[0]}' for k, v in planes.items()))
    (a.case / 'frames.txt').write_text(f'size {size[0]} {size[1]} {size[2]} {size[3]}\n' + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
