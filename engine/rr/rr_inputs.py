#!/usr/bin/env python3
"""Write the game inputs of a capture's denoiser frames for engine/replay_rr.cpp (d4r-rr).

usage: rr_inputs.py CAPTURE OUT_DIR [--frames N] [--no-reset]

The inputs are enc0's textures (colour 160, motion 168, normals 184, roughness 192, the albedos 232 and 240, depth
512); the frame settings come from enc0's argument block: jitter (320), motion scale (344, in output pixels per
render pixel), depth inversion (528) and the camera, whose rotation-times-projection (596) and projection (660) give
back a world-to-view rotation and the view-to-clip matrix that reproduce the block. The harness passes reset on every
frame (scripts/run_rr_engine_compare.sh); --no-reset clears it after the first frame.
"""
import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_capture import Capture  # noqa: E402

OPTIONAL = {'hitdistance': 248, 'disocclusion': 296}
INPUTS = {'color': 160, 'motion': 168, 'normals': 184, 'roughness': 192, 'diffuse': 232, 'specular': 240, 'depth': 512}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('out', type=Path)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--no-reset', action='store_true')
    o = ap.parse_args()
    cap = Capture(o.capture)
    o.out.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in range(1, o.frames + 1):
        enc0 = cap.find('swin_enc0', f)
        a = enc0.args
        u32 = lambda off: struct.unpack_from('<I', a, off)[0]
        f32 = lambda off: struct.unpack_from('<f', a, off)[0]
        mat = lambda off: np.array(struct.unpack_from('<16f', a, off), np.float64).reshape(4, 4)
        rw, rh, ow, oh = u32(144), u32(148), u32(472), u32(476)
        if f == 1:
            lines.append(f'size {rw} {rh} {ow} {oh}')
        images = {i.offset: i for i in enc0.images}
        for name, off in INPUTS.items():
            (o.out / f'{f}-{name}.bin').write_bytes(images[off].path.read_bytes())
        for name, off in OPTIONAL.items():   # DLSSD.SpecularHitDistance, DLSS.DisocclusionMask when the game passed them
            if off in images:
                (o.out / f'{f}-{name}.bin').write_bytes(images[off].path.read_bytes())
        clip = mat(660)
        view = mat(596) @ np.linalg.inv(clip)
        view[3, :3] = 0.0
        jx, jy = f32(320), f32(324)
        mvx, mvy = f32(344) * rw / ow, f32(348) * rh / oh
        reset = 1 if f == 1 or not o.no_reset else 0
        values = ' '.join(f'{v:.9g}' for v in list(view.flatten()) + list(clip.flatten()))
        lines.append(f'frame {f} {jx:.9g} {jy:.9g} {mvx:.9g} {mvy:.9g} 1 {reset} {u32(528)} {values}')
    (o.out / 'frames.txt').write_text('\n'.join(lines) + '\n')
    print(f'{o.frames} frames of {o.capture} in {o.out}')


if __name__ == '__main__':
    main()
