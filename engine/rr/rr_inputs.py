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

OPTIONAL = {'hitdistance': 248, 'disocclusion': 296, 'sss': 264}   # encoder 0 textures
OPTIONAL_OUTPUT = {'particles': 416}                             # output stage textures
INPUTS = {'color': 160, 'motion': 168, 'normals': 184, 'roughness': 192, 'diffuse': 232, 'specular': 240, 'depth': 512}


# preset F (rr2_enc0_kernel): its textures and argument offsets
INPUTS_F = {'color': 144, 'motion': 192, 'normals': 224, 'roughness': 240, 'diffuse': 160, 'specular': 176, 'depth': 552}


def preset_f(cap, o):
    lines = []
    for f in range(1, o.frames + 1):
        enc0 = cap.find('rr2_enc0', f)
        a = enc0.args
        u32 = lambda off: struct.unpack_from('<I', a, off)[0]
        f32 = lambda off: struct.unpack_from('<f', a, off)[0]
        mat = lambda off: np.array(struct.unpack_from('<16f', a, off), np.float64).reshape(4, 4)
        down = cap.find('rr2_downsample', 1).args   # either variant: the sizes are at the same offsets
        rw, rh, iw, ih = u32(128), u32(132), u32(488), u32(492)
        ow, oh = struct.unpack_from('<2I', down, 8)
        if f == 1:
            lines.append(f'size {rw} {rh} {ow} {oh}')
        images = {i.offset: i for i in enc0.images}
        for name, off in INPUTS_F.items():
            (o.out / f'{f}-{name}.bin').write_bytes(images[off].path.read_bytes())
        # the optional inputs: encoder 0's textures 272 / 336 / 288, the particle downsample kernel's texture 208
        for name, off in (('hitdistance', 272), ('disocclusion', 336), ('sss', 288)):
            if off in images:
                (o.out / f'{f}-{name}.bin').write_bytes(images[off].path.read_bytes())
        part = [l for l in cap.launches if l.kernel == 'rr2_downsample_kernel_static_hdr']
        if part:
            n = len(part) // len(cap.find('rr2_enc0'))
            (o.out / f'{f}-particles.bin').write_bytes(part[(f - 1) * n].image(208).path.read_bytes())
        clip = mat(708)
        view = mat(644) @ np.linalg.inv(clip)
        view[3, :3] = 0.0
        reset = 1 if f == 1 or not o.no_reset else 0
        values = ' '.join(f'{v:.9g}' for v in list(view.flatten()) + list(clip.flatten()))
        flags = a[568] | (a[1028] << 1) | (a[466] << 2)   # inverted depth, hardware depth, packed roughness
        lines.append(f'frame {f} {f32(400):.9g} {f32(404):.9g} {f32(448) * rw / iw:.9g} {f32(452) * rh / ih:.9g} 1 {reset} {flags} {values}')
    (o.out / 'frames.txt').write_text('\n'.join(lines) + '\n')
    print(f'{o.frames} preset F frames of {o.capture} in {o.out}')


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
    if any(l.kernel == 'rr2_enc0_kernel' for l in cap.launches):
        return preset_f(cap, o)
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
        out_images = {i.offset: i for i in cap.find('hkpn_output', f).images}
        for name, off in OPTIONAL_OUTPUT.items():   # DLSSD.ColorBeforeParticles
            if off in out_images:
                (o.out / f'{f}-{name}.bin').write_bytes(out_images[off].path.read_bytes())
        clip = mat(660)
        view = mat(596) @ np.linalg.inv(clip)
        view[3, :3] = 0.0
        jx, jy = f32(320), f32(324)
        mvx, mvy = f32(344) * rw / ow, f32(348) * rh / oh
        reset = 1 if f == 1 or not o.no_reset else 0
        values = ' '.join(f'{v:.9g}' for v in list(view.flatten()) + list(clip.flatten()))
        lines.append(f'frame {f} {jx:.9g} {jy:.9g} {mvx:.9g} {mvy:.9g} 1 {reset} {u32(528) | (a[980] << 1)} {values}')
    (o.out / 'frames.txt').write_text('\n'.join(lines) + '\n')
    print(f'{o.frames} frames of {o.capture} in {o.out}')


if __name__ == '__main__':
    main()
