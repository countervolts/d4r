#!/usr/bin/env python3
"""Run ordinary linear HDR frames through the native engine without an NGX capture.

prepare_frames.py FRAMES.json CASE.txt
JSON has output: [width,height] and frames: [{color, motion, depth, render: [w,h],
bits: 16, color_exposure: 1, history_exposure_ratio: 1, network_exposure_scale: 1,
jitter: [0,0], motion_scale: [0,0], motion_offset: [0,0], sharpness: 1.37, reset: true}].
Paths are relative to the JSON. Color is interleaved RGBA16F/32F, motion RG16F/32F,
depth R32F (inverse depth). Subsequent frames keep the engine's own history.
The first frame resets. Attention windows default to a fixed four-token shift;
captured NGX window scheduling is available separately through prepare_k.py.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('frames', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    config = json.loads(a.frames.read_text())
    ow, oh = config['output']
    if not all(isinstance(x, int) and 256 <= x <= 8192 for x in (ow,oh)):
        raise ValueError('output dimensions must be integers in 256..8192')
    frames = config['frames']
    if not frames: raise ValueError('no frames')
    lines = [f'output {ow} {oh}']
    for i, frame in enumerate(frames):
        rw, rh = frame['render']; bits = frame.get('bits',16)
        if not isinstance(rw,int) or not isinstance(rh,int) or not 0 < rw <= ow or not 0 < rh <= oh or bits not in (16,32):
            raise ValueError('invalid render dimensions or component size')
        for j, (key, channels, b) in enumerate((('color',4,bits),('motion',2,bits),('depth',1,32))):
            path = (a.frames.resolve().parent / frame[key]).resolve()
            if path.stat().st_size != rw * rh * channels * b // 8:
                raise ValueError(f'{path}: unexpected frame image size')
            lines.append(f'image {i*3+j} {json.dumps(str(path))} {rw} {rh} {channels} {b}')
        reset = frame.get('reset', i == 0)
        if not isinstance(reset,bool): raise ValueError('reset must be a boolean')
        if i == 0 and not reset: raise ValueError('the first ordinary frame must reset')
        exposure = frame.get('color_exposure',1)
        ratio = frame.get('history_exposure_ratio',1)
        network = frame.get('network_exposure_scale',1)
        if not all(math.isfinite(x) and x > 0 for x in (exposure,ratio,network)):
            raise ValueError('exposure factors must be finite and positive')
        vectors = [frame.get(k,[0,0]) for k in ('jitter','motion_scale','motion_offset')]
        if any(len(v) != 2 or not all(math.isfinite(x) for x in v) for v in vectors):
            raise ValueError('jitter and motion vectors must contain two finite components')
        sharpness = frame.get('sharpness',1.37)
        if not math.isfinite(sharpness): raise ValueError('sharpness must be finite')
        values = [i*3,i*3+1,i*3+2,exposure,ratio,network,int(reset),*vectors[0],*vectors[1],*vectors[2],sharpness]
        lines.append('frame_auto ' + ' '.join(map(str,values)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text('\n'.join(lines)+'\n')
    print(f'Prepared {len(frames)} ordinary frames for {ow}x{oh} native inference')


if __name__ == '__main__': main()
