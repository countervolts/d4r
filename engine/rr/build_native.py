#!/usr/bin/env python3
"""Turn a preset-E Ray Reconstruction model directory (engine/compile_rr.py) into the native one: every Swin stage
from d4r's own shaders instead of translated PTX.

usage: build_native.py CAPTURE MODEL_DIR [--frame N] [--work DIR]

  enc0        engine/rr/enc0_m.comp, generated (gen_enc0_m.py) from the model's translated enc0.generated.comp,
              whose per-pixel feature gather it keeps; weights: NGX's image + enc0_pack.py's blob
  enc1-enc5,  engine/rr/swin_m.comp with swin_pack.py's per-stage header and weights
  dec4-dec1
  dec0        engine/rr/dec0_m.comp with dec0_pack.py's weights
The weights are extracted from ZLUDA replays of the capture's launches (engine/rr/phase_dump.py: every mma's operands
of one window) and the stages' weight images; extraction checks itself against the dump. Like the model directory,
everything written is derived from the user's DLL and stays on the machine. Runs one replay or compile at a time
(phase_dump.py caps its replay at 24 GB).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rr_capture import Capture  # noqa: E402
import swin_ref  # noqa: E402
import dec0_ref  # noqa: E402
import enc0_ref  # noqa: E402

SWIN = ['enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4', 'dec3', 'dec2', 'dec1']


def run(*cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def glsl(source, output, *includes):
    run('glslangValidator', '-V', '--target-env', 'vulkan1.3', *[f'-I{i}' for i in includes], source, '-o', output,
        stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('model')
    ap.add_argument('--frame', type=int, default=2)
    ap.add_argument('--work', help='dumps and build files (default MODEL/native-work)')
    o = ap.parse_args()
    model = Path(o.model)
    work = Path(o.work) if o.work else model / 'native-work'
    work.mkdir(parents=True, exist_ok=True)
    cap = Capture(os.path.expanduser(o.capture))
    manifest = (model / 'rr.txt').read_text().split('\n')

    def dump(stage):
        out = work / f'{stage}_all.npz'
        launch = cap.find(f'swin_{stage}', o.frame)
        block = '5,5' if launch.grid[0] > 5 and launch.grid[1] > 5 else '1,1'
        if not out.exists():
            run(sys.executable, HERE / 'phase_dump.py', o.capture, stage, o.frame, block, out)
        return out, block

    def stage_line(stage, shader, weights):
        for i, line in enumerate(manifest):
            f = line.split()
            if len(f) > 4 and f[0] == 'stage' and f[1] == stage:
                f[2], f[4] = shader, weights
                manifest[i] = ' '.join(x for x in f if x != 'noopt')

    for name in ('rr_exact.glsl', 'rr_attn.glsl', 'rr_decfilter.glsl', 'enc0_m_decl.glsl', 'enc0_m_tail.glsl'):
        shutil.copy(HERE / name, work / name)
    for stage in SWIN:
        d, block = dump(stage)
        print(f'{stage}: extracting', flush=True)
        l, ex, ey, sx, sy, inp, outs, tex = swin_ref.load(o.capture, stage, o.frame)
        bx, by = (int(v) for v in block.split(','))
        swin_ref.extract(stage, d, l.weights, work / f'{stage}_w.npz', inp, 8 * bx - sx, 8 * by - sy, ex, ey)
        sd = work / stage
        sd.mkdir(exist_ok=True)
        run(sys.executable, HERE / 'swin_pack.py', stage, work / f'{stage}_w.npz', model / f'weights-{stage}m.bin',
            sd / 'swin_stage.h')
        glsl(HERE / 'swin_m.comp', model / f'{stage}_m.spv', sd, work)
        stage_line(stage, f'{stage}_m.spv', f'weights-{stage}m.bin')
    # dec0
    d, _ = dump('dec0')
    print('dec0: extracting', flush=True)
    dec0_ref.extract(d, cap.find('swin_dec0', o.frame).weights, work / 'dec0_w.npz')
    run(sys.executable, HERE / 'dec0_pack.py', work / 'dec0_w.npz', model / 'weights-dec0m.bin', work / 'dec0_m_w.h')
    glsl(HERE / 'dec0_m.comp', model / 'dec0_m.spv', work)
    stage_line('dec0', 'dec0_m.spv', 'weights-dec0m.bin')
    # enc0: NGX's weight image (the translated feature gather reads it) followed by the native blob
    d, _ = dump('enc0')
    print('enc0: extracting', flush=True)
    nv = cap.find('swin_enc0', o.frame).weights
    enc0_ref.extract(d, nv, work / 'enc0_w.npz')
    base = (len(nv) + 255) // 256 * 256
    run(sys.executable, HERE / 'enc0_pack.py', work / 'enc0_w.npz', work / 'enc0_native.bin', work / 'enc0_m_w.h', base)
    (model / 'weights-enc0m.bin').write_bytes(nv + bytes(base - len(nv)) + (work / 'enc0_native.bin').read_bytes())
    run(sys.executable, HERE / 'gen_enc0_m.py', model / 'enc0.generated.comp', work / 'enc0_m.comp')
    glsl(work / 'enc0_m.comp', model / 'enc0_m.spv', work)
    stage_line('enc0', 'enc0_m.spv', 'weights-enc0m.bin')
    (model / 'rr.txt').write_text('\n'.join(manifest))
    print(f'{model / "rr.txt"}: native stages written')


if __name__ == '__main__':
    main()
