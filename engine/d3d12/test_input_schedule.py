#!/usr/bin/env python3
"""GPU regression: four-thread K input versus the original token schedule.

test_input_schedule.py MODEL_DIR RUNNER WORK_DIR
Requires a packaged HDR or LDR K model, DXC, and NumPy. No NVIDIA runtime is
loaded. Checks exact recurrent output/history agreement across a window cycle,
motion, jitter, disocclusion, and a mid-sequence reset. Images are synthetic;
this is a scheduling regression check, not a game-quality measurement.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hlsl_build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('runner', type=Path)
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve()
    ldr = json.loads((model / 'model.json').read_text())['variant'] == 'ldr'
    reference = work / 'reference-model'
    shutil.copytree(model, reference, dirs_exist_ok=True)
    defines = ['-DD4R_NATIVE_HALF=1'] + (['-DLDR=1'] if ldr else [])
    hlsl_build.build(Path(hlsl_build.HERE) / 'input_k.comp', reference / 'input_k.dxil', defines, wave=32)
    w, h = 426, 240
    rng = np.random.default_rng(20261022)
    color = rng.random((h, w, 4), dtype=np.float32).astype('<f2')
    color[:, :, 3] = 1
    color.tofile(work / 'color.bin')
    motion = rng.uniform(-3, 3, (h, w, 2)).astype('<f2')
    motion[:8, :, :] = 400  # history outside the active image
    motion.tofile(work / 'motion.bin')
    rng.uniform(0.1, 1, (h, w)).astype('<f4').tofile(work / 'depth.bin')
    lines = ['output 640 360']
    for i, (name, channels, bits) in enumerate([('color', 4, 16), ('motion', 2, 16), ('depth', 1, 32)]):
        lines.append(f'image {i} "{(work / (name + ".bin")).as_posix()}" {w} {h} {channels} {bits}')
    for i in range(16):
        lines.append(f'frame_auto 0 1 2 1 1 1 {int(i in (0, 8))} {(i % 4 - 1.5) / 4} '
                     f'{(i % 3 - 1) / 3} 1 1 0 0 1.37')
    case = work / 'case.txt'
    case.write_text('\n'.join(lines) + '\n')
    for label, package in [('reference', reference), ('candidate', model)]:
        result = subprocess.run([str(args.runner.resolve()), str(package), str(case), str(work / label), '1'],
                                capture_output=True, text=True)
        (work / f'{label}.log').write_text(result.stdout + result.stderr)
        result.check_returncode()
    checked = 0
    for path in sorted((work / 'reference').glob('*.bin')):
        other = work / 'candidate' / path.name
        a = np.fromfile(path, dtype='<u2')
        b = np.fromfile(other, dtype='<u2')
        if not a.size or not np.array_equal(a, b) or not np.isfinite(b.view('<f2')).all():
            raise AssertionError(f'{path.name}: size, finite-value or bitwise agreement failure')
        checked += 1
    if checked != 16 * 6:
        raise AssertionError(f'missing results: {checked} buffers')
    print(f'{"LDR" if ldr else "HDR"}: {checked} buffers finite and bit-identical')


if __name__ == '__main__':
    main()
