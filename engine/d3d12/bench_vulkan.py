#!/usr/bin/env python3
"""Benchmark the existing Vulkan K shader (engine/layer.comp) on its own.

Generates exactly the fixtures test_layer.py writes for --synthetic (params.bin,
row-major weights-row.bin, input.bin), so no D3D12 runner, DXC or Wine is needed.
Each trial launches evk with D4R_EVK_GPU_TIMING=1. evk dispatches once for the
output, then runs 12 batches of REPEATS dispatches, discards 3 as warmup and
prints the median/min of the rest. Every trial's output is checked against the
NumPy oracle used by test_layer.py, and all trials must be bit-identical.

Example (Linux, from the repository root):
  g++ -O2 -std=c++17 engine/evk.cpp -lvulkan -o build/d4r-vulkan-layer
  python engine/d3d12/bench_vulkan.py build/vk-bench --vulkan-runner build/d4r-vulkan-layer \
      --synthetic 160 --heads 8 --width 32 --height 32 --repeats 100 --trials 7
The same work directory can then be passed to test_layer.py --backend wmma for
the D3D12 comparison, which regenerates byte-identical fixtures from the seed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_layer import ENGINE, F16, compare, layer_params, reference, synthetic, validate_model

DEFINES = ['-DKIND=0', '-DPOSATTN=0', '-DEMBED=0', '-DHEAD=0', '-DXC=16', '-DMERGE_N=16']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('work', type=Path)
    p.add_argument('--vulkan-runner', type=Path, required=True)
    p.add_argument('--synthetic', type=int, choices=(32, 64, 96, 128, 160), default=160, metavar='C')
    p.add_argument('--heads', type=int, choices=(2, 4, 6, 8), default=8)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--height', type=int, default=32)
    p.add_argument('--shift', type=int, nargs=2, default=(0, 0))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--trials', type=int, default=7)
    p.add_argument('--min-psnr', type=float, default=60)
    p.add_argument('--glslang', default='glslangValidator')
    a = p.parse_args()
    sx, sy = a.shift
    if min(a.width, a.height) < 8 or max(a.width, a.height) > 4096 or not (0 <= sx < 8 and 0 <= sy < 8):
        p.error('dimensions must be 8..4096, shifts 0..7')
    if not 2 <= a.repeats <= 10000 or not 1 <= a.trials <= 100:
        p.error('repeats must be 2..10000 and trials 1..100')
    a.work.mkdir(parents=True, exist_ok=True)

    # Same generation order as test_layer.py: weights first, then inputs from the same RNG.
    e, weights, rng = synthetic(a.synthetic, a.heads, a.seed)
    inputs = rng.normal(0, .5, (a.height, a.width, e['C'])).astype(F16)
    validate_model(e, weights)
    (a.work/'params.bin').write_bytes(layer_params(e, a.width, a.height, sx, sy))
    weights.tofile(a.work/'weights-row.bin')
    inputs.tofile(a.work/'input.bin')
    (a.work/'model.json').write_text(json.dumps({'dec5': e}, indent=2))
    expected = reference(inputs, e, weights, sx, sy)
    expected.tofile(a.work/'cpu-reference.bin')

    glslang = [a.glslang, '-V', '--target-env', 'vulkan1.3', f'-DC={e["C"]}', f'-DNH={e["H"]}', *DEFINES,
               str(ENGINE/'layer.comp'), '-o', str(a.work/'layer.spv')]
    subprocess.run(glslang, check=True, capture_output=True)
    gx, gy = str(math.ceil((a.width+sx)/8)), str(math.ceil((a.height+sy)/8))
    env = dict(os.environ, D4R_EVK_GPU_TIMING='1')
    trials, first = [], None
    for t in range(a.trials):
        out = f'vulkan-{t}.bin'
        cmd = [str(a.vulkan_runner.resolve()), 'layer.spv', gx, gy, str(a.repeats), 'in:0=params.bin',
               'in:1=weights-row.bin', 'in:2=input.bin', 'in:3=params.bin', f'out:4={out}:{inputs.nbytes}',
               'out:5=unused.bin:16']
        run = subprocess.run(cmd, cwd=a.work, env=env, check=True, capture_output=True, text=True)
        m = re.search(r'([0-9.]+) us per dispatch \(GPU timestamps.*min ([0-9.]+)\)', run.stdout)
        if not m:
            raise ValueError(f'runner did not report GPU timing: {run.stdout!r} {run.stderr!r}')
        got = np.fromfile(a.work/out, F16).reshape(inputs.shape)
        if first is None:
            first = got
        identical = bool(np.array_equal(first.view(np.uint16), got.view(np.uint16)))
        if not identical:
            raise ValueError(f'trial {t} output differs from trial 0')
        trials.append(dict(trial=t, median_us=float(m[1]), min_us=float(m[2]), identical_to_trial0=identical,
                           stdout=run.stdout.strip()))
        print(run.stdout, end='')
    medians = [t['median_us'] for t in trials]
    result = dict(
        kernel='engine/layer.comp (cooperative matrix)', channels=e['C'], heads=e['H'], width=a.width,
        height=a.height, shift=a.shift, seed=a.seed, repeats=a.repeats, groups=[int(gx), int(gy)],
        glslang_command=glslang, runner_command=cmd, timing_env={'D4R_EVK_GPU_TIMING': '1'},
        fixtures={n: sha256(a.work/n) for n in ('params.bin', 'weights-row.bin', 'input.bin', 'layer.spv',
                                                 'cpu-reference.bin')},
        cpu=compare(expected, first, a.min_psnr), trials=trials,
        summary=dict(median_of_trial_medians_us=statistics.median(medians), min_us=min(t['min_us'] for t in trials),
                     max_trial_median_us=max(medians), min_trial_median_us=min(medians),
                     stdev_trial_medians_us=statistics.stdev(medians) if len(medians) > 1 else 0.0))
    result['summary']['spread_percent'] = 100*(max(medians)-min(medians))/statistics.median(medians)
    (a.work/'vulkan-result.json').write_text(json.dumps(result, indent=2))
    print('CPU oracle:', json.dumps(result['cpu']))
    print('Summary:', json.dumps(result['summary']))


if __name__ == '__main__':
    main()
