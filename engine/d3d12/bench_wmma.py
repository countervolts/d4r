#!/usr/bin/env python3
"""Interleave WMMA ablations and optional native Vulkan on byte-identical fixtures.

Compile once per variant, check every trial against the independent CPU oracle,
require stable output within each variant, and save raw trials and source hashes.
The native Vulkan launch prefix is independent of --launch-prefix wine.
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
import numpy as np
from test_layer import (ENGINE, F16, WMMA_VARIANTS, compare, layer_params, reference,
                        synthetic, validate_model, wmma_defines)

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_time(stdout):
    match = re.search(r'([0-9.]+) us per dispatch \(GPU timestamps', stdout)
    if not match or float(match[1]) <= 0:
        raise ValueError(f'runner must report positive GPU timestamp timing: {stdout!r}')
    return float(match[1])


def summarize(trials):
    values = [t['gpu_us'] for t in trials]
    median = statistics.median(values)
    return dict(median_us=median, min_us=min(values), max_us=max(values),
                spread_percent=100*(max(values)-min(values))/median,
                stdev_us=statistics.stdev(values) if len(values)>1 else 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--vulkan-runner', type=Path)
    p.add_argument('--variants', nargs='+', choices=WMMA_VARIANTS, default=list(WMMA_VARIANTS))
    p.add_argument('--channels', type=int, choices=(64, 96, 128, 160), default=160)
    p.add_argument('--heads', type=int, choices=(2, 4, 6, 8), default=8)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--height', type=int, default=32)
    p.add_argument('--shift', type=int, nargs=2, default=(0, 0))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--trials', type=int, default=7)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--min-psnr', type=float, default=60)
    p.add_argument('--dxc', default='dxc')
    p.add_argument('--glslang', default='glslangValidator')
    p.add_argument('--launch-prefix', nargs='+', default=[])
    p.add_argument('--vulkan-launch-prefix', nargs='+', default=[])
    p.add_argument('--debug-validation', action='store_true', help='one separate debug-layer run per WMMA variant; timed trials always disable it')
    a = p.parse_args()
    sx, sy = a.shift
    if not 8 <= min(a.width, a.height) <= max(a.width, a.height) <= 4096 or not (0 <= sx < 8 and 0 <= sy < 8):
        p.error('dimensions must be 8..4096 and shifts 0..7')
    if not 1 <= a.trials <= 100 or not 2 <= a.repeats <= 10000:
        p.error('trials must be 1..100, repeats 2..10000')
    if not math.isfinite(a.min_psnr) or a.min_psnr <= 0 or len(set(a.variants)) != len(a.variants):
        p.error('min-psnr must be finite and positive; variants must be unique')
    a.work.mkdir(parents=True, exist_ok=True)
    e, weights, rng = synthetic(a.channels, a.heads, a.seed)
    validate_model(e, weights)
    inputs = rng.normal(0, .5, (a.height, a.width, a.channels)).astype(F16)
    (a.work/'params.bin').write_bytes(layer_params(e, a.width, a.height, sx, sy))
    weights.tofile(a.work/'weights.bin'); inputs.tofile(a.work/'input.bin')
    print('Computing CPU reference...', flush=True)
    expected = reference(inputs, e, weights, sx, sy)
    expected.tofile(a.work/'cpu-reference.bin')
    gx, gy = str(math.ceil((a.width+sx)/8)), str(math.ceil((a.height+sy)/8))
    commands, outputs, trials, compilation = {}, {}, {}, {}

    def winpath(path):
        path = str(path.resolve())
        return 'Z:'+path.replace('/', '\\') if a.launch_prefix and os.name != 'nt' else path

    for variant in a.variants:
        shader, listing = a.work/f'{variant}.dxil', a.work/f'{variant}.ll'
        cmd = [a.dxc, '-T', 'cs_6_6', '-E', 'main', '-enable-16bit-types', '-O3',
               f'-DC={a.channels}', f'-DNH={a.heads}', *wmma_defines(variant),
               str(HERE/'layer_k_wmma.hlsl'), '-Fo', str(shader), '-Fc', str(listing)]
        print(f'Compiling {variant}...', flush=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        barrier_modes = re.findall(r'call void @dx\.op\.barrier\(i32 80, i32 (\d+)\)', listing.read_text())
        compilation[variant] = dict(command=cmd, shader_sha256=sha256(shader),
            static_dxil_barriers={mode: barrier_modes.count(mode) for mode in sorted(set(barrier_modes))})
        out = a.work/f'{variant}.bin'
        commands[variant] = [*a.launch_prefix, str(a.runner.resolve()), winpath(shader), winpath(a.work/'params.bin'),
                             winpath(a.work/'weights.bin'), winpath(a.work/'input.bin'), winpath(out),
                             gx, gy, str(a.repeats), '--wmma']
        outputs[variant], trials[variant] = out, []
    if a.vulkan_runner:
        cmd = [a.glslang, '-V', '--target-env', 'vulkan1.3', f'-DC={a.channels}', f'-DNH={a.heads}',
               '-DKIND=0', '-DPOSATTN=0', '-DEMBED=0', '-DHEAD=0', '-DXC=16', '-DMERGE_N=16',
               str(ENGINE/'layer.comp'), '-o', str(a.work/'layer.spv')]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        compilation['vulkan'] = dict(command=cmd, shader_sha256=sha256(a.work/'layer.spv'))
        commands['vulkan'] = [*a.vulkan_launch_prefix, str(a.vulkan_runner.resolve()), 'layer.spv', gx, gy,
                              str(a.repeats), 'in:0=params.bin', 'in:1=weights.bin', 'in:2=input.bin',
                              'in:3=params.bin', f'out:4=vulkan.bin:{inputs.nbytes}', 'out:5=unused.bin:16']
        outputs['vulkan'], trials['vulkan'] = a.work/'vulkan.bin', []

    first = {}

    def run(name, debug=False):
        cmd = commands[name] + (['--debug'] if debug else [])
        env = dict(os.environ)
        if name == 'vulkan':
            env['D4R_EVK_GPU_TIMING'] = '1'
        done = subprocess.run(cmd, cwd=a.work if name == 'vulkan' else None,
                              env=env, check=True, capture_output=True, text=True)
        got = np.fromfile(outputs[name], F16).reshape(inputs.shape)
        accuracy = compare(expected, got, a.min_psnr)
        digest = sha256(outputs[name])
        if name in first and not np.array_equal(first[name].view(np.uint16), got.view(np.uint16)):
            raise ValueError(f'{name}: output changed between trials / validation run')
        if name not in first:
            first[name] = got.copy()
        return dict(gpu_us=gpu_time(done.stdout), cpu=accuracy, output_sha256=digest,
                    stdout=done.stdout, stderr=done.stderr, command=cmd)

    validation = {}
    if a.debug_validation:
        for name in a.variants:
            print(f'Debug validation: {name}', flush=True)
            validation[name] = run(name, debug=True)
    names, order = list(commands), []
    for trial in range(a.trials):
        # Rotate the starting kernel each round; every timing includes its own
        # warmup. Keep GPU runs sequential to avoid contention between runners.
        rotation = trial % len(names)
        sequence = names[rotation:] + names[:rotation]
        order.append(sequence)
        for name in sequence:
            result = run(name)
            result['round'] = trial
            trials[name].append(result)
            print(f'round {trial}: {name:10s} {result["gpu_us"]:.3f} us', flush=True)
    summary = {name: summarize(values) for name, values in trials.items()}
    if 'baseline' in summary:
        for name in summary:
            summary[name]['over_baseline'] = summary[name]['median_us']/summary['baseline']['median_us']
    cross = {name: dict(compare(first[names[0]], got, a.min_psnr),
                        bit_identical=bool(np.array_equal(first[names[0]].view(np.uint16), got.view(np.uint16))))
             for name, got in first.items()}
    sources = ['layer_k_wmma.hlsl', 'ags_wmma.hlsli', 'k_math.hlsli', 'replay_layer.cpp', 'test_layer.py', 'bench_wmma.py']
    report = dict(shape=dict(channels=a.channels, heads=a.heads, width=a.width, height=a.height, shift=a.shift),
                  seed=a.seed, repeats=a.repeats, timing='GPU timestamps; ordered writes; debug disabled',
                  fixtures={name: sha256(a.work/name) for name in ['params.bin', 'weights.bin', 'input.bin', 'cpu-reference.bin']},
                  sources={name: sha256(HERE/name) for name in sources},
                  compilation=compilation, debug_validation=validation, order=order, trials=trials,
                  cross_output=cross, cross_output_reference=names[0], summary=summary,
                  environment={key: os.environ[key] for key in ['VK_DRIVER_FILES', 'VK_LOADER_LAYERS_DISABLE',
                      'RADV_DEBUG', 'VKD3D_CONFIG', 'DXIL_SPIRV_CONFIG', 'WINEPREFIX', 'WINEDEBUG',
                      'WINEDLLOVERRIDES', 'VKD3D_SHADER_CACHE_PATH'] if key in os.environ})
    report['sources']['../layer.comp'] = sha256(ENGINE/'layer.comp')
    report['binaries'] = {'d3d12_runner': sha256(a.runner)}
    if a.vulkan_runner:
        report['binaries']['vulkan_runner'] = sha256(a.vulkan_runner)
    report['tool_versions'] = {}
    for name, cmd in [('dxc', [a.dxc, '--version']), ('glslang', [a.glslang, '--version'])]:
        if name == 'glslang' and not a.vulkan_runner:
            continue
        version = subprocess.run(cmd, capture_output=True, text=True)
        report['tool_versions'][name] = dict(returncode=version.returncode, stdout=version.stdout, stderr=version.stderr)
    (a.work/'ablation-result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
