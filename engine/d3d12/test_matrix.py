#!/usr/bin/env python3
"""Validate and benchmark AMD WMMA against the same Vulkan cooperative-matrix tile.

Both kernels compute C + STEPS * A@B with f16 A/B and f32 accumulation.
Inputs are unique per tile. Outputs must match a float64 oracle, and unused
output bytes must remain zero. No model weights or AGS DLL are required.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent


def timing(stdout):
    match = re.search(r'([0-9.]+) us per dispatch', stdout)
    if not match or float(match[1]) <= 0:
        raise ValueError('runner did not report a positive GPU time')
    return float(match[1])


def check_output(path, expected, steps, trailing_zero=False):
    result = np.fromfile(path, np.float32)
    if trailing_zero:
        if result.size != 2*expected.size or np.any(result[expected.size:] != 0):
            raise ValueError(f'{path}: wrong output size or writes outside the tile array')
        result = result[:expected.size]
    if result.size != expected.size or not np.isfinite(result).all():
        raise ValueError(f'{path}: wrong output size or nonfinite results')
    result = result.reshape(expected.shape).astype(np.float64)
    # Account for f32 accumulation over repeated products. Small tile values
    # keep the bound useful; a missing operation, bad layout or ignored C fails.
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=max(1e-6, steps*2e-6))
    return result, float(np.max(np.abs(result-expected)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--vulkan-runner', type=Path)
    p.add_argument('--tiles', type=int, default=1)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--dxc', default='dxc')
    p.add_argument('--glslang', default='glslangValidator')
    p.add_argument('--launch-prefix', nargs='+', default=[])
    p.add_argument('--vulkan-launch-prefix', nargs='+', default=[])
    a = p.parse_args()
    if not 1 <= a.tiles <= 65535 or not 1 <= a.steps <= 1024 or not 2 <= a.repeats <= 10000:
        p.error('tiles must be 1..65535, steps 1..1024, repeats 2..10000')
    a.work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    shape = (a.tiles, 16, 16)
    left = rng.normal(0, .2, shape).astype(np.float16)
    right = rng.normal(0, .2, shape).astype(np.float16)
    accum = rng.normal(0, .1, shape).astype(np.float32)
    expected = accum.astype(np.float64) + a.steps*(left.astype(np.float64) @ right.astype(np.float64))
    data = b''.join(x.tobytes()+y.tobytes()+c.tobytes() for x, y, c in zip(left, right, accum))
    (a.work/'input.bin').write_bytes(data)
    (a.work/'params.bin').write_bytes(bytes(448))  # unused layer-runner binding
    (a.work/'weights.bin').write_bytes(bytes(4))  # unused layer-runner binding
    expected.astype(np.float32).tofile(a.work/'reference.bin')
    shader = a.work/'matrix.dxil'
    subprocess.run([a.dxc, '-T', 'cs_6_6', '-E', 'main', '-enable-16bit-types', '-O3',
                    f'-DMATRIX_STEPS={a.steps}', str(HERE/'matrix_tile.hlsl'), '-Fo', str(shader)], check=True)

    def windows_path(path):
        path = str(path.resolve())
        return 'Z:'+path.replace('/', '\\') if a.launch_prefix and os.name != 'nt' else path

    cmd = [*a.launch_prefix, str(a.runner.resolve()), windows_path(shader), windows_path(a.work/'params.bin'),
           windows_path(a.work/'weights.bin'), windows_path(a.work/'input.bin'), windows_path(a.work/'d3d12.bin'),
           str(a.tiles), '1', str(a.repeats), '--wmma']
    if a.debug:
        cmd.append('--debug')
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(completed.stdout, end='')
    if completed.stderr:
        print(completed.stderr, end='')
    d3d12, error = check_output(a.work/'d3d12.bin', expected, a.steps, trailing_zero=True)
    report = dict(tiles=a.tiles, steps=a.steps, repeats=a.repeats, debug_layer=a.debug,
                  arithmetic='f16 operands, f32 accumulation, C + steps * A@B',
                  d3d12_gpu_us=timing(completed.stdout), d3d12_max_abs_error=error,
                  d3d12_stdout=completed.stdout)
    if a.vulkan_runner:
        subprocess.run([a.glslang, '-V', '--target-env', 'vulkan1.3', f'-DMATRIX_STEPS={a.steps}',
                        str(HERE/'matrix_tile.comp'), '-o', str(a.work/'matrix.spv')], check=True)
        env = dict(os.environ, D4R_EVK_GPU_TIMING='1')
        vk = subprocess.run([*a.vulkan_launch_prefix, str(a.vulkan_runner.resolve()), 'matrix.spv', str(a.tiles), '1',
                             str(a.repeats), 'in:0=input.bin', f'out:1=vulkan.bin:{expected.size*4}'],
                            cwd=a.work, env=env, check=True, capture_output=True, text=True)
        print(vk.stdout, end='')
        if vk.stderr:
            print(vk.stderr, end='')
        vulkan, error = check_output(a.work/'vulkan.bin', expected, a.steps)
        np.testing.assert_allclose(d3d12, vulkan, rtol=2e-5, atol=max(1e-6, a.steps*2e-6))
        report.update(vulkan_gpu_us=timing(vk.stdout), vulkan_max_abs_error=error,
                      vulkan_stdout=vk.stdout, cross_api_max_abs_error=float(np.max(np.abs(d3d12-vulkan))))
        report['d3d12_over_vulkan'] = report['d3d12_gpu_us']/report['vulkan_gpu_us']
    (a.work/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('_stdout')}, indent=2))


if __name__ == '__main__':
    main()
