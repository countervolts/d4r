#!/usr/bin/env python3
"""Interleave the native Windows D3D12 and HIP matrix drafts on common fixtures.

Uses an independent float64 oracle, seeded nonzero f16 A/B and f32 C, and checks
every trial. This measures implementations, not an inherent API speed ranking.
The two backends order repeated writes: UAV barriers vs a dependent HIP graph.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--dxc', type=Path, required=True)
    p.add_argument('--arch', required=True, choices=['gfx1100', 'gfx1101', 'gfx1102', 'gfx1200', 'gfx1201'])
    p.add_argument('--sdk', type=Path)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--repeats', type=int, default=100)
    a = p.parse_args()
    if a.trials < 2 or not 2 <= a.repeats <= 10000:
        p.error('at least two trials and 2..10000 repeats required')
    a.work.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    report = dict(platform=platform.platform(), arch=a.arch, trials=a.trials, repeats=a.repeats,
                  scope='synthetic f16/f32 matrix tiles only; no model/frame/interop/game measurements',
                  runner_sha256=hashlib.sha256(a.runner.read_bytes()).hexdigest(),
                  shader_compiler=subprocess.run([str(a.dxc.resolve()), '--version'], capture_output=True, text=True, check=True).stdout,
                  cases=[])

    def run(command, log):
        result = subprocess.run(command, capture_output=True, text=True)
        log.write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:
            raise RuntimeError(f'command failed ({result.returncode}), see {log}')

    try:
        for tiles, steps in [(1, 1), (128, 1), (128, 128), (4096, 16)]:
            case = dict(tiles=tiles, steps=steps, trials=[])
            report['cases'].append(case)
            for trial in range(a.trials):
                work = a.work/f'tiles{tiles}-steps{steps}'/f'trial{trial}'
                work.mkdir(parents=True, exist_ok=True)
                dx = [sys.executable, str(here/'test_matrix.py'), str(work), '--runner', str(a.runner.resolve()),
                      '--dxc', str(a.dxc.resolve()), '--tiles', str(tiles), '--steps', str(steps), '--repeats', str(a.repeats)]
                hip = [sys.executable, str(here/'bench_hip.py'), str(work), '--arch', a.arch]
                if a.sdk:
                    hip += ['--sdk', str(a.sdk.resolve())]
                # First create the common fixture and validate D3D12. On odd
                # trials rerun D3D12 after HIP to alternate measurement order.
                run(dx, work/'d3d12.log')
                run(hip, work/'hip.log')
                if trial % 2:
                    run(dx, work/'d3d12.log')
                d = json.loads((work/'result.json').read_text())
                h = json.loads((work/'hip-result.json').read_text())
                case['trials'].append(dict(d3d12_us=d['d3d12_gpu_us'], hip_us=h['gpu_us'],
                                           cross_api_max_abs_error=h['cross_api_max_abs_error'], device=h['device']))
                print(f'{tiles} tiles / {steps} steps / trial {trial}: D3D12 {d["d3d12_gpu_us"]:.3f} us; HIP {h["gpu_us"]:.3f} us', flush=True)
            for backend in ('d3d12', 'hip'):
                values = [t[backend+'_us'] for t in case['trials']]
                case[backend] = dict(median_us=statistics.median(values), min_us=min(values), max_us=max(values))
            case['d3d12_over_hip'] = case['d3d12']['median_us']/case['hip']['median_us']
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        (a.work/'comparison.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
