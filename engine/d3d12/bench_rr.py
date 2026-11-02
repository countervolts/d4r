#!/usr/bin/env python3
"""Interleave native D3D12 RR model timings on one common replay case.

Performance repeats advance temporal history. Use compare_rr.py on separate
single-repeat replays for quality; the outputs here are not quality fixtures.
GPU clocks are not changed by this script. The replay idles between inputs and
the GPU takes about half a second of load to reach its boost clock, so short
repeat counts time ramping clocks (10 repeats read about 1.7x slow at 4K). The
defaults keep only each input's second half of 120 repeats.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

STAGES = ('exposure', 'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
          'dec4', 'dec3', 'dec2', 'dec1', 'dec0', 'hkpn')
SAMPLE = re.compile(r'^frame (\d+) repeat (\d+) us ([\d.eE+-]+) stages (.+)$', re.M)


def samples(log, frames, repeats, warmup):
    rows = {}
    for match in SAMPLE.finditer(log):
        key = int(match[1]), int(match[2])
        values = [float(match[3]), *map(float, match[4].split())]
        if key in rows or len(values) != len(STAGES) + 1:
            raise ValueError('duplicate or malformed RR timestamp row')
        if any(not math.isfinite(v) or v < 0 for v in values) or values[0] <= 0:
            raise ValueError('invalid RR GPU timestamp')
        rows[key] = values
    expected = {(frame, repeat) for frame in frames for repeat in range(repeats)}
    if rows.keys() != expected:
        raise ValueError('incomplete RR timing replay')
    return [values for (_, repeat), values in rows.items() if repeat >= warmup]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runner', 'baseline', 'candidate', 'case', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=120)
    parser.add_argument('--warmup', type=int, default=60)
    args = parser.parse_args()
    if args.trials < 1 or not 0 <= args.warmup < args.repeats:
        parser.error('require trials >= 1 and 0 <= warmup < repeats')
    frames = [int(parts[1]) for line in (args.case / 'frames.txt').read_text().splitlines()
              if (parts := line.split()) and parts[0] == 'frame']
    if not frames or len(set(frames)) != len(frames):
        parser.error('require nonempty, unique frame IDs')
    args.output.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('D4R_')}
    result = {'case': str(args.case.resolve()), 'repeats': args.repeats,
              'warmup_per_input': args.warmup, 'quality_replay': False,
              'models': {name: str(getattr(args, name).resolve()) for name in ('baseline', 'candidate')},
              'trials': [], 'summary': {}}
    for trial in range(args.trials):
        order = ('baseline', 'candidate') if trial % 2 == 0 else ('candidate', 'baseline')
        for name in order:
            log_path = args.output / f'{trial}-{name}.log'
            command = [str(args.runner.resolve()), str(getattr(args, name).resolve()),
                       str(args.case.resolve()), str((args.output / (name + '-frames')).resolve()),
                       str(args.repeats)]
            with log_path.open('w') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            data = samples(log_path.read_text(), frames, args.repeats, args.warmup)
            medians = [statistics.median(row[i] for row in data) for i in range(len(STAGES) + 1)]
            result['trials'].append({'trial': trial, 'model': name, 'command': command,
                                     'samples': len(data), 'median_us': medians[0],
                                     'stage_medians_us': dict(zip(STAGES, medians[1:])),
                                     'log': log_path.name})
            print(f'{trial} {name}: {medians[0] / 1000:.4f} ms', flush=True)
    for name in ('baseline', 'candidate'):
        values = [t['median_us'] for t in result['trials'] if t['model'] == name]
        result['summary'][name] = {'median_us': statistics.median(values),
                                   'trial_medians_us': values}
    base = result['summary']['baseline']['median_us']
    candidate = result['summary']['candidate']['median_us']
    result['speedup'] = base / candidate
    result['latency_reduction_percent'] = 100 * (1 - candidate / base)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{result["speedup"]:.3f}x; {result["latency_reduction_percent"]:.2f}% lower latency')


if __name__ == '__main__':
    main()
