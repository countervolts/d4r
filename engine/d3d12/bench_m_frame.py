#!/usr/bin/env python3
"""Interleave complete native M/L replay trials using existing raw-frame inputs.

Includes every inference stage and recurrent frame in each timed GPU batch.
Excludes uploads, readbacks, CPU submission and waits. Close GPU applications
before running. This is a replay measurement, not gameplay or FPS.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def exclusive_benchmark():
    """Serialize this driver's GPU cohorts, including across work directories."""
    with (Path(tempfile.gettempdir()) / 'd4r-native-ml-benchmark.lock').open('a+b') as lock:
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError:
            raise RuntimeError('Another native M/L frame benchmark is running; run GPU cohorts serially.') from None
        try:
            yield
        finally:
            lock.seek(0)
            release()


def capture_provenance(runners, models, inputs):
    return dict(runners={str(p): digest(p) for p in runners},
                inputs={str(p): digest(p) for p in inputs},
                input_bytes={str(p): p.stat().st_size for p in inputs},
                models={str(p): {f.relative_to(p).as_posix(): digest(f) for f in sorted([*p.iterdir(), *(p / 'rdna3').glob('*')])
                        if f.suffix == '.dxil' or f.name in ('d3d12_ml.bin', 'd3d12_ml.json', 'ldr', 'model.json',
                                                            'offsets.bin', 'weights8.bin', 'weights16.bin')}
                        for p in models})


def run_benchmark():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    runners = parser.add_mutually_exclusive_group(required=True)
    runners.add_argument('--runner', type=Path)
    runners.add_argument('--runners', nargs='+', type=Path, help='one runner per model, for runtime schedule comparisons')
    parser.add_argument('--models', required=True, nargs='+', type=Path)
    parser.add_argument('--backends', nargs='+', choices=('vulkan', 'hip', 'vkhip'),
                        help='one replay backend per model (default: vulkan); the same model with different '
                             'backends is compared by label MODEL-BACKEND')
    parser.add_argument('--inputs', required=True, nargs=3, type=Path, metavar=('COLOR', 'MOTION', 'DEPTH'))
    parser.add_argument('--render', nargs=2, type=int, default=(1280, 720))
    parser.add_argument('--output', nargs=2, type=int, default=(2560, 1440))
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=25)
    args = parser.parse_args()
    trial_runners = args.runners or [args.runner] * len(args.models)
    if len(trial_runners) != len(args.models):
        parser.error('--runners must contain one runner per model')
    backends = args.backends or ['vulkan'] * len(args.models)
    if len(backends) != len(args.models):
        parser.error('--backends must contain one backend per model')
    labels = [f'{model.name}-{backend}' for model, backend in zip(args.models, backends)]
    if len(set(labels)) != len(labels):
        parser.error('MODEL-BACKEND labels must be unique')
    if not 1 <= args.trials <= 20 or not 1 <= args.repeats <= 10000:
        parser.error('trials must be in [1,20] and repeats in [1,10000]')
    if any(d < 64 or d > 16384 for d in (*args.render, *args.output)):
        parser.error('dimensions must be in [64,16384]')
    for path, texel_bytes in zip(args.inputs, (8, 4, 4)):
        expected = args.render[0] * args.render[1] * texel_bytes
        try:
            actual = path.stat().st_size
        except OSError as error:
            parser.error(str(error))
        if actual != expected:
            parser.error(f'{path}: expected {expected} bytes for render {args.render}, got {actual}')
    args.work.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, D4R_ENGINE_PROFILE='1')
    for key in ('D4R_M_RESET_AT', 'D4R_M_TEST_DISCARD', 'D4R_M_DISPLAY_MV', 'D4R_M_CAPTURE_DIR'):
        environment.pop(key, None)
    provenance = capture_provenance(trial_runners, args.models, args.inputs)
    initial_provenance = provenance.copy()
    if args.runner is not None:
        provenance['runner_sha256'] = digest(args.runner)
    samples = []
    summary_path = args.work / 'summary.json'
    summary_path.write_text(json.dumps(dict(status='running', render=args.render, output=args.output,
                                           provenance=provenance, samples=samples), indent=2))
    for trial in range(args.trials):
        offset = trial % len(args.models)
        order = list(range(len(args.models)))
        order = order[offset:] + order[:offset]
        for index in order:
            model = args.models[index]
            prefix = args.work / f'{trial}-{index}-{labels[index]}'
            command = [str(trial_runners[index]), str(model), *map(str, args.render), *map(str, args.output),
                       str(prefix), str(args.repeats)]
            if backends[index] != 'vulkan':
                command.append(backends[index])
            command += ['--', *map(str, args.inputs), '0', '0']
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            prefix.with_suffix('.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                summary_path.write_text(json.dumps(dict(status='invalid', reason='runner failed',
                                                       provenance=provenance, samples=samples), indent=2))
                raise RuntimeError(result.stdout + result.stderr)
            if capture_provenance(trial_runners, args.models, args.inputs) != initial_provenance:
                summary_path.write_text(json.dumps(dict(status='invalid', reason='benchmark artifacts changed',
                                                       provenance=provenance, samples=samples), indent=2))
                raise RuntimeError('Runner, model or input changed during the cohort; reject the entire cohort.')
            match = re.search(r'Complete native ([ML]): ([\d.]+)', result.stdout) or \
                re.search(r'(\d+) timed frames: median ([\d.]+) us', result.stdout)
            if not match:
                raise RuntimeError('replay did not report a complete-frame GPU timing')
            sample = dict(trial=trial, model=str(model), label=labels[index], backend=backends[index],
                          runner=str(trial_runners[index]), preset=match[1] if 'Complete' in match.group(0) else 'ML',
                          ms=float(match[2]) / 1000,
                          stages_ms={name: float(value) / 1000 for name, value in
                                     re.findall(r'\s(\w+)\s+median:?\s+([\d.]+)', result.stdout)},
                          output_sha256=digest(Path(str(prefix) + '-0.rgba16f')))
            samples.append(sample)
            print(json.dumps(sample), flush=True)
            summary = dict(status='running', render=args.render, output=args.output, repeats=args.repeats,
                           trials=args.trials, provenance=provenance, samples=samples,
                           medians_ms={label: statistics.median(s['ms'] for s in samples if s['label'] == label)
                                       for label in labels if any(s['label'] == label for s in samples)})
            summary_path.write_text(json.dumps(summary, indent=2))
    summary['status'] = 'complete'
    summary_path.write_text(json.dumps(summary, indent=2))


def main():
    with exclusive_benchmark():
        run_benchmark()


if __name__ == '__main__':
    main()
