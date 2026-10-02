"""Compare resident native modules with exact replay and paired batch timings.

Fixtures can contain private weights; this utility never packages them.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from profile_report import report


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(passed=False, numericalExact=False, fixture=str(args.fixture_dir),
                   module=str(args.module), candidate=str(args.candidate), probe=str(args.probe),
                   pairs=args.pairs, batches=args.batches,
                   timingSources=args.timings, executions=[])
    environment = os.environ.copy()
    environment['D4R_DIAG_DIR'] = str(args.output_dir)
    environment['PATH'] = str(args.hip_root / 'bin') + os.pathsep + environment.get('PATH', '')
    try:
        summary.update(moduleSha256=sha(args.module), candidateSha256=sha(args.candidate),
                       probeSha256=sha(args.probe))
        def execute(name, module, options=()):
            command = [str(args.probe), '--hip-root', str(args.hip_root), '--module', str(module),
                       '--fixture-dir', str(args.fixture_dir), '--output-dir', str(args.output_dir / name),
                       '--iterations', '1', *options]
            with (args.output_dir / (name + '.stdout.log')).open('wb') as out, \
                    (args.output_dir / (name + '.stderr.log')).open('wb') as err:
                try:
                    completed = subprocess.run(command, stdout=out, stderr=err, env=environment, timeout=180)
                except subprocess.TimeoutExpired:
                    summary['executions'].append(dict(name=name, command=command, timedOut=True))
                    raise
            summary['executions'].append(dict(name=name, command=command, exitCode=completed.returncode))
            if completed.returncode:
                raise RuntimeError(f'{name} failed ({completed.returncode}); inspect preserved logs/dump')
            return args.output_dir / name

        control = execute('control-reference', args.module)
        candidate = execute('candidate-reference', args.candidate)
        reference = {p.name: sha(p) for p in control.glob('alloc-*.bin')}
        if not reference:
            raise RuntimeError('Replay produced no allocations')

        def check(folder):
            actual = {p.name: sha(p) for p in folder.glob('alloc-*.bin')}
            if actual != reference:
                changed = sorted(set(actual) ^ set(reference) |
                                 {name for name in reference if actual.get(name) != reference[name]})
                raise RuntimeError(f'{folder.name}: allocations differ from fresh control: {changed}')

        # Stop before timing a candidate that changes the numerical result.
        check(candidate)
        summary['numericalExact'] = True
        summary['allocationSha256'] = reference
        for timing in args.timings:
            for batch in args.batches:
                folder = execute(f'{timing}-{batch}', args.module,
                                 ['--benchmark-module', str(args.candidate), '--benchmark-timing', timing,
                                  '--benchmark-batch', str(batch), '--iterations', str(args.pairs)])
                check(folder)
        measured = report(args.output_dir)
        summary['profile'] = measured
        expected = len(args.timings) * len(args.batches)
        if len(measured['replayPairs']) != expected or any(
                row['controlGpuMs']['samples'] != args.pairs for row in measured['replayPairs']):
            raise RuntimeError('Incomplete/invalid timing pairs; this run is not accepted for performance comparison')
        summary['passed'] = True
        for row in measured['replayPairs']:
            print(f"PAIRED {row['kernel']} {row['timingSource']} batch={row['launchesPerSample']} "
                  f"control={row['controlGpuMs']['median']:.6f} candidate={row['candidateGpuMs']['median']:.6f} ms "
                  f"ratio={row['candidateToControl']['median']:.6f}")
        print('PASS NATIVE_BENCHMARK exact_allocations=1 timings_valid=1 '
              'isolated_replay=1 game_speedup=unmeasured')
    except Exception as error:
        summary['error'] = str(error)
        print(f'FAIL NATIVE_BENCHMARK {error}', file=sys.stderr)
    finally:
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return 0 if summary['passed'] else 5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'hip-root', 'module', 'candidate', 'fixture-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=lambda s: pathlib.Path(s).resolve())
    parser.add_argument('--pairs', type=int, default=32)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 64])
    parser.add_argument('--timings', choices=['events', 'dispatch'], nargs='+', default=['events', 'dispatch'])
    args = parser.parse_args()
    if not 1 <= args.pairs <= 10000 or any(not 1 <= batch <= 256 for batch in args.batches):
        parser.error('pairs must be 1..10000 and batches 1..256')
    if len(set(args.batches)) != len(args.batches) or len(set(args.timings)) != len(args.timings):
        parser.error('duplicate batches/timing sources are not allowed')
    raise SystemExit(run(args))
