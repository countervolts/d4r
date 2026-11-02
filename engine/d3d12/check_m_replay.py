#!/usr/bin/env python3
"""Check fresh-process M/L first-frame determinism, optionally capturing stages.

Each process creates a fresh feature and evaluates the same reset frame once.
This catches intermittent output mismatches; it is not an image-quality oracle.
Only the first output and mismatching outputs are retained, plus all small logs.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--runner', required=True, type=Path)
    parser.add_argument('--models', required=True, nargs='+', type=Path)
    parser.add_argument('--inputs', required=True, nargs=3, type=Path)
    parser.add_argument('--render', nargs=2, type=int, default=(1280, 720))
    parser.add_argument('--output', nargs=2, type=int, default=(2560, 1440))
    parser.add_argument('--trials', type=int, default=64)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    if not 2 <= args.trials <= 10000:
        parser.error('trials must be in [2,10000]')
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    baseline, samples = {}, []
    environment = dict(os.environ)
    for key in ('D4R_ENGINE_PROFILE', 'D4R_M_RESET_AT', 'D4R_M_TEST_DISCARD', 'D4R_M_DISPLAY_MV', 'D4R_M_CAPTURE_DIR'):
        environment.pop(key, None)
    provenance = dict(runner_sha256=digest(args.runner), inputs={str(p): digest(p) for p in args.inputs},
                      shaders={str(p): {f.relative_to(p).as_posix(): digest(f) for f in sorted(p.glob('rdna3/*.dxil'))} for p in args.models})
    for trial in range(args.trials):
        for index, model in enumerate(args.models):
            case = work / f'{trial}-{index}-{model.name}'
            case.mkdir(exist_ok=False)
            prefix = case / 'result'
            env = dict(environment)
            if args.capture:
                env['D4R_M_CAPTURE_DIR'] = str(case / 'capture')
            result = subprocess.run([str(args.runner), str(model), *map(str, args.render), *map(str, args.output),
                                     str(prefix), '--', *map(str, args.inputs), '0', '0'],
                                    env=env, capture_output=True, text=True)
            (work / f'{trial}-{index}.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            output = Path(str(prefix) + '-0.rgba16f')
            hashes = {'output': digest(output)}
            if args.capture:
                capture = case / 'capture' / 'frame-0'
                if not (capture / 'output.bin').exists():
                    raise RuntimeError('runner did not save the requested capture')
                hashes.update({f.name: digest(f) for f in capture.glob('*.bin')})
                if hashes['output'] != hashes['output.bin']:
                    raise AssertionError('independent output readbacks disagree')
            if index not in baseline:
                baseline[index] = hashes
            changed = [name for name in hashes if hashes[name] != baseline[index][name]]
            sample = dict(trial=trial, model=str(model), hashes=hashes, changed=changed)
            samples.append(sample)
            counts = {str(m): dict(Counter(s['hashes']['output'] for s in samples if s['model'] == str(m)))
                      for m in args.models}
            (work / 'summary.json').write_text(json.dumps(dict(provenance=provenance, samples=samples,
                                                               output_counts=counts), indent=2))
            if changed or trial % 8 == 0:
                print(json.dumps(dict(trial=trial, model=str(model), changed=changed)), flush=True)
            if trial and not changed:
                if not case.resolve().is_relative_to(work) or case.resolve() == work:
                    raise AssertionError('generated case escaped the diagnostic directory')
                shutil.rmtree(case)
    print(json.dumps(counts, indent=2))
    if any(len(hashes) != 1 for hashes in counts.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
