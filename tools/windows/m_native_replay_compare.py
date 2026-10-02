"""Replay private M captures and require exact logical outputs against a validated baseline."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import traceback

import numpy as np
from m_capture_compare import logical, me, LAYERS
from m_replay_validate import metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(passed=False, strict=True, architecture='gfx1201', launches=[],
                   hipRoot=str(args.hip_root), modules={},
                   baselineValidation=json.loads(args.baseline_validation.read_text(encoding='utf-8-sig')))
    try:
        validation = summary['baselineValidation']
        if not (validation.get('passed') and validation.get('exact') and validation.get('captured_outputs')):
            raise RuntimeError('An exact captured-output NumPy validation is required for the baseline')
        audited = {row['capture'] for row in validation['launches']}
        captures = []
        for layer in ('enc1', 'enc2', 'enc3_tube', 'dec2', 'dec1'):
            paths = sorted(args.capture_dir.glob(f'replay-*-rrlite_{layer}_4x4'))
            if not paths:
                raise RuntimeError(f'Baseline has no {layer} captures')
            captures.extend((layer, path) for path in paths)
        for layer, capture in captures:
            if capture.name not in audited:
                raise RuntimeError(f'Capture was not included in exact baseline validation: {capture.name}')
            module = args.module_dir / f'rrlite_{layer}_4x4.hsaco'
            summary['modules'][module.name] = sha(module)
            output = args.output_dir / capture.name
            output.mkdir(exist_ok=True)
            env = dict(os.environ, D4R_DIAG_DIR=str(output.resolve()))
            with (output / 'gpu.stdout.log').open('wb') as stdout, (output / 'gpu.stderr.log').open('wb') as stderr:
                process = subprocess.run([str(args.probe.resolve()), '--hip-root', str(args.hip_root.resolve()),
                    '--module', str(module.resolve()), '--fixture-dir', str(capture.resolve()),
                    '--output-dir', str((output / 'gpu-output').resolve()), '--iterations', '1'],
                    stdout=stdout, stderr=stderr, env=env, timeout=90)
            row = dict(capture=capture.name, layer=layer, exitCode=f'0x{process.returncode & 0xffffffff:08x}', fields=[])
            summary['launches'].append(row)
            if process.returncode:
                raise RuntimeError(f'GPU replay failed: {capture.name} ({row["exitCode"]})')
            p = me.Params(str(capture))
            channels, _, merged, _ = LAYERS[layer]
            fields = [('output', p.out, p.W * p.H * channels)]
            if merged:
                fields.append(('merge', p.p48, (p.W // 2) * (p.H // 2) * merged))
            for name, pointer, count in fields:
                reference = logical(p, pointer, count, True, capture)
                actual = logical(p, pointer, count, True, output)
                if reference.size != count or actual.size != count:
                    raise RuntimeError(f'Truncated logical {name}: {capture.name}')
                different = int(np.count_nonzero(reference != actual))
                x, y = me.E4M3[reference], me.E4M3[actual]
                maximum, _, psnr, _ = metrics(x, y, f'M_NATIVE_COMPARE capture={capture.name} field={name}')
                row['fields'].append(dict(name=name, elements=count, differingCodes=different,
                    maxAbs=maximum, referenceSha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                    actualSha256=hashlib.sha256(actual.tobytes()).hexdigest()))
                if different:
                    raise RuntimeError(f'Exact baseline mismatch: {capture.name} {name}, codes={different}, PSNR={psnr}')
            row['passed'] = True
        if args.benchmark_control_dir:
            summary['benchmarks'] = []
            for layer in ('enc1', 'enc2', 'enc3_tube', 'dec2', 'dec1'):
                capture = next(path for item, path in captures if item == layer)
                output = args.output_dir / ('paired-' + layer)
                output.mkdir(exist_ok=True)
                control = args.benchmark_control_dir / f'rrlite_{layer}_4x4.hsaco'
                candidate = args.module_dir / control.name
                with (output / 'paired.stdout.log').open('wb') as stdout, (output / 'paired.stderr.log').open('wb') as stderr:
                    process = subprocess.run([str(args.probe.resolve()), '--hip-root', str(args.hip_root.resolve()),
                        '--module', str(control.resolve()), '--benchmark-module', str(candidate.resolve()),
                        '--fixture-dir', str(capture.resolve()), '--output-dir', str((output / 'output').resolve()),
                        '--iterations', '64'], stdout=stdout, stderr=stderr, timeout=90,
                        env=dict(os.environ, D4R_DIAG_DIR=str(output.resolve())))
                if process.returncode:
                    raise RuntimeError(f'Paired benchmark failed: {layer}, exit={process.returncode:#x}')
                from profile_report import report
                measured = report(output)
                if len(measured['replayPairs']) != 1 or measured['replayPairs'][0]['controlGpuMs']['samples'] != 64:
                    raise RuntimeError(f'Incomplete paired measurements: {layer}')
                summary['benchmarks'].extend(measured['replayPairs'])
                print(f'PAIRED M_NATIVE_COMPARE layer={layer} samples=64 logs={output}', flush=True)
        summary['passed'] = True
        print(f'PASS M_NATIVE_COMPARE launches={len(captures)} exact=1 finite=1 architecture=gfx1201', flush=True)
        return 0
    except Exception as error:
        summary['error'] = str(error)
        traceback.print_exc()
        return 1
    finally:
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-dir', 'module-dir', 'probe', 'hip-root', 'baseline-validation', 'output-dir'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    parser.add_argument('--benchmark-control-dir', type=pathlib.Path)
    raise SystemExit(main(parser.parse_args()))
