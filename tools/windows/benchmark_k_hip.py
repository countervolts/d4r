"""Compare strict HIP K candidates against a fixed model on synthetic temporal frames.

Each candidate overlays only named code objects in a private model copy. All five
frame products must match the baseline hashes exactly. Raw files created by this
run are hashed and deleted immediately after each process; existing evidence is
never removed. Reported frame timestamps include Vulkan front/back and CPU gaps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / '.tools/python/vendor'))
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runner', 'model', 'case', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--candidate', action='append', default=[], metavar='LABEL=OBJECT_DIRECTORY')
    parser.add_argument('--repeats', type=int, default=25)
    args = parser.parse_args()
    if not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('output must stay under ignored build/')
    args.output.mkdir(parents=True, exist_ok=False)
    frame_count = sum(1 for line in args.case.read_text().splitlines()
                      if line.split() and line.split()[0] in ('frame', 'frame_auto'))
    if not frame_count:
        parser.error('case must contain explicit frame or frame_auto entries')
    expected_products = {f'frame-{i}.{suffix}.bin' for i in range(frame_count)
                         for suffix in ('312', '288', '296', '304', 'head')}
    env = {k: v for k, v in os.environ.items() if not k.startswith('D4R_')}
    env['PATH'] = str(repo / '.tools/therock-10.2.0a20260929/_rocm_sdk_core/bin') + os.pathsep + env['PATH']
    report = {'synthetic': True, 'gameAccessed': False, 'exactGate': 'all five frame products',
              'timingScope': 'Vulkan front + HIP network + Vulkan back, including CPU synchronization gaps',
              'runnerSha256': hashlib.sha256(args.runner.read_bytes()).hexdigest(),
              'caseSha256': hashlib.sha256(args.case.read_bytes()).hexdigest(), 'runs': []}
    reference = None
    variants = [('baseline', None)] + [tuple(s.split('=', 1)) for s in args.candidate]
    for label, source in variants:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', label):
            parser.error('candidate label must be letters, digits, underscores or hyphens')
        output = args.output / label
        output.mkdir()
        model = args.model
        row = {'label': label, 'objects': {}}
        if source:
            model = output / 'model'
            shutil.copytree(args.model, model)
            inputs = sorted(Path(source).resolve().glob('*.hsaco')) + sorted(Path(source).resolve().glob('*.spv'))
            for obj in inputs:
                destination = model / 'hip' / obj.name if obj.suffix == '.hsaco' else model / obj.name
                if not destination.is_file():
                    raise RuntimeError('candidate must replace an existing object: ' + obj.name)
                shutil.copy2(obj, destination)
                row['objects'][obj.name] = hashlib.sha256(obj.read_bytes()).hexdigest()
            if not row['objects']:
                raise RuntimeError('candidate contains no objects')
            manifest = json.loads((model / 'model.json').read_text())
            for name, digest in row['objects'].items():
                manifest['sha256'][name] = digest
            manifest['experimentalOverlay'] = row['objects']
            (model / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
        command = [str(args.runner), str(model), str(args.case), str(output / 'frames'), str(args.repeats), 'hip']
        with (output / 'stdout.log').open('wb') as out, (output / 'stderr.log').open('wb') as err:
            result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=240)
        row['exitCode'] = result.returncode
        hashes = {}
        finite = True
        for path in sorted((output / 'frames').glob('frame-*.bin')):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.name.endswith('.312.bin'):
                values = np.fromfile(path, '<f2').astype('f4')
                finite = finite and bool(np.isfinite(values).all() and np.var(values) > 1e-6)
            if not path.resolve().is_relative_to(args.output):
                raise RuntimeError('raw cleanup escaped the fresh output directory')
            path.unlink()
        row['hashes'] = hashes
        row['finiteNonuniformOutput'] = finite
        if reference is None:
            reference = hashes
        row['differentProducts'] = [k for k in sorted(set(hashes) | set(reference)) if hashes.get(k) != reference.get(k)]
        row['exact'] = result.returncode == 0 and set(hashes) == expected_products and not row['differentProducts'] and finite
        text = (output / 'stdout.log').read_text(errors='replace')
        timing = re.search(r'full native frame: ([0-9.]+) us GPU median \(min ([0-9.]+)\)', text)
        if timing:
            row['measuredMedianUs'] = float(timing[1])
            row['measuredMinUs'] = float(timing[2])
        report['runs'].append(row)
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'{label}: exact={row["exact"]} differing={len(row["differentProducts"])} median_us={row.get("measuredMedianUs")}', flush=True)
        if result.returncode:
            return 1
    report['passed'] = all(row['exact'] for row in report['runs'])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
