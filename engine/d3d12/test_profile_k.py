#!/usr/bin/env python3
"""Replay identical K data with normal and Cyberpunk executable identities.

AMD's Windows application profile can change shader arithmetic even when the
executable bytes, model and frame data are identical. Run this on captured dark
HDR scenes as well as synthetic cases; uniform fixtures can miss the failure.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('case', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--runner', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    report = {'runner_sha256': digest(args.runner), 'commands': [], 'files': []}
    for name in ('d4r-k-d3d12', 'Cyberpunk2077'):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        exe = directory / (name + '.exe')
        shutil.copy2(args.runner, exe)
        command = [str(exe), str(args.model.resolve()), str(args.case.resolve()), str(directory / 'results'), '1']
        subprocess.run(command, check=True)
        report['commands'].append(command)
    reference = root / 'd4r-k-d3d12' / 'results'
    test = root / 'Cyberpunk2077' / 'results'
    outputs = sorted(reference.glob('frame-*.bin'))
    if not outputs:
        raise ValueError('runner produced no frame buffers')
    failed = False
    for source in outputs:
        target = test / source.name
        same = target.exists() and digest(source) == digest(target)
        finite = True
        for path in (source, target):
            if not path.exists():
                finite = False
                continue
            data = np.memmap(path, dtype=np.float16, mode='r')
            for start in range(0, data.size, 1048576):
                finite &= bool(np.isfinite(data[start:start + 1048576]).all())
        report['files'].append({'name': source.name, 'byte_identical': same, 'finite': finite})
        failed |= not same or not finite
        print(f'{source.name}: identical={same}, finite={finite}', flush=True)
    report['passed'] = not failed
    (root / 'profile-check.json').write_text(json.dumps(report, indent=2) + '\n')
    if failed:
        raise SystemExit('application-profile replay differs or contains nonfinite data')


if __name__ == '__main__':
    main()
