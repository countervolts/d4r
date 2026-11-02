#!/usr/bin/env python3
"""4K Ray Reconstruction route check: reset and continued frames on every preset, both routes.

The timing suite skips output writes (D4R_RR_NO_WRITE=1) to keep the timed region clean, so the
4K medians have no hashed output of their own. This runs the same two-entry cadence used for the
1440p route check (a reset frame followed by a continued frame) at 1920x1080 -> 3840x2160 and
records coverage, finiteness, hashes and the vkhip-versus-Vulkan difference per preset.

Writes build/local-preset-compare-20261029/logs/route-verification-4k-rr.json.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'build/local-preset-compare-20261029'
RUNNER = Path('/tmp/d4r-preset-compare-20261029/engine')
OUT = W / 'logs/verify'
CASES = [('a', 'models-cnn/rr-a'), ('b', 'models-cnn/rr-b'), ('c', 'models-cnn/rr-c'),
         ('d', 'models/rr-d'), ('e', 'models/rr-e'), ('f', 'models/rr-f')]


def image(path):
    return np.fromfile(path, dtype=np.float16).reshape(-1, 4).astype(np.float32)


def stats(path):
    px = np.fromfile(path, dtype=np.float16).reshape(-1, 4)
    rgb = px[:, :3].astype(np.float32)
    return {'bytes': int(path.stat().st_size), 'pixels': int(px.shape[0]),
            'nonzero_rgb_pixels': int(np.any(rgb != 0, axis=1).sum()),
            'nonfinite_components': int((~np.isfinite(rgb)).sum()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def compare(a, b):
    x, y = image(a), image(b)
    err = np.abs(x[:, :3] - y[:, :3])
    mse = float(np.mean(err ** 2))
    psnr = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)
    return {'differing_pixels': int((err > 0).any(axis=1).sum()), 'pixels': int(x.shape[0]),
            'mean_abs_error': float(np.mean(err)), 'max_abs_error': float(np.max(err)),
            'psnr_db_vs_unit_range': round(psnr, 2) if psnr != float('inf') else None}


def run(preset, model, backend):
    out = OUT / f'rr-{preset}-4k-{backend}'
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob('*.bin'):
        stale.unlink()
    env = dict(os.environ, D4R_RR_OUTPUT_ONLY='1')
    if backend == 'vkhip':
        env['D4R_ENGINE_VKHIP'] = '1'
    r = subprocess.run([str(RUNNER / 'd4r-rr'), str(W / model), str(W / 'inputs/rr-verify-4k'), str(out)],
                       env=env, capture_output=True, text=True)
    (OUT.parent / f'verify-rr-{preset}-4k-{backend}.log').write_text(r.stdout + r.stderr)
    text = r.stdout + r.stderr
    if r.returncode:
        return {'preset': preset, 'backend': backend, 'failed': r.returncode, 'tail': text[-300:]}
    files = sorted(out.glob('*.bin'))
    stages = text.split('vkhip: ')[1].split(' ')[0] if 'vkhip: ' in text else None
    return {'preset': preset, 'backend': backend, 'model': model, 'vkhip_stages': stages,
            'unavailable': 'vkhip unavailable' in text,
            'outputs': {f.name: stats(f) for f in files},
            'continued_differs_from_reset': (files[0].read_bytes() != files[1].read_bytes()) if len(files) > 1 else None}


def main():
    records = []
    for preset, model in CASES:
        row = {b: run(preset, model, b) for b in ('vulkan', 'vkhip')}
        for b in ('vulkan', 'vkhip'):
            if 'failed' in row[b]:
                print(f'{preset} {b} FAILED {row[b]["tail"]}', file=sys.stderr)
                continue
            names = sorted(row[b]['outputs'])
            print(f'{preset} {b}: vkhip_stages={row[b]["vkhip_stages"]} outputs={names} '
                  f'nonzero={row[b]["outputs"][names[-1]]["nonzero_rgb_pixels"]} '
                  f'continued_differs={row[b]["continued_differs_from_reset"]}')
        if 'failed' not in row['vulkan'] and 'failed' not in row['vkhip']:
            row['vkhip_vs_vulkan'] = {'reset': compare(OUT / f'rr-{preset}-4k-vulkan/0-out.bin',
                                                       OUT / f'rr-{preset}-4k-vkhip/0-out.bin'),
                                      'continued': compare(OUT / f'rr-{preset}-4k-vulkan/1-out.bin',
                                                            OUT / f'rr-{preset}-4k-vkhip/1-out.bin')}
        records.append(row)
    dest = W / 'logs/route-verification-4k-rr.json'
    dest.write_text(json.dumps(records, indent=1))
    print(f'wrote {dest}')


if __name__ == '__main__':
    main()
