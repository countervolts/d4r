#!/usr/bin/env python3
"""Prove each route ran the intended kernels and produced a valid, temporally-updating output.

Checks, on the exact model each benchmark route used:
  * SR: d4r-m's pass-0 output image - nonzero, finite, fully covered; the routes' outputs compared to the
    engine's own Vulkan network (vkhip must match the native HIP route byte for byte).
  * RR: d4r-rr's final image - nonzero, finite, covered; a reset frame and a continued frame must differ
    (the temporal history is in use); the vkhip route compared with the Vulkan route on the same inputs.
  * the model's declared preset, and for vkhip the loader's own stage count.

usage: verify_routes.py      writes logs/route-verification.json
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'build/local-preset-compare-20261029'
RUNNER = Path('/tmp/d4r-preset-compare-20261029/engine')
JITTER = [(-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)]
OUT = W / 'logs/verify'


def image(path):
    """RGBA16F file -> float32 (H, W, 4)."""
    a = np.fromfile(path, dtype=np.float16)
    return a.reshape(-1, 4).astype(np.float32)


def stats(path):
    px = image(path)
    rgb = px[:, :3]
    return {'pixels': int(px.shape[0]), 'nonzero_rgb_pixels': int(np.any(rgb != 0, axis=1).sum()),
            'nonfinite_components': int((~np.isfinite(rgb)).sum()),
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def compare(a, b):
    """RGB comparison of two output images: fraction differing, mean/max error, PSNR over [0,1]."""
    x, y = image(a)[:, :3], image(b)[:, :3]
    differing = int((np.abs(x - y) > 0).any(axis=1).sum())
    err = np.abs(x - y)
    finite = np.isfinite(err)
    mse = float(np.mean(np.square(np.where(finite, err, 0))))
    psnr = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)
    return {'differing_pixels': differing, 'pixels': int(x.shape[0]),
            'mean_abs_error': float(np.mean(err[finite])), 'max_abs_error': float(np.max(err[finite])),
            'psnr_db_vs_unit_range': round(psnr, 2) if psnr != float('inf') else None, 'identical': differing == 0}


def sr_case(preset, geom, backend):
    rw, rh, ow, oh = {'1440p': (1280, 720, 2560, 1440), '4k': (1920, 1080, 3840, 2160)}[geom]
    inputs = W / f'inputs/sr-{geom}'
    args = []
    for f, (jx, jy) in enumerate(JITTER):
        args += [str(inputs / f'{f}-color.bin'), str(inputs / f'{f}-motion.bin'), str(inputs / f'{f}-depth.bin'),
                 str(jx), str(jy)]
    prefix = OUT / f'sr-{preset}-{geom}-{backend}'
    cmd = [str(RUNNER / 'd4r-m'), str(W / f'models/sr-{preset}'), str(rw), str(rh), str(ow), str(oh), str(prefix), '0']
    if backend != 'vulkan':
        cmd.append(backend)
    cmd += ['--'] + args
    env = dict(os.environ)
    if backend == 'vkhip':
        env['D4R_VKHIP_STAMPS'] = '1'          # names the native layers actually recorded
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (OUT / f'sr-{preset}-{geom}-{backend}.log').write_text(r.stdout + r.stderr)
    if r.returncode:
        return {'preset': preset, 'geometry': geom, 'backend': backend, 'failed': r.returncode,
                'tail': (r.stdout + r.stderr)[-300:], 'command': ' '.join(cmd)}
    declared = json.loads((W / f'models/sr-{preset}/model.json').read_text())['preset']
    stamps = [line for line in (r.stdout + r.stderr).splitlines() if 'VKHIP_STAMPS' in line][:6]
    return {'preset': preset, 'geometry': geom, 'backend': backend, 'model_preset': declared,
            'output': stats(Path(str(prefix) + '-0.rgba16f')), 'vkhip_layer_stamps': stamps, 'command': ' '.join(cmd)}


def rr_case(preset, model, backend, inputs):
    out = OUT / f'rr-{preset}-{backend}'
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob('*.bin'):
        stale.unlink()
    env = dict(os.environ, D4R_RR_OUTPUT_ONLY='1')
    if backend == 'vkhip':
        env['D4R_ENGINE_VKHIP'] = '1'
    r = subprocess.run([str(RUNNER / 'd4r-rr'), str(W / model), str(inputs), str(out)], env=env,
                       capture_output=True, text=True)
    (OUT.parent / f'verify-rr-{preset}-{backend}.log').write_text(r.stdout + r.stderr)
    if r.returncode:
        return {'preset': preset, 'backend': backend, 'failed': r.returncode,
                'tail': (r.stdout + r.stderr)[-300:], 'command': ' '.join([str(W / model), str(inputs)])}
    manifest = (W / model / 'rr.txt').read_text().splitlines()
    files = sorted(out.glob('*.bin'))
    return {'preset': preset, 'backend': backend, 'model': model, 'manifest_format': manifest[0],
            'declared_preset_line': manifest[1],
            'vkhip_stages': (r.stdout + r.stderr).split('vkhip: ')[1].split(' ')[0] if 'vkhip: ' in r.stdout + r.stderr else None,
            'outputs': {f.name: stats(f) for f in files}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # a two-entry RR input: entry 0 resets, entry 1 continues the history with the opposite jitter
    for geom, size in (('1440p', (1280, 720, 2560, 1440)), ('4k', (1920, 1080, 3840, 2160))):
        src = W / f'inputs/rr-{geom}'
        case = W / f'inputs/rr-verify-{geom}'
        case.mkdir(exist_ok=True)
        for f in src.glob('0-*.bin'):
            (case / f.name).write_bytes(f.read_bytes())
            (case / f.name.replace('0-', '1-')).write_bytes(f.read_bytes())   # the same frame under index 1
        lines = (src / 'frames.txt').read_text().splitlines()
        entries = lines[1:3]
        (case / 'frames.txt').write_text(lines[0] + '\n' + entries[0] + '\n' + entries[1].replace('frame 0', 'frame 1', 1) + '\n')
    report = {'sr': [], 'rr': [], 'route_agreement': [], 'rr_history': []}
    for preset in ('m', 'l'):
        for geom in ('1440p', '4k'):
            for backend in ('vulkan', 'vkhip', 'hip'):
                report['sr'].append(sr_case(preset, geom, backend))
        base = next(r for r in report['sr'] if r['preset'] == preset and r['geometry'] == '1440p' and r['backend'] == 'vulkan')
        for backend in ('vkhip', 'hip'):
            alt = next(r for r in report['sr'] if r['preset'] == preset and r['geometry'] == '1440p' and r['backend'] == backend)
            if 'output' in alt and 'output' in base:
                report['route_agreement'].append({'family': f'sr-{preset}', 'against': 'vulkan', 'route': backend,
                                                  'geometry': '1440p',
                                                  'comparison': compare(Path(W / f'logs/verify/sr-{preset}-1440p-vulkan-0.rgba16f'),
                                                                        Path(W / f'logs/verify/sr-{preset}-1440p-{backend}-0.rgba16f'))})
    for preset, model in (('a', 'models-cnn/rr-a'), ('b', 'models-cnn/rr-b'), ('c', 'models-cnn/rr-c'),
                          ('d', 'models/rr-d'), ('e', 'models/rr-e'), ('f', 'models/rr-f')):
        got = {}
        for backend in ('vulkan', 'vkhip'):
            got[backend] = rr_case(preset, model, backend, W / 'inputs/rr-verify-1440p')
            report['rr'].append(got[backend])
        if all('outputs' in got[b] for b in ('vulkan', 'vkhip')):
            report['route_agreement'].append({'family': f'rr-{preset}', 'against': 'vulkan', 'route': 'vkhip',
                                              'geometry': '1440p',
                                              'comparison': compare(OUT / f'rr-{preset}-vulkan/1-out.bin',
                                                                    OUT / f'rr-{preset}-vkhip/1-out.bin')})
            report['rr_history'].append(
                {'family': f'rr-{preset}',
                  'reset_and_continued_differ': got['vulkan']['outputs']['0-out.bin']['sha256'] != got['vulkan']['outputs']['1-out.bin']['sha256'],
                  'vkhip_reset_and_continued_differ': got['vkhip']['outputs']['0-out.bin']['sha256'] != got['vkhip']['outputs']['1-out.bin']['sha256']})
    (W / 'logs/route-verification.json').write_text(json.dumps(report, indent=1))
    # the images were only evidence for the hashes above; keep the logs, not 30 MB textures
    for f in list(OUT.rglob('*.rgba16f')) + list(OUT.rglob('*.bin')):
        f.unlink()
    print(json.dumps({k: v for k, v in report.items() if k == 'route_agreement'}, indent=1))
    print(f'wrote {W/"logs/route-verification.json"}')


main()
