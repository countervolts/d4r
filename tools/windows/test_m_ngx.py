"""Validate Windows SR preset M on synthetic D3D12 resources, never a game.

Tests packaged native M layers, a translated-PTX control on the same GPU, and
an identical native repeat. Finite/nonuniform output, actual layer selection,
and exact full-frame comparisons are separate gates. This is not NVIDIA
hardware/reference-quality validation or a standalone M engine test.
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

LAYERS = ('rrlite_enc1_4x4', 'rrlite_enc2_4x4', 'rrlite_enc3_tube_4x4', 'rrlite_dec2_4x4', 'rrlite_dec1_4x4')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'shim', 'runtime', 'core', 'dlss', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda s: Path(s).resolve())
    parser.add_argument('--resolution', default='512x288')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--native-objects', type=lambda s: Path(s).resolve(), help='overlay all five candidate M objects in the staged player only')
    args = parser.parse_args()
    if not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('output must stay under ignored build/')
    if not re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*', args.resolution) or not 4 <= args.frames <= 32:
        parser.error('resolution must be WIDTHxHEIGHT; frames must be 4..32')
    ow, oh = map(int, args.resolution.split('x'))
    args.output.mkdir(parents=True, exist_ok=False)
    player = args.output / 'player'
    shutil.copytree(args.runtime, player)
    module = player / 'd4r_nvngx.dll'
    shutil.copy2(args.shim, module)
    for source, relative in ((args.core, 'ngx/_nvngx.dll'), (args.dlss, 'dlss/nvngx_dlss.dll')):
        target = player / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config = (repo / 'packaging/windows/d4r.ini').read_text()
    config, replaced = re.subn(r'^Model\s*=\s*K\s*$', 'Model = M', config, count=1, flags=re.M)
    if replaced != 1:
        raise RuntimeError('expected the default Model = K setting')
    (player / 'd4r.ini').write_text(config)
    native = player / 'native/gfx1101'
    if args.native_objects:
        for name in LAYERS:
            shutil.copy2(args.native_objects / (name + '.hsaco'), native / (name + '.hsaco'))
    manifest_path = native / 'd4r-kernels.txt'
    manifest = manifest_path.read_text()
    control_manifest = ''.join(line for line in manifest.splitlines(keepends=True)
                               if not any(line.startswith(name + ' ') for name in LAYERS))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('D4R_', 'ZLUDA_'))}
    nvapi = player / 'nvapi/gfx1101/nvapi64.dll'
    if not nvapi.is_file():
        nvapi = player / 'nvapi/nvapi64.dll'
    report = {'passed': False, 'synthetic': True, 'gameAccessed': False, 'qualityReference': False,
              'preset': 13, 'model': 'M', 'quality': 'Performance', 'createFlags': 11,
              'render': [ow // 2, oh // 2], 'output': [ow, oh], 'frames': args.frames,
              'nativeFp8': False, 'comparison': 'same-RDNA3 translated PTX and native repeat',
              'artifacts': {str(p): sha(p) for p in (args.probe, args.shim, args.core, args.dlss)},
              'nativeObjects': {name: sha(native / (name + '.hsaco')) for name in LAYERS}, 'runs': []}
    reference_hashes = None
    try:
        for label in ('native', 'translated', 'native-repeat'):
            folder = args.output / label
            folder.mkdir()
            manifest_path.write_text(control_manifest if label == 'translated' else manifest)
            child_env = dict(env, D4R_DIAG_DIR=str(folder))
            command = [str(args.probe), '--module', str(module), '--hip-root', str(player / 'hip'),
                       '--cuda-dll', str(player / 'zluda/nvcuda.dll'), '--nvapi-dll', str(nvapi),
                       '--ngx-core', str(args.core), '--dlss-dll', str(args.dlss), '--ngx-abi', 'project',
                       '--interop-mode', 'command-list', '--player-mode', '1', '--preset', '13',
                       '--ngx-quality', '0', '--ngx-create-flags', '11', '--ngx-output-resolution', args.resolution,
                       '--ngx-input-resolution', f'{ow // 2}x{oh // 2}', '--iterations', str(args.frames)]
            with (folder / 'stdout.log').open('wb') as out, (folder / 'stderr.log').open('wb') as err:
                result = subprocess.run(command, env=child_env, stdout=out, stderr=err, timeout=args.timeout)
            text = (folder / 'stdout.log').read_text(errors='replace')
            logs = (folder / 'stderr.log').read_text(errors='replace')
            player_log = player / 'd4r_nvngx.log'
            if player_log.exists():
                shutil.copy2(player_log, folder / 'player.log')
                logs += player_log.read_text(errors='replace')
            hashes = {p.name: sha(p) for p in sorted(folder.glob('output-*.rgba16f'))}
            selected = {name: bool(re.search(r'\[d4r-native\] "' + re.escape(name) + r'" ->', logs)) for name in LAYERS}
            row = {'label': label, 'exitCode': result.returncode, 'outputHashes': hashes,
                   'nativeLayersLoaded': selected, 'finiteFrames': len(re.findall(r'D3D12_OUTPUT preset=13 frame=\d+ finite=1 ', text)),
                   'playerSelectedM': 'model=M native_fp8=0' in logs,
                   'performanceSelected': 'D3D12_NGX_QUALITY value=0' in text}
            row['executionPassed'] = (result.returncode == 0 and 'PASS D3D12_DLSS' in text
                                      and len(hashes) == args.frames and row['finiteFrames'] == args.frames
                                      and row['playerSelectedM'] and row['performanceSelected'])
            row['selectionPassed'] = not any(selected.values()) if label == 'translated' else all(selected.values())
            if reference_hashes is None:
                reference_hashes = hashes
            row['exactNativeControl'] = hashes == reference_hashes and len(hashes) == args.frames
            row['differences'] = []
            for name in hashes:
                if hashes[name] != reference_hashes.get(name):
                    actual = np.fromfile(folder / name, '<f2').astype('f4')
                    baseline = np.fromfile(args.output / 'native' / name, '<f2').astype('f4')
                    row['differences'].append({'frame': name, 'components': int(np.count_nonzero(actual != baseline)),
                                               'maxAbs': float(np.max(np.abs(actual - baseline)))})
            report['runs'].append(row)
            (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'{label}: execution={row["executionPassed"]} selection={row["selectionPassed"]} exact={row["exactNativeControl"]} frames={row["finiteFrames"]}', flush=True)
            if not row['executionPassed']:
                break
        report['passed'] = len(report['runs']) == 3 and all(r['executionPassed'] and r['selectionPassed'] and r['exactNativeControl'] for r in report['runs'])
    finally:
        manifest_path.write_text(manifest)
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
