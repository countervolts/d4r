"""Stamp or verify local RR build artifacts against sources, tools and DLL identity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('write', 'check'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--preset', choices=('D', 'E'), required=True)
    parser.add_argument('--glslang', type=Path, required=True)
    parser.add_argument('--bash', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    model = args.model.resolve()
    if not model.is_relative_to((repo / 'build').resolve()):
        parser.error('model cache must stay under ignored build/')
    import numpy
    sources = {}
    for directory in ('engine', 'vulkan', 'kernels/tools', 'scripts/windows'):
        for path in sorted((repo / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.ps1', '.sh', '.comp', '.glsl', '.h', '.inc', '.txt'):
                sources[path.relative_to(repo).as_posix()] = sha(path)
    identity = dict(schema=1, preset=args.preset, architecture='gfx1101', arithmetic='exact-gfx11',
                    numpyVersion=numpy.__version__, sources=sources,
                    tools={name: sha(path) for name, path in (
                        ('dll', args.dll), ('glslang', args.glslang),
                        ('bash', args.bash), ('python', Path(sys.executable)))})
    stamp = model / 'windows-rr-cache.json'
    products = {path.name: sha(path) for path in sorted(model.iterdir())
                if path.is_file() and path.name not in (stamp.name, stamp.name + '.tmp')}
    for name in ('rr.txt', 'model.json', 'enc0_m.spv', 'dec0_m.spv', 'hkpn.spv', 'enc0.args', 'hkpn.args'):
        if name not in products:
            raise RuntimeError(f'incomplete model: {name}')
    manifest = (model / 'rr.txt').read_text().splitlines()
    if (not manifest or manifest[0] != 'D4RR0001' or
            f'preset {4 if args.preset == "D" else 5}' not in manifest or
            'arithmetic gfx11' not in manifest):
        raise RuntimeError('RR manifest preset/arithmetic does not match requested cache identity')
    stages = [line.split() for line in manifest if line.startswith('stage ')]
    expected_stages = {'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
                       'dec4', 'dec3', 'dec2', 'dec1', 'dec0', 'hkpn'}
    if len(stages) != 12 or {fields[1] for fields in stages} != expected_stages:
        raise RuntimeError('RR stage list is incomplete or duplicated')
    for fields in stages:
        for name in (fields[2], fields[4]):
            if name != '-' and name not in products:
                raise RuntimeError(f'RR stage artifact missing or outside model directory: {name}')
    if args.action == 'check':
        cached = json.loads(stamp.read_text())
        if cached.get('identity') != identity:
            raise RuntimeError('RR cache identity changed; use a new output directory to rebuild')
        if cached.get('products') != products:
            raise RuntimeError('RR cache artifacts changed; use a new output directory to rebuild')
        print(f'RR_CACHE_HIT preset={args.preset} artifacts={len(products)} model={model}')
    else:
        temporary = stamp.with_name(stamp.name + '.tmp')
        temporary.write_text(json.dumps(dict(identity=identity, products=products), indent=2) + '\n')
        temporary.replace(stamp)
        print(f'RR_CACHE_WRITTEN preset={args.preset} artifacts={len(products)} model={model}')


if __name__ == '__main__':
    main()
