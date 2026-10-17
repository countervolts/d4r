"""Compile a private native K LDR model from verified Windows synthetic captures.

Requires all 11 extracted layers and 16 captured reconstruction tables. Never
substitutes synthetic tables, installs into a game, or creates a release archive.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--output', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--glslang', type=lambda p: Path(p).resolve(), default=repo / '.tools/glslang-16.6.0/bin/glslangValidator.exe')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('private model products must stay under ignored build/')
    validation = json.loads((args.capture / 'validation.json').read_text())
    if (validation.get('passed') is not True or validation.get('reconstructionLutCaptured') is not True
            or validation.get('reconstructionTables') != 16 or validation.get('extractedLayers') != 11
            or validation.get('numerics') != 'explicit-f16-gains'):
        parser.error('requires verified complete 11-layer/16-table capture')
    if set(validation['products']) != {'k.bin', 'k.json', 'lut.bin'}:
        raise RuntimeError('Unexpected/missing capture products')
    weights = args.capture / 'weights'
    for name, expected in validation['products'].items():
        if sha(weights / name) != expected:
            raise RuntimeError('Capture product changed: ' + name)
    if (weights / 'lut.bin').stat().st_size != 16 * 32768:
        raise RuntimeError('Incomplete reconstruction tables')
    if sha(args.glslang) != '9643709c02c83e26773f96b6a87f197ab2e84f78be0879ad8fb3619418b5d9ad':
        raise RuntimeError('Requires pinned glslang 16.6.0')
    inputs = ['engine/compile_k.py', 'engine/test_k.py', 'engine/layer.comp', 'engine/input_k.comp',
              'engine/output_k.comp', 'engine/exposure_k.comp', 'engine/normalized_store.glsl']
    identity = dict(schema=1, preset='K', variant='ldr', numerics='explicit-f16-gains',
                    captureProducts=validation['products'], glslang=sha(args.glslang),
                    python=sha(Path(sys.executable)), sources={name:sha(repo / name) for name in inputs},
                    builder=sha(Path(__file__)))
    stamp = args.output / 'windows-k-model-cache.json'
    def products():
        return {p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != stamp.name}
    if args.output.exists():
        if not args.reuse:
            parser.error('use a fresh output directory or --reuse')
        cached = json.loads(stamp.read_text())
        if cached['identity'] != identity or cached['products'] != products():
            raise RuntimeError('K model cache changed; preserve it and use a fresh output directory')
        print('K_MODEL_CACHE_HIT')
        return
    env = {key:value for key,value in os.environ.items() if not key.startswith(('D4R_', 'K_DEFS_'))}
    env['PYTHONPATH'] = os.pathsep.join(str(repo / name) for name in ('.tools/python/vendor', 'engine', 'kernels/tools'))
    env['PATH'] = str(args.glslang.parent) + os.pathsep + env['PATH']
    args.output.mkdir(parents=True)
    command = [sys.executable, str(repo / 'engine/compile_k.py'), str(weights / 'k'),
               str(weights / 'lut.bin'), str(args.output), '--ldr']
    with (args.output / 'compile.log').open('wb') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process.wait(timeout=30)
            raise RuntimeError('K compilation timed out; process tree terminated')
    if code:
        raise RuntimeError('K compilation failed; inspect compile.log')
    manifest = json.loads((args.output / 'model.json').read_text())
    if manifest.get('numerics') != 'explicit-f16-gains' or manifest.get('variant') != 'ldr' or len(manifest['layers']) != 11:
        raise RuntimeError('Incorrect compiled K model contract')
    for name, expected in manifest['sha256'].items():
        if Path(name).name != name or sha(args.output / name) != expected:
            raise RuntimeError('Compiled model product failed verification')
    stamp.write_text(json.dumps(dict(identity=identity, products=products(), privateLocalModel=True,
                                    gpuValidated=False, gameValidated=False, published=False), indent=2) + '\n')
    print('K_MODEL_BUILT private=1 gpu_validated=0 products=' + str(len(products())))


if __name__ == '__main__':
    main()
