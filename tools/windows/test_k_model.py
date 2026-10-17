"""Run a bounded synthetic Vulkan K model smoke. No NGX, CUDA or game inference.

Authentic captured weights do not make the generated scene a quality reference.
Reports GPU dispatch validity separately from model provenance and game parity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runner', 'model', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if args.output.exists() or not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('use fresh output under ignored build/')
    cache = json.loads((args.model / 'windows-k-model-cache.json').read_text())
    for name, expected in cache['products'].items():
        if Path(name).name != name or hashlib.sha256((args.model / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Compiled K model product changed')
    args.output.mkdir(parents=True)
    rw = rh = 128
    ow = oh = 256
    color, depth = bytearray(), bytearray()
    for y in range(rh):
        for x in range(rw):
            u, v = x / (rw - 1), y / (rh - 1)
            color += struct.pack('<4e', .1 + .7 * u, .1 + .6 * v, .2 + .3 * u * v, 1)
            depth += struct.pack('<f', .5)
    (args.output / 'color.bin').write_bytes(color)
    (args.output / 'depth.bin').write_bytes(depth)
    (args.output / 'motion.bin').write_bytes(bytes(rw * rh * 4))
    lines = [f'output {ow} {oh}']
    for index, (name, channels, bits) in enumerate((('color', 4, 16), ('motion', 2, 16), ('depth', 1, 32))):
        lines.append(f'image {index} {json.dumps(str(args.output / (name + ".bin")))} {rw} {rh} {channels} {bits}')
    for frame in range(4):
        lines.append(f'frame_auto 0 1 2 1 1 1 {int(frame == 0)} 0 0 0 0 0 0 1.37')
    case = args.output / 'case.txt'
    case.write_text('\n'.join(lines) + '\n')
    env = {key:value for key,value in os.environ.items() if not key.startswith('D4R_')}
    report = dict(passed=False, synthetic=True, authenticCapturedWeights=True, qualityReference=False,
                  gameAccessed=False, cudaNgxInference=False, model=str(args.model),
                  runnerSha256=hashlib.sha256(args.runner.read_bytes()).hexdigest())
    try:
        with (args.output / 'stdout.log').open('wb') as out, (args.output / 'stderr.log').open('wb') as err:
            result = subprocess.run([str(args.runner), str(args.model), str(case), str(args.output / 'frames')],
                                    env=env, stdout=out, stderr=err, timeout=180)
        if result.returncode:
            raise RuntimeError('Native K runner failed; inspect logs')
        sys.path.insert(0, str(repo / '.tools/python/vendor'))
        import numpy as np
        outputs = sorted((args.output / 'frames').glob('frame-*.312.bin'))
        if len(outputs) != 4:
            raise RuntimeError('Incomplete native K output set')
        for path in outputs:
            values = np.fromfile(path, '<f2')
            if values.size != ow * oh * 4 or not np.isfinite(values).all() or float(np.ptp(values.astype('f4').reshape(-1, 4)[:, :3])) < 1e-5:
                raise RuntimeError('Invalid native K output: ' + path.name)
        report.update(passed=True, frames=4, finiteNonuniform=True,
                      outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
                      gpuTimeParity=False, gameValidated=False)
        print(json.dumps(report, indent=2))
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (args.output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
