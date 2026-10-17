"""Build bounded synthetic dynamic-size RR references with the standalone engine.

Reference resets at each size change; combined NGX frames deliberately omit that
reset to verify the adapter's implicit history reset. This is engine consistency,
not an NVIDIA image-quality reference. No game files are read or written.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--model', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--output', required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if args.output.exists() or not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('use a fresh directory under ignored build/')
    args.output.mkdir(parents=True)
    inputs, reference = args.output / 'inputs', args.output / 'reference'
    inputs.mkdir(); reference.mkdir()
    lines, extents = ['size 96 96 192 192'], []
    env = {key: value for key, value in os.environ.items() if not key.startswith('D4R_')}
    env['D4R_RR_TIME_REPEAT'] = '1'
    index = 0
    for segment, (width, height) in enumerate(((96, 96), (64, 64), (96, 96), (72, 80))):
        source, result = args.output / f'segment-{segment}', args.output / f'reference-{segment}'
        subprocess.run([sys.executable, str(repo / 'scripts/windows/rr-synthetic-inputs.py'), str(source),
                        '--width', str(width), '--height', str(height), '--frames', '2',
                        '--output-width', '192', '--output-height', '192'], check=True, timeout=60)
        # Exercise optional-view rebinding across reconfiguration and an absent
        # guide frame. Fixture uploads and standalone replay use the same values.
        for frame in (1,):
            for name, texel in (('hitdistance', struct.pack('<f', 2.0)),
                                ('disocclusion', struct.pack('<e', 0.0)),
                                ('sss', struct.pack('<e', 0.125)),
                                ('particles', struct.pack('<4e', 0.2, 0.3, 0.4, 1.0))):
                (source / f'{frame}-{name}.bin').write_bytes(texel * (width * height))
        result.mkdir()
        with (args.output / f'segment-{segment}.log').open('wb') as log:
            subprocess.run([str(args.runner), str(args.model), str(source), str(result)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        frames = (source / 'frames.txt').read_text().splitlines()[1:]
        for local, line in enumerate(frames):
            fields = line.split()
            fields[1] = str(index)
            if segment and local == 0:
                fields[7] = '0'  # omit caller reset on a size change
            lines.append(' '.join(fields))
            extents.append([width, height])
            for path in source.glob(f'{local}-*.bin'):
                shutil.copy2(path, inputs / (str(index) + path.name[len(str(local)):]))
            shutil.copy2(result / f'{local}-out.bin', reference / f'{index}-out.bin')
            index += 1
    (inputs / 'frames.txt').write_text('\n'.join(lines) + '\n')
    (inputs / 'dynamic-extents.txt').write_text(''.join(f'{w} {h}\n' for w, h in extents))
    report = dict(kind='synthetic-dynamic-rr-engine-consistency', extents=extents,
                  frames=index, callerResetOnSizeChange=False, qualityReference=False,
                  model=str(args.model), runner=str(args.runner), gameAccessed=False)
    (args.output / 'fixture.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
