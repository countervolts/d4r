"""Compare bounded FP32 attention dots with sequential FP64 on the GPU."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess


def main(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    count = 262144
    (root / 'alloc-0.bin').write_bytes(bytes(count * 4))
    (root / 'args.bin').write_bytes(struct.pack('<Q', 0x100000000))
    (root / 'manifest.txt').write_text(
        f'kernel exact_pv_probe\nlaunch {count // 32} 1 1 32 1 1 0\nargs 8\n'
        f'alloc 0 0x100000000 {count * 4}\npointer 0 0 0\n')
    with (root / 'stdout.log').open('wb') as out, (root / 'stderr.log').open('wb') as err:
        completed = subprocess.run([
            str(args.probe.resolve()), '--hip-root', str(args.hip_root.resolve()),
            '--module', str(args.module.resolve()), '--fixture-dir', str(root),
            '--output-dir', str(root / 'output'), '--iterations', '1'],
            stdout=out, stderr=err, timeout=120, env=dict(os.environ, D4R_DIAG_DIR=str(root)))
    if completed.returncode:
        raise RuntimeError(f'GPU probe failed: {root}')
    data = (root / 'output/alloc-0.bin').read_bytes()
    mismatches = sum((word & 65535) != (word >> 16) for (word,) in struct.iter_unpack('<I', data))
    summary = dict(passed=len(data) == count * 4 and mismatches == 0,
                   dots=count, differingHalfBits=mismatches)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(summary)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'hip-root', 'module', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
