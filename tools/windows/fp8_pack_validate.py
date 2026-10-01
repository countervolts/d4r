"""Check gfx1201 hardware packing of every finite OCP e4m3 value and signed zero."""
import argparse
import hashlib
import json
import os
import pathlib
import struct
import subprocess

import numpy as np
from m_replay_validate import me


def main(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    codes = np.arange(256, dtype=np.uint8)
    codes[(codes & 127) == 127] = 0  # NaN inputs are outside this finite-value packing contract.
    inputs = me.E4M3[codes].astype(np.float16).tobytes()
    (root / 'alloc-0.bin').write_bytes(inputs)
    (root / 'alloc-1.bin').write_bytes(bytes(256))
    (root / 'args.bin').write_bytes(struct.pack('<QQ', 0x100000000, 0x200000000))
    (root / 'manifest.txt').write_text('kernel fp8_pack_probe\nlaunch 1 1 1 32 1 1 0\nargs 16\n'
        'alloc 0 0x100000000 512\npointer 0 0 0\nalloc 1 0x200000000 256\npointer 8 1 0\n')
    with (root / 'gpu.stdout.log').open('wb') as stdout, (root / 'gpu.stderr.log').open('wb') as stderr:
        result = subprocess.run([str(args.probe.resolve()), '--hip-root', str(args.hip_root.resolve()),
            '--module', str(args.module.resolve()), '--fixture-dir', str(root), '--output-dir', str(root / 'output'),
            '--iterations', '1'], stdout=stdout, stderr=stderr, timeout=90,
            env=dict(os.environ, D4R_DIAG_DIR=str(root)))
    summary = dict(passed=False, architecture='gfx1201', finiteEncodings=254, signedZero=True,
        moduleSha256=hashlib.sha256(args.module.read_bytes()).hexdigest(), exitCode=f'0x{result.returncode & 0xffffffff:08x}')
    if result.returncode == 0:
        actual = np.fromfile(root / 'output/alloc-1.bin', np.uint8)
        summary['passed'] = actual.size == 256 and bool(np.array_equal(actual, codes))
        if actual.size == 256:
            summary['differingCodes'] = int(np.count_nonzero(actual != codes))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(('PASS' if summary['passed'] else 'FAIL') + ' FP8_HARDWARE_PACK finite_encodings=254 signed_zero=1 architecture=gfx1201', flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'module', 'hip-root', 'output-dir'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    raise SystemExit(main(parser.parse_args()))
