"""Public gfx1201 WMMA arithmetic comparison; does not load DLSS or weights."""
import argparse
import json
import os
import pathlib
import struct
import subprocess

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=pathlib.Path, required=True)
    p.add_argument('--module', type=pathlib.Path, required=True)
    p.add_argument('--hip-root', type=pathlib.Path, required=True)
    p.add_argument('--output-dir', type=pathlib.Path, required=True)
    args = p.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    fixture = root / 'fixture'
    fixture.mkdir(exist_ok=True)
    rng = np.random.default_rng(90701201)
    a = rng.uniform(-1, 1, (64, 16, 16)).astype(np.float16)
    b = rng.uniform(-1, 1, a.shape).astype(np.float16)
    c = rng.uniform(-.25, .25, a.shape).astype(np.float16).astype(np.float32)
    initial_output = np.full((64, 3, 16, 16), np.nan, np.float32)
    arrays = (a, b, c, initial_output)
    bases = [0x10000000 * (i + 1) for i in range(4)]
    (fixture / 'args.bin').write_bytes(struct.pack('<4Q', *bases))
    lines = ['kernel d4r_wmma_f16_compare', 'launch 64 1 1 32 1 1 0', 'args 32']
    for i, data in enumerate(arrays):
        data.tofile(fixture / f'alloc-{i}.bin')
        lines += [f'alloc {i} {bases[i]:#x} {data.nbytes}', f'pointer {i * 8} {i} 0']
    (fixture / 'manifest.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    env = os.environ.copy()
    env['D4R_DIAG_DIR'] = str(root)
    with (root / 'gpu.stdout.log').open('wb') as out, (root / 'gpu.stderr.log').open('wb') as err:
        status = subprocess.run([str(args.probe.resolve()), '--hip-root', str(args.hip_root.resolve()),
            '--module', str(args.module.resolve()), '--fixture-dir', str(fixture), '--output-dir',
            str(root / 'output'), '--iterations', '1'], env=env, stdout=out, stderr=err, timeout=120)
    if status.returncode:
        raise RuntimeError(f'GPU probe exit {status.returncode}; see {root}')
    output = np.fromfile(root / 'output/alloc-3.bin', np.float32).reshape(64, 3, 16, 16)
    if not np.isfinite(output).all():
        raise RuntimeError('Nonfinite/unwritten WMMA output')
    rows = []
    for group, steps in enumerate((1, 2, 4, 8)):
        values = output[group * 16:(group + 1) * 16]
        difference = np.abs(values[:, 0].astype(np.float64) - values[:, 1].astype(np.float64))
        row = dict(steps=steps, elements=difference.size, differing=int(np.count_nonzero(difference)),
            maxAbsoluteError=float(difference.max()), rms=float(np.sqrt(np.mean(difference ** 2))))
        rows.append(row)
        print('WMMA_F16_COMPARE ' + json.dumps(row), flush=True)
    adapter_exact = np.array_equal(output[:, 1].view(np.uint32), output[:, 2].view(np.uint32))
    if not adapter_exact:
        raise RuntimeError('Direct packed instruction and layout adapter differ')
    result = dict(passed=True, architecture='gfx1201', publicSynthetic=True, adapterExact=adapter_exact,
        baselineEquivalent=all(row['differing'] == 0 for row in rows), cases=rows,
        gameOptimizationAccepted=False)
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('PASS WMMA_F16_ARITHMETIC_DIAGNOSTIC adapter_exact=1 baseline_equivalent=' +
        str(int(result['baselineEquivalent'])), flush=True)


if __name__ == '__main__':
    main()
