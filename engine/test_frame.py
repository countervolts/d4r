#!/usr/bin/env python3
"""Compare full native frame results, including the engine's own temporal feedback.

test_frame.py CASE_DIR RESULT_DIR [--min-psnr 50]
Exits nonzero for missing data, nonfinite output, or a PSNR regression.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, output):
    a = np.fromfile(reference, np.float16).astype(np.float64)
    b = np.fromfile(output, np.float16).astype(np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError(f'{output}: size differs from reference')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'{output}: nonfinite reference or result')
    error = np.abs(a - b)
    peak = max(np.abs(a).max(), 1e-9)
    psnr = 10 * np.log10(peak**2 / max((error**2).mean(), 1e-30))
    return psnr, error.max(), error.mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', type=Path)
    p.add_argument('results', type=Path)
    p.add_argument('--min-psnr', type=float, default=50)
    a = p.parse_args()
    references = json.loads((a.case / 'references.json').read_text())
    if not references: raise ValueError('no reference frames')
    failed = False
    names = {'288': 'history color', '296': 'hi-res luma', '304': 'token feature', '312': 'final RGB'}
    for i, refs in enumerate(references):
        for off, reference in refs.items():
            psnr, maximum, mean = compare(reference, a.results / f'frame-{i}.{off}.bin')
            print(f'frame {i} {names[off]:14s}: {psnr:6.1f} dB PSNR; max error {maximum:.5g}; mean {mean:.5g}')
            failed |= psnr < a.min_psnr
    if failed: raise SystemExit(f'frame validation failed: PSNR below {a.min_psnr:g} dB')


if __name__ == '__main__': main()
