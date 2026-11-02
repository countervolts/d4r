#!/usr/bin/env python3
"""Compares two d4r-k result directories file by file: typically the Vulkan backend (d4r-k) against the D3D12
backend (d4r-k-d3d12) on the same case. Every file is binary16: the output (.312), the history (.288 colour,
.296 luma, .304 feature) and the network's head (.head).

usage: compare_k.py REFERENCE_DIR TEST_DIR [--min-psnr DB]
Final output uses fixed-peak-1, unclipped linear RGB PSNR, matching the Linux
handoff checker. Auxiliary buffers retain their reference-peak metric.
Exits 1 when a frame's output falls below --min-psnr (default 50 dB) or a file is missing, misshapen or nonfinite.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

KINDS = (('312', 'output'), ('288', 'history colour'), ('296', 'history luma'), ('304', 'feature'), ('head', 'head'),
         ('tokens', 'tokens'))


def compare(a, b, peak=None):
    a, b = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    mismatch = int((~finite).sum())
    diff = np.abs(a[finite] - b[finite])
    if peak is None:
        peak = max(float(np.abs(a[finite]).max(initial=0)), 1e-6)
    mse = float((diff ** 2).mean()) if diff.size else 0.0
    psnr = float('inf') if mse == 0 else 10 * np.log10(peak * peak / mse)
    exact = float((a == b).mean()) * 100
    return psnr, float(diff.max(initial=0)), exact, mismatch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('reference', type=Path)
    p.add_argument('test', type=Path)
    p.add_argument('--min-psnr', type=float, default=50.0)
    a = p.parse_args()
    frames = sorted(int(f.name.split('.')[0].split('-')[1]) for f in a.reference.glob('frame-*.312.bin'))
    if not frames:
        sys.exit(f'no frames in {a.reference}')
    failed = False
    for frame in frames:
        for suffix, name in KINDS:
            ref, test = (d / f'frame-{frame}.{suffix}.bin' for d in (a.reference, a.test))
            if not ref.exists() and suffix == 'tokens':
                continue
            if not test.exists():
                print(f'frame {frame} {name}: missing in {a.test}')
                failed = True
                continue
            x, y = np.fromfile(ref, np.float16), np.fromfile(test, np.float16)
            if x.shape != y.shape or not x.size:
                print(f'frame {frame} {name}: {x.size} against {y.size} values')
                failed = True
                continue
            if suffix == '312':
                if x.size % 4:
                    print(f'frame {frame} output: expected RGBA texels')
                    failed = True
                    continue
                x, y = x.reshape(-1, 4)[:, :3], y.reshape(-1, 4)[:, :3]
            psnr, worst, exact, mismatch = compare(x, y, peak=1.0 if suffix == '312' else None)
            print(f'frame {frame} {name:>14}: {psnr:7.2f} dB, max |diff| {worst:.6g}, {exact:6.2f}% exact'
                  + (f', {mismatch} non-finite samples' if mismatch else ''))
            if mismatch or (suffix == '312' and psnr < a.min_psnr):
                failed = True
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
