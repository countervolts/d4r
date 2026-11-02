#!/usr/bin/env python3
"""Compare full native frame results, including the engine's own temporal feedback.

test_frame.py CASE_DIR RESULT_DIR [--min-psnr 50]
Exits nonzero for missing data, nonfinite output, or a PSNR regression.
Final output uses fixed-peak-1, unclipped linear RGB PSNR; auxiliary buffers
use their reference peak. This is the same final-RGB metric as the Linux handoff.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def dims(name):
    m = re.search(r'-(\d+)x(\d+)x(\d+)-fmt16\.bin$', str(name))
    return tuple(map(int, m.groups())) if m else None


def compare(reference, output, region=None, final_rgb=False):
    a = np.fromfile(reference, np.float16).astype(np.float64)
    b = np.fromfile(output, np.float16).astype(np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError(f'{output}: size differs from reference')
    if region:      # (width, height, channels, used width, used height)
        w, h, c, uw, uh = region
        a = a.reshape(h, w, c)[:uh, :uw].ravel(); b = b.reshape(h, w, c)[:uh, :uw].ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'{output}: nonfinite reference or result')
    if final_rgb:
        a, b = a.reshape(-1, 4)[:, :3], b.reshape(-1, 4)[:, :3]
    error = np.abs(a - b)
    peak = 1.0 if final_rgb else max(np.abs(a).max(), 1e-9)
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
        # The engine leaves out decoder windows that only cover the token grid's padding (k_frame.cpp, kLayerGroups),
        # so the token feature is compared where the next frame can read it: the output's tokens and two more.
        out, feat = dims(refs.get('288', '')), dims(refs.get('304', ''))
        region = feat + (min(feat[0], (out[0] + 3) // 4 + 2), min(feat[1], (out[1] + 3) // 4 + 2)) if out and feat else None
        for off, reference in refs.items():
            psnr, maximum, mean = compare(reference, a.results / f'frame-{i}.{off}.bin',
                                          region if off == '304' else None, final_rgb=off == '312')
            print(f'frame {i} {names[off]:14s}: {psnr:6.1f} dB PSNR; max error {maximum:.5g}; mean {mean:.5g}')
            failed |= psnr < a.min_psnr
    if failed: raise SystemExit(f'frame validation failed: PSNR below {a.min_psnr:g} dB')


if __name__ == '__main__': main()
