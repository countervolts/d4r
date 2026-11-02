#!/usr/bin/env python3
"""Write a preset K model with synthetic weights, for GPU timing without NGX or a capture.

usage: synth_k.py OUT_PREFIX LUT.bin [--seed N]
Writes OUT_PREFIX.bin/.json exactly as extract_k.py would (same layers, tensor shapes and layout), with small
random values in place of NVIDIA's weights, and a single 32768-byte reconstruction table. compile_k.py then packages
it like an extracted model. The network's work does not depend on weight values, so d4r-k timings with this model
measure the real K shapes; its images are meaningless and it must never be used for quality checks.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_k  # noqa: E402

F16 = np.float16


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out_prefix')
    p.add_argument('lut')
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    pm = extract_k.pm
    # Replace the dump readers with generators of the shapes extract_k asks for.
    pm.Dump = lambda d: None
    pm.Shifted = lambda P, base: None
    pm.weights = lambda P, byte, K, N, stride=None: (rng.standard_normal((K, N)) * 0.02).astype(F16)
    pm.vec = lambda P, byte, n: (1 + rng.standard_normal(n) * 0.02).astype(F16)
    pm.bias_table = lambda P, byte: (rng.standard_normal((64, 64)) * 0.1).astype(F16)
    extract_k.glob.glob = lambda pattern: [pattern.replace('*', '0')]
    sys.argv = ['extract_k.py', 'synthetic', a.out_prefix]
    extract_k.main()
    # 2048 sixteen-byte entries of f16 reconstruction weights.
    (rng.random(32768 // 2) * 0.1).astype(F16).tofile(a.lut)
    print(f'{a.lut}: 32768 bytes (synthetic)')


if __name__ == '__main__':
    main()
