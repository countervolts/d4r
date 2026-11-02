#!/usr/bin/env python3
"""Check the production M input WMMA product against a seeded NumPy K32 chain.

The fixture substitutes feature gathering with packed, seeded FP8 features.
It exercises the production product, four-wave indexing and partial blocks.
It does not validate feature gathering or NVIDIA frame parity.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from test_m_layer import run, statistics
from test_m import codes
import hlsl_build
import model_enc3 as me


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--runner', required=True, type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    shader = args.work / 'input.dxil'
    run([hlsl_build.dxc_path(), '-T', 'cs_6_6', '-E', 'main', '-HV', '2021',
         '-enable-16bit-types', '-O3', '-DD4R_NATIVE_HALF=1', HERE / 'test_m_input.hlsl', '-Fo', shader])
    rng = np.random.default_rng(20261023)
    w, h = 26, 18
    features = me.q8(rng.normal(0, .4, (h, w, 128))).astype(np.float16)
    weights = me.q8(rng.normal(0, .08, (128, 64))).astype(np.float16)
    bias = rng.normal(0, .02, 64).astype(np.float16)
    features.tofile(args.work / 'features.bin')
    weights.tofile(args.work / 'weights.bin')
    np.tile(bias, (16, 1)).tofile(args.work / 'bias.bin')
    np.zeros(64, np.uint8).tofile(args.work / 'skip.bin')
    reference = np.broadcast_to(bias.astype(float), (h, w, 64)).copy()
    for k in range(0, 128, 32):
        product = features[..., k:k + 32].astype(float) @ weights[k:k + 32].astype(float)
        reference = np.clip(product + reference, -65504, 65504).astype(np.float16).astype(float)
    expected = codes(me.q8(np.clip(reference, -6.2831855, 6.2831855)))
    records = []
    for shift in (0, 2):
        (args.work / 'params.bin').write_bytes(struct.pack('<4i72I', w, h, shift, shift, *([0] * 72)))
        log = run([args.runner, shader, *[args.work / f for f in
            ('params.bin', 'weights.bin', 'bias.bin', 'features.bin', 'skip.bin', 'output.bin')],
            (w + shift // 2 + 7) // 8, (h + shift // 2 + 7) // 8, 5, w * h * 64, '--wmma'])
        got = np.fromfile(args.work / 'output.bin', np.uint8).reshape(h, w, 64)
        stats = statistics(got, expected)
        if stats['psnr_db'] is not None and stats['psnr_db'] < 50:
            raise AssertionError(stats)
        records.append(dict(shift=shift, cpu=stats, log=log))
    (args.work / 'summary.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
