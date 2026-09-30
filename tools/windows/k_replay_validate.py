#!/usr/bin/env python3
"""Generate public synthetic K replay fixtures and validate native outputs.

Uses the upstream replay format and NumPy model for all eleven K layers.
No NVIDIA weights, PTX or DLLs are used by this fixture generator.
"""
import argparse
import math
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
REFERENCE = pathlib.Path(__file__).resolve().parent / 'reference'
if not (REFERENCE / 'pwin_check.py').exists():
    REFERENCE = ROOT / 'kernels/tools'
sys.path.insert(0, str(REFERENCE))
import pwin_model as pm
pm.NORM_FRAGMENT_ORDER = True
from pwin_check import LAYERS, model_window, Outputs, stored, windows


def generate(layer, directory):
    kind, heads, channels, other, extra = LAYERS[layer]
    cin = 16 if kind == 'enc0' else channels
    low = kind in ('dec', 'dec0')
    cout = other if kind in ('enc', 'enc0') else channels
    layout = pm.Layout(heads, channels, cout)
    core_base = 2 * channels + 2 * 16 * channels if kind == 'enc0' else 0
    if low:
        core_base = 2 * other * 4 * channels + 8 * channels
    weight_bytes = core_base + layout.PMB + 2 * layout.NPA * heads
    weights = np.zeros(weight_bytes, np.uint8)

    def half(offset, value):
        weights[offset:offset + 2] = np.array([value], np.float16).view(np.uint8)

    def matrix(base, k, n, stride, value=1.0):
        half(base + 512 * (k // 16) + stride * (n // 16) + pm.frag_offset(k % 16, n % 16), value)

    if kind == 'enc0':
        for k in range(16):
            matrix(2 * channels, k, k, 512)
            matrix(2 * channels, k, k + 16, 512, .5)
    if low:
        for q in range(4):
            for k in range(min(other, channels)):
                matrix(0, k, q * channels + k, 512 * (other // 16), .25)
    for c in range(channels):
        half(core_base + layout.G1 + 2 * c, 1)
        half(core_base + layout.G2 + 2 * c, 1)
        half(core_base + layout.BO + 2 * c, 1 / 64)
        half(core_base + layout.B2 + 2 * c, 1 / 64)
    position_only = kind in ('enc0', 'dec0') or bool(extra)
    for head in range(heads):
        source_group = head % (channels // 32)
        for j in range(3):
            for k in range(32):
                matrix(core_base + layout.QKV + layout.HEAD * head + 2048 * (3 * source_group + j),
                       k, k, 1024, .5 if j == 2 else .25)
        for k in range(32):
            matrix(core_base + layout.WO + 64 * channels * head, k,
                   32 * source_group + k, 1024, .5)
        if position_only:
            for row in range(64):
                for col in range(64):
                    r, c = row % 16, col % 16
                    offset = core_base + layout.BIAS + head * 8192 + 512 * (4 * (col // 16) + row // 16)
                    offset += 64 * (r & 7) + 16 * ((c & 7) >> 1) + 2 * (c & 1) + 4 * (r >> 3) + 8 * (c >> 3)
                    half(offset, 1 / 64)
    for group in range(channels // 32):
        for k in range(32):
            matrix(core_base + layout.W1 + 64 * channels * group, 32 * group + k, k, 32 * channels, .5)
            matrix(core_base + layout.W2 + 64 * channels * group, k, 32 * group + k, 1024, .5)
    if kind in ('enc', 'enc0'):
        for n in range(cout):
            matrix(core_base + layout.PM, n % channels + channels * ((n // channels) % 4),
                   n, 128 * channels, .5)
    if kind == 'dec0':
        for n in range(40):
            matrix(core_base + layout.PM + 2 * 48, n % channels, n, 1024, .5)
            half(core_base + layout.PM + 2 * n, 1 / 64)

    input_elements = 16 * other if low else 64 * cin
    x = ((np.arange(input_elements) % 9 - 4) / 4).astype(np.float16)
    skip = ((np.arange(64 * channels) % 7 - 3) / 8).astype(np.float16)
    output_channels = 40 if kind == 'dec0' else channels
    allocations = {0: x.view(np.uint8), 1: skip.view(np.uint8), 2: weights}
    if kind in ('enc0', 'enc'):
        allocations[3] = np.full(16 * cout, np.nan, np.float16).view(np.uint8)
        allocations[4] = np.full(64 * channels, np.nan, np.float16).view(np.uint8)
    else:
        allocations[3] = np.full(64 * output_channels, np.nan, np.float16).view(np.uint8)
    parameters = bytearray(176)
    struct.pack_into('<ii', parameters, 0, 8, 8)
    pointers = {8: 0, 64: 2, 24: 3}
    if low:
        pointers[16] = 1
    if kind in ('enc0', 'enc'):
        pointers[32] = 4
    bases = {i: 0x100000000 + i * 0x1000000 for i in allocations}
    for offset, identifier in pointers.items():
        struct.pack_into('<Q', parameters, offset, bases[identifier])
    directory.mkdir(parents=True, exist_ok=True)
    for identifier, data in allocations.items():
        data.tofile(directory / f'alloc-{identifier}.bin')
    (directory / 'args.bin').write_bytes(parameters)
    lines = [f'kernel dltss_pwin_{layer}_layer', 'launch 1 1 1 32 1 4 0']
    lines += [f'alloc {i} {bases[i]:016x} {data.size}' for i, data in allocations.items()]
    lines += [f'pointer {offset} {i} 0' for offset, i in pointers.items()]
    (directory / 'manifest.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'PASS K_FIXTURE kernel={layer} weights={weight_bytes} public_synthetic=1')


def validate(layer, directory, output, window_count=1, real_capture=False):
    dump = pm.Dump(str(directory))
    outputs = Outputs(dump, str(output))
    pairs = []
    for bx, by in windows(dump.grid, window_count):
        model, extra = model_window(dump, layer, bx, by)
        pairs.append(stored(dump, outputs, layer, bx, by, model, extra))
    expected, actual = (np.concatenate(values) for values in zip(*pairs))
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError('K replay contains NaN/Inf or unwritten output')
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    maximum = float(difference.max())
    relative = float((difference / np.maximum(np.abs(expected.astype(np.float64)), 1e-3)).max())
    rms = float(np.sqrt(np.mean(difference ** 2)))
    peak = max(1., float(np.abs(expected.astype(np.float64)).max()))
    psnr = math.inf if not rms else 20 * math.log10(peak / rms)
    print(f'K_REPLAY_REFERENCE kernel={layer} elements={actual.size} max_abs={maximum:.9g} '
          f'max_rel={relative:.9g} rms={rms:.9g} psnr_db={psnr:.6g}')
    # Bound ISA approximate reciprocal/rsqrt and matrix accumulation differences;
    # require the independent PTX reference gate for every real-weight layer.
    absolute_limit = max(.001, .005 * peak) if real_capture else .001
    print(f'K_REPLAY_LIMIT windows={len(pairs)} max_abs_limit={absolute_limit:.9g} min_psnr_db=60 real_capture={int(real_capture)}')
    if maximum > absolute_limit or psnr < 60:
        worst = int(difference.argmax())
        raise RuntimeError(f'K mismatch at {worst}: reference={expected[worst]}, gpu={actual[worst]}')
    print(f'PASS K_REPLAY_REFERENCE kernel={layer} architecture=gfx1201 nan_inf=0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['generate', 'validate'])
    parser.add_argument('--layer', choices=list(LAYERS), required=True)
    parser.add_argument('--fixture-dir', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path)
    parser.add_argument('--windows', type=int, default=1)
    parser.add_argument('--real-capture', action='store_true')
    args = parser.parse_args()
    try:
        if args.mode == 'generate':
            generate(args.layer, args.fixture_dir)
        else:
            if not args.output_dir:
                parser.error('validate requires --output-dir')
            if args.windows < 1:
                parser.error('--windows must be positive')
            validate(args.layer, args.fixture_dir, args.output_dir, args.windows, args.real_capture)
    except Exception as error:
        print(f'FAIL K_REPLAY {error}', file=sys.stderr)
        raise SystemExit(5)
