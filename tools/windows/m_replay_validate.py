"""Public synthetic M fixtures and strict FP16-widening replay validation.

No NVIDIA binaries, weights or extracted PTX are needed to generate fixtures.
The wire format is upstream's replay format, also accepted by native_replay_probe.
"""
import argparse
import math
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
REFERENCE = pathlib.Path(__file__).resolve().parent / 'reference'
if not (REFERENCE / 'swin_check.py').exists():
    REFERENCE = ROOT / 'kernels/tools'
sys.path.insert(0, str(REFERENCE))
import model_enc3 as me
import swin_model as sm
import swin_check
swin_check.CLAMP_PATCH_MERGE = True
from swin_check import LAYERS, blocks, model_block, codes_at


def encode(values):
    """Encode already quantised finite values without depending on vendor FP8."""
    values = me.q8(values)
    table = me.E4M3[:127]
    codes = np.searchsorted(table, np.abs(values)).astype(np.uint8)
    return codes | ((values < 0).astype(np.uint8) << 7)


def generate(layer, directory, size):
    C, heads, NPM, CIN = LAYERS[layer]
    pre = 4 * C * CIN + 8 * C if CIN else 0
    layout = sm.layout(C, heads, pre)
    weights = np.zeros(layout['end'] + 4 * C * NPM + 2 * NPM, np.uint8)

    def half(offset, values):
        data = np.asarray(values, np.float16).reshape(-1).view(np.uint8)
        weights[offset:offset + data.size] = data

    def matrix(base, ks, ns, k, n, source, dest, value=.25):
        offsets = sm.woff_table(base, ks, ns, k, n)
        weights[offsets[source, dest]] = encode(np.array(value))

    if CIN:
        cols = 4 * C // heads
        for head in range(heads):
            for n in range(cols):
                matrix(head * CIN * cols, 512, 16 * CIN, CIN, cols,
                       (head * cols + n) % CIN, n)
        half(4 * C * CIN, np.full(4 * C, 1 / 64))
    half(layout['g1'], np.ones(C))
    half(layout['g2'], np.ones(C))
    half(layout['bo'], np.full(C, 1 / 64))
    half(layout['b2'], np.full(C, 1 / 64))
    for head, base in enumerate(layout['head']):
        for n in range(32):
            source = (32 * head + n) % C
            matrix(base, 1024, 512, C, 32, source, n)
            matrix(base + 32 * C, 1024, 512, C, 32, source, n, .5)
            matrix(base + 64 * C + 512, 0, 512, 32, C, n, source, .5)
    for group, base in enumerate(layout['b1']):
        half(base + 32 * C, np.full(32, 1 / 64))
        for n in range(32):
            source = (32 * group + n) % C
            matrix(base, 512, 16 * C, C, 32, source, n, .5)
            matrix(base + 32 * C + 64, 0, 512, 32, C, n, source, .25)
    for group in range(NPM // 32):
        for n in range(32):
            matrix(layout['end'] + group * 128 * C, 512, 64 * C,
                   4 * C, 32, (group * 32 + n) % (4 * C), n, 8.)
        half(layout['end'] + 4 * C * NPM + 64 * group, np.full(32, 1 / 64))

    input_count = size * size * C if not CIN else (size // 2) ** 2 * CIN
    input_data = encode((np.arange(input_count) % 9 - 4) / 4)
    skip = encode((np.arange(size * size * C) % 7 - 3) / 8) if CIN else np.empty(0, np.uint8)
    allocations = {0: weights, 1: np.concatenate([input_data, skip]),
                   2: np.full(size * size * C, 0x7f, np.uint8)}
    if NPM:
        allocations[3] = np.full((size // 2) ** 2 * NPM, 0x7f, np.uint8)
    bases = {i: 0x100000000 + i * 0x1000000 for i in allocations}
    parameters = bytearray(88 if layer == 'enc3_tube' else 56)
    struct.pack_into('<iiii', parameters, 8, 0, 0, size, size)
    pointers = {0: (0, 0), 24: (1, 0), 40: (2, 0)}
    if CIN:
        pointers[32] = (1, input_count)
    if NPM:
        pointers[48] = (3, 0)
    for offset, (identifier, delta) in pointers.items():
        struct.pack_into('<Q', parameters, offset, bases[identifier] + delta)
    directory.mkdir(parents=True, exist_ok=True)
    for identifier, data in allocations.items():
        data.tofile(directory / f'alloc-{identifier}.bin')
    (directory / 'args.bin').write_bytes(parameters)
    lines = [f'kernel rrlite_{layer}_4x4', f'launch {size // 8} {size // 8} 1 32 1 {heads} 0']
    lines += [f'alloc {i} {bases[i]:016x} {data.size}' for i, data in allocations.items()]
    lines += [f'pointer {offset} {i} {delta}' for offset, (i, delta) in pointers.items()]
    (directory / 'manifest.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'PASS M_FIXTURE kernel={layer} size={size} weights={weights.size} public_synthetic=1')


def stored(parameters, layer, bx, by, allocations, model):
    C, _, NPM, _ = LAYERS[layer]

    def read(ptr, w, h, x, y, channels):
        base, _ = parameters.alloc_of(ptr)
        identifier = next(i for i, (b, _) in parameters.allocs.items() if b == base)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        codes = codes_at(allocations[identifier], base, ptr, w, h, x[valid], y[valid], channels)
        return me.E4M3[codes].ravel(), valid

    x, y = parameters.token_xy(bx, by)
    actual, valid = read(parameters.out, parameters.W, parameters.H, x, y, C)
    got, want = [actual], [model[0][valid].ravel()]
    if NPM:
        r = np.arange(16)
        mx = (8 * bx - parameters.sx) // 2 + (r & 3)
        my = (8 * by - parameters.sy) // 2 + (r >> 2)
        actual, valid = read(parameters.p48, parameters.W // 2, parameters.H // 2, mx, my, NPM)
        got.append(actual)
        want.append(model[1][valid].ravel())
    return np.concatenate(want), np.concatenate(got)


def metrics(expected, actual, label):
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError(f'{label}: NaN/Inf or unwritten output')
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    peak = max(1., float(np.abs(expected).max()))
    maximum = float(error.max())
    relative = float((error / np.maximum(np.abs(expected), .001)).max())
    rms = float(np.sqrt(np.mean(error ** 2)))
    psnr = math.inf if rms == 0 else 20 * math.log10(peak / rms)
    rank = lambda v: np.searchsorted(me._POS, np.abs(v)) * np.sign(v)
    steps = np.abs(rank(actual) - rank(expected))
    print(f'{label} elements={actual.size} max_abs={maximum:.9g} max_rel={relative:.9g} '
          f'rms={rms:.9g} psnr_db={psnr:.6g} identical={np.mean(error == 0):.9g} '
          f'max_fp8_steps={int(steps.max())}', flush=True)
    return maximum, peak, psnr, steps


def validate(layer, directory, output, block_count, real_capture):
    parameters = me.Params(str(directory))
    if parameters.kernel != f'rrlite_{layer}_4x4':
        raise RuntimeError('Replay kernel does not match requested M layer')
    allocations = {i: np.fromfile(output / f'alloc-{i}.bin', np.uint8) for i in parameters.allocs}
    pairs = [stored(parameters, layer, bx, by, allocations, model_block(parameters, layer, bx, by))
             for bx, by in blocks(parameters.grid, block_count)]
    expected, actual = (np.concatenate(values) for values in zip(*pairs))
    maximum, peak, psnr, steps = metrics(expected, actual, f'M_REPLAY_REFERENCE kernel={layer}')
    if (not real_capture and maximum != 0) or (real_capture and (steps.max() > 1 or psnr < 50)):
        raise RuntimeError(f'M mismatch: exact synthetic / <=1 FP8 step and >=50 dB real gate; peak={peak}')
    print(f'PASS M_REPLAY_REFERENCE kernel={layer} blocks={len(pairs)} architecture=gfx1201 nan_inf=0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['generate', 'validate'])
    parser.add_argument('--layer', choices=list(LAYERS), required=True)
    parser.add_argument('--fixture-dir', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path)
    parser.add_argument('--size', type=int, choices=[8, 16, 24, 32], default=16)
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--real-capture', action='store_true')
    args = parser.parse_args()
    try:
        if args.mode == 'generate':
            generate(args.layer, args.fixture_dir, args.size)
        else:
            if not args.output_dir or args.blocks < 1:
                parser.error('validate requires --output-dir and positive --blocks')
            validate(args.layer, args.fixture_dir, args.output_dir, args.blocks, args.real_capture)
    except Exception as error:
        print(f'FAIL M_REPLAY {error}', file=sys.stderr)
        raise SystemExit(5)
