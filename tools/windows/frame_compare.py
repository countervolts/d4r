"""Validate every RGBA16F diagnostic frame against another complete pipeline."""
import argparse
import math
import pathlib
import sys
import numpy as np


def main(args):
    frames = sorted(args.reference.glob('output-*.rgba16f'))
    if not frames:
        raise RuntimeError('Reference contains no frames')
    for path in frames:
        actual = args.actual / path.name
        if not actual.is_file():
            raise RuntimeError(f'Missing output frame {path.name}')
        reference = np.fromfile(path, np.float16)
        output = np.fromfile(actual, np.float16)
        if reference.size != output.size or reference.size != args.width * args.height * 4:
            raise RuntimeError(f'Output size mismatch for {path.name}')
        if not np.isfinite(reference).all() or not np.isfinite(output).all():
            raise RuntimeError(f'NaN/Inf in {path.name}')
        if args.output_storage == 'r11g11b10f':
            # D3D float narrowing truncates. Compare after the independent
            # resource-format quantisation, rather than relaxing the PSNR gate.
            bits = reference.reshape(-1, 4).view(np.uint16).copy()
            bits[bits & 0x8000 != 0] = 0
            bits[:, :2] &= 0x7ff0
            bits[:, 2] &= 0x7fe0
            reference = bits.view(np.float16).reshape(-1)
        elif args.output_storage == 'bgra8':
            reference = (np.rint(np.clip(reference.astype(np.float32), 0, 1) * 255) / 255).astype(np.float16)
        x = reference.reshape(-1, 4)[:, :3].astype(np.float64)
        y = output.reshape(-1, 4)[:, :3].astype(np.float64)
        error = np.abs(x - y)
        rms = float(np.sqrt(np.mean(error ** 2)))
        peak = max(1., float(np.abs(x).max()))
        psnr = math.inf if rms == 0 else 20 * math.log10(peak / rms)
        outliers = float(np.mean(error > .01 * peak))
        print(f'FRAME_REFERENCE name={path.name} max_abs={error.max():.9g} '
              f'max_rel={(error / np.maximum(np.abs(x), .001)).max():.9g} rms={rms:.9g} psnr_db={psnr:.6g} '
              f'fraction_over_1pct={outliers:.9g}')
        if (args.exact and np.any(error != 0)) or psnr < args.min_psnr or error.max() > .025 * peak or outliers > .0001:
            raise RuntimeError(f'Complete pipeline mismatch in {path.name}')
    print(f'PASS FRAME_REFERENCE frames={len(frames)} finite=1 min_psnr_db={args.min_psnr}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=pathlib.Path, required=True)
    parser.add_argument('--actual', type=pathlib.Path, required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--min-psnr', type=float, default=60.)
    parser.add_argument('--output-storage', choices=('rgba16f', 'r11g11b10f', 'bgra8'), default='rgba16f')
    parser.add_argument('--exact', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        print(f'FAIL FRAME_REFERENCE {error}', file=sys.stderr)
        raise SystemExit(5)
