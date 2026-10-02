#!/usr/bin/env python3
"""Compare RDNA4 K enc1/enc2 launches with the existing numpy reference."""
import argparse
import math
import pathlib
import sys

import numpy as np

SCRIPT = pathlib.Path(__file__).resolve()
REFERENCE = SCRIPT.parent / "reference"
if not (REFERENCE / "pwin_model.py").exists():
    REFERENCE = SCRIPT.parents[2] / "kernels" / "tools"
sys.path.insert(0, str(REFERENCE))
from pwin_model import pwin_encoder  # noqa: E402


class SyntheticParams:
    def __init__(self, folder):
        self.input = np.fromfile(folder / "input.bin", dtype=np.uint8)
        self.weights = np.fromfile(folder / "weights.bin", dtype=np.uint8)

    def i32x2(self, offset):
        return (8, 8) if offset == 0 else (0, 0)

    def ptr(self, offset):
        if offset == 8:
            return self.input, 0
        if offset == 64:
            return self.weights, 0
        raise ValueError(f"unsupported synthetic pointer offset {offset}")

    def f16(self, offset, byte, count):
        if offset != 64:
            raise ValueError(f"unsupported synthetic half pointer offset {offset}")
        return self.weights[byte:byte + 2 * count].view(np.float16).copy()


def compare(name, actual, expected):
    if actual.shape != expected.shape:
        raise RuntimeError(f"{name} shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError(f"{name} contains NaN or Inf")
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    maximum = float(diff.max())
    relative = float((diff / np.maximum(np.abs(expected.astype(np.float32)), 1e-3)).max())
    rms = float(np.sqrt(np.mean(np.square(diff.astype(np.float64)))))
    peak = max(1.0, float(np.abs(expected.astype(np.float32)).max()))
    psnr = math.inf if rms == 0 else 20.0 * math.log10(peak / rms)
    mismatch = np.unravel_index(int(diff.argmax()), diff.shape)
    print(f"K_REFERENCE {name} max_abs={maximum:.9g} max_rel={relative:.9g} "
          f"rms={rms:.9g} psnr_db={psnr:.5g} worst={mismatch} "
          f"expected={float(expected[mismatch]):.9g} actual={float(actual[mismatch]):.9g}")
    return maximum, psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", required=True, type=pathlib.Path)
    parser.add_argument("--kernel-name", choices=("enc1", "enc2"), default="enc1")
    args = parser.parse_args()
    folder = args.fixture_dir
    params = SyntheticParams(folder)
    output_channels = 96 if args.kernel_name == "enc2" else 64
    reference_full, reference_merged = pwin_encoder(
        params, 0, 0, H=2, C=64, COUT=output_channels,
        posattn=args.kernel_name == "enc1")
    actual_full = np.fromfile(folder / "full.bin", np.float16).reshape(64, 64)
    actual_merged = np.fromfile(folder / "merged.bin", np.float16).reshape(16, output_channels)
    full_abs, full_psnr = compare("full", actual_full, reference_full)
    merged_abs, merged_psnr = compare("merged", actual_merged, reference_merged)
    # Initial acceptance for the synthetic nonzero FP16 fixture. Tighten this
    # after each K stage has been compared on captured DLSS weights/activations.
    if full_abs > 0.001 or full_psnr < 60.0 or merged_abs > 0.001 or merged_psnr < 60.0:
        raise RuntimeError(f"K {args.kernel_name} numerical mismatch against pwin_model.py")
    print(f"PASS K_REFERENCE reference_layout=gfx12 kernel={args.kernel_name} "
          "nonzero_weights=1 transformer_executed=1 nan_inf=0")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL K_REFERENCE {error}", file=sys.stderr)
        raise SystemExit(5)
