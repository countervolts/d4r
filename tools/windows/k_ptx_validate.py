"""Compare real K captures with the upstream CPU PTX interpreter.

The local DLL and captures remain private. Only numerical metrics are public.
Use this independent reference to audit the NumPy model's reduction uncertainty.
"""
import argparse
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
REFERENCE = pathlib.Path(__file__).resolve().parent / 'reference'
if not (REFERENCE / 'ptxsim.py').exists():
    REFERENCE = ROOT / 'kernels/tools'
sys.path.insert(0, str(REFERENCE))
import pwin_model as pm
pm.NORM_FRAGMENT_ORDER = True
import pwin_check as pc
import ptxsim
from extract_dlss_ptx import fatbins, ptx_entries


def metrics(label, expected, actual):
    expected = expected.astype(np.float64)
    actual = actual.astype(np.float64)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError(f'{label} contains NaN/Inf')
    error = np.abs(expected - actual)
    peak = max(1., float(np.abs(expected).max()))
    maximum = float(error.max())
    relative = float((error / np.maximum(np.abs(expected), .001)).max())
    rms = float(np.sqrt(np.mean(error ** 2)))
    psnr = math.inf if rms == 0 else 20 * math.log10(peak / rms)
    print(f'{label} max_abs={maximum:.9g} max_rel={relative:.9g} rms={rms:.9g} '
          f'peak={peak:.9g} psnr_db={psnr:.6g}', flush=True)
    return maximum, peak, psnr


def validate(args):
    name = f'dltss_pwin_{args.layer}_layer'
    dump = pm.Dump(str(args.capture_dir))
    if dump.kernel != name:
        raise RuntimeError('Capture kernel does not match requested layer')
    native = pc.Outputs(dump, str(args.native_dir))
    candidates = []
    for bx, by in pc.windows(dump.grid, 12):
        model = pc.model_window(dump, args.layer, bx, by)
        expected, actual = pc.stored(dump, native, args.layer, bx, by, *model)
        difference = float(np.abs(expected.astype(np.float64) - actual.astype(np.float64)).max())
        candidates.append((difference, bx, by, model, expected, actual))
    selected = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))[:args.windows]
    pattern = re.compile(rb'\.entry\s+' + name.encode() + rb'\s*\(')
    sources = [source.rstrip(b'\0') for _, fatbin in fatbins(args.dlss_dll.read_bytes())
               for source in ptx_entries(fatbin) if pattern.search(source)]
    if not sources:
        raise RuntimeError('Local DLSS DLL does not contain the captured kernel')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (name + '.ptx')
    path.write_bytes(sources[-1])
    for _, bx, by, model, numpy_expected, actual in selected:
        _, memory, steps = ptxsim.run_block(str(path), name, str(args.capture_dir), (bx, by, 0))
        reference_output = pc.Outputs.__new__(pc.Outputs)
        reference_output.d = dump
        reference_output.allocs = {identifier: (base, data) for identifier, base, data
                                  in zip(memory.ids, memory.bases, memory.data)}
        _, expected = pc.stored(dump, reference_output, args.layer, bx, by, *model)
        print(f'PTX_REFERENCE layer={args.layer} window={bx},{by} statements={steps}', flush=True)
        maximum, peak, psnr = metrics('PTX_VS_NATIVE', expected, actual)
        metrics('PTX_VS_NUMPY', expected, numpy_expected)
        # Half reductions and approximate reciprocal/rsqrt are not bit-exact
        # across ISAs. Bound both the worst element and total error explicitly.
        if maximum > max(.001, .005 * peak) or psnr < 60:
            raise RuntimeError('Native layer differs from independent PTX reference')
    print(f'PASS K_PTX_REFERENCE layer={args.layer} windows={len(selected)} nan_inf=0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', choices=list(pc.LAYERS), required=True)
    parser.add_argument('--capture-dir', type=pathlib.Path, required=True)
    parser.add_argument('--native-dir', type=pathlib.Path, required=True)
    parser.add_argument('--dlss-dll', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--windows', type=int, default=2)
    args = parser.parse_args()
    if args.windows < 1:
        parser.error('--windows must be positive')
    try:
        validate(args)
    except Exception as error:
        print(f'FAIL K_PTX_REFERENCE {error}', file=sys.stderr)
        raise SystemExit(5)
