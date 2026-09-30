"""Independent PTX interpreter validation of real M captures; keep outputs private."""
import argparse
import pathlib
import re
import sys

import numpy as np

from m_replay_validate import me, LAYERS, blocks, model_block, stored, metrics
import ptxsim
from extract_dlss_ptx import fatbins, ptx_entries


def validate(args):
    name = f'rrlite_{args.layer}_4x4'
    parameters = me.Params(str(args.capture_dir))
    if parameters.kernel != name:
        raise RuntimeError('Capture kernel does not match requested M layer')
    native = {i: np.fromfile(args.native_dir / f'alloc-{i}.bin', np.uint8) for i in parameters.allocs}
    candidates = []
    coordinates = args.block or blocks(parameters.grid, args.candidate_blocks)
    for bx, by in coordinates:
        if not (0 <= bx < parameters.grid[0] and 0 <= by < parameters.grid[1]):
            raise RuntimeError(f'Block {bx},{by} is outside the capture grid')
        model = model_block(parameters, args.layer, bx, by)
        expected, actual = stored(parameters, args.layer, bx, by, native, model)
        difference = float(np.abs(expected - actual).max())
        candidates.append((difference, bx, by, model, expected, actual))
    selected = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))[:args.blocks]
    pattern = re.compile(rb'\.entry\s+' + name.encode() + rb'\s*\(')
    sources = [source.rstrip(b'\0') for _, fatbin in fatbins(args.dlss_dll.read_bytes())
               for source in ptx_entries(fatbin) if pattern.search(source)]
    if not sources:
        raise RuntimeError('Local DLSS DLL does not contain this M kernel')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (name + '.ptx')
    path.write_bytes(sources[-1])
    for _, bx, by, model, numpy_expected, actual in selected:
        _, memory, steps = ptxsim.run_block(str(path), name, str(args.capture_dir), (bx, by, 0))
        allocation_data = dict(zip(memory.ids, memory.data))
        _, expected = stored(parameters, args.layer, bx, by, allocation_data, model)
        print(f'PTX_REFERENCE layer={args.layer} block={bx},{by} statements={steps}', flush=True)
        _, _, psnr, fp8_steps = metrics(expected, actual, 'PTX_VS_NATIVE')
        metrics(expected, numpy_expected, 'PTX_VS_NUMPY')
        if fp8_steps.max() > 1 or psnr < 50:
            raise RuntimeError('Native M differs from independent PTX reference')
    print(f'PASS M_PTX_REFERENCE layer={args.layer} blocks={len(selected)} nan_inf=0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', choices=list(LAYERS), required=True)
    parser.add_argument('--capture-dir', type=pathlib.Path, required=True)
    parser.add_argument('--native-dir', type=pathlib.Path, required=True)
    parser.add_argument('--dlss-dll', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--candidate-blocks', type=int, default=12,
                        help='NumPy scan size used to choose the worst blocks for independent PTX')
    parser.add_argument('--block', type=lambda value: tuple(map(int, value.split(','))),
                        action='append', help='Explicit x,y regression block (repeatable)')
    args = parser.parse_args()
    if args.blocks < 1 or args.candidate_blocks < 1:
        parser.error('--blocks must be positive')
    try:
        validate(args)
    except Exception as error:
        print(f'FAIL M_PTX_REFERENCE {error}', file=sys.stderr)
        raise SystemExit(5)
