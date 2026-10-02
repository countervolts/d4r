"""Compare logical M inputs/outputs between private temporal capture sequences."""
import argparse
import pathlib

import numpy as np

from m_replay_validate import me, LAYERS


def logical(parameters, pointer, count, output=False, directory=None):
    base, allocation = parameters.alloc_of(pointer)
    if output:
        identifier = next(i for i, (b, _) in parameters.allocs.items() if b == base)
        allocation = np.fromfile(directory / 'gpu-output' / f'alloc-{identifier}.bin', np.uint8)
    return allocation[pointer - base:pointer - base + count]


def compare(reference, actual):
    paths = []
    order = ['enc1', 'enc2', 'enc3_tube', 'dec2', 'dec1']
    for path in reference.glob('replay-*-rrlite_*_4x4'):
        layer = path.name.split('rrlite_', 1)[1][:-4]
        ordinal = int(path.name.split('-')[1])
        frame = (ordinal - 1) // (6 if layer == 'enc3_tube' else 1)
        paths.append((frame, order.index(layer), ordinal, path, layer))
    for frame, _, ordinal, path, layer in sorted(paths):
        other = actual / path.name
        a, b = me.Params(str(path)), me.Params(str(other))
        c, _, merged, cin = LAYERS[layer]
        if (a.W, a.H, a.sx, a.sy, a.grid) != (b.W, b.H, b.sx, b.sy, b.grid):
            raise RuntimeError(f'Launch geometry differs: {path.name}')
        fields = [('input', 'inp', (a.W // 2) * (a.H // 2) * cin if cin else a.W * a.H * c, False)]
        if cin:
            fields.append(('skip', 'p32', a.W * a.H * c, False))
        fields.append(('output', 'out', a.W * a.H * c, True))
        if merged:
            fields.append(('merge', 'p48', (a.W // 2) * (a.H // 2) * merged, True))
        for label, field, count, output in fields:
            x = logical(a, getattr(a, field), count, output, path)
            y = logical(b, getattr(b, field), count, output, other)
            if x.size != count or y.size != count:
                raise RuntimeError(f'Truncated {path.name} {label}')
            differing = np.flatnonzero(x != y)
            print(f'M_CAPTURE_COMPARE frame={frame} layer={layer} ordinal={ordinal} field={label} '
                  f'codes={count} differing={differing.size} '
                  f'first={int(differing[0]) if differing.size else -1}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=pathlib.Path, required=True)
    parser.add_argument('--actual', type=pathlib.Path, required=True)
    args = parser.parse_args()
    compare(args.reference, args.actual)
