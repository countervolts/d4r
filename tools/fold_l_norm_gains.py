#!/usr/bin/env python3
"""Fold selected outer-L gains into prepared FP8 consumer weights.

Changes quantization: build matching M_AMD_FOLD_* shaders and check reference
PSNR. No selection folds both norms in both outer stages (the original CLI).
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

POS = np.array([(c & 7) / 8 * 2.0 ** -6 if c >> 3 == 0 else
                (1 + (c & 7) / 8) * 2.0 ** ((c >> 3) - 7) for c in range(127)])
DECODE = np.concatenate([POS, [np.nan], -POS, [np.nan]])
SELECTIONS = ('enc:n1', 'enc:n2', 'dec:n1', 'dec:n2')


def encode(values):
    a = np.minimum(np.abs(values), 448.0)
    i = np.clip(np.searchsorted(POS, a), 1, 126)
    lo, hi = POS[i - 1], POS[i]
    upper = (a - lo > hi - a) | ((a - lo == hi - a) & (i % 2 == 0))
    return (np.where(upper, i, i - 1) | np.where(np.signbit(values), 128, 0)).astype(np.uint8)


def fold(model, selected=SELECTIONS):
    model = Path(model)
    if not set(selected) <= set(SELECTIONS):
        raise ValueError('unknown normalization selection')
    head = (model / 'offsets.bin').read_bytes()
    if head[:8] not in (b'D4RL0001', b'D4RL0002') or (len(head) - 40) % 296:
        raise ValueError('requires a compiled L model')
    nblk = (len(head) - 40) // 296
    w8 = np.fromfile(model / 'weights8.bin', np.uint8)
    w16 = np.fromfile(model / 'weights16.bin', np.float16)
    report = []
    for stage, block in (('enc', 0), ('dec', nblk - 1)):
        o8, o16, *o = struct.unpack_from('<74I', head, 40 + block * 296)
        for number, gi in ((1, 0), (2, 2)):
            selection = f'{stage}:n{number}'
            if selection not in selected:
                continue
            gain = w16[o16 // 2 + o[gi]:o16 // 2 + o[gi] + 32].astype(np.float64)
            if not np.isfinite(gain).all():
                raise ValueError('nonfinite norm gains')
            targets = ([o[4 + 4 * h] for h in range(2)] + [o[5 + 4 * h] for h in range(2)]
                       if number == 1 else [o[20 + 3 * c] for c in range(4)])
            max_error = 0.0
            for offset in targets:
                start = o8 + offset
                codes = w8[start:start + 1024].reshape(2, 2, 16, 16)
                values = DECODE[codes] * gain.reshape(2, 1, 1, 16)
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite prepared weights')
                packed = encode(values)
                max_error = max(max_error, float(np.max(np.abs(DECODE[packed] - values))))
                w8[start:start + 1024] = packed.ravel()
            w16[o16 // 2 + o[gi]:o16 // 2 + o[gi] + 32] = 1.0
            report.append(dict(selection=selection, gain_min=float(gain.min()), gain_max=float(gain.max()),
                               weight_max_abs_quantization_error=max_error))
    w8.tofile(model / 'weights8.bin')
    w16.tofile(model / 'weights16.bin')
    manifest = json.loads((model / 'model.json').read_text())
    manifest['folded_outer_norms'] = report
    manifest['files'] = {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                         for name in manifest['files']}
    (model / 'model.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--select', choices=SELECTIONS, action='append')
    args = parser.parse_args()
    print(json.dumps(fold(args.model, args.select or SELECTIONS), indent=1))
