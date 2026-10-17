"""Inspect preserved synthetic K half-output differences without running a GPU."""
import argparse
import json
from pathlib import Path
import sys


def main():
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / '.tools/python/vendor'))
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'output'):
        parser.add_argument('--' + name, required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--width', required=True, type=int)
    args = parser.parse_args()
    if args.output.exists() or not args.output.is_relative_to((repo / 'build').resolve()) or args.width <= 0:
        parser.error('use a fresh JSON output under build/ and a positive width')
    names = sorted(p.name for p in args.reference.glob('output-*.rgba16f'))
    if not names or names != sorted(p.name for p in args.candidate.glob('output-*.rgba16f')):
        raise RuntimeError('Output sets differ or are empty')
    rows = []
    for name in names:
        a = np.fromfile(args.reference / name, '<f2')
        b = np.fromfile(args.candidate / name, '<f2')
        if a.size != b.size or a.size % (args.width * 4):
            raise RuntimeError('Invalid output dimensions')
        indices = np.flatnonzero(a.view('<u2') != b.view('<u2'))
        pixels = indices // 4
        finite = np.isfinite(a) & np.isfinite(b)
        max_difference = float(np.max(np.abs(a[finite].astype('f4') - b[finite].astype('f4')))) if finite.any() else None
        rows.append(dict(frame=name, changedComponents=int(indices.size),
                         channels=[int(np.count_nonzero(indices % 4 == channel)) for channel in range(4)],
                         maxAbsoluteDifference=max_difference,
                         nonfiniteReference=int(np.count_nonzero(~np.isfinite(a))),
                         nonfiniteCandidate=int(np.count_nonzero(~np.isfinite(b))),
                         finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                         boundsXY=[int((pixels % args.width).min()), int((pixels // args.width).min()),
                                   int((pixels % args.width).max()), int((pixels // args.width).max())] if indices.size else None))
    report = dict(exact=all(row['changedComponents'] == 0 for row in rows), rows=rows,
                  changedFrames=sum(row['changedComponents'] > 0 for row in rows),
                  changedComponents=sum(row['changedComponents'] for row in rows),
                  maximumAbsoluteDifference=max((row['maxAbsoluteDifference'] for row in rows if row['maxAbsoluteDifference'] is not None), default=None),
                  gameAccessed=False)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
