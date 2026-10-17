"""Compare rr_ngx_probe outputs of the synthetic noisy scene against its noise-free answer, frame by frame.

The fixture's frames.txt first line is `size RW RH OW OH`; dynamic-extents.txt (when present) gives each frame's render
size. Per frame the reference is clean-<frame>.bin (clean.bin for fixtures from the first generator version) at output
size. When a frame's render size differs from the output size the baselines are the noisy input nearest-resampled to
output size and, if clean-render-<frame>.bin exists, the noise-free render-size image resampled the same way (the best a
render-resolution result could be).
"""
import argparse
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / '.tools/python/vendor'))
import numpy as np


def resampler(render_shape, shape):
    if render_shape == shape:
        return lambda a: a
    ys = np.minimum((np.arange(shape[0]) * render_shape[0] / shape[0]).astype(int), render_shape[0] - 1)
    xs = np.minimum((np.arange(shape[1]) * render_shape[1] / shape[1]).astype(int), render_shape[1] - 1)
    return lambda a: a[ys][:, xs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--ratio', type=float, default=0.5, help='tail RMSE must be below ratio * the input baseline')
    parser.add_argument('--output-float32', action='store_true')
    parser.add_argument('--switch-output', action='store_true')
    args = parser.parse_args()
    sizes = list(map(int, (args.fixture / 'frames.txt').read_text().split('\n')[0].split()[1:5]))
    shape = (sizes[3], sizes[2], 4)  # output size
    n = len(list(args.fixture.glob('*-color.bin')))
    extents_file = args.fixture / 'dynamic-extents.txt'
    extents = [tuple(map(int, line.split())) for line in extents_file.read_text().split('\n') if line.strip()] if extents_file.exists() else None
    sky = np.load(args.fixture / 'sky.npy')
    body = ~sky
    rows = []
    for i in range(n):
        fw, fh = extents[i] if extents else (sizes[0], sizes[1])
        render_shape = (fh, fw, 4)
        resample = resampler(render_shape, shape)
        reference = args.fixture / f'clean-{i}.bin'
        clean = np.fromfile(reference if reference.exists() else args.fixture / 'clean.bin', '<f2').astype('f8').reshape(shape)[..., :3]
        float_output = args.output_float32 and (not args.switch_output or i % 2 == 1)
        out = np.fromfile(args.output / f'{i}-out.bin', '<f4' if float_output else '<f2').astype('f8').reshape(shape)[..., :3]
        noisy = resample(np.fromfile(args.fixture / f'{i}-color.bin', '<f2').astype('f8').reshape(render_shape)[..., :3])
        rmse = lambda a: float(np.sqrt(np.mean((a[body] - clean[body]) ** 2)))
        row = dict(frame=i, renderSize=[fw, fh], finite=bool(np.isfinite(out).all()), outputRmse=rmse(out), inputRmse=rmse(noisy),
                   skyMaxAbs=float(np.abs(out[sky] - clean[sky]).max()) if sky.any() else 0.0,
                   differsFromInput=bool(not np.array_equal(out, noisy)))
        ideal = args.fixture / f'clean-render-{i}.bin'
        if ideal.exists():
            row['idealNearestRmse'] = rmse(resample(np.fromfile(ideal, '<f2').astype('f8').reshape(render_shape)[..., :3]))
        rows.append(row)
    tail = rows[-8:]
    summary = dict(frames=n, creationRenderSize=sizes[:2], outputSize=sizes[2:], dynamicExtents=bool(extents),
                   allFinite=all(r['finite'] for r in rows), allDifferFromInput=all(r['differsFromInput'] for r in rows),
                   tailOutputRmse=float(np.mean([r['outputRmse'] for r in tail])), tailInputRmse=float(np.mean([r['inputRmse'] for r in tail])),
                   tailSkyMaxAbs=max(r['skyMaxAbs'] for r in tail), qualityReference=False, gameAccessed=False, synthetic=True)
    if all('idealNearestRmse' in r for r in tail):
        summary['idealNearestRmse'] = float(np.mean([r['idealNearestRmse'] for r in tail]))
    summary['denoised'] = summary['tailOutputRmse'] < args.ratio * summary['tailInputRmse']
    args.report.write_text(json.dumps(dict(summary=summary, frames=rows), indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary['allFinite'] and summary['allDifferFromInput'] and summary['denoised'] else 1)


if __name__ == '__main__':
    main()
