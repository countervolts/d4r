"""Validate every captured M launch and spatial block; all allocations stay private."""
import argparse
import contextlib
import json
import pathlib
import sys
import traceback

from m_replay_validate import LAYERS, validate


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {'passed': False, 'captured_outputs': True, 'exact': args.exact, 'launches': []}
    status = 1
    try:
        captures = sorted(args.capture_dir.glob('replay-*-rrlite_*_4x4'))
        if not captures:
            raise RuntimeError('No audited M captures were found')
        for capture in captures:
            layer = capture.name.split('rrlite_', 1)[1][:-4]
            if layer not in LAYERS:
                raise RuntimeError(f'Unknown captured M layer: {layer}')
            logfile = args.output_dir / (capture.name + '.log')
            with logfile.open('w', encoding='utf-8') as output, contextlib.redirect_stdout(output):
                validate(layer, capture, capture / 'gpu-output', args.blocks, True, args.exact)
            print(capture.name + ' ' + logfile.read_text().splitlines()[0], flush=True)
            summary['launches'].append({'capture': capture.name, 'layer': layer, 'passed': True})
        present = {item['layer'] for item in summary['launches']}
        if present != set(LAYERS):
            raise RuntimeError('Capture set does not contain all five M layers')
        summary['passed'] = True
        print(f'PASS M_TEMPORAL_REFERENCE launches={len(captures)} blocks={args.blocks} exact={int(args.exact)}', flush=True)
        status = 0
    except Exception as error:
        summary['error'] = str(error)
        summary['failed_capture'] = capture.name if 'capture' in locals() else None
        traceback.print_exc()
    finally:
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--blocks', type=int, default=10000)
    parser.add_argument('--exact', action='store_true')
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error('--blocks must be positive')
    sys.exit(main(args))
