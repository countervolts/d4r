#!/usr/bin/env python3
"""Capture one warm RR stage with RDP CLI, using separate replay submissions.

The replay's split timestamps include CPU waits; use bench_rr.py for latency.
GPU clock settings are left to the caller and the profiler's capture policy.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time

STAGES = ('enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
          'dec4', 'dec3', 'dec2', 'dec1', 'dec0', 'hkpn')


def wait_for(condition, processes, timeout=25):
    deadline = time.monotonic() + timeout
    while not condition():
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('profiling process exited early; inspect capture/replay logs')
        if time.monotonic() >= deadline:
            raise TimeoutError('profiling handshake timed out; inspect capture/replay logs')
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'runner', 'model', 'case', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('output capture already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    capture_log = output.with_suffix('.capture.log')
    replay_log = output.with_suffix('.replay.log')
    env = {k: v for k, v in os.environ.items() if not k.startswith('D4R_')}
    cap = replay = None
    with tempfile.TemporaryDirectory(prefix='rr-profile-', dir=output.parent) as temporary:
        gate = Path(temporary) / 'stage'
        env.update(D4R_RR_PROFILE_SPLIT='1', D4R_RR_PROFILE_GATE=str(gate),
                   D4R_RR_PROFILE_BOUNDARY=str(STAGES.index(args.stage) + 1))
        with capture_log.open('w') as capture_stream, replay_log.open('w') as replay_stream:
            try:
                cap = subprocess.Popen([
                    str(args.panel.resolve()), '--process', args.runner.stem,
                    '--rgp-capture-mode', 'dispatch', '--rgp-render-op-count', '1',
                    '--rgp-instruction-tracing', '--rgp-counter-collection',
                    '--verbose', '--output', str(output)], stdin=subprocess.PIPE,
                    stdout=capture_stream, stderr=subprocess.STDOUT, text=True)
                # RDP buffers its waiting-for-application line until a client
                # connects; the API initialization line is flushed immediately.
                wait_for(lambda: 'Capture API initialized successfully' in capture_log.read_text(), [cap])
                replay = subprocess.Popen([
                    str(args.runner.resolve()), str(args.model.resolve()), str(args.case.resolve()),
                    str(output.with_suffix('.frames')), '4'], env=env,
                    stdout=replay_stream, stderr=subprocess.STDOUT)
                wait_for(lambda: Path(str(gate) + '.ready').exists() and
                         'is ready for capture' in capture_log.read_text(), [cap, replay])
                cap.stdin.write('c\n')
                cap.stdin.flush()
                wait_for(lambda: 'Successfully requested trace' in capture_log.read_text(), [cap, replay])
                Path(str(gate) + '.go').write_text('capture next stage\n')
                wait_for(lambda: 'Trace saved to:' in capture_log.read_text(), [cap], timeout=60)
                replay.wait(timeout=60)
                if replay.returncode:
                    raise RuntimeError('RR replay failed; inspect replay log')
                cap.stdin.write('q\n')
                cap.stdin.flush()
                cap.wait(timeout=15)
                if cap.returncode or not output.is_file():
                    raise RuntimeError('RGP capture failed; inspect capture log')
            finally:
                for process in (replay, cap):
                    if process is not None and process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
    print(f'{args.stage}: {output}')


if __name__ == '__main__':
    main()
