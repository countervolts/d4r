#!/usr/bin/env python3
"""Interleaved warmed benchmark of d4r's local routes against the official NVIDIA preset timings.

Cadence per run, matching tools/bench_official_ngx_cuda.cpp:
  SR (d4r-m): 4 frames with the probe's jitter cycle; 48 untimed evaluations (12 passes), then
              25 passes = 100 timed evaluations. GPU stamps bracket exposure..downsample; the
              host-side input copies and the pass-0 output readback stay outside them.
  RR (d4r-rr): 140 frames.txt entries over one reused static frame; the first 40 are warmup, the
              remaining 100 are measured. Stamps bracket exposure..output; input uploads happen once
              (D4R_RR_NO_UPLOAD) and debug images are not written (D4R_RR_NO_WRITE).
Three trials per geometry group, the run order rotated and reversed so backends alternate; the
reported figure is the median of the three per-run medians (the same statistic as the NVIDIA table).

usage: bench_suite.py GEOMETRY GROUP
  GEOMETRY: 1440p | 4k ; GROUP: sr | rr
"""
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'build/local-preset-compare-20261029'
RUNNER = Path('/tmp/d4r-preset-compare-20261029/engine')
HW = Path('/sys/class/drm/card1/device')
JITTER = [(-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)]
SR_WARMUP_PASSES = 12           # 12 passes * 4 frames = 48 untimed evaluations
SR_REPEATS = 37                 # passes 13..37 -> wait, see sr_command(): timed = (repeats - warmup) * 4

GEOMETRY = {'1440p': (1280, 720, 2560, 1440), '4k': (1920, 1080, 3840, 2160)}


class Sampler(threading.Thread):
    """Records sclk, mclk, PPT, junction temperature and gpu_busy_percent while a run is live."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop = False

    def run(self):
        while not self.stop:
            s = {}
            for key, node, scale in (('sclk_mhz', 'hwmon/hwmon3/freq1_input', 1e-6), ('mclk_mhz', 'hwmon/hwmon3/freq2_input', 1e-6),
                                     ('ppt_w', 'hwmon/hwmon3/power1_average', 1e-6), ('junction_c', 'hwmon/hwmon3/temp2_input', 1e-3),
                                     ('busy_percent', 'gpu_busy_percent', 1)):
                try:
                    s[key] = int((HW / node).read_text()) * scale
                except Exception:
                    pass
            self.samples.append(s)
            time.sleep(0.2)

    def summary(self):
        out = {}
        for key in ('sclk_mhz', 'mclk_mhz', 'ppt_w', 'junction_c', 'busy_percent'):
            v = [s[key] for s in self.samples if key in s]
            if v:
                out[key] = {'median': round(statistics.median(v), 1), 'min': round(min(v), 1),
                            'max': round(max(v), 1)}
        return out


def sr_command(preset, rw, rh, ow, oh, backend, prefix):
    inputs = W / f'inputs/sr-{PREFIX_GEOM}'
    args = []
    for f, (jx, jy) in enumerate(JITTER):
        args += [str(inputs / f'{f}-color.bin'), str(inputs / f'{f}-motion.bin'), str(inputs / f'{f}-depth.bin'),
                 str(jx), str(jy)]
    cmd = [str(RUNNER / 'd4r-m'), str(W / f'models/sr-{preset}'), str(rw), str(rh), str(ow), str(oh), str(prefix),
           str(SR_REPEATS)]
    if backend != 'vulkan':
        cmd.append(backend)
    cmd += ['--'] + args
    env = dict(os.environ, D4R_M_WARMUP_PASSES=str(SR_WARMUP_PASSES), D4R_M_TIME_EACH='1')
    return cmd, env


def rr_command(model, backend, prefix):
    cmd = [str(RUNNER / 'd4r-rr'), str(W / model), str(W / f'inputs/rr-{PREFIX_GEOM}'), str(prefix)]
    env = dict(os.environ, D4R_RR_TIME_REPEAT='1', D4R_RR_NO_UPLOAD='1', D4R_RR_NO_WRITE='1')
    if backend == 'vkhip':
        env['D4R_ENGINE_VKHIP'] = '1'
    return cmd, env


STAGE_NAMES = ['exposure', 'input', 'network', 'expansion', 'reconstruction', 'downsample']


def parse_sr(text):
    times = [float(m.group(1)) for m in re.finditer(r'^TIMED pass=\d+ frame=\d+ ([\d.]+) us$', text, re.M)]
    stages = {}
    for name in STAGE_NAMES:
        m = re.search(rf'{name}\s+median ([\d.]+) us', text)
        if m:
            stages[name] = float(m.group(1))
    return times, stages


def parse_rr(text):
    times = [float(m.group(1)) for m in re.finditer(r'^frame \d+: ([\d.]+) ms', text, re.M)]
    stages = {}
    for key in ('exposure', 'network', 'output'):
        v = [float(m.group(1)) for m in re.finditer(rf'{key} ([\d.]+)\)', text, re.M)]
        if v:
            stages[key] = statistics.median(v[len(v) // 2:]) if len(v) > 40 else statistics.median(v)
    return times, stages


def run_case(case, backend, trial):
    rw, rh, ow, oh = GEOMETRY[PREFIX_GEOM]
    prefix = W / 'logs' / f'{case["id"]}-{backend}-trial{trial}'
    cmd, env = sr_command(case['preset'], rw, rh, ow, oh, backend, prefix) if case['kind'] == 'sr' \
        else rr_command(case['model'], backend, prefix)
    if backend == 'vkhip' and not (W / case['model'] / 'hip').is_dir():
        return {'case': case['id'], 'backend': backend, 'trial': trial, 'skipped': 'no hip/ code objects'}
    if backend in ('vkhip', 'hip') and case['kind'] == 'sr' and not (W / case['model'] / 'hipnet.bin').is_file():
        return {'case': case['id'], 'backend': backend, 'trial': trial, 'skipped': 'no hipnet.bin'}
    sampler = Sampler()
    sampler.start()
    started = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.time() - started
    sampler.stop = True
    time.sleep(0.25)
    text = result.stdout + result.stderr
    prefix.with_suffix('.log').write_text(text)
    if result.returncode:
        return {'case': case['id'], 'backend': backend, 'trial': trial, 'failed': result.returncode,
                'tail': text[-400:], 'command': ' '.join(cmd)}
    times, stages = parse_sr(text) if case['kind'] == 'sr' else parse_rr(text)
    if case['kind'] == 'rr':
        times = times[40:]          # the first 40 entries are warmup
    if len(times) < 100:
        return {'case': case['id'], 'backend': backend, 'trial': trial, 'failed': f'only {len(times)} timed frames'}
    note = []
    if backend == 'vkhip':
        m = re.search(r'vkhip: (\d+) Ray Reconstruction stages', text)
        if m:
            note.append(f'{m.group(1)} vkhip stages')
        if case['kind'] == 'sr':
            note.append('vkhip native layers')
    if 'vkhip unavailable' in text:
        note.append('VKHIP FALLBACK: ' + re.search(r'vkhip unavailable: (.*)', text).group(1))
    return {'case': case['id'], 'backend': backend, 'trial': trial, 'frames': len(times),
            'median_ms': round(statistics.median(times) / (1000 if case['kind'] == 'sr' else 1), 6),
            'min_ms': round(min(times) / (1000 if case['kind'] == 'sr' else 1), 6),
            'p10_ms': round(sorted(times)[len(times) // 10] / (1000 if case['kind'] == 'sr' else 1), 6),
            'p90_ms': round(sorted(times)[len(times) * 9 // 10] / (1000 if case['kind'] == 'sr' else 1), 6),
            'stages_us': {k: v for k, v in stages.items()}, 'wall_s': round(wall, 2),
            'gpu': sampler.summary(), 'note': note, 'command': ' '.join(cmd)}


def main():
    global PREFIX_GEOM
    geom = sys.argv[1]
    group = sys.argv[2]
    PREFIX_GEOM = geom
    cases = []
    if group == 'sr':
        for preset in ('m', 'l'):
            for backend in ('vulkan', 'vkhip', 'hip'):
                cases.append({'id': f'sr-{preset}', 'preset': preset, 'kind': 'sr', 'model': f'models/sr-{preset}',
                              'backend': backend})
    else:
        for preset in ('a', 'b', 'c', 'd', 'e', 'f'):
            model = f'models-cnn/rr-{preset}' if preset in 'abc' else f'models/rr-{preset}'
            for backend in ('vulkan', 'vkhip'):
                cases.append({'id': f'rr-{preset}', 'preset': preset, 'kind': 'rr', 'model': model,
                              'backend': backend})
    order = []
    for trial in range(3):
        rotated = cases[trial:] + cases[:trial]
        order += [(entry, trial) for entry in (list(reversed(rotated)) if trial % 2 else rotated)]
    results = []
    for entry, trial in order:
        r = run_case(entry, entry['backend'], trial)
        results.append(r)
        print(json.dumps(r), flush=True)
    out = W / f'logs/suite-{group}-{geom}.json'
    out.write_text(json.dumps({'geometry': GEOMETRY[geom], 'group': group, 'cadence': {
        'sr': {'frames': 4, 'warmup_evaluations': SR_WARMUP_PASSES * 4, 'timed_evaluations': (SR_REPEATS - SR_WARMUP_PASSES) * 4},
        'rr': {'entries': 140, 'warmup': 40, 'measured': 100}}, 'runs': results}, indent=1))
    print(f'wrote {out}')


main()
