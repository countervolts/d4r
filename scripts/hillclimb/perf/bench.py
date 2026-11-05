#!/usr/bin/env python3
"""Interleaved end-to-end benchmark of d4r_nvngx.dll in d3d12_dlss_harness and vulkan_dlss_harness.

Each case runs the control and the candidate runtime folder alternately (C X X C per round, the order flipped every
round), with game-like frame spacing (the GPU-drawn scene, no sleep between frames) and GPU timestamps around
EvaluateFeature. A run's figure is the median of its measured frames; a side's figure is the median of its run
medians. Every run must log its case's route, or the case is invalid.

usage: bench.py --control RT --candidate RT [--cases REGEX] [--rounds N] [--out FILE]
       bench.py --control RT --candidate RT --self-noise ...   (control against itself: candidate = control)
The runtime folders are portable d4r folders (nvngx.dll, d4r.ini, engine\\, NVIDIA's DLLs).
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESET_IDS = {'K': 11, 'L': 12, 'M': 13}


def load_matrix(path):
    matrix = json.loads(Path(path).read_text())
    return matrix


def case_command(case, matrix, runtime, run_dir):
    """argv and environment for one run of CASE against RUNTIME, writing into RUN_DIR"""
    w, h, ow, oh = case['size']
    frames = matrix['warmup'] + matrix['measured']
    timing = run_dir / 'gpu.txt'
    log = run_dir / 'shim.log'
    out = run_dir / 'out.raw'
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('D4R_'):
            del env[key]
    env['D4R_HARNESS_GPU_TIMING'] = str(timing)
    env['D4R_SHIM_LOG'] = str(log)
    preset = case['preset']
    if case['api'] == 'd3d12':
        exe = Path(matrix['tools']) / 'd3d12_dlss_harness.exe'
        env.update({'D4R_HARNESS_GPU_SCENE': '1', 'D4R_HARNESS_MOTION_SCENE': '1', 'D4R_HARNESS_FRAME_WAIT_MS': '0',
                    'D4R_HARNESS_CREATE_FLAGS': '0x6b'})
        if preset == 'RRF':
            env.update({'D4R_HARNESS_FEATURE_ID': '13', 'D4R_RR_PRESET': '6', 'D4R_HARNESS_RR_PRESET': '6'})
        else:
            env.update({'D4R_HARNESS_DLSS_PRESET': str(PRESET_IDS[preset]), 'D4R_DLSS_PRESET': str(PRESET_IDS[preset])})
        argv = [str(exe), str(Path(runtime) / 'nvngx.dll'), str(out), str(frames), str(w), str(h), str(ow), str(oh)]
    else:
        exe = Path(matrix['tools']) / 'vulkan_dlss_harness.exe'
        env.update({'D4R_HARNESS_ENGINE_DEVICE': '1', 'D4R_HARNESS_PAN': '2', 'D4R_HARNESS_PAN_CANVAS': '1',
                    'D4R_HARNESS_CHECK_EVERY': '1000000', 'D4R_HARNESS_SIZE': f'{w} {h} {ow} {oh}'})
        if preset == 'RRF':
            env.update({'D4R_HARNESS_RR': '1', 'D4R_RR_PRESET': '6'})
        else:
            env['D4R_DLSS_PRESET'] = str(PRESET_IDS[preset])
        argv = [str(exe), str(Path(runtime) / 'nvngx.dll'), str(out), str(frames)]
    env.update({k: str(v) for k, v in case.get('env', {}).items()})
    return argv, env, timing, log, out


def run_once(case, matrix, runtime, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    argv, env, timing, log, out = case_command(case, matrix, runtime, run_dir)
    for f in (timing, log, out):
        if f.exists():
            f.unlink()
    started = time.time()
    try:
        proc = subprocess.run(argv, env=env, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=matrix.get('timeout', 300))
        code, text = proc.returncode, proc.stdout.decode(errors='replace')
    except subprocess.TimeoutExpired as e:
        code, text = 'timeout', (e.stdout or b'').decode(errors='replace')
    (run_dir / 'harness.log').write_text(text)
    result = {'exit': code, 'seconds': round(time.time() - started, 1)}
    shim = log.read_text(errors='replace') if log.exists() else ''
    route = case.get('route')
    result['route_ok'] = bool(route and re.search(route, shim))
    bad = [line for line in shim.splitlines() if re.search(r'\b(refused|failed|rejected)\b', line)
           and not re.search(matrix.get('benign', '$^'), line)]
    result['bad_lines'] = bad[:5]
    samples = []
    if timing.exists():
        for line in timing.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                samples.append(float(parts[1]) / 1000.0)
    measured = samples[matrix['warmup']:]
    result['n'] = len(measured)
    if measured:
        ordered = sorted(measured)
        result['median'] = statistics.median(measured)
        result['p99'] = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
        result['min'] = ordered[0]
    result['samples'] = [round(s, 4) for s in measured]
    if out.exists():
        result['out_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
    result['ok'] = code == 0 and result['route_ok'] and len(measured) >= matrix['measured'] * 0.9
    return result


GPU_LOAD_PS = (
    r"$c = Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 2 -ErrorAction SilentlyContinue; "
    "$c.CounterSamples | Where-Object CookedValue -gt 0.5 | Group-Object { ($_.InstanceName -split '_')[1] } | "
    "ForEach-Object { $p=[int]$_.Name; $n=(Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName; "
    "'{0} {1:N1}' -f $n,(($_.Group | Measure-Object CookedValue -Sum).Sum/2) }")
IDLE_USERS = {'dwm', 'System', 'Idle', '', 'WindowsTerminal', 'explorer', 'zen', 'Discord', 'd3d12_dlss_harness',
              'vulkan_dlss_harness', 'csrss', 'Code', 'claude'}


def other_gpu_load():
    """Summed GPU engine utilization (%) of processes other than the desktop's, and who they are"""
    try:
        text = subprocess.run(['powershell', '-NoProfile', '-Command', GPU_LOAD_PS], capture_output=True,
                              timeout=30).stdout.decode(errors='replace')
    except Exception:
        return 0.0, []
    users = []
    for line in text.splitlines():
        parts = line.rsplit(' ', 1)
        if len(parts) == 2:
            try:
                users.append((parts[0], float(parts[1].replace(',', '.'))))
            except ValueError:
                pass
    busy = [(n, v) for n, v in users if n not in IDLE_USERS]
    return sum(v for _, v in busy), busy


class GpuWatch:
    """Samples other processes' GPU use in the background for the length of a case (each sample is two 1 s counter
    readings), keeping the worst sample and who caused it."""
    def __init__(self):
        self.worst, self.users, self.samples = 0.0, [], 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._stop.is_set():
            load, users = other_gpu_load()
            self.samples += 1
            if load > self.worst:
                self.worst, self.users = load, users
    def __enter__(self):
        self._thread.start()
        return self
    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=30)


def summarize(runs):
    good = [r for r in runs if r.get('ok')]
    if not good:
        return {'ok': False}
    medians = [r['median'] for r in good]
    p99s = [r['p99'] for r in good]
    hashes = sorted({r.get('out_sha256') for r in good})
    return {'ok': len(good) == len(runs), 'median': statistics.median(medians), 'run_medians': medians,
            'spread': (max(medians) - min(medians)) / statistics.median(medians),
            'p99': statistics.median(p99s), 'hashes': hashes}


def geomean(values):
    values = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--matrix', default=str(HERE / 'matrix.json'))
    parser.add_argument('--control', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--cases', default='.', help='regex on case ids')
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--work', default='C:/d4r-hc/bench')
    parser.add_argument('--out', default=None)
    a = parser.parse_args()
    matrix = load_matrix(a.matrix)
    rounds = a.rounds or matrix['rounds']
    cases = [c for c in matrix['cases'] if re.search(a.cases, c['id'])]
    stamp = time.strftime('%Y%m%d-%H%M%S')
    work = Path(a.work) / stamp
    report = {'control': a.control, 'candidate': a.candidate, 'rounds': rounds, 'started': stamp, 'cases': {}}
    for case in cases:
        load, users = other_gpu_load()
        if load > matrix.get('max_other_gpu', 10):
            print(f"{case['id']}: other GPU users {users}; waiting for them to finish", flush=True)
            while load > matrix.get('max_other_gpu', 10):
                time.sleep(60)
                load, users = other_gpu_load()
        runs = {'control': [], 'candidate': []}
        watch = GpuWatch().__enter__()
        for r in range(rounds):
            order = ['control', 'candidate', 'candidate', 'control']
            if r % 2:
                order = order[::-1]
            for i, side in enumerate(order):
                runtime = a.control if side == 'control' else a.candidate
                res = run_once(case, matrix, runtime, work / case['id'] / f'{side}-{r}-{i}')
                runs[side].append(res)
                flag = '' if res['ok'] else f"  INVALID exit={res['exit']} route={res['route_ok']} n={res['n']}"
                print(f"{case['id']:<22} {side:<9} r{r} {res.get('median', float('nan')):7.3f} ms "
                      f"p99 {res.get('p99', float('nan')):7.3f}{flag}", flush=True)
        watch.__exit__()
        ctl, cand = summarize(runs['control']), summarize(runs['candidate'])
        after, users_after = watch.worst, watch.users
        entry = {'case': case, 'control': ctl, 'candidate': cand, 'other_gpu_worst': users_after,
                 'other_gpu_samples': watch.samples, 'busy': after > matrix.get('max_other_gpu', 10),
                 'runs': {s: [{k: v for k, v in r.items() if k != 'samples'} for r in rs] for s, rs in runs.items()}}
        if ctl.get('median') and cand.get('median'):
            entry['delta'] = cand['median'] / ctl['median'] - 1
            entry['identical'] = ctl['hashes'] == cand['hashes'] and len(ctl['hashes']) == 1
            noise = max(ctl['spread'], cand['spread'])
            if entry['busy']:
                print(f"!! {case['id']}: other GPU users during the case {users_after}: result unreliable", flush=True)
            print(f"== {case['id']}: control {ctl['median']:.3f}  candidate {cand['median']:.3f}  "
                  f"({100 * entry['delta']:+.2f}%, spread {100 * noise:.2f}%, identical={entry['identical']})",
                  flush=True)
        report['cases'][case['id']] = entry
    for side in ('control', 'candidate'):
        per_api = {}
        for api in ('d3d12', 'vulkan'):
            per_api[api] = geomean([e[side].get('median') for e in report['cases'].values()
                                    if e['case']['api'] == api])
        report[f'score_{side}'] = {**per_api, 'headline': geomean([v for v in per_api.values() if v])}
    print('score control  ', report['score_control'])
    print('score candidate', report['score_candidate'])
    out = Path(a.out) if a.out else work / 'result.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    sys.exit(main())
