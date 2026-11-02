#!/usr/bin/env python3
"""Per-stage GPU profiles for Ray Reconstruction presets, Vulkan and vkhip routes.

D4R_ENGINE_RR_STAGE_STAMPS=1 makes the engine write a timestamp after every network stage; d4r-rr prints
the minimum over D4R_RR_TIME_REPEAT repeats of each entry. The first entries are warmup and are dropped.

usage: profile_rr_stages.py
Writes logs/profile-rr-stages.json plus logs/profile-<preset>-<backend>-<geom>.log.
"""
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'build/local-preset-compare-20261029'
RUNNER = Path('/tmp/d4r-preset-compare-20261029/engine')
MAKES = Path('/tmp/d4r-preset-compare-20261029/make_rr_inputs')

# stamp index -> stage name, per pipeline family (engine/runtime_rr.inc, runtime_rr2.inc, runtime_rr_cnn.inc)
SWIN = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0']
CNN = ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Dec4', 'Dec3', 'Dec2', 'Dec1', 'Dec0', 'PDL']
RR2 = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4_0', 'dec4_1', 'dec3_0', 'dec3_1',
       'dec2_0', 'dec2_1', 'dec1_0', 'dec1_1', 'dec0_0', 'dec0_1']
CASES = [('a', 'models-cnn/rr-a', CNN), ('d', 'models/rr-d', SWIN), ('e', 'models/rr-e', SWIN), ('f', 'models/rr-f', RR2)]
GEOMETRY = {'1440p': (1280, 720, 2560, 1440), '4k': (1920, 1080, 3840, 2160)}
ENTRIES = 8            # 3 warmup entries, 5 profiled


def main():
    report = []
    for geom, (rw, rh, ow, oh) in GEOMETRY.items():
        inputs = W / f'inputs/rr-prof-{geom}'
        inputs.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(MAKES), str(inputs), str(rw), str(rh), str(ow), str(oh), str(ENTRIES)], check=True)
        for preset, model, names in CASES:
            for backend in ('vulkan', 'vkhip'):
                env = dict(os.environ, D4R_ENGINE_RR_STAGE_STAMPS='1', D4R_RR_NO_UPLOAD='1', D4R_RR_NO_WRITE='1',
                           D4R_RR_TIME_REPEAT='10')
                if backend == 'vkhip':
                    env['D4R_ENGINE_VKHIP'] = '1'
                log = W / 'logs' / f'profile-{preset}-{backend}-{geom}.log'
                r = subprocess.run([str(RUNNER / 'd4r-rr'), str(W / model), str(inputs), str(W / 'logs/profile-out')],
                                   env=env, capture_output=True, text=True)
                text = r.stdout + r.stderr
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(text)
                if r.returncode:
                    report.append({'preset': preset, 'backend': backend, 'geometry': geom, 'failed': r.returncode,
                                   'tail': text[-300:]})
                    continue
                whole = [float(m.group(1)) for m in re.finditer(r'^frame \d+: ([\d.]+) ms', text, re.M)]
                stages = []
                for line in text.splitlines():
                    m = re.match(r'stages \(us\):(.*)', line)
                    if m:
                        stages.append([float(v) for v in m.group(1).split()])
                stages = stages[3:]           # warmup entries
                per = {name: statistics.median([s[i] for s in stages if i < len(s)])
                       for i, name in enumerate(names)}
                network = sum(per.values())
                report.append({'preset': preset, 'backend': backend, 'geometry': geom,
                               'whole_ms_median_of_entries': round(statistics.median(whole[3:]), 4),
                               'network_sum_us_from_stamps': round(network, 1),
                               'stages_us': {k: round(v, 1) for k, v in per.items()},
                               'vkhip_stages': (re.search(r'vkhip: (\d+) Ray Reconstruction stages', text) or [None, '0'])[1]
                                               if backend == 'vkhip' else None,
                               'entries': len(whole)})
                print(json.dumps(report[-1]), flush=True)
    out = W / 'logs/profile-rr-stages.json'
    out.write_text(json.dumps(report, indent=1))
    print(f'wrote {out}')


main()
