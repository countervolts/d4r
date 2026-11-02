#!/usr/bin/env python3
"""Record the machine, toolchain, backend and model provenance of this comparison.

usage: record_environment.py     (reads only; touches no GPU state)
Writes logs/environment.json.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'build/local-preset-compare-20261029'
RUNNER = Path('/tmp/d4r-preset-compare-20261029/engine')
DEV = Path('/sys/class/drm/card1/device')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def sys_read(nodes):
    out = {}
    for key, node in nodes.items():
        try:
            out[key] = (DEV / node).read_text().strip()
        except Exception:
            out[key] = None
    return out


env = {
    'host': {'kernel': run(['uname', '-rs']),
             'cpu': [l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')][0].split(':')[1].strip()},
    'git': {'head': run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
            'status': run(['git', '-C', str(ROOT), 'status', '--short']),
            'patched_files': ['engine/replay_m.cpp', 'engine/replay_rr.cpp'],
            'patch_diff': run(['git', '-C', str(ROOT), 'diff', '--', 'engine/replay_m.cpp', 'engine/replay_rr.cpp'])},
    'toolchain': {'gxx': run(['g++', '--version']).splitlines()[0],
                  'hipconfig': run(['/opt/rocm/bin/hipconfig', '--version']),
                  'glslang': run(['glslangValidator', '--version']).splitlines()[0],
                  'python': sys.version.split()[0]},
    'graphics': {'glxinfo_device': [l for l in run(['glxinfo']).splitlines() if 'Device:' in l][0].strip(),
                 'mesa': [l for l in run(['glxinfo']).splitlines() if l.startswith('OpenGL version string')][0].strip(),
                 'vulkan_loader': run(['vulkaninfo', '--version']).splitlines()[-1],
                 'rocminfo': [l for l in run(['/opt/rocm/bin/rocminfo']).splitlines()
                              if 'Marketing Name' in l or 'gfx12' in l][:4]},
    'gpu': {'pci': run(['lspci', '-mm', '-s', '0000:03:00.0']),
            'sysfs_uevent': sys_read({'uevent': 'uevent'}),
            'hwmon': sys_read({'power_cap_uw': 'hwmon/hwmon3/power1_cap', 'power_cap_default_uw': 'hwmon/hwmon3/power1_cap_default',
                                'power_cap_max_uw': 'hwmon/hwmon3/power1_cap_max', 'sclk_now_hz': 'hwmon/hwmon3/freq1_input',
                                'mclk_hz': 'hwmon/hwmon3/freq2_input', 'junction_c': 'hwmon/hwmon3/temp2_input'}),
            'power_management': sys_read({'performance_level': 'power_dpm_force_performance_level',
                                          'dpm_state': 'power_dpm_state', 'sclk_states': 'pp_dpm_sclk',
                                          'power_profile': 'pp_power_profile_mode'})},
    'runner_binaries': {p.name: sha(p) for p in sorted(RUNNER.glob('d4r-*')) if p.is_file()},
    'engine_library_sha256': sha(RUNNER / 'libd4r_engine.so'),
    'models': {},
    'dlls': {},
    'nvidia_reference_sha256': {'sr': sha(ROOT / 'docs/benchmarks/nvidia-blackwell-sr-20261029.json'),
                                'rr': sha(ROOT / 'docs/benchmarks/nvidia-blackwell-rr-20261029.json')},
}

for model in sorted(W.glob('models/*')) + sorted(W.glob('models-cnn/*')):
    manifest = model / 'model.json' if (model / 'model.json').exists() else model / 'rr.txt'
    entry = {'manifest': manifest.name, 'manifest_sha256': sha(manifest),
             'manifest_head': manifest.read_text()[:160]}
    if (model / 'model.json').exists():
        entry['preset'] = json.loads(manifest.read_text())['preset']
    elif (model / 'rr.txt').exists():
        entry['preset_line'] = (model / 'rr.txt').read_text().splitlines()[1]
    if (model / 'hip').is_dir():
        entry['hip_code_objects'] = {p.name: sha(p) for p in sorted((model / 'hip').glob('*.hsaco'))}
    env['models'][str(model.relative_to(W))] = entry

for dll in sorted((ROOT / 'build/linux-all-presets-handoff-20261022/private/dll').glob('*.dll')):
    env['dlls'][dll.name] = {'sha256': sha(dll), 'bytes': dll.stat().st_size}

env['scripts_sha256'] = {p.name: sha(p) for p in sorted((ROOT / 'tools').glob('*local*.py'))}
env['scripts_sha256']['make_local_rr_inputs.cpp'] = sha(ROOT / 'tools/make_local_rr_inputs.cpp')

(W / 'logs/environment.json').write_text(json.dumps(env, indent=1))
print(f'wrote {W/"logs/environment.json"} ({len(json.dumps(env))} bytes)')
