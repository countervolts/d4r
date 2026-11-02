#!/usr/bin/env python3
"""Build the machine-readable comparison and the markdown tables from the recorded runs.

usage: build_summary.py      writes docs/benchmarks/local-gpu-comparison-20261029.json
Prints the tables used by docs/local-gpu-preset-comparison.md.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'build/local-preset-compare-20261029'
GEOM = {'1440p': '1280x720->2560x1440', '4k': '1920x1080->3840x2160'}
NV = {'sr': json.loads((ROOT / 'docs/benchmarks/nvidia-blackwell-sr-20261029.json').read_text()),
      'rr': json.loads((ROOT / 'docs/benchmarks/nvidia-blackwell-rr-20261029.json').read_text())}
NV_SR_PRESET = {'m': '13', 'l': '12'}
NV_RR_LIBRARY = {'a': '310.3.0', 'b': '310.3.0', 'c': '310.3.0', 'd': '310.9.16', 'e': '310.9.16', 'f': '310.9.16'}
ROUTE_SETTINGS = {
    'vulkan': 'engine Vulkan network and stages',
    'vkhip': 'native HIP code objects as Vulkan pipelines in the engine command list (D4R_ENGINE_VKHIP)',
    'hip': 'native HIP layers through the HIP runtime, engine stages split around them',
}


def nvidia(family, preset, geom):
    if family == 'sr':
        return NV['sr']['results'][GEOM[geom]]['median_of_trial_medians_ms'][NV_SR_PRESET[preset]]
    key = f'{NV_RR_LIBRARY[preset]}/{preset.upper()}/{GEOM[geom]}'
    return NV['rr']['cases'][key]['median_ms']


def local(family, preset, geom):
    suite = json.loads((W / f'logs/suite-{family}-{geom}.json').read_text())
    case_id = f'{family}-{preset}'
    out = {}
    for run in suite['runs']:
        if 'median_ms' not in run or run['case'] != case_id:
            continue
        out.setdefault(run['backend'], []).append(run)
    routes = {}
    for backend, runs in out.items():
        medians = [r['median_ms'] for r in runs]
        stages = {}
        for key in runs[0]['stages_us']:
            stages[key] = round(statistics.median([r['stages_us'][key] for r in runs]), 1)
        routes[backend] = {'trial_medians_ms': [round(m, 6) for m in medians],
                           'median_of_trial_medians_ms': round(statistics.median(medians), 6),
                           'measured_frames': runs[0]['frames'], 'stages_median_us': stages,
                           'gpu_clock_mhz_median': runs[0]['gpu']['sclk_mhz']['median'],
                           'gpu_busy_percent_median': runs[0]['gpu']['busy_percent']['median'],
                           'ppt_w_median': runs[0]['gpu']['ppt_w']['median'],
                           'setting': ROUTE_SETTINGS[backend],
                           'note': sorted({n for r in runs for n in r.get('note', [])}),
                           'command': runs[0]['command']}
    if not routes:
        return {'routes': {}}
    fastest = min(routes, key=lambda b: routes[b]['median_of_trial_medians_ms'])
    return {'routes': routes, 'fastest_route': fastest,
            'fastest_ms': routes[fastest]['median_of_trial_medians_ms']}


def main():
    summary = {
        'date': '2026-10-29', 'gpu': 'AMD Radeon RX 9070 XT (gfx1201, Navi 48, PCI 1002:7550)',
        'driver': 'Mesa 26.2.5-arch3.1 RADV, kernel 7.3.0-5-cachyos, DRM 3.64',
        'power_limit_w': int(Path('/sys/class/drm/card1/device/hwmon/hwmon3/power1_cap').read_text()) / 1e6,
        'power_cap_default_w': int(Path('/sys/class/drm/card1/device/hwmon/hwmon3/power1_cap_default').read_text()) / 1e6,
        'method': {
            'cadence': '40+ warmup and 100 measured evaluations per case, three interleaved trials, median of the three trial medians',
            'timing': 'Vulkan timestamp queries bracketing the evaluation; initialization, input uploads and output readback outside them (same exclusions as the NVIDIA CUDA-event probe)',
            'inputs': 'the official probe\'s inputs recreated byte for byte: SR patterned RGBA16F with zero motion and depth 0.5; RR one static noisy fsr_rr_scene frame with its guides, hit distance 4, view/projection, zero motion, alternating jitter, Reset only on the first evaluation',
            'quality': 'Performance geometry (render half the output per axis); d4r has no separate quality knob in the replay runners',
            'not_comparable': 'whole-evaluation stamps here versus NVIDIA CUDA-event bracketing: both exclude init/upload/readback, but d4r\'s engine runs one submit with barriers while NGX evaluates on its own; cross-backend arithmetic is native FP8, not NVIDIA\'s PTX',
        },
        'model_provenance': {
            'sr_m_l': 'engine models rebuilt from the retained captures of nvngx_dlss.dll 310.7.0; the NVIDIA reference measured 310.9.16',
            'rr_a_b_c': 'engine/compile_rr_cnn.py from nvngx_dlssd_310.3.0.dll, matching the NVIDIA A/B/C library',
            'rr_d_e_f': 'handoff models from nvngx_dlssd_310.9.1.dll; the NVIDIA reference measured 310.9.16',
        },
        'results': {}, 'cost_ratios': {}, 'stage_profiles': {}, 'verification': {}, 'environment': {},
        'sr_l_headroom_projection': {
            'note': 'Derived, not measured: d4r M stage times scaled by NVIDIA per-stage L-over-M work ratios, '
                    'i.e. L costs if its outer stages matched M\'s relative efficiency.',
            'nvidia_l_over_m_stage_ratios_4k': {'exposure': 0.993, 'input': 2.352, 'network': 1.013,
                                               'expansion': 2.949, 'reconstruction': 1.096, 'downsample': 0.981},
            'predicted_l_ms': {'4k': 3.525, '1440p': 1.654},
            'predicted_l_basis': {'4k': 'per-stage scaling of the measured M stages', '1440p': 'M stage sum 1.265 ms scaled by NVIDIA whole-times L/M 1.307 (no 1440p NVIDIA stage profile exists)'},
            'measured_l_ms': {'4k': 4.527, '1440p': 2.002},
            'saving_ms': {'4k': 1.002, '1440p': 0.348},
            'saving_by_stage_4k_ms': {'input': 0.493, 'expansion': 0.404, 'reconstruction': 0.102},
            'predicted_l_over_m_4k': 1.230, 'predicted_l_over_m_1440p': 1.307,
            'measured_l_over_m_4k': 1.580, 'measured_l_over_m_1440p': 1.588,
            'nvidia_l_over_m_4k': 1.238, 'nvidia_l_over_m_1440p': 1.307,
            'predicted_l_vs_nvidia_4k': 1.12, 'predicted_l_vs_nvidia_1440p': 1.17,
            'why_not_lower': 'The 1.54x network ratio is shared with M and is hardware-side; matching M\'s '
                            'efficiency converges L onto M\'s 1.12x, not onto NVIDIA\'s absolute time.',
        },
        'hardware_yardstick': {
            'note': 'Published dense peaks, not measured here; treat each ratio as an upper bound on the hardware gap.',
            'rx_9070_xt': {'fp8_dense_tflops': 389, 'fp16_dense_tflops': 195, 'fp32_tflops': 48.7,
                           'bandwidth_gbs': 640, 'compute_units': 64, 'power_cap_w': 250, 'power_default_w': 330,
                           'source': 'TechPowerUp GPU database and Flopper spec sheet for Navi 48; CU count from rocminfo'},
            'rtx_pro_4500_blackwell': {'fp8_dense_tflops': 811, 'fp16_dense_tflops': 406, 'fp32_tflops': 51,
                                       'bandwidth_gbs': 800, 'power_limit_w': 200,
                                       'source': 'NVIDIA product pages (server edition bandwidth)'},
            'ratios': {'fp8': 2.09, 'fp16': 2.08, 'fp32': 1.05, 'bandwidth': 1.25},
            'verdicts': [
                {'stage_class': 'RR network (Swin D/E, rr2 F)', 'measured': '2.05-2.14x', 'yardstick': 'FP8 2.09x',
                 'verdict': 'hardware'},
                {'stage_class': 'SR network (ten Swin layers)', 'measured': '1.54x', 'yardstick': 'FP8 2.09x',
                 'verdict': 'hardware-limited, inside the gap'},
                {'stage_class': 'RR CNN (A/B/C)', 'measured': '1.49x', 'yardstick': 'FP8 2.09x',
                 'verdict': 'hardware-limited'},
                {'stage_class': 'SR input (enc0)', 'measured': '0.96x', 'yardstick': 'bandwidth 1.25-1.40x',
                 'verdict': 'software advantage'},
                {'stage_class': 'SR reconstruction and downsample', 'measured': '0.71-0.82x', 'yardstick': 'FP32 1.05x',
                 'verdict': 'software advantage'},
                {'stage_class': 'RR outer group', 'measured': '1.33-1.63x', 'yardstick': 'bandwidth 1.25x',
                 'verdict': 'mostly software'},
                {'stage_class': 'SR L non-native outer stages (enc0_l, dec0_l)', 'measured': '1.94x, 2.52x',
                 'yardstick': 'bandwidth 1.25-1.40x', 'verdict': 'software, largest addressable item'},
                {'stage_class': 'RR CNN as Vulkan shaders vs native code objects on the same card',
                 'measured': '17.23 ms vs 2.77 ms', 'yardstick': 'same GPU', 'verdict': 'software'},
            ],
        },
        'limitations': [
            'Model version mismatch: local SR M/L are 310.7.0 weights against NVIDIA 310.9.16; local RR D/E/F are 310.9.1 against 310.9.16. RR A/B/C match (310.3.0).',
            'The replay runners have no HDR/auto-exposure creation flags: the SR case runs the engine\'s exposure stage with the same patterned input, not NGX\'s auto-exposure path (flags 0x4b).',
            'These are isolated d4r engine evaluations on an RX 9070 XT at a 250 W cap, not game frame times, and not RTX 5070 Ti numbers.',
            'Clocks are not pinned: the sclk DPM state list is dynamic (snapshots of 500/1826/2520 and 500/2765/2520 MHz, rocminfo nominal 2520), sampled per-run medians spanned 1228-3178 MHz, GPU busy averaged 91% and the heaviest runs sat at the 250 W cap (median-of-medians 242 W, maximum sample 352 W).',
            'Cross-backend output differences are arithmetic, not failures: vkhip vs the engine\'s Vulkan stages is 55-57 dB (SR) and 47.6-49.7 dB (RR at 1440p) / 48.1-50.5 dB (RR at 4K) on this case.',
        ],
    }
    for geom in ('1440p', '4k'):
        for family in ('sr', 'rr'):
            presets = ('m', 'l') if family == 'sr' else ('a', 'b', 'c', 'd', 'e', 'f')
            for preset in presets:
                key = f'{family}-{preset.upper()}-{geom}'
                loc = local(family, preset, geom)
                nv = nvidia(family, preset, geom)
                if loc.get('fastest_ms'):
                    summary['results'][key] = {'preset': preset.upper(), 'geometry': GEOM[geom], 'family': family,
                                               'nvidia_ms': nv, 'local_fastest_ms': loc['fastest_ms'],
                                               'local_over_nvidia': round(loc['fastest_ms'] / nv, 4),
                                               'fastest_route': loc['fastest_route'], 'routes': loc['routes']}
                else:
                    summary['results'][key] = {'preset': preset.upper(), 'geometry': GEOM[geom], 'family': family,
                                               'nvidia_ms': nv, 'local_fastest_ms': None, 'routes': loc['routes']}
    fast = lambda key: summary['results'][key]['local_fastest_ms']
    summary['cost_ratios'] = {
        'sr_L_over_M': {'local_1440p': round(fast('sr-L-1440p') / fast('sr-M-1440p'), 4),
                        'local_4k': round(fast('sr-L-4k') / fast('sr-M-4k'), 4),
                        'nvidia_1440p': round(NV['sr']['results'][GEOM['1440p']]['L_over_M_percent'] / 100 + 1, 4),
                        'nvidia_4k': round(NV['sr']['results'][GEOM['4k']]['L_over_M_percent'] / 100 + 1, 4)},
        'rr_over_a': {geom: {p: round(fast(f'rr-{p}-{geom}') / fast(f'rr-A-{geom}'), 4) for p in 'BCDEF'} for geom in ('1440p', '4k')},
        'rr_nvidia_over_a': {geom: {p: round(nvidia('rr', p.lower(), geom) / nvidia('rr', 'a', geom), 4) for p in 'BCDEF'}
                             for geom in ('1440p', '4k')},
    }
    # stage profiles: SR from the suites, RR from the dedicated stamp profile
    summary['stage_profiles'] = {'stages_us_median': {}, 'rr_network_stages_us': {}}
    for geom in ('1440p', '4k'):
        for family, presets in (('sr', ('M', 'L')), ('rr', ('A', 'D', 'E', 'F'))):
            for preset in presets:
                entry = summary['results'][f'{family}-{preset}-{geom}']
                for backend, route in entry['routes'].items():
                    summary['stage_profiles']['stages_us_median'][f'{family}-{preset}-{geom}-{backend}'] = route['stages_median_us']
    summary['stage_profiles']['rr_network_stages_us'] = json.loads((W / 'logs/profile-rr-stages.json').read_text())
    summary['verification'] = json.loads((W / 'logs/route-verification.json').read_text())
    rr4k = W / 'logs/route-verification-4k-rr.json'
    if rr4k.exists():
        summary['verification']['rr_4k'] = json.loads(rr4k.read_text())
    env = json.loads((W / 'logs/environment.json').read_text())
    summary['environment'] = {k: env[k] for k in ('host', 'git', 'toolchain', 'graphics', 'gpu', 'runner_binaries',
                                                  'engine_library_sha256', 'dlls', 'nvidia_reference_sha256',
                                                  'scripts_sha256')}
    summary['environment']['models'] = env['models']
    out = ROOT / 'docs/benchmarks/local-gpu-comparison-20261029.json'
    out.write_text(json.dumps(summary, indent=1))
    print(f'wrote {out}')

    print('\n| preset | resolution | fastest local route | local ms | NVIDIA ms | local/NVIDIA |')
    print('|---|---|---|---:|---:|---:|')
    for key in sorted(summary['results'], key=lambda k: (summary['results'][k]['family'], k)):
        r = summary['results'][key]
        if r['local_fastest_ms'] is None:
            continue
        print(f"| {r['preset']} | {r['geometry']} | {r['fastest_route']} | {r['local_fastest_ms']:.3f} | "
              f"{r['nvidia_ms']:.3f} | {r['local_over_nvidia']:.2f}× |")
    print('\nroute table:')
    for key in sorted(summary['results']):
        r = summary['results'][key]
        for backend, route in r['routes'].items():
            print(f"  {key:14s} {backend:6s} {route['median_of_trial_medians_ms']:.3f} ms trials {route['trial_medians_ms']}")
    print('\ncost ratios:', json.dumps(summary['cost_ratios'], indent=1))


main()
