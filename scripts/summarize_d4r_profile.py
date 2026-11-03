#!/usr/bin/env python3
"""Summarize warm D4R_PROFILE samples without merging feature instances or sizes.

GPU pass fields and delayed vk_* fields are summarized independently. Their
per-line difference is deliberately not calculated; they describe different frames.
--stages summarizes the engines' D4R_ENGINE_STAGE_PROFILE rows instead (Vulkan and
native D3D12): GPU milliseconds per stage, in the order the row lists them.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

GROUP_FIELDS = ('render', 'output', 'preset', 'quality', 'flags', 'input_wait',
                'feature', 'native_field', 'native_substitute', 'wire_requested', 'gpu_completion', 'vk_output_mode', 'vk_input_mode')

FIELDS = ('vk_total', 'dlss_gpu', 'vk_input', 'vk_dlss_wait', 'vk_output',
          'native_motion_gpu', 'wire_repair_gpu', 'marker_wait', 'pipeline',
          'throughput_interval')


def summarize(text, min_frame=300, since_clock=None, until_clock=None):
    groups = defaultdict(list)
    instance = 0
    for line in text.splitlines():
        if 'NVSDK_NGX_D3D12_CreateFeature(feature=' in line:
            instance += 1
        if 'D4R_PROFILE frame=' not in line:
            continue
        if since_clock is not None or until_clock is not None:
            clock = re.search(r'^\[\s*([0-9.]+)\]', line)
            if clock is None:
                continue
            seconds = float(clock[1])
            if since_clock is not None and seconds < since_clock:
                continue
            if until_clock is not None and seconds > until_clock:
                continue
        values = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if int(values['frame']) < min_frame or values.get('reset') != '0':
            continue
        key = (instance, *(values.get(k, 'unknown') for k in
               GROUP_FIELDS))
        groups[key].append(values)
    result = []
    for key, samples in groups.items():
        row = dict(zip(('feature_instance', *GROUP_FIELDS), key))
        row.update(samples=len(samples), first_frame=int(samples[0]['frame']), last_frame=int(samples[-1]['frame']))
        row['timings_ms'] = {}
        for field in FIELDS:
            values = [float(s.get(field, '-1')) for s in samples]
            valid = [v for v in values if math.isfinite(v) and v >= 0]
            row['timings_ms'][field] = (dict(n=len(valid), median=statistics.median(valid),
                                           min=min(valid), max=max(valid)) if valid else None)
        result.append(row)
    return result


STAGE_GROUP_FIELDS = ('preset', 'render', 'output')
STAGE_META_FIELDS = ('preset', 'gpu_frame', 'render', 'output')


def _clock_ok(line, since_clock, until_clock):
    if since_clock is None and until_clock is None:
        return True
    clock = re.search(r'^\[\s*([0-9.]+)\]', line)
    if clock is None:
        return False
    seconds = float(clock[1])
    return not ((since_clock is not None and seconds < since_clock) or
                (until_clock is not None and seconds > until_clock))


def summarize_stages(text, min_frame=300, since_clock=None, until_clock=None):
    groups = defaultdict(list)
    instance = 0
    for line in text.splitlines():
        if 'NVSDK_NGX_D3D12_CreateFeature(feature=' in line:
            instance += 1
        if 'D4R_ENGINE_STAGE_PROFILE preset=' not in line or not _clock_ok(line, since_clock, until_clock):
            continue
        fields = re.findall(r'(\w+)=([^\s]+)', line[line.index('D4R_ENGINE_STAGE_PROFILE'):])
        values = dict(fields)
        if int(values['gpu_frame']) < min_frame:
            continue
        stages = [k for k, _ in fields if k not in STAGE_META_FIELDS]
        groups[(instance, *(values[k] for k in STAGE_GROUP_FIELDS))].append((stages, values))
    result = []
    for key, samples in groups.items():
        row = dict(zip(('feature_instance', *STAGE_GROUP_FIELDS), key))
        row.update(samples=len(samples), first_frame=int(samples[0][1]['gpu_frame']),
                   last_frame=int(samples[-1][1]['gpu_frame']))
        order = []
        for stages, _ in samples:
            order += [k for k in stages if k not in order]
        row['timings_ms'] = {}
        for field in order:
            valid = [v for v in (float(s.get(field, '-1')) for _, s in samples) if math.isfinite(v) and v >= 0]
            row['timings_ms'][field] = (dict(n=len(valid), median=statistics.median(valid),
                                             min=min(valid), max=max(valid)) if valid else None)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--min-frame', type=int, default=300)
    parser.add_argument('--since-clock', type=float, help='minimum bracketed log-clock seconds')
    parser.add_argument('--until-clock', type=float, help='maximum bracketed log-clock seconds')
    parser.add_argument('--stages', action='store_true', help='summarize D4R_ENGINE_STAGE_PROFILE rows')
    args = parser.parse_args()
    run = summarize_stages if args.stages else summarize
    print(json.dumps(run(args.log.read_text(errors='replace'), args.min_frame,
                         args.since_clock, args.until_clock), indent=2))


if __name__ == '__main__':
    main()
