"""Summarize native Windows d4r CPU stages and optional HIP event diagnostics.

CPU completion time is not GPU execution time. HIP occupancy API predictions
are not hardware counters. No bandwidth / WMMA utilization is inferred here.
"""
import argparse
import collections
import csv
import json
import math
import pathlib
import re
import statistics


def stats(values):
    values = sorted(values)
    def quantile(q):
        index = (len(values) - 1) * q
        lower = int(index)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)
    return dict(samples=len(values), total=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), p95=quantile(.95), maximum=values[-1])


def metadata(directory):
    result = {}
    if directory is None:
        return result
    for path in sorted(directory.glob('*.txt')):
        text = path.read_text(encoding='utf-8-sig', errors='replace')
        for block in re.split(r'(?m)^  - \.args:', text)[1:]:
            fields = dict(re.findall(r'(?m)^    \.(\w+):\s*([^\r\n]+)', block))
            if 'name' in fields:
                result[fields['name'].strip()] = {key: int(fields[key]) for key in (
                    'wavefront_size', 'vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count',
                    'group_segment_fixed_size', 'private_segment_fixed_size') if key in fields}
    return result


def presentmon(directory):
    # Keep swapchains separate: combining startup/menu/overlay chains invents
    # frame intervals. Missing/NA metrics remain absent rather than becoming 0.
    captures = []
    for path in sorted(directory.glob('presentmon*.csv')):
        groups = collections.defaultdict(list)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('ProcessID') and row.get('SwapChainAddress'):
                    groups[(row['ProcessID'], row['SwapChainAddress'])].append(row)
        for (process, swapchain), rows in groups.items():
            metrics = {}
            for name in ('MsBetweenPresents', 'MsBetweenAppStart', 'MsCPUBusy', 'MsCPUWait',
                         'MsGPULatency', 'MsGPUTime', 'MsGPUBusy', 'MsGPUWait',
                         'MsInPresentAPI', 'DisplayedTime', 'DisplayLatency'):
                values = []
                for row in rows:
                    raw = row.get(name, row.get(name.removeprefix('Ms'), ''))
                    try:
                        value = float(raw)
                    except (ValueError, TypeError):
                        continue
                    if math.isfinite(value) and value >= 0:
                        values.append(value)
                if values:
                    metrics[name] = stats(values)
            item = dict(file=path.name, processId=int(process), swapchain=swapchain,
                        frames=len(rows), metrics=metrics,
                        presentModes=dict(collections.Counter(row.get('PresentMode', '') for row in rows)),
                        syncIntervals=dict(collections.Counter(row.get('SyncInterval', '') for row in rows)))
            interval = metrics.get('MsBetweenPresents', metrics.get('MsBetweenAppStart'))
            if interval and interval['mean'] > 0:
                item['presentFps'] = 1000. / interval['mean']
            captures.append(item)
    captures.sort(key=lambda row: row['frames'], reverse=True)
    return captures


def gpu_telemetry(directory):
    captures = []
    for path in sorted(directory.glob('gpu-telemetry*.csv')):
        unique = {}
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        # Driver telemetry may repeat a cached measurement. Count polls but
        # use each sensor timestamp once; missing/unsupported values stay absent.
        for row in rows:
            try:
                timestamp = int(row.get('sensor_timestamp_ms', ''))
            except (ValueError, TypeError):
                continue
            if timestamp > 0:
                unique.setdefault(timestamp, row)
        metrics = {}
        for name in ('gpu_usage_pct', 'gpu_clock_mhz', 'vram_clock_mhz', 'gpu_power_w',
                     'board_power_w', 'temperature_c', 'hotspot_c', 'poll_ms'):
            values = []
            for row in unique.values():
                try:
                    value = float(row.get(name, ''))
                except (ValueError, TypeError):
                    continue
                if math.isfinite(value) and value >= 0 and (name != 'gpu_usage_pct' or value <= 100):
                    values.append(value)
            if values:
                metrics[name] = stats(values)
        captures.append(dict(file=path.name, polls=len(rows), uniqueSamples=len(unique), metrics=metrics))
    return captures


def report(directory, metadata_directory=None):
    kernel_metadata = metadata(metadata_directory)
    stages = collections.defaultdict(list)
    commands = collections.defaultdict(list)
    kernels = collections.defaultdict(list)
    hook_threads = collections.defaultdict(list)
    apis = collections.defaultdict(list)
    sample_skips = []
    invalid_samples = []
    boundaries = collections.defaultdict(list)
    replay = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in sorted(directory.glob('*.log')):
        data = path.read_bytes()
        text = data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig', errors='replace')
        for name, variant, pair, value in re.findall(
                r'D4R_REPLAY_PROFILE kernel=(\S+) variant=(control|candidate) pair=([0-9]+) gpu_ms=([0-9.]+)', text):
            replay[name][int(pair)][variant] = float(value)
        for name, value in re.findall(r'D4R_STAGE name=(\S+) cpu_ms=([0-9.]+)', text):
            stages[name].append(float(value))
        for kind, fields in re.findall(r'(?m)^D4R_COMMAND_(RECORD|SUBMIT) ([^\r\n]+)', text):
            for name, value in re.findall(r'(\w+_ms)=([0-9.]+)', fields):
                number = float(value)
                if name != 'interval_ms' or number > 0:
                    commands[kind.lower() + '.' + name].append(number)
        for line in text.splitlines():
            if line.startswith('D4R_GPU_BOUNDARY '):
                fields = dict(re.findall(r'(\w+)=(\S+)', line))
                if all(key in fields for key in ('feature', 'ticket', 'input_copy_ms', 'external_span_ms',
                                                 'output_copy_ms', 'total_ms', 'input_start_ticks',
                                                 'output_end_ticks', 'frequency')):
                    boundaries[fields['feature']].append(fields)
            if line.startswith('D4R_CUDA_API_PROFILE '):
                fields = dict(re.findall(r'(\w+)=(\S+)', line))
                if all(key in fields for key in ('api', 'calls', 'total_ms', 'max_ms', 'period_ms', 'thread')):
                    apis[(fields['thread'], fields['api'])].append(fields)
            if line.startswith('D4R_COMMAND_HOOK_SAMPLE '):
                fields = dict(re.findall(r'(\w+)=(\S+)', line))
                if all(name in fields for name in ('thread', 'period_ms', 'calls', 'samples', 'marked',
                                                   'access_sum_ms', 'capture_sum_ms', 'driver_sum_ms')):
                    hook_threads[fields['thread']].append(fields)
            if not line.startswith('D4R_KERNEL_PROFILE '):
                if line.startswith('D4R_KERNEL_PROFILE_SKIPPED '):
                    sample_skips.append(dict(re.findall(r'(\w+)=(\S+)', line)))
                if line.startswith('D4R_KERNEL_PROFILE_INVALID '):
                    invalid_samples.append(dict(re.findall(r'(\w+)=("[^"]+"|\S+)', line)))
                continue
            fields = dict(re.findall(r'(\w+)=("[^"]+"|\S+)', line))
            required = ('kernel', 'backend', 'phase', 'gpu_ms', 'enqueue_ms', 'completion_ms',
                        'grid', 'block', 'static_lds_bytes', 'dynamic_lds_bytes', 'regs', 'private_bytes',
                        'max_threads', 'predicted_blocks_per_multiprocessor', 'serializing')
            if not all(key in fields for key in required):
                continue  # A bounded process stop can leave a partial final line.
            kernels[(fields['kernel'].strip('"'), fields['backend'], fields['phase'],
                     fields['serializing'])].append(fields)
    result = dict(cpuStages={name: stats(values) for name, values in stages.items()},
                  commandStages={name: stats(values) for name, values in commands.items()},
                  commandHookSamples=[],
                  cudaApi=[],
                  kernelSampleSkips=sample_skips,
                  invalidKernelSamples=invalid_samples,
                  gpuBoundaryStages={},
                  presentMon=presentmon(directory),
                  gpuTelemetry=gpu_telemetry(directory),
                  replayPairs=[],
                  kernels=[], notes=[
                      'CPU stage times include waits and host work; they are not isolated GPU timings.',
                      'CUDA API profiles measure host call duration without GPU events or added synchronization; they still include existing waits.',
                      'D3D12 boundary timestamps span copies and HIP/scheduling across the external fence; they do not isolate kernel execution or measure Present.',
                      'Boundary timestamps are read after existing completion fences; only 32 timing bytes are read, with no extra completion wait. GPU clock idle behavior can affect intervals.',
                      'PresentMon ETW GPU busy/wait are per-process frame estimates, not hardware sensor utilization. HWS and cross-API context attribution can affect them.',
                      'ADLX telemetry is driver-reported device-wide utilization/clocks/power, sampled in a separate read-only process. It does not isolate DLSS or WMMA occupancy.',
                      'Command record intervals are between NGX recording calls on one thread, not Present/FPS.',
                      'Hook estimates use random one-in-64 samples; driver timing covers generated forwarding methods only.',
                      'Hook access includes lock waits; summing threads does not measure serial frame latency or CPU execution time.',
                      'Serializing HIP-event profiles wait after every launch. Deferred profiles query ready events and never add completion waits; timing events still add overhead.',
                      'Deferred completionMs is host collection lag, not kernel completion latency. Legacy-stream sampling requires an opt-in because its timing markers retain legacy cross-stream ordering.',
                      'Occupancy is an API prediction, not a measured hardware counter.',
                      'Bandwidth and WMMA utilization require additional supported hardware tooling.'])
    boundary_stages = collections.defaultdict(list)
    for rows in boundaries.values():
        rows.sort(key=lambda row: int(row['ticket']))
        for row in rows:
            for key in ('input_copy_ms', 'external_span_ms', 'output_copy_ms', 'total_ms'):
                boundary_stages[key].append(float(row[key]))
        for previous, current in zip(rows, rows[1:]):
            if int(current['ticket']) != int(previous['ticket']) + 2 or current['frequency'] != previous['frequency']:
                continue
            frequency = int(current['frequency'])
            before = int(current['input_start_ticks']) - int(previous['output_end_ticks'])
            interval = int(current['input_start_ticks']) - int(previous['input_start_ticks'])
            if frequency > 0 and before >= 0 and interval > 0:
                boundary_stages['between_boundaries_ms'].append(before * 1000. / frequency)
                boundary_stages['input_interval_ms'].append(interval * 1000. / frequency)
    result['gpuBoundaryStages'] = {name: stats(values) for name, values in boundary_stages.items()}
    for name, pairs in replay.items():
        complete = [row for row in pairs.values() if set(row) == {'control', 'candidate'}]
        if complete:
            result['replayPairs'].append(dict(kernel=name,
                controlGpuMs=stats([row['control'] for row in complete]),
                candidateGpuMs=stats([row['candidate'] for row in complete]),
                candidateToControl=stats([row['candidate'] / row['control'] for row in complete if row['control'] > 0])))
    for (thread, api), rows in apis.items():
        calls = sum(int(row['calls']) for row in rows)
        total = sum(float(row['total_ms']) for row in rows)
        result['cudaApi'].append(dict(thread=thread, api=api, calls=calls, totalMs=total,
            meanMs=total / calls, maxMs=max(float(row['max_ms']) for row in rows),
            periodMs=sum(float(row['period_ms']) for row in rows)))
    result['cudaApi'].sort(key=lambda row: row['totalMs'], reverse=True)
    for thread, rows in hook_threads.items():
        sums = {name: sum(float(row[name]) for row in rows) for name in
                ('period_ms', 'calls', 'samples', 'marked', 'access_sum_ms', 'capture_sum_ms', 'driver_sum_ms')}
        if not sums['samples'] or not sums['period_ms']:
            continue
        scale = sums['calls'] / sums['samples']
        estimate = scale * (sums['access_sum_ms'] + sums['capture_sum_ms'])
        result['commandHookSamples'].append(dict(thread=int(thread), periods=len(rows), **sums,
            accessMeanUs=1000 * sums['access_sum_ms'] / sums['samples'],
            captureMeanUsPerSample=1000 * sums['capture_sum_ms'] / sums['samples'],
            estimatedAccessCaptureMs=estimate, estimatedAccessCaptureMsPerSecond=1000 * estimate / sums['period_ms']))
    for (name, backend, phase, serializing), rows in kernels.items():
        item = dict(kernel=name, backend=backend, phase=phase,
                    serializing=serializing == '1',
                    legacyStream=any(row.get('legacy_stream') == '1' for row in rows),
                    completionMeaning='host_completion_wait' if serializing == '1' else 'host_collection_lag',
                    gpuMs=stats([float(row['gpu_ms']) for row in rows]),
                    enqueueMs=stats([float(row['enqueue_ms']) for row in rows]),
                    completionMs=stats([float(row['completion_ms']) for row in rows]),
                    configurations=[dict(config) for config in sorted({tuple((key, row[key]) for key in
                        ('grid', 'block', 'static_lds_bytes', 'dynamic_lds_bytes', 'regs', 'private_bytes',
                         'max_threads', 'predicted_blocks_per_multiprocessor')) for row in rows})])
        result['kernels'].append(item)
        if backend == 'native':
            item['elfMetadata'] = kernel_metadata.get(name + ('_prep' if phase == 'prep' else ''), {})
    result['kernels'].sort(key=lambda row: row['gpuMs']['total'], reverse=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--metadata-directory', type=pathlib.Path, help='llvm-readobj --notes output for native code objects')
    args = parser.parse_args()
    result = report(args.directory, args.metadata_directory)
    if not any(result[key] for key in ('cpuStages', 'kernels', 'commandStages', 'commandHookSamples', 'replayPairs', 'cudaApi', 'gpuBoundaryStages', 'presentMon', 'gpuTelemetry')):
        parser.error('no complete stage or HIP-event records found')
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, row in result['cpuStages'].items():
        print(f"CPU {name}: n={row['samples']} mean={row['mean']:.3f} ms median={row['median']:.3f} ms p95={row['p95']:.3f} ms")
    for name, row in result['commandStages'].items():
        print(f"COMMAND {name}: n={row['samples']} mean={row['mean']:.3f} ms median={row['median']:.3f} ms p95={row['p95']:.3f} ms")
    for name, row in result['gpuBoundaryStages'].items():
        print(f"GPU BOUNDARY {name}: n={row['samples']} mean={row['mean']:.3f} ms median={row['median']:.3f} ms p95={row['p95']:.3f} ms")
    for capture in result['presentMon']:
        print(f"PRESENT {capture['file']} process={capture['processId']} swapchain={capture['swapchain']} frames={capture['frames']} fps={capture.get('presentFps', float('nan')):.3f}")
        for name, row in capture['metrics'].items():
            print(f"  {name}: n={row['samples']} mean={row['mean']:.3f} ms median={row['median']:.3f} ms p95={row['p95']:.3f} ms")
    for capture in result['gpuTelemetry']:
        print(f"ADLX {capture['file']}: polls={capture['polls']} unique_samples={capture['uniqueSamples']}")
        for name, row in capture['metrics'].items():
            print(f"  {name}: n={row['samples']} mean={row['mean']:.3f} median={row['median']:.3f} p95={row['p95']:.3f}")
    for row in result['cudaApi'][:20]:
        print(f"CUDA API {row['api']} {row['thread']}: n={row['calls']} mean={row['meanMs']:.6f} ms total={row['totalMs']:.3f} ms max={row['maxMs']:.3f} ms")
    for row in result['commandHookSamples']:
        print(f"HOOK thread={row['thread']} samples={int(row['samples'])} access={row['accessMeanUs']:.3f} us "
              f"capture={row['captureMeanUsPerSample']:.3f} us estimated_cpu={row['estimatedAccessCaptureMsPerSecond']:.3f} ms/s")
    for row in result['replayPairs']:
        print(f"PAIRED {row['kernel']}: n={row['controlGpuMs']['samples']} control_median={row['controlGpuMs']['median']:.6f} ms "
              f"candidate_median={row['candidateGpuMs']['median']:.6f} ms ratio_median={row['candidateToControl']['median']:.6f}")
    for row in result['kernels']:
        times = row['gpuMs']
        print(f"GPU {row['kernel']} {row['backend']} {row['phase']} serializing={int(row['serializing'])}: n={times['samples']} mean={times['mean']:.6f} ms total={times['total']:.3f} ms")


if __name__ == '__main__':
    main()
