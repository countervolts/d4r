"""Summarize native Windows d4r CPU stages and optional serializing HIP events.

CPU completion time is not GPU execution time. HIP occupancy API predictions
are not hardware counters. No bandwidth / WMMA utilization is inferred here.
"""
import argparse
import collections
import json
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


def report(directory, metadata_directory=None):
    kernel_metadata = metadata(metadata_directory)
    stages = collections.defaultdict(list)
    commands = collections.defaultdict(list)
    kernels = collections.defaultdict(list)
    hook_threads = collections.defaultdict(list)
    apis = collections.defaultdict(list)
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
                continue
            fields = dict(re.findall(r'(\w+)=("[^"]+"|\S+)', line))
            kernels[(fields['kernel'].strip('"'), fields['backend'], fields['phase'])].append(fields)
    result = dict(cpuStages={name: stats(values) for name, values in stages.items()},
                  commandStages={name: stats(values) for name, values in commands.items()},
                  commandHookSamples=[],
                  cudaApi=[],
                  gpuBoundaryStages={},
                  replayPairs=[],
                  kernels=[], notes=[
                      'CPU stage times include waits and host work; they are not isolated GPU timings.',
                      'CUDA API profiles measure host call duration without GPU events or added synchronization; they still include existing waits.',
                      'D3D12 boundary timestamps span copies and HIP/scheduling across the external fence; they do not isolate kernel execution or measure Present.',
                      'Boundary timestamps are read after existing completion fences; only 32 timing bytes are read, with no extra completion wait. GPU clock idle behavior can affect intervals.',
                      'Command record intervals are between NGX recording calls on one thread, not Present/FPS.',
                      'Hook estimates use random one-in-64 samples; driver timing covers generated forwarding methods only.',
                      'Hook access includes lock waits; summing threads does not measure serial frame latency or CPU execution time.',
                      'HIP-event profiling synchronizes every sampled launch and changes scheduling.',
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
    for (name, backend, phase), rows in kernels.items():
        item = dict(kernel=name, backend=backend, phase=phase,
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
    if not any(result[key] for key in ('cpuStages', 'kernels', 'commandStages', 'commandHookSamples', 'replayPairs', 'cudaApi', 'gpuBoundaryStages')):
        parser.error('no complete stage or HIP-event records found')
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, row in result['cpuStages'].items():
        print(f"CPU {name}: n={row['samples']} mean={row['mean']:.3f} ms median={row['median']:.3f} ms p95={row['p95']:.3f} ms")
    for name, row in result['commandStages'].items():
        print(f"COMMAND {name}: n={row['samples']} mean={row['mean']:.3f} ms median={row['median']:.3f} ms p95={row['p95']:.3f} ms")
    for name, row in result['gpuBoundaryStages'].items():
        print(f"GPU BOUNDARY {name}: n={row['samples']} mean={row['mean']:.3f} ms median={row['median']:.3f} ms p95={row['p95']:.3f} ms")
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
        print(f"GPU {row['kernel']} {row['backend']} {row['phase']}: n={times['samples']} mean={times['mean']:.6f} ms total={times['total']:.3f} ms")


if __name__ == '__main__':
    main()
