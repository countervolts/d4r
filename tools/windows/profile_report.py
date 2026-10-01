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
    for path in sorted(directory.glob('*.log')):
        text = path.read_text(encoding='utf-8-sig', errors='replace')
        for name, value in re.findall(r'D4R_STAGE name=(\S+) cpu_ms=([0-9.]+)', text):
            stages[name].append(float(value))
        for kind, fields in re.findall(r'(?m)^D4R_COMMAND_(RECORD|SUBMIT) ([^\r\n]+)', text):
            for name, value in re.findall(r'(\w+_ms)=([0-9.]+)', fields):
                number = float(value)
                if name != 'interval_ms' or number > 0:
                    commands[kind.lower() + '.' + name].append(number)
        for line in text.splitlines():
            if not line.startswith('D4R_KERNEL_PROFILE '):
                continue
            fields = dict(re.findall(r'(\w+)=("[^"]+"|\S+)', line))
            kernels[(fields['kernel'].strip('"'), fields['backend'], fields['phase'])].append(fields)
    result = dict(cpuStages={name: stats(values) for name, values in stages.items()},
                  commandStages={name: stats(values) for name, values in commands.items()},
                  kernels=[], notes=[
                      'CPU stage times include waits and host work; they are not isolated GPU timings.',
                      'Command record intervals are between NGX recording calls on one thread, not Present/FPS.',
                      'HIP-event profiling synchronizes every sampled launch and changes scheduling.',
                      'Occupancy is an API prediction, not a measured hardware counter.',
                      'Bandwidth and WMMA utilization require additional supported hardware tooling.'])
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
    if not result['cpuStages'] and not result['kernels']:
        parser.error('no complete stage or HIP-event records found')
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, row in result['cpuStages'].items():
        print(f"CPU {name}: n={row['samples']} mean={row['mean']:.3f} ms median={row['median']:.3f} ms p95={row['p95']:.3f} ms")
    for name, row in result['commandStages'].items():
        print(f"COMMAND {name}: n={row['samples']} mean={row['mean']:.3f} ms median={row['median']:.3f} ms p95={row['p95']:.3f} ms")
    for row in result['kernels']:
        times = row['gpuMs']
        print(f"GPU {row['kernel']} {row['backend']} {row['phase']}: n={times['samples']} mean={times['mean']:.6f} ms total={times['total']:.3f} ms")


if __name__ == '__main__':
    main()
