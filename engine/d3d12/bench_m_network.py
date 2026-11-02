#!/usr/bin/env python3
"""Time isolated native D3D12 M/L network dispatches on seeded synthetic tensors.

The sum excludes frame preparation, input/expansion, reconstruction, downsample,
command submission, and synchronization between stages. It is not a game or a
complete-frame benchmark. Each tube layer has its own weights and input. Run the
small test_m_layer.py accuracy cases before using this script for tuning.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
import numpy as np
from test_m_layer import HERE, run, synthetic
from test_m import codes, dot2_defines, dot2_merge_defines, pack_layer
import hlsl_build
import model_enc3 as me
import m_model as mm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('--runner', required=True, type=Path)
    p.add_argument('--dxc')
    p.add_argument('--render', nargs=2, type=int, default=[1280, 720])
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--repeats', type=int, default=25)
    p.add_argument('--math', choices=['fast', 'fp8'], default='fast')
    a = p.parse_args()
    if not all(64 <= d <= 8192 for d in a.render) or not 1 <= a.trials <= 20 or not 1 <= a.repeats <= 10000:
        p.error('render dimensions must be in [64,8192], trials in [1,20], repeats in [1,10000]')
    a.work.mkdir(parents=True, exist_ok=True)
    extra = ['-DD4R_NATIVE_HALF=1'] + (['-DD4R_M_NO_Q8=1', '-DD4R_M_ACC_ROUND_K32=0'] if a.math == 'fast' else [])
    shaders = {}
    for name in mm.LAYERS:
        shaders[name] = a.work / (name + '.dxil')
        run([hlsl_build.dxc_path(a.dxc), '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types',
             '-O3', *dot2_defines(name), *extra, HERE / 'layer_m_wmma.hlsl', '-Fo', shaders[name]])
        if name in ('enc1', 'enc2'):
            run([hlsl_build.dxc_path(a.dxc), '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types',
                 '-O3', *dot2_merge_defines(name), '-DD4R_NATIVE_HALF=1', HERE / 'merge_m_wmma.hlsl',
                 '-Fo', a.work / (name + '-merge.dxil')])
    tw, th = [(d + 31) // 32 * 16 for d in a.render]
    scales = [1, 2, 4, 4, 4, 4, 4, 4, 2, 1]
    shifts = [2, 2, 2, 0, 2, 0, 2, 0, 0, 0]
    commands, metadata = [], []
    for index, name in enumerate(mm.ORDER):
        d = a.work / f'{index}-{name}'
        d.mkdir(exist_ok=True)
        c, _, merged, expanded = mm.LAYERS[name]
        w, h, shift = tw // scales[index], th // scales[index], shifts[index]
        tensors, rng = synthetic(name, 20261023 + index)
        offsets, b8, b16 = pack_layer(tensors, name)
        (d / 'params.bin').write_bytes(struct.pack('<4i72I', w, h, shift, shift, *offsets))
        me.E4M3[b8].astype(np.float16).tofile(d / 'weights.bin')
        b16.tofile(d / 'vectors.bin')
        shape = (h // 2, w // 2, expanded) if expanded else (h, w, c)
        codes(me.q8(rng.normal(0, .4, shape))).tofile(d / 'input.bin')
        (codes(me.q8(rng.normal(0, .4, (h, w, c)))) if expanded else np.zeros(64, np.uint8)).tofile(d / 'skip.bin')
        common = [d / file for file in ['params.bin', 'weights.bin', 'vectors.bin']]
        commands.append([a.runner, shaders[name], *common, d / 'input.bin', d / 'skip.bin', d / 'out.bin',
                         (w + shift + 3) // 4, (h + shift + 3) // 4, a.repeats, w * h * c, '--wmma'])
        metadata.append(dict(index=index, layer=name, shape=[w, h, c]))
        if merged:
            commands.append([a.runner, a.work / (name + '-merge.dxil'), *common, d / 'out.bin', d / 'skip.bin',
                             d / 'merged.bin', (w // 2 + 3) // 4, (h // 2 + 3) // 4, a.repeats,
                             w * h * merged // 4, '--wmma', '--merge'])
            metadata.append(dict(index=index, layer=name + '-merge', shape=[w // 2, h // 2, merged]))
    samples = []
    # The first pass establishes merge inputs before rotating the process order.
    for trial in range(a.trials):
        order = list(range(len(commands)))
        if trial:
            order = order[trial:] + order[:trial]
        for index in order:
            log = run(commands[index])
            (a.work / f'trial-{trial}-dispatch-{index}.log').write_text(log)
            row = dict(metadata[index], trial=trial, gpu_us=float(re.search(r'([\d.]+) us per dispatch', log)[1]))
            if trial == 0:
                output = np.fromfile(commands[index][7], np.uint8)
                if output.size != int(commands[index][11]) or not np.isfinite(me.E4M3[output]).all():
                    raise RuntimeError(f'{metadata[index]}: invalid output size or nonfinite FP8')
            samples.append(row)
            print(json.dumps(row), flush=True)
    medians = [dict(m, gpu_us=float(np.median([s['gpu_us'] for s in samples if s['index'] == m['index'] and s['layer'] == m['layer']])))
               for m in metadata]
    result = dict(render=a.render, token_grid=[tw, th], math=a.math, trials=a.trials, repeats=a.repeats,
                  isolated_dispatch_sum_ms=sum(m['gpu_us'] for m in medians) / 1000, dispatches=medians, samples=samples,
                  source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                 for name in ['layer_m_wmma.hlsl', 'merge_m_wmma.hlsl', 'ags_wmma.hlsli']})
    (a.work / 'summary.json').write_text(json.dumps(result, indent=2))
    print(f"Isolated dispatch sum: {result['isolated_dispatch_sum_ms']:.3f} ms; frame stages excluded.")


if __name__ == '__main__':
    main()
