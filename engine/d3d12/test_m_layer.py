#!/usr/bin/env python3
"""Validate and time native D3D12 M/L network blocks with seeded, redistributable inputs.

Both presets share these ten network layers; L's pixel-resolution input and expansion
are separate stages. This test does not establish complete-frame accuracy or timing.
Optional Vulkan comparisons use scalar FP16 products because AMD's Windows Vulkan
driver used during development rejects SPV_VALVE_mixed_float_dot_product.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
sys.path.insert(0, str(ENGINE))
import hlsl_build
import m_model as mm
import model_enc3 as me
from test_m import codes, dot2_defines, dot2_merge_defines, pack_layer, widen


def synthetic(name, seed, zero=False):
    c, heads, merged, expanded = mm.LAYERS[name]
    rng = np.random.default_rng(seed)
    f16 = lambda shape: rng.normal(0, .02, shape).astype(np.float16)
    fp8 = lambda shape: me.q8(rng.normal(0, .08, shape)).astype(np.float16)
    t = dict(g1=np.ones(c, np.float16), g2=np.ones(c, np.float16), bo=f16(c), b2=f16(c),
             q=[fp8((c, 32)) for _ in range(heads)], v=[fp8((c, 32)) for _ in range(heads)],
             bias=[f16((16, 16)) for _ in range(heads)], wo=fp8((32 * heads, c)),
             w1=[fp8((c, 32)) for _ in range(c // 8)], b1=[f16(32) for _ in range(c // 8)],
             w2=[fp8((32, c)) for _ in range(c // 8)])
    if merged:
        t.update(merge_w=fp8((4 * c, merged)), merge_b=f16(merged))
    if expanded:
        t.update(expand_w=fp8((expanded, 4 * c)), expand_b=f16(4 * c))
    if zero:
        for key, value in t.items():
            if key not in ('g1', 'g2'):
                for a in value if isinstance(value, list) else [value]:
                    a[...] = 0
    return t, rng


def run(command, **kwargs):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed:\n{result.stdout}{result.stderr}')
    return result.stdout + result.stderr


def statistics(got, expected):
    if got.shape != expected.shape or not np.isfinite(me.E4M3[got]).all():
        raise AssertionError('invalid output shape or nonfinite FP8')
    return dict(psnr_db=None if np.array_equal(got, expected) else float(mm.psnr(me.E4M3[got].astype(float), me.E4M3[expected].astype(float))),
                identical_fraction=float(np.mean(got == expected)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('--runner', required=True, type=Path)
    p.add_argument('--vulkan-runner', type=Path)
    p.add_argument('--glslang', default='glslangValidator')
    p.add_argument('--dxc')
    p.add_argument('--cbv', action='store_true')
    p.add_argument('--split-expand', action='store_true')
    p.add_argument('--vector-io', action='store_true', help='use uint4 network views; decoder layers require --split-expand')
    p.add_argument('--define', action='append', default=[], help='additional DXC definition, NAME=VALUE')
    p.add_argument('--layers', nargs='+', choices=list(mm.LAYERS), default=list(mm.LAYERS))
    p.add_argument('--width', type=int, default=26)
    p.add_argument('--height', type=int, default=18)
    p.add_argument('--shift', nargs=2, type=int, default=[2, 2])
    p.add_argument('--repeats', type=int, default=25)
    p.add_argument('--seed', type=int, default=20261023)
    p.add_argument('--zero', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--merge', action='store_true', help='also validate the encoder patch merge')
    p.add_argument('--fast', action='store_true', help='HIP fast arithmetic: skip intermediate FP8 requantization and K32 half rounding')
    p.add_argument('--backend', choices=['dot2', 'wmma'], default='dot2')
    p.add_argument('--parameter-words', action='store_true')
    a = p.parse_args()
    if min(a.width, a.height) < 16 or a.width % 2 or a.height % 2:
        p.error('dimensions must be even and at least 16')
    if not all(s in (0, 2) for s in a.shift) or not 1 <= a.repeats <= 10000:
        p.error('shifts must be 0 or 2; repeats must be in [1, 10000]')
    if a.vulkan_runner and a.backend != 'dot2':
        p.error('--vulkan-runner currently compares the dot2 backend only')
    if a.fast and a.backend != 'wmma':
        p.error('--fast requires --backend wmma')
    if a.vector_io and (a.backend != 'wmma' or (not a.split_expand and any(mm.LAYERS[n][3] for n in a.layers))):
        p.error('--vector-io requires WMMA and split decoder expansion')
    a.work.mkdir(parents=True, exist_ok=True)
    records = []
    for name in a.layers:
        out = a.work / name
        out.mkdir(exist_ok=True)
        c, heads, merged, expanded = mm.LAYERS[name]
        t, rng = synthetic(name, a.seed + list(mm.LAYERS).index(name), a.zero)
        o, b8, b16 = pack_layer(t, name)
        (out / 'params.bin').write_bytes(struct.pack('<4i72I', a.width, a.height, *a.shift, *o))
        (me.E4M3[b8].astype(np.float16) if a.backend == 'wmma' else widen(b8, o, name)).tofile(out / 'weights.bin')
        b16.tofile(out / 'vectors.bin')
        shape = (a.height // 2, a.width // 2, expanded) if expanded else (a.height, a.width, c)
        x = codes(me.q8(rng.normal(0, .4, shape)))
        skip = codes(me.q8(rng.normal(0, .4, (a.height, a.width, c)))) if expanded else np.zeros(64, np.uint8)
        x.tofile(out / 'input.bin')
        skip.tofile(out / 'skip.bin')
        input_file = out / 'input.bin'
        dxil = out / 'layer.dxil'
        defs = dot2_defines(name) + ['-DD4R_NATIVE_HALF=1'] + (['-DD4R_M_NO_Q8=1', '-DD4R_M_ACC_ROUND_K32=0'] if a.fast else []) + (['-DD4R_M_PARAMETER_WORDS=1'] if a.parameter_words else [])
        defs += ['-D' + value for value in a.define] + (['-DD4R_M_LAYER_CBV=1'] if a.cbv else [])
        if a.split_expand and expanded:
            expand_shader = out / 'expand.dxil'
            run([hlsl_build.dxc_path(a.dxc), '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types', '-O3',
                 *defs, HERE / 'expand_m_wmma.hlsl', '-Fo', expand_shader])
            expanded_file = out / 'expanded.bin'
            log = run([a.runner, expand_shader, out / 'params.bin', out / 'weights.bin', out / 'vectors.bin', input_file, out / 'skip.bin', expanded_file,
                       (a.width // 2 + 5) // 4, (a.height // 2 + 5) // 4, a.repeats, (a.width + 4) * (a.height + 4) * c * 2,
                       '--wmma', *(['--cbv'] if a.cbv else [])])
            (out / 'expand.log').write_text(log)
            input_file = expanded_file
            defs = [d for d in defs if not d.startswith('-DKIND=')] + ['-DKIND=0', '-DD4R_M_INPUT_F16=1']
        if a.backend == 'wmma':
            if a.vector_io: defs += ['-DD4R_M_VECTOR_IO=1']
            run([hlsl_build.dxc_path(a.dxc), '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types', '-O3',
                 *defs, HERE / 'layer_m_wmma.hlsl', '-Fo', dxil])
        else:
            hlsl_build.build(ENGINE / 'layer_m_dot2.comp', dxil, defs, wave=32, dxc=a.dxc)
        tile = 4 if a.backend == 'wmma' else 8
        gx, gy = (a.width + a.shift[0] + tile - 1) // tile, (a.height + a.shift[1] + tile - 1) // tile
        command = [a.runner, dxil, out / 'params.bin', out / 'weights.bin', out / 'vectors.bin',
                   input_file, out / 'skip.bin', out / 'd3d12.bin', gx, gy, a.repeats,
                   a.width * a.height * c]
        if a.backend == 'wmma':
            command += ['--wmma']
        if a.cbv: command += ['--cbv']
        if a.vector_io: command += ['--vector-io']
        if a.parameter_words:
            command += ['--parameter-words']
        if a.debug:
            command += ['--debug']
        log = run(command)
        (out / 'd3d12.log').write_text(log)
        got = np.fromfile(out / 'd3d12.bin', np.uint8).reshape(a.height, a.width, c)
        original_block = mm.block
        def fast_block(*args):
            original_q8 = mm.q8
            try:
                mm.q8 = lambda v: np.asarray(v)
                return original_block(*args)
            finally:
                mm.q8 = original_q8
        original_forward_q8 = mm.q8
        expansion_calls = [0]
        def fast_expansion_q8(v):
            expansion_calls[0] += 1
            return np.asarray(v) if expansion_calls[0] == 1 else original_forward_q8(v)
        try:
            if a.fast:
                mm.block = fast_block
                if expanded: mm.q8 = fast_expansion_q8
            reference, _ = mm.forward(t, name, me.E4M3[x], me.E4M3[skip] if expanded else None, *a.shift)
        finally:
            mm.block = original_block
            mm.q8 = original_forward_q8
        expected = codes(reference)
        stats = statistics(got, expected)
        if stats['identical_fraction'] < 1 and stats['psnr_db'] < (50 if a.fast else 35):
            raise AssertionError(f'{name}: CPU reference agreement too low: {stats}')
        if a.zero:
            np.testing.assert_array_equal(got, skip if expanded else x)
        record = dict(layer=name, shape=[a.width, a.height, c], cpu=stats,
                      d3d12_us=float(re.search(r'([\d.]+) us per dispatch', log)[1]),
                      shader_sha256=hashlib.sha256(dxil.read_bytes()).hexdigest())
        if a.vulkan_runner and a.backend == 'dot2':
            spv = out / 'layer.spv'
            run([a.glslang, '-V', '--target-env', 'vulkan1.3', *dot2_defines(name), '-DD4R_PORTABLE_DOT2=1',
                 ENGINE / 'layer_m_dot2.comp', '-o', spv])
            env = dict(os.environ, D4R_EVK_GPU_TIMING='1')
            vklog = run([a.vulkan_runner, spv, gx, gy, a.repeats,
                         *[f'in:{i}={out / file}' for i, file in enumerate(
                             ['params.bin', 'weights.bin', 'vectors.bin', 'input.bin', 'skip.bin'])],
                         f'out:5={out / "vulkan.bin"}:{a.width * a.height * c}'], env=env)
            (out / 'vulkan.log').write_text(vklog)
            vk = np.fromfile(out / 'vulkan.bin', np.uint8).reshape(got.shape)
            record['vulkan'] = statistics(got, vk)
            if record['vulkan']['identical_fraction'] < 1 and record['vulkan']['psnr_db'] < 38:
                raise AssertionError(f'{name}: Vulkan agreement too low: {record["vulkan"]}')
            record['vulkan_us'] = float(re.search(r'([\d.]+) us per dispatch', vklog)[1])
        if a.merge and merged:
            merge_shader = out / 'merge.dxil'
            if a.backend == 'wmma':
                run([hlsl_build.dxc_path(a.dxc), '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types', '-O3',
                     *dot2_merge_defines(name), '-DD4R_NATIVE_HALF=1', *(['-DD4R_M_LAYER_CBV=1'] if a.cbv else []), HERE / 'merge_m_wmma.hlsl', '-Fo', merge_shader])
                merge_tile = 4
            else:
                hlsl_build.build(ENGINE / 'merge_m_dot2.comp', merge_shader,
                                 dot2_merge_defines(name) + ['-DD4R_NATIVE_HALF=1'], wave=32, dxc=a.dxc)
                merge_tile = 8
            merge_log = run([a.runner, merge_shader, out / 'params.bin', out / 'weights.bin', out / 'vectors.bin',
                             out / 'd3d12.bin', out / 'skip.bin', out / 'merged.bin',
                             (a.width // 2 + merge_tile - 1) // merge_tile,
                             (a.height // 2 + merge_tile - 1) // merge_tile, a.repeats,
                             a.width * a.height * merged // 4, '--merge',
                             *(['--wmma'] if a.backend == 'wmma' else []), *(['--cbv'] if a.cbv else []), *(['--debug'] if a.debug else [])])
            (out / 'merge.log').write_text(merge_log)
            values = me.E4M3[got].astype(float)
            patches = values.reshape(a.height // 2, 2, a.width // 2, 2, c).transpose(0, 2, 1, 3, 4).reshape(a.height // 2, a.width // 2, 4 * c)
            expected_merge = codes(me.q8(np.clip(mm.h(patches @ t['merge_w'].astype(float) + t['merge_b'].astype(float)), -2 * np.pi, 2 * np.pi)))
            merged_got = np.fromfile(out / 'merged.bin', np.uint8).reshape(expected_merge.shape)
            record['merge'] = statistics(merged_got, expected_merge)
            record['merge_us'] = float(re.search(r'([\d.]+) us per dispatch', merge_log)[1])
            if record['merge']['identical_fraction'] < 1 and record['merge']['psnr_db'] < 35:
                raise AssertionError(f'{name}: patch merge agreement too low: {record["merge"]}')
            if a.zero:
                np.testing.assert_array_equal(merged_got, np.zeros_like(merged_got))
        records.append(record)
        print(json.dumps(record), flush=True)
    (a.work / 'summary.json').write_text(json.dumps(dict(arguments={k: str(v) if isinstance(v, Path) else v
                                                                  for k, v in vars(a).items()}, layers=records), indent=2))


if __name__ == '__main__':
    main()
