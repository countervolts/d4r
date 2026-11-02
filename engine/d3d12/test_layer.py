#!/usr/bin/env python3
"""Build, replay and check a plain K layer using the existing extracted model format.

No captures are needed for --synthetic: seeded nonzero weights exercise the full
attention + residual + MLP block against an independent NumPy implementation.
Real replay: --model PREFIX --layer dec5 --input INPUT.f16 --width W --height H.
--backend wmma selects the accelerated AMD path; dot2 remains the default baseline.
The input is the engine's plain f16 token grid, not an NGX-swizzled allocation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from extract_k import Blob, tile16
from test_k import dot2_weights

F16, F32, F64 = np.float16, np.float32, np.float64
WMMA_VARIANTS = {
    'baseline': (0, 0),
    'wave-sync': (1, 0),
    'prefetch': (0, 1),
    'combined': (1, 1),
}


def wmma_defines(variant):
    sync, prefetch = WMMA_VARIANTS[variant]
    return [f'-DWMMA_WAVE_SYNC={sync}', f'-DWMMA_PREFETCH={prefetch}']


def synthetic(channels, heads, seed):
    rng = np.random.default_rng(seed)
    blob = Blob()
    e = dict(kind='plain', C=channels, H=heads, X=0, posattn=False, fold_gains=True,
             qkv=[], table=[], wo=[], w1=[], b1=[], w2=[])

    def matrix(k, n, scale):
        return blob.add(rng.normal(0, scale, (k, n)))

    e['g1'] = blob.add(tile16(np.ones(channels)))
    e['g2'] = blob.add(tile16(np.ones(channels)))
    e['bo'] = blob.add(tile16(rng.normal(0, .03, channels)))
    e['b2'] = blob.add(tile16(rng.normal(0, .03, channels)))
    for _ in range(heads):
        e['qkv'].extend(matrix(channels, 32, .15) for _ in range(3))
        e['table'].append(matrix(64, 64, .04))
        e['wo'].append(matrix(32, channels, .02))
    for _ in range(channels // 8):
        e['w1'].append(matrix(channels, 32, .04))
        e['b1'].append(blob.add(tile16(rng.normal(0, .03, 32))))
        e['w2'].append(matrix(32, channels, .02))
    return e, np.concatenate(blob.parts), rng


def layer_params(e, width, height, sx, sy):
    offsets = [0] * 108
    offsets[:4] = [e[k] for k in ('g1', 'bo', 'g2', 'b2')]
    for h in range(e['H']):
        offsets[4 + 5*h:9 + 5*h] = [*e['qkv'][3*h:3*h+3], e['table'][h], e['wo'][h]]
    for ch in range(e['C'] // 8):
        offsets[44+3*ch:47+3*ch] = [e['w1'][ch], e['b1'][ch], e['w2'][ch]]
    return struct.pack('<4i108I', width, height, sx, sy, *offsets)


def validate_model(e, weights):
    if e['kind'] != 'plain' or e['posattn'] or not e.get('fold_gains', True):
        raise ValueError('prototype supports plain, non-positional K layers with folded gains only')
    c, heads = e['C'], e['H']
    if c not in (32, 64, 96, 128, 160) or heads not in (2, 4, 6, 8):
        raise ValueError('unsupported C/head count')
    if len(weights) % 2 or not np.isfinite(weights).all():
        raise ValueError('weights must be finite and DWORD aligned')
    tensors = [('g1', c), ('bo', c), ('g2', c), ('b2', c)]
    for key, count, size in [('qkv', 3*heads, c*32), ('table', heads, 64*64),
                             ('wo', heads, 32*c), ('w1', c//8, c*32),
                             ('b1', c//8, 32), ('w2', c//8, 32*c)]:
        if len(e[key]) != count:
            raise ValueError(f'{key}: incorrect tensor count')
        tensors.extend((off, size) for off in e[key])
    for key, size in tensors:
        off = e[key] if isinstance(key, str) else key
        if not isinstance(off, int) or off < 0 or off % 2 or off + size > len(weights):
            raise ValueError(f'invalid tensor offset: {key}')


def dot2(a, b, acc=None):
    """f16 pairs, f32 accumulator; one rounding per two products + accumulator.

    Float64 is sufficient for these bounded fixtures. Hardware may choose a
    different dot2 evaluation order, so the comparison permits rounding error.
    """
    a, b = np.asarray(a, F64), np.asarray(b, F64)
    out = np.zeros((len(a), b.shape[1]), F32) if acc is None else np.array(acc, F32)
    for k in range(0, a.shape[1], 2):
        out = (out.astype(F64) + a[:, k:k+1]*b[k] + a[:, k+1:k+2]*b[k+1]).astype(F32)
    return out


def half_mad(a, b, c):
    return (a.astype(F64)*np.asarray(b, F64) + np.asarray(c, F64)).astype(F16)


def block_reference(x, e, weights):
    c = e['C']

    def tensor(off, rows, cols):
        return weights[off:off + rows*cols].reshape(rows, cols)

    def bias(off, cols):
        return np.broadcast_to(weights[off:off+cols], (64, cols))

    sums = np.zeros((64, 2), F32)
    for k in range(0, c, 2):
        pair = x[:, k:k+2].astype(F32)
        sums = (sums + pair*pair).astype(F32)
    inv = (1 / np.sqrt((sums[:, 0] + sums[:, 1]).astype(F16).astype(F32))).astype(F16)
    norm = (x * inv[:, None]).astype(F16)
    acc = np.array(bias(e['bo'], c), F32)
    token_group = (np.arange(64) % 32) & ~7
    for h in range(e['H']):
        q, k, v = [dot2(norm, tensor(off, c, 32)).astype(F16) for off in e['qkv'][3*h:3*h+3]]
        scores = dot2(q, k.T, tensor(e['table'][h], 64, 64)).astype(F16)
        t = np.clip(scores, np.uint16(0xcd69).view(F16), np.uint16(0x4d69).view(F16))
        vv = half_mad((t*t).astype(F16), np.uint16(0x91d5).view(F16), 1)
        exponent = half_mad(t, vv, np.uint16(0xc50d).view(F16))
        probs = np.exp2(exponent.astype(F32)).astype(F16)
        sums = np.zeros((64, 2), F32)
        o = np.zeros((64, 32), F32)
        for d in range(4):
            for half in range(2):
                for j in range(4):
                    key = ((token_group + 8*d) & 31) + 32*half + 2*j
                    p0 = probs[np.arange(64), key]
                    p1 = probs[np.arange(64), key+1]
                    sums = (sums + np.stack((p0, p1), axis=1).astype(F32)).astype(F32)
                    o = (o.astype(F64) + p0[:, None].astype(F64)*v[key].astype(F64)
                         + p1[:, None].astype(F64)*v[key+1].astype(F64)).astype(F32)
        inv = (1 / (sums[:, 0] + sums[:, 1]).astype(F16).astype(F32)).astype(F16)
        o = (o * inv[:, None].astype(F32)).astype(F16)
        acc = dot2(o, tensor(e['wo'][h], 32, c), acc)
    y = (acc.astype(F16) + x).astype(F16)
    acc = (y + bias(e['b2'], c)).astype(F16).astype(F32)
    for w1, b1, w2 in zip(e['w1'], e['b1'], e['w2']):
        hh = dot2(y, tensor(w1, c, 32), bias(b1, 32)).astype(F16)
        clamped = np.clip(hh, F16(-2), F16(2))
        u = (F16(.41216981) - (F16(.08108133)*np.abs(clamped)).astype(F16)).astype(F16)
        gate = (F16(.5) + (clamped*u).astype(F16)).astype(F16)
        hidden = (hh*gate).astype(F16)
        acc = dot2(hidden, tensor(w2, 32, c), acc)
    return acc.astype(F16)


def reference(inputs, e, weights, sx, sy):
    height, width, c = inputs.shape
    output = np.full_like(inputs, np.nan)
    for by in range(math.ceil((height + sy) / 8)):
        for bx in range(math.ceil((width + sx) / 8)):
            yy, xx = np.divmod(np.arange(64), 8)
            yy, xx = yy + 8*by - sy, xx + 8*bx - sx
            my = np.minimum(np.abs(yy), 2*height-2-np.abs(yy))
            mx = np.minimum(np.abs(xx), 2*width-2-np.abs(xx))
            block = block_reference(inputs[my, mx], e, weights)
            live = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            output[yy[live], xx[live]] = block[live]
    return output


def compare(expected, got, min_psnr):
    if expected.shape != got.shape or not np.isfinite(expected).all() or not np.isfinite(got).all():
        raise ValueError('incorrect output shape or nonfinite output/reference')
    error = expected.astype(F64) - got.astype(F64)
    peak = max(float(np.max(np.abs(expected))), 1e-9)
    psnr = 10*math.log10(peak*peak / max(float(np.mean(error*error)), 1e-30))
    result = dict(psnr_db=psnr, max_abs_error=float(np.max(np.abs(error))),
                  exact_percent=100*float(np.mean(expected.view(np.uint16) == got.view(np.uint16))))
    if psnr < min_psnr:
        raise ValueError(f'accuracy failed: {result}; minimum {min_psnr} dB')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--backend', choices=('dot2', 'wmma'), default='dot2')
    p.add_argument('--wmma-variant', choices=WMMA_VARIANTS, default='baseline')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', type=int, choices=(32, 64, 96, 128, 160), metavar='C')
    source.add_argument('--model', type=Path)
    p.add_argument('--heads', type=int, choices=(2, 4, 6, 8), default=8)
    p.add_argument('--layer', default='dec5')
    p.add_argument('--input', type=Path)
    p.add_argument('--reference', type=Path, help='also compare an existing Vulkan/reference f16 result')
    p.add_argument('--vulkan-runner', type=Path, help='also check the existing cooperative-matrix Vulkan shader')
    p.add_argument('--vulkan-time', action='store_true', help='benchmark the Vulkan reference using GPU timestamps')
    p.add_argument('--glslang', default='glslangValidator')
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--height', type=int, default=8)
    p.add_argument('--shift', type=int, nargs=2, default=(0, 0))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--min-psnr', type=float, default=60)
    p.add_argument('--dxc', default='dxc')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--launch-prefix', nargs='+', default=[], help='e.g. wine; configure vkd3d DLL overrides separately')
    p.add_argument('--vulkan-launch-prefix', nargs='+', default=[], help='independent prefix; native Vulkan is the default, including with --launch-prefix wine')
    a = p.parse_args()
    if a.backend != 'wmma' and a.wmma_variant != 'baseline':
        p.error('--wmma-variant requires --backend wmma')
    if a.vulkan_time and (not a.vulkan_runner or a.repeats < 2):
        p.error('--vulkan-time requires --vulkan-runner and at least 2 repeats')
    sx, sy = a.shift
    if min(a.width, a.height) < 8 or max(a.width, a.height) > 4096 or not (0 <= sx < 8 and 0 <= sy < 8):
        p.error('dimensions must be 8..4096, shifts 0..7')
    if not 1 <= a.repeats <= 10000 or not math.isfinite(a.min_psnr) or a.min_psnr <= 0:
        p.error('repeats must be 1..10000 and min-psnr finite and positive')
    a.work.mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        e, weights, rng = synthetic(a.synthetic, a.heads, a.seed)
        inputs = rng.normal(0, .5, (a.height, a.width, e['C'])).astype(F16)
    else:
        if a.input is None:
            p.error('--model requires --input')
        e = json.loads(Path(str(a.model) + '.json').read_text())[a.layer]
        weights = np.fromfile(str(a.model) + '.bin', F16)
        inputs = np.fromfile(a.input, F16).reshape(a.height, a.width, e['C'])
    validate_model(e, weights)
    if a.backend == 'wmma' and e['C'] < 64:
        p.error('the plain WMMA backend requires C >= 64')
    if not np.isfinite(inputs).all():
        raise ValueError('nonfinite input')
    packed = weights if a.backend == 'wmma' else dot2_weights({a.layer: e}, weights)
    (a.work / 'params.bin').write_bytes(layer_params(e, a.width, a.height, sx, sy))
    packed.tofile(a.work / 'weights.bin')
    inputs.tofile(a.work / 'input.bin')
    (a.work / 'model.json').write_text(json.dumps({a.layer: e}, indent=2))
    shader = a.work / 'layer.dxil'
    compiler = shutil.which(a.dxc) or a.dxc
    shader_source = 'layer_k_wmma.hlsl' if a.backend == 'wmma' else 'layer_k.hlsl'
    extra = wmma_defines(a.wmma_variant) if a.backend == 'wmma' else []
    subprocess.run([compiler, '-T', 'cs_6_6', '-E', 'main', '-enable-16bit-types', '-O3',
                    f'-DC={e["C"]}', f'-DNH={e["H"]}', *extra, str(ENGINE / 'd3d12' / shader_source), '-Fo', str(shader)], check=True)
    gx, gy = str(math.ceil((a.width+sx)/8)), str(math.ceil((a.height+sy)/8))

    def replay_path(path):
        absolute = str(path.resolve())
        # Wine takes a Unix executable filename, but its Windows CRT needs
        # Windows filenames for the data arguments. Use the standard Z: mapping.
        if a.launch_prefix and os.name != 'nt':
            return 'Z:' + absolute.replace('/', '\\')
        return absolute

    cmd = [*a.launch_prefix, str(a.runner.resolve()), replay_path(shader), replay_path(a.work/'params.bin'),
           replay_path(a.work/'weights.bin'), replay_path(a.work/'input.bin'), replay_path(a.work/'output.bin'),
           gx, gy, str(a.repeats)]
    if a.debug:
        cmd.append('--debug')
    if a.backend == 'wmma':
        cmd.append('--wmma')
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(completed.stdout, end='')
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end='')
    got = np.fromfile(a.work/'output.bin', F16).reshape(inputs.shape)
    expected = reference(inputs, e, weights, sx, sy)
    expected.tofile(a.work/'cpu-reference.bin')
    result = dict(channels=e['C'], heads=e['H'], width=a.width, height=a.height, shift=a.shift,
                  synthetic=bool(a.synthetic), debug_layer=a.debug, repeats=a.repeats,
                  d3d12_kernel=a.backend, cpu=compare(expected, got, a.min_psnr), runner_stdout=completed.stdout)
    if a.backend == 'wmma':
        result['wmma_variant'] = a.wmma_variant
        result['shader_defines'] = extra
    if a.vulkan_runner:
        # evk's out:FILE:SIZE syntax predates Windows drive letters. Run in the
        # work directory and use relative data filenames to avoid that ambiguity.
        weights.tofile(a.work/'weights-row.bin')
        subprocess.run([a.glslang, '-V', '--target-env', 'vulkan1.3', f'-DC={e["C"]}', f'-DNH={e["H"]}',
                        '-DKIND=0', '-DPOSATTN=0', '-DEMBED=0', '-DHEAD=0', '-DXC=16', '-DMERGE_N=16',
                        str(ENGINE/'layer.comp'), '-o', str(a.work/'layer.spv')], check=True)
        env = dict(os.environ)
        if a.vulkan_time:
            env['D4R_EVK_GPU_TIMING'] = '1'
        vk_run = subprocess.run([*a.vulkan_launch_prefix, str(a.vulkan_runner.resolve()), 'layer.spv', gx, gy,
                        str(a.repeats if a.vulkan_time else 1), 'in:0=params.bin', 'in:1=weights-row.bin',
                        'in:2=input.bin', 'in:3=params.bin', f'out:4=vulkan.bin:{inputs.nbytes}',
                        'out:5=unused.bin:16'], cwd=a.work, env=env, check=True, capture_output=True, text=True)
        print(vk_run.stdout, end='')
        if vk_run.stderr:
            print(vk_run.stderr, file=sys.stderr, end='')
        result['vulkan'] = compare(np.fromfile(a.work/'vulkan.bin', F16).reshape(inputs.shape), got, a.min_psnr)
        result['vulkan_kernel'] = 'cooperative_matrix'
        if a.vulkan_time:
            vk_timing = re.search(r'([0-9.]+) us per dispatch', vk_run.stdout)
            if not vk_timing:
                raise ValueError('Vulkan runner did not report GPU timing')
            result['vulkan_gpu_us'] = float(vk_timing[1])
            result['vulkan_runner_stdout'] = vk_run.stdout
    if a.reference:
        result['external'] = compare(np.fromfile(a.reference, F16).reshape(inputs.shape), got, a.min_psnr)
    timing = re.search(r'([0-9.]+) us per dispatch', completed.stdout)
    if timing:
        result['gpu_us'] = float(timing[1])
    (a.work/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['cpu'], indent=2))
    if 'vulkan' in result:
        print('Vulkan comparison:', json.dumps(result['vulkan']))


if __name__ == '__main__':
    main()
