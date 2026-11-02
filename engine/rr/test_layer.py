#!/usr/bin/env python3
"""Run one Ray Reconstruction stage shader on a captured launch and compare it with NVIDIA's result.

usage: test_layer.py CAPTURE KERNEL FRAME SHADER.spv WORK [--kvk build/rr/kvk] [--time N] [--subgroup 32|64]

KERNEL is a substring of the denoiser kernel's name (e.g. swin_enc1, hkpn_output_kernel_transformer), FRAME the
1-based occurrence in the capture. The shader follows the RR stage contract (engine/rr/README.md):

  binding 0             the parameter block: NVIDIA's argument block with every pointer into the plane arena
                        replaced by its byte offset in the arena, the weight pointer and every texture or surface
                        handle by zero (stage_params below; the runtime builds the same block)
  binding 1             the weight image exactly as NGX uploads it (argument 40), read-only
  binding 2             the plane arena, read/write
  bindings 3, 4, ...    the textures in ascending argument offset, then the surfaces in ascending argument offset

Every surface starts zeroed and the arena starts as it was before NVIDIA's launch; afterwards each surface and the
whole arena are compared with what NVIDIA's launch left. Exit status 0 only when everything is byte-identical.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_capture import Capture  # noqa: E402


# GLSL image formats of the stage shaders' surfaces -> (channels, kvk bits)
SURFACE_FORMATS = {'rgba16f': (4, 16), 'rg16f': (2, 16), 'r16f': (1, 16), 'rgba8_snorm': (4, 9), 'rgba8': (4, 8)}


def stage_params(launch, weights_arg=40):
    args = bytearray(launch.args)
    arena = launch.arena_alloc
    for p in launch.pointers:
        value = p.delta if p.alloc == arena and p.offset != weights_arg else 0
        struct.pack_into('<Q', args, p.offset, value)
    # texture and surface handles: 1 where NGX bound one (the kernels test optional inputs' handles for null, e.g.
    # enc0's specular hit distance at argument 248), 0 elsewhere
    for image in launch.images:
        struct.pack_into('<Q', args, image.offset, 1)
    return bytes(args)


def e4m3_table():
    t = np.zeros(256)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        v = np.nan if (e == 15 and m == 7) else (m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
        t[c] = -v if s else v
    return t


def psnr(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float('nan')
    err = np.mean((a[finite] - b[finite]) ** 2)
    peak = max(np.max(np.abs(b[finite])), 1e-6)
    return float('inf') if err == 0 else 10 * np.log10(peak * peak / err)


def compare_image(name, image, got, ref, ignore_channels=()):
    """IGNORE_CHANNELS: channels NVIDIA's kernel fills from an undefined register (left out of the comparison)"""
    if image.bits == 16:
        a, b = np.frombuffer(got, np.float16).astype(np.float64), np.frombuffer(ref, np.float16).astype(np.float64)
    elif image.bits == 32:
        a, b = np.frombuffer(got, np.float32).astype(np.float64), np.frombuffer(ref, np.float32).astype(np.float64)
    else:
        dt = np.int8 if image.bits == 9 else np.uint8
        a, b = np.frombuffer(got, dt).astype(np.float64), np.frombuffer(ref, dt).astype(np.float64)
    keep = np.ones(len(a), bool)
    for c in ignore_channels:
        keep[c::image.channels] = False
    a, b = a[keep], b[keep]
    element_bytes = len(got) // len(keep)
    same = (np.frombuffer(got, np.uint8).reshape(-1, element_bytes) == np.frombuffer(ref, np.uint8).reshape(-1, element_bytes))[keep]
    elements = len(a)
    equal = np.sum(a == b) + np.sum(np.isnan(a) & np.isnan(b))
    worst = np.nanmax(np.abs(a - b)) if elements else 0
    print(f'  {name}: {image.width}x{image.height}x{image.channels} fmt{image.format}'
          f'{" (channels " + ",".join(map(str, ignore_channels)) + " ignored)" if ignore_channels else ""}: '
          f'{100.0 * equal / elements:.4f}% identical values, max |diff| {worst:.6g}, PSNR {psnr(a, b):.2f} dB, '
          f'nonfinite ours/ref {int(np.sum(~np.isfinite(a)))}/{int(np.sum(~np.isfinite(b)))}')
    return bool(same.all())


def compare_arena(got, before, ref):
    g, p, r = (np.frombuffer(x, np.uint8) for x in (got, before, ref))
    written = (r != p) | (g != p)
    differ = g != r
    print(f'  arena: {int(written.sum())} bytes written by either side, {int(differ.sum())} differ', end='')
    if differ.any():
        idx = np.nonzero(differ)[0]
        t = e4m3_table()
        dv = np.abs(t[g[idx]] - t[r[idx]])
        print(f' (first at byte {idx[0]:#x}, last at {idx[-1]:#x}; as e4m3 values: max |diff| {np.nanmax(dv):.6g}, '
              f'{int(np.sum(dv <= 0.0625 * np.abs(t[r[idx]]) + 1e-12))} within one e4m3 step)')
    else:
        print()
    return not differ.any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('kernel')
    ap.add_argument('frame', type=int)
    ap.add_argument('shader')
    ap.add_argument('work')
    ap.add_argument('--kvk', default=str(Path(__file__).resolve().parents[2] / 'build/rr/kvk'))
    ap.add_argument('--time', type=int, default=1)
    ap.add_argument('--subgroup', type=int, default=32)
    ap.add_argument('--weights', help='bind this file at binding 1 instead of NGX\'s weight image (native stages)')
    ap.add_argument('--weights-arg', type=int, default=40, help='argument offset of the weight pointer (rr2_*: 24)')
    ap.add_argument('--grid', help='X,Y workgroups (default: the launch grid)')
    ap.add_argument('--ignore', action='append', default=[], metavar='SURFACE:CHANNEL',
                    help='leave a channel out of the comparison, e.g. 112:3 for a value NVIDIA stores from an '
                         'undefined register (repeatable)')
    o = ap.parse_args()

    cap = Capture(o.capture)
    launch = [l for l in cap.launches if o.kernel in l.kernel and 'diamond' not in l.kernel or
              ('diamond' in o.kernel and o.kernel in l.kernel)][o.frame - 1]
    work = Path(o.work)
    work.mkdir(parents=True, exist_ok=True)
    print(f'{launch.kernel} (launch {launch.sequence}, grid {launch.grid}, block {launch.block})')

    (work / 'params.bin').write_bytes(stage_params(launch, o.weights_arg))
    (work / 'weights.bin').write_bytes(Path(o.weights).read_bytes() if o.weights else launch.weights_at(o.weights_arg))
    arena = launch.arena_alloc
    arena_before = launch.pre(arena) if arena is not None else bytes(16)
    (work / 'arena.bin').write_bytes(arena_before)
    gx, gy = (int(v) for v in o.grid.split(',')) if o.grid else launch.grid[:2]
    lines = [f'param {work}/params.bin', f'grid {gx} {gy}']
    # Bind in the shader's order (its ptx2rr.py sidecar, SHADER.json: textures then surfaces by argument offset).
    # A handle the shader reads but NGX left null gets a zeroed 1x1 image.
    sidecar = Path(o.shader).with_suffix('.json')
    if sidecar.exists():
        info = json.loads(sidecar.read_text())
        order = [('texture', off, None) for off in info['textures']] + [('surface', off, f) for off, f in info['surfaces']]
    else:
        order = [(i.kind, i.offset, None) for i in launch.images]
    by_key = {(i.kind, i.offset): i for i in launch.images}
    (work / 'zero.bin').write_bytes(bytes(64))
    for kind, offset, fmt in order:
        image = by_key.get((kind, offset))
        if image is None:
            ch, bits = SURFACE_FORMATS.get(fmt, (4, 16))
            lines.append(f'tex {offset} {work}/zero.bin 1 1 4 16 nearest' if kind == 'texture' else
                         f'surf {offset} 1 1 {ch} {bits} - {work}/dummy{offset}.bin')
        elif kind == 'texture':
            lines.append(f'tex {image.offset} {image.path} {image.width} {image.height} {image.channels} {image.bits} '
                         f'{"linear" if image.linear else "nearest"}')
        else:
            lines.append(f'surf {image.offset} {image.width} {image.height} {image.channels} {image.bits} - '
                         f'{work}/surf{image.offset}.bin')
    n_images = len(order)
    lines.append(f'lut {work}/weights.bin')
    lines.append(f'inout {work}/arena.bin {work}/arena.out.bin')
    # further buffers of the stage (ptx2rr.py --buffer), bound after the surfaces
    extra = info.get('buffers', []) if sidecar.exists() else []
    for off in extra:
        present = any(p.offset == off for p in launch.pointers)   # NGX may leave the pointer null
        (work / f'buf{off}.bin').write_bytes(launch.pre(launch.pointer(off).alloc) if present else bytes(16))
        lines.append(f'inout {work}/buf{off}.bin {work}/buf{off}.out.bin')
    probes = info.get('probes', []) if sidecar.exists() else []
    oob = info.get('oob', False) if sidecar.exists() else False
    if probes:   # ptx2rr.py --probe: the debug buffer after the surfaces
        # D4R_PROBE_THREADS: the shader's workgroup size when it is not the launch's block (the hybrid stages)
        threads = int(os.environ.get('D4R_PROBE_THREADS', launch.block[0] * launch.block[1] * launch.block[2]))
        lines.append(f'outbuf {work}/debug.bin {4 * threads * len(probes)}')
    elif oob:    # ptx2rr.py --oob-report: count, lowest and highest address per buffer
        (work / 'oob.init').write_bytes(struct.pack('<9I', *([0, 0xffffffff, 0] * 3)))
        lines.append(f'inout {work}/oob.init {work}/oob.bin')
    lines.append('bindings 0 ' + ' '.join(str(3 + i) for i in range(n_images)) + ' 1 2' +
                 ''.join(f' {3 + n_images + i}' for i in range(len(extra))) +
                 (f' {3 + n_images + len(extra)}' if probes or oob else ''))
    (work / 'manifest').write_text('\n'.join(lines) + '\n')
    r = subprocess.run([o.kvk, o.shader, str(work / 'manifest'), str(o.time), str(o.subgroup)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(f'kvk failed ({r.returncode})')

    identical = True
    bound = {(kind, offset) for kind, offset, _ in order}
    for image in launch.images:
        if image.kind != 'surface':
            continue
        if ('surface', image.offset) not in bound:
            print(f'  surface {image.offset}: not written by this kernel')
            continue
        got = (work / f'surf{image.offset}.bin').read_bytes()
        waived = [int(s.split(':')[1]) for s in o.ignore if int(s.split(':')[0]) == image.offset]
        identical &= compare_image(f'surface {image.offset}', image, got, image.path.read_bytes(), waived)
    if arena is not None:
        identical &= compare_arena((work / 'arena.out.bin').read_bytes(), arena_before,
                                   launch.post_buffer(cap, arena))
    for off in extra:
        if not any(p.offset == off for p in launch.pointers):
            continue
        got, ref = (work / f'buf{off}.out.bin').read_bytes(), launch.post_buffer(cap, launch.pointer(off).alloc)
        same = got[:len(ref)] == ref
        print(f'  buffer {off}: {len(ref)} bytes, ' + ('identical' if same else f'{sum(x != y for x, y in zip(got, ref))} differ'))
        identical &= same
    if oob:
        r = struct.unpack('<9I', (work / 'oob.bin').read_bytes()[:36])
        for k, name in enumerate(('weight reads', 'arena reads', 'arena writes')):
            print(f'  out of range {name}: {r[3 * k]}' + (f' (bytes {r[3 * k + 1]:#x}..{r[3 * k + 2]:#x}, size '
                  f'{len(launch.weights) if k == 0 else len(arena_before):#x})' if r[3 * k] else ''))
    print('IDENTICAL' if identical else 'DIFFERENT')
    sys.exit(0 if identical else 1)


if __name__ == '__main__':
    main()
