#!/usr/bin/env python3
"""Build the engine's preset M model directory from a capture of one frame.

usage: compile_m.py CAPTURE_DIR OUT_DIR [--frame N]
  CAPTURE_DIR/replay must hold the frame's rrlite_enc0 and rrlite_dec0 launches (the `_folded` kernels) and its
  ten Swin layer launches, with their weight allocations (D4R_CUDA_REPLAY_DUMP_DIR, filter `rrlite_`).
Writes weights8.bin (the layers' FP8 matrices in the blocked fixed-16 tile layout of test_m.pack8, with the front-end
enc0/dec0 matrices plain and NGX-ordered as their own stages read them), weights16.bin (f16 vectors and the dither
table), offsets.bin and the compiled shaders. The weights are values read from NVIDIA's DLSS and are not redistributable
with d4r; the shaders are d4r's own.
"""
# D4RM0003 couples the blocked FP8 operands, RGBA32F half-history backing and 8x64 downsample dispatch;
# the previous package's shader workgroup dimensions are not interchangeable with this runtime.
import argparse, glob, hashlib, json, os, struct, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m_model as mm
import model_enc3 as me
import swin_model as sm
from test_m import pack_layer, defines

SHADERS = {'enc1': 'm_enc1', 'enc2': 'm_enc2', 'enc3_tube': 'm_tube', 'dec2': 'm_dec2', 'dec1': 'm_dec1'}


def blob_of(directory, ptr_offset=0):
    args = open(f'{directory}/args.bin', 'rb').read()
    ptr = struct.unpack_from('<Q', args, ptr_offset)[0]
    for line in open(f'{directory}/manifest.txt'):
        q = line.split()
        if q and q[0] == 'alloc':
            base = int(q[2], 16); size = os.path.getsize(f'{directory}/alloc-{q[1]}.bin')
            if base <= ptr < base + size:
                return np.fromfile(f'{directory}/alloc-{q[1]}.bin', np.uint8)[ptr - base:]
    raise ValueError(f'{directory}: the weights allocation is missing')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture'); ap.add_argument('output', type=Path); ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--ldr', action='store_true', help="NGX's *_ldr_* stages, for games without the HDR flag (capture such a frame)")
    ap.add_argument('--hip', type=Path, metavar='KERNEL_DIR', help='also package the network for the native HIP layers: the layer '
                    'weights as NGX uploads them (hipnet.bin) and the rrlite_*_4x4.hsaco kernels of KERNEL_DIR (kernels/build.sh output)')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    def one(kernel):
        d = sorted(glob.glob(f'{a.capture}/replay/replay-*-{kernel}'))
        if len(d) <= a.frame: raise ValueError(f'no launch of {kernel} for frame {a.frame}')
        return d[a.frame]
    R = 'ldr' if a.ldr else 'hdr'
    w8, w16 = [], []
    n8 = n16 = 0
    def add8(x):
        nonlocal n8
        x = np.ascontiguousarray(x, np.uint8).ravel(); pad = (-x.size) % 64
        off = n8; w8.append(x); w8.append(np.zeros(pad + 64, np.uint8)); n8 += x.size + pad + 64
        return off
    def add16(x):
        nonlocal n16
        x = np.ascontiguousarray(x).view(np.uint8).ravel(); pad = (-x.size) % 64
        off = n16; w16.append(x); w16.append(np.zeros(pad + 64, np.uint8)); n16 += x.size + pad + 64
        return off
    table = []      # (w8 offset, w16 offset) per stage, then the layers' offset blocks
    # input stage: embedding [128][64] in natural output order, bias tile, dither table
    wb = blob_of(one(f'rrlite_enc0_4x4_mvlo_{R}_folded'))
    g = sm.gp(np.arange(64))
    enc0 = (add8(wb[sm.woff_table(4096, 512, 2048, 128, 64)][:, g]), add16(np.tile(wb[12288:12288 + 128].view(np.float16)[g][None, :], (16, 1))), add16(wb[:4096]))
    # expansion: the 20 columns per sub-position the stage uses
    wb = blob_of(one('rrlite_dec0_4x4_folded'))
    cols = (np.arange(4)[:, None] * 32 + np.arange(20)[None, :]).ravel()
    dec0 = (add8(wb[sm.woff_table(0, 512, 1024, 64, 128)][:, cols]), add16(np.tile(wb[8192:8192 + 256].view(np.float16)[cols][None, :], (16, 1))))
    Ps, dirs = mm.launches(a.capture, a.frame)
    layers = []
    for P, name in zip(Ps, mm.ORDER):
        o, b8, b16 = pack_layer(mm.plain_layer(P, name), name)
        layers.append((add8(b8), add16(b16), o))
    blob8, blob16 = np.concatenate(w8), np.concatenate(w16)
    blob8.tofile(a.output / 'weights8.bin'); blob16.tofile(a.output / 'weights16.bin')
    head = struct.pack('<8sII', b'D4RM0003', blob8.size, blob16.size) + struct.pack('<5I', *enc0, *dec0)
    for o8, o16, o in layers: head += struct.pack('<2I72I', o8, o16, *o)
    (a.output / 'offsets.bin').write_bytes(head)
    def glsl(src, out, flags=()):
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *flags, str(HERE / src), '-o', str(a.output / out)], check=True, stdout=subprocess.DEVNULL)
    for name, out in SHADERS.items(): glsl('layer_m.comp', out + '.spv', defines(name))
    for s in ('enc0_m', 'dec0_m', 'post_m', 'down_m'): glsl(s + '.comp', s + '.spv', ['-DLDR=1'] if a.ldr and s != 'dec0_m' else [])
    glsl('down_m.comp', 'down_m_rgb10.spv', ['-DOUTPUT_RGB10=1'] + (['-DLDR=1'] if a.ldr else []))
    glsl('post_m.comp', 'post_m_q.spv', ['-DTILE_W=15', '-DTILE_H=15'] + (['-DLDR=1'] if a.ldr else []))
    glsl('post_m.comp', 'post_m_c.spv', ['-DTILE_W=11', '-DTILE_H=11'] + (['-DLDR=1'] if a.ldr else []))
    # the same stages with the token buffers in NGX's plane layout, for a network that runs outside the engine
    for s in ('enc0_m', 'dec0_m'): glsl(s + '.comp', s + '_planes.spv', ['-DPLANES=1'] + (['-DLDR=1'] if a.ldr and s != 'dec0_m' else []))
    if a.hip:
        # hipnet.bin: "D4RMHIP1", layer count, then per layer {kernel name[32], shift x, y, token grid divisor, weight
        # offset, weight bytes} and the weight allocations as NGX uploads them (the native kernels read that layout)
        scale = {'enc1': 1, 'enc2': 2, 'enc3_tube': 4, 'dec2': 2, 'dec1': 1}
        blobs, head, at = [], struct.pack('<8sI4x', b'D4RMHIP1', len(Ps)), 16 + 64 * len(Ps)
        for P, name in zip(Ps, mm.ORDER):
            w = np.ascontiguousarray(P.wbuf); pad = (-w.size) % 64
            head += struct.pack('<32siiIIQQ', f'rrlite_{name}_4x4'.encode(), P.sx, P.sy, scale[name], 0, at, w.size)
            blobs += [w, np.zeros(pad, np.uint8)]; at += w.size + pad
        (a.output / 'hipnet.bin').write_bytes(head + np.concatenate(blobs).tobytes())
        (a.output / 'hip').mkdir(exist_ok=True)
        for name in sorted(set(mm.ORDER)):
            src = a.hip / f'rrlite_{name}_4x4.hsaco'
            if not src.exists(): raise ValueError(f'{src} is missing')
            (a.output / 'hip' / src.name).write_bytes(src.read_bytes())
    marker = a.output / 'ldr'      # tells the runtime to skip exposure
    if a.ldr: marker.write_text('1\n')
    elif marker.exists(): marker.unlink()
    for p in '01': glsl('exposure_k.comp', f'exposure_m{p}.spv', ['-DMODE=1', '-DPASS=' + p])
    files = sorted(a.output.glob('*.bin')) + sorted(a.output.glob('*.spv')) + sorted(a.output.glob('hip/*.hsaco'))
    (a.output / 'model.json').write_text(json.dumps({'format': 'D4RM0003', 'preset': 'M', 'files': {str(f.relative_to(a.output)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}, indent=1))
    print(f'Compiled native preset M model in {a.output} ({blob8.size} FP8 bytes, {blob16.size} bytes of f16 data)')


if __name__ == '__main__':
    main()
