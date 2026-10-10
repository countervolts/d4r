#!/usr/bin/env python3
"""Package extracted K weights and compile d4r's shaders. No PTX translator is used.

compile_k.py MODEL_PREFIX LUT.bin OUTPUT_DIRECTORY
LUT.bin holds NGX's sixteen anisotropic reconstruction tables (2048 sixteen-byte entries each): the whole
allocation behind the output launch's argument 280, as prepare_k.py writes it. NGX picks one by upscaling
ratio. A single 32768-byte table (older captures) is accepted and then serves every ratio.

--hip KERNEL_DIR --capture CAPTURE_DIR also packages the network for the native
HIP layer kernels (engine/hip_net.h): hipnet.bin ("D4RKHIP1", the eleven layers in
network order with the weight allocations exactly as NGX uploads them, which the
native kernels read) and hip/dltss_pwin_*_layer.hsaco copied from KERNEL_DIR.
CAPTURE_DIR/replay must hold one captured launch of each dltss_pwin_*_layer with
its weight allocation (D4R_CUDA_REPLAY_DUMP_DIR, filter dltss_pwin). HDR and LDR
share the weights, so one hipnet.bin serves both variants.
"""
# D4RK0002 couples the package to the exposure stage's sample grid: the engine passes automatic exposure the
# native output/2 source grid (mandatory dispatch size and uniform sampleSize), and the K1 package's exposure
# shader and dispatch assume the render grid, so a K1 offsets blob is not interchangeable with this runtime.
# Nothing else in the package changed meaning: the tensor offsets and the layer/input/output stages are the same.
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from extract_k import LAYERS
from test_k import defines, params

HERE = Path(__file__).resolve().parent
MAGIC = b'D4RK0002'
MANIFEST_VERSION = 2

# Native HIP packaging (hipnet.bin). The layer order is the network order of extract_k.LAYERS; NGX uploads every
# layer's weights back to back into one allocation in the layout the native kernels read
# (kernels/tools/pwin_model.Layout), so each blob is the launch's PwinParams.w up to the next layer's weights
# pointer. `divisor` is the token-grid divisor of makeKShape (2**shift); the sx/sy written per entry are the
# frame-0 window shifts (the backend recomputes the whole eight-frame cycle itself).
HIP_ORDER = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'dec5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0']
HIP_KERNEL = 'dltss_pwin_{}_layer'
HIP_DIVISOR = {'enc0': 1, 'enc1': 2, 'enc2': 4, 'enc3': 8, 'enc4': 16, 'dec5': 32,
               'dec4': 16, 'dec3': 8, 'dec2': 4, 'dec1': 2, 'dec0': 1}
# makeKWindows (engine/runtime.cpp): an eight-frame cycle of per-axis shifts; the five deepest layers move and
# every shallower one follows from the one below as (2 * shift + 4) mod 8. depth 0 = dec5, 1 = enc4/dec4,
# 2 = enc3/dec3; the rest keep shift 4.
HIP_CYCLE = ((0, 2), (5, 6), (4, 0), (1, 4), (7, 5), (2, 1), (3, 7), (6, 3))
HIP_ROLE = {'dec5': 0, 'enc4': 1, 'dec4': 1, 'enc3': 2, 'dec3': 2}


def hip_shift(role, frame):
    x, y = HIP_CYCLE[frame % 8]
    for _ in range(role):
        x, y = (2 * x + 4) % 8, (2 * y + 4) % 8
    return x, y


def hip_weight_place(launch, offset=64):
    """The weight allocation a launch's PwinParams.w (argument at `offset`) points into: (base, size, path, ptr)."""
    ptr = struct.unpack_from('<Q', open(f'{launch}/args.bin', 'rb').read(), offset)[0]
    if ptr == 0:
        raise ValueError(f'{launch}: no weight pointer')
    for line in open(f'{launch}/manifest.txt'):
        q = line.split()
        if q and q[0] == 'alloc':
            path = f'{launch}/alloc-{q[1]}.bin'; base = int(q[2], 16); size = os.path.getsize(path)
            if base <= ptr < base + size:
                return (base, size, path, ptr)
    raise ValueError(f'{launch}: the weights allocation is missing')


def package_hip(output, kernels, capture):
    """Writes hipnet.bin + hip/*.hsaco. The kernels' PwinParams.w is the launch's argument at byte 64.
    NGX packs the eleven layers back to back in one allocation, so a layer's blob is its region up to the
    next layer's weights pointer (the last one to the end of the allocation)."""
    place = {}
    for name in HIP_ORDER:
        found = sorted(glob.glob(f'{capture}/replay/replay-*-{HIP_KERNEL.format(name)}'))
        if not found:
            raise ValueError(f'{capture}: no captured launch of {HIP_KERNEL.format(name)}')
        place[name] = hip_weight_place(found[0])
    # The per-layer region rule below needs the layers to be packed in network order inside one allocation;
    # NGX does that (and packs it identically for HDR and LDR), so a different order must be noticed here.
    offsets = [(place[name][3] - place[name][0]) for name in HIP_ORDER]
    if offsets != sorted(offsets) or len(set(offsets)) != len(offsets) or offsets[0] != 0:
        raise ValueError(f'{capture}: the layers are not packed in network order: {offsets}')
    blobs, at = [], 16 + 64 * len(HIP_ORDER)
    head = struct.pack('<8sI4x', b'D4RKHIP1', len(HIP_ORDER))
    for name in HIP_ORDER:
        base, size, path, ptr = place[name]
        following = [p for b, _, _, p in place.values() if b == base and p > ptr]
        end = min(following) if following else base + size
        with open(path, 'rb') as f:
            f.seek(ptr - base)
            w = bytearray(f.read(end - ptr))
        pad = (-len(w)) % 64
        role = HIP_ROLE.get(name)
        sx, sy = hip_shift(role, 0) if role is not None else (4, 4)
        head += struct.pack('<32siiIIQQ', HIP_KERNEL.format(name).encode(), sx, sy, HIP_DIVISOR[name], 0, at, len(w))
        blobs += [w, bytes(pad)]; at += len(w) + pad
    (output / 'hipnet.bin').write_bytes(head + b''.join(blobs))
    (output / 'hip').mkdir(exist_ok=True)
    for name in HIP_ORDER:
        src = kernels / f'{HIP_KERNEL.format(name)}.hsaco'
        if not src.exists():
            raise ValueError(f'{src} is missing')
        (output / 'hip' / src.name).write_bytes(src.read_bytes())


def tensor_spans(e):
    c, h, x = e['C'], e['H'], e['X']
    for key in ('g1', 'bo', 'g2', 'b2'):
        yield e[key], 16 * c
    for key, size in (('qkv', c * 32), ('table', 64 * 64), ('wo', 32 * c),
                      ('w1', c * 32), ('b1', 16 * 32), ('w2', 32 * c)):
        expected = {'qkv': h * 3, 'table': h, 'wo': h, 'w1': c // 8, 'b1': c // 8, 'w2': c // 8}[key]
        if len(e[key]) != expected:
            raise ValueError(f'{key}: expected {expected} tensors')
        for offset in e[key]:
            yield offset, size
    for key, size in {'embed_w': 16 * c, 'embed_b': 16 * c,
                      'merge_w': 4 * c * e.get('merge_n', 0), 'merge_b': 16 * e.get('merge_n', 0),
                      'expand_w': x * 4 * c, 'expand_b': 16 * 4 * c,
                      'head_w': c * 48, 'head_b': 16 * 48}.items():
        if key in e:
            yield e[key], size


def validate(index, weight_bytes):
    if list(index) != list(LAYERS):
        raise ValueError('expected the eleven preset K layers in network order')
    if weight_bytes <= 0 or weight_bytes % 2:
        raise ValueError('weights must be nonempty f16 data')
    for name, (kind, heads, c, x, pos) in LAYERS.items():
        e = index[name]
        if (e['kind'], e['H'], e['C'], e['X'], e['posattn']) != (kind, heads, c, x, pos):
            raise ValueError(f'{name}: incompatible layer shape')
        required = ('embed_w', 'embed_b') if kind == 'enc0' else ()
        required += ('merge_w', 'merge_b', 'merge_n') if kind in ('enc', 'enc0') else ()
        required += ('expand_w', 'expand_b') if kind in ('dec', 'dec0') else ()
        required += ('head_w', 'head_b') if kind == 'dec0' else ()
        for key in required:
            if key not in e:
                raise ValueError(f'{name}: missing {key}')
        if kind in ('enc', 'enc0') and e['merge_n'] != (x + 15) // 16 * 16:
            raise ValueError(f'{name}: invalid merge stride')
        for offset, size in tensor_spans(e):
            if not isinstance(offset, int) or offset < 0 or offset % 16 or (offset + size) * 2 > weight_bytes:
                raise ValueError(f'{name}: tensor at {offset} with {size} elements is outside/alignment-invalid')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_prefix', type=Path)
    p.add_argument('lut', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--ldr', action='store_true', help='display-referred NGX LDR input and reconstruction')
    p.add_argument('--hip', type=Path, metavar='KERNEL_DIR', help='also package the network for the native HIP layers: '
                   'hipnet.bin (the eleven layers and their weight allocations as NGX uploads them) and '
                   'hip/dltss_pwin_*_layer.hsaco from KERNEL_DIR (kernels/build.sh output)')
    p.add_argument('--capture', type=Path, metavar='CAPTURE_DIR', help='with --hip: the dump directory holding one '
                   'captured frame of the eleven dltss_pwin_*_layer launches (D4R_CUDA_REPLAY_DUMP_DIR)')
    a = p.parse_args()
    if a.hip and a.capture is None:
        p.error('--hip needs --capture (the weights come from a captured frame)')
    weights = Path(str(a.model_prefix) + '.bin')
    index = json.loads(Path(str(a.model_prefix) + '.json').read_text())
    validate(index, weights.stat().st_size)
    # All sixteen tables (prepare_k.py's lut.bin; NGX picks one by upscaling ratio) or, from older captures,
    # the single table of the captured ratio, which the engine then uses for every ratio.
    if a.lut.stat().st_size not in (32768, 16 * 32768):
        raise ValueError('the K reconstruction tables must be one or sixteen tables of 2048 sixteen-byte entries')
    if a.lut.stat().st_size == 32768:
        print('warning: one reconstruction table only; output at other upscaling ratios will not match NGX', file=sys.stderr)
    a.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights, a.output / 'weights.bin')
    shutil.copyfile(a.lut, a.output / 'lut.bin')
    # Only offsets; dimensions and window shifts belong to the Engine, not the Model.
    class ZeroShape:
        def i32x2(self, offset):
            return (0, 0)
    offsets = b''.join(params(ZeroShape(), e)[0][16:] for e in index.values())
    (a.output / 'offsets.bin').write_bytes(MAGIC + struct.pack('<II', 11, weights.stat().st_size) + offsets)
    for name, e in index.items():
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defines(e),
                        str(HERE / 'layer.comp'), '-o', str(a.output / (name + '.spv'))], check=True)
    for name in ('input_k', 'output_k'):
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *(['-DLDR=1'] if a.ldr else []),
                        str(HERE / (name + '.comp')), '-o', str(a.output / (name + '.spv'))], check=True)
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '-DOUTPUT_RGB10=1',
                    *(['-DLDR=1'] if a.ldr else []), str(HERE / 'output_k.comp'),
                    '-o', str(a.output / 'output_k_rgb10.spv')], check=True)
    for stage in '01':
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '-DPASS=' + stage, str(HERE / 'exposure_k.comp'),
                        '-o', str(a.output / f'exposure_k{stage}.spv')], check=True)
    # Shader revision marker. Packages without it double-added the motion origin (the adapter uses copies for
    # their nonzero rectangles); revision 1 shaders ignore the frame flags, so they cannot take regular depth.
    # Revision 2 reads the frame flags; revision 3 also reads the display-motion flag and the game's exposure
    # texture (exposure flag 4).
    (a.output / 'direct_origins.bin').write_bytes(b'D4RO0003')
    if a.hip:
        package_hip(a.output, a.hip, a.capture)
    files = sorted(a.output.glob('*.bin')) + sorted(a.output.glob('*.spv'))
    if a.hip:
        files += sorted(a.output.glob('hip/*.hsaco'))
    manifest = {'format': 'd4r-k', 'version': MANIFEST_VERSION, 'layers': list(index),
                'variant': 'ldr' if a.ldr else 'hdr',
                'numerics': 'folded-gains' if all(e.get('fold_gains', True) for e in index.values()) else 'explicit-f16-gains',
                'sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (a.output / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Compiled native preset K model in {a.output}' + (f' with native HIP kernels from {a.hip}' if a.hip else ''))


if __name__ == '__main__':
    main()
