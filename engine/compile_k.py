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
import numpy as np

from extract_k import LAYERS
from test_k import defines, params
import hlsl_build

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
# makeKWindows (engine/k_frame.cpp): an eight-frame cycle of per-axis shifts; the five deepest layers move and
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


def package_hip(output, kernels, capture, accuracy=None):
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
    # hip/: the layers vkhip runs by default; hip-accuracy/ (optional): the accuracy set, taken with
    # [Kernels] PreferAccuracy. Both read the same hipnet.bin.
    for folder, source in (('hip', kernels), ('hip-accuracy', accuracy)):
        if source is None:
            continue
        (output / folder).mkdir(exist_ok=True)
        for name in HIP_ORDER:
            src = source / f'{HIP_KERNEL.format(name)}.hsaco'
            if not src.exists():
                raise ValueError(f'{src} is missing')
            (output / folder / src.name).write_bytes(src.read_bytes())


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


def d3d12_weight_tiles(index, weights):
    """Transpose individual 16x16 B-operand tiles, preserving binary16 bits and
    all offsets/strides. Nonmatrix data and the canonical Vulkan file are untouched."""
    source = np.frombuffer(weights, dtype='<u2')
    packed = source.copy()
    matrices = {}
    unpacked_offsets = set()
    for e in index.values():
        for key, value in e.items():
            if key in ('g1', 'bo', 'g2', 'b2', 'table', 'b1', 'embed_w', 'embed_b',
                       'merge_w', 'merge_b', 'expand_w', 'expand_b', 'head_w', 'head_b'):
                unpacked_offsets.update(value if isinstance(value, list) else [value])
        for key in ('qkv', 'wo', 'w1', 'w2'):
            k, n = (e['C'], 32) if key in ('qkv', 'w1') else (32, e['C'])
            for offset in e[key]:
                if offset in matrices and matrices[offset] != (k, n):
                    raise ValueError('incompatible aliased D3D12 weight matrices')
                matrices[offset] = (k, n)
    # A transposed tile cannot share storage with a tensor read in its original
    # layout. Exact aliases of the same matrix are supported.
    spans = sorted(set(span for e in index.values() for span in tensor_spans(e)))
    for offset, (k, n) in matrices.items():
        end = offset + k * n
        if offset in unpacked_offsets:
            raise ValueError('D3D12 weight matrix aliases a tensor that must remain row-major')
        if offset < 0 or end > source.size or k % 16 or n % 16:
            raise ValueError('invalid D3D12 weight matrix span')
        if any(other < end and offset < other + size and (other, size) != (offset, k * n)
               for other, size in spans):
            raise ValueError('overlapping D3D12 weight tensors cannot be tile-transposed')
        packed[offset:end] = source[offset:end].reshape(k // 16, 16, n // 16, 16).transpose(0, 3, 2, 1).ravel()
    return packed.tobytes()


def build_dxil(index, output, ldr, target='windows'):
    """Package K's native D3D12 kernels alongside Vulkan's SPIR-V. D3D12 uses
    native half rounding, f16 WMMA accumulation, prefetched column-major weight
    tiles and root CBVs. Network waves are wave32 (one token tile on Windows);
    output waves remain wave64. These backend choices do not change Vulkan's
    kernels or canonical weights.

    Both targets filter every subpixel as Vulkan does. The centre Gaussian
    estimate the 'windows' target used to share between the four subpixels of
    an output pixel (D4R_GAUSSIAN_CENTER, still in output_k.comp) misplaces
    detail near the render pixel pitch (harness blinds; 17-19 dB against the
    ZLUDA reference on reset frames, 48-51 dB without it) and loses 9-14 dB on
    HDR highlights; exact filtering costs 615 against 588 us a frame at 720p
    (RX 9070 XT, 2026-10-23).
    target 'windows' is AMD's D3D12 driver. target 'vkd3d' (vkd3d-proton with
    patches/vkd3d-proton, on RADV) builds output_k as Shader Model 6.9,
    whose DXIL keeps f16 vector arithmetic that
    RADV packs (output_k 2577 -> 2178 instructions, 95 -> 71 VGPRs, with
    patch 0007's fused precise mad), adds AGS element-wise accumulator sums
    (dxil-spirv translates no register arithmetic on matrices; patch 0006
    covers the rest) and runs dec0 with one token tile per wave (329 -> 307
    us at 4K, bit-identical). RX 9070 XT, 250 W, five interleaved trials,
    2026-10-23: 2880 us at 4K and 1552 us at 1440p, against Vulkan's 2867 and
    1528 us. The input now uses four threads per token in wave64, compiled as
    SM6.6 (DXC fails on the SM6.9 version of that schedule); reconstruction
    uses wave shuffles for its highlight quad on RADV. RDNA4 measurements and
    the pending RDNA3 qualification are recorded in linux-k-stage-tuning.md."""
    if target not in ('windows', 'vkd3d'):
        raise ValueError('D3D12 target must be windows or vkd3d')
    vkd3d = target == 'vkd3d'
    rounding = ['-DD4R_NATIVE_HALF=1']
    ldr_define = (['-DLDR=1'] if ldr else []) + rounding
    stage_profile = 'cs_6_9' if vkd3d else 'cs_6_6'
    # AMD's Cyberpunk application profile changes native half accumulation.
    # Keep Gaussian sums in f32 on Windows and round only the filter result;
    # vkd3d/RADV keeps its existing native-half implementation.
    reconstruction = [] if vkd3d else ['-DD4R_FILTER_ACCUM_F32=1']
    for name, e in index.items():
        tpw = 2 if vkd3d and e['C'] == 32 and e['kind'] != 'dec0' else 1
        extra = ['-DCOLUMN_MAJOR_ENC0=0', f'-DTPW={tpw}'] + (['-DPADX=0', '-DPAD4=0'] if e['C'] == 128 else [])
        extra += ['-DD4R_LAYER_CBV=1', '-DD4R_WEIGHT_TILES_COLMAJOR=1']
        extra += rounding + ['-DD4R_ACCUMULATE_F16=1',
                             f'-DPREFETCH_K={e["C"] // 16}', f'-DPREFETCH_N={e["C"] // 16}']
        extra += ['-DD4R_AGS_ELEMENTWISE_ADD=1'] if vkd3d else []
        extra += os.environ.get('K_DEFS_' + name.upper(), '').split()
        hlsl_build.build(HERE / 'layer.comp', output / (name + '.dxil'), defines(e, extra), wave=32, unroll=True)
    for name in ('input_k', 'output_k'):
        output_schedule = ((reconstruction + (['-DD4R_OUTPUT_WAVE_BRIGHT=1'] if vkd3d else []))
                           if name == 'output_k' else ['-DD4R_INPUT_SUBBLOCK=1'])
        # Four lanes share each token's input feature reduction. On RADV the
        # wave64 SM6.6 build avoids DXC's SM6.9 failure for this schedule and
        # halves input-stage time on GFX1201. Reconstruction keeps SM6.9.
        hlsl_build.build(HERE / (name + '.comp'), output / (name + '.dxil'), ldr_define + output_schedule,
                         wave=64 if vkd3d or name == 'output_k' else 32,
                         profile='cs_6_6' if name == 'input_k' else stage_profile)
    hlsl_build.build(HERE / 'output_k.comp', output / 'output_k_rgb10.dxil', ['-DOUTPUT_RGB10=1'] + ldr_define +
                     reconstruction + (['-DD4R_OUTPUT_WAVE_BRIGHT=1'] if vkd3d else []), profile=stage_profile)
    for stage in '01':
        hlsl_build.build(HERE / 'exposure_k.comp', output / f'exposure_k{stage}.dxil', ['-DPASS=' + stage] + rounding)
    marker = output / 'd3d12_layer_cbv.bin'
    packed_weights = output / 'weights_d3d12.bin'
    packed_weights.write_bytes(b'D4RWTB01' + d3d12_weight_tiles(index, (output / 'weights.bin').read_bytes()))
    marker.write_bytes(b'D4RCBV02')


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
    p.add_argument('--dxil', choices=('auto', 'yes', 'no'), default='auto', help='also build the D3D12 backend DXIL '
                   '(hlsl_build.py, needs dxc): auto builds it on Windows when dxc is found')
    p.add_argument('--d3d12-target', choices=('windows', 'vkd3d'), default='windows', help='the D3D12 implementation '
                   'the DXIL is built for: AMD\'s Windows driver, or vkd3d-proton with patches/vkd3d-proton '
                   '(exact per-subpixel reconstruction, SM6.6 input / SM6.9 reconstruction; see build_dxil)')
    p.add_argument('--hip-accuracy', type=Path, metavar='KERNEL_DIR', help='with --hip: also package the accuracy '
                   'layers as hip-accuracy/ and the accuracy reconstruction shaders '
                   '(selected with [Kernels] PreferAccuracy)')
    a = p.parse_args()
    dxil = a.dxil == 'yes'
    if a.dxil == 'auto' and os.name == 'nt':
        try:
            hlsl_build.dxc_path()
            dxil = True
        except FileNotFoundError:
            print('note: dxc not found; the package has no D3D12 shaders', file=sys.stderr)
    if a.hip and a.capture is None:
        p.error('--hip needs --capture (the weights come from a captured frame)')
    if a.hip_accuracy and not a.hip:
        p.error('--hip-accuracy needs --hip')
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
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defines(e, os.environ.get('K_DEFS_' + name.upper(), '').split()),   # per layer, e.g. K_DEFS_DEC0=-DTPW=2
                        str(HERE / 'layer.comp'), '-o', str(a.output / (name + '.spv'))], check=True)
        # the same layer without cooperative matrices (RDNA2: v_dot2_f32_f16); the runtime interleaves the weights.
        # The deep layers (C >= 96) have few windows: up to eight waves per window instead of one per head
        # (Steam Deck, 1920x1080 output: enc3 485 -> 435, enc4 306 -> 259, dec5 182 -> 94, dec4 257 -> 215,
        # dec3 436 -> 383 us). K_DOT2_DEFS / K_DOT2_DEFS_<LAYER> add or override defines.
        if e.get('fold_gains', True):
            nw = [f"-DNW={min(2 * e['H'], 8)}"] if e['C'] >= 96 else []
            # patch merge: the most merged tokens per lane (reuse of each weight load) that keeps 60% of the lanes busy
            # (Steam Deck, 640x360 -> 1280x720: enc2 392 -> 369, enc3 352 -> 340, enc4 278 -> 270 us)
            if e['kind'] in ('enc', 'enc0'):
                waves, pairs = (min(2 * e['H'], 8) if e['C'] >= 96 else e['H']), e['X'] // 2
                def busy(mmt):
                    step = 32 * waves * mmt // 16
                    return pairs / (-(-pairs // step) * step)
                nw.append(f"-DMMT={max(m for m in (2, 4, 8, 16) if waves * m % 16 == 0 and busy(m) >= 0.6)}")
            extra = os.environ.get('K_DOT2_DEFS', '').split() + os.environ.get('K_DOT2_DEFS_' + name.upper(), '').split()
            given = {d.split('=')[0] for d in extra}
            nw = [d for d in nw if d.split('=')[0] not in given]
            subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defines(e, nw + extra),
                            str(HERE / 'layer_dot2.comp'), '-o', str(a.output / (name + '_dot2.spv'))], check=True)
    for name in ('input_k', 'output_k'):
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *(['-DLDR=1'] if a.ldr else []),
                        str(HERE / (name + '.comp')), '-o', str(a.output / (name + '.spv'))], check=True)
    # final stores straight into the game's packed output textures (output_store.glsl); RGBA16F is output_k.spv
    # (output_k_rgb10.spv keeps its name; the newer stores are output_k_store_*.spv, so a stale output_k_rgba8.spv
    # left in a model folder by older experimental builds is never picked up)
    for define, store in (('OUTPUT_RGB10', 'rgb10'), ('OUTPUT_RGBA8', 'store_rgba8'), ('OUTPUT_R11G11B10', 'store_r11g11b10')):
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', f'-D{define}=1',
                        *(['-DLDR=1'] if a.ldr else []), str(HERE / 'output_k.comp'),
                        '-o', str(a.output / f'output_k_{store}.spv')], check=True)
    if a.hip_accuracy:
        for define, store in ((None, ''), ('OUTPUT_RGB10', '_rgb10'), ('OUTPUT_RGBA8', '_store_rgba8'),
                              ('OUTPUT_R11G11B10', '_store_r11g11b10')):
            subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '-DD4R_ACCURACY=1',
                            *([f'-D{define}=1'] if define else []), *(['-DLDR=1'] if a.ldr else []),
                            str(HERE / 'output_k.comp'), '-o', str(a.output / f'output_k{store}_accuracy.spv')], check=True)
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '-DD4R_ACCURACY=1', '-DPASS=1',
                        str(HERE / 'exposure_k.comp'), '-o', str(a.output / 'exposure_k1_accuracy.spv')], check=True)
    for stage in '01':
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '-DPASS=' + stage, str(HERE / 'exposure_k.comp'),
                        '-o', str(a.output / f'exposure_k{stage}.spv')], check=True)
    # Shader revision marker. Packages without it double-added the motion origin (the adapter uses copies for
    # their nonzero rectangles); revision 1 shaders ignore the frame flags, so they cannot take regular depth.
    # Revision 2 reads the frame flags; revision 3 also reads the display-motion flag and the game's exposure
    # texture (exposure flag 4).
    (a.output / 'direct_origins.bin').write_bytes(b'D4RO0003')
    if dxil:
        build_dxil(index, a.output, a.ldr, a.d3d12_target)
    if a.hip:
        package_hip(a.output, a.hip, a.capture, a.hip_accuracy)
    files = sorted(a.output.glob('*.bin')) + sorted(a.output.glob('*.spv')) + sorted(a.output.glob('*.dxil'))
    if a.hip:
        files += sorted(a.output.glob('hip/*.hsaco'))
    accuracy_files = sorted(a.output.glob('hip-accuracy/*.hsaco')) if a.hip_accuracy else []
    manifest = {'format': 'd4r-k', 'version': MANIFEST_VERSION, 'layers': list(index),
                'variant': 'ldr' if a.ldr else 'hdr',
                'numerics': 'folded-gains' if all(e.get('fold_gains', True) for e in index.values()) else 'explicit-f16-gains',
                'sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    if dxil:
        manifest['d3d12_target'] = a.d3d12_target
        manifest['d3d12_input_schedule'] = 'four-threads-per-token'
        manifest['d3d12_input_wave_size'] = 64 if a.d3d12_target == 'vkd3d' else 32
        manifest['d3d12_input_shader_model'] = '6.6'
        manifest['d3d12_highlight_quad'] = 'wave-shuffle' if a.d3d12_target == 'vkd3d' else 'shared-memory'
        manifest['d3d12_reconstruction'] = 'exact'
        manifest['d3d12_gaussian_accumulation'] = ('native-f16' if a.d3d12_target == 'vkd3d'
                                                 else 'f32-sums-f16-result')
        manifest['d3d12_weight_layout'] = 'tile16-column-major'
    if accuracy_files:  # the same names as hip/: keyed by their path
        manifest['sha256'].update({f'hip-accuracy/{f.name}': hashlib.sha256(f.read_bytes()).hexdigest() for f in accuracy_files})
    (a.output / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Compiled native preset K model in {a.output}' + (f' with native HIP kernels from {a.hip}' if a.hip else ''))


if __name__ == '__main__':
    main()
