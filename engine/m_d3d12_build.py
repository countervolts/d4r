#!/usr/bin/env python3
"""Add native D3D12 M/L shaders to a current compile_m.py model package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import hlsl_build
import m_hlsl_build
from test_m import dot2_defines, dot2_merge_defines

HERE = Path(__file__).resolve().parent
NAMES = {'enc1': 'm_enc1', 'enc2': 'm_enc2', 'enc3_tube': 'm_tube', 'dec2': 'm_dec2', 'dec1': 'm_dec1'}


def build(directory, math='fast', dxc=None, glslang=None, layer_params='cbv', expand_mode='split', vector_io=True, stream_mlp=True):
    directory = Path(directory)
    magic = (directory / 'offsets.bin').read_bytes()[:8]
    if magic not in (b'D4RM0007', b'D4RL0002'):
        raise ValueError('native D3D12 requires a D4RM0007 or D4RL0002 model')
    if math not in ('fast', 'fp8'):
        raise ValueError('D3D12 M/L math must be fast or fp8')
    stream_mlp = stream_mlp and math == 'fast'
    if layer_params not in ('srv', 'cbv'): raise ValueError('layer_params must be srv or cbv')
    if expand_mode not in ('fused', 'split'): raise ValueError('expand_mode must be fused or split')
    split = expand_mode == 'split'
    cbv = layer_params == 'cbv'
    if split and not cbv: raise ValueError('split expansion requires cbv parameters')
    marker = b'D4RML004' if split else b'D4RML003' if cbv else b'D4RML002'
    preset_l = magic == b'D4RL0002'
    input_wmma = split and not preset_l
    if input_wmma: marker = b'D4RML005'
    vector_io = vector_io and split
    if vector_io: marker = b'D4RML006'
    # M's expansion (dec0_m) with the binary16 product and NGX's rounding after every 32 products (DOT2 without
    # INT8): the int8 form re-quantizes the tokens (one scale each) and the matrix (column scales), a narrower format
    # than NVIDIA's FP8 (docs/approximations.md on the linux branch). D4RML007: runtimes that predate it refuse the
    # package instead of handing the shader int8 weights.
    # D4RML007 (M and L): the exact RDNA 3 package: binary16 conversions to nearest even on AGS (every pipeline then
    # binds the AGS intrinsic UAV), and M's expansion on f16 WMMA
    exact = vector_io
    f16_expand = exact and not preset_l
    if exact: marker = b'D4RML007'
    ldr = (directory / 'ldr').exists()
    compiler = hlsl_build.dxc_path(dxc)
    # The FP8 WMMA backend (compile_m.py build_dxil) keeps its shaders of the same names beside the weights.
    target = directory / 'rdna3'
    target.mkdir(exist_ok=True)
    # binary16 conversions: AGS's, to nearest even (d3d12/rne16_ags.hlsli), in the exact package; D4R_M_RNE=emulate
    # the integer emulation (dialect_hlsl.h D4R_RNE16_EXACT, for comparisons; wrong in L's enc0_l_network and
    # dec0_l for a reason not found), D4R_M_RNE=native or an older package the driver's own (toward zero)
    mode = os.environ.get('D4R_M_RNE', 'ags' if vector_io else 'native')
    if mode not in ('ags', 'emulate', 'native'): raise ValueError('D4R_M_RNE must be ags, emulate or native')
    if mode != 'native' and not vector_io: raise ValueError('nearest-even conversions need the D4RML007 package')
    native = mode == 'native'
    native_half = {'ags': ['-DD4R_RNE16_EXTERNAL=1', '-DD4R_RNE16_AGS_GFX11=1'], 'emulate': ['-DD4R_RNE16_EXACT=1'],
                   'native': ['-DD4R_NATIVE_HALF=1']}[mode]
    matrix_rounding = native_half
    with tempfile.TemporaryDirectory(prefix='.d3d12-ml-', dir=directory) as tmp:
        out = Path(tmp)
        def matrix(source, output, defines):
            # binary16 conversions to nearest even, as NVIDIA's (dialect_hlsl.h; the driver's native one rounds toward
            # zero): every RDNA 3 shader of the package (D4R_M_NATIVE_HALF=1 restores the native conversion)
            command = [compiler, '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types', '-O3',
                       *matrix_rounding, '-DD4R_M_PAIR_GEMM=1', '-DD4R_M_DIRECT_V=1', f'-DD4R_M_LAYER_CBV={int(cbv)}', *defines, str(HERE / 'd3d12' / source), '-Fo', str(out / (output + '.dxil'))]
            subprocess.run(command, check=True)
        flags = ['-DD4R_M_NO_Q8=1', '-DD4R_M_ACC_ROUND_K32=0', '-DD4R_M_DIRECT_MLP=1'] if math == 'fast' else []
        for name, output in NAMES.items():
            definitions = dot2_defines(name)
            if split and name in ('dec2', 'dec1'):
                matrix('expand_m_wmma.hlsl', 'm_patch_' + name, definitions + flags)
                definitions = [d for d in definitions if not d.startswith('-DKIND=')] + ['-DKIND=0', '-DD4R_M_INPUT_F16=1']
            matrix('layer_m_wmma.hlsl', output, definitions + flags + [f'-DD4R_M_VECTOR_IO={int(vector_io)}', f'-DD4R_M_STREAM_MLP={int(stream_mlp)}'])
        for i, name in enumerate(('enc1', 'enc2')):
            matrix('merge_m_wmma.hlsl', 'm_merge' + str(i + 1), dot2_merge_defines(name))
        def stage(source, output, defines=(), wave=32):
            m_hlsl_build.build(HERE / (source + '.comp'), out / (output + '.dxil'), list(defines), wave=wave, dxc=dxc, glslang=glslang,
                               rounding=native_half)
        light = ['-DLDR=1'] if ldr else []
        if preset_l:
            stage('enc0_l_features', 'enc0_l', light)
            matrix('layer_m_wmma.hlsl', 'enc0_l_network', ['-DC=32', '-DNH=2', '-DKIND=0', '-DXC=64', '-DD4R_M_EMBED=1', '-DD4R_M_UNROLL_C32_IO=1'] + flags)
            if split: matrix('expand_m_wmma.hlsl', 'l_patch_dec0', ['-DC=32', '-DXC=64'] + flags)
            matrix('layer_m_wmma.hlsl', 'dec0_l', ['-DC=32', '-DNH=2', '-DKIND=0' if split else '-DKIND=2', '-DXC=64',
                   '-DD4R_M_INPUT_F16=1' if split else '-DD4R_M_INPUT_F16=0', '-DD4R_M_OUTPUT_F16=1', '-DD4R_M_UNROLL_C32_IO=1'] + flags)
            stage('dec0_l_head', 'dec0_l_head')
            matrix('merge_m_wmma.hlsl', 'merge_l', ['-DC=32', '-DXC=64'])
        else:
            stage('enc0_m', 'enc0_m', ['-DDOT2=1', f'-DD4R_M_INPUT_WMMA={int(input_wmma)}', *light])
            if f16_expand:   # dec0_m.comp's own D3D12 declarations (D4R_HLSL), as compile_m.py builds RDNA 4's
                hlsl_build.build(HERE / 'dec0_m.comp', out / 'dec0_m.dxil', ['-DD4R_M_EXPAND_F16=1', *native_half], wave=32, dxc=dxc)
            else:
                stage('dec0_m', 'dec0_m', ['-DDOT2=1', '-DINT8=1'])
        suffix = '_l' if preset_l else '_m'
        # preset L's smallest tier is 12x12, not 11x11: with 11 its -1..+2 taps now and then read past the tile, into
        # shared memory another wave writes later (RX 7700 XT, 1280x720 -> 2560x1440: identical runs diverged in one
        # 16x16 block every 25-160 frames; 12x12: 3 x 160 frames identical, docs/hillclimb/journal.md)
        for output, tile in [('post' + suffix, 12 if preset_l else 10), ('post' + suffix + '_q', 16 if preset_l else 15), ('post' + suffix + '_c', 12 if preset_l else 11)]:
            stage('post_m', output, ['-DHALF_RGB10=1', '-DDISPLAY_MV=1', '-DD4R_POST_PACKED_TILES=1', f'-DPRESET_L={int(preset_l)}', f'-DTILE_W={tile}', f'-DTILE_H={tile}', *light])
        # the exact package stores the f32 result into the game's format (down_m.comp WIDE_STORE), and has an RGBA8
        # (and BGRA8) output as well; older ones round it to binary16 toward zero first
        wide = ['-DWIDE_STORE=1'] if exact else []
        stage('down_m', 'down_m', ['-DSEG=2', *wide, *light], wave=64)
        stage('down_m', 'down_m_rgb10', ['-DSEG=2', '-DOUTPUT_RGB10=1', *wide, *light], wave=64)
        if exact: stage('down_m', 'down_m_rgba8', ['-DSEG=2', '-DOUTPUT_RGBA8=1', *wide, *light], wave=64)
        for i in range(2):
            hlsl_build.build(HERE / 'exposure_k.comp', out / f'exposure_m{i}.dxil', ['-DMODE=1', f'-DPASS={i}'] + native_half, wave=64, dxc=dxc)
        # d3dhip on RDNA 3: the token-major tokens / dec1 to and from the HIP network's plane layout (d3d12/m_planes.hlsl)
        subprocess.run([compiler, '-nologo', '-T', 'cs_6_0', '-E', 'main', '-O3', str(HERE / 'd3d12' / 'm_planes.hlsl'),
                        '-Fo', str(out / 'm_planes.dxil')], check=True)
        (out / 'd3d12_ml.json').write_text(json.dumps(dict(format=marker.decode(), layer_parameters=layer_params, expansion=expand_mode, attention='direct', reconstruction_tiles='packed half bit patterns', input_matrix='wmma' if input_wmma else 'scalar' if not preset_l else 'wmma pixel block', preset='L' if preset_l else 'M', math=math,
                                                        c32_buffer_loops='unrolled' if preset_l else 'unused', network_buffer_io='uint4' if vector_io else 'uint', network_mlp='one hidden tile in registers' if stream_mlp else 'LDS hidden tiles', network='FP16 WMMA with FP32 accumulation', stages='WMMA pixel blocks with shared feature/head shaders' if preset_l else 'software FP8, f16 WMMA expansion, conversions to nearest even' if f16_expand else 'software FP8/int8'), indent=2))
        (out / 'd3d12_ml.bin').write_bytes(marker)
        # Publish the compatibility marker last; an interrupted update refuses
        # feature creation instead of combining an old marker with new shaders.
        (target / 'd3d12_ml.bin').unlink(missing_ok=True)
        for file in out.iterdir():
            if file.name != 'd3d12_ml.bin': file.replace(target / file.name)
        (out / 'd3d12_ml.bin').replace(target / 'd3d12_ml.bin')
        manifest = directory / 'model.json'
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            for file in [*target.glob('*.dxil'), target / 'd3d12_ml.bin']:
                metadata['files'][file.relative_to(directory).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(metadata, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--math', choices=['fast', 'fp8'], default='fast')
    p.add_argument('--expand-mode', choices=['fused', 'split'], default='split')
    p.add_argument('--layer-params', choices=['srv', 'cbv'], default='cbv')
    p.add_argument('--dxc')
    p.add_argument('--glslang')
    p.add_argument('--scalar-io', action='store_true', help='retain scalar network buffer views for ablation')
    mlp = p.add_mutually_exclusive_group()
    mlp.add_argument('--stream-mlp', dest='stream_mlp', action='store_true', help='use the default single hidden-tile register schedule for fast shared-network blocks')
    mlp.add_argument('--lds-mlp', dest='stream_mlp', action='store_false', help='retain the previous shared-network LDS hidden tiles for comparison')
    p.set_defaults(stream_mlp=True)
    a = p.parse_args()
    build(a.directory, a.math, a.dxc, a.glslang, a.layer_params, a.expand_mode, not a.scalar_io, a.stream_mlp)
