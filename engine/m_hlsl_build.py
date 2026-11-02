#!/usr/bin/env python3
"""Compile the M/L frame-stage GLSL resource ABI to HLSL/DXIL.

This bridge is deliberately limited to the listed M/L stages. glslang performs
conditional preprocessing first; declarations are then translated, while the
stage algorithms remain shared GLSL. The returned binding schema describes the
DXIL registers for the native runtime. Network WMMA kernels are separate HLSL.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import hlsl_build

HERE = Path(__file__).resolve().parent
STAGES = {'enc0_m.comp', 'dec0_m.comp', 'post_m.comp', 'down_m.comp', 'layer_l_dot2.comp', 'merge_l_dot2.comp', 'enc0_l_features.comp', 'dec0_l_head.comp'}
BUFFER = re.compile(r'layout\s*\(([^)]*)\)\s*(readonly|writeonly)?\s*buffer\s+(\w+)\s*\{([^}]+)\}\s*(\w+)\s*;', re.S)
TEXTURE = re.compile(r'layout\s*\(([^)]*)\)\s*uniform\s+(writeonly\s+)?(sampler2D|image2D)\s+(\w+)\s*;')


def convert(source):
    """Translate already-preprocessed storage/texture declarations; reject unknown layouts."""
    bindings, init = [], []
    source = re.sub(r'(#line[^\n]*)', lambda m: m[0].replace('\\', '/'), source)
    # GLSL array constructors become HLSL initializer lists. Match balanced
    # parentheses so vector constructors inside the list remain untouched.
    constructor = re.compile(r'\b(?:[biuf]?[vV]ec[234]|float|uint|int)\s*\[\s*\d+\s*\]\s*\(')
    while True:
        m = constructor.search(source)
        if not m: break
        depth, end = 1, m.end()
        while depth and end < len(source):
            depth += (source[end] == '(') - (source[end] == ')')
            end += 1
        if depth: raise ValueError('unclosed array constructor')
        source = source[:m.start()] + '{' + source[m.end():end - 1] + '}' + source[end:]

    def buffer(m):
        layout, access, typename, fields, name = m.groups()
        binding = int(re.search(r'binding\s*=\s*(\d+)', layout)[1])
        array = re.fullmatch(r'\s*(\w+)\s+(\w+)\s*\[\s*\]\s*;\s*', fields)
        writable = access != 'readonly'
        register = ('u' if writable else 't') + str(binding)
        if array:
            scalar, field = array.groups()
            if scalar == 'uint16_t':
                if writable: raise ValueError('16-bit output buffers are not supported')
                bindings.append(dict(binding=binding, kind='raw'))
                return f'''ByteAddressBuffer {name}_raw : register({register});
struct {typename}Elements {{ ByteAddressBuffer data; uint16_t operator[](uint i) {{ return data.Load<uint16_t>(2u * i); }} }};
struct {typename} {{ {typename}Elements {field}; }};
static {typename} {name} = {{ {{ {name}_raw }} }};'''
            if scalar not in ('uint', 'float', 'uvec4'):
                raise ValueError('unsupported buffer element: ' + scalar)
            bindings.append(dict(binding=binding, kind='rwtyped' if writable else 'typed', scalar=scalar))
            resource = 'RWBuffer' if writable else 'Buffer'
            return f'''{resource}<{scalar}> {name}_{field} : register({register});
struct {typename} {{ {resource}<{scalar}> {field}; }};
static {typename} {name} = {{ {name}_{field} }};'''
        # Parameter blocks remain in UAV state because the exposure pass updates
        # selected words before each input/reconstruction/downsample stage.
        init.append(f'{name} = {name}_buffer[0];')
        bindings.append(dict(binding=binding, kind='params', fields=fields.strip()))
        return f'''struct {typename} {{ {fields} }};
RWStructuredBuffer<{typename}> {name}_buffer : register(u{binding});
static {typename} {name};'''
    source = BUFFER.sub(buffer, source)
    def texture(m):
        layout, write, kind, name = m.groups()
        binding = int(re.search(r'binding\s*=\s*(\d+)', layout)[1])
        if kind == 'sampler2D':
            bindings.append(dict(binding=binding, kind='texture'))
            return f'D4R_SAMPLER({name}, t{binding}, s{binding})'
        if not write: raise ValueError('only write-only images are supported')
        fmt = layout.split(',')[-1].strip()
        value = 'float2' if fmt == 'rg16f' else 'float4'
        qualifier = 'unorm ' if fmt in ('rgba8', 'rgb10_a2') else 'snorm ' if fmt == 'rgba8_snorm' else ''
        bindings.append(dict(binding=binding, kind='rwtexture', format=fmt))
        return f'RWTexture2D<{qualifier}{value}> {name} : register(u{binding});'
    source = TEXTURE.sub(texture, source)
    # The mixed-dot SPIR-V intrinsic becomes the native SM6 packed-half dot.
    source = re.sub(r'spirv_instruction\([^;]+\)\s*float\s+dot2acc\([^;]+;',
                    'float dot2acc(f16vec2 a, f16vec2 b, float c) { return dot2add(a, b, c); }', source)
    arrays = {m[1]: m[2] for m in re.finditer(r'\b\w+\s+(\w+)\s*\[([^\]\n]+)\]\s*;', source)}
    for name, count in arrays.items():
        source = source.replace(name + '.length()', '(' + count + ')')
    # Every M/L dispatch has Z=1. GroupID.z provides the same runtime zero
    # used to keep the downsample filter's parity from being constant-folded.
    source = re.sub(r'int\s*\(\s*gl_NumWorkGroups\.z\s*\)\s*-\s*1', 'int(gl_WorkGroupID.z)', source)
    has_waves = 'gl_Subgroup' in source
    if has_waves:
        # The source's 1D local indexes enumerate logical wave lanes; D3D12 does
        # not guarantee that SV_GroupIndex follows that enumeration.
        source = source.replace('gl_LocalInvocationID.x', '(d4rMWave * 32u + WaveGetLaneIndex())')
        source = source.replace('gl_SubgroupID', 'd4rMWave').replace('gl_SubgroupInvocationID', 'WaveGetLaneIndex()')
        source = '''static uint d4rMWave;
groupshared uint d4rMWaveCounter;
void d4rStageInitWaves() {
    if (gl_LocalInvocationIndex == 0) d4rMWaveCounter = 0;
    barrier();
    uint index = 0;
    if (WaveIsFirstLane()) InterlockedAdd(d4rMWaveCounter, 1, index);
    d4rMWave = WaveReadLaneFirst(index);
}
''' + source
        init.insert(0, 'd4rStageInitWaves();')
    if re.search(r'layout\s*\((?!\s*local_size)', source):
        raise ValueError('untranslated resource layout')
    source = '''#define equal(a, b) ((a) == (b))
#define notEqual(a, b) ((a) != (b))
#define linear d4r_linear
uint pack32(uint16_t2 v) { return uint(v.x) | (uint(v.y) << 16); }
int2 d4rTextureSize(Texture2D<float4> image, int level) { uint w, h, levels; image.GetDimensions(level, w, h, levels); return int2(w, h); }
#define textureSize d4rTextureSize
''' + '#define D4R_HLSL_INIT_CALL ' + ' '.join(init) + '\n' + source
    return source, sorted(bindings, key=lambda b: b['binding'])


def build(shader, output, defines=(), wave=32, dxc=None, glslang=None):
    shader, output = Path(shader).resolve(), Path(output)
    if shader.name not in STAGES:
        raise ValueError('not an M/L frame stage: ' + shader.name)
    glslang = glslang or os.environ.get('D4R_GLSLANG') or shutil.which('glslangValidator')
    if not glslang: raise FileNotFoundError('set D4R_GLSLANG or put glslangValidator on PATH')
    result = subprocess.run([str(glslang), '-E', '-S', 'comp', *defines, str(shader)], capture_output=True, text=True)
    if result.returncode: raise RuntimeError(result.stdout + result.stderr)
    source, schema = convert(result.stdout)
    with tempfile.TemporaryDirectory(prefix='d4r-ml-stage.') as temp:
        path = Path(temp) / shader.name
        path.write_text(source)
        hlsl_build.build(path, output, ['-DD4R_NATIVE_HALF=1'], wave=wave, dxc=dxc)
    output.with_suffix('.bindings.json').write_text(json.dumps(schema, indent=2))
    return schema


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('shader', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('-D', dest='defines', action='append', default=[])
    p.add_argument('--dxc')
    p.add_argument('--glslang')
    a = p.parse_args()
    build(a.shader, a.output, ['-D' + d for d in a.defines], dxc=a.dxc, glslang=a.glslang)
