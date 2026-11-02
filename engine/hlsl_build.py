#!/usr/bin/env python3
"""Compiles one of the engine's GLSL compute shaders to DXIL for the D3D12 backend, from the same source file the
Vulkan backend's SPIR-V is built from (glslangValidator).

The GLSL-only directives (#version, #extension and the workgroup-size layout) are blanked out, keeping the line
numbers; dialect_hlsl.h is included first and supplies everything the two languages spell differently, and the
shader's #ifdef D4R_HLSL blocks declare its resources. The workgroup size becomes the attributes of an entry point,
d4r_entry, that calls the shader's main().

usage: hlsl_build.py SHADER OUTPUT [-DNAME=VALUE ...] [--wave 32|64] [--profile cs_6_N] [--dxc DXC] [--asm]
D4R_DXC_ARGS adds dxc arguments (diagnostics, e.g. -Gis).
"""
import argparse, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARRAY_OPEN = re.compile(r'=\s*\w+\[\d*\]\(\s*$')     # an array constructor whose elements follow on their own lines
SIZE = re.compile(r'^\s*layout\s*\(\s*local_size_x\s*=\s*([^,)]+?)\s*(?:,\s*local_size_y\s*=\s*([^,)]+?)\s*)?'
                  r'(?:,\s*local_size_z\s*=\s*([^,)]+?)\s*)?\)\s*in\s*;\s*(?://.*)?$')


INCLUDE = re.compile(r'^\s*#include\s+"([^"]+\.glsl)"')


def translate(source, unroll=False, directory=None, name=None, seen=None):
    """the shader's text as HLSL input, and whether it declares a workgroup size. Each declaration becomes a definition
    of D4R_WORKGROUP_SIZE, so a shader may choose its size with the preprocessor. With unroll, every loop
    without an attribute is unrolled unless its line says 'rolled' (matrix arrays indexed by a loop counter stay
    in registers only once DXC has unrolled the loop; AMD's GLSL compilers do that themselves)."""
    lines, size, array = [], False, False
    seen = set() if seen is None else seen
    for number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        m = SIZE.match(line)
        inc = INCLUDE.match(line) if directory else None
        if inc and (directory / inc.group(1)).is_file():
            # the engine's GLSL includes are translated as well (their global constants, array constructors), in place
            path = (directory / inc.group(1)).resolve()
            if path in seen:
                raise ValueError(f'{path.name} includes itself')
            seen.add(path)      # the files being translated (an include in every branch of an #if is inlined in each)
            body, inner = translate(path.read_text(), unroll, path.parent, path.as_posix(), seen)
            seen.discard(path)
            size = size or inner
            lines += [f'#line 1 "{path.as_posix()}"', *body, f'#line {number + 1} "{name}"']
        elif m:
            size = True
            lines.append('#define D4R_WORKGROUP_SIZE ' + ', '.join(f'({v or 1})' for v in m.groups()))
        elif stripped.startswith('#version') or stripped.startswith('#extension'):
            lines.append('')
        elif array and stripped == ');':         # the end of a multi-line array constructor: an initializer list
            array = False
            lines.append(line.replace(');', '};'))
        elif line.startswith('const '):         # a global constant: implicitly uniform in HLSL
            array = bool(ARRAY_OPEN.search(line))
            lines.append('static ' + ARRAY_OPEN.sub('= {', line))
        else:
            line = line.replace('[[unroll]]', '[unroll]').replace('[[dont_unroll]]', '[loop]')
            if unroll and 'rolled' not in line:
                line = re.sub(r'(\[(?:unroll(?:\([^)]*\))?|loop|fastopt)\]\s*)?\bfor\s*\(',
                              lambda m: m.group(0) if m.group(1) else '[unroll] ' + m.group(0), line)
            lines.append(line)
    return lines, size


def dxc_path(given=None):
    for candidate in (given, os.environ.get('D4R_DXC'), shutil.which('dxc'), shutil.which('dxc.exe')):
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)
    sdk = os.environ.get('VULKAN_SDK')
    if sdk and (Path(sdk) / 'Bin' / 'dxc.exe').is_file():
        return str(Path(sdk) / 'Bin' / 'dxc.exe')
    raise FileNotFoundError('dxc (DirectX Shader Compiler) not found: set D4R_DXC or put it on PATH')


def build(shader, output, defines=(), wave=64, dxc=None, assembly=False, unroll=False, profile='cs_6_6'):
    shader, output = Path(shader).resolve(), Path(output)
    lines, size = translate(shader.read_text(), unroll, shader.parent, shader.as_posix())
    if not size:
        raise ValueError('no workgroup size (layout(local_size_x = ...) in;)')
    with tempfile.TemporaryDirectory(prefix='d4r-hlsl.') as tmp:
        src = Path(tmp) / (shader.stem + '.hlsl')
        text = ['#include "dialect_hlsl.h"', f'#line 1 "{shader.as_posix()}"', *lines,
                '#ifndef D4R_HLSL_INIT_CALL', '#define D4R_HLSL_INIT_CALL', '#endif',
                '#define D4R_ENTRY_SIZE(...) D4R_ENTRY(__VA_ARGS__)', 'D4R_ENTRY_SIZE(D4R_WORKGROUP_SIZE)', '']
        src.write_text('\n'.join(text))
        command = [dxc_path(dxc), '-nologo', '-T', profile, '-E', 'd4r_entry', '-HV', '2021', '-enable-16bit-types',
                   '-O3', '-I', str(HERE), '-I', str(HERE / 'd3d12'), '-I', str(shader.parent), f'-DD4R_WAVE_SIZE={wave}',
                   *[d if d.startswith('-D') else '-D' + d for d in defines], *os.environ.get('D4R_DXC_ARGS', '').split(), str(src)]
        command += ['-Fc' if assembly else '-Fo', str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            raise RuntimeError(f'dxc failed for {shader.name}')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('shader', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('-D', dest='defines', action='append', default=[], metavar='NAME=VALUE')
    p.add_argument('--wave', type=int, choices=(32, 64), default=64)
    p.add_argument('--profile', default='cs_6_6', help='shader model (cs_6_9 keeps vector arithmetic in the DXIL)')
    p.add_argument('--dxc')
    p.add_argument('--asm', action='store_true', help='write the DXIL disassembly instead')
    p.add_argument('--unroll', action='store_true', help='unroll the loops (see translate)')
    a = p.parse_args()
    build(a.shader, a.output, a.defines, a.wave, a.dxc, a.asm, a.unroll, a.profile)


if __name__ == '__main__':
    main()
