#!/usr/bin/env python3
"""Preset F's hybrid encoder 0 (native-work/enc0/enc0_m.comp after glslangValidator -E) into the GLSL glsl2hip.py
translates, for a vkhip kernel of the stage (kernels/rr/build_vkhip_f.sh, with --include=glsl_hip_enc0.h).

With the gather pass (enc0g) the stage reads its window's features from the gather's records in the arena and none
of the textures, surfaces or extra buffers it declares, so:
- those declarations and their accessors go, and the functions main does not reach
- the module-scope rr2_xf becomes rr2_block's parameter
- gl_NumWorkGroups.x / .y become the grid the runtime dispatches, from the parameter block ((level size + window
  shift + 7) / 8: words 0 / 1 and 20, engine/runtime_rr2.inc; z is 1)
- GLSL spellings glsl2hip does not take: hf literals, spirv_instruction declarations (glsl_hip_enc0.h defines the
  functions), a float[N](...) table, vec4 constructors, accumulator element-type conversions, a division by a power
  of two, and an FP8 operand element read as float

usage: enc0_hip_prep.py ENC0_M.preprocessed.comp OUT.comp
"""
import re
import sys


def strip_bindings(s):
    out, i, lines = [], 0, s.split('\n')
    while i < len(lines):
        line = lines[i]
        if re.match(r'layout\(set = 0, binding = \d+(, \w+)?\) uniform (writeonly )?(sampler2D|image2D)', line) or \
                re.match(r'layout\(set = 0, binding = \d+, std430\) buffer XBuf\d+', line) or \
                re.match(r'(uint|void) [GS]X\d+_\d+\(', line) or \
                line.startswith('layout(constant_id = 0) const bool RR_ROBUST_IMAGES'):
            i += 1
            continue
        if line.startswith('vec4 fetch0(sampler2D'):
            while lines[i].strip() != '}':
                i += 1
            i += 1
            continue
        out.append(line)
        i += 1
    return '\n'.join(out)


def prune(src):
    """Drop top-level functions (and spirv_instruction declarations) unreachable from main or top-level text."""
    lines = src.split('\n')
    funcs = {}   # name -> [(first line, last line)]
    pat = re.compile(r'^(?:const\s+)?[A-Za-z_][\w<>, ]*?\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(\{.*)?$')
    spv = re.compile(r'^spirv_instruction\(.*\)\s*[\w<> ]+\s+([A-Za-z_]\w*)\s*\(.*\);\s*$')
    i = depth = 0
    while i < len(lines):
        line = lines[i]
        if depth == 0:
            m = spv.match(line)
            if m:
                funcs.setdefault(m.group(1), []).append((i, i))
                i += 1
                continue
            m = pat.match(line)
            if m and not line.startswith(('layout', '#', 'return')):
                j = i
                if '{' not in line and i + 1 < len(lines) and lines[i + 1].strip().startswith('{'):
                    j = i + 1
                if '{' in lines[j]:
                    d, k = 0, j
                    while True:
                        d += lines[k].count('{') - lines[k].count('}')
                        if d == 0:
                            break
                        k += 1
                    funcs.setdefault(m.group(1), []).append((i, k))
                    i = k + 1
                    continue
        depth += line.count('{') - line.count('}')
        i += 1
    body = {n: '\n'.join('\n'.join(lines[a:b + 1]) for a, b in r) for n, r in funcs.items()}
    inside = {k for r in funcs.values() for a, b in r for k in range(a, b + 1)}
    ident = re.compile(r'[A-Za-z_]\w*')
    loose = '\n'.join(l for k, l in enumerate(lines) if k not in inside)
    live, todo = set(), ['main'] + [t for t in set(ident.findall(loose)) if t in body]
    while todo:
        n = todo.pop()
        if n in live or n not in body:
            continue
        live.add(n)
        todo.extend(t for t in set(ident.findall(body[n])) if t in body and t not in live)
    drop = {k for n, r in funcs.items() if n not in live for a, b in r for k in range(a, b + 1)}
    return '\n'.join(l for k, l in enumerate(lines) if k not in drop)


def prep(s):
    s = strip_bindings(s)
    if s.count('\nuint rr2_xf;\n') != 1 or 'void rr2_block()' not in s or '    rr2_xf = rr2_rec;' not in s:
        raise SystemExit('enc0_hip_prep: not the gather-split hybrid encoder 0 (engine/rr/gen_rr2_enc0.py --split)')
    s = s.replace('\nuint rr2_xf;\n', '\n')
    s = s.replace('void rr2_block()', 'void rr2_block(uint rr2_xf)').replace('rr2_block();', 'rr2_block(rr2_xf);')
    s = s.replace('    rr2_xf = rr2_rec;', '    uint rr2_xf = rr2_rec;', 1)
    s = s.replace('gl_NumWorkGroups.x', '((param.w[0] + (param.w[20] & 0xffffu) + 7u) / 8u)')
    s = s.replace('gl_NumWorkGroups.y', '((param.w[1] + (param.w[20] >> 16u) + 7u) / 8u)')
    s = s.replace('gl_NumWorkGroups.z', '1u')
    s = prune(s)
    for name in ('sampler2D', 'image2D', 'XBuf', 'gl_NumWorkGroups'):
        if name in s:
            raise SystemExit(f'enc0_hip_prep: {name} still used')
    s = re.sub(r'\b(\d+\.\d*(?:e[+-]?\d+)?)hf\b', r'float16_t(\1)', s)
    s = re.sub(r'(?m)^spirv_instruction\(set = "GLSL\.std\.450", id = \d+\) [^;]*;\n', '', s)
    s = re.sub(r'(?m)^const float (\w+)\[(\d+)\] = float\[\2\] \((.*)\);$', r'GLSL_TABLE float \1[\2] = {\3};', s)
    s = s.replace('(1.0 / 16777216.0)', '5.9604644775390625e-08')
    s = re.sub(r'coopmat < (float|float16_t), gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator > \((?=[A-Za-z_])',
               r'glslAccCvt<\1>(', s)
    s = re.sub(r'\bvec4\(', 'glslV4(', s)
    # through the operand's const accessor: the element proxy's float conversion recurses (an out-of-line call)
    s = s.replace('float(qx[wv][m][i])', 'glslF8f(qx[wv][m], i)')
    return s


if __name__ == '__main__':
    src, dst = sys.argv[1:3]
    open(dst, 'w').write(prep(open(src).read()))
