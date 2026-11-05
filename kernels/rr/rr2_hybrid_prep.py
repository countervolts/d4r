#!/usr/bin/env python3
"""Preset F's hybrid stages (native-work/<stage>/<stage>_m.comp after glslangValidator -E: encoder 0 after the gather
pass, decoder 0's pixel tail dec0_1) into the GLSL glsl2hip.py translates, for vkhip kernels of the stages
(kernels/rr/build_vkhip_f.sh, with --include=glsl_hip_hybrid.h).

- texture, sampler and extra-buffer declarations and their accessors go (encoder 0 reads its window's features from
  the gather's records; neither stage samples a texture), as do the functions main does not reach
- the storage images a stage writes (dec0_1's surfaces) become entries of a descriptor table, binding 3 (glsl2hip.py:
  descriptor set 0's table, which vkhip hands the kernel), in binding order; stores through the hardware's
  image_store, whose bounds check replaces the shader's own (an image size of 2^30)
- encoder 0: the module-scope rr2_xf becomes rr2_block's parameter, and gl_NumWorkGroups.x / .y the grid the runtime
  dispatches, from the parameter block ((level size + window shift + 7) / 8: words 0 / 1 and 20,
  engine/runtime_rr2.inc; z is 1)
- GLSL spellings glsl2hip does not take: hf literals, spirv_instruction declarations (glsl_hip_hybrid.h defines the
  functions), a float[N](...) table, vec4 constructors, accumulator element-type conversions, a division by a power
  of two, and an FP8 operand element read as float

usage: rr2_hybrid_prep.py STAGE_M.preprocessed.comp OUT.comp
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


def two_argument(s, ctor, helper):
    """ctor(a, b) -> helper(a, b): a constructor with more than one argument (one argument is a conversion, which
    the cast form already does)."""
    out, last = [], 0
    for m in re.finditer(r'\b' + ctor + r'\(', s):
        depth, i, commas = 1, m.end(), 0
        while depth:
            c = s[i]
            depth += (c == '(') - (c == ')')
            commas += c == ',' and depth == 1
            i += 1
        if commas:
            out.append(s[last:m.start()] + helper + '(')
            last = m.end()
    return ''.join(out) + s[last:]


IMAGE = re.compile(r'(?m)^layout\(set = 0, binding = (\d+), \w+\) uniform writeonly image2D (\w+);$')


def prep(s):
    images = sorted(((int(b), n) for b, n in IMAGE.findall(s)))
    s = strip_bindings(s)
    for k, (_, name) in enumerate(images):
        s = re.sub(r'\bimageSize\(' + name + r'\)', f'glslImageSize(images.d[{k}])', s)
        s = re.sub(r'\bimageStore\(' + name + r', ', f'glslImageStore(images.d[{k}], ', s)
    table = 'layout(set = 0, binding = 3, std430) readonly buffer Images { glslImage d[]; } images;\n'
    param = 'layout(set = 0, binding = 0, std430) readonly buffer ParamBlock'
    if images:
        s = s.replace(param, table + param, 1)
    if '\nuint rr2_xf;\n' in s:   # encoder 0
        if s.count('\nuint rr2_xf;\n') != 1 or 'void rr2_block()' not in s or '    rr2_xf = rr2_rec;' not in s:
            raise SystemExit('rr2_hybrid_prep: not the gather-split hybrid encoder 0 (engine/rr/gen_rr2_enc0.py --split)')
        s = s.replace('\nuint rr2_xf;\n', '\n')
        s = s.replace('void rr2_block()', 'void rr2_block(uint rr2_xf)').replace('rr2_block();', 'rr2_block(rr2_xf);')
        s = s.replace('    rr2_xf = rr2_rec;', '    uint rr2_xf = rr2_rec;', 1)
        s = s.replace('gl_NumWorkGroups.x', '((param.w[0] + (param.w[20] & 0xffffu) + 7u) / 8u)')
        s = s.replace('gl_NumWorkGroups.y', '((param.w[1] + (param.w[20] >> 16u) + 7u) / 8u)')
        s = s.replace('gl_NumWorkGroups.z', '1u')
    s = prune(s)
    if 'images.d[' not in s:   # encoder 0 declares surfaces it does not write
        s = s.replace(table, '')
    for name in ('sampler2D', 'image2D', 'XBuf', 'gl_NumWorkGroups.x', 'gl_NumWorkGroups.y'):
        if name in s:
            raise SystemExit(f'rr2_hybrid_prep: {name} still used')
    s = re.sub(r'\b(\d+\.\d*(?:e[+-]?\d+)?)hf\b', r'float16_t(\1)', s)
    s = re.sub(r'(?m)^spirv_instruction\(set = "GLSL\.std\.450", id = \d+\) [^;]*;\n', '', s)
    s = re.sub(r'(?m)^const float (\w+)\[(\d+)\] = float\[\2\] \((.*)\);$', r'GLSL_TABLE float \1[\2] = {\3};', s)
    s = s.replace('(1.0 / 16777216.0)', '5.9604644775390625e-08')
    s = re.sub(r'coopmat < (float|float16_t), gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator > \((?=[A-Za-z_])',
               r'glslAccCvt<\1>(', s)
    s = re.sub(r'\bvec4\(', 'glslV4(', s)
    # through the operand's const accessor: the element proxy's float conversion recurses (an out-of-line call)
    s = s.replace('float(qx[wv][m][i])', 'glslF8f(qx[wv][m], i)')
    s = two_argument(s, 'ivec2', 'glslIV2')
    # an FP8 weight byte picked from a pair of pairs: byte i & 3 of the word
    s = s.replace('uintBitsToFloate4m3EXT( ( (uint(i) & 2u) == 0u ? unpack8(xw).xy : unpack8(xw).zw)[uint(i) & 1u])',
                  'uintBitsToFloate4m3EXT(uint8_t(xw >> (8u * (uint(i) & 3u))))')
    return s


if __name__ == '__main__':
    src, dst = sys.argv[1:3]
    open(dst, 'w').write(prep(open(src).read()))
