#!/usr/bin/env python3
"""A preset F stage shader as a vkhip kernel: the engine's GLSL after glslangValidator -E (the model's
<stage>.preprocessed.comp) rewritten into HIP over kernels/rr/glsl_hip.h, so that LLVM compiles the shader the
Vulkan path runs, with its numerics. Handles what engine/rr/rr2_w.comp and rr2_m.comp use: storage buffers bound as
the stage's parameter block (binding 0), weights (1) and arena (2), shared arrays, cooperative matrices, FP8 and
binary16 arithmetic. The kernel takes vkhip's F arguments {param, weights, arena} (engine/vkhip/vkhip_net.cpp).

usage: glsl2hip.py [--sat] [--rolled=VAR,...] [--v-layout=rr2_stage.h] [--record=ARRAY,...] STAGE.preprocessed.comp KERNEL_NAME OUT.hip
--sat: FP8 conversions through MODE.FP16_OVFL (glsl_hip.h GLSL_FP8_SAT), without the infinity tests before them
--rolled=hd,br,ks>2: loops over these variables stay rolled; VAR>N only those with more than N iterations
--record=rv,xr: these per-thread two-dimensional float or float16_t arrays (rr2_m.comp's residual r, a decoder's skip
  input) in the window's record in the arena, one after another, instead of registers (the runtime reserves the
  records of the levels with phases for vkhip); needs --v-layout
--v-layout=H: the stage's rr2_stage.h (rr2_pack.py); the loader then transposes the heads' V weight tiles (rr2_v_layout,
  engine/vkhip/vkhip_net.cpp), the only B operands read from the weights, and their loads become 8 bytes per lane
"""
import re
import sys

BINDINGS = {'0': 'args.param', '1': 'args.weights', '2': 'args.arena'}


FINITE = re.compile(r'\(isinf\(( ?[\w()\[\]+*. ]+?)\) \? uintBitsToFloat\(0x7fc00000u\) : \(( ?[\w()\[\]+*. ]+?)\)\)')


def translate(src, kernel, sat=False, rolled=(), v_layout=None, record=()):
    out, buffers, names, threads = [], [], [], 32
    for line in src.split('\n'):
        s = line.strip()
        m = re.match(r'layout\(local_size_x = (\d+)', s)
        if m:
            threads = int(m.group(1))
        if s.startswith(('#version', '#extension', '#line')) or s.startswith('layout(local_size'):
            continue
        m = re.match(r'layout\(set = 0, binding = (\d), std430\) (readonly )?buffer \w+ \{ (\w+) (\w+)\[\]; \} (\w+);', s)
        if m:
            binding, ro, elem, member, name = m.groups()
            if binding not in BINDINGS:
                raise SystemExit('glsl2hip: unsupported binding: ' + s)
            const = 'const ' if ro else ''
            buffers.append(f'    const struct {{ {const}{elem}* {member}; }} {name} = {{glslGlobal<{const}{elem}>({BINDINGS[binding]})}};')
            names.append(name)
            continue
        if s.startswith('layout('):
            raise SystemExit('glsl2hip: unsupported declaration: ' + s)
        out.append(line)
    text = '\n'.join(out)
    # shared arrays and the module-scope constants
    text = re.sub(r'^(\s*)shared ', r'\1__shared__ ', text, flags=re.M)
    text = re.sub(r'^const (uint|int|float) ', r'constexpr \1 ', text, flags=re.M)
    # RR_ZERO16 (gl_NumWorkGroups.z - 1: zero that the compiler cannot see through) as a constant zero
    text = text.replace('uint16_t RR_ZERO16;', 'constexpr uint16_t RR_ZERO16 = 0;')
    text = re.sub(r'\n\s*RR_ZERO16 = [^;]*;', '\n', text)
    if re.search(r'^(uint|int|float|float16_t|uint16_t|bool)\s+\w+(\[[^\]]*\])*\s*;', text, flags=re.M):
        raise SystemExit('glsl2hip: module-scope private variables are not supported')
    if 'gl_NumWorkGroups.x' in text:   # HIP's hidden arguments, which vkhip's kernel ABI has no room for
        raise SystemExit('glsl2hip: gl_NumWorkGroups.x is not supported')
    text = re.sub(r'\binout\s+', '', text)
    text = re.sub(r'\bout\s+(\w+)\s+(\w+)', r'\1& \2', text)
    # inout parameters are references
    text = re.sub(r'\b(bool|float|uint|int|float16_t)\s+(mid|bad)\)', r'\1& \2)', text)
    text = re.sub(r'\bprecise\s+', '', text)
    if sat:   # FINITE (infinity -> NaN before a saturated FP8 conversion): the stages' f32 sums of FP8 products are finite
        text = FINITE.sub(lambda m: '(' + m.group(1).strip() + ')' if m.group(1).strip() == m.group(2).strip() else m.group(0), text)
    text = text.replace('[[unroll]]', '_Pragma("unroll")').replace('[[dont_unroll]]', '_Pragma("nounroll")')
    # FP8 pair stores (Q8PAIRS, ACT8 and the like): one conversion into the operand's half word
    text = re.sub(r'(\(?[\w\[\]. +*()]*?\)?)\[([^\]]+)\] = (\w+)\.x; \1\[\2 \+ 1u?\] = \3\.y;', r'glslSetPair(\1, \2, \3);', text)
    for spec in rolled:   # loops left rolled: unrolled, LLVM hoists every iteration's weight loads and spills
        v, _, least = spec.partition('>')

        def roll(m):
            bound = re.sub(r'(\d)u\b', r'\1', m.group(2))
            if least and not (re.fullmatch(r'[\d\s()+*/-]+', bound) and eval(bound.replace('/', '//')) > int(least)):
                return m.group(0)
            return '_Pragma("nounroll") ' + m.group(1)
        text = re.sub(r'_Pragma\("unroll"\) (for \(uint ' + v + r' = 0u; ' + v + r' < ([^;]+);)', roll, text)
    # Per-thread arrays in the window's record in the arena (the words rr2_m.comp's phases keep per window, unused by a
    # whole stage): at levels 3 - 5 a window's latency is the stage's time, and r's 80 registers (read only after the
    # MLP) leave LLVM none to issue the weight loads early with; in LDS they would cost the CU its second workgroup
    if record:
        if 'ParamBlock { uint p[]; } param;' not in src or not v_layout:
            raise SystemExit('glsl2hip: --record needs the stage parameter block and --v-layout')
        d = dict(re.findall(r'^#define (\w+) (\d+)u?\b', v_layout, flags=re.M))
        nh, mt, nb = int(d['NH']), int(d['C']) // 16, int(d['NB'])
        sc_win = nh * 512 + mt * 256 + mt * 1024 + nb * 512   # SC_WIN: the runtime's words per window
        grid = '((param.p[0] + (param.p[20] & 0xffffu) + 7u) >> 3u)'
        at = 0
        for name in record:
            m = re.search(r'^(\s*)(float|float16_t)\s+' + name + r'\s*\[([^\]]+)\]\[([^\]]+)\];', text, flags=re.M)
            if not m:
                raise SystemExit(f'glsl2hip: --record: no two-dimensional float or float16_t array {name}')
            ind, elem, rows, cols = m.groups()
            size = eval(re.sub(r'(\d)u\b', r'\1', f'({rows}) * ({cols})').replace('/', '//')) * threads
            if at + size > sc_win:
                raise SystemExit(f'glsl2hip: --record: {name} does not fit in the window record ({sc_win} words)')
            # the window's pointer (uniform: scalar registers) and per element the thread's word at a constant offset from
            # it, one register shared by every access: a per-thread pointer would take two registers per array, and 32-bit
            # indices each a 64-bit address
            put, get = (('floatBitsToUint(', ')'), ('uintBitsToFloat(', ')')) if elem == 'float' else \
                       (('uint(float16BitsToUint16(', '))'), ('uint16BitsToFloat16(uint16_t(', '))'))
            word = lambda r, c: f'{name}_rec[(uint({r}) * ({cols}) + uint({c})) * RR2_THREADS + gl_LocalInvocationIndex]'
            access = r'\(\s*' + name + r'\[([^\[\]]+)\]\)\[([^\[\]]+)\]|\b' + name + r'\[([^\[\]]+)\]\[([^\[\]]+)\]'
            rest = re.sub('(?:' + access + r') = ([^;]+);', lambda a: f'{word(a.group(1) or a.group(3), a.group(2) or a.group(4))}'
                          f' = {put[0]}{a.group(5)}{put[1]};', text[m.end():])
            rest = re.sub(access, lambda a: f'{get[0]}glslRecordLoad(&{word(a.group(1) or a.group(3), a.group(2) or a.group(4))}){get[1]}', rest)
            text = (text[:m.start()] + f'{ind}uint* const {name}_rec = &arena.w[(param.p[26] >> 2u) + (gl_WorkGroupID.y * {grid}'
                    f' + gl_WorkGroupID.x) * {sc_win}u + {at}u];' + rest)
            at += size
    # GLSL floating literals are single precision
    text = re.sub(r'(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?)(?![\w.])', r'\1f', text)
    # 1 / x as RADV compiles it, v_rcp_f32: LLVM would fold float16_t(1.0f / float(h)) into v_rcp_f16, another rounding
    text = re.sub(r'(?<![\w.])1\.0f / (\w+(?:\([^()]*\))?)', r'__builtin_amdgcn_rcpf(\1)', text)
    # constructors of vector types
    for t, f in (('vec2', 'glslVec2'), ('f16vec2', 'glslH2'), ('f16vec3', 'glslH3'), ('f16vec4', 'glslH4'), ('uvec2', 'glslU2')):
        text = re.sub(r'(?<![\w.])' + t + r'\(', f + '(', text)
    main = re.search(r'^void main\(\)\s*\n\{', text, flags=re.M)
    if not main:
        raise SystemExit('glsl2hip: no main')
    # functions that use the buffers (and their callers) become lambdas inside the kernel, after the buffers
    head_text, lambdas, moved = text[:main.start()], [], set()
    funcs = list(re.finditer(r'^([A-Za-z_][\w<>, ]*?)\s+(\w+)\(([^)]*)\)\s*\n?\{', head_text, flags=re.M))
    spans = []
    for f in funcs:
        depth, i = 0, f.end() - 1
        while True:
            depth += {'{': 1, '}': -1}.get(head_text[i], 0)
            if depth == 0:
                break
            i += 1
        spans.append((f, i + 1))
    for f, end in spans:
        code = head_text[f.end():end]
        if any(re.search(r'\b' + n + r'\.', code) for n in names) or any(re.search(r'\b' + m + r'\(', code) for m in moved):
            moved.add(f.group(2))
            lambdas.append(f'    auto {f.group(2)} = [&]({f.group(3)}) -> {f.group(1).strip()} {{' + code + ';')
    for f, end in reversed(spans):
        if f.group(2) in moved:
            head_text = head_text[:f.start()] + head_text[end:]
    text = head_text + text[main.start():]
    main = re.search(r'^void main\(\)\s*\n\{', text, flags=re.M)
    body = '\n'.join(buffers + lambdas)
    text = (text[:main.start()] + '#pragma clang force_cuda_host_device end\n'
            + f'extern "C" __global__ __launch_bounds__(RR2_THREADS) RR2_ATTRS void {kernel}(GlslArgs args)\n{{\n    glslInit();\n' + body
            + text[main.end():])
    vl = ''
    if v_layout:
        d = dict(re.findall(r'^#define (\w+) (\d+)u?\b', v_layout, flags=re.M))
        if not all(k in d for k in ('C', 'NH', 'W_WV0', 'W_WQK0')):
            raise SystemExit('glsl2hip: --v-layout needs C, NH, W_WV0 and W_WQK0 in the stage header')
        head_stride = int(d['W_WQK1']) - int(d['W_WQK0']) if 'W_WQK1' in d else 0
        vl = ('#define GLSL_B_WEIGHTS_TRANSPOSED\n'
              f'extern "C" __device__ __constant__ uint32_t rr2_v_layout[4] = {{{d["C"]}, {d["NH"]}, {d["W_WV0"]}, {head_stride}}};\n')
    head = ('// generated by kernels/rr/glsl2hip.py: do not edit\n' + ('#define GLSL_FP8_SAT\n' if sat else '') + vl + '#include "glsl_hip.h"\n'
            f'#ifndef RR2_THREADS\n#define RR2_THREADS {threads}\n#endif\n'
            '#ifndef RR2_WAVES\n#define RR2_ATTRS\n#else\n#define RR2_ATTRS __attribute__((amdgpu_waves_per_eu(RR2_WAVES)))\n#endif\nextern "C" __device__ __constant__ uint32_t rr2_threads = RR2_THREADS;\n'
            'namespace glsl {\n#pragma clang force_cuda_host_device begin\n')
    return head + text + '\n} // namespace glsl\n'


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    src, kernel, dst = args
    rolled = [v for a in sys.argv if a.startswith('--rolled=') for v in a[9:].split(',') if v]
    v_layout = next((open(a[11:]).read() for a in sys.argv if a.startswith('--v-layout=')), None)
    record = [v for a in sys.argv if a.startswith('--record=') for v in a[9:].split(',') if v]
    open(dst, 'w').write(translate(open(src).read(), kernel, '--sat' in sys.argv, rolled, v_layout, record))


if __name__ == '__main__':
    main()
