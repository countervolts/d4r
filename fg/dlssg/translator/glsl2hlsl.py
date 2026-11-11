#!/usr/bin/env python3
"""glsl2hlsl: a translated kernel (ptxc.py's GLSL, rt.glsl inlined) as HLSL for the DXIL route on native Windows.

The kernel body ptxc emits is regular: PTX registers as locals, structured control flow, memory through
NvxT(address).v buffer references, and rt.glsl helpers. This keeps the body, swaps rt.glsl for rt.hlsl (the same
helpers on D3D12) and rewrites the few GLSL-only forms:
  NvxT(a).v = x;                -> nvx_st_NvxT(a, x);
  NvxT(a).v                     -> nvx_ld_NvxT(a)
  atomicOp(NvxT(a).v, x)        -> nvx_atomOp_NvxT(a, x)
  y = atomicOp(smem[i], x);     -> InterlockedOp(smem[i], x, y);   (and the same without a result)
  float16_t(x) / f16vecN(...)   -> exact round-to-nearest-even conversions (AMD's D3D12 compiler truncates)
  shared / barrier / [[...]]    -> groupshared / GroupMemoryBarrierWithGroupSync / [attributes]
  void main()                   -> nvx_main(), called by an entry point with the root signature, [numthreads] from
                                   NVX_BX/BY/BZ and [WaveSize(32)]
Linux and the Vulkan route do not use this file. Usage: glsl2hlsl.py KERNEL.comp OUT.hlsl  (prints the dxc flags the
kernel's float modes need)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RT_GLSL = open(os.path.join(HERE, 'rt.glsl'), encoding='utf-8').read()

ATOM = {'Add': 'Add', 'Min': 'Min', 'Max': 'Max', 'And': 'And', 'Or': 'Or', 'Xor': 'Xor', 'Exchange': 'Exchange',
        'CompSwap': 'CompareExchange'}


class Err(Exception):
    pass


def split_kernel(src):
    i = src.find(RT_GLSL)
    if i < 0:
        raise Err('rt.glsl is not inlined verbatim (T1 kernel?)')
    return src[:i], src[i + len(RT_GLSL):]


def convert_prefix(prefix):
    keep, modes = [], []
    for line in prefix.splitlines():
        t = line.strip()
        m = re.match(r'spirv_execution_mode\((\d+),\s*(\d+)\);', t)
        if m:
            modes.append((int(m.group(1)), int(m.group(2))))
            continue
        if not t or t.startswith('#version') or t.startswith('#extension') or t.startswith('layout(local_size'):
            continue
        m = re.match(r'layout\(constant_id\s*=\s*3\)\s*const uint NVX_DYN_SMEM\s*=\s*0u;', t)
        if m:
            keep.append('#define NVX_DYN_SMEM (NVX_SMEM_DYN)')
            continue
        m = re.match(r'layout\(constant_id\s*=\s*4\)\s*const uint NVX_NSUB\s*=\s*\d+u;', t)
        if m:   # subgroups per workgroup (the executor's specialization constant 4)
            keep.append('#define NVX_NSUB ((NVX_BX * NVX_BY * NVX_BZ + 31u) / 32u)')
            continue
        if t.startswith('layout('):
            raise Err('unhandled layout line: ' + t)
        # shared memory blocks (GL_EXT_shared_memory_block) alias one another: one uint array, the other views are
        # functions over it (rt.hlsl: nvx_smv_*, nvx_smh_*, nvx_smf_*)
        m = re.match(r'shared\s+NvxSM([UVHF])\s*\{\s*(\w+)\s+(\w+)\[(.*)\];\s*\};$', t)
        if m:
            kind, size = m.group(1), m.group(4)
            words = {'U': '(%s)', 'V': '(%s) * 4u', 'H': '((%s) + 1u) / 2u', 'F': '(%s)'}[kind] % size
            keep.append('#define NVX_SMEM_WORDS_%s (%s)' % (kind, words))
            continue
        keep.append(line)
    text = '\n'.join(keep)
    blocks = re.findall(r'#define NVX_SMEM_WORDS_([UVHF])', text)
    words = None
    if blocks:
        words = 'NVX_SMEM_WORDS_%s' % blocks[-1]
        for k in reversed(blocks[:-1]):
            words = 'NVX_MAXC(NVX_SMEM_WORDS_%s, %s)' % (k, words)
    return text, modes, words


# ---------------------------------------------------------------- cooperative matrices (T1 kernels) -> AGS WMMA
# GLSL_KHR_cooperative_matrix 16x16x16 on shared memory, as the T1 templates use it, on the engine's AGS wrappers
# (engine/d3d12/ags_wmma.hlsli: matrixSharedLoad / Store, matrixConvert, matrixMultiplyAdd with f32 accumulators).
# The kernel's shared array is AGS's matrixShared; GLSL offsets and strides count f16 elements, AGS's 32-bit words.
CM_GLSL = re.compile(r'coopmat<\s*(float16_t|float)\s*,\s*gl_ScopeSubgroup\s*,\s*16\s*,\s*16\s*,\s*gl_MatrixUse(A|B|Accumulator)\s*>')
CM = re.compile(r'\bNVXCM_(float16_t|float)_(A|B|Accumulator)\b')   # CM_GLSL spellings, as one token
CODE = {('float16_t', 'A'): 'MatrixA16', ('float16_t', 'B'): 'MatrixB16', ('float16_t', 'Accumulator'): 'MatrixAcc16',
        ('float', 'Accumulator'): 'MatrixAcc32'}
LAYOUT = {'gl_CooperativeMatrixLayoutRowMajor': '0u', 'gl_CooperativeMatrixLayoutColumnMajor': '0x80u'}


def call_args(s, i):
    """the top-level arguments of the call whose '(' is at s[i], and the index after its ')'"""
    j = find_close(s, i, '(', ')')
    args, depth, start = [], 0, i + 1
    for k in range(i + 1, j):
        c = s[k]
        if c in '([':
            depth += 1
        elif c in ')]':
            depth -= 1
        elif c == ',' and depth == 0:
            args.append(s[start:k].strip())
            start = k + 1
    args.append(s[start:j].strip())
    return args, j + 1


def convert_coopmat(s):
    s = CM_GLSL.sub(r'NVXCM_\1_\2', s)   # no commas inside the type: arguments split at top-level commas
    types = {}
    for m in re.finditer(CM.pattern + r'\s+(\w+)\s*(?:\[[^\]]*\])?\s*;', s):
        types[m.group(3)] = CODE[(m.group(1), m.group(2))]

    def type_of(expr):
        m = re.match(r'(\w+)', expr)
        t = types.get(m.group(1)) if m else None
        if not t:
            raise Err('matrix operand of unknown type: ' + expr)
        return t

    def conv(expr):
        """a matrix expression: a variable, or a GLSL type conversion CMTYPE(x) -> matrixConvert"""
        m = CM.match(expr)
        if not m:
            return expr, type_of(expr)
        to = CODE[(m.group(1), m.group(2))]
        args, _ = call_args(expr, m.end())
        inner, frm = conv(args[0])
        return ('matrixConvert(%s, %s, %s)' % (frm, to, inner) if frm != to else inner), to

    out, i = [], 0
    pat = re.compile(r'\bcoopMat(Load|Store)\s*\(')
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        args, end = call_args(s, m.end() - 1)
        if len(args) != 5 or args[1] != 'smh' or args[4] not in LAYOUT:
            raise Err('coopMat%s form: %s' % (m.group(1), ', '.join(args)))
        off, stride, col = '((%s) / 2u)' % args[2], '((%s) / 2u)' % args[3], LAYOUT[args[4]]
        if m.group(1) == 'Load':
            out.append('%s = matrixSharedLoad(%s | %s, %s, %s)' % (args[0], type_of(args[0]), col, off, stride))
        else:
            expr, t = conv(args[0])
            out.append('matrixSharedStore(%s | %s, %s, %s, %s)' % (t, col, off, stride, expr))
        i = end
    s = ''.join(out)
    # remaining conversions (assignments), then declarations, then products
    out, i = [], 0
    while True:
        m = CM.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        rest = s[m.end():]
        if rest.lstrip().startswith('('):
            k = m.end() + (len(rest) - len(rest.lstrip()))
            args, end = call_args(s, k)
            inner, frm = conv(args[0])
            to = CODE[(m.group(1), m.group(2))]
            out.append('matrixConvert(%s, %s, %s)' % (frm, to, inner) if frm != to else inner)
            i = end
        else:
            out.append('Matrix16')
            i = m.end()
    s = ''.join(out)
    s = re.sub(r'\bcoopMatMulAdd\s*\(', 'matrixMultiplyAdd(', s)
    if 'coopmat' in s or 'coopMat' in s:
        raise Err('unconverted cooperative matrix use')
    return s


ARG = r'((?:[^()]|\([^()]*\))*)'   # an argument with at most one level of nested parentheses


def find_close(s, i, open_ch, close_ch):
    """index of the bracket closing the one at s[i]"""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_ch:
            depth += 1
        elif s[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j
    raise Err('unbalanced ' + open_ch)


def rewrite_view(s, name, kind):
    """name[i] = x;  ->  nvx_<kind>_st(i, x);   name[i]  ->  nvx_<kind>_ld(i)   (an aliased shared-memory view)"""
    out, i = [], 0
    pat = re.compile(r'\b%s\[' % name)
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            return ''.join(out)
        out.append(s[i:m.start()])
        close = find_close(s, m.end() - 1, '[', ']')
        idx = s[m.end():close]
        rest = s[close + 1:]
        a = re.match(r'\s*=(?!=)\s*', rest)
        if a:
            semi = rest.index(';', a.end())
            out.append('nvx_%s_st(%s, %s);' % (kind, idx, rest[a.end():semi]))
            i = close + 1 + semi + 1
        elif re.match(r'\s*\.[xyzw]+\s*=(?!=)', rest):
            raise Err('swizzled store into ' + name)
        else:
            out.append('nvx_%s_ld(%s)' % (kind, idx))
            i = close + 1


PTR = re.compile(r'\b(atomic([A-Za-z]+)|Nvx[A-Za-z0-9]+)\s*\(')


def rewrite_ptr(s):
    """NvxT(a).v = x;  ->  nvx_st_NvxT(a, x);   atomicOp(NvxT(a).v, x)  ->  nvx_atomOp_NvxT(a, x)
    NvxT(a).v  ->  nvx_ld_NvxT(a); addresses and values are rewritten recursively"""
    out, i = [], 0
    while True:
        m = PTR.search(s, i)
        if not m:
            out.append(s[i:])
            return ''.join(out)
        out.append(s[i:m.start()])
        open_at = m.end() - 1
        if m.group(2):   # an atomic: only those on a pointer operand are ours (shared memory ones are converted later)
            args, end = call_args(s, open_at)
            p = re.match(r'(Nvx[A-Za-z0-9]+)\s*\(', args[0])
            if not p:
                out.append(s[m.start():open_at + 1])
                i = open_at + 1
                continue
            op = m.group(2)
            if op not in ATOM:
                raise Err('atomic ' + op)
            close = find_close(args[0], p.end() - 1, '(', ')')
            if args[0][close + 1:].strip() != '.v':
                raise Err('atomic operand form: ' + args[0])
            addr = rewrite_ptr(args[0][p.end():close])
            rest = ', '.join(rewrite_ptr(a) for a in args[1:])
            out.append('nvx_atom%s_%s(%s, %s)' % (op, p.group(1), addr, rest))
            i = end
            continue
        close = find_close(s, open_at, '(', ')')
        after = s[close + 1:]
        if not re.match(r'\.v\b', after):   # not an accessor (a cast to a buffer reference type is not used)
            out.append(s[m.start():open_at + 1])
            i = open_at + 1
            continue
        addr = rewrite_ptr(s[open_at + 1:close])
        a = re.match(r'\.v\s*=(?!=)\s*', after)
        if a:
            semi = after.index(';', a.end())
            out.append('nvx_st_%s(%s, %s);' % (m.group(1), addr, rewrite_ptr(after[a.end():semi])))
            i = close + 1 + semi + 1
        else:
            out.append('nvx_ld_%s(%s)' % (m.group(1), addr))
            i = close + 1 + 2


def convert_body(body):
    s = body
    # memory through pointers: NvxT(a).v as a store, an atomic's operand or a load (balanced, nested in any order)
    s = rewrite_ptr(s)
    # aliased shared-memory views
    for name, kind in (('smv', 'smv'), ('smh', 'smh'), ('smf', 'smf')):
        s = rewrite_view(s, name, kind)

    # shared-memory atomics: GLSL returns the old value, HLSL writes it to an out argument
    def atom_shared(m):
        res, op, dst, args = m.group(1), m.group(2), m.group(3), m.group(4)
        if op not in ATOM:
            raise Err('shared atomic ' + op)
        out = ', ' + res if res else ''
        return 'Interlocked%s(%s, %s%s);' % (ATOM[op], dst, args, out)
    s = re.sub(r'(?:(\w+)\s*=\s*)?\batomic([A-Za-z]+)\(\s*(smem\[[^\]]*\])\s*,\s*([^;]*)\);', atom_shared, s)
    if re.search(r'\batomic[A-Z]', s):
        raise Err('unconverted atomic: ' + re.search(r'.*\batomic[A-Z].*', s).group(0).strip())

    # f32 -> f16 conversions (PTX cvt.rn): exact RNE
    s = re.sub(r'\bfloat16_t\s*\(', 'nvx_cvt_f16(', s)
    s = re.sub(r'\bf16vec([234])\s*\(', r'nvx_mk_f16v\1(', s)
    # 64-bit literals
    s = re.sub(r'\b((?:0x[0-9A-Fa-f]+)|\d+)ul\b', r'\1ull', s)
    s = re.sub(r'\b((?:0x[0-9A-Fa-f]+)|\d+)UL\b', r'\1ull', s)
    # control flow attributes and barriers
    s = s.replace('[[dont_flatten]]', '[branch]').replace('[[flatten]]', '[flatten]')
    s = s.replace('[[unroll]]', '[unroll]').replace('[[dont_unroll]]', '[loop]')
    s = re.sub(r'\bbarrier\(\)', 'GroupMemoryBarrierWithGroupSync()', s)
    s = re.sub(r'\bmemoryBarrierShared\(\)', 'GroupMemoryBarrier()', s)
    s = re.sub(r'\bmemoryBarrier(?:Buffer|Image)?\(\)', 'DeviceMemoryBarrier()', s)
    s = re.sub(r'\bgroupMemoryBarrier\(\)', 'GroupMemoryBarrier()', s)
    # a wave runs in lockstep: subgroup execution barriers need nothing, subgroup shared-memory ordering a group fence
    s = re.sub(r'\bsubgroupBarrier\(\);', ';', s)
    s = re.sub(r'\bsubgroupMemoryBarrier(?:Shared)?\(\)', 'GroupMemoryBarrier()', s)
    s = re.sub(r'\bcontrolBarrier\(gl_ScopeWorkgroup,\s*gl_ScopeWorkgroup,[^;]*\);', 'AllMemoryBarrierWithGroupSync();', s)
    if re.search(r'\bcontrolBarrier\b|\bsubgroupBarrier\b', s):
        raise Err('unhandled barrier form')
    s = re.sub(r'^void main\(\)', 'void nvx_main()', s, flags=re.M)
    if 'nvx_main' not in s:
        raise Err('no main()')
    return s


ENTRY = '''
// A flat group: AMD's D3D12 driver packs 2D / 3D groups into waves in tiles (a 32x8 group's waves are 16x2), while the
// kernels rely on Vulkan's subgroups of consecutive invocation indices (subgroup ids, shuffles, matrix operations).
// A 1D group's waves are consecutive (measured), so the local id is rebuilt from the flat index.
[RootSignature(NVX_RS)]
[numthreads(NVX_BX * NVX_BY * NVX_BZ, 1, 1)]
[WaveSize(32)]
void main(uint3 nvx_g : SV_GroupID, uint nvx_i : SV_GroupIndex)
{
    nvx_gid = nvx_g;
    nvx_lid = uint3(nvx_i % NVX_BX, (nvx_i / NVX_BX) % NVX_BY, nvx_i / (NVX_BX * NVX_BY));
    nvx_ngroups = uint3(nvx_ext.gx, nvx_ext.gy, nvx_ext.gz);
    nvx_main();
}
'''


def convert(src):
    prefix, body = split_kernel(src)
    pre, modes, words = convert_prefix(prefix)
    pre = re.sub(r'^\s*shared\b', 'groupshared', pre, flags=re.M)
    wmma = 'coopmat' in body
    if wmma:
        if not words:
            raise Err('cooperative matrices without shared memory blocks')
        body = convert_coopmat(body)
    out = ['// generated by glsl2hlsl.py from a ptxc.py kernel; do not edit',
           '#ifndef NVX_SMEM_DYN', '#define NVX_SMEM_DYN 0u', '#endif']
    defines = [l for l in pre.splitlines() if l.strip().startswith('#')]
    rest = [l for l in pre.splitlines() if not l.strip().startswith('#')]
    out += defines
    out.append('#include "rt.hlsl"')
    if wmma:
        # AGS's shared-memory matrix operations work on its own groupshared array: the kernel's shared memory is it
        out += ['#define D4R_WMMA_SHARED_WORDS (%s)' % words, '#ifdef NVX_WMMA_EMULATE', '#include "wmma_emul.hlsli"',
                '#else', '#include "ags_wmma.hlsli"', '#endif', '#define smem matrixShared']
    elif words:
        out.append('groupshared uint smem[%s];' % words)
    out += rest
    if re.search(r'\bsm[vhf]\[', body):
        out.append('#include "rt_smem.hlsli"')
    out.append(convert_body(body))
    out.append(ENTRY)
    flags = ['-T', 'cs_6_6', '-E', 'main', '-enable-16bit-types', '-HV', '2021']
    if (4460, 32) in modes:
        flags += ['-denorm', 'ftz']
    return '\n'.join(out), flags, modes


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src = open(sys.argv[1], encoding='utf-8').read()
    try:
        text, flags, modes = convert(src)
    except Err as e:
        sys.exit('glsl2hlsl: %s: %s' % (sys.argv[1], e))
    open(sys.argv[2], 'w', encoding='utf-8', newline='\n').write(text)
    print(' '.join(flags))


if __name__ == '__main__':
    main()
