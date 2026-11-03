# T1: launch-group kernels. A group = a fixed sequence of launches that nvxrt replaces by
# one persistent dispatch (see nvxrt "launch groups"): the fused kernel runs member k's body for every virtual
# workgroup of member k's grid (grid-stride over the real workgroups), then a grid-wide barrier, then member k+1.
# Member bodies are the members' own T1 kernels (same arithmetic, so bit-identical by construction) with
#   gl_WorkGroupID -> nvx_wg (virtual workgroup), gl_NumWorkGroups -> nvx_ng, params read from the member's slot of
#   the group region instead of push constants, and every buffer access coherent (device scope): data written by
#   other workgroups earlier in the same dispatch must not come from a CU-local cache.
# Region (written by nvxrt before submission): [0] count of members to run, [128] grid-barrier counter,
#   [256 + 256k] member k's parameter block, [GRIDOFF + 16k] member k's launch grid (x, y, z).
import re
import ptxc

GRIDOFF = 256 + 48 * 256
SPIN = 20000           # bounded spin per grid barrier (~10-20 ms; in game late workgroups may wait for free CUs)
SPIN0 = 20000          # the first barrier (all workgroups resident) has the same bound

RT_PARAMS_OLD = [
    'uint nvx_p32(uint off) { return pc.w[off >> 2]; }',
    'uint nvx_p16(uint off) { return (pc.w[off >> 2] >> ((off & 2u) * 8u)) & 0xffffu; }',
    'uint nvx_p8(uint off) { return (pc.w[off >> 2] >> ((off & 3u) * 8u)) & 0xffu; }',
    'uint64_t nvx_p64(uint off) { return packUint2x32(uvec2(pc.w[off >> 2], pc.w[(off >> 2) + 1u])); }',
]
RT_PARAMS_NEW = '''uint64_t nvx_pb;      // parameter block of the member being executed (uniform)
uvec3 nvx_wg, nvx_ng;  // its virtual workgroup id / grid
uint nvx_p32(uint off) { return NvxPW(nvx_pb + uint64_t(off & ~3u)).v; }
uint nvx_p16(uint off) { return (nvx_p32(off) >> ((off & 2u) * 8u)) & 0xffffu; }
uint nvx_p8(uint off) { return (nvx_p32(off) >> ((off & 3u) * 8u)) & 0xffu; }
uint64_t nvx_p64(uint off) { return packUint2x32(uvec2(nvx_p32(off), nvx_p32(off + 4u))); }'''


def group_rt(rt):
    """the runtime for fused kernels: coherent buffer references, params from the region"""
    out = rt
    for o in RT_PARAMS_OLD:
        assert o in out, o
    out = out.replace(RT_PARAMS_OLD[0], RT_PARAMS_NEW)
    for o in RT_PARAMS_OLD[1:]:
        out = out.replace(o, '')
    n = 0
    def coh(m):
        nonlocal n
        n += 1
        return m.group(1) + ' coherent buffer '
    out = re.sub(r'(layout\(buffer_reference[^)]*\))\s+buffer\s+', coh, out)
    assert n >= 10, n
    # params: read-only, written by the host before submission (scalar loads)
    out = out.replace('uint nvx_opaque_zero()',
                      'layout(buffer_reference, buffer_reference_align = 4) readonly buffer NvxPW { uint v; };\n'
                      'uint nvx_opaque_zero()', 1)
    return out


def split_kernel(glsl, rt):
    """(smem words, text after the runtime) of a generated T1 kernel"""
    m = re.search(r'shared NvxSMU \{ uint smem\[(\d+)\]; \};', glsl)
    assert m, 'no NvxSMU block'
    i = glsl.index(rt) + len(rt)
    return int(m.group(1)), glsl[i:]


def member_fn(body, name):
    assert body.count('void main()') == 1
    b = body.replace('void main()', 'void %s()' % name)
    b = b.replace('gl_WorkGroupID', 'nvx_wg').replace('gl_NumWorkGroups', 'nvx_ng')
    return b


def fuse(members, rt):
    """members: list of dicts {glsl: complete T1 kernel, grid: (NX, DX, NY, DY, NZ, DZ)}; consecutive identical
    kernels share one body. Returns the fused kernel's GLSL."""
    words = 0
    fns = []          # (fname, grid) per member
    bodies = {}
    for k, m in enumerate(members):
        w, body = split_kernel(m['glsl'], rt)
        words = max(words, w)
        key = body
        if key not in bodies:
            bodies[key] = 'nvx_m%d' % len(bodies)
        fns.append((bodies[key], m['grid']))
    words = (words + 3) // 4 * 4
    src = (ptxc.HEADER +
           '#extension GL_EXT_shared_memory_block : require\n'
           '#extension GL_EXT_control_flow_attributes : require\n'
           'shared NvxSMU { uint smem[%d]; };\n'
           'shared NvxSMH { float16_t smh[%d]; };\n'
           'shared NvxSMV { uvec4 smv[%d]; };\n' % (words, words * 2, words // 4) + group_rt(rt))
    src += r'''
// grid-wide barrier between members. Bounded: a workgroup that waits SPIN polls sets the poison flag (region + 64),
// after which no barrier of this dispatch waits any more (wrong output for that frame, never a GPU hang); nvxrt sees
// the flag and disables the group. nvx_resident(): the same at the start of the dispatch with a short bound, so a
// grid that is not fully co-resident is detected within microseconds.
void nvx_gsync_(uint64_t reg, uint spin)
{
    controlBarrier(gl_ScopeWorkgroup, gl_ScopeQueueFamily,
                   gl_StorageSemanticsBuffer | gl_StorageSemanticsImage | gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
    if (gl_LocalInvocationIndex == 0u) {
        uint n = gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_NumWorkGroups.z;
        uint old = atomicAdd(NvxU32A(reg + 128ul).v, 1u, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);
        uint target = (old / n + 1u) * n;
        for (uint i = 0u; i < spin; i++) {
            if (atomicLoad(NvxU32A(reg + 128ul).v, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsAcquire) >= target)
                break;
            if ((i & 31u) == 0u && atomicLoad(NvxU32A(reg + 64ul).v, gl_ScopeQueueFamily, 0, gl_SemanticsRelaxed) != 0u)
                break;
            if (i + 1u == spin)
                atomicStore(NvxU32A(reg + 64ul).v, 1u, gl_ScopeQueueFamily, 0, gl_SemanticsRelaxed);
        }
    }
    controlBarrier(gl_ScopeWorkgroup, gl_ScopeQueueFamily,
                   gl_StorageSemanticsBuffer | gl_StorageSemanticsImage | gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
}
void nvx_gridsync(uint64_t ctr) { nvx_gsync_(ctr - 128ul, %(SPIN)du); }
void nvx_resident(uint64_t reg) { nvx_gsync_(reg, %(SPIN0)du); }
''' % {'SPIN': SPIN, 'SPIN0': SPIN0}
    for body, fname in bodies.items():
        src += member_fn(body, fname) + '\n'
    # main: members in order, each over its virtual grid
    cases = []
    distinct = []
    for k, fg in enumerate(fns):
        if fg not in distinct:
            distinct.append(fg)
    for (fname, g) in distinct:
        mask = sum(1 << k for k, fg in enumerate(fns) if fg == (fname, g))
        nx, dx, ny, dy, nz, dz = g
        cases.append('''        if (((0x%(mask)xul >> k) & 1ul) != 0ul) {
            uvec3 vg = uvec3((lg.x * %(nx)du + %(dx1)du) / %(dx)du, (lg.y * %(ny)du + %(dy1)du) / %(dy)du,
                             (lg.z * %(nz)du + %(dz1)du) / %(dz)du);
            uint tot = vg.x * vg.y * vg.z;
            for (uint v = gl_WorkGroupID.x; v < tot; v += gl_NumWorkGroups.x) {
                nvx_wg = uvec3(v %% vg.x, (v / vg.x) %% vg.y, v / (vg.x * vg.y));
                nvx_ng = vg;
                barrier();
                %(fn)s();
            }
        }''' % dict(mask=mask, nx=nx, dx=dx, dx1=dx - 1, ny=ny, dy=dy, dy1=dy - 1, nz=nz, dz=dz, dz1=dz - 1, fn=fname))
    src += '''
void main()
{
    uint64_t reg = packUint2x32(uvec2(pc.w[0], pc.w[1]));
    uint count = min(NvxPW(reg).v, %(n)du);
    nvx_resident(reg);
    for (uint k = 0u; k < count; k++) {
        if (k != 0u)
            nvx_gridsync(reg + 128ul);
        nvx_pb = reg + 256ul + uint64_t(k) * 256ul;
        uvec3 lg = uvec3(NvxPW(reg + %(go)dul + uint64_t(k) * 16ul).v, NvxPW(reg + %(go)dul + uint64_t(k) * 16ul + 4ul).v,
                         NvxPW(reg + %(go)dul + uint64_t(k) * 16ul + 8ul).v);
%(cases)s
    }
}
''' % dict(n=len(members), go=GRIDOFF, cases='\n'.join(cases))
    return src


def t1_grid(meta):
    for l in meta:
        if l.startswith('t1_grid'):
            return tuple(int(x) for x in l.split()[1:7])
    return (1, 1, 1, 1, 1, 1)


def t1_block(meta):
    for l in meta:
        if l.startswith('t1_block'):
            return tuple(int(x) for x in l.split()[1:4])
    return None
