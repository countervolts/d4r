#!/usr/bin/env python3
"""Apply T1 kernel overrides to a translated nvxrt kernel cache.
usage: t1_apply.py CACHE_DIR T1_DIR
Each T1_DIR/*.py defines HASH (module hash, as in CACHE/kernels/HASH), ENTRY and patch(glsl) -> glsl | None.
patch() edits the T0 translation (CACHE/kernels/HASH/ENTRY.comp): it must return None when the code it expects is not
there (translator changed), which keeps the T0 kernel. T1 code must be bit-identical to T0 (checked with
on captured launches before a patch is added). The patched GLSL replaces ENTRY.comp/.spv; the T0
translation stays in ENTRY.comp, the patched source goes to ENTRY.t1.comp.

t1_apply.py CACHE_DIR T1_DIR --acc fast (run after the plain pass, in its own process): fast accumulation, fp32
accumulation in the WMMA chains (not exact). The T1 generators are imported with NVX_T1_ACC=fast; every patch / group
whose source differs from the plain pass's gets ENTRY.fa.t1.comp + ENTRY.fa.spv (+ ENTRY.fa.hw.spv when it has a fast
filtering variant), groups/NAME.fa.{comp,spv}; nvxrt loads them with NVXRT_ACC=fast. Stale .fa files of patches
that no longer differ (or no longer apply) are removed, so another DLL or translator falls back to the plain kernels."""
import importlib.util, os, sys
sys.dont_write_bytecode = True
FAST = '--acc' in sys.argv and sys.argv[sys.argv.index('--acc') + 1] == 'fast'
if FAST:
    os.environ['NVX_T1_ACC'] = 'fast'   # read by the T1 generators (_conv.py) at import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptxc

def restore_t0(base, meta_lines):
    # an earlier run may have left a T1 binary: rebuild the T0 one so .spv and .meta agree
    open(base + '.meta', 'w').write(ptxc.with_handle_meta('\n'.join(meta_lines) + '\n', open(base + '.comp').read()))
    ptxc.compile_glsl(base + '.comp', base + '.spv.tmp')
    os.replace(base + '.spv.tmp', base + '.spv')
    ptxc.compile_hw_variant(open(base + '.comp').read(), base)

FA_EXTS = ('.fa.t1.comp', '.fa.spv', '.fa.hw.comp', '.fa.hw.spv')


def drop_fa(base):
    for ext in FA_EXTS:
        if os.path.exists(base + ext):
            os.remove(base + ext)


def write_fa(base, out):
    """fast-accumulation variant of a T1 kernel next to the plain one (only when the source differs)"""
    if not os.path.exists(base + '.t1.comp') or open(base + '.t1.comp').read() == out:
        drop_fa(base)
        return False
    open(base + '.fa.t1.comp', 'w').write(out)
    ptxc.compile_glsl(base + '.fa.t1.comp', base + '.fa.spv.tmp')
    os.replace(base + '.fa.spv.tmp', base + '.fa.spv')
    # fast filtering variant of the fa kernel (same rule as ptxc.compile_hw_variant)
    if os.path.basename(base) in ptxc.HW_FILTER_ENTRIES and ptxc.uses_filtered_tex(out):
        open(base + '.fa.hw.comp', 'w').write(ptxc.hw_filter_source(out))
        ptxc.compile_glsl(base + '.fa.hw.comp', base + '.fa.hw.spv.tmp')
        os.replace(base + '.fa.hw.spv.tmp', base + '.fa.hw.spv')
    else:
        for ext in ('.fa.hw.comp', '.fa.hw.spv'):
            if os.path.exists(base + ext):
                os.remove(base + ext)
    return True


def main_fast(cache, t1dir):
    sys.path.insert(0, os.path.abspath(t1dir))
    ok = same = 0
    for fn in sorted(os.listdir(t1dir)):
        if not fn.endswith('.py') or fn.startswith('_') or fn.startswith('g_'):
            continue
        spec = importlib.util.spec_from_file_location('t1fa_' + fn[:-3], os.path.join(t1dir, fn))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        base = os.path.join(cache, 'kernels', m.HASH, m.ENTRY)
        if not os.path.exists(base + '.comp'):
            continue
        src = open(base + '.comp').read()
        if hasattr(m, 'build'):
            res = m.build(src)
            out = None if res is None else res[0]
        else:
            out = m.patch(src)
        if out is None:
            drop_fa(base)
            continue
        try:
            if write_fa(base, out):
                ok += 1
                print('t1 fa: %s -> %s' % (fn, m.ENTRY))
            else:
                same += 1
        except ptxc.Err as ex:
            drop_fa(base)
            print('t1 fa: %s: compile failed (%s): plain T1 kept' % (fn, ex))
    print('t1 fa applied %d, unchanged %d' % (ok, same))
    gdir = os.path.join(cache, 'groups')
    def get_t0(h, entry):
        p = os.path.join(cache, 'kernels', h, entry + '.comp')
        return open(p).read() if os.path.exists(p) else None
    gok = 0
    for fn in sorted(os.listdir(t1dir)):
        if not (fn.startswith('g_') and fn.endswith('.py')):
            continue
        spec = importlib.util.spec_from_file_location('t1gfa_' + fn[:-3], os.path.join(t1dir, fn))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        base = os.path.join(gdir, m.NAME)
        for ext in ('.fa.comp', '.fa.spv'):
            if os.path.exists(base + ext):
                os.remove(base + ext)
        if not os.path.exists(base + '.comp') or not os.path.exists(base + '.spv'):
            continue      # the plain pass did not build this group
        res = m.build_group(get_t0)
        if res is None or res[0] == open(base + '.comp').read():
            continue
        open(base + '.fa.comp', 'w').write(res[0])
        try:
            ptxc.compile_glsl(base + '.fa.comp', base + '.fa.spv')
        except ptxc.Err as ex:
            for ext in ('.fa.comp', '.fa.spv'):
                if os.path.exists(base + ext):
                    os.remove(base + ext)
            print('t1 group fa %s: compile failed (%s): plain group kept' % (m.NAME, ex))
            continue
        gok += 1
        print('t1 group fa %s' % m.NAME)
    print('t1 groups fa built %d' % gok)


def main():
    cache, t1dir = sys.argv[1], sys.argv[2]
    if FAST:
        return main_fast(cache, t1dir)
    sys.path.insert(0, os.path.abspath(t1dir))   # shared helpers (_*.py) next to the patches
    ok = skipped = 0
    for fn in sorted(os.listdir(t1dir)):
        if not fn.endswith('.py') or fn.startswith('_') or fn.startswith('g_'):
            continue
        spec = importlib.util.spec_from_file_location('t1_' + fn[:-3], os.path.join(t1dir, fn))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        base = os.path.join(cache, 'kernels', m.HASH, m.ENTRY)
        if not os.path.exists(base + '.comp'):
            print('t1: %s: no %s in this cache (other DLL version): skipped' % (fn, m.ENTRY))
            skipped += 1
            continue
        src = open(base + '.comp').read()
        # the T0 meta without earlier T1 lines (dispatch overrides belong to a T1 kernel only)
        meta_lines = [l for l in open(base + '.meta').read().splitlines() if not l.startswith('t1_') and not l.startswith('huse')]
        extra = []
        if hasattr(m, 'build'):
            res = m.build(src)          # full replacement: (glsl, [meta lines]) or None
            out = None if res is None else res[0]
            if res is not None:
                extra = list(res[1])
        else:
            out = m.patch(src)
        if out is None:
            restore_t0(base, meta_lines)
            drop_fa(base)
            print('t1: %s: translation of %s does not match the patch: T0 kept' % (fn, m.ENTRY))
            skipped += 1
            continue
        open(base + '.t1.comp', 'w').write(out)
        try:
            ptxc.compile_glsl(base + '.t1.comp', base + '.spv.tmp')
        except ptxc.Err as ex:
            restore_t0(base, meta_lines)
            drop_fa(base)
            print('t1: %s: compile failed (%s): T0 kept' % (fn, ex))
            skipped += 1
            continue
        os.replace(base + '.spv.tmp', base + '.spv')
        try:
            ptxc.compile_hw_variant(out, base)
        except ptxc.Err as ex:
            restore_t0(base, meta_lines)
            drop_fa(base)
            print('t1: %s: fast filtering variant failed to compile (%s): T0 kept' % (fn, ex))
            skipped += 1
            continue
        open(base + '.meta', 'w').write(ptxc.with_handle_meta('\n'.join(meta_lines + extra) + '\n', out))
        ok += 1
        print('t1: %s -> %s' % (fn, m.ENTRY))
    print('t1 applied %d, skipped %d' % (ok, skipped))
    apply_groups(cache, t1dir)


def apply_groups(cache, t1dir):
    """launch groups: g_*.py define NAME, MEMBERS [(hash, entry)], GRID (x, y, z) and build_group(get_t0) ->
    (glsl, (bx, by, bz)) or None; output cache/groups/NAME.{comp,spv,meta}. Every run rebuilds the directory, so a
    group whose members changed (other DLL / translator) disappears instead of running stale code."""
    gdir = os.path.join(cache, 'groups')
    os.makedirs(gdir, exist_ok=True)
    for f in os.listdir(gdir):
        os.remove(os.path.join(gdir, f))
    def get_t0(h, entry):
        p = os.path.join(cache, 'kernels', h, entry + '.comp')
        return open(p).read() if os.path.exists(p) else None
    ok = skipped = 0
    for fn in sorted(os.listdir(t1dir)):
        if not (fn.startswith('g_') and fn.endswith('.py')):
            continue
        spec = importlib.util.spec_from_file_location('t1g_' + fn[:-3], os.path.join(t1dir, fn))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # a member is (hash, entry) or a list of alternatives [(hash, entry), ...] (nvxrt accepts any of them)
        alts = [mm if isinstance(mm, list) else [mm] for mm in m.MEMBERS]
        if any(get_t0(h, e) is None for a in alts for h, e in a):
            print('t1 group %s: members not in this cache: skipped' % m.NAME)
            skipped += 1
            continue
        res = m.build_group(get_t0)
        if res is None:
            print('t1 group %s: member translations do not match: skipped' % m.NAME)
            skipped += 1
            continue
        glsl, block = res
        base = os.path.join(gdir, m.NAME)
        open(base + '.comp', 'w').write(glsl)
        try:
            ptxc.compile_glsl(base + '.comp', base + '.spv')
        except ptxc.Err as ex:
            print('t1 group %s: compile failed (%s): skipped' % (m.NAME, ex))
            for ext in ('.comp', '.spv'):
                if os.path.exists(base + ext):
                    os.remove(base + ext)
            skipped += 1
            continue
        meta = ['member ' + ' '.join('%s %s' % (h, e) for h, e in a) for a in alts]
        meta += ['block %d %d %d' % tuple(block), 'grid %d %d %d' % tuple(m.GRID)]
        open(base + '.meta', 'w').write('\n'.join(meta) + '\n')
        ok += 1
        print('t1 group %s: %d members' % (m.NAME, len(m.MEMBERS)))
    print('t1 groups built %d, skipped %d' % (ok, skipped))

main()
