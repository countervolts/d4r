#!/usr/bin/env python3
"""Translate every module.ptx found in the nvxrt kernel cache (written by the layer, or by dlssg-setup from a DLL).
usage: build_cache.py CACHE_DIR [--force] [--only NAME]"""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptxc

def main():
    cache = sys.argv[1]
    force = '--force' in sys.argv
    only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
    kdir = os.path.join(cache, 'kernels')
    ok = bad = skipped = 0
    for h in sorted(os.listdir(kdir)):
        mp = os.path.join(kdir, h, 'module.ptx')
        if not os.path.exists(mp):
            continue
        ptx = open(mp, encoding='latin-1').read()
        for name in ptxc.re.findall(r'\.entry\s+([\w$]+)', ptxc.strip_comments(ptx)):
            if only and name != only:
                continue
            base = os.path.join(kdir, h, name)
            if not force and os.path.exists(base + '.spv'):
                skipped += 1
                continue
            t0 = time.time()
            try:
                comp, meta = ptxc.translate(ptx, name)
            except (ptxc.Err, KeyError, IndexError, ValueError, RecursionError) as ex:
                print('FAIL translate %-44s %s: %s %s' % (name, h[:12], type(ex).__name__, ex))
                bad += 1
                continue
            open(base + '.comp', 'w').write(comp)
            try:
                ptxc.compile_glsl(base + '.comp', base + '.spv.tmp')
            except ptxc.Err as ex:
                print('FAIL compile   %-44s %s: %s' % (name, h[:12], ex))
                bad += 1
                continue
            os.replace(base + '.spv.tmp', base + '.spv')
            try:
                ptxc.compile_hw_variant(comp, base)
            except ptxc.Err as ex:
                print('FAIL compile   %-44s %s (fast filtering variant): %s' % (name, h[:12], ex))
                bad += 1
                continue
            open(base + '.meta', 'w').write(ptxc.with_handle_meta(meta, comp))   # + huse lines
            ok += 1
            print('ok   %-44s %s %.2fs' % (name, h[:12], time.time() - t0))
    print('translated %d, failed %d, cached %d' % (ok, bad, skipped))

# the structurizer recurses along the control-flow tree: large kernels (DLSS-G 3.x networks) need a deep stack
import threading
sys.setrecursionlimit(200000)
threading.stack_size(1 << 29)
t = threading.Thread(target=main)
t.start()
t.join()
