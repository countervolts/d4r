#!/usr/bin/env python3
"""build_hlsl.py CACHE: the DXIL route's kernel sources, for native Windows (cubin_d3d12.cpp, [Host] Route=dxil).

Converts every translated kernel of the cache (CACHE/kernels/<module>/<entry>[.fa][.hw][.t1].comp, the files the
Vulkan route compiles) with glsl2hlsl.py into CACHE/hlsl/<module8>_<entry>[.fa][.hw].hlsl, next to the runtime it
includes (rt.hlsl, rt_smem.hlsli, ags_wmma.hlsli copied into CACHE/hlsl). A T1 file replaces the translated one of its
variant, as in the Vulkan cache. The host compiles each source per block shape on first use (dxcompiler.dll) and keeps
the DXIL in CACHE/dxil. The generated sources derive from the translated kernels: they stay in the local cache.
A kernel that does not convert is listed in CACHE/hlsl/failed.txt and keeps the Vulkan route.
"""
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glsl2hlsl  # noqa: E402

RUNTIME = [os.path.join(HERE, 'rt.hlsl'), os.path.join(HERE, 'rt_smem.hlsli'), os.path.join(HERE, 'wmma_emul.hlsli'),
           os.path.join(HERE, '..', '..', '..', 'engine', 'd3d12', 'ags_wmma.hlsli')]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cache = sys.argv[1]
    out = os.path.join(cache, 'hlsl')
    os.makedirs(out, exist_ok=True)
    for f in RUNTIME:
        shutil.copy(f, out)
    ok, failed = 0, []
    for d in sorted(glob.glob(os.path.join(cache, 'kernels', '*', ''))):
        module = os.path.basename(os.path.dirname(d))
        for f in sorted(glob.glob(d + '*.comp')):
            b = os.path.basename(f)
            entry = b.split('.')[0]
            parts = b.split('.')[1:-1]
            variant = ''.join('.' + p for p in parts if p in ('fa', 'hw'))
            if 't1' not in parts and os.path.exists(os.path.join(d, entry + variant + '.t1.comp')):
                continue
            name = '%s_%s%s' % (module[:8], entry, variant)
            try:
                text, flags, modes = glsl2hlsl.convert(open(f, encoding='utf-8').read())
            except glsl2hlsl.Err as e:
                failed.append('%s: %s' % (name, e))
                continue
            # the dxc flags the kernel's float modes need, for the host (first line)
            open(os.path.join(out, name + '.hlsl'), 'w', encoding='utf-8', newline='\n').write(
                '// dxc: %s\n%s' % (' '.join(flags), text))
            ok += 1
    open(os.path.join(out, 'failed.txt'), 'w', encoding='utf-8', newline='\n').write('\n'.join(failed) + '\n')
    print('build_hlsl: %d kernel sources in %s, %d not converted (failed.txt)' % (ok, out, len(failed)))


if __name__ == '__main__':
    main()
