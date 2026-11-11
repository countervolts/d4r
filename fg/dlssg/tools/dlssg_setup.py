#!/usr/bin/env python3
"""dlssg_setup.py --arch gfx11|gfx12 --cache DIR [--scan nvxrt-scan.exe] DLL...: dlssg-setup without bash (native
Windows: install.ps1 through d4r\\tools\\fg_install.py). Makes sure the nvxrt kernel cache DIR holds translations
(PTX -> SPIR-V) of every kernel in each of NVIDIA's DLSS-G DLLs, plus the T1 (hand-written) replacements that match
them: the same steps, files and stamps as tools/dlssg-setup. gfx12: RDNA4; gfx11: RDNA3 (the translator and the T1
kernels build WMMA operands in gfx11's layout). The cache's "arch" file names the layout, and a cache built for
another GPU generation is refused. Needs glslangValidator and spirv-val on PATH. Fast no-op for a DLL set up before.
Everything it writes is derived from the user's own DLLs and stays in the local cache; never publish that folder."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TR = ROOT / 'translator'
T1 = Path(os.environ.get('NVXRT_T1_DIR', ROOT / 't1'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log, env):
    with open(log, 'a', encoding='utf-8', errors='replace') as out:
        return subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, env=env).returncode


def setup(dll, arch, cache, scan, env):
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'kernels').mkdir(exist_ok=True)
    arch_file = cache / 'arch'
    if arch_file.is_file() and arch_file.read_text().strip() != arch:
        raise SystemExit(f'dlssg_setup: {cache} holds {arch_file.read_text().strip()} kernels; use another cache for {arch}')
    arch_file.write_text(arch + '\n')
    t1files = sorted(T1.glob('*.py')) if T1.is_dir() else []
    # the translator's version: a change in it (or in the T1 kernels) invalidates earlier translations
    version = hashlib.sha256()
    version.update(f'{arch} {env.get("NVXRT_CUS", "")}\n'.encode())
    for f in [TR / 'ptxc.py', TR / 'rt.glsl', TR / 't1_apply.py', *t1files]:
        if f.is_file():
            version.update(f.read_bytes())
    tver = version.hexdigest()[:16]
    digest = sha256(dll)
    stamp = cache / 'setup-done.txt'
    done = stamp.read_text().splitlines() if stamp.is_file() else []
    if f'{digest} {tver}' in done:
        return
    print(f'dlssg_setup: translating the kernels of {dll} (sha256 {digest[:16]}, translator {tver})', flush=True)
    log = cache / f'build-{digest[:16]}.log'
    log.write_text('')
    modules = cache / f'modules-{digest[:16]}.txt'
    with open(modules, 'w', encoding='utf-8') as out:
        if subprocess.run([str(scan), str(dll), str(cache)], stdout=out, env=env).returncode:
            raise SystemExit(f'dlssg_setup: {scan} could not read {dll}')
    lines = [l.split() for l in modules.read_text(encoding='utf-8', errors='replace').splitlines() if l.strip()]
    # modules whose PTX lists no kernel could not be read (DLSS-G 3.x stores most of its PTX encrypted)
    unreadable = sum(1 for l in lines if len(l) <= 2)
    if unreadable:
        raise SystemExit(f'dlssg_setup: {unreadable} CUDA modules of {dll} cannot be read (encrypted PTX, DLSS-G 3.x?); '
                         'use a DLSS-G 310.x nvngx_dlssg.dll')
    force = [] if any(l.endswith(' ' + tver) for l in done) else ['--force']
    if run([sys.executable, '-I', str(TR / 'build_cache.py'), str(cache), *force], log, env):
        raise SystemExit(f'dlssg_setup: translation failed, see {log}')
    text = log.read_text(encoding='utf-8', errors='replace')
    print(text.strip().splitlines()[-1] if text.strip() else 'dlssg_setup: translated', flush=True)
    # only this DLL's modules count: a shared cache can hold another DLL version's untranslatable kernels
    own = {l[0][:12] for l in lines if l}
    for line in text.splitlines():
        if line.startswith('FAIL') and any(re.fullmatch(r'[0-9a-f]{12}:', w) and w[:12] in own for w in line.split()):
            raise SystemExit('dlssg_setup: some kernels failed to translate (frame generation would be incomplete)')
    if t1files:
        # T1 replacements are keyed by module hash + entry; kernels of other DLL versions keep their T0 translation
        if run([sys.executable, '-I', str(TR / 't1_apply.py'), str(cache), str(T1)], log, env):
            raise SystemExit(f'dlssg_setup: T1 kernels failed, see {log}')
        # fp32-accumulation variants of the T1 WMMA kernels (ENTRY.fa.spv, used with NVXRT_ACC=fast)
        if run([sys.executable, '-I', str(TR / 't1_apply.py'), str(cache), str(T1), '--acc', 'fast'], log, env):
            raise SystemExit(f'dlssg_setup: fp32-accumulation variants failed, see {log}')
    with open(stamp, 'a', encoding='utf-8') as out:
        out.write(f'{digest} {tver}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arch', choices=('gfx11', 'gfx12'), required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--scan', type=Path, default=HERE.parent / 'nvxrt-scan.exe')
    parser.add_argument('dll', type=Path, nargs='+')
    a = parser.parse_args()
    env = dict(os.environ)
    if a.arch == 'gfx11':
        env['PTXC_LAYOUT'] = 'gfx11'
    for tool in ('glslangValidator', 'spirv-val'):
        if not any((Path(p) / (tool + ('.exe' if os.name == 'nt' else ''))).is_file() for p in env.get('PATH', '').split(os.pathsep) if p):
            raise SystemExit(f'dlssg_setup: {tool} not found on PATH')
    if not a.scan.is_file():
        raise SystemExit(f'dlssg_setup: missing {a.scan}')
    for dll in a.dll:
        setup(dll.resolve(), a.arch, a.cache, a.scan, env)


if __name__ == '__main__':
    main()
