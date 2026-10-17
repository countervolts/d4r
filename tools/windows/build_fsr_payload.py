"""Build the experimental FSR Ray Regeneration payload for native Windows: vkd3d-proton + the amdxc64 shim.

Output (a fresh directory under ignored build/):
  fsr/d4r_vkd3d_core.dll   vkd3d-proton's d3d12core.dll, built for Windows with dxil-spirv's scalar FP8 fallback
                           (upstream PR 307) and extended emulation, renamed so it cannot collide with the system's
  fsr/amdxc64.dll          d4r's capability shim (tools/windows/amdxc64_shim); never forwards to AMD's driver
  licenses/                vkd3d-proton (LGPL-2.1), dxil-spirv (MIT), dxbc-spirv licence texts
  SOURCE.txt               exact commits, patches and build flags: the LGPL source offer for the shipped binary
  fsr-payload.json         hashes of everything above
AMD's amd_fidelityfx_denoiser_dx12.dll is NOT part of the payload (installed hash-pinned by install.bat on request).
Needs network once (git clone of vkd3d-proton and its submodules), the vendored llvm-mingw, pinned glslang 16.6.0,
ninja, and meson (installed with pip --target into .tools/meson-pkg when missing).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
VKD3D_URL = 'https://github.com/HansKristian-Work/vkd3d-proton.git'
VKD3D_COMMIT = '3dfc6f07d0953b1e8b41705275c2c59cc7374fc5'
DXIL_FSR_MERGE = 'f9f6294f3a6795e071bef3a64d2bee99e7c1b7bc'  # dxil-spirv PR 307 merge (scalar FP8 fallback)
GLSLANG_SHA256 = '9643709c02c83e26773f96b6a87f197ab2e84f78be0879ad8fb3619418b5d9ad'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, cwd=None, env=None, log=None):
    print('+ ' + ' '.join(str(part) for part in command), flush=True)
    with open(log, 'ab') if log else open(os.devnull, 'wb') as out:
        result = subprocess.run([str(part) for part in command], cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {command[0]} {command[1] if len(command) > 1 else ""}; see {log}')


def git(args, cwd, log, identity=False):
    prefix = ['git', '-c', 'user.name=d4r', '-c', 'user.email=d4r@localhost'] if identity else ['git']
    run(prefix + list(args), cwd=cwd, log=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--source', type=lambda p: Path(p).resolve(),
                        help='reuse an existing vkd3d-proton clone (must be a checkout of the pinned commit); default: fresh clone under --output/_work')
    args = parser.parse_args()
    if not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('output must stay under ignored build/')
    args.output.mkdir(parents=True, exist_ok=False)
    work = args.output / '_work'
    work.mkdir()
    log = work / 'build.log'
    tools = repo / '.tools'
    mingw = tools / 'llvm-mingw-20260922-ucrt-x86_64/bin'
    glslang = tools / 'glslang-16.6.0/bin/glslangValidator.exe'
    ninja = tools / 'python/bin/ninja.exe'
    for required in (mingw / 'x86_64-w64-mingw32-clang.exe', glslang, ninja):
        if not required.is_file():
            raise RuntimeError('missing tool: ' + str(required))
    if sha(glslang) != GLSLANG_SHA256:
        raise RuntimeError('requires the pinned glslang 16.6.0')
    meson_path = tools / 'meson-pkg'
    if not (meson_path / 'mesonbuild').is_dir():
        run([sys.executable, '-m', 'pip', 'install', '--target', meson_path, 'meson'], log=log)
    source = args.source
    if source is None:
        source = work / 'vkd3d-proton'
        git(['clone', VKD3D_URL, source], work, log)
    git(['checkout', '-q', '-f', '--detach', VKD3D_COMMIT], source, log)
    git(['submodule', 'update', '--init', '--recursive', '-q'], source, log)
    if args.source is not None:
        git(['submodule', 'update', '--force', '--recursive', '-q'], source, log)
    # dxil-spirv: the accepted scalar FP8 fallback (PR 307), cherry-picked as the Linux build script does
    dxil = source / 'subprojects/dxil-spirv'
    git(['fetch', '-q', 'origin', DXIL_FSR_MERGE], dxil, log)
    listing = subprocess.run(['git', 'rev-list', '--reverse', DXIL_FSR_MERGE + '^1..' + DXIL_FSR_MERGE + '^2'], cwd=dxil,
                             capture_output=True, text=True, check=True).stdout.split()
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', DXIL_FSR_MERGE + '^2', 'HEAD'], cwd=dxil).returncode == 0
    if not ancestor:
        git(['cherry-pick', '-x'] + listing, dxil, log, identity=True)
    # libc++ include fixes (GCC pulled these in transitively; libc++ does not)
    for patch, where in (('0001-dxil-spirv-include-exception-for-libcxx.patch', dxil),
                         ('0002-dxbc-spirv-include-new-for-libcxx.patch', dxil / 'subprojects/dxbc-spirv')):
        full = repo / 'patches/vkd3d-proton-windows' / patch
        check = subprocess.run(['git', 'apply', '--check', '--reverse', str(full)], cwd=where, capture_output=True)
        if check.returncode:  # not applied yet
            git(['apply', full], where, log)
    escaped = lambda path: str(path).replace('\\', '/')
    cross = work / 'cross.txt'
    cross.write_text('\n'.join([
        '[binaries]',
        f"c = '{escaped(mingw / 'x86_64-w64-mingw32-clang.exe')}'",
        f"cpp = '{escaped(mingw / 'x86_64-w64-mingw32-clang++.exe')}'",
        f"ar = '{escaped(mingw / 'x86_64-w64-mingw32-ar.exe')}'",
        f"strip = '{escaped(mingw / 'x86_64-w64-mingw32-strip.exe')}'",
        f"windres = '{escaped(mingw / 'x86_64-w64-mingw32-windres.exe')}'",
        f"widl-mingw-tools-fallback = '{escaped(mingw / 'x86_64-w64-mingw32-widl.exe')}'",
        '', '[properties]', 'needs_exe_wrapper = true', '',
        '[host_machine]', "system = 'windows'", "cpu_family = 'x86_64'", "cpu = 'x86_64'", "endian = 'little'", '']))
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join([str(mingw), str(glslang.parent), str(ninja.parent), env['PATH']])
    env['PYTHONPATH'] = str(meson_path)
    build = work / 'build'
    mesonmain = [sys.executable, '-m', 'mesonbuild.mesonmain']
    run(mesonmain + ['setup', '--cross-file', cross, '--buildtype', 'release', '-Denable_tests=false',
                     '-Denable_extended_emulation=true', build, source], cwd=source, env=env, log=log)
    run(mesonmain + ['compile', '-C', build], cwd=source, env=env, log=log)
    fsr = args.output / 'fsr'
    fsr.mkdir()
    core = build / 'libs/d3d12core/d3d12core.dll'
    run([mingw / 'x86_64-w64-mingw32-strip.exe', '-o', fsr / 'd4r_vkd3d_core.dll', core], log=log)
    run([mingw / 'x86_64-w64-mingw32-clang.exe', '-shared', '-O2', '-Wall', '-o', fsr / 'amdxc64.dll',
         repo / 'tools/windows/amdxc64_shim/amdxc64_shim.c', '-Wl,--kill-at'], log=log)
    licenses = args.output / 'licenses'
    licenses.mkdir()
    for src, name in ((source / 'LICENSE', 'vkd3d-proton-LGPL-2.1.txt'), (source / 'COPYING', 'vkd3d-proton-COPYING.txt'),
                      (dxil / 'LICENSE.MIT', 'dxil-spirv-MIT.txt'), (dxil / 'subprojects/dxbc-spirv/LICENSE', 'dxbc-spirv-LICENSE.txt')):
        shutil.copy2(src, licenses / name)
    heads = {name: subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=path, capture_output=True, text=True, check=True).stdout.strip()
             for name, path in (('vkd3d-proton', source), ('dxil-spirv', dxil), ('dxbc-spirv', dxil / 'subprojects/dxbc-spirv'))}
    patches = {p.name: sha(p) for p in sorted((repo / 'patches/vkd3d-proton-windows').glob('*.patch'))}
    (args.output / 'SOURCE.txt').write_text('\n'.join([
        'fsr/d4r_vkd3d_core.dll is vkd3d-proton (LGPL-2.1) built for Windows by tools/windows/build_fsr_payload.py.',
        'Corresponding source: ' + VKD3D_URL + ' at ' + VKD3D_COMMIT + ',',
        'dxil-spirv with upstream PR 307 (merge ' + DXIL_FSR_MERGE + ') cherry-picked, and the patches in',
        'patches/vkd3d-proton-windows/ of the d4r repository (libc++ include fixes). Build: meson cross build with',
        'llvm-mingw clang, -Denable_extended_emulation=true, buildtype release; the DLL is stripped and renamed.',
        'Built HEADs: ' + json.dumps(heads), 'Patches: ' + json.dumps(patches),
        'fsr/amdxc64.dll is d4r original code (tools/windows/amdxc64_shim). It is not AMD software.', '']))
    files = {str(p.relative_to(args.output)).replace('\\', '/'): sha(p) for p in sorted(args.output.rglob('*'))
             if p.is_file() and '_work' not in p.relative_to(args.output).parts}
    (args.output / 'fsr-payload.json').write_text(json.dumps(dict(
        schema=1, kind='d4r-fsr-payload', amdDenoiserBundled=False, vkd3dProtonCommit=VKD3D_COMMIT, dxilSpirvFsrMerge=DXIL_FSR_MERGE,
        builtHeads=heads, patches=patches, files=files), indent=2) + '\n')
    print(json.dumps(files, indent=2))
    print('FSR_PAYLOAD_BUILT ' + str(args.output))


if __name__ == '__main__':
    main()
