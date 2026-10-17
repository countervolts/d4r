"""Build an isolated diagnostic ZLUDA runtime using the already-built LLVM tree.

Never replaces the delivered runtime or rebuilds LLVM. Output is local under
build/. The copied runtime is for audited synthetic captures, not game use.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--output', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('use a fresh directory under ignored build/')
    if args.output.is_relative_to(args.runtime) or not 1 <= args.jobs <= 8:
        parser.error('output must be outside input runtime; jobs must be 1..8')
    source = repo / 'external/ZLUDA'
    if '"hiluma_engine_output_depthreg_mvhi_ldr_max_v2_rel" => Some(328)' not in (source / 'zluda/src/impl/d4r_replay.rs').read_text():
        parser.error('apply tools/windows/zluda-k-output-capture.patch to the prepared Windows ZLUDA source first')
    llvm = repo / 'build/zluda-llvm-windows-gnu'
    rust = repo / '.tools/rust-1.98.1-gnu'
    compiler = repo / '.tools/llvm-mingw-20260922-ucrt-x86_64'
    target = repo / 'build/zluda-rust-windows-gnu'
    for path in (llvm / 'bin/llvm-config.exe', rust / 'bin/cargo.exe', source / 'ptx/lib/zluda_ptx_impl.bc'):
        if not path.is_file():
            parser.error('prebuilt dependency missing: ' + str(path))
    args.output.mkdir(parents=True)
    artifact = target / 'x86_64-pc-windows-gnu/release/nvcuda.dll'
    if artifact.exists():
        shutil.copy2(artifact, args.output / 'previous-cargo-nvcuda.dll')
    environment = os.environ.copy()
    environment.update(CARGO_HOME=str(repo / '.tools/cargo-home'), CARGO_TARGET_DIR=str(target),
                       CARGO_BUILD_JOBS=str(args.jobs), RUSTC=str(rust / 'bin/rustc.exe'),
                       CC=str(compiler / 'bin/x86_64-w64-mingw32-clang.exe'),
                       CXX=str(compiler / 'bin/x86_64-w64-mingw32-clang++.exe'),
                       AR=str(compiler / 'bin/llvm-ar.exe'), CXXSTDLIB='c++',
                       LIBRARY_PATH=str(rust / 'lib/rustlib/x86_64-pc-windows-gnu/lib/self-contained'),
                       CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=str(compiler / 'bin/x86_64-w64-mingw32-clang.exe'),
                       CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS='-C link-arg=-static',
                       ZLUDA_LLVM_BUILD_DIR=str(llvm))
    environment['PATH'] = os.pathsep.join(str(p) for p in (rust / 'bin', compiler / 'bin',
        repo / '.tools/python/cmake/data/bin', repo / '.tools/python/bin', Path('C:/Program Files/Git/usr/bin'))) + os.pathsep + environment['PATH']
    report = dict(passed=False, diagnosticOnly=True, llvmRebuilt=False, deliveredRuntimeChanged=False,
                  originalRuntimeSha256=sha(args.runtime / 'zluda/nvcuda.dll'),
                  captureSourceSha256=sha(source / 'zluda/src/impl/d4r_replay.rs'),
                  llvmConfigSha256=sha(llvm / 'bin/llvm-config.exe'))
    try:
        command = [str(rust / 'bin/cargo.exe'), 'build', '--offline', '--locked', '--release',
                   '--target', 'x86_64-pc-windows-gnu', '-p', 'zluda']
        with (args.output / 'cargo.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False, timeout=30,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                process.wait(timeout=30)
                raise RuntimeError('Capture runtime build timed out; process tree terminated')
            if code:
                raise subprocess.CalledProcessError(code, command)
        runtime = args.output / 'runtime'
        shutil.copytree(args.runtime, runtime)
        shutil.copy2(artifact, runtime / 'zluda/nvcuda.dll')
        for name in ('libc++.dll', 'libunwind.dll'):
            shutil.copy2(compiler / 'bin' / name, runtime / 'zluda' / name)
        if sha(args.runtime / 'zluda/nvcuda.dll') != report['originalRuntimeSha256']:
            raise RuntimeError('Original runtime changed during build')
        report.update(passed=True, runtime=str(runtime), runtimeSha256=sha(runtime / 'zluda/nvcuda.dll'),
                      sourceBase=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
                      sourceWorkingTree='local modified; capture source hash recorded, no clean source assertion',
                      gpuValidation=False, published=False)
        print(json.dumps(report, indent=2))
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (args.output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
