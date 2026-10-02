"""Build a private RDNA4 texture override from the user's local DLSS DLL.

The resulting PTX/code object contains NVIDIA code and must remain local.
No Wine, shell script, GPU, CUDA SDK or system installation is required.
"""
import argparse
import ast
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys


def require_code_object_target(path, target):
    with path.open('rb') as file:
        header = file.read(64)
    if (len(header) != 64 or header[:6] != b'\x7fELF\x02\x01' or
            header[18:20] != b'\xe0\x00' or header[48] != {'gfx1200': 0x48, 'gfx1201': 0x4e}[target]):
        raise RuntimeError(f'Code object does not target {target}: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dlss-dll', type=pathlib.Path, required=True)
    parser.add_argument('--hip-root', type=pathlib.Path, required=True)
    parser.add_argument('--zluda-root', type=pathlib.Path, required=True)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--output-directory', type=pathlib.Path, required=True)
    parser.add_argument('--gpu-arch', choices=['gfx1200', 'gfx1201'], default='gfx1201')
    parser.add_argument('--shader-mode', choices=['cu', 'wgp'], default='cu')
    args = parser.parse_args()
    if not re.fullmatch(r'hiluma_engine_(?:output_depth(?:inv|reg)_mv(?:hi|lo)_(?:hdr|ldr)(?:_max)?|'
                        r'input_depth(?:inv|reg)_mv(?:hi|lo)_(?:hdr|ldr))_v2_rel', args.kernel):
        parser.error('this private experimental build recipe supports K v2 input/output variants only')
    repo = pathlib.Path(__file__).resolve().parents[2]
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for existing in output.glob('*.hsaco'):
        require_code_object_target(existing, args.gpu_arch)
    work = output / ('work-' + args.kernel)
    work.mkdir(exist_ok=True)
    hip = args.hip_root.resolve()
    zluda = args.zluda_root.resolve()
    tools = {name: hip / 'bin' / (name + '.exe') for name in ('clang++', 'llvm-dis', 'llvm-as')}
    emitter = zluda / 'd4r_emit.exe'
    for path in [args.dlss_dll, emitter, *tools.values()]:
        if not path.is_file():
            parser.error(f'missing dependency: {path}')
    env = os.environ.copy()
    env.update(D4R_PREFER_ACCURACY='1', D4R_ZLUDA_WMMA='1', D4R_ZLUDA_WMMA_FP8='1',
               D4R_ZLUDA_WMMA_FP8_NATIVE='0', D4R_ZLUDA_WMMA_F16_REFERENCE='1',
               D4R_ZLUDA_WGP='1' if args.shader_mode == 'wgp' else '0',
               D4R_ZLUDA_IMPLICIT_MAX_BLOCK='256')
    for name in ('D4R_ZLUDA_IGNORE_DENORMAL', 'D4R_ZLUDA_FAST_MATH', 'D4R_ZLUDA_WMMA_F32ACC',
                 'D4R_ZLUDA_WAVE64', 'D4R_ZLUDA_PROFILE'):
        env.pop(name, None)
    env['PATH'] = str(zluda) + os.pathsep + str(hip / 'bin') + os.pathsep + env['PATH']

    def run(label, command):
        with (work / (label + '.log')).open('wb') as log:
            result = subprocess.run([str(arg) for arg in command], env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((work / (label + '.log')).read_text(errors='replace')[-8000:], file=sys.stderr)
            raise RuntimeError(f'{label} failed ({result.returncode}); see {work}')

    ptx = work / 'ptx'
    run('extract', [sys.executable, repo / 'kernels/tools/extract_dlss_ptx.py', args.dlss_dll, ptx])
    env['D4R_DLSS_PTX_DIR'] = str(ptx)
    edited = work / (args.kernel + '.ptx')
    run('make-ptx', [sys.executable, repo / 'kernels/tex/make_ptx.py', args.kernel, edited])
    rewrite_match = re.search(r'surface stores replaced: (\{[^\n]+\})',
                              (work / 'make-ptx.log').read_text(errors='replace'))
    if not rewrite_match:
        raise RuntimeError('PTX recipe did not report its changes')
    rewrite_counts = ast.literal_eval(rewrite_match[1])
    if not any(rewrite_counts.values()):
        raise RuntimeError('PTX recipe made no changes; refuse a no-op override')
    raw, ir, extra = work / 'raw.bc', work / 'raw.ll', work / 'extra.bc'
    run('compile-helper', [tools['clang++'], '-x', 'hip', '-std=c++20', '-nogpuinc', '-nogpulib', '-O3',
                          '-mno-wavefrontsize64', '--offload-device-only', '--offload-arch=' + args.gpu_arch,
                          '-fgpu-rdc', '-emit-llvm', '-c', '-Xclang', '-fdenormal-fp-math=dynamic',
                          '-DD4R_ACCURACY', '-DD4R_KERNEL_NAME=' + args.kernel,
                          '-o', raw, repo / 'kernels/tex/sust_only.hip'])
    run('disassemble', [tools['llvm-dis'], raw, '-o', ir])
    lines = []
    for line in ir.read_text().splitlines():
        if re.search(r'@llvm.used|wchar_size|llvm.module.flags|__hip_cuid', line):
            continue
        line = line.replace('optnone', '').replace(f'"target-cpu"="{args.gpu_arch}"', '')
        lines.append(re.sub(r'"target-features"="[^"]+"', '', line))
    ir.write_text('\n'.join(lines) + '\n')
    run('assemble', [tools['llvm-as'], ir, '-o', extra])
    env['D4R_ZLUDA_EXTRA_BC'] = str(extra)
    run('emit', [emitter, edited, work / 'emitted', args.gpu_arch])
    optimized = (work / 'emitted/opt.ll').read_text()
    definition = re.search(r'^define amdgpu_kernel .*@' + re.escape(args.kernel) + r'\(.*#(\d+)\s*\{', optimized, re.M)
    if not definition:
        raise RuntimeError('Cannot verify shader mode: kernel attributes are missing')
    attributes = re.search(r'^attributes #' + definition[1] + r' = \{([^\n]+)\}', optimized, re.M)
    expected_mode = '-cumode' if args.shader_mode == 'wgp' else '+cumode'
    if not attributes or expected_mode not in attributes[1]:
        raise RuntimeError(f'Emitter did not honor {args.shader_mode} mode; build the corresponding ZLUDA patch first')
    require_code_object_target(work / 'emitted/module.hsaco', args.gpu_arch)
    shutil.copyfile(work / 'emitted/module.hsaco', output / (args.kernel + '.hsaco'))
    run('manifest', [sys.executable, repo / 'kernels/tools/kernel_manifest.py', output, args.dlss_dll])
    metadata = dict(architecture=args.gpu_arch, accuracy=True, nativeFP8=False, shaderMode=args.shader_mode,
                    lastBuiltKernel=args.kernel, lastRecipeRewrites=rewrite_counts,
                    privateNvidiaDerivedCode=True, validation='not yet validated; do not install before comparison',
                    dlssSha256=hashlib.sha256(args.dlss_dll.read_bytes()).hexdigest(),
                    zludaBuild=json.loads((zluda / 'build-info.json').read_text(encoding='utf-8-sig')),
                    compiler=subprocess.check_output([str(tools['clang++']), '--version'], env=env).decode(errors='replace'),
                    objects={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.hsaco')})
    (output / 'build-info.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Built PRIVATE {args.gpu_arch} texture override: {output}; validation required before use')


if __name__ == '__main__':
    main()
