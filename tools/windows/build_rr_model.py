"""Build exact gfx11/gfx12 RR D/E models natively on Windows, without Git Bash.

Derived weights/shaders remain under ignored build/. Nothing is installed into
a game or bundled. Uses the same packers and compiler recipe as the Linux shell.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_fp8_matrices(path):
    """Reject an RDNA4 candidate that lost native FP8 cooperative matrices."""
    data = path.read_bytes()
    if len(data) < 20 or len(data) % 4:
        raise RuntimeError('Invalid SPIR-V: ' + str(path))
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    if words[0] != 0x07230203:
        raise RuntimeError('Invalid SPIR-V magic: ' + str(path))
    capabilities = set()
    offset = 5
    while offset < len(words):
        size, opcode = words[offset] >> 16, words[offset] & 0xffff
        if not size or offset + size > len(words):
            raise RuntimeError('Invalid SPIR-V instruction: ' + str(path))
        if opcode == 17 and size == 2:  # OpCapability
            capabilities.add(words[offset + 1])
        offset += size
    if not {4212, 4213, 6022}.issubset(capabilities):
        raise RuntimeError('Native FP8 cooperative matrices are absent: ' + str(path))


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--preset', choices=('D', 'E'), default='D')
    parser.add_argument('--gpu-arch', choices=('gfx1101', 'gfx1200', 'gfx1201'), default='gfx1101')
    parser.add_argument('--output', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--glslang', type=lambda p: Path(p).resolve(), default=repo / '.tools/glslang-16.6.0/bin/glslangValidator.exe')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('derived models must stay under ignored build/')
    if sha(args.dll) != '4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80':
        parser.error('requires pinned NVIDIA RR 310.9.1 DLL')
    if sha(args.glslang) != '9643709c02c83e26773f96b6a87f197ab2e84f78be0879ad8fb3619418b5d9ad':
        parser.error('requires pinned glslang 16.6.0 executable')
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith('D4R_RR_'):
            del env[key]
    gfx11 = args.gpu_arch.startswith('gfx11')
    env.update(D4R_RR_GFX11='1' if gfx11 else '0', D4R_RR_RELAXED='0',
               PYTHONPATH=str(repo / '.tools/python/vendor'))
    env['PATH'] = str(args.glslang.parent) + os.pathsep + env['PATH']
    numpy_version = subprocess.check_output([sys.executable, '-c', 'import numpy; print(numpy.__version__)'], env=env, text=True, timeout=30).strip()
    sources = {str(Path(__file__).relative_to(repo)).replace('\\', '/'): sha(Path(__file__))}
    for directory in ('engine', 'vulkan', 'kernels/tools'):
        for path in sorted((repo / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.comp', '.glsl', '.h', '.inc', '.sh', '.txt'):
                sources[path.relative_to(repo).as_posix()] = sha(path)
    identity = {'schema': 1, 'preset': args.preset, 'architecture': args.gpu_arch,
                'arithmetic': 'exact-gfx11' if gfx11 else 'exact-gfx12',
                'dll': sha(args.dll), 'glslang': sha(args.glslang), 'python': sha(Path(sys.executable)),
                'numpy': numpy_version, 'sources': sources}
    stamp = args.output / 'windows-native-python-cache.json'
    def products():
        return {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != stamp.name}
    if args.output.exists():
        if not args.reuse:
            parser.error('use a fresh output directory or --reuse')
        cached = json.loads(stamp.read_text())
        if cached['identity'] != identity or cached['products'] != products():
            raise RuntimeError('RR cache changed; use a fresh output directory')
        print('RR_NATIVE_CACHE_HIT preset=' + args.preset)
        return
    args.output.mkdir(parents=True)
    out = args.output
    work = out / 'native-work'
    work.mkdir()
    rr = repo / 'engine/rr'
    serial = 0
    def run(command, label):
        nonlocal serial
        serial += 1
        print(f'RR_BUILD step={serial} {label}', flush=True)
        log = work / f'{serial:02}-{label}.log'
        with log.open('wb') as stream:
            process = subprocess.Popen([str(p) for p in command], cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                else:
                    process.kill()
                process.wait(timeout=30)
                raise RuntimeError('RR build timed out; see ' + str(log))
        if code:
            raise RuntimeError(f'RR build failed ({code}); see {log}')
        return log.read_text(errors='replace').strip()
    def python(script, *arguments, label):
        return run([sys.executable, script, *arguments], label)
    python(repo / 'kernels/tools/extract_dlss_ptx.py', args.dll, work / 'ptx', label='extract')
    python(repo / 'engine/compile_rr.py', '--dll', args.dll, '--preset', args.preset, out,
           '--ptx-dir', work / 'ptx', '--network-bindings-only', label='bindings')
    python(rr / 'dll_weights.py', args.dll, work / 'dll', '--model', '0' if args.preset == 'D' else '1', label='weights')
    manifest = (out / 'rr.txt').read_text()
    arithmetic = 'gfx11' if gfx11 else 'gfx12'
    manifest = manifest.replace(f'preset {4 if args.preset == "D" else 5}\n',
                                f'preset {4 if args.preset == "D" else 5}\narithmetic {arithmetic}\n')
    defines = ['-DRR_EXACT_CONVERT'] + (['-DRR_GFX11'] if gfx11 else [])
    defines += ['-DRR_PRESET_D'] if args.preset == 'D' else []
    def native(stage, shader, include):
        nonlocal manifest
        weights = out / f'weights-{stage}m.bin'
        widening = []
        if gfx11:
            wide = int(python(rr / 'widen_weights.py', weights, label='widen-' + stage).splitlines()[-1])
            widening = [f'-DW_WIDE={wide}u']
        run([args.glslang, '--target-env', 'vulkan1.3', '-V', *defines, *widening,
             '-I' + str(include), '-I' + str(rr), '-o', out / f'{stage}_m.spv', shader], 'compile-' + stage)
        python(rr / 'spv_exact_convert.py', out / f'{stage}_m.spv', label='exact-' + stage)
        if not gfx11:
            require_fp8_matrices(out / f'{stage}_m.spv')
        pattern = rf'^stage {stage} {stage}\.spv 32 weights-{stage}\.bin (.*)$'
        def replacement(match):
            tail = match[1].removesuffix(' noopt')
            return f'stage {stage} {stage}_m.spv 32 weights-{stage}m.bin {tail}'
        manifest, count = re.subn(pattern, replacement, manifest, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError('Missing/duplicate native stage: ' + stage)
    for stage in ('enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4', 'dec3', 'dec2', 'dec1'):
        directory = work / stage
        directory.mkdir()
        shutil.copyfile(work / 'dll' / f'{stage}.npz', directory / 'weights.npz')
        python(rr / 'swin_pack.py', stage, directory / 'weights.npz', out / f'weights-{stage}m.bin', directory / 'swin_stage.h', label='pack-' + stage)
        native(stage, rr / 'swin_m.comp', directory)
    for stage in ('dec0', 'enc0'):
        directory = work / stage
        directory.mkdir()
        shutil.copyfile(work / 'dll' / f'{stage}.npz', directory / 'weights.npz')
        if stage == 'dec0':
            python(rr / 'dec0_pack.py', directory / 'weights.npz', out / 'weights-dec0m.bin', directory / 'dec0_m_w.h', label='pack-dec0')
            native(stage, rr / 'dec0_m.comp', directory)
        else:
            base = 0 if args.preset == 'D' else 4096
            python(rr / 'enc0_pack.py', directory / 'weights.npz', directory / 'enc0_m.bin', directory / 'enc0_m_w.h', str(base), label='pack-enc0')
            noise = b'' if not base else (work / 'dll/noise.bin').read_bytes()
            if len(noise) != base:
                raise RuntimeError('Invalid E noise table')
            (out / 'weights-enc0m.bin').write_bytes(noise + (directory / 'enc0_m.bin').read_bytes())
            python(rr / 'gen_enc0_m.py', out / 'enc0.generated.comp', directory / 'enc0_m.comp', label='gather-enc0')
            native(stage, directory / 'enc0_m.comp', directory)
    metadata = json.loads((out / 'model.json').read_text())
    if metadata['stages']['hkpn'].get('shader') == 'generated from the DLL':
        manifest = manifest.replace('stage hkpn hkpn.spv 32 ', 'stage hkpn hkpn.spv 64 ')
    for stage in ('enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0'):
        path = out / f'weights-{stage}.bin'
        if path.exists():
            path.unlink()
    (out / 'rr.txt').write_text(manifest, newline='\n')
    stages = [line.split() for line in manifest.splitlines() if line.startswith('stage ')]
    if len(stages) != 12 or f'arithmetic {arithmetic}' not in manifest.splitlines():
        raise RuntimeError('Incomplete native manifest')
    for fields in stages:
        if not (out / fields[2]).is_file() or not (out / fields[4]).is_file():
            raise RuntimeError('Missing stage product')
        textures = [] if fields[5] == '-' else list(map(int, fields[5].split(',')))
        surfaces = [] if fields[6] == '-' else [[int(s.split(':')[0]), s.split(':')[1]] for s in fields[6].split(',')]
        (out / fields[2]).with_suffix('.json').write_text(json.dumps({'textures': textures, 'surfaces': surfaces}) + '\n')
    stamp.write_text(json.dumps({'identity': identity, 'products': products()}, indent=2) + '\n')
    print('RR_NATIVE_MODEL_BUILT preset=' + args.preset + ' output=' + str(out))


if __name__ == '__main__':
    main()
