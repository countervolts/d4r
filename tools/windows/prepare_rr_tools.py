"""Prepare a private portable RR model-tool bundle from pinned dependencies.

Downloads/extracts tools locally. Does not install Python, use pip, modify PATH,
change policy, build a model, or install into a game.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import urllib.request
import uuid
import zipfile


DEPENDENCIES = {
    'python-3.12.10-embed-amd64.zip': (
        'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip',
        '4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3'),
    'numpy-2.4.6-cp312-cp312-win_amd64.whl': (
        'https://files.pythonhosted.org/packages/ab/ca/feab00bd44aa5fe1ad2c18f08b4d3bb92e26484b0b1d1443897809ed528c/numpy-2.4.6-cp312-cp312-win_amd64.whl',
        'd8e8286dd7cea7895157318d1b91cdacac64c479f3cbc8dce548331728484751'),
    'glslang-16.6.0-windows-x86_64-release.zip': (
        'https://github.com/KhronosGroup/glslang/releases/download/16.6.0/glslang-16.6.0-windows-x86_64-release.zip',
        '82bf434e69b9bb4829de7e2b4bc2c5e7a7861e53d66cf75e5cc70f5f694a8d9b'),
    'glslang-16.6.0-LICENSE.txt': (
        'https://raw.githubusercontent.com/KhronosGroup/glslang/16.6.0/LICENSE.txt',
        '17e70c676e1521ff3e4686f04a2053d93a7e28a33be8de7ec37ab0ff72feb677'),
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(archive, destination):
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in entry.filename or ':' in entry.filename:
                raise RuntimeError('Unsafe dependency archive path')
            if entry.is_dir():
                continue
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as data, target.open('xb') as out:
                shutil.copyfileobj(data, out)


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--archive', type=lambda p: Path(p).resolve(), help='optional clean portable-tools ZIP, before any model build')
    parser.add_argument('--download-cache', type=lambda p: Path(p).resolve(), default=repo / '.tools/downloads')
    args = parser.parse_args()
    if args.output.exists() or not args.output.is_relative_to((repo / 'build').resolve()):
        parser.error('use a fresh output directory under ignored build/')
    if args.archive and (args.archive.exists() or not args.archive.is_relative_to((repo / 'build').resolve()) or args.archive.is_relative_to(args.output)):
        parser.error('use a fresh archive under build/, outside the bundle directory')
    args.download_cache.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in DEPENDENCIES.items():
        path = args.download_cache / name
        if not path.exists():
            print('RR_TOOLS_DOWNLOAD ' + name, flush=True)
            temporary = path.with_name(name + '.part-' + uuid.uuid4().hex)
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as out:
                shutil.copyfileobj(response, out)
            if sha(temporary) != expected:
                raise RuntimeError('Dependency download hash mismatch: ' + name)
            temporary.rename(path)
        if sha(path) != expected:
            raise RuntimeError('Cached dependency changed: ' + name)
    python = args.output / 'python'
    recipe = args.output / 'recipe'
    extract(args.download_cache / 'python-3.12.10-embed-amd64.zip', python)
    extract(args.download_cache / 'numpy-2.4.6-cp312-cp312-win_amd64.whl', recipe / '.tools/python/vendor')
    glslang = recipe / '.tools/glslang-16.6.0'
    extract(args.download_cache / 'glslang-16.6.0-windows-x86_64-release.zip', glslang)
    shutil.copy2(glslang / 'bin/glslang.exe', glslang / 'bin/glslangValidator.exe')
    if sha(glslang / 'bin/glslangValidator.exe') != '9643709c02c83e26773f96b6a87f197ab2e84f78be0879ad8fb3619418b5d9ad':
        raise RuntimeError('Extracted glslang identity mismatch')
    files = subprocess.check_output(['git', 'ls-files', '-z', 'engine', 'vulkan', 'kernels/tools'], cwd=repo).decode().split('\0')
    suffixes = {'.py', '.comp', '.glsl', '.h', '.inc', '.sh'}
    sources = [repo / name for name in files if name and Path(name).suffix in suffixes]
    sources += [repo / 'engine/rr/native_kernels.txt', repo / 'tools/windows/build_rr_model.py', repo / 'tools/windows/deploy_rr_model.py', repo / 'LICENSE']
    for source in sources:
        target = recipe / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Keep isolated imports: no user site-packages or PYTHONPATH dependence.
    paths = ['python312.zip', '.', '../recipe', '../recipe/engine', '../recipe/engine/rr',
             '../recipe/kernels/tools', '../recipe/vulkan', '../recipe/tools/windows', '../recipe/.tools/python/vendor']
    (python / 'python312._pth').write_text('\n'.join(paths) + '\n')
    licenses = args.output / 'licenses'
    licenses.mkdir()
    shutil.copy2(args.download_cache / 'glslang-16.6.0-LICENSE.txt', licenses / 'glslang-LICENSE.txt')
    shutil.copy2(python / 'LICENSE.txt', licenses / 'Python-LICENSE.txt')
    for source in (recipe / '.tools/python/vendor/numpy-2.4.6.dist-info').glob('licenses/*'):
        if source.is_file():
            shutil.copy2(source, licenses / ('numpy-' + source.name))
    manifest = {'kind': 'local-rr-model-tools', 'dependencies': {name: {'url': url, 'sha256': digest}
                 for name, (url, digest) in DEPENDENCIES.items()}, 'sourceWorkingTree': 'local modified; file hashes authoritative',
                'files': {path.relative_to(args.output).as_posix(): sha(path) for path in sorted(args.output.rglob('*')) if path.is_file()},
                'modelsBundled': False, 'nvidiaBinariesBundled': False}
    (args.output / 'rr-tools.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        # Package only the recorded pristine tool inputs. Never sweep a used
        # bundle's model-build directory into a distributable archive.
        with zipfile.ZipFile(args.archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in (*manifest['files'], 'rr-tools.json'):
                archive.write(args.output / name, name)
        with zipfile.ZipFile(args.archive) as archive:
            for name, expected in manifest['files'].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise RuntimeError('Portable-tools ZIP hash mismatch: ' + name)
        print('RR_TOOLS_ARCHIVE sha256=' + sha(args.archive) + ' bytes=' + str(args.archive.stat().st_size))
    print('RR_TOOLS_READY ' + str(args.output))


if __name__ == '__main__':
    main()
