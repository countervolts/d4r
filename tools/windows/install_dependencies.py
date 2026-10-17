"""Install pinned local NVIDIA dependencies without PowerShell or driver installation.

All downloads and backups stay outside the game. Runtime replacements are staged,
hash verified and rolled back on failure. No model generation is implicit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid


PINS = {
    'dlss': ('nvngx_dlss.dll', 'dlss/nvngx_dlss.dll',
             'https://raw.githubusercontent.com/NVIDIA/DLSS/374959484e79a640feaba44c93ac8cfb0a03f5b5/lib/Windows_x86_64/rel/nvngx_dlss.dll',
             '3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983'),
    'dlssd': ('nvngx_dlssd.dll', 'rr/nvngx_dlssd.dll',
              'https://raw.githubusercontent.com/NVIDIA/DLSS/374959484e79a640feaba44c93ac8cfb0a03f5b5/lib/Windows_x86_64/rel/nvngx_dlssd.dll',
              '4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80'),
    'ngx': ('_nvngx.dll', 'ngx/_nvngx.dll', None,
            '66767018c36b3bab46398dade3adf173daa3730fda75965689ea848c9bc4e79b'),
}
# Optional: AMD's FSR Ray Regeneration denoiser (FidelityFX SDK 2.3), only for [RayReconstruction] Backend = fsr.
# AMD's file, never bundled; the same pin as scripts/windows/setup-fsr-rr.ps1.
FSR_DENOISER = ('amd_fidelityfx_denoiser_dx12.dll', 'amd_fidelityfx_denoiser_dx12.dll',
                'https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.3.0/Kits/FidelityFX/signedbin/amd_fidelityfx_denoiser_dx12.dll',
                '48f1e5888ba6a0a3d59a98b9751e37c392b0f7b8c223d0082d5c1f40642879d3')
DRIVER = ('https://us.download.nvidia.com/Windows/617.14/617.14-desktop-win10-win11-64bit-international-dch-whql.exe',
          '63d3da5ed52547bf3fd86204bb9ea1455947b24e2aa2519f359126f9a6efe13f')
SEVENZIP = ('https://github.com/ip7z/7zip/releases/download/26.04/7zr.exe',
            '256feca8e274e5da655e2a284fabafd9f554365eb164862089dacd4e8276d282')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def external(path, game):
    if path.resolve().is_relative_to(game.resolve()):
        raise RuntimeError('Cache and backups must stay outside game directory: ' + str(path))


def format_eta(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return '{}h {:02d}m {:02d}s'.format(hours, minutes, seconds)
    if minutes:
        return '{}m {:02d}s'.format(minutes, seconds)
    return str(seconds) + 's'


def copy_with_progress(response, out, name, chunk=1 << 20):
    """Copy a download to `out`, rewriting one status line with percent, speed and ETA."""
    total = int(response.headers.get('Content-Length') or 0)
    done, start, last = 0, time.monotonic(), 0.0
    while True:
        block = response.read(chunk)
        if not block:
            break
        out.write(block)
        done += len(block)
        now = time.monotonic()
        if now - last >= 0.25 or (total and done >= total):
            last = now
            speed = done / max(now - start, 1e-6)
            line = '  {} {:.1f} MiB'.format(name, done / 1048576)
            if total:
                line += ' / {:.1f} MiB ({:.0f}%)'.format(total / 1048576, 100 * done / total)
            line += ' at {:.1f} MiB/s'.format(speed / 1048576)
            if total and done < total:
                line += ', ETA ' + format_eta((total - done) / speed)
            print('\r' + line.ljust(79), end='', flush=True)
    print(flush=True)


def verified_download(path, url, expected, offline=False):
    if path.exists():
        if path.is_file() and sha(path) == expected:
            return path
        if offline:
            raise RuntimeError('Invalid offline cache: ' + str(path))
        path.rename(path.with_name(path.name + '.bad-' + uuid.uuid4().hex))
    if offline:
        raise RuntimeError('Missing offline cache: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part-' + uuid.uuid4().hex)
    try:
        print('DOWNLOAD ' + path.name, flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as out:
            copy_with_progress(response, out, path.name)
        if sha(temporary) != expected:
            raise RuntimeError('Download SHA256 mismatch: ' + path.name)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def native_fsr_package(runtime):
    """Only explicit RDNA4-only packages can omit the compatibility payload."""
    target = runtime / 'gpu-target.json'
    architectures = set()
    if target.is_file():
        architectures.add(json.loads(target.read_text(encoding='utf-8-sig')).get('architecture'))
    if (runtime / 'package.json').is_file():
        architectures.update(json.loads((runtime / 'package.json').read_text(encoding='utf-8-sig')).get('targets', {}))
    return bool(architectures) and architectures <= {'gfx1200', 'gfx1201'}


def ini_selects_fsr(runtime):
    """True when d4r.ini sets [RayReconstruction] Backend = fsr (comments and case ignored)."""
    section = ''
    for line in (runtime / 'd4r.ini').read_text(encoding='utf-8-sig').splitlines():
        line = line.split(';', 1)[0].split('#', 1)[0].strip()
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1].strip().lower()
        elif section == 'rayreconstruction' and '=' in line:
            key, value = line.split('=', 1)
            if key.strip().lower() == 'backend' and value.strip().lower() == 'fsr':
                return True
    return False


def select_backend(data, backend):
    """Return d4r.ini bytes with [RayReconstruction] Backend set to `backend`; everything else is preserved."""
    bom = data.startswith(b'\xef\xbb\xbf')
    text = data.decode('utf-8-sig')
    newline = '\r\n' if '\r\n' in text else '\n'
    result, in_rr, found, done = [], False, False, False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('['):
            if in_rr and not done:
                result.append('Backend = ' + backend + newline)
                done = True
            in_rr = stripped.split(']', 1)[0][1:].strip().casefold() == 'rayreconstruction'
            found = found or in_rr
        elif in_rr and '=' in line and not stripped.startswith((';', '#')) and line.split('=', 1)[0].strip().casefold() == 'backend':
            prefix, body = line.split('=', 1)
            ending = '\r\n' if body.endswith('\r\n') else '\n' if body.endswith('\n') else ''
            body = body[:len(body) - len(ending)]
            comment = re.search(r'\s[;#]', body)
            result.append(prefix + '= ' + backend + (body[comment.start():] if comment else '') + ending)
            done = True
            continue
        result.append(line)
    if in_rr and not done:
        if result and not result[-1].endswith(('\n', '\r')):
            result.append(newline)
        result.append('Backend = ' + backend + newline)
    elif not found:
        if result and not result[-1].endswith(('\n', '\r')):
            result.append(newline)
        result.extend((newline, '[RayReconstruction]' + newline, 'Backend = ' + backend + newline))
    return (b'\xef\xbb\xbf' if bom else b'') + ''.join(result).encode('utf-8')


def enable_fsr(runtime, backup_root):
    """Select the FSR backend in d4r.ini with an external backup; idempotent."""
    config = runtime / 'd4r.ini'
    original = config.read_bytes()
    updated = select_backend(original, 'fsr')
    if updated == original:
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / ('fsr-config-' + uuid.uuid4().hex + '.ini')
    with backup.open('xb') as stream:
        stream.write(original)
    temporary = config.with_name('d4r.ini.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(updated)
    if config.read_bytes() != original:
        temporary.unlink()
        raise RuntimeError('d4r.ini changed during installation; preserved current configuration')
    os.replace(temporary, config)
    return backup


def absolutize_library_paths(game, backup_root):
    """Make OptiScaler.ini's [Libraries] folders absolute for this game folder.

    OptiScaler builds without patch 0006 reject a relative NvngxPath ("AllowExternalBackend requires an
    absolute NvngxPath"), which leaves the AMD GPU DLSS-incapable and hides DLSS in its menu. Idempotent;
    the previous file is kept outside the game.
    """
    config = game / 'OptiScaler.ini'
    if not config.is_file():
        return None
    original = config.read_bytes()
    text = original.decode('utf-8-sig')
    section, changed, lines = '', False, []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            section = stripped[1:-1].strip().lower()
        elif section == 'libraries' and '=' in line and not stripped.startswith((';', '#')):
            key, value = line.split('=', 1)
            ending = line[len(line.rstrip('\r\n')):]
            value = value.strip()
            if (key.strip().lower() in ('nvngxpath', 'nvngxdlsspath', 'nvngxfeaturepath')
                    and value and not Path(value).is_absolute()):
                line = key.rstrip() + '=' + str((game / value).resolve()) + ending
                changed = True
        lines.append(line)
    if not changed:
        return None
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / ('optiscaler-config-' + uuid.uuid4().hex + '.ini')
    with backup.open('xb') as stream:
        stream.write(original)
    temporary = config.with_name(config.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write((b'\xef\xbb\xbf' if original.startswith(b'\xef\xbb\xbf') else b'') + ''.join(lines).encode('utf-8'))
    if config.read_bytes() != original:
        temporary.unlink()
        raise RuntimeError('OptiScaler.ini changed during installation; preserved current configuration')
    os.replace(temporary, config)
    return backup


def collect(cache, offline, fsr=False):
    assets = []
    for kind, (name, relative, url, expected) in PINS.items():
        cached = cache / 'nvidia' / name
        if kind == 'ngx' and not (cached.is_file() and sha(cached) == expected):
            if offline:
                raise RuntimeError('Missing or invalid offline NGX 617.14 core')
            driver = verified_download(cache / 'nvidia-driver-617.14.exe', *DRIVER)
            extractor = verified_download(cache / 'tools/7zr-26.04.exe', *SEVENZIP)
            with tempfile.TemporaryDirectory(prefix='d4r-ngx-') as work:
                subprocess.run([str(extractor), 'e', '-y', '-o' + work, str(driver),
                                'Display.Driver/_nvngx.dll'], check=True, timeout=600)
                core = Path(work) / name
                if sha(core) != expected:
                    raise RuntimeError('Extracted NGX core SHA256 mismatch')
                cached.parent.mkdir(parents=True, exist_ok=True)
                if cached.exists():
                    cached.rename(cached.with_name(name + '.bad-' + uuid.uuid4().hex))
                staged = cached.with_name(name + '.part-' + uuid.uuid4().hex)
                try:
                    shutil.copy2(core, staged)
                    if sha(staged) != expected:
                        raise RuntimeError('Staged core SHA256 mismatch')
                    os.replace(staged, cached)
                finally:
                    staged.unlink(missing_ok=True)
        else:
            verified_download(cached, url, expected, offline)
        assets.append((cached, relative, expected))
    if fsr:
        name, relative, url, expected = FSR_DENOISER
        assets.append((verified_download(cache / 'amd' / name, url, expected, offline), relative, expected))
    return assets


def transaction(runtime, assets, backup_root):
    """Back up every replacement before first mutation; preserve concurrent edits."""
    entries, applied, staged_files = [], [], []
    backup = backup_root / ('dlss-install-' + uuid.uuid4().hex)
    try:
        for source, relative, expected in assets:
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise RuntimeError('Invalid runtime destination')
            destination = runtime / relative
            if not destination.resolve().is_relative_to(runtime.resolve()):
                raise RuntimeError('Runtime destination escapes runtime')
            if destination.exists() and not destination.is_file():
                raise RuntimeError('DLL destination is not a file: ' + str(destination))
            previous = sha(destination) if destination.exists() else None
            if previous == expected:
                continue
            original = backup / relative
            if previous is not None:
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, original)
                if sha(original) != previous:
                    raise RuntimeError('Backup SHA256 mismatch')
            entries.append(dict(path=str(destination), sha256=previous, installHash=expected,
                                backup=str(original) if previous is not None else None))
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.with_name(destination.name + '.install-' + uuid.uuid4().hex)
            staged_files.append(staged)
            shutil.copy2(source, staged)
            if sha(staged) != expected:
                raise RuntimeError('Staged DLL SHA256 mismatch')
        if entries:
            backup.mkdir(parents=True, exist_ok=True)
            (backup / 'backup-manifest.json').write_text(json.dumps({'files': entries}, indent=2) + '\n')
        for entry, staged in zip(entries, staged_files):
            destination = Path(entry['path'])
            current = sha(destination) if destination.exists() else None
            if current != entry['sha256']:
                raise RuntimeError('Destination changed after backup: ' + str(destination))
            os.replace(staged, destination)
            applied.append(entry)
            if sha(destination) != entry['installHash']:
                raise RuntimeError('Installed DLL SHA256 mismatch')
        return {'changed': len(entries), 'backup': str(backup) if entries else None,
                'files': {relative: sha(runtime / relative) for _, relative, _ in assets}}
    except BaseException:
        errors = []
        for entry in reversed(applied):
            destination = Path(entry['path'])
            try:
                if not destination.is_file() or sha(destination) != entry['installHash']:
                    errors.append('Concurrent edit preserved: ' + str(destination))
                    continue
                if entry['backup']:
                    original = Path(entry['backup'])
                    if sha(original) != entry['sha256']:
                        raise RuntimeError('Rollback backup changed')
                    staged = destination.with_name(destination.name + '.rollback-' + uuid.uuid4().hex)
                    staged_files.append(staged)
                    shutil.copy2(original, staged)
                    os.replace(staged, destination)
                else:
                    destination.unlink()
            except Exception as error:
                errors.append(str(error))
        if errors:
            print('ROLLBACK ATTENTION: ' + '; '.join(errors), flush=True)
        raise
    finally:
        for staged in staged_files:
            staged.unlink(missing_ok=True)


def prepare_model_tools(bundle, cache):
    """Copy only verified clean inputs; generated products never become inputs."""
    data = (bundle / 'rr-tools.json').read_bytes()
    manifest = json.loads(data)
    if (manifest.get('kind') != 'local-rr-model-tools' or manifest.get('modelsBundled') is not False
            or manifest.get('nvidiaBinariesBundled') is not False):
        raise RuntimeError('Unsupported model-tool manifest')
    target = cache / ('rr-tools-' + hashlib.sha256(data).hexdigest()[:16])
    if not target.resolve().is_relative_to(cache.resolve()):
        raise RuntimeError('Model-tool cache escapes cache root')
    files = manifest['files']
    required = {'python/python.exe', 'recipe/tools/windows/build_rr_model.py',
                'recipe/tools/windows/deploy_rr_model.py', 'recipe/engine/rr/native_kernels.txt'}
    if not required.issubset(files):
        raise RuntimeError('Incomplete model-tool bundle')
    for name, expected in files.items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name:
            raise RuntimeError('Unsafe model-tool path')
        source = bundle.joinpath(*parts.parts)
        if not source.resolve().is_relative_to(bundle.resolve()) or sha(source) != expected:
            raise RuntimeError('Model-tool input changed: ' + name)
        destination = target.joinpath(*parts.parts)
        if not destination.resolve().is_relative_to(target.resolve()):
            raise RuntimeError('Model-tool cache path escaped')
        if destination.exists():
            if sha(destination) != expected:
                raise RuntimeError('Model-tool cache changed; preserved: ' + str(destination))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, destination.open('xb') as out:
                shutil.copyfileobj(src, out)
            if sha(destination) != expected:
                raise RuntimeError('Model-tool copy SHA256 mismatch')
    # A stale or used cache is never swept into a package. Build reuse validates
    # generated products separately through the existing builder's identity.
    return target


def install_model(runtime, cache, backup, preset, enable):
    metadata = json.loads((runtime / 'package.json').read_text(encoding='utf-8-sig'))
    if set(metadata['targets']) != {'gfx1101'}:
        raise RuntimeError('Automatic RR build currently requires gfx1101-only package')
    tools = prepare_model_tools(runtime / 'model-tools', cache)
    python = tools / 'python/python.exe'
    recipe = tools / 'recipe'
    output = recipe / 'build' / ('rr-' + preset.lower())
    # An interrupted build (window closed, Ctrl+C) leaves products without the builder's cache
    # stamp, which --reuse rejects on every later run. Keep it aside and build afresh.
    if output.exists() and not (output / 'windows-native-python-cache.json').is_file():
        aside = output.with_name(output.name + '.incomplete-' + uuid.uuid4().hex)
        output.rename(aside)
        print('Previous RR model build was incomplete; kept at ' + str(aside) + ' and rebuilding.', flush=True)
    # Each builder child has its own 600s deadline and kills a timed-out tree.
    subprocess.run([str(python), str(recipe / 'tools/windows/build_rr_model.py'),
                    '--dll', str(runtime / 'rr/nvngx_dlssd.dll'), '--preset', preset,
                    '--output', str(output), '--reuse'], check=True)
    command = [str(python), str(recipe / 'tools/windows/deploy_rr_model.py'),
               '--model', str(output), '--runtime', str(runtime), '--backup-root', str(backup)]
    if enable:
        command.append('--enable')
    subprocess.run(command, check=True, timeout=120)


def main():
    local = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'd4r'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--cache-root', type=lambda p: Path(p).resolve(), default=local / 'cache')
    parser.add_argument('--backup-root', type=lambda p: Path(p).resolve(), default=local / 'backups')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--rr-preset', choices=('D', 'E'), help='build and deploy optional exact gfx1101 RR model')
    parser.add_argument('--enable-rr', action='store_true', help='select generated native RR model in d4r.ini')
    parser.add_argument('--enable-fsr', action='store_true', help='install the FSR denoiser and select Backend = fsr in d4r.ini')
    parser.add_argument('--install-fsr', action='store_true',
                        help="require AMD's pinned FSR Ray Regeneration denoiser (installed automatically whenever the release has its d4r/fsr payload)")
    args = parser.parse_args()
    runtime = args.package_root / 'd4r' if (args.package_root / 'd4r/d4r.ini').is_file() else args.package_root
    if not (runtime / 'd4r.ini').is_file():
        parser.error('Extract Windows release first; runtime d4r.ini missing')
    game = runtime.parent
    external(args.cache_root, game)
    external(args.backup_root, game)
    if args.enable_rr and not args.rr_preset:
        parser.error('--enable-rr requires --rr-preset D or E')
    pins = (runtime / 'dll-pins.txt').read_text(encoding='utf-8-sig')
    for kind, (_, _, _, expected) in PINS.items():
        if not re.search(r'^\s*' + kind + r'\s+' + expected + r'\s+\S+\s+\S+\s*$', pins, re.M | re.I):
            raise RuntimeError('Release does not recognize pinned ' + kind)
    if args.enable_fsr and args.rr_preset:
        parser.error('--enable-fsr and --rr-preset select different RR backends')
    # Install every backend's files up front so switching [RayReconstruction] Backend later needs no rerun.
    # AMD's denoiser comes along whenever the release carries the FSR payload; requesting FSR without it fails.
    fsr_payload = all((runtime / 'fsr' / name).is_file() for name in ('d4r_vkd3d_core.dll', 'amdxc64.dll'))
    native_fsr = native_fsr_package(runtime)
    if (args.install_fsr or args.enable_fsr or ini_selects_fsr(runtime)) and not (fsr_payload or native_fsr):
        parser.error("FSR Ray Regeneration needs this release's d4r/fsr payload (d4r_vkd3d_core.dll, amdxc64.dll); extract the complete ZIP")
    fsr = fsr_payload or native_fsr
    if not args.yes and input('Download pinned DLSS/RR 310.9.1 and NGX 617.14 (~945 MiB archive)? [Y/n] ').strip().lower().startswith('n'):
        return
    result = transaction(runtime, collect(args.cache_root, args.offline, fsr), args.backup_root)
    print(json.dumps(result, indent=2))
    absolutize_library_paths(game, args.backup_root)
    if args.enable_fsr:
        backup = enable_fsr(runtime, args.backup_root)
        print(json.dumps({'fsrBackendSelected': True, 'configBackup': str(backup) if backup else None}, indent=2))
    if args.rr_preset:
        print('DLL installation verified. Building optional private RR model; a model failure leaves verified DLLs installed.', flush=True)
        install_model(runtime, args.cache_root, args.backup_root, args.rr_preset, args.enable_rr)


if __name__ == '__main__':
    main()
