"""Refresh the verified gfx1101 development ZIP. Never creates a public release."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-zip', 'shim', 'output', 'archive'):
        parser.add_argument('--' + name, required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    build = (repo / 'build').resolve()
    for path in (args.output, args.archive):
        if not path.is_relative_to(build) or path.exists():
            parser.error('use fresh output/archive paths under ignored build/')
    if args.archive.is_relative_to(args.output):
        parser.error('archive must be outside the package directory')
    base_hash = sha(args.base_zip)
    if base_hash != 'e93be288f4013616cc2229be0132b40395d5031d8733bfa63886ba75ce1389ca':
        parser.error('base must be the verified local v3 ZIP')
    with zipfile.ZipFile(args.base_zip) as base:
        manifest = json.loads(base.read('d4r/package.json').decode('utf-8-sig'))
        if not manifest.get('localDevelopment') or set(manifest['targets']) != {'gfx1101'}:
            raise RuntimeError('Unexpected base package identity')
        records = {entry['path']: entry for entry in manifest['files']}
        names = [entry.filename for entry in base.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)) or set(names) != set(records) | {'d4r/package.json'}:
            raise RuntimeError('Base ZIP entries differ from manifest')
        for name in names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name:
                raise RuntimeError('Unsafe ZIP path: ' + name)
            data = base.read(name)
            if name in records and hashlib.sha256(data).hexdigest().upper() != records[name]['sha256'].upper():
                raise RuntimeError('Base ZIP hash mismatch: ' + name)
            destination = args.output.joinpath(*parts.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    overlays = {
        'd4r/_nvngx_gfx1101.dll': args.shim,
        'd4r/d4r.ini': repo / 'packaging/windows/d4r.ini',
        'WINDOWS_README.txt': repo / 'packaging/windows/WINDOWS_README.txt',
        'd4r/d4r-check.ps1': repo / 'scripts/windows/d4r-check.ps1',
        'install.bat': repo / 'packaging/windows/install.bat',
        'install-dlss.ps1': repo / 'scripts/windows/install-dlss.ps1',
        'd4r/dll-pins.txt': repo / 'tools/windows/dll-pins.txt',
    }
    for name, source in overlays.items():
        if name not in records:
            raise RuntimeError('Overlay is not part of the validated package layout: ' + name)
        shutil.copy2(source, args.output / name)
        records[name] = {'path': name, 'sha256': sha(source).upper(), 'version': None}
    prohibited = {'nvngx_dlss.dll', 'nvngx_dlssd.dll'}
    for path in args.output.rglob('*'):
        if path.is_file() and (path.name.lower() in prohibited or path.suffix.lower() in ('.spv', '.ptx', '.cubin', '.weights')):
            raise RuntimeError('Proprietary/model artifact cannot be bundled: ' + str(path))
    manifest.update(localDevelopment=True, packageWorkingTreeDirty=True, binarySourceCommit=None,
                    packageCommit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                    files=list(records.values()),
                    localRefresh={'baseZipSha256': base_hash, 'binaryProvenance': 'local modified build; hashes authoritative',
                                  'overlays': {name: sha(path) for name, path in overlays.items()},
                                  'rr': 'experimental synthetic validation; disabled by default; models generated separately'})
    manifest['targets']['gfx1101']['binarySourceCommit'] = None
    manifest['targets']['gfx1101']['hardwareValidation'] = 'SR prior limited evidence; native RR synthetic NGX lifecycle only; no game RR or performance parity'
    (args.output / 'd4r/package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(args.output).as_posix())
    with zipfile.ZipFile(args.archive) as archive:
        for name, entry in records.items():
            if hashlib.sha256(archive.read(name)).hexdigest().upper() != entry['sha256'].upper():
                raise RuntimeError('Refreshed ZIP hash mismatch: ' + name)
    report = {'zip': str(args.archive), 'sha256': sha(args.archive), 'bytes': args.archive.stat().st_size,
              'allHashesVerified': True, 'privateOverrideBundled': False, 'rrModelBundled': False,
              'sourceCommitCreated': False, 'published': False, 'baseZipSha256': base_hash}
    args.archive.with_suffix('.validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
