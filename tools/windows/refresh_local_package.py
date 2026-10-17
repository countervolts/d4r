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
    parser.add_argument('--rr-tools', required=True, type=lambda value: Path(value).resolve())
    parser.add_argument('--native-fix', type=lambda value: Path(value).resolve(),
                        help='directory of rebuilt gfx1101 K layer .hsaco files that replace same-named packaged objects')
    parser.add_argument('--fsr-payload', type=lambda value: Path(value).resolve(),
                        help='build_fsr_payload.py output to add as d4r/fsr (experimental FSR RR; AMD denoiser never bundled)')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    build = (repo / 'build').resolve()
    for path in (args.output, args.archive):
        if not path.is_relative_to(build) or path.exists():
            parser.error('use fresh output/archive paths under ignored build/')
    if args.archive.is_relative_to(args.output):
        parser.error('archive must be outside the package directory')
    base_hash = sha(args.base_zip)
    # verified local ZIPs this refresher may start from: v3, and v8 (build/d4r-windows-gfx1101-local-v8.zip)
    if base_hash not in ('e93be288f4013616cc2229be0132b40395d5031d8733bfa63886ba75ce1389ca',
                         '962530d1a914268cc6f9d2ec908185f8e4d9e5baafca6800d77ebc3281a647fe'):
        parser.error('base must be a verified local ZIP (v3 or v8)')
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
    native_report = None
    if args.native_fix:
        # Replace named K layer code objects with rebuilt ones (e.g. the pwin_pos.h barrier fix). Each must already be part
        # of the package's gfx1101 native folder and must be a gfx1101 AMDGPU ELF (e_machine 224, EF_AMDGPU_MACH byte 70).
        replaced = {}
        for source in sorted(args.native_fix.glob('*.hsaco')):
            relative = 'd4r/native/gfx1101/' + source.name
            data = source.read_bytes()
            if relative not in records or data[:4] != b'\x7fELF' or data[4] != 2 or data[18] != 0xe0 or data[48] != 70:
                raise RuntimeError('Native fix is not a replacement gfx1101 code object: ' + source.name)
            shutil.copy2(source, args.output / relative)
            replaced[relative] = {'before': records[relative]['sha256'], 'after': sha(source).upper()}
            records[relative] = {'path': relative, 'sha256': sha(source).upper(), 'version': None}
        if not replaced:
            raise RuntimeError('--native-fix directory has no .hsaco files')
        native_report = replaced
    installer = repo / 'tools/windows/install_dependencies.py'
    shutil.copy2(installer, args.output / 'install-dependencies.py')
    records['install-dependencies.py'] = {'path': 'install-dependencies.py', 'sha256': sha(installer).upper(), 'version': None}
    fsr_report = None
    if args.fsr_payload:
        # Experimental FSR RR: d4r-built vkd3d-proton core (LGPL, SOURCE.txt is the source offer) and d4r's amdxc64 shim.
        payload = json.loads((args.fsr_payload / 'fsr-payload.json').read_text(encoding='utf-8'))
        if payload.get('kind') != 'd4r-fsr-payload' or payload.get('amdDenoiserBundled') is not False:
            raise RuntimeError('Unsupported FSR payload manifest')
        additions = {}
        for name, digest in payload['files'].items():
            source = args.fsr_payload / name
            if sha(source) != digest:
                raise RuntimeError('FSR payload file changed: ' + name)
            if name.startswith('fsr/'):
                additions['d4r/' + name] = source
            elif name.startswith('licenses/'):
                additions['d4r/' + name] = source
            elif name == 'SOURCE.txt':
                additions['d4r/fsr/SOURCE.txt'] = source
        for patch in sorted((repo / 'patches/vkd3d-proton-windows').glob('*.patch')):
            additions['d4r/fsr/patches/' + patch.name] = patch
        for relative, source in additions.items():
            if relative in records:
                raise RuntimeError('FSR payload path already in package: ' + relative)
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            records[relative] = {'path': relative, 'sha256': sha(source).upper(), 'version': None}
        fsr_report = {'payloadManifestSha256': sha(args.fsr_payload / 'fsr-payload.json'), 'files': sorted(additions),
                      'vkd3dProtonCommit': payload['vkd3dProtonCommit'], 'amdDenoiserBundled': False}
    # Consume the pristine archive, never an extracted/used tool directory.
    if sha(args.rr_tools) != '297289a57a82142eb952479c99112b1c352a6753eb15bc6b2431bf686aed9b22':
        raise RuntimeError('Requires verified clean portable-tools v2 ZIP')
    with zipfile.ZipFile(args.rr_tools) as tools:
        tool_manifest = json.loads(tools.read('rr-tools.json'))
        expected_files = tool_manifest['files']
        names = [entry.filename for entry in tools.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)) or set(names) != set(expected_files) | {'rr-tools.json'}:
            raise RuntimeError('Portable-tools ZIP entries differ from manifest')
        for name in names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name:
                raise RuntimeError('Unsafe portable-tools path')
            data = tools.read(name)
            digest = hashlib.sha256(data).hexdigest()
            if name in expected_files and digest != expected_files[name]:
                raise RuntimeError('Portable-tools hash mismatch: ' + name)
            relative = 'd4r/model-tools/' + name
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            records[relative] = {'path': relative, 'sha256': digest.upper(), 'version': None}
    prohibited = {'nvngx_dlss.dll', 'nvngx_dlssd.dll', 'amd_fidelityfx_denoiser_dx12.dll'}
    for path in args.output.rglob('*'):
        if path.is_file() and (path.name.lower() in prohibited or path.suffix.lower() in ('.spv', '.ptx', '.cubin', '.weights')):
            raise RuntimeError('Proprietary/model artifact cannot be bundled: ' + str(path))
    manifest.update(localDevelopment=True, packageWorkingTreeDirty=True, binarySourceCommit=None,
                    packageCommit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                    files=list(records.values()),
                    localRefresh={'baseZipSha256': base_hash, 'binaryProvenance': 'local modified build; hashes authoritative',
                                  'overlays': {name: sha(path) for name, path in overlays.items()},
                                  'rrToolsZipSha256': sha(args.rr_tools),
                                  'installerSha256': sha(installer),
                                  'fsrPayload': fsr_report, 'nativeFix': native_report,
                                  'rr': 'experimental synthetic validation; optional local D/E model generation; disabled by default'})
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
