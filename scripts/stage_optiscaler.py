#!/usr/bin/env python3
"""Stage OptiScaler v10 with its complete upstream dependency layout and licenses."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

from configure_optiscaler import configure

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('amd_fidelityfx_loader_dx12.dll', 'amd_fidelityfx_upscaler_dx12.dll',
            'amd_fidelityfx_framegeneration_dx12.dll', 'libxell.dll', 'libxess_fg.dll',
            'dlssg_to_fsr3_amd_is_better.dll')


def stage(source, destination):
    source, destination = Path(source), Path(destination)
    required = ['OptiScaler.dll', 'OptiScaler.ini', 'Licenses/FidelityFX_v2_LICENSE.md',
                'Licenses/XeSS_LICENSE.txt'] + ['OptiScaler/' + name for name in REQUIRED]
    for name in required:
        if not (source / name).is_file():
            raise ValueError(f'OptiScaler v10 release is missing {name}')
    with (source / 'OptiScaler.ini').open(encoding='utf-8-sig', newline='') as handle:
        config = configure(handle.read(), (ROOT / 'packaging/optiscaler.settings').read_text())
    # Keep all of upstream's providers, including nested interop DLLs, together.
    shutil.copytree(source / 'OptiScaler', destination / 'OptiScaler', dirs_exist_ok=True)
    shutil.copy2(source / 'OptiScaler.dll', destination / 'dxgi.dll')
    with (destination / 'OptiScaler.ini').open('w', encoding='utf-8', newline='') as handle:
        handle.write(config)
    licenses = destination / 'd4r/licenses'
    licenses.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / 'Licenses', licenses / 'OptiScaler', dirs_exist_ok=True)
    for original, name in [('FidelityFX_v2_LICENSE.md', 'FidelityFX-LICENSE.txt'),
                           ('XeSS_LICENSE.txt', 'XeSS-LICENSE.txt')]:
        shutil.copy2(source / 'Licenses' / original, licenses / name)
    lock = json.loads((ROOT / 'packaging/optiscaler-release.json').read_text())
    files = [destination / 'dxgi.dll', destination / 'OptiScaler.ini']
    files += sorted(p for p in (destination / 'OptiScaler').rglob('*') if p.is_file())
    hashes = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    incoming = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()
                and (p.relative_to(source).parts[0] in ('OptiScaler', 'Licenses')
                     or p.name in ('OptiScaler.dll', 'OptiScaler.ini'))}
    pinned = incoming == set(lock['files']) and all((source / name).is_file() and
                 hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
                 for name, digest in lock['files'].items())
    provenance = {'release': lock if pinned else {'version': 'custom v10 input'},
                  'files': hashes}
    output = destination / 'd4r/source/optiscaler.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    try:
        stage(*sys.argv[1:])
    except (OSError, ValueError) as error:
        sys.exit(str(error))
