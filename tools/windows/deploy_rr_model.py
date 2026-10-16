"""Verify and deploy a locally built RR model; optionally select it in d4r.ini.

Existing models are preserved. Configuration changes receive an external backup.
No NVIDIA DLL, compiler, driver, package archive or system setting is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid


def sha(data):
    return hashlib.sha256(data).hexdigest()


def select_native(data, model):
    bom = data.startswith(b'\xef\xbb\xbf')
    text = data.decode('utf-8-sig')
    newline = '\r\n' if '\r\n' in text else '\n'
    result, in_rr, found, pending = [], False, False, set()
    values = {'backend': 'native', 'modeldir': model}
    def flush():
        if pending and result and not result[-1].endswith(('\n', '\r')):
            result.append(newline)
        for key in ('backend', 'modeldir'):
            if key in pending:
                result.append(('Backend' if key == 'backend' else 'ModelDir') + ' = ' + values[key] + newline)
        pending.clear()
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('['):
            if in_rr:
                flush()
            in_rr = stripped.split(']', 1)[0][1:].strip().casefold() == 'rayreconstruction'
            if in_rr:
                found = True
                pending = set(values)
        if in_rr and '=' in line and not stripped.startswith((';', '#')):
            key = line.split('=', 1)[0].strip().casefold()
            if key in values:
                prefix, body = line.split('=', 1)
                ending = '\r\n' if body.endswith('\r\n') else '\n' if body.endswith('\n') else ''
                body = body.removesuffix(ending) if ending else body
                comment = re.search(r'\s[;#]', body)
                suffix = body[comment.start():] if comment else ''
                result.append(prefix + '= ' + values[key] + suffix + ending)
                pending.discard(key)
                continue
        result.append(line)
    if in_rr:
        flush()
    if not found:
        if result and not result[-1].endswith('\n'):
            result.append(newline)
        result.extend((newline, '[RayReconstruction]' + newline, 'Backend = native' + newline,
                       'ModelDir = ' + model + newline))
    return (b'\xef\xbb\xbf' if bom else b'') + ''.join(result).encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--runtime', required=True, type=lambda p: Path(p).resolve())
    parser.add_argument('--enable', action='store_true')
    parser.add_argument('--backup-root', type=lambda p: Path(p).resolve(),
                        default=Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'd4r/backups')
    args = parser.parse_args()
    config = args.runtime / 'd4r.ini'
    if not config.is_file():
        parser.error('--runtime must contain d4r.ini')
    if args.enable and args.backup_root.resolve().is_relative_to(args.runtime.parent):
        parser.error('configuration backups must stay outside the game/runtime parent directory')
    cache_data = (args.model / 'windows-native-python-cache.json').read_bytes()
    cache = json.loads(cache_data)
    identity = cache['identity']
    if identity.get('preset') not in ('D', 'E') or identity.get('architecture') != 'gfx1101' or identity.get('arithmetic') != 'exact-gfx11':
        raise RuntimeError('Unsupported local RR cache identity')
    products = cache['products']
    if not {'rr.txt', 'enc0.args', 'hkpn.args'}.issubset(products):
        raise RuntimeError('RR cache omits required manifest/templates')
    for name, expected in products.items():
        if Path(name).name != name or name in ('.', '..') or '\\' in name or ':' in name:
            raise RuntimeError('Invalid model product path: ' + name)
        if sha((args.model / name).read_bytes()) != expected:
            raise RuntimeError('Model product changed: ' + name)
    manifest = (args.model / 'rr.txt').read_text()
    stages = [line.split() for line in manifest.splitlines() if line.startswith('stage ')]
    expected_stages = {'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0', 'hkpn'}
    if (not manifest.startswith('D4RR0001\n') or
        f'preset {4 if identity["preset"] == "D" else 5}' not in manifest.splitlines() or
        len(stages) != 12 or {fields[1] for fields in stages} != expected_stages or 'arithmetic gfx11' not in manifest):
        raise RuntimeError('Incomplete RR model manifest')
    for fields in stages:
        if fields[2] not in products or fields[4] not in products:
            raise RuntimeError('Stage product not covered by cache')
    name = 'rr-' + identity['preset'].lower() + '-' + sha(cache_data)[:12]
    destination = args.runtime / 'engine' / name
    if not destination.resolve().is_relative_to(args.runtime):
        raise RuntimeError('RR destination must stay within the selected runtime')
    reused = destination.exists()
    if not reused:
        destination.mkdir(parents=True)
        for product in products:
            shutil.copy2(args.model / product, destination / product)
        (destination / 'd4r-private-rr.json').write_text(json.dumps({'privateLocalModel': True, 'cacheSha256': sha(cache_data), 'products': products}, indent=2) + '\n')
    for product, expected in products.items():
        if sha((destination / product).read_bytes()) != expected:
            raise RuntimeError('Existing/deployed RR model differs; preserving it: ' + str(destination))
    backup = None
    if args.enable:
        original = config.read_bytes()
        updated = select_native(original, destination.relative_to(args.runtime).as_posix())
        if updated != original:
            args.backup_root.mkdir(parents=True, exist_ok=True)
            backup = args.backup_root / ('rr-config-' + uuid.uuid4().hex + '.ini')
            with backup.open('xb') as stream:
                stream.write(original)
            temporary = config.with_name('d4r.ini.' + uuid.uuid4().hex + '.tmp')
            with temporary.open('xb') as stream:
                stream.write(updated)
            if config.read_bytes() != original:
                temporary.unlink()
                raise RuntimeError('d4r.ini changed during deployment; preserved current configuration')
            os.replace(temporary, config)
    print(json.dumps({'deployed': str(destination), 'preset': identity['preset'], 'productsVerified': len(products),
                      'reused': reused, 'enabled': args.enable, 'configBackup': str(backup) if backup else None,
                      'privateLocalModel': True, 'published': False}, indent=2))


if __name__ == '__main__':
    main()
