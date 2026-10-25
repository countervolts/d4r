#!/usr/bin/env python3
"""Fetch the checksum-pinned OptiScaler archive; print its local path."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch(cache=None):
    release = json.loads((ROOT / 'packaging/optiscaler-release.json').read_text())
    cache = Path(cache) if cache else Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'd4r-optiscaler'
    directory = cache / release['tag']
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / release['url'].rsplit('/', 1)[1]
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != release['sha256']:
            raise ValueError(f'OptiScaler checksum mismatch: {target}; remove it before retrying')
        return target
    descriptor, temporary = tempfile.mkstemp(prefix='.download-', dir=directory)
    try:
        with os.fdopen(descriptor, 'wb') as output, urllib.request.urlopen(release['url'], timeout=60) as response:
            digest = hashlib.sha256()
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
        if digest.hexdigest() != release['sha256']:
            raise ValueError('OptiScaler download checksum mismatch')
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


if __name__ == '__main__':
    try:
        print(fetch(sys.argv[1] if len(sys.argv) > 1 else None))
    except (OSError, ValueError) as error:
        sys.exit(str(error))
