#!/usr/bin/env python3
"""Offline MicroCUDA module pack: ZLUDA used as an offline compiler.

make_pack.py DLL ZLUDA_CACHE_DB ZLUDA_LIBCUDA OUT_DIR [--switches S] [--arch gfx1101]

ZLUDA keeps every module it compiles in its cache database (keyed by the blake3 of the PTX text, the
ZLUDA build and its code-generation switches) as a complete AMDGPU code object. This collects, for
every fatbin in the DLSS DLL, the object ZLUDA compiled for it with one exact ZLUDA build (identified
like ZLUDA does, by the size and mtime of its libcuda.so) and one switch set, so MicroCUDA can load
it without the PTX translator. Fatbins without such an object are listed and left out: MicroCUDA
rejects them instead of running anything else. Native kernel overrides are not part of the pack; the
bridge serves those as for ZLUDA.

Output: OUT_DIR/modules/<fatbin fnv>.co and OUT_DIR/microcuda-pack.txt, read by the runtime. Fatbins without
PTX become 'empty' entries: ZLUDA loads those as modules without functions.
Needs the blake3 Python package. No PTX or NVIDIA code is written; the objects are local build
products like ZLUDA's cache itself.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys

import blake3

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'kernels/tools'))
from extract_dlss_ptx import fatbins, ptx_entries
from kernel_manifest import fnv1a64

# EF_AMDGPU_MACH (low byte of the ELF e_flags) of each supported target
MACH = {'gfx1100': 0x41, 'gfx1101': 0x46, 'gfx1102': 0x47, 'gfx1103': 0x44, 'gfx1200': 0x48, 'gfx1201': 0x4e}

# switches that do not change generated code (see zluda_build_version in ZLUDA's module.rs)
IGNORED_SWITCHES = ('D4R_ZLUDA_CACHE_HOME=',)


def fingerprint(libcuda):
    s = os.stat(libcuda)
    return f'{s.st_size:x}.{int(s.st_mtime):x}.{s.st_mtime_ns % 1_000_000_000:x}'


def switches_of(version):
    # "<git sha>+<library fingerprint>[+<sorted D4R_ZLUDA_ switches>]"
    parts = version.split('+', 2)
    raw = parts[2] if len(parts) == 3 else ''
    return ','.join(s for s in raw.split(',') if s and not s.startswith(IGNORED_SWITCHES))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dll', type=Path)
    p.add_argument('db', type=Path)
    p.add_argument('libcuda', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--switches', help='code-generation switch set to take when the cache holds several')
    p.add_argument('--arch', default='gfx1101')
    a = p.parse_args()

    fp = fingerprint(a.libcuda)
    db = sqlite3.connect(f'file:{a.db}?mode=ro', uri=True)
    rows = collections.defaultdict(list)
    for h, version, attrs, rowid in db.execute(
            'select hash, zluda_version, backend_key, id from modules where device = ? and zluda_version like ?',
            (a.arch, f'%+{fp}%')):
        rows[h].append((switches_of(version), version, attrs, rowid))
    sets = collections.Counter(s for v in rows.values() for s, *_ in v)
    if not sets:
        sys.exit(f'no {a.arch} modules for ZLUDA build {fp} in {a.db}')
    if a.switches is None:
        if len(sets) > 1:
            sys.exit('several switch sets in the cache; pick one with --switches:\n  ' +
                     '\n  '.join(f'{n:4} {s!r}' for s, n in sets.most_common()))
        a.switches = next(iter(sets))

    dll = a.dll.read_bytes()
    (a.out / 'modules').mkdir(parents=True, exist_ok=True)
    lines, missing, attrs_seen, version_seen = [], [], set(), set()
    for index, (_, fatbin) in enumerate(fatbins(dll)):
        ffnv = fnv1a64(fatbin)
        # ZLUDA compiles the last PTX of a fatbin that it can use; take the last one it compiled
        ptxs = list(ptx_entries(fatbin))
        if not ptxs:
            # no PTX (only cubins for NVIDIA GPUs): ZLUDA's release build loads an empty module, so must we
            lines.append(f'empty {ffnv:016x} {len(fatbin)}')
            continue
        chosen = None
        for ptx in reversed(ptxs):
            text = ptx.rstrip(b'\0')
            for s, version, attrs, rowid in rows.get(blake3.blake3(text).hexdigest(), []):
                if s == a.switches:
                    chosen = (text, version, attrs, rowid)
                    break
            if chosen:
                break
        if chosen is None:
            missing.append(f'{index}:{ffnv:016x}')
            continue
        text, version, attrs, rowid = chosen
        attrs_seen.add(attrs)
        version_seen.add(version)
        binary = db.execute('select binary from modules where id = ?', (rowid,)).fetchone()[0]
        if binary[:4] != b'\x7fELF' or struct.unpack_from('<I', binary, 0x30)[0] & 0xff != MACH[a.arch]:
            sys.exit(f'fatbin {index}: cached object is not a {a.arch} code object')
        name = f'{ffnv:016x}.co'
        (a.out / 'modules' / name).write_bytes(binary)
        lines.append(f'module {ffnv:016x} {len(fatbin)} {fnv1a64(text):016x} {len(text)} '
                     f'{hashlib.sha256(binary).hexdigest()} {name}')
    if len(attrs_seen) > 1 or len(version_seen) > 1:
        sys.exit(f'inconsistent compile attributes in the chosen set: {attrs_seen} {version_seen}')
    fp8 = json.loads(next(iter(attrs_seen))).get('wmma_fp8_native', False) if attrs_seen else False
    header = [
        '# d4r MicroCUDA module pack (generated by tools/microcuda/make_pack.py; do not edit)',
        f'# dll sha256 {hashlib.sha256(dll).hexdigest()}',
        f'# zluda {next(iter(version_seen))}',
        f'# attributes {next(iter(attrs_seen))}',
        f'arch {a.arch}',
        f'switches {a.switches}',
        # RDNA4: compiled with native FP8 WMMA (d4r.ini NativeFp8); the runtime refuses a mismatched process
        f'fp8native {int(fp8)}',
    ]
    (a.out / 'microcuda-pack.txt').write_text('\n'.join(header + sorted(lines)) + '\n')
    print(f'{sum(l.startswith("module") for l in lines)} modules and {sum(l.startswith("empty") for l in lines)} empty modules packed for {a.arch}, switches {a.switches!r}; '
          f'{len(missing)} fatbins without a compiled object')
    if missing:
        (a.out / 'missing.txt').write_text('\n'.join(missing) + '\n')


if __name__ == '__main__':
    main()
