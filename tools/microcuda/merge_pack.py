#!/usr/bin/env python3
"""Per-module compile tuning: build a pack from several variant packs of the same DLL, ZLUDA build and target.

merge_pack.py BASE_PACK OUT_PACK [--all VARIANT_PACK] [--choose CHOICES.json]

BASE_PACK fixes the header (switch set the process must run with) and supplies every module not chosen
otherwise. --all takes every module from one variant pack (to measure that variant); --choose takes a JSON
map {fatbin_fnv_hex: variant_pack_dir}. Variants must only differ in code-generation choices that keep
results bit-exact (scheduler strategy, wave size); the caller validates the merged pack's output against the
reference before using it. Each module line keeps its own SHA256, and a comment records its variant.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys


def read(pack):
    header, modules = [], {}
    for line in (pack / 'microcuda-pack.txt').read_text().splitlines():
        if line.startswith(('module ', 'empty ')):
            modules[line.split()[1]] = line
        else:
            header.append(line)
    return header, modules


def field(header, key):
    return next((l.split(' ', 1)[1] if ' ' in l else '' for l in header if l.split(' ', 1)[0] == key), None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('base', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--all', type=Path)
    p.add_argument('--choose', type=Path)
    a = p.parse_args()
    header, modules = read(a.base)
    choice = {}
    if a.all:
        choice = {f: a.all for f in modules}
    if a.choose:
        choice.update({f: Path(v) for f, v in json.loads(a.choose.read_text()).items()})
    variants = {}
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / 'modules').mkdir(parents=True)
    lines, notes = [], []
    for f, line in sorted(modules.items()):
        source = a.base
        if f in choice and line.startswith('module '):
            v = choice[f]
            if v not in variants:
                vh, vm = read(v)
                if field(vh, 'arch') != field(header, 'arch') or \
                        next(l for l in vh if l.startswith('# dll')) != next(l for l in header if l.startswith('# dll')):
                    sys.exit(f'{v}: different DLL or target')
                variants[v] = (vh, vm)
            vh, vm = variants[v]
            if f in vm:
                line, source = vm[f], v
                notes.append(f'# variant {f} {field(vh, "switches")}')
        if line.startswith('module '):
            name = line.split()[6]
            shutil.copyfile(source / 'modules' / name, a.out / 'modules' / name)
        lines.append(line)
    (a.out / 'microcuda-pack.txt').write_text('\n'.join(header + notes + lines) + '\n')
    print(f'{len(lines)} entries, {len(notes)} from variant packs -> {a.out}')


if __name__ == '__main__':
    main()
